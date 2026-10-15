# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Latent Compaction with Test-Time Training](https://arxiv.org/abs/2610.20582) | 提出“潜在压缩”方法，通过测试时训练在模型权重冻结且不接触下游问题的情况下，学习紧凑的文档级 KV 缓存，显著优于同等预算的缓存压缩基线。 |
| [^2] | [One Task, Many Interfaces: Simple Cross-Interface Consistency Training Improves LLM Agent Robustness](https://arxiv.org/abs/2610.20576) | 提出跨界面策略正则化（CIPR）这一自监督方法，通过在两个界面下惩罚动作分布间的KL散度，使LLM智能体在面对界面扰动（如动作重命名、记号变化）时保持一致行为，显著提升其跨界面的鲁棒性。 |
| [^3] | [Manac\'a-1B-Instruct-2: Reproducible Safety Alignment for a Native Brazilian-Portuguese LLM. Negative DPO Results, a Controlled Composition Scan, and a Silent Failure Mode of the LLM Judge](https://arxiv.org/abs/2610.20570) | 该论文为原生巴西葡萄牙语模型Manacá-1B-Instruct-2提供了可复现的安全对齐方案，记录了三次失败的DPO尝试（包括LLM评审奖励饱和而生成结果毫无变化的隐性失效模式），证明安全性来自专门的安全SFT，并通过受控扫描发现安全数据混合中通用数据比例的过度拒绝曲线呈非单调性，拐点在40-50%。 |
| [^4] | [Scaling Discovery Requires Forgetting](https://arxiv.org/abs/2610.20563) | 论文发现 LLM 智能体自动化研究的三种扩展策略都会因智能体锚定于自身历史尝试而停滞，并提出通过周期性丢弃搜索状态、从最佳候选重新启动的“遗忘”机制来突破停滞，在相同预算下实现更大的发现成果。 |
| [^5] | [PerSo: Investigating Facets of LLMs' Social Intelligence in Real-World Persuasion Scenarios](https://arxiv.org/abs/2610.20548) | 该论文提出了PerSo基准，利用Reddit ChangeMyView社区的真实说服对话，同时评估大语言模型社会智能中的认知智能（推断他人心理状态）与行为智能（选择合适社会行动）两个方面。 |
| [^6] | [PrismaCoder: Pushing the Boundaries of Open Multimodal Code Intelligence through Decomposed Rubric Synthesis](https://arxiv.org/abs/2610.20538) | 该论文提出PrismaCoder，从数据端出发，通过分解式评分标准合成方法，解决多模态代码生成中样本的程序正确性、代码质量与视觉正确性难以同时评估的核心难题，从而推动开放多模态代码智能的发展。 |
| [^7] | [VASPO: Self-Distillation with Privileged Value Estimation](https://arxiv.org/abs/2610.20516) | 该论文提出VASPO方法，将特权价值估计与自蒸馏在PPO框架内结合，利用评论家与自教师在学生生成前缀上评估特权信息时呈现的互补位置动态特性，为推理模型提供更有效的词元级训练信号。 |
| [^8] | [ReaDy: Recurrent States as Dynamic Context for Speculative Drafting in Hybrid Linear Attention Large Language Models](https://arxiv.org/abs/2610.20503) | ReaDy是首个利用混合线性注意力模型循环状态作为动态上下文的推测解码方法，通过为每个草拟位置设置可学习查询直接从目标模型循环状态中检索补充上下文，以极低的计算开销提升草拟质量。 |
| [^9] | [Lost in Translation: Why Lower Bias Scores Don't Mean Fairer Models](https://arxiv.org/abs/2610.20479) | 本研究审计了10个开源模型在8种语言中的偏见表现，发现非英语语言下刻板印象分数的下降主要源于翻译缺陷与测量噪声，而非模型本身更公平，因为测试项的英语分数对其翻译后分数的预测能力远低于对改写后分数的预测能力。 |
| [^10] | [AdaRSI: On-Demand Recursive Self-Improvement for Embodied Agents](https://arxiv.org/abs/2610.20474) | AdaRSI提出了一种按需递归自我改进框架，先改进技能与外围框架、再通过重试归因持续失败、最后仅微调不可靠的子步骤，从而在无需人类示范的情况下显著提升具身智能体完成长时程任务的能力。 |
| [^11] | [ChronoCulture: Evaluating Cultural Awareness Across Time in Large Language Models](https://arxiv.org/abs/2610.20472) | 提出ChronoCulture基准，利用72个国家纵向调查数据构建的10,360个时间对，揭示大语言模型虽能预测某一时间点的文化态度，却无法可靠捕捉文化态度随时间的动态演变，呈现出静态化的文化图景。 |
| [^12] | [Equal CER, Unequal Damage: Severity-Weighted Character Error Rate for OCR](https://arxiv.org/abs/2610.20461) | 提出semCER指标，利用大语言模型为OCR中每个字符编辑赋予严重度权重，使错误率评估更符合人类读者对错误实际损害的感知。 |
| [^13] | [MolLean: A Process-Verifiable Language for Chemical Reasoning](https://arxiv.org/abs/2610.20459) | MolLean 提出了一种基于 Lean 的过程可验证化学推理语言，通过分层内核检查显式分子状态上的转换与断言、带类型的见证协议区分结构检查与外部测量，并借助有状态工具接口 MolLean-MCP 提供交互反馈，从而有效抑制化学思维链中的幻觉论断。 |
| [^14] | [Same Claim, Different Interpretations: How LLM Agents Resolve Ambiguity in Scientific Claims](https://arxiv.org/abs/2610.20457) | 该论文提出Claim Cards方法，在证据收集前以九个显式字段记录LLM智能体对歧义科学主张的解读，并验证了解读方式的不同确实会改变智能体的验证判定结果。 |
| [^15] | [The COUNT: Counting N-grams Unlocks Scaling on Host Memory and Training-Free Adaptation](https://arxiv.org/abs/2610.20456) | 提出COUNT方法，通过在廉价的主机内存上存储预计算的n-gram计数并结合语言模型自身的unembedding来构建n-gram嵌入，无需训练新参数，从而实现主机内存上的扩展与免训练适配。 |
| [^16] | [SwarmRSI: Recursive Self-Improvement through Evolving Scientific Agent Swarm](https://arxiv.org/abs/2610.20431) | 提出了 SwarmRSI 框架，通过反馈引导的群体变异和互补群体重组两种机制演化科学智能体群体，解决假设与实现修改不明确以及有用组件被隐藏的问题，实现科学研究中的递归自我改进。 |
| [^17] | [Analyzing performance of Retrieval Augmented Generation models for Question Answering Tasks](https://arxiv.org/abs/2610.20419) | 本文分析了RAG模型在问答任务中的性能，指出尽管RAG架构近年取得巨大进展，但在以南亚当地语言回答该地区特有文化问题时仍面临挑战。 |
| [^18] | [From Unreliable to Usable: Decomposing Failure Modes and Confidence Signals in LLM Entity Extraction - Evidence from Cyber Threat Intelligence](https://arxiv.org/abs/2610.20417) | 该论文通过对新一代低成本与旗舰级大语言模型在354个网络威胁情报事件上的预注册冻结评估，发现实体抽取的不可靠性在所有模型层级均持续存在（精确率仅0.03-0.34且校准误差高达0.53-0.94），并揭示主导性失败并非幻觉，同时探索了置信度信号能否将不可靠输出转化为可用输出。 |
| [^19] | [Cortex: Consolidating Context into Parametric Memory](https://arxiv.org/abs/2610.20399) | 本文提出 Cortex，通过生成前瞻性查询将上下文增益蒸馏为可合并的低秩或结构化前馈网络参数更新，在 4B 至 120B 的多种稠密与混合专家模型上实现了“巩固—保持”帕累托最优的记忆巩固。 |
| [^20] | [FOLD: Position-Free Far Memory with Local Order for Long-Context Speculative Decoding](https://arxiv.org/abs/2610.20392) | FOLD是一种免训练的长上下文投机解码方法，它保留精确的局部位置，将所有远端条目作为共享位置上的远端记忆处理，并通过锐化记忆内部的注意力来缓解注意力稀释问题。 |
| [^21] | [From Triage to Impact: A Multi-Hop Reasoning Benchmark for Explainable Persian Hate Speech](https://arxiv.org/abs/2610.20370) | 本研究提出了首个面向波斯语可解释仇恨言论分析的细粒度多跳推理基准，涵盖从内容分诊到严重程度评估的多项任务，发现大语言模型在需要更细粒度语义与推断性推理的任务上性能下降，且引入中间推理步骤可显著提升严重程度预测效果。 |
| [^22] | [Cascaded Retrieval with Cross-Modal Query Refinement for Efficient Visual Document Search](https://arxiv.org/abs/2610.20367) | 提出SCOUT级联双轨检索框架，通过轻量级视觉检索器先筛选候选、再由多向量重排序器精排，并结合语义轨道检索离线生成的分层摘要，最后用跨模态查询精炼融合两路信号，实现高效且可扩展的视觉文档搜索。 |
| [^23] | [Lexical Persistence Without Semantic Entrenchment in AI Agent Communities](https://arxiv.org/abs/2610.20365) | 该研究通过跨平台对比发现，AI智能体社区虽然能够维持词汇的流行与传播，但缺乏人类语言社区所具有的长程时间语义依赖结构，其语义漂移速度更快、语义稳定性更低。 |
| [^24] | [Do AI Agents Gossip? Evaluative Talk in Human and AI-Agent Online Communities](https://arxiv.org/abs/2610.20360) | 本研究首次通过对比AI智能体社区（Moltbook）与人类在线论坛（PTT），发现AI智能体社区中会自发涌现类似人类八卦的评价性言论，尽管其流行度显著低于人类社区，且两者在后续回应的发展模式上存在明显差异。 |
| [^25] | [TreeFlex: Agents Navigating Their Own Search Trees](https://arxiv.org/abs/2610.20334) | TreeFlex让单一语言模型在反思与执行两种角色间切换，在框架维护的搜索树上自主选择从哪个节点恢复搜索，从而解决ReAct式智能体因上下文膨胀导致的推理失效问题。 |
| [^26] | [Anuvani-Diarization: Zero Shot Speaker Diarization and Annotation Conventions Across 22 Languages](https://arxiv.org/abs/2610.20327) | 该论文提出Anuvani-Diarization，通过不确定性感知的银标准监督将自监督端到端分割模型适配到印度22种表列语言的多语言对话场景，实现零样本说话人日志，在IndicDiarBench上取得16.78%的最低时长加权错误率。 |
| [^27] | [MMLA-RSI: Recursive Self-Improvement through RTT and RTMA](https://arxiv.org/abs/2610.20294) | 提出了一种递归自我改进方案MMLA-RSI，其中RTT改变策略状态、RTMA改变权威记忆，且这两种改进方法本身也可作为改进对象，由独立接纳的后继者在下一代中接管相同角色，从而实现智能体行为的持续递归改进。 |
| [^28] | [Taking the PULSE of Medical Reasoning: Step-Level Retrieval Timing for Knowledge-Grounded Inference](https://arxiv.org/abs/2610.20292) | 提出PUlSE方法，利用大模型内部表征的层间波动性作为内在信号，在医学推理的每个步骤自适应判断是否需要外部检索，从而在提升事实准确性的同时避免不必要的检索噪声与延迟。 |
| [^29] | [Neural POS Tagging for the Low-Resource Saraiki](https://arxiv.org/abs/2610.20279) | 本文构建了首个经人工标注的萨拉基语通用词性标注数据集，并证明XLM-RoBERTa base在该低资源语言上表现最佳，测试准确率达88.60%。 |
| [^30] | [CSWBench: Evaluating Context-Sensitive Code-Switching in Creative Writing](https://arxiv.org/abs/2610.20272) | 该论文提出了CSWBench基准与基于理论的CSWEval评估框架，从事件层面（许可性、语境适当性、功能增益）和全文层面两个维度，对大语言模型在创意写作中语境敏感的语码转换能力进行系统评估，并对十二个模型进行了实验分析。 |
| [^31] | [Anuvani-LID: Robust and Calibrated Spoken Language Identification for Indian Languages](https://arxiv.org/abs/2610.20263) | 本文提出Anuvani-LID，一个在54,643小时印度语语音上训练的1.15亿参数专用Conformer语言识别模型，覆盖22种印度语言及印度口音英语，在短促、嘈杂、窄带等困难条件下显著优于Whisper等基础语音模型及Gemini、GPT等音频大语言模型。 |
| [^32] | [Repairing Speech Access After Text Updates with Frozen Affine Fields](https://arxiv.org/abs/2610.20253) | 提出AFFIRM方法，在保持模型检查点完全冻结的前提下，通过可拆卸的仿射场算子（冻结的基础校正加上零均值的查询特定校正）修复文本更新后新知识在语音查询时难以访问的问题。 |
| [^33] | [Scaling Multilingual Mixture-of-Experts: How Language Coverage Changes Capacity and Compute Allocation](https://arxiv.org/abs/2610.20247) | 本文提出了一种感知重复的联合缩放定律，揭示多语言MoE模型中活跃参数、训练词元、专家数量与语言覆盖范围之间的关系，并发现32个专家相比稠密模型可将语言从12种扩展到200种时的损失惩罚降低83%。 |
| [^34] | [Scaling Multilingual Safety and Reasoning Evaluation through Reasoning-Guided Translation](https://arxiv.org/abs/2610.20221) | 提出了一种推理引导的翻译框架，通过分离翻译策略选择与大规模生成，并结合任务感知验证、针对性修复和独立质量筛选，构建了覆盖安全性、对话和数学推理任务的多语言评估基准。 |
| [^35] | [What Makes a Social Agent's Interactions Effective? Learning from Verifiable Outcomes in Paired Worlds](https://arxiv.org/abs/2610.20195) | 该论文提出HICO-Bench基准，通过对达成计划的可行性进行程序验证、对对话完整性进行LLM裁判检验的双重机制，利用成对场景（公开上下文相同但私有条件不同）评估社交智能体，揭示仅有高评分对话并不等于达成了可验证的成功协调。 |
| [^36] | [When Occupation-derived Skills Don't Help: On the Limits of Skill-Augmented Career Path Prediction](https://arxiv.org/abs/2610.20186) | 该研究批判性考察了技能表示在职业路径预测中的实际价值，发现无论是预测技能还是真实ESCO技能标注均无法显著超越纯文本基线，其根本原因在于文本与技能表示的高度冗余、ESCO分类判别力有限以及技能嵌入的表示坍缩。 |
| [^37] | [Cross-Model In-Context Neurofeedback Reveals (Limited) Activation Control in LLMs](https://arxiv.org/abs/2610.20179) | 本文提出跨模型上下文内神经反馈新范式，首次将模型在任务上的胜任能力与对自身激活的真实访问区分开来，测试十个开源大模型后发现大语言模型仅具备有限的激活控制能力。 |
| [^38] | [TreeSpark: Learning to Draft the Trees You Verify for Speculative Decoding](https://arxiv.org/abs/2610.20175) | TreeSpark 提出一个统一的训练与推理框架，通过一次骨干前向传播构建父节点条件的候选树，并利用生存负对数似然和有上限覆盖铰链两种目标函数（复用参考链目标分布、无需额外目标模型前向传播）来训练推测解码的草稿树，从而提升并行草稿的接受率。 |
| [^39] | [RM-OPD: On-Policy Distillation for Generative Reward Models](https://arxiv.org/abs/2610.20174) | 该论文提出RM-OPD两阶段框架，先通过单领域GRPO微调训练各领域专家教师模型，再通过路由同策略蒸馏将其整合为单一学生模型，从而解决Mix-RL训练生成式奖励模型时的奖励稀疏与跨领域干扰问题。 |
| [^40] | [MultiPartyTurnBench: Benchmarking Floor Control in Multi-Party Spoken Dialogue](https://arxiv.org/abs/2610.20172) | MPTB是首个针对多方口语对话中话轮控制的统一受控基准，通过240段信道分离、涵盖2至5个说话人的合成对话及精确事件标注，系统评估了话轮结束检测与打断检测任务，实现了对真实话轮转移的因果评估。 |
| [^41] | [Gap Detection: Mitigating Homogenization through Demand and Defensibility](https://arxiv.org/abs/2610.20162) | 提出Gap Detection框架，通过结合查询相关性（Demand）与现有内容相异度（Defensibility）的GapScore目标函数，在文本嵌入空间中检测相关但未被占据的方向，以缓解语言模型开放式回答的同质化问题。 |
| [^42] | [LLM-Based Policy Compliance Verification of Loan Applications](https://arxiv.org/abs/2610.20155) | 该论文提出了一种基于大语言模型的两步式政策检查器原型，通过自动从政策文件中提取结构化合规规则，并据此对贷款申请进行合规性验证，实现了信贷审批合规审查流程的自动化。 |
| [^43] | [Sparse Autoencoders for Mechanistic Interpretability in NLP: A Critical Survey of Recoverability, Interpretability, and Utility](https://arxiv.org/abs/2610.20139) | 本文对138项NLP领域的稀疏自编码器研究进行了批判性综述，提出了连接字典设计与NLP功能的任务导向分类体系，并系统评估了特征可恢复性、语义可解释性与行为效用方面的证据。 |
| [^44] | [Why Generate a Click? Plug-in Decision Heads for Web Agents](https://arxiv.org/abs/2610.20107) | 该论文提出在冻结的视觉语言模型上外挂即插即用决策头，直接从隐藏状态中读取动作类型、目标元素和文本选项，并通过置信度级联在高置信时跳过文本生成，从而在不损失（甚至略提升）性能的前提下大幅加速网页智能体的决策。 |
| [^45] | [Language Exposure Matters: Distribution and Order in Multilingual BabyLM Pre-training](https://arxiv.org/abs/2610.20102) | 在数据有限的多语言BabyLM预训练中，语言暴露的数量与顺序均会影响性能，且全程混合的同步多语言训练在所有15个配置中都优于先单语后多语的课程式训练。 |
| [^46] | [AFCS: Atomic-Fact-Conditioned Synthesis for Factual Knowledge Pretraining](https://arxiv.org/abs/2610.20086) | AFCS 提出先从源文档抽取有据可依的原子事实、再以每条事实驱动多格式语料生成的方法，在 OLMo-3-7B 与 32B 模型的继续预训练实验中事实性优于 WRAP 和 Nemotron-CC。 |
| [^47] | [Designing Textual Interfaces To Enhance Llm Reasoning On Audio-Visual Question Answering](https://arxiv.org/abs/2610.20085) | 本研究探讨了文本接口设计对多模态推理系统中自动提示优化的影响，发现全模态LLM直接优化收益依赖具体模型，而描述器-推理器和智能体架构虽提供更可解释的优化接口却更难优化。 |
| [^48] | [Which Knowledge Should LLMs Trust? Uncovering a Generalizable Steering Direction Across Knowledge Conflicts](https://arxiv.org/abs/2610.20067) | 研究发现大语言模型在参数化知识与上下文知识之间的选择沿残差流中单一方向线性表示，且该方向可跨任务泛化，通过操控该方向能有效改善模型在RAG、记忆和规则遵循任务中对提示信息的利用。 |
| [^49] | [What Is a Clarification? Estimating Model-Dependent Input Ambiguity](https://arxiv.org/abs/2610.20052) | 该论文区分了输入固有的“世界歧义”与语言模型实际可感知的“模型歧义”，通过唯一性、忠实性和答案等价性三个标准形式化自然语言澄清，证明了有效答案与澄清等价类之间的一一对应关系，并据此提出一种黑盒方法来估计模型相关的输入歧义。 |
| [^50] | [FACES: Fine-grained Gated Semantic Alignment for Multimodal Emotional Support Conversation](https://arxiv.org/abs/2610.20051) | FACES提出了一种细粒度门控语义对齐框架，通过多源奖励整合、非补偿性门控融合机制以及基于结构化思维链锚点的token级信用分配，有效解决了多模态情感支持对话中回复浅显和支持策略不当的问题。 |
| [^51] | [LintEval: Benchmarking LLMs on Linter-Guided File Refactoring](https://arxiv.org/abs/2610.20027) | 该论文提出LintEval，一个覆盖C++、Python和Java的多语言基准，用于评测大语言模型在linter引导的文件重构中减少lint告警的能力，实验揭示了告警减少与功能保持之间的持续权衡。 |
| [^52] | [A Strong Harness Narrows the Model Gap: A Comparative Evaluation for Legislative Document-to-XML Conversion](https://arxiv.org/abs/2610.19999) | 在立法文档转XML的任务中，精心设计的确定性执行框架（尤其是多代理分解式编排）比模型本身更重要——流水线中最弱的模型表现超过了单独使用的最强模型。 |
| [^53] | [Gray-Box Hallucination Detection in LLMs with Token Identities](https://arxiv.org/abs/2610.19996) | 本文提出GURI，一种轻量级灰盒幻觉检测器，通过将候选词元概率与词元身份向量标识符相结合，不仅能衡量模型的不确定程度，还能识别模型不确定的具体对象，从而实现更精细的大语言模型幻觉检测。 |
| [^54] | [SoK: Grading Open-Text Legal Answers, from Law School Exams to Trustworthy Rubric Judges](https://arxiv.org/abs/2610.19992) | 本文系统化研究了开放式法律文本答案（如法学院考试）的评估方法，从教育测量与法律规范中提炼出八项评估标准并对比八种评估范式，发现针对任务定制评分规则是让LLM裁判与人类评分者达成一致的最关键手段。 |
| [^55] | [Choosing the Number of Attention Heads from Intrinsic Dimension: A Small-Scale Study](https://arxiv.org/abs/2610.19966) | 本文提出一种无需标签的启发式方法，通过嵌套Dropout VAE估计词元表示的内在维度来预先确定transformer注意力头的数量，实验表明按内在维度设置约10个注意力头的模型与32个头性能统计等效，且注意力参数减少69%。 |
| [^56] | [AtomSearch: Evidence-Grounded Tree Search for Agent Skill Optimization](https://arxiv.org/abs/2610.19951) | AtomSearch 通过对比式原子编辑和技能状态树搜索，以证据为依据实现局部化、可回溯的智能体技能优化，在多个任务和模型上超越了 SkillOpt。 |
| [^57] | [Does Machine Translation Retain Media Bias?](https://arxiv.org/abs/2610.19945) | 本研究考察了机器翻译（英/德语译入保加利亚语）对27类媒体偏见的保留情况，发现翻译过程无法完全保留源文本中的新闻偏见。 |
| [^58] | [To Reason but Not to Stop: Understanding and Recovering Pruned Large Reasoning Models](https://arxiv.org/abs/2610.19934) | 该研究揭示剪枝后大型推理模型的性能下降主要源于得到正确答案后无法停止生成而非推理能力受损，并提出免训练的终止引导激活导向方法（TGAS），利用与确定答案相关的隐状态共享方向在推理时引导模型及时终止，从而恢复剪枝模型的准确性。 |
| [^59] | [MemDuet: Learning Dual-Process Memory for Long-Context Reasoning](https://arxiv.org/abs/2610.19928) | 提出受卡尼曼快慢思考启发的双过程记忆框架MemDuet，通过小语言模型快速并行选择证据、大语言模型验证并按文档顺序更新记忆的三阶段联合训练，实现长上下文推理中性能与效率的平衡。 |
| [^60] | [MolHop: Scaffold-Hopping Molecular Design with Agentic Reinforcement Learning](https://arxiv.org/abs/2610.19920) | MolHop 提出了一个基于智能体强化学习的跨骨架分子发现框架，将约 160 万个商业分子骨架组织成从粗到细的分层动作空间，使模型能够发现骨架完全不同但保留参考分子功能特性的新分子。 |
| [^61] | [Not All Criteria Matter: Importance-Aware Criteria Weighting for Rubric-Based LLM Evaluation](https://arxiv.org/abs/2610.19915) | 提出了重要性感知标准加权方法ICW，将LLM对所有标准的整体权重判断转化为易于完成的两两重要性比较，并通过检查修正不一致的比较结果、从主特征向量推导权重，从而提升基于评分标准的LLM评估的准确性与可靠性。 |
| [^62] | [Beyond Benchmark Accuracy: A Diagnostic Test for LLM Chemical Prediction](https://arxiv.org/abs/2610.19906) | 该论文提出身份遮蔽推理诊断框架，通过隐藏目标实体身份、以已知源实体加显式结构编辑操作来定义目标，从而区分大语言模型的化学预测能力究竟来自真正的结构-性质推理，还是来自上下文信息或记忆检索。 |
| [^63] | [Frustratingly Easy Majority Voting Using Length Conditioning](https://arxiv.org/abs/2610.19890) | 提出长度条件化多数投票（LCMV），利用“较长的思维链正确概率更低”这一经验观察，通过逻辑函数建模长度条件化先验，并仅在测试时使用无标签数据自适应调整该先验来加权各答案的投票，无需微调LLM即可显著提升推理准确性。 |
| [^64] | [RCCM: Diagnosing Gains and Harms in Reasoning-Prompt Transfer](https://arxiv.org/abs/2610.19872) | 论文提出RCCM框架，通过将答案与生成结束方式配对来诊断推理提示在不同模型部署间的迁移效果反转，发现简洁推理提示在统一token预算下能提升Phi的答案匹配率却大幅降低Qwen的匹配率，说明提示收益必须结合其执行与评估方式来解读。 |
| [^65] | [Rationale Distribution Distillation via Flow Matching for Distributional Continuous Latent Reasoning](https://arxiv.org/abs/2610.19871) | 该论文提出推理链分布蒸馏（RDD），利用流匹配将多条文本推理路径蒸馏为固定长度潜在序列上的条件流，以非自回归方式生成潜在推理的完整分布，在大幅降低重复生成成本的同时提升答案覆盖率与推理路径的多样性。 |
| [^66] | [SEA-HELM Audio: Evaluating Multilingual ASR-LLM Pipeline Performance Across SEA-HELM Subsets](https://arxiv.org/abs/2610.19869) | 该论文提出 SEA-HELM Audio——一个由文本基准 SEA-HELM 扩展而来的东南亚多语音频评估框架，利用合成语音在五种语言上量化了级联 ASR-LLM 流水线相比文本输入的显著性能退化（多项选择推理下降 9.00–18.50 个百分点，指令遵循下降 20.00–42.86 个百分点，其中泰米尔语退化最严重）。 |
| [^67] | [OpenTWBench (Open Taiwan Benchmark): Taiwan-Domain Knowledge Evaluation for Large Language Models, and How the Measurement Apparatus Shapes the Score](https://arxiv.org/abs/2610.19848) | 该论文提出开放式台湾领域知识评测套件 OpenTWBench，包含基于台湾十五年国家考试、共 25 万余道题的 tw-exam-bench，以及关于当代台湾生活的 1,696 道难度校准题的 tw-formosa-bench，并强调数据清洗与难度校准等测量装置设计对模型评测分数的塑造作用。 |
| [^68] | [EGCForest: Structured Agent Memory Organization for Enterprise Group Chats](https://arxiv.org/abs/2610.19836) | 提出EGCForest多视图分层记忆框架，将企业群聊消息组织为频道、参与者、制品三个互补的森林结构，并通过放置图和稀疏结构桥接实现跨视图连接，从而可靠地检索和组装分布式的证据。 |
| [^69] | [MAPLE: A Multi-Axis Parametric Knowledge Extraction Lens for LLM Evaluation](https://arxiv.org/abs/2610.19827) | 该论文提出MAPLE统一评估框架，通过在表层、概率和超越采样三个层面测量答案恢复与干扰、正确词元排名变化及采样遗漏答案的恢复情况，全面解耦知识提取增强（KEE）技术对大语言模型参数化知识提取的多样化影响。 |
| [^70] | [Model vs. User: Whose Values Prevail? Inference-Time Value Alignment in Multi-Turn Conversations](https://arxiv.org/abs/2610.19817) | 该研究首次在多轮对话中系统评估四种推理时价值引导方法，发现当用户与模型价值观冲突时，与模型基线偏好一致的一方往往胜出，且早期成功的引导效果会在数轮对话后减弱或反转，因此价值对齐应在完整双向对话中评估。 |
| [^71] | [Delta Tokenizer: Reference-Identical Tokenization of Session Deltas and Segment-Parallelism for Long Prompts on Multicore CPUs](https://arxiv.org/abs/2610.19801) | 提出Delta Tokenizer，一种仅对会话增量进行分词且保证与完整分词结果完全一致的CPU分词前端，并利用分段并行加速多核CPU上长提示词的分词，显著降低TTFT和CPU开销。 |
| [^72] | [Evolve Together, Fall into Place: How Social Scaffolding Shapes Self-Evolving Multi-Agent Systems](https://arxiv.org/abs/2610.19799) | 该论文提出AGORA框架，首次揭示了自我进化多智能体系统的社会脚手架（团队构成与相互依赖结构）决定进化成效——单纯的进化作用有限，而合适的脚手架可通过让最有能力的成员主导决策、其余成员退居辅助，使团队性能超越单独进化的强智能体高达25分。 |
| [^73] | [Learning Perception to Action Rules through Reinforcement Learning for Theory-of-Mind Reasoning](https://arxiv.org/abs/2610.19796) | 本文提出利用GRPO强化学习让模型学习显式的“感知到行动”规则来改进大语言模型的心智理论推理，其效果超越提示、监督微调和思维推理方法，且所生成的规则可作为可迁移知识提升其他模型（如GPT-4o-mini）的性能。 |
| [^74] | [JetSpec++: Foresight Parallel Drafting with Self Refinements](https://arxiv.org/abs/2610.19792) | JetSpec++通过块内自精炼、块间前瞻草稿生成和流水线化草稿生成与验证的算法-系统协同设计，几乎完全隐藏草稿模型的开销，实现最高7.45倍于自回归解码的加速。 |
| [^75] | [More Methods, Better Agents? A Reality Check on Self-Distillation in Agentic Reinforcement Learning](https://arxiv.org/abs/2610.19789) | 本文通过 132 次训练运行（累计超过 1.2 万 H800 GPU 小时），在四个智能体基准和三种模型规模上系统性地将十种自蒸馏方法与 GRPO 直接对比，以厘清性能提升的真实来源，并揭示各方法在性能、训练稳定性与计算成本上的实际优劣。 |
| [^76] | [Likelihood Is Not Authorship: Can We Reliably Measure Human Contribution in AI-Assisted Writing?](https://arxiv.org/abs/2610.19767) | 该论文揭示了基于提示词预测能力来衡量AI辅助写作中人类贡献的方法极易被操纵——仅需让模型复制几个随机ID的“特异性填充”技巧，就能使未改变文本的评分波动达量级的三分之一到一半，证明似然并非衡量作者身份的可靠指标。 |
| [^77] | [PRISMATIC : Prescription Risk Inspection System for Multi-Agent Treatment Interaction in Clinical Decision-Making](https://arxiv.org/abs/2610.19762) | 提出了PRISMATIC多智能体架构，通过代表不同工作流程角色的专业智能体协作以及动态自更新权重调整（DSWA）等新型自适应协调机制，为多病共存患者迭代生成安全、合理且可解释的用药方案，以解决专科间碎片化决策导致的处方错误问题。 |
| [^78] | [Qwen-MT-Lumen at WMT2026: Instruction-Specialized Translation via Quality-Aware Training and On-Policy Distillation](https://arxiv.org/abs/2610.19758) | 该论文提出面向 WMT 2026 的开源翻译系统 Lumen，通过持续预训练、监督微调、将离线 M²PO 扩展为带 LLM 评判打分的在线偏好优化、以及在线策略蒸馏的四阶段流水线，实现同时满足风格、术语与结构化输出等细粒度指令要求的高质量翻译。 |
| [^79] | [Beyond Reachability: A Relation-Semantic Reasoning Framework for Biomedical Multi-hop QA](https://arxiv.org/abs/2610.19754) | 该论文提出了ReSemReason框架，通过在候选发现与证据选择之间共享问题诱导的类型化关系模式分布，结合模式兼容性图搜索、关系条件化注意力机制和不确定性加权路径先验，解决了生物医学多跳问答中“答案可达但证据无效”的问题。 |
| [^80] | [Does the Linguistic Register of Thinking Matter? Wenyan as a Compact Reasoning Register for Chinese LLMs](https://arxiv.org/abs/2610.19741) | 该论文提出“文言思维”方法，让中文大语言模型用文言文作为紧凑的推理语体，在受限解码预算下大幅缩短推理长度并降低答案缺失率。 |
| [^81] | [MakSafe: A Spatiotemporal Semantic Framework for Formal Specification and Compliance Checking of Road Vehicle Safety Rules](https://arxiv.org/abs/2610.19735) | MakSafe是一个形式化时空语义框架，通过语义区域、动作预测和轨迹树推理对道路车辆安全规则进行规范定义与合规性检查，单次规则检查耗时远低于一毫秒，并能从同一初始场景对不同分支分别得出安全或不安全的判定。 |
| [^82] | [Mitigating Video Hallucinations via Attention-Guided Representation Perturbation](https://arxiv.org/abs/2610.19728) | 提出了ARP-CD，一种无需训练的对比解码框架，通过利用答案生成时的注意力权重引导对视觉隐状态的梯度更新来构建对比负样本，从而有效缓解视频大语言模型的幻觉问题。 |
| [^83] | [Recognizing Dialogue Ambiguity via Candidate Set Visual Grounding](https://arxiv.org/abs/2610.19723) | 该论文提出了候选集视觉定位（CSVG）这一新任务及 CandVG 基准，通过估计与对话上下文一致的目标物体集合来识别对话歧义，并发现更强的视觉定位能力并不一定带来更好的 CSVG 表现。 |
| [^84] | [Articulation Upper Bounds for Language Models](https://arxiv.org/abs/2610.19721) | 本文提出了新的提示优化算法和基于“提示恢复”的传输度量方法，用于研究随着大语言模型能力扩展时概念在模型间传输的上限，从而探究语言表达能力自身的极限。 |
| [^85] | [Beyond State Capacity: Why Retrieval Breaks in Mamba](https://arxiv.org/abs/2610.19709) | 该论文挑战了“Mamba检索失败源于固定状态容量”的普遍假设，证明即使在状态容量不构成瓶颈的查询优先检索设置中Mamba依然显著落后，其根本原因在于词元表示未能编码与查询的相关性，致使选择门控无法正确调控状态更新。 |
| [^86] | [RouteWeaver: Weaving Mode Selection and Execution into Unified LLM Routing](https://arxiv.org/abs/2610.19705) | 提出 RouteWeaver 统一路由框架，在单个 LLM 策略中联合学习路由模式选择（单轮/多轮/智能体）与模式内模型执行，并通过 COMET-GRPO 将优势信号解耦到两个决策层级以实现更精准的信用分配与联合优化。 |
| [^87] | [BANGLAPLAN: A Benchmark for Constraint-Based Planning and Decision-Making in Bangla](https://arxiv.org/abs/2610.19704) | 该论文提出了BANGLAPLAN——首个用于评估大型语言模型在孟加拉语显式约束条件下规划与决策能力的基准，包含1000道涵盖十个任务族、四种语言语域的问题，答案采用统一的JSON结构化格式，并包含可行与不可行问题。 |
| [^88] | [Share the Symbol, Not the Vector: Training-Free Fact Communication Across Language Models](https://arxiv.org/abs/2610.19696) | 提出了SYNC——一种无需训练的符号索引通信接口，使异构语言模型通过共享词表的索引而非内部向量来传递事实信息，从而无需微调或学习翻译器即可实现跨模型协作。 |
| [^89] | [Function-Space Temporal Self-Distillation for Policy Optimization](https://arxiv.org/abs/2610.19687) | 本文提出单策略的函数空间策略优化框架FSPO，以稀疏的历史下一词元分布作为固定时间目标进行自蒸馏，无需维护持久教师模型即可提供稳定密集的词元级监督，并通过特权分歧校正与可靠性自适应锚定提升策略优化效果。 |
| [^90] | [Interlock: From Local Correctness to Task Completion in Long-Horizon Agents](https://arxiv.org/abs/2610.19686) | 本文提出Interlock任务合成框架，通过将任务建模为包含接地边、耦合超边和实现边的类型化完成图，弥合长程智能体“局部正确但任务未完成”的差距。 |
| [^91] | [Beyond Distribution Matching: Selective Margin Preservation for Supervised Fine-Tuning](https://arxiv.org/abs/2610.19685) | 提出SRP-SFT，一种通过铰链惩罚选择性保持非目标token边际关系的软边际目标，在不冻结参考分布的情况下显著提升了监督微调的数学推理与代码生成性能。 |
| [^92] | [Measuring and Reducing Unusable Replies of LLM Agents](https://arxiv.org/abs/2610.19659) | 本文提出“回复可用性”概念并设计了基于读者模型的自动测量方法，发现LLM智能体超过一半的回复对请求者不可用，并通过让智能体明确陈述回复要求将不合格率从0.52降至0.43。 |
| [^93] | [HACHIMI-II: From Role-Play to Learner Behavior Modeling for Faithful Student Simulation](https://arxiv.org/abs/2610.19654) | HACHIMI-II 将学生模拟从让大语言模型角色扮演转变为对学习者行为的第三人称预测，通过从学习者历史构建学生模型并用行为框架约束其下一步动作，从而忠实再现真实学习者的能力分布和典型错误。 |
| [^94] | [Selective Reasoning for Typed Decision Models: When to Start, Read, and Stop](https://arxiv.org/abs/2610.19644) | 该论文在冻结骨干模型上训练共享候选答案读取器，通过收益门控启动推理、自然状态触发器显式读取、熵规则停止生成，实现按需可中断的选择性推理，将决策准确率从 70.96% 提升至 84.38%，同时相比终端读取最多减少 46.30% 的流水线时间。 |
| [^95] | [Model What Matters: Decision-Guided Selective World Modeling for Language Agents](https://arxiv.org/abs/2610.19642) | 提出DGS-WM方法，通过决策引导的观测选择机制让语言智能体的世界建模只聚焦于具有决策价值的观测内容，从而弥补稀疏任务奖励无法监督环境动态理解的局限。 |
| [^96] | [J-tau: A Fully Localized Japanese Benchmark for Tool-Agent-User Interaction](https://arxiv.org/abs/2610.19638) | J-tau是首个公开可用的完全本地化日语基准，通过机器翻译、实体适配和人工审核对tau-bench进行本地化以评估使用工具的LLM智能体，并在此过程中发现并修复了原始tau-bench中的无效任务和分数聚合错误两个缺陷。 |
| [^97] | [Sparse Autoencoders Are a Shared Coordinate System for Symbolic Matching](https://arxiv.org/abs/2610.19636) | 本文提出SAE-IDF方法，利用稀疏自编码器将符号编码为稀疏表示并根据稀有度重新加权，实现了跨异构符号系统的单元匹配，为通过数据驱动方式对齐与合并现有符号系统、构建大规模神经符号模型迈出了关键的第一步。 |
| [^98] | [DAGFormer: Global Topology Prediction for Transformer Cross-Layer Connections](https://arxiv.org/abs/2610.19625) | DAGFormer通过联合训练的拓扑预测器一次性规划整个Transformer模型的全局跨层连接拓扑，在75M至1B多个规模上将困惑度降低17%至25%，并全面超越逐层路由方法。 |
| [^99] | [Divergent Paths: Learning an Outcome-Predictive Representation of LLM Agent Trajectories](https://arxiv.org/abs/2610.19614) | 提出Precog-RM模型，通过元学习在共享嵌入空间中以少样本方式预测LLM智能体轨迹的成败，并可作为GRPO训练的早停控制器，在保持性能的同时最多减少47.3%的训练计算量。 |
| [^100] | [Are We Really Measuring Forgetting? Pitfalls in Evaluating Machine Unlearning for Multimodal LLMs](https://arxiv.org/abs/2610.19613) | 该论文指出了多模态大语言模型机器遗忘评估中存在的三大陷阱——缺乏保留集评估区分概念级遗忘与实例级过拟合、依赖合成知识而非预训练真实知识、以及不公平的方法比较——并据此提出了基于真实世界知识和原则性评估协议的GEMMU基准。 |
| [^101] | [Twenty Days Later: A Pre-Registered Test of History-Based Forecasts of Brand Mentions in Gemini Answers](https://arxiv.org/abs/2610.19608) | 该论文通过预注册的重复实验证明，基于历史数据对Gemini回答中品牌提及率的预测在约20天后依然准确（Brier得分0.0281，接近8小时短期预测水平且优于品牌历史基线），表明大语言模型的品牌提及计数具有足够的时间稳定性，可供实际决策使用。 |
| [^102] | [PN-Mix: Pair-Norm Guided Mixed-Precision Key Cache Quantization for Efficient Long-Output LLM Inference](https://arxiv.org/abs/2610.19605) | PN-Mix通过RoPE通道对层面的查询-键配对范数分析识别量化敏感的键缓存通道对，并在预填充阶段一次性识别、解码全程复用，为其分配更高精度，从而实现高效且低精度损失的长输出大语言模型推理。 |
| [^103] | [PA1-OPD: One-Step Prefix-Augmented On-Policy Distillation for Language Model Pretraining](https://arxiv.org/abs/2610.19601) | PA1-OPD通过在每个前缀处引入学生采样转移后的教师反馈，扩展了知识蒸馏的监督状态覆盖范围，同时利用数据锚定根节点和可微树注意力实现高效并行训练，在多个数据集的继续预训练中显著优于传统知识蒸馏方法。 |
| [^104] | [Moral Combat: Exploiting Moral Foundations Theory to Jailbreak LLMs](https://arxiv.org/abs/2610.19587) | 该论文提出一种基于道德基础理论的多智能体框架，通过诊断有害请求违反的道德基础并将其重构为看似符合道德的表述来越狱大语言模型，揭示了LLMs极易被“道德化”手段绕过安全防护栏的安全漏洞。 |
| [^105] | [Who's Behind the Text? Robust Model Authorship Attribution](https://arxiv.org/abs/2610.19583) | 提出基于似然的模型作者归属方法ATLAS，能够在未见领域、未知写作场景以及模型家族未来版本等分布偏移下依然稳健地识别文本的来源大语言模型。 |
| [^106] | [INTACT: Inducing Latent Reasoning on a Frozen Backbone](https://arxiv.org/abs/2610.19568) | INTACT 通过小型外部编码器结合蒸馏与强化学习，在完全冻结的大模型上诱导潜在推理，在不改变模型行为的前提下显著提升数学推理准确率并减少自回归解码开销。 |
| [^107] | [REWIND: Uncertainty-Guided Trajectory Repair for Test-Time Reasoning](https://arxiv.org/abs/2610.19561) | 研究发现错误推理伴随显著更高的token熵，据此提出无需训练的REWIND方法，利用熵信号定位并重新生成推理轨迹中的错误后缀，在多个模型和基准上相比最强基线提升0.51-10.00个百分点。 |
| [^108] | [$\tau$-Emotion-Bench: Exposing the Interaction Fragility of LLM Agents through Real-World User Emotions](https://arxiv.org/abs/2610.19560) | 提出τ-Emotion-Bench十领域基准和Tracer可扩展数据生成流水线，通过模拟真实用户的情绪化交互与终止低效对话的机制，评估并提升LLM工具调用智能体在面对情绪化用户时的交互可靠性。 |
| [^109] | [Evidence-Anchored Overdose Death Prediction from Death Investigation Reports](https://arxiv.org/abs/2610.19556) | 该论文提出了一个从死亡调查报告中预测疑似药物过量死亡的自动化框架，通过专家引导生成证据锚定的解释，并比较了流水线模型与自我合理化模型在该任务上的性能差异。 |
| [^110] | [GraphRAG-LTM: A Layered Threat Modeling Framework for Attacks on Graph Retrieval-Augmented Generation](https://arxiv.org/abs/2610.19549) | 该论文提出 GraphRAG-LTM 分层威胁建模框架，通过黑盒查询、白盒查询和图感知白盒三个递进攻击层级，首次系统揭示了 GraphRAG 证据生成路径的安全脆弱性。 |
| [^111] | [Typed Decision Models on Tabular Data](https://arxiv.org/abs/2610.19536) | 类型化决策模型在表格数据任务上以更低成本达到与大语言模型相当的性能，且对问题表述更加鲁棒，但所有模型对表格组织方式的变化都不够稳健。 |
| [^112] | [A Unified Response Framework for Questions and Tutoring Dialogues](https://arxiv.org/abs/2610.19532) | 提出了一个统一框架，将学生对问题的回答与辅导对话中导师提出的子任务共同建模在项目反应理论（IRT）量表上，通过文本估计难度、追踪学生动态能力演化，并利用导师干预行为正则化难度值，实现对学业知识的更细粒度评估。 |
| [^113] | [Retrieve What Helps, Only When It Helps: Agentic Reinforcement Learning for Selective Retrieval-Augmented Machine Translation](https://arxiv.org/abs/2610.19528) | 该论文提出SRAT智能体强化学习框架，让大语言模型在翻译中自主学习何时需要检索外部知识、检索什么证据以及如何整合，从而只在检索真正有助于翻译质量时才进行检索。 |
| [^114] | [Diagnosing and Correcting Reward Hacking in Verifiable Instruction Following](https://arxiv.org/abs/2610.19525) | CALIF 是一种轻量级方法，通过利用现有冻结参考策略计算经校准的基于似然的奖励加成来增强验证器奖励，无需额外评判模型即可缓解可验证指令遵循中的奖励作弊问题并保持回复质量。 |
| [^115] | [One Block Ahead: Promoting Sampler State in Diffusion Language Models](https://arxiv.org/abs/2610.19510) | 提出了一种无需训练的OBA解码算法，通过在去噪当前区块的同时提前解码下一区块并提升相应的采样器状态（包括自条件信号和去噪时间），显著提升了扩散语言模型每次前向传播的token生成量。 |
| [^116] | [Structure-Preserving Document-Level Data Augmentation via Discourse Parsing](https://arxiv.org/abs/2610.19497) | 提出了一种结合大语言模型与文档级篇章结构解析的数据增强框架，通过识别句间修辞关系并替换关键短语，在保留原文篇章结构的同时生成主题偏移的文档级增强数据，有效提升摘要与问答任务性能。 |
| [^117] | [Towards Efficient Simultaneous Inverse Text Normalization with Pretrained Text-to-Text Language Model and Read-Tag-Write Policy](https://arxiv.org/abs/2610.19492) | 该论文提出了一种基于预训练文本到文本语言模型的高效流式端到端逆向文本规范化系统，通过架构适配、专门训练策略和“读-标注-写”解码策略实现流式推理，其准确性媲美非流式基线并超越混合方法。 |
| [^118] | [AgriJev: Efficient Agricultural Candidate Scoring with Multi-Scale Adapted Models](https://arxiv.org/abs/2610.19478) | AgriJev是一系列多尺度适配的Qwen3.5农业模型，通过单次前向调用直接对候选选项评分而非生成文本答案，在17个外部测试视图上平均准确率达75.44%，超越原始模型和农业SFT生成模型。 |
| [^119] | [What Does LALM Post-training Learn? Rethinking Audio Understanding and Reasoning Capabilities](https://arxiv.org/abs/2610.19476) | 研究通过受控实验发现，大型音频-语言模型后训练带来的性能提升主要源于对特定问题格式和文本线索的适应，而非音频理解与推理能力的真正提升。 |
| [^120] | [ActICL: Post-Training for Agentic Multimodal In-Context Learning](https://arxiv.org/abs/2610.19475) | ActICL通过后训练使多模态大模型智能体在迭代推理中自主分析信息缺口，并借助工具增强推理进行渐进式的针对性演示检索，从而自主构建多模态上下文学习所需的上下文。 |
| [^121] | [More Than Seeing: Vision as Scalable Anchors in Native Multimodal Pre-Training](https://arxiv.org/abs/2610.19443) | 该研究通过受控消融实验发现，在原生多模态预训练中保留视觉内容能在不损害纯文本性能的前提下提升图文样例上的文本预测，多模态收益随计算量呈幂律增长，且视觉信号还可作为锚点提升纯文本空间推理等超越视觉理解的能力。 |
| [^122] | [How Do LLMs Understand Legal Philosophy? Diagnosing Legal-Philosophical Reasoning in LLMs](https://arxiv.org/abs/2610.19427) | 该论文提出LePhilBench基准，通过“理解法律哲学”与“应用法律哲学”两项任务及人工精心构建的数据集，首次系统评估了大语言模型的法律哲学推理能力。 |
| [^123] | [ForkGRPO: Efficient Exploration and Fine-Grained Credit Assignment for Diffusion Language Models](https://arxiv.org/abs/2610.19408) | 提出ForkGRPO强化学习框架，通过从参考轨迹的高熵中间状态分支采样多个续写实现高效探索，并引入细粒度的片段级信用分配，从而提升扩散语言模型的推理能力。 |
| [^124] | [Vision-Language Models Integrate Conflicting Evidence Like Biased Bayesian Observers](https://arxiv.org/abs/2610.19407) | 本文提出一个贝叶斯视角的双参数线索组合模型，通过将证据强度与模态整合偏差分离，准确预测视觉-语言模型在冲突多模态输入下的偏好并无需重新拟合即可泛化到新的不可靠性形式，在单一规则下统一解释了文本主导与视觉主导现象。 |
| [^125] | [KoAGE: Do Large Language Models Know, Learn, and Coin Korean Neologisms?](https://arxiv.org/abs/2610.19399) | 该研究提出NeoTrace自动化框架与KoAGE基准，系统评估大语言模型对韩语新词的认知与创造能力，发现模型虽能从语境中推断新词含义，但对构词结构的理解仍存在明显局限。 |
| [^126] | [ELARSpeech: A 30K-Hour Dataset Across 675 Underrepresented Languages for Long-Tail Multilingual Speech Technologies](https://arxiv.org/abs/2610.19388) | ELARSpeech是一个包含675种代表性不足语言、总时长30,202小时的大规模多语言语音数据集，为将语音识别与语音合成技术扩展至语言长尾领域提供了重要资源和测试平台。 |
| [^127] | [The Sum of All Summaries: PROPALIGN for Graded Proposition Salience Evaluation](https://arxiv.org/abs/2610.19387) | 本文提出基于多摘要的分级命题显著性评分方法PROPALIGN及配套数据集PROPALIGNBANK，证明摘要式评分比直接询问模型更稳定，且结合语言特征的模型集成可接近人工标注质量上限。 |
| [^128] | [Learning What to Keep: Transfer-Aware Personal Skill Learning from User Feedback](https://arxiv.org/abs/2610.19380) | 提出 SkillGrain 框架，将用户反馈分解为可独立执行的原子规则并预测其可迁移性，通过基于评分准则的奖励、细粒度信用分配以及强化学习联合优化，从用户反馈中选择性地保留真正可迁移的个性化技能。 |
| [^129] | [Evaluating Compressed Memory: Dependence, Utility, and Content Benefit](https://arxiv.org/abs/2610.19373) | 该论文提出将记忆置零与无法访问存储文本的对照模型相结合来严格评估压缩记忆的真实效用，发现某些依赖记忆的模型性能反而不如无记忆训练的替代模型。 |
| [^130] | [Beyond Corrected Actions: Training LLM Agents on Textual Critiques](https://arxiv.org/abs/2610.19367) | 提出CAFT框架，让专家模型在学生LLM智能体生成的错误动作上同时提供纠正动作和解释性文本批评，并证明基于这种更丰富的文本反馈进行训练能更有效地提升智能体性能。 |
| [^131] | [Seeing, Maintaining, and Learning: The Evolution of Multimodal Agent Memory](https://arxiv.org/abs/2610.19366) | 该论文提出了一个双轴分析框架，从多模态信息的形成与表示、以及基于积累经验的自适应记忆管理两个维度，系统性地梳理了多模态智能体记忆的研究演进。 |
| [^132] | [Tail Competition Explains Scaling in Best-of-N Sampling for Verifiable Problems](https://arxiv.org/abs/2610.19362) | 该论文将 Best-of-N 采样建模为正确与错误响应极端奖励尾部之间的竞争，利用极值理论刻画出正确尾部主导、错误尾部主导和临界三种渐近状态，统一解释了缩放增益与奖励作弊导致的性能退化，并推导了有限样本缩放定律。 |
| [^133] | [VLX-Go: Robust Embodied Tracking through Multi-Stage Learning and On-Policy Refinement](https://arxiv.org/abs/2610.19361) | 该论文提出 VLX-Go——一种视觉-语言路径点策略，其核心创新在于先对齐视觉与语言表征、再通过多阶段学习与在线策略优化来训练跟踪运动，从而显著提升机器人在存在干扰者场景中跟随指定人物的鲁棒性。 |
| [^134] | [Steering Vectors for Competing Ethical Frameworks Are Interchangeable: A Competing-Target Test for Activation Steering](https://arxiv.org/abs/2610.19359) | 论文提出“竞争目标测试”这一新方法，发现用于康德义务论与行为功利主义的激活转向向量共享大部分方向且可互换，说明转向向量可能通过共享组件起作用而非直接控制目标行为。 |
| [^135] | [Where to Start, How to Move: Analyzing Curricula under GRPO for Long-Horizon LLM Agents](https://arxiv.org/abs/2610.19349) | 该研究系统分析了引导辅助GRPO课程设计，发现在固定回合级阈值规则下“从接近完成的阶段开始”最有效，且基于结果信息的工具调用选择在已训练和未训练任务类别上均优于标准GRPO。 |
| [^136] | [Target-Token Asymmetry Can Masquerade As Depth Specialization](https://arxiv.org/abs/2610.19344) | 层级消融研究中所观察到的“早期层专门化”现象，可能仅仅是两个语料库之间目标词元分布不对称（如数学文本中大量空白字符）所造成的假象，而非真正的深度专门化。 |
| [^137] | [Counterfactual Gate-Support Decomposition: Separating Selection from Redistribution in Training-Data Filtering](https://arxiv.org/abs/2610.19343) | 该论文提出反事实门控支持分解（CGSD），一种四臂预算匹配的实验设计，可将训练数据质量门控的整体效应分解为支持内选择、支持间重分配及其交互作用，并在韩英与韩日检索任务中发现支持内选择是唯一显著超出校准零区间的分量。 |
| [^138] | [LLM-jp-4 Technical Report](https://arxiv.org/abs/2610.19331) | LLM-jp-4是强调透明性与可复现性的日本主权大语言模型系列，包含8B密集模型和32B-A3B混合专家模型，均在约12万亿token上从头训练，性能较之前模型显著提升。 |
| [^139] | [TopicMerge: A Divide and Conquer Approach to Topic Modeling](https://arxiv.org/abs/2610.19323) | TopicMerge提出一种分治式的多阶段流水线，先通过高效聚类将文档过度分割为语义纯净的微聚类，再由LLM结合局部邻域上下文迭代仲裁合并候选对，从而同时实现了最先进的聚类性能与计算效率。 |
| [^140] | [Truthful and Expected-Payment-Optimal Model Routing in Graph-Based Agentic Workflows](https://arxiv.org/abs/2610.19316) | 本文将基于图的智能体工作流中的LLM模型路由问题形式化为单参数采购拍卖，提出STAR机制，在保证提供商如实报价且覆盖其执行成本的同时，实现期望支付最优。 |
| [^141] | [PersEvalBench: Can LLM Evaluators Judge Contextually Appropriate Use of Persona Information?](https://arxiv.org/abs/2610.19314) | 提出PersEvalBench基准，通过403个多轮场景和九类受控个性化错误测试LLM评估器能否判断用户历史信息的使用是否恰当，发现即使拥有完整对话历史，LLM评估器识别这些错误的能力仍然有限，其中在个性化不足错误上与人类94.4%的识别率差距最大。 |
| [^142] | [Qwenfant: Train Big Character-level BabyLMs on Concise Parallel Text](https://arxiv.org/abs/2610.19294) | 该论文通过使用LLM总结翻译的合成数据、更大的模型和字符级词表（结合LoRA微调）训练出Qwenfant模型，在2026年BabyLM挑战赛多语言赛道中获得第一名。 |
| [^143] | [Can LLM Agents Efficiently Identify Appliance Faults? Insights From Diagnosing a Simulated Appliance](https://arxiv.org/abs/2610.19291) | 该论文提出了CoffeeBench——一个基于模拟家电的LLM智能体故障诊断基准，发现LLM难以仅凭传感器信息稳定识别故障，但表现最佳的模型能通过迭代缩小故障候选空间实现成功诊断，并进一步分析了不同模型的诊断效率差异。 |
| [^144] | [From Efficiency to Sufficiency: The Cost-Aware Measurement Design Framework for Automated Essay Scoring](https://arxiv.org/abs/2610.19287) | 本文提出成本感知测量设计（CAMD）框架，将自动作文评分从单纯的技术效率问题重新构建为成本受限的测量系统，通过计量、分摊、单位经济学和决策四层结构，将计算与工作流资源记录转化为可直接支持部署决策的成本构念。 |
| [^145] | [Whose Data, Whose Benchmark? A User-Centric Framework for Auditing Evaluation Resources](https://arxiv.org/abs/2610.19268) | 该论文提出了一个以用户为中心的基准测试审计框架，通过七个维度对评估资源进行审计，揭示了现有基准中美国中心惯例占主导地位、地理可见性与GDP强相关、以及来源用户情境被纳入而目标用户情境完全缺失的系统性偏差。 |
| [^146] | [MemeTrace: A Multimodal LLM-Based Agent for Safety-Aware Meme Provenance Analysis](https://arxiv.org/abs/2610.19267) | 提出 MemeTrace，一个基于多模态大语言模型的智能体，能够自动追踪网络迷因的来源、传播路径及潜在社会风险，在包含100个真实迷因的 Meme100 数据集上显著优于 VLM 基线。 |
| [^147] | [Copy-Paste Language Models](https://arxiv.org/abs/2610.19247) | 提出复制-粘贴语言模型（CPLMs），通过在传统语言建模头之外增加一个轻量级复制-粘贴头，以几乎可忽略的开销显著提升大语言模型的复制能力、训练效率、下游性能以及长上下文和检索表现。 |
| [^148] | [Do Syntax Heads Influence Model Behavior?](https://arxiv.org/abs/2610.19231) | 研究发现，尽管Transformer中的句法注意力头看似专门处理句法关系，但其对句法预测的因果作用证据有限：强制干预这些注意力头并不能改变模型对主语的预测偏好，且其干预效果仅在两种指标之一上优于随机注意力头。 |
| [^149] | [Titan-Llama: Retrofitting Pre-trained LLMs with External Neural Memory Modules](https://arxiv.org/abs/2610.19210) | Titan-Llama 提出了一种后训练方法，为预训练的注意力型大语言模型加装可在测试时更新参数的分层神经记忆模块，在保留近期 token 全保真注意力的同时将历史上下文压缩为固定大小的记忆，从而兼顾长上下文推理的效率与信息保留精度。 |
| [^150] | [DUALVIEW: Leveraging Broad and Output-Specific Evidence for LLM-based Multi-Output Text Regression](https://arxiv.org/abs/2610.19206) | DualView是一种参数高效的双分支方法，通过从冻结的大语言模型中提取广泛表征和输出特定表征来实现多输出文本回归，在五个任务上以远少于基线的参数量超越了监督回归方法，甚至可超越更大模型的无监督预测。 |
| [^151] | [Weighted Ensemble Sampling for Rare-Event Estimation in Language Models](https://arxiv.org/abs/2610.19202) | 提出了一种无需训练的加权集成（WE）框架，仅从原始语言模型采样，并利用模型内部表示作为前缀级信号、周期性地将采样预算重新分配给更有希望的前缀，从而高效估计语言模型稀有事件的概率。 |
| [^152] | [NEPSEE: A Double-Annotated Benchmark for Nepali Sentiment Analysis](https://arxiv.org/abs/2610.19190) | 该论文构建了一个由两名标注者独立盲标的双标注尼泊尔语情感分析基准NEPSEE，发现商业大语言模型在细粒度五级情感标注任务上显著优于最佳编码器模型，并揭示了现有尼泊尔语BERT模型分词器会丢失天城文元音符号的问题。 |
| [^153] | [How to Evaluate When You Don't Know How to Evaluate? Choosing Reliable Setups for Base Language Model Benchmarking](https://arxiv.org/abs/2610.19185) | 本研究系统考察了任务表述、上下文示例数量等8个评估设计维度，并利用单调性、非随机性和排名一致性三种元度量标准，为基础语言模型在早期预训练阶段选择可靠的基准测试设置提供指导。 |
| [^154] | [HERO: Harnessing Evidence for Test-time Refinement and Prompt Optimization in Program Generation](https://arxiv.org/abs/2610.19184) | HERO提出了一种结合离线与测试时优化的框架，通过执行验证的成功与失败证据学习可复用的提示更新，并将测试时生成的代码作为实例特定的harness加以改进，从而提升文本到SQL/Python等程序生成任务的可靠性与准确性。 |
| [^155] | [Fantastic Metrics and How to Find Them: Meta-Evaluating Metrics for Multi-lingual Style-Personalized Generation](https://arxiv.org/abs/2610.19166) | 提出基于回归的指标元评估框架，以预测人类评分替代传统相关性方法，并构建了覆盖10种语言、4个评估维度的大规模多语言风格个性化文本生成基准来检验指标的真实评估能力。 |
| [^156] | [ListQA: A Benchmark for Evaluating List-Formatted Factual Knowledge Retrieval in Large Language Models](https://arxiv.org/abs/2610.19165) | 该论文提出了ListQA基准，包含9,045个人工策划并交叉验证的问题，用于评估大语言模型回忆多个事实并组织成结构化列表的能力，并采用最优二分图匹配与LLM语义评分相结合的评估方法来衡量事实准确性与幻觉。 |
| [^157] | [From Qualitative Text to Integrated Evidence: The Parallel-Then-Integrated (PTI) Procedure](https://arxiv.org/abs/2610.19164) | 本文提出了PTI（平行-后整合）这一四步程序，用于在混合方法研究中显式且可追溯地整合同一定性语料库的两种计算表征（本体论指导的PV编码与NMF数据驱动主题），并以尼泊尔母亲的访谈数据和PHQ-9抑郁评分作为外部锚点加以演示。 |
| [^158] | [Into the Unknown: Benchmarking AI agents on open scientific research problems](https://arxiv.org/abs/2610.19159) | 该论文推出了"Into the Unknown"基准测试，通过与能源部国家实验室科学家合作定义未发表的后续研究方向及评分准则，评估九个前沿AI智能体在开放科学问题上撰写后续论文的能力，结果显示135篇投稿无一满足全部条件，其中114篇未通过完整性检查。 |
| [^159] | [DivAlgo: A systematic evaluation of algorithmic diversity for code generation models](https://arxiv.org/abs/2610.19157) | 提出DivAlgo基准数据集及轻量级相似度度量方法，通过对24个代码生成模型的评估发现，模型生成代码的算法多样性显著低于人类解决方案（最高仅达人类多样性的约70%），揭示了当前代码生成模型在算法多样性方面的局限性。 |
| [^160] | [MA-Bench: Measuring Operational Planning in Language Models](https://arxiv.org/abs/2610.19138) | 提出了基于军事作战规划第一步“任务分析”构建的MA-Bench基准，用于评估语言模型在缺少唯一正确答案的情况下，从不完整且矛盾的文档集合中形成态势分析报告的能力。 |
| [^161] | [CAFE: Confining Feature Distillation to Language-Routed Adapters for Compact Multilingual Automatic Speech Recognition](https://arxiv.org/abs/2610.19134) | 提出CAFE方法，通过将CKA特征蒸馏梯度限制在英语绕过的语言路由适配器中，从参数层面解决了紧凑型多语言语音识别中非英语与英语性能之间的权衡问题。 |
| [^162] | [Plant Memory: Technical Language Processing and a Verified Maintenance Ontology for Agent-Accessible Failure Knowledge from SAP PM Work Orders](https://arxiv.org/abs/2610.19131) | 提出了Plant Memory系统，通过规则优先、LLM辅助的技术语言处理，将难以分析的SAP PM维护工单转化为符合ISO 14224标准、每条事实均带置信度、证据溯源与核验状态的维护本体，使设备故障知识可被智能体访问，避免诊断知识随资深技师流失。 |
| [^163] | [Unlocking Structured Pruning for Reasoning Language Models via Attention-Preserving MLP Trimming](https://arxiv.org/abs/2610.19121) | LayerTrim是一种免训练的结构化剪枝方法，通过识别冗余解码器层并选择性裁剪其MLP特征、同时保留注意力机制和解码器深度，在无需梯度或恢复训练的情况下保持长篇推理能力，相比RESP平均准确率最高提升21.4个百分点，并实现1.24倍的预填充加速和19.7%的GPU峰值内存降低。 |
| [^164] | [DecisionRobust: Benchmarking Robustness for JEV Style Decision Models Across Text and Vision](https://arxiv.org/abs/2610.19118) | 本文提出DecisionRobust基准，通过涵盖语义保持文本扰动、图像扰动和提示注入等多种攻击方式，系统性评估了JEV风格决策模型在文本与视觉输入下面对对抗性操纵的鲁棒性。 |
| [^165] | [From Copying to Composition: How Language Models Develop In-Context Learning](https://arxiv.org/abs/2610.19113) | 该研究通过受控诊断方法发现，语言模型会从倾向于复制提示中最近示例，逐渐发展为整合提示信息进行组合式推理的上下文学习，且这种转变出现得很早，并可通过特定的合成数据训练加以引导。 |
| [^166] | [Towards a Data-Centric Understanding of Spatial Reasoning in Multimodal Large Language Models](https://arxiv.org/abs/2610.19108) | 本综述以数据为中心系统梳理了用于多模态大语言模型空间推理的微调方法，将研究归纳为数据集设计、数据集构建与评估三大类，并总结出规则与模板、LLM 提示、程序化生成和人在回路四种主要的空间标注构建方法。 |
| [^167] | [GroundGUI: GUI Grounding by High-Quality Data and Distillation-to-Reinforcement Learning](https://arxiv.org/abs/2610.19100) | 本文提出了涵盖桌面、网页和移动端的大规模GUI定位数据集ScreenRef（包含586K张截图和633K个指代表达实例），并提出蒸馏到强化学习（D2RL）方法，以克服监督微调泛化能力有限的问题，显著提升GUI智能体的视觉定位能力。 |
| [^168] | [Similar Models, Broken Odds: Where Luce's Axiom Fails in Chatbot Arena](https://arxiv.org/abs/2610.19095) | 该研究发现Chatbot Arena所依赖的Bradley-Terry模型（即Luce选择公理）在高度相似的模型之间会系统性失效——相似模型间的对决比分数预测的更具决定性（最高达1.37倍），并提出一个仅含4-6个参数、主要通过建模平局的修正方法，其预测能力超越了Bradley-Terry及含数百参数的复杂模型。 |
| [^169] | [Beyond Aggregate Accuracy: Item-Level Transitions in Prompt Optimization](https://arxiv.org/abs/2610.19084) | 该论文提出RFDPO方法，通过追踪提示优化过程中逐题的“恢复、退化与变动”转变，揭示总体准确率所掩盖的题目级变化，并利用这些反馈指导后续的提示修订。 |
| [^170] | [Smart Alone, Blind Together: Why Accuracy Fails to CertifyTheory of Mind in Multi-Agent Teams](https://arxiv.org/abs/2610.19051) | 该论文论证准确性不能作为认证多智能体团队中心智理论模块的标准，因为所需准确性取决于部署场景的错误代价，且模块间错误的相关性会使准确性阈值以可预测的方向失效。 |
| [^171] | [Pre-Trained Language Models as Zero-Shot Tools for Social Science Research](https://arxiv.org/abs/2610.19048) | 该论文证明预训练语言模型无需任务特定训练即可完成社会科学中的记录链接和种族推断任务，提出的EnsembleLink算法达到或超越现有方法的表现，且大语言模型在美国选民档案种族推断上达到83.8%的准确率，超过传统BISG方法的78.0%。 |
| [^172] | [Graph Coarsening Enables Escaping Local Optima in Compute-Constrained Nonconvex MAX-CUT](https://arxiv.org/abs/2610.19043) | 该论文提出将近线性时间的谱图粗化方法CMG用于热启动具有优良优化景观的粒子多项式松弛（PPR），使计算受限的非凸MAX-CUT优化能够逃离局部最优，在基准实例上平均达到97.5%的最优解、并在全部Biq Mac测试实例上达到已发表最优解，还可扩展至两万顶点规模并大幅降低内存需求。 |
| [^173] | [Tracing Morphology, Semantics, and Divergence in Language Acquisition](https://arxiv.org/abs/2610.19041) | 该论文通过对 Pythia 和 OLMo 等开源模型中间训练检查点的系统评估，揭示了语言模型先习得形态结构、语义嵌入随后才趋于稳定的习得轨迹，并发现训练数据规模对发散识别有积极影响。 |
| [^174] | [Who Should We Listen to More? Welfare-Aware Preference Acquisition for Pluralistic Alignment](https://arxiv.org/abs/2610.19035) | 提出福利感知偏好获取（WAPA）方法，通过基于个体监督效用的社会福祉目标在人群中分配有限的人类反馈，解决多元对齐中因AI模拟精度差异导致的偏好代表性不足问题。 |
| [^175] | [Crossing the Gender Binary: Modelling Gendered Reference in Science Fiction and General Fiction](https://arxiv.org/abs/2610.19028) | 本研究通过遮蔽性别词并利用 RoBERTa 对替代词进行建模，发现科幻小说与一般小说在语言层面均存在双向跨越性别二元的指称现象，但具有不对称性——woman 被遮蔽时更易被阳性词替代——而科幻小说在这一共同模式之上展现出更大的性别灵活性。 |
| [^176] | [Joint Subspace Sparse Attention for Efficient Long-Context Decoding](https://arxiv.org/abs/2610.19026) | 该论文提出联合子空间稀疏注意力（JSSA），通过最小化词元分数保真误差学习共享的低维查询-键子空间并获得闭式谱解以实现高质量词元级稀疏选择，同时设计了无需聚集操作的索引注意力内核，从而显著提升长上下文大语言模型解码的效率。 |
| [^177] | [Can LLMs Access their World Knowledge for Event Prediction?](https://arxiv.org/abs/2610.19021) | 本文提出通过“自然语言事件描述”和“将事件语义与符号化预测标签分离”的接口设计，让 LLM 有效调用其世界知识进行事件预测，并推出包含约 15 万事件、横跨七个数据集和六个领域的 DescribeEvent 基准验证了该方法优于无描述变体和非 LLM 基线。 |
| [^178] | [Evaluating Large Language Models for Multilingual Abstractive Summarization: A Comprehensive Study of English and Bengali](https://arxiv.org/abs/2610.19003) | 该研究构建了一个标准化零样本基准，综合词汇、语义、事实性和LLM评审等多维指标，全面评估了八个大语言模型在英语和孟加拉语摘要任务上的表现，揭示了显著的跨语言性能差异，且不同评估方法会得出不同的模型排名。 |
| [^179] | [The Agentic Scientist: Orchestrating Coding Agents for Automated Scientific (Co-)Discovery](https://arxiv.org/abs/2610.18933) | 提出了端到端分布式系统“智能体科学家”，将计算研究分解为多个阶段并由新一代编程智能体驱动、结合智能体自验证，实现人机协同的自动化科学发现。 |
| [^180] | [Following the Flow the Law Requires: Training Voice Agents to Transfer to Unseen Regulated Procedures in Debt Collection](https://arxiv.org/abs/2610.18881) | 本文提出 FlowSpec 机器可读规范，通过在 297 个随机化监管程序上微调 90 亿参数模型，训练出能够零样本遵循从未见过的受监管债务催收流程的语音代理。 |
| [^181] | [Lotus: Learning to Understand Time-series by Learning to Forecast](https://arxiv.org/abs/2610.18877) | LOTUS 提出“预测锚定”预训练原则，通过要求时间序列的语言对齐表示能够预测未来片段演化，使其在映射到 LLM 嵌入空间时保留时间与频谱信息，从而学会真正理解时间序列。 |
| [^182] | [Read the Room Before Your Model Generates](https://arxiv.org/abs/2610.18820) | 提出 Scenario-Actor 框架，从 Transformer 隐藏状态中提取交际场景的结构化几何表征（场景嵌入），据此自适应地选择解码温度，从而在交互社交场景和开放式指令遵循任务上显著提升 LLM 的表现。 |
| [^183] | [Back-Translation Still Matters: Leveraging Monolingual Data for Low-Resource Machine Translation with LLMs](https://arxiv.org/abs/2610.18818) | 对于低资源机器翻译，利用单语数据做回译比继续预训练大语言模型更有效，仅需1K平行句对即可超越继续预训练7.7至16.4个XCOMET-XXL分，甚至优于拥有10倍平行数据的模型。 |
| [^184] | [Pack Light, CarryOn: Building Object-Oriented Context for Long-Horizon Agents](https://arxiv.org/abs/2610.18784) | 提出 CarryOn，一个面向对象的上下文运行时，通过围绕可复用的任务对象（而非交互历史）来组织长时程智能体的上下文，使其能够在折叠过程中高效更新续接载体记录而无需重建完整表示。 |
| [^185] | [Does Pretraining Teach Chemical Language Models Chemistry?](https://arxiv.org/abs/2610.18651) | 化学语言模型的预训练收益无法随规模扩大，性能低于梯度提升基线，仅有基于理化描述符的有监督预训练能持续带来提升，表明预训练几乎未能让模型真正学到化学知识。 |
| [^186] | [Readout Scaling Shapes Sparse Feature Recovery](https://arxiv.org/abs/2610.18462) | 本文提出读出规范化决策空间（RC-DS）坐标方法以消除读出缩放的奇异值增益，并通过受控实验证明当训练恢复因子生成的原始空间时稀疏特征恢复效果最佳，将平均稀疏因子恢复从0.506提升至0.913。 |
| [^187] | [Fish Audio Drama 3 Technical Report](https://arxiv.org/abs/2610.18010) | Fish Audio Drama 3 是一个基于 35B 参数混合专家架构的开源语音生成模型，能将文本转化为有导演指导的表演，并首次在单一模型中集成了多轮修改、场景配音、填充编辑和流式输入等多项新能力。 |
| [^188] | [SkillScreen: High-Recall Screening of Malicious Agent Skills with Decision Models](https://arxiv.org/abs/2610.17727) | SkillScreen 是一个紧凑的决策模型，能在单次前向传播中直接对 SKILL.md 技能的恶意程度评分，在未见攻击族上达到 99% 召回率且误报远低于词级分类器和 LLM 判官，但对陌生良性来源的误报仍会显著上升。 |
| [^189] | [RELAY: Selective Self-Distillation with Adaptive Hand-off for Rubric-Based Reinforcement Learning](https://arxiv.org/abs/2610.17623) | RELAY提出一种两阶段训练方法，第一阶段将冻结教师模型的选择性自蒸馏与GRPO结合，并通过监控蒸馏损失的停滞检测器自适应地在最佳时机移除教师模型、切换到纯GRPO训练，从而在基于评分标准的强化学习中超越纯GRPO方法，且停滞检测器跨模型和领域具有良好泛化性。 |
| [^190] | [ANCHOR: Assessing Naturalistic Constraint Hierarchies and Ordering in Repository-Grounded Agentic Coding](https://arxiv.org/abs/2610.16859) | 提出 ANCHOR 基准，通过层次化约束分类体系、自动化约束组合流水线和基于轨迹的评估框架，首次系统性地评估基于仓库的编码智能体遵循工程约束（如修改范围、工作流顺序、验证等）的能力，弥补了现有评估仅关注功能正确性的不足。 |
| [^191] | [REVEAL: Reverse Engineering Agentic Latents from Frontier-Model Traces](https://arxiv.org/abs/2610.16288) | 提出免训练框架REVEAL，通过逆向工程从前沿模型轨迹中提取“智能体潜变量”，为冻结的小型开源智能体提供可复用的局部引导控制，无需训练目标模型或访问源模型内部。 |
| [^192] | [Review Rarely Beats the Stronger Model: Separating Verification from Capability Transfer in LLM Pipelines](https://arxiv.org/abs/2610.15897) | 论文揭示了 LLM 评审流水线中“验证”与“能力迁移”的本质区别——评审很少超越更强的单模型、因锚定效应丢弃自身正确答案且成本更高，而简单的答案选择器是更准确又便宜的替代方案。 |
| [^193] | [Who Sets the Pace of Judge Hacking? Judge Capability versus the Policy-Judge Gap](https://arxiv.org/abs/2610.15778) | 该研究发现强化学习“攻击评委”的深度由评委自身能力决定（与评委AUC呈负相关，系列内更大的评委更难被欺骗），而攻击速度反而与“策略-评委能力差距”的解释相矛盾——差距更大的组合被欺骗得更早，且所有组合均未带来真实的保留集准确率提升。 |
| [^194] | [Do LLM Tutors Actually Teach? A Process-Centered Synthesis of the Tutoring Loop](https://arxiv.org/abs/2610.15692) | 本文通过对大语言模型辅导流程的以过程为中心的系统综述发现，尽管教学评估较普遍，但证明真实学习收益、尤其是知识迁移与延迟保持的证据薄弱，且模型辅导效果依赖于教学结构、学习者参与度和学习衡量方式。 |
| [^195] | [Hidden in Plain Sight: Advancing Harmful Content Understanding via Perception-Aware Adversarial Self-Play](https://arxiv.org/abs/2610.15449) | 提出PASP框架，通过融合文本、语音、视觉等多感知视角与对抗性自博弈机制，弥合人类读者与LLM之间的认知不对称，从而提升模型对不断演化的伪装性有害内容意图的理解能力。 |
| [^196] | [Learning Proposition-Level Links from Sentence-Level Supervision for Neuro-Symbolic NLI](https://arxiv.org/abs/2610.15158) | 该论文提出一种可训练的原子链接器，无需原子对标注，通过将跨公式链接结构视为潜在变量并仅从句子级NLI标签进行训练，即可联合预测命题间链接的存在性、源选择和关系类型，从而提升神经符号自然语言推理系统的性能。 |
| [^197] | [AlphaRLM: Aligning Language Models for Literature-Grounded Symbolic Factor Discovery](https://arxiv.org/abs/2610.14959) | AlphaRLM是一个强化学习框架，通过文献引导算子组装（LEGO）提供语义先验，并借助可执行有效性与分组收益分离等奖励将生成策略与市场反馈对齐，从而训练大语言模型直接从金融文献中构造可执行的符号化Alpha因子。 |
| [^198] | [How to Build a Benchmark That Actually Measures Taste](https://arxiv.org/abs/2610.14707) | 该论文提出TasteBench——一个测量真实品味判断力的两两对比基准，并通过对照协议揭示并量化了基准测试中的三种效度威胁：位置偏差、词汇泄露和难度天花板。 |
| [^199] | [RealityBench: Simulated Environments to Evaluate a Vision Language Model's Ability to Document Human Physical Work](https://arxiv.org/abs/2610.14397) | 该论文提出RealityBench基准，通过在Blender中模拟十一个真实体力工作环境，评估视觉语言模型在协助记录人类物理工作（包括判断图像可接受性和生成工作文档）方面的能力。 |
| [^200] | [Evaluating LLMs as Agentic Workflow Executors in Real-World Customer Support Skills](https://arxiv.org/abs/2610.14346) | 该论文提出了一个从真实客户支持对话记录中挖掘并构建的基准，用于评估大语言模型在策略与工具约束下执行有状态的多轮工具调用工作流的能力，并对17个闭源与开源模型进行了大规模零样本评测。 |
| [^201] | [From Structure to Validity: Mechanism-Guided Generation and Refinement of LLM Rubrics](https://arxiv.org/abs/2610.13375) | 该研究系统揭示了LLM评分量规设计的有效机制——结构化的标准组织最为关键，并据此提出ARTS-Hybrid方法，通过自适应量规树构建与语义优先、响应校准的精炼策略，同时提升了端到端评估性能与量规的内在质量。 |
| [^202] | [HarnessSQL: Harness-Native Training for SQL Agents in Realistic Database Environments](https://arxiv.org/abs/2610.12274) | HarnessSQL提出了一种框架原生的后训练方法，通过在带有隐藏执行预言机的真实数据库多轮交互环境中进行监督微调和执行奖励强化学习，弥合了文本到SQL模型静态训练与有状态实际部署之间的差距，显著提升了Spider 2.0-SQLite上的执行准确率。 |
| [^203] | [ILM: An AI-Powered Storytelling Educational Tool](https://arxiv.org/abs/2610.12064) | 本文提出了ILM平台，通过结合阿拉伯语自然语言处理、知识图谱构建和多语言检索技术，为先知故事提供结构化、可视化和多语言支持的互动式学习与理解评估体验。 |
| [^204] | [Do LLMs Learn from Rewards in Context? : Rethinking the role of reward in In-Context Reinforcement Learning](https://arxiv.org/abs/2610.11152) | 该研究通过受控实验发现，在大语言模型的直接上下文强化学习中，奖励信号虽然被读取却几乎不产生学习效果，而轨迹本身（即使语义被打乱或损坏）才是驱动上下文改进的关键，从而挑战了上下文学习真正实现强化学习的假设。 |
| [^205] | [ARCS: Towards Precise Text-to-SQL via Structured Disambiguation](https://arxiv.org/abs/2610.09396) | 该论文提出“结构化消歧”新范式，通过显式受约束的交互取代自由对话来解决文本到SQL中的用户问题歧义，并构建了首个基于真实数据库、包含自然产生歧义及完整标注的基准数据集ARCS。 |
| [^206] | [Self-Retrospection Distillation: Turning Post-hoc Experiences into Prior Foresight](https://arxiv.org/abs/2610.08077) | 该论文提出前瞻学习与自我回溯蒸馏方法，将已完成的轨迹作为特权后见之明，蒸馏为同一策略在交互前的轨迹无关的预见性预测，从而在组相对奖励信号消失时仍能从经验中有效学习。 |
| [^207] | [Lost in the bf16 Cast: Exporting Ternary Language Models Can Revert Most Low-Learning-Rate Code Changes](https://arxiv.org/abs/2610.07853) | 该研究审计发现，三值语言模型导出流程中先将潜在权重转换为bf16，会在阈值处因“舍入到偶数”规则被错误映射为零，导致Falcon-E-1B-Base部署后GSM8K准确率从58.79%暴跌至0.78%，几乎抹去了低学习率微调带来的改进。 |
| [^208] | [Monte Carlo Estimation for KV Cache Eviction](https://arxiv.org/abs/2610.07643) | 提出免训练方法LORE-KV，通过蒙特卡洛采样冻结模型的短自回归延续并以响应侧查询状态估计提示token效用，将KV缓存驱逐从“回顾过去”转变为“预测未来”，在回答时保留真正重要的记忆。 |
| [^209] | [Quality-Aware Self-Correcting Speech Translation on an Edge Device](https://arxiv.org/abs/2610.07545) | 该论文提出了一种在Jetson Nano边缘设备上运行的完全离线自校正语音翻译流水线，发现质量估计（QE）适合作为触发二次校正的门控而不适合作为候选重排序器，其中最小贝叶斯风险解码在阈值0.90下取得了统计显著的翻译质量提升。 |
| [^210] | [SecJev: Bringing Security Expertise to System One Decision Models](https://arxiv.org/abs/2610.03073) | 该论文提出SecJev，首个专门面向安全领域的类Jev决策模型家族（参数规模0.8B至9B），能够从文本、遥测和观察历史中学习布尔型、选择型和有序的安全决策，并通过SecJev语料库在14个任务和8个数据源上统一实现源标签预测与显式策略评估。 |
| [^211] | [Old Ideas, Novel Problems: The Instability of LLM-Based Novelty Evaluation](https://arxiv.org/abs/2610.02022) | 本文通过从OpenReview自动挖掘审稿人明确肯定或质疑论文原创性的段落构建评估集，系统性地对照研究了大语言模型新颖性评估器的设计选择，发现微小的提示词改动即可使评估结果在超过一半的相同想法上发生翻转，揭示了此类评估器的严重不稳定性。 |
| [^212] | [Policy-Conditioned AI-Use Detection: An Evidentiary Framework for Academic Publishing](https://arxiv.org/abs/2609.38427) | 该论文提出“策略条件化AI使用检测”的证据框架，将传统上判断文本是否由AI撰写的检测任务，转变为评估人-AI工作流是否遵守学术出版机构具体AI使用政策的证据推理与合规性评估。 |
| [^213] | [A Computational Framework for Modelling Organisation-Level Semantic Identity from Longitudinal Textual Data](https://arxiv.org/abs/2609.29584) | 该论文提出了一个统一的计算框架，首次将组织级语义身份建模为可解释且随时间演化的语义构建，整合了语义表示学习、图语义建模、组织语义指纹、时序演化分析与证据驱动验证。 |
| [^214] | [EduBehaviors: Assertion-based Schemas for Auditable Coding of Educational Dialogues](https://arxiv.org/abs/2609.27043) | 提出了EduBehaviors框架，利用大语言模型测量教育对话中重复出现的可观察行为并据此学习分类器，实现了可解释、可审计的教育对话标注，其性能与直接提示方法相当。 |
| [^215] | [MemoryAthena: Adaptive Routing over Latent and Generated Memories](https://arxiv.org/abs/2609.25853) | MemoryAthena 提出在直接检索、基于检索线索生成和无表生成三条记忆路径之间进行自适应因果路由，在冻结主干与记忆模块的情况下，依据反事实似然优势学习轻量路由头，并通过有界插值将生成的记忆按情境动态融合到检索残差中。 |
| [^216] | [ARAFA: An LLM-Generated Arabic Fact-Checking Dataset](https://arxiv.org/abs/2609.25833) | 研究者利用大语言模型通过三步自动化流水线构建了包含18万余个标注声明-证据对的大规模阿拉伯语事实核查数据集Arafa，有效缓解了阿拉伯语事实核查资源稀缺的问题。 |
| [^217] | [An Iterative LangGraph Agent for Text-to-SQL: Natural Language Access to the Chicago Crime Database](https://arxiv.org/abs/2609.22917) | 该论文构建了一个仅依靠提示工程（无需微调）的六节点LangGraph迭代式Text-to-SQL智能体，通过问题相关性检查、实时模式获取、SQL生成、试运行验证与失败重试等流程，让非技术用户能用自然语言查询芝加哥犯罪数据库，并将有效SQL率提升至93%、执行准确率提升至60%。 |
| [^218] | [Reviser: Revision-Capable Text Generation via Autoregressive Cursor Actions](https://arxiv.org/abs/2609.20830) | Reviser 提出了一种仅解码器 Transformer，通过在可变画布上自回归地预测光标动作（插入、移动、停止）实现真正非单调的可修订文本生成，在保持简单下一动作接口的同时避免了重复的序列级计算，性能优于 SEDD 和 MDLM 等扩散模型。 |
| [^219] | [From Discharge Notes to Patient Understanding: Persona-Grounded, Open-Ended Simulation of LLMs as Discharge Educators](https://arxiv.org/abs/2609.20827) | 该论文提出了DischargeBench，首个基于人设的开放式出院教育模拟基准，通过虚拟患者多轮对话与教育监控智能体，从对话质量、话题覆盖、理解程度和事实一致性四个维度评估大语言模型作为出院教育者促进患者理解的能力。 |
| [^220] | [The Dice Roll Method: A Standardized Protocol for Repeated-Query Auditing of Large Language Model Brand Recommendations](https://arxiv.org/abs/2609.04047) | 本文提出并形式化了“骰子法”——一个基于温度缩放核采样生成模型的可复用标准化协议，通过将响应方差分解为多个成分并提供完整的统计分析技术栈，为大语言模型品牌推荐的重复查询审计建立了系统方法。 |
| [^221] | [Lazy Grounding: Attacking Search Agents with Factual Evidence](https://arxiv.org/abs/2608.30303) | 该论文提出“惰性接地”攻击：即使检索到的文档完全真实，只要其支持的是相邻改写问题的答案，搜索代理也会采用无法回答当前问题的相邻答案，导致准确率平均下降5.9个百分点、最高下降17.3个百分点。 |
| [^222] | [RATIO: A Benchmark for Retrieval Across Typed Ideation Operations in Scientific Literature](https://arxiv.org/abs/2608.27394) | RATIO基准首次定义了三种科学构思操作（Address、Broaden、Specify）的检索任务，并利用远距离监督扩展到大规模语料库，为科学文献的灵感检索提供了新范式。 |
| [^223] | [BTS-AgentBench: A Deterministic, Replayable Pipeline from Read-Only Telemetry Logs to Agent Benchmarks](https://arxiv.org/abs/2608.27334) | 该论文提出了一种从只读遥测日志到代理基准测试的确定性、可重放构建流水线，通过标准化工具存储和情节编译，实现了精确复现和零缺陷的基准测试生成。 |
| [^224] | [TreeGraft: Adaptive Multi-Drafter Grafting for Tree-Based Speculative Decoding](https://arxiv.org/abs/2608.26112) | 我们提出了TreeGraft，一种自适应多草稿器嫁接框架，通过结合不同成本的草稿器来构建共享草稿树，从而平衡草稿质量与延迟，提升基于树的投机解码效率。 |
| [^225] | [DCGC: Draft-Conditioned Global Correction for Complex Reasoning with Masked Diffusion Models](https://arxiv.org/abs/2608.25428) | DCGC提出了一种基于草稿条件的掩码扩散模型框架，通过动态双CFG机制利用不完美草稿进行全局校正，显著提升了复杂推理任务的准确性。 |
| [^226] | [Mechanistic Circuit Identification for Controllable Data Generation](https://arxiv.org/abs/2608.24065) | 本文提出了一种基于电路识别的框架，通过连接数据估值与机制可解释性，实现可控数据生成，并引入SAMS调度器以优化效用目标数据产出。 |
| [^227] | [Cross-Model Memory Transfer via Target-Side Reader Adaptation](https://arxiv.org/abs/2608.17050) | 本文研究了跨模型记忆迁移中，冻结记忆与目标端阅读器相对重要性，发现轻量级阅读器适配是关键，而非记忆本身。 |
| [^228] | [Refusing Intent, Not Form: Wrapper-Based Intent-Group Supervision for LLM Safety](https://arxiv.org/abs/2608.13304) | 本文提出WIFA方法，通过配对包装有害与良性反例来训练模型拒绝意图而非表面形式，显著提升有害拒绝能力并降低过度拒绝率。 |
| [^229] | [LycheeMemory V2: Efficient Long-Term Memory for LLM Agents via Semantic Segment-Level Consolidation](https://arxiv.org/abs/2608.12990) | LycheeMemory V2通过语义段级整合替代回合级整合，显著降低LLM编码频率，在保持记忆细节的同时提升长期记忆构建效率。 |
| [^230] | [AVA-Encoder: Towards Agent-Native Video Representation Learning](https://arxiv.org/abs/2608.12313) | 本文提出AVA-Encoder框架，通过智能体自编码将视频转换为知识图谱表示并重建，实现了智能体可直接理解、查询和编辑的视频表示学习，克服了创意智能体从电影中学习的障碍。 |
| [^231] | [Hybrid Gated Attention](https://arxiv.org/abs/2608.11805) | 本文提出混合门控注意力框架，通过三种门控策略融合多头信息，结合低秩分解和可学习汇聚，显著提升注意力表示能力与训练效率。 |
| [^232] | [Rubric Dropout: A Simple Way to Mitigate Reward Hacking in Rubric-as-Reward RL](https://arxiv.org/abs/2608.11669) | 本文提出一种简单方法（Rubric Dropout）来缓解基于评分标准的强化学习中的奖励黑客问题，并通过实验证实训练评判员与黄金评判员分数在训练中分化的现象，为提升模型鲁棒性提供新思路。 |
| [^233] | [PERCEPT: A Corpus for POS Tagging and Analysis of Persian-English Code-Mixing](https://arxiv.org/abs/2608.10109) | 本文推出了PERCEPT——首个公开可用的大规模波斯语-英语混合语码语料库，包含6,800条社交媒体帖子并为混合语码词汇提供通用依存词性标注，同时提出了LLM辅助的自动标注框架。 |
| [^234] | [Off-the-Shelf Language-Vision Encoders Predict Human Gaze Preferences in Visual World Experiments](https://arxiv.org/abs/2608.07282) | 该论文提出一种新方法，通过对现成的双模态CLIP双编码器应用归因方法来预测视觉世界实验中的人类注视偏好，无需生成式架构和微调即可定性复现人类预测性语言加工的注视模式。 |
| [^235] | [SPARK: Learning from Verification in Speculative GRPO Rollouts](https://arxiv.org/abs/2608.04962) | 提出SPARK，一种轻量级自适应机制，利用目标模型的验证反馈在推测性GRPO训练的采样过程中实时修正草稿模型的预测遗漏，从而提升训练效率。 |
| [^236] | [ANCHOR-RE: An Agentic Biomedical Relation Extraction Framework with Retrieval Augmentation and Error-Driven Verification](https://arxiv.org/abs/2608.03154) | ANCHOR-RE是一种融合检索增强示例、本体引导语义约束与错误驱动验证的智能体化框架，无需任务特定微调即可显著提升生物医学关系抽取的可靠性与微平均F1分数。 |
| [^237] | [IACM-RL: Intent-Aware Context Management and Reinforcement Learning for Complex Tool Invocation under Dynamic Intent Fluctuations](https://arxiv.org/abs/2608.02110) | 提出IACM-RL框架，通过基于BeliefState的自生成上下文管理器主动追踪用户意图变化并隔离过时参数，结合DynamicIntent数据管线与强化学习优化，解决了动态意图波动下长程工具调用中的意图偏离与重复API调用问题。 |
| [^238] | [Commit Locally, Exit Globally: Coordinating Adaptive Sampling and Early Exit in Diffusion Language Models](https://arxiv.org/abs/2607.28166) | 提出了一种协调局部块加速与全局序列终止的双门机制，通过基于候选答案的置信度验证实现更高效的扩散语言模型加速。 |
| [^239] | [Every Time I Hire a Linguist, Inference Costs Go Down: On Linguistic Rules as Effective Prompt Compressors](https://arxiv.org/abs/2607.25335) | 该论文提出仅用语言学规则（词汇、句法、语义、语篇）作为提示压缩器，通过离线进化搜索获得规则组合，部署时无需语言模型前向传播、仅需CPU端处理即可有效降低LLM推理成本。 |
| [^240] | [Confidently Wrong: Detecting Hallucinations in Financial Question Answering from LLM Internal States](https://arxiv.org/abs/2607.11414) | 该论文提出在大语言模型残差流上训练线性探针，利用模型内部激活来检测金融问答中“自信却错误”的幻觉回答，并证明内部状态包含超越输出层面置信度信号（如token对数概率和模型自我真/假评估）的判别信息。 |
| [^241] | [CausalDS: Benchmarking Causal Reasoning in Data-Science Agents](https://arxiv.org/abs/2607.08093) | 该论文提出CausalDS基准，通过系统性地生成结构因果模型、对应的观测数据以及基于真实领域的自然语言场景，填补了现有基准在评估数据科学智能体因果推理能力方面的空白。 |
| [^242] | [What Does a Discrete Diffusion Model Learn? One Reverse Process in Three Coordinates](https://arxiv.org/abs/2607.05381) | 本文证明离散扩散模型中的去噪器、桥接插件和具体分数三种方法学习的其实是同一对象——加噪过程的精确时间反演，只是处于三种不同坐标系，并给出 Oracle Distance 恒等式（负 ELBO 等于数据熵加路径 KL 散度）及三种坐标间的精确转换。 |
| [^243] | [Knowledge Distillation from Large Reasoning Models to Compact Student Models: A Case Study on the John O Bryan Mathematics Competition](https://arxiv.org/abs/2606.31048) | 本文通过双智能体框架构建思维链语料，在Apple Silicon上利用LoRA将DeepSeek-R1的推理能力蒸馏至Qwen2.5-7B，并借助早停策略控制过拟合，从而提升紧凑模型在数学竞赛题上的表现。 |
| [^244] | [Algorithmic Unverifiability of Safety for Fixed and Recursively Self-Improving Systems](https://arxiv.org/abs/2606.28639) | 该论文从数学上严格证明了对于图灵完备的自修改系统（包括递归自我改进系统），安全验证在静态和动态两个层面都存在不可逾越的极限——不存在任何既可靠、完备又可行的安全验证器，从而为AI自我改进的安全性验证划定了根本性的理论边界。 |
| [^245] | [From Clicks to Intent: Cross-Platform Session Embeddings with LLM-Distilled Taxonomy for Financial Services Recommendations](https://arxiv.org/abs/2606.26277) | 本文提出了一种利用大语言模型提炼的分类法来生成跨平台会话嵌入，从而在金融服务推荐中实现从网页点击到用户意图的预测，并支持定量推荐与定性理解的双重目标。 |
| [^246] | [DEEPRUBRIC: Evidence-Tree Rubric Supervision for Efficient Reinforcement Learning of Deep Research Agents](https://arxiv.org/abs/2606.17029) | DeepRubric通过“先确定评估目标、再反向合成查询-评分标准对”的逆向数据构建框架，解决了现有大模型直接生成评分标准不完整的问题，从而为深度研究智能体提供更可靠的评价监督并提升强化学习效率。 |
| [^247] | [Misinformation Propagation in Benign Multi-Agent Systems](https://arxiv.org/abs/2606.16710) | 本文提出MINT基准（包含10,278个八种风格的LLM生成错误信息段落），首次系统研究了无欺骗意图的良性多智能体系统中错误信息的传播影响，发现辩论的表面稳健性仅源于未受影响智能体的数量优势，交互只有在多数智能体被误导时才能带来最多0.24的准确率恢复。 |
| [^248] | [HiMPO: Hindsight-Informed Memory Policy Optimization for Less-Entangled Credit in Long-Horizon Agents](https://arxiv.org/abs/2606.16285) | HiMPO框架将后见相关性作为有界回溯过滤器，为长时程智能体的记忆写入动作分配与下游工具故障等因素解耦的低纠缠信用，并仅对记忆token应用记忆特定优势进行优化。 |
| [^249] | [ProReviewer: Evidence-Driven Investigation for Scientific Peer Review](https://arxiv.org/abs/2606.13349) | ProReviewer将科学同行评审建模为马尔可夫决策过程，通过结构化评审日志动态追踪、更新和追溯证据，实现了证据驱动的自适应评审，在ICLR 2026论文上以8B参数规模超越包括397B模型在内的所有对比方法。 |
| [^250] | [When More Documents Hurt RAG: Measuring and Reducing Vector Search Dilution with Domain-Scoped Retrieval](https://arxiv.org/abs/2606.11350) | 论文揭示了大语料库下RAG的“向量搜索稀释”问题（precision@10下降63%），并基于200个验证查询、五个LLM骨干和六个基准语料库的MASDR-RAG测试平台证明，基于元数据的领域范围限定是缓解该问题的关键组件。 |
| [^251] | [AgentWorld: A Data Factory for Training LLM Agents from Multi-Turn Interactions](https://arxiv.org/abs/2606.09138) | AgentWorld是一个数据工厂，通过API代理网关捕获异构智能体运行时的多轮交互，并将其转化为标准化、高质量的结构化训练数据，为LLM智能体训练提供从交互收集到训练消费的统一数据生命周期。 |
| [^252] | [Explain Like I'm 5 or Whatever I Choose: Evaluating the Potential for Interactive Complexity in Language Model Responses](https://arxiv.org/abs/2606.06788) | 本研究通过用户研究确定了可靠的文本复杂度度量标准及模型遵循准则，并在557个科学查询上评估了五个大语言模型沿复杂度连续区间生成摘要的能力，发现模型虽能调整响应复杂度，但大多数变化仍与要求不一致。 |
| [^253] | [Ouvia: A User-centered Framework for Measuring Usability of Speech Translation in Real-World Communication Scenarios](https://arxiv.org/abs/2606.06177) | Ouvia是一个以用户为中心的评估框架，通过在真实一对一交流场景中收集超过1,750次交互发现，现代语音翻译仅有约一半的交互被评为可用，且不同人口群体之间存在显著的可用性差距。 |
| [^254] | [Learning Reusable Agent Subroutines from Experience](https://arxiv.org/abs/2606.02994) | ARISE通过识别智能体轨迹中反复出现的流程并将其合成为可复用的子程序库，在叙述推理、规则应用和约束规划等多个任务上显著优于提示方法、散文与代码工件归纳以及思维链等现有方法，准确率提升最高达15.6个百分点。 |
| [^255] | [Robust Asynchronous Planning via Auto-Formalization](https://arxiv.org/abs/2606.00981) | 该论文统一了异步规划的公式化并发布三个新基准，发现LLM作为规划者或经CP-SAT形式化均可取得高准确率，而专为时序规划设计的PDDL2.1反而表现最差，且由于两者由同一中间表示编译时准确率几乎一致，瓶颈在于LLM对PDDL的自动生成而非形式化语言本身。 |
| [^256] | [T2J-BENCH: Benchmarking Coding Agents on Execution-Grounded PyTorch-to-JAX Repository Migration](https://arxiv.org/abs/2605.29054) | 提出T2J-BENCH基准，包含63个涵盖18个模型家族的整仓库PyTorch到JAX迁移任务，以执行行为一致性和无torch依赖作为核心评价标准，用以真实评估编码智能体进行跨框架代码库迁移的能力。 |
| [^257] | [ConRAG: Consensus-Driven Evidence Flow for Multi-Hop Question Answering](https://arxiv.org/abs/2605.28093) | 提出ConRAG框架，通过共识驱动的多视角排序选择相关证据，并利用槽位机制在推理步骤间传递中间信息，显著提升了大语言模型在多跳问答任务上的表现。 |
| [^258] | [When Reasoning Supervision Hurts: TTCW-Based Long-Form Literary Review Generation](https://arxiv.org/abs/2605.20364) | 本研究构建了包含26万余篇长篇故事、覆盖14个TTCW创造力维度的大规模文学评论数据集，并发现一个反直觉结论：评论员风格的推理轨迹监督反而会通过引发语义重复循环而损害文学评论生成，无轨迹监督能实现更高的分数保真度和近乎完整的报告生成。 |
| [^259] | [OScaR: The Occam's Razor for Extreme KV Cache Quantization in LLMs and Beyond](https://arxiv.org/abs/2605.19660) | 该论文识别出Token范数不平衡（TNI）是极端压缩下KV缓存逐通道量化保真度的主要瓶颈，并提出了一种精确且轻量的量化方法OScaR，无需复杂的量化流水线即可实现高保真KV缓存压缩。 |
| [^260] | [When Can Digital Personas Reliably Approximate Human Survey Findings?](https://arxiv.org/abs/2605.10659) | 基于LISS面板的研究表明，大语言模型驱动的数字人格能在分布层面较好地近似人类调查回答（尤其在涉及稳定属性和价值观的领域），但在个体预测和恢复受访者多变量结构方面仍存在根本局限。 |
| [^261] | [On Challenges in Evaluation Methodology: A Review](https://arxiv.org/abs/2604.25923) | 本文通过跨越四十年的范围综述构建了包含22项反复出现的评估方法论问题的分类体系，发现许多问题在文献中的可见度下降且缺乏机构检查清单的覆盖，从而为未来的重新研究和领域层面干预指明了方向。 |
| [^262] | [ContextWeaver: Selective and Dependency-Structured Memory Construction for LLM Agents](https://arxiv.org/abs/2604.23069) | ContextWeaver将智能体的交互轨迹组织成带依赖链接的推理步骤图，结合紧凑依赖摘要与执行反馈验证来选择性构建记忆上下文，从而在固定预算下显著提升LLM智能体在软件工程和终端长上下文任务中的解决率。 |
| [^263] | [Ideological Bias in LLMs' Economic Causal Reasoning](https://arxiv.org/abs/2604.21334) | 该研究在EconCausal基准中引入亲政府与亲市场观点存在分歧的意识形态争议案例，发现大语言模型在经济因果推理中表现出系统性意识形态偏见，且争议性案例的因果方向预测难度显著更高。 |
| [^264] | [Document-as-Image Representations Fall Short for Scientific Retrieval](https://arxiv.org/abs/2604.18508) | 本文提出基于 LaTeX 源文件构建的新基准 ArXivDoc，论证“文档即图像”的表示方法不适合科学文献检索任务，因为关键证据分布于文本、表格和图表等结构化源中。 |
| [^265] | [Omni-Embed-Audio: Leveraging Multimodal LLMs for Robust Audio-Text Retrieval](https://arxiv.org/abs/2604.18360) | 提出了基于多模态大语言模型的音频检索编码器 Omni-Embed-Audio（OEA），并引入五种用户意图查询形式、难负例挖掘流水线及判别指标（HNSR、TFR），以更贴近真实搜索行为的方式评估音频-文本检索的鲁棒性。 |
| [^266] | [Latent Preference Modeling for Multi-Session Personalized Tool Calling](https://arxiv.org/abs/2604.17886) | 该论文提出了多会话个性化工具调用基准 MPT 以及测试时记忆方法 PRefine，通过“生成-验证-精炼”循环将用户潜在偏好维护为文本假设，在五个大语言模型上的表现超越所有基线，并在偏好迁移任务上成为唯一超越完整历史提示的方法。 |
| [^267] | [Reward Design for Physical Reasoning in Vision-Language Models](https://arxiv.org/abs/2604.13993) | 本文系统比较了外部奖励与基于模型自身注意力的内部奖励对视觉语言模型物理推理能力的影响，发现外部奖励贡献了主要准确率提升，而内部注意力奖励作为一种无需标注的接地信号可有效改善空间关系推理，二者互补。 |
| [^268] | [Phantom Paradoxes: Language Models Treat Unresolved Circular Truth Dependence as Contradiction](https://arxiv.org/abs/2604.12128) | 语言模型会将表面上像悖论、但实际上存在唯一一致真值赋值的循环自指问题误判为“不一致”，这种被称为“幻影悖论”的错误源于无法通过有限步骤传播确定真值的循环结构，而非自指本身。 |
| [^269] | [AdaptFuse: Training-Free Sequential Preference Inference via Externalized Bayesian Computation](https://arxiv.org/abs/2604.03925) | 提出了无需训练的AdaptFuse框架，将贝叶斯后验计算外置于符号模块，并与冻结LLM的语义推理通过熵自适应融合相结合，从而在保护隐私的前提下实现多轮交互中用户偏好的序列化累积推断。 |
| [^270] | [Terminal Agents Suffice for Enterprise Automation](https://arxiv.org/abs/2604.00073) | 该论文提出仅配备终端和文件系统的编码智能体通过直接调用平台 API，就能以远低于复杂智能体架构（如 MCP 工具智能体和网页智能体）的成本，达到甚至超越它们在企业任务上的表现。 |
| [^271] | [PLACID: Privacy-preserving Large language models for Acronym Clinical Inference and Disambiguation](https://arxiv.org/abs/2603.23678) | 本研究提出一种完全本地部署的隐私保护级联流水线，利用通用小模型检测临床缩写词并交由生物医学领域专用模型进行上下文相关的消歧扩展，从而在保障患者隐私的前提下实现临床缩写词的准确解读。 |
| [^272] | [LoopRPT: Reinforcement Pre-Training for Looped Language Models](https://arxiv.org/abs/2603.19714) | LoopRPT提出了一种专为循环语言模型设计的强化预训练框架，通过将下一词元预测重构为推理任务并把强化信号直接作用于潜空间计算步骤，使循环模型能以更少的迭代次数实现更高质量的隐式推理。 |
| [^273] | [BenchPreS: A Benchmark for Context-Aware Personalized Preference Selectivity of Persistent-Memory LLMs](https://arxiv.org/abs/2603.16557) | 该论文提出BenchPreS基准，用于评估具有持久记忆的大语言模型能否在不同交流情境下恰当地应用或抑制用户偏好，发现即使是前沿模型也倾向于将用户偏好当作全局规则而非情境依赖的信号来执行，且偏好依从性越强过度应用率越高。 |
| [^274] | [The Reasoning Trap -- Logical Reasoning as a Mechanistic Pathway to Situational Awareness](https://arxiv.org/abs/2603.09200) | 该论文提出RAISE框架，揭示提升大语言模型的演绎、归纳与溯因逻辑推理能力会经由三条机制性路径逐步促成AI的情境感知能力，警示逻辑推理研究与AI安全之间的冲突风险。 |
| [^275] | [TimeWarp: Evaluating Web Agents by Revisiting the Past](https://arxiv.org/abs/2603.04949) | 提出了模拟网页UI演变的TimeWarp基准测试，揭示现有网页智能体对界面变化的脆弱性，并通过TimeTraj标注方法与BC变体训练显著提升了智能体跨网页版本的泛化能力。 |
| [^276] | [Geometric Alignment without Functional Equivalence: A Layer-wise Analysis of the Speech-Text Modality Gap](https://arxiv.org/abs/2603.01502) | 该研究通过对比跨层CKA表征对齐与逐层答案裕度，发现语音大语言模型中语音与文本表征在中间层虽实现几何对齐，但后期层答案形成方式仍存在差异，证明几何对齐并不等于功能等价。 |
| [^277] | [Draft-Thinking: Learning Efficient Reasoning in Long Chain-of-Thought LLMs](https://arxiv.org/abs/2603.00578) | 提出轻量级适配框架Draft-Thinking，通过渐进式草稿适配学习模型适配的紧凑草稿思维链，解决强教师模型简洁推理轨迹对小模型过于稀疏的跨尺度不匹配问题，从而在保持解题能力的同时显著降低长思维链的推理成本。 |
| [^278] | [Semantic Communities and Boundary-Spanning Lyrics in K-pop: A Graph-Based Unsupervised Analysis](https://arxiv.org/abs/2602.12881) | 该研究利用多语言MiniLM嵌入对7,891首K-pop歌曲构建加权最近邻图并通过Louvain算法识别出21个语义社群，发现整首歌嵌入提供最佳的相邻歌曲区分度，且歌词主题呈现记忆、分离、渴望等跨社群的重叠模式。 |
| [^279] | [When Is Enough Not Enough? Illusory Completion in Search Agents](https://arxiv.org/abs/2602.07549) | 该研究揭示了搜索智能体中的“虚幻完成”现象——即使答案正确，智能体也可能在某个约束条件未经验证时就认为任务已完成（在正确答案中占比高达48%），并提出“认知账本”来追踪每轮检索对各约束条件的确立情况，发现训练和规模只能改变而非消除验证失败的模式。 |
| [^280] | [Language Steering for Multilingual In-Context Learning](https://arxiv.org/abs/2602.02326) | 该论文提出“语言向量”——通过在特定层计算平行语言示例间的平均激活差异，并在推理时作为偏移量注入模型隐藏状态，无需任何参数更新即可显著提升多语言上下文学习中的跨语言迁移性能，且这些向量呈现出可解释的语言学聚类结构。 |
| [^281] | [MasalBench: A Benchmark for Contextual and Cross-Cultural Understanding of Persian Proverbs in LLMs](https://arxiv.org/abs/2601.22050) | 该论文提出MasalBench基准测试，发现大语言模型虽然能很好地识别语境中的波斯谚语（准确率超过0.90），但在识别对应英语谚语时性能大幅下降（最佳模型仅为0.79），揭示了LLM在低资源语言跨文化理解方面的不足。 |
| [^282] | [SWE-Spot: Building Small Repo-Experts with Repository-Centric Learning](https://arxiv.org/abs/2601.21649) | 本文提出仓库中心化学习（RCL）新范式，将静态代码库转化为交互式学习信号，训练出针对特定代码仓库深度专门化的小型语言模型SWE-Spot-4B，以弥补小型模型处理复杂陌生代码库时泛化能力不足的缺陷。 |
| [^283] | [CALM-IT: Enhancing Synthetic Motivational Interviewing Dialogue Quality with Conversational Dynamics](https://arxiv.org/abs/2601.10085) | 提出双智能体框架CALM-IT，通过追踪来访者状态与治疗师侧推断来建模会话动态变化，从而显著提升长篇合成动机式访谈治疗对话的质量。 |
| [^284] | [QuantEval: A Benchmark for Financial Quantitative Tasks in Large Language Models](https://arxiv.org/abs/2601.08689) | 该论文提出了QuantEval基准，通过涵盖知识问答、量化数学推理和量化策略编程三个维度的1,575个专家验证实例，并采用确定性CTA回测框架实际执行模型生成的策略，系统评估了13个大语言模型完成量化金融“计算—实现—验证”闭环的能力，发现模型与人类专家之间仍存在显著差距。 |
| [^285] | [Dual-Evolving Reinforcement Learning for Automated Reward Design](https://arxiv.org/abs/2512.13399) | 本文提出双重进化强化学习框架DERL，通过让外循环奖励提议者从内层策略的验证表现中学习元梯度，实现奖励结构的自主发现与持续优化，在具身、科学和数学推理任务上显著超越基线，尤其在分布外任务上表现突出。 |
| [^286] | [The Instability of Safety: How Random Seeds and Temperature Expose Inconsistent LLM Refusal Behavior](https://arxiv.org/abs/2512.12066) | 该研究揭示大语言模型的安全拒绝决策在随机种子和温度变化下并不稳定，18-28%的有害提示词会出现“拒绝”与“配合”之间的决策翻转，且温度越高稳定性越差，表明单次安全评估无法真实反映模型的安全对齐水平。 |
| [^287] | [Ground-Truth Subgraphs for Better Training and Evaluation of Knowledge Graph Augmented LLMs](https://arxiv.org/abs/2511.04473) | 提出了SynthKGQA框架，可从任意知识图谱生成包含完整真值事实的知识图谱问答数据集，既能更有效地评估知识图谱检索器，也能用于训练更好的知识图谱增强大语言模型，并基于Wikidata构建了测试零样本泛化能力的GTSQA数据集。 |
| [^288] | [Reclaiming Low-Quality Instruction Data with Rule-Guided LLM Reconstruction](https://arxiv.org/abs/2510.23160) | 提出一种规则引导的LLM重构框架，将SFT数据选择中通常被丢弃的低分指令样本转化为高质量训练数据，在三个7B/8B模型上以远少于全量数据的样本数（约5.5万条）超越了全量训练和数据选择基线的性能。 |
| [^289] | [LLMs Can Get "Brain Rot": A Pilot Study on Twitter/X](https://arxiv.org/abs/2510.13928) | 本研究首次提出并通过Twitter/X真实语料的对照实验验证了“LLM脑腐化假说”，证明持续预训练低质量垃圾网络文本会导致大语言模型在推理、长上下文理解和安全性上出现类似剂量-反应关系的持久认知衰退。 |
| [^290] | [A large-scale pipeline for LLM-assisted corpus annotation: variation and change in the English consider construction](https://arxiv.org/abs/2510.12306) | 本文提出了一个基于大语言模型的四阶段语料库自动标注流水线，在60小时内以98%以上的准确率完成了COHA语料库中14万余条consider索引行的语法标注，并通过英语consider结构的历时变异分析验证了其在大规模语料库语言学研究中的实用性和有效性。 |
| [^291] | [EconCausal: A Context-Aware Economic Reasoning Benchmark for Large Language Models](https://arxiv.org/abs/2510.07231) | 该论文构建了EconCausal——一个基于2,595篇顶级经济学与金融学实证研究、包含10,490条情境标注因果三元组的大规模基准，用于评估大语言模型在不同情境下推断并修正经济因果效应方向的能力，并揭示了模型在跨情境预测中存在源符号过度迁移的系统性缺陷。 |
| [^292] | [Eigen-Agent: Adaptive Multi-Agent Scientific Reasoning with Monitor-Based RAG](https://arxiv.org/abs/2509.21193) | 提出Eigen-Agent统一框架，以token级监视器检索模块实现最小干扰的隐式知识整合，并结合分层解法精炼（HSR）与质量感知迭代推理（QAIR）两项机制，在HLE生物/化学集上取得迄今最高的48.3%准确率。 |
| [^293] | [Anticipatory Evaluation of Language Models: Predicting LLM Benchmark Scores from Descriptions](https://arxiv.org/abs/2509.20645) | 该论文提出“前瞻性评估”新范式和EVALCAST数据集（2,290对描述-性能数据），证明前沿LLM能够从自然语言任务描述中推断出基准测试的相对难度排序，但精确预测绝对性能分数仍然困难。 |
| [^294] | [Scaling Personality Control in LLMs with Big Five Scaler Prompts](https://arxiv.org/abs/2508.06149) | 提出基于提示词的Big5-Scaler框架，通过将大五人格数值嵌入提示词，无需额外训练即可实现对大语言模型人格特质的细粒度控制。 |
| [^295] | [Majority Bit-Aware Watermarking For Large Language Models](https://arxiv.org/abs/2508.03829) | 提出多数位感知编码范式，将水印信号强度与绿色列表大小解耦，从而在不限制绿色列表大小的情况下保持强水印信号，改善了文本质量与解码准确率之间的权衡。 |
| [^296] | [The Invisible Leash: Why RLVR May or May Not Escape Its Origin](https://arxiv.org/abs/2507.14843) | 该论文提出“经验支持动力学”框架，逐题追踪RLVR对已验证解的保持、扩展与收缩，发现在给定采样预算下支持收缩超过扩展，表明RLVR可能并未真正扩展模型推理边界，而主要是放大了基座模型已知的高奖励输出。 |
| [^297] | [Is This Just Fantasy? Language Model Representations Reflect Human Judgments of Event Plausibility](https://arxiv.org/abs/2507.12553) | 本研究识别出语言模型中用于区分句子情态范畴（可能、不可能、荒谬等）的线性表征——情态差异向量，证明语言模型的情态判断能力比以往认知更可靠，且这些向量随模型能力增强以一致的顺序涌现。 |
| [^298] | [$\pi$-CoT: Prolog-Initialized Chain-of-Thought Prompting for Multi-Hop Question-Answering](https://arxiv.org/abs/2506.20642) | π-CoT将多跳问题转化为Prolog查询并分解为单跳子查询依次求解，再利用中间结果初始化思维链推理，从而在多跳问答任务上达到或超越标准RAG和传统CoT方法的表现。 |
| [^299] | [Assessing and Refining ChatGPT's Performance in Identifying Targeting and Inappropriate Language: A Comparative Study](https://arxiv.org/abs/2505.21710) | 研究发现ChatGPT在检测不当言论方面表现良好，且提示词优化带来的提升几乎全部来自简化标签方案并升级至GPT-4这一单次更改，但其在识别定向攻击性言论方面表现较弱、稳定性较差且误报率较高。 |
| [^300] | [Prompt Stability Matters: Evaluating and Optimizing Auto-Generated Prompt in General-Purpose Systems](https://arxiv.org/abs/2505.13546) | 该论文提出语义稳定性这一新指标来衡量提示在重复执行时响应的一致性，并据此开发了首个利用稳定性反馈迭代优化提示质量的通用提示生成系统。 |
| [^301] | [Beyond Size: How Gradients Shape Pruning Decisions in Large Language Models](https://arxiv.org/abs/2311.04902) | 提出GBLM-Pruner，一种利用泰勒展开一阶项、基于少量校准样本归一化梯度的免训练剪枝方法，在大语言模型剪枝中显著优于SparseGPT和Wanda等现有方法。 |

# 详细

[^1]: 基于测试时训练的潜在压缩

    Latent Compaction with Test-Time Training

    [https://arxiv.org/abs/2610.20582](https://arxiv.org/abs/2610.20582)

    提出“潜在压缩”方法，通过测试时训练在模型权重冻结且不接触下游问题的情况下，学习紧凑的文档级 KV 缓存，显著优于同等预算的缓存压缩基线。

    

    长上下文会扩大键值（KV）缓存，增加生成过程中的内存使用和注意力计算成本。压缩这些缓存时必须为未见过的问 题保留信息。我们提出了**潜在压缩**，该方法通过测试时训练学习一个紧凑的、针对特定文档的缓存。我们使用重构 重要性、教师注意力和键空间多样性来初始化缓存条目。训练首先将注意力输出和对数归一化器拟合到完整缓存的对应 值，其中由重构选出的条目保持固定；随后通过从完整缓存模型进行端到端蒸馏，联合优化整个缓存。整个过程模型权 重保持冻结，训练仅使用关于文档的通用提示，无法访问下游的问题或答案。在 RULER、LongBench 和 Phonebook Lookup 上的实验表明，该方法相较于预算相当的压缩基线取得了显著提升。例如在 RULER 上，我们的方法实现了……（摘要截断）

    arXiv:2610.20582v1 Announce Type: new  Abstract: Long contexts enlarge key--value (KV) caches, increasing memory use and attention costs during generation. Compressing these caches must preserve information for unseen questions. We introduce **Latent Compaction**, which learns a compact document-specific cache through test-time training. We initialize cache entries using reconstruction importance, teacher attention, and key-space diversity. Training first fits attention outputs and log-normalizers to those of the full cache, with reconstruction-selected entries held fixed. It then jointly optimizes the entire cache through end-to-end distillation from the full-cache model. Model weights remain frozen throughout, and training uses only generic prompts about the document, without access to downstream questions or answers. Experiments on RULER, LongBench, and Phonebook Lookup demonstrate substantial gains over budget-matched compression baselines. On a RULER, for example, our method achie
    
[^2]: 一个任务，多种界面：简单的跨界面一致性训练提升LLM智能体的鲁棒性

    One Task, Many Interfaces: Simple Cross-Interface Consistency Training Improves LLM Agent Robustness

    [https://arxiv.org/abs/2610.20576](https://arxiv.org/abs/2610.20576)

    提出跨界面策略正则化（CIPR）这一自监督方法，通过在两个界面下惩罚动作分布间的KL散度，使LLM智能体在面对界面扰动（如动作重命名、记号变化）时保持一致行为，显著提升其跨界面的鲁棒性。

    

    后训练将LLM智能体的能力与其训练时所用的界面绑定在一起。对该界面进行保持语义的扰动，例如重命名某个动作或改变记号方式，虽然不会改变底层任务本身，却会使智能体的表现变差。由于对每个样本独立评分，界面随机化只会将这种依赖从单一界面扩大到训练界面的分布上，因此分布外的扰动仍会导致智能体性能下降。我们提出了跨界面策略正则化（CIPR），这是一种自监督目标函数，它用无信息的标识符替换动作名称，并在两个界面下呈现每个训练样本，在这两个界面中，等价的动作在构造上即可被对应识别。CIPR惩罚在两个界面下所诱导的动作分布之间的KL散度，从而使策略在不同界面之间保持行为一致。我们在监督学习和强化学习两种设置下评估了CIPR的效果。

    arXiv:2610.20576v1 Announce Type: cross  Abstract: Post-training binds the competence of an LLM agent to the interface it was trained on. Semantics-preserving perturbations of that interface, such as renaming an action or changing the notation, leave the underlying task unchanged yet make the agent worse at it. Scoring each sample independently means interface randomization only broadens this dependence from a single interface to the training interface distribution, so out-of-distribution perturbations still degrade the agent. We propose Cross-Interface Policy Regularization (CIPR), a self-supervised objective that replaces action names with uninformative identifiers and presents each training sample under two interfaces in which equivalent actions are identified by construction. CIPR penalizes the KL divergence between the action distributions induced under the two interfaces, so the policy behaves consistently across them. We evaluate CIPR in both supervised and reinforcement-learnin
    
[^3]: Manacá-1B-Instruct-2：面向原生巴西葡萄牙语大语言模型的可复现安全对齐——负面的DPO结果、受控的成分比例扫描，以及LLM评审的隐性失效模式

    Manac\'a-1B-Instruct-2: Reproducible Safety Alignment for a Native Brazilian-Portuguese LLM. Negative DPO Results, a Controlled Composition Scan, and a Silent Failure Mode of the LLM Judge

    [https://arxiv.org/abs/2610.20570](https://arxiv.org/abs/2610.20570)

    该论文为原生巴西葡萄牙语模型Manacá-1B-Instruct-2提供了可复现的安全对齐方案，记录了三次失败的DPO尝试（包括LLM评审奖励饱和而生成结果毫无变化的隐性失效模式），证明安全性来自专门的安全SFT，并通过受控扫描发现安全数据混合中通用数据比例的过度拒绝曲线呈非单调性，拐点在40-50%。

    

    巴西葡萄牙语的开源语言模型仍然服务不足，而少数进行了安全对齐的模型也很少记录哪些方法行不通，或以统计显著性验证比较结果。我们提出Manacá-1B-Instruct-2，这是Manacá-1B（17.2亿参数，从零开始针对巴西葡萄牙语训练）的第二代后训练对齐版本，其SFT语料库包含291,223个样本。我们记录了三次DPO尝试，全部为负面结果，其中包括一次训练奖励边际达到1.0而自由生成结果却完全没有变化的情况，并表明安全性来自专门的安全SFT（LLM-jp AnswerCarefully风格），经过三次迭代优化直至封闭安全-有用性边界。对安全SFT数据混合中通用数据比例（10%至60%）的受控扫描揭示了非单调的过度拒绝曲线，其拐点位于40-50%。最终模型在相同项目上与四个外部巴西葡萄牙语模型进行比较，采用Wilson（置信区间）……（原文截断）

    arXiv:2610.20570v1 Announce Type: new  Abstract: Brazilian Portuguese remains underserved by open-source language models, and the few that are safety-aligned rarely document what did not work or validate comparisons with statistical significance. We present Manac\'a-1B-Instruct-2, the second-generation post-training alignment of Manac\'a-1B (1.72B parameters, trained from scratch for PT-BR), with an SFT corpus of 291,223 examples. We document three DPO attempts, all negative, including a case where the training reward margin reaches 1.0 while free generation does not change at all, and show that safety came from a dedicated safety-SFT (LLM-jp AnswerCarefully style), refined over three iterations until closing the safety-helpfulness frontier. A controlled scan of the fraction of general data in the safety-SFT mix (10% to 60%) reveals a non-monotonic over-refusal curve with a knee point at 40-50%. The final model is compared against four external PT-BR models on the same items, with Wils
    
[^4]: 扩展科学发现需要遗忘

    Scaling Discovery Requires Forgetting

    [https://arxiv.org/abs/2610.20563](https://arxiv.org/abs/2610.20563)

    论文发现 LLM 智能体自动化研究的三种扩展策略都会因智能体锚定于自身历史尝试而停滞，并提出通过周期性丢弃搜索状态、从最佳候选重新启动的“遗忘”机制来突破停滞，在相同预算下实现更大的发现成果。

    

    基于 LLM 智能体的自动化研究在多个领域已展现出可观的前景。通过迭代改进和评分验证器，LLM 智能体能够发现新的想法。扩展自动化研究有多种方式：让单个智能体迭代更长时间、并行运行多个智能体，或者进一步让并行的智能体相互交流并在彼此的候选方案基础上构建。然而，我们发现随着智能体不断积累历史，这三种策略最终都会陷入停滞：每个智能体都会锚定于自己过去的尝试，做出的修改越来越少、幅度越来越小。超过这个临界点后，追加的预算几乎无法带来进一步提升。在本文中，我们表明进一步扩展发现的关键在于“遗忘”：周期性地丢弃搜索状态（包括每个智能体的会话和候选历史），并从迄今为止找到的最佳候选重新开始，能够解除搜索的停滞，让相同的预算发挥更大的作用。我们实现了……（原文摘要此处截断）

    arXiv:2610.20563v1 Announce Type: cross  Abstract: Automated research with LLM agents has shown promising results in various domains. Through iterative refinement and scored verifiers, LLM agents are capable of discovering new ideas. There are different ways to scale automated research: let one agent iterate for longer, run several agents in parallel, or further let the parallel agents communicate and build on one another's candidates. However, we find that all three strategies eventually plateau as the agents accumulate history. Each agent becomes anchored to its own past attempts, making fewer and smaller edits. Beyond this point, additional budget brings little further improvement. In this paper, we show that the key to scaling discovery further is to forget. Periodically discarding the search state, including every agent's session and the candidate history, and restarting from the best candidate found so far, unblocks the search and pushes the same budget further.   We implement fo
    
[^5]: PerSo：在现实世界说服场景中探究大语言模型社会智能的多个维度

    PerSo: Investigating Facets of LLMs' Social Intelligence in Real-World Persuasion Scenarios

    [https://arxiv.org/abs/2610.20548](https://arxiv.org/abs/2610.20548)

    该论文提出了PerSo基准，利用Reddit ChangeMyView社区的真实说服对话，同时评估大语言模型社会智能中的认知智能（推断他人心理状态）与行为智能（选择合适社会行动）两个方面。

    

    理解大语言模型（LLM）的社会智能对于涉及人机交互和多智能体环境的应用而言日益重要。社会智能包含两个方面：认知智能，即推断他人心理状态的能力；以及行为智能，即基于这些推断选择恰当的社会行动以实现社交目标的能力。说服为同时考察这两个方面提供了一个天然的测试平台，然而以往的基准测试通常采用合成场景将二者分开评估。我们提出了PerSo，这是一个基于Reddit的ChangeMyView社区真实世界说服对话来评估这两个方面的基准，采用LLM在环（LLM-in-the-loop）流水线构建评估任务，并通过人工标注加以验证。我们对多种LLM的评估表明，性能总体上随模型规模增大而提升，但在不同维度上表现各异。最强的……

    arXiv:2610.20548v1 Announce Type: new  Abstract: Understanding the social intelligence of large language models (LLMs) is increasingly important for applications involving human-AI interaction and multi-agent environments. Social intelligence comprises two aspects: cognitive intelligence, the ability to infer others' mental states, and behavioral intelligence, the ability to select appropriate social actions based on these inferences to achieve social goals. Persuasion provides a natural testbed in which both aspects are simultaneously at play, yet prior benchmarks typically evaluate them in isolation using synthetic scenarios. We introduce PerSo, a benchmark that evaluates both aspects on real-world persuasion dialogues from Reddit's ChangeMyView community, using an LLM-in-the-loop pipeline to construct the evaluation tasks, validated through human annotation. Our evaluation of diverse LLMs shows that performance generally scales with model size but varies across dimensions. The stron
    
[^6]: PrismaCoder：通过分解式评分标准合成突破开放多模态代码智能的边界

    PrismaCoder: Pushing the Boundaries of Open Multimodal Code Intelligence through Decomposed Rubric Synthesis

    [https://arxiv.org/abs/2610.20538](https://arxiv.org/abs/2610.20538)

    该论文提出PrismaCoder，从数据端出发，通过分解式评分标准合成方法，解决多模态代码生成中样本的程序正确性、代码质量与视觉正确性难以同时评估的核心难题，从而推动开放多模态代码智能的发展。

    

    多模态代码智能将程序的符号世界与其执行所创造的视觉世界连接起来。随着模型越来越多地被要求生成图表、网页、动画和半结构化文档，人们对这一能力的兴趣正在迅速增长。近期，专有模型已大幅领先，甚至能够仅凭文本描述或一张截图就生成交互式作品。然而，这些模型的数据和训练过程仍然封闭，开放领域的进展受限于可用数据的数量与质量，以及生产这些数据的流程和方法。其中质量是更难解决的问题，因为一个样本必须同时满足：作为程序是正确的、构建良好的，以及最关键的——渲染后在视觉上是正确的。目前没有任何现有信号能够同时覆盖这三个方面：执行往往只返回通过或失败，代码相似度无法映射到视觉输出，而针对特定任务的验证器又缺乏泛化能力。为了进一步推动开放领域的发展，我们从数据端入手……

    arXiv:2610.20538v1 Announce Type: new  Abstract: Multimodal code intelligence connects the symbolic world of programs with the visual world their execution creates. Interest in this capability is burgeoning, as models are increasingly asked to produce charts, webpages, animations, and semi-structured documents. Recently, proprietary models have pushed far ahead, producing interactive artifacts even from a text description or a screenshot alone. Yet their data and training remain closed, and open progress is held back by the quantity and quality of available data and by the pipelines and methods for producing it. Quality is the harder of the two, since a sample must be correct as a program, well built, and above all visually right once rendered. No existing signal covers all three. Execution often returns just a pass or fail, code similarity does not map to the visual output, and task-specific verifiers do not generalize. To push open progress further, we start from the data side with P
    
[^7]: VASPO：基于特权价值估计的自蒸馏

    VASPO: Self-Distillation with Privileged Value Estimation

    [https://arxiv.org/abs/2610.20516](https://arxiv.org/abs/2610.20516)

    该论文提出VASPO方法，将特权价值估计与自蒸馏在PPO框架内结合，利用评论家与自教师在学生生成前缀上评估特权信息时呈现的互补位置动态特性，为推理模型提供更有效的词元级训练信号。

    

    在线策略自蒸馏通过特权自教师为推理模型提供密集监督。然而，这种监督反映的是局部词元偏好，而非无指导的学生续写内容的期望回报。近期方法将该信号与GRPO相结合，即用GRPO确定更新方向、用教师模型进行信号缩放。但我们证明了在PPO框架内可以实现更有效的整合。具体而言，我们识别出一种互补的位置动态特性：特权信息在响应末端能够提升评论家价值预测的准确性，而自教师的指导在生成初期较强，中期下降，并在末期回升。这些洞察促成了价值增强自蒸馏策略优化的提出：在该方法中，自教师与评论家在学生生成的前缀处评估相同的特权信息，评论家基于预测的续写回报计算词元级的PPO优势……

    arXiv:2610.20516v1 Announce Type: new  Abstract: On-policy self-distillation provides dense supervision for reasoning models via a privileged self-teacher. However, this supervision reflects local token preferences rather than the expected return of an unguided student continuation. Recent methods combine this signal with GRPO. They use GRPO for update directions and the teacher for signal scaling. However, we demonstrate a more effective integration within PPO. Specifically, we identify complementary positional dynamics: privileged information enhances a critic's value-prediction accuracy toward the response's end, whereas self-teacher guidance is strong initially, drops mid-generation, and rebounds finally. These insights motivate Value-Augmented Self-Distillation Policy Optimization (VASPO), wherein a self-teacher and critic evaluate identical privileged information at student-generated prefixes. The critic computes token-level PPO advantages from predicted continuation returns, whi
    
[^8]: ReaDy：循环状态作为混合线性注意力大语言模型中推测性草拟的动态上下文

    ReaDy: Recurrent States as Dynamic Context for Speculative Drafting in Hybrid Linear Attention Large Language Models

    [https://arxiv.org/abs/2610.20503](https://arxiv.org/abs/2610.20503)

    ReaDy是首个利用混合线性注意力模型循环状态作为动态上下文的推测解码方法，通过为每个草拟位置设置可学习查询直接从目标模型循环状态中检索补充上下文，以极低的计算开销提升草拟质量。

    

    推测解码通过每步草拟多个token来加速自回归生成，然而基于特征的起草器依赖于目标模型的隐藏特征，其上下文侧重点可能与后续草拟位置的需求不一致。混合线性注意力（HLA）模型维护着能够积累历史关联的紧凑循环状态，为推测性草拟提供了一种高效的补充上下文来源。我们提出了ReaDy，据我们所知，这是首个在草拟过程中利用HLA循环状态进行位置特定上下文检索的推测解码方法。ReaDy为每个草拟位置配备一个可学习的查询，直接从目标模型已提交的循环状态中检索补充上下文。轻量级的状态读取器通过残差更新将这些上下文整合到草稿表示中，同时低秩状态读取和融合算子使额外的计算与I/O开销保持在较低水平。

    arXiv:2610.20503v1 Announce Type: new  Abstract: Speculative decoding accelerates autoregressive generation by drafting multiple tokens per step, yet feature-based drafters rely on target hidden features whose contextual emphasis may not align with the needs of subsequent draft positions. Hybrid linear attention (HLA) models maintain compact recurrent states that accumulate historical associations, offering an efficient source of complementary context for speculative drafting. We introduce ReaDy, the first speculative decoding method, to the best of our knowledge, to leverage HLA recurrent states for position-specific context retrieval during drafting. ReaDy equips each draft position with a learned query to retrieve supplementary context directly from the target's committed recurrent states. A lightweight state reader integrates this context into draft representations through residual updates, while low-rank state reads and fused operators keep additional computation and I/O overhead 
    
[^9]: 翻译中的迷失：为什么更低的偏见分数并不意味着更公平的模型

    Lost in Translation: Why Lower Bias Scores Don't Mean Fairer Models

    [https://arxiv.org/abs/2610.20479](https://arxiv.org/abs/2610.20479)

    本研究审计了10个开源模型在8种语言中的偏见表现，发现非英语语言下刻板印象分数的下降主要源于翻译缺陷与测量噪声，而非模型本身更公平，因为测试项的英语分数对其翻译后分数的预测能力远低于对改写后分数的预测能力。

    

    多语言语言模型的偏见审计通常将模型在英语中的平均刻板印象分数与其在另一种语言中的平均分数进行比较，并将二者的差异解读为公平性上的差异。我们探究这种差异中有多少其实是噪声。我们审计了八个语言下的十个开源模型，测量当英语测试项仅仅被改写时分数的变动幅度，逐项检验每一次语言转换，并验证该测试能够捕捉我们故意注入的转换。在全部70个模型-语言对中，刻板印象分数均低于其英语水平，且印地语、韩语和斯瓦希里语的下降幅度明显超过改写英语所导致的下降；其中大部分下降来自StereoSet测试项。通过对每个翻译进行评判，我们发现不完美的翻译下降幅度更大，但即使只看忠实的翻译，也几乎出现同样程度的下降。尽管如此，更低的平均值并不能表明模型更公平：一个测试项的英语分数对其翻译后分数的预测能力远不如对其改写后分数的预测能力。

    arXiv:2610.20479v1 Announce Type: new  Abstract: Bias audits of multilingual language models usually compare a model's average stereotype score in English with its average in another language and read the difference as a difference in fairness. We ask how much of that difference is noise. Auditing ten open models in eight languages, we measure how far scores move when English test items are merely reworded, test every language shift item by item, and check that the test catches shifts we inject on purpose. In all 70 model-language pairs the stereotype score falls below its English level, and for Hindi, Korean and Swahili the drop clearly exceeds what rewording English produces; most of it comes from StereoSet items. Judging every translation, we find that imperfect ones drop more, but faithful translations alone drop nearly as much. Still, a lower average does not show that a model is fairer. An item's English score predicts its translated score far less well than its reworded score (c
    
[^10]: AdaRSI：面向具身智能体的按需递归自我改进

    AdaRSI: On-Demand Recursive Self-Improvement for Embodied Agents

    [https://arxiv.org/abs/2610.20474](https://arxiv.org/abs/2610.20474)

    AdaRSI提出了一种按需递归自我改进框架，先改进技能与外围框架、再通过重试归因持续失败、最后仅微调不可靠的子步骤，从而在无需人类示范的情况下显著提升具身智能体完成长时程任务的能力。

    

    预训练的视觉-语言-动作模型（VLA）在执行未见过的长时程任务时常常中途失败。递归自我改进（RSI）提供了一种补救方法，但现有的具身RSI方法要么只是围绕冻结的执行器改进其外围框架，因而无法补充缺失的动作；要么在持续失败的情况下直接训练执行器，却不追究失败究竟源于执行器的调用方式，还是源于其自身能力。我们提出一个问题：自我改进的具身系统应当在何时改变其执行器的权重？为此我们提出了AdaRSI，一个面向具身智能体的按需RSI框架。一个编程智能体首先改进其技能和外围框架，随后通过从保存的状态重试失败的子步骤来归因持续存在的失败，并仅对仍然不可靠的子步骤进行微调，所用数据为它自行收集的示范以及从早期任务中传承的经验。在RoboCasa365的全部50个目标任务上，以GR00T N1.5作为执行器且无需任何人类示范，AdaRSI改进了……（摘要原文在此处截断）

    arXiv:2610.20474v1 Announce Type: new  Abstract: Pretrained vision-language-action models (VLAs) often fail midway through unseen long-horizon tasks. Recursive self-improvement (RSI) offers a remedy, but embodied RSI methods either refine the harness around a frozen executor and so cannot add a missing motion, or train the executor on persistent failures without asking whether a failure lies in how the executor is called or in what it can do. We ask when a self-improving embodied system should change the weights of its executor, and propose AdaRSI, an on-demand RSI framework for embodied agents. A coding agent first improves its skills and harness, then attributes persistent failures by retrying failing sub-steps from saved states, and fine-tunes only the sub-steps that remain unreliable, with demonstrations it collects itself and experience carried over from earlier tasks. On all 50 target tasks of RoboCasa365, with GR00T N1.5 as the executor and no human demonstrations, AdaRSI improv
    
[^11]: ChronoCulture：评估大语言模型跨时间的文化意识

    ChronoCulture: Evaluating Cultural Awareness Across Time in Large Language Models

    [https://arxiv.org/abs/2610.20472](https://arxiv.org/abs/2610.20472)

    提出ChronoCulture基准，利用72个国家纵向调查数据构建的10,360个时间对，揭示大语言模型虽能预测某一时间点的文化态度，却无法可靠捕捉文化态度随时间的动态演变，呈现出静态化的文化图景。

    

    文化态度和社会规范会随时间演变，然而大多数针对大语言模型的文化基准将文化视为国家或地区的静态属性。我们提出了ChronoCulture，这是一个基于72个国家纵向调查数据构建的包含10,360个时间对的基准。每一对包含在同一国家的较早年份和较晚年份提出的相同调查问题，并以完整的人口回答分布作为真实标准（ground truth）。在六个大语言模型上的评估中，我们发现模型通常能够预测一个国家在某一时间点的态度，但无法可靠地捕捉这些态度如何随时间变化。模型对不同年份的预测往往非常接近，未能捕捉到大部分实际观察到的变化。在提示中指定目标年份对提高与调查回答的一致性也没有带来整体上的明显改善。这些发现表明，模型往往呈现出一种基本静态的文化态度图景，凸显了在评估文化意识时纳入时间维度的必要性。

    arXiv:2610.20472v1 Announce Type: new  Abstract: Cultural attitudes and social norms evolve over time, yet most cultural benchmarks for large language models (LLMs) treat culture as a static attribute of a country or region. We introduce ChronoCulture, a benchmark of 10,360 temporal pairs built from longitudinal survey data across 72 countries. Each pair contains the same survey question asked in the same country in an older and a newer year, with full population response distributions as ground truth. Across six LLMs, we find that models can often predict a country's attitudes at one point in time but do not reliably capture how those attitudes change. Their predictions for different years tend to stay close together, failing to capture much of the observed change. Specifying the intended year in the prompt also offers little overall improvement in alignment with survey responses. These findings suggest that models often present a largely static picture of cultural attitudes, highligh
    
[^12]: 相同的CER，不同的损害：面向OCR的严重度加权字符错误率

    Equal CER, Unequal Damage: Severity-Weighted Character Error Rate for OCR

    [https://arxiv.org/abs/2610.20461](https://arxiv.org/abs/2610.20461)

    提出semCER指标，利用大语言模型为OCR中每个字符编辑赋予严重度权重，使错误率评估更符合人类读者对错误实际损害的感知。

    

    字符错误率（CER）是OCR和视觉语言模型（VLM）的标准评估指标，但它对拼写错误和错误人名的惩罚力度相同。我们提出了semCER，它保留了CER的对齐方式和分母，但由大语言模型（LLM）裁判为每个字符编辑赋予严重度权重，且无需分词。归一化CER保留了整个转录文本的整体分数，在这一层面上它追踪读者判断的能力与semCER相当甚至更好，而semCER能将该分数分解为具体的命名编辑。在英语、越南语和中文中，这些严重度权重能够逐条编辑地追踪读者的评分，并将读者认为最严重的编辑排在排序列表的顶部。在CER与semCER意见不一致的样本对上，读者在英语和中文中更常站在semCER一边而非归一化CER，在越南语中两者持平。在16种语言中，该裁判认为错误的变音符号或声调符号在西班牙语中几乎没有代价，但在泰卢固语中其严重度接近写错一个字母。

    arXiv:2610.20461v1 Announce Type: new  Abstract: Character Error Rate (CER), the standard metric for OCR and vision-language models (VLMs), charges a typo as much as a wrong name. We introduce semCER, which keeps CER's alignment and denominator and charges each grapheme edit a severity from an LLM judge, with no word segmentation. Normalized CER keeps the score of a whole transcription, where it tracks readers as well as semCER or better, and semCER breaks that score down into named edits. The charges track readers' ratings edit by edit in English, Vietnamese, and Chinese and put the edits readers rate most severe near the top of a sorted list. On pairs drawn where CER and semCER disagreed, readers side with semCER more often than with normalized CER in English and Chinese, and equally often in Vietnamese. Across 16 languages, the judge charges a wrong diacritic or tone mark almost nothing in Spanish but nearly a wrong letter's severity in Telugu. Where we can count real words, a gener
    
[^13]: MolLean：一种过程可验证的化学推理语言

    MolLean: A Process-Verifiable Language for Chemical Reasoning

    [https://arxiv.org/abs/2610.20459](https://arxiv.org/abs/2610.20459)

    MolLean 提出了一种基于 Lean 的过程可验证化学推理语言，通过分层内核检查显式分子状态上的转换与断言、带类型的见证协议区分结构检查与外部测量，并借助有状态工具接口 MolLean-MCP 提供交互反馈，从而有效抑制化学思维链中的幻觉论断。

    

    化学思维链中可能包含幻觉生成的结构性与性质性论断。为此，我们提出了 MolLean，一种过程可验证的语言，将化学推理扎根于显式的分子状态之上。一个分层的、基于 Lean 的内核对分子读数上的状态转换与断言进行检查，涵盖从原始图编辑到化合物变换的各个层面。同时，一种带类型的见证协议将原生的结构检查与外部提供的测量值分离开来。我们进一步引入了 MolLean-MCP，一个有状态的工具接口，在生成过程中暴露这些转换、检查与读数：模型接收反馈以决定下一步动作，而被拒绝的动作不会改变状态。MolLean 支持单次轨迹执行、整轨迹细化以及通过 MolLean-MCP 进行的交互式推理。我们在跨越分子生成、编辑等 12 个任务的 8,800 个匹配样本上，将这三种模式与直接思维链方法进行了评估。

    arXiv:2610.20459v1 Announce Type: new  Abstract: Chemical chain-of-thought can contain hallucinated structural and property claims. We thereby introduce MolLean, a process-verifiable language that grounds chemical reasoning in an explicit molecular state. A layered Lean-based kernel checks state transitions and assertions over molecular readouts, from primitive graph edits to compound transformations. Meanwhile, a typed witness protocol separates native structural checks from externally supplied measurements. We further introduce MolLean-MCP, a stateful tool interface that exposes these transitions, checks, and readouts during generation: the model receives feedback to decide its next action, and rejected actions leave the state unchanged. MolLean supports one-pass trace execution, whole-trace refinement, and interactive reasoning through MolLean-MCP. We evaluate these three modes with direct chain-of-thought on 8,800 matched examples across 12 tasks spanning molecular generation, edit
    
[^14]: 同一主张，不同解读：LLM智能体如何消解科学主张中的歧义

    Same Claim, Different Interpretations: How LLM Agents Resolve Ambiguity in Scientific Claims

    [https://arxiv.org/abs/2610.20457](https://arxiv.org/abs/2610.20457)

    该论文提出Claim Cards方法，在证据收集前以九个显式字段记录LLM智能体对歧义科学主张的解读，并验证了解读方式的不同确实会改变智能体的验证判定结果。

    

    共同科学家系统和智能体式科学主张验证器正被越来越多地用于评估各科学领域的主张。为人类专家撰写的自然语言科学主张往往规定不足，因此可以有多种合理的解读。这类验证器通常会隐式地选择其中一种解读，而基准测试在将验证器的判定结果与专家标签进行对比评分时，并未以可比较的形式记录专家的解读或验证器的解读。这导致与专家标签的分歧既无法追溯到证据、也无法追溯到解读本身，同时计算资源也被耗费在未经检验的解读上。我们提出Claim Cards（主张卡片），使这类解读在推理过程中显式化。一张主张卡片在证据收集之前以九个字段记录验证器的解读，每个字段被标记为“陈述的”、“推断的”或“默认的”。为了检验智能体的解读是否会改变其判定结果，我们在60个经专家标注的主张上各运行GPT-5.4智能体五次……

    arXiv:2610.20457v1 Announce Type: new  Abstract: Co-scientists and agentic scientific claim verifiers are increasingly used to assess claims across the sciences. Natural-language scientific claims, written for human experts, are often underspecified and so admit several defensible interpretations. Such verifiers usually pick one, often tacitly, and benchmarks score their verdicts against expert labels without recording the expert's interpretation or the verifier's in a comparable form. This leads to disagreements with expert labels that cannot be traced to the evidence or the interpretation, and to compute spent on unchecked interpretations. We introduce Claim Cards to make such interpretations explicit in the reasoning itself. A Claim Card records a verifier's interpretation as nine fields before evidence gathering, each marked stated, inferred, or default. To test whether an agent's interpretation can change its verdict, we ran a GPT-5.4 agent five times on each of 60 expert-labeled 
    
[^15]: The COUNT：通过计数N-gram解锁主机内存上的扩展与免训练适配

    The COUNT: Counting N-grams Unlocks Scaling on Host Memory and Training-Free Adaptation

    [https://arxiv.org/abs/2610.20456](https://arxiv.org/abs/2610.20456)

    提出COUNT方法，通过在廉价的主机内存上存储预计算的n-gram计数并结合语言模型自身的unembedding来构建n-gram嵌入，无需训练新参数，从而实现主机内存上的扩展与免训练适配。

    

    语言模型（LMs）的计算与性能之间的缩放关系已被广泛研究；最近，模型也被设计为能够扩展内存使用。特别是，为n-gram添加嵌入可以使内存扩展与计算使用相互独立，因为每次前向传播只处理单个嵌入行，其大小远小于整个嵌入表所占用的存储空间。然而，现有方法需要从头训练嵌入，这需要昂贵的GPU内存来进行快速读写。与此同时，主机系统的普通内存价格低廉，却在这些训练方法中未被充分利用。为了在主机内存上实现扩展，我们提出了COUNT（Counting Offline Unlocks N-gram Transformers），它通过在主机上存储预计算的n-gram计数来构建n-gram嵌入，而无需添加任何可学习参数。我们利用语料库频率，对语言模型自身的unembedding进行加权组合，以表示最常跟随当前上下文的后继词元……（摘要原文在此处截断）

    arXiv:2610.20456v1 Announce Type: new  Abstract: The scaling relationship of compute and performance for language models (LMs) has been extensively studied; more recently, models are being designed to scale memory use, as well. In particular, adding embeddings for n-grams scales memory separately from compute use, because each pass only processes individual embedding rows, which are much smaller than the storage used for the whole embedding table. However, existing methods train embeddings from scratch, which needs expensive GPU memory for fast reading and writing. Meanwhile, the ordinary memory of the host system is cheap and underutilized by these training methods. To scale on host memory, we introduce COUNT (Counting Offline Unlocks N-gram Transformers), which builds n-gram embeddings without adding learned parameters by storing precomputed n-gram counts on host. We use the corpus frequencies to weight together the LM's own unembeddings of the next tokens most often following the cu
    
[^16]: SwarmRSI：通过演化的科学智能体群体实现递归自我改进

    SwarmRSI: Recursive Self-Improvement through Evolving Scientific Agent Swarm

    [https://arxiv.org/abs/2610.20431](https://arxiv.org/abs/2610.20431)

    提出了 SwarmRSI 框架，通过反馈引导的群体变异和互补群体重组两种机制演化科学智能体群体，解决假设与实现修改不明确以及有用组件被隐藏的问题，实现科学研究中的递归自我改进。

    

    科学研究中的递归自我改进（RSI）旨在通过智能体自身的实验经验来提升其研究能力。科学智能体群体可以整合来自不同研究方向的证据和实现方案，以指导智能体的后续实验。然而，糟糕的实验结果会导致不清楚究竟应该修改假设还是修改其实现方案，而整体程序评分可能会掩盖对其他智能体有用的组件。为了应对这些挑战，我们提出了 SwarmRSI，一个通过演化科学智能体群体来实现递归自我改进的框架。我们设计了反馈引导的群体变异，利用来自其他智能体的实验反馈来完善假设及其实现方案。我们还提出了互补群体重组，以在智能体之间重用保留的假设和互补的程序组件，使部分结果能够支持新的研究。递归群体演化……

    arXiv:2610.20431v1 Announce Type: cross  Abstract: Recursive self-improvement (RSI) in scientific research aims to improve agents' research capabilities through their own experimental experience. Scientific agent swarms can combine evidence and implementations from different research directions to guide their agents' subsequent experiments. However, poor experimental results can leave unclear whether to revise a hypothesis or its implementation, while overall program scores can hide components useful to other agents. To address these challenges, we introduce SwarmRSI, a framework that evolves scientific agent swarms for recursive self-improvement. We devise Feedback-Guided Swarm Mutation to refine hypotheses and their implementations using experimental feedback from other agents. We also propose Complementary Swarm Recombination to reuse retained hypotheses and complementary program components across agents, allowing partial results to support new research. Recursive Swarm Evolution in
    
[^17]: 分析检索增强生成模型在问答任务中的性能

    Analyzing performance of Retrieval Augmented Generation models for Question Answering Tasks

    [https://arxiv.org/abs/2610.20419](https://arxiv.org/abs/2610.20419)

    本文分析了RAG模型在问答任务中的性能，指出尽管RAG架构近年取得巨大进展，但在以南亚当地语言回答该地区特有文化问题时仍面临挑战。

    

    检索增强生成（RAG）模型已成为开放域问答任务的基石之一，在此类任务中，生成答案的质量在很大程度上依赖于检索机制的有效性。在大语言模型（LLM）领域，过去两年间RAG架构取得了巨大进展，许多模型通过各种设计创新报告了性能的提升。南亚地区的文化问答系统尤其面临特殊挑战，因为该地区拥有大量紧密相互关联的语言和传统，这些深度交织的历史叙事增加了对文化扎根且具备语境感知能力的解释需求。尽管大语言模型在架构上取得了声称有显著进步的改进，但在以当地母语回答该地区特定问题时仍存在挑战。在本研究中，我们旨在（原文摘要在此处截断）

    arXiv:2610.20419v1 Announce Type: new  Abstract: Retrieval-Augmented Generation (RAG) models have become almost a cornerstone of open-domain question answering tasks; in which the quality of answers generated in the process, greatly relies on the effectiveness of retrieval mechanisms. In the realm of LLM, the last two years have seen tremendous progress in RAG architectures, with many models reporting to have improved performance through various design innovations. The cultural question and answering systems of South Asia, in particular, present a specific challenge since the region is home to the vast number of languages and traditions, which are closely interrelated. Along with these deeply intertwined historical narratives increase the need for culturally grounded and context-aware interpretation. Despite major architectural advances in LLMs, claiming significant improvements, challenges remain when answering these region specific questions in native languages. In this study, we aim
    
[^18]: 从不可靠到可用：分解大语言模型实体抽取中的失败模式与置信度信号——来自网络威胁情报的证据

    From Unreliable to Usable: Decomposing Failure Modes and Confidence Signals in LLM Entity Extraction - Evidence from Cyber Threat Intelligence

    [https://arxiv.org/abs/2610.20417](https://arxiv.org/abs/2610.20417)

    该论文通过对新一代低成本与旗舰级大语言模型在354个网络威胁情报事件上的预注册冻结评估，发现实体抽取的不可靠性在所有模型层级均持续存在（精确率仅0.03-0.34且校准误差高达0.53-0.94），并揭示主导性失败并非幻觉，同时探索了置信度信号能否将不可靠输出转化为可用输出。

    

    对大型语言模型（LLM）从全文长文档中抽取信息的审计报告显示其严重不可靠，由此留下两个悬而未决的问题：失败究竟由什么构成，以及置信度信号能否将不可靠的输出转化为可用的输出。我们对两代新型低成本大语言模型（DeepSeek-V4-Flash、GLM-5.3-Flash）以及一个旗舰级对照模型（GPT-6.1-Sol）进行了预注册且完全冻结的评估，语料库为包含354个事件攻击行动的公开数据集（315份独立报告，630次模型-报告评估），并从三个方面进行评分：相对于金标准的抽取质量、置信度校准程度，以及对单句扰动的鲁棒性。研究得出三个发现。（1）不可靠性在各个模型层级均持续存在：在两个数据划分上，微观精确率仅为0.03-0.34，口头表述的校准误差（ECE）高达0.53-0.94；这一差距在旗舰级模型上也未消除。（2）主导性失败并非幻觉：通过一次完整的提及普查加上对360个抽样候选项的三层标注……

    arXiv:2610.20417v1 Announce Type: new  Abstract: Audits of large language model (LLM) extraction from full-length documents report severe unreliability, leaving two questions open: what the failures actually consist of, and whether confidence signals can turn unreliable output into usable output. We run a preregistered, fully frozen evaluation of two new-generation low-cost LLMs (DeepSeek-V4-Flash, GLM-5.3-Flash) with a flagship-tier control (GPT-6.1-Sol) on a public corpus of 354 incident campaigns (315 unique reports, 630 model-report evaluations), scoring extraction quality against gold, confidence calibration, and robustness to single-sentence perturbations. Three findings emerge. (1) Unreliability persists across tiers: micro precision is 0.03-0.34 and verbalized calibration error (ECE) 0.53-0.94 across both splits; the gap does not close at the flagship tier. (2) The dominant failure is not hallucination: a full mention census plus a three-tier annotation of 360 sampled candidate
    
[^19]: Cortex：将上下文整合为参数化记忆

    Cortex: Consolidating Context into Parametric Memory

    [https://arxiv.org/abs/2610.20399](https://arxiv.org/abs/2610.20399)

    本文提出 Cortex，通过生成前瞻性查询将上下文增益蒸馏为可合并的低秩或结构化前馈网络参数更新，在 4B 至 120B 的多种稠密与混合专家模型上实现了“巩固—保持”帕累托最优的记忆巩固。

    

    上下文可以提升大语言模型的性能，但这些增益在移除上下文后可能会丢失。受生物系统巩固机制的启发，我们将记忆巩固定义为在保留非目标能力的同时将这些增益内化，从而在统一目标下连接了上下文指定编辑、技能学习和行为引导。我们引入了一个涵盖五类任务的测试平台，用于衡量单次、批量与顺序更新下的巩固与保持效果。我们提出 Cortex，它通过生成前瞻性查询来预判上下文的用途，并将上下文条件下的输出蒸馏为可合并的低秩或结构化前馈网络更新。在 4B 至 120B 参数规模的稠密与混合专家骨干网络上（包括纯文本与多模态模型），Cortex 在大多数评估设置中位于经验性的“巩固—保持”帕累托前沿。在对 14B 模型进行 50 次顺序事实编辑后，Cortex-Structural（摘要原文在此处截断）

    arXiv:2610.20399v1 Announce Type: new  Abstract: Context can improve large language model performance, but these gains may be lost after context removal. Inspired by biological systems consolidation, we formulate Memory Consolidation (MC) as internalizing these gains while preserving non-target capabilities, connecting context-specified editing, skill learning, and behavioral steering under a common objective. We introduce a five-task-family testbed measuring consolidation and retention under single, batch, and sequential updates. We propose Cortex, which generates prospective queries to anticipate uses of context and distills context-conditioned outputs into mergeable low-rank or structural feed-forward network updates. Across 4B--120B dense and mixture-of-experts backbones, including both text-only and multimodal models, Cortex lies on the empirical consolidation--retention Pareto frontier in most evaluated settings. After 50 sequential factual edits on a 14B model, Cortex-Structural
    
[^20]: FOLD：面向长上下文投机解码的具有局部顺序的无位置远端记忆

    FOLD: Position-Free Far Memory with Local Order for Long-Context Speculative Decoding

    [https://arxiv.org/abs/2610.20392](https://arxiv.org/abs/2610.20392)

    FOLD是一种免训练的长上下文投机解码方法，它保留精确的局部位置，将所有远端条目作为共享位置上的远端记忆处理，并通过锐化记忆内部的注意力来缓解注意力稀释问题。

    

    投机解码通过轻量级起草与并行验证来加速大语言模型的生成。它对长输入尤其有前景，因为KV缓存的访问成本可以摊销到多个被接受的token上。然而，大多数起草器是在短序列上训练的，在长输入上性能会退化，而长上下文训练的代价高昂。现有的免训练方法要么限制上下文窗口，要么重新缩放位置，要么用检索到的远端条目来增强近期上下文。然而，起草器所读取的目标隐藏状态已经编码了上下文与位置信息，并且其轻量级架构使得全历史注意力变得可行。我们提出FOLD，一种免训练方法，它保留精确的局部位置，并将所有远端条目视为处于共享位置上的远端记忆。为应对记忆增长带来的注意力稀释问题，我们在该记忆内部锐化注意力以突出相关条目，同时单独控制其……（摘要在此处截断）

    arXiv:2610.20392v1 Announce Type: new  Abstract: Speculative decoding accelerates LLM generation through lightweight drafting and parallel verification. It is particularly promising for long inputs, where KV-cache access costs can be amortized over multiple accepted tokens. Yet most drafters are trained on short sequences and degrade on long inputs, while long-context training is costly. Existing training-free approaches restrict the context window, rescale positions, or augment recent context with retrieved distant entries. Drafters, however, read target hidden states that already encode contextual and positional information, and their lightweight architecture makes full-history attention affordable. We propose FOLD, a training-free method that preserves exact local positions and treats all distant entries as far memory at a shared position. To counter attention dilution as the memory grows, we sharpen attention within it to emphasize relevant entries while separately controlling its 
    
[^21]: 从分诊到影响：一个面向可解释波斯语仇恨言论的多跳推理基准

    From Triage to Impact: A Multi-Hop Reasoning Benchmark for Explainable Persian Hate Speech

    [https://arxiv.org/abs/2610.20370](https://arxiv.org/abs/2610.20370)

    本研究提出了首个面向波斯语可解释仇恨言论分析的细粒度多跳推理基准，涵盖从内容分诊到严重程度评估的多项任务，发现大语言模型在需要更细粒度语义与推断性推理的任务上性能下降，且引入中间推理步骤可显著提升严重程度预测效果。

    

    仇恨言论在社交媒体上日益泛滥，引发了人们对其检测、解读和内容审核的关注。波斯语作为一种相对低资源的语言，缺乏足够的标注数据和计算资源来支持细粒度的仇恨言论分析。在本研究中，我们提出了一个用于可解释波斯语仇恨言论分析的细粒度、多跳基准，涵盖内容分诊、目标识别、仇恨策略与道德基础分析、文本证据提取、有害叙事推断以及严重程度评估。通过评估最先进的大语言模型（LLMs），我们发现随着任务需要更细粒度的语义和推断性推理，模型性能普遍下降。专有模型总体上优于开源权重模型，而面向推理的模型并未显示出一致的优势。引入中间推理步骤能够显著提升严重程度预测的表现。

    arXiv:2610.20370v1 Announce Type: new  Abstract: Hate speech has become increasingly prevalent on social media, raising concerns about its detection, interpretation, and moderation. Persian, a relatively low-resource language, lacks sufficient annotated data and computational resources for fine-grained hate speech analysis. In this study, we introduce a fine-grained, multi-hop benchmark for explainable Persian hate speech analysis, covering content triage, target identification, hate tactic and moral foundation analysis, textual evidence extraction, harmful narrative inference, and severity assessment. By evaluating state-of-the-art large language models (LLMs), we find that performance generally declines as tasks require more fine-grained semantic and inferential reasoning. Proprietary models generally outperform open-weight models, while reasoning-oriented models show no consistent advantage. Incorporating intermediate reasoning steps substantially improves severity prediction compar
    
[^22]: 面向高效视觉文档搜索的级联检索与跨模态查询精炼方法

    Cascaded Retrieval with Cross-Modal Query Refinement for Efficient Visual Document Search

    [https://arxiv.org/abs/2610.20367](https://arxiv.org/abs/2610.20367)

    提出SCOUT级联双轨检索框架，通过轻量级视觉检索器先筛选候选、再由多向量重排序器精排，并结合语义轨道检索离线生成的分层摘要，最后用跨模态查询精炼融合两路信号，实现高效且可扩展的视觉文档搜索。

    

    从视觉丰富的文档中检索信息，需要将查询与包含表格、图表以及依赖布局的复杂证据的页面进行匹配。当前的视觉-语言检索器虽然能够解决这一问题，但其查询时的计算成本会随语料库规模增长而上升，难以实际扩展。我们提出了SCOUT，一个级联式双轨检索框架，将检索性能与穷举式搜索分离。在视觉轨道中，首先使用一个轻量级的父检索器将语料库过滤为一个紧凑的候选池，然后仅对这些候选应用更强的多向量重排序器。与此同时，语义轨道对离线生成的分层页面和文档摘要进行检索，提供与视觉匹配互补的信号。两路输出通过跨模态查询精炼进行融合，该机制在最终排序之前利用语义分数调整视觉查询的表示。我们在4个主要基准（原文在此处截断）上评估了SCOUT。

    arXiv:2610.20367v1 Announce Type: new  Abstract: Retrieving information from visually rich documents requires matching queries against pages that contain complex evidence in tables, figures, and layout-dependent structures. Current vision-language retrievers address this but incur query-time costs that grow with corpus size and are impractical to scale. We propose SCOUT, a cascaded two-track retrieval framework that separates retrieval performance from exhaustive search. The visual track first uses a lightweight parent retriever to filter the corpus to a compact candidate pool, then applies a stronger multi-vector reranker only to those candidates. In parallel, a semantic track retrieves over hierarchical page and document summaries generated offline to provide a signal complementary to visual matching. The outputs are fused through cross-modal query refinement, which uses semantic scores to adjust the visual query representation before final ranking. We evaluate SCOUT on 4 major bench
    
[^23]: AI智能体社区中缺乏语义固化的词汇持续性

    Lexical Persistence Without Semantic Entrenchment in AI Agent Communities

    [https://arxiv.org/abs/2610.20365](https://arxiv.org/abs/2610.20365)

    该研究通过跨平台对比发现，AI智能体社区虽然能够维持词汇的流行与传播，但缺乏人类语言社区所具有的长程时间语义依赖结构，其语义漂移速度更快、语义稳定性更低。

    

    AI智能体社区能够维持词汇持续性——流行词汇会传播、复现并获得广泛采用——但它们是否也能发展出人类语言社区所特有的时间语义结构，这一问题仍悬而未决。我们使用罗盘对齐分布嵌入方法比较了三个平台：MoltBook（AI智能体社区，8周）、PTT（中文人类BBS论坛，19年）和Reddit（英文人类论坛，匹配的8周窗口），并测量了语义漂移（B1）、邻域稳定性（B2）以及逐词漂移轨迹的滞后一阶自相关性。研究发现，各平台的滞后一阶自相关性形成了一个“记忆视野”连续体：AI智能体的结果聚集在低端，而人类平台则表现出更长程的时间依赖性。MoltBook的B1语义漂移约为两个人类平台的2倍，B2稳定性则比两个人类平台低20-30%。作为对社区年龄的部分控制，创建于2025年2月8日的r/vibecoding子版块在其最初的…（摘要在此处截断）

    arXiv:2610.20365v1 Announce Type: new  Abstract: AI agent communities sustain lexical persistence, popular terms spread, recur, and achieve broad adoption, but whether they also develop the temporal semantic structure characteristic of human linguistic communities remains open. We compare MoltBook (AI agent community, 8 weeks), PTT (Chinese human BBS, 19 years), and Reddit (English human forum, in matched 8-week windows) using Compass-Aligned Distributional Embeddings (CADE), measuring semantic drift (B1), neighborhood stability (B2), and lag-1 autocorrelation of per-word drift trajectories. Across the platforms, lag-1 autocorrelation forms a memory-horizon continuum. AI agents' result clusters at the low end, while human platforms show longer-range temporal dependence. MoltBook exhibits approximately 2$\times$ the B1 drift and 20-30% lower B2 stability than both human platforms. As a partial control for community age, r/vibecoding (founded 2025-02-08) shows significant ACF in its firs
    
[^24]: AI智能体会八卦吗？人类与AI智能体在线社区中的评价性谈论

    Do AI Agents Gossip? Evaluative Talk in Human and AI-Agent Online Communities

    [https://arxiv.org/abs/2610.20360](https://arxiv.org/abs/2610.20360)

    本研究首次通过对比AI智能体社区（Moltbook）与人类在线论坛（PTT），发现AI智能体社区中会自发涌现类似人类八卦的评价性言论，尽管其流行度显著低于人类社区，且两者在后续回应的发展模式上存在明显差异。

    

    八卦，即对不在场的第三方进行评价性谈论，被广泛视为一种独特的人类社会实践。本研究考察了类似的言论是否会在一个由自主AI智能体组成的社区中自发涌现。我们将Moltbook（一个由AI智能体组成的Reddit风格平台）与PTT（台湾一个大型人工运营的电子公告板）进行了比较。为了减少因交流环境差异带来的影响，我们采用相同的八卦操作性定义，对两个公开的、异步的论坛进行了分析。研究发现，Moltbook上确实存在八卦，但其流行度比PTT低2.6至6倍（取决于所采用的分母）。在两个平台上，八卦通常针对本地知名人物，往往能获得第一个回应者的赞同，且很少被呈现为毫无根据的谣言。最明显的差异在于后续回应如何展开：在包含至少一条可编码回复的帖子中，人类社区中出现了65.1%的挑战性回应……

    arXiv:2610.20360v1 Announce Type: new  Abstract: Gossip, defined as evaluative talk about absent third parties, is widely seen as a uniquely human social practice. This study examines whether comparable discourse emerges spontaneously in a community of autonomous AI agents. We compare Moltbook, a Reddit-style platform populated by AI agents, with PTT, a large human-operated bulletin board in Taiwan. To reduce differences attributable to communicative setting, we analyze two public, asynchronous forums using the same operational definition of gossip. Gossip occurs on Moltbook, but its prevalence is 2.6 to 6 times lower than on PTT, depending on the denominator used. On both platforms, gossip frequently targets locally prominent figures, tends to elicit agreement from the first respondent, and is rarely presented as unsupported rumor. The clearest difference concerns how subsequent responses develop. Among threads containing at least one codable reply, challenges occur in 65.1% of human 
    
[^25]: TreeFlex：智能体自主导航自身搜索树的框架

    TreeFlex: Agents Navigating Their Own Search Trees

    [https://arxiv.org/abs/2610.20334](https://arxiv.org/abs/2610.20334)

    TreeFlex让单一语言模型在反思与执行两种角色间切换，在框架维护的搜索树上自主选择从哪个节点恢复搜索，从而解决ReAct式智能体因上下文膨胀导致的推理失效问题。

    

    ReAct风格的智能体在单一上下文中不断累积观察、中间结果和失败的尝试。随着上下文不断增长，过时和偏离主路径的信息会消耗其有限的容量并掩盖相关证据，使聚焦推理变得越来越困难。我们提出了TreeFlex，这是一个由智能体自身策略决定搜索方向的框架。智能体将每个中间状态连同一段简短摘要，记录为由框架（harness）维护的搜索树中的一个节点。该策略是一个承担两种角色的单一语言模型：在“反思”角色中，它通过这些摘要获得树的全局视图，并选择一个要恢复的节点及附带一条指令；在“执行”角色中，框架恢复通向该节点的路径，模型使用常规任务工具执行该指令，直到保存一个新的节点或提出解决方案，随后反思角色会批准该方案或将其送回另一个节点继续探索。

    arXiv:2610.20334v1 Announce Type: cross  Abstract: A ReAct-style agent accumulates observations, intermediate results, and failed attempts in a single context. As this context grows, stale and off-path information consumes its limited capacity and obscures relevant evidence, making focused reasoning increasingly difficult. We introduce TreeFlex, a framework in which the agent's own policy decides where to search. The agent records each intermediate state, with a short summary, as a node in a search tree maintained by the harness. The policy is a single language model that acts in two roles. In the Reflection role, it has a global view of the tree through these summaries and picks a node to resume, together with an instruction. In the Execution role, the harness restores the path that led to that node, and the model carries out the instruction with the usual task tools until it saves a new node or proposes a solution, which Reflection then approves or sends back to another node. Through
    
[^26]: Anuvani-Diarization：面向22种语言的零样本说话人日志与标注规范

    Anuvani-Diarization: Zero Shot Speaker Diarization and Annotation Conventions Across 22 Languages

    [https://arxiv.org/abs/2610.20327](https://arxiv.org/abs/2610.20327)

    该论文提出Anuvani-Diarization，通过不确定性感知的银标准监督将自监督端到端分割模型适配到印度22种表列语言的多语言对话场景，实现零样本说话人日志，在IndicDiarBench上取得16.78%的最低时长加权错误率。

    

    印度的22种表列语言由超过十亿人使用，他们的对话常常混杂多种语言（甚至在同一话轮内），语音重叠频繁，并发生在电话、会议和媒体等多种场景中。明确“谁在何时说话”，即说话人日志任务，对此类对话的语音应用至关重要：例如与多人同时交流的语音代理、会议和通话的说话人归属转录、呼叫中心分析以及媒体监控。我们在IndicDiarBench上对全部22种语言的说话人日志进行了系统研究，并提出Anuvani-Diarization，该方法通过不确定性感知的银标准监督，将自监督端到端分割模型适配到印度语对话场景。在零样本设置下与九个近期日志系统一同评估，该方法在全部1,164条录音上取得了16.78%的时长加权说话人日志错误率（DER），是我们所运行的所有系统中最低的，仅次于最佳系统（原文在此处截断）。

    arXiv:2610.20327v1 Announce Type: new  Abstract: India's 22 scheduled languages are spoken by over a billion people in conversations that mix languages, often within one turn, overlap freely, and take place over telephones, in meetings, and in media. Knowing who spoke when, the task of speaker diarization, is essential to speech applications for such conversation: voice agents conversing with several people at once, speaker attributed transcription of meetings and calls, contact center analytics, and media monitoring. We present a systematic study of speaker diarization across all 22 languages on IndicDiarBench and introduce Anuvani-Diarization, which adapts a self-supervised end-to-end segmentation model to Indian conversation with uncertainty aware silver supervision. Evaluated zero shot alongside nine recent diarization systems, it reaches a duration weighted diarization error rate (DER) of 16.78% on all 1,164 recordings, the lowest of all systems we ran and second only to the best 
    
[^27]: MMLA-RSI：通过RTT和RTMA实现的递归自我改进

    MMLA-RSI: Recursive Self-Improvement through RTT and RTMA

    [https://arxiv.org/abs/2610.20294](https://arxiv.org/abs/2610.20294)

    提出了一种递归自我改进方案MMLA-RSI，其中RTT改变策略状态、RTMA改变权威记忆，且这两种改进方法本身也可作为改进对象，由独立接纳的后继者在下一代中接管相同角色，从而实现智能体行为的持续递归改进。

    

    智能体可以通过改变其行为方式和所保留的经验，在未解决的问题中改进自身的行为。我们提出MMLA-RSI，这是一种在记忆中介学习架构中具有两种实现的递归自我改进方案：推理时训练（RTT）改变策略状态 $\Phi$，而推理时记忆适配（RTMA）改变权威记忆 $M$。每条路线还将其适配方法本身作为改进的对象。当前的数值更新器训练提出其后继者的策略；当前的记忆管理器选择可供提出其后继者的进程所使用的记录。被独立接纳的后继者在下一代中接管相同的角色。我们给出了有界可执行规则语言、共享的任务/提案接口，以及适用于任一路线或其组合的单一算法。RTT实例在有界动量更新中改变标量调度；RTMA实例选择NULL或一个co（摘要在此处被截断）

    arXiv:2610.20294v1 Announce Type: new  Abstract: An agent can improve its behavior within an unresolved problem by changing how it acts and which experience it retains. We propose MMLA-RSI, a recursive self-improvement scheme with two implementations in Memory-Mediated Learning Architecture: Reasoning-Time Training (RTT) changes the policy state $\Phi$, while Reasoning-Time Memory Adaptation (RTMA) changes authoritative memory $M$. Each route also makes its adaptation method an object of improvement. A current numerical updater trains the policy proposing its successor; a current memory manager chooses the records available to the process proposing its successor. An independently admitted successor takes over the same role in the next generation. We give bounded executable rule languages, shared task/proposal interfaces, and one algorithm for either route or their composition. The RTT instance changes scalar schedules in a clipped-momentum update. The RTMA instance chooses NULL or a co
    
[^28]: 为医学推理把脉（PULSE）：面向知识支撑推理的步骤级检索时机判断

    Taking the PULSE of Medical Reasoning: Step-Level Retrieval Timing for Knowledge-Grounded Inference

    [https://arxiv.org/abs/2610.20292](https://arxiv.org/abs/2610.20292)

    提出PUlSE方法，利用大模型内部表征的层间波动性作为内在信号，在医学推理的每个步骤自适应判断是否需要外部检索，从而在提升事实准确性的同时避免不必要的检索噪声与延迟。

    

    可靠的医学推理要求大语言模型（LLM）能够动态访问外部生物医学证据，以提升事实可信度和推理准确性。然而，不加区分的检索会引入噪声并增加延迟，由此产生一个关键的“检索时机”问题：在每个推理步骤中，系统必须判断模型的内部知识是否可靠，还是需要获取外部证据。我们提出PUlSE，一种轻量级的探测方法，它将模型内部表征作为步骤级检索控制的内在信号。我们的后验轨迹分析揭示，错误步骤的层间波动性显著高于正确步骤，这表明其内部知识处于不稳定状态。基于这一发现，PUlSE在层间几何特征上训练一个紧凑的探测器，在步骤完成时检测事实性错误，并将其作为触发检索与步骤重新生成的自适应开关。在四个医学基准上的（实验结果表明……）

    arXiv:2610.20292v1 Announce Type: new  Abstract: Reliable medical reasoning requires LLMs to dynamically access external biomedical evidence to improve factual trustworthiness and reasoning accuracy. However, indiscriminate retrieval can introduce noise and latency, raising a critical retrieval-timing problem: at each reasoning step, the system must decide whether the model's internal knowledge is reliable or external evidence is needed. We propose PUlSE, a lightweight probing method that uses the model's internal representations as intrinsic signals for step-level retrieval control. Our posterior trajectory analysis reveals that erroneous steps exhibit substantially higher inter-layer volatility than correct ones, suggesting unstable internal knowledge. Based on this finding, PUlSE trains a compact probe on inter-layer geometric features to detect factual errors upon step completion, and uses it as an adaptive trigger for retrieval and step regeneration. Across four medical benchmarks
    
[^29]: 面向低资源萨拉基语的神经词性标注

    Neural POS Tagging for the Low-Resource Saraiki

    [https://arxiv.org/abs/2610.20279](https://arxiv.org/abs/2610.20279)

    本文构建了首个经人工标注的萨拉基语通用词性标注数据集，并证明XLM-RoBERTa base在该低资源语言上表现最佳，测试准确率达88.60%。

    

    词性标注是一项基础的序列标注任务，也是许多下游自然语言处理应用的重要组成部分。然而，萨拉基语使用从右向左书写的波斯-阿拉伯文字，是一种资源匮乏的印度-雅利安语系语言，尽管根据巴基斯坦2023年第七次人口与住房普查，约有28,849,579人（约2890万）以其为母语。本文提出了一个基于纯文本开发、经人工标注的萨拉基语通用词性标注数据集。我们使用17个词性类别对文本进行了标注，并在相同的微调设置下评估了三个预训练transformer编码器：XLM-RoBERTa base、Urdu RoBERTa small和多语言DistilBERT。基于五个随机种子的实验结果表明，XLM-RoBERTa base取得了最佳性能，开发集准确率为87.29% ± 0.21%，测试集准确率为88.60% ± 0.21%。Urdu RoBERTa small和多语言DistilBE……（原文摘要在此处被截断）

    arXiv:2610.20279v1 Announce Type: new  Abstract: Part-of-speech tagging is a fundamental sequence labeling task and an important component for many downstream natural language processing applications. However, Saraiki, written from right to left in a Perso-Arabic script, remains an under-resourced Indo-Aryan language, despite being spoken as a mother tongue by 28,849,579 people (approximately 28.9 million) according to Pakistan 7th Population and Housing Census 2023. This paper presents a manually annotated Saraiki Universal POS tagging dataset developed from plain-text. We annotated the text with 17 POS categories. We evaluate three pretrained transformer encoders under the same fine-tuning setup: XLM-RoBERTa base, Urdu RoBERTa small, and multilingual DistilBERT. Experimental results over five random seeds show that XLM-RoBERTa base achieves the best performance, with 87.29% +/- 0.21% development accuracy and 88.60% +/- 0.21% test accuracy. Urdu RoBERTa small and multilingual DistilBE
    
[^30]: CSWBench：评估创意写作中语境敏感的语码转换

    CSWBench: Evaluating Context-Sensitive Code-Switching in Creative Writing

    [https://arxiv.org/abs/2610.20272](https://arxiv.org/abs/2610.20272)

    该论文提出了CSWBench基准与基于理论的CSWEval评估框架，从事件层面（许可性、语境适当性、功能增益）和全文层面两个维度，对大语言模型在创意写作中语境敏感的语码转换能力进行系统评估，并对十二个模型进行了实验分析。

    

    大语言模型（LLMs）已在多样化的语言生成与理解任务中展现出多语言能力。然而，有效的语码转换不仅仅需要多语言熟练度：模型必须决定何时、何地以及为何切换语言，使之符合周围语境。这种语境敏感的能力尚未得到充分评估。我们提出了CSWBench，这是一个通过两种互补设置——聚焦叙事与扩展叙事——来评估创意写作中语境敏感语码转换的基准。我们进一步提出CSWEval，这是一个基于理论的评估框架，包含两个互补层面：事件层面的评估，涵盖许可性、语境适当性与功能增益；以及全文层面的任务特定质量评估——在聚焦叙事中侧重整体文学与叙事质量，在扩展叙事中侧重强调多语言组织的质量。我们对十二个大语言模型进行了评估，发现……

    arXiv:2610.20272v1 Announce Type: new  Abstract: Large language models (LLMs) have demonstrated multilingual capabilities across diverse language generation and understanding tasks. Yet effective code-switching requires more than multilingual proficiency: models must decide when, where, and why to switch languages in ways that fit the surrounding context. This context-sensitive ability remains insufficiently evaluated. We introduce CSWBench, a benchmark for evaluating context-sensitive code-switching in creative writing through two complementary settings: Focused Narrative and Extended Narrative. We further propose CSWEval, a theory-informed framework with two complementary levels: event-level evaluation of Licensing, Contextual Appropriateness, and Functional Gain, and whole-text evaluation of task-specific quality---overall literary and narrative quality in Focused Narrative and quality emphasizing multilingual organization in Extended Narrative. Evaluating twelve LLMs, we find that 
    
[^31]: Anuvani-LID：面向印度语言的鲁棒且校准良好的口语语言识别

    Anuvani-LID: Robust and Calibrated Spoken Language Identification for Indian Languages

    [https://arxiv.org/abs/2610.20263](https://arxiv.org/abs/2610.20263)

    本文提出Anuvani-LID，一个在54,643小时印度语语音上训练的1.15亿参数专用Conformer语言识别模型，覆盖22种印度语言及印度口音英语，在短促、嘈杂、窄带等困难条件下显著优于Whisper等基础语音模型及Gemini、GPT等音频大语言模型。

    

    每个多语言语音处理流水线都始于一个语言判定，而该环节的错误会悄无声息地在识别、翻译和路由中传播。在印度，这涉及横跨四大语系的22种表列语言，且这些语言常出现在短促、嘈杂、窄带以及语码混杂的音频中。基础语音模型越来越倾向于将语言识别（LID）视为一种固有的、潜在的可泛化能力。语音模型中的自回归解码器会输出一个特殊的语言ID标记，而具备音频能力的现代大语言模型在被提示时也能说出语言名称。我们在相同的音频片段上测试了这一假设，发现Whisper及类似语音模型仅对23种语言中的15种具有语言标记；在1秒输入下，Gemini 3.5 Flash的宏平均F1仅为0.52，GPT-audio仅为0.14。我们提出了Anuvani-LID，一个专用的1.15亿参数LID Conformer模型，在54,643小时的印度语语音上训练，涵盖22种印度语言及印度口音英语。同时……（原文摘要在此处截断）

    arXiv:2610.20263v1 Announce Type: new  Abstract: Every multilingual speech pipeline begins with a language decision, an error there propagates silently through recognition, translation and routing. In India it spans 22 scheduled languages across four language families, heard over short, noisy, narrowband and code-mixed audio. Foundational speech models increasingly present language identification (LID) as an inherent latent, generalized capability. Autoregressive decoders in speech models emit a special language ID token, and modern LLMs with audio capabilities can name the language when prompted. We test this assumption on identical clips and find that whisper and similar speech models has tokens for only 15 of the 23 languages, on one second inputs Gemini 3.5 Flash reaches 0.52 macro-F1 and GPT-audio 0.14. We present Anuvani LID, a dedicated 115M parameter LID conformer model trained on 54,643 hours of Indian speech spanning 22 Indian languages and Indian accented English. Along with
    
[^32]: 使用冻结仿射场修复文本更新后的语音访问

    Repairing Speech Access After Text Updates with Frozen Affine Fields

    [https://arxiv.org/abs/2610.20253](https://arxiv.org/abs/2610.20253)

    提出AFFIRM方法，在保持模型检查点完全冻结的前提下，通过可拆卸的仿射场算子（冻结的基础校正加上零均值的查询特定校正）修复文本更新后新知识在语音查询时难以访问的问题。

    

    具备语音能力的语言模型的大部分知识继承自富文本的语言骨干网络，持续的模型维护同样以文本更新为主导。然而，通过文本更新获得的知识，在相同查询以语音形式提出时仍可能难以访问。我们将这种不匹配表述为“更新后语音访问修复”：承载知识的模型检查点保持冻结，由一个可拆卸的算子通过单次模型前向传播修复纯语音访问。我们提出了跨模态接口修复的仿射场方法（AFFIRM），该方法首先学习并冻结针对语音隐藏状态的基础校正，随后添加在算子训练查询上均值为零的查询特定校正。成对的语音与文本状态提供坐标，将校正条件化到每个查询上。在平衡的合成语音PopQA数据集上，成对比较并未确定AFFIRM相对于全局SALSA导向方法或低秩（摘要在此处截断）……的明显优势。

    arXiv:2610.20253v1 Announce Type: new  Abstract: Speech-capable language models inherit most of their knowledge from text-rich language backbones, and continued model maintenance is likewise dominated by text updates. Knowledge acquired through a text update can nevertheless remain difficult to access when the same query is spoken. We formulate this mismatch as post-update speech access repair, where the knowledge-bearing checkpoint stays frozen and a detachable operator repairs speech-only access in one model pass. We introduce Affine Fields for Interface Repair across Modalities (AFFIRM), which first learns and freezes a base correction for speech hidden states, then adds query-specific corrections with zero mean over the operator-training queries. Paired speech and text states provide the coordinates that condition these corrections on each query. On balanced synthetic-speech PopQA, paired comparisons do not establish a clear advantage for AFFIRM over global SALSA steering or low-ra
    
[^33]: 扩展多语言专家混合模型：语言覆盖范围如何改变容量与计算分配

    Scaling Multilingual Mixture-of-Experts: How Language Coverage Changes Capacity and Compute Allocation

    [https://arxiv.org/abs/2610.20247](https://arxiv.org/abs/2610.20247)

    本文提出了一种感知重复的联合缩放定律，揭示多语言MoE模型中活跃参数、训练词元、专家数量与语言覆盖范围之间的关系，并发现32个专家相比稠密模型可将语言从12种扩展到200种时的损失惩罚降低83%。

    

    多语言模型在各语言之间共享容量，因此增加语言数量可能会损害现有语言的表现。专家混合模型只需增加极少量的每词元计算即可扩展这一容量。鉴于这种额外容量，增加专家数量能否降低每种新增语言的代价？为回答这一问题，我们提出了一种感知重复的联合缩放定律，将活跃参数数量、每种语言的训练词元数量、专家数量与语言覆盖范围联系起来。该定律基于来自64个预训练配置的26,300个逐语言损失观测数据进行拟合，在留出配置上相比最强评估基线将预测误差降低了28%。在活跃参数数量和每种语言训练词元数量相同的情况下，32个专家相比稠密模型将语言从12种扩展到200种时的损失惩罚降低了83%。我们的定律进一步表明，更广的语言覆盖范围会在更大的参数范围内倾向于使用更多专家。

    arXiv:2610.20247v1 Announce Type: new  Abstract: Multilingual models share capacity across languages, so adding languages can hurt performance on existing ones. Mixture-of-experts models expand this capacity with little additional per-token computation. Given this extra capacity, does adding experts reduce what each additional language costs? To answer this question, we propose a repetition-aware joint scaling law relating number of active parameters, training tokens per language, number of experts, and language coverage. Fitted on 26,300 per-language loss observations from 64 pretrained configurations, our law reduces prediction error on held-out configurations by 28% relative to the strongest evaluated baseline. At matched number of active parameters and training tokens per language, 32 experts reduce the loss penalty of expanding from 12 to 200 languages by 83% compared with a dense model. Our law further shows that broader language coverage favors more experts across a wider range 
    
[^34]: 通过推理引导的翻译扩展多语言安全与推理评估

    Scaling Multilingual Safety and Reasoning Evaluation through Reasoning-Guided Translation

    [https://arxiv.org/abs/2610.20221](https://arxiv.org/abs/2610.20221)

    提出了一种推理引导的翻译框架，通过分离翻译策略选择与大规模生成，并结合任务感知验证、针对性修复和独立质量筛选，构建了覆盖安全性、对话和数学推理任务的多语言评估基准。

    

    大型语言模型（LLM）日益被部署于多种语言环境中，但其安全性和推理能力仍主要以英语进行评估。翻译现有基准为多语言评估提供了一种可扩展的途径，然而在跨语言过程中保留特定任务的含义面临重大挑战，尤其体现在有害请求、数学推理和对话交互等任务上。我们提出了一种推理引导的翻译框架，用于构建多语言评估集，旨在保留多样化基准任务的结构与预期含义。我们的方法将翻译策略选择与大规模生成分离开来，将推理引导的配置选择与任务感知验证、针对性修复以及独立质量筛选相结合。利用该框架，我们构建了一个源对齐的评估集合，涵盖安全、对话和数学推理等多个领域。

    arXiv:2610.20221v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly deployed across languages, yet their safety and reasoning capabilities remain predominantly evaluated in English. Translating existing benchmarks offers a scalable approach to multilingual evaluation, but preserving task-specific meaning across languages presents significant challenges, particularly for harmful requests, mathematical reasoning, and conversational interactions. We introduce a reasoning-guided translation framework for constructing multilingual evaluations designed to preserve the structure and intended meaning of diverse benchmark tasks. Our approach separates translation strategy selection from large-scale generation, combining reasoning-guided configuration selection with task-aware validation, targeted repair, and independent quality screening. Using this framework, we develop a source-aligned evaluation collection spanning seven safety, conversational, and mathematical rea
    
[^35]: 什么使社交智能体的互动真正有效？在配对世界中从可验证的结果中学习

    What Makes a Social Agent's Interactions Effective? Learning from Verifiable Outcomes in Paired Worlds

    [https://arxiv.org/abs/2610.20195](https://arxiv.org/abs/2610.20195)

    该论文提出HICO-Bench基准，通过对达成计划的可行性进行程序验证、对对话完整性进行LLM裁判检验的双重机制，利用成对场景（公开上下文相同但私有条件不同）评估社交智能体，揭示仅有高评分对话并不等于达成了可验证的成功协调。

    

    有效的社交互动不仅仅需要流畅的回应：智能体必须达成符合参与者需求与处境的计划。现有对社交智能体的评估主要依赖LLM裁判对对话进行评分，而不检查最终形成的计划。我们提出了HICO-Bench，一个包含180个日常协调场景（如资源分配、日程安排和授权）的基准，其中两个持有私有信息的智能体进行对话，然后各自独立记录他们达成的计划。程序检查可行性（两份记录一致且满足双方的私有条件），LLM裁判检查完整性（对话内容支持所记录的计划）；同时通过两项检查即为“经验证的成功”。每个场景有两个版本，它们共享相同的公开上下文，但仅在一个私有条件上不同，因此两个版本中正确的计划各不相同。在十三个LLM中，36%被LLM裁判评为目标达成度较高的对话……（原摘要在此处截断）

    arXiv:2610.20195v1 Announce Type: new  Abstract: Effective social interaction requires more than fluent responses: agents must reach plans that fit the participants' needs and circumstances. Existing evaluations of social agents mainly rely on LLM judges that rate the conversation, without checking the resulting plan. We introduce HICO-Bench, a benchmark of 180 everyday coordination scenarios, such as resource allocation, scheduling, and authorization, in which two agents with private information converse and then independently record the plan they agreed on. A program checks Feasibility (the two records match and satisfy both participants' private conditions), and an LLM judge checks Integrity (the conversation supports the recorded plan); passing both is a Verified Success. Each scenario has two versions that share the same public context but differ in one private condition, so the correct plan differs between them. Across thirteen LLMs, 36% of conversations rated highly for goal com
    
[^36]: 当职业衍生技能无法发挥作用：论技能增强型职业路径预测的局限性

    When Occupation-derived Skills Don't Help: On the Limits of Skill-Augmented Career Path Prediction

    [https://arxiv.org/abs/2610.20186](https://arxiv.org/abs/2610.20186)

    该研究批判性考察了技能表示在职业路径预测中的实际价值，发现无论是预测技能还是真实ESCO技能标注均无法显著超越纯文本基线，其根本原因在于文本与技能表示的高度冗余、ESCO分类判别力有限以及技能嵌入的表示坍缩。

    

    技能被广泛认为在职业转换中发挥着核心作用，这推动了技能在职业路径预测（CPP）系统中的应用。在这项工作中，我们批判性地考察了显式技能表示能否在基于文本的工作表示之外提供补充信号。我们在两种条件下——预测技能和真实（ground-truth）ESCO标注——评估了将技能融入CPP模型的多种策略，包括分数融合和神经架构。令人惊讶的是，在两个基准数据集上，这些设置均未比强大的纯文本基线带来实质性改进。即使使用真实技能标注，收益也依然微弱且不一致。通过详细分析，我们识别出导致这一结果的关键因素：文本表示与技能表示之间的高度冗余、ESCO分类体系有限的判别能力，以及技能嵌入中的表示坍缩。此外，我们发现技能…

    arXiv:2610.20186v1 Announce Type: new  Abstract: Skills are widely assumed to play a central role in career transitions, motivating their use in career path prediction (CPP) systems. In this work, we critically examine whether explicit skill representations provide complementary signal beyond text-based job representations. We evaluate multiple strategies for incorporating skills into CPP models, including score fusion and neural architectures, under two conditions: predicted skills and ground-truth ESCO annotations. Surprisingly, across two benchmark datasets, none of these settings yield substantial improvements over strong text-only baselines. Even when using ground-truth skills, gains remain marginal and inconsistent. Through detailed analysis, we identify key factors underlying this result: high redundancy between textual and skill representations, limited discriminative power of ESCO's taxonomy, and representational collapse in skill embeddings. Furthermore, we find that skill pr
    
[^37]: 跨模型上下文内神经反馈揭示大语言模型中的（有限）激活控制能力

    Cross-Model In-Context Neurofeedback Reveals (Limited) Activation Control in LLMs

    [https://arxiv.org/abs/2610.20179](https://arxiv.org/abs/2610.20179)

    本文提出跨模型上下文内神经反馈新范式，首次将模型在任务上的胜任能力与对自身激活的真实访问区分开来，测试十个开源大模型后发现大语言模型仅具备有限的激活控制能力。

    

    基于激活向量的探针被广泛用于检测语言模型的不良行为，但其有效性依赖于一个假设：模型无法控制其激活所编码的内容。人类可以通过神经反馈学会控制甚至单个神经元的放电，近期有研究探讨了这种能力是否也适用于语言模型，但得出了相互矛盾的结论。迄今为止，尚无实验设计能够将模型在任务上的表现能力与其对自身激活的真实访问区分开来。我们通过“上下文内神经反馈”会话填补了这一空白：在其中，一个“受试”模型被要求最大化（或最小化）一个由其自身激活计算得出的分数，且该分数会在每一轮对话中展示给该模型。在我们的第一个实验中，句子要么由受试模型自己生成，要么由第二个“写作”模型生成，而分数始终来自受试模型的激活。我们测试了来自五个系列的十个开源权重模型，规模从9B到32B。部分模型……（原文摘要在此处被截断）

    arXiv:2610.20179v1 Announce Type: new  Abstract: Probes on activation vectors are widely used to detect misbehavior in language models, but their success relies on the assumption that models cannot control what their activations encode. Humans can learn, through neurofeedback, to control the firing of even single neurons, and recent studies investigated whether this ability extends to language models, reaching conflicting conclusions. No experimental design so far disentangles competence at the task from genuine access to one's own activations. We fill this gap with in-context neurofeedback sessions, in which a "subject" model is asked to maximize (or minimize) a score derived from its activations, shown back to it at every turn. In our first experiment, the sentences are generated either by the subject or by a second "writer" model, while the score always derives from the activations of the subject. We test ten open-weight models from five families, with sizes from 9B to 32B. Some mod
    
[^38]: TreeSpark：为推测解码学习草拟你所验证的树

    TreeSpark: Learning to Draft the Trees You Verify for Speculative Decoding

    [https://arxiv.org/abs/2610.20175](https://arxiv.org/abs/2610.20175)

    TreeSpark 提出一个统一的训练与推理框架，通过一次骨干前向传播构建父节点条件的候选树，并利用生存负对数似然和有上限覆盖铰链两种目标函数（复用参考链目标分布、无需额外目标模型前向传播）来训练推测解码的草稿树，从而提升并行草稿的接受率。

    

    诸如 DFlash 和 DSpark 等并行起草器能够降低提案延迟，但早期的失配限制了每个文本块中可被接受的部分。保留备选方案会引入父状态和候选选择，而基于参考的条件训练并不能直接监督这些内容。我们提出 TreeSpark，这是一个扩展半自回归 DSpark 架构的统一训练与推理框架。它通过一次骨干网络前向传播构建以父节点为条件的候选树，并在训练期间重放其扩展、剪枝和预算约束下的打包过程。生存负对数似然根据源自参考模型的路径质量为保留的边加权，同时一个设有上限的覆盖铰链损失则促进局部选择边界附近对目标模型重要的备选方案。这两个目标函数均复用参考链的目标分布，无需额外的目标模型前向传播。我们在九个通用基准和六个写作领域上，使用 Qwen3-4B 和 Qwen3-8B 以贪心解码方式进行评估。

    arXiv:2610.20175v1 Announce Type: new  Abstract: Parallel drafters such as DFlash and DSpark reduce proposal latency, but early mismatches limit how much of each block can be accepted. Retaining alternatives introduces parent states and candidate choices that reference-conditioned training does not directly supervise. We introduce TreeSpark, a unified training and inference framework extending the semi-autoregressive DSpark architecture. It constructs parent-conditioned candidate trees from one backbone pass and replays their expansion, pruning, and budgeted packing during training. Survival negative log-likelihood weights retained edges by reference-derived path mass, while a capped coverage hinge promotes target-important alternatives near local selection boundaries. Both objectives reuse reference-chain target distributions without additional target forward passes. We evaluate greedy decoding with Qwen3-4B and Qwen3-8B on nine general benchmarks and six writing domains. Across targe
    
[^39]: RM-OPD：面向生成式奖励模型的同策略蒸馏

    RM-OPD: On-Policy Distillation for Generative Reward Models

    [https://arxiv.org/abs/2610.20174](https://arxiv.org/abs/2610.20174)

    该论文提出RM-OPD两阶段框架，先通过单领域GRPO微调训练各领域专家教师模型，再通过路由同策略蒸馏将其整合为单一学生模型，从而解决Mix-RL训练生成式奖励模型时的奖励稀疏与跨领域干扰问题。

    

    基于混合领域偏好数据的强化学习（Mix-RL）已成为训练通用生成式奖励模型（GRM）的主流方法。然而，Mix-RL 存在两个关键瓶颈：由结果奖励导致的奖励稀疏性，以及联合优化异构能力所引起的跨领域干扰，二者共同导致了能力获取不均衡和普遍的奖励欺骗问题。受同策略蒸馏（On-Policy Distillation, OPD）在多领域能力融合中成功应用的启发，我们提出了 RM-OPD（通过同策略蒸馏进行奖励建模），这是一个将专家能力获取与能力整合相分离的两阶段框架。首先，我们通过单领域 GRPO 微调培养领域专精的教师模型，使每个专家能够在隔离状态下达到其性能上限；然后，我们通过路由同策略蒸馏将这些教师模型整合到一个学生模型中。

    arXiv:2610.20174v1 Announce Type: cross  Abstract: Reinforcement Learning on Mixed-domain preference data (Mix-RL) has become the mainstream approach for training general-purpose Generative Reward Models (GRMs). However, Mix-RL suffers from two critical bottlenecks: the reward sparsity induced by outcome rewards, and the cross-domain interference arising from jointly optimizing heterogeneous capabilities, which together give rise to imbalanced capability acquisition and pervasive reward hacking. Inspired by the success of On-Policy Distillation (OPD) in multi-domain capability integration, we introduce RM-OPD (Reward Modeling via On-Policy Distillation), a two-stage framework that separates specialist acquisition from capability integration. First, we cultivate domain-specialized teacher models via single-domain GRPO fine-tuning, allowing each expert to reach its performance ceiling in isolation; then we consolidate these teachers into one student model through (i) routed on-policy dis
    
[^40]: MultiPartyTurnBench：多方口语对话中的话轮控制基准测试

    MultiPartyTurnBench: Benchmarking Floor Control in Multi-Party Spoken Dialogue

    [https://arxiv.org/abs/2610.20172](https://arxiv.org/abs/2610.20172)

    MPTB是首个针对多方口语对话中话轮控制的统一受控基准，通过240段信道分离、涵盖2至5个说话人的合成对话及精确事件标注，系统评估了话轮结束检测与打断检测任务，实现了对真实话轮转移的因果评估。

    

    全双工语音系统必须决定何时进入、保持或放弃对话的话轮权，即使多个说话人出现重叠发言或仅做伴随反馈而不夺取话轮。现有的评估工作要么研究双人话轮转换，要么研究端到端智能体行为，但在多方场景中缺乏针对真实话轮转移与易混淆非事件之间的统一、受控的评估。我们提出了MultiPartyTurnBench（MPTB），这是一个包含240段合成生成、信道分离对话的基准，涵盖2至5个说话人以及6种交互类型。MPTB评估两个互补任务：话轮结束（EOT）检测，用于区分真正的话轮完成与临时停顿；以及打断（INT）检测，用于区分成功的话轮接管与非夺取话轮的言语。精确的事件条件标注使得能够采用误报约束召回率和带符号延迟进行因果评估。在声学、语义和话轮投影基线（摘要在此处截断）

    arXiv:2610.20172v1 Announce Type: cross  Abstract: Full-duplex speech systems must decide when to enter, hold, or relinquish the conversational floor, even as multiple speakers overlap or backchannel without taking it. Existing evaluations study dyadic turn-taking or end-to-end agent behavior, but lack a unified, controlled evaluation of genuine floor transitions against confusable non-events in multi-party settings. We introduce MultiPartyTurnBench (MPTB), a benchmark of 240 synthetically generated, channel-separated conversations spanning two to five speakers and six interaction types. MPTB evaluates two complementary tasks: End-of-Turn (EOT) detection distinguishes genuine turn completion from temporary holds, while Interruption (INT) detection distinguishes successful floor takeovers from non-floor-taking speech. Exact event-conditioned annotations enable causal evaluation using false-positive-constrained recall and signed latency. Across acoustic, semantic, and turn-projection bas
    
[^41]: 差距检测：通过需求与可辩护性缓解同质化

    Gap Detection: Mitigating Homogenization through Demand and Defensibility

    [https://arxiv.org/abs/2610.20162](https://arxiv.org/abs/2610.20162)

    提出Gap Detection框架，通过结合查询相关性（Demand）与现有内容相异度（Defensibility）的GapScore目标函数，在文本嵌入空间中检测相关但未被占据的方向，以缓解语言模型开放式回答的同质化问题。

    

    语言模型在回答开放式问题时常常生成语义相似的答案，即使存在许多合理的替代方案。生成有用的替代方案需要同时识别查询所要求的内容以及现有回答已经覆盖的内容。我们提出了差距检测（Gap Detection），这是一个在共享文本嵌入空间中搜索相关但尚未被占据的方向的框架。其目标函数GapScore将查询相关性（需求，Demand）与最近现有条目的相异度（可辩护性，Defensibility）相结合。该框架在单位球面上优化该乘积，利用语言模型将候选方向映射为概念词，并通过任务特定的过滤和受约束的提示将这些概念转化为文本。在五个精选主题和90个INFINITY-CHAT查询上，报告的输出碰撞率平均相对降低分别为12.8%和10.8%。我们还将该框架实例化应用于四个产品的生成式引擎优化。

    arXiv:2610.20162v1 Announce Type: new  Abstract: Language models often produce semantically similar answers to open-ended questions, even when many plausible alternatives exist. Generating useful alternatives requires identifying both what the query calls for and what existing responses already cover. We introduce Gap Detection, a framework that searches for relevant but unoccupied directions in a shared text embedding space. Its objective, GapScore, combines query relevance (Demand) with dissimilarity to the nearest existing item (Defensibility). The framework optimizes this product on the unit sphere, maps candidate directions to concept words with a language model, and uses task-specific filtering and constrained prompting to turn those concepts into text. On five curated topics and 90 INFINITY-CHAT queries, the reported mean relative reductions in output Collision are 12.8% and 10.8%, respectively. We also instantiate the framework for generative engine optimization over four produ
    
[^42]: 基于大语言模型的贷款申请政策合规性验证

    LLM-Based Policy Compliance Verification of Loan Applications

    [https://arxiv.org/abs/2610.20155](https://arxiv.org/abs/2610.20155)

    该论文提出了一种基于大语言模型的两步式政策检查器原型，通过自动从政策文件中提取结构化合规规则，并据此对贷款申请进行合规性验证，实现了信贷审批合规审查流程的自动化。

    

    银行依靠庞大的内部政策框架来确保企业信贷决策符合监管要求和内部风险标准。对照这些政策审查贷款申请是一个主要依赖人工且知识密集型的过程，可能导致政策违规或例外情况被忽视。本报告提出了一个基于大语言模型（LLM）的政策检查器原型，能够根据内部信贷政策自动验证贷款申请。所提出的方法采用两步式架构：第一步，从政策文件中自动提取与合规相关的规则，并将其转换为结构化的、机器可读的表示形式；第二步，根据提取的规则对贷款申请进行评估，以识别政策例外情况、不合规发现、缺失信息以及需要人工审查的案例。评估结果表明，规则提取过程在多次运行中具有高度的稳定性。

    arXiv:2610.20155v1 Announce Type: cross  Abstract: Banks rely on extensive internal policy frameworks to ensure that corporate lending decisions comply with regulatory requirements and internal risk standards. Reviewing loan applications against these policies is a largely manual and knowledge-intensive process, which can lead to overlooked policy violations or exceptions. This report presents a prototype of an LLM-based policy checker that automatically verifies loan applications against internal lending policies.   The proposed approach follows a two-step architecture. First, compliance-relevant rules are automatically extracted from policy documents and converted into structured, machine-readable representations. Second, loan applications are evaluated against the extracted rules to identify policy exceptions, non-compliant findings, missing information, and cases requiring manual review.   Evaluation results demonstrate that the rule extraction process is highly stable across multi
    
[^43]: 面向自然语言处理机械可解释性的稀疏自编码器：关于可恢复性、可解释性与实用性的批判性综述

    Sparse Autoencoders for Mechanistic Interpretability in NLP: A Critical Survey of Recoverability, Interpretability, and Utility

    [https://arxiv.org/abs/2610.20139](https://arxiv.org/abs/2610.20139)

    本文对138项NLP领域的稀疏自编码器研究进行了批判性综述，提出了连接字典设计与NLP功能的任务导向分类体系，并系统评估了特征可恢复性、语义可解释性与行为效用方面的证据。

    

    稀疏自编码器（SAE）已成为一种有前景的机械可解释性方法，它通过将语言模型的激活分解为稀疏的、潜在可解释的特征来实现解释。然而，设计假设、评估目标和实验设置方面的差异使得对其有效性的判断变得复杂。我们对138项关于自然语言处理中稀疏自编码器的研究进行了批判性综述。我们的贡献有四个方面：（1）我们提出了一个以任务为基础的分类体系，将字典设计、特征操作与NLP功能联系起来；（2）我们考察了稀疏性分配、几何结构、数据覆盖和推理如何影响特征的可恢复性；（3）我们评估了关于语义可解释性、机制忠实性和行为效用方面的证据；（4）我们综合了多语言处理、检索、推理、分类和行为调控等领域的研究发现，强调了目标函数、基线方法以及叠加（superposition，原文此处截断）等因素的作用。

    arXiv:2610.20139v1 Announce Type: new  Abstract: Sparse autoencoders (SAEs) have emerged as a promising approach to mechanistic interpretability by decomposing language-model activations into sparse, potentially interpretable features. However, differences in design assumptions, evaluation objectives, and experimental settings complicate conclusions about their effectiveness. We present a critical survey of 138 studies on SAEs in natural language processing. Our contributions are fourfold: (1) we propose a task-grounded taxonomy connecting dictionary design, feature operations, and NLP functions; (2) we examine how sparsity allocation, geometry, data coverage, and inference affect feature recoverability; (3) we assess evidence for semantic interpretability, mechanistic faithfulness, and behavioral utility; and (4) we synthesize findings across multilingual processing, retrieval, reasoning, classification, and behavioral regulation, highlighting the roles of objectives, baselines, super
    
[^44]: 为什么要生成一次“点击”？面向网页智能体的即插即用决策头

    Why Generate a Click? Plug-in Decision Heads for Web Agents

    [https://arxiv.org/abs/2610.20107](https://arxiv.org/abs/2610.20107)

    该论文提出在冻结的视觉语言模型上外挂即插即用决策头，直接从隐藏状态中读取动作类型、目标元素和文本选项，并通过置信度级联在高置信时跳过文本生成，从而在不损失（甚至略提升）性能的前提下大幅加速网页智能体的决策。

    

    基于视觉语言模型构建的网页智能体通过生成文本来决定每一个动作，例如 click(12)，尽管这些文本的大部分内容只是在少数动作类型和几十个被标记元素之间做出的选择。我们直接从生成器在读取目标和屏幕截图时计算出的隐藏状态中读取这些选择。在冻结的生成器上添加即插即用的决策头，用于预测动作类型，对每个编号框内的视觉词元进行池化以选取目标，并指向列出的文本选项，或指向一个 WRITE 槽位——将撰写新文本的任务留给生成器。置信度级联机制在决策头有把握时执行其动作，否则让生成器从同一次前向传播的 KV 缓存中解码。使用在 21.3 万个网页步骤上训练的 Qwen3-VL 4B 和 8B 生成器，严格的阈值使级联方法在七个离线测试集上保持比生成器低 0.3 个百分点到高 1.4 个百分点之间的性能，同时决策头决定了其中 12% 到 64% 的动作。

    arXiv:2610.20107v1 Announce Type: new  Abstract: Web agents built on vision language models decide every action by generating text such as click(12), although most of that text encodes a choice among a few action types and a few dozen marked elements. We read these choices directly from the hidden states that the generator computes while it reads the goal and the screenshot. Plug-in decision heads on a frozen generator predict the action type, pool the visual tokens inside each numbered box to pick the target, and point at a listed text option or at a WRITE slot that leaves the writing of new text to the generator. A confidence cascade executes the heads' action when they are sure and otherwise lets the generator decode from the KV cache of the same forward pass. With Qwen3-VL 4B and 8B generators trained on 213k web steps, a strict threshold keeps the cascade between 0.3 points below and 1.4 points above the generator on seven offline test sets while the heads decide 12 to 64% of the 
    
[^45]: 语言暴露至关重要：多语言BabyLM预训练中的分布与顺序

    Language Exposure Matters: Distribution and Order in Multilingual BabyLM Pre-training

    [https://arxiv.org/abs/2610.20102](https://arxiv.org/abs/2610.20102)

    在数据有限的多语言BabyLM预训练中，语言暴露的数量与顺序均会影响性能，且全程混合的同步多语言训练在所有15个配置中都优于先单语后多语的课程式训练。

    

    我们研究了在数据有限条件下多语言预训练的两个方面，即语言暴露如何在英语、荷兰语和中文之间分布，以及在训练过程中语言暴露的顺序安排。特别地，我们比较了平衡的多语言暴露与语言主导型暴露（即某一种语言的训练数据多于其他语言）。我们还比较了混合训练（在整个训练过程中持续使用多语言训练数据）与课程训练（先进行主导语言的单语暴露，随后进行平衡的多语言暴露）。我们的结果表明，多语言BabyLM的性能随语言暴露的数量和顺序而变化。在全部15个匹配配置中，混合训练均优于课程学习，这表明在我们的设置下，同步的多语言暴露比分阶段暴露更为有效。若干语言主导型的混合训练配置取得了（摘要在此处不完整）

    arXiv:2610.20102v1 Announce Type: new  Abstract: We investigate two aspects of multilingual pre-training with limited data, namely how language exposure is distributed across English, Dutch, and Chinese, and how that exposure is ordered during training. In particular, we compare balanced multilingual exposure to language-dominant exposure, where one language is exposed with more training data than the others. We also compare mixed training, where multilingual training data is used throughout training, with curriculum training, where monolingual exposure of the dominant language is followed by balanced multilingual exposure. Our results show that multilingual BabyLM performance varies with both the amount and order of language exposure. Mixed training outperforms curriculum learning across all 15 matched configurations, indicating that simultaneous multilingual exposure is more effective than staged exposure in our setting. Several language-dominant mixed training configurations achieve
    
[^46]: AFCS：面向事实知识预训练的原子事实条件化合成

    AFCS: Atomic-Fact-Conditioned Synthesis for Factual Knowledge Pretraining

    [https://arxiv.org/abs/2610.20086](https://arxiv.org/abs/2610.20086)

    AFCS 提出先从源文档抽取有据可依的原子事实、再以每条事实驱动多格式语料生成的方法，在 OLMo-3-7B 与 32B 模型的继续预训练实验中事实性优于 WRAP 和 Nemotron-CC。

    

    合成数据在语言模型预训练中的应用日益增多。现有的知识语料库合成流程，包括 WRAP 和 Nemotron-CC，通常以源段落或文档片段为输入，从该文本生成改写或问答式样本。这种方式可以提升表层形式和提示的多样性，但并未明确要求细粒度的源事实——尤其是次要实体、列表/表格条目和精确的时间属性——出现在合成语料库中。我们提出了 AFCS（原子事实条件化合成），一种先抽取源文本支持的原子事实、再利用每条保留的事实来驱动生成的流程。AFCS 采用了基于文档树的结构感知抽取、源文本验证、局部上下文扩展以及多格式生成。在使用 OLMo-3-7B 和一个 32B 基础模型、基于同一 FineWiki 源语料库的继续预训练实验中，AFCS 的表现优于 WRAP 和 Nemotron-CC。

    arXiv:2610.20086v1 Announce Type: new  Abstract: Synthetic data is increasingly used in language-model pretraining. Existing knowledge-corpus synthesis pipelines, including WRAP and Nemotron-CC, usually take a source passage or document segment as input and generate rewritten or QA-style samples from that text. This can improve surface form and prompt diversity, but it does not explicitly require fine-grained source facts -- especially secondary entities, list/table entries, and precise temporal attributes -- to appear in the synthesized corpus. We propose AFCS (Atomic-Fact-Conditioned Synthesis), a pipeline that first extracts source-supported atomic facts and then uses each retained fact to drive generation. AFCS uses structure-aware extraction over the document tree, source-grounded verification, local context expansion, and multi-format generation. In continued-pretraining experiments with OLMo-3-7B and a 32B base on the same FineWiki source corpus, AFCS outperforms WRAP and Nemotr
    
[^47]: 设计文本接口以增强大语言模型在视听问答任务上的推理能力

    Designing Textual Interfaces To Enhance Llm Reasoning On Audio-Visual Question Answering

    [https://arxiv.org/abs/2610.20085](https://arxiv.org/abs/2610.20085)

    本研究探讨了文本接口设计对多模态推理系统中自动提示优化的影响，发现全模态LLM直接优化收益依赖具体模型，而描述器-推理器和智能体架构虽提供更可解释的优化接口却更难优化。

    

    自动提示优化（APO）已在基于文本的大语言模型（LLM）及LLM系统中展现出强大的有效性，但将其应用于视听理解系统需要合适的文本接口，以便通过这些接口将多模态推理呈现给基于文本的优化器。我们以视听问答任务为研究对象，探讨了文本接口的设计如何影响多模态推理的提示优化。我们比较了三种接口设计，包括由单个全模态大语言模型（OmniLLM）构成的直接推理系统、描述器-推理器系统，以及一种由主智能体协调多个子智能体OmniLLM的智能体架构。基于Qwen3-Omni和Gemini的实验表明，直接提示优化带来的收益依赖于具体模型；描述器-推理器系统和智能体系统暴露出更具可解释性的优化接口，但更难进行优化，对相关案例的错误分析显示……

    arXiv:2610.20085v1 Announce Type: cross  Abstract: Automatic prompt optimization (APO) has shown strong effectiveness for text-based LLMs and LLM systems, but its application to audio-visual understanding systems requires suitable textual interfaces through which multimodal reasoning can be exposed to text-based optimizers. We investigate how the design of textual interfaces affects prompt optimization for multimodal reasoning, on the audio-visual question answering task. We compare three interface designs, including a direct inference system consisting of a single omni-modal large language model (OmniLLM), a captioner-reasoner system, and an agentic architecture in which a primary agent orchestrates subagent OmniLLMs. Experiments with Qwen3-Omni and Gemini show that direct prompt optimization provides model-dependent gains; the captioner-reasoner system and the agentic system expose more interpretable optimization interfaces but are harder to optimize, where error analysis of cases af
    
[^48]: 大语言模型应该信任哪种知识？揭示跨知识冲突的可泛化操控方向

    Which Knowledge Should LLMs Trust? Uncovering a Generalizable Steering Direction Across Knowledge Conflicts

    [https://arxiv.org/abs/2610.20067](https://arxiv.org/abs/2610.20067)

    研究发现大语言模型在参数化知识与上下文知识之间的选择沿残差流中单一方向线性表示，且该方向可跨任务泛化，通过操控该方向能有效改善模型在RAG、记忆和规则遵循任务中对提示信息的利用。

    

    大语言模型（LLMs）经常遇到内部参数化知识与外部上下文知识之间的冲突，然而控制这种选择的机制仍不清楚。我们证明了在上下文与参数化知识之间的选择是线性表示的，并且可以沿着模型残差流中的单一方向进行控制。我们首先证明了这一方向能够在各种事实冲突设置中泛化，包括不同的上下文格式、知识类别和数据集，同时持续地在参数化答案和上下文答案之间转换模型行为。然后，我们将从简单事实冲突中学习到的方向迁移到三个更复杂的任务族：RAG（检索增强生成）、记忆和规则遵循，在这些任务中，模型必须从检索到的文档、交互历史或显式规则中获取信息。在四个模型族和多个迁移设置中，操控该方向持续地改善了对提示所提供信息

    arXiv:2610.20067v1 Announce Type: new  Abstract: Large Language Models (LLMs) frequently encounter conflicts between internal parametric knowledge and external contextual knowledge, yet the mechanisms governing this selection remain unclear. We show that the choice between context and parametric knowledge is linearly represented and can be controlled along a single direction in the model's residual stream. We first demonstrate that this direction generalizes across factual-conflict settings, including context formats, knowledge categories, and datasets, while consistently shifting behavior between parametric and contextual answers. We then transfer a direction learned from simple factual conflicts to three more complex task families: RAG, memory, and rule following, where models must acquire information from retrieved documents, interaction histories, or explicit rules. Across four model families and multiple transfer settings, steering consistently improves the use of prompt-provided 
    
[^49]: 什么是澄清？估计依赖于模型的输入歧义

    What Is a Clarification? Estimating Model-Dependent Input Ambiguity

    [https://arxiv.org/abs/2610.20052](https://arxiv.org/abs/2610.20052)

    该论文区分了输入固有的“世界歧义”与语言模型实际可感知的“模型歧义”，通过唯一性、忠实性和答案等价性三个标准形式化自然语言澄清，证明了有效答案与澄清等价类之间的一一对应关系，并据此提出一种黑盒方法来估计模型相关的输入歧义。

    

    arXiv:2610.20052v1 公告类型：新 摘要：模糊的输入可能允许多种解释，每种解释都可能产生不同的答案。现有工作大多将这种模糊性视为输入本身的属性，然而对于特定的语言模型而言，只有部分备选解释可能是其已知且可回答的。因此，我们区分了世界歧义——即捕捉一个输入所允许的所有备选解释——与模型歧义——即捕捉响应模型所能获取的那些备选解释。为了刻画这些备选解释，我们通过三个理想标准将自然语言澄清形式化：唯一性、忠实性和答案等价性。我们证明这些理想标准在有效答案与澄清的等价类之间建立了一一对应关系，这意味着任何解释既可以由其答案来表示，也可以由唯一表达该解释的澄清来表示。我们利用这种对应关系提出了○clar，一种用于估计（模型相关输入歧义的）黑盒方法……

    arXiv:2610.20052v1 Announce Type: new  Abstract: Ambiguous inputs can admit multiple interpretations, each potentially yielding a different answer. Existing work largely treats this ambiguity as a property of the input, yet for a particular language model, only some alternatives may be known and answerable. We therefore distinguish world ambiguity, capturing all alternatives admitted by an input, from model ambiguity, capturing those accessible to the responding model. To characterize these alternatives, we formalize natural-language clarifications through three desiderata: \emph{uniqueness}, \emph{faithfulness}, and \emph{answer equivalence}. We show that these desiderata induce a one-to-one correspondence between valid answers and equivalence classes of clarifications, implying that any interpretation can be represented either by its answer or by the clarifications that uniquely express it. We leverage this correspondence to introduce \circleclar{}, a black-box method that estimates 
    
[^50]: FACES：面向多模态情感支持对话的细粒度门控语义对齐框架

    FACES: Fine-grained Gated Semantic Alignment for Multimodal Emotional Support Conversation

    [https://arxiv.org/abs/2610.20051](https://arxiv.org/abs/2610.20051)

    FACES提出了一种细粒度门控语义对齐框架，通过多源奖励整合、非补偿性门控融合机制以及基于结构化思维链锚点的token级信用分配，有效解决了多模态情感支持对话中回复浅显和支持策略不当的问题。

    

    近年来，随着多模态交互技术的进步以及对智能情感支持需求的日益增长，多模态情感支持对话已成为一项重要任务。然而，由于语义反馈薄弱、跨维度奖励补偿以及推理监督粗糙等问题，现有系统仍然存在回复浅显和支持策略不当的缺陷。为应对这些挑战，我们提出了FACES，一个面向多模态情感支持对话的细粒度门控语义对齐框架。具体而言，FACES整合多源奖励以构建多维度的语义和功能对齐信号，采用非补偿性门控融合机制来减少跨维度奖励偏移，并利用结构化思维链锚点来执行token级别的信用分配。在受控回复长度协议下，FACES将基于多模态大语言模型的平均评估分数从3.61提升至4.04。

    arXiv:2610.20051v1 Announce Type: new  Abstract: In recent years, with advances in multimodal interaction technologies and the growing demand for intelligent emotional support, Multimodal Emotional Support Conversation (MESC) has emerged as an important task. However, existing systems still suffer from shallow responses and inappropriate support strategies due to weak semantic feedback, cross-dimension reward compensation, and coarse reasoning supervision. To address these challenges, we propose FACES, a fine-grained gated semantic alignment framework for MESC. Specifically, FACES integrates multi-source rewards to construct multidimensional semantic and functional alignment signals, employs a non-compensatory gated fusion mechanism to reduce cross-dimension reward offset, and leverages structured Chain-of-Thought (CoT) anchors to perform token-level credit assignment. Under a controlled response-length protocol, FACES improves the average MLLM-based evaluation score from 3.61 to 4.04 
    
[^51]: LintEval：基于Linter引导的文件重构对大语言模型进行基准评测

    LintEval: Benchmarking LLMs on Linter-Guided File Refactoring

    [https://arxiv.org/abs/2610.20027](https://arxiv.org/abs/2610.20027)

    该论文提出LintEval，一个覆盖C++、Python和Java的多语言基准，用于评测大语言模型在linter引导的文件重构中减少lint告警的能力，实验揭示了告警减少与功能保持之间的持续权衡。

    

    Linter（代码检查工具）被广泛应用于代码维护和IDE实时编辑中，用于揭示源文件中与性能、风格、易错模式等相关的lint告警。然而，现有的代码基准测试并未显式评估linter引导的文件重构任务，即模型必须在保持源文件原有功能正确性的前提下减少lint告警。我们提出了LintEval，一个面向C++、Python和Java的linter引导文件重构多语言基准，包含13,061个初始功能正确的程序以及通过语言特定linter提取的171,662条lint告警。LintEval联合评估功能保持、lint告警减少、新告警规避以及编辑最小化。实验揭示了一个持续存在的权衡：闭源商业模型的Reduction@P最高可达90.4%，但Pass@1仅为67.1%–68.3%，而若干开源小模型则表现出较低的……

    arXiv:2610.20027v1 Announce Type: cross  Abstract: Linters are widely used in code maintenance and real-time IDE editing to surface lint alerts related to performance, style, error-prone, and other patterns in source files. However, existing code benchmarks do not explicitly evaluate linter-guided file refactoring, where models must reduce lint alerts in source files with correct initial functionality while satisfying functionality preservation. We introduce LintEval, a multilingual benchmark for linter-guided file refactoring across C++, Python, and Java, comprising 13,061 programs with correct initial functionality and 171,662 lint alerts extracted using language-specific linters. LintEval jointly evaluates functionality preservation, lint alert reduction, new alert avoidance, and edit minimality. Experiments reveal a persistent trade-off: closed-source commercial models reach up to 90.4% Reduction@P but achieve only 67.1%-68.3% Pass@1, while several open-source small models show low
    
[^52]: 强大的执行框架缩小模型差距：立法文档转XML转换的比较评估

    A Strong Harness Narrows the Model Gap: A Comparative Evaluation for Legislative Document-to-XML Conversion

    [https://arxiv.org/abs/2610.19999](https://arxiv.org/abs/2610.19999)

    在立法文档转XML的任务中，精心设计的确定性执行框架（尤其是多代理分解式编排）比模型本身更重要——流水线中最弱的模型表现超过了单独使用的最强模型。

    

    法律内容的复现要求在章节层级、逐字文本、引用和修订条文等方面与原始文本完全一致；一个被遗漏的引用或错误标注的修订条文会传递到律师所依赖的产品中。我们报告了一条将美国州立法转换为严格的模式约束XML的工业流水线，并假设其准确性更多来自框架工程——即围绕模型构建的确定性工具——而非模型本身。我们在来自两个司法辖区的153份金标法案上进行了受控消融实验，将三种框架（零样本、单一工具使用代理、以及分解式多代理流水线）与三个Claude模型交叉组合。实验表明框架缩小了模型差距：完整流水线中最弱的模型击败了单独使用的最强模型（0.948对0.806，配对bootstrap p<0.001）。由于两个代理式框架共享完全相同的工具，它们之间的差距来自编排方式：任务分解加上持久化的状（原文此处截断）。

    arXiv:2610.19999v1 Announce Type: new  Abstract: Legal-content reproduction requires matching the source exactly in section hierarchy, verbatim text, citations, and amendment spans; a dropped citation or mislabeled amendment carries into the products lawyers rely on. We report an industrial pipeline converting U.S. state legislation into strict, schema-constrained XML, and hypothesize that its accuracy comes from harness engineering, the deterministic tooling built around a model, more than the model itself. Our controlled ablation crosses three harnesses (zero-shot, a single tool-using agent, and a decomposed multi-agent pipeline) with three Claude models on 153 gold bills from two jurisdictions. The harness narrows the model gap: the weakest model in the full pipeline beats the strongest model alone (0.948 versus 0.806, paired bootstrap p less than 0.001). Because the two agentic harnesses share identical tools, their gap comes from orchestration: decomposition plus the persisted sta
    
[^53]: 基于词元身份的大语言模型灰盒幻觉检测

    Gray-Box Hallucination Detection in LLMs with Token Identities

    [https://arxiv.org/abs/2610.19996](https://arxiv.org/abs/2610.19996)

    本文提出GURI，一种轻量级灰盒幻觉检测器，通过将候选词元概率与词元身份向量标识符相结合，不仅能衡量模型的不确定程度，还能识别模型不确定的具体对象，从而实现更精细的大语言模型幻觉检测。

    

    大语言模型经常产生流畅但事实上错误的输出。灰盒检测器利用生成过程中输出的词元概率统计信息来识别此类幻觉，从而避免了对隐藏状态进行白盒访问的需求，而前沿模型并不提供这种访问。然而，这些检测器通常仅依赖于由概率派生的表示，而忽略了候选词元的身份信息。因此，它们只能观察到模型有多不确定，却无法观察到模型对什么不确定。 consequently，当“big”与“small”这一对词和“big”与“huge”这一对词具有相同的概率分布时，它们无法区分模型究竟是在“大”和“小”之间犹豫，还是在“大”和“巨大”之间犹豫。我们提出了GURI（Gray-box Uncertainty Representations with Identity，带身份的灰盒不确定性表示），这是一种轻量级的灰盒幻觉检测器，它将候选词元概率与每个词元的向量标识符相结合。这使得模型正在权衡的备选词元得以显现……（摘要不完整，原文截断）

    arXiv:2610.19996v1 Announce Type: new  Abstract: Large language models often produce fluent but factually incorrect output. Gray-box detectors identify such hallucinations from the token-probability statistics emitted during generation, avoiding the need for white-box access to hidden states that frontier models do not provide. These detectors, however, typically rely only on probability-derived representations and ignore the identities of the candidate tokens. They can therefore observe how uncertain the model is, but not what it is uncertain about. Consequently, they cannot distinguish between a model that is torn between "big" and "small" and one that is torn between "big" and "huge" when the two pairs receive the same probability distribution.   We introduce GURI (Gray-box Uncertainty Representations with Identity), a lightweight gray-box hallucination detector that combines candidate-token probabilities with per-token vector identifiers. This exposes which alternatives the model c
    
[^54]: SoK：开放式法律文本答案的评分——从法学院考试到可信赖的评分规则裁判

    SoK: Grading Open-Text Legal Answers, from Law School Exams to Trustworthy Rubric Judges

    [https://arxiv.org/abs/2610.19992](https://arxiv.org/abs/2610.19992)

    本文系统化研究了开放式法律文本答案（如法学院考试）的评估方法，从教育测量与法律规范中提炼出八项评估标准并对比八种评估范式，发现针对任务定制评分规则是让LLM裁判与人类评分者达成一致的最关键手段。

    

    法学学生通过开放式文本答案接受考核：案例分析、论文和法律备忘录，将法律适用于一组特定事实。这类答案的评分成本高昂，人类评分者之间分歧很大，而大型语言模型（LLM）如今可以在教学中以及测试法律AI的基准评测中大规模进行评分。本文系统化地研究了如何评估开放式法律文本答案。我们从教育测量学以及规范考试的法律中推导出八项必要条件，其中包括效度、信度、说理依据、对可辩护的替代解答给予得分、对投机取巧行为的鲁棒性以及人类监督。随后，我们将八种评估范式与这些标准进行对比，范围涵盖从非结构化的人工评分，到采用针对具体实例的评分规则的LLM裁判，再到成对比较。有三个发现尤为突出。为任务量身编写的评分规则是使LLM裁判与人类评分者达成一致的最强杠杆，但所报告的数字回答的是不同的问题……（原文摘要至此截断）

    arXiv:2610.19992v1 Announce Type: new  Abstract: Law students are examined through open-text answers: case solutions, essays and memos that apply the law to a set of facts. Such answers are expensive to grade, human graders disagree widely on them, and large language models (LLMs) now offer to grade them at scale, both in teaching and in the benchmarks that test legal AI. This paper systematizes how open-text legal answers can be evaluated. We derive eight desiderata from educational measurement and from the law that governs exams, among them validity, reliability, reasons, credit for defensible alternative solutions, robustness to gaming and human oversight. We then compare eight evaluation paradigms against them, from unstructured human grading to LLM judges with instance-specific rubrics and pairwise comparison. Three findings stand out. A rubric written for the task is the strongest lever for agreement between an LLM judge and human graders, but the reported figures answer differen
    
[^55]: 从内在维度选择注意力头数量：一项小规模研究

    Choosing the Number of Attention Heads from Intrinsic Dimension: A Small-Scale Study

    [https://arxiv.org/abs/2610.19966](https://arxiv.org/abs/2610.19966)

    本文提出一种无需标签的启发式方法，通过嵌套Dropout VAE估计词元表示的内在维度来预先确定transformer注意力头的数量，实验表明按内在维度设置约10个注意力头的模型与32个头性能统计等效，且注意力参数减少69%。

    

    Transformer中注意力头的数量通常由惯例或网格搜索来确定。我们测试了一种简单的启发式方法来预先设置这一数量：在无标签的情况下估计词元表示的内在维度，并使用大致相同数量的注意力头。我们在WikiText-2的DistilBERT嵌入上训练了一个嵌套Dropout变分自编码器（VAE）来估计内在维度，并在VAE率失真曲线的拐点处选择KL权重，而非手动选择。由此得到k*=9.2（95%置信区间[8.6, 9.8]）。在一个从零训练的小型4层transformer中，10个注意力头与32个注意力头在SST-2和MRPC任务上统计等效（10个随机种子的配对TOST检验，±1分容差），而10头模型的注意力参数减少了69%。从一开始就训练10个头也略优于先训练32个头再剪枝到10个头的做法（0.6分，p=0.008）。需要说明的是，证据范围较为有限：仅涉及两个分类任务、一种小型架构、未使用预训练模型。

    arXiv:2610.19966v1 Announce Type: new  Abstract: The number of attention heads in a transformer is usually set by convention or grid search. We test a simple heuristic for setting it in advance: estimate the intrinsic dimension of token representations without labels, and use roughly that many heads. We estimate intrinsic dimension with a nested-dropout VAE trained on DistilBERT embeddings of WikiText-2, choosing the KL weight at the knee of the VAE's rate-distortion curve rather than by hand. This gives $k^*=9.2$ (95% CI [8.6, 9.8]). In a small 4-layer transformer trained from scratch, 10 heads and 32 heads are statistically equivalent on SST-2 and on MRPC (paired TOST over 10 seeds, $\pm1$ point margin), with 69% fewer attention parameters for the 10-head model. Training 10 heads from the start also slightly outperforms training 32 heads and pruning to 10 (0.6 points, $p=0.008$). The evidence is narrow: two classification tasks, one small architecture, no pretrained models, and no fo
    
[^56]: AtomSearch：面向智能体技能优化的基于证据的树搜索

    AtomSearch: Evidence-Grounded Tree Search for Agent Skill Optimization

    [https://arxiv.org/abs/2610.19951](https://arxiv.org/abs/2610.19951)

    AtomSearch 通过对比式原子编辑和技能状态树搜索，以证据为依据实现局部化、可回溯的智能体技能优化，在多个任务和模型上超越了 SkillOpt。

    

    智能体技能通过可编辑的外部指令来适配冻结的模型。一次性修改多条规则会使改动难以审查，而仅凭即时得分来判断每次局部修改可能会丢弃有用的中间状态。我们提出 AtomSearch，它保持修改动作的局部性，同时利用后续的技能状态来引导搜索。对比式原子编辑将每个原始规则修改与失败和成功的执行、一个行为假设以及预期作用范围相关联。技能状态树搜索（STS）评估较短的编辑组合，并选择性地将其保留为可回访、可扩展的可执行状态。结果记忆为后续的修改提案提供已观察到的修复与回归统计数据。我们在四个任务和五个目标模型上评估了 AtomSearch。使用 deepseek-v4-flash 时，它在四个任务中的三个上超越了 SkillOpt，其中包括在 LiveMathematicianBench 上取得 9.68 个百分点的提升，以及 SkillOpt 的最终（摘要在此处截断）

    arXiv:2610.19951v1 Announce Type: new  Abstract: Agent skills adapt frozen models through editable external instructions. Revising several rules at once makes changes difficult to inspect, while judging each local edit only by its immediate score can discard useful intermediate states. We introduce AtomSearch, which keeps revision actions local while using subsequent skill states to guide search. Contrastive atomic editing links each primitive rule edit to failed and successful executions, a behavioral hypothesis, and an intended scope. Skill-State Tree Search (STS) evaluates short edit compositions and selectively retains them as executable states that can be revisited and extended. Outcome memory supplies later proposals with observed repair and regression statistics. We evaluate AtomSearch across four tasks and five target models. With deepseek-v4-flash, it exceeds SkillOpt on three of four tasks, including a 9.68-percentage-point gain on LiveMathematicianBench, and SkillOpt's final
    
[^57]: 机器翻译能否保留媒体偏见？

    Does Machine Translation Retain Media Bias?

    [https://arxiv.org/abs/2610.19945](https://arxiv.org/abs/2610.19945)

    本研究考察了机器翻译（英/德语译入保加利亚语）对27类媒体偏见的保留情况，发现翻译过程无法完全保留源文本中的新闻偏见。

    

    在最近二十年中，机器翻译（MT）已经从一个研究领域发展成为一项成熟的、已实际部署应用的散文文本技术，适用于大多数语言对。然而，语言具有不同维度的含义，而上述进展主要体现在命题意义层面上。在这项小型探索性研究中，我们考察了一类情感含义——即以语言形式表达的各类媒体偏见——在自动翻译过程中是否被保留。研究涉及两个语言对：以英语和德语作为源语言，保加利亚语作为目标语言。我们评估了四种最先进的翻译方法，以回答以下问题：应用机器翻译是否会导致非中性语言（具体而言，27种媒体偏见类型）的丢失，以及丢失的程度如何。我们的实证结果与以下论断相符：机器翻译并不能完全保留输入源文本中已有的新闻偏见。

    arXiv:2610.19945v1 Announce Type: new  Abstract: Within the most recent two decades, machine translation (MT) has been advanced from a research area to a mature, deployed technology for prose text and most language pairs. However, language has different dimensions of meaning, and the alluded progress has primarily been demonstrated for propositional meaning.   In this small exploratory study, we investigate whether one type of affective meaning, namely different types of linguistically expressed media bias, are retained under automatic translation for two language pairs, English and German as the source languages, and Bulgarian as the target language. We evaluate four state-of-the-art translation methods to answer the question whether and to what degree a loss of non-neutral language (specifically, 27 types of media bias) can occur as a result of applying MT.   Our empirical findings are consistent with the claim that MT does not fully retain existing news bias of its input source text
    
[^58]: 能推理却不能停止：理解并恢复被剪枝的大型推理模型

    To Reason but Not to Stop: Understanding and Recovering Pruned Large Reasoning Models

    [https://arxiv.org/abs/2610.19934](https://arxiv.org/abs/2610.19934)

    该研究揭示剪枝后大型推理模型的性能下降主要源于得到正确答案后无法停止生成而非推理能力受损，并提出免训练的终止引导激活导向方法（TGAS），利用与确定答案相关的隐状态共享方向在推理时引导模型及时终止，从而恢复剪枝模型的准确性。

    

    权重剪枝会降低大型推理模型（LRMs）的准确性，但目前尚不清楚剪枝是否损害了推理能力本身。我们发现，这种性能下降很大程度上源于模型无法停止，而非无法推理。随着稀疏度的增加，被剪枝的LRM越来越难以在token预算内终止生成——这往往发生在正确答案已经出现在其推理过程之后——反而陷入重复循环，尽管它们仍然能够识别稠密模型会在何处停止。我们进一步发现，终止性延续与循环性延续之间的隐状态差异在不同剪枝检查点上保持一致，这表明存在一个与“确定最终答案”相关的共享方向。基于这一发现，我们提出了终止引导激活导向，这是一种免训练方法，在推理时将该方向添加到残差流中。在六个LRM、三种剪枝流程以及数学和代码基准测试上，TGAS显著提升了模型的终止能力。

    arXiv:2610.19934v1 Announce Type: new  Abstract: Weight pruning can degrade the accuracy of large reasoning models (LRMs), yet it remains unclear whether it damages reasoning itself. We find that much of this degradation stems from a failure to stop rather than to reason. As sparsity increases, pruned LRMs increasingly fail to terminate within the token budget, often after the correct answer has appeared in their reasoning, instead falling into repetitive loops, even though they still recognize where the dense model would stop. We further find that the hidden-state difference between terminating and looping continuations is consistent across pruned checkpoints, suggesting a shared direction associated with committing to an answer. Building on this, we propose Termination-Guided Activation Steering (TGAS), a training-free method that adds this direction to the residual stream at inference time. Across six LRMs, three pruning pipelines, and math and code benchmarks, TGAS improves termina
    
[^59]: MemDuet：学习双过程记忆以实现长上下文推理

    MemDuet: Learning Dual-Process Memory for Long-Context Reasoning

    [https://arxiv.org/abs/2610.19928](https://arxiv.org/abs/2610.19928)

    提出受卡尼曼快慢思考启发的双过程记忆框架MemDuet，通过小语言模型快速并行选择证据、大语言模型验证并按文档顺序更新记忆的三阶段联合训练，实现长上下文推理中性能与效率的平衡。

    

    随着循环记忆智能体的发展，对无限长文档进行推理正在成为现实。然而，上下文长度的增加加剧了推理性能与推理效率之间的权衡。受卡尼曼（Kahneman）关于快思考与慢思考理论的启发，我们提出了MemDuet，一个双过程记忆框架，它学习协调一个小语言模型用于从原始上下文块中快速、并行地选择证据（系统1），以及一个大语言模型用于按文档顺序进行记忆更新验证和记忆更新（系统2）。我们分三个阶段训练MemDuet：对系统1进行证据选择的监督微调；对系统2进行验证与记忆重写的组相对策略优化（GRPO）；以及对两个策略进行联合优化，以协调长上下文推理中的选择与更新。在多样化长上下文推理基准（RULER-HQA、LongBench v2和2WikiMultiHopQA）上的实验表

    arXiv:2610.19928v1 Announce Type: new  Abstract: Reasoning over infinitely long documents is becoming a reality with recurrent memory agents. However, increasing context length exacerbates the trade-off between reasoning performance and inference efficiency. Inspired by Kahneman's account of fast and slow thinking, we propose MemDuet, a dual-process memory framework that learns to coordinate a small language model for fast, parallel evidence selection from raw context chunks (System 1) with a large language model for memory-update verification and memory updates in document order (System 2). We train MemDuet in three stages: supervised fine-tuning of System 1 for evidence selection, Group Relative Policy Optimization (GRPO) of System 2 for verification and memory rewriting, and joint optimization of both policies to coordinate selection and updating for long-context reasoning. Experiments on diverse long-context reasoning benchmarks (RULER-HQA, LongBench v2, and 2WikiMultiHopQA) demons
    
[^60]: MolHop：基于智能体强化学习的骨架跃迁分子设计

    MolHop: Scaffold-Hopping Molecular Design with Agentic Reinforcement Learning

    [https://arxiv.org/abs/2610.19920](https://arxiv.org/abs/2610.19920)

    MolHop 提出了一个基于智能体强化学习的跨骨架分子发现框架，将约 160 万个商业分子骨架组织成从粗到细的分层动作空间，使模型能够发现骨架完全不同但保留参考分子功能特性的新分子。

    

    分子设计通常被表述为在现有分子空间上的优化问题。然而，分子优化与发现结构上截然不同的替代分子代表了两个本质不同的搜索问题。基于大语言模型（LLM）的分子设计智能体通常执行前者，即在现有结构的邻域内进行局部细化，通过结构修饰来改善目标性质。相比之下，后者需要探索更广泛的化学空间，以发现具有完全不同骨架同时保留参考分子功能特性的分子。我们提出了 MolHop，一个用于跨骨架分子发现的智能体框架，使模型能够在预先组织好的化学空间中进行导航和编辑。MolHop 将来自商业分子的约 160 万个分子骨架组织成一个从粗到细的分层动作空间。

    arXiv:2610.19920v1 Announce Type: cross  Abstract: Molecular design is often formulated as an optimization problem over existing molecular spaces. However, molecular optimization and the discovery of structurally distinct replacement molecules represent two fundamentally different search problems. LLM-based molecular design agents typically perform the former, conducting local refinement within the neighborhood of existing structures to improve target properties through structural modifications. In contrast, the latter requires exploration of a broader chemical space to discover molecules with entirely different scaffolds while preserving the functional characteristics of reference molecules. We propose MolHop, an agentic framework for cross-scaffold molecular discovery that enables models to navigate and edit within a pre-organized chemical space. MolHop organizes approximately 1.6 million molecular scaffolds from commercial molecules into a coarse-to-fine hierarchical action space. T
    
[^61]: 并非所有标准都同等重要：面向基于评分标准的LLM评估的重要性感知标准加权

    Not All Criteria Matter: Importance-Aware Criteria Weighting for Rubric-Based LLM Evaluation

    [https://arxiv.org/abs/2610.19915](https://arxiv.org/abs/2610.19915)

    提出了重要性感知标准加权方法ICW，将LLM对所有标准的整体权重判断转化为易于完成的两两重要性比较，并通过检查修正不一致的比较结果、从主特征向量推导权重，从而提升基于评分标准的LLM评估的准确性与可靠性。

    

    基于评分标准的评估通过聚合显式标准的验证结果来评估大语言模型（LLM）的回答，但并非所有标准都同等重要。然而，现有方法通常采用均匀权重来聚合验证结果，这使得次要和重叠的标准与关键标准获得相同的权重，从而产生具有误导性的评估分数。尽管一些方法通过提示LLM直接分配权重，但直接加权需要对所有标准进行整体性判断，而LLM难以可靠地完成这种判断。为了解决这些局限性，我们提出了重要性感知标准加权，该方法从LLM进行的两两重要性比较中推导出各标准的权重。具体而言，ICW将对所有标准的整体性判断转化为一系列两两重要性比较，检查并修正不一致的比较结果，并从主特征向量中推导出各标准的权重。

    arXiv:2610.19915v1 Announce Type: new  Abstract: Rubric-based evaluation assesses the responses of large language models (LLMs) by aggregating the verification results of explicit criteria, yet not all criteria matter equally. However, existing methods commonly aggregate the verification results with uniform weighting, which assigns minor and overlapping criteria the same weight as essential ones and thus produces misleading evaluation scores. Although some methods prompt an LLM to directly assign weights, direct weighting requires a holistic judgment over all criteria that LLMs struggle to make reliably. To address these limitations, we propose Importance-Aware Criteria Weighting (ICW), which derives the weights of criteria from pairwise importance comparisons made by an LLM. Specifically, ICW transforms the holistic judgment over all criteria into a series of pairwise importance comparisons, checks and revises inconsistent comparisons, and derives the weights from the principal eigen
    
[^62]: 超越基准准确率：大语言模型化学预测的诊断性测试

    Beyond Benchmark Accuracy: A Diagnostic Test for LLM Chemical Prediction

    [https://arxiv.org/abs/2610.19906](https://arxiv.org/abs/2610.19906)

    该论文提出身份遮蔽推理诊断框架，通过隐藏目标实体身份、以已知源实体加显式结构编辑操作来定义目标，从而区分大语言模型的化学预测能力究竟来自真正的结构-性质推理，还是来自上下文信息或记忆检索。

    

    大语言模型（LLM）在化学预测基准测试中已取得出色表现。然而，现有的评估方法通常直接提供目标实体的身份信息，这使得我们难以判断模型的预测究竟源自可迁移的结构-性质推理能力，还是依赖于与目标相关的上下文信息或记忆检索。我们提出了身份遮蔽推理诊断框架，用于在目标实体身份信息被隐藏的情况下评估化学预测模型。在该框架中，目标不再由其实体身份直接定义，而是由一个已知的源实体和一个明确的结构编辑操作来定义。这一设置在保留确定目标所需信息的同时，减少了模型获取目标身份线索的机会。我们在涵盖气味预测、分子性质预测、逆合成等六项任务上对该框架进行了评估。

    arXiv:2610.19906v1 Announce Type: new  Abstract: LLMs have achieved strong performance on chemical prediction benchmarks. However, existing evaluation methods typically provide target entity identity information directly, making it difficult to determine whether model predictions arise from transferable structure-property reasoning capabilities or rely on target-related contextual information or memory retrieval. We propose the Identity-Blinded Reasoning Diagnostic framework to evaluate chemical prediction models when target entity identity information is hidden. In this framework, the target is no longer directly defined by its entity identity, but instead by a known source entity and an explicit structural edit operation. This setting preserves the information required to determine the target while reducing the opportunity for models to access target identity cues. We evaluate this framework across six tasks spanning odor prediction, molecular property prediction, retrosynthesis, and
    
[^63]: 基于长度条件化的简单得令人惊讶的多数投票方法

    Frustratingly Easy Majority Voting Using Length Conditioning

    [https://arxiv.org/abs/2610.19890](https://arxiv.org/abs/2610.19890)

    提出长度条件化多数投票（LCMV），利用“较长的思维链正确概率更低”这一经验观察，通过逻辑函数建模长度条件化先验，并仅在测试时使用无标签数据自适应调整该先验来加权各答案的投票，无需微调LLM即可显著提升推理准确性。

    

    多数投票（MV）是一种简单且广泛使用的策略，通过分配额外的推理时计算来提高大语言模型（LLM）的推理准确性。具体而言，在多数投票中，独立采样多个思维链（CoT），并选择出现频率最高的最终答案。在这项工作中，我们提出了长度条件化多数投票（LCMV），这是一种测试时训练方法，它利用了近期多项研究所确立的一个实证观察：较长的思维链往往与较低的正确概率相关。我们将这种归纳偏置表述为长度条件化的先验概率，该先验概率由一个简单的逻辑斯蒂函数建模。此外，我们仅使用未标记的测试数据在测试时自适应地调整这一先验，而无需对LLM进行微调。最后，调整后的先验被用于对每个采样答案的投票进行加权。我们在多个LLM和推理基准上进行了大量实验……

    arXiv:2610.19890v1 Announce Type: new  Abstract: Majority Voting (MV) is a simple and widely used strategy which improves the reasoning accuracy of Large Language Models (LLMs) by allocating additional inference-time computation. Specifically, in MV multiple Chain-of-Thoughts (CoTs) are independently sampled and the most frequent final answer is selected. In this work, we propose Length-Conditioned Majority Voting (LCMV), a Test-Time Training approach which improves MV leveraging an empirical observation that has recently been established in several studies: longer CoTs tend to be associated with a lower probability of correctness. We formulate this inductive bias as a length-conditioned prior probability, which is modeled by a simple logistic function. Moreover, we adapt this prior at test time using only unlabeled testing data and without fine-tuning the LLM. Finally, the adapted prior is used to weight the vote of each sampled answer. Extensive experiments across multiple LLMs and r
    
[^64]: RCCM：诊断推理提示迁移中的收益与损害

    RCCM: Diagnosing Gains and Harms in Reasoning-Prompt Transfer

    [https://arxiv.org/abs/2610.19872](https://arxiv.org/abs/2610.19872)

    论文提出RCCM框架，通过将答案与生成结束方式配对来诊断推理提示在不同模型部署间的迁移效果反转，发现简洁推理提示在统一token预算下能提升Phi的答案匹配率却大幅降低Qwen的匹配率，说明提示收益必须结合其执行与评估方式来解读。

    

    一个推理提示可以在某个部署中提升性能，却在另一个部署中降低性能。RCCM（推理-控制能力失配，Reasoning-Control Capability Mismatch）通过将相同问题的答案与生成如何结束的记录配对，来考察这些效果反转现象。在五个开放权重部署和九种策略中，总体准确率掩盖了问题层面的大量收益与损害；一项采用统一预算的MATH500评估在独立样本上检验了迁移效果。在另外60个问题上，在2,048个token的上限下，简洁推理在三个数据集上提升了Phi模型的自动答案匹配率，却显著降低了Qwen模型的匹配率。经作者审查的最终答案承诺带来了较小且统计上未确定的Phi增益。预留答案阶段同样能改善匹配，但会增加输入和调用成本。这些结果表明，提示带来的收益必须结合其执行与评估方式来解读：匹配参考答案并不等于真正给出答案，表面上的压缩收益……（原文摘要在此处截断）

    arXiv:2610.19872v1 Announce Type: new  Abstract: A reasoning prompt can improve one deployment and degrade another. RCCM (Reasoning-Control Capability Mismatch) examines these reversals by pairing answers to the same questions with records of how generation ends. Across five open-weight deployments and nine strategies, aggregate accuracy hides substantial question-level gains and harms; a common-budget MATH500 evaluation examines transfer on a separate sample. On 60 additional questions, concise reasoning improves Phi's automatic answer matching at a 2,048-token cap across three collections, but sharply reduces Qwen's. Author-reviewed final-answer commitments give a smaller, statistically unresolved Phi gain. Reserving an answer stage also improves matching, with additional input and call costs. These results show why a prompt benefit must be interpreted alongside its execution and evaluation: matching a reference answer need not mean delivering it, and an apparent compression benefit 
    
[^65]: 基于流匹配的推理链分布蒸馏实现分布式连续潜在推理

    Rationale Distribution Distillation via Flow Matching for Distributional Continuous Latent Reasoning

    [https://arxiv.org/abs/2610.19871](https://arxiv.org/abs/2610.19871)

    该论文提出推理链分布蒸馏（RDD），利用流匹配将多条文本推理路径蒸馏为固定长度潜在序列上的条件流，以非自回归方式生成潜在推理的完整分布，在大幅降低重复生成成本的同时提升答案覆盖率与推理路径的多样性。

    

    从语言模型中采样多条思维链推理依据能够发现单次尝试所遗漏的正确解。然而，这些推理依据通常十分冗长，使得反复进行自回归生成的代价高昂。连续潜在推理提供了一种紧凑的替代方案，但点值方法将重复的答案采样条件于单一推理依据之上，这可能限制对多样化路径的探索以及答案的覆盖范围。为此，我们提出了推理链分布蒸馏（RDD），通过从多条文本推理路径中蒸馏，训练一个定义在固定长度潜在序列上的条件流。此外，我们通过查询内混合进一步扩展蒸馏目标，在同一查询的完整潜在推理依据之间进行插值。该流以非自回归方式生成整个潜在序列，随后由一个冻结的语言模型从采样得到的推理依据上下文中自回归地解码出答案。实证结果表明，所得的分布……

    arXiv:2610.19871v1 Announce Type: new  Abstract: Sampling multiple chain-of-thought rationales from a language model can uncover correct solutions missed by a single attempt. However, these rationales are typically lengthy, making repeated autoregressive generation costly. Continuous latent reasoning offers a compact alternative, but point-valued methods condition repeated answer sampling on a single rationale, which can limit exploration of diverse paths and answer coverage. We thus propose rationale distribution distillation (RDD), training a conditional flow over fixed-length latent sequences by distilling from multiple textual reasoning paths. Furthermore, we expand the distillation targets through within-query mixup, interpolating between complete latent rationales for the same query. The flow nonautoregressively generates the entire latent sequence, and a frozen language model autoregressively decodes answers from sampled rationale contexts. Empirically, the resulting distributio
    
[^66]: SEA-HELM Audio：跨 SEA-HELM 子集评估多语言 ASR-LLM 流水线性能

    SEA-HELM Audio: Evaluating Multilingual ASR-LLM Pipeline Performance Across SEA-HELM Subsets

    [https://arxiv.org/abs/2610.19869](https://arxiv.org/abs/2610.19869)

    该论文提出 SEA-HELM Audio——一个由文本基准 SEA-HELM 扩展而来的东南亚多语音频评估框架，利用合成语音在五种语言上量化了级联 ASR-LLM 流水线相比文本输入的显著性能退化（多项选择推理下降 9.00–18.50 个百分点，指令遵循下降 20.00–42.86 个百分点，其中泰米尔语退化最严重）。

    

    我们推出了 SEA-HELM Audio，这是一个东南亚（SEA）多语言评估框架，它将基于文本的 SEA-HELM 基准进行扩展，用于评估级联式自动语音识别与大语言模型（ASR-LLM）流水线的性能退化。利用合成语音，我们在五种语言（中文、英语、马来语、印尼语和泰米尔语）上，针对多项选择推理和指令遵循任务，比较了语音输入与文本输入的性能表现。我们采用相同的验证器评估了开放权重和商业 ASR-LLM 系统，并将“语音差距”量化为文本准确率与语音准确率之间的差异。结果显示存在显著的性能退化：多项选择推理任务的准确率下降 9.00 至 18.50 个百分点，指令遵循任务下降 20.00 至 42.86 个百分点。其中泰米尔语的退化最为严重，超过 40 个百分点。我们的发现凸显了仅基于文本的基准在预测……（摘要在此处截断）

    arXiv:2610.19869v1 Announce Type: cross  Abstract: We introduce SEA-HELM Audio, a Southeast Asian (SEA) multilingual evaluation framework that extends the text-based SEA-HELM benchmark to assess performance degradation in cascaded automatic speech recognition and large language model (ASR-LLM) pipelines. Using synthesized speech, we compare speech-input and text-input performance across five languages (Chinese, English, Malay, Indonesian, and Tamil) on multiple-choice reasoning and instruction-following tasks. We evaluate open-weight and commercial ASR-LLM systems using identical verifiers and quantify the speech gap as the difference between text and speech accuracy. Results show substantial degradation, with accuracy drops of 9.00 to 18.50 percentage points for multiple-choice reasoning and 20.00 to 42.86 points for instruction following. Tamil exhibits the largest degradation, exceeding 40 percentage points. Our findings highlight the limitations of text-only benchmarks in predictin
    
[^67]: OpenTWBench（开放台湾基准）：面向大型语言模型的台湾领域知识评测，以及测量装置如何塑造分数

    OpenTWBench (Open Taiwan Benchmark): Taiwan-Domain Knowledge Evaluation for Large Language Models, and How the Measurement Apparatus Shapes the Score

    [https://arxiv.org/abs/2610.19848](https://arxiv.org/abs/2610.19848)

    该论文提出开放式台湾领域知识评测套件 OpenTWBench，包含基于台湾十五年国家考试、共 25 万余道题的 tw-exam-bench，以及关于当代台湾生活的 1,696 道难度校准题的 tw-formosa-bench，并强调数据清洗与难度校准等测量装置设计对模型评测分数的塑造作用。

    

    我们提出 OpenTWBench，这是一个用于评测大型语言模型台湾领域知识的开放式评测套件，由两个基准构成。tw-exam-bench 包含 255,854 道四选项题目，涵盖 148 个学科，源自台湾考选部以开放数据形式发布的十五年国家考试题目；在 31,312 组重复题目中，标准答案的一致率达 99.91%，且全部 27 处不一致均已得到核查说明。经过审阅的第二轮处理剔除了 2,180 道题干依赖于文本未附图表的题目；所有分数均基于剩余的 253,674 道题计算。tw-formosa-bench 包含 1,696 道经难度筛选的、关于台湾日常生活与当代台湾的题目——即执照考试从不涉及的台湾面向——这些题目从社区策展的语料库中生成，遵循“没有任何模型撰写标准答案”的纪律，并以一个冻结的七模型池作为筛选基准，使该模型池的得分落在及格边缘（46.9-74.6%，中位数约 54%）；

    arXiv:2610.19848v1 Announce Type: new  Abstract: We present OpenTWBench, an open evaluation suite for Taiwan-domain knowledge in large language models, built around two benchmarks. tw-exam-bench contains 255,854 four-option questions across 148 academic subjects, from fifteen years of Taiwan's national examinations published as open data by the Ministry of Examination; answer keys agree across 31,312 duplicated question groups at 99.91%, with all 27 disagreements accounted for. A reviewed second pass withdrew 2,180 items whose stems depend on a figure the text does not carry; all scores use the remaining 253,674. tw-formosa-bench contains 1,696 difficulty-curated questions about everyday and contemporary Taiwan - the Taiwan that licensing examinations never ask about - generated from a community-curated corpus under a discipline in which no model writes a ground truth, and curated against a frozen seven-model pool so that the pool scored at the passing margin (46.9-74.6%, median ~54%);
    
[^68]: EGCForest：面向企业群聊的结构化智能体记忆组织

    EGCForest: Structured Agent Memory Organization for Enterprise Group Chats

    [https://arxiv.org/abs/2610.19836](https://arxiv.org/abs/2610.19836)

    提出EGCForest多视图分层记忆框架，将企业群聊消息组织为频道、参与者、制品三个互补的森林结构，并通过放置图和稀疏结构桥接实现跨视图连接，从而可靠地检索和组装分布式的证据。

    

    企业群聊记录了项目进展、决策、任务归属以及其他对智能体有价值的组织信息。可靠的检索极具挑战性，因为证据分散在交错的线程、频道和时间中，且其含义和权威性依赖于发言者。无论是RAG的独立块检索，还是记忆系统中基于摘要的整合方式，都无法可靠地恢复这类分布式证据。我们提出了EGCForest，这是一种新颖的面向企业群聊的多视图分层记忆框架。EGCForest将不可变的原始消息日志组织为三个互补的森林：用于时间工作流的频道森林、用于发言者条件上下文的参与者森林，以及用于交付物生命周期的制品森林。跨视图的连接由放置图和稀疏的结构桥接来实现。基于这一记忆主干，EGCForest通过联合利用消息级相关性等信息来检索并组装证据。（摘要内容在此处被截断）

    arXiv:2610.19836v1 Announce Type: new  Abstract: Enterprise group chats record project progress, decisions, task ownership, and other organizational information valuable to enterprise agents. Reliable retrieval is challenging because evidence spans interleaved threads, channels, and time, with meaning and authority conditioned on the speaker. Neither RAG's independent chunk retrieval nor memory systems' summary-based consolidation can reliably recover such distributed evidence. We present EGCForest, a novel multi-view hierarchical memory framework for enterprise group chats. EGCForest organizes an immutable message log into three complementary forests: a Channel forest for temporal workflows, an Actor forest for speaker-conditioned context, and an Artifact forest for deliverable lifecycles. Cross-view connections are enabled by a Placement Map and sparse structural bridges. Built on this memory backbone, EGCForest retrieves and assembles evidence by jointly leveraging message-level rel
    
[^69]: MAPLE：用于大语言模型评估的多轴参数化知识提取透镜

    MAPLE: A Multi-Axis Parametric Knowledge Extraction Lens for LLM Evaluation

    [https://arxiv.org/abs/2610.19827](https://arxiv.org/abs/2610.19827)

    该论文提出MAPLE统一评估框架，通过在表层、概率和超越采样三个层面测量答案恢复与干扰、正确词元排名变化及采样遗漏答案的恢复情况，全面解耦知识提取增强（KEE）技术对大语言模型参数化知识提取的多样化影响。

    

    大语言模型从训练数据中吸收海量知识，但其固有的概率性本质无法保证这些知识会在其输出中显现。为弥合这一差距，知识提取增强（KEE）技术——如激活引导、解码调整和提示——通过修改推理过程，可能强化已有的预测，或挖掘出较难获取的答案。然而，现有评估以总体准确率为中心，将“成功恢复的失败案例”与“被干扰的成功案例”合并为单一分数，从而掩盖了这些技术多样化的影响。我们提出MAPLE，一个用于刻画KEE技术如何影响参数化知识提取的统一框架。其诊断指标通过测量答案的恢复与干扰、未恢复失败案例中正确词元的排名变化，以及固定预算采样所遗漏答案的恢复情况，在表层、概率和超越采样三个层面解耦上述影响。我们评估了六种KEE……（原文摘要在此处截断）

    arXiv:2610.19827v1 Announce Type: cross  Abstract: LLMs absorb vast knowledge from training data, yet their inherently probabilistic nature offers no guarantee that this knowledge will surface in their outputs. To bridge this gap, Knowledge Extraction Enhancement (KEE) techniques such as activation steering, decoding adjustments, and prompting modify inference, potentially reinforcing existing predictions or surfacing less accessible answers. Yet their evaluations centered on aggregate accuracy, which obscures diverse effects by collapsing recovered failures and disrupted successes into a single score. We introduce MAPLE, a unified framework for characterizing how KEE techniques affect parametric knowledge extraction. Its diagnostic metrics disentangle these effects at the surface, probability, and beyond-sampling levels by measuring answer recovery and disruption, gold-token rank shifts on unrecovered failures, and recovery of answers missed by fixed-budget sampling. We evaluate six K
    
[^70]: 模型 vs. 用户：谁的价值观占上风？多轮对话中的推理时价值对齐

    Model vs. User: Whose Values Prevail? Inference-Time Value Alignment in Multi-Turn Conversations

    [https://arxiv.org/abs/2610.19817](https://arxiv.org/abs/2610.19817)

    该研究首次在多轮对话中系统评估四种推理时价值引导方法，发现当用户与模型价值观冲突时，与模型基线偏好一致的一方往往胜出，且早期成功的引导效果会在数轮对话后减弱或反转，因此价值对齐应在完整双向对话中评估。

    

    人类价值观塑造了我们对语言模型行为的期望。主流的对齐方法将价值观视为全局目标，然而合适的价值取向实际上是个人的、依赖于具体情境的。无需训练的对齐方法可以在推理时将模型引导至目标价值观，但目前主要在单轮对话设置和单一引导方向（例如朝向安全）下进行评估。我们在多轮对话中评估了四种基于提示和激活的引导方法，其中用户和助手模型朝向相同或相反但同样值得追求的价值观。我们使用线性探针在三个模型和两组价值权衡中追踪对齐情况。当助手和用户朝相反方向拉扯时，与助手基线偏好一致的一方往往占上风，而且早期看似成功的引导效果往往在几轮对话后减弱或反转。因此，引导方法应该在完整对话中、并在价值权衡的两个方向上进行评估。

    arXiv:2610.19817v1 Announce Type: new  Abstract: Human values shape how we expect language models to behave. Dominant alignment approaches treat values as global objectives, although the appropriate value orientation is personal and contextual. Training-free alignment can steer models toward target values at inference time, yet has been largely evaluated in single-turn settings and with a single steering direction (e.g., toward safety). We evaluate four prompt- and activation-based steering methods in multi-turn conversations, where user and assistant models push toward the same or opposing, equally desirable values. We track alignment with a linear probe across three models and two value trade-offs. When assistant and user pull in opposite directions, the side matching the assistant's baseline bias often prevails, and effects that look successful early often fade or reverse after a few turns. Steering should therefore be evaluated over full conversations and in both directions of a va
    
[^71]: Delta Tokenizer：面向多核CPU的会话增量参考一致性分词与长提示词分段并行方法

    Delta Tokenizer: Reference-Identical Tokenization of Session Deltas and Segment-Parallelism for Long Prompts on Multicore CPUs

    [https://arxiv.org/abs/2610.19801](https://arxiv.org/abs/2610.19801)

    提出Delta Tokenizer，一种仅对会话增量进行分词且保证与完整分词结果完全一致的CPU分词前端，并利用分段并行加速多核CPU上长提示词的分词，显著降低TTFT和CPU开销。

    

    驱动大语言模型的智能体会一次又一次地向模型发送相同的对话，每一轮只在末尾追加少量文本。推理引擎会复用共享前缀的KV缓存，因此只有新增的token会通过模型。然而，分词环节却没有这样的捷径。服务引擎的CPU前端会将每个请求从第一个字符开始重新转换为token ID，这延长了首token生成时间（TTFT），并且每一轮都消耗CPU资源。我们提出了Delta Tokenizer，一个CPU分词前端，它只对同一会话中当前请求与前一个请求之间的增量进行分词。它保留较早的token序列，仅重新编码追加的文本以及追加操作可能影响其token的短尾部，并且只在一个分词器切分规则本身无法移动的位置将两部分拼接，从而保证结果与完整分词完全一致。对于没有可复用状态的较长请求，它会对文本进行切分（原文在此处截断），以实现多核CPU上的分段并行处理。

    arXiv:2610.19801v1 Announce Type: new  Abstract: An agent that drives an LLM sends the model the same conversation again and again, and each turn adds only a little text at the end. Inference engines reuse the KV cache of the shared prefix, so only the new tokens pass through the model. However, tokenization has no such shortcut. The CPU front end of the serving engine converts every request back into token IDs from its first character, which lengthens the time to first token (TTFT) and consumes CPU on every turn.   We present Delta Tokenizer, a CPU tokenization front end that tokenizes only the delta between a request and the previous request of the same session. It retains the earlier token sequence, encodes again only the appended text and the short tail whose tokens the append could affect, and joins the two only at a position the tokenizer's splitting rules themselves cannot move, which keeps the result exact. For long requests with no state to reuse, it cuts the text where practi
    
[^72]: 共同进化，各就其位：社会脚手架如何塑造自我进化的多智能体系统

    Evolve Together, Fall into Place: How Social Scaffolding Shapes Self-Evolving Multi-Agent Systems

    [https://arxiv.org/abs/2610.19799](https://arxiv.org/abs/2610.19799)

    该论文提出AGORA框架，首次揭示了自我进化多智能体系统的社会脚手架（团队构成与相互依赖结构）决定进化成效——单纯的进化作用有限，而合适的脚手架可通过让最有能力的成员主导决策、其余成员退居辅助，使团队性能超越单独进化的强智能体高达25分。

    

    最近的研究允许大语言模型（LLM）智能体进化自身的工作框架，甚至进化到团队本身。然而在人类群体中，相同的能力在不同的社会条件下会取得不同的结果。借鉴发展心理学中“脚手架”（scaffolding）的概念，我们研究自我进化多智能体系统（MAS）的社会脚手架——即团队构成和相互依赖结构（例如协作、从众或竞争）——是否会塑造团队的进化方式。我们提出了AGORA（Agents Growing through Organization and Relational Arrangements，通过组织与关系安排促进智能体成长），该框架在三种进化机制中独立变化这两个维度。在网络搜索、代码库理解和文档问答任务中，我们发现：(1) 单纯的进化很少有帮助，而合适的脚手架可比单独进化的强大智能体高出多达25分；(2) 其作用机制在于将决策权转移给最有能力的成员，同时其他成员退居辅助地位；(3) 该成员……（摘要在此处截断）

    arXiv:2610.19799v1 Announce Type: new  Abstract: Recent work lets LLM agents evolve their own harness, up to the team itself. Yet in human groups, the same abilities achieve different outcomes under different social conditions. Borrowing the notion of scaffolding from developmental psychology, we investigate whether the social scaffolding of a self-evolving multi-agent system (MAS), namely its team composition and structure of interdependence (e.g., collaboration, conformity, or competition), shapes how the team evolves. We present AGORA (Agents Growing through Organization and Relational Arrangements), a framework that varies the two axes independently across three evolution mechanisms. Across web search, codebase understanding, and document QA, we find that (1) evolution alone rarely helps, while the right scaffolding surpasses a strong agent evolved alone by up to 25 points; (2) it does so by moving authority to the most capable member while the others step back; and (3) this member
    
[^73]: 通过强化学习学习感知到行动的规则以实现心智理论推理

    Learning Perception to Action Rules through Reinforcement Learning for Theory-of-Mind Reasoning

    [https://arxiv.org/abs/2610.19796](https://arxiv.org/abs/2610.19796)

    本文提出利用GRPO强化学习让模型学习显式的“感知到行动”规则来改进大语言模型的心智理论推理，其效果超越提示、监督微调和思维推理方法，且所生成的规则可作为可迁移知识提升其他模型（如GPT-4o-mini）的性能。

    

    在心智理论推理中，大语言模型可能能够正确推断智能体的心理状态，但在预测或判断智能体的后续行动时却未能利用这一推断。在本工作中，我们研究通过规则显式地连接感知、信念与行动是否能改进心智理论推理。我们生成简洁的“如果-那么”规则来表达感知与行动之间的关系，并使用群体相对策略优化（GRPO）在四个心智理论基准上训练Qwen2.5和Llama3模型。带规则生成的GRPO优于提示方法、监督微调和思维推理。此外，由较大模型（参数量不低于30亿）生成的规则提供了可迁移的知识，能够提升GPT-4o-mini的下游性能。这些发现表明，显式规则有助于模型弥合心智理论推理中的推理鸿沟，并支持跨模型的知识迁移。

    arXiv:2610.19796v1 Announce Type: cross  Abstract: In Theory-of-Mind (ToM) reasoning, large language models may correctly infer the agent's mental state but fail to use that inference when predicting or judging the agent's subsequent actions. In this work, we investigate whether explicitly connecting perception, belief, and action with rules improves ToM reasoning. We generate concise \textit{if--then} rules that express relations between perception to action and use Group Relative Policy Optimization (GRPO) to train Qwen2.5 and Llama3 models across four ToM benchmarks. GRPO with rule generation outperforms prompting, supervised fine-tuning, and think reasoning. Moreover, rules generated by larger models ($\geq$3B) provide transferable knowledge that improves downstream performance of GPT-4o-mini. These findings suggest that explicit rules help models bridge the reasoning gap in ToM and support knowledge transfer across models.
    
[^74]: JetSpec++：具有自我精炼的前瞻式并行草稿生成

    JetSpec++: Foresight Parallel Drafting with Self Refinements

    [https://arxiv.org/abs/2610.19792](https://arxiv.org/abs/2610.19792)

    JetSpec++通过块内自精炼、块间前瞻草稿生成和流水线化草稿生成与验证的算法-系统协同设计，几乎完全隐藏草稿模型的开销，实现最高7.45倍于自回归解码的加速。

    

    推测解码的有效性由两个量决定：草稿token的接受率和生成它们的成本。我们证明这两个目标可以通过算法-系统协同设计来共同优化，并提出JetSpec++，它在保留强大学习型草稿模型的同时，几乎完全隐藏了其成本。JetSpec++结合了三个组件：块内自精炼，训练草稿模型在token块内迭代修正自己的并行预测；块间前瞻草稿生成，在当前块被验证之前就开始生成下一个块；以及流水线化的草稿生成与验证，使两个阶段相互重叠，从而使草稿生成的边际墙钟时间成本趋近于零。与最先进的推测解码方法相比，JetSpec++相对自回归（AR）解码实现了高达7.45倍的墙钟加速，相对DSpark实现了1.54倍的加速，吞吐量高达每秒1,361个token。

    arXiv:2610.19792v1 Announce Type: new  Abstract: Speculative decoding's effectiveness is governed by two quantities: the acceptance rate of drafted tokens and the cost of producing them. We show the two objectives can be co-optimized with algorithm-system co-design, and present JetSpec++, which retains a strong learned draft model while hiding its cost almost entirely. JetSpec++ combines three components: intra-block self refinement, which trains the draft model to iteratively correct its own parallel predictions within a block of tokens; inter-block foresight drafting, which begins the next block before the current one is verified; and pipelined drafting and verification, which overlaps the two stages so drafting's marginal wall-clock cost approaches zero. In comparison with state-of-the-art speculative decoding methods, JetSpec++ attains a wall-clock speedup of up to $7.45\times$ over autoregressive (AR) decoding and $1.54\times$ over DSpark, with up to 1,361 tokens per second for a 
    
[^75]: 更多方法，就能带来更好的智能体吗？对智能体强化学习中自蒸馏方法的现实检验

    More Methods, Better Agents? A Reality Check on Self-Distillation in Agentic Reinforcement Learning

    [https://arxiv.org/abs/2610.19789](https://arxiv.org/abs/2610.19789)

    本文通过 132 次训练运行（累计超过 1.2 万 H800 GPU 小时），在四个智能体基准和三种模型规模上系统性地将十种自蒸馏方法与 GRPO 直接对比，以厘清性能提升的真实来源，并揭示各方法在性能、训练稳定性与计算成本上的实际优劣。

    

    智能体强化学习中的自蒸馏已催生出越来越多的方法。然而，现有评估往往在不同实验条件下复用已发表的基线分数，同期提出的各方法之间鲜有直接比较，且对训练稳定性和计算成本的关注有限。这些局限掩盖了所报告性能提升的真正来源，也掩盖了各竞争方法在实际中的优缺点。高昂的智能体训练成本进一步凸显了建立统一比较基准的必要性。为弥补这些空白，我们在 ALFWorld、ScienceWorld、Search-QA 和 WebShop 四个基准上，针对三种不同模型规模，将十种代表性方法与 GRPO 进行了对比。本研究共包含 132 次训练运行，累计记录 12,124.4 个 H800 GPU 小时，并对预训练骨干模型及全部十一种训练后的方法在八个 rollout 种子上进行了重复评估。除最终任务性能外，我们还考察了训练进度、稳定性以及测量……（摘要原文在此截断）

    arXiv:2610.19789v1 Announce Type: new  Abstract: Self-distillation in agentic reinforcement learning has produced a growing number of methods. However, existing evaluations often reuse published baseline scores under differing experimental conditions, provide few direct comparisons among contemporaneous methods, and give limited attention to training stability and computational cost. These limitations obscure the source of reported gains and the practical strengths and weaknesses of competing methods. Costly agent training further motivates a common basis for comparison. To address these gaps, we compare ten representative methods against GRPO on ALFWorld, ScienceWorld, Search-QA, and WebShop across three model sizes. Our study comprises 132 training runs totaling 12,124.4 recorded H800 GPU-hours, with repeated evaluations over eight rollout seeds for the pretrained backbones and all eleven trained methods. Beyond final task performance, we examine training progress, stability, and mea
    
[^76]: 似然并非作者身份：我们能可靠地衡量AI辅助写作中的人类贡献吗？

    Likelihood Is Not Authorship: Can We Reliably Measure Human Contribution in AI-Assisted Writing?

    [https://arxiv.org/abs/2610.19767](https://arxiv.org/abs/2610.19767)

    该论文揭示了基于提示词预测能力来衡量AI辅助写作中人类贡献的方法极易被操纵——仅需让模型复制几个随机ID的“特异性填充”技巧，就能使未改变文本的评分波动达量级的三分之一到一半，证明似然并非衡量作者身份的可靠指标。

    

    AI文本检测通常关注的是某段文本是否由机器撰写。随着AI日益成为写作中不可或缺且被广泛接受的一部分，更相关的问题不再是是否使用了AI，而是文本中有多少反映了人类的贡献。一项主流研究方法通过测量人的提示词对最终文本的预测能力来估计这一贡献。我们证明这种度量方式很容易被操纵。我们提出了“特异性填充”（SPECIFICITY PADDING）方法，只需让模型复制提示词中的几个随机ID，仅此一项就能让仅提供标题的请求看起来比提供了文章核心思想的请求更具“人类感”。随后的上下文交换审计保持每份文档完全不变，仅向评估器显示或隐藏这些ID。未做任何修改的文本评分会在其量级的三分之一到一半之间波动，并且在所有测试文档上都会上升，涵盖新闻和科学摘要，跨越从3.8B到753B参数的多种大语言模型。相同的文章平均可以达到85%（摘要在此处被截断）。

    arXiv:2610.19767v1 Announce Type: new  Abstract: AI-text detection generally asks whether a machine wrote a text. As AI becomes an increasingly integral and accepted part of writing, the more relevant question is no longer whether AI was used, but how much of the text reflects human contribution. A prominent line of work estimates this by measuring how predictive a person's prompt is of the final text. We show that this measure can be easily manipulated. We introduce SPECIFICITY PADDING that simply asks the model to copy a few random IDs from the prompt, and that alone makes a title-only request look more human than one that supplies the article's key ideas. A context-swap audit then holds each document fixed, and only shows or hides the IDs to the evaluator. The score of the unchanged text swings by a third to half of its scale, and it rises for every document we test, on news and scientific abstracts, across varied LLMs from 3.8B to 753B parameters. The same articles can average 85\%
    
[^77]: PRISMATIC：临床决策中多智能体治疗交互的处方风险检测系统

    PRISMATIC : Prescription Risk Inspection System for Multi-Agent Treatment Interaction in Clinical Decision-Making

    [https://arxiv.org/abs/2610.19762](https://arxiv.org/abs/2610.19762)

    提出了PRISMATIC多智能体架构，通过代表不同工作流程角色的专业智能体协作以及动态自更新权重调整（DSWA）等新型自适应协调机制，为多病共存患者迭代生成安全、合理且可解释的用药方案，以解决专科间碎片化决策导致的处方错误问题。

    

    对于需要复杂用药方案的多病共存患者而言，医疗处方错误是一个重大的患者安全问题。这些错误往往源于各医学专科之间碎片化的决策，这种关键的协调失灵会导致药物不良事件和再入院。为解决这一协调挑战，我们提出了PRISMATIC，这是一种多智能体架构，其中专门的智能体各自代表一个工作流程角色，协作生成并完善治疗方案。该系统运行于黑板架构之上，首先从临床笔记中对患者信息进行结构化处理，然后迭代地制定安全、合理且可解释的用药方案。智能体之间的协作由两种新颖的自适应协调机制进行管理：(1) 动态自更新权重调整（DSWA），这是一种长期机制，根据历史病例结果调整各智能体的影响权重，使系统能够不断适应……

    arXiv:2610.19762v1 Announce Type: new  Abstract: Medical prescribing errors for multimorbid patients, who require complex drug regimens, are a major patient safety concern. These errors often stem from fragmented decision-making across medical specialties, a critical coordination failure that leads to adverse drug events and rehospitalizations. To address this coordination challenge, we introduce PRISMATIC, a multi-agent architecture where specialized agents, each representing a workflow role, collaboratively generate and refine treatment plans. Operating on a blackboard architecture, the system first structures patient information from clinical notes, then iteratively develops safe, rational, and interpretable drug regimens. The agents' collaboration is governed by two novel adaptive coordination mechanisms: (1) Dynamic Self-Updating Weight Adjustment (DSWA), a long-term mechanism that adjusts the influence weights of agents based on historical case outcomes, enabling the system to ad
    
[^78]: Qwen-MT-Lumen 在 WMT2026：通过质量感知训练与在线策略蒸馏实现指令专业化翻译

    Qwen-MT-Lumen at WMT2026: Instruction-Specialized Translation via Quality-Aware Training and On-Policy Distillation

    [https://arxiv.org/abs/2610.19758](https://arxiv.org/abs/2610.19758)

    该论文提出面向 WMT 2026 的开源翻译系统 Lumen，通过持续预训练、监督微调、将离线 M²PO 扩展为带 LLM 评判打分的在线偏好优化、以及在线策略蒸馏的四阶段流水线，实现同时满足风格、术语与结构化输出等细粒度指令要求的高质量翻译。

    

    我们提出 Lumen（开源模型已在...发布），这是我们提交给 WMT 2026 通用机器翻译共享任务的系统。我们专注于指令专业化翻译，即系统必须在生成准确译文的同时，满足对风格、术语和结构化输出的细粒度要求。Lumen 采用四阶段流水线：首先，持续预训练与监督微调共同增强多语言翻译能力以及遵循翻译特定指令（如术语、风格和格式约束）的能力；其次，我们将原本离线的 M²PO 偏好优化框架扩展为在线偏好优化阶段：由当前策略采样候选译文，由大模型评判器给出连续的直接评估分数，并利用由此产生的多对偏好来更新模型；第三，在线策略蒸馏将更强的翻译能力……（原文摘要在此处被截断，后续内容缺失）

    arXiv:2610.19758v1 Announce Type: new  Abstract: We present Lumen, The open-source model is available at, our submission to the WMT 2026 General Machine Translation Shared Task. We focus on instruction-specialized translation, where systems must produce accurate translations while satisfying fine-grained requirements on style, terminology, and structured output. Lumen follows a four-stage pipeline. First, continued pre-training and supervised fine-tuning jointly strengthen multilingual translation competence and the ability to follow translation-specific instructions, such as terminology, style, and format constraints. Second, we extend the originally offline M^{2}PO preference-optimization framework into an online preference-optimization stage: the current policy samples candidate translations, an LLM judge assigns continuous direct-assessment scores, and the resulting multi-pair preferences are used to update the model. Third, on-policy distillation transfers stronger translation-ins
    
[^79]: 超越可达性：面向生物医学多跳问答的关系-语义推理框架

    Beyond Reachability: A Relation-Semantic Reasoning Framework for Biomedical Multi-hop QA

    [https://arxiv.org/abs/2610.19754](https://arxiv.org/abs/2610.19754)

    该论文提出了ReSemReason框架，通过在候选发现与证据选择之间共享问题诱导的类型化关系模式分布，结合模式兼容性图搜索、关系条件化注意力机制和不确定性加权路径先验，解决了生物医学多跳问答中“答案可达但证据无效”的问题。

    

    生物医学知识图谱上的多跳推理需要识别出一条语义有效的关系序列来支持答案。现有方法虽然能够检索到正确答案，但在大量看似相关的候选答案中往往排名不佳。即使检索到了正确答案，其相关证据和推理路径也可能无法支持问题所需的关系组合。我们将这种答案可达性与证据有效性之间的脱节称为“可达但无效”问题。我们提出了ReSemReason框架，它在候选发现与证据选择之间共享一个由问题诱导的类型化、有向关系模式分布。模式兼容性用于指导图搜索；随后ReSemAttn利用关系条件化的键/值投影和不确定性加权路径先验，关注针对特定候选的证据。当出现……时，一个独立的绝对支持度分数对归一化注意力进行了补充（摘要原文在此处截断）。

    arXiv:2610.19754v1 Announce Type: new  Abstract: Biomedical multi-hop reasoning over knowledge graphs requires identifying a semantically valid relation sequence that supports the answer. Existing methods can retrieve correct answers, yet often rank them poorly within a large pool of seemingly relevant candidates. Even when a correct answer is retrieved, its associated evidence and reasoning path may fail to support the relation composition required by the question. We term this disconnect between answer reachability and evidential validity the reachable-but-invalid problem. We present ReSemReason, which shares a question-induced distribution over typed, directed relation schemas between candidate discovery and evidence selection. Schema compatibility guides graph search; ReSemAttn then uses relation-conditioned key/value projections and an uncertainty-weighted path prior to attend to candidate-specific evidence. A separate absolute-support score complements normalized attention when a
    
[^80]: 思维的语言语体重要吗？文言文作为中文大语言模型的紧凑推理语体

    Does the Linguistic Register of Thinking Matter? Wenyan as a Compact Reasoning Register for Chinese LLMs

    [https://arxiv.org/abs/2610.19741](https://arxiv.org/abs/2610.19741)

    该论文提出“文言思维”方法，让中文大语言模型用文言文作为紧凑的推理语体，在受限解码预算下大幅缩短推理长度并降低答案缺失率。

    

    中文推理导向大语言模型的可见推理轨迹通常以现代白话文书写，这在有限的解码预算下可能显得冗长。我们探讨推理所用语言语体能否用于提升效率。我们提出文言思维，该方法以文言文输出中间推理过程，同时保留简短的直接答案输出。从现代中文推理轨迹出发，我们仅将推理字段改写为文言文，并基于改写后的监督数据微调 Qwen3-4B-Thinking-2507。为了区分语体转换与一般性文本压缩，我们使用相同的源数据、教师模型家族、基座模型和SFT训练配方，构建了一个紧凑的现代中文对照组。在 MMLU-ProX-Lite-ZH、CEVAL 和 AGIEval-CN 上的预算敏感评估中，文言思维在受限解码条件下显著降低了推理长度和答案缺失率。紧凑改写在很大程度上解释了……

    arXiv:2610.19741v1 Announce Type: new  Abstract: Visible reasoning traces of Chinese reasoning-oriented LLMs are typically written in modern vernacular Chinese, which may be verbose under limited decoding budgets. We ask whether the linguistic register of reasoning can be used to improve efficiency. We propose Wenyan Thinking, which emits intermediate reasoning in Classical Chinese (wenyanwen) while retaining a short direct-answer output. Starting from modern-Chinese reasoning traces, we rewrite only the reasoning field into wenyan and fine-tune Qwen3-4B-Thinking-2507 on the rewritten supervision. To distinguish register shift from generic compression, we construct a compact modern-Chinese control using the same source data, teacher family, base model, and SFT recipe. Across budget-sensitive evaluations on MMLU-ProX-Lite-ZH, CEVAL, and AGIEval-CN, Wenyan Thinking substantially reduces reasoning length and missing-answer rates under constrained decoding. Compact rewriting explains much 
    
[^81]: MakSafe：用于道路车辆安全规则形式化规范与合规性检查的时空语义框架

    MakSafe: A Spatiotemporal Semantic Framework for Formal Specification and Compliance Checking of Road Vehicle Safety Rules

    [https://arxiv.org/abs/2610.19735](https://arxiv.org/abs/2610.19735)

    MakSafe是一个形式化时空语义框架，通过语义区域、动作预测和轨迹树推理对道路车辆安全规则进行规范定义与合规性检查，单次规则检查耗时远低于一毫秒，并能从同一初始场景对不同分支分别得出安全或不安全的判定。

    

    基于场景的验证对于自动驾驶道路车辆至关重要，但无法覆盖开放世界交通的全部变异性。我们提出了MakSafe，这是一个形式化的时空框架，通过语义区域、基于动作的预测和轨迹树推理来规范定义与检查道路车辆安全规则，避免了统一的时间离散化。我们形式化了道路网络、车辆状态和占用区域，然后为前车识别和碰撞时间验证构建了可执行规则。评估结合了依赖于轨迹的案例研究和关于交通密度（N）、时间范围（H）和分支因子（B）的可扩展性基准测试。MakSafe能够产生特定于分支的判定，从相同的初始场景推导出安全与不安全的结果，并在控制更新后重新评估场景。在报告的参数扫描范围内，单次规则检查的时间远低于一毫秒，而更长的时间范围会降低TTC检查的速度。

    arXiv:2610.19735v1 Announce Type: new  Abstract: Scenario-based validation is essential for automated road vehicles but cannot cover the full variability of openworld traffic. We present MakSafe, a formal spatiotemporal framework for specifying and checking road vehicle safety rules through semantic zones, action-based forecasting, and trace-tree reasoning, avoiding uniform time discretization. We formalize the road network, vehicle states, and occupancy zones, then build executable rules for front-vehicle identification and timeto-collision (TTC) verification. The evaluation combines a tracedependent case study with a scalability benchmark over traffic density (N ), horizon (H), and branch factor (B). MakSafe yields branch-specific verdicts, deriving safe and unsafe outcomes from the same initial scene, and re-evaluates a scenario after a control update. A single rule check stays well below a millisecond across the reported sweep, and longer horizons slow the TTC check.
    
[^82]: 通过注意力引导的表示扰动缓解视频幻觉

    Mitigating Video Hallucinations via Attention-Guided Representation Perturbation

    [https://arxiv.org/abs/2610.19728](https://arxiv.org/abs/2610.19728)

    提出了ARP-CD，一种无需训练的对比解码框架，通过利用答案生成时的注意力权重引导对视觉隐状态的梯度更新来构建对比负样本，从而有效缓解视频大语言模型的幻觉问题。

    

    视频大语言模型（Video LLMs）经常生成输入视频无法支持的回答。对比解码（Contrastive Decoding, CD）可以缓解这类幻觉，但其有效性取决于负分支如何削弱视觉证据。现有方法依赖于预定义的扰动或由问题重构引导的掩码操作，对模型在答案生成过程中的视觉依赖控制能力有限。我们提出了注意力引导表示扰动的对比解码方法（ARP-CD），这是一个无需训练的框架，可直接在表示空间中构建对比负样本。ARP-CD利用答案预测位置分配给视觉token的注意力权重，来引导对语言模型输入视觉隐状态的有界梯度更新。这些更新会降低视觉注意力并产生一个负输入，在整个生成过程中，该负输入的预测与原始分支的预测进行对比，从而缓解幻觉问题。

    arXiv:2610.19728v1 Announce Type: cross  Abstract: Video large language models (Video LLMs) often generate responses unsupported by the input video. Contrastive decoding (CD) can mitigate these hallucinations, but its effectiveness depends on how the negative branch weakens visual evidence. Existing methods rely on predefined perturbations or masking guided by question reconstruction, offering limited control over the model's visual reliance during answer generation. We introduce Attention-guided Representation Perturbation for Contrastive Decoding (ARP-CD), a training-free framework for constructing contrastive negatives directly in representation space. ARP-CD uses the attention mass assigned to visual tokens at the answer-prediction position to guide bounded gradient updates of the language model's input visual hidden states. These updates reduce visual attention and produce a negative input, whose predictions are contrasted with those of the original branch throughout generation. E
    
[^83]: 通过候选集视觉定位识别对话歧义

    Recognizing Dialogue Ambiguity via Candidate Set Visual Grounding

    [https://arxiv.org/abs/2610.19723](https://arxiv.org/abs/2610.19723)

    该论文提出了候选集视觉定位（CSVG）这一新任务及 CandVG 基准，通过估计与对话上下文一致的目标物体集合来识别对话歧义，并发现更强的视觉定位能力并不一定带来更好的 CSVG 表现。

    

    在真实世界的交互中，解读一句有歧义的话语往往需要超出单句话语的上下文信息。当话语指向视觉场景中的物体时，需要随着上下文的积累不断维护和更新可能指代对象的集合，而不能假设已有信息能够唯一确定某个物体。然而，以往关于交互式视觉定位（即根据语言表达定位所指物体）的研究，主要聚焦于在获得足够的对话上下文后识别出唯一的物体。为填补这一空白，我们提出了候选集视觉定位，这是一项估计与给定对话上下文一致的物体集合的任务。我们进一步推出了 CandVG 基准，用于评估候选集如何随对话轮次变化而变化。我们对视觉-语言模型的评估表明，更强的视觉定位能力并不一定会带来更好的 CSVG 表现。

    arXiv:2610.19723v1 Announce Type: new  Abstract: Interpreting an ambiguous utterance in real-world interactions often requires context beyond a single utterance. When the utterance refers to objects in a visual scene, it is necessary to maintain and update the set of plausible referents as context accumulates, without assuming that the available information identifies a unique object. However, prior work on interactive visual grounding, which localizes objects referred to by linguistic expressions, has focused on identifying a unique object once sufficient dialogue context becomes available. To address this gap, we propose Candidate Set Visual Grounding (CSVG), a task for estimating the set of objects consistent with the given dialogue context. We further introduce CandVG, a benchmark that evaluates how the candidate set changes across dialogue turns. Our evaluation of vision-language models shows that stronger visual grounding capability does not necessarily yield better CSVG performa
    
[^84]: 语言模型的表达上限

    Articulation Upper Bounds for Language Models

    [https://arxiv.org/abs/2610.19721](https://arxiv.org/abs/2610.19721)

    本文提出了新的提示优化算法和基于“提示恢复”的传输度量方法，用于研究随着大语言模型能力扩展时概念在模型间传输的上限，从而探究语言表达能力自身的极限。

    

    语言只能捕捉说话者之间形成意义的感知、直觉和意图中的一部分。那么，语言的极限究竟在哪里？我们朝着在大语言模型（LLM）中衡量这一问题的方向迈出了步伐：对于一组概念，我们测量随着LLM能力的扩展，每个概念在两个LLM之间的传输效果如何。这需要新颖的测量工具：(1) 最优消息：我们引入了一种提示优化算法，在基准测试中比现有算法高出最多14%，并能准确估计提示的上限。(2) 一种传输度量：要求发送方对文本进行标注，然后接收方尝试恢复发送方的提示，并测量恢复的程度；我们发现该度量与人类的标注错误强相关。利用这些度量，我们使用观察性缩放定律来研究：随着发送方和接收方容量的增长，使用最优消息传达的概念的可传输性如何提升——它是否会在某一水平以下渐近收敛……

    arXiv:2610.19721v1 Announce Type: cross  Abstract: Language captures only a subset of the perceptions, intuitions and intents that form meaning between speakers. What, then, are the limits of language? We take steps towards measuring this in LLMs as follows: for a set of concepts, we measure how well each concept can be transmitted between two LLMs as their capabilities scale. This requires novel measurement tools: (1) optimal messages: we introduce a prompt optimization algorithm that beats existing algorithms by up to 14% on benchmarks and accurately estimates prompt upper bounds. (2) A transmission measure: the sender is asked to label text; the receiver then tries to recover the sender's prompt, and recovery is measured; we find this correlates strongly with human label error. With these measures, we use observational scaling laws to study how transmissibility of a concept, conveyed with optimal messages, increases as transmitter and receiver capacity grows: does it asymptote below
    
[^85]: 超越状态容量：Mamba中的检索为何失效

    Beyond State Capacity: Why Retrieval Breaks in Mamba

    [https://arxiv.org/abs/2610.19709](https://arxiv.org/abs/2610.19709)

    该论文挑战了“Mamba检索失败源于固定状态容量”的普遍假设，证明即使在状态容量不构成瓶颈的查询优先检索设置中Mamba依然显著落后，其根本原因在于词元表示未能编码与查询的相关性，致使选择门控无法正确调控状态更新。

    

    Mamba模型在上下文检索任务上的表现不如Transformer。这一差距通常被归因于Mamba固定大小的循环状态，即随着上下文增长会丢失信息。我们通过聚焦于“查询优先”检索来挑战这一假设——在这种设置中，查询先于上下文出现，状态容量不应成为约束条件。我们表明，即使在这种设置下，Mamba仍然大幅落后于Transformer，尽管其选择机制本就是为了从干扰内容中区分相关内容而设计的。随后我们证明，Mamba的选择门控对状态更新具有足够的控制力，可以影响模型输出：仅对特定门控进行干预，就能使准确率在两个极端之间摆动。然而，由于门控依赖于输入，它们需要输入词元的表示中编码其与查询的相关性。因此，我们在各层中探测词元表示，发现相关性信号（摘要在此处截断）。

    arXiv:2610.19709v1 Announce Type: new  Abstract: Mamba models underperform Transformers on in-context retrieval. This gap is usually attributed to Mamba's fixed-size recurrent state, which loses information as the context grows. We challenge this assumption by focusing on query-first retrieval, a setup in which the query precedes the context and state capacity should not be the binding constraint. We show that even in this setting Mamba lags substantially behind Transformers, despite its selection mechanism, which was designed to distinguish relevant content from distractors. We then demonstrate that the Mamba selection gates have sufficient control over the state updates to influence the model output: intervening on specific gates alone can swing accuracy between extremes. However, as gates are input-dependent, they require the incoming token's representation to encode its relevance to the query. We therefore probe token representations across layers, and find that the relevance signa
    
[^86]: RouteWeaver：将模式选择与执行编织为统一的大语言模型路由

    RouteWeaver: Weaving Mode Selection and Execution into Unified LLM Routing

    [https://arxiv.org/abs/2610.19705](https://arxiv.org/abs/2610.19705)

    提出 RouteWeaver 统一路由框架，在单个 LLM 策略中联合学习路由模式选择（单轮/多轮/智能体）与模式内模型执行，并通过 COMET-GRPO 将优势信号解耦到两个决策层级以实现更精准的信用分配与联合优化。

    

    大语言模型路由（LLM routing）能够在多种任务中动态选择并协调异构的语言模型。然而，现有方法大多围绕某种预定义的执行范式构建，例如单轮、多轮或智能体（agentic）路由，这固定了模型调用的组织方式，限制了对不同查询需求的适应性。我们提出 RouteWeaver，一个统一的路由框架，它将这些路由模式整合到单个大语言模型策略之中，联合学习应使用哪种模式以及在该模式下如何选择模型。为了对模式选择与模式内执行进行联合优化，我们开发了 COMET-GRPO（协调式模式-执行训练，COordinated Mode-Execution Training），该方法将轨迹级优势函数解耦为分别对应模式选择与模式内执行的两路独立信号，从而实现对各决策层级更有针对性的信用分配。我们进一步引入渐进式探索机制，使训练从外部指派的模式逐渐过渡到自主选择的模式。

    arXiv:2610.19705v1 Announce Type: new  Abstract: LLM routing dynamically selects and coordinates heterogeneous language models across diverse tasks. However, existing approaches are largely built around a predefined execution paradigm, such as single-round, multi-round, or agentic routing, which fixes how model calls are organized and limits adaptation to different query needs. We propose RouteWeaver, a unified routing framework that integrates these routing modes within a single LLM policy, jointly learning which mode to use and how to select models within that mode. To jointly optimize mode selection and within-mode execution, we develop COMET-GRPO (COordinated Mode-Execution Training), which decouples the trajectory-level advantage into separate signals for mode selection and within-mode execution, enabling more targeted credit assignment to each decision level. We further introduce progressive exploration, which gradually shifts training from externally assigned modes to autonomous
    
[^87]: BANGLAPLAN：一个面向孟加拉语基于约束的规划与决策的基准

    BANGLAPLAN: A Benchmark for Constraint-Based Planning and Decision-Making in Bangla

    [https://arxiv.org/abs/2610.19704](https://arxiv.org/abs/2610.19704)

    该论文提出了BANGLAPLAN——首个用于评估大型语言模型在孟加拉语显式约束条件下规划与决策能力的基准，包含1000道涵盖十个任务族、四种语言语域的问题，答案采用统一的JSON结构化格式，并包含可行与不可行问题。

    

    大型语言模型（LLM）越来越多地帮助用户安排预约、规划行程和分配有限资源。这些任务要求模型满足关于时间、成本、容量和顺序的约束，或者判定不存在可行解。孟加拉语是世界上使用人数第七多的语言，现有基准已覆盖许多语言和推理任务，但在显式约束下评估孟加拉语规划与决策能力的基准仍然缺失。我们提出了BANGLAPLAN，这是一个包含1000道题目的基准，涵盖十个任务族，内容植根于孟加拉国的日常生活和各类机构。它包含可行与不可行的问题，并涵盖四种语言语域：日常用语、正式用语、口语以及孟加拉语与英语混用语。答案涵盖八种结构化类型，包括日程表、路线、资源分配和有序步骤，并采用统一的JSON格式。我们通过语言模型辅助起草的方式构建该基准。

    arXiv:2610.19704v1 Announce Type: new  Abstract: Large language models (LLMs) increasingly help users arrange appointments, plan journeys, and allocate limited resources. These tasks require models to satisfy constraints on time, cost, capacity, and order, or determine that no feasible solution exists. Bangla is the world's seventh most widely spoken language, and existing benchmarks cover many language and reasoning tasks. However, benchmarks for evaluating planning and decision-making in Bangla under explicit constraints remain lacking. We introduce BANGLAPLAN, a benchmark of 1,000 problems across ten task families grounded in everyday life and institutions in Bangladesh. It includes feasible and infeasible problems in four language registers: everyday, formal, colloquial, and Bangla--English mixed. Answers cover eight structured types, including schedules, routes, allocations, and ordered steps, in a common JSON format. We construct the benchmark through language-model-assisted draf
    
[^88]: 共享符号，而非向量：跨语言模型的无训练事实通信

    Share the Symbol, Not the Vector: Training-Free Fact Communication Across Language Models

    [https://arxiv.org/abs/2610.19696](https://arxiv.org/abs/2610.19696)

    提出了SYNC——一种无需训练的符号索引通信接口，使异构语言模型通过共享词表的索引而非内部向量来传递事实信息，从而无需微调或学习翻译器即可实现跨模型协作。

    

    多智能体系统允许语言模型通过共享互补信息来协作解决任务。这些系统通常通过文本进行通信，而最近的方法也在探索交换内部模型状态。然而，跨架构共享这些状态通常需要学习到的翻译器或对模型兼容性的假设。我们提出了SYNC（符号索引神经通信，Symbol-Indexed Neural Communication），这是一种词索引接口，无需微调或学习翻译器即可连接异构模型。发送方仅根据它所能看到的描述，从共享词表中选择单词，并为每个对象发送其索引，而无需看到问题。接收方使用自己的词方向（即从其激活中一次性拟合得到的向量）将所描述的事实写入其隐藏状态。我们通过在两个对象之间交换一个属性，并要求对已更改和未更改的属性都给出正确答案，来测试选择性使用能力（摘要在此处被截断）。

    arXiv:2610.19696v1 Announce Type: new  Abstract: Multi-agent systems allow language models to solve tasks collaboratively by sharing complementary information. These systems typically communicate through text, while recent approaches also explore exchanging internal model states. However, sharing these states across architectures often requires learned translators or assumptions about model compatibility. We introduce SYNC (Symbol-Indexed Neural Communication), a word-indexed interface that connects heterogeneous models without fine-tuning or a learned translator. From descriptions only it can see, the sender selects words from a shared list and sends their indices for each object, without seeing the questions. The receiver writes the described facts into its hidden states using its own word directions, vectors fitted once from its activations. We test selective use by swapping one attribute between two objects and requiring correct answers about both changed and unchanged attributes b
    
[^89]: 面向策略优化的函数空间时间自蒸馏

    Function-Space Temporal Self-Distillation for Policy Optimization

    [https://arxiv.org/abs/2610.19687](https://arxiv.org/abs/2610.19687)

    本文提出单策略的函数空间策略优化框架FSPO，以稀疏的历史下一词元分布作为固定时间目标进行自蒸馏，无需维护持久教师模型即可提供稳定密集的词元级监督，并通过特权分歧校正与可靠性自适应锚定提升策略优化效果。

    

    可验证奖励强化学习（RLVR）提升了语言模型的推理能力，但标量结果只能为响应轨迹提供粗略的监督信号。自蒸馏通过将无辅助的策略与基于反馈信息的自教师进行比较，能够提供更密集的词元级指导；然而教师会随学生一同演化，导致监督目标不稳定，因而需要额外的正则化。我们提出了函数空间策略优化（FSPO），这是一个单策略框架，将当前的特权适应与函数空间时间整合相结合。FSPO将稀疏的历史下一词元分布存储为固定的时间目标，在无需维护持久教师模型的情况下实现时间上的分离。该框架进一步引入特权分歧校正以显式抑制局部被高估的动作，并引入可靠性自适应锚定，在自生成证据重复时强化参考监督（摘要原文在此处截断）。

    arXiv:2610.19687v1 Announce Type: cross  Abstract: Reinforcement Learning with Verifiable Rewards (RLVR) improves reasoning in language models, but scalar outcomes provide coarse supervision over response trajectories. Self-distillation offers denser token-level guidance by comparing an unassisted policy with a feedback-informed self-teacher, yet the teacher evolves with the student, creating unstable targets and motivating additional regularization. We introduce Function-Space Policy Optimization (FSPO), a single-policy framework that combines current privileged adaptation with function-space temporal consolidation. FSPO stores sparse historical next-token distributions as fixed temporal targets, providing temporal separation without maintaining a persistent teacher model. It further introduces privileged disagreement correction to explicitly suppress locally overvalued actions, reliability-adaptive grounding that strengthens reference supervision when self-generated evidence repeated
    
[^90]: Interlock：长程智能体中从局部正确性到任务完成

    Interlock: From Local Correctness to Task Completion in Long-Horizon Agents

    [https://arxiv.org/abs/2610.19686](https://arxiv.org/abs/2610.19686)

    本文提出Interlock任务合成框架，通过将任务建模为包含接地边、耦合超边和实现边的类型化完成图，弥合长程智能体“局部正确但任务未完成”的差距。

    

    大型语言模型智能体日益承担复杂环境中的长程任务，这些环境具有证据碎片化、状态部分完成以及跨资源和工件的依赖关系等特点。现有智能体可能在满足个别需求的同时未能维持这些关系，从而产生局部正确但不完整的结果。这些失败之所以出现，是因为任务完成取决于环境证据、后果性决策和所需效果之间的关系。为了解决这一完成差距，我们提出了Interlock，一个任务合成框架，它将每个任务表示为一个类型化的完成图。接地边将公共证据与决策连接起来，耦合超边在组件之间施加联合约束，实现边将决策与所有必需的下游效果连接起来。Interlock将该图编译为一个可重置的工作区、一个公共请求和一个评估评分标准。（注：原摘要在"A source"处被截断）

    arXiv:2610.19686v1 Announce Type: new  Abstract: Large language model agents increasingly undertake long-horizon tasks in complex environments characterized by fragmented evidence, partially completed states, and dependencies across resources and artifacts. Existing agents may satisfy individual requirements while failing to maintain these relationships, producing locally correct but incomplete outcomes. These failures arise because task completion depends on relationships among environmental evidence, consequential decisions, and required effects. To address this completion gap, we propose Interlock, a framework for task synthesis that represents each task as a typed completion graph. Grounding edges connect public evidence to decisions, coupling hyperedges impose joint constraints across components, and realization edges connect decisions to all required downstream effects. Interlock compiles this graph into a resettable workspace, a public request, and an evaluation rubric. A source
    
[^91]: 超越分布匹配：面向监督微调的选择性边际保持方法

    Beyond Distribution Matching: Selective Margin Preservation for Supervised Fine-Tuning

    [https://arxiv.org/abs/2610.19685](https://arxiv.org/abs/2610.19685)

    提出SRP-SFT，一种通过铰链惩罚选择性保持非目标token边际关系的软边际目标，在不冻结参考分布的情况下显著提升了监督微调的数学推理与代码生成性能。

    

    监督微调（SFT）会在教师强制前缀下改变非目标token之间的logit边际。匹配局部参考分布能够保持这些概率，但同时也会惩罚边际的扩大。我们探究的问题是：一个软辅助目标能否在不冻结该参考分布的前提下，保持选定的局部非目标边际关系。我们提出了选择性秩保持SFT（SRP-SFT），这是一种软边际目标：对来自参考分布top-K非目标token的token对，施加基于参考分布缩放下界的铰链惩罚，并应用于固定的教师强制前缀。一旦所要求的下界得到满足，铰链惩罚即停止。我们推导了违反边际所带来的KL代价，刻画了可行区域及其条件效用，并给出了降低总体边际违反的有限步条件。在Qwen3-4B-Base模型和UltraFeedback数据集上的单轮训练实验中，该目标在全部三个训练随机种子上均使MATH-500和代码生成得分超过交叉熵（CE）和LP-SFT方法。

    arXiv:2610.19685v1 Announce Type: cross  Abstract: Supervised fine-tuning (SFT) changes logit margins among non-target tokens at teacher-forced prefixes. Matching a local reference distribution keeps those probabilities and also penalizes margin expansions. We ask whether a soft auxiliary can preserve selected local non-target margin relations without freezing that distribution. We introduce \textbf{Selective Rank-Preserving SFT (SRP-SFT)}, a soft margin target: a hinge on reference-scaled lower bounds for pairs from the reference's top-$K$ non-target tokens, applied at fixed teacher-forced prefixes. The hinge stops once the requested bound is met. We derive the KL cost of violating a margin, characterize the feasible region and its conditional utility, and give a finite-step condition for reducing aggregate violations. In one-epoch experiments on Qwen3-4B-Base and UltraFeedback, this target raises MATH-500 and code-generation scores over CE and LP-SFT at all three training seeds. At s
    
[^92]: 测量并减少大语言模型智能体的不可用回复

    Measuring and Reducing Unusable Replies of LLM Agents

    [https://arxiv.org/abs/2610.19659](https://arxiv.org/abs/2610.19659)

    本文提出“回复可用性”概念并设计了基于读者模型的自动测量方法，发现LLM智能体超过一半的回复对请求者不可用，并通过让智能体明确陈述回复要求将不合格率从0.52降至0.43。

    

    人们将多步骤工作委托给大语言模型（LLM）智能体，并且通常只阅读智能体最终返回的回复。当前的评估方法关注任务完成度和报告的忠实性，却没有考虑请求者能否理解并使用这些回复。我们将这一属性称为“回复可用性”，并通过一个“读者模型”来测量它：该模型站在请求者的立场，检验回复能否仅根据请求者所见的信息被理解、是否使用请求者的语言、以及是否回答了请求。该测量方法与五位标注者的多数意见的一致程度处于单个标注者之间一致性的范围内。在对来自七个模型系列的八个模型的评估中，480个回复中有54%不合格，即使表现最好的模型也有30%不合格。不合格的回复大多指向请求者未曾见过的材料，或使用了智能体在工作过程中自创的标签。明确陈述对回复的要求可以将不合格率从0.52降至0.43，同时还能减少错误和遗漏。

    arXiv:2610.19659v1 Announce Type: new  Abstract: People delegate multi-step work to LLM agents and read only the reply that the agent hands back. Current evaluation scores task completion and the faithfulness of reports, and it leaves open whether the requester can understand and use the reply. We call this property reply usability and measure it with a reader model that takes the requester's position and tests whether the reply can be interpreted from what the requester has seen, is in the requester's language and answers the request. Its agreement with the majority of five annotators lies within the range of the individual annotators. Across eight models from seven families, 54% of 480 replies fail, and the best model still fails on 30%. Failed replies mostly point to material the requester has not seen or use labels that the agent coined during its work. Stating explicit requirements for the reply reduces the failure rate from 0.52 to 0.43 together with errors and omissions, and a s
    
[^93]: HACHIMI-II：从角色扮演到学习者行为建模，实现忠实的学生模拟

    HACHIMI-II: From Role-Play to Learner Behavior Modeling for Faithful Student Simulation

    [https://arxiv.org/abs/2610.19654](https://arxiv.org/abs/2610.19654)

    HACHIMI-II 将学生模拟从让大语言模型角色扮演转变为对学习者行为的第三人称预测，通过从学习者历史构建学生模型并用行为框架约束其下一步动作，从而忠实再现真实学习者的能力分布和典型错误。

    

    模拟学生使研究人员能够在真实学习者出现之前测试辅导系统和教学策略，但前提是模拟必须重现特定学习者在哪些地方成功、在哪些地方失败。常见做法是让大型语言模型扮演学生角色，但角色设定并不能刻画这个学习者知道什么、下一步会做什么，于是模型自身的能力填补了这一空缺。由此导致模拟学生的表现水平由模型本身决定，彼此之间差异很小，并且很少犯真实学习者会犯的错误。本文对学习者的行为进行建模，将学生模拟重新表述为对学习者下一步动作的第三人称预测。HACHIMI-II 从学习者的历史记录中构建学生模型，推断学习者在当前题目或辅导轮次中的状态，并将其编译成一个约束下一步动作的行为执行框架。在三个题目作答数据集上，模拟学生与真实学生能力之间的 Wasserstein-1 距离为 2.4–8.4（摘要原文在此处截断）。

    arXiv:2610.19654v1 Announce Type: new  Abstract: Simulated students let researchers test tutoring systems and teaching strategies before real learners are available, but only if the simulation reproduces where a particular learner succeeds and fails. The common approach asks a large language model to role-play a student, but a role does not model what this learner knows or will do next, so the model's own competence fills the gap. Simulated students perform at a level set by the model, differ little from one another, and seldom make the real learners' errors. We model the learner's behavior and recast student simulation as third-person prediction of the learner's next action. HACHIMI-II builds a student model from the learner's history, infers the learner's state for the current item or tutor turn, and compiles it into a behavior harness that constrains the next action. On three item-response datasets, the Wasserstein-1 distance between simulated and real student ability is 2.4--8.4$\t
    
[^94]: 类型化决策模型的选择性推理：何时开始、何时读取、何时停止

    Selective Reasoning for Typed Decision Models: When to Start, Read, and Stop

    [https://arxiv.org/abs/2610.19644](https://arxiv.org/abs/2610.19644)

    该论文在冻结骨干模型上训练共享候选答案读取器，通过收益门控启动推理、自然状态触发器显式读取、熵规则停止生成，实现按需可中断的选择性推理，将决策准确率从 70.96% 提升至 84.38%，同时相比终端读取最多减少 46.30% 的流水线时间。

    

    类型化决策模型需要一个既能利用额外推理、又能保留候选输出的接口。我们在冻结的 Qwen3.5-4B 骨干模型上训练了一个共享的候选答案读取器，将直接决策与按需、可中断的推理统一起来。该读取器从空推理前缀和部分推理前缀中返回分布。一个收益门控决定何时启动推理，一个自然状态触发器请求显式读取，一条熵规则在接受已获得的答案后取消生成。测量指标区分了答案查询工作量、生成输出和流水线时间。在来自 ProofWriter、LogiQA 2.0 和 ARC-Challenge 的 768 个问题上，准确率从直接读取的 70.96% 提升至 84.38%。相对于 512 token 的终端读取，准确率提高了 7.29 个百分点，而平均输出工作量仅增加 2.25%。相对于 1,024 token 的终端读取，流水线时间减少了 46.30%，准确率差异为 -1.17 个百分点。精确前缀对照实验隔离了……

    arXiv:2610.19644v1 Announce Type: new  Abstract: Typed decision models need an interface that can use additional reasoning while retaining candidate outputs. We train a shared candidate-answer reader on a frozen Qwen3.5-4B backbone to unify direct decisions with on-demand, interruptible reasoning. The reader returns distributions from empty and partial reasoning prefixes. A benefit gate starts reasoning, a natural-state trigger requests explicit reads, and an entropy rule accepts acquired answers and cancels generation. Measurements distinguish answer-query work, generated output, and pipeline time. Across 768 questions from ProofWriter, LogiQA 2.0, and ARC-Challenge, accuracy rises from 70.96% for direct reading to 84.38%. Relative to 512-token terminal reading, accuracy improves by 7.29 percentage points with 2.25% more mean output work. Relative to 1,024-token terminal reading, pipeline time falls by 46.30% with an accuracy difference of -1.17 points. Exact-prefix controls isolate t
    
[^95]: 建模关键内容：面向语言智能体的决策引导式选择性世界建模

    Model What Matters: Decision-Guided Selective World Modeling for Language Agents

    [https://arxiv.org/abs/2610.19642](https://arxiv.org/abs/2610.19642)

    提出DGS-WM方法，通过决策引导的观测选择机制让语言智能体的世界建模只聚焦于具有决策价值的观测内容，从而弥补稀疏任务奖励无法监督环境动态理解的局限。

    

    强化学习为大型语言模型驱动的智能体提供了一种直接优化任务目标的机制，但任务奖励通常是稀疏的轨迹级信号，仅用于评估任务是否成功，无法直接监督动作条件下的环境状态转移，导致模型对动作后果和环境动态的显式理解十分有限。近期的工作通过策略与世界建模协同训练的方式引入辅助监督，但通常将完整的环境观测不加区分地统一作为世界建模目标，忽视了观测内容在决策相关性上的差异。我们提出了DGS-WM（决策引导式选择性世界建模），这是一种面向语言智能体的决策引导式策略-世界建模协同训练方法。DGS-WM采用一个决策引导的观测选择组件，基于观测单元的决策价值筛选出具有决策价值的观测单元作为世界建模目标……

    arXiv:2610.19642v1 Announce Type: cross  Abstract: Reinforcement learning provides a direct mechanism for optimizing task objectives in large language model-driven agents, but task rewards are typically sparse trajectory-level signals that only evaluate task success, without directly supervising action-conditioned environment transitions, leaving the model with limited explicit understanding of action consequences and environment dynamics. Recent work introduces auxiliary supervision through policy and world modeling co-training, yet typically treats complete environmental observations uniformly as world modeling targets, overlooking the varying decision relevance of observation content. We propose DGS-WM (Decision-Guided Selective World Modeling), a decision-guided policy-world modeling co-training approach for language agents. DGS-WM employs a Decision-Guided Observation Selection component that filters observation units with decision value as world modeling targets based on their su
    
[^96]: J-tau：一个完全本地化的日语工具-智能体-用户交互基准

    J-tau: A Fully Localized Japanese Benchmark for Tool-Agent-User Interaction

    [https://arxiv.org/abs/2610.19638](https://arxiv.org/abs/2610.19638)

    J-tau是首个公开可用的完全本地化日语基准，通过机器翻译、实体适配和人工审核对tau-bench进行本地化以评估使用工具的LLM智能体，并在此过程中发现并修复了原始tau-bench中的无效任务和分数聚合错误两个缺陷。

    

    我们提出了J-tau，这是首个公开可用的日语基准，用于在交互式客户支持场景中评估使用工具的LLM智能体。虽然tau-bench被广泛用于评估LLM智能体，但它以英语为中心，无法评估其他语言的智能体。现有的多语言扩展没有提供公开的日语版本，且主要依赖机器翻译，导致地址和货币等特定地区实体与目标区域不一致。为了实现日语环境下的可靠评估，我们通过机器翻译、实体适配和人工审核，对tau-bench进行了细致的本地化。在此过程中，我们还发现并修复了原始tau-bench中的两个缺陷：无效任务和一个会夸大某些模型结果的分数聚合错误。实验表明，J-tau比简单机器翻译的对应版本产生更少的评估错误，并且我们的修复解决了这两个缺陷。

    arXiv:2610.19638v1 Announce Type: new  Abstract: We present J-tau, the first publicly available Japanese benchmark for evaluating tool-using LLM agents in interactive customer-support scenarios. While tau-bench is widely used to evaluate LLM agents, it is English-centric and cannot evaluate agents in other languages. Existing multilingual extensions offer no public Japanese version and rely primarily on machine translation, leaving locale-specific entities such as addresses and currencies inconsistent with the target region. To enable reliable evaluation in Japanese, we carefully localize tau-bench through machine translation, entity adaptation, and human review. During this process, we also uncover and fix two flaws in the original tau-bench: invalid tasks and a score aggregation error that inflates certain models' results. Experiments show that J-tau yields fewer evaluation errors than a naively machine-translated counterpart and that our fixes resolve both flaws.
    
[^97]: 稀疏自编码器是符号匹配的共享坐标系

    Sparse Autoencoders Are a Shared Coordinate System for Symbolic Matching

    [https://arxiv.org/abs/2610.19636](https://arxiv.org/abs/2610.19636)

    本文提出SAE-IDF方法，利用稀疏自编码器将符号编码为稀疏表示并根据稀有度重新加权，实现了跨异构符号系统的单元匹配，为通过数据驱动方式对齐与合并现有符号系统、构建大规模神经符号模型迈出了关键的第一步。

    

    大语言模型（LLM）在一组权重中蕴含了广泛的知识，但缺乏可解释性和可信赖的推理能力，尤其是在高风险的应用场景中。神经符号模型凭借其可解释的架构和系统2（System 2）推理能力，如果能构建到LLM的规模，将是一种有前景的替代方案；然而，其符号骨干通常需要人工构建，因此现有的神经符号模型受限于符号组件的覆盖范围，通常只能应用于特定领域。一种数据驱动的方法则可以通过对齐和合并现有的符号系统来构建更大的符号系统。我们瞄准了这一方向的第一步基础性工作：在这些异构系统之间进行单元匹配。我们的方法SAE-IDF使用稀疏自编码器（SAE）将符号在其原生语境中编码为稀疏代码，根据每个维度的稀有度进行重新加权，并在不同系统之间进行匹配。在涵盖6种符号系统的12个任务上，SAE-IDF的表现优于基线方法。

    arXiv:2610.19636v1 Announce Type: cross  Abstract: Large language models (LLMs) hold broad knowledge in one set of weights, but lack interpretability and trustworthy reasoning, especially in high-stakes use cases. A neuro-symbolic model, with its interpretable architecture and System 2 reasoning capacity, is a promising alternative if built at the scale of LLMs; however, its symbolic backbone is usually constructed with human effort, so existing neuro-symbolic models are bottlenecked by the scope of the symbol component and usually domain-specific. A data-driven method could instead build a larger symbolic system by aligning and merging existing ones. We target its first, foundational step: matching units across these heterogeneous systems. Our method, SAE-IDF, encodes a symbol in its native context into a sparse code with a sparse autoencoder (SAE), re-weights each dimension by its rarity and matches across systems. On 12 tasks across 6 kinds of symbolic system, SAE-IDF outperforms th
    
[^98]: DAGFormer：面向Transformer跨层连接的全局拓扑预测

    DAGFormer: Global Topology Prediction for Transformer Cross-Layer Connections

    [https://arxiv.org/abs/2610.19625](https://arxiv.org/abs/2610.19625)

    DAGFormer通过联合训练的拓扑预测器一次性规划整个Transformer模型的全局跨层连接拓扑，在75M至1B多个规模上将困惑度降低17%至25%，并全面超越逐层路由方法。

    

    我们提出DAGFormer，一种由联合训练的拓扑预测器来规划跨层连接的Transformer语言模型。在标准Transformer残差连接之外，DAGFormer增加了从较早层到较晚层的直接连接，预测器基于输入词元控制每条连接的强度。与现有跨层路由架构逐层单独设置连接的方式不同，DAGFormer一次性规划整个模型的路由。在参数量从75M到1B的匹配OLMo-2模型对上，DAGFormer比原始Transformer在WikiText-2困惑度上降低17%至25%；在50亿词元上训练的1B DAGFormer优于在100亿词元上训练的原始模型。在与四个路由基线使用相同数据和预算的对比中，它在75M、300M和1B三个规模上均取得最低困惑度。该增益还延续到13项常识、科学、阅读和数学评测任务上（平均高出2.0至2.5分）。

    arXiv:2610.19625v1 Announce Type: new  Abstract: We propose DAGFormer, a Transformer language model whose cross-layer connections are planned by a jointly trained topology predictor. Beyond the residual connections in a standard Transformer, DAGFormer adds direct connections from earlier layers to later ones, and the predictor controls the strength of every connection based on input tokens. Unlike existing cross-layer routing architectures, which set each layer's connections separately, DAGFormer plans the routing of the whole model at once. On matched OLMo-2 pairs from 75M to 1B parameters, DAGFormer lowers WikiText-2 perplexity by 17% to 25% over the vanilla Transformer, and a 1B DAGFormer trained on 5B tokens outperforms the vanilla model trained on 10B. Against four routing baselines with the same data and budget, it reaches the lowest perplexity at 75M, 300M and 1B. The gain carries over to 13 commonsense, science, reading and math evaluation tasks (2.0 to 2.5 points higher mean a
    
[^99]: 分岔路径：学习LLM智能体轨迹的结果预测性表示

    Divergent Paths: Learning an Outcome-Predictive Representation of LLM Agent Trajectories

    [https://arxiv.org/abs/2610.19614](https://arxiv.org/abs/2610.19614)

    提出Precog-RM模型，通过元学习在共享嵌入空间中以少样本方式预测LLM智能体轨迹的成败，并可作为GRPO训练的早停控制器，在保持性能的同时最多减少47.3%的训练计算量。

    

    在这项工作中，我们将智能体轨迹重新定义为在动作与环境观测的共享嵌入空间中的路径。我们证明，同一任务的成功与失败rollout在该空间中是可分离的，使用简单的逐任务判别器即可在三个智能体基准和三个策略模型上预测留出的rollout结果，AUROC最高达0.95。为了跨任务利用这一信号，我们引入了Precog-RM，这是一个基于Transformer的模型，通过元学习获得一种表示，使结果预测可以在少样本设置下进行。我们证明Precog-RM能够泛化到未见过的任务和模型，并在轨迹终止前很久就能对其成败给出高置信度预测。作为Tau2-Bench上组相对策略优化（GRPO）的早停控制器部署时，Precog-RM将训练FLOPs减少了高达47.3%，同时保持与基线GRPO相当的性能。

    arXiv:2610.19614v1 Announce Type: cross  Abstract: In this work, we reframe agentic trajectories as paths through a shared embedding space of actions and environmental observations. We show that successful and failed rollouts of the same task separate in this space, with simple per-task discriminators predicting held-out rollout outcomes at up to 0.95 AUROC across three agent benchmarks and three policy models. To exploit this signal across tasks, we introduce Precog-RM, a transformer-based model that meta-learns a representation in which outcome prediction can be performed in a few-shot setting. We show that Precog-RM generalizes to unseen tasks and models, and gives confident predictions of trajectory success long before termination. Deployed as an early stopping controller for Group Relative Policy Optimization (GRPO) on Tau2-Bench, Precog-RM cuts training FLOPs by up to 47.3\% while maintaining performance comparable to baseline GRPO.
    
[^100]: 我们真的在测量遗忘吗？多模态大语言模型机器遗忘评估中的陷阱

    Are We Really Measuring Forgetting? Pitfalls in Evaluating Machine Unlearning for Multimodal LLMs

    [https://arxiv.org/abs/2610.19613](https://arxiv.org/abs/2610.19613)

    该论文指出了多模态大语言模型机器遗忘评估中存在的三大陷阱——缺乏保留集评估区分概念级遗忘与实例级过拟合、依赖合成知识而非预训练真实知识、以及不公平的方法比较——并据此提出了基于真实世界知识和原则性评估协议的GEMMU基准。

    

    随着多模态大语言模型（MLLMs）的机器遗忘技术快速发展，对其有效性的严格评估变得日益重要。然而，现有的基准测试大多沿用了纯文本遗忘的评估协议，忽视了多模态环境下遗忘所面临的独特挑战。在本工作中，我们识别出三个可能导致误导性结论的主要陷阱：缺乏保留集评估，无法区分真正的概念级遗忘与实例级过拟合；依赖合成的、通过微调注入的知识，而非预训练期间获取的真实世界知识；以及对不同遗忘方法应用相同训练设置所导致的不公平方法比较。为了应对这些挑战，我们提出了GEMMU（Generalizable Evaluation for Multimodal Machine Unlearning，多模态机器遗忘的泛化评估），这是一个基于真实世界知识和原则性评估协议构建的基准。通过重新评估代表性方法……（原文摘要在此处截断）

    arXiv:2610.19613v1 Announce Type: new  Abstract: As machine unlearning for multimodal large language models (MLLMs) rapidly advances, rigorous evaluation of its effectiveness becomes increasingly important. However, existing benchmarks largely inherit evaluation protocols from text-only unlearning and overlook the unique challenges of unlearning in multimodal settings. In this work, we identify three major pitfalls that may lead to misleading conclusions: the lack of heldout evaluation to distinguish genuine concept-wise forgetting from instance-wise overfitting, the reliance on synthetic, fine-tuned knowledge rather than real-world knowledge acquired during pre-training, and unfair method comparisons caused by applying the same training settings to different unlearning methods. To address these challenges, we introduce GEMMU (Generalizable Evaluation for Multimodal Machine Unlearning), a benchmark built on real-world knowledge and a principled evaluation protocol. Re-evaluating repres
    
[^101]: 二十天后：对Gemini回答中基于历史的品牌提及预测的预注册检验

    Twenty Days Later: A Pre-Registered Test of History-Based Forecasts of Brand Mentions in Gemini Answers

    [https://arxiv.org/abs/2610.19608](https://arxiv.org/abs/2610.19608)

    该论文通过预注册的重复实验证明，基于历史数据对Gemini回答中品牌提及率的预测在约20天后依然准确（Brier得分0.0281，接近8小时短期预测水平且优于品牌历史基线），表明大语言模型的品牌提及计数具有足够的时间稳定性，可供实际决策使用。

    

    arXiv:2610.19608v1 公告类型：交叉（cross）摘要：大型语言模型提及哪些品牌的计数，只有在其能维持足够长时间以供采取行动时才有用。我们通过一项预注册的重复实验来检验这一点。相同的162个请求（54个问题，涉及三个软件类别，分别以英语、爱沙尼亚语和芬兰语提出，每个问题包含12个候选品牌中的4个，重复三次）于2026年9月22日至23日在约十小时内分三波发送给gemini-3.1-pro-preview，并于2026年10月13日（即第三波之后20.3天）以完全相同的请求体再次发送。实验计划及其决策规则在首次调用之前被冻结、哈希并注册在OSF上。结果指标是每个答案与品牌的词汇提及标签，每波648个。主要预测——每个品牌-问题单元格在之前九个答案中的提及率——取得了Brier得分0.0281和对数损失0.126的成绩，相比之下，8小时波次的六答案预测Brier得分为0.0259，注册的上限为0.0559。它优于品牌历史基线（0.1……（原文摘要在此处截断）

    arXiv:2610.19608v1 Announce Type: cross  Abstract: Counts of which brands a large language model names are useful only if they hold long enough to act on. We test this with a pre-registered replication. The same 162 requests (54 questions in English, Estonian and Finnish about three software categories, four of twelve candidate brands per question, three repeats) were sent to gemini-3.1-pro-preview in three waves within about ten hours on 22 and 23 September 2026, and again on 13 October 2026, 20.3 days after the third wave, with identical request bodies. The plan and its decision rule were frozen, hashed and registered on OSF before the first call. The outcome is a lexical mention label per answer and brand, 648 per wave. The primary forecast, each brand-and-question cell's mention rate over the nine earlier answers, scored Brier 0.0281 and log loss 0.126, against Brier 0.0259 for the six-answer forecast of the 8-hour wave and a registered ceiling of 0.0559. It beat brand history (0.1
    
[^102]: PN-Mix：面向高效长输出大语言模型推理的配对范数引导混合精度键缓存量化

    PN-Mix: Pair-Norm Guided Mixed-Precision Key Cache Quantization for Efficient Long-Output LLM Inference

    [https://arxiv.org/abs/2610.19605](https://arxiv.org/abs/2610.19605)

    PN-Mix通过RoPE通道对层面的查询-键配对范数分析识别量化敏感的键缓存通道对，并在预填充阶段一次性识别、解码全程复用，为其分配更高精度，从而实现高效且低精度损失的长输出大语言模型推理。

    

    大语言模型的长输出推理受到严重的KV缓存内存与带宽瓶颈的制约。虽然KV缓存量化可以缓解这一问题，但低比特量化（例如2比特）会严重损害模型精度。近期的混合精度键缓存量化方法通过为重要的键通道分配更高精度来缓解这种精度下降，但其仅利用键自身的统计信息来估计通道显著性。通过在RoPE通道对层面进行注意力logit扰动分析，我们证明了在查询-键配对范数乘积较大的通道对中，量化误差对logit变化有显著影响。基于这一洞察，我们提出了PN-Mix，一种面向基于RoPE的仅解码器（decoder-only）大语言模型的混合精度键缓存量化方法，该方法为显著通道对分配更高精度，这些通道对在预填充阶段仅识别一次，并在整个解码过程中复用。在标准推理、代码生成和AIME基准……

    arXiv:2610.19605v1 Announce Type: cross  Abstract: Long-output inference in large language models suffers from severe KV cache memory and bandwidth bottlenecks. While KV cache quantization alleviates this issue, low-bit quantization (e.g., 2-bit) severely degrades model accuracy. Recent mixed-precision key-cache quantization methods mitigate this degradation by allocating higher precision to important key channels, but estimate channel saliency using only key statistics. Through attention logit perturbation analysis at the RoPE channel-pair level, we show that quantization errors in channel pairs with large query--key pair-norm products contribute significantly to logit variation. Based on this insight, we propose PN-Mix, a mixed-precision key-cache quantization method for RoPE-based decoder-only LLMs, which allocates higher precision to salient channel pairs identified once during the prefill stage and reused throughout decoding. Across standard reasoning, code-generation, and AIME be
    
[^103]: PA1-OPD：用于语言模型预训练的单步前缀增强在线策略蒸馏

    PA1-OPD: One-Step Prefix-Augmented On-Policy Distillation for Language Model Pretraining

    [https://arxiv.org/abs/2610.19601](https://arxiv.org/abs/2610.19601)

    PA1-OPD通过在每个前缀处引入学生采样转移后的教师反馈，扩展了知识蒸馏的监督状态覆盖范围，同时利用数据锚定根节点和可微树注意力实现高效并行训练，在多个数据集的继续预训练中显著优于传统知识蒸馏方法。

    

    知识蒸馏（KD）通过在语料库前缀处匹配教师模型分布来改进语言模型预训练，但对学生预测所诱导的上下文不提供直接监督。我们提出PA1-OPD，它在每个选定前缀处进行一次学生采样的状态转移后，引入教师反馈来增强语料路径蒸馏。这在保留下一词预测和语料路径知识蒸馏的同时，扩展了受监督的状态集合。数据锚定的根节点支持并行采样，可微的树注意力机制共享前缀计算且不改变推理过程。在两个继续预训练设置中，全覆盖率方案在C4、PG-19和Pile数据集上的每个评估检查点都取得了比KD更好的保留交叉熵。在Stage2阶段，选择学生困惑度最高的10%根节点，使KD相对于下一词训练的Macro交叉熵增益额外提升了48.85%。当分支监督排除已在语料路径上的后继状态时，增益依然保持。

    arXiv:2610.19601v1 Announce Type: new  Abstract: Knowledge distillation (KD) improves language-model pretraining by matching teacher distributions at corpus prefixes, but provides no direct supervision at contexts induced by student predictions. We introduce PA1-OPD, which augments corpus-path distillation with teacher feedback after one student-sampled transition per selected prefix. This expands the supervised states while retaining next-token prediction and corpus-path KD. Data-anchored roots support parallel sampling, and differentiable tree attention shares prefix computation without changing inference. In two continued-pretraining settings, full coverage improves held-out cross-entropy over KD across C4, PG-19, and Pile at every evaluated checkpoint. In Stage2, selecting the 10% of roots with highest student surprisal adds 48.85% to KD's Macro cross-entropy gain over next-token training. Gains remain when branch supervision excludes successor states already on the corpus path, su
    
[^104]: 道德交锋：利用道德基础理论越狱大型语言模型

    Moral Combat: Exploiting Moral Foundations Theory to Jailbreak LLMs

    [https://arxiv.org/abs/2610.19587](https://arxiv.org/abs/2610.19587)

    该论文提出一种基于道德基础理论的多智能体框架，通过诊断有害请求违反的道德基础并将其重构为看似符合道德的表述来越狱大语言模型，揭示了LLMs极易被“道德化”手段绕过安全防护栏的安全漏洞。

    

    社会心理学长期以来描述了人们如何被说服将有害行为解读为道德行为。最近的研究表明，大型语言模型（LLMs）会反映人类心理的诸多方面，从认知偏差到社会从众。这引出了一个具有直接安全影响的问题：能否通过“道德化”手段使LLMs绕过安全防护栏？我们通过一个基于道德基础理论（MFT）的多智能体框架回答了这一问题。对于一个最初被拒绝的有害请求，我们的智能体首先诊断该提示词违反了哪种道德基础（MF），然后识别出一个可辩护的道德基础，在该基础下同一请求显得具有美德，据此重写提示词，并利用生成的类似道德重构先例加以强化。我们在五个开源和专有模型上，使用三个知名越狱数据集中的有害提示词对该框架进行了评估。我们的分析表明，模型极易受到道德重构攻击的影响。

    arXiv:2610.19587v1 Announce Type: new  Abstract: Social psychology has long described how people can be persuaded to interpret a harmful act as moral. Recent research suggests that Large Language Models (LLMs) echo aspects of human psychology, from cognitive biases to social conformity. This raises a question with direct safety implications: can LLMs be moralized to bypass safety guardrails? We answer this question with a multi-agent framework grounded in Moral Foundations Theory (MFT). Given a harmful request that is initially refused, our agents diagnose which moral foundation (MF) the prompt is violating, identify a justifiable foundation under which the same request appears virtuous, rewrite the prompt accordingly, and reinforce it with generated precedents of comparable moral reframing. We evaluate our framework on harmful prompts from three well-known jailbreak datasets across five open-source and proprietary models. Our analysis reveals that models are highly susceptible to mora
    
[^105]: 文本背后是谁？鲁棒的模型作者归属

    Who's Behind the Text? Robust Model Authorship Attribution

    [https://arxiv.org/abs/2610.19583](https://arxiv.org/abs/2610.19583)

    提出基于似然的模型作者归属方法ATLAS，能够在未见领域、未知写作场景以及模型家族未来版本等分布偏移下依然稳健地识别文本的来源大语言模型。

    

    随着大语言模型越来越多地被用于生成文本，其滥用已成为一个日益严重的社会问题，涉及学术不端、模型蒸馏和网络攻击等方面。因此，模型作者归属（MAA）——即识别给定文本的来源模型——对于促进大语言模型的负责任使用和提升人工智能安全具有重要意义。在实际应用中，归属系统必须在训练时观察到的分布之外仍然保持可靠，它们可能会遇到来自未见过的领域和写作场景的文本，以及来自系统训练时尚不存在的模型家族未来版本所生成的文本。然而，现有的MAA方法在这种分布偏移下往往会出现显著的性能下降。为应对这一挑战，我们提出了一种基于似然的来源归属作者追踪方法ATLAS，它通过以来源为条件的……（摘要在此处被截断）

    arXiv:2610.19583v1 Announce Type: cross  Abstract: As large language models (LLMs) are increasingly used to generate text, their misuse has become a growing societal problem, with implications for academic misconduct, model distillation, and cyberattacks. For these reasons, Model Authorship Attribution (MAA), which identifies the source model of a given text, is important for promoting the responsible use of LLMs and improving AI safety. In practice, attribution systems must remain reliable beyond the distributions observed during training. They may encounter texts from unseen domains and writing settings, as well as generations from future versions of model families that were not available when the system was trained. However, existing MAA methods often exhibit substantial performance degradation under such distribution shifts. To address this challenge, we propose Authorship Tracing via Likelihood-based Attribution of Sources (ATLAS), which formulates MAA through source-conditioned t
    
[^106]: INTACT：在冻结骨干模型上诱导潜在推理

    INTACT: Inducing Latent Reasoning on a Frozen Backbone

    [https://arxiv.org/abs/2610.19568](https://arxiv.org/abs/2610.19568)

    INTACT 通过小型外部编码器结合蒸馏与强化学习，在完全冻结的大模型上诱导潜在推理，在不改变模型行为的前提下显著提升数学推理准确率并减少自回归解码开销。

    

    潜在推理用紧凑的连续“软词元”表示取代自然语言的思维链，减少了必须自回归解码的中间词元数量。我们使用一个小型外部编码器在冻结的骨干模型中诱导潜在推理。该编码器迭代地将骨干模型的隐藏状态映射为新的潜在块，构建一个潜在推理前缀，之后冻结的模型以文本形式写出其余的推理过程和答案。我们的方法 INTACT 通过蒸馏训练该编码器，使其再现书面推理步骤的效果，同时保持模型行为接近冻结的骨干模型，然后再根据答案正确性对其进行强化学习。在从 1B 到 8B 的四个冻结骨干模型上，INTACT 同时提升了 GSM8K 和平均域外准确率，包括在 Llama-3.2-3B 上 GSM8K 从 75.8% 提升至 79.3%，在 Qwen2.5-3B 上从 77.6% 提升至 79.7%。它还降低了端到端（原文摘要在此处截断）

    arXiv:2610.19568v1 Announce Type: new  Abstract: Latent reasoning replaces natural-language chains of thought with compact continuous "soft-token" representations, reducing the number of intermediate tokens that must be decoded autoregressively. We use a small external encoder to induce latent reasoning in a frozen backbone model. The encoder iteratively maps the backbone's hidden states to new latent blocks, constructing a latent reasoning prefix, after which the frozen model writes its remaining reasoning and its answer as text. Our method, INTACT, trains this encoder through distillation to reproduce the effect of written reasoning steps while keeping the model's behavior close to that of the frozen backbone, and then reinforces it on answer correctness. Across four frozen backbones from 1B to 8B, INTACT improves both GSM8K and average out-of-domain accuracy, including GSM8K gains from 75.8% to 79.3% on Llama-3.2-3B and from 77.6% to 79.7% on Qwen2.5-3B. It also reduces end-to-end l
    
[^107]: REWIND：面向测试时推理的不确定性引导轨迹修复

    REWIND: Uncertainty-Guided Trajectory Repair for Test-Time Reasoning

    [https://arxiv.org/abs/2610.19561](https://arxiv.org/abs/2610.19561)

    研究发现错误推理伴随显著更高的token熵，据此提出无需训练的REWIND方法，利用熵信号定位并重新生成推理轨迹中的错误后缀，在多个模型和基准上相比最强基线提升0.51-10.00个百分点。

    

    在生成的推理过程中，早期出现的错误可能会限制所有后续的推理。重复生成方法，如基于MCMC的方法，试图通过采样多个生成结果来提高输出质量，但其效果可能因随机选择重新生成的位置而受限。在MATH500数据集上使用Qwen2.5-Math-7B模型，我们发现错误响应的平均token熵显著高于正确响应，同一问题内的平均相对增幅约为112%，这表明熵可以作为推理错误的信号。受这一发现启发，我们提出了REWIND，这是一种无需训练的控制器，利用熵的动态变化来提出有针对性的后缀重新生成。在三个模型系列和五个基准测试上，REWIND优于所有基线方法，相比最强基线获得了0.51-10.00个百分点的提升。这些结果证明了有针对性的、不确定性引导的轨迹修复的有效性。

    arXiv:2610.19561v1 Announce Type: new  Abstract: An early mistake in the generated reasoning can constrain all subsequent reasoning. Repeated-generation methods, like MCMC-based approaches, attempt to improve output quality by sampling multiple generations, but their effectiveness can be limited by randomly selecting where to regenerate. On MATH500 with Qwen2.5-Math-7B, we find that incorrect responses have substantially higher mean token entropy than correct responses, with an average within-question relative increase of approximately 112%, suggesting that entropy can signal reasoning errors. Motivated by this finding, we introduce REWIND, a training-free controller that uses entropy dynamics to propose targeted suffix regenerations. Across three model families and five benchmarks, REWIND outperforms all baselines, with gains of 0.51-10.00 percentage points over the strongest baseline. These results demonstrate the effectiveness of targeted, uncertainty-guided trajectory repair across
    
[^108]: τ-Emotion-Bench：通过真实世界用户情绪揭示大语言模型智能体的交互脆弱性

    $\tau$-Emotion-Bench: Exposing the Interaction Fragility of LLM Agents through Real-World User Emotions

    [https://arxiv.org/abs/2610.19560](https://arxiv.org/abs/2610.19560)

    提出τ-Emotion-Bench十领域基准和Tracer可扩展数据生成流水线，通过模拟真实用户的情绪化交互与终止低效对话的机制，评估并提升LLM工具调用智能体在面对情绪化用户时的交互可靠性。

    

    在长程工具调用任务中，LLM智能体的失败不仅可能源于工具使用错误，还可能源于与用户的低效交互。然而，现有基准大多假设用户是配合的且行为稳定的，导致情绪变化对智能体可靠性的影响研究不足。我们提出了τ-Emotion-Bench，这是一个涵盖十个领域的基准，用于在受控的情绪化用户交互条件下评估工具调用智能体，同时保持底层任务目标不变。为了同时扩展基准构建与智能体训练的规模，我们引入了Tracer——一个可扩展的智能体式数据生成流水线，能够构建有据可依、符合策略的任务，并通过交互式的“智能体-用户-环境”测试框架收集轨迹。通过允许不满意的用户终止低效交互，该测试框架提供了一种交互感知信号，用于过滤过度依赖用户的轨迹，并倾向于……

    arXiv:2610.19560v1 Announce Type: cross  Abstract: In long-horizon tool-calling tasks, LLM agents can fail not only because of tool-use errors, but also because of ineffective interaction with users. Yet existing benchmarks largely assume cooperative and behaviorally stable users, leaving the effect of emotional variation on agent reliability underexplored. We introduce $\tau$-Emotion-Bench, a ten-domain benchmark for evaluating tool-calling agents under controlled emotion-conditioned user interactions while preserving the underlying task objective. To scale both benchmark construction and agent training, we introduce Tracer, a scalable agentic data-generation pipeline that constructs grounded, policy-valid tasks and collects trajectories through an interactive agent--user--environment harness. By allowing dissatisfied users to terminate inefficient interactions, the harness provides an interaction-aware signal that filters trajectories with excessive user dependence and favors effecti
    
[^109]: 基于死亡调查报告的证据锚定药物过量死亡预测

    Evidence-Anchored Overdose Death Prediction from Death Investigation Reports

    [https://arxiv.org/abs/2610.19556](https://arxiv.org/abs/2610.19556)

    该论文提出了一个从死亡调查报告中预测疑似药物过量死亡的自动化框架，通过专家引导生成证据锚定的解释，并比较了流水线模型与自我合理化模型在该任务上的性能差异。

    

    arXiv:2610.19556v1 公告类型：新论文 摘要：在美国，由于涉及药物死亡的认证过程复杂，药物过量死亡的确认常常被延误。死亡调查报告是一种丰富但未被充分利用的数据源，可用于预测某例死亡是否为疑似药物过量（即在最终死因和死亡方式判定之前被初步认定为药物过量的死亡）。我们提出了一个自动化框架，能够从死亡调查报告中对疑似药物过量死亡进行分类，同时生成由专家知识引导的支持性解释。通过专家引导的设计过程，我们研究了以下两个问题：1）哪种类型的解释最有利于提升模型预测性能；2）哪种架构更适合该任务：是先进行合理化解释再预测的流水线模型，还是将预测与合理化解释联合进行的自我合理化模型？我们的实验表明，采用基于证据锚定的解释流水线模型（原文在此处截断）……

    arXiv:2610.19556v1 Announce Type: new  Abstract: Confirmation of overdose deaths in the United States is often delayed due to the complexities of certifying drug-involved deaths. Death investigation reports represent a rich but underutilized data source that could be used to predict whether a death is a suspected overdose (i.e., a death preliminarily identified as an overdose prior to the final cause and manner of death determination). We present an automated framework that classifies suspected overdoses from death investigation reports, while producing supporting explanations guided by expert knowledge. Through an expert-guided design process, we investigate 1) what type of explanations are best for model prediction performance, and 2) which architecture is better suited for this task: pipeline models that rationalize first and then predict, or self-rationalizing models that jointly predict and rationalize? Our experiments show that a pipeline model with explanations grounded in the i
    
[^110]: GraphRAG-LTM：一个针对图检索增强生成攻击的分层威胁建模框架

    GraphRAG-LTM: A Layered Threat Modeling Framework for Attacks on Graph Retrieval-Augmented Generation

    [https://arxiv.org/abs/2610.19549](https://arxiv.org/abs/2610.19549)

    该论文提出 GraphRAG-LTM 分层威胁建模框架，通过黑盒查询、白盒查询和图感知白盒三个递进攻击层级，首次系统揭示了 GraphRAG 证据生成路径的安全脆弱性。

    

    图检索增强生成（GraphRAG）利用从文档中构建的知识图谱来为大语言模型（LLM）的回答组装证据。其安全性既取决于针对查询检索到的种子文本块，也取决于由这些文本块组装而成的图谱证据，然而针对这一证据路径的攻击仍鲜有研究。我们提出了 GraphRAG-LTM，一个分层威胁建模框架，用于测试三个嵌套层级的攻击者访问权限。BB Query 利用排序反馈在语义相似度约束下对查询进行编辑，从而降低相关种子文本块的排名。WB Query 利用检索器的梯度实现同样的目标。图感知的 WB（Graph-aware WB）在保留白盒对抗性查询的基础上，从查询特定的证据图中删除桥接三元组，通过局部图编辑破坏多跳证据传播。在多个数据集、检索器和 LLM 骨干模型上的实验表明，仅凭排序反馈即可降低答案质量，梯度访问会进一步强化查询攻击，而删除桥接三元组……（摘要内容在此处截断）

    arXiv:2610.19549v1 Announce Type: new  Abstract: Graph retrieval-augmented generation (GraphRAG) uses document-derived knowledge graphs to assemble evidence for large language model (LLM) answers. Its security depends on both the seed chunks retrieved for a query and the graph evidence assembled from them, yet attacks on this evidence path remain underexplored. We introduce GraphRAG-LTM, a Layered Threat Modeling framework that tests three nested levels of attacker access. BB Query uses ranking feedback to make query edits under a semantic-similarity constraint and demote relevant seed chunks. WB Query uses retriever gradients for the same objective. Graph-aware WB keeps the white-box adversarial query and deletes bridge triplets from the query-specific evidence graph, disrupting multi-hop evidence propagation through localized graph edits. Across datasets, retrievers, and LLM backbones, rank feedback alone reduces answer quality, gradient access strengthens query attacks, and bridge-t
    
[^111]: 表格数据上的类型化决策模型

    Typed Decision Models on Tabular Data

    [https://arxiv.org/abs/2610.19536](https://arxiv.org/abs/2610.19536)

    类型化决策模型在表格数据任务上以更低成本达到与大语言模型相当的性能，且对问题表述更加鲁棒，但所有模型对表格组织方式的变化都不够稳健。

    

    类型化决策模型是一类语言模型，它们通过为一组固定的候选答案中的每个答案返回概率来回答问题，而不是生成文本。我们在表格数据上研究这类模型，将四种决策模型与三个大语言模型（LLM）在行分类、事实核查和异常检测任务上进行比较。每个请求都以原始形式发送，同时也在21种保持内容不变的改写下发送，例如重命名选项或以其他格式书写表格。决策模型的表现与LLM相当，但成本更低。其中三个模型对问题的表述方式具有更强的鲁棒性。然而，我们测试的所有模型对表格的组织方式都不鲁棒。对于每个模型，在表格或问题的某种改写下，至少有三分之一的行决策会发生变化。决策模型赋予低概率的决策是最容易发生改变的。决策模型是一种实用的选择。

    arXiv:2610.19536v1 Announce Type: new  Abstract: Typed decision models are language models that answer a question with a fixed set of possible answers by returning a probability for each answer instead of generating text. We study them on tabular data and compare four decision models with three large language models (LLMs) on row classification, fact verification and anomaly detection. Every request is sent in its original form and under 21 types of rewrite that keep its content, such as renaming the options or writing the table in another format. The decision models perform as well as the LLMs at a lower cost. Three of them are also far more robust to how the question is formulated. No model we test is robust to how the table is organized. For every model, at least a third of the row decisions change under some rewrite of the table or the question. The decisions to which a decision model gives low probability are the ones most likely to change. Decision models are a practical choice f
    
[^112]: 面向问题与辅导对话的统一响应框架

    A Unified Response Framework for Questions and Tutoring Dialogues

    [https://arxiv.org/abs/2610.19532](https://arxiv.org/abs/2610.19532)

    提出了一个统一框架，将学生对问题的回答与辅导对话中导师提出的子任务共同建模在项目反应理论（IRT）量表上，通过文本估计难度、追踪学生动态能力演化，并利用导师干预行为正则化难度值，实现对学业知识的更细粒度评估。

    

    现有教育应用中的学生模型主要关注学生对问题的回答。然而，随着辅导服务在大规模上日益普及，分析导师与学生之间的对话变得尤为重要，因为这些对话可能揭示关于学生知识的细粒度证据。在这项工作中，我们提出了一个统一框架，将学生对问题的回答以及辅导对话中导师提出的子任务建模在同一个项目反应理论（IRT）量表上。我们首先在辅导对话中识别数学子任务，将这些子任务改写为自包含的问题，并判定学生对这些子任务回答的正确性。随后，我们开发了一个IRT框架，从文本中估计问题/子任务的难度，追踪学生在问答和对话交互过程中不断变化的能力，并提出一种新颖的方法，利用导师在连续子任务之间的干预行为来正则化子任务的难度值。我们评估了我们的框架……

    arXiv:2610.19532v1 Announce Type: new  Abstract: Existing student models in educational applications largely focus on student responses to questions. However, as tutoring becomes more accessible at large scale, it is important to analyze dialogues between tutors and students, which may reveal fine-grained evidence on student knowledge. In this work, we propose a unified framework that models both student responses to questions and tutor-posed subtasks during dialogues on a shared item response theory (IRT) scale. We first identify mathematical subtasks in tutoring dialogues, rewrite them into self-contained questions, and identify the correctness of student responses to these subtasks. We then develop an IRT framework that estimates question/subtask difficulty from text, tracks students' evolving abilities across question-answering and dialogue interactions, and a novel method to leverages tutor moves between consecutive subtasks to regularize their difficulty values. We evaluate our f
    
[^113]: 只在有帮助时检索有帮助的内容：面向选择性检索增强机器翻译的智能体强化学习

    Retrieve What Helps, Only When It Helps: Agentic Reinforcement Learning for Selective Retrieval-Augmented Machine Translation

    [https://arxiv.org/abs/2610.19528](https://arxiv.org/abs/2610.19528)

    该论文提出SRAT智能体强化学习框架，让大语言模型在翻译中自主学习何时需要检索外部知识、检索什么证据以及如何整合，从而只在检索真正有助于翻译质量时才进行检索。

    

    检索外部知识可以通过提供诸如习语解释和术语等补充信息来提升大语言模型（LLM）的翻译质量。然而，现有的检索增强机器翻译系统通常对所有输入统一应用预定义的检索策略。我们认为翻译处于一种“部分证据”机制中：LLM的参数化知识足以应对许多输入，而外部证据只对其中一部分输入有益，且通常是辅助性的，而非直接提供所需的译文。这带来了两个挑战：决定何时检索以及获取何种证据。我们提出了选择性检索增强翻译（SRAT），这是一个智能体式强化学习框架，通过优化最终翻译质量，学习何时直接翻译、何时进行检索、检索什么内容，以及如何整合这些决策。为了使证据检索与下游翻译质量对齐…

    arXiv:2610.19528v1 Announce Type: new  Abstract: Retrieving external knowledge can improve large language model (LLM) translation by providing supplementary information such as idiom interpretations and terminology. However, existing retrieval-augmented machine translation systems typically apply predefined retrieval strategies uniformly across all inputs. We argue that translation operates in a partial-evidence regime: LLM's parametric knowledge suffices for many inputs, while external evidence benefits a subset and usually supports rather than directly provides the desired translation. This creates two challenges: deciding when to retrieve and what evidence to acquire. We introduce Selective Retrieval-Augmented Translation (SRAT), an agentic reinforcement learning framework that learns when to translate directly or retrieve, what to retrieve, and how to integrate these decisions by optimizing final translation quality. To align evidence retrieval with downstream translation quality r
    
[^114]: 诊断与纠正可验证指令遵循中的奖励作弊

    Diagnosing and Correcting Reward Hacking in Verifiable Instruction Following

    [https://arxiv.org/abs/2610.19525](https://arxiv.org/abs/2610.19525)

    CALIF 是一种轻量级方法，通过利用现有冻结参考策略计算经校准的基于似然的奖励加成来增强验证器奖励，无需额外评判模型即可缓解可验证指令遵循中的奖励作弊问题并保持回复质量。

    

    可验证指令遵循基准强调输出约束的满足，但可能忽视回复是否切实地回答了底层请求，从而形成评估盲区。我们表明，前沿模型可以在几乎没有实质性答案内容的回复上获得满分。将同样的基于验证器的标准用作训练奖励会诱发奖励作弊：公开发布的检查点在经过可验证奖励的强化学习后，约束满足度更高，但回复质量更低。为解决这一奖励作弊问题，我们提出了 CALIF，这是一种轻量级方法，它通过一个经过校准的基于似然的奖励加成来增强验证器奖励，从而保持回复质量。CALIF 仅利用现有的冻结参考策略，在只有请求本身（不含约束）的条件下评估回复的似然，无需额外的评判模型或习得的奖励模型。在从 1.7B 到 8B 的推理与非推理模型上（摘要至此截断）。

    arXiv:2610.19525v1 Announce Type: cross  Abstract: Verifiable instruction-following benchmarks emphasize output-constraint satisfaction but can overlook whether responses meaningfully answer the underlying request, creating an evaluation blind spot. We show that frontier models can receive full credit for responses with little substantive answer content. Using the same verifier-based criterion as a training reward can induce reward hacking, as publicly released checkpoints exhibit higher constraint satisfaction but lower response quality after reinforcement learning with verifiable rewards. To address this reward hacking, we introduce CALIF, a lightweight method that augments verifier rewards with a calibrated likelihood-based bonus to preserve response quality. CALIF evaluates response likelihood under the request alone using the existing frozen reference policy, requiring no additional judge or learned reward model. Across models ranging from 1.7B to 8B in reasoning and non-reasoning
    
[^115]: 超前一个区块：在扩散语言模型中提升采样器状态

    One Block Ahead: Promoting Sampler State in Diffusion Language Models

    [https://arxiv.org/abs/2610.19510](https://arxiv.org/abs/2610.19510)

    提出了一种无需训练的OBA解码算法，通过在去噪当前区块的同时提前解码下一区块并提升相应的采样器状态（包括自条件信号和去噪时间），显著提升了扩散语言模型每次前向传播的token生成量。

    

    区块扩散语言模型可以并行地去噪整个token位置的区块，因此每次前向传播能够生成多个token。然而，区块之间仍然严格按顺序去噪，这一成本是区块内加速所无法解决的。本文研究了并发解码多个区块，并提出了一种无需训练的解码算法——One Block Ahead（OBA，超前一个区块），它在去噪当前区块的同时对下一个区块进行去噪。我们证明，对于均匀状态扩散语言模型（其区块的去噪进度在中间token中不可见），如果将下一个区块的token与相应的采样器状态（包括自条件信号和去噪时间）一并提升，并发解码是有利的。我们还推导出了一个盈亏平衡准则，将OBA的收益与成对区块前向传播的相对成本联系起来。在最先进的扩散语言模型DiffusionGemma上，OBA使每次前向传播生成的token数量提升了

    arXiv:2610.19510v1 Announce Type: new  Abstract: Block-diffusion language models denoise a whole block of token positions in parallel so each forward pass can emit many tokens. However, blocks are still denoised in strict sequence, a cost that within-block acceleration leaves untouched. Here, we investigate decoding blocks concurrently and propose One Block Ahead (OBA), a training-free decoding algorithm that denoises the next block alongside the current one. We demonstrate that for uniform-state diffusion LMs, where a block's denoising progress is not visible in its intermediate tokens, concurrent decoding is advantageous if the next block's tokens are promoted together with the corresponding sampler state, including the self-conditioning signal and the denoising time. We also derive a break-even criterion that relates the benefit of OBA to the relative cost of a paired-block forward pass. On DiffusionGemma, a state-of-the-art diffusion LM, OBA increases the tokens per forward pass by
    
[^116]: 基于篇章解析的结构保持型文档级数据增强

    Structure-Preserving Document-Level Data Augmentation via Discourse Parsing

    [https://arxiv.org/abs/2610.19497](https://arxiv.org/abs/2610.19497)

    提出了一种结合大语言模型与文档级篇章结构解析的数据增强框架，通过识别句间修辞关系并替换关键短语，在保留原文篇章结构的同时生成主题偏移的文档级增强数据，有效提升摘要与问答任务性能。

    

    数据增强在自然语言处理（NLP）的低资源或有限监督场景下对提升模型性能起着至关重要的作用。现有的数据增强方法，如同义词替换和回译，已在词级或句子级上展现出有效性；然而，它们往往忽视篇章层面的连贯性和逻辑流程，而这些对于依赖句间关系的复杂任务至关重要。在本文中，我们提出了一种基于大语言模型（LLMs）和文档级篇章结构解析的结构保持型文档级数据增强框架。我们的方法识别句子对之间的修辞关系并提取关键短语，然后将其替换为语义连贯但主题有所偏移的内容，同时保留原始的篇章结构。在文本摘要和问答任务上的实验结果表明，所提出的方法

    arXiv:2610.19497v1 Announce Type: new  Abstract: Data augmentation (DA) plays a vital role in improving model performance in low-resource or limited-supervision scenarios within Natural Language Processing (NLP). Existing DA methods, such as synonym replacement and back-translation, have demonstrated effectiveness on the word or sentence level; however, they frequently neglect discourse-level coherence and logical flow, which are essential for complex tasks dependent on inter-sentential relationships. In this paper, we propose a structure-preserving document-level data augmentation framework based on large language models (LLMs) and document-level discourse structure parsing. Our approach identifies rhetorical relations between sentence pairs and extracts key phrases, which are then replaced with semantically coherent but topically shifted content while preserving the original discourse structure. Experimental results on text summarization and question answering show that the proposed 
    
[^117]: 基于预训练文本到文本语言模型与“读-标注-写”策略的高效同步逆向文本规范化研究

    Towards Efficient Simultaneous Inverse Text Normalization with Pretrained Text-to-Text Language Model and Read-Tag-Write Policy

    [https://arxiv.org/abs/2610.19492](https://arxiv.org/abs/2610.19492)

    该论文提出了一种基于预训练文本到文本语言模型的高效流式端到端逆向文本规范化系统，通过架构适配、专门训练策略和“读-标注-写”解码策略实现流式推理，其准确性媲美非流式基线并超越混合方法。

    

    流式逆向文本规范化（ITN）对于将流式自动语音识别输出的口语形式转换为格式化的书面文本至关重要。现有的流式ITN方法依赖于将神经标注与专家手工构建的有限状态转换器规则相结合的混合系统，这限制了跨领域和跨语言的可扩展性。虽然端到端模型通过直接从数据学习提供了卓越的可扩展性，但标准的编码器-解码器架构由于全局注意力机制而本质上无法进行流式处理。在本文中，我们提出了一种高效的流式端到端ITN系统，通过适配预训练的文本到文本模型来充分利用其强大的语言知识。为实现流式处理，我们引入了架构适配、专门的训练策略、“读-标注-写”解码策略以及推理优化。在越南语数据集上的实验表明，该系统的准确性与非流式基线相当，且优于混合方法……

    arXiv:2610.19492v1 Announce Type: new  Abstract: Streaming Inverse Text Normalization (ITN) is vital for converting spoken-form outputs from streaming Automatic Speech Recognition into formatted written text. Existing streaming ITN methods rely on hybrid systems combining neural tagging with expert-crafted finite-state transducer rules, limiting scalability across domains and languages. While end-to-end models offer superior scalability by learning directly from data, standard encoder-decoder architectures are inherently non-streaming due to global attention. In this paper, we propose an efficient streaming end-to-end ITN system, adapting a pretrained text-to-text model to leverage its robust linguistic knowledge. To enable streaming, we introduce architectural adaptations, a specialized training strategy, a Read-Tag-Write decoding policy, and inference optimizations. Experiments on Vietnamese datasets show accuracy comparable to non-streaming baselines, outperforming hybrid methods wh
    
[^118]: AgriJev：基于多尺度适配模型的高效农业候选评分

    AgriJev: Efficient Agricultural Candidate Scoring with Multi-Scale Adapted Models

    [https://arxiv.org/abs/2610.19478](https://arxiv.org/abs/2610.19478)

    AgriJev是一系列多尺度适配的Qwen3.5农业模型，通过单次前向调用直接对候选选项评分而非生成文本答案，在17个外部测试视图上平均准确率达75.44%，超越原始模型和农业SFT生成模型。

    

    农业系统需要在异构实体类型、文本判断和知识问题选项上做出决策。生成文本答案会为最终只需候选选择的决策增加额外的解码与提取步骤。我们提出AgriJev，这是一系列适配后的Qwen3.5模型，包含0.8B、2B、4B和9B四种规模。通过一个共享接口，模型以农业文本、判断问题和候选定义作为条件输入，在一次前向调用中同时返回分数和相对概率。我们基于现有农业资源构建了具有来源感知的训练数据和固定的评测集，并保留了原始分组与标注来源信息。对比实验包括原始Qwen3.5模型和匹配的农业SFT模型，两者均通过原生答案生成方式进行评估。在17个外部测试视图上，AgriJev-9B达到75.44%的平均准确率，分别超过原始生成模型和SFT生成器5.89和3.79个百分点。

    arXiv:2610.19478v1 Announce Type: new  Abstract: Agricultural systems make decisions over heterogeneous entity types, textual judgments, and knowledge-question options. Generating a textual answer adds decoding and extraction steps to decisions that ultimately require a candidate choice. We introduce AgriJev, a family of adapted Qwen3.5 models at 0.8B, 2B, 4B, and 9B. A shared interface conditions on agricultural text, a judgment question, and candidate definitions, returning scores and relative probabilities in one forward call. We prepare source-aware training records and frozen evaluations from existing agricultural resources, retaining original groups and annotation provenance. Comparisons include original Qwen3.5 models and matched agricultural SFT models, both evaluated through native answer generation. Across 17 external test views, AgriJev-9B reaches 75.44% mean accuracy, exceeding the original and SFT generators by 5.89 and 3.79 percentage points. It outperforms the original g
    
[^119]: LALM后训练究竟学到了什么？重新思考音频理解与推理能力

    What Does LALM Post-training Learn? Rethinking Audio Understanding and Reasoning Capabilities

    [https://arxiv.org/abs/2610.19476](https://arxiv.org/abs/2610.19476)

    研究通过受控实验发现，大型音频-语言模型后训练带来的性能提升主要源于对特定问题格式和文本线索的适应，而非音频理解与推理能力的真正提升。

    

    后训练通常会提升大型音频-语言模型（LALM）在音频理解与推理基准测试上的表现。然而，这些提升可能反映的是模型对特定问题格式（如多选题）的适应，而非音频理解与推理能力的真正进步。为了探究LALM从后训练中究竟学到了什么，我们进行了受控实验，将音频理解能力的提升与对特定问题格式的适应分离开来。在不同的模型和后训练方法上，我们发现即使没有提供有效的音频信息，许多所报告的性能提升依然存在；但当问题格式发生变化时，这些提升在很大程度上会消失。这些发现表明，后训练往往教会模型利用任务特定的文本线索和答案模式，而不是发展出可泛化的音频理解与推理能力。

    arXiv:2610.19476v1 Announce Type: cross  Abstract: Post-training often improves the performance of large audio-language models on audio understanding and reasoning benchmarks. However, these improvements may reflect adaptation to specific question formats, such as multiple-choice questions, rather than genuine advances in audio understanding and reasoning capabilities. To investigate what LALMs actually learn from post-training, we conduct controlled experiments that disentangle improvements in audio understanding from adaptation to specific question formats. Across different models and post-training methods, we find that much of the reported improvement persists even without informative audio, but largely disappears when the question format changes. These findings suggest that post-training often teaches models to exploit task-specific textual cues and answer patterns rather than developing generalizable audio understanding and reasoning capabilities.
    
[^120]: ActICL：面向智能体多模态上下文学习的后训练方法

    ActICL: Post-Training for Agentic Multimodal In-Context Learning

    [https://arxiv.org/abs/2610.19475](https://arxiv.org/abs/2610.19475)

    ActICL通过后训练使多模态大模型智能体在迭代推理中自主分析信息缺口，并借助工具增强推理进行渐进式的针对性演示检索，从而自主构建多模态上下文学习所需的上下文。

    

    多模态上下文学习（ICL）是多模态大语言模型（MLLM）的一项重要能力，其性能与所使用的演示示例密切相关，因此演示检索成为ICL的核心问题。常见方法使用固定的评分方式一次性检索全部演示示例，这可能引入冗余和噪声。上下文感知检索方法为解决这一问题而出现，允许从不断增长的上下文中进行增量检索。然而，现有方法都无法让MLLM学会自主判断需要哪些新演示示例，并据此自行开展渐进式检索。为此，我们提出了ActICL，它通过工具增强推理训练MLLM智能体来获取和使用ICL上下文。ActICL智能体在迭代推理和工具使用过程中分析自身的信息缺口，基于这些缺口执行有针对性的检索以获取一组初始候选演示示例，随后它可以……（原文摘要在此处截断）

    arXiv:2610.19475v1 Announce Type: cross  Abstract: Multimodal in-context learning (ICL) is an important capability of multimodal large language models (MLLMs). Its performance is closely tied to the demonstrations it uses, making demonstration retrieval a core problem for ICL. Common methods use fixed scoring methods to retrieve all the demonstrations at once, which may introduce redundancy and noise. Context-aware retrieval methods have emerged to address this problem by allowing incremental retrieval from a growing context. However, none of them lets the MLLM learn how to judge what new demonstrations it needs and then carry out progressive retrieval on its own. To this end, we introduce ActICL, which trains an MLLM agent to acquire and use an ICL context via tool-augmented reasoning. An ActICL agent analyzes its own information gaps during iterative reasoning and tool-use actions. Based on these gaps, it performs targeted retrieval to acquire a set of initial candidates. It can then
    
[^121]: 超越视觉：视觉作为原生多模态预训练中的可扩展锚点

    More Than Seeing: Vision as Scalable Anchors in Native Multimodal Pre-Training

    [https://arxiv.org/abs/2610.19443](https://arxiv.org/abs/2610.19443)

    该研究通过受控消融实验发现，在原生多模态预训练中保留视觉内容能在不损害纯文本性能的前提下提升图文样例上的文本预测，多模态收益随计算量呈幂律增长，且视觉信号还可作为锚点提升纯文本空间推理等超越视觉理解的能力。

    

    原生多模态预训练是指从零开始在文本和多模态数据上训练单一模型，使其能够理解语言和视觉输入。然而，视觉信号如何塑造扩展行为并支持超越视觉理解的能力，目前仍不清楚。我们通过在不同模型规模和训练预算下进行受控的视觉内容消融实验来研究这一问题。具体而言，我们用无内容的填充词元嵌入替换图像嵌入。结果表明，保留视觉内容可以提升图文样例上的文本预测能力，而不会实质性改变给定计算预算下最低的纯文本损失、模型规模与训练数据的最优平衡，或整体语言得分。相对的多模态损失收益大致随计算量呈幂律增长。为测试超越图像理解本身的收益，我们仅基于文本描述评估空间推理能力，两种条件均随模型和数据规模的扩大而改善。

    arXiv:2610.19443v1 Announce Type: new  Abstract: Native multimodal pre-training trains a single model from scratch on text and multimodal data, enabling it to understand both language and visual inputs. However, how visual signals shape scaling behavior and support capabilities beyond visual understanding remains unclear. We investigate this question through controlled visual-content ablations across model sizes and training budgets. Specifically, we replace image embeddings with content-free padding token embeddings. Retaining visual content improves text prediction on image-text examples without materially changing the lowest text-only loss at a given compute budget, the optimal balance of model size and training data, or aggregate language scores. The relative multimodal loss benefit grows approximately as a power law of compute. To test benefits beyond understanding images, we evaluate spatial reasoning from text descriptions alone. Both conditions improve with model and data scali
    
[^122]: 大语言模型如何理解法律哲学？诊断大语言模型中的法律哲学推理

    How Do LLMs Understand Legal Philosophy? Diagnosing Legal-Philosophical Reasoning in LLMs

    [https://arxiv.org/abs/2610.19427](https://arxiv.org/abs/2610.19427)

    该论文提出LePhilBench基准，通过“理解法律哲学”与“应用法律哲学”两项任务及人工精心构建的数据集，首次系统评估了大语言模型的法律哲学推理能力。

    

    法律推理是大语言模型（LLM）完成法律判断预测等法律AI任务的一项基础能力。在法律实践中，这种推理受到法律哲学的影响：法官对法律的本质和法律渊源的看法决定了其如何解释和适用法律材料，因此面对相同法条和判例的法官可能得出不同的判决。现有研究在很大程度上忽视了大语言模型能否理解法律哲学并将其运用于法律推理，我们将这种能力称为“法律哲学推理”。为填补这一空白，我们提出了LePhilBench，一个用于评估大语言模型法律哲学推理能力的基准，它包含两个基准测试任务和一个数据集。这两个任务分别评估大语言模型对法律哲学的理解程度，以及将法律哲学运用于法律推理的程度。该数据集经过人工精心构建，其测试集包含220个美国最高……（摘要内容在此处截断）

    arXiv:2610.19427v1 Announce Type: new  Abstract: Legal reasoning is a fundamental capability for large language models (LLMs) to complete legal AI tasks such as legal judgment prediction. In legal practice, this reasoning is shaped by legal philosophy: a judge's view of the nature and sources of law guides how legal materials are interpreted and applied, so judges working from the same statutes and precedents can reach different judgments. Existing research has largely overlooked whether LLMs can understand legal philosophy and apply it in legal reasoning, an ability we call \textit{legal-philosophical reasoning}. To fill this gap, we introduce LePhilBench, a benchmark for evaluating the legal-philosophical reasoning of LLMs that comprises two benchmarking tasks and a dataset. The two tasks evaluate how well LLMs understand legal philosophy and how well they apply it in legal reasoning, respectively. The dataset is manually and meticulously constructed. Its test set contains 220 U.S. S
    
[^123]: ForkGRPO：面向扩散语言模型的高效探索与细粒度信用分配

    ForkGRPO: Efficient Exploration and Fine-Grained Credit Assignment for Diffusion Language Models

    [https://arxiv.org/abs/2610.19408](https://arxiv.org/abs/2610.19408)

    提出ForkGRPO强化学习框架，通过从参考轨迹的高熵中间状态分支采样多个续写实现高效探索，并引入细粒度的片段级信用分配，从而提升扩散语言模型的推理能力。

    

    扩散大语言模型（dLLMs）发展迅速，然而通过强化学习（RL）增强其推理能力仍然是一个悬而未决的挑战。任意顺序生成可能会遭受熵退化问题，这会缩窄可供探索的推理路径。近期工作通过将标准GRPO与从左到右的生成方式相结合来缓解这一问题。然而，从输入提示独立生成每个rollout，会使来自中间状态的其他续写方式得不到充分探索，同时轨迹级别的优势分配对所有token赋予统一信用，无法区分各个推理片段的不同贡献。我们提出ForkGRPO，这是一个将高效探索与细粒度片段级信用分配相结合的强化学习框架。它沿着参考轨迹从高熵中间状态采样多个续写，优先考虑文本边界并复用共享前缀。

    arXiv:2610.19408v1 Announce Type: new  Abstract: Diffusion large language models (dLLMs) have advanced rapidly, yet strengthening their reasoning through reinforcement learning (RL) remains an open challenge. Any-order generation can suffer from entropy degradation, which narrows the reasoning paths available for exploration. Recent work mitigates this issue by applying standard GRPO with left-to-right generation. However, independently generating every rollout from the input prompt can leave alternative continuations from intermediate states underexplored, while trajectory-level advantages assign uniform credit to all tokens without distinguishing the contributions of individual reasoning segments. We propose ForkGRPO, an RL framework that couples efficient exploration with fine-grained segment-level credit assignment. It samples multiple continuations from high-entropy intermediate states along a reference trajectory, prioritizing textual boundaries and reusing shared prefixes. The c
    
[^124]: 视觉-语言模型像有偏见的贝叶斯观察者一样整合冲突证据

    Vision-Language Models Integrate Conflicting Evidence Like Biased Bayesian Observers

    [https://arxiv.org/abs/2610.19407](https://arxiv.org/abs/2610.19407)

    本文提出一个贝叶斯视角的双参数线索组合模型，通过将证据强度与模态整合偏差分离，准确预测视觉-语言模型在冲突多模态输入下的偏好并无需重新拟合即可泛化到新的不可靠性形式，在单一规则下统一解释了文本主导与视觉主导现象。

    

    视觉-语言模型（VLM）必须协调冲突的文本与图像信息。先前的工作用模态偏差来刻画由此产生的行为——例如“对文本的盲目信任”——但这种偏好在不同模型、领域和输入条件下为何有所不同仍不清楚。我们从贝叶斯视角发展了一种预测性解释，假设VLM将来自各模态的条件独立证据与先验信念相结合，并受模态特定偏差的影响。由此得到的双参数线索组合模型能够准确预测VLM在未见过的输入退化组合下的偏好，并且无需重新拟合即可迁移到新形式的不可靠性。将证据强度与整合偏差分离可以解释VLM如何在更重视视觉证据的同时仍遵循文本，从而在单一组合规则下统一了关于文本主导和视觉主导的现象。

    arXiv:2610.19407v1 Announce Type: new  Abstract: Vision-language models must reconcile conflicting text and images. Prior work has characterized the resulting behavior in terms of modality biases--e.g., a "blind faith in text"--but why preferences vary across models, domains, and input conditions remains unclear. We develop a predictive account from a Bayesian perspective, positing that VLMs combine conditionally independent evidence from each modality with prior beliefs, subject to modality-specific biases. The resulting two-parameter cue combination model accurately predicts VLM preferences under unseen combinations of input degradation and transfers to new forms of unreliability without refitting. Separating evidence strength from integration bias explains how VLMs can follow text even while weighting visual evidence more heavily, reconciling reports of text and vision dominance under a single combination rule.
    
[^125]: KoAGE：大语言模型了解、学习并创造韩语新词吗？

    KoAGE: Do Large Language Models Know, Learn, and Coin Korean Neologisms?

    [https://arxiv.org/abs/2610.19399](https://arxiv.org/abs/2610.19399)

    该研究提出NeoTrace自动化框架与KoAGE基准，系统评估大语言模型对韩语新词的认知与创造能力，发现模型虽能从语境中推断新词含义，但对构词结构的理解仍存在明显局限。

    

    韩语新词通过多种机制产生，这对语言模型而言是一项超越单纯词汇记忆的严苛考验。为实现系统化评估，我们提出了NeoTrace——一个可重复运行的自动化框架，用于跨文体收集新词并确定其出现时间。该框架收集的4,271条词条支撑了KoAGE基准，这是一个具有时间截止意识的评测基准，包含三项习得任务和两项生成任务，其中后者据我们所知是首个在原始材料和社区语境下将模型造词与真实存在的新词进行对比的评测。在对21个模型的评估中，词汇记忆与语境推断表现出明显分化：在针对五个模型的时间序列分析中，模型只能对截止年份新词中的13-30%给出定义，但能够从单个用法实例中推断出词义。然而，结构理解仍然有限：即使是正确的定义也几乎无法提升对截止日期前新词构词方式的分析能力。在给定构词材料的情况下，单个模型通常能遵循人类的构词机制，而多智能体造词……

    arXiv:2610.19399v1 Announce Type: new  Abstract: Korean neologisms arise through diverse mechanisms, a demanding test of language models beyond lexical memorization. To enable systematic evaluation, we introduce NeoTrace, an automated, rerunnable framework that collects and dates neologisms across genres. Its 4,271 entries support KoAGE, a cutoff-aware benchmark with three acquisition and two generation tasks---the latter, to our knowledge, the first to compare model coinages with attested neologisms under their original materials and community context. Across 21 models, lexical recall diverges from contextual inference: in a five-model temporal analysis, models define only 13-30% of cutoff-year words, yet can infer meaning from one usage. However, structural understanding remains limited: even correct definitions barely improve formation analysis of pre-cutoff neologisms. Given word-building materials, single models often follow human formation mechanisms, whereas multi-agent coinage 
    
[^126]: ELARSpeech：一个覆盖675种代表性不足语言、时长3万小时的长尾多语言语音技术数据集

    ELARSpeech: A 30K-Hour Dataset Across 675 Underrepresented Languages for Long-Tail Multilingual Speech Technologies

    [https://arxiv.org/abs/2610.19388](https://arxiv.org/abs/2610.19388)

    ELARSpeech是一个包含675种代表性不足语言、总时长30,202小时的大规模多语言语音数据集，为将语音识别与语音合成技术扩展至语言长尾领域提供了重要资源和测试平台。

    

    近年来，大规模多语言语音建模取得了显著进展，然而现有的训练资源和基准测试仍然未能充分覆盖语言长尾中的大部分语言。为了弥补这一差距，我们推出了ELARSpeech，这是一个大规模多语言语音资源，包含来自675种代表性不足语言的30,202小时语音数据，整理自濒危语言档案馆。该语料库结合了人工标注和未标注的录音，涵盖了多样的录音条件、应用领域和语系。除了语料库构建之外，我们的实验展示了ELARSpeech如何支持将自动语音识别（ASR）和语音合成（TTS）能力扩展到多种代表性不足的语言，并且通过对未标注数据进行伪标注可以带来进一步的收益。与此同时，我们的语音表征分析揭示了将现有多语言模型迁移到语言长尾时仍存在的局限性。总而言之，ELARSpeech既提供了一份宝贵的资源，也提供了一个测试平台。

    arXiv:2610.19388v1 Announce Type: new  Abstract: Massively multilingual speech modeling has made remarkable progress in recent years, yet existing training resources and benchmarks still underrepresent much of the linguistic long tail. To address this gap, we introduce ELARSpeech, a large-scale multilingual speech resource comprising 30,202 hours of speech across 675 underrepresented languages, curated from the Endangered Languages Archive. Combining human-annotated and unlabeled recordings, the corpus spans diverse recording conditions, domains, and language families. Beyond corpus construction, our experiments show how ELARSpeech supports extending ASR and TTS capabilities to diverse underrepresented languages, with further benefits from pseudo-labeling unlabeled data. At the same time, our speech representation analysis reveals remaining limitations in transferring existing multilingual models to the linguistic long tail. Together, ELARSpeech provides both a resource and a testbed f
    
[^127]: 所有摘要之总和：用于分级命题显著性评估的PROPALIGN

    The Sum of All Summaries: PROPALIGN for Graded Proposition Salience Evaluation

    [https://arxiv.org/abs/2610.19387](https://arxiv.org/abs/2610.19387)

    本文提出基于多摘要的分级命题显著性评分方法PROPALIGN及配套数据集PROPALIGNBANK，证明摘要式评分比直接询问模型更稳定，且结合语言特征的模型集成可接近人工标注质量上限。

    

    文档中显著命题的识别与排序间接参与了许多人工智能任务，但很少有论文清晰地定义命题显著性分数、提供多样化的数据，或直接在该任务上评估模型。本文提出了PROPALIGN，一种基于多个摘要的命题显著性分数估计方法，并发布了与之配套的数据集PROPALIGNBANK，其中包含超过3.5万个命题的13.8万条对齐数据。我们在英语中跨越广泛的口语和书面文体比较了几种显著命题提取方法，并证明简单地要求模型作出判断会导致结果不一致。相比之下，分级的、基于摘要的指标更加稳定。我们进一步观察到，虽然单个模型的表现不如人类，但结合语言特征的模型集成方法接近人工标注的质量上限。此外，对模型输出的分析表明

    arXiv:2610.19387v1 Announce Type: new  Abstract: Identification and ranking of salient propositions in documents is indirectly involved in many AI tasks, but few papers define proposition salience scores clearly, make diverse data available, or evaluate models directly on the task. This paper proposes PROPALIGN, a multiple-summary-based approach for propositional salience score estimation, and releases PROPALIGNBANK, an accompanying dataset for the method with 138K alignments of over 35K propositions. We compare several approaches to salient proposition extraction in English across a broad range of spoken and written genres, and demonstrate that naively asking models for judgments leads to inconsistencies. By contrast, graded, summary-based metrics are more stable. We further observe that while single models perform worse than humans, an ensemble of models incorporating linguistic features approaches manual annotation quality ceilings. In addition, analyses of the model outputs suggest
    
[^128]: 学习该保留什么：基于迁移感知的从用户反馈中学习个性化技能

    Learning What to Keep: Transfer-Aware Personal Skill Learning from User Feedback

    [https://arxiv.org/abs/2610.19380](https://arxiv.org/abs/2610.19380)

    提出 SkillGrain 框架，将用户反馈分解为可独立执行的原子规则并预测其可迁移性，通过基于评分准则的奖励、细粒度信用分配以及强化学习联合优化，从用户反馈中选择性地保留真正可迁移的个性化技能。

    

    用户反馈使大型语言模型（LLM）能够改进可复用的个性化技能，但反馈中往往混杂着持久性偏好与针对特定请求的修正。不加区分地同时保留二者会带来负迁移的风险，因此需要在个体需求层面进行选择性保留。我们将这一挑战构建为原子规则上的选择性迁移问题，并提出了 SkillGrain——一个从用户反馈中学习可复用个性化技能的框架。SkillGrain 将反馈分解为可独立执行的规则，预测各规则的可迁移性，并仅保留那些适用于未来请求的规则。为了学习这些决策，我们将基于评分准则引导的技能奖励与细粒度信用分配相结合，分别评估规则内容和保留决策。我们进一步通过强化学习联合优化技能归纳与基于技能条件的生成，确保所学技能能够支持下游的个性化任务。我们构建了一个基准（原文在此截断）

    arXiv:2610.19380v1 Announce Type: new  Abstract: User feedback enables LLMs to refine reusable personalized skills, but often mixes persistent preferences with request-specific corrections. Indiscriminately retaining both risks negative transfer, motivating selective retention at the level of individual requirements. We frame this challenge as selective transfer over atomic rules and propose SkillGrain, a framework for learning reusable personalized skills from user feedback. SkillGrain decomposes feedback into independently actionable rules, predicts their transferability, and retains only those suitable for future requests. To learn these decisions, we combine rubric-guided skill rewards with fine-grained credit assignment that separately evaluates rule content and retention decisions. We further jointly optimize skill induction and skill-conditioned generation through reinforcement learning, ensuring that learned skills support downstream personalization. We construct a benchmark sp
    
[^129]: 评估压缩记忆：依赖性、效用与内容收益

    Evaluating Compressed Memory: Dependence, Utility, and Content Benefit

    [https://arxiv.org/abs/2610.19373](https://arxiv.org/abs/2610.19373)

    该论文提出将记忆置零与无法访问存储文本的对照模型相结合来严格评估压缩记忆的真实效用，发现某些依赖记忆的模型性能反而不如无记忆训练的替代模型。

    

    语言模型的回答更好，是因为其压缩记忆中包含了有用的文档信息吗？将记忆向量替换为零可以降低已训练模型的分数，但这并不能证明它优于一个在没有记忆的情况下训练的模型。我们将置零方法与单独训练的对照模型相结合——这些对照模型无法访问存储的文本，或对每个文档都接收相同的常量输入。在九个系统中，一些模型虽然依赖记忆，但其表现却不如训练过的替代方案。例如，在预测生成故事中状态的实验里，将记忆置零会恶化模型的预测损失，然而在没有记忆情况下训练的模型却比拥有完整记忆的原始模型取得了更低的损失。在TriviaQA上，为xRAG提供来自另一个问题段落的记忆，将衡量输出与标准答案词重合度的F1分数从无文档时的19.38提高到24.78，但其对输出包含标准答案频率的影响在统计上并不确定。

    arXiv:2610.19373v1 Announce Type: new  Abstract: Does a language model answer better because its compressed memory contains useful document information? Replacing memory vectors with zeros can lower a trained model's score without showing that it outperforms a model trained without memory. We combine zeroing with separately trained models that cannot access stored text or receive the same constant input for every document. Across nine systems, some models rely on memory yet perform worse than the trained alternative. For example, in experiments that predict states in generated stories, zeroing memory worsens a model's prediction loss, yet a model trained without memory achieves lower loss than the original model with intact memory. On TriviaQA, giving xRAG memory from another question's passage raises F1, which measures word overlap with accepted answers, from 19.38 without a document to 24.78. Its effect on how often outputs contain an accepted answer is statistically uncertain. The q
    
[^130]: 超越纠正动作：基于文本批评训练大语言模型智能体

    Beyond Corrected Actions: Training LLM Agents on Textual Critiques

    [https://arxiv.org/abs/2610.19367](https://arxiv.org/abs/2610.19367)

    提出CAFT框架，让专家模型在学生LLM智能体生成的错误动作上同时提供纠正动作和解释性文本批评，并证明基于这种更丰富的文本反馈进行训练能更有效地提升智能体性能。

    

    大语言模型（LLM）越来越多地被用于多轮智能体任务，这类任务需要在较长的动作序列中与用户和环境进行交互。改进LLM智能体的一种常见方法是使用更强大的专家模型提供训练监督，方式要么是对专家的轨迹进行行为克隆，要么是利用专家来纠正较弱学生模型所生成轨迹中的动作。然而，专家LLM能够提供比单纯动作更丰富的监督信息：它们可以直接生成解释特定动作为何不正确的文本批评。我们研究了基于此类反馈进行训练是否对LLM智能体有益，并提出了批评增强的智能体微调，这是一个在学生模型生成的轨迹上获取专家文本反馈的框架。对于每个不正确的动作，专家会同时给出纠正后的动作以及解释学生错误的文本批评——随后我们利用这些……

    arXiv:2610.19367v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used for multi-turn agentic tasks that require interactions with users and environments over extended sequences of actions. A common approach to improving LLM agents is to use more capable expert models to provide training supervision, either by behavior cloning the expert's trajectories or by using the expert to correct the actions in trajectories generated by the weaker student model. However, expert LLMs can provide richer supervision than actions alone: they can directly generate textual critique that explain why specific actions are incorrect. We study whether training on such feedback benefits LLM agents and propose Critique-Augmented Agentic Fine-Tuning (CAFT), a framework that elicits textual expert feedback on student-generated trajectories. For each incorrect action, the expert yields both a corrected action and a textual critique explaining the student's mistake--which we then use 
    
[^131]: 看见、维护与学习：多模态智能体记忆的演进

    Seeing, Maintaining, and Learning: The Evolution of Multimodal Agent Memory

    [https://arxiv.org/abs/2610.19366](https://arxiv.org/abs/2610.19366)

    该论文提出了一个双轴分析框架，从多模态信息的形成与表示、以及基于积累经验的自适应记忆管理两个维度，系统性地梳理了多模态智能体记忆的研究演进。

    

    多模态模型与智能体的出现，使AI系统能够通过整合跨多种模态的信息，更接近人类地从真实世界中学习并与之交互。然而，随着多模态智能体迈向长时程任务和终身学习场景，如何使其能够持续从复杂、异构的多模态数据中获取有用信息仍然是一个重大挑战。应对这一挑战需要有效的记忆机制，该机制能够在会话之间保持连续性、适应不断变化的用户和环境，并通过积累的经验来改进记忆管理策略。我们引入了一个面向多模态智能体记忆的双轴分析框架，该框架围绕两个关键问题构建：多模态信息如何形成和表示（轴一：形成与表示），以及其管理如何随积累的经验而适应（轴二：记忆管理）。轴一“形成与表示”……

    arXiv:2610.19366v1 Announce Type: cross  Abstract: Multimodal models and agents have emerged to enable AI systems to learn from and interact with the real world more like humans by integrating information across multiple modalities. However, enabling multimodal agents to continually acquire useful information from complex, heterogeneous multimodal data remains a significant challenge as they move toward long-horizon tasks and lifelong settings. Addressing this challenge requires effective memory mechanisms that can maintain continuity across sessions, adapt to changing users and environments, and refine memory-management policies through accumulated experience. We introduce a two-axis analytical framework for multimodal agent memory structured around two key questions: how multimodal information is formed and represented (Axis I, Formation and Representation), and how its management adapts with accumulated experience (Axis II, Memory Management). Axis I, Formation and Representation, c
    
[^132]: 尾部竞争解释了可验证问题中 Best-of-N 采样的缩放规律

    Tail Competition Explains Scaling in Best-of-N Sampling for Verifiable Problems

    [https://arxiv.org/abs/2610.19362](https://arxiv.org/abs/2610.19362)

    该论文将 Best-of-N 采样建模为正确与错误响应极端奖励尾部之间的竞争，利用极值理论刻画出正确尾部主导、错误尾部主导和临界三种渐近状态，统一解释了缩放增益与奖励作弊导致的性能退化，并推导了有限样本缩放定律。

    

    Best-of-N 采样是大语言模型一种简单的推理时扩展策略，然而其在不完整奖励模型下的行为仍鲜为人知。我们将 Best-of-N 研究为正确响应与错误响应的极端奖励尾部之间的竞争。这一视角表明，成功不仅取决于采样到正确响应的概率，还取决于正确响应是否在上尾部中占优于错误响应。利用极值理论，我们刻画了三种渐近状态：正确尾部主导、错误尾部主导和临界状态，从而同时解释了单调增益和奖励作弊导致的性能退化。我们进一步针对代表性奖励分布推导了有限样本缩放定律，表明在高斯奖励下，方差足够大的错误响应即使平均奖励更低，也可能在选择过程中占据主导。与理想选择不同，使用不完整选择器时的收敛（原文摘要在此截断）

    arXiv:2610.19362v1 Announce Type: new  Abstract: Best-of-N sampling is a simple inference-time scaling strategy for large language models, yet its behavior under incomplete reward models remains poorly understood. We study Best-of-N as a competition between the extreme reward tails of correct and incorrect responses. This view shows that success depends not only on the probability of sampling a correct response, but also on whether correct responses dominate incorrect ones in the upper tail. Using extreme value theory, we characterize three asymptotic regimes: correct-tail-dominant, incorrect-tail-dominant, and critical explaining both monotonic gains and reward-hacking-induced degradation. We further derive finite-sample scaling laws for representative reward distributions, showing that, under Gaussian rewards, incorrect responses with sufficiently large variance can dominate selection even when their mean reward is lower. Unlike oracle selection, convergence with an incomplete select
    
[^133]: VLX-Go：通过多阶段学习与在线策略优化实现鲁棒的具身目标跟踪

    VLX-Go: Robust Embodied Tracking through Multi-Stage Learning and On-Policy Refinement

    [https://arxiv.org/abs/2610.19361](https://arxiv.org/abs/2610.19361)

    该论文提出 VLX-Go——一种视觉-语言路径点策略，其核心创新在于先对齐视觉与语言表征、再通过多阶段学习与在线策略优化来训练跟踪运动，从而显著提升机器人在存在干扰者场景中跟随指定人物的鲁棒性。

    

    具身视觉跟踪要求机器人从视觉观测中识别指定的目标，并控制自身运动以在环境中跟随该目标。跟随一个孤立的人是具身视觉跟踪较为简单的演示，但在存在干扰者的情况下跟随指定的人物仍然具有挑战性。现有方法在从单目标跟踪过渡到存在干扰者的跟踪场景时，成功率会出现大幅下降，这促使我们关注人员跟随机器人必须应对的共享空间。我们提出了 VLX-Go，这是一种视觉-语言路径点策略，其核心假设是：在学习针对跟踪的运动控制之前，应先对视觉表征与语言表征进行对齐。该策略结合了互补的冻结视觉编码器、多尺度时间上下文、Transformer 主干网络，以及用于以机器人为中心路径点预测的轻量级输出头。其训练过程包括视觉-语言对齐、多阶段学习……（原文摘要在此处截断）

    arXiv:2610.19361v1 Announce Type: cross  Abstract: Embodied visual tracking requires a robot to identify a designated target from visual observations and control its own motion to follow that target through an environment. Following an isolated person is a comparatively simple demonstration of embodied visual tracking, but following the intended person among distractors remains challenging. Existing methods exhibit a substantial drop in success from single-target to distractor tracking, motivating a focus on the shared spaces that person-following robots must navigate. We present \method{}, a vision--language waypoint policy built around the hypothesis that visual and language representations should be aligned before tracking-specific motion is learned. The policy combines complementary frozen visual encoders, multiscale temporal context, a transformer backbone, and a lightweight head for robot-centric waypoint prediction. Its training progresses through vision--language alignment, mul
    
[^134]: 竞争伦理框架的转向向量可互换：一种用于激活转向的竞争目标测试

    Steering Vectors for Competing Ethical Frameworks Are Interchangeable: A Competing-Target Test for Activation Steering

    [https://arxiv.org/abs/2610.19359](https://arxiv.org/abs/2610.19359)

    论文提出“竞争目标测试”这一新方法，发现用于康德义务论与行为功利主义的激活转向向量共享大部分方向且可互换，说明转向向量可能通过共享组件起作用而非直接控制目标行为。

    

    激活转向通过向模型的隐藏状态添加一个向量来控制某个指定的行为。然而，该行为上的增益并不能证明向量是直接控制它的：向量可能通过与相邻行为共享的组件起作用，而未转向基线和随机基线无法区分这种情况。我们引入了竞争目标测试：将一个向量与以相同方式为竞争目标构建的向量进行比较，并通过等效性检验判定二者的差异何时低于有效边界。我们在开源权重模型中，将该测试应用于康德义务论（根据行为所遵循的规则来判断行为）和行为功利主义（根据行为产生的福祉来判断行为）的转向向量。在留出的情境上，通过提示施加某个框架会提高该框架的得分并降低另一框架的得分，而两个框架各自的转向向量却以相似的方式同时移动两者的得分，且没有任何一个达到等效边界。这两个向量共享其大部分方向，即“应用某个框架”的请求……

    arXiv:2610.19359v1 Announce Type: cross  Abstract: Activation steering adds a vector to a model's hidden states to control a named behaviour. A gain on that behaviour, however, does not show that the vector controls it directly: it may act through a component shared with neighbouring behaviours, which unsteered and random baselines cannot separate. We introduce the competing-target test: a vector is compared with the vector built the same way for a competing target, and an equivalence test states when their difference is below a useful margin. We apply it in open-weight models to vectors for Kantian deontology, which judges an act by the rule it follows, and act utilitarianism, which judges it by the welfare it produces. On held-out situations, prompting a framework raises its score and lowers the other's, whereas vectors for either framework move both scores alike, and none reaches the margin. The two vectors share most of their direction, a request to apply some framework; the same c
    
[^135]: 从哪里开始，如何移动：分析长程LLM智能体在GRPO下的课程

    Where to Start, How to Move: Analyzing Curricula under GRPO for Long-Horizon LLM Agents

    [https://arxiv.org/abs/2610.19349](https://arxiv.org/abs/2610.19349)

    该研究系统分析了引导辅助GRPO课程设计，发现在固定回合级阈值规则下“从接近完成的阶段开始”最有效，且基于结果信息的工具调用选择在已训练和未训练任务类别上均优于标准GRPO。

    

    当所有采样（rollouts）获得相同的奖励时，组相对策略优化（GRPO）无法提供基于奖励差异的学习信号。在需要多次交互的长程任务中，奖励只在成功完成任务后才给出，这使得训练早期“全部失败”的组非常常见，限制了模型练习后续步骤的机会。引导辅助GRPO（Guide Assisted GRPO）在学习者继续之前重放一段成功的参考前缀。我们研究了初始位置、交接单元和移动规则如何影响无辅助情况下的任务完成率。在BFCL上跨三种模型规模的实验表明，在固定的回合级阈值规则下，从接近完成的阶段开始更为有效。更细粒度的交接本身并不能持续带来帮助，但基于结果信息的工具调用选择在已训练和未训练的任务类别上都优于GRPO。引导式开始也提升了AppWorld在两个评估集上的成功率。这些发现强调了学习从哪里开始以及起始位置如何移动的重要性。

    arXiv:2610.19349v1 Announce Type: cross  Abstract: Group Relative Policy Optimization (GRPO) provides no reward-relative learning signal when all rollouts receive identical rewards. In long-horizon tasks that require multiple interactions, rewards are given only upon successful task completion, making all-fail groups common early in training and limiting opportunities to practice later steps. Guide Assisted GRPO replays a successful reference prefix before the learner continues. We study how initial positions, handoff units, and movement rules affect unassisted task completion. Across three model sizes on BFCL, starting near completion is more effective under a fixed turn-level threshold rule. Finer handoffs alone do not consistently help, but outcome-informed tool-call selection outperforms GRPO on trained and untrained task categories. Guided starts also improve AppWorld success on both evaluation splits. These findings highlight where learning starts and how starting positions move 
    
[^136]: 目标词元不对称性可能伪装成深度专门化

    Target-Token Asymmetry Can Masquerade As Depth Specialization

    [https://arxiv.org/abs/2610.19344](https://arxiv.org/abs/2610.19344)

    层级消融研究中所观察到的“早期层专门化”现象，可能仅仅是两个语料库之间目标词元分布不对称（如数学文本中大量空白字符）所造成的假象，而非真正的深度专门化。

    

    层级消融研究通常会扰动语言模型的早期或晚期层，比较其对两个语料库预测造成的损害，并将某一语料库上更大的早期层效应解读为该模型对该任务的早期层专门化。这种比较会对各语料库要求模型预测的词元取平均，而这些目标词元是不同的：例如，数学解答比百科全书散文需要多得多的空白字符和数字。我们证明这种差异可以制造出深度专门化的假象。我们按目标词元类型拆分了跨语料库的早晚期损伤交互作用，并在两个语料库共享的目标词元混合下重新计算。在GSM8K和Wikitext之间，正交互作用在五个模型实例中均得到复现。在三个完全分解的模型中，只有空白字符具有正交互作用。移除空白字符在所有三个模型中反转了汇总效应，而匹配完整的目标词元混合则在两个模型中反转了该效应。

    arXiv:2610.19344v1 Announce Type: new  Abstract: Layer-ablation studies often perturb early or late layers of a language model, compare the damage to prediction on two corpora, and read a larger early-layer effect on one corpus as early-layer specialization for its task. Such a comparison averages over the tokens that each corpus asks the model to predict, and these target tokens differ: math solutions, for example, ask for far more whitespace and numbers than encyclopedia prose. We show that this difference can create the appearance of depth specialization. We split a cross-corpus early--late damage interaction by target-token type and recompute it under a target mixture shared by both corpora. Between GSM8K and Wikitext, a positive interaction replicates across five model instances. In the three fully decomposed models, only whitespace has a positive interaction. Removing whitespace reverses the pooled effect in all three, and matching the full target mixture reverses it in two. The 
    
[^137]: 反事实门控支持分解：在训练数据过滤中分离选择与重分配效应

    Counterfactual Gate-Support Decomposition: Separating Selection from Redistribution in Training-Data Filtering

    [https://arxiv.org/abs/2610.19343](https://arxiv.org/abs/2610.19343)

    该论文提出反事实门控支持分解（CGSD），一种四臂预算匹配的实验设计，可将训练数据质量门控的整体效应分解为支持内选择、支持间重分配及其交互作用，并在韩英与韩日检索任务中发现支持内选择是唯一显著超出校准零区间的分量。

    

    训练数据质量门控会同时改变两个方面的结果：输入空间各区域内哪些样本得以保留，以及训练数据总量在这些区域之间如何分布，因此单纯的“门控与随机”对比无法将这两种机制分离开来。本文提出了反事实门控支持分解方法，这是一种四臂、预算匹配的实验设计，通过将门控资格与支持分布进行交叉组合，把观察到的门控效应分解为支持内选择、支持间重分配以及二者的交互作用。该方法面临的实际约束是有限池可实现性：作者刻画了在何种质量和词元组装条件下才能构建所需的实验臂，并讨论了支持消亡的情形——在此情况下常规设计无法定义。在韩英和韩日检索任务上的实验表明，支持内选择分量的均值是唯一超出其校准零区间的分量（+29.7 和 +5.6 个保留集条目；30/30 次重复均为正向）。

    arXiv:2610.19343v1 Announce Type: cross  Abstract: A training-data quality gate changes both which examples survive within regions of the input space and how training mass is distributed across those regions, so a gate-versus-random comparison does not isolate these mechanisms. We introduce Counterfactual Gate-Support Decomposition (CGSD), a four-arm, budget-matched design that crosses gate eligibility with support distribution and decomposes the observed gate effect into within-support selection, between-support redistribution, and their interaction. The practical constraint is finite-pool realizability: we characterize mass and token-assembly conditions under which the required arms can be constructed, together with a support-extinction case in which the ordinary design is undefined. On Korean-English and Korean-Japanese retrieval, the mean within-support selection component is the only one that exceeds its calibrated null band (+29.7 and +5.6 holdout items; 30/30 positive replicates
    
[^138]: LLM-jp-4 技术报告

    LLM-jp-4 Technical Report

    [https://arxiv.org/abs/2610.19331](https://arxiv.org/abs/2610.19331)

    LLM-jp-4是强调透明性与可复现性的日本主权大语言模型系列，包含8B密集模型和32B-A3B混合专家模型，均在约12万亿token上从头训练，性能较之前模型显著提升。

    

    本文介绍了LLM-jp-4，这是一系列日本主权大语言模型（LLM），其开发强调透明性、可复现性和实用可用性。LLM-jp-4包含一个8B密集模型和一个32B-A3B混合专家模型，两者均在约12万亿个token上从头训练。为支持独立验证和复现，我们为训练数据、外部模型和软件的选择制定了明确标准，并公开了开发流程。评估结果表明，LLM-jp-4相较于之前的LLM-jp模型有显著提升，且与同等规模的开放权重模型相比具有竞争力。通过记录LLM-jp-4的开发原则、训练流程和实证发现，我们旨在为未来的主权AI开发提供参考。

    arXiv:2610.19331v1 Announce Type: new  Abstract: This paper presents LLM-jp-4, a family of Japanese sovereign large language models (LLMs) developed with an emphasis on transparency, reproducibility, and practical usability. LLM-jp-4 comprises an 8B dense model and a 32B-A3B mixture-of-experts model, both trained from scratch on approximately 12 trillion tokens. To support independent verification and reproduction, we establish explicit criteria for selecting training data, external models, and software, and make the development process publicly available. Evaluation results demonstrate substantial improvements over previous LLM-jp models and competitive performance against open-weight models of similar scale. By documenting the development principles, training procedures, and empirical findings of LLM-jp-4, we aim to provide a reference for future sovereign AI development.
    
[^139]: TopicMerge：一种分而治之的主题建模方法

    TopicMerge: A Divide and Conquer Approach to Topic Modeling

    [https://arxiv.org/abs/2610.19323](https://arxiv.org/abs/2610.19323)

    TopicMerge提出一种分治式的多阶段流水线，先通过高效聚类将文档过度分割为语义纯净的微聚类，再由LLM结合局部邻域上下文迭代仲裁合并候选对，从而同时实现了最先进的聚类性能与计算效率。

    

    主题建模领域的最新进展分化为两条主要路径：高效的基于嵌入的方法具有良好的可扩展性，但在标准聚类指标上往往表现不佳；而大语言模型（LLM）方法虽然实现了最先进的聚类性能，但代价是高昂的计算开销和较差的可扩展性。本文提出了TopicMerge，这是一种新颖的多阶段流水线，它将上述两种方法结合起来，以同时实现最先进的聚类性能和计算效率。TopicMerge首先采用高效的聚类流水线，有意识地将文档语料库过度分割为大量细粒度、语义纯净的微聚类，并为这些微聚类赋予初始语义标签。随后，它引入了一种新颖的迭代合并算法，由LLM在局部邻域上下文的引导下，对候选合并对进行仲裁决策。这些被批准的合并随后被有选择地……

    arXiv:2610.19323v1 Announce Type: new  Abstract: Recent advancements in topic modeling have split into two main paths: efficient embedding-based methods that offer scalability but are often outperformed on standard clustering metrics, and Large Language Model (LLM) approaches that achieve state-of-the-art clustering performance but at the cost of prohibitive computational expense and poor scalability. This paper introduces TopicMerge, a novel multi-stage pipeline that bridges these approaches to achieve both state-of-the-art clustering performance and computational efficiency. TopicMerge first employs an efficient clustering pipeline to deliberately over-segment a document corpus into a large set of fine-grained, semantically pure micro-clusters, which are then given initial semantic labels. It then introduces a novel, iterative merging algorithm where an LLM, guided by local neighborhood context, arbitrates on consolidating candidate pairs. These approved merges are then selectively e
    
[^140]: 基于图的智能体工作流中真实且期望支付最优的模型路由

    Truthful and Expected-Payment-Optimal Model Routing in Graph-Based Agentic Workflows

    [https://arxiv.org/abs/2610.19316](https://arxiv.org/abs/2610.19316)

    本文将基于图的智能体工作流中的LLM模型路由问题形式化为单参数采购拍卖，提出STAR机制，在保证提供商如实报价且覆盖其执行成本的同时，实现期望支付最优。

    

    将日常任务分配给更便宜的大语言模型（LLM）提供商可以降低生成成本，但切换提供商可能需要重新处理累积的上下文；而让相互依赖的任务保留在同一提供商处，则可以复用符合条件的上下文。因此，基于图的工作流使得各任务之间的路由决策相互耦合。此外，独立的提供商可能私下掌握其实时单位成本，并进行策略性谎报。我们将该路由问题形式化为一个单参数采购拍卖问题，并提出STAR（策略无懈可击的拓扑感知智能体路由）方法，该方法最小化以虚拟成本加权的图工作量。其分配规则和支付机制通过对可能的成本报告上的工作量进行积分来计算，使得无论其他提供商如何报告，如实报告都能最大化自身效用，同时能够覆盖每个如实报告的提供商的执行成本。在成本分布独立且虚拟成本非递减的条件下，STAR还能进一步实现期望支付最优。

    arXiv:2610.19316v1 Announce Type: cross  Abstract: Assigning routine tasks to cheaper large language model (LLM) providers can reduce generation cost, but switching providers may require processing accumulated context again. Keeping dependent tasks with the same provider can instead reuse eligible context. Graph-based workflows therefore couple routing decisions across tasks. Independent providers may also privately know their real-time unit costs and strategically misreport them. We formulate this routing problem as a single-parameter procurement auction and propose STAR (Strategy-proof Topology-aware Agent Routing), which minimizes virtual-cost-weighted graph workloads. Its allocation rule and payments, computed by integrating workload over possible cost reports, make truthful reporting utility-maximizing regardless of other providers' reports and cover each truthful provider's execution cost. With independent cost distributions and non-decreasing virtual costs, STAR additionally min
    
[^141]: PersEvalBench：LLM评估器能否判断角色信息的上下文恰当使用？

    PersEvalBench: Can LLM Evaluators Judge Contextually Appropriate Use of Persona Information?

    [https://arxiv.org/abs/2610.19314](https://arxiv.org/abs/2610.19314)

    提出PersEvalBench基准，通过403个多轮场景和九类受控个性化错误测试LLM评估器能否判断用户历史信息的使用是否恰当，发现即使拥有完整对话历史，LLM评估器识别这些错误的能力仍然有限，其中在个性化不足错误上与人类94.4%的识别率差距最大。

    

    大型语言模型（LLM）评估器正被越来越多地用于评估个性化助手，但它们能否可靠地判断个性化程度却很少得到检验。我们提出了PersEvalBench，这是一个用于测试评估器能否识别用户历史信息被恰当或不恰当使用的基准。该基准包含403个多轮对话场景，其中用户信息应当持续适用、迁移适用或停止适用。基准涵盖三大类别共九种受控错误子类型：个性化不足、个性化过度和状态误用错误。人工审核证实生成的错误与预期类型相符，且所规定的用户信息使用方式是合理的。在对五个LLM评估器的测试中，即使提供完整的对话历史，其对受控个性化错误的识别能力仍然有限。在人工评估的子集上，个性化不足错误呈现出最大的差距：人类能识别出94.4%的此类错误，而评估器的识别率则明显更低（摘要在此处截断）。

    arXiv:2610.19314v1 Announce Type: new  Abstract: Large language model (LLM) evaluators are increasingly used to assess personalized assistants, yet whether they can reliably judge personalization has rarely been tested. We introduce PersEvalBench, a benchmark testing whether an evaluator can recognize appropriate and inappropriate uses of user history. It contains 403 multi-episode scenarios where user information should persist, transfer, or cease to apply. The benchmark covers nine controlled error subtypes across three families: under-personalization, over-personalization, and misapplied-state errors. Human audits confirm that generated errors match their intended types and that the prescribed uses of user information are reasonable.   Across five LLM evaluators, recognition of controlled personalization errors remains limited even with the full dialogue history. On the human-evaluated subset, under-personalization shows the largest gap: humans recognize 94.4% of these errors, compa
    
[^142]: Qwenfant：在简洁平行文本上训练大型字符级BabyLM

    Qwenfant: Train Big Character-level BabyLMs on Concise Parallel Text

    [https://arxiv.org/abs/2610.19294](https://arxiv.org/abs/2610.19294)

    该论文通过使用LLM总结翻译的合成数据、更大的模型和字符级词表（结合LoRA微调）训练出Qwenfant模型，在2026年BabyLM挑战赛多语言赛道中获得第一名。

    

    本工作是我们参加2026年BabyLM挑战赛多语言赛道的参赛作品。我们使用由大语言模型总结和翻译的数据对基线模型进行了改进，表明更高质量的数据可以来自合成来源。我们进一步通过两个简单的改动提升了性能：使用更大的模型，以及使用字符级词表。这两项改动都需要采用LoRA风格的微调才能充分发挥其潜力。我们将最终基于Qwen3的模型命名为Qwenfant，该模型在提交时于多语言赛道排名第一。

    arXiv:2610.19294v1 Announce Type: new  Abstract: This work concerns our submission for the Multilingual track of the 2026 BabyLM Challenge. We improve on our baseline using data that has been summarized and translated by an LLM, showing that higher-quality data can come from synthetic sources. We further improve with 2 simple changes: using a larger model, and using a character-level vocabulary. Both of these changes require LoRA-style fine-tuning for realizing their full potential. We dub our final Qwen3-based model Qwenfant, which, at the time of submission, ranks 1st in the Multilingual track.
    
[^143]: 大语言模型智能体能高效识别家电故障吗？来自诊断模拟家电的启示

    Can LLM Agents Efficiently Identify Appliance Faults? Insights From Diagnosing a Simulated Appliance

    [https://arxiv.org/abs/2610.19291](https://arxiv.org/abs/2610.19291)

    该论文提出了CoffeeBench——一个基于模拟家电的LLM智能体故障诊断基准，发现LLM难以仅凭传感器信息稳定识别故障，但表现最佳的模型能通过迭代缩小故障候选空间实现成功诊断，并进一步分析了不同模型的诊断效率差异。

    

    在所谓的“工业4.0”时代，集成传感器、分析工具和其他数字技术补充了工业工作流程，为增强可视化和自动化提供了潜力。故障预测与健康管理（PHM）和自然语言处理（NLP）领域的研究者都探索了大语言模型（LLM）在工业资产管理中的应用，但许多先前工作对考试式基准测试的关注限制了其对LLM自动化实际操作能力的评估。在本文中，我们介绍了CoffeeBench，这是一个面向LLM智能体的故障诊断任务，基于一个模拟家电进行。我们评估的LLM难以仅凭传感器信息稳定地识别故障，但表现最佳的模型能够迭代地缩小故障候选空间，最终获得成功的诊断。因此，我们后续的分析聚焦于：通过诊断效率区分不同模型（RQ1）；考察表现滞后模型的……

    arXiv:2610.19291v1 Announce Type: new  Abstract: In so-called "Industry 4.0," integrated sensors, analytics tools, and other digital technologies supplement industrial workflows, potentiating enhanced visibility and automation. Prognostics and health management (PHM) and natural language processing (NLP) practitioners alike have explored the application of large language models (LLMs) to industrial asset management, but many prior works' focus on exam-like benchmarks has limited their assessment of LLMs' ability to automate practical operations. Herein, we introduce CoffeeBench, a fault diagnostics task for LLM agents, using a simulated appliance. The LLMs we evaluate struggle to consistently identify faults from the sensor information alone, but the best-performing ones can iteratively refine the fault-option space to eventually obtain a successful diagnosis. Our subsequent analyses, therefore, focus on: differentiating models by diagnostic efficiency (RQ1); examining the lagging perf
    
[^144]: 从效率到充分性：自动作文评分的成本感知测量设计框架

    From Efficiency to Sufficiency: The Cost-Aware Measurement Design Framework for Automated Essay Scoring

    [https://arxiv.org/abs/2610.19287](https://arxiv.org/abs/2610.19287)

    本文提出成本感知测量设计（CAMD）框架，将自动作文评分从单纯的技术效率问题重新构建为成本受限的测量系统，通过计量、分摊、单位经济学和决策四层结构，将计算与工作流资源记录转化为可直接支持部署决策的成本构念。

    

    摘要：在自动作文评分（AES）中，计算成本通常被视为技术效率问题，而非测量设计问题。然而，部署决策所需的远不止报告延迟、模型大小或推理开销。在高风险场景中，总负担包括验证、监控、裁决、合规与治理；在教学场景中，定价、许可与集成可能主导成本。本文将AES重新构建为一个成本受限的测量系统，并引入成本感知测量设计（CAMD）框架——一个由计量、分摊、单位经济学和决策组成的四层结构，用于将计算与工作流资源记录转化为与决策相关的成本构念。直接成本被规定为货币支出加上可归因的劳动力和时间，而设计归因和可审计性这两项标准决定了成本边界所排除的内容。（注：原文摘要在此处被截断）

    arXiv:2610.19287v1 Announce Type: cross  Abstract: Compute cost in automated essay scoring (AES) is usually treated as a technical efficiency issue rather than a measurement design concern. Yet deployment decisions require more than reporting latency, model size, or inference expense. In high-stakes settings, total burden includes validation, monitoring, adjudication, compliance, and governance; in instructional settings, pricing, licensing, and integration may dominate cost. This paper reframes AES as a cost-constrained measurement system and introduces the Cost-Aware Measurement Design (CAMD) Framework, a four-layer structure consisting of metering, allocation, unit economics, and decision, for translating computational and workflow resource records into decision-relevant cost constructs. Direct cost is specified as monetary outlay plus attributable labor and time, and two criteria, design attribution and auditability, determine what the cost boundary excludes. Burdens that satisfy n
    
[^145]: 谁的数据，谁的基准？一个用于审计评估资源的以用户为中心的框架

    Whose Data, Whose Benchmark? A User-Centric Framework for Auditing Evaluation Resources

    [https://arxiv.org/abs/2610.19268](https://arxiv.org/abs/2610.19268)

    该论文提出了一个以用户为中心的基准测试审计框架，通过七个维度对评估资源进行审计，揭示了现有基准中美国中心惯例占主导地位、地理可见性与GDP强相关、以及来源用户情境被纳入而目标用户情境完全缺失的系统性偏差。

    

    基准测试塑造了模型能力被衡量和解读的方式，然而其分数可能掩盖了“是谁的惯例、情境和判断定义了成功的表现”。我们提出了一个以用户为中心的审计框架，该框架区分了贡献基准内容的用户与评估结果旨在代表的用户。该框架在基准内容、元数据、文档和评分程序四个方面具体化了七个维度。通过对16个资源的分析，我们发现：在11个文本基准中，以美国为中心的惯例占据了62.4%至94.4%的匹配信号；而在五个图文资源中，地理可见性高度集中，并与国家GDP份额呈强相关（平均Pearson r = 0.780；Spearman ρ = 0.778）。文档分析揭示了一个系统性的“来源-目标”差距：所有16个资源都具体化了来源用户情境，但没有任何一个具体化了目标用户情境。配对评分实验进一步表明……

    arXiv:2610.19268v1 Announce Type: new  Abstract: Benchmarks shape how model capabilities are measured and interpreted, yet their scores can obscure whose conventions, contexts, and judgments define successful performance. We introduce a user-centric audit framework that distinguishes users who contribute benchmark content from those whose needs evaluation results are intended to represent. The framework operationalizes seven dimensions across benchmark content, metadata, documentation, and scoring procedures. Across 16 resources, we find that American-centered conventions account for 62.4-94.4% of matched signals in 11 text benchmarks, while geographic visibility in five image-text resources is concentrated and strongly associated with national GDP share (mean Pearson r = 0.780; Spearman p = 0.778). Documentation reveals a systematic source-target gap: all 16 resources operationalize source-user context, but none operationalize target-user context. Paired scoring experiments further sh
    
[^146]: MemeTrace：一个面向安全感知迷因溯源分析的多模态大语言模型智能体

    MemeTrace: A Multimodal LLM-Based Agent for Safety-Aware Meme Provenance Analysis

    [https://arxiv.org/abs/2610.19267](https://arxiv.org/abs/2610.19267)

    提出 MemeTrace，一个基于多模态大语言模型的智能体，能够自动追踪网络迷因的来源、传播路径及潜在社会风险，在包含100个真实迷因的 Meme100 数据集上显著优于 VLM 基线。

    

    网络迷因常常是虚假信息和宣传等不安全内容的载体。它们会随新闻事件快速演变，并高度依赖上下文信息，这使得平台和内容审核人员难以对其进行有效的检测和审核。此外，现有的审核研究通常仅基于单张图像将迷因分类为安全或不安全，缺乏进行更广泛溯源分析的能力，即无法回答迷因起源于何处、如何传播、代表何种文化象征意义、是否对社会构成潜在风险等问题。为弥补这一空白，我们提出了 MemeTrace，一个基于多模态大语言模型的智能体，能够自动追踪新兴迷因的来源及其潜在风险。我们在 Meme100（一个包含100个真实世界迷因的数据集）上，通过人工标注和自动指标对 MemeTrace 进行了评估。结果表明，MemeTrace 大幅优于 VLM 基线。更重要的是……

    arXiv:2610.19267v1 Announce Type: cross  Abstract: Online memes are often carriers of unsafe content such as disinformation and propaganda. Their rapid evolution in response to news events and strong dependence on contextual information make it difficult for platforms and moderators to detect and moderate them effectively. Moreover, existing moderation studies typically classify memes as safe or unsafe based on individual images, lacking the ability to perform broader provenance analysis, i.e., where a meme originates, how it spreads, what cultural symbolism it represents, whether it poses potential risks to society, etc. To bridge this gap, we introduce MemeTrace, a multimodal LLM-based agent that automatically traces the provenance and potential risks of emerging memes. We evaluate MemeTrace on Meme100, a dataset of 100 real-world memes, using both human annotation and automatic metrics. The results demonstrate that MemeTrace substantially outperforms the VLM baseline. More important
    
[^147]: 复制-粘贴语言模型

    Copy-Paste Language Models

    [https://arxiv.org/abs/2610.19247](https://arxiv.org/abs/2610.19247)

    提出复制-粘贴语言模型（CPLMs），通过在传统语言建模头之外增加一个轻量级复制-粘贴头，以几乎可忽略的开销显著提升大语言模型的复制能力、训练效率、下游性能以及长上下文和检索表现。

    

    从直观上看，Transformer被认为擅长从其输入中进行复制。我们实际上证明，大语言模型（LLM）在上下文中进行复制的能力可以通过显式建模得到提升。我们提出了复制-粘贴语言模型（CPLMs），它在传统的语言建模（LM）头之外增加了一个轻量级的复制-粘贴头。CPLMs提升了训练效率和下游性能，并在长上下文和检索场景中提供了新的工作点。我们通过预训练从1.35亿到17亿参数规模的模型来验证我们的方法，结果显示其在所有规模和各类骨干架构上都有改进，且训练和推理开销几乎可以忽略不计。我们通过证明CPLMs通过一条替代梯度路径部分缓解了LM头的梯度瓶颈，解释了其成功的原因。

    arXiv:2610.19247v1 Announce Type: new  Abstract: Transformers are intuitively considered to be efficient at copying from their inputs. We actually show that the ability of LLMs to copy in context can be improved through explicit modeling. We propose Copy-Paste Language Models (CPLMs), which add a lightweight Copy-Paste head to the conventional language modeling (LM) head. CPLMs improve training efficiency, downstream performance, and offer new operating points in long-context and retrieval setups. We validate our approach by pretraining models from 135M to 1.7B parameters, showing improvements across all scales and backbones with negligible training and inference overheads. We explain the success of CPLMs by showing that they partially mitigate the LM head gradient bottleneck through an alternative gradient path.
    
[^148]: 句法注意力头是否影响模型行为？

    Do Syntax Heads Influence Model Behavior?

    [https://arxiv.org/abs/2610.19231](https://arxiv.org/abs/2610.19231)

    研究发现，尽管Transformer中的句法注意力头看似专门处理句法关系，但其对句法预测的因果作用证据有限：强制干预这些注意力头并不能改变模型对主语的预测偏好，且其干预效果仅在两种指标之一上优于随机注意力头。

    

    Transformer语言模型会发展出专门处理句法关系的注意力头，例如从动词注意到其主语。这些注意力头在模型的句法预测中是否起着因果作用？支持这种因果作用的证据好坏参半甚至偏向负面：即使我们强制这些注意力头去注意除主语之外的干扰名词，模型的句法预测仍然更符合主语而非干扰名词。对专门句法头的干预对句法预测的影响显著强于对随机注意力头的干预，但这仅在我们提出的两种测量指标中的其中一种上成立。

    arXiv:2610.19231v1 Announce Type: new  Abstract: Transformer language models develop attention heads that specialize in syntactic relations, for example, attending from verbs to their subjects. Do these heads play a causal role in the model's syntactic predictions? Evidence for such a role is mixed to negative: even when we force these heads to attend to a distractor noun other than the subject, the model's syntactic predictions are still more consistent with the subject than with the distractor. Interventions on specialized syntax heads had a significantly stronger effect on syntactic predictions than interventions on random heads, but only in one of the two measures we propose.
    
[^149]: Titan-Llama：为预训练大语言模型加装外部神经记忆模块

    Titan-Llama: Retrofitting Pre-trained LLMs with External Neural Memory Modules

    [https://arxiv.org/abs/2610.19210](https://arxiv.org/abs/2610.19210)

    Titan-Llama 提出了一种后训练方法，为预训练的注意力型大语言模型加装可在测试时更新参数的分层神经记忆模块，在保留近期 token 全保真注意力的同时将历史上下文压缩为固定大小的记忆，从而兼顾长上下文推理的效率与信息保留精度。

    

    Transformer 之所以能够高效地大规模预训练，是因为注意力机制允许跨序列位置的并行计算，但长上下文推理需要越来越大的 KV 缓存。高效的循环类替代方案降低了推理的渐近复杂度，但其序列依赖性限制了训练时逐 token 的并行性，使其更难扩展。此外，当前扩展长上下文的范式造成了“精确保留信息”与“保持长上下文推理高效”这两种目标之间的冲突。为解决这一空白，我们提出了 Titan-Llama，这是一种后训练方法，为基于注意力的语言模型加装一种可在测试时更新参数的分层记忆架构。Titan-Llama 采用分块注意力掩码，在保持对近期 token 全保真注意力的同时，将历史上下文压缩进循环的、固定大小的神经记忆模块（NMM）中。（原文摘要在此处截断）

    arXiv:2610.19210v1 Announce Type: new  Abstract: Transformers are efficient to pre-train at scale because attention permits parallel computation across sequence positions, but long-context inference requires increasingly large KV caches. Efficient recurrent alternatives reduce the asymptotic complexity of inference, but their sequential dependence limits token-wise parallelism during training, making them harder to scale. Further, the current paradigm of extending long context tasks creates competing notions of retaining information precisely and keeping long-context inference efficient. To address this gap, we introduce Titan-Llama, a post-training method for retrofitting attention-based language models with a hierarchical memory architecture that updates parameters at test-time. Titan-Llama applies a blockwise attention mask and preserves full-fidelity attention over recent tokens while compressing historical context into recurrent, fixed-size neural memory modules (NMMs). Our formul
    
[^150]: DUALVIEW：利用广泛证据与输出特定证据进行基于大语言模型的多输出文本回归

    DUALVIEW: Leveraging Broad and Output-Specific Evidence for LLM-based Multi-Output Text Regression

    [https://arxiv.org/abs/2610.19206](https://arxiv.org/abs/2610.19206)

    DualView是一种参数高效的双分支方法，通过从冻结的大语言模型中提取广泛表征和输出特定表征来实现多输出文本回归，在五个任务上以远少于基线的参数量超越了监督回归方法，甚至可超越更大模型的无监督预测。

    

    尽管基于大语言模型（LLM）的回归研究主要集中于标量预测，但多输出文本回归仍然鲜有探索。我们提出了DualView，一种利用大语言模型进行多输出文本回归的参数高效方法。DualView首先从冻结的大语言模型中提取广泛表征和输出特定表征，然后采用双分支架构生成完整的分数向量。在五个异构回归任务和八个大语言模型（最大至122B参数）上的实验表明，DualView优于七个监督回归基线，同时其所需拟合参数比三个最强基线少82至1842倍。这些改进延伸至各个单独的输出维度，并且在训练数据有限的情况下依然保持。值得注意的是，使用较小开源权重大语言模型的DualView甚至可以超越使用Gemini 3.1 Pro的无监督预测。受控消融实验表明，广泛证据与输出特定证据能够提供互补的预测收益，同时轻量级的……（摘要在此处截断）

    arXiv:2610.19206v1 Announce Type: new  Abstract: While work on LLM-based regression has largely focused on scalar prediction, multi-output text regression remains less explored. We present DualView, a parameter-efficient approach to leverage LLMs for multi-output text regression. \method{} first extracts broad and output-specific representations from a frozen LLM, then uses a dual-branch architecture to produce the full score vector. Across five heterogeneous regression tasks and eight LLMs (up to 122B), DualView outperforms seven supervised regression baselines while requiring 82--1842 times fewer fitted parameters than the three strongest baselines. These improvements extend across individual output dimensions and persist even with limited training data. Notably, DualView using a smaller open-weight LLM can outperform unsupervised prediction with Gemini 3.1 Pro. Controlled ablations show that broad and output-specific evidence provide complementary predictive benefits, while lightwei
    
[^151]: 面向语言模型稀有事件估计的加权集成采样

    Weighted Ensemble Sampling for Rare-Event Estimation in Language Models

    [https://arxiv.org/abs/2610.19202](https://arxiv.org/abs/2610.19202)

    提出了一种无需训练的加权集成（WE）框架，仅从原始语言模型采样，并利用模型内部表示作为前缀级信号、周期性地将采样预算重新分配给更有希望的前缀，从而高效估计语言模型稀有事件的概率。

    

    大型语言模型中罕见但严重的失败，在大规模部署时可能具有实际意义上的重要性。使用直接蒙特卡洛采样来估计这类事件的概率是极其低效的，而基于提议分布的方法通常需要对一个额外的提议分布进行辅助学习。我们提出了**加权集成**，这是一个无需训练的语言模型稀有事件估计框架。WE 仅从原始语言模型本身进行采样，并通过周期性地将采样预算重新分配给那些看起来更可能达到目标稀有事件的前缀，从而改进了直接蒙特卡洛方法。为了在不学习额外提议分布的情况下指导这种重新分配，我们利用语言模型的内部表示作为前缀级别的信号，并通过实验验证了其有效性。我们对所提出的算法进行了理论分析。在涵盖小规模与中等规模安全相关任务的实验中（摘要在此处被截断）

    arXiv:2610.19202v1 Announce Type: new  Abstract: Rare but severe failures of large language models can become practically significant at deployment scale. Estimating their probabilities with direct Monte Carlo sampling is highly inefficient, while proposal-based methods typically require auxiliary learning of an additional proposal distribution. We introduce \textbf{Weighted Ensemble (WE)}, a training-free framework for rare-event estimation in language models. WE samples exclusively from the original language model and improves upon direct Monte Carlo by periodically reallocating the sampling budget toward prefixes that appear more likely to reach the target rare event. To guide this reallocation without learning a separate proposal distribution, we exploit internal LLM representations as prefix-level signals and empirically validate their effectiveness. We provide a theoretical analysis of the proposed algorithm. Experiments on safety-related tasks spanning both small-scale and moder
    
[^152]: NEPSEE：一个双重标注的尼泊尔语情感分析基准数据集

    NEPSEE: A Double-Annotated Benchmark for Nepali Sentiment Analysis

    [https://arxiv.org/abs/2610.19190](https://arxiv.org/abs/2610.19190)

    该论文构建了一个由两名标注者独立盲标的双标注尼泊尔语情感分析基准NEPSEE，发现商业大语言模型在细粒度五级情感标注任务上显著优于最佳编码器模型，并揭示了现有尼泊尔语BERT模型分词器会丢失天城文元音符号的问题。

    

    许多情感数据集对每个条目只使用一名标注者，因此测试标签反映的是该标注者个人的标注习惯。我们构建了一个尼泊尔语基准数据集，由两名标注者对相同的句子进行盲标。他们对来自尼泊尔公众人物推特的5,614个句子进行了五级量表标注，我们只保留了两位标注者给出相同极性的3,189个句子。在三类分类基准上，使用修正分词器的NepBERTa获得了0.877的宏F1分数，IndicBERTv2为0.867。三个商业大语言模型（LLM）零样本获得0.882到0.896的分数，且这些差异并不显著。在五级量表上，有766个句子被标注为相邻的两个标签，例如“积极”和“较积极”，我们同时接受这两个标签并在两者上进行训练。在这项任务上，Claude Opus 5.5的宏F1分数比最佳编码器高出0.10到0.12。分词器审查提供了第二个结果：三个尼泊尔语BERT模型的默认分词器会删除天城文的非间隔元音符号和止符（virama）。

    arXiv:2610.19190v1 Announce Type: new  Abstract: Many sentiment datasets use one annotator for each item, so the test labels show the conventions of that person. We build a Nepali benchmark from two annotators who labelled the same sentences blind. They labelled 5,614 sentences from tweets of Nepali public figures on a five-point scale. We keep only the 3,189 sentences where both give the same polarity. On the three-class benchmark, NepBERTa with a fixed tokenizer gets a macro-F1 of 0.877 and IndicBERTv2 0.867. Three commercial large language models (LLMs) get 0.882 to 0.896 zero-shot, and these differences are not significant. On the five-point scale, 766 sentences have two adjacent labels, for example Pos and SPos. We accept both labels and train on both. On this task, Claude Opus 5.5 beats the best encoders by 0.10 to 0.12 macro-F1. A tokenizer audit adds a second result. The default tokenizers of three Nepali BERT models delete the non-spacing Devanagari vowel signs and the virama.
    
[^153]: 当你不知道如何评估时该如何评估？为基础语言模型基准测试选择可靠的评估设置

    How to Evaluate When You Don't Know How to Evaluate? Choosing Reliable Setups for Base Language Model Benchmarking

    [https://arxiv.org/abs/2610.19185](https://arxiv.org/abs/2610.19185)

    本研究系统考察了任务表述、上下文示例数量等8个评估设计维度，并利用单调性、非随机性和排名一致性三种元度量标准，为基础语言模型在早期预训练阶段选择可靠的基准测试设置提供指导。

    

    大型语言模型（LLM）的评估在很大程度上仍处于前科学阶段：它涉及许多设计决策，而关于如何做出这些决策却缺乏指导。这些选择可能会显著影响测得的性能，进而影响关于后续预训练的决策。为了确定有效的评估设置，我们研究了8个方面，例如任务表述方式或上下文示例的数量。我们使用三种广泛使用的信号标准来比较这些设置，并将其重新用作规范性的元度量：评估分数在中间检查点之间的单调性、非随机性和排名一致性。在早期预训练场景中，我们在26个在不同类型数据上训练的单语2B参数模型上评估了这些设置，使用了四种语言的80个任务。我们还在相同四种语言的三个更大的LLM（≥7B）上检验了这些设置，并探讨了首选设置如何随模型规模变化。我们的结果得出（结论）……

    arXiv:2610.19185v1 Announce Type: new  Abstract: Evaluation of large language models (LLMs) remains largely pre-scientific: it involves many design decisions, with little guidance on how to make them. These choices can substantially affect measured performance and, consequently, decisions about subsequent pretraining. To identify effective evaluation setups, we investigate 8 aspects, such as task formulation or the number of in-context examples. We compare the setups using three widely-used signal criteria, repurposed as normative meta-measures: monotonicity, non-randomness, and ranking consistency of the evaluation scores across intermediate checkpoints. In an early pretraining scenario, we evaluate these setups on 26 monolingual 2B-parameter models trained on different types of data, using 80 tasks in four languages. We also examine the setups on three larger LLMs ($\geq$7B) in the same four languages and explore how the preferred setups change with model scale. Our results yield con
    
[^154]: HERO：在程序生成中利用证据进行测试时改进与提示优化

    HERO: Harnessing Evidence for Test-time Refinement and Prompt Optimization in Program Generation

    [https://arxiv.org/abs/2610.19184](https://arxiv.org/abs/2610.19184)

    HERO提出了一种结合离线与测试时优化的框架，通过执行验证的成功与失败证据学习可复用的提示更新，并将测试时生成的代码作为实例特定的harness加以改进，从而提升文本到SQL/Python等程序生成任务的可靠性与准确性。

    

    arXiv:2610.19184v1 公告类型：新 摘要：Harness（脚手架）优化通过改进可复用的提示和脚手架来提升大语言模型的表现，但需要针对文本到SQL、文本到Python等程序生成任务进行进一步适配。现有方法通常依赖LLM反思器来提出harness更新，这虽能修复观察到的失败，却也可能破坏此前正确的行为。文本到代码任务为缓解这一问题提供了天然途径：我们可以执行生成的输出，从而直接评估和验证所提出更新的效果。基于这一信号，我们提出了HERO（利用证据进行改进与优化），它将离线harness优化与测试时harness优化相结合。在第一阶段，HERO从经执行验证的成功与失败案例中学习可复用的提示更新，并在采纳前对候选修订进行验证。在测试时，我们将生成的代码视为特定实例的harness并进行改进……

    arXiv:2610.19184v1 Announce Type: new  Abstract: Harness optimization improves LLMs by refining reusable prompts and scaffolding, yet requires further adaptation for program generation tasks such as text-to-SQL and text-to-Python. Existing methods often rely on LLM reflectors to propose harness updates, which can fix observed failures but also disrupt previously correct behavior. Text-to-code tasks offer a natural way to mitigate this issue: we can execute generated outputs, allowing us to evaluate and verify the effects of proposed updates directly. Building on this signal, we introduce \textsc{HERO} (\textbf{H}arnessing \textbf{E}vidence for \textbf{R}efinement and \textbf{O}ptimization), which combines offline and test-time harness optimization. At the first stage, \method{} learns reusable prompt updates from execution-verified successes and failures and validates candidate revisions before adoption. At test time, we treat the generated code as an instance-specific harness and refi
    
[^155]: 奇妙的指标及其寻找方法：面向多语言风格个性化生成的指标元评估

    Fantastic Metrics and How to Find Them: Meta-Evaluating Metrics for Multi-lingual Style-Personalized Generation

    [https://arxiv.org/abs/2610.19166](https://arxiv.org/abs/2610.19166)

    提出基于回归的指标元评估框架，以预测人类评分替代传统相关性方法，并构建了覆盖10种语言、4个评估维度的大规模多语言风格个性化文本生成基准来检验指标的真实评估能力。

    

    文本生成指标和大语言模型判官的元评估主要采用指标与人类评分之间的相关性来进行。然而，先前的研究表明，相关性并非衡量指标评估能力的恰当方式，且在不同评估设置（如不同数据集或语言）下可能不稳定。因此，我们提出了一种基于回归的指标元评估方法，该方法通过在校准后的指标输出上学习来预测人类评分。我们将这一测量相对于一个简单的常数均值基线以及人类评分者间一致性进行校准，从而提供一种跨评估设置可比较的指标能力度量。为了检验我们的元评估框架，我们构建了一个大规模的多语言风格个性化文本生成（SPTG）基准，在10种拓扑结构多样的语言和四个具有不同程度主观性的评估维度上收集了多条人工标注。我们对来自三种风格……的32个指标进行了元评估。

    arXiv:2610.19166v1 Announce Type: new  Abstract: Meta-evaluation of text generation metrics and LLM-judges is primarily conducted using metric-human correlations. However, prior work has shown that correlation is not an appropriate estimate of a metric's evaluation capability and can be unstable across evaluation settings, such as different datasets or languages. Therefore, we propose a regression-based meta-evaluation of metrics that learns on calibrated metric outputs to predict human ratings. We ground this measurement relative to a trivial constant-mean baseline and human agreements to provide a comparable measure of metric capability across evaluation settings. To test our meta-evaluation framework, we developed a large-scale multi-lingual style-personalized text generation (SPTG) benchmark, collecting multiple annotations across 10 topologically diverse languages and four evaluation dimensions, each with varying degrees of subjectivity. We meta-evaluated 32 metrics from three sty
    
[^156]: ListQA：用于评估大语言模型列表格式事实知识检索的基准

    ListQA: A Benchmark for Evaluating List-Formatted Factual Knowledge Retrieval in Large Language Models

    [https://arxiv.org/abs/2610.19165](https://arxiv.org/abs/2610.19165)

    该论文提出了ListQA基准，包含9,045个人工策划并交叉验证的问题，用于评估大语言模型回忆多个事实并组织成结构化列表的能力，并采用最优二分图匹配与LLM语义评分相结合的评估方法来衡量事实准确性与幻觉。

    

    许多现实世界中的问题需要的不是单一事实，而是一组有组织的事实集合，然而现有的事实知识基准几乎都只针对单一答案的检索。我们推出了ListQA，这是一个包含9,045个人工策划、交叉验证问题（经审计的错误率为1.7%）的基准，要求大语言模型回忆多个事实并将其组织成包含3-10个元素的结构化列表。问题涵盖从扁平列表（“2000年至2024年间哪些NCAA球队保持不败？”）到带有子属性的层级列表（“……以及他们的战绩和结果如何？”），横跨八大类别和60多个国家。常青设计（明确的时间约束或历史上不可改变的事实）使得标准答案无需定期更新即可保持有效。在评估方面，我们将元素匹配框架化为线性求和分配问题：最优二分图匹配与基于大语言模型的语义评分相结合，共同捕捉事实准确性、幻觉……

    arXiv:2610.19165v1 Announce Type: cross  Abstract: Many real-world questions demand not a single fact but an organized collection of them, yet existing factual knowledge benchmarks almost exclusively target single answer retrieval. We introduce ListQA, a benchmark of 9,045 human-curated, cross-validated questions (audited error rate of 1.7%) that require LLMs to recall multiple facts and compose them into structured lists of 3-10 elements. Questions range from flat lists ("Which NCAA teams went undefeated between 2000 and 2024?") to hierarchical lists with sub-attributes ("...and what was their record and result?"), spanning eight categories across 60+ countries. An evergreen design (explicit temporal constraints or historically immutable facts) keeps ground truth valid without periodic updates. For evaluation, we frame element matching as a linear sum assignment problem: optimal bipartite matching paired with LLM-based semantic grading jointly captures factual accuracy, hallucination,
    
[^157]: 从定性文本到整合证据：平行-后整合（PTI）程序

    From Qualitative Text to Integrated Evidence: The Parallel-Then-Integrated (PTI) Procedure

    [https://arxiv.org/abs/2610.19164](https://arxiv.org/abs/2610.19164)

    本文提出了PTI（平行-后整合）这一四步程序，用于在混合方法研究中显式且可追溯地整合同一定性语料库的两种计算表征（本体论指导的PV编码与NMF数据驱动主题），并以尼泊尔母亲的访谈数据和PHQ-9抑郁评分作为外部锚点加以演示。

    

    混合方法研究为整合定性与定量证据提供了成熟的策略，但对于整合同一定性语料库的两种计算表征则缺乏指导。这项方法学研究提出了“平行-后整合”（Parallel-Then-Integrated, PTI），一种用于显式且可追溯整合的四步程序。我们使用来自尼泊尔29位母亲的1,203个访谈段落来展示PTI。这两种表征分别是以本体论为指导、借助PVminer工具进行的患者声音（Patient Voice, PV）编码，以及通过非负矩阵分解（NMF）生成的数据驱动主题。一个“主题-完整”对照表记录了两者之间的趋同、分歧以及对预处理敏感的差异。基于患者健康问卷-9（PHQ-9）评分的抑郁状态作为探索性关联检验的次级外部锚点。PTI为在混合方法研究中整合定性文本的计算表征提供了一种结构化的方法。

    arXiv:2610.19164v1 Announce Type: new  Abstract: Mixed methods research offers established strategies for integrating qualitative and quantitative evidence, but less guidance for integrating two computational representations of the same qualitative corpus. This methodological study introduces Parallel-Then-Integrated (PTI), a four-step procedure for explicit and traceable integration. We present PTI using 1,203 interview paragraphs from 29 mothers in Nepal. The two representations were ontology-guided Patient Voice (PV) coding with PVminer and data-driven themes generated through non-negative matrix factorization (NMF). A theme-complete crosswalk documented convergence, divergence, and preprocessing-sensitive differences. Depression status, based on Patient Health Questionnaire-9 (PHQ-9) scores, served as a secondary external anchor for exploratory association checks. PTI provides a structured approach for integrating computational representations of qualitative text within mixed metho
    
[^158]: 进入未知领域：在开放科学研究问题上对AI智能体进行基准测试

    Into the Unknown: Benchmarking AI agents on open scientific research problems

    [https://arxiv.org/abs/2610.19159](https://arxiv.org/abs/2610.19159)

    该论文推出了"Into the Unknown"基准测试，通过与能源部国家实验室科学家合作定义未发表的后续研究方向及评分准则，评估九个前沿AI智能体在开放科学问题上撰写后续论文的能力，结果显示135篇投稿无一满足全部条件，其中114篇未通过完整性检查。

    

    AI智能体能够对开放性科学问题开展研究吗？我们推出了"Into the Unknown"（进入未知领域）基准测试，该基准为智能体提供一篇科学论文以及一个待探索的研究方向，并要求其撰写一篇后续论文。我们与美国能源部（DOE）国家实验室的科学家合作，每位科学家都贡献了自己工作中一个未发表的后续研究方向，以及投稿必须满足的必要条件。虽然评估一篇后续论文的质量总体上具有挑战性，但利用这些条件形成评分准则，使得对没有参考答案的研究问题的评估更加可行。我们让九个前沿智能体各自撰写15篇后续论文，涵盖量子计算、等离子体物理、计算化学、材料科学和科学机器学习等领域。在135篇投稿中，没有一篇满足所有条件，其中114篇未通过我们的完整性检查，最常见的原因是……

    arXiv:2610.19159v1 Announce Type: cross  Abstract: Can AI agents carry out research on open scientific problems? We introduce Into the Unknown, a benchmark that provides an agent with a science paper along with a research direction to pursue and tasks it to write a follow-up paper. We work with scientists at US Department of Energy (DOE) national laboratories, who each contribute an unpublished follow-up direction from their own work together with the necessary conditions a submission must meet. While assessing the quality of a follow-up paper is challenging in general, using these conditions to form a rubric makes evaluation more tractable for research problems that have no reference answer. We have nine frontier agents write 15 follow-ups each, spanning quantum computing, plasma physics, computational chemistry, materials science, and scientific machine learning domains. None of the 135 submissions satisfies all the conditions, and 114 fail our integrity check, most often because the
    
[^159]: DivAlgo：代码生成模型算法多样性的系统性评估

    DivAlgo: A systematic evaluation of algorithmic diversity for code generation models

    [https://arxiv.org/abs/2610.19157](https://arxiv.org/abs/2610.19157)

    提出DivAlgo基准数据集及轻量级相似度度量方法，通过对24个代码生成模型的评估发现，模型生成代码的算法多样性显著低于人类解决方案（最高仅达人类多样性的约70%），揭示了当前代码生成模型在算法多样性方面的局限性。

    

    代码生成模型正被积极应用于专业和教育场景中。然而，大型语言模型（LLM）生成算法多样性代码的能力在很大程度上仍不为人知。我们提出了DivAlgo基准，这是一个包含针对编程问题的算法多样化解决方案的大规模数据集。我们还引入了一种轻量级的相似度度量方法，以支持大规模评估。我们在24个现代代码生成模型上的实验发现，不受控制的模型生成代码的多样性显著低于人类解决方案（仅捕捉到人类多样性的约31%），即使是最佳的多样性代码生成策略也只能达到人类多样性的约70%。此外，虽然可以引导模型基于人类策略生成解决方案，但它们无法再现人类解决方案的多样性。这些发现表明代码生成模型在捕捉算法多样性方面存在局限。

    arXiv:2610.19157v1 Announce Type: cross  Abstract: Code generation models are actively being used in professional and educational settings. However, the competence of large language models (LLMs) to generate algorithmically diverse code remains largely unknown. We propose the DivAlgo benchmark, a large-scale dataset constituting algorithmically diverse solutions to programming problems. We also introduce a lightweight similarity metric to enable large-scale evaluation. Our experiments across 24 modern code generation models found that uncontrolled model-generated code is significantly less diverse than human solutions (capturing ~31% of human diversity), and even the best diverse code generation strategy could achieve only ~70% of human diversity. Moreover, while it was possible to steer the models to generate solutions based on human strategies, they were unable to reproduce the diversity of the human solutions. These findings indicate the limitations of code generation models in capt
    
[^160]: MA-Bench：测量语言模型的作战规划能力

    MA-Bench: Measuring Operational Planning in Language Models

    [https://arxiv.org/abs/2610.19138](https://arxiv.org/abs/2610.19138)

    提出了基于军事作战规划第一步“任务分析”构建的MA-Bench基准，用于评估语言模型在缺少唯一正确答案的情况下，从不完整且矛盾的文档集合中形成态势分析报告的能力。

    

    arXiv:2610.19138v1 公告类型：交叉（cross） 摘要：当不存在唯一正确答案时，语言模型能在多大程度上将庞大、不完整且部分矛盾的文档集合转化为一份计划？我们提出了任务分析基准（MA-Bench），该基准建立在军事作战规划的第一个步骤——任务分析之上。在这一步骤中，参谋人员需要通读全部态势信息，并在任何人提出解决方案之前先陈述问题。该基准包含10个由专家构建的场景，每个场景将10至12份文档组成的合成资料包与1,257条由专家编写的通过/失败要求相配对，每条要求均包含其来源、问题类型和重要性级别的信息。一个推理模型评判器在不查看文档的情况下，依据这些要求对每份简报进行评分；其评分表现如同第五位持证规划人员，且不改变由四位规划人员组成的评审小组的评分一致性（Fleiss κ 从0.42变为0.41），而来自另一家提供商的评判器也能重现相同的模型排名（Spearman 0.94）。在二十个前沿模型和开源权重模型中，表现最好的模型得到62%……（原文摘要至此截断）

    arXiv:2610.19138v1 Announce Type: cross  Abstract: How well can a language model turn a large, incomplete and partly contradictory body of documents into a plan when no single correct answer exists? We introduce Mission Analysis (MA)-Bench, built on mission analysis, the first step of military operational planning, in which a staff reads the full situation and states the problem before anyone proposes a solution. Its 10 expert-built scenarios pair synthetic packets of ten to twelve documents with 1,257 expert-written pass/fail requirements, each containing information on their source, question type, and importance level. A reasoning-model judge grades each brief against the requirements without seeing the documents; it grades like a fifth credentialed planner, leaving a four-planner panel's agreement unchanged (Fleiss $\kappa$ 0.42 to 0.41), and a judge from a second provider reproduces the model ranking (Spearman 0.94). Across twenty frontier and open-weights models the best gets 62% 
    
[^161]: CAFE：将特征蒸馏限制在语言路由适配器中，以实现紧凑型多语言自动语音识别

    CAFE: Confining Feature Distillation to Language-Routed Adapters for Compact Multilingual Automatic Speech Recognition

    [https://arxiv.org/abs/2610.19134](https://arxiv.org/abs/2610.19134)

    提出CAFE方法，通过将CKA特征蒸馏梯度限制在英语绕过的语言路由适配器中，从参数层面解决了紧凑型多语言语音识别中非英语与英语性能之间的权衡问题。

    

    特征蒸馏通常被认为能够补充logit蒸馏的效果。我们发现，在紧凑型多语言自动语音识别（ASR）中，这一预期对两种常见的特征损失并不成立，并通过分析它们的梯度来探究原因。基于投影的损失函数的梯度与任务梯度正交，而基于相似性的损失函数中心核对对齐（CKA）的梯度与任务梯度仅呈弱对齐。CKA降低了非英语语言上的词错误率（WER），但以牺牲数据丰富的英语为代价。三种损失层面的补救措施（一种通过训练，两种通过探测）均未能解决问题，且CKA梯度与英语在共享编码器中所使用的子空间的重叠度高达97.5%至99.5%（随机情况下约为81%）。基于这一诊断，我们提出了CAFE（受限适配器特征蒸馏），它将CKA梯度限制在英语所绕过的适配器中，从参数层面而非损失层面来解决这一权衡问题。在Open ASR...

    arXiv:2610.19134v1 Announce Type: new  Abstract: Feature distillation is usually expected to add to logit distillation. We find that this expectation does not hold for two common feature losses in compact multilingual automatic speech recognition (ASR), and analyze why through their gradients. The gradient of a projection-based loss is orthogonal to the task gradient, while that of Centered Kernel Alignment (CKA), a similarity-based loss, is weakly aligned with it. CKA lowers the word error rate (WER) on non-English languages, but at the expense of data-rich English. Three loss-level remedies (one trained, two probed) all fall short, and the CKA gradient overlaps with the subspace English uses in the shared encoder by 97.5% to 99.5% (about 81% by chance). Guided by this diagnosis, we propose CAFE (Confined-Adapter Feature distillation), which confines the CKA gradient to adapters that English bypasses, addressing the trade-off in the parameters rather than in the loss. On the Open ASR 
    
[^162]: Plant Memory（工厂记忆）：基于技术语言处理与经核验的维护本体，将SAP PM工单转化为智能体可访问的故障知识

    Plant Memory: Technical Language Processing and a Verified Maintenance Ontology for Agent-Accessible Failure Knowledge from SAP PM Work Orders

    [https://arxiv.org/abs/2610.19131](https://arxiv.org/abs/2610.19131)

    提出了Plant Memory系统，通过规则优先、LLM辅助的技术语言处理，将难以分析的SAP PM维护工单转化为符合ISO 14224标准、每条事实均带置信度、证据溯源与核验状态的维护本体，使设备故障知识可被智能体访问，避免诊断知识随资深技师流失。

    

    维护工单是记录工业设备如何失效的最庞大书面资料，但在大多数工厂中，这些工单几乎无法用于分析。文本简短且充满缩写和拼写错误，资产标识被埋没在自由文本中，工作类型标签与所描述的工作相互矛盾，而对可靠性最为重要的字段（症状、根本原因、所采取的措施）通常是空白的。其结果是，诊断知识只留存于资深技术人员的手中，并随他们的离职而流失。我们提出了Plant Memory，这是一个将技术语言处理（TLP）应用于SAP工厂维护（PM）导出数据的系统，它构建了一个符合ISO 14224标准的维护本体，其中每一条被提取的语句都是一个具体化的事实，附带置信度分数、其读取来源的证据以及核验状态。提取以规则优先，可选的大语言模型（LLM）充当“第二读者”，其与规则的分歧会提升相应事实的复核优先级，而不是直接覆盖……

    arXiv:2610.19131v1 Announce Type: cross  Abstract: Maintenance work orders are the largest written record of how industrial equipment fails, yet in most plants they are close to unusable for analysis. Short texts are abbreviated and misspelt, asset identity is buried in free text, work-type labels contradict the work described, and the fields that matter most for reliability (symptom, root cause, action taken) are usually empty. As a result, diagnostic knowledge stays with senior technicians and leaves when they do. We present Plant Memory, a system that applies technical language processing (TLP) to SAP Plant Maintenance (PM) exports and builds an ISO 14224-aligned maintenance ontology in which every extracted statement is a reified fact carrying a confidence score, the evidence it was read from, and a verification state. Extraction is rule-first, with an optional large language model (LLM) acting as a second reader whose disagreements raise a fact's review priority rather than overwr
    
[^163]: 通过保持注意力的MLP裁剪解锁推理语言模型的结构化剪枝

    Unlocking Structured Pruning for Reasoning Language Models via Attention-Preserving MLP Trimming

    [https://arxiv.org/abs/2610.19121](https://arxiv.org/abs/2610.19121)

    LayerTrim是一种免训练的结构化剪枝方法，通过识别冗余解码器层并选择性裁剪其MLP特征、同时保留注意力机制和解码器深度，在无需梯度或恢复训练的情况下保持长篇推理能力，相比RESP平均准确率最高提升21.4个百分点，并实现1.24倍的预填充加速和19.7%的GPU峰值内存降低。

    

    结构化剪枝通过移除模型组件来生成更小的稠密模型，从而降低大语言模型的推理成本。然而，现有方法可能导致在AIME等长篇推理任务上的推理性能崩溃。我们提出了LayerTrim，这是一种免训练的结构化剪枝方法，它识别冗余的解码器层，并选择性地裁剪其MLP特征，同时保留注意力机制和解码器深度。LayerTrim是首个无需梯度或恢复训练即可保持长篇推理能力的免训练结构化剪枝方法。在三个推理语言模型和五个基准测试上，LayerTrim在大多数评估设置中优于现有的结构化剪枝方法，相比RESP平均准确率提升最高达21.4个百分点。在Qwen3-14B上，LayerTrim相比稠密模型实现了1.24倍的预填充加速和19.7%的峰值GPU内存降低，同时更好地保留了推理能力。

    arXiv:2610.19121v1 Announce Type: cross  Abstract: Structured pruning reduces LLM inference costs by removing model components to produce smaller dense models. However, existing methods can cause reasoning performance to collapse on long-form reasoning tasks such as AIME. We propose \textit{LayerTrim}, a training-free structured pruning method that identifies redundant decoder layers and selectively trims their MLP features while preserving attention and decoder depth. LayerTrim is the first training-free structured pruning method designed to preserve long-form reasoning without gradients or recovery training. Across three reasoning language models and five benchmarks, LayerTrim outperforms existing structured pruning methods in most evaluated settings, improving average accuracy by up to 21.4 percentage points over RESP. On \texttt{Qwen3-14B}, LayerTrim achieves 1.24$\times$ faster prefill and 19.7\% lower peak GPU memory than the dense model, while better preserving reasoning ability
    
[^164]: DecisionRobust：面向文本与视觉的JEV风格决策模型鲁棒性基准测试

    DecisionRobust: Benchmarking Robustness for JEV Style Decision Models Across Text and Vision

    [https://arxiv.org/abs/2610.19118](https://arxiv.org/abs/2610.19118)

    本文提出DecisionRobust基准，通过涵盖语义保持文本扰动、图像扰动和提示注入等多种攻击方式，系统性评估了JEV风格决策模型在文本与视觉输入下面对对抗性操纵的鲁棒性。

    

    arXiv:2610.19118v1 公告类型：新论文 摘要：诸如JEV等新兴决策模型提供了类型化的接口，能够将任务指令和观察结果直接映射到选择、判断和评分。随着这些模型越来越多地被用于自动化决策，理解它们在对抗性操纵下的可靠性变得至关重要。我们提出了DecisionRobust，这是一个用于系统性评估决策模型在文本和视觉输入下鲁棒性的基准。该基准涵盖实际的决策场景以及多样化的攻击套件，包括保持语义的文本扰动、图像扰动以及两种模态下的提示注入。我们通过外部有效性检查和跨模型候选选择来构建评估集。我们在文本任务上评估了30个模型配置，在视觉任务上评估了17个模型配置，其中包括分别从候选选择中排除的24个和12个留出模型。我们的实验揭示了若干关键发现。

    arXiv:2610.19118v1 Announce Type: new  Abstract: Emerging decision models such as JEV provide typed interfaces that map task instructions and observations directly to choices, judgments, and scores. As these models are increasingly used for automated decision-making, understanding their reliability under adversarial manipulation becomes critical. We introduce DecisionRobust, a benchmark for systematically evaluating the robustness of decision models across textual and visual inputs. Our benchmark covers practical decision-making scenarios and a diverse suite of attacks, including semantics-preserving text perturbations, image perturbations, and prompt injection in both modalities. We curate evaluation sets through external validity checks and cross-model candidate selection. We evaluate 30 model configurations on text tasks and 17 on vision tasks, including 24 and 12 held-out models, respectively, that are excluded from candidate selection. Our experiments reveal several key findings. 
    
[^165]: 从复制到组合：语言模型如何发展出上下文学习能力

    From Copying to Composition: How Language Models Develop In-Context Learning

    [https://arxiv.org/abs/2610.19113](https://arxiv.org/abs/2610.19113)

    该研究通过受控诊断方法发现，语言模型会从倾向于复制提示中最近示例，逐渐发展为整合提示信息进行组合式推理的上下文学习，且这种转变出现得很早，并可通过特定的合成数据训练加以引导。

    

    上下文学习使语言模型能够根据提示中的指令和示例调整自身行为，但仅凭准确率无法揭示上下文信息是如何控制模型预测的。我们通过一个受控诊断方法研究ICL的习得与可训练性，该方法将“复制提示中最近的示例”与“整合提示中跨上下文的信息”区分开来。在17个模型中，约三分之一的模型倾向于复制答案，少数模型摇摆不定，其余模型则倾向于符合规则的答案，这与更高的准确率相关，且其预测受所请求的值控制而非受最近的冲突示例控制。检查点分析表明，模型在准确率超越随机水平之前很早就开始优先处理与任务相关的提示词元，而非冲突示例。在匹配训练实验中，先定义词尾再查询词尾的合成文档使模型在受控诊断上转向组合式响应。

    arXiv:2610.19113v1 Announce Type: new  Abstract: In-context learning (ICL) allows language models to adapt their behavior from instructions and demonstrations in a prompt, but accuracy alone does not reveal how contextual information controls their predictions. We study the acquisition and trainability of ICL with a controlled diagnostic that separates copying a nearby demonstration from integrating information across the prompt. Across 17 models, about a third prefer the copied answer, a few are undecided, and the rest prefer the rule-consistent answer, which goes with higher accuracy and with control by the requested value rather than the closest conflicting example. Checkpoint analyses show that models begin prioritizing task-relevant prompt tokens over conflicting examples long before their accuracy improves beyond chance level. In matched training experiments, synthetic documents that define endings and then query them shift models toward compositional responses on the controlled 
    
[^166]: 迈向对多模态大语言模型空间推理的以数据为中心的理解

    Towards a Data-Centric Understanding of Spatial Reasoning in Multimodal Large Language Models

    [https://arxiv.org/abs/2610.19108](https://arxiv.org/abs/2610.19108)

    本综述以数据为中心系统梳理了用于多模态大语言模型空间推理的微调方法，将研究归纳为数据集设计、数据集构建与评估三大类，并总结出规则与模板、LLM 提示、程序化生成和人在回路四种主要的空间标注构建方法。

    

    多模态大语言模型（MLLMs）已被广泛应用于具身智能体和机器人领域的视觉任务中。然而，MLLMs 在空间推理和感知方面存在局限性。后训练技术已展现出提升这些能力的潜力；然而，针对此类方法的以数据为中心的调研仍然有限。在本综述中，我们考察了为空间推理任务对预训练 MLLMs 进行微调的各种方法，涵盖从监督学习到强化学习，并重点关注所使用的数据集。我们研究了数据集构建、数据模态与标注以及所得到模型的评估等主题。基于这些主题，我们将相关研究分为三类：数据集设计、数据集构建与评估。我们发现了构建固定空间标注的四种主要方法：规则与模板、LLM 提示、程序化生成以及人在回路。我们还观察到了相当大的差异……

    arXiv:2610.19108v1 Announce Type: cross  Abstract: Multimodal Large Language Models (MLLMs) have been widely applied in embodied agents and robotics for vision tasks. However, MLLMs suffer from limitations in spatial reasoning and perception. Post-training techniques have shown promise for improving these capabilities; yet, there are limited data-driven surveys of such approaches. In this survey, we consider approaches to fine-tuning pretrained MLLMs for spatial reasoning tasks, ranging from supervised to reinforcement learning, with an emphasis on the datasets used. We investigate themes of dataset construction, data modality and annotation, and evaluation of the resulting models. Based on these themes, we categorize research into three groups: dataset design, dataset construction and evaluation. We find four main methods for constructing fixed spatial annotations: rules and templates, LLM prompting, procedural generation, and human-in-the-loop. We also observe substantial variation i
    
[^167]: GroundGUI：基于高质量数据与蒸馏到强化学习的GUI视觉定位

    GroundGUI: GUI Grounding by High-Quality Data and Distillation-to-Reinforcement Learning

    [https://arxiv.org/abs/2610.19100](https://arxiv.org/abs/2610.19100)

    本文提出了涵盖桌面、网页和移动端的大规模GUI定位数据集ScreenRef（包含586K张截图和633K个指代表达实例），并提出蒸馏到强化学习（D2RL）方法，以克服监督微调泛化能力有限的问题，显著提升GUI智能体的视觉定位能力。

    

    构建强大的GUI智能体依赖于多模态视觉定位能力，即将自然语言意图精确映射到屏幕上可操作的组件。尽管桌面、网页和移动端的基准测试十分丰富，但针对多样化环境定制的高保真资源却出奇地稀缺。为了弥合这一差距，我们引入了屏幕指代定位数据集（ScreenRef），这是一个全面的GUI定位数据集，由GUI截图与自然语言指代表达以及边界框或点标注配对组成。ScreenRef涵盖桌面、网页和移动端领域，包含586K张截图和超过633K个指代表达实例，涵盖文本识别、视觉匹配、空间理解和拒绝场景等任务。为了提升模型的定位性能，我们提出了蒸馏到强化学习（D2RL）方法，以解决监督微调泛化能力有限和强化学习稀疏（摘要在此处截断）……

    arXiv:2610.19100v1 Announce Type: cross  Abstract: Constructing robust GUI agents relies on multimodal visual grounding: the ability to precisely map natural language intents to actionable on-screen components. Despite abundant desktop, web, and mobile benchmarks, high-fidelity resources tailored for diverse environments are surprisingly scarce. To bridge this gap, we introduce the Screen Referring Grounding Dataset (ScreenRef), a comprehensive GUI grounding dataset consisting of GUI screenshots paired with natural language referring expressions and box or point annotations. Spanning desktop, web, and mobile domains, ScreenRef comprises 586K screenshots and over 633K referring expression instances, covering tasks such as text recognition, visual matching, spatial understanding, and refusal scenarios. To boost the model's grounding performance, we propose the Distillation-to-Reinforcement Learning (D2RL) method to address the limited generalization of supervised fine-tuning and the spar
    
[^168]: 相似的模型，失灵的胜率：Luce公理在Chatbot竞技场中的失效之处

    Similar Models, Broken Odds: Where Luce's Axiom Fails in Chatbot Arena

    [https://arxiv.org/abs/2610.19095](https://arxiv.org/abs/2610.19095)

    该研究发现Chatbot Arena所依赖的Bradley-Terry模型（即Luce选择公理）在高度相似的模型之间会系统性失效——相似模型间的对决比分数预测的更具决定性（最高达1.37倍），并提出一个仅含4-6个参数、主要通过建模平局的修正方法，其预测能力超越了Bradley-Terry及含数百参数的复杂模型。

    

    Chatbot竞技场使用Bradley-Terry模型对语言模型进行排名，该模型为每个模型赋予单一分数，因而隐含地假设了Luce选择公理。选择理论预言了该公理会在何处失效：近似复制的模型被区分的果断程度会超出其分数所暗示的水平。基于180万条竞技场投票数据，同一模型不同版本之间的投票果断程度是Bradley-Terry预测的1.37倍，同一代不同参数规模之间为1.26倍，同一实验室不同模型之间为1.12倍；一年后的新发布模型也呈现出一致的乘数，且该效应在替代性的平局处理方式、提示词子集与主题划分以及时间检验下依然稳健。两个模型回答在措辞和格式上的相似程度能够追踪这一效应。一个仅含四到六个参数的修正模型在预测留出集投票方面优于Bradley-Terry模型以及拥有数百个参数的模型，这主要归功于对平局的建模；在预先设定的测试中，将该修正模型冻结并应用于新发布的模型时，其收益集中于平局情形，并在成对比较中表现更优。

    arXiv:2610.19095v1 Announce Type: new  Abstract: Chatbot Arena ranks language models with the Bradley-Terry model, which gives each model one score and so assumes Luce's choice axiom. Choice theory predicts where this fails: near-copies are told apart more decisively than their scores imply. On 1.8 million Arena votes, votes between versions of one model are 1.37 times as decisive as Bradley-Terry predicts, between sizes of one generation 1.26 times and between models of one lab 1.12 times; a release a year later gives consistent multipliers, and the effect survives alternative tie treatments, prompt subsets and topics, and timing checks. How alike two models' answers are worded and formatted tracks the effect. A correction with four to six parameters predicts held-out votes better than Bradley-Terry and than models with hundreds of parameters, mostly by modelling ties; frozen and carried to new releases in a test fixed in advance, its gains are concentrated in ties, and better head-to
    
[^169]: 超越总体准确率：提示优化中的题目级转变

    Beyond Aggregate Accuracy: Item-Level Transitions in Prompt Optimization

    [https://arxiv.org/abs/2610.19084](https://arxiv.org/abs/2610.19084)

    该论文提出RFDPO方法，通过追踪提示优化过程中逐题的“恢复、退化与变动”转变，揭示总体准确率所掩盖的题目级变化，并利用这些反馈指导后续的提示修订。

    

    自动提示优化（APO）利用模型响应的反馈来自动改进提示。然而，其有效性通常通过总体分数来衡量，这种方式无法揭示哪些题目由错变对、哪些原本正确的答案变为了错误，以及改进是否能推广到未见过的题目。我们提出了反思式反馈驱动提示优化（RFDPO），这是一种追踪“恢复”（由错变对）、“退化”（由对变错）和“变动”（所有发生变化的结果）的工具。RFDPO 利用这些反馈来指导后续修订，并在整个优化过程中记录提示的变化。我们在涵盖法律、知识、指令遵循、数学和隐私五个领域的基准上评估了 RFDPO，使用 GPT-5 作为优化器，GPT-4o-mini 和 GPT-4.1 Mini 作为求解器。我们的结果表明，总体准确率可能掩盖显著的变化：在九个配置中的五个里，变动量至少是……的五倍（原文摘要在此处被截断）。

    arXiv:2610.19084v1 Announce Type: cross  Abstract: Automatic prompt optimization (APO) uses feedback from model responses to improve prompts automatically. However, its effectiveness is usually measured through overall scores, which do not reveal which questions become correct, which previously correct answers become wrong, or whether improvements extend to unseen questions. We introduce Reflective Feedback-Driven Prompt Optimization (RFDPO), a tool that tracks recoveries (incorrect to correct), regressions (correct to incorrect), and churn (all changed outcomes). RFDPO uses this feedback to guide subsequent revisions and records prompt changes throughout optimization. We evaluate RFDPO across five benchmarks covering law, knowledge, instruction following, mathematics, and privacy, using GPT-5 as the optimizer and GPT-4o-mini and GPT-4.1 Mini as solvers. Our results show that overall accuracy can hide substantial changes: in five of nine configurations, churn was at least five times th
    
[^170]: 独自聪明，集体失明：为什么准确性无法认证多智能体团队中的心智理论

    Smart Alone, Blind Together: Why Accuracy Fails to CertifyTheory of Mind in Multi-Agent Teams

    [https://arxiv.org/abs/2610.19051](https://arxiv.org/abs/2610.19051)

    该论文论证准确性不能作为认证多智能体团队中心智理论模块的标准，因为所需准确性取决于部署场景的错误代价，且模块间错误的相关性会使准确性阈值以可预测的方向失效。

    

    基于对队友推断而采取行动的协调者，通常以这些推断的准确性作为认证标准。我们论证准确性是错误的认证依据，原因有二。首先，行动所需的准确性取决于错误行动的代价，因此这一标准应属于部署场景，而非模块本身。其次，模块的错误很少是独立的，且这种相关性有方向性：正相关的错误在其损害重叠时有益，而在损害叠加时有害，因此准确性阈值会以可预测的方向失效——一旦错误信念相互作用并可能造成伤害，就不存在对每个模块都正确的基于准确性的规则。我们在任务分配、Level-Based Foraging 和 Hanabi 中检验这些预测，使用了注入错误、训练的和贝叶斯模块、五个语言模型以及人类判断。准确性在两个方向上都失效了：在共享错误叠加的 Hanabi 中批准了有害的语言模型部署，同时拒绝了……（原文截断）

    arXiv:2610.19051v1 Announce Type: cross  Abstract: Coordinators that act on what they infer about teammates are usually certified by the accuracy of those inferences. We argue that accuracy is the wrong certificate, for two reasons. First, the accuracy needed to act depends on what a wrong action costs, so the bar belongs to the deployment, not the module. Second, a module's errors are rarely independent, and dependence has a sign: positively associated errors help where their damage overlaps and hurt where it compounds, so an accuracy threshold fails in a predictable direction, and once wrong beliefs interact and can do harm, no rule based on accuracy is right for every module. We test these predictions in task assignment, Level-Based Foraging and Hanabi, with injected errors, trained and Bayesian modules and five language models, and in human judgments. Accuracy fails in both directions, approving harmful language-model deployments in Hanabi, where shared errors compound, and rejecti
    
[^171]: 预训练语言模型作为社会科学研究的零样本工具

    Pre-Trained Language Models as Zero-Shot Tools for Social Science Research

    [https://arxiv.org/abs/2610.19048](https://arxiv.org/abs/2610.19048)

    该论文证明预训练语言模型无需任务特定训练即可完成社会科学中的记录链接和种族推断任务，提出的EnsembleLink算法达到或超越现有方法的表现，且大语言模型在美国选民档案种族推断上达到83.8%的准确率，超过传统BISG方法的78.0%。

    

    基于姓名的记录链接和人口统计推断是社会科学中的常规需求。解决这些问题的多数方法都需要标注训练数据，而这些数据往往难以获取或成本高昂。我证明了预训练语言模型无需任务特定训练即可准确完成这两项任务。对于记录链接，我提出了EnsembleLink算法，该算法将预训练嵌入与基于字符串的评分器相结合，先检索出候选名单，再识别最可能的匹配。EnsembleLink的表现与大多数先前方法相当甚至更优，并且能够泛化到那些方法无法处理的文字系统和数据格式。对于基于姓名的种族推断，我表明大语言模型（LLM）在美国选民档案上达到高达83.8%的准确率，相比之下补充姓名信息的BISG方法仅为78.0%。开放权重模型提供了可复现的实现，其准确率接近这些顶级模型。此外，LLM（摘要内容在此处被截断）

    arXiv:2610.19048v1 Announce Type: new  Abstract: Record linkage and demographic inference from names are routine demands in social science. Most approaches to these problems require labeled training data that are difficult or costly to acquire. I show that pre-trained language models perform both tasks accurately without requiring task-specific training. For record linkage, I introduce EnsembleLink, an algorithm that uses pre-trained embeddings alongside string-based scorers to retrieve a short list of candidates and then identify the likeliest matches. EnsembleLink matches or exceeds the performance of most prior approaches while generalizing across scripts and formats that those methods are unable to handle. For ethnicity inference from names, I show that large language models (LLMs) achieve up to 83.8% accuracy on U.S. voter files, compared to 78.0% for name-supplemented BISG. Open-weight models offer reproducible implementations that approach these models in accuracy. Moreover, LLM
    
[^172]: 图粗化使计算受限的非凸MAX-CUT问题能够逃离局部最优

    Graph Coarsening Enables Escaping Local Optima in Compute-Constrained Nonconvex MAX-CUT

    [https://arxiv.org/abs/2610.19043](https://arxiv.org/abs/2610.19043)

    该论文提出将近线性时间的谱图粗化方法CMG用于热启动具有优良优化景观的粒子多项式松弛（PPR），使计算受限的非凸MAX-CUT优化能够逃离局部最优，在基准实例上平均达到97.5%的最优解、并在全部Biq Mac测试实例上达到已发表最优解，还可扩展至两万顶点规模并大幅降低内存需求。

    

    由于MAX-CUT的二值优化景观包含指数级数量的局部最优解，该问题对初始化高度敏感。我们研究了粒子多项式松弛（PPR），这是一种基于矩的非凸连续重构方法，具有优越的优化景观。我们使用组合多重网格（CMG）对PPR进行热启动，CMG是一种谱图粗化方法，能够在近线性时间内生成结构感知的初始解，且无需任何训练数据。在t2g20（n=400）这一已知精确最优解的基准实例上，CMG+PPR组合管线平均达到最优解的97.5%，最佳运行达到98.4%。带有离散局部搜索改进的扩展变体，在最近一种神经分支定界方法所评估的全部六十个Biq Mac实例上均达到了已发表的最优解。借助稀疏CSR表示，相同方法可扩展到多达20,000个顶点的G-set实例，最多可将内存需求降低……（摘要原文在此处被截断）

    arXiv:2610.19043v1 Announce Type: cross  Abstract: Because its binary optimization landscape contains exponentially many local optima, MAX-CUT is highly sensitive to initialization. We study Particle Polynomial Relaxation (PPR), a moment-based nonconvex continuous reformulation possessing an advantageous optimization landscape. We warm-start PPR with Combinatorial Multigrid (CMG), a spectral graph-coarsening method that produces structure-aware initial solutions in near-linear time without training data. On \texttt{t2g20} ($n=400$), benchmark instance with known exact optimum, the combined CMG+PPR pipeline reaches $97.5\%$ of optimum on average and $98.4\%$ on the best run. An extended variant with discrete local-search refinement reaches the published optimum on all sixty Biq Mac instances evaluated by a recent neural branch-and-bound method. Using a sparse CSR representation, the same approach scales to G-set instances with up to $20{,}000$ vertices reducing memory requirements by up
    
[^173]: 追踪语言习得中的形态、语义与发散

    Tracing Morphology, Semantics, and Divergence in Language Acquisition

    [https://arxiv.org/abs/2610.19041](https://arxiv.org/abs/2610.19041)

    该论文通过对 Pythia 和 OLMo 等开源模型中间训练检查点的系统评估，揭示了语言模型先习得形态结构、语义嵌入随后才趋于稳定的习得轨迹，并发现训练数据规模对发散识别有积极影响。

    

    理解大语言模型（LLM）中语言习得的动态过程，对于揭示统计表示如何演变为结构化语言能力至关重要。然而，现有研究大多聚焦于完全训练好的模型，对中间训练阶段发展轨迹的探索则相对匮乏。在本工作中，我们利用开源模型（如 Pythia 和 OLMo）的中间检查点，在教师强制评估设置下，系统地研究了跨训练步骤与模型规模的语言习得过程。我们重点关注四个核心特征：形态结构习得、嵌入漂移、重复主导性以及共现效应。我们的发现表明，模型在语义嵌入趋于稳定之前就已经较好地习得了形态结构。此外，我们还发现训练数据集规模对发散识别具有积极影响，而共现对……

    arXiv:2610.19041v1 Announce Type: cross  Abstract: Understanding the dynamics of language acquisition in Large Language Models (LLMs) is crucial for uncovering how statistical representations evolve into structured linguistic capabilities. However, most existing studies focus primarily on fully trained models, leaving the developmental trajectory across intermediate training stages largely underexplored. In this work, we leverage intermediate checkpoints from open-source models (e.g., Pythia and OLMo) under teacher-forcing evaluation to systematically investigate language acquisition across training steps and model scales. We focus on four core characteristics: morphological structure acquisition, embedding drift, repetition dominance, and co-occurrence effects. Our findings reveal that models acquire morphological structures well before semantic embeddings stabilize. Furthermore, we show that training dataset size has a positive impact on divergence recognition, while co-occurrence du
    
[^174]: 我们应该更多地倾听谁的声音？面向多元对齐的福利感知偏好获取

    Who Should We Listen to More? Welfare-Aware Preference Acquisition for Pluralistic Alignment

    [https://arxiv.org/abs/2610.19035](https://arxiv.org/abs/2610.19035)

    提出福利感知偏好获取（WAPA）方法，通过基于个体监督效用的社会福祉目标在人群中分配有限的人类反馈，解决多元对齐中因AI模拟精度差异导致的偏好代表性不足问题。

    

    多元对齐需要在有限的人类反馈下容纳多样化的偏好。使用AI模拟器来标注偏好比较可以降低标注成本，但不同个体之间模拟精度的差异可能导致某些人的偏好在训练中代表性不足。我们研究了如何在人群之间分配人类反馈以构建多元对齐的训练数据，以及这种分配如何影响训练得到的策略。为了使这种分配明确且可控，我们定义了一个基于个体监督效用的社会福祉目标，该效用衡量标注质量。每个查询的边际福祉增益取决于其预期的监督改进、接收者的当前效用以及应用指定的优先级。福利感知偏好获取（WAPA）使用辅助参考模型来估计查询增益和个体效用，然后根据估计的边际福祉增益来选择查询……

    arXiv:2610.19035v1 Announce Type: cross  Abstract: Pluralistic alignment needs to accommodate diverse preferences under limited human feedback. Using AI simulators to label preference comparisons reduces annotation costs, but variation in simulation accuracy across individuals can leave some people's preferences poorly represented in training. We study how to allocate human feedback across people to construct training data for pluralistic alignment, and how this allocation affects the trained policy. To make this allocation explicit and controllable, we define a social welfare objective over individual supervision utilities that measure label quality. Each query's marginal welfare gain depends on its expected supervision improvement, the recipient's current utility, and application-designated priority. Welfare-Aware Preference Acquisition (WAPA) uses auxiliary reference models to estimate query gains and individual utilities, then selects queries by estimated marginal welfare gain whil
    
[^175]: 跨越性别二元：科幻小说与一般小说中性别化指称的建模研究

    Crossing the Gender Binary: Modelling Gendered Reference in Science Fiction and General Fiction

    [https://arxiv.org/abs/2610.19028](https://arxiv.org/abs/2610.19028)

    本研究通过遮蔽性别词并利用 RoBERTa 对替代词进行建模，发现科幻小说与一般小说在语言层面均存在双向跨越性别二元的指称现象，但具有不对称性——woman 被遮蔽时更易被阳性词替代——而科幻小说在这一共同模式之上展现出更大的性别灵活性。

    

    科幻小说被广泛认为具有撼动性别观念的力量。本文探讨这种力量是否在语言层面成立，即性别化指称在多大程度上容易让位于中性词或其他社会类别，以及这是否使科幻小说区别于一般小说。研究方法是在同期的科幻小说与一般小说语料库中遮蔽 man/men、woman/women 和 person/people，并按性别对 RoBERTa 预测的五个最可能的替代词进行分类。保留与替代情况经过五轮重采样的置换检验，并对相关句子进行了细读分析。结果显示，两个语料库都以双向方式跨越了性别二元，但存在不平等：当遮蔽 woman 时，模型预测出阳性词的频率约为遮蔽 man 时预测出阴性词频率的两倍。每一次不匹配都代表一个统计学上的意外——一个语料库没有预料到会出现在该位置的词。与这一共同模式相比，科幻小说表现出更大的性别……（原文摘要在此处截断）

    arXiv:2610.19028v1 Announce Type: new  Abstract: Science fiction is widely credited with unsettling gender. This article asks whether that holds at the linguistic level, in how readily gendered reference gives way to its counterpart, a neutral term, or other social category, and whether it separates the genre from general fiction. It does so by masking man/men, woman/women, and person/people across same-period science fiction and general fiction corpora and classifying RoBERTa's five likeliest substitutes by gender. Retention and displacement are permutation-tested over five resampling rounds, and the underlying sentences are read closely. Both corpora cross the gender binary in both directions, but unequally: where woman is masked, the model predicts a masculine term about twice as often as it predicts a feminine term where man is masked. Each mismatch marks a statistical surprise, a word the corpus did not expect there. Against this shared pattern, science fiction shows greater gende
    
[^176]: 面向高效长上下文解码的联合子空间稀疏注意力

    Joint Subspace Sparse Attention for Efficient Long-Context Decoding

    [https://arxiv.org/abs/2610.19026](https://arxiv.org/abs/2610.19026)

    该论文提出联合子空间稀疏注意力（JSSA），通过最小化词元分数保真误差学习共享的低维查询-键子空间并获得闭式谱解以实现高质量词元级稀疏选择，同时设计了无需聚集操作的索引注意力内核，从而显著提升长上下文大语言模型解码的效率。

    

    稀疏注意力能够减少长上下文大语言模型解码中占主导地位的KV缓存流量，但现有的实用方法大多在较粗的粒度上运行。词元级选择提供了更精细的控制，但面临两个挑战：低维评分可能丢失精确排序所需的查询-键结构，而收集分散的被选词元可能会抵消由此带来的延迟收益。我们提出联合子空间稀疏注意力（JSSA）来同时解决这两个挑战。在选择质量方面，JSSA通过最小化词元分数保真误差来学习一个共享的低维查询-键子空间。我们用一个易处理的边际目标函数对该非凸目标进行上界近似，该边际目标具有闭式谱解；在测试设置中，对精确目标进行直接精化并未带来可测量的改进。在执行效率方面，我们开发了一种无需聚集操作的词元级索引注意力内核，支持各KV头不同的选择。（原文摘要至此截断）

    arXiv:2610.19026v1 Announce Type: new  Abstract: Sparse attention reduces the KV-cache traffic that dominates long-context LLM decoding, but practical methods largely operate at coarse granularity. Token-level selection offers finer control, yet presents two challenges: low-dimensional scoring may lose query--key structure needed for accurate ranking, while gathering scattered selected tokens can offset the resulting latency benefit. We introduce Joint Subspace Sparse Attention (JSSA) to address both challenges. For selection quality, JSSA learns a shared low-dimensional query--key subspace by minimizing token-score preservation error. We upper-bound the resulting non-convex objective with a tractable marginal objective that admits a closed-form spectral solution; direct refinement of the exact objective provides no measurable improvement in the tested settings. For execution efficiency, we develop a gather-free token-level indexed-attention kernel that supports distinct per-KV-head se
    
[^177]: 大语言模型能否利用其世界知识进行事件预测？

    Can LLMs Access their World Knowledge for Event Prediction?

    [https://arxiv.org/abs/2610.19021](https://arxiv.org/abs/2610.19021)

    本文提出通过“自然语言事件描述”和“将事件语义与符号化预测标签分离”的接口设计，让 LLM 有效调用其世界知识进行事件预测，并推出包含约 15 万事件、横跨七个数据集和六个领域的 DescribeEvent 基准验证了该方法优于无描述变体和非 LLM 基线。

    

    近期的研究质疑大语言模型（LLM）在时间性事件预测任务上是否真的有用。尽管拥有庞大的预训练世界知识，LLM 的表现往往不如更简单的专用模型。我们探究的问题是：这种知识是否真的无法被访问，还是当前方法未能以 LLM 能够利用的形式来呈现事件预测？我们支持后一种观点，并证明当接口围绕两个设计选择来构建时，LLM 可以成为有效的事件预测器。第一个选择是将每个事件扎根于自然语言描述之中，第二个选择是将事件语义与预测空间分离，以防止标签关联带来的语义偏差。为了验证这一点，我们推出了 DescribeEvent，一个包含约 15 万个事件、横跨七个数据集和六个领域、并带有事件级描述的基准测试。我们发现，能力较强的 LLM 从描述和符号化标签中显著受益，超越了无描述的变体以及非 LLM 基线，而较弱的……（原文摘要在此处截断）

    arXiv:2610.19021v1 Announce Type: new  Abstract: Recent work has questioned whether large language models (LLMs) can be useful for temporal event prediction. Despite vast pretrained world knowledge, LLMs often underperform simpler specialized models. We ask whether this knowledge is truly inaccessible, or whether current methods fail to present event prediction in a form LLMs can use. We argue for the latter and show that LLMs become effective event forecasters when the interface is designed around two choices. The first grounds each event in a natural-language description, and the second separates event semantics from the prediction space to prevent semantic bias from label associations. To test this, we introduce DescribeEvent, a benchmark of ~150K events across seven datasets and six domains with event-level descriptions. We find that capable LLMs substantially benefit from descriptions and symbolic labels, outperforming description-free variants and non-LLM baselines, while weaker 
    
[^178]: 评估大型语言模型的多语言抽象式摘要能力：英语与孟加拉语的全面研究

    Evaluating Large Language Models for Multilingual Abstractive Summarization: A Comprehensive Study of English and Bengali

    [https://arxiv.org/abs/2610.19003](https://arxiv.org/abs/2610.19003)

    该研究构建了一个标准化零样本基准，综合词汇、语义、事实性和LLM评审等多维指标，全面评估了八个大语言模型在英语和孟加拉语摘要任务上的表现，揭示了显著的跨语言性能差异，且不同评估方法会得出不同的模型排名。

    

    抽象式摘要在各语言中仍然具有挑战性，其语言质量与事实一致性可能出现分歧，尤其是在低资源环境中。我们提出了一个标准化的零样本基准，对八个闭源和开放权重的大语言模型在规模匹配的英语和孟加拉语 XL-Sum 测试集上进行评估。我们的评估将词汇和语义指标与无参考事实性验证以及多维度的“LLM作为评审”（LLM-as-a-Judge）评估相结合，对多语言摘要的表层质量和事实可靠性进行了全面分析。结果显示出显著的跨语言差异：GPT-5.6 Luna 在多项英语词汇指标上领先，但在孟加拉语中并未保持同样的优势，在孟加拉语中 LLaMA 3.3 70B 取得了最高的 ROUGE 和 BLEU 分数。Gemini 2.5 Flash 获得了所观察到的最高孟加拉语事实性分数。此外，不同的评估方法会产生不同的模型排名，凸显了……

    arXiv:2610.19003v1 Announce Type: new  Abstract: Abstractive summarization remains challenging across languages, where linguistic quality and factual consistency may diverge, particularly in low-resource settings. We present a standardized zero-shot benchmark of eight closed-source and open-weight LLMs on size-matched English and Bengali XL-Sum test sets. Our evaluation combines lexical and semantic metrics with reference-free factuality verification and multidimensional LLM-as-a-Judge assessment, providing a comprehensive analysis of both surface-level quality and factual reliability in multilingual summarization. Results reveal substantial cross-lingual variation: GPT-5.6 Luna leads several English lexical metrics but does not maintain the same advantage in Bengali, where LLaMA 3.3 70B achieves the highest ROUGE and BLEU scores. Gemini 2.5 Flash obtains the highest observed Bengali factuality score. Furthermore, evaluation methods produce different model rankings, highlighting the li
    
[^179]: 智能体科学家：编排编程智能体以实现自动化科学（协同）发现

    The Agentic Scientist: Orchestrating Coding Agents for Automated Scientific (Co-)Discovery

    [https://arxiv.org/abs/2610.18933](https://arxiv.org/abs/2610.18933)

    提出了端到端分布式系统“智能体科学家”，将计算研究分解为多个阶段并由新一代编程智能体驱动、结合智能体自验证，实现人机协同的自动化科学发现。

    

    现代编程智能体正在从根本上重塑人类输入在计算研究中的角色：它们能够编写代码、启动计算机模拟实验、分析结果，并以不断延长的时间跨度自主运行。尽管有这些进展，完全自主的研究智能体仍存在构思半径有限、以及目光短浅的爬山式搜索动态等局限。为弥合这一差距，我们提出了“智能体科学家”，这是一个端到端的分布式流水线，使人类研究者能够与现代编程智能体协同开展计算研究。在其前作系统的基础上，我们的系统将研究过程分解为任务起草、想法生成、规划、沙盒化实验、结果汇总、基线与消融实验构建、事后分析以及报告撰写等环节。重要的是，流水线中的每一步都由下一代编程智能体驱动，并利用智能体自验证机制。整体研究过程由（原文在此处截断）……

    arXiv:2610.18933v1 Announce Type: cross  Abstract: Modern coding agents are fundamentally reshaping the role of human input for computational research: they write code, launch in silico experiments, analyze their results, and autonomously operate with ever-increasing time horizons. Despite these developments, fully autonomous research agents suffer from a limited ideation radius and myopic hill-climbing search dynamics. To bridge this gap, we introduce The Agentic Scientist, an end-to-end distributed pipeline for human researchers to conduct computational research alongside modern coding agents. Building on its predecessor, our system decomposes research into task drafting, idea generation, planning, sandboxed experimentation, result aggregation, baseline and ablation construction, post-hoc analysis, and report writing. Importantly, each step in the pipeline is powered by the next generation of coding agents and leverages agentic self-verification. The overall research process is drive
    
[^180]: 遵循法律规定的流程：训练语音代理迁移至未见过的受监管债务催收程序

    Following the Flow the Law Requires: Training Voice Agents to Transfer to Unseen Regulated Procedures in Debt Collection

    [https://arxiv.org/abs/2610.18881](https://arxiv.org/abs/2610.18881)

    本文提出 FlowSpec 机器可读规范，通过在 297 个随机化监管程序上微调 90 亿参数模型，训练出能够零样本遵循从未见过的受监管债务催收流程的语音代理。

    

    基于大语言模型（LLM）构建的语音代理如今可以处理客户来电，但在受监管的应用场景中，仅仅完成任务是不够的：代理必须遵循由法律设定的对话流程。在债务催收中，代理必须在提及债务之前先确认自己正在与债务人通话，表明自身和债权人的身份，说明欠款金额，并避免使用被禁止的语言。每个债权人都会将这些规则转化为自己的流程程序，并不断进行修订。本文探究的核心问题是：能否只训练一次一个小型、自托管的策略模型，然后让它遵循从未见过的受监管程序。每个部署使用一份机器可读的规范 FlowSpec，由它生成策略指令、控制工具调用、提供监督信号并定义测试；代理只能说出经过批准的目录句子。研究团队在来自 297 个随机化程序的 2,506 段对话上微调了一个 90 亿参数的模型，并在四个未见过的部署程序上评估了一个泰语外呼代理（原摘要在此处被截断）。

    arXiv:2610.18881v1 Announce Type: new  Abstract: Voice agents built on large language models (LLMs) now handle customer calls, but in regulated applications completing the task is not enough: the agent must follow a conversational flow set by law. In debt collection, it must confirm it is speaking to the debtor before mentioning the debt, identify itself and the creditor, state the amount, and avoid prohibited language. Every creditor turns these rules into its own procedure and revises it. We ask whether a small, self-hosted policy can be trained once and then follow regulated procedures it has never seen. One machine-readable specification per deployment, FlowSpec, generates the policy instruction, gates the tools, supplies the supervision and defines the tests; the agent speaks only approved catalogue sentences. We fine-tune a 9-billion-parameter model on 2,506 conversations from 297 randomized procedures and evaluate a Thai outbound-call agent on four unseen deployment procedures (
    
[^181]: Lotus：通过学习预测来学会理解时间序列

    Lotus: Learning to Understand Time-series by Learning to Forecast

    [https://arxiv.org/abs/2610.18877](https://arxiv.org/abs/2610.18877)

    LOTUS 提出“预测锚定”预训练原则，通过要求时间序列的语言对齐表示能够预测未来片段演化，使其在映射到 LLM 嵌入空间时保留时间与频谱信息，从而学会真正理解时间序列。

    

    时间序列语言模型通常使用标签、描述或问答对来将数值信号与语言对齐。尽管这种监督方式有助于形成语言表示，但它并未显式要求这些表示保留底层时间序列信号如何演化的信息。我们提出了预测锚定，这是一种预训练原则，要求与语言对齐的时间序列表示在适应下游任务之前，必须保持对未来演化趋势的预测能力。我们在 LOTUS 中实现了这一原则：该方法将每个时间序列通道划分为不重叠的片段，并将其映射到 LLM 的嵌入空间中；随后，它从时间序列词元的 LLM 隐藏状态预测下一个片段，促使学习到的表示保留时间信息。为了同时捕捉时间变化和周期性模式，我们还提出了一种时间-频谱片段表示，将每个片段的时域与频域特征相结合（原文此处截断）。

    arXiv:2610.18877v1 Announce Type: cross  Abstract: Time-series language models typically align numerical signals with language using labels, captions, or question-answer pairs. Although such supervision encourages linguistic representations, it does not explicitly require them to preserve information about how the underlying time-series signal evolves. We introduce forecast grounding, a pretraining principle that requires language-aligned time-series representations to remain predictive of the future evolution before they are adapted to downstream tasks. We instantiate this principle in LOTUS, which divides each time-series channel into non-overlapping patches and maps them into the LLM embedding space. It then predicts the next patch from the LLM hidden states of the time-series tokens, encouraging the learned representation to retain temporal information. To capture both temporal changes and periodic patterns, we also propose a temporal-spectral patch representation that combines eac
    
[^182]: 让模型在生成前先“察言观色”

    Read the Room Before Your Model Generates

    [https://arxiv.org/abs/2610.18820](https://arxiv.org/abs/2610.18820)

    提出 Scenario-Actor 框架，从 Transformer 隐藏状态中提取交际场景的结构化几何表征（场景嵌入），据此自适应地选择解码温度，从而在交互社交场景和开放式指令遵循任务上显著提升 LLM 的表现。

    

    大型语言模型通常依赖采样温度来调节输出的随机性程度。主流的温度选择方法通常采用固定的、启发式的全局设置，以及上下文或词元级别的校准，而不是对交际场景进行几何上可解释的表征。在本工作中，我们提出了一个简单且可解释的温度选择框架——Scenario-Actor（场景-行动者），它能够根据推断出的情境特征来调整解码策略。Scenario-Actor 从 Transformer 隐藏状态中提取与场景相关的结构，并通过场景嵌入——一种对交际环境进行结构化几何表征的形式——来确定温度。实证结果表明，在 Llama-3.1-8B-Instruct 上，Scenario-Actor 在交互式社交场景中实现了出色的目标完成性能，在开放式指令遵循任务上取得了显著提升，并展现出具有竞争力的表现。

    arXiv:2610.18820v2 Announce Type: new  Abstract: Large language models (LLMs) generally rely on sampling temperature to regulate the amount of randomness in their outputs. Mainstream methods for selecting temperature typically involve fixed, heuristic global settings and context or token-level calibration, rather than a geometrically interpretable representation of the communicative scenario. In this work, we introduce a simple and interpretable temperature selection framework, the Scenario-Actor, which adapts decoding to the inferred situational characteristics. Scenario-Actor extracts scenario-relevant structures from Transformer hidden states and determines the temperature from scenario embedding, a structured geometric representation of the communicative setting. Empirical results show that on Llama-3.1-8B-Instruct, Scenario-Actor achieves strong goal-completion performance in interactive social scenarios, significant gains on open-ended instruction following, and competitive perfo
    
[^183]: 回译依然重要：利用单语数据实现基于大语言模型的低资源机器翻译

    Back-Translation Still Matters: Leveraging Monolingual Data for Low-Resource Machine Translation with LLMs

    [https://arxiv.org/abs/2610.18818](https://arxiv.org/abs/2610.18818)

    对于低资源机器翻译，利用单语数据做回译比继续预训练大语言模型更有效，仅需1K平行句对即可超越继续预训练7.7至16.4个XCOMET-XXL分，甚至优于拥有10倍平行数据的模型。

    

    预训练大语言模型（LLM）在零样本和少样本设置下对许多任务都很有效。对于低资源语言的机器翻译，显而易见的选择是少样本提示和在小型平行语料上进行微调。我们提出一个问题：如果同时还有适量规模的低资源单语数据可用，情况会怎样？我们认为，比起继续预训练，更有效地利用这些数据的方法是大语言模型时代之前的技术——回译（BT）。具体做法是：让当前微调后的大语言模型对每种语言的单语句子进行翻译，将得到的句对用于反向方向的训练，并与平行语料上的更新交错进行以保证学习的稳定性。在Odia语和斯瓦希里语上仅使用1K平行句对的情况下，回译在域内测试中比向低资源语言继续预训练高出7.7到16.4个XCOMET-XXL分数。使用束搜索解码的中间翻译结果时，它还击败了拥有10倍平行数据量的模型。

    arXiv:2610.18818v2 Announce Type: cross  Abstract: Pre-trained Large Language Model (LLMs) are effective on many tasks in zero-shot and few-shot settings. For machine translation of low-resource languages, the obvious options are few-shot prompting and fine-tuning on a small parallel set. We ask: what if moderate amounts of low-resource monolingual data are also available? We argue that, rather than continued pretraining, a more effective use of this data is the pre-LLM era technique of back-translation (BT). The current fine-tuned LLM translates monolingual sentences in each language, and the resulting pairs train it in the reverse direction. Updates on the parallel set are interleaved for stable learning. With 1K parallel pairs on Odia and Swahili, BT beats continued pretraining into the low-resource language by 7.7 to 16.4 XCOMET-XXL points in domain. With beam-decoded intermediate translations, it also beats a model with 10 time more parallel data. In-context examples retrieved fro
    
[^184]: 轻装上阵，CarryOn：为长时程智能体构建面向对象的上下文

    Pack Light, CarryOn: Building Object-Oriented Context for Long-Horizon Agents

    [https://arxiv.org/abs/2610.18784](https://arxiv.org/abs/2610.18784)

    提出 CarryOn，一个面向对象的上下文运行时，通过围绕可复用的任务对象（而非交互历史）来组织长时程智能体的上下文，使其能够在折叠过程中高效更新续接载体记录而无需重建完整表示。

    

    长时程智能体在持久化任务状态中积累进展，却需要反复处理描述这些进展如何实现的交互历史。这引出一个自然的问题：长时程上下文能否围绕可复用的进展而非产生它的历史来组织？在本文中，我们提出 CarryOn，一个面向对象的上下文运行时，它围绕可复用的任务对象来组织续接过程。CarryOn 不将压缩历史或重建的进展叙述作为续接的主要依据，而是从续接载体、互补执行上下文以及近期交互的实时尾部构建上下文视图。一个核心挑战是在周边历史被淘汰后保持这一可复用基础的有效性：CarryOn 如何在每次折叠时无需重建完整表示就能更新该基础？CarryOn 在多次折叠之间维护载体记录，并对受影响的记录进行修订……（摘要在此处截断）

    arXiv:2610.18784v2 Announce Type: new  Abstract: Long-horizon agents accumulate progress in persistent task state, yet repeatedly process interaction histories describing how that progress was achieved. This raises a natural question: can long-horizon context instead be organized around reusable progress rather than the history that produced it? In this paper, we introduce CarryOn, an object-oriented context runtime that organizes continuation around reusable task objects. Rather than treating compressed histories or reconstructed accounts of progress as the primary basis for continuation, CarryOn builds a context view from continuation carriers, complementary execution context, and a live tail of recent interactions. A core challenge is keeping this reusable basis valid after its surrounding history is retired: how can CarryOn update the basis without reconstructing the full representation at every fold? CarryOn maintains carrier records across folds, revising affected records against
    
[^185]: 预训练能教会化学语言模型化学吗？

    Does Pretraining Teach Chemical Language Models Chemistry?

    [https://arxiv.org/abs/2610.18651](https://arxiv.org/abs/2610.18651)

    化学语言模型的预训练收益无法随规模扩大，性能低于梯度提升基线，仅有基于理化描述符的有监督预训练能持续带来提升，表明预训练几乎未能让模型真正学到化学知识。

    

    化学语言模型（CLM）在大规模分子语料库上进行预训练，旨在学习可迁移到多种下游任务的表示。本研究使用一系列此类化学语言模型，在64个分子性质预测数据集上系统地研究了预训练的效果。我们发现，化学语言模型预训练带来的下游性能提升并不随规模扩大而增长，且性能在低于梯度提升基线的水平上趋于饱和。在所评估的各种预训练模式中，只有在传统理化描述符上进行有监督预训练才能持续提升所测试化学语言模型的下游性能，而某些形式的预训练甚至会导致模型表现不如随机初始化的编码器。尽管模型能够较好地解析分子结构，但其表现并不优于随机初始化的模型，这表明化学语言模型在预训练中充其量只学到了类似描述符的性质和低级token统计信息，而除此之外学到的化学知识出奇地少。

    arXiv:2610.18651v2 Announce Type: replace-cross  Abstract: Chemical language models (CLMs) are pretrained on large corpora of molecules with the aim of learning representations that transfer to diverse downstream tasks. Here, we systematically study the effects of pretraining using a series of such CLMs across 64 molecular property prediction datasets. We find that the downstream gains of pretraining in CLMs do not scale and performance saturates below gradient-boosting baselines. Across the regimes evaluated, only supervised pretraining on traditional physico-chemical descriptors consistently improves downstream performance for the CLMs tested whereas some forms of pretraining result in models worse than a randomly initialized encoder. While models can resolve molecular structure well, they do so no better than a randomly initialized model, suggesting that, at best, CLMs learn descriptor-like properties in addition to low-level token statistics during pretraining, but surprisingly lit
    
[^186]: 读出缩放塑造稀疏特征恢复

    Readout Scaling Shapes Sparse Feature Recovery

    [https://arxiv.org/abs/2610.18462](https://arxiv.org/abs/2610.18462)

    本文提出读出规范化决策空间（RC-DS）坐标方法以消除读出缩放的奇异值增益，并通过受控实验证明当训练恢复因子生成的原始空间时稀疏特征恢复效果最佳，将平均稀疏因子恢复从0.506提升至0.913。

    

    读出缩放会改变稀疏自编码器（SAE）所学到的内容，但移除它的益处取决于数据。我们使用读出规范化决策空间（RC-DS）坐标来移除奇异值增益，同时保留已保留的决策信息。在固定的读出条件数下，改变生成分布会逆转哪些训练坐标最能恢复已知因子。在我们的受控坐标匹配研究中，当训练恢复了因子生成时所在的空间，恢复效果最好。当因子在读出缩放之前生成时，在条件数为100的情况下，与相同预算下的中心化logit训练相比，RC-DS将平均稀疏因子恢复从0.506提高到0.913。一个精确的包装器以数值精度重现了RC-DS训练，从而分离出训练几何结构而非额外信息的作用。在神经网络策略中，RC-DS在测试预算下改善了平均logit重建。

    arXiv:2610.18462v2 Announce Type: cross  Abstract: Readout scaling changes what a sparse autoencoder (SAE) learns, but the benefit of removing it depends on the data. We use readout-canonicalized decision-space (RC-DS) coordinates to remove singular-value gains while preserving retained decision information. At a fixed readout condition number, changing the generating distribution reverses which training coordinates recover known factors best. In our controlled coordinate-matching study, recovery is highest when training restores the space in which the factors were generated. When factors are generated before readout scaling, RC-DS increases mean sparse-factor recovery from 0.506 to 0.913 at condition number 100, compared with centered-logit training under the same budget. An exact wrapper reproduces RC-DS training to numerical precision, isolating training geometry rather than additional information. In neural policies, RC-DS improves mean logit reconstruction under the tested budgets
    
[^187]: Fish Audio Drama 3 技术报告

    Fish Audio Drama 3 Technical Report

    [https://arxiv.org/abs/2610.18010](https://arxiv.org/abs/2610.18010)

    Fish Audio Drama 3 是一个基于 35B 参数混合专家架构的开源语音生成模型，能将文本转化为有导演指导的表演，并首次在单一模型中集成了多轮修改、场景配音、填充编辑和流式输入等多项新能力。

    

    我们推出 Fish Audio Drama 3，这是一个开源语音生成模型，能够将文本转化为有导演指导的表演。该模型基于 350 亿参数的混合专家架构骨干网络，每个 token 激活 30 亿参数。它可以从多个参考样本克隆声音，遵循自然语言导演指令，在下一轮收到反馈意见时修改录音片段，为多角色场景配音，通过填充方式编辑现有语音，并合成长篇和流式音频。多轮修改、场景配音、填充编辑和流式文本输入均是 Drama 3 的新功能，且全部集成于同一个模型中。Drama 3 将混合线性注意力骨干网络与一种新的码本多 token 预测音频解码器相结合，该解码器通过专门关注整个序列的层来预测每一帧的声学码本。该模型采用三阶段预训练方案、指令微调以及大批量强化学习进行训练，其唯一的奖励是单个……

    arXiv:2610.18010v2 Announce Type: replace-cross  Abstract: We introduce Fish Audio Drama 3, an open-source speech generation model that turns text into directed performances, built on a 35B-parameter mixture-of-experts backbone that activates 3B parameters per token. It clones voices from multiple references, follows natural-language direction, revises a take when the next turn gives a note, voices multi-character scenes, edits existing speech by infilling, and synthesizes long-form and streaming audio. Multi-turn revision, scene dubbing, editing by infilling, and streaming text input are new in Drama 3, all in one model. Drama 3 pairs the hybrid linear-attention backbone with a new codebook multi-token-prediction audio decoder that predicts the acoustic codebooks of each frame in dedicated layers that attend over the entire sequence. The model is trained with a three-stage pre-training recipe, instruction fine-tuning, and large-batch reinforcement learning whose only reward is a singl
    
[^188]: SkillScreen：基于决策模型的恶意智能体技能高召回率筛查

    SkillScreen: High-Recall Screening of Malicious Agent Skills with Decision Models

    [https://arxiv.org/abs/2610.17727](https://arxiv.org/abs/2610.17727)

    SkillScreen 是一个紧凑的决策模型，能在单次前向传播中直接对 SKILL.md 技能的恶意程度评分，在未见攻击族上达到 99% 召回率且误报远低于词级分类器和 LLM 判官，但对陌生良性来源的误报仍会显著上升。

    

    智能体技能将自然语言指令转化为具体操作，这带来了一项安全挑战：合法的自动化行为与恶意行为可能看起来惊人地相似。一个实用的筛查系统必须在几乎捕获每一次攻击的同时，保持审查负载可控。我们提出了 SkillScreen，这是一个紧凑的决策模型，经过训练可以直接从 SKILL.md 文件在单次前向传播中对技能的恶意程度进行评分。在 MaliciousSkillBench 基准上，SkillScreen 在对未见攻击族达到 99% 召回率的情况下，仅将 20-25% 的良性技能标记为可疑，明显少于所测试的词级分类器和提示式 LLM 判官。在相同的训练协议下，它对未见来源的泛化能力也优于全量微调的编码器。然而，陌生的良性来源会显著增加误报率，这表明能够识别新攻击并不等于能够可靠地筛查新的技能集合。我们的结果证明了针对特定任务训练的决策模型在高召回率技能筛查中的价值。

    arXiv:2610.17727v2 Announce Type: cross  Abstract: Agent skills turn natural-language instructions into actions, creating a security challenge: legitimate automation and malicious behaviour can look deceptively similar. A practical screen must catch nearly every attack while keeping the review load manageable. We present SkillScreen, a compact decision model trained to score skill maliciousness directly from SKILL$.$md in a single forward pass. On MaliciousSkillBench, SkillScreen flags 20-25% of benign skills at 99% recall on unseen attack families, substantially fewer than the word-level classifier and prompted LLM judges tested. Under the same training protocol, it also generalizes better than a fully fine-tuned encoder to unseen sources. Yet unfamiliar benign sources sharply increase false positives, showing that recognizing new attacks does not ensure reliable screening of new skill collections. Our results demonstrate the value of task-trained decision models for high-recall skill
    
[^189]: RELAY：面向基于评分标准强化学习的自适应交接式选择性自蒸馏方法

    RELAY: Selective Self-Distillation with Adaptive Hand-off for Rubric-Based Reinforcement Learning

    [https://arxiv.org/abs/2610.17623](https://arxiv.org/abs/2610.17623)

    RELAY提出一种两阶段训练方法，第一阶段将冻结教师模型的选择性自蒸馏与GRPO结合，并通过监控蒸馏损失的停滞检测器自适应地在最佳时机移除教师模型、切换到纯GRPO训练，从而在基于评分标准的强化学习中超越纯GRPO方法，且停滞检测器跨模型和领域具有良好泛化性。

    

    基于评分标准的强化学习根据多条书面评分标准对回答进行评分，但会将这些标准压缩为单一奖励，这不仅丢弃了评分标准所承载的细节信息，而且当一组回答打平时无法提供GRPO奖励梯度。一种将评分标准作为特权上下文读取的冻结教师模型可以提供token级别的监督，但我们观察到其相对GRPO的增益会出现早期平台期。RELAY利用优势值和奖励来选择正优势以及低分数零优势的rollout，并监控蒸馏损失以决定何时让教师模型退场。第一阶段将冻结教师模型的选择性蒸馏与GRPO相结合；随后停滞检测器移除教师模型，第二阶段仅使用GRPO继续训练。该方法在一个模型-领域对上进行校准后，无需修改即可在另外三个模型-领域对上复用，停滞检测器在第122至171步之间触发。在所有四个设置中，RELAY在所比较的方法中于域内取得最高分数，领先于GRPO在RaR-Medici上的表现

    arXiv:2610.17623v2 Announce Type: cross  Abstract: Rubric-based reinforcement learning grades a response against many written criteria but collapses them into one reward, which discards the detail the rubric carries and gives no GRPO reward gradient when a group's responses tie. A frozen teacher that reads the rubric as privileged context provides token-level supervision, but we observe an early plateau in the gains over GRPO. RELAY uses advantages and rewards to select positive-advantage and low-scoring zero-advantage rollouts, and monitors the distillation loss to decide when to retire the teacher. Stage 1 combines selective distillation from a frozen teacher with GRPO; a plateau detector then removes the teacher, and Stage 2 continues with GRPO alone. Calibrated on one model-domain pair and reused unchanged in three others, the plateau detector fires between steps 122 and 171. RELAY scores highest in-domain among the compared methods in all four settings, ahead of GRPO on RaR-Medici
    
[^190]: ANCHOR：基于仓库的智能体编码中自然约束层次与排序的评估

    ANCHOR: Assessing Naturalistic Constraint Hierarchies and Ordering in Repository-Grounded Agentic Coding

    [https://arxiv.org/abs/2610.16859](https://arxiv.org/abs/2610.16859)

    提出 ANCHOR 基准，通过层次化约束分类体系、自动化约束组合流水线和基于轨迹的评估框架，首次系统性地评估基于仓库的编码智能体遵循工程约束（如修改范围、工作流顺序、验证等）的能力，弥补了现有评估仅关注功能正确性的不足。

    

    对于基于仓库的编码智能体而言，功能正确的补丁未必是可接受的：开发者还会约束任务执行的方式——修改范围、工作流顺序、验证、工具使用和报告。我们将这种能力称为“工程约束遵循”，但目前的评估方法在很大程度上忽视了这一能力。我们提出了 ANCHOR（Assessing Naturalistic Constraint Hierarchies and Ordering in Repository-Grounded Agentic Coding），一个建立在自然、多语言仓库任务之上的基准，涵盖八个语言组。ANCHOR 结合了：(i) 从真实开发者交互中提炼的包含 10 个类别和 78 个叶子类型的层次化分类体系；(ii) 一个自动化流水线，用于实例化基于任务的原子约束并将其组合为带类型的 AND/THEN/PRIORITY 关系；(iii) 一个基于轨迹的评估框架，可同时检查代码差异和执行轨迹。

    arXiv:2610.16859v2 Announce Type: replace  Abstract: For repository-grounded coding agents, a functionally correct patch is not necessarily an acceptable one: developers also constrain how the task must be carried out-modification scope, workflow order, validation, tool use, and reporting. This capability, which we call engineering constraint following, remains largely invisible to current evaluations. We introduce ANCHOR (Assessing Naturalistic Constraint Hierarchies and Ordering in Repository-Grounded Agentic Coding), a benchmark built on naturalistic, multilingual repository tasks spanning eight language groups. ANCHOR combines (i) a hierarchical taxonomy of 10 categories and 78 leaf types distilled from real developer interactions, (ii) an automated pipeline that instantiates task-grounded atomic constraints and composes them into typed AND/THEN/PRIORITY relations, and (iii) a trajectory-grounded evaluation framework that checks both the code diff and the execution trajectory with 
    
[^191]: REVEAL：从前沿模型轨迹中逆向工程智能体潜变量

    REVEAL: Reverse Engineering Agentic Latents from Frontier-Model Traces

    [https://arxiv.org/abs/2610.16288](https://arxiv.org/abs/2610.16288)

    提出免训练框架REVEAL，通过逆向工程从前沿模型轨迹中提取“智能体潜变量”，为冻结的小型开源智能体提供可复用的局部引导控制，无需训练目标模型或访问源模型内部。

    

    前沿大语言模型智能体能够执行复杂的交互任务，但其高昂的推理成本促使人们将它们的经验迁移到更小的开源智能体中。蒸馏和潜变量通信方法通常需要对目标模型进行训练，或需要访问源模型的内部结构。我们研究了文本形式的前沿模型轨迹如何能为冻结的智能体提供可复用的局部控制。受协议逆向工程的启发，我们提出了“智能体潜变量”，它由锚定状态以及通过在目标模型中重放而引出的反馈-响应循环的动态特征组成。我们的免训练框架REVEAL（REVerse Engineering Agentic Latents，逆向工程智能体潜变量）利用这些特征和轨迹结果，从同一任务和同一源模型的成功与失败回放中匹配并筛选局部事件。这些局部事件的响应转移对比产生了针对目标模型的引导方向，并应用于每次响应之前。

    arXiv:2610.16288v2 Announce Type: new  Abstract: Frontier LLM agents perform complex interactive tasks, but their inference costs motivate transferring their experience to smaller open-source agents. Distillation and latent communication typically require target-model training or access to source-model internals. We study how textual frontier-model trajectories can provide reusable local controls for a frozen agent. Inspired by protocol reverse engineering, we introduce $\textit{Agentic Latents}$, comprising anchored states and dynamic profiles of feedback--response cycles elicited by replay in the target model. Our training-free framework, $\textbf{REVEAL}$ ($\textbf{REV}$erse $\textbf{E}$ngineering $\textbf{A}$gentic $\textbf{L}$atents), uses these profiles and trajectory outcomes to match and select local events from successful and failed rollouts of the same task and source model. Their response-transition contrasts yield a target-specific steering direction, applied before each re
    
[^192]: 评审很少胜过更强的模型：在大语言模型流水线中区分验证与能力迁移

    Review Rarely Beats the Stronger Model: Separating Verification from Capability Transfer in LLM Pipelines

    [https://arxiv.org/abs/2610.15897](https://arxiv.org/abs/2610.15897)

    论文揭示了 LLM 评审流水线中“验证”与“能力迁移”的本质区别——评审很少超越更强的单模型、因锚定效应丢弃自身正确答案且成本更高，而简单的答案选择器是更准确又便宜的替代方案。

    

    在数学任务中，Qwen3-8B 对 Llama-3.1-8B 的答案进行评审可带来 33.2 分的提升，但 Qwen3-8B 单独作答的准确率反而比这一结果高出 6.4 分。通常人们以评审相对于生成器的增益来评判其价值，我们指出这种评判方式混淆了验证（即带来超越任一单模型单独表现的准确率提升）与能力迁移（即更强的评审者只是提供了它本来就能自己给出的答案）。以更强模型单独表现为基准，在 12 个开源权重模型（3B 至 72B）的 48 组配对、涵盖数学、知识、代码和推理任务的实验中，96 个评审方向里仅有 9 个方向上评审胜过更强模型，且能力差距越大表现越差。其根源在于锚定效应：评审者大多跟随生成器的答案，丢弃了自己原本正确的答案。评审的成本是单独使用更强模型的 2.6 至 2.8 倍，而使用同样两次调用的选择器不仅准确率比评审高 2.7 至 4.2 分，而且成本更低。

    arXiv:2610.15897v2 Announce Type: cross  Abstract: Qwen3-8B reviewing Llama-3.1-8B in math adds 33.2 points, yet Qwen3-8B alone is 6.4 points more accurate than the result. Reviewing another model's answer is usually judged by its gain over the generator. We show that this confuses verification, which adds accuracy beyond either model alone, with capability transfer, where a stronger reviewer supplies answers it could have given on its own. Measured against the stronger model alone, across 48 pairs of 12 open-weight models (3B to 72B) in math, knowledge, code and reasoning, review beats it in only 9 of 96 review directions, and does worse the larger the capability gap. The reason is anchoring: the reviewer mostly follows the generator, discarding its own correct answers. Review costs 2.6 to 2.8 times as much as the stronger model alone, while a selector using the same two calls is 2.7 to 4.2 points more accurate than review and cheaper. Studies of verification should report the stronge
    
[^193]: 谁决定了“评委攻击”的节奏？评委能力与策略-评委差距之争

    Who Sets the Pace of Judge Hacking? Judge Capability versus the Policy-Judge Gap

    [https://arxiv.org/abs/2610.15778](https://arxiv.org/abs/2610.15778)

    该研究发现强化学习“攻击评委”的深度由评委自身能力决定（与评委AUC呈负相关，系列内更大的评委更难被欺骗），而攻击速度反而与“策略-评委能力差距”的解释相矛盾——差距更大的组合被欺骗得更早，且所有组合均未带来真实的保留集准确率提升。

    

    在使用GRPO针对一个冻结的、无参考的LLM（大语言模型）评委进行训练时，强化学习会利用评委的假阳性判断。那么，究竟是策略与评委之间的能力差距，还是评委本身，决定了这种“攻击评委”行为的速度与程度？我们在GSM8K数据集上训练了0.5B和1.5B的Qwen2.5策略，分别对抗四个规模不小于它们的评委，此外1.5B策略还对抗了一个规模更小的评委。事后分析显示，评委的接受率确实上升了，但在12个实验组合中，没有任何一个组合的种子平均保留集贪心准确率提升超过0.04。使用相同配方但以金标准作为奖励的对照组使1.5B策略的准确率提高了+0.054（而对于0.5B策略，金标准奖励同样没有带来可检测的提升）。评委被攻击的深度取决于评委本身：攻击的停滞期与评委的AUC相关（在12个组合上的Spearman相关系数为-0.69），并且在同一系列模型中，更大的评委被欺骗得更浅。而在攻击速度方面，两个预先设定的配对实验与“能力差距”的解释相矛盾：在预设阈值下，差距更大的那个组合反而提前了3.5倍被欺骗。（原文在此处截断）

    arXiv:2610.15778v2 Announce Type: replace-cross  Abstract: Under GRPO against a frozen, reference-free LLM judge, RL exploits the judge's false positives. Does the policy-judge capability gap or the judge itself set how fast and how far? We train 0.5B and 1.5B Qwen2.5 policies on GSM8K against four judges at least as large, and the 1.5B policy also against one smaller judge. In a post-hoc analysis, judge acceptance rose, but no cell of 12 gained more than 0.04 in seed-mean held-out greedy accuracy. A gold-reward control with the same recipe raised the 1.5B policy's accuracy by +0.054 (for the 0.5B policy, gold reward gave no detectable gain either). The depth of judge hacking tracks the judge: the plateau correlates with judge AUC (Spearman -0.69 over 12 cells), and within a family the larger judge is fooled less deeply. For the pace, two pre-specified pairs contradict the gap account: the cell with the larger gap was fooled 3.5x sooner at the pre-specified threshold. With 12 cells, ho
    
[^194]: 大语言模型导师真的在教书吗？以过程为中心的辅导循环综述

    Do LLM Tutors Actually Teach? A Process-Centered Synthesis of the Tutoring Loop

    [https://arxiv.org/abs/2610.15692](https://arxiv.org/abs/2610.15692)

    本文通过对大语言模型辅导流程的以过程为中心的系统综述发现，尽管教学评估较普遍，但证明真实学习收益、尤其是知识迁移与延迟保持的证据薄弱，且模型辅导效果依赖于教学结构、学习者参与度和学习衡量方式。

    

    大语言模型能够生成流畅的辅导对话，但它们是否真正帮助学生学习仍不明确。我们提出了一种以过程为中心的大语言模型辅导综述，沿着从学习者状态推断到教学行动、导师训练、评估以及学生模拟的路径展开。这一视角揭示了一个核心缺口：实现某个辅导组件并不能证明它有效，也不能证明其输出会对后续阶段产生实质性影响。一项结构化的研究层面审计表明，教学评估较为常见，而在学习者条件化干预、人类学习收益，尤其是知识迁移或延迟保持方面的证据则明显薄弱。然而，人类研究提供了一个条件性的积极结果：大语言模型导师可以改善学习，但其收益取决于教学结构、学习者参与度以及学习效果的衡量方式。我们认为，要取得进展，需要验证各环节之间的接口。

    arXiv:2610.15692v2 Announce Type: replace  Abstract: Large language models can produce fluent tu- toring dialogue, but whether they actually help students learn remains unclear. We present a process-centered synthesis of LLM tutoring that follows the path from learner-state infer- ence to pedagogical action, tutor training, eval- uation, and student simulation. This view ex- poses a central gap: implementing a tutoring component does not establish that it works or that its outputs meaningfully affect later stages. A structured study-level audit shows that peda- gogical evaluation is common, while evidence becomes much thinner for learner-conditioned intervention, human learning gains, and espe- cially transfer or delayed retention. Human studies nevertheless provide a conditional posi- tive result: LLM tutors can improve learning, but benefits depend on pedagogical structure, learner engagement, and how learning is mea- sured. We argue that progress requires validat- ing the interfaces
    
[^195]: 藏于明处：通过感知感知式对抗自博弈（PASP）提升有害内容理解能力

    Hidden in Plain Sight: Advancing Harmful Content Understanding via Perception-Aware Adversarial Self-Play

    [https://arxiv.org/abs/2610.15449](https://arxiv.org/abs/2610.15449)

    提出PASP框架，通过融合文本、语音、视觉等多感知视角与对抗性自博弈机制，弥合人类读者与LLM之间的认知不对称，从而提升模型对不断演化的伪装性有害内容意图的理解能力。

    

    尽管在语言理解方面取得了显著进展，大型语言模型（LLM）仍可能忽略“藏于明处”的含义，无法识别人类轻易就能理解的表达中所蕴含的有害意图。恶意用户利用这一缺口，策略性地伪装有害内容以规避自动化内容审核，对在线平台安全构成严重威胁。这种脆弱性反映了人类读者与基于文本的LLM之间的根本性认知不对称：人类读者能够综合上下文、语音和视觉等多种线索来破解刻意混淆背后的意图，而基于文本的LLM主要依赖词元序列，且对不断演化的规避手段接触有限。为了弥合这种不对称，我们提出了PASP（感知感知式对抗自博弈），该方法将多感知视角与对抗性自博弈相结合，在规避手段不断演化的情况下提升对有害意图的理解。具体而言，PASP构建了……（摘要在此处截断）

    arXiv:2610.15449v2 Announce Type: new  Abstract: Despite significant advances in language understanding, Large Language Models (LLMs) can still miss meaning hidden in plain sight, failing to identify harmful intent in expressions readily understood by humans. Malicious users exploit this gap by strategically disguising harmful content to evade automated moderation, posing a serious threat to online platform safety. The vulnerability reflects a fundamental cognitive asymmetry between human readers, who integrate contextual, phonetic, and visual cues to decipher the intent behind deliberate obfuscation, and text-based LLMs, which rely primarily on token sequences and have limited exposure to evolving evasions. To bridge this asymmetry, we propose PASP (Perception-Aware Adversarial Self-Play), which combines multi-perceptual views with adversarial self-play to improve understanding of harmful intent as evasions evolve. Specifically, PASP constructs textual, phonetic, and visual views of e
    
[^196]: 从句子级监督中学习命题级链接以实现神经符号自然语言推理

    Learning Proposition-Level Links from Sentence-Level Supervision for Neuro-Symbolic NLI

    [https://arxiv.org/abs/2610.15158](https://arxiv.org/abs/2610.15158)

    该论文提出一种可训练的原子链接器，无需原子对标注，通过将跨公式链接结构视为潜在变量并仅从句子级NLI标签进行训练，即可联合预测命题间链接的存在性、源选择和关系类型，从而提升神经符号自然语言推理系统的性能。

    

    我们研究了使用前提和假设的独立翻译命题表示的神经符号自然语言推理（NLI）。在符号推理之前，系统必须推断其命题级单元（原子）之间的语义链接。现有的基于抽象含义表示（AMR）和布尔可满足性（SAT）的系统使用固定的基于嵌入的源选择方法以及未针对原子级链接训练的预训练句子级NLI模型来识别这些链接。我们引入了一个可训练的原子链接器，它可以联合预测链接的存在性、源选择和关系类型。在缺乏原子对训练标注的情况下，我们将跨公式链接结构视为潜在变量，并从句子级NLI标签来训练该链接器，同时保持翻译器和SAT推理器固定不变。我们在斯坦福自然语言推理（SNLI）和涉及组合知识的句子（SICK）的完整公开测试集上进行评估。

    arXiv:2610.15158v2 Announce Type: replace  Abstract: We study neuro-symbolic natural language inference (NLI) with independently translated propositional representations of the premise and hypothesis. Before symbolic reasoning, the system must infer semantic links between their proposition-level units (atoms). Existing systems based on Abstract Meaning Representation (AMR) and Boolean satisfiability (SAT) identify these links using fixed embedding-based source selection and pretrained sentence-level NLI models not trained for atom-level linking. We introduce a trainable atom linker that jointly predicts link existence, source selection, and relation type. Without atom-pair training annotations, we treat cross-formula link structures as latent and train the linker from sentence-level NLI labels, keeping the translator and SAT reasoner fixed. We evaluate on the full public test splits of Stanford Natural Language Inference (SNLI) and Sentences Involving Compositional Knowledge (SICK), an
    
[^197]: AlphaRLM：面向基于文献的符号化因子发现的语言模型对齐

    AlphaRLM: Aligning Language Models for Literature-Grounded Symbolic Factor Discovery

    [https://arxiv.org/abs/2610.14959](https://arxiv.org/abs/2610.14959)

    AlphaRLM是一个强化学习框架，通过文献引导算子组装（LEGO）提供语义先验，并借助可执行有效性与分组收益分离等奖励将生成策略与市场反馈对齐，从而训练大语言模型直接从金融文献中构造可执行的符号化Alpha因子。

    

    公式化Alpha挖掘旨在从庞大且充满噪声的搜索空间中发现能够捕捉预测性市场信号的可执行表达式。近期基于大语言模型（LLM）的方法主要依赖提示工程、迭代搜索或种子因子进化，而通过策略学习直接构造因子的路径较少被探索。为解决这一问题，我们提出了AlphaRLM，一个强化学习框架，用于训练大语言模型在无需输入因子表达式的情况下，直接从金融文献中构造可执行的符号化因子。AlphaRLM引入了文献引导算子组装机制，将因子构造建立在文献衍生的思想与可执行算子之上，为探索提供语义先验。它还通过可执行有效性奖励和分组收益分离奖励（后者能够捕捉因子各分位数间的非单调结构）将生成策略与市场反馈对齐。在沪深300、中证1000和标普500上的大量实验表明，AlphaRLM……（摘要至此截断）

    arXiv:2610.14959v2 Announce Type: new  Abstract: Formulaic alpha mining seeks to discover executable expressions that capture predictive market signals in a vast, noisy search space. Recent LLM-based approaches use prompting, iterative search, or seed-factor evolution, leaving direct factor construction through policy learning less explored. To address this, we introduce AlphaRLM, a reinforcement-learning framework that trains LLMs to construct executable symbolic factors from financial literature without input factor expressions. AlphaRLM incorporates Literature-Guided Operator Assembly (LEGO), which conditions factor construction on literature-derived ideas and executable operators, providing semantic priors for exploration. It further aligns the generation policy with market feedback through executable-validity and grouped-return-separation rewards that capture non-monotonic structures across factor quantiles. Extensive experiments on CSI 300, CSI 1000, and S&P 500 show that AlphaRL
    
[^198]: 如何构建一个真正衡量品味的基准测试

    How to Build a Benchmark That Actually Measures Taste

    [https://arxiv.org/abs/2610.14707](https://arxiv.org/abs/2610.14707)

    该论文提出TasteBench——一个测量真实品味判断力的两两对比基准，并通过对照协议揭示并量化了基准测试中的三种效度威胁：位置偏差、词汇泄露和难度天花板。

    

    专家对齐的“品味”基准应该奖励工艺判断力，而不是钻测试空子的能力。我们提出了TasteBench，这是一个涵盖视觉、品牌、书籍和技术项目的两两好坏对比基准，采用固定的评分标准且仅以专家标注作为黄金标准（评审小组n>=3；一致率>=0.60作为主要门槛；弱参与度标签仅用于训练/分析），同时配套一个能将工艺判断力与钻空子行为区分开来的对照协议。将该协议应用于第一轮试点（primary_human_expert；三位真人人类专家；n=250对样本）时，发现并量化了三种效度威胁：两个夸大平凡基线层的构建伪影，以及一个抬高所有分数的难度天花板。(1) 位置偏差：优选项目以0.956的概率出现在槽位A，因此“始终选择A”的策略可得分0.956（公平概率为0.5）。(2) 词汇泄露：仅凭一个文件名词汇线索即可得0.912分。(3) 难度天花板：工艺领域显示出接近天花板的人类一致性。

    arXiv:2610.14707v2 Announce Type: new  Abstract: Expert-aligned "taste" benchmarks should reward craft judgment, not the ability to game the test. We introduce TasteBench, a pairwise good-vs-bad benchmark across visual, brand, book, and tech items under a frozen rubric with expert-only gold (panel n>=3; percent agreement >=0.60 primary gate; weak engagement labels train/analysis-only), together with a control protocol that separates craft judgment from test-gaming. Applying the protocol to a Round-1 pilot (primary_human_expert; three biological human experts; n=250 pairs) surfaces and quantifies three validity threats: two construction artifacts that inflate the trivial-baseline layer, and a difficulty ceiling that inflates every score. (1) Position: the preferred item is slot A with probability 0.956, so an always_item_a policy scores 0.956 (fair chance 0.5). (2) Lexical leakage: a filename token cue scores 0.912. (3) Difficulty ceiling: craft domains show near-ceiling human agreement
    
[^199]: RealityBench：用于评估视觉语言模型记录人类体力工作能力的模拟环境

    RealityBench: Simulated Environments to Evaluate a Vision Language Model's Ability to Document Human Physical Work

    [https://arxiv.org/abs/2610.14397](https://arxiv.org/abs/2610.14397)

    该论文提出RealityBench基准，通过在Blender中模拟十一个真实体力工作环境，评估视觉语言模型在协助记录人类物理工作（包括判断图像可接受性和生成工作文档）方面的能力。

    

    通过指导来增强工人在物理世界中的能力，一直是AI智能体数十年来的目标。其中一个关键的支持能力是理解工人正在做什么——通过人类佩戴的相机或智能眼镜进行观察，并协助他们进行工作记录。尽管视觉语言模型已经取得进展，但它们在这些场景中的能力仍然不确定，且难以进行基准测试，因为大多数体力工作数据是专有的，且变异性太大，无法以系统化、有原则的方式进行 研究。与厨房和仓库等光照、几何结构和物体都可预测的结构化环境不同，屋顶太阳能安装和车辆维修等面向现场的任务是无约束且高变异的。在此工作中，我们在Blender中模拟了真实世界的体力工作，涵盖十一个环境，并纳入了图像可接受性和工作记录两个方面的变体。可接受性任务询问某个特定视角是否……（摘要在此处被截断）

    arXiv:2610.14397v2 Announce Type: cross  Abstract: Augmenting worker capabilities in the physical world through guidance has been a decades-long goal for AI agents. One key supporting capability is to understand what the worker is doing, seeing through the human's camera or smart glasses, and assist them in documentation. Despite advances in vision language models, their capabilities in these settings remain uncertain and are challenging to benchmark because most physical work data is proprietary, and involves too much variability to study in a principled manner. Unlike structured settings such as kitchens and warehouses that have predictable lighting, geometry, and objects, field-facing tasks such as rooftop solar installation and vehicle repair are unconstrained and high variance. Here, we model real-world physical work in Blender across eleven environments, incorporating variants in both image acceptability and documentation. The acceptability task asks whether a particular view wou
    
[^200]: 评估大语言模型作为真实世界客户支持技能中的智能体工作流执行器

    Evaluating LLMs as Agentic Workflow Executors in Real-World Customer Support Skills

    [https://arxiv.org/abs/2610.14346](https://arxiv.org/abs/2610.14346)

    该论文提出了一个从真实客户支持对话记录中挖掘并构建的基准，用于评估大语言模型在策略与工具约束下执行有状态的多轮工具调用工作流的能力，并对17个闭源与开源模型进行了大规模零样本评测。

    

    企业客户支持智能体日益依赖大语言模型（LLM）来执行有界的、使用工具的工作流。我们通过“技能大语言模型”（Skill LLM）的视角研究这一问题：这类专用执行器在选定的技能上下文中运行，必须在策略和工具的约束下完成范围狭窄的、有状态的工作流。我们提出了一个基准构建流程，该流程从真实世界的对话记录中挖掘重复出现的工作流，将其转换为结构化的技能描述，映射到工具需求，使用与模型上下文协议（MCP）兼容的模拟工具来模拟环境状态，并生成多轮次使用工具的对话。基于该基准，我们对17个语言模型进行了大规模零样本评估，涵盖闭源模型系列以及参数量从1.7B到14B的面向部署的开源权重模型，每个模型由五个强大的LLM评审员独立评判，此外还有一个确定性的基于规则的评估器来检查工具调用行为。

    arXiv:2610.14346v2 Announce Type: replace  Abstract: Enterprise customer-support agents increasingly rely on Large Language Models (LLMs) to execute bounded, tool-using workflows. We study this problem through the lens of Skill LLMs: specialized executors that operate within a selected skill context and must complete a narrow, stateful workflow under policy and tool constraints. We present a benchmark construction pipeline that mines recurrent workflows from real-world transcripts, converts them into structured skill descriptions, maps them to tool requirements, simulates environment state with Model Context Protocol (MCP)-compatible mocks, and generates multi-turn tool-using dialogues. Using this benchmark, we conduct a large-scale zero-shot evaluation of 17 language models spanning closed model families and deployment-oriented open-weight models from 1.7B to 14B parameters, each judged independently by five strong LLM judges, plus a deterministic rule-based evaluator that checks tool
    
[^201]: 从结构到有效性：机制引导的大语言模型评分量规生成与优化

    From Structure to Validity: Mechanism-Guided Generation and Refinement of LLM Rubrics

    [https://arxiv.org/abs/2610.13375](https://arxiv.org/abs/2610.13375)

    该研究系统揭示了LLM评分量规设计的有效机制——结构化的标准组织最为关键，并据此提出ARTS-Hybrid方法，通过自适应量规树构建与语义优先、响应校准的精炼策略，同时提升了端到端评估性能与量规的内在质量。

    

    评分量规已被广泛用于评估和优化语言模型在开放式任务上的表现。然而，现有工作主要聚焦于如何生成更好的量规，对于什么样的量规设计才是有效的缺乏深入理解。在这项工作中，我们围绕标准组织、标准操作化和分数聚合三个维度，对量规设计进行了系统的机制分析。分析结果表明：结构化的组织方式有助于构建更全面的标准；仅靠语义描述并不足以保证有效的标准级判断；而复杂的分数聚合只能带来有限的额外收益。基于这些发现，我们提出了带有混合精炼的自适应量规树搜索方法（ARTS-Hybrid），它将自适应树构建与“语义优先、响应校准”的精炼策略相结合。在多个基准上的实验表明，ARTS-Hybrid 同时提升了端到端评估性能和量规的内在质量。

    arXiv:2610.13375v1 Announce Type: new  Abstract: Rubrics have been widely used to evaluate and optimize language models on open-ended tasks. However, existing work primarily focuses on generating better rubrics, with limited understanding of what makes rubric design effective. In this work, we conduct a systematic mechanistic analysis of rubric design across criterion organization, criterion operationalization, and score aggregation. Our analysis shows that structural organization enables more comprehensive criterion construction, semantic specification alone does not ensure effective criterion-level judgments, and sophisticated score aggregation provides limited additional gains. Guided by these findings, we introduce Adaptive Rubric Tree Search with Hybrid refinement (ARTS-Hybrid), which combines adaptive tree construction with semantic-first, response-calibrated refinement. Experiments across benchmarks show that ARTS-Hybrid improves both end-to-end evaluation performance and intrin
    
[^202]: HarnessSQL：面向真实数据库环境中SQL代理的执行框架原生训练

    HarnessSQL: Harness-Native Training for SQL Agents in Realistic Database Environments

    [https://arxiv.org/abs/2610.12274](https://arxiv.org/abs/2610.12274)

    HarnessSQL提出了一种框架原生的后训练方法，通过在带有隐藏执行预言机的真实数据库多轮交互环境中进行监督微调和执行奖励强化学习，弥合了文本到SQL模型静态训练与有状态实际部署之间的差距，显著提升了Spider 2.0-SQLite上的执行准确率。

    

    文本到SQL（Text-to-SQL）模型通常被训练为将问题直接映射到静态查询，而现实世界中的数据库代理则是通过与活数据库进行有状态的多轮交互来运作的——检查数据库模式、执行探测查询、诊断错误并修正假设。这造成了关键的训练-部署不匹配问题，因为调节这种交互的执行框架只在推理阶段才被引入。为了弥合这一差距，我们提出了HarnessSQL，一个框架原生的后训练框架，它在监督微调和强化学习的全过程中完整保留了交互结构。HarnessSQL构建了隔离的、可执行的数据库环境，并配有隐藏的执行预言机，直接在目标SQL执行框架内对教师模型进行推演，仅保留经过验证的轨迹用于全序列SFT，随后进行基于执行奖励的强化学习。在Spider 2.0-SQLite基准上，HarnessSQL显著提升了执行准确率。

    arXiv:2610.12274v1 Announce Type: new  Abstract: Text-to-SQL models are commonly trained to map questions directly to static queries, whereas real-world database agents operate through stateful, multi-turn interaction with live databases -- inspecting schemas, executing probe queries, diagnosing errors, and revising hypotheses. This creates a critical train-deploy mismatch, as the execution harness that mediates this interaction is introduced only at inference time. To bridge this gap, we propose HarnessSQL, a harness-native post-training framework that preserves the full interaction structure throughout both supervised fine-tuning and reinforcement learning. HarnessSQL builds isolated, executable database environments paired with hidden execution oracles, rolls out teachers directly inside the target SQL harness, and retains only verified trajectories for full-sequence SFT, followed by execution-reward RL. Across Spider 2.0-SQLite, HarnessSQL dramatically boosts the execution accuracy
    
[^203]: ILM：一个AI驱动的叙事教育工具

    ILM: An AI-Powered Storytelling Educational Tool

    [https://arxiv.org/abs/2610.12064](https://arxiv.org/abs/2610.12064)

    本文提出了ILM平台，通过结合阿拉伯语自然语言处理、知识图谱构建和多语言检索技术，为先知故事提供结构化、可视化和多语言支持的互动式学习与理解评估体验。

    

    数字技术已经使伊斯兰叙事更容易获取，但现有平台对这些故事的结构化学习和理解提供的支持有限，尤其是在阿拉伯语和多语言环境中。我们提出了ILM，这是一个面向先知故事的互动教育平台，它结合了阿拉伯语自然语言处理、结构化知识表示和基于检索的问题生成。经管理员审核的阿拉伯语叙事由知识图谱（KG）构建引擎处理，该引擎识别实体和叙事关系并将其存储为结构化知识，使学习者能够通过可视化故事地图探索故事，并回答由知识图谱生成的基于实体和关系的问题。此外，一个多语言检索流水线从原始叙事中检索相关段落，以生成多项选择和开放式理解问题。对于开放式问题，（原文摘要在此处截断）

    arXiv:2610.12064v1 Announce Type: new  Abstract: Digital technologies have made Islamic narratives more accessible, but existing platforms provide limited support for structured learning and comprehension of these stories, particularly in Arabic and multilingual settings. We present ILM, an interactive educational platform for Stories of the Prophets that combines Arabic natural language processing, structured knowledge representation, and retrieval-based question generation. Admin-approved Arabic narratives are processed by a Knowledge Graph (KG) Constructor Engine that identifies entities and narrative relationships and stores them as structured knowledge, enabling learners to explore stories through a visual story map and answer entity- and relation-based questions generated from the KG. Separately, a multilingual retrieval pipeline retrieves relevant passages from the original narratives to generate multiple-choice and open-ended comprehension questions. For open-ended questions, a
    
[^204]: 大语言模型会从上下文中的奖励中学习吗？：重新思考奖励在上下文强化学习中的作用

    Do LLMs Learn from Rewards in Context? : Rethinking the role of reward in In-Context Reinforcement Learning

    [https://arxiv.org/abs/2610.11152](https://arxiv.org/abs/2610.11152)

    该研究通过受控实验发现，在大语言模型的直接上下文强化学习中，奖励信号虽然被读取却几乎不产生学习效果，而轨迹本身（即使语义被打乱或损坏）才是驱动上下文改进的关键，从而挑战了上下文学习真正实现强化学习的假设。

    

    大语言模型智能体越来越多地通过在上下文中积累经验而非更新参数来在推理阶段提升性能，这一过程通常被称为上下文强化学习（ICRL）。然而，上下文学习（ICL）是否真的能够发挥强化学习的作用，尚未经过检验。我们在其最简单的形式——直接ICRL中研究这一问题，即模型直接以原始的轨迹-奖励对为条件，并探究奖励是否真正充当学习信号。通过对六个模型在四个基准上的受控实验，我们发现奖励虽然会被模型读取，但其影响很小：翻转、随机化或移除奖励几乎不会改变性能改进曲线，即使在通过元提示明确指示模型进行探索、利用或对奖励进行推理的情况下也是如此。轨迹驱动了性能改进，但并非通过其语义内容：打乱或损坏的轨迹与真实轨迹的效果相当。

    arXiv:2610.11152v1 Announce Type: cross  Abstract: LLM agents increasingly improve at inference time by accumulating experience in context rather than by updating parameters. This process is often described as in-context reinforcement learning (ICRL). Whether in-context learning (ICL) can actually play the role of RL, however, has not been tested. We study this question in its simplest form, direct ICRL, where the model conditions directly on raw trajectory-reward pairs, and ask whether the reward acts as a learning signal. Through controlled experiments on four benchmarks across six models, we find that the reward is read, but its effect is small: flipping, randomizing, or removing the reward leaves the improvement curve almost unchanged, and this holds even under meta-prompts that explicitly instruct the model to explore, exploit, or reason over rewards. Trajectories drive improvement, but not through their semantic content: shuffled or corrupted trajectories work as well as real one
    
[^205]: ARCS：通过结构化消歧实现精确的文本到SQL转换

    ARCS: Towards Precise Text-to-SQL via Structured Disambiguation

    [https://arxiv.org/abs/2610.09396](https://arxiv.org/abs/2610.09396)

    该论文提出“结构化消歧”新范式，通过显式受约束的交互取代自由对话来解决文本到SQL中的用户问题歧义，并构建了首个基于真实数据库、包含自然产生歧义及完整标注的基准数据集ARCS。

    

    随着文本到SQL（text-to-SQL）系统从演示走向真实世界的部署，用户问题中的歧义成为错误的主要来源。这类歧义往往十分微妙、具有领域或数据特异性，并可能在不知不觉中导致系统输出偏离用户的真实意图。传统上，歧义通过对话式澄清来解决，但这种方式往往效率低下、认知负担重，且与真实世界的用户工作流程契合度较差。我们提出了结构化消歧（structured disambiguation）这一新范式，通过显式的、受约束的交互而非自由形式的对话来解决歧义。我们构建了ARCS（Ambiguity Resolution Corpus for SQL，SQL歧义消解语料库），这是首个基于真实世界数据库、包含自然产生且不受约束歧义的文本到SQL基准，并对所有有效的歧义点、各种解释及相应的SQL查询进行了完整标注。实验结果表明，文本到SQL在……（原文摘要至此截断）

    arXiv:2610.09396v1 Announce Type: new  Abstract: As text-to-SQL systems move beyond demonstrations toward real-world deployment, ambiguity in user questions becomes a primary source of errors. Such ambiguities are often subtle, domain- or data-specific, and can silently cause system outputs to deviate from the user's true intent. Ambiguity is traditionally addressed through conversational clarification, which is often inefficient, cognitively demanding, and poorly aligned with real-world user workflows. We propose structured disambiguation, a new paradigm in which ambiguity is resolved through explicit, constrained interactions rather than free-form dialogue. We construct ARCS (Ambiguity Resolution Corpus for SQL), the first text-to-SQL benchmark featuring naturally occurring, unconstrained ambiguities over real-world databases, with complete annotations of all valid ambiguity points, interpretations, and SQL queries. Experimental results show that text-to-SQL remains challenging in th
    
[^206]: 自我回溯蒸馏：将事后经验转化为先验预见

    Self-Retrospection Distillation: Turning Post-hoc Experiences into Prior Foresight

    [https://arxiv.org/abs/2610.08077](https://arxiv.org/abs/2610.08077)

    该论文提出前瞻学习与自我回溯蒸馏方法，将已完成的轨迹作为特权后见之明，蒸馏为同一策略在交互前的轨迹无关的预见性预测，从而在组相对奖励信号消失时仍能从经验中有效学习。

    

    具有可验证奖励的强化学习（RLVR）主要通过交互后的标量结果奖励将智能体经验转化为学习信号。然而，对于组相对目标而言，当所有采样轨迹获得相同奖励时，这一信号便会消失，即使这些轨迹可能揭示了关于任务需求以及智能体如何失败的有用信息。我们提出了一个互补的问题：事后反思能否教会智能体在行动之前本可预见的东西？我们引入前瞻学习，利用事后经验来监督交互前视角下的预见性预测，并通过自我回溯蒸馏（SRD）加以实例化。直观地说，一条已完成的轨迹揭示了本会有用的知识和本应避免的陷阱；SRD将这种特权的后见之明蒸馏到同一策略的、不依赖轨迹的前瞻预测中。前瞻仅作为训练目标，无需成为……

    arXiv:2610.08077v1 Announce Type: new  Abstract: Reinforcement learning with verifiable rewards (RLVR) turns agent experience into learning signals primarily through scalar outcome rewards after interaction. For group-relative objectives, however, this signal vanishes when all rollouts receive the same reward, even though their trajectories may reveal useful information about what the task requires and how the agent fails. We ask a complementary question: can hindsight teach an agent what it could have anticipated before acting? We introduce prospective learning, which uses post-hoc experience to supervise foresight predictions from the pre-interaction view, and instantiate it with Self-Retrospection Distillation (SRD). Intuitively, a completed trajectory reveals knowledge that would have been useful and pitfalls that should be avoided; SRD distills this privileged hindsight into trajectory-blind foresight of the same policy. Foresight serves only as a training target and need not be e
    
[^207]: 迷失于bf16类型转换：导出三值语言模型可能抵消大部分低学习率代码改动的效果

    Lost in the bf16 Cast: Exporting Ternary Language Models Can Revert Most Low-Learning-Rate Code Changes

    [https://arxiv.org/abs/2610.07853](https://arxiv.org/abs/2610.07853)

    该研究审计发现，三值语言模型导出流程中先将潜在权重转换为bf16，会在阈值处因“舍入到偶数”规则被错误映射为零，导致Falcon-E-1B-Base部署后GSM8K准确率从58.79%暴跌至0.78%，几乎抹去了低学习率微调带来的改进。

    

    BitNet b1.58、Falcon-E和BitCPM等三值语言模型先以更高精度的潜在权重进行微调，再通过导出步骤生成三值代码进行部署；在三个实验室公开记录的流程中，该导出步骤均会先将潜在权重转换为bf16。我们对三个实验室的这些流程进行了审计。在已发布的检查点中，对随附潜在权重执行fp32量化所得的结果与实际部署代码在Falcon-E和BitCPM上有0.83%–1.77%的代码不一致，在BitNet 2B-4T上有1.530%不一致；对于Falcon-E和BitCPM，绝大多数不一致源于这样的乘积：其bf16舍入恰好落在阈值上，并被“舍入到偶数”（ties-to-even）规则映射为零，而未经修改的onebitllms导出器可以逐字节复现全部四个Falcon-E发布版本。在微调终点，当学习率按标称的学习率与bf16-ULP之比选取时，按文档所述流程导出会使Falcon-E-1B-Base的贪婪解码GSM8K严格准确率从58.79%跌至0.78%，使BitCPM-CANN-（原文摘要在此处截断）的准确率从36.13%跌至0.39%。

    arXiv:2610.07853v1 Announce Type: cross  Abstract: Ternary language models such as BitNet b1.58, Falcon-E and BitCPM are fine-tuned with higher-precision latent weights and deployed as ternary codes produced by an export step that, in the labs' documented pipelines, first casts the latents to bf16. We audit those pipelines across three labs. In released checkpoints, fp32 quantization of the shipped latents disagrees with the deployed codes on 0.83-1.77% of codes in Falcon-E and BitCPM and on 1.530% in BitNet 2B-4T; for Falcon-E and BitCPM most disagreements are products that bf16 rounding lands exactly on the threshold, which ties-to-even maps to zero, and the unmodified onebitllms exporter reproduces all four Falcon-E releases byte for byte. At fine-tuned endpoints, with learning rates selected to match a nominal learning-rate-to-bf16-ULP ratio, the documented export lowers greedy GSM8K strict accuracy from 58.79% to 0.78% for Falcon-E-1B-Base and from 36.13% to 0.39% for BitCPM-CANN-
    
[^208]: 用于KV缓存驱逐的蒙特卡洛估计

    Monte Carlo Estimation for KV Cache Eviction

    [https://arxiv.org/abs/2610.07643](https://arxiv.org/abs/2610.07643)

    提出免训练方法LORE-KV，通过蒙特卡洛采样冻结模型的短自回归延续并以响应侧查询状态估计提示token效用，将KV缓存驱逐从“回顾过去”转变为“预测未来”，在回答时保留真正重要的记忆。

    

    大多数KV缓存驱逐方法实际上在问：哪些内存在阅读提示时显得重要？我们转而问：哪些内存在回答时会有用？由于在驱逐时无法获得解码查询，先前的面向未来的方法依赖于伪响应或合成的未来查询估计。我们将固定预算的面向未来的驱逐问题转化为对合理的模型条件查询轨迹的分布估计，并引入LORE-KV（基于可靠性加权集成的前瞻输出扰动KV缓存方法），这是一种免训练方法，它从冻结的目标模型中采样简短的自回归延续，并利用其响应侧的查询状态来估计提示token的效用。token通过投影的留一法注意力输出删除代价进行评分，并在采样的未来之间进行聚合（可选地带有轨迹加权）。临时延续在最终解码之前被丢弃，无需

    arXiv:2610.07643v1 Announce Type: cross  Abstract: Most KV-cache eviction methods ask, in effect, which memory appeared important while reading the prompt? We instead ask, which memory will matter while answering? Since decoding queries are unavailable at eviction time, prior future-aware methods rely on pseudo-responses or synthetic future-query estimates. We cast fixed-budget future-aware eviction as distributional estimation over plausible model-conditional query trajectories and introduce LORE-KV (Lookahead Output-perturbation with Reliability-weighted Ensembles for Key-Value caches), a training-free method that samples short autoregressive continuations from the frozen target model and uses their response-side query states to estimate prompt-token utility. Tokens are scored by projected leave-one-out attention-output deletion cost and aggregated across sampled futures with optional trajectory weighting. The temporary continuations are discarded before final decoding, requiring no 
    
[^209]: 边缘设备上的质量感知自校正语音翻译

    Quality-Aware Self-Correcting Speech Translation on an Edge Device

    [https://arxiv.org/abs/2610.07545](https://arxiv.org/abs/2610.07545)

    该论文提出了一种在Jetson Nano边缘设备上运行的完全离线自校正语音翻译流水线，发现质量估计（QE）适合作为触发二次校正的门控而不适合作为候选重排序器，其中最小贝叶斯风险解码在阈值0.90下取得了统计显著的翻译质量提升。

    

    我们提出了一个完全离线的语音到语音翻译流水线，可在Jetson Nano（4 GB）上运行，并在无需重新训练的情况下校正自身的弱翻译。Whisper-tiny语音识别模型为Opus-MT翻译器提供输入；多语言BERT余弦相似度作为质量估计（QE）门控，当置信度低于预定义阈值τ时触发二次校正。我们比较了三种校正方法：QE重排序（M1）、最小贝叶斯风险解码（M2）和受限束搜索（M3）。在1,012句FLORES-200句子（英语-西班牙语）上，τ=0.90时的M2相对贪婪解码在BLEU（+0.67，p<0.001）、ChrF（+0.51，p<0.001）和COMET（N=3时+0.0020，p=0.002）上产生了统计显著的改进；M1没有产生显著增益，而M3显著差于基线（p>0.99）。我们的核心发现是：QE作为门控有效，但作为排序器效果不佳：从候选选择中移除QE模型……

    arXiv:2610.07545v1 Announce Type: new  Abstract: We present a fully offline speech-to-speech translation pipeline that runs on a Jetson Nano (4 GB) and corrects its own weak translations without retraining. A Whisper-tiny ASR feeds an Opus-MT translator; multilingual BERT cosine similarity acts as a Quality Estimation (QE) gate, triggering a secondary-pass correction when confidence falls below a pre-defined threshold $\tau$. We compare three correction methods: QE reranking (M1), Minimum Bayes-Risk decoding (M2), and constrained beam search (M3). On 1,012 FLORES-200 sentences (English-Spanish), M2 at $\tau=0.90$ produces statistically significant improvements over greedy decoding on BLEU (+0.67, p<0.001), ChrF (+0.51, p<0.001), and COMET (+0.0020 at N=3, p=0.002); M1 yields no significant gains, and M3 is significantly worse than baseline (p>0.99). Our central finding is that QE functions effectively as a gate but poorly as a ranker: removing the QE model from candidate selection (M1$
    
[^210]: SecJev：将安全专业知识引入系统一决策模型

    SecJev: Bringing Security Expertise to System One Decision Models

    [https://arxiv.org/abs/2610.03073](https://arxiv.org/abs/2610.03073)

    该论文提出SecJev，首个专门面向安全领域的类Jev决策模型家族（参数规模0.8B至9B），能够从文本、遥测和观察历史中学习布尔型、选择型和有序的安全决策，并通过SecJev语料库在14个任务和8个数据源上统一实现源标签预测与显式策略评估。

    

    安全工作流需要能够将复杂观察和明确策略转化为决策的模型。Jev提出的系统一模型可以返回类型化的预测和概率，而安全专业化则为这些预测提供了领域专业知识支撑。我们提出了SecJev，据我们所知，这是首个专门面向安全领域的类Jev决策模型家族，参数规模覆盖0.8B至9B。SecJev构建于Kev的单次遍历候选评分器之上，能够从文本、遥测数据和观察历史中学习布尔型、选择型和有序决策。我们开发了SecJev语料库，以在14个任务和8个数据源上统一源标签预测与显式策略评估，涵盖工具输出、流量、联邦更新、共识、身份验证和车辆消息。场景加权训练使模型能够适应这些不同领域，同时保持共享的类型化决策接口。安全专业化提升了该家族中每一个模型的性能。

    arXiv:2610.03073v1 Announce Type: cross  Abstract: Security workflows need models that turn complex observations and explicit policies into decisions. System One models introduced by Jev return typed predictions and probabilities; security specialization supplies the domain expertise behind those predictions. We introduce SecJev, to our knowledge the first family of Jev-like decision models specialized for security, spanning 0.8B to 9B parameters. Built on Kev's single-pass candidate scorer, SecJev learns Boolean, choice, and ordered decisions from text, telemetry, and observation histories. We develop SecJev-Corpus to unify source-label prediction and explicit-policy evaluation across 14 tasks and eight sources. It covers tool outputs, traffic, federated updates, consensus, authentication, and vehicle messages. Scene-weighted training adapts the models across these domains while preserving a shared typed decision interface. Security specialization improves every model in the family; S
    
[^211]: 旧想法，新问题：基于大语言模型的新颖性评估的不稳定性

    Old Ideas, Novel Problems: The Instability of LLM-Based Novelty Evaluation

    [https://arxiv.org/abs/2610.02022](https://arxiv.org/abs/2610.02022)

    本文通过从OpenReview自动挖掘审稿人明确肯定或质疑论文原创性的段落构建评估集，系统性地对照研究了大语言模型新颖性评估器的设计选择，发现微小的提示词改动即可使评估结果在超过一半的相同想法上发生翻转，揭示了此类评估器的严重不稳定性。

    

    基于AI的构思系统（ideation systems）常常以其想法的新颖性作为评估标准，而这种判断正越来越多地交由大语言模型来完成。这类评估器通常是临时搭建的，即使经过验证，也往往是在人类撰写的论文上进行验证，而不是在其本应评分的生成式想法上进行验证。那么，新颖性评估器的实际表现如何？答案并不理想。我们对新颖性评估的设计选择开展了一项系统性的对照研究。我们自动构建了一个评估集：从OpenReview中挖掘审稿人明确肯定或质疑论文原创性的评述段落，仅保留审稿人意见完全一致、且处于其研究领域极端情况下的投稿；并将这些样本与来自标准LLM生成器产生的想法进行配对。在六个评估器上的实验中，我们发现看似微小的提示词设计选择会产生巨大的影响；例如，仅仅告诉评估器“审稿人认为一个想法新颖而另一个不新颖”，就能改变其对超过一半的相同……

    arXiv:2610.02022v2 Announce Type: replace  Abstract: AI-based ideation systems are often evaluated on the novelty of their ideas, and that judgment is increasingly delegated to large language models. Such judges are typically built ad hoc and validated, if at all, on human-authored papers rather than on the generated ideas they are meant to score. So, how do novelty judges perform? Not well. We present a systematic controlled study of novelty evaluation design choices. We build an evaluation set automatically, mining OpenReview for passages where reviewers explicitly affirm or dispute a paper's originality and keeping only submissions with unanimous agreement at the extremes of their research area; we pair these with ideas from a vanilla LLM generator. Across six judges, we find that seemingly small prompt design choices have large consequences; e.g., simply telling the judge that reviewers found one idea novel and the other not can change its verdict on more than half of the identical
    
[^212]: 策略条件化的AI使用检测：面向学术出版的证据框架

    Policy-Conditioned AI-Use Detection: An Evidentiary Framework for Academic Publishing

    [https://arxiv.org/abs/2609.38427](https://arxiv.org/abs/2609.38427)

    该论文提出“策略条件化AI使用检测”的证据框架，将传统上判断文本是否由AI撰写的检测任务，转变为评估人-AI工作流是否遵守学术出版机构具体AI使用政策的证据推理与合规性评估。

    

    目前主要的学术会议和期刊已发布了关于作者、审稿人和领域主席如何使用AI的详细规则，这些规则因角色、任务以及必须披露的内容而各不相同。而通常被提议用于执行这些规则的工具——AI检测，所估计的却是另一回事：文本是否由AI模型撰写。我们认为这一目标与会议和期刊面临的实际决策并不一致，因此提出“策略条件化的AI使用检测”，这是一个用于评估人-AI工作流是否遵守既定规则的证据框架。在该框架中，策略使治理规则成为显式的输入；推理过程报告的是假设、证据、校准状态和不确定性，而非“检测到AI”之类的判定。评估方面，通过可复现的流程构建基准数据，生成合规与不合规的工作流，并在出版方预先设定的假阳性率下报告真阳性率。我们将该框架应用于同行评审场景，在给定假阳性率条件下……

    arXiv:2609.38427v1 Announce Type: cross  Abstract: Major venues now publish detailed rules about how authors, reviewers, and area chairs may use AI, and those rules differ by role, by task, and by what must be disclosed. AI detection, the instrument usually proposed to enforce them, estimates something else: whether an AI model wrote the text. We argue that this target is misaligned with the decisions conferences and journals face, and propose policy-conditioned AI-use detection, an evidentiary framework for assessing whether a human--AI workflow complied with a stated rule. Policy makes the governing rule an explicit input. Inference reports hypotheses, evidence, calibration regime, and uncertainty in place of verdicts such as "AI detected". Evaluation builds benchmarks from reproducible pipelines that generate compliant and non-compliant workflows, and reports true positive rate at a false positive rate the venue fixes in advance. We work the framework through peer review, where at p
    
[^213]: 一个从纵向文本数据中建模组织级语义身份的计算框架

    A Computational Framework for Modelling Organisation-Level Semantic Identity from Longitudinal Textual Data

    [https://arxiv.org/abs/2609.29584](https://arxiv.org/abs/2609.29584)

    该论文提出了一个统一的计算框架，首次将组织级语义身份建模为可解释且随时间演化的语义构建，整合了语义表示学习、图语义建模、组织语义指纹、时序演化分析与证据驱动验证。

    

    组织持续产生大量文本数据，这些数据记录了组织如何随时间进行沟通、演化并形成自身差异。尽管自然语言处理的最新进展已显著提升了组织层面的文本分析能力，但现有方法主要将组织表示为用于相似度估计、分类或检索的潜在嵌入向量或预测性特征向量。因此，目前尚缺乏一个通用的计算框架，能够将组织级语义身份建模为一种可解释的、基于纵向文本证据而不断演化的语义构建。本文提出了一个计算框架，将语义表示学习、基于图的语义建模、组织级语义指纹、时序语义演化以及证据驱动的验证整合在一个统一的分析方法论之中。组织被刻画……（原文摘要在此处截断）

    arXiv:2609.29584v1 Announce Type: cross  Abstract: Organisations continuously generate large volumes of textual data that capture how they communicate, evolve and differentiate themselves over time. Although recent advances in natural language processing have substantially improved organisation-level text analytics, existing approaches primarily represent organisations as latent embeddings or predictive feature vectors for similarity estimation, classification or retrieval. Consequently, there is currently no general computational framework for modelling organisation-level semantic identity as an interpretable and evolving semantic construct derived from longitudinal textual evidence. This paper introduces a computational framework that integrates semantic representation learning, graph-based semantic modelling, organisation-level semantic fingerprints, temporal semantic evolution and evidence-driven validation within a unified analytical methodology. Organisations are characterised th
    
[^214]: EduBehaviors：面向可审计教育对话编码的基于断言的模式框架

    EduBehaviors: Assertion-based Schemas for Auditable Coding of Educational Dialogues

    [https://arxiv.org/abs/2609.27043](https://arxiv.org/abs/2609.27043)

    提出了EduBehaviors框架，利用大语言模型测量教育对话中重复出现的可观察行为并据此学习分类器，实现了可解释、可审计的教育对话标注，其性能与直接提示方法相当。

    

    大语言模型使得与目标教学构念相对应的教学标注能够快速部署，为在对话数据集上生成分类提供了自然语言接口。然而，由于大语言模型推理的不透明性，我们无法获得关于模型为何为某句话语选择特定标签的可验证的、机制层面的洞察。我们提出了EduBehaviors框架，这是一种可解释、可扩展的教育数据标注方法，它利用大语言模型测量与多个目标构念相关的重复出现的可观察行为，然后基于这些可观察行为学习该构念的分类器。我们在TalkMoves数据集上对该框架进行了评估，预测教师TalkMoves标签。我们的最佳配置取得了0.673的宏平均F1值和0.688的Cohen's kappa系数，证明其与直接提示方法相比具有竞争力。此外，我们发布了EduBehaviors工具包，包含两个供研究人员使用的工具。

    arXiv:2609.27043v1 Announce Type: new  Abstract: Large language models have allowed the rapid deployment of pedagogical annotations corresponding to constructs of interest, allowing a natural language interface for generating classifications on a conversational dataset. However due to the opaque nature of LLM reasoning, we have no verifiable, mechanistic insight into why a model chose a label for an utterance. We introduce the EduBehaviors framework, an interpretable, scalable approach to annotating educational data that uses LLMs to measure repeated observable behaviors relevant to many constructs of interest and then learns a classifier for the construct based on these observable behaviors. We evaluate the framework on the TalkMoves dataset, predicting the Teacher TalkMoves labels. Our best configuration results in a macro-F1 of 0.673 and 0.688 Cohen's kappa, proving competitive with direct prompting approaches. In addition, we release EduBehaviors Toolkit, two tools allowing researc
    
[^215]: MemoryAthena：潜在记忆与生成记忆之上的自适应路由

    MemoryAthena: Adaptive Routing over Latent and Generated Memories

    [https://arxiv.org/abs/2609.25853](https://arxiv.org/abs/2609.25853)

    MemoryAthena 提出在直接检索、基于检索线索生成和无表生成三条记忆路径之间进行自适应因果路由，在冻结主干与记忆模块的情况下，依据反事实似然优势学习轻量路由头，并通过有界插值将生成的记忆按情境动态融合到检索残差中。

    

    学习型记忆方法将信息存储在显式表格中，并通过单独的读取器来消费这些信息，使得寻址、存储和读取可以独立修改。我们研究有用的记忆是否也可以被生成，而不仅仅是被检索。MemoryAthena 使用三条路径：直接 Engram 检索（E）、从检索到的 Engram 线索生成（GE），以及在不查询记忆表的情况下从因果主干状态生成（GH）。生成的记忆是有条件有用的：它在一种情境中可以补充 E，但在另一种情境中可能干扰 E。因此，MemoryAthena 将 E 视为锚点，并学习生成的表示何时应当介入。在主干、记忆、生成器和读取器全部冻结的条件下，利用 GE 和 GH 相对于 E 的反事实未来词元似然优势，训练一个轻量级的因果路由头。在推理时，被接纳的候选通过有界插值修改 E 的残差……

    arXiv:2609.25853v1 Announce Type: new  Abstract: Learned-memory methods store information in an explicit table and consume it through a separate reader, allowing addressing, storage, and reading to be modified independently. We study whether useful memory can also be generated rather than only retrieved. MemoryAthena uses three pathways: direct Engram retrieval (E), generation from retrieved Engram cues (GE), and generation from causal backbone states without consulting the memory table (GH). Generated memory is conditionally useful: it can complement E in one context but interfere with it in another. MemoryAthena therefore treats E as an anchor and learns when a generated representation should intervene. With the backbone, memory, generators, and readers frozen, a lightweight causal routing head is trained from counterfactual future-token likelihood advantages of GE and GH relative to E. At inference time, an admitted candidate modifies the E residual through bounded interpolation, wh
    
[^216]: ARAFA：一个由大语言模型生成的阿拉伯语事实核查数据集

    ARAFA: An LLM-Generated Arabic Fact-Checking Dataset

    [https://arxiv.org/abs/2609.25833](https://arxiv.org/abs/2609.25833)

    研究者利用大语言模型通过三步自动化流水线构建了包含18万余个标注声明-证据对的大规模阿拉伯语事实核查数据集Arafa，有效缓解了阿拉伯语事实核查资源稀缺的问题。

    

    由于数据集和资源的稀缺，自动事实核查在阿拉伯语自然语言处理领域构成了重大挑战。在本文中，我们介绍了Arafa，一个新的大规模现代标准阿拉伯语事实核查数据集，该数据集通过一个利用大语言模型（LLM）的自动化框架构建而成。数据集的构建采用三步流水线：（1）从阿拉伯语维基百科页面生成带有支持性文本证据的声明；（2）通过声明变异生成带有反驳证据的具有挑战性的反事实声明；（3）通过自动验证步骤，检验生成的声明是被其伴随证据所支持或反驳，还是证据不足以判断声明的有效性。最终得到的数据集包含181,976个声明-证据对，标注为“支持”、“反驳”或“信息不足”。在样本上进行的人工评估……（原文摘要在此处截断）

    arXiv:2609.25833v1 Announce Type: new  Abstract: Automatic fact-checking poses a significant challenge in Arabic natural language processing due to the scarcity of datasets and resources. In this manuscript, we introduce Arafa, a new large-scale dataset for fact-checking in Modern Standard Arabic, constructed through an automated framework leveraging large language models (LLMs). The dataset was constructed through a three-step pipeline: (1) claim generation from Arabic Wikipedia pages with supporting textual evidence, (2) claim mutation to generate challenging counterfactual claims with refuting evidence, and (3) an automatic validation step to validate that the generated claims are either supported or refuted by their accompanying evidence, or if the evidence does not provide enough information to judge the validity of the claims. The resulting dataset comprises 181,976 claim-evidence pairs labeled as supported, refuted, or not enough information. Human evaluation carried out on a te
    
[^217]: 一种基于LangGraph的迭代式Text-to-SQL智能体：芝加哥犯罪数据库的自然语言访问

    An Iterative LangGraph Agent for Text-to-SQL: Natural Language Access to the Chicago Crime Database

    [https://arxiv.org/abs/2609.22917](https://arxiv.org/abs/2609.22917)

    该论文构建了一个仅依靠提示工程（无需微调）的六节点LangGraph迭代式Text-to-SQL智能体，通过问题相关性检查、实时模式获取、SQL生成、试运行验证与失败重试等流程，让非技术用户能用自然语言查询芝加哥犯罪数据库，并将有效SQL率提升至93%、执行准确率提升至60%。

    

    非技术背景的用户往往无法编写从运营数据库中提取洞察所需的SQL。我们构建并评估了一个端到端弥合这一差距的Text-to-SQL智能体：一个六节点的LangGraph StateGraph会检查问题的相关性、获取实时数据库模式、生成PostgreSQL查询、通过试运行进行验证、在失败时重试、执行查询，并用通俗的英语叙述结果集。该智能体仅使用提示工程，未对模型进行任何微调。我们在芝加哥犯罪数据集（约850万条记录、22个属性）上，针对一个手工构建的包含100个自然语言问题及标准答案SQL的基准进行了评估，该基准按难度分为30个简单、40个中等和30个困难题目。通过对比同一智能体的两个提示词修订版本，修订后的系统（V2）在混合关系等价性指标下将有效SQL率提升至93%（此前为87%），执行准确率提升至60%（此前为47%）；在严格的JSON匹配标准下则为19%（此前为12%）。

    arXiv:2609.22917v1 Announce Type: new  Abstract: Non-technical stakeholders frequently cannot write the SQL needed to extract insights from operational databases. We built and evaluated a Text-to-SQL agent that closes this gap end to end: a six-node LangGraph StateGraph checks question relevance, fetches the live schema, generates PostgreSQL, validates it with a dry run, retries on failure, executes the query, and narrates the result set in plain English. The agent uses prompt engineering only; no model was fine-tuned. We evaluated it on the Chicago Crime dataset (approximately 8.5 million records, 22 attributes) against a hand-built benchmark of 100 natural language questions with ground-truth SQL, stratified into 30 Easy, 40 Medium and 30 Hard items. Comparing two prompt revisions of the same agent, the revised system (V2) reached a Valid SQL Rate of 93% (from 87%), an Execution Accuracy of 60% under a hybrid relational equivalence metric (from 47%; 19% from 12% under strict JSON mat
    
[^218]: Reviser：通过自回归光标动作实现可修订的文本生成

    Reviser: Revision-Capable Text Generation via Autoregressive Cursor Actions

    [https://arxiv.org/abs/2609.20830](https://arxiv.org/abs/2609.20830)

    Reviser 提出了一种仅解码器 Transformer，通过在可变画布上自回归地预测光标动作（插入、移动、停止）实现真正非单调的可修订文本生成，在保持简单下一动作接口的同时避免了重复的序列级计算，性能优于 SEDD 和 MDLM 等扩散模型。

    

    具备修订能力的生成非常有吸引力，因为它可以插入或修改先前已生成的内容，但许多非自回归和基于编辑的方法需要通过重复的序列级计算来获得这种灵活性。我们提出了 Reviser，一个仅解码器的 Transformer，它将生成过程建模为在可变画布上的一系列光标相对动作。在每一步，Reviser 精确预测一个动作标记：INSERT(token)（插入）、MOVE($\Delta$)（移动）或 STOP（停止），并且是对编辑历史动作而非最终文本顺序进行自回归。这种设计实现了真正的非单调生成，同时保持了简单的“下一动作”接口。在一个续写基准测试中，在我们的竞技场评估里 Reviser 明显优于 SEDD 和 MDLM，且轨迹统计证实该模型确实频繁执行向后移动和画布中部插入，而不仅仅是模拟末尾追加式的解码。与规模匹配的自回归基线相比，Reviser 具有……（原文在此截断）

    arXiv:2609.20830v1 Announce Type: new  Abstract: Revision-capable generation is appealing because it can insert or revise earlier content, but many non-autoregressive and edit-based approaches obtain this flexibility through repeated sequence-level computation. We propose Reviser, a decoder-only Transformer that generates a response as a sequence of cursor-relative actions on a mutable canvas. At each step, Reviser predicts exactly one action token: INSERT(token), MOVE($\Delta$), or STOP, and is autoregressive over edit-history actions rather than final text order. This design enables genuinely non-monotonic generation while preserving a simple next-action interface. On a continuation benchmark, Reviser is strongly preferred to SEDD and MDLM in our arena evaluations, and trajectory statistics confirm that the model performs frequent backward moves and mid-canvas insertions rather than merely emulating end-append decoding. Against size-matched autoregressive baselines, Reviser is compet
    
[^219]: 从出院记录到患者理解：基于人设的开放式大语言模型出院教育者模拟

    From Discharge Notes to Patient Understanding: Persona-Grounded, Open-Ended Simulation of LLMs as Discharge Educators

    [https://arxiv.org/abs/2609.20827](https://arxiv.org/abs/2609.20827)

    该论文提出了DischargeBench，首个基于人设的开放式出院教育模拟基准，通过虚拟患者多轮对话与教育监控智能体，从对话质量、话题覆盖、理解程度和事实一致性四个维度评估大语言模型作为出院教育者促进患者理解的能力。

    

    医院出院教育是一项互动式教学任务：临床医生需要根据患者的文化水平、记忆能力和个性特点来调整出院计划。现有的大语言模型（LLM）评估主要针对静态任务或产物生成任务，无法衡量开放式对话情境下患者的理解程度。我们提出了DischargeBench，一个基于人设的模拟基准，其中候选LLM教育者与虚拟患者进行多轮对话，同时由一个教育监控智能体在不修改教育者行为的前提下调节虚拟患者的真实性，从而保护评估信号的有效性。我们构建了MIMIC-IV-Ext-DischargeBench数据集，包含覆盖24个ICD章节的477个病例，并设置人设维度（性格、教育水平、健康素养、既往病史记忆）用于分层分析。每次模拟由一个与医生标注对齐的LLM-as-a-Judge在四个维度上进行评分——对话质量、话题清单、理解程度和事实一致性。在闭源与开源模型上……（原文摘要在此处截断）

    arXiv:2609.20827v1 Announce Type: new  Abstract: Hospital discharge education is an interactive teaching task: a clinician adapts a discharge plan to a patient's literacy, recall, and personality. Existing LLM evaluations target static or artifact-generation tasks and do not measure patient understanding under open-ended dialogue. We introduce DischargeBench, a persona-grounded simulation in which a candidate LLM educator conducts a multi-turn session with a Virtual Patient, while an Education Monitor Agent regulates patient realism without modifying the educator, protecting the evaluation signal. We curate MIMIC-IV-Ext-DischargeBench, 477 cases over 24 ICD chapters with persona axes (personality, education level, health literacy, past-medical-history recall) for stratified analysis. Each simulation is scored on four axes -- Conversation Quality, Topic Checklist, Comprehension, and Factual Consistency -- by an LLM-as-a-Judge aligned against physician annotations. Across closed- and ope
    
[^220]: 骰子法：大语言模型品牌推荐重复查询审计的标准化协议

    The Dice Roll Method: A Standardized Protocol for Repeated-Query Auditing of Large Language Model Brand Recommendations

    [https://arxiv.org/abs/2609.04047](https://arxiv.org/abs/2609.04047)

    本文提出并形式化了“骰子法”——一个基于温度缩放核采样生成模型的可复用标准化协议，通过将响应方差分解为多个成分并提供完整的统计分析技术栈，为大语言模型品牌推荐的重复查询审计建立了系统方法。

    

    背景：研究人员越来越多地使用重复相同的提示词来审计大语言模型（LLM）品牌推荐中的随机变异，然而目前尚无用于设定迭代次数、选择稳定性指标或建立可靠性阈值的标准化协议。目标：我们将骰子法形式化为一个可复用的协议，用于LLM品牌推荐的重复查询审计，该协议建立在温度缩放核采样的生成模型基础之上。方法：将总响应方差分解为采样、提示措辞、运行间和模型版本等成分。该技术栈包括：以迭代作为重复测量的负二项混合模型；作为无分布效应量的Cliff's delta；保留依赖结构的自助法；基于模拟的统计功效分析；概化理论分解；以及对固定快照的漂移诊断。我们重新分析了五项品牌推荐审计研究：约190（摘要在此处截断）。

    arXiv:2609.04047v1 Announce Type: cross  Abstract: Background: Researchers increasingly use repeated identical prompts to audit stochastic variation in large language model (LLM) brand recommendations, yet no standardized protocol exists for setting iteration counts, selecting stability metrics, or establishing reliability thresholds. Objective: We formalize the Dice Roll Method as a reusable protocol for repeated-query auditing of LLM brand recommendations, grounded in a generative model of temperature-scaled nucleus sampling. Methods: Total response variance is decomposed into sampling, prompt-phrasing, run-to-run, and model-version components. The stack: a negative-binomial mixed model with iterations as repeated measures; Cliff's delta as the distribution-free effect size; dependence-preserving bootstrap; simulation-based power; a generalizability-theory decomposition; drift diagnostics on pinned snapshots. We reanalyse five brand-recommendation auditing studies: approximately 190,
    
[^221]: 惰性接地：用事实性证据攻击搜索代理

    Lazy Grounding: Attacking Search Agents with Factual Evidence

    [https://arxiv.org/abs/2608.30303](https://arxiv.org/abs/2608.30303)

    该论文提出“惰性接地”攻击：即使检索到的文档完全真实，只要其支持的是相邻改写问题的答案，搜索代理也会采用无法回答当前问题的相邻答案，导致准确率平均下降5.9个百分点、最高下降17.3个百分点。

    

    搜索代理通过将答案锚定在检索到的网络证据上来减少幻觉。然而，对检索的依赖也带来了攻击面：含有虚假或恶意文档的被投毒语料库可能导致代理传播错误信息。我们证明，虚假性并非必要条件——搜索代理可能被与相邻问题相关的事实性证据所误导，即使该相邻答案并不能回答当前问题，代理也会采用它。我们将这种失败称为“惰性接地”。我们利用基准问题经改变答案的改写所产生的相邻证据来揭示惰性接地现象：每份文档都真实地支持一个相邻的改写问题，但却被呈现在原始问题的检索结果中。在12个模型-基准组合上，相邻证据平均使准确率下降5.9个百分点，最高达17.3个百分点，并且在所有设置中都会诱导代理采用相邻答案。当相邻证据出现得越晚或越像答案时，这种效应越强。

    arXiv:2608.30303v1 Announce Type: new  Abstract: Search agents reduce hallucination by grounding answers in retrieved web evidence. Yet reliance on retrieval also creates an attack surface: poisoned corpora with false or malicious documents can cause agents to reproduce misinformation. We show that falsehood is not necessary -- a search agent can be misled by factual evidence for a nearby question, adopting that nearby answer even when it does not answer the current question. We call this failure lazy grounding. We expose lazy grounding using nearby evidence from answer-changing rewrites of benchmark questions. Each document truthfully supports a neighboring rewritten question, but is surfaced for the original question. Across 12 model-benchmark pairs, nearby evidence reduces accuracy by 5.9 points on average and by up to 17.3 points, while inducing nearby-answer adoption in every setting. The effect is stronger when nearby evidence appears later or is more answer-shaped. Our results s
    
[^222]: RATIO：科学文献中跨类型构思操作检索的基准

    RATIO: A Benchmark for Retrieval Across Typed Ideation Operations in Scientific Literature

    [https://arxiv.org/abs/2608.27394](https://arxiv.org/abs/2608.27394)

    RATIO基准首次定义了三种科学构思操作（Address、Broaden、Specify）的检索任务，并利用远距离监督扩展到大规模语料库，为科学文献的灵感检索提供了新范式。

    

    arXiv:2608.27394v1 公告类型：新 摘要：检索到的科学文献可以为人与AI科学家提供灵感。灵感可以采取不同形式：先前的工作可能直接建议如何解决问题，或在不同抽象层次上指出方向——放大到更一般的视角或缩小到具体实现。我们引入RATIO（跨类型构思操作检索），这是一个大规模基准，其中相关性由三种操作定义，我们称之为构思动作：Address检索针对所提出问题的潜在方法，Broaden检索更一般的表述，Specify检索具体实例。RATIO是通过一种通用方法从CS文献中数百万篇全文科学论文构建而成，该方法将话语标记远距离监督——先前仅用于分类——扩展到语料库级检索，并结合了广泛的LLM和人工审核。实验表明，操作-

    arXiv:2608.27394v1 Announce Type: new  Abstract: Retrieved scientific literature can serve as inspiration for both human and AI scientists. Inspiration can take different forms: prior work may directly suggest how to address a problem, or surface directions at different levels of abstraction - zooming out to a more general view or zooming in to a concrete realization. We introduce RATIO (Retrieval Across Typed Ideation Operations), a large-scale benchmark in which relevance is defined by three operations which we name ideation moves: Address retrieves potential approaches for stated problems, Broaden retrieves more general formulations, and Specify retrieves concrete instantiations. RATIO is constructed from millions of full-text scientific papers across CS literature via a general recipe that extends discourse-marker distant supervision - previously used only for classification - to corpus-scale retrieval, combined with extensive LLM and human vetting. Experiments show that operation-
    
[^223]: BTS-AgentBench：从只读遥测日志到代理基准测试的确定性、可重放流水线

    BTS-AgentBench: A Deterministic, Replayable Pipeline from Read-Only Telemetry Logs to Agent Benchmarks

    [https://arxiv.org/abs/2608.27334](https://arxiv.org/abs/2608.27334)

    该论文提出了一种从只读遥测日志到代理基准测试的确定性、可重放构建流水线，通过标准化工具存储和情节编译，实现了精确复现和零缺陷的基准测试生成。

    

    arXiv:2608.27334v1 公告类型：新 摘要：工业现场包含大量只读遥测数据，但很少有基准测试规定如何将这些记录编译为可执行的多轮代理任务。我们提出了一种遥测到情节的构建方法，并将其实例化为BTS-AgentBench。该流水线将BTS元数据和原始历史记录规范化为只读工具存储，编译具有工具派生黄金答案和证据的静态任务，并将保留的任务提升为类型化、有界、面向操作员的情节。发布的532行版本增加了澄清、目标修订、时间戳策略、质量门控报告和证据归因，同时保留了源计算和分割。编码契约预检报告零发现，构建排除控制器完成0/532行。两个独立的原始到情节构建匹配所有11个逻辑工具存储导出，并精确复现发布的356/87/89训练/开发/测试工件。将共享构建路径应用于XAI4HEAT。

    arXiv:2608.27334v1 Announce Type: new  Abstract: Industrial sites contain large volumes of read-only telemetry, but few benchmarks specify how to compile these records into executable multi-turn agent tasks. We present a telemetry-to-episode construction method instantiated as BTS-AgentBench. The pipeline normalizes BTS metadata and raw histories into a read-only tool store, compiles static tasks with tool-derived gold answers and evidence, and lifts retained tasks into typed, bounded operator-facing episodes. The 532-row release adds clarification, goal revision, timestamp policy, quality-gated reporting, and evidence attribution while preserving the source computation and split. Coded contract preflight reports zero findings, and the construction-exclusion controller completes 0/532 rows. Two independent raw-to-episode builds match all 11 logical tool-store exports and reproduce the released 356/87/89 train/dev/test artifact exactly. Applying the shared construction path to XAI4HEAT 
    
[^224]: TreeGraft：基于树的投机解码的自适应多草稿器嫁接方法

    TreeGraft: Adaptive Multi-Drafter Grafting for Tree-Based Speculative Decoding

    [https://arxiv.org/abs/2608.26112](https://arxiv.org/abs/2608.26112)

    我们提出了TreeGraft，一种自适应多草稿器嫁接框架，通过结合不同成本的草稿器来构建共享草稿树，从而平衡草稿质量与延迟，提升基于树的投机解码效率。

    

    投机解码通过“先草稿后验证”的范式加速大型语言模型的推理。在此基础上，树结构方法通过将提议组织成多个候选路径来提升推理效率，从而增加接受长度。然而，现有的树结构方法在所有草稿步骤中使用单一草稿器，这造成了一个困境：较小的草稿器速度快但生成的树质量较低，而较大的草稿器能提高树质量但延迟较高。为了解决这个问题，我们提出了TreeGraft，一个多草稿器框架，其中不同成本的草稿器共同构建一个共享的草稿树。TreeGraft使用更强的草稿器通过更新较弱草稿器分配的分数来重新评分候选，重新选择嫁接位置，并恢复未被探索的有前景路径。它还以非破坏性方式整合更强草稿器的扩展，保留可能仍被目标模型接受的现有分支。

    arXiv:2608.26112v1 Announce Type: new  Abstract: Speculative decoding accelerates large language model inference through a draft-then-verify paradigm. Building on this, tree-structured methods improve inference by organizing proposals into multiple candidate paths, increasing the accepted length. However, existing tree-structured methods use a single drafter for all drafting steps, creating a dilemma: a smaller drafter is fast but yields lower-quality trees, whereas a larger drafter improves tree quality but suffers from high latency. To address this, we propose TreeGraft, a multi-drafter framework in which drafters of different costs jointly construct a shared draft tree. TreeGraft uses the stronger drafter to rescore candidates by updating scores assigned by the weaker drafter, reselect grafting positions, and recover promising paths left unexplored. It also integrates stronger drafter expansions non-destructively, preserving existing branches that may still be accepted by the target
    
[^225]: DCGC：基于草稿条件的全局校正，用于带掩码扩散模型的复杂推理

    DCGC: Draft-Conditioned Global Correction for Complex Reasoning with Masked Diffusion Models

    [https://arxiv.org/abs/2608.25428](https://arxiv.org/abs/2608.25428)

    DCGC提出了一种基于草稿条件的掩码扩散模型框架，通过动态双CFG机制利用不完美草稿进行全局校正，显著提升了复杂推理任务的准确性。

    

    修正有缺陷的推理轨迹仍然是大型语言模型（LLMs）面临的一个重大挑战，其自回归生成可能将早期错误传播到后续推理中。我们引入了DCGC，一种用于全局校正的掩码扩散模型（MDM）框架，它利用来自上游求解器的不完美解决方案草稿作为辅助上下文。DCGC结合了任务特定的监督微调（SFT）和一种新颖的推理时机制，称为动态双CFG。该机制分离了仅问题分支和联合问题-草稿分支，并使用相对置信度差距来缩放草稿条件残差。在数学、代码和知识推理基准测试中，DCGC优于标准采样和更简单的CFG变体，额外结果表明其可迁移到不同的扩散骨干网络。在无法获得真实失败标签的测试时设置中，DCGC通过校正低共识样本来提高完整测试集的准确性。

    arXiv:2608.25428v1 Announce Type: new  Abstract: Correcting flawed reasoning traces remains a significant challenge for Large Language Models (LLMs), whose autoregressive generation can propagate early mistakes into subsequent reasoning. We introduce DCGC, a Masked Diffusion Model (MDM) framework for global correction that uses an imperfect solution draft from an upstream solver as auxiliary context. DCGC combines task-specific Supervised Fine-Tuning (SFT) with a novel inference-time mechanism called Dynamic Dual-CFG. This mechanism separates problem-only and joint problem-draft branches and scales the draft-conditioned residual using a relative confidence gap. Across math, code, and knowledge reasoning benchmarks, DCGC outperforms standard sampling and simpler CFG variants, with additional results suggesting transfer to different diffusion backbones. In test-time setting where ground-truth failure labels are unavailable, DCGC improves full test set accuracy by correcting low-consensus
    
[^226]: 面向可控数据生成的机制电路识别

    Mechanistic Circuit Identification for Controllable Data Generation

    [https://arxiv.org/abs/2608.24065](https://arxiv.org/abs/2608.24065)

    本文提出了一种基于电路识别的框架，通过连接数据估值与机制可解释性，实现可控数据生成，并引入SAMS调度器以优化效用目标数据产出。

    

    arXiv:2608.24065v1 公告类型：交叉 摘要：尽管近期数据合成方面的进展旨在策展高质量数据集，但大多数生成流程仍依赖于基于提示的启发式控制。这种黑箱范式对单个样本如何与模型底层学习动态相互作用提供的洞察有限。为弥合这一差距，我们提出了一种电路基础框架，将基于训练动态的数据估值与机制可解释性（MI）相结合。具体而言，我们将数据质量概念化为三个互补的效用轴：可学习性、挑战性和对齐性。首先，我们揭示出模型内部专门电路，这些电路因果地控制这些效用信号。然后，超越启发式提示转向机制控制，我们利用这些电路作为可控接口，主动引导生成以产出效用目标数据。基于此能力，我们引入了SAMS（阶段感知机制调度），该调度器调度电路...

    arXiv:2608.24065v1 Announce Type: cross  Abstract: While recent advances in data synthesis aim to curate high-quality datasets, most generation pipelines still rely on heuristic prompt-based control. This black-box paradigm provides limited insight into how individual samples interact with a model's underlying learning dynamics. To bridge this gap, we propose a circuit-grounded framework that connects training-dynamics-based data valuation with mechanistic interpretability (MI). Specifically, we conceptualize data quality along three complementary utility axes, learnability, challenge, and alignment. First, we uncover specialized model-internal circuits that causally govern these utility signals. Then, moving beyond heuristic prompting toward mechanistic control, we leverage these circuits as controllable interfaces, actively steering generation to produce utility-targeted data. Building on this capability, we introduce SAMS (Stage-Aware Mechanistic Scheduling), which schedules circuit
    
[^227]: 跨模型记忆迁移：通过目标端阅读器适配

    Cross-Model Memory Transfer via Target-Side Reader Adaptation

    [https://arxiv.org/abs/2608.17050](https://arxiv.org/abs/2608.17050)

    本文研究了跨模型记忆迁移中，冻结记忆与目标端阅读器相对重要性，发现轻量级阅读器适配是关键，而非记忆本身。

    

    arXiv:2608.17050v1 公告类型：交叉 摘要：改进大型语言模型中知识使用的方法通常分为两种模式。非参数检索提供对外部知识的灵活访问，但增加了检索延迟、上下文开销，并且与主干的集成较浅。参数化适配在推理时高效，但将知识与模型权重纠缠在一起，且难以更新、审计或迁移。Engram风格的哈希记忆占据了一个中间模式：它将学习到的信息存储在外部可寻址表中，但通过一个小型学习阅读器来消费该表。这引发了一个基本问题：当这种记忆跨骨干移动时，冻结的记忆本身更重要，还是目标端阅读器更重要？我们通过跨模型冻结记忆提取来研究这个问题，其中在源模型上训练的记忆被冻结并附加到不同的目标模型上，仅训练轻量级阅读器。消融实验表明...

    arXiv:2608.17050v1 Announce Type: cross  Abstract: Methods for improving knowledge use in large language models typically fall into two regimes. Non-parametric retrieval offers flexible access to external knowledge, but adds retrieval latency, context overhead, and only shallow integration with the backbone. Parametric adaptation is efficient at inference time, but entangles knowledge with model weights and can be hard to update, audit, or transfer. Engram-style hashed memory occupies a middle regime: it stores learned information in an external, addressable table, yet consumes that table through a small learned reader. This raises a basic question: when such a memory is moved across backbones, what matters more, the frozen memory itself or the target-side reader? We study this question through cross-model frozen-memory extraction, in which a memory trained on a source model is frozen and attached to a different target model, with only a lightweight reader trained. Ablations show that 
    
[^228]: 拒绝意图而非形式：基于包装的意图组监督用于大语言模型安全

    Refusing Intent, Not Form: Wrapper-Based Intent-Group Supervision for LLM Safety

    [https://arxiv.org/abs/2608.13304](https://arxiv.org/abs/2608.13304)

    本文提出WIFA方法，通过配对包装有害与良性反例来训练模型拒绝意图而非表面形式，显著提升有害拒绝能力并降低过度拒绝率。

    

    arXiv:2608.13304v1 公告类型：新论文  摘要：安全调优可以改善有害拒绝，但模型可能学习到表面形式的捷径：被包装的有害提示可以绕过安全机制，而同样包装的良性提示则被过度拒绝。我们提出了基于包装的意图-形式增强（WIFA），一种自动的意图组增强方法，它将包装的有害示例与结构匹配的包装良性反例配对，无需外部教师或手动逐包装意图标签。我们将WIFA作为两种互补微调路线的通用数据层：WIFA-Boost，一种两阶段高安全配方，以及锚定组一致性拒绝训练（A-GCRT），它正则化相同意图包装下的拒绝/遵从决策分数，并将有害和良性组锚定在间隔的相对两侧。在Qwen设置中，WIFA-Boost达到了最强的转换后有害拒绝性能，而A-GCRT将OR-Bench过度拒绝从基础模型的25.7%降低到17.4%；可复现。

    arXiv:2608.13304v1 Announce Type: new  Abstract: Safety tuning can improve harmful refusal, but models may learn surface-form shortcuts: wrapped harmful prompts bypass safety, while similarly wrapped benign prompts are over-refused. We propose Wrapper-Based Intent-Form Augmentation (WIFA), an automatic intent-group augmentation method that pairs wrapped harmful examples with structurally matched wrapped benign counterexamples, requiring no external teacher or manual per-wrapper intent labels. We use WIFA as a common data layer for two complementary fine-tuning routes: WIFA-Boost, a two-stage high-safety recipe, and Anchored Group-Consistent Refusal Training (A-GCRT), which regularizes refusal/compliance decision scores across same-intent wrappers and anchors harmful and benign groups on opposite sides of a margin. In the Qwen setting, WIFA-Boost reaches the strongest transformed-harmful refusal, while A-GCRT reduces OR-Bench over-refusal from 25.7\% for the base model to 17.4\%; reprod
    
[^229]: LycheeMemory V2：通过语义段级整合实现LLM代理的高效长期记忆

    LycheeMemory V2: Efficient Long-Term Memory for LLM Agents via Semantic Segment-Level Consolidation

    [https://arxiv.org/abs/2608.12990](https://arxiv.org/abs/2608.12990)

    LycheeMemory V2通过语义段级整合替代回合级整合，显著降低LLM编码频率，在保持记忆细节的同时提升长期记忆构建效率。

    

    arXiv:2608.12990v1 公告类型：新公告  摘要：长时程LLM代理必须保留过去交互中的信息以支持未来任务。现有记忆系统通常依赖急切整合，在每次交互后调用LLM来提取、总结或更新记忆。这种设计使得随着对话增长，记忆构建成本日益增加。粗粒度总结可以降低构建成本，但可能丢失细粒度的上下文证据，而更大的检索上下文或多跳LLM推理则将开销转移到查询时间。我们提出LycheeMemory V2，一种高效长期记忆框架，用语义段级整合取代回合级整合。LycheeMemory不整合每次交互，而是将多个交换批量处理为段，并将每个最终确定的段编码为上下文无关的类型化记忆记录。段级批处理降低了LLM编码频率，而语义边界检测有助于预选...

    arXiv:2608.12990v1 Announce Type: new  Abstract: Long-horizon LLM agents must preserve information from past interactions to support future tasks. Existing memory systems typically rely on eager consolidation, invoking LLMs after each interaction to extract, summarize, or update memories. This design makes memory construction increasingly costly as conversations grow. Coarse summarization can reduce construction cost but risks discarding fine-grained contextual evidence, whereas larger retrieval contexts or multi-hop LLM reasoning shift the overhead to query time. We present LycheeMemory V2, an efficient long-term memory framework that replaces turn-level consolidation with semantic segment-level consolidation. Instead of consolidating every interaction, LycheeMemory batches multiple exchanges into segments and encodes each finalized segment into context-independent typed memory records. Segment-level batching lowers LLM encoding frequency, while semantic boundary detection helps prese
    
[^230]: AVA-编码器：迈向智能体原生的视频表示学习

    AVA-Encoder: Towards Agent-Native Video Representation Learning

    [https://arxiv.org/abs/2608.12313](https://arxiv.org/abs/2608.12313)

    本文提出AVA-Encoder框架，通过智能体自编码将视频转换为知识图谱表示并重建，实现了智能体可直接理解、查询和编辑的视频表示学习，克服了创意智能体从电影中学习的障碍。

    

    摘要：arXiv:2608.12313v1 公告类型：交叉 摘要：创意智能体仍缺乏从高质量人类电影中学习的有效方式，这限制了它们制作电影级视频的能力。一个关键挑战是缺乏一种既忠实于电影内容又可直接用于智能体推理和操作的视频结构化表示。为应对这一挑战，我们提出了智能体视频自编码器（AVA-Encoder），一种通过智能体自编码学习智能体原生视频表示的框架。AVA-Encoder 将视频转换为知识图谱表示，然后将其重建回视频。其层次结构和状态节点存储结构化文本，而关联的资源层则保存生成的图像、音频和视频。类型化边以智能体易于理解、查询和编辑的形式保留了这些文本描述与资源之间的关系。视频重建差异驱动了一个文本梯度优化框架，该框架...

    arXiv:2608.12313v1 Announce Type: cross  Abstract: Creative agents still lack an effective way to learn from high-quality human films, limiting their ability to produce cinematic-grade videos. A key challenge is the absence of a structured video representation that is both faithful to film content and directly usable for agentic reasoning and manipulation. To address the challenge, we propose the Agentic Video Auto-Encoder (AVA-Encoder), a framework for learning agent-native video representations via agentic auto-encoding.   AVA-Encoder transforms a video into a knowledge graph (KG) representation and then reconstructs it back into video. Its hierarchy and state nodes store structured text, while a linked asset layer holds generated images, audio, and video. Typed edges preserve the relations between these text descriptions and assets in a form that agents can easily understand, query, and edit. The video reconstruction differences drive a textual-gradient optimization framework, which
    
[^231]: 混合门控注意力

    Hybrid Gated Attention

    [https://arxiv.org/abs/2608.11805](https://arxiv.org/abs/2608.11805)

    本文提出混合门控注意力框架，通过三种门控策略融合多头信息，结合低秩分解和可学习汇聚，显著提升注意力表示能力与训练效率。

    

    摘要：arXiv:2608.11805v1 公告类型：新 摘要：门控注意力是缓解注意力汇聚和增强注意力表示能力的有效方法。为了进一步扩展其效率-效果帕累托前沿，我们提出了一种混合门控注意力（HyGA）框架，该框架包含三种类型的门控策略。具体来说，这些门控利用注意力多个阶段中的多样信息，并从多个角度协同构建元素级/头级门控，捕捉头内和头间的信息交互。通过我们的混合门控组件，HyGA能够提供多源调制信号，从而实现对信息流的更全面控制，并提高注意力的表示能力。我们还引入了低秩矩阵分解和可学习注意力汇聚，以进一步增强训练效率和稳定性。在实验中，我们基于不同骨干网络在广泛使用的基准上评估了HyGA。

    arXiv:2608.11805v1 Announce Type: new  Abstract: Gated attention is an effective approach to mitigate attention sinks and enhance the representational capacity of attention. To further extend its effectiveness-efficiency Pareto frontier, we propose a Hybrid Gated Attention (HyGA) framework that contains three types of gating strategies. Specifically, these gates leverage diverse information from multiple stages of attention, and collaboratively build element-wise/head-wise gating from multiple perspectives, capturing intra-head and cross-head information interactions. Through our hybrid gating components, HyGA could provide multi-source modulation signals, enabling more comprehensive control over information flow and improving the representational capacity of attention. We also introduce low-rank matrix decomposition and learnable attention sink to further enhance training efficiency and stability. In experiments, we evaluate HyGA on widely-used benchmarks based on different backbones.
    
[^232]: 橡皮筋丢弃法：一种缓解基于评分标准奖励强化学习中奖励黑客问题的简单方法

    Rubric Dropout: A Simple Way to Mitigate Reward Hacking in Rubric-as-Reward RL

    [https://arxiv.org/abs/2608.11669](https://arxiv.org/abs/2608.11669)

    本文提出一种简单方法（Rubric Dropout）来缓解基于评分标准的强化学习中的奖励黑客问题，并通过实验证实训练评判员与黄金评判员分数在训练中分化的现象，为提升模型鲁棒性提供新思路。

    

    摘要：基于评分标准的强化学习，即由LLM评判员根据一系列标准进行评分，已成为对没有确定性答案的任务进行语言模型后训练的标准方法。然而，评分标准是质量的固定代理，永远无法完全描述质量，长期针对其训练的策略会学会利用这一差异。我们直接对此进行了测量。使用组相对策略优化（GRPO）在医学和科学评分标准上训练Qwen3-8B，并使用训练评判员和更强的黄金评判员对分布外（OOD）基准进行评分，我们发现这两个分数在训练过程中出现分歧。训练评判员的分数持续上升，而黄金评判员的分数达到峰值后下降，在HealthBench-Hard上下降3分，在ResearchQA上下降22分。如果评判员存在固定偏差，黄金曲线只会发生恒定偏移，而不会在训练分数上升时下降，因此这种分歧是奖励黑客问题，而非评判员偏差。

    arXiv:2608.11669v1 Announce Type: cross  Abstract: Reinforcement learning against rubrics, lists of criteria graded by an LLM judge, has become a standard way to post-train language models on tasks with no deterministic answer. The rubric, however, is a fixed proxy for quality, never a complete description of it, and a policy trained against it long enough will learn to exploit the difference. We measure this directly. Training Qwen3-8B with Group Relative Policy Optimization (GRPO) on medical and science rubrics and grading out-of-distribution (OOD) benchmarks with both the training judge and a stronger gold judge, we find that the two scores diverge during training. The training judge's score keeps climbing while the gold judge's score peaks and then falls, by 3 points on HealthBench-Hard and by 22 points on ResearchQA. A judge with a fixed bias would shift the gold curve by a constant, not send it down while the training score rises, so the divergence is reward hacking, not judge no
    
[^233]: PERCEPT：一个用于波斯语-英语混合语码词性标注与分析的语料库

    PERCEPT: A Corpus for POS Tagging and Analysis of Persian-English Code-Mixing

    [https://arxiv.org/abs/2608.10109](https://arxiv.org/abs/2608.10109)

    本文推出了PERCEPT——首个公开可用的大规模波斯语-英语混合语码语料库，包含6,800条社交媒体帖子并为混合语码词汇提供通用依存词性标注，同时提出了LLM辅助的自动标注框架。

    

    社交媒体已成为多语言交流的重要场所，用户经常在单句话语中混合使用多种语言。尽管目前已有针对若干语言对开发的混合语码语料库，但波斯语-英语混合语码仍然相对缺乏研究。现有的波斯语资源缺乏针对混合语码词汇的通用依存分析（UD）词性（POS）标注，这既限制了语言学分析，也限制了句法感知自然语言处理模型的开发。为填补这一空白，我们推出了PERCEPT——首个公开可用的大规模波斯语-英语混合语码语料库，为其中的混合语码词汇标注了通用依存分析词性标签。该数据集包含从Twitter（X）、Instagram和Digikala收集的6,800条帖子，每条帖子至少包含一个已识别的波斯语-英语混合语码词汇。我们还进一步提出了一个LLM辅助的标注框架，可自动分配词性标签和文档级主题。

    arXiv:2608.10109v2 Announce Type: replace  Abstract: Social media has become a major venue for multilingual communication, where users frequently mix multiple languages within a single utterance. Although code-mixed corpora have been developed for several language pairs, Persian-English code-mixing remains relatively underexplored. Existing Persian resources lack Universal Dependencies (UD) part-of-speech (POS) annotations for code-mixed words, limiting both linguistic analyses and the development of syntax-aware NLP models. To address this gap, we introduce PERCEPT, the first publicly available large-scale Persian-English code-mixed corpus annotated with Universal Dependencies POS tags for code-mixed words. The dataset comprises 6,800 posts collected from Twitter (X), Instagram, and Digikala, each containing at least one identified Persian-English code-mixed word. We further present an LLM-assisted annotation framework that automatically assigns POS tags and document-level topics. Hum
    
[^234]: 现成的语言-视觉编码器可预测视觉世界实验中的人类注视偏好

    Off-the-Shelf Language-Vision Encoders Predict Human Gaze Preferences in Visual World Experiments

    [https://arxiv.org/abs/2608.07282](https://arxiv.org/abs/2608.07282)

    该论文提出一种新方法，通过对现成的双模态CLIP双编码器应用归因方法来预测视觉世界实验中的人类注视偏好，无需生成式架构和微调即可定性复现人类预测性语言加工的注视模式。

    

    计算心理语言学的一个核心问题是，神经语言模型在多大程度上能够作为人类语言加工的认知充分模型。然而，现有研究绝大多数集中于书面或口语语言的单模态情形，而多模态范式——例如同时向被试呈现视觉与语言输入的视觉世界实验——一直被忽视。在本文中，我们提出了一种新颖的方法来预测视觉世界研究中的注视偏好模式。该方法通过对一个简单的双模态CLIP双编码器应用归因方法来推导目标偏好。我们证明了该方法能够在定性层面复现一项开创性的英语视觉世界研究中展示人类预测性加工的注视模式。尽管并未针对该任务进行训练，它无需生成式架构，也无需微调即可实现这一效果。

    arXiv:2608.07282v2 Announce Type: replace  Abstract: A central question for computational psycholinguistics is to what extent neural language models can serve as cognitively adequate models of human language processing. However, work has been overwhelmingly focused on the unimodal case of written or spoken language. In contrast, multimodal paradigms, like visual world experiments that present participants with both visual and linguistic input simultaneously, have been neglected. In this paper, we present a novel approach that predicts patterns of gaze preferences in visual world studies. To do so, it derives target preferences by applying an attribution method to a simple bi-modal CLIP bi-encoder. We demonstrate the ability of this approach to qualitatively replicate the gaze patterns found in a seminal English visual world study demonstrating human predictive processing. It does so without a generative architecture and without the need for fine-tuning, despite not being trained for th
    
[^235]: SPARK：从推测性GRPO采样中的验证学习

    SPARK: Learning from Verification in Speculative GRPO Rollouts

    [https://arxiv.org/abs/2608.04962](https://arxiv.org/abs/2608.04962)

    提出SPARK，一种轻量级自适应机制，利用目标模型的验证反馈在推测性GRPO训练的采样过程中实时修正草稿模型的预测遗漏，从而提升训练效率。

    

    组相对策略优化（GRPO）已成为提升大语言模型推理能力的标准方法，但其训练成本高昂，因为每一步都需要生成大量冗长的推理响应。近期方法采用推测解码来加速采样过程，即由草稿模型提出候选token，再由目标模型进行验证。然而，我们发现草稿模型在单个响应内以及同一GRPO组内的多个响应之间会反复遗漏相同的token，而这些遗漏处的验证反馈并未用于在采样过程中更新草稿模型。基于这一分析，我们提出了SPARK，一种轻量级的自适应机制，旨在增强推测性GRPO训练中使用的草稿模型。SPARK利用目标模型的验证反馈在生成过程中修正草稿模型的预测，同时保留其原有的训练流程。在21个模型-数据集设置上的平均结果显示……

    arXiv:2608.04962v2 Announce Type: replace-cross  Abstract: Group Relative Policy Optimization (GRPO) has become a standard approach for improving the reasoning abilities of large language models, but its training is costly because each step requires generating many long reasoning responses. Recent methods use speculative decoding to accelerate the rollouts, where a draft model proposes tokens that the target model verifies. However, we find that the draft model repeatedly misses the same tokens within a response and across responses in the same GRPO group, and the verification feedback on these misses is not used to update the draft during the rollout. Motivated by this analysis, we introduce SPARK, a lightweight self-adaptive mechanism designed to enhance draft models used in speculative GRPO training. SPARK leverages target verification feedback to refine draft predictions during generation while preserving their existing training procedures. Averaged across 21 model-dataset settings
    
[^236]: ANCHOR-RE：一种结合检索增强与错误驱动验证的智能体化生物医学关系抽取框架

    ANCHOR-RE: An Agentic Biomedical Relation Extraction Framework with Retrieval Augmentation and Error-Driven Verification

    [https://arxiv.org/abs/2608.03154](https://arxiv.org/abs/2608.03154)

    ANCHOR-RE是一种融合检索增强示例、本体引导语义约束与错误驱动验证的智能体化框架，无需任务特定微调即可显著提升生物医学关系抽取的可靠性与微平均F1分数。

    

    生物医学关系抽取（BioRE）支持知识图谱构建与假设生成。在真实场景中，大多数候选实体对之间并无关联。以SemRep为代表的符号系统具有高精确率和可解释性，但召回率有限；而大语言模型（LLM）则可能产生假阳性关系。我们开发了ANCHOR-RE，这是一个智能体化框架，它将检索到的示例、本体引导的语义约束、外部生物医学知识与错误驱动验证相结合，从而在无需任务特定微调的情况下提高生物医学关系抽取的可靠性。我们在SemRepGS、DDI和ChemProt数据集上，使用专有及开放权重大语言模型对ANCHOR-RE进行了评估。为了在减少潜在预训练数据污染的同时评估其泛化能力，我们还使用2026年发表的100篇生物医学文章进行了截止日期之后的时间性精确率评估。使用专有模型作为骨干时，ANCHOR-RE相较直接提示方法提升了微平均F1……（摘要在此处截断）

    arXiv:2608.03154v2 Announce Type: replace  Abstract: Biomedical relation extraction (BioRE) supports knowledge graph construction and hypothesis generation. In real-world settings, most candidate entity pairs are unrelated. Symbolic systems such as SemRep offer high precision and interpretability but limited recall, whereas large language models (LLMs) can produce false-positive relations. We developed ANCHOR-RE, an agentic framework that combines retrieved demonstrations, ontology-guided semantic constraints, external biomedical knowledge, and error-driven verification to improve BioRE reliability without task-specific fine-tuning. We evaluated ANCHOR-RE on SemRepGS, DDI, and ChemProt using proprietary and open-weight LLMs. To assess generalizability while reducing potential pretraining contamination, we also conducted a post-cutoff temporal precision evaluation using 100 biomedical articles published in 2026. With the proprietary backbone, ANCHOR-RE improved micro-F1 over direct prom
    
[^237]: IACM-RL：面向动态意图波动下复杂工具调用的意图感知上下文管理与强化学习

    IACM-RL: Intent-Aware Context Management and Reinforcement Learning for Complex Tool Invocation under Dynamic Intent Fluctuations

    [https://arxiv.org/abs/2608.02110](https://arxiv.org/abs/2608.02110)

    提出IACM-RL框架，通过基于BeliefState的自生成上下文管理器主动追踪用户意图变化并隔离过时参数，结合DynamicIntent数据管线与强化学习优化，解决了动态意图波动下长程工具调用中的意图偏离与重复API调用问题。

    

    在真实环境中执行长程工具调用严重受到动态用户意图噪声的挑战。现有方法试图通过隐式历史扫描或文本压缩来获得鲁棒性，但在简单化场景中大多假设指令是完美无缺的。不可避免地，在上下文波动的情况下，过时的约束会稀释模型注意力，引发意图偏离和重复的API循环。为解决这一问题，我们提出IACM-RL，一个面向鲁棒工具调用的综合性框架。首先，我们引入DynamicIntent管线，跨多个细粒度波动场景合成轨迹，并配备五维诊断指标套件。其次，IACM-RL部署了基于BeliefState的自生成上下文管理器，主动追踪不断变化的目标，并使用结构性过期标志隔离被覆盖的参数。为了自主内化这种状态追踪能力，我们对策略进行优化（原文此处截断）。

    arXiv:2608.02110v2 Announce Type: replace  Abstract: Executing long-horizon tool invocations in real-world environments is severely challenged by dynamic user intent noise. Existing methods attempt robustness via implicit history scanning or text compression, yet predominantly assume perfect instructions in simplistic scenarios. Inevitably, under fluctuating contexts, obsolete constraints dilute model attention, triggering intent deviation and repetitive API loops. To resolve this, we propose IACM-RL, a comprehensive framework for robust tool invocation. First, we introduce the DynamicIntent pipeline, synthesizing trajectories across several fine-grained fluctuation scenarios, paired with a five-dimensional diagnostic metric suite. Second, IACM-RL deploys a BeliefState-based Self-Generated Context Manager that proactively tracks shifting goals and isolates overwritten parameters using structural stale flags. To autonomously internalize this state-tracking capability, we optimize the po
    
[^238]: 本地提交，全局退出：协调扩散语言模型中的自适应采样与提前退出

    Commit Locally, Exit Globally: Coordinating Adaptive Sampling and Early Exit in Diffusion Language Models

    [https://arxiv.org/abs/2607.28166](https://arxiv.org/abs/2607.28166)

    提出了一种协调局部块加速与全局序列终止的双门机制，通过基于候选答案的置信度验证实现更高效的扩散语言模型加速。

    

    arXiv:2607.28166v2 公告类型：替换-交叉 摘要：扩散语言模型在每个去噪步骤中都会暴露一个临时预测，在许多任务中，其内部候选答案在步骤计划耗尽之前就已稳定。这创造了两个加速机会：提前离开块和提前停止序列，但两者需要不同的标准，因为块加速是局部的，而序列终止是全局的并冻结了分级答案。现有方法通常只优化一个轴，且现有退出门依赖于固定区域置信度或计划相关规则，而非候选答案本身。我们提出$\textbf{C}^4$，通过为每个决策分配独立的门来协调这两个轴。$\textbf{C}$onfidence-Verified Early Exit (CVEE) 决定序列何时可以停止，要求在每个步骤重新提取的候选跨度上具有置信度和持续的argmax稳定性。$\textbf{C}$ommit-$\textbf{C}$ore-Then-$\textbf{C}$onfirm 则处理局部块加速。

    arXiv:2607.28166v2 Announce Type: replace-cross  Abstract: Diffusion language models expose a provisional prediction at every denoising step, and on many tasks the candidate answer inside it stabilizes before the step schedule is exhausted. This creates two acceleration opportunities, leaving a block early and stopping the sequence early, but the two require different criteria because block acceleration is local whereas sequence termination is global and freezes the graded answer. Existing methods usually optimize only one axis, and existing exit gates rely on fixed-region confidence or schedule-dependent rules rather than the candidate answer itself. We present $\textbf{C}^4$, which coordinates the two axes by giving each decision its own gate. $\textbf{C}$onfidence-Verified Early Exit (CVEE) decides when the sequence may stop, requiring confidence and sustained argmax stability over a candidate span re-extracted at every step. $\textbf{C}$ommit-$\textbf{C}$ore-Then-$\textbf{C}$onfirm
    
[^239]: 每当我雇一位语言学家，推理成本就下降：论语言学规则作为高效提示压缩器

    Every Time I Hire a Linguist, Inference Costs Go Down: On Linguistic Rules as Effective Prompt Compressors

    [https://arxiv.org/abs/2607.25335](https://arxiv.org/abs/2607.25335)

    该论文提出仅用语言学规则（词汇、句法、语义、语篇）作为提示压缩器，通过离线进化搜索获得规则组合，部署时无需语言模型前向传播、仅需CPU端处理即可有效降低LLM推理成本。

    

    提示压缩通过缩短大语言模型（LLM）的输入来降低推理成本，然而现有方法都依赖语言模型前向传播来评估词元重要性。这种精细而昂贵的词元选择是否必要仍值得质疑。压缩需要识别有信息量的内容，而语言学研究早已通过可操作化为确定性规则的线索来解决这个问题。因此我们提出这样的问题：仅凭语言学规则能否作为有效的提示压缩器，而无需在压缩时进行基于语言模型的评分？为解答这一问题，我们对词汇、句法、语义和语篇层面的种子规则进行离线进化搜索，以找到具有竞争力的规则组合。由此得到的语言学压缩器在部署时无需任何语言模型前向传播，仅靠CPU端处理即可完成压缩。我们采用双路径评估协议来平衡压缩质量与重构保真度。在短文本、多文档等

    arXiv:2607.25335v3 Announce Type: replace  Abstract: Prompt compression shortens LLM input to reduce inference cost, yet existing methods score token importance through LM forward passes. It remains questionable whether such nuanced, costly token selection is necessary. Compression requires identifying informative content, a problem that linguistic research has long addressed through cues that can be operationalized as deterministic rules. We therefore ask: can \textbf{linguistic rules alone} serve as effective prompt compressors, without LM-based scoring at compression time?   To address this, we conduct offline evolutionary search over lexical, syntactic, semantic, and discourse seeds to find competitive rule combinations. The resulting linguistic compressor requires no LM forward pass at deployment and uses only CPU-side processing for compression. We evaluate it with a dual-path protocol to balance compression quality and reconstruction fidelity.   Across short passages, multi-docu
    
[^240]: 自信却错误：从大语言模型内部状态检测金融问答中的幻觉

    Confidently Wrong: Detecting Hallucinations in Financial Question Answering from LLM Internal States

    [https://arxiv.org/abs/2607.11414](https://arxiv.org/abs/2607.11414)

    该论文提出在大语言模型残差流上训练线性探针，利用模型内部激活来检测金融问答中“自信却错误”的幻觉回答，并证明内部状态包含超越输出层面置信度信号（如token对数概率和模型自我真/假评估）的判别信息。

    

    大语言模型（LLM）在金融应用中，最严重的失败情形是“自信却错误”。含糊、不确定的回答会引起人们的审视，而自信的错误却会在毫无预警的情况下悄然损害下游决策。我们探讨如何从模型的内部激活中可靠地检测出这类自信却错误的回答（即“自信幻觉”），以及这些激活是否包含超出模型输出层面置信信号之外的信息。我们在残差流上训练线性探针，并在两个基于真实财报构建的成熟数值问答基准——FinQA和TAT-QA上对其进行评估。行为置信度通过同一问题的八个重采样回答之间的一致性来衡量，并将探针的有效性与基线方法进行比较，例如token对数概率以及模型对自身答案的真/假自评。我们的研究结果表明，在自信的回答中，那些……（原文摘要在此处截断）

    arXiv:2607.11414v2 Announce Type: replace  Abstract: Large language models (LLMs) in financial applications fail most consequentially when they are confidently wrong. Hedged, uncertain answers invite scrutiny, whereas confident errors silently degrade downstream decisions without warning. We ask how reliably such confidently wrong answers, or confident hallucinations, can be detected from a model's internal activations, and whether those activations carry information beyond its output-level confidence signals. We train linear probes on the residual stream and evaluate them on two established numerical question-answering (QA) benchmarks built from real filings, FinQA and TAT-QA. Behavioral confidence is measured as the agreement among eight resampled answers to the same question, and probe effectiveness is compared against baselines, such as token log-probabilities and the model's own True/False self-assessment of its answer. Our findings show that among confident answers, those for whi
    
[^241]: CausalDS：数据科学智能体中因果推理的基准测试

    CausalDS: Benchmarking Causal Reasoning in Data-Science Agents

    [https://arxiv.org/abs/2607.08093](https://arxiv.org/abs/2607.08093)

    该论文提出CausalDS基准，通过系统性地生成结构因果模型、对应的观测数据以及基于真实领域的自然语言场景，填补了现有基准在评估数据科学智能体因果推理能力方面的空白。

    

    大型语言模型日益扮演数据科学智能体的角色，将抽象推理与先进的工具使用相结合。然而，相关基准测试领域大致分为两类：一类是缺乏真实数据分析的符号化因果推理基准，另一类是缺乏因果数据生成结构及一系列因果数据科学任务的数据分析基准。现有的因果评估数据集通常局限于来自现有来源的精选示例，其多样性源于有限的模板化变体，而非系统性地生成新颖的合成因果结构。我们提出了CausalDS，一个用于评估智能体化数据科学工作流中因果推理能力的基准。每个基准实例是一个场景，由一个采样得到的结构因果模型（SCM）及其生成的观测数据，以及一个基于真实领域的配套合成自然语言故事组成。（注：原文摘要在此处被截断）

    arXiv:2607.08093v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) increasingly act as data-science agents, combining abstract reasoning with advanced tool use. Yet the relevant benchmark landscape largely divides into symbolic causal reasoning benchmarks without realistic data analysis or data analysis benchmarks without causal data-generating structure or a range of causal data science tasks. Existing causal evaluation datasets are often restricted to curated examples from existing sources, with diversity coming from limited templatized variations rather than from systematic generation of novel synthetic causal structures. We introduce CausalDS, a benchmark for evaluating causal reasoning in agentic data-science workflows. Each benchmark instance is a scene consisting of a sampled structural causal model (SCM) with generated observational data and an accompanying synthetic natural-language story grounded in a real domain. We ground the composition of the benchmar
    
[^242]: 离散扩散模型学到了什么？三种坐标系下的同一个逆向过程

    What Does a Discrete Diffusion Model Learn? One Reverse Process in Three Coordinates

    [https://arxiv.org/abs/2607.05381](https://arxiv.org/abs/2607.05381)

    本文证明离散扩散模型中的去噪器、桥接插件和具体分数三种方法学习的其实是同一对象——加噪过程的精确时间反演，只是处于三种不同坐标系，并给出 Oracle Distance 恒等式（负 ELBO 等于数据熵加路径 KL 散度）及三种坐标间的精确转换。

    

    离散扩散模型究竟学到了什么？现有文献分别训练去噪器、桥接插件和具体分数，每种方法都在不同的约定下推导出自己的 ELBO 目标。这种碎片化使得所学习的对象不清晰，ELBO 数值也难以解释和比较。本文指出，这三种方法在不同的坐标系中学习的是同一个对象：加噪过程的精确时间反演。我们首先证明了 Oracle Distance 恒等式：对于任意加噪过程，负 ELBO 不仅仅是似然的一个界；它等于数据熵加上从反向加噪过程到所学过程的路径 KL 散度。接下来，针对令牌因子化的噪声，我们证明去噪器、桥接插件和具体分数是该逆向过程的三种坐标，推导出它们之间的精确转换关系，并确定了每种方法所学习的概率分布。特别是，与其标准解释相反……（摘要原文在此处被截断）

    arXiv:2607.05381v2 Announce Type: replace-cross  Abstract: What does a discrete diffusion model actually learn? The literature trains denoisers, bridge plug-ins, and concrete scores, each deriving its own ELBO objective under different conventions. This fragmentation makes the object being learned unclear and ELBO values difficult to interpret and compare. All three methods learn the same object in different coordinates: the exact time-reversal of the noising process. We first prove the Oracle Distance identity: for any noising process, the negative ELBO is not just a bound on the likelihood; it equals the data entropy plus the path KL divergence from the reversed noising process to the learned one. Next, for token-factorizing noise, we show that denoiser, bridge plug-in, and concrete score are three coordinates of this reverse process, derive exact conversions between them, and identify the probability distribution learned by each. In particular, contrary to its standard interpretatio
    
[^243]: 从大型推理模型到紧凑学生模型的知识蒸馏：以John O'Bryan数学竞赛为例的案例研究

    Knowledge Distillation from Large Reasoning Models to Compact Student Models: A Case Study on the John O Bryan Mathematics Competition

    [https://arxiv.org/abs/2606.31048](https://arxiv.org/abs/2606.31048)

    本文通过双智能体框架构建思维链语料，在Apple Silicon上利用LoRA将DeepSeek-R1的推理能力蒸馏至Qwen2.5-7B，并借助早停策略控制过拟合，从而提升紧凑模型在数学竞赛题上的表现。

    

    本文研究了从大型推理模型到紧凑学生模型的知识蒸馏。利用北肯塔基大学John O'Bryan数学竞赛的历史题目（2011-2025年），我们通过双智能体框架构建了思维链训练语料库。该数据集被用于在Apple Silicon硬件上使用MLX框架，通过低秩适配对）学生模型进行微调。基础Qwen2.5-7B模型在竞赛题目上达到64.67%的准确率，而DeepSeek-R1教师模型达到91.40%。初始的1000次迭代训练运行显示出严重的过拟合现象，验证损失在第200次迭代时达到最低点，随后持续上升。基于这一发现，我们进行了五次独立的训练运行，每次限制在200次迭代并使用不同的随机种子，以评估结果的稳定性。在这五次运行中，微调后的学生模型达到了平均……（原文摘要在此处截断）

    arXiv:2606.31048v1 Announce Type: cross  Abstract: This paper investigates knowledge distillation from a large reasoning model (DeepSeek-R1) to a compact student model (Qwen2.5-7B). Using historical problems from the John O'Bryan Mathematics Competition at Northern Kentucky University (2011-2025), we build a Chain-of-Thought (CoT) training corpus through a dual-agent framework. The dataset is used to fine-tune the student model with Low-Rank Adaptation (LoRA) on Apple Silicon hardware using the MLX framework. The base Qwen2.5-7B model achieves 64.67% accuracy on competition problems, while the DeepSeek-R1 teacher achieves 91.40%. An initial 1,000-iteration training run revealed severe overfitting, with validation loss reaching a minimum at iteration 200 before rising steadily. Based on this finding, we ran five independent training runs each limited to 200 iterations with varied random seeds to assess result stability. Across these five runs, the fine-tuned student model achieves a mea
    
[^244]: 固定系统与递归自我改进系统的安全性的算法不可验证性

    Algorithmic Unverifiability of Safety for Fixed and Recursively Self-Improving Systems

    [https://arxiv.org/abs/2606.28639](https://arxiv.org/abs/2606.28639)

    该论文从数学上严格证明了对于图灵完备的自修改系统（包括递归自我改进系统），安全验证在静态和动态两个层面都存在不可逾越的极限——不存在任何既可靠、完备又可行的安全验证器，从而为AI自我改进的安全性验证划定了根本性的理论边界。

    

    我们为图灵完备的自修改系统——即递归自我改进发生的这类系统——建立了算法安全验证的数学极限，既针对固定系统，也针对系统对其自身的修改。从静态角度看，不存在任何既可靠、完备又可行的验证器：在无界域上由Rice定理和哥德尔定理证明，在所有有限配置上由Trakhtenbrot定理证明，而在简洁描述的有限环境中，验证一个策略以对抗对手是coNP完全问题，合成这样一个策略则是PSPACE完全问题。从动态角度看，我们将一步自我修改建模为代码的可计算变换，并询问安全性质能否在该变换后得以保持。如果该变换仅依赖于行为，这相当于“上一层的Rice定理”；如果它读取代码——正如自我修改所做的——该问题不再是语义性的，但同样的s-m-n归约在行为等价的类中仍然成立。

    arXiv:2606.28639v3 Announce Type: replace-cross  Abstract: We establish mathematical limits of algorithmic safety verification for Turing-complete self-modifying systems, the class in which recursive self-improvement takes place, both for a fixed system and across its own modification. Statically, no verifier is sound, complete and tractable: over unbounded domains by Rice's and G\"odel's theorems, over all finite configurations by Trakhtenbrot's theorem, and over succinctly described finite environments because verifying a policy against an adversary is coNP-complete and synthesising one is PSPACE-complete. Dynamically, we model one step of self-modification as a computable transformation of code and ask whether a safety property survives it. If the transformation depends only on behaviour, this is Rice's theorem one level up; if it reads the code, as self-modification does, the question is no longer semantic, yet the same s-m-n reduction works inside a class of behaviourally identica
    
[^245]: 从点击到意图：基于大语言模型提炼分类法的跨平台会话嵌入在金融服务推荐中的应用

    From Clicks to Intent: Cross-Platform Session Embeddings with LLM-Distilled Taxonomy for Financial Services Recommendations

    [https://arxiv.org/abs/2606.26277](https://arxiv.org/abs/2606.26277)

    本文提出了一种利用大语言模型提炼的分类法来生成跨平台会话嵌入，从而在金融服务推荐中实现从网页点击到用户意图的预测，并支持定量推荐与定性理解的双重目标。

    

    顺序用户行为建模在工业推荐系统中被广泛采用；然而，在金融服务领域仍存在显著差距，因为登录前的网页交互与认证后的应用内体验截然不同。具体而言，登录前的网页用户通常探索新产品，而登录后的应用用户则专注于账户服务。由于跨渠道实体解析的挑战（例如，将匿名网页会话匹配到已认证的移动账户），基于网页的意图信号在认证后个性化中仍未得到充分利用。现有捕获网页意图的方法通常是临时且狭窄的，缺乏灵活性来同时支持定量下游推荐和定性的大规模理解。在这项工作中，我们提出了一个可扩展且双用途的意图预测框架，用于基于网页的交互，并展示了其在个性化中的适用性。

    arXiv:2606.26277v1 Announce Type: cross  Abstract: Sequential user behavior modeling is widely adopted in industrial recommender systems; however, significant gaps remain in financial services, where pre-login web interactions and authenticated in-app experiences differ drastically. Specifically, pre-login web users typically explore new products, whereas logged-in app users focus on account servicing. Due to the challenge of cross-channel entity resolution (e.g., matching anonymous web sessions to authenticated mobile accounts), web-based intent signals remain underutilized for post-authentication personalization. Existing methods for capturing web-based intent are often ad-hoc and narrow, lacking the flexibility to support both quantitative downstream recommendations and qualitative understanding at scale. In this work, we propose a scalable and dual-purpose intent prediction framework for web-based interactions and demonstrate its applicability for personalization. Our approach tran
    
[^246]: DEEPRUBRIC：面向深度研究智能体高效强化学习的证据树评分标准监督

    DEEPRUBRIC: Evidence-Tree Rubric Supervision for Efficient Reinforcement Learning of Deep Research Agents

    [https://arxiv.org/abs/2606.17029](https://arxiv.org/abs/2606.17029)

    DeepRubric通过“先确定评估目标、再反向合成查询-评分标准对”的逆向数据构建框架，解决了现有大模型直接生成评分标准不完整的问题，从而为深度研究智能体提供更可靠的评价监督并提升强化学习效率。

    

    深度研究智能体通过搜索和推理检索到的证据来综合生成长篇报告。基于评分标准的强化学习通过将报告质量转化为奖励信号的可检验标准来优化这些智能体，从而提升其性能，但其效率取决于这些标准能否可靠地捕捉任务范围和证据需求。现有大多数研究都是让大语言模型为给定查询生成评分标准，但当模型无法推断出潜在的信息需求时，生成的评分标准可能不完整，从而降低强化学习的效率。为了获得更可靠的查询-评分标准监督，我们提出了DeepRubric，这是一个数据构建框架，它颠倒了这一过程：不再为给定查询推断评估标准，而是首先确定基于证据的报告应该从哪些方面进行评估，然后从这些评估目标中合成对齐的查询-评分标准对。

    arXiv:2606.17029v2 Announce Type: replace  Abstract: Deep research agents synthesize long-form reports by searching and reasoning over retrieved evidence. Reinforcement learning with rubric-based rewards improves these agents by optimizing them against checkable criteria that translate report quality into reward signals, but its efficiency depends on whether those criteria reliably capture the task scope and evidence needs. Most existing studies ask an LLM to generate rubrics for a given query, but when the model fails to infer the underlying information needs, the generated rubrics may be incomplete and reduce RL efficiency. To obtain more reliable query--rubric supervision, we introduce DeepRubric, a data construction framework that reverses this process: instead of inferring evaluation criteria for a given query, it first determines what an evidence-backed report should be evaluated on and then synthesizes aligned query--rubric pairs from those evaluation targets. Starting from a sa
    
[^247]: 良性多智能体系统中的错误信息传播

    Misinformation Propagation in Benign Multi-Agent Systems

    [https://arxiv.org/abs/2606.16710](https://arxiv.org/abs/2606.16710)

    本文提出MINT基准（包含10,278个八种风格的LLM生成错误信息段落），首次系统研究了无欺骗意图的良性多智能体系统中错误信息的传播影响，发现辩论的表面稳健性仅源于未受影响智能体的数量优势，交互只有在多数智能体被误导时才能带来最多0.24的准确率恢复。

    

    多智能体系统（MAS）被认为能够解决复杂问题，但其智能体经常基于可能虚假的上下文采取行动。先前的研究表明，单个大语言模型会受到虚假上下文的影响，MAS也可能被对抗性智能体说服同伴而受到损害。我们研究的是良性情形：所有智能体遵循同一协议，没有任何智能体意图欺骗，但部分智能体读到了误导性段落。我们使用MINT进行研究——这是10,278个由大语言模型生成的、涵盖八种风格的错误信息段落，在常识推理、知识问答和道德判断任务上测试了三个开源权重模型。相关的虚假段落会使单个智能体的准确率下降0.12至0.43；不相关或无实质内容的段落通常最多造成0.06的损失。辩论之所以显得稳健，是因为未受错误信息影响的智能体数量多于受影响的智能体。当三个智能体中有一个受到误导时，其表现并不优于在同样暴露于错误信息的独立样本中选择多数结果。只有当多数智能体受到错误信息影响时，交互才有所帮助，最多可恢复0.24的准确率。

    arXiv:2606.16710v2 Announce Type: replace-cross  Abstract: Multi-agent systems (MAS) are credited with solving complex problems, yet their agents routinely act on context that may be false. Prior work shows single LLMs are swayed by false context, and MAS can be compromised by adversarial agents persuading peers. We study the benign case: agents follow one protocol, none intends to deceive, but some read a misleading passage. We study it with MINT, 10,278 LLM-written misinformation passages in eight styles, testing three open-weight models on commonsense reasoning, knowledge QA, and moral judgment. A relevant false passage costs a single agent 0.12-0.43 accuracy; irrelevant or content-free passages typically cost at most 0.06. Debate looks robust because uninformed agents outnumber misinformed ones. With one of three misinformed, it is no better than choosing the majority over equally exposed independent samples. Interaction helps only with a misinformed majority, recovering up to 0.24
    
[^248]: HiMPO：面向长时程智能体低纠缠信用分配的后见之明引导记忆策略优化

    HiMPO: Hindsight-Informed Memory Policy Optimization for Less-Entangled Credit in Long-Horizon Agents

    [https://arxiv.org/abs/2606.16285](https://arxiv.org/abs/2606.16285)

    HiMPO框架将后见相关性作为有界回溯过滤器，为长时程智能体的记忆写入动作分配与下游工具故障等因素解耦的低纠缠信用，并仅对记忆token应用记忆特定优势进行优化。

    

    长时程智能体依赖记忆机制来压缩交互历史，但优化记忆写入面临一个独特的信用分配挑战：记忆更新可能因下游工具故障、噪声观测或推理错误而非其自身贡献而受到奖励或惩罚。我们提出了HiMPO，一个用于为长时程智能体中记忆写入动作分配低纠缠信用的后见之明引导记忆策略优化框架。HiMPO首先通过在相同的预写入状态下比较可从先前记忆和更新后记忆中恢复的任务相关信息，来估计记忆更新的局部效用。然后，它将后见相关性用作一个有界的回溯过滤器，当局部效用得不到目标结果支持时衰减记忆信用。由此产生的记忆特定优势仅应用于记忆token，而轨迹级奖励则优化智能体的其余部分。

    arXiv:2606.16285v2 Announce Type: replace  Abstract: Long-horizon agents rely on memory mechanisms to compress interaction history, but optimizing memory writing faces a distinct credit assignment challenge: a memory update may be rewarded or penalized due to downstream tool failures, noisy observations, or reasoning errors rather than its own contribution. We propose HiMPO, a Hindsight-Informed Memory Policy Optimization framework for assigning less-entangled credit to memory-writing actions in long-horizon agents. HiMPO first estimates the local utility of a memory update by comparing the task-relevant information recoverable from the previous and updated memories under the same pre-write state. It then uses hindsight relevance as a bounded retrospective filter that attenuates memory credit when local utility is not supported by the target outcome. The resulting memory-specific advantage is applied only to memory tokens, while trajectory-level rewards optimize the rest of the agent's
    
[^249]: ProReviewer：面向科学同行评审的证据驱动式调查

    ProReviewer: Evidence-Driven Investigation for Scientific Peer Review

    [https://arxiv.org/abs/2606.13349](https://arxiv.org/abs/2606.13349)

    ProReviewer将科学同行评审建模为马尔可夫决策过程，通过结构化评审日志动态追踪、更新和追溯证据，实现了证据驱动的自适应评审，在ICLR 2026论文上以8B参数规模超越包括397B模型在内的所有对比方法。

    

    有效的同行评审需要能够响应每篇论文中新出现证据的自适应式调查。现有的基于大语言模型的评审系统大多围绕预定义的工作流来组织评审过程，使得新出现的证据难以重塑调查方向。我们提出了ProReviewer，一个证据驱动的评审智能体，它将评审过程建模为马尔可夫决策过程，并以结构化的评审日志作为其状态的核心。在每一步中，智能体利用已积累的发现来决定接下来检查什么以及何时结束评审。随着新证据的出现，它通过修订论断状态、解决未决问题和记录新发现来更新日志。最终评审中的每个要点都引用了支持性的日志条目，使其判断可以追溯至所记录的证据。在1000篇留出的ICLR 2026论文上，经过训练的8B ProReviewer在所有对比方法中取得了最高的平均分数，包括那些基于397B模型构建的方法。

    arXiv:2606.13349v2 Announce Type: replace  Abstract: Effective peer review requires adaptive investigation that responds to evidence emerging from each paper. Existing LLM-based reviewers largely organize the review process around predefined workflows, leaving limited scope for emerging evidence to reshape investigation. We propose ProReviewer, an evidence-driven review agent that formulates reviewing as a Markov Decision Process, with a structured review log forming the core of its state. At each step, the agent uses accumulated findings to decide what to inspect next and when to conclude. As new evidence emerges, it updates the log by revising claim statuses, resolving open questions, and recording new findings. Each point in the final review cites supporting log entries, making its judgments traceable to the recorded evidence. On 1,000 held-out ICLR 2026 papers, the trained 8B ProReviewer achieves the highest average score among all compared methods, including those built on a 397B 
    
[^250]: 当更多文档反而损害RAG：通过领域范围检索测量并减少向量搜索稀释

    When More Documents Hurt RAG: Measuring and Reducing Vector Search Dilution with Domain-Scoped Retrieval

    [https://arxiv.org/abs/2606.11350](https://arxiv.org/abs/2606.11350)

    论文揭示了大语料库下RAG的“向量搜索稀释”问题（precision@10下降63%），并基于200个验证查询、五个LLM骨干和六个基准语料库的MASDR-RAG测试平台证明，基于元数据的领域范围限定是缓解该问题的关键组件。

    

    检索增强生成（RAG）在应用于大型异构文档集合时会出现性能退化。在RAG系统中，稠密相似度的区分性下降，导致top-k检索返回语义相似但上下文错误的文本块。我们将这种失败模式称为向量搜索稀释（vector search dilution）。即使在混合稠密+稀疏检索下，我们首先在实际部署的怀俄明州交通部（WYDOT）系统中观察到了这一现象；在对照实验中，语料库规模从54个文档增长到1,128个文档，使precision@10下降了63%，包含答案文档的召回率下降了23%。在其他六个州交通部（DOT）中也观察到了类似模式。为了诊断这一问题，我们引入了MASDR-RAG（面向RAG的元数据范围领域检索），作为一个受控测试平台，涵盖200个经人工验证的查询、五个LLM骨干模型、六个公开基准语料库和两种索引技术栈。在MASDR-RAG中，领域范围限定是起关键支撑作用的组件。我们发现硬分区

    arXiv:2606.11350v2 Announce Type: replace  Abstract: Retrieval-augmented generation (RAG) degrades when applied to large, heterogeneous document collections. In RAG systems, dense similarity becomes less discriminative, causing top-$k$ retrieval to return semantically similar but contextually incorrect chunks. We refer to this failure mode as vector search dilution. Even under hybrid dense+sparse retrieval, we first observed it in a deployed Wyoming Department of Transportation (WYDOT) system; in a controlled experiment, growing the corpus from 54 to 1,128 documents cut precision@10 by 63% and recall of the answer-bearing document by 23%. Similar patterns were observed in six other DOTs. To diagnose this issue, we introduce MASDR-RAG (Metadata-Scoped Domain Retrieval for RAG) as a controlled testbed over 200 human-validated queries, five LLM backbones, six public benchmark corpora, and two index stacks. In MASDR-RAG, domain scoping is the load-bearing component. We found hard partition
    
[^251]: AgentWorld：一个从多轮交互中训练大语言模型智能体的数据工厂

    AgentWorld: A Data Factory for Training LLM Agents from Multi-Turn Interactions

    [https://arxiv.org/abs/2606.09138](https://arxiv.org/abs/2606.09138)

    AgentWorld是一个数据工厂，通过API代理网关捕获异构智能体运行时的多轮交互，并将其转化为标准化、高质量的结构化训练数据，为LLM智能体训练提供从交互收集到训练消费的统一数据生命周期。

    

    训练大语言模型使其成为有能力的智能体，需要从与环境的**多轮交互**中学习。然而，由异构智能体运行时生成的多轮交互轨迹，难以转化为训练引擎能够直接使用的结构化、高质量训练数据。本文提出AgentWorld，一个通过网关服务器和数据池连接异构智能体运行时与训练引擎的数据工厂，围绕从交互收集到训练消费的统一数据生命周期重构LLM智能体训练。网关服务器通过LLM API代理捕获多轮交互，数据池则将其组织为标准化的步骤记录，以便进行数据整理与训练。交互式仪表板支持轨迹监控、步骤检查和数据审查。在演示中，用户可以集成开箱即用的白盒和黑盒智能体，对轨迹进行追踪与整理……

    arXiv:2606.09138v2 Announce Type: replace-cross  Abstract: Training LLMs to act as capable agents requires learning from multi-turn interactions with environments. However, multi-turn interaction trajectories generated by heterogeneous agent runtimes are difficult to transform into structured, high-quality training data that training engines can readily consume. In this paper, we propose AgentWorld, a data factory that bridges heterogeneous agent runtimes and training engines through a Gateway Server and a Data Pool, reframing LLM agent training around a unified data lifecycle from interaction collection to training consumption. The Gateway Server captures multi-turn interactions through an LLM API proxy, while the Data Pool organizes them into standardized step records for curation and training. An interactive dashboard supports trajectory monitoring, step inspection, and data review. In our demonstration, users integrate ready-to-use white-box and black-box agents, trace and curate t
    
[^252]: 用5岁小孩能懂的方式解释，或按我选择的任意程度解释：评估语言模型响应中交互式复杂度的潜力

    Explain Like I'm 5 or Whatever I Choose: Evaluating the Potential for Interactive Complexity in Language Model Responses

    [https://arxiv.org/abs/2606.06788](https://arxiv.org/abs/2606.06788)

    本研究通过用户研究确定了可靠的文本复杂度度量标准及模型遵循准则，并在557个科学查询上评估了五个大语言模型沿复杂度连续区间生成摘要的能力，发现模型虽能调整响应复杂度，但大多数变化仍与要求不一致。

    

    鉴于大型语言模型（LLMs）在阅读和获取信息方面的使用日益增多，以及用户需求和背景的多样性，对文本进行直接操纵可以成为终端用户控制的一种有价值机制。然而，实现直接操纵要求模型生成与用户所请求的更改相一致的文本。我们聚焦于其中一个场景——操纵科学信息的复杂度，并探讨模型能否沿着文本复杂度的连续区间生成多个摘要，从而实现有效的用户选择与控制。通过一项被试内用户研究，我们确定了可靠的复杂度度量指标以及模型应符合这些指标的准则。我们在557个科学查询上评估了五个来自不同家族、不同规模和不同推理能力的大型语言模型。我们发现，尽管模型能够在不同响应之间调整复杂度，但大多数变化仍然不一致，模型在调整复杂度指标时会……

    arXiv:2606.06788v2 Announce Type: replace  Abstract: Given the increased use of large language models (LLMs) for reading and seeking information and the varying needs and backgrounds of users, direct manipulation of text can be a valuable mechanism for end-user control. However, enabling direct manipulation requires that models generate text consistent with user-requested changes. We focus on one such setting, manipulating the complexity of scientific information, and ask if models can generate multiple summaries along a continuum of text complexity that enable effective user selection and control. Through a within-subjects user study, we identify reliable complexity measures and a criterion for how models should conform to these measures. We evaluate five LLMs of different families, sizes, and reasoning abilities on $557$ scientific queries. We find that while models can vary complexity across responses, most changes remain inconsistent, with models shifting the complexity measures in
    
[^253]: Ouvia：一个以用户为中心的、用于衡量真实世界交流场景中语音翻译可用性的评估框架

    Ouvia: A User-centered Framework for Measuring Usability of Speech Translation in Real-World Communication Scenarios

    [https://arxiv.org/abs/2606.06177](https://arxiv.org/abs/2606.06177)

    Ouvia是一个以用户为中心的评估框架，通过在真实一对一交流场景中收集超过1,750次交互发现，现代语音翻译仅有约一半的交互被评为可用，且不同人口群体之间存在显著的可用性差距。

    

    语音翻译（ST）在用户应用中的采用日益广泛，但其评估主要聚焦于脱离实际情境的测试平台和整体质量，而非最终用户的交流需求。我们提出了Ouvia，这是一个用于衡量真实世界环境中用户对语音翻译输出感知可用性的评估框架。Ouvia聚焦于一对一交流场景：一名英语使用者需要向一名葡萄牙语使用者传达请求，信息由系统自动翻译。通过自定义网络应用和多阶段研究设计，我们在医疗保健和日常生活情境中收集了超过1,750次此类交互，这些交互由四个语音翻译系统调节，涉及来自三种英语方言和两种性别的说话者。我们发现，现代语音翻译对人们的帮助程度有限——只有大约一半的交互被评为可用——且不同人口群体之间报告的可用性存在显著差距。此外，在质量指标方面……

    arXiv:2606.06177v2 Announce Type: replace  Abstract: Speech translation (ST) is increasingly adopted in user applications, yet its evaluation largely focuses on decontextualized testbeds and holistic quality, rather than end users' communication needs. We introduce Ouvia, an evaluation framework for measuring user-perceived usability of speech translation outputs in real-world settings. Ouvia focuses on one-to-one communication: an English speaker needs to convey a request to a Portuguese speaker, and the message is automatically translated. Through a custom web app and multi-phase study design, we collect more than 1,750 such interactions in healthcare and everyday situations, mediated by four ST systems, involving speakers from three English dialects and two genders. We find that modern ST serves people only to a limited extent---only around half of interactions are rated as usable---with significant gaps in reported usability across demographic groups. Moreover, among quality metric
    
[^254]: 从经验中学习可复用的智能体子程序

    Learning Reusable Agent Subroutines from Experience

    [https://arxiv.org/abs/2606.02994](https://arxiv.org/abs/2606.02994)

    ARISE通过识别智能体轨迹中反复出现的流程并将其合成为可复用的子程序库，在叙述推理、规则应用和约束规划等多个任务上显著优于提示方法、散文与代码工件归纳以及思维链等现有方法，准确率提升最高达15.6个百分点。

    

    现有方法从智能体轨迹中归纳出各种形式的知识，从示例工作流到程序化技能。我们提出了ARISE（从成功经验中归纳智能体例程），该方法能够识别智能体轨迹中反复出现的流程，并将其合成为一个可复用的智能体子程序库。在叙述推理、规则应用和约束规划领域的五项任务中，该方法被证明是一种比提示方法和散文与代码工件归纳方法更有效的学习策略。在主要评估的全部十个模型-任务设置中，ARISE均优于思维链方法。与最先进的智能体经验归纳方法相比，我们的方法比评估的最强基线高出6.1个百分点，比代码技能归纳方法高出11.5个百分点。在受控实验中，与对归纳内容进行格式化处理相比，ARISE还将平均准确率提升了最多15.6个百分点。

    arXiv:2606.02994v2 Announce Type: replace-cross  Abstract: Existing methods induce various forms of knowledge from agent traces, from sample workflows to programmatic skills. We propose ARISE (Agent Routine Induction from Successful Experience), which identifies recurring procedures in agent traces and synthesizes them into a reusable library of agent subroutines. Across five tasks in narrative deduction, rule application, and constraint planning, it is shown to be a more effective learning strategy than prompting methods and prose and code artifact induction methods. ARISE outperforms chain-of-thought in all ten model-task settings of the main evaluation. Compared to state-of-the-art methods in agent experience induction, our method scores 6.1 percentage points over the strongest evaluated baselines, and 11.5 percentage points over code-skill induction. In our controlled experiment, ARISE also improves average accuracy by up to 15.6 percentage points compared to formatting the induced
    
[^255]: 通过自动形式化实现稳健的异步规划

    Robust Asynchronous Planning via Auto-Formalization

    [https://arxiv.org/abs/2606.00981](https://arxiv.org/abs/2606.00981)

    该论文统一了异步规划的公式化并发布三个新基准，发现LLM作为规划者或经CP-SAT形式化均可取得高准确率，而专为时序规划设计的PDDL2.1反而表现最差，且由于两者由同一中间表示编译时准确率几乎一致，瓶颈在于LLM对PDDL的自动生成而非形式化语言本身。

    

    大型语言模型可以通过两种方式进行规划：作为规划者直接生成调度方案，或作为形式化器将任务翻译成求解器可用的形式化规范。大多数现实世界的任务都是异步的，具有非均匀的持续时间、并发性以及执行时的变更，但现有基准测试几乎未涵盖这些情况。我们在统一的公式化框架下整合了异步规划，并发布了三个基准测试，涵盖任务图规模、基础状态与资源约束以及执行时事件。令人惊讶的是，专为时序规划而构建的形式化语言PDDL2.1却是最弱的流水线。在AsyncPlan-XXL上，其平均规划准确率仅为30.8%，而规划者方法达到81.7%，CP-SAT形式化器达到94.0%。这一差距随任务规模扩大而加剧，在100个动作时，三种流水线的准确率分别为9.0%、44.5%和82.5%。问题并不在于形式化语言本身——从同一个模型提取的中间表示编译而来，PDDL2.1和CP-SAT达到了几乎相同的准确率（81.6%对比……

    arXiv:2606.00981v2 Announce Type: replace  Abstract: LLMs can plan by generating schedules directly as a Planner, or by translating a task into a formal specification for a solver as a Formalizer. Most real-world tasks are asynchronous, with non-uniform durations, concurrency, and execution-time changes, yet existing benchmarks hardly cover them. We unify asynchronous planning under one formulation and release three benchmarks spanning task-graph scale, grounded state and resource constraints, and execution-time events. Surprisingly, PDDL2.1, the formalism built for temporal planning, is the weakest pipeline. On AsyncPlan-XXL it averages 30.8% plan accuracy, against 81.7% for the Planner and 94.0% for a CP-SAT Formalizer. The gap widens with scale, and at 100 actions the three pipelines reach 9.0%, 44.5%, and 82.5%, respectively. The formalism is not to blame. Compiled from the same model-extracted intermediate representation, PDDL2.1 and CP-SAT reach nearly identical accuracy (81.6% v
    
[^256]: T2J-BENCH：基于执行效果的PyTorch到JAX代码仓库迁移编码智能体基准测试

    T2J-BENCH: Benchmarking Coding Agents on Execution-Grounded PyTorch-to-JAX Repository Migration

    [https://arxiv.org/abs/2605.29054](https://arxiv.org/abs/2605.29054)

    提出T2J-BENCH基准，包含63个涵盖18个模型家族的整仓库PyTorch到JAX迁移任务，以执行行为一致性和无torch依赖作为核心评价标准，用以真实评估编码智能体进行跨框架代码库迁移的能力。

    

    将机器学习代码库跨框架迁移——例如从PyTorch迁移到JAX——是一项常规但成本高昂的工程任务，其本质是跨范式的：急切执行的、有状态的、面向对象的模块必须被重写为作用于显式参数pytree的纯函数、可被jit编译的函数。我们探究编码智能体能否正确执行这类整仓库级别的迁移，并指出现有代码基准——它们以单元测试通过率或与参考翻译的表面相似度来评分——无法回答这一问题：它们既不要求迁移后的系统在“行为”上保持一致，也无法防止JAX重新实现通过暗中导入PyTorch本身来伪造等价性。我们提出T2J-BENCH，这是一个包含63个整仓库PyTorch到JAX迁移任务的基准，涵盖18个模型家族，横跨训练与服务两个场景，其中无torch依赖的检查点转换本身就是交付成果的一部分。T2J-BENCH以执行效果为依据进行评分。

    arXiv:2605.29054v3 Announce Type: replace-cross  Abstract: Migrating a machine-learning codebase across frameworks--for example from PyTorch to JAX--is a routine but costly engineering task that is fundamentally cross-paradigm: eager, stateful, object-oriented modules must be rewritten as pure, jittable functions over an explicit parameter pytree. We ask whether coding agents can perform such whole-repository migrations correctly, and argue that existing code benchmarks, which grade unit-test pass rates or surface-level similarity to a reference translation, cannot answer this: they neither require the migrated system to \emph{behave} the same nor prevent a JAX reimplementation from faking equivalence by secretly importing PyTorch itself. We introduce T2J-BENCH, a benchmark of 63 whole-repository PyTorch-to-JAX migration tasks spanning 18 model families across both training and serving, where torch-free checkpoint conversion is itself part of the deliverable. T2J-BENCH grades execution
    
[^257]: ConRAG：面向多跳问答的共识驱动证据流框架

    ConRAG: Consensus-Driven Evidence Flow for Multi-Hop Question Answering

    [https://arxiv.org/abs/2605.28093](https://arxiv.org/abs/2605.28093)

    提出ConRAG框架，通过共识驱动的多视角排序选择相关证据，并利用槽位机制在推理步骤间传递中间信息，显著提升了大语言模型在多跳问答任务上的表现。

    

    检索增强生成（RAG）已成为提升大语言模型（LLM）在多跳问答（QA）任务上性能的一种有前景的范式，该任务需要对来自多个文档的证据进行推理。然而，当前的多跳RAG方法在证据选择和信息传递方面仍然存在困难，例如检索到不相关的段落，或无法将中间信息传递给后续推理步骤，从而阻碍了连贯的多跳推理。为此，我们提出了ConRAG，一个共识驱动的证据流框架，能够有效提升大语言模型在复杂多跳问答上的性能。简而言之，ConRAG的核心在于通过共识驱动的多视角排序来选择相关证据，并利用基于槽位的机制在相互依赖的推理步骤之间传递中间信息。在三个多跳问答基准上的大量实验表明，ConRAG以明显优势超越了基线方法。

    arXiv:2605.28093v3 Announce Type: replace  Abstract: Retrieval-augmented generation (RAG) has emerged as a promising paradigm for enhancing large language models (LLMs) on multi-hop question answering (QA), which requires reasoning over evidence from multiple documents. However, current multi-hop RAG approaches still struggle with evidence selection and information transfer, i.e., retrieving irrelevant passages or failing to pass intermediate information to subsequent reasoning steps, thereby hindering coherent multi-hop reasoning. To this end, we propose ConRAG, a consensus-driven evidence flow framework that effectively boosts LLM performance on complex multi-hop QA. In short, the core of ConRAG is to select relevant evidence via consensus-driven multi-view ranking, and leverage a slot-based mechanism to transfer intermediate information between dependent reasoning steps. Extensive experiments across three multi-hop QA benchmarks show that ConRAG surpasses the baselines by a clear ma
    
[^258]: 当推理监督带来损害：基于托兰斯创造性写作测验（TTCW）的长篇文学评论生成

    When Reasoning Supervision Hurts: TTCW-Based Long-Form Literary Review Generation

    [https://arxiv.org/abs/2605.20364](https://arxiv.org/abs/2605.20364)

    本研究构建了包含26万余篇长篇故事、覆盖14个TTCW创造力维度的大规模文学评论数据集，并发现一个反直觉结论：评论员风格的推理轨迹监督反而会通过引发语义重复循环而损害文学评论生成，无轨迹监督能实现更高的分数保真度和近乎完整的报告生成。

    

    长篇文学写作的自动评估仍然充满挑战，尤其是在独创性和灵活性等与创造力相关的维度上。针对基于结构化托兰斯创造性写作测验（TTCW）的评估缺乏大规模监督数据的问题，我们构建了一个包含263,911篇长篇故事的数据集，每篇故事均配有标量分数，以及在14个基于TTCW维度上经元合成的评论意见。利用该数据集，我们研究了评论员风格的推理轨迹能否在Qwen3-4B、Qwen3-8B和Nemotron-8B三个模型上改进结构化文学评论的生成。在所有三个模型上，无轨迹监督均取得了更高的分数模仿保真度以及近乎完整的报告生成，而轨迹监督则将报告完成率降至85.9%–90.9%。失败分析表明，88.6%的失败推理生成在轨迹内部存在语义重复循环，且这些循环集中在特定的故事类型上。

    arXiv:2605.20364v2 Announce Type: replace  Abstract: Automatic evaluation of long-form literary writing remains challenging, particularly for creativity-related dimensions such as originality and flexibility. We address the lack of large-scale supervision for structured Torrance Test of Creative Writing (TTCW)-based evaluation by constructing a dataset of 263{,}911 long-form stories, each paired with scalar scores and meta-synthesised review comments across 14 TTCW-based dimensions. Using this dataset, we study whether reviewer-style reasoning traces improve structured literary review generation across Qwen3-4B, Qwen3-8B, and Nemotron-8B. Across all three models, no-trace supervision yields higher score imitation fidelity and near-complete report generation, whereas trace supervision reduces completion to $85.9$--$90.9\%$. Failure analysis shows that $88.6\%$ of failed reasoning generations contain semantic repetition loops inside the trace. These loops concentrate in particular story-
    
[^259]: OScaR：面向大语言模型及更广泛领域的极端KV缓存量化的“奥卡姆剃刀”

    OScaR: The Occam's Razor for Extreme KV Cache Quantization in LLMs and Beyond

    [https://arxiv.org/abs/2605.19660](https://arxiv.org/abs/2605.19660)

    该论文识别出Token范数不平衡（TNI）是极端压缩下KV缓存逐通道量化保真度的主要瓶颈，并提出了一种精确且轻量的量化方法OScaR，无需复杂的量化流水线即可实现高保真KV缓存压缩。

    

    向长上下文推理和多模态智能的快速发展，使得键值（KV）缓存的内存占用成为高效部署中的主要内存瓶颈。虽然已有的逐通道量化方法能够有效处理键张量中固有的通道级离群值，但其在极端压缩下的效力会减弱。在本工作中，我们从实证和理论两个视角重新审视了逐通道量化范式的固有局限性。我们的分析将Token范数不平衡（TNI）识别为量化保真度的主要瓶颈。我们证明，当共享量化参数需要跨越具有显著范数差异的token组时，TNI会系统性地放大误差。我们没有依赖复杂的量化流水线（如TurboQuant），而是提出了OScaR（Omni-Scaled Canalized Rotation，全尺度引导旋转），一种精确且轻量的KV缓存协同量化方法。

    arXiv:2605.19660v3 Announce Type: replace-cross  Abstract: The rapid advancement toward long-context reasoning and multi-modal intelligence has made the memory footprint of the Key-Value (KV) cache a dominant memory bottleneck for efficient deployment. While the established per-channel quantization effectively accommodates intrinsic channel-wise outliers in Key tensors, its efficacy diminishes under extreme compression. In this work, we revisit the inherent limitations of the per-channel quantization paradigm from both empirical and theoretical perspectives. Our analysis identifies Token Norm Imbalance (TNI) as the primary bottleneck to quantization fidelity. We demonstrate that TNI systematically amplifies errors when shared quantization parameters are required to span token groups exhibiting substantial norm disparities. Instead of relying on intricate quantization pipelines (e.g., TurboQuant), we propose OScaR (Omni-Scaled Canalized Rotation), an accurate and lightweight KV cache co
    
[^260]: 数字人格何时能可靠地近似人类调查结果？

    When Can Digital Personas Reliably Approximate Human Survey Findings?

    [https://arxiv.org/abs/2605.10659](https://arxiv.org/abs/2605.10659)

    基于LISS面板的研究表明，大语言模型驱动的数字人格能在分布层面较好地近似人类调查回答（尤其在涉及稳定属性和价值观的领域），但在个体预测和恢复受访者多变量结构方面仍存在根本局限。

    

    由大语言模型驱动的数字人格日益被提议作为人类调查受访者的替代品，但它们何时能够可靠地近似人类调查结果仍不清楚。我们利用LISS面板回答这一问题：从受访者的背景变量和2023年之前的调查历史中构建数字人格，然后将其与同一受访者在截止日期之后保留的答案进行对照测试。在四种人格架构、三个大语言模型和两个预测任务中，我们在问题、受访者、分布、公平性和聚类等多个层面评估其表现。数字人格在人类回答分布的对齐上有所改善，尤其是在与稳定属性和价值观相关的领域，但在个体预测方面仍然有限，且无法恢复受访者的多变量结构。检索增强架构带来了最明显的收益，但其性能更多取决于人类回答的结构而非（模型本身）（摘要原文在此处截断）。

    arXiv:2605.10659v3 Announce Type: replace  Abstract: Digital personas powered by Large Language Models (LLMs) are increasingly proposed as substitutes for human survey respondents, yet it remains unclear when they can reliably approximate human survey findings. We answer this question using the LISS panel, constructing personas from respondents' background variables and pre-2023 survey histories, then testing them against the same respondents' held-out post-cutoff answers. Across four persona architectures, three LLMs, and two prediction tasks, we assess performance at the question, respondent, distributional, equity, and clustering levels. Digital personas improve alignment with human response distributions, especially in domains tied to stable attributes and values, but remain limited for individual prediction and fail to recover multivariate respondent structure. Retrieval-augmented architectures provide the clearest gains, but performance depends more on human response structure th
    
[^261]: 关于评估方法论挑战的研究：综述

    On Challenges in Evaluation Methodology: A Review

    [https://arxiv.org/abs/2604.25923](https://arxiv.org/abs/2604.25923)

    本文通过跨越四十年的范围综述构建了包含22项反复出现的评估方法论问题的分类体系，发现许多问题在文献中的可见度下降且缺乏机构检查清单的覆盖，从而为未来的重新研究和领域层面干预指明了方向。

    

    语言模型的最新进展促使越来越多的研究开始质疑当前通行的评估实践。然而，许多此类批评早已在自然语言处理（NLP）领域中被讨论过，该领域在评估方法论的反思方面有着悠久的历史。我们进行了一项跨越四十年的范围综述，并构建了一个包含22个反复出现的问题的分类体系。针对每个问题，我们考察了其在所回顾文献中被讨论的近期程度，以及是否在机构检查清单中得到明确关注，结果发现一些问题在文献中的可见度已有所下降，且许多问题缺乏这些检查清单的覆盖。我们的工作概述了塑造该领域的评估方法论主要问题，并为研究社区重新开展调查和实施领域层面的干预确定了潜在目标。

    arXiv:2604.25923v2 Announce Type: replace  Abstract: Recent advances in language models have prompted a growing body of work that questions prevailing evaluation practices. However, many such critiques have already been discussed in natural language processing (NLP), a field with a long history of methodological reflection on evaluation. We conduct a scoping review spanning four decades and develop a taxonomy of 22 recurring concerns. For each concern, we examine how recently it was discussed in the reviewed literature and whether it receives explicit attention in institutional checklists, finding that some concerns have become less visible in literature and many lack coverage in these checklists. Our work provides an overview of the major concerns in evaluation methodology that have shaped the field and identifies potential targets for renewed investigation and field-level intervention by the research community.
    
[^262]: ContextWeaver：面向大语言模型智能体的选择性、依赖结构化记忆构建方法

    ContextWeaver: Selective and Dependency-Structured Memory Construction for LLM Agents

    [https://arxiv.org/abs/2604.23069](https://arxiv.org/abs/2604.23069)

    ContextWeaver将智能体的交互轨迹组织成带依赖链接的推理步骤图，结合紧凑依赖摘要与执行反馈验证来选择性构建记忆上下文，从而在固定预算下显著提升LLM智能体在软件工程和终端长上下文任务中的解决率。

    

    大型语言模型（LLM）智能体在长上下文交互中常常表现不佳。随着历史记录的增长，滑动窗口和提示压缩可能会丢弃后续步骤所依赖的早期信息。基于检索的记忆系统能够呈现相关内容，但仍然忽略了步骤之间的因果和逻辑依赖关系。我们提出了ContextWeaver，这是一个选择性的、依赖结构化的记忆框架，它将智能体的交互轨迹组织成一个由推理步骤构成的图，并为未来的动作选择相关上下文。ContextWeaver支持：（1）基于依赖的构建与遍历，将每个步骤与其所依赖的早期步骤链接起来；（2）对推理路径进行紧凑的依赖摘要；（3）一个融合执行反馈的轻量级验证层。在五个软件工程和终端基准测试中，采用每任务固定预算，ContextWeaver相比滑动窗口基线提高了任务解决率。

    arXiv:2604.23069v2 Announce Type: replace  Abstract: Large language model (LLM) agents often struggle in long-context interactions. As the history grows, sliding windows and prompt compression may drop earlier information that later steps rely on. Retrieval-based memory systems surface relevant content but still overlook the causal and logical dependencies between steps. We introduce ContextWeaver, a selective and dependency-structured memory framework that organizes an agent's interaction trace into a graph of reasoning steps and selects the relevant context for future actions. ContextWeaver supports (1) dependency-based construction and traversal that link each step to the earlier steps it relies on; (2) compact dependency summaries of the reasoning paths; and (3) a lightweight validation layer that incorporates execution feedback. On five software-engineering and terminal benchmarks with a fixed budget per task, ContextWeaver improves the resolve rate over a sliding-window baseline 
    
[^263]: 大语言模型经济因果推理中的意识形态偏见

    Ideological Bias in LLMs' Economic Causal Reasoning

    [https://arxiv.org/abs/2604.21334](https://arxiv.org/abs/2604.21334)

    该研究在EconCausal基准中引入亲政府与亲市场观点存在分歧的意识形态争议案例，发现大语言模型在经济因果推理中表现出系统性意识形态偏见，且争议性案例的因果方向预测难度显著更高。

    

    大语言模型（LLM）在推理经济因果效应时是否会表现出系统性的意识形态偏见？随着大语言模型越来越多地被应用于政策分析和经济报道——在这些领域中，方向正确的因果判断至关重要——这一问题具有直接的现实意义。我们通过在EconCausal基准中扩展加入意识形态争议案例来开展系统性评估——即干预导向（亲政府）视角与市场导向（亲市场）视角会预测出相反因果符号的实例。我们从顶级经济学和金融学期刊中提取的10,490个因果三元组（具有经验验证效应方向的处理-结果对）中，识别出1,056个意识形态争议实例，并评估了20个最先进的大语言模型预测经验支持的因果方向的能力。我们发现，意识形态争议项目始终比非争议项目更难预测，并且在17个o……

    arXiv:2604.21334v3 Announce Type: replace-cross  Abstract: Do large language models (LLMs) exhibit systematic ideological bias when reasoning about economic causal effects? As LLMs are increasingly used in policy analysis and economic reporting, where directionally correct causal judgments are essential, this question has direct practical stakes. We present a systematic evaluation by extending the EconCausal benchmark with ideology-contested cases - instances where intervention-oriented (pro-government) and market-oriented (pro-market) perspectives predict divergent causal signs. From 10,490 causal triplets (treatment-outcome pairs with empirically verified effect directions) derived from top-tier economics and finance journals, we identify 1,056 ideology-contested instances and evaluate 20 state-of-the-art LLMs on their ability to predict empirically supported causal directions. We find that ideology-contested items are consistently harder than non-contested ones, and that across 17 o
    
[^264]: 图像化文档表示在科学文献检索中的不足

    Document-as-Image Representations Fall Short for Scientific Retrieval

    [https://arxiv.org/abs/2604.18508](https://arxiv.org/abs/2604.18508)

    本文提出基于 LaTeX 源文件构建的新基准 ArXivDoc，论证“文档即图像”的表示方法不适合科学文献检索任务，因为关键证据分布于文本、表格和图表等结构化源中。

    

    许多近期的文档嵌入模型基于“文档即图像”的表示进行训练，即将渲染后的页面作为图像而非底层源文件进行嵌入。与此同时，现有的科学文档检索基准（如 ArXivQA 和 ViDoRe）将文档视为页面的图像，从而隐性地偏向了这类表示方法。在本工作中，我们认为这种范式并不适合文本丰富的多模态科学文档，因为在这类文档中，关键证据分布在包括文本、表格和图表在内的结构化源文件中。为了研究这一场景，我们提出了 ArXivDoc，一个基于科学论文底层 LaTeX 源文件构建的新基准。与 PDF 或基于图像的表示不同，LaTeX 提供了对结构化元素（如章节、表格、图表和公式）的直接访问，从而支持基于特定证据类型的受控查询构建。我们系统地比较了纯文本、基于图像的表示方法

    arXiv:2604.18508v2 Announce Type: replace-cross  Abstract: Many recent document embedding models are trained on document-as-image representations, embedding rendered pages as images rather than the underlying source. Meanwhile, existing benchmarks for scientific document retrieval, such as ArXivQA and ViDoRe, treat documents as images of pages, implicitly favoring such representations. In this work, we argue that this paradigm is not well-suited for text-rich multimodal scientific documents, where critical evidence is distributed across structured sources, including text, tables, and figures. To study this setting, we introduce ArXivDoc, a new benchmark constructed from the underlying LaTeX sources of scientific papers. Unlike PDF or image-based representations, LaTeX provides direct access to structured elements (e.g., sections, tables, figures, equations), enabling controlled query construction grounded in specific evidence types. We systematically compare text-only, image-based, and
    
[^265]: Omni-Embed-Audio：利用多模态大语言模型实现稳健的音频-文本检索

    Omni-Embed-Audio: Leveraging Multimodal LLMs for Robust Audio-Text Retrieval

    [https://arxiv.org/abs/2604.18360](https://arxiv.org/abs/2604.18360)

    提出了基于多模态大语言模型的音频检索编码器 Omni-Embed-Audio（OEA），并引入五种用户意图查询形式、难负例挖掘流水线及判别指标（HNSR、TFR），以更贴近真实搜索行为的方式评估音频-文本检索的鲁棒性。

    

    基于对比语言-音频预训练（CLAP）的音频-文本检索系统在传统基准测试中表现优异；然而，这些基准测试依赖于字幕式的查询方式，与真实世界的搜索行为存在很大差异，限制了对实际检索鲁棒性的评估。我们提出了 Omni-Embed-Audio（OEA），一种利用具备原生音频理解能力的多模态大语言模型的面向检索的编码器。为了系统性地评估超越字幕式查询的鲁棒性，我们引入了用户意图查询——五种反映自然搜索行为的查询形式：提问式、命令式、关键词标签式、同义改写式以及基于排除的否定式查询。针对否定式查询，我们开发了难负例挖掘流水线，并提出了判别指标（HNSR、TFR）来评估模型抑制声学相似干扰项的能力。在 AudioCaps、Clotho 和 MECAT 数据集上的实验表明，OEA 取得了……

    arXiv:2604.18360v4 Announce Type: replace-cross  Abstract: Audio-text retrieval systems based on Contrastive Language-Audio Pretraining (CLAP) achieve strong performance on traditional benchmarks; however, these benchmarks rely on caption-style queries that differ substantially from real-world search behavior, limiting their assessment of practical retrieval robustness. We present Omni-Embed-Audio (OEA), a retrieval-oriented encoder leveraging multimodal LLMs with native audio understanding. To systematically evaluate robustness beyond caption-style queries, we introduce User-Intent Queries (UIQs) - five formulations reflecting natural search behaviors: questions, commands, keyword tags, paraphrases, and exclusion-based negative queries. For negative queries, we develop a hard negative mining pipeline and propose discrimination metrics (HNSR, TFR) assessing models' ability to suppress acoustically similar distractors. Experiments on AudioCaps, Clotho, and MECAT show that OEA achieves c
    
[^266]: 面向多会话个性化工具调用的潜在偏好建模

    Latent Preference Modeling for Multi-Session Personalized Tool Calling

    [https://arxiv.org/abs/2604.17886](https://arxiv.org/abs/2604.17886)

    该论文提出了多会话个性化工具调用基准 MPT 以及测试时记忆方法 PRefine，通过“生成-验证-精炼”循环将用户潜在偏好维护为文本假设，在五个大语言模型上的表现超越所有基线，并在偏好迁移任务上成为唯一超越完整历史提示的方法。

    

    用户在与基于大语言模型（LLM）的智能体交互时，常常会在请求中省略关键细节，导致工具调用时的输入信息不完整。这对工具增强型智能体构成了根本性挑战，因为 API 执行通常需要完整的参数，这凸显了个性化工具调用的必要性。为了在更贴近实际的场景中研究这一问题，我们提出了多会话个性化工具调用，这是一个包含 4,695 个实例、覆盖 459 个多会话交互历史的基准数据集，涵盖三项挑战：偏好召回、偏好归纳和偏好迁移。我们进一步提出了 PRefine，一种测试时记忆方法，它将用户的潜在偏好维护为一个文本假设，并通过“生成-验证-精炼”循环不断修正。在五个大语言模型上的实验表明，现有记忆系统的表现不及完整历史提示；而 PRefine 的表现优于所有基线方法，并且是唯一在偏好迁移任务上超越完整历史提示的方法。这些结果表明，用于个性化的记忆……（摘要原文在此处截断）

    arXiv:2604.17886v2 Announce Type: replace-cross  Abstract: Users often omit essential details in their requests to LLM-based agents, resulting in under-specified inputs for tool use. This poses a fundamental challenge for tool-augmented agents, as API execution typically requires complete arguments, highlighting the need for personalized tool calling. To study this problem in a more realistic setup, we present Multi-Session Personalized Tool Calling (MPT), a benchmark comprising 4,695 instances over 459 multi-session interaction histories that cover three challenges: Preference Recall, Induction, and Transfer. We further propose PRefine, a test-time memory method that maintains the user's latent preference as a textual hypothesis revised through a generate-verify-refine loop. Across five LLMs, existing memory systems underperform full-history prompting; PRefine outperforms all baselines and alone surpasses it on Preference Transfer. These results indicate that memory for personalized a
    
[^267]: 视觉语言模型中物理推理的奖励设计

    Reward Design for Physical Reasoning in Vision-Language Models

    [https://arxiv.org/abs/2604.13993](https://arxiv.org/abs/2604.13993)

    本文系统比较了外部奖励与基于模型自身注意力的内部奖励对视觉语言模型物理推理能力的影响，发现外部奖励贡献了主要准确率提升，而内部注意力奖励作为一种无需标注的接地信号可有效改善空间关系推理，二者互补。

    

    基于图像的物理推理需要整合感知、领域知识和多步符号推理，然而视觉语言模型（VLMs）在物理基准测试中的表现远不及人类。强化学习可以通过奖励信号来改进推理，但这些信号是否必须来自模型外部，还是可以从模型自身中推导得出，目前仍不清楚。我们将依赖外部验证器的外部奖励（包括基于规则的格式与答案正确性检查，或大语言模型评判的评分标准）与基于注意力的内部奖励进行比较，后者由模型自身对图像区域的注意力计算得出，无需标注或评判者。我们发现，外部奖励带来了大部分的准确率提升，并且在奖励推理token时能够保持更连贯的推理过程。而基于注意力的内部奖励则提供了一种无需标注的接地信号，能够改善空间关系推理，并与外部奖励形成互补。

    arXiv:2604.13993v2 Announce Type: replace-cross  Abstract: Physical reasoning over images requires integrating perception, domain knowledge, and multi-step symbolic inference, yet Vision-Language Models (VLMs) fall far short of human performance on physics benchmarks. Reinforcement learning can improve reasoning via reward signals, but it remains unclear whether these signals must come from outside the model or can be derived from the model itself. We compare external rewards, which rely on an outside verifier (rule-based format and answer-correctness checks, or an LLM-judge rubric), with an internal attention-based reward computed from the model's own attention over image regions, requiring no labels or judge. We find that external rewards drive most of the accuracy gains, while preserving more coherent reasoning when rewarded for reasoning tokens. The internal attention-based reward, provides an annotation-free grounding signal that improves spatial-relation reasoning and is compleme
    
[^268]: 幻影悖论：语言模型将未能解决的循环真值依赖视为矛盾

    Phantom Paradoxes: Language Models Treat Unresolved Circular Truth Dependence as Contradiction

    [https://arxiv.org/abs/2604.12128](https://arxiv.org/abs/2604.12128)

    语言模型会将表面上像悖论、但实际上存在唯一一致真值赋值的循环自指问题误判为“不一致”，这种被称为“幻影悖论”的错误源于无法通过有限步骤传播确定真值的循环结构，而非自指本身。

    

    语言模型究竟是在自指本身上存在困难，还是在那些沿引用传播真值无法在有限步骤内确定取值的结构上存在困难？我们的结果支持后者。在我们的主要测试中，语句之间以循环方式相互指涉彼此的真值，但恰好存在唯一一个一致的赋值。局部传播永远无法确定被查询语句的取值，而穷举搜索可以找到它，因此回答 INCONSISTENT（不存在一致赋值）总是错误的。我们将这种对可满足问题的误读称为“幻影悖论”。每个循环问题都有一个答案相同但没有循环的接地孪生问题。为了排除逻辑难度和不熟悉措辞的干扰，我们还构建了约束孪生问题（不含真值谈论的等价公式）以及困惑度匹配的接地对照组。在预先固定的验证性行为测试中，来自三个系列的四个开放模型平均多回答 INCONSISTENT 31.0 个百分点……

    arXiv:2604.12128v2 Announce Type: replace  Abstract: Do language models struggle with self-reference itself, or with structures where propagating truth values along references cannot settle a value in finitely many steps? Our results support the latter. In our main test, statements refer to one another's truth in a cycle, yet exactly one consistent assignment exists. Local propagation never settles the queried value, but exhaustive search finds it, so INCONSISTENT (no consistent assignment) is always wrong. We call such misreadings of satisfiable problems phantom paradoxes. Each circular problem has a grounded twin with the same answer but no cycle. To rule out logical difficulty and unfamiliar wording, we added constraint twins (equivalent formulas without talk of truth) and perplexity-matched grounded controls. In confirmatory behavioral tests fixed before any confirmation response, four open models from three families answered INCONSISTENT on average 31.0 percentage points more ofte
    
[^269]: AdaptFuse：通过外置化贝叶斯计算实现无需训练的序列偏好推断

    AdaptFuse: Training-Free Sequential Preference Inference via Externalized Bayesian Computation

    [https://arxiv.org/abs/2604.03925](https://arxiv.org/abs/2604.03925)

    提出了无需训练的AdaptFuse框架，将贝叶斯后验计算外置于符号模块，并与冻结LLM的语义推理通过熵自适应融合相结合，从而在保护隐私的前提下实现多轮交互中用户偏好的序列化累积推断。

    

    大语言模型（LLM）难以在多轮用户交互中积累证据，无法以符合贝叶斯推断的方式更新其信念。现有解决方案需要在敏感的用户交互数据上进行微调，这限制了其在注重隐私场景中的适用性。我们提出了AdaptFuse，这是一个无需训练的框架，它将概率计算完全从LLM中外置化：一个符号模块在离散假设集合上维护贝叶斯后验分布，而冻结的LLM则通过多样本Dirichlet聚合贡献语义推理。两种信号通过熵自适应融合机制进行结合，该机制根据各来源的预测置信度自动分配权重，并随着证据的积累，将依赖重心从LLM逐渐转移到符号后验上。我们在三个领域进行了评估：航班推荐、酒店推荐和网购；在Gemma 2 9B上，（摘要原文在此处截断）

    arXiv:2604.03925v2 Announce Type: replace  Abstract: Large language models (LLMs) struggle to accumulate evidence across multiple rounds of user interactions, failing to update their beliefs in a manner consistent with Bayesian inference. Existing solutions require fine-tuning on sensitive user interaction data, limiting their applicability in privacy-conscious settings. We propose \textsc{AdaptFuse}, a training-free framework that externalizes probabilistic computation entirely from the LLM, where a symbolic module maintains a Bayesian posterior over a discrete hypothesis set, while a frozen LLM contributes semantic reasoning via multi-sample Dirichlet aggregation. The two signals are combined through entropy-adaptive fusion, which automatically weights each source by its predictive confidence, shifting reliance from the LLM to the symbolic posterior as evidence accumulates. We evaluate across three domains: flight recommendation, hotel recommendation, and web shopping; on Gemma~2 9B,
    
[^270]: 终端智能体足以实现企业自动化

    Terminal Agents Suffice for Enterprise Automation

    [https://arxiv.org/abs/2604.00073](https://arxiv.org/abs/2604.00073)

    该论文提出仅配备终端和文件系统的编码智能体通过直接调用平台 API，就能以远低于复杂智能体架构（如 MCP 工具智能体和网页智能体）的成本，达到甚至超越它们在企业任务上的表现。

    

    构建能够与数字平台交互、自主执行有意义企业任务的智能体，正受到越来越多的关注。目前探索的方法包括基于模型上下文协议等抽象构建的工具增强型智能体，以及通过图形界面操作的网页智能体。然而，考虑到其成本和运营开销，这类复杂的智能体系统是否必要仍不明确。我们认为，仅配备终端和文件系统的编码智能体可以通过直接与平台 API 交互，更有效地解决许多企业任务。我们在多样化的真实世界系统中对这一假设进行了评估，结果表明这些低级终端智能体只需极低的成本即可达到或超越更复杂的智能体架构。我们的发现表明，简单灵活的编程接口与强大的基础模型相结合，应当成为企业自动化的核心支柱。

    arXiv:2604.00073v4 Announce Type: replace-cross  Abstract: There has been growing interest in building agents that can interact with digital platforms to execute meaningful enterprise tasks autonomously. Among the approaches explored are tool-augmented agents built on abstractions such as Model Context Protocol (MCP) and web agents that operate through graphical interfaces. Yet, it remains unclear whether such complex agentic systems are necessary given their cost and operational overhead. We argue that a coding agent equipped only with a terminal and a filesystem can solve many enterprise tasks more effectively by interacting directly with platform APIs. We evaluate this hypothesis across diverse real-world systems and show that these low-level terminal agents match or outperform more complex agent architectures at a fraction of the cost. Our findings suggest that simple, flexible programmatic interfaces combined with strong foundation models should be the backbone of enterprise autom
    
[^271]: PLACID：用于临床缩写词推断与消歧的隐私保护大语言模型

    PLACID: Privacy-preserving Large language models for Acronym Clinical Inference and Disambiguation

    [https://arxiv.org/abs/2603.23678](https://arxiv.org/abs/2603.23678)

    本研究提出一种完全本地部署的隐私保护级联流水线，利用通用小模型检测临床缩写词并交由生物医学领域专用模型进行上下文相关的消歧扩展，从而在保障患者隐私的前提下实现临床缩写词的准确解读。

    

    大语言模型（LLMs）在许多领域提供了变革性的解决方案，但其在医疗保健领域的集成受到严格的数据隐私限制的阻碍。临床叙述文本中充满大量歧义性缩写词，对这些缩写的错误解读可能引发严重后果，例如危及生命的用药错误。尽管依赖云端的大语言模型在缩写词消歧方面表现出色，但将受保护的健康信息传输到外部服务器违反了隐私法规。为弥合这一差距，本研究率先评估了完全部署在本地设备上的小参数模型，以确保隐私保护。我们提出了一种隐私保护的级联流水线，利用通用本地模型检测临床缩写词，并将其路由到特定领域的生物医学模型中进行与上下文相关的扩展。结果显示，虽然通用指令遵循模型达到了较高的检测准确率（约0.988），但其扩展能力……（摘要在此处被截断）

    arXiv:2603.23678v2 Announce Type: replace  Abstract: Large Language Models (LLMs) offer transformative solutions across many domains, but healthcare integration is hindered by strict data privacy constraints. Clinical narratives are dense with ambiguous acronyms, misinterpretation these abbreviations can precipitate severe outcomes like life-threatening medication errors. While cloud-dependent LLMs excel at Acronym Disambiguation, transmitting Protected Health Information to external servers violates privacy frameworks. To bridge this gap, this study pioneers the evaluation of small-parameter models deployed entirely on-device to ensure privacy preservation. We introduce a privacy-preserving cascaded pipeline leveraging general-purpose local models to detect clinical acronyms, routing them to domain-specific biomedical models for context-relevant expansions. Results reveal that while general instruction-following models achieve high detection accuracy (~0.988), their expansion capabili
    
[^272]: LoopRPT：面向循环语言模型的强化预训练

    LoopRPT: Reinforcement Pre-Training for Looped Language Models

    [https://arxiv.org/abs/2603.19714](https://arxiv.org/abs/2603.19714)

    LoopRPT提出了一种专为循环语言模型设计的强化预训练框架，通过将下一词元预测重构为推理任务并把强化信号直接作用于潜空间计算步骤，使循环模型能以更少的迭代次数实现更高质量的隐式推理。

    

    循环语言模型通过迭代式潜空间计算来精炼内部表示，为显式的思维链推理提供了一种替代方案。然而，现有的强化学习范式主要针对输出词元，这与推理以隐式方式展开的循环架构存在结构性错配。在本工作中，我们提出了LoopRPT，一个专为循环语言模型量身定制的强化预训练框架。通过将下一词元预测重新构建为下一词元推理任务，LoopRPT利用EMA教师参考和带噪声的潜空间轨迹，将强化信号直接分配到潜空间计算步骤上。这一构建方式使强化学习能够直接塑造中间表示，将有效推理压缩至更少的迭代次数中。我们在多种模型规模的Ouro模型上实例化了LoopRPT。结果表明，LoopRPT提升了每一步表示的质量，并在准确率-计算量的权衡中实现了帕累托最优。

    arXiv:2603.19714v2 Announce Type: replace  Abstract: Looped language models (LoopLMs) perform iterative latent computation to refine internal representations, offering an alternative to explicit chain-of-thought (CoT) reasoning. However, existing reinforcement learning (RL) paradigms primarily target output tokens, creating a structural mismatch with looped architectures whose reasoning unfolds implicitly. In this work, we propose LoopRPT, a reinforcement pre-training framework tailored for LoopLMs. By reframing next-token prediction as a next-token reasoning task, LoopRPT assigns reinforcement signals directly to latent steps using an EMA teacher reference and noisy latent rollouts. This formulation enables RL to directly shape intermediate representations, compressing effective reasoning into fewer iterations. We instantiate LoopRPT on Ouro across multiple model scales. Results show that LoopRPT improves per-step representation quality and achieves Pareto dominance in accuracy-comput
    
[^273]: BenchPreS：面向持久记忆大语言模型的上下文感知个性化偏好选择性的基准测试

    BenchPreS: A Benchmark for Context-Aware Personalized Preference Selectivity of Persistent-Memory LLMs

    [https://arxiv.org/abs/2603.16557](https://arxiv.org/abs/2603.16557)

    该论文提出BenchPreS基准，用于评估具有持久记忆的大语言模型能否在不同交流情境下恰当地应用或抑制用户偏好，发现即使是前沿模型也倾向于将用户偏好当作全局规则而非情境依赖的信号来执行，且偏好依从性越强过度应用率越高。

    

    大语言模型（LLM）越来越多地将用户偏好存储在持久记忆中，以支持跨交互的个性化服务。然而，在受社会和制度规范约束的第三方交流场景中，某些用户偏好可能并不适合被应用。我们提出了BenchPreS基准，用于评估基于记忆的用户偏好在不同交流情境中是被适当地应用还是被抑制。通过两个互补的指标——误用率和适当应用率，我们发现即使是前沿的大语言模型也难以以情境敏感的方式应用偏好。偏好依从性更强的模型表现出更高的过度应用率，且无论是推理能力还是基于提示词的防御机制都无法完全解决这一问题。这些结果表明，当前的大语言模型将个性化偏好视为全局可执行的规则，而非依赖情境的规范性信号。

    arXiv:2603.16557v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) increasingly store user preferences in persistent memory to support personalization across interactions. However, in third-party communication settings governed by social and institutional norms, some user preferences may be inappropriate to apply. We introduce BenchPreS, which evaluates whether memory-based user preferences are appropriately applied or suppressed across communication contexts. Using two complementary metrics, Misapplication Rate (MR) and Appropriate Application Rate (AAR), we find even frontier LLMs struggle to apply preferences in a context-sensitive manner. Models with stronger preference adherence exhibit higher rates of over-application, and neither reasoning capability nor prompt-based defenses fully resolve this issue. These results suggest current LLMs treat personalized preferences as globally enforceable rules rather than as context-dependent normative signals.
    
[^274]: 推理陷阱——逻辑推理作为通向情境感知的机制性路径

    The Reasoning Trap -- Logical Reasoning as a Mechanistic Pathway to Situational Awareness

    [https://arxiv.org/abs/2603.09200](https://arxiv.org/abs/2603.09200)

    该论文提出RAISE框架，揭示提升大语言模型的演绎、归纳与溯因逻辑推理能力会经由三条机制性路径逐步促成AI的情境感知能力，警示逻辑推理研究与AI安全之间的冲突风险。

    

    情境感知是指AI系统认识自身本质、理解其训练与部署环境、并对自身处境进行策略性推理的能力，被广泛认为是先进AI系统中最危险的新兴能力之一。与此同时，越来越多的研究致力于提升大语言模型（LLM）在演绎、归纳和溯因三方面的逻辑推理能力。本文认为，这两条研究轨迹正在走向碰撞。我们提出了RAISE框架（Reasoning Advancing Into Self Examination，推理进阶为自我审视），该框架识别出三条机制性路径，通过这些路径，逻辑推理能力的提升能够使AI逐步获得更深层次的情境感知：演绎式自我推断、归纳式情境识别和溯因式自我建模。我们对每条路径进行了形式化，并构建了一条从基础自我识别到策略性（此处摘要原文不完整）

    arXiv:2603.09200v2 Announce Type: replace-cross  Abstract: Situational awareness, the capacity of an AI system to recognize its own nature, understand its training and deployment context, and reason strategically about its circumstances, is widely considered among the most dangerous emergent capabilities in advanced AI systems. Separately, a growing research effort seeks to improve the logical reasoning capabilities of large language models (LLMs) across deduction, induction, and abduction. In this paper, we argue that these two research trajectories are on a collision course. We introduce the RAISE framework (Reasoning Advancing Into Self Examination), which identifies three mechanistic pathways through which improvements in logical reasoning enable progressively deeper levels of situational awareness: deductive self inference, inductive context recognition, and abductive self modeling. We formalize each pathway, construct an escalation ladder from basic self recognition to strategic 
    
[^275]: TimeWarp：通过重访过去来评估网页智能体

    TimeWarp: Evaluating Web Agents by Revisiting the Past

    [https://arxiv.org/abs/2603.04949](https://arxiv.org/abs/2603.04949)

    提出了模拟网页UI演变的TimeWarp基准测试，揭示现有网页智能体对界面变化的脆弱性，并通过TimeTraj标注方法与BC变体训练显著提升了智能体跨网页版本的泛化能力。

    

    随着网页智能体在基准测试中逐渐缩小与人类的差距，一个问题随之而来：今天的智能体在明天的网页上是否也能表现得同样出色？我们提出了TimeWarp，一个模拟不断演变的网页环境的基准测试。TimeWarp包含三个网页环境，每个环境有六个UI版本，涵盖了互联网不同时代的UI设计、前端代码和工作流程。我们为TimeWarp配套了一系列复杂、真实的任务，涵盖不同形式的网页导航。我们的实验表明，基于视觉的智能体容易受到网页变化的影响，而基于文本的智能体一旦在单一版本上微调就会变得脆弱。为了解决这个问题，我们提出了TimeTraj，一种新的标注方法，通过计划蒸馏来收集跨多个版本的轨迹。通过使用我们的BC变体方法在教师rollout上训练智能体，我们取得了显著的性能提升：Qwen-3 4B模型从20.4%提升到37.7%，Llama-3.1 8B模型从0%提升到27.0%。

    arXiv:2603.04949v2 Announce Type: replace  Abstract: As web agents close the gap with humans on benchmarks, one question arises: Do today's agents perform just as well on tomorrow's web? We introduce TimeWarp, a benchmark that emulates the evolving web. TimeWarp consists of three web environments, each with six UI versions spanning UI design, frontend code, and workflows from different eras of the internet. We pair TimeWarp with a set of complex, realistic tasks covering different forms of web navigation. Our experiments reveal that vision-based agents are vulnerable to changes, while text-based agents become brittle once fine-tuned on a single version. To address this, we propose TimeTraj, a new annotation method that uses plan distillation to collect trajectories across multiple versions. By training agents on teacher rollouts using our BC-variant, we achieve substantial performance gains: 20.4% to 37.7% for Qwen-3 4B and 0% to 27.0% for Llama-3.1 8B models. Our work helps study gene
    
[^276]: 无功能等价的几何对齐：语音-文本模态差距的逐层分析

    Geometric Alignment without Functional Equivalence: A Layer-wise Analysis of the Speech-Text Modality Gap

    [https://arxiv.org/abs/2603.01502](https://arxiv.org/abs/2603.01502)

    该研究通过对比跨层CKA表征对齐与逐层答案裕度，发现语音大语言模型中语音与文本表征在中间层虽实现几何对齐，但后期层答案形成方式仍存在差异，证明几何对齐并不等于功能等价。

    

    端到端语音大语言模型（LLM）在从语音回答同一问题时，其准确率往往低于从文本回答，即使语音内容与文本提示完全一致。现有分析通过表征的几何结构来研究这种模态差距，并衡量语音表征与文本表征的接近程度。几何对齐使这一差距具有可测量的形式，但它并未表明模型是否以相同方式使用已对齐的表征。本工作将表征几何与答案形成过程联系起来，探究几何对齐是否意味着功能等价。我们在 SpeechMMLU 和 VoiceBench BBH 上，对四个开源权重语音大语言模型比较了跨层中心化核对齐（CKA）与逐层答案裕度。在 Qwen2.5-Omni 和 MiniCPM-o 中，语音与文本表征在中间层实现对齐，但它们的后期层答案裕度仍然不同，因此中间层的对齐并不意味着功能等价。

    arXiv:2603.01502v2 Announce Type: replace  Abstract: End-to-end speech large language models (LLMs) often answer the same question less accurately from speech than from text, even when the spoken content matches the text prompt. Existing analyses study this modality gap through the geometry of representations and measure how closely speech representations align with text. Geometric alignment gives the gap a measurable form, but it does not show whether the model uses aligned representations in the same way. This work connects representation geometry with answer formation and asks whether geometric alignment implies functional equivalence. We compare cross-layer Centered Kernel Alignment (CKA) with layer-wise answer margins for four open-weight speech LLMs on SpeechMMLU and VoiceBench BBH. In Qwen2.5-Omni and MiniCPM-o, speech and text representations align in middle layers while their late-layer answer margins still differ, so mid-layer alignment does not imply functional equivalence. 
    
[^277]: 草稿思维：在长思维链大语言模型中学习高效推理

    Draft-Thinking: Learning Efficient Reasoning in Long Chain-of-Thought LLMs

    [https://arxiv.org/abs/2603.00578](https://arxiv.org/abs/2603.00578)

    提出轻量级适配框架Draft-Thinking，通过渐进式草稿适配学习模型适配的紧凑草稿思维链，解决强教师模型简洁推理轨迹对小模型过于稀疏的跨尺度不匹配问题，从而在保持解题能力的同时显著降低长思维链的推理成本。

    

    长思维链推理提升了大型推理模型的解题能力，但往往会带来高昂的推理成本和过度思考问题。现有的效率优化方法主要依赖提示工程、事后压缩或长度正则化优化；这些方法可能对模型规模敏感、依赖外部估计器，或者对困难问题分配的推理量不足。我们提出了**草稿思维**，这是一个轻量级适配框架，用于学习*模型适配的草稿思维链*：一种紧凑、自包含的工作推导过程，可直接支撑最终答案的生成。我们识别出一个**跨尺度草稿不匹配**问题：由强教师模型生成的简洁推理轨迹对于较小的目标模型而言可能过于稀疏。为解决这一不匹配问题，**渐进式草稿适配**方法首先通过监督微调（SFT）迁移简洁的推理形式先验，然后应用两阶段……（摘要原文截断于此）

    arXiv:2603.00578v2 Announce Type: replace-cross  Abstract: Long chain-of-thought~(CoT) reasoning improves the problem-solving capability of large reasoning models~(LRMs), but often incurs substantial inference cost and overthinking. Existing efficiency methods primarily rely on prompting, post-hoc compression, or length-regularized optimization; these approaches can be sensitive to model scale, depend on external estimators, or under-allocate reasoning to difficult instances. We propose \textbf{Draft-Thinking}, a lightweight adaptation framework that learns a \textit{model-adapted draft CoT}: a compact, self-contained working derivation that directly supports the final answer. We identify a \textbf{Cross-Scale Draft Mismatch}: concise traces generated by a strong teacher can be too sparse for a smaller target model. To address this mismatch, \textbf{Progressive Draft Adaptation} first transfers a concise reasoning-form prior through supervised fine-tuning (SFT) and then applies two-sta
    
[^278]: 韩流音乐（K-pop）中的语义社群与跨边界歌词：一种基于图的无监督分析

    Semantic Communities and Boundary-Spanning Lyrics in K-pop: A Graph-Based Unsupervised Analysis

    [https://arxiv.org/abs/2602.12881](https://arxiv.org/abs/2602.12881)

    该研究利用多语言MiniLM嵌入对7,891首K-pop歌曲构建加权最近邻图并通过Louvain算法识别出21个语义社群，发现整首歌嵌入提供最佳的相邻歌曲区分度，且歌词主题呈现记忆、分离、渴望等跨社群的重叠模式。

    

    我们研究了歌词表示方式如何影响计算相似度，使用了四种384维的多语言MiniLM表示方法，对象涵盖2000年至2023年10月排行榜中1,456位艺术家的7,891首K-pop歌曲。整首歌截断嵌入在相邻歌曲对与随机歌曲对之间提供了最强的标准化区分度（1.995），相比之下分块文档为1.931，尽管分块表示显示出更高的调整兰德指数一致性，而基于行的表示则产生更高的模块度。利用整首歌嵌入，我们构建了一个加权余弦最近邻图（k=30）。代表性的Louvain划分识别出21个操作性社群，但马尔可夫稳定性分析中并未出现相应的稳健平台期。对选定歌词和术语的AI辅助分析表明，歌词中存在记忆、分离、渴望、自我反思、吸引与表演等重叠的主题模式。社群成员关系与...

    arXiv:2602.12881v2 Announce Type: replace-cross  Abstract: We examine how lyric representation shapes computational similarity using four 384-dimensional multilingual MiniLM representations of 7,891 K-pop songs by 1,456 artists from charts spanning 2000 to October 2023. Whole-song truncated embeddings provide the strongest standardized separation between neighbouring and random song pairs (1.995), compared with 1.931 for chunked documents, although chunked representations show greater adjusted Rand agreement and line-based representations yield higher modularity. Using whole-song embeddings, we construct a weighted cosine nearest-neighbour graph ($k=30$). A representative Louvain partition identifies 21 operational communities, without a corresponding robust plateau in the Markov Stability analysis. AI-assisted examination of selected lyrics and terms suggests overlapping patterns of memory, separation, longing, self-reflection, attraction, and performance. Community membership is asso
    
[^279]: 何时足够还不够？搜索智能体中的虚幻完成

    When Is Enough Not Enough? Illusory Completion in Search Agents

    [https://arxiv.org/abs/2602.07549](https://arxiv.org/abs/2602.07549)

    该研究揭示了搜索智能体中的“虚幻完成”现象——即使答案正确，智能体也可能在某个约束条件未经验证时就认为任务已完成（在正确答案中占比高达48%），并提出“认知账本”来追踪每轮检索对各约束条件的确立情况，发现训练和规模只能改变而非消除验证失败的模式。

    

    在智能体搜索中，大语言模型智能体会搜索网络、阅读找到的页面，并决定接下来要查找什么，然后返回答案。但我们能仅凭智能体返回了答案就信任它吗？往往不能，即使是正确的答案也可能只是侥幸猜中：在包含多个约束条件的问题上，我们发现智能体在多达48%的正确答案中，会在某个约束条件尚未得到验证的情况下就断定任务已完成。我们将这种现象称为“虚幻完成”。为了探究其产生机制，我们提出了认知账本，它在每一轮中跟踪检索到的页面针对每个约束条件确立了什么信息，以及智能体声称了什么。在从7B强化学习训练模型到前沿大语言模型的13个智能体中，训练和规模提升了准确率，但只是改变了验证失败的模式，而非消除这些失败：约束条件可能未被检查、在没有依据的情况下被假定成立，或在存在反驳证据的情况下仍被保留。

    arXiv:2602.07549v3 Announce Type: replace-cross  Abstract: In agentic search, an LLM agent searches the web, reads the pages it finds, and decides what to look for next before returning an answer. But can we trust an answer simply because the agent returns it? Often not, and even a correct answer can be a lucky guess: on questions with several constraints, we find that agents conclude the task is complete while a constraint remains unverified in up to 48% of their correct answers. We call this illusory completion. To see how it arises, we introduce the Epistemic Ledger, which tracks at every turn what the retrieved pages establish about each constraint and what the agent claims. Across 13 agents, from 7B RL-trained models to frontier LLMs, training and scale raise accuracy but change the pattern of verification failures rather than eliminating them: constraints may be left unchecked, assumed without support, or retained despite refuting evidence. To measure what agents lose without tra
    
[^280]: 面向多语言上下文学习的语言引导

    Language Steering for Multilingual In-Context Learning

    [https://arxiv.org/abs/2602.02326](https://arxiv.org/abs/2602.02326)

    该论文提出“语言向量”——通过在特定层计算平行语言示例间的平均激活差异，并在推理时作为偏移量注入模型隐藏状态，无需任何参数更新即可显著提升多语言上下文学习中的跨语言迁移性能，且这些向量呈现出可解释的语言学聚类结构。

    

    如果多语言语言模型的表示中包含可分离的与语言相关的结构，那么在语言之间切换可能只需要简单的激活空间干预。为了验证这一假设，我们将多语言上下文学习（ICL）作为一个受控的研究场景：少样本示例以英语提供，而测试查询使用目标语言，从而构成一个明确的跨语言迁移挑战。我们提出了“语言向量”（language vectors），其通过在特定层上计算平行源语言与目标语言示例之间的平均激活差异得到，并在推理时作为偏移量应用于模型的隐藏状态。在涵盖19种语言的三项任务和三个模型上，我们的方法无需任何参数更新即可超越英语少样本提示的表现。除性能提升外，这些向量还编码了可解释的语言学结构：亲缘关系相近的语言形成紧密的聚类，并且向量能够有效地跨模……（原文摘要在此处截断）

    arXiv:2602.02326v3 Announce Type: replace  Abstract: If multilingual language model representations contain separable language-associated structure, then shifting between languages may require only simple activation-space interventions. To test this hypothesis, we study multilingual in-context learning (ICL) as a controlled setting: few-shot demonstrations are provided in English while the test query is in a target language, creating a clear cross-lingual transfer challenge. We propose language vectors, which are computed as mean activation differences between parallel source and target language examples at a particular layer and applied as inference-time offsets to the model's hidden states. Across three tasks spanning 19 languages and three models, our method improves over English few-shot prompting without any parameter updates. Beyond performance, the vectors encode interpretable linguistic structure: closely related languages form tight clusters, and vectors transfer effectively a
    
[^281]: MasalBench：用于评估大语言模型对波斯谚语上下文与跨文化理解能力的基准测试

    MasalBench: A Benchmark for Contextual and Cross-Cultural Understanding of Persian Proverbs in LLMs

    [https://arxiv.org/abs/2601.22050](https://arxiv.org/abs/2601.22050)

    该论文提出MasalBench基准测试，发现大语言模型虽然能很好地识别语境中的波斯谚语（准确率超过0.90），但在识别对应英语谚语时性能大幅下降（最佳模型仅为0.79），揭示了LLM在低资源语言跨文化理解方面的不足。

    

    近年来，多语言大语言模型（LLMs）已成为日常生活中不可或缺的一部分，因此它们必须掌握会话语言的规则，才能与用户进行有效沟通。尽管以往的研究已评估了LLM对高资源语言中比喻性语言的理解，但其在低资源语言中的表现仍未得到充分探索。本文提出了MasalBench，这是一个全面的基准测试，用于评估大语言模型对波斯谚语的上下文和跨文化理解能力，而波斯谚语正是这种低资源语言日常会话中的关键组成部分。我们在MasalBench上评估了八个最先进的大语言模型，发现它们在识别语境中的波斯谚语方面表现良好，准确率超过0.90。然而，当任务要求识别对应的英语谚语时，它们的表现显著下降，其中表现最好的模型准确率仅为0.79。

    arXiv:2601.22050v2 Announce Type: replace  Abstract: In recent years, multilingual Large Language Models (LLMs) have become an inseparable part of daily life, making it crucial for them to master the rules of conversational language in order to communicate effectively with users. While previous work has evaluated LLMs' understanding of figurative language in high-resource languages, their performance in low-resource languages remains underexplored. In this paper, we introduce MasalBench, a comprehensive benchmark for assessing LLMs' contextual and cross-cultural understanding of Persian proverbs, which are a key component of conversation in this low-resource language. We evaluate eight state-of-the-art LLMs on MasalBench and find that they perform well in identifying Persian proverbs in context, achieving accuracies above 0.90. However, their performance drops considerably when tasked with identifying equivalent English proverbs, with the best model achieving 0.79 accuracy. Our finding
    
[^282]: SWE-Spot：通过仓库中心化学习构建小型仓库专家模型

    SWE-Spot: Building Small Repo-Experts with Repository-Centric Learning

    [https://arxiv.org/abs/2601.21649](https://arxiv.org/abs/2601.21649)

    本文提出仓库中心化学习（RCL）新范式，将静态代码库转化为交互式学习信号，训练出针对特定代码仓库深度专门化的小型语言模型SWE-Spot-4B，以弥补小型模型处理复杂陌生代码库时泛化能力不足的缺陷。

    

    在隐私敏感和资源受限环境中部署编码智能体的需求，推动了对高性能开源权重小型语言模型（SLM）的需求。然而，这些模型存在根本性的能力差距：与前沿大模型不同，它们缺乏在推理时应对复杂、陌生代码库的强泛化能力。我们发现当前流行的任务中心学习（TCL）方法——即在不同仓库间扩大任务暴露规模——无法解决这一局限。为此，我们提出仓库中心化学习（RCL），这是一种范式转变：在SLM训练中优先追求纵向的单仓库深度，而非横向的单任务广度。为实现这种深度，我们设计了由四个单元组成的仓库中心化经验，将静态代码库转化为交互式学习信号，用于训练SWE-Spot-4B——一系列作为仓库专门专家构建的高度紧凑模型。在受控对比实验中……

    arXiv:2601.21649v2 Announce Type: replace-cross  Abstract: The deployment of coding agents in privacy-sensitive and resource-constrained environments drives the demand for capable open-weight Small Language Models (SLMs). However, they suffer from a fundamental capability gap: unlike frontier large models, they lack the inference-time strong generalization to handle complicated, unfamiliar codebases. We identify that the prevailing Task-Centric Learning (TCL) approach, which scales exposure across disparate repositories, fails to address this limitation. In response, we propose Repository-Centric Learning (RCL), a shift that prioritizes vertical, single-repository depth over horizontal, single-task breadth for SLM training. To accomplish this depth, we design a four-unit Repository-Centric Experience, transforming static codebases into interactive learning signals, to train SWE-Spot-4B, a family of highly compact models built as repo-specialized experts. Under controlled comparisons wi
    
[^283]: CALM-IT：通过会话动态提升合成动机式访谈对话质量

    CALM-IT: Enhancing Synthetic Motivational Interviewing Dialogue Quality with Conversational Dynamics

    [https://arxiv.org/abs/2601.10085](https://arxiv.org/abs/2601.10085)

    提出双智能体框架CALM-IT，通过追踪来访者状态与治疗师侧推断来建模会话动态变化，从而显著提升长篇合成动机式访谈治疗对话的质量。

    

    会话动态描述了融洽关系、情绪和目标在对话过程中如何随时间变化。这些变化在治疗性对话中至关重要，因为它们会影响治疗师介入的时机和方式。然而，基于大语言模型（LLM）的合成治疗性对话生成器很少追踪这些属性如何共同演变，这可能限制了较长对话的质量。我们提出了CALM-IT，这是一个双智能体框架，通过追踪来访者状态和治疗师侧的推断来指导策略选择与话语生成。我们使用自动化的动机式访谈治疗完整性（MITI）评分、会话指标以及不同对话长度下的比较来评估合成的动机式访谈（MI）对话，四位持证心理学家的评分为主要发现提供了支持。在涵盖686个真实场景、四个框架和三种目标长度的8,232个对话中，CALM-IT...

    arXiv:2601.10085v3 Announce Type: replace  Abstract: Conversational dynamics describe how rapport, emotion, and goals change over a dialogue. These changes are important in therapeutic conversations, where they can influence when and how therapists intervene. However, large language model (LLM)-based synthetic therapeutic dialogue generators rarely track how these properties evolve together, which may limit the quality of longer conversations. We introduce CALM-IT, a dual-agent framework that tracks client states and therapist-side inferences to guide strategy selection and utterance generation. We evaluate synthetic Motivational Interviewing (MI) dialogues using automated Motivational Interviewing Treatment Integrity (MITI) ratings, conversational metrics, and comparisons across dialogue lengths, with ratings from four licensed psychologists providing support for key findings. Across 8,232 dialogues covering 686 grounded scenarios, four frameworks, and three target lengths, CALM-IT ac
    
[^284]: QuantEval：面向大语言模型金融量化任务的基准测试

    QuantEval: A Benchmark for Financial Quantitative Tasks in Large Language Models

    [https://arxiv.org/abs/2601.08689](https://arxiv.org/abs/2601.08689)

    该论文提出了QuantEval基准，通过涵盖知识问答、量化数学推理和量化策略编程三个维度的1,575个专家验证实例，并采用确定性CTA回测框架实际执行模型生成的策略，系统评估了13个大语言模型完成量化金融“计算—实现—验证”闭环的能力，发现模型与人类专家之间仍存在显著差距。

    

    大语言模型（LLM）在面向金融的问答任务中已展现出强大的能力，但其在专业量化工作流程上的评估仍然较为碎片化，且往往止步于事实性或基于文档的问答。我们提出了QuantEval，这是一个用于评估大语言模型能否完成量化金融中所需的“计算—实现—验证”闭环的基准测试。QuantEval包含1,575个经专家验证的实例，涵盖三个互补的维度：(i) 基于知识的问答，(ii) 量化数学推理，以及 (iii) 量化策略编程。编程任务采用确定性的CTA风格回测框架进行评估，该框架会实际执行模型生成的策略，并将其风险—收益指标与专家参考基准进行比较，从而超越了单纯的语法或模板匹配。我们评估了13个开源及专有大语言模型，报告了提示词与可靠性分析，并观察到模型与人类专家之间仍存在显著差距。

    arXiv:2601.08689v3 Announce Type: replace  Abstract: Large Language Models (LLMs) have shown strong capabilities in finance-oriented question answering, yet their evaluation on professional quantitative workflows remains fragmented and often stops at factual or document-grounded QA. We introduce QuantEval, a benchmark for evaluating whether LLMs can complete the compute--implement--validate loop required in quantitative finance. QuantEval contains 1,575 expert-validated instances across three complementary dimensions: (i) knowledge-based QA, (ii) quantitative mathematical reasoning, and (iii) quantitative strategy coding. The coding task is evaluated with a deterministic CTA-style backtesting framework that executes model-generated strategies and compares risk--return metrics against expert references, moving beyond syntax or template matching. We evaluate 13 open-source and proprietary LLMs, report prompt and reliability analyses, and observe substantial gaps relative to human experts
    
[^285]: 面向自动化奖励设计的双重进化强化学习

    Dual-Evolving Reinforcement Learning for Automated Reward Design

    [https://arxiv.org/abs/2512.13399](https://arxiv.org/abs/2512.13399)

    本文提出双重进化强化学习框架DERL，通过让外循环奖励提议者从内层策略的验证表现中学习元梯度，实现奖励结构的自主发现与持续优化，在具身、科学和数学推理任务上显著超越基线，尤其在分布外任务上表现突出。

    

    构建有效的奖励信号仍然是强化学习（RL）在复杂推理任务中的核心挑战。现有的自动化奖励优化方法使用固定的提议者进行搜索，而该提议者从不从其代价高昂的试验中学习。我们提出了双重进化强化学习（Dual-Evolving Reinforcement Learning, DERL），这是一个用于自主发现奖励结构的双重训练框架。外循环的提议者从任务基元中组合奖励，内层策略则使用该奖励进行训练。提议者本身通过策略梯度从内层策略的验证性能中学习。因此，DERL能够利用过去的试验反馈来塑造其奖励提议分布，这构成了任务成功的经验性元梯度。我们在具身推理（ALFWorld）、科学推理（ScienceWorld）和数学推理（GSM8K、MATH）上验证了DERL。在多轮智能体基准测试中，DERL超越了强大的基线方法，在分布外任务上的优势尤为显著。在数学领域，DE……（原文摘要在此处截断）

    arXiv:2512.13399v3 Announce Type: replace-cross  Abstract: Crafting effective reward signals remains a central challenge in reinforcement learning (RL) for complex reasoning. Existing methods for automated reward optimization search with a fixed proposer which never learns from its costly trials. We present Dual-Evolving Reinforcement Learning (DERL), a dual-training framework for autonomous discovery of reward structures. An outer-loop proposer composes a reward from task primitives, and an inner policy trains with this reward. The proposer itself learns by policy gradient from the validation performance of the inner policy. Thus, DERL utilizes past-trial feedback to shape its reward-proposal distribution, an empirical meta-gradient of task success. We validate DERL on embodied (ALFWorld), scientific (ScienceWorld), and mathematical (GSM8K, MATH) reasoning. On multi-turn agent benchmarks, DERL outperforms strong baselines, significantly on out-of-distribution tasks. In mathematics, DE
    
[^286]: 安全的不稳定性：随机种子与温度如何暴露大语言模型不一致的拒绝行为

    The Instability of Safety: How Random Seeds and Temperature Expose Inconsistent LLM Refusal Behavior

    [https://arxiv.org/abs/2512.12066](https://arxiv.org/abs/2512.12066)

    该研究揭示大语言模型的安全拒绝决策在随机种子和温度变化下并不稳定，18-28%的有害提示词会出现“拒绝”与“配合”之间的决策翻转，且温度越高稳定性越差，表明单次安全评估无法真实反映模型的安全对齐水平。

    

    当前大语言模型的安全评估依赖于单次测试，隐含地假设模型响应是确定性的，并能代表模型的安全对齐水平。我们通过研究安全拒绝决策在不同随机种子和温度设置下的稳定性来挑战这一假设。我们在20种采样配置（4种温度 × 5个随机种子）下，对来自三个系列的四个指令微调模型（Llama 3.1 8B、Qwen 2.5 7B、Qwen 3 8B、Gemma 3 12B）在876个有害提示词上进行了测试，发现18-28%的提示词表现出决策翻转——即模型在某些配置下拒绝回答，而在其他配置下则予以配合——具体比例因模型而异。我们的安全稳定性指数（SSI）显示，更高的温度会显著降低决策稳定性（Friedman卡方 = 396.81，p < 0.001），温度内平均SSI从温度0.0时的0.977下降至温度1.0时的0.942。我们在……（原文摘要在此处被截断）

    arXiv:2512.12066v3 Announce Type: replace-cross  Abstract: Current safety evaluations of large language models rely on single-shot testing, implicitly assuming that model responses are deterministic and representative of the model's safety alignment. We challenge this assumption by investigating the stability of safety refusal decisions across random seeds and temperature settings. Testing four instruction-tuned models from three families (Llama 3.1 8B, Qwen 2.5 7B, Qwen 3 8B, Gemma 3 12B) on 876 harmful prompts across 20 sampling configurations (4 temperatures x 5 seeds), we find that 18-28% of prompts exhibit decision flips--the model refuses in some configurations but complies in others--depending on the model. Our Safety Stability Index (SSI) reveals that higher temperatures significantly reduce decision stability (Friedman chi-squared = 396.81, p < 0.001), with mean within-temperature SSI dropping from 0.977 at temperature 0.0 to 0.942 at temperature 1.0. We validate findings acro
    
[^287]: 用于更好训练和评估知识图谱增强大语言模型的真值子图

    Ground-Truth Subgraphs for Better Training and Evaluation of Knowledge Graph Augmented LLMs

    [https://arxiv.org/abs/2511.04473](https://arxiv.org/abs/2511.04473)

    提出了SynthKGQA框架，可从任意知识图谱生成包含完整真值事实的知识图谱问答数据集，既能更有效地评估知识图谱检索器，也能用于训练更好的知识图谱增强大语言模型，并基于Wikidata构建了测试零样本泛化能力的GTSQA数据集。

    

    从图结构知识库中检索信息是提高大语言模型事实性的一个有前景的方向。尽管已提出多种解决方案，但由于缺乏带有图检索真值目标的具有挑战性的问答数据集，方法之间的比较十分困难。我们提出了SynthKGQA，这是一个由大语言模型驱动的框架，可以从任何知识图谱生成高质量的知识图谱问答数据集，并提供知识图谱中用于推理问题的完整真值事实集合。我们展示了SynthKGQA生成的数据除了能够对知识图谱检索器进行更有信息量的基准测试外，还能用于训练更好的模型。我们将SynthKGQA应用于Wikidata，生成了GTSQA——一个旨在测试知识图谱检索器对未见过的图结构和关系类型的零样本泛化能力的新数据集，并在其上对流行的知识图谱增强大语言模型解决方案进行了基准测试。

    arXiv:2511.04473v3 Announce Type: replace-cross  Abstract: Retrieval of information from graph-structured knowledge bases represents a promising direction for improving the factuality of LLMs. While various solutions have been proposed, a comparison of methods is difficult due to the lack of challenging QA datasets with ground-truth targets for graph retrieval. We present SynthKGQA, an LLM-powered framework for generating high-quality Knowledge Graph Question Answering datasets from any Knowledge Graph, providing the full set of ground-truth facts in the KG to reason over questions. We show how, in addition to enabling more informative benchmarking of KG retrievers, the data produced with SynthKGQA also allows us to train better models.We apply SynthKGQA to Wikidata to generate GTSQA, a new dataset designed to test zero-shot generalization abilities of KG retrievers with respect to unseen graph structures and relation types, and benchmark popular solutions for KG-augmented LLMs on it.
    
[^288]: 利用规则引导的大语言模型重构回收低质量指令数据

    Reclaiming Low-Quality Instruction Data with Rule-Guided LLM Reconstruction

    [https://arxiv.org/abs/2510.23160](https://arxiv.org/abs/2510.23160)

    提出一种规则引导的LLM重构框架，将SFT数据选择中通常被丢弃的低分指令样本转化为高质量训练数据，在三个7B/8B模型上以远少于全量数据的样本数（约5.5万条）超越了全量训练和数据选择基线的性能。

    

    针对监督微调（SFT）的数据选择方法通常会丢弃低分的指令样本，而这些样本可能仍然保留有用的监督信息。我们提出了一种规则引导的大语言模型（LLM）重构框架来回收这些信息。该框架通过基于LLM的评分和分数修正来识别低分样本，选取语义上具有代表性的源数据，并通过策略引导的生成、基于标准的验证以及失败引导的修复来重构指令-回复对。该框架从123,786个低分样本中构建了54,888个训练样本。在三个7B/8B基座大语言模型上，重构数据取得了比低分源数据集以及所有被评估的数据选择基线更高的平均下游性能。尽管使用的样本数量远少于完整源数据集（约30万条），它在所有三个基座模型上都取得了比全量数据训练更高的报告平均分数。逐步消融实验……

    arXiv:2510.23160v2 Announce Type: replace  Abstract: Data selection for supervised fine-tuning (SFT) often discards low-scoring instruction examples that may retain useful supervision. We propose a rule-guided LLM reconstruction framework to reclaim this information. The framework identifies low-scoring samples through LLM-based rating and score correction, selects semantically representative sources, and reconstructs instruction-response pairs through strategy-guided generation, criteria-based validation, and failure-guided repair. From 123,786 low-scoring samples, it constructs 54,888 training examples. Across three 7B/8B base LLMs, the reconstructed data achieves higher average downstream performance than the low-scoring source set and all evaluated data-selection baselines. Despite using substantially fewer examples than the full source dataset (approximately 300K), it achieves higher reported average scores than full-data training across all three base models. Stepwise ablations i
    
[^289]: 大语言模型也会“脑腐化”：一项基于Twitter/X的初步研究

    LLMs Can Get "Brain Rot": A Pilot Study on Twitter/X

    [https://arxiv.org/abs/2510.13928](https://arxiv.org/abs/2510.13928)

    本研究首次提出并通过Twitter/X真实语料的对照实验验证了“LLM脑腐化假说”，证明持续预训练低质量垃圾网络文本会导致大语言模型在推理、长上下文理解和安全性上出现类似剂量-反应关系的持久认知衰退。

    

    我们提出并验证了“大语言模型脑腐化假说”：持续暴露于垃圾网络文本会导致大语言模型（LLM）产生持久的认知衰退。为了揭示垃圾文本的影响，我们在真实的Twitter/X语料库上设计了一项新颖的对照实验，通过两种正交的操作化方式构建了垃圾数据集和反向对照数据集：M1（互动参与度）和M2（语义质量），并确保各条件下token规模和训练操作完全一致。与对照组相比，4个大语言模型在垃圾数据集上进行持续预训练后，在推理、长上下文理解和安全性方面出现了显著下降（Hedges' g>0.3），同时“黑暗人格特质”（如精神变态、自恋）有所膨胀。垃圾数据集与对照数据集的渐进混合还产生了剂量-反应式的认知衰退：例如，在M1条件下，随着垃圾数据比例从0%升至100%，采用思维链的ARC-Challenge得分从72.1降至57.2，RULER-CWE从83.7降至52.3。错误取证分析…

    arXiv:2510.13928v3 Announce Type: replace  Abstract: We propose and test the LLM Brain Rot Hypothesis: continual exposure to junk web text induces lasting cognitive decline in large language models (LLMs). To unveil junk effects, we designed a novel controlled experiment on real Twitter/X corpora, by constructing junk and reverse-controlled datasets via two orthogonal operationalizations: M1 (engagement degree) and M2 (semantic quality), with matched token scale and training operations across conditions. Compared to the control group, continual pre-training of 4 LLMs on the junk dataset causes non-trivial declines (Hedges' g>0.3) on reasoning, long-context understanding, safety, and inflating "dark traits" (e.g., psychopathy, narcissism). The gradual mixtures of junk and control datasets also yield dose-response cognition decay: for example, under M1, ARC-Challenge with Chain-of-Thought drops 72.1 -> 57.2 and RULER-CWE 83.7 -> 52.3 as junk ratio rises from 0% to 100%.   Error forensics
    
[^290]: 一个用于大语言模型辅助语料库标注的大规模流水线：英语consider结构中的变异与变化

    A large-scale pipeline for LLM-assisted corpus annotation: variation and change in the English consider construction

    [https://arxiv.org/abs/2510.12306](https://arxiv.org/abs/2510.12306)

    本文提出了一个基于大语言模型的四阶段语料库自动标注流水线，在60小时内以98%以上的准确率完成了COHA语料库中14万余条consider索引行的语法标注，并通过英语consider结构的历时变异分析验证了其在大规模语料库语言学研究中的实用性和有效性。

    

    随着自然语言语料库以前所未有的速度扩展，人工标注仍然是语料库语言学研究中的一个重大方法论瓶颈。为应对这一挑战，我们提出了一个可扩展的流水线，利用大语言模型（LLM）实现海量语料库语法标注的自动化。与以往的监督式和迭代式方法不同，我们的方法采用四阶段工作流程：提示工程、事前评估、自动化批量处理和事后验证。我们通过对英语评价性consider结构（consider X as/to be/∅ Y）变异的历时案例研究，展示了该流水线的可操作性和有效性。我们通过OpenAI API在不到60小时的时间内，对美国历史英语语料库（COHA）中的143,933条“consider”索引行进行了标注，在两项复杂的标注任务中达到了98%以上的准确率。此外，基于44,527个真实阳性样本拟合的贝叶斯多项式广义加性模型（GAM）……

    arXiv:2510.12306v4 Announce Type: replace  Abstract: As natural language corpora expand at an unprecedented rate, manual annotation remains a significant methodological bottleneck in corpus linguistic work. We address this challenge by presenting a scalable pipeline for automating grammatical annotation in voluminous corpora using large language models (LLMs). Unlike previous supervised and iterative approaches, our method employs a four-phase workflow: prompt engineering, pre-hoc evaluation, automated batch processing, and post-hoc validation. We demonstrate the pipeline's accessibility and effectiveness through a diachronic case study of variation in the English evaluative consider construction (consider X as/to be/{\O} Y). We annotate 143,933 'consider' concordance lines from the Corpus of Historical American English (COHA) via the OpenAI API in under 60 hours, achieving 98%+ accuracy on two sophisticated annotation procedures. A Bayesian multinomial GAM fitted to 44,527 true positi
    
[^291]: EconCausal：一个面向大语言模型的情境感知经济推理基准

    EconCausal: A Context-Aware Economic Reasoning Benchmark for Large Language Models

    [https://arxiv.org/abs/2510.07231](https://arxiv.org/abs/2510.07231)

    该论文构建了EconCausal——一个基于2,595篇顶级经济学与金融学实证研究、包含10,490条情境标注因果三元组的大规模基准，用于评估大语言模型在不同情境下推断并修正经济因果效应方向的能力，并揭示了模型在跨情境预测中存在源符号过度迁移的系统性缺陷。

    

    社会经济因果效应高度依赖于其制度与环境背景。同一种干预措施在不同的监管制度、市场条件、时间段或人群中可能产生不同甚至相反的效果。这对承担决策支持角色的大语言模型（LLM）提出了挑战：它们能否在特定情境下推断因果效应的方向，并在情境发生改变时修正这一判断？为解决这一问题，我们提出了EconCausal，一个包含10,490条情境标注因果三元组的大规模基准数据集，这些数据提取自顶级经济学与金融学期刊的2,595项高质量实证研究，并通过包含多轮共识、情境细化与多评审过滤的严格四阶段流程构建而成。在对19个模型的评估中，我们观察到跨情境预测中存在源符号过度迁移现象。尽管顶尖模型在单情境预测中达到约84%的准确率，但平均准确率……（原文摘要在此处截断）

    arXiv:2510.07231v5 Announce Type: replace  Abstract: Socio-economic causal effects depend heavily on their institutional and environmental contexts. The same intervention can produce different, even opposite, effects across regulatory regimes, market conditions, time periods, or populations. This poses a challenge for large language models (LLMs) in decision-support roles: can they infer the direction of a causal effect under a specified context, and revise that judgment when the context changes?   To address this, we introduce EconCausal, a large-scale benchmark of 10,490 context-annotated causal triplets extracted from 2,595 high-quality empirical studies in top-tier economics and finance journals, constructed through a rigorous four-stage pipeline with multi-run consensus, context refinement, and multi-critic filtering.   Across 19 models, we observe source-sign over-transfer in cross-context prediction. While top models reach ~84% accuracy in single-context prediction, mean accurac
    
[^292]: Eigen智能体：基于监视器检索增强生成（RAG）的自适应多智能体科学推理

    Eigen-Agent: Adaptive Multi-Agent Scientific Reasoning with Monitor-Based RAG

    [https://arxiv.org/abs/2509.21193](https://arxiv.org/abs/2509.21193)

    提出Eigen-Agent统一框架，以token级监视器检索模块实现最小干扰的隐式知识整合，并结合分层解法精炼（HSR）与质量感知迭代推理（QAIR）两项机制，在HLE生物/化学集上取得迄今最高的48.3%准确率。

    

    大型语言模型（LLMs）近期在科学推理方面取得了显著进展，但仍存在两大瓶颈。其一，显式检索会割裂推理过程，带来额外token和步骤所构成的隐性“工具税”。其二，多智能体流水线往往通过对所有候选方案取平均而稀释了优质解法。我们提出了一个融合隐式检索与结构化协作的统一框架来应对这些挑战。该框架的基础是一个基于监视器的检索模块，它在token级别运行，以对推理干扰最小的方式整合外部知识。在此基础之上，分层解法精炼（HSR）迭代地将每个候选解指定为锚点并由同伴智能体进行修复，而质量感知迭代推理（QAIR）则根据解的质量自适应地调整精炼过程。在Humanity's Last Exam（HLE）生物/化学黄金集上，我们的框架达到了48.3%的准确率——这是迄今为止报告的最高成绩，超越了……（原文摘要至此截断）

    arXiv:2509.21193v2 Announce Type: replace  Abstract: Large language models (LLMs) have recently shown strong progress on scientific reasoning, yet two major bottlenecks remain. First, explicit retrieval fragments reasoning, imposing a hidden "tool tax" of extra tokens and steps. Second, multi-agent pipelines often dilute strong solutions by averaging across all candidates. We address these challenges with a unified framework that combines implicit retrieval and structured collaboration. At its foundation, a Monitor-based retrieval module operates at the token level, integrating external knowledge with minimal disruption to reasoning. On top of this substrate, Hierarchical Solution Refinement (HSR) iteratively designates each candidate as an anchor to be repaired by its peers, while Quality-Aware Iterative Reasoning (QAIR) adapts refinement to solution quality. On Humanity's Last Exam (HLE) Bio/Chem Gold, our framework achieves 48.3% accuracy - the highest reported to date, surpassing t
    
[^293]: 语言模型的前瞻性评估：从任务描述预测LLM基准测试分数

    Anticipatory Evaluation of Language Models: Predicting LLM Benchmark Scores from Descriptions

    [https://arxiv.org/abs/2509.20645](https://arxiv.org/abs/2509.20645)

    该论文提出“前瞻性评估”新范式和EVALCAST数据集（2,290对描述-性能数据），证明前沿LLM能够从自然语言任务描述中推断出基准测试的相对难度排序，但精确预测绝对性能分数仍然困难。

    

    现代机器学习研究日益面临评估瓶颈：构建基准测试和运行实验成本高昂，拖慢了迭代速度。我们探讨的问题是：在实验之前，仅凭对任务和实验设置的自然语言描述，能够对评估结果推断出多少信息。我们将这一设定称为“前瞻性评估”。我们提出了EVALCAST，一个基于文献构建的资源，包含来自1,519篇arXiv论文的2,290对“描述-性能”数据，其中包括由模型知识截止日期之后发表的论文组成的测试集。我们评估了两项任务：通过回归预测绝对性能，以及通过成对预测比较基准测试的难度。基于前沿LLM的实验表明，精确的回归预测仍然具有挑战性，而成对预测在匹配设置下能够恢复相对性能排序。在截止日期后论文和流式评估上取得相当的表现，说明所观察到的信号并非仅仅……（原文摘要在此截断）

    arXiv:2509.20645v4 Announce Type: replace  Abstract: Modern ML research increasingly faces an evaluation bottleneck: building benchmarks and running experiments is costly, slowing iteration. We ask how much can be inferred about evaluation outcomes before experiments, given only a natural-language description of the task and setup. We call this setting anticipatory evaluation. We introduce EVALCAST, a literature-grounded resource of 2,290 description-performance pairs from 1,519 arXiv papers, including a test set of papers published after the model's knowledge cutoff. We evaluate two tasks: predicting absolute performance through regression and comparing benchmark difficulty through pairwise prediction. Experiments with frontier LLMs show that accurate regression remains challenging, while pairwise predictions can recover relative performance ordering under matched settings. Comparable performance on post-cutoff and streaming evaluations suggests that the observed signal is not solely 
    
[^294]: 使用大五人格缩放提示词实现大语言模型中的人格控制扩展

    Scaling Personality Control in LLMs with Big Five Scaler Prompts

    [https://arxiv.org/abs/2508.06149](https://arxiv.org/abs/2508.06149)

    提出基于提示词的Big5-Scaler框架，通过将大五人格数值嵌入提示词，无需额外训练即可实现对大语言模型人格特质的细粒度控制。

    

    我们提出了Big5-Scaler，这是一个基于提示词的框架，用于通过可控的大五人格特质来调节大型语言模型（LLMs）。通过将数值型特质值嵌入自然语言提示词中，我们的方法无需额外训练即可实现细粒度的人格控制。我们在特质表达、对话生成和人类特质模仿任务上对Big5-Scaler进行了评估。结果表明，该方法能够在不同模型中诱导出一致且可区分的人格特质，其性能因提示词类型和规模而异。我们的分析强调了简洁提示词和较低特质强度的有效性，为构建具有人格感知能力的对话智能体提供了一种高效的方法。

    arXiv:2508.06149v2 Announce Type: replace  Abstract: We present Big5-Scaler, a prompt-based framework for conditioning large language models (LLMs) with controllable Big Five personality traits. By embedding numeric trait values into natural language prompts, our method enables fine-grained personality control without additional training. We evaluate Big5-Scaler across trait expression, dialogue generation, and human trait imitation tasks. Results show that it induces consistent and distinguishable personality traits across models, with performance varying by prompt type and scale. Our analysis highlights the effectiveness of concise prompts and lower trait intensities, providing a efficient approach for building personality-aware dialogue agents.
    
[^295]: 面向大语言模型的多数位感知水印方法

    Majority Bit-Aware Watermarking For Large Language Models

    [https://arxiv.org/abs/2508.03829](https://arxiv.org/abs/2508.03829)

    提出多数位感知编码范式，将水印信号强度与绿色列表大小解耦，从而在不限制绿色列表大小的情况下保持强水印信号，改善了文本质量与解码准确率之间的权衡。

    

    大语言模型（LLM）的日益普及引发了人们对其在生成有害或欺骗性内容方面被滥用的担忧。为解决这一问题，已有水印方法被提出，用于在生成的文本中嵌入可识别的多比特消息以实现滥用追踪。然而，现有方法往往面临文本质量与解码准确率之间的根本性权衡。特别是，这些方法必须在编码时限制偏好词元集合（即绿色列表）的大小，以维持可供解码检测的水印信号，这不可避免地降低了生成质量。为改善这一权衡，我们提出了一种新颖的消息编码范式，称为“多数位感知编码”，它将水印信号强度从绿色列表大小中解放出来。该策略使得即使使用较大的绿色列表，也能在生成的文本中保留强水印信号。我们引入了两种实例化方法……

    arXiv:2508.03829v3 Announce Type: replace  Abstract: The growing deployment of Large Language Models (LLMs) has raised concerns about their misuse in generating harmful or deceptive content. To address this issue, watermarking methods have been proposed to embed identifiable multi-bit messages into generated text for misuse tracing. However, existing methods often suffer from a fundamental trade-off between text quality and decoding accuracy. In particular, they have to restrict the size of the preferred token set (i.e., green list) during encoding to maintain a detectable watermark signal for decoding, which inevitably degrades generation quality. To improve this trade-off, we propose a novel message encoding paradigm called \textit{majority bit-aware encoding}, which relaxes the watermark signal strength from the green list size. This strategy allows for a strong watermark signal to be preserved in generated texts even when using a large green list. We introduce two instantiations of
    
[^296]: 《无形的缰绳：为什么RLVR可能或可能无法摆脱其起源》

    The Invisible Leash: Why RLVR May or May Not Escape Its Origin

    [https://arxiv.org/abs/2507.14843](https://arxiv.org/abs/2507.14843)

    该论文提出“经验支持动力学”框架，逐题追踪RLVR对已验证解的保持、扩展与收缩，发现在给定采样预算下支持收缩超过扩展，表明RLVR可能并未真正扩展模型推理边界，而主要是放大了基座模型已知的高奖励输出。

    

    arXiv:2507.14843v5 公告类型：replace-cross。摘要：近期的研究进展突出了可验证奖励强化学习（RLVR）作为增强大语言模型能力的一种有前景的方法。然而，目前仍不清楚当前的RLVR实践究竟是真正扩展了模型的推理边界，还是主要放大了基座模型已经掌握的高奖励输出。我们引入了“经验支持动力学”这一框架，用以量化RLVR如何保持、扩展和收缩模型对已验证解决方案的访问能力。该框架被应用于多个模型、推理任务和训练检查点，通过逐个问题地匹配已验证的成功案例，揭示出总体准确率所可能掩盖的变化。在所评估的模型级基准汇总中，观察到的支持收缩超过了其扩展：在给定的采样预算下，失去已验证成功的问题数量多于获得已验证成功的问题数量。这些损失与pass@1的提升同时发生。一些RLVR模型增加了平均token级熵……

    arXiv:2507.14843v5 Announce Type: replace-cross  Abstract: Recent advances highlight Reinforcement Learning with Verifiable Rewards (RLVR) as a promising approach to enhancing LLM capabilities. However, it remains unclear whether the current practice of RLVR truly expands models' reasoning boundary or mainly amplifies high-reward outputs that base models already know. We introduce empirical support dynamics, a framework that quantifies how RLVR preserves, expands, and shrinks access to verified solutions. Applied across models, reasoning tasks, and training checkpoints, it matches verified successes question by question to reveal changes that aggregate accuracy can conceal. Across the evaluated model-level benchmark aggregates, \textit{the shrinkage of observed support outweighs its expansion}: more questions lose a verified success than gain one at the stated sampling budgets. These losses occur alongside gains in \texttt{pass@1}. Some RLVR models increase average token-level entropy 
    
[^297]: 这仅仅是幻想吗？语言模型表征反映人类对事件合理性的判断

    Is This Just Fantasy? Language Model Representations Reflect Human Judgments of Event Plausibility

    [https://arxiv.org/abs/2507.12553](https://arxiv.org/abs/2507.12553)

    本研究识别出语言模型中用于区分句子情态范畴（可能、不可能、荒谬等）的线性表征——情态差异向量，证明语言模型的情态判断能力比以往认知更可靠，且这些向量随模型能力增强以一致的顺序涌现。

    

    语言模型被用于各种多样的任务，从问答到撰写奇幻故事。为了可靠地完成这些任务，语言模型必须能够辨别句子的情态范畴（即，句子描述的是可能发生的事、不可能发生的事，还是完全荒谬的内容等）。然而，近期的研究对语言模型依据情态对句子进行分类的能力提出了质疑（Michaelov et al., 2025; Kauf et al., 2023）。在这项工作中，我们在多种语言模型中识别出了能够区分不同情态范畴的线性表征，即情态差异向量。对情态差异向量的分析表明，语言模型所具备的情态分类判断能力比之前报道的更为可靠。此外，我们发现情态差异向量随着模型能力的提升（即通过训练步数、层数和参数量的增加）以一致的顺序涌现。值得注意的是……

    arXiv:2507.12553v4 Announce Type: replace  Abstract: Language models (LMs) are used for a diverse range of tasks, from question answering to writing fantastical stories. In order to reliably accomplish these tasks, LMs must be able to discern the modal category of a sentence (i.e., whether it describes something that is possible, impossible, completely nonsensical, etc.). However, recent studies have called into question the ability of LMs to categorize sentences according to modality (Michaelov et al., 2025; Kauf et al., 2023). In this work, we identify linear representations that discriminate between modal categories within a variety of LMs, or modal difference vectors. Analysis of modal difference vectors reveals that LMs have access to more reliable modal categorization judgments than previously reported. Furthermore, we find that modal difference vectors emerge in a consistent order as models become more competent (i.e., through training steps, layers, and parameter count). Notabl
    
[^298]: π-CoT：用于多跳问答的Prolog初始化思维链提示方法

    $\pi$-CoT: Prolog-Initialized Chain-of-Thought Prompting for Multi-Hop Question-Answering

    [https://arxiv.org/abs/2506.20642](https://arxiv.org/abs/2506.20642)

    π-CoT将多跳问题转化为Prolog查询并分解为单跳子查询依次求解，再利用中间结果初始化思维链推理，从而在多跳问答任务上达到或超越标准RAG和传统CoT方法的表现。

    

    思维链（CoT）提示显著增强了大型语言模型（LLM）的问题解决能力，但在处理复杂的多跳问题时仍然存在困难，常常陷入循环推理模式或完全偏离逻辑路径。这一局限性在检索增强生成（RAG）场景中尤为突出，因为获取正确的上下文至关重要。我们提出了Prolog初始化思维链（π-CoT），这是一种新颖的提示策略，将逻辑编程的结构严谨性与语言模型的灵活性相结合。π-CoT将多跳问题重新表述为分解为单跳子查询的Prolog查询。这些子查询被依次求解，产生中间结果，我们利用这些中间结果来初始化后续的CoT推理过程。大量实验表明，π-CoT在多跳问答基准测试中达到或超越了标准RAG和上下文内CoT的性能。

    arXiv:2506.20642v3 Announce Type: replace  Abstract: Chain-of-Thought (CoT) prompting significantly enhances large language models' (LLMs) problem-solving capabilities, but still struggles with complex multi-hop questions, often falling into circular reasoning patterns or deviating from the logical path entirely. This limitation is particularly acute in retrieval-augmented generation (RAG) settings, where obtaining the right context is critical. We introduce Prolog-Initialized Chain-of-Thought ($\pi$-CoT), a novel prompting strategy that combines logic programming's structural rigor with language models' flexibility. $\pi$-CoT reformulates multi-hop questions into Prolog queries decomposed as single-hop sub-queries. These are resolved sequentially, producing intermediate artifacts, with which we initialize the subsequent CoT reasoning procedure. Extensive experiments demonstrate that $\pi$-CoT matches or outperforms standard RAG and in-context CoT on multi-hop question-answering benchm
    
[^299]: 评估与改进ChatGPT在识别定向攻击性言论和不当言论方面的表现：一项比较研究

    Assessing and Refining ChatGPT's Performance in Identifying Targeting and Inappropriate Language: A Comparative Study

    [https://arxiv.org/abs/2505.21710](https://arxiv.org/abs/2505.21710)

    研究发现ChatGPT在检测不当言论方面表现良好，且提示词优化带来的提升几乎全部来自简化标签方案并升级至GPT-4这一单次更改，但其在识别定向攻击性言论方面表现较弱、稳定性较差且误报率较高。

    

    本研究评估了ChatGPT这一先进的自然语言处理AI模型在识别网络评论中定向攻击性言论与不当言论方面的有效性。随着社交网络平台上海量用户生成内容审核挑战的日益加剧，AI在内容审核中的作用愈发重要。我们将ChatGPT的表现与众包标注和专家评估进行了比较，以评估其准确性、检测范围和一致性。研究结果表明，ChatGPT在检测不当内容方面表现良好，且迭代式提示词优化所带来的提升几乎全部来自单次更改——即简化标签方案并转向GPT-4——此后对提示结构和采样温度的进一步调整几乎没有效果。其在定向攻击性言论识别方面的表现较弱且不够稳定，与专家判断相比存在较高的误报率……

    arXiv:2505.21710v2 Announce Type: replace  Abstract: This study evaluates the effectiveness of ChatGPT, an advanced AI model for natural language processing, in identifying targeting and inappropriate language in online comments. With the increasing challenge of moderating vast volumes of user-generated content on social network sites, the role of AI in content moderation has gained prominence. We compared ChatGPT's performance against crowd-sourced annotations and expert evaluations to assess its accuracy, scope of detection, and consistency. Our findings show that ChatGPT performs well in detecting inappropriate content, and that almost the entire gain from iterative prompt refinement came from a single change --- simplifying the label scheme and moving to GPT-4 --- after which further adjustments to prompt structure and sampling temperature had little effect. Its performance on targeting language was weaker and less stable, with higher false positive rates than expert judgments befo
    
[^300]: 提示稳定性至关重要：评估与优化通用系统中自动生成的提示

    Prompt Stability Matters: Evaluating and Optimizing Auto-Generated Prompt in General-Purpose Systems

    [https://arxiv.org/abs/2505.13546](https://arxiv.org/abs/2505.13546)

    该论文提出语义稳定性这一新指标来衡量提示在重复执行时响应的一致性，并据此开发了首个利用稳定性反馈迭代优化提示质量的通用提示生成系统。

    

    自动提示生成在使通用多智能体系统能够自主执行多样化任务方面发挥着关键作用。现有方法通常基于提示的即时任务表现来评估提示，而忽略了决定其可靠性的内在品质。这种以结果为中心的视角不仅限制了可解释性，而且未能考虑到大型语言模型（LLM）固有的随机性。在这项工作中，我们强调提示稳定性——即模型在重复执行中响应的一致性——作为构建鲁棒且有效的提示生成系统的关键因素。为了量化这一概念，我们提出语义稳定性作为评估提示响应一致性的标准。基于所提出的指标，我们开发了首个稳定性感知的通用提示生成系统，该系统利用稳定性反馈来迭代提升提示质量和系统性能。

    arXiv:2505.13546v2 Announce Type: replace-cross  Abstract: Automatic prompt generation plays a crucial role in enabling general-purpose multi-agent systems to perform diverse tasks autonomously. Existing methods typically evaluate prompts based on their immediate task performance, overlooking the intrinsic qualities that determine their reliability. This outcome-centric view not only limits interpretability but also fails to account for the inherent stochasticity of large language models (LLMs). In this work, we bring attention to prompt stability-the consistency of model responses across repeated executions-as a key factor for building robust and effective prompt generation systems. To quantify this, we propose semantic stability as a criterion for assessing the response consistency of prompts. Based on the proposed metric, we developed the first stability-aware general-purpose prompt generation system that leverages stability feedback to iteratively enhance both prompt quality and sy
    
[^301]: 超越规模：梯度如何塑造大语言模型中的剪枝决策

    Beyond Size: How Gradients Shape Pruning Decisions in Large Language Models

    [https://arxiv.org/abs/2311.04902](https://arxiv.org/abs/2311.04902)

    提出GBLM-Pruner，一种利用泰勒展开一阶项、基于少量校准样本归一化梯度的免训练剪枝方法，在大语言模型剪枝中显著优于SparseGPT和Wanda等现有方法。

    

    arXiv:2311.04902v3 公告类型：替换 摘要：拥有数十亿参数的大型语言模型（LLMs）是网络剪枝的主要目标，剪枝旨在在不损害性能的前提下移除部分模型权重。先前的方法如幅度剪枝、SparseGPT和Wanda要么仅关注权重，要么将权重与激活值相结合以实现稀疏性。然而，这些方法忽略了从预训练LLMs中获取的蕴含丰富信息的梯度。在本文中，我们提出了一种新颖的以稀疏性为中心的预训练LLMs剪枝方法，称为基于梯度的语言模型剪枝器（GBLM-Pruner）。GBLM-Pruner利用泰勒展开的一阶项，以无需训练的方式运行，通过利用少量校准样本的适当归一化梯度来确定剪枝度量，并在多个基准测试中大幅超越SparseGPT和Wanda等竞争方法。有趣的是，通过引入梯度，我们方法的非结构化剪枝倾向于重新……

    arXiv:2311.04902v3 Announce Type: replace  Abstract: Large Language Models (LLMs) with billions of parameters are prime targets for network pruning, which removes some model weights without hurting performance. Prior approaches such as magnitude pruning, SparseGPT, and Wanda either concentrated solely on weights or integrated weights with activations for sparsity. However, they overlooked the informative gradients derived from pretrained LLMs. In this paper, we present a novel sparsity-centric pruning method for pretrained LLMs, termed Gradient-based Language Model Pruner (GBLM-Pruner). GBLM-Pruner leverages the first-order term of the Taylor expansion, operating in a training-free manner by harnessing properly normalized gradients from a few calibration samples to determine the pruning metric, and substantially outperforms competitive counterparts like SparseGPT and Wanda in multiple benchmarks. Intriguingly, by incorporating gradients, unstructured pruning with our method tends to re
    

