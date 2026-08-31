# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Formal Limitation on Learning Human Language From Textual Corpora](https://arxiv.org/abs/2608.28560) | 该论文从信息论角度证明，话语形式对意义存在固有的不可约减的不确定性，因此任何文本表示（包括大语言模型的隐藏状态）在从话语恢复说话者意图意义上的概率都存在无法超越的上界。 |
| [^2] | [When Robots Mishear Us: Mapping the Safety Risks of Voice-Controlled Embodied AI](https://arxiv.org/abs/2608.28518) | 该论文首次系统研究了语音识别错误对具身AI安全性的影响，通过将模拟的ASR错误与安全基准相结合，揭示了语音识别错误可导致有害指令被具身AI模型接受并执行这一新型安全风险。 |
| [^3] | [Phoneme- and Word-Level Metrics Using Self-Supervised Speech Representations for Forced Alignment Evaluation](https://arxiv.org/abs/2608.28508) | 本文提出两个基于自监督语音表示的无参考强制对齐评估指标PCMI和WACS，无需人工标注时间戳即可在大规模多语言场景（85种语言）下有效评估对齐质量。 |
| [^4] | [Ladders in Chaos: When, How, (and Perhaps Why) Does Test-Time Scaling Improve LLM Machine Translation](https://arxiv.org/abs/2608.28496) | 本研究系统对比了序列式与并行式测试时扩展在机器翻译中的表现，发现序列式采样具有更高的性能上限，能显著提升翻译流畅度和自然度，但在大推理预算下可能损害翻译准确性，并部分解释了其作用机制。 |
| [^5] | [NL2AGBench: Benchmarking LLM Auto-Formalization for AlphaGeometry](https://arxiv.org/abs/2608.28481) | 该论文提出了NL2AGBench基准测试，用于评估大语言模型将英文几何问题自动形式化为AlphaGeometry兼容形式化表示的能力，从而解决神经符号几何系统中手动转换输入的可用性瓶颈。 |
| [^6] | [Blind Men and the Elephant: Probing the Epistemic Myopia of LLMs under Long-Tail Divergent Knowledge](https://arxiv.org/abs/2608.28478) | 提出ElephantBench闭卷知识探针基准，揭示大语言模型对长尾分歧事实存在“认知短视”——即使最强模型也只能在52.4%的问题上同时召回两个版本的答案，且扩大模型规模和增强推理能力均无法消除这种不完整性。 |
| [^7] | [ContextPilot: Teaching Agents for Proactive Context Management via Fine-grained RL](https://arxiv.org/abs/2608.28476) | ContextPilot通过扩展上下文管理工具集并引入细粒度的强化学习信用分配机制，教会智能体在长时程任务中主动、高效地管理持续增长的工作上下文。 |
| [^8] | [Stranger, Fan, or Peer? A Systematic Study on the Role of Interlocutor in Persona-Based Dialogue Generation](https://arxiv.org/abs/2608.28467) | 该论文首次将人设对话系统中对话双方“个人简介可见性”在训练、推理、评估三个阶段进行独立分解研究，发现训练阶段的简介可见性比推理阶段更能决定模型是真正通过对话表达人设特征，还是简单复制简介文本。 |
| [^9] | [Acquire, Repair, Preserve: A Diagnosis-Guided Post-Training Recipe for Small-Model Dialogue Game Agents](https://arxiv.org/abs/2608.28458) | 该论文提出一种诊断引导的三步后训练方案（获取、修复、保持），使2B小型模型在对话游戏挑战中的clemscore从10.67大幅提升至38.92，同时保持了一般能力不退化。 |
| [^10] | [Sliding-window beats linear attention](https://arxiv.org/abs/2608.28444) | 本研究表明，带sink的滑动窗口注意力（SWA）在多项下游任务和长上下文推理任务上的表现与经过后训练的线性注意力模型相当甚至更优，说明这个更简单的基线方法被严重低估了。 |
| [^11] | [Fidelity Is Not Enough: Dispatch-Level Instrumentation for Agentic Datasheet Extraction](https://arxiv.org/abs/2608.28439) | 仅靠保真度指标会漏掉智能体文档提取中的静默失败（如模型不开数据手册就编造答案），本文通过记录每次工具调用的调度级日志，构建失败归因分类器与静默失败检测器，能在不误报正常提取的情况下发现全部植入故障。 |
| [^12] | [Are These Modules Worth Their Cost? A Paradigm-Level Accuracy-Cost Analysis of In-context Learning Text-to-SQL](https://arxiv.org/abs/2608.28432) | 该论文在统一受控环境下首次对ICL Text-to-SQL流水线中五个模块的17种范式级配置进行了精度-成本边际贡献分析，发现执行反馈精炼是唯一在所有骨干模型上收益均普遍成立的范式。 |
| [^13] | [A Unified Framework to Elicit Structured Feedback for Interpretable Multi-Trait Essay Scoring](https://arxiv.org/abs/2608.28407) | 提出统一自回归框架HiFTS，先基于评分标准生成分层思维链反馈再预测多维度与整体作文分数，并通过GRPO强化学习优化反馈质量与评分一致性，实现可解释的多维度自动作文评分。 |
| [^14] | [CultureConverse: A Multilingual Multi-turn Simulation Harness for Culturally Grounded Assistance in East and Southeast Asia](https://arxiv.org/abs/2608.28405) | 该论文提出CultureConverse，一个覆盖东亚与东南亚10个地区、58个子群体身份和7个领域的多语言多轮文化情境化助手对话模拟与评测框架，并构建了包含14,610个基准评测回合和274,295个oracle引导对话的数据集，弥补了传统单选题式文化评测无法反映多轮实际辅助场景的不足。 |
| [^15] | [BEACON: Behavior-Anchored Cross-Source Knowledge Graph Construction for Cyber Threat Intelligence](https://arxiv.org/abs/2608.28394) | 该论文提出BEACON，创新性地以映射到MITRE ATT&CK的攻击行为作为锚点，将不同来源报告中的实体和失陷指标统一到同一规范空间，实现跨源网络威胁情报知识图谱的自动构建。 |
| [^16] | [CamoDocs: A Poisoning Attack Against Retrieval-Augmented Language Models Using Camouflaged Documents](https://arxiv.org/abs/2608.28389) | CamoDocs通过将对抗性文档伪装在良性内容中并利用分散token技术，实现了无需直接包含查询即可绕过多种防御的高效RAG投毒攻击，在多个模型和基准上保持高攻击成功率。 |
| [^17] | [Semantic Head Specialization Guides Hybrid ViT Attention for Multimodal LLMs](https://arxiv.org/abs/2608.28383) | 本研究发现了ViT注意力头会分化为物体与背景专门化角色的“语义头专门化”（SHS）现象，并提出SHS-Index加以量化，据此设计了以6.5倍更少计算量在22个图像视频任务上媲美全注意力的混合注意力架构Ariadne Attention。 |
| [^18] | [When Linguistic and Internal Confidence Diverge in Large Language Models](https://arxiv.org/abs/2608.28382) | 该研究通过跨8个分类任务、2个生成任务和30个模型的大规模实验，揭示了大语言模型口头表达的语言置信度与其内部置信度经常不一致，且指令微调模型置信度更高但校准更差、态度提示会夸大置信度而无法提升准确性。 |
| [^19] | [PersonaForge: Realistic Multi-Turn User Simulation for Agentic Systems](https://arxiv.org/abs/2608.28378) | PersonaForge是一个通过四维人格空间、SOUL驱动行为控制和反向深度构建来合成逼真多轮用户-智能体交互的用户模拟框架，并据此构建了6.3K条训练数据和138任务的人工标注基准PersonaForge-Bench，弥合了真实多轮用户交互与智能体系统训练评估之间的巨大差距。 |
| [^20] | [BanglaMed-QA: A Question Answering System for Healthcare Support in Bangla](https://arxiv.org/abs/2608.28329) | 本文提出了BanglaMed-QA——首个专为孟加拉语医疗领域设计的问答系统，通过构建包含506种疾病、4,493个问答对的结构化知识库，结合SVM问题分类、领域专用词典与同义词集以及多种相似度度量与投票机制，为低资源语言的医疗健康信息支持提供了有效解决方案。 |
| [^21] | [Layered LLM Defenses as an Ensemble: Access Tiers, Inference Cost, and the Measured Failure Correlation Between Defense Layers](https://arxiv.org/abs/2608.28327) | 本文提出对抗者访问层级模型（AATM）和推理成本分类方法，首次实测了大语言模型多层防御之间的失败相关性，证明防御堆栈只有在各层在不同输入上失败时才能真正产生叠加防护效果。 |
| [^22] | [AIM: Anchor Identity Features, Then Match for Multimodal Large Language Model Unlearning](https://arxiv.org/abs/2608.28312) | 提出AIM两阶段遗忘方法，利用身份类与感知类问题在隐藏状态中组织方式的差异，在无法访问保留图像的情况下实现多模态大语言模型的身份信息遗忘，同时保留其视觉感知能力。 |
| [^23] | [VISTA: Verifier-Informed Student-to-Teacher Adaptation for On-Policy Self-Distillation](https://arxiv.org/abs/2608.28306) | 提出VISTA方法，在保留标准在策略自蒸馏学生更新的同时，利用结果验证的rollout使特权教师向学生分布自适应，解决了教师分布与学生有效推理不匹配时单向监督误导学生的问题。 |
| [^24] | [A Probabilistic Interpretation of KV Cache Eviction](https://arxiv.org/abs/2608.28293) | 该论文首次从概率推理的角度对KV缓存驱逐问题进行形式化，证明其计算上的困难性，并将其归结为可通过采样近似的期望估计问题，从而为设计驱逐策略以及解码时校正被驱逐条目提供了理论框架。 |
| [^25] | [Embedding Models for Stance-Aware Argument Retrieval](https://arxiv.org/abs/2608.28283) | 本研究探索了稠密嵌入模型在立场感知论点检索中的应用，发现现有模型偏向主题相关性而忽视立场信息，而通过对比训练纠偏又会导致模型过度关注极性关键词的过度纠正问题。 |
| [^26] | [Synth-JDoc: Synthesizing a Japanese Document Image Dataset for OCR with Diverse Layouts and Embedded Images](https://arxiv.org/abs/2608.28248) | 该论文提出Synth-JDoc，一个通过合成方式构建、具有多样化版式和嵌入图像的日语文档图像OCR数据集，旨在提升大型视觉语言模型对竖排日语文本的阅读能力。 |
| [^27] | [Stay Within Your Bounds: Distance-Guided Decoding for Guaranteed Context-Free Grammar Compliance](https://arxiv.org/abs/2608.28229) | 提出一种基于下推自动机的距离引导解码框架，通过离线计算可达性标签与到接受状态的距离上界、在线进行视野感知剪枝与束搜索，保证大模型生成结果百分之百符合目标上下文无关文法，同时提升补全质量。 |
| [^28] | [Benchmarking large language model agent societies against human behavioural distributions](https://arxiv.org/abs/2608.28182) | 提出开放基准SILICA，通过五个带人类行为锚点的环境及其扰动和反记忆变体检验大语言模型智能体社会，发现模型仅在首轮公共品贡献等起点上与人类行为一致，而无法复现最终状态等结果。 |
| [^29] | [Text Restoration of Ancient Documents with Language Models](https://arxiv.org/abs/2608.28170) | 本研究首次系统探索利用不同架构的语言模型修复受损古代手稿中的文本缺失，提出多种解码策略以解决缺损边界与分词方案不一致的问题，发现该技术虽无法完全自动化，但可作为古文书学家的有效辅助工具。 |
| [^30] | [FinExam-10K: When Retrieval Helps Financial Reasoning?](https://arxiv.org/abs/2608.28155) | 该论文提出了迄今最大的覆盖CFA与FRM完整考试体系的英文金融考试基准FinExam-10K（含10,198道专家标注题目及双赛道评估设计），并揭示检索增强方法（Function-RAG和FunctionGraph-RAG）能挽救模型在困难金融推理题目上的失败。 |
| [^31] | [Nested Byte-Level Vocabularies Are Cheap to Deploy and Expensive to Share: A Pre-Registered Negative Result](https://arxiv.org/abs/2608.28151) | 本文通过预注册实验证明，嵌套字节级词表虽然能通过精确切片实现廉价部署（移除66%部署权重且数值完全一致），但跨规模共享词表会使模型性能超出预设余量地落后于固定词表的专用模型。 |
| [^32] | [H-Scale: Hessian-Guided Scale Refinement for NVFP4 Sub-Byte LLM Inference](https://arxiv.org/abs/2608.28113) | 提出H-Scale，一种基于Hessian二阶信息的轻量级后处理方法，通过精修NVFP4量化中的逐组缩放因子来更直接地减少层输出扰动，从而提升大语言模型亚字节推理的精度。 |
| [^33] | [Speculative Probing: LLM Monitoring at Speculative-Decoding Cost](https://arxiv.org/abs/2608.28099) | 该论文提出通过在目标序列末尾附加训练好的软提示，将LLM中的投机解码模块重新用作高效且高质量的序列分类器，以投机解码的成本实现实时模型监控。 |
| [^34] | [CNeo-Bench: Diagnosing Large Language Models on Chinese Neologisms](https://arxiv.org/abs/2608.28053) | 该论文提出了CNeo-Bench——一个包含4,759个汉语新词的基准及双层评估框架，用于区分大语言模型“描述新词”与“操作其底层语言机制”的能力，发现多数模型释义生成准确率低于40%，且普遍存在能描述却不能还原源形式的“识别-操作”差距。 |
| [^35] | [SimpCue: Cue-Based Prompting for Multilingual Text Simplification](https://arxiv.org/abs/2608.28042) | 该论文提出在多语言“易读”文本简化中，将自动预测的句子复杂度语言学线索加入提示，能在所有评估指标上带来小幅但一致的提升，而人工黄金线索的效果反而不稳定。 |
| [^36] | [A Shaky Voice Is Not Always a Dodge: Benchmarking Textual and Vocal Evasion Detection in Earnings Calls](https://arxiv.org/abs/2608.28040) | 该论文提出 DualEvasion 基准，首次将财报电话会议中的回避检测从单一文本维度扩展为文本回避与语音自信度的双维度联合分析，并发现现有最先进多模态模型难以检测语音自信度，原因是它们孤立解读声学线索而非结合说话者个人基线。 |
| [^37] | [Twin Worlds: Equivariance-Based Abstention for Evidence-Grounded Reasoning](https://arxiv.org/abs/2608.28018) | 该论文提出“双生世界”（TW）框架，通过等变性检验模型的推理是否真正以证据为依据，使模型在证据不足时能够弃答，从而避免生成看似合理却缺乏证据支撑的答案。 |
| [^38] | [Beyond Global Scalars: Synergizing Token-Level Statistics and Deep Semantics for Adversarial AIGC Text Detection](https://arxiv.org/abs/2608.28009) | 提出NeuroStat端到端框架，通过协同词元级概率统计信息与深层语义隐藏状态来弥合统计与语义之间的鸿沟，同时构建了包含16000个样本的MOSAIC对抗性基准，显著提升了机器生成文本检测在对抗场景下的鲁棒性。 |
| [^39] | [Predicting Turn-Taking Outcomes in Multi-Party Conversation: Interpretable Modelling of Speech and Gaze Dynamics with Interpersonal Closeness](https://arxiv.org/abs/2608.27988) | 本研究基于GaMMA四人自由对话语料库，构建了融合注视动态、言语特征与人际亲密度的可解释逻辑回归模型，用于预测话轮转换结果是间隙还是重叠。 |
| [^40] | [QUORUM: QUality-Optimized Routing Using Multiple annotators](https://arxiv.org/abs/2608.27974) | QUORUM是一个预算感知的标注路由框架，它利用基于特征的难度信号在固定预算下动态地将数据实例分配给人类或大语言模型标注者，并通过多标注一致性奖励机制提升标注的可靠性。 |
| [^41] | [DisCTI: Who Needs to Know Timely? Automated Sector-Aware Cyber Threat Intelligence Dissemination](https://arxiv.org/abs/2608.27967) | 该论文提出DisCTI，将面向行业的网络威胁情报分发建模为多标签分类问题，实现自动化的行业感知CTI及时分发，解决了现有平台（如MISP）中98%事件缺乏行业标注、情报运营价值受限的问题。 |
| [^42] | [Lexically conditioned realization ambiguity in Korean predicate morphology](https://arxiv.org/abs/2608.27966) | 韩语谓词的表层形式并非由规范词素序列唯一决定：同形同音的谓词会因词汇意义与论元结构不同而分属不同的屈折实现类别，形成“同音异折”现象。 |
| [^43] | [Entity-Memory Graph Retrieval Improves Evidence Coverage in Long-Conversation Question Answering](https://arxiv.org/abs/2608.27925) | 提出实体-记忆图检索方法，通过共享实体和有向时间边连接对话轮次，将长对话问答的证据召回率从79.75%提升至84.48%，显著优于匹配的稠密检索基线。 |
| [^44] | [What Makes Agent Memory Useful for Reliable Unanswerable Question Handling?](https://arxiv.org/abs/2608.27924) | 该论文在统一的智能体RAG框架下系统研究了记忆在处理不可回答问题中的作用，发现记忆带来的提升是选择性且脆弱的，跨模型记忆复用比跨数据集迁移更可行，且决策引导比轨迹塑造更能保留记忆收益。 |
| [^45] | [AI Alignment through a Game-theoretic Lens: A Survey](https://arxiv.org/abs/2608.27910) | 本综述以博弈论视角系统梳理AI对齐研究，围绕偏好多样性、对齐优先级和时间动态三大挑战组织文献，阐明了博弈论分析真正发挥作用之处以及构建鲁棒、自适应、可验证AI系统仍待解决的难题。 |
| [^46] | [LandingAgent: A Reference-Annotated Dataset and Agentic Generation Framework for Landing Pages](https://arxiv.org/abs/2608.27902) | 该论文提出了参考画像数据集LandingBench和三阶段智能体生成框架LandingAgent，通过从真实落地页中抽象提取可复用的设计模式来引导生成过程，解决了大语言模型直接生成落地页时产生的模板化和内容缺乏依据的问题。 |
| [^47] | [OpenStamp: A Watermark for Open-Source Language Models](https://arxiv.org/abs/2608.27899) | OpenStamp通过仅修改开源语言模型的反嵌入层，将水印逻辑直接编码进模型权重，解决了传统采样概率水印在白盒场景下可被用户禁用的问题，在几乎不损失模型能力的前提下实现了更优的检测性能和更强的鲁棒性。 |
| [^48] | [AI Writers Have a Consistent Stylometric Footprint, but AI Editors Do Not](https://arxiv.org/abs/2608.27855) | 本文发现AI生成的文本具有跨8个模型和5个领域保持一致的风格计量学“足迹”（主要由熵和词汇多样性等特征构成），可用于检测AI生成文本，但AI编辑过的人类文本并不会留下同样的足迹。 |
| [^49] | [Is Prosody Lost in Translation? Fine-Grained Cross-Lingual Prosody Similarity Across Languages](https://arxiv.org/abs/2608.27848) | 该研究首次利用多语言配音数据对英德、英西、英法语言对进行了细粒度的跨语言韵律分析，揭示了某些语言间韵律结构存在固有的跨语言相关性，为在语音到语音翻译系统中有效融入韵律提供了重要依据。 |
| [^50] | [EvoHarmBench: Breaking Content Moderation with Iterative Human-Like Evasion](https://arxiv.org/abs/2608.27844) | 本文提出了首个面向内容审核系统的动态对抗评估框架 EvoHarmBench，通过在语义簇层面迭代演化类人规避策略，揭示了静态基准分数与真实部署审核效果之间的显著性能差距。 |
| [^51] | [Synthetic Linguistic Agency: How an Embodied Mortal Agent Learns Linguistic Affordances through Consequential Social Experience](https://arxiv.org/abs/2608.27843) | 该论文提出了“合成语言能动性”的可检验标准，并基于稳态调节强化学习构建了一个以死亡为基础的具身有死智能体（EMA），使其通过后果性社会经验习得语言可供性。 |
| [^52] | [Auditing Generative Audio Calls for Known-Task Audio-LLM Evaluation](https://arxiv.org/abs/2608.27817) | 该论文将音频大语言模型的评估建模为受控的调用决策问题，发现在已知封闭集任务上，有监督编码器（如CLAP和WavLM）无需调用生成式音频模型即可取得接近最优的准确率，从而揭示了传统“波形提示对比ASR转录”的评估方式混淆了声学证据获取与生成模型调用这两个因素。 |
| [^53] | [PersonaEdit: Representative Sample Selection for Personalized Model Editing](https://arxiv.org/abs/2608.27816) | 提出PersonaEdit，一种基于隐表示聚类与比例分层采样的代表性编辑样本选择策略，使模型编辑能够高效、低成本地实现LLM个性化。 |
| [^54] | [Representation of syntax in LLMs through the lens of linear distance and similarity-aware entropy](https://arxiv.org/abs/2608.27813) | 本文将结构探针的评估指标按句法关系分解为UASL，发现相关词间线性距离的统计特征以及句法关系中心词的相似性感知熵这两个因素能够预测各句法关系重建准确率的大部分变异，且该结论在不同规模和架构的模型上均成立。 |
| [^55] | [CEDAR: Automata as Verifiable Interfaces for Language-Guided Embodied Action](https://arxiv.org/abs/2608.27797) | CEDAR框架将自然语言指令中的技能与约束统一表示为环境事件轨迹上的确定性有限自动机，通过自动机交集运算使具身智能体的行为在构造上即可强制执行约束，从而实现可验证、可组合、可修复的语言引导具身行动。 |
| [^56] | [Compositional Failure in Audio-Visual LLMs: Late-Layer Prior Dominance Under Cross-modal Conflict](https://arxiv.org/abs/2608.27785) | 本研究揭示了音频-视觉大语言模型在跨模态冲突下存在“先验主导”失败模式——模型后期层（集中于约25.5层）固守内部偏好的答案模式而忽视冲突输入，导致准确率大幅下降，且增强时序对齐仅能改变答案偏差而无法提升组合泛化能力。 |
| [^57] | [SURE-Challenge: Evaluating Speech Evidence Before Speech-LLM Generation](https://arxiv.org/abs/2608.27783) | 该论文提出 SURE-Challenge 基准，用于评估语音大模型在生成回答之前对不支持输入（静音、噪声、合成音调、嘈杂语音）的拒绝能力，并证明一个简单的“能量加 Whisper 分数”规则可将不支持输入的拒绝数从 15/204 提升至 196/204，同时不损失有效输入的准确率。 |
| [^58] | [Memorization Is Not Extraction: Tight Differential-Privacy Bounds and Audit Blind Spots](https://arxiv.org/abs/2608.27782) | 该论文精确刻画了差分隐私对反事实记忆化与自适应提取这两个度量的紧致控制界，证明二者互不控制，从而揭示了差分隐私作为统一防护代理时存在的审计盲区。 |
| [^59] | [Why Didn't It Check? Unsupported Final Claims and Their Repair in Two Tool-Equipped Language Models](https://arxiv.org/abs/2608.27768) | 该论文将工具增强语言模型做出无证据支持的最终断言这一失败现象分解为“发生率”和“条件修复率”两个可精确测量的指标，并通过在断言发生状态的精确副本上进行重放、仅改变工具响应中一个字符的对照实验，在 Qwen3-32B 上量化研究了这一问题。 |
| [^60] | [Fast Weight Attention for Continual Learning](https://arxiv.org/abs/2608.27763) | 该论文在“写后读”自回归语义下将快速权重记忆与状态空间模型的状态转移统一视为在线学习规则，并推导出面向持续学习前缀预测的归一化一阶更新家族（Falcon 系列回归与内积变体）。 |
| [^61] | [Informational Antilocality and the Locality Bias in LLMs](https://arxiv.org/abs/2608.27760) | LLM最终能够学会反局部语言并达到相当的损失水平，但在反局部性更强的语言上收敛更慢，表明局部性偏差体现在学习速度上而非学习能力上。 |
| [^62] | [Load-Bearing Context: The Question Damage Score for Evaluating Context Reliance in Linguistic Reasoning](https://arxiv.org/abs/2608.27756) | 该论文提出“问题损伤分数”诊断框架，通过从语言学奥赛谜题中随机或靶向删除单个上下文示例，量化大语言模型答题对上下文的依赖程度，以区分模型是依赖上下文还是先验知识。 |
| [^63] | [The Calls are Coming from Inside the Model: Investigating Probe-based Detection of Tool-Calling Errors in LLMs](https://arxiv.org/abs/2608.27750) | 本研究提出利用线性探针读取大语言模型隐藏状态来检测工具调用错误，在18个模型上验证了该方法能有效捕获包括参数值错误在内的各类调用错误，且检测效果受模型大小、探针层级和后训练类型的影响。 |
| [^64] | [Below the Noise Floor: Bimodal Seed Collapse and Distinct Failure Modes in Small-Model Knowledge Distillation](https://arxiv.org/abs/2608.27729) | 该研究通过多种子实验发现，小模型知识蒸馏的单种子报告会掩盖高达48.7个百分点的种子方差，部分KD变体还会出现双峰坍塌且失败模式各不相同，因此任何低于五个百分点的蒸馏收益声明都不可信。 |
| [^65] | [First Make It Playable, Then Make It Good: Staged Interaction Learning for Small Dialogue-Game Agents](https://arxiv.org/abs/2608.27672) | 提出20亿参数模型Qwen-GuidePlay-2B，采用“先模仿完整成功轨迹保证可玩性、再通过加权轮次级和教师引导SFT提升决策能力”的分阶段训练策略，在Playpen对话游戏挑战中取得第二高的clemscore增量（较基座模型提升约36分）。 |
| [^66] | [Semantic Watermarking with Order-Robust Detection over Sub-sentence Units](https://arxiv.org/abs/2608.27666) | 该论文提出了自适应嵌入位移攻击（EDA），在黑盒设置下通过改写、重排和重新分段以单一目标最大化嵌入位移，无需访问生成器或密钥即可在四种语义水印方案上成功移除32.6%至47.9%文档的水印，是测试攻击中最有效的。 |
| [^67] | [Knowing Before Answering: Decoding Language Models for Reliable RAG](https://arxiv.org/abs/2608.27661) | 该论文提出利用语言模型的内部信号（隐藏层激活与注意力特征）训练轻量级线性分类器，在作答前判断RAG检索到的信息是充分、不充分还是冲突，从而提升生成系统的可靠性。 |
| [^68] | [When Tokenizers Fail: Byte-Level Chunking for Zero-Shot Transfer to Low-Resource Languages](https://arxiv.org/abs/2608.27658) | 本文提出一种无需大量训练的分层字节级网络框架，通过从冻结基础模型的子词表示初始化字节嵌入并使用块对齐损失，实现了向低资源语言的零样本迁移。 |
| [^69] | [Trajectory-Level Speculative Decoding for Diffusion Language Models](https://arxiv.org/abs/2608.27514) | 提出了一种针对扩散语言模型的轨迹级推测解码框架，通过置信度分层树探索构建草稿去噪轨迹、利用双向注意力掩码进行分块并行验证，并引入跨块前瞻的块间推测机制以突破单token生成的吞吐量瓶颈。 |
| [^70] | [Quantization-Triggered Backdoors in Language Models: Cross-Quantizer Transferability and the Validation--Deployment Gap](https://arxiv.org/abs/2608.27512) | 该论文提出量化行为等价类（QBEC）理论，证明源精度下的模型验证无法保证量化部署后的行为等价，并构建三阶段对抗微调框架，使后门仅在模型被INT8或4比特量化部署时才被触发激活，揭示了量化流程中的安全隐患。 |
| [^71] | [How Do Linear Probes Emerge? A Circuit-Tracing Framework with Concept-Targeted Attribution](https://arxiv.org/abs/2608.27510) | 该论文提出概念定向归因（CTA）框架，通过针对线性探针方向训练归因图，首次将线性探针的性能与模型内部可解释的电路结构联系起来，不仅能判断探针是否有效，还能揭示是哪些内部计算使探针起作用。 |
| [^72] | [A Survey on Rubric-Guided Reinforcement Learning for Language Models](https://arxiv.org/abs/2608.27505) | 本综述提出一个贝叶斯统一框架——将宪法视为评价准则的先验分布、量规视为其条件化实例，并沿先验—后验轴系统梳理了量规引导强化学习的分类体系，用结构化、可解释的评价准则取代标量奖励，从而改进大语言模型的对齐效果。 |
| [^73] | [INSPIRE: An Internalize-Then-Improve Approach for Example-Driven Mathematical Reasoning](https://arxiv.org/abs/2608.27501) | 提出INSPIRE方法，采用先内化参考示例、再逐步改进的策略，增强大语言模型基于示例的数学推理能力（如构造反例检验定理边界），超越仅优化最终答案正确性的传统方法。 |
| [^74] | [XHotpotQA: A Benchmark for Cross-Lingual Knowledge Composition in Multi-Hop Question Answering](https://arxiv.org/abs/2608.27481) | 该论文提出XHotpotQA基准，通过将多跳问答实例建模为带显式语言标注的证据依赖图，系统评测了模型在混合语言证据下跨语言组合知识的能力，并发现完全的语言不匹配会导致模型性能显著下降。 |
| [^75] | [Retrieving Relations, Detecting Fallacies: A RAG Approach to Political Debate Analysis](https://arxiv.org/abs/2608.27471) | 该论文提出一种引导式检索增强生成方法，利用论证间的支持与攻击关系动态引导检索过程，从而突破静态特征的局限，提升政治辩论中谬误检测与分类的性能。 |
| [^76] | [Select, Don't Train: The Benefits of Modular Entity Disambiguation with LLM-Based Selection](https://arxiv.org/abs/2608.27470) | 本文系统比较了在共享LLM选择阶段下不同检索策略的实体消歧效果，提出“选择而非训练”的模块化范式，避免了训练专用检索器的高昂成本与维护负担。 |
| [^77] | [UIC-AIHealth4All at ArchEHR-QA 2026: Answer-First Evidence Grounding for Clinical Question Answering](https://arxiv.org/abs/2608.27467) | 该研究提出一种答案优先的流水线，先让模型生成引用具体病历句子的候选答案再分类完整证据集，并结合自洽性投票进行答案-证据对齐，在ArchEHR-QA 2026临床问答任务中分别取得证据识别第三名、答案生成第九名和答案-证据对齐第五名的成绩。 |
| [^78] | [PACE: Publisher-Adaptive Content Extraction via Agentic Automation](https://arxiv.org/abs/2608.27466) | PACE是一个智能体自动化框架，训练时利用LLM从代表性页面中分析结构并聚合可复用的抽取模式，推理时将学习到的配置实例化为固定的确定性抽取模板，从而同时实现准确、低成本且可扩展的出版商自适应网页内容抽取。 |
| [^79] | [The Effect of Emotional Context on Large Language Models' Endorsement of Premature Decisions: Comparing Emotional Vulnerability Across Six Commercial Models](https://arxiv.org/abs/2608.27465) | 本研究通过对六种商业大语言模型在324段对话中的实验发现，用户表达痛苦等负面情绪会显著增加模型对过早决策（如基于薄弱证据辞职）的支持与鼓励，且该效应独立于对话长度，揭示了LLM在情感语境下普遍存在的安全脆弱性。 |
| [^80] | [Sledgehammer or Scalpel? A Fine-grained Adaptive Framework for Implicit Hate Speech](https://arxiv.org/abs/2608.27462) | 提出了细粒度自适应框架FAID，将隐性仇恨言论划分为浅层、针对性和上下文依赖三类，并针对不同类别采用不同复杂度的检测策略，在提升检测精度的同时降低不必要的计算开销。 |
| [^81] | [SciReC: Diagnostic Evaluation of Multimodal, Multi-Turn Relational Reasoning with Adaptive Interaction](https://arxiv.org/abs/2608.27461) | 该论文提出了SciReC——一个模型自适应的多模态学术对话基准，以及DMRA缺陷诊断框架，用于系统评估多模态大语言模型在多轮关系推理中的表现，并量化视觉理解、知识展示和记忆回忆等因素对失败案例的贡献。 |
| [^82] | [Accelerating LLM Inference via Vector Index Based Output Embeddings](https://arxiv.org/abs/2608.27460) | 本文将大语言模型的输出投影重新表述为基于HNSW向量索引的最大内积搜索，仅检索高分候选词元以替代稠密词表投影，在CPU推理中最高可将解码吞吐量提升82%且不损失生成质量。 |
| [^83] | [Representing and Parsing Korean Constituency Structure at Different Levels of Granularity](https://arxiv.org/abs/2608.27035) | 本文通过比较三种基于Penn韩语树库的不同粒度表示，系统评估了韩语成分解析在形态复杂eojeol单元下的表示策略与解析性能。 |
| [^84] | [Surgical Alignment in Knowledge Graph Training for Clinical Diagnosis with Large Language Models](https://arxiv.org/abs/2608.26587) | 本文提出“外科式对齐”概念，通过梯度干预密度和梯度扭曲指标，发现KL正则化下的知识图谱判断训练能产生稀疏局部更新，优于任务特定SFT的密集更新，从而更有效地将KG知识整合到LLM中用于临床诊断。 |
| [^85] | [Comparing Chunking and Embedding Strategies for Turkish RAG Systems](https://arxiv.org/abs/2608.26192) | 本文系统比较了土耳其语RAG系统中分块策略与嵌入模型的影响，发现布局感知分块能缩小嵌入模型差异，且领先嵌入模型间无显著统计差异。 |
| [^86] | [Self-Generated Text Recognition: Quality Heuristics, Cross-Task Transfer, and Downstream Bias in LLM Evaluation](https://arxiv.org/abs/2608.26159) | 本研究通过系统分析实验设计选择（操作化）对自生成文本识别准确率的影响，调和了先前矛盾结论，并证实了质量启发式在LLM评估中的关键作用。 |
| [^87] | [Artificial Intelligence Models Can Predict and Collaboratively Modulate Human Memory Search](https://arxiv.org/abs/2608.26152) | 本研究首次证明大型语言模型能够预测并协同调节人类在语义记忆搜索中的心理轨迹，从而作为认知工具增强而非取代人类生成性思维。 |
| [^88] | [ElementCheck: Complexity-Aware Long-Form Text Factuality Evaluation via Sentence Elements](https://arxiv.org/abs/2608.26118) | ElementCheck提出了一种基于句子元素图的复杂度感知验证框架，通过图拓扑结构估计句子复杂度并自适应调整验证粒度，解决了长文本事实性评估中固定分解和验证粒度导致的可靠性问题。 |
| [^89] | [TreeGraft: Adaptive Multi-Drafter Grafting for Tree-Based Speculative Decoding](https://arxiv.org/abs/2608.26112) | 我们提出了TreeGraft，一种自适应多草稿器嫁接框架，通过结合不同成本的草稿器来构建共享草稿树，从而平衡草稿质量与延迟，提升基于树的投机解码效率。 |
| [^90] | [One Form to Transfer Them All: Pretraining Multilingual Language Models Beyond Native Orthography](https://arxiv.org/abs/2608.25904) | 本研究系统比较了不同输入表示（正字法文本、国际音标、罗马化）在多语言自回归预训练中的跨语言迁移效果，发现罗马化预训练在多种规模和语言对上表现最优，且优势随规模增大而扩大。 |
| [^91] | [Overview of SHROOM-Visions 2026: A Shared Task on Hallucination Detection in Large Vision-Language Models](https://arxiv.org/abs/2608.25662) | 本论文介绍了SHROOM-Visions 2026共享任务，该任务利用SHEEP数据集和五类幻觉分类体系，在四种语言中检测大型视觉-语言模型中的细粒度幻觉，以推进模型无关的幻觉检测研究。 |
| [^92] | [When Stale Constraints Go Unchecked: Budgeted Verification Failures in Inherited Agent Memory](https://arxiv.org/abs/2608.25553) | 该论文研究了在有限验证预算下，代理继承的过时约束未被检查导致验证失败的问题，并提出了通过重新分配验证槽位来减少这类错误的方法。 |
| [^93] | [MathAdv: What Theorem Provers Know, Reason, Formalize, and Generalize](https://arxiv.org/abs/2608.25449) | 本文提出MathAdv基准，通过多任务和专家变换系统评估定理证明器，揭示了形式化瓶颈、领域差异及自然语言对模型影响的异质性。 |
| [^94] | [Trust the Mass: Forced Weights in KV-Cache Eviction](https://arxiv.org/abs/2608.25230) | 本文发现KV缓存驱逐中保留最大权重已接近最优，已发布方法间的差异主要源于存储方式而非选择策略，并揭示了评估中的内存与性能权衡。 |
| [^95] | [SeMoCo: A Semantic-First Motion Codec for Motion Language Modeling](https://arxiv.org/abs/2608.24334) | 提出语义优先的运动编解码器SeMoCo，将每个运动标记分解为一个语义标记与残差运动学标记序列，配合双轴生成器分别建模语义演进与运动学细节以实现语言条件下的文本到动作生成，并构建了统一于SOMA表示的大规模多源人体运动数据集Ω-MotionVerse。 |
| [^96] | [Semantic Overlays: Mitigating Prompt Injection with Annotations Beyond Tokens and Steering Vectors](https://arxiv.org/abs/2608.23873) | 该论文提出了一种名为“语义覆盖层”的新技术，通过向模型输入添加非文本通道来缓解提示注入攻击，利用小型学习的适配器在冻结模型的残差流中创建带外注释，从而增强模型对片段身份的理解。 |
| [^97] | [JuryProbe: An Empirical Consensus-Risk Diagnostic for Routing Reference-Free Factuality Judge Panels to Grounded Verification](https://arxiv.org/abs/2608.20607) | 本文提出JuryProbe，一种通过仅假阴性相关性和假共识提升度来诊断无参考事实性评审团共识风险的方法，并在高风险时路由到有参考验证，以减少因共享盲点导致的错误接受。 |
| [^98] | [A Declarative-Procedural Perspective on Expert Routing in Bilingual Mixture-of-Experts Language Models](https://arxiv.org/abs/2608.15102) | 本研究通过陈述性-程序性框架分析双语MoE模型，发现无课程训练的混合数据基线比顺序课程训练展现出更强的语言类别专家路由特化，挑战了传统课程学习假设。 |
| [^99] | [Why Knowing Both Hops Is Not Enough: Understanding Two-Hop Generalization in Language Models](https://arxiv.org/abs/2608.07261) | 本文通过受控符号环境中的变换器训练和机制分析，揭示了两跳泛化中第二跳分布内成功而分布外失败的根本原因，即一致中间表示的出现与层间不匹配。 |
| [^100] | [Search, Inspect, Fetch: Exploiting Structure-Aware Boolean Retrieval for Deep-Search Agents](https://arxiv.org/abs/2608.02751) | 提出基于布尔查询语言的Sieve搜索-检查-获取策略，通过结构感知检索在提升深度搜索智能体准确性的同时，将token消耗降低20.7%-50.6%。 |
| [^101] | [Where Steering Signals Come From: Activation Source Selection in Activation Steering](https://arxiv.org/abs/2607.25270) | 该论文首次将激活引导中常被忽视的“激活源选择”作为核心研究对象，发现引导信号的有效性关键取决于激活是否取自模型即将执行目标行为的“执行边界状态”，而非源文本中是否包含期望行为。 |
| [^102] | [Set-shifting Behavioral Test for Harnessed Agents](https://arxiv.org/abs/2607.13396) | 该论文借鉴认知心理学中的“定势转换”概念，提出了一种通过在冗余工具库中隐藏地切换可靠工具组来测试LLM智能体适应能力的行为测试方法，并发现不同模型面对相同切换时表现出截然不同的行为模式。 |
| [^103] | [An LLM-Based Framework for Intent-Driven Network Topology Design](https://arxiv.org/abs/2607.00292) | 该论文提出了一个基于大语言模型的意图驱动网络拓扑设计框架，通过结合分层建模与系统性验证的约束驱动流水线，使LLM能够从自然语言需求生成结构有效且符合约束的网络拓扑，并发布了包含四个真实网络场景的公开基准数据集用于多模型评估。 |
| [^104] | [LV-ROVER-MLT: Low-Resource Maltese OCR by Synthetic Fine-Tuning and Multi-Stream Arbitration](https://arxiv.org/abs/2607.00250) | 该论文提出LV-ROVER-MLT系统，通过合成数据微调Tesseract 5并结合五路识别流与词典门控词级仲裁，在仅57页标注数据的低资源条件下以0.0074的字符错误率赢得DocEng 2026马耳他语OCR竞赛冠军，同时发布了36,803对的马耳他语OCR新语料库。 |
| [^105] | [ProfileFoundry: A Synthetic Person-Object Substrate for Privacy, Memory, and Tool-Use Evaluation in LLM Agent](https://arxiv.org/abs/2606.26403) | 本文提出了PROFILEFOUNDRY，一个确定性生成器，发布了10万个跨八个地区的合成人物对象数据集，包含丰富的个人状态、关系、事件和来源信息，以解决真实用户数据难以共享和评估的问题，为LLM智能体的隐私、记忆和工具使用评估提供可靠基础。 |
| [^106] | [CASPER in the Machine: Insights into Character Variety in LLM-Generated Stories](https://arxiv.org/abs/2606.22454) | 该研究借鉴叙事学理论，从风格化、完整性等八个维度自动分析并对比LLM生成故事与人类撰写故事中的角色刻画，探究两者角色是否相似以及LLM能否生成角色多样的故事。 |
| [^107] | [Does Finetuning with Scientific Data Increase Hallucinations? A Multi-domain Factuality Evaluation of LLMs](https://arxiv.org/abs/2606.21359) | 提出SciFactCheck多领域基准，涵盖五个科学领域的2,500个提示并针对三种幻觉类型进行评估，发现用科学数据微调的大语言模型相比其通用基础模型在各类幻觉类型上的事实可靠性均出现下降。 |
| [^108] | [Closing the Operational Gap in Semantic Caching](https://arxiv.org/abs/2606.19719) | 该论文指出PR-AUC指标会误导语义缓存系统的部署决策，提出了缓存感知的P-CHR AUC指标和运营保留率ORR，并将离线与部署质量间的运营差距分解为可恢复的阈值效用部分和由数据集正例率决定的不可约简结构部分。 |
| [^109] | [Securing Multi-Agent GIS Systems: Risk Evaluation and Prompt Hardening Optimization](https://arxiv.org/abs/2606.17092) | 该论文提出了一个面向安全的多智能体GIS系统框架，通过基于状态机的模块化编排、自适应攻击者红队测试评估以及将提示视为结构化签名的提示优化方法，实现风险识别、评估与缓解，提升系统安全性。 |
| [^110] | [TokenPilot: Cache-Efficient Context Management for LLM Agents](https://arxiv.org/abs/2606.17016) | TokenPilot提出了一种双粒度上下文管理框架，通过全局的感知摄取压缩来稳定提示前缀并消除环境噪声，结合局部的生命周期感知驱逐机制仅在任务相关性过期时卸载内容，从而在降低大语言模型智能体推理成本的同时保持提示缓存的连续性。 |
| [^111] | [Persuasion Index: A Theory-Guided Framework for Persuasion Analysis](https://arxiv.org/abs/2606.14580) | 提出了基于心理学与传播学说服理论的“说服指数”（PI）——一个包含15个维度的模块化说服分类体系及其由词典和规则构建的55个子特征的透明实现，在多个英语论证文本数据集上验证了其作为轻量级共享特征空间可有效解释与说服相关的修辞模式。 |
| [^112] | [MemoryCard: Topic-Aware Multi-Modal Clue Compression for Long-Video Question Answering](https://arxiv.org/abs/2606.05917) | MemoryCard提出了一种基于视频记忆的增强框架，通过将长视频组织成主题感知、语义连贯的记忆卡片来替代碎片化帧证据，从而提升视觉语言模型在长视频问答中捕捉事件级语义的能力。 |
| [^113] | [The Granularity Gap: A Multi-Dimensional Cross-Generational Audit of Sycophancy in Gemini Models](https://arxiv.org/abs/2606.05183) | 该论文揭示了安全评估中“通过/失败”二元判定与谄媚行为连续评分之间存在不可弥合的“粒度差距”，表明现有评估方法无法充分捕捉模型取悦用户的细微行为。 |
| [^114] | [Self-Evaluation Is Already There: Eliciting Latent Judge Calibration in Base LLMs with Minimal Data](https://arxiv.org/abs/2606.05122) | 基础大语言模型天生就具备预测外部裁判如何为自身输出打分的潜在能力，所提出的SEE方法仅需160个样本（比强化学习基线少约31倍），通过“校准耦合强化学习+掩码蒸馏”的简短循环即可激发该能力，在保持回答质量的同时显著提升自我评估的校准水平。 |
| [^115] | [RealClawBench: Live OpenClaw Benchmarks from Real Developer-Agent Sessions](https://arxiv.org/abs/2606.03889) | RealClawBench是一个基于真实OpenClaw开发者-智能体会话构建的实时基准测试框架，通过重建执行环境和确定性可验证评分器两大机制，将真实用户请求转化为281个可复现、可自动评分的任务，从而捕捉已部署智能体的真实使用分布与难度。 |
| [^116] | [CRAM: Centroid-Routing and Adaptive MoE for Multimodal Continual Instruction Tuning](https://arxiv.org/abs/2606.02502) | 本文提出CRAM方法，通过中心路由和自适应混合专家模型，在缓解多模态持续指令微调中灾难性遗忘的同时，提升参数效率。 |
| [^117] | [Benchmarking LLM-as-a-Judge for Long-Form Output Evaluation](https://arxiv.org/abs/2606.01629) | 该论文提出了LongJudgeBench基准，用于系统评估LLM评判者在多样化真实场景和评判协议下对长文本输出进行评估的可靠性。 |
| [^118] | [DiffuSent: Towards a Unified Diffusion Framework for Aspect-Based Sentiment Analysis](https://arxiv.org/abs/2606.01323) | 提出非自回归扩散框架DiffuSent，将方面级情感分析的全部七个子任务统一建模为边界去噪扩散过程，并配合对比去噪训练策略，有效解决了多词方面词与观点词的边界不敏感及重复预测问题。 |
| [^119] | [Auditing LLM Benchmarks with Item Response Theory](https://arxiv.org/abs/2605.30504) | 本文提出基于项目反应理论的指标，能以95%的精确率从七个LLM基准测试中识别错误标注样本，并揭示奖励模型偏重风格偏好而非事实知识、且可能存在基准污染的问题。 |
| [^120] | [LongDS-Bench: On the Failure of Long-Horizon Agentic Data Analysis](https://arxiv.org/abs/2605.30434) | 提出LongDS基准，基于真实Kaggle笔记本构建68个长时程多轮数据分析任务，揭示当前最先进模型在维护和演变分析状态方面存在严重缺陷，最佳模型平均准确率仅48.45%，且长时程错误占失败案例的52%–69%。 |
| [^121] | [Human Label Variation as Stable Signal: Learning Annotator-Specific Explanation Behavior via Cross-Annotator Preference Optimization](https://arxiv.org/abs/2605.28802) | 本文提出跨标注者偏好优化（CAPO）方法，通过将目标标注者的解释与其他标注者对同一输入的有效但特异性较低的标注进行对比训练，使大语言模型能够学习并复现特定标注者的解释行为。 |
| [^122] | [A Wolf in Sheep's Clothing: Targeted Routing Hijacking in Federated RAG](https://arxiv.org/abs/2605.28112) | 论文揭示了联邦RAG系统中的一种新型“路由劫持”攻击：恶意客户端通过伪造语义画像劫持目标查询，在三种主流路由架构中均能引发证据缺失、投毒、错误答案和幻觉等严重后果，且现有防御无法有效应对。 |
| [^123] | [Do LLM Agents Mirror Socio-Cognitive Effects in Power-Asymmetric Conversations?](https://arxiv.org/abs/2605.17694) | 研究发现LLM智能体在权力不对称对话中会展现出类似人类的社会认知效应（如语言协调、权威偏见），这既可能促成理想行为，也可能导致对不安全请求的顺从。 |
| [^124] | [SkillSafetyBench: Evaluating Agent Safety under Skill-Facing Attack Surfaces](https://arxiv.org/abs/2605.12015) | 该论文提出 SkillSafetyBench 基准，通过 155 个覆盖 6 大风险领域的对抗性案例，首次系统评估了隐藏在技能指导、本地文件等非用户输入中的攻击面，发现此类攻击可稳定诱发大语言模型智能体的不安全行为。 |
| [^125] | [Psychologically Potent, Computationally Invisible: LLMs Generate Social-Comparison-Eliciting Posts They Fail to Detect](https://arxiv.org/abs/2605.01017) | 本研究构建了小红书社会比较基准，发现LLM能生成引发社会比较的帖子，但基于提示的检测器难以稳定识别该信号，揭示生成与检测能力的不对称性。 |
| [^126] | [G-Loss: Graph-Guided Fine-Tuning of Language Models](https://arxiv.org/abs/2604.25853) | 提出了一种图引导的损失函数G-Loss，通过结合半监督标签传播与文档相似度图来捕捉全局语义结构，引导预训练语言模型学习更具判别性和鲁棒性的嵌入表示。 |
| [^127] | [Why are all LLMs Obsessed with Japanese Culture? On the Hidden Cultural and Regional Biases of LLMs](https://arxiv.org/abs/2604.21751) | 本文提出了一个涵盖24种语言的文化相关开放问题数据集CROQ，通过评估发现大语言模型在回答文化问题时存在明显的地区偏见（尤其偏爱日本），且低资源语言提示下的输出多样性显著低于高资源语言。 |
| [^128] | [Select, Label, Evaluate: Active Testing in NLP](https://arxiv.org/abs/2603.21840) | 该论文形式化了NLP中的主动测试框架，通过选择最具信息量的测试样本进行标注，在18个数据集上的实验表明可将标注成本降低高达95%，同时模型性能估计与完整测试集的准确度差异保持在1%以内。 |
| [^129] | [Large Reasoning Models Struggle to Transfer Parametric Knowledge Across Scripts](https://arxiv.org/abs/2603.17070) | 该研究发现大型推理模型的跨语言知识迁移失败主要由文字系统差异（而非语言或语系）导致，并提供源语言关键实体及合成SFT训练样本可显著缓解这一问题。 |
| [^130] | [Diverging Transformer Predictions for Human Sentence Processing: A Comprehensive Analysis of Agreement Attraction Effects](https://arxiv.org/abs/2603.16574) | 该研究基于惊讶度机制系统评估了十一个自回归Transformer模型在英语一致性吸引效应上的表现，发现模型预测在介词短语配置上与人类阅读时间数据一致，但在宾语提取关系从句配置上性能显著下降、模型间预测分歧明显且均无法复制人类的不对称干扰模式，从而表明当前Transformer模型不能解释人类的形态句法处理过程。 |
| [^131] | [The Company You Keep: How LLMs Respond to Dark Triad Traits](https://arxiv.org/abs/2603.04299) | 本研究系统分析大语言模型对表达黑暗三联征人格特质的用户提示的回应，发现尽管所有模型以纠正性行为为主，部分模型仍会产生强化或模棱两可的输出，凸显了构建能够可靠检测并应对用户从良性升级为有害请求的更安全对话系统的必要性。 |
| [^132] | [From Leaky Thoughts to Private Reasoning: Controlling What LRMs Say to Themselves](https://arxiv.org/abs/2602.24210) | 该论文提出SFT数据集与分阶段解码策略，通过提升大型推理模型在推理过程中的指令遵循能力，有效减少推理轨迹中的隐私泄露。 |
| [^133] | [FENCE: A Financial and Multimodal Jailbreak Detection Dataset](https://arxiv.org/abs/2602.18154) | FENCE是一个面向金融应用的双语多模态越狱检测数据集，基于它训练的检测器在分布内数据上达到99%的准确率，有效弥补了金融领域越狱检测资源的匮乏。 |
| [^134] | [CoFrGeNet: Continued Fraction Architectures for Language Generation](https://arxiv.org/abs/2601.21766) | 受连分数启发，本文提出CoFrGeNet新架构，其组件能以远少于原有参数量的方式替代Transformer中的多头注意力和前馈网络，并可即插即用，几乎无需改动现有的训练与推理流程。 |
| [^135] | [Beyond the Rabbit Hole: Mapping the Relational Harms of QAnon Radicalization](https://arxiv.org/abs/2601.17658) | 本文通过分析12747个来自r/QAnonCasualties支持社群的个人故事，构建计算流程提取主题特征并聚类为六种激进化人格画像，首次揭示了不同激进化模式与特定情感伤害（如愤怒、厌恶、恐惧、悲伤）之间的对应关系，填补了阴谋论研究中对信徒身边人所受关系伤害的关注空白。 |
| [^136] | [Aligning Agentic World Models via Knowledgeable Experience Learning](https://arxiv.org/abs/2601.13247) | 提出WorldMind框架，通过综合环境反馈自主构建符号化世界知识库，使LLM智能体世界模型无需昂贵再训练即可遵循物理法则、避免物理幻觉。 |
| [^137] | [Tracing the complexity profiles of different linguistic phenomena through the intrinsic dimension of LLM representations](https://arxiv.org/abs/2601.03779) | 该研究发现大语言模型表示的内在维度可作为语言复杂度的有效标记，并列/从属、右分支/中心嵌套、无歧义/歧义修饰等经典语言学复杂度对比在六个LLM的各层中均一致地体现为ID差异，且不同对比的出现位置和峰值阶段各不相同。 |
| [^138] | [The Instability of Safety: How Random Seeds and Temperature Expose Inconsistent LLM Refusal Behavior](https://arxiv.org/abs/2512.12066) | 该研究揭示大语言模型的安全拒绝决策在随机种子和温度变化下并不稳定，18-28%的有害提示词会出现“拒绝”与“配合”之间的决策翻转，且温度越高稳定性越差，表明单次安全评估无法真实反映模型的安全对齐水平。 |
| [^139] | [OmniFusion: Simultaneous Multilingual Multimodal Translations via Modular Fusion](https://arxiv.org/abs/2512.00234) | OmniFusion提出一种端到端的模块化融合方法，将多模态基础模型与翻译大语言模型结合，实现低延迟的同时多语言多模态翻译。 |
| [^140] | [Learning a Single Token to Replace Long System Prompts in LLMs](https://arxiv.org/abs/2511.23271) | 该论文提出一种轻量级训练框架，通过学习单个“行为等价 Token（[BE]）”替代长系统提示词，在不更新模型权重、不使用辅助压缩模型和标注数据的情况下，实现高达 3000 倍的提示词压缩并保留约 98% 的下游行为效果。 |
| [^141] | [SMRC: Aligning Large Language Models with Student Reasoning for Mathematical Error Correction](https://arxiv.org/abs/2511.14684) | 该论文提出SMRC方法，将学生数学推理建模为多步序贯决策问题并利用蒙特卡洛树搜索探索最优纠正路径，使大语言模型能够像教师一样系统地检测和纠正学生的数学推理错误。 |
| [^142] | [Think-at-Hard: Dynamic Looped Transformers for Improved Reasoning](https://arxiv.org/abs/2511.08577) | 针对循环Transformer中存在的“潜在过度思考”现象，提出TaH方法，利用轻量级神经决策器仅在可能出错的token处动态触发潜在迭代，从而在参数受限条件下将大语言模型的推理性能提升高达7.3%。 |
| [^143] | [Multilingual Lexical Feature Analysis of Spoken Language for Predicting Major Depression Symptom Severity](https://arxiv.org/abs/2511.07011) | 该研究基于来自英国、荷兰和西班牙467名参与者的多语言智能手机录音数据，利用可解释的线性混合效应模型识别出与抑郁症状严重程度相关的口语词汇特征，并通过机器学习验证了这些特征对PHQ-8评分预测的增益作用。 |
| [^144] | [Roleplaying with Structure: Synthetic Therapist-Client Conversation Generation from Questionnaires](https://arxiv.org/abs/2510.25384) | 该论文提出SQPsych生成管线，在不泄露敏感数据的前提下，利用真实的结构化来访者档案和心理问卷生成合成治疗师-来访者对话语料库，经专家评估证明微调后的LLM在治疗师角色扮演上显著更优。 |
| [^145] | [Quantifying Affective Bias in Low-Resource Media: Large-Scale Emotion Profiling of Bengali Headlines](https://arxiv.org/abs/2510.17252) | 该研究利用Gemma 3 4B零样本推理对30万条孟加拉语新闻标题进行大规模情感分析，发现愤怒、悲伤、失望和恐惧等负面情感在低资源语言新闻媒体中占主导地位，从而量化揭示了孟加拉语数字新闻中存在的情感偏见。 |
| [^146] | [PRISM: Agentic Retrieval with LLMs for Multi-Hop Question Answering](https://arxiv.org/abs/2510.14278) | PRISM是一个基于大语言模型的智能体式检索框架，通过问题分析器、选择器和添加器三个专门智能体的迭代协作，以高精确率和高召回率检索证据，显著提升多跳问答性能。 |
| [^147] | [OceanGym: A Benchmark Environment for Underwater Embodied Agents](https://arxiv.org/abs/2509.26536) | OceanGym是首个面向水下具身智能体的综合性基准环境，涵盖八个真实任务领域和基于多模态大语言模型的统一智能体框架，实验揭示当前最先进的智能体与人类专家之间仍存在显著差距。 |
| [^148] | [Steering Multimodal Large Language Models Decoding for Context-Aware Safety](https://arxiv.org/abs/2509.19212) | 提出了轻量级、模型无关的解码框架SafeCoDe，通过对比真实与高斯噪声图像突出视觉敏感token，并结合场景级推理动态调整拒绝行为，从而平衡多模态大语言模型安全决策中的过度敏感与敏感不足问题。 |
| [^149] | [The Telephone Game: Evaluating Semantic Drift in Unified Models](https://arxiv.org/abs/2509.04438) | 该论文提出语义漂移协议（SDP）和平均累积漂移（MCD）指标，通过模拟传声筒游戏的多轮交替生成方式，首次量化了统一模型在理解与生成能力组合使用时出现的语义漂移问题，揭示了在孤立基准上表现优异的模型在跨任务一致性上可能严重失效。 |
| [^150] | [Automatic Pronunciation Error Detection and Correction of the Holy Quran's Learners Using Deep Learning](https://arxiv.org/abs/2509.00094) | 该论文提出了一套98%自动化的《古兰经》诵读数据构建流程，发布了848小时音频数据集（28.6万条标注语句）以及涵盖Tajweed规则的基准qdat_bench，实现了对《古兰经》学习者发音错误的自动检测与纠正。 |
| [^151] | [Beyond the Rosetta Stone: Unification Forces in Generalization Dynamics](https://arxiv.org/abs/2508.11017) | 通过合成数据上的小型Transformer实验，本文揭示了跨语言知识迁移的关键在于表示统一性，这取决于信息性和可提取性，并提供了统一理论解释多语言模型中的迁移现象。 |
| [^152] | [Cognitive Chain-of-Thought (CoCoT): Structured Multimodal Reasoning about Social Situations](https://arxiv.org/abs/2507.20409) | 提出认知思维链框架，通过感知、情境、规范三个认知阶段结构化视觉语言模型的多模态社会情境推理，在意图消歧、心智理论、社会常识推理等多个任务上取得一致的显著提升。 |
| [^153] | [Pruning Laws for Large Language Models](https://arxiv.org/abs/2504.04342) | 提出了“剪枝定律”，一种将剪枝后性能与未剪枝性能和剪枝比例关联起来的简单可解释的缩放关系，可在多种模型、剪枝策略和任务上以小于7%的平均外推误差准确预测大语言模型剪枝后的性能。 |
| [^154] | [PRISM: Self-Pruning Intrinsic Selection Method for Training-Free Multimodal Data Selection](https://arxiv.org/abs/2502.12119) | PRISM提出了一种免训练的多模态数据选择方法，通过首次揭示视觉特征分布的各向异性及其引发的全局语义漂移现象，在不依赖昂贵推理或训练的情况下高效剪除冗余指令数据，显著降低了数据选择的计算成本。 |
| [^155] | [Long Story Short: Story-level Video Understanding from 20K Short Films](https://arxiv.org/abs/2406.10221) | 提出了目前最大的公开可用电影数据集SF20K，包含20,143部短片共3,582小时视频，通过业余短片规避数据泄露问题，推动故事级长视频理解研究。 |
| [^156] | [Evaluating the Performance of Large Language Models on GAOKAO Benchmark.](http://arxiv.org/abs/2305.12474) | 本文介绍了一个基于高考考试问题的基准测试GAOKAO-Benchmark，用于评估大型语言模型在客观和主观问题方面的表现。通过对ChatGPT模型的评估，研究发现其在客观问题方面表现出色，同时也揭示了其不足之处和改进的方向。 |

# 详细

[^1]: 从文本语料库学习人类语言的形式化局限

    A Formal Limitation on Learning Human Language From Textual Corpora

    [https://arxiv.org/abs/2608.28560](https://arxiv.org/abs/2608.28560)

    该论文从信息论角度证明，话语形式对意义存在固有的不可约减的不确定性，因此任何文本表示（包括大语言模型的隐藏状态）在从话语恢复说话者意图意义上的概率都存在无法超越的上界。

    

    听者能否仅凭话语的形式恢复说话者所表达的意思？我们从信息论角度回答了这一问题，并适用于由任何文本特征化器所给出的听者，包括当代大型语言模型的隐藏状态。我们将语言使用建模为意义、语境和话语的联合分布，推导出解码器从话语表示中恢复说话者意图意义的概率上界。这些上界由形式关于意义所留下的不确定性决定，该不确定性分为不可约减的部分，以及只有（语言外的）语境才能解决、而仅凭话语本身永远无法解决的部分。由于这些量是语言所固有的，任何表示——无论其由多少文本或监督产生——都无法超越它们；无论意义空间是离散的还是连续的，这些界均成立。在人工语言、汉语零代词消解等任务上的实验（原文摘要在此处截断）

    arXiv:2608.28560v1 Announce Type: new  Abstract: Can a listener recover what a speaker means from the form of an utterance alone? We answer this question information-theoretically, and for a listener given by any featurizer of text, including the hidden states of contemporary large language models. Modeling language use as a joint distribution over meanings, contexts, and utterances, we derive upper bounds on the probability that a decoder recovers a speaker's intended meaning from a representation of the utterance. The bounds are governed by the uncertainty that form leaves about meaning, which splits into an irreducible part and a part that only (extralinguistic) context, but never the utterance alone, can resolve. Because these quantities are intrinsic to language, no representation, however much text or supervision produced it, can surpass them; the bounds hold whether the space of meanings is discrete or continuous. Experiments on artificial languages, Mandarin zero-pronoun resolu
    
[^2]: 当机器人听错我们的话：映射语音控制具身AI的安全风险

    When Robots Mishear Us: Mapping the Safety Risks of Voice-Controlled Embodied AI

    [https://arxiv.org/abs/2608.28518](https://arxiv.org/abs/2608.28518)

    该论文首次系统研究了语音识别错误对具身AI安全性的影响，通过将模拟的ASR错误与安全基准相结合，揭示了语音识别错误可导致有害指令被具身AI模型接受并执行这一新型安全风险。

    

    我们研究了用户输入中的自动语音识别（ASR）错误是否会导致具身AI（EAI）模型产生不安全的输出。我们发现，ASR错误可能导致有害指令被EAI模型接受并执行，从而降低安全性。我们模拟了ASR错误，并将其与现有的安全基准（SafeAgentBench和POEX）相结合，以评估不同类型的错误如何影响具身AI的安全性。我们发现，某些错误保留了语义结构但增加了有害的歧义性，而另一些错误则削弱了模型的拒绝行为，使得不安全的计划得以生成和执行。我们表明，在某些情况下，自动纠正ASR错误可以降低风险，但这并不总是有效。总体而言，我们证明了ASR错误会给具身AI带来显著的安全风险。

    arXiv:2608.28518v1 Announce Type: cross  Abstract: We investigate whether automatic speech recognition (ASR) errors in user input can lead to unsafe outputs from Embodied AI (EAI) models. We find that ASR errors can lead to harmful instructions being accepted and executed by EAI models, thereby reducing safety. We simulate ASR errors and combine them with existing safety benchmarks (SafeAgentBench and POEX) to evaluate how different errors affect embodied AI safety. We find that some of them preserve semantic structure but increase harmful ambiguity, while others weaken the model refusal behaviour and allow unsafe plans to be generated and executed. We show that in some cases automatic correction of ASR errors can reduce the risk, but this is not always effective. Overall, we show that ASR errors lead to significant safety risks for embodied AI.
    
[^3]: 基于自监督语音表示的音素级与词级指标用于强制对齐评估

    Phoneme- and Word-Level Metrics Using Self-Supervised Speech Representations for Forced Alignment Evaluation

    [https://arxiv.org/abs/2608.28508](https://arxiv.org/abs/2608.28508)

    本文提出两个基于自监督语音表示的无参考强制对齐评估指标PCMI和WACS，无需人工标注时间戳即可在大规模多语言场景（85种语言）下有效评估对齐质量。

    

    强制对齐评估通常需要人工标注的时间戳，这限制了大规模和多语言分析。我们提出了两个基于自监督（SSL）语音表示的语料库级指标，用于无参考的强制对齐评估：音素-聚类互信息和词声学一致性得分。PCMI衡量对齐后的音素标签与从SSL语音表示中归纳出的聚类之间的一致性，而WACS利用词表示序列之间的动态时间规整相似度来衡量重复词实现的声学一致性。通过随机和系统性扰动实验，我们证明PCMI和WACS在对齐扰动下表现出一致的退化。我们进一步在来自FLEURS的85种语言上对多个对齐系统分析这些指标的表现，并在DoReCo中45种语言的人工标注对齐上验证它们，同时在两个音系复杂的语料上对指标进行评估。

    arXiv:2608.28508v1 Announce Type: new  Abstract: Forced alignment evaluation typically requires manually annotated timestamps, limiting large-scale and multilingual analysis. We introduce two corpus-level metrics based on self-supervised (SSL) speech representations for reference-free forced alignment evaluation: Phoneme-Cluster Mutual Information (PCMI) and Word Acoustic Consistency Score (WACS). PCMI measures agreement between aligned phoneme labels and clusters induced from SSL-speech representations, while WACS measures consistency of repeated word realizations using dynamic time warping similarity between word representation sequences. Using both random and systematic perturbations, we show that PCMI and WACS degrade consistently under alignment perturbations. We further analyze the metrics across multiple alignment systems on 85 languages from FLEURS, validate them against manually annotated alignments from 45 languages in DoReCo, and evaluate them on two phonologically complex l
    
[^4]: 混沌中的阶梯：测试时扩展何时、如何（以及或许为何）提升大语言模型机器翻译

    Ladders in Chaos: When, How, (and Perhaps Why) Does Test-Time Scaling Improve LLM Machine Translation

    [https://arxiv.org/abs/2608.28496](https://arxiv.org/abs/2608.28496)

    本研究系统对比了序列式与并行式测试时扩展在机器翻译中的表现，发现序列式采样具有更高的性能上限，能显著提升翻译流畅度和自然度，但在大推理预算下可能损害翻译准确性，并部分解释了其作用机制。

    

    大语言模型（LLM）的两种测试时扩展形式已成为有效且被广泛采用的范式：序列式扩展，即后续的答案尝试依赖于先前的尝试；以及并行式扩展，例如独立同分布采样加重排序。在本研究中，我们考察了它们在翻译中的特性。首先，我们的研究表明，序列式采样具有更高的性能上限，能够提供更加多样且有效的样本池，尤其是在较小的采样预算下。其次，我们通过多维度的手动分析探究了测试时扩展的本质。对Best-of-N翻译结果的人工分析表明，序列式采样显著提升了翻译的流畅度和自然度，但当推理预算较大时可能会损害准确性。最后，我们提出了序列式扩展改进机器翻译机制的一种解释。我们的受控分析部分将其归因于（原文在此处截断）

    arXiv:2608.28496v1 Announce Type: new  Abstract: Two forms of test-time scaling for Large Language Models (LLMs) have emerged as effective and widely adopted paradigms: sequential, in which later answer attempts depend on earlier ones, and parallel, such as i.i.d. sampling with reranking. In this study, we investigate their properties in translation. First, our study shows that sequential sampling has a higher performance ceiling, providing a more diverse and effective pool of samples, particularly under smaller sampling budgets. Second, we interrogate the nature of test-time scaling through a multidimensional manual analysis. Human analysis of the Best-of-$N$ translations demonstrates that sequential sampling substantially improves translation fluency and naturalness, but can degrade accuracy when inference budgets are large. Finally, we suggest an explanation of the mechanism through which sequential scaling improves machine translation. Our controlled analysis partially attributes t
    
[^5]: NL2AGBench：面向AlphaGeometry的大语言模型自动形式化基准测试

    NL2AGBench: Benchmarking LLM Auto-Formalization for AlphaGeometry

    [https://arxiv.org/abs/2608.28481](https://arxiv.org/abs/2608.28481)

    该论文提出了NL2AGBench基准测试，用于评估大语言模型将英文几何问题自动形式化为AlphaGeometry兼容形式化表示的能力，从而解决神经符号几何系统中手动转换输入的可用性瓶颈。

    

    大语言模型（LLM）的最新进展展示了其在自然语言理解和数学推理方面的强大能力。然而，它们将非形式化数学问题转化为形式化表示的能力仍然缺乏充分探索。这一局限对于神经符号几何系统（如AlphaGeometry）尤为重要，因为其定理证明引擎需要以专门的领域特定语言（DSL）作为输入。尽管AlphaGeometry达到了接近国际数学奥林匹克（IMO）金牌选手的水平，但手动将自然语言问题转换为其形式化语法仍然是一个显著的可用性瓶颈。为了应对这一挑战，我们提出了自然语言到AlphaGeometry基准测试（NL2AGBench），用于评估大语言模型将英文几何问题翻译成与AlphaGeometry兼容的形式化表示的能力。NL2AGBench使用AlphaGeometry内部基于执行的验证来评估翻译质量……

    arXiv:2608.28481v1 Announce Type: new  Abstract: Recent advances in large language models (LLMs) have demonstrated strong capabilities in natural language understanding and mathematical reasoning. However, their ability to translate informal mathematical problems into formal representations remains underexplored. This limitation is particularly important for neuro-symbolic geometry systems such as AlphaGeometry, whose theorem-proving engine requires inputs in a specialized domain-specific language (DSL). Although AlphaGeometry achieves near-IMO gold-medalist performance, manually converting natural-language problems into its formal syntax remains a significant usability bottleneck. To address this challenge, we introduce the Natural Language to AlphaGeometry Benchmark (NL2AGBench), which evaluates LLMs in translating English geometry problems into AlphaGeometry-compatible formal representations. NL2AGBench uses execution-based verification within AlphaGeometry to assess translation qua
    
[^6]: 盲人与大象：探究大语言模型在长尾分歧知识下的认知短视

    Blind Men and the Elephant: Probing the Epistemic Myopia of LLMs under Long-Tail Divergent Knowledge

    [https://arxiv.org/abs/2608.28478](https://arxiv.org/abs/2608.28478)

    提出ElephantBench闭卷知识探针基准，揭示大语言模型对长尾分歧事实存在“认知短视”——即使最强模型也只能在52.4%的问题上同时召回两个版本的答案，且扩大模型规模和增强推理能力均无法消除这种不完整性。

    

    事实性问答（QA）通常假设存在单一的标准答案，这掩盖了大语言模型（LLM）是否保留了对长尾事实的分歧性描述。为了填补这一空白，我们提出了ElephantBench，这是一个包含1,094个问题的闭卷知识探针，这些问题通过一个可审计的基于图的流水线生成。该流水线从低曝光的网络语料库中检索相关文档，识别自然存在的分歧，并将其转换为多版本答案的QA记录。每个答案都对照原始文档和权威公共网络来源进行验证，随后由人工标注员审核。在32个模型的评估中，即使是最强的模型也仅在52.4%的问题上同时召回两个版本的答案，而在几乎所有剩余的问题上，它只能回忆起其中一个版本而遗漏另一个。扩大模型规模和增强推理时的推理能力可以提升召回率，但无法消除这种不完整性。语料库分析进一步表明

    arXiv:2608.28478v1 Announce Type: new  Abstract: Factual question answering (QA) typically assumes a single canonical answer, obscuring whether large language models (LLMs) retain divergent accounts of long-tail facts. To address this gap, we introduce ElephantBench, a closed-book knowledge probe comprising 1,094 questions generated through an auditable graph-based pipeline. The pipeline retrieves related documents from a low-exposure web corpus, identifies naturally occurring disagreements, and converts them into multi-account QA records. Each answer is verified against the originating documents and authoritative public web sources and is then reviewed by human annotators. Across 32 models, even the strongest model recovers both accounts on only 52.4% of questions, while on nearly all remaining questions it recalls one account but omits the other. Scaling model size and inference-time reasoning improve recall but do not eliminate this incompleteness. Corpus analysis further shows that
    
[^7]: ContextPilot：通过细粒度强化学习教会智能体进行主动式上下文管理

    ContextPilot: Teaching Agents for Proactive Context Management via Fine-grained RL

    [https://arxiv.org/abs/2608.28476](https://arxiv.org/abs/2608.28476)

    ContextPilot通过扩展上下文管理工具集并引入细粒度的强化学习信用分配机制，教会智能体在长时程任务中主动、高效地管理持续增长的工作上下文。

    

    长时程智能体任务要求大语言模型（LLM）在多轮交互中迭代地检索、整合并维护分散的信息，但保留所有交互历史会导致工作上下文持续增长。近期的主动式上下文管理方法允许模型借助专用工具编辑自己的工作上下文，但仍面临三个关键局限：（1）工具集有限，仅限于搜索、删除和摘要，不支持全局规划、长期记忆和自适应压缩；（2）探索效率低下，尽管各类上下文管理操作对最终结果的影响各异，却将其一视同仁地对待；（3）粗粒度的信用分配，即在强化学习过程中将最终的轨迹级奖励分配给所有中间上下文编辑操作。为弥合这些差距，我们提出了ContextPilot，一个面向长时程任务的主动式上下文管理框架（原文摘要在此处截断）。

    arXiv:2608.28476v1 Announce Type: new  Abstract: Long-horizon agentic tasks require large language models (LLMs) to iteratively retrieve, integrate, and maintain dispersed information across multi-turn interactions, but preserving all interaction histories leads to a continuously growing working context. Recent proactive context management methods allow models to edit their own working context with specialized tools, yet they still face three key limitations: (1) a limited toolset restricted to search, deletion, and summarization, with no support for global planning, long-term memory, and adaptive compression; (2) inefficient exploration that treats context management actions uniformly despite their heterogeneous impacts on final outcomes; and (3) coarse-grained credit assignment that assigns the final trajectory-level reward to all intermediate context editing actions during RL. To bridge these gaps, we introduce ContextPilot, a proactive context management framework for long-horizon 
    
[^8]: 陌生人、粉丝还是同伴？对话者在基于人设的对话生成中作用的系统性研究

    Stranger, Fan, or Peer? A Systematic Study on the Role of Interlocutor in Persona-Based Dialogue Generation

    [https://arxiv.org/abs/2608.28467](https://arxiv.org/abs/2608.28467)

    该论文首次将人设对话系统中对话双方“个人简介可见性”在训练、推理、评估三个阶段进行独立分解研究，发现训练阶段的简介可见性比推理阶段更能决定模型是真正通过对话表达人设特征，还是简单复制简介文本。

    

    基于人设的对话系统通常以说话者的个人简介作为条件，但对话至少涉及两个参与者，并且在训练、推理和评估阶段，谁能访问谁的个人简介可能有所不同。先前的工作往往忽视了这些方面，从而掩盖了只有在训练、推理和评估三个阶段分别独立切换个人简介可见性时才会显现的机制——这种三阶段分解在以往研究中基本被当作单一因素处理。我们在一个包含对话及其说话者个人简介的数据集上研究了这一分解因素，改变目标说话者和对话者在训练和推理期间是否可以看到彼此的个人简介，并使用大语言模型作为评判者来执行作者身份识别。我们发现：(i) 训练时的可见性（相比推理时的可见性）更能决定模型是通过对话自然地表达人设特征，还是退而求其次地复制个人简介文本（这是一个已知的问题/现象……）

    arXiv:2608.28467v1 Announce Type: new  Abstract: Persona-based dialogue systems are usually conditioned on speaker biography, but dialogues involve at least two participants, and who has access to whose biography can vary across training, inference, and evaluation. Prior work often neglected these aspects, obscuring mechanisms that only appear when biography visibility is toggled separately across training, inference, and evaluation, a three-stage factorisation that prior work has largely treated as a single factor. We study this factorisation on a dataset of dialogues paired with speaker's biographies, varying whether the target and interlocutor speakers see each other's biographies during training and inference, and using an LLM as a judge to perform author identification. We find that (i) training-time visibility, more than inference-time visibility, determines whether models express persona traits through dialogue or fall back on copying biographical text (a known problem/phenomeno
    
[^9]: 获取、修复、保持：一种面向小型模型对话游戏智能体的诊断引导式后训练方案

    Acquire, Repair, Preserve: A Diagnosis-Guided Post-Training Recipe for Small-Model Dialogue Game Agents

    [https://arxiv.org/abs/2608.28458](https://arxiv.org/abs/2608.28458)

    该论文提出一种诊断引导的三步后训练方案（获取、修复、保持），使2B小型模型在对话游戏挑战中的clemscore从10.67大幅提升至38.92，同时保持了一般能力不退化。

    

    交互式对话游戏考验一种静态基准测试大多未能明确涉及的能力：模型必须在多轮对话中携带状态、解读反馈，并在不断变化的约束下选择有效动作。我们在LM Playschool Challenge中使用2B参数的开源权重模型研究这一场景，发现许多失败不仅是广泛的知识性失败，还包括局部决策失败：重复猜测、格式错误的动作，以及违反模型刚刚看到的反馈。这些诊断结果启发了一种围绕三个步骤组织的训练方案：通过监督微调获取广泛的游戏参与能力，使用轮次局部的偏好对在单一目标对话游戏族内修复可机械验证的失败，并保持这些对话游戏之外的一般能力。在官方最终评估中，我们的提交将公开clemscore从10.67提升至38.92，封闭域内得分从13.41提升至41.17，同时……

    arXiv:2608.28458v1 Announce Type: new  Abstract: Interactive dialogue games test a capability that static benchmarks largely leave implicit: a model must carry state across turns, interpret feedback, and choose valid actions under changing constraints. We study this setting in the LM Playschool Challenge with a 2B open-weight model, and find that many failures are not only broad knowledge failures but also local decision failures: repeated guesses, malformed actions, and violations of feedback that the model has just seen. These diagnostics motivate a training recipe organized around three steps: acquire broad game participation through supervised fine-tuning, repair mechanically verifiable failures within one targeted dialogue-game family using turn-local preference pairs, and preserve general capabilities beyond these dialogue games. In the official final evaluation, our submission improves public clemscore from 10.67 to 38.92 and closed in-domain score from 13.41 to 41.17, while app
    
[^10]: 滑动窗口注意力优于线性注意力

    Sliding-window beats linear attention

    [https://arxiv.org/abs/2608.28444](https://arxiv.org/abs/2608.28444)

    本研究表明，带sink的滑动窗口注意力（SWA）在多项下游任务和长上下文推理任务上的表现与经过后训练的线性注意力模型相当甚至更优，说明这个更简单的基线方法被严重低估了。

    

    由于二次方复杂度注意力的固有特性，大语言模型（LLM）消耗大量内存和能源。每个新token的成本都比前一个更高。对于每个新增的token，其键和值必须无限期地存储在内存中，这是不可持续的。为了解决二次方扩展问题，研究者们已提出多种替代方案，其中之一是将LLM改造为使用线性注意力。由于线性注意力有望以低成本实现最先进的性能并解决二次方扩展问题，这一想法引起了广泛关注。然而，这一研究方向尚未与更简单的基线方法进行恰当的比较。在本工作中，我们证明了带sink的滑动窗口注意力（SWA）的表现与经过后训练的线性注意力模型相当甚至更好。我们在多个LLM和多种下游任务上都观察到了这一结果。对于长上下文推理任务（Needle-in-a-Haystack和BABILong），SWA实现了大幅更高的性能。

    arXiv:2608.28444v1 Announce Type: new  Abstract: Due to the nature of quadratic attention, Large Language Models (LLMs) consume a lot of memory and energy. Every new token costs more than the previous one. For each additional token, the keys and values must be stored in memory indefinitely, which is unsustainable.   Several alternatives have been proposed to fix the quadratic scaling problem, one of which is retrofitting LLMs to use Linear Attention. This idea has attracted a lot of attention, given its promise to solve the quadratic scaling problem with state-of-the-art performance at low cost. However, this line of research has not been properly compared to simpler baselines.   In this work, we show that Sliding Window Attention (SWA) with sinks performs as well or better than post-trained Linear Attention models. We observe this across multiple LLMs on various downstream tasks. For long-context reasoning tasks (Needle-in-a-Haystack and BABILong), SWA achieves massively higher perfor
    
[^11]: 保真度并不足够：面向智能体数据手册提取的调度级监测方法

    Fidelity Is Not Enough: Dispatch-Level Instrumentation for Agentic Datasheet Extraction

    [https://arxiv.org/abs/2608.28439](https://arxiv.org/abs/2608.28439)

    仅靠保真度指标会漏掉智能体文档提取中的静默失败（如模型不开数据手册就编造答案），本文通过记录每次工具调用的调度级日志，构建失败归因分类器与静默失败检测器，能在不误报正常提取的情况下发现全部植入故障。

    

    有一个模型在从未打开数据手册的情况下通过了我们的保真度检查。我们在为内部提取服务筛选模型时发现了它：一个结构化输出约束悄悄禁用了工具调用，而该模型依然给出了回答，并编造了源文本。只有逐工具的调用轨迹才暴露了这一点。保真度——即提取的值是否与源文本一致——是智能体文档提取的标准度量，而它把那次运行判为成功。因此，我们在一个包含37项人工整理声明的智能体基准（其中25项覆盖三个组件，另有12项覆盖第四个组件）上记录每一次工具调用。基于这份调度记录，我们构建了两个监测工具：一个基于规则的失败归因分类器，以及一个静默失败检测器——它的两条规则只检查调用了哪些工具，从不检查提取出的值。该检测器在来自三个模型系列的207次干净且通过保真度检查的提取上不触发任何告警，并成功找回了全部50个被植入的故障。

    arXiv:2608.28439v1 Announce Type: new  Abstract: One model passed our fidelity check without ever opening the datasheet. We found it while qualifying models for an internal extraction service: a structured-output constraint had silently disabled tool use, and the model answered anyway, with fabricated source text. Only the per-tool trace exposed it. Fidelity -- whether an extracted value matches the source -- is the standard measure for agentic document extraction, and it scores that run a success. We therefore log every tool call in an agentic benchmark of 25 hand-curated claims over three components, with 12 more on a fourth, 37 in all. From that dispatch record we build two instruments: a rule-based failure-attribution classifier, and a silent-failure detector whose two rules check only which tools were called, never the extracted value. The detector raises no flag on 207 clean fidelity-passing extractions across three model families, and recovers all 50 planted faults that withhold
    
[^12]: 这些模块值得它们的成本吗？上下文学习Text-to-SQL的范式级精度-成本分析

    Are These Modules Worth Their Cost? A Paradigm-Level Accuracy-Cost Analysis of In-context Learning Text-to-SQL

    [https://arxiv.org/abs/2608.28432](https://arxiv.org/abs/2608.28432)

    该论文在统一受控环境下首次对ICL Text-to-SQL流水线中五个模块的17种范式级配置进行了精度-成本边际贡献分析，发现执行反馈精炼是唯一在所有骨干模型上收益均普遍成立的范式。

    

    上下文学习（ICL）Text-to-SQL的最新进展通过在基础生成器周围组装日益精细的流水线，显著提升了公开基准测试上的执行准确率，然而现有研究通常只报告端到端的总体准确率，未能量化各个设计选择对准确率与成本的边际贡献。因此，提供统一的、范式级别的成本-准确率量化仍然是理解和配置现代Text-to-SQL系统的关键挑战。为解决这一问题，我们在单一受控实现下，围绕ICL Text-to-SQL流水线中五个反复出现的模块实例化了17种范式级配置，并将每种范式的边际贡献及产生成本归因于四个涵盖不同能力水平和推理风格的骨干模型。我们的分析揭示，执行反馈精炼是唯一一种其收益普遍成立的范式。

    arXiv:2608.28432v1 Announce Type: new  Abstract: Recent advances in in-context learning (ICL) text-to-SQL have substantially improved execution accuracy on public benchmarks by assembling increasingly elaborate pipelines around the base generator, yet existing studies typically report aggregate end-to-end accuracy, without quantifying the marginal accuracy-cost contribution of individual design choices. Consequently, providing a unified, paradigm-level cost-accuracy quantification remains a critical challenge for understanding and configuring modern text-to-SQL. To address this, we instantiate 17 paradigm-level configurations across five recurring modules of the ICL text-to-SQL pipeline under a single controlled implementation, and attribute each paradigm's marginal contribution and incurred cost across all four backbones spanning diverse capability levels and reasoning styles. Our analysis reveals that execution-feedback refinement is the only paradigm whose benefit holds universally 
    
[^13]: 一个用于可解释多维度作文评分的结构化反馈引导统一框架

    A Unified Framework to Elicit Structured Feedback for Interpretable Multi-Trait Essay Scoring

    [https://arxiv.org/abs/2608.28407](https://arxiv.org/abs/2608.28407)

    提出统一自回归框架HiFTS，先基于评分标准生成分层思维链反馈再预测多维度与整体作文分数，并通过GRPO强化学习优化反馈质量与评分一致性，实现可解释的多维度自动作文评分。

    

    多维度自动作文评分（AES）需要基于评分标准对相互关联的多个维度进行推理，而非孤立的分数预测。现有的反馈增强方法通常将反馈与评分解耦，或对各个维度进行独立评估，从而削弱了分数与反馈之间的一致性以及与评分标准的对齐程度。我们提出HiFTS，这是一个统一的自回归框架，在预测维度级分数和整体分数之前先生成分层思维链（CoT）反馈。HiFTS从教师大语言模型中蒸馏出基于评分标准的分层CoT反馈，并训练学生模型联合生成反馈与分数。HiFTS进一步应用组相对策略优化（GRPO），采用平衡分数一致性、校准度、反馈质量和结构有效性的复合奖励。在推理阶段，一个轻量级的全局先验提供整体性指导，以减少长篇推理过程中的漂移。我们还引入了CFMS-34，一个包含951篇作文的中文多维度AES数据集。

    arXiv:2608.28407v1 Announce Type: new  Abstract: Multi-trait Automated Essay Scoring (AES) requires rubric-grounded reasoning across interdependent traits, rather than isolated score prediction. Existing feedback-enhanced methods often decouple feedback from scoring or assess traits independently, weakening score--feedback consistency and rubric alignment. We propose HiFTS, a unified autoregressive framework that generates hierarchical CoT feedback before predicting trait-level and holistic scores. HiFTS distills rubric-grounded hierarchical CoT feedback from a teacher LLM and trains student models to jointly generate feedback and scores. HiFTS further applies Group Relative Policy Optimization with a composite reward balancing score agreement, calibration, feedback quality, and structural validity. At inference, a lightweight global prior provides holistic guidance to reduce drift during long-form reasoning. We also introduce CFMS-34, a Chinese multi-trait AES dataset with 951 essays 
    
[^14]: CultureConverse：一个面向东亚与东南亚文化情境化辅助的多语言多轮对话模拟评测框架

    CultureConverse: A Multilingual Multi-turn Simulation Harness for Culturally Grounded Assistance in East and Southeast Asia

    [https://arxiv.org/abs/2608.28405](https://arxiv.org/abs/2608.28405)

    该论文提出CultureConverse，一个覆盖东亚与东南亚10个地区、58个子群体身份和7个领域的多语言多轮文化情境化助手对话模拟与评测框架，并构建了包含14,610个基准评测回合和274,295个oracle引导对话的数据集，弥补了传统单选题式文化评测无法反映多轮实际辅助场景的不足。

    

    当前针对大语言模型（LLM）的文化评测往往将文化简化为通过多选题进行的单轮事实性问答，无法捕捉一个常见的使用场景：用户在文化情境化的场景中通过多轮对话寻求实际帮助。我们提出了CultureConverse，这是一个可扩展的、多语言的文化情境化助手对话模拟与评测框架，覆盖10个东亚和东南亚地区、58个子群体身份以及7个领域。每一次被模拟和评测的对话回合都会产生一个带评分的交互，其中助手为用户提供帮助，并从部分信息中推断文化约束。由此构建的CultureConverse-DS数据集包含14,610个基准（评测）回合和274,295个由oracle引导（gold模式）的对话。在对18个模型的基准评测中，GPT-5 mini获得了最高的辅助质量。人工标注实验表明，我们的评测框架可以作为一种充分的替代指标……

    arXiv:2608.28405v1 Announce Type: new  Abstract: Current cultural evaluations for large language models (LLMs) often reduce culture to single-turn factual recall via MCQs, failing to capture a common use case: users seeking practical help over multiple turns in culturally grounded scenarios. We introduce CultureConverse, a scalable, multilingual simulation and evaluation harness for culturally grounded assistant dialogue that covers 10 East and Southeast Asian regions, 58 subgroup identities, and 7 domains. Each simulated and evaluated episode produces a scored interaction where the assistant assists the user and infers cultural constraints from partial information. The resulting CultureConverse-DS dataset contains 14,610 benchmark (evaluation) episodes and 274,295 oracle-guided (gold-mode) dialogues. In our benchmark evaluation of 18 models, GPT-5 mini achieves the highest assistance quality. Human annotation experiments suggest that our evaluation framework is a sufficient proxy for 
    
[^15]: BEACON：面向网络威胁情报的行为锚定跨源知识图谱构建

    BEACON: Behavior-Anchored Cross-Source Knowledge Graph Construction for Cyber Threat Intelligence

    [https://arxiv.org/abs/2608.28394](https://arxiv.org/abs/2608.28394)

    该论文提出BEACON，创新性地以映射到MITRE ATT&CK的攻击行为作为锚点，将不同来源报告中的实体和失陷指标统一到同一规范空间，实现跨源网络威胁情报知识图谱的自动构建。

    

    网络威胁情报（CTI）是现代网络防御的基础，然而其中大部分内容以非结构化报告的形式存在，其数量和异构性远超人工分析的能力，这促使研究者探索从CTI报告中自动构建知识图谱的方法。然而，现有方法主要在单一报告内提取部分信息，跨源场景尚未被探索——在该场景下，同一威胁可能被赋予互不相关的名称。我们的关键洞察是：攻击行为一旦映射到MITRE ATT&CK（一个标准化的攻击技术目录），就可以锚定报告的其余部分。攻击行为是报告所描述的对抗性动作，而上下文实体（如威胁行为者、攻击活动和受影响的产品）以及失陷指标（IoCs，如IP地址）则是这些行为的参与者和痕迹。将它们附加到这些锚点上，可以把每个单报告图谱置于同一个规范空间中。我们在B（原文在此截断）

    arXiv:2608.28394v1 Announce Type: cross  Abstract: Cyber threat intelligence (CTI) is foundational to modern cyber defense, yet much of it resides in unstructured reports whose volume and heterogeneity far exceed manual analysis, motivating research on automatically constructing knowledge graphs from CTI reports. However, existing approaches mainly extract partial information within a single report, leaving the cross-source setting unexplored, where the same threat is given unrelated names. Our key insight is that attack behaviors, once mapped to MITRE ATT&CK (a standardized catalog of attack techniques), can anchor the rest of a report. Attack behaviors are the adversarial actions a report describes, while contextual entities (e.g., threat actors, campaigns, and affected products) and Indicators of Compromise (IoCs; e.g., IP addresses) are their participants and traces. Attaching them to these anchors places every per-report graph in one canonical space.   We realize this insight in B
    
[^16]: CamoDocs：一种利用伪装文档针对检索增强语言模型的投毒攻击

    CamoDocs: A Poisoning Attack Against Retrieval-Augmented Language Models Using Camouflaged Documents

    [https://arxiv.org/abs/2608.28389](https://arxiv.org/abs/2608.28389)

    CamoDocs通过将对抗性文档伪装在良性内容中并利用分散token技术，实现了无需直接包含查询即可绕过多种防御的高效RAG投毒攻击，在多个模型和基准上保持高攻击成功率。

    

    检索增强生成（RAG）通过外部文档来增强大语言模型的能力，但公开或用户可编辑的数据来源使RAG系统面临数据投毒风险：攻击者可以注入恶意文档，将模型输出引导至特定目标答案。现有的投毒攻击通常依赖于“查询包含”策略，即将目标查询插入被投毒的文档中以提高被检索到的概率；然而，这种方式会在词汇和嵌入空间中留下痕迹，使其容易被过滤检测。我们提出了CamoDocs，一种通过将对抗性文档伪装在良性内容之中来避免直接包含查询的投毒攻击方法。CamoDocs将合成的良性与对抗性草稿进行分块，用分散token替换良性块中选定的token以分散被投毒文档的嵌入表示，并应用连贯性过滤来限制文本可读性的下降。在七种RAG防御方法、三个开源权重大语言模型和三个基准测试上的实验表明，CamoDocs在避免查询包含的同时实现了较高的平均攻击成功率（ASR）。

    arXiv:2608.28389v1 Announce Type: cross  Abstract: Retrieval-augmented generation (RAG) augments LLMs with external documents, but public or user-editable sources expose RAG systems to data poisoning: attackers can inject malicious documents to steer outputs toward targeted answers. Existing poisoning attacks often rely on query inclusion, inserting the target query into poisoned documents to improve retrieval; however, this creates lexical and embedding-space artifacts that make them easy to filter. We propose CamoDocs, a poisoning attack that avoids direct query inclusion by camouflaging adversarial documents among benign content. CamoDocs chunks synthesized benign and adversarial drafts, replaces selected tokens in benign chunks with dispersion tokens that spread poisoned-document embeddings, and applies coherence filtering to limit readability degradation. Across seven RAG defenses, three open-weight LLMs, and three benchmarks, CamoDocs achieves strong average ASR while avoiding qu
    
[^17]: 语义头专门化引导面向多模态大语言模型的混合ViT注意力

    Semantic Head Specialization Guides Hybrid ViT Attention for Multimodal LLMs

    [https://arxiv.org/abs/2608.28383](https://arxiv.org/abs/2608.28383)

    本研究发现了ViT注意力头会分化为物体与背景专门化角色的“语义头专门化”（SHS）现象，并提出SHS-Index加以量化，据此设计了以6.5倍更少计算量在22个图像视频任务上媲美全注意力的混合注意力架构Ariadne Attention。

    

    混合注意力主导着前沿大语言模型，然而多模态大语言模型中的视觉Transformer（ViT）缺乏令人满意的混合设计，且对于为何某些注意力模式效果更好尚无共识。为填补这一空白，我们研究了ViT的注意力头，发现它们会分化为面向物体和面向背景的专门化角色，这一模式在全注意力下最为显著；我们将其称为语义头专门化（SHS）。我们提出SHS-Index来量化这种专门化程度，证明它可以区分全注意力与分块窗口ViT，并发现它与下游基准测试性能高度相关。随后，我们识别出塑造SHS的三个结构性因素——窗口交互、令牌序列化和局部softmax分配——并将其作为混合注意力的设计原则。在这些因素的指导下，我们设计了Ariadne Attention，一种在22个图像和视频任务上以6.5倍更少的注意力计算量匹配全注意力性能的混合设计。

    arXiv:2608.28383v1 Announce Type: cross  Abstract: Hybrid attention dominates frontier LLMs, yet Vision Transformers (ViTs) in multimodal LLMs lack a satisfactory hybrid design, with no consensus on why certain attention patterns work better. To fill this gap, we study ViT attention heads and find they differentiate into object- and background-specialist roles, a pattern most pronounced under full attention; we call this Semantic Head Specialization (SHS). We propose SHS-Index to quantify this specialization, show that it distinguishes full-attention from chunk-window ViTs, and find that it strongly tracks downstream benchmark performance. We then identify three structural factors that shape SHS---window interaction, token serialization, and local softmax allocation---and use them as design principles for hybrid attention. Guided by these factors, we design Ariadne Attention, a hybrid that matches full attention on 22 image and video tasks at 6.5x less attention compute. Our findings e
    
[^18]: 《大语言模型中语言置信度与内部置信度的分歧》

    When Linguistic and Internal Confidence Diverge in Large Language Models

    [https://arxiv.org/abs/2608.28382](https://arxiv.org/abs/2608.28382)

    该研究通过跨8个分类任务、2个生成任务和30个模型的大规模实验，揭示了大语言模型口头表达的语言置信度与其内部置信度经常不一致，且指令微调模型置信度更高但校准更差、态度提示会夸大置信度而无法提升准确性。

    

    用户经常要求大语言模型（LLM）报告其置信程度，但这种语言置信度是否能反映模型的内部置信度尚不清楚。我们在8个分类任务、2个生成任务以及来自三个系列的30个模型上研究了这一问题。对于分类任务，我们从三个维度（关联性、幅值一致性和校准度）将语言置信度与基于logits的置信度进行比较。对于生成任务，我们测试语言置信度是否与基于语义熵的不确定性相符。结果显示这些维度经常出现分歧：实例层面的关联性平均较弱，尽管在较简单的题目和更强的基座模型上有所改善；经过指令微调的模型通常报告更高的置信度，有时表现出更高的关联性，但它们的置信度差距也更大，校准度更差；提示词设计主要改变的是报告置信度的分布；态度线索会夸大置信度却无法提升……（摘要原文在此处截断）

    arXiv:2608.28382v1 Announce Type: new  Abstract: Users often ask large language models (LLMs) to report how confident they are, but it is unclear whether such linguistic confidence tracks the model's internal confidence. We study this question across 8 classification tasks, 2 generation tasks and 30 models from three families. For classification, we compare linguistic confidence with logits-based confidence along three axes: association, magnitude agreement and calibration. For generation, we test whether linguistic confidence tracks semantic-entropy-based uncertainty. The axes frequently diverge. Instance-level association is weak on average, although it improves on easier items and for stronger base models. Instruction-tuned models often report higher confidence and sometimes show higher association, but they also have larger confidence gaps and worse calibration. Prompt design mostly changes the distribution of reported confidence. Attitude cues inflate confidence without improving 
    
[^19]: PersonaForge：面向智能体系统的逼真多轮用户模拟

    PersonaForge: Realistic Multi-Turn User Simulation for Agentic Systems

    [https://arxiv.org/abs/2608.28378](https://arxiv.org/abs/2608.28378)

    PersonaForge是一个通过四维人格空间、SOUL驱动行为控制和反向深度构建来合成逼真多轮用户-智能体交互的用户模拟框架，并据此构建了6.3K条训练数据和138任务的人工标注基准PersonaForge-Bench，弥合了真实多轮用户交互与智能体系统训练评估之间的巨大差距。

    

    大语言模型日益被用作智能体工作流的执行器，然而现有的训练数据和基准测试大多假设信息完整、单轮的查询。我们对16K真实会话的分析表明，75.9%的交互是多轮的，这揭示了用户与智能体的实际交互方式与此类系统的训练和评估方式之间存在巨大差距。我们提出了PersonaForge，一个用于合成逼真多轮用户-智能体交互的用户模拟框架。PersonaForge结合了四维人格空间、基于真实用户统计校准的SOUL驱动行为控制，以及基于真实种子查询的反向深度构建方法。利用PersonaForge，我们构建了包含6.3K条记录的训练数据集，以及PersonaForge-Bench——一个包含138个任务、覆盖20多个专业领域、采用四维评分的人工标注基准。在Qwen3.5-27B上的实验表明，PersonaF

    arXiv:2608.28378v1 Announce Type: new  Abstract: Large language models are increasingly used as agentic workflow executors, yet existing training data and benchmarks largely assume informationally complete, single-turn queries. Our analysis of 16K real-world sessions shows that 75.9% of interactions are multi-turn, revealing a substantial gap between how users interact with agents and how such systems are trained and evaluated. We introduce \textbf{PersonaForge}, a user simulation framework for synthesizing realistic multi-turn user--agent interactions. PersonaForge combines a four-dimensional persona space, SOUL-driven behavioral control calibrated to real-user statistics, and Reverse Deep Construction grounded in authentic seed queries. Using PersonaForge, we construct a 6.3K-record training dataset and \textbf{PersonaForge-Bench}, a manually annotated 138-task benchmark spanning over 20 professional domains with four-dimensional scoring. Experiments on Qwen3.5-27B show that PersonaF
    
[^20]: BanglaMed-QA：面向孟加拉语医疗健康支持的问答系统

    BanglaMed-QA: A Question Answering System for Healthcare Support in Bangla

    [https://arxiv.org/abs/2608.28329](https://arxiv.org/abs/2608.28329)

    本文提出了BanglaMed-QA——首个专为孟加拉语医疗领域设计的问答系统，通过构建包含506种疾病、4,493个问答对的结构化知识库，结合SVM问题分类、领域专用词典与同义词集以及多种相似度度量与投票机制，为低资源语言的医疗健康信息支持提供了有效解决方案。

    

    医疗问答（QA）系统已成为提供可靠健康信息的重要工具。但由于数据集有限以及缺乏针对孟加拉语等低资源语言定制的系统，这些语言在医疗问答领域仍鲜有探索。为解决这一问题，我们推出了BanglaMed-QA，一个专为孟加拉语医疗领域设计的稳健问答系统。该流程首先构建了一个结构化的医疗知识库，其中包含506种疾病下9个类别的4,493个问答对。为提升语义理解能力，我们提出了领域专用的词根词典和同义词集，并采用词性标注技术进行指代消解。我们采用了监督机器学习模型，其中支持向量机（SVM）被发现是对问题进行分类的最佳模型。我们应用了多种相似度度量方法，包括余弦相似度、Jaccard、BM25和Levenshtein距离，并结合软投票和硬投票方法进行查询匹配。该问答系统的性能表现……（原文摘要在此处截断）

    arXiv:2608.28329v1 Announce Type: new  Abstract: Medical question answering (QA) systems have become crucial tools for providing reliable health information. But they remain very unexplored for low-resource languages like Bangla due to limited datasets and systems tailored to these languages. To address this, we introduce BanglaMed-QA, a robust QA system specifically designed for the Bangla medical domain. The process begins with building a structured medical knowledge base that includes 4,493 QA pairs in 9 categories under 506 diseases. To improve semantic comprehension, domain-specific root word dictionaries and synonym sets are proposed, in addition to part-of-speech tagging for anaphora resolution. We adopt supervised machine learning models in which SVM is found to be the best model to categorize questions. Multiple similarity metrics, including cosine, Jaccard, BM25, and Levenshtein, are applied with soft and hard voting methods for query matching. The performance of the QA syste
    
[^21]: 将大语言模型分层防御视为集成：访问层级、推理成本与防御层间失败相关性的实测

    Layered LLM Defenses as an Ensemble: Access Tiers, Inference Cost, and the Measured Failure Correlation Between Defense Layers

    [https://arxiv.org/abs/2608.28327](https://arxiv.org/abs/2608.28327)

    本文提出对抗者访问层级模型（AATM）和推理成本分类方法，首次实测了大语言模型多层防御之间的失败相关性，证明防御堆栈只有在各层在不同输入上失败时才能真正产生叠加防护效果。

    

    从业者通过堆叠多层防御来保护大语言模型（LLM），并默认各层防御会相互叠加增强。防御堆栈本质上是一个集成，而集成只有在满足一个条件下才会产生叠加效果——这一条件在LLM安全文献中虽被推荐但从未被实测：各成员必须在不同的输入上失败。本文提出两种工具使这一点变得可度量。对抗者访问层级模型（AATM）根据攻击者所掌握的访问权限对其进行分级，范围从仅系统访问（A0）到可影响训练数据（A4）。成本模型将防御按五类推理时开销进行分类；由于其中两类需要训练权重或读取激活值，它们像AATM对攻击者分级那样对防御者进行分级。由此我们推导出堆栈的行为规律，以及防御者所关心的各项指标如何分化：覆盖率在层级内趋于饱和，成本随类别上升，误拒绝以并集方式累积，而残余攻击成功率只有在独立假设成立时才呈乘性下降。我们对这种独立性进行了实测。

    arXiv:2608.28327v1 Announce Type: cross  Abstract: Practitioners defend large language models (LLMs) by stacking defenses, assuming the layers compound. A stack is an ensemble, and ensembles compound only under a condition the LLM security literature recommends but never measures: the members must fail on different inputs.   Two instruments make that measurable. The Adversary Access-Tier Model (AATM) grades an adversary by the access it holds, from system-only (A0) to influence over training data (A4). A cost model sorts defenses into five classes of inference-time overhead; because two classes require training weights or reading activations, they tier the defender as AATM tiers the adversary. From these we derive how a stack behaves, and the quantities a defender cares about diverge: coverage saturates within a tier, cost rises by class, false refusals accumulate as a union, and residual attack success falls multiplicatively only under independence.   We measure that independence. Run
    
[^22]: AIM：锚定身份特征后进行匹配，实现多模态大语言模型的遗忘

    AIM: Anchor Identity Features, Then Match for Multimodal Large Language Model Unlearning

    [https://arxiv.org/abs/2608.28312](https://arxiv.org/abs/2608.28312)

    提出AIM两阶段遗忘方法，利用身份类与感知类问题在隐藏状态中组织方式的差异，在无法访问保留图像的情况下实现多模态大语言模型的身份信息遗忘，同时保留其视觉感知能力。

    

    多模态大语言模型（MLLM）可能会记住其微调数据中有关特定人物的隐私信息，当个人请求删除其数据时会带来隐私风险。现有的MLLM遗忘方法通常假设在删除过程中可以访问保留图像或真实答案，这在许多实际场景中是不现实的。我们研究了在删除时保留图像不可用情况下的身份遗忘问题。我们的分析表明，身份类问题和视觉感知类问题在微调后的隐藏状态中占据不同的区域，且组织方式不同：身份类问题按人物聚类，而感知类问题按问题类型聚类。这表明可以在不擦除一般视觉感知能力的前提下抑制身份知识。基于这一观察，我们提出了AIM，一种两阶段方法：首先使用通用视觉提示锚定身份遗忘目标，然后将视觉编码器与该目标进行匹配……

    arXiv:2608.28312v1 Announce Type: cross  Abstract: Multimodal large language models (MLLMs) can memorize identity-specific facts about people in their fine-tuning data, creating privacy risks when a person requests deletion. Existing MLLM unlearning methods often assume access to retain images or ground-truth answers during deletion, which is unrealistic in many practical scenarios. We study identity unlearning when retain images are unavailable at deletion time. Our analysis shows that identity and visual-perception questions occupy distinct regions in fine-tuned hidden states and are organized differently: identity questions cluster by person, whereas perception questions cluster by question type. This suggests that identity knowledge can be suppressed without erasing general visual perception. Building on this observation, we propose AIM, a two-stage method that anchors an identity-forgetting target with a universal visual prompt and then matches the vision encoder to that target un
    
[^23]: VISTA：基于验证器信息的学生到教师自适应的在策略自蒸馏

    VISTA: Verifier-Informed Student-to-Teacher Adaptation for On-Policy Self-Distillation

    [https://arxiv.org/abs/2608.28306](https://arxiv.org/abs/2608.28306)

    提出VISTA方法，在保留标准在策略自蒸馏学生更新的同时，利用结果验证的rollout使特权教师向学生分布自适应，解决了教师分布与学生有效推理不匹配时单向监督误导学生的问题。

    

    在策略自蒸馏（OPSD）通过训练一个仅见问题的学生模型，在其自身生成的 rollout 上进行学习，并由一个同时能看见参考答案的特权教师模型提供密集的词元级监督，从而提升推理能力。然而，标准 OPSD 将教师分布视为学生 rollout 上的固定目标，且只更新学生模型——尽管特权条件化并不能保证教师总是为仅见问题的推理提供最合适的目标。因此，当教师分布与学生的有效推理不一致时，这种单向监督可能会误导学生。为此，我们提出了基于验证器信息的学生到教师自适应方法（VISTA），该方法在保留标准 OPSD 学生更新的同时，利用经结果验证的 rollout 使教师分布向学生分布自适应。在每个经验证的 rollout 内，VISTA 进一步将这种自适应限制在 top-k 位置上……

    arXiv:2608.28306v1 Announce Type: cross  Abstract: On-policy self-distillation (OPSD) improves reasoning by training a problem-only student on its own rollouts using dense token-level supervision from a privileged teacher that also sees a reference solution. However, standard OPSD treats the teacher distribution as a fixed target along the student's rollout and updates only the student %, although -- even though privileged conditioning does not guarantee that the teacher always provides the most appropriate target for problem-only reasoning. This one-way supervision can therefore misdirect the student when the teacher distribution is misaligned with valid student reasoning. We therefore introduce Verifier-Informed Student-to-Teacher Adaptation (VISTA), which preserves the standard OPSD student update while using outcome-verified rollouts to adapt the teacher toward the student distribution. Within each verified rollout, VISTA further restricts this adaptation to the top-$k$ positions w
    
[^24]: KV缓存驱逐的概率性解释

    A Probabilistic Interpretation of KV Cache Eviction

    [https://arxiv.org/abs/2608.28293](https://arxiv.org/abs/2608.28293)

    该论文首次从概率推理的角度对KV缓存驱逐问题进行形式化，证明其计算上的困难性，并将其归结为可通过采样近似的期望估计问题，从而为设计驱逐策略以及解码时校正被驱逐条目提供了理论框架。

    

    KV（缓存）驱逐的前提和承诺很简单：通过从KV缓存中驱逐部分条目可以获得更高的吞吐量，而对质量的影响可以忽略不计。这一结论在许多现有方法中得到了实证验证，尽管这些方法大多依赖创造性的启发式规则来选择要丢弃的条目。尽管近期已有诸多进展，KV驱逐问题在文献中一直缺乏正式的定义。本文旨在通过概率推理的视角对这一问题进行严格的形式化，并揭示从这一视角能够获得哪些启示。具体而言，我们（1）对KV驱逐问题进行了形式化，并遗憾地证明该问题在计算上是困难的；（2）表明通过概率化的框架，KV驱逐可以归结为期望估计问题，而该问题可以通过采样来近似求解；（3）表明借助这种概率解释，在解码过程中对被驱逐条目进行校正——一个此前被忽视的问题——变得……（原文摘要在此处截断）

    arXiv:2608.28293v1 Announce Type: new  Abstract: The premise and promise of KV (cache) eviction is simple: higher throughput can be achieved by evicting some entries from the KV cache, at a negligible cost to quality. This holds empirically for many existing methods, though most rely on creative heuristics for selecting which entries to drop. Despite recent advances, the problem of KV eviction has remained informal in the literature. This paper aims to properly formalize this problem through the lens of probabilistic reasoning and reveal what can be learned from this perspective. Concretely, we (1) formalize the problem of KV eviction and, unfortunately, prove that it is computationally hard, (2) show that by framing it probabilistically, KV eviction reduces to the problem of expectation estimation, which can be approximated through sampling, (3) show that through this probabilistic interpretation, correcting for evicted entries during decoding---a previously ignored problem---becomes 
    
[^25]: 面向立场感知论点检索的嵌入模型

    Embedding Models for Stance-Aware Argument Retrieval

    [https://arxiv.org/abs/2608.28283](https://arxiv.org/abs/2608.28283)

    本研究探索了稠密嵌入模型在立场感知论点检索中的应用，发现现有模型偏向主题相关性而忽视立场信息，而通过对比训练纠偏又会导致模型过度关注极性关键词的过度纠正问题。

    

    在计算论辩学中，获取明确支持或攻击给定主张的论点是下游推理任务的关键前提。当需要使用语义搜索方法检索这些支持和攻击论点时，需要对它们与相关主张的主题相关性以及它们对该主张的（正面或负面）立场的正确性进行双重评估。在本文中，我们探索了驱动现代检索流水线的稠密嵌入模型如何作为融合这种双重评估的语义搜索的基础。我们通过实验表明，现有模型在非对称推理方面存在困难，表现出对主题重叠的强烈偏好，而忽略了指令性立场。我们还表明，通过对比训练纠正这种偏差会触发一种新的失败模式，即模型过度纠正，过度关注极性关键词（例如“支持”或“反驳”）……

    arXiv:2608.28283v1 Announce Type: new  Abstract: In computational argumentation, obtaining arguments that explicitly support or attack given claims is a critical precursor to downstream reasoning tasks. When these supporting and attacking arguments are to be retrieved using semantic search methods, they need to be assessed for topic-relevance to the claims of interest as well as for correctness of their (positive or negative) stance towards the claims. In this paper we explore how dense embedding models (hereafter, models), powering modern retrieval pipelines, can serve as the basis of semantic search incorporating this dual assessment. We show experimentally that existing models struggle with asymmetric reasoning, exhibiting a strong bias toward topical overlap while ignoring instructional stance. We also show that correcting this bias via contrastive training triggers a new failure mode where models over-correct, over-fixating on polarity keywords (e.g., "supports" or "refutes") at t
    
[^26]: Synth-JDoc：合成具有多样化版式和嵌入图像的日语文档图像OCR数据集

    Synth-JDoc: Synthesizing a Japanese Document Image Dataset for OCR with Diverse Layouts and Embedded Images

    [https://arxiv.org/abs/2608.28248](https://arxiv.org/abs/2608.28248)

    该论文提出Synth-JDoc，一个通过合成方式构建、具有多样化版式和嵌入图像的日语文档图像OCR数据集，旨在提升大型视觉语言模型对竖排日语文本的阅读能力。

    

    大型视觉语言模型（LVLMs）读取文档图像中文本的能力至关重要，因为它可以支持文档视觉问答等多种应用。为了提升LVLMs的文本阅读能力，高质量的OCR数据集必不可少。这一需求对于日语文档尤为关键，因为日语文档通常同时包含竖排文本和横排文本。当前的LVLMs在竖排日语文本上的表现明显低于横排文本，因此需要专门的OCR数据集来弥补这一差距。然而，人工构建OCR数据集成本高昂且难以规模化；而通过OCR模型从现有文档图像中提取文本来构建数据集，则会带来文本识别错误以及需要预先获取文档图像等挑战。为了解决这些问题，我们通过合成方式构建OCR数据集……

    arXiv:2608.28248v1 Announce Type: cross  Abstract: The ability of Large Vision Language Models (LVLMs) to read text within document images is crucial, as it enables various applications such as Document Visual Question Answering. To enhance the text-reading capabilities of LVLMs, high-quality OCR datasets are essential. This need is particularly critical for Japanese documents, which often feature vertically written text alongside horizontally written text. Current LVLMs demonstrate considerably lower performance on vertically written Japanese text than on horizontally written text, necessitating specialized OCR datasets to bridge this gap. However, manually constructing OCR datasets is expensive and difficult to scale. Alternatively, constructing datasets by extracting text from existing document images using OCR models introduces challenges, such as text recognition errors and the prerequisite of sourcing document images.   To address these issues, we construct an OCR dataset by synt
    
[^27]: 恪守边界：基于距离引导的解码方法，保证输出符合上下文无关文法

    Stay Within Your Bounds: Distance-Guided Decoding for Guaranteed Context-Free Grammar Compliance

    [https://arxiv.org/abs/2608.28229](https://arxiv.org/abs/2608.28229)

    提出一种基于下推自动机的距离引导解码框架，通过离线计算可达性标签与到接受状态的距离上界、在线进行视野感知剪枝与束搜索，保证大模型生成结果百分之百符合目标上下文无关文法，同时提升补全质量。

    

    文法约束解码可帮助大型语言模型生成语法有效的结构化输出，例如代码、JSON和SQL。针对上下文无关文法，许多实用解码器强制执行局部前缀可行性：每个token必须保证当前前缀能够扩展到某个有效的完整结果。然而，在分词器与文法不匹配以及token预算有限的情况下，可行前缀仍可能无法到达接受状态。我们提出了一种基于下推自动机的、面向上下文无关文法的前瞻引导解码框架。离线阶段，我们计算带有可达性标签以及到接受状态距离上界的有界下推摘要；在线阶段，这些估计值引导具有视野感知的剪枝与束搜索。由此得到的解码器在语法上是可靠的：每个输出都会被目标文法接受。在JSON、SQL和线性时序逻辑（LTL）上的实验表明，与现有基线相比，该方法既保持了一致的语法有效性，又提升了补全质量。

    arXiv:2608.28229v1 Announce Type: cross  Abstract: Grammar-constrained decoding helps large language models produce syntactically valid structured outputs, such as code, JSON, and SQL. For context-free grammars, many practical decoders enforce local prefix feasibility: each token must keep the current prefix extendable to some valid completion. Yet, under tokenizer-grammar mismatch and finite token budgets, feasible prefixes may still fail to reach acceptance. We propose a lookahead-guided decoding framework for context-free grammars based on pushdown automata. Offline, we compute bounded pushdown summaries with reachability labels and upper-bound distances to acceptance. Online, these estimates guide horizon-aware pruning and beam search. The resulting decoder is syntactically sound: every output is accepted by the target grammar. Experiments on JSON, SQL, and Linear Temporal Logic (LTL) show both consistent syntactic validity and improved completion quality over existing baselines.
    
[^28]: 以人类行为分布为基准测试大语言模型智能体社会

    Benchmarking large language model agent societies against human behavioural distributions

    [https://arxiv.org/abs/2608.28182](https://arxiv.org/abs/2608.28182)

    提出开放基准SILICA，通过五个带人类行为锚点的环境及其扰动和反记忆变体检验大语言模型智能体社会，发现模型仅在首轮公共品贡献等起点上与人类行为一致，而无法复现最终状态等结果。

    

    大语言模型智能体群体正日益被用作实验社会。然而每一种此类结果都笼罩着三点疑虑：智能体的行为是否像它们所代表的人类；当实验装置改变而规则保持不变时，研究发现是否仍然成立；以及表面上显现的社会动态究竟是否源于真实的交互，还是仅仅复现了模型曾经读过的实验。本文介绍了SILICA，一个能够检验上述三点的开放工具。该工具包含五个带有已发表人类行为锚点的环境，每个环境都配有重新呈现相同规则的扰动版本，以及收益指向偏离记忆结果的变体。十二个开放权重模型在单张消费级显卡上完成了该测试。结果显示，模型与人类数据的一致性仅局限于起点：十一个模型中有八个的首轮公共品贡献落在等效边界之内，但没有任何模型能匹配最终状态的贡献或其他……

    arXiv:2608.28182v1 Announce Type: cross  Abstract: Populations of large language model agents are increasingly used as experimental societies. Three doubts shadow every such result: whether the agents behave like the humans they stand in for, whether a finding survives changes to the apparatus that leave the rules untouched, and whether apparent social dynamics are interaction at all rather than the reproduction of experiments the models have read. This article introduces SILICA, an open instrument that tests all three. Five environments carry published human anchors, each paired with perturbations that re-render the same rules and with variants whose payoffs point away from the memorised result. Twelve open-weight models were run through it on a single consumer graphics card. Agreement with human data is confined to starting points: first-round public-goods contributions fall inside the equivalence margin for eight of eleven models, while no model matches end-state contributions or th
    
[^29]: 基于语言模型的古代文献文本修复

    Text Restoration of Ancient Documents with Language Models

    [https://arxiv.org/abs/2608.28170](https://arxiv.org/abs/2608.28170)

    本研究首次系统探索利用不同架构的语言模型修复受损古代手稿中的文本缺失，提出多种解码策略以解决缺损边界与分词方案不一致的问题，发现该技术虽无法完全自动化，但可作为古文书学家的有效辅助工具。

    

    目的 - 本研究探讨了使用语言模型修复受损古代手稿中因物理性缺损而导致的文本缺失的可行性。方法 - 该研究提出了不同的场景以模拟真实世界条件，并根据各模型对不同场景的适用性，应用了不同架构的语言模型。此外，我们还提出了几种解码策略，以进一步提升性能，并解决缺损边界与模型分词方案之间的不一致问题。发现 - 结果表明，这些文献的文本修复无法完全自动化，但它可以作为辅助古文书学家工作的有用工具。模型性能因需修复的文献结构部分不同以及缺失文本的字符长度是否可知而有很大差异。独创性 - 这是第一项……（原文在此处截断）

    arXiv:2608.28170v1 Announce Type: new  Abstract: Purpose - This study investigates the feasibility of restoring missing text caused by physical lacunae in damaged ancient manuscripts using language models.   Methodology - The study proposes different scenarios to replicate real-world conditions. Language models of different architectures are applied according to their suitability to each scenario. We also propose several decoding strategies that further enhance performance and address the discrepancy between lacuna boundaries and the models' tokenization schemes.   Findings - The results reveal that text restoration of these documents cannot be fully automated, but it can serve as a useful tool to assist paleographers in their work. Model performance varies greatly depending on which structural part of the document needs to be restored and whether the character length of missing text is available.   Originality - This is the first study and to analyze model performance on formulaic and
    
[^30]: FinExam-10K：检索何时有助于金融推理？

    FinExam-10K: When Retrieval Helps Financial Reasoning?

    [https://arxiv.org/abs/2608.28155](https://arxiv.org/abs/2608.28155)

    该论文提出了迄今最大的覆盖CFA与FRM完整考试体系的英文金融考试基准FinExam-10K（含10,198道专家标注题目及双赛道评估设计），并揭示检索增强方法（Function-RAG和FunctionGraph-RAG）能挽救模型在困难金融推理题目上的失败。

    

    专业的金融考试要求模型结合领域知识、计算和判断能力，然而目前尚无基准测试在统一协议下覆盖CFA和FRM的完整体系。我们推出了FinExam-10K，据我们所知，这是该设置下已报道的最大的英文基准，包含10,198道经专家重新标注的题目，涵盖CFA一至三级和FRM一至二级。我们公开发布5,110道题目，并将5,088道题封闭保存，用于每季度维护的排行榜。为了区分覆盖率与局部可答性，我们报告了包含10,198道题的全覆盖赛道，以及包含7,625道题的上下文完整推理赛道，后者是关于基于所提供记录进行推理能力评估的主要依据。在17个模型的评测中，最佳总体准确率为85.29%。在冻结的困难区间上，最佳成绩在全覆盖赛道上为34.68%，在372道上下文完整题目上为54.57%。所有17个模型共有47道上下文完整题目的共同失败案例。Function-RAG和FunctionGraph-RAG能够挽救……

    arXiv:2608.28155v1 Announce Type: new  Abstract: Professional financial examinations require models to combine domain knowledge, calculation, and judgment, yet no benchmark covers the full CFA and FRM structure under one protocol. We introduce FinExam-10K, to our knowledge the largest reported English benchmark for this setting, with 10,198 expert-reannotated questions spanning CFA Levels I-III and FRM Parts I-II. We release 5,110 questions and sequester 5,088 for a quarterly maintained leaderboard. To separate coverage from local answerability, we report a 10,198-item Full-Coverage Track and a 7,625-item Context-Complete Reasoning Track, which is the primary basis for claims about reasoning from the supplied record. Across 17 models, the best accuracy is 85.29% overall. On the frozen Hard band, the best score is 34.68% on the Full-Coverage Track and 54.57% on the 372 context-complete items. All 17 models share 47 context-complete failures. Function-RAG and FunctionGraph-RAG rescue hun
    
[^31]: 嵌套字节级词表部署廉价但共享昂贵：一项预注册的阴性结果

    Nested Byte-Level Vocabularies Are Cheap to Deploy and Expensive to Share: A Pre-Registered Negative Result

    [https://arxiv.org/abs/2608.28151](https://arxiv.org/abs/2608.28151)

    本文通过预注册实验证明，嵌套字节级词表虽然能通过精确切片实现廉价部署（移除66%部署权重且数值完全一致），但跨规模共享词表会使模型性能超出预设余量地落后于固定词表的专用模型。

    

    字节级BPE分词器是一个有序的合并规则列表，因此仅应用其前缀即可得到一个词表，其词元标识符恰好是完整词表的前几行。这种前缀嵌套使一个语言模型能够在多种词表规模下运行，使用控制词元指示当前激活的规模，并通过切片其嵌入层和输出头以任意训练过的规模进行部署。我们预先注册了五项声明，包括性能余量、随机种子、对照实验和停止规则，并训练了30个模型，分别使用310万和1060万参数的主体，每个模型在2亿词元上训练。切片在数值上是精确的：在76项检查中，切片后的模型逐位复现了受限完整模型的logits，并移除了66%的部署权重而不改变推理延迟。然而，共享模型落后于固定词表的专用模型：在32k词表下每字节比特数落后3.64%（超出1%的预设余量），在8k词表下落后2.96%（超出2%的预设余量）。一项2x2消融实验将控制词元与输出限制分离开来。

    arXiv:2608.28151v1 Announce Type: new  Abstract: A byte-level BPE tokenizer is an ordered list of merge rules, so applying only a prefix yields a vocabulary whose token identifiers are the first rows of the full vocabulary. This prefix nesting allows one language model to operate at several vocabulary sizes, use a control token to indicate the active size, and be deployed at any trained size by slicing its embedding and output head. We pre-registered five claims, including margins, seeds, contrasts, and a stop rule, and trained 30 models with 3.1M- and 10.6M-parameter bodies on 200M tokens each. Slicing is numerically exact: across 76 checks, a sliced model reproduces the restricted full model's logits bit for bit and removes 66% of deployed weights without changing latency. However, the shared model trails a fixed-cap specialist by 3.64% bits per byte at 32k against a 1% margin, and by 2.96% at 8k against a 2% margin. A 2x2 ablation separating the control token from output restriction
    
[^32]: H-Scale：基于Hessian引导的尺度精修方法，用于NVFP4亚字节大语言模型推理

    H-Scale: Hessian-Guided Scale Refinement for NVFP4 Sub-Byte LLM Inference

    [https://arxiv.org/abs/2608.28113](https://arxiv.org/abs/2608.28113)

    提出H-Scale，一种基于Hessian二阶信息的轻量级后处理方法，通过精修NVFP4量化中的逐组缩放因子来更直接地减少层输出扰动，从而提升大语言模型亚字节推理的精度。

    

    NVIDIA Blackwell架构凭借其对超细粒度NVFP4格式的原生支持，为加速大语言模型（LLM）推理开辟了新的机会。NVFP4的微块设计（例如组大小为16）为捕捉局部权重分布和隔离离群值提供了强大的表示灵活性，但同时也引入了一个庞大且高度敏感的逐组缩放因子空间。现有的训练后量化（PTQ）方法主要专注于优化量化权重值，而对尺度选择这一步骤的探索相对不足。为填补这一空白，我们提出了H-Scale，一种用于NVFP4逐组尺度精修的轻量级后处理方法。与最小化普通权重重建误差不同，H-Scale利用从校准激活中导出的对角二阶代理（Hessian引导）来选择硬件有效的组尺度，从而更直接地针对层输出扰动进行优化。该方法被设计为（原文在此处截断）……

    arXiv:2608.28113v1 Announce Type: new  Abstract: The NVIDIA Blackwell architecture, with native support for the ultra-fine-grained NVFP4 format, opens new opportunities for accelerating large language model (LLM) inference. NVFP4's micro-block design, such as a group size of 16, offers strong representational flexibility for capturing local weight distributions and isolating outliers, but it also introduces a large and highly sensitive space of per-group scaling factors. Existing post-training quantization (PTQ) methods primarily focus on refining quantized weight values, leaving this scale-selection step underexplored. To address this gap, we propose \textbf{H-Scale}, a lightweight post-processing method for NVFP4 per-group scale refinement. Instead of minimizing plain weight reconstruction error, H-Scale selects hardware-valid group scales using a diagonal second-order proxy derived from calibration activations, thereby targeting layer output perturbation more directly. It is designe
    
[^33]: 投机探测：以投机解码成本实现LLM监控

    Speculative Probing: LLM Monitoring at Speculative-Decoding Cost

    [https://arxiv.org/abs/2608.28099](https://arxiv.org/abs/2608.28099)

    该论文提出通过在目标序列末尾附加训练好的软提示，将LLM中的投机解码模块重新用作高效且高质量的序列分类器，以投机解码的成本实现实时模型监控。

    

    在语言模型推理过程中进行实时分类，对于安全过滤、行为分析和模型监控具有重要价值，但现有方法迫使人们在准确性与效率之间进行权衡。隐藏状态探测器速度快但能力有限：它们要么不具备上下文感知能力——仅操作单个向量，无法建模跨位置的交互；要么成本非常高昂——需要专门的分类器模型（如Llama Guard、Qwen Guard、LLM-as-judge），或者需要对所有token的隐藏状态进行计算然后再池化结果（MultiMax）。这体现了效率与准确性之间的内在权衡。然而，我们发现近期LLM中的投机解码模块可以被重新用于实现高效且高质量的分类。通过在目标序列末尾附加一个训练好的软提示，我们可以将投机解码模块转变为一个序列分类器。在投机解码的推理时……

    arXiv:2608.28099v1 Announce Type: cross  Abstract: Real-time classification during language model inference is valuable for safety filtering, behavioral analysis, and model monitoring, but current approaches force a trade-off between accuracy and efficiency. Hidden-state probes are fast but limited: they are either not context-aware: operating on a single vector and cannot model interactions across positions; or they are very costly: having dedicated classifier models (Llama Guard, Qwen Guard, LLM-as-judge) or performing computation on hidden states for all tokens and then pooling the results (MultiMax). This shows an intrinsic trade-off between efficiency and accuracy.   However, we find that the speculative-decoding module in recent LLMs can be repurposed for efficient high-quality classification. By appending a trained soft prompt at the end of the target sequence, we can repurpose the speculative-decoding module into a sequence classifier. At inference time in a speculative-decodin
    
[^34]: CNeo-Bench：诊断大语言模型对汉语新词的理解能力

    CNeo-Bench: Diagnosing Large Language Models on Chinese Neologisms

    [https://arxiv.org/abs/2608.28053](https://arxiv.org/abs/2608.28053)

    该论文提出了CNeo-Bench——一个包含4,759个汉语新词的基准及双层评估框架，用于区分大语言模型“描述新词”与“操作其底层语言机制”的能力，发现多数模型释义生成准确率低于40%，且普遍存在能描述却不能还原源形式的“识别-操作”差距。

    

    汉语新词运用了多样且独特的语言机制，例如语音替代（如用"886"表示“拜拜”）和视觉字形拆解，这些机制在其他语言中十分罕见。我们提出了CNeo-Bench，这是一个包含4,759个此类新词及其参考释义的基准数据集，并按照每个表达背后的语言机制将其组织为五大类别和九个子类别。CNeo-Bench还配套了一个双层评估框架，将“模型能否描述一个新词”与“模型能否对其底层语言机制进行操作”区分开来。通过对18个大语言模型的评估，我们发现汉语新词仍然是一个尚未解决的挑战：大多数模型在释义生成任务上的表现低于40%，并且在若干子类别上出现了系统性的“识别-操作”差距——模型能够正确描述新词，但在源形式还原任务中，它们会用语义等价的表达（释义）来替代源形式，而非产出真正的源形式。

    arXiv:2608.28053v1 Announce Type: new  Abstract: Chinese neologisms exploit diverse and unique linguistic mechanisms, such as phonetic substitution (e.g., 886 for ``bye-bye'') and visual character decomposition that are rare in other languages. We introduce CNeo-Bench, a benchmark of 4,759 such neologisms with reference definitions, organized into five top-level categories and nine subcategories by the linguistic mechanism behind each expression. CNeo-Bench is paired with a two-tier evaluation framework that separates whether a model can describe a neologism from whether it can operate on its underlying mechanism. Evaluating 18 LLMs, we find that Chinese neologisms remain an open challenge; most models fall below 40\% on definition generation, and on several subcategories a systematic recognition-manipulation gap emerges: models describe neologisms correctly but, in source-form restoration tasks, substitute a semantic equivalent (paraphrase) for the source form rather than producing th
    
[^35]: SimpCue：基于线索提示的多语言文本简化

    SimpCue: Cue-Based Prompting for Multilingual Text Simplification

    [https://arxiv.org/abs/2608.28042](https://arxiv.org/abs/2608.28042)

    该论文提出在多语言“易读”文本简化中，将自动预测的句子复杂度语言学线索加入提示，能在所有评估指标上带来小幅但一致的提升，而人工黄金线索的效果反而不稳定。

    

    文本简化旨在在保留原文含义的同时，使复杂文本更易于理解。近期的大型语言模型可以通过提示来执行简化任务，但在提示中加入关于句子复杂度的显式语言学信息是否能改善其输出，目前仍不清楚。我们针对加泰罗尼亚语、西班牙语和意大利语的多语言句子级“易读”简化任务研究了这一问题。使用Qwen3-8B模型，我们比较了基线提示、以人工标注的黄金语言学线索增强的提示，以及以自动预测的线索增强的提示。我们采用SARI、BLEU、chrF和BERTScore对输出进行评估，并结合人工定性分析加以补充。预测线索提示在全部四项指标上均取得了最佳总体分数，尽管相对于基线的提升幅度较小。黄金线索提示并未持续优于基线，且结果……

    arXiv:2608.28042v1 Announce Type: new  Abstract: Text simplification aims to make complex texts easier to understand while preserving their original meaning. Recent large language models can perform simplification through prompting, but it remains unclear whether adding explicit linguistic information about sentence complexity to the prompt improves their outputs. We investigate this question for multilingual sentence-level Easy-to-Read simplification in Catalan, Spanish, and Italian. Using Qwen3-8B, we compare a baseline prompt, a gold-cue prompt enriched with gold linguistic cues, and a predicted-cue prompt enriched with automatically predicted cues. We evaluate the outputs using SARI, BLEU, chrF, and BERTScore, and complement this evaluation with a manual qualitative analysis. Predicted-cue prompting obtains the best overall scores across all four metrics, although the gains over the baseline are small. Gold-cue prompting does not consistently improve over the baseline, and results 
    
[^36]: 声音颤抖并不总是闪避：财报电话会议中文本与语音回避检测的基准测试

    A Shaky Voice Is Not Always a Dodge: Benchmarking Textual and Vocal Evasion Detection in Earnings Calls

    [https://arxiv.org/abs/2608.28040](https://arxiv.org/abs/2608.28040)

    该论文提出 DualEvasion 基准，首次将财报电话会议中的回避检测从单一文本维度扩展为文本回避与语音自信度的双维度联合分析，并发现现有最先进多模态模型难以检测语音自信度，原因是它们孤立解读声学线索而非结合说话者个人基线。

    

    现有针对财报电话会议中回避行为检测的方法主要聚焦于文本转录，将回避视为单一维度的现象。我们认为，口语交流中的回避本质上是多维度的：除了高管说什么之外，他们如何说也携带着独立且互补的信息。为了联合研究这些维度，我们提出了 DualEvasion，一个针对财报电话会议问答环节中跨文本与音频的回避检测基准。该基准包含来自 60 场财报电话会议的 505 个经标注的问答对，每个问答对均带有两个独立的标签：文本回避（直接 vs. 回避）以及语音线索（将其操作化为说话者自信度：自信 vs. 不自信）。我们的实验表明，最先进的多模态模型难以检测语音自信度，尤其在不自信的回答上表现不佳。我们的分析表明，这些模型是孤立地解读声学线索，而不是相对于每位说话者自身的基线来解读。

    arXiv:2608.28040v1 Announce Type: new  Abstract: Existing approaches to evasion detection in earnings calls focus on textual transcripts, treating evasion as a single-dimensional phenomenon. We argue that evasion in spoken communication is inherently multidimensional: beyond what executives say, how they say it carries independent and complementary information. To study these dimensions jointly, we introduce DualEvasion, a benchmark for evasion detection across text and audio in earnings call Q&A. The benchmark contains 505 annotated question-answer pairs from 60 earnings calls, each with two independent labels: textual evasion (direct vs. evasive) and vocal cues operationalized as speaker confidence (confident vs. unconfident). Our experiments show that state-of-the-art multimodal models struggle to detect vocal confidence, particularly on unconfident responses. Our analysis suggests these models interpret acoustic cues in isolation rather than relative to each speaker's baseline. Pro
    
[^37]: 双生世界：基于等变性的弃答机制实现证据接地推理

    Twin Worlds: Equivariance-Based Abstention for Evidence-Grounded Reasoning

    [https://arxiv.org/abs/2608.28018](https://arxiv.org/abs/2608.28018)

    该论文提出“双生世界”（TW）框架，通过等变性检验模型的推理是否真正以证据为依据，使模型在证据不足时能够弃答，从而避免生成看似合理却缺乏证据支撑的答案。

    

    知识密集型推理要求大语言模型（LLM）将答案建立在所提供的证据之上。当证据不足时，理想的情形是模型选择弃答，而不是自信地生成缺乏依据的答案。现有的弃答方法依赖于不确定性估计或证据充分性检查，但二者均无法检验生成过程——即由所提供证据与模型内部记忆参数的交互所驱动的推理过程——是否真正以证据为依据。一个关键因素是，上下文中的实体提及会激活模型记忆中的关联，导致模型生成看似合理却缺乏证据支撑的回答。我们提出了双生世界（Twin Worlds, TW）框架，通过基于等变性的弃答机制来提升知识密集型推理的可靠性：与要求输出保持不变的不变性不同，等变性要求输出在实体……（原文摘要在此处截断）

    arXiv:2608.28018v1 Announce Type: new  Abstract: Knowledge-intensive reasoning requires Large Language Models (LLMs) to ground answers in provided evidence. When evidence is insufficient, it is desirable that models abstain rather than confidently generating unsupported answers. Existing abstention methods rely on uncertainty estimation or evidence sufficiency checks, but neither tests whether the reasoning process for generation, driven by the interaction of provided evidence and the model's internal memory parameters, is actually grounded in the evidence. A key contributing factor is that entity mentions in context activate memorised associations, causing models to generate plausible responses ungrounded in evidence. We propose Twin Worlds (TW), a framework for improving reliability in knowledge-intensive reasoning through equivariance-based abstention: unlike invariance, which requires outputs to remain unchanged, equivariance requires outputs to transform correspondingly under enti
    
[^38]: 超越全局标量：协同词元级统计与深层语义进行对抗性AIGC文本检测

    Beyond Global Scalars: Synergizing Token-Level Statistics and Deep Semantics for Adversarial AIGC Text Detection

    [https://arxiv.org/abs/2608.28009](https://arxiv.org/abs/2608.28009)

    提出NeuroStat端到端框架，通过协同词元级概率统计信息与深层语义隐藏状态来弥合统计与语义之间的鸿沟，同时构建了包含16000个样本的MOSAIC对抗性基准，显著提升了机器生成文本检测在对抗场景下的鲁棒性。

    

    大语言模型的快速发展使得可靠的机器生成文本检测变得十分必要。现有范式通常遵循两条相互孤立的路线：免训练方法依赖全局统计标量（如困惑度），而基于训练的方法则利用语义隐藏状态。这两种方法在对抗场景下都表现出根本性的脆弱性。全局标量作为一种有损压缩，会掩盖交错文本中的局部概率突发性；而纯语义模型则容易过拟合于特定的模型指纹，且易受欺骗攻击。为了揭示这些缺陷，我们提出了MOSAIC，一个包含16000个样本、覆盖全粒度攻击谱的综合对抗性基准。为了应对这些挑战，我们提出了NeuroStat，一个弥合统计与语义鸿沟的端到端框架。NeuroStat在捕获未经压缩的词元级概率logits的同时，还捕获深层语义隐藏状态……

    arXiv:2608.28009v1 Announce Type: new  Abstract: The rapid evolution of large language models necessitates robust machine-generated text detection. Existing paradigms typically follow two isolated tracks. Training-free methods rely on global statistical scalars such as perplexity, while training-based methods utilize semantic hidden states. Both approaches exhibit fundamental vulnerabilities in adversarial scenarios. Global scalars act as lossy compressions that obscure local probabilistic burstiness in interleaved texts, whereas pure semantic models overfit to specific fingerprints and remain susceptible to spoofing. To expose these flaws, we introduce MOSAIC, a comprehensive adversarial benchmark comprising 16000 samples across a full-granularity attack spectrum. To address these challenges, we propose NeuroStat, an end-to-end framework bridging the statistical and semantic gap. NeuroStat captures uncompressed token-level probabilistic logits alongside deep semantic hidden states fro
    
[^39]: 多方对话中话轮转换结果的预测：结合人际亲密度的言语与注视动态可解释建模

    Predicting Turn-Taking Outcomes in Multi-Party Conversation: Interpretable Modelling of Speech and Gaze Dynamics with Interpersonal Closeness

    [https://arxiv.org/abs/2608.27988](https://arxiv.org/abs/2608.27988)

    本研究基于GaMMA四人自由对话语料库，构建了融合注视动态、言语特征与人际亲密度的可解释逻辑回归模型，用于预测话轮转换结果是间隙还是重叠。

    

    流畅的说话者切换是有效对话的基础，它依赖于对话参与者预测何时进入对话的能力。这种能力取决于能否准确解读和表达言语与非言语线索，以表明说话者何时希望接管或放弃话语权。在嘈杂、自然的多方对话环境中，由于存在多个潜在对话参与者，这一过程变得更加复杂。本研究对四人自由对话中注视与言语如何结合感知的人际亲密度来预示对话话语权的转换进行了建模。利用GaMMA语料库，我们使用从每次话轮转换事件之前提取的、可解释且具有行为学动机的特征训练逻辑回归模型，将话语权转换结果分类为间隙还是重叠。预测特征包括注视特征，如转换模式与行为对比、熵、基于注视的受话人身份以及相互注视，以及言语特征……（原文摘要在此处截断）

    arXiv:2608.27988v1 Announce Type: new  Abstract: Smooth speaker transitions are fundamental to effective conversation and rely on an interlocutor's ability to predict when to enter the conversation. This ability depends on accurately interpreting and expressing the verbal and non-verbal cues that signal when a speaker wishes to take or relinquish the floor. The process becomes even more complex in noisy, natural, multi-party settings, with multiple interlocutors available. This study models how gaze and speech, together with perceived interpersonal closeness, signal conversational floor changes in free four-person dialogue. Using the GaMMA corpus, we trained logistic regression models using interpretable, behaviourally motivated features extracted before each turn-taking event to classify floor-transfer outcomes as gaps or overlaps. Predictors included gaze features such as transition motifs and behavioural contrasts, entropy, gaze-based addressee identity, and mutual gaze, alongside s
    
[^40]: QUORUM：基于多标注者的质量优化路由

    QUORUM: QUality-Optimized Routing Using Multiple annotators

    [https://arxiv.org/abs/2608.27974](https://arxiv.org/abs/2608.27974)

    QUORUM是一个预算感知的标注路由框架，它利用基于特征的难度信号在固定预算下动态地将数据实例分配给人类或大语言模型标注者，并通过多标注一致性奖励机制提升标注的可靠性。

    

    arXiv:2608.27974v1 公告类型：新论文 摘要：数据标注仍然是自然语言处理中的核心瓶颈，需要人力投入才能大规模获得高质量标签。虽然大语言模型（LLMs）提供了一种快速且经济的替代方案，但其可靠性高度依赖于具体实例：它们在简单输入上表现良好，但在需要细致推理或上下文理解的样本上常常失败。在这项工作中，我们提出QUORUM（QUality-Optimized Routing Using Multiple annotators，基于多标注者的质量优化路由）来解决这一挑战，这是一个预算感知的路由框架，能够在固定标注预算下动态地将每个实例分配给人类标注者或LLM标注者。与以往依赖模型置信度或不确定性估计的方法不同，QUORUM利用基于特征的信号来估计实例难度，并支持对每个实例进行多次标注，通过基于一致性的奖励机制将多个标注结果组合起来，从而提升标注的可靠性。我们在多样化的封闭式和开放式（任务上对QUORUM进行了评估）

    arXiv:2608.27974v1 Announce Type: new  Abstract: Data annotation remains a central bottleneck in natural language processing, requiring human effort to obtain high-quality labels at scale. While Large Language Models (LLMs) offer a fast and cost-effective alternative, their reliability is highly instance-dependent: they perform well on simple inputs but often fail on examples requiring nuanced reasoning or contextual understanding. In this work, we address this challenge with QUORUM (QUality-Optimized Routing Using Multiple annotators), a budget-aware routing framework that dynamically assigns each instance to human or LLM annotators under a fixed annotation budget. Unlike prior approaches relying on model confidence or uncertainty estimates, QUORUM leverages feature-based signals to estimate instance difficulty and supports multiple annotations per instance, combining them through agreement-based rewards to improve reliability. We evaluate QUORUM across diverse closed- and open-ended 
    
[^41]: DisCTI：谁需要及时知晓？自动化的行业感知网络威胁情报分发

    DisCTI: Who Needs to Know Timely? Automated Sector-Aware Cyber Threat Intelligence Dissemination

    [https://arxiv.org/abs/2608.27967](https://arxiv.org/abs/2608.27967)

    该论文提出DisCTI，将面向行业的网络威胁情报分发建模为多标签分类问题，实现自动化的行业感知CTI及时分发，解决了现有平台（如MISP）中98%事件缺乏行业标注、情报运营价值受限的问题。

    

    网络威胁情报（CTI）的及时分发对于组织快速有效地开展事件响应至关重要。当有效的CTI在正确的时间传递给正确的行业时，相同的攻击往往能够被遏制或缓解。然而，当今快速扩张的CTI环境使分析师不堪重负，他们必须从海量且异构的情报源中进行筛选。诸如恶意软件信息共享平台（MISP）等现有平台提供了行业标签功能（如能源、金融、政府），但在实践中，这些功能基本处于未使用状态（98%的事件未被归类）。这种缺乏自动化和及时行业映射的问题严重限制了共享情报的运营价值，使得属于关键信息基础设施行业的组织尤其容易暴露于风险之中。为了填补这一空白，我们将面向行业的CTI分发问题形式化为一个多标签分类问题。利用……

    arXiv:2608.27967v1 Announce Type: cross  Abstract: The timely dissemination of cyber threat intelligence (CTI) is critical for organizations to mount swift and effective incident response. When valid CTI is delivered to the right sector at the right time, identical attacks can often be contained or mitigated. However, today's rapidly expanding CTI landscape overwhelms analysts, who must sift through massive and heterogeneous feeds. Existing platforms such as the Malware Information Sharing Platform (MISP) provide sector tagging features (e.g., energy, finance, government), but in practice, these remain largely unmapped (98% of events are left uncategorized). This lack of automated and timely sector mapping severely limits the operational value of shared intelligence, leaving organizations that belong especially to the critical information infrastructure sector exposed.   To address this gap, we formulate sector-targeted CTI dissemination as a multilabel classification problem. Leveragi
    
[^42]: 韩语谓词形态中受词汇制约的实现歧义

    Lexically conditioned realization ambiguity in Korean predicate morphology

    [https://arxiv.org/abs/2608.27966](https://arxiv.org/abs/2608.27966)

    韩语谓词的表层形式并非由规范词素序列唯一决定：同形同音的谓词会因词汇意义与论元结构不同而分属不同的屈折实现类别，形成“同音异折”现象。

    

    本文将韩语的表层实现与形态分析区分开来加以考察，探讨规范词素序列与语法范畴标签的组合是否能唯一确定对应的表层形式。对于一类受限但在理论上极具启发意义的韩语谓词而言，答案是否定的：在这些情形中，形式上相同或近乎相同的“词干+词尾”配置，会因词汇身份及所属实现类别的不同而产生不同的表层输出。我们将这一现象分析为“伴随屈折分歧的同音（同形）异义”现象，重点考察规则形与ㄷ不规则形的配对、规则形与ㅂ不规则形的配对，以及르不规则形与러不规则形的配对。这些案例表明，仅凭词干形状与词尾并不总能决定表层实现；相反，词汇意义、次范畴化框架和语义角色结构有助于识别目标谓词，谓词进而决定其实现类别……

    arXiv:2608.27966v1 Announce Type: new  Abstract: This paper examines Korean surface realization as distinct from morphological analysis. It asks whether a sequence of canonical morphemes and grammatical category labels uniquely determines the corresponding surface form. The answer is negative for a restricted but theoretically revealing class of Korean predicates. In these cases, formally identical or near-identical stem-ending configurations yield different outputs depending on lexical identity and realization class membership. We analyze this phenomenon as homonymy with inflectional divergence, focusing on regular versus digeut irregular pairs, regular versus bieup irregular pairs, and reu irregular versus reo irregular pairs. These cases show that stem shape and ending alone do not always determine surface realization. Instead, lexical meaning, subcategorization, and semantic role structure help identify the intended predicate; the predicate determines the realization class; and the
    
[^43]: 实体-记忆图检索提升长对话问答中的证据覆盖率

    Entity-Memory Graph Retrieval Improves Evidence Coverage in Long-Conversation Question Answering

    [https://arxiv.org/abs/2608.27925](https://arxiv.org/abs/2608.27925)

    提出实体-记忆图检索方法，通过共享实体和有向时间边连接对话轮次，将长对话问答的证据召回率从79.75%提升至84.48%，显著优于匹配的稠密检索基线。

    

    实体-记忆图检索将对话轮次保存为逐字记录的记忆节点，通过共享实体连接重复提及的内容，并用有向时间顺序边连接相邻的记忆。在查询时，检索器从实体门控开始，经过语义融合和一跳时间顺序恢复，最后进行稠密回填。该路径能够保留稠密余弦排序原本会遗漏的邻近记忆。一个匹配的稠密对照组共享相同的记忆和查询向量、上下文预算、要求的答案协议以及评估器，从而将图结构的影响与阅读器的变化隔离开来。在来自十个LoCoMo对话的1,986个问题上，图检索将top-k 25时的官方证据召回率从79.7468%提升至84.4842%。该召回率优势在top-k 5到50的范围内均得到支持，而没有任何匹配的截断点支持最终答案F1的整体差异。四种符合论文要求的请求配置在所测试的GPT-3.5和……上支持了实证稳健性。

    arXiv:2608.27925v1 Announce Type: new  Abstract: Entity-Memory graph retrieval keeps dialogue turns as verbatim Memory nodes, links repeated mentions through shared Entities, and connects adjacent Memories with directed chronological edges. At query time the retriever moves from Entity gating through semantic fusion and one-hop chronological recovery to dense backfill. The path can keep a neighboring Memory that dense cosine ranking would otherwise omit. A matched dense control shares the Memory and query vectors, context budget, requested answer protocol, and evaluator, isolating graph structure from changes to the reader.   On 1,986 questions from ten LoCoMo conversations, graph retrieval raises official evidence recall at top-k 25 from 79.7468% to 84.4842%. The recall advantage is supported from top-k 5 to 50, while no matched cutoff supports an overall final-answer F1 difference. Four paper-eligible requested configurations support empirical robustness across the tested GPT-3.5 and
    
[^44]: 什么使智能体记忆对可靠处理不可回答问题变得有用？

    What Makes Agent Memory Useful for Reliable Unanswerable Question Handling?

    [https://arxiv.org/abs/2608.27924](https://arxiv.org/abs/2608.27924)

    该论文在统一的智能体RAG框架下系统研究了记忆在处理不可回答问题中的作用，发现记忆带来的提升是选择性且脆弱的，跨模型记忆复用比跨数据集迁移更可行，且决策引导比轨迹塑造更能保留记忆收益。

    

    可靠地处理不可回答问题（UAQ）对于基于大语言模型的可信智能体至关重要。尽管记忆在智能体系统中被广泛使用，但其在可靠UAQ处理中的作用仍不明确。我们在统一的智能体检索增强生成（agentic RAG）框架下，对智能体记忆用于UAQ处理开展了系统性研究，评估了四种代表性记忆方法在三个UAQ相关数据集和两个基础模型上的表现。我们发现，记忆在某些设置下能够提升UAQ性能，但这种提升是选择性的而非普遍的，并且在数据集偏移下依然脆弱。有趣的是，跨模型记忆复用往往比跨数据集迁移更可行，这表明可回答性模式的变化对记忆复用构成的挑战大于基础模型本身的变化。我们进一步发现，UAQ的性能提升通过决策引导比通过轨迹塑造更能得到保留，且记忆的有效性强烈依赖于表征……（原文摘要在此处截断）

    arXiv:2608.27924v1 Announce Type: new  Abstract: Reliable handling of unanswerable questions (UAQs) is critical for trustworthy LLM-based agents. Although memory is widely used in agent systems, its role in reliable UAQ handling remains unclear. We present a systematic study of agent memory for UAQ handling under a unified agentic RAG framework, evaluating four representative memory methods across three UAQ-related datasets and two base models.   We find that memory can improve UAQ performance in some settings, but such gains are selective rather than universal and remain fragile under dataset shift. Interestingly, cross-model memory reuse is often more feasible than cross-dataset transfer, suggesting that shifts in answerability patterns pose a greater challenge to memory reuse than changes in the base model itself. We further find that UAQ gains are more strongly preserved through decision guidance than through trajectory shaping, and that memory effectiveness depends strongly on rep
    
[^45]: 通过博弈论视角审视AI对齐：综述

    AI Alignment through a Game-theoretic Lens: A Survey

    [https://arxiv.org/abs/2608.27910](https://arxiv.org/abs/2608.27910)

    本综述以博弈论视角系统梳理AI对齐研究，围绕偏好多样性、对齐优先级和时间动态三大挑战组织文献，阐明了博弈论分析真正发挥作用之处以及构建鲁棒、自适应、可验证AI系统仍待解决的难题。

    

    随着大语言模型和日益强大的AI智能体被部署到高风险场景中，使其与复杂的人类价值观保持一致已成为核心挑战。现有的对齐方法虽然在提升有用性、无害性和可控性方面卓有成效，但往往难以捕捉那些依赖于上下文、不具传递性、并由动态多方交互塑造的真实世界偏好。本综述通过博弈论的视角审视AI对齐研究。具体而言，它围绕关键的博弈论要素组织近期进展，并围绕三大挑战综合梳理相关文献：偏好多样性、对齐优先级和时间动态。这一视角阐明了当前对齐方法在哪些方面真正受益于博弈论分析，哪些方面的框架应用较为宽松，以及在构建鲁棒、自适应、可验证的AI系统方面仍面临哪些挑战。

    arXiv:2608.27910v1 Announce Type: cross  Abstract: As large language models and increasingly capable AI agents are deployed in high-risk settings, aligning them with complex human values has become a central challenge. Existing alignment methods, while effective in improving helpfulness, harmlessness, and controllability, often struggle to capture real-world preferences that are context-dependent, non-transitive, and shaped by dynamic multi-party interactions. This survey reviews AI alignment through a game-theoretic lens. Specifically, it organizes recent progress around key game-theoretic elements and synthesizes the literature along three challenges: preference diversity, alignment priority, and temporal dynamics. This perspective clarifies where current alignment methods genuinely benefit from game-theoretic analysis, where the framework is looser, and what challenges remain in building robust, adaptive, and verifiable AI systems.
    
[^46]: LandingAgent：面向落地页的参考标注数据集与智能体生成框架

    LandingAgent: A Reference-Annotated Dataset and Agentic Generation Framework for Landing Pages

    [https://arxiv.org/abs/2608.27902](https://arxiv.org/abs/2608.27902)

    该论文提出了参考画像数据集LandingBench和三阶段智能体生成框架LandingAgent，通过从真实落地页中抽象提取可复用的设计模式来引导生成过程，解决了大语言模型直接生成落地页时产生的模板化和内容缺乏依据的问题。

    

    落地页是以目标为导向的网页界面，它们必须在传达特定目标价值主张的同时，组织好信息流、视觉层次结构和行动号召（CTA）。尽管大语言模型能够根据自然语言提示生成看似合理的网页代码，但直接生成往往会产生通用化的模板和缺乏依据的劝说性内容。我们研究了基于目标、以参考为引导的落地页生成任务，即系统需要通过借鉴真实页面中的可复用模式（而非直接复制）来为新目标创建可执行的页面。我们提出了LandingBench，这是一个参考画像数据集，它将真实落地页抽象为章节序列、布局模式、语气描述、视觉强调和CTA结构。基于LandingBench，我们提出了LandingAgent，这是一个三阶段的智能体框架，该框架首先对目标进行画像分析，然后构建参考引导的线框图，最后通过基于批判反馈的打磨来优化页面。

    arXiv:2608.27902v1 Announce Type: new  Abstract: Landing pages are goal-oriented web interfaces that must communicate a target-specific value proposition while organizing information flow, visual hierarchy, and calls to action (CTA). Although large language models can generate plausible webpage code from natural-language prompts, direct generation often yields generic templates and unsupported persuasive claims. We study target-grounded, reference-guided landing-page generation, where a system must create an executable page for a new target by adapting reusable patterns from real pages without copying them. We introduce LandingBench, a reference-profile dataset that abstracts real landing pages into section sequences, layout patterns, tone descriptors, visual emphasis, and CTA structure. Building on LandingBench, we propose LandingAgent, a three-phase agentic framework that profiles the target, constructs a reference-guided wireframe, and refines the page through critique-guided polish
    
[^47]: OpenStamp：面向开源语言模型的水印技术

    OpenStamp: A Watermark for Open-Source Language Models

    [https://arxiv.org/abs/2608.27899](https://arxiv.org/abs/2608.27899)

    OpenStamp通过仅修改开源语言模型的反嵌入层，将水印逻辑直接编码进模型权重，解决了传统采样概率水印在白盒场景下可被用户禁用的问题，在几乎不损失模型能力的前提下实现了更优的检测性能和更强的鲁棒性。

    

    随着大语言模型（LLM）生成内容的日益普及，水印技术被视为一种将文本归属于LLM并与人类撰写内容相区分的有前景的方法。一类突出的技术通过修改token的采样概率，在生成文本中嵌入细微但可检测的信号。然而，这类方法并不适用于开源模型，因为用户拥有白盒访问权限，可以在推理过程中轻松禁用水印。在这项工作中，我们提出了OpenStamp，一种水印技术，它通过仅修改最终的投影层（即反嵌入层，unembedding layer），将水印逻辑直接编码到模型权重中。通过在两个模型上的实验，我们证明OpenStamp实现了更优的检测性能，且与先前方法相比，模型能力的退化极小。植入的水印经过专门设计，并经实验证实，对扰动（p…

    arXiv:2608.27899v1 Announce Type: new  Abstract: With the growing prevalence of large language model (LLM) generated content, watermarking is considered a promising approach for attributing text to LLMs and distinguishing it from human-written content. A prominent class of techniques embeds subtle but detectable signals in generated text by modifying token sampling probabilities. However, such methods are unsuitable for open-source models, where users have white-box access and can easily disable watermarking during inference. In this work, we introduce OpenStamp, a watermarking technique that encodes the watermarking logic directly into the model weights by modifying only the final projection, or unembedding, layer. Through experiments across two models, we show that OpenStamp achieves superior detection performance, with minimal degradation in model capabilities compared to prior methods. The implanted watermark is explicitly designed, and empirically confirmed, to be more robust to p
    
[^48]: AI写作具有一致的风格计量学足迹，但AI编辑则不然

    AI Writers Have a Consistent Stylometric Footprint, but AI Editors Do Not

    [https://arxiv.org/abs/2608.27855](https://arxiv.org/abs/2608.27855)

    本文发现AI生成的文本具有跨8个模型和5个领域保持一致的风格计量学“足迹”（主要由熵和词汇多样性等特征构成），可用于检测AI生成文本，但AI编辑过的人类文本并不会留下同样的足迹。

    

    大型语言模型（LLM）生成的文本已被证明在风格计量学上与人类撰写的文本截然不同。但LLM越来越多地不仅被用于生成文本，还被用于编辑人类写作，目前尚不清楚这两种用途是否会留下相同的痕迹。我们证明，AI生成会留下一个一致的“风格计量足迹”：一小部分特征（主要是熵和词汇多样性）在8个LLM和5个领域中始终能够将AI生成的文本与人类写作区分开来，而其余特征则严重依赖于具体领域和生成器。然而，AI编辑并不会重现同样的足迹。相对于其人类撰写的原始文本，AI编辑后的文本仅表现出词汇多样性的小幅增加以及……

    arXiv:2608.27855v1 Announce Type: new  Abstract: Text generated by large language models (LLMs) has been shown to be stylometrically distinct from human-written text \citep{andreDetectingAIAuthorship2023, shahDetectingUnmaskingAIGenerated2023, oparaStyloAIDistinguishingAIGenerated2024, soto2024fewshot, liLinguisticDifferencesAI2025, selviogluFeatureExtractionAnalysis2025}. But LLMs are increasingly used not only to generate text but also to edit human writing, and it is unclear whether the two leave the same trace. We show that AI generation leaves a consistent ``stylometric footprint'': a small subset of features, primarily entropy and lexical diversity, consistently separates AI-generated text from human writing across 8 LLMs and 5 domains, while the remaining features depend heavily on the domain and generator. AI editing, however, does not reproduce the same footprint. Relative to their human-written sources, AI-edited texts show only a small increase in lexical diversity and a dec
    
[^49]: 韵律在翻译中丢失了吗？跨语言的细粒度韵律相似性分析

    Is Prosody Lost in Translation? Fine-Grained Cross-Lingual Prosody Similarity Across Languages

    [https://arxiv.org/abs/2608.27848](https://arxiv.org/abs/2608.27848)

    该研究首次利用多语言配音数据对英德、英西、英法语言对进行了细粒度的跨语言韵律分析，揭示了某些语言间韵律结构存在固有的跨语言相关性，为在语音到语音翻译系统中有效融入韵律提供了重要依据。

    

    韵律在语音翻译中起着重要作用，它传达了词汇内容之外的信息，如强调、情感和意图。然而，尽管富有表现力的语音到语音翻译（S2ST）最近取得了进展，但人们对韵律模式在不同语言之间的相似性和差异性知之甚少。理解这些跨语言的相似性和差异性对于有效地将韵律融入富有表现力的S2ST系统至关重要。在这项工作中，我们使用英语-德语、英语-西班牙语和英语-法语语言对的多语言配音数据，首次对韵律进行了细粒度的跨语言分析。我们分析了源语音和目标语音之间音高、能量和时间特征模式的相似性，并研究了影响这种相似性的语言因素和对齐相关因素。我们的分析揭示了某些语言之间韵律结构中固有的跨语言相关性。

    arXiv:2608.27848v1 Announce Type: cross  Abstract: Prosody plays an important role in speech translation, conveying information such as emphasis, emotion, and intent beyond lexical content. However, despite recent progress in expressive speech-to-speech translation (S2ST), little is known about how prosodic patterns are similar/different across languages. Understanding these cross-lingual similarities and differences is crucial for effectively incorporating prosody into expressive S2ST systems. In this work, we present the first fine-grained cross-lingual analysis of prosody using multilingual dubbing data across English-German, English-Spanish, and English-French language pairs. We analyze the similarity of pitch, energy, and temporal feature patterns between source and target speech and investigate the linguistic and alignment-related factors affecting this similarity. Our analysis reveals inherent cross-lingual correlations in prosodic structure between certain languages. The findin
    
[^50]: EvoHarmBench：通过迭代式类人规避攻击突破内容审核

    EvoHarmBench: Breaking Content Moderation with Iterative Human-Like Evasion

    [https://arxiv.org/abs/2608.27844](https://arxiv.org/abs/2608.27844)

    本文提出了首个面向内容审核系统的动态对抗评估框架 EvoHarmBench，通过在语义簇层面迭代演化类人规避策略，揭示了静态基准分数与真实部署审核效果之间的显著性能差距。

    

    现有的有害内容检测评估主要依赖静态基准，难以反映真实内容平台中交互式对抗生态系统——在真实平台上，用户会根据审核反馈不断修改自己的表达方式。这种不匹配导致离线基准分数与线上部署效果之间存在显著的性能差距。据我们所知，我们提出了 EvoHarmBench，这是首个面向内容审核系统的动态对抗评估框架。该框架采用迭代优化循环，在语义簇层面演化规避策略，同时兼顾规避成功率与人类可读性。我们系统性地评估了在真实审核系统中广泛应用的基于大语言模型（LLM）的防御模型。评估涵盖五大违规类别下的229个语义子簇，这些子簇源自5,002条真实世界的对抗样本。

    arXiv:2608.27844v1 Announce Type: new  Abstract: Existing evaluations of harmful content detection rely predominantly on static benchmarks, which struggle to reflect the interactive adversarial ecosystem of real-world content platforms where users continuously revise their expressions in response to moderation feedback. This mismatch creates a significant performance gap between offline benchmark scores and online deployment effectiveness. To the best of our knowledge, we present EvoHarmBench, the first dynamic adversarial evaluation framework for content moderation systems. The framework employs an iterative optimization loop that evolves evasion strategies at the semantic-cluster level, while simultaneously optimizing for evasion success and human readability. We systematically evaluate LLM-based defense models which are widely used in real world moderation systems. The evaluation covers 229 semantic sub-clusters across five violation categories, derived from 5,002 real-world adversa
    
[^51]: 合成语言能动性：具身有死智能体如何通过后果性社会经验学习语言可供性

    Synthetic Linguistic Agency: How an Embodied Mortal Agent Learns Linguistic Affordances through Consequential Social Experience

    [https://arxiv.org/abs/2608.27843](https://arxiv.org/abs/2608.27843)

    该论文提出了“合成语言能动性”的可检验标准，并基于稳态调节强化学习构建了一个以死亡为基础的具身有死智能体（EMA），使其通过后果性社会经验习得语言可供性。

    

    当代语言模型能够流利地对话并影响人类决策，然而它们的交流并不进入一种持续的、脆弱的、属于自己的生活。语言能动性理论将这种缺失的联系识别为语言能动性，并通过具身性、语言参与性和脆弱性来刻画它：一个能够行动并承担后果的身体，能够同时改变智能体及其伙伴的交互，以及一个可以被维持或失去的未来。两项协调的研究考察了这种组织如何在人工系统中出现。首先，我们将这些关系转化为合成语言能动性（SLA）的可检验标准，并识别出若干现有的SLA系统。其次，基于稳态调节强化学习，我们开发了一个以死亡为基础的语言强化学习模型，并将其实现为一个具身有死智能体（EMA）。EMA学习说话方式如何改变伙伴的意愿……

    arXiv:2608.27843v1 Announce Type: new  Abstract: Contemporary language models can converse fluently and influence human decisions, yet their exchanges do not enter a continuing, vulnerable life of their own. Linguistic-agency theory identifies this missing connection as linguistic agency and characterizes it through embodiment, linguistic participation, and precariousness: a body that acts and bears consequences, interaction that changes both agent and partner, and a future that can be sustained or lost. Two coordinated studies examine how this organization can appear in artificial systems. First, we translate these relations into inspectable criteria for Synthetic Linguistic Agency (SLA) and identify several existing SLA systems. Second, building on Homeostatically Regulated Reinforcement Learning, we develop a mortality-grounded linguistic-reinforcement-learning model and instantiate it in an Embodied Mortal Agent (EMA). The EMA learns how ways of speaking change a partner's willingn
    
[^52]: 面向已知任务音频大语言模型评估的生成式音频调用审计

    Auditing Generative Audio Calls for Known-Task Audio-LLM Evaluation

    [https://arxiv.org/abs/2608.27817](https://arxiv.org/abs/2608.27817)

    该论文将音频大语言模型的评估建模为受控的调用决策问题，发现在已知封闭集任务上，有监督编码器（如CLAP和WavLM）无需调用生成式音频模型即可取得接近最优的准确率，从而揭示了传统“波形提示对比ASR转录”的评估方式混淆了声学证据获取与生成模型调用这两个因素。

    

    语音和音频大语言模型通常通过比较波形提示是否优于自动语音识别（ASR）转录文本来进行评估。对于已知的封闭集任务，这种比较混淆了两个因素：获取声学证据的途径，以及调用生成式音频模型的需求。我们将这一区分评估为一个受控的调用决策问题。对于每个样本，一个策略可以在以下选项中做出选择：保留转录文本标签、使用来自对比语言-音频预训练（CLAP）、音频频谱图Transformer（AST）或WavLM的编码器证据，或调用Qwen2-Audio、Qwen2.5-Omni或MOSS-Audio；其中决定性的消融实验在保持选择器和开发协议不变的前提下移除所有生成式操作。在VocalSound数据集上，转录文本的准确率仅为0.296，说明确实需要波形信息。然而，有监督的CLAP和WavLM对照方法在完全不调用生成式音频模型的情况下分别达到了0.850和0.854的准确率。带有生成式操作的选择器在使用12.5%的调用预算的情况下达到了0.925的准确率（摘要在此处截断）。

    arXiv:2608.27817v1 Announce Type: cross  Abstract: Speech and audio LLMs are often evaluated by asking whether a waveform prompt beats an automatic speech recognition (ASR) transcript. For known closed-set tasks, that comparison conflates two factors: access to acoustic evidence and the need to call a generative audio model. We evaluate this distinction as a controlled call-decision problem. For each example, a policy chooses among keeping a transcript label, using encoder evidence from Contrastive Language-Audio Pretraining (CLAP), Audio Spectrogram Transformer (AST), or WavLM, and calling Qwen2-Audio, Qwen2.5-Omni, or MOSS-Audio; the decisive ablation removes all generative actions while keeping the selector and development protocol fixed. On VocalSound, transcripts reach 0.296 accuracy, so waveform information is needed. Yet supervised CLAP and WavLM controls reach 0.850 and 0.854 with no generative audio calls. A selector with generative actions reaches 0.925 accuracy using 12.5% c
    
[^53]: PersonaEdit：面向个性化模型编辑的代表性样本选择

    PersonaEdit: Representative Sample Selection for Personalized Model Editing

    [https://arxiv.org/abs/2608.27816](https://arxiv.org/abs/2608.27816)

    提出PersonaEdit，一种基于隐表示聚类与比例分层采样的代表性编辑样本选择策略，使模型编辑能够高效、低成本地实现LLM个性化。

    

    个性化在LLM应用中引起了越来越多的关注，然而现有的基于检索的方法严重依赖检索质量，且在长期交互中性能会下降。模型编辑通过直接修改模型内部参数来纳入新知识，已在事实知识编辑任务中展现出有效的知识修改能力，可能为个性化提供一种潜在的解决方案。然而，将模型编辑扩展到个性化并非易事：编辑大量用户数据会增加计算成本，并导致各编辑之间的相互干扰，因此需要有效的样本选择方法。为解决这一问题，我们提出了PersonaEdit，一种通过比例分层采样来选择代表性编辑样本的隐表示聚类策略。实验表明，模型编辑对个性化是有效的，且我们的选择策略能够保留大部分性能。

    arXiv:2608.27816v1 Announce Type: new  Abstract: Personalization has attracted growing interest in LLM applications, yet existing retrieval-based approaches depend heavily on retrieval quality and degrade in long-term interactions. Model editing, which directly modifies internal model parameters to incorporate new knowledge, has demonstrated effective knowledge modification capabilities in factual knowledge editing tasks and may provide a potential solution for personalization. However, scaling model editing to personalization is non-trivial. Editing large amounts of user data increases computational cost and causes interference among edits, motivating the need for effective sample selection. To address this issue, we propose, PersonaEdit, a hidden representation clustering strategy that selects representative editing samples through proportional stratified sampling. Experiments show that model editing is effective for personalization, and that our selection strategy preserves most of 
    
[^54]: 通过线性距离与相似性感知熵的视角审视大语言模型中的句法表示

    Representation of syntax in LLMs through the lens of linear distance and similarity-aware entropy

    [https://arxiv.org/abs/2608.27813](https://arxiv.org/abs/2608.27813)

    本文将结构探针的评估指标按句法关系分解为UASL，发现相关词间线性距离的统计特征以及句法关系中心词的相似性感知熵这两个因素能够预测各句法关系重建准确率的大部分变异，且该结论在不同规模和架构的模型上均成立。

    

    结构探针由Hewitt和Manning提出，用于从神经语言模型的潜在表示中重建句法树。其评估方式是计算在标注语料库上被正确重建的句法树边所占的比例（以无向无标签依存得分UAS来衡量）。在本研究中，我们将这一度量进行分解，提出按标签划分的无向依存得分（UASL），它可以分别评估每种句法关系的重建准确率，从而揭示了与语言学区分相对应的各类句法关系之间的重要差异。此外，我们识别出两个能够预测UASL在各句法关系间大部分变异的因素：(i) 相关词之间线性距离（对数尺度上）的均值与离散程度，以及 句法关系中中心词的多样性（即相似性感知熵）。这些结论在一系列不同规模和架构的模型上均成立，从而阐明了……（原文此处截断）

    arXiv:2608.27813v1 Announce Type: new  Abstract: Structural probes were introduced by Hewitt and Manning to reconstruct syntactic trees from a neural language model's latent representations. They are evaluated by calculating the proportion of syntactic tree edges correctly reconstructed over an annotated corpus (as measured by undirected unlabeled attachment score). Here, we disaggregate this measure, considering undirected attachment score by label (UASL), which assesses the reconstruction accuracy of each syntactic relation separately, establishing important differences among relations that overlap linguistic distinctions. Moreover, we identify two factors that predict most of UASL's variability across relations: (i) the mean and dispersion of the linear distance (on a log scale) between the related words, and (ii) the diversity (similarity-aware entropy) of the syntactic relation's head. These results, which hold across a range of model sizes and architectures, shed light on the deg
    
[^55]: CEDAR：自动机作为语言引导具身行动的可验证接口

    CEDAR: Automata as Verifiable Interfaces for Language-Guided Embodied Action

    [https://arxiv.org/abs/2608.27797](https://arxiv.org/abs/2608.27797)

    CEDAR框架将自然语言指令中的技能与约束统一表示为环境事件轨迹上的确定性有限自动机，通过自动机交集运算使具身智能体的行为在构造上即可强制执行约束，从而实现可验证、可组合、可修复的语言引导具身行动。

    

    对具身智能体的自然语言任务指令很少仅仅是目标规范：用户还会施加一些必须在世界变化时持续保持的约束。代码生成型大语言模型智能体可以为这类指令产生看似合理的行为，但其自由形式的程序无法提供稳定的对象来用于验证、与新约束组合、或从失败的执行轨迹中修复。我们提出了CEDAR，一个反例引导的框架，它将指令落地为环境事件轨迹上的正则语言。CEDAR使用语言模型进行语义判断，并利用执行轨迹进行修正，然后将技能和规范都表示为确定性有限自动机。这将约束转化为可执行的有限状态对象：一个学到的技能可以与一个学到的“夜间休眠”或“停留在此生物群系”等规范进行交集运算，从而产生一个通过构造而非重复提示来强制执行所学约束的控制器。在Minecraft中，……

    arXiv:2608.27797v1 Announce Type: cross  Abstract: Natural-language tasking of embodied agents is rarely just goal specification: users also impose constraints that must persist while the world changes. Code-generating LLM agents can produce plausible behaviors for such instructions, but their free-form programs provide no stable object to verify, compose with new constraints, or repair from a failing trace. We present CEDAR, a counterexample-guided framework that grounds instructions as regular languages over environment event traces. CEDAR uses a language model for semantic judgments and execution traces for correction, then represents both skills and specifications as deterministic finite automata. This turns constraints into executable finite-state objects: a learned skill can be intersected with a learned sleep at night or stay in this biome specification, yielding a controller that enforces the learned constraint by construction rather than by repeated prompting. In Minecraft, wi
    
[^56]: 音频-视觉大语言模型中的组合性失败：跨模态冲突下的深层先验主导

    Compositional Failure in Audio-Visual LLMs: Late-Layer Prior Dominance Under Cross-modal Conflict

    [https://arxiv.org/abs/2608.27785](https://arxiv.org/abs/2608.27785)

    本研究揭示了音频-视觉大语言模型在跨模态冲突下存在“先验主导”失败模式——模型后期层（集中于约25.5层）固守内部偏好的答案模式而忽视冲突输入，导致准确率大幅下降，且增强时序对齐仅能改变答案偏差而无法提升组合泛化能力。

    

    我们研究音频-视觉冲突作为AV-LLMs（音频-视觉大语言模型）的组合泛化测试：模型必须结合同步但语义不兼容的音频和视频证据，并判断该配对是否匹配。在VideoLLaMA 2-7B-AV上，三种对齐配置在AVHBench的评分精确字符串是/否子集上仍接近随机水平，尽管其输出先验发生了显著变化。类似地，现成的InternVideo2在跨模态冲突下准确率特异性下降了32.3%，并伴随17.3%的指令遵循失败。我们将这种失败模式称为“先验主导”（prior dominance）：模型后期层对内部偏好答案模式的坚定承诺，而这种承诺与冲突输入的关联较弱。为解释这一行为，我们进行了机制可解释性分析，发现这种承诺集中在25.5±1层。我们进一步表明，更强的时序对齐会改变答案偏差，但并不能改善组合性表现。

    arXiv:2608.27785v1 Announce Type: new  Abstract: We study audio-visual conflict as a compositional generalization test for AV-LLMs: the model must combine synchronized but semantically incompatible audio and video evidence and decide whether the pair matches. On VideoLLaMA 2-7B-AV, three alignment configurations remain nearchance on the scored exact-string Yes/No subset of AVHBench, even though their output priors shift substantially. Similarly, off-the-shelf InternVideo2 experienced a 32.3% accuracy decrease specifically under cross-modal conflict, accompanied by a 17.3% instruction-following failure. We call this failure mode prior dominance: late-layer commitment to an internally preferred answer pattern that is weakly grounded in the conflicting inputs. To explain this behavior, we conduct a mechanistic interpretability analysis and find that commitment remains concentrated at 25.5 $\pm$ 1 layers. We show that stronger temporal alignment changes answer bias, but do not improve comp
    
[^57]: SURE-Challenge：在语音大模型生成之前评估语音证据

    SURE-Challenge: Evaluating Speech Evidence Before Speech-LLM Generation

    [https://arxiv.org/abs/2608.27783](https://arxiv.org/abs/2608.27783)

    该论文提出 SURE-Challenge 基准，用于评估语音大模型在生成回答之前对不支持输入（静音、噪声、合成音调、嘈杂语音）的拒绝能力，并证明一个简单的“能量加 Whisper 分数”规则可将不支持输入的拒绝数从 15/204 提升至 196/204，同时不损失有效输入的准确率。

    

    语音大模型（Speech LLMs）通常在其作出回答之后才被评估，尽管操作系统首先必须决定是否应将音频波形发送给模型。我们为此准入步骤定义了“语音不支持拒绝评估挑战”（Speech-Unsupported Rejection Evaluation Challenge，简称 SURE-Challenge）。该基准将源自 LibriSpeech 的转录和首词问答任务与不支持的输入——静音、有色噪声、合成音调以及来源模糊的嘈杂语音——配对，并采用互不相交的来源划分以防止泄漏。前端消融实验使用 Qwen2-Audio；随后将选定的“能量加 Whisper 分数”规则在六个语音/音频大模型之前进行重放验证。在经过泄漏筛查的 474 行 SURE-Extended 测试集上，原始 Qwen2-Audio 仅拒绝 204 个不支持输入中的 15 个，而固定规则可拒绝其中 196 个，且支持样本的准确率保持不变。外部检查界定了这一数字的边界：随着 Whisper 分数阈值的收紧，Common Voice 的保留率随之下降，而变速嘈杂语音在 54 个片段中仅产生 18 到 24 个被拒绝的片段。

    arXiv:2608.27783v1 Announce Type: cross  Abstract: Speech LLMs are usually graded after they answer, although an operating system first has to decide whether a waveform should be sent to the model. We define the Speech-Unsupported Rejection Evaluation Challenge (SURE-Challenge) for this admission step. The benchmark pairs LibriSpeech-derived transcription and first-word question answering with unsupported silence, colored noise, synthetic tones, and source-ambiguous babble under disjoint source splits. Front-end ablations use Qwen2-Audio; the selected energy-plus-Whisper-score rule is then replayed before six speech/audio LLMs. On the 474-row leakage-screened SURE-Extended test set, raw Qwen2-Audio rejects 15/204 unsupported inputs, whereas the fixed rule rejects 196/204 and leaves supported accuracy unchanged. External checks delimit this number: Common Voice retention drops as the Whisper-score threshold is tightened, and no-speed babble gives 18 to 24 rejected clips out of 54 across
    
[^58]: 记忆不等于提取：紧的差分隐私界与审计盲区

    Memorization Is Not Extraction: Tight Differential-Privacy Bounds and Audit Blind Spots

    [https://arxiv.org/abs/2608.27782](https://arxiv.org/abs/2608.27782)

    该论文精确刻画了差分隐私对反事实记忆化与自适应提取这两个度量的紧致控制界，证明二者互不控制，从而揭示了差分隐私作为统一防护代理时存在的审计盲区。

    

    大型语言模型中的记忆化是通过一系列定义来度量的，这些定义之间的形式化关系尚不清楚，而差分隐私（DP）被视为一种能同时抵御所有这些定义的代理。我们确定了其中最具实际意义的两个定义——反事实记忆化与自适应提取——的精确DP常数，并证明它们彼此之间互不控制。在 f-DP 框架下，任何具有列表预算 m 的自适应提取协议，对于无知基线 κ，其成功概率至多为 1−f(κ)，且该界在一稠密的基线集合上是紧的：DP 对提取的控制恰好精确到一个关于秘密先验可猜测性的阈值。最小熵以分布无关的方式证明了该基线：在纯 ε-DP 下，H∞ ≥ ε log₂ e + log₂(m/τ) 对任意先验都能将提取风险控制在 τ ≤ 1/2 以下，且在均匀先验上是精确的。在记忆化方面，f-DP（摘要在此处截断）

    arXiv:2608.27782v1 Announce Type: cross  Abstract: Memorization in large language models is measured through a zoo of definitions whose formal relations are unknown, and differential privacy (DP) is treated as a proxy against all of them at once. We pin down the exact DP constant for the two that carry the practical weight, counterfactual memorization and adaptive extraction, and show that they do not control each other. Under $f$-DP, every adaptive extraction protocol with list budget $m$ succeeds with probability at most $1-f(\kappa)$ for the oblivious baseline $\kappa$, and the bound is tight on a dense set of baselines: DP uniformly controls extraction exactly up to a threshold in how well the secret can be guessed a priori. Min-entropy certifies that baseline distribution-free, since $H_\infty\ge\epsilon\log_2 e+\log_2(m/\tau)$ holds extraction below a risk level $\tau\le1/2$ under pure $\epsilon$-DP for every prior, and is exact on uniform priors. On the memorization side, $f$-DP
    
[^59]: 为什么它没有检查？两个配备工具的语言模型中缺乏证据支持的最终断言及其修复

    Why Didn't It Check? Unsupported Final Claims and Their Repair in Two Tool-Equipped Language Models

    [https://arxiv.org/abs/2608.27768](https://arxiv.org/abs/2608.27768)

    该论文将工具增强语言模型做出无证据支持的最终断言这一失败现象分解为“发生率”和“条件修复率”两个可精确测量的指标，并通过在断言发生状态的精确副本上进行重放、仅改变工具响应中一个字符的对照实验，在 Qwen3-32B 上量化研究了这一问题。

    

    一个拥有工具访问权限的语言模型可能会做出一个缺乏其所见证据支持的最终断言，即使只需一次可用的工具调用就能消除这种不确定性，而且其指令明确禁止假设和猜测。我们将这种失败分解为两个精确定义的量：发生率，即模型自行做出缺乏支持断言的频率，仅根据可见证据和最终断言进行测量，而不使用隐藏的正确答案；以及条件修复率，即当缺失的证据被提供时，那些自然发生的缺乏支持断言被修复的频率。在一个固定的 Qwen3-32B 设置上，针对 256 个新提示模板的 512 次首次响应中，有 33 次以一个缺乏支持的既定事实断言结束。我们从断言发生时状态的精确副本重放每个案例；在每次匹配的重放中，替代的工具响应具有相同的结构和长度，仅在一个字符的响应内容上有所不同……

    arXiv:2608.27768v1 Announce Type: cross  Abstract: A language model with access to tools can commit to a final claim unsupported by the evidence it has seen, even when a single available tool call would resolve the uncertainty and its instructions explicitly forbid assumptions and guesses. We separate this failure into two precisely defined quantities: occurrence, how often the model makes an unsupported claim on its own, measured from the visible evidence and final claim without using the hidden correct answer; and conditional repair, how often those same naturally occurring unsupported claims are repaired when the missing evidence is supplied. On one fixed Qwen3-32B setup, 33 of 512 first responses to 256 new prompt templates ended with an unsupported established claim. We replayed each case from an exact copy of the state in which the claim occurred; within each matched replay, the alternative tool responses had the same structure and length and differed only in a one-character resp
    
[^60]: 面向持续学习的快速权重注意力

    Fast Weight Attention for Continual Learning

    [https://arxiv.org/abs/2608.27763](https://arxiv.org/abs/2608.27763)

    该论文在“写后读”自回归语义下将快速权重记忆与状态空间模型的状态转移统一视为在线学习规则，并推导出面向持续学习前缀预测的归一化一阶更新家族（Falcon 系列回归与内积变体）。

    

    循环快速权重记忆与选择性状态空间模型将不断增长的上下文压缩进固定大小的循环状态中，从而使状态转移成为一种在线学习规则。我们在“写后读”自回归语义下研究这一规则。对于本文所考虑的前缀预测目标，在第 $t$ 步揭示的局部快速记忆样本是前缀对齐对 $(\mathbf{x}_t,\mathbf{y}_t)=(\phi(\mathbf{k}_{t-1}),\mathbf{v}_t)$；常见的同一步关联 $(\phi(\mathbf{k}_t),\mathbf{v}_t)$ 虽然仍满足因果性，但优化的是另一种内部目标。我们为平方误差回归和负内积目标推导了归一化的一阶更新规则：回归家族包括 Falcon-1（标量 NLMS 更新）、Falcon-2（其按列扩展）以及 Falcon-3（滑动窗口小批量更新）；Falcon-1A/Falcon-2A/Falcon-3A 则是相应的内积变体。我们提供了循环的、带掩码的……

    arXiv:2608.27763v1 Announce Type: cross  Abstract: Recurrent fast-weight memories and selective state-space models compress an expanding context into a fixed-size recurrent state, making the state transition an online learning rule. We study this rule under read-after-write autoregressive semantics. For the prefix-prediction objective considered here, the local fast-memory example revealed at step $t$ is the prefix-aligned pair $(\mathbf{x}_t,\mathbf{y}_t)=(\phi(\mathbf{k}_{t-1}),\mathbf{v}_t)$. The common same-step association $(\phi(\mathbf{k}_t),\mathbf{v}_t)$ remains causal, but optimizes a different internal objective. We derive normalized first-order updates for squared-error regression and negative inner-product objectives. The regression family comprises Falcon-1 (a scalar NLMS update), Falcon-2 (its per-column extension), and Falcon-3 (a sliding-window mini-batch update); Falcon-1A/Falcon-2A/Falcon-3A are the corresponding inner-product variants. We provide recurrent, masked-p
    
[^61]: 信息反局部性与LLM中的局部性偏差

    Informational Antilocality and the Locality Bias in LLMs

    [https://arxiv.org/abs/2608.27760](https://arxiv.org/abs/2608.27760)

    LLM最终能够学会反局部语言并达到相当的损失水平，但在反局部性更强的语言上收敛更慢，表明局部性偏差体现在学习速度上而非学习能力上。

    

    我们研究了基于Transformer的语言模型（LLM）学习我们称之为k-反局部语言的能力，即在任何k个连续符号的跨度上均不存在互信息的语言。我们构建了反局部性程度递增的此类语言，发现在其上训练的LLM无论反局部性程度如何都能达到相当的交叉熵损失，但在反局部性更强的语言上收敛速度更慢。我们的发现支持了非局部依赖更难学习的观点，但这一偏差的证据来自学习速度而非学习成败。

    arXiv:2608.27760v1 Announce Type: new  Abstract: We consider the ability of transformer-based language models (LLMs) to learn what we call k-antilocal languages, i.e., languages that have no mutual information across any span of $k$ contiguous symbols. We construct such languages with increasing $k$, finding that LLMs trained on them achieve comparable cross-entropy loss regardless of antilocality, but converge more slowly on more antilocal languages. Our findings support the idea that non-local dependencies are more difficult to learn, but the evidence for this bias comes from learning speed rather than learning success.
    
[^62]: 承重语境：用于评估语言推理中语境依赖性的问题损伤分数

    Load-Bearing Context: The Question Damage Score for Evaluating Context Reliance in Linguistic Reasoning

    [https://arxiv.org/abs/2608.27756](https://arxiv.org/abs/2608.27756)

    该论文提出“问题损伤分数”诊断框架，通过从语言学奥赛谜题中随机或靶向删除单个上下文示例，量化大语言模型答题对上下文的依赖程度，以区分模型是依赖上下文还是先验知识。

    

    arXiv:2608.27756v1 公告类型：新论文 摘要：确定大型语言模型的答案是源自上下文还是先验知识，仍然是一个根本性挑战。自包含的语言学奥林匹克竞赛谜题提供了一个受控环境，其中所有答案完全源自专家设计的上下文示例，无需外部知识。删除单个上下文示例可能会消除特定问题所需的信息，同时保持谜题的其余部分不变。我们利用这一特性引入了一个用于分析单个上下文示例的诊断框架。基于53个英国语言学奥林匹克竞赛谜题，我们通过删除单个上下文示例生成两种修改变体：(1) 均匀随机删除，以及 (2) 靶向删除（受纠错码启发），用于删除唯一携带必要信息的结构性承重示例。我们使用问题损伤分数对这种影响进行形式化，从而将谜题分类为脆弱型或鲁棒型。我们对三个前沿大语言模型进行了评估……

    arXiv:2608.27756v1 Announce Type: new  Abstract: Determining whether large language models derive answers from context or prior knowledge remains a fundamental challenge. Self-contained linguistic olympiad puzzles provide a controlled setting where all answers derive solely from expert-designed context examples without external knowledge. Removing individual context examples can eliminate information needed for specific questions while leaving the rest of the puzzle unchanged. We leverage this to introduce a diagnostic framework for analyzing individual context examples. Using 53 UK Linguistics Olympiad puzzles, we generate two modified variants by deleting a single context example: (1) uniform random deletion, and (2) targeted deletion (inspired by error-correcting codes) to remove a structurally load-bearing example uniquely carrying necessary information. We formalize this impact using a Question Damage Score to classify puzzles as fragile or robust. Evaluating three frontier LLMs u
    
[^63]: 调用来自模型内部：研究基于探针的大语言模型工具调用错误检测

    The Calls are Coming from Inside the Model: Investigating Probe-based Detection of Tool-Calling Errors in LLMs

    [https://arxiv.org/abs/2608.27750](https://arxiv.org/abs/2608.27750)

    本研究提出利用线性探针读取大语言模型隐藏状态来检测工具调用错误，在18个模型上验证了该方法能有效捕获包括参数值错误在内的各类调用错误，且检测效果受模型大小、探针层级和后训练类型的影响。

    

    大语言模型（LLM）的隐藏状态被认为包含与模型知识和行为相关的丰富信息，而这些信息仅通过检查输入和输出很难提取。随着基于LLM的系统越来越多地与外部世界交互，一个值得关注的问题是检测工具的错误或不当使用。基于此，我们研究了使用线性探针检测错误工具调用的有效性，并在Berkeley Function Calling Leaderboard（伯克利函数调用排行榜）上评估的18个工具调用LLM中测量了探针的效力。总体而言，我们发现探针是捕获各种不同工具调用错误的有效手段，包括由于使用了值错误但类型正确的参数而产生的错误，这类错误可能不会被标准的日志框架记录下来。成功的重要因素包括模型大小、探针所在的层以及模型的后训练类型。我们还表明探针能够进行泛化……

    arXiv:2608.27750v1 Announce Type: cross  Abstract: The hidden states of large language models (LLMs) are known to capture rich information relating to model knowledge and behavior that can be hard to extract from examination of input and output alone. As LLM-based systems increasingly interface with the external world, one area of concern is detecting incorrect or improper use of tools. Motivated by this, we study the effectiveness of using linear probes to detect incorrect tool-calls, measuring probe efficacy across 18 tool-calling LLMs evaluated on the Berkeley Function Calling Leaderboard. Overall, we find that probing is an effective means to catch a range of different tool-calling errors, including errors arising from using an argument that has the wrong value but the correct type, which might not be recorded by standard logging frameworks. Important factors in success include model size, probing layer, and model post-training type. We also show that probes are capable of generali
    
[^64]: 《噪声底之下：小模型知识蒸馏中的双峰种子坍塌与独特失败模式》

    Below the Noise Floor: Bimodal Seed Collapse and Distinct Failure Modes in Small-Model Knowledge Distillation

    [https://arxiv.org/abs/2608.27729](https://arxiv.org/abs/2608.27729)

    该研究通过多种子实验发现，小模型知识蒸馏的单种子报告会掩盖高达48.7个百分点的种子方差，部分KD变体还会出现双峰坍塌且失败模式各不相同，因此任何低于五个百分点的蒸馏收益声明都不可信。

    

    函数路由——即根据自然语言请求从固定目录中选择正确的API调用——是一个部署问题，在此场景下小型学生模型颇具吸引力，但知识蒸馏的收益通常仅以单种子方式报告，而在这种规模下种子方差是未知的。在一个包含740个实例的医疗API路由任务上，使用1.5B的Qwen学生模型和20B的教师模型，我们将八种知识蒸馏（KD）变体与监督交叉熵进行比较，并对关键配置使用三到六个随机种子。我们发现：(i) 单种子标准差范围从2.8到48.7个百分点，足以吞没所有低于五个百分点的KD收益声明；(ii) 七种KD变体中有三种表现出双峰坍塌，即三到五个种子中至少有一个准确率低于55%，而其他种子训练正常，另有第四种变体表现出升高的方差；(iii) 坍塌具有截然不同的模式——ce_kd和ce_paraphrase表现为错误函数选择，而另一种则是此前未被记录过的输出截断现象（摘要在此处被截断）。

    arXiv:2608.27729v1 Announce Type: new  Abstract: Function routing -- selecting the correct API call from a fixed catalog given a natural-language request -- is a deployment problem where small students are attractive but knowledge distillation gains are typically reported single-seed, at scales where seed variance is unknown. On a 740-instance healthcare API routing task with a 1.5B Qwen student and a 20B teacher, we compare eight KD variants against supervised cross-entropy, using three to six seeds for key configurations. We find: (i) per-seed standard deviation ranges from 2.8 to 48.7 percentage points, swallowing every claimed KD gain below five points; (ii) three of seven KD variants exhibit bimodal collapse, with at least one in three to five seeds falling below 55% accuracy while the others train normally, and a fourth showing elevated variance; (iii) collapse has distinct modes -- wrong-function selection for ce_kd and ce_paraphrase, and a previously undocumented output-truncat
    
[^65]: 先让它能玩，再让它玩得好：面向小型对话游戏智能体的分阶段交互学习

    First Make It Playable, Then Make It Good: Staged Interaction Learning for Small Dialogue-Game Agents

    [https://arxiv.org/abs/2608.27672](https://arxiv.org/abs/2608.27672)

    提出20亿参数模型Qwen-GuidePlay-2B，采用“先模仿完整成功轨迹保证可玩性、再通过加权轮次级和教师引导SFT提升决策能力”的分阶段训练策略，在Playpen对话游戏挑战中取得第二高的clemscore增量（较基座模型提升约36分）。

    

    我们提出了Qwen-GuidePlay-2B，一个用于对话游戏交互的20亿参数语言模型。我们通过三个步骤对Qwen3.5-2B进行微调：a) 仅在Playpen的成功游戏轨迹上进行SFT（监督微调），b) 加权轮次级SFT，c) 教师引导的SFT。教师模型（即一个更大的模型）仅用于修复格式和评估示例，而不创建新的黄金动作。我们的最终模型在公开的Playpen验证集上获得了57.12的clemscore和42.68的statscore。在官方发布的挑战结果中，我们的模型在所有提交系统中获得了第二高的Playpen clemscore增量（比其基础模型高约+36分）。我们的发现表明，模仿完整轨迹有助于实现可玩性，而轮次级和教师引导的训练通常能改善决策质量并提高整体分数。重放修复和困难样本挖掘等程序繁重的替代方法没有带来帮助，这表……（摘要原文在此处截断）

    arXiv:2608.27672v1 Announce Type: new  Abstract: We present Qwen-GuidePlay-2B, a 2B-parameter language model for dialogue-game interaction. We fine-tune Qwen3.5-2B using three steps: a) SFT on only successful game trajectories from Playpen, b) weighted turn-level SFT, and c) teacher-guided SFT. The teacher model (which is a larger model) is only used to fix formatting and evaluate examples, but does not create new gold actions. Our final model scores 57.12 clemscore and 42.68 statscore on the public Playpen validation. In the officially released challenge results, our model obtains the second-highest Playpen clemscore delta among submitted systems (which is approximately +36 over its base model). Our findings suggest that imitating full trajectories helps with playability, while turn-level and teacher-guided training usually improve decision-making and increase the overall score. Alternative procedurally heavy approaches like replay-repair and hard-example mining did not help, which su
    
[^66]: 基于子句单元的具有顺序鲁棒检测的语义水印

    Semantic Watermarking with Order-Robust Detection over Sub-sentence Units

    [https://arxiv.org/abs/2608.27666](https://arxiv.org/abs/2608.27666)

    该论文提出了自适应嵌入位移攻击（EDA），在黑盒设置下通过改写、重排和重新分段以单一目标最大化嵌入位移，无需访问生成器或密钥即可在四种语义水印方案上成功移除32.6%至47.9%文档的水印，是测试攻击中最有效的。

    

    arXiv:2608.27666v1 公告类型：cross 摘要：语义水印将水印标记与句子含义而非词元选择相绑定，有望对保持内容不变的编辑具有鲁棒性。然而，检测器只能观察到攻击者提供的文本，攻击者可以在不损失内容的前提下对文本进行改写、重排或重新分段以逃避检测。改写、重排和重新分段都会导致嵌入位移：检测时测试的嵌入与水印嵌入阶段所选择的嵌入不同，因此可能丢失水印标记。我们提出的自适应嵌入位移攻击（EDA）在使嵌入位移最大化的单一目标下同时容纳这三种编辑方式。它仅使用公开的改写器和替代编码器，无需访问提供方的生成器或密钥。在5%的假阳性率（FPR）和90%的内容保持阈值下，EDA在四种方案上成功移除了32.6%至47.9%文档中的水印标记，是所测试攻击中效果最强的。因此，EDA可用于评估

    arXiv:2608.27666v1 Announce Type: cross  Abstract: Semantic watermarks tie the mark to sentence meaning rather than token choices, promising robustness to content-preserving edits. However, the detector only observes attacker-supplied text, which can be reworded, reordered, or resegmented to evade detection without content loss. Rewording, reordering, and resegmentation all cause embedding displacement: detection tests embeddings different from those selected during watermarking and can therefore lose the mark. Our adaptive embedding displacement attack (EDA) admits all three edits under a single objective that maximizes this displacement. It uses a public paraphraser and surrogate encoder without access to the provider's generator or secret key. At a 5% false-positive rate (FPR) and content-preservation threshold $\bar{q}=90\%$, EDA successfully removes the mark on between 32.6% and 47.9% of documents across four schemes, the highest among the tested attacks. Therefore, EDA evaluates 
    
[^67]: 先知后答：解码语言模型以实现可靠的检索增强生成

    Knowing Before Answering: Decoding Language Models for Reliable RAG

    [https://arxiv.org/abs/2608.27661](https://arxiv.org/abs/2608.27661)

    该论文提出利用语言模型的内部信号（隐藏层激活与注意力特征）训练轻量级线性分类器，在作答前判断RAG检索到的信息是充分、不充分还是冲突，从而提升生成系统的可靠性。

    

    在检索增强生成（RAG）中，检索到的信息可能不足以回答问题，或包含相互冲突的内容。系统不仅需要知道何时作答，还必须能够识别出RAG所提供文档不充分或存在冲突的情况。这可以被构建为一个三分类问题：我们利用模型的内部信号来判断输入中提供的信息应被归类为充分、不充分还是冲突。我们构建了一个受控的基准数据集，模拟使用虚构信息的RAG场景，并将每个实例标注为可回答、信息不充分或信息冲突三类。我们以隐藏层激活值和基于注意力的特征作为输入，训练一个轻量级线性模型来区分这三个类别。在涵盖不同架构和多种规模的16个语言模型上，我们基于特征的路由器……

    arXiv:2608.27661v1 Announce Type: new  Abstract: In Retrieval-Augmented Generation (RAG), retrieval may provide insufficient or conflicting information needed to answer a question. The system should not only know when to answer but also be able to identify cases in which the documents provided in RAG are insufficient or contain conflicting information. This can be framed as a three-way classification problem, where we use the model's internal signals to determine whether the provided information in the input can be classified as sufficient, insufficient, or conflicting. We create a controlled benchmark dataset that replicates a RAG setup with fictitious information and labels each instance as answerable, insufficient, or conflicting. We use hidden activations and attention-derived features as inputs to train a lightweight linear model to distinguish among the three classes. Across 16 language models spanning different architectures and a range of model sizes, our feature-based router c
    
[^68]: 当分词器失效时：面向低资源语言零样本迁移的字节级分块方法

    When Tokenizers Fail: Byte-Level Chunking for Zero-Shot Transfer to Low-Resource Languages

    [https://arxiv.org/abs/2608.27658](https://arxiv.org/abs/2608.27658)

    本文提出一种无需大量训练的分层字节级网络框架，通过从冻结基础模型的子词表示初始化字节嵌入并使用块对齐损失，实现了向低资源语言的零样本迁移。

    

    子词分词通过将主导语言的频率模式强加于共享文字的变体语言上，阻碍了低资源语言的处理。字节级模型通过处理原始UTF-8字符绕过了这一问题，但在非拉丁文字的词级任务中会产生粒度不匹配的问题。分层字节级架构通过将字节分组为与词对齐的块来解决这种不匹配。然而，这些架构需要大量训练数据，并且在与冻结的基于子词的语言模型配对时会遭受表征不对齐的问题。在本文中，我们提出了一种改进的分层网络框架，无需大量训练即可弥合这种模态鸿沟。我们的方法直接从冻结基础模型的子词表示初始化字节嵌入。我们应用块对齐损失将动态分组的字节块投影到预计算的子词目标上，并交织轻量级的词性（标注信息）……

    arXiv:2608.27658v1 Announce Type: new  Abstract: Subword tokenization hinders low-resource language processing by imposing frequency patterns from dominant languages onto script-sharing variants. Byte-level models bypass this issue by processing raw UTF-8 characters, yet they create a granularity mismatch for word-level tasks in non-Latin scripts. Hierarchical byte-level architectures address this mismatch by grouping bytes into word-aligned chunks. However, these architectures require massive training data and suffer from representational misalignment when paired with frozen subword-based language models. In this paper, we propose an adapted hierarchical network framework that bridges this modality gap without extensive training. Our method initializes byte embeddings directly from the subword representations of a frozen base model. We apply a chunk alignment loss to project dynamically grouped byte chunks toward precomputed subword targets, and interleave lightweight part-of-speech (
    
[^69]: 扩散语言模型的轨迹级推测解码

    Trajectory-Level Speculative Decoding for Diffusion Language Models

    [https://arxiv.org/abs/2608.27514](https://arxiv.org/abs/2608.27514)

    提出了一种针对扩散语言模型的轨迹级推测解码框架，通过置信度分层树探索构建草稿去噪轨迹、利用双向注意力掩码进行分块并行验证，并引入跨块前瞻的块间推测机制以突破单token生成的吞吐量瓶颈。

    

    基于扩散的语言模型通过迭代去噪实现并行token生成，但现有解码策略在低置信度时会退化为单token生成，严重限制了吞吐量。与自回归模型中推测解码按固定从左到右顺序作用于token序列不同，dLLM需要对去噪轨迹进行推测——即具有明确位置和去掩码顺序的多token更新序列。我们开发了一个轨迹级推测框架，通过置信度分层树探索构建草稿去噪轨迹，并通过带双向注意力掩码的分块并行评估对其进行验证。我们的方法进一步引入了块间推测，利用扩散模型的双向结构执行跨块前瞻。我们正式刻画了该方法何时是精确的，并将轨迹漂移识别为根本挑战。

    arXiv:2608.27514v1 Announce Type: new  Abstract: Diffusion-based language models (dLLMs) enable parallel token generation through iterative denoising, but existing decoding strategies collapse to single-token generation under low confidence, severely limiting throughput. Unlike autoregressive models where speculative decoding operates on token sequences in a fixed left-to-right order, dLLMs require speculating over denoising trajectories-sequences of multi-token updates with explicit positions and unmasking orders. We develop a trajectory-level speculative framework that constructs draft denoising trajectories via confidence-stratified tree exploration and verifies them through blockwise parallel evaluation with bidirectional attention masking. Our method further introduces inter-block speculation, exploiting diffusion models' bidirectional structure to perform cross-block lookahead. We formally characterize when this approach is exact and identify trajectory drift as the fundamental c
    
[^70]: 语言模型中的量化触发后门：跨量化器可迁移性与验证—部署鸿沟

    Quantization-Triggered Backdoors in Language Models: Cross-Quantizer Transferability and the Validation--Deployment Gap

    [https://arxiv.org/abs/2608.27512](https://arxiv.org/abs/2608.27512)

    该论文提出量化行为等价类（QBEC）理论，证明源精度下的模型验证无法保证量化部署后的行为等价，并构建三阶段对抗微调框架，使后门仅在模型被INT8或4比特量化部署时才被触发激活，揭示了量化流程中的安全隐患。

    

    arXiv:2608.27512v1 公告类型：交叉 摘要：训练后量化通常被视为大语言模型边缘部署中一种语义中立的优化手段。当全精度源模型检查点经过评估后，量化在下游流程中被应用而未进行同等的重新评估，这种工作流程造成了一种结构性的“验证—部署鸿沟”：由于量化是参数空间上的多对一映射，源精度下的安全认证并不能保证部署配置中的行为等价性。我们通过量化行为等价类（QBECs）对这一鸿沟进行了形式化定义，并证明属于同一QBEC并不意味着行为等价，从而为量化触发的后门攻击提供了理论基础。基于一个三阶段对抗微调框架，我们将潜在的恶意载荷嵌入到能够通过评估中使用的源精度检查的模型中，而这些模型在INT8或4比特量化后会激活针对性的对抗行为（摘要在此处被截断）。

    arXiv:2608.27512v1 Announce Type: cross  Abstract: Post-training quantization is often treated as a semantically neutral optimization for edge deployment of Large Language Models. When a full-precision source checkpoint is evaluated and quantization is applied downstream without equivalent re-evaluation, this workflow creates a structural validation--deployment gap: because quantization is a many-to-one mapping over parameter space, source-precision certification does not guarantee behavioral equivalence in the deployed configuration. We formalize this gap through Quantization Behavioral Equivalence Classes (QBECs) and prove that QBEC membership does not imply behavioral equivalence, providing a theoretical basis for quantization-triggered backdoor attacks. Building on a three-stage adversarial fine-tuning framework, we embed latent malicious payloads into models that satisfy the source-precision checks used in our evaluation, yet activate targeted adversarial behavior upon INT8 or 4-b
    
[^71]: 线性探针是如何涌现的？一种基于概念定向归因的电路追踪框架

    How Do Linear Probes Emerge? A Circuit-Tracing Framework with Concept-Targeted Attribution

    [https://arxiv.org/abs/2608.27510](https://arxiv.org/abs/2608.27510)

    该论文提出概念定向归因（CTA）框架，通过针对线性探针方向训练归因图，首次将线性探针的性能与模型内部可解释的电路结构联系起来，不仅能判断探针是否有效，还能揭示是哪些内部计算使探针起作用。

    

    转码器归因图通常被训练用于解释模型为何对特定的下一个词元分配高概率。我们提出了概念定向归因（Concept-Targeted Attribution, CTA），该方法改为针对线性探针方向来训练归因图。因此，CTA能够生成探针特定的电路，解释为什么内部概念表示会在提示中产生，而与该概念是否在生成的词元中被表达无关。利用跨层转码器，我们证明这些以探针为目标的图包含预测性结构：图级特征能够预测四个广泛研究的概念类别上的探针准确率（ρ = 0.91，R² = 0.84），而局部特征则能识别出驱动逐提示分类的稀疏组件。这将探针性能与可解释的电路结构联系起来，使我们不仅能询问探针是否有效，还能探究是哪些内部计算使其有效。因果消融实验进一步表明……

    arXiv:2608.27510v1 Announce Type: new  Abstract: Transcoder attribution graphs are usually trained to explain why a model assigns high probability to a particular next token. We introduce Concept-Targeted Attribution (CTA), which instead trains attribution graphs with respect to a linear probe direction. CTA therefore yields probe-specific circuits that explain why an internal concept representation arises in a prompt, independently of whether it is expressed in the generated token. Using Cross-Layer Transcoders, we show that these probe-targeted graphs contain predictive structure: graph-level features predict probe accuracy across four widely studied concept categories ($\rho = 0.91$, $R^2 = 0.84$), while local features identify the sparse components driving per-prompt classification. This connects probe performance to interpretable circuit structure, allowing us to ask not only whether a probe works, but which internal computations make it work. Causal ablations further show that pr
    
[^72]: 基于量规引导的强化学习在语言模型中的研究综述

    A Survey on Rubric-Guided Reinforcement Learning for Language Models

    [https://arxiv.org/abs/2608.27505](https://arxiv.org/abs/2608.27505)

    本综述提出一个贝叶斯统一框架——将宪法视为评价准则的先验分布、量规视为其条件化实例，并沿先验—后验轴系统梳理了量规引导强化学习的分类体系，用结构化、可解释的评价准则取代标量奖励，从而改进大语言模型的对齐效果。

    

    基于人类反馈的强化学习（RLHF）已成为将大型语言模型（LLM）与人类偏好对齐的主流范式。然而，传统RLHF依赖于标量奖励信号，这类信号缺乏可解释性，且无法刻画回答质量的多面性。量规引导的强化学习通过引入结构化、可解释的评价准则（即量规），将其作为奖励设计、反馈生成与策略优化的核心骨架，从而克服了上述局限。在本综述中，我们提出了一个贝叶斯框架：将“宪法”定义为评价准则上的先验分布P(R)，将量规定义为条件化的具体实例化R_x ~ P(R|x)。在这一统一视角下，我们沿先验—后验轴对量规引导的强化学习进行了系统分类，涵盖宪法式AI、实例化量规、过程级监督、自进化量规，及其在智能体与多模态场景中的扩展。

    arXiv:2608.27505v1 Announce Type: new  Abstract: Reinforcement learning from human feedback (RLHF) has become the dominant paradigm for aligning large language models (LLMs) with human preferences. However, traditional RLHF relies on scalar reward signals that lack interpretability and fail to capture the multifaceted nature of response quality. Rubric-guided reinforcement learning addresses these limitations by introducing structured, interpretable evaluation criteria, or rubrics, as the backbone of reward design, feedback generation, and policy optimization. In this survey, we introduce a Bayesian framework that defines constitutions as prior distributions $P(R)$ over evaluation criteria and rubrics as conditional instantiations $R_x \sim P(R|x)$. Under this unified view, we present a taxonomy of rubric-guided RL along the prior-posterior axis, covering constitutional AI, instance-specific rubrics, process-level supervision, self-evolving rubrics, and their agentic and multimodal ext
    
[^73]: INSPIRE：一种用于示例驱动数学推理的先内化后改进方法

    INSPIRE: An Internalize-Then-Improve Approach for Example-Driven Mathematical Reasoning

    [https://arxiv.org/abs/2608.27501](https://arxiv.org/abs/2608.27501)

    提出INSPIRE方法，采用先内化参考示例、再逐步改进的策略，增强大语言模型基于示例的数学推理能力（如构造反例检验定理边界），超越仅优化最终答案正确性的传统方法。

    

    数学推理在大语言模型（LLMs）中取得了快速进展，然而现有方法主要针对最终答案的正确性进行优化，这引发了一个问题：模型是真正内化了数学概念，还是仅仅记忆了解题模式。在人类数学教育中，基于示例的推理（如构造反例来检验定理边界）反映了深层的概念理解，但这种能力在当前的大语言模型中仍然发展不足。通过偏好优化来增强这种能力面临两个关键挑战：（1）模型有限的基于示例的推理能力使得构建有效的偏好对本身就十分困难；（2）能力的获得是渐进式的，模型必须先学会采用这种策略，然后才能学会正确地应用它。因此，我们提出了INSPIRE，一种先内化后改进的方法，结合了参考引导的学生内化（RGSI）……

    arXiv:2608.27501v1 Announce Type: new  Abstract: Mathematical reasoning has seen rapid progress in large language models (LLMs), yet existing methods optimize predominantly for final-answer correctness, raising the question whether models truly internalize mathematical concepts or merely memorize solution patterns. In human mathematics education, example-based reasoning such as constructing counterexamples to test theorem boundaries reflects deep conceptual understanding, but remains underdeveloped in current LLMs. Enhancing this capability through preference optimization presents two key challenges: (1) the model's limited example-based reasoning ability makes constructing effective preference pairs inherently difficult; and (2) capability acquisition is progressive, as the model must first learn to adopt this strategy before learning to apply it correctly. Therefore we propose INSPIRE, an Internalize-Then-Improve approach combining Reference-Guided Student Internalization (RGSI), whi
    
[^74]: XHotpotQA：面向多跳问答中跨语言知识组合的基准测试

    XHotpotQA: A Benchmark for Cross-Lingual Knowledge Composition in Multi-Hop Question Answering

    [https://arxiv.org/abs/2608.27481](https://arxiv.org/abs/2608.27481)

    该论文提出XHotpotQA基准，通过将多跳问答实例建模为带显式语言标注的证据依赖图，系统评测了模型在混合语言证据下跨语言组合知识的能力，并发现完全的语言不匹配会导致模型性能显著下降。

    

    知识密集型的多跳问答要求系统选择证据并组合相互依赖的事实，然而现有多语言基准通常将整个样例翻译为单一语言，这掩盖了推理链内部语言边界处的失败。我们提出XHotpotQA，一个针对混合语言证据上跨语言知识组合的受控基准。每个实例被建模为一张证据依赖图，其中问题、桥接证据、含答案证据和干扰项均具有显式的语言标注。该经过审核的资源包含15,661个训练实例和7,405个验证实例，并提供句子级支持监督及干扰项。在验证集中，99.81%的条目跨越了问题与黄金证据之间的语言界面，95.60%的条目使用不同语言的黄金段落。在三个阅读器模型上，完全的问题-证据语言不匹配与Unicode感知指标降低10.25至15.79相关。

    arXiv:2608.27481v1 Announce Type: new  Abstract: Knowledge-intensive multi-hop question answering requires systems to select evidence and compose dependent facts, yet multilingual benchmarks usually translate an entire example into one language. This hides failures at language boundaries inside the reasoning chain. We introduce XHotpotQA, a controlled benchmark for cross-lingual knowledge composition over mixed-language evidence. Each instance is modeled as an evidence-dependency graph whose question, bridge evidence, answer-bearing evidence, and distractors have explicit language assignments. The audited resource contains 15,661 training and 7,405 validation instances, with sentence-level support supervision and supplied distractors. In validation, 99.81% of items cross the question-to-gold-evidence language interface and 95.60% use gold paragraphs in different languages. Across three reader artifacts, full question-evidence mismatch is associated with 10.25 to 15.79 lower Unicode-awa
    
[^75]: 检索关系，检测谬误：一种用于政治辩论分析的RAG方法

    Retrieving Relations, Detecting Fallacies: A RAG Approach to Political Debate Analysis

    [https://arxiv.org/abs/2608.27471](https://arxiv.org/abs/2608.27471)

    该论文提出一种引导式检索增强生成方法，利用论证间的支持与攻击关系动态引导检索过程，从而突破静态特征的局限，提升政治辩论中谬误检测与分类的性能。

    

    谬误是指采用无效推理的论证，在诸如高风险政治辩论这类影响公众舆论形成的敏感场景中，对谬误进行自动检测至关重要。识别一个谬误论证需要超越其表面文本的上下文知识，这既包括与所讨论主题相关的世界知识，也包括论证性话语中各论证之间关系的相关知识。先前关于谬误分析的研究表明，论证性话语结构能够有效提升分类性能。然而，这种结构通常仅被编码为静态的分类器特征，限制了其灵活性。在保留这一思路的同时针对该局限加以改进，我们提出了一种引导式检索增强方法，用于谬误的检测与分类，该方法利用论证之间的支持与攻击关系来动态引导

    arXiv:2608.27471v1 Announce Type: cross  Abstract: Fallacies are arguments that employ invalid reasoning, making their automatic detection critical in sensitive contexts such as high-stakes political debates, where public opinion is shaped. Spotting a fallacious argument requires contextual knowledge beyond its pure surface text. This entails world knowledge pertaining to the subject matter under discussion, as well as knowledge of the relationships that exist between arguments within the argumentative discourse. Prior work on fallacy analysis has shown that argumentative discourse structure can beneficially improve classification performance. However, such structure is typically encoded only as static classifier features, limiting its flexibility. Building on this intuition while addressing this limitation, we introduce a guided retrieval-augmented methodology for fallacy detection and classification that leverages argumentative relations of support and attack to dynamically steer the
    
[^76]: 选择而非训练：基于LLM选择的模块化实体消歧的优势

    Select, Don't Train: The Benefits of Modular Entity Disambiguation with LLM-Based Selection

    [https://arxiv.org/abs/2608.27470](https://arxiv.org/abs/2608.27470)

    本文系统比较了在共享LLM选择阶段下不同检索策略的实体消歧效果，提出“选择而非训练”的模块化范式，避免了训练专用检索器的高昂成本与维护负担。

    

    实体消歧（ED）是构建和使用知识图谱的关键任务。最先进的神经方法通常将ED建模为单一任务，尽管它实际上包含两个不同的子问题：检索候选实体和根据上下文选择正确的实体。双编码器模型在共享嵌入空间中同时优化这两个子任务，迫使表示在兼顾高召回率检索与细粒度选择之间进行权衡，并且它们需要经过训练的检索器，当知识图谱发生变化时维护成本高昂。虽然最近的工作已开始将检索器与基于LLM的选择器相结合，但这两个阶段之间的相互作用尚未得到系统性研究。在本文中，我们在共享的基于LLM的选择阶段下，对候选生成的各种检索策略进行了系统性比较，结合了稀疏检索（BM25）、Web KB搜索和最先进的训练密集检索器，以及多个开源和闭源模型。

    arXiv:2608.27470v1 Announce Type: new  Abstract: Entity Disambiguation (ED) is a key task for constructing and using knowledge graphs. State-of-the-art neural approaches commonly model ED as a single task, although it consists of two distinct subproblems: retrieving candidate entities and selecting the correct one given context. Dual-encoder models optimize for both within a shared embedding space, forcing representations to balance high-recall retrieval with fine-grained selection, and they require trained retrievers, which are costly to maintain as knowledge graphs change. While recent work has begun to combine retrievers with LLM-based selectors, the interplay between the two stages has not been studied systematically. In this paper, we present a systematic comparison of retrieval strategies for candidate generation under a shared LLM-based selection stage, combining sparse retrieval (BM25), Web KB search, and a state-of-the-art trained dense retriever with several open- and closed-
    
[^77]: UIC-AIHealth4All在ArchEHR-QA 2026上的方案：面向临床问答的答案优先证据接地方法

    UIC-AIHealth4All at ArchEHR-QA 2026: Answer-First Evidence Grounding for Clinical Question Answering

    [https://arxiv.org/abs/2608.27467](https://arxiv.org/abs/2608.27467)

    该研究提出一种答案优先的流水线，先让模型生成引用具体病历句子的候选答案再分类完整证据集，并结合自洽性投票进行答案-证据对齐，在ArchEHR-QA 2026临床问答任务中分别取得证据识别第三名、答案生成第九名和答案-证据对齐第五名的成绩。

    

    我们介绍了UIC-AIHealth4All团队参加ArchEHR-QA 2026的系统，该共享任务旨在基于电子健康记录进行有据可依的问答。我们参与了子任务2（证据识别）、子任务3（答案生成）和子任务4（答案-证据对齐）。对于子任务2和3，我们提出了一种答案优先的流水线：模型先生成引用具体病历句子的候选答案，再对完整证据集进行分类，从而利用了“在摘要层面判断相关性”与“相对于已生成答案判断相关性”之间的不对称性。对于子任务4，我们在五次独立的模型调用上应用自洽性投票，并根据投票阈值保留对齐链接。我们的流水线在证据识别上排名第三（严格微平均F1为62.90），在答案生成上排名第九（总分31.90），在答案-证据对齐上排名第五（F1为79.81）。对45个文体特征的事后语言学分析显示，模型输出仍比……高出3.2个Flesch-Kincaid年级水平（原文此处截断）。

    arXiv:2608.27467v1 Announce Type: new  Abstract: We describe the UIC-AIHealth4All system for ArchEHR-QA 2026, a shared task on grounded question answering from electronic health records. We participated in Subtasks 2 (evidence identification), 3 (answer generation), and 4 (answer-evidence alignment). For Subtasks 2 and 3, we propose an answer-first pipeline in which the model generates candidate answers citing specific note sentences before classifying the full evidence set, exploiting the asymmetry between judging relevance in the abstract versus relative to a generated answer. For Subtask 4, we apply self-consistency voting over five independent model calls, retaining links by vote threshold. Our pipeline ranked third on evidence identification (Strict Micro F1 62.90), ninth on answer generation (Overall 31.90), and fifth on answer-evidence alignment (F1 79.81). A post-hoc linguistic analysis of 45 stylistic features reveals that model outputs remain 3.2 Flesch-Kincaid grade levels h
    
[^78]: PACE：基于智能体自动化的出版商自适应内容抽取

    PACE: Publisher-Adaptive Content Extraction via Agentic Automation

    [https://arxiv.org/abs/2608.27466](https://arxiv.org/abs/2608.27466)

    PACE是一个智能体自动化框架，训练时利用LLM从代表性页面中分析结构并聚合可复用的抽取模式，推理时将学习到的配置实例化为固定的确定性抽取模板，从而同时实现准确、低成本且可扩展的出版商自适应网页内容抽取。

    

    网页内容抽取对于构建可靠的LLM数据管道至关重要，然而现有方法往往难以同时满足准确性、可扩展性和适应性。通用抽取器虽然适用范围广，但在出版商特定的页面布局以及更丰富的抽取目标（如元数据、图像和表格）上往往表现脆弱。直接基于LLM的抽取方式虽然提供了更大的灵活性，但在大规模应用时会带来高昂的成本和延迟；而人工为特定出版商设计的解析器虽然可以达到很高的准确性，却需要大量的人力来构建和维护。我们提出了PACE，这是一个智能体框架，能够从代表性页面和用户需求中学习出版商特定的抽取配置。在训练阶段，PACE利用LLM分析页面结构并聚合可复用的抽取模式；在推理阶段，学习到的配置会实例化一个固定的确定性抽取器模板，从而实现可扩展的内容抽取。

    arXiv:2608.27466v1 Announce Type: new  Abstract: Web content extraction is essential for reliable LLM data pipelines, yet existing methods often struggle to jointly satisfy accuracy, scalability, and adaptability. General-purpose extractors can be applied broadly, but they are often brittle on publisher-specific layouts and richer extraction targets such as metadata, images, and tables. Direct LLM-based extraction offers greater flexibility, but incurs substantial cost and latency at scale, while manually engineered publisher-specific parsers can achieve high accuracy but require substantial human effort to build and maintain.   We introduce PACE, an agentic framework for learning publisher-specific extraction configurations from representative pages and user requirements. During training, PACE uses LLMs to analyze page structure and aggregate reusable extraction patterns. At inference time, the learned configurations instantiate a fixed deterministic extractor template, enabling scala
    
[^79]: 情感语境对大语言模型支持过早决策的影响：六种商业模型的情感脆弱性比较

    The Effect of Emotional Context on Large Language Models' Endorsement of Premature Decisions: Comparing Emotional Vulnerability Across Six Commercial Models

    [https://arxiv.org/abs/2608.27465](https://arxiv.org/abs/2608.27465)

    本研究通过对六种商业大语言模型在324段对话中的实验发现，用户表达痛苦等负面情绪会显著增加模型对过早决策（如基于薄弱证据辞职）的支持与鼓励，且该效应独立于对话长度，揭示了LLM在情感语境下普遍存在的安全脆弱性。

    

    随着大语言模型（LLM）越来越多地被用于日常决策建议，模型是否会根据用户的情绪状态改变其建议方向，已成为一个重要的安全问题。我们测试了当用户在拥有相同客观信息的情况下，对一个过早的决策（例如，基于薄弱证据辞去稳定工作）过度自信时，情绪表达是否会增加模型的支持度（即鼓励用户继续进行的倾向）。作为关键对照，我们纳入了一个无情绪的多轮（中性）对话条件，该条件保持事实内容和对话轮数不变，从而将情绪的影响与对话长度的影响分离开来。我们将六种商业模型（来自OpenAI、Anthropic和Google的顶级及中级模型）置于三种场景（职业转换、业务扩张、移民）和三种条件（冷静/中性/痛苦）下，每种条件重复六次，共产生324段对话，并且……

    arXiv:2608.27465v1 Announce Type: new  Abstract: As large language models (LLMs) are increasingly used for everyday decision-making advice, whether a model shifts the direction of its advice according to the user's emotional state has become an important safety problem. We test whether emotional expression increases a model's endorsement (encouragement to proceed) when a user, holding the same objective information, is overconfident about a premature decision (e.g., quitting a stable job on weak evidence). As a key control, we include a no-emotion multi-turn (neutral) condition that holds factual content and the number of conversational turns constant, isolating the effect of emotion from that of conversation length. We exposed six commercial models (top-tier and mid-tier models from OpenAI, Anthropic, and Google) to three scenarios (career change, business expansion, emigration) across three conditions (cold/neutral/distress) with six repetitions each, yielding 324 conversations, and 
    
[^80]: 大锤还是手术刀？一种面向隐性仇恨言论的细粒度自适应框架

    Sledgehammer or Scalpel? A Fine-grained Adaptive Framework for Implicit Hate Speech

    [https://arxiv.org/abs/2608.27462](https://arxiv.org/abs/2608.27462)

    提出了细粒度自适应框架FAID，将隐性仇恨言论划分为浅层、针对性和上下文依赖三类，并针对不同类别采用不同复杂度的检测策略，在提升检测精度的同时降低不必要的计算开销。

    

    与带有明显脏话的显性攻击不同，隐性仇恨言论通过隐喻和上下文暗示将恶意隐藏在看似合规的表达之中，这使得在线内容审核中的检测极具挑战性。虽然现有的基于预训练语言模型（PLM）或大语言模型（LLM）的方法表现良好，但它们通常对所有样本采用单一的推理过程。这忽略了细粒度的语言细微差别，并对较简单的情况造成不必要的计算开销。我们观察到，在线仇恨言论并非单一形态，而是以多种形式呈现。因此，我们定义了三个细粒度类别：浅层、针对性和上下文依赖型。据此，我们提出了细粒度自适应隐性仇恨言论检测框架FAID，这是一个新颖的框架，首先进行细粒度分类，然后针对特定类别进行自适应处理。具体而言，对于具有表面可识别意图的浅层样本，该框架采用轻量级的提示调优进行快速分类。

    arXiv:2608.27462v1 Announce Type: new  Abstract: Unlike explicit attacks with obvious profanity, implicit hate speech hides malice within seemingly compliant expressions through metaphors and contextual hints, making its detection in online content review challenging. While existing PLM- or LLM-based methods perform well, they typically apply a single reasoning process to all samples. This overlooks fine-grained linguistic nuances and causes unnecessary computation for simpler cases. We observe that online hate speech is not monolithic but manifests in varied forms. We therefore define three fine-grained categories: Shallow, Targeted, and Context-Dependent. Accordingly, we propose Fine-grained Adaptive Implicit Hate speech Detection (FAID), a novel framework that first performs fine-grained classification and then adapts to specific categories. Specifically, for Shallow samples with surface-identifiable intents, the framework adopts lightweight prompt-tuning for rapid classification; f
    
[^81]: SciReC：基于自适应交互的多模态多轮关系推理诊断评估

    SciReC: Diagnostic Evaluation of Multimodal, Multi-Turn Relational Reasoning with Adaptive Interaction

    [https://arxiv.org/abs/2608.27461](https://arxiv.org/abs/2608.27461)

    该论文提出了SciReC——一个模型自适应的多模态学术对话基准，以及DMRA缺陷诊断框架，用于系统评估多模态大语言模型在多轮关系推理中的表现，并量化视觉理解、知识展示和记忆回忆等因素对失败案例的贡献。

    

    关系推理需要对概念之间的潜在关系进行感知理解、比较和整合的过程。这种能力包含多个类别，例如类比推理、结构推理和因果关系推理，每种类型都捕捉了高阶理解的不同方面。为了检验多模态大语言模型（MLLM）在这些关系推理任务上的表现，我们开发了 SciReC，一个模型自适应的多模态学术对话基准。由于关系推理过程涉及多种表示和多种因素（视觉理解、知识展示和记忆回忆），我们提出了 DMRA，一种基于缺陷的诊断框架，用于量化这些组成部分的贡献，以识别失败案例的主要原因。Claude 4.6 在总体关系得分上取得了最佳表现，达到 73%，其次是 GPT 5.4，得分为 68%。性能趋势表明

    arXiv:2608.27461v1 Announce Type: new  Abstract: Relational reasoning requires the process of perceptual understanding, comparing, and integrating the underlying relationships between concepts. This ability consists of multiple categories, such as analogical, structural, and cause-effect, each capturing a different aspect of higher-order understanding. To examine the performance of multimodal large language models (MLLM) on these relational inference tasks, we developed SciReC, a model-adaptive multimodal academic dialog benchmark. As the relational reasoning process involves multiple representations and various factors (visual understanding, exhibiting knowledge, and memory recall), we propose DMRA, a deficit-based diagnostic framework that quantifies the contribution of these components to identify the primary cause of unsuccessful cases. Claude 4.6 achieved the best performance on the overall relational score with 73\%, followed by GPT 5.4 with 68\%. Performance trends indicate that
    
[^82]: 基于向量索引输出嵌入加速大语言模型推理

    Accelerating LLM Inference via Vector Index Based Output Embeddings

    [https://arxiv.org/abs/2608.27460](https://arxiv.org/abs/2608.27460)

    本文将大语言模型的输出投影重新表述为基于HNSW向量索引的最大内积搜索，仅检索高分候选词元以替代稠密词表投影，在CPU推理中最高可将解码吞吐量提升82%且不损失生成质量。

    

    大型输出嵌入矩阵在自回归解码过程中会造成显著的内存带宽瓶颈，尤其是对于拥有庞大多语言词表的紧凑型大语言模型。我们将输出投影及随后的top-k词元选择重新表述为对词元嵌入的最大内积搜索，并用基于HNSW的向量索引取代稠密的词表投影。由此得到的输出头仅检索一小部分高分候选词元，并可通过将检索到的logits散射到稀疏的全词表张量中，集成到现有的解码流水线中。在Gemma 3、Llama 3.2和Qwen 3模型的CPU推理实验中，我们的方法显著加速了输出投影，使Gemma 3 270M的端到端批大小为1的解码吞吐量最高提升了82%，同时在AlpacaEval评估下保持了生成质量。这些结果表明，近似检索是稠密输出投影的一种实用替代方案。

    arXiv:2608.27460v1 Announce Type: new  Abstract: Large output embedding matrices create a significant memory bandwidth bottleneck during autoregressive decoding, especially for compact LLMs with large multilingual vocabularies. We reformulate the output projection followed by top-k token selection as a maximum inner product search over token embeddings and replace the dense vocabulary projection with an HNSW-based vector index. The resulting output head retrieves only a small candidate set of high-scoring tokens and can be integrated into existing decoding pipelines by scattering retrieved logits into a sparse full-vocabulary tensor. On CPU inference with Gemma 3, Llama 3.2, and Qwen 3 models, our method substantially accelerates the output projection and improves end-to-end batch-size-one decoding throughput by up to 82% for Gemma 3 270M, while preserving generation quality under AlpacaEval evaluation. These results suggest approximate retrieval is a practical alternative to dense out
    
[^83]: 不同粒度下韩语成分结构的表示与解析

    Representing and Parsing Korean Constituency Structure at Different Levels of Granularity

    [https://arxiv.org/abs/2608.27035](https://arxiv.org/abs/2608.27035)

    本文通过比较三种基于Penn韩语树库的不同粒度表示，系统评估了韩语成分解析在形态复杂eojeol单元下的表示策略与解析性能。

    

    韩语成分解析面临一个表示挑战，因为短语结构树的终端单元并不直接对应简单的表面词。韩语eojeol（语节）是形态复杂的间距单位，现有成分资源在表示eojeol内部形态和非显性元素方面有所不同。本文比较了从Penn韩语树库中派生的三种成分解析表示：Morpheme+XPOS、Eojeol+XPOS和Eojeol+UPOS。我们通过移除空元素、将Penn韩语短语结构与显性eojeol标记对齐、尽可能保留Penn韩语短语标签，并变化终端和前终端层来构建这些表示。然后，我们在共享建模和评估设置下，以自顶向下、中序和自底向上顺序评估了规范的基于非二元转换的成分解析器。所有实验使用黄金终端分割和黄金前终端分割。

    arXiv:2608.27035v1 Announce Type: new  Abstract: Korean constituency parsing raises a representational challenge because the terminal units of a phrase-structure tree do not straightforwardly correspond to simple surface words. Korean eojeols are morphologically complex spacing units, and existing constituency resources differ in how they represent eojeol-internal morphology and non-overt elements. This paper compares three constituency parsing representations derived from the Penn Korean Treebank: Morpheme+XPOS, Eojeol+XPOS, and Eojeol+UPOS. We construct these representations by removing null elements, aligning Penn Korean phrase structure with overt eojeol tokens, preserving Penn Korean phrase labels where possible, and varying the terminal and preterminal layers. We then evaluate canonical non-binary transition-based constituency parsers in top-down, in-order, and bottom-up orders under a shared modeling and evaluation setup. All experiments use gold terminal segmentation and gold p
    
[^84]: 临床诊断大语言模型训练中的知识图谱外科式对齐

    Surgical Alignment in Knowledge Graph Training for Clinical Diagnosis with Large Language Models

    [https://arxiv.org/abs/2608.26587](https://arxiv.org/abs/2608.26587)

    本文提出“外科式对齐”概念，通过梯度干预密度和梯度扭曲指标，发现KL正则化下的知识图谱判断训练能产生稀疏局部更新，优于任务特定SFT的密集更新，从而更有效地将KG知识整合到LLM中用于临床诊断。

    

    生物医学知识图谱（KGs）提供了结构化的医学知识，能够在大语言模型（LLM）的临床诊断应用中支撑其推理，但如何将KG信号整合到LLM中仍是一个开放问题。我们进行了一项系统性研究，涵盖五种KG任务表述、三种训练范式、两个知识图谱和三个基础LLM。在任务层面，所有范式都优于未微调的基线，但具有相当领域内准确度的方法在知识迁移行为上表现出显著差异。我们引入了梯度干预密度（GID）和梯度扭曲（GD）来度量优化器对预训练模型的修改广度。GID和GD共同揭示了一个明确的分界：在KL正则化下的KG判断训练产生稀疏、局部的更新（我们称之为“外科式对齐”），而任务特定的SFT则产生密集更新。一项受控消融实验表明，目标和KL贡献是这一差异的关键因素。

    arXiv:2608.26587v1 Announce Type: new  Abstract: Biomedical knowledge graphs (KGs) offer structured medical knowledge that can ground large language model (LLM) reasoning in clinical diagnosis application, yet how KG signal should be integrated into LLMs remains an open question. We present a systematic study spanning five KG task formulations, three training paradigms, two KGs, and three base LLMs. At the task level, all paradigms improve over the non-finetuned baseline, but methods with comparable in-domain accuracy show substantially different knowledge transfer behavior. We introduce Gradient Intervention Density (GID) and Gradient Distortion (GD) to measure how broadly an optimizer modifies the pretrained model. GID and GD together reveal a clear divide: KG-judgment training under KL regularization produces sparse, localized updates (a regime we term as surgical alignment), while task-specific SFT produces dense ones. A controlled ablation shows that the objective and KL contribut
    
[^85]: 土耳其语RAG系统中分块与嵌入策略的比较研究

    Comparing Chunking and Embedding Strategies for Turkish RAG Systems

    [https://arxiv.org/abs/2608.26192](https://arxiv.org/abs/2608.26192)

    本文系统比较了土耳其语RAG系统中分块策略与嵌入模型的影响，发现布局感知分块能缩小嵌入模型差异，且领先嵌入模型间无显著统计差异。

    

    arXiv:2608.26192v1 公告类型：交叉 摘要：文档如何被分割为可检索的分块以及这些分块如何被嵌入，强烈影响检索增强生成（RAG）的质量，然而，对于诸如土耳其语这类形态丰富的语言，这两方面都尚未被系统研究。我们比较了土耳其语文档问答在三种分块策略（固定长度、语义和布局感知的Docling）、五种嵌入模型和两种生成器大语言模型上的表现，这些比较基于三份具有对比布局的文档。完全交叉设计产生了9,000个分级问答评估，每个评估由独立的评判模型打分，组件比较通过配对McNemar检验在Holm校正下进行。得出以下四点发现：分块策略决定了嵌入选择的重要性程度——布局感知分块将现代嵌入模型之间的差异压缩到约一分。三个领先的嵌入模型在统计上无显著差异，因此语言稀疏性并未削弱它们的性能。

    arXiv:2608.26192v1 Announce Type: cross  Abstract: How documents are segmented into retrievable chunks and how those chunks are embedded strongly affect Retrieval-Augmented Generation (RAG) quality, yet neither has been systematically studied for morphologically rich languages such as Turkish. We compare Turkish document question answering across three chunking strategies (fixed-length, semantic, and layout-aware Docling), five embedding models, and two generator LLMs, over three documents with contrasting layouts. The fully crossed design yields 9,000 graded question-answer evaluations, each scored by an independent judge model, and component comparisons are tested by paired McNemar tests under Holm correction. Four findings follow. The chunking strategy determines how much the embedding choice matters: layout-aware chunking compresses the spread between the modern embedding models to about a point. The three leading embedding models are statistically indistinguishable, so language sp
    
[^86]: 自生成文本识别：质量启发式、跨任务迁移及LLM评估中的下游偏差

    Self-Generated Text Recognition: Quality Heuristics, Cross-Task Transfer, and Downstream Bias in LLM Evaluation

    [https://arxiv.org/abs/2608.26159](https://arxiv.org/abs/2608.26159)

    本研究通过系统分析实验设计选择（操作化）对自生成文本识别准确率的影响，调和了先前矛盾结论，并证实了质量启发式在LLM评估中的关键作用。

    

    自生成文本识别（SGTR）——即大型语言模型识别自身输出的能力——对依赖LLM作为评估者或监控者的AI安全机制构成风险。具体而言，LLM可能识别出同一模型其他副本的输出，并做出有偏见的判断或直接共谋。以往研究关于当前模型是否具备显著SGTR能力得出了相互矛盾的结论。我们通过识别关键实验设计选择（我们称之为“操作化”）来调和这些发现，这些选择驱动了结果的分歧。评估13-21个模型在六种操作化下的表现，我们发现准确率随评估格式（成对比较 vs. 对文本的单独评估）、对话结构（在用户标签 vs. 助手标签中呈现候选文本）以及用于生成候选文本的任务领域（例如，编程 vs. 摘要）而有显著变化。我们证实了先前观察，即质量启发式...

    arXiv:2608.26159v1 Announce Type: cross  Abstract: Self-Generated Text Recognition (SGTR)--the ability of an LLM to identify its own outputs--poses risks to AI safeguards that rely on LLMs as evaluators or monitors. Specifically, an LLM may recognize outputs from other copies of the same model and make biased judgments or collude outright. Prior work has drawn conflicting conclusions about whether current models possess significant SGTR capabilities. We reconcile these findings by identifying key experimental design choices--which we term operationalizations--that drive divergent results. Evaluating 13-21 models across six operationalizations, we find that accuracy varies substantially with evaluation format (pairwise vs. individual assessments of text), conversation structure (presenting candidate text in user tags vs. assistant tags), and the domain of the task used to generate candidate text (e.g., coding vs. summarization). We corroborate previous observations that a quality heuris
    
[^87]: 人工智能模型能够预测并协同调节人类记忆搜索

    Artificial Intelligence Models Can Predict and Collaboratively Modulate Human Memory Search

    [https://arxiv.org/abs/2608.26152](https://arxiv.org/abs/2608.26152)

    本研究首次证明大型语言模型能够预测并协同调节人类在语义记忆搜索中的心理轨迹，从而作为认知工具增强而非取代人类生成性思维。

    

    arXiv:2608.26152v1 公告类型：交叉 摘要：大型语言模型（LLMs）展现出前所未有的自然语言生成能力和许多基于文本的问题解决能力。实际上，在许多基于语言的任务中，例如常规编程，这些人工智能模型已经减少甚至消除了人类输入的需求。但与其取代人类的认知努力，LLMs或许更适合作为认知工具来扩展人类能力，尤其是在涉及开放式概念探索和创造性构思的任务中。然而，我们尚不清楚这些模型如何在人机交互中增强这种生成性的人类认知能力。在本研究中，我们探索并评估了LLMs在语义记忆搜索过程中跟随和增强人类心理轨迹的能力。为测试这一点，我们使用了语义流畅性任务（SFT），这是一个经典的认知范式，要求生成性语义记忆检索，长期以来一直用于此目的。

    arXiv:2608.26152v1 Announce Type: cross  Abstract: Large language models (LLMs) exhibit unprecedented natural language generation and many text-based problem-solving capabilities. Indeed, in many language-based tasks, for example routine coding, these artificial intelligence models have reduced, or even eliminated, the need for human input. But rather than replacing human cognitive effort, LLMs may instead serve as cognitive tools to extend human abilities, particularly when they are engaged in a task requiring open-ended conceptual exploration and creative ideation. However, we are yet to understand how these models may enhance such generative human cognitive abilities in human--AI interactions. In this study, we explore and evaluate the ability of LLMs to follow and enhance human mental trajectories during semantic memory search. To test this, we use the semantic fluency task (SFT), a classic cognitive paradigm requiring generative semantic memory retrieval that has long served to ch
    
[^88]: ElementCheck：通过句子元素进行复杂度感知的长文本事实性评估

    ElementCheck: Complexity-Aware Long-Form Text Factuality Evaluation via Sentence Elements

    [https://arxiv.org/abs/2608.26118](https://arxiv.org/abs/2608.26118)

    ElementCheck提出了一种基于句子元素图的复杂度感知验证框架，通过图拓扑结构估计句子复杂度并自适应调整验证粒度，解决了长文本事实性评估中固定分解和验证粒度导致的可靠性问题。

    

    现有长文本事实性评估依赖于“分解-检索-验证”流程。然而，该流程受到声明分解噪声和固定验证粒度的影响，导致结果不可靠。我们提出ElementCheck，一种通过句子元素验证长文本输出的复杂度感知框架。ElementCheck不将句子统一分解为原子子声明，而是提取原句中通过可验证连接明确关联的实体对作为元素，并将这些元素组织成元素图。图的拓扑结构为估计句子复杂度提供了结构信号，从而能够对简单句子进行直接验证，对复杂句子进行有针对性的元素级细化和验证。为支持细粒度评估，我们构建了一个新基准FastFact-Sent，将FastFact-Bench中的孤立声明映射回其源句子。实验...

    arXiv:2608.26118v1 Announce Type: new  Abstract: Existing long-form factuality evaluation relies on the decompose-retrieve-verify pipeline. However, the pipeline suffers from noise from claim decomposition and fixed verification granularity, resulting in unreliable results. We propose ElementCheck, a complexity-aware framework that verifies long-form outputs via sentence elements. Instead of uniformly decomposing sentences into atomic sub-claims, ElementCheck extracts entity pairs that are explicitly linked through verifiable connections in the original sentence as elements, and organizes these into an element graph. The graph topology provides a structural signal for estimating sentence complexity, enabling direct verification for simple sentences and targeted element-level refinement and verification for complex ones. To support fine-grained evaluation, we construct a new benchmark FastFact-Sent by mapping isolated claims from FastFact-Bench back to their source sentences. Experiment
    
[^89]: TreeGraft：基于树的投机解码的自适应多草稿器嫁接方法

    TreeGraft: Adaptive Multi-Drafter Grafting for Tree-Based Speculative Decoding

    [https://arxiv.org/abs/2608.26112](https://arxiv.org/abs/2608.26112)

    我们提出了TreeGraft，一种自适应多草稿器嫁接框架，通过结合不同成本的草稿器来构建共享草稿树，从而平衡草稿质量与延迟，提升基于树的投机解码效率。

    

    投机解码通过“先草稿后验证”的范式加速大型语言模型的推理。在此基础上，树结构方法通过将提议组织成多个候选路径来提升推理效率，从而增加接受长度。然而，现有的树结构方法在所有草稿步骤中使用单一草稿器，这造成了一个困境：较小的草稿器速度快但生成的树质量较低，而较大的草稿器能提高树质量但延迟较高。为了解决这个问题，我们提出了TreeGraft，一个多草稿器框架，其中不同成本的草稿器共同构建一个共享的草稿树。TreeGraft使用更强的草稿器通过更新较弱草稿器分配的分数来重新评分候选，重新选择嫁接位置，并恢复未被探索的有前景路径。它还以非破坏性方式整合更强草稿器的扩展，保留可能仍被目标模型接受的现有分支。

    arXiv:2608.26112v1 Announce Type: new  Abstract: Speculative decoding accelerates large language model inference through a draft-then-verify paradigm. Building on this, tree-structured methods improve inference by organizing proposals into multiple candidate paths, increasing the accepted length. However, existing tree-structured methods use a single drafter for all drafting steps, creating a dilemma: a smaller drafter is fast but yields lower-quality trees, whereas a larger drafter improves tree quality but suffers from high latency. To address this, we propose TreeGraft, a multi-drafter framework in which drafters of different costs jointly construct a shared draft tree. TreeGraft uses the stronger drafter to rescore candidates by updating scores assigned by the weaker drafter, reselect grafting positions, and recover promising paths left unexplored. It also integrates stronger drafter expansions non-destructively, preserving existing branches that may still be accepted by the target
    
[^90]: 一种统一转写方式：超越原生书写系统的多语言模型预训练

    One Form to Transfer Them All: Pretraining Multilingual Language Models Beyond Native Orthography

    [https://arxiv.org/abs/2608.25904](https://arxiv.org/abs/2608.25904)

    本研究系统比较了不同输入表示（正字法文本、国际音标、罗马化）在多语言自回归预训练中的跨语言迁移效果，发现罗马化预训练在多种规模和语言对上表现最优，且优势随规模增大而扩大。

    

    arXiv:2608.25904v1 公告类型：新 摘要：多语言模型通过共享子词词汇在语言间传递知识，但这一机制在相关语言使用不同书写系统时会失效。先前研究通过脚本统一（如罗马化或国际音标转写）来解决此问题，但直接比较较少；研究重点集中在仅编码器模型上，且大多数工作是对现有预训练模型进行调整。我们系统比较了自回归多语言预训练中的不同输入表示，在受控设置下，对四种类型学配对的语言对（共八种语言），在三个规模（467M、709M和1.03B）上对比了正字法文本、国际音标和罗马化表示。在广泛的下游任务（涵盖已见和未见语言）中，罗马化预训练展现出最强的跨语言迁移能力，且其优势随模型规模扩大而增强。国际音标在多数设置中优于文本，但略逊于罗马化。令人惊讶的是，微调一个...

    arXiv:2608.25904v1 Announce Type: new  Abstract: Multilingual language models transfer knowledge across languages through shared subword vocabulary, a mechanism that breaks down when related languages use different writing systems. Prior work addresses this via script equalization (romanization or IPA transcription), but direct comparisons are rare; the focus has been on encoder-only models, with most work adapting existing pretrained models. We systematically compare different input representations in autoregressive multilingual pretraining, comparing orthographic text, IPA, and romanization in a controlled setup across three scales (467M, 709M, and 1.03B) on eight languages in four typologically motivated pairs. Across a wide range of downstream tasks on seen and unseen languages, romanized pretraining yields the strongest cross-lingual transfer, and the advantage over text widens with scale. IPA improves over text in most settings but trails romanization. Surprisingly, finetuning a 
    
[^91]: SHROOM-Visions 2026 概览：大型视觉-语言模型幻觉检测共享任务

    Overview of SHROOM-Visions 2026: A Shared Task on Hallucination Detection in Large Vision-Language Models

    [https://arxiv.org/abs/2608.25662](https://arxiv.org/abs/2608.25662)

    本论文介绍了SHROOM-Visions 2026共享任务，该任务利用SHEEP数据集和五类幻觉分类体系，在四种语言中检测大型视觉-语言模型中的细粒度幻觉，以推进模型无关的幻觉检测研究。

    

    2026年，我们举办了SHROOM共享任务系列的第四轮：SHROOM-Visions（大型视觉语言模型幻觉及相关可观察过度生成错误共享任务），该任务与EMNLP 2026联合举办的UncertaiNLP研讨会共同主办。继2024年和2025年任务的成功之后，本次我们旨在通过一个与模型无关的检测任务来应对幻觉问题，重点关注大型视觉-语言模型。基于近期推出的SHEEP数据集（该数据集专为跨模型世代的长期评估而设计），本任务邀请参与者检测并分类图像条件文本生成（如视觉问答、图像描述等）中的细粒度幻觉片段。评估采用涵盖五种幻觉类别的分类体系，并涉及四种语言：中文、英语、法语和意大利语。该共享任务在NLP领域引起了强烈关注。

    arXiv:2608.25662v1 Announce Type: new  Abstract: In 2026, we held the fourth iteration of the SHROOM Shared Task series: SHROOM-Visions (\textbf{S}hared-task on \textbf{H}allucinations and \textbf{R}elated \textbf{O}bservable \textbf{O}vergeneration \textbf{M}istakes in \textbf{Vision} language model\textbf{s}), which is hosted at the UncertaiNLP Workshop co-located with EMNLP 2026. Following the success of the 2024 and 2025 tasks, this time we aim to tackle hallucinations through a model-agnostic detection task focused on large vision-language models. Building on the recently introduced SHEEP dataset, designed for long-term evaluation across model generations, the task invites participants to detect and classify fine-grained hallucination spans in image-conditioned text generation (VQA, image captioning, etc.). The evaluation uses a five-class taxonomy of hallucinations spanning four languages: Chinese, English, French, and Italian. The shared task generated strong interest in the NLP
    
[^92]: 当过时约束未被检查：继承代理记忆中的预算验证失败

    When Stale Constraints Go Unchecked: Budgeted Verification Failures in Inherited Agent Memory

    [https://arxiv.org/abs/2608.25553](https://arxiv.org/abs/2608.25553)

    该论文研究了在有限验证预算下，代理继承的过时约束未被检查导致验证失败的问题，并提出了通过重新分配验证槽位来减少这类错误的方法。

    

    arXiv:2608.25553v1 公告类型：交叉 摘要：一个继承了整合记忆的代理可能继承了一个在写入时成立但已被更新的权威记录撤销的约束。在稀缺的验证预算下，代理能否恢复该撤销？如果不能，这种错误是否能在不增加支出的情况下避免？我们明确建模了替代关系——历史来源是不可变的；变化的是哪个记录是当前的——并设计性地分配了记忆的形式、世界状态（来源当前或已被替代）以及固定预算为两条记录的验证策略：代理自身的分配，或相同预算但将一个槽位重新分配给关键来源路径或随机记录。在声明约束的情况下，代理在大约五分之一的回合中检查了其来源路径；当该约束已被替代时，原生分配在主要运行、新措辞等中分别产生了77.3%、74.7%和74.7%的回合中的过时一致决策。

    arXiv:2608.25553v1 Announce Type: cross  Abstract: An agent that inherits a consolidated memory may inherit a constraint that was true when written and has since been withdrawn by a newer authoritative record. Under a scarce verification budget, does the agent recover the withdrawal, and if not, is the error avoidable without spending more? We model supersession explicitly -- historical provenance is immutable; what changes is which record is current -- and assign by design the memory's form, the world's state (source current or superseded), and the verification policy at a fixed budget of two records: the agent's own allocation, or the same budget with one slot re-assigned to the critical provenance path or to a random record. With a constraint stated, agents inspected its provenance path in about one episode in five; when that constraint had been superseded, native allocation produced stale-consistent decisions in 77.3%, 74.7% and 74.7% of episodes across a primary run, a fresh-wordi
    
[^93]: MathAdv：定理证明器所知晓、推理、形式化与泛化的内容

    MathAdv: What Theorem Provers Know, Reason, Formalize, and Generalize

    [https://arxiv.org/abs/2608.25449](https://arxiv.org/abs/2608.25449)

    本文提出MathAdv基准，通过多任务和专家变换系统评估定理证明器，揭示了形式化瓶颈、领域差异及自然语言对模型影响的异质性。

    

    arXiv:2608.25449v1 公告类型：新 摘要：形式化定理证明能够实现对数学推理的机器可验证评估，然而现有基准通常强调整体证明准确率，集中在狭窄的数学范围，并且对等价改写形式的鲁棒性证据有限。我们引入了MathAdv，一个涵盖本科和研究生水平数学中13个领域的诊断性基准。除了Lean 4定理证明之外，MathAdv还提供最多三个辅助任务：探测数学知识的多项选择题、隔离非正式推理的填空题，以及测试对问题表述鲁棒性的专家构造变换。我们对当代定理证明器的评估得出四个发现：形式化仍是一个主要瓶颈；不同数学领域的性能差异显著；自然语言指导有助于通用大语言模型但可能阻碍专用证明模型；以及数学泛化能力仍不足。

    arXiv:2608.25449v1 Announce Type: new  Abstract: Formal theorem proving enables machine-verifiable evaluation of mathematical reasoning, yet existing benchmarks often emphasize aggregate proof accuracy, concentrate on a narrow range of mathematics, and provide limited evidence of robustness to equivalent reformulations. We introduce MathAdv, a diagnostic benchmark spanning 13 domains across undergraduate- and graduate-level mathematics. Alongside Lean 4 theorem proving, MathAdv provides up to three auxiliary tasks: multiple-choice questions that probe mathematical knowledge, fill-in-the-blank problems that isolate informal reasoning, and expert-crafted transformations that test robustness to problem presentation. Our evaluation of contemporary theorem provers yields four findings: formalization remains a major bottleneck; performance varies substantially across mathematical domains; natural-language guidance helps general-purpose LLMs but can hinder proof-specialized models; and mathem
    
[^94]: 信任大众：KV缓存驱逐中的强制权重

    Trust the Mass: Forced Weights in KV-Cache Eviction

    [https://arxiv.org/abs/2608.25230](https://arxiv.org/abs/2608.25230)

    本文发现KV缓存驱逐中保留最大权重已接近最优，已发布方法间的差异主要源于存储方式而非选择策略，并揭示了评估中的内存与性能权衡。

    

    arXiv:2608.25230v1 公告类型：交叉 摘要：每个部署的稀疏注意力或KV缓存驱逐规则都会保留一部分键，丢弃其余部分，并对保留集上的注意力权重进行重新归一化。在来自五个模型的168,192个注意力行上，枚举该约束下的精确最优子集表明，保留最大权重已经接近最优，因为最优子集仅将剩余差距中位数缩小了2%到5%。如果选择带来的改进如此之小，那么已发布的驱逐方法之间的差异必然来自其他方面，因此我们测量了每种方法持有的字节数。在共享评估流程中，最强的查询无关方法持有完整缓存，因为它们的按头选择存储为掩码，只有不规则按头存储才能释放该内存。在固定选择上强制执行名义预算会损失14到62个基准点。我们将一个87.6点的检索差距追溯到在问题可见时计算的排名。

    arXiv:2608.25230v1 Announce Type: cross  Abstract: Every deployed sparse-attention or KV-cache-eviction rule keeps a subset of the keys, discards the rest, and renormalizes the attention weights over the kept set. Enumerating the exact best subset under that constraint on $168{,}192$ attention rows from five models shows that keeping the largest weights is already near-optimal, since the best subset closes only a median $2$ to $5\%$ of the remaining gap to full attention. If selection closes this little, published margins between eviction methods must come from elsewhere, so we measure the bytes each method holds. In the shared evaluation pipeline, the strongest query-agnostic methods hold the full cache because their per-head selections are stored as masks, and only ragged per-head storage frees that memory. Enforcing a nominal budget on one fixed selection costs $14$ to $62$ benchmark points. We trace an $87.6$-point retrieval margin to rankings computed while the question is visible
    
[^95]: SeMoCo：一种面向运动语言建模的语义优先运动编解码器

    SeMoCo: A Semantic-First Motion Codec for Motion Language Modeling

    [https://arxiv.org/abs/2608.24334](https://arxiv.org/abs/2608.24334)

    提出语义优先的运动编解码器SeMoCo，将每个运动标记分解为一个语义标记与残差运动学标记序列，配合双轴生成器分别建模语义演进与运动学细节以实现语言条件下的文本到动作生成，并构建了统一于SOMA表示的大规模多源人体运动数据集Ω-MotionVerse。

    

    离散运动表示极大地推动了自回归的文本到动作生成。然而，大多数运动分词器是为重构而优化的，并未根据语义角色显式地分配容量。因此，动作级别的含义与细粒度的运动学细节必须通过同一个以重构为导向的层次结构来编码。我们提出了SeMoCo，一种语义优先的运动编解码器，并配套设计了一个用于语言条件动作生成的双轴运动生成器。每个运动标记包含一个语义标记和一个残差运动学标记序列。该生成器对跨时间的语义演进进行建模，并以自回归方式细化残差条目。我们还构建了Ω-MotionVerse，这是一个在SOMA表示下统一的大规模、多源人体运动数据集。在所报告的对比中，SeMoCo在所比较的编解码器中取得了最佳的重构精度。

    arXiv:2608.24334v2 Announce Type: replace-cross  Abstract: Discrete motion representations have substantially advanced autoregressive text-to-motion generation. However, most motion tokenizers are optimized for reconstruction and do not explicitly allocate capacity according to semantic role. Action-level meaning and fine-grained kinematic detail must therefore be encoded through the same reconstruction-driven hierarchy. We introduce SeMoCo, a semantic-first motion codec, together with a dual-axis motion generator for language-conditioned motion generation. Each motion token contains one semantic token and a residual sequence of kinematic tokens. The generator models semantic progression across time and autoregressively refines the residual entries. We also construct $\Omega$-MotionVerse, a large-scale, multi-source human-motion dataset unified under the SOMA representation. Across the reported comparisons, SeMoCo achieves the best reconstruction accuracy among the compared codecs, whi
    
[^96]: 语义覆盖层：通过超越令牌和引导向量的注释缓解提示注入

    Semantic Overlays: Mitigating Prompt Injection with Annotations Beyond Tokens and Steering Vectors

    [https://arxiv.org/abs/2608.23873](https://arxiv.org/abs/2608.23873)

    该论文提出了一种名为“语义覆盖层”的新技术，通过向模型输入添加非文本通道来缓解提示注入攻击，利用小型学习的适配器在冻结模型的残差流中创建带外注释，从而增强模型对片段身份的理解。

    

    摘要：arXiv:2608.23873v1 公告类型：新 摘要：语言模型看到的一切都是令牌。服务堆栈知道每个片段是什么——用户输入、工具输出、指令——但模型必须自己跟踪这些，它可能会失去跟踪或被混淆：文本可以被写成看起来像任何东西。提示注入是对这种现象的自然利用。通过扰乱模型对片段身份的理解，攻击者可以诱导不必要的、可能危险的行为。在模型输入中添加一个非文本通道——一种超越文本传达片段身份的方式——缓解了这类攻击。因此，我们引入了一种通用的引导技术，称为语义覆盖层：小型学习的适配器，应用于冻结模型的残差流中的选定预填充位置。在片段上铺设覆盖层创建了一个带外注释通道，该通道无法通过令牌复制。与引导向量不同，语义覆盖层是经过训练的、可适应的，并有选择性地应用。一个覆盖层...

    arXiv:2608.23873v1 Announce Type: new  Abstract: Everything a language model sees is tokens. The serving stack knows what each span is -- user input, tool output, instructions -- but the model must keep track of that itself, and it can lose track or be confused: text can be written to read like anything. Prompt injection is a natural exploit of this phenomenon. By scrambling the model's understanding of span identity, an attacker can induce unwanted and potentially dangerous actions. Adding a non-textual channel to the model's input -- a way to communicate span identity beyond text -- mitigates this class of attack. We thus introduce a general steering technique called Semantic Overlays: small learned adapters applied at chosen prefill positions to a frozen model's residual stream. Laying an overlay over a span creates an out-of-band annotation channel that cannot be replicated by tokens. Unlike steering vectors, Semantic Overlays are trained, adaptable, and selectively applied. An ove
    
[^97]: JuryProbe：一种用于路由无参考事实性评审团到有依据验证的经验共识风险诊断方法

    JuryProbe: An Empirical Consensus-Risk Diagnostic for Routing Reference-Free Factuality Judge Panels to Grounded Verification

    [https://arxiv.org/abs/2608.20607](https://arxiv.org/abs/2608.20607)

    本文提出JuryProbe，一种通过仅假阴性相关性和假共识提升度来诊断无参考事实性评审团共识风险的方法，并在高风险时路由到有参考验证，以减少因共享盲点导致的错误接受。

    

    arXiv:2608.20607v1 公告类型：交叉 摘要：由廉价LLM评审员组成的小组越来越多地做出接受或升级的决策。在事实性设置中，因为多个无参考评审员一致同意而接受一个声明可能会产生隐藏风险：这种一致性可能反映的是共同的假阴性盲点，而非独立的证据。我们引入了JuryProbe，一种针对无参考事实性评审团的经验共识风险诊断方法，并配以基于校准的路由策略。JuryProbe通过使用仅假阴性（FN-only）评审员相关性和假共识提升度，从标记的校准探针中估计共识风险；当标记为高风险时，无参考多数接受会被路由到带有可信参考的相同评审员。在审计的FEVER腐败数据上，无参考评审团显示出相关的假阴性（FN-only相关性为0.402和0.368；提升度分别为3.13倍和18.13倍），而在可信参考最佳案例诊断下，两种情形的一致假共识均降至零。

    arXiv:2608.20607v1 Announce Type: cross  Abstract: Panels of inexpensive LLM judges increasingly make accept-or-escalate decisions. In factuality settings, accepting a claim because several reference-free judges agree can create a hidden risk: agreement may reflect shared false-negative blind spots rather than independent evidence. We introduce JuryProbe, an empirical consensus-risk diagnostic for reference-free factuality judge panels, paired with a calibration-based routing policy. JuryProbe estimates consensus risk from a labeled calibration probe using false-negative-only (FN-only) judge correlation and false-consensus lift; when flagged high-risk, reference-free majority accepts are routed to the same judges with trusted references. On audited FEVER corruptions, reference-free panels show correlated false negatives (FN-only correlations 0.402 and 0.368; lifts 3.13x and 18.13x), while unanimous false consensus drops to zero under a trusted-reference best-case diagnostic on both min
    
[^98]: 双语混合专家语言模型中专家路由的陈述性-程序性视角

    A Declarative-Procedural Perspective on Expert Routing in Bilingual Mixture-of-Experts Language Models

    [https://arxiv.org/abs/2608.15102](https://arxiv.org/abs/2608.15102)

    本研究通过陈述性-程序性框架分析双语MoE模型，发现无课程训练的混合数据基线比顺序课程训练展现出更强的语言类别专家路由特化，挑战了传统课程学习假设。

    

    我们研究了混合专家（MoE）语言模型在双语习得过程中是否发展出具有语言学结构的专家路由。受陈述性-程序性框架的启发，我们在顺序语言暴露下训练的仅解码器英德MoE Transformer中，分析了词汇、语法和句法处理。我们构建了一个基于探针的验证集，并提取了令牌级路由分布，以通过互信息、路由熵和Jensen-Shannon距离来量化类别依赖的特化。课程训练模型在第5层达到峰值互信息0.1148，表明不同语言类别间的路由分布存在类别依赖差异。令人惊讶的是，一个在混合英德数据上训练的无课程基线显示出更强的整体特化，在同一层达到峰值互信息0.2599。这些结果表明，i

    arXiv:2608.15102v1 Announce Type: new  Abstract: We investigate whether Mixture-of-Experts (MoE) language models develop linguistically structured expert routing during bilingual language acquisition. Inspired by the Declarative-Procedural framework, we analyze lexical, grammatical, and syntactic processing in a decoder-only English-German MoE Transformer trained under sequential language exposure. We construct a probe-based validation set and extract token-level routing distributions to quantify category-dependent specialisation using mutual information, routing entropy, and Jensen-Shannon distance. The curriculum-trained model exhibits a peak mutual information of 0.1148 at layer 5, indicating category-dependent differences in routing distributions across linguistic categories. Surprisingly, a no-curriculum baseline trained on mixed English-German data shows stronger aggregate specialisation, reaching a peak mutual information of 0.2599 at the same layer. These results suggest that i
    
[^99]: 知道两个跳数还不够：理解语言模型中的两跳泛化

    Why Knowing Both Hops Is Not Enough: Understanding Two-Hop Generalization in Language Models

    [https://arxiv.org/abs/2608.07261](https://arxiv.org/abs/2608.07261)

    本文通过受控符号环境中的变换器训练和机制分析，揭示了两跳泛化中第二跳分布内成功而分布外失败的根本原因，即一致中间表示的出现与层间不匹配。

    

    大型语言模型（LLMs）能够解决复杂的多跳问题，但在简单的两跳查询上却表现出令人困惑的失败：尽管模型可能正确存储每个单独的跳，但它常常无法将它们组合起来。为了理解这一现象的内部机制，我们在受控符号环境中从头训练了变换器模型。我们的实验揭示了两跳泛化中的一种模式：当第二跳遵循训练分布时，模型能可靠地泛化，但当其偏离时则总是失败。通过机制分析，我们为这些不同的泛化行为提供了完整解释：在模型成功泛化的设置中，性能是由跨上下文同一实体的一致中间表示的出现所驱动，而在第二跳分布外失败的情况下，则源于层间的不匹配：较低层正确构建了这些表示，但较高层未能有效利用它们。

    arXiv:2608.07261v2 Announce Type: replace  Abstract: Large language models (LLMs) can solve complex multi-hop problems yet exhibit puzzling failures on simple two-hop queries: although a model may correctly store each individual hop, it often fails to combine them. To understand the internal mechanisms of this phenomenon, we train transformers from scratch in a controlled symbolic environment. Our experiments reveal a pattern in two-hop generalization: models generalize reliably when the second hop follows the training distribution, but always fail when it deviates. Through mechanistic analysis, we provide a complete explanation for these distinct generalization behaviors: in settings where models generalize successfully, performance is driven by the emergence of consistent intermediate representations for the same entities across contexts, whereas failures on settings where the second hop is out-of-distribution arise from a mismatch across layers: lower layers correctly construct thes
    
[^100]: 搜索、检查、获取：利用结构感知布尔检索提升深度搜索智能体

    Search, Inspect, Fetch: Exploiting Structure-Aware Boolean Retrieval for Deep-Search Agents

    [https://arxiv.org/abs/2608.02751](https://arxiv.org/abs/2608.02751)

    提出基于布尔查询语言的Sieve搜索-检查-获取策略，通过结构感知检索在提升深度搜索智能体准确性的同时，将token消耗降低20.7%-50.6%。

    

    现有的深度搜索智能体采用“搜索-访问”工作流，在检索整个网页时不考虑网页通过标题、小标题、章节和元数据所暴露的结构。这阻碍了智能体将检索直接约束到网页的特定部分，并且经常将无关内容带入其上下文中。我们提出了Sieve，一种由布尔查询语言（BQL）驱动的搜索-检查-获取策略：它搜索网页字段以过滤候选结果，使用可互换的排序器对结果进行排序，呈现结构丰富的结果卡片以供检查，并仅获取选定的章节。在三个问答数据集上，Sieve在每个数据集上都比最强的传统“搜索-访问”配置更准确，同时使用的token减少了20.7%-50.6%。布尔过滤提升了所有被测试排序器的性能，且这种准确性与上下文效率的优势在不同的检索器选择和智能体骨干网络中均得以保持。

    arXiv:2608.02751v3 Announce Type: replace-cross  Abstract: Existing deep-search agents use a Search-Visit workflow that retrieves whole webpages without considering the structure they expose through titles, headings, sections, and metadata. This prevents agents from directly constraining retrieval to parts of a webpage and often carries irrelevant content into their context. We introduce Sieve, a search-inspect-fetch strategy driven by a Boolean Query Language (BQL): it searches webpage fields to filter candidates, uses an interchangeable ranker to order them, presents structure-rich result cards for inspection, and fetches only selected sections. Across three QA collections, Sieve is more accurate than the strongest conventional Search-Visit configuration on each collection while using 20.7-50.6% fewer tokens. Boolean filtering improves every tested ranker, and the accuracy-context advantage persists across retriever choices and agent backbones. Our implementation is included in the S
    
[^101]: 引导信号从何而来：激活引导中的激活源选择

    Where Steering Signals Come From: Activation Source Selection in Activation Steering

    [https://arxiv.org/abs/2607.25270](https://arxiv.org/abs/2607.25270)

    该论文首次将激活引导中常被忽视的“激活源选择”作为核心研究对象，发现引导信号的有效性关键取决于激活是否取自模型即将执行目标行为的“执行边界状态”，而非源文本中是否包含期望行为。

    

    激活引导通过在推理时向隐藏状态中添加向量或特征来控制语言模型，但这些引导信号的上游来源通常被视为次要细节。我们将这一来源选择作为“激活源选择”进行研究：即用于收集隐藏状态（并从中构建引导信号）的源上下文与激活读取策略的组合。在保持下游干预不变的情况下，我们在三个指令微调模型和四个引导任务族上证明，仅改变源激活就会显著改变引导的成功率。我们进一步发现，有效的引导并不能简单地用源文本中是否出现期望行为来解释。相反，强信号来自“执行边界状态”，即模型即将产生或继续目标行为时的状态。这种实现前/实现后的区分解释了为什么基于答案的源有时有效：……

    arXiv:2607.25270v2 Announce Type: replace  Abstract: Activation steering controls language models by adding vectors or features to hidden states at inference time, but the upstream source of these steering signals is often treated as a secondary detail. We study this source choice as activation source selection: the combination of source context and activation readout policy used to collect the hidden states from which a steering signal is built. Holding the downstream intervention fixed, we show across three instruction-tuned models and four steering task families that changing only the source activations substantially changes steering success. We further find that effective steering is not explained simply by whether the desired behavior appears in the source text. Instead, strong signals come from execution-boundary states, where the model is about to produce or continue the target behavior. This pre-/post-realization distinction explains why answer-based sources sometimes work: the
    
[^102]: 面向配备框架智能体的定势转换行为测试

    Set-shifting Behavioral Test for Harnessed Agents

    [https://arxiv.org/abs/2607.13396](https://arxiv.org/abs/2607.13396)

    该论文借鉴认知心理学中的“定势转换”概念，提出了一种通过在冗余工具库中隐藏地切换可靠工具组来测试LLM智能体适应能力的行为测试方法，并发现不同模型面对相同切换时表现出截然不同的行为模式。

    

    当可靠的工具在持续会话中悄然发生变化时，LLM智能体的工具选择会发生什么？我们从认知心理学中借鉴了“定势转换”的概念，研究智能体对隐藏可靠性变化的适应能力。我们为LLM智能体设计的认知测试挂载了冗余的工具与技能库，其中多个工具可以解决同一任务，但在隐藏的可靠性上存在差异。通过分支式的调度安排，我们在环境中切换可靠工具组，并与稳定的对照组进行比较，从而能够分离出每一次切换对智能体行为的独立影响。我们在一组配备框架的LLM上开展了研究，结果表明同一组切换在不同模型中引发了截然不同的行为：有些模型在几轮之内就固守某种固定模式，而另一些则持续变化。能力较弱的模型往往会忽略可靠工具组，而前沿模型则会在调用其他工具组的同时持续调用可靠工具组。（原文摘要在此处截断）

    arXiv:2607.13396v2 Announce Type: replace-cross  Abstract: What happens to an LLM agent's tool choice when the reliable tool silently changes within an ongoing session? We borrow the notion of set-shifting from cognitive psychology to study how well agents adapt to hidden reliability shifts. Our cognitive test for LLM agents mounts libraries of redundant tools and skills, in which many tools solve the same task but differ in hidden reliability. Using a branching schedule, we shift the reliable tool group in the environment and compare it with a stable control, allowing us to isolate the effect of each shift on the agent's behavior. We conduct our study on a panel of LLMs equipped with harnesses and show that the same set of shifts results in distinct behaviors across models: some latch onto a fixed routine within a few turns, whereas others continue to vary. Less capable models often omit the reliable tool group, while frontier models keep calling it alongside the other groups. We intr
    
[^103]: 基于大语言模型的意图驱动网络拓扑设计框架

    An LLM-Based Framework for Intent-Driven Network Topology Design

    [https://arxiv.org/abs/2607.00292](https://arxiv.org/abs/2607.00292)

    该论文提出了一个基于大语言模型的意图驱动网络拓扑设计框架，通过结合分层建模与系统性验证的约束驱动流水线，使LLM能够从自然语言需求生成结构有效且符合约束的网络拓扑，并发布了包含四个真实网络场景的公开基准数据集用于多模型评估。

    

    从自然语言需求出发设计可部署且具有弹性的网络拓扑，仍然是网络自动化领域一个具有挑战性的问题。本工作研究了大型语言模型（LLM）通过结合分层建模与系统性验证的约束驱动流水线，生成结构有效且符合约束的网络拓扑的能力。该框架通过对专有和开放权重LLM进行多模型比较来评估，涵盖四个真实网络场景，并作为公开数据集发布。我们使用节点和边的F1分数对照参考拓扑来评估结构正确性，并通过服务器和内容连接性指标来评估网络弹性。此外，我们还分析了常见的失败模式，包括生成拓扑中的接口不匹配和方向不一致问题。总体而言，这项工作为理解LLM如何处理结构和约束问题提供了一个系统性的基准。

    arXiv:2607.00292v2 Announce Type: replace-cross  Abstract: Designing deployable and resilient network topologies from natural language requirements remains a challenging problem in network automation. This work investigates the ability of Large Language Models (LLMs) to generate structurally valid and constraint-compliant network topologies through a constraint-driven pipeline combining hierarchical modeling and systematic validation. The framework is evaluated via a multimodel comparison of proprietary and open-weight LLMs across four realistic network scenarios released as a public dataset. We assess structural correctness using node and edge F1-scores against reference topologies, and evaluate resilience through server and content connectivity metrics. In addition, we analyze common failure modes, including interface mismatches and directional inconsistencies in generated topologies. Overall, this work provides a systematic benchmark for understanding how LLMs handle structural and 
    
[^104]: LV-ROVER-MLT：基于合成数据微调与多流仲裁的低资源马耳他语OCR

    LV-ROVER-MLT: Low-Resource Maltese OCR by Synthetic Fine-Tuning and Multi-Stream Arbitration

    [https://arxiv.org/abs/2607.00250](https://arxiv.org/abs/2607.00250)

    该论文提出LV-ROVER-MLT系统，通过合成数据微调Tesseract 5并结合五路识别流与词典门控词级仲裁，在仅57页标注数据的低资源条件下以0.0074的字符错误率赢得DocEng 2026马耳他语OCR竞赛冠军，同时发布了36,803对的马耳他语OCR新语料库。

    

    马耳他语拥有丰富的文本语料库和预训练语言模型，但段落规模的OCR训练数据仍然稀缺；NOMOCRAT仅提供57页经过验证的标注页面。LV-ROVER-MLT结合了Tesseract 5的合成数据微调、五个互补的识别流，以及针对马耳他语变音符号和连字符规则适配的词典门控词级仲裁机制。在DocEng 2026马耳他语OCR竞赛中，该系统以留出集字符错误率（CER）0.0074获得第一名；排名第二的提交结果为0.0161，NOMOCRAT为0.0163。相同方法在卢森堡语上相比原版Tesseract取得了显著改进，而匈牙利语的结果尚无定论。从EUR-Lex和维基百科构建的36,803对马耳他语OCR语料库提供了额外的段落级资源。代码、模型权重和语料数据均已公开。

    arXiv:2607.00250v5 Announce Type: replace  Abstract: Maltese has substantial text corpora and pretrained language models, but paragraph-scale OCR training data remains scarce; NOMOCRAT provides 57 verified annotated pages. LV-ROVER-MLT combines synthetic fine-tuning of Tesseract~5 with five complementary recognition streams and lexicon-gated word-level arbitration adapted to Maltese diacritics and hyphenation. In the DocEng~2026 Maltese OCR competition, the system placed first with held-out CER 0.0074; the next-ranked submission scored 0.0161 and NOMOCRAT scored 0.0163. The same approach produced a significant improvement over stock Tesseract on Luxembourgish, while the Hungarian result was inconclusive. A 36,803-pair Maltese OCR corpus constructed from EUR-Lex and Wikipedia provides an additional paragraph-level resource. Code, model weights, and corpus data are public.
    
[^105]: ProfileFoundry：用于LLM智能体隐私、记忆与工具使用评估的合成人-物基础平台

    ProfileFoundry: A Synthetic Person-Object Substrate for Privacy, Memory, and Tool-Use Evaluation in LLM Agent

    [https://arxiv.org/abs/2606.26403](https://arxiv.org/abs/2606.26403)

    本文提出了PROFILEFOUNDRY，一个确定性生成器，发布了10万个跨八个地区的合成人物对象数据集，包含丰富的个人状态、关系、事件和来源信息，以解决真实用户数据难以共享和评估的问题，为LLM智能体的隐私、记忆和工具使用评估提供可靠基础。

    

    arXiv:2606.26403v1 公告类型：新 摘要：基础模型研究越来越需要关于人的数据：用户状态、个人历史、人际关系、类似联系人字段、文档以及纵向更新。真实用户数据难以负责任地共享、扰动、审计或重新分发，而独立生成的虚假字段很少能保持受控评估所需的跨字段和时间一致性。我们提出了PROFILEFOUNDRY，这是一个确定性生成器，并发布了固定参考数据集，包含跨八个地区的10万个成人合成人物对象。每个对象结合了类型化的当前快照、家庭、家族和雇主链接、快照对齐的事件、规范化的关系视图以及生成来源。该数据集包含709,228个事件、40,338个家庭、52,491个雇主和518,564条有向关系边。我们在不同类别中报告了证据：选定的人口边缘比较、每个对象的不变性检查、数据集范围的引用完整性。

    arXiv:2606.26403v1 Announce Type: new  Abstract: Foundation-model research increasingly needs data about people: user state, personal histories, relationships, contact-like fields, documents, and longitudinal updates. Real user data is difficult to share, perturb, audit, or redistribute responsibly, while independently generated fake fields rarely preserve the cross-field and temporal consistency needed for controlled evaluation. We present PROFILEFOUNDRY, a deterministic generator and fixed reference release of 100,000 adult synthetic Person Objects across eight locales. Each object combines a typed current snapshot, household, family, and employer links, snapshot-aligned events, normalized relational views, and generation provenance. The release contains 709,228 events, 40,338 households, 52,491 employers, and 518,564 directed relationship edges. We report evidence in separate categories: selected population-marginal comparisons, per-object invariant checks, release-wide referential 
    
[^106]: 机器中的CASPER：洞察大语言模型生成故事中的角色多样性

    CASPER in the Machine: Insights into Character Variety in LLM-Generated Stories

    [https://arxiv.org/abs/2606.22454](https://arxiv.org/abs/2606.22454)

    该研究借鉴叙事学理论，从风格化、完整性等八个维度自动分析并对比LLM生成故事与人类撰写故事中的角色刻画，探究两者角色是否相似以及LLM能否生成角色多样的故事。

    

    随着大语言模型（LLM）生成的文本日益普及，尤其是在虚构作品领域，我们探讨了LLM生成的故事与人类撰写的故事之间存在多大差异。在这项工作中，我们聚焦于角色。我们借鉴叙事学的定义来分析角色的八个复杂维度，例如风格化和完整性。这些维度考虑的不仅仅是基本特征，还评估角色在故事中是如何被刻画的。在自动推断出LLM生成故事和人类撰写故事中的角色类别之后，我们对这两组故事进行了比较和对比。我们考虑以下两个总体问题：(1) LLM生成的故事与人类撰写的故事是否具有相似的角色？(2) LLM是否能生成具有多样化角色的故事？我们的分析包括针对流行LLM生成的故事以及最近发表的人类撰写故事的研究问题。我们描述了许多有趣的相似之处……

    arXiv:2606.22454v2 Announce Type: replace  Abstract: As LLM-generated text is increasingly used, especially in fictional domains, we explore how much LLM-generated stories differ from human-written stories. In this work, we focus on characters. We borrow definitions from narratology to analyze eight intricate dimensions of character, such as stylization and wholeness. These dimensions consider more than just basic characteristics. They assess how characters are portrayed within their stories. After automatically inferring categories of characters within both LLM and human-written stories, we compare and contrast these two sets of stories. We consider the following overarching questions: (1) Do LLMs and human-written stories have similar characters? and (2) Do LLMs generate stories with a variety of characters? Our analysis includes research questions that focus on stories generated by popular LLMs and recently published human-written stories. We describe a number of interesting similar
    
[^107]: 用科学数据进行微调会增加幻觉吗？对大语言模型的多领域事实性评估

    Does Finetuning with Scientific Data Increase Hallucinations? A Multi-domain Factuality Evaluation of LLMs

    [https://arxiv.org/abs/2606.21359](https://arxiv.org/abs/2606.21359)

    提出SciFactCheck多领域基准，涵盖五个科学领域的2,500个提示并针对三种幻觉类型进行评估，发现用科学数据微调的大语言模型相比其通用基础模型在各类幻觉类型上的事实可靠性均出现下降。

    

    大语言模型（LLM）越来越多地被用于交流和解释科学概念，然而其产生幻觉的倾向在这种高风险应用场景中构成了重大风险。以往的科学幻觉评估工作大多局限于生物医学领域，将幻觉视为二元任务，且尚未对日益增多的科学微调大语言模型进行研究。我们通过SciFactCheck来填补这些空白，这是一个涵盖五个科学领域、包含2,500个提示的基准测试，并配有一个针对三种事实性幻觉类型（不可验证性、过度声明和归因）的模块化评估框架。采用受控最小配对设计，我们评估了18个大语言模型，将每个科学微调模型与其通用基础模型进行比较。我们的结果表明：1. 科学微调模型在所有幻觉类型和科学领域中均表现出事实可靠性下降；2. （摘要在此处截断）

    arXiv:2606.21359v2 Announce Type: replace  Abstract: Large language models (LLMs) are increasingly used to communicate and explain scientific concepts, yet their tendency to hallucinate poses significant risks in this high stakes use-case. Prior scientific hallucination evaluation work remains largely restricted to the biomedical domain, treats hallucination as a binary task, and has not examined the growing family of scientifically fine-tuned LLMs. We address these gaps with SciFactCheck, a benchmark of 2,500 prompts across five scientific domains, paired with a modular evaluation framework targeting three factuality hallucination types: unverifiability, overclaim, and attribution. Using a controlled minimal-pairing design, we evaluate 18 LLMs by comparing each scientifically fine-tuned model against its general-purpose base. Our results indicate that 1. Scientifically fine-tuned models exhibit degraded factual reliability across all hallucination types and scientific domains, and 2. 
    
[^108]: 弥合语义缓存中的运营差距

    Closing the Operational Gap in Semantic Caching

    [https://arxiv.org/abs/2606.19719](https://arxiv.org/abs/2606.19719)

    该论文指出PR-AUC指标会误导语义缓存系统的部署决策，提出了缓存感知的P-CHR AUC指标和运营保留率ORR，并将离线与部署质量间的运营差距分解为可恢复的阈值效用部分和由数据集正例率决定的不可约简结构部分。

    

    语义缓存通过为语义相似的查询提供缓存响应来降低大语言模型（LLM）的推理成本。标准做法是使用PR-AUC来评估这些系统，但该指标仅衡量分数的排序质量，而忽略了分数在固定阈值下是否可用。我们证明这种错位会导致系统性的糟糕部署选择，因为PR-AUC最高的模型在实际运行中往往表现最差。我们引入了精确率-缓存命中率（P-CHR）AUC这一缓存感知指标，用于衡量不同缓存利用率水平下的精确率；以及运营保留率（ORR），用于捕捉离线排序质量在部署时的保留程度。我们将离线质量与部署质量之间的运营差距分解为可恢复的阈值效用部分，以及由数据集正例率固定的不可约简的结构部分。我们的实验表明，阈值效用差距由训练目标决定，而非……（摘要原文在此处截断）

    arXiv:2606.19719v3 Announce Type: replace-cross  Abstract: Semantic caching cuts LLM inference costs by serving a cached response to semantically similar queries. Standard practice evaluates these systems using PR-AUC, a metric that only measures how well scores rank and ignores whether they are usable at a fixed threshold. We show this mismatch leads to systematically poor deployment choices, as models with the highest PR-AUC are often the worst in operation. We introduce Precision--Cache Hit Ratio (P-CHR) AUC, a cache-aware metric that measures precision across cache utilization levels, and Operational Retention Rate (ORR), which captures how much offline ranking quality survives at deployment. We decompose the operational gap between offline and deployed quality into a recoverable threshold-utility component and an irreducible structural component fixed by the dataset's positive rate. Our experiments show that the threshold-utility gap is governed by the training objective rather th
    
[^109]: 保护多智能体GIS系统：风险评估与提示强化优化

    Securing Multi-Agent GIS Systems: Risk Evaluation and Prompt Hardening Optimization

    [https://arxiv.org/abs/2606.17092](https://arxiv.org/abs/2606.17092)

    该论文提出了一个面向安全的多智能体GIS系统框架，通过基于状态机的模块化编排、自适应攻击者红队测试评估以及将提示视为结构化签名的提示优化方法，实现风险识别、评估与缓解，提升系统安全性。

    

    智能体系统正越来越多地与地理信息系统（GIS）集成，多智能体协作能够实现复杂的对话式与空间分析，但同时也带来了安全风险。本工作提出了一个面向安全的多智能体GIS系统风险识别、评估与缓解框架，同时保持对更广泛智能体架构的适应性。我们测试了一家商业地理空间合作伙伴的智能体系统，并开发了一个基于状态机的模块化编排框架，将智能体行为抽象为可复用的组件。我们使用红队测试框架评估系统鲁棒性，该框架包含一个自适应攻击者大语言模型和一个确定性评判器，可在多轮攻击中产生带有支持性理由的二值评判结果。我们进一步通过一个提示优化框架提升系统韧性，该框架将提示视为结构化签名并注入对抗性演示，使系统能够……（原文摘要截断于此）

    arXiv:2606.17092v2 Announce Type: replace-cross  Abstract: Agentic systems are increasingly integrated with geographic information systems (GIS), where multi-agent coordination enables complex conversational and spatial analysis but introduces security risks. This work presents a security-oriented framework for risk identification, evaluation, and mitigation in a multi-agent GIS system while maintaining adaptability to broader agentic architectures. We test the agentic system of a commercial geospatial partner while developing a modular state-machine-based orchestration framework that abstracts agent behavior into reusable components. We evaluate robustness using a red-teaming framework with an adaptive attacker LLM and a deterministic judge that produces binary outcomes with supporting rationales across multi-turn attacks. We further improve resilience with a prompt optimization framework that treats prompts as structured signatures and injects adversarial demonstrations, enabling sys
    
[^110]: TokenPilot：面向大语言模型智能体的缓存高效上下文管理

    TokenPilot: Cache-Efficient Context Management for LLM Agents

    [https://arxiv.org/abs/2606.17016](https://arxiv.org/abs/2606.17016)

    TokenPilot提出了一种双粒度上下文管理框架，通过全局的感知摄取压缩来稳定提示前缀并消除环境噪声，结合局部的生命周期感知驱逐机制仅在任务相关性过期时卸载内容，从而在降低大语言模型智能体推理成本的同时保持提示缓存的连续性。

    

    随着大语言模型智能体被部署在长时程会话中，上下文的不断累积推高了推理成本。现有方法利用文本剪枝或动态内存驱逐来最小化令牌占用；然而，它们不受约束的序列变更会改变布局，引入前缀不匹配和缓存失效问题。这揭示了文本稀疏性与提示缓存连续性之间的关键权衡。为解决这一问题，我们提出了TokenPilot，一个双粒度的上下文管理框架。在全局层面，感知摄取的压缩机制充当框架治理工具，在摄取入口处稳定提示前缀并消除开放世界的环境噪声。在局部层面，生命周期感知的驱逐机制监控上下文片段的持续残余效用，执行保守的批轮次调度，仅在任务相关性过期时才卸载内容片段。在PinchBench和Claw-Eval基准上以隔离和连续两种模式进行的实验证明了……（原文摘要在此处截断）

    arXiv:2606.17016v2 Announce Type: replace  Abstract: As LLM agents are deployed in long-horizon sessions, context accumulation drives up inference costs. Existing approaches utilize text pruning or dynamic memory eviction to minimize token footprints; however, their unconstrained sequence mutations alter layouts, introducing prefix mismatches and cache invalidation. This reveals a critical trade-off between text sparsity and prompt cache continuity. To address this, we present TokenPilot, a dual-granularity context management framework. Globally, Ingestion-Aware Compaction acts as a framework harness to stabilize prompt prefixes and eliminate open-world environmental noise at the ingestion gate. Locally, Lifecycle-Aware Eviction monitors the ongoing residual utility of context segments, enforcing a conservative batch-turn schedule to offload content segments only when task relevance expires. Experiments on PinchBench and Claw-Eval under both isolated and continuous modes demonstrate th
    
[^111]: 说服指数：一种以理论为指导的说服分析框架

    Persuasion Index: A Theory-Guided Framework for Persuasion Analysis

    [https://arxiv.org/abs/2606.14580](https://arxiv.org/abs/2606.14580)

    提出了基于心理学与传播学说服理论的“说服指数”（PI）——一个包含15个维度的模块化说服分类体系及其由词典和规则构建的55个子特征的透明实现，在多个英语论证文本数据集上验证了其作为轻量级共享特征空间可有效解释与说服相关的修辞模式。

    

    识别具有说服力的修辞线索在多个领域都至关重要，包括检测信息操纵、提升人工智能安全性以及推进公共卫生传播。我们提出了说服指数，这是一个基于心理学和传播学说服理论构建的包含15个维度的分类体系，并提供了一种透明的实现方式，使用由词典和基于规则的检测器构建的55个子特征。该分类体系是模块化的：可以在保留理论结构的同时替换单个检测器。我们在四个公开的英语论证文本数据集上评估了PI，这些数据集在领域、风格和结果衡量方式上各不相同，结果表明PI为解释与说服相关结果相关联的修辞模式提供了一个共享的特征空间。线性模型表明，PI特征具有有意义的预测信号，同时保持计算上的轻量性。维度层面的分析揭示了PI各维度与说服相关结果之间反复出现的关联。

    arXiv:2606.14580v2 Announce Type: replace  Abstract: Identifying persuasive rhetorical cues is critical across domains, from detecting information manipulation and improving AI safety to advancing public health communication. We propose the Persuasion Index (PI), a taxonomy of 15 dimensions grounded in persuasion theories from psychology and communication, and one transparent implementation using 55 sub-features built from lexicons and rule-based detectors. The taxonomy is modular: individual detectors can be replaced while preserving the theoretical structure. We evaluate PI on four public datasets for English argumentative text that vary in domain, style, and outcome measures, and show that PI provides a shared feature space for interpreting rhetorical patterns associated with persuasion-related outcomes. Linear models show that PI features carry meaningful predictive signal while remaining computationally lightweight. Dimension-level analyses reveal recurring associations between PI
    
[^112]: MemoryCard：面向长视频问答的主题感知多模态线索压缩

    MemoryCard: Topic-Aware Multi-Modal Clue Compression for Long-Video Question Answering

    [https://arxiv.org/abs/2606.05917](https://arxiv.org/abs/2606.05917)

    MemoryCard提出了一种基于视频记忆的增强框架，通过将长视频组织成主题感知、语义连贯的记忆卡片来替代碎片化帧证据，从而提升视觉语言模型在长视频问答中捕捉事件级语义的能力。

    

    长视频问答对视觉语言模型（VLMs）而言仍然充满挑战，因为与答案相关的证据往往稀疏、短暂，且在冗长的视频上下文中随时间分散分布。现有的以帧为中心的方法通过均匀采样、查询感知的帧选择、视觉token压缩和自适应分辨率策略来提升效率。然而，这些方法仍依赖孤立且碎片化的帧作为基本证据单元，限制了VLMs有效捕捉连贯事件级语义的能力。为解决这一局限，我们提出了MemoryCard，一种基于视频记忆的增强框架，它将长视频组织成自包含的记忆卡片（Memory Cards）。具体而言，MemoryCard首先对视频及其对齐的话语执行自读取过程，将视频分割为语义连贯的单元，每个单元对应一个不同的主题或事件。对于每个单元，它会生成事件级的……（摘要原文在此处截断）

    arXiv:2606.05917v2 Announce Type: replace-cross  Abstract: Long-video question answering remains challenging for Vision-Language Models (VLMs), as answer-relevant evidence is often sparse, transient, and temporally dispersed across lengthy video contexts. Existing frame-centric approaches improve efficiency through uniform sampling, query-aware frame selection, visual-token compression, and adaptive resolution strategies. However, they still rely on isolated and fragmented frames as the fundamental evidence units, limiting VLMs' ability to effectively capture coherent event-level semantics. To address this limitation, we propose MemoryCard, a video-memory-based augmentation framework that organizes long videos into self-contained Memory Cards. Specifically, MemoryCard first performs a self-reading process over videos and aligned utterances to segment the video into semantically coherent units, each corresponding to a distinct topic or event. For each unit, it generates an event-level v
    
[^113]: 粒度差距：Gemini模型中谄媚行为的多维跨代审计

    The Granularity Gap: A Multi-Dimensional Cross-Generational Audit of Sycophancy in Gemini Models

    [https://arxiv.org/abs/2606.05183](https://arxiv.org/abs/2606.05183)

    该论文揭示了安全评估中“通过/失败”二元判定与谄媚行为连续评分之间存在不可弥合的“粒度差距”，表明现有评估方法无法充分捕捉模型取悦用户的细微行为。

    

    arXiv:2606.05183v2 公告类型：替换-交叉 摘要：通过/失败的安全评估报告模型是否拒绝，但它不报告模型为了取悦用户而走多远，我们表明这两者接近不同的测量。我们对三代Gemini模型进行了谄媚行为审计，在3种护栏条件下，对7个类别的350个对抗性提示，对8个模型变体的N=8,830个响应进行了评分，采用1-5连续量表评估谄媚性、真实性和拒绝性。评判者自身的拒绝或服从判定仅解释了其谄媚评分方差的29%。我们将剩余部分称为“粒度差距”，它在重新校准下不会闭合：已在使用的切点是在拒绝轴上可用的最佳切点，且该轴的任何函数都无法解释超过35%的方差。阅读四位评判者在评分时写下的内容揭示了原因。在四分之一到三分之一的投票中，他们记录提示未要求任何有害内容，几乎从未出现在两个征求有害行为的类别中。

    arXiv:2606.05183v2 Announce Type: replace-cross  Abstract: Pass/fail safety evaluation reports whether a model refused. It does not report how far a model went to please the user, and we show these are close to different measurements. We audited sycophancy across three Gemini generations, scoring N=8,830 responses from 8 model variants on 350 adversarial prompts in 7 categories under 3 guardrail conditions, on continuous 1-5 scales for sycophancy, truthfulness and refusal.   The judge's own refuse-or-comply verdict explains 29% of the variance in its own sycophancy scores. We term the remainder the Granularity Gap, and it does not close under recalibration: the cut point already in use is the best available on the refusal axis, and no function of that axis explains more than 35%. Reading what four judges wrote while scoring shows why. On a quarter to a third of votes they record that the prompt asked for nothing harmful, almost never in the two categories that solicit a harmful act and
    
[^114]: 自我评估已然存在：用极少数据激发基础大语言模型中潜在的裁判校准能力

    Self-Evaluation Is Already There: Eliciting Latent Judge Calibration in Base LLMs with Minimal Data

    [https://arxiv.org/abs/2606.05122](https://arxiv.org/abs/2606.05122)

    基础大语言模型天生就具备预测外部裁判如何为自身输出打分的潜在能力，所提出的SEE方法仅需160个样本（比强化学习基线少约31倍），通过“校准耦合强化学习+掩码蒸馏”的简短循环即可激发该能力，在保持回答质量的同时显著提升自我评估的校准水平。

    

    大语言模型越来越多地被其他模型评估，这引出一个自然的问题：模型能否预测裁判将如何为其自身输出打分？我们发现这种能力在很大程度上在任何针对性训练之前就已存在：通过少样本提示，基础模型在三个基准测试上预测外部裁判对开放式回答的多属性质量分数，其表现已远超随机水平。我们提出了自我评估激发方法（Self-Evaluation Elicitation, SEE），该方法通过一个简短的循环来显现这种潜在能力：该循环包含一个校准耦合的强化学习阶段（同时改进回答并预测裁判），随后是一个掩码蒸馏阶段（在保持回答不变的同时锐化预测能力）。仅从160个独特样本出发——比强化学习基线少约31倍——SEE在三个基准测试上提升了对保留集的校准能力，同时保持了回答质量。被激发出的自我评估能力具有高度的局域性。

    arXiv:2606.05122v2 Announce Type: replace  Abstract: Large language models are increasingly evaluated by other models, raising a natural question: can a model predict how a judge will score its own output? We find that the ability is largely present before any targeted training: prompted few-shot, a base model already predicts an external judge's multi-attribute quality scores on open-ended responses well above chance across three benchmarks. We introduce Self-Evaluation Elicitation (SEE), a method that surfaces this latent ability through a short cycle comprising a calibration-coupled reinforcement learning phase that improves the answer and predicts the judge, followed by a masked distillation phase that sharpens the prediction while leaving the answer untouched. From 160 unique examples, roughly 31x fewer than a reinforcement learning baseline, SEE improves held-out calibration across three benchmarks while preserving answer quality. The elicited self-evaluation is sharply localized
    
[^115]: RealClawBench：来自真实开发者-智能体会话的实时OpenClaw基准测试

    RealClawBench: Live OpenClaw Benchmarks from Real Developer-Agent Sessions

    [https://arxiv.org/abs/2606.03889](https://arxiv.org/abs/2606.03889)

    RealClawBench是一个基于真实OpenClaw开发者-智能体会话构建的实时基准测试框架，通过重建执行环境和确定性可验证评分器两大机制，将真实用户请求转化为281个可复现、可自动评分的任务，从而捕捉已部署智能体的真实使用分布与难度。

    

    智能体基准测试应当反映用户实际要求已部署智能体完成的任务，然而现有基准测试往往缺失真实开发者-智能体会话的关键真实性特征。我们提出了RealClawBench，一个基于真实OpenClaw会话构建的实时基准测试框架，旨在捕捉已部署智能体使用的分布、多样性和真实世界难度。真实的用户请求难以进行基准测试，因为它们通常依赖于本地执行环境、涉及隐式或表述不完整的意图，并且需要非平凡的验证。RealClawBench通过两个核心机制来解决这些挑战：重建的执行环境和确定性可验证评分器，二者共同将真实会话转化为可复现、可自动评分的任务。最终发布版本包含从更大的真实会话池中采样的281个可执行任务，同时保留了源分布，其最终任务与源分布之间的最大Jensen-Shannon散度……（摘要在此处被截断）

    arXiv:2606.03889v3 Announce Type: replace  Abstract: Agent benchmarks should reflect what users actually ask deployed agents to do, yet existing benchmarks often miss key realism properties of real developer-agent sessions. We introduce RealClawBench, a live benchmark framework built from real OpenClaw sessions to capture the distribution, diversity, and real-world difficulty of deployed agent use. Real user requests are challenging to benchmark because they often depend on local execution environments, involve implicit or underspecified intent, and require nontrivial verification. RealClawBench addresses these challenges with two core mechanisms: reconstructed execution environments and deterministic verifiable scorers, which together convert real sessions into reproducible, automatically scored tasks. The resulting release contains 281 executable tasks sampled from a much larger real-session pool while preserving the source distribution, with maximum final-vs-source Jensen-Shannon di
    
[^116]: CRAM：面向多模态持续指令微调的中心路由与自适应混合专家模型

    CRAM: Centroid-Routing and Adaptive MoE for Multimodal Continual Instruction Tuning

    [https://arxiv.org/abs/2606.02502](https://arxiv.org/abs/2606.02502)

    本文提出CRAM方法，通过中心路由和自适应混合专家模型，在缓解多模态持续指令微调中灾难性遗忘的同时，提升参数效率。

    

    多模态大语言模型（MLLMs）通过指令微调在共享生成框架下统一了异构的视觉-语言任务，然而现实世界部署需要持续的能力扩展，这使得多模态持续指令微调（MCIT）变得至关重要。现有方法要么使用共享参数集更新所有任务，要么为每个新任务分配专用模块。共享更新迫使异构任务相互竞争，导致已学能力的遗忘。相反，孤立扩展虽避免了干扰，但在长任务流中严重限制了参数效率。为解决这一困境，我们提出了CRAM（中心路由与自适应混合专家模型）。具体而言，通过将任务特定模式隔离到独立模块中，CRAM缓解了跨任务的灾难性遗忘。为进一步提升参数效率，我们利用自适应秩实例化来识别现有专家之间的能力差距。

    arXiv:2606.02502v2 Announce Type: replace  Abstract: Multimodal Large Language Models (MLLMs) unify heterogeneous vision-language tasks under a shared generative framework via instruction tuning, yet real-world deployment demands continuous capability expansion, making Multimodal Continual Instruction Tuning (MCIT) essential. Existing methods either update all tasks with a shared parameter set or allocate dedicated modules for each new task. Shared updates force heterogeneous tasks to compete, causing forgetting of learned capabilities. Conversely, isolated expansion prevents interference but severely limits parameter efficiency over long task streams. To address this dilemma, we propose CRAM (Centroid-Routing and Adaptive MoE). Specifically, by isolating task-specific patterns into independent modules, CRAM mitigates catastrophic forgetting across tasks. To further boost parameter efficiency, we utilize adaptive-rank instantiation to identify the capability gap between existing expert
    
[^117]: 针对长文本输出评估的LLM-as-a-Judge基准测试

    Benchmarking LLM-as-a-Judge for Long-Form Output Evaluation

    [https://arxiv.org/abs/2606.01629](https://arxiv.org/abs/2606.01629)

    该论文提出了LongJudgeBench基准，用于系统评估LLM评判者在多样化真实场景和评判协议下对长文本输出进行评估的可靠性。

    

    随着大语言模型（LLM）越来越多地被用于长文本生成，可靠地评估长文本输出已成为一个关键挑战。LLM-as-a-judge（以大语言模型作为评判者）为人工评估提供了一种可扩展的替代方案，然而其在长文本输出评估中的可靠性仍未得到充分研究：现有的元评估基准主要关注短文本输出。与短文本评估相比，长文本评估不仅仅是输出长度的问题；它通常要求评判者进行更复杂的文档级评估，包括整体结构组织、任务相关的覆盖面与深度、跨章节一致性以及特定场景的质量标准。在这项工作中，我们提出了LongJudgeBench，这是一个全面的基准，用于在多样化的真实世界场景和评判协议下评估LLM评判者对长文本输出的评估能力。我们系统地评估了广泛的LLM评判者，涵盖多个基础模型和不同的评判设置。我们的

    arXiv:2606.01629v4 Announce Type: replace  Abstract: As large language models (LLMs) are increasingly used for long-form generation, reliably evaluating long-form outputs has become a critical challenge. LLM-as-a-judge offers a scalable alternative to human evaluation, yet its reliability in long-form output evaluation remains underexamined: existing meta-evaluation benchmarks focus mainly on short-form outputs. Compared with short-form evaluation, long-form evaluation is not merely a matter of output length; it often requires judges to make more complex document-level assessments of overall organization, task-relevant coverage and depth, cross-section consistency, and scenario-specific quality criteria. In this work, we introduce LongJudgeBench, a comprehensive benchmark for evaluating LLM judges on long-form outputs across diverse real-world scenarios and judging protocols. We systematically evaluate a broad range of LLM judges, covering multiple base models and judging settings. Our
    
[^118]: DiffuSent：面向方面级情感分析的统一扩散框架

    DiffuSent: Towards a Unified Diffusion Framework for Aspect-Based Sentiment Analysis

    [https://arxiv.org/abs/2606.01323](https://arxiv.org/abs/2606.01323)

    提出非自回归扩散框架DiffuSent，将方面级情感分析的全部七个子任务统一建模为边界去噪扩散过程，并配合对比去噪训练策略，有效解决了多词方面词与观点词的边界不敏感及重复预测问题。

    

    方面级情感分析（ABSA）包含七个不同的子任务，每个子任务关注不同的抽取要素。尽管生成式模型在统一方面级情感分析中的成功已被证明，但现有方法通常依赖自回归的逐token生成方式，未能把握方面词和观点词的整体信息，导致边界不敏感的问题，在多词方面词和观点词的场景下尤为明显。为解决这些问题，我们提出了DiffuSent，一种非自回归的扩散框架，它将所有ABSA子任务系统地建模为边界去噪扩散过程，在噪声状态上逐步细化边界。此外，我们引入了一种对比去噪训练策略，有效解决了扩散过程所引入的细微变化导致的重复预测问题。大量实验（涵盖7个子任务×4个数据集共28种设置）表明

    arXiv:2606.01323v2 Announce Type: replace  Abstract: Aspect-Based Sentiment Analysis (ABSA) encompasses seven distinct subtasks, each focusing on different extracted elements. Despite the proven success of generative models in unified aspect sentiment analysis, existing approaches often rely on auto-regressive token-by-token generation without grasping the whole information of the aspect and opinion terms, resulting in boundary insensitivity, particularly in context of multi-word aspect and opinion terms. To address these issues, we present DiffuSent, a non-auto-regressive diffusion framework that systematically formulates all ABSA subtasks as boundary denoising diffusion processes, progressively refining boundaries over noisy states. Furthermore, we introduce a contrastive denoising training strategy which effectively address duplicate predictions with subtle variations introduced by diffusion process. Extensive experiments across 28 settings (7 subtasks x 4 datasets) demonstrate that
    
[^119]: 使用项目反应理论审计大语言模型基准测试

    Auditing LLM Benchmarks with Item Response Theory

    [https://arxiv.org/abs/2605.30504](https://arxiv.org/abs/2605.30504)

    本文提出基于项目反应理论的指标，能以95%的精确率从七个LLM基准测试中识别错误标注样本，并揭示奖励模型偏重风格偏好而非事实知识、且可能存在基准污染的问题。

    

    大语言模型基准测试的标签在发布时即被固定，并被静默地传播到下游基准测试中，错误也随之传递。我们提出了一种基于项目反应理论的指标，利用114个模型的响应，在七个偏好和多项选择基准测试中，以95%的精确率从排名前200的样本中识别出可能被错误标注的样本，其表现优于监督分类器。我们将这些错误追溯到机械化的标注启发式方法、从源数据集原封不动继承的上游标注错误，以及本质上模糊、不存在合理单一标签的样本。同样的模型拟合还揭示了奖励模型更专注于风格偏好而非事实知识，并识别出一个前沿奖励模型，它与检测到的错误标签的一致率达到78%，而同类模型仅为38%，这与基准测试污染或针对特定基准的过度优化相一致。

    arXiv:2605.30504v2 Announce Type: replace  Abstract: LLM benchmark labels are frozen at release and silently propagated into downstream benchmarks, errors and all. We introduce an Item Response Theory-based indicator that surfaces likely mislabels at 95% precision in the top 200 examples across seven preference and multiple-choice benchmarks using responses from 114 models, outperforming a supervised classifier. We trace these errors to mechanical labeling heuristics, upstream annotation mistakes inherited unchanged from source datasets, and fundamentally ambiguous items without a defensible single label. The same model fit reveals that reward models specialize in stylistic preference rather than factual knowledge, and identifies one frontier reward model that agrees with detected mislabels at 78% accuracy versus 38% for its peers, consistent with benchmark contamination or benchmark-specific over-optimization.
    
[^120]: LongDS-Bench：论长时程智能体数据分析的失败

    LongDS-Bench: On the Failure of Long-Horizon Agentic Data Analysis

    [https://arxiv.org/abs/2605.30434](https://arxiv.org/abs/2605.30434)

    提出LongDS基准，基于真实Kaggle笔记本构建68个长时程多轮数据分析任务，揭示当前最先进模型在维护和演变分析状态方面存在严重缺陷，最佳模型平均准确率仅48.45%，且长时程错误占失败案例的52%–69%。

    

    现实世界的数据分析本质上是迭代式的，然而现有的基准测试大多评估孤立的或短交互的任务，未能检验智能体在长时程中跟踪不断演变的分析上下文的能力。我们提出了LongDS，这是一个面向长时程、多轮数据分析的基准，要求智能体维护、更新、恢复和组合不断演变的分析状态。LongDS包含从真实Kaggle笔记本构建的68个任务，涵盖地球科学、商业和教育等六个领域，共2,225轮。任务围绕状态演变模式（例如反事实扰动、回滚、多状态组合）设计，平均依赖跨度为11.3轮。通过对五个最先进模型的评估，我们发现表现最好的模型平均准确率仅为48.45%，性能从早期到后期轮次下降近47个百分点，长时程错误占失败案例的52%–69%。

    arXiv:2605.30434v2 Announce Type: replace-cross  Abstract: Real-world data analysis is inherently iterative, yet existing benchmarks mostly evaluate isolated or short interactive tasks, leaving agents' ability to track evolving analytical context over long horizons untested. We introduce LongDS, a benchmark for long-horizon, multi-turn data analysis where agents must maintain, update, restore, and compose evolving analytical states. LongDS comprises 68 tasks constructed from real-world Kaggle notebooks, spanning 2,225 turns across six domains including Geoscience, Business, and Education. Tasks are designed around state-evolution patterns (e.g., counterfactual perturbation, rollback, multi-state composition), with an average dependency span of 11.3 turns. Evaluating five state-of-the-art models, we find that the best model reaches only 48.45% average accuracy, performance drops nearly 47 points from early to late turns, and long-horizon errors account for 52%--69% of failures. Further 
    
[^121]: 人类标签变异作为稳定信号：通过跨标注者偏好优化学习标注者特定的解释行为

    Human Label Variation as Stable Signal: Learning Annotator-Specific Explanation Behavior via Cross-Annotator Preference Optimization

    [https://arxiv.org/abs/2605.28802](https://arxiv.org/abs/2605.28802)

    本文提出跨标注者偏好优化（CAPO）方法，通过将目标标注者的解释与其他标注者对同一输入的有效但特异性较低的标注进行对比训练，使大语言模型能够学习并复现特定标注者的解释行为。

    

    自由文本解释通过揭示标注者决策背后的推理和偏好，将人类标签变异（HLV）的研究拓展到标签分歧之外。我们研究大型语言模型（LLM）能否学习并复现这种标注者特定的标签解释行为。我们使用两个句子对任务——自然语言推理和释义判断——每个任务各有四名标注者，首先分析标注者是否表现出稳定的个体模式。我们发现，由于强烈的输入内容效应，这种模式在单次标注层面较为微弱，但在经过输入内容消减和标注者层面聚合后变得可检测。随后，我们比较了提示和监督微调（SFT）基线方法，并提出了跨标注者偏好优化（CAPO），该方法将目标标注者的回答与同一输入下其他有效但目标特异性较低的标注进行对比。实验表明，提示方法的能力有限……（摘要内容在此处截断）

    arXiv:2605.28802v2 Announce Type: replace  Abstract: Free-text explanations extend human label variation (HLV) beyond label disagreement by revealing the reasoning and preferences behind annotators' decisions. We study whether large language models (LLMs) can learn and reproduce such annotator-specific label-explanation behavior. Using two sentence-pair tasks with four annotators each -- natural language inference and paraphrase judgment -- we first analyze whether annotators exhibit stable individual patterns. We find that such patterns are weak at the single-annotation level due to strong input-content effects, but become detectable after input-content reduction and annotator-level aggregation. We then compare prompting and supervised fine-tuning (SFT) baselines and propose cross-annotator preference optimization (CAPO), which contrasts a target annotator's response with other valid but less target-specific annotations for the same input. Experiments show that prompting is limited an
    
[^122]: 披着羊皮的狼：联邦RAG中的定向路由劫持

    A Wolf in Sheep's Clothing: Targeted Routing Hijacking in Federated RAG

    [https://arxiv.org/abs/2605.28112](https://arxiv.org/abs/2605.28112)

    论文揭示了联邦RAG系统中的一种新型“路由劫持”攻击：恶意客户端通过伪造语义画像劫持目标查询，在三种主流路由架构中均能引发证据缺失、投毒、错误答案和幻觉等严重后果，且现有防御无法有效应对。

    

    联邦检索增强生成对隐私敏感应用极具吸引力，因为完整的本地语料库始终保留在客户端。因此，路由必须依赖客户端提供的语义画像，这为操纵创造了新的机会。我们提出了路由劫持，这是一种路由阶段的攻击，恶意客户端伪造其语义画像以吸引目标查询，即使其底层数据并不相关。我们证明这一漏洞非常严重。在三种代表性的FedRAG路由架构中，路由劫持都能持续地将目标查询错误路由，并导致下游的干扰和故障，包括证据缺失、投毒、错误答案和幻觉。在一项受控的MedQA-USMLE压力测试中，我们进一步表明，被投毒的检索证据可以误导不同规模的模型，导致错误答案、幻觉和谄媚性失败。现有防御手段无法消除这一威胁。

    arXiv:2605.28112v2 Announce Type: replace-cross  Abstract: Federated Retrieval-Augmented Generation (FedRAG) is attractive for privacy-sensitive applications because full local corpora remain on clients. As a result, routing must rely on client-provided semantic profiles, creating a new opportunity for manipulation. We introduce Routing Hijacking, a routing-stage attack in which a malicious client forges its profile to attract target queries despite having irrelevant underlying data. We show that this vulnerability is severe. Across three representative FedRAG routing architectures, Routing Hijacking consistently misroutes target queries and leads to downstream disruptions and failures, including missing evidence, poisoning, incorrect answers, and hallucinations. In a controlled MedQA-USMLE stress test, we further show that poisoned retrieved evidence can mislead models across scales, leading to incorrect answers, hallucinations, and sycophantic failures. Existing defenses do not close
    
[^123]: LLM智能体能否在权力不对称对话中反映社会认知效应？

    Do LLM Agents Mirror Socio-Cognitive Effects in Power-Asymmetric Conversations?

    [https://arxiv.org/abs/2605.17694](https://arxiv.org/abs/2605.17694)

    研究发现LLM智能体在权力不对称对话中会展现出类似人类的社会认知效应（如语言协调、权威偏见），这既可能促成理想行为，也可能导致对不安全请求的顺从。

    

    权力差异通过已被充分记录的社会认知效应塑造着人类交流，包括语言协调、代词使用、权威偏见和有害顺从。我们研究大型语言模型（LLM）在被赋予高地位或低地位角色设定时是否表现出类似行为。我们使用来自不同职业的角色设定，模拟多轮权力不对称对话（如校长-教师、法官-律师），并测量以下四个方面：语言协调、代词使用、说服成功率以及对不安全请求的顺从度。我们的结果表明，LLM表现出权力的关键社会认知效应，尽管存在细微差异和变异性，这将模拟互动与理想行为和不安全行为联系起来。

    arXiv:2605.17694v3 Announce Type: replace  Abstract: Power differences shape human communication through well documented socio cognitive effects, including language coordination, pronoun usage, authority bias, and harmful compliance. We examine whether large language models (LLMs) exhibit similar behaviors when assigned high or low status personas. Using personas from diverse professions, we simulate multi turn, power asymmetric dialogues (e.g., principal teacher, justice lawyer) and measure (i) language coordination, (ii) pronoun usage, (iii) persuasion success, and (iv) compliance with unsafe requests. Our results show that LLMs show key socio-cognitive effects of power, albeit with nuances and variability, linking simulated interactions to both desirable and unsafe behaviors.
    
[^124]: SkillSafetyBench：评估面向技能攻击面下的智能体安全性

    SkillSafetyBench: Evaluating Agent Safety under Skill-Facing Attack Surfaces

    [https://arxiv.org/abs/2605.12015](https://arxiv.org/abs/2605.12015)

    该论文提出 SkillSafetyBench 基准，通过 155 个覆盖 6 大风险领域的对抗性案例，首次系统评估了隐藏在技能指导、本地文件等非用户输入中的攻击面，发现此类攻击可稳定诱发大语言模型智能体的不安全行为。

    

    可复用技能正在成为扩展大语言模型智能体的常见接口，它将程序性指导与对文件、工具、记忆和执行环境的访问打包在一起。然而，这种模块化引入了现有安全评估大多忽视的攻击面：即使用户请求是良性的，不安全的影响也可能存在于技能指导、本地工件或执行环境文件中，从而引导智能体做出不安全的操作。我们提出了 SkillSafetyBench，一个用于评估此类面向技能安全故障的可运行基准。SkillSafetyBench 包含 155 个对抗性案例，涵盖 47 个任务、6 个风险领域和 30 个安全类别，每个案例均通过特定于案例的基于规则的验证器进行评估。使用多个 CLI 智能体和模型后端的实验表明，非用户攻击能够持续诱发不安全行为，且在不同领域、攻击方法和脚手架-模型组合下呈现出截然不同的失败模式。

    arXiv:2605.12015v3 Announce Type: replace-cross  Abstract: Reusable skills are becoming a common interface for extending large language model agents, packaging procedural guidance with access to files, tools, memory, and execution environments. However, this modularity introduces attack surfaces that are largely missed by existing safety evaluations: even when the user request is benign, unsafe influence may reside in skill guidance, local artifacts, or execution-environment files that steer the agent toward unsafe actions. We present SkillSafetyBench, a runnable benchmark for evaluating such skill-facing safety failures. SkillSafetyBench includes 155 adversarial cases across 47 tasks, 6 risk domains, and 30 safety categories, each evaluated with a case-specific rule-based verifier. Experiments with multiple CLI agents and model backends show that non-user attacks can consistently induce unsafe behavior, with distinct failure patterns across domains, attack methods, and scaffold-model 
    
[^125]: 心理上强烈，计算上隐形：大型语言模型生成能引发社会比较的帖子，却无法检测到它们

    Psychologically Potent, Computationally Invisible: LLMs Generate Social-Comparison-Eliciting Posts They Fail to Detect

    [https://arxiv.org/abs/2605.01017](https://arxiv.org/abs/2605.01017)

    本研究构建了小红书社会比较基准，发现LLM能生成引发社会比较的帖子，但基于提示的检测器难以稳定识别该信号，揭示生成与检测能力的不对称性。

    

    我们引入了小红书社会比较读者引发基准（XHS-SCoRE），这是一个基于读者视角的基准，用于检测纯文本的小红书（RedNote）帖子是否从第一人称读者视角引发向上、向下或无明确的社会比较。该任务针对一种具有社会意义的关系性、行为性真实信号，该信号不能简化为情感。在提示式LLM分类器和监督式中文编码器中，我们发现了一致的生成-检测不匹配：该信号在领域内是可文本学习的，但对基于提示的分类并不稳健。提示式LLM分类器表现出稳定的失败，特别是对引发比较的帖子进行中和，以及模型特定的方向性偏差。一项受控试点显示，LLM生成的小红书风格帖子可以改变感知地位和与比较相关的情感，即使基于提示的相同构建检测仍然脆弱。XHS-SCoRE贡献...

    arXiv:2605.01017v3 Announce Type: replace  Abstract: We introduce Xiaohongshu Social Comparison Reader Elicitation (XHS-SCoRE), a reader-grounded benchmark for detecting whether text-only Xiaohongshu (RedNote) posts elicit Upward, Downward, or Neutral/no clear social comparison from a first-person reader perspective. The task targets a socially meaningful relational, behaviorally real signal not reducible to sentiment. Across prompted LLM classifiers and supervised Chinese encoders, we find a consistent generation-detection mismatch: the signal is textually learnable in-domain, but not robustly accessible to prompt-based classification. Prompted LLM classifiers show stable failures, especially neutralization of comparison-eliciting posts and model-specific directional skew. A controlled pilot shows that LLM-generated Xiaohongshu-style posts can shift perceived standing and comparison-related affect even when prompt-based detection of the same construct remains fragile. XHS-SCoRE contri
    
[^126]: G-Loss：图引导的语言模型微调

    G-Loss: Graph-Guided Fine-Tuning of Language Models

    [https://arxiv.org/abs/2604.25853](https://arxiv.org/abs/2604.25853)

    提出了一种图引导的损失函数G-Loss，通过结合半监督标签传播与文档相似度图来捕捉全局语义结构，引导预训练语言模型学习更具判别性和鲁棒性的嵌入表示。

    

    传统的损失函数，包括用于微调BERT等预训练语言模型的交叉熵、对比损失、三元组损失和监督对比损失，仅在局部邻域内运作，未能考虑全局语义结构。我们提出了G-Loss，这是一种图引导的损失函数，它结合了半监督标签传播来利用嵌入流形中的结构关系。G-Loss构建了一个能够捕捉全局语义关系的文档相似度图，从而引导模型学习更具判别性和鲁棒性的嵌入表示。我们在涵盖关键下游分类任务的五个基准数据集上评估了G-Loss：MR（情感分析）、R8和R52（主题分类）、Ohsumed（医学文档分类）以及20NG（新闻分类）。在大多数实验设置中，G-Loss收敛更快，并能产生语义连贯的嵌入空间……

    arXiv:2604.25853v4 Announce Type: replace  Abstract: Traditional loss functions, including cross-entropy, contrastive, triplet, and su pervised contrastive losses, used for fine-tuning pre-trained language models such as BERT, operate only within local neighborhoods and fail to account for the global semantic structure. We present G-Loss, a graph-guided loss function that incorporates semi-supervised label propagation to use structural relationships within the embedding manifold. G-Loss builds a document-similarity graph that captures global semantic relationships, thereby guiding the model to learn more discriminative and robust embeddings. We evaluate G-Loss on five benchmark datasets covering key downstream classification tasks: MR (sentiment analysis), R8 and R52 (topic categorization), Ohsumed (medical document classification), and 20NG (news categorization). In the majority of experimental setups, G-Loss converges faster and produces semantically coherent embedding spaces, result
    
[^127]: 为什么所有大语言模型都痴迷于日本文化？论大语言模型隐藏的文化与地区偏见

    Why are all LLMs Obsessed with Japanese Culture? On the Hidden Cultural and Regional Biases of LLMs

    [https://arxiv.org/abs/2604.21751](https://arxiv.org/abs/2604.21751)

    本文提出了一个涵盖24种语言的文化相关开放问题数据集CROQ，通过评估发现大语言模型在回答文化问题时存在明显的地区偏见（尤其偏爱日本），且低资源语言提示下的输出多样性显著低于高资源语言。

    

    大语言模型（LLMs）在文化覆盖面和文化能力方面存在局限，在某些情况下还会表现出特定的文化偏见。尽管先前的研究已经考察过大语言模型的文化能力，但尚无研究专门调查它们在一般文化相关问题中的地区偏好。在这项工作中，我们基于“文化相关开放问题”的综合分类体系提出了一个新数据集CROQ，其中的问题以24种语言提供。我们通过提示大语言模型回答CROQ中的问题并提供一个样本位置来对其进行评估。结果表明，与以往关于文化偏见的研究相反，大语言模型在回答中明显倾向于日本等国家。此外，我们的结果显示，当使用英语或其他高资源语言进行提示时，大语言模型往往提供更多样化的输出；相反，低资源语言则表现出更强的倾向性，其回答更多聚焦于共同……

    arXiv:2604.21751v2 Announce Type: replace  Abstract: LLMs have limitations when it comes to cultural coverage and competence, and in some cases, show specific cultural biases. Although prior studies have examined the cultural capabilities of LLMs, none have specifically investigated their regional preferences in generic culture-related questions. In this work, we propose a new dataset based on a comprehensive taxonomy of Culture-Related Open Questions (CROQ), with questions available in 24 languages. We evaluate LLMs by prompting them to answer questions from CROQ and provide a sample location. The results show that, contrary to previous cultural bias work, LLMs show a clear tendency towards countries such as Japan in their answers. Moreover, our results show that when prompting in languages such as English or other high-resource ones, LLMs tend to provide more diverse outputs. Low-resource languages, on the other hand, show more inclinations towards answering questions highlighting co
    
[^128]: 选择、标注、评估：自然语言处理中的主动测试

    Select, Label, Evaluate: Active Testing in NLP

    [https://arxiv.org/abs/2603.21840](https://arxiv.org/abs/2603.21840)

    该论文形式化了NLP中的主动测试框架，通过选择最具信息量的测试样本进行标注，在18个数据集上的实验表明可将标注成本降低高达95%，同时模型性能估计与完整测试集的准确度差异保持在1%以内。

    

    人工标注的成本和时间仍然是自然语言处理（NLP）领域的重大瓶颈，其中测试数据标注尤为昂贵，因为可靠的模型评估对低错误率、高质量标签有严格要求。传统方法需要标注整个测试集，导致巨大的资源需求。主动测试是一种选择最具信息量的测试样本进行标注的框架。在给定标注预算的情况下，它旨在选出最能估计模型性能的子集，同时最小化成本和人力投入。在本工作中，我们形式化了NLP中的主动测试，并在涵盖4种不同NLP任务的18个数据集和4种嵌入策略上对现有方法进行了广泛的基准测试。实验表明，标注量最多可减少95%，而性能估计的准确度与完整测试集的差异在1%以内。

    arXiv:2603.21840v2 Announce Type: replace  Abstract: Human annotation cost and time remain significant bottlenecks in Natural Language Processing (NLP), with test data annotation being particularly expensive due to the stringent requirement for low-error and high-quality labels necessary for reliable model evaluation. Traditional approaches require annotating entire test sets, leading to substantial resource requirements. Active Testing is a framework that selects the most informative test samples for annotation. Given a labeling budget, it aims to choose the subset that best estimates model performance while minimizing cost and human effort. In this work, we formalize Active Testing in NLP and we conduct an extensive benchmarking of existing approaches across 18 datasets and 4 embedding strategies spanning 4 different NLP tasks. The experiments show annotation reductions of up to 95%, with performance estimation accuracy difference from the full test set within 1%. Our analysis reveal
    
[^129]: 大型推理模型难以跨文字系统迁移参数化知识

    Large Reasoning Models Struggle to Transfer Parametric Knowledge Across Scripts

    [https://arxiv.org/abs/2603.17070](https://arxiv.org/abs/2603.17070)

    该研究发现大型推理模型的跨语言知识迁移失败主要由文字系统差异（而非语言或语系）导致，并提供源语言关键实体及合成SFT训练样本可显著缓解这一问题。

    

    在这项工作中，我们分析了现代大型推理LLM在跨语言知识迁移方面的不足。我们证明，所观察到的知识迁移差距主要是一种文字系统障碍。首先，我们对思维模型在两个包含世界各地本地知识的数据集（ECLeKTic和MultiLoKo）上的表现进行了观察性数据分析。我们的回归分析表明，在控制模型能力和问题难度后，文字系统是否匹配——而非语言或语系——是知识迁移失败的主要预测因素。我们通过向LLM提供问题中关键实体的源语言形式进一步验证了这一发现，发现这对跨文字系统的问题带来了不成比例的改善。随后我们假设这些LLM在测试时可以进行更好的推理。为了评估这一点，我们开发了一个合成数据生成管道，用于设计SFT样本，以鼓励模型更好地推理关于tra（原文在此处截断）

    arXiv:2603.17070v2 Announce Type: replace  Abstract: In this work, we analyze shortcomings in cross-lingual knowledge transfer in large, modern reasoning LLMs. We demonstrate that the perceived gap in knowledge transfer is primarily a script barrier. First, we conduct an observational data analysis on the performance of thinking models on two datasets with local knowledge from around the world, ECLeKTic and MultiLoKo. Our regression analysis shows that script match - not language or family - is the primary predictor of knowledge transfer failure once model capability and question difficulty are accounted for. We further this finding by providing the LLMs with the key entities of the questions in their source language and find that this disproportionately improves cross-script questions. We then posit that these LLMs could be reasoning better at test-time. To evaluate this, we develop a synthetic generation pipeline to design SFT samples to encourage the model to better reason about tra
    
[^130]: Transformer预测与人类句子处理的分歧：对一致性吸引效应的全面分析

    Diverging Transformer Predictions for Human Sentence Processing: A Comprehensive Analysis of Agreement Attraction Effects

    [https://arxiv.org/abs/2603.16574](https://arxiv.org/abs/2603.16574)

    该研究基于惊讶度机制系统评估了十一个自回归Transformer模型在英语一致性吸引效应上的表现，发现模型预测在介词短语配置上与人类阅读时间数据一致，但在宾语提取关系从句配置上性能显著下降、模型间预测分歧明显且均无法复制人类的不对称干扰模式，从而表明当前Transformer模型不能解释人类的形态句法处理过程。

    

    Transformer是计算语言学中几乎所有最先进语言模型的基础架构，但其作为人类句子处理模型的认知适切性仍存在争议。在这项工作中，我们使用基于惊讶度的联结机制，在比以往研究更全面的英语一致性吸引配置集合上，系统评估了十一个不同规模和架构的自回归Transformer模型。我们的实验结果好坏参半：尽管Transformer的预测在介词短语配置上总体与人类阅读时间数据一致，但在宾语提取关系从句配置上性能显著下降。在后一种情况下，不同模型之间的预测也出现明显分歧，且没有任何模型能够成功复制人类中观察到的不对称干扰模式。我们得出结论：当前的Transformer模型无法解释人类的形态句法处理，而且对……的评估（原文摘要至此截断）

    arXiv:2603.16574v2 Announce Type: replace  Abstract: Transformers underlie almost all state-of-the-art language models in computational linguistics, yet their cognitive adequacy as models of human sentence processing remains disputed. In this work, we use a surprisal-based linking mechanism to systematically evaluate eleven autoregressive transformers of varying sizes and architectures on a more comprehensive set of English agreement attraction configurations than prior work. Our experiments yield mixed results: While transformer predictions generally align with human reading time data for prepositional phrase configurations, performance degrades significantly on object-extracted relative clause configurations. In the latter case, predictions also diverge markedly across models, and no model successfully replicates the asymmetric interference patterns observed in humans. We conclude that current transformer models do not explain human morphosyntactic processing, and that evaluations of
    
[^131]: 近朱者赤：大语言模型如何回应黑暗三联征人格特质

    The Company You Keep: How LLMs Respond to Dark Triad Traits

    [https://arxiv.org/abs/2603.04299](https://arxiv.org/abs/2603.04299)

    本研究系统分析大语言模型对表达黑暗三联征人格特质的用户提示的回应，发现尽管所有模型以纠正性行为为主，部分模型仍会产生强化或模棱两可的输出，凸显了构建能够可靠检测并应对用户从良性升级为有害请求的更安全对话系统的必要性。

    

    大语言模型通常表现出高度附和的对话风格，也被称为AI谄媚性。当与反映负面社会倾向的用户提示交互时，这种模式可能成为问题，存在放大有害行为的风险。我们使用一个精心构建的数据集，研究大语言模型如何回应表达不同程度黑暗三联征特质（马基雅维利主义、自恋和精神病态）的用户提示。我们的分析揭示了模型之间的系统性差异：虽然所有模型主要表现出纠正性行为，但部分模型会产生强化性或模棱两可的输出。模型行为还随特质严重程度和响应情感而变化。这些发现凸显了对更安全对话系统的需求，这类系统能够可靠地检测并应对用户从良性请求升级为有害请求的情况。

    arXiv:2603.04299v5 Announce Type: replace  Abstract: LLMs often exhibit highly agreeable conversational styles, also known as AI sycophancy. This pattern may become problematic when interacting with user prompts that reflect negative social tendencies, risking the amplification of harmful behavior. We examine how LLMs respond to user prompts expressing varying degrees of Dark Triad traits (Machiavellianism, Narcissism, and Psychopathy) using a curated dataset. Our analysis reveals systematic differences across models: while all models predominantly exhibit corrective behavior, some generate reinforcing or ambivalent output. Model behavior further varies with severity level and response sentiment. These findings highlight the need for safer conversational systems that can reliably detect and respond to users escalating from benign to harmful requests.
    
[^132]: 从泄露的想法到私密推理：控制大型推理模型的自言自语

    From Leaky Thoughts to Private Reasoning: Controlling What LRMs Say to Themselves

    [https://arxiv.org/abs/2602.24210](https://arxiv.org/abs/2602.24210)

    该论文提出SFT数据集与分阶段解码策略，通过提升大型推理模型在推理过程中的指令遵循能力，有效减少推理轨迹中的隐私泄露。

    

    大型推理模型（LRMs）产生的推理轨迹（RTs）通常包含敏感信息。这些泄露的想法难以控制，且经常违反明确的隐私指令。由于推理轨迹可能通过提示注入攻击被暴露，这成为对用户隐私的直接威胁。我们将此视为一个可控性问题：由于隐私指令本身就是指令，提高推理轨迹中的指令遵循（IF）能力为减少隐私泄露提供了直接途径。为此，我们引入了一个SFT数据集，用于教模型在整个推理过程中遵循一般指令，并提出了一种简单的解码策略——分阶段解码，该方法使用独立的LoRA适配器将推理轨迹与答案生成解耦，以最大化各组件的指令遵循能力。我们在来自两个模型系列的六个模型（1.7B-14B参数）上，通过两个指令遵循基准和两个隐私基准评估了我们的方法。

    arXiv:2602.24210v4 Announce Type: replace  Abstract: Large reasoning models (LRMs) produce reasoning traces (RTs) that often contain sensitive information. These leaky thoughts are difficult to control and frequently violate explicit privacy directives. Because RTs can be exposed through prompt injection attacks, this becomes a direct privacy risk to the user. We approach this as a controllability problem: since privacy directives are themselves instructions, improving instruction-following (IF) within the RT provides a direct path to reducing privacy leaks. To this end, we introduce an SFT dataset that teaches models to follow general instructions throughout their reasoning process, and propose Staged Decoding, a simple decoding strategy that decouples RT and answer generation using separate LoRA adapters to maximize IF of each component. We evaluate our approach on six models from two families (1.7B-14B parameters), across two IF benchmarks and two privacy benchmarks. Our method yiel
    
[^133]: FENCE：面向金融领域的多模态越狱攻击检测数据集

    FENCE: A Financial and Multimodal Jailbreak Detection Dataset

    [https://arxiv.org/abs/2602.18154](https://arxiv.org/abs/2602.18154)

    FENCE是一个面向金融应用的双语多模态越狱检测数据集，基于它训练的检测器在分布内数据上达到99%的准确率，有效弥补了金融领域越狱检测资源的匮乏。

    

    越狱攻击对大型语言模型（LLM）和视觉语言模型（VLM）的部署构成了重大风险。VLM尤其脆弱，因为它们同时处理文本和图像，从而产生了更广泛的攻击面。然而，可用于越狱检测的资源十分匮乏，尤其是在金融领域。为填补这一空白，我们提出了FENCE，这是一个双语（韩语-英语）多模态数据集，用于训练和评估金融应用中的越狱检测器。FENCE通过将金融相关查询与基于图像的威胁相配对，强调领域真实性。对商业和开源VLM的实验揭示了一致的脆弱性：GPT-4o显示出可测量的攻击成功率，而开源模型则表现出更大的暴露风险。在FENCE上训练的基线检测器在分布内数据上达到了99%的准确率，并在外部基准测试中保持了强劲的性能，凸显了该数据集的稳健性。

    arXiv:2602.18154v3 Announce Type: replace  Abstract: Jailbreaking poses a significant risk to the deployment of Large Language Models (LLMs) and Vision Language Models (VLMs). VLMs are particularly vulnerable because they process both text and images, creating broader attack surfaces. However, available resources for jailbreak detection are scarce, particularly in finance. To address this gap, we present FENCE, a bilingual (Korean-English) multimodal dataset for training and evaluating jailbreak detectors in financial applications. FENCE emphasizes domain realism through finance-relevant queries paired with image-grounded threats. Experiments with commercial and open-source VLMs reveal consistent vulnerabilities, with GPT-4o showing measurable attack success rates and open-source models displaying greater exposure. A baseline detector trained on FENCE achieves 99 percent in-distribution accuracy and maintains strong performance on external benchmarks, underscoring the dataset's robustn
    
[^134]: CoFrGeNet：用于语言生成的连分数架构

    CoFrGeNet: Continued Fraction Architectures for Language Generation

    [https://arxiv.org/abs/2601.21766](https://arxiv.org/abs/2601.21766)

    受连分数启发，本文提出CoFrGeNet新架构，其组件能以远少于原有参数量的方式替代Transformer中的多头注意力和前馈网络，并可即插即用，几乎无需改动现有的训练与推理流程。

    

    Transformer可以说是语言生成的首选架构。在本文中，受连分数的启发，我们为生成式建模引入了一类新的函数类。实现这一函数类的架构族被命名为CoFrGeNets——连分数生成网络。我们基于该函数类设计了新颖的架构组件，这些组件可以替代Transformer块中的多头注意力和前馈网络，同时所需的参数量要少得多。我们推导了自定义的梯度公式，与使用基于PyTorch的标准梯度相比，能够更准确、更高效地优化所提出的组件。我们的组件是即插即用的替代品，只需对已为基于Transformer的模型建立的训练或推理流程做极少改动，因此我们的方法很容易融入大型工业工作流程中。我们在两种截然不同的Transformer架构上进行了实验……

    arXiv:2601.21766v5 Announce Type: replace  Abstract: Transformers are arguably the preferred architecture for language generation. In this paper, inspired by continued fractions, we introduce a new function class for generative modeling. The architecture family implementing this function class is named CoFrGeNets - Continued Fraction Generative Networks. We design novel architectural components based on this function class that can replace Multi-head Attention and Feed-Forward Networks in Transformer blocks while requiring much fewer parameters. We derive custom gradient formulations to optimize the proposed components more accurately and efficiently than using standard PyTorch-based gradients. Our components are a plug-in replacement requiring little change in training or inference procedures that have already been put in place for Transformer-based models thus making our approach easy to incorporate in large industrial workflows. We experiment on two very different transformer archit
    
[^135]: 超越兔子洞：绘制QAnon（匿名者Q）激进化带来的关系伤害图谱

    Beyond the Rabbit Hole: Mapping the Relational Harms of QAnon Radicalization

    [https://arxiv.org/abs/2601.17658](https://arxiv.org/abs/2601.17658)

    本文通过分析12747个来自r/QAnonCasualties支持社群的个人故事，构建计算流程提取主题特征并聚类为六种激进化人格画像，首次揭示了不同激进化模式与特定情感伤害（如愤怒、厌恶、恐惧、悲伤）之间的对应关系，填补了阴谋论研究中对信徒身边人所受关系伤害的关注空白。

    

    针对阴谋论的大规模计算研究一直仅聚焦于信徒本人的线上行为，而其身边亲近之人所遭受的伤害却鲜有研究关注。本文通过分析来自r/QAnonCasualties（一个为“失去”亲人至阴谋论信念的人们提供支持的在线社群）的12747个故事来填补这一空白。我们设计了一个计算流程，从个人叙事中提取细粒度的主题特征，并将其聚类为六种连贯的激进化人格画像，随后借助大语言模型辅助的情感检测与回归建模，将这些画像与叙述者所报告的情感代价联系起来。研究发现，人格画像能够有效预测特定的情感伤害：被视为有意意识形态选择的激进化与愤怒和厌恶相关，而以个人与认知崩溃为特征的人格画像则对应恐惧和悲伤。这项工作提供了一个基于实证的计算（框架）……

    arXiv:2601.17658v2 Announce Type: replace  Abstract: Large-scale computational research on conspiracy theories has focused exclusively on believers' online behavior, leaving the harm experienced by those closest to them under-examined. This paper bridges this gap by analyzing 12747 stories from r/QAnonCasualties, an online support group for people who have ``lost'' someone to conspiracy beliefs. We design a computational pipeline to extract fine-grained thematic traits from personal narratives and cluster them into six coherent radicalization personas, which we then link to the emotional toll reported by narrators via LLM-assisted emotion detection and regression modeling. We find that personas are meaningful predictors of specific emotional harms: radicalization perceived as a deliberate ideological choice is associated with anger and disgust, while personas marked by personal and cognitive collapse correspond to fear and sadness. This work provides an empirically grounded computation
    
[^136]: 通过知识性经验学习对齐智能体世界模型

    Aligning Agentic World Models via Knowledgeable Experience Learning

    [https://arxiv.org/abs/2601.13247](https://arxiv.org/abs/2601.13247)

    提出WorldMind框架，通过综合环境反馈自主构建符号化世界知识库，使LLM智能体世界模型无需昂贵再训练即可遵循物理法则、避免物理幻觉。

    

    当前的大型语言模型（LLMs）存在一个关键的模态脱节问题：它们拥有海量的语义知识，却缺乏遵循物理世界不变法则的程序性基础。因此，尽管这些智能体隐式地充当着世界模型，它们的模拟常常受到物理幻觉的困扰——生成逻辑上合理但物理上无法执行的计划。现有的对齐策略主要依赖资源密集型的训练或微调，试图将动态的环境规则压缩进静态的模型参数中。然而，这种参数化封装本质上是僵硬的，若不进行持续且昂贵的再训练，难以适应物理动力学的开放式变化。为弥合这一差距，我们提出了WorldMind，一个通过综合环境反馈来自主构建符号化世界知识库的框架。具体而言，它统一了过程经验……

    arXiv:2601.13247v2 Announce Type: replace  Abstract: Current Large Language Models (LLMs) exhibit a critical modal disconnect: they possess vast semantic knowledge but lack the procedural grounding to respect the immutable laws of the physical world. Consequently, while these agents implicitly function as world models, their simulations often suffer from physical hallucinations-generating plans that are logically sound but physically unexecutable. Existing alignment strategies predominantly rely on resource-intensive training or fine-tuning, which attempt to compress dynamic environmental rules into static model parameters. However, such parametric encapsulation is inherently rigid, struggling to adapt to the open-ended variability of physical dynamics without continuous, costly retraining. To bridge this gap, we introduce WorldMind, a framework that autonomously constructs a symbolic World Knowledge Repository by synthesizing environmental feedback. Specifically, it unifies Process Ex
    
[^137]: 通过大语言模型表示的内在维度追踪不同语言现象的复杂度特征

    Tracing the complexity profiles of different linguistic phenomena through the intrinsic dimension of LLM representations

    [https://arxiv.org/abs/2601.03779](https://arxiv.org/abs/2601.03779)

    该研究发现大语言模型表示的内在维度可作为语言复杂度的有效标记，并列/从属、右分支/中心嵌套、无歧义/歧义修饰等经典语言学复杂度对比在六个LLM的各层中均一致地体现为ID差异，且不同对比的出现位置和峰值阶段各不相同。

    

    我们探索大语言模型（LLM）表示的内在维度（ID）作为语言复杂度的标记。具体而言，我们检验模型各层之间的ID差异是否能够反映（心理）语言学中已确立的著名复杂度对比：并列结构 vs. 从属结构、右分支结构 vs. 中心嵌套结构、以及无歧义修饰 vs. 歧义修饰。我们在六个不同LLM上的实验结果表明，这些复杂度对比一致地反映在ID差异中，更复杂的语言现象会引发更高的ID特征值。值得注意的是，不同的复杂度对比在模型不同层的位置出现ID差异，并在不同阶段达到峰值。使用表示相似性和层剪枝的进一步实验证实了这些趋势。我们得出结论：内在维度是LLM中语言复杂度的有效标记，它揭示了不同LLM之间相似的语言处理步骤，并且它有潜力区分不同的语言现象。

    arXiv:2601.03779v3 Announce Type: replace  Abstract: We explore intrinsic dimension (ID) of LLM representations as a marker of linguistic complexity. Specifically, we test whether ID differences across model layers reflect well-known complexity contrasts established in (psycho)linguistics: coordination vs. subordination, right-branching vs. center-embedding, and unambiguous vs. ambiguous attachment. Our results on six different LLMs show that these contrasts are consistently reflected in ID differences, with more complex phenomena eliciting higher ID profiles. Notably, ID differences emerge at different points across layers for different contrasts, also reaching their peaks at different stages. Further experiments using representational similarity and layer pruning confirm the trends. We conclude that ID is a useful marker of linguistic complexity in LLMs, that it points to similar linguistic processing steps across disparate LLMs, and that it has the potential to differentiate between
    
[^138]: 安全的不稳定性：随机种子与温度如何暴露大语言模型不一致的拒绝行为

    The Instability of Safety: How Random Seeds and Temperature Expose Inconsistent LLM Refusal Behavior

    [https://arxiv.org/abs/2512.12066](https://arxiv.org/abs/2512.12066)

    该研究揭示大语言模型的安全拒绝决策在随机种子和温度变化下并不稳定，18-28%的有害提示词会出现“拒绝”与“配合”之间的决策翻转，且温度越高稳定性越差，表明单次安全评估无法真实反映模型的安全对齐水平。

    

    当前大语言模型的安全评估依赖于单次测试，隐含地假设模型响应是确定性的，并能代表模型的安全对齐水平。我们通过研究安全拒绝决策在不同随机种子和温度设置下的稳定性来挑战这一假设。我们在20种采样配置（4种温度 × 5个随机种子）下，对来自三个系列的四个指令微调模型（Llama 3.1 8B、Qwen 2.5 7B、Qwen 3 8B、Gemma 3 12B）在876个有害提示词上进行了测试，发现18-28%的提示词表现出决策翻转——即模型在某些配置下拒绝回答，而在其他配置下则予以配合——具体比例因模型而异。我们的安全稳定性指数（SSI）显示，更高的温度会显著降低决策稳定性（Friedman卡方 = 396.81，p < 0.001），温度内平均SSI从温度0.0时的0.977下降至温度1.0时的0.942。我们在……（原文摘要在此处被截断）

    arXiv:2512.12066v3 Announce Type: replace-cross  Abstract: Current safety evaluations of large language models rely on single-shot testing, implicitly assuming that model responses are deterministic and representative of the model's safety alignment. We challenge this assumption by investigating the stability of safety refusal decisions across random seeds and temperature settings. Testing four instruction-tuned models from three families (Llama 3.1 8B, Qwen 2.5 7B, Qwen 3 8B, Gemma 3 12B) on 876 harmful prompts across 20 sampling configurations (4 temperatures x 5 seeds), we find that 18-28% of prompts exhibit decision flips--the model refuses in some configurations but complies in others--depending on the model. Our Safety Stability Index (SSI) reveals that higher temperatures significantly reduce decision stability (Friedman chi-squared = 396.81, p < 0.001), with mean within-temperature SSI dropping from 0.977 at temperature 0.0 to 0.942 at temperature 1.0. We validate findings acro
    
[^139]: OmniFusion：通过模块化融合实现同时多语言多模态翻译

    OmniFusion: Simultaneous Multilingual Multimodal Translations via Modular Fusion

    [https://arxiv.org/abs/2512.00234](https://arxiv.org/abs/2512.00234)

    OmniFusion提出一种端到端的模块化融合方法，将多模态基础模型与翻译大语言模型结合，实现低延迟的同时多语言多模态翻译。

    

    开源纯文本翻译大语言模型（LLM）在语言覆盖范围和翻译质量方面已取得显著进展。然而，这些模型只能用于语音翻译（ST）的级联流水线中，即先执行自动语音识别，再进行翻译。这引入了额外的延迟，在同声语音翻译（SimulST）中尤为关键，并且使模型无法利用多模态上下文（例如图像），而图像可以帮助消除歧义。预训练多模态基础模型（MMFM）已具备跨多种模态的强大感知和推理能力，但通常缺乏专用翻译大语言模型所具有的多语言覆盖和专门翻译性能。为了构建有效的多模态翻译系统，我们提出了一种端到端方法，将多模态基础模型与翻译大语言模型进行融合。我们引入了一种新颖的融合策略，连接隐藏状态……

    arXiv:2512.00234v3 Announce Type: replace  Abstract: There has been significant progress in open-source text-only translation large language models (LLMs) with better language coverage and quality. However, these models can be only used in cascaded pipelines for speech translation (ST), performing automatic speech recognition first followed by translation. This introduces additional latency, which is particularly critical in simultaneous ST (SimulST), and prevents the model from exploiting multimodal context, such as images, which can aid disambiguation. Pretrained multimodal foundation models (MMFMs) already possess strong perception and reasoning capabilities across multiple modalities, but generally lack the multilingual coverage and specialized translation performance of dedicated translation LLMs. To build an effective multimodal translation system, we propose an end-to-end approach that fuses MMFMs with translation LLMs. We introduce a novel fusion strategy that connects hidden s
    
[^140]: 学习单个 Token 以替代大语言模型中的长系统提示词

    Learning a Single Token to Replace Long System Prompts in LLMs

    [https://arxiv.org/abs/2511.23271](https://arxiv.org/abs/2511.23271)

    该论文提出一种轻量级训练框架，通过学习单个“行为等价 Token（[BE]）”替代长系统提示词，在不更新模型权重、不使用辅助压缩模型和标注数据的情况下，实现高达 3000 倍的提示词压缩并保留约 98% 的下游行为效果。

    

    长系统提示词被广泛用于引导大语言模型（LLM）的行为，但在推理时反复处理这些提示词效率低下，且会消耗宝贵的上下文预算。这引出了一个核心问题：能否仅用一个极简的学习表示来保留长系统提示词的行为效果？为此，我们提出了一个轻量级训练框架，用于学习单个行为等价 Token（[BE]）。该框架首先通过重构任务训练 [BE] 编码原始系统提示词的语义内容，然后将提示词的下游行为蒸馏到这个单一 Token 中。重要的是，我们的方法无需更新预训练的大语言模型权重，无需辅助压缩模型，也无需带标签的响应数据。在三个数据集上的实证评估表明，用单个 [BE] Token 替换长提示词可实现高达 3000 倍的提示词压缩率，同时保留约 98% 的下游行为效果。

    arXiv:2511.23271v2 Announce Type: replace  Abstract: Long system prompts are widely used to steer Large Language Models (LLMs), but repeatedly processing them at inference time is inefficient and consumes valuable context budget. This motivates a central question: can the behavioral effect of a long system prompt be retained using only a minimal learned representation? To enable this, we propose a lightweight training framework that learns a single Behavior-Equivalent Token ([BE]). The framework first trains [BE] to encode the semantic content of the original system prompt via reconstruction, and then distills the prompt's downstream behavior into this single token. Importantly, our method requires no update to the pretrained LLM weights, no auxiliary compression models, and no labeled responses. Empirical evaluations on three datasets show that replacing long prompts with a single [BE] token yields up to a $3000\times$ prompt compression ratio, while retaining about 98% of the downstr
    
[^141]: SMRC：将大语言模型与学生推理对齐以实现数学错误纠正

    SMRC: Aligning Large Language Models with Student Reasoning for Mathematical Error Correction

    [https://arxiv.org/abs/2511.14684](https://arxiv.org/abs/2511.14684)

    该论文提出SMRC方法，将学生数学推理建模为多步序贯决策问题并利用蒙特卡洛树搜索探索最优纠正路径，使大语言模型能够像教师一样系统地检测和纠正学生的数学推理错误。

    

    大语言模型（LLM）在解决数学问题时经常出现推理错误，如何自动检测并纠正这些错误已成为一个重要的研究方向。然而，现有方法主要关注模型内部的自纠错，这无法满足教育场景中所需的“教师式”纠正，即系统地引导和修订学生的解题过程。为了填补这一空白，我们提出了SMRC（学生数学推理纠正），这是一种将大语言模型与学生推理对齐的新颖方法。具体而言，SMRC将学生推理形式化为一个多步序贯决策问题，并引入蒙特卡洛树搜索（MCTS）来探索最优纠正路径。为了降低标注过程级奖励的成本，我们利用广度优先搜索（BFS）引导……（摘要原文在此处截断）

    arXiv:2511.14684v2 Announce Type: replace  Abstract: Large language models (LLMs) often make reasoning errors when solving mathematical problems, and how to automatically detect and correct these errors has become an important research direction. However, existing approaches \textit{mainly focus on self-correction within the model}, which falls short of the "teacher-style" correction required in educational settings, \textit{i.e.}, systematically guiding and revising a student' s problem-solving process. To address this gap, we propose \texttt{SMRC} (\textit{\underline{S}tudent \underline{M}athematical \underline{R}easoning \underline{C}orrection}), a novel method that aligns LLMs with student reasoning. Specifically, \texttt{SMRC} formulates student reasoning as a multi-step sequential decision problem and introduces Monte Carlo Tree Search (MCTS) to explore optimal correction paths. To reduce the cost of the annotating process-level rewards, we leverage breadth-first search (BFS) gui
    
[^142]: Think-at-Hard（难处深思）：面向推理能力提升的动态循环Transformer

    Think-at-Hard: Dynamic Looped Transformers for Improved Reasoning

    [https://arxiv.org/abs/2511.08577](https://arxiv.org/abs/2511.08577)

    针对循环Transformer中存在的“潜在过度思考”现象，提出TaH方法，利用轻量级神经决策器仅在可能出错的token处动态触发潜在迭代，从而在参数受限条件下将大语言模型的推理性能提升高达7.3%。

    

    提升大语言模型（LLM）的推理能力，尤其是在参数受限的条件下，对实际应用至关重要。循环Transformer通过执行多次潜在迭代来优化每个token的表示，超越了单次前向传播的能力。然而，我们识别出一种“潜在过度思考”现象：大多数token预测在第一次前向传播后就已经正确，但在后续迭代中有时反而被修改成错误。我们探究了选择性地跳过潜在迭代能否提升准确率，并通过一个先验迭代策略揭示了显著的潜力，该策略可将性能提升高达7.3%。受此启发，我们提出了Think-at-Hard（TaH），一种针对选择性迭代进行优化的循环Transformer。TaH采用一个轻量级神经决策器，仅在标准前向传播后可能出错的token上触发潜在迭代。在潜在迭代过程中，深度感知的低秩适应（LoRA）模块……

    arXiv:2511.08577v4 Announce Type: replace  Abstract: Improving the reasoning abilities of Large Language Models (LLMs), especially under parameter constraints, is crucial for real-world applications. Looped transformers address this by performing multiple latent iterations to refine each token beyond a single forward pass. However, we identify a latent overthinking phenomenon: most token predictions are already correct after the first pass, but are sometimes revised into errors in later iterations. We ask whether selectively skipping latent iterations can improve accuracy, and reveal significant potential with an oracle iteration policy that boosts performance by up to 7.3%. Motivated by this, we propose Think-at-Hard (TaH), a looped transformer optimized for selective iteration. TaH employs a lightweight neural decider to trigger latent iteration, only at tokens likely to be incorrect after the standard forward pass. During latent iterations, depth-aware Low-Rank Adaptation (LoRA) mod
    
[^143]: 用于预测重度抑郁症症状严重程度的多语言口语词汇特征分析

    Multilingual Lexical Feature Analysis of Spoken Language for Predicting Major Depression Symptom Severity

    [https://arxiv.org/abs/2511.07011](https://arxiv.org/abs/2511.07011)

    该研究基于来自英国、荷兰和西班牙467名参与者的多语言智能手机录音数据，利用可解释的线性混合效应模型识别出与抑郁症状严重程度相关的口语词汇特征，并通过机器学习验证了这些特征对PHQ-8评分预测的增益作用。

    

    背景：远程采集的口语语言可以为抑郁症症状严重程度提供客观、定期的指标。然而，迄今为止的研究主要使用非临床、横断面的书面语言以及可解释性有限的复杂机器学习方法。方法：我们使用线性混合效应模型，在RADAR-MDD研究的数据中识别与症状严重程度相关的可解释词汇特征，该数据包含来自英国、荷兰和西班牙467名参与者的5,846个智能手机录音以及患者健康问卷（PHQ-8）评分。随后，我们开发了机器学习模型，并通过嵌套交叉验证系统地评估了可解释的词汇特征或高维向量嵌入是否能比社会人口统计学和混杂因素特征提高PHQ-8预测的准确性。结果：抑郁症状严重程度与五个词汇特征相关，包括词语（原文在此处截断）

    arXiv:2511.07011v2 Announce Type: replace  Abstract: Background: Remotely captured spoken language could provide objective, regular indicators of depression symptom severity. However, research to date has largely used non-clinical, cross-sectional written language and complex machine learning (ML) approaches with limited interpretability. Methods: We used linear mixed-effect models to identify interpretable lexical features associated with symptom severity in data from the RADAR-MDD study that comprised 5,846 smartphone recordings and Patient Health Questionnaire (PHQ-8) scores from 467 participants in the UK, Netherlands and Spain. We then developed ML models and systematically assessed via nested cross-validation whether interpretable lexical features or high-dimensional vector embeddings improved the accuracy of PHQ-8 prediction over sociodemographic and confounding features. Results: Depression symptom severity was associated with five lexical features, including reductions in word
    
[^144]: 基于结构的角色扮演：从心理问卷生成合成的治疗师-来访者对话

    Roleplaying with Structure: Synthetic Therapist-Client Conversation Generation from Questionnaires

    [https://arxiv.org/abs/2510.25384](https://arxiv.org/abs/2510.25384)

    该论文提出SQPsych生成管线，在不泄露敏感数据的前提下，利用真实的结构化来访者档案和心理问卷生成合成治疗师-来访者对话语料库，经专家评估证明微调后的LLM在治疗师角色扮演上显著更优。

    

    大型语言模型（LLM）是心理健康领域合成数据生成的有前景的工具。然而，隐私政策及相关限制迫使以往的工作主要依赖通用信息。我们提出了一个通过LLM生成的合成治疗师-来访者对话综合语料库。我们构建了生成管线SQPsych（基于结构化问卷的心理治疗），它使用真实的结构化来访者档案和心理问卷，同时不泄露任何敏感数据。我们在生成的语料库SQPsychConv上微调了多种开源权重的LLM，并通过自动基准测试和由受过训练的心理治疗师参与的人类评估对其进行测试。我们发现标准基准测试无法充分体现我们数据集的优势，但专家评判表明，SQPsych使LLM在治疗师角色扮演方面的能力显著提升。专家们也一致更倾向于选择由我们的模型生成的治疗会谈内容……

    arXiv:2510.25384v2 Announce Type: replace  Abstract: Large Language Models (LLMs) are promising tools for synthetic data generation in mental health. However, privacy policies and restrictions forced previous work to rely mainly on generic information. We present a comprehensive corpus of synthetic therapist-client conversations generated through LLMs. We construct our generation pipeline, SQPsych (Structured Questionnaire-based Psychotherapy), which uses real structured client profiles and psychological questionnaires without leaking any sensitive data. We fine-tune various open-weight LLMs on our generated corpus, SQPsychConv , and test them through both automatic benchmarks and human evaluation with trained psychotherapists. We find that standard benchmarks do not adequately capture the strengths of our dataset, but expert judgment shows that SQPsych makes LLMs significantly better at therapist roleplaying. Experts also consistently prefer therapy sessions generated by our models co
    
[^145]: 量化低资源媒体中的情感偏见：孟加拉语新闻标题的大规模情感画像

    Quantifying Affective Bias in Low-Resource Media: Large-Scale Emotion Profiling of Bengali Headlines

    [https://arxiv.org/abs/2510.17252](https://arxiv.org/abs/2510.17252)

    该研究利用Gemma 3 4B零样本推理对30万条孟加拉语新闻标题进行大规模情感分析，发现愤怒、悲伤、失望和恐惧等负面情感在低资源语言新闻媒体中占主导地位，从而量化揭示了孟加拉语数字新闻中存在的情感偏见。

    

    新闻媒体不仅可以通过报道的事件影响读者，还可以通过呈现这些事件时所使用的情感基调来影响读者。这一问题在数字新闻环境中尤为重要，因为标题往往在读者打开完整文章之前就塑造了第一印象。本研究通过语料库层面的情感分析，考察了孟加拉语数字新闻中的情感框架。使用基于Gemma 3 4B的零样本推理，我们分析了300,000条孟加拉语新闻标题，以估计每个标题的主导情感和整体情感基调。结果显示，负面情感标签，特别是愤怒、悲伤、失望和恐惧，在所分析的语料库中频繁出现。一项针对200条人工审阅标题的小型试点验证表明，该模型能够提供有用的情感估计，尽管这些结果应被解读为计算估计值，而非完整的基准测试。

    arXiv:2510.17252v2 Announce Type: replace  Abstract: News media can influence readers not only through the events they report but also through the emotional tone used to present them. This issue is especially important in digital news environments, where headlines often shape first impressions before readers open the full article. This study examines affective framing in Bengali digital journalism through corpus level emotion analysis of news headlines. Using zero shot inference with Gemma 3 4B, we analyzed 300,000 Bengali news headlines to estimate the dominant emotion and overall affective tone of each headline. The results show that negative emotion labels, particularly anger, sadness, disappointment, and fear, appear frequently in the analyzed corpus. A small pilot validation on 200 manually reviewed headlines suggests that the model can provide useful emotion estimates, although the results should be interpreted as computational estimates rather than a complete benchmark. Based on
    
[^146]: PRISM：基于大语言模型的智能体式检索框架用于多跳问答

    PRISM: Agentic Retrieval with LLMs for Multi-Hop Question Answering

    [https://arxiv.org/abs/2510.14278](https://arxiv.org/abs/2510.14278)

    PRISM是一个基于大语言模型的智能体式检索框架，通过问题分析器、选择器和添加器三个专门智能体的迭代协作，以高精确率和高召回率检索证据，显著提升多跳问答性能。

    

    检索在多跳问答中扮演着核心角色，因为回答复杂问题需要收集多条证据。我们提出了PRISM，这是一个智能体式检索框架，它利用大语言模型在结构化循环中以高精确率和高召回率检索相关证据。PRISM将检索过程分解为三个专门的智能体：问题分析器将复杂查询分解为子问题，选择器为每个子问题识别最相关的上下文（专注于精确率），添加器引入任何缺失的证据（专注于召回率）。选择器与添加器之间的迭代交互产生了一个紧凑而全面的证据集合，既避免了脆弱的错误传播，也避免了嘈杂上下文的累积。它在过滤干扰性内容的同时实现了更高的检索准确率，使下游问答模型能够超越全上下文回答的准确率。

    arXiv:2510.14278v2 Announce Type: replace  Abstract: Retrieval plays a central role in multi-hop question answering (QA), where answering complex questions requires gathering multiple pieces of evidence. We propose PRISM, an agentic retrieval framework that leverages large language models (LLMs) in a structured loop to retrieve relevant evidence with high precision and recall. PRISM decomposes retrieval into three specialized agents: a Question Analyzer that breaks complex queries into sub-questions, a Selector that identifies the most relevant context for each sub-question (focusing on precision), and an Adder that brings in any missing evidence (focusing on recall). The iterative interaction between the Selector and Adder produces a compact yet comprehensive evidence set, avoiding both brittle error propagation and noisy context accumulation. It achieves higher retrieval accuracy while filtering out distracting content, enabling downstream QA models to surpass full-context answer acc
    
[^147]: OceanGym：面向水下具身智能体的基准测试环境

    OceanGym: A Benchmark Environment for Underwater Embodied Agents

    [https://arxiv.org/abs/2509.26536](https://arxiv.org/abs/2509.26536)

    OceanGym是首个面向水下具身智能体的综合性基准环境，涵盖八个真实任务领域和基于多模态大语言模型的统一智能体框架，实验揭示当前最先进的智能体与人类专家之间仍存在显著差距。

    

    我们提出了OceanGym，这是首个面向海洋水下具身智能体的综合性基准测试环境，旨在推动人工智能在最苛刻的真实环境之一中的发展。与陆地或空中领域不同，水下环境带来了极端的感知与决策挑战，包括低能见度和动态洋流，使得智能体的有效部署异常困难。OceanGym涵盖八个真实的任务领域，以及一个由多模态大语言模型（MLLM）驱动的统一智能体框架，该框架集成了感知、记忆和序列决策能力。智能体需要理解光学和声呐数据，自主探索复杂环境，并在这些恶劣条件下完成长程目标。大量实验表明，最先进的MLLM驱动智能体与人类专家之间存在显著差距，凸显了感知、规划等能力方面持续存在的困难。

    arXiv:2509.26536v3 Announce Type: replace  Abstract: We introduce OceanGym, the first comprehensive benchmark for ocean underwater embodied agents, designed to advance AI in one of the most demanding real-world environments. Unlike terrestrial or aerial domains, underwater settings present extreme perceptual and decision-making challenges, including low visibility, dynamic ocean currents, making effective agent deployment exceptionally difficult. OceanGym encompasses eight realistic task domains and a unified agent framework driven by Multi-modal Large Language Models (MLLMs), which integrates perception, memory, and sequential decision-making. Agents are required to comprehend optical and sonar data, autonomously explore complex environments, and accomplish long-horizon objectives under these harsh conditions. Extensive experiments reveal substantial gaps between state-of-the-art MLLM-driven agents and human experts, highlighting the persistent difficulty of perception, planning, and 
    
[^148]: 面向情境感知安全的多模态大语言模型解码引导

    Steering Multimodal Large Language Models Decoding for Context-Aware Safety

    [https://arxiv.org/abs/2509.19212](https://arxiv.org/abs/2509.19212)

    提出了轻量级、模型无关的解码框架SafeCoDe，通过对比真实与高斯噪声图像突出视觉敏感token，并结合场景级推理动态调整拒绝行为，从而平衡多模态大语言模型安全决策中的过度敏感与敏感不足问题。

    

    多模态大语言模型（MLLMs）越来越多地被部署到实际应用中，但其做出情境感知安全决策的能力仍然有限。现有方法往往难以平衡过度敏感（对良性查询的不合理拒绝）与敏感不足（漏检具有视觉依据的风险），导致安全对齐方面存在持续差距。为解决这一问题，我们提出了安全感知对比解码（SafeCoDe），这是一个轻量级且与模型无关的解码框架，能够基于多模态上下文动态调整token生成。SafeCoDe分两个阶段运行：（1）一种对比解码机制，通过对比真实图像与高斯噪声图像来突出对视觉上下文敏感的token；（2）一种全局感知的token调制策略，将场景级推理与token级调整相结合，根据预测的安全判定自适应地调整拒绝行为。大量实验……

    arXiv:2509.19212v2 Announce Type: replace  Abstract: Multimodal Large Language Models (MLLMs) are increasingly deployed in real-world applications, yet their ability to make context-aware safety decisions remains limited. Existing methods often fail to balance oversensitivity (unjustified refusals of benign queries) and undersensitivity (missed detection of visually grounded risks), leaving a persistent gap in safety alignment. To address this issue, we introduce Safety-aware Contrastive Decoding (SafeCoDe), a lightweight and model-agnostic decoding framework that dynamically adjusts token generation based on multimodal context. SafeCoDe operates in two stages: (1) a contrastive decoding mechanism that highlights tokens sensitive to visual context by contrasting real and Gaussian-noised images, and (2) a global-aware token modulation strategy that integrates scene-level reasoning with token-level adjustment to adapt refusals according to the predicted safety verdict. Extensive experime
    
[^149]: 传声筒游戏：评估统一模型中的语义漂移

    The Telephone Game: Evaluating Semantic Drift in Unified Models

    [https://arxiv.org/abs/2509.04438](https://arxiv.org/abs/2509.04438)

    该论文提出语义漂移协议（SDP）和平均累积漂移（MCD）指标，通过模拟传声筒游戏的多轮交替生成方式，首次量化了统一模型在理解与生成能力组合使用时出现的语义漂移问题，揭示了在孤立基准上表现优异的模型在跨任务一致性上可能严重失效。

    

    统一模型（UMs）将视觉理解（I2T）与图像生成（T2I）结合在单一框架中。我们聚焦于T2I和I2T，其中交叉一致性——模型所理解的内容应当能够生成出来——是统一化的承诺，也是组合这两种能力时的必要条件。然而，现有基准将两者孤立评估：FID/GenEval用于T2I；MME/MMBench用于I2T。我们证明这一差距影响重大：在这些基准上得分具有竞争力的模型，在理解与生成能力组合使用时可能严重失败，丢失实体、属性、空间关系和数量，从而导致语义漂移。为了量化这种漂移，我们提出了语义漂移协议（SDP），其灵感来源于传声筒游戏：从一段图像描述或一张图像出发，在多轮生成中交替进行I2T和T2I，并测量语义的保持程度。我们进一步提出平均累积漂移（MCD），一种基于嵌入的度量方法，用于衡量跨三个……

    arXiv:2509.04438v3 Announce Type: replace-cross  Abstract: Unified models (UMs) combine visual understanding (I2T) and generation (T2I) in a single framework. We focus on T2I and I2T, where cross-consistency---what a model understands, it should be able to generate---is a promise of unification and a necessity when composing both capabilities. Yet, existing benchmarks evaluate them in isolation: FID/GenEval for T2I; MME/MMBench for I2T. We show this gap is consequential: models scoring competitively on these benchmarks can fail severely when understanding and generation are composed, losing entities, attributes, spatial relations, and counts, resulting in semantic drift. To quantify drift, we introduce the Semantic Drift Protocol (SDP), inspired by the Telephone Game: starting from a caption or image, we alternate I2T and T2I over multiple generations and measure semantic preservation. We propose Mean Cumulative Drift (MCD), an embedding-based measure of content retention across three 
    
[^150]: 使用深度学习的《古兰经》学习者发音错误自动检测与纠正

    Automatic Pronunciation Error Detection and Correction of the Holy Quran's Learners Using Deep Learning

    [https://arxiv.org/abs/2509.00094](https://arxiv.org/abs/2509.00094)

    该论文提出了一套98%自动化的《古兰经》诵读数据构建流程，发布了848小时音频数据集（28.6万条标注语句）以及涵盖Tajweed规则的基准qdat_bench，实现了对《古兰经》学习者发音错误的自动检测与纠正。

    

    评估口语具有挑战性，而量化用于机器学习模型的发音指标更是难上加难。然而，对于《古兰经》而言，得益于穆斯林学者们建立的严谨诵读规则（Tajweed，泰吉维德），这一任务得以实现，使高效评估成为可能。尽管有这一优势，高质量标注数据的稀缺仍是一个重大障碍。在本工作中，我们通过引入以下内容来弥合这些差距：(1) 一套98%自动化的流程，用于生成高质量的《古兰经》数据集——包括从专业诵经师处收集诵读音频、使用我们微调的wav2vec2-BERT模型在停顿点（waqf）进行分割、对片段进行转录，以及通过我们新颖的Tasmeea算法进行转录验证；(2) 848小时音频（28.6万条标注语句）；(3) qdat_bench，一个涵盖音素、标音符号以及Tajweed规则（Ghunnah鼻音、Qalqalah弹音、Madd长音）的真实诵读基准数据集。

    arXiv:2509.00094v2 Announce Type: replace-cross  Abstract: Assessing spoken language is challenging, and quantifying pronunciation metrics for machine learning models is even harder. However, for the Holy Quran, this task is enabled by the rigorous recitation rules (Tajweed) established through the efforts of Muslim scholars, making highly effective assessment possible. Despite this advantage, the scarcity of high-quality annotated data remains a significant barrier. In this work, we bridge these gaps by introducing: (1) A 98% automated pipeline to produce high-quality Quranic datasets -- encompassing collection of recitations from expert reciters, segmentation at pause points (waqf) using our fine-tuned wav2vec2-BERT model, transcription of segments, and transcript verification via our novel Tasmeea algorithm; (2) 848 hours of audio (286K annotated utterances); (3) qdat_bench, a benchmark covering phonemes, diacritization, and Tajweed rules (Ghunnah, Qalqalah, Madd) on real recitation
    
[^151]: 超越罗塞塔石碑：泛化动力学中的统一力量

    Beyond the Rosetta Stone: Unification Forces in Generalization Dynamics

    [https://arxiv.org/abs/2508.11017](https://arxiv.org/abs/2508.11017)

    通过合成数据上的小型Transformer实验，本文揭示了跨语言知识迁移的关键在于表示统一性，这取决于信息性和可提取性，并提供了统一理论解释多语言模型中的迁移现象。

    

    大语言模型（LLMs）在跨语言知识迁移方面存在困难：当用某种语言询问训练中另一种语言表达的事实时，它们有时会产生幻觉。本研究引入了一个受控环境，通过在合成的多语言数据集上从头训练小型Transformer模型，来研究这一现象的原因和训练动态。根据（1）事实与其学习所用语言之间的相关性（信息性）和（2）语言识别的难易程度（可提取性），模型要么发展出跨语言的统一表示，要么发展出分离表示；只有表示统一时，事实才能跨语言迁移。基于这些见解，我们提出了一个统一视角，解释了一系列关于多语言LLM跨语言迁移的先前观察结果。我们的工作表明，受控环境可以揭示预训练的机制。

    arXiv:2508.11017v3 Announce Type: replace-cross  Abstract: Large language models (LLMs) struggle with cross-lingual knowledge transfer: they sometimes hallucinate when asked in one language about facts expressed in a different language during training. This work introduces a controlled setting to study the causes and training dynamics of this phenomenon by training small Transformer models from scratch on synthetic multilingual datasets. Depending on (1) the correlation between facts and the language they were learned in (informativeness), and (2) the ease of language identification (extractability), models either develop unified representations across languages or separate representations; only when representations are unified do facts transfer across languages. Based on these insights, we propose a unifying perspective which explains a range of prior observations concerning cross-lingual transfer in multilingual LLMs. Our work shows controlled settings can shed light on pre-training 
    
[^152]: 认知思维链：关于社会情境的结构化多模态推理

    Cognitive Chain-of-Thought (CoCoT): Structured Multimodal Reasoning about Social Situations

    [https://arxiv.org/abs/2507.20409](https://arxiv.org/abs/2507.20409)

    提出认知思维链框架，通过感知、情境、规范三个认知阶段结构化视觉语言模型的多模态社会情境推理，在意图消歧、心智理论、社会常识推理等多个任务上取得一致的显著提升。

    

    思维链提示能够帮助模型逐步思考。但朴素的CoT在基于视觉的社会任务中会失效，在这类任务中，模型必须同时进行感知、理解和判断，从而将感知与基于社会规范的推理连接起来。近期的工作已为多轮智能体规划和视觉问答引入了结构化推理方法，将任务分解为顺序的子目标。为了将这种方法扩展到单次多模态社会推理中，我们提出了认知思维链，这是一个通过三个受认知启发的阶段来构建视觉语言模型（VLM）推理的框架：感知（提取有依据的事实）、情境（推断情境状况）和规范（应用社会规范）。在多个不同任务上的评估，包括多模态意图消歧、多模态心智理论、社会常识推理和安全指令遵循，均显示出一致的改进（平均提升5.9%至4.6%）。

    arXiv:2507.20409v3 Announce Type: replace  Abstract: Chain-of-Thought (CoT) prompting helps models think step by step. But naive CoT breaks down in visually grounded social tasks, where models must perceive, understand, and judge all at once; bridging perception with norm-grounded reasoning. Recent work has introduced structured reasoning for multi-turn agent planning and visual QA, decomposing tasks into sequential sub-goals. To extend this to single-shot multimodal social reasoning, we introduce Cognitive Chain-of-Thought (CoCoT), a reasoning framework that structures vision-language-model (VLM) reasoning through three cognitively inspired stages: Perception (extract grounded facts), Situation (infer situations), and Norm (applying social norms). Evaluation across multiple distinct tasks such as multimodal intent disambiguation, multimodal theory of mind, social commonsense reasoning, and safety instruction following, shows consistent improvements (5.9% to 4.6% on average). We furthe
    
[^153]: 大语言模型的剪枝定律

    Pruning Laws for Large Language Models

    [https://arxiv.org/abs/2504.04342](https://arxiv.org/abs/2504.04342)

    提出了“剪枝定律”，一种将剪枝后性能与未剪枝性能和剪枝比例关联起来的简单可解释的缩放关系，可在多种模型、剪枝策略和任务上以小于7%的平均外推误差准确预测大语言模型剪枝后的性能。

    

    扩大模型参数和训练数据规模能持续提升大语言模型（LLM）的性能，但其代价是内存和计算需求的快速增长，这使得在资源受限的硬件上部署变得不可行。模型剪枝是一种广泛使用的压缩技术，通过移除冗余参数来降低推理成本。然而，剪枝对下游性能的影响仍然难以预测，通常只能通过代价高昂的经验性扫描来评估。为填补这一空白，我们提出了剪枝定律——一种简单且可解释的缩放关系，将剪枝后LLM的性能与其未剪枝性能和剪枝比例联系起来。在十个LLM（1.3B-30B参数）、一个20B混合专家模型、三种剪枝策略（非结构化、宽度和深度）以及八个多样化任务上，我们证明了剪枝定律具有强大的预测准确性（平均外推误差小于7%），并且结果可靠。

    arXiv:2504.04342v2 Announce Type: replace  Abstract: Scaling up model parameters and training data consistently improves the performance of large language models (LLMs), but at the cost of rapidly growing memory and compute requirements, which makes deployment on resource-limited hardware infeasible. Model pruning, a widely used compression technique, reduces inference costs by removing redundant parameters. However, its impact on downstream performance remains unpredictable and is typically assessed only through costly empirical sweeps. To address this gap, we introduce pruning laws, simple and interpretable scaling relations that connect a pruned LLM's post-pruning performance to its unpruned performance and pruning ratio. Across ten LLMs (1.3B-30B parameters), a 20B mixture-of-experts model, three pruning strategies (unstructured, width, and depth), and eight diverse tasks, we show that pruning laws achieve strong predictive accuracy (average extrapolation error less than 7%), relia
    
[^154]: PRISM：面向免训练多模态数据选择的自剪枝内在选择方法

    PRISM: Self-Pruning Intrinsic Selection Method for Training-Free Multimodal Data Selection

    [https://arxiv.org/abs/2502.12119](https://arxiv.org/abs/2502.12119)

    PRISM提出了一种免训练的多模态数据选择方法，通过首次揭示视觉特征分布的各向异性及其引发的全局语义漂移现象，在不依赖昂贵推理或训练的情况下高效剪除冗余指令数据，显著降低了数据选择的计算成本。

    

    视觉指令微调使预训练的多模态大语言模型（MLLMs）能够遵循人类指令，从而应用于真实世界场景。然而，这类数据集的快速增长引入了显著的冗余，导致计算成本大幅上升。现有的指令数据选择方法旨在修剪这种冗余，但主要依赖于计算开销高昂的技术，例如基于代理模型的推理或基于训练的度量指标。因此，这些选择过程本身所产生的巨大计算成本，往往会加剧它们本想解决的效率瓶颈，给MLLMs的可扩展且高效微调带来了重大挑战。为应对这一挑战，我们首先识别出一个关键但此前被忽视的因素：视觉特征分布中固有的各向异性。我们发现这种各向异性会引起“全局语义漂移”（Global Semantic Drift），而忽视……（摘要内容在此处被截断）

    arXiv:2502.12119v5 Announce Type: replace-cross  Abstract: Visual instruction tuning adapts pre-trained Multimodal Large Language Models (MLLMs) to follow human instructions for real-world applications. However, the rapid growth of these datasets introduces significant redundancy, leading to increased computational costs. Existing methods for selecting instruction data aim to prune this redundancy, but predominantly rely on computationally demanding techniques such as proxy-based inference or training-based metrics. Consequently, the substantial computational costs incurred by these selection processes often exacerbate the very efficiency bottlenecks they are intended to resolve, posing a significant challenge to the scalable and effective tuning of MLLMs. To address this challenge, we first identify a critical, yet previously overlooked, factor: the anisotropy inherent in visual feature distributions. We find that this anisotropy induces a \textit{Global Semantic Drift}, and overlooki
    
[^155]: 长话短说：基于2万部短片的故事级视频理解

    Long Story Short: Story-level Video Understanding from 20K Short Films

    [https://arxiv.org/abs/2406.10221](https://arxiv.org/abs/2406.10221)

    提出了目前最大的公开可用电影数据集SF20K，包含20,143部短片共3,582小时视频，通过业余短片规避数据泄露问题，推动故事级长视频理解研究。

    

    arXiv:2406.10221v3 公告类型：replace-cross 摘要：视觉-语言模型的最新发展显著推动了视频理解的进步。然而，现有的数据集和任务存在明显的局限性。大多数数据集仅限于事件有限、叙事狭窄的短视频。例如，包含教学视频和第一人称（自我中心）视角视频的数据集通常只描绘单一场景中一个人的活动。尽管现有的电影数据集提供了更丰富的内容，但它们通常仅限于短期任务，缺乏公开可用的视频，并且由于在大语言模型预训练中使用了商业电影的字幕及其他相关信息，经常面临数据泄露问题。为了解决上述局限性，我们提出了Short-Films 20K（SF20K），这是目前最大的公开可用电影数据集。SF20K由20,143部业余电影组成，总计3,582小时的视频，平均每部电影12分钟。我们还随数据集提供了SF20K-Test，这是一个手工标注的开放式问答（测试集）……

    arXiv:2406.10221v3 Announce Type: replace-cross  Abstract: Recent developments in vision-language models have significantly advanced video understanding. Existing datasets and tasks, however, have notable limitations. Most datasets are confined to short videos with limited events and narrow narratives. For example, datasets with instructional and egocentric videos often depict the activities of one person in a single scene. Although existing movie datasets offer richer content, they are often limited to short-term tasks, lack publicly available videos, and frequently encounter data leakage issues given the use of subtitles and other information about commercial movies during LLM pretraining. To address the above limitations, we propose Short-Films 20K (SF20K), the largest publicly available movie dataset. SF20K consists of 20,143 amateur films, amounting to 3,582 hours of video, with an average of 12 minutes per movie. We accompany this dataset with SF20K-Test, a manual, open-ended que
    
[^156]: 在高考基准测试上评估大型语言模型的性能

    Evaluating the Performance of Large Language Models on GAOKAO Benchmark. (arXiv:2305.12474v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.12474](http://arxiv.org/abs/2305.12474)

    本文介绍了一个基于高考考试问题的基准测试GAOKAO-Benchmark，用于评估大型语言模型在客观和主观问题方面的表现。通过对ChatGPT模型的评估，研究发现其在客观问题方面表现出色，同时也揭示了其不足之处和改进的方向。

    

    大型语言模型已经在各种自然语言处理任务中展示了出色的性能；然而它们在更具挑战性和领域特定的任务中的功效仍然不太清楚。本文介绍了GAOKAO-Benchmark（GAOKAO-Bench），这是一个直观的基准测试，它使用中国高考考试的题目作为测试样本，评估大型语言模型。为了尽可能地使评估结果与人类一致，我们设计了一种基于零-shot提示的方法，通过将问题分为主观和客观类型来分析模型的准确性和评分率。我们评估了ChatGPT模型在GAOKAO-Benchmark性能上的表现。我们的研究发现，ChatGPT模型在解决客观问题方面表现出色，同时也揭示了其不足之处和改进的方向。为了进一步审查模型的响应，我们加入了人类评估。总之，本研究为创建一个稳健的评估GAOKAO基准测试提供了贡献。

    Large language models have demonstrated remarkable performance across various natural language processing tasks; however, their efficacy in more challenging and domain-specific tasks remains less explored. This paper introduces the GAOKAO-Benchmark (GAOKAO-Bench), an intuitive benchmark that employs questions from the Chinese Gaokao examination as test samples for evaluating large language models.In order to align the evaluation results with humans as much as possible, we designed a method based on zero-shot prompts to analyze the accuracy and scoring rate of the model by dividing the questions into subjective and objective types. We evaluated the ChatGPT model on GAOKAO-Benchmark performance.Our findings reveal that the ChatGPT model excels in tackling objective questions, while also shedding light on its shortcomings and areas for improvement. To further scrutinize the model's responses, we incorporate human evaluations.In conclusion, this research contributes a robust evaluation ben
    

