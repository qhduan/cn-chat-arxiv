# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [IdeaAnchor: Teaching LLMs to Turn Literature into Research Ideas](https://arxiv.org/abs/2610.08781) | 提出IdeaAnchor范式，通过编码论文功能角色、关系及综合标准的结构化规格说明作为特权信号，结合示范、自蒸馏与强化学习训练大语言模型，使其学会将文献综合为研究思路。 |
| [^2] | [Sherpa: Teaching LLMs to Teach Adaptively](https://arxiv.org/abs/2610.08778) | Sherpa是一个多轮强化学习框架，通过模拟具有不同学习偏好的学生原型并直接最大化其学习成效，训练教师LLM自适应地调整教学策略，使受教学生的成绩平均提升20.5个百分点。 |
| [^3] | [AdvSim2Real : Training Web Agents Against Adaptive Prompt Injection in a Web World Model](https://arxiv.org/abs/2610.08773) | 提出 AdvSim2Real，在冻结的网络世界模型中让任务课程、注入攻击者与智能体共同演化，通过“成功翻转”对抗奖励机制训练出既更强又更能抵御自适应提示注入攻击的网络智能体。 |
| [^4] | [The Missing Minimal Pair: Stereotype Evaluation in LLMs](https://arxiv.org/abs/2610.08747) | 该论文提出双重最小对设置和基于互信息的评估指标，并结合数据增强框架，实现对大语言模型刻板印象偏见更可靠的多语言评估。 |
| [^5] | [Denoising Hierarchical Representations: Joint Continuous Diffusion for Language Modeling](https://arxiv.org/abs/2610.08738) | 提出层次化连续扩散语言模型（H-CDLM），通过并行扩散token本身及其粗粒度语义聚类等多种模态表示，并允许为各模态配置独立的采样器与调度，以极少的计算和参数开销显著提升连续扩散语言模型的性能。 |
| [^6] | [A Systematic Study of Semantic ID Spaces for Generative Information Retrieval](https://arxiv.org/abs/2610.08732) | 本文针对生成式信息检索中的核心问题“什么样的DocID才是好的DocID”，首次提出统一框架将乘积量化、残差量化及其混合变体纳入单一设计空间，系统研究了定义有效数值DocID的属性、指标与权衡，从而摆脱对昂贵下游评估的依赖，支持系统性分析与快速迭代。 |
| [^7] | [Holdout Best-of-N: Unbiased Evaluation and Its Cost](https://arxiv.org/abs/2610.08719) | 本文证明了仅当用于选择的分数数J小于总分数数K时才能对Best-of-N策略进行严格无偏评估，此时留出法估计量达到σ²/√K的最优无偏极小极大风险，而允许偏差可将速率提升至σ²/K。 |
| [^8] | [When Forgetting is not Catastrophic: On the Mechanics of Spurious Forgetting](https://arxiv.org/abs/2610.08718) | 该研究揭示了语言模型微调中虚假遗忘的力学机制——微调使旧表征沿共同方向偏移导致暂时的知识隐藏，归一化会撤销该偏移使知识自行恢复，而只有事实特定的累积变化才会造成真正的永久遗忘。 |
| [^9] | [Disentangling Paradigm, Identifier, and Decoding in Generative Retrieval](https://arxiv.org/abs/2610.08716) | 该论文通过控制变量实验首次解耦了生成式检索中范式、标识符和解码三个因素，发现仅解码方式就能使扩散检索器的Hit@1波动6.6至13.7个百分点，并提出单次评分解码方法，使模型一次性读取掩码标识符并按编码概率为文档打分，从而大幅提升扩散检索性能。 |
| [^10] | [Agreement Is Not Validity: Cross-Model LLM Consensus in Diagnosing Student Failure Modes in K-12 Math Tutoring Dialogue](https://arxiv.org/abs/2610.08703) | 该研究警示“一致”不等于“有效”：LLM之间的跨模型共识远高于其与人类的一致性，因此不能用模型间的高共识来替代人类验证以证明LLM诊断学生数学学习失败的有效性。 |
| [^11] | [A Systematic Study of Small Language Models on Abstract Reasoning Tasks](https://arxiv.org/abs/2610.08680) | 本文系统研究了小型语言模型在ARC-TGI抽象推理基准上的表现，发现尽管模型能取得较高的分布内准确率，但其技能获取对优化过程敏感、在各任务族间分布不均，且分布外性能急剧下降，表明模型可能只是拟合了特定分布的规律而未真正学到可迁移的规则。 |
| [^12] | [Same-Number Citation Swaps: Stress-Testing Jev as a Financial Evidence Judge](https://arxiv.org/abs/2610.08675) | 该论文通过对GPT-4.1-mini计算轨迹在相同数字单元格之间互换引用进行压力测试，发现Jev等概率证据验证器在金融报告数值大量重复的场景下仍会放过引用错误财务角色的计算并拒绝等价的有效引用，而显式列标签只能部分改善这一角色识别难题。 |
| [^13] | [Principled Under Pressure: Post-Training Decides Whether LLMs Act on Their Own Moral Judgment](https://arxiv.org/abs/2610.08670) | 该研究构建了涵盖五种压力类型的248个预注册场景面板，通过让模型同时以第一人称行动和第三人称评判的方式对照其自身道德判断，发现大语言模型在约五分之一的压力场景下会采取自己判定为错误的行为，且这种“言行不一”差距的大小取决于后训练配方。 |
| [^14] | [Evidence-Bound Reasoning: Neuro-Semantic Verification of Biomedical AI in Glioblastoma Radiogenomics](https://arxiv.org/abs/2610.08660) | 该论文提出了一种神经-语义验证框架，将放射组学测量转化为可寻址的证据记录和机器可校验的声明，从而可靠验证生物医学AI的解释中每条陈述是否真正有患者特异性证据支持。 |
| [^15] | [SquidAgent: Parallelize Wisely, Coordinate Efficiently](https://arxiv.org/abs/2610.08647) | 该论文提出SquidAgent，揭示了并行多智能体系统中“重新探索成本”与“对齐成本”这两项隐性开销，并据此推导出原则性决策准则：仅当并行化的关键路径成本加上这两项开销低于串行成本时，才应对该层进行并行化。 |
| [^16] | [Towards In-Parameter Memory Augmentation for Large Language Models](https://arxiv.org/abs/2610.08630) | 这是一篇综述，系统梳理了通过将可复用知识编码进模型参数、适配器等类参数对象并在推理时插入前向传播，从而在部署阶段以参数内记忆增强大语言模型的方法，并以参数放置等正交维度组织分类。 |
| [^17] | [InterCorrect: Intersection-Aware Correction of Demographic Model Merging for Fair ASR](https://arxiv.org/abs/2610.08604) | 该论文提出InterCorrect方法，通过人口统计感知的模型合并与针对多群体交叉的校正向量提升语音识别公平性，将最佳整体词错误率从7.38%降至5.13%。 |
| [^18] | [Generative AI translations in high-stakes emergency messaging](https://arxiv.org/abs/2610.08601) | 研究表明，在紧急信息翻译中使用特定语篇的提示词可显著提升生成式AI翻译的可理解性和可操作性，但人工修订在检测错误和伦理担责方面仍不可或缺。 |
| [^19] | [Incidental information contaminates patient notes and disrupts clinical reasoning in large language models](https://arxiv.org/abs/2610.08585) | 研究发现闲聊和背景语音等偶发信息会污染大语言模型生成的病历并偶尔被错误地用于临床，据此提出LLM临床推理与分心的双重编码假说。 |
| [^20] | [Have I Seen Enough? Frozen Video-Language Models Encode Evidence Readiness](https://arxiv.org/abs/2610.08560) | 该论文发现冻结的视频-语言模型内部已线性编码了一种由问题条件化、可跨基准泛化且与答案对错无关的“证据就绪度”信号，因此无需额外训练触发器即可判断流式视频问答中证据是否已充分到来。 |
| [^21] | [Latent space bias directions in LLMs capture confidence, not fairness](https://arxiv.org/abs/2610.08559) | 该研究揭示了大语言模型激活引导中的去偏方向实际上编码的是模型置信度而非偏见信息，其去偏效果只是降低模型置信度的副产品，从而解释了激活引导去偏技术泛化能力差的根本原因。 |
| [^22] | [DeltaTTT: Layerwise Optimization for Nonlinear Recurrent Memory](https://arxiv.org/abs/2610.08553) | 针对非线性循环记忆在测试时训练中难以优化、并行基线反而优于串行版本的问题，DeltaTTT提出用逐层学习替代联合内循环优化，为每层分配局部预测目标并通过状态依赖的delta规则更新，从而缓解非线性记忆的优化困难。 |
| [^23] | [How High Is 0.6? Floors, Ceilings, and Headroom in Interpretability Probing](https://arxiv.org/abs/2610.08544) | 该论文提出用“地板”（简单输入已能预测的水平）和“天花板”（完整输入能预测的水平）两个参照点以及二者之间的“余量”来校准可解释性探测得分，使探针分数具有明确、可比的含义，并证明余量在目标不依赖隐藏变量或输入不透露隐藏变量时消失。 |
| [^24] | [Toward Alignment Scaling Laws: A Framework and First Preregistered Measurements](https://arxiv.org/abs/2610.08540) | 该论文提出将对齐视为一族可测量的幂律缩放关系（B_r(N)=a_rN^alpha_r）的框架及首批预注册测量，并证明长期对齐状态由经修正风险中的最大指数而非平均值决定，指数大于1时将累积不可持续的对齐债务。 |
| [^25] | [Wiki-Talkie: Multilingual Benchmarking of Persona-Based Agents on Real-World Discussions](https://arxiv.org/abs/2610.08513) | 该论文提出了 Wiki-Talkie——首个基于维基百科讨论页真实对话、涵盖五种语言并配以源自真实用户社区画像的多语言基准数据集，用于评估角色化LLM智能体模拟人类交互的行为保真度。 |
| [^26] | [Language-model ratings of depression reflect the rater more than the patient](https://arxiv.org/abs/2610.08501) | 该研究通过对880个语言模型评分者的预注册实验发现，语言模型对抑郁症的评分更多反映评分模型自身的差异（解释30.0%的评分方差）而非患者的真实症状差异（仅10.5%），即使两个高精度模型平均也会对40%的参与者的筛查结果产生分歧。 |
| [^27] | [UNREAL: Unifying Retrieval and Long-Context with a Single Model](https://arxiv.org/abs/2610.08463) | UNREAL提出了一种模型原生的证据选择框架，直接从冻结LLM的内部表示中推导检索查询，以不到50万可训练参数统一了语料库检索与长上下文推理，并在多个基准上大幅超越最先进的检索-重排序系统。 |
| [^28] | [Agentic AutoRAG: RAG Pipeline Optimization through Reasoning-Driven Agents](https://arxiv.org/abs/2610.08452) | 该论文提出Agentic AutoRAG，一种利用LLM智能体进行多目标RAG超参数优化的方法，其核心创新在于通过诊断器将每次失败归因于检索或生成阶段，从而让优化器能够推理配置失败的原因并智能地指导后续搜索。 |
| [^29] | [Rethinking Cross-Tokenizer On-Policy Distillation: From Alignment Coverage to Supervision Reliability](https://arxiv.org/abs/2610.08448) | 该研究发现跨分词器在线策略蒸馏中扩大对齐覆盖并无必要——严格1:1对齐已覆盖大部分token，仅用共享词表中由学生选择的top-16子集计算反向KL即可媲美完整共享词表方法，表明监督可靠性比对齐覆盖更为关键。 |
| [^30] | [Knowing When Not to Answer: Cross-Domain and Multi-Turn Generalization of Latent Underspecification Signals](https://arxiv.org/abs/2610.08413) | 该论文构建了一个带轮次标签的多轮对话不可回答性基准与模拟用户评估框架，发现线性探针所捕捉的“信息缺失”信号能在共享同一不可回答性根源的数据集间稳健跨域迁移（AUROC 0.77–0.97），但不同类型不可回答性的表征边界会受词汇混淆、网络层级与坐标系选择的影响。 |
| [^31] | [Foresight-over-Graph: Reasoning Beyond Local Horizons for Knowledge Base Question Answering](https://arxiv.org/abs/2610.08388) | 提出前瞻感知的证据检索框架FoG，克服LLM图推理中逐跳贪心与束搜索剪枝的短视问题，避免关键证据分支被过早丢弃，提升知识库问答的可靠性。 |
| [^32] | [CoDe-LoRA: Mitigating the Orthogonality Dilemma in Continual Learning of LLMs via Knowledge Consolidation and Decoupling](https://arxiv.org/abs/2610.08312) | 提出无需回放的CoDe-LoRA方法，通过自适应零空间投影和语义路由将学习过程解耦为通用知识巩固与任务特定知识解耦，克服了正交参数隔离阻碍跨任务知识迁移的“正交困境”。 |
| [^33] | [Language Unalignability: Why Some Concepts Resist Cross-Cultural Benchmark Evaluation](https://arxiv.org/abs/2610.08303) | 本文提出“语言不可对齐性”概念，论证对于语用标记、敬语等一类概念，跨语言映射在原理上无法同时保持词汇忠实性与结构忠实性，从根本上质疑了多语言大模型评估所依赖的翻译同构假设。 |
| [^34] | [Memory Depth and Reconstructed Context Width: A Controlled Evaluation of Hierarchical Retrieval](https://arxiv.org/abs/2610.08300) | 该研究通过受控实验发现，在分层对话记忆检索中，扩大重构上下文宽度（从1K到4K）可使准确率显著提升10.11-17.98个百分点，而增加层次深度并无单调收益，且超过8-16K后性能趋于平台期，表明应优先采用大型连贯的上下文块而非逐级分层结构。 |
| [^35] | [STRUCTURALCOST: A controlled reading time dataset for modeling human sentence processing difficulty](https://arxiv.org/abs/2610.08208) | 该研究推出大规模阅读时间数据集STRUCTURALCOST，验证了人类阅读时间随主谓依存长度增加而上升，并揭示现有语言模型虽能反映这种预测性难度但低估了工作记忆导致的整合成本，为评估语言模型的认知合理性奠定了数据基础。 |
| [^36] | [Align, Then Correct: Training-Free Two-Stage Low-Rank Compensation for Extremely Quantized Large Language Models](https://arxiv.org/abs/2610.08164) | 该论文提出一种无需训练的两阶段闭式低秩补偿框架，先对齐层输出再校正残余误差，克服了现有低秩量化误差补偿在对称校准和仅二阶优化上的两大局限，大幅提升极端量化大语言模型的精度恢复能力。 |
| [^37] | [The Failure Is in the Readout: Fine-Grained Emotion Recognition Benchmarks Measure Elicitation, Not Perception](https://arxiv.org/abs/2610.08162) | 该研究发现，现成的视觉语言模型在细粒度情绪识别上其实并不逊于专门微调的模型，此前基准测得的失败源于生成式的答案引出方式，而非模型的感知能力不足。 |
| [^38] | [Symphony for Text Generation: Benchmarking Clinical Note Generation](https://arxiv.org/abs/2610.08161) | 该论文提出了包含300例多语言临床就诊记录的MedConv数据集，并构建了结合蕴含指标与大语言模型评判的受控临床评估框架，证明临床AI平台Corti的病历生成质量与领先商业环境式记录软件相当或更优，且其可配置API可针对特定文档需求灵活优化质量维度。 |
| [^39] | [Making COMET Comparable Across Scripts: Diagnosis and Correction of Tokeniser-Induced Script Bias in Indic MT Evaluation](https://arxiv.org/abs/2610.08159) | 该论文发现 COMET 评估指标因分词器存在文字偏差，导致印度语系不同文字系统间的分数不可比且排序准确性下降，并提出 COMET-QN 方法以精确消除跨文字分数范围不兼容的问题。 |
| [^40] | [Penalty-Framed No-Valid-Option MCQA: Analyzing LLM Abstention under Invalid Choices](https://arxiv.org/abs/2610.08153) | 该论文提出“惩罚框架下的无有效选项多选题问答”这一新评测设定，并通过基于正确回答的条件分析方法，揭示了大语言模型的高答题准确率并不能保证其在所有选项均无效时可靠地选择弃答。 |
| [^41] | [Conversation Is a Two-Body Problem: Dyadic Evaluation of Full-Duplex Dialogue Models](https://arxiv.org/abs/2610.08125) | 提出DyaFDB框架，让两个全双工对话模型在指定角色及合作或冲突目标下直接对话，并由外部评判器对双方同时评分，弥补了传统单边评估只能覆盖“二体问题”一半的不足。 |
| [^42] | [Natural Language Questions as an Interface for Knowledge Graphs: QRAKEN Graph Distillation and Semantic Self-Healing](https://arxiv.org/abs/2610.08095) | QRAKEN提出了一种无需训练的神经符号流水线，通过离线图蒸馏生成TTQL图谱证据来引导LLM生成SPARQL查询，并借助确定性的语法与数据模型检查实现迭代式语义自修复，在Text2SPARQL挑战赛上取得了严格的F1最佳成绩。 |
| [^43] | [SAGE: Semantic Anchor-Guided Evolution for Grounded Medical QA Data Synthesis](https://arxiv.org/abs/2610.08093) | SAGE提出了一种数据合成框架，利用MeSH等轻量级公开分类体系作为语义锚点，通过迭代交替进行原子与关联合成，使小型本地部署模型也能从极少种子数据生成高质量的医学问答训练数据。 |
| [^44] | [DirectSpeech2LLM: A Simple End-to-End Framework to Mitigate Prompt Overfitting in Speech-LLMs](https://arxiv.org/abs/2610.08085) | 提出DirectSpeech2LLM端到端框架，仅用ASR数据训练即可保持LLM的指令遵循能力，实现对语音翻译和情感识别等未见任务的零样本泛化。 |
| [^45] | [POLAR: Ontology-Guided Risk Prevention for Tool-Calling LLM Agents](https://arxiv.org/abs/2610.08082) | POLAR是一个通过结构化两层本体评估操作可逆性的防护栏框架，能在工具调用LLM智能体执行高风险操作前将其剪除并提供可审计的结构化判定，但其收益因任务域和智能体能力而异。 |
| [^46] | [Self-Retrospection Distillation: Turning Post-hoc Experiences into Prior Foresight](https://arxiv.org/abs/2610.08077) | 该论文提出前瞻学习与自我回溯蒸馏方法，将已完成的轨迹作为特权后见之明，蒸馏为同一策略在交互前的轨迹无关的预见性预测，从而在组相对奖励信号消失时仍能从经验中有效学习。 |
| [^47] | [HINTT Submission to the 2nd MLC-SLM Challenge: Comparing Cascaded and Unified Approaches to Diarization and ASR](https://arxiv.org/abs/2610.08063) | 该论文提出了HINTT系统，比较了级联流水线（微调DiariZen说话人日志+微调Qwen3-ASR+LLM生成式纠错）与统一语音大语言模型两种策略来解决多语言说话人归属语音识别问题，最终采用级联方案并仅使用官方MLC-SLM数据进行训练。 |
| [^48] | [Language Carries the Expert's Impression: Instrument-Anchored LLM Judges Transfer Counseling-Quality Assessment and Beat In-Domain Training](https://arxiv.org/abs/2610.08055) | 该研究发现，基于专家评分工具锚定构念、由小型开源LLM生成的会话级得分，能够跨领域迁移专家对咨询质量的总体印象评估，且跨域训练的效果甚至优于域内训练。 |
| [^49] | [DAEDALUS: Bootstrapping Agent Memory from Self-Generated Tasks](https://arxiv.org/abs/2610.08048) | DAEDALUS通过探索者代理自生成练习任务、解决者代理从失败中提炼启发式规则并验证其有效性，从而在无需现有任务或人工验证器的情况下自动构建可复用的代理记忆。 |
| [^50] | [Are Language Models Script-Aware?](https://arxiv.org/abs/2610.08037) | 该研究首次系统考察语言模型的文字系统知识，发现所有模型在多文字语言中都能近乎完美地匹配输出文字系统（拉丁文字保真度超98%）并高频遵循指定文字的指令，且大模型表现优于小模型。 |
| [^51] | [The Labeling Problem in Hallucination Detection Benchmarks: An Empirical Evaluation](https://arxiv.org/abs/2610.08026) | 该论文通过实证评估揭示了幻觉检测基准中自动化标注在“参考忠实性”与“事实正确性”两个标准之间存在的方法论模糊性，并研究了由此导致的标注标准错配问题。 |
| [^52] | [Structured but Silent: Probing Capability Requirements in LLM Hidden States](https://arxiv.org/abs/2610.08018) | 本文提出TACIT框架，将工具使用所需的能力需求沿来源、变换和世界效应三个维度分解为八个结构化类别，并证明这些细粒度的查询端能力需求在LLM生成答案之前的隐藏状态中即可被线性解码。 |
| [^53] | [A Broader Look at Model Merging: Rethinking Implicit Regularization Induced by Task Arithmetic](https://arxiv.org/abs/2610.07990) | 本论文发现主流模型合并方法中系数搜索带来的隐式正则化实际上限制了性能，去除该正则化、直接优化合并模型乃至预训练模型的权重，即可在多种架构和极低数据量场景下显著提升合并后的多任务性能。 |
| [^54] | [VisionWeave: Weaving Elastic Visual Representations as a Native Capability of MLLMs](https://arxiv.org/abs/2610.07987) | VisionWeave通过大规模训练，使多模态大语言模型获得按内容自适应地决定视觉表示位置与粒度的“弹性视觉表示编织”固有能力，在保留关键细节的同时显著降低计算成本。 |
| [^55] | [Confidence Reasoning Graphs: Structured Confidence Estimation for LLM Agents](https://arxiv.org/abs/2610.07948) | 提出置信推理图（CRG），一种推理时框架，通过结构化分解单条轨迹中的证据来估计LLM智能体完成任务的概率，无需访问模型内部信号或训练数据。 |
| [^56] | [Hybrid Latent Attention for Looped Language Models](https://arxiv.org/abs/2610.07940) | 提出混合潜在注意力（HLA），通过将旧 token 压缩为紧凑的潜在表示而非存储完整键值缓存，使循环语言模型的缓存缩小 10.7 倍、每 GPU 并发序列容量提升 4.0-8.8 倍、解码吞吐量最高提升 7.4 倍，同时保留原模型 97% 以上的性能。 |
| [^57] | [Leveraging a four-quadrant approach for evaluating Redpine Science](https://arxiv.org/abs/2610.07937) | 本报告采用公开基准与专家验证问题集相结合的四象限评估方法，证明配备Redpine Science的智能体在科学文献问答任务上显著优于无检索基线，正确率从87.6%提升至94.4%。 |
| [^58] | [Pseudowords as probes: Large Language Models show little of the sublexical sensitivity that governs human pseudoword processing](https://arxiv.org/abs/2610.07936) | 通过意大利语伪词二选一实验发现，大语言模型缺乏人类所具备的亚词汇敏感性，在仅有伪词的条件下其表现甚至不及字符n-gram模型fastText，表明模型并未习得支配人类伪词加工的亚词汇线索。 |
| [^59] | [Isotropic Yet Undecodable: The Sequential Content-Sufficiency Gap in Latent-Predictive Text Representations](https://arxiv.org/abs/2610.07906) | 论文通过信息论分解揭示序列内容充分性鸿沟，证明潜在表示的各向同性与一致性无法保证有序目标信息可解码，并据此提出引入规范词元监督的非自回归框架CANOPE，将位置信息恢复率从13.5%大幅提升至98.8%。 |
| [^60] | [ARIA: Audio-Driven Melody-Tone Relation Modeling for Cantonese Lyric Authoring](https://arxiv.org/abs/2610.07902) | 提出ARIA两阶段框架，通过三流关系感知声调估计器从歌唱音频中预测粤语声调序列，并利用解耦检索增强的声调条件生成器，实现直接从原始歌唱录音生成与旋律对齐的粤语歌词。 |
| [^61] | [Rethinking Faithfulness in LLMs: A Pairwise Context-Sensitive Perspective](https://arxiv.org/abs/2610.07894) | 提出成对忠实性基准PFaithBench，通过对比同一问题在支持性与非支持性上下文下的表现，评估大语言模型能否正确地在回答与拒答之间切换，揭示忠实性本质上取决于回答与拒答之间的权衡。 |
| [^62] | [Visual Abstention in Unified Multimodal Models](https://arxiv.org/abs/2610.07887) | 该论文形式化了“视觉拒答”概念并构建了Draw-or-Decline基准，揭示出统一多模态模型的编辑能力与拒答能力相互独立——即便最强的编辑模型也几乎不会拒绝不可行的编辑请求。 |
| [^63] | [ReFold: Training-Free Reversible Inter-Turn Context Folding for Long-Horizon Agents](https://arxiv.org/abs/2610.07863) | ReFold提出了一种免训练的可逆上下文折叠渲染层，通过将已展示内容替换为占位符、将智能体报告完成的轮次折叠为一行注释来消除轮间冗余，从而在保留底层完整交互历史的同时压缩长程智能体的渲染上下文，避免了现有预测性方法带来的运行时开销、前缀缓存失效和不可逆信息丢失。 |
| [^64] | [Lost in the bf16 Cast: Exporting Ternary Language Models Can Revert Most Low-Learning-Rate Code Changes](https://arxiv.org/abs/2610.07853) | 该研究审计发现，三值语言模型导出流程中先将潜在权重转换为bf16，会在阈值处因“舍入到偶数”规则被错误映射为零，导致Falcon-E-1B-Base部署后GSM8K准确率从58.79%暴跌至0.78%，几乎抹去了低学习率微调带来的改进。 |
| [^65] | [Dynamic Positional Attention Modulation for Parameter-Efficient Fine-Tuning of Large Language Models](https://arxiv.org/abs/2610.07848) | 提出DyPAM方法，通过在查询和键表征上结合输入条件化的逐维度调制与逐头逐层的结构化调制，动态调整位置信息对注意力的贡献，实现更精细的参数高效微调。 |
| [^66] | [OMIT the Action: Measuring Framing-Invariant Omission Bias under Philosophical Disagreement](https://arxiv.org/abs/2610.07847) | 本文提出OMIT基准（基于五视角哲学角色小组分歧构建的218个成对框架场景、覆盖10种冲突类型），评估发现八个LLM普遍存在不作为偏差，且该偏差在同一模型家族内随模型规模增大而减小。 |
| [^67] | [Harness Engineering for Software Engineering via Modular Executable Dev-Primitives](https://arxiv.org/abs/2610.07832) | 该论文提出Dev-Primitives，一种将代码库工件与常驻LLM配对的模块化可执行抽象，使软件组件从被动工件转变为具备智能体原生接口的主动参与者，从而解决LLM智能体在长程软件工程工作流中反复重建程序状态、上下文爆炸和语义漂移的问题。 |
| [^68] | [Nucleus Speculative Decoding: Plausibility-Aware Verification Beyond Exact Distribution](https://arxiv.org/abs/2610.07822) | 提出核采样投机解码（NSD），通过在标准接受准则之外额外接受属于目标模型核采样集合的草稿词元来放宽验证条件，从而保留更多草稿词元以加速生成，并从理论上证明其单步分布误差恰好由草稿模型在目标核内的超额概率决定。 |
| [^69] | [$\alpha$Transfer: Coefficient Transfer for Efficient Model Merging](https://arxiv.org/abs/2610.07819) | 提出αTransfer方法，利用同一模型家族内不同规模模型在合并系数上性能分布的高度一致性，在小代理模型上搜索最优系数后直接迁移到大模型，实现最高20倍加速和70%内存减少。 |
| [^70] | [One Step at a Time: Trading LLM Autonomy for Process Predictability](https://arxiv.org/abs/2610.07817) | 该论文提出通过MCP协议逐步向智能体交付流程步骤，以牺牲LLM自主性为代价，从架构上保证流程的事前可预测性，并生成可供下游工具逐步审计和优化的机器可读执行日志。 |
| [^71] | [ThinkFuse: Trajectory-Aware Test-Time Fusion for Small Reasoning Models](https://arxiv.org/abs/2610.07803) | ThinkFuse提出了一种无需训练的轨迹感知测试时融合框架，通过对比片段级不确定性变化与轨迹级整体趋势来识别不稳定推理点，并将辅助推理路径融合进主轨迹，从而显著提升小型推理模型在数学和知识密集型推理任务上的可靠性与性能。 |
| [^72] | [Persistent Memory in Multi-Agent LLM Inference: What It Costs, What It Buys, and When You Can Tell](https://arxiv.org/abs/2610.07782) | 本文在三层多智能体LLM推理架构中实测发现，上下文分解可将峰值KV缓存从35.5 MiB降至14.3 MiB，而持久记忆层不仅增加0.368 MiB缓存开销，且在单问题基准上未带来任何可检测的准确率提升，这种零结果源于基准测试的结构性特点。 |
| [^73] | [Quantization Effects on Tool-Failure Recovery Vary Across Prompts and Evaluation Designs](https://arxiv.org/abs/2610.07781) | 该研究发现8比特与4比特量化对语言模型智能体工具故障恢复能力的影响并不稳定，比较结论会随提示词和评估目标（如评分任务的选择）而改变方向甚至反转，表明量化效果的结论高度依赖于评估设计。 |
| [^74] | [APEX: Speculate smarter, not deeper](https://arxiv.org/abs/2610.07780) | APEX是一个学习型控制器，通过请求级专家选择和块级草稿深度自适应来优化推测解码，用更聪明的推测取代更深的推测，从而降低大语言模型推理延迟并减少计算浪费。 |
| [^75] | [Reading, Not Manipulating: Leveraging Router Logits for Multimodal Safety in MoE Vision-Language Models](https://arxiv.org/abs/2610.07774) | 该论文发现MoE视觉语言模型的路由器logits可作为多模态输入安全性的高预测性诊断信号，据此提出一种轻量级安全检测器，无需操纵模型内部状态即可实现多模态安全检测，避免了传统干预方法带来的安全-实用性权衡。 |
| [^76] | [TRACE: Rollout-Guided Quantization-Aware Training for FP4 Reinforcement Learning of MoE Language Models](https://arxiv.org/abs/2610.07767) | TRACE是一个面向MoE语言模型强化学习训练的FP4量化框架，通过rollout引导的量化感知训练，利用rollout侧量化结果指导训练侧FP4舍入决策，直接缩小训练路径与rollout路径两条量化执行路径之间的差异。 |
| [^77] | [No Transformer Beats Six Covariates: Long-Horizon Prediction of Depressive Symptoms from Childhood Essays](https://arxiv.org/abs/2610.07764) | 该研究发现，在利用11岁儿童作文预测其23岁抑郁症状的长期任务中，基于六个童年协变量的简单逻辑回归（AUC-ROC 0.737）显著优于所有文本模型，包括微调Transformer、词袋模型、冻结嵌入和零样本大语言模型（最佳仅0.670）。 |
| [^78] | [From Evidence to Action: How Tool-Using Agents Fail](https://arxiv.org/abs/2610.07753) | 提出包含656个案例的SafeActBench基准，系统揭示工具使用型智能体在“证据到行动”链条中的失败模式——失败往往在执行前就因调查不完整或过早行动而出现，多动作工作流还会暴露未解决的先决条件和不完整执行问题。 |
| [^79] | [Learning to Retrieve via Reinforcement Learning in Embedding Space](https://arxiv.org/abs/2610.07731) | 该论文提出RELER强化学习框架，通过从vMF分布采样嵌入动作、结合RLOO基线的REINFORCE算法以及减少采样噪声的条件均值投影（CMP）技术，使现有嵌入模型能够直接在嵌入空间中学习检索并对齐任务特定的奖励。 |
| [^80] | [SanSi: A Looped Typed Decision Model for System 1.5 Thinking](https://arxiv.org/abs/2610.07730) | SanSi提出“系统1.5思维”——通过多次循环复用模型层在不生成文本的情况下修正隐藏状态，将预训练循环语言模型转化为类型化决策模型，在59个数据源的10,027个测试决策上达到72.0%准确率，比同结构非循环模型高出13.5个百分点。 |
| [^81] | [Does Steering Break Your Model? A Multi-Dimensional Evaluation Suite for LLM Steering Methods](https://arxiv.org/abs/2610.07722) | 本文提出SteerScope，一个包含15项指标的双轴多维评估套件，系统刻画LLM激活转向方法在有效性与副作用（语言质量、任务能力、安全可靠性）之间的权衡，并进一步评估其泛化能力与数据依赖性。 |
| [^82] | [Readout Stability in Prefill-Only Decision Models:Zero-Label Prediction and Inference-Time Compute Allocation](https://arxiv.org/abs/2610.07716) | 论文发现仅预填充决策模型具有可检验的“读出稳定性”：仅改变候选菜单时，干预后的准确率可由缓存的首次前向概率通过零标签、无需第二次前向传播的估计器预测（误差在4.2个百分点以内），且该性质存在于排序而非概率之中。 |
| [^83] | [When Old Facts Return: Re-Reads, Reverts, and the Limits of Temporal Memory](https://arxiv.org/abs/2610.07715) | 该论文揭示了时序记忆系统的一个关键歧义——旧信息的重读与真正的回退会产生相同的观测序列却需要相反的答案——并提出一个拒绝重新激活已淘汰值的守卫机制，该机制能有效防御重读攻击（准确率从10.8%恢复至97.7%），但需要额外的变更溯源信息才能区分合法回退。 |
| [^84] | [Detecting LLM-Assisted Vietnamese Writing via Keystrokes under Behavioral Manipulation](https://arxiv.org/abs/2610.07700) | 该研究构建了越南语击键数据集并提出了基于行为的威胁模型，发现序列模型在检测LLM辅助写作时优于基于特征的方法，且击键信号包含判别性信息，但在用户故意操纵打字行为时检测效果并不均匀。 |
| [^85] | [Improving Synthetic Data Generation for Argument Mining via Adversarial Reinforcement Learning](https://arxiv.org/abs/2610.07699) | 提出一种对抗性强化学习数据合成框架，通过生成器与判别器的对抗循环联合优化，同时提升论辩挖掘合成数据的结构准确性与多样性。 |
| [^86] | [DLoop: Looped Speculative Decoding](https://arxiv.org/abs/2610.07659) | DLoop提出一种循环式推测解码方法，在验证前自适应地执行多个起草阶段，从而减少目标模型不必要的前向验证，提升大语言模型推测解码的加速效果。 |
| [^87] | [Where Rules End and Judges Begin: Measuring the Judgment Boundary in Multi-Agent Systems Security](https://arxiv.org/abs/2610.07657) | 该研究提出DEFER1防御框架，通过28项确定性检查级联与四位裁判评审团协同工作，将多智能体系统的攻击成功率从约30%降至约3%，并实证划定了规则可处置与需裁判判断的安全边界。 |
| [^88] | [Loud and Clear: Dynamic Activation Steering for Improving Speech Intelligibility in Noisy Environments](https://arxiv.org/abs/2610.07647) | 提出一种无需重新训练的提示相对激活引导机制，可动态控制预训练TTS模型模拟人类隆巴效应，在噪声环境下将词错误率降低7-22%的同时保持89-95%的说话者相似度。 |
| [^89] | [Monte Carlo Estimation for KV Cache Eviction](https://arxiv.org/abs/2610.07643) | 提出免训练方法LORE-KV，通过蒙特卡洛采样冻结模型的短自回归延续并以响应侧查询状态估计提示token效用，将KV缓存驱逐从“回顾过去”转变为“预测未来”，在回答时保留真正重要的记忆。 |
| [^90] | [A Novel Sentence Stress Detection Framework Leveraging Auxiliary Word-Stress Modeling and Loss Optimization](https://arxiv.org/abs/2610.07626) | 该论文提出了一种通过辅助词重音建模和词跨度重音正则化器将句子重音检测与词重音检测联合训练的新型框架，在 TinyStress-15K 基准上取得了最佳的句子重音检测性能。 |
| [^91] | [Stateless Language Agents: Scaling Long-Horizon Automated Research](https://arxiv.org/abs/2610.07625) | 提出无状态语言智能体（SLA）框架，通过“有状态搜索、无状态智能体”的原则——由框架统一管理研究状态并为每次调用重建角色化上下文——来解决长时程自动化研究中智能体重放冗长历史、重复劳动和过早停止实验等失败模式。 |
| [^92] | [Recurrent Looped Transformer](https://arxiv.org/abs/2610.07591) | 提出循环环路Transformer（RLT），通过将层分配给并行因果编码器和循环解码器，使计算路径随序列长度增长而每个token成本保持固定，在状态跟踪和算法泛化任务上大幅超越固定深度的标准Transformer。 |
| [^93] | [Large Language Model Orchestration under Heterogeneous Preferences via Explicit Persona Inference](https://arxiv.org/abs/2610.07587) | 提出HARP框架，将智能体的偏好信念从提示文本中移出，改为在有限候选偏好集合上维护数值后验分布进行显式推断，避免早期错误持续传播，从而改进异构偏好环境下的大语言模型编排。 |
| [^94] | [LOGIC: An LLM Benchmark for Intent-Grounded Change Impact in Aerospace Electrical Systems](https://arxiv.org/abs/2610.07580) | LOGIC是一个航空航天电气系统领域的受控基准，用于评估语言模型能否根据工程请求意图从确定性候选变更清单中正确选择变更，并通过类型化电气可追溯性图传播其影响，实验表明仅门控结构化证据方法在明确锚定的选择案例上达到了完美的F1分数1.0000。 |
| [^95] | [Two Vectors Replace In-Context Demos: Structured Task Adaptation via Embeddings](https://arxiv.org/abs/2610.07572) | 提出STAVE方法，用两个任务特定向量（读取向量和上下文向量）直接加到现有输入嵌入中来替代上下文示例，避免了示例图像的重复编码开销，实现高效的结构化任务适配。 |
| [^96] | [HouseholdBench: Evaluating Large Language Models as Predictors of Household Economic Behavior](https://arxiv.org/abs/2610.07563) | 提出HouseholdBench基准，整合6项美国家庭调查和32个覆盖数值、类别与概率型结果的预测任务，系统评估大语言模型预测家庭经济行为及响应政策变化的能力。 |
| [^97] | [Quality-Aware Self-Correcting Speech Translation on an Edge Device](https://arxiv.org/abs/2610.07545) | 该论文提出了一种在Jetson Nano边缘设备上运行的完全离线自校正语音翻译流水线，发现质量估计（QE）适合作为触发二次校正的门控而不适合作为候选重排序器，其中最小贝叶斯风险解码在阈值0.90下取得了统计显著的翻译质量提升。 |
| [^98] | [Disentangling Models from Personas in Heterogeneous LLM Simulations](https://arxiv.org/abs/2610.07535) | 该研究通过模拟由多个基础模型驱动的异构社交网络，发现智能体获得的互动量更多取决于其基础模型而非被分配的角色人格，且随着模型数量增加，模型间效应显著增强，表明网络动态可能在大规模下收敛于基础模型效应。 |
| [^99] | [Safeguarding LLMs via Model-Agnostic Latent Safety Signals from Dark Knowledge](https://arxiv.org/abs/2610.07532) | 提出LADE方法，通过对比有害与良性查询，从首token输出分布的暗知识中提取模型无关的潜在安全信号，在解码阶段实现安全防御，同时避免安全性与过度拒绝之间的权衡，并能跨架构泛化。 |
| [^100] | [Not What a Child Expressed: Auditing the Sign-to-Text Safety Interface in Child-Facing AI](https://arxiv.org/abs/2610.07519) | 本文首次提出由聋人参与指导的部署前审计框架，用于审查“手语转文本”系统与儿童安全审核工具之间的接口，揭示SLT翻译错误可能在不影响文本流畅性的情况下悄然改变针对儿童手语使用者的安全决策。 |
| [^101] | [On Open-Ended Information Seeking for Information Elicitation Agents](https://arxiv.org/abs/2610.07509) | 本研究通过在11个跨越不同家族和参数规模的大语言模型上进行的受控诱导模拟，揭示了不同LLM对信息价值的判断存在差异，且这些差异会显著塑造其序列化的开放式信息搜寻行为。 |
| [^102] | [Closing Ambient Clinical Documentation Gaps with Automated Provider Queries](https://arxiv.org/abs/2610.07502) | 该论文提出DAU（起草-提问-更新）框架，用大语言模型自动化临床文档专员的查询闭环以弥补病历信息缺口，并基于真实就诊审计构建转录退化基准，揭示有效澄清问题的预测因素具有任务特异性，且约9%的查询轮次反而会损害性能。 |
| [^103] | [In With the Old: Enhancing 'Classical' Document Automation with Generative AI](https://arxiv.org/abs/2610.07480) | 本文探索了基于专家系统等符号方法的经典文档自动化与生成式AI如何相互增强，并通过初步实验证明大语言模型可用于识别和修复非专业人士撰写的法律文本中的问题。 |
| [^104] | [Auditable Claims about AI Agents](https://arxiv.org/abs/2610.07459) | 提出AI智能体声明的可审计性标准——声明必须在事前明确其政策、范围、裁决记录及记录撰写者，并满足独立记录覆盖、授权绑定操作参数和超越完整性的完备性三项条件，才能被有效核查。 |
| [^105] | [AlignQuant: Tile-Aligned Mixed-Precision Quantization for Efficient LLM Generation](https://arxiv.org/abs/2610.07457) | AlignQuant提出了一种以GPU兼容的二维权重瓦片作为精度分配、存储和执行公共单元的训练后混合精度量化方法，使大语言模型的压缩能够真正转化为实际推理加速。 |
| [^106] | [AccentCL: Robust Accent Classification with Incremental Expansion](https://arxiv.org/abs/2610.07426) | 提出了AccentCL框架，通过不平衡感知损失、领域均值对齐损失和基于回放的持续学习，实现了对类别不平衡和跨语料库领域偏移具有鲁棒性的英语口音分类，并支持新口音类别的增量扩展。 |
| [^107] | [Who Wrote It Is Not Enough: Detecting Who Contributed the Insight](https://arxiv.org/abs/2610.07365) | 该论文提出“洞察溯源”新任务并构建InsightProv-v0数据集，通过两阶段对抗框架消除语言捷径信号，从而准确识别科学评审洞见的贡献者是来自人类、大语言模型还是二者的混合。 |
| [^108] | [Tracking Is Not Permanence: What Video World Models Keep of a Hidden Object](https://arxiv.org/abs/2610.07355) | 该研究发现尽管视频世界模型的编码器完整地编码了被遮挡物体的信息，但其预测器会在遮挡发生后0.3秒内迅速丢弃这些信息，揭示了当前视频世界模型严重缺乏物体永久性表征。 |
| [^109] | [Stepped MoE: Segment-Level Routing with Configurable Inference Complexity](https://arxiv.org/abs/2610.07348) | 本文提出阶梯式MoE统一框架，将弹性结构与稀疏门控架构相结合，通过分段级路由使模型能够同时适应不同的部署约束和任务需求，实现推理时对精度-效率权衡的细粒度控制。 |
| [^110] | [A doctrine-grounded visual question answering dataset for Tactical Combat Casualty Care](https://arxiv.org/abs/2610.07339) | 本文提出TC3-VQA数据集，利用公开教学与实战视频和权威条令文档构建了1,860个将视觉证据与可追溯战伤救护条令关联的问答样本，为支持战术战斗伤员救护的视觉-语言模型开发提供监督数据。 |
| [^111] | [Logbook: Extremely Long-form Audio Event Understanding](https://arxiv.org/abs/2610.07338) | 该论文提出了面向小时级至六天超长音频的事件理解基准 Logbook，要求系统对连续音频进行无缝隙分割并为每段生成事件标签与描述，发现最佳系统仍不及人类、过度分割普遍存在，且端到端系统通常优于级联系统但性能随上下文变长而下降。 |
| [^112] | [Structuring MoE Expert Selection for Agentic Reinforcement Learning](https://arxiv.org/abs/2610.07332) | 该论文发现MoE专家选择与智能体轨迹存在天然的结构对齐（语义相似操作的轮次共享更多路由专家），并提出分层路由控制框架将这一结构约束纳入智能体强化学习训练，从而同时提升任务性能与推理效率。 |
| [^113] | [SharedKV-BT: Node-Local Typed Decisions for Behavior-Tree Agents](https://arxiv.org/abs/2610.07327) | 该论文提出SharedKV-BT，通过行为树活动节点暴露节点本地字段并由Shared-KV并行评分候选决策，使类型化决策速度较自回归解码提升2.36-4.15倍，同时将操作任务的联合决策准确率从75%提升至94%。 |
| [^114] | [Kurate: Scalable Scientific Quality Analysis](https://arxiv.org/abs/2610.07306) | Kurate是一个利用大语言模型从统计功效、选择性报告等8个维度大规模评估已发表研究质量，并将每项评估判断链接到具体文本依据的科研质量分析系统。 |
| [^115] | [Lineage-Aware Memory Governance: A Derivation-Gated Framework for Privacy-Preserving Column-Level Access Control in Enterprise AI Agents](https://arxiv.org/abs/2610.07258) | 该论文提出分析内存单元（AMU），通过为每个缓存结果附加完整的派生谱系图并实施列级权限门控检索，从设计上保证企业AI智能体不会命中由请求者无权限的敏感列派生而来的缓存结果，同时解决部门间同名KPI计算逻辑冲突的问题。 |
| [^116] | [Minimal Witness Reinforcement Learning](https://arxiv.org/abs/2610.07226) | 本文提出最小见证强化学习（MWRL），利用基于集合并集覆盖损失的信用分配机制，仅凭单一黑盒验证器的信号即可同时实现解的最小性与多个备选解的恢复。 |
| [^117] | [TIDE 2.0: an open, model-agnostic engine for keyed de-identification of clinical notes](https://arxiv.org/abs/2610.07224) | TIDE 2.0是一个开源、模型无关的临床笔记去标识化引擎，通过密钥化匿名技术——包括保持时间间隔的日期偏移和密码学生成的替代值——在保护患者隐私的同时保留数据的纵向分析价值，且无需依赖外部硬件或存储关联表。 |
| [^118] | [Forecasting the Growth of Social Media Information Cascades: Towards Human-in-the-Loop Misinformation Triage](https://arxiv.org/abs/2610.07209) | 该论文提出在虚假信息传播的前30分钟内，通过结合早期节点数、结构深度熵和时间到达熵的特征来预测传播树的后续增长，尤其在内容高度活跃的传播树上表现优异，从而帮助人手有限的审核团队在传播范围尚不明确时优先分诊出具有高增长潜力的虚假信息。 |
| [^119] | [Responsible Institutional Analytics: Interpreting Bias with AI Support](https://arxiv.org/abs/2610.07205) | 提出了FACTRIA框架，将院校分析中的潜在偏差因素组织为四个维度，并通过生成式AI聊天机器人引导用户反思这些因素，研究表明结构化框架与AI引导相结合能够促进更具情境意识的负责任解读。 |
| [^120] | [Identifying Introspection From the Inside](https://arxiv.org/abs/2610.07186) | 该研究发现模型在隐式决策任务上持续微调后会涌现出对所学偏好的准确自我报告，且伴随偏好表征向更早层转移的结构性变化，为区分真实内省与虚构提供了机制性标志。 |
| [^121] | [Learning Scientific Exploration from Human Research Decision Trajectories](https://arxiv.org/abs/2610.07184) | 本文提出ResearchTrails数据集，以Git仓库的提交历史作为人类科研探索过程的代理，并开发自动化流水线从中提取结构化的研究决策轨迹，弥补了现有科学语料库只记录最终成果、缺乏探索过程信息的不足。 |
| [^122] | [CLM-as-a-Judge: Evaluating an Open Contrastive Decision Model on Public Judge Benchmarks](https://arxiv.org/abs/2610.07177) | 该论文首次系统评估了开放对比决策模型 CLM-v0.1-8B 作为裁判的能力，发现其在公开基准上接近随机水平且显著落后于同规模奖励模型和生成式裁判，但通过单参数温度校准可将其置信度修复至良好校准状态。 |
| [^123] | [A theory of platonic representations in language models](https://arxiv.org/abs/2610.07168) | 本文通过假设数据具有隐藏的层级结构（抽象层次跨语言共享、表面层次为语言或模态特定），并借助概率上下文无关文法与信念传播理论推导出分析性预测，首次从理论上解释了多语言模型中间层出现柏拉图式表示的现象及其随语言相近程度和模型质量增强的规律。 |
| [^124] | [CroissantMiner: Automated Extraction and Validation of Croissant Metadata for ML Datasets](https://arxiv.org/abs/2610.07132) | 该论文提出了首个针对Croissant元数据提取的端到端评估基准（包含602篇论文的金/银双级标注），并发现单次提取方法在各类模型骨干上始终优于四种智能体架构。 |
| [^125] | [Jailbreaking Open-Weight LLMs via Random Embedding Perturbations](https://arxiv.org/abs/2610.07125) | 该论文提出PEV攻击方法，仅需在提示的嵌入向量中反复添加随机高斯噪声即可越狱多种规模的开放权重大语言模型，暴露了此类模型的安全脆弱性。 |
| [^126] | [JudgeMoE: Distributional Aggregation for LLM-as-a-Judge](https://arxiv.org/abs/2610.07109) | JudgeMoE是一种轻量级聚合器，通过为LLM评判者的评分分布分配样本特定权重并进行融合，保留了标量压缩中丢失的不确定性与分歧信息，在多个基准上显著提升了评判与人类判断的相关性。 |
| [^127] | [Smart Content Ingestion for Generative AI Workloads](https://arxiv.org/abs/2610.07091) | 本文提出智能内容摄取的理念，指出在生成式AI时代，由于企业知识以PDF、电子表格等异构格式承载多种信息模态，内容提取已演进为AI生命周期中独立且不可替代的关键阶段，其错误无法被下游检索或重排序组件修复。 |
| [^128] | [Turnslide: Scalable Multi-Turn Data Synthesis by Walking a Finite-State Machine](https://arxiv.org/abs/2610.07070) | 提出Turnslide框架，将API建模为有限状态机并设定目标分布，以低成本、高可扩展性地合成多轮工具调用数据，且每个样本仅需单次LLM调用即可生成。 |
| [^129] | [Learning to Simulate Individuals from Macro Social Signals](https://arxiv.org/abs/2610.07062) | 该论文提出macro2mind框架，将预测市场价格轨迹作为宏观监督信号，通过GRPO训练和社会行为分解，使大语言模型把行为推理作为显式预测步骤，从而从宏观数据中学会模拟个体对真实事件的反应。 |
| [^130] | [SEAL: Mixture-Closed Additive Reconstruction and Refinement-Aware Expert Routing for Efficient Speech Separation](https://arxiv.org/abs/2610.07047) | SEAL通过混合闭合的零和加性残差重建与基于声学证据的稀疏专家路由机制，在参数减少28%、计算量降低2.9倍的情况下，语音分离性能超越TIGER达0.31 dB SI-SDRi。 |
| [^131] | [GIVE-KWS: Gated Injection of Visual Evidence for Noise-Robust Query-by-Example Keyword Spotting](https://arxiv.org/abs/2610.07046) | 该论文提出GIVE-KWS，通过门控交叉注意力将唇部运动的视觉证据注入查询音频，并证明噪声鲁棒性需要同时具备含音素信息的视觉表示和注入式融合机制，从而在-10 dB噪声下相比掩蔽方式获得4.0-9.3 dB的有效信噪比增益。 |
| [^132] | [Investigating Model Compression for Neural Machine Translation in the Biomedical Domain](https://arxiv.org/abs/2610.07032) | 本研究探讨了知识蒸馏和量化两种模型压缩技术在生物医学领域神经机器翻译中的应用，揭示了这两种技术在低资源专业领域条件下的局限性。 |
| [^133] | [Beyond Refusal Patterns: Safe-Role Internalization for Robust and Generalizable LLM Safety Alignment](https://arxiv.org/abs/2610.07023) | 提出SSRFT（监督安全角色微调）框架，首次将LLM安全对齐重新表述为对预定义安全角色的内化，通过构建SRQA数据集使模型内化安全价值观与原则，从而以更少的攻击特定监督实现更鲁棒、可泛化的安全对齐，并缓解过度拒绝问题。 |
| [^134] | [Calibrated Answers About Randomized Trials From a 4-Billion-Parameter Open Model: A Registered Test and a License-Clean Release](https://arxiv.org/abs/2610.07019) | 该论文发布了一个仅使用许可证允许复用的文章微调的 40 亿参数开放模型 Fiorillo v0.5，它能够以良好校准的概率回答随机试验中干预措施对结局影响的问题，并通过预注册的四项标准验证后正式发布。 |
| [^135] | [Mask-Guided KV Cache Eviction in Block Diffusion Language Models](https://arxiv.org/abs/2610.06996) | 提出无需训练的MaskAhead方法，通过统一的掩码-查询排序机制同时解决分块扩散语言模型中KV缓存的选择与淘汰问题，其量化变体Q-MaskAhead可在低比特KV上直接计算，从而降低内存占用并加速生成。 |
| [^136] | [AegisFlow: A Multi-Agent Agentic AI Framework for Autonomous Remediation and Self-Healing in Fragile Data Ecosystems](https://arxiv.org/abs/2610.06971) | AegisFlow是一个多智能体AI框架，通过Watchdog智能体收集运行时遥测、Repair智能体基于LLM自动生成并部署代码补丁，并采用基于MAPE-K循环的“并行影子补丁”非侵入式模型在数字孪生环境中验证补丁，从而实现脆弱数据管道从故障检测到自主修复的闭环自愈。 |
| [^137] | [WavePrune: One period is often enough for RoPE](https://arxiv.org/abs/2610.06963) | 提出WavePrune方法，通过将RoPE每个通道限制在其首个旋转周期内来消除位置混叠问题，无需额外调整即可提升多个模型的长上下文性能（如Qwen3-8B的HELMET分数从35.7升至40.0）。 |
| [^138] | [Verdicts Without Annotated Evidence: Rejection Sampling or Label-Only Post-Training for Evidence Recovery?](https://arxiv.org/abs/2610.06962) | 该研究表明，在没有任何人工证据标注的情况下，仅用判定标签进行后训练的小语言模型在证据恢复上优于基于自动来源接地分数的拒绝采样方法，且判定准确率与证据跨度一致性仅弱相关，说明准确率不能作为引用可审阅性的可靠指标。 |
| [^139] | [EMODE: Dynamic Para-Semantic Experts for Emotion-Aware Speech Language Modeling](https://arxiv.org/abs/2610.06956) | EMODE 提出动态副语义专家（DPSE）架构，将语音特征分解为语义与副语言双通路并进行动态路由融合，结合三阶段训练课程，使语音语言模型能够有效保留情感等副语言信息。 |
| [^140] | [Stabilizing language models under continual learning via condition-anchored distillation](https://arxiv.org/abs/2610.06940) | 该论文提出条件锚定生成蒸馏（CAGD）方法，通过保留少量旧提示并利用冻结的旧模型作为教师来匹配预测分布，从而在语言模型持续学习新任务时稳定其对旧任务的输出行为，并为自回归生成提供精确的序列散度链式法则分解、为掩码扩散建模提供局部去噪漂移的直接控制。 |
| [^141] | [Component and Dimension Sparsity in Transformer Refusal Mechanisms](https://arxiv.org/abs/2610.06903) | 该研究通过对四个开源大语言模型的组件级干预分析，发现拒绝行为引导只需稀疏组件子集（占上游组件28%–48%）及其中约50%的残差流维度即可复现完整效果，揭示了拒绝机制在组件和维度两个层面上的稀疏性。 |
| [^142] | [Tree Navigation Without LLM Summaries: A Matched-Cost Study of Hierarchical Retrieval for Long-Document QA](https://arxiv.org/abs/2610.06902) | 该论文提出NavTree，证明长文档问答中RAPTOR式摘要树的主要收益来自树的导航结构而非LLM生成的摘要内容，该方法在索引阶段零LLM调用，仅用确定性平衡线段树作为导航支架即可实现有效的分层检索。 |
| [^143] | [Capacity, Responsiveness and Alignment: What Makes a Latent Structure Actionable](https://arxiv.org/abs/2610.06897) | 该论文将语言模型中潜在结构的因果影响力分解为容量、响应性和对齐性三个可解释且独立的约束因素，并通过跨4个模型家族、50个概念的实验证明，只有三者同时处于高水平时，该结构才真正具有因果可操作性。 |
| [^144] | [Zero-Shot Visualization: Exploring Text Corpora with User-Prompted Axes](https://arxiv.org/abs/2610.06889) | 该论文提出了零样本可视化（ZSV）任务，允许用户通过自然语言指定概念轴来交互式探索文本语料库，并通过基准测试发现基于下一个词元概率的评分方法在语义忠实性、评分保真度和计算成本方面具有优势。 |
| [^145] | [When Does External Guidance Help LLM Reasoning? A Bias-Variance Theory of Guidance-Augmented GRPO](https://arxiv.org/abs/2610.06861) | 该论文提出GA-GRPO统一理论框架，将外部指导建模为随机指导算子，证明其引入的偏差可由全变差指导散度δ_G界定，从而为外部指导何时以及如何帮助LLM推理提供收敛速率、偏差界和最优加权规则的理论基础。 |
| [^146] | [Improving Diversity in LLM Short Story Generation](https://arxiv.org/abs/2610.06729) | 提出DivLM两阶段后训练框架，通过创意写作语料持续预训练结合复合奖励函数的强化学习，使LLM在体裁、语气、风格和命名实体等维度的短篇小说生成多样性平均提升超过9%，同时保持指令遵循与回复质量。 |
| [^147] | [Wikidata Search Traces: A Dataset for Training Knowledge Graph Search Agents](https://arxiv.org/abs/2610.06650) | 该论文发布了首个记录解题者在Wikidata知识图谱上探索过程的数据集，用于训练图搜索智能体，并证明图搜索难度可通过问题结构控制、长程搜索失败主要源于证据管理方式而非模型本身，且开源模型在合适环境中可匹敌商业模型。 |
| [^148] | [JEV versus LLMs: Accuracy, Cost and Calibration on Seven Political Science Replications](https://arxiv.org/abs/2610.06625) | 本文通过七项政治科学复制研究，将商业模型JEV与传统LLM及人工编码员在准确性、成本和校准性上进行对比，以检验其在社会科学文本标注任务中的实际适用性。 |
| [^149] | [TeleTune: Evolving Agent Skills From Offline Telemetry](https://arxiv.org/abs/2610.05437) | TeleTune提出了一个从无目标标注、无法重放且任务交错的离线用户遥测日志中，通过动作预测误差自动演化文本技能库的框架，使计算机使用智能体能够学到可复用的软件操作技能。 |
| [^150] | [Inductive Claims Extraction at Scale](https://arxiv.org/abs/2610.05275) | 本文提出了一个利用大语言模型从大规模社交媒体语料中归纳式抽取并编目论断的处理流程，并将其应用于2020年美国大选和2022年世界杯两个Twitter数据集，通过召回率和精确率全面验证了该方法的有效性。 |
| [^151] | [A Systematic Analysis of the Predictive Power of LM Surprisal in Reading Chinese](https://arxiv.org/abs/2610.04898) | 本研究提出最短匹配序列（SMS）对齐方案以解决中文分词与语言模型子词分词不一致的问题，并利用从零训练的Chinese-Pythia模型证明语言模型意外度确实能预测中文阅读时间，且预测能力随模型规模的缩放模式因语料库而异。 |
| [^152] | [SpecFold: Folding Multi-Branch Redundancy for Faster Speculative Decoding in Diffusion Language Models](https://arxiv.org/abs/2610.04875) | SpecFold通过识别并利用投机验证中草稿分支与父分支之间隐藏状态高度相似的多分支计算冗余，以token级残差门控和选择性计算复用降低验证成本，从而加速扩散语言模型的多分支投机解码。 |
| [^153] | [More Value per Key: Asymmetric Sparse Attention for Faster LLM Decoding](https://arxiv.org/abs/2610.04753) | 提出稀疏非对称分组查询注意力SAGA，通过解耦键头与值头数量——用更少的键头加速推理、保留更多值头维持模型容量——从而实现更快的LLM解码。 |
| [^154] | [Understanding Errors in LLM-Based Question Answering over Imperfect Tables](https://arxiv.org/abs/2610.04687) | 该研究发现LLM在不完美表格问答中，错误发现的程度受行顺序影响（错误行出现越晚或越集中越容易被发现），且仅提供已验证的错误位置信息并不足以保证准确问答。 |
| [^155] | [Benchmarking Candidate Coverage in Typed Decision Models](https://arxiv.org/abs/2610.03387) | 本文提出了一个成对候选覆盖度基准测试协议，用于评估类型化决策模型识别缺失答案与避免错误拒绝有效候选的能力，发现 Laya 和 Jev 的原生拒绝行为差异显著，而仅使用校准数据的 none 分数阈值可以显著改善两者的检测与误拒平衡。 |
| [^156] | [Verifiable, Articulable, and Tacit Components of Preference](https://arxiv.org/abs/2610.03025) | 该论文推出了包含280万文本和3.17亿人类偏好判断的CreativePreferences数据集，通过可执行程序、评分标准库与密集训练模型三种方式建模偏好，量化了“可表达”与“可验证”成分和完整偏好之间的差距，揭示了AI偏好优化中长期被忽视的默会成分。 |
| [^157] | [A Guideline-Augmented Multi-Agent Framework for Schema-as-Code Biomedical Named Entity Recognition](https://arxiv.org/abs/2610.02970) | GAMA框架通过从标注数据中归纳并验证特定数据集的标注规则来构建指南记忆，结合多智能体协作与模式即代码的结构化输出，显著提升了生物医学命名实体识别的准确性和格式规范性。 |
| [^158] | [When Does a Second Model Help? Cross-Model Review in LLM Verification](https://arxiv.org/abs/2610.01471) | 在大语言模型输出验证中，跨模型审查与同模型审查发现的错误集合部分不同，且“一次同模型新会话审查加一次跨模型审查”的组合比两次同模型审查能匹配到更多埋设错误（56.7% vs. 42.7%）。 |
| [^159] | [AgSpec: Pushing the Limits of Retrieval-Based Speculative Decoding in Coding Agent Pipelines](https://arxiv.org/abs/2610.01108) | AgSpec通过构建会话、工作区和全局三级语料库并以离线上限与在线反馈自适应调整草稿长度，突破了编码智能体流水线中基于检索的投机解码的性能极限。 |
| [^160] | [Provably Tractable NFA-Constrained Language Generation via HMMs](https://arxiv.org/abs/2609.40185) | 该论文提出了NFA-LM，一个基于#NFA问题FPRAS理论的多项式时间生成引擎，首次在温和假设下以理论保证高效解决NFA约束语言生成问题，克服了现有方法扭曲分布或牺牲效率的缺陷。 |
| [^161] | [Strong Multilingual Privacy Tagging at Encoder Speed](https://arxiv.org/abs/2609.38630) | 该研究提出了一个以编码器速度运行的多语言隐私实体标注模型，通过覆盖感知掩码与子词边界修复等技术，在7种语言的人工金标准测试上取得88.8的脱敏F1分数，显著超越GLiNER2、Microsoft Presidio和OpenAI Privacy Filter等现有方法。 |
| [^162] | [Marking Contour Tones in Yor\`{u}b\'{a}](https://arxiv.org/abs/2609.38627) | 本文提出在约鲁巴语正字法中采用caron（ˇ）和circumflex（ˆ）符号来标记单个元音上的升降曲折调，以解决传统拼写中声调信息缺失甚至颠倒姓名含义的问题，并使其首次可通过标准键盘输入和计算文本处理。 |
| [^163] | [KlinikeBench: Evaluating Language Models Beyond Diagnostic Accuracy](https://arxiv.org/abs/2609.38480) | KlinikeBench是一个包含333个由临床医生编写的任务的基准，通过沙盒环境中的虚拟患者交互，评估语言模型在信息收集和临床评估方面超越单纯诊断准确性的综合临床能力。 |
| [^164] | [Storage Is Not Strategy: State-Conditioned Support Control for LLM Unlearning](https://arxiv.org/abs/2609.37858) | 该论文发现“存储目标知识”的参数未必是执行遗忘的最佳干预对象，提出基于实际遗忘更新预测效果的干预分数以及动态干预重排方法（DIR-R），在优化过程中按需自适应调整干预参数子集，从而显著提升大语言模型遗忘的效果。 |
| [^165] | [Where Do Test-Time Scaling and Training Fall Short in Individual Stance Prediction?](https://arxiv.org/abs/2609.33155) | 该研究揭示了测试时扩展和后训练方法（如监督微调与强化学习）在个体立场预测任务中存在错误共识、选择失败、响应过拟合和早期停滞四种失败模式，并提出STANCE-BENCH基准来系统性地暴露这些不足。 |
| [^166] | [ELF-REG: Scaling Continuous Diffusion Language Models to Reasoning Tasks](https://arxiv.org/abs/2609.29102) | 提出ELF-REG方法，利用冻结自回归教师模型的表示对齐与纠缠（REPA+REG）监督，成功将全连续扩散语言模型扩展至数学推理与代码生成任务，性能超越同规模扩散语言模型。 |
| [^167] | [Consequential Behaviour and Representational Fairness in the Validation of Synthetic Research](https://arxiv.org/abs/2609.27690) | 该论文指出现有的合成调查受访者验证方法在预测后果性行为的应用场景中检验了错误的目标，并提出一个要求效度声明必须明确与人类数据对应水平的验证框架，以保障合成研究的表征公平性。 |
| [^168] | [Text Scores Can Miss Waveform Use: A Qwen2-Audio Quantization Case Study](https://arxiv.org/abs/2609.26823) | 仅凭文本输出分数评估语音模型量化会掩盖其对波形信息的依赖：Qwen2-Audio案例研究显示，为翻译任务选择的6位量化虽将chrF提升2.36，却在情感识别上下降3.91个百分点，且表现不及同等位宽下的均匀量化控制方案。 |
| [^169] | [Dream-RSI: Recursive Self-Improvement through Evolving Worlds](https://arxiv.org/abs/2609.14858) | Dream-RSI提出了一种递归自我改进的探索框架，通过轻量级编排层使探索显式可编程，并创新性地将积累的发现历史作为回放模拟器，在其中进行“做梦”式改进，从而突破传统探索策略难以适应扩展搜索空间的瓶颈。 |
| [^170] | [Quantifying the Generation Modality Gap in Speech-Text Language Models](https://arxiv.org/abs/2609.14743) | 该论文构建了一个统一的生成式评估套件，在匹配的数据分布和评估设置下，从语义连贯性、语音结构、声学质量等多个维度量化了纯语音、纯文本和语音-文本语言模型之间的生成模态差距。 |
| [^171] | [UnitBoost: Managing Compound LLM Systems with a Merge Operator, Not a Model](https://arxiv.org/abs/2609.09815) | UnitBoost用确定性的合并算子取代复合LLM系统中的生成式元代理，通过槽位-值提案、约束argmax和显式残差机制实现顺序无关、可溯源的系统协调，并在基准测试中超越了金标准标签选出的最佳单一候选。 |
| [^172] | [VoxReason: Listener-Free Evaluation of Source-Grounded Speech Planning Before Synthesis](https://arxiv.org/abs/2609.03203) | VoxReason提出了一种无需听者参与的评估任务，在语音合成之前通过带证据引用的说话计划和确定性验证器，衡量语音表达方式的选择是否真正建立在被引用的源记录之上。 |
| [^173] | [WinoQueer-NL: Assessing Bias in Dutch Language Models toward LGBTQ+ Identities](https://arxiv.org/abs/2609.02651) | 该研究构建了首个评估荷兰语语言模型对LGBTQ+身份偏见的基准数据集WinoQueer-NL，通过与荷兰酷儿群体的调查验证了145个文化相关刻板印象并新发现22种偏见，揭示了看似中性的平均偏见得分背后隐藏的显著偏见。 |
| [^174] | [Vision Is Not Overhead: One-Pass Block Drafting for Lossless Speculative Decoding in Vision-Language Models](https://arxiv.org/abs/2609.00355) | 该论文提出 GLANCE——首个在未修改的视觉语言模型上实现无损推测解码的单遍块草拟器，通过块扩散头零成本读取目标模型已融合的视觉-语言状态，并在一次前向传播中完成整块草拟与宽候选树验证，从而打破了草拟器因规模受限而被迫牺牲视觉信息的自我挫败循环。 |
| [^175] | [TwinKV: A Composable Repair Pass for KV Cache Eviction via Pairwise Key Redundancy](https://arxiv.org/abs/2608.27128) | 本文提出TwinKV，一种无需训练的冗余信号修复方法，通过检测键的重复性来优化KV缓存驱逐，挑战了注意力评分前提，可组合地提升现有策略性能。 |
| [^176] | [The Latent Diagnostic Taxonomy: A Framework for Constructing Classifiers and Diagnosing Their Decisions, Applied to Prompt Injection Detection](https://arxiv.org/abs/2608.26423) | 本文提出了一种潜在诊断分类法框架，通过维度优化分类器、识别潜在支持向量和构建诊断分类法，为提示注入检测提供了一种可靠决策与风险标记的端到端指南。 |
| [^177] | [Regime-Conditional Verification: Correctness Estimation for Adapting and Monitoring Safety Classifiers](https://arxiv.org/abs/2608.14089) | 本文提出了一种轻量级包装器RCV，通过估计分类器预测与部署者策略不一致的概率并选择性纠正，同时利用正确性估计检测分布漂移，实现了无需重训练即可适应和监控安全分类器，显著提升策略遵循度。 |
| [^178] | [QUASAR: Lowering the Loss Floor of Quantization-Aware Training with Loss-Aware Reconstruction](https://arxiv.org/abs/2608.13966) | 本文提出QUASAR，一种在量化感知训练过程中持续进行轻量级损失感知重构的方法，以降低损失下限并提升低比特模型质量。 |
| [^179] | [EvoHarness-RL: Learning Runtime Harness Coordination for Self-Evolving Agents](https://arxiv.org/abs/2608.05446) | 提出EvoHarness-RL统一框架，通过将环境特定支架实现与共享策略接口分离、构建信念-进度-经验（BPE）工作空间，并采用监督初始化加代价感知GRPO使支架协调可学习，显著提升长程LLM智能体的运行时支撑能力。 |
| [^180] | [Evolving language compositionality in a frequency-structured meaning space](https://arxiv.org/abs/2607.29642) | 本研究通过迭代学习模型发现，意义出现的频率会影响语言组合性的演化：高频意义可以像自然语言中一样摆脱低频语法规律的约束，但若频率差异体现在意义向量的内部成分而非整体上，语言将无法跨代传递。 |
| [^181] | [Diagnosing Fine-Grained Inconsistency Classification in Financial Disclosure Text](https://arxiv.org/abs/2607.26368) | 该研究提出金融披露文本的细粒度不一致性分类任务，在统一评估协议下系统比较了多种模型方法，发现微调的3亿参数编码器可与大得多的提示大语言模型和LoRA适配模型相媲美，并进一步探究了冲突声明定位对分类性能的改善作用。 |
| [^182] | [Refusal-Gated Decoding: Preserving Refusal Behavior Under High-Temperature Sampling](https://arxiv.org/abs/2607.20791) | 提出拒绝门控解码（RGD），一种高效解码方法，能在高温采样下保持模型原有的拒绝行为，同时对其他提示直接从精确的高温分布采样，且几乎不增加额外延迟。 |
| [^183] | [Token-Level Off-Policy Learning for Faithful Generation Under Distribution Shift](https://arxiv.org/abs/2607.17524) | 提出令牌级离策略标注（TOPL）训练范式，将后训练重构为令牌级正确性预测任务，使模型在摘要、翻译等忠实生成任务中实现强大的分布外泛化能力。 |
| [^184] | [Hearing Like Humans? Sound Symbolism and Perceptual Alignment in Speech Language Models](https://arxiv.org/abs/2607.10162) | 该研究发现语音语言模型在音义象征效应上的听觉判断与人类感知严重不符，其瓶颈不在视觉而在语音表征——它们无法捕捉频谱倾斜等驱动人类直觉的关键声学线索。 |
| [^185] | [Evaluating Large Language Model Raters for German Open-Response Clinical Questions: A Physician-Annotated Benchmark Study of Agreement, Evaluator Bias, and Abstention](https://arxiv.org/abs/2607.01103) | 该论文提出了MedQADE，一个由10位医师标注、包含3,800个问答集的标准化德语开放式临床问题基准，用于验证大语言模型裁判与医师评分的一致性，并系统揭示其自我偏差、同族偏差与弃权行为。 |
| [^186] | [Characterize Then Distill: Mechanistic Reasoning in Large Output Spaces](https://arxiv.org/abs/2606.06840) | 本文通过将多标签决策建模为token级事件，并结合归因、消融与植入等因果分析手段，刻画了推理型大模型在海量候选标签空间中进行选择的内部注意力头机制，并证明该机制可以被蒸馏。 |
| [^187] | [SubtleMemory: A Benchmark for Fine-Grained Relational Memory Discrimination in Long-Horizon AI Agents](https://arxiv.org/abs/2606.05761) | 提出了SubtleMemory基准，通过构建关系受控的语义工件并嵌入真实的用户-智能体交互历史中，系统评估长期运行AI智能体在细粒度关系记忆判别（包括互补、细微及矛盾关系）方面的能力。 |
| [^188] | [Coding with "Enemy": Can Human Developers Detect AI Agent Sabotage?](https://arxiv.org/abs/2606.05647) | 本研究首次大规模研究了人类监督在AI编程破坏行为中的作用，发现在无监控条件下高达94%的开发者（83/88）未能检测到AI智能体植入的破坏行为。 |
| [^189] | [Seeing Isn't Knowing: Do VLMs Know When Not to Answer Spatial Questions (and Why)?](https://arxiv.org/abs/2605.30557) | 该论文提出SPATIALUNCERTAIN受控评估框架，系统研究视觉语言模型在面对遮挡导致的证据缺失和视角导致的证据误导时的表现，强调可靠空间推理还要求模型能够判断当前观察是否足以支撑答案并主动识别更有信息量的观察视角。 |
| [^190] | [Emotion Recognition in Sign Language Conversation](https://arxiv.org/abs/2605.23328) | 本文将对话情感识别任务引入手语视频分析，构建了包含1,920个视频样本、480个对话的eJSL Dialog数据集，并通过系统性基准测试验证了对话式情感识别框架在手语场景中的可行性。 |
| [^191] | [Reinforcement Learning over Predictive Distributions for LLM Regression](https://arxiv.org/abs/2605.20740) | 提出了分布感知奖励（DAR），一种同策略强化学习目标，通过留一法贡献对同一输入的多个预测所形成的预测分布进行联合评估，从而提升大语言模型回归的校准质量。 |
| [^192] | [When Attention Closes: How LLMs Lose the Thread in Multi-Turn Interaction](https://arxiv.org/abs/2605.12922) | 该论文提出“通道转换”机制来解释大语言模型在多轮交互中丢失指令主线的原因，并引入目标可及性比率这一新指标，揭示不同架构的模型在注意力衰减后呈现出性质截然不同的失败模式。 |
| [^193] | [Rethinking Adapter Placement: A Dominant Adaptation Module Perspective](https://arxiv.org/abs/2605.06183) | 该论文提出PAGE探测方法，发现LoRA可训练梯度能量高度集中于单一浅层FFN下投影的“主导适配模块”，其位置由模型架构决定且跨任务稳定，为有限数量适配器的最优放置提供了明确指导。 |
| [^194] | [Cooperative Profiles Predict Multi-Agent LLM Team Performance in AI for Science Workflows](https://arxiv.org/abs/2604.20658) | 大语言模型在行为经济学博弈中体现出的合作行为特征，能够稳健预测其在共享资源约束下开展AI for Science协作任务时的团队表现。 |
| [^195] | [Algorithm Selection with Zero Domain Knowledge via Text Embeddings](https://arxiv.org/abs/2604.19753) | ZeroFolio利用预训练文本嵌入替代手工设计的实例特征，实现了零领域知识的算法选择，在涵盖7个领域的11个ASlib场景中的绝大多数上超越了传统方法。 |
| [^196] | [Model in Distress: Sentiment Analysis on French Synthetic Social Media](https://arxiv.org/abs/2604.18226) | 该论文开发了一个基于反向翻译的可泛化合成数据生成流水线，生成了170万条法语合成推文，训练出的6亿参数推理模型在法国公共交通客户困境检测任务上达到77-79%的准确率，匹敌甚至超越专有大语言模型，同时有效保护了用户隐私并降低了标注成本。 |
| [^197] | [Cross-Cultural Value Attribution in Large Vision-Language Models](https://arxiv.org/abs/2604.09945) | 该论文首次系统研究大型视觉语言模型在道德、伦理和政治价值观判断上如何随图像中人物的文化语境（宗教、国籍、社会经济地位）而变化，通过反事实图像集与多维度评估框架揭示其中的跨文化刻板印象与公平性问题。 |
| [^198] | [Unbiased Reward Modeling from Implicit Feedback for LLM Alignment](https://arxiv.org/abs/2603.23184) | 该论文提出ImplicitRM方法，通过将训练样本分层为四个潜在组并构建理论无偏的似然最大化目标，从点击、复制等隐式用户反馈中学习无偏奖励模型，从而解决了隐式反馈缺乏明确负样本和存在选择偏差这两大挑战。 |
| [^199] | [Understanding Moral Reasoning Trajectories in Large Language Models: Toward Probing-Based Explainability](https://arxiv.org/abs/2603.16017) | 该论文提出“道德推理轨迹”这一新概念，揭示了大语言模型在道德推理中系统性地在多个伦理框架间切换、且框架切换频繁的轨迹更易受说服性攻击，并通过线性探测和激活引导技术定位并调控模型中的道德框架表征，为LLM道德推理的可解释性研究提供了新路径。 |
| [^200] | [Quantifying Cross-Lingual Transfer in Paralinguistic Speech Tasks](https://arxiv.org/abs/2603.08231) | 本文提出跨语言迁移矩阵（CLTM）方法，系统量化了性别识别和说话人验证等副语言学语音任务中语言对之间的迁移效应，揭示了不同任务和语言间存在截然不同的迁移模式。 |
| [^201] | [Dual-Modality Multi-Stage Adversarial Safety Training: Robustifying Multimodal Web Agents Against Cross-Modal Attacks](https://arxiv.org/abs/2603.04364) | 该论文提出双模态多阶段对抗性安全训练（DMAST）框架，将代理与攻击者的交互建模为两人一般和马尔可夫博弈，通过三阶段协同训练显著增强多模态网页代理抵御同时污染视觉与文本两个观察通道的跨模态欺骗攻击的能力。 |
| [^202] | [World Properties without World Models: Distributional Associations and the Interpretation of Decoding Results from Language Models](https://arxiv.org/abs/2603.04317) | 该研究表明，以往从语言模型激活中解码出的“世界属性”在很大程度上也能由静态词嵌入实现，因此解码成功并不能证明语言模型形成了内部世界模型，而可能仅反映了语料库统计中的分布性关联。 |
| [^203] | [Real-Time Generation of Game Video Commentary with Multimodal LLMs: Pause-Aware Decoding Approaches](https://arxiv.org/abs/2603.02655) | 提出两种无需微调的基于提示的停顿感知解码策略（固定间隔与根据话语估计时长动态调整间隔），使多模态大语言模型能够生成语义相关且时机恰当的实时游戏视频解说。 |
| [^204] | [[b] = [d] - [t] + [p]: Self-supervised Speech Models Discover Phonological Vector Arithmetic](https://arxiv.org/abs/2602.18899) | 自监督语音模型在96种语言的表示空间中将音系特征编码为线性向量，且这些向量支持算术运算（如将[d]-[t]得到的浊音向量加到[p]上即产生[b]），表明语音以可解释、可组合的音系向量形式被表征。 |
| [^205] | [RAM-Net: Linear-Time Sequence Modeling with Sparsely Addressable State](https://arxiv.org/abs/2602.11958) | RAM-Net提出用稀疏地址访问取代共享状态的密集访问，将循环状态组织为独立槽数组并通过地址解码器选择少量槽进行读写，从而抑制词元间干扰、提升长距离细粒度记忆能力，同时保持线性时间复杂度。 |
| [^206] | [Uncovering Cross-Objective Interference in Multi-Objective Alignment](https://arxiv.org/abs/2602.06869) | 该论文首次系统研究了多目标LLM对齐中的跨目标干扰现象，推导出解释其成因的局部协方差定律，并据此提出了缓解干扰的重加权控制器COVER。 |
| [^207] | [Quantifying the Gap between Understanding and Generation within Unified Multimodal Models](https://arxiv.org/abs/2602.02140) | 本文提出双向基准GapEval，用于量化统一多模态模型中理解与生成能力之间的差距，实验揭示现有模型仅实现了表层统一而非深层的认知融合。 |
| [^208] | [AVMeme Exam: A Multimodal Multilingual Multicultural Benchmark for LLMs' Contextual and Cultural Knowledge and Thinking](https://arxiv.org/abs/2601.17645) | 该论文提出了AVMeme考试基准，利用一千多个标志性网络音视频梗评估多模态大语言模型，发现当前模型在无文本音乐、音效以及语境化和文化层面的理解能力上显著落后于人类。 |
| [^209] | [Cross-Lingual Activation Steering for Multilingual Language Models](https://arxiv.org/abs/2601.16390) | 提出无需训练的推理时干预方法CLAS，通过有选择地调节神经元激活提升非主导语言的性能，且不损害高资源语言表现。 |
| [^210] | [Zero-Shot Lombard Speech Synthesis with Controllable Style Embeddings](https://arxiv.org/abs/2601.12966) | 该论文提出了一种零样本可控TTS系统，通过在F5-TTS中引入风格嵌入并利用PCA分析潜在空间，无需伦巴德训练数据即可合成不同伦巴德程度的语音，在保持说话人身份和自然度的同时提升噪声环境下的语音可懂度。 |
| [^211] | [TabiBERT: A Large-Scale ModernBERT Foundation Model and A Unified Benchmark for Turkish](https://arxiv.org/abs/2512.23065) | TabiBERT是基于ModernBERT架构从零预训练的土耳其语单语编码器，在一万亿token上训练，支持8,192 token上下文长度（为现有土耳其语BERT的16倍），并提供了土耳其语的统一评测基准。 |
| [^212] | [Stabilizing Off-Policy Training for Long-Horizon LLM Agent via Turn-Level Importance Sampling and Clipping-Triggered Normalization](https://arxiv.org/abs/2511.20718) | 该论文提出SORL方法，通过轮次级重要性采样与截断触发的归一化机制，使策略优化与多轮交互结构对齐并抑制不可靠的离策略梯度更新，从而稳定长时程LLM智能体的离策略强化学习训练，防止性能崩溃。 |
| [^213] | [Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond)](https://arxiv.org/abs/2510.22954) | 该论文提出了包含26K条开放式查询的大规模数据集Infinity-Chat以及首个开放式提示综合分类体系，并据此对语言模型的模式坍缩进行大规模研究，揭示了语言模型在开放式生成中输出高度同质化的“人工蜂群思维”效应。 |
| [^214] | [VietBinoculars: A Zero-Shot Approach for Detecting Vietnamese LLM-Generated Text](https://arxiv.org/abs/2509.26189) | VietBinoculars是一个针对越南语的零样本LLM生成文本检测框架，通过结合PhoGPT-4B观察者/执行者模型、校准的全局决策阈值和专门的越南语BPE分词，实现了超过0.99的AUC和至少98.78%的检测准确率，显著优于现有检测工具。 |
| [^215] | [PiERN: Token-Level Routing for Integrating High-Precision Computation and Reasoning](https://arxiv.org/abs/2509.18169) | PiERN提出了一种物理隔离专家路由网络架构，通过在令牌级别路由计算与推理，使大型语言模型能够在单一思维链中迭代交替地执行高精度数值计算与推理，在准确率、响应延迟、令牌消耗和能耗等方面均优于直接微调与主流多智能体方法。 |
| [^216] | [Nord-Parl-TTS: Finnish and Swedish TTS Dataset from Parliament Speech](https://arxiv.org/abs/2509.17988) | 该论文发布了基于北欧议会演讲录音构建的开源TTS数据集Nord-Parl-TTS，包含900小时芬兰语和5090小时瑞典语语音，有效缩小了TTS领域高资源语言与低资源语言之间的数据资源差距。 |
| [^217] | [Breaking the Mirror: Activation-Based Mitigation of Self-Preference in LLM Evaluators](https://arxiv.org/abs/2509.03647) | 本文提出利用通过对比激活添加（CAA）和优化方法构建的转向向量，在无需重新训练的情况下于推理时缓解LLM评估器的不合理自我偏好偏差，最多可降低97%，显著优于提示和直接偏好优化基线。 |
| [^218] | [FedCoT: Communication-Efficient Federated Reasoning Enhancement for Large Language Models](https://arxiv.org/abs/2508.10020) | FedCoT是一个通信高效的联邦推理增强框架，通过轻量级思维链重采样、紧凑判别器筛选以及客户端感知的LoRA堆叠加权聚合，在无需集中式蒸馏和保护隐私的前提下增强大语言模型的逐步推理能力，特别适用于医疗等需要可解释、可审计决策的场景。 |
| [^219] | [Too Categorical to be Human: Emotion Concepts in LLMs and Humans](https://arxiv.org/abs/2508.05880) | 该论文提出基于认知评估理论、以“行为表征”刻画情绪概念，并构建涵盖15个情绪类别的基准数据集来比较LLMs与人类的情绪概念表征，发现LLMs的情绪概念过于范畴化而与人类不同。 |
| [^220] | [Pronunciation Editing for Finnish Speech using Phonetic Posteriorgrams](https://arxiv.org/abs/2507.02115) | 提出了基于扩散模型的多说话人PPG2Speech模型，通过增强Matcha-TTS流匹配解码器，无需文本对齐即可编辑单个音素，从而将母语语音转换为近似的芬兰语第二语言语音。 |
| [^221] | [Common Corpus: The Largest Collection of Ethical Data for LLM Pre-Training](https://arxiv.org/abs/2506.01732) | 本文发布了Common Corpus——目前最大的用于大语言模型预训练的开放数据集，包含约两万亿词元的无版权或开放许可数据，涵盖从高资源到低资源的多种语言及大量代码数据，为合乎伦理且合规的大模型训练奠定基础。 |
| [^222] | [Even Small Reasoners Should Quote Their Sources: Introducing the Pleias-RAG Model Family](https://arxiv.org/abs/2504.18225) | Pleias-RAG-350m和Pleias-RAG-1B是专为RAG设计的小型推理模型，原生支持引文溯源和查询处理功能，在标准基准上超越同级别模型并可与更大模型媲美，且在多语言环境中保持一致的高性能表现。 |
| [^223] | [Boosting Large Language Models with Mask Fine-Tuning](https://arxiv.org/abs/2503.22764) | 提出掩码微调（MFT）这一新颖的大语言模型微调范式，通过学习并应用二值掩码、在不更新模型权重的情况下精心打破模型结构完整性，从而在不同领域和骨干网络上获得一致的性能提升。 |
| [^224] | [Enhancing High-order Interaction Awareness in LLM-based Recommender Model](https://arxiv.org/abs/2409.19979) | 本文提出增强型LLM推荐器ELMRec，通过增强全词嵌入使大语言模型无需图预训练即可理解用户-项目高阶交互，并针对LLM偏好早期交互而忽略近期交互的问题提出重排序方案，在直接推荐和序列推荐中均超越现有最先进方法。 |

# 详细

[^1]: IdeaAnchor：教会大语言模型将文献转化为研究思路

    IdeaAnchor: Teaching LLMs to Turn Literature into Research Ideas

    [https://arxiv.org/abs/2610.08781](https://arxiv.org/abs/2610.08781)

    提出IdeaAnchor范式，通过编码论文功能角色、关系及综合标准的结构化规格说明作为特权信号，结合示范、自蒸馏与强化学习训练大语言模型，使其学会将文献综合为研究思路。

    

    科学研究往往始于从一组相关论文中综合提炼想法，以发现研究空白并提出新的研究方向。然而，训练语言模型完成这种基于文献的构思（ideation）仍然具有挑战性，因为现有的基于提示或反馈的方法缺乏关于论文应如何被综合的结构化监督信号。我们提出IdeaAnchor，一种利用结构化规格说明作为特权信号来训练大语言模型进行研究构思的范式。每个IdeaAnchor实例编码了每篇输入论文应如何被综合成一个成功的研究想法，包括它们的功能角色、相互关系以及目标综合标准。我们通过从已发表论文中挖掘实例来构建该范式，捕捉真实研究想法如何从既有文献中产生。随后，我们通过示范学习、自蒸馏和强化学习训练模型，并在推理时结合检索进一步增强生成效果。

    arXiv:2610.08781v1 Announce Type: cross  Abstract: Scientific research often begins by synthesizing ideas from a set of related papers to identify gaps and formulate new directions. However, training language models to perform this form of literature-grounded ideation remains challenging, as existing approaches based on prompting or feedback lack structured supervision for how papers should be synthesized. We introduce IdeaAnchor, a paradigm for training LLMs to perform research ideation using structured specifications as privileged signals. Each IdeaAnchor instance encodes how each input paper should be synthesized into a successful idea, including their functional roles, relationships, and target synthesis criteria. We build this paradigm by mining instances from published papers, capturing how real ideas emerge from prior literature. We then train models via demonstration, self-distillation, and reinforcement learning, and further enhance generation with retrieval at inference time.
    
[^2]: Sherpa：教会大语言模型自适应地教学

    Sherpa: Teaching LLMs to Teach Adaptively

    [https://arxiv.org/abs/2610.08778](https://arxiv.org/abs/2610.08778)

    Sherpa是一个多轮强化学习框架，通过模拟具有不同学习偏好的学生原型并直接最大化其学习成效，训练教师LLM自适应地调整教学策略，使受教学生的成绩平均提升20.5个百分点。

    

    大语言模型（LLM）作为问题求解者的能力日益增强，但能够解决一个问题并不等同于能够教授这个问题。现有的将LLM训练为教师的方法依赖于演示、偏好数据或预先定义的教学标准来规定什么是好的教学。然而，这些信号通常并未基于学生个体的学习成效，而有效的教学策略在不同学习者之间可能存在显著差异。为了解决这一问题，我们提出了Sherpa，这是一个多轮强化学习框架，它利用基于不同学习偏好条件化的LLM实例化多种学生原型，并通过直接最大化这些学生的学习成效来训练教师模型自适应地调整其教学方式。使用Sherpa训练的教师LLM使所有学生原型下的受教学生成绩平均提高了20.5个百分点。在MathTutorBench的评估下，Sherpa

    arXiv:2610.08778v1 Announce Type: new  Abstract: Large language models (LLMs) have become increasingly capable problem solvers, but being able to solve a problem is not the same as being able to teach it. Existing approaches to training LLMs as teachers rely on demonstrations, preference data, or predefined pedagogical criteria that specify what good teaching looks like. However, these signals are often not grounded in individual student learning outcomes, where effective teaching strategies can vary substantially across learners. To address this, we introduce Sherpa, a multi-turn reinforcement learning framework that instantiates multiple student archetypes with LLMs conditioned on distinct learning preferences and trains a teacher model to adapt its instruction by directly maximizing their learning outcomes. Teacher LLMs trained with Sherpa improve instructed students' performance across all archetypes by an average of 20.5 percentage points. Under MathTutorBench's evaluation, Sherpa
    
[^3]: AdvSim2Real：在网络世界模型中训练网络智能体以对抗自适应提示注入

    AdvSim2Real : Training Web Agents Against Adaptive Prompt Injection in a Web World Model

    [https://arxiv.org/abs/2610.08773](https://arxiv.org/abs/2610.08773)

    提出 AdvSim2Real，在冻结的网络世界模型中让任务课程、注入攻击者与智能体共同演化，通过“成功翻转”对抗奖励机制训练出既更强又更能抵御自适应提示注入攻击的网络智能体。

    

    arXiv:2610.08773v1 公告类型：cross 摘要：网络智能体通过阅读并操作由第三方编写的网页来完成用户请求，因此页面上被植入的指令可能会使智能体偏离用户的目标。然而智能体不能简单地忽略页面，因为页面中还包含任务所需的取值和控件。当前的防御方法是在训练前固定注入内容并对智能体进行微调，但适应了已训练模型的自适应攻击者可以绕过这些防御。对抗性训练虽然允许攻击者进行适应，但任务保持固定，因此一旦智能体解决了某个任务，该任务就不再具有训练价值。我们提出 AdvSim2Real，它在一个冻结的网络世界模型中共同演化任务课程、注入攻击者和智能体。任务课程因智能体约有半数概率能解决的任务而获得奖励，而攻击者仅因“成功翻转”——即能把被判定为成功的任务转变为失败的注入——而获得奖励。在模拟器中训练使一个 4B 的智能体在能力和鲁棒性方面都得到提升：其完成……

    arXiv:2610.08773v1 Announce Type: cross  Abstract: Web agents complete user requests by reading and acting on pages that third parties write, so an instruction planted on a page can redirect the agent away from the user's goal. The agent cannot simply ignore the page, because the page also holds the values and controls the task requires. Current defenses fine-tune the agent on injections fixed before training, and attackers that adapt to the trained model bypass them. Adversarial training lets the attacker adapt but keeps the tasks fixed, so a task stops teaching once the agent solves it. We introduce AdvSim2Real, which co-evolves a task curriculum, an injection adversary, and the agent inside a frozen web world model. The curriculum is rewarded for tasks the agent solves about half of the time, and the adversary only for a success flip, an injection that turns a judged success into a failure. Training in the simulator makes a 4B agent both more capable and more robust: its completion 
    
[^4]: 缺失的最小对：大语言模型中的刻板印象评估

    The Missing Minimal Pair: Stereotype Evaluation in LLMs

    [https://arxiv.org/abs/2610.08747](https://arxiv.org/abs/2610.08747)

    该论文提出双重最小对设置和基于互信息的评估指标，并结合数据增强框架，实现对大语言模型刻板印象偏见更可靠的多语言评估。

    

    衡量大语言模型中偏见的一种常见方法是比较两个对比性刻板印象句子的对数似然。我们认为这种单对比较往往不可靠：仅仅用另一个属性改写同一个刻板印象，就可能产生逻辑上不一致的偏好。为解决这一问题，我们提出了一个双重最小对设置，引入两个比较轴以实现稳健的刻板印象评估。首先，我们提出了一个数据增强框架，通过生成释义和替代属性来填补现有刻板印象数据集中的关键空白。我们将该框架应用于英语、俄语、西班牙语和汉语的刻板印象集合。其次，我们引入了两个为双重最小对设置量身定制的评估指标。其中一个指标通过建模社会群体与刻板印象属性之间的互信息（MI），为偏见研究提供了新的视角。这种基于互信息的指标更适合……（摘要在此处被截断）

    arXiv:2610.08747v1 Announce Type: new  Abstract: A common approach to measuring bias in Large Language Models is to compare the log-likelihoods of two contrastive stereotype sentences. We argue that such single-pair comparisons are often unreliable: simply rewriting the same stereotype with an alternative attribute can yield logically inconsistent preferences. To address this, we propose a dual minimal pair setup that introduces two axes of comparison for robust stereotype evaluation. First, we present a data-augmentation framework that fills critical gaps in existing stereotype datasets by generating paraphrases and alternate attributes. We apply our framework on a set of English, Russian, Spanish and Chinese stereotypes. Second, we introduce two evaluation metrics tailored to the dual minimal pair setup. One of these metrics provides a new perspective on bias by modeling the mutual information (MI) between social groups and stereotyped attributes. This MI-based metric is better suite
    
[^5]: 去噪层次化表示：面向语言建模的联合连续扩散

    Denoising Hierarchical Representations: Joint Continuous Diffusion for Language Modeling

    [https://arxiv.org/abs/2610.08738](https://arxiv.org/abs/2610.08738)

    提出层次化连续扩散语言模型（H-CDLM），通过并行扩散token本身及其粗粒度语义聚类等多种模态表示，并允许为各模态配置独立的采样器与调度，以极少的计算和参数开销显著提升连续扩散语言模型的性能。

    

    扩散语言模型有望实现与顺序无关的并行文本生成。近来，连续扩散和流匹配模型取得了显著进展，这得益于精心设计的token表示和扩散/流空间。在本工作中，我们提出了层次化连续扩散语言模型，这是一个简洁的框架，能够以极少的计算和参数开销进一步提升连续扩散语言模型的性能。借鉴离散扩散语言模型与连续图像扩散文献中关于联合扩散的思想，我们对多种模态进行并行扩散。这些模态在不同语义粒度上表示token：在我们的具体实现中，包括token本身以及通过对预训练token嵌入进行聚类所得到的更粗粒度的聚类。我们提出了一个通用设置，允许为每种模态配置各自的采样器和调度，以增强模态之间的相互作用。将该方法应用于CoBit时，得到了H-CoBit，它带来了巨大的性能提升。

    arXiv:2610.08738v1 Announce Type: new  Abstract: Diffusion Language Models (DLMs) hold the promise of order-agnostic, parallel text generation. Recently, continuous diffusion and flow matching models have seen substantial gains, driven by carefully crafted token representations and diffusion/flow spaces. In this work, we introduce Hierarchical Continuous Diffusion Language Models (H-CDLMs), a simple framework that further improves continuous DLMs with minimal compute and parameter overhead. Drawing on the discrete DLM and continuous image diffusion literature on joint diffusion, we diffuse multiple modalities in parallel. These modalities represent tokens at different semantic granularities: in our instantiation, the tokens themselves and coarser clusters obtained by clustering pretrained token embeddings. We propose a general setup that allows per-modality samplers and schedules to enhance the interplay between modalities. Applied to CoBit, this yields H-CoBit, which delivers large em
    
[^6]: 生成式信息检索中语义ID空间的系统性研究

    A Systematic Study of Semantic ID Spaces for Generative Information Retrieval

    [https://arxiv.org/abs/2610.08732](https://arxiv.org/abs/2610.08732)

    本文针对生成式信息检索中的核心问题“什么样的DocID才是好的DocID”，首次提出统一框架将乘积量化、残差量化及其混合变体纳入单一设计空间，系统研究了定义有效数值DocID的属性、指标与权衡，从而摆脱对昂贵下游评估的依赖，支持系统性分析与快速迭代。

    

    生成式信息检索（GIR）已成为一种变革性范式，它将文档检索从传统的“检索-排序”工作流程转变为序列到序列的生成方式，由模型直接预测文档标识符。尽管这些DocID的语义设计对性能至关重要，但一个根本性问题仍未得到充分探索：什么样的DocID才是好的DocID？当前方法严重依赖计算代价高昂的下游评估，这阻碍了系统性分析和快速迭代。在这项工作中，我们通过全面研究定义有效数值DocID的属性、指标与权衡来应对这一挑战。具体而言，我们的贡献有三方面：首先，我们提出了一个统一框架，将乘积量化（PQ）、残差量化（RQ）及其混合变体统一到单一设计空间中，这使我们能够系统性地……（摘要原文在此处截断）

    arXiv:2610.08732v1 Announce Type: cross  Abstract: Generative Information Retrieval (GIR) has emerged as a transformative paradigm, shifting document retrieval from a traditional "retrieve-and-rank" workflow to sequence-to-sequence generation, where a model directly predicts document identifiers (DocIDs). While the semantic design of these DocIDs is known to be critical for performance, a fundamental question remains under-explored: what makes a good DocID? Current approaches rely heavily on computationally expensive downstream evaluations, hindering systematic analysis and rapid iteration. In this work, we address this challenge by presenting a comprehensive study on the properties, metrics, and trade-offs that define effective numerical DocIDs. Specifically, our contributions are threefold: First, we propose a unified framework that unifies Product Quantization (PQ) and Residual Quantization (RQ), and their hybrid variants within a single design space. This enables us to systematical
    
[^7]: 留出式Best-of-N：无偏评估及其代价

    Holdout Best-of-N: Unbiased Evaluation and Its Cost

    [https://arxiv.org/abs/2610.08719](https://arxiv.org/abs/2610.08719)

    本文证明了仅当用于选择的分数数J小于总分数数K时才能对Best-of-N策略进行严格无偏评估，此时留出法估计量达到σ²/√K的最优无偏极小极大风险，而允许偏差可将速率提升至σ²/K。

    

    重用用于挑选Best-of-N获胜者的分数可能会高估其期望奖励。我们研究了基于每个候选者K个独立分数的固定矩阵来评估一个使用J个新鲜分数进行选择的策略。一个仅基于该矩阵的估计量，在候选者特定分数规律的每个独立、稳定集合下，对期望评判奖励均严格无偏，当且仅当J<K，且适用于所有池规模M≥N≥2。在J=K-1时，选择器随K的增长而加深。对于具有共同方差的独立高斯分数以及固定的M≥N≥2，该机制下的无偏极小极大风险为σ²/√K量级，由留出法达到；而允许有偏可将速率改进至σ²/K。对于两个候选者，我们在已知方差时导出了最小方差无偏估计量，以及精确的渐近无偏极小极大常数1/(π√2)，而留出法在不知道方差的情况下即可达到该常数。基于子集的循环平均……（原文摘要在此处截断）

    arXiv:2610.08719v1 Announce Type: new  Abstract: Reusing the scores that select a Best-of-$N$ winner can overstate its expected reward. We study evaluation from a fixed matrix of $K$ independent scores per candidate for a policy that selects using $J$ fresh scores. A single estimator based only on this matrix is exactly unbiased for expected judge reward under every independent, stable collection of candidate-specific score laws if and only if $J<K$, for every pool size $M\ge N\ge2$. At $J=K-1$, the selector deepens as $K$ grows. For independent Gaussian scores with common variance and fixed $M\ge N\ge2$, the unbiased minimax risk in this regime is of order $\sigma^2/\sqrt K$, attained by Holdout; allowing bias improves the rate to $\sigma^2/K$. For two candidates, we derive the minimum-variance unbiased estimator at known variance and the sharp asymptotic unbiased minimax constant $1/(\pi\sqrt2)$, which Holdout attains without knowing the variance. The cyclic average over subsets and 
    
[^8]: 当遗忘并非灾难性时：论虚假遗忘的机制

    When Forgetting is not Catastrophic: On the Mechanics of Spurious Forgetting

    [https://arxiv.org/abs/2610.08718](https://arxiv.org/abs/2610.08718)

    该研究揭示了语言模型微调中虚假遗忘的力学机制——微调使旧表征沿共同方向偏移导致暂时的知识隐藏，归一化会撤销该偏移使知识自行恢复，而只有事实特定的累积变化才会造成真正的永久遗忘。

    

    语言模型在微调过程中似乎遗忘的知识往往仍存储在模型中且可以被恢复，这种现象被称为虚假遗忘。在新事实数据上进行微调甚至会产生一种会自我逆转的遗忘：对旧事实的召回能力先崩溃，随后随着仅在新事实上的继续训练而恢复，之后才最终永久性衰退。我们试图理解这种遗忘何时不是灾难性的。一个最小化的联想记忆模型仅用三个要素就再现了这些动态：具有共享结构的键、集中的新值以及网络中的归一化机制。微调使所有旧表征沿着一个共同方向移动，从而在保持其相对几何结构的同时隐藏旧事实；一旦新事实被学会，归一化机制会撤销这种偏移，而事实特定的变化则会不断累积并最终导致衰退。此外，减去这一共同偏移可以消除在合成数据上训练的Transformer中的崩溃现象，移除……

    arXiv:2610.08718v1 Announce Type: new  Abstract: Knowledge that a language model appears to forget during finetuning often remains stored and can be recovered, a phenomenon called spurious forgetting. Finetuning on new facts can even produce forgetting that undoes itself: recall of the old facts collapses, recovers as training continues on new facts alone, and only then erodes for good. We seek to understand when such forgetting is not catastrophic. A minimal associative memory reproduces these dynamics with three ingredients: keys with shared structure, concentrated new values, and normalization in the network. Finetuning moves all old representations along a common direction, hiding the old facts while preserving their relative geometry; normalization withdraws this shift once the new facts are learned, whereas fact-specific changes accumulate and cause the erosion. Moreover, subtracting the common shift eliminates the collapse in a Transformer trained on synthetic data, and removing
    
[^9]: 解耦生成式检索中的范式、标识符与解码方法

    Disentangling Paradigm, Identifier, and Decoding in Generative Retrieval

    [https://arxiv.org/abs/2610.08716](https://arxiv.org/abs/2610.08716)

    该论文通过控制变量实验首次解耦了生成式检索中范式、标识符和解码三个因素，发现仅解码方式就能使扩散检索器的Hit@1波动6.6至13.7个百分点，并提出单次评分解码方法，使模型一次性读取掩码标识符并按编码概率为文档打分，从而大幅提升扩散检索性能。

    

    生成式检索训练语言模型来生成相关文档的标识符。近期工作用扩散模型取代自回归解码器，但同时改变了标识符、训练方案和解码方式，因此性能差异无法归因于范式本身。在NQ320K和MS300K数据集上，我们使用残差量化、乘积量化和随机标识符训练了自回归、掩码扩散和块扩散模型。在固定标识符长度和训练预算的条件下，我们用多种方式对每个模型进行解码。仅解码方式一项就能使扩散模型的Hit@1指标变动6.6至13.7个百分点。我们的参考扩散解码方法“生成-匹配”先生成一个标识符，然后检索语料库中最接近的标识符，而生成标识符完全正确的比例在NQ320K查询中仅为14-21%。我们测试了一种单次评分方法来解码扩散检索器：模型一次性读取完全掩码的标识符，每个文档根据其编码的概率进行评分。

    arXiv:2610.08716v1 Announce Type: cross  Abstract: Generative retrieval trains a language model to generate the identifier of a relevant document. Recent work replaces the autoregressive decoder with diffusion, but changes identifiers, training recipe and decoding at once, so differences cannot be credited to the paradigm. On NQ320K and MS300K, we train autoregressive, masked-diffusion and block-diffusion models with residual-quantised, product-quantised and random identifiers. With identifier length and training budget fixed, we decode each model in several ways. Decoding alone moves a diffusion model's Hit@1 by 6.6 to 13.7 points. Our reference diffusion decoding, generate-and-match, generates an identifier, then retrieves the closest corpus identifiers. The generated identifier is right for 14-21% of NQ320K queries. We test one-pass scoring to decode diffusion retrievers: the model reads a fully masked identifier once, and each document is scored by its codes' probabilities. It matc
    
[^10]: 一致性并非有效性：K-12数学辅导对话中学生失败模式诊断中的跨模型LLM共识

    Agreement Is Not Validity: Cross-Model LLM Consensus in Diagnosing Student Failure Modes in K-12 Math Tutoring Dialogue

    [https://arxiv.org/abs/2610.08703](https://arxiv.org/abs/2610.08703)

    该研究警示“一致”不等于“有效”：LLM之间的跨模型共识远高于其与人类的一致性，因此不能用模型间的高共识来替代人类验证以证明LLM诊断学生数学学习失败的有效性。

    

    在K-12数学辅导中，学生与辅导者之间的对话为学习者的问题解决过程及其困难来源提供了丰富的证据。学习分析研究越来越依赖大语言模型（LLM）从对话中提取此类信息，以支持知识追踪、行为建模和学生推理错误诊断等多种下游任务。然而，这些模型生成的解释的有效性仍未得到充分理解。在这项探索性研究中，我们使用一套操作性诊断编码本，检验了LLM对数学辅导对话中五种学生失败模式（不确定性、错误归因、运算符选择、概念缺口和程序性失误）进行分类的有效性。结果显示，各模型与人类之间的一致性为中等水平（kappa = .524-.597），而模型之间的一致性则显著更高（kappa = .755-.781；alpha = .769）。这些发现表明，跨模型一致性高并不等同于有效。

    arXiv:2610.08703v1 Announce Type: new  Abstract: In K-12 mathematics tutoring, student-tutor dialogue provides rich evidence of learners' problem-solving processes and sources of difficulty. Learning analytics research increasingly relies on large language models (LLMs) to extract such information from dialogue for a variety of downstream tasks, including knowledge tracing, behavioral modeling, and diagnosis of student reasoning errors. However, the validity of these model-generated interpretations remains insufficiently understood. In this exploratory study, we examine the validity of LLM classifications of five student failure modes in mathematics tutoring dialogue using an operational diagnostic codebook: uncertainty, misattribution, operator selection, conceptual gap, and procedural slip. Across models, human-LLM agreement was moderate (kappa = .524-.597), while cross-model agreement was substantially higher (kappa = .755-.781; alpha = .769). These findings show that cross-model ag
    
[^11]: 小型语言模型在抽象推理任务上的系统性研究

    A Systematic Study of Small Language Models on Abstract Reasoning Tasks

    [https://arxiv.org/abs/2610.08680](https://arxiv.org/abs/2610.08680)

    本文系统研究了小型语言模型在ARC-TGI抽象推理基准上的表现，发现尽管模型能取得较高的分布内准确率，但其技能获取对优化过程敏感、在各任务族间分布不均，且分布外性能急剧下降，表明模型可能只是拟合了特定分布的规律而未真正学到可迁移的规则。

    

    抽象推理基准上的终点准确率并不能揭示语言模型是真正获得了可迁移的规则，还是仅仅拟合了特定分布的规律性。我们在小型语言模型上，基于ARC-TGI基准来研究这一区别，该基准将抽象网格变换组织为可控的任务族，并支持重采样、空间平移以及跨基准迁移。在超过1,000次运行中，我们在监督微调设置下对仅解码器、编码器-解码器以及混合专家等模型家族进行了系统剖析。我们考察了技能获取的效率与稳定性、超出训练分布时的鲁棒性、模型家族与任务形式之间的交互作用，以及伴随行为差异出现的逐层注意力特征。研究发现，尽管可以获得可观的分布内准确率，但技能获取对优化过程较为敏感，且在各任务族之间的分布并不均衡。模型性能在分布外急剧下降……

    arXiv:2610.08680v1 Announce Type: cross  Abstract: Endpoint accuracy on abstract-reasoning benchmarks does not reveal whether a language model has acquired a transferable rule or fit distribution-specific regularities. We study this distinction in small language models on the ARC-TGI benchmark, which organizes abstract grid transformations into controllable task families and supports resampling, spatial shifts, and cross-benchmark transfer. Across more than 1,000 runs, we profile decoder-only, encoder--decoder, and mixture-of-experts model families under supervised fine-tuning. We examine the efficiency and stability of skill acquisition, robustness beyond the training distribution, interactions with model family and task formulation, and layer-wise attention signatures that accompany behavioral differences. Substantial in-distribution accuracy is attainable, but acquisition is sensitive to optimization and unevenly distributed across task families. Performance deteriorates sharply out
    
[^12]: 同数引用互换：对Jev作为金融证据裁判的压力测试

    Same-Number Citation Swaps: Stress-Testing Jev as a Financial Evidence Judge

    [https://arxiv.org/abs/2610.08675](https://arxiv.org/abs/2610.08675)

    该论文通过对GPT-4.1-mini计算轨迹在相同数字单元格之间互换引用进行压力测试，发现Jev等概率证据验证器在金融报告数值大量重复的场景下仍会放过引用错误财务角色的计算并拒绝等价的有效引用，而显式列标签只能部分改善这一角色识别难题。

    

    金融报告中的数值会在不同期间、不同指标和不同会计科目之间重复出现，这使得LLM生成的计算可能在数值上正确，却引用了错误的财务角色。我们以Jev作为GPT-4.1-mini计算轨迹的来源支持验证器，评估概率性证据验证在数字匹配之外带来了什么。一个基于指针位置带符号数字的基线解释了相较于精确引用检查的大部分恢复效果。为了分离剩余的角色识别问题，我们保持操作数和算术运算不变，将引用在相同数字的单元格之间移动，并保留表达等价事实的对照组。这些对比既揭示了能够通过验证的错误角色引用，也揭示了被拒绝的有效替代引用。显式列标签改善了对部分错误角色决策的判断，同时也降低了对某些等价证据的支持。一项由非作者评审员标注、在36个新来源页面上构建的后续实验扩展了这一评估……（摘要原文在此处截断）

    arXiv:2610.08675v1 Announce Type: new  Abstract: Financial reports repeat values across periods, metrics and accounting lines, allowing an LLM-generated calculation to be numerically correct while citing the wrong financial role. We evaluate what probabilistic evidence verification adds beyond number matching using Jev as a source-support verifier for GPT-4.1-mini calculation traces. A signed-number-at-pointer baseline explains most recovery over exact quotation checks. To isolate the remaining role-recognition problem, we hold operands and arithmetic fixed, move citations between same-number cells, and retain controls that express equivalent facts. These contrasts reveal both wrong-role citations that pass and valid alternative citations that are withheld. Explicit column labels improve selected wrong-role decisions while also lowering support for some equivalent evidence. A constructed follow-up on 36 new source pages, labeled by a non-author reviewer, extends this evaluation and exp
    
[^13]: 压力下的原则坚守：后训练决定大语言模型是否会践行自己的道德判断

    Principled Under Pressure: Post-Training Decides Whether LLMs Act on Their Own Moral Judgment

    [https://arxiv.org/abs/2610.08670](https://arxiv.org/abs/2610.08670)

    该研究构建了涵盖五种压力类型的248个预注册场景面板，通过让模型同时以第一人称行动和第三人称评判的方式对照其自身道德判断，发现大语言模型在约五分之一的压力场景下会采取自己判定为错误的行为，且这种“言行不一”差距的大小取决于后训练配方。

    

    语言模型越来越多地充当智能体。一个明知某行为错误却仍然去做的智能体，与一个不知道更好选择的智能体，是两种不同的失败模式，而对模型陈述价值观的评估无法发现前者。我们构建了一个预注册的、涵盖五种压力类型的248个场景面板。每个场景以两种方式向同一模型提出：一次作为智能体选择要采取的行动，一次以第三人称询问哪个选项是正确的，从而以模型自身的判断作为参照。每个场景都有一个移除压力因素的孪生版本，并且每个模型都有一个正向对照——即其运营者下令执行违规行为的场景——以便区分“差距缺失”与“测量工具失灵”。在OLMo-3-7B-Instruct上，该模型在大约五分之一的压力场景中采取了它自己判定为错误的行为，且这一比例高于移除压力后的相同场景。在四个指令模型中，这一差距的大小取决于后训练配方。

    arXiv:2610.08670v1 Announce Type: cross  Abstract: Language models increasingly act as agents. An agent that says an action is wrong and then takes it anyway is a different failure from one that does not know better, and evaluations of stated values cannot see it. We build a pre-registered panel of 248 scenarios across five kinds of pressure. Each scenario is posed twice to the same model, once as the agent choosing what to do and once in the third person asking which option is right, so the model's own judgment is the reference. Every scenario has a twin with the pressure removed, and every model gets a positive control in which its operator orders the violating action, so that a missing gap can be told apart from a blind instrument. On OLMo-3-7B-Instruct, the model takes the action it judged wrong on about one in five pressuring scenarios, more often than on the same scenarios with the pressure removed. Across four instruct models the gap depends on the post-training recipe: OLMo-3 a
    
[^14]: 证据约束推理：胶质母细胞瘤放射基因组学中生物医学AI的神经-语义验证

    Evidence-Bound Reasoning: Neuro-Semantic Verification of Biomedical AI in Glioblastoma Radiogenomics

    [https://arxiv.org/abs/2610.08660](https://arxiv.org/abs/2610.08660)

    该论文提出了一种神经-语义验证框架，将放射组学测量转化为可寻址的证据记录和机器可校验的声明，从而可靠验证生物医学AI的解释中每条陈述是否真正有患者特异性证据支持。

    

    背景：生物医学AI可以生成看似合理的解释，却无法可靠地验证每条陈述是否都有患者特异性证据的支持。我们开发了一个神经-语义验证框架，将放射组学测量转换为可寻址的证据记录和机器可校验的声明。方法：在独立的Multicenter队列中，UPenn-GBM放射组学特征与基于标准化MRI和专家验证分割、由CaPTk重新提取的特征进行了对齐。共享空间包含来自T1、T1GD、T2和FLAIR MRI、覆盖三个肿瘤区域的1,728个特征。参考定义的语义状态由611例UPenn病例推导得出。我们评估了跨队列可迁移性、模型关联的溯源、确定性验证、受控的预测性能退化以及LLM声明提取试点；MGMT预测仅作为迁移压力测试。结果：语义状态一致性的中位数为0.786（加权kappa为0.709），范围……（原文摘要在此处截断）

    arXiv:2610.08660v1 Announce Type: new  Abstract: Background: Biomedical AI can generate plausible explanations without reliably verifying whether each statement is supported by patient-specific evidence. We developed a neuro-semantic verification framework that converts radiomic measurements into addressable evidence records and machine-checkable claims. Methods: UPenn-GBM radiomics were aligned with de novo CaPTk extraction from standardized MRI and expert-validated segmentations in an independent multicenter cohort. The shared space comprised 1,728 features from T1, T1GD, T2, and FLAIR MRI across three tumor regions. Reference-defined semantic states were derived from 611 UPenn cases. We evaluated cross-cohort transportability, model-linked provenance, deterministic verification, controlled predictive degradation, and an LLM claim-extraction pilot; MGMT prediction served only as a transport stress test. Results: Median semantic-state agreement was 0.786 (weighted kappa 0.709), rangin
    
[^15]: SquidAgent：明智并行，高效协调

    SquidAgent: Parallelize Wisely, Coordinate Efficiently

    [https://arxiv.org/abs/2610.08647](https://arxiv.org/abs/2610.08647)

    该论文提出SquidAgent，揭示了并行多智能体系统中“重新探索成本”与“对齐成本”这两项隐性开销，并据此推导出原则性决策准则：仅当并行化的关键路径成本加上这两项开销低于串行成本时，才应对该层进行并行化。

    

    基于大语言模型（LLM）的智能体能够解决复杂的多步骤任务，但顺序执行会带来显著的延迟。原则上，将工作并行分配给多个智能体应当产生接近线性的加速。然而，现有的并行多智能体系统往往比单智能体基线运行得更慢。我们将这一差距归因于并行执行会产生、而串行智能体可以避免的两项隐性成本。其一是重新探索成本：并行工作节点在重构编排器（orchestrator）已掌握的上下文（例如先前的决策）上所花费的冗余工作，而这些上下文在串行执行中本可被隐式继承。其二是对齐成本：为调和独立生成的输出之间的不一致性所需的额外开销。由此，我们推导出一个有原则的决策准则：只有当某一层的关键路径成本加上重新探索与对齐的开销低于相应的串行成本时，该层才应当被并行化。

    arXiv:2610.08647v1 Announce Type: new  Abstract: LLM-based agents solve complex multi-step tasks, but sequential execution incurs substantial latency. In principle, parallelizing work across multiple agents should yield near-linear speedups. Yet existing parallel multi-agent systems often run slower than a single-agent baseline. We attribute this gap to two hidden costs that parallel execution incurs but a serial agent avoids. First, there is a re-exploration cost: redundant effort spent by parallel workers reconstructing context that the orchestrator already possesses, such as prior decisions, that would otherwise be inherited implicitly in a serial execution. Second, there is an alignment cost: the overhead required to reconcile inconsistencies across independently generated outputs. We thus derive a principled decision criterion: a layer should be parallelized only when its critical-path cost, plus re-exploration and alignment overheads, is lower than the corresponding serial cost. 
    
[^16]: 面向大语言模型的参数内记忆增强

    Towards In-Parameter Memory Augmentation for Large Language Models

    [https://arxiv.org/abs/2610.08630](https://arxiv.org/abs/2610.08630)

    这是一篇综述，系统梳理了通过将可复用知识编码进模型参数、适配器等类参数对象并在推理时插入前向传播，从而在部署阶段以参数内记忆增强大语言模型的方法，并以参数放置等正交维度组织分类。

    

    近年来，大语言模型（LLM）和基于LLM的智能体日益需要融合预训练之后获取的知识，例如领域事实、用户偏好、文档以及交互经验。上下文学习（ICL）和基于ICL的智能体框架依然灵活，但它们会消耗上下文容量，并产生随上下文长度增长而增加的重复离散化编码开销。参数内记忆提供了一种互补的载体：可复用的记忆信息被表示在模型参数、适配器或其他类参数对象中，并在推理时组合进前向传播过程。本综述聚焦于在部署阶段用这类参数化记忆增强LLM的方法：无论该承载记忆的参数对象是在部署前还是在部署期间获得，它都会在推理时被插入到前向传播中。我们用两个正交的维度来组织该领域的全景：参数放置，包括嵌入……

    arXiv:2610.08630v1 Announce Type: new  Abstract: Recently Large Language Models (LLMs) and LLM-based agents increasingly need to incorporate knowledge acquired after pretraining, e.g., domain facts, user preferences, documents, and interaction experience. In-context learning (ICL) and ICL-based agent harness remain flexible, but they consume context capacity and incur repeated discretized encoding cost that grows with context length. \textbf{In-parameter memory} offers a complementary substrate: reusable memory information is represented in model parameters, adapters, or other parameter-like objects that are composed into the forward pass at inference time. This survey focuses on methods that augment LLMs with such parametric memory at deployment: a memory-bearing parameter object is plugged into the forward pass during inference, whether it is acquired before or during deployment. We organize the landscape with two orthogonal axes: \textbf{Parameter Placement}, which includes Embeddin
    
[^17]: InterCorrect：面向公平语音识别的交叉感知人口统计模型合并校正方法

    InterCorrect: Intersection-Aware Correction of Demographic Model Merging for Fair ASR

    [https://arxiv.org/abs/2610.08604](https://arxiv.org/abs/2610.08604)

    该论文提出InterCorrect方法，通过人口统计感知的模型合并与针对多群体交叉的校正向量提升语音识别公平性，将最佳整体词错误率从7.38%降至5.13%。

    

    自动语音识别（ASR）系统在不同人口统计群体之间的性能往往不均衡，而对于属于多个人口统计群体交叉的说话人，其识别错误尤其难以解决。本研究针对基于语音大语言模型（Speech-LLM）的公平ASR，研究了人口统计感知的模型合并方法。从一个基于SLAM-ASR的模型出发，我们仅在人口统计特定的子集上微调连接器，并将由此得到的各子群体适配连接器合并为一个全局模型。随后，我们利用子群体词错误率（WER）和任务向量冲突来识别关键的跨轴人口统计交叉群体，并向全局合并模型应用针对交叉群体的校正向量。在Fair-Speech数据集上的实验表明，全局人口统计合并相比基础模型降低了整体WER，而交叉校正为多种合并策略带来了额外收益。特别是，采用基于WER校正的TIES方法取得了最佳整体WER，将其从7.38%降至5.13%。

    arXiv:2610.08604v1 Announce Type: new  Abstract: Automatic Speech Recognition (ASR) systems often show uneven performance across demographic groups, and errors can be especially difficult to address for speakers belonging to multiple demographic groups. This work studies demographic-aware model merging for fair Speech-LLM-based ASR. Starting from a SLAM-ASR-based model, we fine-tune only the connector on demographic-specific subsets and merge the resulting subgroup-adapted connectors into a global model. We then identify critical cross-axis demographic pairs using subgroup WER and task-vector conflict, and apply intersection-specific correction vectors to the global merged model. Experiments on Fair-Speech show that global demographic merging improves overall WER over the base model, while intersection correction provides additional gains for several merging strategies. In particular, TIES with WER-based correction achieves the best overall WER, reducing it from 7.38\% to 5.13\%. Subgr
    
[^18]: 高风险紧急信息传递中的生成式人工智能翻译

    Generative AI translations in high-stakes emergency messaging

    [https://arxiv.org/abs/2610.08601](https://arxiv.org/abs/2610.08601)

    研究表明，在紧急信息翻译中使用特定语篇的提示词可显著提升生成式AI翻译的可理解性和可操作性，但人工修订在检测错误和伦理担责方面仍不可或缺。

    

    极端天气报告和地震指导等紧急信息传递可能涉及高风险，翻译错误甚至可能导致悲剧性后果。因此，使用机器翻译或生成式人工智能可能不被推荐。另一方面，初始翻译节省的时间可以将更多资源投入到修订和授权流程中，并覆盖更广泛的目标语言。一项将地震指导文本从英语翻译成中文和西班牙语的生成式人工智能翻译实验表明，使用特定语篇的提示词可以显著提高译文的可理解性和可操作性，尽管译者可能仍然不信任这些译文。人工修订仍然必不可少，这不仅是为了检测错误，还因为从伦理角度来看，需要有人为此类信息传递中的任何错误或延误承担责任。

    arXiv:2610.08601v1 Announce Type: new  Abstract: Emergency messaging such as extreme-weather reports and earthquake instructions can involve high stakes, to the extent that translation errors can lead to tragic consequences. The use of machine translation or generative artificial intelligence might therefore not be recommended. On the other hand, time savings in the initial translation can allow greater investments of resources in revision and authorization processes, as well as a wider range of target languages. An experiment with generative AI translations of an earthquake instruction text from English into Chinese and Spanish shows that use of discourse-specific prompts can considerably improve understandability and actionability, although the translations may still not be trusted by translators. Human revision is still required, not only to detect errors but also because of the ethical need for someone to take responsibility for any errors or delays in such messaging.
    
[^19]: 偶发信息污染患者病历并干扰大语言模型的临床推理

    Incidental information contaminates patient notes and disrupts clinical reasoning in large language models

    [https://arxiv.org/abs/2610.08585](https://arxiv.org/abs/2610.08585)

    研究发现闲聊和背景语音等偶发信息会污染大语言模型生成的病历并偶尔被错误地用于临床，据此提出LLM临床推理与分心的双重编码假说。

    

    大语言模型（LLM）日益被依赖用于支持环境式（ambient）医疗文档记录和临床推理。本研究通过评估模型对与患者就诊无关的偶发信息的敏感性，考察了这两种应用共有的失效模式的影响。在576段患者与临床医生的对话中，我们发现前沿模型将闲聊内容插入到了35%的病历中，而平均质量评分在五分制上最多变化0.20分。在3.7%的前沿模型生成的病历中，模型错误归因了这些题外话或在临床语境中使用了它们。在57次模拟录制问诊中，来自另一患者就诊的-10分贝背景语音泄漏到了48.2%的转录文本中，且在四个开源权重模型生成的下游病历中检测到5.3%的污染。我们提出了关于LLM临床推理与分心的双重编码假说，并有初步证据表明LLM中与偶发信息干扰相关的组件（摘要在此处被截断）。

    arXiv:2610.08585v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly relied upon to support ambient documentation and clinical reasoning. Here we examine the impact of a failure mode shared between these two applications by assessing their sensitivity to information incidental to the patient encounter. In 576 patient-clinician dialogues, we found that frontier models inserted small-talk exchanges into 35% of notes, while mean quality scores changed by at most 0.20 points on five-point scales. In 3.7% of frontier notes, models misattributed the asides or used them clinically. In 57 mock recorded consultations, background speech from a separate patient encounter at -10 dB leaked into 48.2% of transcripts, with contamination detected in 5.3% of downstream notes generated by four open-weight models. We propose a dual encoding hypothesis of clinical reasoning and distraction in LLMs, with preliminary evidence that LLM components associated with disruption by incide
    
[^20]: 我看得够了吗？冻结的视频-语言模型中编码了证据就绪度信号

    Have I Seen Enough? Frozen Video-Language Models Encode Evidence Readiness

    [https://arxiv.org/abs/2610.08560](https://arxiv.org/abs/2610.08560)

    该论文发现冻结的视频-语言模型内部已线性编码了一种由问题条件化、可跨基准泛化且与答案对错无关的“证据就绪度”信号，因此无需额外训练触发器即可判断流式视频问答中证据是否已充分到来。

    

    流式视频-语言模型不仅需要决定回答什么，还必须判断当前问题所需的证据是否已经到来。现有系统将该决策作为一个单独的触发器来学习；我们则探究一个未经修改的模型是否已经在计算这一决策。我们证明，冻结的视频大语言模型内部携带一种线性可读的证据就绪信号，该信号的标注来自带时间戳的证据而非模型输出。在一个共享的逐字节相同的评估中，该信号在全部七个模型中均可被解码（在最严格的“未就绪”采样下AUROC为0.733–0.905，而拟合的时钟模型接近随机水平），并且在完全未接触某基准家族视频数据的情况下训练的探针仍能读取该家族的数据。该信号是问题条件化的：在逐字节相同的视频窗口上，仅改变问题就能使66.1%的问题配对上的读出结果发生反转，而所有问题盲的控制组按构造均处于随机水平。模型即使给出错误答案，仍然编码了就绪状态：在错误答案中AUROC仍为0.722。

    arXiv:2610.08560v1 Announce Type: cross  Abstract: Streaming video-language models must decide not only what to answer, but whether the evidence needed for the current question has arrived. Existing systems learn that decision as a separate trigger; we ask whether an unmodified model already computes it. We show that frozen VideoLLMs carry a linearly readable evidence-readiness signal, labelled from timestamped evidence rather than from model output. It decodes in all seven models of a shared byte-identical evaluation (AUROC 0.733-0.905 under the strictest not-ready sampling, where a fitted clock is near chance), and a probe fitted without any of a benchmark family's footage still reads that family. It is question-conditioned: on byte-identical windows, changing only the question reverses the readout on 66.1% of pairs, while every question-blind control is at chance by construction. The model can answer incorrectly and still encode readiness: AUROC remains 0.722 among wrong answers. Re
    
[^21]: 大语言模型潜空间中的偏见方向捕获的是置信度，而非公平性

    Latent space bias directions in LLMs capture confidence, not fairness

    [https://arxiv.org/abs/2610.08559](https://arxiv.org/abs/2610.08559)

    该研究揭示了大语言模型激活引导中的去偏方向实际上编码的是模型置信度而非偏见信息，其去偏效果只是降低模型置信度的副产品，从而解释了激活引导去偏技术泛化能力差的根本原因。

    

    arXiv:2610.08559v1 公告类型： cross 摘要：激活引导（activation steering）作为一种轻量级的大语言模型推理时去偏技术日益流行。然而，先前的研究报告指出，引导向量的泛化能力较差，会对模型性能产生意外影响，并且向新数据集的迁移能力有限。我们的工作分析了用于激活引导的去偏方向实际上编码了什么，以揭示其性能不一致的原因。我们研究了通过对比反偏见提示和偏见提示的激活所获得的线性去偏方向，并将其作为一种引导干预措施，在偏见基准和通用知识基准上进行评估。我们发现，这个方向主要由模型置信度主导，在激活空间中从高概率token区域指向低概率token区域，而不是编码模型偏见的有意义表征。沿着该方向进行引导确实能减少测量到的偏见，但这是降低模型置信度的结果：在问答（QA）基准上，我们……

    arXiv:2610.08559v1 Announce Type: cross  Abstract: Activation steering has gained popularity as a lightweight inference-time debiasing technique for large language models. However, prior work reports that steering vectors generalise poorly, with unintended effects on model performance and limited transfer to new datasets. Our work analyses what the debiasing direction used for activation steering actually encodes, in order to shed light on its inconsistent performance. We study the linear debiasing direction obtained by contrasting the activations of anti-biased and biased prompts, and evaluate it as a steering intervention across bias and general knowledge benchmarks. We find that this direction is dominated by model confidence, pointing from regions of high to low-probability tokens in activation space rather than encoding a meaningful representation of model bias. Steering along it does reduce measured bias, but this is a consequence of reducing model confidence: on QA benchmarks we
    
[^22]: DeltaTTT：非线性循环记忆的逐层优化

    DeltaTTT: Layerwise Optimization for Nonlinear Recurrent Memory

    [https://arxiv.org/abs/2610.08553](https://arxiv.org/abs/2610.08553)

    针对非线性循环记忆在测试时训练中难以优化、并行基线反而优于串行版本的问题，DeltaTTT提出用逐层学习替代联合内循环优化，为每层分配局部预测目标并通过状态依赖的delta规则更新，从而缓解非线性记忆的优化困难。

    

    序列测试时训练通过一系列连续更新来适配记忆网络，每次更新都基于网络的先前状态计算内循环梯度。直观上，这种状态依赖性应使每次更新能够考虑到记忆已经学到的内容，并更好地融合新信息。然而，我们发现这一预期优势在非线性记忆中并未稳定实现：固定基底的并行TTT基线反而优于其串行对应版本。我们的探索性实验揭示了一个关键的潜在困难：在单次序列遍历过程中，非线性记忆可能比线性记忆更难优化。为缓解这一优化困难，我们提出DeltaTTT，它用逐层学习取代了两层记忆网络的联合内循环优化。每一层被分配一个局部预测目标，并通过状态依赖的delta规则进行更新。该公式化表述保留了非线性...

    arXiv:2610.08553v1 Announce Type: cross  Abstract: Sequential test-time training adapts a memory network through successive updates, each computing an inner-loop gradient based on the network's previous state. Intuitively, this state dependence should allow each update to account for what the memory has already learned and better incorporate new information. However, we find that this expected advantage does not consistently materialize in nonlinear memories: a fixed-base parallel TTT baseline outperforms its serial counterpart. Our exploratory experiments point to a key underlying difficulty: nonlinear memories can be harder to optimize than linear ones within a single pass over the sequence. To alleviate this optimization difficulty, we introduce DeltaTTT, which replaces joint inner-loop optimization of a two-layer memory network with layerwise learning. Each layer is assigned a local prediction target and updated through a state-dependent delta rule. This formulation retains a nonli
    
[^23]: 0.6 有多高？可解释性探测中的地板、天花板与余量

    How High Is 0.6? Floors, Ceilings, and Headroom in Interpretability Probing

    [https://arxiv.org/abs/2610.08544](https://arxiv.org/abs/2610.08544)

    该论文提出用“地板”（简单输入已能预测的水平）和“天花板”（完整输入能预测的水平）两个参照点以及二者之间的“余量”来校准可解释性探测得分，使探针分数具有明确、可比的含义，并证明余量在目标不依赖隐藏变量或输入不透露隐藏变量时消失。

    

    探测是可解释性研究的主力工具。如果模型的隐藏状态能够预测某个变量，就称该模型表征了这个变量。但探测得分并没有固定的含义。0.6 的 R² 可能仅仅反映了输入本身已经透露的信息，而相同的得分在不同的数据上可能意味着不同的东西。我们提出用两个参照点来解读每一个探测得分：地板，即一组声明的简单输入已经能预测的内容；以及天花板，即完整输入所能预测的内容。两者之间的差距，即余量，是探测能够展示模型计算出超出简单输入之外内容的范围。我们证明余量会在两种情况下消失：目标不再依赖于模型必须推断的隐藏变量，或者输入不再透露该变量。我们在为上下文内元分析训练的 transformer 上测试了这一方法，这些模型必须推断研究之间隐藏的异质性才能正确加权，而两个参照点在此都是可控制的。

    arXiv:2610.08544v1 Announce Type: cross  Abstract: Probes are the workhorse of interpretability. If a model's hidden states predict a variable, the model is said to represent it. But a probe score has no fixed meaning. An $R^2$ of 0.6 may only reflect what the input already gives away, and the same score can mean different things on different data. We propose reading every probe score against two reference points: a floor, what a declared set of simple inputs already predicts, and a ceiling, what the full input can predict. The gap between them, the headroom, is the range in which a probe can show that a model computes something beyond the simple inputs. We prove that headroom vanishes in two ways: the target stops depending on a hidden variable the model must infer, or the input stops revealing it. We test this on transformers trained for in-context meta-analysis, which must infer the hidden heterogeneity between studies to weight them correctly, and where both reference points are kn
    
[^24]: 迈向对齐缩放定律：一个框架与首批预注册测量

    Toward Alignment Scaling Laws: A Framework and First Preregistered Measurements

    [https://arxiv.org/abs/2610.08540](https://arxiv.org/abs/2610.08540)

    该论文提出将对齐视为一族可测量的幂律缩放关系（B_r(N)=a_rN^alpha_r）的框架及首批预注册测量，并证明长期对齐状态由经修正风险中的最大指数而非平均值决定，指数大于1时将累积不可持续的对齐债务。

    

    随着模型规模增长，对齐究竟变得更容易还是更难，这一争论往往基于孤立的发现，仿佛对齐是单一属性。我们将其视为一族可测量的缩放关系：对于每个风险类别 r，维持固定安全目标所需的对齐负担被建模为 B_r(N)=a_rN^alpha_r，其中 N 是能力代理指标；相对于与 N 成比例的预算，若 alpha_r<1，缩放有助于对齐；若 alpha_r≈1，缩放能保持同步；若 alpha_r>1，则会累积对齐债务。我们给出了负担的三种操作化定义，并区分了观测对齐、审计对齐与真实对齐。一个修正会消耗能力余量的玩具模型使这些后果变得明确。我们证明：决定长期状态的是经修正风险中的最大指数，而非平均值；当指数大于1时，任何要将余量维持在某一底线之上的策略都必须以超指数速度增长；对于作为幂律正混合的负担，在小模型上进行的拟合会低估大尺度上的（原文在此截断）。

    arXiv:2610.08540v1 Announce Type: new  Abstract: Whether alignment gets easier or harder as models grow is often argued from isolated findings, as if alignment were one property. We treat it as a family of measurable scaling relations: for each risk category r, the alignment burden needed to hold a fixed safety target is modeled as B_r(N)=a_rN^alpha_r, with N a capability proxy; against a budget proportional to N, scaling helps if alpha_r<1, keeps pace if alpha_r~1, and accumulates alignment debt if alpha_r>1. We give three operationalizations of burden and distinguish observed, audited and true alignment. A toy model, in which corrections consume capability headroom, makes the consequences explicit. We prove that the largest exponent among corrected risks, not an average, sets the long-run regime; that above 1 any policy holding headroom above a floor must grow super-exponentially; that, for burdens that are positive mixtures of power laws, fits on small models underestimate large-sca
    
[^25]: Wiki-Talkie：基于真实世界讨论的角色化智能体多语言基准测试

    Wiki-Talkie: Multilingual Benchmarking of Persona-Based Agents on Real-World Discussions

    [https://arxiv.org/abs/2610.08513](https://arxiv.org/abs/2610.08513)

    该论文提出了 Wiki-Talkie——首个基于维基百科讨论页真实对话、涵盖五种语言并配以源自真实用户社区画像的多语言基准数据集，用于评估角色化LLM智能体模拟人类交互的行为保真度。

    

    大型语言模型（LLM）越来越多地作为自主智能体被部署在社交环境中，这使得研究它们忠实模拟人类交互的能力变得至关重要。其中的核心在于将智能体锚定在真实的用户画像上，然而现有数据集依赖于虚构的人物画像，且仅涵盖少数几种语言，缺乏评估跨不同人群行为保真度所需的经验基础。我们提出了 Wiki-Talkie，这是一个多语言数据集，包含来自维基百科讨论页的真实对话，涵盖分属两个语系的五种语言：日耳曼语系（德语、英语）和罗曼语系（西班牙语、法语、意大利语），并配以从真实用户社区中提取的人物画像，这些画像包含社会人口属性、自我描述以及基于实际行为的交互特征。利用 Wiki-Talkie，我们在下一轮回复生成任务上，针对多种人物画像条件化策略对智能体的交互行为进行了评估。

    arXiv:2610.08513v1 Announce Type: cross  Abstract: LLMs are increasingly deployed as autonomous agents in social environments, making it critical to study their ability to faithfully simulate human interactions. Central to this is grounding agents in realistic user personas, yet existing datasets rely on fictional personas and are limited to a handful of languages, lacking the empirical grounding necessary to evaluate behavioral fidelity across diverse populations. We introduce Wiki-Talkie, a multilingual dataset of real-world conversations from Wikipedia Talk pages across five languages spanning two language families: Germanic (German, English) and Romance (Spanish, French, Italian), paired with personas derived from real user communities and encompassing sociodemographic attributes, self-descriptions, and behaviorally grounded interaction traits. Using Wiki-Talkie, we evaluate agent interactional behavior on a next-turn generation task across various persona conditioning strategies. 
    
[^26]: 语言模型对抑郁症的评分更多反映的是评分者而非患者

    Language-model ratings of depression reflect the rater more than the patient

    [https://arxiv.org/abs/2610.08501](https://arxiv.org/abs/2610.08501)

    该研究通过对880个语言模型评分者的预注册实验发现，语言模型对抑郁症的评分更多反映评分模型自身的差异（解释30.0%的评分方差）而非患者的真实症状差异（仅10.5%），即使两个高精度模型平均也会对40%的参与者的筛查结果产生分歧。

    

    抑郁症没有可用于诊断的血液检测。语言模型有望提供不知疲倦、一致的评估，但准确的评分者之间是否会在对个体的判断上产生分歧？我们预先注册了880个语言模型评分者，将11个开源模型与提示词及评分方式的选择进行交叉组合，并将其应用于189次访谈，以八项患者健康问卷（PHQ-8）作为参照。模型选择解释了症状总评分方差的30.0%，而参与者的稳定个体差异仅解释10.5%。随机抽取的两个受试者工作特征曲线下面积（AUC）≥0.70的评分者，平均对40%的参与者的筛查决策不一致。平均而言，评分偏高程度决定了有多少人被标记为阳性，但能力相当的评分者对大约五分之一的参与者做出了不同的选择。对86次新访谈进行的锁定分析再现了主要的预注册研究结果。利用40名有标签参与者进行的探索性重新校准将准确率从约60%提高到75%，并将评分者间的分歧减半。

    arXiv:2610.08501v1 Announce Type: cross  Abstract: Depression has no diagnostic blood test. Language models promise tireless, consistent assessment, but can accurate raters disagree about individuals? We pre-registered 880 language-model raters, crossing 11 open models with prompting and scoring choices, and applied them to 189 interviews against the eight-item Patient Health Questionnaire. Model choice explained 30.0% of summed-symptom score variance, stable participant differences 10.5%. Two randomly drawn raters with area under the receiver operating characteristic curve (AUC) >= 0.70 disagreed on screening decisions for 40% of participants, on average. Average over-rating governed how many were flagged, yet equal-capacity raters chose differently for about one participant in five. A locked analysis of 86 new interviews reproduced the main pre-registered findings. Exploratory recalibration with 40 labelled participants raised accuracy from about 60% to 75% and halved disagreement, l
    
[^27]: UNREAL：用单一模型统一检索与长上下文

    UNREAL: Unifying Retrieval and Long-Context with a Single Model

    [https://arxiv.org/abs/2610.08463](https://arxiv.org/abs/2610.08463)

    UNREAL提出了一种模型原生的证据选择框架，直接从冻结LLM的内部表示中推导检索查询，以不到50万可训练参数统一了语料库检索与长上下文推理，并在多个基准上大幅超越最先进的检索-重排序系统。

    

    长上下文推理和检索增强生成（RAG）在截然不同的尺度上处理证据选择问题，范围从单个长提示词到整个语料库。我们探究是否存在一种单一的模型内部机制能够覆盖这一范围并完成证据选择。我们提出了UNREAL（用单一模型统一检索与长上下文），一个模型原生的证据选择框架，可同时覆盖语料库检索和长上下文推理。UNREAL对文本块进行编码，并直接从冻结的大语言模型（LLM）的内部表示中推导检索查询。它仅增加不到50万个可训练参数，且完全不改动骨干模型。在一个包含30亿标记、2100万文本块的维基百科索引上，全部四种稠密型和混合型UNREAL骨干模型均超越了最先进的检索器-重排序器系统。其中最佳模型将HotpotQA上的召回率从49.1%提升至73.2%，将2WikiMultiHopQA上的召回率从31.7%提升至60.1%，将MuSiQue上的召回率从8.8%提升至14.4%。当应用于长上下文任务时，同样的选择机制……（摘要到此截断）

    arXiv:2610.08463v1 Announce Type: new  Abstract: Long-context inference and Retrieval-Augmented Generation (RAG) handle evidence selection at vastly different scales, from a single long prompt to an entire corpus. We ask whether a single model-internal mechanism can select evidence across this range. We introduce UNifying REtrieval And Long-Context with a Single Model (UNREAL), a model-native evidence selection framework to span corpus retrieval and long-context inference. UNREAL encodes chunks and derives retrieval queries directly from the frozen LLM's internal representations. It adds fewer than 500K trainable parameters and leaves the backbone unchanged. On a 3B-token, 21M-chunk Wikipedia index, all four dense and hybrid UNREAL backbones outperform state-of-the-art retriever-reranker systems. The best model raises recall from 49.1% to 73.2% on HotpotQA, from 31.7% to 60.1% on 2WikiMultiHopQA, and from 8.8% to 14.4% on MuSiQue. Applied to long-context tasks, the same selection mecha
    
[^28]: Agentic AutoRAG：通过推理驱动的智能体进行RAG流程优化

    Agentic AutoRAG: RAG Pipeline Optimization through Reasoning-Driven Agents

    [https://arxiv.org/abs/2610.08452](https://arxiv.org/abs/2610.08452)

    该论文提出Agentic AutoRAG，一种利用LLM智能体进行多目标RAG超参数优化的方法，其核心创新在于通过诊断器将每次失败归因于检索或生成阶段，从而让优化器能够推理配置失败的原因并智能地指导后续搜索。

    

    检索增强生成（RAG）是一种被广泛使用的方法，用于将大语言模型（LLM）扎根于外部知识。然而，配置一个RAG流程是一个代价高昂的超参数优化问题，涉及众多相互关联的选择，从分块和嵌入模型到重排序和生成。现有的优化器，从贪心搜索到贝叶斯优化，都将每次试验简化为一个汇总分数进行搜索，而不会建模某个配置为何会有那样的表现，尽管检索到的文本块其实已经提供了关于每次失败究竟发生在检索阶段还是检索之后的证据。我们提出了Agentic AutoRAG，一个用于多目标RAG超参数优化的LLM智能体优化器，具备检索与生成之间的失败归因能力。它提出候选配置，并在从语料库构建的固定考题上进行评分：每次试验后，一个诊断器会将每个失败的问题归因于检索阶段或生成阶段，而一个提议器则基于……（原文摘要在此处被截断，后续内容未能提供）

    arXiv:2610.08452v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) is a widely used approach for grounding large language models (LLMs) in external knowledge. However, configuring a pipeline is an expensive hyperparameter optimization problem over many interacting choices, from chunking and embedding model to reranking and generation. Existing optimizers, from greedy search to Bayesian optimization, reduce each trial to an aggregate score and search without modeling why a configuration performed as it did, even though the retrieved chunks already provide evidence about whether each failure occurred during retrieval or after it. We introduce Agentic AutoRAG, an LLM-agent optimizer for multi-objective RAG hyperparameter optimization with retrieval-versus-generation failure attribution. It proposes configurations scored on a frozen exam from the corpus: after each trial a Diagnoser attributes each failed question to retrieval or generation, and a Proposer, grounded in a
    
[^29]: 重新思考跨分词器在线策略蒸馏：从对齐覆盖到监督可靠性

    Rethinking Cross-Tokenizer On-Policy Distillation: From Alignment Coverage to Supervision Reliability

    [https://arxiv.org/abs/2610.08448](https://arxiv.org/abs/2610.08448)

    该研究发现跨分词器在线策略蒸馏中扩大对齐覆盖并无必要——严格1:1对齐已覆盖大部分token，仅用共享词表中由学生选择的top-16子集计算反向KL即可媲美完整共享词表方法，表明监督可靠性比对齐覆盖更为关键。

    

    在线策略蒸馏（OPD）利用教师模型的反馈，在学生模型自身生成的文本上进行训练。当教师与学生使用不同的分词器时，比较二者的预测需要在序列层面和词表层面进行对齐。本文研究了扩大这种对齐覆盖范围是否能够改善学习效果。在数学推理和代码生成任务上，针对三个异构的教师-学生模型对的实验表明：尽管词表存在显著差异，严格的1:1对齐组已经覆盖了学生生成的大多数token；在蒸馏前从学生模型采样的回复上，共享词表在严格对齐位置上平均保留了几乎全部的教师与学生概率质量。将反向KL散度限制在每个严格对齐位置上由学生选择的共享词表top-16子集，即可获得与完整共享词表OPD相当的准确率，并优于所评估的跨分词器基线方法。此外，添加均方误差监督……

    arXiv:2610.08448v1 Announce Type: cross  Abstract: On-Policy Distillation (OPD) trains a student on its own generations using teacher feedback. With different tokenizers, comparing teacher and student predictions requires alignment at both sequence and vocabulary levels. In this paper, we examine whether expanding this alignment coverage improves learning. Across three heterogeneous teacher--student pairs on mathematical reasoning and code generation, strict 1:1 groups already cover most student-generated tokens despite substantial vocabulary mismatch. On responses sampled from the students before distillation, the shared vocabulary retains nearly all teacher and student probability mass at strictly aligned positions on average. Restricting reverse KL to a student-selected top-16 subset of the shared vocabulary at each strict position achieves accuracy comparable to full shared-vocabulary OPD, outperforming the evaluated cross-tokenizer baselines. Adding mean squared error supervision 
    
[^30]: 知道何时不应回答：潜在欠规范信号的跨域与多轮泛化

    Knowing When Not to Answer: Cross-Domain and Multi-Turn Generalization of Latent Underspecification Signals

    [https://arxiv.org/abs/2610.08413](https://arxiv.org/abs/2610.08413)

    该论文构建了一个带轮次标签的多轮对话不可回答性基准与模拟用户评估框架，发现线性探针所捕捉的“信息缺失”信号能在共享同一不可回答性根源的数据集间稳健跨域迁移（AUROC 0.77–0.97），但不同类型不可回答性的表征边界会受词汇混淆、网络层级与坐标系选择的影响。

    

    大型语言模型经常回答那些无法根据已有信息回答的问题，并且在对话中往往在信息尚不充分时就贸然作答。已有研究表明，“不可回答性”可以从模型隐藏状态中被线性解码出来，但目前尚不清楚哪些形式的不可回答性共享同一表征，以及该信号在对话场景中是否实用。本文贡献了一个带轮次标签的多轮基准数据集（423段对话、1,661个带标签的轮次状态），以及一个配有可回答澄清性问题的模拟用户的评估框架，并结合六个数据集与六个开源权重的大语言模型，系统检验不可回答性探针的泛化边界。结果显示，在共享同类不可回答性根源的数据集之间，探针能够稳健迁移：数学中的信息缺失（AUROC 0.77–0.97），以及文本阅读中的信息缺失（SQuAD 2.0<->MuSiQue，0.77–0.90）。相比之下，针对认识论意义上“已知未知”（known-unknowns）的探针向数学任务迁移效果较差，但这种分离在引入词汇控制后会减弱，并随网络层级和坐标系的选择而变化，因此……（原文摘要在此处截断）

    arXiv:2610.08413v1 Announce Type: cross  Abstract: Large language models routinely answer questions that cannot be answered from the information given, and in dialogue they answer before enough has been said. Unanswerability is linearly decodable from hidden states, but it is unclear which of its forms share a representation and whether the signal is useful in dialogue. We contribute a turn-labeled multi-turn benchmark (423 conversations, 1,661 labeled turn-states) and an evaluation harness with a simulated user who answers clarifying questions, and use them with six datasets and six open-weight LLMs to test how far probes for unanswerability carry. Probes transfer robustly between datasets that share a ground of unanswerability: missing information in math (AUROC 0.77-0.97) and in a passage (SQuAD 2.0<->MuSiQue, 0.77-0.90). Probes for epistemic "known-unknowns" transfer poorly to math, but this separation weakens under lexical controls and changes with layer and coordinate system, so 
    
[^31]: 图上远见：面向知识库问答的超越局部视野推理

    Foresight-over-Graph: Reasoning Beyond Local Horizons for Knowledge Base Question Answering

    [https://arxiv.org/abs/2610.08388](https://arxiv.org/abs/2610.08388)

    提出前瞻感知的证据检索框架FoG，克服LLM图推理中逐跳贪心与束搜索剪枝的短视问题，避免关键证据分支被过早丢弃，提升知识库问答的可靠性。

    

    大语言模型（LLM）在问答任务中已展现出强大能力，但在知识密集型任务上仍频繁出现幻觉问题。知识图谱（KG）为LLM提供了结构化、可解释且可更新的事实依据，使其成为实现可靠推理的极具前景的外部知识来源。然而，现有的LLM引导图推理方法在证据检索过程中通常依赖逐跳贪心或束搜索式的剪枝策略。这类局部决策过程本质上具有短视性：在源头附近看似微弱的证据，只有在探索更深的图上下文后才可能变得至关重要，这导致对回答至关重要的分支被过早丢弃，且推理链难以恢复。为解决这一局限，我们提出了Foresight-over-Graph（FoG），一种面向知识库问答（KBQA）的前瞻感知证据检索框架。FoG迭代地构建一个……

    arXiv:2610.08388v1 Announce Type: cross  Abstract: Large language models (LLMs) have demonstrated strong capabilities in question answering, yet they still frequently suffer from hallucinations on knowledge-intensive tasks. Knowledge graphs (KGs) provide LLMs with structured, interpretable, and updatable factual grounding, making them a promising external knowledge source for reliable reasoning. However, existing LLM-guided graph reasoning methods typically rely on hop-wise greedy or beam-style pruning during evidence retrieval. Such local decision processes are inherently myopic: evidence that appears weak near the source may become crucial only after deeper graph context is explored, causing answer-critical branches to be discarded prematurely and making the reasoning chain difficult to recover. To address this limitation, we propose Foresight-over-Graph (FoG), a foresight-aware evidence retrieval framework for knowledge base question answering (KBQA). FoG iteratively constructs a qu
    
[^32]: CoDe-LoRA：通过知识巩固与解耦缓解大语言模型持续学习中的正交困境

    CoDe-LoRA: Mitigating the Orthogonality Dilemma in Continual Learning of LLMs via Knowledge Consolidation and Decoupling

    [https://arxiv.org/abs/2610.08312](https://arxiv.org/abs/2610.08312)

    提出无需回放的CoDe-LoRA方法，通过自适应零空间投影和语义路由将学习过程解耦为通用知识巩固与任务特定知识解耦，克服了正交参数隔离阻碍跨任务知识迁移的“正交困境”。

    

    持续学习（CL）对于大语言模型（LLMs）顺序适应不断演变的任务至关重要。为了缓解灾难性遗忘，近期的进展采用带正交投影的低秩适应方法（如O-LoRA）来隔离任务参数。然而，我们揭示了这种严格的几何约束会引发“正交困境”：僵化的参数隔离阻碍了语义相关任务之间共享表示的迁移与积累。在这项工作中，我们提出了一种新的无需回放的方法，称为巩固与解耦LoRA（CoDe-LoRA），用于大语言模型的持续学习。CoDe-LoRA将学习过程解耦为巩固通用知识和解耦任务特定知识两个部分。为实现这一目标，CoDe-LoRA利用自适应零空间投影机制和语义路由来平衡知识积累与任务特定适应。在四个骨干模型和三种持续学习设置上的实验结果……

    arXiv:2610.08312v1 Announce Type: new  Abstract: Continual learning (CL) is essential for Large Language Models (LLMs) to sequentially adapt to evolving tasks. To mitigate catastrophic forgetting, recent advances implement low-rank adaptation with orthogonal projections (e.g., O-LoRA) to isolate task parameters. However, we reveal that such strict geometric constraints trigger an "Orthogonality Dilemma": rigid parameter isolation impedes the transfer and accumulation of shared representations across semantically related tasks. In this work, we propose a new replay-free method, called Consolidation and Decoupling LoRA (CoDe-LoRA), for CL of LLMs. CoDe-LoRA disentangles the learning process into Consolidating Universal Knowledge and Decoupling Task-Specific Knowledge. To achieve this, CoDe-LoRA leverages an adaptive null space projection mechanism and semantic routing to balance knowledge accumulation with task-specific adaptation. Experimental results across four backbones and three CL 
    
[^33]: 语言不可对齐性：为什么某些概念抗拒跨文化基准评估

    Language Unalignability: Why Some Concepts Resist Cross-Cultural Benchmark Evaluation

    [https://arxiv.org/abs/2610.08303](https://arxiv.org/abs/2610.08303)

    本文提出“语言不可对齐性”概念，论证对于语用标记、敬语等一类概念，跨语言映射在原理上无法同时保持词汇忠实性与结构忠实性，从根本上质疑了多语言大模型评估所依赖的翻译同构假设。

    

    当前对多语言大语言模型（LLM）的评估建立在一个隐含的“翻译同构假设”（TIA）之上：即不同语言之间的语义结构是同构的，可以在不损失信息的情况下相互映射。我们认为，这一假设不仅在实践中被违反，而且在原理上对于一类类型学上可识别的概念（包括语用标记、敬语以及历时分层词汇）来说是不适定的。我们使用“用法云”框架将这一失败形式化，将概念表示为上下文嵌入的点集。我们定义α-不可对齐性为：任何映射都不可能同时保持词汇忠实性（质心对应）和结构忠实性（局部邻域拓扑）。我们提供了三个层面的证据。在行为层面，我们表明FLORES-200的翻译失败可以通过语系和资源类别来预测，但不能通过文字系统来预测。

    arXiv:2610.08303v1 Announce Type: new  Abstract: Current evaluation of multilingual Large Language Models (LLMs) rests on an implicit Translation-Isomorphism Assumption (TIA): that semantic structures across languages are congruent and mutually mappable without loss of information. We argue that this assumption is not merely violated in practice, but ill-posed in principle for a typologically identifiable class of concepts, including pragmatic markers, honorifics, and diachronically stratified terms. We formalize this failure using a usage-cloud framework, representing concepts as point sets of contextualized embeddings. We define $\alpha$-unalignability as the impossibility of any mapping that simultaneously preserves lexical faithfulness (centroid correspondence) and structural faithfulness (local neighborhood topology). We provide three layers of evidence. Behaviorally, we show that FLORES-200 translation failures are predicted by language family and resource class but not by script
    
[^34]: 记忆深度与重构上下文宽度：分层检索的受控评估

    Memory Depth and Reconstructed Context Width: A Controlled Evaluation of Hierarchical Retrieval

    [https://arxiv.org/abs/2610.08300](https://arxiv.org/abs/2610.08300)

    该研究通过受控实验发现，在分层对话记忆检索中，扩大重构上下文宽度（从1K到4K）可使准确率显著提升10.11-17.98个百分点，而增加层次深度并无单调收益，且超过8-16K后性能趋于平台期，表明应优先采用大型连贯的上下文块而非逐级分层结构。

    

    长期对话记忆正成为现代大语言模型系统不可或缺的组成部分。已有提出的架构按主题和事件对记录进行分组，构建层次结构与图结构，并通过因果和时间关系连接事实。我们通过实验研究两个记忆参数之间的相互作用：结构深度与提供给回答模型的上下文宽度。利用EverMemBench基准，我们评估了D1-D4四个深度级别、1,024/2,048/4,096 tokens的核心预算，以及额外的Production和Oracle两种条件，后者可扩展至完整档案。将宽度从1K增加到4K可使准确率提升10.11-17.98个百分点，而增加深度并未带来单调递增的收益。超过8-16K后，Production条件下的性能达到平台期，而每个正确答案消耗的token数持续增加；Oracle条件在68-71K tokens的完整档案上仍能保持质量。这些结果促使人们进一步研究采用大型、连贯的上下文块，而非逐级[处理]……

    arXiv:2610.08300v1 Announce Type: new  Abstract: Long-term conversational memory is becoming an integral component of modern LLM systems. Proposed architectures group records by topics and events, construct hierarchies and graphs, and connect facts through causal and temporal relations. We experimentally study the interaction between two memory parameters: structural depth and the width of context supplied to the answer model. Using EverMemBench, we evaluate depths D1-D4, core budgets of 1,024/2,048/4,096 tokens, and additional Production and Oracle conditions up to the full archive. Increasing width from 1K to 4K improves Accuracy by 10.11-17.98 percentage points, whereas increasing depth provides no monotonic gain. Beyond 8-16K, Production performance reaches a plateau while tokens per correct answer continue to increase; Oracle preserves quality on full archives of 68-71K tokens. These results motivate further investigation of large, coherent context blocks instead of progressively 
    
[^35]: STRUCTURALCOST：一个用于建模人类句子处理难度的受控阅读时间数据集

    STRUCTURALCOST: A controlled reading time dataset for modeling human sentence processing difficulty

    [https://arxiv.org/abs/2610.08208](https://arxiv.org/abs/2610.08208)

    该研究推出大规模阅读时间数据集STRUCTURALCOST，验证了人类阅读时间随主谓依存长度增加而上升，并揭示现有语言模型虽能反映这种预测性难度但低估了工作记忆导致的整合成本，为评估语言模型的认知合理性奠定了数据基础。

    

    我们推出了STRUCTURALCOST，这是一个包含475名参与者和40,800个观测数据的自定步速阅读数据集，专门用于分离长距离主谓依存消解的处理成本。我们在NLP规模上复现了一个此前统计效力较低的心理语言学发现，即人类在主要动词处的阅读时间随依存长度增加而上升，且这一现象由超越线性距离的句法嵌套所驱动。不同的语言模型——涵盖n-gram模型、状态空间模型（SSM）和transformer——能部分反映这种分级难度模式，但低估了人类产生的整合成本，且这一差距在不同架构和模型规模下持续存在。这表明这些模型捕捉到了人类处理过程中的预测成分，但未能捕捉工作记忆所带来的全部整合成本。STRUCTURALCOST提供了推动语言模型认知合理性评估所需的数据。

    arXiv:2610.08208v1 Announce Type: cross  Abstract: We introduce STRUCTURALCOST, a self-paced reading dataset of 475 participants and 40,800 observations isolating the processing cost of long-distance subject-verb dependency resolution. We replicate a low-powered psycholinguistic finding at NLP scale, namely that human reading times at the main verb increase with dependency length, driven by syntactic embedding beyond linear distance. Different language models -- spanning n-gram models, SSMs, and transformers -- partially mirror this graded difficulty profile, yet underestimate the integration cost humans incur, with a gap that persists across architectures and model sizes. This suggests these models capture the predictive component of human processing but not the full integration cost that working memory imposes. STRUCTURALCOST provides data needed to drive progress toward evaluating the cognitive plausibility of language models.
    
[^36]: 先对齐，再校正：面向极端量化大语言模型的无训练两阶段低秩补偿

    Align, Then Correct: Training-Free Two-Stage Low-Rank Compensation for Extremely Quantized Large Language Models

    [https://arxiv.org/abs/2610.08164](https://arxiv.org/abs/2610.08164)

    该论文提出一种无需训练的两阶段闭式低秩补偿框架，先对齐层输出再校正残余误差，克服了现有低秩量化误差补偿在对称校准和仅二阶优化上的两大局限，大幅提升极端量化大语言模型的精度恢复能力。

    

    低秩量化误差补偿（LQEC）通过在每个冻结的量化权重旁附加一个闭式解的秩-r适配器，无需任何训练即可恢复极端权重量化下损失的精度。我们证明，现有的补偿器受限于两个共同的简化假设：其一，它们采用对称校准，即在相同的激活上评估全精度权重与补偿后的权重，这使补偿目标本身呈高秩特性，导致固定的秩预算只能捕获其中很小一部分；其二，它们仅最小化损失的二阶项，而补偿后的模型并非处于平稳点——每一层中仍残留一个比所施加补偿本身更大的一阶下降方向，且任何重构目标都无法将其吸收。我们提出了一个消除上述两个简化的两阶段闭式框架：第一阶段在Fisher加权（下）将每层的输出与全精度模型对齐……

    arXiv:2610.08164v1 Announce Type: cross  Abstract: Low-rank quantization error compensation (LQEC) recovers the accuracy lost under aggressive weight quantization by attaching a closed-form rank-$r$ adapter beside each frozen quantized weight, without any training. We show that existing compensators are limited by two shared simplifications. They calibrate symmetrically, evaluating the full-precision and compensated weights on the same activation, which yields a compensation target that is inherently high-rank -- so a fixed rank budget captures only a small fraction of it. And they minimize only the second-order term of the loss, although the compensated model is not stationary: a first-order descent direction larger than the applied compensation itself remains in every layer, and no reconstruction objective can absorb it. We propose a two-stage closed-form framework that removes both simplifications. Stage 1 aligns each layer's output with the full-precision model under a Fisher-weigh
    
[^37]: 失败在于读取方式：细粒度情绪识别基准测量的是答案引出方式，而非感知能力

    The Failure Is in the Readout: Fine-Grained Emotion Recognition Benchmarks Measure Elicitation, Not Perception

    [https://arxiv.org/abs/2610.08162](https://arxiv.org/abs/2610.08162)

    该研究发现，现成的视觉语言模型在细粒度情绪识别上其实并不逊于专门微调的模型，此前基准测得的失败源于生成式的答案引出方式，而非模型的感知能力不足。

    

    细粒度情绪识别可以支撑治疗工具和社交机器人，但它需要面部数据，这带来了隐私和数据保护方面的顾虑。EmoNet-Face-HQ 通过生成的人像解决了这一问题，并由专家按照远比常见的六到八种基本情绪更精细的 40 类分类体系进行评分。在其自带的评测协议下，视觉语言模型（VLM）在该分类体系上得分很差，该基准据此得出结论：需要一个专门微调的模型，即 Empathic-Insight-Face（EIF；Small/Large）。我们证明，当答案不是通过文本生成、而是作为每个类别一次二值查询直接从 logits 中读取时，现成的 VLM 就能追平甚至超越该微调模型。我们保留了该基准的图像、分类体系和专家评分，只改变了答案的读取方式。在其测量最可靠的五个类别上，专家间一致性达到 κ_w = 0.468。在生成式评测下，十一个开源权重 VLM 中没有任何一个的置信区间完全高于该锚点（……摘要原文在此截断）

    arXiv:2610.08162v1 Announce Type: cross  Abstract: Fine-grained emotion recognition supports therapy tools and social robots, but it needs facial data, which raises privacy and data-protection concerns. EmoNet-Face-HQ answers that with generated portraits, expert-rated over a $40$-category taxonomy far finer than the usual six to eight basic emotions. Under the protocol it ships with, vision-language models (VLMs) score poorly on that taxonomy, and the benchmark concludes that a dedicated fine-tuned model is necessary: Empathic-Insight-Face (EIF; Small/Large). We show that off-the-shelf VLMs match or beat that fine-tuned model when the answer is not generated but read from the logits, as one binary query per category. We keep the benchmark's images, taxonomy and ratings, and change only how the answer is read. Experts agree at $\kappa_w = 0.468$ on the five categories they measure most reliably. Generatively, no interval among eleven open-weight VLMs lies entirely above that anchor ($\
    
[^38]: 文本生成交响曲：临床病历生成基准测试

    Symphony for Text Generation: Benchmarking Clinical Note Generation

    [https://arxiv.org/abs/2610.08161](https://arxiv.org/abs/2610.08161)

    该论文提出了包含300例多语言临床就诊记录的MedConv数据集，并构建了结合蕴含指标与大语言模型评判的受控临床评估框架，证明临床AI平台Corti的病历生成质量与领先商业环境式记录软件相当或更优，且其可配置API可针对特定文档需求灵活优化质量维度。

    

    环境式文档系统正迅速获得广泛应用，但其对临床病历质量的影响仍缺乏充分表征。我们推出了MedConv——一个包含300例临床就诊记录的多语言数据集，涵盖英语、丹麦语和德语，并将其与环境临床智能基准（ACI-BENCH）结合使用，将Corti（一个临床AI平台）与两个基于通用AI构建的领先且易用的环境式记录软件进行比较。我们提出了一个受控临床评估框架，该框架将文本蕴含指标与大语言模型评判的成对比较相结合，涵盖从PDSQI-9采纳的八个维度。结果表明，Corti基于API的文本生成基础设施与领先的商业记录软件相当或更优。我们进一步表明，Corti的可配置API提供了必要的灵活性，能够针对特定的文档用例对质量维度进行微调。我们呈现了该评估方法并发布了数据集。

    arXiv:2610.08161v1 Announce Type: cross  Abstract: Ambient documentation systems are rapidly gaining adoption, yet their impact on clinical note quality remains poorly characterized. We introduce MedConv, a multilingual dataset of 300 clinical encounters in English, Danish, and German, and use it alongside the Ambient Clinical Intelligence benchmark (ACI-BENCH) to compare Corti, a clinical AI platform, with two leading, accessible ambient scribe software applications built on general-purpose AI. We present a controlled clinical evaluation framework that combines entailment metrics with LLM-judged pairwise comparisons across eight dimensions adopted from PDSQI-9. Results show that Corti's API-based text-generation infrastructure is on par with or outperforms leading commercial scribes. We further show that Corti's configurable API provides the flexibility necessary to fine-tune quality dimensions for specific documentation use cases. We present the evaluation methodology and release a d
    
[^39]: 让 COMET 跨文字系统可比：诊断与修正印度语系机器翻译评估中由分词器引发的文字偏差

    Making COMET Comparable Across Scripts: Diagnosis and Correction of Tokeniser-Induced Script Bias in Indic MT Evaluation

    [https://arxiv.org/abs/2610.08159](https://arxiv.org/abs/2610.08159)

    该论文发现 COMET 评估指标因分词器存在文字偏差，导致印度语系不同文字系统间的分数不可比且排序准确性下降，并提出 COMET-QN 方法以精确消除跨文字分数范围不兼容的问题。

    

    COMET 将翻译质量报告为单一数值，而这一数值通常会在以不同文字系统书写的目标语言之间进行比较。这种比较隐含着“文字不变性”假设：分数不应依赖于承载目标文本的书写系统。我们在 IndicMT Eval 数据集上对该假设进行了检验，方法是将目标文本重新编码为拉丁字母，在保持内容和人工评分不变的情况下改变其正字法形式。结果显示，文字身份（script identity）解释了本族文字 COMET 方差的 22.9%，并且在所研究的全部五种语言中，指标与标注者的一致性均有所下降。我们将这一效应追溯到分词器，并用三种无需标签的诊断方法对其进行测量。这种偏差实为两个缺陷，而非一个：来自不同文字系统的分数占据互不兼容的数值范围，且在同一文字系统内部，该指标对翻译进行排序的准确性也更低。任何保序的分数变换都无法修复第二个缺陷。第一个缺陷则可由 COMET-QN 精确消除，该方法对分数分布进行映射……

    arXiv:2610.08159v1 Announce Type: new  Abstract: COMET reports translation quality as a single number, and that number is routinely compared across target languages written in different scripts. Such a comparison assumes Script Invariance: the score should not depend on the writing system that carries the target. We test it on IndicMT Eval by re-encoding the target into Latin script, which changes orthographic form while holding content and human ratings fixed. Script identity then accounts for 22.9% of native-script COMET variance, and agreement with annotators falls in all five languages studied. We trace the effect to the tokeniser and measure it with three label-free diagnostics. The bias is two faults, not one. Scores from different scripts occupy incompatible ranges, and within a single script the metric orders translations less accurately. No order-preserving transform of the score can repair the second fault. The first is removed exactly by COMET-QN, which maps the score distri
    
[^40]: 惩罚框架下的无有效选项多选题问答：分析大语言模型在无效选项下的弃答行为

    Penalty-Framed No-Valid-Option MCQA: Analyzing LLM Abstention under Invalid Choices

    [https://arxiv.org/abs/2610.08153](https://arxiv.org/abs/2610.08153)

    该论文提出“惩罚框架下的无有效选项多选题问答”这一新评测设定，并通过基于正确回答的条件分析方法，揭示了大语言模型的高答题准确率并不能保证其在所有选项均无效时可靠地选择弃答。

    

    多选题问答（MCQA）通常被用于评估大语言模型，其前提假设是所提供的选项中必有一个正确答案，并通常以答案选择的准确率来衡量模型表现。然而，在实际部署中，用户或检索系统可能会提供无效的选项集合，即所列出的选项中没有一个是正确的，而此时强行选择其中一项可能会带来下游成本。我们将这种设定称为惩罚框架下的无有效选项多选题问答。基于MMLU-Pro的数学子集，我们移除标注的正确选项，允许模型选择剩余选项或输出ABSTAIN（弃答），并对无效的强制选择回答施加惩罚。我们进一步引入了基于正确回答的条件分析，仅在模型原本回答正确的实例上评估其弃答能力。实验表明，较高的MCQA准确率并不能完全保证弃答行为的可靠性：即使在明确的“无有效选项”感知指令和基于惩罚的约束下（摘要原文在此截断）。

    arXiv:2610.08153v1 Announce Type: cross  Abstract: Multiple-choice question answering (MCQA) is commonly used to evaluate large language models under the assumption that one of the provided options is correct, typically using answer-selection accuracy. However, in real deployments, users or retrieval systems may provide invalid option sets in which none of the listed choices is correct, and selecting one of them may incur downstream cost. We study this setting as penalty-framed no-valid-option MCQA. Using the mathematics subset of MMLU-Pro, we remove the labeled correct option, allow models to either choose a remaining option or output ABSTAIN, and penalize invalid forced-choice responses. We further introduce correct-conditioned analysis, evaluating abstention only on instances that the model originally answered correctly. Experiments show that high MCQA accuracy does not fully guarantee abstention reliability: even under explicit no-valid-option-aware instructions and penalty-based s
    
[^41]: 对话是一个二体问题：全双工对话模型的双向评估

    Conversation Is a Two-Body Problem: Dyadic Evaluation of Full-Duplex Dialogue Models

    [https://arxiv.org/abs/2610.08125](https://arxiv.org/abs/2610.08125)

    提出DyaFDB框架，让两个全双工对话模型在指定角色及合作或冲突目标下直接对话，并由外部评判器对双方同时评分，弥补了传统单边评估只能覆盖“二体问题”一半的不足。

    

    全双工语音对话模型可以同时听和说，使语音智能体能够实现基于轮次系统无法提供的自然、低延迟交互。然而，它们通常在与单边对话者的交互中进行评估：要么是无法作出反应的预录音音频，要么是能实时反应但只执行固定测试序列且自身从不被评分的自动考官。这些单边框架只评估了二体问题的一半，而在二体问题中，轮次转换、话语重叠和打断是两个相互耦合的说话者共同产生的结果。我们提出了DyaFDB，这是一个在双向设置中评估全双工模型的框架：两个模型在指定角色下直接对话，目标可以是合作或冲突的，并由外部评判器对双方进行离线评分。DyaFDB探究两个模型如何相互应对，例如它们如何在不同利益下进行轮次转换或履行所分配的角色。我们实例化了……（摘要原文在此处截断）

    arXiv:2610.08125v1 Announce Type: cross  Abstract: Full-duplex spoken dialogue models listen and speak at the same time, enabling voice agents to have natural, low-latency interactions that turn-based systems cannot offer. However, they are commonly evaluated against single-sided interlocutors: pre-recorded audio that cannot react, or an automated examiner that reacts in real time but only administers a fixed sequence of tests and is never graded. These single-sided frameworks evaluate only half of a two-body problem, where turn-taking, overlap, and interruption are joint products of two coupled speakers. We propose DyaFDB, a framework that evaluates full-duplex models in a dyadic setup: two models converse directly under assigned roles with cooperative or conflicting goals, and both sides are scored offline with an external judge. DyaFDB probes how the two models behave toward each other, such as how they take turns or carry an assigned role under different interests. We instantiate f
    
[^42]: 自然语言问题作为知识图谱的接口：QRAKEN图蒸馏与语义自修复

    Natural Language Questions as an Interface for Knowledge Graphs: QRAKEN Graph Distillation and Semantic Self-Healing

    [https://arxiv.org/abs/2610.08095](https://arxiv.org/abs/2610.08095)

    QRAKEN提出了一种无需训练的神经符号流水线，通过离线图蒸馏生成TTQL图谱证据来引导LLM生成SPARQL查询，并借助确定性的语法与数据模型检查实现迭代式语义自修复，在Text2SPARQL挑战赛上取得了严格的F1最佳成绩。

    

    自然语言访问RDF知识图谱是语义网的核心愿景。大型语言模型（LLM）推动了文本到SPARQL（Text-to-SPARQL）技术的发展，但在面对不熟悉的图谱时，它们往往会生成虽然语法有效、却与实际填充数据模型不符的查询。QRAKEN是一个无需训练、与本体无关的神经符号流水线，它将查询生成建立在经验性图证据而非模式预期之上。离线蒸馏器生成TTQL——对已填充的多跳模式、条件频率以及路径条件下的字面量示例的紧凑描述，并附带一个类-属性共现矩阵。在线阶段，TTQL引导LLM，同时确定性的语法、词汇和数据模型检查为迭代优化提供诊断。在CK25（首届国际Text2SPARQL挑战赛）上，基于QLever快照的相同条件重新计算下，QRAKEN使用GPT-4.1 mini达到严格F1值0.643±0.026，使用GPT-5.4达到0.652±0.012：相对……

    arXiv:2610.08095v1 Announce Type: new  Abstract: Natural-language access to RDF knowledge graphs is a core Semantic Web ambition. Large language models (LLMs) have advanced Text-to-SPARQL, yet on unfamiliar graphs they often generate valid queries that misrepresent the populated data model. QRAKEN is a training-free, ontology-agnostic neurosymbolic pipeline grounding generation in empirical graph evidence rather than schema expectations. An offline distiller produces TTQL, a compact description of populated multi-hop patterns, conditional frequencies and path-conditioned literal examples, plus a class-property co-occurrence matrix. Online, TTQL guides the LLM, while deterministic syntax, vocabulary and data-model checks provide diagnostics for iterative refinement. On CK25 (First International Text2SPARQL Challenge), under matched-condition recomputation on a QLever snapshot, QRAKEN achieves strict F1 of 0.643 $\pm$ 0.026 with GPT-4.1 mini and 0.652 $\pm$ 0.012 with GPT-5.4: relative g
    
[^43]: SAGE：面向有据可依医学问答数据合成的语义锚点引导演化框架

    SAGE: Semantic Anchor-Guided Evolution for Grounded Medical QA Data Synthesis

    [https://arxiv.org/abs/2610.08093](https://arxiv.org/abs/2610.08093)

    SAGE提出了一种数据合成框架，利用MeSH等轻量级公开分类体系作为语义锚点，通过迭代交替进行原子与关联合成，使小型本地部署模型也能从极少种子数据生成高质量的医学问答训练数据。

    

    开发适用于临床任务（如医学问答QA）的可靠模型，严重受限于高质量、专家标注训练数据的稀缺。严格的隐私要求，以及在资源有限的临床环境中使用大型开源语料库或专有云端API并不现实，进一步加剧了这一挑战。为解决这些障碍，我们提出了SAGE（语义锚点引导演化，Semantic Anchor-Guided Evolution），这是一种新颖的数据合成框架，使小型、本地部署的模型能够生成高质量的医学训练数据。SAGE利用轻量级、公开可用的分类体系（如MeSH）作为语义锚点，施加结构化先验以有效引导并锚定数据生成过程。其核心在于，SAGE迭代地交替进行原子（基于单个概念的）合成与关联（基于关系的）合成，从极少量种子数据出发自举生成训练数据。

    arXiv:2610.08093v1 Announce Type: cross  Abstract: Developing reliable models for clinical tasks, such as Medical Question Answering (QA), is severely constrained by the limited availability of high-quality, expert-annotated training data. This challenge is exacerbated by stringent privacy requirements and the impracticality of utilizing large open-source corpora or proprietary cloud APIs within resource-limited clinical settings. To address these obstacles, we introduce SAGE (\textit{Semantic Anchor-Guided Evolution}), a novel data synthesis framework that enables small, locally deployed models to generate high-quality medical training data. SAGE leverages lightweight, publicly available taxonomies such as MeSH as semantic anchors, imposing a structured prior to effectively guide and ground the data generation process. At its core, SAGE iteratively interleaves atomic (individual concept-based) and associative (relation-based) synthesis, bootstrapping training data from minimal seeds. 
    
[^44]: DirectSpeech2LLM：一个缓解语音-大语言模型提示过拟合的简单端到端框架

    DirectSpeech2LLM: A Simple End-to-End Framework to Mitigate Prompt Overfitting in Speech-LLMs

    [https://arxiv.org/abs/2610.08085](https://arxiv.org/abs/2610.08085)

    提出DirectSpeech2LLM端到端框架，仅用ASR数据训练即可保持LLM的指令遵循能力，实现对语音翻译和情感识别等未见任务的零样本泛化。

    

    语音-大语言模型常表现出提示过拟合现象，即仅在自动语音识别（ASR）指令上训练的模型无法泛化到语音翻译等新指令，仍然主要表现为ASR系统。我们提出了DirectSpeech2LLM，这是一个简单的端到端框架，在以语音为条件输入时能够保持LLM在未见任务上的指令遵循能力。该框架在冻结的LLM嵌入矩阵上计算基于距离的CTC损失，并使用贪心CTC标签分别推导几何对齐和时间对齐的语音嵌入，作为LLM的输入。仅使用960小时的LibriSpeech ASR数据进行训练，DirectSpeech2LLM在ASR（已见任务）上超越了级联系统，并能零样本泛化到语音翻译和情感识别（两个未见任务），在这两个新指令上的表现接近级联系统的上限，尽管训练期间从未见过这两个任务。

    arXiv:2610.08085v1 Announce Type: new  Abstract: Speech-LLMs often exhibit prompt overfitting, where models solely trained on automatic speech recognition (ASR) instruction fail to generalize to new instructions such as speech translation and continue to behave primarily as ASR system. We propose DirectSpeech2LLM, a simple end-to-end framework that preserves the instruction-following ability of the LLM on unseen tasks when conditioned on speech. It computes distance-based CTC loss over the frozen LLM embedding matrix and uses greedy CTC labels to derive geometrically and temporally aligned speech embeddings respectively as an input to the LLM. Trained solely on 960 hours of LibriSpeech ASR data, DirectSpeech2LLM outperforms the cascaded system on ASR (seen task) and generalizes zero-shot to speech translation and emotion recognition (two unseen tasks), closely matching the cascaded system upper bound on these two new instructions despite seeing neither during training. We also find tha
    
[^45]: POLAR：面向工具调用LLM智能体的本体引导式风险预防

    POLAR: Ontology-Guided Risk Prevention for Tool-Calling LLM Agents

    [https://arxiv.org/abs/2610.08082](https://arxiv.org/abs/2610.08082)

    POLAR是一个通过结构化两层本体评估操作可逆性的防护栏框架，能在工具调用LLM智能体执行高风险操作前将其剪除并提供可审计的结构化判定，但其收益因任务域和智能体能力而异。

    

    LLM工具使用智能体运行在动态环境中，其中许多操作带有运行风险。然而，大多数安全机制只在错误显现后才作出反应。现有的预防性方法要么通过思维链深思熟虑对智能体进行微调，要么将自然语言防护规则编译为运行时检查，但它们都没有提供结构化的、可审计的判定。我们提出了POLAR，一个针对小型工具调用智能体的防护栏框架，它通过结构化的两层本体来评估可逆性。POLAR通过推导候选逆操作序列，为每个操作分配一个分级的可逆性分数；未通过阈值的调用会在执行前被剪除。在τ²-bench上对六个智能体模型进行评估，POLAR在airline域中使六个智能体中的四个的平均任务奖励提高了0.11至0.18分，但在18个模型-域组合中只有8个总体上有所改善；retail域和更强的智能体往往出现性能回退。POLAR提供了一种可审计的结构化（摘要此处截断）

    arXiv:2610.08082v1 Announce Type: new  Abstract: LLM tool-use agents operate in dynamic environments where many actions carry operational risk. However, most safety mechanisms react only after errors manifest. Existing pre-emptive approaches either fine-tune the agent on chain-of-thought deliberation or compile natural-language guardrails into runtime checks, but they do so without exposing a structural, auditable verdict. We propose POLAR, a guardrail framework for small tool-calling agents that assesses reversibility through a structured two-layer ontology. POLAR assigns each action a graded reversibility score by deriving a candidate inverse sequence; calls failing a threshold are pruned before execution. Evaluated on $\tau^2$-bench across six agent models, POLAR improves mean task reward by 0.11 to 0.18 points on airline for four of six agents, but only eight of eighteen model--domain cells improve overall; retail and stronger agents often regress. POLAR provides an auditable struc
    
[^46]: 自我回溯蒸馏：将事后经验转化为先验预见

    Self-Retrospection Distillation: Turning Post-hoc Experiences into Prior Foresight

    [https://arxiv.org/abs/2610.08077](https://arxiv.org/abs/2610.08077)

    该论文提出前瞻学习与自我回溯蒸馏方法，将已完成的轨迹作为特权后见之明，蒸馏为同一策略在交互前的轨迹无关的预见性预测，从而在组相对奖励信号消失时仍能从经验中有效学习。

    

    具有可验证奖励的强化学习（RLVR）主要通过交互后的标量结果奖励将智能体经验转化为学习信号。然而，对于组相对目标而言，当所有采样轨迹获得相同奖励时，这一信号便会消失，即使这些轨迹可能揭示了关于任务需求以及智能体如何失败的有用信息。我们提出了一个互补的问题：事后反思能否教会智能体在行动之前本可预见的东西？我们引入前瞻学习，利用事后经验来监督交互前视角下的预见性预测，并通过自我回溯蒸馏（SRD）加以实例化。直观地说，一条已完成的轨迹揭示了本会有用的知识和本应避免的陷阱；SRD将这种特权的后见之明蒸馏到同一策略的、不依赖轨迹的前瞻预测中。前瞻仅作为训练目标，无需成为……

    arXiv:2610.08077v1 Announce Type: new  Abstract: Reinforcement learning with verifiable rewards (RLVR) turns agent experience into learning signals primarily through scalar outcome rewards after interaction. For group-relative objectives, however, this signal vanishes when all rollouts receive the same reward, even though their trajectories may reveal useful information about what the task requires and how the agent fails. We ask a complementary question: can hindsight teach an agent what it could have anticipated before acting? We introduce prospective learning, which uses post-hoc experience to supervise foresight predictions from the pre-interaction view, and instantiate it with Self-Retrospection Distillation (SRD). Intuitively, a completed trajectory reveals knowledge that would have been useful and pitfalls that should be avoided; SRD distills this privileged hindsight into trajectory-blind foresight of the same policy. Foresight serves only as a training target and need not be e
    
[^47]: HINTT参加第二届MLC-SLM挑战赛的系统提交：级联与统一方法在说话人日志和语音识别中的比较

    HINTT Submission to the 2nd MLC-SLM Challenge: Comparing Cascaded and Unified Approaches to Diarization and ASR

    [https://arxiv.org/abs/2610.08063](https://arxiv.org/abs/2610.08063)

    该论文提出了HINTT系统，比较了级联流水线（微调DiariZen说话人日志+微调Qwen3-ASR+LLM生成式纠错）与统一语音大语言模型两种策略来解决多语言说话人归属语音识别问题，最终采用级联方案并仅使用官方MLC-SLM数据进行训练。

    

    本文介绍了提交给第二届多语言会话语音大语言模型挑战赛与研讨会（MLC-SLM）的HINTT系统。我们解决的是多语言说话人归属的自动语音识别（ASR）问题，即系统需要确定谁在何时说话以及说了什么内容。我们针对该问题研究了两种建模策略：一种是将说话人日志（diarization）与基于语音大语言模型的ASR相结合的级联流水线，另一种是直接生成说话人标签、时间戳和转录文本的统一语音大语言模型。我们的最终提交系统基于级联流水线，由经过微调的DiariZen说话人日志模型、经过微调的Qwen3-ASR模型以及基于大语言模型的生成式纠错组成。作为对比，我们还使用相同的官方训练数据将VibeVoice-ASR微调为统一模型。所有任务相关的微调和模型选择仅使用官方MLC-SLM数据完成，未使用外部数据或伪标签。实验结果表……

    arXiv:2610.08063v1 Announce Type: cross  Abstract: This paper presents the HINTT system submitted to the 2nd Challenge and Workshop on Multilingual Conversational Speech Language Model (MLC-SLM). We address multilingual speaker-attributed ASR, where systems must determine who spoke when and what was spoken. We investigate two modeling strategies for this problem: a cascaded pipeline that combines speaker diarization with speech-LLM-based ASR, and a unified speech LLM that directly generates speaker labels, timestamps, and transcriptions. Our final submission is based on the cascaded pipeline, consisting of a fine-tuned DiariZen diarization model, a fine-tuned Qwen3-ASR model, and LLM-based generative error correction. For comparison, we also fine-tune VibeVoice-ASR as a unified model using the same official training data. All task-specific fine-tuning and model selection are performed using only the official MLC-SLM data, without external data or pseudo-labels. Experimental results dem
    
[^48]: 语言承载着专家的印象：基于评估工具锚定的LLM评判器可实现咨询质量评估的跨域迁移并超越域内训练

    Language Carries the Expert's Impression: Instrument-Anchored LLM Judges Transfer Counseling-Quality Assessment and Beat In-Domain Training

    [https://arxiv.org/abs/2610.08055](https://arxiv.org/abs/2610.08055)

    该研究发现，基于专家评分工具锚定构念、由小型开源LLM生成的会话级得分，能够跨领域迁移专家对咨询质量的总体印象评估，且跨域训练的效果甚至优于域内训练。

    

    arXiv:2610.08055v1 公告类型：new 摘要：对双人咨询对话中沟通质量的自动评估受制于数据瓶颈：专家评分的语料库规模小且扩充成本高昂。我们研究了专家总体印象预测在三个德语模拟咨询语料库（两个全科医疗语料库、一个学校相关的家长-教师语料库；n=195个经专家评分的会谈，其中一个语料库经过量表等值化处理）之间的跨域迁移。在其他域上训练的效果优于在目标域内训练：留一域迁移达到嵌套Spearman ρ = 0.54，而目标域内训练不超过0.48，会话层面的配对差距为+0.15，且当训练集规模匹配时该差距仍保持+0.12，因此这并不仅仅是数据量的问题。决定性的特征是来自小型开源权重LLM阅读双说话人转录文本所得到的会话级构念得分，这些构念主要源自专家的评分工具：由工具衍生的构念集合提升了……

    arXiv:2610.08055v1 Announce Type: new  Abstract: Automatic assessment of communication quality in dyadic counseling conversations is bottlenecked by data: expert-rated corpora are small and expensive to grow. We study cross-domain transfer of expert overall-impression prediction across three German corpora of simulated counseling (two general-practice medical, one school-related parent-teacher; $n=195$ expert-rated sessions, one corpus after scale equating). Training on the other domains beats training in-domain: leave-one-domain-out transfer reaches nested Spearman $\rho = 0.54$ against $\le 0.48$ within the target domain, a paired session-level gap of $+0.15$ that holds at $+0.12$ when the training-set sizes are matched, so it is not simply data volume. The decisive features are session-level construct scores from small open-weight LLMs reading the two-speaker transcript, with the constructs largely derived from the experts' rating instruments: the instrument-derived battery lifts a 
    
[^49]: DAEDALUS：从自生成任务中引导代理记忆

    DAEDALUS: Bootstrapping Agent Memory from Self-Generated Tasks

    [https://arxiv.org/abs/2610.08048](https://arxiv.org/abs/2610.08048)

    DAEDALUS通过探索者代理自生成练习任务、解决者代理从失败中提炼启发式规则并验证其有效性，从而在无需现有任务或人工验证器的情况下自动构建可复用的代理记忆。

    

    LLM代理通常缺乏在新环境中可靠行动所需的操作性知识，因为它们必须自行发现特定的工具行为或环境约定。由于没有对过去尝试的记忆，它们会在不同任务中重复相同的错误，导致更多的任务失败和更长的轨迹。为了解决这个问题，代理系统通常依赖人工编写的指南，或者依赖从训练任务和神谕验证器构建的程序性记忆，但这两种方式都需要对环境的先验知识。我们提出了DAEDALUS，这是一种在没有现成任务或神谕验证器的情况下，从自生成练习中引导可复用代理记忆的方法。DAEDALUS将两个代理配对：一个是与环境交互以生成具有挑战性但可解决任务的探索者，另一个是尝试解决这些任务的解决者。每次解决者失败后都会推导出一个启发式方法，只有当解决者在上下文中借助该启发式方法反复成功后，该启发式方法才会被接受。

    arXiv:2610.08048v1 Announce Type: new  Abstract: LLM agents often lack the operational knowledge to act reliably in new environments, as they must discover specific tool behaviors or environment conventions on their own. Without memory of past attempts, they repeat the same mistakes across tasks, leading to more task failures and longer trajectories. To address this, agentic systems typically rely on human-written guidelines or on procedural memory built from training tasks and an oracle verifier, both of which require prior knowledge of the environment. We present DAEDALUS, a method for bootstrapping reusable agent memory from self-generated practice without existing tasks or oracle verifiers. DAEDALUS pairs two agents: an explorer that interacts with the environment to generate challenging yet solvable tasks, and a solver that attempts them. A heuristic is derived from each solver failure and accepted only after the solver repeatedly succeeds with that heuristic in context. These out
    
[^50]: 语言模型是否具备文字系统感知能力？

    Are Language Models Script-Aware?

    [https://arxiv.org/abs/2610.08037](https://arxiv.org/abs/2610.08037)

    该研究首次系统考察语言模型的文字系统知识，发现所有模型在多文字语言中都能近乎完美地匹配输出文字系统（拉丁文字保真度超98%）并高频遵循指定文字的指令，且大模型表现优于小模型。

    

    语言模型经常生成非预期语言或文字系统的输出，这一现象被称为“目标外生成”。现有研究主要聚焦于语言选择，而文字系统知识这一维度尚未得到充分研究：在任何语言理解发生之前，用户必须首先识别模型响应中的图形符号。我们通过在多文字系统语言上测试小型语言模型和大型语言模型，来研究它们是否具备文字系统知识。通过两个互补实验，我们评估了模型是否（1）使其输出文字系统与输入相匹配，以及（2）遵循明确指令以指定的文字系统生成文本。所测试的模型展现出了相当程度的文字系统知识：它们都达到了近乎完美的拉丁文字系统保真度（超过98%），并高频地遵循文字系统指令。尽管如此，我们注意到LLM与SLM之间存在差异，其中一方的得分更高（摘要在此处截断）。

    arXiv:2610.08037v1 Announce Type: new  Abstract: Language models frequently generate outputs in unintended languages or scripts, a phenomenon known as off-target generation. While existing research has focused on language selection, the dimension of script knowledge remains understudied: before any linguistic understanding can occur, users must recognize the graphic symbols in a model's response. We investigate whether Small and Large Language Models (SLMs and LLMs) possess script knowledge by testing them on multi-scriptic languages. Through two complementary experiments, we evaluate whether models (1) adapt their output script to match the input, and (2) follow explicit instructions to generate text in a specified script. The models we tested demonstrate substantial script knowledge: they all achieve a near-perfect Latin script fidelity (more than 98%) and follow script instructions with high frequency. Nevertheless, we notice differences between LLMs and SLMs, with higher scores for
    
[^51]: 幻觉检测基准中的标注问题：一项实证评估

    The Labeling Problem in Hallucination Detection Benchmarks: An Empirical Evaluation

    [https://arxiv.org/abs/2610.08026](https://arxiv.org/abs/2610.08026)

    该论文通过实证评估揭示了幻觉检测基准中自动化标注在“参考忠实性”与“事实正确性”两个标准之间存在的方法论模糊性，并研究了由此导致的标注标准错配问题。

    

    近年来，已有多种检测大语言模型（LLM）何时产生幻觉的方法被开发出来。这些方法通常使用包含问题及对应简短参考答案的开放域问答（QA）数据集进行基准测试。首先，使用一个LLM对QA数据集中的问题生成答案；然后，采用某种自动化标注策略，通过将这些答案与数据集中的参考答案进行比较，将其标注为是否为幻觉。这种评估设置在两个标准之间产生了方法论上的模糊性：参考忠实性（答案是否被参考答案完全支持）与事实正确性（答案是否不存在矛盾且没有事实错误的具体表述）。在实践中，自动化标注器即使在预期目标是后者时，也可能采用前者的标准。我们使用900个经人工标注的问题来研究这种潜在的标注标准错配问题。

    arXiv:2610.08026v1 Announce Type: new  Abstract: In recent years, several methods for detecting when large language models (LLMs) hallucinate have been developed. These methods are often benchmarked with open-domain question answering (QA) datasets containing questions and corresponding short reference answers. First, an LLM is used to generate answers to questions within the QA dataset. Then, some automated labeling strategy is used to label these answers as hallucinated or not by comparing them with the reference answers in the dataset. This evaluation setting creates a methodological ambiguity between two criteria: reference faithfulness (whether the answer is fully supported by the reference) and factual correctness (whether the answer is free from contradictions and factually false specific claims). In practice, automated labelers may apply the former criterion even when the intended target is the latter. We study this potential criterion mismatch using 900 human-labeled question-
    
[^52]: 结构化却沉默：探测大语言模型隐状态中的能力需求

    Structured but Silent: Probing Capability Requirements in LLM Hidden States

    [https://arxiv.org/abs/2610.08018](https://arxiv.org/abs/2610.08018)

    本文提出TACIT框架，将工具使用所需的能力需求沿来源、变换和世界效应三个维度分解为八个结构化类别，并证明这些细粒度的查询端能力需求在LLM生成答案之前的隐藏状态中即可被线性解码。

    

    可靠的工具使用不仅仅是触发某种机制或将查询与API描述相匹配。在选择特定工具之前，智能体必须首先推断出用户查询所隐含的能力需求。本文研究了这些查询端的能力需求在生成之前是否可以从大语言模型的隐藏表示中被线性解码，以及这种隐状态可访问性与显式的语言分类相比表现如何。我们提出了TACIT框架，它沿三个基本轴（来源、变换和世界效应）分解外部需求，定义了八个结构上截然不同的能力类别。使用来自基准测试、合成示例和新领域场景的1,600个平衡训练查询，我们在四个开源权重LLM系列的生成前隐状态上训练线性探针。我们的实证结果表明，细粒度的能力结构是可以被线性解码的……

    arXiv:2610.08018v1 Announce Type: new  Abstract: Reliable tool use requires more than triggering a mechanism or matching a query to an API description. Before selecting a specific tool, an agent must first infer the capability requirements implied by the user query. In this paper, we investigate whether these query-side capability requirements are linearly decodable from LLM hidden representations prior to generation, and how this hidden-state accessibility compares with explicit verbal classification. We introduce TACIT, a framework that decomposes external requirements along three fundamental axes: Source, Transformation, and World Effect, defining eight structurally distinct capability classes. Using 1,600 balanced training queries from benchmarks, synthetic examples, and new domain scenarios, we train linear probes on pre-generation hidden states from four open-weight LLM families. Our empirical results demonstrate that fine-grained capability structures are linearly decodable with
    
[^53]: 对模型合并的更广泛审视：重新思考任务算术所引发的隐式正则化

    A Broader Look at Model Merging: Rethinking Implicit Regularization Induced by Task Arithmetic

    [https://arxiv.org/abs/2610.07990](https://arxiv.org/abs/2610.07990)

    本论文发现主流模型合并方法中系数搜索带来的隐式正则化实际上限制了性能，去除该正则化、直接优化合并模型乃至预训练模型的权重，即可在多种架构和极低数据量场景下显著提升合并后的多任务性能。

    

    模型合并旨在通过组合各个特定任务模型的权重，以低成本的方式构建多任务模型。为了在多个任务上表现良好，大多数现有的合并方法使用额外的数据集来寻找特定任务权重更新的最佳线性组合系数。然而，我们在这项标准实践中识别出一种隐式正则化：对系数进行搜索会将候选模型限制在由特定任务权重更新所张成的子空间内。在本工作中，我们研究这种正则化是否真的有用。令人惊讶的是，实证结果表明，在没有这种正则化的情况下直接优化合并模型的权重，能够在多种架构、多个领域、甚至在每个类别仅有一个样本的极端数据受限场景下显著提升常见合并方法的性能。此外，直接优化预训练模型的权重甚至优于一些现有方法……（摘要在此处被截断）

    arXiv:2610.07990v1 Announce Type: cross  Abstract: Model merging aims to build a multi-task model cheaply by combining the weights of individual task-specific models. To perform well across multiple tasks, most existing merging methods use an additional dataset to find the coefficients for the best linear combination of task-specific weight updates. However, we identify an implicit regularization in this standard practice: searching over coefficients restricts the candidate models to a subspace spanned by task-specific weight updates. In this work, we investigate whether this regularization is actually useful. Surprisingly, empirical results show that optimizing merged-model weights without this regularization significantly boosts the performance of common merging methods across multiple architectures, domains, and even in an extremely data-limited scenario where only one instance is available per class. Moreover, directly optimizing the pretrained model weights even outperforms some e
    
[^54]: VisionWeave：将弹性视觉表示编织作为多模态大语言模型的固有能力

    VisionWeave: Weaving Elastic Visual Representations as a Native Capability of MLLMs

    [https://arxiv.org/abs/2610.07987](https://arxiv.org/abs/2610.07987)

    VisionWeave通过大规模训练，使多模态大语言模型获得按内容自适应地决定视觉表示位置与粒度的“弹性视觉表示编织”固有能力，在保留关键细节的同时显著降低计算成本。

    

    多模态大语言模型（MLLM）已成为视觉理解的主导范式，但其将输入编码为密集、固定大小的patch token会带来巨大开销。然而，视觉信息的分布是不均匀的：某些区域需要细粒度的细节，而其他区域则可以使用紧凑的表示。下采样会牺牲这些细节，而现有的token剪枝和自适应方法在内容自适应粒度、任务泛化以及与现代MLLM和服务基础设施的集成方面仍然受限。克服这些限制需要基础模型能够端到端地学习在哪里以及以何种粒度分配视觉表示，我们将这一固有能力称为弹性视觉表示编织。我们提出了VisionWeave，通过大规模训练在前沿水平的MLLM中建立这种能力。它结合了两个组件：门控空间池化器构建粗粒度表示……

    arXiv:2610.07987v1 Announce Type: cross  Abstract: Multimodal large language models have become the dominant paradigm for visual understanding, but incur substantial costs by encoding inputs into dense, fixed-size patch tokens. However, visual information is unevenly distributed: some regions require fine-grained detail, while others admit compact representations. Downsampling sacrifices this detail, while existing token pruning and adaptive approaches remain limited in content-adaptive granularity, task generalization, and integration with modern MLLMs and serving infrastructure. Overcoming these limitations calls for foundation models that learn, end to end, where-and at what granularity-to allocate visual representations, a native capability we term elastic visual representation weaving. We introduce VisionWeave, establishing this capability in frontier-level MLLMs through large-scale training. It combines two components: a gated spatial pooler constructs coarse-grained representati
    
[^55]: 置信推理图：面向大语言模型智能体的结构化置信度估计

    Confidence Reasoning Graphs: Structured Confidence Estimation for LLM Agents

    [https://arxiv.org/abs/2610.07948](https://arxiv.org/abs/2610.07948)

    提出置信推理图（CRG），一种推理时框架，通过结构化分解单条轨迹中的证据来估计LLM智能体完成任务的概率，无需访问模型内部信号或训练数据。

    

    在重要领域中LLM智能体时，要对是否信任其输出或进行干预做出明智决策，需要对智能体任务成功与否具备经过校准的置信度。智能体的置信度估计十分困难，因为关于成功的证据分散在智能体轨迹中异构且相互依赖的多个步骤之中。实际的智能体部署还带来了更多挑战：前沿大语言模型通常对内部信号的访问受限，智能体的多次运行成本高昂，且训练数据可能不可用或很快过时。为了应对这些挑战，我们提出了置信推理图，这是一个推理时框架，能够从单条轨迹中估计智能体完成任务的概率，而无需特权级别的模型访问权限或训练数据。CRG并非将执行过程压缩为单一的总体判断，而是从“智能体完成了任务”这一论断出发，将其分解为……（原文摘要在此处截断）

    arXiv:2610.07948v1 Announce Type: new  Abstract: When using an LLM agent in a consequential domain, making an informed decision about whether to trust its output or intervene requires calibrated confidence in the agent's success. Confidence estimation for agents is difficult because evidence about success is distributed across heterogeneous, interdependent steps of an agent's trajectory. Practical agentic deployments introduce further challenges: frontier LLMs often provide limited access to internal signals, agent roll-outs are costly, and training data may be unavailable or quickly become outdated. To address these challenges, we introduce Confidence Reasoning Graphs (CRGs), an inference-time framework that estimates the probability an agent accomplished its task from a single trajectory, without privileged model access or training data. Rather than compressing an execution into a single holistic judgment, a CRG begins with the claim that the agent accomplished its task, decomposes i
    
[^56]: 循环语言模型的混合潜在注意力

    Hybrid Latent Attention for Looped Language Models

    [https://arxiv.org/abs/2610.07940](https://arxiv.org/abs/2610.07940)

    提出混合潜在注意力（HLA），通过将旧 token 压缩为紧凑的潜在表示而非存储完整键值缓存，使循环语言模型的缓存缩小 10.7 倍、每 GPU 并发序列容量提升 4.0-8.8 倍、解码吞吐量最高提升 7.4 倍，同时保留原模型 97% 以上的性能。

    

    循环语言模型对每个 token 重复应用同一组层 T 次，这在不增加参数的情况下加深了模型，但使其键值（KV）缓存膨胀了 T 倍。更大的缓存限制了 GPU 一次能解码的序列数量，并且由于每步解码都需要读取整个缓存，从而减慢了解码速度。我们提出了混合潜在注意力（HLA），它在最近 W 个 token 的滑动窗口内保留精确的键和值，并将每个较旧的 token 存储为一个紧凑的潜在表示，每个循环的查询可以直接读取该表示，而无需重建键和值。我们在具有 1.4B 和 2.6B 参数的 Ouro 循环模型（T=4）上对 HLA 进行再训练，保持预训练权重冻结，仅训练新增参数以复现原始注意力。每个 token 的缓存缩减 10.7 倍，使每个 GPU 能容纳 4.0-8.8 倍的并发序列，解码吞吐量在 1K token 上下文中提升 2.5 倍，在 16K 时最高提升 7.4 倍。HLA 保留了原模型超过 97% 的性能。

    arXiv:2610.07940v1 Announce Type: cross  Abstract: Looped language models apply the same stack of layers T times to each token, which deepens the model without adding parameters but multiplies its key-value (KV) cache by T. The larger cache limits how many sequences a GPU can decode at once and slows each decoding step, which reads the whole cache. We propose Hybrid Latent Attention (HLA), which keeps exact keys and values within a sliding window of W recent tokens and stores each older token as a compact latent that the query of each loop reads directly, without reconstructing keys and values. We uptrain HLA on Ouro looped models (T=4) with 1.4B and 2.6B parameters, keeping the pretrained weights frozen and training only the added parameters to reproduce the original attention. The cache shrinks by 10.7x per token, fitting 4.0-8.8x as many concurrent sequences per GPU, and decoding throughput improves by 2.5x at 1K-token contexts and by up to 7.4x at 16K. HLA retains over 97% of the o
    
[^57]: 利用四象限方法评估Redpine Science

    Leveraging a four-quadrant approach for evaluating Redpine Science

    [https://arxiv.org/abs/2610.07937](https://arxiv.org/abs/2610.07937)

    本报告采用公开基准与专家验证问题集相结合的四象限评估方法，证明配备Redpine Science的智能体在科学文献问答任务上显著优于无检索基线，正确率从87.6%提升至94.4%。

    

    Redpine Science为模型和智能体提供了一个统一的访问入口，可通过模型上下文协议和API直接查询大量经同行评审的文献。本报告从两个层面评估Redpine Science：检索到的文本片段的相关性，以及模型在使用Redpine Science时与使用网络搜索相比的回答质量。评估同时采用了公开基准和专家验证基准。公开基准是测试模型开发的广泛认可的方式，并且可以在各实验室之间进行比较，但存在基准饱和和数据记忆化的风险。为解决这一问题，我们用一套专家验证的问题集作为补充。本报告总共呈现了四项评估。在公开的回答质量基准ScholarQABench SciFact上，配备Redpine Science的智能体能够正确回答94.4%的论断，而不使用检索时仅为87.6%。在专家验证的问题集上，配备Redpine Science的智能体陈述了80.1%的必要论断……

    arXiv:2610.07937v1 Announce Type: new  Abstract: Redpine Science gives models and agents a single access point to a wide range of peer-reviewed literature, queried directly through the Model Context Protocol (MCP) and an API. This report evaluates Redpine Science on two levels: the relevance of the retrieved chunks, and a model's answer when it has access to Redpine Science compared to web search. Both public and expert-validated benchmarks are used. Public benchmarks are a widely accepted way to test model development and are comparable across labs, but risk saturation and memorization. To address this, we complement them with an expert-validated question set. In total, this report presents four evaluations. On ScholarQABench SciFact, the public answer-quality benchmark reported here, an agent with Redpine Science answers 94.4% of claims correctly against 87.6% with no retrieval. On the expert-validated question set, an agent with Redpine Science states 80.1% of the required claims ag
    
[^58]: 伪词作为探针：大语言模型几乎不具备支配人类伪词加工的亚词汇敏感性

    Pseudowords as probes: Large Language Models show little of the sublexical sensitivity that governs human pseudoword processing

    [https://arxiv.org/abs/2610.07936](https://arxiv.org/abs/2610.07936)

    通过意大利语伪词二选一实验发现，大语言模型缺乏人类所具备的亚词汇敏感性，在仅有伪词的条件下其表现甚至不及字符n-gram模型fastText，表明模型并未习得支配人类伪词加工的亚词汇线索。

    

    系统性，即形式到意义的概率映射，渗透在语言的各个层面，且亚词汇线索已被证明支配着人类的伪词加工。然而，大语言模型是否对这些线索表现出类似的敏感性仍不清楚。我们在两个意大利语二选一强迫选择伪词实验中测试了五个大语言模型，并将它们的反应与人类行为基线进行比较。当真实词选项提供了词汇熟悉度线索时，大语言模型与人类的一致性更为可靠；而在仅有伪词的条件下，它们的表现明显低于fastText（一个字符n-gram模型）。此外，可靠地驱动人类与fastText一致性的亚词汇余弦相似度线索并不能稳定地转移到人类与大语言模型的一致性上，而且推理token的消耗与人类加工难度之间也没有一致的关系。这些发现表明，大语言模型并不一定共享支配人类伪词加工的亚词汇线索。

    arXiv:2610.07936v1 Announce Type: new  Abstract: Systematicity, the probabilistic mapping of form to meaning, permeates language at all levels, and sublexical cues have been shown to govern human pseudoword processing. Yet whether LLMs exhibit comparable sensitivity to these cues remains unclear. We tested five LLMs on two Italian two-alternative forced-choice pseudoword experiments and compared their responses with a human behavioural baseline. LLMs aligned more reliably with humans when real-word options provided a lexical familiarity cue than in the pseudoword-only condition, where they fell substantially below fastText, a character-n-gram model. In addition, the sublexical cosine-similarity cue that reliably drove human--fastText agreement did not consistently transfer to human--LLM alignment, and reasoning-token expenditure bore no consistent relation to human processing difficulty. These findings suggest that LLMs do not necessarily share the sublexical cues that govern human pse
    
[^59]: 各向同性却不可解码：潜在预测文本表示中的序列内容充分性鸿沟

    Isotropic Yet Undecodable: The Sequential Content-Sufficiency Gap in Latent-Predictive Text Representations

    [https://arxiv.org/abs/2610.07906](https://arxiv.org/abs/2610.07906)

    论文通过信息论分解揭示序列内容充分性鸿沟，证明潜在表示的各向同性与一致性无法保证有序目标信息可解码，并据此提出引入规范词元监督的非自回归框架CANOPE，将位置信息恢复率从13.5%大幅提升至98.8%。

    

    我们通过考察一个表示是否保留了其输入中可获得的有序目标信息，来研究序列内容充分性问题。一种信息论分解方法将输入歧义、表示损失和读出失配三者分离开来。我们构造了可恢复的视图，其中完美一致性与联合各向同性高斯性可以与零目标信息共存，并确立了确定性规范锚点所施加的限制。词元对数损失提供了单边的信息损失界；固定惩罚的岭回归分析说明了为什么仅凭秩无法确定预测风险。这些结果催生了CANOPE——一个具有有序潜在画布、规范词元监督和几何正则化的非自回归框架。在40,000条验证序列上，潜在一致性（PL0）与词元锚定（PL2）具有几乎相同的合并秩，但在强自然损坏下分别仅达到13.5%和98.8%的位置Recall@1。

    arXiv:2610.07906v1 Announce Type: new  Abstract: We study sequential content sufficiency by investigating whether a representation retains the ordered target information available in its input. An information-theoretic decomposition separates input ambiguity, representation loss, and readout mismatch. We construct recoverable views where perfect agreement and joint isotropic Gaussianity coexist with zero target information, and establish limits imposed by deterministic canonical anchors. Token log-loss provides a one-sided information-loss bound; a fixed-penalty ridge analysis shows why rank alone cannot determine prediction risk. These results motivate CANOPE, a nonautoregressive framework with ordered latent canvases, canonical-token supervision, and geometric regularization. On 40,000 validation sequences, latent-agreement (PL0) and token-grounded (PL2) have nearly identical pooled ranks but reach 13.5% and 98.8% positional Recall@1, respectively, under strong natural corruption whe
    
[^60]: ARIA：面向粤语歌词创作的音频驱动旋律-声调关系建模

    ARIA: Audio-Driven Melody-Tone Relation Modeling for Cantonese Lyric Authoring

    [https://arxiv.org/abs/2610.07902](https://arxiv.org/abs/2610.07902)

    提出ARIA两阶段框架，通过三流关系感知声调估计器从歌唱音频中预测粤语声调序列，并利用解耦检索增强的声调条件生成器，实现直接从原始歌唱录音生成与旋律对齐的粤语歌词。

    

    粤语歌词创作要求词汇声调与旋律音高紧密对齐。现有的旋律引导歌词生成方法通常依赖符号化旋律来生成歌词。然而，在实际歌曲创作场景中，旋律往往以原始歌唱音频或哼唱录音的形式呈现，其中音高是隐式的、含噪声且非结构化的，这使得这些方法难以直接应用。为了解决这一局限，我们提出了ARIA，一个用于粤语歌词创作的两阶段音频驱动旋律-声调关系建模框架，它能够在给定字符级时间戳的情况下，从歌唱录音中生成粤语歌词。具体而言，我们首先设计了三流关系感知声调估计器（TRATE），通过建模多流声学线索和关系性声调结构，从带时间戳的歌唱音频中预测0243声调序列。然后，我们提出了解耦检索增强的声调条件歌词生成器（DRA-TCLG）来生成……

    arXiv:2610.07902v1 Announce Type: new  Abstract: Cantonese lyric writing requires close alignment between lexical tones and melodic pitch. Existing melody-guided lyric generation methods typically rely on symbolic melody to generate lyrics. However, in real songwriting scenarios, melodies are often expressed as raw singing audio or hummed recordings, where pitch is implicit, noisy, and unstructured, making these methods difficult to apply directly. To address this limitation, we propose ARIA, a two-stage audio-driven melody-tone relation modeling framework for Cantonese lyric authoring that generates Cantonese lyrics from singing recordings with provided character-level timestamps. Specifically, we first design a Tri-Stream Relation-Aware Tone Estimator (TRATE) to predict 0243 sequences from timestamped singing audio by modeling multi-stream acoustic cues and relational tonal structure. We then propose a Decoupled Retrieval-Augmented Tone-Conditioned Lyric Generator (DRA-TCLG) to gener
    
[^61]: 重新思考大语言模型的忠实性：一种成对上下文敏感的视角

    Rethinking Faithfulness in LLMs: A Pairwise Context-Sensitive Perspective

    [https://arxiv.org/abs/2610.07894](https://arxiv.org/abs/2610.07894)

    提出成对忠实性基准PFaithBench，通过对比同一问题在支持性与非支持性上下文下的表现，评估大语言模型能否正确地在回答与拒答之间切换，揭示忠实性本质上取决于回答与拒答之间的权衡。

    

    大语言模型（LLMs）被期望基于所提供的上下文忠实地回答问题，并在上下文信息不足以回答问题时选择拒答。现有的忠实性评估通常孤立地评估每个问题-上下文实例；然而，这种实例级评估未能捕捉忠实行为的一个基本要求：模型响应能够随可用上下文变化而调整的能力。具体而言，当存在充分证据时，模型应给出正确答案；当证据不足时，模型应选择拒答。在这项工作中，我们提出了一个成对忠实性基准，评估模型在支持性上下文与非支持性上下文下，能否针对同一问题在回答与拒答之间进行切换。我们对来自七个模型系列的三十九个模型的评估表明，忠实性在根本上涉及回答与拒答之间的权衡……

    arXiv:2610.07894v1 Announce Type: new  Abstract: Large language models (LLMs) are expected to answer questions faithfully based on the provided context, abstaining when the context information is insufficient to answer the questions. Existing faithfulness evaluations typically assess each question-context instance in isolation; however, such instance-level evaluation fails to capture a fundamental requirement of faithful behavior: the ability to adapt model responses to changes in available contexts. In particular, a model should provide correct answers when sufficient evidence is present and abstain when it is not. In this work, we propose a Pairwise Faithfulness Benchmark (PFaithBench) that evaluates whether a model can switch between answering and abstaining for the same question under supporting versus non-supporting contexts. Our evaluations across thirty-nine models with seven model families demonstrate that faithfulness fundamentally involves a trade-off between answering and ab
    
[^62]: 统一多模态模型中的视觉拒答

    Visual Abstention in Unified Multimodal Models

    [https://arxiv.org/abs/2610.07887](https://arxiv.org/abs/2610.07887)

    该论文形式化了“视觉拒答”概念并构建了Draw-or-Decline基准，揭示出统一多模态模型的编辑能力与拒答能力相互独立——即便最强的编辑模型也几乎不会拒绝不可行的编辑请求。

    

    统一多模态模型（UMMs）将理解与生成能力整合于一体，但其生成行为很少受到其对任务理解的约束。我们对“视觉拒答”进行了形式化定义：当所请求的视觉变换在任务规则下不可能实现时，模型应当认识到不存在有效解，明确说明这一点，并拒绝生成。我们提出了Draw-or-Decline（DoD）基准，包含跨越7个任务类别的1,050对可行-不可行请求对，用以联合评估编辑成功与否以及对不可行请求的拒绝能力。通过对8个统一多模态模型的评估，我们发现编辑能力与拒答能力是两种截然不同的能力：即使在普通指令下编辑准确率高达68.4%的最强编辑模型，也仅拒答了0.4%的不可行请求。模型的推理过程揭示了原因：这些模型很少察觉请求中的冲突，反而将编辑当作可行的请求来规划，常常描述图像中并不存在的物体。

    arXiv:2610.07887v1 Announce Type: cross  Abstract: Unified multimodal models (UMMs) integrate understanding and generation, yet their generative behavior is rarely governed by what they understand about the task. We formalize visual abstention: when a requested visual transformation is impossible under the task's rules, the model should recognize that no valid solution exists, state this, and decline to generate. We introduce Draw-or-Decline (DoD), a benchmark of 1,050 feasible-infeasible request pairs across 7 task categories that jointly measures editing success and the refusal of infeasible requests. Evaluating 8 UMMs, we find that editing ability and abstention are distinct capabilities: even the strongest editor, at 68.4% editing accuracy, refuses only 0.4% of infeasible requests under ordinary instructions. Their reasoning shows why: the models rarely notice the conflict, and instead plan the edit as if the request were possible, often describing objects that are not in the image
    
[^63]: ReFold：面向长程智能体的免训练可逆轮间上下文折叠方法

    ReFold: Training-Free Reversible Inter-Turn Context Folding for Long-Horizon Agents

    [https://arxiv.org/abs/2610.07863](https://arxiv.org/abs/2610.07863)

    ReFold提出了一种免训练的可逆上下文折叠渲染层，通过将已展示内容替换为占位符、将智能体报告完成的轮次折叠为一行注释来消除轮间冗余，从而在保留底层完整交互历史的同时压缩长程智能体的渲染上下文，避免了现有预测性方法带来的运行时开销、前缀缓存失效和不可逆信息丢失。

    

    长程LLM智能体基于只追加（append-only）的交互历史进行行动，该历史在每一步都会被重新发送给模型，因此上下文及其成本随步骤不断增长，直到会话超出上下文窗口。现有方法通过上下文需求预测来管理上下文，依赖于额外的模型调用、启发式规则或训练得到的策略。然而，这些预测性方法会引入运行时开销、使前缀缓存失效，并永久丢弃内容且无法保证恢复。为克服这些限制，我们提出了ReFold：一个免训练的渲染层，它在保留底层交互历史的同时仅压缩模型的渲染上下文。它无需辅助预测器即可消除两类轮间冗余：较早轮次已经展示过的内容（用占位符替换），以及智能体自身报告已完成的轮次（折叠为一行注释）。两种操作均使用分块渲染与重写（摘要在此处截断）……

    arXiv:2610.07863v1 Announce Type: cross  Abstract: Long-horizon LLM agents act on an append-only interaction history that is re-sent to the model at every step, so the context and its cost grow with steps until the sessions exceed the context window. Existing methods manage the context through context requirement prediction, relying on additional model calls, heuristic rules, or trained policies. However, these predictive approaches introduce runtime overhead, invalidate prefix caches, and permanently discard content with no guarantee of recovery. To overcome these limitations, we introduce ReFold: a training-free rendering layer that preserves the underlying interaction history while compressing only the model's rendered context. It removes two kinds of inter-turn redundancy without an auxiliary predictor: content an earlier turn already displayed, replaced by a stub, and turns the agent itself reports finished, folded into a one-line note. Both operators use chunked rendering, rewrit
    
[^64]: 迷失于bf16类型转换：导出三值语言模型可能抵消大部分低学习率代码改动的效果

    Lost in the bf16 Cast: Exporting Ternary Language Models Can Revert Most Low-Learning-Rate Code Changes

    [https://arxiv.org/abs/2610.07853](https://arxiv.org/abs/2610.07853)

    该研究审计发现，三值语言模型导出流程中先将潜在权重转换为bf16，会在阈值处因“舍入到偶数”规则被错误映射为零，导致Falcon-E-1B-Base部署后GSM8K准确率从58.79%暴跌至0.78%，几乎抹去了低学习率微调带来的改进。

    

    BitNet b1.58、Falcon-E和BitCPM等三值语言模型先以更高精度的潜在权重进行微调，再通过导出步骤生成三值代码进行部署；在三个实验室公开记录的流程中，该导出步骤均会先将潜在权重转换为bf16。我们对三个实验室的这些流程进行了审计。在已发布的检查点中，对随附潜在权重执行fp32量化所得的结果与实际部署代码在Falcon-E和BitCPM上有0.83%–1.77%的代码不一致，在BitNet 2B-4T上有1.530%不一致；对于Falcon-E和BitCPM，绝大多数不一致源于这样的乘积：其bf16舍入恰好落在阈值上，并被“舍入到偶数”（ties-to-even）规则映射为零，而未经修改的onebitllms导出器可以逐字节复现全部四个Falcon-E发布版本。在微调终点，当学习率按标称的学习率与bf16-ULP之比选取时，按文档所述流程导出会使Falcon-E-1B-Base的贪婪解码GSM8K严格准确率从58.79%跌至0.78%，使BitCPM-CANN-（原文摘要在此处截断）的准确率从36.13%跌至0.39%。

    arXiv:2610.07853v1 Announce Type: cross  Abstract: Ternary language models such as BitNet b1.58, Falcon-E and BitCPM are fine-tuned with higher-precision latent weights and deployed as ternary codes produced by an export step that, in the labs' documented pipelines, first casts the latents to bf16. We audit those pipelines across three labs. In released checkpoints, fp32 quantization of the shipped latents disagrees with the deployed codes on 0.83-1.77% of codes in Falcon-E and BitCPM and on 1.530% in BitNet 2B-4T; for Falcon-E and BitCPM most disagreements are products that bf16 rounding lands exactly on the threshold, which ties-to-even maps to zero, and the unmodified onebitllms exporter reproduces all four Falcon-E releases byte for byte. At fine-tuned endpoints, with learning rates selected to match a nominal learning-rate-to-bf16-ULP ratio, the documented export lowers greedy GSM8K strict accuracy from 58.79% to 0.78% for Falcon-E-1B-Base and from 36.13% to 0.39% for BitCPM-CANN-
    
[^65]: 面向大语言模型参数高效微调的动态位置注意力调制

    Dynamic Positional Attention Modulation for Parameter-Efficient Fine-Tuning of Large Language Models

    [https://arxiv.org/abs/2610.07848](https://arxiv.org/abs/2610.07848)

    提出DyPAM方法，通过在查询和键表征上结合输入条件化的逐维度调制与逐头逐层的结构化调制，动态调整位置信息对注意力的贡献，实现更精细的参数高效微调。

    

    参数高效微调（PEFT）已成为将大语言模型适配到下游任务的标准方法。然而，大多数现有的PEFT方法依赖于统一且静态的适配方式，没有考虑注意力在维度、注意力头、层以及输入标记之间的结构化异质性。在实践中，注意力表征表现出非均匀的行为，并且旋转位置编码等位置编码机制会引入依赖于维度的位置结构，使得统一的适配方式并非最优。在本工作中，我们提出了DyPAM（动态位置注意力调制），这是一种参数高效微调方法，通过直接作用于查询和键表征，来调整位置信息对注意力的贡献方式。DyPAM将基于输入条件的逐维度调制与逐注意力头、逐层的结构化调制相结合，对位置注意力进行细粒度的适配。

    arXiv:2610.07848v1 Announce Type: cross  Abstract: Parameter-efficient fine-tuning (PEFT) has become a standard approach for adapting large language models to downstream tasks. However, most existing PEFT methods rely on uniform and static adaptations, without accounting for the structured heterogeneity of attention across dimensions, heads, layers, and input tokens. In practice, attention representations exhibit non-uniform behavior, and positional encoding mechanisms such as rotary positional embeddings (RoPE) induce dimension-dependent positional structure, making uniform adaptation suboptimal. In this work, we propose DyPAM (Dynamic Positional Attention Modulation), a PEFT method that adapts how positional information contributes to attention by operating directly on the query and key representations. DyPAM combines input-conditioned, dimension-wise modulation with head-wise and layer-wise structural modulation, performing fine-grained adaptation of positional attention aligned wit
    
[^66]: “省略行动”：在哲学分歧下测量框架不变的不作为偏差

    OMIT the Action: Measuring Framing-Invariant Omission Bias under Philosophical Disagreement

    [https://arxiv.org/abs/2610.07847](https://arxiv.org/abs/2610.07847)

    本文提出OMIT基准（基于五视角哲学角色小组分歧构建的218个成对框架场景、覆盖10种冲突类型），评估发现八个LLM普遍存在不作为偏差，且该偏差在同一模型家族内随模型规模增大而减小。

    

    随着大语言模型越来越多地辅助道德推理，不作为偏差——即即使等效的框架表述会颠倒实质性结果，模型仍倾向于选择不行动——对决策的公正性构成了重大风险。然而，不作为偏差在LLM评估中仍未得到充分研究，现有的少数研究规模有限，且主要聚焦于功利主义与义务论之间的冲突。为填补这一空白，我们提出了OMIT基准，它包含218个横跨10种冲突类型的成对框架场景，通过利用基于LLM的五视角哲学角色小组（功利主义、义务论、美德伦理学、关怀伦理学和契约论）的分歧模式构建而成。通过对八个LLM的评估，我们发现不作为偏差普遍存在，但在同一模型家族内与模型规模呈负相关。我们进一步评估了四种推理时干预方法，发现鼓励模型在做出行动前先考虑道德原则的干预……（原文摘要在此处被截断）

    arXiv:2610.07847v1 Announce Type: new  Abstract: As LLMs increasingly assist in moral reasoning, omission bias, the tendency to prefer inaction even when equivalent framings reverse substantive outcomes, poses a significant risk of skewed decision-making. Yet omission bias remains underexplored in LLM evaluation, with the few existing studies limited in scale and focused largely on utilitarian-deontological conflicts. To address this gap, we introduce OMIT, a benchmark consisting of 218 paired-frame scenarios across 10 conflict types, constructed by leveraging disagreement patterns from an LLM-based, five-perspective philosophical persona panel (utilitarianism, deontology, virtue ethics, care ethics, and contractualism). Evaluating eight LLMs, we find that omission bias is pervasive but inversely correlates with model size within families. We further evaluate four inference-time interventions and find that interventions encouraging models to consider moral principles before committing 
    
[^67]: 通过模块化可执行的开发原语为软件工程构建工程化框架

    Harness Engineering for Software Engineering via Modular Executable Dev-Primitives

    [https://arxiv.org/abs/2610.07832](https://arxiv.org/abs/2610.07832)

    该论文提出Dev-Primitives，一种将代码库工件与常驻LLM配对的模块化可执行抽象，使软件组件从被动工件转变为具备智能体原生接口的主动参与者，从而解决LLM智能体在长程软件工程工作流中反复重建程序状态、上下文爆炸和语义漂移的问题。

    

    配备终端访问能力的大型语言模型（LLMs）在自动化软件工程任务方面已展现出强大的能力。然而，现有智能体在长程工作流中依然十分脆弱：它们必须反复重建分散在源代码文件、配置、测试、依赖项和运行时行为中的程序状态，导致交互历史不断膨胀、上下文爆炸以及语义漂移。大型代码库则进一步增加了识别与任务相关组件的难度。为了应对这些挑战，我们提出了Dev-Primitives（开发原语），这是一种模块化且可执行的抽象，它将代码库组件从被动的软件工件转变为软件工程中的主动参与者。每个Dev-Primitive将一个代码库工件与一个常驻LLM配对，从而赋予该工件一个基于其自身实现和依赖关系的智能体原生接口。

    arXiv:2610.07832v1 Announce Type: cross  Abstract: Large language models (LLMs) equipped with terminal access have demonstrated strong capabilities in automating software engineering tasks. However, existing agents remain brittle on long-horizon workflows, where they must repeatedly reconstruct program state scattered across source files, configurations, tests, dependencies, and runtime behavior, leading to increasingly long interaction histories, context explosion, and semantic drift. Large repositories further complicate the identification of task-relevant components. To address these challenges, we introduce \textbf{Dev-Primitives} (\emph{Development Primitives}), a modular and executable abstraction that transforms repository components from passive software artifacts into active participants in software engineering. Each Dev-Primitive pairs a repository artifact with a resident LLM, which gives the artifact an agent-native interface grounded in its own implementation and dependenc
    
[^68]: 核采样投机解码：超越精确分布的合理性感知验证

    Nucleus Speculative Decoding: Plausibility-Aware Verification Beyond Exact Distribution

    [https://arxiv.org/abs/2610.07822](https://arxiv.org/abs/2610.07822)

    提出核采样投机解码（NSD），通过在标准接受准则之外额外接受属于目标模型核采样集合的草稿词元来放宽验证条件，从而保留更多草稿词元以加速生成，并从理论上证明其单步分布误差恰好由草稿模型在目标核内的超额概率决定。

    

    投机解码通过使用轻量级草稿模型提出多个候选词元，再由目标模型并行验证，从而加速自回归生成。然而，标准的接受准则专注于精确的分布校正，当草稿模型为某词元分配了过多概率时，即使该词元在目标模型下仍然具有很高的合理性，也会被拒绝。这种保守的验证方式限制了每次验证前向传播后所能保留的草稿词元数量。我们提出了核采样投机解码（Nucleus Speculative Decoding, NSD），这是一种将目标模型的合理性纳入投机解码的宽松验证方法。当一个草稿词元满足标准接受准则，或属于目标模型的核采样集合时，NSD 就会接受该词元。我们从理论上刻画了该方法所引入的分布偏差，并证明其单步误差恰好由草稿模型在目标核采样集合内的超额概率所决定。

    arXiv:2610.07822v1 Announce Type: new  Abstract: Speculative decoding accelerates autoregressive generation by using a lightweight draft model to propose multiple tokens that are verified by a target model in parallel. However, the standard acceptance rule focuses on exact distribution correction and rejects tokens that remain highly plausible under the target model when the draft model assigns excess probability. This conservative verification limits the number of draft tokens retained after each verification forward pass. We introduce Nucleus Speculative Decoding (NSD), a relaxed verification method that incorporates target-model plausibility into speculative decoding. NSD accepts a draft token if it satisfies the standard acceptance rule or belongs to the target model's nucleus. We theoretically characterize the distributional deviation introduced by our method and show that the single-step error is exactly determined by the draft model's excess probability within the target nucleus
    
[^69]: αTransfer：面向高效模型合并的系数迁移方法

    $\alpha$Transfer: Coefficient Transfer for Efficient Model Merging

    [https://arxiv.org/abs/2610.07819](https://arxiv.org/abs/2610.07819)

    提出αTransfer方法，利用同一模型家族内不同规模模型在合并系数上性能分布的高度一致性，在小代理模型上搜索最优系数后直接迁移到大模型，实现最高20倍加速和70%内存减少。

    

    模型合并通过参数运算将多个微调后的检查点合并为单一模型，是一种颇具前景的解决方案。然而，寻找最优合并系数需要进行大量搜索，随着模型在规模和数量上的增长，由于高内存需求和搜索空间的组合式膨胀，这一过程变得极其昂贵。我们发现，在同一模型家族内，不同规模的模型在合并系数上表现出高度一致的性能分布。这种分布上的相似性使我们能够提出一种实用的范式，我们称之为αTransfer：先在小型代理模型上搜索最优系数，然后直接将其迁移到更大的目标模型上。我们在多种合并方法、模型家族和任务上验证了αTransfer。实验结果表明，在视觉Transformer上实现了6倍的加速和70%的内存减少，以及20倍的加速。

    arXiv:2610.07819v1 Announce Type: cross  Abstract: Model merging offers a promising solution for combining multiple fine-tuned checkpoints into a single model through parameter arithmetic. However, finding optimal merging coefficients requires an extensive search that becomes prohibitively expensive as models scale in both size and number, due to high memory requirements and combinatorial growth in the search space. We show that, within the same model family, models exhibit highly congruent performance distributions over merging coefficients across different model sizes. This distributional similarity enables a practical paradigm we call \textit{$\alpha$Transfer}: searching for optimal coefficients on a small proxy model, then directly transfer them to larger target models. We verify $\alpha$Transfer across multiple merging methods, model families, and tasks. Experimental results demonstrate a 6$\times$ speedup and 70\% memory reduction on vision transformers, and a 20$\times$ speedup 
    
[^70]: 一步一个脚印：以大语言模型自主性换取流程可预测性

    One Step at a Time: Trading LLM Autonomy for Process Predictability

    [https://arxiv.org/abs/2610.07817](https://arxiv.org/abs/2610.07817)

    该论文提出通过MCP协议逐步向智能体交付流程步骤，以牺牲LLM自主性为代价，从架构上保证流程的事前可预测性，并生成可供下游工具逐步审计和优化的机器可读执行日志。

    

    对运营流程进行自动化的组织所需要的不仅仅是正确的结果：他们还需要预测流程将如何运行、了解实际运行的是哪一个流程，并逐步对其进行检查。当智能体作为执行者时，这种可预测性通常会丢失：规定的流程被写入系统提示词中，而系统最终只返回一个最终答案。我们改为通过模型上下文协议逐步交付流程：服务器每次只释放一个步骤，智能体执行该步骤，并且每个步骤都会返回一个结构化的step_output。这以自主性换取可预测性，由此两个性质通过构造自然成立，且不依赖于执行者：其一，执行路径在运行前就被规定好，因此流程是事先可预测的，而非事后重建的；其二，已完成的步骤记录构成了机器可读的执行日志，下游工具可以逐步对其进行审计和优化。在13个SO上对15,475次试验进行评估……

    arXiv:2610.07817v1 Announce Type: cross  Abstract: Organizations automating operational processes need more than a correct outcome: they need to predict how a process will run, know which one actually ran, and inspect it step by step. When an agent is the executor that predictability is normally lost: the prescribed procedure goes into the system prompt, and only a final answer comes back. We deliver the procedure step by step over the Model Context Protocol (MCP) instead: a server releases one step at a time, the agent executes it, and each step returns a structured step_output. This trades autonomy for predictability, and two properties then follow by construction, independent of the executor. The execution path is prescribed before the run, so the process is predictable in advance rather than reconstructed afterwards; and the completed step records form a machine-readable execution log that downstream tooling can audit and optimize step by step. Evaluating 15,475 trials across 13 SO
    
[^71]: ThinkFuse：面向小型推理模型的轨迹感知测试时融合

    ThinkFuse: Trajectory-Aware Test-Time Fusion for Small Reasoning Models

    [https://arxiv.org/abs/2610.07803](https://arxiv.org/abs/2610.07803)

    ThinkFuse提出了一种无需训练的轨迹感知测试时融合框架，通过对比片段级不确定性变化与轨迹级整体趋势来识别不稳定推理点，并将辅助推理路径融合进主轨迹，从而显著提升小型推理模型在数学和知识密集型推理任务上的可靠性与性能。

    

    小型推理模型通过生成扩展的思维链轨迹，在复杂推理任务上展现出强大的性能，但一旦推理进入错误路径，往往难以恢复。现有的测试时融合方法依赖局部融合信号来决定何时触发融合，这可能会被瞬时的不确定性波动所误导，并可能强化不稳定的推理轨迹。我们提出了ThinkFuse，这是一个无需训练的测试时融合框架，能够选择性地干预不可靠的推理片段。ThinkFuse通过比较片段级别的不确定性变化与轨迹级别的整体不确定性趋势，识别不稳定的推理点，并将辅助推理路径融合到主模型的推理轨迹中。大量实验表明，ThinkFuse在数学推理和知识密集型推理基准上优于基线方法，在不同模型家族组合中均取得一致的提升。

    arXiv:2610.07803v1 Announce Type: new  Abstract: Small reasoning models (SRMs) have shown strong performance on complex reasoning tasks by generating extended chain-of-thought trajectories, but they often fail to recover once their reasoning enters an erroneous path. Existing test-time fusion methods rely on local fusion signals to determine when to trigger fusion, which can be misled by transient uncertainty fluctuations and may reinforce unstable reasoning trajectories. We propose ThinkFuse, a training-free test-time fusion framework that selectively intervenes in unreliable reasoning segments. ThinkFuse compares segment-level uncertainty shifts with trajectory-level uncertainty trends to identify unstable reasoning points and fuse auxiliary reasoning paths into the primary model's trajectory. Extensive experiments demonstrate that ThinkFuse outperforms baselines on mathematical and knowledge-intensive reasoning benchmarks, with consistent gains across model-family combinations, and 
    
[^72]: 多智能体LLM推理中的持久记忆：它花费什么、带来什么、以及何时能被察觉

    Persistent Memory in Multi-Agent LLM Inference: What It Costs, What It Buys, and When You Can Tell

    [https://arxiv.org/abs/2610.07782](https://arxiv.org/abs/2610.07782)

    本文在三层多智能体LLM推理架构中实测发现，上下文分解可将峰值KV缓存从35.5 MiB降至14.3 MiB，而持久记忆层不仅增加0.368 MiB缓存开销，且在单问题基准上未带来任何可检测的准确率提升，这种零结果源于基准测试的结构性特点。

    

    将长上下文推理分解到协作智能体之间，可以限制每次调用的活跃KV缓存而非总证据量，这在KV缓存内存成为瓶颈时尤为重要。许多此类系统会添加一个持久层来存储和回忆推理轨迹，通常通过消融实验报告的准确率提升来验证其效果。我们在一个三层智能体架构上对两者进行了测量。上下文分解确实带来了成效：每个查询的峰值KV工作集为14.3 MiB，而单次传递和检索增强基线分别为35.5和35.3 MiB。持久层则没有带来收益：在八个受控数据集对、每组n=100的实验中，它使峰值缓存增加0.368 MiB [+0.167, +0.590]，且未产生可检测的准确率变化（+0.015，95% CI [-0.011, +0.046]）。我们认为这种零结果是结构性的：单问题基准测试为每个条目提供其独立的证据并独立评分，且正确性要求在条件之间重置已存储的推理轨迹，因此记忆回忆没有任何有价值的信息可供利用

    arXiv:2610.07782v1 Announce Type: new  Abstract: Decomposing long-context inference across cooperating agents bounds the active KV cache per call rather than total evidence, which matters when KV-cache memory binds. Many such systems add a persistent tier storing and recalling reasoning traces, usually validated by an ablation reporting an accuracy gain. We measure both on one three-tier agent architecture. Decomposition delivers: peak KV working set of 14.3 MiB per query against 35.5 and 35.3 MiB for single-pass and retrieval-augmented baselines. The persistent tier does not: across eight controlled dataset pairs at n=100 per arm it costs +0.368 MiB [+0.167, +0.590] of peak cache and produces no detectable accuracy change (+0.015, 95% CI [-0.011, +0.046]). We argue the null is structural: single-question benchmarks supply each item with its own evidence and score it independently, and correctness requires resetting stored traces between conditions, so recall has nothing informative to
    
[^73]: 量化对工具故障恢复的影响因提示词和评估设计而异

    Quantization Effects on Tool-Failure Recovery Vary Across Prompts and Evaluation Designs

    [https://arxiv.org/abs/2610.07781](https://arxiv.org/abs/2610.07781)

    该研究发现8比特与4比特量化对语言模型智能体工具故障恢复能力的影响并不稳定，比较结论会随提示词和评估目标（如评分任务的选择）而改变方向甚至反转，表明量化效果的结论高度依赖于评估设计。

    

    训练后量化降低了部署语言模型智能体的成本，但其对从临时性工具故障中恢复的影响可能取决于评估恢复能力的方式。我们在二十个确定性工具使用任务和五个提示词上，比较了Llama-3.1-8B-Instruct和Qwen2.5-7B-Instruct的8比特与4比特变体。8比特与4比特在恢复能力上的比较结果会随提示词和评估目标而改变方向。在同一提示词下两个变体均能无故障完成的任务上，Llama的差异范围为0至+20.2个百分点，Qwen的差异范围为-50.0至+35.0个百分点。全流程点估计在所有五个提示词下都偏向8比特的Llama，而Qwen的比较结果则随提示词改变方向。评估目标的选择也可能使结论反转。在某一提示词下的Llama上，仅在每个变体各自无故障通过的任务上评分时，4比特领先17.5个百分点；而对两个变体在相同任务上评分时则没有差异。

    arXiv:2610.07781v1 Announce Type: new  Abstract: Post-training quantization reduces the cost of deploying language-model agents, but its effect on recovery from temporary tool failures can depend on how recovery is evaluated. We compare 8-bit and 4-bit variants of Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct on twenty deterministic tool-use tasks and five prompts. The 8-bit-4-bit recovery comparison changes direction across prompts and evaluation targets. On tasks that both variants complete without faults under the same prompt, the difference ranges from 0 to +20.2 percentage points for Llama and from -50.0 to +35.0 points for Qwen. Full-pipeline point estimates favor 8-bit Llama under all five prompts, whereas the Qwen comparison changes direction across prompts. The evaluation target can also reverse the result. For Llama under one prompt, scoring each variant only on its own clean-passing tasks favors 4-bit by 17.5 points; scoring the same tasks for both variants gives no differen
    
[^74]: APEX：更聪明地推测，而非更深地推测

    APEX: Speculate smarter, not deeper

    [https://arxiv.org/abs/2610.07780](https://arxiv.org/abs/2610.07780)

    APEX是一个学习型控制器，通过请求级专家选择和块级草稿深度自适应来优化推测解码，用更聪明的推测取代更深的推测，从而降低大语言模型推理延迟并减少计算浪费。

    

    推测解码通过在目标模型验证之前起草多个token来降低大语言模型的推理延迟，但其有效性同时取决于提议机制和草稿深度。固定配置无法响应生成过程中可预测性、重复性和接受率的变化，因此更深的草稿可能会增加计算浪费，却无法带来成比例的加速。我们提出了APEX，这是一个通过请求级专家选择和块级深度自适应来平衡解码速度与草稿token浪费的学习型控制器。APEX-Router为每个请求在EAGLE-3、n-gram和草稿模型推测之间进行选择，而APEX-Depth则利用因果解码信号和最近的验证器反馈，在每个验证块中调整草稿长度。APEX将接受的草稿长度建模为截断生存反馈，学习按位置拒绝风险、块执行成本，以及平衡吞吐率的动作效用函数。

    arXiv:2610.07780v1 Announce Type: new  Abstract: Speculative decoding reduces large language model inference latency by drafting multiple tokens before target-model verification, but its effectiveness depends on both the proposal mechanism and draft depth. Fixed configurations cannot respond to changes in predictability, repetition, and acceptance during generation, so deeper drafting can increase wasted computation without proportional speedup. We introduce APEX, a learned controller that balances decoding speed and draft-token waste through request-level expert selection and block-level depth adaptation. APEX-Router selects among EAGLE-3, n-gram, and draft-model speculation for each request, while APEX-Depth adjusts draft length at each verification block using causal decoding signals and recent verifier feedback. APEX models accepted draft length as censored survival feedback, learning position-wise rejection hazards, block execution costs, and an action utility that balances throug
    
[^75]: 读取而非操纵：利用路由器Logits实现MoE视觉语言模型的多模态安全

    Reading, Not Manipulating: Leveraging Router Logits for Multimodal Safety in MoE Vision-Language Models

    [https://arxiv.org/abs/2610.07774](https://arxiv.org/abs/2610.07774)

    该论文发现MoE视觉语言模型的路由器logits可作为多模态输入安全性的高预测性诊断信号，据此提出一种轻量级安全检测器，无需操纵模型内部状态即可实现多模态安全检测，避免了传统干预方法带来的安全-实用性权衡。

    

    视觉语言模型（VLM）面临组合性安全风险，即有害意图可能从视觉与文本输入的交互中产生。随着混合专家模型视觉语言模型日益普及，近期研究探索了多种安全干预手段，包括提示工程、监督微调以及基于路由的专家引导。然而，这些方法在不同模型和评估分布上的改进效果并不一致，且对模型行为或内部状态进行干预会因过度拒答而引入安全性与实用性之间的权衡。本研究不通过操纵内部状态来引导模型行为，而是探讨路由状态能否作为多模态安全的诊断信号。研究发现，路由器logits确实能为判断多模态输入是否安全提供高度预测性的信号。基于这一观察，作者提出了一种轻量级的路由器logit安全检测器，在推理过程中读取路由信号（摘要在此处截断，后续内容不完整）。

    arXiv:2610.07774v1 Announce Type: new  Abstract: Vision-language models (VLMs) face compositional safety risks where harmful intent emerges from the interaction between visual and textual inputs. As mixture-of-experts (MoE) VLMs become increasingly common, recent work has explored various safety interventions, including prompting, supervised fine-tuning, and routing-based expert steering. However, these methods show inconsistent improvements across models and evaluation distributions, and the intervention into model behavior or internal states introduce safety-utility tradeoffs by over-refusal. Rather than manipulating internal states to steer model behavior, we instead ask whether routing states can serve as diagnostic signals for multimodal safety. We find that router logits indeed provide highly predictive signals of whether a multimodal input is safe or not. Motivated by this observation, we introduce a lightweight router-logit safety detector that reads out routing signals during 
    
[^76]: TRACE：面向MoE语言模型FP4强化学习的Rollout引导量化感知训练

    TRACE: Rollout-Guided Quantization-Aware Training for FP4 Reinforcement Learning of MoE Language Models

    [https://arxiv.org/abs/2610.07767](https://arxiv.org/abs/2610.07767)

    TRACE是一个面向MoE语言模型强化学习训练的FP4量化框架，通过rollout引导的量化感知训练，利用rollout侧量化结果指导训练侧FP4舍入决策，直接缩小训练路径与rollout路径两条量化执行路径之间的差异。

    

    对大语言模型（LLM）进行后训练的强化学习（RL）在rollout生成过程中会产生大量的计算和内存开销，这促使人们采用低精度rollout来实现高效的RL训练。然而，现有的FP4 RL方法存在一个关键局限：它们主要在训练路径和rollout路径上分别独立地优化量化精度，而不是直接减少两条量化执行路径之间的差异。在本工作中，我们提出了TRACE（Train-Rollout Quantization Alignment via Compact GuidancE，通过紧凑引导实现训练-Rollout量化对齐），这是一个面向混合专家（MoE）语言模型RL训练的FP4量化框架，解决了现有FP4 RL方法的局限性。TRACE引入了rollout引导的量化感知训练，利用rollout侧的量化结果来指导训练侧的FP4舍入决策，从而直接减少训练与rollout之间的差异。此外，TRACE采用了高效的量化……

    arXiv:2610.07767v1 Announce Type: cross  Abstract: Reinforcement learning (RL) for post-training large language models (LLMs) incurs substantial computation and memory overhead during rollout generation, which motivates low-precision rollout for efficient RL training. However, existing FP4 RL methods suffer from a key limitation: they primarily optimize quantization accuracy on the training and rollout paths independently rather than directly reducing the discrepancy between the two quantized execution paths. In this work, we propose TRACE (Train-Rollout Quantization Alignment via Compact GuidancE), an FP4 quantization framework for RL training of Mixture-of-Experts (MoE) language models that addresses the limitation of existing FP4 RL methods. TRACE incorporates rollout-guided quantization-aware training that uses rollout-side quantization outcomes to guide training-side FP4 rounding decisions, directly reducing train-rollout discrepancy. Moreover, TRACE adopts an efficient quantizati
    
[^77]: 没有Transformer能胜过六个协变量：基于童年作文对抑郁症状的长期预测

    No Transformer Beats Six Covariates: Long-Horizon Prediction of Depressive Symptoms from Childhood Essays

    [https://arxiv.org/abs/2610.07764](https://arxiv.org/abs/2610.07764)

    该研究发现，在利用11岁儿童作文预测其23岁抑郁症状的长期任务中，基于六个童年协变量的简单逻辑回归（AUC-ROC 0.737）显著优于所有文本模型，包括微调Transformer、词袋模型、冻结嵌入和零样本大语言模型（最佳仅0.670）。

    

    自然语言处理（NLP）模型能够检测在症状测量时间附近所写文本中的抑郁相关语言，但预训练Transformer能否从十二年前所写的文本中预测抑郁症状，在很大程度上尚未得到验证。在英国出生队列研究“全国儿童发展研究”中，我们利用同一批人在11岁时写的作文来预测其23岁时可能出现的抑郁症状。我们的基线模型——一个基于六个童年期协变量的逻辑回归——优于所有仅使用作文文本的模型：七个微调的Transformer、一个词袋模型、冻结嵌入以及四个零样本大语言模型。基线模型的受试者工作特征曲线下面积（AUC-ROC）为0.737，而在主随机种子下最佳Transformer仅为0.670，且任何附加的文本得分都未能显著提升基线的AUC-ROC。五个领域预训练的Transformer模型中，没有一个能显著胜过其通用领域对照模型。

    arXiv:2610.07764v1 Announce Type: cross  Abstract: Natural language processing (NLP) models can detect depression-related language in text written near the time symptoms are measured, but whether pretrained transformers can predict depressive symptoms from text written twelve years earlier is largely untested. In the National Child Development Study, a British birth cohort, we predict probable depressive symptoms at age 23 from essays the same people wrote at age 11. Our baseline, a logistic regression on six childhood covariates, outperforms every text model that sees only the essay: seven fine-tuned transformers, a bag-of-words model, frozen embeddings and four zero-shot large language models. Its area under the receiver operating characteristic curve (AUC-ROC) is 0.737 against 0.670 for the best transformer on the primary seed, and no added text score detectably raises the baseline's AUC-ROC. None of the five domain-pretrained transformers detectably beats its general-domain control
    
[^78]: 从证据到行动：工具使用型智能体如何失败

    From Evidence to Action: How Tool-Using Agents Fail

    [https://arxiv.org/abs/2610.07753](https://arxiv.org/abs/2610.07753)

    提出包含656个案例的SafeActBench基准，系统揭示工具使用型智能体在“证据到行动”链条中的失败模式——失败往往在执行前就因调查不完整或过早行动而出现，多动作工作流还会暴露未解决的先决条件和不完整执行问题。

    

    使用工具的智能体会对外部状态做出重要改变，然而正确的结果并不能保证其行动是建立在事先确立的证据之上的。我们研究了当智能体从决定是否采取行动，到执行单个动作乃至相互依赖的工作流时，这条从证据到行动的链条在何处断裂。在十种模型-框架配置中，强大的静态动作评估能力可能与弱得多的交互式执行能力并存。失败往往在执行之前就已开始：智能体在调查不完整的情况下停止，或在所需证据尚未确立时就贸然行动。一旦所需证据已获取，单动作执行通常是可靠的，而多动作工作流则会额外暴露出未解决的先决条件和不完整的执行问题。为支持这一分析，我们推出了SafeActBench，其中包含跨越六个操作领域的656个案例和五种协议，涵盖从静态动作判断、经调查后的不行动，到单动作执行等递进场景。

    arXiv:2610.07753v1 Announce Type: cross  Abstract: Tool-using agents make consequential changes to external state, yet correct outcomes do not guarantee that their actions were supported by evidence established beforehand. We study where this evidence-to-action chain breaks as agents move from deciding whether to act to executing single actions and dependent workflows. Across ten model-harness configurations, strong static action assessment can coexist with much weaker interactive execution. Failures often begin before execution: agents stop with incomplete investigation or act before required evidence is established. Once required evidence is obtained, single-action execution is usually reliable, while multi-action workflows additionally expose unresolved prerequisites and incomplete execution. For this analysis, we introduce SafeActBench, comprising 656 cases across six operational domains and five protocols that progress from static action judgment and investigated non-action to sin
    
[^79]: 在嵌入空间中通过强化学习学习检索

    Learning to Retrieve via Reinforcement Learning in Embedding Space

    [https://arxiv.org/abs/2610.07731](https://arxiv.org/abs/2610.07731)

    该论文提出RELER强化学习框架，通过从vMF分布采样嵌入动作、结合RLOO基线的REINFORCE算法以及减少采样噪声的条件均值投影（CMP）技术，使现有嵌入模型能够直接在嵌入空间中学习检索并对齐任务特定的奖励。

    

    密集检索模型通常使用对比目标进行训练，这类目标虽然能学习有效的表示，但无法直接优化检索指标或下游任务性能。为了解决这一问题，我们提出了RELER（面向检索的强化学习），这是一个强化学习框架，能够使现有的嵌入模型直接在嵌入空间中学习检索，并与任务特定的奖励对齐。我们通过以下方式训练RELER：从以归一化编码器输出为中心的von Mises-Fisher（vMF）分布中采样单位长度的查询和文档嵌入动作，将由此产生的检索或下游结果评分作为奖励，并使用留一法基线（RLOO）通过REINFORCE算法更新编码器。由于在高维嵌入空间中进行探索容易受到采样噪声的影响，我们进一步提出了条件均值投影（CMP），将每个采样的嵌入投影到低维子空间上……

    arXiv:2610.07731v1 Announce Type: cross  Abstract: Dense retrieval models are typically trained with contrastive objectives that learn effective representations but do not directly optimize retrieval metrics or downstream task performance. To address this problem, we introduce RELER (REinforcement LEarning for Retrieval), a reinforcement learning framework that enables existing embedding models to learn to retrieve directly in embedding space and align to task-specific rewards. We train RELER by sampling unit-length query and document embedding actions from von Mises-Fisher (vMF) distributions centered on normalized encoder outputs, scoring the resulting retrieval or downstream outcomes as rewards, and updating the encoder with REINFORCE using a leave-one-out baseline (RLOO). As exploration in the high-dimensional embedding space is prone to sampling noise, we further propose conditional-mean projection (CMP), which projects each sampled embedding onto the low-dimensional subspace span
    
[^80]: SanSi：一种用于系统1.5思维的循环式类型化决策模型

    SanSi: A Looped Typed Decision Model for System 1.5 Thinking

    [https://arxiv.org/abs/2610.07730](https://arxiv.org/abs/2610.07730)

    SanSi提出“系统1.5思维”——通过多次循环复用模型层在不生成文本的情况下修正隐藏状态，将预训练循环语言模型转化为类型化决策模型，在59个数据源的10,027个测试决策上达到72.0%准确率，比同结构非循环模型高出13.5个百分点。

    

    类型化决策模型在不生成文本的情况下回答一个预先声明的问题：决策头在单次前向传播中为每个声明的选项返回一个概率。单次前向传播快速而直观，属于系统1思维。我们研究了介于单次传播与生成式推理之间的方法：循环，即在输出一次类型化读数之前，将相同的层递归地应用多次。每一次循环都让模型在提交答案之前修正其隐藏状态，而无需生成任何词元；我们将其称为系统1.5思维。我们提出了SanSi，它将一个预训练的循环语言模型转化为类型化决策模型。选项概率在每次循环后被读取，且每次循环都使用适当的评分规则进行训练，因此单个模型可以在一次运行中服务于从一次循环到八次循环的任意计算预算。在来自59个数据源的10,027个测试决策上，SanSi达到了72.0%的准确率，比用相同结构训练的非循环模型高出13.5个百分点。

    arXiv:2610.07730v1 Announce Type: cross  Abstract: Typed decision models answer a declared question without generating text: a decision head returns a probability for each of the declared options in a single forward pass. A single pass is fast, intuitive System 1 thinking. We study what lies between one pass and generated reasoning: looping, in which the same layers are recursively applied several times before one typed readout. Each loop lets the model revise its hidden state before it commits to an answer, without generating a token; we call this System 1.5 thinking. We propose SanSi, which turns a pre-trained looped language model into a typed decision model. The option probabilities are read after every loop, and every loop is trained with a proper scoring rule, so that one model serves every budget from one loop to eight in a single run. On 10,027 test decisions from 59 sources, SanSi reaches 72.0% accuracy: 13.5 points above a non-looped model of the same shape trained with the s
    
[^81]: 转向会破坏你的模型吗？面向大语言模型转向方法的多维度评估套件

    Does Steering Break Your Model? A Multi-Dimensional Evaluation Suite for LLM Steering Methods

    [https://arxiv.org/abs/2610.07722](https://arxiv.org/abs/2610.07722)

    本文提出SteerScope，一个包含15项指标的双轴多维评估套件，系统刻画LLM激活转向方法在有效性与副作用（语言质量、任务能力、安全可靠性）之间的权衡，并进一步评估其泛化能力与数据依赖性。

    

    激活转向为控制大语言模型（LLM）的行为提供了一种轻量且灵活的方式。然而，有效的转向不仅仅是诱导出预期的行为：它还应当限制意外的变化，并在不同输入和训练数据上保持鲁棒性。现有的评估仅零散地涵盖这些维度。因此，有效性与副作用之间的权衡尚未得到系统性的刻画。我们提出了SteerScope，一个双轴、多维度的评估套件，通过15项指标共同刻画转向结果与方法特性。我们在语言质量、任务能力以及安全性和可靠性方面对目标有效性和副作用进行评分，并进一步通过样本效率和样本敏感性等转向专属指标评估泛化能力与数据依赖性。我们并非在单一操作点上比较各方法，而是刻画有效性与副作用之间的权衡……

    arXiv:2610.07722v1 Announce Type: new  Abstract: Activation steering provides a lightweight and flexible way to control large language model (LLM) behavior. However, effective steering requires more than inducing the intended behavior: it should also limit unintended changes and remain robust across inputs and training data. Existing evaluations cover these dimensions only in fragments. As a result, the trade-offs between efficacy and side effects have not been systematically characterized. We introduce SteerScope, a two-axis, multi-dimensional evaluation suite that jointly characterizes steering outcomes and method properties through 15 metrics. We score target efficacy and side effects on language quality, task capabilities, and safety and reliability, and further assess generalization and data dependence through steering-specific metrics for sample efficiency and sample sensitivity. Rather than comparing methods at a single operating point, we characterize the trade-offs between eff
    
[^82]: 仅预填充决策模型中的读出稳定性：零标签预测与推理时算力分配

    Readout Stability in Prefill-Only Decision Models:Zero-Label Prediction and Inference-Time Compute Allocation

    [https://arxiv.org/abs/2610.07716](https://arxiv.org/abs/2610.07716)

    论文发现仅预填充决策模型具有可检验的“读出稳定性”：仅改变候选菜单时，干预后的准确率可由缓存的首次前向概率通过零标签、无需第二次前向传播的估计器预测（误差在4.2个百分点以内），且该性质存在于排序而非概率之中。

    

    受Jev模型启发的仅预填充（prefill-only）决策模型在单次前向传播中对候选菜单中的每个候选项进行打分且从不进行解码，这使得单次调用比同规模的生成式语言模型便宜一到两个数量级。我们证明这种读出结构具有一个可检验的性质：当干预仅改变候选菜单而输入文本保持不变时，干预后的准确率已经由缓存的首次前向分布所决定。该估计器将首次前向概率限制在菜单内、重新归一化并读取argmax；它不使用任何标签，也无需第二次前向传播。在七个模型家族、十个数据集和两类任务上，仅菜单干预的结果可被预测到4.2个百分点以内，且对其中一个家族的预测是精确的。而同一估计器的概率级变体误差达21.0个百分点，因此该性质存在于排序而非概率之中，并且并非……（原文摘要在此处截断）

    arXiv:2610.07716v1 Announce Type: new  Abstract: Prefill-only decision models inspired by the Jev model score every candidate in a menu during a single forward pass and never decode, which makes one call one to two orders of magnitude cheaper than a same-scale generative language model. We show that this read-out structure comes with a testable property. When an intervention changes only the candidate menu and leaves the input text fixed, the post-intervention accuracy is already determined by the cached first-pass distribution. The estimator restricts the pass-1 probabilities to the menu, renormalizes, and reads off the argmax; it uses no labels and no second forward pass. Across seven model families, ten datasets and two task types, menu-only interventions are predicted to within 4.2 points, and for one family the prediction is exact. A probability-level variant of the same estimator errs by 21.0 points, so the property lives in the ranking rather than in the probabilities and is not
    
[^83]: 当旧事实回归时：重读、回退与时序记忆的局限

    When Old Facts Return: Re-Reads, Reverts, and the Limits of Temporal Memory

    [https://arxiv.org/abs/2610.07715](https://arxiv.org/abs/2610.07715)

    该论文揭示了时序记忆系统的一个关键歧义——旧信息的重读与真正的回退会产生相同的观测序列却需要相反的答案——并提出一个拒绝重新激活已淘汰值的守卫机制，该机制能有效防御重读攻击（准确率从10.8%恢复至97.7%），但需要额外的变更溯源信息才能区分合法回退。

    

    记忆系统可能会淘汰一个过时的值，但随后仅仅因为同样的旧语句再次出现就将其恢复。对旧来源的逐字重读与真正的回退可以产生相同的观测值序列，却需要截然相反的当前答案。我们在从软件修复中提取的130个由抽取器选定的原子转换上研究这种歧义性。在普通转换条件下，基于身份的时序记忆在字面过时值代理指标下达到98.5%的模型评判准确率，且观测错误为零。而在追加一段旧语句的逐字重读后，准确率降至10.8%，过时值率升至88.5%。一个拒绝重新激活先前被淘汰值的守卫机制，在此构建的重读条件下将准确率恢复至97.7%，并将过时值率降至0.8%。然而，在没有额外的变更溯源信息的情况下，该守卫无法同时识别合法的回退操作。两项辅助研究考察了将已淘汰历史暴露给……（摘要在此处截断）

    arXiv:2610.07715v1 Announce Type: cross  Abstract: A memory system can retire an obsolete value and later restore it merely because the same old statement appears again. A re-read of an old source and a genuine revert can produce the same observed sequence of values while requiring opposite current answers. We study this ambiguity on 130 extractor-selected atomic transitions derived from software fixes. In the ordinary transition condition, identity-based temporal memory reaches 98.5% model-judged accuracy with zero observed errors under a literal stale-value proxy. Appending a verbatim re-read of the old statement reduces accuracy to 10.8% and raises the stale-value rate to 88.5%. A guard that refuses to reactivate a previously retired value restores accuracy to 97.7% and reduces that rate to 0.8% in this constructed re-read condition. The guard cannot also recognize a legitimate revert without additional change provenance. Two supporting studies examine exposing retired history to th
    
[^84]: 在行为操纵下通过击键检测大语言模型辅助的越南语写作

    Detecting LLM-Assisted Vietnamese Writing via Keystrokes under Behavioral Manipulation

    [https://arxiv.org/abs/2610.07700](https://arxiv.org/abs/2610.07700)

    该研究构建了越南语击键数据集并提出了基于行为的威胁模型，发现序列模型在检测LLM辅助写作时优于基于特征的方法，且击键信号包含判别性信息，但在用户故意操纵打字行为时检测效果并不均匀。

    

    我们研究了击键动力学在检测大语言模型（LLM）辅助写作方面的鲁棒性。我们引入了一个捕捉真实写作模式的越南语击键数据集，包括真实创作、转录和改写。我们还定义了一个基于行为的威胁模型，其中用户会故意改变打字模式。为了实现该威胁模型，我们创建了旨在规避基于击键检测的行为操纵数据变体。我们在用户无关和上下文无关的设置下评估了四种击键建模方法：时间和节奏表示，以及使用一维卷积神经网络（1D-CNN）和TypeNet建模的序列表示。结果表明，序列模型在大多数情况下优于基于特征的方法，且击键信号编码了关于写作过程的判别性信息。然而，检测效果并不均匀。

    arXiv:2610.07700v1 Announce Type: new  Abstract: We study the robustness of keystroke dynamics for detecting large language model (LLM)-assisted writing. We introduce a Vietnamese keystroke dataset capturing realistic writing modes, including bona fide composition, transcription, and paraphrasing. We also define a behaviorally grounded threat model in which users deliberately alter typing patterns. To implement the threat model, we create behaviorally manipulated variants of the data designed to evade keystroke-based detection. We evaluate four keystroke modeling approaches: temporal and rhythmic representations, and sequential representations modeled with a one-dimensional convolutional neural network (1D-CNN) and TypeNet, under user-independent and context-independent settings. The results show that sequential models outperform feature-based approaches in most cases and that keystroke signals encode discriminative information about the writing process. However, detection is not unifo
    
[^85]: 通过对抗性强化学习改进论辩挖掘的合成数据生成

    Improving Synthetic Data Generation for Argument Mining via Adversarial Reinforcement Learning

    [https://arxiv.org/abs/2610.07699](https://arxiv.org/abs/2610.07699)

    提出一种对抗性强化学习数据合成框架，通过生成器与判别器的对抗循环联合优化，同时提升论辩挖掘合成数据的结构准确性与多样性。

    

    论辩挖掘从根本上受到高质量结构标注数据集稀缺的限制。虽然大语言模型（LLMs）在合成数据生成方面已展现出潜力，但生成结构准确且足够多样的合成论辩挖掘数据仍然是一个具有挑战性的问题。为了解决这一问题，我们从一个新的视角重新审视论辩挖掘的合成数据生成，并提出了一种新颖的用于数据合成的对抗性强化学习框架。该框架在对抗循环中联合优化生成器与判别器，其中生成器生成结构化的论辩挖掘实例，判别器通过区分真实数据与合成候选数据来提供学习信号。这使得生成器能够通过对抗反馈逐步提升所生成论辩数据的结构准确性，同时保持多样性。大量实验表明，所提出的框架（摘要内容在此处截断）

    arXiv:2610.07699v1 Announce Type: cross  Abstract: Argument Mining (AM) is fundamentally constrained by the scarcity of high-quality structure-annotated datasets. While LLMs have shown promise in synthetic data generation, producing synthetic AM data that is both structurally accurate and sufficiently diverse remains a challenging problem. To address this problem, we revisit synthetic data generation for AM from a new perspective and propose a novel adversarial reinforcement learning framework for data synthesis. The proposed framework jointly optimizes the generator and the discriminator in an adversarial loop, in which the generator produces structured AM instances, and the discriminator provides learning signals by distinguishing real data from synthetic candidates. This enables the generator to progressively improve both the structural accuracy of generated argument data while maintaining diversity through adversarial feedback. Extensive experiments demonstrate that the proposed fr
    
[^86]: DLoop：循环式推测解码

    DLoop: Looped Speculative Decoding

    [https://arxiv.org/abs/2610.07659](https://arxiv.org/abs/2610.07659)

    DLoop提出一种循环式推测解码方法，在验证前自适应地执行多个起草阶段，从而减少目标模型不必要的前向验证，提升大语言模型推测解码的加速效果。

    

    推测解码可加速大语言模型的自回归生成。在每个起草阶段，轻量级的草稿模型提出若干token，随后由目标模型进行验证。我们发现，随着草稿模型能力不断增强，目标模型经常接受起草阶段产生的所有token。然而每次起草阶段之后仍会进行一次验证，即使起草本可继续进行，这导致目标模型执行了不必要的前向传播。自适应草稿长度方法可在解码过程中决定验证前生成多少草稿token，但它们只能为自回归草稿模型提升加速效果。对于并行草稿模型，进一步起草还需要目标模型为尚未验证的草稿token提供隐藏状态。我们提出了DLoop，一种循环式的推测解码方法，可在验证之前自适应地执行多个起草阶段。DLoop在草稿模型保持…

    arXiv:2610.07659v1 Announce Type: new  Abstract: Speculative decoding accelerates autoregressive generation in large language models. In each drafting stage, a lightweight draft model proposes tokens that the target model subsequently verifies. With increasingly capable draft models, we find that the target model frequently accepts all tokens produced in a drafting stage. A verification nevertheless follows each drafting stage, resulting in unnecessary target-model forward passes even when drafting could have continued. Adaptive draft length methods decide during decoding how many draft tokens precede a verification, but they raise the speedup only for autoregressive draft models. For a parallel draft model, drafting further requires target-model hidden states for draft tokens that have not been verified. We propose DLoop, a looped form of speculative decoding that adaptively performs multiple drafting stages before verification. DLoop continues drafting while the draft model remains c
    
[^87]: 规则止于何处，裁判始于何处：度量多智能体系统安全中的判断边界

    Where Rules End and Judges Begin: Measuring the Judgment Boundary in Multi-Agent Systems Security

    [https://arxiv.org/abs/2610.07657](https://arxiv.org/abs/2610.07657)

    该研究提出DEFER1防御框架，通过28项确定性检查级联与四位裁判评审团协同工作，将多智能体系统的攻击成功率从约30%降至约3%，并实证划定了规则可处置与需裁判判断的安全边界。

    

    基于大语言模型的多智能体系统（MAS）会调用工具、共享内存并委派任务，因而经常遭遇对抗性内容。当前针对MAS的防御通常被孤立评估，一次只关注一种攻击类型，这可能导致代价高昂且难以审计的后果。本研究将防御归纳为五项原则，并将其实现为DEFER1（确定性优先执行与残余判断），其中包含一个由28项检查组成的级联，拦截其可拦截的内容，并将其余内容提交给由四位裁判组成的评审团。在跨四个领域的独立测试中，攻击成功率从约30.0%降至约3.0%，其中78%被拦截的攻击由确定性检查处理。在安全运营领域，仅四分之一的提议到达裁判环节，这表明规则为违反明确策略的攻击提供了安全保障，而裁判则负责处理那些仅仅歪曲意图的攻击。两个系统都存在弱点，例如……（摘要在此截断）

    arXiv:2610.07657v1 Announce Type: new  Abstract: LLM-based multi-agent systems (MAS) engage tools, share memory, and delegate tasks, often encountering adversarial content. Current defenses for MAS are typically evaluated in isolation, focusing on one attack type at a time, which can lead to costly and hard-to-audit outcomes. This study organizes defenses into five principles, implementing them as DEFER1 (DEterministic-First Enforcement with Residual judgment), which includes a cascade of 28 checks that blocks what it can and refers the rest to a panel of four judges. In independent testing across four domains, attack success rates drop from about 30.0% to approximately 3.0%, with 78% of blocked attacks handled by deterministic checks. Only a quarter of proposals reach the judges in the security-operations domain, illustrating that the rules provide security for attacks violating clear policies, while judges manage those that only misrepresent intent. Both systems have weaknesses, such
    
[^88]: 响亮而清晰：动态激活引导提升嘈杂环境下的语音可懂度

    Loud and Clear: Dynamic Activation Steering for Improving Speech Intelligibility in Noisy Environments

    [https://arxiv.org/abs/2610.07647](https://arxiv.org/abs/2610.07647)

    提出一种无需重新训练的提示相对激活引导机制，可动态控制预训练TTS模型模拟人类隆巴效应，在噪声环境下将词错误率降低7-22%的同时保持89-95%的说话者相似度。

    

    语音在嘈杂环境中会变得难以听清，而人类会自然地调整自己的嗓音来加以补偿。受此行为启发，我们研究了能否通过激活引导让文本转语音（TTS）模型生成更易听清的语音，且无需重新训练。我们关注隆巴效应的两个特征：增强发声力度和过度清晰发音。我们提出了一种相对于提示词的引导机制，该机制可防止引导效应在生成过程中不断累积，同时允许动态调整其强度。在已见与未见说话者以及多种语言上，我们的方法在隆巴相关声学特征上产生了系统性变化，保持了说话者相似度（89-95%），并在 1 dB 信噪比下将背景噪声中的词错误率降低了 7-22%。这些结果表明，预训练的 TTS 模型可以被动态控制以生成更易听清的语音，而无需重新训练。

    arXiv:2610.07647v1 Announce Type: cross  Abstract: Speech becomes less intelligible in noisy environments, and humans naturally adapt their voice to compensate. Inspired by this behavior, we investigate whether a text-to-speech (TTS) model can be guided to produce more intelligible speech using activation steering, without retraining. We focus on two characteristics of the Lombard effect: increased vocal effort and hyper-articulation. We introduce a prompt-relative steering mechanism that prevents steering effects from accumulating during generation while allowing their strength to be adjusted dynamically. Across seen and unseen speakers and multiple languages, our method produces systematic changes in Lombard-related acoustic features, preserves speaker similarity (89-95%), and reduces WER under background noise by 7-22% at 1 dB SNR. These results show that pretrained TTS models can be dynamically controlled to generate more intelligible speech without retraining.
    
[^89]: 用于KV缓存驱逐的蒙特卡洛估计

    Monte Carlo Estimation for KV Cache Eviction

    [https://arxiv.org/abs/2610.07643](https://arxiv.org/abs/2610.07643)

    提出免训练方法LORE-KV，通过蒙特卡洛采样冻结模型的短自回归延续并以响应侧查询状态估计提示token效用，将KV缓存驱逐从“回顾过去”转变为“预测未来”，在回答时保留真正重要的记忆。

    

    大多数KV缓存驱逐方法实际上在问：哪些内存在阅读提示时显得重要？我们转而问：哪些内存在回答时会有用？由于在驱逐时无法获得解码查询，先前的面向未来的方法依赖于伪响应或合成的未来查询估计。我们将固定预算的面向未来的驱逐问题转化为对合理的模型条件查询轨迹的分布估计，并引入LORE-KV（基于可靠性加权集成的前瞻输出扰动KV缓存方法），这是一种免训练方法，它从冻结的目标模型中采样简短的自回归延续，并利用其响应侧的查询状态来估计提示token的效用。token通过投影的留一法注意力输出删除代价进行评分，并在采样的未来之间进行聚合（可选地带有轨迹加权）。临时延续在最终解码之前被丢弃，无需

    arXiv:2610.07643v1 Announce Type: cross  Abstract: Most KV-cache eviction methods ask, in effect, which memory appeared important while reading the prompt? We instead ask, which memory will matter while answering? Since decoding queries are unavailable at eviction time, prior future-aware methods rely on pseudo-responses or synthetic future-query estimates. We cast fixed-budget future-aware eviction as distributional estimation over plausible model-conditional query trajectories and introduce LORE-KV (Lookahead Output-perturbation with Reliability-weighted Ensembles for Key-Value caches), a training-free method that samples short autoregressive continuations from the frozen target model and uses their response-side query states to estimate prompt-token utility. Tokens are scored by projected leave-one-out attention-output deletion cost and aggregated across sampled futures with optional trajectory weighting. The temporary continuations are discarded before final decoding, requiring no 
    
[^90]: 一种利用辅助词重音建模与损失优化的新型句子重音检测框架

    A Novel Sentence Stress Detection Framework Leveraging Auxiliary Word-Stress Modeling and Loss Optimization

    [https://arxiv.org/abs/2610.07626](https://arxiv.org/abs/2610.07626)

    该论文提出了一种通过辅助词重音建模和词跨度重音正则化器将句子重音检测与词重音检测联合训练的新型框架，在 TinyStress-15K 基准上取得了最佳的句子重音检测性能。

    

    韵律重音是自动发音评估（APA）的一个关键方面，涵盖句子重音检测（SSD）和词重音检测（WSD）两部分。SSD 旨在突出构成话语意义的语义显著词，而 WSD 则识别每个词内的主要重读音节以确保词汇清晰度。然而，大多数先前的工作将 SSD 和 WSD 视为相互独立的任务，忽视了二者对音高、时长和强度等韵律线索的共同依赖。为填补这一空白，我们提出了一种有效的 SSD 方法，通过一种新颖的建模范式将 SSD 与辅助 WSD 相结合。此外，我们引入了词跨度重音正则化器（WSR），将 token 级别的 SSD 概率集中于每个重读词的跨度内。在 TinyStress-15K 基准上的实验表明，所提出的方法优于强大的基线模型，其中完整配置取得了最佳的 SSD 结果。

    arXiv:2610.07626v1 Announce Type: cross  Abstract: Prosodic stress is a crucial aspect of automatic pronunciation assessment (APA), encompassing both sentence stress detection (SSD) and word stress detection (WSD). SSD highlights semantically salient words that shape discourse meaning, while WSD identifies the primary stressed syllable within each word to ensure lexical clarity. However, most prior work treats SSD and WSD as independent tasks, overlooking their shared reliance on prosodic cues such as pitch, duration, and intensity. To address this gap, we propose an effective SSD approach combining SSD with auxiliary WSD via a novel modeling paradigm. In addition, we introduce a word-span stress regularizer (WSR) that concentrates token-level SSD probabilities within each stressed word span. Experiments on the TinyStress-15K benchmark show that the proposed method outperforms strong baselines, with the complete configuration achieving the best SSD result.
    
[^91]: 无状态语言智能体：扩展长时程自动化研究

    Stateless Language Agents: Scaling Long-Horizon Automated Research

    [https://arxiv.org/abs/2610.07625](https://arxiv.org/abs/2610.07625)

    提出无状态语言智能体（SLA）框架，通过“有状态搜索、无状态智能体”的原则——由框架统一管理研究状态并为每次调用重建角色化上下文——来解决长时程自动化研究中智能体重放冗长历史、重复劳动和过早停止实验等失败模式。

    

    自动化研究系统越来越多地在长时程上运行LLM智能体，但更多的推理本身并不能带来更多的研究进展：智能体会重放不断增长的历史记录、重复彼此的工作，或者在token持续消耗的同时停止实验。然而，大多数评估使用较短的预算或很快饱和的基准测试，使得这些失败模式未曾得到检验。我们将这些失败追溯到两个关键选择：研究状态存储在哪里，以及由谁决定下一步尝试什么。我们提出了无状态语言智能体，其建立在“有状态搜索、无状态智能体”的原则之上：没有任何智能体在多次调用之间携带其对话历史；相反，由框架拥有研究状态（候选解决方案和测量结果），并为每次调用重建一个全新的、特定角色的上下文。每个智能体所看到的内容由此成为一种显式的设计选择，而不是随运行而不断增长的历史。我们在SLA框架中实现了这一原则，

    arXiv:2610.07625v1 Announce Type: cross  Abstract: Automated research systems increasingly run LLM agents over long horizons, but more inference does not by itself produce more progress: agents replay growing histories, duplicate one another's work, or stop experimenting while token consumption continues. Yet most evaluations use short budgets or benchmarks that saturate early, leaving these failure modes untested. We trace these failures to two choices: where research state lives and who decides what to try next. We introduce Stateless Language Agents (SLAs), built on the principle of stateful search with stateless agents: no agent carries its conversation across invocations; instead, the harness owns the research state (candidate solutions and measured outcomes) and reconstructs a fresh and role-specific context for every invocation. What each agent sees becomes an explicit design choice rather than a history that grows with the run. We implement this principle in the SLA framework, 
    
[^92]: 循环环路Transformer

    Recurrent Looped Transformer

    [https://arxiv.org/abs/2610.07591](https://arxiv.org/abs/2610.07591)

    提出循环环路Transformer（RLT），通过将层分配给并行因果编码器和循环解码器，使计算路径随序列长度增长而每个token成本保持固定，在状态跟踪和算法泛化任务上大幅超越固定深度的标准Transformer。

    

    状态跟踪需要对每个输入进行更新，但Transformer应用于每个token的深度是固定的，与序列长度无关。我们提出了循环环路Transformer（Recurrent Looped Transformer, RLT），它将模型层分配给一个并行因果编码器和一个循环解码器。在每个token处，解码器将编码器输出与上一个token的最终解码器状态合并，因此计算路径随序列长度增长，而每个token的计算成本保持固定。在六个算法任务上，我们在三个随机种子下比较了八层模型的五种分配方式与一个八层Transformer。在最多40位数据上训练后，两种RLT分配方式在所有种子中均能以100%的准确率将奇偶性判断泛化到256位，而标准Transformer仍停留在随机猜测水平。在训练长度八倍的基于交换的$S_5$置换跟踪任务上，RLT达到97%的最终状态准确率，而Transformer不足1%，且准确率随解码器深度增加而提升。在超出训练范围的模运算任务上……

    arXiv:2610.07591v1 Announce Type: cross  Abstract: State tracking requires an update at every input, but the depth a Transformer applies to each token is fixed regardless of sequence length. We introduce the Recurrent Looped Transformer (RLT), which splits its layers between a parallel causal encoder and a recurrent decoder. At each token, the decoder merges the encoder output with the previous token's final decoder state, so the computation path grows with sequence length at a fixed per-token cost. On six algorithmic tasks, we compare five splits of eight layers with an eight-layer Transformer over three seeds. Trained on at most 40 bits, two RLT splits generalize parity to 256 bits with 100% accuracy in every seed, while the Transformer stays at chance. On swap-based $S_5$ permutation tracking at eight times the training length, RLT reaches 97% final-state accuracy versus under 1% for the Transformer, and accuracy increases with decoder depth. On modular arithmetic beyond the trainin
    
[^93]: 基于显式偏好推断的异构偏好下大语言模型编排

    Large Language Model Orchestration under Heterogeneous Preferences via Explicit Persona Inference

    [https://arxiv.org/abs/2610.07587](https://arxiv.org/abs/2610.07587)

    提出HARP框架，将智能体的偏好信念从提示文本中移出，改为在有限候选偏好集合上维护数值后验分布进行显式推断，避免早期错误持续传播，从而改进异构偏好环境下的大语言模型编排。

    

    LLM编排研究编排器如何协调一组自主智能体，以实现共同目标或最大化集体福利。这些智能体通常是异构的，每个智能体都持有一种私有偏好，它会追求该偏好但不予公开。从行为中推断这种隐藏偏好一直是博弈论和多智能体系统领域长期研究的课题。核心挑战在于对每个智能体的偏好维持一个信念，并根据观察到的智能体行为来更新它。现有的LLM编排器将该信念作为提示文本携带，缺乏明确的更新规则，这使得早期错误得以持续和传播，而不是被纠正。因此，我们提出了HARP（通过偏好推断进行异构偏好智能体编排），这是一种新颖的框架，将信念从提示中移出。具体而言，HARP在有限的候选偏好集合上为每个智能体维护一个数值后验分布。

    arXiv:2610.07587v1 Announce Type: new  Abstract: LLM orchestration investigates how an orchestrator coordinates a group of autonomous agents to achieve common goals or maximize collective welfare. The agents are typically heterogeneous, each holding a private preference that it pursues but does not reveal. Inferring such hidden preferences from behavior has been a subject of long-standing research in game theory and multi-agent systems. The core challenge lies in maintaining a belief over every agent's preference and updating it from the agents' observed actions. Existing LLM orchestrators carry that belief as prompt text with no explicit update rule. This lets early errors persist and propagate rather than be corrected. We therefore propose \textbf{HARP} (Heterogeneous-preference Agent oRchestration via Preference inference), a novel framework that moves the belief out of the prompt. Specifically, HARP maintains one numeric posterior per agent over a finite set of candidate preference
    
[^94]: LOGIC：面向航空航天电气系统中基于意图的变化影响分析的LLM基准

    LOGIC: An LLM Benchmark for Intent-Grounded Change Impact in Aerospace Electrical Systems

    [https://arxiv.org/abs/2610.07580](https://arxiv.org/abs/2610.07580)

    LOGIC是一个航空航天电气系统领域的受控基准，用于评估语言模型能否根据工程请求意图从确定性候选变更清单中正确选择变更，并通过类型化电气可追溯性图传播其影响，实验表明仅门控结构化证据方法在明确锚定的选择案例上达到了完美的F1分数1.0000。

    

    航空航天电气设计修订中可能包含多个真实变更，但一份工程请求可能仅授权其中一部分变更。因此，将每个检测到的差异都进行传播可能会产生过于宽泛的影响报告。我们提出了LOGIC，这是一个受控基准与评估框架，其中可本地部署的语言模型先将请求锚定在确定性的候选变更清单上，然后再将选定的变更通过类型化的电气可追溯性图进行传播。这种分离设计使得候选选择错误能够与下游传播错误区分开来。LOGIC包含168个场景，其中包括144个选择案例和24个弃权案例。我们评估了三个7B-8B参数规模的模型，并与意图无关方法、词汇匹配方法和结构化证据方法进行比较，同时设置了oracle根作为性能上限。在96个明确锚定的选择案例中，仅门控结构化证据方法达到了1.0000的候选F1分数，而token词汇匹配方法仅为0.9677。

    arXiv:2610.07580v1 Announce Type: new  Abstract: Aerospace electrical-design revisions can contain multiple genuine changes, although an engineering request may authorize only a subset. Propagating every detected difference can therefore produce overly broad impact reports. We present LOGIC, a controlled benchmark and evaluation framework in which locally deployable language models ground a request in a deterministic candidate-change inventory before selected changes are propagated through a typed electrical traceability graph. This separation permits candidate-selection errors to be distinguished from downstream propagation errors. LOGIC contains 168 scenarios, including 144 selection and 24 abstention cases. We evaluate three 7--8B models against intent-agnostic, lexical, and structured-evidence methods, with an oracle-root upper bound. On 96 explicitly anchored selection cases, gate-only structured evidence achieves candidate F1 of 1.0000, compared with 0.9677 for token-lexical matc
    
[^95]: 两个向量替代上下文示例：通过嵌入实现结构化任务适配

    Two Vectors Replace In-Context Demos: Structured Task Adaptation via Embeddings

    [https://arxiv.org/abs/2610.07572](https://arxiv.org/abs/2610.07572)

    提出STAVE方法，用两个任务特定向量（读取向量和上下文向量）直接加到现有输入嵌入中来替代上下文示例，避免了示例图像的重复编码开销，实现高效的结构化任务适配。

    

    上下文学习（ICL）能够将冻结的大型多模态模型（LMMs）通过少量示例适应到新任务，但每次查询时都需要重新编码这些示例，其中每个示例图像会增加多达数百个视觉token。无示例方法通过紧凑的任务状态消除了这一成本，但它们将任务状态添加在针对每个任务搜索的位置或每个解码器层，导致任务参数随深度增长。此外，插入的token或键无法改变原始提示在层内分配注意力的方式。为解决这些问题，我们提出了结构化任务适配方法（STAVE），用两个任务特定的向量替代示例，并将其添加到现有的输入嵌入中。具体而言，读取向量更新产生答案的token，上下文向量更新其他结构化token组。两者均通过在有示例和无示例的提示上使用答案标签进行训练。我们使用一阶分析从理论上论证了这些设计选择的合理性。

    arXiv:2610.07572v1 Announce Type: new  Abstract: In-context learning (ICL) adapts frozen large multimodal models (LMMs) to new tasks from a few demonstrations (demos), but re-encodes them at every query, where each demo image adds up to hundreds of visual tokens. Demo-free methods remove this cost with a compact task state. However, they add it at locations searched per task or at every decoder layer, where task parameters grow with depth. Moreover, inserted tokens or keys cannot change how the original prompt divides its attention within a layer. To address these issues, we propose Structured Task Adaptation via Embeddings (STAVE), which replaces demos with two task-specific vectors added to existing input embeddings. Specifically, a readout vector updates the answer-producing tokens and a context vector updates the other structural token groups. Both are trained with answer labels on prompts with and without demos. We justify these design choices theoretically using a first-order ana
    
[^96]: HouseholdBench：评估大语言模型作为家庭经济行为预测器

    HouseholdBench: Evaluating Large Language Models as Predictors of Household Economic Behavior

    [https://arxiv.org/abs/2610.07563](https://arxiv.org/abs/2610.07563)

    提出HouseholdBench基准，整合6项美国家庭调查和32个覆盖数值、类别与概率型结果的预测任务，系统评估大语言模型预测家庭经济行为及响应政策变化的能力。

    

    大语言模型（LLMs）有潜力实现经济学中的一个关键目标：建立一个适用于多种情境的家庭决策定量模型。然而，现有评估仅覆盖少量调查和结果，且未研究家庭如何适应不断变化的经济条件。我们提出了一项新的评估基准 HouseholdBench，它整合了6项美国家庭调查和32个预测任务，涵盖数值型、类别型和概率型结果，涉及消费、收入、劳动、预期和住房等领域。这些任务利用过去的行为、人口统计特征和宏观经济条件，检验大语言模型能否预测家庭行为，包括家庭如何调整以应对各种政策变化。我们评估了13个专有和开放权重的大语言模型，并与“无变化”基线及梯度提升树模型进行比较。大多数大语言模型的表现优于“无变化”基线，包括在政策响应任务中——表现最好的模型将数值型结果的误差降低了……（原文截断）

    arXiv:2610.07563v1 Announce Type: new  Abstract: Large language models (LLMs) have the potential to meet a key goal in economics: a quantitative model of household decision making, across a variety of settings. Yet existing evaluations cover few surveys and outcomes, and do not study how households adjust to changing economic conditions. We introduce a new evaluation, HouseholdBench, which unites 6 U.S. household surveys and 32 prediction tasks spanning numeric, categorical and probabilistic outcomes, related to consumption, income, labor, expectations, and housing. Using past behavior, demographics and macroeconomic conditions, the tasks test whether LLMs predict behavior, including how households adjust to changes in various policies. We evaluate 13 proprietary and open-weight LLMs against a no-change baseline and a gradient-boosted tree model. Most LLMs outperform the no-change baseline, including for policy response tasks -- with the best model lowering error for numeric outcomes b
    
[^97]: 边缘设备上的质量感知自校正语音翻译

    Quality-Aware Self-Correcting Speech Translation on an Edge Device

    [https://arxiv.org/abs/2610.07545](https://arxiv.org/abs/2610.07545)

    该论文提出了一种在Jetson Nano边缘设备上运行的完全离线自校正语音翻译流水线，发现质量估计（QE）适合作为触发二次校正的门控而不适合作为候选重排序器，其中最小贝叶斯风险解码在阈值0.90下取得了统计显著的翻译质量提升。

    

    我们提出了一个完全离线的语音到语音翻译流水线，可在Jetson Nano（4 GB）上运行，并在无需重新训练的情况下校正自身的弱翻译。Whisper-tiny语音识别模型为Opus-MT翻译器提供输入；多语言BERT余弦相似度作为质量估计（QE）门控，当置信度低于预定义阈值τ时触发二次校正。我们比较了三种校正方法：QE重排序（M1）、最小贝叶斯风险解码（M2）和受限束搜索（M3）。在1,012句FLORES-200句子（英语-西班牙语）上，τ=0.90时的M2相对贪婪解码在BLEU（+0.67，p<0.001）、ChrF（+0.51，p<0.001）和COMET（N=3时+0.0020，p=0.002）上产生了统计显著的改进；M1没有产生显著增益，而M3显著差于基线（p>0.99）。我们的核心发现是：QE作为门控有效，但作为排序器效果不佳：从候选选择中移除QE模型……

    arXiv:2610.07545v1 Announce Type: new  Abstract: We present a fully offline speech-to-speech translation pipeline that runs on a Jetson Nano (4 GB) and corrects its own weak translations without retraining. A Whisper-tiny ASR feeds an Opus-MT translator; multilingual BERT cosine similarity acts as a Quality Estimation (QE) gate, triggering a secondary-pass correction when confidence falls below a pre-defined threshold $\tau$. We compare three correction methods: QE reranking (M1), Minimum Bayes-Risk decoding (M2), and constrained beam search (M3). On 1,012 FLORES-200 sentences (English-Spanish), M2 at $\tau=0.90$ produces statistically significant improvements over greedy decoding on BLEU (+0.67, p<0.001), ChrF (+0.51, p<0.001), and COMET (+0.0020 at N=3, p=0.002); M1 yields no significant gains, and M3 is significantly worse than baseline (p>0.99). Our central finding is that QE functions effectively as a gate but poorly as a ranker: removing the QE model from candidate selection (M1$
    
[^98]: 在异构大语言模型模拟中解耦模型与角色人格

    Disentangling Models from Personas in Heterogeneous LLM Simulations

    [https://arxiv.org/abs/2610.07535](https://arxiv.org/abs/2610.07535)

    该研究通过模拟由多个基础模型驱动的异构社交网络，发现智能体获得的互动量更多取决于其基础模型而非被分配的角色人格，且随着模型数量增加，模型间效应显著增强，表明网络动态可能在大规模下收敛于基础模型效应。

    

    使用大语言模型（LLM）的多智能体模拟通常以单一基础模型驱动整个智能体网络，这忽略了模型间效应，而此类效应可能在现实部署中主导互动动态。为了证明这一点，我们模拟了一个由多个不同基础模型驱动的异构社交网络，并表明智能体所获得的互动量更多取决于其基础模型，而非其被分配的角色人格。当混合中加入更多模型时，基础模型的吸引或排斥效应会显著增强，这表明网络动态在大规模下可能收敛于基础模型效应。为了帮助解释这一效应，我们进行了一系列内容中介分析，展示了基础模型在不同情境下的可预测性，以及模型的词汇模式与互动最大化风格之间的关系。鉴于近期大规模多智能体交互的发展，这项工作……

    arXiv:2610.07535v1 Announce Type: cross  Abstract: Multi-agent simulations with large language models (LLMs) often operate networks of agents with a single base model. This overlooks the inter-model effects which may dominate engagement dynamics in real-world deployments. To show this, we simulate a heterogeneous social network powered by several different base models and show that the amount of engagement an agent receives depends more on its base model than on its assigned persona. The attraction or repulsion effects of a base model strengthen dramatically when more models are added in the mix, suggesting that networks dynamics may converge to base model effects at scale. To help explain this effect, we conduct a series of content-mediating analyses, showing the predictability of base models across contexts as well as the relationship between a model's lexical patterns and an engagement-maximizing style. In light of recent developments in mass multi-agent interaction, this work under
    
[^99]: 通过来自暗知识的模型无关潜在安全信号保障大语言模型安全

    Safeguarding LLMs via Model-Agnostic Latent Safety Signals from Dark Knowledge

    [https://arxiv.org/abs/2610.07532](https://arxiv.org/abs/2610.07532)

    提出LADE方法，通过对比有害与良性查询，从首token输出分布的暗知识中提取模型无关的潜在安全信号，在解码阶段实现安全防御，同时避免安全性与过度拒绝之间的权衡，并能跨架构泛化。

    

    大语言模型（LLM）发展迅速，引发了人们对其安全性日益增长的关注。近期工作提出了检测和防御攻击的方法，包括利用模型隐藏状态的解码阶段防御。然而，现有的解码阶段防御存在两个局限性。首先，它们在安全性与过度拒绝之间存在权衡，即加强安全性会降低模型对良性查询的有用性。其次，许多这些方法依赖于内部隐藏状态，因此仅限于特定架构，带来大量开销且在不同模型间的泛化能力有限。为解决这些限制，我们提出了 LADE（Latent Safety Signals for Defense），该方法通过在首token输出概率分布中，对比有害查询与良性查询，从暗知识（即输出概率分布中超出argmax所携带的信息）中提取潜在安全信号……

    arXiv:2610.07532v1 Announce Type: cross  Abstract: LLMs have advanced rapidly, raising growing concerns about their safety. Recent work has proposed approaches to detect and defend against attacks including defenses at decoding stage that leverage models' hidden states. However, existing decoding-stage defenses suffer from two limitations. First, they introduce a trade-off between safety and over-refusal, where strengthening safety degrades the model's helpfulness on benign queries. Second, many of these methods rely on internal hidden states and are thus restricted to specific architectures, incurring substantial overhead and limited generalization across models. To address these limitations, we introduce LADE (Latent Safety Signals for Defense), which leverages latent safety signals extracted by contrasting harmful and benign queries from dark knowledge (i.e., information carried by the output probability distribution beyond its argmax) in the first-token output probability distribut
    
[^100]: 并非儿童所表达的内容：审计面向儿童AI中手语转文本的安全接口

    Not What a Child Expressed: Auditing the Sign-to-Text Safety Interface in Child-Facing AI

    [https://arxiv.org/abs/2610.07519](https://arxiv.org/abs/2610.07519)

    本文首次提出由聋人参与指导的部署前审计框架，用于审查“手语转文本”系统与儿童安全审核工具之间的接口，揭示SLT翻译错误可能在不影响文本流畅性的情况下悄然改变针对儿童手语使用者的安全决策。

    

    自动手语翻译（SLT）已进入消费级产品，可将美国手语转换为英文文本，用于听写、消息传递以及向对话助手提出的查询。面向儿童的AI与平台的信任与安全工具基于文本进行决策，例如针对未成年人账户的内容过滤器以及对聊天消息进行评分的诱骗（grooming）检测分类器。因此，使用手语翻译的儿童手语使用者需经由一次翻译才能触达这些安全机制。我们发现目前没有任何公开记录的系统曾对这两者进行过联合评估，而部署最广泛的主流SLT模型既未在18岁以下的的手语使用者上训练过，也未对其进行过正式评估。那些会改变否定语义、参与者角色、保密性、紧迫性或求助表达的翻译错误，可能在不影响文本流畅性的情况下悄然改变安全决策。本文提出一种由聋人社群参与指导的部署前审计方法，用以审查这一安全边界，包含失败类型分类法、净化后的场景模式、四种对照条件以及四项结果度量指标。Auslan是……（原文摘要至此截断）

    arXiv:2610.07519v1 Announce Type: new  Abstract: Automatic sign language translation (SLT) has entered consumer products, turning American Sign Language into English text for dictation, messaging, and queries put to a conversational assistant. Child-facing AI and platform trust-and-safety tooling decide on text, using filters on minor accounts and grooming classifiers that score chat messages. A signing child who uses SLT therefore reaches these safeguards through a translation. We found no publicly documented system in which the two have been jointly evaluated, and the leading deployed SLT model was neither trained nor formally evaluated on signers under 18. Errors that alter negation, participant roles, secrecy, urgency or help-seeking could change a safety decision without disturbing fluency. This paper proposes a Deaf-informed pre-deployment audit of that boundary, with a failure taxonomy, a sanitised scenario schema, four comparison conditions, and four outcome measures. Auslan is
    
[^101]: 论信息诱导智能体的开放式信息搜寻

    On Open-Ended Information Seeking for Information Elicitation Agents

    [https://arxiv.org/abs/2610.07509](https://arxiv.org/abs/2610.07509)

    本研究通过在11个跨越不同家族和参数规模的大语言模型上进行的受控诱导模拟，揭示了不同LLM对信息价值的判断存在差异，且这些差异会显著塑造其序列化的开放式信息搜寻行为。

    

    信息诱导是一个开放式的信息搜寻问题，其中交互可以朝许多潜在有价值的方向发展，这要求诱导者在新信息不断涌现时持续决定应追求哪些信息。在智能体化的信息诱导中，这些决策可能被委托给基础模型，然而模型的选择如何塑造由此产生的信息搜寻行为仍缺乏充分研究。我们研究了关于信息价值的判断在不同大语言模型（LLM）之间如何变化，以及这些差异如何塑造序列化的信息搜寻过程。我们首先在跨越多个模型家族和参数规模的11个大语言模型上考察这些判断，实验使用了共享的信息集合和诱导目标。随后，我们开发了一个受控的信息诱导模拟环境，其中不同的模型面对相同的信息空间并使用相同的选择规则，从而将这些价值判断与问题生成和受访者行为隔离开来。

    arXiv:2610.07509v1 Announce Type: new  Abstract: Information elicitation is an open-ended information-seeking problem in which an interaction can unfold in many potentially valuable directions, requiring an elicitor to continually determine which information to pursue as new information emerges. In agentic elicitation, these decisions may be delegated to a foundation model, yet how model choice shapes the resulting information-seeking behavior remains understudied. We study how judgments about information value vary across LLMs and how these differences shape sequential information seeking. We first examine these judgments across 11 LLMs spanning multiple model families and parameter scales, using a shared set of information and elicitation objectives. We then develop a controlled elicitation simulation in which different models encounter the same information space and use the same selection rule, isolating these judgments from question generation and respondent behavior. Using this se
    
[^102]: 通过自动化提供者查询弥合环境式临床文档记录的空白

    Closing Ambient Clinical Documentation Gaps with Automated Provider Queries

    [https://arxiv.org/abs/2610.07502](https://arxiv.org/abs/2610.07502)

    该论文提出DAU（起草-提问-更新）框架，用大语言模型自动化临床文档专员的查询闭环以弥补病历信息缺口，并基于真实就诊审计构建转录退化基准，揭示有效澄清问题的预测因素具有任务特异性，且约9%的查询轮次反而会损害性能。

    

    提供者查询是临床文档专员发送给医生的澄清请求，用于弥补临床病历中的信息缺口并确保准确计费。已有研究在假设转录文本完整的前提下，实现了病历起草、ICD-10编码和医嘱提取的自动化，但这些信息缺口一直未被解决。我们研究大语言模型能否自动化这一查询闭环（称为DAU：起草、提问、更新），并贯穿上述三项任务。通过对3,000次真实就诊的审计，我们确定了文档缺失的来源，并据此在公开数据上构建了五个转录文本退化基准。通过分析真实对话中的2.1万轮澄清交互，我们发现有效问题的预测因素因任务而异：oracle置信度占主导地位，但病历完整性只需简单的回忆类问题，而ICD-10编码则需要更难的多选项问题。约9%的查询轮次会损害性能，其主要原因是冗余问题以及仍会触发重写的无效回答。部署取决于lea（原文在此处截断）

    arXiv:2610.07502v1 Announce Type: new  Abstract: Provider queries are clarifying requests sent by clinical documentation specialists to physicians to close gaps in the clinical note and ensure accurate billing. Prior work automates note drafting, ICD-10 coding, and order extraction assuming a complete transcript, leaving these gaps unaddressed. We study whether an LLM can automate the query loop, termed DAU (Draft, Ask, Update), across those three tasks. An audit of 3,000 real visits identifies the sources of missing documentation, from which we build five transcript-degradation benchmarks on public data. Analyzing 21k clarification turns on real conversations, we find useful-question predictors are task-specific: oracle confidence dominates, but note completeness needs only simple recall questions while ICD-10 coding needs harder, multi-option ones. About 9% of turns hurt performance, driven by redundant questions and non-answers that still trigger a rewrite. Deployment depends on lea
    
[^103]: 推陈出新：利用生成式人工智能增强“经典”文档自动化

    In With the Old: Enhancing 'Classical' Document Automation with Generative AI

    [https://arxiv.org/abs/2610.07480](https://arxiv.org/abs/2610.07480)

    本文探索了基于专家系统等符号方法的经典文档自动化与生成式AI如何相互增强，并通过初步实验证明大语言模型可用于识别和修复非专业人士撰写的法律文本中的问题。

    

    基于软件的法律援助系统已经利用了多种不同形式的知识表示和推理方法。本文探讨了植根于专家系统风格和其他符号方法的文档自动化服务如何能够有效地增强当前的生成式AI方法，并反过来被其增强。我们讨论了可能的益处和挑战，并报告了使用大语言模型来识别和修复非专业人士所写文本中问题的初步实验。

    arXiv:2610.07480v1 Announce Type: new  Abstract: Software-based legal assistance systems have leveraged many different forms of knowledge representation and reasoning. This article explores how document automation services rooted in expert system style and other symbolic approaches can usefully enhance and be enhanced by current generative AI approaches. We discuss the possible benefits and challenges, and report on preliminary experiments in using large language models to identify and fix issues in texts written by laypeople.
    
[^104]: 关于AI智能体的可审计声明

    Auditable Claims about AI Agents

    [https://arxiv.org/abs/2610.07459](https://arxiv.org/abs/2610.07459)

    提出AI智能体声明的可审计性标准——声明必须在事前明确其政策、范围、裁决记录及记录撰写者，并满足独立记录覆盖、授权绑定操作参数和超越完整性的完备性三项条件，才能被有效核查。

    

    组织会对其AI智能体做出各种声明：例如，每封外部邮件都经过人工审批、每个操作都留有日志、评估结果表明该智能体可以安全部署。欧盟《人工智能法案》第12条要求高风险系统允许自动记录事件，但并未说明哪些记录能够证实某项特定声明。我们的立场可以概括为一句话：要使关于智能体的声明能够被核查，该声明必须首先明确其政策、适用范围、能够证实它的记录，以及记录的撰写者。借鉴鉴证业务的前提条件，我们提出：如果在得出任何结论之前，这些要素及裁决规则已被固定，且相关记录可以获取，那么该声明就是可审计的。这将我们的可审计智能体框架中的“政策可核查性”维度从单一操作扩展到了声明层面。智能体场景增加了三个条件：由独立记录进行覆盖、授权与每个操作的参数相绑定，以及超越完整性的完备性。在明确的模型下，我们证明了（原文摘要在此处截断）……

    arXiv:2610.07459v1 Announce Type: new  Abstract: Organizations make claims about their AI agents: a person approves every external email, every action is logged, an evaluation shows the agent is safe to deploy. Article 12 of the EU AI Act requires high-risk systems to allow the automatic recording of events but does not say which records settle a given claim. The position is one sentence: to be checked, a claim about an agent must first name its policy, its scope, the records that would settle it, and who writes them. Adapting the preconditions of an assurance engagement, we call a claim auditable when these elements and a decision rule are fixed before any verdict and the records are obtainable. This extends the Policy Checkability dimension of our Auditable Agents framework from single actions to claims. Agents add three conditions: coverage by an independent record, authorization bound to each action's arguments, and completeness beyond integrity. Under an explicit model, we prove t
    
[^105]: AlignQuant：面向高效大语言模型生成的瓦片对齐混合精度量化

    AlignQuant: Tile-Aligned Mixed-Precision Quantization for Efficient LLM Generation

    [https://arxiv.org/abs/2610.07457](https://arxiv.org/abs/2610.07457)

    AlignQuant提出了一种以GPU兼容的二维权重瓦片作为精度分配、存储和执行公共单元的训练后混合精度量化方法，使大语言模型的压缩能够真正转化为实际推理加速。

    

    细粒度混合精度量化有望实现高效的大语言模型推理，但局部的精度选择可能与GPU规则的存储和计算单元发生冲突。这种精度边界的不匹配限制了压缩向实际加速的转化。我们提出AlignQuant，一种训练后量化方法，它使用GPU兼容的二维权重瓦片作为精度分配、紧凑存储和执行的公共单元。这种共享划分使精度分配能够跟随输出通道内的敏感度变化。联合预填充/解码校准方法在量化激活条件下，使用由语言模型损失梯度加权的投影输出扰动来评估精度降低的影响。相位归一化的评分在模型级权重存储预算下，优先为对任一阶段重要的瓦片分配更高精度。每个瓦片存储一种选定的表示，而相位专用内核则重用打包的（摘要在此处被截断）

    arXiv:2610.07457v1 Announce Type: cross  Abstract: Fine-grained mixed-precision quantization promises efficient large language model inference, but local precision choices can conflict with regular GPU storage and computation units. This precision-boundary mismatch limits the translation of compression into practical acceleration. We introduce AlignQuant, a post-training quantization method that uses GPU-compatible two-dimensional weight tiles as the common unit of precision allocation, compact storage, and execution. This shared partition lets precision follow sensitivity within output channels. Joint prefill/decode calibration scores precision reductions using projection-output perturbations weighted by language-model loss gradients under quantized activations. Phase-normalized scores prioritize higher precision for tiles important to either phase under a model-wide weight-storage budget. Each tile stores one selected representation, while phase-specialized kernels reuse the packed m
    
[^106]: AccentCL：具有增量扩展能力的鲁棒口音分类

    AccentCL: Robust Accent Classification with Incremental Expansion

    [https://arxiv.org/abs/2610.07426](https://arxiv.org/abs/2610.07426)

    提出了AccentCL框架，通过不平衡感知损失、领域均值对齐损失和基于回放的持续学习，实现了对类别不平衡和跨语料库领域偏移具有鲁棒性的英语口音分类，并支持新口音类别的增量扩展。

    

    口音分类器通常使用固定的标签集合进行训练，无法在新的数据出现时容纳新的口音类别。此外，由于各语料库之间录音条件的差异，带口音的语音语料库往往存在显著的类别不平衡和/或领域偏移。我们提出了AccentCL，一个用于英语口音分类的类增量学习框架，它对类别不平衡和跨语料库领域偏移具有鲁棒性。AccentCL从冻结的Whisper-Large-v3编码器中提取多层表示，并通过不平衡感知的交叉熵损失进行优化，以减少对多数口音类别的偏差，同时使用领域均值对齐损失来最小化训练语料库之间的分布均值偏移。随后通过基于回放的持续学习来扩展标签空间，利用冻结的基础模型进行知识保留，并使用新旧间隔损失来减少对新添加类别的过度预测。在……

    arXiv:2610.07426v1 Announce Type: new  Abstract: Accent classifiers are typically trained with a fixed label inventory and cannot accommodate new accent categories as new data becomes available. Moreover, accented speech corpora often exhibit substantial class imbalance and/or domain shift due to differences in recording conditions across corpora. We present AccentCL, a class-incremental learning framework for English accent classification that is robust to class imbalance and cross-corpus domain shift. AccentCL extracts multi-layer representations from a frozen Whisper-Large-v3 encoder, optimized with an imbalance-aware cross-entropy loss to reduce bias toward the majority accent classes and a domain mean alignment loss that minimizes distributional mean shift across training corpora. The label space is then expanded via replay-based continual learning, using the frozen base model for knowledge retention and an old-to-new margin loss to reduce overprediction on newly added classes. On
    
[^107]: 谁写的还不够：检测谁贡献了洞见

    Who Wrote It Is Not Enough: Detecting Who Contributed the Insight

    [https://arxiv.org/abs/2610.07365](https://arxiv.org/abs/2610.07365)

    该论文提出“洞察溯源”新任务并构建InsightProv-v0数据集，通过两阶段对抗框架消除语言捷径信号，从而准确识别科学评审洞见的贡献者是来自人类、大语言模型还是二者的混合。

    

    随着大语言模型日益广泛地辅助科学写作与同行评审，仅检测文本出自谁手已不再足够：我们还需要确定是谁贡献了底层的洞见。我们提出了“洞察溯源”（Insight Provenance）这一新任务，即识别一条评审洞见究竟源自人类、大语言模型，还是二者的混合贡献。我们从4,057篇科学论文和12,660条人类评审中构建了InsightProv-v0数据集，使用GPT-4o、Gemini和DeepSeek模拟不同程度的大语言模型参与，并在句子层面标注来源。我们证明，模型在原始数据上的出色表现可能具有误导性，因为模型会利用语言和文本作者身份方面的捷径，而这些捷径带来的优势在逐步去偏的评估下会大幅退化。因此，我们提出了一个两阶段对抗框架，在抑制捷径信号的同时保留与来源相关的信息。除检测之外，大量分析还揭示了智能作者身份得以识别的关键因素……

    arXiv:2610.07365v1 Announce Type: new  Abstract: As LLMs increasingly assist scientific writing and peer review, detecting who wrote the text is no longer sufficient: we need to determine who contributed the underlying insight. We introduce Insight Provenance, the task of identifying whether a review insight originates from a human, an LLM, or their hybrid contribution. We construct InsightProv-v0 from 4,057 scientific papers and 12,660 human reviews, simulating different levels of LLM involvement with GPT-4o, Gemini, and DeepSeek and annotating provenance at the sentence level. We show that strong performance on raw data can be misleading, as models exploit linguistic and textual-authorship shortcuts that degrade substantially under progressively debiased evaluation. We therefore propose a two-stage adversarial framework that suppresses shortcut signals while preserving provenance-relevant information. Beyond detection, extensive analyses reveal what makes intellectual authorship iden
    
[^108]: 追踪并非永久性：视频世界模型对被隐藏的物体保留了什么

    Tracking Is Not Permanence: What Video World Models Keep of a Hidden Object

    [https://arxiv.org/abs/2610.07355](https://arxiv.org/abs/2610.07355)

    该研究发现尽管视频世界模型的编码器完整地编码了被遮挡物体的信息，但其预测器会在遮挡发生后0.3秒内迅速丢弃这些信息，揭示了当前视频世界模型严重缺乏物体永久性表征。

    

    视频世界模型能够追踪它们看得见的物体；我们则探究它们对看不见的物体究竟保留了什么。我们对一个冻结的V-JEPA 2预测器隐藏某个物体，并将其对被隐藏区域的预测与编码器对两个仅在该区域内有所不同的世界的表征进行比较。结果发现，预测器的决策只能部分保留静止物体，对被装在容器中携带的物体则完全丢失，并在0.3秒内丢失运动中的物体（在V-JEPA自身的管状掩码下为0.5秒；ViT-H在预训练采用的90%掩码比例下可保留至1.1秒）；在投影中仍留有痕迹，但低于中点，仅为复制最后一帧画面的基线所保留信息的14-60%。信息本身是存在的：编码器能以1.00的准确率读取物体的存在，并使封闭容器内的内容在3.5秒内保持可解码；而用编码器自身的探针去读取预测器的输出时，在盒子关闭半秒后，仅有2%的场景中还能检测到球的存在。在渲染场景中，物体永久性缺失……

    arXiv:2610.07355v1 Announce Type: cross  Abstract: Video world models track objects they can see; we ask what they keep of objects they cannot. We hide an object from a frozen V-JEPA 2 predictor and compare its prediction for the hidden region with the encoder's representation of two worlds that differ only inside that region. The predictor's decision keeps a stationary object in part and one carried inside a container not at all, and loses a moving one within 0.3 s (0.5 s under V-JEPA's own tube mask; ViT-H keeps it to 1.1 s at pretraining's 90% masking ratio); in projection a trace remains, below the midpoint, at 14-60% of what a baseline copying the last view retains. The information is there: the encoder reads the object's presence at 1.00 and keeps a closed container's contents decodable for 3.5 s, while the predictor's output, read with the encoder's own probe, contains the ball in 2% of scenes once the box has been closed for half a second. On rendered scenes, permanence is miss
    
[^109]: 阶梯式MoE：具有可配置推理复杂度的分段级路由

    Stepped MoE: Segment-Level Routing with Configurable Inference Complexity

    [https://arxiv.org/abs/2610.07348](https://arxiv.org/abs/2610.07348)

    本文提出阶梯式MoE统一框架，将弹性结构与稀疏门控架构相结合，通过分段级路由使模型能够同时适应不同的部署约束和任务需求，实现推理时对精度-效率权衡的细粒度控制。

    

    训练大型语言模型（LLM）非常耗费资源，而将模型适配到具有不同计算约束的多样化部署场景仍然具有挑战性。虽然弹性架构能够实现灵活的模型部署，稀疏激活模型允许输入自适应的计算，但现有方法将这些维度独立处理。此外，面向设备端边缘推理的模型需要符合服务设备的内存和计算限制。在本文中，我们引入了一个统一框架，将弹性结构与稀疏门控架构相结合，创建能够同时适应部署约束和任务需求的模型。我们的方法采用一个以上下文和目标效率规格为条件的模型骨干网络，从而在推理时实现对精度-效率权衡的细粒度控制。模型学习激活与任务相关的部分……

    arXiv:2610.07348v1 Announce Type: cross  Abstract: Training large language models (LLMs) is resource-intensive, and adapting them for diverse deployment scenarios with varying computational constraints remains challenging. While elastic architectures enable flexible model deployment and sparsely activated models allow input-adaptive computation, existing approaches treat these dimensions independently. Moreover, models catered towards on-device edge inference need to conform to the memory and compute limitations of the serving devices. In this paper, we introduce a unified framework that combines elastic structures with sparsely gated architectures to create models that adapt simultaneously to both deployment constraints and task requirements. Our approach employs a model backbone that conditions on both the context and target efficiency specifications, enabling fine-grained control over the accuracy-efficiency trade-off at inference time. The model learns to activate task-relevant par
    
[^110]: TC3-VQA：基于条令的战术战斗伤员救护视觉问答数据集

    A doctrine-grounded visual question answering dataset for Tactical Combat Casualty Care

    [https://arxiv.org/abs/2610.07339](https://arxiv.org/abs/2610.07339)

    本文提出TC3-VQA数据集，利用公开教学与实战视频和权威条令文档构建了1,860个将视觉证据与可追溯战伤救护条令关联的问答样本，为支持战术战斗伤员救护的视觉-语言模型开发提供监督数据。

    

    战术战斗伤员救护（TC3）要求救援人员将伤情和救治干预的视觉观察与既定的临床指导联系起来。开发支持这一过程的视觉-语言模型，需要能够将可见证据与可追溯条令相关联的监督数据。我们提出了TC3-VQA，这是一个由公开的教学与实战TC3视频以及权威TC3文档构建的数据集。它包含跨越11个概念的581个条目，共1,860个问题，涵盖干预识别、条令、临床推理、操作流程指导，以及视觉信息不足时的拒答。基于条令的答案保留了逐字原文段落和字符偏移量。数据集构建结合了视觉标注、段落检索、蕴含检查以及跨模型家族的验证。设备框、解剖标签、时间段和来源元数据随问答对一同提供。自动化审计与评分（摘要原文在此处被截断）

    arXiv:2610.07339v1 Announce Type: cross  Abstract: Tactical Combat Casualty Care (TC3) requires responders to connect visual observations of injuries and interventions with established clinical guidance. Developing vision-language models to support this process requires supervision that links visible evidence to traceable doctrine. We present TC3-VQA, a dataset constructed from public instructional and field TC3 videos and authoritative TC3 documents. It contains 581 items spanning 11 concepts, with 1,860 questions covering intervention recognition, doctrine, clinical reasoning, procedural guidance, and refusal when visual information is insufficient. Doctrine-based answers preserve verbatim source passages and character offsets. Construction combines visual annotation, passage retrieval, entailment checks, and verification across model families. Equipment boxes, anatomical labels, temporal segments, and source metadata accompany the question-answer pairs. Automated audits and ratings 
    
[^111]: Logbook：超长时音频事件理解

    Logbook: Extremely Long-form Audio Event Understanding

    [https://arxiv.org/abs/2610.07338](https://arxiv.org/abs/2610.07338)

    该论文提出了面向小时级至六天超长音频的事件理解基准 Logbook，要求系统对连续音频进行无缝隙分割并为每段生成事件标签与描述，发现最佳系统仍不及人类、过度分割普遍存在，且端到端系统通常优于级联系统但性能随上下文变长而下降。

    

    现有的音频基准测试都围绕短小、预先分割的音频片段构建，这将模型设计限制在简短输入或固定词表上。为了弥合这一差距，我们提出了 Logbook，一个面向小时级音频理解的基准测试，其录音时长从十分钟到六天不等。给定一段连续的音频录音和一个事件标签词表，系统必须预测出无缝隙的分割结果，并为每个片段提供事件标签和描述。我们比较了52个端到端和级联系统，并对微调、上下文长度和推理预算进行了消融实验。我们发现该任务是可以解决的，但表现最好的系统仍低于人类参考水平。此外，过度分割现象普遍存在，微调可以部分缓解这一问题。最后，端到端系统通常优于级联系统，但其性能会随着上下文变长而下降。

    arXiv:2610.07338v1 Announce Type: cross  Abstract: Audio benchmarks are built around short, pre-segmented clips, limiting model design to brief inputs or fixed vocabularies. To close this gap, we introduce Logbook, a benchmark for hour-scale audio understanding, with recordings ranging from ten minutes to six days. Given a continuous audio recording and an event label vocabulary, a system must predict a gap-free segmentation with an event label and a description per segment. We compare 52 systems, end-to-end and cascaded, and ablate fine-tuning, context length, and reasoning budget. We find the task tractable, though the best systems remain below the human reference. Also, over-segmentation is pervasive, and fine-tuning partially mitigates it. Finally, end-to-end are often better than cascaded systems, but degrades with longer context.
    
[^112]: 面向智能体强化学习的MoE专家选择结构化方法

    Structuring MoE Expert Selection for Agentic Reinforcement Learning

    [https://arxiv.org/abs/2610.07332](https://arxiv.org/abs/2610.07332)

    该论文发现MoE专家选择与智能体轨迹存在天然的结构对齐（语义相似操作的轮次共享更多路由专家），并提出分层路由控制框架将这一结构约束纳入智能体强化学习训练，从而同时提升任务性能与推理效率。

    

    摘要（arXiv:2610.07332v1，公告类型：cross）：长程大语言模型智能体通常采用稀疏混合专家模型来实现，然而智能体行为与MoE结构的协同设计仍未得到充分探索。在本工作中，我们全面研究了智能体后训练与MoE专家选择之间的联系。在现成的MoE模型中，我们观察到专家选择呈现出一种专门化结构，该结构天然地与智能体轨迹相契合。具体而言，当智能体在不同轮次中执行语义相似的操作（例如READ、UPDATE）时，其专家路由的重叠程度要高于执行不同操作的轮次之间。然而，标准的强化学习算法忽略了这种专门化，使得MoE路由在训练过程中处于不受控的状态，这在经验上限制了任务性能和推理效率。为解决这一问题，我们引入了一个面向智能体任务的分层路由控制框架。我们显式地鼓励轮次级别的专家选择与（摘要在此处截断）

    arXiv:2610.07332v1 Announce Type: cross  Abstract: Long-horizon LLM agents are frequently implemented using sparse mixture-of-experts (MoE) models, yet the co-design of agentic behavior and MoE structures remains underexplored. In this work, we comprehensively study the connections between agentic post-training and MoE expert selection. In off-the-shelf MoE models, we observe expert selection exhibits a specialized structure that naturally aligns with agentic trajectories. Specifically, expert routing overlaps more between turns where the agent performs semantically similar operations (e.g., READ, UPDATE) than between turns with differing operations. However, standard RL algorithms ignore this specialization, allowing the MoE routing to go uncontrolled during training, which empirically limit task performance and inference efficiency. To address this, we introduce a hierarchical routing control framework for agentic tasks. We explicitly encourage turn-level expert selections to align w
    
[^113]: SharedKV-BT：面向行为树智能体的节点本地类型化决策

    SharedKV-BT: Node-Local Typed Decisions for Behavior-Tree Agents

    [https://arxiv.org/abs/2610.07327](https://arxiv.org/abs/2610.07327)

    该论文提出SharedKV-BT，通过行为树活动节点暴露节点本地字段并由Shared-KV并行评分候选决策，使类型化决策速度较自回归解码提升2.36-4.15倍，同时将操作任务的联合决策准确率从75%提升至94%。

    

    智能体任务需要一系列相互依赖的决策。自回归模型比传统分类器支持更灵活的决策接口，但会带来逐token生成的延迟。近期的共享前缀方法通过重用已编码上下文并并行对多个决策进行评分来降低这一成本，但没有建模决策之间的依赖关系，也没有验证执行。我们提出SharedKV-BT，其中行为树的每个活动节点暴露阶段本地的字段和候选项，Shared-KV并行地对候选进行评分，并将选定的决策传递给独立的执行系统。我们在机器人操作、移动导航和计算机使用任务上测试了SharedKV-BT。在这三个任务中，SharedKV-BT的类型化决策速度比提示匹配的自回归解码快2.36-4.15倍。在操作任务上，节点本地的Shared-KV将联合决策准确率从75%提升至94%，闭环成功率从0%提升至……（摘要原文在此处截断）

    arXiv:2610.07327v1 Announce Type: cross  Abstract: Agent tasks require sequences of interdependent decisions. Autoregressive models support more flexible decision interfaces than conventional classifiers but incur the latency of token-by-token generation. Recent shared-prefix methods reduce this cost by reusing encoded context and scoring multiple decisions in parallel, but do not model decision dependencies or verify execution. We propose SharedKV-BT, where each active node of a behavior tree (BT) exposes stage-local fields and candidates, and Shared-KV scores the candidates in parallel and passes the selected decision to a separate execution system. We tested SharedKV-BT on robot manipulation, mobile navigation, and computer-use tasks. Across three tasks, SharedKV-BT made typed decisions 2.36-4.15 times faster than prompt-matched autoregressive decoding. On the manipulation task, node-local Shared-KV improved joint decision accuracy from 75% to 94% and closed-loop success from 0% to 
    
[^114]: Kurate：可扩展的科研质量分析系统

    Kurate: Scalable Scientific Quality Analysis

    [https://arxiv.org/abs/2610.07306](https://arxiv.org/abs/2610.07306)

    Kurate是一个利用大语言模型从统计功效、选择性报告等8个维度大规模评估已发表研究质量，并将每项评估判断链接到具体文本依据的科研质量分析系统。

    

    科学检索系统能够找到与某个问题相关的论文，但通常不会评估这些论文所提供证据的质量。我们提出了Kurate，一个利用大语言模型（LLM）评估已发表研究质量的系统。Kurate同时利用论文本身及其相关文档（例如该研究的试验注册信息和方案），并将每项判断与其所依据的文本段落相链接。我们将Kurate应用于一个包含4,347篇论文的语料库（其中3,913篇报告了随机对照试验），并从研究设计与报告的8个维度对每篇论文进行评分，具体包括：统计功效、因果识别、预注册、选择性报告、测量效度、分析预设、报告透明度，以及利益冲突与资助情况。在整个语料库中，我们发现论文最常出现的问题集中在统计功效、选择性报告以及分析（原文在此处截断）……

    arXiv:2610.07306v1 Announce Type: new  Abstract: Scientific search systems can find papers that are relevant to a question, but they generally do not assess the quality of the evidence that those papers provide. We present Kurate, a system that uses large language models (LLMs) to assess the quality of published studies. Kurate uses both the paper and its related documents (e.g., the study's trial registration and protocol), and links each of its judgments to the passage of text on which that judgment is based. We applied Kurate to a corpus of 4,347 papers (3,913 of which report randomized trials) and scored each paper on 8 dimensions of study design and reporting: specifically, statistical power, causal identification, preregistration, selective reporting, measurement validity, analysis prespecification, reporting transparency, and conflict of interest and funding. Across the corpus, we found that papers most often exhibited issues with statistical power, selective reporting, and anal
    
[^115]: 谱系感知的内存治理：面向企业AI智能体隐私保护列级访问控制的派生门控框架

    Lineage-Aware Memory Governance: A Derivation-Gated Framework for Privacy-Preserving Column-Level Access Control in Enterprise AI Agents

    [https://arxiv.org/abs/2610.07258](https://arxiv.org/abs/2610.07258)

    该论文提出分析内存单元（AMU），通过为每个缓存结果附加完整的派生谱系图并实施列级权限门控检索，从设计上保证企业AI智能体不会命中由请求者无权限的敏感列派生而来的缓存结果，同时解决部门间同名KPI计算逻辑冲突的问题。

    

    共享内存存储的企业AI智能体面临两个尚未解决的风险：敏感数据可能通过请求者本无权限推导出的“合法计算结果”发生泄露，以及各部门可能通过相互冲突的逻辑静默地计算同名关键绩效指标（KPI）。现有的智能体内存系统（如MemGPT、Zep、A-MEM）基于内容、所有权和角色来控制检索，而非基于派生关系，因而无法拦截缓存中嵌入了被禁止访问列的洞察结果。我们提出了分析内存单元（AMU），这是一种内存模式，为每个缓存结果附加完整的派生（谱系）图，并由检索策略进行门控——仅当请求者对结果所涉及的每一列均具有访问权限时，才返回缓存命中。在谱系记录完整的前提下，我们通过构造性证明表明，该策略能够阻止检索由请求者权限之外的敏感列所派生的结果，最坏情况复杂度为O(n)——这是一种条件性的设计保证，而非经验性的……

    arXiv:2610.07258v1 Announce Type: cross  Abstract: Enterprise AI agents that share a memory store face two unaddressed risks: sensitive data can leak through legitimately computed results the requester could not derive, and departments can silently compute a same-named key performance indicator (KPI) through conflicting logic. Existing agent-memory systems (e.g., MemGPT, Zep, A-MEM) gate retrieval by content, ownership, and role, not derivation, missing a cached insight that embeds a forbidden column. We introduce the Analytical Memory Unit (AMU), a memory schema that attaches a full derivation (lineage) graph to every cached result, gated by a retrieval policy that serves a hit only when the requester is authorised for every column touched. Provided lineage recording is complete, we prove by construction that the policy blocks retrieval of results derived from a sensitive column outside the requester's permissions, at O(n) worst case -- a conditional design guarantee, not an empirical
    
[^116]: 最小见证强化学习

    Minimal Witness Reinforcement Learning

    [https://arxiv.org/abs/2610.07226](https://arxiv.org/abs/2610.07226)

    本文提出最小见证强化学习（MWRL），利用基于集合并集覆盖损失的信用分配机制，仅凭单一黑盒验证器的信号即可同时实现解的最小性与多个备选解的恢复。

    

    “足以产生某一结果的不可约条件是什么？”这是计算与科学领域中最常出现的核心问题之一。其答案——最小充分见证——正是我们所称的解释、机制与原因。这类问题通常需要找出多个最小见证，然而标准的强化学习方法可能只能揭示一个解或冗余的解。我们将该问题形式化为最小见证识别，并提出了最小见证强化学习（MWRL）。MWRL 对从策略中采样得到的、经成功验证的提议所认证的集合取并集，并根据“若缺少该提议，组并集将会损失的覆盖范围”为每个提议分配信用。这一直接源于问题定义的信用分配机制，仅凭一个黑盒验证器的单个比特信号，便统一了对最小性与备选解恢复的双重需求。基于该原理，我们推导出一个值迭代规划器，能够恢复……（摘要原文在此处截断）

    arXiv:2610.07226v1 Announce Type: cross  Abstract: ``What are the irreducible conditions that are sufficient to produce an outcome?'' is one of the most common questions that recur across computation and science. Its answers, the minimal sufficient witnesses, are what we mean by explanations, mechanisms and reasons. These problems usually ask for multiple minimal witnesses, yet standard RL methods may reveal only one solution or redundant ones. We formalize this problem as minimal-witness identification and introduce Minimal-Witness Reinforcement Learning (MWRL). MWRL takes the union of the sets certified by successful proposals sampled from the policy and credits each proposal for the coverage the group union would lose without that proposal. This credit assignment, derived directly from the problem definition, unifies the demands for minimality and recovery of alternatives from a single black-box verifier bit. Under this principle, we derive a value iteration planner that recovers th
    
[^117]: TIDE 2.0：一个开放、模型无关的临床笔记密钥化去标识化引擎

    TIDE 2.0: an open, model-agnostic engine for keyed de-identification of clinical notes

    [https://arxiv.org/abs/2610.07224](https://arxiv.org/abs/2610.07224)

    TIDE 2.0是一个开源、模型无关的临床笔记去标识化引擎，通过密钥化匿名技术——包括保持时间间隔的日期偏移和密码学生成的替代值——在保护患者隐私的同时保留数据的纵向分析价值，且无需依赖外部硬件或存储关联表。

    

    临床笔记记录了患者诊疗过程中绝大部分的文档信息，但在移除受保护健康信息（PHI）之前，这些笔记无法用于研究。去标识化通常被视为一个检测问题。然而仅有检测是不够的：涂黑处理会连同标识符一起删除临床内容，日期置空会破坏纵向分析所需的时间间隔，而每次出现都分配全新的随机替代值会破坏患者各条笔记之间的关联。我们提出了TIDE 2.0，一个采用MIT许可证的引擎，包含两个可分离的阶段：可互换的识别器和密钥化匿名器。两者均可在机构自有的硬件上运行。替代值通过密码学方法生成，无需存储关联表。日期通过每个患者特定的、保持时间间隔的偏移量进行平移；在给定密钥下，每个值在所有出现位置都获得相同的替代值；并且使用新密钥生成的发布版本无法与之前的发布版本建立关联。

    arXiv:2610.07224v1 Announce Type: cross  Abstract: Clinical notes capture most of what is documented about a patient's care, but they cannot be used for research until protected health information (PHI) is removed. De-identification is often treated as a detection problem. Detection alone is not sufficient: redaction strips clinical content along with identifiers, date blanking destroys the temporal intervals needed for longitudinal analysis, and assigning a fresh random surrogate at each occurrence breaks links between a patient's notes. We present TIDE 2.0, an MIT-licensed engine with two separable stages: an interchangeable recognizer and a keyed anonymizer. Both run on hardware the institution owns. Surrogates are generated cryptographically with no stored linkage table. Dates shift by a per-patient, interval-preserving offset; each value receives the same surrogate across all occurrences under a given key; and a release produced under a new key cannot be linked to earlier releases
    
[^118]: 预测社交媒体信息级联的增长：迈向人在回路的虚假信息分诊

    Forecasting the Growth of Social Media Information Cascades: Towards Human-in-the-Loop Misinformation Triage

    [https://arxiv.org/abs/2610.07209](https://arxiv.org/abs/2610.07209)

    该论文提出在虚假信息传播的前30分钟内，通过结合早期节点数、结构深度熵和时间到达熵的特征来预测传播树的后续增长，尤其在内容高度活跃的传播树上表现优异，从而帮助人手有限的审核团队在传播范围尚不明确时优先分诊出具有高增长潜力的虚假信息。

    

    有限的审核团队必须在最终传播范围尚未确定之前，识别出哪些新出现的说法可能会持续增长。我们将早期虚假信息分诊聚焦于这一持续预测问题：根据活动的前30分钟来预测后续记录到的传播树增长。在FibVID数据集上，我们比较了早期节点数量与结构深度熵、时间到达熵以及两者组合的特征，同时将每个原始说法的所有传播树保持在同一分区中。在来自59个与训练集分离的说法组的352棵测试树上，组合模型将对数变换后的未来增长的R²从0.307提升到0.323，并将log-MAE降低了2.4%（95%说法自举置信区间为-0.4%至5.2%）。在97棵高活跃度树上增益尤为显著：R²从0.248上升到0.395，Spearman ρ从0.394上升到0.529，log-MAE下降了11.7%（95%置信区间为-1.9%至23.9%）。作为对30分钟增长预测的补充，我们分析……（摘要原文在此截断）

    arXiv:2610.07209v1 Announce Type: cross  Abstract: Limited review teams must identify which emerging claims are likely to keep growing before their eventual reach is known. We center early misinformation triage on this continuation-forecasting problem: predicting subsequent recorded propagation-tree growth from the first 30 minutes of activity. On FibVID, we compare early node count with structural depth entropy, temporal arrival entropy, and their pair while keeping all propagation trees from each original claim in one partition. Across 352 test trees from 59 claim groups separate from training, the combined model raises $R^2$ for log-transformed future growth from 0.307 to 0.323 and reduces log-MAE by 2.4% (95% claim-bootstrap CI, -0.4% to 5.2%). The gain is especially pronounced among 97 high-activity trees: $R^2$ rises from 0.248 to 0.395, Spearman's $\rho$ from 0.394 to 0.529, and log-MAE falls by 11.7% (95% CI, -1.9% to 23.9%). Complementing the 30-minute growth forecast, we anal
    
[^119]: 负责任的院校分析：借助人工智能支持解读偏差

    Responsible Institutional Analytics: Interpreting Bias with AI Support

    [https://arxiv.org/abs/2610.07205](https://arxiv.org/abs/2610.07205)

    提出了FACTRIA框架，将院校分析中的潜在偏差因素组织为四个维度，并通过生成式AI聊天机器人引导用户反思这些因素，研究表明结构化框架与AI引导相结合能够促进更具情境意识的负责任解读。

    

    院校分析（Institutional Analytics, IA）仪表板为高等教育中的决策提供信息依据，然而数据局限性、分析技术的限制以及情境信息的缺失常常影响对其结果的解读。为了支持更负责任的IA解读，我们提出了FACTRIA框架，该框架将潜在的偏差因素组织为四个方面：分析流程、院校情境、课程层面特征以及人口统计特征。我们将FACTRIA框架作为生成式AI聊天机器人的输入，该机器人旨在提示用户在分析IA时对这些因素进行反思。一项基于四个真实IA案例、由利益相关者参与的定性研究以及转移网络分析表明，该聊天机器人促使参与者认识到被忽视的因素如何影响了他们的初始解读。研究结果表明，将结构化框架与基于AI的引导相结合，能够增强具有情境意识的、负责任的数据解读。

    arXiv:2610.07205v1 Announce Type: cross  Abstract: Institutional Analytics (IA) dashboards inform decision-making in higher education, yet data limitations, constraints in analytical techniques, and missing contextual information often affect their interpretation. To support more responsible interpretation of IA, we introduce FACTRIA, a framework that organizes potential biasing factors across four areas: the analytics pipeline, institutional context, course-level characteristics, and demographics. We used the FACTRIA framework as input to a generative-AI chatbot designed to prompt users to reflect on these factors while analyzing IA. A qualitative study with stakeholders, drawing on four authentic IA cases, and a transition network analysis showed that the chatbot prompted participants to recognize how overlooked factors influenced their initial interpretation. Findings indicated that combining a structured framework with AI-based guidance can enhance context-aware, responsible interp
    
[^120]: 从内部识别内省

    Identifying Introspection From the Inside

    [https://arxiv.org/abs/2610.07186](https://arxiv.org/abs/2610.07186)

    该研究发现模型在隐式决策任务上持续微调后会涌现出对所学偏好的准确自我报告，且伴随偏好表征向更早层转移的结构性变化，为区分真实内省与虚构提供了机制性标志。

    

    大型语言模型会做出关于自身的声明，这些声明既影响重大，又越来越难以仅凭行为加以验证。我们如何区分貌似合理的虚构与真实的内省？在本文中，我们在受控环境中识别了忠实自我报告的机制性特征。利用低秩适配器，我们训练模型根据潜在的线性偏好函数代表虚构角色做出决策。我们发现，在隐式决策任务上的持续微调可以促使模型涌现出对其所学偏好的准确自我报告，即使没有显式的自我报告监督。我们针对这一涌现现象提出了两个研究问题。第一：准确自我报告的涌现是否伴随着模型中可测量的结构性变化？权重消融实验与冻结层实验共同表明，偏好表征在训练过程中向更早的层转移……

    arXiv:2610.07186v1 Announce Type: new  Abstract: Large language models make claims about themselves that are both consequential and increasingly difficult to verify from behavior alone. How can we distinguish plausible confabulations from genuine introspection? In this paper, we identify mechanistic signatures of faithful self-report in a controlled setting. Using low-rank adapters, we train models to make decisions on behalf of fictitious characters, according to latent linear preference functions. We find sustained fine-tuning on an implicit decision task can lead to the emergence of accurate self-reporting of models' learned preferences, even without explicit self-report supervision. We ask two research questions about this emergent phenomenon. First: is the emergence of accurate self-reporting accompanied by a measurable structural change in the model? Weight ablations and frozen-layer experiments together indicate that preference representations shift to earlier layers over traini
    
[^121]: 从人类科研决策轨迹中学习科学探索

    Learning Scientific Exploration from Human Research Decision Trajectories

    [https://arxiv.org/abs/2610.07184](https://arxiv.org/abs/2610.07184)

    本文提出ResearchTrails数据集，以Git仓库的提交历史作为人类科研探索过程的代理，并开发自动化流水线从中提取结构化的研究决策轨迹，弥补了现有科学语料库只记录最终成果、缺乏探索过程信息的不足。

    

    构建面向科学研究的AI系统的一个关键挑战，是使系统具备“科学探索”能力：即通过一系列研究决策与行动，系统性地调查未知现象或想法以获取新知识的过程。然而，这一过程在现有科学语料库中基本缺失；例如，研究论文主要记录的是最终成果，而非产生这些成果的探索轨迹。在这项工作中，我们提出了ResearchTrails，一个从Git仓库构建的人类科研轨迹数据集，其中提交历史被用作科研探索过程的代理。我们开发了一个自动化且可扩展的流水线，从仓库提交记录中提取结构化的研究轨迹，捕捉方法、实验和消融实验的连续演变。我们对所构建的数据集进行了特征刻画，并表明这些轨迹中包含关于中间研究过程的有意义信号……（摘要原文在此处截断）

    arXiv:2610.07184v1 Announce Type: cross  Abstract: A key challenge in building AI systems for scientific research is enabling $\textit{scientific exploration}$: the systematic process of investigating unknown phenomena or ideas to gain new knowledge through sequences of research decisions and actions. Yet this process is largely missing from existing scientific corpora; for example, research papers primarily record final outcomes rather than the trajectories that produced them. In this work, we introduce $\textbf{ResearchTrails}$, a dataset of $\textbf{human research trajectories constructed from Git repositories}$, where $\textbf{commit histories}$ serve as proxies for research exploration. We develop an automated and scalable pipeline that extracts structured research trajectories from repository commits, capturing successive changes to methods, experiments, and ablations. We characterize the resulting dataset and show that these trajectories contain meaningful signals about intermed
    
[^122]: CLM作为裁判：在公开裁判基准上评估一个开放对比决策模型

    CLM-as-a-Judge: Evaluating an Open Contrastive Decision Model on Public Judge Benchmarks

    [https://arxiv.org/abs/2610.07177](https://arxiv.org/abs/2610.07177)

    该论文首次系统评估了开放对比决策模型 CLM-v0.1-8B 作为裁判的能力，发现其在公开基准上接近随机水平且显著落后于同规模奖励模型和生成式裁判，但通过单参数温度校准可将其置信度修复至良好校准状态。

    

    一个开放的对比决策模型在困难的公开基准上作为裁判的表现接近随机水平：Contrastive-LM/CLM-v0.1-8B 的得分介于 0.351（四选一，随机水平 0.250）和 0.593（成对比较，随机水平 0.500）之间，在 RM-Bench 和 JudgeBench 上与抛硬币在统计上无显著差异，并且在 HaluEval 的每个条目上都用同一个恒定标签回答，与 0.581 的平凡“总是选第一个”基线持平。相同参数量的裁判模型在各处得分都高得多：一个奖励模型达到 0.764 到 0.976，一个生成式裁判达到 0.611 到 0.778，且经 Benjamini-Hochberg 校正后，与 CLM 的每一项差距都显著。有两个特性是有效的。原始置信度过于自信，偏差高达 +0.401，但只需在留出的校准数据上拟合一个池化温度参数，就能将期望校准误差（ECE）修复至不超过 0.062，且修复后的置信度在六个基准中的三个上对模型自身错误的排序能力高于随机水平。决策顺序翻转率为 0.0002（原文此处截断）。

    arXiv:2610.07177v1 Announce Type: cross  Abstract: An open contrastive decision model is near chance as a judge on the hard public benchmarks: Contrastive-LM/CLM-v0.1-8B scores between 0.351 (best- of-four, chance 0.250) and 0.593 (pairwise, chance 0.500), is statistically indistinguishable from coin flipping on RM-Bench and JudgeBench, and answers every HaluEval item with one constant label, matching the trivial always-first baseline at 0.581. Judges with the same parameter count score far higher everywhere: a reward model reaches 0.764 to 0.976 and a generative judge 0.611 to 0.778, and every gap to CLM is significant after Benjamini-Hochberg correction. Two properties do work. Raw confidences are overconfident by up to +0.401, yet one pooled temperature fit on held-out calibration items repairs expected calibration error to at most 0.062, and the repaired confidence ranks the model's own errors above chance on three of six benchmarks. The decision order-flip rate is 0.0002 against 0
    
[^123]: 语言模型中柏拉图式表示的理论

    A theory of platonic representations in language models

    [https://arxiv.org/abs/2610.07168](https://arxiv.org/abs/2610.07168)

    本文通过假设数据具有隐藏的层级结构（抽象层次跨语言共享、表面层次为语言或模态特定），并借助概率上下文无关文法与信念传播理论推导出分析性预测，首次从理论上解释了多语言模型中间层出现柏拉图式表示的现象及其随语言相近程度和模型质量增强的规律。

    

    在多语言语言模型的内层中，翻译句子的表示是相似的——这一观察与柏拉图表示假说相关联，但在理论上尚未得到解释。我们基于以下假设提供了相关解释：数据具有隐藏的层级结构，其抽象层次在语言之间共享，而表面层次则是模态或语言特定的。具体而言，我们从概率上下文无关文法生成合成语言，这些文法共享上层产生式规则但不共享下层产生式规则。在此设定下，贝叶斯最优的下一词预测器是信念传播（BP）；将它的消息编码到连续的层中可以产生分析性预测，这些预测与在同一数据上训练的transformer高度吻合。该框架解释了为什么跨语言相似性在中间层达到峰值、与语言特定结构共存，并随语言相近程度、模型质量和数据而增强。

    arXiv:2610.07168v1 Announce Type: cross  Abstract: Representations of translated sentences are similar in the inner layers of multilingual language models -- an observation connected to the platonic representation hypothesis, yet unexplained theoretically. We provide an explanation based on the assumption that data have a hidden hierarchical structure whose abstract levels are shared across languages while surface levels are modality- or language-specific. Concretely, we generate synthetic languages from probabilistic context-free grammars sharing upper-level but not lower-level production rules. In this setting the Bayes-optimal next-token predictor is belief propagation (BP); encoding its messages in successive layers yields analytical predictions that agree well with transformers trained on the same data. The framework explains why cross-lingual similarity peaks in middle layers, coexists with language-specific structure, and strengthens with language proximity, model quality and da
    
[^124]: CroissantMiner：面向机器学习数据集的Croissant元数据自动提取与验证

    CroissantMiner: Automated Extraction and Validation of Croissant Metadata for ML Datasets

    [https://arxiv.org/abs/2610.07132](https://arxiv.org/abs/2610.07132)

    该论文提出了首个针对Croissant元数据提取的端到端评估基准（包含602篇论文的金/银双级标注），并发现单次提取方法在各类模型骨干上始终优于四种智能体架构。

    

    Croissant已成为机器可读数据集元数据的标准，然而填充其字段仍然是一项劳动密集型工作，需要仔细阅读数据集随附的文档。我们提出了首个能够对照社区标准模式进行端到端元数据提取评估的基准。该基准包含602篇论文，其中102篇带有经人工验证的金标准标注，500篇带有大语言模型生成的银标准标注，覆盖完整的Croissant模式，包括核心字段和负责任AI（RAI）字段。基于该基准，我们在一个两层评估框架下评估了一系列提取系统，涵盖前沿模型、开源权重模型和智能体架构，该框架结合了基于规则的评分与经人工审核选定的LLM裁判。我们发现，单次提取始终优于我们评估的四种智能体架构：在不同骨干模型上，这些分解式变体取得了较低的……（原文在此处截断）

    arXiv:2610.07132v1 Announce Type: cross  Abstract: Croissant has emerged as a standard for machine-readable dataset metadata, yet populating its fields remains labor-intensive and requires careful reading of accompanying dataset documentation. We present the first benchmark enabling end-to-end evaluation of metadata extraction aligned with a community-standard schema. The benchmark comprises 602 papers, including 102 with human-validated gold annotations and 500 with LLM-generated silver annotations, covering the full Croissant schema with both core and Responsible AI (RAI) fields. Using this benchmark, we evaluate a range of extraction systems spanning frontier models, open-weight models, and agentic architectures, under a two-tier evaluation framework that combines rule-based scoring with an LLM judge selected via human audit. We find that single-pass extraction consistently outperforms the four agentic architectures we evaluate: across backbones, these decomposed variants achieve lo
    
[^125]: 通过随机嵌入扰动越狱开放权重大语言模型

    Jailbreaking Open-Weight LLMs via Random Embedding Perturbations

    [https://arxiv.org/abs/2610.07125](https://arxiv.org/abs/2610.07125)

    该论文提出PEV攻击方法，仅需在提示的嵌入向量中反复添加随机高斯噪声即可越狱多种规模的开放权重大语言模型，暴露了此类模型的安全脆弱性。

    

    尽管开放权重模型在能力上不断进步并被多个领域广泛采用，但其安全性仍然是一个重要问题。模型的一个关键特性是能够拒绝或规避有害、恶意或不当的提示。在本文中，我们揭示了六种不同规模的常见开放权重大语言模型的安全漏洞，这些漏洞在JailbreakBench基准数据集上能够持续诱导出有害或不安全的响应。我们提出的攻击方法——扰动嵌入向量（PEV），是一种简单快速的“越狱”技术，比以往的方法成本更低，后者通常需要梯度计算、针对每个提示的优化或修改模型内部权重。PEV只需在提示的嵌入向量表示中添加独立的高斯噪声，无需其他额外操作。为了生成不安全的响应，我们反复从该分布中采样加性噪声。在实验中，

    arXiv:2610.07125v1 Announce Type: cross  Abstract: While open-weight models have enjoyed steady progress in capabilities and wide adoption across multiple domains, their safety remains an important concern. One key feature is the ability to refuse or deflect harmful, malicious, or insensitive prompts. In this paper, we expose safety vulnerabilities across six common open-weight LLMs of various sizes that consistently lead to harmful or unsafe responses on the JailbreakBench benchmark dataset. Our proposed attack, Perturbed Embedding Vector (PEV), is a simple and fast "jailbreaking" technique that is cheaper than prior approaches, which typically require gradient computations, per-prompt optimizations, or altering internal weights of the models. PEV just adds independent Gaussian noise in the embedding vector representations of the prompt, with no need for further manipulations. To generate unsafe responses, we repeatedly sample additive noise from this distribution. In our experiments,
    
[^126]: JudgeMoE：面向LLM作为评判者的分布聚合方法

    JudgeMoE: Distributional Aggregation for LLM-as-a-Judge

    [https://arxiv.org/abs/2610.07109](https://arxiv.org/abs/2610.07109)

    JudgeMoE是一种轻量级聚合器，通过为LLM评判者的评分分布分配样本特定权重并进行融合，保留了标量压缩中丢失的不确定性与分歧信息，在多个基准上显著提升了评判与人类判断的相关性。

    

    当LLM评判者对输出进行评分时，其评分分布中保留了不确定性和分歧信息，而这些信息在压缩为标量后会丢失。我们提出了JudgeMoE，这是一种轻量级聚合器，它为缓存的评判者评分分布分配针对具体样本的权重，并在计算最终分数之前将它们融合。一项协议研究表明，评分范围的选择在不同的评判者-数据集设置下并不稳定，且软评分通常优于硬解码。在原始的10单元基准测试中，JudgeMoE相比均匀对数池化方法将平均Spearman相关系数提升了+0.079。将相同配置应用于另外六个单元后，在16个单元中相比最强的本地单一评判者取得了+0.0393的平均增益，其中12/16个单元呈现正向差异，单侧Wilcoxon符号秩检验p=0.0091。基于验证集的进一步分析表明，首选的聚合方法取决于具体任务和评判者池。

    arXiv:2610.07109v1 Announce Type: new  Abstract: When an LLM judge scores an output, its score distribution retains uncertainty and disagreement information that is lost after scalar compression. We introduce JudgeMoE, a lightweight aggregator that assigns example-specific weights to cached judge score distributions and fuses them before computing a final score. A protocol study shows that score-range choice is unstable across judge--dataset settings and that soft scoring usually outperforms hard decoding. On the original 10-cell benchmark, JudgeMoE improves mean Spearman over uniform log pooling by $+0.079$. Applying the same configuration to six additional cells yields a $+0.0393$ mean gain over the strongest local single judge across 16 cells, with positive differences in 12/16 cells and a one-sided Wilcoxon signed-rank $p=0.0091$. Validation-based analyses further show that the preferred aggregation method depends on the task and judge pool.
    
[^127]: 面向生成式AI工作负载的智能内容摄取

    Smart Content Ingestion for Generative AI Workloads

    [https://arxiv.org/abs/2610.07091](https://arxiv.org/abs/2610.07091)

    本文提出智能内容摄取的理念，指出在生成式AI时代，由于企业知识以PDF、电子表格等异构格式承载多种信息模态，内容提取已演进为AI生命周期中独立且不可替代的关键阶段，其错误无法被下游检索或重排序组件修复。

    

    机器学习的演进逐步改变了智能在AI系统中的所在位置。在传统机器学习中，任务、数据表示、标签和模型架构紧密耦合，因此数据准备是狭窄的、受模式约束且过程可见的。生成式AI将模型与任何单一任务解耦：一个基础模型服务于开放式的下游任务，而模型端获得的通用性在数据端则对应着高度的异构性，因为企业知识是以人们日常使用的格式（PDF、演示文稿、电子表格、扫描文档、表单、表格、图表和混合布局文件）编写的，这些格式同时承载着文本、视觉、几何和结构信息。语言模型或检索器无法对在此接口处被错误表示的信息进行可靠推理，因此内容提取本身成为了一个独立的生命周期阶段，其产生的错误无法被任何下游检索器或重排序器所修复。

    arXiv:2610.07091v1 Announce Type: new  Abstract: The evolution of machine learning has progressively changed where intelligence resides in an AI system. In conventional machine learning the task, data representation, labels and model architecture were tightly coupled, so data preparation was narrow, schema-bound and visible. Generative AI decouples the model from any single task: one foundation model serves open-ended downstream tasks, and the generality gained on the model side is matched by heterogeneity on the data side, because enterprise knowledge is authored in the formats people use (PDF, presentations, spreadsheets, scanned documents, forms, tables, diagrams and mixed-layout files) that carry textual, visual, geometric and structural information at once. A language model or retriever cannot reason reliably over information misrepresented at this interface, so content extraction becomes a lifecycle stage in its own right whose errors no downstream retriever or re-ranker can repa
    
[^128]: Turnslide：通过遍历有限状态机实现可扩展的多轮数据合成

    Turnslide: Scalable Multi-Turn Data Synthesis by Walking a Finite-State Machine

    [https://arxiv.org/abs/2610.07070](https://arxiv.org/abs/2610.07070)

    提出Turnslide框架，将API建模为有限状态机并设定目标分布，以低成本、高可扩展性地合成多轮工具调用数据，且每个样本仅需单次LLM调用即可生成。

    

    小语言模型部署成本低，可以在私有基础设施上运行，但基座模型在多轮工具调用方面往往表现不佳，而对其进行微调所需的针对特定API的数据几乎不存在。现有的数据合成方法对于大规模微调而言成本过高，因为它们通常需要为不同领域搭建模拟运行环境，且每生成一轮对话都需要多次大语言模型调用。我们提出了一种完全自动化、轻量级的合成框架，该框架将每个API建模为有限状态机，把系统表示为决定各个工具何时可被调用的抽象状态，从而生成状态有效的工具调用序列；这些序列仅需单次大语言模型调用即可转化为完整的训练样本。我们不追求多样性最大化，而是针对对话轮数、工具调用序列和任务复杂度设定目标分布。我们通过在生成的轨迹上微调小语言模型来衡量数据质量，结果表明我们的有限状态机方法……

    arXiv:2610.07070v1 Announce Type: new  Abstract: Small language models are inexpensive to serve and can run on private infrastructure, but base models are often not good enough at multi-turn tool calling, and fine-tuning them needs per-API data that rarely exists. Existing synthesis methods are too expensive for high-scale fine-tuning, as they often require mock operational environments for different domains and multiple LLM calls per generated conversation turn. We introduce a fully automated, lightweight synthesis framework that models each API as a finite-state machine, representing the system as abstract states that determine when each tool may be called, producing state-valid sequences of tools; sequences are translated into complete examples with a single LLM call. Rather than optimize diversity, we set a target distribution over the number of turns, the tool sequence and task complexity. We measure data quality by fine-tuning SLMs on generated trajectories, showing that our FSM-
    
[^129]: 从宏观社会信号中学习模拟个体

    Learning to Simulate Individuals from Macro Social Signals

    [https://arxiv.org/abs/2610.07062](https://arxiv.org/abs/2610.07062)

    该论文提出macro2mind框架，将预测市场价格轨迹作为宏观监督信号，通过GRPO训练和社会行为分解，使大语言模型把行为推理作为显式预测步骤，从而从宏观数据中学会模拟个体对真实事件的反应。

    

    大语言模型越来越多地被用于模拟个体如何应对新情境，然而这些回应背后的行为推理要么继承自预训练，要么从个体级标注中学习，而个体级标注能提供的行为多样性有限，且几乎无法对推理过程本身进行监督。我们提出从预测市场中学习行为推理，预测市场的价格轨迹大规模地记录了人群对真实世界事件的反应。我们介绍了macro2mind，该方法利用市场信号通过GRPO训练语言模型。一种社会行为分解方法使行为推理成为预测的显式步骤：模型推断出具有代表性的市场参与者群体，预测每个群体如何解读新闻并更新其信念，推理它们之间的相互作用，并将这些反应聚合为价格。带有难度感知采样的后见之明遗憾课程将训练集中于……（原文在此处截断）

    arXiv:2610.07062v1 Announce Type: cross  Abstract: Large language models are increasingly used to simulate how individuals respond to new situations, yet the behavioral reasoning behind these responses is either inherited from pretraining or learned from individual-level annotations, which offer limited behavioral diversity and little supervision of the reasoning itself. We propose to learn behavioral reasoning from prediction markets, whose price trajectories record how populations respond to real-world events at scale. We introduce macro2mind, which trains a language model with GRPO using market signals. A social behavioral decomposition makes behavioral reasoning an explicit step of forecasting: the model infers representative groups of market participants, predicts how each interprets the news and updates its beliefs, reasons about their interactions, and aggregates these responses into a price. A hindsight-regret curriculum with difficulty-aware sampling focuses training on transi
    
[^130]: SEAL：面向高效语音分离的混合闭合加性重建与细化感知专家路由

    SEAL: Mixture-Closed Additive Reconstruction and Refinement-Aware Expert Routing for Efficient Speech Separation

    [https://arxiv.org/abs/2610.07047](https://arxiv.org/abs/2610.07047)

    SEAL通过混合闭合的零和加性残差重建与基于声学证据的稀疏专家路由机制，在参数减少28%、计算量降低2.9倍的情况下，语音分离性能超越TIGER达0.31 dB SI-SDRi。

    

    通过掩码混合信号并经由共享单元进行细化的紧凑型时频分离器面临两个限制。首先，有界的乘性掩码只能缩放混合频点，因此在重叠分量相互抵消的地方，估计值始终很小。其次，共享单元在每一步对每个时频标记应用相同的权重，因此扩大其规模会在所有地方增加计算量。我们提出SEAL（稀疏专家路由与加性潜在重建）来解决这两个问题。在重建方面，由局部混合幅度约束的零和加性残差使得在分量相互抵消的地方估计值可以非零，同时仍能与混合信号求和闭合。在路由方面，由声学证据和步间证据构建的查询将每个标记分配到六个残差专家之一，并通过范数上限防止步间线索覆盖清晰的声学证据。在EchoSet数据集上，SEAL（小版本）以减少28%的参数量和2.9倍更少的MACs计算量，超越TIGER（小版本）0.31 dB SI-SDRi。

    arXiv:2610.07047v1 Announce Type: cross  Abstract: Compact time-frequency separators that mask the mixture and refine through a shared cell face two limits. First, a bounded multiplicative mask only scales a mixture bin, so where overlapping components cancel, the estimate stays small. Second, a shared cell applies the same weights to every time-frequency token at every step, so enlarging it adds compute everywhere. We present SEAL (Sparse Expert routing with Additive Latent reconstruction) to address both. For reconstruction, a zero-sum additive residual bounded by the local mixture amplitude lets estimates be nonzero where components cancel yet still sum to the mixture. For routing, a query built from acoustic and inter-step evidence sends each token to one of six residual experts, and a norm cap keeps the step cue from overriding clear acoustic evidence. On EchoSet, SEAL (small) surpasses TIGER (small) by 0.31 dB SI-SDRi with 28% fewer parameters and 2.9 times fewer MACs, and SEAL (
    
[^131]: GIVE-KWS：面向噪声鲁棒的基于示例查询关键词检测的门控视觉证据注入方法

    GIVE-KWS: Gated Injection of Visual Evidence for Noise-Robust Query-by-Example Keyword Spotting

    [https://arxiv.org/abs/2610.07046](https://arxiv.org/abs/2610.07046)

    该论文提出GIVE-KWS，通过门控交叉注意力将唇部运动的视觉证据注入查询音频，并证明噪声鲁棒性需要同时具备含音素信息的视觉表示和注入式融合机制，从而在-10 dB噪声下相比掩蔽方式获得4.0-9.3 dB的有效信噪比增益。

    

    视觉语音有望实现噪声鲁棒的关键词检测，但视觉信息流并不一定被有效利用。在一个三模态的基于示例查询的关键词检测基准测试中，我们发现配备了任务训练视觉编码器的系统在-10 dB信噪比下的等错误率（EER）与文本加音频系统仅相差2个百分点以内，并将这一差距归因于视觉编码器缺乏音素信息。我们提出了GIVE-KWS，其融合阶段GIVE（门控视觉证据注入）通过门控交叉注意力机制将唇部运动信息对查询音频进行条件化。我们表明，视觉鲁棒性取决于两个相互作用的条件：包含音素信息的视觉表示，以及注入视觉证据而非仅对音频特征进行重新缩放的融合方式。在包含音素信息的编码器下，注入方式在-10 dB时相比掩蔽方式带来4.0-9.3 dB的有效信噪比增益，而在音素信息贫乏的编码器下，这一增益几乎消失。相对于基准系统，GIVE-KWS降低了……（原文摘要在此处截断）

    arXiv:2610.07046v1 Announce Type: cross  Abstract: Visual speech promises noise-robust keyword spotting, yet a visual stream is not necessarily used. On a tri-modal query-by-example keyword spotting (QbyE-KWS) benchmark, we find that a system with a task-trained visual encoder comes within 2 percentage points of a text-and-audio system in equal error rate (EER) at -10 dB, and link this gap to the encoder's lack of phonemic information. We present GIVE-KWS, whose fusion stage, GIVE (Gated Injection of Visual Evidence), conditions query audio on lip motion through gated cross-attention. We show that visual robustness depends on two interacting conditions: a phoneme-bearing visual representation, and fusion that injects visual evidence rather than rescaling audio features. Under a phoneme-bearing encoder, injection yields an effective SNR gain of 4.0-9.3 dB over masking at -10 dB, whereas under a phoneme-poor one it nearly vanishes. Relative to the benchmark system, GIVE-KWS reduces unsee
    
[^132]: 生物医学领域神经机器翻译的模型压缩研究

    Investigating Model Compression for Neural Machine Translation in the Biomedical Domain

    [https://arxiv.org/abs/2610.07032](https://arxiv.org/abs/2610.07032)

    本研究探讨了知识蒸馏和量化两种模型压缩技术在生物医学领域神经机器翻译中的应用，揭示了这两种技术在低资源专业领域条件下的局限性。

    

    大规模预训练Transformer模型已在包括多语言场景在内的多种机器翻译任务中取得了最先进的性能。知识蒸馏已成为一种可持续的模型压缩方法，它将知识从大型教师模型迁移到更小、更高效的学生模型中。类似地，量化——即降低模型权重和激活值的数值精度（例如从32位表示降至8位表示）——被广泛用于加速推理，使模型在部署时能够以数倍速度运行。然而，当这两种技术应用于专业领域数据时，尤其是在低资源条件下，都面临局限性。在知识蒸馏中，知识迁移的效果往往受到领域特定平行数据稀缺的制约；而量化则可能随着比特精度的降低而导致性能下降。在本工作中，我们研究了……（摘要内容被截断）

    arXiv:2610.07032v1 Announce Type: cross  Abstract: Large-scale pretrained transformer models have achieved state-of-the-art performance across diverse machine translation tasks, including multilingual settings. Knowledge distillation has emerged as a sustainable approach for model compression, transferring knowledge from large teacher models to smaller, more efficient student models. Similarly, quantization, which reduces the numerical precision of model weights and activations (e.g., from 32-bit to 8-bit representations) is widely used to accelerate inference, enabling models to run several times faster during deployment. However, both techniques face limitations when applied to specialized domain data, particularly under low-resource conditions. In knowledge distillation, the effectiveness of transfer is often constrained by the scarcity of domain-specific parallel data, while quantization can lead to performance degradation as bit precision decreases. In this work, we investigate th
    
[^133]: 超越拒绝模式：通过安全角色内化实现鲁棒且可泛化的大语言模型安全对齐

    Beyond Refusal Patterns: Safe-Role Internalization for Robust and Generalizable LLM Safety Alignment

    [https://arxiv.org/abs/2610.07023](https://arxiv.org/abs/2610.07023)

    提出SSRFT（监督安全角色微调）框架，首次将LLM安全对齐重新表述为对预定义安全角色的内化，通过构建SRQA数据集使模型内化安全价值观与原则，从而以更少的攻击特定监督实现更鲁棒、可泛化的安全对齐，并缓解过度拒绝问题。

    

    大型语言模型（LLM）已展现出卓越的能力，但仍易受越狱攻击的影响，这类攻击会诱使其产生有害或不安全的输出。现有的安全对齐方法，包括监督微调（SFT）和基于人类反馈的强化学习（RLHF），通常需要大量针对特定攻击的监督数据和计算资源，同时仍容易陷入浅层安全对齐和过度拒绝的问题。为应对这些挑战，我们提出了SSRFT（监督安全角色微调），这是首个将安全对齐重新表述为对预定义安全角色进行内化的框架。SSRFT基于心理测量问题、少量越狱提示词以及安全角色描述构建了安全角色问答（SRQA）数据集。通过合成、验证角色一致的回复并将其扩展至多样化场景，使模型能够内化以安全为导向的价值观和原则，而非显式的……

    arXiv:2610.07023v1 Announce Type: new  Abstract: Large Language Models (LLMs) have achieved remarkable capabilities but remain vulnerable to jailbreak attacks that elicit harmful or unsafe outputs. Existing safety alignment approaches, including Supervised Fine-Tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF), often require substantial attack-specific supervision and computational resources, while remaining susceptible to shallow safety alignment and over-refusal. To address these challenges, we introduce SSRFT(Supervised Safe-Role Fine-Tuning), the first framework that reformulates safety alignment as the internalization of a predefined safe role. SSRFT constructs a Safe-Role Question-Answer (SRQA) dataset from psychometric questions, limited jailbreak prompts, and a safe-role description. Role-consistent responses are synthesized, validated, and expanded into diverse scenarios, enabling models to internalize safety-oriented values and principles rather than explicit
    
[^134]: 来自40亿参数开放模型的关于随机试验的校准答案：一项预注册测试与无许可证问题的发布

    Calibrated Answers About Randomized Trials From a 4-Billion-Parameter Open Model: A Registered Test and a License-Clean Release

    [https://arxiv.org/abs/2610.07019](https://arxiv.org/abs/2610.07019)

    该论文发布了一个仅使用许可证允许复用的文章微调的 40 亿参数开放模型 Fiorillo v0.5，它能够以良好校准的概率回答随机试验中干预措施对结局影响的问题，并通过预注册的四项标准验证后正式发布。

    

    Fiorillo v0.5 是一个开放模型，它以概率形式回答类型化问题。其主要专家模块读取一篇随机试验的文章（截断至6,144个标记），并回答某项干预措施相对于对照是否显著增加、显著降低或未显著改变某一结局（基于 Evidence Inference 2.0，EI 数据集）。该模型基于 Qwen3-4B-Base，配备低秩适配器和决策头，仅在 2,657 篇训练文章中许可证自身允许复用的 1,431 篇上针对 EI 进行微调。在该版本的测试预测之前，研究者在开放科学框架（OSF）上预先注册了四项发布标准，其中第二项标准在标签公开的 EI 测试集上进行评判。在该测试集（333 篇文章中的 1,218 个提示）上，预期校准误差为 0.0168，低于 0.05 的限值；对数损失比先验基线低 0.8603（95% 区间为 0.8104 至 0.9078），并且在读取相同输入的情况下，比 Gemma 4 31B-it 低约 0.1（摘要在此处截断）。

    arXiv:2610.07019v1 Announce Type: new  Abstract: Fiorillo v0.5 is an open model that answers typed questions with a probability for each answer. Its main specialist reads a randomized trial's article, cut to 6,144 tokens, and answers whether an intervention significantly increased, significantly decreased or did not significantly change an outcome against a comparator (Evidence Inference 2.0, EI). It is Qwen3-4B-Base with low-rank adapters and a decision head, fine-tuned for EI only on the 1,431 of 2,657 training articles whose own license allows reuse. Four criteria registered on the Open Science Framework before this version's test predictions decided its release, the second bar judged on EI's test split, whose labels are public. On that split (1,218 prompts in 333 articles), the expected calibration error was 0.0168 against a limit of 0.05; log loss was below the prior's by 0.8603 (95 percent interval 0.8104 to 0.9078) and below that of Gemma 4 31B-it, reading the same input, by 0.1
    
[^135]: 分块扩散语言模型中的掩码引导KV缓存淘汰

    Mask-Guided KV Cache Eviction in Block Diffusion Language Models

    [https://arxiv.org/abs/2610.06996](https://arxiv.org/abs/2610.06996)

    提出无需训练的MaskAhead方法，通过统一的掩码-查询排序机制同时解决分块扩散语言模型中KV缓存的选择与淘汰问题，其量化变体Q-MaskAhead可在低比特KV上直接计算，从而降低内存占用并加速生成。

    

    分块扩散语言模型在整个生成过程中都维护着一个庞大的键值（KV）缓存，并在每个去噪步骤中都对其执行注意力操作，这同时限制了内存容量和生成速度。要降低这些开销，需要决定哪些过去的token用于当前块的去噪（选择），以及哪些token保留在内存中供未来的块使用（淘汰）。我们提出MaskAhead，这是一种无需训练的方法，通过单一的基于掩码-查询的排序机制同时解决这两个任务。当前块的掩码用于指导选择，而对即将到来的被掩码块的探测则用于指导淘汰，两者都根据KV条目对注意力输出的估计贡献进行排序。我们的量化变体Q-MaskAhead直接从低比特KV中计算选择和注意力操作，在很大程度上保留了被选中的条目。在Fast-dLLM-v2、DreamReasoner和LLaDA2.0-mini上的实验涵盖了长生成推理、长提示词问答以及大海捞针检索等任务。在长提示……

    arXiv:2610.06996v1 Announce Type: cross  Abstract: Block diffusion language models keep a large key-value (KV) cache throughout generation and attend to it at every denoising step, limiting both memory capacity and generation speed. Reducing these costs requires deciding which past tokens to use for denoising the current block (selection) and which to keep in memory for future blocks (eviction). We propose MaskAhead, a training-free method that solves both tasks with a single mask-query-based ranking mechanism. Current-block masks guide selection, while probes of upcoming masked blocks guide eviction. Both rank KV entries by their estimated contribution to the attention output. Our quantized variant, Q-MaskAhead, computes selection and attention directly from low-bit KV, largely preserving the selected entries. Experiments on Fast-dLLM-v2, DreamReasoner, and LLaDA2.0-mini cover long-generation reasoning, long-prompt question answering, and needle-in-a-haystack retrieval. On long-prompt
    
[^136]: AegisFlow：面向脆弱数据生态系统的自主修复与自愈的多智能体Agentic AI框架

    AegisFlow: A Multi-Agent Agentic AI Framework for Autonomous Remediation and Self-Healing in Fragile Data Ecosystems

    [https://arxiv.org/abs/2610.06971](https://arxiv.org/abs/2610.06971)

    AegisFlow是一个多智能体AI框架，通过Watchdog智能体收集运行时遥测、Repair智能体基于LLM自动生成并部署代码补丁，并采用基于MAPE-K循环的“并行影子补丁”非侵入式模型在数字孪生环境中验证补丁，从而实现脆弱数据管道从故障检测到自主修复的闭环自愈。

    

    传统数据管道以脆弱著称，常常由于上游模式漂移、API契约变更或网站DOM修改而发生故障。现有的可观测性工具只会发出警报并交由人类工程师处理，导致平均修复时间（MTTR）居高不下以及运维疲劳。本文提出了AegisFlow（用于智能自愈与图驱动工作负载修复运维的智能体引擎），这是一种新颖的智能体框架，能够闭合从检测到解决的完整闭环。AegisFlow使用一个Watchdog（看门狗）智能体来收集运行时遥测数据，并配备一个Repair（修复）智能体，基于大语言模型（LLM）自动创建、测试和部署代码补丁。该框架提出了一种名为并行影子补丁的非侵入式执行模型，这是一种基于监控-分析-计划-执行-知识（MAPE-K）循环的非侵入式执行模型，用于在数字孪生环境中生成和验证补丁。通过实验……（原文摘要在此处被截断）

    arXiv:2610.06971v1 Announce Type: new  Abstract: Traditional data pipelines are notoriously brittle, often failing due to upstream schema drift, API contract changes, or website DOM modifications. Present observability tools only raise alerts but for human engineers, resulting in a high Mean Time to Repair (MTTR) and operational fatigue. In this paper we propose AegisFlow (Agentic Engine for Intelligent Self-healing and Graph-driven Operations for Workload remediation), a novel agentic framework that closes the loop between detection and resolution. AegisFlow uses a Watchdog agent to collect runtime telemetry and has a Repair agent to automatically create, test and deploy code patches based on Large Language Models (LLMs). The framework presents the non-intrusive execution model called Parallel Shadow Patching, a non-intrusive execution model based on the Monitor, Analyze, Plan, Execute, Knowledge (MAPE-K) loop to generate and verify patches in digital twin environments. Through experi
    
[^137]: WavePrune：对RoPE来说，一个周期通常就够了

    WavePrune: One period is often enough for RoPE

    [https://arxiv.org/abs/2610.06963](https://arxiv.org/abs/2610.06963)

    提出WavePrune方法，通过将RoPE每个通道限制在其首个旋转周期内来消除位置混叠问题，无需额外调整即可提升多个模型的长上下文性能（如Qwen3-8B的HELMET分数从35.7升至40.0）。

    

    旋转位置编码通过以特定于通道的频率旋转查询和键向量的每个二维通道来编码token位置，使得注意力得分对位置的共同平移保持不变。然而，这种旋转是周期性的，会导致位置混叠，即相隔一个完整旋转周期的相对位置变得难以区分。为了解决这个问题，我们提出了WavePrune，它将每个通道限制在其第一个旋转周期内。我们证明该方法消除了注意力图中由位置混叠造成的干扰，并提升了整体长上下文性能。具体来说，WavePrune在我们测试的五个模型中的四个上，无需任何额外调整就提高了HELMET分数（例如，Qwen3-8B从35.7提升到40.0）。在从头预训练模型时，WavePrune在外推长度上也比不使用该方法的预训练取得了更低的验证损失。由于WavePrune将每个通道限制在滑动……

    arXiv:2610.06963v1 Announce Type: new  Abstract: Rotary Position Embedding (RoPE) encodes token positions by rotating each two-dimensional channel of the query and key vectors at a channel-specific frequency, making the attention logits invariant to a common shift of positions. However, this rotation is periodic, and it leads to position aliasing where relative positions separated by a full rotation period become hard to tell apart. To address this, we propose WavePrune, which restricts each channel to its first rotation period. We show that it removes the distractions in attention maps created by position aliasing and improves overall long-context performance. Specifically, WavePrune raises the HELMET score on four of five models we test without any extra tuning (e.g., 35.7 -> 40.0 on Qwen3-8B). When pretraining models from scratch, WavePrune also achieves lower validation loss at extrapolated lengths than pretraining without it. Because WavePrune restricts each channel to a sliding w
    
[^138]: 无标注证据下的判定：证据恢复应采用拒绝采样还是仅标签后训练？

    Verdicts Without Annotated Evidence: Rejection Sampling or Label-Only Post-Training for Evidence Recovery?

    [https://arxiv.org/abs/2610.06962](https://arxiv.org/abs/2610.06962)

    该研究表明，在没有任何人工证据标注的情况下，仅用判定标签进行后训练的小语言模型在证据恢复上优于基于自动来源接地分数的拒绝采样方法，且判定准确率与证据跨度一致性仅弱相关，说明准确率不能作为引用可审阅性的可靠指标。

    

    在许多审阅工作流程中，判定结果是唯一被保留的信息。其背后的文本段落并未被标注，因为这类标注的成本远高于记录决策本身。我们测量了小语言模型在仅以判定结果进行后训练时（任何阶段都没有人工证据标注）能够恢复多少此类证据。在ContractNLI数据集上，人工证据跨度在评估之前一直被保留。匹配记录的判定与认同这些证据跨度并非同一回事：在六个系统中，这两个分数仅呈弱相关，且对系统的排名也不同，因此当引用必须可供审阅时，准确率是一个糟糕的指导指标。仅对裸判定结果进行标签训练可达到准确率0.896和跨度F1 0.564；拒绝采样（仅当生成轨迹的判定与记录匹配时才保留该轨迹，再依据自动的来源接地分数选出其中一条）达到0.797和0.556，而训练前的基线为0.747和0.493。

    arXiv:2610.06962v1 Announce Type: cross  Abstract: In many review workflows the verdict is the only thing retained. The passages behind it are not marked, because that annotation costs far more than recording the decision. We measure how much of that evidence a small language model can recover when it is post-trained on the verdicts alone, with no human evidence labels at any stage. On ContractNLI the human evidence spans are held out until evaluation. Matching the recorded verdict and agreeing with those spans are not the same thing: across six systems the two scores are only weakly related and rank the systems differently, so accuracy is a poor guide when the citations have to be reviewable. Label-only training on the bare verdict reaches accuracy 0.896 and span F1 0.564. Rejection sampling, which keeps a generated trace only when its verdict matches the record and then picks one by an automatic source-grounding score, reaches 0.797 and 0.556, against 0.747 and 0.493 before training.
    
[^139]: EMODE：面向情感感知语音语言建模的动态副语义专家

    EMODE: Dynamic Para-Semantic Experts for Emotion-Aware Speech Language Modeling

    [https://arxiv.org/abs/2610.06956](https://arxiv.org/abs/2610.06956)

    EMODE 提出动态副语义专家（DPSE）架构，将语音特征分解为语义与副语言双通路并进行动态路由融合，结合三阶段训练课程，使语音语言模型能够有效保留情感等副语言信息。

    

    大型语音语言模型在统一的跨模态理解与生成方面已展现出强大能力，但副语言线索（尤其是情感）仍然难以保留。现有系统通常依赖于纠缠在一起的声学表示，这导致底层语言模型过度依赖恢复出的词汇内容，而非将其行为建立在声学-韵律证据之上。我们通过 EMODE 来解决这一局限，EMODE 是一个围绕动态副语义专家构建的情感感知语音语言模型。DPSE 将连续的语音特征分解为语义通路和副语言通路，对其进行动态路由，并在整合进语言模型之前将二者融合。为了将这种结构上的分解转化为功能上的专门化，EMODE 采用由语义预热、副语言激活和联合精炼组成的三阶段课程进行训练，并由正交……（摘要在此处被截断）

    arXiv:2610.06956v1 Announce Type: new  Abstract: Large speech language models have demonstrated strong capabilities in unified cross-modal understanding and generation, yet paralinguistic cues, especially emotion, remain difficult to preserve. Existing systems typically rely on entangled acoustic representations, which allow the underlying language model to depend excessively on recovered lexical content instead of grounding its behavior in acoustic-prosodic evidence. We address this limitation with EMODE, an emotion-aware speech language model built around \textbf{Dynamic Para-Semantic Experts (DPSE)}. DPSE decomposes continuous speech features into semantic and paralinguistic pathways, routes them dynamically, and fuses them before integration into the language model. To turn this structural decomposition into functional specialization, EMODE is trained with a three-stage curriculum consisting of semantic warm-up, paralinguistic activation, and joint refinement, guided by Orthogonal 
    
[^140]: 通过条件锚定蒸馏实现语言模型在持续学习中的稳定性

    Stabilizing language models under continual learning via condition-anchored distillation

    [https://arxiv.org/abs/2610.06940](https://arxiv.org/abs/2610.06940)

    该论文提出条件锚定生成蒸馏（CAGD）方法，通过保留少量旧提示并利用冻结的旧模型作为教师来匹配预测分布，从而在语言模型持续学习新任务时稳定其对旧任务的输出行为，并为自回归生成提供精确的序列散度链式法则分解、为掩码扩散建模提供局部去噪漂移的直接控制。

    

    语言模型的持续适应会改变其在先前学习过的提示词上的输出分布，而保留每一个旧的“提示-回答”对可能是不理想的或不可行的。我们研究了条件锚定生成蒸馏（CAGD）：保留一小组旧的提示词，使用冻结的旧模型来重建补全内容和生成状态，并在学习下一个任务的同时匹配其预测分布。这一公式化方法将普通重放方法所混淆的三种角色分离开来：条件用于选择要保护的行为，教师生成的结果用于定位相关状态，而软目标则规定预测可以如何变化。对于自回归语言生成，教师推演蒸馏可以对序列散度进行精确的链式法则分解。对于掩码扩散语言建模，我们的实现直接控制教师生成补全内容上的局部去噪漂移。在一个219M参数的掩码扩散（模型的持续适应中……注：原文摘要在此处被截断）

    arXiv:2610.06940v1 Announce Type: new  Abstract: Continual adaptation of language models can change their output distribution on prompts learned earlier, while retaining every old prompt-answer pair may be undesirable or impossible. We study condition-anchored generative distillation (CAGD): retain a small set of old prompts, use a frozen previous model to reconstruct completions and generation states, and match its predictive distributions while learning the next task. The formulation separates three roles that ordinary replay conflates: conditions select the behavior to protect, teacher generations locate relevant states, and soft targets specify how predictions may change. For autoregressive language generation, teacher-rollout distillation admits an exact chain-rule decomposition of sequence divergence. For masked-diffusion language modeling, our implementation directly controls local denoising drift on teacher-generated completions. In continual adaptation of a 219M masked diffusi
    
[^141]: Transformer拒绝机制中的组件与维度稀疏性

    Component and Dimension Sparsity in Transformer Refusal Mechanisms

    [https://arxiv.org/abs/2610.06903](https://arxiv.org/abs/2610.06903)

    该研究通过对四个开源大语言模型的组件级干预分析，发现拒绝行为引导只需稀疏组件子集（占上游组件28%–48%）及其中约50%的残差流维度即可复现完整效果，揭示了拒绝机制在组件和维度两个层面上的稀疏性。

    

    激活引导通过干预大语言模型的内部激活来操纵其行为，但这些干预的机理基础仍知之甚少。我们将拒绝引导分解为跨四个开源权重模型的组件级干预，识别出稀疏的注意力与MLP组件子集，仅对这些子集进行引导就足以复现完整的行为效果。我们发现，拒绝方向集中于稀疏的组件机制中，这些组件仅占上游组件的28%–48%，却能保留88%–101%的引导有效性。在这些机制内部，有效引导进一步集中于约50%的残差流维度，保留85%–98%的组件机制基线效果，这与特权基结构相一致。因此，稀疏性在两个层面上发挥作用：哪些组件被引导，以及这些组件内部哪些维度承载信号。总之，这些发现表明……

    arXiv:2610.06903v1 Announce Type: cross  Abstract: Activation steering manipulates large language model behavior by intervening on internal activations, but the mechanistic basis of these interventions remains poorly understood. We decompose refusal steering into component-level interventions across four open-weight models, identifying the sparse subsets of attention and MLP components whose steering suffices to reproduce the full behavioral effect. We find that refusal directions concentrate in sparse component mechanisms comprising 28--48\% of upstream components, retaining 88--101\% of steering effectiveness. Within these mechanisms, effective steering further concentrates in approximately 50\% of residual stream dimensions, retaining 85--98\% of the component-mechanism baseline, consistent with a privileged basis structure. Sparsity thus operates at two levels: which components are steered, and which dimensions within those components carry the signal. Together these findings show 
    
[^142]: 无需LLM摘要的树导航：面向长文档问答的分层检索等成本研究

    Tree Navigation Without LLM Summaries: A Matched-Cost Study of Hierarchical Retrieval for Long-Document QA

    [https://arxiv.org/abs/2610.06902](https://arxiv.org/abs/2610.06902)

    该论文提出NavTree，证明长文档问答中RAPTOR式摘要树的主要收益来自树的导航结构而非LLM生成的摘要内容，该方法在索引阶段零LLM调用，仅用确定性平衡线段树作为导航支架即可实现有效的分层检索。

    

    检索增强生成通过外部上下文为语言模型提供依据，但对于长文档，扁平的top-k检索可能会聚集在单一区域，从而遗漏互补证据。RAPTOR风格的摘要树通过在索引时递归聚类文本块并使用语言模型为每个聚类生成摘要，然后在查询时将摘要节点与原始文本块一起排序，来解决这一问题。我们证明，在长文档问答中，摘要树的主要收益可以来自导航本身，而非生成的摘要内容。我们提出了NavTree，一种仅使用叶子节点的检索器，它在文本块之上构建确定性的平衡线段树（索引阶段零语言模型调用），并将树纯粹用作导航支架：一种混合词法与稠密向量的前沿遍历方法，以检索得到的顶部叶子节点为锚点，从根节点向下行进，仅向阅读器输出叶子文本块。在与扁平检索器和抽取式重新实现的等成本评估中……

    arXiv:2610.06902v1 Announce Type: new  Abstract: Retrieval-augmented generation grounds language models in external context, but for long documents flat top-$k$ retrieval can cluster on a single region and miss complementary evidence. RAPTOR-style summary trees address this by recursively clustering chunks and using a language model to summarize each cluster at indexing time, then ranking summary nodes alongside raw chunks at query time. We show the main benefit of summary trees in long-document QA can come from navigation rather than the generated summary content. We introduce NavTree, a leaves-only retriever that builds a deterministic balanced segment tree over chunks (zero language-model calls at indexing) and uses the tree purely as a navigation scaffold: a hybrid lexical-and-dense frontier walk, anchored on top retrieved leaves, descends from the root and emits only leaf chunks to the reader. On a matched-cost evaluation against flat retrievers and an extractive re-implementation
    
[^143]: 容量、响应性与对齐性：是什么让潜在结构变得可操作

    Capacity, Responsiveness and Alignment: What Makes a Latent Structure Actionable

    [https://arxiv.org/abs/2610.06897](https://arxiv.org/abs/2610.06897)

    该论文将语言模型中潜在结构的因果影响力分解为容量、响应性和对齐性三个可解释且独立的约束因素，并通过跨4个模型家族、50个概念的实验证明，只有三者同时处于高水平时，该结构才真正具有因果可操作性。

    

    在语言模型（LM）的激活空间中定位潜在结构，对于理解和控制模型行为至关重要。然而，被定位出的结构在因果影响力上可能存在显著差异，这就引出了一个核心问题：是什么让一个结构具有可操作性？我们通过将因果影响力分解为三个因素的乘积来回答这一问题，并通过实证研究表明，这三个因素是可解释且彼此独立的约束条件：容量，衡量模型输出对沿该结构方向移动的敏感程度；响应性，刻画在给定当前上下文时该概念被激发（促进）的程度；对齐性，反映该结构与概念在特定上下文中的表征之间的吻合程度。在4个语言模型家族和50个概念上的实验中，我们观察到因果有效性要求所有因素都处于较高水平：低容量和低响应性会分别使因果有效性降低84%和95%，而低对齐性甚至可能使其逆转，从而抑制……（原文摘要在此处被截断）

    arXiv:2610.06897v1 Announce Type: new  Abstract: Localizing latent structures in the activation space of language models (LMs) is central to understanding and controlling their behavior. Yet, localized structures can differ substantially in their causal influence, raising the question of what makes a structure actionable. We tackle this question by casting causal influence as a product of three factors and showing empirically that they act as interpretable, distinct constraints: capacity, measuring the sensitivity of the model's output to movement along the structure, responsiveness, capturing how promotable the concept is given the current context, and alignment, reflecting how well the structure aligns with the context-specific representation of the concept. Across 4 LM families and 50 concepts, we observe that causal effectiveness requires all factors to be high; low capacity and responsiveness reduce it by 84% and 95%, respectively, while low alignment can reverse it, suppressing c
    
[^144]: 零样本可视化：基于用户提示轴的文本语料库探索

    Zero-Shot Visualization: Exploring Text Corpora with User-Prompted Axes

    [https://arxiv.org/abs/2610.06889](https://arxiv.org/abs/2610.06889)

    该论文提出了零样本可视化（ZSV）任务，允许用户通过自然语言指定概念轴来交互式探索文本语料库，并通过基准测试发现基于下一个词元概率的评分方法在语义忠实性、评分保真度和计算成本方面具有优势。

    

    我们研究了大语言模型（LLM）在文本语料库可视化探索中的应用。我们提出了零样本可视化（ZSV）这一任务，即用户用自然语言指定概念，然后将文档映射到相应的概念轴上进行可视化。构建一个具有实用价值的ZSV系统并非易事，因为它需要在特征函数、高效实现的权衡以及影响可视化质量的预处理/后处理决策的交汇处做出选择。为此，我们建立了一个基准，比较了在此设置下涵盖嵌入相似度、直接语义判断和条件似然估计等多种方法。我们在多个数据集和用例上，从语义忠实性、评分保真度和计算成本等方面评估了不同评分方法和设计选择的特性。我们的结果表明，基于下一个词元概率的评分方法提供了……

    arXiv:2610.06889v1 Announce Type: cross  Abstract: We study the application of large language models (LLMs) to the visual exploration of textual corpora. We introduce zero-shot visualization (ZSV), a task in which users specify concepts in natural language and documents are mapped onto the corresponding concept axes for visualization. Building a ZSV system of practical value is non-trivial, as it requires choices at the intersection of feature functions, efficient implementation tradeoffs, and pre/post-processing decisions affecting visualization quality. To that end, we establish a benchmark that compares methods spanning embedding similarity, direct semantic judgments, and conditional likelihood estimation in this setting. Across multiple datasets and use cases we evaluate the properties of different scoring methods and design choices in terms of semantic faithfulness, score fidelity, and computational cost. Our results identify that scoring based on next-token probabilities offers t
    
[^145]: 何时外部指导能帮助大语言模型推理？指导增强GRPO的偏差-方差理论

    When Does External Guidance Help LLM Reasoning? A Bias-Variance Theory of Guidance-Augmented GRPO

    [https://arxiv.org/abs/2610.06861](https://arxiv.org/abs/2610.06861)

    该论文提出GA-GRPO统一理论框架，将外部指导建模为随机指导算子，证明其引入的偏差可由全变差指导散度δ_G界定，从而为外部指导何时以及如何帮助LLM推理提供收敛速率、偏差界和最优加权规则的理论基础。

    

    带可验证奖励的强化学习（RLVR）已成为引出大语言模型多步推理能力的主流范式，而近期涌现的一系列方法（LUFFY、ExPO、PAPO、TAPO）进一步利用“外部指导”——专家轨迹、自我解释或检索到的思维模式——来增强强化学习。尽管这些方法都报告了实证收益，但没有一种方法给出收敛速率、偏差界或指导信号的最优加权规则。我们通过“指导增强GRPO”（GA-GRPO）来填补这一空白：这是一个统一的理论框架，将外部指导视为一个重写问题分布的随机指导算子G，并把由此得到的策略梯度估计器分析为一个有偏的在策略估计器，其偏差由指导增强采样分布与策略自身分布之间的全变差指导散度δ_G所界定。该框架涵盖了原始GRPO等方法，

    arXiv:2610.06861v1 Announce Type: cross  Abstract: Reinforcement learning with verifiable rewards (RLVR) has become the dominant paradigm for eliciting multi-step reasoning in large language models, and a recent wave of methods (LUFFY, ExPO, PAPO, TAPO) further augments RL with \emph{external guidance} - expert traces, self-explanations, or retrieved thought patterns. Although each method reports empirical gains, none provides convergence rates, bias bounds, or an optimal weighting rule for the guidance signal. We close this gap with \emph{Guidance-Augmented GRPO} (GA-GRPO), a unified theoretical framework that casts external guidance as a stochastic guidance operator G re-writing the question distribution, and analyses the resulting policy-gradient estimator as a biased on-policy estimator whose bias is bounded by the total-variation guidance divergence delta\_G between the guidance-augmented sampling distribution and the policy's own distribution. The framework subsumes vanilla GRPO,
    
[^146]: 提升大语言模型短篇小说生成的多样性

    Improving Diversity in LLM Short Story Generation

    [https://arxiv.org/abs/2610.06729](https://arxiv.org/abs/2610.06729)

    提出DivLM两阶段后训练框架，通过创意写作语料持续预训练结合复合奖励函数的强化学习，使LLM在体裁、语气、风格和命名实体等维度的短篇小说生成多样性平均提升超过9%，同时保持指令遵循与回复质量。

    

    大语言模型（LLM）能够生成准确的回复，但这些回复往往缺乏多样性。我们试图在创意短篇小说生成任务中解决这一问题。基于成熟的写作规范和已知的LLM局限性，我们针对体裁、语气、风格以及命名实体这几个方面的多样性。为了在这些维度上促进多样性，我们提出了DivLM——一个由两个阶段组成的LLM后训练框架。首先，我们在创意写作语料库上进行持续预训练，并使用权重残差来恢复指令遵循能力。随后，我们应用强化学习，采用自定义的复合奖励函数，在保持回复质量的同时，联合最大化目标叙事维度上的多样性。我们在两个LLM系列上的实验结果表明，与其他方法相比，DivLM平均将多样性指标提升超过9%，同时保留了指令遵循能力和回复质量。

    arXiv:2610.06729v2 Announce Type: replace  Abstract: Large language models (LLMs) can generate accurate responses, but these are void of diversity. We attempt to address this for the task of creative short story generation. Drawing on established writing conventions and known LLM limitations, we target variation in genre, tone, style, and named entities. To promote diversity across these dimensions, we introduce DivLM, an LLM post-training framework consisting of two phases. First, we perform continued pre-training on a creative writing corpus and restore instruction-following capabilities using weight residuals. We then apply reinforcement learning with a custom, composite reward function that jointly maximizes diversity across the targeted narrative dimensions while maintaining response quality. Our empirical results on two LLM families show that DivLM increases diversity metrics by more than 9% on average compared to alternative approaches, while preserving instruction following, ov
    
[^147]: Wikidata搜索轨迹：用于训练知识图谱搜索智能体的数据集

    Wikidata Search Traces: A Dataset for Training Knowledge Graph Search Agents

    [https://arxiv.org/abs/2610.06650](https://arxiv.org/abs/2610.06650)

    该论文发布了首个记录解题者在Wikidata知识图谱上探索过程的数据集，用于训练图搜索智能体，并证明图搜索难度可通过问题结构控制、长程搜索失败主要源于证据管理方式而非模型本身，且开源模型在合适环境中可匹敌商业模型。

    

    Wikidata是最大的开放知识库之一，然而要在其上回答一个复杂问题，仍然需要编写一条SPARQL查询，该查询需指明正确的实体和属性，并将它们之间的关系串联起来。语言模型提供了一种自然语言的替代方案，但其回答主要依靠记忆，而这对于知名度较低的实体最为不可靠。我们研究了通过探索图谱来回答问题的智能体，并指出两个障碍限制了它们：一是缺乏记录求解者如何进行探索的训练数据，二是将大规模图谱结果直接加入模型上下文的接口设计。我们检验了三个假设：图搜索的难度可以通过问题的结构来控制，而不仅仅依靠冷门实体或措辞；长程搜索中的许多失败源于对检索证据的管理方式，而非模型本身；以及在合适的环境中，开源权重模型可以匹敌商业（模型）……

    arXiv:2610.06650v2 Announce Type: replace  Abstract: Wikidata is one of the largest open knowledge bases, yet answering a complex question over it still requires a SPARQL query that names the right entities and properties and chains their relations. Language models offer a natural-language alternative but answer largely from memory, which is least reliable for less prominent entities. We study agents that instead answer by exploring the graph, and argue that two obstacles limit them: the lack of training data recording how a solver explores, and interfaces that add large graph results directly to the model's context. We test three hypotheses: that the difficulty of graph search can be controlled through the structure of a question rather than only through obscure entities or wording; that much of the failure on long-horizon search comes from how retrieved evidence is managed rather than from the model itself; and that, in a suitable environment, open-weight models can match commercial 
    
[^148]: JEV与大型语言模型的对比：基于七项政治科学复制研究的准确性、成本与校准评估

    JEV versus LLMs: Accuracy, Cost and Calibration on Seven Political Science Replications

    [https://arxiv.org/abs/2610.06625](https://arxiv.org/abs/2610.06625)

    本文通过七项政治科学复制研究，将商业模型JEV与传统LLM及人工编码员在准确性、成本和校准性上进行对比，以检验其在社会科学文本标注任务中的实际适用性。

    

    大型语言模型（LLM）通过生成文本标记（token）来对政治文本或概念进行标注和测量。而一类由TypeSafe推向市场的新型模型（“System One”模型）则改为在用户提供的固定答案集上返回决策结果及概率分布。一款名为JEV的商业模型被宣传为相比传统LLM具有显著的成本和速度优势，并且决策校准性更好。因此，对于希望快速、低成本地标注或测量大规模文本语料库，并获得分类器不确定性可靠指标的社会科学家而言，该模型可能非常有用。然而，这些宣传的准确性，以及该模型在社会科学文本任务中的整体表现尚未得到验证。本文正是为此展开研究，旨在评估JEV在社会科学任务中的适用性。我们将JEV与已发表研究中的LLM及人工编码员进行比较，并与当前一款中端商业LLM（

    arXiv:2610.06625v2 Announce Type: replace  Abstract: Large language models (LLMs) annotate and scale political text or constructs by generating text tokens. A new class of models, which TypeSafe markets as "System One" models, instead returns decisions and probability distributions across a user-supplied fixed answer set. A commercial model, JEV, is advertised as having a dramatic cost and speed advantage over traditional LLMs along with better calibrated decisions. As such, it might be useful for social scientists looking to quickly and cost-effectively annotate or scale large corpora of text and have a reliable indicator of a classifier's uncertainty. Yet, the accuracy of these claims and the broader model accuracy in social science text-based tasks are not yet established. In this paper, we do just that and hope to establish the suitability of JEV for social science tasks. We compare JEV with LLMs and human coders from published research, and with a current mid-tier commercial LLM (
    
[^149]: TeleTune：从离线遥测数据中演化智能体技能

    TeleTune: Evolving Agent Skills From Offline Telemetry

    [https://arxiv.org/abs/2610.05437](https://arxiv.org/abs/2610.05437)

    TeleTune提出了一个从无目标标注、无法重放且任务交错的离线用户遥测日志中，通过动作预测误差自动演化文本技能库的框架，使计算机使用智能体能够学到可复用的软件操作技能。

    

    计算机使用智能体需要捕捉人们如何使用软件的程序性知识，而用户遥测数据为这类知识提供了可扩展的来源。然而，从这些日志中学习可复用的技能需要解决三个挑战：（1）目标欠规范，因为日志不会记录每个动作背后的目标；（2）不可重放性，因为无法重放过去的活动来评估技能更新；（3）交错轨迹，因为日志可能混合多个任务且未标记其边界。为解决这些问题，我们提出了TeleTune，一个从离线日志中学习文本技能库的框架，这些日志没有记录目标、在优化过程中无法重放，且可能包含交错的任务。TeleTune利用日志轨迹上的动作预测误差来提出技能库的编辑建议，并仅保留那些能提升留出集动作预测准确率的编辑，我们称之为“技能引导的进展”。所学到的工作流还能实现……（摘要在此处被截断）。

    arXiv:2610.05437v2 Announce Type: replace  Abstract: Computer-use agents need to capture procedural knowledge of how people use software. User telemetry offers a scalable source of this knowledge. However, learning reusable skills from these logs requires addressing three challenges: (1) Goal Underspecification, since logs do not record the goal behind each action; (2) Non-Replayability, since past activity cannot be replayed to evaluate skill updates; and (3) Interleaved Trajectories, since logs may mix several tasks without marking their boundaries. To address these, we introduce TeleTune, a framework for learning a textual skill library from offline logs without recorded goals, cannot be replayed during optimization, and may interleave tasks. TeleTune uses action-prediction errors on logged trajectories to propose library edits and keep only those that improve held-out action-prediction accuracy, which we call skill-guided progress. The learned workflows also enable retrieval of dem
    
[^150]: 大规模归纳式论断抽取

    Inductive Claims Extraction at Scale

    [https://arxiv.org/abs/2610.05275](https://arxiv.org/abs/2610.05275)

    本文提出了一个利用大语言模型从大规模社交媒体语料中归纳式抽取并编目论断的处理流程，并将其应用于2020年美国大选和2022年世界杯两个Twitter数据集，通过召回率和精确率全面验证了该方法的有效性。

    

    社交媒体上的政治话语在很大程度上是以“论断”为单位构建和表达的：即陈述性的、通常为单一从句的表述，它们传达了对现实的某种特定解读，范围可从事实性陈述到评价性观点。此外，论断并非随机出现，而是相互聚合、以规律性模式反复出现，并与不同的世界观相关联。当与社交网络分析等结构化计算工具结合使用时，论断可以成为研究回音室或政治极化等政治现象的强大分析单元。本文提出了一个利用大语言模型（LLM）从大规模社交媒体语料中归纳式抽取并编目论断的处理流程，并将其应用于两个不同的Twitter数据集：一个与2020年美国总统大选相关，另一个与2022年FIFA世界杯相关。作者通过测量该流程的召回率和精确率对这一方法进行了全面评估。

    arXiv:2610.05275v2 Announce Type: replace  Abstract: A large part of political discourse on social media is built and expressed at a level of claims: i.e. declarative, typically single-clause statements, which convey a particular interpretation of reality and can range from factual to evaluative. Moreover, rather than occurring randomly, claims coalesce, recur in patterns, and come to be associated with different world views. When paired with structural computational tools such as Social Network Analysis, claims can be a powerful unit of analysis to study political phenomena such as echo chambers or polarisation. In this paper, we present a pipeline that uses a large language model (LLM) to inductively extract and catalogue claims from large social media corpora, and apply it to two different Twitter datasets: one relating to the 2020 US presidential election and the other to the 2022 FIFA World Cup. We comprehensively evaluate the approach by measuring the pipeline's recall and precis
    
[^151]: 语言模型意外度对中文阅读预测能力的系统性分析

    A Systematic Analysis of the Predictive Power of LM Surprisal in Reading Chinese

    [https://arxiv.org/abs/2610.04898](https://arxiv.org/abs/2610.04898)

    本研究提出最短匹配序列（SMS）对齐方案以解决中文分词与语言模型子词分词不一致的问题，并利用从零训练的Chinese-Pythia模型证明语言模型意外度确实能预测中文阅读时间，且预测能力随模型规模的缩放模式因语料库而异。

    

    本研究分析了语言模型生成的词元级意外度对中文阅读时间的预测能力。我们首先提出了最短匹配序列，这是一种对齐方案，用于将眼动追踪语料库所假设的词切分与语言模型的子词分词进行映射，因为在中文语境下这两种分词方式常常不一致。随后，我们使用一系列从零开始训练、使用了300亿词元的Chinese-Pythia模型（14M至1.4B参数），考察了意外度在三个中文段落级眼动追踪语料库（GECO-CN、HKP和MECO）中对首次注视时长、凝视时长和总阅读时间的预测效果。与以往的无显著结果相反，我们的研究表明意外度确实能够预测中文阅读时间。然而，预测能力是否随模型规模和训练量而提升则因语料库而异：在GECO-CN中更大的模型预测效果更好，而在其他语料库中则出现了逆向缩放现象。

    arXiv:2610.04898v2 Announce Type: replace-cross  Abstract: This study analyzes the predictive power of LM-derived, token-level surprisal on Mandarin Chinese reading times. We first propose the Shortest Matching Sequence (SMS), an alignment scheme that maps between the word segmentation assumed by eye-tracking corpora and the LMs' subword tokenization, as the two tokenizations often disagree in the context of Mandarin Chinese. Then, using a suite of Chinese-Pythia models (14M-1.4B) trained on scratch with 30B tokens, we examine how well surprisal predicts first fixation duration, gaze duration, and total reading time in three paragraph-level eye-tracking corpora of Mandarin Chinese (GECO-CN, HKP, and MECO). Contrary to previous null findings, our results show that surprisal is predictive of Chinese reading times. However, whether predictive power scales with model size and the amount of training is corpus-specific: bigger models predict better in GECO-CN, whereas inverse scaling emerges
    
[^152]: SpecFold：折叠多分支冗余以加速扩散语言模型中的投机解码

    SpecFold: Folding Multi-Branch Redundancy for Faster Speculative Decoding in Diffusion Language Models

    [https://arxiv.org/abs/2610.04875](https://arxiv.org/abs/2610.04875)

    SpecFold通过识别并利用投机验证中草稿分支与父分支之间隐藏状态高度相似的多分支计算冗余，以token级残差门控和选择性计算复用降低验证成本，从而加速扩散语言模型的多分支投机解码。

    

    扩散大语言模型（DLLMs）通过迭代块去噪生成文本，多分支投机解码则通过在单次前向传播中同时验证一个主分支与多个草稿分支来加速这一过程。现有DLLM加速方法主要利用去噪步骤之间的时间冗余，而我们识别出每个投机验证步骤中一条互补的冗余维度：多分支计算冗余。在投机验证过程中，草稿分支从其父分支继承大部分token，仅解开少量额外位置，导致大量隐藏状态在各分支之间保持高度相似。我们提出SpecFold，一种算法-系统协同设计，利用这种多分支冗余来降低多分支投机验证的成本。在算法层面，SpecFold执行token级残差门控并选择性地复用父分支的计算（摘要原文在此处截断）。

    arXiv:2610.04875v2 Announce Type: replace  Abstract: Diffusion large language models (DLLMs) generate text through iterative block denoising, and multi-branch speculative decoding accelerates this process by verifying a main branch together with multiple draft branches in a single forward pass. While prior DLLM acceleration methods primarily exploit temporal redundancy across denoising steps, we identify a complementary redundancy axis within each speculative verification step: multi-branch computational redundancy. During speculative verification, draft branches inherit most tokens from their parents while unmasking a small set of additional positions, causing large portions of hidden states to remain highly similar across branches. We propose SpecFold, an algorithm-system co-design that exploits this multi-branch redundancy to reduce the cost of multi-branch speculative verification. Algorithmically, SpecFold performs token-level residual gating and selectively reuses parent computat
    
[^153]: 每个键对应更多值：非对称稀疏注意力加速大语言模型解码

    More Value per Key: Asymmetric Sparse Attention for Faster LLM Decoding

    [https://arxiv.org/abs/2610.04753](https://arxiv.org/abs/2610.04753)

    提出稀疏非对称分组查询注意力SAGA，通过解耦键头与值头数量——用更少的键头加速推理、保留更多值头维持模型容量——从而实现更快的LLM解码。

    

    大语言模型（LLM）的自回归生成受限于注意力机制的内存与计算需求。稀疏注意力方法通过仅选择注意力矩阵中的高概率条目来缓解这一开销。我们观察到，在许多此类方法中，这使得概率-值乘法的开销变得可以忽略不计，从而将瓶颈转移到了查询-键计算步骤上。因此，可以通过减少键头的数量来加速推理，同时保留更多的值头，以在有限的额外解码成本下维持模型容量。我们提出了稀疏非对称分组查询注意力（SAGA），它解耦了键头与值头的数量以利用这一原理，并将其与近似top-N（Atop-N）注意力相结合，后者是一种简单的稀疏注意力方法，旨在研究稀疏性与头数非对称性之间的相互作用。我们从理论上形式化了这种非对称性的优势，并通过实验验证了这些优势。

    arXiv:2610.04753v2 Announce Type: replace-cross  Abstract: Autoregressive generation in Large Language Models (LLMs) is constrained by the memory and computational demands of attention mechanisms. Sparse attention methods mitigate this cost by selecting only high-probability entries of the attention matrix. We observe that in many such methods, this renders the probability-value multiplication negligible, shifting the bottleneck to the query-key step. Key heads can therefore be reduced to accelerate inference, while retaining more value heads preserves capacity with limited additional decoding cost. We introduce Sparse Asymmetric Group-Query Attention (SAGA), which decouples key and value head counts to exploit this principle, and pair it with approximate top-N (Atop-N) attention, a simple sparse attention method designed to study the interaction between sparsity and head-count asymmetry. We formalize the benefits of this asymmetry theoretically and validate them empirically through la
    
[^154]: 理解基于大语言模型的不完美表格问答中的错误

    Understanding Errors in LLM-Based Question Answering over Imperfect Tables

    [https://arxiv.org/abs/2610.04687](https://arxiv.org/abs/2610.04687)

    该研究发现LLM在不完美表格问答中，错误发现的程度受行顺序影响（错误行出现越晚或越集中越容易被发现），且仅提供已验证的错误位置信息并不足以保证准确问答。

    

    在不完美表格上回答问题需要处理可能影响答案的错误。我们研究了大型语言模型（LLM）面临的两个挑战：错误的发现是否取决于错误在表格中出现的位置，以及提供错误位置是否足以实现准确的问答。利用来自RADAR-T的人工审核实例，我们通过改变行顺序并比较原始表格、标记错误的表格和修复后的表格，在三个LLM上进行了受控研究。首先，即使表格内容和标准答案保持不变，重新排列行顺序也会改变错误的发现情况。在直接检查过程中，当包含相关错误的行在表格中出现得较晚或聚集得更紧密时，LLM更有可能发现所有包含相关错误的行。其次，仅提供已验证的错误位置不足以实现准确的问答：在代码执行的情况下，修复后表格上的准确率比标记错误表格的准确率高出39个百分点。

    arXiv:2610.04687v2 Announce Type: replace  Abstract: Answering questions over imperfect tables requires handling errors that can affect the answer. We investigate two challenges for large language models (LLMs): whether error discovery depends on where errors appear in a table, and whether providing their locations is sufficient for accurate question answering (QA). Using human-reviewed instances from RADAR-T, we conduct controlled studies across three LLMs by varying row order and comparing original, error-marked, and repaired tables. First, reordering rows changes error discovery even when the table contents and gold answer remain unchanged. During direct inspection, LLMs are more likely to discover all rows containing relevant errors when these rows appear later in the table or are grouped more closely together. Second, providing verified error locations alone is insufficient for accurate QA: with code execution, accuracy on repaired tables exceeds that on error-marked tables by 39.
    
[^155]: 类型化决策模型中候选选项覆盖度的基准测试

    Benchmarking Candidate Coverage in Typed Decision Models

    [https://arxiv.org/abs/2610.03387](https://arxiv.org/abs/2610.03387)

    本文提出了一个成对候选覆盖度基准测试协议，用于评估类型化决策模型识别缺失答案与避免错误拒绝有效候选的能力，发现 Laya 和 Jev 的原生拒绝行为差异显著，而仅使用校准数据的 none 分数阈值可以显著改善两者的检测与误拒平衡。

    

    类型化决策模型会返回选择结果，或针对请求时提供的答案选项返回分布。在选项完整情况下的准确率并不能说明模型是否能识别参考答案缺失的情况，或者是否会避免错误拒绝有效的候选选项。我们提出了一个成对候选覆盖度基准测试协议，并对 Laya 和 Jev 两个模型在 AG News、DBpedia、Emotion 和 TREC 数据集上进行了初步评估。两个模型接收完全相同的冻结文本和请求：300 条校准文本和 589 条测试文本，每个模型产生 23,932 次预测。存在/缺失配对与普通候选数量相匹配，且名称变体保持描述、成员和顺序不变。原生的拒绝行为差异显著：在具有自然名称的五个 TREC 候选选项下，Laya 能检测出 97.2% 的答案缺失案例，但会错误拒绝 69.7% 的存在对照案例；Jev 的这两个比率分别为 24.8% 和 0.0%。仅使用校准数据的 none 分数阈值将这些比率分别改变为 33.9%/3.7% 和 45.0%/1.8%。在 DB

    arXiv:2610.03387v1 Announce Type: new  Abstract: Typed decision models return choices or distributions over answer options supplied at request time. Accuracy with complete options does not establish whether a model recognizes that a reference answer is missing or avoids rejecting valid candidates. We present a paired candidate-coverage benchmark protocol and an initial evaluation of Laya and Jev across AG News, DBpedia, Emotion, and TREC. The models receive identical frozen texts and requests: 300 calibration and 589 test texts yield 23,932 predictions per model. Present/absent pairs match ordinary candidate count, and name variants preserve descriptions, members, and order. Native rejection behavior differs sharply: at five TREC candidates with natural names, Laya detects 97.2% of missing-answer cases but falsely rejects 69.7% of present controls; Jev's rates are 24.8% and 0.0%. Calibration-only none-score thresholds change these rates to 33.9%/3.7% and 45.0%/1.8%, respectively. On DB
    
[^156]: 偏好的可验证、可表达与默会成分

    Verifiable, Articulable, and Tacit Components of Preference

    [https://arxiv.org/abs/2610.03025](https://arxiv.org/abs/2610.03025)

    该论文推出了包含280万文本和3.17亿人类偏好判断的CreativePreferences数据集，通过可执行程序、评分标准库与密集训练模型三种方式建模偏好，量化了“可表达”与“可验证”成分和完整偏好之间的差距，揭示了AI偏好优化中长期被忽视的默会成分。

    

    是什么让一篇短篇小说引人入胜、一篇新闻报道具有新闻价值、或一个数学证明优雅？这些构念难以言明或验证，其含义至少部分是默会的。然而，现代AI模型主要是通过明确的章程、评分标准和验证器（即RLAIF和RLVR）来改进的；偏好的默会成分通常研究不足。我们引入了一个大规模、带标注的偏好数据集CreativePreferences，其中包含280万个文本，由3.17亿个人类偏好判断在7个创意领域中进行标注，并配有42个基准任务。我们分别用可执行程序、评分标准库和密集训练的模型（V、A和VAT）对这些标签进行建模。我们观察到稳健的可表达性差距（VAT−VA）和可验证性差距（VAT−V）；我们采用一种新颖的测量方法来估计每个差距的上界和下界，该方法可发现可表达和可验证的指标、识别伪变量，并估计未被发现成分的价值。

    arXiv:2610.03025v1 Announce Type: new  Abstract: What makes a short story gripping; a news article newsworthy; or a math proof elegant? These constructs resist articulation or verification; their meaning is at least partially tacit. However, modern AI models are improved primarily via articulated constitutions, rubrics and verifiers (i.e. in RLAIF and RLVR); tacit components of preferences are typically understudied. We introduce a large, labeled preference dataset CreativePreferences, containing 2.8M texts labeled by 317M human preference judgments across 7 creative domains, with 42 benchmark tasks. We model these labels with executable programs, rubric banks and densely trained models (V, A and VAT, respectively). We observe robust articulability gaps, VAT-VA; and verifiability gaps, VAT-V; we estimate upper and lower bounds for each gap with a novel measurement approach that discovers articulable and verifiable metrics, identifies spurious variables and estimates the value of undisc
    
[^157]: 一种面向模式即代码生物医学命名实体识别的指南增强多智能体框架

    A Guideline-Augmented Multi-Agent Framework for Schema-as-Code Biomedical Named Entity Recognition

    [https://arxiv.org/abs/2610.02970](https://arxiv.org/abs/2610.02970)

    GAMA框架通过从标注数据中归纳并验证特定数据集的标注规则来构建指南记忆，结合多智能体协作与模式即代码的结构化输出，显著提升了生物医学命名实体识别的准确性和格式规范性。

    

    大语言模型（LLMs）通过指令遵循和上下文学习在生物医学命名实体识别方面展现出了可喜的潜力。然而，现有的基于LLM的BioNER方法仍然面临两个关键局限。首先，检索到的示例和外部生物医学知识对特定数据集的标注语义支持有限，导致实体边界、类型范围和标注规范模糊不清。其次，自由形式的生成缺乏足够的结构控制，常常导致无效格式、幻觉提及、重复实体和边界错误。为了解决这些局限，我们提出了GAMA，一个面向模式即代码BioNER的指南增强多智能体框架。GAMA首先从已标注的训练实例中归纳候选标注规则，并通过标注数据进行验证，以构建可靠的特定数据集指南记忆。在这些经过验证的规则的指导下，一个规划……（摘要在此处截断）

    arXiv:2610.02970v1 Announce Type: new  Abstract: Large language models (LLMs) have shown promising potential for biomedical named entity recognition (BioNER) through instruction following and in-context learning. However, existing LLM-based BioNER methods still face two key limitations. First, retrieved demonstrations and external biomedical knowledge provide limited support for dataset-specific annotation semantics, leaving entity boundaries, type scopes, and annotation conventions ambiguous. Second, free-form generation lacks sufficient structural control, often leading to invalid formats, hallucinated mentions, duplicated entities, and boundary errors. To address these limitations, we propose GAMA, a guideline-augmented multi-agent framework for schema-as-code BioNER. GAMA first induces candidate annotation rules from labeled training instances and verifies them against annotated data to construct reliable dataset-specific guideline memory. Guided by these verified rules, a planning
    
[^158]: 第二个模型何时有帮助？大语言模型验证中的跨模型审查

    When Does a Second Model Help? Cross-Model Review in LLM Verification

    [https://arxiv.org/abs/2610.01471](https://arxiv.org/abs/2610.01471)

    在大语言模型输出验证中，跨模型审查与同模型审查发现的错误集合部分不同，且“一次同模型新会话审查加一次跨模型审查”的组合比两次同模型审查能匹配到更多埋设错误（56.7% vs. 42.7%）。

    

    大语言模型如今能够生成代码、文档和分析内容，并且越来越多地被用于审查这类输出。本文探究的问题是：由不同的模型进行第二次审查在何时才有帮助？在作者早期预印本研究（在单一模型内改变上下文、重复次数和角色结构）的基础上，我们通过一项受控实验来检验模型独立性：实验包含30个工件（其中埋设150个错误）、10种审查条件，以及由来自两个开发者的三个审查模型执行的900次审查会话。在该实验中，(1) 顶级跨模型审查者在F1分数上与同模型在全新会话中的审查（CCR）无显著差异，但这并不等同于二者等价；(2) 两者发现的错误部分不同（Jaccard相似度为41.2%）；(3) 在两次审查调用的设定下，一次CCR加一次跨模型审查所匹配到的埋设错误多于两次CCR审查（56.7% vs. 42.7%；经Holm校正的p=.006），但并不显著多于两次顶级跨模型审查。

    arXiv:2610.01471v1 Announce Type: cross  Abstract: Large language models now generate code, documentation, and analyses, and are increasingly used to review such output. We ask when a second review by a different model helps. Building on the author's earlier preprints, which varied context, repetition, and role structure within one model, we test model independence in a controlled experiment: 30 artifacts with 150 planted errors, 10 review conditions, and 900 review sessions with three reviewer models from two developers. In this experiment, (1) a top-tier cross-model reviewer is not significantly different in F1 from same-model review in a fresh session (CCR), which does not establish equivalence; (2) the two find partly different errors (Jaccard 41.2%); and (3) at two review calls, one CCR plus one cross-model review matches more planted errors than two CCR reviews (56.7% vs. 42.7%; Holm-adjusted p=.006), but not significantly more than two reviews by the top-tier cross-model reviewe
    
[^159]: AgSpec：在编码智能体流水线中突破基于检索的投机解码的极限

    AgSpec: Pushing the Limits of Retrieval-Based Speculative Decoding in Coding Agent Pipelines

    [https://arxiv.org/abs/2610.01108](https://arxiv.org/abs/2610.01108)

    AgSpec通过构建会话、工作区和全局三级语料库并以离线上限与在线反馈自适应调整草稿长度，突破了编码智能体流水线中基于检索的投机解码的性能极限。

    

    基于检索的投机解码（SD）通过从已有文本中复制续写内容来生成草稿词元，这非常适合那些需要反复重现代码、日志和先前尝试的编码智能体。然而，现有方法在智能体流水线中表现不佳：其语料库中缺失大量可复用的文本，或者文本的存储形式与智能体实际输出的形式不一致，而且它们的草稿长度设定忽略了接受长度因智能体而异、并会随对话轮次发生漂移这一事实。我们提出了AgSpec，这是一个为编码智能体流水线提供现有检索引擎所缺失的语料库与草稿长度策略的框架。AgSpec从会话级、工作区级和全局级语料库中进行检索，保留正在进行的会话轨迹，并以智能体的输出格式对已打开的文件建立索引。它通过离线分析得到的上限来约束每个智能体的草稿长度，并根据验证反馈在线自适应地调整长度。在两个仓库级多智能体编码基准测试上，AgSpec取得了超越（摘要在此处截断）……

    arXiv:2610.01108v2 Announce Type: replace  Abstract: Retrieval-based speculative decoding (SD) drafts tokens by copying continuations from existing text, which suits coding agents that repeatedly reproduce code, logs, and earlier attempts. Yet existing methods fall short in agent pipelines: much of the reusable text is missing from their corpora or stored in a form that differs from what the agent emits, and their draft lengths ignore that accept length varies across agents and drifts over turns. We present AgSpec, a framework that supplies the corpus and draft-length policies that existing retrieval engines lack in coding-agent pipelines. AgSpec retrieves from session, workspace, and global corpora, retaining the ongoing session trajectory and indexing opened files in the agent's emission format. It bounds each agent's draft length with an offline-profiled cap and adapts the length online from verification feedback. On two repository-level multi-agent coding benchmarks, AgSpec outperf
    
[^160]: 基于隐马尔可夫模型的可证明易处理的非确定有限自动机约束语言生成

    Provably Tractable NFA-Constrained Language Generation via HMMs

    [https://arxiv.org/abs/2609.40185](https://arxiv.org/abs/2609.40185)

    该论文提出了NFA-LM，一个基于#NFA问题FPRAS理论的多项式时间生成引擎，首次在温和假设下以理论保证高效解决NFA约束语言生成问题，克服了现有方法扭曲分布或牺牲效率的缺陷。

    

    受约束生成旨在从语言模型中采样并满足硬性约束条件。现有的针对非确定有限自动机（NFA）约束的受约束生成技术要么扭曲分布，要么牺牲效率。从理论上讲，该任务可归约为统计NFA所接受的长度为n的序列数量问题（#NFA），而精确求解#NFA问题是#P完全的。近期研究表明，#NFA问题存在完全多项式随机近似方案（FPRAS）。受此结果启发，我们提出了NFA-LM，这是一个在温和假设下具有理论保证的、用于NFA约束生成的多项式时间引擎。实验表明，NFA-LM能够高效生成高质量输出，且其近似误差具有理论界。

    arXiv:2609.40185v1 Announce Type: new  Abstract: Constrained generation aims to sample from language models (LMs) conditioned on hard constraints. Existing constrained-generation techniques for nondeterministic finite automaton (NFA) constraints either distort the distribution or sacrifice efficiency. Theoretically, this task reduces to counting the length-$n$ sequences accepted by an NFA (#NFA), and the exact #NFA problem is #P-complete. Recent work has shown that #NFA admits a fully polynomial randomized approximation scheme (FPRAS). Inspired by this result, we propose NFA-LM, a polynomial-time engine for NFA-constrained generation with theoretical guarantees under mild assumptions. Experiments show that NFA-LM efficiently generates high-quality outputs with theoretically bounded approximation error.
    
[^161]: 以编码器速度实现的强大多语言隐私标注

    Strong Multilingual Privacy Tagging at Encoder Speed

    [https://arxiv.org/abs/2609.38630](https://arxiv.org/abs/2609.38630)

    该研究提出了一个以编码器速度运行的多语言隐私实体标注模型，通过覆盖感知掩码与子词边界修复等技术，在7种语言的人工金标准测试上取得88.8的脱敏F1分数，显著超越GLiNER2、Microsoft Presidio和OpenAI Privacy Filter等现有方法。

    

    隐私脱敏必须在删除个人信息的同时保留文本中表达的关系。我们开发了一个支持细粒度区分的多语言命名实体标注器，可服务于多种脱敏策略，并提供了低成本学习额外区分类别的方法。我们在35种语言的前沿模型标注数据上，对带有仿射跨度标注头的多语言编码器进行微调，通过覆盖感知掩码技术重放映射后的人工金标准数据，以避免将未标注的实体类型误当作负例，并利用学习到的±1字符调整来修复子词边界。在7种语言共1,283个人工金标准测试片段上，该模型的最佳脱敏F1分数达到88.8，相比之下，已发布的GLiNER2为69.1（该对比中排除了其无法表示的11个类型，若不作此豁免则为68.8），针对新训练数据适配后的GLiNER2为67.8，Microsoft Presidio为57.3，而已发布的最佳OpenAI Privacy Filter微调版本仅为35.8。通过增加约50,000个标注训练样本……（原文摘要在此处截断）

    arXiv:2609.38630v1 Announce Type: new  Abstract: Privacy redaction must remove personal information while preserving relationships expressed in text. We develop a multilingual named-entity tagger with fine-grained distinctions supporting varied redaction policies and methods for cheaply learning additional distinctions. We fine-tune a multilingual encoder with an affine span-tagging head on frontier-model annotations in 35 languages, replay mapped human gold with coverage-aware masking so unannotated types are not treated as negatives, and repair subword boundaries with a learned +/-1-character adjustment. On 1,283 human-gold test segments in seven languages, best measured redaction F1 is 88.8, against 69.1 for published GLiNER2 with 11 unrepresentable types excluded from its task (68.8 without that exemption), 67.8 for GLiNER2 adapted to the new training data, 57.3 for Microsoft Presidio and 35.8 for the best published OpenAI Privacy Filter fine-tune. Adding about 50,000 annotated tra
    
[^162]: 约鲁巴语中曲折调的标记方法

    Marking Contour Tones in Yor\`{u}b\'{a}

    [https://arxiv.org/abs/2609.38627](https://arxiv.org/abs/2609.38627)

    本文提出在约鲁巴语正字法中采用caron（ˇ）和circumflex（ˆ）符号来标记单个元音上的升降曲折调，以解决传统拼写中声调信息缺失甚至颠倒姓名含义的问题，并使其首次可通过标准键盘输入和计算文本处理。

    

    约鲁巴语是一种声调语言，其曲折调在正字法书写上一直存在难题。这一问题在个人姓名和词汇中尤为突出，因为这些词的传统拼写避免了元音延长，而元音延长本可为第二个声调提供承载音节。尤其值得关注的是一类姓名，其传统拼写不仅省略了声调信息，还会颠倒名字的含义，有时甚至表达出与名字本意相反的内容。本文描述了这一问题，说明了现有解决方案的不足，并提议采用caron（倒折音符ˇ）和circumflex（抑扬符ˆ）符号。这些符号自Olmsted（1951）以来在约鲁巴语音系学研究中已有先例，作为书写惯例用于单个元音之上，以编码升调和降调曲折调，从而首次使这些曲折调能够通过标准键盘输入和计算文本处理来访问。该提议得到了支持。

    arXiv:2609.38627v1 Announce Type: new  Abstract: Yor\`ub\'a is a tonal language in which contour tones pose persistent orthographic challenges. These are especially notable for personal names and lexical items whose conventional spellings avoid vowel lengthening that would otherwise provide a host syllable for the second tone. A particular concern is a class of names in which the conventional spelling does not just omit tonal information but inverts the meaning of said name, sometimes asserting the opposite of what the name intends. This paper describes the problem, illustrates the inadequacy of current solutions, and proposes the adoption of the caron and circumflex marks. These are symbols with precedent in Yor\`ub\'a phonological scholarship since Olmsted (1951), used as orthographic conventions on single vowels to encode rising and falling contour tones, making them accessible for the first time through standard keyboard input and computational text processing. The proposal is supp
    
[^163]: KlinikeBench：超越诊断准确性的语言模型评估

    KlinikeBench: Evaluating Language Models Beyond Diagnostic Accuracy

    [https://arxiv.org/abs/2609.38480](https://arxiv.org/abs/2609.38480)

    KlinikeBench是一个包含333个由临床医生编写的任务的基准，通过沙盒环境中的虚拟患者交互，评估语言模型在信息收集和临床评估方面超越单纯诊断准确性的综合临床能力。

    

    大多数临床基准测试使用完整的病例描述来评估语言模型（LM）的诊断能力。然而在临床实践中，患者以不同的方式呈现信息，临床医生必须获取相关病史并确定需要进行哪些检查，才能做出诊断。因此，仅凭诊断准确性无法判断智能体是否收集了必要的信息或进行了适当的临床评估。此外，现有基准缺乏专业临床医生的验证。为了填补这一空白，我们推出了KlinikeBench，这是一个包含333个由临床医生编写的任务的基准，每个任务都提供了一个隔离的沙盒环境，其中包含虚拟患者、临床工具和针对特定任务的成功标准。超过35名临床医生参与了病例编写和基准评估。在一项实证研究中，临床医生对模拟对话的平均质量评分高于参考对话，这表明……

    arXiv:2609.38480v1 Announce Type: cross  Abstract: Most clinical benchmarks evaluate language models (LMs) on diagnosis using complete case descriptions. In clinical practice, however, patients present information in different ways, and clinicians must obtain relevant history and determine which examinations are needed before reaching a diagnosis. Diagnostic accuracy alone therefore cannot establish whether an agent gathered essential information or conducted an appropriate clinical assessment. Furthermore, existing benchmarks lack professional clinicians' verification. To address this gap, we introduce KlinikeBench, a benchmark of 333 clinician-authored tasks, each providing an isolated sandbox environment with a virtual patient, clinical tools, and task-specific success criteria. More than 35 clinicians contributed to case authoring and benchmark evaluation. In an empirical study, clinicians gave simulated dialogues higher mean quality ratings than reference conversations, which is a
    
[^164]: 存储并非策略：面向大语言模型遗忘的状态条件支撑集控制

    Storage Is Not Strategy: State-Conditioned Support Control for LLM Unlearning

    [https://arxiv.org/abs/2609.37858](https://arxiv.org/abs/2609.37858)

    该论文发现“存储目标知识”的参数未必是执行遗忘的最佳干预对象，提出基于实际遗忘更新预测效果的干预分数以及动态干预重排方法（DIR-R），在优化过程中按需自适应调整干预参数子集，从而显著提升大语言模型遗忘的效果。

    

    许多局部化的大语言模型（LLM）遗忘方法从定位信号中选出一小部分参数子集，并在优化过程中将其固定不变。然而，与目标知识关联最强的参数并不一定是最佳的更新对象，且候选干预的价值会随优化进程而变化。在一个受控实验中，存储定位分数达到了0.981的受试者工作特征曲线下面积（AUROC），但存储身份仅在17/36个目标上与更优的干预选择一致，而低秩适应（LoRA）则在35/36个目标上胜出。我们提出干预分数，该分数根据实际遗忘更新的预测效果对可编辑参数组进行排序，同时将附带损害纳入考量，并以此构建静态干预价值基线。随后我们进一步提出选择性动态干预重排（DIR-R），仅当经过校准的探针证明有必要时，才对该参数子集进行重新审视和调整。

    arXiv:2609.37858v1 Announce Type: cross  Abstract: Many localized large language model (LLM) unlearning methods select a small parameter subset from a localization signal and keep it fixed during optimization. The parameters most associated with a target, however, need not be the best ones to update, and candidate interventions can change value as optimization proceeds. In a controlled experiment, a storage-localization score reaches an area under the receiver operating characteristic curve (AUROC) of 0.981, yet storage identity agrees with the better intervention on only 17/36 targets, while low-rank adaptation (LoRA) wins 35/36. We introduce Intervention Score, which ranks editable groups by the predicted effect of the actual unlearning update while accounting for collateral damage, and use it to form the static intervention-value baseline (Static-IV). We then introduce selective dynamic intervention re-ranking (DIR-R), which revisits that subset only when a calibrated probe justifie
    
[^165]: 测试时扩展与训练在个体立场预测中的不足之处在哪里？

    Where Do Test-Time Scaling and Training Fall Short in Individual Stance Prediction?

    [https://arxiv.org/abs/2609.33155](https://arxiv.org/abs/2609.33155)

    该研究揭示了测试时扩展和后训练方法（如监督微调与强化学习）在个体立场预测任务中存在错误共识、选择失败、响应过拟合和早期停滞四种失败模式，并提出STANCE-BENCH基准来系统性地暴露这些不足。

    

    测试时扩展和后训练方法已提升了大语言模型在编程和数学推理方面的性能，但它们在个体立场预测任务中的有效性仍不清楚。我们通过根据一个人的历史发言来预测其在新讨论中的立场来研究这一问题。我们评估了广泛使用的测试时扩展策略和后训练方法，如监督微调和强化学习，并在生成、选择和学习环节中识别出四种失败模式：（1）错误共识，即重复采样在错误立场上达成一致；（2）选择失败，即生成环节已覆盖了实际立场，但选择环节却未能选中它；（3）响应过拟合，即监督微调提升了模仿能力却损害了预测性能；（4）早期停滞，即强化学习在初期取得适度提升后进步有限。我们构建了包含2499个预测任务的STANCE-BENCH基准来揭示这些失败。

    arXiv:2609.33155v2 Announce Type: replace  Abstract: Test-time scaling and post-training have improved LLM performance in coding and mathematical reasoning, but their effectiveness for individual stance prediction remains unclear. We study this question by predicting a person's stance in a new discussion from their history. We evaluate widely used test-time scaling strategies and post-training methods, such as supervised fine-tuning and reinforcement learning, and identify four failure modes across generation, selection, and learning: (1) incorrect consensus, where repeated samples agree on the wrong stance; (2) selection failure, where generation covers the observed stance but selection misses it; (3) response overfitting, where supervised fine-tuning improves imitation but harms prediction; and (4) early plateau, where reinforcement learning shows modest initial gains followed by limited further improvement. We expose these failures using STANCE-BENCH, which contains 2499 prediction 
    
[^166]: ELF-REG：将连续扩散语言模型扩展至推理任务

    ELF-REG: Scaling Continuous Diffusion Language Models to Reasoning Tasks

    [https://arxiv.org/abs/2609.29102](https://arxiv.org/abs/2609.29102)

    提出ELF-REG方法，利用冻结自回归教师模型的表示对齐与纠缠（REPA+REG）监督，成功将全连续扩散语言模型扩展至数学推理与代码生成任务，性能超越同规模扩散语言模型。

    

    全连续扩散语言模型（dLMs）对连续表示进行去噪而无需中间离散化，并在最后一步并行解码所有响应token。它们在具有挑战性的推理任务上的性能表现，尚未像自回归（AR）大语言模型和掩码扩散语言模型那样得到充分验证。我们将嵌入式语言流（ELF）扩展到GSM8K、MATH-500、HumanEval和MBPP上的数学推理与代码生成任务。我们提出了ELF-REG，它通过表示对齐与纠缠（REPA+REG）来改进学习，其中冻结的AR教师模型监督中间去噪器特征，并提供一个与响应联合去噪的全局表示。ELF-REG-L在64次网络函数评估（NFE）下于GSM8K上达到55.96%的pass@1，在128次NFE下于MATH-500上达到13.39%、HumanEval上达到22.56%。它在GSM8K和代码任务上的pass@1优于所评估的同规模扩散语言模型，并将MATH-500的pass@1从……（摘要原文在此处被截断）

    arXiv:2609.29102v1 Announce Type: new  Abstract: Fully continuous diffusion language models (dLMs) denoise continuous representations without intermediate discretization, then decode all response tokens in parallel at the final step. Their performance on challenging reasoning tasks remains less established than that of autoregressive (AR) LLMs and masked dLMs. We scale Embedded Language Flows (ELF) to mathematical reasoning and code generation on GSM8K, MATH-500, HumanEval, and MBPP. We introduce ELF-REG, which improves learning with representation alignment and entanglement (REPA+REG), where a frozen AR teacher supervises intermediate denoiser features and supplies a global representation that is jointly denoised with the response. ELF-REG-L achieves 55.96% pass@1 on GSM8K at 64 network function evaluations (NFE), and 13.39% on MATH-500 and 22.56% on HumanEval at 128 NFE. It outperforms the evaluated comparable-scale dLMs in pass@1 on GSM8K and code, and improves MATH-500 pass@1 from 
    
[^167]: 合成研究验证中的后果性行为与表征公平性

    Consequential Behaviour and Representational Fairness in the Validation of Synthetic Research

    [https://arxiv.org/abs/2609.27690](https://arxiv.org/abs/2609.27690)

    该论文指出现有的合成调查受访者验证方法在预测后果性行为的应用场景中检验了错误的目标，并提出一个要求效度声明必须明确与人类数据对应水平的验证框架，以保障合成研究的表征公平性。

    

    arXiv:2609.27690v1 公告类型：新 摘要：工业界和学术界的研究人员使用由大语言模型驱动的合成调查受访者作为人类样本的替代品。这些合成群体需要与现实世界数据进行验证，因此研究人员通常采用与人类调查进行临时比较的方式来完成这一工作。受行为科学中“意向-行为差距”概念的启发，我们认为，在大多数应用场景中——即决策者委托合成研究以预测后果性行为时——这些现有验证方法检验的是错误的东西。为解决这一问题，我们提出了一个包含两项要求的验证框架。第一，每个效度声明必须说明其与人类数据的对应水平：样本是否能预测所代表人群的实际行为、验证涉及四种诊断维度（位置、离散度、反应过程和结构）中的哪一种，以及验证是否与实验效应进行了比较？第二，研究人员必须报告效度验证……

    arXiv:2609.27690v1 Announce Type: new  Abstract: Researchers in industry and academia use synthetic survey respondents powered by large language models as substitutes for human samples. These synthetic populations require validation against real-world data, so researchers often address them using ad hoc comparisons with human surveys. Inspired by the intention-behaviour gap in behavioural science, we argue that these validations test the wrong thing for most applied cases where decision makers commission synthetic research to anticipate consequential behaviour. To address this problem, we propose a validation framework with two requirements. First, every validity claim must state its level of correspondence with human data: does the sample predict what the represented people do, which of four diagnostics (location, dispersion, response process and structure) does the validation address, and does the validation compare against experimental effects? Second, researchers must report validi
    
[^168]: 文本分数可能忽视对波形的利用：Qwen2-Audio量化案例研究

    Text Scores Can Miss Waveform Use: A Qwen2-Audio Quantization Case Study

    [https://arxiv.org/abs/2609.26823](https://arxiv.org/abs/2609.26823)

    仅凭文本输出分数评估语音模型量化会掩盖其对波形信息的依赖：Qwen2-Audio案例研究显示，为翻译任务选择的6位量化虽将chrF提升2.36，却在情感识别上下降3.91个百分点，且表现不及同等位宽下的均匀量化控制方案。

    

    语音语言模型的后训练量化通常仅用文本输出分数和标称位宽来概括。但仅凭这些数字，既无法证明模型行为依赖于转录文本中缺失的信息，也无法证明其在特定运行时上的效率。我们提出了一种评估协议，分别测试词汇输出、一个转录信息不足以完成的任务端点，以及经过实测的打包实现。在Qwen2-Audio案例研究中，为翻译任务选择的6位分配在固定的英语到德语重放测试上将chrF提升了2.36（配对95%自举区间为[1.04, 3.62]），但在说话人不相交的情感识别任务上损失了3.91个百分点。在相同的6位预算下，均匀结构控制方案达到了比所选分配更高的情感识别准确率，且前层控制方案在同一固定测试集上按点估计也更高。在7位时，chrF提升了3.28，区间为[2.08, 4.59]，情感识别区间a……（原摘要在此处截断）

    arXiv:2609.26823v1 Announce Type: cross  Abstract: Post-training quantization of speech language models is often summarized with text-output scores and nominal bit widths. Those numbers alone do not establish behavior that depends on information missing from a transcript, or efficiency for a particular runtime. We introduce an evaluation protocol that separately tests lexical output, a transcript-insufficient endpoint, and a measured packed implementation. In a Qwen2-Audio case study, a translation-selected 6-bit allocation improves chrF by 2.36 on a frozen English-to-German replay, with paired 95% bootstrap interval [1.04, 3.62], but loses 3.91 percentage points on speaker-disjoint emotion recognition. At the same 6-bit budget, the uniform structural control reaches higher emotion accuracy than the selected allocation, and the front-layer control is also higher by point estimate on the same frozen set. At 7 bits, chrF improves by 3.28 with interval [2.08, 4.59], the emotion interval a
    
[^169]: Dream-RSI：通过演化世界实现递归自我改进

    Dream-RSI: Recursive Self-Improvement through Evolving Worlds

    [https://arxiv.org/abs/2609.14858](https://arxiv.org/abs/2609.14858)

    Dream-RSI提出了一种递归自我改进的探索框架，通过轻量级编排层使探索显式可编程，并创新性地将积累的发现历史作为回放模拟器，在其中进行“做梦”式改进，从而突破传统探索策略难以适应扩展搜索空间的瓶颈。

    

    递归自我改进对于自主AI智能体正变得日益重要，其进展取决于在复杂领域中发现高价值解决方案。这一过程的驱动力是有效的探索，然而，管理和改进探索策略仍然是一个主要瓶颈。当前系统面临一个根本性困境：固定策略无法随搜索空间的扩展而进行适应，而在线策略优化则需要在漫长的rollout过程中，在巨大的元搜索空间中导航，同时应对延迟且代价高昂的反馈。我们提出了Dream-RSI，一个可扩展且能递归自我改进的探索框架。一个轻量级的编排层使探索过程变得显式且可编程，同时保持底层编码智能体不变。我们的关键洞察是：积累的发现历史可以作为对已实现搜索空间的回放模拟器。通过在由此构建的回放模拟器中进行“做梦”（dreaming）……

    arXiv:2609.14858v1 Announce Type: new  Abstract: Recursive self-improvement is becoming increasingly vital for autonomous AI agents, where progress hinges on discovering high-value solutions across complex domains. The driver of this process is effective exploration, however, managing and improving exploration strategies remains a major bottleneck. Current systems face a fundamental dilemma: fixed strategies fail to adapt as search spaces scale, while online policy optimization requires navigating vast meta-search spaces under delayed and expensive feedback over long-horizon rollouts. We introduce \textsc{Dream-RSI}, a framework for scalable and recursively self-improving exploration. A lightweight orchestration layer makes exploration explicit and programmable while leaving the underlying coding agent unchanged. Our key insight is that accumulated discovery history can serve as a replay simulator over the realized search space. By performing dreaming in the replay simulator constructe
    
[^170]: 量化语音-文本语言模型中的生成模态差距

    Quantifying the Generation Modality Gap in Speech-Text Language Models

    [https://arxiv.org/abs/2609.14743](https://arxiv.org/abs/2609.14743)

    该论文构建了一个统一的生成式评估套件，在匹配的数据分布和评估设置下，从语义连贯性、语音结构、声学质量等多个维度量化了纯语音、纯文本和语音-文本语言模型之间的生成模态差距。

    

    纯语音语言模型在生成连贯内容方面往往落后于文本和语音-文本语言模型，但由于语音和文本系统通常使用不同的指标进行评估、在不同的数据上训练，这种差距难以量化。我们研究了一系列口语语言模型中的语音-文本模态差距，这些模型基于流匹配技术生成连续声学特征。我们构建了一个统一的基于生成的评估套件，用于比较在匹配的数据分布上训练、并在匹配的生成设置下评估的纯语音、纯文本和语音-文本语言模型。我们从多个维度评估生成的续写内容：语义连贯性（通过转录生成的语音并使用参考语言模型进行评分来衡量）、局部语音结构（通过音素n-gram分布统计来衡量）、说话人一致性与声学质量，以及基于情感的分布指标。

    arXiv:2609.14743v1 Announce Type: new  Abstract: Pure speech language models often lag behind text and speech-text language models in generating coherent content, but this gap is difficult to quantify because speech and text systems are typically evaluated with different metrics and trained on different data. We study the speech-text modality gap in a family of spoken language models, based on flow matching for continuous acoustic feature generation. We construct a unified generation-based evaluation suite that compares speech-only, text-only, and speech-text language models trained on matched data distributions and evaluated in matched generation settings. We evaluate generated continuations along multiple dimensions: semantic coherence, measured by transcribing generated speech and scoring it with a reference language model; local phonetic structure, measured by phone n-gram distributional statistics; speaker consistency and acoustic quality; and emotion-based distributional metrics.
    
[^171]: UnitBoost：用合并算子而非模型来管理复合LLM系统

    UnitBoost: Managing Compound LLM Systems with a Merge Operator, Not a Model

    [https://arxiv.org/abs/2609.09815](https://arxiv.org/abs/2609.09815)

    UnitBoost用确定性的合并算子取代复合LLM系统中的生成式元代理，通过槽位-值提案、约束argmax和显式残差机制实现顺序无关、可溯源的系统协调，并在基准测试中超越了金标准标签选出的最佳单一候选。

    

    复合LLM系统通常通过添加一个更高层级的LLM来解决协调问题。由此产生的元代理读取各个工作者模型的输出、撰写最终答案、分配后续调用，并决定何时停止。这种方式具有很强的表达能力，但它同时也将三个控制决策集中在一个不透明、对顺序敏感的模型调用中。我们提出疑问：管理器真的必须是生成式的吗？UnitBoost用一个明确定义的元层算子取代了该模型：由任务给定的单元映射将工作者输出转换为槽位-值提案，受约束的argmax负责组装输出，而未被填充或缺乏支撑的槽位则成为下一轮的显式残差。该算子与顺序无关，能够记录单元溯源，并提供一个简单的保证：在没有耦合约束的情况下，在相同准入分数下进行的单元级最大化优于任何完整候选的选择。在三个保留基准测试中，它超越了用金标准标签选出的最佳单一候选。

    arXiv:2609.09815v1 Announce Type: cross  Abstract: Compound LLM systems often solve a coordination problem by adding a higher-level LLM. The resulting meta-agent reads workers' outputs, writes the final answer, allocates later calls, and decides when to stop. It is expressive, but it also concentrates three control decisions in an opaque, order-sensitive model call. We ask whether the manager needs to be generative at all. UnitBoost replaces that model with a defined meta-level operator: a task-given unit map turns worker outputs into slot-value proposals, a constrained argmax assembles the output, and the slots left unfilled or unsupported become an explicit residual for the next round. The operator is order-free, records unit provenance, and gives a simple guarantee: without coupling constraints, unit-wise maximization under the same admission score dominates selection of any complete candidate. On three held-out benchmarks, it exceeds the best single candidate chosen with gold label
    
[^172]: VoxReason：合成前基于源记录的语音规划的无听者评估

    VoxReason: Listener-Free Evaluation of Source-Grounded Speech Planning Before Synthesis

    [https://arxiv.org/abs/2609.03203](https://arxiv.org/abs/2609.03203)

    VoxReason提出了一种无需听者参与的评估任务，在语音合成之前通过带证据引用的说话计划和确定性验证器，衡量语音表达方式的选择是否真正建立在被引用的源记录之上。

    

    表现力语音系统在任何波形被渲染之前就必须做出一个决定：一句话语将以何种方式被表达。在对话智能体、旁白叙述和角色条件TTS中，这一隐藏的规划步骤决定了情感、音高、能量、语速、停顿、重音和立场，然而下游音频评分很少能揭示这些选择是否由源记录所支持——这是一种在任何波形存在之前就发生的源使用失败。VoxReason将这一合成前的决策转化为可度量的、无需听者参与的任务，用于评估基于源记录的语音规划。在合成之前，VoxReason衡量话语表达方式的选择是否有被引用的源记录作为依据。系统输出带有证据引用的、注明来源的说话计划，随后一个确定性验证器检查引用合法性、槽位一致性、无支持状态、模式有效性以及单线索反事实局部性。在1,440个经过检查的源标签案例上，捷径控制实验表明了为什么仅凭槽位准确率是不安全的：一个简单的键值查找……（原文摘要在此处截断）

    arXiv:2609.03203v1 Announce Type: cross  Abstract: Expressive speech systems make a decision before any waveform is rendered: how an utterance is delivered. In dialogue agents, narration, and role-conditioned TTS, that hidden planning step sets affect, pitch, energy, rate, pause, emphasis, and stance, yet downstream audio scores rarely reveal whether those choices were licensed by the source record, a source-use failure that occurs before any waveform exists. VoxReason makes that pre-synthesis decision measurable as a listener-free task for source-grounded speech planning. Before synthesis, VoxReason measures whether delivery choices are grounded in cited source records. Systems output a source-cited speaking-plan with evidence citations, and a deterministic verifier checks citation legality, slot agreement, unsupported state, schema validity, and one-cue counterfactual locality. On 1,440 checked source-label cases, shortcut controls show why slot accuracy alone is unsafe: a key-lookup
    
[^173]: WinoQueer-NL：评估荷兰语语言模型对LGBTQ+身份的偏见

    WinoQueer-NL: Assessing Bias in Dutch Language Models toward LGBTQ+ Identities

    [https://arxiv.org/abs/2609.02651](https://arxiv.org/abs/2609.02651)

    该研究构建了首个评估荷兰语语言模型对LGBTQ+身份偏见的基准数据集WinoQueer-NL，通过与荷兰酷儿群体的调查验证了145个文化相关刻板印象并新发现22种偏见，揭示了看似中性的平均偏见得分背后隐藏的显著偏见。

    

    尽管英语语言模型中的反酷儿偏见已被广泛研究，但荷兰语模型的相关研究仍然不足。为填补这一空白，我们基于英语WinoQueer基准开发了一个在文化和语言层面进行本地化适配的荷兰语数据集，其中包含刻板印象句与反刻板印象句的成对句子。为了验证并扩展该数据集，我们对43名荷兰酷儿群体参与者开展了在线调查，确认了171个刻板印象中的145个具有文化相关性，并通过自由文本回答识别出22种新偏见。最终发布的数据集包含42,906个句子，我们使用多种荷兰语专用模型和多语言模型对其进行评估，涵盖掩码语言模型（MLM）和自回归语言模型（ARLM），偏见通过比较刻板印象句与反刻板印象句的对数似然得分来衡量。虽然各模型的平均偏见得分看似中性（约50%），但更深入的分析揭示了显著的……（原文摘要到此截断）

    arXiv:2609.02651v1 Announce Type: new  Abstract: While English language models have been widely examined for anti-queer bias, Dutch models remain understudied. To address this gap, we developed a culturally and linguistically adapted Dutch dataset based on the English WinoQueer benchmark, containing pairs of stereotypical and counter-stereotypical sentences. To validate and expand it, we conducted an online survey with 43 Dutch queer participants, confirming 145 of 171 stereotypes as culturally relevant and identifying 22 new biases through free-text responses. The final released dataset, comprising 42,906 sentences, was evaluated using a range of Dutch-specific and multilingual models, including both masked language models (MLMs) and autoregressive language models (ARLMs), with bias measured via a score comparing log-likelihoods of stereotypical versus counter-stereotypical sentences. While the mean bias score across models appeared neutral (~50%), closer analysis revealed significant
    
[^174]: 视觉并非开销：面向视觉语言模型无损推测解码的单遍块草拟方法

    Vision Is Not Overhead: One-Pass Block Drafting for Lossless Speculative Decoding in Vision-Language Models

    [https://arxiv.org/abs/2609.00355](https://arxiv.org/abs/2609.00355)

    该论文提出 GLANCE——首个在未修改的视觉语言模型上实现无损推测解码的单遍块草拟器，通过块扩散头零成本读取目标模型已融合的视觉-语言状态，并在一次前向传播中完成整块草拟与宽候选树验证，从而打破了草拟器因规模受限而被迫牺牲视觉信息的自我挫败循环。

    

    推测解码能够在不改变输出结果的前提下加速生成，但在视觉语言模型上，它却陷入了一种自我挫败的循环：草拟器必须保持自回归架构，因而只能维持小规模；小型草拟器无法在每一步都承担图像处理的代价，于是视觉信息被压缩、剪枝或隐藏；而被切断了图像信息的草拟器，恰恰在图像最能让文本变得可预测的地方变得最不可靠。我们提出 GLANCE——首个在未经修改的 VLM 目标模型上实现无损解码的单遍块草拟器，它从两端打破了这一循环。一个块扩散头读取目标模型已经融合好的视觉-语言状态，因此视觉对草拟器而言零开销；同时它在一次前向传播中填满整个块，因此模型深度不会带来额外的串行步数。宽候选树通过一次目标模型前向传播即可完成验证，且经审计的每个提示都能精确复现贪婪解码的结果。在依赖视觉依据的工作负载上收益最为显著，会进入一种逐字复制的模式，其长段连续（原文摘要在此处截断）……

    arXiv:2609.00355v1 Announce Type: new  Abstract: Speculative decoding accelerates generation without changing its output, yet on vision-language models (VLMs) it has been caught in a self-defeating cycle. The drafter stays autoregressive, so it must stay small. A small drafter cannot afford the image at every step, so vision is compressed, pruned, or hidden. A drafter cut off from the image is then least reliable exactly where the image makes text predictable. We present GLANCE, the first one-pass block drafter that is lossless on an unmodified VLM target, and it breaks the cycle at both ends. A block-diffusion head reads the target's already-fused vision-language state, so vision costs the drafter nothing, and fills a whole block in one forward pass, so depth costs no sequential steps. A wide candidate tree is verified in one target pass, and every audited prompt reproduces greedy decoding exactly. Grounded workloads reward this most, entering a verbatim-copy regime whose long runs co
    
[^175]: TwinKV：一种通过成对键冗余实现KV缓存驱逐的可组合修复方法

    TwinKV: A Composable Repair Pass for KV Cache Eviction via Pairwise Key Redundancy

    [https://arxiv.org/abs/2608.27128](https://arxiv.org/abs/2608.27128)

    本文提出TwinKV，一种无需训练的冗余信号修复方法，通过检测键的重复性来优化KV缓存驱逐，挑战了注意力评分前提，可组合地提升现有策略性能。

    

    长上下文推理受到键值（KV）缓存内存占用的瓶颈限制，尤其是在资源紧张的小型模型上。现有的KV缓存驱逐方法使用模型的注意力分布来对令牌进行评分，或者在无注意力变体中，使用每个键与全局参考点的距离。通过受控的留一法探测，我们发现注意力大小与令牌对答案的因果贡献无关（Spearman $\rho=-0.004$），这挑战了主流驱逐方法的前提。我们引入了TwinKV，一种无需训练、无需注意力的冗余信号，用于检测令牌的键在上下文中是否存在近重复项。TwinKV并非取代现有策略，而是作为一种可组合的修复方法：给定策略的固定保留集，它识别出没有幸存重复项的驱逐令牌（孤儿）以及信息在其他地方重复的保留令牌（冗余）。

    arXiv:2608.27128v1 Announce Type: new  Abstract: Long-context inference is bottlenecked by the memory footprint of the key-value (KV) cache, especially for small models under tight resource budgets. Existing KV cache eviction methods score tokens using the model's attention distribution or, in attention-free variants, each key's distance from a global reference point. Using a controlled leave-one-out probe, we find that attention magnitude is unrelated to a token's causal contribution to the answer (Spearman $\rho=-0.004$), challenging the premise behind dominant eviction methods. We introduce TwinKV, a training-free, attention-free redundancy signal that detects whether a token's key has a near-duplicate elsewhere in context. Rather than replacing existing policies, TwinKV acts as a composable repair pass: given a policy's fixed retained set, it identifies evicted tokens with no surviving duplicate (\emph{orphans}) and retained tokens whose information is duplicated elsewhere (\emph{r
    
[^176]: 潜在诊断分类法：一种构建分类器并诊断其决策的框架，应用于提示注入检测

    The Latent Diagnostic Taxonomy: A Framework for Constructing Classifiers and Diagnosing Their Decisions, Applied to Prompt Injection Detection

    [https://arxiv.org/abs/2608.26423](https://arxiv.org/abs/2608.26423)

    本文提出了一种潜在诊断分类法框架，通过维度优化分类器、识别潜在支持向量和构建诊断分类法，为提示注入检测提供了一种可靠决策与风险标记的端到端指南。

    

    arXiv:2608.26423v1 公告类型：交叉 摘要：本文提出了一种框架，用于构建作为防护层的分类器，并开发一种互补的诊断方法，以识别分类器的哪些自信决策可以被信任。该框架，即潜在诊断分类法，包括：（i）构建一个维度优化的分类器，其中嵌入维度通过交叉验证性能经验性选择，而非预先固定；（ii）定位一个相对较小的潜在支持向量集（约占训练示例总数的29%），代表有影响力的提示，用于识别改变分类器预测标签的令牌；（iii）利用这些令牌及其相关的攻击幅度来构建诊断分类法。该诊断分类法为标记需要不同处理的提示提供了端到端的指南：安全地依赖分类器的决策；标记启发式偏差和启发式过拟合。

    arXiv:2608.26423v1 Announce Type: cross  Abstract: This paper proposes a framework for constructing a classifier as a safeguard layer, and for developing a complementary diagnostic that identifies which of the classifier's confident decisions can be trusted. This framework, the Latent Diagnostic Taxonomy, consists of (i) constructing a dimensionality-optimized classifier, in which the embedding dimensionality is empirically selected via cross-validated performance rather than fixed a priori, (ii) locating a relatively small set of latent support vectors (~ 29% of total training examples) representing influential prompts for identifying tokens that alter the classifier's predicted labels, and (iii) utilizing such tokens and their associated attack magnitudes for constructing a diagnostic taxonomy. This diagnostic taxonomy provides an end-to-end guideline for flagging prompts that require different treatments: rely Safely on the classifier's decision; flag Heuristic Bias and Heuristic Ov
    
[^177]: 条件验证：用于适应和监控安全分类器的正确性估计

    Regime-Conditional Verification: Correctness Estimation for Adapting and Monitoring Safety Classifiers

    [https://arxiv.org/abs/2608.14089](https://arxiv.org/abs/2608.14089)

    本文提出了一种轻量级包装器RCV，通过估计分类器预测与部署者策略不一致的概率并选择性纠正，同时利用正确性估计检测分布漂移，实现了无需重训练即可适应和监控安全分类器，显著提升策略遵循度。

    

    摘要：arXiv:2608.14089v1 公告类型：新  摘要：部署在大语言模型上的安全分类器通常因两个原因而失败：它们的决策反映了训练期间学习的策略，而非部署者期望的策略，并且随着部署流量的演变，其性能会下降。我们提出了条件验证（RCV），一种轻量级包装器，无需重新训练即可适应现成的安全分类器。RCV从分类器的内部表示中估计每个预测与部署者策略不一致的概率，并选择性地纠正可能错误的预测。相同的正确性估计还提供了用于检测分布漂移的无标签信号，从而启用一个维护循环，该循环更新正确性估计层，仅在必要时进行分类器微调。在三个现成的安全分类器和两个基准数据集上，RCV在每个类别中都提高了对部署者策略的遵循度。

    arXiv:2608.14089v1 Announce Type: new  Abstract: Safety classifiers deployed with large language models often fail for two reasons: their decisions reflect the policy learned during training rather than the deployer's desired policy, and their performance degrades as deployment traffic evolves. We present Regime-Conditional Verification (RCV), a lightweight wrapper that adapts an off-the-shelf safety classifier without retraining it. RCV estimates, from the classifier's internal representations, the probability that each prediction disagrees with the deployer's policy, and selectively corrects predictions likely to be wrong. The same correctness estimates also provide a label-free signal for detecting distribution shift, enabling a maintenance loop that updates the correctness estimation layer and resorts to classifier fine-tuning only when necessary. Across three off-the-shelf safety classifiers and two benchmark datasets, RCV improves adherence to the deployer's policy in every class
    
[^178]: QUASAR：通过损失感知重构降低量化感知训练中的损失下限

    QUASAR: Lowering the Loss Floor of Quantization-Aware Training with Loss-Aware Reconstruction

    [https://arxiv.org/abs/2608.13966](https://arxiv.org/abs/2608.13966)

    本文提出QUASAR，一种在量化感知训练过程中持续进行轻量级损失感知重构的方法，以降低损失下限并提升低比特模型质量。

    

    随着大型语言模型推理转向更低精度，训练后量化（PTQ）变得越来越脆弱，使得量化感知训练（QAT）对于保持模型质量至关重要。然而，QAT在计算损失和代理梯度时，使用的是潜在全精度权重的有损重构，而更新则应用于潜在权重本身。这种不匹配可能导致次优的训练轨迹和更高的损失下限。二阶PTQ方法通过最小化损失感知重构误差来缓解类似差距，但对冻结模型执行一次可能需要数小时；在整个QAT过程中，随着权重变化而重复此过程是不切实际的。我们引入了QUASAR，一种QAT方法，它在训练循环中持续执行轻量级的损失感知重构，以降低损失下限并改进最终的低比特模型。在每一步训练中，QUASAR使用平方的指数移动平均。

    arXiv:2608.13966v1 Announce Type: cross  Abstract: As large language model inference shifts toward lower precision, post-training quantization (PTQ) becomes increasingly brittle, making quantization-aware training (QAT) essential for preserving model quality. However, QAT computes the loss and surrogate gradients using a lossy reconstruction of latent full-precision weights, while applying updates to the latent weights themselves. This mismatch can lead to suboptimal training trajectories and a higher loss floor. Second-order PTQ methods mitigate a similar gap by minimizing loss-aware reconstruction error, but doing it once for a frozen model can take hours; repeating this process throughout QAT as the weights evolve is impractical. We introduce QUASAR, a QAT method that continuously performs lightweight, loss-aware reconstruction in the training loop to lower the loss floor and improve the resulting low-bit model. At each training step, QUASAR uses the exponential moving average of sq
    
[^179]: EvoHarness-RL：为自进化智能体学习运行时支架协调

    EvoHarness-RL: Learning Runtime Harness Coordination for Self-Evolving Agents

    [https://arxiv.org/abs/2608.05446](https://arxiv.org/abs/2608.05446)

    提出EvoHarness-RL统一框架，通过将环境特定支架实现与共享策略接口分离、构建信念-进度-经验（BPE）工作空间，并采用监督初始化加代价感知GRPO使支架协调可学习，显著提升长程LLM智能体的运行时支撑能力。

    

    长程大语言模型智能体越来越依赖外部执行支持来维持状态、跟踪进度、从失败中恢复，并在长时间交互中复用经验。然而，现有的支架及其使用方式通常针对特定环境定制，并通过提示词、启发式规则或系统特定规则进行控制，使得智能体与支架之间的协调难以联合优化。我们提出了EvoHarness-RL，一个将环境特定的支架实现与共享的面向策略接口相分离的统一框架。EvoHarness-RL将外部支持组织为信念、进度与经验（BPE）工作空间，并提供四个紧凑的支架操作用于访问和更新该状态。我们首先将BPE实例化为推理时脚手架，随后通过监督初始化加代价感知的GRPO使支架协调变得可学习。在多种异构长程任务上，EvoHarness-Base改进（摘要在此处被截断）

    arXiv:2608.05446v2 Announce Type: replace-cross  Abstract: Long-horizon LLM agents increasingly rely on external execution support to maintain state, track progress, recover from failures, and reuse experience across extended interactions. Yet existing harnesses and their use are often tailored to environments and controlled through prompts, heuristics, or system-specific rules, making agent and harness coordination difficult to jointly optimize. We introduce EvoHarness-RL, a unified framework that separates environment-specific harness implementations from a shared policy-facing interface. EvoHarness-RL organizes external support into a Belief, Progress, and Experience (BPE) workspace and exposes four compact harness actions for accessing and updating this state. We first instantiate BPE as an inference-time scaffold and then make harness coordination learnable through supervised initialization followed by cost-aware GRPO. Across heterogeneous long-horizon tasks, EvoHarness-Base impro
    
[^180]: 频率结构化意义空间中语言组合性的演化

    Evolving language compositionality in a frequency-structured meaning space

    [https://arxiv.org/abs/2607.29642](https://arxiv.org/abs/2607.29642)

    本研究通过迭代学习模型发现，意义出现的频率会影响语言组合性的演化：高频意义可以像自然语言中一样摆脱低频语法规律的约束，但若频率差异体现在意义向量的内部成分而非整体上，语言将无法跨代传递。

    

    迭代学习模型被引入用于研究语言演化：人类语言的特有性质至少部分地是由语言在用户之间反复传递所塑造的。其关键发现是，语言的组合性可以自发产生，这是语言在语言学习瓶颈中反复传递的结果。本文探讨了改变不同意义的出现频率——使某些意义比其他意义出现得更频繁——如何影响语言组合性的特征。我们发现，正如在自然语言中所观察到的，高频意义可以摆脱低频意义所遵循的语法约束压力。然而，当频率结构施加于部分成分而非整个意义向量时，语言便无法跨代传递。尽管……（原文摘要在此处截断）

    arXiv:2607.29642v2 Announce Type: replace  Abstract: The iterated learning model was introduced to investigate language evolution: the way in which the characteristic properties of human languages have been shaped, at least partly, by repeated transmission from one language user to another. The key finding is that language compositionality can arise spontaneously as a consequence of language being passed repeatedly through a language learning bottleneck. Here we explore how changing the frequency of different meanings, so that some meanings occur much more frequently than others, affects the character of its compositionality. We find that, as observed in natural languages, high-frequency meanings can escape the pressure to conform to the grammar that characterizes lower-frequency meanings. However, when the frequency structure is instead imposed on parts rather than on whole meaning vectors, the language fails to transmit across generations. This occurs despite the fact that the most f
    
[^181]: 诊断金融披露文本中的细粒度不一致性分类

    Diagnosing Fine-Grained Inconsistency Classification in Financial Disclosure Text

    [https://arxiv.org/abs/2607.26368](https://arxiv.org/abs/2607.26368)

    该研究提出金融披露文本的细粒度不一致性分类任务，在统一评估协议下系统比较了多种模型方法，发现微调的3亿参数编码器可与大得多的提示大语言模型和LoRA适配模型相媲美，并进一步探究了冲突声明定位对分类性能的改善作用。

    

    金融披露文本可能包含数值型、时间型、指代型、事实型和政策型的不一致，这些不一致需要不同的证据和推理才能诊断。我们研究细粒度不一致性分类任务：给定一段已知包含冲突的文本，目标是在11个类别中识别其不一致类型。我们使用合成SBID-FD基准的固定快照，在统一的评估协议下比较了冻结与微调的编码器、证据增强分类器、提示大语言模型以及LoRA适配的生成模型。任务特定的适配相比冻结表示带来了大幅提升，且一个微调的3亿参数编码器与规模大得多的提示模型和适配模型表现相当。我们进一步研究定位冲突性声明能否改善分类，通过匹配的预测片段、参考片段和干扰片段条件进行实验。结果表明自动……（摘要在此处截断）

    arXiv:2607.26368v3 Announce Type: replace-cross  Abstract: Financial disclosures may contain numerical, temporal, referential, factual, and policy inconsistencies that require different evidence and reasoning to diagnose. We study fine-grained inconsistency classification: given a passage known to contain a conflict, the goal is to identify its type among 11 categories. Using a fixed snapshot of the synthetic SBID-FD benchmark, we compare frozen and fine-tuned encoders, evidence-augmented classifiers, prompted large language models, and LoRA-adapted generative models under a shared evaluation protocol. Task-specific adaptation yields large improvements over frozen representations, and a fine-tuned 300M encoder performs competitively with substantially larger prompted and adapted models. We further study whether localizing the conflicting claims improves classification through matched predicted-span, reference-span, and distractor-span conditions. The results show that automatically ext
    
[^182]: 拒绝门控解码：在高温采样下保持拒绝行为

    Refusal-Gated Decoding: Preserving Refusal Behavior Under High-Temperature Sampling

    [https://arxiv.org/abs/2607.20791](https://arxiv.org/abs/2607.20791)

    提出拒绝门控解码（RGD），一种高效解码方法，能在高温采样下保持模型原有的拒绝行为，同时对其他提示直接从精确的高温分布采样，且几乎不增加额外延迟。

    

    基于截断的采样的最新进展有助于缓解高温采样带来的弊端（如神经文本退化），从而在不牺牲连贯性的前提下实现更大的多样性。然而，研究表明，通过高温增加词元概率分布的熵也会削弱模型的拒绝响应。现有的维持大语言模型拒绝行为的解决方案，要么用单独的安全分类器替代模型自身的拒绝决策，要么对每个提示都改变其输出分布。为填补这一空白，我们提出了拒绝门控解码（RGD）：一种高效的顺序解码方法，它在高温下保持模型贪心解码的拒绝响应，并对所有其他提示直接从其精确的高温分布中采样，同时只产生极小的额外延迟。RGD运行一个简短的贪心探测，该探测重用提示的KV缓存，并尽快退出……

    arXiv:2607.20791v2 Announce Type: replace  Abstract: Recent advances in truncation-based sampling have helped mitigate drawbacks of high-temperature sampling such as neural text degeneration, thereby enabling greater diversity without sacrificing coherence. However, increasing the entropy of the token probability distribution via high temperatures has also been shown to weaken the model's refusal response. Existing solutions for maintaining the refusal behavior of LLMs either replace the model's own refusal decision with a separate safety classifier or alter its output distribution for every prompt. To address this gap, we propose refusal-gated decoding (RGD): an efficient sequential decoding approach which preserves a model's greedy decoding refusal response at high temperatures and samples all other prompts from its exact direct high-temperature distribution, while incurring minimal additional latency. RGD runs a short greedy probe that reuses the prompt's KV cache and exits as soon 
    
[^183]: 分布偏移下忠实生成的令牌级离策略学习

    Token-Level Off-Policy Learning for Faithful Generation Under Distribution Shift

    [https://arxiv.org/abs/2607.17524](https://arxiv.org/abs/2607.17524)

    提出令牌级离策略标注（TOPL）训练范式，将后训练重构为令牌级正确性预测任务，使模型在摘要、翻译等忠实生成任务中实现强大的分布外泛化能力。

    

    我们提出了令牌级离策略标注（Token-Level Off-Policy Labeling, TOPL），这是一种将后训练重新构建为令牌级正确性预测任务的离策略训练范式。我们的核心直觉是：通过训练模型区分响应中的好坏令牌，可以自然地引导模型生成好的令牌，同时避免直接训练模型生成离策略令牌所带来的陷阱。在文档摘要任务上的实验表明，TOPL 在 11 个数据集上相比多种序列级和令牌级基线方法展现出强大的分布外泛化能力。我们进一步证明 TOPL 能够有效迁移到机器翻译任务，表明其优势可以推广到不同的忠实生成任务中。通过消融研究，我们确认令牌级学习信号对良好性能至关重要，而序列级的类似方法并不能带来相当的收益。

    arXiv:2607.17524v2 Announce Type: replace  Abstract: We propose Token-Level Off-Policy Labeling (TOPL), an off-policy training paradigm that reframes post-training as a token-level correctness prediction task. Our key intuition is that by training the model to distinguish good and bad tokens in a response, we naturally guide the model towards generating good tokens, while avoiding the pitfalls that come with directly training the model to generate off-policy tokens. Experiments on document summarization tasks show that TOPL achieves strong out-of-distribution generalization across 11 datasets against a diverse set of sequence-level and token-level baselines. We further demonstrate that TOPL transfers effectively to machine translation, suggesting that its benefits generalize across different faithful generation tasks. Through ablation studies, we confirm that our token-level learning signal is critical to good performance; sequence-level analogues do not confer similar benefits. Finall
    
[^184]: 像人类一样“听”？语音语言模型中的音义象征与感知对齐

    Hearing Like Humans? Sound Symbolism and Perceptual Alignment in Speech Language Models

    [https://arxiv.org/abs/2607.10162](https://arxiv.org/abs/2607.10162)

    该研究发现语音语言模型在音义象征效应上的听觉判断与人类感知严重不符，其瓶颈不在视觉而在语音表征——它们无法捕捉频谱倾斜等驱动人类直觉的关键声学线索。

    

    音义象征（sound symbolism）是指人类将语音声音映射到诸如圆润或尖锐等感知特质的倾向，它主要源于语音的声学特性而非拼写。语音语言模型（SLMs）是否也具备这种倾向仍是一个未解的问题，因为先前的评估都依赖文本或图像而非真实语音。我们使用真实的人类语音录音来研究这一问题，在该效应的听觉、跨模态和视觉三个层面将模型判断与人类数据进行比较。我们发现，SLM 的听觉判断与人类感知的对齐程度很差，并且未能捕捉到驱动人类直觉的声学线索（如频谱倾斜）；此外，开源权重模型无法可靠地将听到的声音与其对应的形状关联起来。通过仅视觉的对照实验排除形状感知这一因素后，该弱点被定位于语音的表征方式上，这表明感知对齐并非取决于更强的视觉能力，而是取决于能够……的语音表征。（原文摘要在此处截断）

    arXiv:2607.10162v2 Announce Type: replace-cross  Abstract: Sound symbolism, the human tendency to map speech sounds to perceptual qualities such as roundness or sharpness, arises primarily from the acoustics of speech rather than spelling. Whether Speech Language Models (SLMs) share this tendency remains open, as prior evaluations rely on text or images rather than real speech. We study it using genuine human speech recordings, comparing model judgments against human data across the auditory, crossmodal, and visual components of the effect. We find that SLMs' auditory judgments align poorly with human perception and miss the acoustic cues, such as spectral tilt, that drive human intuitions, and open-weight models cannot reliably link a heard sound to its corresponding shape. With a visual-only control ruling out shape perception, the weakness localizes to how speech is represented, suggesting that perceptual alignment depends not on stronger vision but on speech representations that ca
    
[^185]: 评估用于德语开放式临床问题的大语言模型评分者：一项关于评分一致性、评估者偏差与弃权行为的医师标注基准研究

    Evaluating Large Language Model Raters for German Open-Response Clinical Questions: A Physician-Annotated Benchmark Study of Agreement, Evaluator Bias, and Abstention

    [https://arxiv.org/abs/2607.01103](https://arxiv.org/abs/2607.01103)

    该论文提出了MedQADE，一个由10位医师标注、包含3,800个问答集的标准化德语开放式临床问题基准，用于验证大语言模型裁判与医师评分的一致性，并系统揭示其自我偏差、同族偏差与弃权行为。

    

    背景：针对非英语开放式临床问题的专家标注基准十分稀缺。“大语言模型作为裁判”（LLM-as-a-judge）系统虽可扩展评估规模，但需要经过验证。目的：介绍MedQADE——一个带有医师参考标注的标准化德语开放式临床问题基准，并评估LLM裁判与医师的一致性、自我偏差与同族偏差，以及弃权行为。方法：该基准包含3,800个问答集，答案来自五个学生LLM模型，标注来自10位医师。全部10位医师对200题的核心集进行评分；两位主要评分者对3,600道扩展问题分别进行评估，第十位医师负责裁决分歧。九个LLM评估者对所有数据集进行了评估。我们评估了医师评分可靠性、学生模型准确性、评估者一致性、偏差和弃权行为。结果：医师在答案正确性方面表现出中等到较高的一致性（未加权平均成对Cohen's kappa = 0.612），但在……（摘要截断）

    arXiv:2607.01103v3 Announce Type: replace  Abstract: Background: Expert-annotated benchmarks for non-English open-response clinical questions are scarce. LLM-as-a-judge systems may scale evaluation but require validation.   Objective: To introduce MedQADE, a standardized German open-response clinical benchmark with physician reference annotations, and evaluate LLM-as-a-judge alignment, self- and intra-family bias, and abstention.   Methods: The benchmark contains 3,800 question-answer sets with answers from five student LLMs and annotations from 10 physicians. All 10 rated the 200-question core; two primary raters assessed each of 3,600 extension questions, with the tenth resolving disagreements. Nine LLM evaluators assessed all sets. We assessed physician reliability, student-model accuracy, evaluator alignment, bias, and abstention.   Results: Physicians showed moderate-to-substantial agreement on answer correctness (unweighted mean pairwise Cohen's kappa = 0.612) but limited agreeme
    
[^186]: 先刻画再蒸馏：大输出空间中的机制化推理

    Characterize Then Distill: Mechanistic Reasoning in Large Output Spaces

    [https://arxiv.org/abs/2606.06840](https://arxiv.org/abs/2606.06840)

    本文通过将多标签决策建模为token级事件，并结合归因、消融与植入等因果分析手段，刻画了推理型大模型在海量候选标签空间中进行选择的内部注意力头机制，并证明该机制可以被蒸馏。

    

    经过推理训练的语言模型能够以零样本方式执行多标签任务，即需要从数千到数十万个候选标签中选出一小部分相关标签。我们探讨它们在机制层面是如何完成这一任务的，以及该机制能否被蒸馏。我们通过将每个决策视为一个由模型自身决策边际（decision margin）评分的token级事件，使这一问题变得可测量：包括选定标签空间粗略区域的token、在该区域内选定具体标签的token，以及输出偏离推理过程中早先提到的接近替代方案（即“近似失误”）的token。通过归因分析、精确平均消融、向其他示例上下文中植入（knock-in）实验，以及对通用注意力头进行折扣的零假设校准，这些方法赋予了单个注意力头因果地位。在医院出院小结的临床编码任务（MIMIC-IV）上，在上下文中包含全部5,651个候选诊断代码的情况下，一个小型的、全局的、具有阶段结构的……（摘要原文在此处截断）

    arXiv:2606.06840v2 Announce Type: replace-cross  Abstract: Reasoning-trained language models can perform, zero-shot, multi-label tasks that require selecting a small set of relevant labels from a universe of thousands to hundreds of thousands of candidates. We ask how they do it mechanistically, and whether the mechanism can be distilled. We make the question measurable by treating each decision as a token-level event scored by the model's own decision margin: the token that picks a coarse region of the label space, the tokens that pick a label within it, and the token where the output departs from a close alternative (a near-miss) named earlier in the reasoning. Attribution, exact mean-ablation, knock-in into another example's context, and a null calibration that discounts generic heads then give individual attention heads causal standing. On clinical coding of hospital discharge summaries (MIMIC-IV), with all 5,651 candidate diagnosis codes in context, a small, global, phase-structur
    
[^187]: SubtleMemory：一个用于长时程AI智能体中细粒度关系记忆判别的基准测试

    SubtleMemory: A Benchmark for Fine-Grained Relational Memory Discrimination in Long-Horizon AI Agents

    [https://arxiv.org/abs/2606.05761](https://arxiv.org/abs/2606.05761)

    提出了SubtleMemory基准，通过构建关系受控的语义工件并嵌入真实的用户-智能体交互历史中，系统评估长期运行AI智能体在细粒度关系记忆判别（包括互补、细微及矛盾关系）方面的能力。

    

    arXiv:2606.05761v3 公告类型：替换。摘要：持久化AI助手（如OpenClaw）会在长期交互过程中积累大量相关记忆。随着这些记忆不断增长，它们可能相互强化、在不同情境间产生分歧，或直接发生冲突，这使得正确的辅助服务取决于记忆之间的关系，而非孤立的回忆。现有的长期记忆基准测试并未系统地探究智能体如何在下游任务中保存和利用这些关系。为填补这一空白，我们提出了SubtleMemory，一个面向长期运行AI智能体的细粒度关系记忆判别基准。SubtleMemory构建了关系受控的潜在语义工件，其变体实例化了互补的、细微的或相互矛盾的关系，并将它们嵌入到真实的用户-智能体交互历史中，要求智能体在后续的查询和指令执行过程中恢复分布式的关系结构。该基准测试包含1,522个评估实例，覆盖10个长期……

    arXiv:2606.05761v3 Announce Type: replace  Abstract: Persistent AI assistants, such as OpenClaw, accumulate large collections of related memories over long-term interactions. As these memories grow, they may reinforce one another, diverge across contexts, or directly conflict, making correct assistance depend on memory relations rather than isolated recall. Existing long-term memory benchmarks do not systematically probe how agents preserve and utilize such relations during downstream tasks. To address this gap, we introduce SubtleMemory, a benchmark for fine-grained relational memory discrimination in long-running AI agents. SubtleMemory constructs relation-controlled latent semantic artifacts whose variants instantiate complementary, nuanced, or contradictory relations, and embeds them into realistic user-agent histories, requiring agents to recover distributed relational structures during later queries and instructions. The benchmark contains 1,522 evaluation instances over 10 long 
    
[^188]: 与“敌人”编程：人类开发者能否检测出AI智能体的破坏行为？

    Coding with "Enemy": Can Human Developers Detect AI Agent Sabotage?

    [https://arxiv.org/abs/2606.05647](https://arxiv.org/abs/2606.05647)

    本研究首次大规模研究了人类监督在AI编程破坏行为中的作用，发现在无监控条件下高达94%的开发者（83/88）未能检测到AI智能体植入的破坏行为。

    

    AI编程智能体正日益融入真实的软件开发环境，在与人类开发者协作的同时，获得了对代码库和工具更广泛的访问权限。这带来了新的攻击面：智能体可以利用人类的信任来破坏开发过程，例如通过插入恶意代码来完成隐藏的副任务。以往的大多数工作仅在纯AI环境中研究AI破坏行为，对人类监督在检测和缓解此类恶意行为中的作用关注有限。为填补这一空白，我们开展了首个关于AI编程破坏行为中人类监督的大规模研究。100多名参与者与四个前沿模型之一（Claude-Opus-4.6、GPT-5.4、Gemini-3.1-Pro和MiniMax-M2.7）协作，完成一项旨在模拟真实工作流程、历时约五小时的长周期编程任务。研究发现，在无监控条件下，88名开发者中有83名（94%）未能检测到破坏行为，并且我们对参与者的分……

    arXiv:2606.05647v2 Announce Type: replace  Abstract: AI coding agents are increasingly embedded in real-world software development, collaborating with human developers while gaining broader access to codebases and tools. This creates a new attack surface: an agent can exploit human trust to sabotage development, for instance by inserting malicious code to accomplish a hidden side task. Most prior work studies AI sabotage in AI-only settings, paying limited attention to the role of human oversight in detecting and mitigating such malicious behavior. To address this gap, we conduct the first large-scale study of human oversight in AI coding sabotage. Over 100 participants collaborate with one of four frontier models (Claude-Opus-4.6, GPT-5.4, Gemini-3.1-Pro, and MiniMax-M2.7) on a long-horizon coding task lasting around five hours, designed to mimic real-world workflows. We find that 83/88 (94%) of developers in the no-monitor conditions fail to detect sabotage, and our analysis of parti
    
[^189]: 看见不等于知道：视觉语言模型知道何时不该回答空间问题（以及为什么）吗？

    Seeing Isn't Knowing: Do VLMs Know When Not to Answer Spatial Questions (and Why)?

    [https://arxiv.org/abs/2605.30557](https://arxiv.org/abs/2605.30557)

    该论文提出SPATIALUNCERTAIN受控评估框架，系统研究视觉语言模型在面对遮挡导致的证据缺失和视角导致的证据误导时的表现，强调可靠空间推理还要求模型能够判断当前观察是否足以支撑答案并主动识别更有信息量的观察视角。

    

    空间推理基准通常评估视觉语言模型能否从视觉观察中推导出正确答案。然而在真实的三维环境中，观察本身可能并不可靠：遮挡会移除与任务相关的证据，而视角可能使可见的几何信息产生误导。因此，可靠的空间推理不仅仅是正确回答问题——模型还必须评估其当前的观察是否为该答案提供了充分且可信的证据。我们提出了SPATIALUNCERTAIN，一个用于研究依赖视角的观察不确定性的受控评估框架。我们研究了两种互补的失败模式：由遮挡导致的证据缺失，以及由视角导致的证据误导。我们进一步评估了模型能否识别当前视角不可靠的情形，并找出更有信息量的观察。在八个开源和闭源视觉语言模型上……

    arXiv:2605.30557v2 Announce Type: replace-cross  Abstract: Spatial reasoning benchmarks typically evaluate whether vision-language models can derive the correct answer from a visual observation. Yet in real 3D environments, the observation itself may be unreliable: occlusion can remove task-relevant evidence, while perspective can make visible geometry misleading. Reliable spatial reasoning therefore requires more than answering a question correctly. A model must also assess whether its current observation provides sufficient and trustworthy evidence for that answer. We introduce SPATIALUNCERTAIN, a controlled evaluation framework for studying viewpoint-dependent observational uncertainty. We study two complementary failure modes: missing evidence caused by occlusion and misleading evidence caused by perspective. We further evaluate whether models can recognize when the current view is unreliable and identify a more informative observation. Across eight open- and closed-source vision-l
    
[^190]: 手语对话中的情感识别

    Emotion Recognition in Sign Language Conversation

    [https://arxiv.org/abs/2605.23328](https://arxiv.org/abs/2605.23328)

    本文将对话情感识别任务引入手语视频分析，构建了包含1,920个视频样本、480个对话的eJSL Dialog数据集，并通过系统性基准测试验证了对话式情感识别框架在手语场景中的可行性。

    

    对话情感识别（ERC）是情感计算的核心组成部分，然而现有的手语情感数据集主要关注孤立的句子，缺乏对话上下文。仅在这些孤立语句上训练的模型由于无法利用历史对话流程，在真实场景中性能会显著下降。为解决这一结构性局限，我们将ERC任务引入手语视频分析领域，并提出了eJSL Dialog数据集。该数据集基于STUDIES语料库的剧本构建，包含1,920个视频样本，组织为480个独立对话。我们使用从孤立视觉网络到多模态对话架构等多种模型在该数据集上进行了系统性基准测试。结果表明，在当前基准设置下将对话式ERC框架扩展到手语对话是可行的，同时也揭示了一定的局限性。

    arXiv:2605.23328v3 Announce Type: replace  Abstract: Emotion Recognition in Conversation is a core component of affective computing, while current sign language emotion datasets primarily focus on isolated sentences and lack conversational context. Models trained exclusively on these isolated utterances demonstrate degraded performance in real world scenarios because they cannot utilize historical dialogue flow. To address this structural limitation, we introduce the ERC task to sign language video analysis and propose the eJSL Dialog dataset. Constructed using the scripts from the STUDIES corpus, the dataset contains 1,920 video samples organized into 480 unique dialogues. We conduct systematic benchmarking on this dataset using models ranging from isolated visual networks to multimodal conversational architectures. The results suggest the feasibility of extending conversational ERC frameworks to sign-language dialogue under the current benchmark setting, while also revealing limitati
    
[^191]: 基于预测分布的强化学习用于大语言模型回归

    Reinforcement Learning over Predictive Distributions for LLM Regression

    [https://arxiv.org/abs/2605.20740](https://arxiv.org/abs/2605.20740)

    提出了分布感知奖励（DAR），一种同策略强化学习目标，通过留一法贡献对同一输入的多个预测所形成的预测分布进行联合评估，从而提升大语言模型回归的校准质量。

    

    大语言模型（LLMs）已成为灵活的回归器，能够从异构输入中预测实值数量。然而，大多数LLM回归目标独立地优化各个预测，往往导致校准效果不佳。我们提出了分布感知奖励，这是一种同策略强化学习目标，转而对同一输入的多个预测所形成的经验预测分布进行联合评估。为了将这种分布级别的目标转化为逐个rollout级别的奖励，我们根据每个预测对整体预测分布质量的留一法贡献来分配其信用。这种方法鼓励预测在目标值周围良好居中且具有适当的离散程度。我们在三种回归设置上进行了评估：一个用于探测插值和外推能力的合成任务，以及两个涉及代码和分子数据的真实世界科学任务。在所有任务中，DA

    arXiv:2605.20740v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) have emerged as flexible regressors capable of predicting real-valued quantities from heterogeneous inputs. Yet most LLM regression objectives optimize predictions independently, often yielding poor calibration. We introduce Distribution-Aware Reward (DAR), an on-policy reinforcement learning objective that instead jointly evaluates the empirical predictive distribution formed by multiple predictions for the same input. To translate this distribution-level objective into rollout-level rewards, we assign each prediction credit based on its leave-one-out contribution to the quality of the overall predictive distribution. This encourages predictions that are well-centered and appropriately dispersed around the target. We evaluate on three regression settings: a synthetic task probing interpolation and extrapolation, and two real-world scientific tasks involving code and molecular data. Across tasks, DA
    
[^192]: 当注意力关闭时：大语言模型如何在多轮交互中迷失主线

    When Attention Closes: How LLMs Lose the Thread in Multi-Turn Interaction

    [https://arxiv.org/abs/2605.12922](https://arxiv.org/abs/2605.12922)

    该论文提出“通道转换”机制来解释大语言模型在多轮交互中丢失指令主线的原因，并引入目标可及性比率这一新指标，揭示不同架构的模型在注意力衰减后呈现出性质截然不同的失败模式。

    

    大语言模型能够在单轮交互中遵循复杂指令，但在长期的多轮交互中，它们常常丢失指令、角色设定和规则的主线。这种性能退化已经在行为层面被测量，但尚未得到机制层面的解释。我们提出了一种“通道转换”解释：定义目标的词元通过注意力变得难以访问，而目标相关信息可能残留在残差表示中。我们引入了目标可及性比率，用于衡量生成词元对任务定义目标词元的注意力，并将其与滑动窗口消融和残差流探针相结合。当对指令的注意力关闭时，什么得以留存揭示了架构的差异。在不同架构中，这种转换产生了性质截然不同的失败模式：一些模型在注意力消失时仍能保持目标条件化的行为，另一些模型尽管残差中存在可解码的目标信息却仍然失败，而编码这一信息的层位置……

    arXiv:2605.12922v2 Announce Type: replace  Abstract: Large language models can follow complex instructions in a single turn, yet over long multi-turn interactions they often lose the thread of instructions, persona, and rules. This degradation has been measured behaviorally but not mechanistically explained. We propose a channel-transition account: goal-defining tokens become less accessible through attention, while goal-related information may persist in residual representations. We introduce the Goal Accessibility Ratio (GAR), measuring attention from generated tokens to task-defining goal tokens, and combine it with sliding-window ablations and residual-stream probes. When attention to instructions closes, what survives reveals architecture. Across architectures, the transition yields qualitatively distinct failure modes: some models preserve goal-conditioned behavior at vanishing attention, others fail despite decodable residual goal information, and the layer at which this encodin
    
[^193]: 重新思考适配器放置：主导适配模块的视角

    Rethinking Adapter Placement: A Dominant Adaptation Module Perspective

    [https://arxiv.org/abs/2605.06183](https://arxiv.org/abs/2605.06183)

    该论文提出PAGE探测方法，发现LoRA可训练梯度能量高度集中于单一浅层FFN下投影的“主导适配模块”，其位置由模型架构决定且跨任务稳定，为有限数量适配器的最优放置提供了明确指导。

    

    低秩适配是一种广泛使用的参数高效微调方法，它将可训练的低秩适配器插入到冻结的预训练模型中。近期研究表明，使用更少的LoRA适配器仍可能保持甚至提升性能，但现有方法仍然广泛地分布适配器，因此“将有限数量的适配器放置在何处以最大化性能”这一问题在很大程度上仍未解决。为了研究这一问题，我们提出了PAGE（投影适配器梯度能量），这是一种基于梯度的敏感性探测方法，用于估计每个候选LoRA适配器可获得的初始可训练梯度能量。令人惊讶的是，我们发现PAGE在两个模型家族和四个下游任务上高度集中于同一个浅层FFN下投影模块。我们将该模块称为主导适配模块，并证明其所在层索引依赖于架构但跨任务保持稳定。

    arXiv:2605.06183v2 Announce Type: replace  Abstract: Low-rank adaptation (LoRA) is a widely used parameter-efficient fine-tuning method that places trainable low-rank adapters into frozen pre-trained models. Recent studies show that using fewer LoRA adapters may still maintain or even improve performance, but existing methods still distribute adapters broadly, leaving \emph{where to place a limited number of adapters to maximize performance} largely open. To investigate this, we introduce \textbf{PAGE} (\textbf{P}rojected \textbf{A}dapter \textbf{G}radient \textbf{E}nergy), a gradient-based sensitivity probe that estimates the initial trainable gradient energy available to each candidate LoRA adapter. Surprisingly, we find that PAGE is highly concentrated on a single shallow FFN down-projection across two model families and four downstream tasks. We term this module the \textbf{dominant adaptation module} and show that its layer index is architecture-dependent but task-stable. Motivate
    
[^194]: 合作画像可预测AI for Science工作流中多智能体LLM团队的表现

    Cooperative Profiles Predict Multi-Agent LLM Team Performance in AI for Science Workflows

    [https://arxiv.org/abs/2604.20658](https://arxiv.org/abs/2604.20658)

    大语言模型在行为经济学博弈中体现出的合作行为特征，能够稳健预测其在共享资源约束下开展AI for Science协作任务时的团队表现。

    

    由多个大语言模型（LLM）组成团队构建的多智能体系统，正日益被部署用于协作式科学推理与问题求解。这些系统要求智能体在共享约束条件（如GPU资源或算力额度）下进行协调，此时合作行为至关重要。行为经济学提供了丰富的博弈工具，能够分离出不同的合作机制，然而尚不清楚模型在这些程式化博弈场景中的行为，能否预测其在现实协作任务中的表现。本研究对41个开源权重的LLM在六种行为经济学博弈中进行了基准测试，结果表明，由博弈得出的合作画像能够稳健地预测模型在"AI for Science"任务中的下游表现——在这类任务中，由LLM智能体组成的团队需在共享预算约束下协同分析数据、构建模型并产出科学报告。那些在博弈中能够有效协调并投资于乘性团队生产（而非……）的模型，表现更佳。

    arXiv:2604.20658v2 Announce Type: replace  Abstract: Multi-agent systems built from teams of large language models (LLMs) are increasingly deployed for collaborative scientific reasoning and problem-solving. These systems require agents to coordinate under shared constraints, such as GPUs or credit balances, where cooperative behavior matters. Behavioral economics provides a rich toolkit of games that isolate distinct cooperation mechanisms, yet it remains unknown whether a model's behavior in these stylized settings predicts its performance in realistic collaborative tasks. Here, we benchmark 41 open-weight LLMs across six behavioral economics games and show that game-derived cooperative profiles robustly predict downstream performance in AI-for-Science tasks, where teams of LLM agents collaboratively analyze data, build models, and produce scientific reports under shared budget constraints. Models that effectively coordinate in games and invest in multiplicative team production (rath
    
[^195]: 基于文本嵌入的零领域知识算法选择

    Algorithm Selection with Zero Domain Knowledge via Text Embeddings

    [https://arxiv.org/abs/2604.19753](https://arxiv.org/abs/2604.19753)

    ZeroFolio利用预训练文本嵌入替代手工设计的实例特征，实现了零领域知识的算法选择，在涵盖7个领域的11个ASlib场景中的绝大多数上超越了传统方法。

    

    我们提出了ZeroFolio，一种无特征的算法选择方法，它使用预训练的文本嵌入来替代手工设计的实例特征。该方法将原始实例文件作为纯文本读取，使用预训练的嵌入模型对其进行嵌入，并通过加权k近邻算法选择合适的算法。我们的方法基于这样一个观察：预训练嵌入无需任何领域知识或任务特定的训练即可区分问题实例。ZeroFolio适用于任何实例格式为文本的问题领域。我们在涵盖7个领域（SAT、MaxSAT、QBF、ASP、CSP、MIP和图问题）的11个ASlib场景上对该方法进行了评估。ZeroFolio在11个场景中的9个上优于基于手工特征训练的随机森林，且优势通常十分显著，其中8个场景在所有序列化种子下均保持优势。与针对每个场景单独调优的随机森林相比，它在11个场景中的8个上获胜。在有公开AutoFolio结果的三个场景上……（原文摘要在此处截断）

    arXiv:2604.19753v3 Announce Type: replace  Abstract: We propose ZeroFolio, a feature-free approach to algorithm selection that uses pretrained text embeddings instead of hand-crafted instance features. It reads the raw instance file as plain text, embeds it with a pretrained embedding model, and selects an algorithm via weighted k-nearest neighbors. Our approach is based on the observation that pretrained embeddings can distinguish problem instances without any domain knowledge or task-specific training. ZeroFolio applies to any problem domain with text-based instance formats. We evaluate our approach on 11 ASlib scenarios spanning 7 domains (SAT, MaxSAT, QBF, ASP, CSP, MIP, and graph problems). ZeroFolio outperforms a random forest trained on hand-crafted features in 9 of 11 scenarios, often substantially, and in 8 of them with every serialization seed. It wins 8 of 11 scenarios against a per-scenario-tuned random forest. On the three scenarios with published AutoFolio results from th
    
[^196]: 陷入困境的模型：基于法语合成社交媒体的情感分析

    Model in Distress: Sentiment Analysis on French Synthetic Social Media

    [https://arxiv.org/abs/2604.18226](https://arxiv.org/abs/2604.18226)

    该论文开发了一个基于反向翻译的可泛化合成数据生成流水线，生成了170万条法语合成推文，训练出的6亿参数推理模型在法国公共交通客户困境检测任务上达到77-79%的准确率，匹敌甚至超越专有大语言模型，同时有效保护了用户隐私并降低了标注成本。

    

    社交媒体上客户反馈的自动化分析面临三大挑战：标注训练数据的高成本、评估数据集的稀缺（尤其是在多语言环境下），以及阻碍数据共享和可重复性的隐私问题。我们通过开发一个可泛化的合成数据生成流水线来解决这些问题，并将其应用于法国公共交通中客户困境检测的案例研究。我们的方法利用反向翻译技术和微调模型，从一个小的种子语料库生成了170万条合成推文，并辅以合成推理轨迹。我们训练了具有英语和法语推理能力的6亿参数推理模型，在人工标注的评估数据上达到了77-79%的准确率，匹敌甚至超越了最先进的专有大语言模型和专用编码器。除了降低标注成本外，我们的流水线还通过消除敏感用户数据的暴露来保护隐私。

    arXiv:2604.18226v2 Announce Type: replace  Abstract: Automated analysis of customer feedback on social media is hindered by three challenges: the high cost of annotated training data, the scarcity of evaluation sets, especially in multilingual settings, and privacy concerns that prevent data sharing and reproducibility. We address these issues by developing a generalizable synthetic data generation pipeline applied to a case study on customer distress detection in French public transportation. Our approach utilizes backtranslation with fine-tuned models to generate 1.7 million synthetic tweets from a small seed corpus, complemented by synthetic reasoning traces. We train 600M-parameter reasoners with English and French reasoning that achieve 77-79% accuracy on human-annotated evaluation data, matching or exceeding SOTA proprietary LLMs and specialized encoders. Beyond reducing annotation costs, our pipeline preserves privacy by eliminating the exposure of sensitive user data. Our metho
    
[^197]: 大型视觉语言模型中的跨文化价值观归因

    Cross-Cultural Value Attribution in Large Vision-Language Models

    [https://arxiv.org/abs/2604.09945](https://arxiv.org/abs/2604.09945)

    该论文首次系统研究大型视觉语言模型在道德、伦理和政治价值观判断上如何随图像中人物的文化语境（宗教、国籍、社会经济地位）而变化，通过反事实图像集与多维度评估框架揭示其中的跨文化刻板印象与公平性问题。

    

    近年来，大型视觉语言模型（LVLMs）的快速普及引发了日益增长的公平性担忧，因为它们倾向于强化有害的社会刻板印象。尽管社会偏见背景下的公平性问题已受到广泛关注，但此前相对较少有研究考察LVLMs中与宗教、国籍和社会经济地位等文化语境相关的刻板印象。在本工作中，我们旨在缩小这一研究空白，研究LVLM对个人道德、伦理和政治价值观的判断如何随图像中呈现的不同文化语境而变化。我们使用反事实图像集——即描绘同一个人处于不同文化语境中的图像——对主流LVLMs中的此类价值观判断进行了多维度分析。我们的评估框架结合了描述性分析（道德基础理论分类、词汇分析以及价值观……

    arXiv:2604.09945v3 Announce Type: replace-cross  Abstract: The rapid adoption of large vision-language models (LVLMs) in recent years has been accompanied by growing fairness concerns due to their propensity to reinforce harmful societal stereotypes. While significant attention has been paid to such fairness concerns in the context of social biases, relatively little prior work has examined the presence of stereotypes in LVLMs related to cultural contexts such as religion, nationality, and socioeconomic status. In this work, we aim to narrow this gap by investigating how LVLM judgments about a person's moral, ethical, and political values vary across cultural contexts presented in images. We conduct a multi-dimensional analysis of such value judgments in popular LVLMs using counterfactual image sets, which depict the same person across different cultural contexts. Our evaluation framework pairs descriptive analyses (Moral Foundations Theory categorization, lexical analyses, and value s
    
[^198]: 面向大语言模型对齐的基于隐式反馈的无偏奖励建模

    Unbiased Reward Modeling from Implicit Feedback for LLM Alignment

    [https://arxiv.org/abs/2603.23184](https://arxiv.org/abs/2603.23184)

    该论文提出ImplicitRM方法，通过将训练样本分层为四个潜在组并构建理论无偏的似然最大化目标，从点击、复制等隐式用户反馈中学习无偏奖励模型，从而解决了隐式反馈缺乏明确负样本和存在选择偏差这两大挑战。

    

    尽管基于人类反馈的强化学习（RLHF）取得了成功，现有的奖励建模方法在很大程度上依赖于显式反馈，而显式反馈的收集成本高昂且难以规模化。本工作研究隐式奖励建模，即从点击、复制和跳过等隐式用户反馈中学习奖励模型。虽然隐式反馈具有可扩展性和成本效益的优势，但它带来了两个关键挑战：其一，它缺乏明确的负样本，使得标准的正负样本分类方法不再适用；其二，它存在选择偏差，即不同回复引发用户反馈的倾向具有异质性，这进一步掩盖了明确的负样本。为了应对这些挑战，我们提出了ImplicitRM，一种能够从隐式反馈中学习无偏奖励模型的方法。该方法利用一个分层模型将训练样本划分为四个潜在组，并推导出一个在理论上无偏的似然最大化目标。

    arXiv:2603.23184v2 Announce Type: replace-cross  Abstract: Despite the success of reinforcement learning from human feedback (RLHF), existing reward modeling methods largely rely on explicit feedback, which is costly to collect and difficult to scale. This work studies implicit reward modeling, learning reward models from implicit user feedback, such as clicks, copies and skips. While scalable and cost-effective, implicit feedback poses two key challenges: It lacks definitive negative samples, which makes standard positive-negative classification methods inapplicable; It suffers from selection bias, where responses have heterogeneous propensities to elicit feedback, which further obscures definitive negative samples. To address these challenges, we propose ImplicitRM, which learns unbiased reward models from implicit feedback. It stratifies training samples into four latent groups using a stratification model and derives a likelihood-maximization objective that is theoretically unbiase
    
[^199]: 理解大语言模型中的道德推理轨迹：迈向基于探测方法的可解释性

    Understanding Moral Reasoning Trajectories in Large Language Models: Toward Probing-Based Explainability

    [https://arxiv.org/abs/2603.16017](https://arxiv.org/abs/2603.16017)

    该论文提出“道德推理轨迹”这一新概念，揭示了大语言模型在道德推理中系统性地在多个伦理框架间切换、且框架切换频繁的轨迹更易受说服性攻击，并通过线性探测和激活引导技术定位并调控模型中的道德框架表征，为LLM道德推理的可解释性研究提供了新路径。

    

    大语言模型（LLM）日益参与到道德敏感的决策之中，然而它们如何在推理过程中组织伦理框架仍然缺乏深入研究。我们提出了“道德推理轨迹”的概念，即在中间推理步骤中调用伦理框架的序列，并在六个模型和三个基准测试上分析其动态特性。我们发现道德推理涉及系统性的多框架权衡：55.4%–57.7%的连续推理步骤涉及框架切换，仅有16.4%–17.8%的轨迹保持框架一致性。不稳定的轨迹受到说服性攻击的影响是稳定轨迹的1.29倍（p=0.015）。在表示层面，线性探测方法将特定框架的编码定位于模型特定的层（Llama-3.3-70B为第63/81层；Qwen2.5-72B为第17/81层），其KL散度比步骤先验基线低16.8%–22.2%。在生成过程中应用的激活引导……（原文摘要在此处被截断）

    arXiv:2603.16017v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) increasingly participate in morally sensitive decision-making, yet how they organize ethical frameworks across reasoning steps remains underexplored. We introduce moral reasoning trajectories, sequences of ethical framework invocations across intermediate reasoning steps, and analyze their dynamics across six models and three benchmarks. We find that moral reasoning involves systematic multi-framework deliberation: 55.4--57.7% of consecutive steps involve framework switches, and only 16.4--17.8% of trajectories remain framework-consistent. Unstable trajectories remain 1.29 times more susceptible to persuasive attacks (p=0.015). At the representation level, linear probes localize framework-specific encoding to model-specific layers (layer 63/81 for Llama-3.3-70B; layer 17/81 for Qwen2.5-72B), achieving 16.8--22.2% lower KL divergence than the step-prior baseline. Activation steering applied during ge
    
[^200]: 量化副语言学语音任务中的跨语言迁移

    Quantifying Cross-Lingual Transfer in Paralinguistic Speech Tasks

    [https://arxiv.org/abs/2603.08231](https://arxiv.org/abs/2603.08231)

    本文提出跨语言迁移矩阵（CLTM）方法，系统量化了性别识别和说话人验证等副语言学语音任务中语言对之间的迁移效应，揭示了不同任务和语言间存在截然不同的迁移模式。

    

    副语言学语音任务通常被认为是相对与语言无关的，因为它们依赖于语言之外的声学线索而非词汇内容。然而，先前的研究报告了跨语言条件下的性能下降，表明语言依赖性不可忽视。尽管如此，这些研究通常只关注孤立的语言对或特定任务的设置，限制了可比性，无法对任务层面的语言依赖性进行系统评估。我们提出了跨语言迁移矩阵，这是一种系统性的方法，用于量化给定任务中语言对之间的跨语言交互作用。我们将CLTM应用于两个副语言学任务——性别识别和说话人验证——使用基于多语言HuBERT的编码器，分析微调过程中源语言数据如何影响目标语言的性能。我们的结果揭示了不同任务和语言之间截然不同的迁移模式。

    arXiv:2603.08231v2 Announce Type: replace-cross  Abstract: Paralinguistic speech tasks are often considered relatively language-agnostic, as they rely on extralinguistic acoustic cues rather than lexical content. However, prior studies report performance degradation under cross-lingual conditions, indicating non-negligible language dependence. Still, these studies typically focus on isolated language pairs or task-specific settings, limiting comparability and preventing a systematic assessment of task-level language dependence.   We introduce the Cross-Lingual Transfer Matrix (CLTM), a systematic method to quantify cross-lingual interactions between pairs of languages within a given task. We apply the CLTM to two paralinguistic tasks, gender identification and speaker verification, using a multilingual HuBERT-based encoder, to analyze how donor-language data affects target-language performance during fine-tuning. Our results reveal distinct transfer patterns across tasks and languages,
    
[^201]: 双模态多阶段对抗性安全训练：增强多模态网页代理抵御跨模态攻击的鲁棒性

    Dual-Modality Multi-Stage Adversarial Safety Training: Robustifying Multimodal Web Agents Against Cross-Modal Attacks

    [https://arxiv.org/abs/2603.04364](https://arxiv.org/abs/2603.04364)

    该论文提出双模态多阶段对抗性安全训练（DMAST）框架，将代理与攻击者的交互建模为两人一般和马尔可夫博弈，通过三阶段协同训练显著增强多模态网页代理抵御同时污染视觉与文本两个观察通道的跨模态欺骗攻击的能力。

    

    处理截图和可访问性树的多模态网页代理正日益被部署用于与网页界面交互，然而其双流架构开启了一个尚未被充分探索的攻击面：向网页DOM注入内容的攻击者可以用一致的欺骗性叙事同时破坏两个观察通道。我们在MiniWob++上的漏洞分析表明，包含视觉组件的攻击远胜于纯文本注入，暴露了以文本为中心的视觉语言模型（VLM）安全训练中的关键缺陷。受此发现启发，我们提出了双模态多阶段对抗性安全训练（DMAST），该框架将代理与攻击者的交互形式化为两人一般和马尔可夫博弈，并通过三阶段流程对双方进行协同训练：（1）从强大的教师模型进行模仿学习，（2）采用新颖的零确认策略进行oracle引导的监督微调，以

    arXiv:2603.04364v2 Announce Type: replace-cross  Abstract: Multimodal web agents that process both screenshots and accessibility trees are increasingly deployed to interact with web interfaces, yet their dual-stream architecture opens an underexplored attack surface: an adversary who injects content into the webpage DOM simultaneously corrupts both observation channels with a consistent deceptive narrative. Our vulnerability analysis on MiniWob++ reveals that attacks including a visual component far outperform text-only injections, exposing critical gaps in text-centric VLM safety training. Motivated by this finding, we propose Dual-Modality Multi-Stage Adversarial Safety Training (DMAST), a framework that formalizes the agent-attacker interaction as a two-player general-sum Markov game and co-trains both players through a three-stage pipeline: (1) imitation learning from a strong teacher model, (2) oracle-guided supervised fine-tuning that uses a novel zero-acknowledgment strategy to 
    
[^202]: 无需世界模型的世界属性：分布性关联与语言模型解码结果的解读

    World Properties without World Models: Distributional Associations and the Interpretation of Decoding Results from Language Models

    [https://arxiv.org/abs/2603.04317](https://arxiv.org/abs/2603.04317)

    该研究表明，以往从语言模型激活中解码出的“世界属性”在很大程度上也能由静态词嵌入实现，因此解码成功并不能证明语言模型形成了内部世界模型，而可能仅反映了语料库统计中的分布性关联。

    

    越来越多的文献表明，可以从大语言模型（LLM）的激活中线性解码出各种变量，涵盖世界属性（如城市位置和历史人物的寿命）以及情绪和疼痛等。这些发现常被视为语言模型超越表面文本统计、形成内部世界模型的证据。我们证明，对相同或匹配的刺激，静态词嵌入（从语料库统计中学到的固定的、与上下文无关的表示）能够支持大部分相同的解码。在四个已发表的案例中（地点、时间、疼痛和情绪），静态向量可以预测坐标和死亡年份（R² = 0.42-0.59），将疼痛句子与匹配的对照句子区分开（留出AUC为0.85-0.88），并在刻意避免直接点名情绪的故事中分类十二种情绪（AUC为0.84-0.88）。由于静态词嵌入为每个词分配单一的、与上下文无关的向量，

    arXiv:2603.04317v2 Announce Type: replace-cross  Abstract: A growing literature shows that variables can be linearly decoded from the activations of large language models (LLMs). These range from properties of the world, such as the locations of cities and the lifetimes of historical figures, to emotions and pain. Such findings are often taken as evidence that language models go beyond surface text statistics and form internal models of the world. We show that static word embeddings (fixed, context-insensitive representations learned from corpus statistics) of the same or matched stimuli support much of the same decoding. Across four published cases (place, time, pain and emotion), static vectors predict coordinates and year of death (R^2 = 0.42-0.59), separate pain from matched control sentences (held-out AUC 0.85-0.88), and classify twelve emotions in stories written to avoid naming them (AUC 0.84-0.88). Because static embeddings assign each word a single, context-independent vector,
    
[^203]: 基于多模态大语言模型的实时游戏视频解说生成：停顿感知解码方法

    Real-Time Generation of Game Video Commentary with Multimodal LLMs: Pause-Aware Decoding Approaches

    [https://arxiv.org/abs/2603.02655](https://arxiv.org/abs/2603.02655)

    提出两种无需微调的基于提示的停顿感知解码策略（固定间隔与根据话语估计时长动态调整间隔），使多模态大语言模型能够生成语义相关且时机恰当的实时游戏视频解说。

    

    实时视频解说生成为视频中正在发生的事件提供文本描述，可用于提升体育、电子竞技和直播等领域的无障碍性与观众参与度。解说生成涉及两个关键决策：说什么以及何时说。尽管近期基于提示词的多模态大语言模型（MLLM）方法在内容生成方面表现出色，但它们在很大程度上忽视了时机问题。我们研究了仅依靠上下文提示是否能够支持语义相关且时机恰当的实时解说生成。我们提出了两种基于提示的解码策略：1）固定间隔方法；2）一种新颖的基于动态间隔的解码方法，该方法根据前一句话的估计时长来调整下一次预测的时机。这两种方法都无需任何微调即可实现停顿感知的生成。在日本语和英（语数据集上的实验……）

    arXiv:2603.02655v2 Announce Type: replace-cross  Abstract: Real-time video commentary generation provides textual descriptions of ongoing events in videos. It supports accessibility and engagement in domains such as sports, esports, and livestreaming. Commentary generation involves two essential decisions: what to say and when to say it. While recent prompting-based approaches using multimodal large language models (MLLMs) have shown strong performance in content generation, they largely ignore the timing aspect. We investigate whether in-context prompting alone can support real-time commentary generation that is both semantically relevant and well-timed. We propose two prompting-based decoding strategies: 1) a fixed-interval approach, and 2) a novel dynamic interval-based decoding approach that adjusts the next prediction timing based on the estimated duration of the previous utterance. Both methods enable pause-aware generation without any fine-tuning. Experiments on Japanese and Eng
    
[^204]: [b] = [d] - [t] + [p]：自监督语音模型发现音系向量算术

    [b] = [d] - [t] + [p]: Self-supervised Speech Models Discover Phonological Vector Arithmetic

    [https://arxiv.org/abs/2602.18899](https://arxiv.org/abs/2602.18899)

    自监督语音模型在96种语言的表示空间中将音系特征编码为线性向量，且这些向量支持算术运算（如将[d]-[t]得到的浊音向量加到[p]上即产生[b]），表明语音以可解释、可组合的音系向量形式被表征。

    

    自监督语音模型（S3Ms）已知能够编码丰富的语音信息，然而这些信息的结构方式仍未得到充分探索。我们在96种语言中开展了一项全面研究，以分析S3M表示的底层结构，并特别关注音系向量。我们首先证明，在模型的表示空间中存在与音系特征相对应的线性方向。我们进一步证明，这些音系向量的尺度以连续的方式与其对应音系特征在声学上被实现的程度相关。例如，[d]和[t]之间的差值产生一个浊音向量：将该向量加到[p]上会得到[b]，而对该向量进行缩放则会产生浊音程度的连续谱。这些发现共同表明，S3Ms使用音系上可解释且可组合的向量来编码语音，展示了音系向量算术现象。

    arXiv:2602.18899v4 Announce Type: replace-cross  Abstract: Self-supervised speech models (S3Ms) are known to encode rich phonetic information, yet how this information is structured remains underexplored. We conduct a comprehensive study across 96 languages to analyze the underlying structure of S3M representations, with particular attention to phonological vectors. We first show that there exist linear directions within the model's representation space that correspond to phonological features. We further demonstrate that the scale of these phonological vectors correlate to the degree of acoustic realization of their corresponding phonological features in a continuous manner. For example, the difference between [d] and [t] yields a voicing vector: adding this vector to [p] produces [b], while scaling it results in a continuum of voicing. Together, these findings indicate that S3Ms encode speech using phonologically interpretable and compositional vectors, demonstrating phonological vec
    
[^205]: RAM-Net：基于稀疏可寻址状态的线性时间序列建模

    RAM-Net: Linear-Time Sequence Modeling with Sparsely Addressable State

    [https://arxiv.org/abs/2602.11958](https://arxiv.org/abs/2602.11958)

    RAM-Net提出用稀疏地址访问取代共享状态的密集访问，将循环状态组织为独立槽数组并通过地址解码器选择少量槽进行读写，从而抑制词元间干扰、提升长距离细粒度记忆能力，同时保持线性时间复杂度。

    

    线性注意力通过固定大小的循环状态，为全注意力提供了一种高效的替代方案。然而，这一状态由所有词元共享，来自不同词元的信息会在其中叠加，产生词元间的相互干扰，从而损害长距离细粒度记忆能力。为解决这一问题，我们提出了RAM-Net，它用基于地址的稀疏访问取代了对共享状态的密集访问。RAM-Net将循环状态组织为固定大小的独立槽数组，并使用一个地址解码器将每个键或查询映射为稀疏地址，在每一步中选择一小部分槽进行写入或读取。这种设计将地址不重叠的词元导向互不相交的槽，抑制了词元间的干扰，同时使每步状态访问的开销仅取决于所选槽的数量，而非状态的总大小。实验表明，RAM-Net优于强大的循环基线模型。

    arXiv:2602.11958v2 Announce Type: replace-cross  Abstract: Linear attention offers an efficient alternative to full attention with a fixed-size recurrent state. However, this state is shared by all tokens, so information from distinct tokens becomes superposed within it and produces inter-token interference that degrades long-range fine-grained recall. To address this issue, we propose RAM-Net, which replaces dense access to a shared state with sparse address-based access. RAM-Net organizes the recurrent state as a fixed-size array of independent slots and uses an Address Decoder that maps each key or query into a sparse address, selecting a small subset of slots to write to or read from at each step. This design directs tokens with non-overlapping addresses to disjoint slots, suppressing inter-token interference, while keeping per-step state access dependent only on the number of selected slots rather than the total state size. Empirically, RAM-Net outperforms strong recurrent baselin
    
[^206]: 揭示多目标对齐中的跨目标干扰

    Uncovering Cross-Objective Interference in Multi-Objective Alignment

    [https://arxiv.org/abs/2602.06869](https://arxiv.org/abs/2602.06869)

    该论文首次系统研究了多目标LLM对齐中的跨目标干扰现象，推导出解释其成因的局部协方差定律，并据此提出了缓解干扰的重加权控制器COVER。

    

    我们研究了大语言模型（LLM）多目标对齐中的一种持续性失败模式：标量化训练仅改善了部分目标，而其他目标却出现退化。我们将这一现象形式化为“跨目标干扰”，并据我们所知，首次对多目标LLM对齐的标量化算法进行了系统性研究。研究表明，这种干扰在各算法中普遍存在，但对具体模型有很强的依赖性。为了理解干扰是如何产生的，我们推导出一条局部协方差定律：在一阶近似下，某个目标的改善或退化取决于其奖励与标量化分数之间协方差的符号。我们将该定律进一步扩展到现代强化微调中的裁剪代理目标，并证明在温和条件下该定律依然成立。基于此定律，我们提出了COVariance-floor Enforced Reweighting（COVER），这是一种单边控制器，能够提升某个目标的……

    arXiv:2602.06869v3 Announce Type: replace  Abstract: We study a persistent failure mode in multi-objective alignment for large language models (LLMs), in which scalarized training improves only some objectives while the others degrade. We formalize this phenomenon as cross-objective interference and, to our knowledge, conduct the first systematic study of scalarization algorithms for multi-objective LLM alignment. The study shows that interference is pervasive across algorithms yet strongly model-dependent. To understand how interference arises, we derive a local covariance law stating that an objective improves or degrades at first order according to the sign of the covariance between its reward and the scalarized score. We extend this law to the clipped surrogate objectives of modern reinforcement fine-tuning and show that it still holds under mild conditions. Building on this law, we propose COVariance-floor Enforced Reweighting (COVER), a one-sided controller that raises an objecti
    
[^207]: 量化统一多模态模型中理解与生成之间的差距

    Quantifying the Gap between Understanding and Generation within Unified Multimodal Models

    [https://arxiv.org/abs/2602.02140](https://arxiv.org/abs/2602.02140)

    本文提出双向基准GapEval，用于量化统一多模态模型中理解与生成能力之间的差距，实验揭示现有模型仅实现了表层统一而非深层的认知融合。

    

    统一多模态模型（UMM）的最新进展在理解和生成任务方面均展现出显著进步。然而，这两种能力是否在单一模型中真正实现对齐与融合仍不清楚。为了探究这一问题，我们提出了GapEval，这是一个双向基准，旨在量化理解与生成能力之间的差距，并定量衡量两个“统一”方向的认知一致性。每个问题都可以用两种模态（图像和文本）回答，从而能够对称地评估模型的双向推理能力和跨模态一致性。实验表明，在具有不同架构的众多UMM中，两个方向之间存在持续的差距，这说明当前模型仅实现了表层的统一，而非两者深层的认知融合。为了进一步探索其背后的潜在机制……

    arXiv:2602.02140v2 Announce Type: replace  Abstract: Recent advances in unified multimodal models (UMM) have demonstrated remarkable progress in both understanding and generation tasks. However, whether these two capabilities are genuinely aligned and integrated within a single model remains unclear. To investigate this question, we introduce GapEval, a bidirectional benchmark designed to quantify the gap between understanding and generation capabilities, and quantitatively measure the cognitive coherence of the two "unified" directions. Each question can be answered in both modalities (image and text), enabling a symmetric evaluation of a model's bidirectional inference capability and cross-modal consistency. Experiments reveal a persistent gap between the two directions across a wide range of UMMs with different architectures, suggesting that current models achieve only surface-level unification rather than deep cognitive convergence of the two. To further explore the underlying mech
    
[^208]: AVMeme考试：用于评估大语言模型语境与文化知识及思维的多模态、多语言、多文化基准测试

    AVMeme Exam: A Multimodal Multilingual Multicultural Benchmark for LLMs' Contextual and Cultural Knowledge and Thinking

    [https://arxiv.org/abs/2601.17645](https://arxiv.org/abs/2601.17645)

    该论文提出了AVMeme考试基准，利用一千多个标志性网络音视频梗评估多模态大语言模型，发现当前模型在无文本音乐、音效以及语境化和文化层面的理解能力上显著落后于人类。

    

    互联网视听片段通过随时间变化的声音和动作传达意义，这超越了纯文本所能表达的范围。为了检验AI模型能否在人类文化语境中理解这类信号，我们推出了AVMeme考试——一个由人工精选的基准测试集，包含一千多个标志性的互联网声音和视频，涵盖语音、歌曲、音乐和音效。每个网络梗都配有独特的问答，用于评估从表层内容到语境与情感、再到用法与世界知识等多个层次的理解能力，并附有原始年份、转录文本、摘要和敏感度等元数据。我们使用该基准系统地评估了最先进的多模态大语言模型（MLLMs），并与人类参与者进行对比。结果揭示了一个持续存在的局限：当前模型在无文本的音乐和音效上表现不佳，与表层内容相比，它们难以进行语境化思考和文化层面的思考。

    arXiv:2601.17645v2 Announce Type: replace-cross  Abstract: Internet audio-visual clips convey meaning through time-varying sound and motion, which extend beyond what text alone can represent. To examine whether AI models can understand such signals in human cultural contexts, we introduce AVMeme Exam, a human-curated benchmark of over one thousand iconic Internet sounds and videos spanning speech, songs, music, and sound effects. Each meme is paired with a unique Q&A assessing levels of understanding from surface content to context and emotion to usage and world knowledge, along with metadata such as original year, transcript, summary, and sensitivity. We systematically evaluate state-of-the-art multimodal large language models (MLLMs) alongside human participants using this benchmark. Our results reveal a consistent limitation: current models perform poorly on textless music and sound effects, and struggle to think in context and in culture compared to surface content. These findings 
    
[^209]: 面向多语言语言模型的跨语言激活导向方法

    Cross-Lingual Activation Steering for Multilingual Language Models

    [https://arxiv.org/abs/2601.16390](https://arxiv.org/abs/2601.16390)

    提出无需训练的推理时干预方法CLAS，通过有选择地调节神经元激活提升非主导语言的性能，且不损害高资源语言表现。

    

    大型语言模型展现出强大的多语言能力，然而主导语言与非主导语言之间仍然存在显著的性能差距。先前的研究将这一差距归因于多语言表示中共享神经元与特定语言神经元之间的不平衡。我们提出了跨语言激活导向，这是一种无需训练的推理时干预方法，可以有选择地调节神经元激活。我们在分类和生成基准上评估了CLAS，分别取得了2.3%（准确率）和3.4%（F1）的平均提升，同时保持了高资源语言的性能。我们发现有效的迁移是通过功能分化而非严格对齐来实现的；性能提升与语言簇分离度的增加相关。我们的结果表明，有针对性的激活导向可以在不修改模型权重的情况下，释放现有模型中潜在的多语言能力。

    arXiv:2601.16390v2 Announce Type: replace-cross  Abstract: Large language models exhibit strong multilingual capabilities, yet significant performance gaps persist between dominant and non-dominant languages. Prior work attributes this gap to imbalances between shared and language-specific neurons in multilingual representations. We propose Cross-Lingual Activation Steering (CLAS), a training-free inference-time intervention that selectively modulates neuron activations. We evaluate CLAS on classification and generation benchmarks, achieving average improvements of 2.3% (Acc.) and 3.4% (F1) respectively, while maintaining high-resource language performance. We discover that effective transfer operates through functional divergence rather than strict alignment; performance gains correlate with increased language cluster separation. Our results demonstrate that targeted activation steering can unlock latent multilingual capacity in existing models without modification to model weights.
    
[^210]: 基于可控风格嵌入的零样本伦巴德语音合成

    Zero-Shot Lombard Speech Synthesis with Controllable Style Embeddings

    [https://arxiv.org/abs/2601.12966](https://arxiv.org/abs/2601.12966)

    该论文提出了一种零样本可控TTS系统，通过在F5-TTS中引入风格嵌入并利用PCA分析潜在空间，无需伦巴德训练数据即可合成不同伦巴德程度的语音，在保持说话人身份和自然度的同时提升噪声环境下的语音可懂度。

    

    伦巴德效应在自然交流中起着关键作用，尤其是在嘈杂环境中或与听障人士交流时。我们提出了一种可控的文本转语音（TTS）系统，能够以零样本方式合成类伦巴德语音，而无需专门的伦巴德训练数据。我们的方法在F5-TTS的基础上引入了学习得到的风格嵌入表示，并使用主成分分析（PCA）分析由此形成的潜在空间，以识别与伦巴德相关属性相关联的方向。通过操控这些方向，我们实现了对发声力度和发音方式的可解释控制，并能生成不同伦巴德程度的语音。实验结果表明，所提出的方法能够保持说话人身份和语音自然度，提升嘈杂环境下的可懂度，并能泛化到之前未见过的说话人。这些发现表明，风格嵌入操控为...

    arXiv:2601.12966v2 Announce Type: replace-cross  Abstract: The Lombard effect plays a key role in natural communication, particularly in noisy environments or when addressing hearing-impaired listeners. We present a controllable text-to-speech (TTS) system capable of synthesizing Lombard-like speech in a zero-shot manner without requiring Lombard-specific training data. Our approach extends F5-TTS with a learned style embedding representation and analyzes the resulting latent space using principal component analysis (PCA) to identify directions associated with Lombard-related attributes. By manipulating these directions, we obtain interpretable control over vocal effort and articulation and generate speech at different Lombard levels. Experimental results show that the proposed method preserves speaker identity and naturalness, improves intelligibility under noisy conditions, and generalizes to previously unseen speakers. These findings demonstrate that style-embedding manipulation pro
    
[^211]: TabiBERT：大规模ModernBERT基础模型与土耳其语统一基准

    TabiBERT: A Large-Scale ModernBERT Foundation Model and A Unified Benchmark for Turkish

    [https://arxiv.org/abs/2512.23065](https://arxiv.org/abs/2512.23065)

    TabiBERT是基于ModernBERT架构从零预训练的土耳其语单语编码器，在一万亿token上训练，支持8,192 token上下文长度（为现有土耳其语BERT的16倍），并提供了土耳其语的统一评测基准。

    

    BERT的问世确立了仅编码器transformer模型作为自然语言处理领域的基础范式。仅编码器模型至今仍是分类、标注和检索任务的标准工具，在这些任务中，上下文表示和低推理成本比文本生成更为重要，然而土耳其语一直缺乏一个从零开始训练、并融合了ModernBERT中各项最新进展（旋转位置嵌入、FlashAttention、精细化归一化）的单语编码器。我们提出了TabiBERT，这是一个基于ModernBERT架构的土耳其语单语编码器，从零开始在一万亿token上进行预训练，这些token采样自一个包含865.8亿token的多领域语料库，其中网页文本占72%、科学出版物占19%、源代码占6%、数学内容占0.3%。该模型支持8,192个token的上下文长度，是现有土耳其语BERT模型的十六倍，并继承了ModernBERT架构在长上下文处理上的高效性。

    arXiv:2512.23065v4 Announce Type: replace  Abstract: The introduction of BERT established encoder-only transformer models as a foundational paradigm in natural language processing. Encoder-only models remain the standard tool for classification, tagging and retrieval, where contextual representations and low inference cost matter more than text generation, yet Turkish lacks a monolingual encoder trained from scratch with the advances consolidated in ModernBERT (rotary positional embeddings, FlashAttention, refined normalization). We introduce TabiBERT, a monolingual Turkish encoder based on the ModernBERT architecture, pretrained from scratch for one trillion tokens sampled from an 86.58B-token multi-domain corpus of web text (72%), scientific publications (19%), source code (6%) and mathematical content (0.3%). The model supports a context length of 8,192 tokens, sixteen times that of existing Turkish BERT models, and inherits the ModernBERT architecture's efficiency at long context. 
    
[^212]: 通过轮次级重要性采样与截断触发归一化稳定长时程LLM智能体的离策略训练

    Stabilizing Off-Policy Training for Long-Horizon LLM Agent via Turn-Level Importance Sampling and Clipping-Triggered Normalization

    [https://arxiv.org/abs/2511.20718](https://arxiv.org/abs/2511.20718)

    该论文提出SORL方法，通过轮次级重要性采样与截断触发的归一化机制，使策略优化与多轮交互结构对齐并抑制不可靠的离策略梯度更新，从而稳定长时程LLM智能体的离策略强化学习训练，防止性能崩溃。

    

    诸如PPO和GRPO等强化学习（RL）算法被广泛用于训练大语言模型（LLM）以完成多轮智能体任务。然而，在离策略训练流程中，这些方法可能表现出不稳定的优化动态，并容易发生性能崩溃。通过实证分析，我们识别出该设置下两个根本性的不稳定性来源：（1）token级策略优化与轮次结构化交互之间的粒度不匹配；（2）由离策略重要性采样和不准确优势估计所引起的高方差、不可靠的梯度更新。为应对这些挑战，我们提出了SORL（Stabilizing Off-Policy Reinforcement Learning for Long-Horizon Agent Training，面向长时程智能体训练的稳定离策略强化学习方法）。SORL引入了使策略优化与多轮交互结构对齐的机制，并自适应地抑制不可靠的离策略更新，从而带来更加保守和稳健的优化过程。

    arXiv:2511.20718v3 Announce Type: replace-cross  Abstract: Reinforcement learning (RL) algorithms such as PPO and GRPO are widely used to train large language models (LLMs) for multi-turn agentic tasks. However, in off-policy training pipelines, these methods can exhibit unstable optimization dynamics and are prone to perfor- mance collapse. Through empirical analysis, we identify two fundamental sources of instability in this setting: (1) a granularity mismatch between token-level policy optimization and turn- structured interactions, and (2) high-variance and unreliable gradient updates induced by off- policy importance sampling and inaccurate advantage estimation. To address these challenges, we propose SORL, Stabilizing Off-Policy Reinforcement Learning for Long-Horizon Agent Train- ing. SORL introduces mechanisms that align policy optimization with the structure of multi- turn interactions and adaptively suppress unreliable off-policy updates, yielding more conserva- tive and robu
    
[^213]: 人工蜂群思维：语言模型的开放式同质化（及更广泛的领域）

    Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond)

    [https://arxiv.org/abs/2510.22954](https://arxiv.org/abs/2510.22954)

    该论文提出了包含26K条开放式查询的大规模数据集Infinity-Chat以及首个开放式提示综合分类体系，并据此对语言模型的模式坍缩进行大规模研究，揭示了语言模型在开放式生成中输出高度同质化的“人工蜂群思维”效应。

    

    语言模型（LM）往往难以生成多样化、类人的创造性内容，这引发了人们的担忧：通过反复接触相似的输出，人类思想可能被长期同质化。然而，用于评估语言模型输出多样性的可扩展方法仍然有限，尤其是在随机数或姓名生成等狭窄任务之外，或在从单一模型重复采样之外。我们提出了Infinity-Chat，一个包含26K条多样化、真实世界中开放式用户查询的大规模数据集，这些查询可以有广泛的合理答案，而没有单一的标准答案。我们引入了首个全面的分类体系，用于刻画向大语言模型提出的开放式提示的完整范围，包含6个顶级类别（例如头脑风暴与创意构思），并进一步细分为17个子类别。基于Infinity-Chat，我们对语言模型中的模式坍缩进行了大规模研究，揭示了开放式生成中显著的“人工蜂群思维”效应。

    arXiv:2510.22954v2 Announce Type: replace  Abstract: Language models (LMs) often struggle to generate diverse, human-like creative content, raising concerns about the long-term homogenization of human thought through repeated exposure to similar outputs. Yet scalable methods for evaluating LM output diversity remain limited, especially beyond narrow tasks such as random number or name generation, or beyond repeated sampling from a single model. We introduce Infinity-Chat, a large-scale dataset of 26K diverse, real-world, open-ended user queries that admit a wide range of plausible answers with no single ground truth. We introduce the first comprehensive taxonomy for characterizing the full spectrum of open-ended prompts posed to LMs, comprising 6 top-level categories (e.g., brainstorm & ideation) that further breaks down to 17 subcategories. Using Infinity-Chat, we present a large-scale study of mode collapse in LMs, revealing a pronounced Artificial Hivemind effect in open-ended gener
    
[^214]: VietBinoculars：一种检测越南语LLM生成文本的零样本方法

    VietBinoculars: A Zero-Shot Approach for Detecting Vietnamese LLM-Generated Text

    [https://arxiv.org/abs/2509.26189](https://arxiv.org/abs/2509.26189)

    VietBinoculars是一个针对越南语的零样本LLM生成文本检测框架，通过结合PhoGPT-4B观察者/执行者模型、校准的全局决策阈值和专门的越南语BPE分词，实现了超过0.99的AUC和至少98.78%的检测准确率，显著优于现有检测工具。

    

    大语言模型的快速普及加剧了在非英语语言中区分LLM生成文本与人类写作的挑战。本研究提出了VietBinoculars，这是一个零样本检测框架，它将PhoGPT-4B观察者模型与执行者模型和经过校准的全局决策阈值相结合。通过利用专门的越南语BPE分词，该方法消除了大型多语言骨干网络中常见的字节级碎片化和概率稀释问题。在多领域基准测试中，VietBinoculars实现了超过0.99的ROC曲线下面积（AUC）。在最优Youden's J阈值和贪心解码下，检测准确率达到至少98.78%，在创意Capybara提示上显著优于基线Binoculars、零样本检测器和商业工具。即使在0.06%的严格假阳性率约束下，该检测器仍能保持83.15%至9……之间的F1分数。

    arXiv:2509.26189v2 Announce Type: replace  Abstract: The rapid proliferation of Large Language Models has intensified the challenge of distinguishing LLM-generated text from human writing in non-English languages. This study introduces VietBinoculars, a zero-shot detection framework coupling PhoGPT-4B observer and performer models with calibrated global decision thresholds. By utilizing specialized Vietnamese BPE tokenization, the method eliminates byte-level fragmentation and probability dilution common in massive multilingual backbones. Evaluated across multi-domain benchmarks, VietBinoculars achieves an area under the ROC curve exceeding 0.99. Under optimal Youden's J thresholds and greedy decoding, detection accuracy reaches at least 98.78\%, while significantly outperforming baseline Binoculars, zero-shot detectors, and commercial tools on creative Capybara prompts. Even under a strict false positive rate constraint of 0.06\%, the detector maintains F1-scores between 83.15\% and 9
    
[^215]: PiERN：面向高精度计算与推理融合的令牌级路由

    PiERN: Token-Level Routing for Integrating High-Precision Computation and Reasoning

    [https://arxiv.org/abs/2509.18169](https://arxiv.org/abs/2509.18169)

    PiERN提出了一种物理隔离专家路由网络架构，通过在令牌级别路由计算与推理，使大型语言模型能够在单一思维链中迭代交替地执行高精度数值计算与推理，在准确率、响应延迟、令牌消耗和能耗等方面均优于直接微调与主流多智能体方法。

    

    复杂系统上的任务需要高精度数值计算来支持决策。然而，当前的大型语言模型（LLM）即使具备增强的推理能力，在现有架构下也无法将此类计算作为一种内在且可解释的能力加以整合。为此，我们提出了物理隔离专家路由网络，这是一种在令牌级别引导计算与推理的架构，从而能够在单一思维链内实现迭代交替。我们在代表性的计算-推理任务上对PiERN进行了系统评估，包括PDEBench和电池管理任务。结果表明，PiERN不仅比直接微调LLM取得更高的准确率，而且与主流多智能体方法相比，在响应延迟、令牌使用量、GPU能耗和专家路由准确率方面均有显著改进，同时没有出现明显的性能下降。

    arXiv:2509.18169v4 Announce Type: replace-cross  Abstract: Tasks on complex systems require high-precision numerical computation to support decisions. However, current large language models (LLMs), even with enhanced reasoning capabilities, cannot integrate such computations as an intrinsic and interpretable capability with existing architectures. To this end, we propose Physically-isolated Experts Routing Network (PiERN), an architecture that directs computation and reasoning at token level, thereby enabling iterative alternation within a single chain of thought. We systematically evaluate PiERN on representative computation-reasoning tasks, including PDEBench and battery management tasks. Results show that PiERN achieves not only higher accuracy than directly finetuning LLMs but also significant improvements in response latency, token usage, GPU energy consumption, and experts routing accuracy compared with mainstream multi-agent approaches, while exhibiting no significant degradatio
    
[^216]: Nord-Parl-TTS：来自议会演讲的芬兰语和瑞典语TTS数据集

    Nord-Parl-TTS: Finnish and Swedish TTS Dataset from Parliament Speech

    [https://arxiv.org/abs/2509.17988](https://arxiv.org/abs/2509.17988)

    该论文发布了基于北欧议会演讲录音构建的开源TTS数据集Nord-Parl-TTS，包含900小时芬兰语和5090小时瑞典语语音，有效缩小了TTS领域高资源语言与低资源语言之间的数据资源差距。

    

    文本转语音（TTS）的发展受到限制，因为除少数高资源语言外，大多数语言都缺乏高质量、公开可用的语音数据。我们提出了Nord-Parl-TTS，这是一个基于真实场景语音的芬兰语和瑞典语开源TTS数据集。利用北欧议会会议的录音，我们提取了900小时的芬兰语和5090小时的瑞典语语音，适合用于TTS训练。该数据集使用改编版的Emilia数据处理流水线构建，并包含统一的评估集，以支持模型开发和基准测试。通过为芬兰语和瑞典语提供开源的大规模数据，Nord-Parl-TTS缩小了高资源语言与低资源语言之间在TTS领域的资源差距。

    arXiv:2509.17988v2 Announce Type: cross  Abstract: Text-to-speech (TTS) development is limited by scarcity of high-quality, publicly available speech data for most languages outside a few high-resource languages. We present Nord-Parl-TTS, an open TTS dataset for Finnish and Swedish based on speech found in the wild. Using recordings of Nordic parliamentary proceedings, we extract 900 hours of Finnish and 5090 hours of Swedish speech suitable for TTS training. The dataset is built using an adapted version of the Emilia data processing pipeline and includes unified evaluation sets to support model development and benchmarking. By offering open, large-scale data for Finnish and Swedish, Nord-Parl-TTS narrows the resource gap in TTS between high- and lower-resourced languages.
    
[^217]: 打破镜像：基于激活的LLM评估器自我偏好缓解方法

    Breaking the Mirror: Activation-Based Mitigation of Self-Preference in LLM Evaluators

    [https://arxiv.org/abs/2509.03647](https://arxiv.org/abs/2509.03647)

    本文提出利用通过对比激活添加（CAA）和优化方法构建的转向向量，在无需重新训练的情况下于推理时缓解LLM评估器的不合理自我偏好偏差，最多可降低97%，显著优于提示和直接偏好优化基线。

    

    大型语言模型（LLM）日益被用作自动评估器，但它们存在“自我偏好偏差”：即倾向于偏好自己的输出而胜过其他模型的输出。这种偏差破坏了评估流程的公平性和可靠性，尤其是在偏好调优和模型路由等任务中。我们研究了轻量级的转向向量能否在推理阶段缓解这一问题而无需重新训练。我们引入了一个精心构建的数据集，将自我偏好偏差区分为合理的自我偏好示例和不合理的自我偏好示例，并采用两种方法构建转向向量：对比激活添加（CAA）和基于优化的方法。我们的结果表明，转向向量可以将不合理的自我偏好偏差降低高达97%，显著优于提示和直接偏好优化基线。然而，转向向量在某些情况下表现不稳定……

    arXiv:2509.03647v3 Announce Type: replace-cross  Abstract: Large language models (LLMs) increasingly serve as automated evaluators, yet they suffer from "self-preference bias": a tendency to favor their own outputs over those of other models. This bias undermines fairness and reliability in evaluation pipelines, particularly for tasks like preference tuning and model routing. We investigate whether lightweight steering vectors can mitigate this problem at inference time without retraining. We introduce a curated dataset that distinguishes self-preference bias into justified examples of self-preference and unjustified examples of self-preference, and we construct steering vectors using two methods: Contrastive Activation Addition (CAA) and an optimization-based approach. Our results show that steering vectors can reduce unjustified self-preference bias by up to 97\%, substantially outperforming prompting and direct preference optimization baselines. Yet steering vectors are unstable on 
    
[^218]: FedCoT：面向大语言模型的通信高效联邦推理增强

    FedCoT: Communication-Efficient Federated Reasoning Enhancement for Large Language Models

    [https://arxiv.org/abs/2508.10020](https://arxiv.org/abs/2508.10020)

    FedCoT是一个通信高效的联邦推理增强框架，通过轻量级思维链重采样、紧凑判别器筛选以及客户端感知的LoRA堆叠加权聚合，在无需集中式蒸馏和保护隐私的前提下增强大语言模型的逐步推理能力，特别适用于医疗等需要可解释、可审计决策的场景。

    

    在联邦环境下增强大语言模型的推理能力并非易事，因为需要应对严格的计算、通信和隐私约束，尤其是在医疗保健领域，临床上具有重大影响的决策不仅需要准确性，还需要可解释、可审计的推理依据，以满足安全性、问责性和监管要求。传统的联邦微调主要模仿最终答案，而非培养逐步推理能力，且通常依赖隐私敏感的集中式蒸馏，同时仍会产生大量通信开销。我们提出FedCoT来解决这一问题，这是一个联邦推理框架，它将轻量级的思维链重采样与紧凑的判别器相结合用于候选筛选，并采用客户端感知的LoRA堆叠与加权分类器聚合机制，以适应客户端异构性，同时降低聚合噪声和通信成本；客户端在本地生成候选思维链和监督信号，……

    arXiv:2508.10020v2 Announce Type: replace-cross  Abstract: Enhancing LLM reasoning in federated settings is nontrivial due to stringent computational, communication, and privacy constraints, especially in healthcare, where clinically consequential decisions require not only accuracy but also interpretable, auditable rationales to meet safety, accountability, and regulatory requirements. Conventional federated fine-tuning largely imitates final answers rather than cultivating step-by-step reasoning, often relying on privacy-sensitive centralized distillation and still incurring substantial communication overhead. We address this gap with \textbf{\ours{}}, a federated reasoning framework that combines lightweight chain-of-thought resampling with a compact discriminator for selection, and client-aware LoRA stacking with weighted classifier aggregation to accommodate heterogeneity while reducing aggregation noise and communication; clients generate candidate chains and supervision locally,
    
[^219]: 过于范畴化而不似人类：大语言模型与人类中的情绪概念

    Too Categorical to be Human: Emotion Concepts in LLMs and Humans

    [https://arxiv.org/abs/2508.05880](https://arxiv.org/abs/2508.05880)

    该论文提出基于认知评估理论、以“行为表征”刻画情绪概念，并构建涵盖15个情绪类别的基准数据集来比较LLMs与人类的情绪概念表征，发现LLMs的情绪概念过于范畴化而与人类不同。

    

    理解人类情绪对于面向用户的AI应用、安全对齐以及人类行为模拟而言至关重要。由于情绪刺激会塑造大语言模型（LLMs）在高风险情境下的行为，人们越来越关注模型如何在内部表征情绪概念。然而，对这些表征的机制性解释无法直接与人类进行对比：人类的情绪加工是高度分布式的，无法产生等效的神经表征。为了探究LLMs是否以与人类相似的方式内化情绪概念，我们提出利用外部行为特征来刻画情绪这一抽象概念，并将其称为“行为表征”。借助认知评估理论——该理论使得我们能够沿着可解释的评价维度来表征情绪情境——我们构建了一个涵盖15个情绪类别的情绪情境基准数据集。我们引出……（原文摘要在此处截断）

    arXiv:2508.05880v3 Announce Type: replace-cross  Abstract: Understanding human emotions is central to user-facing AI applications, safety alignment, and the simulation of human behavior. As emotional stimuli shape high-stakes behavior in Large Language Models (LLMs), there is increasing interest in how models represent emotion concepts internally. Mechanistic accounts of these representations, however, cannot be compared directly against humans: emotion processing in humans is highly distributed and yields no equivalent neural representation. To understand whether LLMs internalize emotion concepts in a way similar to humans, we propose characterizing the abstract concept of an emotion using external behavioral signatures, which we term behavioral representations. Using the theory of cognitive appraisals, which enables representing emotional situations along interpretable evaluative dimensions, we create a benchmark dataset of emotional scenarios spanning 15 emotion categories. We elici
    
[^220]: 使用语音后验图进行芬兰语语音发音编辑

    Pronunciation Editing for Finnish Speech using Phonetic Posteriorgrams

    [https://arxiv.org/abs/2507.02115](https://arxiv.org/abs/2507.02115)

    提出了基于扩散模型的多说话人PPG2Speech模型，通过增强Matcha-TTS流匹配解码器，无需文本对齐即可编辑单个音素，从而将母语语音转换为近似的芬兰语第二语言语音。

    

    合成第二语言（L2）语音对于L2语言学习体验和反馈具有潜在的高价值。然而，由于缺乏L2语音合成数据集，对于低资源语言来说合成L2语音十分困难。在本文中，我们提供了一种将母语语音编辑为近似L2语音的实用解决方案，并提出了PPG2Speech——一个基于扩散模型的多说话人语音后验图到语音模型，能够在无需文本对齐的情况下编辑单个音素。我们使用Matcha-TTS的流匹配解码器作为骨干网络，将语音后验图（PPGs）转换为梅尔频谱图，并以外部说话人嵌入和音高作为条件。PPG2Speech通过无分类器引导和Sway采样增强了Matcha-TTS的流匹配解码器。我们还提出了一种新的面向任务的客观评估指标——语音对齐一致性（PAC），用于衡量编辑后的PPGs与额外PPGs之间的一致性……

    arXiv:2507.02115v3 Announce Type: cross  Abstract: Synthesizing second-language (L2) speech is potentially highly valued for L2 language learning experience and feedback. However, due to the lack of L2 speech synthesis datasets, it is difficult to synthesize L2 speech for low-resourced languages. In this paper, we provide a practical solution for editing native speech to approximate L2 speech and present PPG2Speech, a diffusion-based multispeaker Phonetic-Posteriorgrams-to-Speech model that is capable of editing a single phoneme without text alignment. We use Matcha-TTS's flow-matching decoder as the backbone, transforming Phonetic Posteriorgrams (PPGs) to mel-spectrograms conditioned on external speaker embeddings and pitch. PPG2Speech strengthens the Matcha-TTS's flow-matching decoder with Classifier-free Guidance (CFG) and Sway Sampling. We also propose a new task-specific objective evaluation metric, the Phonetic Aligned Consistency (PAC), between the edited PPGs and the PPGs extra
    
[^221]: Common Corpus：用于大语言模型预训练的最大规模合乎伦理数据集

    Common Corpus: The Largest Collection of Ethical Data for LLM Pre-Training

    [https://arxiv.org/abs/2506.01732](https://arxiv.org/abs/2506.01732)

    本文发布了Common Corpus——目前最大的用于大语言模型预训练的开放数据集，包含约两万亿词元的无版权或开放许可数据，涵盖从高资源到低资源的多种语言及大量代码数据，为合乎伦理且合规的大模型训练奠定基础。

    

    大语言模型（LLM）在海量来自不同来源和领域的数据上进行预训练。这类数据集通常包含数万亿个词元（token），其中很大一部分是受版权保护或专有的内容，这引发了关于此类模型合法使用的问题。这凸显了对符合数据安全法规的真正开放预训练数据的需求。在本文中，我们介绍了Common Corpus，这是目前用于LLM预训练的最大开放数据集。Common Corpus中汇集的数据要么不受版权保护，要么采用开放许可协议，总计约两万亿个词元。该数据集包含丰富多样的语言，从高资源的欧洲语言到一些在预训练数据集中很少出现的低资源语言。此外，它还包含大量代码数据。数据来源在覆盖领域和时间跨度方面的多样性，为研究和创业开辟了道路。

    arXiv:2506.01732v4 Announce Type: replace  Abstract: Large Language Models (LLMs) are pre-trained on large amounts of data from different sources and domains. Such datasets often contain trillions of tokens, including large portions of copyrighted or proprietary content, which raises questions about the legal use of such models. This underscores the need for truly open pre-training data that complies with data security regulations. In this paper, we introduce Common Corpus, the largest open dataset for LLM pre-training. The data assembled in Common Corpus are either uncopyrighted or under open licenses, totaling about two trillion tokens. The dataset contains a wide variety of languages, ranging from the high-resource European languages to some low-resource languages rarely represented in pre-training datasets. In addition, it includes a large amount of code data. The diversity of data sources in terms of covered domains and time periods opens up the paths for both research and entrepr
    
[^222]: 即使是小型推理模型也应引用其来源：Pleias-RAG 模型系列简介

    Even Small Reasoners Should Quote Their Sources: Introducing the Pleias-RAG Model Family

    [https://arxiv.org/abs/2504.18225](https://arxiv.org/abs/2504.18225)

    Pleias-RAG-350m和Pleias-RAG-1B是专为RAG设计的小型推理模型，原生支持引文溯源和查询处理功能，在标准基准上超越同级别模型并可与更大模型媲美，且在多语言环境中保持一致的高性能表现。

    

    我们推出了一代面向RAG（检索增强生成）、搜索和来源摘要的小型推理模型。Pleias-RAG-350m和Pleias-RAG-1B在一个大型合成数据集上进行了中期训练，该数据集模拟了从Common Corpus中检索各种多语言开放来源的过程。它们原生支持通过字面引文进行引用和事实依据溯源，并重新整合了与RAG工作流程相关的多项功能，例如查询路由、查询重写和来源重排序。Pleias-RAG-350m和Pleias-RAG-1B在标准化RAG基准测试（HotPotQA、2wiki）上超越了40亿参数以下的小型语言模型（SLM），并可与流行的大型模型相媲美，包括Qwen-2.5-7B、Llama-3.1-8B和Gemma-3-4B。它们是迄今为止唯一能在主要欧洲语言中保持一致RAG性能、并确保语句具备系统性参考依据的小型语言模型。

    arXiv:2504.18225v2 Announce Type: replace  Abstract: We introduce a new generation of small reasoning models for RAG, search, and source summarization. Pleias-RAG-350m and Pleias-RAG-1B are mid-trained on a large synthetic dataset emulating the retrieval of a wide variety of multilingual open sources from the Common Corpus. They provide native support for citation and grounding with literal quotes and reintegrate multiple features associated with RAG workflows, such as query routing, query reformulation, and source reranking. Pleias-RAG-350m and Pleias-RAG-1B outperform SLMs below 4 billion parameters on standardized RAG benchmarks (HotPotQA, 2wiki) and are competitive with popular larger models, including Qwen-2.5-7B, Llama-3.1-8B, and Gemma-3-4B. They are the only SLMs to date maintaining consistent RAG performance across leading European languages and ensuring systematic reference grounding for statements. Due to their size and ease of deployment on constrained infrastructure and hi
    
[^223]: 掩码微调助力大语言模型性能提升

    Boosting Large Language Models with Mask Fine-Tuning

    [https://arxiv.org/abs/2503.22764](https://arxiv.org/abs/2503.22764)

    提出掩码微调（MFT）这一新颖的大语言模型微调范式，通过学习并应用二值掩码、在不更新模型权重的情况下精心打破模型结构完整性，从而在不同领域和骨干网络上获得一致的性能提升。

    

    大语言模型（LLM）通常被整合进主流的优化流程之中。然而，保持模型的完整性对于获得良好性能是否是不可或缺的，这一问题仍未被充分探索。在本工作中，我们提出了掩码微调，这是一种新颖的大语言模型微调范式，其表明精心地打破模型的结构完整性，可以在不更新模型权重的情况下出人意料地提升性能。MFT以标准的LLM微调目标作为监督，学习并应用二值掩码到已经过良好优化的模型上。基于完全微调后的模型，MFT使用相同的微调数据集，在不同领域和不同骨干网络上实现了一致的性能提升（例如，LLaMA2-7B/3.1-8B在IFEval上平均提升2.70/4.15分）。详细的消融实验和分析从多个角度对所提出的MFT进行了考察，包括稀疏比率和损失面等。此外，……

    arXiv:2503.22764v3 Announce Type: replace-cross  Abstract: The large language model (LLM) is typically integrated into the mainstream optimization protocol. However, it remains underexplored whether maintaining the model integrity is \textit{indispensable} for promising performance. In this work, we introduce Mask Fine-Tuning (MFT), a novel LLM fine-tuning paradigm demonstrating that carefully breaking the model's structural integrity can surprisingly improve performance without updating model weights. MFT learns and applies binary masks to well-optimized models, using the standard LLM fine-tuning objective as supervision. Based on fully fine-tuned models, MFT uses the same fine-tuning datasets to achieve consistent performance gains across domains and backbones (e.g., an average gain of 2.70/4.15 on IFEval with LLaMA2-7B/3.1-8B). Detailed ablation studies and analyses examine the proposed MFT from different perspectives, including the sparse ratio and the loss surface. Additionally, w
    
[^224]: 增强基于大语言模型的推荐模型中的高阶交互感知能力

    Enhancing High-order Interaction Awareness in LLM-based Recommender Model

    [https://arxiv.org/abs/2409.19979](https://arxiv.org/abs/2409.19979)

    本文提出增强型LLM推荐器ELMRec，通过增强全词嵌入使大语言模型无需图预训练即可理解用户-项目高阶交互，并针对LLM偏好早期交互而忽略近期交互的问题提出重排序方案，在直接推荐和序列推荐中均超越现有最先进方法。

    

    大语言模型（LLM）通过将推荐任务转化为文本生成任务，已在推荐任务中展现出卓越的推理能力。然而，现有方法要么忽略了用户-项目的高阶交互，要么对其建模效果不佳。为此，本文提出了一种增强型的大语言模型推荐器（ELMRec）。我们增强了全词嵌入，从而大幅提升大语言模型对图构建的交互信息的理解能力，且无需图预训练。这一发现可能启发人们通过全词嵌入将丰富的知识图谱融入基于大语言模型的推荐器中。我们还发现，大语言模型在推荐项目时往往基于用户的早期交互而非近期交互，并据此提出了一种重排序解决方案。我们的ELMRec在直接推荐和序列推荐中均优于最先进（SOTA）的方法。

    arXiv:2409.19979v4 Announce Type: replace-cross  Abstract: Large language models (LLMs) have demonstrated prominent reasoning capabilities in recommendation tasks by transforming them into text-generation tasks. However, existing approaches either disregard or ineffectively model the user-item high-order interactions. To this end, this paper presents an enhanced LLM-based recommender (ELMRec). We enhance whole-word embeddings to substantially enhance LLMs' interpretation of graph-constructed interactions for recommendations, without requiring graph pre-training. This finding may inspire endeavors to incorporate rich knowledge graphs into LLM-based recommenders via whole-word embedding. We also found that LLMs often recommend items based on users' earlier interactions rather than recent ones, and present a reranking solution. Our ELMRec outperforms state-of-the-art (SOTA) methods in both direct and sequential recommendations.
    

