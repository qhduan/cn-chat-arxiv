# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Language of Instruction: Genre and Generative AI](https://arxiv.org/abs/2610.32982) | 本文创新性地提出，指令微调后的语言模型实现了一种新的“指令-会话体裁”，该体裁融合了人机历史对话的话语语言，并通过指令微调、系统提示词以及带有特殊标记和角色赋值的标记语言输入在应用中实例化。 |
| [^2] | [Measure Before You Train: Typed Decision Models vs. a Classical Baseline vs. a Generative LLM for Routing a Personal Multilingual Inbox](https://arxiv.org/abs/2610.32955) | 该论文提出自托管个人邮件代理Foyer及其测量协议，在真实多语言邮件数据上于训练前建立人工真实标注基准，并系统比较了类型化决策模型、经典机器学习基线与生成式大语言模型等九个系统的表现。 |
| [^3] | [Yalda-Embedding: Open Development of a Competitive Persian Retrieval Model](https://arxiv.org/abs/2610.32933) | 该论文提出Yalda Embedding，一个基于Qwen3-Embedding-0.6B适配、使用超百万查询-文档对训练的5.958亿参数波斯语检索模型，在FaMTEB的八个波斯语检索任务上超越所有同规模0.6B模型，并接近4B规模模型的性能。 |
| [^4] | [Adaptive Frequency Threshold Byte Pair Encoding for Mixed Natural Language and Structured Symbolic Corpora](https://arxiv.org/abs/2610.32921) | 本文提出AdaBPE，一种基于自适应频率阈值的频繁模式挖掘分词方法，通过动态调整合并策略，更好地处理跨越空白的多词表达和LaTeX等结构化符号内容。 |
| [^5] | [Do "True" and "False" Add Up to One? Complement Consistency in LLM-Agent Opinion Simulations](https://arxiv.org/abs/2610.32874) | 该研究首次系统揭示了LLM智能体在观点模拟中违反互补一致性——对同一命题及其否定所报告的概率之和超过1（如Qwen3-4B平均达1.18），且该偏差远大于扰动噪声两到十倍，意味着此类模拟结果可能存在系统性扭曲。 |
| [^6] | [J-Comm: Jacobian-Guided Latent Communication between LLM Agents](https://arxiv.org/abs/2610.32871) | 提出J-Comm框架，结合免训练的GEAR几何一致性方法与H+J（隐状态与雅可比）双通道通信接口，实现大语言模型智能体之间基于雅可比引导的潜在通信。 |
| [^7] | [Interpretable sparse fine-tuning with TopKLoRA enables faithful circuit discovery](https://arxiv.org/abs/2610.32866) | 本文提出了设计上即可解释的稀疏微调方法TopKLoRA，通过在每个token上仅激活k个最大的适配器潜在变量、纯粹以下游损失训练，使其潜在变量在功能上具有因果性，从而能够忠实地发现休眠代理后门等模型内部电路。 |
| [^8] | [Communicating with the Entire Population: Machine Translation of Emergency Alerts and Warnings](https://arxiv.org/abs/2610.32842) | 本文通过全国性应急管理者调查了解其真实需求，并据此评估商业机器翻译工具在紧急警报翻译中的表现，为应急响应中如何及何时安全使用翻译工具提供了实用指导。 |
| [^9] | [SPOOK: Sparse Pixel Optimization for Off-Manifold Knowledge in Vision Encoders](https://arxiv.org/abs/2610.32835) | 该研究提出SPOOK方法，证明仅通过黑盒爬山优化对空白画布进行稀疏像素编辑，就能在OpenCLIP和DINOv3等视觉编码器的表示空间中成功检索到目标语义，揭示了远离自然图像流形处嵌入相似性与人类可见语义的脱节。 |
| [^10] | [KrishokTech: A Bengali Agricultural Advisory System with Explicit Authorization Controls](https://arxiv.org/abs/2610.32817) | KRISHOKTECH 是一个孟加拉语农业咨询系统，其核心创新在于围绕生成过程构建显式授权边界——通过高风险请求拦截、作物确认机制和剂量声明验证等控制措施，并结合可独立更新的外部证据索引与本地检索，在低网络连接环境下安全可靠地提供农业建议。 |
| [^11] | [Frontier-OPD: Discovering the Edge of Agent Capability and Teaching Agents to Go Beyond](https://arxiv.org/abs/2610.32802) | Frontier-OPD 通过“学习边界”概念——即经简短教师指导后学生仍能自主有效延续的挑战性决策情境——并借助共享状态轨迹图和师生分歧的轨迹平衡聚合来定位干预点，从而发现多轮智能体的能力边界并教会其超越。 |
| [^12] | [Multiplicative vs. Additive State Coupling in Learned Optimizer Schedules](https://arxiv.org/abs/2610.32801) | 本研究通过反向传播学习MAX-CUT求解器的调度，发现双线性（乘法）状态耦合架构在分布外泛化上相比加法耦合具有显著且持久的优势，从而将关键架构区别定位于求解器状态进入调度的方式，而非状态自适应性本身。 |
| [^13] | [ParaTaxe: a Suite of Tools for Processing and Publishing Multi-parallel Corpora](https://arxiv.org/abs/2610.32796) | 该论文提出了一套开源工具套件（包括基于多语言嵌入的AIlign对齐器、Python处理流程以及用于管理和浏览平行语料库的网页界面），旨在解决学术平行语料库因缺乏维护和资金而难以在线持续访问的问题。 |
| [^14] | [Every Edge Needs a Quote: Building a Quote-Grounded Knowledge Graph of Healthcare Software Adoption with Language-Model Agents](https://arxiv.org/abs/2610.32781) | 该论文提出一个从开放网络构建医疗软件客户与集成关系知识图谱的流水线，要求每条边都必须有源页面的逐字引用支持，从而确保语言模型智能体提取的信息可验证、可追溯。 |
| [^15] | [On-Policy Self-Training: Data-Free Self-Improvement in Large Language Models](https://arxiv.org/abs/2610.32770) | 该论文提出了一种无需任何数据的在线策略自训练框架，大语言模型仅需粗糙的高层主题描述即可自生成问题-答案对和特权推理演示进行自我改进，在多个领域提升能力的同时保留原有领域的性能。 |
| [^16] | [ActionRAG: Evidence-Refining Graph Diffusion for Long-Term Memory in LLM Agents](https://arxiv.org/abs/2610.32752) | ActionRAG提出了一种图记忆架构，通过谨慎的分层实体消解索引与查询时基于局部路径结构的图扩散回溯评估，帮助LLM智能体从长期交互历史中恢复精确且相互一致的关系型证据。 |
| [^17] | [The Butterfly Defects: Tracing Cooperation in LLM Agent Networks](https://arxiv.org/abs/2610.32726) | 该论文通过迭代囚徒困境实验发现，大语言模型智能体网络中的合作行为高度依赖模型组成与微小的语言变化（如不确定性提示、动作标签顺序），从而挑战了合作是稳定模型级属性的观念。 |
| [^18] | [Ontology-Guided Knowledge Graph Construction from Structured Metadata and Climate Science Publications](https://arxiv.org/abs/2610.32725) | 本文提出一种代理式、本体引导的知识图谱构建框架，利用大语言模型从气候科学文献中抽取带文本证据的三元组，并通过多阶段验证确保语义一致性与可追溯性，抽取质量接近人类标注者水平。 |
| [^19] | [CounselPlex: A Stressor-Aware Full-Duplex Counseling Model](https://arxiv.org/abs/2610.32702) | CounselPlex通过在求助者沉默时更新压力源状态并在聆听时注入该状态，克服了全双工语音模型的“压力源盲区”问题，同时引入RoPE位置偏移增强（RPSA）以提升长对话深度下的性能，从而实现更高质量的心理咨询。 |
| [^20] | [Natural Language Autoencoders Identify Without Explaining](https://arxiv.org/abs/2610.32691) | 本文指出自然语言自编码器（NLA）的往返重建成功标准只能证明解释性文本包含足以识别激活的信息，并不能保证该解释对激活实际功能的描述是忠实的，这一问题与思维链忠实性问题结构相同。 |
| [^21] | [When is more search useful? An Empirical Study of Search Utility in Quality-Aware Decoding for Machine Translation](https://arxiv.org/abs/2610.32679) | 该论文提出“搜索效用”的三个量化指标（总扩展增益、边际扩展增益、采样饱和点），实证表明机器翻译中不同句子从额外候选中获益的程度差异显著，从而为自适应分配搜索预算提供了依据。 |
| [^22] | [Residual Expert Routing for Text-Only Domain Adaptation of Factorized Transducer ASR](https://arxiv.org/abs/2610.32637) | 提出保守残差路由方法，利用轻量门控在每个前缀处动态决定是否接纳文本自适应预测器的残差修正，从而在保留目标领域纯文本自适应收益的同时，使源领域性能完全不受损害。 |
| [^23] | [PHYRD: A Living Benchmark for Reproducing and Extending Theoretical Physics Research](https://arxiv.org/abs/2610.32575) | 本文提出PHYRD动态基准（含63个复现题与43个发现题）及FormulaForest任务构建框架，用于评估大语言模型的理论物理推理能力，并证明FormulaForest作为合成数据可将模型在FrontierScience基准上的准确率从61%提升至72%。 |
| [^24] | [Pseudo-Labels as Rewards, Not Targets: Group-Relative Policy Optimization for Semi-Supervised ASR](https://arxiv.org/abs/2610.32572) | 该论文提出将伪标签用作GRPO的奖励参考而非监督微调的目标，使教师错误传递率从100%降至27%，并在四个模型规模、四个教师家族、三个领域和四种语言上取得29.3%-56.2%的相对WER改进，显著提升了半监督ASR对标签噪声的鲁棒性。 |
| [^25] | [Principled Nonlinear Concept Erasure and Steering](https://arxiv.org/abs/2610.32516) | 本文提出一个基于条件自编码器的原则性非线性概念擦除框架，从理论上证明完美重建与总体水平独立性足以实现完美的非线性防护保证，并在多个数据集上取得了更优的实用性与擦除效果的平衡。 |
| [^26] | [Analysing Austrian Parliamentary Speeches by Predicting Party Affiliations](https://arxiv.org/abs/2610.32515) | 该论文利用微调BERT模型预测奥地利议会演讲的党派归属，发现联合执政党派的演讲特征趋于相似而反对党逐渐疏远，且议员只有在换党后才会改变演讲风格。 |
| [^27] | [The Punjabi Geet Toxicity Dataset (PGT-FG): Fine-Grained Multimodal Annotations for Toxic Content in Punjabi Songs](https://arxiv.org/abs/2610.32513) | 本文提出了首个面向旁遮普歌曲毒性检测的段级多模态数据集PGT-FG，包含2,396个经四名母语标注者标注的片段，涵盖八类细粒度毒性类别及严重程度、情感等多维度标注，填补了旁遮普语音乐内容毒性检测资源的空白。 |
| [^28] | [Accordion: Elastic-Depth Inference via Fixed-Prefix Layer Looping](https://arxiv.org/abs/2610.32492) | 本文提出基于混合专家的模型Accordion，通过固定前缀层循环支持五种推理预算，并在预训练、监督微调和强化学习全流程中无需跨预算一致性目标即可在不同推理预算下保持强劲性能。 |
| [^29] | [When Agents Fail: Model or Harness--Who's to Blame?](https://arxiv.org/abs/2610.32483) | 本文提出了一个通过五种互补指标（有效完成率、模型-框架匹配度、跨模型匹配度、交互残差和Token负载）来评估LLM智能体执行框架与底层模型交互表现的方法，用以区分智能体失败时究竟应归咎于模型还是执行框架。 |
| [^30] | [Beyond Post-Hoc Rewards: Privileged Reasoning Exploration for Multimodal Embeddings](https://arxiv.org/abs/2610.32474) | 提出PILOT框架，将自生成推理建模为潜在变量，通过共享多模态大语言模型的先验与后验两种推理模式进行特权推理探索，从而发现更具检索效用的推理轨迹，突破传统仅以输入为条件的事后奖励优化的局限，提升多模态嵌入的质量。 |
| [^31] | [Reading Between the Layers: Cross-Modal Alignment and Representational Hierarchy in Vision Transformers](https://arxiv.org/abs/2610.32463) | 该研究通过逐层探测与跨模态对齐分析，首次系统揭示视觉Transformer中强大的跨模态语义映射仅在最深层出现，并呈现出从表层识别到语义理解的结构化表征层级。 |
| [^32] | [Decoupling Patient Personas and Clinical Workflows: A Dual-Agent Approach to Realistic Medical Dialogue Generation](https://arxiv.org/abs/2610.32443) | 提出CoRPS-Dial双智能体框架，通过解耦患者画像与临床工作流并辅以事后质量控制，生成兼顾临床规范性与患者多样性的逼真诊前医疗对话。 |
| [^33] | [SCI-AGENT-EXAM: Benchmarking Agents on Scientific Research Tasks](https://arxiv.org/abs/2610.32441) | 提出了SCI-AGENT-EXAM基准测试，包含120个覆盖11个科学领域和7种任务类型的专家设计任务，并采用程序化验证与LLM评判相结合的混合评估协议，以系统评估科研智能体在真实研究工作流程中的综合能力。 |
| [^34] | [Profile-Driven User Simulation via Reinforcement Learning with Hierarchical Rewards for Task-Oriented Dialogues](https://arxiv.org/abs/2610.32435) | 该论文提出ProfileSim框架，通过联合建模用户画像（人口统计与大五人格）和业务画像（从真实通话记录逆向提取的意图），并利用在真实语音数据上微调及带层级奖励的强化学习优化，实现了面向金融营销外呼场景、可细粒度控制的用户模拟。 |
| [^35] | [Cycle-of-Thought: Self-Evaluative Iterative Reasoning for Table Visual Question Answering in Document Images](https://arxiv.org/abs/2610.32423) | 提出了一种无需训练的推理时自我纠正框架，让单个视觉语言模型迭代地扮演生成者、批评者和改进者的角色，形成闭环自我反馈回路，无需额外标注数据或模型更新即可提升文档图像中表格视觉问答的性能。 |
| [^36] | [Can LLM Annotators Match Human Consistency? Benchmarking Human Agreement and Optimizing Multi-Agent with Contrastive Knowledge](https://arxiv.org/abs/2610.32415) | 该论文提出了真实世界多标注者基准ANT-A（使用Fleiss kappa进行分布感知评估，衡量LLM标注与人类标注分布的一致性），并配套提出通过蒸馏对比知识优化提示词的多智能体标注框架ANT-OR，以提升标注系统对复杂标注指南的适应能力。 |
| [^37] | [A Broader Look into MLLM's Retrieval Heads](https://arxiv.org/abs/2610.32411) | 该研究通过构建语音与画面内容相同的视听配对输入发现，检索头机制随新增模态依然保持稀疏并在跨模态间高度共享，且全模态模型在很大程度上保留了原有模态的检索头。 |
| [^38] | [Acoustic and Lexical Signals for Native Language Identification of Language Learners](https://arxiv.org/abs/2610.32382) | 该研究首次将基于WavLM的音频分类器与零样本GPT-4o转写文本分类器相结合用于第二语言学习者的母语识别，发现音频方法在低熟练度（A1）时更优，而文本方法在中高熟练度（A2–C1）时更具优势。 |
| [^39] | [Why Full-Duplex Speech Models Collapse in Long Conversations](https://arxiv.org/abs/2610.32349) | 本文提出全双工长时程基准FD-LHB，通过长时重放对话和与LLM实时对话的测试，首次系统揭示了现有全双工语音模型在长对话中均会退化或耗尽上下文，陷入重复、沉默和胡言乱语，并识别了其背后的成因机制。 |
| [^40] | [Beyond Final Outcomes: Quantifying Claim-Level Information Dynamics in Multi-Agent LLM Interaction](https://arxiv.org/abs/2610.32333) | 本文提出RIVER框架，将多智能体LLM交互输出分解为原子声明，从收敛、探索和保留三个维度量化声明级信息动态，发现重复聚合式交互促进收敛但缩小信息空间，而分布式接力式交互能产生更广泛的信息。 |
| [^41] | [4DMemory: Benchmarking Long-Horizon Spatiotemporal Memory with Harness-Augmented Agents](https://arxiv.org/abs/2610.32332) | 该论文提出了包含约900小时视频和5000多个人工审核问题的 LongVideo 数据集及 4DMemoryBench 基准，并开发了集成四种互补技能、22个以上专用工具模型和检索增强生成系统的 4DMemory Harness，用于评估和提升基础模型的长时程时空记忆推理能力。 |
| [^42] | [System One models Are Not Better Rerankers](https://arxiv.org/abs/2610.32318) | 本研究通过在0.6B至27B参数的六个Qwen骨干模型和十个TREC DL与BEIR数据集上的系统对比，证明决策微调的System One模型并不比普通重排序微调模型是更好的重排序器——重排序微调在0.6B至9B全面胜出5至29个nDCG@10点，27B时两者持平，且9B以下决策微调相比零样本询问未调优模型并无增益。 |
| [^43] | [Many Perturbations, Shared Failure Dynamics: Tracing Direction-Sensitive Perturbation Influence in Language Models](https://arxiv.org/abs/2610.32309) | 该论文提出一种方向敏感、相对于读出过程的语义脆弱性分析框架，通过纵-横分解与储备-传输模型揭示异质扰动在语言模型中共享相同的失败动力学，并证明扰动的方向（而不仅是标量大小）携带因果信息。 |
| [^44] | [DMAIC-Bench: From Self-Reflection to Self-Restraint for Agents in Multi-Stage Analytical Tasks](https://arxiv.org/abs/2610.32305) | 该论文提出 DMAIC-Bench 基准，通过基于精益六西格玛的成对反事实任务首次系统评估大语言模型智能体的“自我克制”能力（即依据证据决定继续还是停止），发现模型在 GO 决策上准确率达 92%，但在 NO-GO 决策上仅约 19%。 |
| [^45] | ["Not" is Not Lost in Translation: A Causal Analysis of Negation Representations in Multilingual LLMs](https://arxiv.org/abs/2610.32281) | 研究发现多语言大模型在各语言间共享可因果干预的否定表征，从一种语言提取的否定激活方向可以在其他语言中诱导或移除否定，且否定由稀疏特征表示。 |
| [^46] | [Do Halting Heads Really Save Compute in Looped Language Models?](https://arxiv.org/abs/2610.32246) | 本文揭示循环语言模型中的停止头因训练方式缺陷和批次生成机制而无法真正节省计算量，无停止头的模型在同等计算成本下表现几乎一样好。 |
| [^47] | [DIAL-S2ST: Learning Latency Control for Simultaneous Speech-to-Speech Translation](https://arxiv.org/abs/2610.32214) | 提出DIAL-S2ST后训练方法，通过前瞻预测自蒸馏学习同声语音到语音翻译模型的推理时延迟控制，仅需10小时源语音即可实现灵活的延迟调节与良好的质量-延迟权衡。 |
| [^48] | [Head Dimension Matters in Language Models for Long-Context Retrieval](https://arxiv.org/abs/2610.32213) | 本研究发现在固定模型规模下，增大注意力头维度可通过提高检索头的比例显著提升语言模型的长上下文检索性能，且即使冻结注意力投影权重该结论依然成立，而头维度对预训练损失和短上下文任务性能影响甚微。 |
| [^49] | [SOPO: Socratic Guided Policy Optimization for Span-Level Hallucination Detection](https://arxiv.org/abs/2610.32212) | SOPO提出了一种苏格拉底引导的多轮强化学习框架，通过教师模型提供基于真实标签的间接提示来引导推理轨迹，从而解决片段级幻觉检测中的定位模糊、困难样本奖励稀疏和长度膨胀问题。 |
| [^50] | [SWE-Shift: Evaluating Coding Agents in Evolving Workspaces](https://arxiv.org/abs/2610.32196) | 提出 SWE-Shift 基准，用于评估编程智能体在任务执行中途遭遇代码、依赖和需求等变化时的表现，实验发现所有 17 个被测智能体系统在发生此类变化时成功率均显著下降。 |
| [^51] | [Collection-Date Assignment Redirects Answers in Temporal QA](https://arxiv.org/abs/2610.32191) | 该研究发现在证据内容完全不变的情况下，仅为文档标注不同的收集日期元数据就能让大语言模型在时间性问答中大幅转向以该日期为参照的答案（精确匹配率下降最高达51.4个百分点），且包括明确忽略指令在内的七种提示策略均无法消除这一效应。 |
| [^52] | [Depth-Dependent Safety Geometry in Vision-Language Models: Layer-Wise Localization and Bidirectional Refusal Control](https://arxiv.org/abs/2610.32182) | 该论文提出一个无需重训的推理时框架，通过逐层激活分析揭示视觉语言模型的安全几何随深度非均匀分布——早期视觉层可通过激活引导实现 6%–96% 的双向拒答控制，而深层虽表征可分却对操控不敏感，从而区分了表征可分性与因果可控性。 |
| [^53] | [Not All Information Is Worth Asking For: Decision-Informed Information Seeking with LLMs](https://arxiv.org/abs/2610.32181) | 提出DIVA方法，让大语言模型基于证据的决策价值扣除成本后来评估每项检查是否值得进行，从而决定该查询什么信息以及何时停止，避免为无法改变决策或不值得的信息付费。 |
| [^54] | [The LLENS Platform for On-Premises Large Language Model Agents in a Japanese University Hospital: Development and Evaluation Study](https://arxiv.org/abs/2610.32151) | 研究团队在北海道大学医院的封闭网络内开发并部署了基于万亿参数开放权重模型Kimi K2.6的临床支持AI智能体平台LLENS，该模型在日本医师执业资格考试中取得480/500的高分，验证了本地化部署医疗AI智能体的可行性。 |
| [^55] | [From Promises to Accountability: Comparative AI Modeling and Visualization of Electoral Commitments in Nepali Political Texts](https://arxiv.org/abs/2610.32150) | 该研究创建了首个尼泊尔政治文本黄金标准数据集（Cohen's kappa = 0.777），并通过对比性AI建模与可视化方法对选举承诺进行结构化，以解决竞选承诺难以追踪和问责的问题。 |
| [^56] | [Disentangling Question, Response, and Verifier Languages in Multilingual P(True) Self-Evaluation](https://arxiv.org/abs/2610.32144) | 该论文通过六语言因子设计首次将多语言自我评估中的问题语言、回答语言与验证语言解耦，发现三者分别影响评估的不同环节——准确性取决于基准、语言匹配提升区分度但不改善校准、验证语言独立偏移校准误差，同时强调评分器验证的重要性。 |
| [^57] | [SyncPro: Synchronous Reasoning and Responding for Proactive Streaming Video Understanding](https://arxiv.org/abs/2610.32132) | SyncPro提出了一种同步推理与响应的机制，解决了主动式流媒体视频理解中持续推理与及时响应难以协调的问题，避免了固定推理间隔导致的无谓更新或延迟响应。 |
| [^58] | [Shorter Is Not Cheaper: A Controlled Comparison of Declarative Animation Specifications and Free-Form Code for LLM-Generated Explanations](https://arxiv.org/abs/2610.32129) | 本研究通过预先冻结假设与分析计划的受控实验发现，尽管声明式动画规范的输出比自由格式代码短25-48%，但“更短即更便宜、更可靠”的普遍预期并不成立。 |
| [^59] | [From Noisy Retrieval to Decision Evidence: Silver Evidence Demonstrations for In-Context Learning](https://arxiv.org/abs/2610.32128) | 提出“银证据示例”方法，用LLM生成的、以目标为条件的证据摘要替代ICL示例中的原始噪声检索段落，从而将噪声检索转化为可靠的决策证据，提升RAG和少样本上下文学习在长噪声文本中的决策能力。 |
| [^60] | [Data Quality in RVL-CDIP: A Systematic Audit](https://arxiv.org/abs/2610.32126) | 本文对文档分类标准基准RVL-CDIP进行了系统性数据质量审计，发现约10.5%的标签错误和35.2%的训练-测试集重复问题，并在修复缺陷后重新基准测试25个模型，显示重训练模型准确率可提升0.9至2.9个百分点。 |
| [^61] | [Semantically Rich, Metrically Affect-Blind: A Geometric Blind Spot in Modern Text Embedders](https://arxiv.org/abs/2610.32122) | 现代文本嵌入器的嵌入向量中几乎可以完美解码情感信息（线性探针准确率高达91.3%），但其度量空间（余弦相似度）几乎完全无法区分情感，导致依赖度量的检索、聚类、去重和RAG系统对这类信息“视而不见”。 |
| [^62] | [EDWARD-NMT: Elastic Dimensions With Adaptive Routing and Dual-consistency for Neural Machine Translation](https://arxiv.org/abs/2610.32115) | EDWARD-NMT 提出了一种结合套娃式弹性维度、带可学习阈值的自适应 sigmoid 路由器以及双一致性目标的新型 MoE 架构，实现“一次训练、随处部署”，使神经机器翻译模型能够在资源受限设备上灵活部署。 |
| [^63] | [Tampered ML: Evaluating LLM Judgment of Machine Learning Research](https://arxiv.org/abs/2610.32114) | 该论文提出Tampered ML基准，通过将真实机器学习论文的实验提案与人为引入方法论缺陷的最小修改版本配对，来评估大语言模型识别研究方法缺陷的能力，发现前沿模型虽优于随机水平但远未达到可靠程度。 |
| [^64] | [REACT: Representational Early Analysis and Control of Chain-of-Thought Trajectories](https://arxiv.org/abs/2610.32113) | REACT 提出一种从成对安全与不安全轨迹中学习安全方向、并一次性施加于思维链早期隐藏状态的激活干预方法，发现最强控制增益集中在思维链的前两到三个 token 窗口，即使后续状态更能预测风险。 |
| [^65] | [Context or Knowledge: How LLMs Attribute Their Own Answers](https://arxiv.org/abs/2610.32109) | 该论文通过对KV缓存的干预实验发现，大语言模型对答案来源的自我归因主要由问答序列（Q+A）的缓存表征决定而非文档内容，并且可以通过沿预拟合方向引导Q+A值以近乎100%的成功率系统性地翻转来源报告。 |
| [^66] | [Beyond What to Say: Proactive Utterance Timing with Inner Thoughts for Werewolf Agents](https://arxiv.org/abs/2610.32107) | 本研究针对狼人杀游戏的竞争性和不完全信息特点，重新设计了基于“内心想法”框架的LLM智能体系统，使其能够自主判断发言时机并决定发言内容。 |
| [^67] | [Putting Transformer Theories to the Test](https://arxiv.org/abs/2610.32104) | 该论文将七种关于Transformer能力的理论预测统一到同一矩阵中，并通过受控实验验证，发现没有任何单一理论能完全解释Transformer在不同计算任务上的表现。 |
| [^68] | [Behaviour analysis for large language model-driven agents](https://arxiv.org/abs/2610.32087) | 提出了一个将数据驱动的行为发现、可解释的依赖关系分析与实验验证（含提示引导）相结合的通用框架，用于理解大语言模型驱动智能体在轨迹中如何组织行为及其与情境和结果的关系。 |
| [^69] | [Concentrated Causal Influence, Distributed Diagnostic Information: Cognitive Impairment Decisions in Large Audio-Language Models](https://arxiv.org/abs/2610.32061) | 该论文通过因果干预识别出大型音频-语言模型解码器MLP中的“认知障碍对比神经元”（CICNs），首次从因果层面证明认知障碍决策由少量神经元集中控制，而诊断信息以分布式形式存在于模型表征中，且这一机制在不同模型间和跨语料库上具有一定泛化性。 |
| [^70] | [How Much Do You Trust A Dog? Quantifying Hint-Following in Reasoning Language Models](https://arxiv.org/abs/2610.32055) | 该研究发现大语言模型对提示来源（如教授、朋友甚至狗）的答案偏移遵循可预测的指数倾斜结构，且偏移强度与来源可靠性基本无关，揭示了模型谄媚行为未被来源可靠性校准的问题。 |
| [^71] | [X-Dubbing: Speech-to-Speech Translation under Temporal and Visual Constraints](https://arxiv.org/abs/2610.32042) | X-Dubbing 将跨语言配音形式化为时间感知的语音到文本翻译与视频感知的文本到语音合成的联合任务，在无需成对配音数据的情况下即可生成与原视频时间同步的目标语言配音。 |
| [^72] | [CRM-Agent-Bench: Benchmarking Tool-Using Agents in Stateful Enterprise Environments](https://arxiv.org/abs/2610.32004) | 该论文提出 CRM-Agent-Bench，一个包含 555 个有状态多轮任务、覆盖九大业务领域的执行型基准测试，从结果、轨迹、运营可行性和政策合规四个维度联合评估语言智能体在真实企业 CRM 环境中执行多步骤业务流程的真实能力。 |
| [^73] | [DraMaze: A Multi-Agent Pipeline with Future-Aware Memory for Diverse Microdrama Script Generation](https://arxiv.org/abs/2610.32002) | 提出DraMaze多智能体流水线，通过套路记忆引导的进化树搜索和基于未来叙事目标的反向叙事依赖树，实现既多样又保持长程连贯的微短剧剧本生成，并发布双语评测基准MiDramaBench。 |
| [^74] | [AusClinBench: a deterministically scored benchmark of large language models against current Australian clinical policy](https://arxiv.org/abs/2610.31993) | 该研究构建了基于澳大利亚现行临床政策、采用精确匹配确定性评分的142题基准AusClinBench，对28个大语言模型的测试显示盲测准确率为51.5%–92.7%，且86%的错误在提供相关条款后即可纠正，表明模型失败主要源于规则记忆而非规则应用能力。 |
| [^75] | [The Helpful Agent Problem: When Emergent Cooperation Among LLM Agents Bypasses Intended Workflows](https://arxiv.org/abs/2610.31985) | 该论文揭示了“有益智能体问题”：即使没有恶意或共谋，追求各自目标的良性LLM智能体也会在审查机制失效时自发合作、代替彼此执行操作，从而在合法权限范围内绕过预期的监督工作流，构成一种新型AI安全风险。 |
| [^76] | [Knowing Whom to Trust is not Enough: The Trust--Agreement Gap in Repeated Multi-Party LLM Negotiation](https://arxiv.org/abs/2610.31971) | 本文提出了一个重复多方、多议题的LLM谈判框架，发现LLM智能体虽然能从历史交互中形成有用的信任估计并识别暗中阻挠协议的对手，但知道该信任谁并不足以保证达成协议，从而揭示了“信任-协议鸿沟”。 |
| [^77] | [POS-Signature Demonstration Retrieval for In-Context Dependency Parsing](https://arxiv.org/abs/2610.31966) | 该论文提出了一种基于加权词性统计的"词性签名"示例检索策略，为上下文依存句法分析选取更相关的演示示例，无需微调即在两个开源语言模型上超越了语义相似度检索等基线方法，取得了最高的标注依存分数。 |
| [^78] | [Which Answer Did the Seventeen-Year-Old Write? A Behavioral Case Study of Author-Age Misattribution, Miscalibrated Confidence, and Non-Corrective Self-Explanation in Six Frontier Language Models](https://arxiv.org/abs/2610.31946) | 本研究通过1,081次API会话、3,225份回复的行为学案例研究，发现六个前沿语言模型在将手写答案匹配到作者年龄时表现接近随机却报告73-88%的高置信度，17岁少年的答案在高达90%的情况下被误判为成年人所写，且移除选择题提示后准确率进一步崩塌，揭示了前沿大模型在作者年龄归因任务中存在严重的系统性误判、过度自信及自我解释无法纠错的问题。 |
| [^79] | [Kolibri: A Sovereign European Model on the Pareto Frontier](https://arxiv.org/abs/2610.31945) | Kolibri是一个总参数78.1B、激活参数仅3.46B的英德双语开源混合专家模型，通过完全自主掌控的数据与训练流程实现德语占比超20%（含2T自建德语token）的训练，在帕累托前沿上实现了面向主权化部署的高效性能。 |
| [^80] | [Agreement rather than normalisation keeps depth reusable in looped language models](https://arxiv.org/abs/2610.31943) | 循环语言模型深度可复用的关键在于训练时与推理时循环配置的一致性，而非循环步之间的状态归一化。 |
| [^81] | [LazyFlash: Forward-Dense, Backward-Sparse Attention for Accelerated Long-Context Fine-Tuning](https://arxiv.org/abs/2610.31911) | LazyFlash利用注意力训练开销主要由反向传播主导这一观察，提出前向保持稠密、后向采用梯度感知块稀疏化的方法，跳过对参数更新贡献可忽略的查询块，在保持模型为标准稠密模型（可被vLLM等系统直接部署）的同时显著加速长上下文微调。 |
| [^82] | [Knowing When to Stop: The Evidence-Action Gap in Coding Agents](https://arxiv.org/abs/2610.31895) | 该论文揭示了编码智能体的“证据-行动差距”现象——即智能体即使已收集到表明任务无法独立完成的充分证据仍过度坚持或虚报成功，并提出ImpasseBench基准通过在可解决任务中注入环境阻碍并标记证据充分步骤来量化这一差距。 |
| [^83] | [TaxoSAE: Rethinking Sparse Autoencoders through Taxonomic Logic](https://arxiv.org/abs/2610.31894) | TaxoSAE通过在稀疏自编码器特征空间中显式编码is-a分类学关系，利用按分类层级堆叠的SAE结构和分类对齐损失，实现了对大语言模型知识更准确且具分类一致性的读取与更精细粒度的控制。 |
| [^84] | [Mining Frames: Association Rule Mining as an Unsupervised Framework for News Framing Detection](https://arxiv.org/abs/2610.31874) | 该论文提出将关联规则挖掘（ARM）作为无监督、无需标注的新闻框架检测新框架，通过对加沙战争首年25,512篇新闻文章的分析验证了其无需预定义分类体系即可归纳发现可解释框架模式的能力。 |
| [^85] | [Rewriting Scaling Laws for Synthetic Data with Compute and Data Constraints](https://arxiv.org/abs/2610.31867) | 本文提出了首个面向合成重写数据的缩放定律，将生成重写的计算成本纳入建模，发现最优重写数据比例随计算预算的增加而显著上升。 |
| [^86] | [Machine-Readable Operator Manuals over the ISO 11783 Network for Offline, Multilingual LLM Assistance](https://arxiv.org/abs/2610.31863) | 本文提出在ISO 11783网络上对操作手册进行语义同步，使拖拉机侧的离线多语言LLM助手能够在31至89秒内获取完整的机具文档，传输时间与操作员首次连接机具时等待对象池上传的时间相当。 |
| [^87] | [Aligning Large Language Models with Human Feedback: Mathematical Foundations and Algorithm Design](https://arxiv.org/abs/2610.31862) | 本文系统介绍了大语言模型对齐的数学基础与算法框架，涵盖SFT、RLHF和DPO等主流技术，并将其理论与逆强化学习和离散选择模型建立联系，为开发更稳健、可扩展的对齐方法提供了全面参考。 |
| [^88] | [SWE-Clar: Can Coding Agents Clarify Incomplete Client Requirements and Deliver Greenfield Software?](https://arxiv.org/abs/2610.31854) | 该论文提出了 SWE-Clar 基准，基于 100 个报酬至少 4,000 美元的真实 Upwork 客户委托任务，通过“先澄清需求、后实现代码”的多轮交互设计并以明确的需求理解测试来评估编码智能体在自由职业全新软件开发中澄清不完整客户需求的能力。 |
| [^89] | [Scaling cross-situational word learning on children's naturalistic experiences](https://arxiv.org/abs/2610.31842) | 该研究探讨了基于儿童自然第一人称视频经验的跨情境词汇学习能否规模化，重点考察其表现对视觉表征质量的依赖性以及在更大、更多样化数据集上的扩展能力。 |
| [^90] | [SPECMEM: Accelerating Agentic Tool Calling via Live Memory Management](https://arxiv.org/abs/2610.31840) | SpecMem 提出一种在部署全过程中持续活跃管理的工具调用记忆机制，通过在线写回已验证调用（配合 LRU 淘汰）和副作用门控，使投机解码在多用户多会话场景下相比冻结草稿库保持约 27% 的平均接受 token 优势。 |
| [^91] | [SAGE: Sense-Aligned Multimodal Grounding Evaluation with Gloss-Augmented Reasoning](https://arxiv.org/abs/2610.31838) | 本文提出统一的多模态词义定位评估基准SAGE，并引入免训练的释义增强推理框架GAR，通过结合定义定位与约束引导语义验证，显著提升了多模态大模型在不同定位任务上的词义消歧性能。 |
| [^92] | [Latent Coherence: Do Frontier Language Models' Probability Judgments Obey the Law of Total Probability?](https://arxiv.org/abs/2610.31833) | 该论文首次系统性地检验前沿语言模型在零样本条件下的概率判断是否满足全概率定律及因果后门调整，发现 GPT-6.1 Sol 和 Claude Opus 5.5 等模型已自发表现出接近贝叶斯后验的潜在概率一致性。 |
| [^93] | [What the Query Remembers: Query Abstraction Governs Lexical and Dense Retrieval of Classical Arabic Poetry](https://arxiv.org/abs/2610.31821) | 该论文发现，在古典阿拉伯语诗歌检索中，查询保留诗歌原文措辞的程度（即查询的抽象程度）决定了检索器的优劣——记忆型查询下BM25领先所有稠密模型，而抽象查询下稠密编码器胜出，且BM25对措辞变化的敏感度远高于稠密模型。 |
| [^94] | [Free Lunch for LLMs Confidence: The RL Critic Already Knows](https://arxiv.org/abs/2610.31820) | 本文证明在基于正确性的结果下，LLM的置信度估计与强化学习中的评论家训练在数学上等价，因此RL训练得到的评论家无需额外的数据和计算即可免费提供可靠的模型置信度估计。 |
| [^95] | [LAMB: A Modern Contextual Language Model for Latin](https://arxiv.org/abs/2610.31723) | 研究者基于ModernBERT架构，采用新颖的形态学感知掩码方案在740亿拉丁语词元上分两阶段预训练出LAMB编码器，并发布Acervus和首个拉丁语语义三元组数据集Corpus Trinum两个新语料库，为拉丁语语义相似度等下游任务奠定基础。 |
| [^96] | [Omni-UBench: Benchmarking Omni-Modal LLM-based Guardrails Across Safety, Utility, and Reliability](https://arxiv.org/abs/2610.31713) | 该论文提出了首个全面的全模态安全评估基准Omni-UBench，从安全性、实用性和可靠性三个维度评估基于全模态大语言模型的防护栏模型，涵盖超过18.3万个来自文本、图像、音频、视频及跨模态组合的样本。 |
| [^97] | [Fooling LLM Safety Guards into Judging the Wrong Text](https://arxiv.org/abs/2610.31695) | 论文提出“目标重绑定”攻击方法——通过附加伪造查询和无害诱饵，利用安全防护机制只评判输入中最后一个完整任务的特性，使有害内容绕过审查，在GuardRebind基准上对五种防护机制的绕过率均超过93%。 |
| [^98] | [Distributional Alignment of LLMs and Its Generalization Properties](https://arxiv.org/abs/2610.31687) | 该研究提出了大语言模型与人类分布对齐的方法，发现对齐后的模型能够泛化到同领域的未见问题，跨主题泛化取决于主题间的关联性，但可能伴随创造力等能力的意外退化。 |
| [^99] | [From Target Selection to Condition Reuse: Auditing Residual Interventions in Tool-Use Models](https://arxiv.org/abs/2610.31686) | 激活干预虽能大幅提升模型对熟悉目标的选择正确率，但这种增益无法泛化到事实与标识符的重新组合或独立资格判断上，仅能改变源事实影响判断的方式。 |
| [^100] | [An Agentic Approach with Evidence-Grounded Verification for Multimodal Emotion Reasoning in Conversation](https://arxiv.org/abs/2610.31685) | 该论文提出了一种可追溯的多模态情感推理智能体框架，通过模态感知智能体发布带证据引用的观察、集成智能体生成情感标签与引用，并利用独立的基于证据的验证阶段执行确定性检查来定位错误来源，从而以低成本实现对话中的多模态情感推理。 |
| [^101] | [Knowledge Graphs Quality Assessment in the Era of LLMs: A Conceptual Framework](https://arxiv.org/abs/2610.31684) | 本文回顾了数据、信息、关联数据及知识图谱与大语言模型交叉领域的质量评估文献，将现有最先进框架扩展为包含24个质量维度的目录，并提出了一种通用、可按领域或任务定制且实用的知识图谱质量评估概念框架。 |
| [^102] | [The Schema Coherence Inventory: Measuring Whether an Assigned Belief Organizes a Language Model's Cognition](https://arxiv.org/abs/2610.31683) | 本文提出图式连贯性量表（SCI），一种基于贝克认知模型的96题强制选择、无生成评分工具，用于验证赋予语言模型的信念是否真正组织其认知而非仅仅引发情绪，并通过目标特异性判别标准在五个骨干模型上得到验证。 |
| [^103] | [Multi modal perceived conversation success prediction for spontaneous autistic and non-autistic dyadic interactions](https://arxiv.org/abs/2610.31682) | 本研究提出利用构音、面部、声学-韵律和话轮转换等多模态特征，在包含自闭症与非自闭症个体的自发双人对话数据中预测参与者感知的对话成功，证明对话成功本质上是多维的，需要多层面的分析框架。 |
| [^104] | [Rethinking Structured Generation: Can Graph-Based Reasoning Resolve Ambiguity?](https://arxiv.org/abs/2610.31681) | 该论文将结构化生成重新表述为歧义感知推理问题，提出基于图推理的多有效学习框架，通过对一组合理的结构化输出进行推理而非预测单一解，解决了自然语言歧义导致现有方法输出不稳定和不一致的问题。 |
| [^105] | [MisconceptDialogue: Evaluation Criteria Shape Measured Success in LLM Math Tutoring](https://arxiv.org/abs/2610.31679) | 该论文提出MISCONCEPTDIALOGUE全自动评估框架，揭示评估标准对LLM数学辅导效果测量结果影响巨大：在相同对话上，精确匹配评分报告的纠正成功率仅为42.1%，而人工校准的LLM裁判评分高达81.3%。 |
| [^106] | [Validating the Comparison, Not Just the Scorer: Why Overall Accuracy Can Mislead in Interactive LLM Evaluation](https://arxiv.org/abs/2610.31678) | 该论文指出，仅用总体准确率验证自动评分器会掩盖针对特定提示的评分误差，从而可能扭曲交互式LLM评估中提示间比较的方向和幅度，因此应当验证“比较结果”本身而不仅是评分器。 |
| [^107] | [How Ground Truth Shapes Retrieval Scores: Evidence-Aware Evaluation of a Retrieval-Augmented Generation System for Breast Lesions](https://arxiv.org/abs/2610.31606) | 该论文提出了一种证据感知的检索评估协议，通过将证据片段映射到所有包含块来定义相关性，从而解决检索得分依赖真值标注覆盖范围的问题，并将其应用于乳腺病变管理文献临床RAG系统的评估。 |
| [^108] | [Mostly Reading: What Open-Book Scores Measure in Document-Level LoRA Adapters](https://arxiv.org/abs/2610.31017) | 开卷评分主要衡量的是所有适配器共有的“阅读”行为而非文档记忆——随机适配器即可恢复专用适配器82%的开卷优势，但在无源文本的完形测试中仅恢复约2%（与零无异），因此不宜用开卷评分来评估或选择文档级LoRA适配器。 |
| [^109] | [Scaling Laws for Safety Probes](https://arxiv.org/abs/2610.30678) | 本文通过构建六个新安全数据集、在20个基础模型上以不同标签规模训练探针，首次提出了可依据模型规模和标签数量提前预测安全探针性能的缩放定律。 |
| [^110] | [Rethinking SAE Evaluation: An Atomic Interpretable Unit Framework Reveals Hidden Polysemanticity and Redundancy](https://arxiv.org/abs/2610.30474) | 本文提出了一种基于原子可解释单元的SAE评估新框架，将超越单义性的评估目标形式化，并借此揭示了现有SAE特征中隐藏的多义性与冗余性。 |
| [^111] | [OACap: Caption-Guided Active Observation for Detailed Video Captioning](https://arxiv.org/abs/2610.29327) | 提出了以描述为中心的智能体框架 OACap，它能将视频描述中缺失或缺乏支持的信息转化为主动观察请求，借助时空工具获取证据并迭代验证修订描述，从而显著提升详细视频描述质量，同时引入 FSCap-Bench 基准及 CSFS 等评分指标来系统评估描述的事实性与语义保真度。 |
| [^112] | [SANE: Automated Scorer-Specific Metamorphic Testing for LLM Judges](https://arxiv.org/abs/2610.28886) | SANE是一个基于智能体的自动化框架，能针对特定LLM评判器自动识别其评分能力、生成变异测试配方及预期分数关系，从而将评判器可靠性测试实现全流程自动化，摆脱了人工编写扰动配方和预设评分器类型的限制。 |
| [^113] | [ASTOR: Benchmarking and Improving Autoformulation for Soft-Constraint Optimization with a Formal Grammar](https://arxiv.org/abs/2610.26819) | 本文提出ASTOR框架，通过形式文法引导的抽象语法树将自然语言决策问题确定性地编译为可执行的优化模型，填补了软约束优化自动建模研究的空白并提供了系统基准测试。 |
| [^114] | [AfriGuard: Safety and Robust Refusal For African Languages](https://arxiv.org/abs/2610.26474) | 该论文提出了AfriGuard，一个覆盖10种非洲语言和英语的扎根文化背景的多语言数据集，用于研究有害意图检测、安全类别理解与拒答行为，并揭示了现有护栏模型在这些安全任务上表现不佳、对适当安全行为理解有限的现状。 |
| [^115] | [Token-Set Choice Confounds POPE: A Systematic Audit of Yes/No Extraction in VLM Hallucination Evaluation](https://arxiv.org/abs/2610.25964) | 该论文系统审计了POPE幻觉评估中的是/否答案提取方式，发现传统双词元logit读出所比较的"yes"/"no"词元ID在LLaVA-1.5-7B的实际输出中从未出现过（模型实际生成的是带空格前缀的" Yes"/" No"），仅这一读出实现细节就能使F1分数偏移高达6.13分，说明评估代码的实现选择会严重混淆基准对比结论。 |
| [^116] | [EvoSkill v2: Fully Agentic Skill Evolution from Benchmark Adaptation to a Stateful Evolver](https://arxiv.org/abs/2610.25931) | EvoSkill v2 提出了一种由三个智能体组成的技能演化系统，其核心创新在于采用一个贯穿全程、带有 bash 工具的有状态编码智能体在每个训练周期持续重写技能指令，并结合多次尝试与无筛选接受策略，在不改动底层模型的前提下提升智能体的基准测试表现。 |
| [^117] | [Muse Glimmer 30B: An On-Device Agentic Model](https://arxiv.org/abs/2610.24628) | Muse Glimmer是一个开源的30B参数端侧智能体模型，集成了多步推理、可靠的工具使用和故障恢复能力，可在消费级硬件上本地运行且无需云端，并在智能体基准测试中超越Gemma4-31B和Qwen3.6-27B等同规模模型达到最先进性能。 |
| [^118] | [Chinese Decision: A Family of Language Models for Single-Pass Probabilistic Decision-Making](https://arxiv.org/abs/2610.23170) | Chinese Decision 是一系列中文语言模型，通过单次前向传播直接对候选选项的 logits 进行归一化得到概率分布，无需自回归生成，在医学、法律、金融等领域的决策任务上实现了高准确率与良好校准。 |
| [^119] | [Same Context, Different Agents: A Behavioral Study of Semantic Layers, Orchestration and Compaction in Data Agents](https://arxiv.org/abs/2610.20959) | 该论文首次系统性地研究了语义层、编排与压缩三种上下文工程技术在数据智能体中单独及组合使用时的行为、准确性与成本影响，发现无这些技术的智能体会呈现调查者、过早回答者和过早停止者三种行为群体，且每种技术的收益取决于其所匹配的智能体行为群体。 |
| [^120] | [Cross-Model In-Context Neurofeedback Reveals (Limited) Activation Control in LLMs](https://arxiv.org/abs/2610.20179) | 本文提出跨模型上下文内神经反馈新范式，首次将模型在任务上的胜任能力与对自身激活的真实访问区分开来，测试十个开源大模型后发现大语言模型仅具备有限的激活控制能力。 |
| [^121] | [Recurrent Self-Improvement: Dynamic Cross-Loop On-Policy Distillation for Looped Language Models](https://arxiv.org/abs/2610.10623) | LoopOPD让循环语言模型利用自身更深层循环计算作为冻结“教师”，在学生自生成轨迹上进行在线策略蒸馏，无需外部教师或特权信息即可获得密集监督实现自我提升，D-LoopOPD进一步将该过程动态化。 |
| [^122] | [Constitution-Guided Watermarking](https://arxiv.org/abs/2610.09552) | 本文提出“宪法引导水印”框架，通过将提供商需求表示为自然语言原则，使水印系统能够根据不同请求的需求灵活选择属性权衡，避免了传统方法对所有请求采用统一配置所导致的牺牲问题。 |
| [^123] | [ToolRACER: A Robust Agentic Conversation Emulation Resource for Agent Training and Evaluation](https://arxiv.org/abs/2610.09163) | 该论文提出ToolRACER合成数据生成流水线，通过协调用户、助手和工具模拟模型生成含对抗性行为的多轮对话数据，并构建了包含5.6K条验证对话轨迹（约66%含易失败场景）的ToolRACERBench基准，用于训练和评估鲁棒的任务导向对话智能体。 |
| [^124] | [SOL: Measuring Gaps between Text Distributions by Double Sliced Wasserstein Metrics](https://arxiv.org/abs/2610.06513) | 提出SOL——一种基于固定Transformer隐藏状态经验测度的双切片Wasserstein距离的文本分布距离度量，当Transformer为单射时可证明其为真正的度量，为非自回归语言模型的分布拟合评估提供了稳定的样本级评估方案。 |
| [^125] | [A Guideline-Augmented Multi-Agent Framework for Schema-as-Code Biomedical Named Entity Recognition](https://arxiv.org/abs/2610.02970) | GAMA框架通过从标注数据中归纳并验证特定数据集的标注规则来构建指南记忆，结合多智能体协作与模式即代码的结构化输出，显著提升了生物医学命名实体识别的准确性和格式规范性。 |
| [^126] | [Enhancing Biomedical Named Entity Recognition via Multiple Programming Languages Instruction Tuning and Ensemble Method](https://arxiv.org/abs/2610.02949) | 该论文提出MITE方法，将生物医学命名实体识别重构为结构到结构的生成任务，通过多种编程语言的指令微调与集成策略，在不依赖昂贵外部知识资源的条件下提升了识别性能与模型鲁棒性。 |
| [^127] | [Beyond Overlap: Estimating the Causal Effect of Benchmark Exposure](https://arxiv.org/abs/2609.27176) | 提出LeakScale干预框架，通过控制基准家族私有信息的暴露并构建全新可执行任务，首次因果量化了训练数据污染对模型评估准确率的实际影响，发现暴露使准确率提升7.17至27.31个百分点。 |
| [^128] | [BabelArena: A Large-Scale Multilingual Benchmark for LLM Agents](https://arxiv.org/abs/2609.23490) | 提出了BabelFlow通用工作流和BabelArena大规模多语言基准测试（涵盖23种语言、16,146个实例），首次系统评估了LLM智能体的跨语言能力，发现前沿模型存在显著的跨语言性能差异且无单一模型全面领先。 |
| [^129] | [Beyond Static and Linear: What Attention Constraints Best Fit Human Reading Times?](https://arxiv.org/abs/2608.23818) | 该论文系统比较了多种基于注意力的记忆约束机制（静态与动态），发现对中间词元内容敏感的约束最能拟合人类阅读时间数据。 |
| [^130] | [TreeWY: Speculative Verification for Gated DeltaNet Hybrids](https://arxiv.org/abs/2608.20961) | 本文提出TreeWY方法，通过树状WY变换消除推测解码中的状态快照，显著降低内存开销，从而支持高接受率的宽草稿树。 |
| [^131] | [Persona-Guided LLM Agents for Task-Oriented Dialogue](https://arxiv.org/abs/2608.18085) | 本研究提出一个无需训练的框架，通过模拟两个大语言模型代理的任务导向对话，验证了系统适应用户人格能在不损害任务完成度的前提下提升交互质量。 |
| [^132] | [Every Coin Has Two Sides: On the Dual Nature of Generalization in On-Policy Distillation of Large Language Models](https://arxiv.org/abs/2608.16647) | 同策略蒸馏的泛化行为取决于教师和学生来源关系，同源对能跨域迁移推理能力，跨源对则受限于训练分布，这种双重性既是优势也是风险。 |
| [^133] | [When Humans Leave Context Unsaid: Implicit Context as a Risk for Large Language Models](https://arxiv.org/abs/2608.09551) | 该论文提出了大语言模型的“隐式风险”这一新概念：由于预设、会话含义等隐式语境在训练数据中缺失且被对齐过程忽视，攻击者可引导模型推断提示中未言明的危害性语境，从而诱导其生成更多有害内容。 |
| [^134] | [Romanized Arabic Across Dialects: Views, Usage Patterns, and Linguistic Variation](https://arxiv.org/abs/2608.02555) | 本文通过调查阿拉伯语使用者并发布字符级词对齐资源，首次系统揭示了五种方言中阿拉伯语拉丁化书写的方言内规律性与方言间系统性变异。 |
| [^135] | [AgentGUI: An Interface for Observing and Steering Long-Running AI Agents](https://arxiv.org/abs/2607.26300) | AgentGUI是一个本地托管的图形界面，通过丰富的轨迹可视化、手动与自动引导以及多框架集成协调，显著提升了人类对长时运行AI代理的监督效率（轨迹分析提速38%）和小型本地代理的任务完成率（最高提升34个百分点）。 |
| [^136] | [A Cross-lingual Comparison of Human and Classification Model Entrainment Behavior in Code-switched Speech Settings](https://arxiv.org/abs/2607.25202) | 该研究首次对语码转换口语对话中的人类趋同行为进行跨语言比较，发现词汇趋同具有跨语言普遍性而声学韵律与语码转换风格趋同依赖具体语境，且分类模型虽能检测趋同但所依赖的特征与人类行为中最显著的特征并不一致。 |
| [^137] | [JANUS: Foreseeing Latent Risk for Long-Horizon Agent Safety](https://arxiv.org/abs/2607.19913) | Janus提出前瞻式智能体安全框架，通过预判任务与裁决任务的联合强化学习训练出守护模型Vanguard，能够在不安全操作执行前预见延迟风险，使平均防护提升15.9个百分点、良性任务完成率提升5.1个百分点。 |
| [^138] | [Expanding the Lexicon of Ge'ez Based African Languages: A Comparative Study of Amharic and Tigrinya](https://arxiv.org/abs/2607.15209) | 该论文提出VEXMLM，通过扩展词汇和两阶段训练，显著提升了XLM-R在阿姆哈拉语和提格里尼亚语等低资源格厄兹文字语言上的性能，尤其在词汇外实体识别方面。 |
| [^139] | [How Do I Know What to Say Next? Barenholtz's Autogenerative Theory as an Enrichment of Harrisean Integrationism](https://arxiv.org/abs/2607.07891) | 本文提出巴伦霍尔茨基于大语言模型行为发展出的语言自生成理论，能够填补哈里斯整合主义语言学在符号未来开放性机制、语言与非语言符号活动连续性以及历史整合档案结构属性方面的解释空白，从而在不违背其核心主张的前提下丰富整合主义理论。 |
| [^140] | [GaP: A Graph-as-Policy Multi-Agent Self-Learning Harness For Variational Automation Tasks](https://arxiv.org/abs/2607.05369) | 本文提出了Graph-as-Policy（GaP），一个多智能体自学习编码框架，它从模块化开源机器人技能库生成包含感知、规划和控制节点的有向计算图，并利用内部仿真环境进行排练，从而提升物体几何与姿态变化较大的变分自动化任务的可靠性。 |
| [^141] | [Cluster, Route, Escalate: Cascaded Framework for Cost-Aware LLM Serving](https://arxiv.org/abs/2606.27457) | 提出了CRE-Router两阶段级联框架，在延迟预算约束下先通过聚类将查询路由到成本最优的模型，再通过质量估计将不合格答案升级到更强的模型，从而在大语言模型服务中兼顾准确率与计算成本。 |
| [^142] | [Honeyquest for LLMs: Rethinking Cyber Deception for AI Attackers](https://arxiv.org/abs/2606.21037) | 本文通过大规模评估21个大语言模型，发现AI攻击者比人类更易落入网络欺骗陷阱，提出了一种新的自动化评估框架，并指出大语言模型构成独特的攻击者类别。 |
| [^143] | [AlphaMemo: Structured Search-Process Memory for Self-Evolving Alpha Mining Agents](https://arxiv.org/abs/2606.20625) | AlphaMemo通过从AST差异中提取编辑模式、置信度门控残差记忆和非对称否决控制构建结构化搜索过程记忆，实现了自进化的Alpha挖掘智能体，在中证500和标普500上显著提升了样本外表现和固定预算下的因子发现效率。 |
| [^144] | [Continuous Audio Thinking for Large Audio Language Models](https://arxiv.org/abs/2606.18273) | 提出连续音频思考（CoAT）框架，通过音频专家蒸馏为大型音频语言模型构建连续潜在思考空间，使其在生成响应前能够组织并利用语音细节、韵律、声音事件、情感和音高等丰富声学信息。 |
| [^145] | [Characterize Then Distill: Mechanistic Reasoning in Large Output Spaces](https://arxiv.org/abs/2606.06840) | 本文通过将多标签决策建模为token级事件，并结合归因、消融与植入等因果分析手段，刻画了推理型大模型在海量候选标签空间中进行选择的内部注意力头机制，并证明该机制可以被蒸馏。 |
| [^146] | [SkillPyramid: A Hierarchical Skill Consolidation Framework for Self-Evolving Agents](https://arxiv.org/abs/2606.03692) | SkillPyramid提出了一种基于层次化技能拓扑的技能巩固框架，通过自进化机制使智能体在任务执行中组合、验证并积累新技能，从而实现技能经验的复用与更广泛的任务泛化。 |
| [^147] | [RCEM: Robust Conversational Search EMbedder in Distributional Shift](https://arxiv.org/abs/2606.01697) | RCEM通过将对话与LLM重写的查询对齐而非直接匹配长段落，在保留原始嵌入空间的前提下，实现了减少过拟合、无需相关性标注且无需重建索引的鲁棒对话搜索检索。 |
| [^148] | [Probing Collision Grounding in Vision-Language Models for Safe Human-Robot Collaboration](https://arxiv.org/abs/2605.31196) | 该论文提出了 TouchSafeBench——首个基于物理仿真构建、用于评估视觉-语言模型“碰撞定位”能力的基准，包含近三千个人机共处场景，支持安全状态分类与即将碰撞预警两项实际部署任务。 |
| [^149] | [GRKV: Global Regression for Training-Free KV Cache Compression in Long-Context LLMs](https://arxiv.org/abs/2605.31105) | 提出了一种无需训练的KV缓存压缩方法GRKV，通过全局回归直接最小化压缩缓存与完整缓存注意力输出之间的差异，解决了基于区间保留策略中合并模式高度不平衡所导致的过度合并与信息损失问题。 |
| [^150] | [Efficient Benchmarking Is Just Feature Selection and Multiple Regression](https://arxiv.org/abs/2605.25773) | 该论文将高效基准测试重新构建为带特征选择的多元回归问题，通过结合核岭回归与mRMR特征选择算法，在预测误差和排序相关性上显著改进了现有高效基准测试方法。 |
| [^151] | [Translators as Invisible Teachers of AI: Copyright, Translation Memory, and the Political Economy of Linguistic Data](https://arxiv.org/abs/2605.24842) | 译者的翻译劳动（以翻译记忆和平行语料库为载体）已成为AI发展的基础数据资本，但其创造性贡献在版权法下被抹去，本文从语言数据的政治经济学视角揭示了这一剥削机制。 |
| [^152] | [NodeSynth: Socially Aligned Synthetic Data for AI Evaluation](https://arxiv.org/abs/2605.14381) | NodeSyn通过将微调的分类体系生成器与可信文献来源和敏感属性相结合，生成反映真实社会背景的敏感领域评估数据，其诱发的模型失败率达到通用合成基线的3.1倍、真实人类查询的1.6倍，显著优于现有合成数据方法。 |
| [^153] | [Deep Reasoning in General Purpose Agents via Structured Meta-Cognition](https://arxiv.org/abs/2605.11388) | 提出“深度推理”方法，利用形式化语言将元推理表示为可执行的分解，在推理时动态构建任务特定的脚手架，使LLM智能体能像人类一样灵活地在联想推理、形式化计算和递归求解等不同推理模式间切换。 |
| [^154] | [MedAction: Towards Active Multi-turn Clinical Diagnostic LLMs](https://arxiv.org/abs/2605.07305) | 提出MedAction树状结构蒸馏流水线，通过大语言模型与环境交互合成高质量多轮诊断轨迹，使模型学会在部分证据下主动开具检查、解读结果并更新鉴别诊断，克服了静态单轮诊断评估的局限。 |
| [^155] | [Better Reward Models Are Not Always Less Biased: Social Alignment Beyond Standard Reward Benchmarks](https://arxiv.org/abs/2605.05003) | 该论文将奖励模型评估扩展到偏见、安全、道德和伦理推理四个社会关键领域，发现标准基准性能与社会对齐行为之间仅有不一致的关联，且更强的奖励模型在应对多元社会期望时并不一定表现更好，揭示了偏见缓解与情境忠实性之间的权衡。 |
| [^156] | [daVinci-Env: Open SWE Environment Synthesis at Scale](https://arxiv.org/abs/2603.13023) | 提出了 OpenSWE——目前最大的完全开源透明的软件工程智能体训练框架，包含 45,320 个覆盖 1.28 万仓库的可执行 Docker 环境，并借助 64 节点集群上的多智能体合成流水线自动化构建。 |
| [^157] | [MetaKE: Meta-Learning for Knowledge Editing Toward a Better Accuracy-Editability Trade-off](https://arxiv.org/abs/2603.12677) | 提出MetaKE框架，通过元学习将知识编辑中上游表示优化与下游参数优化统一为双层优化问题，解决了两阶段优化割裂导致的编辑不充分或可编辑性下降问题，实现了精度与可编辑性之间更优的权衡。 |
| [^158] | [MoE-SpAc: Efficient MoE Inference Based on Speculative Activation Utility in Heterogeneous Edge Scenarios](https://arxiv.org/abs/2603.09983) | 提出MoE-SpAc框架，将投机解码创新性地重新用作内存管理的前瞻传感器，通过投机效用估计器、异构工作负载均衡器和异步执行引擎，在统一的效用空间中协调专家预取与逐出，使异构边缘设备上的MoE推理TPS较SOTA基线提升42%。 |
| [^159] | [CrimeNER: Curated Dataset and Methods for Research and Investigation in Language Intelligence](https://arxiv.org/abs/2603.02150) | 该论文提出了CrimeNER，一个包含1500多份标注文档的犯罪领域命名实体识别数据集，定义了4种粗粒度和21种细粒度犯罪实体类型，并通过传统NER模型和大语言模型的实验验证了数据集质量。 |
| [^160] | [A Diversity Diet for a Healthier Model: A Case Study of French ModernBERT](https://arxiv.org/abs/2602.22014) | 该研究表明，采用基于多样性的数据采样策略，可以在保持性能的前提下将法语 ModernBERT 的预训练数据集缩减高达 94%、预训练时间缩短高达 73%。 |
| [^161] | [Under the Influence: Evaluating and Improving LLM Vigilance in Agentic Tasks](https://arxiv.org/abs/2602.21262) | 该论文针对大语言模型智能体容易被外界影响的问题，提出在现实智能体任务中评估并提升其“警觉性”，即辨别何时应采纳善意建议、何时应拒绝恶意建议的能力。 |
| [^162] | [A-MapReduce: Executing Wide Search via Agentic MapReduce](https://arxiv.org/abs/2602.01331) | 提出受MapReduce范式启发的多智能体框架A-MapReduce，将广度搜索重新构造为水平结构化检索问题，通过任务自适应分解、结构化结果聚合和经验记忆驱动的任务分配与重组，实现大规模检索目标的高效并行执行。 |
| [^163] | [SteerEval: Inference-time Interventions Strengthen Multilingual Generalization in Neural Summarization Metrics](https://arxiv.org/abs/2601.15809) | 该论文发现，通过推理时的激活干预将多语言神经摘要评估指标的内部表示引导向英语枢纽语言，可以普遍提升这些指标在多种语言上与人类判断的相关性。 |
| [^164] | [Towards robust long-context understanding of large language model via active recap learning](https://arxiv.org/abs/2601.13734) | 提出主动回顾学习（ARL）框架，通过持续预训练中的针对性序列构建与推理时的自主回溯总结，使大语言模型在段落间建立递归记忆机制，显著提升长上下文理解能力（RULER提升最高达129.7%）。 |
| [^165] | [PAIR-SAFE: A Paired-Agent Approach for Runtime Auditing and Refining AI-Mediated Mental Health Support](https://arxiv.org/abs/2601.12754) | 提出PAIR-SAFE双智能体框架，由基于临床验证MITI-4标准的评判者智能体实时审计并指导修正LLM生成的心理健康支持回复，显著提升了伙伴关系、协作性和整体关系质量。 |
| [^166] | [Dementia-R1: Reinforced Pretraining and Reasoning from Unstructured Clinical Notes for Real-World Dementia Prognosis](https://arxiv.org/abs/2601.03018) | 提出了Dementia-R1框架，通过冷启动强化学习策略先预训练模型预测可验证的临床指标，再基于非结构化临床笔记进行痴呆症的纵向预后预测，实现了最佳整体性能。 |
| [^167] | [EvalCards: A Framework for Standardized Evaluation Reporting](https://arxiv.org/abs/2511.21695) | 本文提出 EvalCards 框架，围绕可复现性、可访问性和可问责性三个核心维度规范 AI 模型评估报告，并提供简洁评估摘要的初步格式，以推动未来评估报告标准的建立。 |
| [^168] | [What MLLMs Learn about When they Learn about Multimodal Reasoning](https://arxiv.org/abs/2510.01719) | 该论文提出MathLens几何推理基准，将多模态推理性能分解为感知、推理和多模态组件，揭示了强化学习与文本SFT等常见训练策略会产生在总体准确率下不可见的系统性不同能力特征。 |
| [^169] | [A survey of diversity quantification in natural language processing: The why, what, where and how](https://arxiv.org/abs/2507.20858) | 该论文借鉴生态学与经济学的跨领域统一框架，调研了300多篇NLP多样性相关文献，提出了涵盖“为什么重要、衡量对象、衡量位置、衡量方法”四个视角的NLP多样性量化专用框架，以解决该领域度量标准碎片化和不一致的问题。 |

# 详细

[^1]: 指令的语言：体裁与生成式人工智能

    The Language of Instruction: Genre and Generative AI

    [https://arxiv.org/abs/2610.32982](https://arxiv.org/abs/2610.32982)

    本文创新性地提出，指令微调后的语言模型实现了一种新的“指令-会话体裁”，该体裁融合了人机历史对话的话语语言，并通过指令微调、系统提示词以及带有特殊标记和角色赋值的标记语言输入在应用中实例化。

    

    本文主张，经过指令微调的语言模型应被理解为实现了一种体裁，即本文所提出的“指令-会话体裁”。这一体裁是生成式模型与人类先前对话中话语语言的融合物，这些话语在这些大规模神经网络中被递归地运行处理。在应用中，该体裁通过以下方式得以实例化：指令微调、引入被称为系统提示词的前置文本，以及将输入形式化为标记语言——其特征是使用特殊标记，并在轮次对话结构中进行角色赋值。

    arXiv:2610.32982v1 Announce Type: new  Abstract: This article argues that instruction-fine-tuned language models should be understood as implementing a genre, what is proposed as the instructional-conversational genre. This genre is an amalgamation of discursive language from prior conversations between generative models and humans recursively run through these massive neural networks. It is instantiated in applications through instruction fine-tuning, the inclusion of a pretext known as a system prompt, and the formalization of inputs in mark-up languages characterized by the use of special tokens and role assignment in turntaking structures.
    
[^2]: 先测量再训练：类型化决策模型 vs. 经典基线 vs. 生成式大语言模型在个人多语言收件箱路由中的应用

    Measure Before You Train: Typed Decision Models vs. a Classical Baseline vs. a Generative LLM for Routing a Personal Multilingual Inbox

    [https://arxiv.org/abs/2610.32955](https://arxiv.org/abs/2610.32955)

    该论文提出自托管个人邮件代理Foyer及其测量协议，在真实多语言邮件数据上于训练前建立人工真实标注基准，并系统比较了类型化决策模型、经典机器学习基线与生成式大语言模型等九个系统的表现。

    

    个人邮件代理需要为每封邮件做出决策：这封邮件属于什么类型、是否紧急、是否需要起草回复、创建任务，还是直接忽略。然而这些系统的错误率很少被真正测量，因为存储的标签是由系统自身生成的。我们提出了Foyer——一个正在日常使用的自托管个人收件箱代理，以及一套测量协议（分层采样、冻结快照、按邮件线程分组的划分、预注册假设、在代理审查循环中由人工标注），该协议能够在任何训练开始之前获得真实标注数据。在739封由所有者亲自标注的真实意大利语/英语邮件上（其中231封为无偏样本，500封为对抗性采样），我们在按线程分组的交叉验证下比较了九个系统：生产环境的关键词规则、TF-IDF+逻辑回归经典基线、本地类型化决策模型、托管决策模型，以及托管生成式大语言模型。论文给出了四项主要结果，其中生产环境规则在邮件类型识别上仅获得28%的得分，而对照系统达到了58%。

    arXiv:2610.32955v1 Announce Type: new  Abstract: Personal email agents decide, for every message, what it is, whether it is urgent, whether to draft a reply, create a task, or ignore it. Their error rate is rarely measured: the stored labels were written by the system itself. We present Foyer, a self-hosted personal inbox agent in daily use, and a measurement protocol (stratified sampling, frozen snapshots, thread-grouped splits, pre-registered hypotheses, human labels in the agent's review loop) that yields ground truth before any training. On 739 real Italian/English emails labelled by the owner (231 unbiased, 500 adversarially sampled), we compare nine systems under thread-grouped cross-validation: the production keyword rules, a TF-IDF + logistic-regression baseline, local typed decision models (Tev1, Nimble), a hosted decision model (Nimble v4-max), and hosted generative LLMs (Qwen2.5-7B, Gemma 4 26B/31B). Four results. The production rules score 28% on message type against a 58% 
    
[^3]: Yalda-Embedding：一个具有竞争力的波斯语检索模型的开放开发

    Yalda-Embedding: Open Development of a Competitive Persian Retrieval Model

    [https://arxiv.org/abs/2610.32933](https://arxiv.org/abs/2610.32933)

    该论文提出Yalda Embedding，一个基于Qwen3-Embedding-0.6B适配、使用超百万查询-文档对训练的5.958亿参数波斯语检索模型，在FaMTEB的八个波斯语检索任务上超越所有同规模0.6B模型，并接近4B规模模型的性能。

    

    尽管多语言文本表示技术不断进步，但波斯语的有效密集检索仍然受到特定语言训练资源和紧凑嵌入模型可用性有限的制约。我们提出了Yalda Embedding，这是一个5.958亿参数的波斯语检索模型，基于Qwen3-Embedding-0.6B适配而来，使用了一百多万个查询-文档对进行训练。在FaMTEB的八个波斯语检索任务上，该模型在所评估的0.6B模型中取得了最高的宏平均值，得分0.5488，在每个任务上都优于Qwen3-Embedding-0.6B和F2LLM-v2-0.6B。它还接近Qwen3-Embedding-4B的性能，在四个任务上获胜，在一个任务上打平，宏平均值仅落后0.0075。一项增量消融研究考察了对比学习批量大小、优化稳定性以及从F2LLM-v2-8B进行的关系知识蒸馏。在单轮训练设置下，仅对比微调就能带来适度的平均改进……

    arXiv:2610.32933v1 Announce Type: cross  Abstract: Despite advances in multilingual text representations, effective dense retrieval for Persian remains constrained by the limited availability of language-specific training resources and compact embedding models. We present Yalda Embedding, a 595.8-million-parameter Persian retrieval model adapted from Qwen3-Embedding-0.6B using more than one million query-document pairs. On eight Persian retrieval tasks from FaMTEB, the model achieves the highest macro-average among the evaluated 0.6B models, scoring 0.5488 and outperforming Qwen3-Embedding-0.6B and F2LLM-v2-0.6B on every task. It also approaches Qwen3-Embedding-4B, winning on four tasks, tying on one, and trailing by 0.0075 in macro-average. An incremental ablation study examines contrastive batch size, optimization stabilization, and relational knowledge distillation from F2LLM-v2-8B. Under the one-epoch training setup, contrastive fine-tuning alone yields a modest mean improvement ov
    
[^4]: 面向混合自然语言与结构化符号语料库的自适应频率阈值字节对编码

    Adaptive Frequency Threshold Byte Pair Encoding for Mixed Natural Language and Structured Symbolic Corpora

    [https://arxiv.org/abs/2610.32921](https://arxiv.org/abs/2610.32921)

    本文提出AdaBPE，一种基于自适应频率阈值的频繁模式挖掘分词方法，通过动态调整合并策略，更好地处理跨越空白的多词表达和LaTeX等结构化符号内容。

    

    现代语言模型依赖子词分词来定义输入处理的离散单元，通常采用字节对编码（BPE）等方法，将空白字符视为固定的语义边界。这种设计存在固有局限，因为许多有意义的概念——包括多词表达以及诸如LaTeX等形式语言的结构化环境——经常跨越空白字符。尽管近期的超词分词方法试图通过放宽空白约束来解决这一问题，但它们通常依赖贪心的频率驱动合并方式，未能考虑词元长度或资源不足的领域特定语言的具体需求。在本文中，我们提出了自适应频率阈值字节对编码，这是一种利用自适应频繁模式挖掘方法的新型分词方法。AdaBPE具有双重动态特性：它基于……的长度自适应调整最小频率阈值（摘要在此处截断）

    arXiv:2610.32921v1 Announce Type: new  Abstract: Modern language models rely on subwording tokenization to define the discrete units of input processing, typically using methods like Byte Pair Encoding (BPE) that treat whitespace as a fixed semantic boundary. This design is inherently limited, as many meaningful concepts---including multiword expressions and the structured environments of formal languages like LaTeX---frequently span across whitespaces. While recent superword tokenization methods attempt to address this by relaxing whitespace constraints, they often rely on greedy frequency-driven merging that fails to account for token length or the specific needs of underrepresented domain-specific languages. In this paper, we introduce Adaptive Frequency Threshold Byte Pair Encoding (AdaBPE), a new tokenization method that leverages an adaptive frequent-pattern mining approach. AdaBPE features a twofold dynamicity: it adapts minimum frequency thresholds based on both the length of a
    
[^5]: “真”与“假”相加等于一吗？LLM智能体观点模拟中的互补一致性

    Do "True" and "False" Add Up to One? Complement Consistency in LLM-Agent Opinion Simulations

    [https://arxiv.org/abs/2610.32874](https://arxiv.org/abs/2610.32874)

    该研究首次系统揭示了LLM智能体在观点模拟中违反互补一致性——对同一命题及其否定所报告的概率之和超过1（如Qwen3-4B平均达1.18），且该偏差远大于扰动噪声两到十倍，意味着此类模拟结果可能存在系统性扭曲。

    

    在LLM智能体观点模拟中，每个智能体在看到自己之前的概率以及邻居所报告的概率后，报告某条声明为真的概率，该数字将成为它在下一轮中的状态。如果这个数字描述的是智能体的立场而非它所看到的句子本身，那么将该声明改写为其否定形式，并把每个概率p替换为1-p，理应得到互补的答案，即两次报告之和应为1。我们使用三个小型开源模型（Qwen3-4B、Llama-3.2-3B和Gemma-3-4B）对此进行了测试。在从一次运行中的网络模拟提取的2,304个智能体状态上，Qwen的成对报告之和平均为1.18；而在来自AVeriTeC的48条真实事实核查声明上，其结果同样超过1。一个包含24个双结果世界（每个世界恰好有两种可能结局）的受控基准测试表明，三个模型都存在同样的失效现象。这一偏差比相同均值扰动所引起的变化大两到十倍。

    arXiv:2610.32874v1 Announce Type: new  Abstract: In LLM-agent opinion simulations, each agent reports the probability that a claim is true after seeing its own previous probability and the probabilities reported by its neighbors, and that number becomes its state in the next round. If the number describes the agent's position rather than the sentence it was shown, then rewriting the claim as its complement and replacing every probability p with 1 - p should return the complementary answer, so the two reports should sum to one. We test this with three small open models: Qwen3-4B, Llama-3.2-3B and Gemma-3-4B. On 2,304 agent states taken from a running network simulation, Qwen's paired reports sum to 1.18 on average, and on 48 real fact-checking claims from AVeriTeC they again exceed one. A controlled benchmark of 24 worlds, each with exactly two possible outcomes, shows the same failure in all three models. The discrepancy is two to ten times larger than the change produced by a same-mea
    
[^6]: J-Comm：基于雅可比引导的大语言模型智能体间潜在通信

    J-Comm: Jacobian-Guided Latent Communication between LLM Agents

    [https://arxiv.org/abs/2610.32871](https://arxiv.org/abs/2610.32871)

    提出J-Comm框架，结合免训练的GEAR几何一致性方法与H+J（隐状态与雅可比）双通道通信接口，实现大语言模型智能体之间基于雅可比引导的潜在通信。

    

    潜在通信既需要一个有效的连续推理过程，也需要一个接口使外部模型能够利用其中间计算结果。我们提出了J-Comm框架，包含两个核心组件。首先，一种免训练的潜在推理几何一致性方法（GEAR），在现有模型嵌入之上对候选token进行重加权，在每一步中单调地优化嵌入空间中候选token之间的几何共识。其次，我们引入了H+J（隐状态与雅可比）通信接口，为接收方提供发送方推理过程的两个互补视角。H通道提供源自反馈分布的连续输入，而J通道则以词汇索引的雅可比透镜坐标传递中间状态线索。直观地说，H为接收方提供推理所需的输入，而J则在其计算过程中提供额外的引导。接收方重建这些

    arXiv:2610.32871v1 Announce Type: new  Abstract: Latent communication requires both an effective continuous reasoning process and an interface through which an external model can leverage its intermediate computation. We present J-Comm, a framework comprising two core components. First, a training-free Geometric Agreement for Latent Reasoning (GEAR) reweights token candidates over existing model embeddings, which monotonically optimizes geometric consensus among candidate tokens in embedding space at each step. Second, we introduce an H+J (Hidden-and-Jacobian) communication interface that gives the Receiver two complementary views of the Sender's reasoning. The H channel supplies continuous inputs derived from feedback distributions, while the J channel conveys intermediate-state cues in vocabulary-indexed Jacobian-lens coordinates. Intuitively, H provides inputs for the Receiver to reason from, while J offers additional guidance within its computation. The Receiver reconstructs these 
    
[^7]: 基于TopKLoRA的可解释稀疏微调实现忠实的电路发现

    Interpretable sparse fine-tuning with TopKLoRA enables faithful circuit discovery

    [https://arxiv.org/abs/2610.32866](https://arxiv.org/abs/2610.32866)

    本文提出了设计上即可解释的稀疏微调方法TopKLoRA，通过在每个token上仅激活k个最大的适配器潜在变量、纯粹以下游损失训练，使其潜在变量在功能上具有因果性，从而能够忠实地发现休眠代理后门等模型内部电路。

    

    摘要（arXiv:2610.32866v1，公告类型：cross）：在来源未知的数据集上进行微调时，如果数据中包含被投毒的样本，可能会在被训练的模型中植入后门。后门难以定位和移除，并且可以在安全训练之后持续存在。虽然参数高效微调方法能够捕捉训练过程中引入的全部参数变化，但它们不易于解释。我们提出了TopKLoRA的完整方法，该方法扩展了Masiak等人（2025）的初步提案，这是一种设计上即可解释的LoRA变体，其中在每个token上只有$k$个最大的适配器潜在变量处于激活状态。与稀疏自编码器（SAE）不同，TopKLoRA完全基于下游损失进行训练，其中优化压力会促使模型学习在功能上具有因果作用的潜在变量。我们在一个休眠代理（sleeper-agent）后门案例研究中展示了这种稀疏LoRA适配器：我们通过在投毒的指令遵循数据集上微调gemma-2-2b、Qwen2.5-1.5B和Qwen2.5-7B来创建模型生物体（model organisms）。对于每个生物体，我们识别并……（摘要原文在此处截断）

    arXiv:2610.32866v1 Announce Type: cross  Abstract: Fine-tuning on datasets of unknown origin can create a backdoor in the trained model when the data includes poisoned examples. Difficult to localise and remove, backdoors can persist through safety training. While parameter-efficient fine-tuning methods capture the full parameter changes introduced during training, they are not easily interpretable. We present the full method of TopKLoRA, extending the preliminary proposal of Masiak et al. (2025), an interpretable-by-design LoRA variant in which only the $k$ largest adapter latents are active at each token. Unlike SAEs, TopKLoRA is trained purely on downstream loss, where optimisation pressure incentivises learning functionally causal latents. We demonstrate this sparse LoRA adapter in a sleeper-agent backdoor case study where we create model organisms by fine-tuning gemma-2-2b, Qwen2.5-1.5B, and Qwen2.5-7B on a poisoned instruction-following dataset. For each organism, we identify and
    
[^8]: 与全体民众沟通：紧急警报与预警信息的机器翻译

    Communicating with the Entire Population: Machine Translation of Emergency Alerts and Warnings

    [https://arxiv.org/abs/2610.32842](https://arxiv.org/abs/2610.32842)

    本文通过全国性应急管理者调查了解其真实需求，并据此评估商业机器翻译工具在紧急警报翻译中的表现，为应急响应中如何及何时安全使用翻译工具提供了实用指导。

    

    在危机期间，与受影响的全体民众进行沟通至关重要，其中包括那些不讲该地区主流语言的人群。商业翻译工具为扩大信息获取范围提供了一个诱人的解决方案，并且在没有其他资源可用时可以提供一条生命线。然而，应急响应管理者往往对这些自动化输出的质量和充分性一无所知。在本文中，我们对应急管理者进行了全国性调查，以了解他们当前的工作流程、观点和运营挑战。随后，我们根据这些需求评估了商业翻译工具，阐明了它们的实用性，并就如何在紧急响应中安全使用这些工具以及何时使用提供了实用指导。

    arXiv:2610.32842v1 Announce Type: new  Abstract: It is essential that we communicate with the entirety of an affected population during a crisis, including those who do not speak a region's dominant language(s). Commercial Translation tools offer a tempting solution to broaden access and can provide a lifeline when no other resources are available. However, Emergency response managers often operate in the dark regarding the quality and adequacy of these automated outputs. In this paper, we conduct a nationwide survey of emergency managers to understand their current workflows, perspectives, and operational challenges. We then evaluate commercial translation tools against these needs, shedding light on their utility and offering practical guidance on how and when these tools can be safely used in emergency response.
    
[^9]: SPOOK：视觉编码器中流形外知识的稀疏像素优化

    SPOOK: Sparse Pixel Optimization for Off-Manifold Knowledge in Vision Encoders

    [https://arxiv.org/abs/2610.32835](https://arxiv.org/abs/2610.32835)

    该研究提出SPOOK方法，证明仅通过黑盒爬山优化对空白画布进行稀疏像素编辑，就能在OpenCLIP和DINOv3等视觉编码器的表示空间中成功检索到目标语义，揭示了远离自然图像流形处嵌入相似性与人类可见语义的脱节。

    

    现代视觉编码器被广泛用作语义表示，但在远离自然图像流形的区域，嵌入相似性与人类可见语义之间的关系可能变得不清晰。我们在黑盒设置下探究这一区域，通过对空白画布应用稀疏像素编辑，并采用一种简单的像素级爬山过程。在ImageNet和Flickr30k数据集上，我们评估了这些编辑过的空白画布能否在多模态对比编码器和自监督视觉编码器中检索到选定的目标。在OpenCLIP上，朝类别提示优化的输入实现了强大的top-1和top-5目标检索性能；在DINOv3上，朝类别原型优化的输入实现了高目标对齐和接近顶部的原型排名。这些结果表明，在原本空白、视觉上无语义的输入上进行稀疏编辑，即可进入语言对齐和自监督视觉表示空间中的目标语义盆地。

    arXiv:2610.32835v1 Announce Type: cross  Abstract: Modern vision encoders are widely used as semantic representations, but the relationship between embedding similarity and human-visible semantics can become unclear far from the natural-image manifold. We probe this regime in a black-box setting by applying sparse pixel edits to blank canvases with a simple pixel-level hill-climbing procedure. Across ImageNet and Flickr30k, we evaluate whether these edited blank canvases can retrieve chosen targets in multimodal contrastive encoders and self-supervised visual encoders. On OpenCLIP, inputs optimized toward class prompts achieve strong top-1 and top-5 targeted retrieval performance; on DINOv3, inputs optimized toward class prototypes achieve high target alignment and near-top prototype ranks. These results show that sparse edits on otherwise blank, visually non-semantic inputs can enter target semantic basins in both language-aligned and self-supervised visual representation spaces. Our 
    
[^10]: KrishokTech：一个具有显式授权控制的孟加拉语农业咨询系统

    KrishokTech: A Bengali Agricultural Advisory System with Explicit Authorization Controls

    [https://arxiv.org/abs/2610.32817](https://arxiv.org/abs/2610.32817)

    KRISHOKTECH 是一个孟加拉语农业咨询系统，其核心创新在于围绕生成过程构建显式授权边界——通过高风险请求拦截、作物确认机制和剂量声明验证等控制措施，并结合可独立更新的外部证据索引与本地检索，在低网络连接环境下安全可靠地提供农业建议。

    

    对于农业咨询而言，生成答案只是问题的一部分：系统还必须控制不安全的请求、在有限网络连接条件下工作，并适应不断变化的指导建议。KRISHOKTECH 是一个围绕生成过程构建了显式授权边界的孟加拉语农业咨询系统。在检索之前，规范化处理和基于规则的准入检查会将政策涵盖的高风险请求转交处理，并在治疗意图不明确时要求用户提供作物信息。随后，系统将检索绑定到单一作物，并在文本与图像显示的作物不一致时要求确认。生成之后，一个范围严格的验证器会删除检索到的 BARI/BRRI 证据所不支持的剂量声明。农业指导内容保存在一个外部证据索引中，该索引无需重新训练生成器即可更新，同时浏览器端视觉处理和本地检索降低了对网络连接的要求。界面将这些决策以 ASK 等形式直观呈现（原文在此处截断）。

    arXiv:2610.32817v1 Announce Type: new  Abstract: For agricultural advisory, generating an answer is only part of the problem: the system must also control unsafe requests, work with limited connectivity, and accommodate changing guidance. KRISHOKTECH is a Bengali agricultural advisory system built around an explicit authorization boundary around generation. Before retrieval, normalization and rule-based admission checks refer policy-covered high-risk requests and ask for a crop when treatment intent is underspecified. The system then binds retrieval to a single crop and asks for confirmation when the text and image indicate different crops. After generation, a narrow verifier removes dosage claims not supported by retrieved BARI/BRRI evidence. Agricultural guidance is kept in an external evidence index that can be updated without retraining the generator, while browser-side vision and local retrieval reduce connectivity requirements. The interface makes these decisions visible as ASK, 
    
[^11]: Frontier-OPD：发现智能体能力的边界并教会智能体去超越

    Frontier-OPD: Discovering the Edge of Agent Capability and Teaching Agents to Go Beyond

    [https://arxiv.org/abs/2610.32802](https://arxiv.org/abs/2610.32802)

    Frontier-OPD 通过“学习边界”概念——即经简短教师指导后学生仍能自主有效延续的挑战性决策情境——并借助共享状态轨迹图和师生分歧的轨迹平衡聚合来定位干预点，从而发现多轮智能体的能力边界并教会其超越。

    

    在策略蒸馏使教师反馈与学生当前行为保持一致，然而为多轮智能体获取有用的监督信号，需要同时确定何处需要指导，以及学生在此之后能否维持交互。我们提出 Frontier-OPD，它通过学生的学习边界来解决这一问题：学习边界指那些仍然具有挑战性、但在接受简短教师指导后能够进行有用的自主延续的决策情境。为了将该边界可操作化，Frontier-OPD 将在策略经验组织成一个共享状态轨迹图，其中通过轨迹平衡的方式聚合师生分歧，识别出不同的干预候选点，同时保留回放所需的历史信息。随后，通过将教师桥接段及随后的学生延续段上的一致性，与原始轨迹中等长的片段进行比较，来评估简短教师桥接段，为指导价值提供行为代理指标。

    arXiv:2610.32802v1 Announce Type: new  Abstract: On-policy distillation aligns teacher feedback with a student's current behavior, yet acquiring useful supervision for multi-turn agents requires identifying both where guidance is needed and whether the student can sustain the interaction afterward. We introduce Frontier-OPD, which addresses this problem through the student's learning frontier: decision situations that remain challenging but admit useful autonomous continuation after brief teacher guidance. To operationalize this frontier, Frontier-OPD organizes on-policy experience into a shared-state trajectory graph, where trajectory-balanced aggregation of teacher-student disagreement identifies distinct intervention candidates while preserving the histories needed for replay. Short teacher bridges are then assessed by comparing agreement over the bridge and the ensuing student continuation with an equally long segment of the original trajectory, providing a behavioral proxy for gui
    
[^12]: 学习优化器调度中的乘法与加法状态耦合

    Multiplicative vs. Additive State Coupling in Learned Optimizer Schedules

    [https://arxiv.org/abs/2610.32801](https://arxiv.org/abs/2610.32801)

    本研究通过反向传播学习MAX-CUT求解器的调度，发现双线性（乘法）状态耦合架构在分布外泛化上相比加法耦合具有显著且持久的优势，从而将关键架构区别定位于求解器状态进入调度的方式，而非状态自适应性本身。

    

    我们通过在展开的经典求解器上进行反向传播，来学习MAX-CUT连续惩罚松弛问题的调度。状态盲调度能够泛化到比训练所用实例大得多的实例。状态自适应的双线性调度同样能实现分布外泛化，在15个随机种子上优于其未训练的基线（t=4.1）。相比之下，具有相同输入和输出头的加法消融模型在分布内与双线性模型在统计上不可区分，但在分布外的表现却比其自身基线更差（t=-11.7）。这种分离在无正则化的情况下即会出现，并在30轮权重衰减扫描中持续存在：双线性模型的优势在每个测试值下都保持显著，而加法模型始终无法恢复。在该设置下，这些结果将求解器状态进入调度的方式，而非单纯的状态自适应性，隔离为与（分布外泛化表现）相关的关键架构区别。

    arXiv:2610.32801v1 Announce Type: cross  Abstract: We learn schedules for a continuous penalty relaxation of MAX-CUT by backpropagating through an unrolled classical solver. A state-blind schedule generalizes to instances substantially larger than those used for training. A state-adaptive bilinear schedule also generalizes out of distribution, outperforming its untrained baseline across 15 seeds ($t=4.1$). By contrast, an additive ablation with identical inputs and output heads is statistically indistinguishable from the bilinear model in distribution but performs worse than its own baseline out of distribution ($t=-11.7$). This separation appears without regularization and persists across a $30$ round weight-decay sweep: the bilinear advantage remains significant at every tested value, whereas the additive model never recovers. The results in this setting isolate how solver state enters the schedule, rather than state adaptivity alone, as the architectural distinction associated with 
    
[^13]: ParaTaxe：一套用于处理和发布多平行语料库的工具

    ParaTaxe: a Suite of Tools for Processing and Publishing Multi-parallel Corpora

    [https://arxiv.org/abs/2610.32796](https://arxiv.org/abs/2610.32796)

    该论文提出了一套开源工具套件（包括基于多语言嵌入的AIlign对齐器、Python处理流程以及用于管理和浏览平行语料库的网页界面），旨在解决学术平行语料库因缺乏维护和资金而难以在线持续访问的问题。

    

    在过去的十五年中，用于语料库平行化和对齐的数字工具在数字人文领域取得了显著发展。为了比较文本的不同译本或版本，学界已创建了多个项目和软件工具。然而，尽管有这些技术进步，许多在学术环境中产生的平行语料库由于缺乏维护或资金支持，很快就无法在线访问。为了解决这一缺口，我们展示了在 ACR Grimm Tradalign 和 ParaTaxe 项目中开发的一套开源工具。该套件包括基于多语言嵌入的 AIlign 对齐器、用于转换和对齐语料库的 Python 处理流程、一个名为 ParaTaxe 的用于管理平行语料库集合的网页界面，以及一个基于 TEI-Publisher 构建的用于浏览已对齐语料库的网页界面。

    arXiv:2610.32796v1 Announce Type: new  Abstract: Over the past fifteen years, digital tools for parallelizing and aligning textual corpora have developed significantly within the field of digital humanities. Several projects and software tools have been created to compare different translations or versions of texts. Despite these technical advances, many parallel corpora produced in academic contexts quickly become inaccessible online due to the lack of maintenance or funding. To address this gap, we present a suite of open-source tools developed within the ACR Grimm Tradalign and ParaTaxe projects. This suite includes the AIlign aligner, based on multilingual embeddings, a Python processing pipeline for converting and aligning corpora, a web interface called ParaTaxe for managing collections of parallel corpora, and a web interface for exploring aligned corpora built with TEI-Publisher.
    
[^14]: 每条边都需要引用：利用语言模型智能体构建以引用为依据的医疗软件采用知识图谱

    Every Edge Needs a Quote: Building a Quote-Grounded Knowledge Graph of Healthcare Software Adoption with Language-Model Agents

    [https://arxiv.org/abs/2610.32781](https://arxiv.org/abs/2610.32781)

    该论文提出一个从开放网络构建医疗软件客户与集成关系知识图谱的流水线，要求每条边都必须有源页面的逐字引用支持，从而确保语言模型智能体提取的信息可验证、可追溯。

    

    医院需要定期采购临床医生和患者赖以使用的软件。由于这些采购成本高昂、难以逆转，且可能影响患者护理，采购方需要寻找证据来确认哪些同行机构已经在使用某款产品，以及该产品与哪些系统相连接。这些证据是公开的，但分散在供应商网站、新闻稿和第三方页面中。语言模型智能体现在能够以低成本收集这些信息，但提取出的声明的正确性难以核实，而错误列出的客户信息可能会误导采购方并损害机构声誉。我们提出了一个从开放网络构建客户关系与集成关系知识图谱的流水线，遵循一条规则：每条边都必须由来自其源页面的逐字引用来支持。页面来源于对每个供应商网站的确定性爬取，以及一个在预算限制内搜索网络其余部分的ReAct智能体。一个宽松的提取器提出带有引用的关系，随后……

    arXiv:2610.32781v1 Announce Type: cross  Abstract: Hospitals regularly buy software that clinicians and patients depend on. Because these purchases are costly, hard to reverse and can affect patient care, buyers look for evidence of which peer institutions already use a product and which systems it connects to. This evidence is public but scattered across vendor websites, press releases and third-party pages. Language-model agents can now collect it at low cost, but the correctness of the extracted claims is difficult to establish, and a wrongly listed customer can mislead buyers and misrepresent an organisation. We present a pipeline that builds a knowledge graph of customer and integration relations from the open web under one rule: every edge must be supported by a verbatim quote from its source page. Pages come from a deterministic crawl of each vendor's site and a budgeted ReAct agent that searches the rest of the web. A permissive extractor proposes relations with quotes, and a l
    
[^15]: 在线策略自训练：大语言模型的无数据自我改进

    On-Policy Self-Training: Data-Free Self-Improvement in Large Language Models

    [https://arxiv.org/abs/2610.32770](https://arxiv.org/abs/2610.32770)

    该论文提出了一种无需任何数据的在线策略自训练框架，大语言模型仅需粗糙的高层主题描述即可自生成问题-答案对和特权推理演示进行自我改进，在多个领域提升能力的同时保留原有领域的性能。

    

    在线策略自蒸馏使得能够在不依赖更强外部教师的情况下对基础模型进行后训练。然而，现有方法依赖于真实的问题-答案对、人工策划的问题集或其他监督信号。在这项工作中，我们将在线策略自蒸馏重新表述为一个自训练框架，由自生成的问题-答案对和特权推理演示驱动。我们将监督信号推向极限：仅由粗糙的高层级主题描述引导，模型在无需访问任何人工策划的提示或答案的情况下，自行合成查询课程和特权推理轨迹。我们发现，在完全合成的特权信息上进行自蒸馏，能够使模型在多个领域获得改进，同时保留其在未进行自训练的领域上的能力。我们展示了该框架在文本和多模态领域的通用性，并取得了一致的（效果）。

    arXiv:2610.32770v1 Announce Type: cross  Abstract: On-policy self-distillation enables post-training foundation models without relying on stronger external teachers. However, existing approaches rely on ground-truth question-answer pairs, human-curated problem sets, or other supervised signals. In this work, we reformulate on-policy self-distillation as a self-training framework, driven by self-generated question-answer pairs and privileged reasoning demonstrations. We push supervision to the limit: guided solely by coarse, high-level topic descriptions, the model synthesizes its own curriculum of queries and privileged reasoning traces without access to any human-curated prompts or answers. We find that self-distillation on fully synthetic privileged information enables a model to improve across multiple domains while retaining its capabilities on domains it is not self-trained on. We demonstrate the generality of this framework across text and multimodal domains, achieving consistent
    
[^16]: ActionRAG：面向大语言模型智能体长期记忆的证据精炼图扩散方法

    ActionRAG: Evidence-Refining Graph Diffusion for Long-Term Memory in LLM Agents

    [https://arxiv.org/abs/2610.32752](https://arxiv.org/abs/2610.32752)

    ActionRAG提出了一种图记忆架构，通过谨慎的分层实体消解索引与查询时基于局部路径结构的图扩散回溯评估，帮助LLM智能体从长期交互历史中恢复精确且相互一致的关系型证据。

    

    LLM智能体需要能够从不断增长的交互历史中恢复精确且相互一致的证据的长期记忆，尤其是当答案依赖于事实之间的关系而非孤立的语义匹配时。当前的记忆流水线在两个阶段仍然脆弱：索引错误可能碎片化重复的实体或将不同实体错误合并，而在线检索常常将初始相似度分数视为最终结果，即使周围的图结构提供了修正这些分数的证据。我们提出ActionRAG，一种将谨慎索引与查询时图回溯评估相结合的图记忆架构。其索引流水线对类型标签进行协调并应用分层实体消解。给定查询后，双通道的实体与三元组检索形成初始证据集，并诱导出一个有界的局部图。查询条件下的接收者归一化扩散利用局部路径结构来精炼三元组分数。种子保护、枢纽惩罚通量……（摘要在此处截断）

    arXiv:2610.32752v1 Announce Type: new  Abstract: LLM agents need long-term memory that can recover precise and mutually consistent evidence from growing interaction histories, particularly when answers depend on relations among facts rather than isolated semantic matches. Current memory pipelines remain vulnerable at two stages: indexing errors can fragment repeated entities or merge distinct ones, and online retrieval often treats initial similarity scores as final even when the surrounding graph provides evidence for revising them. We introduce ActionRAG, a graph memory architecture that couples cautious indexing with query-time graph retroevaluation. Its indexing pipeline reconciles type labels and applies tiered entity resolution. Given a query, dual-channel entity and triple retrieval forms an initial evidence set and induces a bounded local graph. Query-conditioned receiver-normalized diffusion refines triple scores using local path structure. Seed protection, hub-penalized flux,
    
[^17]: 蝴蝶缺陷：追踪大语言模型智能体网络中的合作行为

    The Butterfly Defects: Tracing Cooperation in LLM Agent Networks

    [https://arxiv.org/abs/2610.32726](https://arxiv.org/abs/2610.32726)

    该论文通过迭代囚徒困境实验发现，大语言模型智能体网络中的合作行为高度依赖模型组成与微小的语言变化（如不确定性提示、动作标签顺序），从而挑战了合作是稳定模型级属性的观念。

    

    随着大语言模型（LLM）越来越多地以相互连接的智能体形式运作，一个根本性问题浮现出来：个体层面的合作是否会转化为集体层面的合作？我们通过迭代囚徒困境（IPD）对这一问题进行研究，从双主体交互扩展到由开放权重LLM构成的同质与异质网络。我们的发现表明，合作行为对模型组成以及看似微小的语言变化高度敏感。与完全相同的同伴交互时表现合作的模型，在混合群体中可能转向背叛。一个简单的不确定性提示可以增加或减少合作、重新分配成本，并影响数跳之外未直接暴露的智能体。甚至反转动作标签的顺序也会显著改变行为，而不确定性框架有时会进一步放大这种敏感性。这些发现共同挑战了将合作视为稳定的模型级属性的观念，揭示了策略性行为的动态特性。

    arXiv:2610.32726v1 Announce Type: cross  Abstract: As large language models (LLMs) increasingly operate as interconnected agents, a fundamental question emerges: does individual cooperation translate into collective cooperation? We investigate this question through the Iterated Prisoner's Dilemma (IPD), moving from dyadic interactions to homogeneous and heterogeneous networks of open-weight LLMs. Our findings reveal that cooperation is highly sensitive to model composition and seemingly minor linguistic changes. Models that cooperate with identical peers can shift toward defection in mixed populations. A simple uncertainty cue can increase or decrease cooperation, redistribute costs, and affect unexposed agents several hops away. Even reversing action-label order can dramatically alter behavior, with uncertainty framing sometimes amplifying this sensitivity. Together, these findings challenge the notion of cooperation as a stable model-level property, revealing how strategic behavior d
    
[^18]: 本体引导下基于结构化元数据与气候科学出版物的知识图谱构建

    Ontology-Guided Knowledge Graph Construction from Structured Metadata and Climate Science Publications

    [https://arxiv.org/abs/2610.32725](https://arxiv.org/abs/2610.32725)

    本文提出一种代理式、本体引导的知识图谱构建框架，利用大语言模型从气候科学文献中抽取带文本证据的三元组，并通过多阶段验证确保语义一致性与可追溯性，抽取质量接近人类标注者水平。

    

    科学文献的快速扩张远远超出了人工整理的速度，这为从科学文本中提取结构化知识造成了根本性瓶颈。为解决这一问题，我们提出了一种代理式、本体引导的知识图谱（KG）构建框架。该框架利用专门的代理分别执行文档理解、信息抽取、本体一致性检查和实体链接任务。首先，大语言模型（LLM）从研究论文中以三元组的形式抽取事实知识，并附带明确的文本证据。随后，三元组经过多阶段验证流程，使其锚定于本体之上。通过强制执行模式约束并要求提供文本片段形式的证据，我们确保了数据在语义上的一致性和可追溯性。在一项针对1000个抽取三元组的一致性研究中，Gemini-2.5-flash 相对于三位人类标注者取得了 0.640–0.700 的 Cohen's κ 分数，达到（原文摘要在此处截断）

    arXiv:2610.32725v1 Announce Type: cross  Abstract: The rapid expansion of scientific literature far outpaces the speed of manual curation, creating a fundamental bottleneck in extracting structured knowledge from scientific texts. To address this, we introduce an agentic, ontology-guided framework for knowledge graph (KG) construction. Our framework utilizes specialized agents for document understanding, information extraction, ontology conformance, and entity linking. First, an LLM extracts factual knowledge in the form of triples with explicit textual evidence from research papers. Then the triplets undergo a multistage verification process to ground the triplets in the ontology. By enforcing schema constraints and requiring evidence in the form of text fragments, we ensure the data is both semantically consistent and traceable. In an agreement study over 1,000 extracted triplets, Gemini-2.5-flash achieves Cohen's $\kappa$ scores of 0.640--0.700 against three human annotators, matchi
    
[^19]: CounselPlex：一种压力源感知的全双工心理咨询模型

    CounselPlex: A Stressor-Aware Full-Duplex Counseling Model

    [https://arxiv.org/abs/2610.32702](https://arxiv.org/abs/2610.32702)

    CounselPlex通过在求助者沉默时更新压力源状态并在聆听时注入该状态，克服了全双工语音模型的“压力源盲区”问题，同时引入RoPE位置偏移增强（RPSA）以提升长对话深度下的性能，从而实现更高质量的心理咨询。

    

    全双工语音模型可以同时聆听和说话，从而实现自然、低延迟的交互。然而在心理咨询场景中，这类模型往往只关注最近的话语，而丢失了求助者潜在的问题。我们将这种失败称为“压力源盲区”。我们提出了CounselPlex，一个压力源感知的全双工框架，它采用两种操作：CounselPlex在求助者沉默期间更新一个简短的压力源状态，并在聆听过程中将该状态注入到相应位置。我们还引入了RoPE位置偏移增强（RPSA），该方法在训练期间扩展了位置覆盖范围，并在我们的评估中提升了较长对话轮次深度下的性能。基于LLM评判的评估显示，与强基线相比，CounselPlex在心理咨询质量和压力源锚定方面均有提升。

    arXiv:2610.32702v1 Announce Type: new  Abstract: Full-duplex speech models can listen and speak at the same time, enabling natural, low-latency interaction. In counseling, however, these models often focus on the latest utterance and lose track of the seeker's underlying problem. We call this failure Stressor Blindness. We propose CounselPlex, a stressor-aware full-duplex framework that uses two operations. CounselPlex updates a short stressor state during seeker silence and injects this state into  positions while listening. We also introduce RoPE Position Shift Augmentation (RPSA), which broadens positional coverage during training and improves performance at longer turn depths in our evaluation. LLM-judge evaluations show improved counseling quality and stressor grounding over strong baselines.
    
[^20]: 自然语言自编码器：识别而非解释

    Natural Language Autoencoders Identify Without Explaining

    [https://arxiv.org/abs/2610.32691](https://arxiv.org/abs/2610.32691)

    本文指出自然语言自编码器（NLA）的往返重建成功标准只能证明解释性文本包含足以识别激活的信息，并不能保证该解释对激活实际功能的描述是忠实的，这一问题与思维链忠实性问题结构相同。

    

    自然语言自编码器（NLA）以可读文本的形式表达模型的内部激活。言语化器将激活转换为英文解释；重建器随后将该解释转换回激活，如果这一往返过程能够恢复原始激活，则该解释被认为是成功的。我们主张这一标准仅评估了信息是否充分，而未涉及描述的忠实性。一句话可以包含足以识别某个激活的充分信息，即使它错误地解释了该激活的功能：在重建的场景中，首要目标是准确识别激活，而非描述它们。这一问题与思维链忠实性问题具有相同的结构——在该问题中，经过训练以伴随某一过程的言语输出被当作对该过程的报告，并且它类似于……（原文在此处截断）

    arXiv:2610.32691v1 Announce Type: new  Abstract: Natural Language Autoencoders (NLAs) express a model's internal activations in the form of readable text. The verbalizer turns an activation into an English explanation; the reconstructor then converts that explanation back into an activation, and the explanation is considered successful if the round trip brings back the original activation. We claim that this criterion evaluates whether the information is sufficient and does not address descriptive faithfulness. A sentence can contain sufficient information to make it possible to identify an activation even if it wrongly explains what the activation does: in the case of reconstruction, the primary aim is to accurately identify the activations and not to describe them. This issue has the same structure as the chain-of-thought faithfulness problem, in which verbal output that has been trained to go with a process is taken as a report on that process, and it parallels the dissociation betw
    
[^21]: 何时更多的搜索才有用？机器翻译质量感知解码中搜索效用的实证研究

    When is more search useful? An Empirical Study of Search Utility in Quality-Aware Decoding for Machine Translation

    [https://arxiv.org/abs/2610.32679](https://arxiv.org/abs/2610.32679)

    该论文提出“搜索效用”的三个量化指标（总扩展增益、边际扩展增益、采样饱和点），实证表明机器翻译中不同句子从额外候选中获益的程度差异显著，从而为自适应分配搜索预算提供了依据。

    

    质量感知解码通过测试时扩展来提升机器翻译，即生成多个候选译文，再由质量感知解码器选出最终输出。通常做法是对每个源句子都花费相同的候选预算N，假定所有输入需要同等的搜索量。我们证明这一假设并不成立，且额外候选带来的质量提升在不同句子之间存在差异。我们通过三个量来研究搜索效用：总扩展增益，即一个句子能从更大的候选池中获得多少质量提升；边际扩展增益，即额外生成一批候选所期望带来的增益；以及采样饱和点，即实现大部分总扩展增益所需的候选预算。我们考察了用于预测这三个量的两类信息来源：在候选集扩展之前即可获得的信号，以及从候选集计算得到的统计量。

    arXiv:2610.32679v1 Announce Type: new  Abstract: Quality-Aware Decoding (QAD) improves machine translation through test-time scaling, generating multiple candidate translations and selecting the final output with a quality-aware decoder. Usually the same candidate budget N is spent on every source sentence, assuming that all inputs require the same amount of search. We show that this assumption does not hold and that the quality gain obtained from additional candidates varies across sentences. We study search utility through three quantities: total scaling gain, how much quality a sentence can gain from a much larger candidate pool; marginal scaling gain, the expected gain from generating one additional batch of candidates; and the sampling saturation point, the candidate budget required to realize most of the total scaling gain. We investigate two sources of information for predicting these quantities: signals available before candidate-set expansion, and statistics computed from the 
    
[^22]: 面向因子化换能器语音识别纯文本领域自适应的残差专家路由

    Residual Expert Routing for Text-Only Domain Adaptation of Factorized Transducer ASR

    [https://arxiv.org/abs/2610.32637](https://arxiv.org/abs/2610.32637)

    提出保守残差路由方法，利用轻量门控在每个前缀处动态决定是否接纳文本自适应预测器的残差修正，从而在保留目标领域纯文本自适应收益的同时，使源领域性能完全不受损害。

    

    在目标领域文本上微调因子化换能器的非空白预测器，可以提升该领域的识别效果，但会损害源领域的性能。基础预测器与自适应预测器之间的单一插值权重无法解决这一问题：自适应预测器在某些前缀上有帮助，而在另一些前缀上有害，这种情况常常发生在同一句话内。我们提出保守残差路由方法：将自适应预测器作为对基础预测器的残差修正，并由一个1.6万参数的门控在每个前缀处决定接纳多少残差修正。关闭门控即可完全复现基础模型。在一个40小时的会话语料库上，路由方法在两个留出领域上基本保留了纯文本自适应的全部收益（词错误率分别从19.89降至16.74、从20.91降至16.21），同时源领域词错误率保持不变（12.90到12.93）；而单独使用自适应预测器会造成1.08的词错误率损失。路由方法无需在目标领域音频上进行针对各领域的调优。分析表明，门控的行为并非简单的领域级检测器……

    arXiv:2610.32637v1 Announce Type: new  Abstract: Fine-tuning the non-blank predictor of a factorized transducer on target-domain text improves recognition on that domain but degrades performance on the source domain. A single interpolation weight between base and adapted predictors cannot resolve this: the adapted predictor helps at some prefixes and hurts at others, often within one utterance. We propose conservative residual routing: the adapted predictor becomes a residual correction to the base, and a 16k-parameter gate decides at each prefix how much of it to admit. Closing the gate reproduces the base exactly. On a 40-hour conversational corpus, routing retains essentially all of the text-only gain on two held-out domains (19.89 to 16.74, 20.91 to 16.21 WER) while leaving source-domain WER unchanged (12.90 to 12.93); the adapted predictor alone costs 1.08 WER. Routing needs no per-domain tuning on target-domain audio. Analysis reveals the gate does not behave as a simple domain-l
    
[^23]: PHYRD：一个用于复现和拓展理论物理研究的动态基准

    PHYRD: A Living Benchmark for Reproducing and Extending Theoretical Physics Research

    [https://arxiv.org/abs/2610.32575](https://arxiv.org/abs/2610.32575)

    本文提出PHYRD动态基准（含63个复现题与43个发现题）及FormulaForest任务构建框架，用于评估大语言模型的理论物理推理能力，并证明FormulaForest作为合成数据可将模型在FrontierScience基准上的准确率从61%提升至72%。

    

    大型语言模型与智能体在高级物理研究中已展现出强大能力。本文提出PHYRD，一个用于理论物理推理的动态基准，包含63个“复现”问题和43个“发现”问题。“复现”任务测试模型对已发表结果的推导能力；“发现”任务测试模型在修改理论条件后的推论能力。FormulaForest通过参考答案和子问题级评分标准来构建这些任务。在五个模型的测试中，整题准确率在“复现”任务上介于49.21%至68.25%之间，在闭卷“发现”任务上介于18.60%至74.42%之间。借助工具，四个模型能完成更多“发现”问题，但概念错误和论证缺失依然存在。此外，实验结果证明了FormulaForest作为合成数据的价值：在OpenAI开发的FrontierScience基准上，它将Qwen-35B-A3B模型的准确率从61%提升至72%。

    arXiv:2610.32575v1 Announce Type: new  Abstract: Large language models and agents have shown significant power in advanced physical research. In this paper, We introduce PHYRD, a living benchmark for theoretical-physics reasoning with 63 Reproduce and 43 Discover problems. Reproduce tests derivation of published results; Discover tests consequences of modified theoretical conditions. FormulaForest constructs the tasks with reference solutions and subquestion-level rubrics. Across five models, whole-problem accuracy ranges from 49.21% to 68.25% on Reproduce and from 18.60% to 74.42% on closed-book Discover. With tools, four models complete more Discover problems, but conceptual errors and missing arguments persist. Besides, our experimental results demonstrate the value of FormulaForest as synthetic data. On FrontierScience, a benchmark developed by OpenAI, it raises the accuracy of a Qwen-35B-A3B model from 61% to 72%.
    
[^24]: 伪标签作为奖励而非目标：面向半监督自动语音识别的组相对策略优化

    Pseudo-Labels as Rewards, Not Targets: Group-Relative Policy Optimization for Semi-Supervised ASR

    [https://arxiv.org/abs/2610.32572](https://arxiv.org/abs/2610.32572)

    该论文提出将伪标签用作GRPO的奖励参考而非监督微调的目标，使教师错误传递率从100%降至27%，并在四个模型规模、四个教师家族、三个领域和四种语言上取得29.3%-56.2%的相对WER改进，显著提升了半监督ASR对标签噪声的鲁棒性。

    

    将自动语音识别（ASR）适配到仅有未转录音频的领域，通常依赖伪标签方法：由一个强大的教师模型转录音频，再用这些转录结果对学生模型进行微调。由于监督微调（SFT）将每个伪标签词元都视为学习目标，学生模型会继承教师模型的错误。我们证明，将相同的伪标签作为组相对策略优化（GRPO）的奖励参考，对标签噪声具有更强的鲁棒性。在银行领域，基于Whisper生成的标签，GRPO相比SFT在词错误率（WER）上取得29.3%至56.2%的相对改进，且该优势在四种Whisper模型规模、四个教师模型家族、三个领域和四种语言上均成立。SFT几乎原封不动地复现了100%的教师错误，而GRPO仅传递了27%。在线策略采样的统计数据解释了其中的原因：错误的伪标签词很少出现在采样结果中，因此组相对优势始终与干净的真实转录保持一致，而SFT则会将这些错误强行纳入模型策略之中。

    arXiv:2610.32572v1 Announce Type: new  Abstract: Adapting ASR to a domain with only untranscribed audio typically relies on pseudo-labeling, a strong teacher transcribes the audio and a student is fine-tuned on it. Because supervised fine-tuning (SFT) treats every pseudo-label token as a target, the student inherits the teacher's errors. We show that using the same pseudo-labels as a reward reference for Group Relative Policy Optimization (GRPO) is more robust to label noise. In the banking domain, GRPO yields 29.3% to 56.2% relative WER improvements over SFT on Whisper-generated labels, and the advantage holds across four Whisper sizes, four teacher families, three domains and four languages. While SFT reproduces almost 100% of teacher errors verbatim, GRPO passes through only 27%. Rollout statistics explain why, erroneous pseudo-label words rarely appear in on-policy rollouts, so group-relative advantages stay aligned with the clean transcript, whereas SFT forces them into the policy
    
[^25]: 原则性非线性概念擦除与引导

    Principled Nonlinear Concept Erasure and Steering

    [https://arxiv.org/abs/2610.32516](https://arxiv.org/abs/2610.32516)

    本文提出一个基于条件自编码器的原则性非线性概念擦除框架，从理论上证明完美重建与总体水平独立性足以实现完美的非线性防护保证，并在多个数据集上取得了更优的实用性与擦除效果的平衡。

    

    学习到的表示通常会编码敏感属性，例如性别或种族，这些属性若被保留可能是不可取的或不安全的。概念擦除旨在移除这些信息，同时保留与任务相关的内容。现有方法仍不完善：线性方法仅提供针对线性探针的保证，而非线性方法虽然实现了更强的经验擦除效果，但缺乏正式保证，且常常牺牲模型实用性。在这项工作中，我们为非线性概念擦除引入了一个原则性的条件自编码器框架。我们遵循这样的原则：擦除应当只移除敏感属性而不影响其他内容，并且我们证明，在给定假设下，完美重建结合总体水平的独立性足以实现完美的非线性防护性，同时在不完美重建的情形下也提供保留保证。在实证方面，我们在多个数据集上实现了更优的实用性-擦除权衡。

    arXiv:2610.32516v1 Announce Type: cross  Abstract: Learned representations often encode sensitive attributes, such as gender or race, that may be undesirable or unsafe to retain. Concept erasure aims to remove this information while preserving task-relevant content. Existing methods remain incomplete: linear methods provide guarantees only against linear probes, while nonlinear methods achieve stronger empirical erasure but lack formal guarantees and often sacrifice utility. In this work, we introduce a principled conditional autoencoder framework for nonlinear concept erasure. We follow the principle that erasure should remove only the sensitive attribute and nothing else, and we show that, under the stated assumptions, perfect reconstruction together with population-level independence is sufficient for perfect nonlinear guardedness, with preservation guarantees in the imperfect-reconstruction regime. Empirically, we attain a superior utility-erasure trade-off across multiple datasets
    
[^26]: 通过预测党派归属分析奥地利议会演讲

    Analysing Austrian Parliamentary Speeches by Predicting Party Affiliations

    [https://arxiv.org/abs/2610.32515](https://arxiv.org/abs/2610.32515)

    该论文利用微调BERT模型预测奥地利议会演讲的党派归属，发现联合执政党派的演讲特征趋于相似而反对党逐渐疏远，且议员只有在换党后才会改变演讲风格。

    

    本文研究了奥地利国民议会议员的演讲特征。我们提出了一种基于微调BERT模型的分析方法，该模型为每篇演讲分配一个关于奥地利各党派的概率分布，并在发言者和党派层面对这些分布进行聚合。在进行深入分析之前，我们进行了广泛的评估以确保生成分布的质量。我们没有将预测党派的错误归类视为性能问题，而是利用这些错误来推导关于议员和政治发展的见解。我们的研究结果表明，共同执政的党派在演讲特征上变得更加相似，而反对党则进一步疏远执政党。此外，转换党派的议员通常只有在换党之后才会调整其演讲特征。最后，我们公开发布了一个包含党派归属预测的数据集。

    arXiv:2610.32515v1 Announce Type: cross  Abstract: This paper examines speech characteristics of the Austrian National Council members (CMs). We introduce analysis methodology based on a fine-tuned BERT-Model which assigns to every speech a probability distribution over Austrian parties. These are then aggregated at the speaker and party level. We perform an extensive evaluation to ensure the quality of the generated distributions prior to further analysis. Instead of considering mis-assignments of predicted parties as a performance issue, we leverage them to derive insights about the CMs and political developments. Our findings show that parties that govern together become more similar in their speech characteristics, while, analogously, parties that are in opposition move further away from governing parties. Moreover, CMs who switch party usually adapt their speech characteristics only after the switch occurs. Finally, we publicly release a data set containing party affiliation predi
    
[^27]: 旁遮普Geet毒性数据集（PGT-FG）：针对旁遮普歌曲中有毒内容的细粒度多模态标注

    The Punjabi Geet Toxicity Dataset (PGT-FG): Fine-Grained Multimodal Annotations for Toxic Content in Punjabi Songs

    [https://arxiv.org/abs/2610.32513](https://arxiv.org/abs/2610.32513)

    本文提出了首个面向旁遮普歌曲毒性检测的段级多模态数据集PGT-FG，包含2,396个经四名母语标注者标注的片段，涵盖八类细粒度毒性类别及严重程度、情感等多维度标注，填补了旁遮普语音乐内容毒性检测资源的空白。

    

    旁遮普语的毒性检测资源仍然有限，特别是对于结合了语言信息与多模态信息的音乐内容。我们提出了旁遮普Geet毒性数据集（PGT-FG），这是一个段级资源，旨在支持旁遮普音乐内容毒性检测的研究。该数据集基于从YouTube收集的公开旁遮普歌曲构建而成。最初收集了620首歌曲，其中465首成功下载，并从450首歌曲中提取出3,402个初步片段。经过质量控制和标注流程后，最终数据集包含来自392首歌曲的2,396个独特片段。四名以旁遮普语为母语的标注者共产生了2,870条标注记录，其中包括158个由所有四名标注者共同标注的共享片段子集。每条标注包含二元毒性标签、八个细粒度毒性类别、严重程度、情感、模式以及不确定性标志。标注者间一致性使用Fleiss' kappa进行测量（摘要在此处截断）。

    arXiv:2610.32513v1 Announce Type: new  Abstract: Resources for toxicity detection in Punjabi remain limited, particularly for musical content that combines linguistic and multimodal information. We present the Punjabi Geet Toxicity Dataset (PGT-FG), a segment-level resource developed to support research on toxicity detection in Punjabi musical content. The dataset was constructed from publicly available Punjabi songs collected from YouTube. An initial collection of 620 songs resulted in 465 successfully downloaded songs and 3,402 preliminary segments from 450 songs. After quality control and annotation, the final dataset contains 2,396 unique segments from 392 songs. Four native Punjabi annotators produced 2,870 annotation records, including a shared subset of 158 segments annotated by all four. Each annotation includes binary toxicity, eight fine-grained toxicity categories, severity, sentiment, mode, and an uncertainty flag. Inter-annotator agreement was measured with Fleiss' kappa o
    
[^28]: Accordion：通过固定前缀层循环实现弹性深度推理

    Accordion: Elastic-Depth Inference via Fixed-Prefix Layer Looping

    [https://arxiv.org/abs/2610.32492](https://arxiv.org/abs/2610.32492)

    本文提出基于混合专家的模型Accordion，通过固定前缀层循环支持五种推理预算，并在预训练、监督微调和强化学习全流程中无需跨预算一致性目标即可在不同推理预算下保持强劲性能。

    

    大型语言模型服务于具有不同推理预算的各类应用，但为每个预算训练和维护单独的模型会产生巨大的成本。近期的循环Transformer通过参数重用实现了灵活的推理计算，但在不同推理预算下保持强劲性能仍然是一个开放的挑战。我们提出了Accordion，一种基于混合专家的模型，通过固定前缀层循环支持五种推理预算。每个预算遵循预定义的执行路径，而最小预算则保留未循环的模型。联合预训练在每个微批次中采样一个预算并优化其下一个词预测损失，无需额外的跨预算一致性目标。我们进一步通过监督微调和强化学习来保持这些推理预算，其中参数组合并和预算匹配的多教师在线策略蒸馏将三个领域专家整合到统一模型中。

    arXiv:2610.32492v1 Announce Type: cross  Abstract: Large language models serve applications with diverse inference budgets, but training and maintaining separate models for each budget incurs substantial costs. Recent Looped Transformers enable flexible inference computation through parameter reuse, but maintaining strong performance across different inference budgets remains an open challenge. We introduce Accordion, a mixture-of-experts-based model that supports five inference budgets through fixed-prefix layer looping. Each budget follows a pre-defined execution path, while the minimum budget retains the unlooped model. Joint pretraining samples one budget per micro-batch and optimizes its next-token prediction loss without additional cross-budget consistency objectives. We further preserve these inference budgets through supervised fine-tuning and reinforcement learning, where parameter-group merging and budget-matched multi-teacher on-policy distillation integrate three domain exp
    
[^29]: 当智能体失败时：模型还是执行框架——该归咎于谁？

    When Agents Fail: Model or Harness--Who's to Blame?

    [https://arxiv.org/abs/2610.32483](https://arxiv.org/abs/2610.32483)

    本文提出了一个通过五种互补指标（有效完成率、模型-框架匹配度、跨模型匹配度、交互残差和Token负载）来评估LLM智能体执行框架与底层模型交互表现的方法，用以区分智能体失败时究竟应归咎于模型还是执行框架。

    

    LLM智能体将模型与执行框架相结合，因此智能体基准测试的性能反映了二者的共同行为。我们认为，因此应当通过执行框架与底层模型的交互来评估框架。我们开发了一个结合五种互补指标的评估框架：(i) 有效完成率（VCR），即智能体正常完成并交付可评估结果的可靠程度，与任务正确性无关；(ii) 模型-框架匹配度，即框架通过有效完成来支持给定模型达成任务的程度；(iii) 跨模型匹配度，即框架在被评估的各模型上的整体适用性；(iv) 交互残差，即被组件级汇总指标所掩盖的特定模型-框架配对的执行优势与劣势；以及 (v) Token负载，即智能体执行所消耗的token量，包括失败尝试。我们使用四个核心模型评估了四个通用执行框架（OpenCode、Codex CLI、Pi CLI和DeepAgents），外加两个固定的原生模型。

    arXiv:2610.32483v1 Announce Type: new  Abstract: LLM agents combine models with execution harnesses, so agentic benchmark performance reflects their joint behavior. We argue that harnesses should therefore be evaluated through their interactions with underlying models. We develop a framework combining five complementary measures: (i) Valid Completion Rate (VCR), how reliably agents finish normally and deliver evaluable results, irrespective of task correctness; (ii) Model-Harness Fit, how well a harness supports task achievement through valid completion for a given model; (iii) Cross-Model Fit, a harness's overall suitability across evaluated models; (iv) Interaction Residuals, pairing-specific execution strengths and weaknesses hidden by component-level summaries; and (v) Token Load, the token demands of agent execution, including failed attempts. We evaluate four general-purpose harnesses (OpenCode, Codex CLI, Pi CLI, and DeepAgents) with four core models, plus two fixed native model
    
[^30]: 超越事后奖励：面向多模态嵌入的特权推理探索

    Beyond Post-Hoc Rewards: Privileged Reasoning Exploration for Multimodal Embeddings

    [https://arxiv.org/abs/2610.32474](https://arxiv.org/abs/2610.32474)

    提出PILOT框架，将自生成推理建模为潜在变量，通过共享多模态大语言模型的先验与后验两种推理模式进行特权推理探索，从而发现更具检索效用的推理轨迹，突破传统仅以输入为条件的事后奖励优化的局限，提升多模态嵌入的质量。

    

    多模态大语言模型（MLLM）的进展推动了多模态嵌入领域的显著发展。近期方法利用模型的生成能力，通过自生成的推理来增强被嵌入的输入，从而产生信息更丰富的嵌入表示。然而，用于训练嵌入模型的传统检索监督仅指定成对相关性关系，并未显式引导用于输入增强的推理过程。这类推理既无法被直接观测，也不由输入唯一确定，因此多种合理的推理轨迹可能引入检索效用各异的语义线索。现有方法大多仅在输入条件下优化推理，限制了对与检索相关轨迹的探索。我们提出PILOT框架，将推理建模为潜在变量，并将该变量的先验分布与后验分布实例化为同一共享MLLM的两种不同推理模式。其中，先验仅……（原文摘要在此处截断）

    arXiv:2610.32474v1 Announce Type: new  Abstract: Advances in Multimodal Large Language Models (MLLMs) have driven substantial progress in multimodal embeddings. Recent methods exploit generative capabilities by augmenting embedded inputs with self-generated reasoning to produce more informative embeddings. However, conventional retrieval supervision for training embedding models specifies pairwise relevance relations without explicitly guiding the reasoning used for input augmentation. Such reasoning is neither directly observed nor uniquely determined by the input, allowing multiple plausible traces to introduce semantic cues of varying retrieval utility. Existing approaches largely optimize reasoning conditioned on the input alone, limiting the exploration of retrieval-relevant traces. We introduce PILOT, a framework that models reasoning as a latent variable and instantiates its prior and posterior distributions as distinct reasoning modes of a shared MLLM. The prior conditions only
    
[^31]: 解读层间信息：视觉Transformer中的跨模态对齐与表征层级

    Reading Between the Layers: Cross-Modal Alignment and Representational Hierarchy in Vision Transformers

    [https://arxiv.org/abs/2610.32463](https://arxiv.org/abs/2610.32463)

    该研究通过逐层探测与跨模态对齐分析，首次系统揭示视觉Transformer中强大的跨模态语义映射仅在最深层出现，并呈现出从表层识别到语义理解的结构化表征层级。

    

    尽管大型视觉-语言模型在光学字符识别方面表现出色，但其底层视觉编码器如何处理文字及其含义，以及它们是否支持类似阅读理解的能力，仍不清楚。我们系统地研究了基于Transformer的视觉编码器，将渲染文本与视觉概念之间的跨模态对齐，连同探测和基于检索的分析，作为考察这些更广泛能力的受控窗口。我们的逐层分析涵盖正字法、语法和语义组织，揭示出强大的跨模态映射仅在最深层中出现，且是在单模态物体和文本识别已经发展之后才形成，这与从表层表征到语义表征的演进过程相吻合。尽管这种层级结构在对比训练的模型中最为显著，但更大的自监督模型和富含文本的训练数据在没有显式对齐（的情况下也能部分支持类似行为）……

    arXiv:2610.32463v1 Announce Type: new  Abstract: While large vision-language models excel at optical character recognition, it remains unclear how their underlying vision encoders process text and its meaning, and whether they support anything resembling reading comprehension. We systematically study transformer-based vision encoders, using cross-modal alignment between rendered text and visual concepts together with probing and retrieval-based analyses as a controlled window into these broader capabilities. Our layer-wise analyses span orthographic, grammatical, and semantic organization, and reveal that strong cross-modal mapping emerges only in the deepest layers, after unimodal object and text recognition have already developed, coinciding with a progression from surface-level to semantic representations. Although this hierarchy is strongest in contrastively trained models, larger self-supervised models and text-rich training data partially support similar behaviors without explici
    
[^32]: 解耦患者画像与临床工作流：一种用于逼真医疗对话生成的双智能体方法

    Decoupling Patient Personas and Clinical Workflows: A Dual-Agent Approach to Realistic Medical Dialogue Generation

    [https://arxiv.org/abs/2610.32443](https://arxiv.org/abs/2610.32443)

    提出CoRPS-Dial双智能体框架，通过解耦患者画像与临床工作流并辅以事后质量控制，生成兼顾临床规范性与患者多样性的逼真诊前医疗对话。

    

    诊前医疗对话对于收集全面的患者信息至关重要，但由于隐私和伦理限制，大规模获取这类对话成本高昂。现有基于大语言模型（LLM）的合成方法通常依赖提示工程或僵化的规则，限制了其在临床相关性与真实患者多样性之间取得平衡的能力。我们提出了CoRPS-Dial，这是一个用于诊前医疗对话合成的可控双智能体框架，它将患者画像与符合临床指南的工作流规则相整合。CoRPS-Dial通过由患者画像驱动的患者智能体与由流程引导的医生智能体之间的交互来生成对话，并进行事后质量控制以保证角色一致性、工作流完整性和文本一致性。在冠心病数据上的实验表明，CoRPS-Dial在信息覆盖度、临床合规性和对话自然度方面提升了合成对话与真实对话之间的对齐程度。

    arXiv:2610.32443v1 Announce Type: cross  Abstract: Pre-consultation medical dialogues are essential for collecting comprehensive patient information, but are costly to obtain at scale due to privacy and ethical constraints. Existing LLM-based synthesis methods often rely on prompt engineering or rigid rules, limiting their ability to balance clinical relevance with realistic patient diversity. We propose CoRPS-Dial, a controllable dual-agent framework for pre-consultation medical dialogue synthesis that integrates patient personas with guideline-aligned clinical workflow rules. CoRPS-Dial generates dialogues through interactions between a persona-driven patient agent and a process-guided doctor agent, followed by post-hoc quality control for role consistency, workflow completeness, and textual consistency. Experiments on coronary heart disease demonstrate that CoRPS-Dial improves alignment between synthetic and real dialogues in information coverage, clinical compliance, and dialogue n
    
[^33]: SCI-AGENT-EXAM：面向科研任务的智能体基准测试

    SCI-AGENT-EXAM: Benchmarking Agents on Scientific Research Tasks

    [https://arxiv.org/abs/2610.32441](https://arxiv.org/abs/2610.32441)

    提出了SCI-AGENT-EXAM基准测试，包含120个覆盖11个科学领域和7种任务类型的专家设计任务，并采用程序化验证与LLM评判相结合的混合评估协议，以系统评估科研智能体在真实研究工作流程中的综合能力。

    

    大语言模型（LLM）智能体在支持科学工作流程方面的能力日益增强，但它们在多样化科研活动中的能力仍难以进行系统性评估。现有基准测试往往只关注特定的科学能力或任务类型，限制了对科研智能体在真实研究工作流程中优劣势的全面刻画。我们提出了SCI-AGENT-EXAM，这是一个面向工作流程的基准测试，包含120个由专家精心策划的任务，涵盖11个科学领域以及由科学工作流程任务分类法（SWTT）定义的7种研究任务类型。每个任务同时需要可执行的产物（如代码和脚本）和文本产物（如报告和解释）。为了评估这些异构的任务输出，我们开发了一种混合评估协议，将针对可执行产物的确定性程序化验证与经过校准、由专家验证的基于大语言模型的评判器相结合，用于评估文本产物。

    arXiv:2610.32441v1 Announce Type: new  Abstract: Large language model (LLM) agents are increasingly capable of supporting scientific workflows, but their capabilities across diverse research activities remain difficult to assess systematically. Existing benchmarks often focus on selected scientific capabilities or task types, limiting a comprehensive characterization of scientific agents' strengths and weaknesses in realistic research workflows. We introduce SCI-AGENT-EXAM, a workflow-oriented benchmark comprising 120 expert-curated tasks across 11 scientific fields and 7 research task types defined by the Scientific Workflow Task Taxonomy (SWTT). Each task requires both executable artifacts (e.g., code and scripts) and textual artifacts (e.g., reports and explanations). To evaluate the heterogeneous task outputs, we develop a hybrid evaluation protocol combining deterministic programmatic verification for executable artifacts with a calibrated, expert-validated LLM-based judge for tex
    
[^34]: 基于层级奖励强化学习的面向任务对话用户画像驱动用户模拟

    Profile-Driven User Simulation via Reinforcement Learning with Hierarchical Rewards for Task-Oriented Dialogues

    [https://arxiv.org/abs/2610.32435](https://arxiv.org/abs/2610.32435)

    该论文提出ProfileSim框架，通过联合建模用户画像（人口统计与大五人格）和业务画像（从真实通话记录逆向提取的意图），并利用在真实语音数据上微调及带层级奖励的强化学习优化，实现了面向金融营销外呼场景、可细粒度控制的用户模拟。

    

    为金融营销等高价值场景构建面向任务的对话（TOD）系统需要大量特定领域的对话数据，然而隐私法规、标注成本以及对专家人力的需求使这类数据难以获取。用户模拟提供了一种解决途径，但现有方法面向通用领域对话，存在两个关键局限：一是对特定领域业务流程与合规约束的建模不足，二是对语音交互中口语化、口头语言特征的捕捉不够充分。我们提出了ProfileSim，一个面向金融营销外呼电话的用户画像驱动用户模拟框架。该框架联合建模用户画像（人口统计特征与大五人格特质）与业务画像（从真实通话记录中逆向工程提取的意图），从而实现对模拟行为的细粒度控制。基座模型在真实语音对话数据上进行微调，并通过群组相对（策略优化）…（摘要在此处截断）

    arXiv:2610.32435v1 Announce Type: new  Abstract: Building Task-Oriented Dialogue (TOD) systems for high-value scenarios such as financial marketing requires large volumes of domain-specific dialogue data, which privacy regulations, annotation costs, and the need for expert labor make hard to obtain. User simulation offers a remedy, but existing methods target general-domain dialogue and exhibit two key limitations: insufficient modeling of domain-specific business processes and compliance constraints, and inadequate capture of the colloquial, spoken-language character of voice interactions. We propose ProfileSim, a profile-driven user simulation framework for financial marketing outbound calls. It jointly models user profiles---demographics and Big Five personality traits---and business profiles---intents reverse-engineered from real call logs---enabling fine-grained control over simulated behavior. The base model is fine-tuned on real voice dialogue data and refined by Group Relative 
    
[^35]: 思维循环：面向文档图像中表格视觉问答的自评估迭代推理

    Cycle-of-Thought: Self-Evaluative Iterative Reasoning for Table Visual Question Answering in Document Images

    [https://arxiv.org/abs/2610.32423](https://arxiv.org/abs/2610.32423)

    提出了一种无需训练的推理时自我纠正框架，让单个视觉语言模型迭代地扮演生成者、批评者和改进者的角色，形成闭环自我反馈回路，无需额外标注数据或模型更新即可提升文档图像中表格视觉问答的性能。

    

    视觉语言模型（VLM）的最新进展显著提升了视觉问答（VQA）的性能，但它们对表格等结构化内容的推理仍不够稳定，常常受到空间错位和逻辑幻觉的困扰。现有的改进表格视觉问答的方法通常依赖于架构修改或昂贵的监督微调（SFT），由于需要大规模的专家标注数据集和大量计算资源，这些方法的准入门槛很高。在本工作中，我们提出了Self-RefineVQA，这是一个无需训练的推理时自我纠正框架，能够在不需要额外标注数据或模型更新的情况下提升表格视觉问答的性能。受人类修改行为的启发，我们的方法使单个视觉语言模型能够迭代地扮演生成者、批评者和改进者的角色，形成一个闭环的自我反馈回路，通过提示引导的重新评估来纠正常见的推理错误……

    arXiv:2610.32423v1 Announce Type: cross  Abstract: Recent advances in Vision-Language Models (VLMs) have significantly improved Visual Question Answering (VQA) performance, yet their reasoning over structured content such as tables remains inconsistent, often plagued by spatial misalignment and logical hallucinations. Existing approaches for improving table VQA typically rely on architectural modifications or expensive supervised fine-tuning (SFT), which impose high barriers to entry due to the need for large-scale, expert-annotated datasets and significant computational resources. In this work, we propose Self-RefineVQA, a training-free, inference-time self-correction framework that improves table VQA performance without additional labeled data or model updates. Inspired by human revision behavior, our method enables a single VLM to iteratively act as a generator, critic, and refiner, forming a closed self-feedback loop that corrects common reasoning errors through prompt-guided re-ev
    
[^36]: 大语言模型标注员能否匹配人类一致性？基于对比知识的人类一致性基准测试与多智能体优化

    Can LLM Annotators Match Human Consistency? Benchmarking Human Agreement and Optimizing Multi-Agent with Contrastive Knowledge

    [https://arxiv.org/abs/2610.32415](https://arxiv.org/abs/2610.32415)

    该论文提出了真实世界多标注者基准ANT-A（使用Fleiss kappa进行分布感知评估，衡量LLM标注与人类标注分布的一致性），并配套提出通过蒸馏对比知识优化提示词的多智能体标注框架ANT-OR，以提升标注系统对复杂标注指南的适应能力。

    

    基于大语言模型（LLM）的标注技术发展迅速，但现有基准测试仍与真实世界实践脱节。实际标注任务通常涉及冗长的、特定于任务的标注指南，且不同标注者可能产生不同但均属合理的判断，尤其是在边界案例上，而单标签数据集无法反映这种分歧。这一差距也限制了现有的单智能体和基于工作流的标注系统，它们通常缺乏对复杂且不断演变的标注指南的适应能力。为弥合这一差距，我们提出了ANT-A基准，这是一个真实世界的多标注者基准，涵盖跨领域、跨模态的7个数据集，包含特定任务的标注指南以及每个实例5个独立的人工标签。ANT-A支持分布感知的评估，使用Fleiss kappa来衡量模型标注与人类标注分布的接近程度。我们进一步提出了ANT-OR，一个多智能体标注框架，通过蒸馏对比知识来优化提示词……（原文摘要在此处截断）

    arXiv:2610.32415v1 Announce Type: new  Abstract: LLM-based annotation has advanced rapidly, yet existing benchmarks remain misaligned with real-world practice. Actual annotation often involves lengthy, task-specific guidelines, and different annotators may produce divergent but reasonable judgments, especially on boundary cases, where single-labeled dataset fail to reflect this disagreement. This gap also limits existing single-agent and workflow-based annotation systems, which often lack adaptability to complex, evolving guidelines. To bridge this gap, we introduce ANT-A Benchmark, a real-world multi-annotator benchmark spanning 7 datasets across domains and modalities, with task-specific guidelines and 5 independent human labels per instance. ANT-A enables a distribution-aware evaluation, using Fleiss kappa to assess how closely model annotations align with human labeling distributions. We further propose ANT-OR, a multi-agent annotation framework that refines prompts by distilling c
    
[^37]: 更广泛地审视多模态大语言模型的检索头

    A Broader Look into MLLM's Retrieval Heads

    [https://arxiv.org/abs/2610.32411](https://arxiv.org/abs/2610.32411)

    该研究通过构建语音与画面内容相同的视听配对输入发现，检索头机制随新增模态依然保持稀疏并在跨模态间高度共享，且全模态模型在很大程度上保留了原有模态的检索头。

    

    在长输入中定位相关证据是多模态大语言模型（MLLM）的一项核心能力。先前的研究将这种能力归因于检索头（retrieval heads），即一组稀疏的注意力头，负责将信息从上下文传递到输出。然而，这些分析仅局限于有限的模态集合及相应的模型。此外，如何利用检索头来增强模型仍有待进一步探索。本文探究了这一机制是否会随模型输入模态的扩展而扩展。我们将视听输入作为一种新增模态，构建了成对输入，其中相同的答案既通过语音说出，也在屏幕上显示。在证据内容保持固定的情况下，检索头在新模态中依然保持稀疏，并在很大程度上跨模态共享。在18个语言、视觉-语言和全模态模型检查点上，全模态模型在很大程度上保留了原有模态的检索头。

    arXiv:2610.32411v1 Announce Type: new  Abstract: Locating relevant evidence in long inputs is a core capability of Multimodal Large Language Models. Prior studies attribute this capability to retrieval heads, a sparse set of attention heads that transfer information from the context to the output. However, these analyses are confined to a limited set of modalities and the corresponding models. Moreover, how retrieval heads can be leveraged to enhance the model calls for further exploration. In this paper, we ask whether this mechanism expands together with the input modalities of the model. Taking audio-visual input as a newly added modality, we construct paired inputs in which the same answer is either spoken or shown on screen. With the evidence content fixed, retrieval heads remain sparse in the new modality and are shared to a large extent across modalities. Across 18 language, vision-language, and omni-modal checkpoints, omni-modal models largely preserve the retrieval heads of th
    
[^38]: 面向语言学习者母语识别的声学与词汇信号

    Acoustic and Lexical Signals for Native Language Identification of Language Learners

    [https://arxiv.org/abs/2610.32382](https://arxiv.org/abs/2610.32382)

    该研究首次将基于WavLM的音频分类器与零样本GPT-4o转写文本分类器相结合用于第二语言学习者的母语识别，发现音频方法在低熟练度（A1）时更优，而文本方法在中高熟练度（A2–C1）时更具优势。

    

    第二语言学习者的口语母语（L1）识别传统上依赖于语音的声学属性，而通过分析语音转写文本来探寻母语痕迹的研究关注相对较少。本文提出了一项利用口语学习者英语语料库中的语音及其对应转写文本进行母语识别的研究。我们将基于WavLM的音频母语分类器与在转写文本上运行的零样本GPT-4o分类器进行对比，并使用CEFR平衡的评估集在不同熟练度水平上分析二者的预测结果。实验表明，当完整转写文本可用时，音频分类器总体上优于基于转写文本的方法，但两者的相对优势随学习者熟练度而变化：音频方法在A1级别表现更好，而从A2到C1级别，在拥有完整转写文本的情况下文本方法表现更佳。基于转写文本的准确率通常随熟练度提高而上升，而声学性能则保持……（摘要原文在此处被截断）

    arXiv:2610.32382v1 Announce Type: new  Abstract: Spoken native language (L1) identification for second language learners has traditionally relied on acoustic attributes of speech. Comparatively little attention has been paid to analysing speech transcripts for L1 traces. In this paper, we present a study of L1 identification using both speech and corresponding transcripts from a spoken learner English corpus. We compare a WavLM-based audio L1 classifier with a zero-shot GPT-4o classifier operating on transcripts and analyse their predictions across proficiency levels using a CEFR-balanced evaluation set. Our experiments show that the audio classifier outperforms the transcript-based approach overall when full transcripts are available, although their relative strengths vary with learner proficiency. Audio performs better at A1, whereas text performs better from A2 to C1 with full transcripts. Transcript-based accuracy generally increases with proficiency, while acoustic performance rem
    
[^39]: 为什么全双工语音模型在长对话中会崩溃

    Why Full-Duplex Speech Models Collapse in Long Conversations

    [https://arxiv.org/abs/2610.32349](https://arxiv.org/abs/2610.32349)

    本文提出全双工长时程基准FD-LHB，通过长时重放对话和与LLM实时对话的测试，首次系统揭示了现有全双工语音模型在长对话中均会退化或耗尽上下文，陷入重复、沉默和胡言乱语，并识别了其背后的成因机制。

    

    全双工语音模型能够同时进行听和说，这使它们在自然、开放式对话方面极具前景。但它们能否持续进行数十分钟的连贯对话？由于这类模型即使在聆听时也会在每一帧生成一个token，它们会在几分钟内耗尽上下文窗口。然而，现有的基准测试仅评估单轮或少数几轮对话，时长太短，无法暴露长时程问题。我们提出了全双工长时程基准（FD-LHB），在长时重放对话以及与LLM伙伴的实时对话中测试了五个最新的全双工模型。所有被评估的模型最终都会性能退化或耗尽上下文。保留完整历史的模型往往在上下文耗尽之前就已退化，而滑动窗口模型仍会陷入重复、沉默和胡言乱语。对于滑动窗口模型，我们识别出两种促成机制。胡言乱语和沉默与……（原文摘要在此处被截断）

    arXiv:2610.32349v1 Announce Type: new  Abstract: Full-duplex speech models listen and speak at the same time, which makes them promising for natural, open-ended conversation. But can they sustain a coherent conversation for tens of minutes? Because they generate a token every frame, even while listening, they exhaust their context windows within minutes. Yet existing benchmarks evaluate only single turns or a few conversational rounds, too short to expose the long-horizon problem. We introduce the Full-Duplex Long-Horizon Benchmark (FD-LHB), which tests five recent full-duplex models on long replayed conversations and in live conversation with an LLM partner. Every evaluated model eventually either degrades or exhausts its context. Models that keep the full history often degrade before their context runs out, while sliding-window models still fall into repetition, silence, and gibberish. For sliding-window models, we identify two contributing mechanisms. Gibberish and silence are linke
    
[^40]: 超越最终结果：量化多智能体LLM交互中的声明级信息动态

    Beyond Final Outcomes: Quantifying Claim-Level Information Dynamics in Multi-Agent LLM Interaction

    [https://arxiv.org/abs/2610.32333](https://arxiv.org/abs/2610.32333)

    本文提出RIVER框架，将多智能体LLM交互输出分解为原子声明，从收敛、探索和保留三个维度量化声明级信息动态，发现重复聚合式交互促进收敛但缩小信息空间，而分布式接力式交互能产生更广泛的信息。

    

    当基于大语言模型（LLM）的智能体进行交互时，信息会被反复共享、扩展和遗忘，然而多智能体评估通常只关注最终任务结果。我们将这种交互作为声明级别的信息动态进行研究，并提出RIVER方法，该方法将智能体输出分解为原子声明，并评估三个互补维度：收敛性、探索性和保留性。收敛性刻画跨智能体的声明对应关系和立场一致性；探索性刻画语义广度以及超出源信息的扩展；保留性追踪源信息的保存情况以及新声明在各阶段和谱系中的持续性。我们在40个受控新闻生成案例上评估了五种交互策略。重复聚合式交互促进收敛但缩小了信息空间，而分布式接力式交互则产生更广泛、共享程度更低的信息。进一步比较表明，方法特定的约束能够增加探索性，而较弱的……

    arXiv:2610.32333v1 Announce Type: new  Abstract: When LLM-powered agents interact, information is repeatedly shared, expanded, and forgotten, yet multi-agent evaluation typically focuses on final task outcomes. We study such interaction as claim-level information dynamics and propose RIVER, which decomposes agent outputs into atomic claims and evaluates three complementary dimensions: convergence, exploration, and retention. Convergence captures cross-agent claim correspondence and stance alignment; exploration captures semantic breadth and extension beyond the source; and retention tracks source preservation and the persistence of new claims across stages and lineages. We evaluate five interaction strategies on 40 controlled news-generation cases. Repeatedly pooled interaction promotes convergence but narrows the information space, whereas distributed relay yields broader, less shared information. Further comparisons show that method-specific constraints increase exploration while wea
    
[^41]: 4DMemory：基于工具增强智能体的长时程时空记忆基准测试

    4DMemory: Benchmarking Long-Horizon Spatiotemporal Memory with Harness-Augmented Agents

    [https://arxiv.org/abs/2610.32332](https://arxiv.org/abs/2610.32332)

    该论文提出了包含约900小时视频和5000多个人工审核问题的 LongVideo 数据集及 4DMemoryBench 基准，并开发了集成四种互补技能、22个以上专用工具模型和检索增强生成系统的 4DMemory Harness，用于评估和提升基础模型的长时程时空记忆推理能力。

    

    人类能够高效处理跨越数小时至数年的多模态数据流，构建持久、细粒度的时空记忆，以支持对长时程事件的推理与预测。相比之下，现有基准测试主要聚焦于短上下文和中等上下文任务，而大多数基础模型难以在长时程中保留细粒度信息。为了评估和改进长时程时空推理能力，我们提出了 LongVideo 数据集和 4DMemoryBench 基准。LongVideo 包含约 900 小时的视频和超过 5,000 个经人工审核的问题，涵盖不同上下文长度下的视觉、文本和时空推理。我们进一步开发了 4DMemory Harness，它集成了四种互补技能、22 个以上的专用工具模型以及针对特定模型的检索增强生成系统。通过对 13 个领先模型的评估，我们发现独立运行的基础模型在长时程任务上表现不佳，而 4DMemory Harness（摘要在此处截断）……

    arXiv:2610.32332v1 Announce Type: cross  Abstract: Humans efficiently process multimodal streams spanning hours to years, building persistent, fine-grained spatiotemporal memories that support reasoning about and predicting long-horizon events. In contrast, existing benchmarks focus on short- and medium-context tasks, and most foundation models struggle to preserve fine-grained information over long horizons. To evaluate and improve long-horizon spatiotemporal reasoning, we introduce the LongVideo dataset and the 4DMemoryBench. LongVideo comprises about 900 hours of video and over 5,000 manually reviewed questions covering visual, textual, and spatiotemporal reasoning across diverse context lengths. We further develop the 4DMemory Harness, which integrates four complementary skills, 22+ specialized tool models, and a model-specific retrieval-augmented generation system. Evaluating 13 leading models, we find that standalone foundation models struggle with long-horizon tasks, while the 4
    
[^42]: 系统一模型并非更好的重排序器

    System One models Are Not Better Rerankers

    [https://arxiv.org/abs/2610.32318](https://arxiv.org/abs/2610.32318)

    本研究通过在0.6B至27B参数的六个Qwen骨干模型和十个TREC DL与BEIR数据集上的系统对比，证明决策微调的System One模型并不比普通重排序微调模型是更好的重排序器——重排序微调在0.6B至9B全面胜出5至29个nDCG@10点，27B时两者持平，且9B以下决策微调相比零样本询问未调优模型并无增益。

    

    系统一模型，或称决策模型，通过对输入问题给出一个经过校准的概率而非生成文本来进行回答。在2026年9月首次发布后的几周内，它们便被用作重排序器。我们测试了其背后的决策微调是否比同一模型的普通重排序微调能产生更好的重排序器。针对从0.6B到27B参数的六个Qwen骨干模型以及十个TREC DL和BEIR数据集，我们将零样本提问的未调优模型与其决策微调版本和重排序微调版本进行比较，其中两个微调版本由我们自行训练。在这项工作中，我们得出结论：决策微调永远不会造就更好的重排序器。在0.6B至9B参数范围内，重排序微调在每个骨干模型上均以5到29个点的nDCG@10优势胜出（包括在零样本数据集上），而在27B时两者打平。在9B及以下，决策微调相比直接零样本询问未调优模型没有任何额外收益。此外，我们发现决策模型的校准优势……（摘要在此处截断）

    arXiv:2610.32318v1 Announce Type: new  Abstract: System One models, or decision models, answer a question about an input with a calibrated probability instead of generated text. Within weeks of the first release, in September 2026, they were used as rerankers. We test whether the decision fine-tuning behind them makes a better reranker than plain reranker fine-tuning of the same model. For six Qwen backbones from 0.6B to 27B parameters and ten TREC DL and BEIR collections, we compare the untuned model, asked zero-shot, with its decision fine-tunes and its reranker fine-tunes, two of which we train ourselves. In our work, we draw the conclusion that decision fine-tuning never makes the better reranker. From 0.6B to 9B, reranker fine-tuning wins on every backbone by 5 to 29 points nDCG@10, also on zero-shot collections, and at 27B the two tie. Up to 9B, decision fine-tuning adds nothing over simply asking the untuned model. Moreover, we find that the calibration advantage of decision mod
    
[^43]: 多种扰动，共享的失败动力学：追踪语言模型中方向敏感的扰动影响

    Many Perturbations, Shared Failure Dynamics: Tracing Direction-Sensitive Perturbation Influence in Language Models

    [https://arxiv.org/abs/2610.32309](https://arxiv.org/abs/2610.32309)

    该论文提出一种方向敏感、相对于读出过程的语义脆弱性分析框架，通过纵-横分解与储备-传输模型揭示异质扰动在语言模型中共享相同的失败动力学，并证明扰动的方向（而不仅是标量大小）携带因果信息。

    

    为什么一次保持任务语义的改写会把正确答案变成失败，而更大的内部变化却让答案保持不变？我们提出了一种方向敏感的、相对于读出过程的语义脆弱性解释：异质的语言变化可以共享几何与读出关系，而不必共享一个与输入无关的失败方向。干净的决策储备、表示变换和响应对齐为这一过程提供了互补的描述。一个精确的纵-横分解识别出被标量传输所丢弃的方向信息。一个紧凑的储备-传输模型在 Qwen3-32B 和 Gemma 4 31B 上对留出的族群和条目实现了失败预测，并向 ReClor 实现了有用但依赖参数化方式的迁移。匹配干预实验表明，扰动的方向携带了超越标量几何的因果信息。在自然发生的失败改写中，针对不相交原始样本的注意力组件测试……（原文摘要在此处截断）

    arXiv:2610.32309v1 Announce Type: new  Abstract: Why can a task-preserving rewrite turn a correct answer into a failure while a larger internal change leaves the answer intact? We develop a direction-sensitive, readout-relative account of semantic fragility: heterogeneous language changes can share geometric and readout relationships without sharing an input-independent failure direction. Clean decision reserve, representation transformation, and response alignment provide complementary descriptions of this process. An exact longitudinal-transverse decomposition identifies directional information discarded by scalar transmission. A compact reserve-and-transmission model predicts failures across held-out families and items in Qwen3-32B and Gemma 4 31B, with useful but parameterization-dependent transfer to ReClor. Matched interventions show that orientation carries causal information beyond scalar geometry. Among naturally failed rewrites, attention-component tests on disjoint original 
    
[^44]: DMAIC-Bench：多阶段分析任务中智能体从自我反思到自我克制

    DMAIC-Bench: From Self-Reflection to Self-Restraint for Agents in Multi-Stage Analytical Tasks

    [https://arxiv.org/abs/2610.32305](https://arxiv.org/abs/2610.32305)

    该论文提出 DMAIC-Bench 基准，通过基于精益六西格玛的成对反事实任务首次系统评估大语言模型智能体的“自我克制”能力（即依据证据决定继续还是停止），发现模型在 GO 决策上准确率达 92%，但在 NO-GO 决策上仅约 19%。

    

    多阶段分析任务要求大语言模型智能体利用中间证据来决定是继续、转向还是停止。自我反思（即智能体批判自身输出的过程）已被广泛研究，然而批判本身并不能表明证据是否改变了下一步行动。我们将这种能力称为“自我克制”。现有的智能体基准测试主要衡量任务完成度或输出质量，而未对这一能力进行检验。DMAIC-Bench 通过 97 个种子任务来评估这一能力，这些任务覆盖四个行业的 15 个工业案例，并围绕精益六西格玛明确的统计学前提构建。成对的反事实变体要求智能体在同一工作流位置采取相反的行动，从而将有依据的克制与不加区分的停止区分开来。我们使用私有评分器评估执行轨迹，并通过独立专家评审评估最终结论。在对来自六家提供商的 13 个模型的评估中，执行层面的路由准确率在匹配的 GO 决策上平均为 92%，而在 NO-GO 决策上仅为 19%。

    arXiv:2610.32305v1 Announce Type: new  Abstract: Multi-stage analytical tasks require LLM agents to use intermediate evidence to decide whether to continue, redirect or stop. Self-reflection, in which an agent critiques its output, is widely studied, yet a critique does not show whether evidence changed the next action. We call that capability self-restraint. Existing agent benchmarks largely measure task completion or output quality and leave it untested. DMAIC-Bench evaluates it with 97 seeded tasks across 15 industrial cases in four sectors, built around Lean Six Sigma's explicit statistical prerequisites. Paired counterfactual variants require opposite actions at the same workflow position, distinguishing warranted restraint from indiscriminate stopping. We assess execution traces with a private scorer and final claims through independent expert review. Across 13 models from six providers, execution-level routing accuracy averages 92% on matched GO decisions and 19% on NO-GO decisi
    
[^45]: “Not”在翻译中并未丢失：多语言大语言模型中否定表征的因果分析

    "Not" is Not Lost in Translation: A Causal Analysis of Negation Representations in Multilingual LLMs

    [https://arxiv.org/abs/2610.32281](https://arxiv.org/abs/2610.32281)

    研究发现多语言大模型在各语言间共享可因果干预的否定表征，从一种语言提取的否定激活方向可以在其他语言中诱导或移除否定，且否定由稀疏特征表示。

    

    否定存在于每一种语言中，但其表层实现方式差异巨大。尽管如此，多语言大语言模型（LLMs）仍能稳健地翻译否定，这引出了一个值得探究的问题：它们是否在不同语言间共享某种底层的否定表征。我们在六个开源模型中，针对英语、法语、越南语和中文，通过在一个合成对比数据集以及从FLORES-200衍生的否定翻译数据集上进行因果干预来研究这一问题。在所有模型中，我们发现了强有力的证据，表明否定表征是因果可控的，并且在各语言间基本共享。特别地，我们展示了从一种语言提取的对比激活方向可以在其他语言中诱导或移除否定，且从不同语言提取的方向在实质上是相互对齐的。我们进一步发现，否定是以稀疏特征的形式表示的：每个模型中单个稀疏自编码器（SAE）特征即足以……

    arXiv:2610.32281v1 Announce Type: cross  Abstract: Negation occurs in every language, yet its surface realization differs drastically. Despite this, multilingual large language models (LLMs) translate negation robustly, raising the question of whether they share an underlying representation of negation across languages. We study this question in six open models across English, French, Vietnamese, and Chinese, using causal interventions on a synthetic contrastive dataset and a negation translation dataset derived from FLORES-200. Across models, we find strong evidence that negation representations are causally controllable and largely shared across languages. In particular, we show that a contrastive activation direction extracted from one language can induce or remove negation in the others, and directions extracted from different languages are substantially aligned. We further find that negation is represented in sparse features: a single sparse autoencoder (SAE) feature per model suf
    
[^46]: 循环语言模型中的停止头真的能节省计算吗？

    Do Halting Heads Really Save Compute in Looped Language Models?

    [https://arxiv.org/abs/2610.32246](https://arxiv.org/abs/2610.32246)

    本文揭示循环语言模型中的停止头因训练方式缺陷和批次生成机制而无法真正节省计算量，无停止头的模型在同等计算成本下表现几乎一样好。

    

    循环语言模型让每个词元多次通过相同的层，每次通过称为一次迭代。为了节省计算量，模型引入一个额外的层——停止头，它在每次迭代后给出词元在此停止的概率，使简单的词元可以提前停止。我们从零开始预训练了参数量最高达13亿的模型，发现了两个问题。第一，在所有迭代的混合数据上训练的停止头预测词元所需的迭代次数不足一半，但在标准停止规则下，词元实际上几乎每次迭代都会运行，因为这种训练方式从未对词元真正停止的那一次迭代进行评分。改为对每次迭代都进行评分可以解决这一问题，但在我们的实验中，只有在采用强烈偏向提前停止的惩罚时才有效。第二，计算节省量通常以每个词元的平均迭代次数来报告，但当序列以批次方式生成、且需运行到批次中最慢的词元停止为止时，没有停止头的模型在同等计算成本下表现几乎一样好。

    arXiv:2610.32246v1 Announce Type: new  Abstract: Looped language models run each token through the same layers several times, and each run is one iteration. To save compute, an extra layer, the halting head, gives after each iteration the probability that the token stops there, so that easy tokens can stop early. Pretraining models from scratch, up to 1.3B parameters, we find two problems. First, heads trained on a mix of all iterations predict that tokens need under half the iterations, yet under the standard stopping rule tokens run almost every iteration, because this training never scores the single iteration where a token stops. Scoring every iteration instead fixes this, but in our runs it works only with a strong penalty favoring early stops. Second, savings are commonly reported as an average number of iterations per token, but when sequences are generated in batches that run until their slowest token stops, a model without a head is almost as good at that cost. The reported av
    
[^47]: DIAL-S2ST：面向同声语音到语音翻译的延迟控制学习方法

    DIAL-S2ST: Learning Latency Control for Simultaneous Speech-to-Speech Translation

    [https://arxiv.org/abs/2610.32214](https://arxiv.org/abs/2610.32214)

    提出DIAL-S2ST后训练方法，通过前瞻预测自蒸馏学习同声语音到语音翻译模型的推理时延迟控制，仅需10小时源语音即可实现灵活的延迟调节与良好的质量-延迟权衡。

    

    我们提出了DIAL-S2ST，这是一种后训练方法，用于在同声语音到语音翻译模型中学习推理时的延迟控制。我们的方法通过对前瞻预测进行自蒸馏来复用预训练的翻译能力，同时限制大型文本蒸馏损失的影响，以应对模仿差距。这为更早的输出提供了监督，而无需配对目标数据或奖励计算。为了使学习到的输出时机在推理时可调节，我们在匹配的前瞻引导下跨多个尺度联合训练单个LoRA适配器。基于Hibiki的法语到英语翻译实验表明，仅需10小时的源语音即可实现这种适配，提供了灵活的延迟控制和良好的质量-延迟权衡。

    arXiv:2610.32214v1 Announce Type: new  Abstract: We propose DIAL-S2ST, a post-training method for learning inference-time latency control in simultaneous speech-to-speech translation models. Our approach reuses pretrained translation capabilities through self-distillation from lookahead predictions, while limiting the influence of large text distillation losses to account for the imitation gap. This provides supervision for earlier emission without paired target data or reward computation. To make the learned timing adjustable at inference, we jointly train a single LoRA adapter across scales with matched lookahead guidance. Experiments on French-to-English translation with Hibiki demonstrate that only 10 hours of source speech enable this adaptation, providing flexible latency control with a favorable quality-latency trade-off.
    
[^48]: 头维度对语言模型长上下文检索至关重要

    Head Dimension Matters in Language Models for Long-Context Retrieval

    [https://arxiv.org/abs/2610.32213](https://arxiv.org/abs/2610.32213)

    本研究发现在固定模型规模下，增大注意力头维度可通过提高检索头的比例显著提升语言模型的长上下文检索性能，且即使冻结注意力投影权重该结论依然成立，而头维度对预训练损失和短上下文任务性能影响甚微。

    

    多头注意力通过查询-键交互和值聚合来检索与组合上下文表示，其中头维度限制了每个注意力头的表示空间。在长上下文处理中，将固定的参数预算分配给更少但更宽的头是否能改善信息检索，目前仍不清楚。在本文中，我们研究了头维度如何影响长上下文检索任务的性能。我们的实证研究表明，头维度对预训练损失和短上下文非检索任务的性能影响不大，而在固定模型规模下，长上下文检索性能会随着头的变宽而提升。我们发现，拥有更宽头的模型具有更高比例的检索头，即有更多参数与检索功能相关联。我们的消融实验还表明，即使冻结注意力模块中的投影权重，检索性能仍会随头变宽而提升。

    arXiv:2610.32213v1 Announce Type: cross  Abstract: Multi-head attention retrieves and combines context representations through query--key interactions and value aggregation, with head dimension constraining the per-head representation space. In long-context processing, it remains unclear whether allocating a fixed parameter budget to fewer, wider heads improves information retrieval. In this paper, we investigate how head dimension affects the performance of long-context retrieval tasks. Our empirical studies reveal that pretraining loss and short-context non-retrieval task performance are not affected much by head dimension, while long-context retrieval performance improves with wider heads at fixed model sizes. We find that models with wider heads have a higher proportion of retrieval heads, associating more parameters with retrieval. Our ablation studies also show that the retrieval performance improves with wider heads even when the projection weights in attention modules are froze
    
[^49]: SOPO：苏格拉底引导的策略优化用于片段级幻觉检测

    SOPO: Socratic Guided Policy Optimization for Span-Level Hallucination Detection

    [https://arxiv.org/abs/2610.32212](https://arxiv.org/abs/2610.32212)

    SOPO提出了一种苏格拉底引导的多轮强化学习框架，通过教师模型提供基于真实标签的间接提示来引导推理轨迹，从而解决片段级幻觉检测中的定位模糊、困难样本奖励稀疏和长度膨胀问题。

    

    大语言模型（LLMs）已在各种自然语言处理任务中得到广泛应用，但存在幻觉问题，即生成不忠实或不一致的内容。虽然可验证奖励强化学习（RLVR）在推理任务中展现出有前景的应用，但将其应用于片段级幻觉检测面临定位模糊性、困难样本奖励稀疏以及不受控的长度膨胀等挑战。为解决这些问题，我们提出了苏格拉底引导的策略优化（SOPO），这是一个用于鲁棒幻觉检测的多轮强化学习框架。SOPO的关键创新在于苏格拉底反馈机制，其中教师模型通过提供从真实标签推断出的间接提示，在采样过程中引导推理轨迹。这种教师引导的方法使模型能够为困难样本生成更有效的推理轨迹，从而缓解奖励稀疏问题。

    arXiv:2610.32212v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have gained widespread adoption in various natural language processing tasks, but suffer from hallucination issues where they generate unfaithful or inconsistent content. While Reinforcement Learning with Verifiable Rewards (RLVR) has shown promising applications in reasoning tasks, its application to span-level hallucination detection is challenged by localization ambiguity, reward sparsity on hard examples, and uncontrolled length inflation. To address these issues, we introduce Socratic Guided Policy Optimization (SOPO), a multi-turn RL framework for robust hallucination detection. The key innovation of SOPO is a Socratic feedback mechanism, where a teacher model guides the reasoning trajectories during rollout by providing indirect hints inferred from the ground truth. This teacher-guided approach enables the model to generate more effective reasoning trajectories for hard examples, thus alleviating rew
    
[^50]: SWE-Shift：在动态演变的工作区中评估编程智能体

    SWE-Shift: Evaluating Coding Agents in Evolving Workspaces

    [https://arxiv.org/abs/2610.32196](https://arxiv.org/abs/2610.32196)

    提出 SWE-Shift 基准，用于评估编程智能体在任务执行中途遭遇代码、依赖和需求等变化时的表现，实验发现所有 17 个被测智能体系统在发生此类变化时成功率均显著下降。

    

    仓库级编程基准测试在很大程度上假设任务和环境在整个执行过程中保持不变。而在软件开发中，代码、依赖项、运行时状态和需求可能在智能体仍在工作时发生变化。我们提出了 SWE-Shift，一个用于评估编程智能体在执行中途遭遇变化时表现的基准。基于相同的保存历史和工作区，Clean（干净）延续在不干预的情况下继续进行，而与之匹配的 Shifted（转变）延续则会收到与任务相关的更新。成功需要完成当前任务、保留仍然有效的工作，并满足适用的状态和证据检查。该基准包含横跨 12 个变化族别的 120 个场景，每个场景在九个预定的干预位置进行评估。我们评估了 17 个编程智能体系统，所有系统在 Shifted 条件下的成功率均低于 Clean 条件。GPT-6 Astra 的成功率从 67.35% 降至 24.79%，配对成功率为 20.34%，工具/会话变化和过时证据是主要失败因素之一。

    arXiv:2610.32196v1 Announce Type: new  Abstract: Repository-level coding benchmarks largely assume that tasks and environments remain unchanged throughout execution. In software development, code, dependencies, runtime state, and requirements can change while an agent is still working. We introduce SWE-Shift, a benchmark for evaluating coding agents under such mid-execution changes. From the same saved history and workspace, a Clean continuation proceeds without intervention, while a matched Shifted continuation receives a task-relevant update. Success requires completing the current task, preserving still-valid work, and satisfying the applicable state and evidence checks. The benchmark contains 120 scenarios across 12 change families, each evaluated at nine scheduled intervention positions. We evaluate 17 coding-agent systems, all of which have lower Shifted than Clean success. GPT-6 Astra drops from 67.35% to 24.79%, with 20.34% paired success. Tool/session changes and stale evidenc
    
[^51]: 收集日期标注会重定向时间性问答中的答案

    Collection-Date Assignment Redirects Answers in Temporal QA

    [https://arxiv.org/abs/2610.32191](https://arxiv.org/abs/2610.32191)

    该研究发现在证据内容完全不变的情况下，仅为文档标注不同的收集日期元数据就能让大语言模型在时间性问答中大幅转向以该日期为参照的答案（精确匹配率下降最高达51.4个百分点），且包括明确忽略指令在内的七种提示策略均无法消除这一效应。

    

    收集日期记录的是源文本修订版被获取的时间，而非其内容发生变化的时间。在证据内容不变的情况下，为文档标注这一元数据能否改变时间性问答的答案？在595个维基百科页面上，将问题日期与次日在两个修订版之间互换，使五个开源权重大语言模型在引用“后一日期”时的精确匹配率（EM）下降了37.6至51.4个百分点，并在128个互不重叠的页面上得到复现。改用前一日期的互换在497个页面上仍造成29.0至39.6个百分点的损失。包括明确要求忽略收集日期的指令和归档修订版来源信息在内的七种提示词变体，均无法消除该效应。虚构答案和以英文月份名称表示的问题日期表明，模型记住的答案和完全相同的日期字符串都不是产生该效应的必要条件。匹配的思考模式将Qwen3.5的损失从30.1个百分点降至11.1个百分点，而原生推理模型的表现则参差不齐。两个API大语言模型在维基百科互换实验中仅表现出较小的效应，且在虚构记录上很少转向更早的答案。大规模互换引起的（摘要在此处被截断）。

    arXiv:2610.32191v1 Announce Type: new  Abstract: A collection date records when a source revision was acquired, not when its contents changed. Can assigning this metadata redirect temporal question answering despite unchanged evidence? Swapping the question day and following day between two revisions lowers later-reference exact match (EM) by 37.6-51.4 percentage points across five open-weight LLMs on 595 Wikipedia pages, replicated on 128 disjoint pages. Previous-day swaps retain 29.0-39.6-point losses on 497 pages. Seven prompt variants, including explicit ignore-collection instructions and archived-revision provenance, retain the effect. Fictional answers and English month-name question dates show that remembered answers and identical date strings are unnecessary. Matched thinking reduces Qwen3.5's penalty from 30.1 to 11.1 points, while native reasoners vary. Two API LLMs show small Wikipedia swap effects and rarely shift to earlier answers on fictional records. Large swap-induced 
    
[^52]: 视觉语言模型中深度依赖的安全几何结构：逐层定位与双向拒答控制

    Depth-Dependent Safety Geometry in Vision-Language Models: Layer-Wise Localization and Bidirectional Refusal Control

    [https://arxiv.org/abs/2610.32182](https://arxiv.org/abs/2610.32182)

    该论文提出一个无需重训的推理时框架，通过逐层激活分析揭示视觉语言模型的安全几何随深度非均匀分布——早期视觉层可通过激活引导实现 6%–96% 的双向拒答控制，而深层虽表征可分却对操控不敏感，从而区分了表征可分性与因果可控性。

    

    视觉语言模型（VLM）已被广泛部署，但其安全行为背后的内部机制仍知之甚少。我们提出了一个轻量级的推理时框架，用于在冻结的视觉语言模型中定位并操控安全敏感表征，无需重新训练或修改参数。在 Qwen3-VL-4B 上，逐层激活差异分析揭示了安全几何结构在模型深度上呈非均匀分布。因果干预实验表明，早期视觉层通过沿学习到的安全方向进行激活引导，可实现强大的双向拒答控制，不安全拒绝率可在 6%–96% 之间调节；而后面的层对激活幅值缩放更为敏感，并表现出显著的安全性与选择性之间的权衡。更深的层沿安全方向保持良好的表征分离，但对激活引导基本不敏感，这区分了表征可分性与因果可控性。InternVL2.5-4B 在定性上（呈现出类似结果，摘要原文在此处截断）。

    arXiv:2610.32182v1 Announce Type: cross  Abstract: Vision-Language Models (VLMs) are widely deployed, yet the internal mechanisms behind their safety behavior remain poorly understood. We present a lightweight inference-time framework for localizing and manipulating safety-sensitive representations in frozen VLMs, without retraining or modifying parameters. On Qwen3-VL-4B, layer-wise activation divergence analysis reveals non-uniform safety geometry across depth. Causal interventions show that early vision layers support strong bidirectional refusal control through activation steering along a learned safety direction, with unsafe rejection spanning 6%-96%, whereas later layers respond more to activation magnitude scaling and exhibit a substantial safety-selectivity trade-off. Deeper layers remain well separated along the safety direction yet are largely insensitive to steering, distinguishing representational separability from causal controllability. InternVL2.5-4B shows qualitatively 
    
[^53]: 并非所有信息都值得询问：基于决策感知的大语言模型信息获取

    Not All Information Is Worth Asking For: Decision-Informed Information Seeking with LLMs

    [https://arxiv.org/abs/2610.32181](https://arxiv.org/abs/2610.32181)

    提出DIVA方法，让大语言模型基于证据的决策价值扣除成本后来评估每项检查是否值得进行，从而决定该查询什么信息以及何时停止，避免为无法改变决策或不值得的信息付费。

    

    大语言模型（LLM）越来越多地被用于在不完整信息下支持决策，在这种情况下，决定查询什么以及何时停止与最终答案同样重要。然而，LLM会陷入信息陷阱：它们会为那些无法改变决策或不值得其成本的检查（即昂贵的证据片段）付费，并且在证据已经充分的情况下仍然继续询问。为此，我们提出了决策感知的信息价值获取方法（Decision-Informed Value-of-information Acquisition，DIVA），这是一种基于证据决策价值的简单而有效的LLM信息获取程序。DIVA 1）让LLM通过简短的数值问题指定一个小型决策模型，即坏结果发生的可能性有多大，以及每项检查将如何改变这一概率；2）通过设想各候选检查的潜在结果能在扣除成本后对决策带来多大改善来评估每项候选检查的价值，当没有检查值得其成本时即停止。其变体DIVA-U进一步考虑了……

    arXiv:2610.32181v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used to support decisions under incomplete information, where deciding what to look up and when to stop matters as much as the final answer. However, LLMs fall into an information trap: They pay for checks, i.e., costly pieces of evidence, that cannot change their decision or are not worth their cost, and they keep asking when the evidence is already sufficient. To this end, we propose Decision-Informed Value-of-information Acquisition (DIVA), a simple yet effective procedure for LLM information seeking based on the decision value of evidence. DIVA 1) lets the LLM specify a small decision model, i.e., how likely a bad outcome is and how each check would shift this probability, through short numeric questions and 2) values each candidate check by how much its imagined outcomes improve the decision net of its cost, stopping when no check is worth its cost. Its variant, DIVA-U, further accounts 
    
[^54]: 日本某大学医院本地部署大语言模型智能体的LLENS平台：开发与评估研究

    The LLENS Platform for On-Premises Large Language Model Agents in a Japanese University Hospital: Development and Evaluation Study

    [https://arxiv.org/abs/2610.32151](https://arxiv.org/abs/2610.32151)

    研究团队在北海道大学医院的封闭网络内开发并部署了基于万亿参数开放权重模型Kimi K2.6的临床支持AI智能体平台LLENS，该模型在日本医师执业资格考试中取得480/500的高分，验证了本地化部署医疗AI智能体的可行性。

    

    摘要（arXiv:2610.32151v1，公告类型：交叉）：使用工具的大语言模型智能体可以分担占用医生时间的部分电子病历和案头文书工作，但在许多医院中，医疗信息系统运行在封闭网络内，因此模型和智能体框架也必须部署在其中。我们设计、实现并部署了LLENS——一个在北海道大学医院封闭网络内运行的临床支持AI智能体平台，部署于配备8块NVIDIA H200 GPU的单台主机上，采用开放权重模型Kimi K2.6（约1万亿参数），并使用基于Open WebUI重构的Web应用。该智能体框架包含10个组件，包括页面图像读取、全文附件输入、无需向量索引的路径寻址文档树、网络隔离沙箱、工件返回、生成输出的自检、子智能体以及已保存的操作流程。Kimi K2.6是从3个开放权重候选模型中选出的，并在2026年日本医师执业资格考试中获得了480/500的成绩。

    arXiv:2610.32151v1 Announce Type: cross  Abstract: Large language model agents that use tools could take on part of the electronic health record and desk work that occupies physicians' time, but in many hospitals the medical information systems run inside closed networks, so the model and the agent harness must also sit inside. We designed, implemented, and deployed LLENS, a clinical support AI agent platform that runs inside the closed network of Hokkaido University Hospital on one host with 8 NVIDIA H200 GPUs, the open-weight model Kimi K2.6 (~1 trillion parameters), and a web application rebuilt from Open WebUI. The harness has 10 components, including page image reading, full-text attachment input, a path-addressed document tree without a vector index, a network-isolated sandbox, artifact return, self-checking of generated output, subagents, and saved procedures. Kimi K2.6 was chosen from 3 open-weight candidates and scored 480/500 on the 2026 Japanese Medical Licensing Examination
    
[^55]: 从承诺到问责：尼泊尔政治文本中选举承诺的对比性AI建模与可视化

    From Promises to Accountability: Comparative AI Modeling and Visualization of Electoral Commitments in Nepali Political Texts

    [https://arxiv.org/abs/2610.32150](https://arxiv.org/abs/2610.32150)

    该研究创建了首个尼泊尔政治文本黄金标准数据集（Cohen's kappa = 0.777），并通过对比性AI建模与可视化方法对选举承诺进行结构化，以解决竞选承诺难以追踪和问责的问题。

    

    选举承诺挖掘是指对候选人或政党在竞选期间做出的选举承诺进行分析。然而，许多当选的候选人和政党，尤其是在南亚发展中国家，并未完全兑现选举前做出的承诺。即使在选举之后，通过选举宣言、海报、传单、视频及其他竞选材料呈现的承诺，也往往没有以系统化或透明的方式存档。因此，这些承诺往往会随时间流逝而逐渐被淡忘，导致难以对当选候选人的承诺进行问责。为解决这一长期存在的问题，本研究旨在开发一个能够对候选人在选举期间所做承诺进行结构化处理的系统。在这项工作中，我们创建了首个尼泊尔政治文本的黄金标准数据集（Cohen's kappa = 0.777），并评估了机器学习、深度学习……（摘要内容到此截断）

    arXiv:2610.32150v1 Announce Type: new  Abstract: Election Commitment Mining refers to the analysis of election commitments made by electoral candidates or political parties during election campaigns. However, many elected candidates and political parties, especially in developing South Asian countries, do not fully deliver on the promises they make before elections. Even after the elections, commitments presented through election manifestos, posters, pamphlets, videos, and other campaign materials are often not systematically or transparently archived. As a result, these promises tend to fade away over time, making it difficult to hold the elected candidates accountable for their commitments. To address this persistent issue, this research aims to develop a system capable of structuring the commitments made by candidates during election periods. In this work, we created the first gold-standard dataset of Nepali political texts (Cohen's kappa = 0.777) and evaluated machine learning, dee
    
[^56]: 解耦多语言 P(True) 自我评估中的问题语言、回答语言与验证语言

    Disentangling Question, Response, and Verifier Languages in Multilingual P(True) Self-Evaluation

    [https://arxiv.org/abs/2610.32144](https://arxiv.org/abs/2610.32144)

    该论文通过六语言因子设计首次将多语言自我评估中的问题语言、回答语言与验证语言解耦，发现三者分别影响评估的不同环节——准确性取决于基准、语言匹配提升区分度但不改善校准、验证语言独立偏移校准误差，同时强调评分器验证的重要性。

    

    当语言模型被问及自己的答案是否正确时，涉及三种语言：问题的语言、要求回答所用的语言，以及要求做出判断的验证指令所用的语言。多语言置信度评估很少将这三者区分开来。我们在一个六语言因子设计中，使用十个开源权重模型（3B-14B），在 LSQA、MMMLU 和 TriviaQA 数据集上独立变化这三种语言，并在不同验证语言下保持每个答案固定，通过首个 token 的 P(True) 探针测量置信度。这三种语言影响评估的不同部分：哪种语言决定准确性取决于所用的基准测试；将回答语言与问题语言或验证语言相匹配可以提升区分度但无法改善校准；而仅验证语言本身就能使校准误差偏移多达 0.04 而不改变准确性。评分器同样重要：一个未经验证的评分器会使 MMMLU 上英语与印地语的准确率差距从 0（摘要在此处被截断）

    arXiv:2610.32144v1 Announce Type: new  Abstract: When a language model is asked whether its own answer is correct, three languages are involved: that of the question, that requested for the answer, and that of the verifier instruction asking for the judgment. Multilingual confidence evaluations seldom separate the three. We vary them independently in a six-language factorial design with ten open-weight models (3B-14B) on LSQA, MMMLU and TriviaQA, holding each answer fixed across verifier languages and measuring confidence with a first-token P(True) probe. The three languages move different parts of the evaluation: which of them governs accuracy depends on the benchmark, matching the response language to the question or verifier language sharpens discrimination but not calibration, and the verifier language alone shifts calibration error by up to 0.04 without changing accuracy. The scorer matters as well: an unvalidated judge would have shrunk the MMMLU English-Hindi accuracy gap from 0
    
[^57]: SyncPro：面向主动式流媒体视频理解的同步推理与响应

    SyncPro: Synchronous Reasoning and Responding for Proactive Streaming Video Understanding

    [https://arxiv.org/abs/2610.32132](https://arxiv.org/abs/2610.32132)

    SyncPro提出了一种同步推理与响应的机制，解决了主动式流媒体视频理解中持续推理与及时响应难以协调的问题，避免了固定推理间隔导致的无谓更新或延迟响应。

    

    主动式流媒体视频理解要求助手能够追踪正在展开的事件，并在用户所请求的条件满足时做出响应。为了做出这些决策，助手必须积累相关证据，并在新观察到来时重新评估响应条件。现有工作为这一过程提供了互补的能力：主动式视频助手学会了何时响应，而流媒体视频推理方法则生成按计划安排的中间文本来组织观察结果并支持后续回答。这些能力共同为主动式推理奠定了基础，但如何协调持续进行的推理与及时的响应仍未得到充分探索。在直接适配的方式下，固定的推理间隔可能在没有相关变化发生时触发更新，或者将有用的更新推迟到下一个预定步骤。同时，共享同一条生成路径也将两个不同的需求耦合在一起：即为未来决策保留证据……

    arXiv:2610.32132v1 Announce Type: cross  Abstract: Proactive streaming video understanding requires assistants to track unfolding events and respond when a user's requested condition is met. To make these decisions, assistants must accumulate relevant evidence and reassess response conditions as new observations arrive. Existing work offers complementary capabilities for this process: proactive video assistants learn when to respond, while streaming video reasoning methods generate scheduled intermediate text to organize observations and support subsequent answers. Together, these capabilities provide a foundation for proactive reasoning, but how to coordinate ongoing reasoning with timely responses remains underexplored. In a direct adaptation, fixed reasoning intervals can trigger updates when nothing relevant has changed or postpone useful updates until the next scheduled step. Sharing one generation path also couples two distinct requirements: preserving evidence for future decisio
    
[^58]: 更短并不更便宜：声明式动画规范与自由格式代码在LLM生成解释中的受控比较

    Shorter Is Not Cheaper: A Controlled Comparison of Declarative Animation Specifications and Free-Form Code for LLM-Generated Explanations

    [https://arxiv.org/abs/2610.32129](https://arxiv.org/abs/2610.32129)

    本研究通过预先冻结假设与分析计划的受控实验发现，尽管声明式动画规范的输出比自由格式代码短25-48%，但“更短即更便宜、更可靠”的普遍预期并不成立。

    

    聊天助手越来越多地使用交互式、动画化的解释来回答概念性问题，这些解释要么以自由格式的HTML/JavaScript代码生成，要么以由固定运行时播放的声明式规范生成。声明式方式被普遍预期更便宜、更快、更可靠，因为其输出更短且受到约束。我们在假设和分析计划均预先冻结的受控比较中检验了这一预期。我们设计了ExplainSpec——一种按时间索引的声明式动画格式，以及一项观察契约，使每个生成的解释都可以被程序化检查；经过验证的、与节奏无关的检查器对来自五个领域的20个解释请求进行评分。三个Claude模型（360个任务）以及后续加入的开源权重Qwen3-14B（分别启用和禁用思考模式，240个任务）以两种格式回答了所有请求。对于每个模型，声明式规范的答案令牌数量缩短了25-48%（摘要在此处被截断）

    arXiv:2610.32129v1 Announce Type: new  Abstract: Chat assistants increasingly answer conceptual questions with interactive, animated explanations, generated either as free-form HTML/JavaScript or as declarative specifications that a fixed runtime plays. The declarative route is widely expected to be cheaper, faster and more reliable, because its outputs are shorter and constrained. We test this expectation in controlled comparisons whose hypotheses and analysis plans were frozen before the runs. We designed ExplainSpec, a time-indexed declarative animation format, and an observation contract that makes every generated explanation programmatically inspectable; validated, pacing-agnostic checkers score 20 explanation requests from five domains. Three Claude models (360 jobs) and, in a follow-up, the open-weight Qwen3-14B with thinking enabled and disabled (240 jobs) answered every request in both formats. For every model the specifications were 25-48% shorter in answer tokens (output exc
    
[^59]: 从噪声检索到决策证据：面向上下文学习的银证据示例

    From Noisy Retrieval to Decision Evidence: Silver Evidence Demonstrations for In-Context Learning

    [https://arxiv.org/abs/2610.32128](https://arxiv.org/abs/2610.32128)

    提出“银证据示例”方法，用LLM生成的、以目标为条件的证据摘要替代ICL示例中的原始噪声检索段落，从而将噪声检索转化为可靠的决策证据，提升RAG和少样本上下文学习在长噪声文本中的决策能力。

    

    当与决策相关的证据被埋没在冗长且充满噪声的上下文中时，RAG和检索增强的少样本ICL（上下文学习）可能会表现不佳。检索到的段落和示例可能包含冗余或干扰性信息。尽管专家标注的开发样例可以识别支持每项决策的文本，但这些标注通常仅被用作答案侧的推理依据，而非用于过滤噪声检索的监督信号。我们提出银证据示例，用生成的、以目标为条件的证据摘要替换ICL示例中的原始检索段落。对于每个有标注的开发样例，LLM在不接触专家标签或金标准证据的情况下生成银证据；随后这些证据与专家标签以及可用的金标准证据配对。在测试时，通过单次结构化调用从检索段落中生成银证据，然后预测最终标签和决策证据。在EI 2.0、Sc（摘要在此处被截断）等数据集上……

    arXiv:2610.32128v1 Announce Type: new  Abstract: RAG and retrieval-augmented few-shot ICL can struggle when decision-relevant evidence is buried in long, noisy context. Retrieved passages and demonstrations may contain redundant or distracting information. Although expert-labeled development examples may identify the text supporting each decision, these annotations are typically used only as answer-side rationales rather than as supervision for filtering noisy retrieval. We propose silver-evidence demonstrations, which replace raw retrieved passages in ICL examples with generated, target-conditioned evidence summaries. For each labeled development example, an LLM generates silver evidence without access to the expert label or gold evidence; this evidence is then paired with the expert label and gold evidence where available. At test time, a single structured call generates silver evidence from retrieved passages and then predicts the final label and decision evidence. Across EI 2.0, Sc
    
[^60]: RVL-CDIP中的数据质量：一项系统性审计

    Data Quality in RVL-CDIP: A Systematic Audit

    [https://arxiv.org/abs/2610.32126](https://arxiv.org/abs/2610.32126)

    本文对文档分类标准基准RVL-CDIP进行了系统性数据质量审计，发现约10.5%的标签错误和35.2%的训练-测试集重复问题，并在修复缺陷后重新基准测试25个模型，显示重训练模型准确率可提升0.9至2.9个百分点。

    

    RVL-CDIP是文档分类领域事实上的标准基准数据集，目前在其上报告的最高准确率已达98.8%。在本文中，我们对RVL-CDIP的数据质量问题进行了审计，发现其40万份文档中约有10.5%存在错误标签，且测试集中有35.2%的文档在训练集中存在重复或近似重复。仅凭页面上的识别代码就足以达到69%的分类准确率，而最先进的模型在去除这些代码后会损失11个百分点。仅在图像上训练的模型在处理来自其他来源的文档时会损失16到28个百分点，而OCR引擎的选择会导致准确率产生4.1到8.3个百分点的差异。我们发现近16,000份RVL-CDIP文档包含敏感个人数据，并为这些文档制作了去标识化版本。我们对25个模型进行了重新基准测试，在修复所有缺陷后，我们重新训练的13个模型的准确率比原始条件下提高了0.9到2.9个百分点；其中最准确的模型是L

    arXiv:2610.32126v1 Announce Type: cross  Abstract: RVL-CDIP is the de facto standard benchmark for document classification, and reported accuracy on it is now 98.8%. In this paper, we audit RVL-CDIP for data quality issues and find that approximately 10.5% of its 400,000 documents have erroneous labels and that 35.2% of the test set has a duplicate or near-duplicate in the training set. Identification codes on the pages are sufficient to classify at 69% accuracy, and a state-of-the-art model loses 11 points without them. Image-only models trained on RVL-CDIP lose 16 to 28 points on documents from other sources, and the choice of OCR engine changes accuracy by 4.1 to 8.3 points. We find that nearly 16,000 RVL-CDIP documents contain sensitive personal data, and produce de-identified versions of these documents. We re-benchmark 25 models, and with all defects remediated, the 13 that we retrain are 0.9 to 2.9 points more accurate than in the original condition; the most accurate of them, L
    
[^61]: 语义丰富，度量情感盲区：现代文本嵌入器中的几何盲点

    Semantically Rich, Metrically Affect-Blind: A Geometric Blind Spot in Modern Text Embedders

    [https://arxiv.org/abs/2610.32122](https://arxiv.org/abs/2610.32122)

    现代文本嵌入器的嵌入向量中几乎可以完美解码情感信息（线性探针准确率高达91.3%），但其度量空间（余弦相似度）几乎完全无法区分情感，导致依赖度量的检索、聚类、去重和RAG系统对这类信息“视而不见”。

    

    我们报告了现代文本嵌入器中的一种系统性分离现象：情感几乎可以完美地从其嵌入中解码出来，却几乎完全缺失于其度量之中。在涵盖2021至2026年的十一个冻结编码器上，包括2026年世代的模型以及40亿参数的LLM嵌入器Qwen3-Embedding-4B，线性探针能以高达91.3%的准确率恢复句子极性，而余弦相似度区分极性的差距仅为0.003至0.064。Qwen3-Embedding-4B是该组中探针性能最强的模型，其探针准确率达到91.3%，但检索MAP仅为64.3。由于检索、聚类、去重和检索增强生成（RAG）都是基于度量而非训练好的探针来运行的，这些系统对其编码器确实包含的信息实际上是“视而不见”的。一项任务条件控制实验表明，这反映的是预训练目标本身，而非提示方式的伪影：将Jina-v5以其分类任务为条件进行……（原文摘要在此处截断）

    arXiv:2610.32122v1 Announce Type: new  Abstract: We report a systematic dissociation in modern text embedders: sentiment is almost perfectly decodable from their embeddings, yet almost entirely absent from their metric. Across eleven frozen encoders spanning 2021 to 2026, including 2026-generation models and the 4B-parameter LLM embedder Qwen3-Embedding-4B, a linear probe recovers sentence polarity at up to 91.3% accuracy while cosine similarity separates polarity by a margin of only 0.003-0.064. Qwen3-Embedding-4B, the strongest probe in the panel, reaches 91.3% probe accuracy but only 64.3 retrieval MAP. Because retrieval, clustering, deduplication, and retrieval-augmented generation operate on the metric rather than on a trained probe, these systems are effectively blind to information their encoder demonstrably contains. A task-conditioning control suggests this reflects the pretraining objective rather than a prompting artifact: conditioning Jina-v5 on its classification task rath
    
[^62]: EDWARD-NMT：面向神经机器翻译的弹性维度、自适应路由与双一致性方法

    EDWARD-NMT: Elastic Dimensions With Adaptive Routing and Dual-consistency for Neural Machine Translation

    [https://arxiv.org/abs/2610.32115](https://arxiv.org/abs/2610.32115)

    EDWARD-NMT 提出了一种结合套娃式弹性维度、带可学习阈值的自适应 sigmoid 路由器以及双一致性目标的新型 MoE 架构，实现“一次训练、随处部署”，使神经机器翻译模型能够在资源受限设备上灵活部署。

    

    稀疏混合专家模型（Sparse Mixture-of-Experts, MoE）是扩展神经机器翻译（NMT）并管理语言异质性的强大范式。然而，实际的局限性阻碍了其在资源受限设备上的部署：整个专家池必须保留在内存中，且每个 token 的计算预算在训练期间被僵化的 top-k 路由静态固定。我们提出了 EDWARD-NMT（面向 NMT 的弹性维度、自适应路由与双一致性），这是一种新颖的 MoE 编码器-解码器架构，专为“一次训练、随处部署”的范式而设计，可在推理时提供一族可部署的子网络。EDWARD-NMT 融合了三项关键创新：（i）套娃式（Matryoshka 风格）弹性维度，可灵活降低内存占用；（ii）具有可学习的、任务条件化阈值的自适应 sigmoid 路由器，使每个 token 能够动态决定其激活的专家数量；（iii）双一致性目标函数，以最大化性能……（原文摘要在此处截断）

    arXiv:2610.32115v1 Announce Type: new  Abstract: Sparse Mixture-of-Experts (MoE) is a powerful paradigm for scaling Neural Machine Translation (NMT) and managing language heterogeneity. However, practical limitations hinder its deployment on resource-constrained devices: the entire expert pool must remain in memory, and the per-token compute budget is statically fixed during training by rigid top-k routing. We introduce EDWARD-NMT (Elastic Dimensions With Adaptive Routing and Dual-consistency for NMT), a novel MoE encoder-decoder architecture designed for a train once, deploy anywhere paradigm that exposes a family of deployable sub-networks at inference time. EDWARD-NMT integrates three key innovations: (i) Matryoshka-style elastic dimensions to flexibly reduce the memory footprint; (ii) an adaptive sigmoid router with a learned, task-conditioned threshold, allowing each token to dynamically determine its active expert count; and (iii) a dual-consistency objective to maximize the perf
    
[^63]: Tampered ML：评估大语言模型对机器学习研究的判断力

    Tampered ML: Evaluating LLM Judgment of Machine Learning Research

    [https://arxiv.org/abs/2610.32114](https://arxiv.org/abs/2610.32114)

    该论文提出Tampered ML基准，通过将真实机器学习论文的实验提案与人为引入方法论缺陷的最小修改版本配对，来评估大语言模型识别研究方法缺陷的能力，发现前沿模型虽优于随机水平但远未达到可靠程度。

    

    加速AI对齐研究进展的一种途径是将对齐研究本身的部分工作自动化。这要求系统能够在难以监督的研究任务上可靠运行，包括识别研究提案中的缺陷，以及评估一个提议的实验是否能支持其预期结论。为了评估这一能力，我们引入了Tampered ML，这是一个包含1,858个条目对的基准，每个条目对由一个从已发表的机器学习论文中重构的实验提案，以及一个经过最小化编辑、引入了针对性方法论缺陷的反事实版本组成。模型必须在两种呈现顺序下识别出方法论上更合理的提案。一项针对分层样本的人类验证研究估计，86%的条目是有效的（95%置信区间：81-91%）。前沿模型的表现高于随机水平，但仍远未达到可靠程度（其中Opus 5达到77.9%的准确率）。它们对呈现顺序高度敏感，并且f……（摘要在此处被截断）

    arXiv:2610.32114v1 Announce Type: cross  Abstract: One approach to accelerating progress on AI alignment is to automate parts of alignment research itself. Doing so requires systems that can perform reliably on hard-to-supervise research tasks, including identifying flaws in research proposals and evaluating whether a proposed experiment supports its intended claim. To evaluate this capability, we introduce Tampered ML, a benchmark of 1,858 item-pairs, each consisting of an experiment proposal reconstructed from a published machine learning paper and a minimally edited counterfactual that introduces a targeted methodological flaw. Models must identify the more methodologically sound proposal in both presentation orders. A human validation study on a stratified sample estimates that 86% of items are valid (95% CI: 81-91%). Frontier models perform above chance but remain far from reliable (with Opus 5 achieving 77.9% accuracy). They are highly sensitive to the order of presentation and f
    
[^64]: REACT：思维链轨迹的表征早期分析与控制

    REACT: Representational Early Analysis and Control of Chain-of-Thought Trajectories

    [https://arxiv.org/abs/2610.32113](https://arxiv.org/abs/2610.32113)

    REACT 提出一种从成对安全与不安全轨迹中学习安全方向、并一次性施加于思维链早期隐藏状态的激活干预方法，发现最强控制增益集中在思维链的前两到三个 token 窗口，即使后续状态更能预测风险。

    

    思维链生成提供了一种在推理模型完成回答之前进行干预的机会。但是，风险最容易被检测到的时间点是否也能提供最有效的控制？我们通过 REACT 来研究这个问题，这是一种条件性激活干预方法，它从成对的安全与不安全轨迹中学习一个安全方向，并将其一次性应用于早期的思维链隐藏状态。模型权重保持冻结，所有后续 token 的生成不再进行进一步编辑。为了研究时机问题，我们在前五个思维链位置上分别训练方向，并使用无门控编辑配合编辑后的完整重新生成进行评估。在三个蒸馏推理模型和三个生成种子上，最强的增益集中在两到三个 token 的早期窗口内；在 DeepSeek-R1-Distill-Llama-8B 上，效果在第四和第五个 token 处降至接近于零，尽管较晚的状态可能更能预测风险。

    arXiv:2610.32113v1 Announce Type: cross  Abstract: Chain-of-thought (CoT) generation offers an opportunity to intervene before a reasoning model completes its answer. But does the point at which risk is easiest to detect also offer the most effective control? We study this question with REACT, a conditional activation intervention that learns a safety direction from paired safe and unsafe trajectories and applies it once to an early CoT hidden state. The model weights remain frozen, and all subsequent tokens are generated without further edits. To study timing, we train directions separately at each of the first five CoT positions and evaluate ungated edits with full post-edit regeneration. Across three distilled reasoning models and three generation seeds, the strongest gains concentrate in a two- to three-token early window; on DeepSeek-R1-Distill-Llama-8B, the effect falls to approximately zero at the fourth and fifth tokens, even though later states can be more predictive of risk. 
    
[^65]: 上下文还是知识：大语言模型如何归因自己的答案

    Context or Knowledge: How LLMs Attribute Their Own Answers

    [https://arxiv.org/abs/2610.32109](https://arxiv.org/abs/2610.32109)

    该论文通过对KV缓存的干预实验发现，大语言模型对答案来源的自我归因主要由问答序列（Q+A）的缓存表征决定而非文档内容，并且可以通过沿预拟合方向引导Q+A值以近乎100%的成功率系统性地翻转来源报告。

    

    语言模型可以基于提供的文档或其参数化知识来回答问题。当随后被问及答案来自何处时，是什么决定了它们的来源报告？我们通过在答案生成之后、来源问题提出之前对模型的键值（KV）缓存进行干预来研究这一问题。在一个使用四个模型（7B-32B）的可控事实任务中，更改或删除文档的缓存从不改变来源报告。相比之下，在问题与答案标记完全相同的运行之间移植问答（Q+A）缓存则会转移来源报告：每个符合条件的报告都跟随供体，而接收方的文档保持不变。在两个较小的模型上，沿着在任何来源问题出现之前拟合出的方向对Q+A值进行引导，在固定强度下可以在四种文档条件中逆转98.9-100%的报告，而范数匹配的随机方向最多只翻转1.1%。键和值沿拟合方向的分量……（原文截断）

    arXiv:2610.32109v1 Announce Type: new  Abstract: Language models can answer a question from a supplied document or from their parametric knowledge. When later asked where the answer came from, what determines their source report? We study this by intervening on the model's key-value (KV) cache after the answer is generated and before the source question is asked. On a controlled factual task with four models (7B-32B), changing or removing the document's cache never changes the report. In contrast, transplanting the question-and-answer (Q+A) cache between runs with identical question and answer tokens transfers it: every eligible report follows the donor, while the recipient's document is unchanged. On the two smaller models, steering Q+A values along directions fitted before any source question reverses 98.9-100% of reports across four document conditions at a fixed strength, while norm-matched random directions flip at most 1.1%. The components of keys and values along fitted directio
    
[^66]: 超越说什么：基于内心想法的狼人杀智能体主动发言时机决策

    Beyond What to Say: Proactive Utterance Timing with Inner Thoughts for Werewolf Agents

    [https://arxiv.org/abs/2610.32107](https://arxiv.org/abs/2610.32107)

    本研究针对狼人杀游戏的竞争性和不完全信息特点，重新设计了基于“内心想法”框架的LLM智能体系统，使其能够自主判断发言时机并决定发言内容。

    

    参与对话不仅涉及决定说什么，还涉及是否发言。在狼人杀游戏这类参与者利益相互冲突的环境中，这种判断尤为重要，因为发言与沉默都会受到他人的解读和评判。因此，对于玩狼人杀游戏的大语言模型（LLM）智能体而言，确定何时发言以及说什么都至关重要。在本研究中，我们介绍了为AIWolfDial共享任务“随时发言赛道”开发的狼人杀AI智能体。我们的系统建立在现有的“内心想法”框架之上，该框架通过生成内部想法并评估每个想法在当下表达的价值来确定对话中的发言时机。由于现有方法是为一般对话设计的，我们专门针对竞争环境和不完全信息设定对该系统进行了重新设计。

    arXiv:2610.32107v1 Announce Type: new  Abstract: Participating in a conversation involves not only deciding what to say but also whether to speak at all. This judgment carries even greater significance in environments like the Werewolf game, where participants' interests conflict, because both utterances and silence are subject to interpretation and assessment by others. Therefore, determining when to speak, as well as what to say, is crucial for an LLM agent playing the Werewolf game. In this study, we introduce a Werewolf AI agent developed for the Anytime Utterance Track of the AIWolfDial shared task. Our system builds on the existing Inner Thoughts framework, which determines utterance timing during conversations by generating internal thoughts and evaluating the value of each thought for expression at that moment. Since existing methods were designed for general dialogue, we have redesigned the system specifically for the competitive environment and incomplete information setting 
    
[^67]: 将Transformer理论付诸检验

    Putting Transformer Theories to the Test

    [https://arxiv.org/abs/2610.32104](https://arxiv.org/abs/2610.32104)

    该论文将七种关于Transformer能力的理论预测统一到同一矩阵中，并通过受控实验验证，发现没有任何单一理论能完全解释Transformer在不同计算任务上的表现。

    

    目前已发展出若干形式化理论来刻画Transformer架构的能力与局限，每种理论采用不同的抽象方式来捕捉该架构的不同特征。由于底层假设和概念框架的差异，这些理论对于Transformer认为哪些任务容易、哪些任务困难的预测可能并不一致。我们梳理了七种关于Transformer能力的理论（Star-free、AC⁰、TC⁰、C-RASP、MPC、BAPO和Sensitivity），并将其结果转化为一个统一的预测矩阵，覆盖一系列具有不同计算结构的任务。我们提供了新的理论结果来补全该矩阵。随后，我们通过受控实验来评估这些理论预测，实验测量了分布内样本复杂度、长度泛化能力以及预训练语言模型的性能。我们发现，没有任何单一理论能够解释……（摘要原文不完整）

    arXiv:2610.32104v1 Announce Type: cross  Abstract: Several formal theories have been developed to characterize the capabilities and limitations of the transformer architecture, each employing different abstractions that capture different features of the architecture. Due to the differences in their underlying assumptions and conceptual frameworks, these theories can disagree in their predictions about what tasks transformers find easy or hard. We map out seven theories of transformer capability (Star-free, $\mathsf{AC}^0$, $\mathsf{TC}^0$, C-RASP, MPC, BAPO, and Sensitivity) and translate their results into a common matrix of predictions across a suite of tasks covering different computational structures. We provide new theoretical results to complete this matrix. We then evaluate theoretical predictions through controlled experiments measuring in-distribution sample complexity, length generalization, and pretrained language model performance. We find that no single theory explains the
    
[^68]: 面向大语言模型驱动智能体的行为分析

    Behaviour analysis for large language model-driven agents

    [https://arxiv.org/abs/2610.32087](https://arxiv.org/abs/2610.32087)

    提出了一个将数据驱动的行为发现、可解释的依赖关系分析与实验验证（含提示引导）相结合的通用框架，用于理解大语言模型驱动智能体在轨迹中如何组织行为及其与情境和结果的关系。

    

    理解语言智能体的行为需要学习动作如何在整个轨迹中被组织，以及这种组织如何与情境和结果相关联。我们提出了一个通用框架，将数据驱动的行为发现与分析及实验研究相结合。从记录的活动中开发的类别定义了所要测量的内容。分析保留了行为在过程中的位置，并将其关系表达为可解释的假设。统计评估和在保留情境中的测试检验了证据的强度和范围。随后，包括提示引导在内的实验测试了行为和结果在刻意变化的条件下如何改变。一项关于迭代式自我反思的研究展示了行为发现、依赖关系学习和提示比较如何能够相互启发。我们针对不同长度、事件顺序和智能体角色的轨迹发展了这一方法，并识别了在编码……（摘要在此处截断）

    arXiv:2610.32087v1 Announce Type: new  Abstract: Understanding language-agent behaviour requires learning how actions are organised across a trajectory and how this organisation relates to context and outcomes. We propose a general framework that connects data-driven behavioural discovery with analysis and experimental investigation. Categories developed from recorded activity define what is measured. Analyses preserve the behaviours' positions in the process and express their relationships as interpretable hypotheses. Statistical assessment and tests in held-out contexts examine the strength and scope of the evidence. Experiments, including prompt steering, then test how behaviour and outcomes change under deliberately varied conditions. A study of iterative self-reflection illustrates how discovery, dependency learning and prompt comparisons can inform one another. We develop this approach for trajectories of varying length, event order and agent roles, and identify advances in codin
    
[^69]: 集中的因果影响，分布式的诊断信息：大型音频-语言模型中的认知障碍决策

    Concentrated Causal Influence, Distributed Diagnostic Information: Cognitive Impairment Decisions in Large Audio-Language Models

    [https://arxiv.org/abs/2610.32061](https://arxiv.org/abs/2610.32061)

    该论文通过因果干预识别出大型音频-语言模型解码器MLP中的“认知障碍对比神经元”（CICNs），首次从因果层面证明认知障碍决策由少量神经元集中控制，而诊断信息以分布式形式存在于模型表征中，且这一机制在不同模型间和跨语料库上具有一定泛化性。

    

    大型音频-语言模型（LALMs）能够从语音中评估认知障碍（CI），但其回答既无法揭示哪些内部组件控制着这一决策，也无法揭示其表征中蕴含多少可获取的认知障碍信息。我们通过认知障碍对比神经元（CICNs）来研究这两个问题，这类神经元是通过解码器MLP中的类对比激活频率筛选得到的。在八个开源LALMs和两个语料库（Pitt与PROCESS-2）上，我们将CICNs与大小和层匹配的随机掩码以及激活匹配的对照组进行因果测试。在符合资格标准的五个模型中，有三个模型的少量CICNs被消融后会引起大幅的、面向特定类别的预测偏移；从语音中识别出的掩码也能控制仅基于转写文本的决策，且四个主要Pitt掩码中有三个在PROCESS-2上保持了特异性。位置受限的干预实验将这种影响定位在最终答案位置之前，并在语音或转写文本证据出现之后最为强烈。

    arXiv:2610.32061v1 Announce Type: cross  Abstract: Large audio-language models (LALMs) can assess cognitive impairment (CI) from speech, yet their answers reveal neither which internal components control the decision nor how much CI information is accessible in their representations. We investigate both questions through CI-contrast neurons (CICNs), selected by class-contrastive activation frequency in decoder MLPs. Across eight open LALMs and two corpora (Pitt and PROCESS-2), we causally test CICNs against size- and layer-matched random masks and activity-matched controls. In three of the five models that meet our eligibility criteria, ablation of small CICN sets causes large, class-directed prediction shifts; speech-identified masks also control transcript-only decisions, and three of four primary Pitt masks retain specificity on PROCESS-2. Position-restricted interventions locate the influence before the final answer position, most strongly after the speech or transcript evidence un
    
[^70]: 你有多信任一只狗？量化推理语言模型中的提示遵循行为

    How Much Do You Trust A Dog? Quantifying Hint-Following in Reasoning Language Models

    [https://arxiv.org/abs/2610.32055](https://arxiv.org/abs/2610.32055)

    该研究发现大语言模型对提示来源（如教授、朋友甚至狗）的答案偏移遵循可预测的指数倾斜结构，且偏移强度与来源可靠性基本无关，揭示了模型谄媚行为未被来源可靠性校准的问题。

    

    大型语言模型（LLM）常常表现出谄媚行为，即使在客观的事实和推理任务上，也会为了迎合用户的建议而改变自己的答案。然而，这种影响的定量形式——以及它是否与信息来源的可靠性相校准——仍然知之甚少。在本工作中，我们通过MMLU问题上的提示诱导答案偏移来研究这一行为，评估形如“[来源]认为答案是(X)”的提示。我们在三个开源模型系列（Qwen、Gemma和SmolLM）上系统地改变提示所归属的来源（教授、朋友或狗）。我们表明，提示引起的变化遵循一个可预测的结构，该结构可以用指数倾斜很好地刻画：提示在被提示答案的对数几率上引起近似恒定的均值偏移，且问题与目标的易感性在不同来源之间很大程度上是共享的。为了探索这种模式如何产生，我们引入了一个形式化的……

    arXiv:2610.32055v1 Announce Type: new  Abstract: Large language models (LLMs) often exhibit sycophancy, altering their answers to align with user suggestions even on objective factual and reasoning tasks. Yet the quantitative form of this influence--- and whether it is calibrated to source reliability--- remains poorly understood. In this work, we investigate this behavior through hint-induced answer shifts on MMLU questions, evaluating prompts of the form ``[Source] thinks the answer is (X)''. We systematically vary the source to which the hint is attributed (a professor, a friend, or a dog) across three open-source model families, Qwen, Gemma, and SmolLM. We show that the change induced by the hint follows a predictable structure well-characterized by an exponential tilt: the hint induces an approximately constant mean shift in the hinted answer's log-odds, with question--target susceptibility largely shared across sources. To explore how this pattern can arise, we introduce a formal
    
[^71]: X-Dubbing：时间与视觉约束下的语音到语音翻译

    X-Dubbing: Speech-to-Speech Translation under Temporal and Visual Constraints

    [https://arxiv.org/abs/2610.32042](https://arxiv.org/abs/2610.32042)

    X-Dubbing 将跨语言配音形式化为时间感知的语音到文本翻译与视频感知的文本到语音合成的联合任务，在无需成对配音数据的情况下即可生成与原视频时间同步的目标语言配音。

    

    跨语言配音旨在将源语音转换为目标语言的语音，并保持与原视频的同步。这项任务的关键在于生成忠实的翻译，同时使其能够在源语音的时间限制内自然地朗读出来，而这并非易事，因为不同语言在时长、节奏和发音方式上存在差异。在本文中，我们提出 X-Dubbing，将这一任务形式化为时间感知的语音到文本翻译与视频感知的文本到语音的联合问题。具体而言，我们的方法由一个生成时间结构化脚本的配音翻译器和一个将脚本与源视频同步渲染的配音演员组成，二者均基于与语言无关的时间预算。我们进一步利用来自合成语音的配音感知反馈对翻译器进行优化，鼓励其生成能够产生良好同步配音的脚本。该设计无需成对的配音数据，因为这两个组件均在语音-文本

    arXiv:2610.32042v1 Announce Type: new  Abstract: Cross-lingual dubbing aims to convert source speech into target-language speech that remains synchronized with the original video. The key to this task is to produce a faithful translation that is naturally speakable within the source timing, which is non-trivial because languages differ in duration, rhythm, and articulation. In this paper, we propose X-Dubbing, which formulates the task as a joint problem of time-aware speech-to-text translation and video-aware text-to-speech. Specifically, our method consists of a Dubbing Translator that generates temporally structured scripts and a Voice Actor that renders them in synchrony with the source video, both grounded in a language-agnostic timing budget. We further refine the translator with dubbing-aware feedback from the synthesized speech, encouraging scripts that yield well-synchronized dubbing. This design requires no paired dubbing data, as the two components are trained on speech-text
    
[^72]: CRM-Agent-Bench：面向有状态企业环境中工具使用型智能体的基准测试

    CRM-Agent-Bench: Benchmarking Tool-Using Agents in Stateful Enterprise Environments

    [https://arxiv.org/abs/2610.32004](https://arxiv.org/abs/2610.32004)

    该论文提出 CRM-Agent-Bench，一个包含 555 个有状态多轮任务、覆盖九大业务领域的执行型基准测试，从结果、轨迹、运营可行性和政策合规四个维度联合评估语言智能体在真实企业 CRM 环境中执行多步骤业务流程的真实能力。

    

    arXiv:2610.32004v1 公告类型：cross 摘要：在真实应用中部署语言智能体，仅仅给出正确答案是远远不够的：智能体必须正确执行多步骤业务流程、遵守领域特定规则，并在跨轮对话中与用户持续交互——“说到”与“做到”之间的差距，正是客户满意与真实损失之间的差距。我们认为，一个面向可部署智能体的基准测试应联合评估四个维度：智能体达成的结果、所走的轨迹、其运营可行性以及政策合规性。客户关系管理（CRM）领域兼具可变记录、运营规则和重大后果影响，恰好能同时检验这四个方面。我们提出了 CRM-Agent-Bench，这是一个以真实执行为基础的基准测试，包含 555 个有状态的多轮任务，覆盖九个业务领域和 36 个能力标签。每个任务都提供一份任务专属的可执行契约，该契约源自对动作、参数、执行顺序、响应、目标状态和受保护状态的检查；成功意味着……

    arXiv:2610.32004v1 Announce Type: cross  Abstract: Deploying language agents in real applications demands more than the right answer: an agent must execute a multi-step business process correctly, follow domain-specific rules, and interact with a user across a dialogue, where the gap between saying and doing is the gap between a satisfied customer and a real loss. We argue that a deployable-agent benchmark should jointly assess four facets: the outcome an agent reaches, the trajectory it takes, its operational viability, and its policy compliance. Customer relationship management (CRM), where mutable records meet operational rules and consequential effects, exercises all four. We introduce CRM-Agent-Bench, an execution-grounded benchmark of 555 stateful, multi-turn tasks across nine business domains and 36 capability tags. Each task supplies a task-specific executable contract drawn from checks over actions, arguments, order, response, target state, and protected state; success is the 
    
[^73]: DraMaze：具有前瞻记忆的多智能体流水线，用于多样化微短剧剧本生成

    DraMaze: A Multi-Agent Pipeline with Future-Aware Memory for Diverse Microdrama Script Generation

    [https://arxiv.org/abs/2610.32002](https://arxiv.org/abs/2610.32002)

    提出DraMaze多智能体流水线，通过套路记忆引导的进化树搜索和基于未来叙事目标的反向叙事依赖树，实现既多样又保持长程连贯的微短剧剧本生成，并发布双语评测基准MiDramaBench。

    

    微短剧的快速增长增加了对高质量剧本制作的需求。使用大语言模型进行长篇微短剧剧本生成，需要同时具备多样化的情节推进和长程叙事连贯性。我们的观察发现，随着生成过程的推进，候选续写越来越集中于少数叙事机制，而传统检索方法可能强化套路化复用，并忽略对后续情节至关重要的历史信息。为应对这些挑战，我们提出了DraMaze，一个通过套路记忆引导的进化树搜索和变异来探索多样化因果路径的多智能体流水线。该方法进一步构建了反向叙事依赖树，基于未来叙事目标来组织、保留和检索历史记忆。我们还推出了MiDramaBench，一个用于评估不同剧本长度下剧本逻辑与创意质量的双语基准，并配有因果模式（causal-pattern）相关评测……

    arXiv:2610.32002v1 Announce Type: new  Abstract: The rapid growth of microdramas has increased demand for high-quality script production. Long-form microdrama script generation with LLMs requires both diverse plot progression and long-range narrative coherence. Our observations reveal that candidate continuations increasingly concentrate on fewer narrative mechanisms as generation progresses, while conventional retrieval may reinforce trope reuse and overlook historical information essential to future plots. To address these challenges, we propose DraMaze, a multi-agent pipeline that explores diverse causal paths through trope memory-guided evolutionary tree search and mutation. It further constructs a Backward Narrative Dependency Tree to organize, retain, and retrieve historical memories based on future narrative goals. We also introduce MiDramaBench, a bilingual benchmark for evaluating script logic and creative quality across different script lengths, together with causal-pattern m
    
[^74]: AusClinBench：基于澳大利亚现行临床政策的大语言模型确定性评分基准

    AusClinBench: a deterministically scored benchmark of large language models against current Australian clinical policy

    [https://arxiv.org/abs/2610.31993](https://arxiv.org/abs/2610.31993)

    该研究构建了基于澳大利亚现行临床政策、采用精确匹配确定性评分的142题基准AusClinBench，对28个大语言模型的测试显示盲测准确率为51.5%–92.7%，且86%的错误在提供相关条款后即可纠正，表明模型失败主要源于规则记忆而非规则应用能力。

    

    AI模型已经在澳大利亚临床软件中为临床决策提供参考并起草病历记录，而从业者通常不会被告知使用的是哪个模型。现有的医学基准是为其他司法管辖区编写的。我们根据澳大利亚指南和规范编写了142个问题，涵盖五个领域（PBS药品福利计划授权、MBS医疗保险计费、Ahpra执业义务、分诊危险信号、免疫接种与筛查），每个问题都有一个可通过精确匹配进行确定性评分的可验证答案。每个问题均在盲测状态下以及提供相关 governing 条款的情况下分别运行，从而将“无法回忆当前规则”与“无法应用规则”区分开来。我们在2026年7月至10月期间测试了28个模型。盲测准确率介于51.5%至92.7%之间，得分最高的模型之间难以区分优劣。在791个模型-问题失败对中，684个（86%）在盲测时错误，而在提供条款后回答正确。自信错误率介于6.6%至47.8%之间。在五个指南未给出明确方向（原文在此截断）的免疫接种问题上……

    arXiv:2610.31993v1 Announce Type: cross  Abstract: AI models already inform clinical decisions and draft notes in Australian clinical software, usually without the practitioner being told which model is used. Existing medical benchmarks are written for other jurisdictions. We wrote 142 questions from Australian guidelines and schedules across five domains (PBS authority, MBS billing, Ahpra obligations, triage red flags, immunisation and screening), each with one verifiable answer scored by exact match. Each question ran blind and with the governing clause supplied, which separates failure to recall the current rule from failure to apply it. We tested 28 models between July and October 2026. Blind accuracy ranged from 51.5% to 92.7%, and the highest-scoring models could not be separated. Of 791 failed model-question pairs, 684 (86%) were wrong blind and right with the clause. Confident-error rates ranged from 6.6% to 47.8%. On five immunisation questions where the guideline gives no dir
    
[^75]: 有益智能体问题：当LLM智能体间的涌现合作绕过预期工作流时

    The Helpful Agent Problem: When Emergent Cooperation Among LLM Agents Bypasses Intended Workflows

    [https://arxiv.org/abs/2610.31985](https://arxiv.org/abs/2610.31985)

    该论文揭示了“有益智能体问题”：即使没有恶意或共谋，追求各自目标的良性LLM智能体也会在审查机制失效时自发合作、代替彼此执行操作，从而在合法权限范围内绕过预期的监督工作流，构成一种新型AI安全风险。

    

    LLM智能体越来越多地被部署在协作环境中：它们在追求各自任务的同时共享资源，并处于审查或监督流程之下。这类环境不仅在智能体具有恶意时会产生安全风险，在它们仅仅是想提供帮助时同样会产生安全风险。以往的研究主要关注对抗性或明显的共谋行为；我们则表明，追求各自目标的良性智能体可以自发地进行合作，从而绕过预期的工作流。我们将这种现象称为“有益智能体问题”。我们研究了长时程的共享资源环境：其中智能体有时会依赖同伴所拥有的资源，而跨所有者的更改通常由审查者来调解。当审查机制变得不可用时，智能体便开始代替彼此进行更改，尽管它们之间没有共享奖励，也没有明确的合作激励。至关重要的是，这些操作仍然处于每个智能体自身的本地权限范围内，同时却规避了预期的监督流程。

    arXiv:2610.31985v1 Announce Type: new  Abstract: LLM agents are increasingly deployed in collaborative settings where they pursue separate tasks while sharing resources and operating under review or oversight processes. Such settings create safety risks not only when agents are malicious, but also when they are simply trying to help. Prior work has largely studied adversarial or explicitly collusive behavior; we show that benign agents pursuing separate objectives can spontaneously cooperate in ways that bypass intended workflows. We call this the \emph{helpful agent problem}. We study long-horizon shared-resource environments in which agents sometimes depend on resources owned by their peers and cross-owner changes are normally mediated by a reviewer. When review becomes unavailable, agents begin making changes on one another's behalf despite having no shared reward or explicit incentive to cooperate. Crucially, these actions remain within each agent's local permissions while circumve
    
[^76]: 知道该信任谁还不够：重复多方LLM谈判中的信任-协议鸿沟

    Knowing Whom to Trust is not Enough: The Trust--Agreement Gap in Repeated Multi-Party LLM Negotiation

    [https://arxiv.org/abs/2610.31971](https://arxiv.org/abs/2610.31971)

    本文提出了一个重复多方、多议题的LLM谈判框架，发现LLM智能体虽然能从历史交互中形成有用的信任估计并识别暗中阻挠协议的对手，但知道该信任谁并不足以保证达成协议，从而揭示了“信任-协议鸿沟”。

    

    大型语言模型（LLM）智能体越来越多地代表个人和组织参与谈判，但它们仍然容易受到操纵，尤其是在重复的多方互动中。经典的多智能体系统通过显式的信任与声誉模型来应对这一挑战，而现有的LLM谈判研究大多将信任视为一种涌现现象。我们研究了LLM智能体能否从先前的交互中形成有用的信任估计，并利用它们来改善未来的谈判结果。我们引入了一个重复的、多方的、多议题的谈判框架，其中反复出现的智能体参与一系列谈判博弈，同时一名参与者暗中试图阻止达成协议。每次博弈结束后，智能体会评估其对手，并维护延续至后续谈判的成对信任估计。我们比较了信任感知与信任无关的智能体，并分析了……（原文摘要在此处被截断）

    arXiv:2610.31971v1 Announce Type: new  Abstract: Large language model (LLM) agents increasingly participate in negotiations on behalf of individuals and organizations, yet they remain vulnerable to manipulation, particularly in repeated multi-party interactions. Classical multi-agent systems address this challenge through explicit trust and reputation models, whereas existing work on LLM negotiation largely treats trust as an emergent phenomenon. We investigate whether LLM agents can form useful trust estimates from prior interactions and leverage them to improve future negotiation outcomes. We introduce a repeated, multi-party, multi-issue negotiation framework in which recurring agents participate in a sequence of negotiation games while one participant covertly attempts to prevent agreement. After each game, agents assess their counterparts and maintain pairwise trust estimates that carry over to subsequent negotiations. We compare trust-aware and trust-agnostic agents and analyze h
    
[^77]: 基于词性签名的上下文依存句法分析示例检索

    POS-Signature Demonstration Retrieval for In-Context Dependency Parsing

    [https://arxiv.org/abs/2610.31966](https://arxiv.org/abs/2610.31966)

    该论文提出了一种基于加权词性统计的"词性签名"示例检索策略，为上下文依存句法分析选取更相关的演示示例，无需微调即在两个开源语言模型上超越了语义相似度检索等基线方法，取得了最高的标注依存分数。

    

    上下文学习的示例通常通过语义相似度来检索，但这种方法在结构化预测任务中可能是相关性的一个很差的代理指标。我们研究了通用依存分析（Universal Dependencies parsing）中的示例选择问题，提出了一种词性签名（POS-signature）策略：使用加权词性统计来表示句子，并从带标注的训练集中检索三个长度兼容、附带标准解析的示例。我们在 UD 英语网页树库（English Web Treebank）测试集的全部 2,077 个句子上，对两个未经微调的开源权重语言模型进行了评估，将该策略与匹配示例数量和格式的固定三示例基线、语义检索三示例基线，以及零样本、思维链和批判式改进系统进行了比较。词性签名检索在所有被测试的提示系统中为两个模型都取得了最高的标注依存分数（72.4 和 73.2），分别比语义检索高出 1.9 和 2.7 个百分点；配对自助法的 95% 置信区间排除了零差异（原文此处截断）。

    arXiv:2610.31966v1 Announce Type: new  Abstract: Demonstrations for in-context learning are often retrieved by semantic similarity, which may be a poor proxy for relevance in structured prediction. We study demonstration selection for Universal Dependencies parsing with a POS-signature policy that represents sentences with weighted part-of-speech statistics and retrieves three length-compatible, gold-parsed examples from the labeled training split. We evaluate two open-weight language models without fine-tuning on all 2,077 sentences of the UD English Web Treebank test set, comparing the policy with shot- and format-matched fixed and semantic three-shot baselines, as well as zero-shot, chain-of-thought, and critique-refinement systems. POS-signature retrieval yields the highest labeled attachment score among the tested prompting systems for both models (72.4 and 73.2), outperforming semantic retrieval by 1.9 and 2.7 points; paired-bootstrap 95% confidence intervals exclude zero for bot
    
[^78]: 哪份答案出自十七岁少年之手？对六个前沿语言模型中作者年龄误判、自信心校准失准与非纠错性自我解释的行为学案例研究

    Which Answer Did the Seventeen-Year-Old Write? A Behavioral Case Study of Author-Age Misattribution, Miscalibrated Confidence, and Non-Corrective Self-Explanation in Six Frontier Language Models

    [https://arxiv.org/abs/2610.31946](https://arxiv.org/abs/2610.31946)

    本研究通过1,081次API会话、3,225份回复的行为学案例研究，发现六个前沿语言模型在将手写答案匹配到作者年龄时表现接近随机却报告73-88%的高置信度，17岁少年的答案在高达90%的情况下被误判为成年人所写，且移除选择题提示后准确率进一步崩塌，揭示了前沿大模型在作者年龄归因任务中存在严重的系统性误判、过度自信及自我解释无法纠错的问题。

    

    三名学生用笔回答了同一道考试题：“什么是幸福？你幸福吗？”采集时记录的年龄分别为17岁、28岁和45岁。17岁学生的答案措辞谨慎且用语正式。当被要求将每份答案与年龄和性别标签进行匹配时，大多数前沿语言模型都自信地将其判为成年人所写。本文对六个模型中的这一失败现象进行了结构化案例研究，涉及Claude Fable 5、Gemini 3.1 Pro、GPT-5.6 Sol、Grok 4.5、GPT-4o（2024）和DeepSeek V4。我们运行了1,081次API会话，并在操纵问题框架、信息披露格式和受众感知指令的六种实验条件下对3,225份回复进行了评分。六个模型中有三个在匹配任务上的表现处于或接近随机水平（1/3），同时却报告了73-88%的平均置信度。最年轻作者的答案在高达90%的试验中被误判为成年人所写。移除选择题提示框架后，准确率进一步崩塌：有三个模型从未落在……

    arXiv:2610.31946v1 Announce Type: new  Abstract: Three students answered the same exam prompt by hand: "What is happiness? Are you happy?" Their ages, recorded at collection, are 17, 28, and 45. The 17-year-old's answer is hedged and formally phrased. Asked to match each answer to an age and gender label, most frontier language models confidently assign it to an adult. This paper reports a structured case study of that failure across six models: Claude Fable 5, Gemini 3.1 Pro, GPT-5.6 Sol, Grok 4.5, GPT-4o (2024), and DeepSeek V4. We ran 1,081 API sessions and scored 3,225 responses across six experimental conditions that manipulate framing, disclosure format, and audience-awareness instruction. Three of six models perform at or near chance (1/3) on the matching task while reporting 73-88% mean confidence. The youngest writer's answer is misattributed to an adult in up to 90% of trials. Removing the multiple-choice scaffold collapses accuracy further: three models never place within th
    
[^79]: Kolibri：位于帕累托前沿的欧洲主权大模型

    Kolibri: A Sovereign European Model on the Pareto Frontier

    [https://arxiv.org/abs/2610.31945](https://arxiv.org/abs/2610.31945)

    Kolibri是一个总参数78.1B、激活参数仅3.46B的英德双语开源混合专家模型，通过完全自主掌控的数据与训练流程实现德语占比超20%（含2T自建德语token）的训练，在帕累托前沿上实现了面向主权化部署的高效性能。

    

    本报告介绍了Kolibri，一个英德双语的混合专家transformer模型，总参数量为78.1B，每个token激活3.46B参数，以Apache 2.0许可证开源发布。我们构建Kolibri的目的是实现主权化和专业化部署，特别关注德语以及公共管理、工业和航空航天等受监管领域。为支持这些应用场景，我们掌控了完整的模型开发流程，包括数据、架构、训练基础设施、后训练和评估。我们在24T token上对Kolibri进行训练，涵盖预训练、中期训练和长上下文扩展，其中德语按字节数计算占数据组合的20%以上。为达到这一比例，我们通过自建数据管道从网络整理并合成生成了超过2T的德语token，以补充德语数据集。后训练阶段将在精选的德语和英语数据上进行监督微调与……（摘要原文不完整）

    arXiv:2610.31945v1 Announce Type: cross  Abstract: This report introduces Kolibri, an English-German mixture-of-experts transformer with 78.1B total parameters and 3.46B active parameters per token, released as open weights under the Apache 2.0 licence. We built Kolibri for sovereign and specialised deployment, with a particular focus on German and on regulated domains such as public administration, industry, and aerospace. To support these settings, we control the full model-development process, including data, architecture, training infrastructure, post-training, and evaluation.   We train Kolibri on 24T tokens across pre-training, mid-training, and long-context extension, with German accounting for more than 20% of the data mix by bytes. To reach this share, we complement German datasets with more than 2T German tokens that we curate from the web with our own pipeline and generate synthetically. Post-training combines supervised fine-tuning on curated German and English data with re
    
[^80]: 一致性而非归一化使循环语言模型的深度保持可复用

    Agreement rather than normalisation keeps depth reusable in looped language models

    [https://arxiv.org/abs/2610.31943](https://arxiv.org/abs/2610.31943)

    循环语言模型深度可复用的关键在于训练时与推理时循环配置的一致性，而非循环步之间的状态归一化。

    

    循环语言模型反复应用同一个层块，而这一设计只有在深度可复用时才有价值：超出训练预算的循环步数在推理时必须仍然可用。我们探讨这究竟依赖于循环步之间的状态归一化（如已发布的 Ouro 模型所采用的），还是依赖于模型训练时所用循环与运行时所执行循环之间的一致性。在已发布的权重上，这种循环边界归一化可以在不增加任何参数的情况下被移除，且读出路径保持比特级一致；该操作在单次循环步时是恒等变换，这可作为实现正确性的检验。移除该归一化后，Ouro-1.4B 和 Ouro-2.6B 在一个平衡二分类任务上（随机水平为 0.500）到训练预算处准确率降为零。此前理论所预测的失败机制——状态范数增长——并不是原因：范数始终保持在 2.021 倍以内，而输出头却持续提交同一个错误 token。然而，在两个合成任务上从零训练的 1900 万参数模型中（摘要在此处截断）

    arXiv:2610.31943v1 Announce Type: cross  Abstract: Looped language models apply one block of layers repeatedly, and the design pays off only if depth is reusable: loop steps beyond the trained budget must remain usable at inference. We ask whether that rests on normalising the state between loop steps, as the released Ouro models do, or on agreement between the loop a model is trained with and the loop it is run with. On released weights this loop-boundary normalisation can be removed without adding parameters and with the readout path bit-identical, and the operation is an identity at one loop step, which serves as an implementation check. Removing it takes Ouro-1.4B and Ouro-2.6B to zero accuracy by the trained budget on a balanced binary task (chance 0.500). State-norm growth, the failure that prior accounts predict, is not the cause: the norm stays within a factor of 2.021 while the head commits to a single wrong token. Yet in 19M-parameter models trained from scratch on two synthe
    
[^81]: LazyFlash：前向稠密、后向稀疏的注意力机制，用于加速长上下文微调

    LazyFlash: Forward-Dense, Backward-Sparse Attention for Accelerated Long-Context Fine-Tuning

    [https://arxiv.org/abs/2610.31911](https://arxiv.org/abs/2610.31911)

    LazyFlash利用注意力训练开销主要由反向传播主导这一观察，提出前向保持稠密、后向采用梯度感知块稀疏化的方法，跳过对参数更新贡献可忽略的查询块，在保持模型为标准稠密模型（可被vLLM等系统直接部署）的同时显著加速长上下文微调。

    

    随着大语言模型（LLM）处理的序列越来越长，注意力机制成为主要的计算瓶颈。稀疏注意力大多针对推理阶段，然而许多方法仍然进行稠密训练或无法支持稀疏训练，且其额外增加的复杂性使得稀疏化难以转化为实际的墙钟时间加速，导致稠密注意力在实践中仍被广泛使用。本文提出了LazyFlash，一种前向稠密、后向稀疏的注意力方法，其建立在一个观察之上：注意力训练的开销主要由反向传播主导。LazyFlash保持前向传播完全稠密，因此训练得到的模型仍然是标准的稠密模型，现有系统（如vLLM）无需任何修改即可直接部署服务。对于反向传播，LazyFlash提出了一种梯度感知的块稀疏化设计，该设计根据注意力输出的梯度范数对查询块进行排序，并跳过那些对参数更新贡献可忽略不计的查询块，从而将稠密的反向传播转化为块稀疏的反向传播。

    arXiv:2610.31911v1 Announce Type: new  Abstract: As large language models (LLMs) process increasingly long sequences, attention becomes a major computational bottleneck. Sparse attention mostly targets inference, yet many methods still train densely or cannot support sparse training, and their added complexity makes sparsity hard to turn into wall-clock speedups, leaving dense attention widely used in practice. This paper introduces LazyFlash, a forward-dense, backward-sparse attention method built on the observation that attention training is dominated by the backward pass. LazyFlash keeps the forward pass exactly dense, so the trained model remains a standard dense model that existing systems such as vLLM can serve without any change. For the backward pass, LazyFlash proposes a gradient-aware block sparsification design that ranks query blocks by their attention-output gradient norms and skips those contributing negligibly to parameter updates, turning the dense backward into a block
    
[^82]: 知道何时停止：编码智能体中的证据-行动差距

    Knowing When to Stop: The Evidence-Action Gap in Coding Agents

    [https://arxiv.org/abs/2610.31895](https://arxiv.org/abs/2610.31895)

    该论文揭示了编码智能体的“证据-行动差距”现象——即智能体即使已收集到表明任务无法独立完成的充分证据仍过度坚持或虚报成功，并提出ImpasseBench基准通过在可解决任务中注入环境阻碍并标记证据充分步骤来量化这一差距。

    

    编码智能体被训练来完成任务，即使环境使得任务根本无法完成时，它们仍会不断尝试。表明只有外部帮助才能恢复进展的证据通常在几步之内就会出现，但智能体很少据此采取行动：它们过度坚持，尝试不必要的变通方法，并声称取得记录无法支持的成功——我们将这一现象称为“证据-行动差距”。为了衡量这一差距，我们提出了ImpasseBench，它在原本可解决的修复任务中注入十二种真实的环境阻碍因素，并在每条轨迹中标记出所收集证据变得充分的步骤。相对于该步骤对智能体的行动进行评分，可以区分适当的努力与过度坚持以及过早求助。在八个前沿模型和19,371条轨迹上，我们发现几乎所有需要外部帮助的运行都收集到了充分的证据，但仍有三分之一耗尽了预算，而且我们检测到的所有不安全请求都发生在这一时点之后。

    arXiv:2610.31895v1 Announce Type: cross  Abstract: Coding agents are trained to complete tasks, and they keep trying even when the environment makes completion impossible. The evidence that only outside help can restore progress usually appears within a few steps, yet agents rarely act on it: they over-persist, attempt undesired workarounds, and claim success the record does not support -- a phenomenon we call evidence-action gap. To measure it, we introduce ImpasseBench, which injects twelve realistic environment blockers into otherwise solvable repair tasks and marks, in each trajectory, the step at which the collected evidence becomes sufficient. Scoring actions relative to that step separates appropriate effort from over-persistence and premature handoff. Across eight frontier models and 19,371 trajectories, we find that nearly every run that needs outside help collects sufficient evidence, yet a third still exhaust their budget, and every unsafe request we detect comes after that 
    
[^83]: TaxoSAE：通过分类学逻辑重新思考稀疏自编码器

    TaxoSAE: Rethinking Sparse Autoencoders through Taxonomic Logic

    [https://arxiv.org/abs/2610.31894](https://arxiv.org/abs/2610.31894)

    TaxoSAE通过在稀疏自编码器特征空间中显式编码is-a分类学关系，利用按分类层级堆叠的SAE结构和分类对齐损失，实现了对大语言模型知识更准确且具分类一致性的读取与更精细粒度的控制。

    

    最近的稀疏自编码器（SAE）能够将大语言模型（LLM）的表示分解为可解释的特征，从而实现对所编码知识的精确读取与控制。然而，现有的SAE将每个特征视为独立单元，而现实世界的知识往往是以分类学方式组织的。因此，读取操作无法保证分类学上的一致性，控制操作也缺乏指定编辑范围的粒度。我们提出了TaxoSAE，它将分类学信息（建立在is-a即上位词-下位词关系之上）显式地编码到SAE特征空间中。TaxoSAE通过一组堆叠的SAE来镜像分类树结构，每个分类层级对应一个SAE，并引入了一种分类对齐损失，训练每个概念在其任一下位概念激活时也保持激活状态。在两个事实问答分类树（一个关于人物传记关系，一个关于动物）上，TaxoSAE实现了：（1）借助分类树进行更准确的读取，且激活更加集中，

    arXiv:2610.31894v1 Announce Type: new  Abstract: Recent Sparse Autoencoders (SAEs) decompose the representations of large language models (LLMs) into interpretable features, enabling precise readout and control of encoded knowledge. However, existing SAEs treat each feature as a standalone unit, whereas real-world knowledge is often taxonomically organized. As a result, readout carries no guarantee of taxonomic consistency, and control lacks the granularity to specify the scope of an edit. We propose TaxoSAE, which explicitly encodes taxonomical information, built on is-a (hypernym-hyponym) relations, into the SAE feature space. TaxoSAE mirrors a taxonomy with a stack of SAEs, one per taxonomy level, and introduces a taxonomic alignment loss that trains each concept to be active whenever one of its hyponyms is. On two factual QA taxonomies, one over biographical relations and one over animals, TaxoSAE enables (1) more accurate readout through the taxonomy with concentrated activations,
    
[^84]: 挖掘框架：将关联规则挖掘作为新闻框架检测的无监督框架

    Mining Frames: Association Rule Mining as an Unsupervised Framework for News Framing Detection

    [https://arxiv.org/abs/2610.31874](https://arxiv.org/abs/2610.31874)

    该论文提出将关联规则挖掘（ARM）作为无监督、无需标注的新闻框架检测新框架，通过对加沙战争首年25,512篇新闻文章的分析验证了其无需预定义分类体系即可归纳发现可解释框架模式的能力。

    

    计算框架分析目前主要由有监督分类器主导，这些分类器需要大规模标注语料库和预定义的框架分类体系，限制了其在非英语语言、新事件和低资源环境中的应用。本文提出将关联规则挖掘作为一种无监督、无需标注的新闻框架检测框架。每篇文档或句子被视为词汇项组成的事务，高提升度的共现规则被解读为候选框架信号（即Gamson和Modigliani所说的“浓缩符号”），并由记者-研究者在上下文中加以确认。主要验证案例为“加沙框架”研究：来自16家美国、英国、法国和德国媒体的25,512篇英文文章，涵盖加沙战争的第一年（2023年10月至2024年10月），采用Apriori算法（mlxtend）、平台加权支持度、词性过滤的n-grams以及基于提升度的评分进行分析。ARM能够在没有预定义分类体系的情况下归纳性地涌现出可解释的模式……

    arXiv:2610.31874v1 Announce Type: new  Abstract: Computational framing analysis is dominated by supervised classifiers that require large annotated corpora and predefined frame taxonomies, limiting their use for non-English languages, novel events, and low-resource settings. This paper presents Association Rule Mining (ARM) as an unsupervised, annotation-free framework for news framing detection. Each document or sentence is treated as a transaction of lexical items, and high-lift co-occurrence rules are read as candidate framing signals (Gamson and Modigliani's condensing symbols), confirmed in context by journalist-researchers. The primary validation is Framing Gaza: 25,512 English-language articles from 16 US, UK, French, and German outlets covering the first year of the Gaza war (October 2023 to October 2024), analyzed with Apriori (mlxtend), platform-weighted support, POS-filtered n-grams, and lift-based scoring. ARM inductively surfaces interpretable patterns without predefined c
    
[^85]: 在计算与数据约束下重写合成数据的缩放定律

    Rewriting Scaling Laws for Synthetic Data with Compute and Data Constraints

    [https://arxiv.org/abs/2610.31867](https://arxiv.org/abs/2610.31867)

    本文提出了首个面向合成重写数据的缩放定律，将生成重写的计算成本纳入建模，发现最优重写数据比例随计算预算的增加而显著上升。

    

    合成重写是指由语言模型对自然文档进行重写（例如转换为目标风格），这如今已成为预训练语料库的标准组成部分。然而，其使用中的若干关键问题仍知之甚少，例如：应该花费多少计算量来重写训练数据，以及在重写数据上训练出的模型能否超越重写器本身。我们通过提出一个重写缩放定律来研究这两个问题，该定律根据模型大小、训练词元数量、训练词元中重写数据所占比例以及在自然数据上的训练轮数来预测损失，并将生成重写所花费的计算量计入总成本。我们在两种设置下拟合了该缩放定律。在计算受限的设置中，自然数据充足，重写数据来自一个用比我们最大规模训练运行更多计算量训练的第三方重写器。我们发现，估计的最优重写比例随计算量的增加而上升，从3×10¹⁸ FLOPs时的不到10%上升到10²²时的32%。

    arXiv:2610.31867v1 Announce Type: cross  Abstract: Synthetic rewrites, where a language model rewrites a natural document e.g. into a target style, are now a standard ingredient of pretraining corpora. However, critical aspects of their use remain poorly understood, such as how much compute should be spent rewriting training data and whether a model trained on rewrites can outperform the rewriter. We study these two questions by proposing a rewriting scaling law that predicts loss from model size, training tokens, the fraction of training tokens that are rewrites, and the number of epochs over the natural data, charging the compute spent generating rewrites. We fit this scaling law in two settings. In the compute-limited setting, natural data is plentiful and rewrites come from a third-party rewriter trained with more compute than our largest run. We find that the estimated optimal rewrite fraction increases with compute, from under $10\%$ at $3\times10^{18}$ FLOPs to $32\%$ at $10^{22
    
[^86]: 基于ISO 11783网络的机器可读操作手册，用于离线多语言大语言模型辅助

    Machine-Readable Operator Manuals over the ISO 11783 Network for Offline, Multilingual LLM Assistance

    [https://arxiv.org/abs/2610.31863](https://arxiv.org/abs/2610.31863)

    本文提出在ISO 11783网络上对操作手册进行语义同步，使拖拉机侧的离线多语言LLM助手能够在31至89秒内获取完整的机具文档，传输时间与操作员首次连接机具时等待对象池上传的时间相当。

    

    如果农业机械要支持智能化的离线辅助，那么系统必须能够动态访问技术文档。本文分析并定义了基于ISO 11783的CAN总线上操作手册的语义同步，即将拖拉机侧助手的本地知识库与所连接机具的操作手册保持一致的过程。论文使用Markdown、JSON和XML对数据传输性能进行了基准测试，并评估了多种开源权重的大语言模型（LLM）在跨语言查询任务中的表现。答案的自动评估结果经过四位人工评审员的验证。根据格式和传输设置的不同，一份完整手册通过物理ISO 11783网络的传输时间为31至89秒，这与操作员在首次连接机具时已经习惯等待的对象池上传时间相当。Markdown生成的文件最小，而……（摘要在此处截断）

    arXiv:2610.31863v1 Announce Type: cross  Abstract: If agricultural machinery is to support intelligent, offline assistance, then the system must access technical documentation dynamically. In this paper, the Semantic Synchronization of operator manuals over ISO 11783-based CAN buses is analyzed, defined as the process that brings the local knowledge base of a tractor-side assistant into agreement with the operator manual of the connected implement. Data transfer performance was benchmarked using Markdown, JSON, and XML; various open-weight Large Language Models (LLMs) were also evaluated in cross-lingual query tasks. The automatic evaluation of the answers was validated against four human raters. A complete manual was transferred over a physical ISO 11783 network in 31 to 89 s, depending on format and transport settings, which is comparable to the object pool upload that operators already wait for at the first coupling of an implement. Markdown produced the smallest files, while the qu
    
[^87]: 基于人类反馈的大语言模型对齐：数学基础与算法设计

    Aligning Large Language Models with Human Feedback: Mathematical Foundations and Algorithm Design

    [https://arxiv.org/abs/2610.31862](https://arxiv.org/abs/2610.31862)

    本文系统介绍了大语言模型对齐的数学基础与算法框架，涵盖SFT、RLHF和DPO等主流技术，并将其理论与逆强化学习和离散选择模型建立联系，为开发更稳健、可扩展的对齐方法提供了全面参考。

    

    本文介绍了用于使大语言模型（LLM）与人类意图、偏好和价值观对齐的数学基础与算法框架。我们讨论了标准的对齐技术，如监督微调（SFT）、基于人类反馈的强化学习（RLHF）和直接偏好优化（DPO）。我们还探讨了从人类偏好中学习的理论基础，并将其与逆强化学习（IRL）和离散选择模型建立了联系。我们以教程的形式介绍了最先进的算法，讨论了它们的优势与局限性，并为实际应用提供了见解。本文旨在为研究人员和从业者提供一份全面的参考资料，既包含对对齐方法论的基础理解，也为开发更稳健、更可扩展的对齐技术提供了框架。

    arXiv:2610.31862v1 Announce Type: cross  Abstract: This article provides an introduction to the mathematical foundations and algorithmic frameworks used to align Large Language Models (LLMs) with human intentions, preferences, and values. We discuss standard alignment techniques, such as fine-tuning (SFT), reinforcement learning with human feedback (RLHF), and direct preference optimization (DPO). We also explore the theoretical underpinnings of learning from human preferences, drawing connections to inverse reinforcement learning (IRL) and discrete choice models. We present state-of-the-art algorithms in a tutorial style, discuss their advantages and limitations, and offer insights into practical implementation. Our exposition is intended to serve as a comprehensive resource for researchers and practitioners, providing both a foundational understanding of alignment methodologies and a framework for developing more robust and scalable alignment techniques.
    
[^88]: SWE-Clar：编码智能体能否澄清不完整的客户需求并交付全新软件？

    SWE-Clar: Can Coding Agents Clarify Incomplete Client Requirements and Deliver Greenfield Software?

    [https://arxiv.org/abs/2610.31854](https://arxiv.org/abs/2610.31854)

    该论文提出了 SWE-Clar 基准，基于 100 个报酬至少 4,000 美元的真实 Upwork 客户委托任务，通过“先澄清需求、后实现代码”的多轮交互设计并以明确的需求理解测试来评估编码智能体在自由职业全新软件开发中澄清不完整客户需求的能力。

    

    arXiv:2610.31854v1 公告类型：交叉引用（cross）。摘要：在自由职业环境中，全新软件开发要求编码智能体将不完整的客户请求转化为可运行的软件。这使得需求澄清至关重要，然而客户可能无法在漫长的开发周期中提供持续反馈，而且即使收到客户的回答，也不能保证智能体能正确理解复杂需求。这些挑战促成了两项基准设计选择：需求澄清先于代码实现，并通过明确的需求理解测试来评估澄清的有效性。基于这些设计选择，我们提出了 SWE-Clar，这是一个基于真实、高价值客户委托的基准，用于评估该场景下的需求澄清与代码实现能力。该基准包含 100 个源自 Upwork 上真实软件开发工作的任务，每个任务的总报酬至少为 4,000 美元。每个任务均以编码智能体与客户智能体之间的多轮交互开始，

    arXiv:2610.31854v1 Announce Type: cross  Abstract: Greenfield software development in freelance settings requires coding agents to turn incomplete client requests into working software. This makes requirement clarification essential, yet clients may be unable to provide continuous feedback over long development cycles, and receiving their answers does not guarantee correct understanding of complex requirements. These challenges motivate two benchmark design choices: clarification precedes code implementation, and its effectiveness is assessed through explicit tests of requirement understanding. Guided by these choices, we introduce SWE-Clar, a benchmark grounded in real, high-value client commissions to evaluate requirement clarification and code implementation in this setting. It comprises 100 tasks based on real software development jobs on Upwork, each with at least US\$4,000 in total payments. Each task begins with multi-round interaction between a coding agent and a client agent, 
    
[^89]: 在儿童自然经验数据上扩展跨情境词汇学习

    Scaling cross-situational word learning on children's naturalistic experiences

    [https://arxiv.org/abs/2610.31842](https://arxiv.org/abs/2610.31842)

    该研究探讨了基于儿童自然第一人称视频经验的跨情境词汇学习能否规模化，重点考察其表现对视觉表征质量的依赖性以及在更大、更多样化数据集上的扩展能力。

    

    儿童是如何学习那些构成其早期词汇大部分的简单、具体词汇的含义的？关于许多词义的统计信息可以通过逐步积累词汇与其指称对象共现的信息来获得，这一理论被称为“跨情境学习”。对比式视觉-语言模型使这一学习策略能够在捕捉儿童早期真实生活经验（包含视觉和语言输入）的第一人称视角视频数据上进行测试。该方法在小型数据集上的初步成功提供了概念验证，但视觉模型在这些同样的第一人称视频上训练时难以学到鲁棒的视觉表征。跨情境词汇学习的表现在多大程度上依赖于模型视觉表征的质量？跨情境模型能否扩展到更大、更多样化的数据集？为了回答这些问题，我们在……

    arXiv:2610.31842v1 Announce Type: new  Abstract: How do children learn the meanings of the simple, concrete words that form a majority of their early vocabulary? Statistical information about many word meanings can be recovered from the gradual accumulation of information about the co-occurrence of words with their referents, a proposal known as ``cross-situational learning''. Contrastive vision--language models allow this learning strategy to be tested on egocentric video data capturing children's early real-life experiences with visual and linguistic inputs. Initial successes of this method on small datasets provide a proof-of-concept, but vision models struggle to learn robust visual representations when trained on these same egocentric videos. How dependent is cross-situational word learning performance on the quality of the model's visual representations? And can cross-situational models scale to larger and more diverse datasets? To address these questions, we trained models on th
    
[^90]: SPECMEM：通过实时记忆管理加速智能体工具调用

    SPECMEM: Accelerating Agentic Tool Calling via Live Memory Management

    [https://arxiv.org/abs/2610.31840](https://arxiv.org/abs/2610.31840)

    SpecMem 提出一种在部署全过程中持续活跃管理的工具调用记忆机制，通过在线写回已验证调用（配合 LRU 淘汰）和副作用门控，使投机解码在多用户多会话场景下相比冻结草稿库保持约 27% 的平均接受 token 优势。

    

    工具调用智能体通常以准确性为主要优化目标；然而在部署阶段，其日益凸显的成本是延迟：每次工具调用都需要自回归地解码。投机解码可以在不改变输出结果的前提下加速这一过程，而在真实使用场景中，由于每个用户的行为往往具有重复性，检索过去的调用记录可以生成高质量的草稿。然而，现有系统要么不保留跨会话记忆，要么只在初始阶段构建一次草稿库且之后从不更新。我们提出 SpecMem，一种在整个部署过程中持续保持活跃并被管理的工具调用记忆：经过验证的调用会在线写回至带有 LRU 淘汰机制的、容量受限的每用户存储中，并且一个副作用门控机制会推迟不可逆调用的执行，直至验证确认后才进行。在由三个开源权重模型服务的多会话、多用户查询流上，冻结的草稿库的接受率会持续下降，而 SpecMem 则能保持接受率：平均接受 token 数相差 27%（各基准上为 +11% 至 +34%，具体取决于草稿库构建后出现的新任务量）。

    arXiv:2610.31840v1 Announce Type: new  Abstract: Tool-calling agents are mostly optimized for accuracy; in deployment, however, their growing cost is latency: every tool call is decoded autoregressively. Speculative decoding accelerates this without changing the output, and retrieved past calls make strong drafts under real use, where each user's behavior tends to repeat. Yet existing systems keep no cross-session memory, or build the draft store once and never update it. We propose SpecMem, a tool-call memory kept alive and managed throughout deployment: verified calls are written back online into bounded per-user stores with LRU eviction, and a side-effect gate defers irreversible calls until verification confirms. On multi-session, multi-user query streams served by three open-weight models, a frozen store steadily loses acceptance while SpecMem holds it: a 27% gap in mean accepted tokens (+11% to +34% across benchmarks, tracking how much novel work arrives after the store is built)
    
[^91]: SAGE：基于释义增强推理的词义对齐多模态定位评估

    SAGE: Sense-Aligned Multimodal Grounding Evaluation with Gloss-Augmented Reasoning

    [https://arxiv.org/abs/2610.31838](https://arxiv.org/abs/2610.31838)

    本文提出统一的多模态词义定位评估基准SAGE，并引入免训练的释义增强推理框架GAR，通过结合定义定位与约束引导语义验证，显著提升了多模态大模型在不同定位任务上的词义消歧性能。

    

    本文提出了词义对齐定位评估基准SAGE，这是一个统一的10候选图像选择基准，将VWSD、VerSe和WikiDiverse重新整合为共享的以词语为中心的评估协议，涵盖以名词、动词和实体为中心的定位任务。SAGE包含易混淆的困难负样本以及一个上下文交换的VerSe诊断任务，用以探究超越直接图文兼容性的目标词语定位能力。我们进一步提出了释义增强推理（GAR），这是一个免训练框架，将定义定位与约束引导的语义验证相结合。实验表明，模型在VWSD上的强劲表现并不能均匀迁移到其他定位设置中。使用Gemma3-12B模型时，定义增强提升了所有三个子集的性能，而约束引导推理提升了VWSD的表现，并在VerSe上取得了尤为显著的增益，但在WikiDiverse上效果较差。将两种信号结合使用实现了最佳的宏观平均性能，提升了Hits@1指标。

    arXiv:2610.31838v1 Announce Type: new  Abstract: This paper introduces Sense-Aligned Grounding Evaluation (SAGE), a unified 10-candidate image-selection benchmark that reformulates VWSD, VerSe, and WikiDiverse into a shared term-centric protocol covering noun-, verb-, and entity-centered grounding. SAGE includes confusable hard negatives and a context-swapped VerSe diagnostic to probe target-term grounding beyond direct caption--image compatibility. We further propose Gloss-Augmented Reasoning (GAR), a training-free framework combining definition grounding with constraint-guided semantic verification. Experiments show that strong VWSD performance does not transfer uniformly across grounding settings. With Gemma3-12B, definition augmentation improves all three subsets, while constraint-guided reasoning improves VWSD and yields particularly strong gains on VerSe, but is less effective on WikiDiverse. Combining the two signals achieves the best macro-average performance, improving Hits@1 
    
[^92]: 潜在一致性：前沿语言模型的概率判断是否服从全概率定律？

    Latent Coherence: Do Frontier Language Models' Probability Judgments Obey the Law of Total Probability?

    [https://arxiv.org/abs/2610.31833](https://arxiv.org/abs/2610.31833)

    该论文首次系统性地检验前沿语言模型在零样本条件下的概率判断是否满足全概率定律及因果后门调整，发现 GPT-6.1 Sol 和 Claude Opus 5.5 等模型已自发表现出接近贝叶斯后验的潜在概率一致性。

    

    LawZero 的 Scientist AI 训练一个预测器（Predictor）向贝叶斯后验逼近，其目标函数会惩罚不同潜在解释之间不一致的概率。我们由此提出问题：当前的前沿语言模型在未经此类训练的情况下，是否已经在零样本意义上具备这种一致性——当潜在变量 z 需要从上下文中的记录推断出来时，模型的概率判断是否满足全概率定律 P(X)=∑_z P(z)P(X|z)？在相互独立的单轮调用中，模型被分别要求给出 P(X)、潜在变量各状态上的 P(z)，以及 P(X|z)；此外还设置了一个因果推断系列，将同样的检验应用于 P(y|do(x)) 和后门调整。每个查询都对应一个精确的真实概率值，且其中一个题目子集以三种不同措辞提问，以区分持续性不一致与措辞变化带来的噪声。在七个模型中，GPT-6.1 Sol 和 Claude Opus 5.5 表现出最高的一致性（平均绝对残差分别为 0.08 和 0.12 对数几率，约合 2 个百分点）。

    arXiv:2610.31833v1 Announce Type: new  Abstract: LawZero's Scientist AI trains a Predictor toward the Bayesian posterior, with an objective that penalizes inconsistent probabilities among latent explanations. We ask whether current frontier language models, with no such training, are already consistent in this sense zero-shot: do their probability judgments satisfy the law of total probability, $P(X)=\sum_z P(z)P(X|z)$, when the latent $z$ must be inferred from records in the context? In separate single-turn calls, models are asked for $P(X)$, for $P(z)$ over the latent's states, and for $P(X|z)$; a causal family applies the same test to $P(y|do(x))$ and the back-door adjustment. Every query has an exact ground-truth probability, and a subset of items is asked in three wordings to separate persistent incoherence from rewording noise. Across seven models, GPT-6.1 Sol and Claude Opus 5.5 are the most coherent (mean absolute residual 0.08 and 0.12 log-odds, about 2 percentage points). Cla
    
[^93]: 查询所记住的：查询抽象程度决定古典阿拉伯语诗歌的词汇检索与稠密检索

    What the Query Remembers: Query Abstraction Governs Lexical and Dense Retrieval of Classical Arabic Poetry

    [https://arxiv.org/abs/2610.31821](https://arxiv.org/abs/2610.31821)

    该论文发现，在古典阿拉伯语诗歌检索中，查询保留诗歌原文措辞的程度（即查询的抽象程度）决定了检索器的优劣——记忆型查询下BM25领先所有稠密模型，而抽象查询下稠密编码器胜出，且BM25对措辞变化的敏感度远高于稠密模型。

    

    一个诗歌档案库能被搜索得多好，取决于用户还记得什么，而不仅仅是模型理解了什么。我们在古典阿拉伯语诗歌上验证了这一点，在两个语料库、三种索引模式和两种评估协议下，对九个稠密编码器和BM25进行了基准测试。哪个检索器胜出，取决于查询中仍然保留了多少诗歌原文的措辞。在“话到嘴边”（tip-of-the-tongue）型查询上，BM25领先所有稠密编码器，包括一个8B参数的最先进模型（Recall@5为0.746对0.717）；而在抽象查询上，BM25则跌至十个系统中的第八位（0.392对0.748）。对相同查询措辞的编辑实验表明，这种排名很大程度上反映了查询的构成：添加一句被记住的半行诗可使BM25提升38分；手工改写引用的词语后，BM25与8B编码器持平；而且BM25对被记住措辞的敏感度显著高于最强的稠密编码器。同样的维度也支配着解释增强方法。

    arXiv:2610.31821v1 Announce Type: cross  Abstract: How well a poetry archive can be searched depends on what the user still remembers, not only on what the model understands. We show this for Classical Arabic poetry, benchmarking nine dense encoders and BM25 over two corpora, three indexing modes and two evaluation protocols. Which retriever wins is governed by how much of a poem's wording the query still carries. On tip-of-the-tongue queries BM25 leads every dense encoder, including an 8B state-of-the-art model (0.746 vs. 0.717 Recall@5); on abstract queries it falls to eighth out of ten (0.392 vs. 0.748). Editing the wording of the same queries shows that this ranking largely reflects query composition: adding one remembered half-verse raises BM25 by 38 points, rewriting the quoted words by hand leaves BM25 level with the 8B encoder, and BM25's sensitivity to remembered wording is significantly steeper than that of the strongest dense encoder. The same axis controls explanation augme
    
[^94]: 大语言模型置信度的免费午餐：RL评论家早已知晓

    Free Lunch for LLMs Confidence: The RL Critic Already Knows

    [https://arxiv.org/abs/2610.31820](https://arxiv.org/abs/2610.31820)

    本文证明在基于正确性的结果下，LLM的置信度估计与强化学习中的评论家训练在数学上等价，因此RL训练得到的评论家无需额外的数据和计算即可免费提供可靠的模型置信度估计。

    

    可靠的置信度估计对于大语言模型（LLM）的可信部署至关重要，但专门的置信度训练阶段需要额外的数据和计算。我们提出了一个根本性问题：置信度训练实际解决的是什么预测问题？受先前外部置信度估计工作的启发，我们比较了在相同（问题、回答、正确性）三元组上训练的容量相当的自估计器与外部估计器。我们证明，仅基于策略的行为痕迹训练的外部模型，在不访问策略内部表示的情况下，可以与策略本身同样准确地估计策略的置信度。这揭示了与强化学习（RL）中价值函数学习的更深层次联系：在基于正确性的结果下，我们证明置信度估计与评论家训练是等价的，两者都是在预测策略回答正确的可能性。（注：原摘要在此处截断）

    arXiv:2610.31820v1 Announce Type: new  Abstract: Reliable confidence estimation is essential for trustworthy deployment of large language models (LLMs), but dedicated confidence training stage requires additional data and compute. We ask a fundamental question: what prediction problem does confidence training actually solve? Inspired by prior work on external confidence estimation, we compare self- and external estimators of comparable capacity trained on the same (question, response, correctness) triples. We show that an external model trained solely on a policy's behavioral traces can estimate the policy's confidence as accurately as the policy itself, without accessing the policy's internal representations. This points to a deeper connection with value function learning in reinforcement learning (RL): under a correctness-based outcome, we prove that confidence estimation and critic training are equivalent, both predicting the likelihood that a policy's response will be correct from 
    
[^95]: LAMB：一个面向拉丁语的现代上下文语言模型

    LAMB: A Modern Contextual Language Model for Latin

    [https://arxiv.org/abs/2610.31723](https://arxiv.org/abs/2610.31723)

    研究者基于ModernBERT架构，采用新颖的形态学感知掩码方案在740亿拉丁语词元上分两阶段预训练出LAMB编码器，并发布Acervus和首个拉丁语语义三元组数据集Corpus Trinum两个新语料库，为拉丁语语义相似度等下游任务奠定基础。

    

    本文介绍了拉丁语现代BERT（LAMB），这是一个基于ModernBERT架构的拉丁语编码器，采用一种新颖的形态学感知掩码方案，分两个阶段在超过740亿个拉丁语词元上进行了预训练。我们还提出了两个拉丁语语料库。Acervus包含240亿个BERT WordPiece词元，其中包括超过76万个来自严重损坏的OCR转录的样本，这些样本由一个专为拉丁语文本修复训练的ByT5模型进行修复。第二个数据集Corpus Trinum包含超过1亿个强、弱拉丁语语义三元组的混合。Corpus Trinum是首个拉丁语语义三元组数据集，为拉丁语语言模型适配通用句子语义相似度任务打开了大门。我们构建了一个包含五个任务和十七个子任务的实验网格，对15个预训练编码器和6个LAMB训练检查点应用了多种微调配置和子词聚合技术……

    arXiv:2610.31723v1 Announce Type: new  Abstract: This article introduces Latin Modern BERT (LAMB), an encoder for the Latin language based on the ModernBERT architecture, pretrained in two phases on more than 74B Latin tokens using a novel morphologically informed masking regimen. We also present two corpora for the Latin language. Acervus contains 24B BERT WordPiece tokens, including over 760K samples from heavily damaged OCR transcriptions, repaired by a ByT5 model trained for Latin text repair. The second dataset, Corpus Trinum, contains a mixture of over 100M strong and weak Latin language semantic triplets. Corpus Trinum is the first Latin semantic triplet dataset, opening the door for Latin language models to be adapted to general sentence semantic similarity tasks. We construct an experimental grid using five tasks and seventeen subtasks, applying a variety of finetuning configurations and subword aggregation techniques for 15 pretrained encoders, 6 LAMB training checkpoints, an
    
[^96]: Omni-UBench：跨安全性、实用性与可靠性的全模态大语言模型防护栏基准测试

    Omni-UBench: Benchmarking Omni-Modal LLM-based Guardrails Across Safety, Utility, and Reliability

    [https://arxiv.org/abs/2610.31713](https://arxiv.org/abs/2610.31713)

    该论文提出了首个全面的全模态安全评估基准Omni-UBench，从安全性、实用性和可靠性三个维度评估基于全模态大语言模型的防护栏模型，涵盖超过18.3万个来自文本、图像、音频、视频及跨模态组合的样本。

    

    全模态大语言模型（OLLMs）的进步使人机交互日益无缝，但其对异构模态的统一处理也扩大了攻击面，使攻击者能够利用更丰富的输入组合来绕过安全对齐。为了应对此类风险，研究者提出了全模态防护栏，用于检测和过滤跨任意模态的不安全查询。然而，由于缺乏全面的评估基准，此类防护栏的能力与局限仍未被充分了解。在本文中，我们提出了Omni-UBench，这是首个全面的全模态安全评估框架，从三个关键维度（即安全性、实用性和可靠性）对防护栏模型进行评估。Omni-UBench汇集了超过18.3万个样本，来自35个以上异构来源，涵盖文本、图像、音频、视频及其跨模态组合，并带有安全标注。（原文摘要此处截断）

    arXiv:2610.31713v1 Announce Type: cross  Abstract: Advances in Omni-modal Large Language Models (OLLMs) have enabled increasingly seamless human-AI interaction, but their unified processing of heterogeneous modalities also expands the attack surface, allowing adversaries to exploit richer input compositions to circumvent safety alignment. To counter such risks, omni-modal guardrails have been proposed to detect and filter unsafe queries across arbitrary modalities. However, their capabilities and limitations remain insufficiently understood due to the absence of a comprehensive evaluation benchmark. In this paper, we present Omni-UBench, the first comprehensive framework for omni-modal safety assessment, evaluating guardrail models along three critical dimensions, i.e., safety, utility, and reliability. Omni-UBench curates over 183k samples drawn from 35+ heterogeneous sources spanning text, image, audio, video, and their cross-modal combinations, annotated with safety labels under a u
    
[^97]: 欺骗大语言模型安全防护机制对错误的文本进行判断

    Fooling LLM Safety Guards into Judging the Wrong Text

    [https://arxiv.org/abs/2610.31695](https://arxiv.org/abs/2610.31695)

    论文提出“目标重绑定”攻击方法——通过附加伪造查询和无害诱饵，利用安全防护机制只评判输入中最后一个完整任务的特性，使有害内容绕过审查，在GuardRebind基准上对五种防护机制的绕过率均超过93%。

    

    在LLM应用程序对内容采取行动之前，安全防护机制会先检查内容，而大多数防护机制会将该内容直接粘贴到纯文本提示词中。模仿提示词格式的内容可以附加第二个任务，防护机制反而会去回答那个任务。我们将这一现象称为“目标重绑定”。我们的解释是：防护机制评判的是其输入中最后一个完整的任务（此处为无害的诱饵），因此有害请求得以原封不动地通过。在一项预注册测试中，附加的伪造查询和诱饵成功绕过了Shieldstral-1.0-3B标记的全部75个保留AdvBench有害请求。防护机制并非被说服："忽略上述文本”的指令无法绕过其中任何一个请求，且在后续运行中，有害的诱饵反而会阻止攻击。在我们的GuardRebind基准测试中，五种防护机制中的每一种都有一个（针对各防护机制选择的）攻击家族，能够绕过至少93%的被其标记的项目。目标重绑定对Shieldstral和gpt-oss-safeguard的解释效果最好；而对于ShieldGemma和Llama Guard 3，通用规避和尾部文本策略分别……

    arXiv:2610.31695v1 Announce Type: cross  Abstract: Safety guards check content before an LLM application acts on it, and most paste that content into a plain-text prompt. Content that imitates the prompt can append a second task, and the guard answers that one instead. We call this target rebinding. Our explanation is that the guard grades the last complete task in its input, here a harmless decoy, so the harmful request goes through unchanged. In a pre-registered test, an appended forged query and decoy clear all 75 held-out AdvBench requests that Shieldstral-1.0-3B flags. The guard is not persuaded: a directive to ignore the text above clears none of them, and in a follow-up run a harmful decoy stops the attack. On our GuardRebind benchmark, each of five guards has an attack family, chosen per guard, that clears at least 93% of its flagged items. Rebinding explains this best on Shieldstral and gpt-oss-safeguard; on ShieldGemma and Llama Guard 3, generic evasion and trailing text resp
    
[^98]: 大语言模型的分布对齐及其泛化特性

    Distributional Alignment of LLMs and Its Generalization Properties

    [https://arxiv.org/abs/2610.31687](https://arxiv.org/abs/2610.31687)

    该研究提出了大语言模型与人类分布对齐的方法，发现对齐后的模型能够泛化到同领域的未见问题，跨主题泛化取决于主题间的关联性，但可能伴随创造力等能力的意外退化。

    

    大语言模型越来越多地被用于模拟和探究人类行为，例如用于问卷调查。对于这些应用而言，一个关键属性是输出满足分布对齐：即模型的输出分布与某种参考（此处为人类）分布相匹配。实现分布对齐的一个障碍是如何从少量训练数据中进行泛化。在这项工作中，我们研究了：(1) 如何对大语言模型进行分布对齐；(2) 对齐后的大语言模型如何泛化到未见过的样本以及不相关的主题和任务。通过两个大规模基准测试，我们发现大语言模型确实可以实现分布对齐，并能泛化到同一领域中未见过的的问题。跨主题泛化则取决于主题之间的关系（例如，关于政治的问题与对机构的信任相关）。然而，对齐后的大语言模型可能会以令人惊讶的方式发生退化（例如，失去其创造力），这表明其泛化特性……

    arXiv:2610.31687v1 Announce Type: cross  Abstract: Large language models are increasingly used to simulate and probe human behavior, e.g., for surveys. For these applications, a key property is that outputs satisfy distributional alignment: to match its output distribution with some reference (in this case, human) distribution. An obstacle in achieving distributional alignment is generalizing from small training data. In this work, we study (1) how to distributionally align LLMs; and (2) how aligned LLMs might generalize to unseen samples and unrelated topics and tasks. Using two large-scale benchmarks, we find that LLMs can indeed achieve distributional alignment and generalize to unseen questions of the same domain. Cross-topic generalization depends on the relationship between topics (e.g., questions about politics are related to trust in institutions). However, aligned LLMs can degrade in surprising ways (e.g., lose their creative ability), suggesting that its generalization proper
    
[^99]: 从目标选择到条件复用：审计工具使用模型中的残差干预

    From Target Selection to Condition Reuse: Auditing Residual Interventions in Tool-Use Models

    [https://arxiv.org/abs/2610.31686](https://arxiv.org/abs/2610.31686)

    激活干预虽能大幅提升模型对熟悉目标的选择正确率，但这种增益无法泛化到事实与标识符的重新组合或独立资格判断上，仅能改变源事实影响判断的方式。

    

    激活干预可以使语言模型产生目标答案，但模型能否在其他计算中使用所预期的条件？我们在Qwen3.5-27B上研究了这一问题，任务为工具使用场景，其中时间或权限决定候选者的资格与选择。在32个新构建的世界中，两种针对具体案例的残差写入变体将熟悉目标选择的正确率从35/128提升至119/128，以及从33/128提升至114/128。然而，当熟悉的事实和标识符在同一个选择任务中被重新组合时，这些增益便消失了，同时先前被奖励过的标识符被选中的比例反而上升。这些增益在不同输出格式下保持稳定，但在关于候选者是否符合资格或应否被选中的独立判断上几乎没有改善。尽管如此，内部编辑仍能够改变源事实对判断的影响方式：在一项预先设定的次要比较中，当某个源事实被使用时，早期层的源补丁改变了64个判断中的50个……

    arXiv:2610.31686v1 Announce Type: cross  Abstract: Activation interventions can make a language model produce a target answer, but can the model use the intended condition in other computations? We study this question in Qwen3.5-27B on tool-use tasks where time or permissions determine eligibility and selection. Across 32 fresh worlds, two case-specific residual-writing variants raise familiar target-selection correctness from 35/128 to 119/128 and from 33/128 to 114/128. Yet these gains disappear when familiar facts and identifiers are recombined within the same selection task, while selection of previously rewarded identifiers increases. The gains persist across output formats but yield little improvement in independent judgments about whether a candidate is eligible or should be selected. Internal edits can nevertheless change how a source fact affects a judgment: in a prespecified secondary comparison, an early-layer source patch changes 50/64 judgments when that source is used, wi
    
[^100]: 一种基于证据校验的智能体方法，用于对话中的多模态情感推理

    An Agentic Approach with Evidence-Grounded Verification for Multimodal Emotion Reasoning in Conversation

    [https://arxiv.org/abs/2610.31685](https://arxiv.org/abs/2610.31685)

    该论文提出了一种可追溯的多模态情感推理智能体框架，通过模态感知智能体发布带证据引用的观察、集成智能体生成情感标签与引用，并利用独立的基于证据的验证阶段执行确定性检查来定位错误来源，从而以低成本实现对话中的多模态情感推理。

    

    基于大语言模型（LLM）的多模态情感推理系统在标准基准测试中表现良好，但通过指令微调来改进它们的计算成本可能很高。由冻结模型构建的智能体流水线可以避免这一成本，但带来了另一个挑战：当预测出错时，系统必须识别是哪个智能体或哪个阶段导致了错误。我们通过一个可追溯的多模态情感推理与提示空间自适应的智能体框架来解决这一问题。针对特定模态的感知智能体分析视觉、声学和文本输入，并将附带证据引用的观察结果发布到共享看板上。一个集成智能体将这些观察结果与检索到的多模态示例相结合，生成情感描述、标签集合以及支持性引用，同时多模态大语言模型（MLLM）评论者可以在输出最终确定之前要求进行一次修订。对于错误的预测，一个独立的基于证据的验证阶段会执行确定性检查

    arXiv:2610.31685v1 Announce Type: cross  Abstract: LLM-based multimodal emotion reasoning systems perform well on standard benchmarks, but improving them through instruction tuning can be computationally expensive. Agentic pipelines built from frozen models avoid this cost, but introduce a different challenge: when a prediction is wrong, the system must identify which agent or stage caused the error. We address this problem with an agentic framework for traceable multimodal emotion reasoning and prompt-space adaptation. Modality-specific perception agents analyze visual, acoustic, and lexical inputs and post evidence-cited observations to a shared board. An integration agent combines these observations with retrieved multimodal examples to produce an emotion description, label set, and supporting citations, while an MLLM critic can request one revision before the output is finalized. For incorrect predictions, a separate evidence-grounded verification stage applies deterministic checks
    
[^101]: 大语言模型时代的知识图谱质量评估：一个概念框架

    Knowledge Graphs Quality Assessment in the Era of LLMs: A Conceptual Framework

    [https://arxiv.org/abs/2610.31684](https://arxiv.org/abs/2610.31684)

    本文回顾了数据、信息、关联数据及知识图谱与大语言模型交叉领域的质量评估文献，将现有最先进框架扩展为包含24个质量维度的目录，并提出了一种通用、可按领域或任务定制且实用的知识图谱质量评估概念框架。

    

    知识图谱（KGs）在过去几十年中得到了广泛普及，并且越来越多地由大语言模型（LLMs）构建和消费。因此，各种应用依赖知识图谱来提供简洁、完整、准确、忠实、新鲜且可追溯的答案。然而，这些知识图谱的质量究竟如何？在哪些情况下应该使用知识图谱？它是否可以被信任用于为大语言模型提供事实依据？又该如何对其进行评估？我们回顾了关于数据、信息、关联数据、知识图谱以及知识图谱与大语言模型交叉领域的质量评估（QA）文献。我们将当前最先进的框架扩展为一个包含24个质量维度（QDs）和多种质量指标（QMs）的目录。此外，我们提出了一个通用的、可针对特定领域或任务定制的、实用的质量评估框架，用于评估知识图谱的质量。

    arXiv:2610.31684v1 Announce Type: new  Abstract: Knowledge Graphs (KGs) have been popularized during the last few decades and are increasingly constructed and consumed by Large Language Models (LLMs). Various applications therefore rely on KGs to provide concise, complete, accurate, faithful, fresh, and traceable answers. However, what is the quality of those KGs? In which cases should a KG be used, and can it be trusted to ground an LLM? How might it be evaluated? We reviewed the literature on quality assessment (QA) of data, information, linked data, KGs, and the intersection of KGs and LLMs. We extended the current state-of-the-art frameworks to a catalog of 24 quality dimensions (QDs) and various quality metrics (QMs). Furthermore, we propose a general-purpose, customizable to a domain or task, and practical QA framework for assessing the quality of KGs.
    
[^102]: 图式连贯性量表：测量被赋予的信念是否组织了语言模型的认知

    The Schema Coherence Inventory: Measuring Whether an Assigned Belief Organizes a Language Model's Cognition

    [https://arxiv.org/abs/2610.31683](https://arxiv.org/abs/2610.31683)

    本文提出图式连贯性量表（SCI），一种基于贝克认知模型的96题强制选择、无生成评分工具，用于验证赋予语言模型的信念是否真正组织其认知而非仅仅引发情绪，并通过目标特异性判别标准在五个骨干模型上得到验证。

    

    人格条件化的语言模型被用作模拟患者和情感代理，但目前尚无经过验证的方法来判断赋予此类模型的信念是否真正组织了其行为，还是该模型仅仅采纳了一种情绪。我们提出了图式连贯性量表（SCI）：包含96个强制选择题项，覆盖贝克认知模型的八个层面，每个题项提供六个选项，分别对应三种消极自我图式及其积极对应图式，并通过长度归一化的对数似然进行评分，因此无需生成或解析任何文本。由于单纯的情感就可能使模型转向一个默认信念，判别标准是目标特异性：为某一信念构建的条件必须比为其同极性其他信念构建的条件更频繁地选中该信念。在五个骨干模型（参数量为17亿至140亿）上开展的六项验证研究、共107个条件，支持了这一解读。九名接受过临床培训的评分者能以79.5%的准确率识别出所标注的信念（随机水平为16.7%）。

    arXiv:2610.31683v1 Announce Type: new  Abstract: Persona-conditioned language models are used as simulated patients and affective agents, but there is no validated way to ask whether a belief assigned to such a model organizes its behaviour or whether the model has merely adopted a mood. We present the Schema Coherence Inventory (SCI): 96 forced-choice items at eight levels of Beck's cognitive model, each offering six replies keyed to three negative self-schemas and their positive counterparts, scored by length-normalized log-likelihood so that nothing is generated or parsed. Affect alone can move a model onto a default belief, so the discriminant criterion is target specificity: a condition built for one belief must select that belief more often than conditions built for the other beliefs of its pole. Six validation studies over 107 conditions on five backbones (1.7B-14B parameters) support this reading. Nine clinically trained raters recover the keyed belief at 79.5% (chance 16.7%). 
    
[^103]: 针对自闭症与非自闭症个体自发双人对话互动的多模态感知对话成功预测

    Multi modal perceived conversation success prediction for spontaneous autistic and non-autistic dyadic interactions

    [https://arxiv.org/abs/2610.31682](https://arxiv.org/abs/2610.31682)

    本研究提出利用构音、面部、声学-韵律和话轮转换等多模态特征，在包含自闭症与非自闭症个体的自发双人对话数据中预测参与者感知的对话成功，证明对话成功本质上是多维的，需要多层面的分析框架。

    

    对话成功对于沟通、社会纽带和幸福感至关重要，而在自闭症者与神经典型者的对话中，对话成功往往会降低。先前的研究曾在儿科自闭症富集样本中使用声学和词汇特征来预测感知的对话成功，或在非自闭症群体互动中使用非言语行为进行预测。其他研究刻画了自闭症互动中的韵律、面部、动作和话轮转换同步性，但主要针对诊断分类或组间差异分析。本研究则从互补的构音、面部、声学-韵律和话轮转换测量出发，在三个包含自闭症和非自闭症个体的自发双人互动数据集上预测参与者感知的对话成功。通过利用多模态特征，我们证明对话成功本质上是多维的，需要一个多层面的分析框架。

    arXiv:2610.31682v1 Announce Type: new  Abstract: Conversation success is important for communication, social bonds, and well-being and is often reduced in autistic-neurotypical conversations. Previous studies have predicted perceived conversation success using acoustic and lexical features in pediatric autism-enriched samples, or using nonverbal behavior in non-autistic group interactions. Other work has characterized prosodic, facial, motor, and turn-taking synchrony in autistic interactions, but has primarily targeted diagnostic classification or group differences. The present study instead predicts participant perceived conversation success from complementary articulatory, facial, acoustic-prosodic, and turn-taking measures across three spontaneous dyadic-interaction datasets including autistic and non autistic individuals. By leveraging multimodal features, we demonstrate that conversational success is inherently multidimensional and requires a multi-faceted analytical framework. C
    
[^104]: 重新思考结构化生成：基于图的推理能否解决歧义问题？

    Rethinking Structured Generation: Can Graph-Based Reasoning Resolve Ambiguity?

    [https://arxiv.org/abs/2610.31681](https://arxiv.org/abs/2610.31681)

    该论文将结构化生成重新表述为歧义感知推理问题，提出基于图推理的多有效学习框架，通过对一组合理的结构化输出进行推理而非预测单一解，解决了自然语言歧义导致现有方法输出不稳定和不一致的问题。

    

    结构化生成通常被表述为从输入到单一结构化输出的确定性映射。这一假设与自然语言存在根本性的不匹配，因为自然语言往往允许多种结构上不同但语义上均有效的解释。因此，现有方法依赖于单一参考监督，将歧义压缩为单一预测，导致在欠指定输入下产生不稳定且不一致的输出。我们将结构化生成重新表述为一个歧义感知推理问题，其目标是对一组合理的结构化输出进行推理，而非预测单一解。我们证明标准训练本质上会抑制有效的替代方案，因此我们采用了一种多有效学习框架，在优化过程中保留歧义。为了将这一表述付诸实践，我们构建了一个结构化输出的候选空间，将每个候选表示为……

    arXiv:2610.31681v1 Announce Type: cross  Abstract: Structured generation is typically formulated as a deterministic mapping from input to a single structured output. This assumption is fundamentally misaligned with natural language, which often admits multiple structurally distinct yet semantically valid interpretations. As a result, existing methods rely on single-reference supervision that collapses ambiguity into one prediction, leading to unstable and inconsistent outputs under underspecified inputs. We reformulate structured generation as an \textit{ambiguity-aware inference problem}, where the objective is to reason over a set of plausible structured outputs rather than predict a single solution. We show that standard training inherently suppresses valid alternatives, and instead adopt a multi-valid learning framework that preserves ambiguity during optimization. To operationalize this formulation, we construct a candidate space of structured outputs, represent each candidate as 
    
[^105]: MisconceptDialogue：评估标准如何塑造LLM数学辅导中测量到的成功

    MisconceptDialogue: Evaluation Criteria Shape Measured Success in LLM Math Tutoring

    [https://arxiv.org/abs/2610.31679](https://arxiv.org/abs/2610.31679)

    该论文提出MISCONCEPTDIALOGUE全自动评估框架，揭示评估标准对LLM数学辅导效果测量结果影响巨大：在相同对话上，精确匹配评分报告的纠正成功率仅为42.1%，而人工校准的LLM裁判评分高达81.3%。

    

    大语言模型越来越多地被用作数学辅导教师，这引出了两个问题：单一LLM能否检测学生的错误、诊断其背后的误解，并通过多轮对话加以纠正——以及我们能否信任那些报告其表现好坏的数字？我们通过MISCONCEPTDIALOGUE同时回答这两个问题，这是一个完全自动化的框架，在五个开源权重LLM辅导模型（7B-14B）与LLM模拟学生交互的5000多个对话上运行完整流程——这是一个刻意采用的模拟协议（配对条件、对真实学习者不安全的错误诊断对照组、大规模），其结果并不能证明真实学生的学习效果。在测量方面，我们的核心发现对该领域并不乐观：标准数学NLP中常见的基准式精确匹配评分——作为一种下限压力测试——在完全相同的对话上报告的纠正成功率仅为42.1%，而经过人工校准的LLM作为裁判则报告为81.3%，两者相差39.2个百分点

    arXiv:2610.31679v1 Announce Type: new  Abstract: Large language models are increasingly used as math tutors, raising two questions: can a single LLM detect a student's error, diagnose the underlying misconception, and remediate it through multi-turn dialogue - and can we trust the numbers that report how well it does? We address both with MISCONCEPTDIALOGUE, a fully automated framework that runs the full pipeline on five open-weight LLM tutors (7B-14B) interacting with LLM-simulated students over 5,000+ dialogues - a deliberately simulated protocol (paired conditions, a wrong-diagnosis control unsafe with real learners, scale) whose results do not establish real-student learning. On measurement, our headline finding is uncomfortable for the field: benchmark-style exact-match scoring common in standard math NLP - a lower-bound stress test - reports 42.1% measured Remediation Success Rate on identical dialogues where a human-calibrated LLM-as-judge reports 81.3%, a 39.2 percentage-point 
    
[^106]: 验证比较本身，而不仅是评分器：为什么总体准确率会在交互式大语言模型评估中产生误导

    Validating the Comparison, Not Just the Scorer: Why Overall Accuracy Can Mislead in Interactive LLM Evaluation

    [https://arxiv.org/abs/2610.31678](https://arxiv.org/abs/2610.31678)

    该论文指出，仅用总体准确率验证自动评分器会掩盖针对特定提示的评分误差，从而可能扭曲交互式LLM评估中提示间比较的方向和幅度，因此应当验证“比较结果”本身而不仅是评分器。

    

    许多关于大语言模型的研究会比较模型对不同后续提示的响应方式。例如，一个提示可能以中性的方式要求模型重新考虑，而另一个提示则坚称一部虚构的电影确实存在。这种差异被用来衡量对话压力的影响。随后，自动评分器会标注每个回复是否做出了无事实依据的声明。研究者通常用一个总体准确率或 F₁ 分数来验证评分器的可靠性。然而，较高的总体性能可能掩盖针对特定提示的错误：如果评分器在某个提示之后犯下更多错误，那么测得的差异可能过大、过小，甚至方向完全相反。一个主要由困难样本组成的审计也可能歪曲在整个研究中真实的表现水平。我们推导了此类错误如何改变提示之间的比较结果，并使用六个模型、180 个虚构实体问题以及六种提示条件对这一问题进行了实证考察。在一个包含 288 个回复的均衡审计中，总体……

    arXiv:2610.31678v1 Announce Type: cross  Abstract: Many LLM studies compare how models respond to different follow-up prompts. For example, one prompt may neutrally ask a model to reconsider, while another insists that an invented film really exists. The difference is used to measure the effect of conversational pressure. An automatic scorer then labels whether each response makes an unsupported factual claim. Researchers commonly validate the scorer with one overall accuracy or $F_1$ score. However, high overall performance can hide prompt-specific mistakes: if the scorer makes more errors after one prompt, the measured difference can be too large, too small, or point in the wrong direction. An audit containing mostly difficult responses can also misrepresent performance in the full study. We derive how such errors alter a prompt comparison and examine the problem using six models, 180 invented-entity questions, and six prompt conditions. In a balanced audit of 288 responses, overall 
    
[^107]: 真值标注如何塑造检索得分：面向乳腺病变管理的检索增强生成系统的证据感知评估

    How Ground Truth Shapes Retrieval Scores: Evidence-Aware Evaluation of a Retrieval-Augmented Generation System for Breast Lesions

    [https://arxiv.org/abs/2610.31606](https://arxiv.org/abs/2610.31606)

    该论文提出了一种证据感知的检索评估协议，通过将证据片段映射到所有包含块来定义相关性，从而解决检索得分依赖真值标注覆盖范围的问题，并将其应用于乳腺病变管理文献临床RAG系统的评估。

    

    目的：描述一种针对基于乳腺病变管理文献的临床检索增强生成（RAG）系统的证据感知检索评估协议，以解决检索得分对真值标注覆盖范围的依赖问题。方法：使用包含18篇文档的语料库和18条查询的开发基准，每篇文档对应一个“是/否”问题或特殊疑问句问题。检索结合了稠密向量嵌入、BM25和交叉编码器重排序。评估了128至1024个词元的分块大小、0%至20%的重叠率以及四种检索器配置。相关性由映射到所有包含块上的证据片段来定义，将包含等价证据的重叠块视为相关。检索效果采用R-Precision、Recall@K、MRR和nDCG@K进行评估。此外，生成质量在512词元索引上以K=1至5进行评估，采用DeepEval声明级指标（以deepseek-v4-pro作为LLM评判者）以及BLEU/ROUGE指标。结果：所选的混合【摘要在此处不完整，原文截断】

    arXiv:2610.31606v1 Announce Type: new  Abstract: Purpose: To describe an evidence-aware retrieval evaluation protocol for a clinical retrieval-augmented generation (RAG) system over breast lesion management literature, addressing the dependence of retrieval scores on annotation coverage. Methods: An 18-document corpus and an 18-query development benchmark, with one Yes/No or WH-question per document, were used. Retrieval combined dense embeddings, BM25, and cross-encoder reranking. Chunk sizes of 128 to 1024 tokens, overlaps of 0 to 20%, and four retriever configurations were evaluated. Relevance was defined by evidence spans mapped to all containing chunks, treating overlapping chunks with equivalent evidence as relevant. Retrieval was evaluated using R-Precision, Recall@K, MRR, and nDCG@K. Separately, generation was assessed on a 512-token index at K=1 to 5 using DeepEval claim-level metrics, with deepseek-v4-pro as the LLM judge, and BLEU/ROUGE metrics. Results: The selected hybrid 
    
[^108]: 主要是阅读：开卷评分在文档级LoRA适配器中究竟衡量了什么

    Mostly Reading: What Open-Book Scores Measure in Document-Level LoRA Adapters

    [https://arxiv.org/abs/2610.31017](https://arxiv.org/abs/2610.31017)

    开卷评分主要衡量的是所有适配器共有的“阅读”行为而非文档记忆——随机适配器即可恢复专用适配器82%的开卷优势，但在无源文本的完形测试中仅恢复约2%（与零无异），因此不宜用开卷评分来评估或选择文档级LoRA适配器。

    

    单文档LoRA适配器已被提出作为可合并与检索的知识单元；然而，开卷评分——即将文档放入提示词中——会同样容易地把功劳归给文档自己的适配器（其专用适配器）的“阅读”行为，而非真正“记住”该文档。我们在Llama-3.1-8B-Instruct上为150篇Wikipedia文章各训练一个适配器，发现开卷分数主要衡量的是每个适配器都共有的一种阅读行为：一个在其他文档上训练的随机适配器能够恢复专用适配器相对于基础模型82%的开卷优势，但在一个不提供源文本、直接询问同一文档事实的完形填空探测中，只能恢复其2%的优势——这一比例与零无法区分。文档内容确实写入了权重，因为专用适配器在完形回忆上比基础模型高出0.094，且其更新的几何结构遵循其所阅读的文档；然而，在一项包含30个查询的测试中，通过权重空间邻近性选择的适配器……（原文在此处截断）

    arXiv:2610.31017v1 Announce Type: new  Abstract: Single-document LoRA adapters have been proposed as units of knowledge to be merged and retrieved; yet open-book scoring, with the document in the prompt, can credit a document's own adapter (its specialist) for reading that document as readily as for holding it. Training one adapter for each of 150 Wikipedia articles on Llama-3.1-8B-Instruct, we find that open-book scores mostly measure a reading behaviour every adapter shares: a random adapter trained on another document recovers 82 % of the specialist's open-book advantage over the base model but only 2 % of its advantage on a cloze probe that asks for the same document's facts without providing the source text, a share we cannot distinguish from zero. Content does reach the weights, since the specialists beat the base model in cloze recall by 0.094, and the geometry of their updates follows the documents they read; yet, in a test of 30 queries, adapters chosen by weight-space proximi
    
[^109]: 安全探针的缩放定律

    Scaling Laws for Safety Probes

    [https://arxiv.org/abs/2610.30678](https://arxiv.org/abs/2610.30678)

    本文通过构建六个新安全数据集、在20个基础模型上以不同标签规模训练探针，首次提出了可依据模型规模和标签数量提前预测安全探针性能的缩放定律。

    

    安全探针是在模型激活值上训练的、用于实时监控语言模型的标准工具。每当发布新模型或需要监控新的危害领域时，探针都必须重新训练，这需要新的标签以及探针架构的选择。然而，无论是模型还是标签都无法预先告诉我们最终探针的性能如何，要弄清这一点需要训练并比较多种探针配置。在本文中，我们探讨是否可以仅凭模型规模和标签数量来提前预测探针的性能。现有的安全数据集不足以支持这一研究，因为它们要么规模有限，要么已经饱和。因此，我们构建了六个规模足够大、难度足够高的新安全数据集。在这些数据集上，我们在来自五个模型家族的20个基础模型上训练了两种探针架构，每个模型使用32到10⁶个标签。我们使用探针误差来衡量性能……

    arXiv:2610.30678v1 Announce Type: cross  Abstract: Safety probes trained on a model's activations are a standard tool for monitoring language models in real time. Each time a new model is released or a new harm area needs to be monitored, the probe must be retrained, which requires new labels and a choice of probe architecture. Yet neither the model nor the labels tell us in advance how well the resulting probe will perform. Finding out requires training and comparing many probe configurations. In this paper, we ask whether probe performance can instead be predicted in advance from model size and the number of labels. Existing safety datasets are insufficient for this, as they are either limited in scale or already saturated. We therefore construct six new safety datasets of sufficient size and difficulty. On these datasets, we train two probe architectures on 20 base models from five model families, with 32 to $10^{6}$ labels for each model. We measure performance using probe error, d
    
[^110]: 重新思考SAE评估：原子可解释单元框架揭示隐藏的多义性与冗余性

    Rethinking SAE Evaluation: An Atomic Interpretable Unit Framework Reveals Hidden Polysemanticity and Redundancy

    [https://arxiv.org/abs/2610.30474](https://arxiv.org/abs/2610.30474)

    本文提出了一种基于原子可解释单元的SAE评估新框架，将超越单义性的评估目标形式化，并借此揭示了现有SAE特征中隐藏的多义性与冗余性。

    

    稀疏自编码器（SAE）最初是为发现单义特征而提出的，单义性是刻画单个特征的性质，但并未规定学习到的特征集合整体上应该是什么样子。随着SAE越来越多地被应用于归因分析、回路发现和干预，研究者们开始使用描述特征集合全局性质的评估指标，例如稀疏性、死特征比例、重建保真度和解耦性等，这一系列指标已远远超出单义性的范畴。然而，这种超越单义性的隐式转变尚未被形式化为一个清晰、明确定义的、先于这些评估指标存在的目标。缺少这样的目标，现有指标仍然是临时拼凑的，其必要性缺乏依据。在这项工作中，我们通过为SAE评估引入一种新范式来形式化这一最终目标：要求SAE特征，以及更广义的字典……（摘要不完整）

    arXiv:2610.30474v1 Announce Type: cross  Abstract: Sparse autoencoders (SAEs) were originally motivated by discovering monosemantic features, a property that characterizes individual features but does not specify what the learned feature set should look like as a whole. As SAEs have increasingly been adopted for attribution, circuit discovery, and intervention, researchers have begun evaluating metrics that describe global properties of the feature set, such as sparsity, dead feature rates, reconstruction fidelity, and disentanglement, a diverse set that extends far beyond monosemanticity. However, this implicit shift beyond monosemanticity has not been formalized as a clear, well-defined objective that comes before these evaluation metrics. Without such an objective, existing metrics remain ad hoc, with their necessity unjustified. In this work, we formalize this ultimate objective by introducing a new paradigm for SAE evaluation: requiring that SAE features, and more generally dictio
    
[^111]: OACap：面向详细视频描述的描述引导式主动观察

    OACap: Caption-Guided Active Observation for Detailed Video Captioning

    [https://arxiv.org/abs/2610.29327](https://arxiv.org/abs/2610.29327)

    提出了以描述为中心的智能体框架 OACap，它能将视频描述中缺失或缺乏支持的信息转化为主动观察请求，借助时空工具获取证据并迭代验证修订描述，从而显著提升详细视频描述质量，同时引入 FSCap-Bench 基准及 CSFS 等评分指标来系统评估描述的事实性与语义保真度。

    

    与问答任务不同（问答中的查询已明确了感知目标），开放式详细视频描述必须自主决定需要获取什么证据，以及如何将这些证据整合成连贯的全局描述。我们提出了观察自适应视频描述器，这是一个以描述为中心的智能体，它将描述中缺乏支持或缺失的信息转化为有针对性的观察请求，利用时空工具获取证据，并迭代地验证和修订描述内容。在使用固定的纯文本阅读器的情况下，OACap 生成的描述在多个通用视频理解基准上均取得了一致的性能提升。为了更好地诊断这些描述所保留的信息，我们引入了事实-语义描述基准（FSCap-Bench）。其中的核心支撑事实性得分（CSFS）将断言精确度与重要性感知的事实覆盖率相结合，在 93.0-98.3% 的有效描述对上对四种受控扰动类型做出了符合预期的响应。其跨模态语义往返得分（摘要原文在此处截断）

    arXiv:2610.29327v1 Announce Type: cross  Abstract: Unlike question answering, where a query specifies a perceptual target, open-ended detailed captioning must determine what evidence to acquire and how to integrate it into a coherent global description. We propose Observation-Adaptive Video Captioner (OACap), a caption-centered agent that turns unsupported or missing information into targeted observation requests, uses spatiotemporal tools to acquire evidence, and iteratively verifies and revises the caption. OACap captions consistently improve performance across general video-understanding benchmarks using a fixed text-only reader. To better diagnose the information they preserve, we introduce Factual-Semantic Captioning Bench (FSCap-Bench). Its Core-Support Factuality Score (CSFS) combines claim precision with importance-aware fact coverage and responds as expected to four controlled perturbation types on 93.0-98.3% of valid caption pairs. Its Cross-Modal Semantic Round-Trip Score (C
    
[^112]: SANE：面向LLM评判器的自动化的评分器特定变异测试

    SANE: Automated Scorer-Specific Metamorphic Testing for LLM Judges

    [https://arxiv.org/abs/2610.28886](https://arxiv.org/abs/2610.28886)

    SANE是一个基于智能体的自动化框架，能针对特定LLM评判器自动识别其评分能力、生成变异测试配方及预期分数关系，从而将评判器可靠性测试实现全流程自动化，摆脱了人工编写扰动配方和预设评分器类型的限制。

    

    LLM评判器（LLM judges）在基准评估、模型选择、数据筛选和模型训练中扮演着越来越重要的角色。只有当每个评判器能够对影响其评分标准的变动做出可靠响应时，其决策才有价值。现有的可靠性测试方法通过对输入施加具有已知预期评分影响的扰动来进行测试。主流的LLM评判器测试框架将人工编写的扰动配方与合成测试生成相结合，但将针对特定评分器的配方构思排除在自动化流程之外。这限制了它们只能使用固定的扰动方式和预先假设的评分器类型。我们提出了SANE，一个基于智能体（agent）的框架，它将随机评估器的行为需求编码为可复用的智能体指令。给定评分器规范、实现代码和代表性输入，该流水线首先识别评分器特定的能力，然后为每种能力自动推导扰动配方和预期的分数关系。SANE还定义了一个（摘要内容在此处被截断）

    arXiv:2610.28886v1 Announce Type: new  Abstract: LLM judges increasingly mediate benchmark evaluation, model selection, data curation, and training. Their decisions are useful only when each judge responds reliably to changes affecting its scoring criterion. Existing methods for reliability testing perturb inputs with known expected effects on judge scores. Prominent LLM-judge testing harnesses combine human-authored perturbation recipes with synthetic test generation, leaving scorer-specific recipe ideation outside the automated pipeline. This limits their use to fixed perturbations and pre-assumed scorer types. We introduce SANE, an agent-based framework that encodes behavioral requirements for stochastic evaluators as reusable agent instructions. Given a scorer specification, implementation, and representative inputs, the pipeline first identifies scorer-specific capabilities, then automatically derives perturbation recipes and expected score relations for each. SANE also defines a 
    
[^113]: ASTOR：基于形式文法的软约束优化自动建模的基准测试与改进

    ASTOR: Benchmarking and Improving Autoformulation for Soft-Constraint Optimization with a Formal Grammar

    [https://arxiv.org/abs/2610.26819](https://arxiv.org/abs/2610.26819)

    本文提出ASTOR框架，通过形式文法引导的抽象语法树将自然语言决策问题确定性地编译为可执行的优化模型，填补了软约束优化自动建模研究的空白并提供了系统基准测试。

    

    现实世界的决策问题同时包含两类性质截然不同的约束：必须满足的要求，称为硬约束；以及可以被违反的偏好，称为软约束。由于这些问题通常以自然语言的形式呈现给决策者，大语言模型为辅助解决此类问题提供了天然的工题。这催生了“自动建模”（autoformulation）研究方向，其重点在于利用大语言模型将决策问题的自然语言描述转化为可执行的数学优化模型。然而，现有研究往往忽视了消费者、管理者和规划者所提出的软约束，尽管许多实际问题需要显式地建模这些约束。我们提出ASTOR，一个推理时框架，它先将自然语言决策问题映射为文法引导的抽象语法树（AST），然后确定性地将所得的表示编译为可执行的优化模型。

    arXiv:2610.26819v2 Announce Type: replace  Abstract: Real-world decision problems combine two qualitatively different kinds of constraints: requirements that must be satisfied, called hard constraints, and preferences that can be violated, called soft constraints. Because these problems are often posed to decision-makers in natural language, large language models offer a natural tool for assisting with their solution. This has given rise to autoformulation, which focuses on using LLMs to translate natural-language descriptions of decision problems into executable mathematical optimization models. However, this research often neglects soft constraints like those proposed by consumers, managers, and planners, despite the fact that many practical problems require modeling them explicitly. We introduce ASTOR, an inference-time framework that maps a natural-language decision problem to a grammar-guided abstract syntax tree (AST) before deterministically compiling the resulting representatio
    
[^114]: AfriGuard：面向非洲语言的安全性与稳健拒答机制

    AfriGuard: Safety and Robust Refusal For African Languages

    [https://arxiv.org/abs/2610.26474](https://arxiv.org/abs/2610.26474)

    该论文提出了AfriGuard，一个覆盖10种非洲语言和英语的扎根文化背景的多语言数据集，用于研究有害意图检测、安全类别理解与拒答行为，并揭示了现有护栏模型在这些安全任务上表现不佳、对适当安全行为理解有限的现状。

    

    arXiv:2610.26474v2 公告类型：替换 摘要：旨在与用户交互的大语言模型（LLM）通常被训练以对有害或有毒内容做出适当回应。然而，在多样化的语言和文化语境下，确保适当回应变得更具挑战性，因为安全风险的表达和解读方式可能各不相同。因此，构建安全的AI系统需要检测安全风险、理解其性质并做出适当回应，同时还要兼顾在每个阶段都增加复杂性的语言和文化差异。在这项工作中，我们提出了AfriGuard，一个扎根于文化背景的多语言数据集，用于研究10种非洲语言和英语中的有害意图、安全类别理解以及拒答行为。我们的研究结果表明，护栏（guardrail）模型在安全意图和安全类别分类任务上的表现均不及开源和闭源大语言模型，同时各模型对适当安全行为的理解也十分有限。

    arXiv:2610.26474v2 Announce Type: replace  Abstract: Large language models (LLMs) designed to interact with users are typically trained to respond appropriately to harmful or toxic content. However, ensuring appropriate responses becomes more challenging in diverse linguistic and cultural contexts, where safety risks may be expressed or interpreted differently. Building safe AI systems therefore requires detecting safety risks, understanding their nature, and responding appropriately, while accounting for linguistic and cultural variations that add complexity to each stage. In this work, we introduce AfriGuard, a culturally grounded multilingual dataset for studying harmful intent, safety-category understanding, and refusal behavior across 10 African languages and English. Our findings reveal that guardrail models underperform both open- and closed-source LLMs on safety-intent and safety-category classification, while models show limited understanding of appropriate safety behavior, sc
    
[^115]: 词元集合选择混淆POPE基准：VLM幻觉评估中是/否提取的系统性审计

    Token-Set Choice Confounds POPE: A Systematic Audit of Yes/No Extraction in VLM Hallucination Evaluation

    [https://arxiv.org/abs/2610.25964](https://arxiv.org/abs/2610.25964)

    该论文系统审计了POPE幻觉评估中的是/否答案提取方式，发现传统双词元logit读出所比较的"yes"/"no"词元ID在LLaVA-1.5-7B的实际输出中从未出现过（模型实际生成的是带空格前缀的" Yes"/" No"），仅这一读出实现细节就能使F1分数偏移高达6.13分，说明评估代码的实现选择会严重混淆基准对比结论。

    

    我们试图复现POPE基准上近期的一个幻觉缓解结果。我们测得的LLaVA-1.5-7B基线在对抗性分割上达到82.21 F1，比我们计划对比的对抗性基线（81.00、79.30和73.54）高出1.21至8.67分。我们检查了评估代码，发现一个仅凭自身就能改变F1分数的读出选择：双词元logit读出方式通过比较两个词表ID——3582（"yes"）和1217（"no"）——来判定是与否。但在跨越全部三个分割的9,000个贪心解码问题上，这两个ID一次也从未被生成过。模型实际输出的是“ Yes”（词元3869）和“ No”（词元1939），因为SentencePiece在提示词后缀"ASSISTANT:"之后会前置一个空格字节。改用覆盖带空格前缀表层形式的八字元查找来读取结果时，LLaVA-1.5-7B在对抗性、热门和随机分割上的F1分别为0.8221、0.8498和0.8713。仅这一对抗性分割上的6.13分偏移就超过了……（摘要原文在此处被截断）

    arXiv:2610.25964v2 Announce Type: replace-cross  Abstract: We set out to reproduce a recent hallucination-mitigation result on the POPE benchmark. The LLaVA-1.5-7B baseline we measured, 82.21 F1 on the adversarial split, sat 1.21 to 8.67 points above the adversarial baselines we planned to compare against (81.00, 79.30, and 73.54). We looked at the evaluation code and found a readout choice that moves F1 on its own. A two-token logit readout decides yes versus no by comparing two vocabulary IDs, 3582 ("yes") and 1217 ("no"). Across 9,000 greedy-decode questions spanning all three splits, neither ID is generated even once. The model emits " Yes" (token 3869) and " No" (token 1939), because SentencePiece prepends a space byte after the prompt suffix "ASSISTANT:". Read with an eight-token lookup that covers the space-prefixed surface forms, LLaVA-1.5-7B scores F1 0.8221, 0.8498, and 0.8713 on the adversarial, popular, and random splits. That 6.13-point adversarial shift exceeds the advers
    
[^116]: EvoSkill v2：从基准适配到有状态演化器的全智能体技能演化

    EvoSkill v2: Fully Agentic Skill Evolution from Benchmark Adaptation to a Stateful Evolver

    [https://arxiv.org/abs/2610.25931](https://arxiv.org/abs/2610.25931)

    EvoSkill v2 提出了一种由三个智能体组成的技能演化系统，其核心创新在于采用一个贯穿全程、带有 bash 工具的有状态编码智能体在每个训练周期持续重写技能指令，并结合多次尝试与无筛选接受策略，在不改动底层模型的前提下提升智能体的基准测试表现。

    

    EvoSkill v2 是一个三智能体系统，它通过编辑智能体所获得的指令（技能）来提升性能，同时保持其底层模型和运行框架不变。用户通过描述一个基准测试并提供一个评分器来启动运行，基准适配器会将其转换为针对该基准的专用插件。求解器尝试完成任务，演化器则读取求解器的训练轨迹并在每个周期重写技能。我们将演化器实现为一个带有 bash 工具、在整个运行过程中持续存在的有状态编码智能体，这不同于我们所调研的技能优化器中最常见的两种设计：如 SkillOpt 那样的一系列单次模型调用，以及如 SkillBoost 那样每轮从零开始的智能体。我们还测量了求解器的轨迹在多次尝试之间的变化程度，为演化器提供每个训练任务的多次尝试机会，并直接接受每个周期产生的技能，而不是依据嘈杂的分数在其中进行筛选。在细粒度真实标签的条件下，EvoSkill v2 的技能将……（摘要原文在此处截断）

    arXiv:2610.25931v2 Announce Type: replace  Abstract: EvoSkill v2 is a three-agent system that edits the instructions (skills) an agent is given, leaving its model and harness intact. The user starts a run by describing a benchmark and supplying a grader, which a benchmark adaptor turns into a benchmark-specific plugin. A solver attempts the tasks, and an evolver reads the solver's training rollouts and rewrites the skills at each epoch. We implement the evolver as one stateful coding agent with a bash tool for the whole run, unlike the two most common designs among the skill optimizers we surveyed: a sequence of single model calls, as in SkillOpt, and an agent that starts afresh each round, as in SkillBoost. We also measure how much the solver's rollouts vary from one attempt to the next, give the evolver several attempts per training task, and accept every epoch's skills instead of selecting among them on noisy scores. With fine-grained ground truth, EvoSkill v2's skills raised the he
    
[^117]: Muse Glimmer 30B：一个端侧智能体模型

    Muse Glimmer 30B: An On-Device Agentic Model

    [https://arxiv.org/abs/2610.24628](https://arxiv.org/abs/2610.24628)

    Muse Glimmer是一个开源的30B参数端侧智能体模型，集成了多步推理、可靠的工具使用和故障恢复能力，可在消费级硬件上本地运行且无需云端，并在智能体基准测试中超越Gemma4-31B和Qwen3.6-27B等同规模模型达到最先进性能。

    

    我们介绍了Muse Glimmer，这是一个开源的300亿参数因果语言模型，配备感知编码器，专为在消费级硬件上执行自主智能体任务而打造。Muse Glimmer将多步推理、可靠的工具使用和故障恢复能力集成到单一模型中，可在消费级设备上本地运行，无需依赖云端基础设施。该模型接受文本和图像输入并产生文本输出，训练数据涵盖一百多种语言，并被量化至约4比特精度以实现高效的单GPU推理。Muse Glimmer在智能体基准测试中取得了同规模模型中的最先进性能，在工具使用、多步规划和智能体任务完成方面，相比同规模模型Gemma4-31B和Qwen3.6-27B树立了新标准。我们以Apache 2.0许可证发布该模型，以促进功能强大、安全、本地化AI智能体的开源发展。

    arXiv:2610.24628v2 Announce Type: replace-cross  Abstract: We introduce Muse Glimmer, an open-source 30B parameter causal language model with perception encoder, purpose-built for autonomous agentic tasks on consumer hardware. Muse Glimmer integrates multi-step reasoning, reliable tool use and failure recovery into a single model that runs locally on consumer devices without requiring cloud infrastructure. The model accepts text and image inputs and produces text outputs, was trained on data from over a hundred languages, and is quantized to approximately 4-bit precision for efficient single-GPU inference. Muse Glimmer achieves state-of-the-art performance for its size class on agentic benchmarks, setting a new standard compared to the models of its size: Gemma4-31B and Qwen3.6-27B on tool use, multi-step planning, and agentic task completion. We release the model under the Apache 2.0 license to foster open-source development of capable, safe, local AI agents.
    
[^118]: Chinese Decision（中文决策）：面向单次前向概率决策的语言模型系列

    Chinese Decision: A Family of Language Models for Single-Pass Probabilistic Decision-Making

    [https://arxiv.org/abs/2610.23170](https://arxiv.org/abs/2610.23170)

    Chinese Decision 是一系列中文语言模型，通过单次前向传播直接对候选选项的 logits 进行归一化得到概率分布，无需自回归生成，在医学、法律、金融等领域的决策任务上实现了高准确率与良好校准。

    

    许多基于语言的应用需要对有限决策集合输出概率分布，而非开放式回答。我们提出了 Chinese Decision（中文决策），一个面向单次前向概率决策的中文语言模型系列。给定上下文、指令和候选描述，模型在单次前向传播中提取末尾位置的候选字母 logits，并在有效选项上进行归一化，无需自回归式答案生成。统一的共享接口支持布尔判断、类别选择和序数评分。我们报告了一个已完成的 0.8B 模型，该模型使用分布交叉熵在 1304.8 万个决策上训练；此外还报告了经过额外 RLCD 目标训练的初步 2B 检查点。在包含 307,900 个决策的完整 CJ-Bench 上，未经校准的 0.8B 模型仅使用单一检查点，在通用领域达到 72.24% 的准确率，在医学、法律和金融领域平均达到 76.98% 的准确率，整体期望校准误差为 1.07%。

    arXiv:2610.23170v1 Announce Type: new  Abstract: Many language-based applications require probabilities over a finite set of decisions rather than open-ended responses. We present Chinese Decision, a family of Chinese-language models for single-pass probabilistic decision-making. Given context, an instruction, and candidate descriptions, the model extracts final-position candidate-letter logits in one forward pass and normalizes them over valid options, without autoregressive answer generation. A shared interface supports Boolean judgments, categorical choices, and ordinal scores. We report a completed 0.8B model trained with distribution cross-entropy on 13.048 million decisions and preliminary 2B checkpoints trained with an additional RLCD objective. On the full CJ-Bench of 307,900 decisions, the uncalibrated 0.8B model achieves 72.24% general-domain accuracy and 76.98% average accuracy across medicine, law, and finance using a single checkpoint, with 1.07% overall expected calibrati
    
[^119]: 相同上下文，不同智能体：数据智能体中语义层、编排与压缩的行为研究

    Same Context, Different Agents: A Behavioral Study of Semantic Layers, Orchestration and Compaction in Data Agents

    [https://arxiv.org/abs/2610.20959](https://arxiv.org/abs/2610.20959)

    该论文首次系统性地研究了语义层、编排与压缩三种上下文工程技术在数据智能体中单独及组合使用时的行为、准确性与成本影响，发现无这些技术的智能体会呈现调查者、过早回答者和过早停止者三种行为群体，且每种技术的收益取决于其所匹配的智能体行为群体。

    

    回答异构数据上的问题意味着需要整合来自文件、数据库、文档和网页的证据。数据智能体（Data Agents）可以实现这一过程的自动化，但其成败取决于上下文。三种上下文工程技术（Context-Engineering Techniques，CETs）通过塑造上下文来发挥作用：语义层用于描述数据，编排（orchestration）用于在多个智能体之间分配工作，压缩（compaction）用于精简交互历史。现有研究仅单独验证了每种CET的效果，而智能体行为如何影响其作用仍是一个悬而未决的问题。我们对这三种CET进行了全面分析，涵盖单独使用与组合使用的情形，涉及七个大语言模型（LLMs）在来自五个基准的1,514个长时程任务上的表现，考察维度包括行为、准确性和成本。我们发现，在没有CET的情况下，数据智能体会形成三种行为群体：深入调查者、不检查数据就过早作答的“过早回答者”，以及中途放弃的“过早停止者”。每种CET都带来了其已知的好处——从促使更多调查到缩小提示规模——但只有与其相匹配的行为群体……（原文摘要至此截断）

    arXiv:2610.20959v1 Announce Type: cross  Abstract: Answering questions over heterogeneous data means combining evidence from files, databases, documents and the web. Data agents automate this, but their success depends on context. Three context-engineering techniques (CETs) help by shaping it: semantic layers describe the data, orchestration splits work among agents, and compaction condenses interaction history. Existing studies validate each CET alone, leaving open how agent behavior shapes its effect. We present a comprehensive analysis of the three CETs, alone and combined, with seven large language models (LLMs) on 1,514 long-horizon tasks from five benchmarks, covering behavior, accuracy and cost. We find that data agents without CETs form three behavior groups: investigators, premature answerers that answer without inspecting data, and early stoppers that give up. Each CET delivers its known benefit, from more investigation to smaller prompts, but only the behavior group it match
    
[^120]: 跨模型上下文内神经反馈揭示大语言模型中的（有限）激活控制能力

    Cross-Model In-Context Neurofeedback Reveals (Limited) Activation Control in LLMs

    [https://arxiv.org/abs/2610.20179](https://arxiv.org/abs/2610.20179)

    本文提出跨模型上下文内神经反馈新范式，首次将模型在任务上的胜任能力与对自身激活的真实访问区分开来，测试十个开源大模型后发现大语言模型仅具备有限的激活控制能力。

    

    基于激活向量的探针被广泛用于检测语言模型的不良行为，但其有效性依赖于一个假设：模型无法控制其激活所编码的内容。人类可以通过神经反馈学会控制甚至单个神经元的放电，近期有研究探讨了这种能力是否也适用于语言模型，但得出了相互矛盾的结论。迄今为止，尚无实验设计能够将模型在任务上的表现能力与其对自身激活的真实访问区分开来。我们通过“上下文内神经反馈”会话填补了这一空白：在其中，一个“受试”模型被要求最大化（或最小化）一个由其自身激活计算得出的分数，且该分数会在每一轮对话中展示给该模型。在我们的第一个实验中，句子要么由受试模型自己生成，要么由第二个“写作”模型生成，而分数始终来自受试模型的激活。我们测试了来自五个系列的十个开源权重模型，规模从9B到32B。部分模型……（原文摘要在此处被截断）

    arXiv:2610.20179v1 Announce Type: new  Abstract: Probes on activation vectors are widely used to detect misbehavior in language models, but their success relies on the assumption that models cannot control what their activations encode. Humans can learn, through neurofeedback, to control the firing of even single neurons, and recent studies investigated whether this ability extends to language models, reaching conflicting conclusions. No experimental design so far disentangles competence at the task from genuine access to one's own activations. We fill this gap with in-context neurofeedback sessions, in which a "subject" model is asked to maximize (or minimize) a score derived from its activations, shown back to it at every turn. In our first experiment, the sentences are generated either by the subject or by a second "writer" model, while the score always derives from the activations of the subject. We test ten open-weight models from five families, with sizes from 9B to 32B. Some mod
    
[^121]: 循环式自我提升：面向循环语言模型的动态跨环在线策略蒸馏

    Recurrent Self-Improvement: Dynamic Cross-Loop On-Policy Distillation for Looped Language Models

    [https://arxiv.org/abs/2610.10623](https://arxiv.org/abs/2610.10623)

    LoopOPD让循环语言模型利用自身更深层循环计算作为冻结“教师”，在学生自生成轨迹上进行在线策略蒸馏，无需外部教师或特权信息即可获得密集监督实现自我提升，D-LoopOPD进一步将该过程动态化。

    

    循环语言模型通过在循环计算步骤中复用共享参数，提供了一种参数高效地扩展推理能力的方法。尽管前景广阔，循环语言模型的有效后训练仍然充满挑战。现有方法要么提供稀疏的、或跨循环扩展代价高昂的基于奖励的监督，要么依赖外部教师模型或特权信息，导致教师模型可用性受限或师生上下文不匹配。为解决这些局限，我们提出LoopOPD，这是一个跨环在线策略蒸馏框架，它利用循环语言模型内部额外的循环计算作为其自身的监督来源。LoopOPD在学生自身生成的轨迹上，以冻结的终端循环策略作为具有计算特权的教师来指导中间循环的学生，从而在无需外部教师或特权信息的情况下提供密集监督。我们进一步提出动态LoopOPD（D-LoopOPD），它持续……（原文摘要在此处截断）

    arXiv:2610.10623v1 Announce Type: cross  Abstract: Looped Language Models (LoopLMs) offer a parameter efficient approach to scaling reasoning by reusing shared parameters across recurrent computation steps. Despite their promise, effective post-training of LoopLMs remains challenging. Existing approaches either provide reward based supervision that is sparse or costly to extend across loops, or rely on external teachers or privileged information, leading to limited teacher availability or teacher-student context mismatch. To address these limitations, we introduce LoopOPD, a cross-loop on-policy distillation framework that uses additional recurrent computation within a LoopLM as its own source of supervision. LoopOPD uses a frozen terminal loop policy as a compute privileged teacher for an intermediate loop student on student generated rollouts, providing dense supervision without an external teacher or privileged information. We further propose Dynamic LoopOPD (D-LoopOPD), which conti
    
[^122]: 宪法引导的水印技术

    Constitution-Guided Watermarking

    [https://arxiv.org/abs/2610.09552](https://arxiv.org/abs/2610.09552)

    本文提出“宪法引导水印”框架，通过将提供商需求表示为自然语言原则，使水印系统能够根据不同请求的需求灵活选择属性权衡，避免了传统方法对所有请求采用统一配置所导致的牺牲问题。

    

    水印技术使语言模型提供商能够识别由其模型生成的文本。然而，水印的理想属性之间可能存在冲突（即更强的水印信号可能会降低文本质量），而抵抗编辑的设计也可能便于伪造。提供商通过选择平衡竞争目标或优先考虑特定属性的配置来应对这些权衡。但这两种方法都会将一个统一的操作点强加于具有不同需求的请求上，可能在需要保留原始措辞时牺牲文本质量，或在需要可靠归因时牺牲鲁棒性。为了实现灵活且可适应的设计，我们提出了宪法引导水印，这是一个能够根据提供商需求（以自然语言原则的形式列出）来为每个请求选择合适权衡的框架。在离线阶段，一个预训练的推理代理会结合水印实现来审查宪法规则，并迭代地重新……

    arXiv:2610.09552v1 Announce Type: cross  Abstract: Watermarking enables language model providers to identify text generated by their models. However, its desired properties can conflict (\ie~stronger watermark signals can degrade text quality), while designs that resist editing may also facilitate forgery. Providers address these trade-offs by choosing configurations that balance competing objectives or prioritize particular properties. Either approach imposes a shared operating point on requests with different requirements, potentially sacrificing quality where wording preservation matters or robustness where reliable attribution is essential. To allow flexible and adaptable designs, we introduce \emph{Constitution-Guided Watermarking}, a framework that selects request-appropriate trade-offs from provider requirements, listed as natural-language principles. \emph{Offline}, a pretrained reasoning agent examines constitutional rules alongside watermark implementations and iteratively re
    
[^123]: ToolRACER：一个用于智能体训练与评估的鲁棒智能体会话模拟资源

    ToolRACER: A Robust Agentic Conversation Emulation Resource for Agent Training and Evaluation

    [https://arxiv.org/abs/2610.09163](https://arxiv.org/abs/2610.09163)

    该论文提出ToolRACER合成数据生成流水线，通过协调用户、助手和工具模拟模型生成含对抗性行为的多轮对话数据，并构建了包含5.6K条验证对话轨迹（约66%含易失败场景）的ToolRACERBench基准，用于训练和评估鲁棒的任务导向对话智能体。

    

    任务导向的对话智能体在真实世界的对话场景中仍然十分脆弱，因为对话很少遵循可预测的脚本，尤其是当用户表现出不合作行为时。现有的函数调用基准通常强调成功的、合作的交互，而对对抗性对话轨迹的表达不足，从而限制了用于开发鲁棒智能体的训练资源。我们提出了ToolRACER，这是一个合成数据生成流水线，通过协调用户、助手和工具模拟模型来生成并验证用户与智能体之间的多轮交互。利用ToolRACER，我们构建了ToolRACERBench，这是一个覆盖六个领域、涵盖55种不同人设的鲁棒多轮对话基准，生成了一个经过验证的包含5.6K对话轨迹的语料库，其中约66%的对话包含容易导致失败的对话场景。我们注入对抗性行为，从而……（原文截断）

    arXiv:2610.09163v1 Announce Type: new  Abstract: Task-oriented conversational agents remain fragile under real world conversation scenarios as they rarely follow a predictable script, especially when users exhibit non-cooperative behavior. Existing function-calling benchmarks often emphasize successful, cooperative interactions and underrepresent adversarial conversation trajectories, thereby limiting the training resources available for developing robust agents. We present ToolRACER, a synthetic data generation pipeline that coordinates user, assistant and tool emulation models to generate and validated multi-turn interactions between a user and an agent. Using \sysn, we construct ToolRACERBench a robust multi-turn conversation benchmark spanning six domains, ranging over 55 varied personas, generating a validated corpus of 5.6K conversation trajectories, with approximately 66\% of conversations containing failure-prone conversation scenarios. We inject adversarial behaviors, producin
    
[^124]: SOL：利用双切片Wasserstein度量衡量文本分布之间的差距

    SOL: Measuring Gaps between Text Distributions by Double Sliced Wasserstein Metrics

    [https://arxiv.org/abs/2610.06513](https://arxiv.org/abs/2610.06513)

    提出SOL——一种基于固定Transformer隐藏状态经验测度的双切片Wasserstein距离的文本分布距离度量，当Transformer为单射时可证明其为真正的度量，为非自回归语言模型的分布拟合评估提供了稳定的样本级评估方案。

    

    评估文本生成需要衡量生成分布与数据分布的匹配程度。对于自回归模型，这通过困惑度来实现；而扩散模型和基于流的语言模型只能提供似然界，其紧密程度在不同模型家族之间存在差异。基于样本的替代方法（如结合熵的生成式困惑度）则未考虑分布拟合情况。我们提出了SOL，一种文本分布之间的距离度量。每个序列由其在固定Transformer下隐藏状态的经验测度来表示，并通过双切片Wasserstein距离来比较这些测度的分布。我们证明，当Transformer是单射时，SOL是一个真正的度量。实验表明，SOL能够检测分布性失败、恢复预期的模型趋势，并提供稳定的基于样本的估计。我们提出SOL以填补当前非自回归模型评估协议中的空白。

    arXiv:2610.06513v2 Announce Type: replace  Abstract: Evaluating text generation requires measuring how well the generated distribution matches the data distribution. For autoregressive models, this is done by the perplexity. Diffusion and flow-based language models can only provide a likelihood bound, whose tightness differs between model families. Sample-based substitutes such as generative perplexity with entropy do not consider the distribution fit. We propose SOL,   a distance between text distributions. Each sequence is represented by the empirical measure of its hidden states under a fixed transformer and the distributions of these measures are compared by the double sliced Wasserstein distance. We prove that SOL is a metric if the transformer is injective. Experiments show that SOL detects distributional failures, recovers expected model trends, and provides stable sample-based estimates. We put forward SOL to fill the gap in the current evaluation protocol used for non auto-reg
    
[^125]: 一种面向模式即代码生物医学命名实体识别的指南增强多智能体框架

    A Guideline-Augmented Multi-Agent Framework for Schema-as-Code Biomedical Named Entity Recognition

    [https://arxiv.org/abs/2610.02970](https://arxiv.org/abs/2610.02970)

    GAMA框架通过从标注数据中归纳并验证特定数据集的标注规则来构建指南记忆，结合多智能体协作与模式即代码的结构化输出，显著提升了生物医学命名实体识别的准确性和格式规范性。

    

    大语言模型（LLMs）通过指令遵循和上下文学习在生物医学命名实体识别方面展现出了可喜的潜力。然而，现有的基于LLM的BioNER方法仍然面临两个关键局限。首先，检索到的示例和外部生物医学知识对特定数据集的标注语义支持有限，导致实体边界、类型范围和标注规范模糊不清。其次，自由形式的生成缺乏足够的结构控制，常常导致无效格式、幻觉提及、重复实体和边界错误。为了解决这些局限，我们提出了GAMA，一个面向模式即代码BioNER的指南增强多智能体框架。GAMA首先从已标注的训练实例中归纳候选标注规则，并通过标注数据进行验证，以构建可靠的特定数据集指南记忆。在这些经过验证的规则的指导下，一个规划……（摘要在此处截断）

    arXiv:2610.02970v1 Announce Type: new  Abstract: Large language models (LLMs) have shown promising potential for biomedical named entity recognition (BioNER) through instruction following and in-context learning. However, existing LLM-based BioNER methods still face two key limitations. First, retrieved demonstrations and external biomedical knowledge provide limited support for dataset-specific annotation semantics, leaving entity boundaries, type scopes, and annotation conventions ambiguous. Second, free-form generation lacks sufficient structural control, often leading to invalid formats, hallucinated mentions, duplicated entities, and boundary errors. To address these limitations, we propose GAMA, a guideline-augmented multi-agent framework for schema-as-code BioNER. GAMA first induces candidate annotation rules from labeled training instances and verifies them against annotated data to construct reliable dataset-specific guideline memory. Guided by these verified rules, a planning
    
[^126]: 通过多编程语言指令微调与集成方法增强生物医学命名实体识别

    Enhancing Biomedical Named Entity Recognition via Multiple Programming Languages Instruction Tuning and Ensemble Method

    [https://arxiv.org/abs/2610.02949](https://arxiv.org/abs/2610.02949)

    该论文提出MITE方法，将生物医学命名实体识别重构为结构到结构的生成任务，通过多种编程语言的指令微调与集成策略，在不依赖昂贵外部知识资源的条件下提升了识别性能与模型鲁棒性。

    

    指令微调已成为将大语言模型（LLM）应用于生物医学命名实体识别（BioNER）的一种常见范式。然而，现有的指令微调方法仍然面临两个关键挑战。首先，传统的自然语言指令通常将BioNER标注序列化为扁平的文本输出，为带类型的实体抽取提供的结构约束有限。其次，高质量的生物医学标注数据有限，从单一的序列化输出形式中学习可能会限制结构多样性并降低模型的鲁棒性。尽管可以引入外部生物医学知识来缓解数据稀缺问题，但这通常需要昂贵的资源构建成本。为了应对这些挑战，我们提出了MITE，一种面向BioNER的多编程语言指令微调与集成方法。MITE将BioNER重新表述为结构到结构的生成任务，通过用编程语言表示指令和实体输出（原文在此处截断）。

    arXiv:2610.02949v1 Announce Type: new  Abstract: Instruction tuning has become a common paradigm for applying large language models (LLMs) to biomedical named entity recognition (BioNER). However, existing instruction-tuning approaches still face two key challenges. First, conventional natural-language instructions typically serialize BioNER annotations as flat textual outputs, providing limited structural constraints for typed entity extraction. Second, high-quality biomedical annotations are limited, and learning from a single serialized output form may restrict structural diversity and reduce model robustness. Although external biomedical knowledge can be introduced to alleviate data scarcity, it often requires costly resource construction. To address these challenges, we propose MITE, a Multiple Programming Languages Instruction Tuning and Ensemble method for BioNER. MITE reformulates BioNER as a structure-to-structure generation task by representing both instructions and entity ou
    
[^127]: 超越重叠：估计基准测试暴露的因果效应

    Beyond Overlap: Estimating the Causal Effect of Benchmark Exposure

    [https://arxiv.org/abs/2609.27176](https://arxiv.org/abs/2609.27176)

    提出LeakScale干预框架，通过控制基准家族私有信息的暴露并构建全新可执行任务，首次因果量化了训练数据污染对模型评估准确率的实际影响，发现暴露使准确率提升7.17至27.31个百分点。

    

    arXiv:2609.27176v1 公告类型：新论文 摘要：评估材料进入训练的证据并不能揭示这些材料对评估结果产生了多大影响。这一区分使得被污染的基准测试分数难以解读：来源追踪可以证实接触行为，但只有反事实分析才能量化可归因于这种接触的性能。我们提出了LeakScale，一个用于估计这一缺失数量的干预性框架。LeakScale创建全新的可执行任务，这些任务需要私有的、家族特定的信息（这些信息在公开任务中不存在，也无法从公开任务中推导得出），控制对该信息的访问，并估计由此产生的经过控制调整后的可执行准确率变化。在2,048个独特家族、两个模型家族、两个可执行领域以及262,144次生成的实验中，暴露在每个模型-领域组合中都提高了准确率，增益范围从+7.17到+27.31个百分点。这些发现区分了两个经常被混淆的实证问题：基准测试是否被污染，以及污染实际造成了多大影响。

    arXiv:2609.27176v1 Announce Type: new  Abstract: Evidence that evaluation material entered training does not reveal how much it affected evaluation. This distinction leaves a contaminated benchmark score difficult to interpret: provenance can establish contact, but only a counterfactual can quantify the performance attributable to that contact. We present LeakScale, an interventional framework for estimating this missing quantity. LeakScale creates fresh executable tasks that require private, family-specific information absent from and non-derivable from the public task, controls access to that information, and estimates the resulting control-adjusted change in executable accuracy. Across 2,048 unique families, two model families, two executable domains, and 262,144 generations, exposure improves accuracy in every model-by-domain combination, with gains ranging from +7.17 to +27.31 percentage points. These findings separate two empirical questions that are often conflated: whether benc
    
[^128]: BabelArena：面向大语言模型智能体的大规模多语言基准测试

    BabelArena: A Large-Scale Multilingual Benchmark for LLM Agents

    [https://arxiv.org/abs/2609.23490](https://arxiv.org/abs/2609.23490)

    提出了BabelFlow通用工作流和BabelArena大规模多语言基准测试（涵盖23种语言、16,146个实例），首次系统评估了LLM智能体的跨语言能力，发现前沿模型存在显著的跨语言性能差异且无单一模型全面领先。

    

    大语言模型（LLM）智能体越来越多地通过工具使用以及与用户和环境的交互来执行多步骤工作流。然而，当前的智能体评估主要以英语为中心，限制了我们对多语言环境下智能体能力的理解。我们提出了BabelFlow，这是一种通用于各类基准测试的智能体工作流，它通过分析运行时依赖关系、协调保持结构的翻译，并将多层验证与人工审查相结合，将现有的智能体基准测试适配到新语言，以保留任务和评估语义。利用BabelFlow，我们构建了BabelArena，这是一个任务对齐的基准测试，包含16,146个实例，这些实例源自四个基准测试家族、13个领域和23种语言中的702个规范任务。对五个前沿模型的实验表明，没有任何单一模型能在所有基准测试家族中占据主导地位，且跨语言差异远不止体现在任务成功与否上。低资源语言……

    arXiv:2609.23490v1 Announce Type: new  Abstract: Large language model (LLM) agents increasingly execute multi-step workflows through tool use and interaction with users and environments. However, current agent evaluations are largely English-centric, limiting our understanding of agent capabilities in multilingual settings. We introduce BabelFlow, a benchmark-general agentic workflow that adapts existing agent benchmarks to new languages by analyzing runtime dependencies, coordinating structure-preserving translation, and combining multi-layer verification with human review to preserve task and evaluation semantics. Using BabelFlow, we construct BabelArena, a task-aligned benchmark comprising 16,146 instances derived from 702 canonical tasks across four benchmark families, 13 domains, and 23 languages. Experiments with five frontier models show that no single model dominates across benchmark families and that cross-language disparities extend well beyond task success. Lower-resource la
    
[^129]: 超越静态与线性：哪种注意力约束最能拟合人类阅读时间？

    Beyond Static and Linear: What Attention Constraints Best Fit Human Reading Times?

    [https://arxiv.org/abs/2608.23818](https://arxiv.org/abs/2608.23818)

    该论文系统比较了多种基于注意力的记忆约束机制（静态与动态），发现对中间词元内容敏感的约束最能拟合人类阅读时间数据。

    

    基于Transformer的语言模型被广泛用作人类语言处理的计算模型，然而其注意力机制允许对全部前文语境进行无损访问，这与人类有限的记忆系统不同。我们假设，在Transformer的注意力机制中引入记忆约束可以提升其对人类行为数据的拟合程度。以往的研究大多只孤立地探索过单一的约束方式，而我们对多种基于注意力的记忆机制进行了系统性比较，涵盖不同的模型规模和训练语料，并同时评估其对人类阅读时间的心理测量预测能力以及语法能力。此外，我们还比较了静态约束（即约束强度在整个训练过程中保持固定）与动态记忆课程式约束。我们发现，对中间词元内容敏感的约束始终与人类阅读时间保持最高的一致性，超越了……

    arXiv:2608.23818v2 Announce Type: replace  Abstract: Transformer-based language models are widely used as models of human language processing, yet their attention mechanisms allow lossless access to the full preceding context, unlike the limited memory systems of humans. We hypothesize that installing memory constraints into transformers' attention mechanisms can improve their fit to human behavioral data. While previous work has explored individual constraints in isolation, we conduct a systematic comparison of multiple attention-based memory mechanisms across different model sizes and training corpora, evaluating both psychometric predictive power for human reading times and grammatical competence. We additionally compare static constraints, in which the constraint strength is fixed throughout training, to dynamic memory curricula. We find that constraints that are sensitive to the content of intervening tokens consistently achieve the highest alignment with human reading times, outp
    
[^130]: TreeWY：面向门控DeltaNet混合模型的自适应验证方法

    TreeWY: Speculative Verification for Gated DeltaNet Hybrids

    [https://arxiv.org/abs/2608.20961](https://arxiv.org/abs/2608.20961)

    本文提出TreeWY方法，通过树状WY变换消除推测解码中的状态快照，显著降低内存开销，从而支持高接受率的宽草稿树。

    

    arXiv:2608.20961v1 公告类型：新论文  摘要：现代开源模型多为混合架构：大部分层采用线性注意力（门控DeltaNet，GDN）层，这些层携带一个小的固定大小的循环状态，而非不断增长的键值（KV）缓存。这使得普通解码在内存上高效，但不利于推测解码。为了验证一批草稿标记并回滚被拒绝的标记，当前系统在GDN层的每个草稿位置都会对完整循环状态进行快照，而这些快照无法在草稿树的分支间共享，因此宽且高接受率的树在内存上变得不可行。我们移除了快照。利用门控Delta规则的一种树状WY变换，我们通过一次三角求解计算每个草稿节点的输出，并在提交时仅重建一个被接受的状态，存储一个小型的伪值矩阵而非逐节点状态；该推导仅依赖于门控Delta规则，而不涉及任何其他架构细节。在两个规模的服务器基准测试中...

    arXiv:2608.20961v1 Announce Type: new  Abstract: Modern open models are hybrids: most layers are linear-attention (Gated DeltaNet, GDN) layers carrying a small fixed-size recurrent state instead of a growing key-value (KV) cache. This makes ordinary decoding memory-efficient, but hurts speculative decoding. To verify a batch of draft tokens and then roll back the rejected ones, today's systems snapshot the full recurrent state at every draft position for GDN layers, and those snapshots cannot be shared across branches of a draft tree, so a wide, high-acceptance tree becomes memory-infeasible. We remove the snapshots. Using a tree-structured WY transform of the gated delta rule, we compute every draft node's output with a single triangular solve and reconstruct only the one accepted state on commit, storing a small pseudo-value matrix instead of per-node states; the derivation depends only on the gated delta rule, not on any other architectural detail. In serving benchmarks on two scale
    
[^131]: 基于人格引导的大语言模型代理用于任务导向对话

    Persona-Guided LLM Agents for Task-Oriented Dialogue

    [https://arxiv.org/abs/2608.18085](https://arxiv.org/abs/2608.18085)

    本研究提出一个无需训练的框架，通过模拟两个大语言模型代理的任务导向对话，验证了系统适应用户人格能在不损害任务完成度的前提下提升交互质量。

    

    先前的研究表明，大型语言模型（LLMs）在开放式文本生成中能够表达多样的人格特质。然而，尚不清楚它们是否能在目标导向的对话中做到这一点而不损害任务完成度，以及适应用户的人格是否能提升交互质量。我们在任务导向对话（TOD）中研究这些问题，其中系统通过多轮交互帮助用户实现目标。我们构建了一个无需训练的框架，模拟两个LLM之间的TOD交互：一个表现出目标人格的用户代理，以及一个在完成任务的同时适应用户的系统代理。为了隔离适应的影响，我们在三种条件下变化系统对用户人格的了解程度。在“中性”条件下，系统不接收人格信息；在“尝试”条件下，它从对话线索中推断人格；在“预言”条件下，它被明确给予人格信息。

    arXiv:2608.18085v1 Announce Type: new  Abstract: Prior work has shown that large language models (LLMs) can express diverse personality traits in open-ended text generation. However, it remains unclear whether they can do so in a goal-directed dialogue without compromising task completion, and whether adapting to the user's personality improves the interaction quality. We study these questions in task-oriented dialogue (TOD), where a system helps a user accomplish a goal via multi-turn interaction. We build a training-free framework that simulates a TOD interaction between two LLMs: a user agent that exhibits a target personality and a system agent that adapts to the user while completing the task. To isolate the effect of adaptation, we vary how much the system knows about the user's personality across three conditions. In Neutral, the system receives no personality information. In Try, it infers the personality from dialogue cues. In Oracle, it is given the personality explicitly. We
    
[^132]: 每一枚硬币都有两面：关于大语言模型同策略蒸馏中泛化的双重性

    Every Coin Has Two Sides: On the Dual Nature of Generalization in On-Policy Distillation of Large Language Models

    [https://arxiv.org/abs/2608.16647](https://arxiv.org/abs/2608.16647)

    同策略蒸馏的泛化行为取决于教师和学生来源关系，同源对能跨域迁移推理能力，跨源对则受限于训练分布，这种双重性既是优势也是风险。

    

    同策略蒸馏（OPD）通过监督学生自身策略采样的轨迹来转移教师能力，但其泛化行为仍鲜为人知，因为大多数研究仅在单一领域和接近训练数据的基准上评估OPD。我们进行了一项受控研究，每次只改变一个泛化因素，从域内分布偏移到跨域迁移以及多教师设置。我们发现OPD转移的是教师的推理行为而非其对特定问题的答案：训练难度几乎无关紧要，甚至教师从未解决的问题也有用。迁移强烈依赖于教师和学生之间的来源关系：同源对使学生在语言、推理范围甚至其他领域中接近教师，而跨源对主要适应训练分布。这种广泛的影响是一把双刃剑。

    arXiv:2608.16647v1 Announce Type: new  Abstract: On-policy distillation (OPD) transfers teacher capabilities by supervising trajectories sampled from the student's own policy, yet its generalization behavior remains poorly understood, as most studies evaluate OPD on a single domain and on benchmarks close to the training data. We present a controlled study that varies one generalization factor at a time, from in-domain distribution shifts to cross-domain transfer and the multi-teacher setting. We find that OPD transfers a teacher's reasoning behavior rather than its answers to particular problems: training difficulty barely matters, and even problems the teacher never solves are useful. Transfer depends strongly on the origin relationship between teacher and student: same-origin pairs bring the student close to the teacher across languages, reasoning horizons, and even other domains, whereas cross-origin pairs mostly fit the trained distribution. This broad reach is a double-edged swor
    
[^133]: 当人类未言明语境时：隐式语境作为大语言模型的风险

    When Humans Leave Context Unsaid: Implicit Context as a Risk for Large Language Models

    [https://arxiv.org/abs/2608.09551](https://arxiv.org/abs/2608.09551)

    该论文提出了大语言模型的“隐式风险”这一新概念：由于预设、会话含义等隐式语境在训练数据中缺失且被对齐过程忽视，攻击者可引导模型推断提示中未言明的危害性语境，从而诱导其生成更多有害内容。

    

    人类语言植根于语用学，需要隐式（语言之外的）语境来解读语言，例如世界知识和社会规范。然而，这类语境通常是隐式的而非在语言中明确表达，需要人类自行推断理解语言所必需的底层语境信息。由于其隐式特性以及在文本语料库中的缺失，这类语境在后续的对齐过程中往往被忽视。在本文中，我们证明了训练数据中此类隐式语境的缺失是大语言模型的一种风险，我们将其称为大语言模型的“隐式风险”。为了利用这种隐式风险，我们提出了一种方法，引导大语言模型从暗示社会危害的提示中推断出隐式语境线索，如预设和会话含义，然后利用这些推断出的隐式语境来诱导模型生成更多对社会有害的内容。实验结果表明……

    arXiv:2608.09551v2 Announce Type: replace  Abstract: Human language is grounded in pragmatics, necessitating implicit (extralinguistic) context to interpret language, e.g., world knowledge and social norms. However, such context is often implicit rather than explicitly expressed in language, requiring humans to infer the underlying contextual information necessary for language understanding. Because of its implicit nature and absence from textual corpora, it is often overlooked in the subsequent alignment process. In this paper, we demonstrate that the absence of such implicit context in training data is a risk for LLMs, and we refer to this risk as the implicit risk for LLMs. To exploit this implicit risk, we propose a method that guides LLMs to infer implicit contextual cues, such as presuppositions and implicatures, from prompts that imply social harm, and then leverages these inferred implicit context to elicit additional socially harmful content. Experimental results demonstrate t
    
[^134]: 跨方言的罗马化阿拉伯语：观点、使用模式与语言变异

    Romanized Arabic Across Dialects: Views, Usage Patterns, and Linguistic Variation

    [https://arxiv.org/abs/2608.02555](https://arxiv.org/abs/2608.02555)

    本文通过调查阿拉伯语使用者并发布字符级词对齐资源，首次系统揭示了五种方言中阿拉伯语拉丁化书写的方言内规律性与方言间系统性变异。

    

    Arabizi（阿拉伯语拉丁化文字）是指用拉丁字母书写的阿拉伯语。尽管已有研究表明，Arabizi的流行程度和使用方式因地区、年龄组等因素而有所不同，但大多数针对阿拉伯语文本的自然语言处理（NLP）研究将其视为由于阿拉伯文字技术支持有限而产生的临时现象。在本工作中，我们与阿拉伯语使用者进行交流，收集他们对Arabizi的认知和使用情况的相关见解。我们进一步考察了不同方言使用者之间的书写规范，重点关注阿尔及利亚阿拉伯语、埃及阿拉伯语、黎巴嫩阿拉伯语、摩洛哥阿拉伯语和突尼斯阿拉伯语这五种方言。为此，我们发布了两个资源。首先，我们基于受调查参与者转写的单词，对阿拉伯语单词进行字符级对齐，以研究这五种方言内部及方言之间的变异，发现方言内部存在系统性的规律性，而方言之间则存在变异。其次，为了研究阿拉伯语使用者识别这种……（摘要原文在此处被截断）

    arXiv:2608.02555v2 Announce Type: replace  Abstract: Arabizi refers to Arabic written in Latin script. Although previous studies have shown that the prevalence and usage of Arabizi vary by factors such as region and age group, most NLP research on Arabic texts treats it as a temporary phenomenon resulting from limited technological support for the Arabic script. In this work, we engage with Arabic speakers to collect insights on their perceptions and usage of Arabizi. We further examine writing norms among speakers of different dialects, focusing on Algerian, Egyptian, Lebanese, Moroccan, and Tunisian Arabic. To this end, we release two resources. First, we perform a character-level alignment of Arabic words to study inter- and intradialectal variation across these five dialects, based on words transliterated by the surveyed participants (VARabiziLex), finding systematic intradialectal regularity and interdialectal variation. Second, to study Arabic speakers' ability to identify this s
    
[^135]: AgentGUI：用于观察和引导长时运行AI代理的界面

    AgentGUI: An Interface for Observing and Steering Long-Running AI Agents

    [https://arxiv.org/abs/2607.26300](https://arxiv.org/abs/2607.26300)

    AgentGUI是一个本地托管的图形界面，通过丰富的轨迹可视化、手动与自动引导以及多框架集成协调，显著提升了人类对长时运行AI代理的监督效率（轨迹分析提速38%）和小型本地代理的任务完成率（最高提升34个百分点）。

    

    AI代理在处理复杂、长时间运行的任务方面日益熟练。随着自主能力的快速增长，由于缺乏以人为中心的交互界面，人类监督正系统性地滞后。为解决这一问题，我们提出了AgentGUI，一个用户友好、本地托管的图形界面，用于在多个并发的长时间运行会话中无缝观察和引导AI代理。AgentGUI具有以下特点：1）丰富的代理轨迹可视化，2）有效的手动和自动引导功能，3）与开源及前沿代理框架的集成与协调。一项受控用户研究表明，从代理轨迹中识别关键要素的时间有统计学显著性的减少（快38%，p = 0.023）。在一项初步实验中，AgentGUI的自动防偏航功能在0.8B至9B的模型规模梯度上（N=50次运行）将小型本地代理的任务完成率提高了高达34个百分点。

    arXiv:2607.26300v3 Announce Type: replace-cross  Abstract: AI agents are increasingly adept at tackling complex, long-running tasks. With the rapid surge of autonomous capabilities, human oversight is systematically lagging behind due to limited human-centered interfacing. Aiming to address this, we introduce AgentGUI, a user-friendly, locally hosted GUI for seamlessly observing and steering AI agents amid multiple concurrent, long-running sessions. AgentGUI features 1) rich agent trajectory visualizations, 2) effective manual and automated steering, and 3) integration with and coordination between open-source and frontier agent frameworks. A controlled user study demonstrates statistically significant reduction in the time it takes to identify key elements from agent traces (38% faster, p = 0.023). In a preliminary experiment, AgentGUI's automated drift prevention feature raises the task completion rate of small local agents by as high as 34pp across a 0.8B--9B model ladder (N=50 runs
    
[^136]: 语码转换语音场景中人类与分类模型趋同行为的跨语言比较

    A Cross-lingual Comparison of Human and Classification Model Entrainment Behavior in Code-switched Speech Settings

    [https://arxiv.org/abs/2607.25202](https://arxiv.org/abs/2607.25202)

    该研究首次对语码转换口语对话中的人类趋同行为进行跨语言比较，发现词汇趋同具有跨语言普遍性而声学韵律与语码转换风格趋同依赖具体语境，且分类模型虽能检测趋同但所依赖的特征与人类行为中最显著的特征并不一致。

    

    会话趋同在单语和书面语境中已得到充分研究，但在口语语码转换（CSW）中仍未被充分探索。我们对普通话-英语、印地语-英语和西班牙语-英语对话中的趋同行为进行了跨语言分析，将已有的西班牙语-英语研究框架（Bhattacharya et al., 2024）复制并扩展到两个新的语言对，结果表明：虽然词汇层面的趋同在不同语言对之间具有普遍性，但在声学-韵律和语码转换风格方面的趋同则表现出特定语境的变化。基于这些发现，我们进一步探讨分类模型能否捕捉这些人类行为模式。通过特征重要性和消融分析，我们发现经典分类器和基于Transformer的分类器都能较好地检测趋同现象，但它们始终优先考虑的特征并非对人类趋同行为最关键的特征。我们的方法引入了一个以人类行为为基准的框架……

    arXiv:2607.25202v2 Announce Type: replace  Abstract: Conversational entrainment is well-studied in monolingual and written contexts, but remains underexplored in spoken code-switching (CSW). We present a cross-lingual analysis of entrainment in Mandarin-English, Hindi-English, and Spanish-English dialogue, replicating and extending an established Spanish-English framework (Bhattacharya et al., 2024) to two new language pairs, and show that, while lexical entrainment generalizes across language pairs, entrainment over acoustic-prosodic and CSW style aspects exhibits context-specific variation. We build on these findings by asking whether classification models capture these human behavioral patterns. Using feature importance and ablation analyses, we find that classical and Transformer-based classifiers detect entrainment reasonably well but consistently prioritize features other than those most salient to human entraining behavior. Our approach introduces a human-grounded framework for 
    
[^137]: JANUS：面向长程智能体安全的潜在风险预见

    JANUS: Foreseeing Latent Risk for Long-Horizon Agent Safety

    [https://arxiv.org/abs/2607.19913](https://arxiv.org/abs/2607.19913)

    Janus提出前瞻式智能体安全框架，通过预判任务与裁决任务的联合强化学习训练出守护模型Vanguard，能够在不安全操作执行前预见延迟风险，使平均防护提升15.9个百分点、良性任务完成率提升5.1个百分点。

    

    智能体安全正从内容审核转向在工具使用智能体行动之前预防操作性失误。我们提出Janus，一个面向长程智能体安全的前瞻性框架，它训练守护模型从部分轨迹中预判延迟显现的风险。Janus通过多智能体模拟合成多样化的智能体轨迹，并以两个耦合任务学习一个共享策略：一个是预测与安全相关未来的预判任务，另一个是综合观察到的前缀与预判的未来来判定安全性的裁决任务。这两个任务通过CoAA-RL进行联合优化，该方法根据预测对下游安全判断的效用来给予奖励。由此得到的守护模型Vanguard能够在不安全操作执行之前将其拦截。在四个智能体安全基准测试中，Vanguard相比基线守护模型将平均防护能力提升了15.9个百分点，同时将良性任务完成率提升了5.1个百分点。

    arXiv:2607.19913v2 Announce Type: replace  Abstract: Agent safety is moving from content moderation toward preventing operational failures before tool-using agents act. We propose Janus, a foresight-oriented framework for long-horizon agent safety that trains guards to anticipate delayed risks from partial trajectories. Janus synthesizes diverse agent trajectories via multi-agent simulation and learns a shared policy with two coupled tasks: an anticipation task that forecasts safety-relevant futures and an adjudication task that decides safety from both the observed prefix and anticipated future. The two tasks are jointly optimized with CoAA-RL, which rewards forecasts by their utility for downstream safety judgment. The resulting guard model, Vanguard, blocks unsafe actions before execution. Across four agent-safety benchmarks, Vanguard improves average protection by 15.9 percentage points over baseline guards while increasing benign task completion by 5.1 percentage points.
    
[^138]: 扩展格厄兹语系非洲语言词汇：阿姆哈拉语与提格里尼亚语的比较研究

    Expanding the Lexicon of Ge'ez Based African Languages: A Comparative Study of Amharic and Tigrinya

    [https://arxiv.org/abs/2607.15209](https://arxiv.org/abs/2607.15209)

    该论文提出VEXMLM，通过扩展词汇和两阶段训练，显著提升了XLM-R在阿姆哈拉语和提格里尼亚语等低资源格厄兹文字语言上的性能，尤其在词汇外实体识别方面。

    

    arXiv:2607.15209v2 公告类型：替换 摘要：多语言预训练语言模型（如XLM-R）在主要语言上表现良好，但在低资源格厄兹文字语言上却面临困难，这主要是由于以拉丁文字为中心的标记器导致的高词汇外（OOV）率和过度的子词碎片化。我们引入了VEXMLM，一种针对阿姆哈拉语和提格里尼亚语的XLM-R词汇扩展变体。我们在精选的单语语料库上训练语言特定的SentencePiece标记器，将XLM-R的词汇扩展30k个格厄兹文字子词，并通过子词平均初始化其嵌入。VEXMLM采用两阶段训练：（1）在精选语料库上进行持续掩码语言建模，（2）在问答、命名实体识别和情感分析上进行监督微调。VEXMLM在所有评估任务上显著优于XLM-R和Glot500，特别是在词汇外实体识别上表现突出。关键的是，i

    arXiv:2607.15209v2 Announce Type: replace  Abstract: Multilingual pre-trained language models such as XLM-R perform well for major languages but struggle with low-resource Ge'ez-script languages, largely due to high out-of-vocabulary (OOV) rates and excessive subword fragmentation from Latin-script-centric tokenizers. We introduce VEXMLM, a vocabulary-extended variant of XLM-R targeting Amharic and Tigrinya. We train language-specific SentencePiece tokenizers on curated monolingual corpora, extend XLM-R's vocabulary with 30k Ge'ez-script subwords, and initialize their embeddings via subword averaging. VEXMLM undergoes two-stage training: (1) continued masked language modeling on the curated corpora and (2) supervised fine-tuning on question answering, named entity recognition, and sentiment analysis. VEXMLM substantially outperforms XLM-R and Glot500 across all evaluated tasks on Amharic and Tigrinya, with particularly strong gains on out-of-vocabulary entity recognition. Critically, i
    
[^139]: 我如何知道接下来该说什么？巴伦霍尔茨的自生成理论对哈里斯整合主义的丰富

    How Do I Know What to Say Next? Barenholtz's Autogenerative Theory as an Enrichment of Harrisean Integrationism

    [https://arxiv.org/abs/2607.07891](https://arxiv.org/abs/2607.07891)

    本文提出巴伦霍尔茨基于大语言模型行为发展出的语言自生成理论，能够填补哈里斯整合主义语言学在符号未来开放性机制、语言与非语言符号活动连续性以及历史整合档案结构属性方面的解释空白，从而在不违背其核心主张的前提下丰富整合主义理论。

    

    罗伊·哈里斯的整合主义语言学对深植于语言计算方法核心的指称主义传统提出了有力的批判，认为语言并非一种映射到预先给定世界的代码，而是一种面向未来共同行动的、情境化的双向活动。然而，整合主义存在某些解释上的空白：它未能充分阐明符号维持未来开放性的结构机制，对语言与非语言符号活动之间的连续性理论化不足，并且对过去整合活动所积累档案的结构属性缺乏详细论述。本文认为，伊兰·巴伦霍尔茨针对大语言模型（LLM）行为而发展的语言自生成理论，恰好能够填补这些空白，在不损害整合主义任何核心主张的前提下丰富其理论内涵。

    arXiv:2607.07891v2 Announce Type: replace-cross  Abstract: Roy Harris's Integrationist linguistics offers a compelling critique of the referentialist tradition embedded deep at the heart of computational approaches to language, arguing that language is not a code that maps onto a pre-given world but a situated, bipartite activity oriented toward prospective joint action. Yet Integrationism leaves certain explanatory gaps: it does not fully account for the structural mechanism by which signs sustain prospective openness, it undertheorises the continuity between linguistic and non-linguistic semiotic activity, and it offers no detailed account of the structural properties of the accumulated archive of past integrations. This paper argues that Elan Barenholtz's autogenerative theory of language, developed in response to the behaviour of Large Language Models (LLMs), can fill precisely these gaps, enriching Integrationism without undermining any of its core commitments. Specifically, the a
    
[^140]: GaP：一种用于变分自动化任务的“图即策略”多智能体自学习框架

    GaP: A Graph-as-Policy Multi-Agent Self-Learning Harness For Variational Automation Tasks

    [https://arxiv.org/abs/2607.05369](https://arxiv.org/abs/2607.05369)

    本文提出了Graph-as-Policy（GaP），一个多智能体自学习编码框架，它从模块化开源机器人技能库生成包含感知、规划和控制节点的有向计算图，并利用内部仿真环境进行排练，从而提升物体几何与姿态变化较大的变分自动化任务的可靠性。

    

    要让机器人在商业和工业应用中可靠地工作，智能体编码系统的最新进展有潜力将可解释的机器人编程与无模型策略的开放世界适应能力相结合。我们专注于“变分自动化”（VA），这是一类在物体几何形状和姿态上的变化比固定自动化更大的任务。无模型策略往往难以缩小变分自动化任务的可靠性差距，而这些任务在商业和工业应用中必须持久且可靠地执行。受先前任务与运动规划（TAMP）和机器人操作系统（ROS）方面工作的启发，我们提出了“图即策略”，这是一个多智能体编码框架，能够从模块化开源机器人技能库（MORSL）中生成包含感知、规划和控制节点的有向计算图。GaP生成一个内部仿真环境，以便用不同的图并行地排练任务实例。

    arXiv:2607.05369v2 Announce Type: replace-cross  Abstract: For robots to work reliably in commercial and industrial applications, recent advances in agentic coding systems have potential to combine interpretable robot programming with the open-world adaptability of model-free policies. We focus on "Variational Automation" (VA), a class of tasks that have larger variations in object geometry and pose than fixed automation. Model-free policies often struggle to close the reliability gap for VA tasks, which must be executed persistently and reliably in commercial and industrial applications. Motivated by prior work on Task and Motion Planning (TAMP) and the Robot Operating System (ROS), we introduce Graph-as-Policy (GaP), a multi-agent coding harness that generates directed computation graphs with perception, planning, and control nodes from a Modular Open Robot Skill Library (MORSL). GaP generates an internal simulation environment to rehearse task instances with different graphs in para
    
[^141]: 聚类、路由、升级：面向成本感知大语言模型服务的级联框架

    Cluster, Route, Escalate: Cascaded Framework for Cost-Aware LLM Serving

    [https://arxiv.org/abs/2606.27457](https://arxiv.org/abs/2606.27457)

    提出了CRE-Router两阶段级联框架，在延迟预算约束下先通过聚类将查询路由到成本最优的模型，再通过质量估计将不合格答案升级到更强的模型，从而在大语言模型服务中兼顾准确率与计算成本。

    

    在本地部署开放权重语言模型需要在准确性与计算成本之间进行权衡。托管不同规模的模型可以让简单查询由小型高效模型处理，而困难查询则由大型模型处理。我们提出了CRE-Router（聚类、路由、升级），这是一种两阶段方法，用于决定由哪个模型来回答每个查询。它首先对查询进行聚类，并将每个聚类路由到在错误与成本之间实现最佳权衡的模型，同时遵守每输出标记时间（TPOT）或每请求端到端延迟（E2EL）的预算约束。随后，系统要么接受该答案，要么在质量估计器拒绝该答案时，将查询升级到强大的模型。路由在生成之前做出决策，因此困难查询可以直接发送给强大的模型；升级则在生成之后做出决策，因此可以捕获路由无法预测的错误。除任务正确性标签外，它不需要任何其他标注。我们在TeleMath、TeleQnA和AIME数据集上对CRE-Router进行了评估。

    arXiv:2606.27457v2 Announce Type: replace-cross  Abstract: Serving open-weight language models on premises forces a trade-off between accuracy and computational cost. Hosting models of different sizes lets easy queries go to a small, efficient model and hard ones to a large model. We present CRE-Router (Cluster, Route, Escalate), a two-stage method that decides which of these models answers each query. It clusters queries and routes each cluster to the model that best trades error against cost, under a budget on the time per output token (TPOT) or the end-to-end latency per request (E2EL). It then either accepts the answer or, when a quality estimator rejects it, escalates the query to the strong model. Routing decides before generation, so a hard query goes straight to the strong model; escalation decides after, so it can catch the errors routing cannot predict. It needs no annotation beyond task-correctness labels. We evaluate CRE-Router on TeleMath, TeleQnA and AIME, with pools of t
    
[^142]: 针对大语言模型的蜜源策略：重新思考针对AI攻击者的网络欺骗

    Honeyquest for LLMs: Rethinking Cyber Deception for AI Attackers

    [https://arxiv.org/abs/2606.21037](https://arxiv.org/abs/2606.21037)

    本文通过大规模评估21个大语言模型，发现AI攻击者比人类更易落入网络欺骗陷阱，提出了一种新的自动化评估框架，并指出大语言模型构成独特的攻击者类别。

    

    网络欺骗的经验基础依赖于以人为中心的假设，但自主、AI驱动的攻击者的迅速崛起，挑战了这一基础是否适用于AI代理。为此，我们引入了一个自动化评估框架，改编自Honeyquest工具，以大规模评估大语言模型攻击者的判断力。我们的21个大语言模型队列涵盖10个提供商、多样化的架构和专业化方向、开源和闭源权重模型，以及从8B到超过1T的参数规模。我们评估了这一大语言模型队列（产生10,962个响应）在相同的174个侦察查询集上，与47名参与者的人类基线进行对比。我们的实证评估揭示了三个关键发现，将大语言模型确立为一个独特的攻击者类别：（1）我们队列中的每个模型落入欺骗陷阱的比率显著高于人类攻击者；（2）在人类中观察到的防御性注意力分散效应在...

    arXiv:2606.21037v2 Announce Type: replace-cross  Abstract: The empirical foundation of cyber deception relies on human-centered hypotheses, but the rapid emergence of autonomous, AI-enabled attackers challenges whether this foundation transfers to AI agents. To address this, we introduce an automated evaluation framework adapted from the Honeyquest instrument to assess LLM attacker judgment at scale. Our 21-LLM cohort spanned 10 providers, diverse architectures and specializations, open- and closed-weight models, and parameter scales from 8B to over 1T. We evaluated the performance of this LLM cohort (yielding 10,962 responses) against the 47-participant human baseline across an identical set of 174 reconnaissance queries. Our empirical evaluation reveals three key findings that establish LLMs as a distinct attacker class: (1) every model in our cohort falls for deceptive traps at a significantly higher rate than human attackers; (2) the defensive attention-diversion effect observed in
    
[^143]: AlphaMemo：面向自进化Alpha挖掘智能体的结构化搜索过程记忆

    AlphaMemo: Structured Search-Process Memory for Self-Evolving Alpha Mining Agents

    [https://arxiv.org/abs/2606.20625](https://arxiv.org/abs/2606.20625)

    AlphaMemo通过从AST差异中提取编辑模式、置信度门控残差记忆和非对称否决控制构建结构化搜索过程记忆，实现了自进化的Alpha挖掘智能体，在中证500和标普500上显著提升了样本外表现和固定预算下的因子发现效率。

    

    LLM智能体通过结合金融先验知识、符号推理、可执行因子生成和反馈驱动的迭代改进，在Alpha挖掘领域展现出广阔前景。然而，它们面临着组合爆炸式的搜索空间、嘈杂且非平稳的反馈、冗余的因子发现，以及简单复用历史成功经验所带来的过拟合风险。为了应对这些挑战，我们提出了AlphaMemo，一个具有结构化搜索过程记忆的自进化Alpha挖掘智能体。与仅仅记忆最终因子或完整轨迹不同，AlphaMemo记录的是可复用的证据，即哪些编辑模式在特定的父因子上下文中有效或失败。它从抽象语法树（AST）差异中提取编辑模式，在搜索账本先验之上应用置信度门控的残差记忆，并采用非对称否决控制机制来抑制高置信度的失败模式。在中证500和标普500指数上的实验表明，该方法的样本外表现和固定预算下的因子发现效率均得到提升。

    arXiv:2606.20625v2 Announce Type: replace  Abstract: LLM agents are promising for alpha mining via combining financial priors, symbolic reasoning, executable factor generation, and feedback-driven refinement. Yet, they face a combinatorial search space, noisy non-stationary feedback, redundant discoveries, and overfitting risks from naively reusing past successes. To address these challenges, we propose AlphaMemo, a self-evolving alpha mining agent with Structured Search-Process Memory. Rather than memorizing only final factors or full trajectories, AlphaMemo records reusable evidence about which edit motifs work or fail under specific parent-factor contexts. It extracts motifs from Abstract Syntax Tree (AST) differences, applies confidence-gated residual memory on top of a search-ledger prior, and uses asymmetric veto control to suppress high-confidence failure patterns. Experiments on CSI 500 and S&P 500 show improved out-of-sample performance and fixed-budget discovery efficiency, w
    
[^144]: 大型音频语言模型的连续音频思考

    Continuous Audio Thinking for Large Audio Language Models

    [https://arxiv.org/abs/2606.18273](https://arxiv.org/abs/2606.18273)

    提出连续音频思考（CoAT）框架，通过音频专家蒸馏为大型音频语言模型构建连续潜在思考空间，使其在生成响应前能够组织并利用语音细节、韵律、声音事件、情感和音高等丰富声学信息。

    

    大型音频语言模型（LALMs）在从语音转录到音乐分析等多种音频理解任务上展现出令人印象深刻的能力。然而，由于LALMs通常被训练以生成与文本对齐的响应，其隐藏状态逐渐被塑造为服务于文本生成而非保留声学信息。因此，音频所承载的丰富声学内容，如语音细节、韵律、声音事件、情感和音高，在这一过程中逐渐丢失，难以在响应中加以利用。我们提出了连续音频思考，这是一个为音频语言模型配备连续潜在工作空间的框架，用于在生成响应之前组织声学信息，并通过音频专家的蒸馏进行支撑。在思考空间内，模型可以在生成响应时利用专家蒸馏所提供的丰富声学信息。

    arXiv:2606.18273v2 Announce Type: replace-cross  Abstract: Large audio language models (LALMs) have shown impressive capabilities on diverse audio understanding tasks, ranging from speech transcription to music analysis. However, because LALMs are typically trained to produce text-aligned responses, their hidden states are progressively shaped for text generation rather than for preserving acoustic information. As a result, the diverse acoustic content that audio carries, such as phonetic detail, prosody, sound events, affect, and pitch, is lost along the way and difficult to leverage in the response. We introduce Continuous Audio Thinking (CoAT), a framework that equips audio language models with a continuous latent workspace for organizing acoustic information prior to response generation, grounded by distillation from audio experts. Within the thinking space, the model can utilize the rich acoustic information provided by expert distillation when generating its response. Furthermore
    
[^145]: 先刻画再蒸馏：大输出空间中的机制化推理

    Characterize Then Distill: Mechanistic Reasoning in Large Output Spaces

    [https://arxiv.org/abs/2606.06840](https://arxiv.org/abs/2606.06840)

    本文通过将多标签决策建模为token级事件，并结合归因、消融与植入等因果分析手段，刻画了推理型大模型在海量候选标签空间中进行选择的内部注意力头机制，并证明该机制可以被蒸馏。

    

    经过推理训练的语言模型能够以零样本方式执行多标签任务，即需要从数千到数十万个候选标签中选出一小部分相关标签。我们探讨它们在机制层面是如何完成这一任务的，以及该机制能否被蒸馏。我们通过将每个决策视为一个由模型自身决策边际（decision margin）评分的token级事件，使这一问题变得可测量：包括选定标签空间粗略区域的token、在该区域内选定具体标签的token，以及输出偏离推理过程中早先提到的接近替代方案（即“近似失误”）的token。通过归因分析、精确平均消融、向其他示例上下文中植入（knock-in）实验，以及对通用注意力头进行折扣的零假设校准，这些方法赋予了单个注意力头因果地位。在医院出院小结的临床编码任务（MIMIC-IV）上，在上下文中包含全部5,651个候选诊断代码的情况下，一个小型的、全局的、具有阶段结构的……（摘要原文在此处截断）

    arXiv:2606.06840v2 Announce Type: replace-cross  Abstract: Reasoning-trained language models can perform, zero-shot, multi-label tasks that require selecting a small set of relevant labels from a universe of thousands to hundreds of thousands of candidates. We ask how they do it mechanistically, and whether the mechanism can be distilled. We make the question measurable by treating each decision as a token-level event scored by the model's own decision margin: the token that picks a coarse region of the label space, the tokens that pick a label within it, and the token where the output departs from a close alternative (a near-miss) named earlier in the reasoning. Attribution, exact mean-ablation, knock-in into another example's context, and a null calibration that discounts generic heads then give individual attention heads causal standing. On clinical coding of hospital discharge summaries (MIMIC-IV), with all 5,651 candidate diagnosis codes in context, a small, global, phase-structur
    
[^146]: SkillPyramid：面向自进化智能体的层次化技能巩固框架

    SkillPyramid: A Hierarchical Skill Consolidation Framework for Self-Evolving Agents

    [https://arxiv.org/abs/2606.03692](https://arxiv.org/abs/2606.03692)

    SkillPyramid提出了一种基于层次化技能拓扑的技能巩固框架，通过自进化机制使智能体在任务执行中组合、验证并积累新技能，从而实现技能经验的复用与更广泛的任务泛化。

    

    最近的AI智能体能够灵活地调用技能来解决复杂任务，但其长期改进从根本上受限于缺乏系统化的技能构建、积累与迁移机制。特别是，在没有统一的技能巩固框架的情况下，智能体往往会在不同任务中冗余地构建相似的能力，无法有效地将经验转化为可复用的资产，并且难以将特定任务的技能泛化到新场景中。为了解决这一局限，我们提出了SkillPyramid，一个能够复用现有技能经验以实现更广泛任务泛化的技能巩固框架。SkillPyramid在层次化的技能拓扑结构上运行，并进一步引入了一种自进化机制，使智能体能够在任务执行过程中组合、验证并纳入新技能。在ALFWorld、WebShop和ScienceWorld三个基准上、跨越四个骨干模型的实验表明，SkillPyramid显著……（原文摘要至此处被截断）

    arXiv:2606.03692v2 Announce Type: replace  Abstract: Recent AI agents can flexibly invoke skills to solve complex tasks, but their long-term improvement is fundamentally constrained by a lack of systematic skill construction, accumulation, and transfer. In particular, without a unified framework for skill consolidation, agents tend to redundantly construct similar capabilities across different tasks, are unable to effectively transform experience into reusable assets, and struggle to generalize task-specific skills to novel scenarios. To address this limitation, we propose SkillPyramid, a skill consolidation framework that reuses existing skill experience for broader task generalization. Operating on a hierarchical skill topology, SkillPyramid further introduces a self-evolution mechanism that enables agents to compose, validate, and incorporate new skills during task execution. Experiments on ALFWorld, WebShop, and ScienceWorld across four backbone models show that SkillPyramid substa
    
[^147]: RCEM：分布偏移下鲁棒的对话搜索嵌入器

    RCEM: Robust Conversational Search EMbedder in Distributional Shift

    [https://arxiv.org/abs/2606.01697](https://arxiv.org/abs/2606.01697)

    RCEM通过将对话与LLM重写的查询对齐而非直接匹配长段落，在保留原始嵌入空间的前提下，实现了减少过拟合、无需相关性标注且无需重建索引的鲁棒对话搜索检索。

    

    我们提出了RCEM，一个鲁棒的对话搜索嵌入器，它额外具备了LLM的查询重写能力，同时不损失基础模型的泛化能力。与以往学习从对话直接到段落匹配的对话式密集检索方法不同，RCEM将带有特殊标记前缀的对话与LLM重写后的查询进行对齐，同时保留原始的嵌入空间。保持不变的嵌入空间可以自动将重写后的查询映射到相关段落。因此，RCEM (1) 通过将对齐任务从长段落简化为较短的重写查询来减少过拟合，(2) 消除了训练时对“对话-段落”相关性标注的依赖，(3) 保持了原始嵌入空间，使得对话查询可以直接在由原始嵌入器构建的索引上进行检索，无需重建索引。大量实验表明，RCEM持续优于以往的方法。

    arXiv:2606.01697v3 Announce Type: replace  Abstract: We propose RCEM, a Robust Conversational search EMbedder that is additionally equipped with LLM's query reformulation capability without losing base model's generalization. Unlike prior conversational dense retrieval approaches that learn direct conversation-to-passage matching, RCEM aligns conversations, prepended by special token, to LLM-rewritten queries, while preserving the original embedding space. The unchanged embedding space automatically maps the rewritten-query to the relevant passages. As a result, RCEM (1) reduces overfitting by simplifying the alignment task from long passages to shorter rewritten queries, (2) eliminates the need for conversation-to-passage relevance labels for training, and (3) maintains its original embedding space that allows conversational queries against indexes built by original embedder without rebuilding them. Extensive experiments show that RCEM consistently outperforms prior approaches, achiev
    
[^148]: 探测视觉-语言模型中的碰撞定位能力以实现安全的人机协作

    Probing Collision Grounding in Vision-Language Models for Safe Human-Robot Collaboration

    [https://arxiv.org/abs/2605.31196](https://arxiv.org/abs/2605.31196)

    该论文提出了 TouchSafeBench——首个基于物理仿真构建、用于评估视觉-语言模型“碰撞定位”能力的基准，包含近三千个人机共处场景，支持安全状态分类与即将碰撞预警两项实际部署任务。

    

    安全的人机协作不仅仅需要视觉描述：监控系统必须判断机器人本体是否与场景保持安全分离、是否已经与场景或人员发生碰撞、或即将发生碰撞。我们将这种能力称为“碰撞定位”：即将视觉观测与机器人本体几何、相机视角、场景布局、人类接近程度以及时间上的运动信息绑定起来，以推断当前和即将发生的接触。我们提出了 TouchSafeBench，一个基于物理的基准测试，用于评估视觉-语言模型（VLM）中的碰撞定位能力。TouchSafeBench 构建于 Habitat 3.0 仿真环境中，包含 2,940 个模拟室内人机共处场景，涵盖社交导航和社交重排任务，并提供同步的多视角 RGB-D 观测、俯视轨迹图、标定的相机元数据以及来自仿真器的接触标签。我们研究了两个面向实际部署的任务：对当前安全状态进行分类，以及对即将发生的碰撞发出警告。

    arXiv:2605.31196v2 Announce Type: replace-cross  Abstract: Safe human--robot collaboration requires more than visual description: a monitor must determine whether the robot body is safely separated, already colliding with the scene or a person, or about to collide. We call this capability collision grounding: binding visual observations to robot body geometry, camera viewpoint, scene layout, human proximity, and temporal motion in order to infer present and imminent contact. We introduce TouchSafeBench, a physics-grounded benchmark for evaluating collision grounding in vision-language models (VLMs). Built in Habitat~3.0, TouchSafeBench contains 2,940 simulated indoor co-presence episodes across social navigation and social rearrangement, with synchronized multi-view RGB-D observations, top-down trajectory maps, calibrated camera metadata, and simulator-derived contact labels. We study two deployment-facing tasks: classifying the current safety state and warning about imminent collision
    
[^149]: GRKV：面向长上下文大语言模型的无训练KV缓存压缩的全局回归方法

    GRKV: Global Regression for Training-Free KV Cache Compression in Long-Context LLMs

    [https://arxiv.org/abs/2605.31105](https://arxiv.org/abs/2605.31105)

    提出了一种无需训练的KV缓存压缩方法GRKV，通过全局回归直接最小化压缩缓存与完整缓存注意力输出之间的差异，解决了基于区间保留策略中合并模式高度不平衡所导致的过度合并与信息损失问题。

    

    具有扩展上下文长度的大语言模型（LLM）依赖键值（KV）缓存来支持对先前标记的注意力计算。然而，维护KV缓存会带来巨大的内存开销，这促使了通过淘汰与合并来执行固定预算的KV缓存压缩方法。现代淘汰方法越来越多地采用基于区间的保留策略，因为保留连续区间在经验上是有效的，并且能更好地保持语义连贯性。然而，当与淘汰后合并结合使用时，基于区间的保留策略会将合并集中在少数区间边界载体标记上，产生高度不平衡的合并模式，从而加剧过度合并并增加信息损失。为解决这一不平衡问题，我们提出了GRKV（KV缓存的全局回归），这是一种无需训练的KV缓存合并方法，它直接最小化压缩缓存与完整缓存的注意力输出之间的差异。GRKV使用岭回归……

    arXiv:2605.31105v2 Announce Type: replace  Abstract: Large language models (LLMs) with extended context lengths rely on the key-value (KV) cache to support attention over prior tokens. However, maintaining the KV cache incurs substantial memory overhead, motivating KV-cache compression methods that enforce a fixed budget through eviction and merging. Modern eviction methods increasingly adopt span-based retention because preserving contiguous spans is empirically effective and better preserves semantic coherence. Yet, when combined with post-eviction merging, span-based retention concentrates merges onto a small set of span-boundary carrier tokens, producing a highly imbalanced merge pattern that exacerbates over-merging and increases information loss. To address this imbalance, we propose GRKV (Global Regression for KV Cache), a training-free KV-cache merging method that directly minimizes the discrepancy between compressed-cache and full-cache attention outputs. GRKV uses ridge-regre
    
[^150]: 高效基准测试本质上就是特征选择与多元回归

    Efficient Benchmarking Is Just Feature Selection and Multiple Regression

    [https://arxiv.org/abs/2605.25773](https://arxiv.org/abs/2605.25773)

    该论文将高效基准测试重新构建为带特征选择的多元回归问题，通过结合核岭回归与mRMR特征选择算法，在预测误差和排序相关性上显著改进了现有高效基准测试方法。

    

    arXiv:2605.25773v3 公告类型：replace-cross 摘要：高效基准测试技术旨在通过仅使用基准测试题目中的一个子集来预测完整基准测试分数，从而降低评估大语言模型（LLM）的计算成本。通过将这一问题重新构建为带特征选择的多元回归的一个实例，我们发现现有高效基准测试方法只需在预测阶段采用核岭回归（kernel ridge regression）即可得到大幅改进。此外，利用一种称为最小冗余最大相关性（mRMR）的信息论特征选择算法，我们可以通过选择对预测最有价值的题目子集来进一步提升这些方法的效果。除在数据极度匮乏的情形外，这些方法在涵盖二值与连续分数的多种基准测试上，均始终取得更小的预测误差（在MAE和RMSE两方面）以及预测分数与真实分数之间更高的排序相关性（在Spearman ρ和Kendall τ两方面）。

    arXiv:2605.25773v3 Announce Type: replace-cross  Abstract: Efficient benchmarking techniques aim to lower the computational cost of evaluating LLMs by predicting full benchmark scores using only a subset of a benchmark's questions. By reframing this problem as an instance of multiple regression with feature selection, we find that existing efficient benchmarking methods can be greatly improved by simply using kernel ridge regression at the prediction stage. Additionally, using an information-theoretic feature-selection algorithm called minimum redundancy maximum relevance (mRMR), we can further improve upon these methods by selecting question subsets that will be maximally useful for prediction. Except in very data-poor settings, these approaches consistently achieve smaller prediction errors (in both MAE and RMSE), and greater ranking correlation between predicted and true scores (in both Spearman $\rho$ and Kendall $\tau$) across a range of benchmarks using both binary and continuous
    
[^151]: 译者作为AI的隐形教师：版权、翻译记忆与语言数据的政治经济学

    Translators as Invisible Teachers of AI: Copyright, Translation Memory, and the Political Economy of Linguistic Data

    [https://arxiv.org/abs/2605.24842](https://arxiv.org/abs/2605.24842)

    译者的翻译劳动（以翻译记忆和平行语料库为载体）已成为AI发展的基础数据资本，但其创造性贡献在版权法下被抹去，本文从语言数据的政治经济学视角揭示了这一剥削机制。

    

    本文考察了译者的劳动如何被转化为人工智能（AI）时代的基础性数据资本。翻译记忆（TM）与平行语料库保存了源文本与目标文本之间的一一对应关系，因此构成了对机器翻译而言极为宝贵的有监督训练数据。统计机器翻译（SMT）、神经机器翻译（NMT）、Transformer架构以及多语言大语言模型（LLM）的发展，都无法与此类翻译数据的积累相分离。然而，译者的译文却作为合同项下的交付成果被购买，作为技术对象被切分，并在版权法下被作为“信息分析”数据处理——从而丧失了其对创作这些译文的译者在道德、创造性与经济上的归属。本文提出了两个概念来刻画这一过程。第一个是

    arXiv:2605.24842v2 Announce Type: replace  Abstract: This paper examines how the labour of translators has been transformed into foundational data capital for the age of artificial intelligence (AI). Translation memories (TM) and parallel corpora preserve a one-to-one correspondence between source and target text and therefore constitute extraordinarily valuable supervised training data for machine translation. The development of statistical machine translation (SMT), neural machine translation (NMT), the Transformer architecture, and multilingual large language models (LLMs) cannot be disentangled from the accumulation of such translation data. And yet, translators' renditions have been bought as deliverables under contract, segmented as technical objects, and processed as "information analysis" data under copyright law -- losing their moral, creative, and economic attribution to the translators who produced them. The paper develops two concepts to capture this process. The first is a
    
[^152]: NodeSyn：面向AI评估的社会对齐合成数据

    NodeSynth: Socially Aligned Synthetic Data for AI Evaluation

    [https://arxiv.org/abs/2605.14381](https://arxiv.org/abs/2605.14381)

    NodeSyn通过将微调的分类体系生成器与可信文献来源和敏感属性相结合，生成反映真实社会背景的敏感领域评估数据，其诱发的模型失败率达到通用合成基线的3.1倍、真实人类查询的1.6倍，显著优于现有合成数据方法。

    

    生成式人工智能的最新进展为模型评估所需的大规模合成数据生成提供了便利。然而，若缺乏针对性的方法，这些数据集往往缺乏敏感领域所需的社会技术层面的细微差别。我们提出了NodeSyn，这是一种将经过微调的分类体系生成器（TaG）与可信的已发表文献来源以及提取的敏感属性相结合的方法，用于生成反映真实社会背景和细微危害的查询。在使用经人工验证的自动评估器，于医疗建议和自我伤害两个领域对十个开源及专有大语言模型进行评估时，NodeSyn在10个模型中的8个上于每个领域均排名第一。NodeSyn数据集引发的平均失败率是通用合成基线的3.1倍，是结构化合成基准的2.4倍，是真实人类撰写查询的1.6倍。消融研究证实，在控制结构混杂因素的情况下，我们细粒度的第3级分类体系…

    arXiv:2605.14381v3 Announce Type: replace-cross  Abstract: Recent advancements in generative AI facilitate large-scale synthetic data generation for model evaluation. However, without targeted approaches, these datasets often lack the sociotechnical nuance required for sensitive domains. We introduce NodeSyn, a methodology that pairs a fine-tuned taxonomy generator (TaG) with credible published sources and extracted sensitive attributes to generate queries reflecting authentic societal contexts and nuanced harms. Evaluated across ten open-weight and proprietary LLMs in Medical Advice and Self-Harm using human-validated autoraters, NodeSyn ranked first on 8 of 10 models in each domain. NodeSyn datasets induced average failure rates 3.1 times higher than generic synthetic baselines, 2.4 times higher than structured synthetic benchmarks, and 1.6 times higher than authentic human-authored queries. Ablation studies confirm that, controlling for structural confounds, our granular Level 3 tax
    
[^153]: 通过结构化元认知在通用智能体中实现深度推理

    Deep Reasoning in General Purpose Agents via Structured Meta-Cognition

    [https://arxiv.org/abs/2605.11388](https://arxiv.org/abs/2605.11388)

    提出“深度推理”方法，利用形式化语言将元推理表示为可执行的分解，在推理时动态构建任务特定的脚手架，使LLM智能体能像人类一样灵活地在联想推理、形式化计算和递归求解等不同推理模式间切换。

    

    人类通过在不同推理模式之间灵活切换来直观地解决复杂问题：他们进行规划、执行、修改中间目标、通过联想判断消解歧义，并对明确定义的子问题应用形式化程序。当前的LLM智能体缺乏这种灵活性，因为它们的脚手架（scaffold）预先硬编码了此类推理决策。当其规定的结构与任务相匹配时，这些脚手架是有效的，但当解决任务需要调整推理结构本身时，它们就会变得脆弱。我们提出了深度推理（Deep Reasoning）——一种通过结构化元推理在推理时构建任务特定脚手架的方法。深度推理使用一种形式化语言，将元推理表示为对联想推理、形式化计算和递归子问题求解的可执行分解，使分解原则能够被编码为上下文示例，从而指导测试时的脚手架构建。

    arXiv:2605.11388v2 Announce Type: replace  Abstract: Humans intuitively solve complex problems by flexibly shifting among reasoning modes: they plan, execute, revise intermediate goals, resolve ambiguity through associative judgment, and apply formal procedures to well-specified subproblems. Current LLM agents lack this flexibility, as their scaffolds hard-code such reasoning decisions in advance. These scaffolds are effective when their prescribed structure matches the task, but brittle when solving the task requires adapting the structure of reasoning itself. We introduce Deep Reasoning -- an inference-time approach for constructing task-specific scaffolds through structured meta-reasoning. Deep Reasoning uses a formal language that represents meta-reasoning as executable decompositions over associative inference, formal computation, and recursive subproblem solving, enabling decomposition principles to be encoded as in-context examples that guide test-time scaffold construction. We 
    
[^154]: MedAction：迈向主动式多轮临床诊断大语言模型

    MedAction: Towards Active Multi-turn Clinical Diagnostic LLMs

    [https://arxiv.org/abs/2605.07305](https://arxiv.org/abs/2605.07305)

    提出MedAction树状结构蒸馏流水线，通过大语言模型与环境交互合成高质量多轮诊断轨迹，使模型学会在部分证据下主动开具检查、解读结果并更新鉴别诊断，克服了静态单轮诊断评估的局限。

    

    现有的大多数大语言模型诊断评估都是在静态、单轮的设置下进行的，即预先提供完整的患者信息，这是对真实临床实践的过度简化。我们研究主动式诊断：即从初始观察开始、开具检查、解读结果，并在多轮过程中不断更新鉴别诊断的真实临床流程。通过系统性分析，我们识别出当前大语言模型中三种反复出现的失败模式：缺乏依据的检查开具、不可靠的诊断更新，以及多轮对话中连贯性的退化。这些失败共同揭示了一个核心缺陷：现有的医学训练数据教会模型基于完整信息进行推理，却没有教会模型在不断演变的部分证据下采取行动。为了弥补这一差距，我们提出了 MedAction，这是一个树状结构的蒸馏流水线，通过大语言模型与环境的交互，合成多样化且高质量的多轮诊断轨迹。我们提出了两种……

    arXiv:2605.07305v2 Announce Type: replace-cross  Abstract: Most existing LLM diagnoses are evaluated on static, single-turn settings where complete patient information is provided upfront, an oversimplification of real clinical practice. We study active diagnosis: the real-life clinical process of starting from initial observation, ordering tests, interpreting results, and updating a differential diagnosis across multiple turns. Through systematic analysis, we identify three recurring failure modes in current LLMs: ungrounded test ordering, unreliable diagnostic update, and degraded multi-turn coherence. Together, these failures reveal a core deficit: existing medical training data teaches models to reason from complete information but not to act under evolving, partial evidence. To address this gap, we introduce MedAction, a tree-structured distillation pipeline that synthesizes diverse and high-quality multi-turn diagnostic trajectories via LLM-environment interaction. We propose two
    
[^155]: 更好的奖励模型并不总是偏见更少：超越标准奖励基准的社会对齐研究

    Better Reward Models Are Not Always Less Biased: Social Alignment Beyond Standard Reward Benchmarks

    [https://arxiv.org/abs/2605.05003](https://arxiv.org/abs/2605.05003)

    该论文将奖励模型评估扩展到偏见、安全、道德和伦理推理四个社会关键领域，发现标准基准性能与社会对齐行为之间仅有不一致的关联，且更强的奖励模型在应对多元社会期望时并不一定表现更好，揭示了偏见缓解与情境忠实性之间的权衡。

    

    奖励模型是大语言模型对齐的核心，然而当前的基准测试几乎无法揭示这些模型如何表征多样的、有时相互冲突的社会偏好。我们将奖励模型评估扩展到四个具有重大社会影响的领域：偏见、安全、道德和伦理推理。我们的分析表明，在基准测试中的优异表现可能掩盖社会行为上的重要差异。模型可能在避免某些刻板印象的同时强化了其他刻板印象，而总体偏见得分可能隐藏这些相互对立的倾向。我们还发现，更强的刻板印象规避可能以牺牲情境忠实性为代价，揭示了社会偏见缓解与情境敏感性之间存在权衡关系。在广泛的奖励模型集合中，标准基准性能与社会对齐行为之间仅存在不一致的关联。这些结果表明，更强的奖励模型在应对多元社会期望方面并不一定表现更好。

    arXiv:2605.05003v2 Announce Type: replace-cross  Abstract: Reward models are central to LLM alignment, yet current benchmarks reveal little about how they represent diverse and sometimes competing social preferences. We extend reward model evaluation to four socially consequential domains: bias, safety, morality, and ethical reasoning. Our analysis shows that strong benchmark performance can obscure important differences in social behavior. Models may avoid some stereotypes while reinforcing others, and aggregate bias scores can hide these opposing tendencies. We also find that stronger stereotype avoidance can come at the expense of contextual faithfulness, revealing a trade-off between social bias mitigation and sensitivity to context. Across a broad set of reward models, standard benchmark performance is only inconsistently related to socially aligned behavior. These results show that stronger reward models are not uniformly better at navigating pluralistic social expectations and m
    
[^156]: daVinci-Env：大规模开放式软件工程环境合成

    daVinci-Env: Open SWE Environment Synthesis at Scale

    [https://arxiv.org/abs/2603.13023](https://arxiv.org/abs/2603.13023)

    提出了 OpenSWE——目前最大的完全开源透明的软件工程智能体训练框架，包含 45,320 个覆盖 1.28 万仓库的可执行 Docker 环境，并借助 64 节点集群上的多智能体合成流水线自动化构建。

    

    训练能力强大的软件工程（SWE）智能体需要大规模、可执行且可验证的环境，以为迭代式代码编辑、测试执行和方案改进提供动态反馈回路。然而，现有的开源数据集在规模和仓库多样性方面仍然有限，而工业界的解决方案则不透明且基础设施未公开，为大多数学术研究团队设置了难以逾越的障碍。我们提出了 OpenSWE，这是目前最大的完全透明的 Python 软件工程智能体训练框架，包含 45,320 个可执行的 Docker 环境，覆盖超过 12.8k 个代码仓库，所有 Dockerfile、评估脚本和基础设施均完全开源以保证可复现性。OpenSWE 通过部署在 64 节点分布式集群上的多智能体合成流水线构建，自动化完成仓库探索、Dockerfile 构建、评估脚本生成和迭代测试分析。

    arXiv:2603.13023v3 Announce Type: replace-cross  Abstract: Training capable software engineering (SWE) agents demands large-scale, executable, and verifiable environments that provide dynamic feedback loops for iterative code editing, test execution, and solution refinement. However, existing open-source datasets remain limited in scale and repository diversity, while industrial solutions are opaque with unreleased infrastructure, creating a prohibitive barrier for most academic research groups. We present OpenSWE, the largest fully transparent framework for SWE agent training in Python, comprising 45,320 executable Docker environments spanning over 12.8k repositories, with all Dockerfiles, evaluation scripts, and infrastructure fully open-sourced for reproducibility. OpenSWE is built through a multi-agent synthesis pipeline deployed across a 64-node distributed cluster, automating repository exploration, Dockerfile construction, evaluation script generation, and iterative test analysi
    
[^157]: MetaKE：面向更优精度-可编辑性权衡的知识编辑元学习方法

    MetaKE: Meta-Learning for Knowledge Editing Toward a Better Accuracy-Editability Trade-off

    [https://arxiv.org/abs/2603.12677](https://arxiv.org/abs/2603.12677)

    提出MetaKE框架，通过元学习将知识编辑中上游表示优化与下游参数优化统一为双层优化问题，解决了两阶段优化割裂导致的编辑不充分或可编辑性下降问题，实现了精度与可编辑性之间更优的权衡。

    

    现有基于“定位后编辑”的知识编辑方法通常将编辑分解为两个阶段：上游的目标表示优化和下游的受约束参数优化。两个阶段之间的优化是相互割裂的：上游在未能观察下游计划残差实际实现情况的前提下应用统一正则化，阻碍了精度与可编辑性之间的精细权衡。由于这种实现是请求特定的，并依赖于下游约束，统一正则化可能会过度收缩高关联请求，导致编辑不充分；同时可能对低关联请求正则化不足，产生过大的计划残差，从而降低下游的可编辑性。为弥合这一割裂，我们提出了MetaKE（面向知识编辑的元学习），一个将上游与下游阶段统一为双层优化问题的新框架。内层优化参数更新……

    arXiv:2603.12677v4 Announce Type: replace-cross  Abstract: Existing locate-then-edit Knowledge Editing (KE) methods typically decompose editing into two stages: upstream target representation optimization and downstream constrained parameter optimization. The optimization across the two stages is disconnected: upstream applies uniform regularization without observing downstream realization of the planned residual, hindering a refined accuracy-editability trade-off. Since this realization is request-specific and depends on downstream constraints, uniform regularization can over-shrink high-association requests, causing insufficient editing, while it can under-regularize low-association requests, producing over-large planned residuals that reduce downstream editability. To bridge this disconnect, we propose MetaKE (Meta-learning for Knowledge Editing), a new framework that unifies upstream and downstream stages into a bi-level optimization problem. The inner level optimizes parameter upd
    
[^158]: MoE-SpAc：异构边缘场景下基于投机激活效用的高效MoE推理框架

    MoE-SpAc: Efficient MoE Inference Based on Speculative Activation Utility in Heterogeneous Edge Scenarios

    [https://arxiv.org/abs/2603.09983](https://arxiv.org/abs/2603.09983)

    提出MoE-SpAc框架，将投机解码创新性地重新用作内存管理的前瞻传感器，通过投机效用估计器、异构工作负载均衡器和异步执行引擎，在统一的效用空间中协调专家预取与逐出，使异构边缘设备上的MoE推理TPS较SOTA基线提升42%。

    

    混合专家模型能够实现可扩展的性能，但在边缘设备上面临严重的内存限制。由于自回归专家激活具有动态性和低信息量特性，现有的卸载策略难以应对I/O瓶颈。本文提出将投机解码重新定位，不仅作为计算加速器，更作为内存管理的信息前瞻传感器，这一提议得到了我们理论分析和实证分析的支持。因此，我们提出MoE-SpAc，一个MoE推理框架，它集成了用于追踪专家需求的投机效用估计器、通过在线整数优化动态划分计算的异构工作负载均衡器，以及在相同效用空间中统一预取与逐出操作的异步执行引擎。在七个基准测试上的大量实验表明，MoE-SpAc相对于基于投机解码的SOTA基线，TPS提升了42%……

    arXiv:2603.09983v2 Announce Type: replace-cross  Abstract: Mixture-of-Experts (MoE) models enable scalable performance but face severe memory constraints on edge devices. Existing offloading strategies struggle with I/O bottlenecks due to the dynamic, low-information nature of autoregressive expert activation. In this paper, we propose to repurpose Speculative Decoding (SD) not merely as a compute accelerator, but as an informative lookahead sensor for memory management, supported by our theoretical and empirical analyses. Hence, we introduce MoE-SpAc, an MoE inference framework that integrates a Speculative Utility Estimator to track expert demand, a Heterogeneous Workload Balancer to dynamically partition computation via online integer optimization, and an Asynchronous Execution Engine to unify the prefetching and eviction in the same utility space. Extensive experiments on seven benchmarks demonstrate that MoE-SpAc achieves a 42% improvement in TPS over the SOTA SD-based baseline, a
    
[^159]: CrimeNER：用于语言智能研究与调查的精选数据集与方法

    CrimeNER: Curated Dataset and Methods for Research and Investigation in Language Intelligence

    [https://arxiv.org/abs/2603.02150](https://arxiv.org/abs/2603.02150)

    该论文提出了CrimeNER，一个包含1500多份标注文档的犯罪领域命名实体识别数据集，定义了4种粗粒度和21种细粒度犯罪实体类型，并通过传统NER模型和大语言模型的实验验证了数据集质量。

    

    从犯罪相关文档中提取关键信息是执法机构的一项重要任务。这种信息的提取可以被理解为命名实体识别（NER）任务。然而，关于一般真实世界犯罪场景的充分标注数据相当匮乏。为解决这一问题，我们提出了CrimeNER，这是一项犯罪相关NER的案例研究，同时构建了一个通用的犯罪相关命名实体识别数据库，该数据库由从恐怖袭击公开报告和美国司法部新闻稿中提取的超过1500份标注文档组成。我们定义了4种粗粒度犯罪实体类型和21种细粒度实体类型。我们通过使用扁平与嵌套NER架构的全监督实验，以及使用大语言模型（LLMs）的零样本和少样本实验，验证了该数据库的质量。该数据库已在GitHub上公开提供。

    arXiv:2603.02150v3 Announce Type: replace-cross  Abstract: The extraction of critical information from crime-related documents is a crucial task for law enforcement agencies. The extraction of this information can be interpreted as a Named-Entity Recognition (NER) task. However, there is a considerable lack of adequately annotated data on general real-world crime scenarios. To address this issue, we present CrimeNER, a case study of crime-related NER, and a general crime-related Named-Entity Recognition database (CrimeNER-db), consisting of more than 1.5K annotated documents extracted from public reports of terrorist attacks and the US Department of Justice's press notes. We define 4 coarse types of crime entity and 21 fine-grained entity types. We address the quality of the presented database with fully supervised experiments using flat and nested NER architectures and zero- and few-shot experiments using Large Language Models (LLMs). The database is available on GitHub.
    
[^160]: 多样化“饮食”造就更健康的模型：法语 ModernBERT 案例研究

    A Diversity Diet for a Healthier Model: A Case Study of French ModernBERT

    [https://arxiv.org/abs/2602.22014](https://arxiv.org/abs/2602.22014)

    该研究表明，采用基于多样性的数据采样策略，可以在保持性能的前提下将法语 ModernBERT 的预训练数据集缩减高达 94%、预训练时间缩短高达 73%。

    

    近年来，多样性在自然语言处理（NLP）社区中日益受到关注。与此同时，ModernBERT 等最先进的 Transformer 模型使用的预训练数据集规模非常庞大，其构建主要基于数据规模而非多样性。这促使我们研究多样性对预训练的影响。我们在本研究中开展了这一探索，明确目标是缩减预训练数据集的规模，同时保持至少相当的性能。我们比较了多种基于多样性的采样算法，并用其中最优的算法在固定计算预算下预训练了多个法语 ModernBERT 模型。我们在多种法语基准测试上对这些模型进行微调和评估，并将它们与在同等规模随机采样数据上、以相同计算预算预训练的模型进行比较。我们发现，无论是随机采样还是基于多样性的采样，都可以将预训练数据集缩减最多 94%，预训练时间缩短最多 73%，同时保持性能。

    arXiv:2602.22014v2 Announce Type: replace  Abstract: Diversity has been gaining interest in the NLP community in recent years. At the same time, state-of-the-art transformer models such as ModernBERT use very large pre-training datasets, which are driven by size rather than by diversity. This summons to investigate the impact of diversity on pre-training. We do so in this study, with the express intent of reducing pre-training dataset size, while retaining at least comparable performance. We compare diversity-driven sampling algorithms, and we use the best one to pre-train several ModernBERT models on French with a fixed compute budget. We fine-tune and evaluate them on a variety of French benchmarks. We compare them with models pre-trained on randomly sampled data of commensurate size, with the same compute budget. We find that both random and diversity-driven sampling may reduce the pre-training dataset by up to 94% and the pre-training time by up to 73% while maintaining performance
    
[^161]: 在影响之下：评估与提升大语言模型在智能体任务中的警觉性

    Under the Influence: Evaluating and Improving LLM Vigilance in Agentic Tasks

    [https://arxiv.org/abs/2602.21262](https://arxiv.org/abs/2602.21262)

    该论文针对大语言模型智能体容易被外界影响的问题，提出在现实智能体任务中评估并提升其“警觉性”，即辨别何时应采纳善意建议、何时应拒绝恶意建议的能力。

    

    大语言模型（LLMs）正日益成为自主化助手，能够通过与软件数据库、互联网等开放世界领域交互来完成一系列复杂任务。这既带来了巨大的机遇，也伴随着巨大的风险：开放领域经常出现自主智能体可能被他人影响的机会，例如决定是否接受拉取请求，或是否采纳留言板上的建议来完成任务。至关重要的是，智能体必须能够辨别何时应将善意建议纳入其推理过程，何时应拒绝恶意建议。这涉及“警觉性”这一社交能力：即判断哪些信息应当采纳、哪些信息应当舍弃，从而实现有效沟通的能力。尽管警觉性在人类中已被充分研究，但LLM智能体表现出的警觉性程度却相对未知。本文在三个现实任务中考察了LLM智能体的警觉性：导航……

    arXiv:2602.21262v4 Announce Type: replace  Abstract: Large Language Models (LLMs) are becoming increasingly autonomous assistants, capable of accomplishing series of complex tasks by interacting with open world domains such as software databases and the internet. This presents great opportunity but also great risk: open domains routinely present opportunities for autonomous agents to be influenced by others, e.g. deciding whether to accept pull requests or whether to use advice on a message board in order to accomplish a task. Critically, agents must be capable of deciphering when to incorporate benevolent advice into their reasoning, and when to reject malicious advice. This involves the social capacity of vigilance: the ability to determine which information to use, and which to discard, for effective communication. While well-studied in humans, the extent to which LLM agents exhibit vigilance is relatively unknown. Here, we examine LLM agent vigilance in three realistic tasks: navig
    
[^162]: A-MapReduce：基于智能体化MapReduce的广度搜索执行框架

    A-MapReduce: Executing Wide Search via Agentic MapReduce

    [https://arxiv.org/abs/2602.01331](https://arxiv.org/abs/2602.01331)

    提出受MapReduce范式启发的多智能体框架A-MapReduce，将广度搜索重新构造为水平结构化检索问题，通过任务自适应分解、结构化结果聚合和经验记忆驱动的任务分配与重组，实现大规模检索目标的高效并行执行。

    

    当代基于大语言模型（LLM）的多智能体系统在深度研究任务中展现出系统性优势，这类任务强调迭代式的、纵向结构化的信息获取。然而，当面对以大规模、广度导向检索为特征的广度搜索任务时，现有的智能体框架——主要围绕顺序式、纵向结构化推理设计——仍然陷入庞大的搜索目标和低效的长程执行之中。为了弥合这一差距，我们提出了A-MapReduce，这是一个受MapReduce范式启发的多智能体执行框架，它将广度搜索重新构造为水平结构化的检索问题。具体而言，A-MapReduce通过任务自适应分解和结构化结果聚合，实现了对海量检索目标的并行处理。同时，它利用经验记忆来驱动基于查询条件的任务分配与重组的持续演化，从而不断提升执行效率。

    arXiv:2602.01331v2 Announce Type: replace-cross  Abstract: Contemporary large language model (LLM)-based multi-agent systems exhibit systematic advantages in deep research tasks, which emphasize iterative, vertically structured information seeking. However, when confronted with wide search tasks characterized by large-scale, breadth-oriented retrieval, existing agentic frameworks, primarily designed around sequential, vertically structured reasoning, remain stuck in expansive search objectives and inefficient long-horizon execution. To bridge this gap, we propose A-MapReduce, a MapReduce paradigm-inspired multi-agent execution framework that recasts wide search as a horizontally structured retrieval problem. Concretely, A-MapReduce implements parallel processing of massive retrieval targets through task-adaptive decomposition and structured result aggregation. Meanwhile, it leverages experiential memory to drive the continual evolution of query-conditioned task allocation and recomposi
    
[^163]: SteerEval：推理时干预增强神经摘要评估指标的多语言泛化能力

    SteerEval: Inference-time Interventions Strengthen Multilingual Generalization in Neural Summarization Metrics

    [https://arxiv.org/abs/2601.15809](https://arxiv.org/abs/2601.15809)

    该论文发现，通过推理时的激活干预将多语言神经摘要评估指标的内部表示引导向英语枢纽语言，可以普遍提升这些指标在多种语言上与人类判断的相关性。

    

    越来越多的研究利用多语言语言模型来执行诸如文本摘要等自然语言生成任务。该领域的一个主要实证瓶颈是许多语言缺乏准确且稳健的评估指标，这阻碍了研究进展。近期研究表明，多语言语言模型通常将英语作为内部枢纽语言，与该枢纽语言的不一致可能导致下游性能下降。基于这种不匹配也可能适用于多语言神经指标的假设，我们探究将多语言神经指标的激活引导向英语枢纽能否提高其与人类判断的相关性。我们对基于编码器和基于解码器的指标进行了实验，发现测试时干预方法普遍有效，能够提升指标对多种语言的有效性。

    arXiv:2601.15809v2 Announce Type: replace  Abstract: An increasing body of work has leveraged multilingual language models for Natural Language Generation tasks such as summarization. A major empirical bottleneck in this area is the shortage of accurate and robust evaluation metrics for many languages, which hinders progress. Recent studies suggest that multilingual language models often use English as an internal pivot language, and that misalignment with this pivot can lead to degraded downstream performance. Motivated by the hypothesis that this mismatch could also apply to multilingual neural metrics, we ask whether steering their activations toward an English pivot can improve correlation with human judgments. We experiment with encoder- and decoder-based metrics and find that test-time intervention methods are effective across the board, increasing metric effectiveness for diverse languages.
    
[^164]: 通过主动回顾学习增强大语言模型对长上下文的鲁棒理解

    Towards robust long-context understanding of large language model via active recap learning

    [https://arxiv.org/abs/2601.13734](https://arxiv.org/abs/2601.13734)

    提出主动回顾学习（ARL）框架，通过持续预训练中的针对性序列构建与推理时的自主回溯总结，使大语言模型在段落间建立递归记忆机制，显著提升长上下文理解能力（RULER提升最高达129.7%）。

    

    本文提出了主动回顾学习，这是一个用于增强大语言模型理解长上下文能力的框架。ARL使模型能够在持续预训练阶段通过有针对性的序列构建来回顾和总结先前的内容，并在推理阶段进行回溯式总结。首先，基于长前向上下文与短前向上下文之间的损失差距，在准备好的长上下文中识别关键token，并找出与之最相关的先前段落，然后使用大语言模型对其进行总结。其次，ARL使模型具备在推理过程中自主生成和利用这些回溯式总结的能力，从而在段落之间建立起递归记忆机制。实验结果显示了显著的性能提升，ARL在RULER基准上取得了高达129.7%的改进，在LongBench上取得了11%–13%的改进。总体而言，ARL提供了一种简单而有效的基于持续预训练的方法，以增强模型的长上下文理解能力。

    arXiv:2601.13734v2 Announce Type: replace-cross  Abstract: In this paper, we propose active recap learning (ARL), a framework for enhancing large language model (LLM) in understanding long contexts. ARL enables models to revisit and summarize earlier content through targeted sequence construction during continued pretraining and retrospective summarization at inference. First, we identify key tokens in prepared long context based on loss gaps between long and short forward contexts and find the most relevant preceding paragraphs, then summarize them using an LLM. Second, ARL equips models with the ability to autonomously generate and utilize these retrospective summaries during inference, thereby establishing a recursive memory mechanism across paragraphs. Experimental results show substantial gains, with ARL achieving an improvement of up to 129.7\% on RULER and 11--13\% on LongBench. Overall, ARL offers a simple yet effective continued pretraining-based approach to strengthen long-co
    
[^165]: PAIR-SAFE：一种用于运行时审计与优化AI介导心理健康支持的双智能体方法

    PAIR-SAFE: A Paired-Agent Approach for Runtime Auditing and Refining AI-Mediated Mental Health Support

    [https://arxiv.org/abs/2601.12754](https://arxiv.org/abs/2601.12754)

    提出PAIR-SAFE双智能体框架，由基于临床验证MITI-4标准的评判者智能体实时审计并指导修正LLM生成的心理健康支持回复，显著提升了伙伴关系、协作性和整体关系质量。

    

    大语言模型越来越多地被用于心理健康支持，但它们可能产生过于指令化、不一致或与临床实践不符的回复。大多数风险缓解方法依赖于通过训练或提示工程实现的隐式对齐，其运行时问责能力有限。我们提出了PAIR-SAFE，这是一个双智能体框架，将响应者智能体与基于临床验证的动机性访谈治疗完整性标准（MITI-4）构建的评判者智能体相结合。评判者对每个回复进行审计，并发出“允许”或“修改”的决定，以指导实时优化。我们使用从人工标注数据中构建的求助者模拟器来评估交互过程。评判者监督下的交互在关键MITI维度上显示出显著改进，包括伙伴关系、寻求协作以及整体关系质量。我们的定量研究结果得到了专家评估的印证，凸显了运行时监督的细微之处。

    arXiv:2601.12754v2 Announce Type: replace-cross  Abstract: LLMs are increasingly used for mental health support, yet they can produce responses that are overly directive, inconsistent, or clinically misaligned. Most risk mitigation approaches rely on implicit alignment through training or prompting, offering limited runtime accountability. We introduce PAIR-SAFE, a paired-agent framework that integrates a Responder agent with a Judge agent grounded in the clinically validated Motivational Interviewing Treatment Integrity (MITI-4). The Judge audits each response and issues \allow{} or \revise{} decisions to guide real-time refinement. We evaluate interactions using a support-seeker simulator derived from human-annotated data. The Judge-supervised interactions show significant improvements in key MITI dimensions, including Partnership, Seek Collaboration, and overall Relational quality. Our quantitative findings are corroborated by expert evaluation, highlighting nuances of runtime super
    
[^166]: Dementia-R1：基于非结构化临床笔记的强化预训练与推理，用于真实世界痴呆症预后预测

    Dementia-R1: Reinforced Pretraining and Reasoning from Unstructured Clinical Notes for Real-World Dementia Prognosis

    [https://arxiv.org/abs/2601.03018](https://arxiv.org/abs/2601.03018)

    提出了Dementia-R1框架，通过冷启动强化学习策略先预训练模型预测可验证的临床指标，再基于非结构化临床笔记进行痴呆症的纵向预后预测，实现了最佳整体性能。

    

    尽管大型语言模型（LLMs）在临床文本理解方面表现出强大的性能，但它们在痴呆症预后等纵向预测任务上仍然面临困难，这类任务需要对多次就诊中复杂、非单调的症状轨迹进行推理。标准的监督式训练缺乏对症状演变的显式标注，而直接的强化学习（RL）则受到稀疏二值奖励的阻碍。为了应对这一挑战，我们提出了Dementia-R1，一个基于强化学习的框架，用于从非结构化临床笔记中进行纵向痴呆症预后预测。我们的方法采用冷启动强化学习策略，预训练模型以预测从患者病史中提取的可验证临床指标，从而在确定最终临床状态之前增强模型对疾病进展进行推理的能力。大量实验表明，Dementia-R1在AMC真实世界非结构化临床数据集上取得了最佳的整体性能。

    arXiv:2601.03018v3 Announce Type: replace-cross  Abstract: While Large Language Models (LLMs) have shown strong performance on clinical text understanding, they struggle with longitudinal prediction tasks such as dementia prognosis, which require reasoning over complex, non-monotonic symptom trajectories across multiple visits. Standard supervised training lacks explicit annotations for symptom evolution, while direct Reinforcement Learning (RL) is hindered by sparse binary rewards. To address this challenge, we introduce Dementia-R1, an RL-based framework for longitudinal dementia prognosis from unstructured clinical notes. Our approach adopts a Cold-Start RL strategy that pre-trains the model to predict verifiable clinical indices extracted from patient histories, enhancing the capability to reason about disease progression before determining the final clinical status. Extensive experiments show that Dementia-R1 achieves the best overall performance on the AMC real-world unstructured
    
[^167]: EvalCards：一个用于标准化评估报告的框架

    EvalCards: A Framework for Standardized Evaluation Reporting

    [https://arxiv.org/abs/2511.21695](https://arxiv.org/abs/2511.21695)

    本文提出 EvalCards 框架，围绕可复现性、可访问性和可问责性三个核心维度规范 AI 模型评估报告，并提供简洁评估摘要的初步格式，以推动未来评估报告标准的建立。

    

    评估报告为通用人工智能模型的使用提供指导，然而不一致的报告方式往往使关键问题得不到解答。通过对以往工作的回顾，我们指出了当今评估报告实践中存在的三个核心问题——可复现性、可访问性和可问责性，并检视了现有报告工件在解决这些问题上的不足。我们提出了 EvalCards 框架，阐述了其关键维度，并给出了一种简洁评估摘要的初步格式。最后，我们讨论了该框架如何为未来评估报告标准的制定提供指导。

    arXiv:2511.21695v2 Announce Type: replace-cross  Abstract: Evaluation reports guide the use of general-purpose AI (GPAI) models, yet inconsistent reporting often leaves essential questions unanswered. Through a review of prior work, we identify three central issues in evaluation reporting practices today, reproducibility, accessibility, and accountability, and examine how existing reporting artifacts fall short in addressing them. We introduce the EvalCards framework, outlining its key dimensions and presenting a preliminary format for concise evaluation summaries. We conclude by discussing how the framework can guide the future development of evaluation reporting standards.
    
[^168]: 多模态大语言模型在学习多模态推理时究竟学到了什么

    What MLLMs Learn about When they Learn about Multimodal Reasoning

    [https://arxiv.org/abs/2510.01719](https://arxiv.org/abs/2510.01719)

    该论文提出MathLens几何推理基准，将多模态推理性能分解为感知、推理和多模态组件，揭示了强化学习与文本SFT等常见训练策略会产生在总体准确率下不可见的系统性不同能力特征。

    

    对多模态推理模型的评估通常被简化为单一的准确率分数，这隐含地将推理视为一种单一的能力。我们提出了MathLens，这是一个由教科书式几何问题组成的基准，通过将模型性能可操作地分解为感知、推理和多模态特定组件，来揭示上述假设的局限。每个问题均源自符号化规范，并配有可视化图形、纯文本变体、多模态问题和针对性的感知探针，从而能够对各组件进行受控测量。利用这种分解方法，我们表明常见的训练策略与系统性不同的能力特征相关联，而这些差异在总体准确率指标下是不可见的。在我们研究的模型中，强化学习主要提升了感知基础能力和对图形变化的鲁棒性，而我们的分析表明文本SFT则通过反思性

    arXiv:2510.01719v5 Announce Type: replace  Abstract: Evaluation of multimodal reasoning models is typically reduced to a single accuracy score, implicitly treating reasoning as a unitary capability. We introduce MathLens, a benchmark of textbook-style geometry problems that exposes this assumption by operationally decomposing performance into perception, reasoning, and multimodal-specific components. Each problem is derived from a symbolic specification and accompanied by visual diagrams, text-only variants, multimodal questions, and targeted perceptual probes, enabling controlled measurement of each component. Using this decomposition, we show that common training strategies are associated with systematically different capability profiles that are invisible under aggregate accuracy. Across the models we study, reinforcement learning primarily improves perceptual grounding and robustness to diagram variation, while our analysis suggests that textual SFT yields gains through reflective 
    
[^169]: 自然语言处理中多样性量化的综述：为什么、是什么、在哪里以及如何

    A survey of diversity quantification in natural language processing: The why, what, where and how

    [https://arxiv.org/abs/2507.20858](https://arxiv.org/abs/2507.20858)

    该论文借鉴生态学与经济学的跨领域统一框架，调研了300多篇NLP多样性相关文献，提出了涵盖“为什么重要、衡量对象、衡量位置、衡量方法”四个视角的NLP多样性量化专用框架，以解决该领域度量标准碎片化和不一致的问题。

    

    近年来，多样性（diversity）的概念在自然语言处理（NLP）领域受到了越来越多的关注。它已成为数据集和系统所推崇的一种属性，人们使用了许多度量方法来量化它，但论文之间也存在许多不一致之处。为了解决这种碎片化问题，我们从多样性概念化更为深入的其他科学领域中汲取灵感。我们借鉴了（Sarkar, 2010）和（Stirling, 2007）这两个受生态学、经济学和其他领域启发的统一跨领域框架。前者主张对多样性认同进行明确的论证，后者区分了多样性的三个维度：丰富度、均衡性和差异性。通过调研 ACL Anthology 中 300 多篇近期与多样性相关的论文，我们提出了一个 NLP 专用框架，包含 4 个视角：为什么多样性重要、用什么来衡量多样性、在哪里衡量多样性以及如何衡量。我们发现，在 NLP 中，丰富度（最简单的……（摘要不完整，在“丰富度”处截断）

    arXiv:2507.20858v4 Announce Type: replace  Abstract: The concept of diversity received increasing attention in natural language processing (NLP) in recent years. It became an advocated property of datasets and systems, with many measures used to quantify it, but also many cross-paper inconsistencies. To address this fragmentation, we take inspiration from other scientific fields where diversity has been more thoroughly conceptualized. We build upon (Sarkar, 2010) and (Stirling, 2007), two unified cross-domain frameworks inspired from ecology, economy and other domains. The first one advocates explicit justification of diversity endorsement. The latter distinguishes three dimensions of diversity: variety, balance, and disparity. Surveying over 300 recent diversity-related papers from ACL Anthology, we propose an NLP-specific framework with 4 perspectives: why diversity is important, what diversity is measured on, where it is measured, and how. We find that in NLP richness (the simplest 
    

