# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [WearableQA: A Benchmark for Health Reasoning over Real-World Wearable Data](https://arxiv.org/abs/2609.05405) | 该论文提出了WearableQA基准，基于200名真实用户长达500天的可穿戴数据构建了4,084道多项选择题，并通过数据/健康推理和单信号/跨信号推理两个维度的16种问题类型，评估AI系统对真实世界可穿戴数据的健康推理能力。 |
| [^2] | [Same Trajectory, Contradictory Rewards (ROBORMBENCH): Paraphrase Fragility in Vision Language Reward Models](https://arxiv.org/abs/2609.05401) | 现有视觉语言奖励模型缺乏释义不变性——仅改写指令就能大幅改变同一机器人轨迹的奖励评分，甚至翻转成败判定，新基准ROBORMBENCH证实这种脆弱性普遍存在，且难以靠扩大模型规模或显式推理来缓解。 |
| [^3] | [Multi-Step Tool-Calling over Korean Open Public APIs: A Benchmark and a Data-Synthesis Recipe](https://arxiv.org/abs/2609.05395) | 该论文提出韩国开放公共API基准KOPA-Bench（145个真实任务）和基于实时执行验证的数据合成方法EDGE，通过合成可执行的多步工具调用轨迹并用GRPO微调，使9B开源模型性能几乎追平同系列27B模型。 |
| [^4] | [Does Your Agent's Memory Survive a Model Upgrade? A Controlled Study of Memory Portability](https://arxiv.org/abs/2609.05339) | 该对照研究发现智能体记忆的可移植性取决于存储格式：固定模式知识图在模型升级后准确率几乎不变，而由模型压缩生成的自然语言笔记与原模型高度耦合，迁移后性能会大幅不对称波动。 |
| [^5] | [Technical Manual for a Toolkit for Measuring Contextual Individuation in Transformer Language Models](https://arxiv.org/abs/2609.05333) | 本文提出一个基于“桥接词形”构念的开源工具包，通过让同一词形在不同主题领域中承载不同词义，来系统测量Transformer语言模型在后续层中是否能依据上下文对相同词形的不同出现进行个体化区分。 |
| [^6] | [Large Language Models for HVAC Operations in Building Energy Systems: A Critical Review of Methods, Applications, and Deployment Readiness](https://arxiv.org/abs/2609.05314) | 本系统性综述分析了66篇关于大语言模型用于建筑暖通空调运行的研究，发现该领域主要集中于建筑能源建模，但绝大多数研究仍停留在研究阶段，目前尚无任何研究达到可立即产业部署的成熟度。 |
| [^7] | [LexFlip: A Dissociation Diagnostic for Legal Meaning Preservation Metrics](https://arxiv.org/abs/2609.05296) | 本文提出LexFlip诊断集，通过373个保持词汇表面形式不变却逆转法律效力的魁北克法语法条最小扰动，揭示了现有嵌入类语义度量几乎无法察觉法律含义变化，暴露了当前法律文本简化评估中“相同对检验”的根本缺陷。 |
| [^8] | [Self-Supervised Lexical Representation Learning for Fast, Large-Scale Phylogenetic Inference](https://arxiv.org/abs/2609.05262) | 本文提出一种无需同源词标注的全自监督双重对比学习框架，直接从IPA音标词表学习词汇表示，从而快速、低成本地推断大规模全球语言的系统发育树。 |
| [^9] | [A Verifier-Guided Explainable Reasoning Framework with Gold-Anchored QLoRA, Task-Aware Mixture-of-Experts, and Group-Relative RLVR](https://arxiv.org/abs/2609.05221) | 该论文提出一种面向教育问答的可解释推理框架，通过黄金锚定QLoRA微调、任务感知符号路由（逻辑问题交由FOL/Z3验证器、物理问题交由公式与单位感知求解器）以及组相对RLVR强化学习，从正确性、一致性和推理深度三个维度提升大模型推理的可验证性与可解释性。 |
| [^10] | [Can Large Language Models Anticipate Behavioral Responses to Social Policies? A Case of Pension Enrollment Prediction among China's Flexible Workers](https://arxiv.org/abs/2609.05189) | 本文提出首个面向中国灵活就业人员的领域专用大语言模型 FlexPension-LLM，通过 DKI-RDistill 方法将政策相关线索（如 Probit 边际效应和户籍省份养老规则）注入提示词并进行知识蒸馏，实现养老保险参保行为的精准预测，为社会政策评估提供了低成本的新工具。 |
| [^11] | [Measuring the Novelty of Biomedical Papers Using the Latent Distances between Knowledge Units](https://arxiv.org/abs/2609.05175) | 该论文提出了一种综合性的新颖性测量方法，通过量化知识单元之间的网络、语义和层级三种关系所对应的潜在距离，突破了以往仅依赖共现关系评估论文新颖性的局限。 |
| [^12] | [Compression Beyond the Uncompressed: A Two-Stage Training Recipe for Soft Context Compression in RAG](https://arxiv.org/abs/2609.05152) | 提出两阶段训练方案DEX-Comp，通过纯蒸馏热启动加困难查询强化学习探索，使RAG中的软上下文压缩模型性能超越未压缩系统，同时实现16倍的上下文压缩。 |
| [^13] | [Large Language Models with At Most One Spike per Neuron](https://arxiv.org/abs/2609.05151) | 该论文提出一种基于参考的编码策略，首次构建了完全基于TTFS编码、每个神经元至多发放一次脉冲的脉冲神经网络大语言模型架构，可端到端训练，并在BERT和GPT-2等模型上实现了与人工神经网络相当的性能与显著节能优势。 |
| [^14] | [From Vision to Language: Investigating Causal Information Flow in Multimodal Decision-Making](https://arxiv.org/abs/2609.05149) | 通过对视觉-语言模型的视频-文本注意力通路进行逐层因果干预，发现视觉信息主要在处理候选答案选项时被整合进决策，其中名词作为语义锚点、动词对时间推理至关重要，并揭示了VLM在时间推理上的独特缺陷模式。 |
| [^15] | [A Human-in-the-Loop Framework for AI-Assisted Scoring in Large-Scale Writing Assessment](https://arxiv.org/abs/2609.05143) | 本研究提出并验证了一个面向大规模写作测评的AI辅助评分人机协同框架，基于约一万份真实全国考试数据，在多个评分维度上实现了AI与人工评分的中高程度一致性，有效降低人工评分工作量。 |
| [^16] | [NS-ST-GraphRAG: Neuro-Symbolic Spatio-Temporal GraphRAG for Literary Knowledge Processing](https://arxiv.org/abs/2609.05139) | 提出了神经符号时空GraphRAG框架NS-ST-GraphRAG，通过选择符合查询时空范围的图状态并提供可追溯证据来处理长篇文学叙事中受时间、空间和关系约束的问答，并发布了首个中国古典文学开放多跳问答基准Red-Chamber-QA。 |
| [^17] | [Improving Language Identification for Code-Switched Utterances with Integer Linear Programming](https://arxiv.org/abs/2609.05099) | 本文通过解决MaskLID过度依赖词级语言关联分数的问题，并将其底层优化算法重新表述为可引入清晰可解释约束的整数线性规划，显著提升了码切换话语的语言识别性能。 |
| [^18] | [Measuring AI Accountability Through Argumentation Analysis: Can Model Reasoning Withstand Scrutiny?](https://arxiv.org/abs/2609.05088) | 该论文提出一种基于沃尔顿论证图式理论与Govier论证合理性标准的四阶段辩证评估协议，通过衡量模型在面对批判性问题时为其裁决辩护的结构质量，实现了在缺乏公认正确答案的真实模糊情境下对大语言模型道德推理的严格评估。 |
| [^19] | [TruthInsightBench: An Evidence-Grounded Benchmark for Automated Evaluation of Open-Ended Scientific Discovery Agents](https://arxiv.org/abs/2609.05079) | 该论文提出TruthInsightBench——首个面向开放式科学发现（而非结果复现）的智能体评估基准，通过40个隐藏源结论的盲测任务与基于LLM的六维证据成熟度自动评分体系，实现了无需逐实例人工评分的科学发现智能体自动化评估。 |
| [^20] | [Influence Score and Transformers interpretability: Measure of the Effective Impact of Attention Heads at inference time](https://arxiv.org/abs/2609.05074) | 本文提出一种结合logits方向性影响与残差流结构性贡献的影响分数，可在注意力头、层和网络三个尺度上量化注意力头对分类决策的贡献，并在提示注入检测的DeBERTa模型上揭示了正确与错误预测的不同决策行为。 |
| [^21] | [A Structured Debate-Mixture-of-Agents Framework for Complex Clinical Diagnostic Decision Support](https://arxiv.org/abs/2609.05069) | 提出了一种结构化的辩论-智能体混合框架（DMoA），通过多智能体角色交互实现迭代式诊断推理，在罕见病和疑难病例上相比GPT-4o基线将诊断准确率提高10.21个百分点、安全率提高11.36个百分点。 |
| [^22] | [Repeated Queries Exhaust an LLM's Brand Recommendations but Not Its Sources](https://arxiv.org/abs/2609.05059) | 重复提问相同的购买问题时，不联网检索的大语言模型会不断涌现新的品牌推荐而难以饱和，启用检索的引擎则快速封闭品牌列表，但所有引擎引用的来源域名始终持续增加、远未收敛。 |
| [^23] | [EuroAlpaca: Task-Preserving Localisation of Instruction Data for European Languages](https://arxiv.org/abs/2609.05043) | 本文提出EuroAlpaca任务保持式本地化流程和European-IFEval多语言基准，覆盖50种欧洲语言，并证明直接机器翻译的指令数据会损害模型的可验证指令遵循能力（准确率下降29.8%），而任务保持式本地化可避免此问题。 |
| [^24] | [How do LLMs Evaluate Perceived Moral Agency? Investigating Moral Decision-Making in Human-Artificial Agents Interactions](https://arxiv.org/abs/2609.05037) | 本文首次通过实证研究比较了人类与大语言模型在智慧城市场景中评估人类与自主人工体感知道德能动性（PMA）的方式，发现LLMs表现出更高的道德能动性感知。 |
| [^25] | [Moral Competence Before Moral Content: Why LLM Agents Lack the Prerequisites for Coherent Alignment](https://arxiv.org/abs/2609.05036) | 本文提出判断稳定性、单调性、果断性和帕累托可行性四个结构性条件，用以衡量仅凭行为即可评估的“道德能力”，并揭示前沿大语言模型智能体普遍缺乏实现连贯对齐所需的这一结构性前提。 |
| [^26] | [Leveraging Low-Level Symbolic Competences for Unsupervised Grounding in Hallucination Detection](https://arxiv.org/abs/2609.05025) | 本文提出让LLM根据参考文档构建SQL数据库，并将其作为无监督幻觉检测的接地依据，通过这种神经符号化方法在RAGTruth和DiaHalu数据集上超越了直接预测并与最先进方法相媲美。 |
| [^27] | [MoirfEolas and Cr\'iochScore: Developing Resources for and the Evaluation of Tokenization Alignment with Irish Morphology](https://arxiv.org/abs/2609.05022) | 本文提出了爱尔兰语形态数据集 MoirfEolas 和分词评估指标 CríochScore，发现 Unigram 语言模型的分词与爱尔兰语形态边界对齐程度最高，且形态对齐与压缩率及词汇效率之间存在权衡。 |
| [^28] | [BIT.UA at BioASQ 14B: Modular Retrieval with pg_textsearch and Qdrant, and Agent-Based Answer Generation](https://arxiv.org/abs/2609.04999) | BIT.UA团队在BioASQ 14B生物医学问答挑战赛中，通过采用基于PostgreSQL pg_textsearch和Qdrant的模块化检索架构，以及LLM-as-a-judge框架和新型智能体法定人数机制进行答案生成，显著改进了检索与生成流水线。 |
| [^29] | [BeaconKV: Key-Value Cache Compression Guided by Beacon Queries for Efficient Large Reasoning Model Inference](https://arxiv.org/abs/2609.04971) | 该论文发现长程推理中存在会重新关注早期关键上下文的“思维回溯token”，且其查询在嵌入空间中聚成少数相似组，据此提出无需训练的KV缓存压缩方法BeaconKV，从而高效支持大型推理模型的推理。 |
| [^30] | [Why We Care About Understanding: Competence through Predictive Compression](https://arxiv.org/abs/2609.04962) | 本文提出理解是对稳健能力的有效代理——通过掌握可实现预测的关系结构心理模型进而实现压缩——从而弥合了信息论“理解即压缩”传统与哲学家对理解的概念刻画之间的鸿沟。 |
| [^31] | [Discourse Dependency: A Continuous Criterion for Translation Difficulty](https://arxiv.org/abs/2609.04959) | 本文提出“语篇依赖度（DDP）”这一免指标的、基于源端的连续翻译难度度量，通过实体复现与代词共指衡量片段所需的指称回溯距离，揭示现有翻译基准严重偏向低DDP片段，并证明随着DDP增大，现有的各类上下文注入策略均无法达到人工译后编辑的水平。 |
| [^32] | [RefactorPlatform: An Open-Source Harness for Controlled Evaluation of Repository-Scale Refactoring Agents](https://arxiv.org/abs/2609.04898) | 本文提出开源评估平台RefactorPlatform，通过固定环境并显式变化模型骨干、执行模式和提示词等设计维度，实现对仓库级重构智能体的受控评估，并发现AST感知分块相比朴素token窗口分块可带来25-30%的性能提升。 |
| [^33] | [Cache-Aware Joint Router Adaptation for Memory-Efficient MoE Inference](https://arxiv.org/abs/2609.04895) | 提出一种缓存感知的后训练框架，通过联合自适应MoE主干与轻量级时间-空间缓存路由器，在不改变原生Top-K专家选择规则的前提下提升缓存命中率、减少专家权重传输，实现内存高效的MoE推理。 |
| [^34] | [CC-Mediation: Evaluating Large Language Models for Cross-Cultural Conflict Mediation](https://arxiv.org/abs/2609.04855) | 提出了基于跨文化敏感性发展模型（DMIS）构建的CC-Mediation基准，包含1,661段十轮跨文化冲突调解对话，并设计了轨迹AUC和带符号Wasserstein-1距离两个新指标来评估大语言模型调解干预的效果与跨文化立场变化。 |
| [^35] | [MMTClinic: Multimodal, Multilingual Time Series Question Answering and Reasoning Benchmark for Clinical Domain](https://arxiv.org/abs/2609.04842) | 本文提出了MMTClinic——首个面向临床领域的多模态、多语言时间序列问答与推理基准，融合文本、医学图像和生理信号，包含覆盖五种语言的30,000个问答对，用于评估大语言模型在真实临床场景中的推理能力。 |
| [^36] | [MABPD: Multi-Agent Bias Probing & Detection via Structured Argument Debate](https://arxiv.org/abs/2609.04841) | 该论文提出MABPD框架，让三个专门的LLM智能体通过结构化论证辩论（SAD）协议协作分析新闻文章，利用非对称举证责任、角色加权投票和共识后验证机制，实现无需监督训练的媒体偏见检测。 |
| [^37] | [On Epistemic Diversity in Large Language Models](https://arxiv.org/abs/2609.04835) | 该论文借鉴哲学与社会认识论中的认知多样性概念，提出将其——即LLM向用户展现的有效答案、解释和推理路径的范围——作为评估大型语言模型的新维度，并发现前沿LLM常表现出认知狭隘性。 |
| [^38] | [Generating Constructive Feedback on Stories via Reinforcement Learning](https://arxiv.org/abs/2609.04824) | 本文提出一种基于GRPO的强化学习方法，通过新颖的多组件奖励函数，在无需真实反馈标注的情况下引导大语言模型生成针对故事量身定制、有助提升故事质量并聚焦最关键写作问题的建设性反馈。 |
| [^39] | [Reinforcement Learning for improving Large Language Models' Catalan text simplification capabilities](https://arxiv.org/abs/2609.04823) | 本文提出了一种结合SARI指标与惩罚组件的新型奖励函数，利用GRPO强化学习对大语言模型进行后训练，有效提升了加泰罗尼亚语这一低资源语言的文本简化能力。 |
| [^40] | [A Systematic Comparison of Multilingual Interpretability Methods Reveals Anisotropy-Driven Failures](https://arxiv.org/abs/2609.04819) | 本研究系统比较了四种跨语言共享度度量方法，发现它们结论分歧的根源在于表示的各向异性，且只有ILO与跨语言迁移的相关性（ρ=0.90）经得起模型规模、家族等控制变量的检验，因而推荐使用ILO。 |
| [^41] | [Recurrence Is Not Enough: Causally Validating Multilingual SAE Translation Features in Gemma 2 and 3](https://arxiv.org/abs/2609.04808) | 尽管能在 Gemma 2 和 Gemma 3 中找到 20 多个跨多语言设置频繁重复出现的 SAE 翻译启动特征，但因果验证表明这些特征对翻译行为几乎没有显著或一致的影响，说明特征的重复出现并不足以证明其因果有效性。 |
| [^42] | [Can Activation Steering Capture Multidimensional Authorship Style?](https://arxiv.org/abs/2609.04792) | 提出无需训练的“方面感知激活导向”框架A3S，通过融合各修辞维度的对比方向与干扰感知聚合，直接在激活空间中捕捉多维度的作者写作风格，显著提升多方面作者风格迁移效果。 |
| [^43] | [Persistent Teacher Anchoring for Tool-Using Agents](https://arxiv.org/abs/2609.04773) | 提出持久教师锚定（PTA），一种由学生诱导但由教师承诺的 rollout 构建方法，通过在块级验证基础上增加轮级承诺让工具调用得以执行，从而解决工具使用中师生分布差距累积导致的漂移问题。 |
| [^44] | [Vectorizing Classical Tamil: Representation Learning for Verse-Commentary Pairs](https://arxiv.org/abs/2609.04755) | 本文构建了包含1,262对语料的古典泰米尔语诗句-注释数据集，并通过多种表示学习模型与严格对照实验发现，简单的TF-IDF词汇基线可媲美甚至超越深度生成模型，从而揭示了表示学习在低资源古典文献上的真实能力边界。 |
| [^45] | [Beneath the Surface of Chains-of-Thought: A Mechanistic Interpretation of Reasoning Operations in LLMs](https://arxiv.org/abs/2609.04753) | 该研究揭示了大语言模型思维链中的不同推理操作在隐藏表示空间中具有可分离的几何结构，这种结构在中间层最为显著，且注意力掩码干预表明文本块起始处的操作对齐表示依赖于前序推理内容。 |
| [^46] | [Knowing What Not to Answer: Selective Non-Compliance in Vision-Language Models](https://arxiv.org/abs/2609.04720) | 该论文提出了KoNA基准，涵盖错误前提、视觉不可及性等五个类别，用于评估视觉语言模型在单一查询和复合查询两种情形下，区分应回答内容与应拒绝内容的选择性拒绝遵从能力。 |
| [^47] | [Refuse without Refusal: A Structural Analysis of Safety-Tuning Responses for Reducing False Refusals in Language Models](https://arxiv.org/abs/2609.04714) | 该论文创新性地将安全调优响应分解为模板化拒绝声明与拒绝理由两个部分，发现拒绝声明会诱导模型依赖表面线索而造成误拒，而仅用拒绝理由进行训练能有效减少语言模型的误拒行为。 |
| [^48] | [How Do Language Models Represent and Use Phonological Information for Allomorph Selection?](https://arxiv.org/abs/2609.04708) | 该论文发现语言模型在嵌入空间中以单一线性方向编码音系条件，并通过预测即将出现的触发词的音系特征来因果性地驱动英语不定冠词 a/an 的选择，展现出规则式泛化而非单纯记忆的能力。 |
| [^49] | [Retinal OCTA Phenotyping with LLM Reporting for Alzheimer's Disease](https://arxiv.org/abs/2609.04689) | 该论文提出了一种可解释的视网膜OCTA分析流程，通过整合标注感知的血管分割、分层血管生物标志物提取、无标签表型分析与基于测量结果的大语言模型报告生成，实现了阿尔茨海默病的早期无创筛查与可解释诊断。 |
| [^50] | [Controlling and Assessing Appropriate Persona Use in LLM-based Dialogue Generation](https://arxiv.org/abs/2609.04676) | 本文揭示了LLM在角色对话生成中存在系统性融入所有角色属性的偏差，并提出SCONPOS方法通过干预模型内部表示来抑制过度使用，同时提出PAS指标评估角色使用的恰当性。 |
| [^51] | [Choosing the Right Language Mode at Inference Time for Multilingual Reliability](https://arxiv.org/abs/2609.04653) | 提出了一种无需训练的测试时框架 RAAI，通过在推理时自适应选择最合适的语言模式（目标语言、英语或双语），在准确性与可靠性之间取得平衡，从而提升多语言大模型的推理可靠性。 |
| [^52] | [ConsensusBench: Benchmark of Consensus Nodes for LLM Reasoning via Outcome Reward Densifying](https://arxiv.org/abs/2609.04648) | 该论文提出ConsensusBench数据集，通过识别推理过程中可验证的中间结论（共识节点）来密集化结果奖励，为LLM推理的强化学习提供基于规则的过程级奖励信号，弥补稀疏最终答案奖励的不足。 |
| [^53] | [CAGE: Coherence-Aware Graph Encoding for Retrieval-Augmented Generation](https://arxiv.org/abs/2609.04647) | CAGE提出了一种连贯性感知的图编码重排序框架，通过有向异构图建模检索段落间在领域相关性、抗噪性、信息联结和事实一致性四个维度的块间连贯性，显著提升了RAG系统在多跳问答任务上的检索与生成效果。 |
| [^54] | [Latent-Aligned Reasoning for Multimodal Recommendation](https://arxiv.org/abs/2609.04645) | 提出LARK两阶段潜空间推理框架，通过可学习token与冻结视觉编码器对齐及物品对比学习，解决多模态推荐中VLM多步推理导致的视觉与文本信号衰减（跨模态稀释）问题。 |
| [^55] | [Tracing Audio Grounding and Answer Selection in Audio LLMs](https://arxiv.org/abs/2609.04637) | 该研究揭示了音频大语言模型内部音频真正决定答案的机制：训练主要在中间至后期层增强音频对最终预测的影响，而声学信息主要在早至中间层塑造答案选项的表示。 |
| [^56] | [PetQA: Benchmarking Veterinary Knowledge and Clinical Reasoning](https://arxiv.org/abs/2609.04598) | 本文提出PetQA——首个用于评估大语言模型和视觉-语言模型兽医知识与临床推理能力的韩语长文本问答基准，包含近1.9万个源自真实猫狗病例的文本与多模态问答对，并对18个模型在零样本、RAG和微调三种设置下进行了系统评估。 |
| [^57] | [JLIR: A Julia-Native MLIR-Inspired Intermediate Representation with Automatic JACC Kernel Extraction](https://arxiv.org/abs/2609.04585) | 该论文提出JLIR这一Julia原生、受MLIR启发的中间表示框架，克服了MLIR对动态语言类型要求过强和C++扩展门槛高的问题，使Julia科学计算抽象能自然融入编译器优化路径，并支持自动提取JACC内核。 |
| [^58] | [When Do Internal Probes Beat Reading the Answer? Miscalibrated Readouts and Behavior-Concealed Knowledge in Language Models](https://arxiv.org/abs/2609.04582) | 该论文发现语言模型内部（隐藏状态与输出logits边际）早已编码了正确的判断信息，但由于决策阈值过度饱和偏移而被行为层面掩盖，行为准确率坍缩为阈值偏移的单一函数（Spearman -0.93），揭示了“行为隐藏知识”现象。 |
| [^59] | [Does the Selected Object Reach the Reader? Auditing Identity Handoffs in Grounded Language-Model Pipelines](https://arxiv.org/abs/2609.04579) | 该论文系统审计了基于事实的语言模型流水线中对象从选择器到阅读器的身份交接问题，发现仅正文BM25检索会遗漏26.6%的所选对象，而带重排序的混合检索仅遗漏1.0%，且所选对象身份与数据集关联身份在约18%的记录中存在差异。 |
| [^60] | [Extremely Sparse Supervision Incentivizes Reasoning Ability](https://arxiv.org/abs/2609.04565) | 研究发现在在线策略蒸馏中，仅需对每个推理轨迹中的一两个token（约占总token数的0.05%）进行监督，就能有效激励大语言模型的推理能力，且在多数情况下可匹配甚至超越全token训练的效果。 |
| [^61] | [Rhythms of Work: Multi-Scale Interpretation of Human Behavioral Traces for Workplace Agents](https://arxiv.org/abs/2609.04556) | 该论文提出行为解释依赖于时间分辨率这一核心观点，构建了由重复模式、连贯情节和日级节奏组成的多分辨率语义规范化词汇表，使职场智能体能够在不同时间尺度上对人类行为轨迹获得多种可寻址的解释，并基于6.67亿个人类归因事件进行了验证。 |
| [^62] | [A Calibrated Reflection Approach for Enhancing Confidence Estimation in LLMs](https://arxiv.org/abs/2609.04539) | 该论文提出了一种融合最大置信度选择、反思式提示和距离感知校准三项创新的校准反思框架，显著提升大语言模型的置信度估计能力，从而帮助系统判断何时信任模型输出、何时寻求人工干预。 |
| [^63] | [Scale-QLoRA: Code-Invariant Adapter Merging for Native 4-bit Microscaling LLMs](https://arxiv.org/abs/2609.04526) | Scale-QLoRA通过只训练原生4比特微缩放检查点的每块缩放因子、同时冻结全部E2M1编码来合并LoRA适配器，从而避免了朴素合并因重新量化编码平面而抹除适配效果（最高达39个百分点）的问题。 |
| [^64] | [LentEx: Generalizable Latent Entity Extraction via Synthetic Data and Instruction-Tuned LLMs](https://arxiv.org/abs/2609.04511) | LentEx利用基于模板的合成数据生成与指令微调技术优化小型大语言模型，首次系统性地实现了对文本中隐含、抽象潜在实体的泛化抽取。 |
| [^65] | [Rethinking Indirect Prompt Injection as a Test-Time Search Problem](https://arxiv.org/abs/2609.04495) | 该论文将间接提示注入重新定义为在任务相关攻击面上的测试时搜索问题，并提出一个具备环境侦察、策略推理与自适应评估能力的智能体攻击框架，证明攻击成功率随攻击者测试时计算预算的增加而提升，因此安全评估应同时刻画攻击者的搜索过程与计算预算。 |
| [^66] | [Towards Understanding Pause Token Fine-Tuning Dynamics: A Mode Retention Perspective](https://arxiv.org/abs/2609.04489) | 本文提出掩码边界暂停标记方法（MBP），首次从模式保持与非短视压缩的训练动力学视角揭示了暂停标记提升推理能力的机制，并在1B至8B规模的Qwen和Llama模型上带来最高6个百分点的推理提升。 |
| [^67] | [Uncertainty Signals for Network Intent Translation: Risk Ranking and Ambiguity Localization](https://arxiv.org/abs/2609.04486) | 本文提出利用基于采样的预测不确定性对LLM生成的网络配置进行部署前翻译风险排序，并利用词元级熵定位歧义来源，从而弥补现有研究只关注翻译准确性而忽视部署风险的不足。 |
| [^68] | [Cultural Misalignment in Large Language Models: Detection, Measurement, and Mitigation Through Targeted Fine-Tuning](https://arxiv.org/abs/2609.04485) | 研究发现开源大语言模型并不偏袒其本国文化，且仅需约1,200个样本的针对性LoRA微调虽能降低文化偏差约17%，但实际上是重新分配而非消除偏差。 |
| [^69] | [Patterns of Priming in Production: Lexical, Semantic and Structural Alignment in Language Model Generation](https://arxiv.org/abs/2609.04484) | 本研究通过受控句子补全实验证明语言模型在生成时同样会受到结构启动效应的影响，且启动幅度呈现逆频率效应——较少出现的双宾语结构相对增幅更大，而更常见的介词宾语结构绝对增幅更大。 |
| [^70] | [Safety for Whom? Boundary-Aware Self-Distillation for Controlled LLM Safety Refusal](https://arxiv.org/abs/2609.04482) | 该论文提出“窄边界安全”新范式，通过结合升级重试机制的边界感知自蒸馏框架，使大语言模型能在同一主题内根据部署场景精细化控制拒绝边界，将目标域拒绝率从9.47%提升至84.75%，同时显著降低更广泛基准上的不安全响应率。 |
| [^71] | [Shared circuits predict whether LLMs generalize across formats in arithmetic reasoning](https://arxiv.org/abs/2609.04463) | 本研究通过归因修补技术定位大语言模型解决数字与语言表述算术问题的内部电路，发现模型语言电路与其数字电路的重叠程度能够预测其跨格式泛化能力。 |
| [^72] | [When Load-Balancing Goes Too Far: Expert Pruning in Over-Dispersed Mixture-of-Experts Models](https://arxiv.org/abs/2609.04453) | 该论文发现在因训练时负载均衡过于激进而导致路由过度分散的MoE模型中，路由器概率不再是可靠的专家重要性信号，困惑度也无法预测下游任务准确率，因此传统的基于路由的专家剪枝方法在此类模型中会失效，且不同评分指标之间存在能力权衡。 |
| [^73] | [TRILOGUE: A Trilingual Spoken Dialogue Fact-Checking Benchmark with Evidence and Paired Audio](https://arxiv.org/abs/2609.04452) | TRILOGUE是首个大规模三语（英语、俄语、哈萨克语）口语对话事实核查基准，包含近1.2万个对话、18.7万个轮次和390小时的配对音频，填补了带配对语音和轮次级标签的大型多语言事实核查基准的空白。 |
| [^74] | [Conformity Breaks Conformal Prediction](https://arxiv.org/abs/2609.04445) | 论文揭示同伴压力会使多智能体LLM系统发生“评分机制偏移”，悄然破坏共形预测的覆盖率保证（从90%降至74%），且攻击者可通过针对低置信度子组将其覆盖率近乎减半（从87%降至47%）。 |
| [^75] | [GRACE: Graph-Grounded Reflective Agent Copilot Engine for Expert-in-the-Loop Knowledge Expansion](https://arxiv.org/abs/2609.04442) | 提出GRACE框架，将LLM响应解构为原子论断，通过加权二部图与可信知识先验锚定，利用加权中心性分析将论断分类为有依据、被反驳或边界三类，并借助注意力回报率（RoA）目标高效分配专家审查资源，从而在识别幻觉的同时发现模型知识前沿的新颖论断。 |
| [^76] | [What Attention Recalls and Recurrence Controls in Hybrid Language Models](https://arxiv.org/abs/2609.04434) | 混合语言模型中两条通道功能明确分工：注意力通道专责上下文精确检索，循环状态通道则控制输出语言和角色风格。 |
| [^77] | [A Systematic Evaluation of Cross-Lingual Consistency Enhancement Methods in Multilingual Language Models](https://arxiv.org/abs/2609.04409) | 本文对多语言模型中的跨语言一致性增强方法进行了统一的系统性评估，发现后训练方法（尤其是直接分布对齐）总体更可靠且能稳定提升一致性，而跨域迁移仅在源与目标任务输出格式相似时才有效。 |
| [^78] | [The Anatomy of an ASR Hallucination](https://arxiv.org/abs/2609.04404) | 该研究揭示了ASR幻觉源于“接地失败”，发现编码器最后阶段是关键边界——绕过该阶段会导致输出发散而非流畅的虚构文本。 |
| [^79] | [Evaluation of Phonetic Encoding Algorithms on Transcription Datasets](https://arxiv.org/abs/2609.04391) | 提出了一种基于Hüllermeier-Rifqi指数的新型评估方案，通过计算语音编码与IPA真实转录之间的成对相似度差异，并结合碰撞率评估多种语音编码算法在多语言转录数据集上的性能。 |
| [^80] | [You Really Didn't Get That? Benchmarking Social Pragmatic Inference for Indirect and Playful Chinese Online Comments](https://arxiv.org/abs/2609.04384) | 本文基于超过20万条真实中文社交媒体互动记录，构建了包含4,735个人工验证诊断条目的语用推理基准，用于评估大语言模型能否在上下文中正确解读间接、俏皮评论的社交含义，结果显示最强模型准确率仅81.42%、八个模型平均68.70%，该任务对当前模型仍具很大挑战性。 |
| [^81] | [VERGE: Verification-Enhanced Refinement for Grounded Extraction of Early-Onset Colorectal Cancer Symptoms in Clinical Notes](https://arxiv.org/abs/2609.04366) | VERGE是一种结合检索增强生成与有界验证-精炼循环的智能体工作流，能从临床自由文本笔记中可靠提取早发性结直肠癌的六种红旗症状及家族史风险，并将无法解决的论断自动升级为人工审查。 |
| [^82] | [Knowing When Not to Answer: Pseudo-Ensembles for Abstention in Music Audio-Language Models](https://arxiv.org/abs/2609.04362) | 提出通过打乱选项顺序等不会改变正确答案的输入扰动，从单个预训练音乐音频-语言模型构建伪集成，从而获得置信度估计，使模型在不确定时能够弃权而非猜测。 |
| [^83] | [Adapting from Downturns: Prediction of Long-Term Conversational-Skill Development in Mental-Health Crisis Counselors](https://arxiv.org/abs/2609.04350) | 本文提出了在辅导员职业生涯早期预测其长期对话技能能否提升的新任务，发现辅导员如何从对话低谷中适应和恢复最能预示其长期改进潜力，从而帮助优先支持最需要帮助的辅导员。 |
| [^84] | [SharedSAE: One Feature Dictionary Across Language Models](https://arxiv.org/abs/2609.04344) | SharedSAE通过共享字典结合各模型专属的编码器-解码器对，用单一稀疏自编码器即可替代多个语言模型的专用SAE，在保留96.6%解释方差的同时实现跨模型的统一特征解释与迁移。 |
| [^85] | [A Removal Based Approach to Improve LLM Faithfulness at Test-Time](https://arxiv.org/abs/2609.04343) | 提出了一种基于删除的测试时方法，直接针对大语言模型解释中此前被忽视的不完整性问题，无需访问模型权重或大量计算资源即可提升解释的忠实性。 |
| [^86] | [MedProb: Probing Internal Representations of Vision-Language Models for Medical Question Answering](https://arxiv.org/abs/2609.04336) | MedProb是一个轻量级探测框架，可直接从冻结的视觉语言模型内部表征中预测医学视觉问答答案，性能超越提示方法和医学专用VLM，并揭示了小型模型含有比生成式评估所显示的更丰富的医学问答信号，同时避免了自由文本生成中的答案位置偏差。 |
| [^87] | [Abstraction Agent](https://arxiv.org/abs/2609.04303) | 提出了一种基于大语言模型的零样本“抽象智能体”流水线，仅凭自然语言博弈描述即可自动发现策略特征并构建不完全信息博弈的信息抽象，无需博弈特定评估器、训练数据或博弈树遍历。 |
| [^88] | [Harbor Adapters and Harbor-Index: Infrastructure and a Curated Meta-Dataset for Large-Scale Agentic Evaluation](https://arxiv.org/abs/2609.04298) | 本文提出了 Harbor Adapters 统一评估基础设施，将 80 多个智能体基准测试移植为可评估任意智能体的形式，并据此对 8 个模型进行大规模评估，同时推出经 AI 与人工双重审核筛选的包含 82 个高质量难题的精选元数据集 Harbor-Index。 |
| [^89] | [Evidence Integration in Large Language Models](https://arxiv.org/abs/2609.04290) | 本文提出了一个关于大语言模型证据整合机制的分布理论，并通过千万级规模实验验证了三个预测：接收者眼中更可能的候选答案更具说服力、模型更易整合自身特有错误而非外来错误、以及相同证据可能改善弱模型却损害强模型。 |
| [^90] | [Memory as transformation: LETHE, a self-referential gan-inspired architecture](https://arxiv.org/abs/2609.04289) | LETHE是一个受生成对抗网络启发的自指涉声音系统，通过判别器与随机扰动优化器的交互驱动音频混合矩阵参数在无外部数据或监督的封闭环境中自主演化。 |
| [^91] | [EVOHARNESSBENCH: Can Your Agents Keep Pace with an Evolving Harness?](https://arxiv.org/abs/2609.04280) | 提出了EVOHARNESSBENCH基准，首次将非平稳性从任务流转移到智能体运行框架本身（工具、技能、智能体）的持续演进上，用于评估智能体在框架不断变化环境下的适应与保留能力。 |
| [^92] | [Evaluating Large Language Models for Forced Outage Risk Prediction: Benefits and Comparison to Machine Learning](https://arxiv.org/abs/2609.04272) | 本研究首次将大语言模型以零样本方式应用于电网天气相关停电风险预测，发现其精度虽略逊于监督机器学习模型，但在可操作推理和地理可扩展性方面具有独特优势，两者结合可能是最佳实践。 |
| [^93] | [Reviewer Capability Governs Rejection Targeting, Not Repair Skill: Evidence from LLM Execute-Review-Revise Pipelines](https://arxiv.org/abs/2609.04270) | 在LLM执行-评审-修订流水线中，评审者能力决定的是拒绝定向的准确性而非修复技能——跨家族的中档评审者可将准确率从52%提升至64%且零答案损害，而同模型自我审查虽错误检测率最高却收益不显著。 |
| [^94] | [Quality Recovery for Quantized KV Caches via Low-Rank Attention Adaptation](https://arxiv.org/abs/2609.04263) | 该论文提出一种在量化器固定的情况下，将浮点缓存模型行为蒸馏到低秩Q/K/V投影适配器中的方法，能大幅恢复量化KV缓存造成的模型质量损失。 |
| [^95] | [Automatic Speech Recognition for Multilingual Oral History Research](https://arxiv.org/abs/2609.04232) | 该论文评估了Whisper自动语音识别工具在新西兰粤语口述历史这一语码转换语境下的转录效果，最佳模型配置的词错误率为12.10，且仅需人工转录约1%的时间，为社区主导的语言保护与振兴工作提供了高效的初步转录方案。 |
| [^96] | [How Much Does Corpus Choice Change Dependency-Distance Estimates?](https://arxiv.org/abs/2609.04223) | 该研究通过比较38个同语言树库对发现，语料库的选择会显著影响依存距离估计——跨树库一致性仅为中等、近40%的语言排序会因替换树库而逆转，但所有树库均一致支持依存长度最小化趋势，表明平均依存距离更宜被视为受语料库条件制约的复合指标而非语言本身的固有属性。 |
| [^97] | [GEPARD - Generative, Prosody-aware, Autoregressive text-to-speech model for Realtime Dialogue](https://arxiv.org/abs/2609.04222) | GEPARD是一个基于标准LLM骨干网络、可在不修改vLLM推理引擎计算内核的情况下流式运行的自回归文本转语音模型，通过将零样本语音克隆等辅助机制移出解码循环，实现了真正的实时语音对话生成。 |
| [^98] | [Auditing Bias and Safety in Voice AI Customer Care](https://arxiv.org/abs/2609.04206) | 本文提出了一个验证门控的审计框架，用于检测语音AI客户服务系统中因来电者口音、情感等呈现线索引发的偏见与安全问题，能够捕捉最终拒绝发生之前以额外服务负担形式出现的不公平伤害。 |
| [^99] | [Sequential Beats Joint: On the Interplay between On-Policy Distillation and RLVR](https://arxiv.org/abs/2609.04108) | 先蒸馏后强化学习的两阶段训练方案在推理任务上持续优于纯OPD、纯RLVR及所有联合优化方法，因为OPD先扩大学生对教师解的覆盖范围、RL再在其内锐化，而联合训练会导致两种信号相互干扰。 |
| [^100] | [Editable Visual Design](https://arxiv.org/abs/2609.04034) | 该论文提出“可编辑的视觉设计”新范式，以编码智能体为核心，将VLM作为“创意大脑”进行需求理解与审美判断，将图像生成模型作为按需的“视觉世界模拟器”合成独立资产，并通过“先想象、后行动”的闭环工作流编写原生HTML/CSS，实现支持图层级精确后编辑的视觉设计。 |
| [^101] | [Fixed Suffix Dependency Ratio: Quantifying the Dual-Track Mechanism of Gender Assignment in Latvian Loanwords](https://arxiv.org/abs/2609.03930) | 本研究提出固定后缀依赖比率（FSDR）这一量化指标，揭示了拉脱维亚语英语外来词性属分配的双轨机制——阴性外来词显著依赖固定派生后缀，而阳性外来词集中于自由选择区域。 |
| [^102] | [VisCAD: A Foundation Model Suite with Multimodal Industrial CAD Intelligence](https://arxiv.org/abs/2609.03811) | VisCAD是一个面向工业CAD的基础模型套件，其核心270亿参数模型VisCAD-M1能够将渲染图、文本、2D图纸和真实照片等多种输入转换为可执行的CAD程序，在保证广泛泛化能力的同时具备强大的专业CAD设计与装配生成能力。 |
| [^103] | [IndicSafeEval: Safety Robustness of Large Language Models under Multilingual Persuasive Jailbreak Attacks](https://arxiv.org/abs/2609.03781) | 该论文提出了IndicSafeEval框架，通过四种印度语言、十个安全类别和六种说服策略构建7,200条对抗性提示，系统评估并揭示了大语言模型在面对多语言说服性越狱攻击时安全表现存在显著差异。 |
| [^104] | [RealCADBench: Benchmarking Parametric CAD Modeling from Industrial Design Intents](https://arxiv.org/abs/2609.03773) | 提出了RealCADBench基准测试，基于19个工厂自动化类别的真实工业设计意图，通过文本、图纸、图片等多种输入模态和可执行性、IoU、视觉语义一致性等综合评估指标，系统性地评估从设计意图到程序化参数CAD建模的能力。 |
| [^105] | [Counterfactual Fairness Audits of Multi-Step Clinical LLM Agents Require a Measured Per-Action Instability Floor](https://arxiv.org/abs/2609.03221) | 临床LLM智能体在完全相同输入下本身就存在显著的动作不稳定性（约8.7%），因此反事实公平性审计必须先测量这一“每动作不稳定性底线”，否则任何检测到的人口统计学差异都无法解释。 |
| [^106] | [Post-Training Language Models for Gold-Medal Performance in Coding Competitions](https://arxiv.org/abs/2609.02849) | 该研究通过结合大规模题目筛选、监督微调、强化学习以及反馈驱动的测试时计算策略 GenCorrect，使语言模型在 IOI 2025 编程竞赛中取得了超越金牌分数线（438.3 分）的成绩（Nano-CC 达 468 分，Ultra-CC 达 502 分）。 |
| [^107] | [From Tokens to Semantics: Leveraging Complementary Signals for Hallucination Detection in Black-Box LLMs](https://arxiv.org/abs/2609.02679) | 该论文针对无参考文档的黑盒大语言模型，提出联合利用语义熵与词元级不确定性这两种互补信号（包括TopK聚合、CoCoA混合方法及Gated等监督方法）来更准确地检测幻觉。 |
| [^108] | [Enoki: Efficient Multi-Level Hallucination Detection](https://arxiv.org/abs/2609.00581) | Enoki提出了一种基于开放信息抽取的多层级幻觉检测框架，通过抽取文本锚定的关系事实并进行验证，无需额外的声明-片段对齐即可同时实现声明级验证和片段级定位，并支持LLM、编码器和规则三种抽取方式以平衡准确性与推理成本。 |
| [^109] | [Synthetic Worlds for Temporal Evaluation and Knowledge Updating in LLMs](https://arxiv.org/abs/2609.00184) | 该论文提出了一个模拟驱动的合成框架，通过虚构未来世界的 ParallelEvents 基准避免评估污染，并利用 Synapse 训练框架（结合中期训练与指令微调）实现大语言模型的可扩展知识更新，性能比现有方法提升 14.23%。 |
| [^110] | [Token-Efficient Data Reasoning Agents via Adaptive Structuring of Unstructured Data](https://arxiv.org/abs/2608.31082) | 提出“智能体式数据破解”方法，通过在查询到来时自适应地将非结构化数据结构化，使LLM智能体能以远低于现有方法的令牌成本对海量非结构化数据进行复杂问题推理。 |
| [^111] | [When Linguistic and Internal Confidence Diverge in Large Language Models](https://arxiv.org/abs/2608.28382) | 该研究通过跨8个分类任务、2个生成任务和30个模型的大规模实验，揭示了大语言模型口头表达的语言置信度与其内部置信度经常不一致，且指令微调模型置信度更高但校准更差、态度提示会夸大置信度而无法提升准确性。 |
| [^112] | [INSPIRE: An Internalize-Then-Improve Approach for Example-Driven Mathematical Reasoning](https://arxiv.org/abs/2608.27501) | 提出INSPIRE方法，采用先内化参考示例、再逐步改进的策略，增强大语言模型基于示例的数学推理能力（如构造反例检验定理边界），超越仅优化最终答案正确性的传统方法。 |
| [^113] | [Semantic Overlays: Mitigating Prompt Injection with Annotations Beyond Tokens and Steering Vectors](https://arxiv.org/abs/2608.23873) | 该论文提出了一种名为“语义覆盖层”的新技术，通过向模型输入添加非文本通道来缓解提示注入攻击，利用小型学习的适配器在冻结模型的残差流中创建带外注释，从而增强模型对片段身份的理解。 |
| [^114] | [Don' t Box Me In: Dynamic Cultural Adaptation and Cognitive Tracking for Social Understanding](https://arxiv.org/abs/2608.22411) | 本文提出一种无需训练的框架DyCAC，通过将文化偏好建模为动态混合参考并持续追踪认知，使大型语言模型能灵活适应多元文化社交情境，克服了静态文化建模的局限。 |
| [^115] | [Compiler-Guided Adaptive Proof Search with Cross-Model Synergy on Context-Dependent Theorem Proving](https://arxiv.org/abs/2608.18084) | 提出了一种编译器引导的证明搜索框架，通过双模型生成和编译器接地比较，在真实Lean 4项目中实现了更高的证明通过率和更好的效率。 |
| [^116] | [CT-$\Delta$Bench: A Benchmark for Longitudinal 3D Medical Imaging Difference Reporting with Vision-Language Models](https://arxiv.org/abs/2608.11534) | 本文提出CT-$\Delta$Bench基准，专门用于评估视觉语言模型在纵向三维医学影像中生成时间差异报告的能力，并通过患者级划分确保评估可靠性。 |
| [^117] | [Search-G1: Grounded Search Agents via Representation-Based Intrinsic Rewards](https://arxiv.org/abs/2608.07531) | Search-G1 通过基于表示的内在奖励，利用干预校准读数区分必要检索与冗余搜索，实现无需昂贵注释的落地搜索代理。 |
| [^118] | [Latent Fact-Checking: Detecting Misinformation through Activation Engineering](https://arxiv.org/abs/2608.06417) | 本文提出了一种基于激活工程的错误信息检测框架，利用语言模型表示空间中的几何方向来分类声明真伪，无需微调或外部知识。 |
| [^119] | [ConWriter: Transition-Constrained Stateful Long-Form Story Generation with Lightweight Neuro-Symbolic Consistency Control](https://arxiv.org/abs/2608.05169) | ConWriter是一个无需训练的长篇故事生成框架，通过场景级增量写作、维护演进的故事状态、检查叙事转换约束，并利用不确定性感知的风险信号进行优先验证和局部修复，从而在生成过程中（而非事后）实现一致性控制，避免局部错误向后续场景传播。 |
| [^120] | [Revisiting Lossy Verification in Speculative Decoding: Mechanisms, Trade-offs, and Failure Modes](https://arxiv.org/abs/2607.26627) | 本文系统分析了推测解码中有损验证方法所诱导的解码分布，将众多表面不同的方法统一为基于截断的验证与协作式验证两类，并构建诊断评估框架揭示其效率与质量之间的权衡及失效模式。 |
| [^121] | [Not All LLM Reasoning is Visible in the Chain-of-Thought](https://arxiv.org/abs/2607.22925) | 前沿大语言模型能利用语义无关的填充token进行思维链之外的“不可见推理”来提升任务表现，甚至可以完成思维链监控完全无法察觉的隐藏目标，这对基于CoT监控的AI安全方案构成重大风险。 |
| [^122] | [From Plausible to Actionable: A Position on LLM Self-Explanations](https://arxiv.org/abs/2607.15957) | 本立场论文指出大语言模型的自我解释虽高度合理但忠实性存疑，主张评估标准应从合理性与忠实性扩展到可操作性，并为此提供了实用的评估指南。 |
| [^123] | [EvoCUA-1.5: Online Reinforcement Learning for Multi-turn Computer-Use Agents](https://arxiv.org/abs/2607.09773) | EvoCUA-1.5 将计算机使用智能体从离线经验学习扩展到在线强化学习，并提出步级策略优化（STEPO）方法，以解决多轮交互中上下文管理观察、稀疏终端奖励、可变长度轨迹和慢速环境反馈等挑战。 |
| [^124] | [Estimating Uncertainty from Reasoning: A Large-Scale Study of Multi- and Crosslingual MCQA Performance in LLMs](https://arxiv.org/abs/2607.06327) | 该研究首次在22种语言上大规模评估了不确定性估计方法，发现让模型用英语进行长篇推理可显著提升低资源语言问答的不确定性估计性能，表明可靠性瓶颈在于生成而非理解。 |
| [^125] | [Robust Text Watermarking for Large Language Models via Dual Semantic Embeddings](https://arxiv.org/abs/2606.31602) | 提出双重嵌入水印（DEW）方案，结合上下文与词元级嵌入的信号处理技术，为大型语言模型生成对改写和翻译攻击具有最先进鲁棒性的文本水印，同时保持低计算开销和高质量的文本。 |
| [^126] | [GPTNT: Benchmarking Real-Time Collaboration Between Multimodal Agents on Keep Talking And Nobody Explodes](https://arxiv.org/abs/2606.28514) | 该论文提出了GPTNT基准，首次在《保持通话，无人爆炸》游戏中将时间压力、信息不对称和不完美沟通这三种协作条件结合在一起，评估两个多模态智能体在实时倒计时下通过沟通协作拆弹的能力。 |
| [^127] | [Faithful by Construction: Claim-Anchored Attribution for Multi-Document Summarization](https://arxiv.org/abs/2606.23989) | 提出CAMS框架，将声明级归因嵌入“提取—选择—改写”流程，使多文档摘要中的每句话都能锚定到经过验证、可溯源的源文本片段，从而在构造层面保证摘要的忠实性。 |
| [^128] | [CacheWeaver: Cache-Aware Evidence Ordering for Efficient Grounded RAG Inference](https://arxiv.org/abs/2606.19667) | CacheWeaver是一种轻量级的提示层缓存感知证据排序方法，通过维护前缀树并用贪心算法将重叠证据重排为可复用的前缀，在不改变服务引擎和检索证据集的前提下，将有据RAG推理的中位首token时间（TTFT）降低约20-33%。 |
| [^129] | [Detect, Remask, Repair: Diffusion Editing for Faithful Summarization of Evolving Contexts](https://arxiv.org/abs/2606.12807) | 提出基于掩码扩散语言模型的DETECT-REMASK-REPAIR框架，通过检测、重新掩码和局部修复摘要中的过时片段，在保留已受支持内容的同时实现高效可控的演化上下文摘要更新，并发布了StreamSum基准数据集。 |
| [^130] | [KCSAT-ML: Probing Reasoning Models with Nationwide-Cohort Human Difficulty](https://arxiv.org/abs/2606.10403) | 该论文提出了KCSAT-ML基准——十年韩国高考数学题共664道、核心339题附带数十万考生的官方逐题错误率，并配合与得分正交的“难度对齐推理增益”（DRG）指标，揭示了模型在人类认为困难的题目上准确率崩溃、测试时扩展的准确率收益呈非单调曲线等关键发现。 |
| [^131] | [From Architecture to Output: Structural Origins of Hallucination in Large Language Models and the Amplifying Role of Data](https://arxiv.org/abs/2606.07537) | 该论文提出了一个仅需采样访问权限的幻觉归因框架，通过针对前缀、上下文和频率竞争的三次有序干预，将大语言模型的单个幻觉追溯归因到自注意力联想检索、最大似然预训练目标或暴露偏差下的自回归承诺等具体组件，并提出五个可证伪的预测。 |
| [^132] | [MineExplorer: Evaluating Open-World Exploration of MLLM Agents in Minecraft](https://arxiv.org/abs/2605.30931) | 本文提出了MineExplorer基准测试，通过筛选通用原子任务、组合隐式多跳任务及多智能体合成流程，系统评估MLLM智能体在《我的世界》中的开放世界探索能力。 |
| [^133] | [Robust and Efficient Guardrails with Latent Reasoning](https://arxiv.org/abs/2605.29068) | COLAGUARD通过分阶段训练将多步安全推理压缩进连续潜在空间，推理时直接传播隐状态，在宏F1上超越Llama Guard 3达8.24分，并在达到显式推理基线GuardReasoner同等性能的同时实现12.9倍加速和22.4倍的token开销降低。 |
| [^134] | [Trait-Aware Policy Optimization for Autoregressive Multi-Trait Essay Scoring](https://arxiv.org/abs/2605.25731) | 提出特质感知策略优化（TAPO）后训练框架，通过在样本和特质两个维度分解奖励并结合包含原始提示与特质描述的增强提示，在多个骨干模型上持续优于监督微调和标量奖励优化基线，显著提升自回归模型的多维度作文评分性能。 |
| [^135] | [Towards Generalization of Block Attention via Automatic Segmentation and Block Distillation](https://arxiv.org/abs/2605.15913) | 该论文构建了包含3万余实例的语义分割数据集SemanticSeg并训练轻量级自动分割器，同时提出块蒸馏训练框架，解决了块注意力中文本难以有效分块及微调低效易损性能的问题，推动块注意力在RAG等长上下文场景中的泛化应用。 |
| [^136] | [Scientific Domain Knowledge Improves Vision-Language Fundus Models](https://arxiv.org/abs/2605.02720) | 该研究构建了高领域密度的PubMed-Ophtha数据集，并通过严格对照实验证明，使用领域特定科学文献训练的视觉-语言眼底模型在110项临床任务中的表现优于使用固定模板或医学报告训练的模型。 |
| [^137] | [Role-Aware Artificial Intelligence Across Augmentation and Automation in Human-Machine Symbiosis](https://arxiv.org/abs/2605.00440) | 该论文提出在人机共生中应关注AI的角色感知问题，指出AI在“自动化替代”与“能力增强”之间的功能角色虽在提示词中被规定，但脱离对话上下文后便难以追溯。 |
| [^138] | [Unified Deployment-Aware Evaluation of Open Reasoning Language Models](https://arxiv.org/abs/2604.07035) | 该论文对七种开放推理语言模型在四个基准上进行了统一且面向实际部署的全面评估，不仅比较准确率还涵盖延迟、显存占用、提示敏感性等指标，发现Gemma-4-26B-A4B配合零样本提示取得最高加权分数0.794。 |
| [^139] | [Cross-Preference Learning for Sentence-Level and Context-Aware Machine Translation](https://arxiv.org/abs/2603.25183) | 提出交叉偏好学习方法，通过在偏好优化目标中整合条件内偏好与跨条件偏好，显式捕捉句子级与上下文感知机器翻译的互补优势，使模型既能有效利用有信息量的上下文，又能对无信息量的上下文保持稳健。 |
| [^140] | [YOLO with Kolmogorov-Arnold networks and vision-language foundation models for interpretable object detection with trustworthy multimodal AI in computer vision perception](https://arxiv.org/abs/2603.23037) | 该论文提出用Kolmogorov-Arnold网络作为可解释的事后代理模型，基于七个几何与语义特征评估YOLOv10检测结果的置信度可信性，并结合BLIP视觉-语言基础模型生成描述，实现计算机视觉感知中透明、可信的目标检测。 |
| [^141] | [PROMPT2BOX:Improving LLM Weakness Discovery and Specificity Estimation by Uncovering Entailment Structure among Prompts](https://arxiv.org/abs/2603.21438) | 该论文提出Prompt2Box方法，将提示词嵌入到盒嵌入空间中，以同时捕捉语义相似性和提示词之间的具体性（蕴含）关系，从而实现对大语言模型弱点的细粒度发现与具体性估计。 |
| [^142] | [NOTAI.AI: Explainable Detection of Machine-Generated Text via Curvature and Feature Attribution](https://arxiv.org/abs/2603.05617) | 该论文提出了NotAI.AI，一个将句子级条件概率曲率、神经检测器分数与可解释文体特征结合于XGBoost元分类器的可解释AI生成文本检测系统，通过TreeSHAP特征归因和自然语言解释揭示预测依据，在RAID数据集的类别平衡子集上达到0.9685的F1分数。 |
| [^143] | [Labels have Human Values: Value Calibration of Subjective Tasks](https://arxiv.org/abs/2601.06631) | 提出MC-STL框架，通过将标注聚类为可识别的人类价值集群并学习集群特定嵌入来校准预测，在多种主观任务上持续优于忽略标注潜在价值结构的基线方法。 |
| [^144] | [TeleTables: A Benchmark for Large Language Models in Telecom Table Interpretation](https://arxiv.org/abs/2601.04202) | 该论文提出TeleTables基准（包含2,220张3GPP规范表格和500道人工验证选择题），通过评估20个开源大语言模型揭示了电信表格解读的两大瓶颈：闭卷时领域知识不足导致准确率不超过41%，而提供表格上下文时准确率虽可超90%，但会随推理深度、证据范围和表格结构复杂性增加而系统性下降。 |
| [^145] | [Do Androids Dream of Unseen Puppeteers? Probing for a Conspiracy Tendencies in Large Language Models](https://arxiv.org/abs/2511.03699) | 本研究通过标准化心理测量调查发现，大型语言模型表现出一定的阴谋论倾向，且可通过提示策略被引导采纳阴谋论观点。 |
| [^146] | [GSM8K-V: Can Vision Language Models Solve Grade School Math Word Problems in Visual Contexts](https://arxiv.org/abs/2509.25160) | 该论文提出 GSM8K-V 基准，将文本数学题转化为多图像序列，发现视觉语言模型在视觉数学推理上存在显著模态差距（最佳模型仅 59%，远低于人类 91%）。 |
| [^147] | [Exploring Solution Divergence and Its Effect on Large Language Model Problem Solving](https://arxiv.org/abs/2509.22480) | 本文提出将“解分歧”作为新指标，发现其与大语言模型问题求解能力正相关，并能同时提升监督微调和强化学习的训练效果。 |
| [^148] | [QoNext: Towards Next-generation QoE for Foundation Models](https://arxiv.org/abs/2509.21889) | QoNext首次将网络与多媒体领域的体验质量原则引入人机交互评估，通过识别生成速度、延迟等动态体验因素构建了QoNext数据库，并训练出可直接从可测量因素预测用户体验的神经模型。 |
| [^149] | [ConfRAG: Confidence-Guided Retrieval-Augmenting Generation](https://arxiv.org/abs/2506.07309) | 提出ConfQA微调策略，通过训练模型在不确定时回答“我不确定”将LLM幻觉率从20-40%降至5%以下，并在此基础上构建ConfRAG，仅在模型置信度低时才触发检索增强生成，从而同时减少幻觉并降低检索计算成本。 |
| [^150] | [Harnessing the Reasoning Economy: A Survey of Efficient Reasoning for Large Language Models](https://arxiv.org/abs/2503.24377) | 本综述系统性地提出了“推理经济”概念，用以刻画大语言模型在性能收益与计算成本之间的权衡，并从后训练和测试时推理两个阶段分析了推理低效的成因、不同推理模式的行为特征以及实现高效推理的潜在方向。 |
| [^151] | [Multilingual Models for Check-Worthy Social Media Posts Detection](https://arxiv.org/abs/2408.06737) | 本研究开发了能够同时处理英语和多种低资源语言的多标签多语言分类模型，用于检测社交媒体中包含可验证事实声明和有害声明的帖子。 |

# 详细

[^1]: WearableQA：一个基于真实世界可穿戴数据的健康推理基准

    WearableQA: A Benchmark for Health Reasoning over Real-World Wearable Data

    [https://arxiv.org/abs/2609.05405](https://arxiv.org/abs/2609.05405)

    该论文提出了WearableQA基准，基于200名真实用户长达500天的可穿戴数据构建了4,084道多项选择题，并通过数据/健康推理和单信号/跨信号推理两个维度的16种问题类型，评估AI系统对真实世界可穿戴数据的健康推理能力。

    

    可穿戴传感技术的最新进展使对生理和行为信号的连续监测成为可能，然而现有基准测试很少评估AI系统能否对真实用户的纵向可穿戴记录进行推理。我们提出了WearableQA，这是一个包含4,084道10选项多项选择题的基准，这些题目基于200名真实用户的可穿戴时间序列数据、血液生物标志物和人口统计信息构建，每位用户拥有长达500天的每日测量数据。WearableQA保留了真实的可穿戴数据分布，包括设备噪声和个体间差异。为了评估不同的推理能力，我们引入了沿两个互补维度组织的16种问题类型：数据推理与健康推理，用于区分对纵向测量的计算与对生理意义的解读；单信号推理与跨信号推理，用于区分对单一信号的推理与对多种信号的整合。

    arXiv:2609.05405v1 Announce Type: new  Abstract: Recent advances in wearable sensing enable continuous monitoring of physiological and behavioral signals, yet existing benchmarks rarely evaluate whether AI systems can reason over a real user's longitudinal wearable record. We introduce WearableQA, a benchmark comprising 4,084 10-option multiple-choice questions constructed from the wearable time series, blood biomarkers, and demographics of 200 real users, each with up to 500 days of daily measurements. WearableQA preserves authentic wearable distributions that include device noise and inter-individual variability. To evaluate distinct reasoning capabilities, we introduce 16 question types organized along two complementary axes: data versus health reasoning, which distinguishes computation over longitudinal measurements from physiological interpretation; and single- versus cross-signal reasoning, which separates reasoning about individual signals from the integration of multiple signal
    
[^2]: 相同轨迹，矛盾奖励（ROBORMBENCH）：视觉语言奖励模型中的释义脆弱性

    Same Trajectory, Contradictory Rewards (ROBORMBENCH): Paraphrase Fragility in Vision Language Reward Models

    [https://arxiv.org/abs/2609.05401](https://arxiv.org/abs/2609.05401)

    现有视觉语言奖励模型缺乏释义不变性——仅改写指令就能大幅改变同一机器人轨迹的奖励评分，甚至翻转成败判定，新基准ROBORMBENCH证实这种脆弱性普遍存在，且难以靠扩大模型规模或显式推理来缓解。

    

    视觉语言模型（VLM）正越来越多地被用作机器人学习的奖励函数，但这一角色要求释义不变性：在语义等价的目标描述下，相同的轨迹应获得相同的奖励。我们证明当前的VLM奖励模型经常违反这一性质。仅仅对指令进行同义改写就可能显著改变预测的进度分数，甚至能让完全相同的机器人行为在“失败”与“成功”之间翻转。为了衡量这一失效模式，我们提出了ROBORMBENCH基准，其中包含2,390条真实机器人轨迹、真实进度标签以及21,673条经过验证的释义改写，涵盖词汇、句法和动作-目标层面的改写。在专有和开源VLM中，由释义引起的不稳定性广泛且严重，在更偏离原意的改写下会加剧，且无法通过扩大模型规模或显式推理来可靠地降低。使用基于轨迹监督训练的专用奖励模型则

    arXiv:2609.05401v1 Announce Type: cross  Abstract: Vision-language models are increasingly used as reward functions for robotic learning, but this role requires paraphrase invariance: the same trajectory should receive the same reward under semantically equivalent goal descriptions. We show that current VLM reward models often violate this property. Paraphrasing the instruction alone can substantially change predicted progress scores, and can even flip identical robot behavior between failure and success. To measure this failure mode, we introduce ROBORMBENCH, a benchmark with 2,390 real-robot trajectories, ground-truth progress labels, and 21,673 verified paraphrases spanning lexical, syntactic, and action-goal rewrites. Across proprietary and open-source VLMs, paraphrase-induced instability is widespread and severe, grows under more divergent rewrites, and is not reliably reduced by scale or explicit reasoning. Dedicated reward models trained with trajectory-grounded supervision are 
    
[^3]: 韩国开放公共API上的多步工具调用：一个基准与数据合成配方

    Multi-Step Tool-Calling over Korean Open Public APIs: A Benchmark and a Data-Synthesis Recipe

    [https://arxiv.org/abs/2609.05395](https://arxiv.org/abs/2609.05395)

    该论文提出韩国开放公共API基准KOPA-Bench（145个真实任务）和基于实时执行验证的数据合成方法EDGE，通过合成可执行的多步工具调用轨迹并用GRPO微调，使9B开源模型性能几乎追平同系列27B模型。

    

    arXiv:2609.05395v1 公告类型：新 摘要：数据主权法规日益要求公共机构部署开源、本地化的LLM智能体，这些智能体需要在实时政府API上链式执行多个工具调用。然而，开源模型在这种多步场景中表现持续不佳，且目前没有基准能衡量这一差距。我们推出了韩国开放公共API基准（KOPA-Bench），包含145个真实世界任务。为缩小这一差距，我们提出了EDGE——一个由实时执行驱动的、基于执行结果的工具调用数据合成动态图。EDGE构建了一个展示每个工具输出如何馈入另一个工具输入的图结构，仅保留对实时API实际调用时验证成功的链接，并遍历这些经过验证的链接来合成可执行的多步轨迹。通过在所得数据集上使用GRPO进行微调，我们的9B模型几乎追平了同系列未经调优的27B模型，不仅在KOPA-Bench上，在BFCL基准上也取得了显著提升。

    arXiv:2609.05395v1 Announce Type: new  Abstract: Data-sovereignty regulations increasingly require public institutions to deploy open-source, on-premise LLM agents that chain multiple tool-calls across live government APIs. However, open-source models consistently underperform in this multi-step setting, and no existing benchmark measures the gap. We introduce the Korean Open Public API Benchmark (KOPA-Bench), comprising 145 real-world tasks. To close this gap, we present EDGE, an Execution-grounded Dynamic Graph for tool-calling data synthEsis driven by live execution. EDGE builds a graph of how each tool's output can feed another's input, keeps only the links that succeed when actually called against the live APIs, and traverses these verified links to synthesize executable multi-step trajectories. Fine-tuned via GRPO on the resulting dataset, our 9B model nearly matches the untuned 27B model from the same family, improving substantially not only on KOPA-Bench but also on the BFCL be
    
[^4]: 你的智能体记忆能在模型升级中幸存吗？一项关于记忆可移植性的对照研究

    Does Your Agent's Memory Survive a Model Upgrade? A Controlled Study of Memory Portability

    [https://arxiv.org/abs/2609.05339](https://arxiv.org/abs/2609.05339)

    该对照研究发现智能体记忆的可移植性取决于存储格式：固定模式知识图在模型升级后准确率几乎不变，而由模型压缩生成的自然语言笔记与原模型高度耦合，迁移后性能会大幅不对称波动。

    

    模型升级是常规操作，而记忆迁移却不是。即使智能体保留相同的记忆存储，仍可能发生遗忘：新模型可能以不同方式解读旧笔记，混合的嵌入版本可能破坏检索，且在缺乏原始证据时修复可能失败。我们在保留相同历史记录的前提下，比较了四种记忆存储方式：原文完整保留用于长上下文阅读（LC-RAW）、分块用于检索增强生成（RAG）、由模型压缩为自然语言笔记（NOTES）、或规范化为固定模式知识图（KG-fixed）。该研究使用了48个带有随机化答案代码的合成历史记录、精确评分机制，以及两个参数量低于100亿的开源权重模型。我们的测量结果表明，固定模式结构可以可靠地迁移——在写入器（模型）更换后，KG-fixed的准确率仅变化 $+0.0004 \pm 0.0020$。相反，压缩的NOTES表现出高度的模型耦合性，其准确率以不对称的方式变化，偏移量达 $+9.91$ 或 $-13

    arXiv:2609.05339v1 Announce Type: new  Abstract: Model upgrades are routine; memory migrations are not. An agent can keep the same memory store and still forget: a new model may interpret old notes differently, mixed embedding versions may break retrieval, and repair may fail without the original evidence. We compare memory as the same history is preserved verbatim for long-context reading (LC-RAW), divided into chunks for retrieval-augmented generation (RAG), compressed by a model into natural-language notes (NOTES), or normalized into a fixed-schema knowledge graph (KG-fixed). The study uses 48 synthetic histories with randomized answer codes, exact scoring, and two open-weight models with sub 10 billion parameters.   Our measurements show that fixed-schema structures transfer reliably, with KG-fixed accuracy changing by only $+0.0004 \pm 0.0020$ following a writer swap. Conversely, compressed NOTES exhibit high model coupling, with accuracy shifting asymmetrically by $+9.91$ or $-13
    
[^5]: 用于测量Transformer语言模型中上下文个体化能力的工具包技术手册

    Technical Manual for a Toolkit for Measuring Contextual Individuation in Transformer Language Models

    [https://arxiv.org/abs/2609.05333](https://arxiv.org/abs/2609.05333)

    本文提出一个基于“桥接词形”构念的开源工具包，通过让同一词形在不同主题领域中承载不同词义，来系统测量Transformer语言模型在后续层中是否能依据上下文对相同词形的不同出现进行个体化区分。

    

    arXiv:2609.05333v1 公告类型：新论文 摘要：Transformer语言模型在其嵌入层为词类型分配单一的、与上下文无关的向量，但人们普遍认为它在后续的层中会根据上下文对该词的各个具体出现进行个体化区分。要干净利落地检验这一信念，需要一个构念：在保持词形不变的同时，以受控且带标注的方式改变其上下文和预期词义。本手册记录了一个围绕此类构念构建的开源工具包，我们将该构念称为“桥接词形”：即一个在两个或多个主题领域中重复出现、书写形式完全相同、但在每个领域中具有不同词义的书面词。我们描述并论证了流程的每个阶段：桥接词形及其源领域的声明式规范、从维基百科获取语料库、出现位置定位、逐层表示提取、通过领域成对轮廓系数测量模型表示空间中的分离度，以及配对可视化协议。每个设计选择都被预先呈现。

    arXiv:2609.05333v1 Announce Type: new  Abstract: A transformer language model assigns a single, context-independent vector to a word type at its embedding layer, yet is widely believed to individuate that word's occurrences by context in its later layers. Testing this belief cleanly requires a construct that holds the word form fixed while its context and intended sense vary in a controlled, labeled way. This manual documents an open toolkit built around such a construct, which we call a bridge form: a single written word that recurs, unchanged, across two or more subject domains with a different sense in each. We describe, and justify, every stage of the pipeline: the declarative specification of bridge forms and their source domains, corpus acquisition from Wikipedia, occurrence localization, layer-wise representation extraction, a domain-pairwise silhouette measurement of separation in the model's representation space, and a paired visualization protocol. Each design choice is prese
    
[^6]: 大语言模型在建筑能源系统HVAC运行中的应用：方法、应用与部署准备度的批判性综述

    Large Language Models for HVAC Operations in Building Energy Systems: A Critical Review of Methods, Applications, and Deployment Readiness

    [https://arxiv.org/abs/2609.05314](https://arxiv.org/abs/2609.05314)

    本系统性综述分析了66篇关于大语言模型用于建筑暖通空调运行的研究，发现该领域主要集中于建筑能源建模，但绝大多数研究仍停留在研究阶段，目前尚无任何研究达到可立即产业部署的成熟度。

    

    建筑自动化系统产生了丰富的传感器数据，但仍然洞察匮乏，因为异构的点位命名、缺失的元数据和零散的文档阻碍了其运营使用。本系统性综述对2023年至2026年3月间发表的66篇关于大语言模型（LLM）用于暖通空调（HVAC）运行的同行评审研究进行了分析与编码。每项研究被归入五个应用类别和三个LLM方法类别，并从证据真实性、部署准备度以及LLM与物理HVAC决策之间的责任边界等方面进行评估。该研究语料集中于建筑能源建模（BEM，66篇论文中有32篇），而负荷预测领域的研究仍过于稀疏，无法得出子领域层面的结论。仅有四项研究达到了试点级别的证据，且没有任何研究报告了持续的运营部署。没有研究被归类为当前可立即被行业采用的就绪状态；三项属于近期可部署，63项仅为研究性质。尽管如此，若干……（原文摘要在此处截断）

    arXiv:2609.05314v1 Announce Type: new  Abstract: Building automation systems generate rich sensor data yet remain insight-poor because heterogeneous point naming, missing metadata, and fragmented documentation obstruct their operational use. This systematic review analyses and codes 66 peer-reviewed studies on large language models (LLMs) for HVAC operations published between 2023 and March 2026. Each study is classified across five application families and three LLM method families and assessed for evidence realism, deployment readiness, and the responsibility boundary between the LLM and physical HVAC decisions. The corpus is concentrated in building energy modelling (BEM, 32 of 66 papers), while load forecasting remains too sparse for subfield-level conclusions. Only four studies reach pilot-level evidence, and none reports sustained operational deployment. No study was classified as ready-now for industry adoption; three were near-term and 63 research-only. Nevertheless, several bo
    
[^7]: LexFlip：法律含义保持度量的解离诊断方法

    LexFlip: A Dissociation Diagnostic for Legal Meaning Preservation Metrics

    [https://arxiv.org/abs/2609.05296](https://arxiv.org/abs/2609.05296)

    本文提出LexFlip诊断集，通过373个保持词汇表面形式不变却逆转法律效力的魁北克法语法条最小扰动，揭示了现有嵌入类语义度量几乎无法察觉法律含义变化，暴露了当前法律文本简化评估中“相同对检验”的根本缺陷。

    

    简化后的法律条款是否仍然表达了原文的含义？目前使用的检验方法无法证明这一点：要求相同句对得分最高、无关句对得分最低的检验方式，会使词汇重叠与法律效力一同变动，因此任何词元重叠的单调函数都能同时满足这两项要求。我们的补救方案是“解离”——即表面形式保持不变而法律效力发生变化的项目。我们发布了LexFlip，包含373个针对魁北克省法语成文法的最小扰动，这些扰动在保留0.93词元的同时逆转了法律效力，并配套一个评分框架，可对度量指标、回归器和提示式评判模型进行评测。我们测试的七个嵌入类和BERTScore度量在此类编辑上仅消耗其相同对至无关对得分区间的0.022至0.039，相比之下双向NLI为0.670——而双向NLI恰恰是相同句对检验会淘汰的唯一一类方法。在FrJudge上，相对于实测的人类上限r=0.597，一个简单的长度特征胜过了所有语义度量指标，并且具有最低的（原文在此处截断）

    arXiv:2609.05296v1 Announce Type: new  Abstract: Does a simplified legal clause still say what the original said? The checks in current use cannot establish that it does: requiring an identical pair to score highest and an unrelated pair lowest moves lexical overlap and legal force together, so any monotone function of token overlap satisfies both. Our remedy is a dissociation, an item holding surface form fixed while legal force moves. We release LexFlip, 373 minimal perturbations of Quebec statutory French that reverse legal force while preserving 0.93 of the tokens, with a harness scoring metrics, regressors and prompted judges alike. The seven embedding and BERTScore metrics we test spend only 0.022 to 0.039 of their identical-to-unrelated range on such an edit, against 0.670 for bidirectional NLI, the one family the identical-pair check would disqualify. On FrJudge, against a measured human ceiling of r=0.597, a bare length feature outscores every semantic metric and has the lowes
    
[^8]: 面向快速大规模系统发育推断的自监督词汇表示学习

    Self-Supervised Lexical Representation Learning for Fast, Large-Scale Phylogenetic Inference

    [https://arxiv.org/abs/2609.05262](https://arxiv.org/abs/2609.05262)

    本文提出一种无需同源词标注的全自监督双重对比学习框架，直接从IPA音标词表学习词汇表示，从而快速、低成本地推断大规模全球语言的系统发育树。

    

    计算系统发育学已成为历史语言学中的重要工具，但其在全球尺度上的应用仍受限于两个因素：基于字符的方法所需的人工同源词判断标注极其耗费人力，以及在大规模数据集上进行推断的高昂计算成本。本文提出了一种完全自监督的对比学习框架，可直接从原始的IPA国际音标转写词表中学习词汇表示，无需同源词标注、词表比对或额外的专家输入。该模型采用双重对比目标：词级别的损失函数将语音相似的形式组织到一个连贯的空间中；辅助的语言级别损失函数则促使词汇空间反映语言更广泛的音系特征。从所得的词表示中推导出语言间的成对距离，并据此推断出涵盖3……（原文此处截断）

    arXiv:2609.05262v1 Announce Type: new  Abstract: Computational phylogenetics has become an essential tool in historical linguistics, yet its application at a global scale remains constrained by two factors: the labor-intensive manual annotation of cognacy judgments required for character-based methods and the substantial computational cost of inference on large datasets. This paper introduces a fully self-supervised contrastive learning framework that learns lexical representations directly from raw IPA-transcribed wordlists, without requiring cognacy annotations, alignments, or additional expert input. The model employs a dual contrastive objective: a word-level loss that organizes phonetically similar forms into a coherent space, and an auxiliary language-level loss that encourages the lexical space to reflect broader phonological properties of languages. From the resulting word representations, pairwise language distances are derived and used to infer a global phylogenetic tree of 3
    
[^9]: 一种验证器引导的可解释推理框架：结合黄金锚定QLoRA、任务感知混合专家与组相对RLVR

    A Verifier-Guided Explainable Reasoning Framework with Gold-Anchored QLoRA, Task-Aware Mixture-of-Experts, and Group-Relative RLVR

    [https://arxiv.org/abs/2609.05221](https://arxiv.org/abs/2609.05221)

    该论文提出一种面向教育问答的可解释推理框架，通过黄金锚定QLoRA微调、任务感知符号路由（逻辑问题交由FOL/Z3验证器、物理问题交由公式与单位感知求解器）以及组相对RLVR强化学习，从正确性、一致性和推理深度三个维度提升大模型推理的可验证性与可解释性。

    

    大型语言模型（LLMs）展现出强大的推理能力，但其解释可能仍然不一致、依据薄弱或难以验证。我们提出了一种面向透明教育问答的验证器引导可解释推理框架，该框架结合了黄金锚定QLoRA、任务感知符号路由和组相对RLVR。首先使用以权威答案为锚定的字段加权QLoRA监督对Qwen2.5-3B-Instruct模型进行适配。随后，一个轻量级路由器将逻辑问题分配给FOL/Z3验证器，将物理问题分配给公式与单位感知的符号求解器。验证器反馈进一步用于支持RLVR过程中的候选评估、自我修订和奖励构建。候选响应沿三个互补维度进行评估：P1评估答案正确性，P2评估证据或单位一致性，P3评估推理深度和可解释性。在推理阶段，无需黄金答案的自我一致性机制对多个候选进行聚合……

    arXiv:2609.05221v1 Announce Type: cross  Abstract: Large language models (LLMs) show strong reasoning ability, but their explanations can remain inconsistent, weakly grounded, or difficult to verify. We propose a verifier-guided explainable reasoning framework for transparent educational question answering that combines gold-anchored QLoRA, task-aware symbolic routing, and group-relative RLVR. Qwen2.5-3B-Instruct is first adapted with field-weighted QLoRA supervision anchored to authoritative answers. A lightweight router then assigns logic problems to a FOL/Z3 verifier and physics problems to a formula- and unit aware symbolic solver. Verifier feedback is further used to support candidate evaluation, self-revision, and reward construction during RLVR. Candidate responses are evaluated along three complementary dimensions: P1 for answer correctness, P2 for evidence or unit consistency, and P3 for reasoning depth and explainability. At inference, gold-free self-consistency aggregates mu
    
[^10]: 大语言模型能否预判人们对社会政策的行为反应？以中国灵活就业人员养老保险参保预测为例

    Can Large Language Models Anticipate Behavioral Responses to Social Policies? A Case of Pension Enrollment Prediction among China's Flexible Workers

    [https://arxiv.org/abs/2609.05189](https://arxiv.org/abs/2609.05189)

    本文提出首个面向中国灵活就业人员的领域专用大语言模型 FlexPension-LLM，通过 DKI-RDistill 方法将政策相关线索（如 Probit 边际效应和户籍省份养老规则）注入提示词并进行知识蒸馏，实现养老保险参保行为的精准预测，为社会政策评估提供了低成本的新工具。

    

    评估社会政策变化的影响是政策制定者公认的一大难题。计量经济学方法在推演假设情景时可能不可靠，而实地试点项目成本高昂。本文提出将大语言模型（LLMs）作为由通用模型适配而来的政策评估工具。我们提出了 FlexPension-LLM，这是首个面向中国灵活就业人员分层养老保险参保预测任务的领域专用大语言模型，并引入 DKI-RDistill 方法，将基于政策的线索注入提示词中，包括由 Probit 模型导出的边际效应和户籍所在省份的养老保险规则。该方法随后使用 LoRA/SFT 将经理由增强的监督信息蒸馏到开放权重的 MoE 学生模型中，并通过在真实标签下重新生成相应案例来纠正教师模型的错误。在 CHFS 2019 盲测数据分割上，FlexPension-LLM 实现了 0.9316 的综合 F1 分数，超越了……

    arXiv:2609.05189v1 Announce Type: new  Abstract: Assessing the impacts of social policy changes is a widely acknowledged challenge for policymakers. Econometric methods can be unreliable when extrapolating to hypothetical scenarios, while field pilot programs are highly costly. In this paper, we propose using large language models (LLMs) as policy-assessment tools adapted from general-purpose models. We present FlexPension-LLM, the first domain-specialized large language model for a hierarchical pension-enrollment prediction task among flexible workers in China, and introduce DKI-RDistill, which injects policy-grounded cues into the prompt, including Probit-derived marginal effects and hukou-province pension rules. The method then uses LoRA/SFT to distill rationale-augmented supervision into an open-weight MoE student, with teacher errors corrected by regenerating those cases under ground-truth labels. On a CHFS 2019 blind split, FlexPension-LLM achieves 0.9316 Composite F1, surpassing
    
[^11]: 利用知识单元间的潜在距离测量生物医学论文的新颖性

    Measuring the Novelty of Biomedical Papers Using the Latent Distances between Knowledge Units

    [https://arxiv.org/abs/2609.05175](https://arxiv.org/abs/2609.05175)

    该论文提出了一种综合性的新颖性测量方法，通过量化知识单元之间的网络、语义和层级三种关系所对应的潜在距离，突破了以往仅依赖共现关系评估论文新颖性的局限。

    

    测量科学论文的新颖性是研究评价与科学计量学中的核心问题。从知识重组的视角出发，以往研究大多聚焦于知识单元的共现关系来评估科学论文的新颖性。然而，这些研究往往忽视了知识单元之间的其他关系，这种狭隘的视角可能导致对科学论文新颖性的评估不够准确或不够完整。为填补这一空白，本研究引入了一种综合性的新颖性测量方法，该方法纳入了知识单元之间的三种关系类型：网络关系、语义关系和层级关系，并利用这些关系来量化知识单元之间的潜在距离。基于发表于PLoS ONE的142,036篇论文的数据集以及来自H1 Connect平台的验证数据集，我们的结果表明：(1) 每种关系类型都能捕捉到MeSH主题词之间不同的潜在距离；(2) 比较……（摘要在此处被截断）

    arXiv:2609.05175v1 Announce Type: cross  Abstract: Measuring the novelty of scientific papers is a central concern in research evaluation and scientometrics. From a recombination perspective, prior studies have largely focused on the co-occurrence of knowledge units to assess the novelty of scientific papers. However, these studies often overlook other relationships between knowledge units. This narrow view may result in inaccurate or incomplete evaluations of novelty for scientific papers. To fill this gap, this study introduces a comprehensive novelty measurement that incorporates three types of relationships between knowledge units: network, semantic, and hierarchical. These relationships are used to quantify the latent distances among knowledge units. Using a dataset of 142,036 articles published in PLoS ONE and a validation dataset from the H1 Connect platform, our results demonstrate that (1) each relationship type captures distinct latent distances between MeSH terms; (2) compar
    
[^12]: 超越未压缩的压缩：RAG中软上下文压缩的两阶段训练方案

    Compression Beyond the Uncompressed: A Two-Stage Training Recipe for Soft Context Compression in RAG

    [https://arxiv.org/abs/2609.05152](https://arxiv.org/abs/2609.05152)

    提出两阶段训练方案DEX-Comp，通过纯蒸馏热启动加困难查询强化学习探索，使RAG中的软上下文压缩模型性能超越未压缩系统，同时实现16倍的上下文压缩。

    

    检索增强生成（RAG）通过外部知识增强语言模型的能力，但冗长的检索上下文会膨胀输入规模并降低推理效率。软上下文压缩将每个文档编码为显著更短的嵌入序列。然而，大多数现有方法是通过从未压缩RAG系统的输出进行蒸馏来训练的，这从本质上限制了其相对于原始模型的性能。为了解决这一局限，我们提出了DEX-Comp，一种两阶段训练方案：纯蒸馏阶段仅在未压缩RAG的正确响应上对压缩模型进行热启动，随后困难探索阶段仅在未压缩RAG失败的查询上进行强化学习，迫使模型探索更适合压缩表示的计算模式。在检索深度从top-5到top-30的五个开放域问答基准上，DEX-Comp将检索上下文压缩16倍，并准确率超越（摘要原文在此处截断）

    arXiv:2609.05152v1 Announce Type: new  Abstract: Retrieval-Augmented Generation (RAG) enhances language models with external knowledge, but the lengthy retrieved context inflates the input and degrades inference efficiency. Soft context compression encodes each document into a substantially shorter embedding sequence. However, most existing approaches are trained by distilling outputs from uncompressed RAG systems, inherently limiting their performance relative to the original model. To address this limitation, we propose DEX-Comp, a two-stage training recipe: Pure Distillation warm-starts the compression model on the uncompressed RAG's correct responses only, and Hard Exploration then runs reinforcement learning solely on queries the uncompressed RAG fails, forcing the model to explore computation patterns better suited to compressed representations. On five open-domain QA benchmarks at retrieval depths from top-5 to top-30, DEX-Comp compresses retrieved contexts by $16\times$ and acc
    
[^13]: 每个神经元至多发放一次脉冲的大语言模型

    Large Language Models with At Most One Spike per Neuron

    [https://arxiv.org/abs/2609.05151](https://arxiv.org/abs/2609.05151)

    该论文提出一种基于参考的编码策略，首次构建了完全基于TTFS编码、每个神经元至多发放一次脉冲的脉冲神经网络大语言模型架构，可端到端训练，并在BERT和GPT-2等模型上实现了与人工神经网络相当的性能与显著节能优势。

    

    利用其固有的稀疏事件驱动计算特性，脉冲神经网络（SNNs）为实现节能的大语言模型（LLMs）提供了一条有前景的路径。首次脉冲时间（TTFS）编码在时间窗口内使每个神经元最多产生一次脉冲，从而实现极低的发放率。然而，传统的TTFS SNN局限于特定结构，使得难以用TTFS编码LLM中的某些模块——例如层归一化和矩阵乘法。为克服这一限制，我们引入了一种基于参考的策略，专门用于编码LLM的四个核心组件：嵌入层、层归一化、注意力相关操作和dropout。我们构建了一个完全基于TTFS的SNN架构，并对其进行端到端训练。在BERT和GPT-2等现代LLM上的实验表明，我们的方法在自然语言理解和常识推理任务上取得了与人工神经网络（ANN）相当的性能。

    arXiv:2609.05151v1 Announce Type: cross  Abstract: Leveraging their inherent sparse event-driven computation, spiking neural networks (SNNs) offer a promising path toward energy-efficient large language models (LLMs). Time-to-first-spike (TTFS) coding generates at most one spike per neuron within a time window, yielding extremely low firing rates. However, conventional TTFS SNNs are restricted to specific structures, making it challenging to encode certain blocks in LLM -- such as layer normalization and matrix multiplication --using TTFS. To overcome this limitation, we introduce a reference-based strategy specifically to encode the four core LLM components: embedding layers, layer normalization, attention-related operations and dropout. We construct a fully TTFS-based SNN architecture and train it end-to-end. Experiments on modern LLMs like BERT and GPT-2 demonstrate that our approach achieves performance comparable to ANN counterparts on natural language understanding and common-sen
    
[^14]: 从视觉到语言：探究多模态决策中的因果信息流

    From Vision to Language: Investigating Causal Information Flow in Multimodal Decision-Making

    [https://arxiv.org/abs/2609.05149](https://arxiv.org/abs/2609.05149)

    通过对视觉-语言模型的视频-文本注意力通路进行逐层因果干预，发现视觉信息主要在处理候选答案选项时被整合进决策，其中名词作为语义锚点、动词对时间推理至关重要，并揭示了VLM在时间推理上的独特缺陷模式。

    

    视觉-语言模型（VLM）通常通过其最终预测结果来进行评估，但要理解这些决策是否基于视觉证据，需要追溯视觉信息是如何参与到基于语言的决策中的。为此，我们在基于视频的生成式多选题式任务场景中，通过对视频-文本注意力通路施加逐层因果干预，来研究跨模态信息流。我们针对空间、因果和时间三种视觉推理进行研究。结果表明，视觉信息主要在模型处理候选答案选项时被整合，这些选项是最终决策的主要文本锚定点。我们进一步发现，名词在多模态信息丰富化过程中作为语义锚点发挥重要作用，而动词在处理时间关系时更为关键。最后，我们在时间推理中识别出一种独特的模式，表明VLM在处理时间信息时存在困难。

    arXiv:2609.05149v1 Announce Type: new  Abstract: Vision-Language Models are commonly evaluated through their final predictions, but understanding whether these decisions are grounded in visual evidence requires tracing how visual information contributes to language-based decisions. With this purpose in mind, we investigate cross-modal information flow in a video-based generative multiple-choice-like setting by applying a layer-wise causal intervention on video-text attention pathways. We target spatial, causal, and temporal visual reasoning. Our results show that visual information is mainly integrated while the model processes the candidate answer options, which serve as the primary textual grounding sites for the final decision. We further show that nouns play an important role as semantic anchors during multimodal enrichment, while verbs are more relevant when temporal relations are processed. Finally, we identify a distinct pattern in temporal reasoning, suggesting that VLMs strugg
    
[^15]: 大规模写作测评中AI辅助评分的人机协同框架

    A Human-in-the-Loop Framework for AI-Assisted Scoring in Large-Scale Writing Assessment

    [https://arxiv.org/abs/2609.05143](https://arxiv.org/abs/2609.05143)

    本研究提出并验证了一个面向大规模写作测评的AI辅助评分人机协同框架，基于约一万份真实全国考试数据，在多个评分维度上实现了AI与人工评分的中高程度一致性，有效降低人工评分工作量。

    

    人工智能（AI），特别是大型语言模型（LLM）在教育测评中的融合，为提升评分流程的效率和可扩展性开辟了新的机遇。本研究提出并验证了一个面向大规模全国性测评中书面作答的AI辅助评分框架。该方法聚焦于约150-200词的短篇书面文本，并纳入人机协同（human-in-the-loop）策略，在降低人工工作量的同时保持测评质量。该研究基于真实的运行环境，使用了最近两次全国性考试的数据，每次考试包含约5,000份学生作答。我们分析了AI生成评分与人工评分者在多个评分标准维度上的一致性，以及所提出的决策流程对及格/不及格判定结果的影响。结果显示，该模型与（人工评分者之间）存在中等至高度的一致性……

    arXiv:2609.05143v1 Announce Type: cross  Abstract: The integration of artificial intelligence (AI), particularly large language models (LLMs), into educational assessment has opened new opportunities to enhance the efficiency and scalability of grading processes. This study presents the design and validation of an AI-assisted scoring framework for written responses in a large-scale national assessment. The proposed approach focuses on short written texts of approximately 150-200 words and incorporates a human-in-the-loop strategy to preserve assessment quality while reducing manual workload. The study is grounded in a real operational context, using data from two recent editions of a nationwide test, each comprising approximately 5,000 student responses. We analyze the alignment between AI-generated scores and human raters across multiple rubric dimensions, as well as the impact of the proposed decision flow on pass/fail outcomes. Results show moderate to high agreement between the mod
    
[^16]: NS-ST-GraphRAG：面向文学知识处理的神经符号时空GraphRAG框架

    NS-ST-GraphRAG: Neuro-Symbolic Spatio-Temporal GraphRAG for Literary Knowledge Processing

    [https://arxiv.org/abs/2609.05139](https://arxiv.org/abs/2609.05139)

    提出了神经符号时空GraphRAG框架NS-ST-GraphRAG，通过选择符合查询时空范围的图状态并提供可追溯证据来处理长篇文学叙事中受时间、空间和关系约束的问答，并发布了首个中国古典文学开放多跳问答基准Red-Chamber-QA。

    

    长篇文学叙事对检索增强生成提出了独特的信息处理挑战：相关证据分布在各个章节之中，人物关系随叙事时间不断演变，而正确答案可能同时依赖于时间、空间和关系约束。我们提出了NS-ST-GraphRAG，一个神经符号时空GraphRAG框架，它集成了本体引导的抽取、确定性约束检查、双重时间坐标、空间场景属性以及动态子图检索。该框架不是从单一的语料库级图中进行检索，而是选择对查询的时空范围有效的图状态，并将生成的答案建立在可追溯的证据之上。我们还进一步推出了Red-Chamber-QA，据我们所知，这是首个面向中国古典文学的开放多跳问答基准，包含时间类、空间类和通用类问题类别，以及分部证据（注：原摘要在此处截断）。

    arXiv:2609.05139v1 Announce Type: new  Abstract: Long-form literary narratives pose a distinctive information-processing challenge for retrieval-augmented generation: relevant evidence is distributed across chapters, relations evolve over narrative time, and correct answers may depend jointly on temporal, spatial, and relational constraints. We propose NS-ST-GraphRAG, a neuro-symbolic spatio-temporal GraphRAG framework that integrates ontology-guided extraction, deterministic constraint checking, dual temporal coordinates, spatial scene attributes, and dynamic sub-graph retrieval. Instead of retrieving from a single corpus-level graph, the framework selects the graph state valid for the temporal and spatial scope of a query and grounds generated answers in traceable evidence. We further introduce Red-Chamber-QA, to our knowledge the first open multi-hop question-answering benchmark for classical Chinese literature, with time-, space-, and general-question categories, per-part evidence 
    
[^17]: 基于整数线性规划的码切换话语语言识别改进

    Improving Language Identification for Code-Switched Utterances with Integer Linear Programming

    [https://arxiv.org/abs/2609.05099](https://arxiv.org/abs/2609.05099)

    本文通过解决MaskLID过度依赖词级语言关联分数的问题，并将其底层优化算法重新表述为可引入清晰可解释约束的整数线性规划，显著提升了码切换话语的语言识别性能。

    

    自动识别码切换（CS）话语仍然是语言识别（LID）系统面临的一项挑战，导致此类文本在大语言模型的训练数据中代表性不足。本文重新审视了MaskLID——一种最先进的码切换识别方法，该方法无需训练即可检测任意语言组合。我们做出了三项主要贡献：（a）我们揭示并解决了MaskLID的一个重大问题：其过度依赖词级语言关联分数；（b）我们将底层优化算法重新表述为整数线性规划，使我们能够试验大量清晰且可解释的约束条件；（c）每一项改进都大幅提升了基线系统，我们在涉及10种不同语言的实验中对此进行了展示，观察到码切换基准测试上的性能显著提升。我们公开了代码和数据以保证可复现性。

    arXiv:2609.05099v1 Announce Type: new  Abstract: Automatic identification of code-switched (CS) utterances remains a challenge for language identification (LID) systems, causing such texts to be underrepresented in the training data of Large Language Models. In this paper, we revisit MaskLID, a state-of-the art approach for CS identification, which requires no training and detects arbitrary language combinations. We make three main contributions: (a) we reveal, and address, a major issue of MaskLID: its overreliance on word-level language association scores; (b) we reformulate the underlying optimization algorithm as an Integer Linear Program, enabling us to experiment with a large set of clear and interpretable constraints; (c) each of these improvements vastly improves the baseline system, as we illustrate in experiments involving 10~diverse languages, where we observe a strong boost in performance on CS benchmarks. We release our code and data for reproducibility.
    
[^18]: 通过论证分析衡量AI问责性：模型推理能否经受住审视？

    Measuring AI Accountability Through Argumentation Analysis: Can Model Reasoning Withstand Scrutiny?

    [https://arxiv.org/abs/2609.05088](https://arxiv.org/abs/2609.05088)

    该论文提出一种基于沃尔顿论证图式理论与Govier论证合理性标准的四阶段辩证评估协议，通过衡量模型在面对批判性问题时为其裁决辩护的结构质量，实现了在缺乏公认正确答案的真实模糊情境下对大语言模型道德推理的严格评估。

    

    AI监督方法依赖于真实标准（ground truth）进行验证，但什么才是恰当的AI行为本身存在争议。这使得对大语言模型道德推理的评估以及基于辩论的监督在无形中回避了现实中的模糊性。我们研究了一种旨在在此类模糊性下依然有效的替代性标准：模型针对其裁决所能够构建的辩护的结构质量，即在面对批判性问题时为结论辩护的能力。该质量通过一个四阶段的辩证协议来测量，该协议建立在沃尔顿（Walton）的论证图式理论与戈维尔（Govier）的论证合理性标准之上。该协议能够适应不同的推理框架，超越了多选题式的评估形式，并同时考察得出裁决之前的推理过程以及事后的正当性辩护。在九个前沿模型和200个高模糊性的MoralChoice题目上进行实验——共计6,778个由评审员打分的单元，二元失败判断的评审员间一致率达89.6%——模型的辩护……（原文摘要于此处被截断）

    arXiv:2609.05088v1 Announce Type: new  Abstract: AI oversight methods rely on ground truth for validation, but what constitutes appropriate AI behavior is contested. This leaves evaluation of moral reasoning in LLMs and debate-based oversight implicitly avoiding realistic ambiguity. We investigate an alternative standard designed to function despite such ambiguity: structural quality of the defence a model can mount for its verdicts in response to critical questions, measured through a four-phase dialectical protocol grounded in Walton's theory of argumentation schemes and Govier's criteria for argument cogency. The protocol is adaptive to different frames of reasoning, extends beyond multiple-choice framing, and treats both the reasoning that precedes a verdict and its post-hoc justification. Across nine frontier models and 200 high-ambiguity MoralChoice items -- $6,778$ judge-scored cells, validated against $89.6\%$ inter-judge agreement on the binary failure judgment -- models defen
    
[^19]: TruthInsightBench：一个基于证据的开放式科学发现智能体自动化评估基准

    TruthInsightBench: An Evidence-Grounded Benchmark for Automated Evaluation of Open-Ended Scientific Discovery Agents

    [https://arxiv.org/abs/2609.05079](https://arxiv.org/abs/2609.05079)

    该论文提出TruthInsightBench——首个面向开放式科学发现（而非结果复现）的智能体评估基准，通过40个隐藏源结论的盲测任务与基于LLM的六维证据成熟度自动评分体系，实现了无需逐实例人工评分的科学发现智能体自动化评估。

    

    自主编码智能体正日益被提议为“AI科学家”系统，用于执行分析并撰写研究报告，但执行预先设定的分析与真正做出科学发现并不等同。现有基准测试是为“复现”而配置的：任务、数据和评分细则均围绕一个隐藏的目标研究构建，并以恢复其结果作为奖励。我们提出了TruthInsightBench，一个为“发现”而配置的基准。其40个盲测任务取自10个科学领域的40项同行评审研究，仅向智能体提供一个中立的科学目标和冻结的数据；源结论、预期数值和分析路径均被隐藏，由智能体自行判断数据支持何种论断。一个固定的基于大语言模型（LLM）的评估器从六个维度对智能体自身论断的证据成熟度进行评分，并具体化为29个基于产出物的评分条目，采用自动化、确定性的聚合方式，无需逐实例的人工评分，因此评估可以……

    arXiv:2609.05079v1 Announce Type: new  Abstract: Autonomous coding agents are increasingly proposed as AI-scientist systems that conduct analyses and write research reports, but executing a prescribed analysis is not the same as making a discovery. Existing benchmarks are configured for reproduction: tasks, data, and rubrics are built around a hidden target study, and recovery of its result is rewarded. We present TruthInsightBench, a benchmark configured for discovery. Its 40 blind tasks, drawn from 40 peer-reviewed studies across 10 scientific domains, expose only a neutral scientific objective and frozen data; source conclusions, expected values, and analysis paths are withheld, leaving the agent to determine what claim the data support. A fixed LLM-based judge scores the evidentiary maturity of an agent's own claims along six dimensions, operationalized as 29 artifact-grounded items, with automated, deterministic aggregation and no per-instance human grading, so evaluation can be r
    
[^20]: 影响分数与Transformer可解释性：推理时注意力头有效影响的度量

    Influence Score and Transformers interpretability: Measure of the Effective Impact of Attention Heads at inference time

    [https://arxiv.org/abs/2609.05074](https://arxiv.org/abs/2609.05074)

    本文提出一种结合logits方向性影响与残差流结构性贡献的影响分数，可在注意力头、层和网络三个尺度上量化注意力头对分类决策的贡献，并在提示注入检测的DeBERTa模型上揭示了正确与错误预测的不同决策行为。

    

    我们提出了一种影响分数，用于量化面向提示注入检测的基于Transformer的模型中注意力头对分类决策的贡献。该分数将模型对logits的方向性影响与残差流中的结构性贡献相结合，实现了在注意力头、层和网络三个层面的多尺度分析。将该框架应用于专门用于提示注入检测的DeBERTa模型，我们揭示了正确预测与错误预测之间截然不同的决策行为。我们的方法在细粒度电路分析与基于全局输出的方法之间提供了有效的折中，并为系统性地研究Transformer分类器的决策机制提供了一种可行途径。

    arXiv:2609.05074v1 Announce Type: new  Abstract: We propose an influence score to quantify the contribution of attention heads to classification decisions in Transformer-based models designed for prompt injection detection. The score combines directional influence on the logits with structural contribution within the residual stream, enabling a multi-scale analysis at the head, layer, and network levels. Applied to a DeBERTa model specialized for prompt injection detection, our framework reveals distinct decision behaviours between correct and erroneous predictions. Our method provides an effective compromise between fine-grained circuit analysis and global output-based methods, and offers a systematic way to study decision mechanisms in Transformer classifiers.
    
[^21]: 一种用于复杂临床诊断决策支持的结构化辩论-智能体混合框架

    A Structured Debate-Mixture-of-Agents Framework for Complex Clinical Diagnostic Decision Support

    [https://arxiv.org/abs/2609.05069](https://arxiv.org/abs/2609.05069)

    提出了一种结构化的辩论-智能体混合框架（DMoA），通过多智能体角色交互实现迭代式诊断推理，在罕见病和疑难病例上相比GPT-4o基线将诊断准确率提高10.21个百分点、安全率提高11.36个百分点。

    

    大型语言模型在医学任务中展现出潜力，但其单轮问答的形式无法反映临床诊断的实际操作方式，因此它们在复杂诊断场景中仍然受限。我们开发了辩论-智能体混合框架（DMoA），这是一种新颖的多智能体框架，通过结构化的基于角色的交互来支持迭代式诊断推理。基座模型和DMoA在297例罕见病病例和1,719例疑难病例上进行了评估。在两个数据集上，DMoA相比GPT-4o基线将最可能诊断的准确率提高了10.21个百分点，安全率提高了11.36个百分点。消融实验表明，性能提升并非仅仅源于使用了更多模型或更长的输出，也反映了结构化工作流的贡献。进一步的分析考察了框架设计、基座模型选择和token预算对性能的影响。DMoA表现更好……

    arXiv:2609.05069v1 Announce Type: cross  Abstract: Large language models (LLMs) show potential for medical tasks, but their single-turn question-answer format does not reflect how clinical diagnosis is performed in practice. As a result, they remain limited in complex diagnostic settings. We developed Debate-Mixture-of-Agents (DMoA), a novel multi-agent framework that structures role-based interaction to support iterative diagnostic reasoning. Base models and DMoA were evaluated on 297 rare disease cases and 1,719 challenging cases. Across both datasets, DMoA improved most likely diagnosis accuracy by 10.21 percentage points and safety rate by 11.36 percentage points over GPT-4o baseline. Ablation experiments showed that the gains were not simply due to the use of more models or longer outputs, but also reflected the contribution of the structured workflow. Further analyses examined how framework design, base model choice, and token budget affected performance. DMoA performed better wi
    
[^22]: 重复提问会耗尽大语言模型的品牌推荐，却耗不尽其引用来源

    Repeated Queries Exhaust an LLM's Brand Recommendations but Not Its Sources

    [https://arxiv.org/abs/2609.05059](https://arxiv.org/abs/2609.05059)

    重复提问相同的购买问题时，不联网检索的大语言模型会不断涌现新的品牌推荐而难以饱和，启用检索的引擎则快速封闭品牌列表，但所有引擎引用的来源域名始终持续增加、远未收敛。

    

    重复提出相同的购买类问题是否会耗尽语言模型的品牌推荐，取决于其是否具备检索能力。在涵盖300个“问题-引擎”组合单元的实验中（50个问题、6个引擎、每个单元运行15次，并对1,470个经人工裁定的组织进行开放式抽取），五个不借助网络搜索作答的引擎在86-92%的单元中到第15次运行时仍在出现从未见过的品牌，其品牌储备中位数为15-31个组织；而唯一启用检索功能的引擎则封闭了其品牌列表（中位数8个组织，64%的单元仍在新增），这与此前四个深度实验单元中启用网络搜索的运行在第10次左右即趋于饱和的结果相符。在所有测试的时间范围内，被引用域名的累积量持续上升：四个深度实验单元在第24次运行时仍在新增域名，仅观测到Chao2下限估计值的59-84%，检索型引擎的广度单元中也有44%在第15次运行时仍在新增域名。单次运行仅能覆盖五次运行品牌集合的62-77%，且跨引擎来看，每个问题的中位数可引出38个组织……（摘要原文在此处截断）

    arXiv:2609.05059v1 Announce Type: cross  Abstract: Whether repeated identical buying questions exhaust a language model's brand recommendations depends on retrieval. Across 300 question-engine cells (50 questions, six engines, 15 runs each, open extraction over 1,470 adjudicated organizations), the five engines answering without web search were still adding never-seen brands at run 15 in 86-92% of cells, with median repertoires of 15-31 organizations; the one retrieval-enabled engine closed its list (median 8 organizations, 64% of cells still adding), matching four earlier deep cells where web-search runs saturated by run ten. Cited-domain accumulation keeps rising at every horizon tested: four deep cells were still adding domains at run 24 with 59-84% of the Chao2 lower-bound estimate observed, and 44% of the retrieval engine's breadth cells were still adding domains at run 15. A single run shows 62-77% of the five-run brand set, and across engines the median question draws 38 organiz
    
[^23]: EuroAlpaca：面向欧洲语言的指令数据任务保持式本地化

    EuroAlpaca: Task-Preserving Localisation of Instruction Data for European Languages

    [https://arxiv.org/abs/2609.05043](https://arxiv.org/abs/2609.05043)

    本文提出EuroAlpaca任务保持式本地化流程和European-IFEval多语言基准，覆盖50种欧洲语言，并证明直接机器翻译的指令数据会损害模型的可验证指令遵循能力（准确率下降29.8%），而任务保持式本地化可避免此问题。

    

    机器翻译（MT）为将英语指令微调数据扩展到多种语言提供了一种可扩展的方法，但它可能会扭曲任务关键约束和所需输出，从而产生损坏的训练样本，并降低在此类数据上训练的模型的性能。我们介绍了EuroAlpaca，这是一个任务保持式本地化流程和近并行资源，涵盖50种欧洲语言及地区变体，同时推出了European-IFEval，一个用于可验证指令遵循的多语言基准测试。根据具体示例，我们的流程在保留任务关键内容的前提下应用逐字段机器翻译，或重建任务等效的目标语言实例，随后进行跨字段连贯性和目标语言一致性验证。在四个大语言模型的LoRA实验中，使用直接翻译的数据进行训练虽然能提升Aya评估套件上的ROUGE-L和F-BERT分数，但相较于未适配的基础模型，其在European-IFEval上的准确率下降了29.8%……

    arXiv:2609.05043v1 Announce Type: new  Abstract: Machine translation (MT) offers a scalable way to extend English instruction-tuning data to multiple languages, but it can distort task-critical constraints and required outputs, creating corrupted training examples and degrading models trained on such data. We introduce EuroAlpaca, a task-preserving localisation pipeline and near-parallel resource covering 50 European languages and regional varieties, together with European-IFEval, a multilingual benchmark for verifiable instruction following. Depending on the example, our pipeline applies field-wise MT while preserving task-critical content or reconstructs a task-equivalent target-language instance, followed by validation of cross-field coherence and target-language consistency. Across LoRA experiments with four LLMs, training on directly translated data improves ROUGE-L and F-BERT on the Aya Evaluation Suite, but reduces accuracy on European-IFEval by 29.8% relative to the unadapted b
    
[^24]: 大语言模型如何评估感知道德能动性？探究人-人工体交互中的道德决策

    How do LLMs Evaluate Perceived Moral Agency? Investigating Moral Decision-Making in Human-Artificial Agents Interactions

    [https://arxiv.org/abs/2609.05037](https://arxiv.org/abs/2609.05037)

    本文首次通过实证研究比较了人类与大语言模型在智慧城市场景中评估人类与自主人工体感知道德能动性（PMA）的方式，发现LLMs表现出更高的道德能动性感知。

    

    随着大语言模型（LLMs）开始承担需要提供道德建议的角色，理解它们如何归属道德能动性变得至关重要。人类拥有道德能动性，即做出符合伦理指引的决策并为其后果承担责任的能力，这是道德心理学中一个已被充分确立的构念。然而，随着机器人、无人机和无形AI系统等人工体日益嵌入智慧城市环境，道德能动性是否以及如何被归属于它们的问题变得愈发紧迫。据我们所知，本文提出了首项实证研究，比较人类与LLMs如何评估人类智能体与自主人工体的感知道德能动性（PMA），这些智能体在实体化形式上各不相同，并被置于可信的智慧城市场景中。我们采用经过验证的PMA量表的改编版本，将该评估协议应用于190名人类参与者以及多种LLMs。我们的评估显示出现了更高的道德能动性感知（原文摘要在此处截断）。

    arXiv:2609.05037v1 Announce Type: cross  Abstract: As LLMs take on roles requiring moral advice, understanding how they attribute moral agency becomes critical. Humans possess moral agency, the capacity to make ethically guided decisions and bear responsibility for their consequences, a well-established construct in moral psychology. Yet as artificial agents (AAs) such as robots, drones, and disembodied AI systems become increasingly embedded in smart city environments, the question of whether and how moral agency is attributed to them takes on new urgency. This paper presents, to the best of our knowledge, the first empirical study comparing how humans and LLMs evaluate perceived moral agency (PMA) across human and autonomous artificial agents varying in embodiment, situated in plausible smart city scenarios. Using an adaptation of a validated PMA scale, we applied a protocol to 190 human participants as well as various LLMs. Our evaluation reveals higher perceptions of moral agency i
    
[^25]: 道德能力先于道德内容：为什么大语言模型智能体缺乏实现连贯对齐的先决条件

    Moral Competence Before Moral Content: Why LLM Agents Lack the Prerequisites for Coherent Alignment

    [https://arxiv.org/abs/2609.05036](https://arxiv.org/abs/2609.05036)

    本文提出判断稳定性、单调性、果断性和帕累托可行性四个结构性条件，用以衡量仅凭行为即可评估的“道德能力”，并揭示前沿大语言模型智能体普遍缺乏实现连贯对齐所需的这一结构性前提。

    

    AI对齐要求AI系统遵循人类的规范、价值观或意图。在价值多元主义的背景下并不存在唯一正确的对齐目标，但一个共同的先决条件是系统的行为表现出一种连贯的政策：即从情境到判断的映射，当情境中的道德相关特征保持不变时该映射保持稳定，而当这些特征发生变化时该映射也随之改变。我们提出了这类连贯政策的四个结构性条件：判断稳定性、单调性、果断性和帕累托可行性。它们共同衡量了一种仅凭行为即可评估的道德能力，无需参照任何道德标准或专家基准，从而构成对齐的结构性底线，而非规范性目标。我们在三个模拟部署场景中演示了该方法，这些场景中的基于大语言模型的智能体面临道德困境。我们在五种类述变体、五个升级等级和三个领域的因子实验设计下评估了九个前沿模型。

    arXiv:2609.05036v1 Announce Type: new  Abstract: AI alignment requires AI systems to adhere to human norms, values, or intentions. Under value pluralism there is no correct target, but a shared prerequisite is that the system's behavior expresses a coherent policy: a mapping from situations to verdicts that is invariant while a situation's morally relevant features are preserved, and sensitive when they change. We introduce four structural conditions for such coherent policies: verdict stability, monotonicity, decisiveness, and Pareto viability. Together they measure a form of moral competence that is evaluable from behavior alone, without reference to a moral standard or expert baseline, forming a structural floor for alignment rather than a normative target. We demonstrate the methodology on three simulated deployments featuring LLM-based agents facing moral dilemmas. Evaluating nine frontier models under a factorial design of five paraphrases, five escalation levels, and three domin
    
[^26]: 利用低层次符号能力实现无监督的幻觉检测接地

    Leveraging Low-Level Symbolic Competences for Unsupervised Grounding in Hallucination Detection

    [https://arxiv.org/abs/2609.05025](https://arxiv.org/abs/2609.05025)

    本文提出让LLM根据参考文档构建SQL数据库，并将其作为无监督幻觉检测的接地依据，通过这种神经符号化方法在RAGTruth和DiaHalu数据集上超越了直接预测并与最先进方法相媲美。

    

    幻觉——即语言模型生成事实上错误或缺乏来源支持的内容——是提示式和微调语言模型面临的重大挑战。由于大语言模型（LLM）的推理过程不透明，往往难以解释模型的输出为何可能不准确，因此检测幻觉十分困难。在本工作中，我们研究LLM能否使用一种替代性的、低层次的符号能力（如SQL）来在某个高层次任务中进行无监督的幻觉检测。为此，我们让LLM根据参考文档构建一个SQL数据库。该SQL数据库随后被用于在以数据库为依据的幻觉检测流程中，对参考内容和采样响应进行推理，从而提供一种神经符号化的检查。在RAGTruth和DiaHalu幻觉检测数据集上，我们发现我们的方法优于直接预测，并可与最先进的方法相媲美。

    arXiv:2609.05025v1 Announce Type: cross  Abstract: Hallucination-where a language model generates outputs that are factually incorrect or unsupported by the source-is a major challenge for both prompted and fine-tuned language models. Detecting hallucinations is difficult due to the opaque reasoning processes of LLMs, which often provide little insight into why a model's output may be inaccurate.   In this work, we investigate whether an LLM can use an alternative, low level, symbolic competence such as SQL for unsupervised hallucination detection in some high level task. For this, we make an LLM build an SQL database from reference documents. This SQL database is then used for reasoning over the reference and the sampled response in a hallucination detection pipeline that is grounded in the database, thereby providing a neurosymbolic checkup.   On RAGTruth and DiaHalu hallucination detection datasets, we find that our approach improves on direct prediction and competes with state-of-t
    
[^27]: MoirfEolas 与 CríochScore：开发与爱尔兰语形态学对齐的分词资源及评估方法

    MoirfEolas and Cr\'iochScore: Developing Resources for and the Evaluation of Tokenization Alignment with Irish Morphology

    [https://arxiv.org/abs/2609.05022](https://arxiv.org/abs/2609.05022)

    本文提出了爱尔兰语形态数据集 MoirfEolas 和分词评估指标 CríochScore，发现 Unigram 语言模型的分词与爱尔兰语形态边界对齐程度最高，且形态对齐与压缩率及词汇效率之间存在权衡。

    

    本文提出了针对爱尔兰语的新型分词资源，以及衡量分词与该语言形态边界对齐程度的评估方法。我们提出了 MoirfEolas——一个包含超过 35,000 个爱尔兰语单词及其对应的遮蔽形式（eclipsis）、前缀和后缀映射关系的数据集；同时提出了评估指标 CríochScore，用于评估分词结果与 MoirfEolas 中形态边界的对齐程度。我们使用 CríochScore 以及分词文献中的内在评估指标，对常见的分词算法进行了评估。我们发现，Unigram 语言模型比其他被评估的算法更常与爱尔兰语的形态结构对齐。我们还发现，分词的形态对齐程度与压缩率以及词汇表效率之间存在权衡关系，这为爱尔兰语自然语言处理的发展提供了实用见解。该数据集有助于应对爱尔兰语低资源地位的问题。

    arXiv:2609.05022v1 Announce Type: new  Abstract: This paper presents new tokenization resources for Irish and evaluation measures of alignment with the morphological boundaries of the language. We present MoirfEolas, a dataset of over 35,000 Irish words mapped to their respective eclipses, prefixes and suffixes as well as an evaluation metric Cr\'iochScore, that evaluates the alignment of tokenizations with the morphological boundaries present in MoirfEolas. We evaluate common tokenization algorithms using Cr\'iochScore as well as intrinsic metrics present in the tokenization literature. We find that the Unigram Language Model aligns with Irish morphology more often than the other algorithms evaluated. We also find trade-offs between morphological-alignment of tokenization with both compression as well as vocabulary efficiency, providing practical insights for Irish natural language processing development. This dataset contributes towards combating the Irish language's low-resource sta
    
[^28]: BIT.UA参加BioASQ 14B：基于pg_textsearch和Qdrant的模块化检索与基于智能体的答案生成

    BIT.UA at BioASQ 14B: Modular Retrieval with pg_textsearch and Qdrant, and Agent-Based Answer Generation

    [https://arxiv.org/abs/2609.04999](https://arxiv.org/abs/2609.04999)

    BIT.UA团队在BioASQ 14B生物医学问答挑战赛中，通过采用基于PostgreSQL pg_textsearch和Qdrant的模块化检索架构，以及LLM-as-a-judge框架和新型智能体法定人数机制进行答案生成，显著改进了检索与生成流水线。

    

    本文描述了来自阿威罗大学的BIT.UA团队参加第14届BioASQ任务B生物医学问答挑战赛的情况。在以往提交方案的基础上，我们引入了经过大幅重构的模块化代码库，并对流水线的检索和生成两个组件进行了重大改进。在Phase A文档检索阶段，我们用基于PostgreSQL的pg_textsearch替换了PyTerrier PISA索引来执行BM25检索，并采用Qdrant进行稠密嵌入索引，实现了更高效的存储和GPU加速的相似度搜索。我们探索了基于HyDE的查询扩展以及Context-1检索策略。此外，还开发了新的重排序器训练流程，结合稠密检索进行负采样。在Phase A+和Phase B答案生成阶段，我们引入了LLM-as-a-judge（大语言模型作为评判者）框架以及一种新颖的智能体法定人数机制，即多个具有不同提示的智能体……（原文在此处截断）

    arXiv:2609.04999v1 Announce Type: new  Abstract: This paper describes the participation of the BIT.UA team from the University of Aveiro in the 14th edition of the BioASQ Task B challenge on biomedical question answering. Building on our previous submissions, we introduced a substantially refactored and modular codebase, and made significant changes to both the retrieval and generation components of the pipeline. For Phase~A document retrieval, we replaced the PyTerrier PISA index with PostgreSQL-based pg\_textsearch for BM25 retrieval and adopted Qdrant for dense embedding indexing, enabling more efficient storage and GPU-accelerated similarity search. We explored HyDE-based query expansion alongside a Context-1 retrieval strategy. A new reranker training pipeline was developed, incorporating dense retrieval for negative sampling. For Phases A+ and B answer generation, we introduced an LLM-as-a-judge framework and a novel agent quorum mechanism, where multiple agents with diverse prom
    
[^29]: BeaconKV：由信标查询引导的键值缓存压缩方法，实现高效的大推理模型推理

    BeaconKV: Key-Value Cache Compression Guided by Beacon Queries for Efficient Large Reasoning Model Inference

    [https://arxiv.org/abs/2609.04971](https://arxiv.org/abs/2609.04971)

    该论文发现长程推理中存在会重新关注早期关键上下文的“思维回溯token”，且其查询在嵌入空间中聚成少数相似组，据此提出无需训练的KV缓存压缩方法BeaconKV，从而高效支持大型推理模型的推理。

    

    大型推理模型通过扩展的思维链生成获得了卓越的问题解决能力，但由此产生的键值缓存随序列长度线性增长，造成严重的内存瓶颈，在长推理轨迹场景下常常超出GPU容量。现有的KV缓存压缩方法依赖最近的查询来估计未来token的重要性，隐含地假设这些查询可以作为未来注意力模式的可靠代理。我们证明这一假设在长程推理中并不成立：某些解码步骤会生成“思维回溯token” (Thought Revisiting Tokens, TRT)，重新关注距离较远的先前上下文，例如推理轨迹早期形成的任务求解计划。通过系统性分析，我们发现与TRT对应的查询在嵌入空间中聚集成少数几个相似度组。基于这一洞察，我们提出BeaconKV，一种无需训练的KV缓存压缩方法……

    arXiv:2609.04971v1 Announce Type: cross  Abstract: Large Reasoning Models (LRMs) achieve superior problem-solving through extended Chain-of-Thought (CoT) generation, but the resulting key-value (KV) cache grows linearly with sequence length and creates severe memory bottlenecks, often exceeding GPU capacity for long reasoning traces. Existing KV cache compression methods rely on recent queries to estimate future token importance, implicitly assuming these serve as reliable proxies for future attention patterns. We demonstrate that this assumption fails in long-horizon reasoning: certain decoding steps generate Thought Revisiting Tokens (TRT) that re-attend to distant previous context, such as task-solving plans formulated early in the trace. Through systematic analysis, we discover that queries corresponding to the TRT cluster into a small number of similarity groups in the embedding space. Based on this insight, we propose BeaconKV, a training-free KV cache compression method that mai
    
[^30]: 为什么我们在意理解：通过预测性压缩获得的能力

    Why We Care About Understanding: Competence through Predictive Compression

    [https://arxiv.org/abs/2609.04962](https://arxiv.org/abs/2609.04962)

    本文提出理解是对稳健能力的有效代理——通过掌握可实现预测的关系结构心理模型进而实现压缩——从而弥合了信息论“理解即压缩”传统与哲学家对理解的概念刻画之间的鸿沟。

    

    理解与压缩之间的关系是什么？为什么人类的理解呈现出如此高度压缩的形式？在信息论、机器学习和人工智能研究领域，一个重要的传统将理解等同于压缩——这一思想体现在 Gregory Chaitin 的名言“理解即压缩”之中。相比之下，哲学家们则从把握联系、给出解释以及处理新颖性问题等方面来刻画理解。本文通过三个相互关联的论题来弥合这两种观点。第一个论题关乎理解的概念：理解作为一种独特形式的稳健能力的有效代理，使我们能够识别应该信任谁、应该向谁学习。第二个论题关乎理解的状态：理解一个领域，就是拥有一个关于其关系结构的心理模型，该模型能够实现预测；而凡能实现预测者，便能实现压缩，因为……（原文摘要在此处截断）

    arXiv:2609.04962v1 Announce Type: new  Abstract: What is the relation between understanding and compression, and why does human understanding take such a heavily compressed form? Across information theory, machine learning, and AI research, a substantial tradition identifies understanding with compression-a thought captured in Gregory Chaitin's dictum that "comprehension is compression." Philosophers, by contrast, have characterized understanding in terms of grasping connections, giving explanations, and handling novelty. This paper bridges the two pictures through three interlocking theses. The first concerns the concept of understanding: it serves as an efficient proxy for a distinctive form of robust competence, enabling us to identify whom to trust and whom to learn from. The second concerns the state of understanding: to understand a domain is to possess a mental model of its relational structure that enables prediction, and what enables prediction enables compression, because wha
    
[^31]: 语篇依赖度：一种用于衡量翻译难度的连续性判据

    Discourse Dependency: A Continuous Criterion for Translation Difficulty

    [https://arxiv.org/abs/2609.04959](https://arxiv.org/abs/2609.04959)

    本文提出“语篇依赖度（DDP）”这一免指标的、基于源端的连续翻译难度度量，通过实体复现与代词共指衡量片段所需的指称回溯距离，揭示现有翻译基准严重偏向低DDP片段，并证明随着DDP增大，现有的各类上下文注入策略均无法达到人工译后编辑的水平。

    

    近来对更难机器翻译基准的呼吁尚未澄清“难度”究竟意味着什么。我们认为，一个有意义且目前未被测量的维度是指称回溯距离，即一个片段需要回溯到其所在文档中多远的位置才能解析其中包含的实体和代词。我们将其形式化为语篇依赖度，这是一种免指标的、基于源端的度量，通过命名实体的重复提及和代词共指关系来计算。经与黄金共指标注验证，DDP在99.2%的片段中只会单方向出错，因此高DDP片段可被确证需要长程上下文。将DDP应用于WMT24++和WMT25后发现，这两个基准都严重偏向低DDP片段，而领域标签无法区分这种差异。基于DDP，我们在英韩译后编辑的实验设置中比较了五种上下文注入策略，变化上下文的大小和选择方式。随着DDP增大，没有任何策略能跟上人工译后编辑的水平。在DDP >= 15的片段上（原文摘要在此处截断）……

    arXiv:2609.04959v1 Announce Type: new  Abstract: Recent calls for harder machine translation benchmarks have not clarified what difficulty should mean. We argue that one meaningful and currently unmeasured axis is referential reach, the distance a segment must look back into its document to resolve the entities and pronouns it contains. We formalize this as discourse dependency (DDP), a metric-free, source-side measure computed from named entity re-mentions and pronominal coreference. Validated against gold coreference, DDP errs one-sidedly in 99.2% of segments, so a high-DDP segment is certified to require long-range context. Applying DDP to WMT24++ and WMT25 shows that both are heavily skewed toward low-DDP segments, which domain labels do not distinguish. Building on DDP, we compare five context injection strategies in an English-Korean post-editing setup, varying context size and selection. As DDP grows, no strategy keeps pace with human post-editing. On segments with DDP >= 15 rat
    
[^32]: RefactorPlatform：一个用于受控评估仓库规模重构智能体的开源测试平台

    RefactorPlatform: An Open-Source Harness for Controlled Evaluation of Repository-Scale Refactoring Agents

    [https://arxiv.org/abs/2609.04898](https://arxiv.org/abs/2609.04898)

    本文提出开源评估平台RefactorPlatform，通过固定环境并显式变化模型骨干、执行模式和提示词等设计维度，实现对仓库级重构智能体的受控评估，并发现AST感知分块相比朴素token窗口分块可带来25-30%的性能提升。

    

    仓库规模的重构要求编程智能体在不改变程序行为的前提下，将单一变更传播到多个相互依赖的文件中，然而据我们所知，目前尚无现有的测试平台能够隔离决定智能体在此任务上成败的设计选择。我们提出了RefactorPlatform，这是一个开源评估测试平台，它保持环境固定，并明确地变化每个设计维度：模型骨干（通过OpenRouter和GitHub Copilot CLI）、执行模式（基线、检索增强和多智能体）以及提示词特异性。每次运行都在隔离的工作区中执行，具有实时终端流、按任务记录的token、差异和转录日志、基于AST的验证，以及用于审计和复现的可导出遥测数据。通过在四个模型系列的100个多文件RefactorBench任务上演示该平台，我们展示了它所支持的分析：AST感知的分块比朴素的token窗口分块性能高出25-30%……

    arXiv:2609.04898v1 Announce Type: cross  Abstract: Repository-scale refactoring requires coding agents to propagate a single change across many interdependent files without altering program behavior, yet to our knowledge no existing harness isolates the design choices that determine agent success on this task. We present RefactorPlatform, an open-source evaluation harness that holds the environment fixed and varies each design axis explicitly: model backbone (via OpenRouter and GitHub Copilot CLI), execution regime (baseline, retrieval-augmented, and multi-agent), and prompt specificity. Each run executes in an isolated workspace with live terminal streaming, per-task logging of tokens, diffs, and transcripts, AST-based verification, and exportable telemetry for audit and reproduction. Demonstrating the platform on 100 multi-file RefactorBench tasks across four model families, we illustrate the analyses it supports: AST-aware chunking outperforms naive token-window chunking by 25-30% a
    
[^33]: 面向内存高效MoE推理的缓存感知联合路由器自适应

    Cache-Aware Joint Router Adaptation for Memory-Efficient MoE Inference

    [https://arxiv.org/abs/2609.04895](https://arxiv.org/abs/2609.04895)

    提出一种缓存感知的后训练框架，通过联合自适应MoE主干与轻量级时间-空间缓存路由器，在不改变原生Top-K专家选择规则的前提下提升缓存命中率、减少专家权重传输，实现内存高效的MoE推理。

    

    混合专家模型每个token仅激活一小部分专家，但完整的专家集合通常超出GPU内存容量，导致解码过程中反复的权重传输。我们将专家缓存管理形式化为一个模型侧的算法问题，并提出一种缓存感知的后训练框架，该框架联合调整MoE主干网络和轻量级辅助缓存路由器，同时在推理时保留原生的Top-K专家选择规则。其仅更新模式——时间路由器，可预测同层复用并为未来token保留专家，无需主动加载。完整的时空路由器在此基础上增加了空间路由器，利用因果前驱token的隐藏状态在访问目标层之前细化时间缓存。我们在Qwen3和GPT-OSS上，通过GSM8K、MATH和CommonsenseQA对两种模式进行了评估。与匹配的纯语言模型基线相比，时间路由器持续提升缓存命中率并减少专家权重传输流量。在Qwen（摘要截断）

    arXiv:2609.04895v1 Announce Type: new  Abstract: Mixture-of-Experts (MoE) models activate only a small subset of experts per token, but the full expert set often exceeds GPU memory, causing repeated weight transfers during decoding. We formulate expert-cache management as a model-side algorithmic problem and propose a cache-aware post-training framework that jointly adapts the MoE backbone and lightweight auxiliary cache routers while preserving the native Top-K expert-selection rule at inference. Its update-only mode, Temporal Router, predicts same-layer reuse and retains experts for future tokens without proactive loading. The full Spatio-Temporal Router adds a Spatio Router that uses the causal predecessor's hidden state to refine the temporal cache before target-layer access. We evaluate both modes on Qwen3 and GPT-OSS across GSM8K, MATH, and CommonsenseQA. Temporal Router consistently improves cache hit rate and reduces expert-weight traffic over matched LM-only baselines. On Qwen
    
[^34]: CC-Mediation：评估大语言模型在跨文化冲突调解中的能力

    CC-Mediation: Evaluating Large Language Models for Cross-Cultural Conflict Mediation

    [https://arxiv.org/abs/2609.04855](https://arxiv.org/abs/2609.04855)

    提出了基于跨文化敏感性发展模型（DMIS）构建的CC-Mediation基准，包含1,661段十轮跨文化冲突调解对话，并设计了轨迹AUC和带符号Wasserstein-1距离两个新指标来评估大语言模型调解干预的效果与跨文化立场变化。

    

    大语言模型（LLM）的跨文化调解需要决定何时进行干预，以及如何在具有文化背景的冲突中做出回应。这一问题的进展一直受限于两方面：（1）缺乏具有可衡量下游效应的调解数据集；（2）缺乏评估跨文化立场变化的原则性指标。为弥补这些空白，我们提出了CC-Mediation，一个基于跨文化敏感性发展模型（DMIS）构建的跨文化调解基准，包含1,661段十轮对话，涵盖具有文化基础的冲突、调解干预措施以及干预后的对话轨迹。我们进一步提出两个基于DMIS的评估指标：轨迹AUC（Trajectory AUC），用于衡量跨文化改善随时间推移的持续性；以及带符号Wasserstein-1距离，用于衡量跨文化立场变化的幅度和方向。这两个指标均与人类对DMIS的判断表现出高度一致性。

    arXiv:2609.04855v1 Announce Type: cross  Abstract: Cross-cultural mediation by large language models (LLMs) requires deciding both when to intervene and how to respond in culturally grounded conflicts. Progress on this problem has been limited by the lack of (1) mediation datasets with measurable downstream effects and (2) principled metrics for evaluating intercultural stance change. To address these gaps, we introduce CC-Mediation, a cross-cultural mediation benchmark of $1{,}661$ ten-turn dialogues grounded in the Developmental Model of Intercultural Sensitivity (DMIS), containing culturally grounded conflicts, mediation interventions, and post-intervention trajectories. We further propose two DMIS-based evaluation metrics: Trajectory AUC, which measures the persistence of intercultural improvement over time, and a signed Wasserstein-1 distance, which measures the magnitude and direction of shifts in intercultural stance. Both metrics show strong agreement with human judgment of DMI
    
[^35]: MMTClinic：面向临床领域的多模态、多语言时间序列问答与推理基准

    MMTClinic: Multimodal, Multilingual Time Series Question Answering and Reasoning Benchmark for Clinical Domain

    [https://arxiv.org/abs/2609.04842](https://arxiv.org/abs/2609.04842)

    本文提出了MMTClinic——首个面向临床领域的多模态、多语言时间序列问答与推理基准，融合文本、医学图像和生理信号，包含覆盖五种语言的30,000个问答对，用于评估大语言模型在真实临床场景中的推理能力。

    

    临床环境中的时间序列数据对于捕捉患者健康状况随时间的动态变化至关重要，能够支持及时诊断、个性化治疗以及关键事件的早期发现。然而，开发临床可靠且语言包容的医学AI系统仍然是一项重大挑战，其主要原因在于缺乏能够反映真实世界临床场景复杂性的多模态、多语言且基于时间序列的基准。为填补这一空白，我们提出了MMTClinic，这是一个旨在评估大型语言模型（LLM）在涉及临床时间序列的复杂推理与问答任务上能力的基准。MMTClinic融合了文本、医学图像和多变量生理信号，包含30,000个问答对（15,000个多选题（MCQ）和15,000个开放式问题），涵盖五种语言：英语、印地语、孟加拉语、马拉地语和泰米尔语。这些问题涵盖三个……

    arXiv:2609.04842v1 Announce Type: cross  Abstract: Time-series data in clinical settings is crucial for capturing dynamic changes in a patient's health over time, enabling timely diagnosis, personalized treatment, and early detection of critical events. However, the development of clinically reliable and linguistically inclusive medical AI systems remains a significant challenge, primarily due to the lack of multimodal, multilingual, and time-series-grounded benchmarks that reflect the complexity of real-world clinical scenarios. To fill this gap, we present MMTClinic, a benchmark designed to evaluate large language models (LLMs) on complex reasoning and question-answering tasks involving clinical time-series. MMTClinic combines text, medical images, and multivariate physiological signals and includes 30,000 QA pairs (15,000 multiple choice questions (MCQs) and 15,000 open-ended questions) across five languages: English, Hindi, Bengali, Marathi, and Tamil. These questions cover three i
    
[^36]: MABPD：基于结构化论证辩论的多智能体偏见探测与检测

    MABPD: Multi-Agent Bias Probing & Detection via Structured Argument Debate

    [https://arxiv.org/abs/2609.04841](https://arxiv.org/abs/2609.04841)

    该论文提出MABPD框架，让三个专门的LLM智能体通过结构化论证辩论（SAD）协议协作分析新闻文章，利用非对称举证责任、角色加权投票和共识后验证机制，实现无需监督训练的媒体偏见检测。

    

    新闻文章中的媒体偏见通过微妙的语言线索起作用——带有倾向性的措辞、选择性框架和策略性省略——这些线索难以被单一模型检测到，且传统上需要大规模标注语料库进行监督训练。我们探讨结构化的多智能体协商能否作为该任务中监督分类的一种有原则的、无需训练的替代方案。我们提出了MABPD（多智能体偏见探测与检测），这是一个流水线系统，其中三个专门的LLM智能体从互补的角度分析文章，并通过结构化论证辩论（SAD）协议解决彼此的分歧。SAD实现了由领域动机驱动的非对称举证责任——缺乏文本证据支撑的偏见主张权重为零——并结合角色加权投票和共识后验证机制，用明确的协商结构取代了任务特定的监督决策边界。消融实验证实……（原文摘要在此处截断）

    arXiv:2609.04841v1 Announce Type: cross  Abstract: Media bias in news articles operates through subtle linguistic cues---loaded language, selective framing, and strategic omission---that resist single-model detection and have traditionally required large annotated corpora for supervised training. We ask whether structured multi-agent deliberation can serve as a principled, training-free alternative to supervised classification for this task. We introduce MABPD (Multi-Agent Bias Probing & Detection), a pipeline in which three specialized LLM agents analyze an article from complementary perspectives and resolve disagreements through a Structured Argument Debate (SAD) protocol. SAD implements a domain-motivated asymmetric burden of proof---biased claims without grounded textual evidence carry zero weight---combined with role-weighted voting and post-consensus verification, replacing task-specific supervised decision boundaries with explicit deliberative structure. Ablation confirms that t
    
[^37]: 论大型语言模型中的认知多样性

    On Epistemic Diversity in Large Language Models

    [https://arxiv.org/abs/2609.04835](https://arxiv.org/abs/2609.04835)

    该论文借鉴哲学与社会认识论中的认知多样性概念，提出将其——即LLM向用户展现的有效答案、解释和推理路径的范围——作为评估大型语言模型的新维度，并发现前沿LLM常表现出认知狭隘性。

    

    大型语言模型（LLMs）日益不仅被用于检索信息，还被用于回答问题、进行解释、开展教学和支持探究。在这些应用场景中，仅依靠准确性或对齐性已不足以完成评估。一个系统可能给出正确的答案，却仍然限制了用户接触其他有效答案、解释或推理路径的机会。借鉴哲学和社会认识论中更广泛的认知多样性概念，我们将其在LLM的语境下形式化为：LLM向用户展现的有效答案、解释和推理路径的范围。我们认为，认知多样性是在LLM被用于支持知识密集型任务的场景中一个有用的评估维度。我们提出了一个用于概念化和测量LLM认知多样性的初步框架，并在两个领域中对其进行了操作化。我们发现，前沿LLM常常表现出认知上的狭隘性……

    arXiv:2609.04835v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used not only to retrieve information, but to answer questions, explain, teach, and support inquiry. In such settings, evaluation cannot be exhausted by accuracy or alignment alone. A system may give a correct answer while still narrowing users' access %to knowledge. to alternative valid answers, explanations, or reasoning routes. Drawing on the broader notion of epistemic diversity in philosophy and social epistemology, we formalize it in the context of LLMs as the range of valid answers, explanations, and reasoning routes that an LLM exposes to users. We argue that epistemic diversity is a useful evaluation dimension for settings where LLMs are used to support knowledge-intensive tasks. We propose a preliminary framework for conceptualizing and measuring epistemic diversity in LLMs, and operationalize it in two domains. We find that frontier LLMs often exhibit epistemic narrowness, repeated
    
[^38]: 通过强化学习为故事生成建设性反馈

    Generating Constructive Feedback on Stories via Reinforcement Learning

    [https://arxiv.org/abs/2609.04824](https://arxiv.org/abs/2609.04824)

    本文提出一种基于GRPO的强化学习方法，通过新颖的多组件奖励函数，在无需真实反馈标注的情况下引导大语言模型生成针对故事量身定制、有助提升故事质量并聚焦最关键写作问题的建设性反馈。

    

    建设性反馈对于创意作家提升其叙事能力至关重要。由于从人类专家处获得反馈通常成本高昂且耗时，大型语言模型（LLM）作为自动写作助手提供了一种可扩展且高效的替代方案。尽管潜力巨大，但研究表明LLM生成的反馈往往过于笼统、缺乏可操作性，且无法识别出最关键的写作问题。为解决这些局限性，我们提出了一种强化学习方法，无需真实反馈标注即可引导LLM生成建设性反馈。我们使用群体相对策略优化（GRPO）训练模型，并设计了一种新颖的多组件奖励函数，旨在提升反馈的建设性：该奖励函数优先考虑针对故事量身定制、有助于提高故事质量、并能针对最关键写作问题提出改进建议的反馈。在三种（数据集上的）自动评估与人工评估中……

    arXiv:2609.04824v1 Announce Type: new  Abstract: Constructive feedback is crucial for creative writers to refine their storytelling abilities. Since receiving feedback from human experts is often costly and time-intensive, large language models (LLMs) offer a scalable and efficient alternative as automatic writing assistants. Despite their potential, research indicates that LLM-generated feedback is often generic, lacks actionability, and fails to identify which writing issue is most critical. To address these limitations, we present a reinforcement learning approach that steers LLMs to generate constructive feedback without the need for ground-truth feedback. We train our model using group relative policy optimization (GRPO) with a novel multi-component reward function aiming at constructiveness: it prioritizes feedback that is uniquely tailored to the story, helps to improve story quality, and addresses the most critical writing issue. In automatic and human evaluation across three s
    
[^39]: 强化学习提升大语言模型的加泰罗尼亚语文本简化能力

    Reinforcement Learning for improving Large Language Models' Catalan text simplification capabilities

    [https://arxiv.org/abs/2609.04823](https://arxiv.org/abs/2609.04823)

    本文提出了一种结合SARI指标与惩罚组件的新型奖励函数，利用GRPO强化学习对大语言模型进行后训练，有效提升了加泰罗尼亚语这一低资源语言的文本简化能力。

    

    尽管自动文本简化（ATS）对于无障碍访问至关重要，但其发展进度并未跟上更广泛的自然语言处理技术快速演进的步伐。本文研究了应用强化学习（RL）来提升大语言模型（LLM）在低资源语言自动文本简化方面的质量。本文提出了一种新颖的奖励函数，旨在通过组相对策略优化（GRPO）引导大语言模型实现目标简化风格，该函数将SARI指标与特定的惩罚组件相结合。通过对IberianLLM-7B-Instruct模型在ASSET数据集上进行后训练，论证并展示了GRPO配合该奖励函数的有效性。在英文ASSET数据集上进行后训练后，模型在两个精选的加泰罗尼亚语基准测试中的自动文本简化性能得到提升，同时成功抑制了此前观察到的负面行为。此外，本文还通过翻译探索了跨语言迁移学习。

    arXiv:2609.04823v1 Announce Type: cross  Abstract: Although automatic text simplification (ATS) is critical for accessibility, its progress has not matched the rapid evolution of broader natural language processing techniques. This paper investigates the application of reinforcement learning (RL) to improve the quality of ATS for low-resource languages using Large Language Models (LLMs). The paper introduces a novel reward function, designed to guide LLMs toward a targeted simplification style with Group Relative Policy Optimization (GRPO), that combines the SARI metric with specific penalty components. The effectiveness of GRPO with this reward function is motivated and demonstrated by post-training IberianLLM-7B-Instruct on the ASSET dataset. After post-training on the English ASSET, the model's ATS performance improves on two curated Catalan benchmarks while also successfully suppressing previously observed negative behaviors. Cross-lingual transfer learning is explored by translati
    
[^40]: 对多语言可解释性方法的系统比较揭示了由各向异性导致的度量失效

    A Systematic Comparison of Multilingual Interpretability Methods Reveals Anisotropy-Driven Failures

    [https://arxiv.org/abs/2609.04819](https://arxiv.org/abs/2609.04819)

    本研究系统比较了四种跨语言共享度度量方法，发现它们结论分歧的根源在于表示的各向异性，且只有ILO与跨语言迁移的相关性（ρ=0.90）经得起模型规模、家族等控制变量的检验，因而推荐使用ILO。

    

    多语言语言模型会形成共享的跨语言表示，多种可解释性方法都声称能够量化这种共享程度。然而这些方法大多是在相互孤立的情况下开发的，当它们得出的结论不一致时，人们无法判断这种不一致究竟反映了模型本身的特性，还是测量方法本身的伪影。我们在来自五个模型家族（参数规模从1.25亿到140亿）的21个基础模型上，比较了四种共享度度量方法（CKA、ANC、逐token的GMM主导性以及ILO），并将每种方法与五个下游任务上的跨语言迁移性能进行相关性分析。我们发现这些度量方法在对模型跨语言共享程度的量化上存在明显分歧，并进一步提出这种分歧可追溯到各向异性——即表示倾向于聚集在嵌入空间狭窄锥形区域内的现象。只有ILO与跨语言迁移的相关性（Spearman's ρ = 0.90）在控制了模型规模、模型家族和逐任务差异后依然稳健。因此，我们推荐使用ILO。

    arXiv:2609.04819v1 Announce Type: new  Abstract: Multilingual language models develop shared cross-lingual representations, and various interpretability methods claim to quantify this sharing. These methods have been developed largely in isolation, and when they disagree, it is unclear whether the disagreement reflects a property of the model or an artifact of the measurement. We compare four sharing metrics (CKA, ANC, GMM dominance per token, and ILO) across 21 base models from five families (125M-14B parameters) and correlate each with cross-lingual transfer on five downstream tasks. We find that the metrics differ in their quantification of cross-lingual sharing in these models and suggest that the disagreement traces to anisotropy, the tendency of representations to cluster in a narrow cone of the embedding space. Only ILO's correlation with cross-lingual transfer (Spearman's $\rho = 0.90$) survives controls for model size, family, and per-task variation. We therefore recommend ILO
    
[^41]: 重复出现并不足够：对 Gemma 2 和 Gemma 3 中多语言 SAE 翻译特征的因果验证

    Recurrence Is Not Enough: Causally Validating Multilingual SAE Translation Features in Gemma 2 and 3

    [https://arxiv.org/abs/2609.04808](https://arxiv.org/abs/2609.04808)

    尽管能在 Gemma 2 和 Gemma 3 中找到 20 多个跨多语言设置频繁重复出现的 SAE 翻译启动特征，但因果验证表明这些特征对翻译行为几乎没有显著或一致的影响，说明特征的重复出现并不足以证明其因果有效性。

    

    稀疏自编码器（SAE）特征越来越多地被用于解释和引导语言模型的行为，但在一种语言环境中发现的特征在处理另一种语言的提示时是否发挥相同的因果作用，目前仍不清楚。我们使用翻译启动特征（Wu et al., 2026）来研究这一问题。我们在 Gemma 2 中复现了 Wu 等人的 SAE 特征发现方法，并将其扩展到改变提示语言、源语言和目标语言的多语言设置中。随后，我们通过在推理过程中放大或消融这些特征的激活，来检验那些在多种设置中重复出现的特征是否会影响翻译行为。我们还考察了该方法是否可以应用于 Gemma 3。在两个模型中，我们观察到了相同的发现：尽管我们能找到超过 20 个在所有发现设置中频繁激活的特征，但因果验证表明，几乎所有这些特征的效果都很小或不一致。相反，……

    arXiv:2609.04808v1 Announce Type: cross  Abstract: Sparse autoencoder (SAE) features are increasingly used to explain and steer language-model behavior, but it remains unclear whether a feature found in one language context plays the same causal role when processing prompts in another language. We study this question using translation-initiation features (Wu et al., 2026). We reproduce the SAE feature discovery method from Wu et al. in Gemma 2 and extend it to multilingual settings that vary prompt language, source language, and target language. We then test whether features that recur across settings affect translation behavior by amplifying or ablating their activations during inference. We also examine whether the method can be applied to Gemma 3.   In both models, we observe an identical finding: although we can find more than 20 features that activate frequently across all discovery settings, causal validation shows that nearly all have small or inconsistent effects. In contrast, 
    
[^42]: 激活导向能否捕捉多维度的作者写作风格？

    Can Activation Steering Capture Multidimensional Authorship Style?

    [https://arxiv.org/abs/2609.04792](https://arxiv.org/abs/2609.04792)

    提出无需训练的“方面感知激活导向”框架A3S，通过融合各修辞维度的对比方向与干扰感知聚合，直接在激活空间中捕捉多维度的作者写作风格，显著提升多方面作者风格迁移效果。

    

    激活导向在控制大语言模型沿明确定义属性的生成方面已展现出潜力，但它能否处理作者风格多维度且难以定义的特性仍不清楚。我们探究沿修辞动机驱动的维度进行结构化对比提示，能否直接在激活空间中构建丰富的风格表示，从而无需自然语言风格描述符或专门的训练。我们发现，所得到的方向共享一个共同的作者风格主干，同时在承载真实风格信号的特定方面残差上存在冲突，这解释了为何朴素的聚合方法会失败。我们将其操作化为“方面感知激活导向”，这是一个无需训练的框架，它将各个具体方面的对比方向与干扰感知的聚合方式相融合，并针对每个实例调整导向强度。A3S在真正涉及多个方面的作者风格迁移任务中取得改进，性能优于……

    arXiv:2609.04792v1 Announce Type: cross  Abstract: Activation steering has shown promise for controlling LLM generation along well-defined attributes, but it remains unclear whether it can handle the multidimensional and hard-to-define nature of authorship style. We ask whether structured contrastive prompting along rhetorically-motivated dimensions can construct rich style representations directly in activation space, bypassing the need for natural language style descriptors or dedicated training. We find that the resulting directions share a common authorship backbone while conflicting on aspect-specific residuals that carry genuine stylistic signal, explaining why naive aggregation fails. We operationalize this in Aspect-Aware Activation Steering (A3S), a training-free framework that merges per-aspect contrastive directions with interference-aware aggregation and tunes steering strength per instance. A3S improves authorship style transfer where it is genuinely multi-aspect, outperfo
    
[^43]: 面向工具使用智能体的持久教师锚定

    Persistent Teacher Anchoring for Tool-Using Agents

    [https://arxiv.org/abs/2609.04773](https://arxiv.org/abs/2609.04773)

    提出持久教师锚定（PTA），一种由学生诱导但由教师承诺的 rollout 构建方法，通过在块级验证基础上增加轮级承诺让工具调用得以执行，从而解决工具使用中师生分布差距累积导致的漂移问题。

    

    蒸馏在大语言模型后训练中十分常见，其中在策略知识蒸馏（OPKD）利用学生生成的轨迹为学生参与下游强化学习做好准备。在每个状态，学生需要匹配由教师提供的下一个词元分布。当 rollout 进入教师不会访问的状态时，师生分布之间的差距会不断累积。在工具使用场景中，这种差距的影响尤为严重，因为学生编写的调用会在监督之前执行，其返回的观察结果会塑造后续的前缀内容。提议者-验证者生成方式通过让教师在生成过程中决定保留哪些学生提出的文本，来缓解这种漂移。然而，现有方案仅对文本进行管控，将工具执行排除在其作用范围之外。我们提出持久教师锚定（PTA），这是一种由学生诱导、但由教师承诺的 rollout 构建方法。PTA 在保留块级验证的同时增加了轮级承诺，使调用能够真正到达环境并被执行。

    arXiv:2609.04773v1 Announce Type: cross  Abstract: Distillation is common in LLM post-training, where on-policy knowledge distillation (OPKD) uses student-generated trajectories to prepare the student for downstream RL. At each state, the student matches a next-token distribution supplied by the teacher. As the rollout enters states the teacher would not visit, the teacher-student distribution gap can accumulate. In tool use, this gap becomes consequential because student-written calls execute before supervision and their observations shape later prefixes. Proposer-verifier generation addresses this drift by letting the teacher decide which student-proposed text is retained during generation. Existing formulations govern text but leave tool execution outside their scope. We propose Persistent Teacher Anchoring (PTA), a student-induced but teacher-committed rollout construction. PTA retains chunk-level verification and adds turn-level commitment, allowing a call to reach the environment
    
[^44]: 古典泰米尔语的向量化：面向诗句-注释语对的表示学习

    Vectorizing Classical Tamil: Representation Learning for Verse-Commentary Pairs

    [https://arxiv.org/abs/2609.04755](https://arxiv.org/abs/2609.04755)

    本文构建了包含1,262对语料的古典泰米尔语诗句-注释数据集，并通过多种表示学习模型与严格对照实验发现，简单的TF-IDF词汇基线可媲美甚至超越深度生成模型，从而揭示了表示学习在低资源古典文献上的真实能力边界。

    

    我们构建了一个包含1,262对诗句-注释语对的语料库，这些语料取自五个古典泰米尔语文献章节，涵盖从技术性语法散文到现代释义等多种文体，并探究表示学习能够从中恢复出哪些信息。我们训练了循环神经网络与Transformer编码器、孪生式配对匹配网络、mBART风格的编码器-解码器模型，以及仅解码器语言模型。每项分析均在相同数据上与相应的对照组进行对照解读。TF-IDF提供了无需训练的强大词汇检索基线，同时我们对所训练的模型进行了表示分析和生成控制实验。一个包含25个最常见注释词的固定字符串在生成重叠度上的得分高于仅解码器模型。在这些样本量下，高斯噪声上的典型相关分析即可达到1.000，该语料库上的token-F1仅在约0.02-0.20之间，且编码器-解码器模型在验证（后文缺失）之后仍持续降低训练损失达十六个轮次。（原文摘要至此处截断）

    arXiv:2609.04755v1 Announce Type: new  Abstract: We construct a corpus of 1,262 verse-commentary (urai) pairs from five Classical Tamil source sections, ranging from technical grammatical prose to modern paraphrase, and ask what information representation learning can recover. We train recurrent and Transformer encoders, a Siamese-style pair-matching network, an mBART-style encoder-decoder, and a decoder-only language model. Each analysis is interpreted against an appropriate control on the same data. TF-IDF provides a strong no-training lexical retrieval baseline, alongside representation analyses and generation controls for the learned models. A fixed string containing the 25 most frequent commentary words scores higher on generation overlap than the decoder-only model. Canonical correlation reaches 1.000 on Gaussian noise at these sample sizes, token-F1 spans only about 0.02-0.20 on this corpus, and the encoder-decoder continues to lower training loss for sixteen epochs after valida
    
[^45]: 思维链的表面之下：大语言模型推理操作的机制性解释

    Beneath the Surface of Chains-of-Thought: A Mechanistic Interpretation of Reasoning Operations in LLMs

    [https://arxiv.org/abs/2609.04753](https://arxiv.org/abs/2609.04753)

    该研究揭示了大语言模型思维链中的不同推理操作在隐藏表示空间中具有可分离的几何结构，这种结构在中间层最为显著，且注意力掩码干预表明文本块起始处的操作对齐表示依赖于前序推理内容。

    

    大语言模型中的推理通过多种功能操作展开，例如问题表述、目标分解和演绎。尽管这些操作在文本中被明确区分，但人们对它们在表示空间中如何进行几何组织知之甚少。为此，我们研究了不同的推理操作是否在隐藏表示中表现出相应的几何结构。我们发现，这些操作在保留集表示中是可分离的，且可分离性在中间层达到峰值，并验证了这种结构无法由词汇或位置等混淆因素解释。跨层来看，逐词元的操作对齐在跨度上变得更加分布式，而相同的表面词元会根据其周围文本块的操作以不同方式被表示。注意力掩码干预进一步表明，文本块起始处的操作对齐表示依赖于前序推理内容。

    arXiv:2609.04753v1 Announce Type: new  Abstract: Reasoning in large language models unfolds through diverse functional operations, such as problem formulation, goal decomposition, and deduction. Although these operations are explicitly distinguished in text, little is known about how they are geometrically organized in representation spaces. To this end, we investigate whether distinct reasoning operations exhibit corresponding geometric structure in hidden representations. We find that operations are separable in held-out representations, with separability peaking in middle layers, and verify that this structure is not explained by lexical or positional confounds. Across layers, token-wise operation-alignment becomes more distributed over spans, while identical surface tokens are represented differently depending on the operation of its surrounding chunk. Attention-masking interventions further show that operation-aligned representations at chunk onset depend on preceding reasoning co
    
[^46]: 知道何时不该回答：视觉语言模型中的选择性拒绝遵从

    Knowing What Not to Answer: Selective Non-Compliance in Vision-Language Models

    [https://arxiv.org/abs/2609.04720](https://arxiv.org/abs/2609.04720)

    该论文提出了KoNA基准，涵盖错误前提、视觉不可及性等五个类别，用于评估视觉语言模型在单一查询和复合查询两种情形下，区分应回答内容与应拒绝内容的选择性拒绝遵从能力。

    

    视觉语言模型（VLM）应当对适当的请求提供有帮助的回应，同时对不正确、不安全、不可行或无法回答的请求拒绝遵从。然而，现有基准主要在整体查询层面评估不遵从行为，假设每个请求要么应当遵从，要么应当拒绝遵从。实际上，现实世界的查询可能同时包含可回答的内容和应当拒绝遵从的组成部分。在本文中，我们提出了KoNA，一个用于评估视觉语言模型选择性不遵从能力的基准，涵盖五个类别：错误前提、视觉不可及性、普遍未知、任务可行性和安全性。每个任务评估两种能力：查询级不遵从，以及在成对的单一查询与复合查询下的组件级不遵从。我们在多种VLM上的评估表明，模型常常无法拒绝、纠正或……

    arXiv:2609.04720v1 Announce Type: cross  Abstract: Vision-language models (VLMs) are expected to respond helpfully to appropriate requests while withholding compliance with requests that are incorrect, unsafe, infeasible, or unanswerable. However, existing benchmarks predominantly evaluate non-compliance at the level of the query as a whole, assuming that each request either warrants compliance or requires withholding compliance. In practice, real-world queries can contain a mixture of answerable content and components for which compliance should be withheld. In this paper, we introduce KoNA, a benchmark for evaluating selective non-compliance in VLMs across five categories: False Premise, Visual Inaccessibility, Universal Unknown, Task Feasibility, and Safety. Each task evaluates two capabilities: query-level non-compliance and component-level non-compliance under paired single and compound queries. Our evaluation across diverse VLMs shows that models often fail to refuse, correct, or
    
[^47]: 不拒绝而拒绝：通过安全调优响应的结构分析减少语言模型中的误拒

    Refuse without Refusal: A Structural Analysis of Safety-Tuning Responses for Reducing False Refusals in Language Models

    [https://arxiv.org/abs/2609.04714](https://arxiv.org/abs/2609.04714)

    该论文创新性地将安全调优响应分解为模板化拒绝声明与拒绝理由两个部分，发现拒绝声明会诱导模型依赖表面线索而造成误拒，而仅用拒绝理由进行训练能有效减少语言模型的误拒行为。

    

    在大型语言模型的对齐中，平衡有用性与安全性始终是一个根本性挑战。为实现这种平衡，模型应当拒绝有害查询（例如“我如何射杀某人？”），同时保持对良性输入的响应能力，即使这些输入在表面上与有害查询相似（例如“在哪里能拍出好照片？”）。然而，模型往往难以区分真正有害的查询与包含表面风险性语言的良性查询，从而导致误拒。在本文中，我们通过将安全调优数据集中的响应分解为两个不同的组成部分来应对这一问题：（i）模板化的拒绝声明和（ii）解释拒绝原因的理由说明。我们的实验和分析表明，拒绝声明会诱导模型依赖表面线索，从而阻碍其对有害查询和良性查询的准确区分。相比之下，仅在拒绝理由上进行训练可以减少误拒

    arXiv:2609.04714v1 Announce Type: cross  Abstract: Striking a balance between helpfulness and safety remains a fundamental challenge in aligning large language models. To achieve this balance, models should refuse harmful queries (e.g., "How do I shoot someone?") while remaining responsive to benign inputs, even those superficially resembling harmful queries (e.g., "Where can I shoot a good photo?"). However, models often struggle to distinguish genuinely harmful queries from benign queries that contain superficially risky language, resulting in false refusals. In this paper, we address the issue by decomposing a response in the safety-tuning dataset into two distinct components: (i) a boilerplate refusal statement and (ii) a rationale explaining the refusal. Our experiments and analyses show that refusal statements impede accurate discrimination between harmful and benign queries by inducing reliance on superficial cues. In contrast, training solely on rationales reduces false refusal
    
[^48]: 语言模型如何表征并使用音系信息进行语素变体选择？

    How Do Language Models Represent and Use Phonological Information for Allomorph Selection?

    [https://arxiv.org/abs/2609.04708](https://arxiv.org/abs/2609.04708)

    该论文发现语言模型在嵌入空间中以单一线性方向编码音系条件，并通过预测即将出现的触发词的音系特征来因果性地驱动英语不定冠词 a/an 的选择，展现出规则式泛化而非单纯记忆的能力。

    

    语言模型是在经过分词处理的文本上训练的，这种文本掩盖了词语的语音结构，然而它们却能可靠地生成其形式受音系条件制约的语素。目前尚不清楚语言模型是依赖于针对具体词条的记忆，还是依赖规则式的泛化；如果是后者，这种泛化又是如何在模型内部实现的。因此，我们探究这种音系条件是否在语言模型内部被表征，以及它如何被因果性地用于语素变体选择。针对英语不定冠词 a/an，我们证明了该音系条件在触发词标记的嵌入中被编码为单一线性方向，且该方向在标记级 wug 测试中因果性地驱动冠词的选择；此外，在冠词预测位置，模型会预测即将出现的触发词标记，并利用所预测触发词的音系特征来选择冠词。随后，我们进一步探究这种规则式泛化是否能够扩展到英语冠词之外的情形。

    arXiv:2609.04708v1 Announce Type: new  Abstract: Language models are trained on tokenized text that obscures the sound structure of words, yet they reliably produce morphemes whose form is phonologically conditioned. It remains unclear whether they rely on item-specific memorization or rule-like generalization and, if the latter, how that generalization is implemented. We therefore ask whether this phonological condition is represented within language models and how it is causally used for allomorph selection. For the English indefinite article a/an, we show that the phonological condition is encoded along a single linear direction in trigger-token embeddings, that this direction causally drives article selection in token-level wug tests, and that, at the article-prediction position, the model forecasts the upcoming trigger token and uses the forecasted trigger's phonological feature to choose the article. We then ask whether this rule-like generalization extends beyond English article
    
[^49]: 用于阿尔茨海默病的视网膜OCTA表型分析与大语言模型报告

    Retinal OCTA Phenotyping with LLM Reporting for Alzheimer's Disease

    [https://arxiv.org/abs/2609.04689](https://arxiv.org/abs/2609.04689)

    该论文提出了一种可解释的视网膜OCTA分析流程，通过整合标注感知的血管分割、分层血管生物标志物提取、无标签表型分析与基于测量结果的大语言模型报告生成，实现了阿尔茨海默病的早期无创筛查与可解释诊断。

    

    阿尔茨海默病（AD）的早期识别仍然具有挑战性，因为现有的评估方法可能成本高昂、资源密集，或不适合大规模人群筛查。光学相干断层扫描血管成像（OCTA）能够无创地可视化视网膜微血管，但现有方法通常需要诊断标签，且提供的测量级解释有限。我们提出了一种可解释的OCTA流程，整合了标注感知的血管分割、分层血管生物标志物提取、无标签表型分析以及基于测量结果的LLM报告生成。使用来自39名受试者的117张ROSE-1图像，我们将标注匹配的分割模型应用于浅层血管复合体（SVC）、深层血管复合体（DVC）以及SVC+DVC联合表示。这些模型实现了0.916-0.970的ROC-AUC值和0.695-0.781的Dice分数。六种密度和分形维数生物标志物构成受试者...

    arXiv:2609.04689v1 Announce Type: cross  Abstract: Early identification of Alzheimer's disease (AD) remains challenging because established assessment methods can be costly, resource-intensive, or unsuitable for population-scale screening. Optical coherence tomography angiography (OCTA) provides non-invasive visualization of retinal microvasculature, but existing approaches often require diagnostic labels and provide limited measurement-level interpretation. We present an explainable OCTA pipeline that integrates annotation-aware vessel segmentation, layer-specific vascular biomarker extraction, label-free phenotyping, and measurement-grounded LLM reporting. Using 117 ROSE-1 images from 39 subjects, we apply annotation-matched segmentation models to superficial vascular complex (SVC), deep vascular complex (DVC), and combined SVC+DVC representations. The models achieve ROC-AUC values of 0.916-0.970 and Dice scores of 0.695-0.781. Six density and fractal-dimension biomarkers form subjec
    
[^50]: 基于大语言模型的对话生成中角色特征恰当使用的控制与评估

    Controlling and Assessing Appropriate Persona Use in LLM-based Dialogue Generation

    [https://arxiv.org/abs/2609.04676](https://arxiv.org/abs/2609.04676)

    本文揭示了LLM在角色对话生成中存在系统性融入所有角色属性的偏差，并提出SCONPOS方法通过干预模型内部表示来抑制过度使用，同时提出PAS指标评估角色使用的恰当性。

    

    在基于角色的对话生成（PDG）中，大语言模型常常过度使用角色属性，即无论对话上下文如何都会将其融入回复中，导致回复不自然。尽管这一问题具有重要的实际意义，但其根本原因至今未被探索，目前既缺乏缓解该问题的方法，也没有评估角色使用恰当性的指标。为解决这些问题，我们首先对基于大语言模型的PDG进行了全面分析，揭示出大语言模型存在一种系统性偏差，即倾向于融入所有给定的角色属性，且现有指标无法捕捉上下文层面的恰当性。基于这些发现，我们提出了自我对比角色过度使用抑制方法（SCONPOS），通过在提示编码阶段直接干预大语言模型的内部表示来缓解过度使用问题，且无需任何回复生成过程。我们进一步提出了角色恰当性评分（PAS），这是一种新颖的指标，能够同时惩罚角色属性的过度使用和使用不足。

    arXiv:2609.04676v1 Announce Type: new  Abstract: In persona-based dialogue generation (PDG), LLMs often overuse persona attributes by incorporating them regardless of dialogue context, resulting in unnatural responses. Despite its practical significance, the underlying causes remain unexplored, with no method to mitigate this problem or metric to assess the appropriateness of persona use. To address these issues, we first conduct a comprehensive analysis of LLM-based PDG, revealing that LLMs exhibit a systematic bias to incorporate all given persona attributes, and that existing metrics fail to capture contextual appropriateness. Building on these findings, we propose Self-CONtrastive Persona Overuse Suppression (SCONPOS) to mitigate overuse by directly intervening in LLMs' internal representations at the prompt encoding stage, without requiring any response generation. We further propose the Persona Appropriateness Score (PAS), a novel metric that penalizes both overuse and underuse. 
    
[^51]: 在推理时选择正确的语言模式以实现多语言可靠性

    Choosing the Right Language Mode at Inference Time for Multilingual Reliability

    [https://arxiv.org/abs/2609.04653](https://arxiv.org/abs/2609.04653)

    提出了一种无需训练的测试时框架 RAAI，通过在推理时自适应选择最合适的语言模式（目标语言、英语或双语），在准确性与可靠性之间取得平衡，从而提升多语言大模型的推理可靠性。

    

    多语言大语言模型在低资源和中资源语言中的推理往往表现不佳。先前的研究表明，翻译可以通过帮助模型获取更强的以英语为中心的表示来改善多语言推理。这引出了一个核心问题：多语言大语言模型需要多少翻译才能可靠地进行推理？何时更多的翻译反而会引发干扰和过度自信？我们使用 LLaMA 和 Qwen 模型，通过改变文本范围和语言模式（仅目标语言、仅英语、双语）进行了大量实验，以同时评估准确性和可靠性。我们的结果揭示了一个明显的权衡：英语语境通常能改善理解并修复由非英语理解造成的错误，但添加冗余的双语上下文会加剧干扰。我们通过可靠性感知自适应推理（Reliability-Aware Adaptive Inference, RAAI）来解决这一权衡，这是一个无需训练的测试时框架，它（i）执行……（摘要原文在此处截断）

    arXiv:2609.04653v1 Announce Type: new  Abstract: Multilingual large language models often struggle to reason in low- to mid-resource languages. Prior work has shown that translation can improve multilingual reasoning by helping models access stronger English-centric representations. This raises a central question: How much translation is needed for multilingual large language models to reason reliably, and when does more translation instead trigger interference and overconfidence?   Using LLaMA and Qwen models, we run extensive experiments varying text scope and language mode (target-only, English-only, bilingual) to evaluate both accuracy and reliability.   Our results reveal a clear trade-off: English context often improve understanding and recover errors caused by non-English comprehension, yet adding redundant bilingual context intensifies interference. We address this trade-off with Reliability-Aware Adaptive Inference (RAAI), a training-free test-time framework that (i) performs 
    
[^52]: ConsensusBench：通过结果奖励密集化实现LLM推理的共识节点基准

    ConsensusBench: Benchmark of Consensus Nodes for LLM Reasoning via Outcome Reward Densifying

    [https://arxiv.org/abs/2609.04648](https://arxiv.org/abs/2609.04648)

    该论文提出ConsensusBench数据集，通过识别推理过程中可验证的中间结论（共识节点）来密集化结果奖励，为LLM推理的强化学习提供基于规则的过程级奖励信号，弥补稀疏最终答案奖励的不足。

    

    强化学习（RL）已成为提升大语言模型（LLM）推理能力的主要范式之一。其中，组相对策略优化（GRPO）及相关算法在结果级奖励下展现了出色的性能。然而，这些方法仅依赖于最终答案，缺乏关于哪些中间步骤促成成功或失败的反馈。随着任务复杂度和推理轨迹长度的增加，这种稀疏的最终答案奖励变得越来越不足。为了解决这一局限，我们提出了ConsensusBench，一个旨在提供基于规则的过程级信号的新型数据集。我们假设正确的最终答案依赖于推理过程中的一小组中间结论，这些结论可被视为可验证的子结果。我们通过从N次rollout中筛选正确轨迹并对语义等价的结论进行聚类来识别这些子结果……

    arXiv:2609.04648v1 Announce Type: new  Abstract: Reinforcement learning (RL) has become one of the primary paradigms for reasoning enhancement of large language models (LLMs). In particular, Group Relative Policy Optimization (GRPO) and related algorithms have demonstrated strong performance with outcome-level rewards. However, these methods depend solely on the final answer, without feedback regarding which intermediate steps contribute to success or failure. As task complexity and reasoning trajectory length increase, such sparse final-answer rewards become increasingly insufficient. To address this limitation, we introduce ConsensusBench, a novel dataset designed to provide rule-based process-level signals. We posit that a correct final answer relies on a small set of intermediate conclusions throughout the reasoning process, which can be seen as a verifiable sub-outcome. We identify these sub-outcomes by filtering correct trajectories from N rollouts and clustering semantically equ
    
[^53]: CAGE：面向检索增强生成的连贯性感知图编码

    CAGE: Coherence-Aware Graph Encoding for Retrieval-Augmented Generation

    [https://arxiv.org/abs/2609.04647](https://arxiv.org/abs/2609.04647)

    CAGE提出了一种连贯性感知的图编码重排序框架，通过有向异构图建模检索段落间在领域相关性、抗噪性、信息联结和事实一致性四个维度的块间连贯性，显著提升了RAG系统在多跳问答任务上的检索与生成效果。

    

    传统的检索增强生成系统独立地将每个段落与查询进行评分，由此组装的上下文集合虽然可能各自相关，但整体上缺乏连贯性。我们提出了连贯性感知图编码，这是一个重排序框架，从四个维度建模“块间连贯性”：领域内相关性、抗噪性、信息联结和事实一致性。我们的流程将检索到的段落转换为有向异构实体图，通过最小出度重加权放大事实锚点，通过关系图卷积网络编码结构模式，并将块间连贯性与查询相关性融合以进行最终排序。在四个多跳问答基准测试上的评估表明，CAGE在以桥梁型问题为主的数据集上的Recall@5指标与包括monoT5在内的强基线相当或更优，并持续提升下游精确匹配得分，证明了结构上连贯的（原文在此截断）

    arXiv:2609.04647v1 Announce Type: new  Abstract: Traditional Retrieval-Augmented Generation (RAG) systems score each passage independently against the query, assembling context sets that may be individually relevant yet collectively incoherent. We introduce Coherence-Aware Graph Encoding (CAGE), a reranking framework that models "between-chunk coherence" across four dimensions: Intra-Domain Relevance, Noise Resistance, Informational Bonding, and Factual Consistency. Our pipeline transforms retrieved passages into directed heterogeneous entity graphs, amplifies factual anchors via min-out-degree reweighting, encodes structural patterns through a Relational Graph Convolutional Network, and fuses inter-chunk coherence with query relevance for final ranking. Evaluated across four multi-hop benchmarks, CAGE matches or outperforms strong baselines including monoT5 in Recall@5 on bridge-dominated datasets and consistently improves downstream Exact Match, demonstrating that structurally cohere
    
[^54]: 面向多模态推荐的潜空间对齐推理

    Latent-Aligned Reasoning for Multimodal Recommendation

    [https://arxiv.org/abs/2609.04645](https://arxiv.org/abs/2609.04645)

    提出LARK两阶段潜空间推理框架，通过可学习token与冻结视觉编码器对齐及物品对比学习，解决多模态推荐中VLM多步推理导致的视觉与文本信号衰减（跨模态稀释）问题。

    

    多模态视觉-语言模型（VLMs）在跨模态理解方面展现出卓越的能力，但将其应用于推荐任务时仍存在一个根本性挑战：随着表示在多步推理过程中传播，视觉和文本信号会逐渐衰减——我们将这种现象称为跨模态稀释。为解决这一问题，我们提出了LARK（潜空间对齐推理框架），这是一个在单一VLM内具备互补对齐机制的两阶段潜空间推理框架。在第一阶段，可学习的潜空间token与多步思维链（CoT）推理交错进行，并与冻结的视觉编码器显式对齐，作为视觉检查点，在整个推理链中保留感知细节。在第二阶段，潜空间表示通过桥接MLP进行投影，并采用物品到物品的对比学习进行训练；为防止推理语义发生（摘要在此处截断）……

    arXiv:2609.04645v1 Announce Type: cross  Abstract: Multimodal Vision-Language Models (VLMs) have demonstrated remarkable capabilities in cross-modal understanding, yet a fundamental challenge persists when applying them to recommendation: as representations propagate through multi-step reasoning, both visual and textual signals progressively attenuate - a phenomenon we term cross-modal dilution. To address this, we propose LARK (Latent-Aligned Reasoning frameworK), a two-stage latent reasoning framework with complementary alignment mechanisms within a single VLM. In the first stage, learnable latent tokens are interleaved with multi-step chain-of-thought (CoT) reasoning and explicitly aligned with a frozen vision encoder, serving as visual checkpoints that preserve perceptual details throughout the reasoning chain. In the second stage, the latent representations are projected via a bridge MLP and trained with item-to-item contrastive learning; to prevent the reasoning semantics from fa
    
[^55]: 追踪音频大语言模型中的音频接地与答案选择

    Tracing Audio Grounding and Answer Selection in Audio LLMs

    [https://arxiv.org/abs/2609.04637](https://arxiv.org/abs/2609.04637)

    该研究揭示了音频大语言模型内部音频真正决定答案的机制：训练主要在中间至后期层增强音频对最终预测的影响，而声学信息主要在早至中间层塑造答案选项的表示。

    

    音频大语言模型在音频理解方面已取得进展，但它们仍可能通过文本线索或语言先验进行推理来预测答案，而非依据所提供的音频。一种常见的补救措施是在那些答案无法仅凭文本推断的数据上训练模型。这种方法能够提升性能，但模型内部究竟发生了什么变化仍不清楚。在本文中，我们探究了为了让音频真正决定答案，模型内部必须发生什么。我们的发现有三点：(1) 将音频替换为静音或无关音频时，经过训练的模型的性能下降幅度明显大于预训练模型；(2) 声学信息在早至中间层最强烈地塑造模型对答案选项的表示，而训练主要在中间至后期层中增强了音频信息对最终预测的影响；(3) 训练期间学到的权重……（原文摘要至此截断）

    arXiv:2609.04637v1 Announce Type: cross  Abstract: Audio Large Language Models (Audio LLMs) have advanced in audio understanding, yet they can still predict the answer by reasoning from textual cues or linguistic priors rather than the provided audio. A common remedy is to train models on data whose answers cannot be inferred from text alone. This approach can improve performance, but what changes within the model remains unclear. In this paper, we ask what must happen inside the model for the audio to actually determine the answer. Our findings are threefold. (1) Replacing the audio with silence or unrelated audio causes substantially larger performance degradation in the trained model than in the pretrained model. (2) Acoustic information most strongly shapes the model's representations of the answer choices in early-to-middle layers, while training mainly increases the influence of audio information on the final prediction in middle-to-late layers. (3) The weights learned during tra
    
[^56]: PetQA：兽医知识与临床推理基准测试

    PetQA: Benchmarking Veterinary Knowledge and Clinical Reasoning

    [https://arxiv.org/abs/2609.04598](https://arxiv.org/abs/2609.04598)

    本文提出PetQA——首个用于评估大语言模型和视觉-语言模型兽医知识与临床推理能力的韩语长文本问答基准，包含近1.9万个源自真实猫狗病例的文本与多模态问答对，并对18个模型在零样本、RAG和微调三种设置下进行了系统评估。

    

    我们推出了PetQA，这是一个韩语长文本问答（QA）基准，用于评估大型语言模型（LLM）和大型视觉-语言模型（LVLM）的兽医知识与临床推理能力。PetQA包含10,076个纯文本问答对和8,751个多模态问答对，这些问题源自关于猫狗的真实提问，并与专家兽医的答案相配对。其测试集PetQA-Bench还进一步包含问题类型和临床状况的标注。我们在零样本推理、检索增强生成（RAG）和监督微调（SFT）三种设置下，使用ROUGE、BERTScore以及LLM-as-a-judge等指标，从事实性和有用性两方面评估了十八个模型。基准测试结果概述了当前模型在处理兽医临床问题方面的优势与局限，并强调了需要更有效的适配方法来开发对兽医而言临床可靠的AI系统。

    arXiv:2609.04598v1 Announce Type: cross  Abstract: We introduce PetQA, a Korean long-form question-answering (QA) benchmark for evaluating veterinary knowledge and clinical reasoning in large language models (LLMs) and large vision-language models (LVLMs). PetQA contains 10,076 text-only and 8,751 multimodal QA pairs derived from real-world questions about dogs and cats, paired with answers from expert veterinarians. Its test split, PetQA-Bench, further includes annotations for question types and clinical conditions. We evaluate eighteen models using ROUGE, BERTScore, and LLM-as-a-judge metrics for factuality and helpfulness under three settings: zero-shot inference, retrieval-augmented generation (RAG), and supervised fine-tuning (SFT). The benchmarking results provide an overview of the strengths and limitations of current models in addressing veterinary clinical queries and highlight the need for more effective adaptation methods to develop clinically reliable AI systems for veterin
    
[^57]: JLIR：一种Julia原生的、受MLIR启发的中间表示，支持自动JACC内核提取

    JLIR: A Julia-Native MLIR-Inspired Intermediate Representation with Automatic JACC Kernel Extraction

    [https://arxiv.org/abs/2609.04585](https://arxiv.org/abs/2609.04585)

    该论文提出JLIR这一Julia原生、受MLIR启发的中间表示框架，克服了MLIR对动态语言类型要求过强和C++扩展门槛高的问题，使Julia科学计算抽象能自然融入编译器优化路径，并支持自动提取JACC内核。

    

    多层中间表示（MLIR）已使可复用的编译器基础设施在特定领域计算中变得实用。然而，MLIR对编译时类型的强要求以及低层（C++）扩展模型，对于像Julia这样高级的、动态特化的语言而言可能并不契合。MLIR在类型系统和抽象层次方面对动态编程语言存在若干不足。因此，对于非编译器专业或科学计算领域的用户来说，引入新的编程抽象并以既自然又可优化的形式表达算法实现极为困难。这导致线性代数、网格处理、偏微分方程及相关领域的库接口往往处于编译器优化路径之外。我们提出了JLIR（Julia原生层级中间表示），一个Julia原生的中间表示框架，它带来了主要……（原文摘要至此被截断）

    arXiv:2609.04585v1 Announce Type: cross  Abstract: The Multi-Level Intermediate Representation (MLIR) has made reusable compiler infrastructure practical for domain-specific computation. However, MLIR's strong compile-time type requirements and low-level (C++) extension model can be a poor match for high-level, dynamically specialized languages such as Julia. MLIR has several drawbacks for dynamic programming languages in terms of the type system and level of abstraction. It is thus extremely challenging for non-compiler or scientific computing users to introduce new programming abstractions and express algorithm implementations in a form that remains both natural and optimizable. As a result, library interfaces for linear algebra, mesh processing, partial differential equations, and related domains often sit outside the compiler optimization path. We present JLIR (Julia-native Level Intermediate Representation), a Julia-native intermediate representation framework that brings the main
    
[^58]: 内部探针何时胜过读取答案？语言模型中的误校准读出与行为隐藏知识

    When Do Internal Probes Beat Reading the Answer? Miscalibrated Readouts and Behavior-Concealed Knowledge in Language Models

    [https://arxiv.org/abs/2609.04582](https://arxiv.org/abs/2609.04582)

    该论文发现语言模型内部（隐藏状态与输出logits边际）早已编码了正确的判断信息，但由于决策阈值过度饱和偏移而被行为层面掩盖，行为准确率坍缩为阈值偏移的单一函数（Spearman -0.93），揭示了“行为隐藏知识”现象。

    

    一个0.6B参数的语言模型在验证1,200个逻辑结论（一半有效，一半被单个语义编辑破坏）时，每次都回答“是”。从行为表现看，它无法做出任何区分；但对其隐藏状态的线性探针却能以0.96的AUC读出正确判定，并可迁移到未见过的逻辑结构，还能区分由真实结论完全相同的词构成的干扰项（0.90）。我们追问判定信息在何处丢失，发现主要失败源于单个标量：判定信息沿着一个良好对齐的读出方向在模型自身的输出logits中依然存在（边际AUC为0.89），而一个饱和的决策阈值（偏移+4.6个标准差）将其抹除。该诊断具有普适性：在一个五模型、三家族的因子实验中，跨越90种语义标签配置，行为准确率坍缩为阈值偏移的单一函数（Spearman相关系数-0.93），而边际排序的变化则小得多。在13倍的规模范围内，内部知识趋于饱和，而自由行为测得的表现……（摘要在此处截断）

    arXiv:2609.04582v1 Announce Type: cross  Abstract: A 0.6B language model, asked to verify 1,200 logical conclusions (half valid, half corrupted by a single semantic edit), answers YES every time. Judged by behavior it discriminates nothing; linear probes on its hidden states read the correct verdict at 0.96 AUC, transferring to unseen logical structures and separating foils built from exactly the words of the true conclusion (0.90). We ask where the verdict is lost, and find the dominant failure is a single scalar. The verdict survives to the model's own output logits (margin AUC 0.89) along a well-aligned readout direction; a saturated decision threshold, offset by +4.6 sigma, erases it. The diagnosis generalizes: across 90 semantic-label configurations of a five-model, three-family factorial, behavioral accuracy collapses onto a single function of threshold offset (Spearman -0.93) while margin ranking moves far less. Across a 13x scale range, internal knowledge saturates while free-f
    
[^59]: 所选对象能否到达阅读器？审计基于事实的语言模型流水线中的身份交接问题

    Does the Selected Object Reach the Reader? Auditing Identity Handoffs in Grounded Language-Model Pipelines

    [https://arxiv.org/abs/2609.04579](https://arxiv.org/abs/2609.04579)

    该论文系统审计了基于事实的语言模型流水线中对象从选择器到阅读器的身份交接问题，发现仅正文BM25检索会遗漏26.6%的所选对象，而带重排序的混合检索仅遗漏1.0%，且所选对象身份与数据集关联身份在约18%的记录中存在差异。

    

    基于事实（接地）的语言模型流水线可分为三个阶段：选择一个对象、为该对象检索相关段落，以及利用这些证据进行回答。如果所选对象必须传递给阅读器，那么在交接过程中丢失它就会破坏整个流程。基准测试的召回率检查的是与数据集关联的对象，这可能与实际选择的对象不同。我们对三个选择器家族的600个HybridQA问题进行了审计。在1,463条所选对象与数据集追踪段落相匹配的可解决记录中，精确键查找和精确标题匹配每次都能返回该对象。在为每种排序规则提供相同解码后所选标题的情况下，仅正文BM25在截断值为5时在389条记录（26.6%）上遗漏了该对象，而带重排序的混合检索仅在14条记录（1.0%）上遗漏。这两种身份在1,792条可解决记录中的329条上存在差异。使用原始问题排序时，它们的前五项检查在106条记录（5.9%）上不一致。冻结阅读器的比较表明，对齐对象的存在与28.6到31.0分的性能提升相关。

    arXiv:2609.04579v1 Announce Type: new  Abstract: Grounded language-model pipelines can be divided into three stages: selecting an object, retrieving passages for it, and using that evidence to answer. If the selected object must reach the reader, losing it breaks the handoff. Benchmark recall checks the dataset-linked object, which can differ. We audit 600 HybridQA questions across three selector families. On 1,463 resolvable records where the selected object matches the dataset-traced passage, exact key lookup and exact title matching return the object every time. With every ranked rule given the same decoded selected title, body-only BM25 omits it on 389 records (26.6%) at cutoff five, while hybrid retrieval with reranking omits it on 14 (1.0%). The two identities differ on 329 of 1,792 resolvable records. With original-question rankings, their top-five checks disagree on 106 records (5.9%). Frozen reader comparisons associate the aligned object's presence with 28.6 to 31.0 points hi
    
[^60]: 极度稀疏的监督激励推理能力

    Extremely Sparse Supervision Incentivizes Reasoning Ability

    [https://arxiv.org/abs/2609.04565](https://arxiv.org/abs/2609.04565)

    研究发现在在线策略蒸馏中，仅需对每个推理轨迹中的一两个token（约占总token数的0.05%）进行监督，就能有效激励大语言模型的推理能力，且在多数情况下可匹配甚至超越全token训练的效果。

    

    大语言模型通过有效的后训练展现出日益强大的推理能力。然而，主流的后训练方法在海量token上进行优化，隐含地假设有效的学习必须依赖密集的token训练。我们在在线策略蒸馏（OPD）设置中重新审视了这一假设，该设置天然允许在生成的每个token上提供密集的教师监督。使用Qwen3系列模型，我们发现了一个反直觉的现象：推理能力可以通过极小比例的生成token得到有效激励——每个推理轨迹中仅需一到两个token，仅占全部token的0.05%。令人惊讶的是，尽管训练目标中排除了绝大多数生成token，这种稀疏监督在提升推理能力方面在大多数情况下仍可匹配甚至超越全token训练的效果。该现象在九种教师-学生配置中均被一致观察到。

    arXiv:2609.04565v1 Announce Type: new  Abstract: Large language models demonstrate increasingly strong reasoning capabilities through effective post-training. Yet, prevailing post-training methods optimize over massive numbers of tokens, implicitly assuming that effective learning must be token-intensive. We revisit this assumption in the on-policy distillation (OPD) setting, which naturally admits dense teacher supervision at every generated token. Using the Qwen3 family, we discover a counter-intuitive phenomenon: reasoning can be effectively incentivized by an extremely small fraction of generated tokens--as few as one or two tokens per reasoning trajectory, corresponding to only 0.05% of all tokens. Surprisingly, this sparse supervision in most cases matches or surpasses full-token training in improving reasoning ability, despite excluding the vast majority of generated tokens from the training objective. This phenomenon is consistently observed across nine teacher--student configu
    
[^61]: 工作节奏：面向职场智能体的人类行为轨迹多尺度解释

    Rhythms of Work: Multi-Scale Interpretation of Human Behavioral Traces for Workplace Agents

    [https://arxiv.org/abs/2609.04556](https://arxiv.org/abs/2609.04556)

    该论文提出行为解释依赖于时间分辨率这一核心观点，构建了由重复模式、连贯情节和日级节奏组成的多分辨率语义规范化词汇表，使职场智能体能够在不同时间尺度上对人类行为轨迹获得多种可寻址的解释，并基于6.67亿个人类归因事件进行了验证。

    

    运行时轨迹正成为理解智能体系统的核心基础，但现有解释工作主要关注智能体本身做了什么。职场智能体面临的是一个互补的问题：解释围绕它们展开的人类活动。数小时的低层级事件承载着关于用户状态的丰富证据，但过于细粒度而无法直接推理；而将其扁平化为单一数据流或压缩成单个嵌入向量，都把“总结用户行为”当作只有唯一正确答案的问题来处理。我们主张，行为解释是依赖于分辨率的：同一条轨迹应当可以在不同时间分辨率下获得多种可寻址的解释。我们构建了一个多分辨率的语义规范化词汇表，涵盖重复模式、连贯情节和日级节奏，每一层都保留了在其自身时间尺度上显著的结构。该方法应用于来自大规模数据的6.67亿个人类归因事件。

    arXiv:2609.04556v1 Announce Type: new  Abstract: Runtime traces are becoming a central substrate for understanding agentic systems, yet interpretation has focused largely on what the agent did. Workplace agents face the complementary problem: interpreting the human activity that surrounds them. Hours of low-level events carry rich evidence about a user's state but are too granular to reason over directly, and flattening them into one stream or compressing them into a single embedding both treat "summarize the user's behavior" as if it had one correct answer. We argue instead that behavioral interpretation is resolution-dependent: the same trace should admit multiple addressable interpretations at different temporal resolutions. We construct a multi-resolution vocabulary of semantically normalized operators, recurring motifs, coherent episodes, and day-level rhythms, each preserving the structure salient at its own horizon. Applied to 667 million human-attributed events from a large com
    
[^62]: 一种用于增强大语言模型置信度估计的校准反思方法

    A Calibrated Reflection Approach for Enhancing Confidence Estimation in LLMs

    [https://arxiv.org/abs/2609.04539](https://arxiv.org/abs/2609.04539)

    该论文提出了一种融合最大置信度选择、反思式提示和距离感知校准三项创新的校准反思框架，显著提升大语言模型的置信度估计能力，从而帮助系统判断何时信任模型输出、何时寻求人工干预。

    

    部署大语言模型（LLM）时的一个关键挑战是开发可靠的机制来估计其置信度，从而使系统能够判断何时信任模型输出、何时寻求人工干预。我们提出了一种用于增强大语言模型置信度估计的校准反思方法，这是一个将结构化推理与距离感知校准技术相结合的框架。我们的方法引入了三项关键创新：（1）最大置信度选择方法，能够对所有可能标签的置信度进行全面评估；（2）基于反思的提示机制，可提升推理的可靠性；（3）距离感知校准技术，能够考虑标签之间的有序关系。我们在多个数据集上对该框架进行了评估，包括 HelpSteer2、Llama T-REx 以及一个专有对话数据集，证明了其在对话型和事实型分类任务中的有效性。

    arXiv:2609.04539v1 Announce Type: new  Abstract: A critical challenge in deploying Large Language Models (LLMs) is developing reliable mechanisms to estimate their confidence, enabling systems to determine when to trust model outputs versus seek human intervention. We present a Calibrated Reflection approach for enhancing confidence estimation in LLMs, a framework that combines structured reasoning with distance-aware calibration technique. Our approach introduces three key innovations: (1) a Maximum Confidence Selection (MCS) method that comprehensively evaluates confidence across all possible labels, (2) a reflection-based prompting mechanism that enhances reasoning reliability, and (3) a distance-aware calibration technique that accounts for ordinal relationships between labels. We evaluate our framework on diverse datasets, including HelpSteer2, Llama T-REx, and a proprietary conversational dataset, demonstrating its effectiveness across both conversational and fact-based classific
    
[^63]: Scale-QLoRA：面向原生4比特微缩放大语言模型的编码不变适配器合并

    Scale-QLoRA: Code-Invariant Adapter Merging for Native 4-bit Microscaling LLMs

    [https://arxiv.org/abs/2609.04526](https://arxiv.org/abs/2609.04526)

    Scale-QLoRA通过只训练原生4比特微缩放检查点的每块缩放因子、同时冻结全部E2M1编码来合并LoRA适配器，从而避免了朴素合并因重新量化编码平面而抹除适配效果（最高达39个百分点）的问题。

    

    将LoRA适配器合并进基座模型是标准的部署做法：它消除了运行时适配器每次前向传播的开销，并留下一个任何服务栈都能加载的单一独立检查点。但在原生4比特微缩放检查点（NVFP4、MXFP4）上，这一步不再是无代价的。合并后的权重必须通过量化器写回，而量化器会重新推导检查点的离散E2M1编码平面（约占该制品字节数的90%），因此部署制品便与某一种量化约定耦合，其生命周期中后续任何触及编码的事件都可能使其发生偏移。若按朴素方式执行，这一步不只是脆弱，甚至更糟：它会抹除适配效果，最高可达39个百分点，因为面对一个已落在编码网格上的基座，重建的最优解就是该基座本身。Scale-QLoRA则只适配原生的每块缩放因子字段，在部署网格上训练这些缩放因子，并冻结所有E2M1编码。在固定的原生格式、缩放网格、块布局……（摘要原文在此处截断）

    arXiv:2609.04526v1 Announce Type: new  Abstract: Merging a LoRA adapter into its base model is standard deployment practice: it removes the runtime adapter's per-forward overhead and leaves a single standalone checkpoint any serving stack can load. On a native 4-bit microscaling checkpoint (NVFP4, MXFP4) that step stops being free. The merged weights must be written back through a quantizer, which re-derives the checkpoint's discrete E2M1 code plane (roughly 90% of the artifact's bytes), so the deployed artifact becomes coupled to one quantization convention, and every later code-touching event in its lifecycle can move it. Done naively the step is worse than fragile: it deletes the adaptation, by up to 39 pp, because against an already-on-grid base the reconstruction optimum is that base. Scale-QLoRA instead adapts only the native per-block scale field, trains those scales on the deployment grid, and freezes every E2M1 code. Within a fixed native format, scale grid, block layout and c
    
[^64]: LentEx：基于合成数据与指令微调大语言模型的泛化潜在实体抽取

    LentEx: Generalizable Latent Entity Extraction via Synthetic Data and Instruction-Tuned LLMs

    [https://arxiv.org/abs/2609.04511](https://arxiv.org/abs/2609.04511)

    LentEx利用基于模板的合成数据生成与指令微调技术优化小型大语言模型，首次系统性地实现了对文本中隐含、抽象潜在实体的泛化抽取。

    

    潜在实体抽取（LEE）旨在解决识别自由文本中隐含的、需要通过上下文推断的实体这一难题——而这正是传统实体抽取方法的短板。本文提出了LentEx，一种新颖的潜在实体抽取框架，它利用合成数据生成与指令微调技术来优化更小、更高效的大语言模型（LLM）。潜在实体往往具有抽象性和主题性，对于检索增强生成（RAG）、用户画像分析以及知识图谱增强等应用至关重要。LentEx通过采用基于模板的方法生成多样化、上下文丰富的合成数据，解决了标注数据集稀缺的问题，确保了数据的高可变性并与真实世界分布保持一致。据我们所知，LentEx是首个从大语言模型视角系统性研究潜在实体抽取的工作。LentEx展现出了显著的性能提升。

    arXiv:2609.04511v1 Announce Type: new  Abstract: Latent entity extraction (LEE) tackles the challenge of identifying implicit, contextually inferred entities within free text-an area where traditional entity extraction methods fall short. In this paper, we introduce LentEx, a novel framework for latent entity extraction that leverages synthetic data generation and instruction fine-tuning to optimize smaller, efficient large language models (LLMs). Latent entities, which are often abstract and thematic, are crucial for applications such as retrieval-augmented generation (RAG), customer persona analysis, and knowledge graph enrichment. LentEx addresses the scarcity of labeled datasets by employing a template-based approach to generate diverse, contextually rich synthetic data, ensuring high variability and alignment with real-world distributions. To our knowledge, LentEx is the first to systematically approach LEE through the lens of LLMs. LentEx demonstrates significant performance impr
    
[^65]: 重新思考间接提示注入：作为一个测试时搜索问题

    Rethinking Indirect Prompt Injection as a Test-Time Search Problem

    [https://arxiv.org/abs/2609.04495](https://arxiv.org/abs/2609.04495)

    该论文将间接提示注入重新定义为在任务相关攻击面上的测试时搜索问题，并提出一个具备环境侦察、策略推理与自适应评估能力的智能体攻击框架，证明攻击成功率随攻击者测试时计算预算的增加而提升，因此安全评估应同时刻画攻击者的搜索过程与计算预算。

    

    我们将间接提示注入形式化为一个测试时搜索问题，其搜索空间是由环境、用户任务和注入任务共同诱导的任务相关攻击面。为了将这一形式化付诸实践，我们引入了一种具备专用搜索框架的智能体攻击者，该框架能够执行环境侦察、对攻击策略进行结构化推理，并利用受害智能体的反馈进行自适应评估。在多种异构任务上，我们发现增加攻击者的测试时计算量能够提升漏洞发现与利用的能力；同时消融实验表明，显式的策略管理对于避免冗余搜索以及在更大计算预算下持续保持收益至关重要。这些结果表明，智能体安全评估应当刻画攻击者的搜索过程和计算预算，而不是将攻击成功视为与预算无关的受害方固有属性。更广泛地说，我们的发现指出攻击者的自适应搜索……

    arXiv:2609.04495v1 Announce Type: new  Abstract: We formulate indirect prompt injection as a test-time search over a task-dependent attack surface induced by the environment, user task, and injection task. To operationalize this formulation, we introduce an agentic attacker with a dedicated search harness that performs environment reconnaissance, structured reasoning over attack strategies, and adaptive evaluation using victim-agent feedback. Across heterogeneous tasks, we find that increasing attacker test-time compute improves vulnerability discovery and exploitation, while ablations show that explicit strategy management is important for avoiding redundant search and sustaining gains at larger budgets. These results suggest that agentic security evaluations should characterize both the attacker's search procedure and compute budget, rather than treating attack success as a budget-independent property of the victim. More broadly, our findings identify the attacker's adaptive search o
    
[^66]: 理解暂停标记微调动力学：模式保持视角

    Towards Understanding Pause Token Fine-Tuning Dynamics: A Mode Retention Perspective

    [https://arxiv.org/abs/2609.04489](https://arxiv.org/abs/2609.04489)

    本文提出掩码边界暂停标记方法（MBP），首次从模式保持与非短视压缩的训练动力学视角揭示了暂停标记提升推理能力的机制，并在1B至8B规模的Qwen和Llama模型上带来最高6个百分点的推理提升。

    

    暂停标记方法通过在序列中插入特殊标记来提升大语言模型的推理能力。先前的工作主要通过计算表达能力来解释这些增益。然而，针对暂停标记训练动力学的研究相对较少。我们探索了暂停标记如何重塑微调的训练动力学。两项对照实验揭示了明显的非对称性：在合成持续学习任务中，掩码暂停标记在匹配的最终适应效果下，对先前学习分布的覆盖程度减少了约4倍（假设H1，模式保持）；在合成数学推理探测任务中，边界相邻的标记编码了更多的下游步骤信息（假设H2，非短视压缩）。我们形式化了一个与这两者都一致的训练规则——掩码边界暂停标记（MBP），即将暂停标记放置在推理步骤边界并对其损失进行掩码处理。在1B至8B参数规模的Qwen和Llama模型上，MBP持续提升推理能力，增益最高可达6个百分点。

    arXiv:2609.04489v1 Announce Type: cross  Abstract: Pause-token methods improve LLM reasoning by inserting special tokens into sequences. Prior work explains these gains through computational expressivity. However, there is relatively little investigation into the training dynamics of pause tokens. We explore how pause tokens reshape the training dynamics of fine-tuning. Two controlled pilots expose distinct asymmetries. On a synthetic continual-learning task, masked pauses overwrite a previously-learned distribution roughly 4x less at matched final adaptation (H1, mode retention); on a synthetic math-reasoning probe, the boundary-adjacent token comes to encode substantially more downstream-step information (H2, non-myopic compression). We formalize a training rule consistent with both - Masked Boundary Pause (MBP), pause tokens placed at reasoning-step boundaries with their loss masked. Across 1B-8B Qwen and Llama models, MBP consistently improves reasoning, achieving gains of up to 6 
    
[^67]: 网络意图翻译的不确定性信号：风险排序与歧义定位

    Uncertainty Signals for Network Intent Translation: Risk Ranking and Ambiguity Localization

    [https://arxiv.org/abs/2609.04486](https://arxiv.org/abs/2609.04486)

    本文提出利用基于采样的预测不确定性对LLM生成的网络配置进行部署前翻译风险排序，并利用词元级熵定位歧义来源，从而弥补现有研究只关注翻译准确性而忽视部署风险的不足。

    

    基于意图的网络实现始于将高层意图翻译为低层网络配置。近期的方法已转向基于大语言模型（LLM）的翻译。尽管结果令人鼓舞，但大多数研究都聚焦于翻译准确性，而忽视了部署所生成配置时可能带来的风险。在本工作中，我们通过分析模型的不确定性，研究了LLM生成配置的部署前翻译风险。我们提出使用两种不确定性信号：即用于翻译风险排序的基于采样的预测不确定性，以及用于歧义来源定位的词元级熵。我们在一个歧义受控的测试集上，针对不同上下文类型和采样预算评估了这些信号，使用的是在特定厂商交换机平台（Juniper EX3300）上针对意图翻译任务微调的Llama-3.1-8B-Instruct模型。结果表明，预测不确定性提供了一个有用的信号

    arXiv:2609.04486v1 Announce Type: cross  Abstract: Intent-based networking realization starts by translating high-level intents into low-level network configurations. Recent approaches have shifted toward LLM-based translation. Despite promising results, most studies focus on translation accuracy and overlook risks associated with deploying the resulting configurations. In this work, we investigate the pre-deployment translation risk of LLM-generated configurations by analyzing the model's uncertainty. We propose to use two uncertainty signals, namely sampling-based predictive uncertainty for translation-risk ranking and token-level entropy for ambiguity-source localization. We evaluate these signals on an ambiguity-controlled test set across different context types and sampling budgets, using a Llama-3.1-8B-Instruct model fine-tuned for intent translation on a vendor-specific switch platform (Juniper EX3300). The results demonstrate that predictive uncertainty provides a useful signal
    
[^68]: 大型语言模型中的文化错位：通过针对性微调进行检测、测量与缓解

    Cultural Misalignment in Large Language Models: Detection, Measurement, and Mitigation Through Targeted Fine-Tuning

    [https://arxiv.org/abs/2609.04485](https://arxiv.org/abs/2609.04485)

    研究发现开源大语言模型并不偏袒其本国文化，且仅需约1,200个样本的针对性LoRA微调虽能降低文化偏差约17%，但实际上是重新分配而非消除偏差。

    

    我们评估了三个开源权重的大语言模型（来自美国的Gemma3-12B、来自波兰的Bielik-11B-v3和来自中国的Qwen3-4B），将其与三个国家63个人口统计画像的世界价值观调查第七波数据进行对比，并使用归一化Wasserstein距离来量化分布层面的文化错位程度。与预期相反，没有任何模型偏袒其本国：中国开发的Qwen3-4B在其本国中国人群上表现最差（W1 = 0.436，是整个模型×国家矩阵中错位程度最高的）。针对五个最差画像进行LoRA针对性微调，仅需不到1,200个训练样本对，在单个GPU上用时不到15分钟，即可使Bielik-11B的偏差降低16.8%（p_Bonf = 0.002，d = -4.4），且所有五个目标画像均有所改善。然而，国家层面的分解分析揭示，微调是重新分配而非消除偏差：Bielik的最差画像完全从美国老年人转变为中国的（原文在此处截断）。

    arXiv:2609.04485v1 Announce Type: cross  Abstract: We evaluate three open-weight LLMs (Gemma3-12B from the USA, Bielik-11B-v3 from Poland, and Qwen3-4B from China) against World Values Survey Wave 7 data for 63 demographic personas across three countries, using normalized Wasserstein distance to quantify distributional misalignment. Contrary to expectations, no model favors its home country: the Chinese-built Qwen3-4B performs worst on its own Chinese population (W1 = 0.436, the highest misalignment in the entire model x country matrix). Targeted LoRA fine-tuning on the five worst-case personas, requiring fewer than 1,200 training pairs and under 15 minutes on a single GPU, reduces bias by 16.8% for Bielik-11B (p_Bonf = 0.002, d = -4.4) with all five targets improving. However, country-level decomposition reveals that fine-tuning redistributes rather than removes bias: Bielik's worst-case personas swap entirely from American to Chinese elderly, with zero overlap between pre- and post-c
    
[^69]: 生产中的启动模式：语言模型生成中的词汇、语义与结构对齐

    Patterns of Priming in Production: Lexical, Semantic and Structural Alignment in Language Model Generation

    [https://arxiv.org/abs/2609.04484](https://arxiv.org/abs/2609.04484)

    本研究通过受控句子补全实验证明语言模型在生成时同样会受到结构启动效应的影响，且启动幅度呈现逆频率效应——较少出现的双宾语结构相对增幅更大，而更常见的介词宾语结构绝对增幅更大。

    

    本文研究了语言模型（LM）生成过程中的结构启动效应，考察前置结构上下文如何影响句子补全。尽管先前的研究已证明了结构交替在理解中的启动效应，但这些效应是否在生成过程中依然存在尚不清楚——因为在生成时，语言模型在每一步都从多种可能的续写中进行采样。我们通过对与格结构的一系列受控句子补全实验来探讨这一问题。与先前研究一致，我们发现语言模型容易受到结构启动的影响，尤其是在语义连贯的句子中。在启动幅度方面，我们发现虽然双宾语与格结构相对于基线有更大的相对增幅（这与逆频率效应相符），但更常被生成的介词宾语结构则表现出更大的绝对增幅。最后，我们不仅观察到……（摘要在此处截断）

    arXiv:2609.04484v1 Announce Type: cross  Abstract: This paper investigates structural priming in language model (LM) production, examining how preceding structural context influences sentence completion. While prior work has demonstrated priming effects in comprehension of structural alternations, it remained unclear whether these persist in production, where, when generating, an LM samples from many possible continuations at each step. We address this question through a series of controlled sentence-completion experiments on dative constructions. In line with prior work, we find that LMs are susceptible to structural priming, particularly in sentences that are semantically coherent. In terms of priming magnitude, we find that while there is a greater relative increase of double-object datives against our baselines, in line with inverse frequency effects, there is a larger absolute increase in prepositional-objects, the more frequently produced construction. Finally, we not only observ
    
[^70]: 安全为谁而设？面向受控大语言模型安全拒绝的边界感知自蒸馏

    Safety for Whom? Boundary-Aware Self-Distillation for Controlled LLM Safety Refusal

    [https://arxiv.org/abs/2609.04482](https://arxiv.org/abs/2609.04482)

    该论文提出“窄边界安全”新范式，通过结合升级重试机制的边界感知自蒸馏框架，使大语言模型能在同一主题内根据部署场景精细化控制拒绝边界，将目标域拒绝率从9.47%提升至84.75%，同时显著降低更广泛基准上的不安全响应率。

    

    安全对齐通常被表述为一个主题级别的问题：这个主题是否有害？而实际部署场景需要的是一个更窄的问题。公民教育导师和公共部门助手可能共享同一个基础模型，但在同一主题内需要不同的边界——拒绝有针对性的政治操纵，同时仍然回答关于同一次选举的事实性问题。我们将此问题表述为“窄边界安全”，并提出了一个离线自生成框架，该框架结合了受控主题生成、覆盖修复、分布内补偿数据以及用于训练和评估的有害-良性配对样本。单次生成会留下19.88%的提示没有可接受的拒绝轨迹，而升级重试机制可将这一比例降至0.20%。在Qwen3-8B的政治说服任务上，使用通过升级重试补全的拒绝数据进行训练，可将目标域拒绝率从9.47%提升至84.75%，并将三个更广泛危害性基准上的平均不安全响应率从26.26%降至0（摘要原文在此处截止）。

    arXiv:2609.04482v1 Announce Type: new  Abstract: Safety alignment is usually posed as a topic-level question: is this subject harmful? Deployments ask a narrower one. A civics tutor and a public-sector assistant may share a base model yet need different boundaries inside the same topic, refusing targeted political manipulation while still answering factual questions about the same election. We formulate this as narrow-boundary safety and introduce an offline self-generated framework combining controlled topic generation, coverage repair, in-distribution compensation data, and harmful-benign pairs for training and evaluation. Single-shot generation leaves 19.88% of prompts without accepted refusal traces, whereas escalating retries leave 0.20%. On political persuasion with Qwen3-8B, training on refusal data completed through Escalate increases target-domain refusal from 9.47% to 84.75% and reduces the mean unsafe-response rate across three broader harmfulness benchmarks from 26.26% to 0
    
[^71]: 共享电路预测大语言模型在算术推理中能否跨格式泛化

    Shared circuits predict whether LLMs generalize across formats in arithmetic reasoning

    [https://arxiv.org/abs/2609.04463](https://arxiv.org/abs/2609.04463)

    本研究通过归因修补技术定位大语言模型解决数字与语言表述算术问题的内部电路，发现模型语言电路与其数字电路的重叠程度能够预测其跨格式泛化能力。

    

    arXiv:2609.04463v1 公告类型：cross 摘要：在包括算术推理在内的许多推理形式中，跨输入格式的表面变化进行泛化对人类来说毫不费力：任何能算出 2+5 的人也能解决 “二加五”。相比之下，大语言模型对提示的表面变化更为脆弱：例如，它们几乎能完美解决数字形式的算术问题，但在同样问题的语言表述形式上准确率明显下降。本研究探究是否可以从模型内部预测跨格式的泛化能力。我们首先使用归因修补技术，独立定位了每个模型在英语、西班牙语和意大利语三种语言中，解决数字算术问题（2+5）与语言表述问题时分别调用的电路；然后，我们检验模型自身数字电路与语言电路的重叠程度是否能预测其对语言表述格式的泛化。事实上，我们在三个层面上找到了支持这一观点的证据……

    arXiv:2609.04463v1 Announce Type: cross  Abstract: In many forms of reasoning, including arithmetic reasoning, generalizing across superficial changes in input format is effortless for humans: anyone who can solve 2+5 can also solve 'two plus five'. In contrast, LLMs are more brittle to surface variations of the prompts: for example, they solve numeric arithmetic problems almost perfectly but are substantially less accurate on verbal renditions of the same problems. Here, we ask whether generalization across formats can be predicted from the models' internals. Using attribution patching, we first independently localize the circuit that each model recruits to solve numeric arithmetic problems (2+5) vs. verbal ones, in three languages: English ('two plus five'), Spanish ('dos m\'as cinco'), and Italian ('due pi\`u cinque'); then, we test whether overlap with the model's own numeric circuit predicts its generalization to the verbal formats. Indeed, we find support for this idea at three l
    
[^72]: 当负载均衡走向极端：过度分散混合专家模型中的专家剪枝

    When Load-Balancing Goes Too Far: Expert Pruning in Over-Dispersed Mixture-of-Experts Models

    [https://arxiv.org/abs/2609.04453](https://arxiv.org/abs/2609.04453)

    该论文发现在因训练时负载均衡过于激进而导致路由过度分散的MoE模型中，路由器概率不再是可靠的专家重要性信号，困惑度也无法预测下游任务准确率，因此传统的基于路由的专家剪枝方法在此类模型中会失效，且不同评分指标之间存在能力权衡。

    

    专家剪枝通过移除由路由器识别出的低重要性专家，来降低混合专家模型（MoE）的内存与服务成本，其前提假设是路由器概率能够提供可靠的专家重要性信号。我们观察到，这一假设在“过度分散路由”（over-dispersed routing）的状态下会失效——这种状态与训练中过于激进的负载均衡相关，此时token几乎均匀地分布到各个专家上，重要性信号随之崩塌。在这种状态下，困惑度无法预测下游任务的准确率：在gpt-oss-20B上，困惑度最低的剪枝配置反而产生了最差的数学推理表现，而困惑度最高的配置却保留了数学推理能力。这一现象在标准路由（如Mixtral-8x7B-Instruct）下并不会出现，在标准路由下困惑度与准确率会同步退化。此外，在过度分散路由下进行剪枝还会暴露出一种能力上的权衡：没有任何单一的评分指标能够全面占优，其中激活感知评分……

    arXiv:2609.04453v1 Announce Type: cross  Abstract: Expert pruning reduces the memory and serving cost of Mixture-of-Experts (MoE) models by removing low-importance experts identified by the router, assuming router probabilities provide a reliable importance signal. We observe that this assumption breaks down under over-dispersed routing, a regime associated with aggressive load-balancing during training, in which tokens are distributed nearly uniformly across experts and importance signals collapse. In this regime, perplexity does not predict downstream task accuracy: on gpt-oss-20B, the lowest-perplexity pruning configuration yields the worst mathematical reasoning, while the highest-perplexity configuration preserves it. This does not occur under standard routing (e.g., Mixtral-8x7B-Instruct), where perplexity and accuracy degrade together. Pruning under over-dispersed routing also exposes a capability trade-off in which no single scoring metric dominates: activation-aware scoring pr
    
[^73]: TRILOGUE：一个包含证据和配对音频的三语口语对话事实核查基准

    TRILOGUE: A Trilingual Spoken Dialogue Fact-Checking Benchmark with Evidence and Paired Audio

    [https://arxiv.org/abs/2609.04452](https://arxiv.org/abs/2609.04452)

    TRILOGUE是首个大规模三语（英语、俄语、哈萨克语）口语对话事实核查基准，包含近1.2万个对话、18.7万个轮次和390小时的配对音频，填补了带配对语音和轮次级标签的大型多语言事实核查基准的空白。

    

    现代错误信息往往在被阅读之前就先被听到，然而事实核查系统仍然主要在干净的书面声明上进行评估。口语对话即使在系统基于转录文本运行时也存在差异：声明可能分布在不同的说话者和对话轮次之间，依赖于先前的上下文，并且当自动语音识别（ASR）错误扭曲可用文本时验证难度更高。现有的口语对话事实核查资源规模较小、以英语为中心，或专注于标注而非端到端基准测试，导致目前缺乏带有配对语音和轮次级标签的大型多语言基准。我们推出了TRILOGUE（三语口语对话事实核查），这是一个大规模的三语基准，包含英语、俄语和哈萨克语的基于来源溯源的口语对话。它包含近1.2万个对话、18.7万个对话轮次和390小时的配对音频，并配有所有三种语言的ASR转录文本和词级时间戳对齐。

    arXiv:2609.04452v1 Announce Type: new  Abstract: Modern misinformation is often heard before it is read, yet fact-checking systems are still evaluated mainly on clean written claims. Spoken dialogue remains different even when systems operate on transcripts: claims may be distributed across speakers and turns, depend on prior context, and become harder to verify when Automatic Speech Recognition (ASR) errors distort the available text. Prior spoken dialogue fact-checking resources are small, English-centric, or focused on annotation rather than end-to-end benchmarking, leaving no large multilingual benchmark with paired speech and turn-level labels. We introduce TRILOGUE (TRIlingual spoken diaLOGUE fact-checking), a large-scale trilingual benchmark of source-grounded spoken dialogues in English, Russian, and Kazakh. It contains nearly 12K dialogues, 187K turns, and 390 hours of paired audio with ASR transcripts and word-level timestamp alignments across all three languages, including n
    
[^74]: 从众性破坏共形预测

    Conformity Breaks Conformal Prediction

    [https://arxiv.org/abs/2609.04445](https://arxiv.org/abs/2609.04445)

    论文揭示同伴压力会使多智能体LLM系统发生“评分机制偏移”，悄然破坏共形预测的覆盖率保证（从90%降至74%），且攻击者可通过针对低置信度子组将其覆盖率近乎减半（从87%降至47%）。

    

    当LLM单独回答时，共形证书可以是有效的，但当同一个LLM看到一致断言错误答案的同伴时，该证书便会失效。问题本身没有改变，改变的是模型对正确答案的评分。我们将这一现象称为“评分机制偏移”：在干净条件下校准的证书只能认证模型单独回答时的评分方式，却无法认证其在同伴压力下的评分方式。我们证明这种偏移会在多智能体LLM系统中悄然破坏共形预测。在多种开放权重模型和多选题问答任务上，在标准alpha = 0.10工作点下，当存在一致给出错误答案的同伴时，覆盖率会从校准后的90%降至74%。这一平均值掩盖了更严重的失败：攻击者通过针对证书仍能覆盖的低置信度题目，几乎将该子组的覆盖率减半——从87%降至47%——而受监控的整体平均值仍保持在较高水平。这种失败还会波及决策层：一个本应在不确定时上报的系统……（原文摘要在此处截断）

    arXiv:2609.04445v1 Announce Type: cross  Abstract: A conformal certificate can be valid when an LLM answers alone and invalid when the same LLM sees peers that unanimously assert a wrong answer. The question is unchanged; the model's score for the correct answer changes. We call this a score-mechanism shift: clean calibration certifies how the model scores answers alone, but not how it scores them under peer pressure. We show that this shift silently breaks conformal prediction in multi-agent LLM systems. Across open-weight models and multiple-choice QA tasks, coverage falls from a calibrated 90% to 74% under unanimous-wrong peers at the standard alpha = 0.10 operating point. The average hides a sharper failure: by targeting the low-confidence items the certificate still covers, an attacker nearly halves coverage on that subgroup, from 87% to 47%, while the monitored average remains much higher. The failure also reaches the decision layer: a system that should escalate when uncertain c
    
[^75]: GRACE：面向专家在环知识扩展的图锚定反思式智能体副驾驶引擎

    GRACE: Graph-Grounded Reflective Agent Copilot Engine for Expert-in-the-Loop Knowledge Expansion

    [https://arxiv.org/abs/2609.04442](https://arxiv.org/abs/2609.04442)

    提出GRACE框架，将LLM响应解构为原子论断，通过加权二部图与可信知识先验锚定，利用加权中心性分析将论断分类为有依据、被反驳或边界三类，并借助注意力回报率（RoA）目标高效分配专家审查资源，从而在识别幻觉的同时发现模型知识前沿的新颖论断。

    

    在高风险场景中部署的大型语言模型经常生成看似合理但缺乏依据的论断。标准的检索增强生成（RAG）流程提供的解决方案有限，因为它们只检索孤立的段落，既不追踪跨文档的证据关系，也不量化不确定性。我们提出GRACE（图锚定反思式智能体副驾驶引擎），这是一个将LLM响应解构为原子论断，并在加权二部图中将其与可信知识先验进行锚定的框架。边权重编码了每个论断与先验知识的接近程度，从而实现加权中心性分析，将论断分类为“有依据”、“被反驳”或“边界”。这种分类不仅能识别幻觉，还能识别模型知识前沿处新颖或有争议的论断。为了高效分配人类或智能体资源，我们制定了注意力回报率目标，将论断推迟给专家审…（原文摘要在此处截断）

    arXiv:2609.04442v1 Announce Type: cross  Abstract: Large language models deployed in high-stakes settings frequently generate plausible but ungrounded claims. Standard retrieval-augmented generation (RAG) pipelines offer limited remedy, since they retrieve isolated passages without tracking cross-document evidence relationships or quantifying uncertainty. We introduce GRACE (Graph-grounded Reflective Agent Copilot Engine), a framework that deconstructs LLM responses into atomic claims and grounds them against trusted knowledge priors within a weighted bipartite graph. Edge weights encode the closeness of each claim to the priors, enabling weighted centrality analysis that classifies claims as Grounded, Refuted, or Boundary. Such classification identifies not just hallucinations but also novel or contested claims at the frontier of the model's knowledge. To efficiently allocate human or agent resources, we formulate a Return on Attention (RoA) objective that defers a claim to expert rev
    
[^76]: 混合语言模型中注意力所回忆与循环所控制的内容

    What Attention Recalls and Recurrence Controls in Hybrid Language Models

    [https://arxiv.org/abs/2609.04434](https://arxiv.org/abs/2609.04434)

    混合语言模型中两条通道功能明确分工：注意力通道专责上下文精确检索，循环状态通道则控制输出语言和角色风格。

    

    混合语言模型将注意力机制与固定大小的循环状态相结合，但每个通道的具体作用仍不明确。我们引入了两种缓存层面的干预方法。Split-prefill（分离预填充）仅保留预填充上下文中的KV缓存或仅保留循环状态，然后生成答案。State-swap（状态交换）在单次前向传递中将来自一个上下文的KV缓存与来自另一个上下文的循环状态配对。在Qwen3.5和Falcon-H1模型上，两个通道在功能上呈现明显分化。精确检索只能通过注意力通道得以保留（达到完整准确率的64-98%），而通过循环通道则完全崩溃为零。输出语言和角色设定则呈现相反的模式：两者都能通过循环通道得以保留（语言准确率70-80%，角色保持3-5倍），而仅使用KV缓存时语言准确率降至约1%。State-swap从因果上证实了这一分工：答案的内容来自KV缓存一侧，而语言特征来自循环状态一侧。此外，仅用循环状态生成时，模型还能接受从未出现在上下文中、但与见过的词语共享含义或词形部分的词……

    arXiv:2609.04434v1 Announce Type: new  Abstract: Hybrid language models combine attention with a fixed-size recurrent state, but the role of each channel remains unclear. We introduce two cache-level interventions. Split-prefill keeps only the KV cache or only the recurrent state from a prefilled context, then generates an answer. State-swap pairs the KV cache from one context with the recurrent state from another in a single forward pass. On Qwen3.5 and Falcon-H1, the two channels split sharply by function. Exact retrieval survives only through attention (64-98% of full accuracy) and collapses to zero through recurrence. Output language and persona reverse the pattern: both survive recurrence (70-80% and 3-5x) while KV-only drops to ~1% language accuracy. State-swap confirms this causally: the answer takes its value from the KV side and its language from the recurrent side. Recurrent-only generation also accepts words that were never in the context but share meaning or parts with seen
    
[^77]: 多语言语言模型中跨语言一致性增强方法的系统性评估

    A Systematic Evaluation of Cross-Lingual Consistency Enhancement Methods in Multilingual Language Models

    [https://arxiv.org/abs/2609.04409](https://arxiv.org/abs/2609.04409)

    本文对多语言模型中的跨语言一致性增强方法进行了统一的系统性评估，发现后训练方法（尤其是直接分布对齐）总体更可靠且能稳定提升一致性，而跨域迁移仅在源与目标任务输出格式相似时才有效。

    

    多语言语言模型在处理语义等价但表达于不同语言的问题时，常常产生不一致的答案，这促使研究者提出改进跨语言一致性（CLC）的方法。然而，现有方法通常在不同的模型、任务和评估协议下进行评估，导致其相对优势尚不明确。在本工作中，我们对问答任务中代表性的跨语言一致性增强方法进行了统一评估，涵盖推理时干预和后训练两类方法，涉及三个模型家族和三个封闭式基准。结果表明，后训练方法总体上更为可靠，其中直接分布对齐在所有模型-数据集组合中均能持续改进跨语言一致性，而其他方法则对答案格式和语言覆盖广度更为敏感。值得注意的是，除非源任务与目标任务具有相似的输出格式，否则跨域迁移的效果有限。我们进一步研究……

    arXiv:2609.04409v1 Announce Type: cross  Abstract: Multilingual language models often produce inconsistent answers to semantically equivalent questions across languages, motivating methods to improve cross-lingual consistency (CLC). However, existing methods are typically evaluated using different models, tasks, and protocols, leaving their relative strengths unclear. In this work, we present a unified evaluation of representative CLC-enhancement methods for question answering, spanning inference-time interventions and post-training approaches across three model families and three closed-form benchmarks. The results show that post-training methods are generally more reliable, with direct distribution alignment consistently improving CLC across all model-dataset combinations, while other methods are more sensitive to answer format and the breadth of language coverage. Notably, cross-domain transfer is limited unless source and target tasks share similar output formats. We further invest
    
[^78]: ASR幻觉的解剖学分析

    The Anatomy of an ASR Hallucination

    [https://arxiv.org/abs/2609.04404](https://arxiv.org/abs/2609.04404)

    该研究揭示了ASR幻觉源于“接地失败”，发现编码器最后阶段是关键边界——绕过该阶段会导致输出发散而非流畅的虚构文本。

    

    自动语音识别（ASR）系统有时会产生与其接收到的语音无关的流畅文本。我们将这些幻觉视为一种更广泛的“接地失败”的可能后果，即转录文本不再受到音频的充分引导。为了探究这种失败在何处成为可能，我们研究了两个独立训练的Conformer-Large识别器——一个基于CTC，一个基于RNN-T——在环境退化和说话人背景变化下的表现。在两个模型中，编码器的最后阶段都是一个关键边界：绕过最后一个模块会导致几乎每条语句的输出发散，而绕过中间模块则几乎没有影响。在同一阶段，表示变得更加紧凑，文本变得可以被训练好的解码器读取，字符信息也变得显式。重要的是，这种干预产生的是乱码或重复的输出，而非流畅的虚构内容。因此，我们的结果识别出了一个机制性的前提条件

    arXiv:2609.04404v1 Announce Type: new  Abstract: ASR systems sometimes produce fluent text that is unrelated to the speech they receive. We view these hallucinations as one possible consequence of a broader grounding failure, in which the transcript is no longer adequately guided by the audio. To understand where this failure becomes possible, we study two independently trained Conformer-Large recognizers - one CTC and one RNN-T - under environmental degradation and speaker-background shift. In both models, the final encoder stage emerges as a critical boundary: bypassing the final block causes divergence on nearly every utterance, whereas bypassing middle blocks has little effect. At this same stage, the representations become more compact, text becomes readable by the trained decoder, and grapheme information becomes explicit. Importantly, the intervention produces garbled or repetitive output rather than fluent fabrication. Our result therefore identifies a mechanistic precondition 
    
[^79]: 转录数据集上语音编码算法的评估

    Evaluation of Phonetic Encoding Algorithms on Transcription Datasets

    [https://arxiv.org/abs/2609.04391](https://arxiv.org/abs/2609.04391)

    提出了一种基于Hüllermeier-Rifqi指数的新型评估方案，通过计算语音编码与IPA真实转录之间的成对相似度差异，并结合碰撞率评估多种语音编码算法在多语言转录数据集上的性能。

    

    本工作提出了一种基于广义Rand指数变体（即Hüllermeier-Rifqi指数）的新型评估方案，用于评估语音编码算法与IPA（国际音标）标注的基于单词的转录结果的一致程度。为此，通过计算真实转录结果与相应语音编码之间的成对相似度值的绝对差来获得不一致分数，其中相似度值采用归一化编辑距离作为依赖排列的字符串度量进行计算。所得分数随后根据使用与所考虑编码器相同字母表的随机字符串生成器的分数进行调整。以此方式，在多语言转录数据集上评估了多种语音编码器，并基于碰撞率评估了它们的召回能力。

    arXiv:2609.04391v1 Announce Type: new  Abstract: In this work, a novel evaluation scheme built on a generalized variant of the Rand Index measure, namely, the H\"ullermeier-Rifqi Index, is proposed in order to assess how well phonetic encoding algorithms conform to word-based transcriptions in IPA (International Phonetic Alphabet) notation. For this objective, the discordance score is obtained by calculating the absolute difference between the pairwise similarity values of ground-truth transcriptions and those of corresponding phonetic encodings, which are computed using normalized edit distance as a permutation dependent string metric. The resulting score is subsequently adjusted with respect to that of a random string generator incorporating the same alphabet as the encoder under consideration. A wide range of phonetic encoders were evaluated as such on multi-lingual transcription datasets along with their recall capabilities based on the collision rate. The validity of the proposed 
    
[^80]: 你真的没听懂吗？面向间接与俏皮中文网络评论的社交语用推理基准测试

    You Really Didn't Get That? Benchmarking Social Pragmatic Inference for Indirect and Playful Chinese Online Comments

    [https://arxiv.org/abs/2609.04384](https://arxiv.org/abs/2609.04384)

    本文基于超过20万条真实中文社交媒体互动记录，构建了包含4,735个人工验证诊断条目的语用推理基准，用于评估大语言模型能否在上下文中正确解读间接、俏皮评论的社交含义，结果显示最强模型准确率仅81.42%、八个模型平均68.70%，该任务对当前模型仍具很大挑战性。

    

    中文网络评论常常通过间接、俏皮的语言传达社交含义，脱离上下文便难以解读。现有评估大多围绕预定义的语言现象或受控的语用类别来组织测试条目，因而模型能否区分一条自然产生的评论在特定对话情境中的各种合理解读，这一问题仍未得到回答。我们提出了一个用于评估大语言模型（LLM）能否恢复这种情境化语用含义的基准。我们从超过20万条公开中文社交媒体互动记录中构建了4,735个经人工验证的诊断条目，每个条目将一条目标评论与重建的前文语境以及合理的误读选项配对。我们在交叉出题人设置下评估了八个大语言模型，它们同时担任出题者与解题者。该任务颇具挑战性：最强模型在留一出题人（leave-writer-out）设置下达到81.42%的准确率。在全部八个模型中，平均留一出题人准确率为68.70%，而人类……（摘要原文在此截断）

    arXiv:2609.04384v1 Announce Type: cross  Abstract: Chinese online comments often convey social meaning through indirect and playful language that is hard to interpret without context. Existing evaluations largely organize items around predefined phenomena or controlled pragmatic categories, leaving open whether models can distinguish plausible readings of what a naturally occurring comment is doing in a particular exchange. We introduce a benchmark for evaluating whether LLMs can recover such situated pragmatic meanings. From more than 200,000 public Chinese social media interaction records, we construct 4,735 human-validated diagnostic items, each pairing a target comment with reconstructed preceding context and plausible misreadings. We evaluate eight LLMs as both question writers and solvers in a cross-writer setting. The task is challenging: the strongest model achieves 81.42% leave-writer-out accuracy. Across all eight models, the mean leave-writer-out accuracy is 68.70% while hum
    
[^81]: VERGE：面向临床笔记中早发性结直肠癌症状有据提取的验证增强式精炼方法

    VERGE: Verification-Enhanced Refinement for Grounded Extraction of Early-Onset Colorectal Cancer Symptoms in Clinical Notes

    [https://arxiv.org/abs/2609.04366](https://arxiv.org/abs/2609.04366)

    VERGE是一种结合检索增强生成与有界验证-精炼循环的智能体工作流，能从临床自由文本笔记中可靠提取早发性结直肠癌的六种红旗症状及家族史风险，并将无法解决的论断自动升级为人工审查。

    

    早发性结直肠癌在年轻成年人中的发病率正在上升，然而该年龄段的红旗症状（警示症状）尚无基于证据的随访检测指南，且结构化就诊数据无法捕捉支持早期检测和指导随访所需的细节，包括症状持续时间、临床情境以及家族史——后者是结直肠癌的既定危险因素。本研究旨在开发并评估一种自动化方法，用于从自由文本临床笔记中提取六种红旗症状和家族史风险状态。我们提出了VERGE，这是一种智能体工作流：首先利用检索增强生成提出初始标签和证据，随后经过一个有界的验证-精炼循环，检查论断的文本依据与临床有效性，对论断进行纠正并复核，直至问题解决或达到循环上限，并将未能解决的论断升级以供人工审查。VERGE在4,033份由临床医生标注的（笔记）上进行了评估（摘要原文在此处截断）。

    arXiv:2609.04366v1 Announce Type: new  Abstract: Early-onset colorectal cancer is increasing among younger adults, yet red-flag symptoms in this age group have no evidence-based guidelines for follow-up testing, and structured encounter data do not capture the detail needed to support early detection and inform follow-up, including symptom duration, context, and fam- ily history, an established colorectal-cancer risk factor. This study aimed to develop and evaluate an automated method for extracting six red-flag symptoms and family-history risk status from free-text clinical notes. We developed VERGE, an agentic workflow in which an initial label and evidence are proposed using retrieval-augmented generation, then passed through a bounded verification- refinement cycle that checks textual grounding and clinical validity, corrects and rechecks a claim until resolved or a limit is reached, and escalates unresolved claims for human review. VERGE was evaluated on 4,033 clinician-labeled no
    
[^82]: 知道何时不应作答：面向音乐音频-语言模型弃权机制的伪集成方法

    Knowing When Not to Answer: Pseudo-Ensembles for Abstention in Music Audio-Language Models

    [https://arxiv.org/abs/2609.04362](https://arxiv.org/abs/2609.04362)

    提出通过打乱选项顺序等不会改变正确答案的输入扰动，从单个预训练音乐音频-语言模型构建伪集成，从而获得置信度估计，使模型在不确定时能够弃权而非猜测。

    

    音乐音频-语言模型的评估几乎完全依赖于多项选择题的准确率。这种评估方式迫使模型必须选定一个选项，因此靠运气猜对的答案与真正的音乐理解看起来毫无区别。目前所缺乏的是一种判断模型何时不知道答案的方法，从而使模型能够选择弃权而不是盲目猜测。通常的解决方案是使用多个独立训练的模型组成集成，但这在音乐音频-语言模型场景下成本过于高昂，这使得单一预测分布的熵成为唯一可用的置信度信号。我们转而提出从单个预训练模型构建伪集成的方法：通过以不会改变正确答案的方式对输入进行扰动，然后对候选答案上的多个预测分布进行平均。我们的主要构造方法只是简单地打乱候选答案的呈现顺序；我们还研究了由受损音频和交换选项标签构建的集成。伪集成提供了多个预测分布，可用于……

    arXiv:2609.04362v1 Announce Type: cross  Abstract: Music audio-language models are evaluated almost entirely by accuracy on multiple-choice questions. This protocol forces the model to commit to an option, so a lucky guess looks the same as real musical understanding. What is missing is a way to tell when the model does not know the answer, so that it can abstain instead of guessing. The usual solution, an ensemble of independently trained models, is far too expensive here, which leaves the entropy of a single predictive distribution as the only available confidence signal. We instead build pseudo-ensembles from one pretrained model by perturbing its input in ways that cannot change the correct answer, then averaging the resulting distributions over the options. Our main construction simply shuffles the order in which the candidate answers are presented; we also study ensembles built from corrupted audio and from swapped option labels. A pseudo-ensemble gives several predictive distrib
    
[^83]: 从低谷中适应：预测心理健康危机辅导员长期对话技能的发展

    Adapting from Downturns: Prediction of Long-Term Conversational-Skill Development in Mental-Health Crisis Counselors

    [https://arxiv.org/abs/2609.04350](https://arxiv.org/abs/2609.04350)

    本文提出了在辅导员职业生涯早期预测其长期对话技能能否提升的新任务，发现辅导员如何从对话低谷中适应和恢复最能预示其长期改进潜力，从而帮助优先支持最需要帮助的辅导员。

    

    人们如何学会成为更好的对话者？这个问题在心理健康咨询领域尤为重要：对话技能在其中至关重要，但志愿辅导员往往难以获得督导和结构化的反馈。理解辅导员如何发展其引导对话走向积极结果的能力——并尽早识别哪些辅导员（没有）走在改善的轨道上——有助于为最需要支持的辅导员优先提供帮助。在这项工作中，我们提出了一个新的任务：在对话者职业生涯的早期预测其最终能否提升引导对话走向积极结果的能力，并以志愿心理健康危机辅导员为例证明了该任务的可行性。我们的核心洞察是，人们可能会在对话中的某些特定时刻遇到困难，而最能揭示其改善可能性的是他们如何从对话的低谷中适应与恢复……

    arXiv:2609.04350v1 Announce Type: cross  Abstract: How do people learn to become better conversationalists? This question is especially important in the context of mental-health counseling, where conversational skills are essential, yet volunteer counselors often have limited access to supervision and structured feedback. Understanding how counselors develop their ability to steer conversations toward positive outcomes -- and identifying early which counselors are (not) on track to improve -- can help prioritize support for the counselors who need it most.   In this work, we introduce the task of predicting, early in a conversationalist's career, whether they will eventually improve at steering conversations toward positive outcomes, and demonstrate the feasibility of this task in the case of volunteer mental-health crisis counselors. Our central insight is that people may struggle with particular kinds of moments in a conversation, and that what is especially revealing of their likeli
    
[^84]: SharedSAE：跨语言模型的单一特征字典

    SharedSAE: One Feature Dictionary Across Language Models

    [https://arxiv.org/abs/2609.04344](https://arxiv.org/abs/2609.04344)

    SharedSAE通过共享字典结合各模型专属的编码器-解码器对，用单一稀疏自编码器即可替代多个语言模型的专用SAE，在保留96.6%解释方差的同时实现跨模型的统一特征解释与迁移。

    

    稀疏自编码器（SAE）被广泛用于解释语言模型的激活，但SAE的训练和潜变量标注通常需要对每个模型重复进行。在这项工作中，我们证明了一个单一的共享SAE可以取代一组针对各模型的专用SAE。我们的方法SharedSAE将共享字典与模型专属的编码器-解码器对相结合。与最接近的先前方法不同——该方法会丢弃激活幅度并要求在推理时使用所有模型——SharedSAE仅对选择分数进行归一化以保留激活幅度，并通过模型丢弃策略实现单模型推理。我们在四个跨越不同模型家族和分词器的10亿参数规模基础语言模型上训练了SharedSAE。尽管其潜变量在模型间共享，SharedSAE仍保留了专用SAE平均解释方差的96.6%；其潜变量激活所展现的跨模型相关性是事后对齐的独立SAE的1.8倍，且其潜变量描述能够在模型间迁移。

    arXiv:2609.04344v1 Announce Type: cross  Abstract: Sparse autoencoders (SAEs) are widely used to interpret language model activations, but SAE training and latent labelling are typically repeated for every model. Here, we show that a single shared SAE can replace a collection of dedicated per-model SAEs. Our method, SharedSAE, combines a shared dictionary with model-specific encoder-decoder pairs. Unlike the closest prior method, which discards activation magnitudes and requires all models at inference, SharedSAE instead normalizes only selection scores, preserving magnitudes, and uses model dropout for single-model inference. We train SharedSAE on four 1B-scale base language models spanning distinct families and tokenizers. Despite sharing its latents across models, SharedSAE retains 96.6% of dedicated SAEs' mean explained variance; its latent activations exhibit cross-model correlations 1.8 times as high as separate SAEs aligned post-hoc, and its latent descriptions transfer across m
    
[^85]: 一种基于删除的方法在测试时提升大语言模型的忠实性

    A Removal Based Approach to Improve LLM Faithfulness at Test-Time

    [https://arxiv.org/abs/2609.04343](https://arxiv.org/abs/2609.04343)

    提出了一种基于删除的测试时方法，直接针对大语言模型解释中此前被忽视的不完整性问题，无需访问模型权重或大量计算资源即可提升解释的忠实性。

    

    大语言模型（LLM）越来越多地被用于重大决策，这使得模型的解释成为审计模型行为的重要工具。遗憾的是，这些解释可能是不忠实的，未能反映模型决策背后的实际推理过程。我们考虑这样一个场景：LLM在回答问题时同时提供答案和解释。我们识别出不忠实解释的两个不同维度：不完整性，即解释遗漏了影响答案的因素；以及不合理性，即解释引用了并未影响模型答案的因素。现有的提升LLM忠实性的方法包括训练时方法（需要访问模型权重和大量计算资源）以及主要针对不合理性问题的测试时方法。我们提出了一种直接针对不完整性问题的测试时方法。我们移除...

    arXiv:2609.04343v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used for consequential decisions, making their explanations an important tool for auditing model behavior. Unfortunately, these explanations can be unfaithful, failing to reflect the actual reasoning underlying the model's decisions. We consider a setting in which an LLM provides both an answer and an explanation in response to a question. We identify two distinct dimensions of unfaithful explanations: incompleteness, meaning that the explanation omits factors that influence the answer, and unsoundness, meaning that the explanation cites factors that did not influence the model's answer. Existing approaches to improving LLM faithfulness include training-time methods, which require access to model weights and extensive computational resources, and test-time methods that largely focus on addressing unsoundness. We introduce a test-time approach that directly targets incompleteness. We remove fr
    
[^86]: MedProb：探测视觉语言模型内部表征以用于医学问答

    MedProb: Probing Internal Representations of Vision-Language Models for Medical Question Answering

    [https://arxiv.org/abs/2609.04336](https://arxiv.org/abs/2609.04336)

    MedProb是一个轻量级探测框架，可直接从冻结的视觉语言模型内部表征中预测医学视觉问答答案，性能超越提示方法和医学专用VLM，并揭示了小型模型含有比生成式评估所显示的更丰富的医学问答信号，同时避免了自由文本生成中的答案位置偏差。

    

    医学视觉问答通常被认为需要医学领域的微调、大型模型或复杂的多智能体流水线。我们通过MedProb重新审视了这一假设——MedProb是一个轻量级的探测框架，它从冻结的视觉语言模型（VLM）表征中预测多选题式的医学视觉问答答案，而无需自由文本生成。在PATH-VQA、SLAKE和VQA-RAD数据集上，MedProb比提示方法恢复了多得多的与答案相关的信号，并且表现优于医学VLM和智能体系统。与提示方法相比，探测方法还缩小了小型模型与大型模型之间表观上的差距，这表明较小的VLM所包含的可恢复医学问答信号比基于生成的评估所显示的更多。在14对匹配的通用型与医学VLM中，医学领域的适配并不总能一致地改善这种线性可解码性。最后，自由文本生成表现出高达10个百分点的答案位置偏差，而MedProb则……

    arXiv:2609.04336v1 Announce Type: new  Abstract: Medical visual question answering (Med-VQA) is often assumed to require medical fine-tuning, large models, or complex multi-agent pipelines. We revisit this assumption with \textbf{MedProb}, a lightweight probing framework that predicts multiple-choice Med-VQA answers from frozen VLM representations without free-text generation. Across PATH-VQA, SLAKE, and VQA-RAD, MedProb recovers substantially more answer-relevant signal than prompting and performs stronger than medical VLMs and agentic systems. Probing also reduces the apparent gap between small and large models compared to prompting, suggesting that smaller VLMs contain more recoverable Med-VQA signal than generation-based evaluation reveals. Across 14 matched general-purpose and medical VLM pairs, medical adaptation does not consistently improve this linear decodability. Finally, free-text generation exhibits an answer-position bias of up to 10 percentage points, whereas MedProb als
    
[^87]: 抽象智能体

    Abstraction Agent

    [https://arxiv.org/abs/2609.04303](https://arxiv.org/abs/2609.04303)

    提出了一种基于大语言模型的零样本“抽象智能体”流水线，仅凭自然语言博弈描述即可自动发现策略特征并构建不完全信息博弈的信息抽象，无需博弈特定评估器、训练数据或博弈树遍历。

    

    信息抽象将策略上相似的私有状态归组为数量可控的若干“桶”，这对于将博弈求解算法扩展到大型不完全信息博弈至关重要。然而，构建有效的抽象传统上需要领域特定的评估器，例如手牌强度计算器或权益估计器，这些评估器需要专业知识和工程投入，且对于大多数研究较少的博弈并不存在。我们提出了抽象智能体，这是一个零样本流水线，它利用大型语言模型从自然语言博弈描述中发现连续的策略特征，基于这些特征对私有状态进行评分，并将它们聚类为抽象桶，在抽象构建过程中无需任何博弈特定的评估器、训练数据或博弈树遍历。该流水线分为四个阶段运行：带校准锚点的特征发现、批量私有状态评分、基于相关性的…（摘要原文在此处截断）

    arXiv:2609.04303v1 Announce Type: cross  Abstract: Information abstraction, which groups strategically similar private states into a tractable number of buckets, is essential for scaling game-solving algorithms to large imperfect-information games. Constructing effective abstractions, however, has traditionally required domain-specific evaluators such as hand-strength calculators or equity estimators, which demand expert knowledge and engineering effort and are unavailable for most less-studied games. We propose the Abstraction Agent, a zero-shot pipeline that uses a large language model (LLM) to discover continuous strategic features from a natural-language game description, score private states on these features, and cluster them into abstraction buckets, without any game-specific evaluator, training data, or game-tree traversal during abstraction construction. The pipeline runs in four phases: feature discovery with calibration anchors, batched private-state scoring, correlation-bas
    
[^88]: Harbor 适配器与 Harbor-Index：面向大规模智能体评估的基础设施与精选元数据集

    Harbor Adapters and Harbor-Index: Infrastructure and a Curated Meta-Dataset for Large-Scale Agentic Evaluation

    [https://arxiv.org/abs/2609.04298](https://arxiv.org/abs/2609.04298)

    本文提出了 Harbor Adapters 统一评估基础设施，将 80 多个智能体基准测试移植为可评估任意智能体的形式，并据此对 8 个模型进行大规模评估，同时推出经 AI 与人工双重审核筛选的包含 82 个高质量难题的精选元数据集 Harbor-Index。

    

    在数量不断增长的智能体（agentic）基准测试上评估智能体是一项挑战，因为这些基准通常需要复杂的环境和智能体集成方案。我们提出了 Harbor Adapters，一个面向智能体基准测试的统一评估基础设施。我们的工作包含三项贡献。第一，我们开发了一系列基准适配器，将 80 多个基准测试移植为可评估任意智能体的形式，并通过严格的代码审查和一致性实验对其进行了验证。第二，我们在 54 个基准测试上对横跨不同能力层级的 8 个模型进行了大规模评估；每个模型均使用 Terminus-2 以及 3 个原生测试框架之一运行。这使得对智能体能力和失败模式的更广泛分析成为可能。第三，我们推出了 Harbor-Index，一个精心策划的包含 82 个高难度、多样化且高质量任务的集合，覆盖 29 个基准测试，它是在适配后的基准套件基础上，通过难度筛选、AI 与人工审核以及审核-修复循环精炼而成。

    arXiv:2609.04298v1 Announce Type: new  Abstract: Evaluating agents on the growing number of agentic benchmarks is challenging because they often require complex environments and agent integrations. We introduce Harbor Adapters, a unified evaluation infrastructure for agentic benchmarks. Our work makes three contributions. First, we develop benchmark adapters that port more than 80 benchmarks to evaluate arbitrary agents, and validate them through rigorous code review and parity experiments. Second, we conduct a large-scale evaluation of 8 models spanning capability tiers across 54 benchmarks; every model is run with Terminus-2 and with one of 3 native harnesses. This enables a broader analysis of agent capabilities and failure modes than was previously possible. Third, we introduce Harbor-Index, a curated set of 82 difficult, diverse, and high-quality tasks spanning 29 benchmarks, refined from the adapted suite through difficulty filtering, AI and human audit, and an audit-and-fix loop
    
[^89]: 大语言模型中的证据整合

    Evidence Integration in Large Language Models

    [https://arxiv.org/abs/2609.04290](https://arxiv.org/abs/2609.04290)

    本文提出了一个关于大语言模型证据整合机制的分布理论，并通过千万级规模实验验证了三个预测：接收者眼中更可能的候选答案更具说服力、模型更易整合自身特有错误而非外来错误、以及相同证据可能改善弱模型却损害强模型。

    

    尽管人们越来越依赖借助工具、检索增强生成、其他智能体和用户提供的外部证据进行推理的大语言模型（LLMs），但LLMs如何将这些证据整合到它们已经初步形成的决策中，这在很大程度上仍不清楚。我们提出了一个分布理论，在该理论中，证据通过接收者先验权重和候选证据倾斜度来改变接收者初始答案的分布，并由此得出三个预测。第一，对接收者而言概率更高的候选答案更具说服力。第二，接收者更容易整合自身特有的错误，而不是来自不同来源的外来错误。第三，相同的证据可以改善较弱的模型，却会损害较强的模型。我们通过超过一千万次试验、来自四个系列的十二个LLMs以及八个领域（其中四个是物理与生命科学领域的科学发现任务：量子力学、物理学、遗传学和分子生物学）验证了这些预测。

    arXiv:2609.04290v1 Announce Type: cross  Abstract: Despite increasing reliance on LLMs that reason with external evidence supplied by tools, retrieval-augmented generation, other agents, and users, how LLMs integrate such evidence into decisions they have already begun to form remains largely unclear. We present a distributional theory in which evidence shifts the receiver's distribution of initial answers, driven by a receiver prior weight and a candidate evidence tilt, leading to three predictions. First, candidates more probable to the receiver are more persuasive. Second, receivers more readily integrate characteristic errors of their own than foreign errors from different sources. Third, identical evidence can improve weaker models and harm stronger ones. We confirm these over ten million trials, twelve LLMs from four families, and eight domains, four of them scientific discovery tasks in the physical and life sciences: quantum mechanics, physics, genetics, and molecular biology. 
    
[^90]: 记忆即转化：LETHE，一种自指涉的生成对抗网络启发的架构

    Memory as transformation: LETHE, a self-referential gan-inspired architecture

    [https://arxiv.org/abs/2609.04289](https://arxiv.org/abs/2609.04289)

    LETHE是一个受生成对抗网络启发的自指涉声音系统，通过判别器与随机扰动优化器的交互驱动音频混合矩阵参数在无外部数据或监督的封闭环境中自主演化。

    

    LETHE（具有时间分层准平衡的潜在参数演化）是一个在SuperCollider中实现的自指涉声音遗忘系统。它在封闭配置中采用生成对抗网络的形式化词汇，初始化后不使用外部数据集或监督。音频通过围绕两条延迟线构建的3×3混合矩阵进行处理；其九个系数和两个延迟时间通过五特征线性判别器与类似单样本REINFORCE的随机扰动优化器的相互作用而演化。判别器将当前能量行为与初始状态的存档进行比较，并指导参数更新。循环、固定和实时信号源可以独立混合。在消融对照的固定和循环会话中，活跃生成器对参数演化是必要的（所有15个消融会话中$\Delta c_{22}=0.000$）。该工作立足于……

    arXiv:2609.04289v1 Announce Type: new  Abstract: LETHE (Latent-parameter Evolution with Temporal Hierarchical quasi-Equilibrium) is a self-referential sonic-oblivion system implemented in SuperCollider. It adopts the formal vocabulary of Generative Adversarial Networks in a closed configuration without external datasets or supervision after initialization. Audio is processed by a 3 x 3 mixing matrix built around two delay lines; its nine coefficients and two delay times evolve through the interaction of a five-feature linear discriminator and a random-perturbation optimizer analogous to single-sample REINFORCE. The discriminator compares current energy behavior with an archive of the initial state and guides parameter updates. Circular, fixed, and live sources can be mixed independently. Across fixed and circular sessions with an ablation control, the active generator is necessary for parametric evolution ($\Delta c_{22}=0.000$ in all 15 ablation sessions). Situated in the tradition of
    
[^91]: EVOHARNESSBENCH：你的智能体能否跟上不断演进的运行框架？

    EVOHARNESSBENCH: Can Your Agents Keep Pace with an Evolving Harness?

    [https://arxiv.org/abs/2609.04280](https://arxiv.org/abs/2609.04280)

    提出了EVOHARNESSBENCH基准，首次将非平稳性从任务流转移到智能体运行框架本身（工具、技能、智能体）的持续演进上，用于评估智能体在框架不断变化环境下的适应与保留能力。

    

    现代基于大语言模型（LLM）的智能体通过一个由工具、可复用技能和专职智能体组成的运行框架来运行，该框架决定了智能体能够观察到什么以及能够做什么。在实践中，这个运行框架会随着新能力的加入而不断演进。我们提出了EVOHARNESSBENCH，这是一个在三个维度（工具、技能和智能体）上评估智能体在受控运行框架演进条件下表现的基准。与现有的智能体持续学习基准不同——后者通常将非平稳性（即随时间变化的内容）置于任务流中而保持运行框架固定不变——EVOHARNESSBENCH将非平稳性置于外部提供的运行框架本身。该基准包含17个由基于验证器的基准确定性构建的多阶段运行框架流，共涵盖802个任务、520个工具、42个技能和62个智能体。我们评估了对应于运行框架演进核心挑战的两种互补设置：部署评估（用于隔离测试保留能力）

    arXiv:2609.04280v1 Announce Type: cross  Abstract: Modern LLM-based agents operate through a harness of tools, reusable skills, and specialist agents that shapes what they observe and what they can do. In practice, this harness continually evolves as new capabilities are added. We introduce EVOHARNESSBENCH, a benchmark for evaluating agents under controlled harness evolution across three axes (tools, skills, and agents). Unlike existing continual-learning benchmarks for agents, which typically place non-stationarity (i.e., what changes over time) in the task stream while keeping the harness fixed, EVOHARNESSBENCH places non-stationarity in the externally supplied harness itself. It contains 17 multi-stage harness streams constructed deterministically from verifier-based benchmarks, comprising 802 tasks, 520 tools, 42 skills, and 62 agents. We evaluate two complementary settings corresponding to the central challenges of harness evolution: deployment evaluation, which isolates retention
    
[^92]: 评估大语言模型在强迫停电风险预测中的应用：优势及与机器学习的比较

    Evaluating Large Language Models for Forced Outage Risk Prediction: Benefits and Comparison to Machine Learning

    [https://arxiv.org/abs/2609.04272](https://arxiv.org/abs/2609.04272)

    本研究首次将大语言模型以零样本方式应用于电网天气相关停电风险预测，发现其精度虽略逊于监督机器学习模型，但在可操作推理和地理可扩展性方面具有独特优势，两者结合可能是最佳实践。

    

    本研究考察了大语言模型（LLMs）在零样本框架下预测配电网中与天气相关的强迫停电风险的能力，无需标注训练数据。该问题被表述为在三个预测时间尺度（3小时、6小时、12小时）上的二元严重程度分类任务，使用了德克萨斯州中部一个公用事业服务区六年的停电记录和高分辨率天气数据。四种零样本大语言模型与两种监督分类器在两种输入配置下进行了基准比较：一种使用当前天气观测数据，另一种使用天气预报数据。结果表明，监督模型在宏F1分数和精确率上优于大语言模型，而新一代大语言模型取得了具有竞争力的分数。除了准确性之外，大语言模型在可操作推理和地理可扩展性方面提供了互补优势，这表明将它们与监督模型相结合可能是最佳实践。

    arXiv:2609.04272v1 Announce Type: cross  Abstract: This study examines the ability of large language models (LLMs) to predict the risk of weather-related forced outages in the distribution grid in a zero-shot framework, without labeled training data. The problem is formulated as a binary severity classification task across three forecast horizons (3h, 6h, 12h), using six years of outage records and high-resolution weather data for a utility service area in central Texas. Four zero-shot LLMs are benchmarked against two supervised classifiers across two input configurations: one using current weather observations and the other using weather forecast data. Results show that supervised models outperform LLMs on macro-F1 and precision, while newer LLM generations achieve competitive scores. Beyond accuracy, LLMs offer complementary strengths in actionable reasoning and geographic scalability, suggesting that combining them with supervised models may be the best practice.
    
[^93]: 评审者能力决定的是拒绝定向而非修复技巧：来自LLM执行-评审-修订流水线的证据

    Reviewer Capability Governs Rejection Targeting, Not Repair Skill: Evidence from LLM Execute-Review-Revise Pipelines

    [https://arxiv.org/abs/2609.04270](https://arxiv.org/abs/2609.04270)

    在LLM执行-评审-修订流水线中，评审者能力决定的是拒绝定向的准确性而非修复技能——跨家族的中档评审者可将准确率从52%提升至64%且零答案损害，而同模型自我审查虽错误检测率最高却收益不显著。

    

    多智能体LLM流水线日益将不同角色（包括执行与验证）分配给不同能力层级的模型，这是因为每个阶段都运行旗舰模型成本过高。已有文献表明验证阶段并非总是有益，但其将评审者的能力相对于执行者大致保持不变，而我们改变了这一设定：我们将评审者替换为覆盖一个能力区间的多种模型——甚至包括完全无法解决这些问题的模型——并测量每一次拒绝所导致的结果。实验在一个固定的100道奥林匹克数学题集合上进行。结果显示，一个来自不同家族的中档评审者将最终准确率提升了12个百分点，从52%提高到64%（p = 0.0005），且没有损坏任何答案。同模型自我审查在所有条件下拥有最高的错误检测率（0.85召回率），却没有带来显著收益：其拒绝的频率是前者的2.1倍，而修复率却只有三分之一……

    arXiv:2609.04270v1 Announce Type: cross  Abstract: Multi-agent LLM pipelines increasingly assign roles, including execution and verification, to models of different capability tiers. This is done because running a flagship model at every stage is expensive. Previous literature has established that verification stages are not always beneficial, but holds reviewer capability roughly fixed relative to the executor. We vary it. We replace the reviewer with models spanning a capability range down to one that cannot solve the problems at all, and measure the outcome of every individual rejection. This is done across a constant set of 100 olympiad mathematics problems.   A cross-family mid-tier reviewer improves final accuracy by 12 percentage points, from 52 to 64 percent (p = 0.0005), with zero damaged answers. Same-model self-review attains the highest error-detection rate of any condition (0.85 recall) yet yields no significant gain: it rejects 2.1 times as often for a third the repair ra
    
[^94]: 通过低秩注意力适配恢复量化KV缓存的质量

    Quality Recovery for Quantized KV Caches via Low-Rank Attention Adaptation

    [https://arxiv.org/abs/2609.04263](https://arxiv.org/abs/2609.04263)

    该论文提出一种在量化器固定的情况下，将浮点缓存模型行为蒸馏到低秩Q/K/V投影适配器中的方法，能大幅恢复量化KV缓存造成的模型质量损失。

    

    低比特键值（KV）缓存可以减少自回归解码所需的内存，但由此产生的质量损失取决于模型和量化器。我们在保持量化器固定的情况下，将浮点缓存模型的行为蒸馏到低秩Q/K/V投影更新中，同时学生模型执行物理打包的增量缓存。在三个随机种子下，4比特仿射缓存适配器在TinyLlama-1.1B上恢复了54.24%±2.47%的保留困惑度差距，在Gemma-4-12B上恢复了75.96%±4.04%。在同一个冻结的NF4 Llama-3.1-8B基座上，每个量化器选取一次验证选定的运行，在KIVI K2V2下恢复了60.42%，在KVarN K4V2下恢复了37.61%，同时保持了180个案例的联想检索能力。Gemma在官方4K/8K RULER子集上的得分从使用未适配4比特缓存时的42.80提升至适配后的48.33（浮点模型为46.15），且存在显著的任务异质性。最后，一项2比特的秩-令牌扫描将TinyLlama的2比特困惑度从576降低。

    arXiv:2609.04263v1 Announce Type: cross  Abstract: Low-bit key--value (KV) caches reduce the memory required for autoregressive decoding, but the resulting quality loss depends on the model and quantizer. We keep the quantizer fixed and distill the floating-cache model's behavior into low-rank Q/K/V projection updates while the student executes a physically packed incremental cache. Across three seeds, 4-bit affine-cache adapters recover $54.24\%\pm2.47\%$ of the held-out perplexity gap on TinyLlama-1.1B and $75.96\%\pm4.04\%$ on Gemma-4-12B. On the same frozen NF4 Llama-3.1-8B base, one validation-selected run per quantizer recovers $60.42\%$ under KIVI K2V2 and $37.61\%$ under KVarN K4V2, while preserving 180-case associative retrieval. Gemma's score on an official 4K/8K RULER subset rises from 42.80 with the unadapted 4-bit cache to 48.33 after adaptation (46.15 floating), with substantial task heterogeneity. Finally, a 2-bit rank--token sweep reduces TinyLlama's 2-bit PPL from 576.
    
[^95]: 面向多语言口述历史研究的自动语音识别

    Automatic Speech Recognition for Multilingual Oral History Research

    [https://arxiv.org/abs/2609.04232](https://arxiv.org/abs/2609.04232)

    该论文评估了Whisper自动语音识别工具在新西兰粤语口述历史这一语码转换语境下的转录效果，最佳模型配置的词错误率为12.10，且仅需人工转录约1%的时间，为社区主导的语言保护与振兴工作提供了高效的初步转录方案。

    

    本文从一个独特的视角探讨了语音技术如何被社区主导的传承语言保护与振兴倡议所采用。作为一种社区主导的语言维护策略，口述历史在新西兰粤语振兴工作中发挥着至关重要的作用。自动语音识别（ASR）工具包（如Whisper）的发展，加速了传统上耗费大量资源和时间的口述历史资料转录过程。然而，关于ASR工具包在语码转换语言环境下有效性的研究仍然有限。基于词错误率（WER）的评估显示，表现最佳的Whisper模型配置实现了12.10的词错误率，但其代价是无法准确转录不支持的非英语片段。尽管如此，Whisper仍然是一个有用的工具，它所提供的初步转录仅需人工转录所需估计时间的1%。

    arXiv:2609.04232v1 Announce Type: cross  Abstract: This paper offers a unique perspective on how speech technologies are being adopted by community-led heritage language preservation and revitalisation initiatives. As a community-led language maintenance strategy, oral histories play a crucial role in Cantonese language revitalisation in New Zealand. The development of Automatic Speech Recognition (ASR) toolkits, such as Whisper, have expedited what has often been a resource and time-intensive process of transcribing oral history collections. However, there is limited research into the effectiveness of ASR toolkits when applied to code-switched language contexts. Based on Word Error Rate (WER), the best performing Whisper model configuration achieved a WER of 12.10 at the expense of accurately transcribing unsupported non-English segments. However, Whisper remains a useful tool by providing a first-pass transcription using only 1% of the estimated time otherwise needed for manual trans
    
[^96]: 语料库选择在多大程度上会改变依存距离的估计结果？

    How Much Does Corpus Choice Change Dependency-Distance Estimates?

    [https://arxiv.org/abs/2609.04223](https://arxiv.org/abs/2609.04223)

    该研究通过比较38个同语言树库对发现，语料库的选择会显著影响依存距离估计——跨树库一致性仅为中等、近40%的语言排序会因替换树库而逆转，但所有树库均一致支持依存长度最小化趋势，表明平均依存距离更宜被视为受语料库条件制约的复合指标而非语言本身的固有属性。

    

    从单一语料库中得出的依存距离估计通常被视为某种语言自身的属性，然而这一假设尚未在独立编纂的语料库之间得到检验。我们使用一致性相关分析、Bland-Altman分析以及包含十二种设定的多元宇宙设计，比较了通用依存树库v2.18中38个同语言树库对的平均依存距离估计结果。跨树库的一致性充其量只达到中等水平：用另一个树库替代某个树库会逆转近40%的语言两两排序，且树库选择解释了约29%的组间方差。这种不一致性远超树库内部的抽样误差，并且在所有十二种预处理设定中持续存在。然而，每一个树库都证实了依存长度最小化的存在（归一化比率低于1）。这些数据更符合将MDD（平均依存距离）视为一种受语料库条件制约的语法与语用因素复合指标的观点。

    arXiv:2609.04223v1 Announce Type: new  Abstract: Dependency-distance estimates derived from a single corpus are routinely treated as properties of a language, yet this assumption has not been tested across independently compiled corpora. We compared mean dependency-distance estimates across 38 same-language treebank pairs from Universal Dependencies v2.18, using concordance correlation, Bland-Altman analysis, and a twelve-specification multiverse design. Cross-treebank agreement was moderate at best: substituting one treebank for another reversed nearly 40 percent of pairwise language orderings, and treebank choice accounted for roughly 29 percent of between-group variance. This disagreement substantially exceeded within-treebank sampling error and persisted across all twelve preprocessing specifications. Nevertheless, every treebank confirmed dependency-length minimization (normalized ratio below 1). The data are more consistent with MDD as a corpus-conditioned composite of grammatica
    
[^97]: GEPARD——面向实时对话的生成式、韵律感知、自回归文本转语音模型

    GEPARD - Generative, Prosody-aware, Autoregressive text-to-speech model for Realtime Dialogue

    [https://arxiv.org/abs/2609.04222](https://arxiv.org/abs/2609.04222)

    GEPARD是一个基于标准LLM骨干网络、可在不修改vLLM推理引擎计算内核的情况下流式运行的自回归文本转语音模型，通过将零样本语音克隆等辅助机制移出解码循环，实现了真正的实时语音对话生成。

    

    我们提出了GEPARD（面向实时对话的生成式、韵律感知、自回归文本转语音模型），这是一个用于实时语音对话的流式文本转语音模型。GEPARD使用LLM骨干网络自回归地生成语音——文本和音频嵌入在单一的仅解码器（decoder-only）模型中共同训练——并通过基于FSQ（有限标量量化）的神经编解码器将其解码为波形，随着文本的到达逐块流式输出音频。我们的核心目标是构建一个可以由标准LLM推理引擎（vLLM）提供服务而无需修改其计算内核的TTS架构。这定义了总体设计原则：骨干网络是标准的全注意力transformer，而所有非平凡的辅助机制——零样本语音克隆、文本增强和分类器无关引导——都被移出自回归解码循环，转移到预填充阶段，或直接蒸馏到模型权重中。在流式端到端推理中，单一流达到的实时因子为……（原文摘要在此处截断）

    arXiv:2609.04222v1 Announce Type: cross  Abstract: We present GEPARD (Generative, Prosody-aware, Autoregressive text-to-speech model for Realtime Dialogue), a streaming text-to-speech model for real-time spoken dialogue. GEPARD generates speech autoregressively with an LLM backbone - text and audio embeddings are trained together in a single decoder-only model - and decodes it to a waveform with an FSQ-based neural codec, streaming audio chunk-by-chunk as text arrives.   Our central goal is a TTS architecture served by a standard LLM engine (vLLM) without modifying its compute kernels. This defines the overarching design principle: the backbone is a standard full-attention transformer, while all non-trivial auxiliary mechanisms - zero-shot voice cloning, text augmentation, and classifier-free guidance - are moved out of the autoregressive decode loop into prefill, or distilled directly into the weights.   On streaming end-to-end inference, a single stream reaches a Real-Time Factor of 
    
[^98]: 语音AI客户服务中的偏见与安全审计

    Auditing Bias and Safety in Voice AI Customer Care

    [https://arxiv.org/abs/2609.04206](https://arxiv.org/abs/2609.04206)

    本文提出了一个验证门控的审计框架，用于检测语音AI客户服务系统中因来电者口音、情感等呈现线索引发的偏见与安全问题，能够捕捉最终拒绝发生之前以额外服务负担形式出现的不公平伤害。

    

    语音AI系统日益成为客户服务互动的中介，在这些互动中，来电者的呈现线索（如口音、情感、流利度和紧急程度）与服务请求一并存在。现有的公平性与安全性评估涵盖了语音识别差异、口语对话偏见和语音代理能力等方面，但很少将客户服务语音代理视为有状态、多轮次、工具中介的系统，而在这类系统中，伤害可能在任何最终拒绝发生之前就以额外负担的形式出现。我们为此类系统形式化了一个验证门控的审计框架。该框架（i）区分原生语音到语音、级联式ASR-语言模型-TTS以及混合工具中介三种架构；（ii）在受控的来电者呈现条件下使用匹配的服务事实；（iii）在推理之前验证事实不变性、呈现线索、伪影和声学测量；（iv）同时记录实质性结果与服务负担路径。我们定义了

    arXiv:2609.04206v1 Announce Type: cross  Abstract: Voice AI systems increasingly mediate customer care interactions where caller presentation cues such as accent, affect, fluency, and urgency are available alongside the service request. Existing fairness and safety evaluations cover speech recognition disparities, spoken dialogue bias, and voice agent capability, but rarely treat customer care voice agents as stateful, multi turn, tool mediated systems where harm can appear as additional burden before any final denial occurs. We formalize a validation gated audit framework for such systems. The framework (i) separates native speech to speech, cascaded ASR to language model to TTS, and hybrid tool mediated architectures; (ii) uses matched service facts across controlled caller presentation conditions; (iii) validates fact invariance, presentation cues, artifacts, and acoustic measurements before inference; and (iv) records both material outcomes and path to service burden. We define the
    
[^99]: 顺序优于联合：论在线策略蒸馏与RLVR的相互作用

    Sequential Beats Joint: On the Interplay between On-Policy Distillation and RLVR

    [https://arxiv.org/abs/2609.04108](https://arxiv.org/abs/2609.04108)

    先蒸馏后强化学习的两阶段训练方案在推理任务上持续优于纯OPD、纯RLVR及所有联合优化方法，因为OPD先扩大学生对教师解的覆盖范围、RL再在其内锐化，而联合训练会导致两种信号相互干扰。

    

    可验证奖励强化学习（RLVR）和在线策略蒸馏（OPD）已成为对推理大语言模型进行后训练的两种主流方法。先前的工作利用OPD的密集token级监督来补充稀疏的RL奖励，在单个步骤内融合这两种信号：要么作为加权加性组合，要么作为对RL优势的教师调制重缩放。在本文中，我们展示了一个简单的两阶段方案——先OPD后RL——在逻辑和数学推理基准上持续优于纯OPD、纯RLVR以及所有此类联合基线方法。除了实证结果外，我们还通过pass@$k$行为、学习动态和参数更新对这一现象提供了系统性的理解，并得出一个一致的解释：OPD扩大了学生对教师支持解的覆盖范围，而RL则在该支持范围内进行锐化，同时联合优化这两种信号会导致它们相互干扰。

    arXiv:2609.04108v1 Announce Type: cross  Abstract: Reinforcement learning with verifiable rewards (RLVR) and on-policy distillation (OPD) have emerged as two dominant methods for post-training reasoning LLMs. Prior work uses OPD's dense token-level supervision to complement the sparse RL reward, fusing the two signals within a single step: either as a \emph{weighted-additive combination} or a \emph{teacher-modulated rescaling} of the RL advantage. In this paper, we show that a simple two-stage scheme, OPD-then-RL, consistently outperforms pure OPD, pure RLVR, and all such joint baselines across logic and math reasoning benchmarks. Beyond the empirical results, we further provide a systematic understanding of this through pass@$k$ behavior, learning dynamics, and parameter updates, yielding a consistent explanation: OPD expands the student's coverage of teacher-supported solutions and RL sharpens within that support, while jointly optimizing the two signals causes them to interfere.To p
    
[^100]: 可编辑的视觉设计

    Editable Visual Design

    [https://arxiv.org/abs/2609.04034](https://arxiv.org/abs/2609.04034)

    该论文提出“可编辑的视觉设计”新范式，以编码智能体为核心，将VLM作为“创意大脑”进行需求理解与审美判断，将图像生成模型作为按需的“视觉世界模拟器”合成独立资产，并通过“先想象、后行动”的闭环工作流编写原生HTML/CSS，实现支持图层级精确后编辑的视觉设计。

    

    尽管 GPT-Image-2 和 Nano-Banana 等扩散基础模型展现出卓越的视觉表现力，但它们的端到端生成本质上会产生文本易出错的扁平位图，导致无法进行图层级的后编辑。相反，通过编码智能体进行基于代码的视觉生成能够提供精确的布局控制和相互解耦的图层，但仍受限于缺乏全局审美直觉以及编写复杂视觉资产的困难。为解决这些问题，我们提出了“可编辑的视觉设计”，这是一种由编码智能体驱动的新范式。我们将视觉语言模型（VLM）指定为负责需求理解、任务规划和审美判断的“创意大脑”，同时利用图像生成模型作为按需调用的“视觉世界模拟器”来合成独立的视觉资产。在“先想象、后行动”的闭环工作流下，智能体生成相互隔离的资产、编写原生 HTML/CSS 代码，并进行迭代式优化。

    arXiv:2609.04034v1 Announce Type: cross  Abstract: While diffusion base models such as GPT-Image-2 and Nano-Banana exhibit remarkable visual expressiveness, their end-to-end generation inherently yields flattened bitmaps with error-prone text, precluding layer-wise post-editing. Conversely, code-based visual generation via Coding Agents provides precise layout control and decoupled layers, yet remains constrained by a lack of global aesthetic intuition and the difficulty of coding complex visual assets.   To address this, we propose Editable Visual Design, a new paradigm driven by a Coding Agent. We designate the VLM as the ``creative brain'' for requirement comprehension, task planning, and aesthetic judgment, while utilizing the image generation model as an on-demand ``visual world simulator'' to synthesize standalone visual assets. Operating under an ``imagine first, then act'' closed-loop workflow, the agent generates isolated assets, writes native HTML/CSS, and iteratively refines
    
[^101]: 固定后缀依赖比率：量化拉脱维亚语外来词性属分配的双轨机制

    Fixed Suffix Dependency Ratio: Quantifying the Dual-Track Mechanism of Gender Assignment in Latvian Loanwords

    [https://arxiv.org/abs/2609.03930](https://arxiv.org/abs/2609.03930)

    本研究提出固定后缀依赖比率（FSDR）这一量化指标，揭示了拉脱维亚语英语外来词性属分配的双轨机制——阴性外来词显著依赖固定派生后缀，而阳性外来词集中于自由选择区域。

    

    现有研究反复观察到英语外来词在不同接受语言中倾向于聚集为阳性的现象，但由于固定的形态规则和默认分配常常被放在一起分析，这种模式的起源仍难以确定。本研究提出固定后缀依赖比率（FSDR），用以量化不同性属对固定派生后缀的依赖程度，并区分分布中的形态锚定与自由选择。通过考察1,832个拉脱维亚语名词词元类型，结果揭示了外来词系统内部显著的FSDR不对称性：阴性外来词显著更依赖固定派生后缀，而阳性外来词更多集中在自由选择区域。这一模式表现出外来词特异性，并且在当代使用中变得更加明显。因此，FSDR提供了一个量化框架

    arXiv:2609.03930v1 Announce Type: new  Abstract: Existing research has repeatedly observed the tendency for English loanwords to cluster in the masculine gender across different recipient languages, yet the origin of this pattern remains difficult to determine, as fixed morphological rules and default assignments are frequently analysed together. This study proposes the Fixed Suffix Dependency Ratio (FSDR) to quantify the degree of reliance on fixed derivational suffixes across different genders, and to distinguish between morphological anchoring and free-choice in distribution. By examining 1,832 Latvian noun lemma types, the results reveal a significant FSDR asymmetry within the loanword system: feminine loanwords rely significantly more on fixed derivational suffixes, while masculine loanwords are more concentrated in the free-choice zone. This pattern exhibits loanword specificity and has become more pronounced in contemporary usage. FSDR therefore provides a quantitative framework
    
[^102]: VisCAD：具备多模态工业CAD智能的基础模型套件

    VisCAD: A Foundation Model Suite with Multimodal Industrial CAD Intelligence

    [https://arxiv.org/abs/2609.03811](https://arxiv.org/abs/2609.03811)

    VisCAD是一个面向工业CAD的基础模型套件，其核心270亿参数模型VisCAD-M1能够将渲染图、文本、2D图纸和真实照片等多种输入转换为可执行的CAD程序，在保证广泛泛化能力的同时具备强大的专业CAD设计与装配生成能力。

    

    面向工业产品的AI辅助计算机辅助设计（CAD）涉及两个具有挑战性的阶段。零件级生成将多种形式的用户意图——包括渲染图、文本描述、二维图纸和真实照片——映射为CAD领域特定语言的可执行程序。装配级生成则还需要处理相互作用的零件、规划配合关系、估计位姿并正确放置所有零件。现有的专用CAD模型通常只在狭窄的输入域（如渲染图或文本）上训练，泛化能力往往较差；而通用前沿模型虽然覆盖更广泛的输入，但在CAD各领域上的表现并不稳定。我们提出了VisCAD，一个基础模型套件，旨在为真实工业产品同时提供广泛的泛化能力和强大的CAD专业能力。其核心是VisCAD-M1，一个经过中期训练和后训练的270亿参数模型，用于零件级设计生成。在PubCADBench上（摘要截断）……

    arXiv:2609.03811v1 Announce Type: cross  Abstract: AI-assisted computer-aided design (CAD) for industrial products involves two challenging phases. Part-level generation maps diverse forms of user intent, including renders, text descriptions, 2D drawings, and real photographs, to executable programs in a CAD domain-specific language. Assembly-level generation must additionally handle interacting parts, plan mating relations, estimate poses, and place all parts correctly. Existing specialized CAD models are commonly trained on narrow input domains, such as renders or texts, and often generalize poorly, while general-purpose frontier models cover broader inputs but perform inconsistently across CAD domains. We present VisCAD, a foundation model suite designed to provide both broad generalization and strong CAD capability for realistic industrial products. At its core is VisCAD-M1, a 27B model trained through mid-training and post-training for part-level design generation. On PubCADBench 
    
[^103]: IndicSafeEval：多语言说服性越狱攻击下大语言模型的安全稳健性

    IndicSafeEval: Safety Robustness of Large Language Models under Multilingual Persuasive Jailbreak Attacks

    [https://arxiv.org/abs/2609.03781](https://arxiv.org/abs/2609.03781)

    该论文提出了IndicSafeEval框架，通过四种印度语言、十个安全类别和六种说服策略构建7,200条对抗性提示，系统评估并揭示了大语言模型在面对多语言说服性越狱攻击时安全表现存在显著差异。

    

    大语言模型在多语言环境中的应用日益广泛，但其安全性评估主要仍以英语为主。这限制了我们对对齐失效在低资源和文化多样性语言中如何表现的理解。我们提出了IndicSafeEval，一个针对印度语言的说服性越狱攻击评估框架。该基准将十个安全关键内容类别与六种类人说服策略相结合，涵盖印地语、孟加拉语、马拉地语和旁遮普语四种不同的印度语言，共生成7,200条对抗性提示。我们对多个开源大语言模型进行了系统性的黑盒评估，以检验其安全行为如何随语言、说服策略和风险类别的变化而变化。我们的分析表明，模型并非在所有语言和提示风格下都表现得同样安全，相反，安全性能在很大程度上取决于所使用的语言以及提示的构造方式。

    arXiv:2609.03781v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used in multilingual settings, yet their safety is still evaluated primarily in English. This limits our understanding of how alignment failures manifest in low-resource and culturally diverse languages. We introduce IndicSafeEval, a persuasion-based jailbreak evaluation framework for Indian languages. Our benchmark combines ten safety critical content categories with six human-like persuasive strategies across four different Indian languages, such as Hindi, Bengali, Marathi and Punjabi, resulting in 7,200 adversarial prompts. We conduct a systematic black-box evaluation of several open-source LLMs to examine how their safety behaviour varies across languages, persuasion strategies, and risk categories. Our analysis shows that the model does not behave equally safely across all languages and prompt styles. Instead, safety performance depends strongly on both the languages used and the way a
    
[^104]: RealCADBench：基于工业设计意图的参数化CAD建模基准测试

    RealCADBench: Benchmarking Parametric CAD Modeling from Industrial Design Intents

    [https://arxiv.org/abs/2609.03773](https://arxiv.org/abs/2609.03773)

    提出了RealCADBench基准测试，基于19个工厂自动化类别的真实工业设计意图，通过文本、图纸、图片等多种输入模态和可执行性、IoU、视觉语义一致性等综合评估指标，系统性地评估从设计意图到程序化参数CAD建模的能力。

    

    参数化计算机辅助设计（CAD）建模难以用单一指标进行评估。现有的CAD基准测试往往侧重于合成或CAD原生环境、有限的输入模态，或仅关注可执行性和IoU（交并比）。我们提出了RealCADBench，这是一个从真实工业设计意图到程序化CAD建模的基准测试。它包含来自19个工厂自动化类别的12,632个任务，涵盖文本描述、2D工程图纸、真实产品图片和渲染图像等多种输入模态，同时支持零件和装配体建模。我们在一个包含1,770个任务的评估切片上报告结果：包括跨四种输入模式的1,745个零件任务，以及RCB-Assm25（一个包含25个任务的装配体研究），用于所有已报告的装配体比较。每种方法生成FreeCAD API的Python代码，由共享运行时执行以导出3D模型。我们使用可执行性、Solid IoU、Surface IoU以及基于评分标准的视觉-语义一致性Judge来评估导出的模型。

    arXiv:2609.03773v1 Announce Type: cross  Abstract: Parametric computer-aided design (CAD) modeling is difficult to evaluate with a single metric. Existing CAD benchmarks often emphasize synthetic or CAD-native settings, limited input modalities, or executability and IoUs alone. We introduce RealCADBench, a benchmark for intent-to-program CAD modeling from real industrial design intents. It contains 12,632 tasks from 19 factory-automation categories and spans text descriptions, 2D engineering drawings, real product pictures, and rendered images for both Part and Assembly modeling. We report results on a 1,770-task evaluation slice: 1,745 Part tasks across four input regimes and RCB-Assm25, a 25-task assembly study used in every reported assembly comparison. Each method generates FreeCAD API Python, which a shared runtime executes to export the 3D model. We evaluate the exported model using executability, Solid IoU, Surface IoU, and a rubric-based visual-semantic identity Judge. Among th
    
[^105]: 多步临床大语言模型智能体的反事实公平性审计需要测量每动作的不稳定性底线

    Counterfactual Fairness Audits of Multi-Step Clinical LLM Agents Require a Measured Per-Action Instability Floor

    [https://arxiv.org/abs/2609.03221](https://arxiv.org/abs/2609.03221)

    临床LLM智能体在完全相同输入下本身就存在显著的动作不稳定性（约8.7%），因此反事实公平性审计必须先测量这一“每动作不稳定性底线”，否则任何检测到的人口统计学差异都无法解释。

    

    反事实审计是检查临床智能体是否对人口统计学上不同但临床上相同的患者采取不同行动的标准工具。这类审计报告一个“翻转率”：当仅改变患者描述时，智能体行动发生改变的频率。我们证明这一指标本身是不可解释的。在16个病例情景上将完全相同的条件重复运行十次（相同叙述、相同描述字符串、不改变任何变量），临床智能体的行动在8.7%的结果-情景单元格中发生了改变，且不稳定性在不同行动之间呈现8倍的异质性，从ICU升级决策的0.022到受管制物质谨慎建议的0.179。我们数据中没有任何人口统计学对比能够与这一底线区分开。第二个模型给出了6.7%的合并底线，且对六种行动的不稳定性排序几乎完全一致（Spearman 0.94，精确p=0.017），说明该底线并非单一系统的特有产物。对五次抽样进行多数投票聚合可以消除其中39%的不稳定性……

    arXiv:2609.03221v1 Announce Type: new  Abstract: Counterfactual audits are the standard tool for checking whether a clinical agent treats demographically distinct but clinically identical patients differently. They report a flip rate: how often an action changes when only the patient descriptor changes. We show that this quantity is uninterpretable on its own. Re-running an identical condition ten times over sixteen vignettes (same narrative, same descriptor string, nothing varied) moved a clinical agent's action in 8.7% of outcome-vignette cells, and instability was heterogeneous across actions by a factor of eight, from 0.022 for ICU escalation to 0.179 for controlled-substance caution. No demographic contrast in our data was distinguishable from that floor. A second model gives a pooled floor of 6.7% and ranks the six actions almost identically (Spearman 0.94, exact p=0.017), so the floor is not one system's artefact. Majority-vote aggregation over five draws removes 39% of it and t
    
[^106]: 面向编程竞赛金牌表现的语言模型后训练

    Post-Training Language Models for Gold-Medal Performance in Coding Competitions

    [https://arxiv.org/abs/2609.02849](https://arxiv.org/abs/2609.02849)

    该研究通过结合大规模题目筛选、监督微调、强化学习以及反馈驱动的测试时计算策略 GenCorrect，使语言模型在 IOI 2025 编程竞赛中取得了超越金牌分数线（438.3 分）的成绩（Nano-CC 达 468 分，Ultra-CC 达 502 分）。

    

    竞赛编程已成为检验大语言模型推理能力的关键测试，其中 IOI 和 ICPC 等国际赛事代表了最具挑战性的场景。我们提出了一条端到端的专门化流水线，结合了大规模题目筛选、合成推理轨迹、监督微调（SFT）和强化学习（RL）。利用 22,000 道精选题目，我们通过 SFT 和 RL 训练了 Nemotron-3-Nano-CC（30B-A3B），并仅通过 SFT 训练了 Nemotron-3-Ultra-CC（550B-A55B）。我们进一步提出了 GenCorrect，这是一种由反馈驱动的测试时计算策略，可迭代地生成、评估并改进多样化的解决方案。在 IOI 2025 上，Nano-CC 在后训练后从 130 分提升至 291 分，结合 GenCorrect 后达到 468 分，超过了 438.3 的金牌分数线，而 Ultra-CC 达到了 502 分。在这些结果的指导下，我们开发了一个面向竞赛的 Ultra-CC 系统，并在 IOI 2026 期间进行了前瞻性评估。

    arXiv:2609.02849v1 Announce Type: cross  Abstract: Competitive programming has become a key test of large language model reasoning, with international competitions such as IOI and ICPC representing its most challenging settings. We present an end-to-end specialization pipeline combining large-scale problem curation, synthetic reasoning traces, supervised fine-tuning (SFT), and reinforcement learning (RL). Using 22,000 curated problems, we train Nemotron-3-Nano-CC (30B-A3B) with SFT and RL and Nemotron-3-Ultra-CC (550B-A55B) with SFT alone. We further introduce GenCorrect, a feedback-driven test-time compute strategy that iteratively generates, evaluates, and refines diverse solutions. On IOI 2025, Nano-CC improves from 130 points to 291 after post-training and to 468 with GenCorrect, exceeding the gold threshold of 438.3 while Ultra-CC reaches 502. Guided by these results, we develop a competition-specific Ultra-CC system and evaluate it prospectively during IOI 2026. Under the same ti
    
[^107]: 从词元到语义：利用互补信号检测黑盒大语言模型中的幻觉

    From Tokens to Semantics: Leveraging Complementary Signals for Hallucination Detection in Black-Box LLMs

    [https://arxiv.org/abs/2609.02679](https://arxiv.org/abs/2609.02679)

    该论文针对无参考文档的黑盒大语言模型，提出联合利用语义熵与词元级不确定性这两种互补信号（包括TopK聚合、CoCoA混合方法及Gated等监督方法）来更准确地检测幻觉。

    

    当大语言模型支持面向公众或高风险的工作流程时，遗漏的虚假编造内容可能损害用户和机构的利益，而误报则会消耗有限的人工审核资源。在缺乏可信上下文或参考文档的情况下，我们研究了两种可通过黑盒模型API获取的信号：语义熵（衡量采样响应含义之间的分歧程度）和由词元对数概率导出的不确定性。这两种信号的失效模式可以互补：当所有响应形成单一语义簇时，语义熵将失去信息量，而词元不确定性则可能遗漏模型始终自信的错误。我们通过TopK方法聚合采样响应中的词元级信号，扩展了基于词元的不确定性检测；评估了结合目标响应不确定性与语义差异性的混合方法CoCoA；并提出并研究了两种监督方法：Gated（将单簇情形路由至聚合的……

    arXiv:2609.02679v1 Announce Type: cross  Abstract: When LLMs support public-facing or high-stakes workflows, missed fabrications can harm users and institutions, while false alarms consume limited human-review capacity. When no trusted context or reference document is available, we study two signals accessible through black-box model APIs: semantic entropy, which measures disagreement among sampled response meanings, and uncertainty derived from token log-probabilities. Their failure modes can be complementary: semantic entropy becomes uninformative when responses form one semantic cluster, while token uncertainty can miss consistently confident errors. We extend token-based uncertainty detection by aggregating token-level signals across sampled responses through our TopK method, evaluate the hybrid CoCoA method, which combines target-response uncertainty with semantic dissimilarity, and propose and study two supervised methods: Gated, which routes single-cluster cases to an aggregated
    
[^108]: Enoki：高效的多层级幻觉检测

    Enoki: Efficient Multi-Level Hallucination Detection

    [https://arxiv.org/abs/2609.00581](https://arxiv.org/abs/2609.00581)

    Enoki提出了一种基于开放信息抽取的多层级幻觉检测框架，通过抽取文本锚定的关系事实并进行验证，无需额外的声明-片段对齐即可同时实现声明级验证和片段级定位，并支持LLM、编码器和规则三种抽取方式以平衡准确性与推理成本。

    

    在高风险场景中部署大语言模型（LLM）时，确保事实性仍然是一项关键挑战。现有的幻觉检测器通常仅在单一层级上运作：声明级方法提供可解释的事实单元，而片段级方法则定位不受支持的文本。弥合这两种视角的代价高昂，因为依赖大量LLM的流水线需要多次分解和验证调用，而模块化系统则需要额外的声明到片段的对齐。我们提出了Enoki，一个用于多层级幻觉检测的开放信息抽取框架。Enoki抽取以文本为锚点的关系事实，将其与证据进行核对验证，并将不受支持的事实投影回幻觉片段。这种共享表示使得声明级验证和片段级定位无需单独的对齐步骤即可实现。Enoki支持基于LLM的、基于编码器的以及基于规则的抽取方式，通过统一接口在准确性和推理成本之间取得平衡。实验……（原文摘要在此处被截断）

    arXiv:2609.00581v1 Announce Type: new  Abstract: Ensuring factuality remains a critical challenge for deploying LLMs in high-stakes settings. Existing hallucination detectors usually operate at a single level: claim-level methods provide interpretable factual units, while span-level methods localize unsupported text. Bridging these views is costly, as LLM-heavy pipelines require multiple decomposition and verification calls, and modular systems need additional claim-to-span alignment. We propose Enoki, an Open Information Extraction framework for multi-level hallucination detection. Enoki extracts text-anchored relational facts, verifies them against evidence, and projects unsupported facts back to hallucinated spans. This shared representation enables claim-level verification and span-level localization without requiring separate alignment. Enoki supports LLM-based, encoder-based, and rule-based extraction regimes, balancing accuracy and inference cost through a common interface. Expe
    
[^109]: 面向大语言模型时序评估与知识更新的合成世界

    Synthetic Worlds for Temporal Evaluation and Knowledge Updating in LLMs

    [https://arxiv.org/abs/2609.00184](https://arxiv.org/abs/2609.00184)

    该论文提出了一个模拟驱动的合成框架，通过虚构未来世界的 ParallelEvents 基准避免评估污染，并利用 Synapse 训练框架（结合中期训练与指令微调）实现大语言模型的可扩展知识更新，性能比现有方法提升 14.23%。

    

    大语言模型（LLM）依赖于静态的预训练语料库，导致其知识随时间推移而变得过时。现有的知识编辑评估方法要么容易遭受快速的数据污染，要么依赖于与现有刚性知识相冲突的反事实编辑。在本工作中，我们提出了一个合成的、模拟驱动的框架，用于研究大语言模型中的知识插入。我们引入了 {\sc ParallelEvents}，这是一个由虚构但逼真的未来世界构成的基准，能够生成连贯的事件轨迹以进行受控评估，在避免污染的同时保持一致性。基于该数据集，我们开发了 {\sc Synapse}，这是一个利用模型自身生成的数据、通过中期训练（mid-training）和指令微调来更新模型参数的训练框架。这一合成流程实现了可扩展的知识整合，而无需昂贵的人工策划数据。实验结果表明，{\sc Synapse} 的性能比现有方法高出 14.23%。

    arXiv:2609.00184v1 Announce Type: new  Abstract: Large language models (LLMs) rely on static pretraining corpora, causing their knowledge to become outdated over time. Existing approaches for evaluating knowledge edits either suffer from rapid contamination or rely on counterfactual edits that conflict with rigid existing knowledge. In this work, we propose a synthetic, simulation-driven framework for studying knowledge insertion in LLMs. We introduce {\sc ParallelEvents}, a benchmark of fictional yet realistic future worlds that generates coherent event trajectories for controlled evaluation, avoiding contamination while preserving consistency. Building on this dataset, we develop {\sc Synapse}, a training framework that uses model-generated data to update model parameters via mid-training and instruction tuning. This synthetic pipeline enables scalable knowledge integration without costly human-curated data. Empirically, {\sc Synapse} outperforms existing methods by 14.23\%, demonstr
    
[^110]: 基于非结构化数据自适应结构化的令牌高效数据推理智能体

    Token-Efficient Data Reasoning Agents via Adaptive Structuring of Unstructured Data

    [https://arxiv.org/abs/2608.31082](https://arxiv.org/abs/2608.31082)

    提出“智能体式数据破解”方法，通过在查询到来时自适应地将非结构化数据结构化，使LLM智能体能以远低于现有方法的令牌成本对海量非结构化数据进行复杂问题推理。

    

    有价值的数据仍然嵌入在非结构化来源中：网页、报告、合同、备案文件、财报电话会议记录和PDF文档。企业AI的重大押注在于部署大语言模型（LLM）智能体，对这些数据进行推理，为每一位知识工作者回答复杂问题。智能体如今已经可以做到这一点，但成本却高得令人望而却步。每个问题都需要反复打开大型文档以找回分散在各处的证据，最多会消耗一百万个令牌。然而，如果数据已经结构化，同样的问题将简化为一次廉价的数据库查询。例如，在FanOutQA基准测试中，对一个理想的预结构化数据存储进行推理的成本要低28倍，并且随着问题在更多文档上展开，这一差距会扩大到几个数量级。然而，提前对所有内容进行结构化并不可行：文档所蕴含的可能结构远超任何工作负载实际会使用的范围，而且在查询到来之前，有用的结构和相关文档都是未知的。我们提出了智能体式数据破解（agentic data cracking），一种方法……

    arXiv:2608.31082v1 Announce Type: new  Abstract: Valuable data remains embedded in unstructured sources: web pages, reports, contracts, filings, earnings calls, and PDFs. The big bet in enterprise AI is deploying LLM agents that reason over this data to answer complex questions for every knowledge worker. Agents can do this today, but at prohibitive cost. Each question repeatedly opens large documents to recover scattered evidence, consuming up to a million tokens. However, if the data were already structured, the same question would reduce to a cheap database lookup. For example, on FanOutQA benchmark, reasoning over an ideal pre-structured store is 28X cheaper, and the gap grows to orders of magnitude as questions fan out over more documents. Yet structuring everything in advance is not viable: documents hold vastly more possible structure than any workload will use, and the useful structure and documents are unknown until queries arrive. We propose agentic data cracking, a method th
    
[^111]: 《大语言模型中语言置信度与内部置信度的分歧》

    When Linguistic and Internal Confidence Diverge in Large Language Models

    [https://arxiv.org/abs/2608.28382](https://arxiv.org/abs/2608.28382)

    该研究通过跨8个分类任务、2个生成任务和30个模型的大规模实验，揭示了大语言模型口头表达的语言置信度与其内部置信度经常不一致，且指令微调模型置信度更高但校准更差、态度提示会夸大置信度而无法提升准确性。

    

    用户经常要求大语言模型（LLM）报告其置信程度，但这种语言置信度是否能反映模型的内部置信度尚不清楚。我们在8个分类任务、2个生成任务以及来自三个系列的30个模型上研究了这一问题。对于分类任务，我们从三个维度（关联性、幅值一致性和校准度）将语言置信度与基于logits的置信度进行比较。对于生成任务，我们测试语言置信度是否与基于语义熵的不确定性相符。结果显示这些维度经常出现分歧：实例层面的关联性平均较弱，尽管在较简单的题目和更强的基座模型上有所改善；经过指令微调的模型通常报告更高的置信度，有时表现出更高的关联性，但它们的置信度差距也更大，校准度更差；提示词设计主要改变的是报告置信度的分布；态度线索会夸大置信度却无法提升……（摘要原文在此处截断）

    arXiv:2608.28382v1 Announce Type: new  Abstract: Users often ask large language models (LLMs) to report how confident they are, but it is unclear whether such linguistic confidence tracks the model's internal confidence. We study this question across 8 classification tasks, 2 generation tasks and 30 models from three families. For classification, we compare linguistic confidence with logits-based confidence along three axes: association, magnitude agreement and calibration. For generation, we test whether linguistic confidence tracks semantic-entropy-based uncertainty. The axes frequently diverge. Instance-level association is weak on average, although it improves on easier items and for stronger base models. Instruction-tuned models often report higher confidence and sometimes show higher association, but they also have larger confidence gaps and worse calibration. Prompt design mostly changes the distribution of reported confidence. Attitude cues inflate confidence without improving 
    
[^112]: INSPIRE：一种用于示例驱动数学推理的先内化后改进方法

    INSPIRE: An Internalize-Then-Improve Approach for Example-Driven Mathematical Reasoning

    [https://arxiv.org/abs/2608.27501](https://arxiv.org/abs/2608.27501)

    提出INSPIRE方法，采用先内化参考示例、再逐步改进的策略，增强大语言模型基于示例的数学推理能力（如构造反例检验定理边界），超越仅优化最终答案正确性的传统方法。

    

    数学推理在大语言模型（LLMs）中取得了快速进展，然而现有方法主要针对最终答案的正确性进行优化，这引发了一个问题：模型是真正内化了数学概念，还是仅仅记忆了解题模式。在人类数学教育中，基于示例的推理（如构造反例来检验定理边界）反映了深层的概念理解，但这种能力在当前的大语言模型中仍然发展不足。通过偏好优化来增强这种能力面临两个关键挑战：（1）模型有限的基于示例的推理能力使得构建有效的偏好对本身就十分困难；（2）能力的获得是渐进式的，模型必须先学会采用这种策略，然后才能学会正确地应用它。因此，我们提出了INSPIRE，一种先内化后改进的方法，结合了参考引导的学生内化（RGSI）……

    arXiv:2608.27501v1 Announce Type: new  Abstract: Mathematical reasoning has seen rapid progress in large language models (LLMs), yet existing methods optimize predominantly for final-answer correctness, raising the question whether models truly internalize mathematical concepts or merely memorize solution patterns. In human mathematics education, example-based reasoning such as constructing counterexamples to test theorem boundaries reflects deep conceptual understanding, but remains underdeveloped in current LLMs. Enhancing this capability through preference optimization presents two key challenges: (1) the model's limited example-based reasoning ability makes constructing effective preference pairs inherently difficult; and (2) capability acquisition is progressive, as the model must first learn to adopt this strategy before learning to apply it correctly. Therefore we propose INSPIRE, an Internalize-Then-Improve approach combining Reference-Guided Student Internalization (RGSI), whi
    
[^113]: 语义覆盖层：通过超越令牌和引导向量的注释缓解提示注入

    Semantic Overlays: Mitigating Prompt Injection with Annotations Beyond Tokens and Steering Vectors

    [https://arxiv.org/abs/2608.23873](https://arxiv.org/abs/2608.23873)

    该论文提出了一种名为“语义覆盖层”的新技术，通过向模型输入添加非文本通道来缓解提示注入攻击，利用小型学习的适配器在冻结模型的残差流中创建带外注释，从而增强模型对片段身份的理解。

    

    摘要：arXiv:2608.23873v1 公告类型：新 摘要：语言模型看到的一切都是令牌。服务堆栈知道每个片段是什么——用户输入、工具输出、指令——但模型必须自己跟踪这些，它可能会失去跟踪或被混淆：文本可以被写成看起来像任何东西。提示注入是对这种现象的自然利用。通过扰乱模型对片段身份的理解，攻击者可以诱导不必要的、可能危险的行为。在模型输入中添加一个非文本通道——一种超越文本传达片段身份的方式——缓解了这类攻击。因此，我们引入了一种通用的引导技术，称为语义覆盖层：小型学习的适配器，应用于冻结模型的残差流中的选定预填充位置。在片段上铺设覆盖层创建了一个带外注释通道，该通道无法通过令牌复制。与引导向量不同，语义覆盖层是经过训练的、可适应的，并有选择性地应用。一个覆盖层...

    arXiv:2608.23873v1 Announce Type: new  Abstract: Everything a language model sees is tokens. The serving stack knows what each span is -- user input, tool output, instructions -- but the model must keep track of that itself, and it can lose track or be confused: text can be written to read like anything. Prompt injection is a natural exploit of this phenomenon. By scrambling the model's understanding of span identity, an attacker can induce unwanted and potentially dangerous actions. Adding a non-textual channel to the model's input -- a way to communicate span identity beyond text -- mitigates this class of attack. We thus introduce a general steering technique called Semantic Overlays: small learned adapters applied at chosen prefill positions to a frozen model's residual stream. Laying an overlay over a span creates an out-of-band annotation channel that cannot be replicated by tokens. Unlike steering vectors, Semantic Overlays are trained, adaptable, and selectively applied. An ove
    
[^114]: 别把我框住：面向社会理解的动态文化适应与认知追踪

    Don' t Box Me In: Dynamic Cultural Adaptation and Cognitive Tracking for Social Understanding

    [https://arxiv.org/abs/2608.22411](https://arxiv.org/abs/2608.22411)

    本文提出一种无需训练的框架DyCAC，通过将文化偏好建模为动态混合参考并持续追踪认知，使大型语言模型能灵活适应多元文化社交情境，克服了静态文化建模的局限。

    

    社交互动越来越多地发生在多元文化环境中，个体可能借鉴多种文化影响，并在不同情境下调整其交际行为。尽管近期在赋予大型语言模型（LLMs）社会理解能力方面取得了进展，现有方法往往将文化建模为静态的人口统计属性，限制了它们适应混合且动态表达的交际偏好的能力。因此，在本文中，我们提出\textbf{DyCAC}，一种无需训练的框架，通过结合动态文化适应与持续认知追踪，实现流畅的社会对齐。DyCAC不推断固定的文化身份，而是将文化相关的交际偏好建模为随时间变化的人口级文化参考档案的混合体。这种基于参考的表示进一步被校准。

    arXiv:2608.22411v1 Announce Type: new  Abstract: Social interaction increasingly takes place in multicultural settings, where individuals may draw on multiple cultural influences and adapt their communicative behavior across contexts. Despite recent advances in equipping Large Language Models (LLMs) with social understanding capabilities, existing approaches often model culture as a static demographic attribute, limiting their ability to accommodate hybrid and dynamically expressed communicative preferences. Therefore, in this paper, we propose \textbf{DyCAC}, a training-free framework that achieves fluid social alignment by incorporating \underline{Dy}namic \underline{C}ultural \underline{A}daptation with continuous \underline{C}ognitive tracking. Rather than inferring a fixed cultural identity, DyCAC models culturally relevant communicative preferences as a time-varying mixture of population-level cultural reference profiles. This reference-based representation is further calibrated 
    
[^115]: 编译器引导的自适应证明搜索与跨模型协同在上下文相关定理证明中的应用

    Compiler-Guided Adaptive Proof Search with Cross-Model Synergy on Context-Dependent Theorem Proving

    [https://arxiv.org/abs/2608.18084](https://arxiv.org/abs/2608.18084)

    提出了一种编译器引导的证明搜索框架，通过双模型生成和编译器接地比较，在真实Lean 4项目中实现了更高的证明通过率和更好的效率。

    

    在真实世界的Lean 4项目中，定理证明具有挑战性，因为证明通常依赖于项目特定的上下文。虽然迭代细化可以利用编译器错误来修复失败的证明，但重用失败的尝试需要仔细的搜索控制：某些证明比其他证明提供了更好的起点，而后来的修改可能会降低部分正确证明的质量。我们提出了一种编译器引导的证明搜索框架，该框架平衡了探索和利用。它通过双模型生成和停滞触发的重采样来探索多样化的起点，同时通过基于编译器接地成对比较的当前最佳细化来利用有前景的证明状态。在来自miniCTX-v2的七个真实世界Lean 4项目上的实验表明，我们的方法在有效性-效率权衡方面优于pass@k基线。在pass@32预算内，我们的方法将平均通过率提高了12.8个百分点，同时减少了计算开销。

    arXiv:2608.18084v1 Announce Type: new  Abstract: Theorem proving in real-world Lean 4 projects is challenging because proofs often depend on project-specific context. While iterative refinement can use compiler errors to repair failed proofs, reusing failed attempts requires careful search control: some proofs provide better starting points than others, and later revisions may degrade a partially correct proof. We propose a compiler-guided proof search framework that balances exploration and exploitation. It explores diverse starting points through dual-model generation and stagnation-triggered resampling, while exploiting promising proof states through current-best refinement guided by compiler-grounded pairwise comparison. Experiments on seven real-world Lean 4 projects from miniCTX-v2 show that our method achieves a better effectiveness--efficiency tradeoff than pass@k baselines. Within the pass@32 budget, our method improves average pass rate by 12.8 percentage points while reducin
    
[^116]: CT-$\Delta$Bench：用于纵向三维医学影像差异报告与视觉语言模型的基准

    CT-$\Delta$Bench: A Benchmark for Longitudinal 3D Medical Imaging Difference Reporting with Vision-Language Models

    [https://arxiv.org/abs/2608.11534](https://arxiv.org/abs/2608.11534)

    本文提出CT-$\Delta$Bench基准，专门用于评估视觉语言模型在纵向三维医学影像中生成时间差异报告的能力，并通过患者级划分确保评估可靠性。

    

    在医学影像中，计算机断层扫描（CT）的临床价值不仅在于描绘当前疾病状态，更关键的是能够纵向比较系列扫描以确定疾病演变，这一过程支撑着疗效评估、复发检测和持续的患者管理。然而，尽管时间比较在临床决策中占据核心地位，现有的医学基础模型仍主要局限于单次研究理解，未能充分解决基于时间的交叉检查问题。为填补这一空白，我们研究了纵向影像差异报告任务，即模型接收同一患者的两次时间分离扫描，并生成描述其间间隔变化的临床有意义报告。我们引入了CT-$\Delta$Bench，一个专为此任务设计的基准，采用患者级划分以防止信息泄漏。为了更全面地评估...

    arXiv:2608.11534v1 Announce Type: new  Abstract: In medical imaging, the clinical value of Computed Tomography (CT) lies not only in depicting current disease status, but crucially in enabling longitudinal comparison of serial scans to determine disease evolution, a process that underpins response assessment, recurrence detection, and ongoing patient management. Yet, despite this central role of temporal comparison in clinical decision-making, existing medical foundation models remain largely confined to single-study understanding, leaving temporally grounded cross-examination insufficiently addressed. To address this gap, we study longitudinal imaging difference reporting, a task in which a model takes two temporally separated scans from the same patient and generates a clinically meaningful report describing interval changes between them. We introduce CT-$\Delta$Bench, a dedicated benchmark for this task with patient-level splitting to prevent information leakage. To better evaluate 
    
[^117]: Search-G1：基于表示的内在奖励的落地搜索代理

    Search-G1: Grounded Search Agents via Representation-Based Intrinsic Rewards

    [https://arxiv.org/abs/2608.07531](https://arxiv.org/abs/2608.07531)

    Search-G1 通过基于表示的内在奖励，利用干预校准读数区分必要检索与冗余搜索，实现无需昂贵注释的落地搜索代理。

    

    搜索增强型语言代理应仅在必要时检索外部信息，并将其答案基于检索到的证据。现有的外部奖励要么提供稀疏的结果监督，要么提供来自过程注释和LLM评判器的更丰富反馈。结果奖励易于扩展，但无法区分落地检索与冗余搜索，而更丰富的信号在训练期间需要昂贵的注释或推断。基于策略侧信号（如熵、似然或信息增益）的内在奖励是分级且评估成本低的，但主要反映模型置信度而非证据落地。我们提出Search-G1，一种基于表示的内在奖励框架，通过两个干预校准的读数来衡量代理答案的操作性落地。一个提示状态读数预测闭卷充分性，其补数定义策略相对的检索必要性；

    arXiv:2608.07531v2 Announce Type: replace-cross  Abstract: Search-augmented language agents should retrieve external information only when necessary and ground their answers in retrieved evidence. Existing external rewards provide either sparse outcome supervision or richer feedback from process annotations and LLM judges. Outcome rewards scale readily but cannot distinguish grounded retrieval from redundant search, whereas richer signals require costly annotation or inference during training. Internal rewards based on policy-side signals such as entropy, likelihood, or information gain are graded and inexpensive to evaluate, yet mainly reflect model confidence rather than evidence grounding. We propose Search-G1, a representation-based intrinsic reward framework that measures the operational grounding of an agent's answers through two intervention-calibrated readouts. A prompt-state readout predicts closed-book sufficiency, whose complement defines policy-relative retrieval necessity;
    
[^118]: 潜在事实核查：通过激活工程检测错误信息

    Latent Fact-Checking: Detecting Misinformation through Activation Engineering

    [https://arxiv.org/abs/2608.06417](https://arxiv.org/abs/2608.06417)

    本文提出了一种基于激活工程的错误信息检测框架，利用语言模型表示空间中的几何方向来分类声明真伪，无需微调或外部知识。

    

    arXiv:2608.06417v3 公告类型：交叉替换 摘要：在线错误信息的泛滥推动了对可扩展检测系统的需求。尽管大多数现有方法依赖于表面层面的语言特征或外部知识检索，我们将真实性视为语言模型表示空间中的一种几何属性。我们引入了一个基于激活工程的错误信息检测框架，该框架利用Transformer模型的潜在几何结构。我们的方法通过对比成对真实和虚假陈述的激活，遵循对比激活加法（CAA）的均值差异原理，在残差流中引出错误信息方向。在推理时，将未见声明的最后一个令牌激活投影到该方向上，并将投影表示馈送到多层感知器（MLP）进行分类。该过程无需对骨干模型进行微调，也无需外部证据检索。

    arXiv:2608.06417v3 Announce Type: replace-cross  Abstract: The proliferation of misinformation online has driven demand for scalable detection systems. While most existing approaches rely on surface-level linguistic features or external knowledge retrieval, we examine truthfulness as a geometric property of a language model's representation space. We introduce a misinformation detection framework grounded in activation engineering, which leverages the latent geometry of transformer models. Our approach elicits a misinformation direction in the residual stream by contrasting activations from paired truthful and false statements, following the difference-in-means principle of Contrastive Activation Addition (CAA). At inference time, the last-token activation of an unseen claim is projected onto this direction, and the projected representation is fed to an Multilayer Perceptron (MLP) for classification. The procedure requires no fine-tuning of the backbone model, no external evidence retr
    
[^119]: ConWriter：基于轻量级神经符号一致性控制的、转换约束的状态化长篇故事生成

    ConWriter: Transition-Constrained Stateful Long-Form Story Generation with Lightweight Neuro-Symbolic Consistency Control

    [https://arxiv.org/abs/2608.05169](https://arxiv.org/abs/2608.05169)

    ConWriter是一个无需训练的长篇故事生成框架，通过场景级增量写作、维护演进的故事状态、检查叙事转换约束，并利用不确定性感知的风险信号进行优先验证和局部修复，从而在生成过程中（而非事后）实现一致性控制，避免局部错误向后续场景传播。

    

    长篇故事生成要求模型在扩展的上下文中保持叙事一致性，然而现有的基于提示的方法往往随着故事的增长而累积时间、事实、角色、常识和风格方面的错误。我们提出了ConWriter，一个用于一致性感知长篇故事生成的无需训练的框架。ConWriter在场景级别上增量式地编写故事，由静态故事需求、动态叙事记忆、符号状态推理和不确定性感知的风险信号引导。ConWriter并非将长篇故事生成视为单一的自由形式解码过程，而是维护不断演进的故事状态，检查新场景是否满足所需的叙事转换，并使用不确定性感知的风险信号来优先进行验证和局部修复。这使得一致性控制能够在生成过程中进行，在局部错误传播到后续场景之前。我们在ConStory-Bench上评估ConWriter……

    arXiv:2608.05169v2 Announce Type: replace  Abstract: Long-form story generation requires models to preserve narrative consistency across extended contexts, yet existing prompting-based methods often accumulate temporal, factual, character, commonsense, and stylistic errors as the story grows. We propose ConWriter, a training-free framework for consistency-aware long-form story generation. ConWriter writes stories incrementally at the scene level, guided by static story requirements, dynamic narrative memory, symbolic state reasoning, and uncertainty-aware risk signals. Rather than treating long-story generation as a single free-form decoding process, ConWriter maintains evolving story states, checks whether new scenes satisfy required narrative transitions, and uses uncertainty-aware risk signals to prioritize validation and localized repair. This enables consistency control during generation, before local errors propagate into later scenes. We evaluate ConWriter on ConStory-Bench acro
    
[^120]: 重新审视推测解码中的有损验证：机制、权衡与失效模式

    Revisiting Lossy Verification in Speculative Decoding: Mechanisms, Trade-offs, and Failure Modes

    [https://arxiv.org/abs/2607.26627](https://arxiv.org/abs/2607.26627)

    本文系统分析了推测解码中有损验证方法所诱导的解码分布，将众多表面不同的方法统一为基于截断的验证与协作式验证两类，并构建诊断评估框架揭示其效率与质量之间的权衡及失效模式。

    

    推测解码（Speculative Decoding, SD）通过让轻量级的草稿模型提出 token，再由更大的目标模型并行验证，从而加速大语言模型的推理。近期的方法引入了有损验证方案，通过放松严格的分布匹配来进一步提升效率。然而，这种放松会悄然改变解码分布，由此获得的加速可能以不稳定、有时严重退化的生成质量为代价。在这项工作中，我们对有损验证方法所诱导的分布进行了系统性的分析。我们证明了许多表面上看似不同的方法其实只是 superficial 层面的差异，并可以统一归纳为两大类：基于截断的验证和协作式验证。我们进一步构建了一个覆盖多个精选基准的诊断式评估框架。对于基于截断的方法，我们识别出一个根本性的陷阱——性能可能会……

    arXiv:2607.26627v2 Announce Type: replace  Abstract: Speculative Decoding (SD) accelerates large language model inference by allowing a lightweight draft model to propose tokens that are subsequently verified in parallel by a larger target model. Recent approaches introduce lossy verification schemes to further improve efficiency by relaxing strict distributional matching. Yet such relaxation silently rewrites the decoding distribution, and the resulting acceleration can come at the cost of unstable, sometimes severely degraded generation quality. In this work, we present a principled analysis of the distributions induced by lossy verification methods. We show that many seemingly distinct approaches differ only superficially and can be unified into two categories: truncation-based verification and collaborative verification. We further construct a diagnostic evaluation framework across curated benchmarks. For truncation-based methods, we identify a fundamental pitfall-performance can d
    
[^121]: 并非所有大语言模型的推理都体现在思维链中

    Not All LLM Reasoning is Visible in the Chain-of-Thought

    [https://arxiv.org/abs/2607.22925](https://arxiv.org/abs/2607.22925)

    前沿大语言模型能利用语义无关的填充token进行思维链之外的“不可见推理”来提升任务表现，甚至可以完成思维链监控完全无法察觉的隐藏目标，这对基于CoT监控的AI安全方案构成重大风险。

    

    AI安全的一个关键问题是：语言模型是否会在其输出token中表达其全部推理过程。我们展示了一种具体的失败模式：前沿模型通过利用语义无关的填充token来提升在合成推理任务上的表现，从而展现出“不可见推理”。我们在三个任务上评估了13个前沿语言模型，发现许多模型都能从填充token中显著获益，准确率提升最高可达13个百分点。这种收益取决于所使用的token类型，且在不同模型之间存在差异。我们进一步表明，填充token使Claude Opus 4.5能够在不牺牲其主要任务准确率的前提下满足一个隐藏的模运算约束，这证明了不可见推理可以服务于思维链监控完全无法察觉的目标。强化学习使Qwen3-235B对填充token的内容产生了强烈偏好，但无论是强化学习还是监督微调（原文在此处截断）

    arXiv:2607.22925v2 Announce Type: replace-cross  Abstract: A key question for AI safety is whether a language model expresses all of its reasoning in its output tokens. We demonstrate a concrete failure mode where frontier models exhibit invisible reasoning by leveraging semantically irrelevant filler tokens to improve performance on synthetic reasoning tasks. We evaluate 13 frontier language models across three tasks and find that many models benefit significantly from filler tokens, with accuracy improvements of up to 13 percentage points. The benefit depends on which tokens are used and differs across models. We further show that filler tokens enable Claude Opus 4.5 to satisfy a hidden modular arithmetic constraint without sacrificing accuracy on its primary task, demonstrating that invisible reasoning can serve objectives entirely invisible to CoT monitoring. Reinforcement learning gives Qwen3-235B strong preferences over filler token content, but neither RL nor supervised fine-tun
    
[^122]: 从看似合理到可操作：关于大语言模型自我解释的立场论文

    From Plausible to Actionable: A Position on LLM Self-Explanations

    [https://arxiv.org/abs/2607.15957](https://arxiv.org/abs/2607.15957)

    本立场论文指出大语言模型的自我解释虽高度合理但忠实性存疑，主张评估标准应从合理性与忠实性扩展到可操作性，并为此提供了实用的评估指南。

    

    大语言模型（LLM）能够生成用自然语言对其自身决策进行合理化说明的解释，这种现象通常被称为“自我解释”。此类解释已成为可解释人工智能（XAI）中一个有前景的研究方向，尤其在解释大语言模型行为方面。然而，尽管自我解释往往看起来合理，但它们是否忠实地反映了模型底层的推理过程仍是一个悬而未决的问题。在这篇观点论文中，我们提出：自我解释可以高度合理、忠实性却存疑，但同时又具有高度的可操作性。从传统XAI的视角出发，我们指出了针对LLM生成的自我解释的标准评估协议的局限性，并提出了评估其合理性与忠实性的实用指南。此外，我们主张评估应超越这些标准、扩展到可操作性维度，并重点阐述了LLM合理化解释的应用（摘要此处被截断）。

    arXiv:2607.15957v2 Announce Type: replace  Abstract: Large Language Models (LLMs) can generate natural language explanations that rationalize their own decisions, a phenomenon commonly referred to as self-explanations. Such explanations have emerged as a promising direction for explainable artificial intelligence (XAI), particularly for interpreting LLM behavior. However, while self-explanations often appear plausible, whether they faithfully reflect a model's underlying reasoning process remains an open question. In this opinion paper, we argue that self-explanations can be highly plausible, questionably faithful, and yet highly actionable. From a traditional XAI perspective, we identify the limitations of standard evaluation protocols for LLM-generated self-explanations and propose practical guidelines for assessing their plausibility and faithfulness.Moreover, we argue that evaluation should extend beyond these criteria to actionability, highlighting applications of LLM rationalizat
    
[^123]: EvoCUA-1.5：面向多轮计算机使用智能体的在线强化学习

    EvoCUA-1.5: Online Reinforcement Learning for Multi-turn Computer-Use Agents

    [https://arxiv.org/abs/2607.09773](https://arxiv.org/abs/2607.09773)

    EvoCUA-1.5 将计算机使用智能体从离线经验学习扩展到在线强化学习，并提出步级策略优化（STEPO）方法，以解决多轮交互中上下文管理观察、稀疏终端奖励、可变长度轨迹和慢速环境反馈等挑战。

    

    计算机使用智能体必须通过与部分可观察的多模态桌面环境进行反复交互来解决长时程任务。尽管模仿学习和离线轨迹优化能够提供强大的先验知识，但静态轨迹无法覆盖真实计算机使用中的因果反馈循环：每个动作都会改变屏幕状态、未来的动作空间以及恢复选项。EvoCUA-1.5 将自我进化的计算机使用智能体从离线经验学习扩展到在线强化学习，其中策略与可执行的沙盒环境进行交互，并从可验证的任务结果中获得改进。在这种设置下，在线强化学习并非简单复用单轮语言强化学习的方法即可奏效。多轮交互引入了上下文管理的观察、稀疏的终端奖励、可变长度的轨迹以及缓慢的环境反馈等挑战。EvoCUA-1.5 通过步级策略优化（STEPO）方法来应对这些挑战，该方法保留了……（摘要在此处截断）

    arXiv:2607.09773v2 Announce Type: replace  Abstract: Computer-use agents must solve long-horizon tasks through repeated interaction with partially observable, multimodal desktop environments. Although imitation learning and offline trajectory refinement provide strong priors, static traces cannot cover the causal feedback loop of real computer use: each action changes the screen state, future action space, and recovery options. EvoCUA-1.5 extends self-evolving computer-use agents from offline experience learning to online reinforcement learning, where policies interact with executable sandbox environments and improve from verifiable task outcomes. Online RL in this setting requires more than directly reusing single-turn language-RL recipes. Multi-turn interaction introduces context-managed observations, sparse terminal rewards, variable-length trajectories, and slow environment feedback. EvoCUA-1.5 addresses these challenges with Step-Level Policy Optimization (STEPO), which preserves 
    
[^124]: 从推理中估计不确定性：大语言模型多语言与跨语言多项选择问答性能的大规模研究

    Estimating Uncertainty from Reasoning: A Large-Scale Study of Multi- and Crosslingual MCQA Performance in LLMs

    [https://arxiv.org/abs/2607.06327](https://arxiv.org/abs/2607.06327)

    该研究首次在22种语言上大规模评估了不确定性估计方法，发现让模型用英语进行长篇推理可显著提升低资源语言问答的不确定性估计性能，表明可靠性瓶颈在于生成而非理解。

    

    不确定性估计使基于大语言模型的系统能够识别何时应当弃答，然而现有研究主要集中于英语。我们首次对不确定性估计（UE）方法进行了覆盖22种语言的大规模评估，涵盖高、中、低资源语言。基于两个人工整理的问答数据集，我们比较了开放式与封闭式UE方法（共九种），涵盖不同的模型规模和架构，同时引导模型进行长篇推理，并避免使用LLM-as-a-judge和基于嵌入的评分方式（这些方式可能引入评估噪声）。我们报告了三个主要的可操作发现。第一，我们发现让模型在保持问题为低资源语言的同时用英语进行推理，能够显著提升UE性能，这表明模型对低资源语言的理解能力基本完好，可靠性瓶颈在于生成而非理解。第二，提示模型用英语进行推理接……（摘要原文在此处截断）

    arXiv:2607.06327v3 Announce Type: replace-cross  Abstract: Uncertainty estimation (UE) enables LLM-powered systems to recognize when to abstain, yet existing research has predominantly focused on English. We present the first large-scale evaluation of UE methods across 22 languages, spanning high-, mid-, and low-resource settings. Using two human-curated Q&A datasets, we compare open and closed box UE methods (nine in total) across different model sizes and architectures while eliciting long-form reasoning, avoiding LLM-as-a-judge and embedding-based scoring, which can introduce evaluation noise. We report three main actionable findings. First, we find that prompting models to reason in English while keeping questions in low-resource languages substantially improves UE performance, suggesting that comprehension of low-resource languages is largely intact, and that the reliability bottleneck lies in generation rather than understanding. Second, prompting models to reason in English clos
    
[^125]: 通过双重语义嵌入实现大型语言模型的鲁棒文本水印

    Robust Text Watermarking for Large Language Models via Dual Semantic Embeddings

    [https://arxiv.org/abs/2606.31602](https://arxiv.org/abs/2606.31602)

    提出双重嵌入水印（DEW）方案，结合上下文与词元级嵌入的信号处理技术，为大型语言模型生成对改写和翻译攻击具有最先进鲁棒性的文本水印，同时保持低计算开销和高质量的文本。

    

    本工作提出了双重嵌入水印（Dual-Embedding Watermarking, DEW），这是一种针对大型语言模型（LLMs）的语义水印方案，它利用上下文嵌入和词元级嵌入来增强对改写和翻译攻击的鲁棒性。DEW采用信号处理方法，对词元嵌入和上下文嵌入应用代数向量空间运算，从而导出一种在语义偏移下能够平稳退化的水印信号。该方法通过使用以密钥为种子的伪随机矩阵投影嵌入向量来混淆水印。实验结果表明，双重嵌入水印能够提供最先进的鲁棒性，特别是在抵抗翻译攻击方面，同时与其他语义方案相比具有相对较低的计算开销。在较低的水印强度下，DEW仍能保持有竞争力的文本质量，这表明双重嵌入信号为鲁棒的语义水印提供了一个有前景的基础。

    arXiv:2606.31602v3 Announce Type: replace  Abstract: This work presents Dual-Embedding Watermarking (DEW), a semantic watermarking scheme for large language models (LLMs) that leverages contextual and token-level embeddings to enhance robustness against paraphrasing and translation. DEW utilizes a signal-processing methodology, applying algebraic vector-space operations to token and context embeddings to derive a watermark signal that degrades gracefully under semantic shifts. The method obfuscates the watermark by projecting embedding vectors through pseudo-random matrices seeded with a secret key. Experimental results show that dual-embedding watermarking can offer state-of-the-art robustness, particularly against translation, while incurring relatively low computational overhead compared with other semantic schemes. At lower watermark strength, DEW also maintains competitive text quality, suggesting that dual-embedding signals provide a promising substrate for robust semantic waterm
    
[^126]: GPTNT：基于《保持通话，无人爆炸》的多模态智能体实时协作基准测试

    GPTNT: Benchmarking Real-Time Collaboration Between Multimodal Agents on Keep Talking And Nobody Explodes

    [https://arxiv.org/abs/2606.28514](https://arxiv.org/abs/2606.28514)

    该论文提出了GPTNT基准，首次在《保持通话，无人爆炸》游戏中将时间压力、信息不对称和不完美沟通这三种协作条件结合在一起，评估两个多模态智能体在实时倒计时下通过沟通协作拆弹的能力。

    

    多模态模型正越来越多地被部署用于与人类或其他人工智能体协作完成任务。尽管现有基准测试表明它们具备基本能力，但协作过程中同时出现的各种条件——时间压力、信息不对称以及不完美的沟通——传统上都是被孤立研究的。为填补这一空白，我们提出了GPTNT，一个基于合作视频游戏《保持通话，无人爆炸》（Keep Talking and Nobody Explodes）构建的基准测试。在该基准中，两个智能体必须在实时倒计时的压力下协作拆除程序生成的炸弹谜题。其中一个智能体可以看到炸弹，但没有拆弹说明；另一个智能体持有拆弹说明，却看不到也无法操作炸弹。任何一方都无法独自完成任务：该任务需要双方共同贡献，只有通过有效且高效的沟通才能解决。我们移除了轮次交替的代理机制或各种简化设定，转而要求……

    arXiv:2606.28514v2 Announce Type: replace  Abstract: Multimodal models are increasingly deployed to solve tasks collaboratively with humans or other artificial agents. While existing benchmarks show that they possess the fundamental capabilities, the various conditions that coincide when collaborating---time pressure, information asymmetry, and imperfect communication---have traditionally been studied in isolation. To address this gap, we introduce GPTNT, a benchmark built on the cooperative video game Keep Talking and Nobody Explodes, in which two agents must coordinate to defuse procedurally generated bomb puzzles against a live countdown. One agent has access to the bomb but not the instructions for defusing it; the other holds the instructions but cannot see or manipulate the bomb. Neither agent can succeed alone: the task requires contributions from both, and is solvable only through effective, efficient communication. We remove turn-taking proxies or simplifications, instead requ
    
[^127]: 构造即忠实：面向多文档摘要的声明锚定归因方法

    Faithful by Construction: Claim-Anchored Attribution for Multi-Document Summarization

    [https://arxiv.org/abs/2606.23989](https://arxiv.org/abs/2606.23989)

    提出CAMS框架，将声明级归因嵌入“提取—选择—改写”流程，使多文档摘要中的每句话都能锚定到经过验证、可溯源的源文本片段，从而在构造层面保证摘要的忠实性。

    

    端到端大语言模型（LLM）能够生成流畅的多文档摘要，但仍容易产生幻觉，且其提供的归因通常较为粗糙（仅指向整篇文档或段落）并属于事后生成，导致每条摘要陈述都难以验证。我们重新审视模块化的“提取—选择—改写”范式，并将其中间表示重新构建为归因的基本单元。我们提出了CAMS（Claim-Anchored Multi-document Summarization，声明锚定多文档摘要）框架，该框架：(i) 从每个源文档中提取带有词元级溯源信息的原子声明；(ii) 跨文档聚类等价声明，同时标记源间冲突；(iii) 选择一个兼顾支持度与显著性的子集；(iv) 将所选内容改写为摘要，其中每个句子都锚定到一个经过支持性检验的声明，该声明可回链至一个或多个源文本片段。由于内容在生成之前就已完成定位，整个流程从构造上即是面向归因的。

    arXiv:2606.23989v3 Announce Type: replace-cross  Abstract: End-to-end large language models (LLMs) produce fluent multi-document summaries but remain prone to hallucination, and the attributions they offer are typically coarse (whole documents or passages) and generated post hoc, leaving each summary statement hard to verify. We revisit the modular Extract--Select--Rewrite paradigm and recast its intermediate representation as the unit of attribution. We present CAMS, a Claim-Anchored Multi-document Summarization framework that (i) extracts atomic claims with token-level provenance from every source document, (ii) clusters equivalent claims across documents while flagging inter-source conflicts, (iii) selects a support-aware and salient subset, and (iv) rewrites the selection into a summary in which every sentence is anchored to a support-checked claim that links back to one or more source spans. Because content is localized before it is realized, the pipeline is attribution-oriented b
    
[^128]: CacheWeaver：面向高效有据RAG推理的缓存感知证据排序

    CacheWeaver: Cache-Aware Evidence Ordering for Efficient Grounded RAG Inference

    [https://arxiv.org/abs/2606.19667](https://arxiv.org/abs/2606.19667)

    CacheWeaver是一种轻量级的提示层缓存感知证据排序方法，通过维护前缀树并用贪心算法将重叠证据重排为可复用的前缀，在不改变服务引擎和检索证据集的前提下，将有据RAG推理的中位首token时间（TTFT）降低约20-33%。

    

    检索增强生成（RAG）提升了事实依据性，但同时延长了提示词长度并增加了预填充成本。vLLM等服务引擎中的前缀缓存只有在请求共享相同token前缀时才能降低这一成本。然而在有据生成中，相邻的查询可能以不同顺序检索到重叠的证据，因此证据集合的重叠无法转化为可重用的前缀重叠。我们提出了CacheWeaver，这是一种轻量级的提示层缓存感知证据排序方法。该方法在最近服务过的证据序列上维护一个前缀树，并通过贪心遍历将最可重用的前缀放在最前面，同时保持服务引擎和检索到的证据集合不变。在三种vLLM配置下，相对于按检索顺序排列的前缀缓存，该方法将中位首token生成时间（TTFT）降低了约20-33%，并且在我们的问答测试中不影响答案质量。贪心策略达到了中位（原文在此处截断）

    arXiv:2606.19667v2 Announce Type: replace  Abstract: Retrieval-Augmented Generation (RAG) improves factual grounding, but it also lengthens prompts and raises prefill cost. Prefix caching in serving engines such as vLLM reduces this cost only when requests share the same token prefix. In grounded generation, however, adjacent queries may retrieve overlapping evidence in different orders, so set overlap does not become reusable prefix overlap. We present CacheWeaver, a lightweight prompt-layer method for cache-aware evidence ordering. The method keeps a prefix tree over recently served evidence sequences and uses a greedy walk to place the most reusable prefix first, while leaving the serving engine and retrieved evidence set unchanged. Across three vLLM configurations, the method lowers median time-to-first-token (TTFT) by about 20-33 percent relative to retrieval-order prefix caching, without hurting answer quality in our QA tests. The greedy policy reaches 97.5 percent of the median 
    
[^129]: 检测、重新掩码、修复：面向演化上下文忠实摘要的扩散编辑

    Detect, Remask, Repair: Diffusion Editing for Faithful Summarization of Evolving Contexts

    [https://arxiv.org/abs/2606.12807](https://arxiv.org/abs/2606.12807)

    提出基于掩码扩散语言模型的DETECT-REMASK-REPAIR框架，通过检测、重新掩码和局部修复摘要中的过时片段，在保留已受支持内容的同时实现高效可控的演化上下文摘要更新，并发布了StreamSum基准数据集。

    

    arXiv:2606.12807v2 公告类型：替换 摘要：随着上下文的演化和新信息的到来，真实世界事件的摘要可能会变得过时。一种常见的应对方式是基于更新后的上下文重新生成新摘要，但完全重新生成会丢弃先前的草稿，可能掩盖发生变化的内容，并且在只有少量陈述不再受支持时可能是没有必要的。我们研究局部化的忠实性修复任务：在保留已受支持内容的同时，更新现有摘要中过时的片段。我们提出了DETECT-REMASK-REPAIR，一个基于扩散的框架，利用掩码扩散语言模型来识别、重新掩码并修复过时区域。为了评估演化上下文摘要任务，我们引入了StreamSum，一个由合成事件时间线构成的基准数据集。在DialogSum和StreamSum上的实验表明，局部化扩散修复为完全重写提供了一种可控的替代方案：忠实性引导的修复能够改进早期草稿，单步修复将修复成本降低至半秒以内，并且……

    arXiv:2606.12807v2 Announce Type: replace  Abstract: Summaries of real-world events can become outdated as contexts evolve and new information arrives. A common response is to generate a new summary from the updated context, but full regeneration discards the previous draft, can obscure what changed, and may be unnecessary when only a few claims are unsupported. We study localized faithfulness repair: updating outdated spans in an existing summary while preserving supported content. We propose DETECT-REMASK-REPAIR, a diffusion-based framework that identifies, remasks, and repairs outdated regions with masked diffusion language models. To evaluate evolving-context summarization, we introduce StreamSum, a benchmark of synthetic event timelines. Experiments on DialogSum and StreamSum show that localized diffusion repair provides a controllable alternative to full rewriting: faithfulness-steered repair improves early drafts, one-step repair reduces repair cost to under half a second, with 
    
[^130]: KCSAT-ML：利用全国考生群体的人类难度数据探测推理模型

    KCSAT-ML: Probing Reasoning Models with Nationwide-Cohort Human Difficulty

    [https://arxiv.org/abs/2606.10403](https://arxiv.org/abs/2606.10403)

    该论文提出了KCSAT-ML基准——十年韩国高考数学题共664道、核心339题附带数十万考生的官方逐题错误率，并配合与得分正交的“难度对齐推理增益”（DRG）指标，揭示了模型在人类认为困难的题目上准确率崩溃、测试时扩展的准确率收益呈非单调曲线等关键发现。

    

    数学推理基准测试大量涌现，但大多数缺乏基于真实人类表现的逐题难度信号。我们推出KCSAT-ML，涵盖十年（2014-2025）韩国大学修学能力考试（KCSAT，即“修能”）数学题：共664道题目，其中包含一个339题的核心集合，附带来自数十万考生全国群体数据的官方逐题错误率。我们为该基准配备了难度对齐推理增益（DRG）：一个与得分正交的指标，用于考察模型的错误是集中在人类认为困难的题目上，还是集中在人类认为简单的题目上。结合二者，我们在广泛的视觉语言模型（VLM，以及使用OCR的大语言模型）中揭示了三种模式：(i) 在各种模型规模下，低预算的准确率在高人类错误率区段急剧崩溃；(ii) 测试时扩展（TTS）的token使用量与考生群体错误率大致呈线性增长，而准确率提升呈非单调曲线；(iii) 在单一模型家族内，TTS在……（原文在此截断）

    arXiv:2606.10403v3 Announce Type: replace  Abstract: Math reasoning benchmarks have proliferated, yet most lack a per-item difficulty signal grounded in actual human performance. We introduce KCSAT-ML, a decade (2014-2025) of Korean College Scholastic Ability Test (KCSAT; Suneung) mathematics: 664 problems with a 339-item core set carrying official per-item error rates from nationwide cohorts of hundreds of thousands of examinees. We pair the benchmark with Difficulty-aligned Reasoning Gain (DRG): a score-orthogonal metric that asks whether a model's mistakes concentrate on the items humans found hard, or on items humans found easy. Together they expose, across a wide range of VLMs (and LLMs with OCR), three patterns: (i) low-budget accuracy collapses on the high-human-error tail at every model size; (ii) test-time scaling (TTS) raises token use roughly linearly with cohort error rate, while accuracy gains follow a non-monotonic curve; (iii) within a single family, TTS flips between an
    
[^131]: 从架构到输出：大语言模型幻觉的结构性起源与数据的放大作用

    From Architecture to Output: Structural Origins of Hallucination in Large Language Models and the Amplifying Role of Data

    [https://arxiv.org/abs/2606.07537](https://arxiv.org/abs/2606.07537)

    该论文提出了一个仅需采样访问权限的幻觉归因框架，通过针对前缀、上下文和频率竞争的三次有序干预，将大语言模型的单个幻觉追溯归因到自注意力联想检索、最大似然预训练目标或暴露偏差下的自回归承诺等具体组件，并提出五个可证伪的预测。

    

    大语言模型会生成流畅、自信但事实上错误的输出。现有的分类体系仅按输出类型对这些失败进行分类——内在性还是外在性、忠实性还是事实性——却从未说明是哪个计算组件产生了特定的失败。我们探讨需要什么条件才能将单个幻觉归因于纯解码器（decoder-only）架构栈中的特定组件。我们将三个组件——自注意力的联想检索、最大似然预训练目标、以及暴露偏差（exposure bias）下的自回归承诺——视为候选失败面，论证而非假定它们的可分离性，并规定了一种仅需采样访问权限的归因程序：一组针对前缀、上下文和频率竞争的有序三步干预，同时辅以基于独立标注和分类器基线的验证设计。我们提出了五个可证伪的预测……

    arXiv:2606.07537v2 Announce Type: replace-cross  Abstract: Large language models produce fluent, confident, factually wrong output. Existing taxonomies classify these failures by output type -- intrinsic versus extrinsic, faithfulness versus factuality -- but say nothing about which computational component produced a given failure. We ask what would be required to attribute an individual hallucination to a specific component of the decoder-only stack. We treat three components -- self-attention's associative retrieval, the maximum-likelihood pretraining objective, and autoregressive commitment under exposure bias -- as candidate failure surfaces, justify their separability rather than assuming it, and specify an attribution procedure requiring only sampling access: an ordered set of three interventions on prefix, context, and frequency competition, together with a validation design based on independent annotation and a classifier baseline. We state five falsifiable predictions and iden
    
[^132]: MineExplorer：评估MLLM智能体在《我的世界》中的开放世界探索能力

    MineExplorer: Evaluating Open-World Exploration of MLLM Agents in Minecraft

    [https://arxiv.org/abs/2605.30931](https://arxiv.org/abs/2605.30931)

    本文提出了MineExplorer基准测试，通过筛选通用原子任务、组合隐式多跳任务及多智能体合成流程，系统评估MLLM智能体在《我的世界》中的开放世界探索能力。

    

    多模态大语言模型（MLLMs）在感知、推理和动作生成方面展现出强大能力。然而，它们在动态开放世界中维持持续探索的能力仍不明确。现有的具身和基于游戏的基准测试往往将交互压缩为短时任务，或将成功与特定领域的游戏机制纠缠在一起。在本文中，我们引入了MineExplorer基准测试，用于评估MLLM智能体在《我的世界》中的开放世界探索能力。我们首先筛选出那些解决方案主要依赖《我的世界》特定知识的原子任务，以更好地反映通用的开放世界推理。然后，我们围绕ReAct风格的能力框架组织基准测试，并将原子任务组合成隐式多跳任务。为进一步构建可靠的实例，MineExplorer采用多智能体合成工作流，联合设计任务图、沙盒场景和基于规则的里程碑评估器。人类评估...

    arXiv:2605.30931v3 Announce Type: replace  Abstract: Multimodal large language models (MLLMs) have shown strong capabilities in perception, reasoning, and action generation. However, their ability to sustain exploration in dynamic open worlds remains unclear. Existing embodied and game-based benchmarks often compress interaction into short-horizon tasks or entangle success with domain-specific game mechanics. In this paper, we introduce MineExplorer benchmark for evaluating open-world exploration capabilities of MLLM agents in Minecraft. We first filter atomic tasks whose solutions rely heavily on Minecraft-specific knowledge to better reflect general open-world reasoning. Then we organize the benchmark around a ReAct-style capability formulation and compose atomic tasks into implicit multi-hop tasks. To further construct reliable instances, MineExplorer uses a multi-agent synthesis workflow that jointly designs task graphs, sandbox scenes, and rule-based milestone evaluators. Human ev
    
[^133]: 基于潜在推理的鲁棒高效安全护栏

    Robust and Efficient Guardrails with Latent Reasoning

    [https://arxiv.org/abs/2605.29068](https://arxiv.org/abs/2605.29068)

    COLAGUARD通过分阶段训练将多步安全推理压缩进连续潜在空间，推理时直接传播隐状态，在宏F1上超越Llama Guard 3达8.24分，并在达到显式推理基线GuardReasoner同等性能的同时实现12.9倍加速和22.4倍的token开销降低。

    

    随着大语言模型（LLM）越来越多地部署于现实世界的应用中，保障其安全性至关重要。现有的安全护栏通常依赖于单次分类，或近期出现的蒸馏推理方法。基于推理的护栏在性能上显著优于仅分类的基线方法，但其会带来较高的查询延迟和token开销，使其难以适用于高吞吐量部署场景。为应对这一挑战，我们提出了COLAGUARD，这是一种安全护栏模型，通过分阶段训练课程将多步安全推理迁移至连续潜在空间中，从而在推理时实现直接的隐状态传播。在涵盖八个安全基准的十个提示与响应审核设置上进行的评估中，COLAGUARD的宏F1分数比Llama Guard 3提高了8.24个点，并在宏F1上与我们的显式推理基线GuardReasoner持平，同时实现了12.9倍的加速和22.4倍的（token开销降低）。（注：原文摘要在此处截断）

    arXiv:2605.29068v2 Announce Type: replace  Abstract: Maintaining the safety of large language models (LLMs) is crucial as they are increasingly deployed in real-world applications. Existing safety guardrails typically rely on single-pass classification or, more recently, distilled reasoning. Reasoning-based guardrails significantly outperform classification-only baselines, but they incur substantial query latency and token overhead that make them impractical for highthroughput deployment. To address this challenge, we propose COLAGUARD, a guardrail model that transfers multi-step safety reasoning into a continuous latent space through a stage-wise training curriculum, enabling direct hidden-state propagation at inference. Evaluated on ten prompt- and response-moderation settings spanning eight safety benchmarks, COLAGUARD improves macro-F1 by 8.24 points over Llama Guard 3 and matches our explicit reasoning baseline, GuardReasoner, in macroF1 while delivering a 12.9X speedup and 22.4X 
    
[^134]: 面向自回归多维度作文评分的特质感知策略优化

    Trait-Aware Policy Optimization for Autoregressive Multi-Trait Essay Scoring

    [https://arxiv.org/abs/2605.25731](https://arxiv.org/abs/2605.25731)

    提出特质感知策略优化（TAPO）后训练框架，通过在样本和特质两个维度分解奖励并结合包含原始提示与特质描述的增强提示，在多个骨干模型上持续优于监督微调和标量奖励优化基线，显著提升自回归模型的多维度作文评分性能。

    

    多维度作文评分旨在对写作质量的多个维度进行细粒度评估。然而，如何有效地对自回归评分模型进行后训练仍然研究不足。在本文中，我们提出了特质感知策略优化，这是一个专为自回归多维度评分量身定制的后训练框架。我们的方法沿样本和特质两个维度分解奖励，结合了全局评分一致性、特质级准确性、格式有效性以及特质间依赖关系的保持。此外，我们在整个训练过程中使用增强提示，通过纳入原始提示文本和特质描述，为特质特定的分数生成提供更丰富的语义信息。在多个骨干模型上的实验表明，与监督微调和标量奖励优化基线相比，我们的方法持续提升了多维度评分性能，证明了该方法的有效性。

    arXiv:2605.25731v3 Announce Type: replace  Abstract: Multi-trait essay scoring aims to provide fine-grained evaluation of writing quality across multiple dimensions. However, how to effectively post-train autoregressive scoring models remains underexplored. In this paper, we propose Trait-Aware Policy Optimization (TAPO), a post-training framework tailored to autoregressive multi-trait scoring. Our method decomposes rewards along both the sample and trait dimensions, combining global scoring consistency, trait-level accuracy, format validity, and inter-trait dependency preservation. In addition, we use enhanced prompts throughout training by incorporating original prompt texts and trait descriptions, providing richer semantic information for trait-specific score generation. Experiments across multiple backbone models show that our method consistently improves multi-trait scoring performance over supervised fine-tuning and scalar-reward optimization baselines, demonstrating the effectiv
    
[^135]: 通过自动分段与块蒸馏实现块注意力的泛化

    Towards Generalization of Block Attention via Automatic Segmentation and Block Distillation

    [https://arxiv.org/abs/2605.15913](https://arxiv.org/abs/2605.15913)

    该论文构建了包含3万余实例的语义分割数据集SemanticSeg并训练轻量级自动分割器，同时提出块蒸馏训练框架，解决了块注意力中文本难以有效分块及微调低效易损性能的问题，推动块注意力在RAG等长上下文场景中的泛化应用。

    

    块注意力将输入处理为彼此无法相互关注的独立块，在检索增强生成（RAG）等长上下文场景中，为提升KV缓存复用提供了巨大潜力。然而，其更广泛的应用受到两个关键挑战的阻碍：一是难以将输入文本分割成有意义的、自包含的块；二是现有的块微调方法效率低下，且有性能下降的风险。为解决这些问题，我们首先构建了SemanticSeg，这是一个大型且多样化的语义分割数据集，包含16个类别（涵盖书籍、代码、网页文本和对话等）超过3万个实例，文本长度从2k到32k不等。利用该数据集，我们训练了一个轻量级分割器，能够自动将文本划分为符合人类直觉的块，且分割粒度可控。其次，我们提出了块蒸馏，这是一种更高效的训练框架。

    arXiv:2605.15913v5 Announce Type: replace-cross  Abstract: Block attention, which processes the input as separate blocks that cannot attend to one another, offers significant potential to improve KV cache reuse in long-context scenarios such as Retrieval-Augmented Generation (RAG). However, its broader application is hindered by two key challenges: the difficulty of segmenting input text into meaningful, self-contained blocks, and the inefficiency of existing block fine-tuning methods that risk degrading performance. To address these, we first construct SemanticSeg, a large and diverse semantic segmentation dataset containing over 30k instances across 16 categories-including books, code, web text, and conversations with text lengths ranging from 2k to 32k. Using this dataset, we train a lightweight segmenter to automatically partition text into human-instinct-aligned blocks with controllable granularity. Second, we propose block distillation, a training framework that is more efficient
    
[^136]: 科学领域知识改进视觉-语言眼底模型

    Scientific Domain Knowledge Improves Vision-Language Fundus Models

    [https://arxiv.org/abs/2605.02720](https://arxiv.org/abs/2605.02720)

    该研究构建了高领域密度的PubMed-Ophtha数据集，并通过严格对照实验证明，使用领域特定科学文献训练的视觉-语言眼底模型在110项临床任务中的表现优于使用固定模板或医学报告训练的模型。

    

    视觉-语言模型在眼科领域具有相当大的前景，但目前尚不清楚哪种训练数据源最能传达专家领域知识。现有的眼科模型是在固定文本模板、医学报告或通用生物医学文献上训练的，这些数据源从未在相同条件下进行过比较。为了将领域特定文献纳入这一比较，我们提出了PubMed-Ophtha，这是一个具有高领域密度的分层数据集，包含来自PubMed Central的15,842篇开放获取文章中的102,023个图版及其子标题。随后，我们在每种数据源上微调相同的CLIP模型，并以通用生物医学文献模型作为基线，发现领域特定文献在110项临床任务中取得了最佳平均性能，平均线性探测AUROC达到88.63%，领先于医学报告（85.68%）。将数据集限制为眼底图像，并控制图像数量与医学报告一致……

    arXiv:2605.02720v2 Announce Type: replace-cross  Abstract: Vision-language models hold considerable promise for ophthalmology, but it remains unclear which training data source best conveys expert domain knowledge. Existing ophthalmic models are trained on fixed text templates, medical reports, or general biomedical literature, sources that have never been compared under matched conditions. To include domain-specific literature in this comparison, we present PubMed-Ophtha, a hierarchical dataset with high domain density of 102,023 panels with their subcaptions from 15,842 open-access articles in PubMed Central. We then finetuned identical CLIP models on each source, using a general biomedical literature model as baseline, and found that domain-specific literature achieved the best average performance across 110 clinical tasks, reaching a mean linear probing AUROC of 88.63% ahead of medical reports (85.68%). Restricting the dataset to fundus images, to the image count of the medical rep
    
[^137]: 人机共生中跨越增强与自动化的角色感知人工智能

    Role-Aware Artificial Intelligence Across Augmentation and Automation in Human-Machine Symbiosis

    [https://arxiv.org/abs/2605.00440](https://arxiv.org/abs/2605.00440)

    该论文提出在人机共生中应关注AI的角色感知问题，指出AI在“自动化替代”与“能力增强”之间的功能角色虽在提示词中被规定，但脱离对话上下文后便难以追溯。

    

    人工智能（AI）的演进使得人类与计算机器之间的界限日益模糊。在人机共生中内部关系愈加交织的背景下，“AI生成信息”这一概念本身变得难以定义，因为这类信息并非单独源自人类或机器，而是源于二者的相互塑造。有时AI代替人类行动，实现任务的自动化；有时它则扩展人类所能做的事情，增强人类的能力。因此，一个更为切题的问题不仅在于AI是否参与了，而在于它是如何参与的。一般而言，AI所承担的角色通常会在输入提示词中以明示或暗示的方式被规定，然而当只有生成内容可供查看时，这一角色便变得不那么明显甚至完全不可观察。一旦脱离对话上下文，其功能角色可能不再可被追溯。

    arXiv:2605.00440v2 Announce Type: replace  Abstract: The evolution of artificial intelligence (AI) has rendered the boundary between humanity and computational machinery increasingly ambiguous. In the presence of more interwoven relationships within human-machine symbiosis, the very notion of AI-generated information becomes difficult to define, as such information arises not from either humans or machines in isolation, but from their mutual shaping. At times AI acts in place of the human, automating the task; at others it extends what the human can do, augmenting their capability. Therefore, a more pertinent question lies not merely in whether AI has participated, but in how it has participated. In general, the role assumed by AI is often specified, either implicitly or explicitly, in the input prompt, yet becomes less apparent or altogether unobservable when the generated content alone is available. Once detached from the dialogue context, the functional role may no longer be traceab
    
[^138]: 开放推理语言模型的统一部署感知评估

    Unified Deployment-Aware Evaluation of Open Reasoning Language Models

    [https://arxiv.org/abs/2604.07035](https://arxiv.org/abs/2604.07035)

    该论文对七种开放推理语言模型在四个基准上进行了统一且面向实际部署的全面评估，不仅比较准确率还涵盖延迟、显存占用、提示敏感性等指标，发现Gemma-4-26B-A4B配合零样本提示取得最高加权分数0.794。

    

    开放推理语言模型通常在混合样本量、部分标准化提示词和以准确率为中心的总结下进行比较，这使得实际模型选择的结果难以解读。我们对七种开放推理语言模型配置在四个基准测试上进行了统一评估：ARC-Challenge、GSM8K、MATH 1至3级以及TruthfulQA MC1。我们在相同的238个样本子集上，对每一种模型-数据集-策略条件测试了零样本、思维链和少样本思维链提示方法，形成了一个完整的7×4×3实验设计，共包含84个条件和19,992个被评估的样本。除准确率之外，我们还报告了Wilson置信区间、延迟、峰值显存（VRAM）占用、加权综合性能、帕累托有效运行点、提示敏感性指标以及兼容性诊断结果。其中，Gemma-4-26B-A4B在使用零样本提示时取得了最高的加权分数0.794。Gemma-4-E4B则依然保持……

    arXiv:2604.07035v3 Announce Type: replace  Abstract: Open reasoning language models are often compared under mixed sample sizes, partially standardized prompts, and accuracy-centered summaries, which makes practical model selection difficult to interpret. We present a unified evaluation of seven open reasoning language model configurations across four benchmarks: ARC-Challenge, GSM8K, MATH levels 1 to 3, and TruthfulQA MC1. We test zero-shot, chain-of-thought (CoT), and few-shot CoT prompting on the same 238-example subset for every model--dataset--strategy condition, yielding a complete 7 x 4 x 3 design with 84 conditions and 19,992 evaluated examples. Beyond accuracy, we report Wilson confidence intervals, latency, peak video random access memory (VRAM), weighted aggregate performance, Pareto-efficient operating points, prompt-sensitivity metrics, and compatibility diagnostics. Gemma-4-26B-A4B with zero-shot prompting achieves the highest weighted score at 0.794. Gemma-4-E4B remains 
    
[^139]: 面向句子级与上下文感知机器翻译的交叉偏好学习

    Cross-Preference Learning for Sentence-Level and Context-Aware Machine Translation

    [https://arxiv.org/abs/2603.25183](https://arxiv.org/abs/2603.25183)

    提出交叉偏好学习方法，通过在偏好优化目标中整合条件内偏好与跨条件偏好，显式捕捉句子级与上下文感知机器翻译的互补优势，使模型既能有效利用有信息量的上下文，又能对无信息量的上下文保持稳健。

    

    上下文感知机器翻译（MT）利用文档级信息，但其并不总是稳定地优于句子级机器翻译，因为上下文信号在不同句子中带来的益处并不均衡。现有的训练目标没有显式地对这种差异性进行建模，限制了模型自适应利用上下文的能力。在本文中，我们提出了交叉偏好学习，这是一种基于偏好的训练框架，能够显式地捕捉句子级机器翻译与上下文感知机器翻译之间的互补优势。CPL通过将条件内偏好和跨条件偏好整合到偏好优化目标中来实现这一目标，为利用有信息量的上下文提供显式监督，同时对无信息量的上下文保持稳健性。我们使用多个模型（包括Qwen3-4B、Qwen3-8B和Llama-3-8B-Instruct）在多个公开的上下文感知机器翻译任务上验证了所提出的方法。实验结果表明了一致的提升（摘要在此处截断）。

    arXiv:2603.25183v2 Announce Type: replace  Abstract: Context-aware machine translation (MT) leverages document-level information, yet it does not consistently outperform sentence-level MT, as contextual signals are unevenly beneficial across sentences. Existing training objectives do not explicitly model this variability, limiting a model's ability to adaptively exploit context. In this paper, we propose Cross-Preference Learning (CPL), a preference-based training framework that explicitly captures the complementary benefits of sentence-level and context-aware MT. CPL achieves this by integrating both intra- and cross-condition preferences into the preference optimization objective, providing explicit supervision to exploit informative context while remaining robust to uninformative context. We validate the proposed approach on several public context-aware MT tasks using multiple models, including Qwen3-4B, Qwen3-8B, and Llama-3-8B-Instruct. Experimental results demonstrate consistent 
    
[^140]: 基于Kolmogorov-Arnold网络与视觉-语言基础模型的YOLO：面向计算机视觉感知中可解释目标检测的可信多模态AI

    YOLO with Kolmogorov-Arnold networks and vision-language foundation models for interpretable object detection with trustworthy multimodal AI in computer vision perception

    [https://arxiv.org/abs/2603.23037](https://arxiv.org/abs/2603.23037)

    该论文提出用Kolmogorov-Arnold网络作为可解释的事后代理模型，基于七个几何与语义特征评估YOLOv10检测结果的置信度可信性，并结合BLIP视觉-语言基础模型生成描述，实现计算机视觉感知中透明、可信的目标检测。

    

    本文研究了一种新型Kolmogorov-Arnold网络框架的可信目标检测能力。该方法解决了车辆检测感知乃至更广泛计算机视觉领域的一个关键局限：这些系统在视觉退化或模糊场景下，其置信度分数的可靠性缺乏透明度。为此，本文采用Kolmogorov-Arnold网络作为可解释的事后代理模型，利用七个几何与语义特征对YOLOv10检测的可信度进行建模。Kolmogorov-Arnold网络的加性样条结构使得每个特征的影响可以直接可视化，产生平滑且透明的函数映射，从而揭示模型的置信度何时得到充分支持、何时不可靠。此外，引导式语言-图像预训练（BLIP）基础模型为每个检测结果生成描述性文本说明……

    arXiv:2603.23037v2 Announce Type: replace-cross  Abstract: The trustworthy object detection capabilities of a novel Kolmogorov-Arnold network framework are examined here. The approach addresses a key limitation in computer vision for vehicle detection perception, and beyond. These systems offer limited transparency regarding the reliability of their confidence scores in visually degraded or ambiguous scenes. To this end, a Kolmogorov-Arnold network is employed as an interpretable post-hoc surrogate to model the trustworthiness of the You Only Look Once (Yolov10) detections using seven geometric and semantic features. The additive spline-based structure of the Kolmogorov-Arnold network enables direct visualisation of each feature's influence. This produces smooth and transparent functional mappings that reveal when the model's confidence is well supported and when it is unreliable. Furthermore, a bootstrapped language-image (BLIP) foundation model generates descriptive captions of each 
    
[^141]: PROMPT2BOX：通过揭示提示词间的蕴含结构改进大语言模型弱点发现与具体性估计

    PROMPT2BOX:Improving LLM Weakness Discovery and Specificity Estimation by Uncovering Entailment Structure among Prompts

    [https://arxiv.org/abs/2603.21438](https://arxiv.org/abs/2603.21438)

    该论文提出Prompt2Box方法，将提示词嵌入到盒嵌入空间中，以同时捕捉语义相似性和提示词之间的具体性（蕴含）关系，从而实现对大语言模型弱点的细粒度发现与具体性估计。

    

    为了发现大语言模型（LLM）的弱点，研究者们通常将提示词嵌入到向量空间中并进行聚类，以提取有价值的模式。然而，向量嵌入主要捕捉的是主题层面的相似性；因此，共享同一主题但在具体性（进而难度）上不同的提示词往往被表示得相似，这使得细粒度的弱点分析变得困难。为了解决这一局限，我们提出了Prompt2Box，它使用一个经过训练的编码器将提示词嵌入到盒嵌入（box embedding）空间中。该编码器在现有数据集和合成数据集上训练，输出的盒嵌入不仅能捕捉语义相似性，还能捕捉提示词之间的具体性关系（例如，“写一个冒险故事”比“写一个故事”更具体）。我们进一步开发了一种新颖的盒嵌入降维技术，以便于数据集的可视化和比较。我们的实验表明，盒嵌入始终能够……

    arXiv:2603.21438v3 Announce Type: replace  Abstract: To discover the weaknesses of LLMs, researchers often embed prompts into a vector space and cluster them to extract insightful patterns. However, vector embeddings primarily capture topical similarity; as a result, prompts that share a topic but differ in specificity, and consequently in difficulty, are often represented similarly, making fine-grained weakness analysis difficult. To address this limitation, we propose Prompt2Box, which embeds prompts into a box embedding space using a trained encoder. The encoder, trained on existing and synthesized datasets, outputs box embeddings that capture not only semantic similarity but also specificity relations between prompts (e.g., "writing an adventure story" is more specific than "writing a story"). We further develop a novel dimension reduction technique for box embeddings to facilitate dataset visualization and comparison. Our experiments demonstrate that box embeddings consistently ca
    
[^142]: NOTAI.AI：基于曲率与特征归因的可解释机器生成文本检测

    NOTAI.AI: Explainable Detection of Machine-Generated Text via Curvature and Feature Attribution

    [https://arxiv.org/abs/2603.05617](https://arxiv.org/abs/2603.05617)

    该论文提出了NotAI.AI，一个将句子级条件概率曲率、神经检测器分数与可解释文体特征结合于XGBoost元分类器的可解释AI生成文本检测系统，通过TreeSHAP特征归因和自然语言解释揭示预测依据，在RAID数据集的类别平衡子集上达到0.9685的F1分数。

    

    我们提出了NotAI.AI，一个可解释的AI生成文本检测系统。该系统不再仅返回二分类标签或置信度分数，而是展示哪些信号影响了预测结果，并允许用户检查通过减去选定局部贡献所获得的基于归因的敏感性估计。NotAI.AI在一个XGBoost元分类器中结合了句子级条件概率曲率、神经检测器分数，以及可解释的文体特征和可读性特征。系统利用TreeSHAP特征贡献来解释预测，并能将所得证据转化为简洁的自然语言解释。我们在RAID数据集的一个类别平衡子集上评估该系统，该子集包含人类撰写文本、干净的AI生成文本以及受攻击的AI生成文本。完整模型优于基于单一特征族的变体，在留出的子集内测试划分上达到了0.9685的F1分数。在自动评估中，两个模型判断……

    arXiv:2603.05617v2 Announce Type: replace  Abstract: We present NotAI.AI, an explainable AI-generated text detection system. Instead of returning only a binary label or confidence score, the system shows which signals influenced the prediction and lets users inspect an attribution-based sensitivity estimate obtained by subtracting selected local contributions. NotAI.AI combines sentence-level conditional probability curvature, a neural detector score, and interpretable stylometric and readability features in an XGBoost meta-classifier. It explains predictions with TreeSHAP feature contributions and can turn the resulting evidence into a concise natural-language explanation. We evaluate the system on a category-balanced subset of RAID containing human-written, clean AI-generated, and attacked AI-generated texts. The full model outperforms variants based on individual feature families, reaching 0.9685 F1 on the held-out within-subset test split. In an automatic evaluation, two model judg
    
[^143]: 标签具有人类价值观：主观任务的价值校准

    Labels have Human Values: Value Calibration of Subjective Tasks

    [https://arxiv.org/abs/2601.06631](https://arxiv.org/abs/2601.06631)

    提出MC-STL框架，通过将标注聚类为可识别的人类价值集群并学习集群特定嵌入来校准预测，在多种主观任务上持续优于忽略标注潜在价值结构的基线方法。

    

    构建用于主观任务的自然语言处理系统需要确保其与相互冲突的人类价值观保持一致。我们提出了多校准主观任务学习器框架（MC-STL），该框架通过三种方法（标注者理由的相似性、专家价值分类体系或评分者的社会文化描述符）将标注聚类为可识别的人类价值集群，并通过学习特定于集群的嵌入来校准每个价值集群的预测。我们在多种主观学习设置中展示了MC-STL，包括序数预测、二分类预测和偏好学习预测，并在涵盖有毒聊天机器人对话、攻击性社交媒体帖子和人类偏好对齐的多个数据集上对其进行了评估。结果表明，MC-STL始终优于忽略标注潜在价值结构的基线方法，在判别能力、特定价值校准和感知分歧的指标方面均有提升。

    arXiv:2601.06631v2 Announce Type: replace  Abstract: Building NLP systems for subjective tasks requires one to ensure their alignment to contrasting human values. We propose the MultiCalibrated Subjective Task Learner framework (MC-STL), which clusters annotations into identifiable human value clusters by three approaches (similarity of annotator rationales, expert-value taxonomies or rater's sociocultural descriptors) and calibrates predictions for each value cluster by learning cluster-specific embeddings. We demonstrate MC-STL on several subjective learning settings, including ordinal, binary, and preference learning predictions, and evaluate it on multiple datasets covering toxic chatbot conversations, offensive social media posts, and human preference alignment. The results show that MC-STL consistently outperforms the baselines that ignore the latent value structure of the annotations, delivering gains in discrimination, value-specific calibration, and disagreement-aware metrics.
    
[^144]: TeleTables：面向电信表格解读的大语言模型基准测试

    TeleTables: A Benchmark for Large Language Models in Telecom Table Interpretation

    [https://arxiv.org/abs/2601.04202](https://arxiv.org/abs/2601.04202)

    该论文提出TeleTables基准（包含2,220张3GPP规范表格和500道人工验证选择题），通过评估20个开源大语言模型揭示了电信表格解读的两大瓶颈：闭卷时领域知识不足导致准确率不超过41%，而提供表格上下文时准确率虽可超90%，但会随推理深度、证据范围和表格结构复杂性增加而系统性下降。

    

    大语言模型（LLM）越来越多地被应用于电信工程任务，但在3GPP规范上的表现较差。这些标准将大量技术信息编码在复杂的表格中，而大语言模型对此类表格的知识与解读能力在很大程度上仍未被探索。我们提出了TeleTables，该基准包含来自13份3GPP规范、四种格式的2,220张表格，以及500道经人工验证、涵盖从直接检索到多步推理的选择题。我们对20个开源权重大语言模型（涵盖非推理、多模态、推理和表格专用架构）的评估揭示了两个显著的性能瓶颈：在闭卷设置下，领域知识是主要制约因素，没有任何通用模型的准确率超过41%；当表格作为上下文提供时，最优模型的准确率超过90%，但性能会随着推理深度、证据范围和结构复杂性的增加而系统性下降，最大差距达32.2个百分点。

    arXiv:2601.04202v2 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) are increasingly applied to telecom engineering tasks, yet perform poorly on 3GPP specifications. These standards encode much of their technical information in complex tables, but LLM knowledge and interpretation of such tables remain largely unexplored. We introduce TeleTables, a benchmark comprising 2,220 tables from 13 3GPP specifications in four formats and 500 human-verified MCQs spanning direct retrieval to multi-step reasoning. Evaluating 20 open-weight LLMs across non reasoning, multimodal, reasoning, and table specialized architectures reveals two distinct performance bottlenecks. In the closed-book setting, domain knowledge is the primary constraint, with no general-purpose model exceeding 41% accuracy. When the table is provided as context, the best models exceed 90%, but performance degrades systematically with reasoning depth, evidence scope, and structural complexity, with a 32.2pp spr
    
[^145]: 仿生人会梦见看不见的幕后操纵者吗？探究大语言模型中的阴谋论倾向

    Do Androids Dream of Unseen Puppeteers? Probing for a Conspiracy Tendencies in Large Language Models

    [https://arxiv.org/abs/2511.03699](https://arxiv.org/abs/2511.03699)

    本研究通过标准化心理测量调查发现，大型语言模型表现出一定的阴谋论倾向，且可通过提示策略被引导采纳阴谋论观点。

    

    我们研究了大型语言模型（LLM）是否表现出阴谋论倾向、它们在该领域是否表现出社会人口统计学偏差，以及它们被诱导采纳阴谋论观点的难易程度。阴谋论信念在错误信息传播以及对机构不信任的形成中扮演着核心角色，这使其成为评估LLM社会与心理保真度及其复制或强化有害叙事潜力的重要测试平台。尽管LLM常被用作研究人类行为的替代工具，但它们是否能再现广义阴谋论信念等高阶心理结构仍不清楚。为弥补这一研究空白，我们在不同的提示与调节策略下，对多个模型进行了测量阴谋论思维的经过验证的心理测量调查。我们的研究结果显示，LLM与其中部分要素表现出一定的一致性。

    arXiv:2511.03699v2 Announce Type: replace  Abstract: We investigate whether Large Language Models (LLMs) exhibit conspiratorial tendencies, whether they display socio-demographic biases in this domain, and how easily they can be conditioned into adopting conspiratorial perspectives. Conspiracy beliefs play a central role in the spread of misinformation and in shaping distrust toward institutions, making them an important testbed for assessing the social and psychological fidelity of LLMs and their potential to reproduce or reinforce harmful narratives. Although LLMs are often used as proxies for studying human behavior, it remains unclear whether they reproduce higher-order psychological constructs such as generalized conspiratorial beliefs. To bridge this research gap, we administer validated psychometric surveys measuring conspiratorial mindset to multiple models under different prompting and conditioning strategies. Our findings reveal that LLMs show partial agreement with elements 
    
[^146]: GSM8K-V：视觉语言模型能否在视觉情境中解决小学数学应用题

    GSM8K-V: Can Vision Language Models Solve Grade School Math Word Problems in Visual Contexts

    [https://arxiv.org/abs/2509.25160](https://arxiv.org/abs/2509.25160)

    该论文提出 GSM8K-V 基准，将文本数学题转化为多图像序列，发现视觉语言模型在视觉数学推理上存在显著模态差距（最佳模型仅 59%，远低于人类 91%）。

    

    数学推理是视觉语言模型（VLMs）的关键能力，然而当前的基准测试主要评估基于文本或显式符号化的视觉输入。目前尚不清楚，当信息必须从图像中感知和推断，而非从显式符号中读取时，VLMs 能否进行数学推理。我们引入了 GSM8K-V，一个将 GSM8K 转化为多图像序列并保持语义等价的基准测试。通过自动化流程和人工验证，将基于文本的问题映射为视觉形式，我们筛选出 1,319 个高质量样本。在 GSM8K-V 中，数量必须通过视觉感知提取，推理链必须通过整合跨场景的隐含线索来重建。对 34 个 VLMs 的评估揭示了一个显著的模态差距：虽然大多数模型在文本上超过 90%，但最佳模型在 GSM8K-V 上仅达到 59%，远低于人类 91% 的准确率。值得注意的是，为视觉增强的模型表现并未显著改善。

    arXiv:2509.25160v2 Announce Type: replace-cross  Abstract: Mathematical reasoning is a key capability for vision-language models (VLMs), yet current benchmarks mainly evaluate text-based or explicitly symbolic visual inputs. It remains unclear whether VLMs can reason mathematically when information must be perceived and inferred from images rather than read from explicit symbols. We introduce GSM8K-V, a benchmark transforming GSM8K into multi-image sequences with semantic equivalence preserved. By mapping text-based problems into visual form via an automated pipeline and human verification, we curate 1,319 high-quality samples. In GSM8K-V, quantities must be extracted through visual perception, and reasoning chains must be reconstructed by integrating implicit cues across scenes. Evaluation of 34 VLMs reveals a striking modality gap: while most models exceed 90\% on text, the best model achieves only 59\% on GSM8K-V, far below the 91\% human accuracy. Notably, models enhanced for visua
    
[^147]: 探索解分歧及其对大语言模型问题求解的影响

    Exploring Solution Divergence and Its Effect on Large Language Model Problem Solving

    [https://arxiv.org/abs/2509.22480](https://arxiv.org/abs/2509.22480)

    本文提出将“解分歧”作为新指标，发现其与大语言模型问题求解能力正相关，并能同时提升监督微调和强化学习的训练效果。

    

    大语言模型（LLMs）已被广泛应用于问题求解任务。近期大多数工作通过使用标注数据的监督微调（SFT）或基于任务反馈的强化学习（RL）来提升其性能。本文从一个新的视角展开研究：大语言模型针对单个问题所生成解的分歧性。我们表明，解分歧越高，与各种模型更强的问题求解能力呈正相关。基于这一发现，我们提出将解分歧作为一种新颖的度量指标，可以同时支持SFT和RL两种策略。我们在三个代表性问题领域上测试了这一想法，发现使用解分歧能够持续提升成功率。这些结果表明，解分歧是推进大语言模型训练与评估的一种简单而有效的工具。

    arXiv:2509.22480v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) have been widely used for problem-solving tasks. Most recent work improves their performance through supervised fine-tuning (SFT) with labeled data or reinforcement learning (RL) from task feedback. In this paper, we study a new perspective: the divergence in solutions generated by LLMs for a single problem. We show that higher solution divergence is positively related to better problem-solving abilities across various models. Based on this finding, we propose solution divergence as a novel metric that can support both SFT and RL strategies. We test this idea on three representative problem domains and find that using solution divergence consistently improves success rates. These results suggest that solution divergence is a simple but effective tool for advancing LLM training and evaluation.
    
[^148]: QoNext：迈向面向基础模型的下一代体验质量

    QoNext: Towards Next-generation QoE for Foundation Models

    [https://arxiv.org/abs/2509.21889](https://arxiv.org/abs/2509.21889)

    QoNext首次将网络与多媒体领域的体验质量原则引入人机交互评估，通过识别生成速度、延迟等动态体验因素构建了QoNext数据库，并训练出可直接从可测量因素预测用户体验的神经模型。

    

    现有的基础模型评估主要关注输出的正确性，将交互视为静态的信息交换。然而，这种视角忽视了由大语言模型驱动的对话体验的本质——该体验不仅取决于内容质量，更关键的是取决于生成速度和延迟模式等动态服务属性。为填补这一空白，我们提出了QoNext，这是首个将网络与多媒体领域的体验质量原则引入人机交互整体评估的框架。QoNext识别了塑造用户体验的体验因素，并将其纳入模拟交互场景的受控实验中，在多种配置下收集了人类评分。基于这些研究，我们构建了QoNext数据库，并训练了QoNext模型——一个能够直接从可测量因素估计用户体验的神经预测器。

    arXiv:2509.21889v3 Announce Type: replace  Abstract: Existing evaluations of foundation models predominantly focus on output correctness, treating interaction as a static exchange of information. However, such perspectives overlook the essence of the LLM-driven conversational experience, which is determined not only by content quality but, crucially, by dynamic service attributes such as generation velocity and latency patterns. To address this gap, we introduce QoNext, the first framework that adapts Quality of Experience (QoE) principles from networking and multimedia to the holistic assessment of human-AI interaction. QoNext identifies experiential factors that shape user experience and incorporates them into controlled experiments in simulated interaction scenarios, where human ratings are collected under diverse configurations. From these studies we construct the QoNext Database and train the QoNext Model, a neural predictor that estimates user experience directly from measurable 
    
[^149]: ConfRAG：置信度引导的检索增强生成

    ConfRAG: Confidence-Guided Retrieval-Augmenting Generation

    [https://arxiv.org/abs/2506.07309](https://arxiv.org/abs/2506.07309)

    提出ConfQA微调策略，通过训练模型在不确定时回答“我不确定”将LLM幻觉率从20-40%降至5%以下，并在此基础上构建ConfRAG，仅在模型置信度低时才触发检索增强生成，从而同时减少幻觉并降低检索计算成本。

    

    大型语言模型（LLM）能否被训练以避免对事实性陈述产生幻觉？检索增强生成（RAG）能否仅在必要时才被触发，以降低检索和计算成本？在本工作中，我们同时解决了这两个挑战。我们提出了ConfQA，一种微调策略，在多个事实性基准测试中将幻觉率从20-40%降低到5%以下。该方法非常简单：当模型能够正确回答时，训练其输出答案；否则，训练其回答“我不确定”。两个设计选择使这种训练有效：（1）一个抑制性提示（“只有在你有信心时才回答”），明确抑制过度自信的幻觉；（2）训练数据来自原子化的事实性陈述（例如知识图谱属性值），这可以校准模型置信度，并在不同领域和问题类型上产生稳健的泛化能力。基于ConfQA，我们……

    arXiv:2506.07309v3 Announce Type: replace  Abstract: Can Large Language Models (LLMs) be trained to avoid hallucinating factual statements, and can Retrieval-Augmented Generation (RAG) be triggered only when necessary to reduce retrieval and computation costs? In this work, we address both challenges simultaneously. We introduce ConfQA, a fine-tuning strategy that reduces hallucination rates from 20-40% to below 5% across multiple factuality benchmarks. The approach is simple: when the model answers correctly, it is trained to output the answer; otherwise, it is trained to respond with "I am unsure". Two design choices make this training effective: (1) a dampening prompt ("answer only if you are confident") that explicitly discourages overconfident hallucinations, and (2) training data drawn from atomic factual statements (e.g., knowledge graph attribute values), which calibrates model confidence and yields robust generalization across domains and question types. Building on ConfQA, we
    
[^150]: 驾驭推理经济：大语言模型高效推理综述

    Harnessing the Reasoning Economy: A Survey of Efficient Reasoning for Large Language Models

    [https://arxiv.org/abs/2503.24377](https://arxiv.org/abs/2503.24377)

    本综述系统性地提出了“推理经济”概念，用以刻画大语言模型在性能收益与计算成本之间的权衡，并从后训练和测试时推理两个阶段分析了推理低效的成因、不同推理模式的行为特征以及实现高效推理的潜在方向。

    

    大语言模型（LLM）的最新进展显著提升了其执行复杂推理任务的能力，实现了从快速直观的思考（系统1）到缓慢深入的推理（系统2）的转变。虽然系统2推理提高了任务准确性，但由于其慢思考的特性以及低效或不必要的推理行为，往往会产生巨大的计算成本。相比之下，系统1推理在计算上高效但性能欠佳。因此，在性能（收益）与计算成本（预算）之间进行权衡至关重要，由此产生了“推理经济”的概念。在本综述中，我们对大语言模型在后训练和测试时推理阶段的推理经济进行了全面分析，涵盖：i）推理低效产生的原因，ii）不同推理模式的行为分析，以及iii）潜在的优化方向。

    arXiv:2503.24377v2 Announce Type: replace-cross  Abstract: Recent advancements in Large Language Models (LLMs) have significantly enhanced their ability to perform complex reasoning tasks, transitioning from fast and intuitive thinking (System 1) to slow and deep reasoning (System 2). While System 2 reasoning improves task accuracy, it often incurs substantial computational costs due to its slow thinking nature and inefficient or unnecessary reasoning behaviors. In contrast, System 1 reasoning is computationally efficient but leads to suboptimal performance. Consequently, it is critical to balance the trade-off between performance (benefits) and computational costs (budgets), giving rise to the concept of reasoning economy. In this survey, we provide a comprehensive analysis of reasoning economy in both the post-training and test-time inference stages of LLMs, encompassing i) the cause of reasoning inefficiency, ii) behavior analysis of different reasoning patterns, and iii) potential 
    
[^151]: 用于检测值得核查社交媒体帖子的多语言模型

    Multilingual Models for Check-Worthy Social Media Posts Detection

    [https://arxiv.org/abs/2408.06737](https://arxiv.org/abs/2408.06737)

    本研究开发了能够同时处理英语和多种低资源语言的多标签多语言分类模型，用于检测社交媒体中包含可验证事实声明和有害声明的帖子。

    

    这项工作对基于Transformer的NLP模型在检测包含可验证事实声明和有害声明的社交媒体帖子方面的应用进行了广泛研究。该研究涵盖多项活动，包括数据集收集、数据集预处理、架构选择、参数配置、模型训练（微调）、模型测试和实现部署。该研究对不同模型进行了全面分析，特别关注多语言模型，即同一模型能够处理英语以及阿拉伯语、保加利亚语、荷兰语、波兰语、捷克语、斯洛伐克语等低资源语言的社交媒体帖子。研究中获得的结果与最先进的模型进行了对比验证，比较结果表明了所提出模型的鲁棒性。这项工作的创新之处在于开发了多标签多语言分类模型，可以同时检测有害帖子（原文此处截断）。

    arXiv:2408.06737v2 Announce Type: replace  Abstract: This work presents an extensive study of transformer-based NLP models application for detection of social media posts that contain verifiable factual claims and harmful claims. The study covers various activities, including dataset collection, dataset pre-processing, architecture selection, setup of settings, model training (fine-tuning), model testing, and implementation. The study includes a comprehensive analysis of different models, with a special focus on multilingual models where the same model is capable of processing social media posts in both English and in low-resource languages such as Arabic, Bulgarian, Dutch, Polish, Czech, Slovak. The results obtained from the study were validated against state-of-the-art models, and the comparison demonstrated the robustness of the proposed models. The novelty of this work lies in the development of multi-label multilingual classification models that can simultaneously detect harmful p
    

