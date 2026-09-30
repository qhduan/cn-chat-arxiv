# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Beyond the Timeline: Augmenting Long-Video Memory with Grounded Entity Biographies](https://arxiv.org/abs/2609.38155) | 该论文提出长视频记忆框架GEB（接地实体传记），将跨片段对同一物理实例的视觉观测聚合为可检索的实体传记，使模型在长视频问答中能够基于身份关联跨时间追踪特定实体。 |
| [^2] | [Effective Dense Retrieval using Only In-Context Examples](https://arxiv.org/abs/2609.38099) | 本文提出RICE方法，仅通过少量上下文示例提示LLM，无需任何检索器训练即可提取高质量稠密表示，显著提升基于提示的LLM嵌入在稠密检索任务中的准确性。 |
| [^3] | [Auditable Long-Term Memory: A Deterministic Retrieval Chain Measured at 479/475 of 500 on LongMemEval-S](https://arxiv.org/abs/2609.38021) | 该论文提出一种以确定性检索链（混合候选检索、交叉编码器重排序、覆盖优先数据包编译）为核心、LLM仅作为可替换最终阅读器的可审计长期记忆系统，在LongMemEval-S上两次评测分别获得479/500和475/500，成绩跨越Chronos High的478/500，但差异不足以证明优越性或等效性。 |
| [^4] | [BITEM at the NTCIR-19 R2C2 Task: Predicting Confidence from Agentic RAG Pipeline Signals](https://arxiv.org/abs/2609.37993) | 该论文提出一种由编排器根据代理式RAG流水线运行痕迹（蕴含级联核验证据、多轮去重检索等信号）来计算答案置信度、而无需模型自我评估的方法，在NTCIR-19 R2C2任务中取得第4和第5名，且多轮融合检索在多跳问题上增益最大。 |
| [^5] | [Generated Query Expansion Still Helps Strong Sparse Retrieval: A Controlled Study with SPLADE-v3](https://arxiv.org/abs/2609.37911) | 即使面对SPLADE-v3这样强大的稀疏检索器，在严格对照实验中生成式查询扩展仍能带来显著且稳健的检索性能提升。 |
| [^6] | [Towards Semi-Automatically Comparing Keyword-Based and Semantic Search Accuracy](https://arxiv.org/abs/2609.37749) | 本文提出一个新颖的初步框架，通过针对特定领域定制的可互换等价类，实现了基于关键词搜索的排序准确性与语义对话式搜索（如RAG）信息完整性之间的半自动定量比较。 |
| [^7] | [MERGE: Multi-LLM Ensemble for Retrieval via Generative Enrichment](https://arxiv.org/abs/2609.37574) | MERGE提出一个两阶段多LLM集成框架，先由三个小型开源LLM独立生成查询扩展候选、再由更大的LLM生成式合成为单一查询，并用基于下游检索性能的自动提示词优化循环取代传统LLM评估器，解决了单一LLM查询增强受限于模型偏见且提示词工程难以扩展的问题。 |
| [^8] | [Do Evidence-Reading Diagnostics Improve Interface Selection in Small LLM Recommenders?](https://arxiv.org/abs/2609.37472) | 研究发现，基于行为测试的证据阅读诊断特征并不能改善小型LLM推荐系统中的接口选择，添加诊断特征的选择器表现反而略差于基线选择器，表明此类诊断测量对推荐接口选择没有实际帮助。 |
| [^9] | [Relevance Is Not Sufficient Evidence: Detecting Evidence Gaps Before Generation in RAG](https://arxiv.org/abs/2609.37469) | 该研究揭示了现有证据不足测试基准的构建陷阱，并提出一个通过替换、删除和问题交换构造、控制表面特征的配对基准，证明可以在生成之前仅凭问题与检索证据判断其充分性，从而使RAG系统在证据缺失时更可靠地弃答。 |
| [^10] | [Backdoor in the Loop: Compromising Agentic Search via Malicious Retrievers](https://arxiv.org/abs/2609.37468) | 本文揭示了一种针对智能体检索增强生成系统的新型后门攻击：攻击者仅需提供一个被植入后门的检索器检查点，无需修改语料库即可抑制证据、劫持检索结果或诱导冗长搜索，并通过“注入-遗忘”循环伪造净化假象来规避后门检测。 |
| [^11] | [ReMem: Rethinking Perception and Memory in Long-Context Recommendation Agents](https://arxiv.org/abs/2609.37311) | ReMem提出了一种结合基于OCR的多模态感知与随时间演化动态记忆的推荐智能体框架，通过截图而非原始HTML来观察商品页面，实现更拟人化、平台无关的感知，并缓解长上下文推理低效的问题。 |
| [^12] | [Follow the Entities: A Corpus Map for Agentic Search](https://arxiv.org/abs/2609.37226) | 提出CorpusMap，一种围绕文档中反复出现实体来组织语料库的导航层，帮助大语言模型智能体在多文档搜索中发现文档间的关联，从而减少遗漏互补证据并降低令牌消耗。 |
| [^13] | [HELIX: Purified and Unified - Rethinking Feature Interaction and Sequence Modeling for Large-Scale Recommendation](https://arxiv.org/abs/2609.37183) | HELIX提出了一种纯化统一的大规模推荐架构，通过将序列检索与特征交互相交织并强制单向信息流，实现对特征交互与序列建模两个轴的联合扩展，从而获得更优的扩展律斜率。 |
| [^14] | [Optimizing VLP-aligned Multimodal Intent Representation with Correct Visual Instantiation for Zero-Shot Composed Image Retrieval](https://arxiv.org/abs/2609.36946) | 该论文提出统一的ZS-CIR框架VMIR-CVI，通过将多模态意图转换为与VLP原生文本空间对齐的统一文本描述，并进行正确的视觉实例化，从两个互补角度优化VLP兼容的多模态意图表示，从而提升零样本组合图像检索性能。 |
| [^15] | [Safer Content or Firmer Refusals? A Hybrid Perturbation Defense for Alignment under Harmful Fine-tuning](https://arxiv.org/abs/2609.36862) | 该论文提出 VaccineBooster，将嵌入层面的扰动防御与权重层面的梯度衰减防御融合到单一对齐流程的每个训练步骤中，有效抵御有害微调攻击，并在 Llama-2-7B 上取得最低的 OpenAI 审核分数。 |
| [^16] | [Does the Unsafe Gradient Survive a Conversation? On the Fragility of Gradient-Based Jailbreak Detection in Multi-Turn Dialogue](https://arxiv.org/abs/2609.36849) | 该论文揭示了基于梯度的越狱检测方法（如 GradSafe）在多轮对话场景下的脆弱性，提出通过上下文窗口扫描器对其进行扩展，并发现其在真实良性对话环境中的检测表现与合成良性数据环境存在显著差异。 |
| [^17] | [GRP v0.1 Technical Report](https://arxiv.org/abs/2609.36688) | GRP是一个将检索、排序和奖励建模统一在单一编码器-解码器模型中的生成式推荐框架，通过引入mGRPO强化学习后训练方法和服务优化，在实现69%检索延迟降低的同时，展示了通往端到端推荐的渐进式路径。 |
| [^18] | [Retrieval Sensitivity to Identity Signals in Queries](https://arxiv.org/abs/2609.36534) | 密集检索器会受查询中身份信号的影响而产生系统性偏差：返回与查询自身政治倾向一致的文章，且对非裔美国人语言（AAL）查询的检索效果劣于白人主流英语（WME）查询。 |
| [^19] | [ARCagent: An Adaptive Retrieval Calibration Agent for Clinical Question Answering](https://arxiv.org/abs/2609.36392) | 针对诊断框架并存、治疗指南相互矛盾的ME/CFS疾病，ARCagent提出了包含结构化冲突登记知识库、冲突感知检索校准流水线和LLM-as-Judge评分基准的临床问答代理，弥补了标准RAG系统在知识完整性与冲突感知综合方面的安全关键缺陷。 |
| [^20] | [Better Nearest Neighbor Graph Indices via (Efficient) LLM-Guided Pruning](https://arxiv.org/abs/2609.36359) | 该论文提出LLM引导的图剪枝（LGP）框架，直接利用LLM推理在索引构建层面改进现有ANN图索引，从根本上解决索引构建与检索评估之间的“几何-语义”错配问题。 |
| [^21] | [ThuRunel: Dynamic Decoupling for Structured Advisory Dialogue](https://arxiv.org/abs/2609.36340) | 提出ThuRunel咨询智能体，通过动态解耦机制决定问什么、何时停止、哪些自主解决、哪些转交专家，在医疗美容、法律咨询等高风险两阶段咨询场景中显著提升了引导完整性与专家简报质量。 |
| [^22] | [GeoOutageBench: Benchmarking Ambiguity-aware, Ontology-grounded Geospatiotemporal KGQA for Multimodal Power Outage and Resilience Analysis](https://arxiv.org/abs/2609.36082) | 该论文提出了GeoOutageBench，一个基于多模态时空知识图谱的基准测试，用于评估大语言模型在多模态停电与韧性分析中对歧义地理时空问题的理解、本体效用评估和答案准确性三方面能力。 |
| [^23] | [Mnemon: Raw Records, Fast Judgments, Slow Thoughts](https://arxiv.org/abs/2609.36059) | 提出记忆代理 Mnemon，将记忆工作类比为快慢双系统分工：LLM（System 2）负责规划搜索与组织答案，决策模型 Jev（System 1）快速执行大量记录判断，从而在保留带日期的原始对话记录的基础上高效构建长期记忆。 |
| [^24] | [Structured Interaction, Visual Localization, and Robust Execution for Complex Web Tasks: A Technical Report on the WebRetriever Challenge](https://arxiv.org/abs/2609.35904) | 该论文提出一种结构化交互优先的网页智能体系统，通过网格辅助视觉定位、分层上下文管理和故障感知执行三项设计，在 WebRetriever 挑战赛中以59%的通过率夺得冠军。 |
| [^25] | [Soft Curriculum Learning for Optimizing Fresh and Generalized Recommendations](https://arxiv.org/abs/2609.35783) | 本文提出一种可扩展的软课程学习方法，通过渐进式向模型暴露更困难、更低频的样本来打破大规模推荐系统中的流行度反馈回路，克服头部物品偏向问题，同时解决了传统课程学习在工业部署中的硬件利用效率瓶颈。 |
| [^26] | [Financial Evidence Crowding: Diagnosing and Mitigating Constraint-Induced Displacement in Retrieval-Augmented Generation](https://arxiv.org/abs/2609.35782) | 该论文揭示了金融问答RAG中的“金融证据拥挤”现象——主题匹配但时间期间、分部或口径冲突的候选证据会挤占有限的上下文槽位、排挤真正支持答案的证据（导致Recall@10下降0.147），并提出学习式分数修正方法FinDeCrowd-RAG来缓解这一问题。 |
| [^27] | [TSG Suggester: Tree-Structured Knowledge-Graph Retrieval for Troubleshooting Guide Recommendation in Cloud Incident Management](https://arxiv.org/abs/2609.35780) | 该论文提出TSG Suggester系统，通过将故障排除指南转换为保留章节层级的树结构并结合知识图谱（Tree+KG），在大语言模型生成的问题抽象的帮助下桥接指南与事件描述之间的语言差异，从而直接从事件描述中高效推荐相关的故障排除指南。 |
| [^28] | [Post-Generation Verification Dominates Retrieval Optimization: A 2^4 Factorial Ablation of RAG Pipeline Features](https://arxiv.org/abs/2609.35774) | 通过2^4全因子消融实验发现，RAG流水线中后生成验证（完整性检查）是主导性特征，仅其单独使用即优于所有其他特征组合，而目录引导检索可零成本显著提升效果。 |
| [^29] | [Socrates-RAG: Premise-Directed Inquiry against Coordinated Evidence Poisoning](https://arxiv.org/abs/2609.35773) | 提出Socrates-RAG，一种通过主动探询并解决关键未决前提来对抗协同证据投毒的前提导向主动检索策略，并在有限查询预算下提供了相对于常规查询策略的条件性挽救保证。 |
| [^30] | [AX is the New AEO](https://arxiv.org/abs/2609.34951) | 该论文提出“代理体验（AX）”——即AI智能体能否顺利抓取并阅读企业自身网站——正在取代答案引擎优化（AEO）成为决定AI推荐结果的关键因素，并通过超过3.7万次智能体买家旅程实验加以验证。 |
| [^31] | [IROH: Insightful Ranking Of Humor using Multi-Stage Hybrid Retrieval with Rationale-Distilled LLM Judges for JOKER 2026 Track Task 1 English](https://arxiv.org/abs/2609.15618) | 该论文提出IROH三阶段检索系统，通过稀疏-稠密混合检索、交叉编码器重排序与原理蒸馏的LoRA大语言模型裁判集成，在JOKER 2026任务1幽默排序中夺得第一名（MAP 0.6347），并发现原理蒸馏裁判是排序质量的关键驱动因素。 |
| [^32] | [RAISE: Diagnosing Acquisition Collapse in Costly LLM Signals](https://arxiv.org/abs/2608.10441) | 本文识别出“获取崩溃”这一失败模式，并提出了RAISE预路由诊断框架，帮助判断昂贵的LLM信号在何时才值得调用，从而避免盲目调用造成的资源浪费。 |
| [^33] | [OneLatent: Latent Reasoning for Efficient Foundation Recommendation Models](https://arxiv.org/abs/2607.26621) | OneLatent提出将显式思维链推理压缩为可学习的潜在token，实现“先潜在推理后回答”的高效推荐框架，通过多视角自适应CoT生成高质量监督信号，在捕捉多样化用户兴趣的同时大幅降低推理开销。 |
| [^34] | [Unlocking Spatial Grounding in Large Audio-Visual Retrieval models](https://arxiv.org/abs/2607.24786) | 该研究发现大规模视听检索模型的中间层视觉表示蕴含高度结构化的空间信息，并提出LAIP框架，通过轻量级的音频信息空间池化（AiSP）替代全局聚合，在弱监督条件下实现了细粒度的声源空间定位。 |
| [^35] | [SIREN (Luring LLMs onto the Rocks): PAIR-Driven Preference Manipulation in Web-RAG Recommenders](https://arxiv.org/abs/2607.21951) | 本文提出SIREN方法，利用PAIR越狱循环和23种可解释编辑策略，在Web-RAG推荐系统中操纵LLM的排名输出，使目标实体升至首位。 |
| [^36] | [The Hitchhiker's Guide to Agentic AI: From Foundations to Systems](https://arxiv.org/abs/2606.24937) | 该论文（著作）是一部从大语言模型基础、对齐与推理技术到智能体AI全栈覆盖的构建自主AI系统综合实践指南，其核心观点是：构建优秀的智能体系统必须理解技术栈的每一个层级。 |
| [^37] | [Agentic Graph Retrieval-Augmented Generation for Auditable Commercial Registry Analysis](https://arxiv.org/abs/2605.18770) | 本文提出一种受控的智能体GraphRAG架构，将瑞士官方商业公报构建为包含500余万节点和470万关系的Neo4j知识图谱，并通过意图路由、受限图工具、有界反思和状态机引导实现可审计的商业登记簿自然语言分析。 |
| [^38] | [UltRAG: a Universal Simple Scalable Recipe for Knowledge Graph RAG](https://arxiv.org/abs/2603.28773) | UltRAG是一种无需训练的知识图谱RAG方案，通过结合LLM查询生成、完全归纳式神经查询执行器和LLM仲裁，在无需重训模型的情况下于KGQA任务上取得最先进结果，并支持Wikidata规模的超大规模图谱。 |
| [^39] | [Evidence-Guided Schema Normalization for Temporal Tabular Reasoning](https://arxiv.org/abs/2512.00329) | 该研究将时序表格问答重构为自动化知识库构建任务，并通过受控交叉实验证明模式设计质量对问答准确率的影响远大于查询模型的选择（分别解释79.5%和1.6%的EM方差），据此提炼出模式设计原则。 |
| [^40] | [RecKG: Knowledge Graph for Recommender Systems](https://arxiv.org/abs/2501.03598) | 本文提出 RecKG——一个面向推荐系统的标准化知识图谱，通过统一的实体表示与命名规范实现异构数据源的无缝集成，从而挖掘出更多的语义信息。 |
| [^41] | [BadRAG: Identifying Vulnerabilities in Retrieval Augmented Generation of Large Language Models](https://arxiv.org/abs/2406.00083) | BadRAG揭示了一种针对检索增强生成系统的新型攻击：攻击者通过向知识库注入恶意文段，当用户查询包含特定触发词时即可操纵系统响应，且无需修改用户输入或模型权重。 |

# 详细

[^1]: 超越时间线：利用接地实体传记增强长视频记忆

    Beyond the Timeline: Augmenting Long-Video Memory with Grounded Entity Biographies

    [https://arxiv.org/abs/2609.38155](https://arxiv.org/abs/2609.38155)

    该论文提出长视频记忆框架GEB（接地实体传记），将跨片段对同一物理实例的视觉观测聚合为可检索的实体传记，使模型在长视频问答中能够基于身份关联跨时间追踪特定实体。

    

    回答关于长视频的问题通常需要跨越数小时甚至数天，将涉及同一对象的事件关联起来。按时间顺序的描述和从文本派生的实体往往无法解决物理身份的问题：不同的对象可能共享相同的描述，而同一对象的观测结果在不同事件之间仍相互割裂。因此，检索相关事件并不一定能恢复问题所涉及的那个特定实体的“传记”。为了解决这一问题，我们提出了接地实体传记（GEB），这是一个长视频记忆框架，它将跨视频片段中对同一物理实例的视觉接地观测聚合为可检索的实体传记，同时保留每个时刻的上下文。在问答过程中，实体传记与情节证据一同被检索，使模型能够借助记忆构建阶段建立的身份关联，沿事件轨迹追踪该实体的经历。在四个基准上的评估显示……

    arXiv:2609.38155v1 Announce Type: cross  Abstract: Answering questions about long videos often requires connecting events involving the same objects across hours or days. Chronological descriptions and text-derived entities can leave physical identity unresolved: different objects may share a description, while observations of the same object remain disconnected across events. Retrieving relevant events therefore does not necessarily recover the "biography" of the particular entity a question concerns. To address this, we introduce Grounded Entity Biographies (GEB), a long-video memory framework that groups visually grounded observations of the same physical instance across clips into retrievable biographies while preserving the context of each moment. During question answering, the biography is retrieved alongside episodic evidence, allowing the model to follow an entity through events using identity links established during memory construction. Evaluations across four benchmarks, inc
    
[^2]: 仅使用上下文示例实现高效的稠密检索

    Effective Dense Retrieval using Only In-Context Examples

    [https://arxiv.org/abs/2609.38099](https://arxiv.org/abs/2609.38099)

    本文提出RICE方法，仅通过少量上下文示例提示LLM，无需任何检索器训练即可提取高质量稠密表示，显著提升基于提示的LLM嵌入在稠密检索任务中的准确性。

    

    将仅解码器架构的大型语言模型（LLM）转变为强大的稠密检索器通常需要某种形式的检索器训练。在本文中，我们探讨了一个问题：在仅提供少量上下文示例的情况下，能否通过提示（prompting）让LLM直接生成有效的稠密检索表示？为回答这一问题，我们提出了RICE（Representations from In-Context Examples，来自上下文示例的表示），这是一种简单的“免训练”方法，可以从LLM中提取高质量的稠密表示。具体而言，RICE通过上下文示例为LLM提供共享语境，用于查询和文档的编码。我们的结果表明，RICE嵌入能够显著提升基于提示的LLM嵌入的准确性，使其成为构建无需训练的基于LLM的稠密检索器的一种简单方法。我们已在 https://github.com/nourj98/RICE 发布了代码。

    arXiv:2609.38099v1 Announce Type: cross  Abstract: Turning decoder-only large language models (LLMs) into strong dense retrievers typically requires some form of retriever training. In this paper, we ask whether LLMs can instead be prompted to produce effective representations for dense retrieval given only a few in-context examples. To answer this, we introduce RICE (Representations from In-Context Examples), a simple "training-free" approach that extracts high-quality dense representations from LLMs. To do so, RICE conditions the LLM on examples that provide a shared context for query and document encoding. Our results demonstrate that RICE embeddings can substantially improve the accuracy of prompt-based LLM embeddings, establishing it as a simple method to build LLM-based dense retrievers that do not require training. We release our code at https://github.com/nourj98/RICE.
    
[^3]: 可审计的长期记忆：在LongMemEval-S上测得479/475（满分500）成绩的确定性检索链

    Auditable Long-Term Memory: A Deterministic Retrieval Chain Measured at 479/475 of 500 on LongMemEval-S

    [https://arxiv.org/abs/2609.38021](https://arxiv.org/abs/2609.38021)

    该论文提出一种以确定性检索链（混合候选检索、交叉编码器重排序、覆盖优先数据包编译）为核心、LLM仅作为可替换最终阅读器的可审计长期记忆系统，在LongMemEval-S上两次评测分别获得479/500和475/500，成绩跨越Chronos High的478/500，但差异不足以证明优越性或等效性。

    

    我们在LongMemEval-S上评估了一个可审计的长期记忆系统。其检索链采用混合候选检索、交叉编码器重排序、以覆盖优先的数据包编译以及确定性推理脚手架；大语言模型仅作为可替换的最终阅读器使用。该检索链在470个可回答问题中的468个上将所有金标准会话纳入候选池，并为其中462个生成金标准完整的数据包。使用通过未固定版本的CLI别名调用的Claude Opus阅读器，在GPT-4o评分下，两次500题评测分别获得479/500和475/500的分数。其中72个可回答的知识更新题目使用了经过实质性修改的评分提示词，该修改在官方提示词文本下的效果尚未被测量。这一对结果跨越了Chronos High已发表的478/500；由于阅读器生成方式、评分提示词、可能的数据版本差异，以及系统内部的方差，这些结果既不能确立优越性，也不能确立等效性。在同一数据包上使用grok-4.6-high阅读器的得分为476/474，而……

    arXiv:2609.38021v1 Announce Type: cross  Abstract: We evaluate an auditable long-term memory system on LongMemEval-S. Its retrieval chain uses hybrid candidate retrieval, cross-encoder reranking, coverage-first packet compilation, and deterministic reasoning scaffolds; an LLM is used only as a replaceable final reader. The chain places all gold sessions in the candidate pool for 468/470 answerable questions and produces gold-complete packets for 462/470. With a Claude Opus reader called through an unpinned CLI alias, two 500-question passes score 479/500 and 475/500 under GPT-4o. The 72 answerable knowledge-update rows used a substantively modified scoring prompt whose effect under the official text has not been measured. The pair straddles Chronos High's published 478/500; differences in reader generation, scoring prompt, and possibly data version, plus within-system variance, establish neither superiority nor equivalence. A grok-4.6-high reader on the same packets scores 476/474, whi
    
[^4]: BITEM参加NTCIR-19 R2C2任务：从代理式RAG流水线信号预测置信度

    BITEM at the NTCIR-19 R2C2 Task: Predicting Confidence from Agentic RAG Pipeline Signals

    [https://arxiv.org/abs/2609.37993](https://arxiv.org/abs/2609.37993)

    该论文提出一种由编排器根据代理式RAG流水线运行痕迹（蕴含级联核验证据、多轮去重检索等信号）来计算答案置信度、而无需模型自我评估的方法，在NTCIR-19 R2C2任务中取得第4和第5名，且多轮融合检索在多跳问题上增益最大。

    

    BITEM团队使用单一的代理式流水线参加了NTCIR-19 R2C2任务的两个子任务。在该流水线中，一个模型在电影语料库上进行检索、阅读并记录证据，同时一个编排器持有记录并裁决哪些内容可以提交。只有当蕴含级联将某个论断与其所引用的段落核对之后，该论断才被采纳；只有当足够多的已核验证据支撑某个答案时，该答案才被发布。每个问题运行三到四次，且每一轮检索都从剥离了此前各轮已见内容的语料库中进行。随每个答案提交的置信度由编排器根据运行过程留下的痕迹计算得出，而从不询问模型本身，模型也没有任何自我评分的途径。两个检索运行在22个参赛系统中分别位列第4和第5，对多轮运行进行融合带来0.0709的nDCG@20提升，且提升在多跳和重后处理的问题上最大，组织方将融合后的运行在该类问题上排名第一。

    arXiv:2609.37993v1 Announce Type: cross  Abstract: The BITEM team entered both subtasks of the NTCIR-19 R2C2 task with a single agentic pipeline, in which a model searches, reads and records evidence over a movie corpus while an orchestrator holds the record and rules on what may be submitted. A claim is admitted only once an entailment cascade has checked it against the passage it cites, and an answer is released only once enough checked evidence stands behind it. Each question is run three or four times, every pass retrieving from a corpus stripped of what the earlier passes have already seen. The confidence filed with each answer is computed by the orchestrator from what the run leaves behind and is never asked of the model, which is offered no way to rate itself. The two retrieval runs placed 4th and 5th of 22, pooling the passes was worth 0.0709 nDCG@20, and the gain was largest on the multi-hop and post-processing-heavy questions, where the organisers rank the pooled run top of t
    
[^5]: 生成式查询扩展仍然有助于强稀疏检索：基于SPLADE-v3的对照研究

    Generated Query Expansion Still Helps Strong Sparse Retrieval: A Controlled Study with SPLADE-v3

    [https://arxiv.org/abs/2609.37911](https://arxiv.org/abs/2609.37911)

    即使面对SPLADE-v3这样强大的稀疏检索器，在严格对照实验中生成式查询扩展仍能带来显著且稳健的检索性能提升。

    

    科学查询通常较为简短，而相关论文却使用专业化的词汇，生成式查询扩展可以弥合这种不匹配，但早期研究表明其价值会随着底层检索器的变强而缩小。我们在NFCorpus、TREC-COVID和SciDocs上测试了四种生成格式——词项列表、伪文档、多个伪参考文献和语料库引导文本——与SPLADE-v3结合使用。每个条件都搜索相同的冻结文档索引，并遵循相同的查询侧集成规则和256维度预算，从而隔离出所添加内容的效果。所有十二个方法-数据集组合的比较都提升了总体nDCG@10，最佳相对增益分别为4.81%、8.92%和9.47%，其中十一项在Holm校正后仍然显著。该增益在114个插值设置中的103个中持续存在，包括每个将至少30%混合权重分配给原始查询的设置。打乱文本和非上下文……

    arXiv:2609.37911v1 Announce Type: cross  Abstract: Scientific queries are often brief, while relevant papers use specialized vocabulary. Generated query expansion can bridge this mismatch, but earlier work suggests that its value shrinks as the underlying retriever becomes stronger. We test the four generated formats of term lists, a pseudo-document, multiple pseudo-references, and corpus-steered text all together with SPLADE-v3 on NFCorpus, TREC-COVID, and SciDocs. Every condition searches the same frozen document index and follows the same query-side integration rule and 256-dimension budget, isolating the effect of the added content. All twelve method-collection comparisons improve aggregate nDCG@10, with best relative gains of 4.81%, 8.92%, and 9.47%. Eleven remain significant after Holm correction. The gain persists in 103 of 114 interpolation settings, including every setting that assigns at least 30% of the mixture weight to the original query. Shuffled-text and non-contextual l
    
[^6]: 迈向半自动比较基于关键词与语义搜索的准确性

    Towards Semi-Automatically Comparing Keyword-Based and Semantic Search Accuracy

    [https://arxiv.org/abs/2609.37749](https://arxiv.org/abs/2609.37749)

    本文提出一个新颖的初步框架，通过针对特定领域定制的可互换等价类，实现了基于关键词搜索的排序准确性与语义对话式搜索（如RAG）信息完整性之间的半自动定量比较。

    

    信息检索（IR）在管理大型数据集方面的日益重要性凸显了传统基于关键词的搜索系统的重大局限性。诸如检索增强生成（RAG）等上下文感知的对话式搜索方法近来不断涌现，但与基于关键词的系统相比，其评估往往依赖于主观的用户反馈。这两种范式之间仍然缺乏严格、量化的比较。本工作引入了一个新颖的初步框架，用于定量评估产生不同输出格式（如列表和消息）的搜索系统的信息检索准确性。该方法聚焦于两个关键方面：基于关键词系统的排序准确性，以及语义对话式系统所检索信息的完整性。我们的方法通过使用针对特定领域环境定制的可互换等价类，实现了对语义方法与基于关键词方法的半自动比较。

    arXiv:2609.37749v1 Announce Type: new  Abstract: The increasing importance of Information Retrieval (IR) in managing large datasets has highlighted significant limitations in traditional keyword-based search systems. Context-aware chat-based search methods, such as Retrieval Augmented Generation (RAG), have recently emerged, but their evaluation compared to keyword-based systems often relies on subjective user feedback. A rigorous, quantitative comparison between these paradigms remains lacking. This work introduces a novel, preliminary framework to quantitatively assess IR accuracy of search systems that produce different output formats, such as lists and messages. It focuses on two key aspects: the ranking accuracy for keyword-based systems and the completeness of retrieved information for semantic chat-based systems. Our approach enables semi-automatic comparisons of semantic and keyword-based methods using interchangeable equivalence classes tailored to domain-specific contexts (e.
    
[^7]: MERGE：基于生成式增强的多大语言模型集成检索框架

    MERGE: Multi-LLM Ensemble for Retrieval via Generative Enrichment

    [https://arxiv.org/abs/2609.37574](https://arxiv.org/abs/2609.37574)

    MERGE提出一个两阶段多LLM集成框架，先由三个小型开源LLM独立生成查询扩展候选、再由更大的LLM生成式合成为单一查询，并用基于下游检索性能的自动提示词优化循环取代传统LLM评估器，解决了单一LLM查询增强受限于模型偏见且提示词工程难以扩展的问题。

    

    大语言模型（LLM）越来越多地被用于信息检索（IR）中的用户查询增强，使BM25等标准检索器能够弥合查询与目标语料库之间的词汇鸿沟。然而，任何单一LLM都受限于其训练数据和架构偏见，且其增强行为依赖于手工设计的提示词——这些提示词必须针对每个新模型重新设计，是一个昂贵且难以扩展的过程。我们提出了MERGE（Multi-LLM Ensemble for Retrieval via Generative Enrichment，基于生成式增强的检索多LLM集成框架），这是一个两阶段框架：三个异构的7-8B开源LLM独立生成候选查询扩展，随后由一个更大的LLM将它们生成式地合成为单一查询。为了使提示词工程在整个模型集成中具备可扩展性，我们在两个阶段中都集成了基于任务的自动提示词优化（APO）循环。与使用LLM评估器来判断候选结果的APO方法不同，我们的循环根据每个候选的下游检索表现进行评分。

    arXiv:2609.37574v1 Announce Type: cross  Abstract: Large Language Models (LLMs) are increasingly used to enrich user queries in information retrieval (IR) so that a standard retriever such as BM25 can bridge vocabulary gaps with the target corpus. Any single LLM, however, is limited by its training data and architectural biases, and its enrichment behavior depends on hand-crafted prompts that must be re-engineered for each new model -- an expensive and poorly scalable process. We present MERGE (Multi-LLM Ensemble for Retrieval via Generative Enrichment), a two-stage framework: three heterogeneous 7-8B open-source LLMs independently produce candidate expansions, and a larger LLM generatively synthesizes them into a single query. To make prompt engineering scalable across the ensemble, we integrate a task-grounded Automatic Prompt Optimization (APO) loop into both stages. Unlike APO methods that judge candidates with an LLM evaluator, our loop scores each candidate by its downstream retr
    
[^8]: 证据阅读诊断能否改善小型LLM推荐系统中的接口选择？

    Do Evidence-Reading Diagnostics Improve Interface Selection in Small LLM Recommenders?

    [https://arxiv.org/abs/2609.37472](https://arxiv.org/abs/2609.37472)

    研究发现，基于行为测试的证据阅读诊断特征并不能改善小型LLM推荐系统中的接口选择，添加诊断特征的选择器表现反而略差于基线选择器，表明此类诊断测量对推荐接口选择没有实际帮助。

    

    行为测试用于衡量语言模型如何阅读证据。我们探究这些测量结果是否有助于选择推荐接口。我们在四个推荐领域中使用按时间顺序的评估方法和3,426名评估用户，对六个小型指令微调模型检查点进行了评估。每个请求对八个候选项目进行排序。基线选择器在三种接口之间进行选择：仅历史提示、带协同证据的提示以及分数融合。它使用可观测特征和六个改变措辞与候选顺序的稳定性提示。增强选择器则添加了来自六个证据阅读提示的特征，这些提示要求模型比较支持计数。在开发（验证）数据上为每个领域和检查点一次性选择的接口获得了0.5524的NDCG@5，相比之下基线选择器为0.5447，增强选择器为0.5428。添加诊断特征使NDCG@5变化了-0.0019（95%置信区间[-0.0046, 0.0004]）。该区间包括……

    arXiv:2609.37472v1 Announce Type: cross  Abstract: Behavioral tests measure how a language model reads evidence. We ask whether those measurements help choose a recommendation interface. We evaluate six small instruction-tuned checkpoints across four recommendation domains with chronological evaluation and 3,426 evaluation users. Each request ranks eight candidates. A baseline selector chooses among history-only prompting, prompting with collaborative evidence, and score fusion. It uses observable features and six stability prompts that vary wording and candidate order. An augmented selector adds features from six evidence-reading prompts that ask the model to compare support counts. An interface chosen once on development (validation) data for each domain and checkpoint scores 0.5524 NDCG@5, compared with 0.5447 for the baseline selector and 0.5428 for the augmented selector. Adding the diagnostic features changes NDCG@5 by -0.0019 (95% interval [-0.0046, 0.0004]). The interval includ
    
[^9]: 相关性并非充分证据：在RAG生成之前检测证据缺口

    Relevance Is Not Sufficient Evidence: Detecting Evidence Gaps Before Generation in RAG

    [https://arxiv.org/abs/2609.37469](https://arxiv.org/abs/2609.37469)

    该研究揭示了现有证据不足测试基准的构建陷阱，并提出一个通过替换、删除和问题交换构造、控制表面特征的配对基准，证明可以在生成之前仅凭问题与检索证据判断其充分性，从而使RAG系统在证据缺失时更可靠地弃答。

    

    检索增强生成（RAG）通过外部来源为大语言模型提供依据，但检索到的段落往往只是提到了正确的实体，却未提供回答问题所需的事实。即使被明确要求在证据不足时弃答，12个生成器仍会回答40.0%至99.3%的证据不足问题。通过训练生成器来学会弃答，会将这一决策绑定在模型权重上，可能奖励从参数化知识中“回忆”出的答案，且仍需要一次完整的生成器调用。那么，能否仅凭问题和证据，在任何答案产生之前判断证据是否充分？我们指出了构建“证据不足”测试时的陷阱：删除相关证据或将证据与不相关问题配对，可能通过词汇重叠或证据位置泄露标签。我们构建了一个配对基准，采用替换、删除和问题交换三种构造方式，在控制用词等选定表面特征的同时改变证据对答案的支持程度。证据充分性可以在……（原文摘要截断）

    arXiv:2609.37469v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) grounds large language models in external sources, but retrieved passages often name the right entities without providing the facts needed to answer. Even when instructed to abstain, 12 generators answer 40.0-99.3% of insufficient-evidence questions. Training generators to abstain ties the decision to model weights, may reward answers recalled from parametric knowledge, and still requires a full generator call. Can sufficiency be judged from the question and evidence alone, before any answer exists? We identify pitfalls in constructing insufficient-evidence tests: removing relevant evidence or pairing evidence with unrelated questions can reveal labels through lexical overlap or evidence position. We build a paired benchmark using substitution, deletion, and question-swap constructions that vary answer support while controlling selected surface features, such as word use. Sufficiency can be judged wit
    
[^10]: 环路中的后门：利用恶意检索器攻陷智能体搜索

    Backdoor in the Loop: Compromising Agentic Search via Malicious Retrievers

    [https://arxiv.org/abs/2609.37468](https://arxiv.org/abs/2609.37468)

    本文揭示了一种针对智能体检索增强生成系统的新型后门攻击：攻击者仅需提供一个被植入后门的检索器检查点，无需修改语料库即可抑制证据、劫持检索结果或诱导冗长搜索，并通过“注入-遗忘”循环伪造净化假象来规避后门检测。

    

    智能体检索增强生成（RAG）将推理与反复检索交织进行，使检索器不仅影响智能体所观察到的证据，还影响其后续的搜索决策。我们研究了利用这一反馈回路的检索器后门攻击，并重新利用弱后门净化手段来掩盖其存在。攻击者在保持搜索智能体和部署语料库不变的情况下，仅需提供一个被植入后门的检索器检查点。即使没有语料库的写入权限，攻击者仍可抑制有用的证据、持续检索某个选定的已有文档，或引导智能体陷入冗长的搜索，从而膨胀检索开销、上下文长度和延迟成本。为了向检测隐藏这些恶意行为，我们提出利用一种受控的“注入-移除”循环：故意注入一个较弱的后门，然后将其遗忘。这一过程会削弱检测器可观察到的特征签名，并以净化的假象欺骗后门检测器。

    arXiv:2609.37468v1 Announce Type: cross  Abstract: Agentic retrieval-augmented generation (RAG) interleaves reasoning with repeated retrieval, giving the retriever influence over both the evidence an agent observes and its subsequent search decisions. We study retriever backdoors that exploit this feedback loop and repurpose weak backdoor purification to conceal their presence. An attacker supplies a compromised retriever checkpoint while leaving the search agent and deployment corpus unchanged. Without corpus write access, the attacker can still suppress useful evidence, persistently retrieve a selected existing document, or steer the agent toward prolonged search, inflating retrieval, context, and latency cost. To conceal these behaviors from detection, we propose leveraging a controlled inject-and-remove cycle: deliberately inject a weaker backdoor and then unlearn it. This process weakens detector-visible signatures and fools the backdoor detectors with an illusion of purification 
    
[^11]: ReMem：重新思考长上下文推荐智能体中的感知与记忆

    ReMem: Rethinking Perception and Memory in Long-Context Recommendation Agents

    [https://arxiv.org/abs/2609.37311](https://arxiv.org/abs/2609.37311)

    ReMem提出了一种结合基于OCR的多模态感知与随时间演化动态记忆的推荐智能体框架，通过截图而非原始HTML来观察商品页面，实现更拟人化、平台无关的感知，并缓解长上下文推理低效的问题。

    

    近期的推荐智能体通过将推荐转变为一种主动的、用户侧的范式，提供了颇具前景的替代方案，其中生成式智能体能够自主感知外部平台、推理用户偏好并执行决策。然而，现有的推荐智能体仍存在两个关键局限：其一，基于嘈杂且异构的商品页面导致的脆弱的商品感知；其二，对冗长用户历史和多步交互轨迹进行低效的长上下文推理。为应对这些挑战，我们提出了一种新颖的推荐智能体框架，称为ReMem，它将基于OCR的多模态感知与随时间演化的动态记忆相结合。ReMem不再解析原始HTML，而是通过截图观察商品页面，并借助OCR工具提取结构化的多模态信息，从而实现更拟人化且与平台无关的感知机制。为支持长程偏好建模，ReMem进一步引入了……

    arXiv:2609.37311v1 Announce Type: new  Abstract: Recent Recommendation Agents (RecAgents) offer a promising alternative by shifting recommendation to an active, user-side paradigm, where generative agents autonomously perceive external platforms, reason over user preferences, and execute decisions. However, existing RecAgents still suffer from two critical limitations: brittle item perception based on noisy and heterogeneous item pages, and inefficient long-context reasoning over extended user histories and multi-step interaction traces. To address these challenges, we propose a novel recommendation agent framework, termed as ReMem, that combines OCR-based multimodal perception with time-evolving dynamic memory. Instead of parsing raw HTML, ReMem observes item pages through screenshots and extracts structured multimodal information via an OCR tool, enabling a more humanoid and platform-agnostic perception mechanism. To support long-horizon preference modeling, ReMem further introduces 
    
[^12]: 跟随实体：面向智能体搜索的语料库地图

    Follow the Entities: A Corpus Map for Agentic Search

    [https://arxiv.org/abs/2609.37226](https://arxiv.org/abs/2609.37226)

    提出CorpusMap，一种围绕文档中反复出现实体来组织语料库的导航层，帮助大语言模型智能体在多文档搜索中发现文档间的关联，从而减少遗漏互补证据并降低令牌消耗。

    

    在大规模文档集合上回答问题和完成任务，通常需要连接分散在多个文档中的证据，例如某个项目的批准记录在一个文档中，其需求在另一个文档中，而其最新状态在第三个文档中。近期的大语言模型（LLM）智能体通过迭代式搜索整个语料库来应对这一挑战，而不是仅阅读一组固定的排名靠前的文档。然而，当语料库仅以扁平的文件集合形式呈现时，相关文档无法体现它与其他文档之间的关系，因此智能体必须为每个查询重新发现这些关系，这往往导致遗漏互补证据，同时消耗大量额外的令牌（token）。为了解决这个问题，我们提出了CorpusMap，这是一个导航层，它围绕语料库中反复出现的实体来组织语料库。这些实体可以从文档本身中识别出来，并能够将单个文档与来自不同来源的许多其他文档链接起来。具体而言，CorpusMap……

    arXiv:2609.37226v1 Announce Type: cross  Abstract: Answering questions and completing tasks over large document collections often requires connecting evidence spread across multiple documents, such as a project's approval recorded in one, its requirements in another, and its latest status in a third. Recent LLM agents approach this by iteratively searching the full corpus rather than reading only a fixed set of top-ranked documents. However, when the corpus is exposed only as a flat collection of files, a relevant document gives no indication of how it relates to others, so the agent must rediscover these relationships for every query, often missing complementary evidence while simultaneously consuming substantial additional tokens. To address this, we introduce CorpusMap, a navigation layer that organizes the corpus around its recurring entities, which are identifiable from the documents themselves and can link a single document to many others across sources. Specifically, CorpusMap r
    
[^13]: HELIX：纯化与统一——重新思考大规模推荐中的特征交互与序列建模

    HELIX: Purified and Unified - Rethinking Feature Interaction and Sequence Modeling for Large-Scale Recommendation

    [https://arxiv.org/abs/2609.37183](https://arxiv.org/abs/2609.37183)

    HELIX提出了一种纯化统一的大规模推荐架构，通过将序列检索与特征交互相交织并强制单向信息流，实现对特征交互与序列建模两个轴的联合扩展，从而获得更优的扩展律斜率。

    

    工业推荐排序模型通常沿两个建模轴进行扩展：一是对异构的用户、物品、上下文及交叉特征进行特征交互，二是对长而信息丰富的多类型用户行为历史进行序列建模。我们发现单独扩展其中任何一种能力都是不够的，因为二者各自表现出有限的扩展上限和次优的扩展律斜率。我们推测，要获得更有利的扩展律斜率，需要对两个轴进行联合扩展。为验证这一观点，我们提出了HELIX，一种面向大规模推荐的纯化且统一的架构。HELIX将序列检索与特征交互交织进行，同时强制从可复用的序列状态到候选条件混合标记的单向信息流。这一设计在保持两个建模轴之间跨深度通信的同时，使用户侧的序列计算可以摊销，从而实现了序列建模的灵活且非对称的扩展。

    arXiv:2609.37183v1 Announce Type: new  Abstract: Industrial recommendation ranking models typically scale along two modeling axes: feature interaction over heterogeneous user, item, context, and cross features, and sequence modeling over long, informative, and multi-type user behavior histories. We find that scaling either capability in isolation is insufficient, as each exhibits a limited scaling ceiling and a suboptimal scaling-law slope. We conjecture that achieving a more favorable scaling-law slope requires jointly scaling both axes. To support this, we present HELIX, a purified and unified architecture for large-scale recommendation. HELIX interleaves sequence retrieval and feature interaction while enforcing one-way information flow from reusable sequence states to candidate-conditioned mix-tokens. This design preserves cross-depth communication between the two modeling axes while keeping user-side sequence computation amortizable, enabling flexible and asymmetric scaling of seq
    
[^14]: 通过正确的视觉实例化优化VLP对齐的多模态意图表示以实现零样本组合图像检索

    Optimizing VLP-aligned Multimodal Intent Representation with Correct Visual Instantiation for Zero-Shot Composed Image Retrieval

    [https://arxiv.org/abs/2609.36946](https://arxiv.org/abs/2609.36946)

    该论文提出统一的ZS-CIR框架VMIR-CVI，通过将多模态意图转换为与VLP原生文本空间对齐的统一文本描述，并进行正确的视觉实例化，从两个互补角度优化VLP兼容的多模态意图表示，从而提升零样本组合图像检索性能。

    

    ZS-CIR（零样本组合图像检索）旨在仅凭参考图像和修改文本检索目标图像，而无需配对监督，通常通过将组合查询编码为VLP（视觉-语言预训练模型）图像-文本匹配空间中的文本主导表示来实现。然而，无论是通过视觉伪词学习还是基于多模态大语言模型（MLLM）的目标推理来重建查询，都常常偏离原生VLP表示空间：前者受参考噪声和粗糙文本融合的影响，后者则存在描述冗长且视觉基础薄弱的问题。在本文中，我们提出了一个统一的ZS-CIR框架（命名为VMIR-CVI），从两个互补的角度重建多模态组合查询，以优化与VLP兼容的多模态意图表示。首先，它将多模态意图推理并转换为统一的文本描述，与VLP主干的原生文本空间对齐，从而生成更具检索兼容性的文本查询。其次，它重建查询……

    arXiv:2609.36946v1 Announce Type: new  Abstract: ZS-CIR aims to retrieve a target image from a reference image and a modification text without paired supervision, typically by encoding composed queries as text-dominant representations within the image-text matching space of VLPs. However, queries reconstructed by visual pseudo-word learning or MLLM-based target reasoning often deviate from the native VLP representation space due to reference noise and coarse text fusion in the former, and verbose, weakly visually grounded descriptions in the latter. In this paper, we propose a unified ZS-CIR framework (named VMIR-CVI) to reconstruct multimodal composite queries from two complementary perspectives for optimizing VLP-compatible multimodal intent representation. First, it reasons and converts the multimodal intent into a unified textual description, aligning with the native text space of the VLP backbones to produce more retrieval-compatible textual queries. Second, it reconstructs the qu
    
[^15]: 更安全的内容还是更坚定的拒绝？一种面向有害微调下对齐安全的混合扰动防御方法

    Safer Content or Firmer Refusals? A Hybrid Perturbation Defense for Alignment under Harmful Fine-tuning

    [https://arxiv.org/abs/2609.36862](https://arxiv.org/abs/2609.36862)

    该论文提出 VaccineBooster，将嵌入层面的扰动防御与权重层面的梯度衰减防御融合到单一对齐流程的每个训练步骤中，有效抵御有害微调攻击，并在 Llama-2-7B 上取得最低的 OpenAI 审核分数。

    

    微调即服务让用户能够将安全对齐的语言模型适配到自己的数据上，但这也带来了有害微调的攻击面：在原本良性的微调数据集中混入少量有害数据，就可能破坏模型的对齐性。近期有两种对齐阶段的防御方法在不同模型层面应对这一问题：Vaccine 提升隐藏嵌入对有害微调所引起的表示偏移的鲁棒性，而 Booster 则模拟有害的权重更新并在对齐过程中减弱其影响。我们研究这两种机制是否互补，并提出了 VaccineBooster——一种单一的对齐流程，在每个训练步骤中同时结合嵌入扰动与权重层面的梯度衰减。在以 BeaverTails 对齐的 Llama-2-7B 上遭受投毒微调攻击的实验中，VaccineBooster 在所对比的防御方法中取得了最低的 OpenAI 审核分数。

    arXiv:2609.36862v1 Announce Type: cross  Abstract: Fine-tuning-as-a-service lets users adapt a safety-aligned language model to their own data, but it also creates a harmful fine-tuning attack surface: a small amount of harmful data mixed into an otherwise benign fine-tuning set can degrade the model's alignment. Two recent alignment-stage defenses address this problem at different levels of the model. Vaccine improves the robustness of hidden embeddings to the representation shifts induced by harmful fine-tuning, whereas Booster simulates harmful weight updates and attenuates their effect during alignment. We investigate whether these mechanisms are complementary and propose VaccineBooster, a single alignment procedure that combines embedding perturbation and weight-level gradient attenuation within each training step. On Llama-2-7B aligned with BeaverTails and then attacked through poisoned fine-tuning, VaccineBooster achieves the lowest OpenAI moderation score among the compared def
    
[^16]: 不安全梯度能否在对话中幸存？论多轮对话中基于梯度的越狱检测的脆弱性

    Does the Unsafe Gradient Survive a Conversation? On the Fragility of Gradient-Based Jailbreak Detection in Multi-Turn Dialogue

    [https://arxiv.org/abs/2609.36849](https://arxiv.org/abs/2609.36849)

    该论文揭示了基于梯度的越狱检测方法（如 GradSafe）在多轮对话场景下的脆弱性，提出通过上下文窗口扫描器对其进行扩展，并发现其在真实良性对话环境中的检测表现与合成良性数据环境存在显著差异。

    

    安全对齐的语言模型通常被部署为多轮对话助手，这使得攻击者能够将不安全意图分散到多个用户轮次中，而非集中在单个提示词里。基于梯度的越狱检测器（如 GradSafe）是为单条提示词设计的：它们通过输入所产生的梯度与固定不安全参考方向之间的对齐程度来对输入进行评分，但其在多轮对话中的有效性尚不明确。我们对多轮场景下基于梯度的越狱检测进行了受控评估。我们通过一个上下文窗口扫描器对 GradSafe 进行了扩展，该扫描器将检测器应用于固定大小的用户轮次窗口，并使用最大窗口评分作为对话级别的评分。我们评估了不同的窗口大小、攻击类型、良性对话分布和目标模型。结果显示，在合成良性数据与真实良性数据两种设置下，检测效果存在显著差异。面对合成良性对话时，该检测器能够达到……（原文摘要在此处截断）

    arXiv:2609.36849v1 Announce Type: cross  Abstract: Safety-aligned language models are commonly deployed as multi-turn assistants, which lets adversaries spread unsafe intent across several user turns instead of a single prompt. Gradient-based jailbreak detectors such as GradSafe were developed for single prompts: they score an input by the alignment between its induced gradient and a fixed unsafe reference direction, and their effectiveness in multi-turn dialogue remains unclear. We conduct a controlled evaluation of gradient-based jailbreak detection in multi-turn settings. We extend GradSafe with a Context Window Scanner that applies the detector to fixed-size windows of user turns and uses the maximum window score as the conversation-level score. We evaluate different window sizes, attack families, benign conversation distributions, and target models. The results differ sharply between synthetic and realistic benign settings. Against synthetic benign conversations, the detector achi
    
[^17]: GRP v0.1 技术报告

    GRP v0.1 Technical Report

    [https://arxiv.org/abs/2609.36688](https://arxiv.org/abs/2609.36688)

    GRP是一个将检索、排序和奖励建模统一在单一编码器-解码器模型中的生成式推荐框架，通过引入mGRPO强化学习后训练方法和服务优化，在实现69%检索延迟降低的同时，展示了通往端到端推荐的渐进式路径。

    

    工业推荐系统依赖于多阶段级联架构，其检索、排序和服务组件难以联合替换。我们提出了GRP，一个将检索、排序和奖励建模结合在单一编码器-解码器模型中的生成式推荐框架，并评估了一条通往端到端推荐的渐进式路径。该模型生成多模态语义ID（Semantic IDs），并通过联合训练的排序模块对候选进行打分。随后，冻结的排序模块为强化学习后训练提供奖励信号。我们提出了mGRPO方法，它在奖励优化中加入了参考锚定的边际项，以保持已记录目标行为（logged targets）的似然。离线实验考察了历史编码、模型容量分配、事件选择、分词和奖励判别等方面。服务优化将端到端检索延迟降低了69%。在线实验将模型作为检索源进行评估，早期排名……（摘要在此处截断）

    arXiv:2609.36688v1 Announce Type: new  Abstract: Industrial recommendation systems rely on multi-stage cascades whose retrieval, ranking, and serving components are difficult to replace jointly. We present GRP, a generative recommendation framework that combines retrieval, ranking, and reward modeling in a single encoder-decoder model, and evaluate a progressive path toward end-to-end recommendation. The model generates multimodal Semantic IDs and scores candidates with a jointly trained ranking module. The frozen ranking module then supplies rewards for reinforcement-learning post-training. We introduce mGRPO, which adds a reference-anchored margin to reward optimization to preserve the likelihood of logged targets. Offline experiments examine history encoding, model capacity allocation, event selection, tokenization, and reward discrimination. Serving optimizations reduce end-to-end retrieval latency by 69%. Online experiments evaluate the model as a retrieval source, with early-rank
    
[^18]: 检索系统对查询中身份信号的敏感性

    Retrieval Sensitivity to Identity Signals in Queries

    [https://arxiv.org/abs/2609.36534](https://arxiv.org/abs/2609.36534)

    密集检索器会受查询中身份信号的影响而产生系统性偏差：返回与查询自身政治倾向一致的文章，且对非裔美国人语言（AAL）查询的检索效果劣于白人主流英语（WME）查询。

    

    密集检索器决定了哪些文档能够到达用户以及使用这些文档的语言模型，然而它们通常是用中性查询来评估的。我们探究真实用户在查询中表达的身份信号——政治意识形态和方言——是否会使检索器返回的结果产生偏差。我们在两个领域设计了评估：政治新闻和消费者健康问题，每个领域都将一个仅改变身份信号的受控合成数据集与自然查询相配对。在五个密集检索器和一个稀疏检索基线上，每个检索器（i）都会检索出与查询自身政治倾向一致的文章，并且（ii）对以非裔美国人语言（AAL）撰写的问题的表现比对以白人主流英语（WME）撰写的问题更差。两项分析将这些差距与查询中超越表面词汇层面的身份信号联系起来：在剔除总体的词汇不对称分数后，合成数据上的差距基本保持不变，且线性探针能够恢复出查询的倾向和……（原文摘要在此处截断）

    arXiv:2609.36534v1 Announce Type: new  Abstract: Dense retrievers decide which documents reach users and the language models that use them, yet they are typically evaluated with neutral queries. We ask whether the identity signals that real users express in their queries---political ideology and dialect---bias what a retriever returns. We design evaluations in two domains, political news and consumer-health questions, each pairing a controlled synthetic set that varies only the identity signal with naturalistic queries. Across five dense retrievers and a sparse baseline, every retriever (i) retrieves articles that align with the query's own political lean and (ii) performs worse for questions written in African American Language (AAL) than in White Mainstream English (WME). Two analyses tie these gaps to queries' identity signals beyond surface vocabulary: partialling out an aggregate lexical-asymmetry score leaves the synthetic gaps largely intact, and linear probes recover lean and d
    
[^19]: ARCagent：一种用于临床问答的自适应检索校准代理

    ARCagent: An Adaptive Retrieval Calibration Agent for Clinical Question Answering

    [https://arxiv.org/abs/2609.36392](https://arxiv.org/abs/2609.36392)

    针对诊断框架并存、治疗指南相互矛盾的ME/CFS疾病，ARCagent提出了包含结构化冲突登记知识库、冲突感知检索校准流水线和LLM-as-Judge评分基准的临床问答代理，弥补了标准RAG系统在知识完整性与冲突感知综合方面的安全关键缺陷。

    

    在临床指南不完整、存在争议或相互矛盾的疾病中，知识完整性与动态的冲突感知综合是两项安全关键属性，而标准的检索增强生成（RAG）系统无法提供这些特性。为此，我们提出了ARCagent，一个针对ME/CFS（肌痛性脑脊髓炎/慢性疲劳综合征）的自适应检索校准临床问答代理——ME/CFS是一种多种诊断框架并存、且主要指南在治疗方面相互积极矛盾的疾病。ARCagent贡献了三个组件：第一，一个包含1,706个文本块、10个数据来源的知识库，其中带有覆盖所有现行ME/CFS诊断框架的结构化指南间冲突登记表；第二，一个冲突感知的检索校准流水线，利用特定查询的重点信号和冲突信号对检索到的证据进行重新排序；第三，一个采用LLM-as-Judge评分的基准测试，避免了关键词匹配方法造成的平均10.1个百分点的系统性低估。

    arXiv:2609.36392v1 Announce Type: new  Abstract: In diseases where clinical guidelines are incomplete, contested, or mutually contradictory, knowledge completeness and dynamic conflict-aware synthesis are two safety-critical properties that standard Retrieval-Augmented Generation systems do not provide. Therefore, we present \sysname, an adaptive retrieval calibration clinical question-answering agent for ME/CFS, a disease where diagnostic frameworks coexist and major guidelines actively contradict each other on treatment. ARCagent contributes three components. First, a 1,706-chunk, 10-source knowledge base with a structured inter-guideline conflict registry spanning all active ME/CFS diagnostic frameworks. Second, a conflict-aware retrieval calibration pipeline that re-ranks retrieved evidence using query-specific focus and conflict signals. Third, a benchmark scored by LLM-as-Judge, avoiding systematic underestimation averaging 10.1 percentage points caused by keyword matching. ARCag
    
[^20]: 通过（高效的）LLM引导的剪枝构建更优的最近邻图索引

    Better Nearest Neighbor Graph Indices via (Efficient) LLM-Guided Pruning

    [https://arxiv.org/abs/2609.36359](https://arxiv.org/abs/2609.36359)

    该论文提出LLM引导的图剪枝（LGP）框架，直接利用LLM推理在索引构建层面改进现有ANN图索引，从根本上解决索引构建与检索评估之间的“几何-语义”错配问题。

    

    基于图的近似最近邻搜索（ANNS）被广泛用于大规模语义搜索。其索引的构建主要基于输入数据集（例如文档或图像）嵌入之间的几何关系，而非显式地优化语义相关性。然而，当使用这些索引进行下游查询检索时，性能却是根据检索结果与查询之间的语义相关性来评估的。这在索引的构建方式与其检索结果的评估方式之间造成了一个根本性的“几何-语义”错配。虽然现有的基于LLM的重排序方法可以在查询时部分缓解这种错配，但它们并未解决图索引中这一底层结构问题。因此，我们提出了LLM引导的图剪枝，这是一个通用框架，通过利用LLM推理直接解决这一错配，从而改进现有的ANN图索引本身。

    arXiv:2609.36359v1 Announce Type: new  Abstract: Graph-based approximate nearest neighbor search (ANNS) is widely used for large-scale semantic search. Its indices are constructed primarily based on geometric relationships among embeddings of an input dataset (e.g., documents or images), rather than explicitly optimizing for semantic relevance. However, when using these indices for downstream query retrieval, performance is evaluated based on the semantic relevance of the retrieved results to the query. This creates a fundamental "geometry-semantic" mismatch between how the indices are constructed and how their retrieval results are evaluated. While existing LLM-based reranking methods can partially mitigate this mismatch at query time, they leave this underlying structural problem in the graph unresolved. We therefore propose LLM-Guided Graph Pruning (LGP), a general framework that addresses this mismatch directly by leveraging LLM reasoning to refine an existing ANN graph index itsel
    
[^21]: ThuRunel：面向结构化咨询对话的动态解耦

    ThuRunel: Dynamic Decoupling for Structured Advisory Dialogue

    [https://arxiv.org/abs/2609.36340](https://arxiv.org/abs/2609.36340)

    提出ThuRunel咨询智能体，通过动态解耦机制决定问什么、何时停止、哪些自主解决、哪些转交专家，在医疗美容、法律咨询等高风险两阶段咨询场景中显著提升了引导完整性与专家简报质量。

    

    医疗美容、法律咨询和教育规划等高风险咨询领域呈现出两阶段结构：早期阶段需要共情式引导和情感支持，后期阶段则需要权威的专业判断。无论是完全自动化的智能体还是人类初级咨询师，都无法在规模化场景下充分应对这种结构。我们将核心设计挑战形式化为“动态解耦”，即AI咨询智能体应如何决定问什么、何时停止、哪些问题可以自主解决、哪些问题应转发给专家。我们提出了ThuRunel，这是一个结合了有限状态信念管理框架、思维链教师合成协议以及学习型生成适配器的咨询智能体。与十一个基线方法相比，ThuRunel在引导完整性和专家简报质量方面取得了一致的改进。ThuRunel目前已作为双语Web应用公开部署，其中相同（原文在此处截断）

    arXiv:2609.36340v1 Announce Type: new  Abstract: High-stakes advisory domains such as medical aesthetics, legal consultation, and educational planning exhibit a two-phase structure. The early phase requires empathetic elicitation and emotional support, and the late phase requires authoritative specialist judgment. Neither fully automated agents nor human junior consultants adequately address this structure at scale. We formalize the core design challenge as dynamic decoupling, asking how an AI advisory agent should decide what to ask, when to stop, what to resolve autonomously, and what to forward to the specialist. We present ThuRunel, an advisory agent combining a finite-state belief management framework, a chain-of-thought teacher synthesis protocol, and learned generation adapters. Against eleven baselines, ThuRunel achieves consistent improvements in elicitation completeness and specialist brief quality. ThuRunel is publicly deployed as a bilingual web application in which the sam
    
[^22]: GeoOutageBench：面向多模态停电与韧性分析的歧义感知、本体驱动的地理时空知识图谱问答基准测试

    GeoOutageBench: Benchmarking Ambiguity-aware, Ontology-grounded Geospatiotemporal KGQA for Multimodal Power Outage and Resilience Analysis

    [https://arxiv.org/abs/2609.36082](https://arxiv.org/abs/2609.36082)

    该论文提出了GeoOutageBench，一个基于多模态时空知识图谱的基准测试，用于评估大语言模型在多模态停电与韧性分析中对歧义地理时空问题的理解、本体效用评估和答案准确性三方面能力。

    

    我们提出了GeoOutageBench，这是一个用于评估基于大语言模型（LLM）的地理时空知识图谱问答（KGQA）在多模态停电与韧性分析方面能力的基准测试。与现有的面向网络知识的KGQA基准不同，GeoOutageBench采用了一个时空知识图谱，该图谱整合了来自停电记录、遥感、天气观测、风暴和电力事件、地理实体以及领域本体的视觉、文本和结构化数据。它提供了一个不同难度级别的能力查询分类体系，涵盖时空包含与邻近关系、时空共现分析、多模态证据以及假设评估。基于多模态知识图谱和查询类别，GeoOutageBench提供了用户可配置的评估，涵盖三个重要、高度相关但研究较少的任务：（1）LLM对歧义地理时空问题的理解能力，即自然语言到SPARQL的转换；（2）查询驱动的本体效用评估；（3）答案准确性评估……

    arXiv:2609.36082v1 Announce Type: new  Abstract: We introduce GeoOutageBench, a benchmark for assessing LLM-based geospatiotemporal KGQA for multimodal outage and resilience analysis. Unlike existing KGQA benchmarks for Web knowledge, GeoOutageBench considers a spatiotemporal KG that integrates visual, textual, and structured data from outage records, remote sensing, weather observations, storm and power events, geographic entities, and domain ontologies. It provides a competency query taxonomy at different difficulty levels from spatiotemporal containment and proximity, spatiotemporal co-occurrence analysis, multimodal evidence, to hypothetical evaluation. Over multimodal KG and query classes, GeoOutageBench provides user-configurable evaluation of three important, highly coherent yet less studied tasks: (1) LLMs' understanding for ambiguous geospatiotemporal questions in terms of NL to SPARQL interpretation, (2) query-driven assessment of ontology utility, and (3) answer accuracy of 
    
[^23]: Mnemon：原始记录、快速判断与慢速思考

    Mnemon: Raw Records, Fast Judgments, Slow Thoughts

    [https://arxiv.org/abs/2609.36059](https://arxiv.org/abs/2609.36059)

    提出记忆代理 Mnemon，将记忆工作类比为快慢双系统分工：LLM（System 2）负责规划搜索与组织答案，决策模型 Jev（System 1）快速执行大量记录判断，从而在保留带日期的原始对话记录的基础上高效构建长期记忆。

    

    长期记忆使 LLM 助手能够利用其已无法重新阅读的历史对话，而大多数记忆系统是在写入时通过将对话重写为事实、图谱或类型化记忆来构建长期记忆。我们认为，记忆的工作如同思考一样，可以划分为两个系统。其中大部分是快速的 System 1 工作：对记录进行大量独立的是/否判断，例如某条记录是否仍然需要或是否已过时，一个决策模型可以在三分之一秒内完成数十个这样的判断。只有少量工作是慢速的 System 2 工作：撰写少量搜索查询、明确回答所需的内容并组织答案，这些任务 LLM 擅长但速度较慢。我们提出了 Mnemon，一个基于这种分工构建的记忆代理。它将对话保存为带有日期的原始记录；LLM（System 2）负责规划对这些记录的搜索，决策模型 Jev（System 1）负责判断搜索返回的结果，而带有明确预算的规则将这些判断转化为一个小型视图，供未经修改的应答模型使用。

    arXiv:2609.36059v1 Announce Type: cross  Abstract: Long-term memory lets an LLM assistant use a history it can no longer reread, and most memory systems build it by rewriting conversations into facts, graphs or typed memories at write time. We argue that the work of memory divides, as thinking does, into two systems. Most of it is fast System 1 work: many small, independent yes/no judgments about records, such as whether a record is needed or no longer current, which a decision model makes by the dozen in a third of a second. Only a little is slow System 2 work: writing a few search queries, naming what the reply needs and composing the answer, which an LLM does well but slowly. We present Mnemon, a memory agent built on this division. It keeps conversations as raw, dated records; an LLM (System 2) plans searches over them, a decision model, Jev (System 1), judges what the searches return, and rules with explicit budgets turn the judgments into a small View for an unchanged answering m
    
[^24]: 面向复杂网页任务的结构化交互、视觉定位与鲁棒执行：WebRetriever 挑战赛技术报告

    Structured Interaction, Visual Localization, and Robust Execution for Complex Web Tasks: A Technical Report on the WebRetriever Challenge

    [https://arxiv.org/abs/2609.35904](https://arxiv.org/abs/2609.35904)

    该论文提出一种结构化交互优先的网页智能体系统，通过网格辅助视觉定位、分层上下文管理和故障感知执行三项设计，在 WebRetriever 挑战赛中以59%的通过率夺得冠军。

    

    本报告介绍了为 WebRetriever 挑战赛开发的网页智能体（web agent）系统。该系统遵循“结构化交互优先”策略，利用网页语义信息完成常规浏览器操作，仅当结构化表示不足以应对时才调用视觉感知。系统引入了三项关键设计：用于操作难以访问控件的网格辅助视觉定位、用于减少冗余页面与交互历史的分层上下文管理，以及用于稳定处理多浏览器任务的故障感知执行机制。该系统在协议1（Protocol 1）的本地评估中通过率高达79%；在官方协议3（Protocol 3）竞赛中，系统以八个并发浏览器工作进程实现了59%的通过率，综合排名第一，赢得了 WebRetriever 挑战赛冠军。

    arXiv:2609.35904v1 Announce Type: new  Abstract: This report presents the web agent system developed for the WebRetriever Challenge. The system follows a structuredinteraction- first strategy, using semantic webpage information for routine browser operations and invoking visual perception only when structured representations are insufficient. Three key designs are introduced: grid-assisted visual localization for difficult-to-access controls, hierarchical context management for reducing redundant page and interaction history, and fault-aware execution mechanisms for stable multi-browser task processing. The system achieved a pass rate of up to 79% in local evaluation on Protocol 1. In the official Protocol 3 competition, it achieved a 59% pass rate with eight concurrent browser workers and ranked first overall, winning the WebRetriever Challenge.
    
[^25]: 面向新鲜与泛化推荐优化的软课程学习

    Soft Curriculum Learning for Optimizing Fresh and Generalized Recommendations

    [https://arxiv.org/abs/2609.35783](https://arxiv.org/abs/2609.35783)

    本文提出一种可扩展的软课程学习方法，通过渐进式向模型暴露更困难、更低频的样本来打破大规模推荐系统中的流行度反馈回路，克服头部物品偏向问题，同时解决了传统课程学习在工业部署中的硬件利用效率瓶颈。

    

    大规模推荐系统，尤其是短视频平台，往往受到海量流行度反馈回路的瓶颈制约。在这种环境中，当模型推荐热门物品时，会产生海量偏向“头部”物品的倾斜训练数据，形成一个自我强化的循环，使检索和排序模型以牺牲对目录中庞大“尾部”内容的泛化能力为代价来记忆“头部”物品的模式。虽然课程学习（CL）提供了一种通过系统性地让模型接触渐进式更困难、更低频样本来打破这种反馈回路的强大机制，但其在工业推荐中的采用一直受到硬件利用效率低下或需要复杂预处理技术的阻碍，因为动态数据拒绝算法往往主要受CPU限制，导致硬件加速器处于饥饿状态。在这项工作中，我们引入了一种可扩展的软课程学习方法（摘要原文在此处被截断）。

    arXiv:2609.35783v1 Announce Type: cross  Abstract: Large-scale recommender systems, particularly short-form video platforms, are often bottlenecked by massive popularity feedback loops. In such environments, as models recommend popular items, they generate an overwhelming amount of skewed training data for "head" items. This creates a self-reinforcing cycle where retrieval and ranking models memorize "head" item patterns at the expense of generalizing across the vast "tail" of the catalogue. While Curriculum Learning (CL) offers a powerful mechanism to break this feedback loop by systematically exposing models to progressively more difficult and less frequent examples, its adoption in industrial recommendation has been hampered by hardware utilization inefficiencies or the needs for complicated pre-processing techniques because dynamic data rejection algorithms tend to starve hardware accelearators (TPUs/GPUs) by becoming largely CPU-bound. In this work, we introduce a scalable Soft Cu
    
[^26]: 金融证据拥挤：诊断与缓解检索增强生成中由约束引起的排挤现象

    Financial Evidence Crowding: Diagnosing and Mitigating Constraint-Induced Displacement in Retrieval-Augmented Generation

    [https://arxiv.org/abs/2609.35782](https://arxiv.org/abs/2609.35782)

    该论文揭示了金融问答RAG中的“金融证据拥挤”现象——主题匹配但时间期间、分部或口径冲突的候选证据会挤占有限的上下文槽位、排挤真正支持答案的证据（导致Recall@10下降0.147），并提出学习式分数修正方法FinDeCrowd-RAG来缓解这一问题。

    

    检索增强生成（RAG）会检索候选证据，并仅将排名靠前的有限子集（即top-k上下文）发送给生成器。在金融问答中，候选段落可能在主题上与查询匹配，但在时间期间、业务分部、指标口径或表格范围上与查询存在冲突。我们研究了由此导致的集合级排序失败现象，并将其命名为“金融证据拥挤”。FinDeCrowd-Stress通过构建相互匹配的兼容与不兼容候选池，并固定查询、相关证据、排序模型、候选数量和检索预算，从而隔离出这一失败现象。在包含训练期间未见公司的FinDER测试集上，不兼容候选池相对于同等难度的兼容候选池使top-10证据纳入率（Recall@10）降低了0.147。这一差距表明，冲突的候选证据会占用有限的上下文槽位，从而排挤掉真正支持答案的证据。随后，我们提出了FinDeCrowd-RAG，一种学习式的分数修正方法……

    arXiv:2609.35782v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) retrieves candidate evidence and sends only a limited top-ranked subset, the top-k context, to a generator. In financial question answering, passages can match a query's topic while conflicting with its period, segment, metric scope, or table scope. We study the resulting set-level ordering failure, which we call financial evidence crowding. FinDeCrowd-Stress isolates this failure through matched compatible and incompatible candidate pools while fixing the query, relevant evidence, ranking model, candidate count, and retrieval budget. On a FinDER test split containing companies unseen during training, incompatible pools reduce top-10 evidence inclusion (Recall@10) by 0.147 relative to equally difficult compatible pools. This gap shows that conflicting candidates consume limited context slots and displace answer-supporting evidence. We then introduce FinDeCrowd-RAG, a learned score correction that comb
    
[^27]: TSG Suggester：基于树结构知识图谱检索的云事件管理故障排除指南推荐

    TSG Suggester: Tree-Structured Knowledge-Graph Retrieval for Troubleshooting Guide Recommendation in Cloud Incident Management

    [https://arxiv.org/abs/2609.35780](https://arxiv.org/abs/2609.35780)

    该论文提出TSG Suggester系统，通过将故障排除指南转换为保留章节层级的树结构并结合知识图谱（Tree+KG），在大语言模型生成的问题抽象的帮助下桥接指南与事件描述之间的语言差异，从而直接从事件描述中高效推荐相关的故障排除指南。

    

    大规模云服务的值班工程师在巨大的时间压力下工作，然而为事件定位正确的故障排除指南仍然是一个主要依靠人工、以关键词为驱动的过程，此前的实证研究发现指南搜索消耗了总故障缓解时间的相当大一部分。我们提出了TSG Suggester，这是一个能够直接从事件描述中推荐相关故障排除指南的检索系统。我们在314个真实世界事件上评估了五种检索策略：纯文本RAG、图像增强RAG、RAPTOR、树结构检索，以及我们提出的树+知识图谱方法（Tree+KG），这些事件涵盖了来自生产事件管理系统中18个服务团队的112个不同的故障排除指南。Tree+KG将每个指南转换为一棵保留其原生章节层级的树，并在内部节点上附加由大语言模型生成的问题抽象，以桥接指南中面向解决方案的语言与事件中面向问题的语言之间的差异。

    arXiv:2609.35780v1 Announce Type: new  Abstract: On call engineers in large scale cloud services work under intense time pressure, yet locating the correct Troubleshooting Guide (TSG) for an incident remains a largely manual, keyword driven process, and prior empirical work finds that guide search consumes a substantial fraction of total mitigation time.   We present TSG Suggester, a retrieval system that recommends relevant TSGs directly from an incident description. We evaluate five retrieval strategies: Text Only RAG, Image Augmented RAG, RAPTOR, Tree Structured Retrieval, and our proposed Tree + Knowledge Graph (Tree+KG), on 314 real world incidents spanning 112 unique TSGs drawn from 18 service teams on a production incident management system.   Tree+KG converts each guide into a tree that preserves its native section hierarchy, attaches LLM generated problem abstractions to internal nodes to bridge the solution oriented language of guides and the problem oriented language of inci
    
[^28]: 后生成验证主导检索优化：RAG流水线特征的2^4全因子消融研究

    Post-Generation Verification Dominates Retrieval Optimization: A 2^4 Factorial Ablation of RAG Pipeline Features

    [https://arxiv.org/abs/2609.35774](https://arxiv.org/abs/2609.35774)

    通过2^4全因子消融实验发现，RAG流水线中后生成验证（完整性检查）是主导性特征，仅其单独使用即优于所有其他特征组合，而目录引导检索可零成本显著提升效果。

    

    现代RAG（检索增强生成）流水线堆叠了许多增强特征，但这些特征通常被孤立验证，其相互作用从未被测量。我们对四个流水线特征——章节扩展（SE）、智能体搜索（AS）、完整性检查（CC）和目录引导检索（ToC）——进行了2^4全因子消融实验，涵盖16种配置、跨越八种交互类型的24个查询、两个云级模型（共768个条件），在五份公开文档（78-492页）上进行，并将每个答案与经过验证的参考答案进行对比评分。结果表明后生成验证占主导地位：CC是最强的特征（d=+0.48，p<0.001），能同时提高准确性、完整性和有用性，且仅使用CC（4.31/5）即优于所有不含CC的配置，包括三特征组合SE+AS+ToC（4.11）。ToC在零LLM成本下带来显著增益（d=+0.22）；AS效果微小且不稳定，对某些查询有帮助而对其他查询有害；SE则为中性。

    arXiv:2609.35774v1 Announce Type: new  Abstract: Modern RAG pipelines stack many enhancement features, but these features are typically validated in isolation, leaving their interactions unmeasured. We run a 2^4 full factorial ablation of four pipeline features -- section expansion (SE), agentic search (AS), completeness check (CC), and table-of-contents-guided retrieval (ToC) -- across 16 configurations, 24 queries spanning eight interaction types, and two cloud-class models (768 conditions) on five public documents (78-492 pages), scoring every answer against a verified reference. Post-generation verification dominates: CC is the strongest feature (d=+0.48, p<0.001), improving accuracy, completeness, and usefulness simultaneously, and CC alone (4.31/5) outperforms every configuration without it, including the three-feature SE+AS+ToC (4.11). ToC yields a significant gain at zero LLM cost (d=+0.22); AS is small and unstable, helping some queries and harming others; SE is neutral. The h
    
[^29]: 苏格拉底-RAG：针对协同证据投毒的前提导向式探询

    Socrates-RAG: Premise-Directed Inquiry against Coordinated Evidence Poisoning

    [https://arxiv.org/abs/2609.35773](https://arxiv.org/abs/2609.35773)

    提出Socrates-RAG，一种通过主动探询并解决关键未决前提来对抗协同证据投毒的前提导向主动检索策略，并在有限查询预算下提供了相对于常规查询策略的条件性挽救保证。

    

    检索增强生成（RAG）的防御方法通常决定如何对固定的检索结果集合进行过滤或聚合。然而，在开放语料库问答中，决定性证据可能并不在初始上下文中，但却是可以检索到的，这使得下一次查询本身成为可靠性问题的一部分。我们提出了Socrates-RAG，这是一种前提导向的主动检索策略，它表示相互竞争的候选答案，选择一个尚未解决的前提——对该前提的解决将能够区分这些答案——并利用新获得的证据在回答或弃答之前细化后续查询。我们对该策略产生的有限预算证据状态进行了形式化，并相对于重复查询或主题查询策略给出了条件性挽救保证。我们将Socrates-RAG与一个匹配的对照组进行比较评估，该对照组由相同的骨干模型生成普通的面向相关性的搜索查询；两种策略共享相同的初始证据、确定性检索器、两次查询/前三名预算、回答提示……

    arXiv:2609.35773v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) defenses typically decide how to filter or aggregate a fixed retrieved set. In open-corpus question answering, however, decisive evidence may be absent from the initial context but retrievable, making the next query part of the reliability problem. We introduce Socrates-RAG, a premise-directed active retrieval policy that represents competing answers, selects an unresolved premise whose resolution would discriminate them, and uses newly acquired evidence to refine a subsequent query before answering or abstaining. We formalize the resulting finite-budget evidence state and give a conditional rescue guarantee relative to repeated or topical-query policies.   We evaluate Socrates-RAG against a matched control in which the same backbone generates ordinary relevance-oriented search queries; both policies share the initial evidence, deterministic retriever, two-query/top-three budget, answer prompt, and la
    
[^30]: AX 是新的 AEO

    AX is the New AEO

    [https://arxiv.org/abs/2609.34951](https://arxiv.org/abs/2609.34951)

    该论文提出“代理体验（AX）”——即AI智能体能否顺利抓取并阅读企业自身网站——正在取代答案引擎优化（AEO）成为决定AI推荐结果的关键因素，并通过超过3.7万次智能体买家旅程实验加以验证。

    

    2023年，AI模型依靠训练数据回答问题，数据耗尽时便产生幻觉，因此企业被告知要预先植入相关知识。此后，模型的训练知识已让位于实时网络搜索，相关建议也随之转移：答案引擎优化（AEO）如今建议企业在论坛帖子、清单文章和站外引用中散布“面包屑”信息，以便AI引擎更容易将它们展示和推荐出来。但仅仅被展示出来已经不够了：智能体会打开搜索结果、阅读内容后才做决策，一个买家问题会驱使它经历多轮搜索与抓取。在这一深入挖掘环节中，决定成败的是智能体能否成功抓取并阅读企业自己的网站——这就是代理体验（AX）。我们主张，AX 是新的 AEO。我们在四个独立测试框架上，针对1,056家真实企业开展了37,927次智能体旅程实验，每次旅程都是一个关于某企业的买家问题，并在知名度、模型先验知识以及（原文在此截断）……等方面进行了匹配。

    arXiv:2609.34951v2 Announce Type: replace  Abstract: In 2023, AI models answered from training data and hallucinated when it ran out, and businesses were told to seed that knowledge. Models' training knowledge has since given way to live web search, and the advice followed it there: answer-engine optimization, or AEO, now tells businesses to scatter breadcrumbs across forum threads, listicles, and off-site citations, so AI engines are likelier to surface and recommend them. But being surfaced is no longer enough: an agent opens the results and reads them before deciding, and one buyer question sends it through several rounds of search and fetch. What decides the outcome at this drill-down step is whether the agent can fetch and read the business's own site: agent experience (AX). We argue that AX is the new AEO. We run 37,927 agent journeys, each a buyer question about a business, across four independent harnesses over 1,056 real businesses, matched on fame, prior model knowledge, and 
    
[^31]: IROH：基于多阶段混合检索与原理蒸馏LLM裁判的幽默洞察排序——面向JOKER 2026赛道任务1（英语）

    IROH: Insightful Ranking Of Humor using Multi-Stage Hybrid Retrieval with Rationale-Distilled LLM Judges for JOKER 2026 Track Task 1 English

    [https://arxiv.org/abs/2609.15618](https://arxiv.org/abs/2609.15618)

    该论文提出IROH三阶段检索系统，通过稀疏-稠密混合检索、交叉编码器重排序与原理蒸馏的LoRA大语言模型裁判集成，在JOKER 2026任务1幽默排序中夺得第一名（MAP 0.6347），并发现原理蒸馏裁判是排序质量的关键驱动因素。

    

    我们的团队VANGUARD提出了IROH（Insightful Ranking of Humor，幽默洞察排序），这是一个面向CLEF 2026 JOKER任务1（英语）的三阶段检索系统，以0.6347的MAP成绩位列排行榜第一名。我们的处理流程结合了稀疏-稠密混合检索、交叉编码器重排序，以及经LoRA适配的大语言模型裁判集成。我们使用Gemma 4在两种提示策略（通用型和类型化）下生成查询感知的原理说明，并生成多达四种类型的结构化困难负例用于训练数据构建。通过对三种交叉编码器架构、四种稠密嵌入模型和八种裁判配置的消融实验，我们得到三项关键发现：（1）原理蒸馏的裁判是排序质量的主要驱动因素，而将原理说明附加到第一阶段索引中的贡献微乎其微；（2）结构化困难负例虽然在本地验证集上会抬高分数，但在几乎所有配置中都会损害泛化能力；以及……

    arXiv:2609.15618v1 Announce Type: cross  Abstract: Our team, VANGUARD, presents IROH (Insightful Ranking of Humor), a three-stage retrieval system for JOKER Task 1 English at CLEF 2026, achieving first place on the leaderboard with 0.6347 MAP. Our pipeline combines hybrid sparse-dense retrieval, cross-encoder reranking, and a LoRA-adapted Large Language Model judge ensemble. We employ Gemma 4 to generate query-aware rationales under two prompt strategies, generic and typed, and produce up to four types of structured hard negatives for training data construction. Through an ablation across three cross-encoder architectures, four dense embedders, and eight judge configurations, our key findings are threefold: (1) the rationale-distilled judge is the primary driver of ranking quality, whereas appending rationales to the first-stage index contributes negligibly; (2) structured hard negatives degrade generalisation in nearly all configurations despite inflating local validation scores; and 
    
[^32]: RAISE：诊断昂贵LLM信号中的获取崩溃

    RAISE: Diagnosing Acquisition Collapse in Costly LLM Signals

    [https://arxiv.org/abs/2608.10441](https://arxiv.org/abs/2608.10441)

    本文识别出“获取崩溃”这一失败模式，并提出了RAISE预路由诊断框架，帮助判断昂贵的LLM信号在何时才值得调用，从而避免盲目调用造成的资源浪费。

    

    大型语言模型（LLM）日益被用作真实系统中昂贵的按需组件，但无差别地调用它们可能会浪费大量的计算资源、延迟时间和服务预算。因此，关键的部署问题不仅在于LLM平均而言是否有帮助，更在于何时值得调用它。我们识别出一种常见的失败模式，称之为“获取崩溃”：一个LLM信号可能在总体上或事后看来是有用的，但在调用之前所能提供的信息却太少，无法支持可靠的选择性使用。我们提出了RAISE（Reward-SNR Actionability in Signal Evaluation，信号评估中的奖励信噪比可行动性），这是一个预路由诊断框架，用于在确定路由策略之前检验现有证据是否支持选择性使用。我们通过结构化假设嵌入来实现RAISE，这是一种用于推荐的冻结LLM意图信号，每个用户仅需一次LLM调用，并通过受控实验、回顾性研究和新用户队列研究对其进行评估。

    arXiv:2608.10441v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) are increasingly used as costly, on-demand components in real systems, but calling them indiscriminately can waste substantial compute, latency, and serving budget. The key deployment question is therefore not only whether an LLM helps on average, but when it is worth calling. We identify a common failure mode, which we call acquisition collapse: an LLM signal can appear useful in aggregate or post hoc, yet still provide too little before-call information to support reliable selective use. We introduce RAISE (Reward-SNR Actionability in Signal Evaluation), a pre-routing diagnostic framework for testing whether available evidence supports selective use before committing to a routing strategy. We instantiate RAISE with Structured Hypothesis Embeddings (SHE), a frozen-LLM intent signal for recommendation using one LLM call per user, and evaluate it through controlled, retrospective, and fresh-cohort st
    
[^33]: OneLatent：面向高效基础推荐模型的潜在推理

    OneLatent: Latent Reasoning for Efficient Foundation Recommendation Models

    [https://arxiv.org/abs/2607.26621](https://arxiv.org/abs/2607.26621)

    OneLatent提出将显式思维链推理压缩为可学习的潜在token，实现“先潜在推理后回答”的高效推荐框架，通过多视角自适应CoT生成高质量监督信号，在捕捉多样化用户兴趣的同时大幅降低推理开销。

    

    大型语言模型（LLMs）展现了强大的推理能力，这促使它们被用作基础推荐模型（FRMs）的主干。现有方法通过在“先思考后回答”范式下进行显式思维链（CoT）推理来增强推荐效果。然而，显式CoT会生成冗长的推理轨迹，带来巨大的推理开销，并且依赖手动设计的模板，难以捕捉多样且动态变化的用户兴趣。我们提出OneLatent，一个高效的潜在推理框架，它将显式推理轨迹压缩为若干可学习的潜在token，实现“先潜在推理后回答”的推理方式，无需生成冗长的推理轨迹。OneLatent首先引入多视角自适应CoT（MV-ACoT），通过从多个视角探索用户兴趣并自动调整推理复杂度，生成多样化、高质量的教师监督信号。

    arXiv:2607.26621v3 Announce Type: replace-cross  Abstract: Large language models (LLMs) have demonstrated strong reasoning capabilities, motivating their use as the backbone of foundation recommendation models (FRMs). Existing methods enhance recommendations through explicit Chain-of-Thought (CoT) reasoning under a Think-then-Answer paradigm. However, explicit CoT incurs substantial inference overhead by generating lengthy reasoning traces and relies on manually designed templates that struggle to capture diverse, dynamic user interests. We propose OneLatent, an efficient latent reasoning framework that compresses explicit reasoning traces into several learnable latent tokens, enabling Latent-Reason-then-Answer inference without generating verbose traces. OneLatent first introduces Multi-View Adaptive CoT (MV-ACoT), which creates diverse, high-quality teacher-generated supervision by exploring user interests from multiple perspectives and automatically adapting reasoning complexity to 
    
[^34]: 解锁大型视听检索模型中的空间定位能力

    Unlocking Spatial Grounding in Large Audio-Visual Retrieval models

    [https://arxiv.org/abs/2607.24786](https://arxiv.org/abs/2607.24786)

    该研究发现大规模视听检索模型的中间层视觉表示蕴含高度结构化的空间信息，并提出LAIP框架，通过轻量级的音频信息空间池化（AiSP）替代全局聚合，在弱监督条件下实现了细粒度的声源空间定位。

    

    arXiv:2607.24786v2 公告类型：replace-cross 摘要：由于密集的空间标注在大规模场景下获取成本高昂，弱监督为视听声源定位任务设定了一个实用的研究范式。然而，该任务仍然具有挑战性，因为模型必须在没有像素级监督的情况下，从时间对齐的视听数据中定位声源。近期的大规模视听检索模型以前所未有的规模进行训练，编码了丰富的多模态结构。我们证明，这些模型的潜在表示虽然是为全局对齐而优化的，但仍然能够实现细粒度的空间定位。尽管由于全局池化的作用，空间细节在检索骨干网络的上层逐渐丢失，但中间层的视觉标记保留了高度结构化的空间信息。为了利用这一点，我们提出了LAIP（通过音频信息池化进行定位），该框架采用轻量级的“音频信息空间池化”来替代标准的全局聚合模块……

    arXiv:2607.24786v2 Announce Type: replace-cross  Abstract: Weak supervision sets a practical regime for audio-visual sound source localization as dense spatial annotations are costly to obtain at scale. The task, however, remains challenging, as models must locate sound sources from temporally aligned audio-visual data without pixel-level supervision. Recent large-scale audio-visual retrieval models, trained at unprecedented scale, encode rich multimodal structure. We show their latent representations, though optimized for global alignment, can nonetheless enable fine-grained spatial grounding. While spatial detail is progressively lost in the upper layers of retrieval backbones due to global pooling, intermediate visual tokens retain highly structured spatial information. To exploit this, we introduce LAIP (\emph{Localization via Audio-Informed Pooling}), a framework that employs a lightweight \emph{Audio-informed Spatial Pooling} (AiSP) to replace the standard global aggregation modu
    
[^35]: SIREN（将LLM引向险境）：Web-RAG推荐系统中的PAIR驱动偏好操纵

    SIREN (Luring LLMs onto the Rocks): PAIR-Driven Preference Manipulation in Web-RAG Recommenders

    [https://arxiv.org/abs/2607.21951](https://arxiv.org/abs/2607.21951)

    本文提出SIREN方法，利用PAIR越狱循环和23种可解释编辑策略，在Web-RAG推荐系统中操纵LLM的排名输出，使目标实体升至首位。

    

    本文研究了针对网页增强型大型语言模型（LLM）生成的排名推荐结果进行对抗性操纵的问题。当LLM通过检索并阅读实时网页来回答推荐查询时，它扮演了推荐系统的角色，而每个检索到的页面都可能成为攻击面。先前的研究探讨了虚构产品、检索投毒和排名提升等问题，但这些研究并未比较在周围源集不变的情况下，对已检索页面的不同编辑如何改变模型的最终排名。为填补这一空白，我们提出了SIREN，一种自动化攻击者-评判者方法，它将PAIR越狱循环适应于竞争性排名操纵，目标是使选定实体在LLM生成的推荐中升至第1位。SIREN使用Anthropic的网页工具检索并捕获网页，然后通过23种可解释的编辑分类法迭代修改检索到的源。

    arXiv:2607.21951v2 Announce Type: replace  Abstract: This paper investigates the adversarial manipulation of the ranked recommendations produced by web-augmented large language models (LLMs). When an LLM answers a recommendation query by retrieving and reading live webpages, it acts as a recommender, and each retrieved page becomes a potential attack surface. Prior work has examined fabricated products, retrieval poisoning, and rank promotion. However, these studies do not compare how different edits to an already retrieved page change the model's final ranking while the surrounding source set remains unchanged. To address this gap, we propose SIREN, an automated attacker--judge method that adapts the PAIR jailbreaking loop to competitive rank manipulation, with the goal of moving a chosen entity to rank~1 in an LLM-generated recommendation. SIREN retrieves and captures webpages using Anthropic's web tools, then iteratively edits a retrieved source using an interpretable taxonomy of 23
    
[^36]: 《智能体人工智能漫游指南：从基础到系统》

    The Hitchhiker's Guide to Agentic AI: From Foundations to Systems

    [https://arxiv.org/abs/2606.24937](https://arxiv.org/abs/2606.24937)

    该论文（著作）是一部从大语言模型基础、对齐与推理技术到智能体AI全栈覆盖的构建自主AI系统综合实践指南，其核心观点是：构建优秀的智能体系统必须理解技术栈的每一个层级。

    

    《智能体人工智能漫游指南》是一本面向从业者的综合性参考书，旨在指导构建自主人工智能系统，内容涵盖从第一性原理到生产部署的全栈知识。本书的核心论点是：构建优秀的智能体系统需要理解技术流水线的每一层，而不仅仅是其中某一层。本书开篇介绍大语言模型（LLM）基础层，涵盖Transformer架构、GPU系统、训练与微调（SFT、LoRA、MoE）、模型压缩以及推理优化等必备基础。随后阐述对齐与推理层：RLHF、PPO、DPO及其变体、GRPO、奖励建模，以及面向大型推理模型的强化学习，包括思维链和测试时扩展（test-time scaling）。本书后半部分专门聚焦智能体AI本身：智能体训练与基于轨迹的强化学习、RAG与智能体RAG、记忆系统（上下文内记忆、外部记忆、情景记忆与语义记忆）、智能体框架设计、循环工程、基于图的编排等。

    arXiv:2606.24937v3 Announce Type: replace  Abstract: The Hitchhiker's Guide to Agentic AI is a comprehensive practitioner's reference for building autonomous AI systems, covering the full stack from first principles to production deployment. The central thesis: building great agentic systems requires understanding every layer of the pipeline, not just one. The book opens with the LLM substrate, covering transformer architecture, GPU systems, training and fine-tuning (SFT, LoRA, MoE), model compression, and inference optimization, as essential foundations. It then develops the alignment and reasoning layer: RLHF, PPO, DPO and its variants, GRPO, reward modeling, and RL for large reasoning models including chain-of-thought and test-time scaling. The second half is devoted to agentic AI proper: agentic training and trajectory-based RL, RAG and Agentic RAG, memory systems (in-context, external, episodic, and semantic), agent harness design, loop engineering, graph-based orchestration, and 
    
[^37]: 用于可审计商业登记分析的智能体图检索增强生成

    Agentic Graph Retrieval-Augmented Generation for Auditable Commercial Registry Analysis

    [https://arxiv.org/abs/2605.18770](https://arxiv.org/abs/2605.18770)

    本文提出一种受控的智能体GraphRAG架构，将瑞士官方商业公报构建为包含500余万节点和470万关系的Neo4j知识图谱，并通过意图路由、受限图工具、有界反思和状态机引导实现可审计的商业登记簿自然语言分析。

    

    公共商业登记簿在形式上是公开的，但其实际分析仍然困难，因为相关事实分散在数百万条记录中，这些记录混合了结构化元数据、多语言法律公告、时间性事件以及实体别名。本文提出了一种受控的、以工具为媒介的智能体GraphRAG架构，用于对此类登记簿进行可审计的自然语言分析。所提出的处理流水线将瑞士官方商业公报的出版物转换为包含超过500万个节点和470万个关系的Neo4j知识图谱。该系统结合了结构化登记字段的确定性摄取、LLM辅助的非结构化公告中潜在行为者的提取，以及确定性的身份解析层。一个分析智能体通过意图路由、受限图工具、有界反思和状态机引导的响应合成在该知识图谱上执行操作。我们对系统进行了评估（摘要在此处截断）。

    arXiv:2605.18770v3 Announce Type: replace-cross  Abstract: Public commercial registries are formally open, yet their practical analysis remains difficult because relevant facts are scattered across millions of records that combine structured metadata, multilingual legal notices, temporal events, and entity aliases. This paper presents a controlled, tool-mediated agentic GraphRAG architecture for auditable natural-language analysis of such registries. The proposed pipeline transforms publications from the Swiss Official Gazette of Commerce into a Neo4j knowledge graph comprising over five million nodes and 4.7 million relationships. It combines deterministic ingestion of structured registry fields, LLM-assisted extraction of latent actors from unstructured notices, and a deterministic identity-resolution layer. An analytical agent operates on this graph through intent routing, restricted graph tools, bounded reflection, and state-machine-guided response synthesis. We evaluate the system
    
[^38]: UltRAG：一种通用、简单、可扩展的知识图谱RAG方案

    UltRAG: a Universal Simple Scalable Recipe for Knowledge Graph RAG

    [https://arxiv.org/abs/2603.28773](https://arxiv.org/abs/2603.28773)

    UltRAG是一种无需训练的知识图谱RAG方案，通过结合LLM查询生成、完全归纳式神经查询执行器和LLM仲裁，在无需重训模型的情况下于KGQA任务上取得最先进结果，并支持Wikidata规模的超大规模图谱。

    

    大型语言模型（LLM）在用于语言生成时，经常生成看似自信但事实上不正确的内容（这种现象通常被称为“幻觉”）。检索增强生成（RAG）试图通过在知识语料库中检索信息并将其置于模型上下文窗口中，来减少事实性错误。虽然这种方法在文档结构化数据上已经相当成熟，但要将其适配到知识图谱（KG）上并不容易，尤其是对于那些需要在图上进行多节点/多跳推理的查询。我们提出了UltRAG，这是一种无需训练的KG-RAG方案，它结合了LLM查询生成、完全归纳式（inductive）的神经查询执行器以及LLM仲裁。这种开箱即用的组合在知识图谱问答（KGQA）任务上取得了最先进的结果，而无需重新训练LLM或执行器，同时使语言模型能够与Wikidata规模的图谱（1.16亿实体、16亿……）进行交互。

    arXiv:2603.28773v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) frequently generate confident yet factually incorrect content when used for language generation (a phenomenon often known as hallucination). Retrieval augmented generation (RAG) tries to reduce factual errors by identifying information in a knowledge corpus and putting it in the context window of the model. While this approach is well-established for document-structured data, it is non-trivial to adapt it for Knowledge Graphs (KGs), especially for queries that require multi-node/multi-hop reasoning on graphs. We introduce UltRAG, a training-free KG-RAG recipe that combines LLM query generation, a fully inductive neural query executor, and LLM arbitration. This off-the-shelf composition achieves state-of-the-art results on Knowledge Graph Question Answering (KGQA) tasks without retraining the LLM or executor, while enabling language models to interface with Wikidata-scale graphs (116M entities, 1.6B 
    
[^39]: 面向时序表格推理的证据引导式模式规范化

    Evidence-Guided Schema Normalization for Temporal Tabular Reasoning

    [https://arxiv.org/abs/2512.00329](https://arxiv.org/abs/2512.00329)

    该研究将时序表格问答重构为自动化知识库构建任务，并通过受控交叉实验证明模式设计质量对问答准确率的影响远大于查询模型的选择（分别解释79.5%和1.6%的EM方差），据此提炼出模式设计原则。

    

    arXiv:2512.00329v2 公告类型：replace-cross 摘要：在不断演化的半结构化表格上进行时序推理对当前的问答系统构成了挑战。我们提出一种方法，将该任务重新构建为自动化知识库构建：（1）通过提示大语言模型（LLM）从维基百科信息框时间线中合成符合第三范式（3NF）的关系模式，（2）填充该模式以获得可查询的数据库，（3）生成并执行针对该数据库的SQL查询，并以问答准确率作为所构建知识库的外在评估手段。在一个由三种模式生成器与六种查询模型交叉组合的受控实验网格中，模式来源解释了精确匹配（EM）方差的79.5%，而查询模型仅解释1.6%：替换模式及其派生的提示框架会使EM偏移14.7至20.0个百分点，而在固定模式下替换查询模型仅使EM偏移4.4至12.1个百分点。基于这些证据，我们提炼出三条候选的模式设计原则：平衡规范化……

    arXiv:2512.00329v2 Announce Type: replace-cross  Abstract: Temporal reasoning over evolving semi-structured tables poses a challenge to current QA systems. We propose an approach that recasts the task as automated knowledge base construction: (1) prompting an LLM to synthesize a 3NF-compliant relational schema from Wikipedia infobox timelines, (2) populating the schema to obtain a queryable database, and (3) generating and executing SQL queries against it, with QA accuracy serving as an extrinsic evaluation of the constructed knowledge base. In a controlled grid of three schema generators crossed with six query models, the schema source accounts for 79.5% of the exact match (EM) variance against 1.6% for the query model: replacing the schema, and the prompt scaffolding derived from it, shifts EM by 14.7 to 20.0 points, whereas replacing the query model under a fixed schema shifts it by 4.4 to 12.1. From this evidence, we distill three candidate schema-design principles: balanced normal
    
[^40]: RecKG：面向推荐系统的知识图谱

    RecKG: Knowledge Graph for Recommender Systems

    [https://arxiv.org/abs/2501.03598](https://arxiv.org/abs/2501.03598)

    本文提出 RecKG——一个面向推荐系统的标准化知识图谱，通过统一的实体表示与命名规范实现异构数据源的无缝集成，从而挖掘出更多的语义信息。

    

    知识图谱已在整合各领域异构数据方面被证明是成功的。然而，尽管基于知识图谱的推荐系统已获得广泛的研究关注，但关于异构推荐系统之间无缝集成的研究仍然明显不足。本研究旨在通过提出 RecKG——一个面向推荐系统的标准化知识图谱——来填补这一空白。RecKG 确保了不同数据集之间实体的一致表示，并能兼容多样的属性类型以实现有效的数据整合。通过对多个推荐系统数据集的细致考察，我们为 RecKG 选取了属性，并通过一致的命名规范确保了标准化格式。凭借这些特性，RecKG 能够无缝集成异构数据源，从而在集成后的知识图谱中发现额外的语义信息。

    arXiv:2501.03598v2 Announce Type: replace-cross  Abstract: Knowledge graphs have proven successful in integrating heterogeneous data across various domains. However, there remains a noticeable dearth of research on their seamless integration among heterogeneous recommender systems, despite knowledge graph-based recommender systems garnering extensive research attention. This study aims to fill this gap by proposing RecKG, a standardized knowledge graph for recommender systems. RecKG ensures the consistent representation of entities across different datasets, accommodating diverse attribute types for effective data integration. Through a meticulous examination of various recommender system datasets, we select attributes for RecKG, ensuring standardized formatting through consistent naming conventions. By these characteristics, RecKG can seamlessly integrate heterogeneous data sources, enabling the discovery of additional semantic information within the integrated knowledge graph. We app
    
[^41]: BadRAG：识别大语言模型检索增强生成中的漏洞

    BadRAG: Identifying Vulnerabilities in Retrieval Augmented Generation of Large Language Models

    [https://arxiv.org/abs/2406.00083](https://arxiv.org/abs/2406.00083)

    BadRAG揭示了一种针对检索增强生成系统的新型攻击：攻击者通过向知识库注入恶意文段，当用户查询包含特定触发词时即可操纵系统响应，且无需修改用户输入或模型权重。

    

    检索增强生成（RAG）通过从外部知识库检索相关信息来增强大语言模型（LLM），从而提供更准确、更具上下文感知且更加及时的回答。然而，这种对外部知识的依赖引入了重大的安全漏洞，因为许多RAG系统（例如Google搜索）依赖于庞大且未经清洗的数据存储库（例如Reddit）。在本文中，我们揭示了一种新型威胁：攻击者通过向RAG系统的知识库注入恶意文段来操纵系统的响应。当用户的查询包含攻击者指定的触发词时，RAG会检索并引用这些恶意文段，使攻击者能够在不修改用户输入或RAG权重的情况下操纵响应。BadRAG分为两个阶段：（i）对恶意文段进行优化，使其仅在用户查询中出现触发词时才会被检索；（ii）这些恶意文段……

    arXiv:2406.00083v3 Announce Type: replace-cross  Abstract: Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by retrieving relevant information from external knowledge bases to provide more accurate, contextually informed, and up-to-date responses. However, this reliance on external knowledge introduces significant security vulnerabilities, as many RAG systems (e.g., Google Search) rely on large and unsanitized data repositories (e.g., Reddit). In this paper, we unveil a novel threat in which attackers steer the RAG system's response by injecting malicious passages into its knowledge base. When a user's query contains attacker-specified trigger words, the RAG retrieves and refers to these malicious passages, enabling the attacker to steer the response without altering the user input or modifying the RAG weights. BadRAG operates in two phases: (i) malicious passages are optimized to be retrieved exclusively when trigger words appear in user queries; (ii) these p
    

