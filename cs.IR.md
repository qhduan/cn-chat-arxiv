# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Task-Centric Survey of Long-Document Visual Question Answering: Taxonomy, Evolution, and Future Directions](https://arxiv.org/abs/2610.31385) | 该论文提出首个以任务为中心的长文档视觉问答综述，将其分解为文档解析、证据组织与检索、生成器组织与学习及评估四个组件，建立细粒度分类体系，并通过时间维度分析梳理研究重点从多模态内容理解到证据获取与管理的演进脉络及未来方向。 |
| [^2] | [SurveyGraph: Generating Academic Surveys via Methods-Entity Knowledge Graphs](https://arxiv.org/abs/2610.31317) | SurveyGraph通过构建编码方法级后续、竞争与资源共享关系的“方法-实体知识图谱”（MEKG）来生成学术综述，使图感知检索的召回率相比扁平文本基线近乎翻倍，并显著提升了生成内容的引用支持率。 |
| [^3] | [The Delayed Countdown Method for Validating Review-Oriented AI Workflows in eDiscovery](https://arxiv.org/abs/2610.31311) | 本文提出延迟倒计时方法，为电子取证中审查导向型AI工作流的有效性验证提供了一种简单实用、易于集成、兼容任意停止规则并能给出有效置信区间的评估方案。 |
| [^4] | [Chunk Recomputation under Real Document Revisions: Measuring Update Cost and Retrieval Quality of Fixed, Recursive, and Content-Defined Chunking](https://arxiv.org/abs/2610.31019) | 该论文首次在真实文档持续修订的场景下量化比较了固定大小、递归和内容定义三种分块策略的更新成本与检索质量，发现递归分块的重计算开销最低（38-56%），而三者在检索召回率上的差异很小。 |
| [^5] | [Formal-Informal Mathematical Alignment through and beyond Human Annotations](https://arxiv.org/abs/2610.31011) | 本文构建了将 Stacks Project 的非形式化数学陈述与 Lean mathlib 形式化内容相连接的数据集，并利用检索、LLM 重排序和 LLM 辅助验证等方法，在人工标注之外自动发现新的形式化与非形式化数学之间的对应关系。 |
| [^6] | [PAIR: Preference-Aligned Intent Retrieval for Premium People Recommendation at LinkedIn](https://arxiv.org/abs/2610.30941) | LinkedIn提出PAIR检索框架，通过建模用户意图偏好而非个人资料相似度，解决了专业网络中跨角色互补匹配的人物推荐问题。 |
| [^7] | [UserArena: Benchmarking Interactive User Simulation for Long-Horizon Shopping Agents](https://arxiv.org/abs/2610.30932) | UserArena是一个包含6.4万余条真实用户会话轨迹的电商交互式基准，通过五级评估体系并对照生产环境A/B测试结果进行离线验证，用于开发和评估长周期购物智能体的推荐策略。 |
| [^8] | [Lean Finder V2: Semantic Search for Mathlib That Disambiguates User Queries](https://arxiv.org/abs/2610.30843) | Lean Finder V2 通过互补的模型级对抗训练与输入级查询重写，实现了能够消除歧义的 Mathlib 语义搜索，有效解决了短而模糊查询的检索难题。 |
| [^9] | [Bezier-Guided Adaptive Flow for Cross-domain Sequential Recommendation](https://arxiv.org/abs/2610.30825) | 提出BGAFlow，一个结合贝塞尔引导的状态自适应语义演化与用户级迁移控制的个性化跨域序列推荐框架，利用LLM推断的用户画像和基于兴趣重叠度的个性化迁移视界，实现对跨域知识迁移的精确调控。 |
| [^10] | [Automated Enrichment Analysis with GraphRAG: Efficient Multi-Database ORA Extraction and Biological Interpretation](https://arxiv.org/abs/2610.30710) | 提出GBOR-RAG框架，将多个异构基因集数据库统一为术语-基因知识图谱，通过检索证据子图条件化大语言模型生成可追溯的生物学富集解释，并引入声明级忠实度指标评估生成内容的证据支持。 |
| [^11] | [Lychee: Certified Screening of Inverted Lists for High-Recall Vector Search](https://arxiv.org/abs/2610.30697) | 提出Lychee方法，利用精确搜索的认证下界在无需读取向量的前提下筛除绝大多数被探测的向量，以远低于全精度评分的成本实现0.999以上召回率的高效最近邻搜索。 |
| [^12] | [OneLatent: Latent Reasoning for Efficient Foundation Recommendation Models](https://arxiv.org/abs/2610.30675) | OneLatent提出了一种潜在推理框架，将显式思维链推理压缩为可学习的潜在标记，并通过多视角自适应CoT生成高质量监督，从而在无需生成冗长推理轨迹的情况下实现高效的基础推荐模型。 |
| [^13] | [Compiling Autoregressive LLM Control into Policy-Exact Parallel Inference for Hierarchical RAG](https://arxiv.org/abs/2610.30664) | 该论文提出将分层RAG中自回归LLM的控制决策编译为策略精确的并行推理——只需一次根预填充加一次树因果探测，即可在两次前向计算的关键路径、零输出token的条件下，精确复现串行语法约束解码的动作分布。 |
| [^14] | [The Fusion Tax: A Diagnostic Analysis of When Hybrid Retrieval Hurts](https://arxiv.org/abs/2610.30612) | 该研究通过诊断分析发现，混合检索（如RRF融合）虽然在总体指标上优于纯稠密检索，但在8-48%的个体查询上反而损害检索效果，因此“不融合”应被视为值得认真对待的一等选项。 |
| [^15] | [Better Reasoning, Not Just Bigger Models: A Structure-Grounded Framework for Efficient and Reliable Inference](https://arxiv.org/abs/2610.30611) | 提出了无需训练的 SGUR 框架，通过将不完整观测表示为查询图，并利用先验加权的部分图匹配和主动信息增益策略在结构化假设空间中主动消减不确定性，从而提升多轮推理的可靠性与效率。 |
| [^16] | [Journal Basket: Supporting Cross-Disciplinary Literature Searching and Methodological Transparency](https://arxiv.org/abs/2610.30544) | 本文介绍了Journal Basket——一个帮助商科学生基于ABDC期刊质量列表构建期刊集合、生成多数据库检索式并进行跨学科文献检索的浏览器工具，同时提升文献综述的方法论透明度。 |
| [^17] | [Getting a GRIP on Graph-Based Search for Learned Sparse Retrieval](https://arxiv.org/abs/2610.30425) | 本文挑战了“图索引在学习型稀疏检索中劣于倒排索引”的传统观点，提出将倒排索引的优势融入图搜索，从而使基于图的ANN方法在LSR任务上也能达到有竞争力的性能。 |
| [^18] | [When Agentic Retrieval Fails Silently: Framework and Production Evidence from a Creative AI Assistant](https://arxiv.org/abs/2610.30423) | 该论文指出传统检索评估指标（CTR、MRR、行为NDCG）在智能体助手中因“归因鸿沟”而悄然失效，并提出无需新增建模的LARA框架，通过检索质量、参与与结果、产品成功三个独立层级以及四字段溯源模式，在生产环境Adobe Express中评估智能体式检索的真实效果。 |
| [^19] | [Data-Void Exploits in Web-Searching LLMs](https://arxiv.org/abs/2610.30365) | 该论文揭示了一种针对网络搜索型大语言模型的新攻击方式——“数据空洞”漏洞利用：虚假信息散布者在可靠内容稀缺的搜索空间中植入叙事内容，诱导LLM在回答提问时检索并引用这些不可靠来源，并通过实验考察了提示语气、模型选择和攻击者控制手段对攻击效果的影响。 |
| [^20] | [FAIRGate: Agentic Runtime Policy Selection and Verified Repair for Fairer Recommendation Delivery](https://arxiv.org/abs/2610.30359) | FAIRGate提出了一种模型无关的推荐分发控制器，通过智能体在运行时从预定义的公平策略库中动态选择合适策略，并借助确定性执行与多重约束验证的修复机制，实现更公平且可审计的推荐结果分发。 |
| [^21] | [Protection Is Not One Event: A Survey of RAG Protection Research by Measured Endpoint](https://arxiv.org/abs/2610.30193) | 该论文通过对118篇RAG防御论文按“实际度量终点”而非声称目标进行系统性分类综述，构建了七类分类体系，并揭示了防御机制与保护结果不对应、评估标准割裂等三大研究空白。 |
| [^22] | [Deep Search in Real-World Web Environments](https://arxiv.org/abs/2610.30181) | 该论文提出了 OpenDS 任务及 OpenDS-World 基准，用于评估智能体在真实、开放的互联网环境中自主进行多步骤深度搜索、跨多个独立来源发现并整合证据，从而得出唯一事实性答案的能力。 |
| [^23] | [QProtFuse: Query-Guided Layer Fusion for Retrieval-Augmented Protein Function Prediction](https://arxiv.org/abs/2610.30180) | QProtFuse利用查询引导自适应融合蛋白质语言模型的多层特征，并通过联合对比学习实现蛋白质与注释的协同检索，显著提升检索增强式蛋白质功能预测性能。 |
| [^24] | [Association between Player Perception and AI Disclosure on Steam: An Exploratory Study](https://arxiv.org/abs/2610.30178) | 该研究通过分析Steam评论发现，披露使用生成式AI的游戏在负面评论中会引来更多AI相关的批评，为理解玩家对AI披露的感知提供了首个探索性证据。 |
| [^25] | [FlexVDR: Learning to Allocate Matryoshka Representation Budgets for Visual Document Retrieval](https://arxiv.org/abs/2610.30177) | FlexVDR 提出了一个在单一视觉文档检索器中根据查询自适应分配套娃表示预算的框架，通过联合调节编码器深度、嵌入宽度和词元数量，并利用预算分配器预测各路由的检索效用，从而在计算开销与检索质量之间实现灵活权衡。 |
| [^26] | [FormBench: A Benchmark for Fine-Grained Retrieval in Formulation Patents](https://arxiv.org/abs/2610.30175) | 该论文提出了FormBench基准，用于评估模型从配方专利等技术文档中检索精确化学信息的能力，发现商业API编码器表现极差，而基于FormBench三元组进行任务自适应微调的开源编码器性能显著提升。 |
| [^27] | [IRAH: Hierarchical Agentic Tree Search for Retrieval Pipeline Optimization](https://arxiv.org/abs/2610.30167) | IRAH将UCT引导的分层智能体树搜索与离线元学习初始化相结合，在有限评估预算下联合优化RAG检索流程的结构组件与数值超参数，性能最高超越TPE达6.4%。 |
| [^28] | [Semantics Delivery Network: Rethinking Web Retrieval Infrastructure for LLM Agents](https://arxiv.org/abs/2609.22486) | 该论文提出“语义投递网络”，主张将Web检索基础设施从面向人类客户端的URL/摘要检索模式，重构为直接向LLM智能体提供面向任务效用的语义块检索服务，并支持跨轮次有状态检索与智能体间的工作共享。 |
| [^29] | [POI Recommendation with LLM-Augmented Multi-Graph Learning and Contrastive Alignment](https://arxiv.org/abs/2608.16407) | 本文提出LLM-MGCL，通过结合语义和地理信息的多图学习与对比对齐，有效解决了POI推荐中的冷启动问题。 |
| [^30] | [Structure then Query: Enabling Precise Analytical Queries over Unstructured Documents](https://arxiv.org/abs/2608.13384) | 本文提出了AnnoIndex系统，通过自动生成分层注释模式并提取结构化值，将非结构化文档转化为可查询的结构化索引，从而克服了传统向量模糊匹配的局限性，实现了精确的分析查询。 |
| [^31] | [OneLatent: Latent Reasoning for Efficient Foundation Recommendation Models](https://arxiv.org/abs/2607.26621) | OneLatent提出将显式思维链推理压缩为可学习的潜在token，实现“先潜在推理后回答”的高效推荐框架，通过多视角自适应CoT生成高质量监督信号，在捕捉多样化用户兴趣的同时大幅降低推理开销。 |
| [^32] | [Xetrieval: Explaining Dense Retrieval through Feature Elicitation](https://arxiv.org/abs/2605.29507) | 提出Xetrieval框架，通过特征引出器生成辅助视图使文档中与检索相关的语义更加显式，并借助共享稀疏字典以可解释的SAE特征表达查询-文档关系，从而解释稠密检索的相关性得分。 |
| [^33] | [A Memory-Efficient Distributed Algorithm for Approximate Nearest Neighbour Search with Arbitrary Distances](https://arxiv.org/abs/2405.13795) | 本文提出分布式近似最近邻搜索算法PDASC，其索引设计同时支持任意不相似性度量、原生分布式执行以及内存受限环境下的高效运行，且无需专用硬件。 |

# 详细

[^1]: 长文档视觉问答的任务中心综述：分类体系、演进与未来方向

    A Task-Centric Survey of Long-Document Visual Question Answering: Taxonomy, Evolution, and Future Directions

    [https://arxiv.org/abs/2610.31385](https://arxiv.org/abs/2610.31385)

    该论文提出首个以任务为中心的长文档视觉问答综述，将其分解为文档解析、证据组织与检索、生成器组织与学习及评估四个组件，建立细粒度分类体系，并通过时间维度分析梳理研究重点从多模态内容理解到证据获取与管理的演进脉络及未来方向。

    

    视觉文档理解对于从视觉内容丰富的文档中获取知识至关重要，正吸引着越来越多的研究兴趣。现有综述大多侧重于模型架构或技术方法，而对任务瓶颈如何塑造技术选择与研究优先级的关注有限。我们提出了一项以任务为中心的综述，将长文档视觉问答（VQA）分解为四个组成部分：文档解析、证据组织与检索、生成器组织与学习，以及评估。针对每个部分，我们构建了细粒度的分类体系，并梳理了研究动机、技术应对方案以及各组件之间的关联。我们聚焦于显式检索，同时也涵盖直接阅读方式。我们通过时间维度的分析，追踪了研究重点的演进过程：从多模态内容理解，扩展到充分且高保真的证据获取，再到持续的证据管理。我们进一步展望了证据充分性、可验证性等方面的未来研究方向。

    arXiv:2610.31385v1 Announce Type: cross  Abstract: Visual document understanding is essential for acquiring knowledge from visually rich documents and is attracting growing research interest. Existing surveys emphasize architectures or technical approaches, with limited attention to how task bottlenecks shape technical choices and research priorities. We present a task-centric survey that decomposes long-document visual question answering (VQA) into four components: document parsing, evidence organization and retrieval, generator organization and learning, and evaluation. For each, we develop a fine-grained taxonomy and trace motivations, technical responses, and cross-component connections. We focus on explicit retrieval while covering direct reading. Our temporal analysis traces expanding priorities from multimodal content understanding to sufficient, high-fidelity evidence acquisition and sustained evidence management. We outline future directions in evidence sufficiency, verifiable
    
[^2]: SurveyGraph：基于方法-实体知识图谱生成学术综述

    SurveyGraph: Generating Academic Surveys via Methods-Entity Knowledge Graphs

    [https://arxiv.org/abs/2610.31317](https://arxiv.org/abs/2610.31317)

    SurveyGraph通过构建编码方法级后续、竞争与资源共享关系的“方法-实体知识图谱”（MEKG）来生成学术综述，使图感知检索的召回率相比扁平文本基线近乎翻倍，并显著提升了生成内容的引用支持率。

    

    大多数自动化综述生成系统将论文作为扁平文档进行检索，限制了其捕捉研究结构的能力。我们提出SurveyGraph，一个围绕方法-实体知识图谱（MEKG）构建的框架，该图谱编码了方法层面的后续研究、竞争以及共享资源关系。该图谱支持核心论文发现、检索、混合大纲构建以及基于证据门控的叙事综合。我们评估了检索、语义边、引用依据、文献集覆盖率以及生成阶段组件。在一项受控的七种方法等预算基准测试中，以MEKG为锚点的融合方法在召回率与结构密度之间取得平衡，并通过独立的共被引检验加以验证。在另一项独立的存档图原生诊断中，图感知检索在八次运行中的召回率几乎是扁平词汇基线的两倍（p = 0.004）。语义证据门控达到AUC = 0.75。对160个生成句子的审计发现，80.6%的句子具有锚定或部分的引用支持。

    arXiv:2610.31317v1 Announce Type: new  Abstract: Most automated survey generation systems retrieve papers as flat documents, limiting their ability to capture research structure. We present SurveyGraph, a framework built around a Methods-Entity Knowledge Graph (MEKG) encoding method-level follow-on, competition, and shared-resource relations. The graph supports core-paper discovery, retrieval, hybrid outlining, and evidence-gated narrative synthesis. We evaluate retrieval, semantic edges, citation grounding, collection coverage, and generation-stage components. In a controlled seven-method equal-budget benchmark, MEKG-anchored fusion balances recall and structural density, with an independent co-citation check. In a separate archived graph-native diagnostic, graph-aware retrieval nearly doubles recall over flat lexical baselines across eight runs (p = 0.004). A semantic evidence gate achieves AUC = 0.75. Audits of 160 generated sentences find 80.6% anchored-or-partial citation support.
    
[^3]: 用于验证电子取证中审查导向型AI工作流的延迟倒计时方法

    The Delayed Countdown Method for Validating Review-Oriented AI Workflows in eDiscovery

    [https://arxiv.org/abs/2610.31311](https://arxiv.org/abs/2610.31311)

    本文提出延迟倒计时方法，为电子取证中审查导向型AI工作流的有效性验证提供了一种简单实用、易于集成、兼容任意停止规则并能给出有效置信区间的评估方案。

    

    面向审查的工作流已成为电子取证中AI辅助文档审查的主流方法。然而，如何验证审查导向型工作流的有效性，一直是法律领域技术辅助审查（TAR）中一个令人困惑且充满争议的话题。我们分析了这一争议的根源，并为审查导向型工作流提出了一种简单而实用的评估方法。我们的延迟倒计时方法易于融入现有的审查导向型工作流和软件中，兼容任意停止规则，并能产生关于有效性的有效频率学派置信区间。我们阐释了该方法的概念基础，提供了示例计算，讨论了对逻辑上不可能的召回率值的处理，并就降低审查成本的后续研究提出了建议。

    arXiv:2610.31311v1 Announce Type: new  Abstract: Review-oriented workflows have become the dominant approach to AI-assisted document review in electronic discovery (eDiscovery). How to validate the effectiveness of review-oriented workflows has, however, remained a confusing and controversial topic in technology-assisted review (TAR) in the law. We examine the sources of this controversy, and propose a simple and practical evaluation approach for review-oriented workflows. Our Delayed Countdown method is easily incorporated into existing review-oriented workflows and software, is compatible with arbitrary stopping rules, and produces valid frequentist confidence intervals on effectiveness. We explain the conceptual foundations of the method, provide example computations, discuss the handling of logically impossible recall values, and suggest future work on decreasing review costs.
    
[^4]: 真实文档修订下的分块重计算：测量固定分块、递归分块与内容定义分块的更新成本与检索质量

    Chunk Recomputation under Real Document Revisions: Measuring Update Cost and Retrieval Quality of Fixed, Recursive, and Content-Defined Chunking

    [https://arxiv.org/abs/2610.31019](https://arxiv.org/abs/2610.31019)

    该论文首次在真实文档持续修订的场景下量化比较了固定大小、递归和内容定义三种分块策略的更新成本与检索质量，发现递归分块的重计算开销最低（38-56%），而三者在检索召回率上的差异很小。

    

    面向检索增强生成（RAG）的分块策略几乎总是在静态快照上进行评估，然而知识库底层的文档却在持续变化。由于分块是独立进行向量嵌入的，编辑后边界的移动会迫使未发生变化的文本被重新嵌入。我们在120对相邻的中文和英文维基百科修订版本上，在严格的字节级相同复用规则和冻结的稠密编码器条件下，测量了三种无状态分块器（无重叠固定大小分块、递归的段落优先分块，以及内容定义分块）的更新成本。在由116篇文章组成的共同成功集合上，使用冻结的、按长度校准的参数，固定大小分块需要重计算其完整新版嵌入输入工作负载的57-68%，递归分块为38-56%，内容定义分块为43-54%（后者在62篇中文文章中有4篇处理失败），而在这些参数下，在固定的2048词元检索预算内，三者的平均证据召回率差异小于（原文摘要在此处截断）

    arXiv:2610.31019v1 Announce Type: new  Abstract: Chunking strategies for retrieval-augmented generation are almost always evaluated on a static snapshot, yet the documents underlying a knowledge base continue to change. Because chunks are embedded independently, a boundary that moves after an edit forces unchanged text to be re-embedded. We measure this update cost on 120 pairs of adjacent Chinese and English Wikipedia revisions for three stateless chunkers (non-overlapping fixed-size, recursive paragraph-first, and content-defined chunking) under a strict byte-identical reuse rule and a frozen dense encoder. On the 116-article common-success pool, with frozen, length-calibrated parameters, fixed-size chunking recomputes 57-68% of its full new-version embedding-input workload, recursive chunking 38-56%, and content-defined chunking 43-54% (it failed on 4 of 62 Chinese articles), while at these parameters mean evidence recall under a fixed 2048-token retrieval budget differs by less tha
    
[^5]: 通过人工标注及超越人工标注的形式化-非形式化数学对齐

    Formal-Informal Mathematical Alignment through and beyond Human Annotations

    [https://arxiv.org/abs/2610.31011](https://arxiv.org/abs/2610.31011)

    本文构建了将 Stacks Project 的非形式化数学陈述与 Lean mathlib 形式化内容相连接的数据集，并利用检索、LLM 重排序和 LLM 辅助验证等方法，在人工标注之外自动发现新的形式化与非形式化数学之间的对应关系。

    

    诸如 Lean 的 mathlib 这类形式化数学库提供了数学知识的机器可验证表示，包括定义、定理及其相互关系。如果能够与非形式化数学资源有效连接，形式化数学将有助于揭示数学概念之间此前隐含的联系，并改善对现有形式化成果的获取。然而，目前非形式化数学与形式化数学库之间的连接稀疏且缺乏系统性。在本文中，我们朝着更好的形式化整合迈出了第一步。我们通过利用专家标注，构建了一个将 Stacks Project 中的非形式化陈述与其在 mathlib 中的形式化版本相互关联的数据集。我们评估了检索、基于大语言模型（LLM）的重排序以及 LLM 辅助验证等方法，以发现新的、未经标注的非形式化到形式化的连接。为了便于形式化知识的探索与复用，我们将新发现的（连接）集成到了……（摘要此处不完整）

    arXiv:2610.31011v1 Announce Type: new  Abstract: Formalized mathematics libraries such as Lean's \texttt{mathlib} provide machine-verifiable representations of mathematical knowledge, including definitions, theorems, and their relationships. If effectively connected to informal mathematical resources, formal mathematics could help uncover previously implicit links between mathematical concepts and improve access to existing formalizations. However, connections between informal and formal mathematics libraries are currently sparse and unsystematic. In this paper, we make first strides towards better formal integration. We construct a dataset linking informal statements from the Stacks Project to their formalizations in \texttt{mathlib} by exploiting expert annotations. We evaluate retrieval, LLM-based reranking, and LLM-assisted verification to discover new unannotated informal-to-formal links. To facilitate exploration and reuse of formalized knowledge, we integrated the newly discover
    
[^6]: PAIR：面向LinkedIn高端人物推荐的偏好对齐意图检索框架

    PAIR: Preference-Aligned Intent Retrieval for Premium People Recommendation at LinkedIn

    [https://arxiv.org/abs/2610.30941](https://arxiv.org/abs/2610.30941)

    LinkedIn提出PAIR检索框架，通过建模用户意图偏好而非个人资料相似度，解决了专业网络中跨角色互补匹配的人物推荐问题。

    

    在专业网络中，基于个人资料相似度训练的通用嵌入常常会错误地对候选人进行排序，因为其相关性在本质上是互补的而非对称的。例如，求职者通常与招聘人员互动，而不是与拥有相似资历的同行互动。当前的用户推荐系统通常建立在个人资料相似度或社交信号（如共同好友）之上，这类方法在以交友为导向的社交网络中是有效的，因为用户倾向于与相似的人建立联系或在现有社交圈内互动。然而，这一假设在LinkedIn等专业平台上并不成立，因为这些平台上的用户互动往往是由目标驱动的跨角色互动，需要基于互补意图而非相似性或社交邻近度来进行匹配。在本文中，我们提出了偏好对齐意图检索，这是一个旨在建模用户意图偏好而非个人资料相似度的检索框架。

    arXiv:2610.30941v1 Announce Type: cross  Abstract: General-purpose embeddings trained on profile similarity often misrank candidates in professional networks, where relevance is inherently complementary rather than symmetric. For example, job seekers usually engage with recruiters rather than peers with similar credentials. Current user recommendation systems are often built on profile similarity or social signals such as shared connections, which are effective in friend-oriented social networks where users tend to connect with similar people or within existing social circles. However, this does not hold in professional platforms like LinkedIn, where interactions are often goal-driven and cross-role interactions, requiring matching based on complementary intents rather than similarity or social proximity. In this paper, we propose Preference-Aligned Intent Retrieval, a retrieval framework that is designed to model user intent preference instead of profile similarity. Our approach is in
    
[^7]: UserArena：面向长周期购物智能体的交互式用户模拟基准测试

    UserArena: Benchmarking Interactive User Simulation for Long-Horizon Shopping Agents

    [https://arxiv.org/abs/2610.30932](https://arxiv.org/abs/2610.30932)

    UserArena是一个包含6.4万余条真实用户会话轨迹的电商交互式基准，通过五级评估体系并对照生产环境A/B测试结果进行离线验证，用于开发和评估长周期购物智能体的推荐策略。

    

    智能体推荐通过一系列依赖上下文的决策来实现购物者目标。开发和评估此类策略需要忠实的推荐世界模型，既能刻画交互式购物环境，又能捕捉用户在长程交互中的行为反应。现有资源很少提供长周期、细粒度的真实用户轨迹，也难以支持轨迹级评估以及与生产环境A/B测试结果对照的离线验证。我们提出了UserArena，这是一个真实电商场景的交互式基准，包含来自11,010名用户的64,729条会话轨迹，涵盖566,102个商品和13种动作类型。UserArena将这些轨迹与交互式环境相结合，并支持与受控生产环境A/B测试结果对照的离线验证，实现了从输出有效性和局部决策到序列行为、个性化和最终结果的五级评估。为了提高模拟器的保真度……（原文摘要在此处截断）

    arXiv:2610.30932v1 Announce Type: new  Abstract: Agentic recommendation pursues shopper goals through sequences of context-dependent decisions. Developing and evaluating such policies requires faithful recommendation world models that capture both the interactive shopping environment and users' behavioral responses over long interactions. Existing resources rarely provide long-horizon, fine-grained real-user trajectories or support trajectory-level evaluation and offline validation against production A/B results. We introduce UserArena, a real-world e-commerce interactive benchmark containing 64,729 session trajectories from 11,010 users, spanning 566,102 items and 13 action types. UserArena pairs these trajectories with an interactive environment and supports offline validation against outcomes from a controlled production A/B test, enabling five-level evaluation from output validity and local decisions to sequential behavior, personalization, and outcomes. To improve simulator fideli
    
[^8]: Lean Finder V2：为 Mathlib 提供可消除用户查询歧义的语义搜索

    Lean Finder V2: Semantic Search for Mathlib That Disambiguates User Queries

    [https://arxiv.org/abs/2610.30843](https://arxiv.org/abs/2610.30843)

    Lean Finder V2 通过互补的模型级对抗训练与输入级查询重写，实现了能够消除歧义的 Mathlib 语义搜索，有效解决了短而模糊查询的检索难题。

    

    我们提出了 Lean Finder V2，这是一个面向 Mathlib 的语义搜索框架，通过互补的模型级对抗训练和输入级查询重写来消除用户查询的歧义。人类数学家和编码智能体很少通过提供详细上下文来搜索 Mathlib。在实践中，他们查询的是简短的性质、记号、别名或假设，而将大部分预期的数学内容隐含在查询之外。相比之下，现有的 Mathlib 搜索引擎不切实际地假设查询带有丰富的上下文。由于缺乏与真实标准（ground-truth）Mathlib 陈述配对的真实世界短查询，这种歧义问题变得更加难以处理。我们认为这本质上是一个数据问题。通过对抗训练，我们扩大了带标注的短且含歧义查询的生成规模，并在模型级微调和输入级查询重写两个层面上学习消除用户查询的歧义。在人工策划的 MathlibQR 基准测试上，我们的 Lean Finder V2 创造了新的最佳纪录（原文在此处截断）。

    arXiv:2610.30843v1 Announce Type: new  Abstract: We introduce Lean Finder V2, a semantic search framework for Mathlib that disambiguates user queries through complementary model-level adversarial training and input-level query rewriting. Human mathematicians and coding agents rarely search Mathlib by providing detailed context. In practice, they query short properties, notation, aliases, or hypotheses, leaving much of the intended mathematics implicit. In contrast, existing Mathlib search engines unrealistically assume context-rich queries. The scarcity of real-world short queries paired with ground-truth Mathlib statements makes this ambiguity even more intractable. We argue that this is essentially a data problem. Through adversarial training, we scale up the generation of labeled short and ambiguous queries, and learn to disambiguate user queries at both model-level fine-tuning and input-level query rewriting. On the human-curated MathlibQR benchmark, our Lean Finder V2 sets a new s
    
[^9]: 贝塞尔引导的自适应流用于跨域序列推荐

    Bezier-Guided Adaptive Flow for Cross-domain Sequential Recommendation

    [https://arxiv.org/abs/2610.30825](https://arxiv.org/abs/2610.30825)

    提出BGAFlow，一个结合贝塞尔引导的状态自适应语义演化与用户级迁移控制的个性化跨域序列推荐框架，利用LLM推断的用户画像和基于兴趣重叠度的个性化迁移视界，实现对跨域知识迁移的精确调控。

    

    跨域序列推荐旨在通过跨领域迁移有用的偏好知识来缓解数据稀疏和冷启动问题。然而，用户在不同领域间表现出不同程度的偏好一致性，而现有方法通常依赖于全局对齐目标或粗粒度的迁移策略。这类方法可能会引入无关信息，并且在根据个人用户偏好调整语义表示方面灵活性有限。为了应对这些挑战，我们提出了BGAFlow，这是一个将状态自适应语义演化与用户特定迁移控制相结合的个性化框架。BGAFlow首先通过统一的双域预训练建立共享语义空间，其中用户画像由大语言模型（LLM）推断得到。随后，基于潜在兴趣重叠度，它估计出一个个性化视界，以确定每个用户适当的迁移程度。为了实现……（摘要原文在此处截断）

    arXiv:2610.30825v1 Announce Type: new  Abstract: Cross-domain sequential recommendation aims to alleviate data sparsity and cold-start problems by transferring useful preference knowledge across domains. However, users exhibit varying degrees of preference consistency across domains, while existing methods often rely on global alignment objectives or coarse-grained transfer strategies. Such approaches may introduce irrelevant information and provide limited flexibility in adapting semantic representations to individual user preferences. To address these challenges, we propose BGAFlow, a personalized framework that combines state-adaptive semantic evolution with user-specific transfer control. BGAFlow first establishes a shared semantic space through unified dual-domain pretraining with user profiles inferred by a large language model (LLM). Based on latent interest overlap, it then estimates a personalized horizon to determine the appropriate transfer extent for each user. To enable se
    
[^10]: 基于GraphRAG的自动化富集分析：高效的多数据库ORA提取与生物学解释

    Automated Enrichment Analysis with GraphRAG: Efficient Multi-Database ORA Extraction and Biological Interpretation

    [https://arxiv.org/abs/2610.30710](https://arxiv.org/abs/2610.30710)

    提出GBOR-RAG框架，将多个异构基因集数据库统一为术语-基因知识图谱，通过检索证据子图条件化大语言模型生成可追溯的生物学富集解释，并引入声明级忠实度指标评估生成内容的证据支持。

    

    过度表示分析（ORA）被广泛用于解读基因表达研究，但其结果往往分散在多个基因集数据库中，并以扁平的术语列表形式呈现，掩盖了相关富集信号之间的基因层面联系。我们提出了基于图的过度表示检索增强生成（GBOR-RAG），这是一个用于ORA解读的集成且可追溯的框架。GBOR-RAG将异构的基因集数据库表示为统一的术语-基因知识图谱，其中共享的基因将来自不同来源的相关术语连接起来。在基于图的ORA评分之后，该框架检索出一个紧凑的连通证据子图，该子图同时保留了跨数据库关系和基因层面的支持。该子图作为大语言模型（LLM）的条件输入以生成生物学解释，并且我们引入了一个声明级忠实度指标来评估生成的声明是否有明确证据支持。实验表明，基于图的ORA保留了……

    arXiv:2610.30710v1 Announce Type: new  Abstract: Over-representation analysis (ORA) is widely used to interpret gene-expression studies, but its outputs are often fragmented across gene-set databases and presented as flat term lists that obscure gene-level links among related enrichment signals. We propose Graph-Based Over-Representation Retrieval-Augmented Generation (GBOR-RAG), an integrated and traceable framework for ORA interpretation. GBOR-RAG represents heterogeneous gene-set databases as a unified term--gene knowledge graph, where shared genes connect related terms across sources. After graph-based ORA scoring, the framework retrieves a compact connected evidence subgraph that preserves both cross-database relationships and gene-level support. The subgraph conditions an LLM to generate biological interpretations, and we introduce a claim-level faithfulness metric to assess whether generated claims are supported by explicit evidence. Experiments show that graph-based ORA preserv
    
[^11]: Lychee：面向高召回率向量搜索的倒排列表认证筛选

    Lychee: Certified Screening of Inverted Lists for High-Recall Vector Search

    [https://arxiv.org/abs/2610.30697](https://arxiv.org/abs/2610.30697)

    提出Lychee方法，利用精确搜索的认证下界在无需读取向量的前提下筛除绝大多数被探测的向量，以远低于全精度评分的成本实现0.999以上召回率的高效最近邻搜索。

    

    对嵌入向量的最近邻搜索将人脸与监控名单、交易与已知欺诈模式、医学影像与确诊病例进行匹配，在这些应用中，漏掉一个近邻就意味着漏掉一个威胁、一起欺诈或一个诊断。由于近似索引无法报告哪些查询丢失了真实的近邻，这些应用要求召回率达到0.999至0.9999，同时成本远低于对数十亿向量进行精确搜索。倒排文件索引通过探测更多列表来达到高召回率，但它们需要以全精度读取并为每个被探测的向量评分，因此吞吐量与被探测列表的数量成反比下降。图索引在未达到这些目标之前就已饱和，而量化列表则必须使用完整向量对其候选进行重排序。我们的洞见是，借助精确搜索的认证下界，无需读取向量即可判定几乎每个被探测的向量都无法进入被探测列表的前k名……（原文摘要在此处截断）

    arXiv:2610.30697v1 Announce Type: new  Abstract: Nearest-neighbor search over embedding vectors matches faces against watchlists, transactions against known fraud patterns, and medical images against confirmed cases, where a missed neighbor is a missed threat, fraud, or diagnosis. Because an approximate index cannot report which queries lost a true neighbor, these applications require recall of 0.999 to 0.9999, at a cost far below that of exact search over billions of vectors. Inverted-file indexes reach high recall by probing more lists, but they read and score every probed vector at full precision, so that throughput falls in inverse proportion to the number of probed lists. Graph indexes saturate below these targets, whereas quantized lists must rerank their candidates with the full vectors. Our insight is to decide for almost every probed vector that it cannot enter the top k of the probed lists without reading the vector, through the certified lower bound of exact search by certif
    
[^12]: OneLatent：面向高效基础推荐模型的潜在推理

    OneLatent: Latent Reasoning for Efficient Foundation Recommendation Models

    [https://arxiv.org/abs/2610.30675](https://arxiv.org/abs/2610.30675)

    OneLatent提出了一种潜在推理框架，将显式思维链推理压缩为可学习的潜在标记，并通过多视角自适应CoT生成高质量监督，从而在无需生成冗长推理轨迹的情况下实现高效的基础推荐模型。

    

    大语言模型（LLMs）已展现出强大的推理能力，这促使它们被用作基础推荐模型（FRMs）的主干网络。现有方法在“思考后回答”（Think-then-Answer）范式下，通过显式的思维链（Chain-of-Thought, CoT）推理来增强推荐效果。然而，显式CoT需要生成冗长的推理轨迹，带来巨大的推理开销，并且依赖手工设计的模板，难以捕捉多样且动态变化的用户兴趣。我们提出OneLatent，一个高效的潜在推理框架，它将显式的推理轨迹压缩为若干可学习的潜在标记，实现“潜在推理后回答”的推理方式，而无需生成冗长的轨迹。OneLatent首先引入多视角自适应思维链（Multi-View Adaptive CoT, MV-ACoT），通过从多个视角探索用户兴趣，并自动为每个实例调整推理复杂度，从而生成多样且高质量的教师监督信号。

    arXiv:2610.30675v1 Announce Type: new  Abstract: Large language models (LLMs) have demonstrated strong reasoning capabilities, motivating their use as the backbone of foundation recommendation models (FRMs). Existing methods enhance recommendations through explicit Chain-of-Thought (CoT) reasoning under a Think-then-Answer paradigm. However, explicit CoT incurs substantial inference overhead by generating lengthy reasoning traces and relies on manually designed templates that struggle to capture diverse, dynamic user interests. We propose OneLatent, an efficient latent reasoning framework that compresses explicit reasoning traces into several learnable latent tokens, enabling Latent-Reason-then-Answer inference without generating verbose traces. OneLatent first introduces Multi-View Adaptive CoT (MV-ACoT), which creates diverse, high-quality teacher-generated supervision by exploring user interests from multiple perspectives and automatically adapting reasoning complexity to each insta
    
[^13]: 将自回归LLM控制编译为面向分层RAG的策略精确并行推理

    Compiling Autoregressive LLM Control into Policy-Exact Parallel Inference for Hierarchical RAG

    [https://arxiv.org/abs/2610.30664](https://arxiv.org/abs/2610.30664)

    该论文提出将分层RAG中自回归LLM的控制决策编译为策略精确的并行推理——只需一次根预填充加一次树因果探测，即可在两次前向计算的关键路径、零输出token的条件下，精确复现串行语法约束解码的动作分布。

    

    大型语言模型越来越多地在结构化决策流水线中充当控制器，但其决策仍然通过自回归文本解码来执行。分层RAG（Hierarchical RAG）正是这样一种场景：在文档树的每个前沿节点，LLM需要在依赖输入的动作语法约束下，从k个章节中选定b个进行提交，其执行方式要么是容易出错的自由格式JSON，要么是可靠但严格串行的语法约束生成。我们证明，对于有界视界控制而言，这种串行化是不必要的：可达的未来形成一个浅层的动作前缀树，我们将其编译——仅需一次根预填充加一次树因果探测，即可利用串行解码所对应的token、位置和因果祖先，评估每个前缀条件下的动作分布。一个前缀状态引理保证编译后的控制在精确算术意义下达到策略精确：即相同的策略，但关键路径仅包含两次前向计算且零输出token。再经过一次探测过程即可验证精确的序边际化……（摘要在此处被截断）

    arXiv:2610.30664v1 Announce Type: new  Abstract: Large language models increasingly act as controllers in structured decision pipelines, yet their decisions are still executed by autoregressive text decoding. Hierarchical RAG is one such setting: at each frontier of a document tree, an LLM commits to $b$ of $k$ sections under an input-dependent action grammar, executed as either failure-prone free-form JSON or reliable but strictly serial grammar-constrained generation. We show that this serialization is unnecessary for bounded-horizon control: the reachable future forms a shallow action-prefix tree, which we compile --- one root prefill plus one tree-causal probe evaluate every prefix-conditioned action distribution with serial decoding's tokens, positions, and causal ancestry. A prefix-state lemma makes compiled control policy-exact in exact arithmetic: the same policy with a two-forward critical path and zero output tokens. One further probe pass certifies the exact order-marginaliz
    
[^14]: 融合税：混合检索何时造成损害的诊断性分析

    The Fusion Tax: A Diagnostic Analysis of When Hybrid Retrieval Hurts

    [https://arxiv.org/abs/2610.30612](https://arxiv.org/abs/2610.30612)

    该研究通过诊断分析发现，混合检索（如RRF融合）虽然在总体指标上优于纯稠密检索，但在8-48%的个体查询上反而损害检索效果，因此“不融合”应被视为值得认真对待的一等选项。

    

    混合检索——即将稀疏词法检索器与稠密神经检索器相融合——在2026年9月调研的软件文档中被广泛推荐，通常采用倒数排序融合（RRF），其依据是总体性能上的提升。我们研究融合在单个查询上付出了什么代价，并将“不融合”作为一个一等公民级别的结果来对待。实验设置正是混合检索实际部署的典型场景：标准的Anserini BM25、五个跨越2200万至5.96亿参数且涵盖三代模型的稠密编码器、一个学习型稀疏（SPLADE）替代方案，以及交叉编码器第二阶段重排，在七个BEIR数据集和三种融合函数上进行评估。即使是最好的方法——稠密加权分数插值——在13-48%的查询上，其将相关文档的排名低于两个组件中较好的那个，在8-21%的查询上低于可直接部署的纯稠密检索器（在留一语料库验证下选择权重时，这两个比例分别为14-48%和11-25%），同时在每个语料库的总体指标上仍优于纯稠密检索。

    arXiv:2610.30612v1 Announce Type: cross  Abstract: Hybrid retrieval, which fuses a sparse lexical retriever with a dense neural one, is widely recommended in the software documentation surveyed in September 2026, usually with reciprocal rank fusion (RRF), on the strength of aggregate gains. We ask what fusion costs on individual queries, treating "do not fuse" as a first-class outcome, in the setting in which hybrid retrieval is deployed: a standard Anserini BM25, five dense encoders spanning 22M-596M parameters and three model generations, a learned-sparse (SPLADE) alternative, and a cross-encoder second stage, on seven BEIR datasets and three fusion functions. Even the best method, dense-weighted score interpolation, ranks relevant documents below the better of its two components on 13-48% of queries and below the deployable dense-only retriever on 8-21% (14-48% and 11-25% at a weight selected under leave-one-corpus-out validation), while beating dense-only in aggregate on every corp
    
[^15]: 更好的推理，而非仅仅是更大的模型：一个基于结构的高效可靠推理框架

    Better Reasoning, Not Just Bigger Models: A Structure-Grounded Framework for Efficient and Reliable Inference

    [https://arxiv.org/abs/2610.30611](https://arxiv.org/abs/2610.30611)

    提出了无需训练的 SGUR 框架，通过将不完整观测表示为查询图，并利用先验加权的部分图匹配和主动信息增益策略在结构化假设空间中主动消减不确定性，从而提升多轮推理的可靠性与效率。

    

    扩展大型语言模型显著提升了语言流畅性，然而可靠的自主智能体所需要的不仅仅是流畅的文本生成。在多轮推理中，当前的智能体系统常常产生语义上合理但结构上缺乏依据的推理，尤其是在观测信息不完整的情况下。我们提出了结构化落地的 uncertainty 消减框架 SGUR（Structure-Grounded Uncertainty Reduction），这是一个无需训练的框架，通过对结构化假设空间进行主动的不确定性消减来补充词元级别的生成。SGUR 将不完整的观测表示为查询图，并通过先验加权的部分图匹配机制将其与候选假设图进行对齐，其中由数据库导出的经验先验会赋予与决策相关的关系更大的重要性，而不是对所有图的边一视同仁。由此得到的推理状态通过主动信息增益策略进行迭代优化，该策略会选择下一个信息量最大的节…

    arXiv:2610.30611v1 Announce Type: cross  Abstract: Scaling large language models has substantially improved linguistic fluency, yet reliable autonomous agents require more than fluent generation. In multi-turn inference, current agentic systems often produce semantically plausible but structurally ungrounded reasoning, especially under incomplete observations. We propose Structure-Grounded Uncertainty Reduction (SGUR), a training-free framework that complements token-level generation with active uncertainty reduction over structured hypothesis spaces. SGUR represents incomplete observations as a query graph and aligns them with candidate hypothesis graphs through a prior-weighted partial graph matching mechanism, where database-derived empirical priors assign greater importance to decision-relevant relations rather than treating all graph edges uniformly. The resulting inference state is iteratively refined by an active information-gain strategy that selects the next most informative n
    
[^16]: 期刊篮：支持跨学科文献检索与方法论透明度

    Journal Basket: Supporting Cross-Disciplinary Literature Searching and Methodological Transparency

    [https://arxiv.org/abs/2610.30544](https://arxiv.org/abs/2610.30544)

    本文介绍了Journal Basket——一个帮助商科学生基于ABDC期刊质量列表构建期刊集合、生成多数据库检索式并进行跨学科文献检索的浏览器工具，同时提升文献综述的方法论透明度。

    

    开始研究项目的商科学生必须学习如何识别相关的学术出版物、区分学科来源、构建数据库检索式，以及报告其文献综述背后的决策过程。由于期刊分类系统和书目数据库以不同的方式组织学术信息，这些任务可能颇为困难。本文介绍了Journal Basket，一个专为帮助商科学生进行期刊限定式文献检索而开发的基于浏览器的工具，目前已进入beta测试阶段。Journal Basket使用户能够从澳大利亚商学院院长理事会（ABDC）期刊质量列表构建期刊集合，可以从42个商科学科预设、完整的ABDC学科领域或《金融时报》FT50列表中进行选择，应用期刊评级阈值，并为Scopus、Web of Science和Business Source数据库生成检索表达式。用户还可以查看和修改期刊集合，下载…

    arXiv:2610.30544v1 Announce Type: cross  Abstract: Business students beginning a research project must learn how to identify relevant scholarly publications, distinguish disciplinary sources, construct database searches, and report the decisions behind their literature reviews. These tasks can be difficult because journal classification systems and bibliographic databases organize scholarly information differently. This paper introduces Journal Basket, a browser-based tool developed to help business students conduct journal-bounded literature searches and currently available in beta. Journal Basket enables users to construct journal collections from the Australian Business Deans Council (ABDC) Journal Quality List, choose from 42 business discipline presets, entire ABDC fields or the Financial Times FT50 list, apply journal rating thresholds, and generate search expressions for Scopus, Web of Science and Business Source. Users can also inspect and modify journal collections, download j
    
[^17]: 掌握学习型稀疏检索中基于图的搜索（Getting a GRIP）

    Getting a GRIP on Graph-Based Search for Learned Sparse Retrieval

    [https://arxiv.org/abs/2610.30425](https://arxiv.org/abs/2610.30425)

    本文挑战了“图索引在学习型稀疏检索中劣于倒排索引”的传统观点，提出将倒排索引的优势融入图搜索，从而使基于图的ANN方法在LSR任务上也能达到有竞争力的性能。

    

    学习型稀疏检索（LSR）近年来在信息检索领域受到广泛关注，因为它能够将神经表示的泛化能力与基于关键词搜索的可解释性相结合。近期多项研究表明，倒排索引在针对学习型稀疏嵌入进行适当修改后，能够为LSR实现最先进的召回率-延迟权衡，在固定召回率目标下，其延迟通常大幅优于基于图的索引。因此，学界普遍认为，对于LSR而言，基于图的近似最近邻（ANN）搜索不如倒排索引。本文挑战了这一传统观点。我们的核心洞察是：虽然将标准的基于图的ANN索引直接应用于稀疏向量会导致缺乏竞争力的性能，但通过将倒排索引的优势融入图搜索中，我们可以克服这些不足……

    arXiv:2610.30425v1 Announce Type: new  Abstract: Learned sparse retrieval (LSR) has recently gained traction in the information retrieval community for the ability to combine the generalization capabilities of neural representations with the interpretability of keyword-based search. A number of recent works have demonstrated that inverted indexes, when appropriately modified for learned sparse embeddings, achieve state-of-the-art recall-latency tradeoffs for LSR, often outperforming graph-based indexes by substantial margins in latency at a fixed recall target. As a result, the community has generally concluded that graph-based approximate nearest neighbor (ANN) search is inferior to inverted indexes for LSR. This paper challenges this conventional wisdom. Our key insight is that, while applying standard graph-based ANN indexes out-of-the-box to sparse vectors leads to uncompetitive performance, we can overcome these shortcomings by infusing the strengths of inverted indexes into graph
    
[^18]: 当智能体式检索悄然失效时：来自创意AI助手的框架与生产环境证据

    When Agentic Retrieval Fails Silently: Framework and Production Evidence from a Creative AI Assistant

    [https://arxiv.org/abs/2610.30423](https://arxiv.org/abs/2610.30423)

    该论文指出传统检索评估指标（CTR、MRR、行为NDCG）在智能体助手中因“归因鸿沟”而悄然失效，并提出无需新增建模的LARA框架，通过检索质量、参与与结果、产品成功三个独立层级以及四字段溯源模式，在生产环境Adobe Express中评估智能体式检索的真实效果。

    

    CTR（点击率）、MRR（平均倒数排名）和行为NDCG等指标均假设用户会看到排序后的结果并据此采取行动。而在智能体式助手（agentic assistant）中，生成层会首先消费检索输出，这一假设在三个方面被打破：智能体先于用户读取检索结果（“谁”）；用户的行动可能发生在检索之后的数轮对话或多个会话（“何时”）；最终结果融合了检索内容与模型自身的知识（“何因”）。我们将这一现象称为“归因鸿沟”（attribution gap）。我们提出分层智能体检索评估框架（LARA），将检索质量、参与度与结果、产品成功作为三个相互独立的层级加以测量，并通过一个四字段的溯源模式（lineage schema）将各层连接起来，且无需引入新的建模。我们在Adobe Express上对这一动机进行了考察：同一套检索基础设施同时服务于画廊搜索界面与智能体式助手，后者面向相同的用户群体，但会话量仅约为前者的八分之一。聊天用户的查询重述比例高于画廊用户（44.4% 对 33……

    arXiv:2610.30423v1 Announce Type: new  Abstract: CTR, MRR, and behavioral NDCG assume the user sees ranked results and acts on them. In an agentic assistant, a generation layer consumes retrieval output first, and that assumption breaks in three ways. The agent reads the results before the user does (who). The user's action can come several turns or sessions after retrieval (when). Outcomes blend retrieved content with the model's own knowledge (what caused it). We call this the attribution gap. We propose Layered Agentic Retrieval Assessment (LARA), which measures retrieval quality, engagement and outcome, and product success as three independent layers joined by a four-field lineage schema that requires no new modeling. We examine the motivation on Adobe Express, where one retrieval infrastructure serves a gallery search surface and an agentic assistant at roughly one-eighth its session volume for the same user population. Chat users reformulate more than gallery users (44.4% vs. 33.
    
[^19]: 网络搜索型大语言模型中的数据空洞漏洞利用

    Data-Void Exploits in Web-Searching LLMs

    [https://arxiv.org/abs/2610.30365](https://arxiv.org/abs/2610.30365)

    该论文揭示了一种针对网络搜索型大语言模型的新攻击方式——“数据空洞”漏洞利用：虚假信息散布者在可靠内容稀缺的搜索空间中植入叙事内容，诱导LLM在回答提问时检索并引用这些不可靠来源，并通过实验考察了提示语气、模型选择和攻击者控制手段对攻击效果的影响。

    

    arXiv:2610.30365v1 公告类型：交叉发布。摘要：具备网络搜索功能的大语言模型（LLM）改变了用户在线获取信息的方式：用户不再逐条浏览排好序的链接列表，而是直接阅读模型综合生成的答案。这一转变催生了一种新的LLM漏洞：数据空洞（data-void）漏洞利用。虚假信息散布者将他们的叙事内容发布到相关且可靠内容稀缺的搜索空间（即“数据空洞”）中，然后依赖LLM的网络搜索功能检索这些页面，并在响应用户针对数据空洞的提问时使用、引用或列出这些页面。与传统搜索不同（用户可以看到排名提示和相互竞争的搜索结果），LLM可能掩盖其答案依赖于稀疏、未经验证或对抗性来源的事实。我们通过改变用户提示的语气——怀疑、兴奋或中性——以及LLM的选择——ChatGPT、Gemini或Sonar Pro——再加上攻击者的三种控制手段来研究这一威胁。攻击者可以使用五种已知策略之一来构建并填充数据空洞以植入其叙事内容，并托管其受控的……（原文摘要在此处被截断）

    arXiv:2610.30365v1 Announce Type: cross  Abstract: Web-searching Large Language Models (LLMs) have changed how users seek online information: instead of sifting through ranked lists of links, users read synthesized answers. This shift creates a new LLM vulnerability: a data-void exploit. Disinformers publish their narratives in search spaces where relevant and reliable content is scarce, i.e. data voids, then rely on the LLM's web-search to retrieve and then use, cite, or list those pages in its responses to data-void seeking prompts. Unlike classical search, where users see ranking cues and competing results, LLMs can obscure when an answer relies on sparse, unverified, or adversarial sources. We study this threat by varying the user's prompt tone - skeptical, excited, or neutral - and LLM choice - ChatGPT, Gemini, or Sonar Pro - as well as three attacker controls. The attacker can use one of five known strategies to construct and fill a data void with their narrative, host their cont
    
[^20]: FAIRGate：面向更公平推荐分发的智能体化运行时策略选择与可验证修复

    FAIRGate: Agentic Runtime Policy Selection and Verified Repair for Fairer Recommendation Delivery

    [https://arxiv.org/abs/2610.30359](https://arxiv.org/abs/2610.30359)

    FAIRGate提出了一种模型无关的推荐分发控制器，通过智能体在运行时从预定义的公平策略库中动态选择合适策略，并借助确定性执行与多重约束验证的修复机制，实现更公平且可审计的推荐结果分发。

    

    公平感知的推荐系统通常在训练阶段编码单一公平性目标，或在评分后应用固定的重排序规则。然而在部署阶段，合适的公平性操作可能因请求而异，因为候选池、物品元数据、曝光债务、用户偏好历史和效用预算都会在运行时发生变化。我们提出FAIRGate，一个模型无关的分发控制器，它在基础排序器和最终top-K列表之间放置了一个智能体化的策略选择层。该智能体并非从头定义公平性目标，而是从一个预定义的策略库中进行选择，该策略库涵盖长尾曝光、物品组曝光均衡、集中度控制、校准保护和弃权等策略。随后，一个确定性执行器通过受限修复来应用所选策略，并且只有在相关性、可行性、元数据、校准和审计约束均得到满足时，验证器才会接受该结果。（注：原摘要在此处被截断）

    arXiv:2610.30359v1 Announce Type: new  Abstract: Fairness-aware recommender systems often encode one fairness objective during training or apply a fixed re-ranking rule after scoring. In deployment, however, the appropriate fairness action can vary across requests because the candidate pool, item metadata, exposure debt, user preference history, and utility budget all change at runtime. We propose FAIRGate, a model-agnostic delivery controller that places an agentic policy-selection layer between a base ranker and the final top-K list. The agent does not define fairness objectives from scratch. It chooses from a predefined policy library covering long-tail exposure, item-group exposure balance, concentration control, calibration protection, and abstention. A deterministic executor then applies the selected policy through constrained repair, and a validator accepts the result only when relevance, feasibility, metadata, calibration, and audit constraints are satisfied. The final language
    
[^21]: 保护并非单一事件：基于度量终点的RAG保护研究综述

    Protection Is Not One Event: A Survey of RAG Protection Research by Measured Endpoint

    [https://arxiv.org/abs/2610.30193](https://arxiv.org/abs/2610.30193)

    该论文通过对118篇RAG防御论文按“实际度量终点”而非声称目标进行系统性分类综述，构建了七类分类体系，并揭示了防御机制与保护结果不对应、评估标准割裂等三大研究空白。

    

    检索增强生成（RAG）系统面临安全、隐私和版权风险，包括内容泄露、来源重构、未授权访问、数据投毒和提示注入。然而，RAG保护研究使用了互不兼容的评估指标、数据集和威胁模型。我们呈现了一项系统性综述和人工编码的证据图谱，涵盖118篇经全文确认的RAG防御论文，并按所度量的终点而非声称的目标或防御机制进行组织。我们的七类分类体系涵盖内容泄露、形式化隐私代理指标、访问控制与抑制、语料库使用检测以及流水线完整性与归因。32名标注者对论文进行了双重编码，kappa系数为0.867。该综述识别出三个研究空白。首先，检索重排序和上下文重写各自跨越五到六个终点类别，因此防御机制本身并不能确定保护结果。其次，仅有17.8%的研究使用内容泄露或重构（原文此处截断）

    arXiv:2610.30193v1 Announce Type: cross  Abstract: Retrieval-augmented generation (RAG) systems face security, privacy, and copyright risks including content leakage, source reconstruction, unauthorized access, data poisoning, and prompt injection. Yet RAG protection studies use incompatible evaluation metrics, datasets, and threat models. We present a systematic survey and human-coded evidence map of 118 full-text-confirmed RAG defense papers, organized by measured endpoint rather than claimed goal or defense mechanism. Our seven-family taxonomy covers content disclosure, formal privacy proxies, access control and suppression, corpus-use detection, and pipeline integrity and attribution. Thirty-two annotators double-coded the papers kappa=0.867. The survey identifies three gaps. First, retrieval reranking and context rewriting each span five or six endpoint families, so mechanisms do not identify protection outcomes. Second, only 17.8% of studies use content disclosure or reconstructi
    
[^22]: 真实世界网络环境中的深度搜索

    Deep Search in Real-World Web Environments

    [https://arxiv.org/abs/2610.30181](https://arxiv.org/abs/2610.30181)

    该论文提出了 OpenDS 任务及 OpenDS-World 基准，用于评估智能体在真实、开放的互联网环境中自主进行多步骤深度搜索、跨多个独立来源发现并整合证据，从而得出唯一事实性答案的能力。

    

    尽管深度搜索智能体已取得长足进步，但其在真实世界开放网络环境中进行端到端研究的能力仍未得到充分评估。与静态检索设置不同，真实网络中的信息分布在多样的来源、模态和异构格式之中，通常涉及复杂的交互机制，要求智能体在执行过程中自主地搜索、访问并整合相关证据。为填补这一空白，我们提出了 OpenDS 任务，旨在评估智能体在实时、开放的网络中进行多步骤深度搜索的能力。该任务要求智能体自主探索真实网络，通过网络搜索、浏览器交互和证据处理，从多个独立来源中发现并关联相关证据，并最终确定唯一的事实性答案。基于该任务设定，我们构建了 OpenDS-World，这是一个包含 20（原文摘要至此截断）……

    arXiv:2610.30181v1 Announce Type: new  Abstract: Despite substantial progress in Deep Search agents, their ability to conduct end-to-end research in real-world open Web environments remains insufficiently evaluated. Unlike static retrieval settings, information on the real-world Web is distributed across diverse sources, modalities, and heterogeneous formats, often involving complex interaction mechanisms that require agents to autonomously search for, access, and integrate relevant evidence during execution. To address this gap, we introduce the OpenDS task, designed to evaluate agents' ability to perform multi-step deep search in the live, open Web. This task requires agents to autonomously explore the real Web, discover and connect relevant evidence from multiple independent sources through Web search, browser interaction, and evidence processing, and ultimately determine a unique factual answer. Building on this task formulation, we construct OpenDS-World, a benchmark comprising 20
    
[^23]: QProtFuse：面向检索增强蛋白质功能预测的查询引导层级融合方法

    QProtFuse: Query-Guided Layer Fusion for Retrieval-Augmented Protein Function Prediction

    [https://arxiv.org/abs/2610.30180](https://arxiv.org/abs/2610.30180)

    QProtFuse利用查询引导自适应融合蛋白质语言模型的多层特征，并通过联合对比学习实现蛋白质与注释的协同检索，显著提升检索增强式蛋白质功能预测性能。

    

    从氨基酸序列预测蛋白质功能对于理解生物过程至关重要。大型语言模型（LLM）能够针对自然语言查询提供灵活的功能描述，但其对蛋白质的理解能力有限，这促使了检索增强生成（RAG）的应用。现有的蛋白质RAG方法主要侧重于扩展外部检索信号，而忽视了单个蛋白质语言模型（PLM）内部编码的丰富知识。我们证明了检索性能在PLM的不同层以及不同生物任务之间存在差异，这使得传统的最终层特征选择并非最优。为此，我们提出了QProtFuse，它利用查询自适应地融合多层蛋白质特征，并通过模态内和跨模态对比目标联合学习蛋白质检索与注释检索。在Prot-Inst-OOD数据集上，QProtFuse取得了最优的蛋白质和注释检索分数，通过注释重排达到了16.4%的Recall@1。

    arXiv:2610.30180v1 Announce Type: new  Abstract: Predicting protein functions from amino acid sequences is essential to understanding biological processes. Large language models (LLMs) offer flexible function descriptions to natural-language queries, yet their limited protein understanding motivates retrieval-augmented generation (RAG). Existing protein RAG methods primarily expand external retrieval signals while overlooking the rich knowledge encoded within a single protein language model (PLM). We show that retrieval performance varies across PLM layers and biological tasks, making conventional final-layer selection suboptimal. To this end, we propose QProtFuse, which uses queries to adaptively fuse multi-layer protein features and jointly learns protein and annotation retrieval through intra-modal and cross-modal contrastive objectives. On Prot-Inst-OOD, QProtFuse achieves the best protein and annotation retrieval scores, reaching 16.4\% Recall@1 with annotation reranking. These ga
    
[^24]: Steam平台上玩家认知与AI披露之间的关联：一项探索性研究

    Association between Player Perception and AI Disclosure on Steam: An Exploratory Study

    [https://arxiv.org/abs/2610.30178](https://arxiv.org/abs/2610.30178)

    该研究通过分析Steam评论发现，披露使用生成式AI的游戏在负面评论中会引来更多AI相关的批评，为理解玩家对AI披露的感知提供了首个探索性证据。

    

    生成式AI在游戏开发中的应用正迅速增长，并成为激烈辩论的焦点，其中玩家的反应扮演着核心角色。本文通过分析Steam评论，探讨披露使用AI的游戏是否与不同的评论模式相关联，从而为这一讨论做出贡献。我们使用跨越多个披露组别的数据集，应用包括关键词分析、词频分析和特征词分析在内的语言分析方法，来研究Steam上与AI相关的讨论。结果表明，披露使用AI的游戏在负面评论中与更高程度的AI相关批评相关联。我们的发现为理解玩家如何感知AI披露迈出了首个探索性步伐。

    arXiv:2610.30178v1 Announce Type: new  Abstract: The use of generative AI in game development is growing rapidly and is the subject of intense debate in which player reactions play a central role. This paper contributes to this discussion by analyzing Steam reviews to examine whether games with AI disclosure are associated with different review patterns. Using a dataset across multiple disclosure groups, we apply linguistic analysis including keyword-based, word frequency, and characteristic word analysis to investigate the AI-related discussion on Steam. Our results show that AI-disclosed games are associated with higher levels of AI-related criticism in negative reviews. Our findings provide a first exploratory step towards understanding how players perceive AI disclosure.
    
[^25]: FlexVDR：学习为视觉文档检索分配套娃表示预算

    FlexVDR: Learning to Allocate Matryoshka Representation Budgets for Visual Document Retrieval

    [https://arxiv.org/abs/2610.30177](https://arxiv.org/abs/2610.30177)

    FlexVDR 提出了一个在单一视觉文档检索器中根据查询自适应分配套娃表示预算的框架，通过联合调节编码器深度、嵌入宽度和词元数量，并利用预算分配器预测各路由的检索效用，从而在计算开销与检索质量之间实现灵活权衡。

    

    视觉文档检索依赖丰富的多向量表示来捕获文档页面中的细粒度证据，但对这些表示进行编码和匹配可能计算开销高昂。现有的压缩与弹性表示方法虽然提供了更小的配置，但提供多种预算选择并不能决定某个查询应使用哪种配置。分配决策必须在承担完整计算开销之前估计备选表示的检索价值，同时兼顾不同的资源偏好。我们提出 FlexVDR，一个在单一检索器内实现查询自适应预算分配的框架。我们构建了覆盖编码器深度、嵌入宽度和词元数量的联合路由族，通过检索监督与蒸馏来训练紧凑路由，同时以原生旁路保留全预算参考。预算分配器从……预测特定路由的检索效用（摘要原文在此处截断）。

    arXiv:2610.30177v1 Announce Type: new  Abstract: Visual document retrieval relies on rich multi-vector representations to capture fine-grained evidence in document pages, but encoding and matching these representations can be computationally expensive. Existing compression and elastic representation methods expose smaller configurations, yet making multiple budgets available does not determine which configuration a query should use. The allocation decision must estimate the retrieval value of alternative representations before incurring their full computation, while accommodating different resource preferences. We introduce FlexVDR, a framework for query-adaptive budget allocation within a single retriever. We construct a joint family of routes over encoder depth, embedding width, and token count. Retrieval supervision and distillation train the compact routes, while a native bypass preserves the full-budget reference. A Budget Allocator predicts route-specific retrieval utility from e
    
[^26]: FormBench：面向配方专利细粒度检索的基准测试

    FormBench: A Benchmark for Fine-Grained Retrieval in Formulation Patents

    [https://arxiv.org/abs/2610.30175](https://arxiv.org/abs/2610.30175)

    该论文提出了FormBench基准，用于评估模型从配方专利等技术文档中检索精确化学信息的能力，发现商业API编码器表现极差，而基于FormBench三元组进行任务自适应微调的开源编码器性能显著提升。

    

    我们提出了FormBench，一个用于测试模型从技术文档中检索精确化学信息能力的平台。FormBench包含55,352个查询和两个检索语料库变体，每个变体约有100万个段落：FormBench-Random（C0）使用随机采样的干扰项，FormBench-Structured（C1）使用结构上相近的干扰项。每个查询都配有一个锚定段落、一个硬负例和一个软负例——这些通过材料本体中的实体重叠评分构建，为模型的判别能力提供了更深层次的分级信号。我们评估了10种模型配置，涵盖商业API编码器（OpenAI、Voyage）、开源编码器（Nomic、GTE-large、Qwen3-Embed）及其任务自适应微调（TAPT）变体。我们发现：(1) 商业API在所有配置上表现都很差（OpenAI-3-large在C1上仅为0.009，在C0上为0.213）；(2) 在FormBench三元组上进行TAPT使Qwen3-Embed在C1上的性能提升至0.4085。

    arXiv:2610.30175v1 Announce Type: new  Abstract: We introduce FormBench, a platform for testing the ability of models to retrieve precise chemical information from technical documents. FormBench contains 55,352 queries and two retrieval corpus variants each with ~1M passages: FormBench-Random (C0), with randomly sampled distractors, and FormBench-Structured (C1), with structurally proximate distractors. Each query is paired with an anchor passage, a hard negative, and a soft negative - derived from a materials ontology through entity overlap scoring, providing a deeper graded signal on a model's discriminatory power. We evaluate 10 model configurations spanning commercial API encoders (OpenAI, Voyage), open-source encoders (Nomic, GTE-large, Qwen3-Embed), and their task-adaptive fine-tuned (TAPT) variants. We find: (1) commercial APIs score very poorly across all configurations (OpenAI-3-large: 0.009 on C1, 0.213 on C0); (2) TAPT on FormBench triplets lifts Qwen3-Embed to 0.4085 on C1,
    
[^27]: IRAH：面向检索流程优化的分层智能体树搜索

    IRAH: Hierarchical Agentic Tree Search for Retrieval Pipeline Optimization

    [https://arxiv.org/abs/2610.30167](https://arxiv.org/abs/2610.30167)

    IRAH将UCT引导的分层智能体树搜索与离线元学习初始化相结合，在有限评估预算下联合优化RAG检索流程的结构组件与数值超参数，性能最高超越TPE达6.4%。

    

    优化用于检索增强生成（RAG）的信息检索（IR）流程，需要在有限的评估预算下联合选择结构组件和数值超参数。我们提出了信息检索智能体框架IRAH，它将面向信息检索的语言智能体树搜索（LATS-IR）与离线元学习初始化相结合。LATS-IR执行由UCT引导的分层搜索，首先探索宏观结构层面的流程选择，然后细化数值参数，并通过自然语言反思来指导后续的变异操作，同时提供可审查的搜索轨迹。我们在三个分布内基准和一个分布外基准上评估IRAH，以考察其在分布偏移下的泛化能力。在相同的元学习初始化条件下，LATS-IR在三个分布内基准中的两个上优于TPE，提升最高达6.4%，并在分布外基准上保持竞争力。

    arXiv:2610.30167v1 Announce Type: cross  Abstract: Optimizing Information Retrieval (IR) pipelines for Retrieval-Augmented Generation (RAG) requires jointly selecting structural components and numerical hyperparameters under limited evaluation budgets. We introduce the Information Retrieval Agentic Harness (IRAH), which combines Language Agent Tree Search for IR (LATS-IR) with offline meta-learned initialization. LATS-IR performs UCT-guided hierarchical search, first exploring macro-structural pipeline choices and then refining numerical parameters, with natural-language reflections to guide subsequent mutations and provide inspectable search traces. We evaluate IRAH on three in-distribution benchmarks and one out of-distribution benchmark to assess generalization under distribution shift. Under the same meta-learned initialization, LATS-IR outperforms TPE on two of three in-distribution benchmarks, with gains of up to 6.4%, and remains competitive on the out-of-distribution benchmark.
    
[^28]: 语义投递网络：为LLM智能体重构Web检索基础设施

    Semantics Delivery Network: Rethinking Web Retrieval Infrastructure for LLM Agents

    [https://arxiv.org/abs/2609.22486](https://arxiv.org/abs/2609.22486)

    该论文提出“语义投递网络”，主张将Web检索基础设施从面向人类客户端的URL/摘要检索模式，重构为直接向LLM智能体提供面向任务效用的语义块检索服务，并支持跨轮次有状态检索与智能体间的工作共享。

    

    大语言模型（LLM）在回答需要专有信息或最新实时网页内容的问题时，越来越多地依赖外部来源，其方式既包括传统的单次检索增强生成（RAG），也包括多轮智能体式RAG（agentic RAG）。然而，当今的网络基础设施仍然是为人类客户端构建的：给定一个查询，当前的搜索服务返回的是按通用相关性排序的URL和摘要列表；内容分发网络（CDN）缓存以URL为地址的对象（文本、图像、视频等），但并不知道智能体究竟需要哪一段内容。相比之下，LLM消费的是简短且语义连贯的段落（下文称为“语义块”），这些语义块应根据下游任务的效用而非仅仅依据相似性来选择，且LLM可能在多个推理轮次中进行有状态的检索。此外，缺乏协调的智能体还会重复执行搜索、数据获取和语义处理，导致本可共享的工作被重复完成。我们主张，语义块的检索应当成为

    arXiv:2609.22486v1 Announce Type: cross  Abstract: Large language models (LLMs) increasingly rely on external sources when answering questions that require proprietary information or up-to-date live web content, through both traditional single-shot retrieval-augmented generation (RAG) and multi-turn agentic RAG. Yet today's web infrastructure is still built for human clients. Given a query, current search services return a list of URLs and snippets ranked for generic relevance; content delivery networks (CDNs) cache URL-addressed objects (texts, images, videos, etc.) without knowing which passage an agent needs. LLMs, in contrast, consume short, semantically coherent passages, hereafter "chunks", selected for downstream task utility rather than similarity alone, and may retrieve statefully across reasoning turns. Uncoordinated agents also repeat search, data acquisition, and semantic processing, duplicating work that could be shared. We argue that semantic chunk retrieval should become
    
[^29]: 基于大语言模型增强的多图学习与对比对齐的兴趣点推荐

    POI Recommendation with LLM-Augmented Multi-Graph Learning and Contrastive Alignment

    [https://arxiv.org/abs/2608.16407](https://arxiv.org/abs/2608.16407)

    本文提出LLM-MGCL，通过结合语义和地理信息的多图学习与对比对齐，有效解决了POI推荐中的冷启动问题。

    

    摘要：基于图神经网络的兴趣点（POI）推荐模型通过在用户-物品交互上传播协同信号实现强性能，但它们在冷启动问题上表现不佳，即交互很少或没有交互的物品无法被有效表示。在本文中，我们提出了大语言模型增强的多图对比学习（LLM-MGCL），这是一种多图神经网络，利用物品的语义和空间信息，通过两个辅助物品-物品图扩展LightGCN骨干网络：一个是由大语言模型生成的图片摘要和关键词的句子嵌入构建的语义图，另一个是根据商业地点之间的哈弗辛距离推导出的地理图。物品嵌入在所有三个图上并行传播，通过加性融合，并通过双向InfoNCE对比目标跨视图对齐，连接同一物品的行为、语义和空间表示。

    arXiv:2608.16407v1 Announce Type: cross  Abstract: Point-of-interest (POI) recommendation models based on graph neural networks achieve strong performance by propagating collaborative signals over user-item interactions, yet they struggle with the cold-start problem, where items with few or no interactions are not represented. In this paper, we propose LLM-augmented Multi-Graph Contrastive Learning (LLM-MGCL), a multi-graph neural network that uses semantic and spatial information about items to extend the LightGCN backbone with two auxiliary item-item graphs: a semantic graph constructed from sentence embeddings of LLM-generated photo summaries and keywords, and a geographic graph derived from Haversine distances between business locations. Item embeddings are propagated over all three graphs in parallel, fused additively, and aligned across views through a bidirectional InfoNCE contrastive objective that connects behavioral, semantic, and spatial representations of the same items. Ex
    
[^30]: 先结构化再查询：实现对非结构化文档的精确分析查询

    Structure then Query: Enabling Precise Analytical Queries over Unstructured Documents

    [https://arxiv.org/abs/2608.13384](https://arxiv.org/abs/2608.13384)

    本文提出了AnnoIndex系统，通过自动生成分层注释模式并提取结构化值，将非结构化文档转化为可查询的结构化索引，从而克服了传统向量模糊匹配的局限性，实现了精确的分析查询。

    

    非结构化文档构成了企业和网络数据的主体。随着大型语言模型（LLMs）的快速发展，研究人员开始构建像操作数据库一样分析非结构化文本数据的数据系统。然而，由于主流检索方法仍依赖于基于向量相似性的模糊匹配，精确获取信息并进行结构化分析和推理仍然是一个重大挑战。为了解决这些局限性，AnnoIndex引入了两个核心基础组件。第一个是注释索引。该系统使用一个名为SchemaLoop的模块，从原始语料库中自动创建分层注释模式，然后使用轻量级语言模型提取特定值。它将分散的非结构化文本转化为物化的、结构化的索引，从而实现低成本的过滤和查询。注释索引避免了向量黑盒匹配的缺陷。

    arXiv:2608.13384v1 Announce Type: new  Abstract: Unstructured documents constitute the majority of enterprise and web data. With the rapid development of large language models(LLMs), researchers have started to build data systems that analyze unstructured textual documents like operating on databases. However, because mainstream retrieval methods still relies on fuzzy matching based on vector similarity, accurately obtaining information and performing structured analysis and reasoning remains a major challenge. To address these limitations, AnnoIndex introduces two core fundamental components. The first is Annotation Index. The system uses a module called SchemaLoop to automatically create hierarchical annotation schemas from the raw corpus, and then uses lightweight language model to extract specific values. It turns scattered unstructured text into a materialized, structured index that enables low-cost filtering and querying. The annotation index avoids the black-box matching of vect
    
[^31]: OneLatent：面向高效基础推荐模型的潜在推理

    OneLatent: Latent Reasoning for Efficient Foundation Recommendation Models

    [https://arxiv.org/abs/2607.26621](https://arxiv.org/abs/2607.26621)

    OneLatent提出将显式思维链推理压缩为可学习的潜在token，实现“先潜在推理后回答”的高效推荐框架，通过多视角自适应CoT生成高质量监督信号，在捕捉多样化用户兴趣的同时大幅降低推理开销。

    

    大型语言模型（LLMs）展现了强大的推理能力，这促使它们被用作基础推荐模型（FRMs）的主干。现有方法通过在“先思考后回答”范式下进行显式思维链（CoT）推理来增强推荐效果。然而，显式CoT会生成冗长的推理轨迹，带来巨大的推理开销，并且依赖手动设计的模板，难以捕捉多样且动态变化的用户兴趣。我们提出OneLatent，一个高效的潜在推理框架，它将显式推理轨迹压缩为若干可学习的潜在token，实现“先潜在推理后回答”的推理方式，无需生成冗长的推理轨迹。OneLatent首先引入多视角自适应CoT（MV-ACoT），通过从多个视角探索用户兴趣并自动调整推理复杂度，生成多样化、高质量的教师监督信号。

    arXiv:2607.26621v3 Announce Type: replace-cross  Abstract: Large language models (LLMs) have demonstrated strong reasoning capabilities, motivating their use as the backbone of foundation recommendation models (FRMs). Existing methods enhance recommendations through explicit Chain-of-Thought (CoT) reasoning under a Think-then-Answer paradigm. However, explicit CoT incurs substantial inference overhead by generating lengthy reasoning traces and relies on manually designed templates that struggle to capture diverse, dynamic user interests. We propose OneLatent, an efficient latent reasoning framework that compresses explicit reasoning traces into several learnable latent tokens, enabling Latent-Reason-then-Answer inference without generating verbose traces. OneLatent first introduces Multi-View Adaptive CoT (MV-ACoT), which creates diverse, high-quality teacher-generated supervision by exploring user interests from multiple perspectives and automatically adapting reasoning complexity to 
    
[^32]: Xetrieval：通过特征引出解释稠密检索

    Xetrieval: Explaining Dense Retrieval through Feature Elicitation

    [https://arxiv.org/abs/2605.29507](https://arxiv.org/abs/2605.29507)

    提出Xetrieval框架，通过特征引出器生成辅助视图使文档中与检索相关的语义更加显式，并借助共享稀疏字典以可解释的SAE特征表达查询-文档关系，从而解释稠密检索的相关性得分。

    

    稠密检索能够实现有效的语义匹配，但其相关性得分背后的语义因素仍然是不透明的。稀疏自编码器（SAEs）通过将稠密嵌入分解为稀疏语义特征，为特征层面的解释提供了支持。然而，检索器相关性得分所依据的文档语义在嵌入中可能不够显式，难以被恢复为可解释的SAE特征。我们提出了Xetrieval，一个通过特征引出（feature elicitation）来解释稠密检索的框架。特征引出器通过辅助视图使文档含义中与检索相关的方面更加显式，从而便于将其识别为可解释的SAE特征。一个共享的稀疏字典从这些多样化的表示中学习，通过可描述的特征来表达查询与文档之间的关系。我们通过对原始文档嵌入进行干预来评估这些特征对相关性得分的影响……

    arXiv:2605.29507v2 Announce Type: replace-cross  Abstract: Dense retrieval enables effective semantic matching, yet the semantic factors underlying its relevance scores remain opaque. Sparse autoencoders (SAEs) support feature-level explanations by decomposing dense embeddings into sparse semantic features. However, the document semantics underlying a retriever's relevance scores may not be sufficiently explicit in the embeddings to be recovered as interpretable SAE features. We introduce Xetrieval, a framework for explaining dense retrieval through feature elicitation. Feature Elicitors make retrieval-relevant aspects of document meaning more explicit through auxiliary views, facilitating their identification as interpretable SAE features. A shared sparse dictionary learns from these varied representations to express query--document relations through describable features. We assess their influence on relevance scores through interventions on the original document embeddings. Experimen
    
[^33]: 一种支持任意距离、内存高效的分布式近似最近邻搜索算法

    A Memory-Efficient Distributed Algorithm for Approximate Nearest Neighbour Search with Arbitrary Distances

    [https://arxiv.org/abs/2405.13795](https://arxiv.org/abs/2405.13795)

    本文提出分布式近似最近邻搜索算法PDASC，其索引设计同时支持任意不相似性度量、原生分布式执行以及内存受限环境下的高效运行，且无需专用硬件。

    

    近似最近邻（ANN）搜索已成为现代数据密集型应用中的核心任务，尤其是在处理大规模、异构或高维数据集时。然而，许多现有的ANN方法在这类场景下表现不佳，原因要么是它们依赖于度量性假设，要么是其索引策略并不适合分布式环境或内存资源受限的设置。本工作提出了PDASC（Parametrizable Distributed Approximate Similarity Search with Clustering，可参数化的基于聚类的分布式近似相似性搜索），这是一种分布式ANN搜索算法，其索引设计同时解决了ANN文献中通常被分开处理的三个问题：支持任意的不相似性函数、原生分布式执行，以及在内存受限环境下的高效运行，且无需依赖专用硬件加速器或昂贵的内存资源。PDASC构建了一个分布式层次化索引……

    arXiv:2405.13795v5 Announce Type: replace  Abstract: Approximate nearest neighbour (ANN) search has become a central task in modern data-intensive applications, particularly when operating on large-scale, heterogeneous, or high-dimensional datasets. However, many existing ANN methods struggle in such scenarios, either because they rely on metric assumptions or because their indexing strategies are not well suited to distributed environments or to settings with constrained memory resources. This work introduces PDASC (Parametrizable Distributed Approximate Similarity Search with Clustering), a distributed ANN search algorithm whose index design jointly addresses three concerns commonly treated separately in the ANN literature: support for arbitrary dissimilarity functions, native distributed execution, and efficient operation under constrained memory environments, avoiding reliance on specialised hardware accelerators or costly memory resources. PDASC builds a distributed hierarchical i
    

