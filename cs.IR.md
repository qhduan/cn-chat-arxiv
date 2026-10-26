# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Separating Thinking from Spelling for Efficient Generative Recommendation](https://arxiv.org/abs/2610.32980) | 该论文提出Deco方法，将阅读用户历史与书写物品标识符的过程解耦——骨干网络将历史一次性压缩为紧凑的“下一物品概念”，再由轻量的KV缓存解码器独立执行整个束搜索，从而大幅降低生成式推荐中随骨干规模、SID深度和候选数量增长的解码开销。 |
| [^2] | [Knowledge Distillation across Item and Code Spaces for Parallel Generative Recommendation](https://arxiv.org/abs/2610.32943) | 该论文提出面向并行语义ID推荐器的两个互补知识蒸馏目标（如物品分布匹配IDM），将序列推荐器的物品级预测分布知识跨空间迁移到代码空间，弥补了硬标签监督缺乏备选物品分布信息的不足。 |
| [^3] | [Every Edge Needs a Quote: Building a Quote-Grounded Knowledge Graph of Healthcare Software Adoption with Language-Model Agents](https://arxiv.org/abs/2610.32781) | 该论文提出一个从开放网络构建医疗软件客户与集成关系知识图谱的流水线，要求每条边都必须有源页面的逐字引用支持，从而确保语言模型智能体提取的信息可验证、可追溯。 |
| [^4] | [From archive to resource: bringing legacy data back to life at scale with large language models](https://arxiv.org/abs/2610.32616) | 本研究由两个行业团队独立开发基于大语言模型的提取流水线，从制药公司遗留的临床前研究报告档案中大规模恢复禁食、麻醉等 rarely captured 的研究背景描述符，并返回带页码引用的逐字证据，对400多份文档的评估显示与专家参考值高度一致，有效实现了历史数据的再利用。 |
| [^5] | [UAE GovTrust Scope Filtering Gains and Routing Failures in Bilingual Public Service Retrieval](https://arxiv.org/abs/2610.32608) | 本文提出双语公共服务检索诊断集 UAE GovTrust，证明基于角色范围过滤可将首条结果相关率从 65.3% 大幅提升至 95.8%，但现有路由器在扰动场景下几乎完全失效（168 次扰动中仅 2 次正确）。 |
| [^6] | [Agentic Memory Management Is the Next Frontier of Information Retrieval](https://arxiv.org/abs/2610.32464) | 本文提出智能体记忆管理（AMM）是信息检索研究的自然延伸与下一个前沿方向，并为此构建了系统级框架，同时将记忆对象创新性地划分为常量、函数和算子三类计算分类法。 |
| [^7] | [Hierarchical GTV Estimation: Bridging the Gap between Container-Level Ranking and Item-Level Conversion](https://arxiv.org/abs/2610.32420) | 针对电商容器级排序与商品级交易之间的粒度不匹配问题，提出层级式GTV估计框架HGE，通过混合集合感知编码器与代理标签学习策略，解决GTV估计中的标签方差、标签稀疏和商品异质性等挑战。 |
| [^8] | [SkillRep: Characterizing and Understanding the Representation Design Space of Agent Skill Retrieval](https://arxiv.org/abs/2610.32336) | 该论文首次将五种表示选择（编码器微调、查询侧与文档侧扩展、背景投影和词法通道）作为统一检索器 SkillRep 的模块进行联合评估，系统刻画了智能体技能检索的表示设计空间，并在多个基准上取得最优性能。 |
| [^9] | [System One models Are Not Better Rerankers](https://arxiv.org/abs/2610.32318) | 本研究通过在0.6B至27B参数的六个Qwen骨干模型和十个TREC DL与BEIR数据集上的系统对比，证明决策微调的System One模型并不比普通重排序微调模型是更好的重排序器——重排序微调在0.6B至9B全面胜出5至29个nDCG@10点，27B时两者持平，且9B以下决策微调相比零样本询问未调优模型并无增益。 |
| [^10] | [PersonalQuery-Benchmark:Evaluating Retrieval Robustness to Personalized User Expression Differences](https://arxiv.org/abs/2610.32164) | 该论文提出了首个系统性基准 PQB，通过从用户历史文本中建模个性化句法表达和书写错误构建 6,516 个查询实例，用以评估检索系统对同一需求下不同用户个性化表达差异的鲁棒性。 |
| [^11] | [From Item Scores to Semantic Tokens: Test-Time Collaborative Adaptation for Generative Recommendation](https://arxiv.org/abs/2610.32098) | 提出TCBSA方法，在测试时利用独立判别式推荐器的互补评分来引导冻结生成式推荐器的束搜索前缀决策，无需重新训练即可缓解新兴兴趣被剪枝遗漏的问题。 |
| [^12] | [Scaling Graph Query Processing: Focused Exploration on Partitions Using Load Balancing](https://arxiv.org/abs/2610.32015) | 本文提出将图划分与分布式范式相结合，通过负载均衡驱动的聚焦探索，在任意规模的图上实现表达能力丰富的图查询的高效、可扩展处理。 |
| [^13] | [A Model-Agnostic Correction Framework for Short-Term Building Load Forecasting](https://arxiv.org/abs/2610.31959) | 提出了一种模型无关的天气-负荷检索增强校正框架WeL-RAC，通过在预测时基于近期负荷状态和天气预报检索相似历史案例，并聚合其观测到的未来负荷轨迹来校正初始预测，从而提升短期建筑负荷预测在工况快速变化下的可靠性。 |
| [^14] | [MICE to Meet You: From Re-ranking Only to End-to-End Retrieval](https://arxiv.org/abs/2610.31932) | 本文提出MICE-P，首个基于MICE架构的端到端检索管线，通过PLAID索引上下文化的文档token表示，解决了MICE缺乏实用文档表示存储方案的局限，从而将MICE从仅能用于重排序扩展到完整的检索流程。 |
| [^15] | [Mining Frames: Association Rule Mining as an Unsupervised Framework for News Framing Detection](https://arxiv.org/abs/2610.31874) | 该论文提出将关联规则挖掘（ARM）作为无监督、无需标注的新闻框架检测新框架，通过对加沙战争首年25,512篇新闻文章的分析验证了其无需预定义分类体系即可归纳发现可解释框架模式的能力。 |
| [^16] | [Machine-Readable Operator Manuals over the ISO 11783 Network for Offline, Multilingual LLM Assistance](https://arxiv.org/abs/2610.31863) | 本文提出在ISO 11783网络上对操作手册进行语义同步，使拖拉机侧的离线多语言LLM助手能够在31至89秒内获取完整的机具文档，传输时间与操作员首次连接机具时等待对象池上传的时间相当。 |
| [^17] | [What the Query Remembers: Query Abstraction Governs Lexical and Dense Retrieval of Classical Arabic Poetry](https://arxiv.org/abs/2610.31821) | 该论文发现，在古典阿拉伯语诗歌检索中，查询保留诗歌原文措辞的程度（即查询的抽象程度）决定了检索器的优劣——记忆型查询下BM25领先所有稠密模型，而抽象查询下稠密编码器胜出，且BM25对措辞变化的敏感度远高于稠密模型。 |
| [^18] | [Lean Finder V2: Semantic Search for Mathlib That Disambiguates User Queries](https://arxiv.org/abs/2610.30843) | Lean Finder V2 通过互补的模型级对抗训练与输入级查询重写，实现了能够消除歧义的 Mathlib 语义搜索，有效解决了短而模糊查询的检索难题。 |
| [^19] | [Driving Video Retrieval for Complex Queries with Structured Grounding](https://arxiv.org/abs/2606.09109) | 提出STRIVE-D数据校准检索框架，通过弱标注视频评估并自适应调整查询规则，将校准后的规则分数与视觉-语言及关键词检索信号融合，从而在驾驶视频中可靠检索加塞、急刹等复杂动态事件。 |
| [^20] | [Deep learning enables urban change profiling through alignment of historical maps](https://arxiv.org/abs/2602.02154) | 提出了一种基于深度学习的模块化框架，通过空间地图对齐、多时相目标提取和变化估计，实现了从大规模历史地图集合中进行细粒度、定量的城市变化分析。 |

# 详细

[^1]: 将思考与拼写分离以实现高效的生成式推荐

    Separating Thinking from Spelling for Efficient Generative Recommendation

    [https://arxiv.org/abs/2610.32980](https://arxiv.org/abs/2610.32980)

    该论文提出Deco方法，将阅读用户历史与书写物品标识符的过程解耦——骨干网络将历史一次性压缩为紧凑的“下一物品概念”，再由轻量的KV缓存解码器独立执行整个束搜索，从而大幅降低生成式推荐中随骨干规模、SID深度和候选数量增长的解码开销。

    

    生成式推荐器通过在Trie树约束的束搜索下逐码书写物品的语义标识符（SID）来检索物品，这一方法目前已部署于大型网络平台。标准的自回归接口为SID长度K付出了双倍代价：T个物品的历史序列以TK个代码token的形式进入骨干网络，且每一次束搜索步骤都要经过骨干网络，因此为每个用户返回B个候选需要O(BK)个骨干网络解码token。第二项开销通常占据主导地位，并随骨干网络规模、SID深度和候选预算的增长而增长。我们提出了Deco（从概念解码，decoding from concepts），将阅读历史与书写标识符相分离。骨干网络将每个历史物品读取为一个融合token，并将历史一次性压缩为紧凑的“下一物品概念”；一个轻量的、带KV缓存的解码器以该概念和已生成前缀为条件运行整个束搜索，而标识符、Trie树以及自回归分解保持不变。

    arXiv:2610.32980v1 Announce Type: new  Abstract: Generative recommenders retrieve items by writing their semantic IDs (SIDs) code by code under trie-constrained beam search, an approach now deployed on large web platforms. Standard autoregressive interfaces pay for the SID length K twice: a history of T items enters the backbone as TK code tokens, and every beam-search step runs through the backbone, so returning B candidates costs O(BK) backbone decode tokens per user. The second cost often dominates, and it grows with the backbone, the SID depth and the candidate budget.   We introduce Deco (decoding from concepts), which separates reading the history from writing the identifier. The backbone reads each history item as one fused token and condenses the history once into a compact next-item concept; a light KV-cached decoder conditioned on the concept and the generated prefix runs the entire beam search, while the identifiers, the trie and the autoregressive factorization stay unchang
    
[^2]: 面向并行生成式推荐的物品空间与代码空间知识蒸馏

    Knowledge Distillation across Item and Code Spaces for Parallel Generative Recommendation

    [https://arxiv.org/abs/2610.32943](https://arxiv.org/abs/2610.32943)

    该论文提出面向并行语义ID推荐器的两个互补知识蒸馏目标（如物品分布匹配IDM），将序列推荐器的物品级预测分布知识跨空间迁移到代码空间，弥补了硬标签监督缺乏备选物品分布信息的不足。

    

    基于语义ID的生成式推荐将每个物品表示为从其语义表示中导出的离散代码序列，并通过预测这些代码来生成推荐。现有的语义ID推荐器通常使用观测到的下一个物品的硬标签进行训练。尽管这种监督方式能够标识目标物品，但它无法提供在同一用户历史条件下其他备选物品的显式分布信息。序列推荐器可以提供这种物品级别的预测知识，但由于教师模型和学生模型分别工作在物品空间和代码空间，将其迁移到语义ID生成中具有挑战性。我们为并行语义ID推荐器提出了两个互补的知识蒸馏目标，这类推荐器无需自回归前缀条件即可同时预测所有代码位置。物品分布匹配（IDM）从编目层面重构物品分布……（原文摘要在此处截断）

    arXiv:2610.32943v1 Announce Type: new  Abstract: Generative recommendation with semantic IDs represents each item as a sequence of discrete codes derived from its semantic representation and predicts these codes to generate recommendations. Existing semantic-ID recommenders are typically trained with hard labels for the observed next item. Although such supervision identifies the target, it provides no explicit distributional information about alternative items conditioned on the same user history. Sequential recommenders can supply this item-level predictive knowledge, but transferring it to semantic-ID generation is challenging because the teacher and student operate in item and code spaces, respectively. We propose two complementary knowledge-distillation objectives for parallel semantic-ID recommenders, which predict all code positions simultaneously without autoregressive prefix conditioning. Item Distribution Matching (IDM) reconstructs a catalogue-level item distribution from th
    
[^3]: 每条边都需要引用：利用语言模型智能体构建以引用为依据的医疗软件采用知识图谱

    Every Edge Needs a Quote: Building a Quote-Grounded Knowledge Graph of Healthcare Software Adoption with Language-Model Agents

    [https://arxiv.org/abs/2610.32781](https://arxiv.org/abs/2610.32781)

    该论文提出一个从开放网络构建医疗软件客户与集成关系知识图谱的流水线，要求每条边都必须有源页面的逐字引用支持，从而确保语言模型智能体提取的信息可验证、可追溯。

    

    医院需要定期采购临床医生和患者赖以使用的软件。由于这些采购成本高昂、难以逆转，且可能影响患者护理，采购方需要寻找证据来确认哪些同行机构已经在使用某款产品，以及该产品与哪些系统相连接。这些证据是公开的，但分散在供应商网站、新闻稿和第三方页面中。语言模型智能体现在能够以低成本收集这些信息，但提取出的声明的正确性难以核实，而错误列出的客户信息可能会误导采购方并损害机构声誉。我们提出了一个从开放网络构建客户关系与集成关系知识图谱的流水线，遵循一条规则：每条边都必须由来自其源页面的逐字引用来支持。页面来源于对每个供应商网站的确定性爬取，以及一个在预算限制内搜索网络其余部分的ReAct智能体。一个宽松的提取器提出带有引用的关系，随后……

    arXiv:2610.32781v1 Announce Type: cross  Abstract: Hospitals regularly buy software that clinicians and patients depend on. Because these purchases are costly, hard to reverse and can affect patient care, buyers look for evidence of which peer institutions already use a product and which systems it connects to. This evidence is public but scattered across vendor websites, press releases and third-party pages. Language-model agents can now collect it at low cost, but the correctness of the extracted claims is difficult to establish, and a wrongly listed customer can mislead buyers and misrepresent an organisation. We present a pipeline that builds a knowledge graph of customer and integration relations from the open web under one rule: every edge must be supported by a verbatim quote from its source page. Pages come from a deterministic crawl of each vendor's site and a budgeted ReAct agent that searches the rest of the web. A permissive extractor proposes relations with quotes, and a l
    
[^4]: 从档案到资源：利用大语言模型大规模盘活遗留数据

    From archive to resource: bringing legacy data back to life at scale with large language models

    [https://arxiv.org/abs/2610.32616](https://arxiv.org/abs/2610.32616)

    本研究由两个行业团队独立开发基于大语言模型的提取流水线，从制药公司遗留的临床前研究报告档案中大规模恢复禁食、麻醉等 rarely captured 的研究背景描述符，并返回带页码引用的逐字证据，对400多份文档的评估显示与专家参考值高度一致，有效实现了历史数据的再利用。

    

    制药公司拥有大量遗留的临床前研究报告档案，其中包含研究背景细节，如禁食、麻醉和实验室仪器等信息。这些信息很少以结构化数据格式保存，但它们可能与临床病理学参数值相关联，并限制历史数据的重复利用。这项工作展示了实用且可扩展的解决方案：由两个行业团队独立开发的基于大语言模型（LLM）的提取流水线，能够从异构的报告文本中恢复此类描述符。这些流水线可以识别和规范化元数据字段，并返回带有页码和章节引用的逐字证据，从而加速人工验证和迭代改进。对超过400份文档对照专家整理的参考值进行评估，结果显示在广泛的元数据字段范围内具有高度一致性。对于报告方式一致的项目表现最佳，而对于（摘要在此处截断）

    arXiv:2610.32616v1 Announce Type: new  Abstract: Pharmaceutical companies maintain vast archives of legacy preclinical study reports that contain study-context details, such as fasting, anesthesia, and laboratory instrumentation, that are rarely captured in structured data formats, yet they can correlate with clinical pathology parameters values and limit the reuse of historical data. This work demonstrates practical, scalable solutions to recover such descriptors from heterogeneous report texts using large language model (LLM)-based extraction pipelines developed independently by two industry teams. The pipelines identify and normalize metadata fields and return verbatim evidence with page and section reference, hasting human verification and iterative refinement. Evaluation of over 400 documents against expert-curated reference values showed high agreement across a broad set of metadata fields. Performance was strongest for consistently reported items and lower for descriptors subjec
    
[^5]: 阿联酋 GovTrust：双语公共服务检索中的范围过滤收益与路由失败

    UAE GovTrust Scope Filtering Gains and Routing Failures in Bilingual Public Service Retrieval

    [https://arxiv.org/abs/2610.32608](https://arxiv.org/abs/2610.32608)

    本文提出双语公共服务检索诊断集 UAE GovTrust，证明基于角色范围过滤可将首条结果相关率从 65.3% 大幅提升至 95.8%，但现有路由器在扰动场景下几乎完全失效（168 次扰动中仅 2 次正确）。

    

    公共服务查询可能同时提及多个地点，而其中只有一个地点负责所请求的服务。我们在 UAE GovTrust 中研究这一区别，这是一个受控的英语-阿拉伯语检索诊断集，包含 22 篇人工撰写的服务摘要，以及 50 个场景下的 110 个查询变体。研究对比了十二种方法，涵盖词法与稠密排序、排名融合、范围过滤、多语言交叉编码器以及直接服务查找。在 72 个带有双语证据的可匹配可回答查询上，采用角色过滤的倒数排名融合方法为 69 个查询（95.8%）检索到相关的首条结果，而未过滤的融合仅为 47 个（65.3%）；过滤后的稠密检索和直接查找各达到 68 个（94.4%）。在仅有英语证据的情况下，排除零分词法候选使过滤融合的正确首条命中数从 52 提升到 64。然而，保持不变的路由器在 168 次审计后扰动（涵盖改述、无关事实等）中，仅正确选择操作和范围 2 次（摘要在此处被截断）。

    arXiv:2610.32608v1 Announce Type: new  Abstract: Public-service queries can mention several locations while only one controls the requested service. We study this distinction in UAE GovTrust, a controlled English-Arabic retrieval diagnostic comprising 22 authored service summaries and 50 scenarios with 110 query variants. Twelve methods separate lexical and dense ranking, rank fusion, scope filtering, a multilingual cross-encoder, and direct service lookup. On 72 matched answerable queries with bilingual evidence, role-filtered reciprocal rank fusion retrieves a relevant first result for 69 queries (95.8%), versus 47 (65.3%) for unfiltered fusion; filtered dense retrieval and direct lookup each achieve 68 (94.4%). With English-only evidence, excluding zero-score lexical candidates raises filtered fusion from 52 to 64 correct first hits. However, the unchanged router selects the correct action and scope for only 2 of 168 post-audit perturbations covering paraphrases, irrelevant facts, n
    
[^6]: 智能体记忆管理是信息检索的下一个前沿

    Agentic Memory Management Is the Next Frontier of Information Retrieval

    [https://arxiv.org/abs/2610.32464](https://arxiv.org/abs/2610.32464)

    本文提出智能体记忆管理（AMM）是信息检索研究的自然延伸与下一个前沿方向，并为此构建了系统级框架，同时将记忆对象创新性地划分为常量、函数和算子三类计算分类法。

    

    信息检索长期以来一直专注于连接人与完成任务所需的信息。近来，随着越来越多的AI智能体代表用户消费、保留并基于信息采取行动，这一含义发生了巨大变化。因此，信息管理现在也以记忆管理的形式发生在智能体内部。在这篇观点论文中，我们认为智能体记忆管理（AMM）——研究智能体记忆中信息的结构、分析、组织、存储、搜索、检索与使用——是信息检索研究的自然延伸，也是信息检索学界能够做出独特贡献的领域。具体而言，我们首先提出了一个AMM的系统级框架，并引入了一种记忆对象的计算分类法，将记忆对象划分为常量、函数和算子三类，每类具有不同的检索语义。然后，我们……

    arXiv:2610.32464v1 Announce Type: new  Abstract: Information retrieval has long been focusing on connecting people with information they need to complete their tasks. Recently, the meaning of this has changed dramatically as more and more AI agents now consume, retain, and act on information on a user's behalf. Consequently, information management now also takes place \emph{inside} the agent in the form of memory management. In this perspective paper, we argue that \emph{agentic memory management} (AMM), which studies the structure, analysis, organization, storage, searching, retrieval, and \emph{use} of information in an intelligent agent's memory, is a natural extension of IR research and an area in which the IR community can make distinctive contributions. Specifically, we first present a system-level framework for AMM and introduce a computational taxonomy of memory objects as \emph{constants}, \emph{functions}, and \emph{operators}, each with distinct retrieval semantics. Then, we
    
[^7]: 层级式GTV估计：弥合容器级排序与商品级转化之间的鸿沟

    Hierarchical GTV Estimation: Bridging the Gap between Container-Level Ranking and Item-Level Conversion

    [https://arxiv.org/abs/2610.32420](https://arxiv.org/abs/2610.32420)

    针对电商容器级排序与商品级交易之间的粒度不匹配问题，提出层级式GTV估计框架HGE，通过混合集合感知编码器与代理标签学习策略，解决GTV估计中的标签方差、标签稀疏和商品异质性等挑战。

    

    新兴电商平台，如直播和线上到线下（O2O）服务，本质上运行在一种“基于容器”的展示范式之下。在这种场景中，存在一个关键的粒度不匹配问题：平台在容器层面（如直播间或店铺）进行排序和估计，而实际交易却发生在这些容器内部的商品层面。这种结构性差异使得成交总额（GTV）的估计变得复杂，并带来了独特的挑战，包括高标签方差、严重的标签稀疏性以及动态的商品异质性。为应对这些挑战，我们提出了一个名为层级式GTV估计的新颖框架。HGE采用混合集合感知编码器来建模商品的动态组成与特征交互。为缓解标签方差问题，我们引入了一种代理标签学习策略，将点击后GTV分解为购买概率、数……

    arXiv:2610.32420v1 Announce Type: new  Abstract: Emerging e-commerce platforms, such as live-streaming and Online-to-Offline (O2O) services, inherently operate under a "Container-based" display paradigm. In this setting, a critical granularity mismatch exists: the platform performs ranking and estimation at the container level (e.g., livestream rooms or stores), whereas the actual transactions occur at the item level within these containers. This structural discrepancy complicates Gross Transaction Value (GTV) estimation, introducing unique challenges including high label variance, severe label sparsity, and dynamic item heterogeneity. To address these challenges, we propose a novel framework named Hierarchical GTV Estimation (HGE). HGE employs a Hybrid Set-Aware Encoder (HSAE) to model the dynamic composition of items and feature interactions. To mitigate label variance, we introduce a Proxy-Label Learning (PLL) strategy that decomposes post-click GTV into purchase probability, quanti
    
[^8]: SkillRep：刻画与理解智能体技能检索的表示设计空间

    SkillRep: Characterizing and Understanding the Representation Design Space of Agent Skill Retrieval

    [https://arxiv.org/abs/2610.32336](https://arxiv.org/abs/2610.32336)

    该论文首次将五种表示选择（编码器微调、查询侧与文档侧扩展、背景投影和词法通道）作为统一检索器 SkillRep 的模块进行联合评估，系统刻画了智能体技能检索的表示设计空间，并在多个基准上取得最优性能。

    

    从庞大的技能库中检索出正确的技能对 LLM 智能体至关重要，然而请求与技能的撰写目的和用词各不相同。现有的检索器各自仅通过一到两种表示选择来缩小这种“请求-技能”差距，因此尚不清楚这些方法结合使用时其收益是否依然成立。此前没有工作测量这一差距本身，因此也不清楚每种选择分别缩小了差距的哪一部分。我们通过五种选择——编码器微调、查询侧与文档侧扩展、背景投影以及词法通道——定义了一个表示设计空间，并首次将它们作为一个统一检索器 SkillRep 的模块进行联合评估。借助 0.6B 规模的编码器和 LLM 扩展，SkillRep 在 SRA-Bench 上使用官方技能池时，MRR@10 相比最强的同规模检索器提升了 7.02 分；使用我们基于公共注册库构建的、包含 111,247 个技能的 UniPool 时提升了 8.79 分，并在 SkillsBench 上取得最佳表现。每种选择的价值……（原文在此处截断）

    arXiv:2610.32336v1 Announce Type: new  Abstract: Retrieving the right skill from a large library is essential for LLM agents, yet requests and skills are written for different purposes and in different words. Existing retrievers narrow this request-skill gap with one or two representation choices each, so it is unknown whether their gains hold in combination. No prior work measures the gap itself, so it is also unclear which part of it each choice narrows. We define a representation design space by five choices -- encoder fine-tuning, query-side and document-side expansion, background projection and a lexical channel -- and evaluate them together for the first time, as modules of one retriever, SkillRep. With a 0.6B encoder and LLM expansion, SkillRep improves MRR@10 on SRA-Bench over the strongest same-size retriever by 7.02 points with the official skill pool and 8.79 with UniPool, a 111,247-skill pool we build from public registries, and is best on SkillsBench. The value of each cho
    
[^9]: 系统一模型并非更好的重排序器

    System One models Are Not Better Rerankers

    [https://arxiv.org/abs/2610.32318](https://arxiv.org/abs/2610.32318)

    本研究通过在0.6B至27B参数的六个Qwen骨干模型和十个TREC DL与BEIR数据集上的系统对比，证明决策微调的System One模型并不比普通重排序微调模型是更好的重排序器——重排序微调在0.6B至9B全面胜出5至29个nDCG@10点，27B时两者持平，且9B以下决策微调相比零样本询问未调优模型并无增益。

    

    系统一模型，或称决策模型，通过对输入问题给出一个经过校准的概率而非生成文本来进行回答。在2026年9月首次发布后的几周内，它们便被用作重排序器。我们测试了其背后的决策微调是否比同一模型的普通重排序微调能产生更好的重排序器。针对从0.6B到27B参数的六个Qwen骨干模型以及十个TREC DL和BEIR数据集，我们将零样本提问的未调优模型与其决策微调版本和重排序微调版本进行比较，其中两个微调版本由我们自行训练。在这项工作中，我们得出结论：决策微调永远不会造就更好的重排序器。在0.6B至9B参数范围内，重排序微调在每个骨干模型上均以5到29个点的nDCG@10优势胜出（包括在零样本数据集上），而在27B时两者打平。在9B及以下，决策微调相比直接零样本询问未调优模型没有任何额外收益。此外，我们发现决策模型的校准优势……（摘要在此处截断）

    arXiv:2610.32318v1 Announce Type: new  Abstract: System One models, or decision models, answer a question about an input with a calibrated probability instead of generated text. Within weeks of the first release, in September 2026, they were used as rerankers. We test whether the decision fine-tuning behind them makes a better reranker than plain reranker fine-tuning of the same model. For six Qwen backbones from 0.6B to 27B parameters and ten TREC DL and BEIR collections, we compare the untuned model, asked zero-shot, with its decision fine-tunes and its reranker fine-tunes, two of which we train ourselves. In our work, we draw the conclusion that decision fine-tuning never makes the better reranker. From 0.6B to 9B, reranker fine-tuning wins on every backbone by 5 to 29 points nDCG@10, also on zero-shot collections, and at 27B the two tie. Up to 9B, decision fine-tuning adds nothing over simply asking the untuned model. Moreover, we find that the calibration advantage of decision mod
    
[^10]: PersonalQuery-Benchmark：评估检索系统对个性化用户表达差异的鲁棒性

    PersonalQuery-Benchmark:Evaluating Retrieval Robustness to Personalized User Expression Differences

    [https://arxiv.org/abs/2610.32164](https://arxiv.org/abs/2610.32164)

    该论文提出了首个系统性基准 PQB，通过从用户历史文本中建模个性化句法表达和书写错误构建 6,516 个查询实例，用以评估检索系统对同一需求下不同用户个性化表达差异的鲁棒性。

    

    现有的个性化信息检索方法主要通过用户偏好和查询意图来建模用户需求的差异，而很少关注不同用户在表达同一检索需求时产生的个性化表达差异。现有的真实查询数据通常缺乏稳定的用户身份、长期的表达历史以及意图匹配的对应项，因而难以系统地评估个性化表达差异对检索结果的影响。为此，我们提出了 PersonalQuery-Benchmark（PQB）。PQB 从用户的历史文本中建模个性化句法表达和个性化书写错误，构建了个性化句法表达查询和个性化书写错误查询，并通过大语言模型评估和独立的人类评价来验证基准的质量。PQB 涵盖三个 Amazon 产品领域，共包含 6,516 个查询实例。

    arXiv:2610.32164v1 Announce Type: new  Abstract: Existing personalized information retrieval methods mainly model differences in user needs through user preferences and query intent, while paying little attention to the personalized expression differences that arise when different users express the same retrieval need. Existing real query data typically lack stable user identities, long-term expression histories, and intent-matched counterparts, making it difficult to systematically evaluate the impact of personalized expression differences on retrieval outcomes. To this end, we propose PersonalQuery-Benchmark (PQB). PQB models personalized syntactic expression and personalized writing errors from users' historical texts, constructs personalized syntactic expression queries and personalized writing-error queries, and validates benchmark quality through LLM assessment and independent human evaluation. PQB covers three Amazon product domains with 6,516 query instances in total, and is ev
    
[^11]: 从物品评分到语义标记：面向生成式推荐的测试时协同自适应

    From Item Scores to Semantic Tokens: Test-Time Collaborative Adaptation for Generative Recommendation

    [https://arxiv.org/abs/2610.32098](https://arxiv.org/abs/2610.32098)

    提出TCBSA方法，在测试时利用独立判别式推荐器的互补评分来引导冻结生成式推荐器的束搜索前缀决策，无需重新训练即可缓解新兴兴趣被剪枝遗漏的问题。

    

    使用语义标识符的生成式推荐器通过连续的标记决策来预测物品，但束搜索可能会不可逆地剪枝掉有前景的物品路径。当少量近期交互表明一种新兴兴趣、而早期行为偏向既有偏好时，这一局限性尤为严重。然而，一个独立训练的判别式推荐器可能基于相同的交互历史对相同的物品给出高分排名。这类互补的评分在测试时即可获得，从而提供了一种无需重新训练任何模型即可适配冻结生成器的途径。挑战在于如何在剪枝发生之前利用这些评分：合并已完成的推荐列表无法改变早期的束搜索决策，而针对完整物品的评分也无法直接指导在多个物品共享的语义前缀之间做出选择。我们提出了测试时协同束搜索自适应（TCBSA），这是一种训练……

    arXiv:2610.32098v1 Announce Type: new  Abstract: Generative recommenders using semantic identifiers predict items through successive token decisions, but beam search may irreversibly prune promising item paths. This limitation is particularly consequential when a few recent interactions signal an emerging interest while earlier behavior favors established preferences. Yet an independently trained discriminative recommender may rank the same items highly from the same interaction history. Such complementary scores are available at test time, offering a way to adapt a frozen generator without retraining either model. The challenge is to use these scores before pruning: combining completed recommendation lists cannot alter earlier beam decisions, while scores over complete items cannot directly inform choices among semantic prefixes shared by multiple items. We propose \textbf{T}est-Time \textbf{C}ollaborative \textbf{B}eam \textbf{S}earch \textbf{A}daptation (\textbf{TCBSA}), a training-
    
[^12]: 扩展图查询处理：基于负载均衡的分区聚焦探索

    Scaling Graph Query Processing: Focused Exploration on Partitions Using Load Balancing

    [https://arxiv.org/abs/2610.32015](https://arxiv.org/abs/2610.32015)

    本文提出将图划分与分布式范式相结合，通过负载均衡驱动的聚焦探索，在任意规模的图上实现表达能力丰富的图查询的高效、可扩展处理。

    

    本文的重点是查询处理而非搜索（无论是近似的还是精确的）。目标是找出全部或k个答案，并且查询的表达能力（与通常较为简单的搜索表达式相比）被视为重要因素。尽管SQL查询处理在过去几十年中已经成熟，但针对图数据的查询处理却未受到同等的关注。直到最近，Neo4J等系统才开始大规模使用图进行数据表示，但它们的方法和算法仍然是专有的。本文致力于解决图查询的回答问题及其在任意规模图上的高效处理。我们提出对图进行划分，并利用分布式范式来实现高效性和可扩展性。我们的方法以一个子图作为查询输入，其中包括通配符和比较操作。我们的分区方法利用聚焦探索……（原文在此处截断）

    arXiv:2610.32015v1 Announce Type: cross  Abstract: The focus of this paper is on query processing and not search (approximate or exact). The goal is to find all or k answers and the expressiveness of a query (as opposed to a search expression which is typically simpler) is considered important. Although SQL query processing has matured over the last several decades, query processing over graph data has not received the same attention. Only recently, systems, such as Neo4J, and others have started using graphs for data representation on a large scale. Their approaches and algorithms remain proprietary.   In this paper, we address the problem of answering graph queries and their efficient processing on arbitrary-sized graphs. We propose the partitioning of graphs and the use of the distributed paradigm to achieve efficiency and scalability. Our approach takes a subgraph as query input including wild cards and comparison operations. Our partitioned approach leverages focused exploration a
    
[^13]: 一种用于短期建筑负荷预测的模型无关校正框架

    A Model-Agnostic Correction Framework for Short-Term Building Load Forecasting

    [https://arxiv.org/abs/2610.31959](https://arxiv.org/abs/2610.31959)

    提出了一种模型无关的天气-负荷检索增强校正框架WeL-RAC，通过在预测时基于近期负荷状态和天气预报检索相似历史案例，并聚合其观测到的未来负荷轨迹来校正初始预测，从而提升短期建筑负荷预测在工况快速变化下的可靠性。

    

    短期建筑负荷预测仍然具有挑战性，因为负荷模式会随着天气和运行条件的变化而快速改变。当当前运行状态在训练数据中罕见或代表性不足时，即使历史记录中其他地方存在类似情况，数据驱动的预测模型也可能变得不太可靠。检索增强时间序列预测的最新进展提供了一种有前景的方法，即通过在推理时直接重用相关的历史案例，使预测能够适应不断变化的条件。受这一思想的启发，我们提出了WeL-RAC，一种模型无关的天气-负荷检索增强校正框架，该框架在预测时显式地检索此类证据。WeL-RAC基于近期负荷状态和与预测时域对齐的天气预报，检索与当前条件相似的历史案例，然后聚合这些案例观测到的未来负荷轨迹，用于校正初始预测结果。

    arXiv:2610.31959v1 Announce Type: new  Abstract: Short-term building load forecasting remains challenging because load patterns can change rapidly with weather and operating conditions. Data-driven forecasting models may become less reliable when the current operating regime is rare or poorly represented during training, even when similar situations exist elsewhere in the historical record. Recent advances in retrieval-augmented time series forecasting offer a promising way to adapt predictions to changing conditions by directly reusing relevant historical cases at inference time. Motivated by this idea, we propose WeL-RAC, a model-agnostic Weather-Load Retrieval-Augmented Correction framework that explicitly retrieves such evidence at prediction time. WeL-RAC retrieves historical cases that are similar to the current condition based on both recent load states and horizon-aligned weather forecasts. Their observed future load trajectories are then aggregated and used to correct the init
    
[^14]: MICE很高兴见到你：从仅用于重排序到端到端检索

    MICE to Meet You: From Re-ranking Only to End-to-End Retrieval

    [https://arxiv.org/abs/2610.31932](https://arxiv.org/abs/2610.31932)

    本文提出MICE-P，首个基于MICE架构的端到端检索管线，通过PLAID索引上下文化的文档token表示，解决了MICE缺乏实用文档表示存储方案的局限，从而将MICE从仅能用于重排序扩展到完整的检索流程。

    

    晚期交互模型通过允许预先计算文档表示并对其进行高效索引，为神经检索提供了最先进的效率与效果权衡。然而，其相对简单的交互机制（通常是MaxSim）使得其效果 compared to 交叉编码器有所受限。MICE是近期提出的一种重排序架构，旨在通过富有表现力的基于Transformer的评分获得类似交叉编码器的效果，同时借助文档侧预计算来降低推理时的计算开销。然而，此前并未探索出存储这些文档表示的实用方法。本文针对这一局限，提出了MICE-P——首个基于MICE的端到端检索管线，该管线基于PLAID对上下文化的文档token表示进行索引。我们通过使用Ettin Family在匹配的实验条件下训练可比模型，对MICE、ColBERT与交叉编码器进行了受控比较……

    arXiv:2610.31932v1 Announce Type: new  Abstract: Late-interaction models offer state-of-the-art efficiency-effectiveness trade-offs for neural retrieval by enabling document representations to be pre-computed and efficiently indexed. Yet, their relatively simple interaction mechanisms, typically MaxSim, can limit their effectiveness compared to cross-encoders. MICE, a recently introduced re-ranking architecture, aims to combine cross-encoder-like effectiveness through expressive transformer-based scoring with reduced inference-time computation through document-side pre-computation. However, no practical way to store these document representations was explored. We address this limitation here and introduce MICE-P, the first end-to-end retrieval pipeline with MICE, based on PLAID to index the contextualized document token representations. We conduct a controlled comparison of MICE, ColBERT and cross-encoders by training comparable models using the Ettin Family under matched experimental 
    
[^15]: 挖掘框架：将关联规则挖掘作为新闻框架检测的无监督框架

    Mining Frames: Association Rule Mining as an Unsupervised Framework for News Framing Detection

    [https://arxiv.org/abs/2610.31874](https://arxiv.org/abs/2610.31874)

    该论文提出将关联规则挖掘（ARM）作为无监督、无需标注的新闻框架检测新框架，通过对加沙战争首年25,512篇新闻文章的分析验证了其无需预定义分类体系即可归纳发现可解释框架模式的能力。

    

    计算框架分析目前主要由有监督分类器主导，这些分类器需要大规模标注语料库和预定义的框架分类体系，限制了其在非英语语言、新事件和低资源环境中的应用。本文提出将关联规则挖掘作为一种无监督、无需标注的新闻框架检测框架。每篇文档或句子被视为词汇项组成的事务，高提升度的共现规则被解读为候选框架信号（即Gamson和Modigliani所说的“浓缩符号”），并由记者-研究者在上下文中加以确认。主要验证案例为“加沙框架”研究：来自16家美国、英国、法国和德国媒体的25,512篇英文文章，涵盖加沙战争的第一年（2023年10月至2024年10月），采用Apriori算法（mlxtend）、平台加权支持度、词性过滤的n-grams以及基于提升度的评分进行分析。ARM能够在没有预定义分类体系的情况下归纳性地涌现出可解释的模式……

    arXiv:2610.31874v1 Announce Type: new  Abstract: Computational framing analysis is dominated by supervised classifiers that require large annotated corpora and predefined frame taxonomies, limiting their use for non-English languages, novel events, and low-resource settings. This paper presents Association Rule Mining (ARM) as an unsupervised, annotation-free framework for news framing detection. Each document or sentence is treated as a transaction of lexical items, and high-lift co-occurrence rules are read as candidate framing signals (Gamson and Modigliani's condensing symbols), confirmed in context by journalist-researchers. The primary validation is Framing Gaza: 25,512 English-language articles from 16 US, UK, French, and German outlets covering the first year of the Gaza war (October 2023 to October 2024), analyzed with Apriori (mlxtend), platform-weighted support, POS-filtered n-grams, and lift-based scoring. ARM inductively surfaces interpretable patterns without predefined c
    
[^16]: 基于ISO 11783网络的机器可读操作手册，用于离线多语言大语言模型辅助

    Machine-Readable Operator Manuals over the ISO 11783 Network for Offline, Multilingual LLM Assistance

    [https://arxiv.org/abs/2610.31863](https://arxiv.org/abs/2610.31863)

    本文提出在ISO 11783网络上对操作手册进行语义同步，使拖拉机侧的离线多语言LLM助手能够在31至89秒内获取完整的机具文档，传输时间与操作员首次连接机具时等待对象池上传的时间相当。

    

    如果农业机械要支持智能化的离线辅助，那么系统必须能够动态访问技术文档。本文分析并定义了基于ISO 11783的CAN总线上操作手册的语义同步，即将拖拉机侧助手的本地知识库与所连接机具的操作手册保持一致的过程。论文使用Markdown、JSON和XML对数据传输性能进行了基准测试，并评估了多种开源权重的大语言模型（LLM）在跨语言查询任务中的表现。答案的自动评估结果经过四位人工评审员的验证。根据格式和传输设置的不同，一份完整手册通过物理ISO 11783网络的传输时间为31至89秒，这与操作员在首次连接机具时已经习惯等待的对象池上传时间相当。Markdown生成的文件最小，而……（摘要在此处截断）

    arXiv:2610.31863v1 Announce Type: cross  Abstract: If agricultural machinery is to support intelligent, offline assistance, then the system must access technical documentation dynamically. In this paper, the Semantic Synchronization of operator manuals over ISO 11783-based CAN buses is analyzed, defined as the process that brings the local knowledge base of a tractor-side assistant into agreement with the operator manual of the connected implement. Data transfer performance was benchmarked using Markdown, JSON, and XML; various open-weight Large Language Models (LLMs) were also evaluated in cross-lingual query tasks. The automatic evaluation of the answers was validated against four human raters. A complete manual was transferred over a physical ISO 11783 network in 31 to 89 s, depending on format and transport settings, which is comparable to the object pool upload that operators already wait for at the first coupling of an implement. Markdown produced the smallest files, while the qu
    
[^17]: 查询所记住的：查询抽象程度决定古典阿拉伯语诗歌的词汇检索与稠密检索

    What the Query Remembers: Query Abstraction Governs Lexical and Dense Retrieval of Classical Arabic Poetry

    [https://arxiv.org/abs/2610.31821](https://arxiv.org/abs/2610.31821)

    该论文发现，在古典阿拉伯语诗歌检索中，查询保留诗歌原文措辞的程度（即查询的抽象程度）决定了检索器的优劣——记忆型查询下BM25领先所有稠密模型，而抽象查询下稠密编码器胜出，且BM25对措辞变化的敏感度远高于稠密模型。

    

    一个诗歌档案库能被搜索得多好，取决于用户还记得什么，而不仅仅是模型理解了什么。我们在古典阿拉伯语诗歌上验证了这一点，在两个语料库、三种索引模式和两种评估协议下，对九个稠密编码器和BM25进行了基准测试。哪个检索器胜出，取决于查询中仍然保留了多少诗歌原文的措辞。在“话到嘴边”（tip-of-the-tongue）型查询上，BM25领先所有稠密编码器，包括一个8B参数的最先进模型（Recall@5为0.746对0.717）；而在抽象查询上，BM25则跌至十个系统中的第八位（0.392对0.748）。对相同查询措辞的编辑实验表明，这种排名很大程度上反映了查询的构成：添加一句被记住的半行诗可使BM25提升38分；手工改写引用的词语后，BM25与8B编码器持平；而且BM25对被记住措辞的敏感度显著高于最强的稠密编码器。同样的维度也支配着解释增强方法。

    arXiv:2610.31821v1 Announce Type: cross  Abstract: How well a poetry archive can be searched depends on what the user still remembers, not only on what the model understands. We show this for Classical Arabic poetry, benchmarking nine dense encoders and BM25 over two corpora, three indexing modes and two evaluation protocols. Which retriever wins is governed by how much of a poem's wording the query still carries. On tip-of-the-tongue queries BM25 leads every dense encoder, including an 8B state-of-the-art model (0.746 vs. 0.717 Recall@5); on abstract queries it falls to eighth out of ten (0.392 vs. 0.748). Editing the wording of the same queries shows that this ranking largely reflects query composition: adding one remembered half-verse raises BM25 by 38 points, rewriting the quoted words by hand leaves BM25 level with the 8B encoder, and BM25's sensitivity to remembered wording is significantly steeper than that of the strongest dense encoder. The same axis controls explanation augme
    
[^18]: Lean Finder V2：为 Mathlib 提供可消除用户查询歧义的语义搜索

    Lean Finder V2: Semantic Search for Mathlib That Disambiguates User Queries

    [https://arxiv.org/abs/2610.30843](https://arxiv.org/abs/2610.30843)

    Lean Finder V2 通过互补的模型级对抗训练与输入级查询重写，实现了能够消除歧义的 Mathlib 语义搜索，有效解决了短而模糊查询的检索难题。

    

    我们提出了 Lean Finder V2，这是一个面向 Mathlib 的语义搜索框架，通过互补的模型级对抗训练和输入级查询重写来消除用户查询的歧义。人类数学家和编码智能体很少通过提供详细上下文来搜索 Mathlib。在实践中，他们查询的是简短的性质、记号、别名或假设，而将大部分预期的数学内容隐含在查询之外。相比之下，现有的 Mathlib 搜索引擎不切实际地假设查询带有丰富的上下文。由于缺乏与真实标准（ground-truth）Mathlib 陈述配对的真实世界短查询，这种歧义问题变得更加难以处理。我们认为这本质上是一个数据问题。通过对抗训练，我们扩大了带标注的短且含歧义查询的生成规模，并在模型级微调和输入级查询重写两个层面上学习消除用户查询的歧义。在人工策划的 MathlibQR 基准测试上，我们的 Lean Finder V2 创造了新的最佳纪录（原文在此处截断）。

    arXiv:2610.30843v1 Announce Type: new  Abstract: We introduce Lean Finder V2, a semantic search framework for Mathlib that disambiguates user queries through complementary model-level adversarial training and input-level query rewriting. Human mathematicians and coding agents rarely search Mathlib by providing detailed context. In practice, they query short properties, notation, aliases, or hypotheses, leaving much of the intended mathematics implicit. In contrast, existing Mathlib search engines unrealistically assume context-rich queries. The scarcity of real-world short queries paired with ground-truth Mathlib statements makes this ambiguity even more intractable. We argue that this is essentially a data problem. Through adversarial training, we scale up the generation of labeled short and ambiguous queries, and learn to disambiguate user queries at both model-level fine-tuning and input-level query rewriting. On the human-curated MathlibQR benchmark, our Lean Finder V2 sets a new s
    
[^19]: 基于结构化定位的驾驶视频复杂查询检索

    Driving Video Retrieval for Complex Queries with Structured Grounding

    [https://arxiv.org/abs/2606.09109](https://arxiv.org/abs/2606.09109)

    提出STRIVE-D数据校准检索框架，通过弱标注视频评估并自适应调整查询规则，将校准后的规则分数与视觉-语言及关键词检索信号融合，从而在驾驶视频中可靠检索加塞、急刹等复杂动态事件。

    

    大规模视频检索是自动驾驶数据整理与安全验证的核心任务，用户不仅希望检索到场景，还希望找到诸如加塞、急刹车等动态事件。现有的视觉-语言检索与基于关键词的检索方法常常会遗漏这些事件，原因在于相关运动可能未在文本中被明确描述，或无法通过词汇重叠被捕获。基于规则的检索能够更直接地编码这类事件，但它较为脆弱：当生成或手写的规则与真实驾驶数据的假设不匹配时往往会失效。我们提出了STRIVE-D，一个面向驾驶视频的数据校准检索框架。该框架利用弱标注的域内视频来估计查询规则何时可靠，对与观测数据不匹配的规则进行自适应调整，并将校准后的规则分数与视觉-语言检索及关键词检索信号相融合。在三个驾驶基准上的实验（包括新发布的Dr...（摘要原文在此处被截断）

    arXiv:2606.09109v3 Announce Type: replace-cross  Abstract: Video retrieval at scale is central to data curation and safety validation in autonomous driving, where users want to find not only scenes but also dynamic events such as cut-ins and hard braking. Existing vision-language and keyword-based retrieval methods often miss these events because the relevant motion may not be explicitly described in text or captured by lexical overlap. Rule-based retrieval can encode such events more directly, but it is brittle: generated or hand-written rules often fail when their assumptions do not match real driving data. We propose STRIVE-D, a data-calibrated retrieval framework for driving videos. It uses weakly labeled in-domain videos to estimate when a query rule is reliable, adapt rules that mismatch observed data, and fuse calibrated rule scores with vision-language and keyword-based retrieval signals. Across three driving benchmarks, including newly released human-annotated event data on Dr
    
[^20]: 深度学习通过历史地图对齐实现城市变化特征分析

    Deep learning enables urban change profiling through alignment of historical maps

    [https://arxiv.org/abs/2602.02154](https://arxiv.org/abs/2602.02154)

    提出了一种基于深度学习的模块化框架，通过空间地图对齐、多时相目标提取和变化估计，实现了从大规模历史地图集合中进行细粒度、定量的城市变化分析。

    

    在现代对地观测技术出现之前，历史地图为长期城市变迁提供了独特的记录，也是观察城市不断演变的特征的一个视角。然而，由于空间错位、制图差异以及文档质量退化等问题，从历史地图系列中提取一致且细粒度的变化信息仍然具有挑战性，这使得大多数分析只能局限于小规模或定性的方法。我们提出了一种基于深度学习的框架，用于从大规模历史地图集合中进行细粒度的城市变化分析，该框架采用模块化设计，集成了空间地图对齐、多时相目标提取和变化估计。该框架将历史地图分析从临时的视觉比较转向系统的、定量的城市变化特征描述。实验证明了所提出的对齐和目标提取方法的鲁棒性能。

    arXiv:2602.02154v2 Announce Type: replace-cross  Abstract: Prior to modern Earth observation technologies, historical maps provide a unique record of long-term urban transformation and offer a lens on the evolving identity of cities. However, extracting consistent and fine-grained change information from historical map series remains challenging due to spatial misalignment, cartographic variation, and degrading document quality, limiting most analyses to small-scale or qualitative approaches. We propose a deep-learning-based framework for fine-grained urban change analysis from large collections of historical maps, built on a modular design that integrates spatial map alignment, multi-temporal object extraction, and change estimation. This framework shifts the analysis of historical maps from ad hoc visual comparison toward systematic, quantitative characterization of urban change. Experiments demonstrate the robust performance of the proposed alignment and object extraction methods. A
    

