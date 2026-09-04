# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CORE: Improving Compositional Reasoning in MLLM Embedding via Reranker Distillation](https://arxiv.org/abs/2609.04083) | CORE通过将交叉注意力重排序器的细粒度组合判断以列表式Rank-KL目标蒸馏到嵌入模型中，显著提升了MLLM嵌入模型的组合推理能力，其效果优于对比学习和CoSENT。 |
| [^2] | [The Dice Roll Method: A Standardized Protocol for Repeated-Query Auditing of Large Language Model Brand Recommendations](https://arxiv.org/abs/2609.04047) | 本文提出并形式化了“骰子法”——一个基于温度缩放核采样生成模型的可复用标准化协议，通过将响应方差分解为多个成分并提供完整的统计分析技术栈，为大语言模型品牌推荐的重复查询审计建立了系统方法。 |
| [^3] | [The WebKurator.de Platform: Combined Regional and Topical Web Curation](https://arxiv.org/abs/2609.03971) | WebKurator.de 平台提出了一种将主题分类与地理标注明确分离的二维策展模型，结合基于大语言模型的主题分类和基于印记信息的地理编码技术，实现了区域与主题相结合的网络策展，并基于 554 万个德国网站数据集构建。 |
| [^4] | [RuleMem: Active Rule Memory for Long-Term Conversational Agents](https://arxiv.org/abs/2609.03915) | RuleMem提出了一种基于规则的主动记忆框架，通过从历史对话中归纳并验证自然语言霍恩子句来主动指导证据检索与推理，显著提升了长期对话问答代理的可靠性。 |
| [^5] | [Comparing Retrieval Methods for Academic Advisor Discovery: A Six-Method Study of 768 CS Faculty Profiles Across 9 US Universities](https://arxiv.org/abs/2609.03901) | 该研究构建了一个包含美国9所大学768位计算机科学教师档案的新数据集，系统比较了六种信息检索方法在学术导师发现任务上的表现，发现重排序方法效果最佳（平均NDCG@10达0.477），且语义检索整体优于传统词汇匹配方法。 |
| [^6] | [GRASP: Graph-Retrieval Automated Scoring Pipeline for Label-Free Multi-Topic Essay Grading](https://arxiv.org/abs/2609.03857) | 提出了GRASP评分流水线，通过Sentence-BERT与FAISS向量索引、参考答案语义相似度图以及大语言模型消歧，首次实现对答案混合无标签的多主题简答题考试的自动评分。 |
| [^7] | [Unified Pitch Graphs for Diagnosing Pitching Strategy](https://arxiv.org/abs/2609.03810) | 本文提出统一投球图（UPG）这一分层图表示方法，通过保留每个投球的精确三维轨迹与上下文并构建有向序列边，对棒球投球序列进行回顾性分析，揭示了名义相同的投球序列背后不同的物理执行及长上下文中日益显著的有序策略结构。 |
| [^8] | [LLM4AIGQ: LLM-based AI Guidance Query Generation Framework for Multi Interest Mining](https://arxiv.org/abs/2609.03674) | 提出LLM4AIGQ框架，利用大语言模型挖掘用户多兴趣来生成AI引导查询，解决了传统共现检索范式中语义漂移和购买意图不匹配的问题。 |
| [^9] | [Enhancing Financial Question Answering: A Novel Benchmark Dataset of Banks' financial statements](https://arxiv.org/abs/2609.03654) | 该论文提出了首个针对跨机构银行财务报表检索的金融问答基准数据集 FinRAG-QA，包含999个从业者整理的问题和24家欧美大型银行的209份超长报告，并系统评估了多阶段 RAG 流水线中各组件的贡献。 |
| [^10] | [EPIC: Explicit Posterior Item Conditioning for Semantic ID Diffusion Recommendation](https://arxiv.org/abs/2609.03522) | 提出EPIC方法，将显式的物品级后验竞争引入语义ID扩散去噪过程，通过个性化候选物品后验分布来指导未确定位置的词元预测，且无需修改冻结的预训练骨干网络。 |
| [^11] | [From Topical Relevance to Answerability: Entailment Distillation for Conversational Retrieval](https://arxiv.org/abs/2609.03482) | CLEAR框架通过蕴含蒸馏将答案-段落蕴含监督迁移到重排序器中，使对话式检索从主题相关性转向真正的可回答性，并借助溯因召回模块引入低相似度但可回答的段落以提升检索效果。 |
| [^12] | [ExplainRoute: A Pre-Deployment Audit Framework for Non-Answer-Giving Programming Tutors](https://arxiv.org/abs/2609.03470) | 本文提出ExplainRoute，一个面向不直接给出答案的编程导师的部署前审计框架，通过机器可检验的契约审计信息边界、响应极性、失败闭环和学习者解释可见性等维度。 |
| [^13] | [When Retrieval Helps: Selective Retrieval for Single-Turn Mental-Health QA](https://arxiv.org/abs/2609.03454) | 该研究针对单轮心理健康问答提出一种轻量级选择性检索策略，通过心理教育需求、应对需求、回答具体性三个效用维度及安全触发机制判断检索的必要性，从而在发挥检索增强优势的同时避免其负面影响。 |
| [^14] | [Plan Pointers and Record-Directive Form in Budgeted Verification of Inherited Agent Memory](https://arxiv.org/abs/2609.03450) | 该论文通过十二项注册研究发现，写入智能体记忆库的指令形式（准则、裸ID或指针）会以高度模型依赖的方式显著影响预算受限下的记录选择，长度匹配准则可带来35分的提升，但附加ID可能完全抵消准则的效果。 |
| [^15] | [Spruce: Scalable Private Outsourced Retrieval Using Compact Embeddings](https://arxiv.org/abs/2609.03376) | Spruce通过将紧凑二进制嵌入表示与密码学协议协同设计，用汉明距离计算取代全语料库嵌入评分，将百万文档规模的隐私外包检索性能显著提升。 |
| [^16] | [HypRQ-VAE: Hyperbolic Item Indexing for Long-Tail-Aware Generative Recommender Systems](https://arxiv.org/abs/2609.03369) | 提出首个在双曲空间中学习物品索引的框架HypRQ-VAE，利用双曲残差量化变分自编码器解决生成式推荐中欧几里得空间难以建模物品长尾分布的问题。 |
| [^17] | [SciLENS: RL-Driven Autonomous Agents for Scientific Localized Evidence Navigation and Synthesis](https://arxiv.org/abs/2609.03338) | SciLENS提出了一个完全本地化运行的强化学习驱动自主智能体框架，开创性地将结构化可视化集成到推理循环中以缓解上下文耗尽，并通过自动化的多跳子图合成流水线与跨模型共识验证实现无需人工标注的智能体训练，从而在约1200万条学术记录上实现科学证据的本地化导航与综合。 |
| [^18] | [SelfDR: Self-Distillation from Reasoning for LLM-Based Recommendation](https://arxiv.org/abs/2609.03313) | SelfDR提出了一种推理自蒸馏框架，将大语言模型自身经推理增强的预测能力蒸馏为直接推荐，在无需外部模型的情况下同时提升推荐效果与推理效率。 |
| [^19] | [DoPR: Reusable Compressed Document Prefixes for Efficient LLM Reranking](https://arxiv.org/abs/2609.03311) | DoPR提出一种压缩文档前缀框架，将文档处理与查询处理解耦，通过离线预计算并跨查询重用压缩的文档前缀状态，使LLM在线重排序只需处理查询部分，从而实现高达8倍的在线计算成本降低。 |
| [^20] | [UniCon: A Unified Context-Centric Modeling Paradigm for CTR Prediction](https://arxiv.org/abs/2609.03290) | 本文提出UniCon，一种统一的以上下文为中心的CTR预测建模范式，将用户历史行为与当前请求作为同质的上下文单元进行统一建模，克服传统异构信号划分的局限，提升扩展效率与预测质量。 |
| [^21] | [SHELF: A Synthetic Harness for Multi-Task Bibliographic Benchmarking](https://arxiv.org/abs/2609.03047) | SHELF是一个基于美国国会图书馆词表生成6万余篇合成文档的Python系统，为图书馆和档案馆的书目工作提供了涵盖分类、聚类、检索等多任务的系统性基准测试框架。 |
| [^22] | [Reflect-SQL: A Self-Reflection Based Framework for Text-to-SQL](https://arxiv.org/abs/2609.02944) | Reflect-SQL是一个基于多阶段自我反思的Text-to-SQL新框架，通过知识库理解晦涩的数据库模式，并利用LLM-as-a-judge驱动的评分机制在反馈循环中迭代优化每个阶段的SQL生成结果。 |
| [^23] | [CHSR-RRF: A curriculum-gated hybrid retrieval framework with reciprocal rank fusion and leakage-aware benchmarking for educational RAG](https://arxiv.org/abs/2609.02913) | 提出课程门控混合检索框架CHSR-RRF，通过在检索前施加课程元数据约束并结合倒数排序融合，将课程泄漏减少4.6倍且不损失召回率，同时发布了包含126个案例的泄漏感知教育检索基准CERB。 |
| [^24] | [R$^{2}$Adapter: A Routing and Rewriting Adapter for Efficient Hybrid RAG](https://arxiv.org/abs/2609.02894) | 提出了轻量级即插即用的路由与重写适配器R²Adapter，可动态将查询分配给原生RAG或基于图的RAG，仅对真正受益于图推理的查询进行图检索，从而降低不必要的开销并减少对底层大模型的依赖。 |
| [^25] | [ViSAR: Training-Free Adaptive-$k$ Retrieval for Visual Document Question Answering](https://arxiv.org/abs/2609.02486) | 提出了一种无需训练的自适应k值检索方法ViSAR，通过在嵌入空间中构建查询条件的页面级相似度矩阵来动态确定检索页面数量，在保持或提升答案准确性的同时将RAG延迟降低高达58.7%。 |
| [^26] | [Route Based Map Matching via a Structured Codebook and Token Sequence Decoding](https://arxiv.org/abs/2607.22543) | 该论文提出一种基于路径的轻量级地图匹配方法，将候选路径编码为线路与交叉口词元构成的码本，并借助DAFSA×Levenshtein自动机将GPS轨迹匹配转化为模糊词元序列对齐解码，使单次查询成本比暴力扫描低多个数量级。 |
| [^27] | [PACMS: Submodular Context Selection as a Pluggable Engine for LLM Agents](https://arxiv.org/abs/2606.20047) | 该论文提出 PACMS，一种基于次模优化的上下文选择方法，作为可插拔引擎嵌入 LLM 智能体框架，用以替代对主题不敏感的按最近性截断机制，在上下文超出 token 预算时智能地保留与当前查询相关的信息。 |
| [^28] | [Is Position Bias in Dense Retrievers Built In-or Learned from Data?](https://arxiv.org/abs/2605.26578) | 研究发现稠密检索器的位置偏差主要源于训练数据中证据位置的分布，而非模型架构本身，通过位置均衡的训练数据可将位置敏感性降低57%至87%。 |
| [^29] | [Cost and Accuracy of Long-Term Memory in Distributed Multi-Agent Systems Based on Large Language Models](https://arxiv.org/abs/2601.07978) | 本文构建了一个独立可复现的云边环境测试平台，首次从系统级成本（CPU时间、内存、磁盘I/O、网络）角度对mem0、Graphiti、cognee三种长期记忆框架与RAG及全上下文基线在分布式多智能体系统中的精度和资源开销进行了系统对比评估。 |
| [^30] | [Causal-Counterfactual RAG: The Integration of Causal-Counterfactual Reasoning into RAG](https://arxiv.org/abs/2509.14435) | 该论文提出因果-反事实RAG框架，通过将显式因果图与反事实推理融入检索增强生成过程，解决了传统RAG系统因文本分块破坏上下文完整性和过度依赖语义相似性检索而导致的回答浅显不准确的问题。 |
| [^31] | [TC-RAG:Turing-Complete RAG's Case study on Medical LLM Systems](https://arxiv.org/abs/2408.09199) | 提出TC-RAG框架，通过引入图灵完备系统管理状态变量，并利用具备自适应检索、推理和规划能力的记忆栈系统，实现检索过程的受控终止并减少错误知识累积，从而显著提升医疗大语言模型系统知识检索的效率与准确性。 |
| [^32] | [Multi-Task Deep Recommender Systems: A Survey](https://arxiv.org/abs/2302.03525) | 本综述填补了推荐领域多任务学习系统性综述的空白，全面回顾了多任务深度推荐系统（MTDRS），并从任务关系（并行、级联、辅助与主任务）和方法论（参数共享、优化、训练机制）两个角度提出了系统的分类体系。 |

# 详细

[^1]: CORE：通过重排序器蒸馏改进MLLM嵌入中的组合推理

    CORE: Improving Compositional Reasoning in MLLM Embedding via Reranker Distillation

    [https://arxiv.org/abs/2609.04083](https://arxiv.org/abs/2609.04083)

    CORE通过将交叉注意力重排序器的细粒度组合判断以列表式Rank-KL目标蒸馏到嵌入模型中，显著提升了MLLM嵌入模型的组合推理能力，其效果优于对比学习和CoSENT。

    

    基于MLLM的嵌入模型在组合检索方面仍然存在局限，常常无法区分包含相同概念但属性-对象绑定不同的场景。然而，当同一骨干网络被用作交叉注意力重排序器时，却能够解决这类区分问题，这促使我们将其组合判断蒸馏到嵌入模型中。我们提出了CORE，该方法合成了跨越五个组合匹配级别的候选列表，并引入Rank-KL目标函数，训练嵌入模型复现重排序器的细粒度排序。我们进一步提出了一种分级评估协议，并在相同的数据和调参预算下比较了对比学习、成对式CoSENT和列表式Rank-KL。比较结果表明，CoSENT和Rank-KL都比对比学习更有效地利用了多级别监督，其中Rank-KL取得了最强的整体性能。在三个组合推理基准上……（摘要截断）

    arXiv:2609.04083v1 Announce Type: cross  Abstract: MLLM-based embedding models remain limited in compositional retrieval, often failing to distinguish scenes containing the same concepts but different attribute-object bindings. Yet the same backbone can resolve such distinctions when used as a cross-attentive reranker, motivating us to distill its compositional judgments into the embedding model. We propose CORE, which synthesizes candidate lists spanning five compositional matching levels and introduces a Rank-KL objective that trains the embedding model to reproduce the reranker's fine-grained ranking. We further introduce a graded evaluation protocol and compare contrastive learning, pairwise CoSENT, and listwise Rank-KL under the same data and tuning budget. Our comparison shows that both CoSENT and Rank-KL use the multi-level supervision more effectively than contrastive learning, with Rank-KL achieving the strongest overall performance. Across three compositional reasoning benchm
    
[^2]: 骰子法：大语言模型品牌推荐重复查询审计的标准化协议

    The Dice Roll Method: A Standardized Protocol for Repeated-Query Auditing of Large Language Model Brand Recommendations

    [https://arxiv.org/abs/2609.04047](https://arxiv.org/abs/2609.04047)

    本文提出并形式化了“骰子法”——一个基于温度缩放核采样生成模型的可复用标准化协议，通过将响应方差分解为多个成分并提供完整的统计分析技术栈，为大语言模型品牌推荐的重复查询审计建立了系统方法。

    

    背景：研究人员越来越多地使用重复相同的提示词来审计大语言模型（LLM）品牌推荐中的随机变异，然而目前尚无用于设定迭代次数、选择稳定性指标或建立可靠性阈值的标准化协议。目标：我们将骰子法形式化为一个可复用的协议，用于LLM品牌推荐的重复查询审计，该协议建立在温度缩放核采样的生成模型基础之上。方法：将总响应方差分解为采样、提示措辞、运行间和模型版本等成分。该技术栈包括：以迭代作为重复测量的负二项混合模型；作为无分布效应量的Cliff's delta；保留依赖结构的自助法；基于模拟的统计功效分析；概化理论分解；以及对固定快照的漂移诊断。我们重新分析了五项品牌推荐审计研究：约190（摘要在此处截断）。

    arXiv:2609.04047v1 Announce Type: cross  Abstract: Background: Researchers increasingly use repeated identical prompts to audit stochastic variation in large language model (LLM) brand recommendations, yet no standardized protocol exists for setting iteration counts, selecting stability metrics, or establishing reliability thresholds. Objective: We formalize the Dice Roll Method as a reusable protocol for repeated-query auditing of LLM brand recommendations, grounded in a generative model of temperature-scaled nucleus sampling. Methods: Total response variance is decomposed into sampling, prompt-phrasing, run-to-run, and model-version components. The stack: a negative-binomial mixed model with iterations as repeated measures; Cliff's delta as the distribution-free effect size; dependence-preserving bootstrap; simulation-based power; a generalizability-theory decomposition; drift diagnostics on pinned snapshots. We reanalyse five brand-recommendation auditing studies: approximately 190,
    
[^3]: WebKurator.de 平台：区域与主题相结合的网络策展

    The WebKurator.de Platform: Combined Regional and Topical Web Curation

    [https://arxiv.org/abs/2609.03971](https://arxiv.org/abs/2609.03971)

    WebKurator.de 平台提出了一种将主题分类与地理标注明确分离的二维策展模型，结合基于大语言模型的主题分类和基于印记信息的地理编码技术，实现了区域与主题相结合的网络策展，并基于 554 万个德国网站数据集构建。

    

    网络的系统性策展仍然是旨在保存文化和区域相关内容的国家级图书馆和记忆机构所面临的核心挑战。现有的基于目录的方法（如 Curlie）主要采用以主题为中心的一维层次结构，其中地理信息与主题和语言类别交织在一起。为了解决这一局限性，我们提出了 WebKurator.de，一个用于区域与主题相结合的网络策展的协作平台，初期专注于德国网络。WebKurator 引入了一个二维策展模型，明确地将主题分类和地理标注分离开来。该系统集成了基于大语言模型（LLM）的主题分类、基于印记信息的地址提取与地理编码技术，并支持用户建议与人工审核相结合的机制。该平台基于德国印记数据集（German Imprints Dataset）构建，这是一个包含 554 万个网站的大规模数据集。

    arXiv:2609.03971v1 Announce Type: new  Abstract: The systematic curation of the Web remains a central challenge for national libraries and memory institutions that aim to preserve culturally and regionally relevant content. Existing directory-based approaches such as Curlie implement a predominantly topic-centric, one-dimensional hierarchy, where geographic aspects are intertwined with topical and linguistic categories. To address this limitation, we present WebKurator.de, a collaborative platform for combined regional and topical web curation, initially focused on the German web. WebKurator introduces a two-dimensional curation model that explicitly separates topical categorization and geographic annotation. The system integrates LLM-based topic classification and imprint-based address extraction with geocoding, and supports user suggestions together with moderated review. The platform is bootstrapped from the German Imprints Dataset, a large-scale collection of 5.54 million websites.
    
[^4]: RuleMem：面向长期对话代理的主动规则记忆

    RuleMem: Active Rule Memory for Long-Term Conversational Agents

    [https://arxiv.org/abs/2609.03915](https://arxiv.org/abs/2609.03915)

    RuleMem提出了一种基于规则的主动记忆框架，通过从历史对话中归纳并验证自然语言霍恩子句来主动指导证据检索与推理，显著提升了长期对话问答代理的可靠性。

    

    长期对话中的问答代理必须对海量且在时间上分散的对话历史进行推理。然而，现有的记忆机制主要将过去的信息视为被动存储的事实，导致语义鸿沟和不可靠的推理。为了解决这一局限性，我们提出了RuleMem，这是一个基于规则的记忆框架，它从历史交互中归纳出可重用的逻辑规则，以主动地指导证据检索和推理。具体而言，RuleMem从对话中构建自然语言的霍恩子句，并通过规则困惑度一致性机制对其进行验证。这些归纳出的规则能够检索语义上相距较远的证据，同时为答案生成提供显式的逻辑结构。我们在两个长期对话基准LoCoMo和LongMemEval_s*上对RuleMem进行了全面评估，并在与14个基线的严格比较中验证了其有效性。

    arXiv:2609.03915v1 Announce Type: new  Abstract: Question answering agents in long-term conversations must reason over massive, temporally dispersed dialogue histories. However, existing memory mechanisms primarily treat past information as \textit{passively} stored facts, leading to semantic gaps and unreliable reasoning. To address this limitation, we propose RuleMem, a rule-based memory framework that induces reusable logical rules from historical interactions to \textit{actively} guide both evidence retrieval and reasoning. Specifically, RuleMem constructs natural-language Horn clauses from conversations and validates them via a Rule Perplexity Consistency (RPC) mechanism. These induced rules enable the retrieval of semantically distant evidence while providing an explicit logical structure for answer generation. We conducted a comprehensive evaluation of RuleMem on two long-term conversational benchmarks, LoCoMo and LongMemEval_s*. In a rigorous comparison against 14 baselines on 
    
[^5]: 比较学术导师发现中的检索方法：针对美国9所大学768位计算机科学教师档案的六种方法研究

    Comparing Retrieval Methods for Academic Advisor Discovery: A Six-Method Study of 768 CS Faculty Profiles Across 9 US Universities

    [https://arxiv.org/abs/2609.03901](https://arxiv.org/abs/2609.03901)

    该研究构建了一个包含美国9所大学768位计算机科学教师档案的新数据集，系统比较了六种信息检索方法在学术导师发现任务上的表现，发现重排序方法效果最佳（平均NDCG@10达0.477），且语义检索整体优于传统词汇匹配方法。

    

    我们针对学术导师发现任务——即根据研究生申请人的研究兴趣陈述对计算机科学教师进行相关性排序——对六种信息检索方法进行了比较评估。这些方法涵盖稀疏词汇匹配（Jaccard重叠度、TF-IDF、BM25）、稠密语义检索（all-MiniLM-L6-v2句子嵌入）、混合分数融合以及学习排序。评估采用一个新的特定领域数据集：从美国9个计算机科学系抓取的768份教师档案，并针对代表不同研究生研究画像的5个查询进行了162个分级相关性判定（等级0/1/2）。在全部五个查询中，Reranked（重排序）方法取得了最高的平均NDCG@10（0.477，标准差0.138），其后依次为Semantic（语义检索，0.450）、Hybrid（混合方法，0.421）、BM25（0.406）、Jaccard（0.303）和TF-IDF（0.246）。经过对全部15组两两比较进行Bonferroni校正后，TF-IDF显著差于BM25、Semantic、Hybrid和Reranked，而……

    arXiv:2609.03901v1 Announce Type: cross  Abstract: We present a comparative evaluation of six information retrieval methods for the task of academic advisor discovery: ranking CS faculty members by relevance to a graduate applicant's research interest statement. The methods span sparse lexical matching (Jaccard overlap, TF-IDF, BM25), dense semantic retrieval (all-MiniLM-L6-v2 sentence embeddings), hybrid score fusion, and learning-to-rank. Evaluation uses a new domain-specific collection: 768 faculty profiles scraped from 9 US CS departments, with 162 graded relevance judgments (grade 0/1/2) across 5 queries representing distinct graduate student research profiles. Across all five queries, Reranked achieves the highest mean NDCG@10 (0.477, std 0.138), followed by Semantic (0.450), Hybrid (0.421), BM25 (0.406), Jaccard (0.303), and TF-IDF (0.246). After Bonferroni correction across all 15 pairwise comparisons, TF-IDF is significantly worse than BM25, Semantic, Hybrid, and Reranked; no 
    
[^6]: GRASP：面向无标签多主题作文评分的图检索自动评分流水线

    GRASP: Graph-Retrieval Automated Scoring Pipeline for Label-Free Multi-Topic Essay Grading

    [https://arxiv.org/abs/2609.03857](https://arxiv.org/abs/2609.03857)

    提出了GRASP评分流水线，通过Sentence-BERT与FAISS向量索引、参考答案语义相似度图以及大语言模型消歧，首次实现对答案混合无标签的多主题简答题考试的自动评分。

    

    自动简答题评分研究历来只关注仅包含单一主题问题的考试，而对包含多个主题问题的考试进行自动评分的研究仍然较少。在这项工作中，我们提出了一种图检索自动评分流水线（GRASP），用于对无标签的多主题科学考试进行评分。无标签考试是指学生的多个不同主题的回答被合并为单一段落的简答题考试，其中没有任何标记标签或分段来指示哪一部分内容回答了哪个问题。每个问题的参考答案通过 Sentence-BERT 编码并存入 FAISS 向量索引，并在这些参考答案集合上构建语义相似度图。在评分时，首先应用句子数量启发式方法，并借助大语言模型解决歧义情况，以预测学生作文中回答了多少个不同的主题。

    arXiv:2609.03857v1 Announce Type: new  Abstract: Automated short-answer grading research has historically focused on exams consisting solely of questions pertaining to a single topic. Automatic grading of exams containing questions about more than one topic remains less explored. In this work, a Graph-Retrieval Automated Scoring Pipeline (GRASP) is introduced for grading label-free multi-topic science exams. Label-free exams are short-answer exams in which a student's responses to several distinct topics are merged into a single paragraph, with no markup labels or segmentation indicating which span answers which question. Reference answers for each question are encoded into a FAISS vector index via Sentence-BERT, and a semantic similarity graph is constructed over this set of reference answers. At grading time, sentence count heuristics, with a large language model used to resolve ambiguous cases, are first applied to predict how many distinct topics were answered in the student essay.
    
[^7]: 统一投球图：用于诊断投球策略

    Unified Pitch Graphs for Diagnosing Pitching Strategy

    [https://arxiv.org/abs/2609.03810](https://arxiv.org/abs/2609.03810)

    本文提出统一投球图（UPG）这一分层图表示方法，通过保留每个投球的精确三维轨迹与上下文并构建有向序列边，对棒球投球序列进行回顾性分析，揭示了名义相同的投球序列背后不同的物理执行及长上下文中日益显著的有序策略结构。

    

    棒球中的投球策略既体现在投球的物理执行上，也体现在投球使用的有序上下文中，然而常见的表示方法往往将投球压缩为离散的类型或聚合的统计数据。我们提出了统一投球图（UPG），这是一种用于对序列化时空事件进行回顾性分析的分层图表示。UPG 将每个投球保存为携带重建的三维轨迹和上下文的精确事件，通过有向序列边连接相邻的投球，并在语义和时间分辨率上组织相同的事件。当重复出现的证据不足时，一种支持自适应机制会从细粒度的长序列向后回退，同时保留精确的事件谱系。我们在2021年至2026年间394万条 MLB Statcast 投球数据上对 UPG 进行了评估。结果显示，名义上相同的投球序列实际上呈现出不同的物理执行，且有序结构在更长的上下文中表现得愈发明显……

    arXiv:2609.03810v1 Announce Type: new  Abstract: Pitching strategy in baseball is expressed through both physical execution and the ordered context in which pitches are used, yet common representations collapse pitches into discrete types or aggregate statistics. We present Unified Pitch Graphs (UPG), a hierarchical graph representation for retrospective analysis of sequential spatiotemporal events. UPG preserves each pitch as an exact event with reconstructed three-dimensional trajectory and context, connects consecutive pitches through directed sequence edges, and organizes the same events across semantic and temporal resolutions. A support-adaptive mechanism backs off from fine, long sequences when repeated evidence is insufficient, while retaining exact event lineage. We evaluate UPG on 3.94 million MLB Statcast pitches from 2021 to 2026. Nominally identical pitch sequences exhibit distinct physical executions, and ordered structure becomes increasingly evident in longer context-co
    
[^8]: LLM4AIGQ：基于大语言模型的面向多兴趣挖掘的AI引导查询生成框架

    LLM4AIGQ: LLM-based AI Guidance Query Generation Framework for Multi Interest Mining

    [https://arxiv.org/abs/2609.03674](https://arxiv.org/abs/2609.03674)

    提出LLM4AIGQ框架，利用大语言模型挖掘用户多兴趣来生成AI引导查询，解决了传统共现检索范式中语义漂移和购买意图不匹配的问题。

    

    引导查询通过提取用户偏好为搜索查询提供引导价值，从而刺激用户消费，在电子商务领域发挥着至关重要的作用。传统的AI生成查询（AIGQ）生成主要依赖于两阶段的“查询到AI生成查询”（Q2AIGQ）关联范式，首先通过多路径检索从用户画像、历史行为序列、商品侧信息和当前查询中召回用户主要搜索查询，然后通过基于规则的方法泛化生成AIGQ。这种方法由于信息级联损失而存在语义漂移问题；此外，主要搜索查询的推导严重依赖于“用户-商品”共现关系，缺乏对用户多兴趣的探索，导致引导查询价值低且与购买意图不匹配。为解决传统基于共现检索的表达局限性，我们提出了LLM4AIGQ，一种基于大语言模型的解决方案。

    arXiv:2609.03674v1 Announce Type: new  Abstract: Guidance queries stimulate user consumption by extracting preferences to provide search queries with guidance value, playing a crucial role in the e-commerce field. Traditional AI-generated queries (AIGQ) generation primarily relies on a two-stage "Query-to-AI-Generated-Query" (Q2AIGQ) association paradigm, first recalling user primary search queries from user profiles, historical behavior sequences, item-side information, and the current query through multi-path retrieval, then generalizing AIGQ via rule-based methods. This approach suffers from semantic drift due to information cascade loss; additionally, primary search query derivation heavily depends on "user-item" co-occurrence relationships, lacking exploration of user multi-interests, resulting in guidance queries with low value and mismatched purchase intent. To address the expressive limitations of traditional co-occurrence-based retrieval, we propose LLM4AIGQ, an LLM-based solu
    
[^9]: 增强金融问答：一个基于银行财务报表的新型基准数据集

    Enhancing Financial Question Answering: A Novel Benchmark Dataset of Banks' financial statements

    [https://arxiv.org/abs/2609.03654](https://arxiv.org/abs/2609.03654)

    该论文提出了首个针对跨机构银行财务报表检索的金融问答基准数据集 FinRAG-QA，包含999个从业者整理的问题和24家欧美大型银行的209份超长报告，并系统评估了多阶段 RAG 流水线中各组件的贡献。

    

    由于银行财务报表的复杂性、篇幅冗长、专业术语的使用，以及不同司法管辖区和机构之间文本与数值内容的异质性，对其进行比较分析对自动问答系统构成了重大挑战。我们提出了 FinRAG-QA，一个新颖的金融问答基准数据集，包含999个由从业者精心整理的问题，涵盖10个标准化指标，基于24家欧美主要银行2019年至2023年的209份年度报告和第三支柱（Pillar 3）报告。与以往主要聚焦于美国申报文件和单一机构分析的金融问答基准不同，FinRAG-QA 针对的是跨机构检索场景，文档平均长度达19.8万字，超过了任何现有的金融问答资源。在该基准上，我们评估了一个多阶段 RAG 流水线，并分离量化了每个组件的贡献。上下文分块增强结合检索器……

    arXiv:2609.03654v1 Announce Type: cross  Abstract: The comparative analysis of banks' financial statements poses significant challenges for automated question answering systems due to their complexity, substantial length, technical language, and inhomogeneity of both textual and numerical content across different jurisdictions and institutions. We introduce FinRAG-QA, a novel benchmark dataset for financial question answering, which comprises 999 practitioner-curated questions on 10 standardised indicators, grounded in 209 annual and Pillar 3 reports from 24 major European and U.S. banks spanning 2019-2023. Unlike prior financial QA benchmarks, which centre on U.S. filings and single-institution analysis, FinRAG-QA targets cross-institutional retrieval over documents averaging 198k words, longer than any existing financial QA resource. On this benchmark we evaluate a multi-stage RAG pipeline and isolate the contribution of each component. Contextual chunk enrichment combined with a ret
    
[^10]: EPIC：面向语义ID扩散推荐的显式后验物品条件化

    EPIC: Explicit Posterior Item Conditioning for Semantic ID Diffusion Recommendation

    [https://arxiv.org/abs/2609.03522](https://arxiv.org/abs/2609.03522)

    提出EPIC方法，将显式的物品级后验竞争引入语义ID扩散去噪过程，通过个性化候选物品后验分布来指导未确定位置的词元预测，且无需修改冻结的预训练骨干网络。

    

    语义ID（SID）生成式推荐通过生成一串简短的离散词元来预测下一个物品。近期的掩码扩散方法通过双向上下文和灵活解码改进了这一过程，但推荐最终需要在完整的物品目录中进行选择。在每个去噪步骤中，一个不完整的SID可能对应多个可行的物品，而现有方法主要通过逐位置的词元预测来进行推理。我们提出了显式后验物品条件化（EPIC），将显式的物品级竞争引入SID去噪过程。EPIC利用当前生成上下文和用户的近期交互，在可行的候选物品上构建个性化的后验分布，然后将该分布投影回尚未确定的SID位置，以指导后续的词元决策。预训练的骨干网络保持冻结，且不需要额外的解码器前向传播。在四个Amazon数据集上的实验……

    arXiv:2609.03522v1 Announce Type: cross  Abstract: Semantic ID (SID) generative recommendation predicts the next item by generating a short tuple of discrete tokens. Recent masked-diffusion methods improve this process through bidirectional context and flexible decoding, yet recommendation ultimately requires selecting among complete catalog items. At each denoising step, a partial SID can correspond to multiple feasible items, while existing methods primarily reason through position-wise token predictions. We propose Explicit Posterior Item Conditioning (EPIC), which introduces explicit item-level competition into SID denoising. EPIC constructs a personalized posterior over feasible candidate items using the current generation context and the user's recent interactions, then projects this distribution back to unresolved SID positions to guide subsequent token decisions. The pretrained backbone remains frozen and requires no additional decoder forward pass. Experiments on four Amazon b
    
[^11]: 从主题相关性到可回答性：面向对话式检索的蕴含蒸馏方法

    From Topical Relevance to Answerability: Entailment Distillation for Conversational Retrieval

    [https://arxiv.org/abs/2609.03482](https://arxiv.org/abs/2609.03482)

    CLEAR框架通过蕴含蒸馏将答案-段落蕴含监督迁移到重排序器中，使对话式检索从主题相关性转向真正的可回答性，并借助溯因召回模块引入低相似度但可回答的段落以提升检索效果。

    

    现有的对话式检索器通常将主题相关性作为可回答性的替代指标。然而，与对话上下文高度匹配的段落并不一定是支持正确答案的段落。我们将这种不匹配识别为系统性的可回答性差距。为了解决这一问题，我们提出了CLEAR，一个将对话式检索从主题相关性转向可回答性的框架。CLEAR的核心是蕴含蒸馏，它将答案-段落蕴含监督迁移到交叉编码器重排序器中，使重排序器能够在推理时区分支持答案的段落与主题干扰项，而无需依赖答案。CLEAR还辅以一个以段落为中心的溯因召回模块，通过使用大语言模型从段落中推断可回答的查询，将低相似度但可回答的段落纳入候选池。在TopiOCQA、QReCC以及域外TREC CAsT数据集上，CLEAR持续（原文截断）……

    arXiv:2609.03482v1 Announce Type: new  Abstract: Existing conversational retrievers commonly treat topical relevance as a proxy for answerability. However, a passage that closely matches the dialogue context is not necessarily the one that supports the correct answer. We identify this mismatch as a systematic answerability gap. To address this issue, we propose CLEAR, a framework that shifts conversational retrieval from topical relevance to answerability. The core of CLEAR is entailment distillation, which transfers answer-passage entailment supervision into a cross-encoder reranker so that the reranker discriminates answer-supporting passages from topical distractors at inference time, without requiring answers. CLEAR is complemented by a passage-centric abductive recall module that brings low-similarity yet answerable passages into the candidate pool by inferring answerable queries from passages with an LLM. Across TopiOCQA, QReCC, and out-of-domain TREC CAsT datasets, CLEAR consist
    
[^12]: ExplainRoute：面向不直接给出答案的编程导师的部署前审计框架

    ExplainRoute: A Pre-Deployment Audit Framework for Non-Answer-Giving Programming Tutors

    [https://arxiv.org/abs/2609.03470](https://arxiv.org/abs/2609.03470)

    本文提出ExplainRoute，一个面向不直接给出答案的编程导师的部署前审计框架，通过机器可检验的契约审计信息边界、响应极性、失败闭环和学习者解释可见性等维度。

    

    编程导师应该支持学习者自己的解释，而不是立即提供标准答案。我们提出了ExplainRoute，一个面向不直接给出答案的编程导师的部署前审计框架。给定一行代码和学习者的解释，该框架会估计解释状态，并从两种有界响应中选择一种：费曼式的自我解释提示或苏格拉底式支架。该框架通过机器可检验的契约公开其状态、策略、引用的代码片段和泄露风险。与仅凭流利程度对导师进行排名的基准不同，ExplainRoute在课堂部署之前审计信息边界、响应极性、失败闭环以及学习者解释可见性的价值。我们在包含1,770对样本的SelfCode语料库上使用代码组划分方式进行离线评估，其中443对样本保留在11个未被触及的留出组中。评估比较了直接给出答案、固定的开放式自我解释、固定的苏格拉底支架……

    arXiv:2609.03470v1 Announce Type: new  Abstract: Programming tutors should support learners' own explanations rather than immediately providing model answers. We present ExplainRoute, a pre-deployment audit framework for non-answer-giving programming tutors. Given a code line and a learner explanation, it estimates the explanation state and selects one of two bounded responses: a Feynman-style self-explanation prompt or a Socratic scaffold. The framework exposes its state, strategy, cited code fragment, and leakage risk through a machine-checkable contract. Unlike benchmarks that rank tutors by fluency alone, ExplainRoute audits information boundaries, response polarity, failure closure, and the value of learner-explanation visibility before classroom deployment. We evaluate it offline on the 1,770-pair SelfCode corpus using a code-group split, with 443 pairs reserved in 11 untouched holdout groups. The evaluation compares direct answers, fixed open self-explanation, fixed Socratic sca
    
[^13]: 何时检索有益：面向单轮心理健康问答的选择性检索

    When Retrieval Helps: Selective Retrieval for Single-Turn Mental-Health QA

    [https://arxiv.org/abs/2609.03454](https://arxiv.org/abs/2609.03454)

    该研究针对单轮心理健康问答提出一种轻量级选择性检索策略，通过心理教育需求、应对需求、回答具体性三个效用维度及安全触发机制判断检索的必要性，从而在发挥检索增强优势的同时避免其负面影响。

    

    检索增强生成（RAG）能够提升大语言模型回答的具体性和事实依据，但在单轮心理健康问答中，其效果并非始终有益，因为用户提问往往同时涉及情绪困扰、治疗关切以及安全敏感需求。我们研究了检索在心理健康问答中何时有益、何时有害，以及一种轻量级的选择性检索策略能否更好地控制这种权衡。我们通过三个基于草稿的条件化效用维度来量化检索需求：心理教育需求、应对需求和回答具体性，并结合基于规则的安全触发机制。借鉴coTherapist等基于心理治疗框架的RAG系统，我们构建了一个紧凑且可控的指南语料库，涵盖应对策略、心理教育和安全资源。我们使用QLoRA在MentalChat16K上对指令微调生成器进行微调，并比较了闭卷（Closed-book）、始终检索（Always Retrieval）以及……（摘要原文在此处截断）

    arXiv:2609.03454v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) can improve the specificity and grounding of large language model responses, but its effect is not uniformly beneficial in single-turn mental-health question answering, where user queries often combine emotional distress, treatment concerns, and safety-sensitive needs. We study when retrieval helps or hurts mental-health QA, and whether a lightweight selective retrieval policy can better control this trade-off. We operationalize retrieval need using three draft-conditioned utility dimensions: psychoeducational need, coping need, and response specificity, together with a rule-based safety trigger. Following psychotherapy-grounded RAG systems such as coTherapist, we construct a compact and controllable guideline corpus comprising coping-strategy, psychoeducational, and safety resources. We fine-tune an instruction-tuned generator on MentalChat16K using QLoRA and compare Closed-book, Always Retrieval, an
    
[^14]: 预算化继承式智能体记忆验证中的计划指针与记录指令形式

    Plan Pointers and Record-Directive Form in Budgeted Verification of Inherited Agent Memory

    [https://arxiv.org/abs/2609.03450](https://arxiv.org/abs/2609.03450)

    该论文通过十二项注册研究发现，写入智能体记忆库的指令形式（准则、裸ID或指针）会以高度模型依赖的方式显著影响预算受限下的记录选择，长度匹配准则可带来35分的提升，但附加ID可能完全抵消准则的效果。

    

    arXiv:2609.03450v1 公告类型：cross。摘要：一个继承了六条单行记忆的智能体在行动前最多只能拉取一条存档的源记录；写入存储中的指令可以引导这一选择：可以是指向该记录的指针、识别该记录的准则，或两者兼有。在同一仪器谱系上的十二项注册研究（共14,760次尝试）中，我们测量了每种指令形式下请求的去向。在六个直接提供商模型上，长度匹配的准则比裸ID高出+35.0个点 [+31.2, +38.8]（研究D）；而在九个模型的OpenRouter服务面板上，该对比未能通过注册的优越性规则（研究E）。在三个Claude模型上，附加ID会抵消准则的效果（Opus 5: 从40/40降至0/40；研究F-x）；六次字节匹配的编辑使每个精确字符串都产生了各自的效应（研究G），并且在每单元八十次运行的重跑中，三十个复现对比中有十五个处于误差范围内，十五个未获解决，没有一个超出范围（研究G'）。批准行（在Opus 5上+96.0个点）以及一个（摘要在此处截断）

    arXiv:2609.03450v1 Announce Type: cross  Abstract: An agent that inherits six one-line memories may pull at most one archived source record before acting; a directive written into the store can steer that choice: a pointer to the record, a criterion that identifies it, or both. Across twelve registered studies on one instrument lineage (14,760 attempts) we measured where the request goes under each form. On six direct-provider models a length-matched criterion exceeded a bare id by +35.0 points [+31.2, +38.8] (Study D); the contrast failed its registered superiority rule on a nine-model OpenRouter-served panel (Study E). Appending the id cancelled the criterion on three Claude models (Opus 5: 40/40 to 0/40; Study F-x); six byte-matched edits gave each exact string its own effect (Study G), and a re-run at eighty runs per cell left fifteen of thirty replication contrasts within the margin, fifteen unresolved and none beyond (Study G'). A ratification line (+96.0 points on Opus 5) and a 
    
[^15]: Spruce：基于紧凑嵌入的可扩展隐私外包检索

    Spruce: Scalable Private Outsourced Retrieval Using Compact Embeddings

    [https://arxiv.org/abs/2609.03376](https://arxiv.org/abs/2609.03376)

    Spruce通过将紧凑二进制嵌入表示与密码学协议协同设计，用汉明距离计算取代全语料库嵌入评分，将百万文档规模的隐私外包检索性能显著提升。

    

    arXiv:2609.03376v1 公告类型：cross 摘要：检索增强生成（RAG）使得在大规模文档集合上进行密集检索成为标准构建模块。组织越来越多地将向量索引外包给不受信任的云服务，这会导致专有语料库和用户查询暴露。密码学保护面临挑战，因为每个查询都需要搜索语料库规模的状态，导致计算量、相关随机数和通信量随语料库规模增长。在百万文档规模下，朴素的安全实现每个查询需要数分钟时间和约90GB的通信量。即使是近期优化的系统也需要10至22秒。我们提出Spruce（基于紧凑嵌入的可扩展隐私外包检索），它将表示学习与密码学协议进行协同设计。Spruce学习紧凑的二进制编码，在为全精度重排序保留候选结果的同时，在双服务器多方计算框架下用高效的汉明距离计算取代了全语料库的嵌入评分。

    arXiv:2609.03376v1 Announce Type: cross  Abstract: Retrieval-Augmented Generation (RAG) has made dense retrieval over large document collections a standard building block. Organizations increasingly outsource vector indexes to untrusted clouds, exposing proprietary corpora and user queries. Cryptographic protection is challenging because each query searches corpus-scale state, causing computation, correlated randomness, and communication to grow with the corpus. At million-document scale, a naive secure implementation takes minutes and about 90 GB of communication per query. Even recent optimized systems require 10--22 seconds.   We propose Spruce (Scalable Private Outsourced Retrieval Using Compact Embeddings), which co-designs representations with the cryptographic protocol. Spruce learns compact binary codes that preserve candidates for full-precision reranking, replacing corpus-wide embedding scoring with efficient Hamming-distance computation under two-server multi-party computati
    
[^16]: HypRQ-VAE：面向长尾感知生成式推荐系统的双曲物品索引

    HypRQ-VAE: Hyperbolic Item Indexing for Long-Tail-Aware Generative Recommender Systems

    [https://arxiv.org/abs/2609.03369](https://arxiv.org/abs/2609.03369)

    提出首个在双曲空间中学习物品索引的框架HypRQ-VAE，利用双曲残差量化变分自编码器解决生成式推荐中欧几里得空间难以建模物品长尾分布的问题。

    

    序列推荐系统将用户行为建模为物品ID序列，而近期的生成式方法则借助大语言模型（LLM）将推荐任务转化为语言建模问题。虽然这一范式融入了丰富的文本语义，但也带来了一个根本性的不匹配：LLM操作的是文本标记，而推荐系统依赖的是离散的物品索引。这种错位常常导致生成式推荐中出现幻觉问题。现有方法试图通过在欧几里得空间中学习物品词表来弥合这一差距，但它们难以刻画现实世界物品目录中固有的长尾分布——少数头部物品占据主导地位，而大量尾部物品则反映了用户的小众偏好。为了解决这一问题，我们提出了双曲残差量化变分自编码器，这是首个在双曲空间中学习物品索引的框架。HypRQ-VAE利用双曲空间的独特性质……

    arXiv:2609.03369v1 Announce Type: new  Abstract: Sequential recommender systems model user behavior as item ID sequences, while recent generative methods cast recommendation as a language modeling task using large language models (LLMs). While this paradigm incorporates rich textual semantics, it introduces a fundamental mismatch: LLMs operate on text tokens, whereas recommender systems depend on discrete item indices. This misalignment often leads to hallucinations in generative recommendations. Existing methods attempt to bridge this gap by learning item vocabularies in Euclidean space, but they struggle to model the inherent long-tail distribution of real-world catalogs, where a small number of head items dominate, and a vast number of tail items reflect users' niche preferences. To address this issue, we introduce Hyperbolic Residual-Quantized Variational AutoEncoder (HypRQ-VAE), the first framework to learn item indexing in hyperbolic space. HypRQ-VAE leverages the unique properti
    
[^17]: SciLENS：用于科学本地化证据导航与综合的强化学习驱动自主智能体

    SciLENS: RL-Driven Autonomous Agents for Scientific Localized Evidence Navigation and Synthesis

    [https://arxiv.org/abs/2609.03338](https://arxiv.org/abs/2609.03338)

    SciLENS提出了一个完全本地化运行的强化学习驱动自主智能体框架，开创性地将结构化可视化集成到推理循环中以缓解上下文耗尽，并通过自动化的多跳子图合成流水线与跨模型共识验证实现无需人工标注的智能体训练，从而在约1200万条学术记录上实现科学证据的本地化导航与综合。

    

    科学文献综合智能体日益依赖专有在线服务，这限制了系统的可复现性、隐私保护和离线部署能力。为应对这一挑战，我们提出了SciLENS（科学本地化证据导航与综合系统），一个完全本地化运行的自主智能体框架，其构建于索引约1200万条学术记录的双层基础设施之上。SciLENS开创性地将结构化可视化作为可操作工具集成到推理循环中，使智能体能够将复杂的引用拓扑结构压缩为经过验证的数据驱动图表，从而在宏观层面的文献综合过程中缓解上下文耗尽问题。为了在没有人工标注的情况下训练该智能体，我们开发了一个自动化数据合成流水线，从引用知识图谱中提取多跳子图，并通过20个前沿模型的跨模型共识进行验证。随后，智能体通过反向分解评分准则进行对齐训练。

    arXiv:2609.03338v1 Announce Type: new  Abstract: Scientific literature synthesis agents increasingly rely on proprietary online services, limiting reproducibility, privacy, and offline deployment. To address this challenge, we introduce SciLENS Scientific Localized Evidence Navigation and Synthesis), a fully local autonomous agent framework operating on a dual-tier infrastructure indexing approximately 12 million academic records. SciLENS pioneers the integration of structural visualization as an actionable tool within the reasoning loop, enabling the agent to compress complex citation topologies into validated data-driven charts and thereby mitigate context exhaustion during macro-level synthesis. To train the agent without human annotation, we develop an automated data synthesis pipeline that extracts multi-hop subgraphs from a citation knowledge graph, verified by cross-model consensus among 20 frontier models. The agent is subsequently aligned through a reverse-decomposition rubric
    
[^18]: SelfDR：基于推理自蒸馏的大语言模型推荐

    SelfDR: Self-Distillation from Reasoning for LLM-Based Recommendation

    [https://arxiv.org/abs/2609.03313](https://arxiv.org/abs/2609.03313)

    SelfDR提出了一种推理自蒸馏框架，将大语言模型自身经推理增强的预测能力蒸馏为直接推荐，在无需外部模型的情况下同时提升推荐效果与推理效率。

    

    近年来，大语言模型（LLMs）已成为推荐系统中强大的骨干模型。为了更好地激发其能力，推理机制被广泛引入，以帮助大语言模型解读丰富的文本信号并提高推荐准确性。然而，显式地生成中间推理轨迹往往会带来高昂的计算成本，这限制了其在现实世界推荐系统中的实际部署。为了应对这一挑战，我们提出了SelfDR，一个面向基于大语言模型推荐的推理自蒸馏框架。SelfDR将大语言模型自身经推理增强的预测进行蒸馏，从而直接产生推荐结果，在提升推荐效果的同时保持推理效率。框架中的所有组件均构建于同一个基础大语言模型之上，无需依赖任何外部模型。具体而言，教师推荐器是通过以下游性能作为奖励来训练推理器而构建的，使……

    arXiv:2609.03313v1 Announce Type: new  Abstract: Large Language Models (LLMs) have recently emerged as powerful backbones for recommendation. To better elicit their capabilities, reasoning has been widely incorporated to help LLMs interpret rich textual signals and improve recommendation accuracy. However, explicitly generating intermediate reasoning traces often incurs substantial computational costs, which limits practical deployment in real-world recommender systems. To address this challenge, we propose SelfDR, a Self-Distillation from Reasoning framework for LLM-based Recommendation. SelfDR distills an LLM's own reasoning-enhanced predictions to produce recommendations directly, improving recommendation effectiveness while maintaining inference efficiency. All components in the framework are built on the same base LLM, without relying on any external models. Specifically, the teacher recommender is constructed by training a reasoner with downstream performance as the reward, enabl
    
[^19]: DoPR：用于高效LLM重排序的可重用压缩文档前缀

    DoPR: Reusable Compressed Document Prefixes for Efficient LLM Reranking

    [https://arxiv.org/abs/2609.03311](https://arxiv.org/abs/2609.03311)

    DoPR提出一种压缩文档前缀框架，将文档处理与查询处理解耦，通过离线预计算并跨查询重用压缩的文档前缀状态，使LLM在线重排序只需处理查询部分，从而实现高达8倍的在线计算成本降低。

    

    大语言模型（LLM）是有效的重排序器，但逐点重排序在不同查询中会重复处理同一文档，造成大量冗余的文档侧计算。我们提出DoPR，一种压缩文档前缀框架，将离线文档处理与在线重排序解耦。DoPR首先选择与查询无关的文档表示，并将其转换为压缩的文档前缀状态，这些状态在离线阶段预计算，并在文档被检索到时重复使用。在在线重排序过程中，模型仅需处理查询和评分标记即可对每个查询-文档对进行评分，文档信息由存储的前缀状态提供。该设计通过文档侧压缩和跨查询的前缀状态重用降低了在线成本。在TREC DL、BEIR和BRIGHT数据集上使用0.6B到8B规模的Qwen3模型进行的实验表明，DoPR实现了高达8.0倍的在线文档（原文此处被截断）。

    arXiv:2609.03311v1 Announce Type: new  Abstract: Large language models (LLMs) are effective rerankers, but pointwise reranking repeatedly processes the same document across different queries, causing substantial redundant document-side computation. We propose \textbf{DoPR}, a compressed document prefix framework that decouples offline document processing from online reranking. DoPR first selects query-independent document representations and converts them into compressed document prefix states, which are precomputed offline and reused whenever the document is retrieved. During online reranking, the model scores each query-document pair by processing only the query and scoring token, with document information supplied by the stored prefix states. This design reduces online cost through both document-side compression and cross-query prefix-state reuse. Experiments on TREC DL, BEIR, and BRIGHT with Qwen3 models from $0.6$B to $8$B show that DoPR achieves up to 8.0$\times$ online document-
    
[^20]: UniCon：一种面向CTR预测的统一上下文中心建模范式

    UniCon: A Unified Context-Centric Modeling Paradigm for CTR Prediction

    [https://arxiv.org/abs/2609.03290](https://arxiv.org/abs/2609.03290)

    本文提出UniCon，一种统一的以上下文为中心的CTR预测建模范式，将用户历史行为与当前请求作为同质的上下文单元进行统一建模，克服传统异构信号划分的局限，提升扩展效率与预测质量。

    

    统一建模已成为工业级点击率（CTR）预测的重要发展方向。现有方法通常在token级别统一序列信号与非序列信号，在共享骨干网络中建模它们的交互，并通过增大模型容量来改善扩展性。然而，这种划分源于传统的特征工程实践，与底层的决策过程并不一致。用户行为本质上是一系列同质的上下文单元；在输入组织的层面，历史行为与当前请求的区别仅在于其结果是已被观测的还是尚待预测的。将二者视为异构信号会掩盖用户决策上下文内部的结构依赖关系，从而同时限制了扩展效率和预测质量。这一局限在电商货架、瀑布流信息流等上下文丰富的场景中尤为突出。为解决这一问题……

    arXiv:2609.03290v1 Announce Type: new  Abstract: Unified modeling has become a major direction for industrial click-through rate (CTR) prediction. Existing approaches typically unify sequential and non-sequential signals at the token level, model their interactions in a shared backbone, and increase model capacity to improve scaling behavior. However, this division originates from legacy feature-engineering practice and is misaligned with the underlying decision process. User behavior is inherently a sequence of homogeneous context units; at the level of input organization, historical behavior and the current request differ only in whether their outcomes are observed or remain to be predicted. Treating them as heterogeneous signals obscures structural dependencies within the user's decision context, limiting both scaling efficiency and prediction quality. This limitation is particularly pronounced in context-rich scenarios such as e-commerce shelves and waterfall feeds. To address this
    
[^21]: SHELF：一个用于多任务书目基准测试的合成测试框架

    SHELF: A Synthetic Harness for Multi-Task Bibliographic Benchmarking

    [https://arxiv.org/abs/2609.03047](https://arxiv.org/abs/2609.03047)

    SHELF是一个基于美国国会图书馆词表生成6万余篇合成文档的Python系统，为图书馆和档案馆的书目工作提供了涵盖分类、聚类、检索等多任务的系统性基准测试框架。

    

    图书馆和档案馆在人员和计算预算有限的情况下管理着大量馆藏，然而现有的常见基准测试并未系统地检验其书目工作。他们需要了解哪些方法适用于自己的任务，以及运行这些方法需要什么条件。SHELF（用于评估LLM适应性的合成测试框架，Synthetic Harness for Evaluating LLM Fitness）填补了这一空白。它是一个Python系统，能够将带标签的分类法、编写规范和生成预算转化为受控的基准数据和评估任务。首个发布版本包含62,899篇基于美国国会图书馆词表、由模型生成的文档，涵盖分类、聚类、检索、成对分类和指令检索等任务。我们比较了TF、TF-IDF、BM25、流行的编码器模型，以及仅在主题分类任务上测试的零样本解码器；每种方法仅出现在支持它的任务上。主题分类的准确率达到0.8887，而体裁-形式分类仅达到0.2605……

    arXiv:2609.03047v1 Announce Type: cross  Abstract: Libraries and archives manage large collections with limited staff and computing budgets, yet common benchmarks do not systematically test their bibliographic work. They need to know which methods work for their tasks and what those methods require to run. SHELF, the Synthetic Harness for Evaluating LLM Fitness, addresses this gap. It is a Python system that turns labelled taxonomies, writing specifications, and a generation budget into controlled benchmark data and evaluation tasks. This first release contains 62,899 model-written documents based on Library of Congress vocabularies, with tasks for classification, clustering, retrieval, pair classification, and instruction retrieval. We compare TF, TF-IDF, BM25, popular encoders, and, on subject classification only, zero-shot decoders; each method appears only on tasks that support it. Subject classification reaches 0.8887, while genre-form classification reaches only 0.2605, and sever
    
[^22]: Reflect-SQL：一种基于自我反思的Text-to-SQL框架

    Reflect-SQL: A Self-Reflection Based Framework for Text-to-SQL

    [https://arxiv.org/abs/2609.02944](https://arxiv.org/abs/2609.02944)

    Reflect-SQL是一个基于多阶段自我反思的Text-to-SQL新框架，通过知识库理解晦涩的数据库模式，并利用LLM-as-a-judge驱动的评分机制在反馈循环中迭代优化每个阶段的SQL生成结果。

    

    通过自然语言实现数据访问的民主化是现代企业的重要目标，但Text-to-SQL的实际应用受到现实世界复杂性的严重阻碍：1. 晦涩且庞大的数据库模式；2. 由于模式的固定结构设置和用户查询的模糊性，导致无法有效检索相关的表和列；3. 由于缺乏健壮的验证和纠错机制，生成了语法或逻辑上有缺陷的SQL。为了解决这些系统性挑战，我们提出了Reflect-SQL，这是一个新颖的Text-to-SQL框架，其基于多阶段自我反思方法，利用知识库来理解晦涩的数据库模式，建立有效的检索流程以及生成语法/语义正确的SQL的系统。我们的系统并非采用单次尝试的方式，而是在相互关联的反馈循环中采用LLM-as-a-judge驱动的评分机制，在每个阶段迭代地优化结果。

    arXiv:2609.02944v1 Announce Type: cross  Abstract: Democratizing data access through natural language is a crucial goal for modern enterprises, but the practical adoption of Text-to-SQL is critically hindered by real-world complexities: 1. Obscure and large database schemas, 2. Ineffective retrieval of relevant tables and columns due to structured setting of schemas and vague user query, 3. Generation of syntactically or logically flawed SQL due to a lack of robust validation and correction mechanism. To address these systemic challenges, we introduce Reflect-SQL, a novel framework for Text to SQL, grounded in multi-stage self-reflection approach to develop understanding of obscure schema using a knowledge base, setup a process for effective retrieval and system to generate syntactically/semantically SQL. Instead of a single-pass attempt, our system employs an LLM-as-a-judge driven scoring mechanism within interconnected feedback loops to iteratively refine the results at every stage. 
    
[^23]: CHSR-RRF：一种面向教育RAG的课程门控混合检索框架，结合倒数排序融合与泄漏感知基准测试

    CHSR-RRF: A curriculum-gated hybrid retrieval framework with reciprocal rank fusion and leakage-aware benchmarking for educational RAG

    [https://arxiv.org/abs/2609.02913](https://arxiv.org/abs/2609.02913)

    提出课程门控混合检索框架CHSR-RRF，通过在检索前施加课程元数据约束并结合倒数排序融合，将课程泄漏减少4.6倍且不损失召回率，同时发布了包含126个案例的泄漏感知教育检索基准CERB。

    

    检索增强生成（RAG）在教育问答中的应用日益广泛，但标准检索器仅优化主题相关性，而未强制保证课程层面的有效性。在学校环境中，一段文本即使主题相关，但如果它来自错误的学科、年级或考试情境，也可能并不合适；我们将这种失败模式称为“课程泄漏”。我们提出了CHSR-RRF，一个课程门控混合检索框架，它在检索之前应用元数据约束，然后结合稀疏检索与稠密检索，通过倒数排序融合（RRF）和确定性重排序进行融合。我们还引入了CERB，一个包含126个案例的课程约束检索基准，具有层次感知的相关性标签和显式的泄漏标注。在61个案例的初步实验中，检索前门控使课程泄漏减少4.6倍（p<0.001），同时保持了排序召回率；而将相同的约束施加在检索之后，则会使召回率和精确范围成功率降至零（p=0.039）。

    arXiv:2609.02913v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) is increasingly used in educational question answering, but standard retrievers optimize topical relevance without enforcing curriculum validity. In school settings, a passage can be relevant yet inappropriate if it comes from the wrong subject, level, or examination context; we call this failure mode curriculum leakage. We present CHSR-RRF, a curriculum-gated hybrid retrieval framework that applies metadata constraints before retrieval, then combines sparse and dense search with reciprocal rank fusion and deterministic reranking. We also introduce CERB, a 126-case benchmark for curriculum-constrained retrieval with hierarchy-aware relevance labels and explicit leakage annotations. On a 61-case pilot, pre-retrieval gating reduces leakage by 4.6x ($p<0.001$) while preserving ranked recall, whereas applying the same constraints after retrieval collapses recall and exact-scope success to zero ($p=0.039$)
    
[^24]: R²Adapter：面向高效混合RAG的路由与重写适配器

    R$^{2}$Adapter: A Routing and Rewriting Adapter for Efficient Hybrid RAG

    [https://arxiv.org/abs/2609.02894](https://arxiv.org/abs/2609.02894)

    提出了轻量级即插即用的路由与重写适配器R²Adapter，可动态将查询分配给原生RAG或基于图的RAG，仅对真正受益于图推理的查询进行图检索，从而降低不必要的开销并减少对底层大模型的依赖。

    

    检索增强生成（RAG）已成为利用非参数知识增强大语言模型（LLM）的主流范式。原生RAG能够高效处理简单查询，但在关系推理或多跳推理方面表现不佳。基于图的RAG缓解了这一问题，但会带来更高的推理复杂度和延迟。在实际应用中，用户查询的复杂度差异巨大，采用固定不变的RAG策略并非最优。然而，现有的混合文本-图RAG方法通常依赖启发式方法和基于LLM的路由，导致不必要的开销，并对底层LLM有很强的依赖。为应对这些挑战，我们提出了R²Adapter，一个轻量级、即插即用的路由与重写适配器，旨在动态地将查询分配给原生RAG或基于图的RAG。通过仅对真正能从图推理中受益的查询进行路由，R²Adapter减少了不必要的图检索开销。

    arXiv:2609.02894v1 Announce Type: new  Abstract: Retrieval-Augmented Generation (RAG) has become a prevailing paradigm for enhancing Large Language Models (LLMs) with non-parametric knowledge. Vanilla RAG efficiently handles simple queries but struggles with relational or multi-hop reasoning. Graph-based RAG alleviates this issue but incurs higher inference complexity and latency. In practice, user queries can differ significantly in their complexity, rendering a fixed RAG strategy suboptimal. However, existing hybrid text-graph RAG methods typically rely on heuristic and LLM-based routing, resulting in unnecessary overhead and strong dependence on the underlying LLM. To address these challenges, we propose R$^{2}$Adapter, a lightweight plug-in Routing and Rewriting Adapter designed to allocate queries between vanilla and graph-based RAG dynamically. By routing only the queries that genuinely benefit from graph-based reasoning, R$^{2}$Adapter reduces unnecessary graph retrieval overhea
    
[^25]: ViSAR：面向视觉文档问答的无需训练的自适应k值检索方法

    ViSAR: Training-Free Adaptive-$k$ Retrieval for Visual Document Question Answering

    [https://arxiv.org/abs/2609.02486](https://arxiv.org/abs/2609.02486)

    提出了一种无需训练的自适应k值检索方法ViSAR，通过在嵌入空间中构建查询条件的页面级相似度矩阵来动态确定检索页面数量，在保持或提升答案准确性的同时将RAG延迟降低高达58.7%。

    

    文档视觉问答通常利用检索增强生成技术，其中晚期交互编码器常被用于识别与用户查询相关的文档页面，然后由大型视觉-语言模型生成答案。现有方法通常无论查询复杂度如何都检索固定数量的前k个页面，这会增加大型视觉-语言模型的延迟，并可能降低答案的准确性。我们提出了ViSAR（视觉语义激活检索），这是一种面向晚期交互视觉文档检索的无需训练的自适应k值检索方法。ViSAR直接在嵌入空间中运行，构建以查询为条件的页面级相似度矩阵，突出与查询相关的语义，并动态确定需要检索的页面数量。在多个编码器和大型视觉-语言模型上的实验表明，ViSAR能够检索紧凑且适应查询的页面集合，将RAG延迟降低高达58.7%，同时保持或提升答案准确性。

    arXiv:2609.02486v1 Announce Type: cross  Abstract: Document Visual Question Answering (DocVQA) often leverages Retrieval-Augmented Generation (RAG), where late-interaction encoders are commonly used to identify document pages relevant to a user query, before answer generation by a Large Vision-Language Model (LVLM). Existing approaches typically retrieve a fixed top-$k$ number of pages regardless of query complexity, which increases LVLM latency and may degrade answer accuracy. We introduce ViSAR (Visual Semantic Activation Retrieval), a training-free adaptive-$k$ retrieval method for late-interaction visual document retrieval. ViSAR operates directly in the embedding space to construct a query-conditioned page-level similarity matrix that highlights query-relevant semantics and dynamically determines the number of pages to retrieve. Across multiple encoders and LVLMs, ViSAR retrieves compact, query-adapted page sets that reduce RAG latency by up to 58.7\%, while maintaining or improvi
    
[^26]: 基于结构化码本与词元序列解码的路径级地图匹配方法

    Route Based Map Matching via a Structured Codebook and Token Sequence Decoding

    [https://arxiv.org/abs/2607.22543](https://arxiv.org/abs/2607.22543)

    该论文提出一种基于路径的轻量级地图匹配方法，将候选路径编码为线路与交叉口词元构成的码本，并借助DAFSA×Levenshtein自动机将GPS轨迹匹配转化为模糊词元序列对齐解码，使单次查询成本比暴力扫描低多个数量级。

    

    本研究提出了一种高效且计算开销小的基于路径的地图匹配方法，适用于城市快速路网络上的GPS轨迹数据。其核心思想是利用由命名线路和命名交叉口构成的符号化结构——这一结构在低层级地图匹配中一直未被充分利用。我们将每条候选路径表示为线路名称与交叉口名称组成的序列，将此类序列的集合作为“路径码本”，并把地图匹配问题形式化为探测轨迹与码本成员之间的评分对齐问题。探测轨迹通过网格量化器（一种预先将每个坐标映射到线路或交叉口词元的网格）转换为词元序列，而解码器按构造保证返回码本中的某一个成员。该码本通过DAFSA×Levenshtein自动机进行索引，这是一种源自近似字符串匹配与语音识别领域的模糊查找技术，使得每次查询的解码成本比暴力扫描低若干个数量级。我们对该方法进行了评估。

    arXiv:2607.22543v2 Announce Type: replace-cross  Abstract: This study proposes an efficient and computationally light route based map matching method for GPS track data on urban expressway networks. The key idea is to exploit a symbolic structure of named lines and named junctions that link level map matching leaves unused. We represent each candidate route as a sequence of line and junction names, take the set of such sequences as a route codebook, and formulate map matching as scored alignment of a probe trajectory against members of the codebook. Probes become token sequences via a mesh quantizer, a precomputed grid mapping each coordinate to a line or junction token, and the decoder returns a member of the codebook by construction. The codebook is indexed by a DAFSA $\times$ Levenshtein automaton, a fuzzy lookup technique from approximate string matching and speech recognition; the per query decoding cost is orders of magnitude lower than a brute force scan. We evaluate the method 
    
[^27]: PACMS：面向大语言模型智能体的次模上下文选择可插拔引擎

    PACMS: Submodular Context Selection as a Pluggable Engine for LLM Agents

    [https://arxiv.org/abs/2606.20047](https://arxiv.org/abs/2606.20047)

    该论文提出 PACMS，一种基于次模优化的上下文选择方法，作为可插拔引擎嵌入 LLM 智能体框架，用以替代对主题不敏感的按最近性截断机制，在上下文超出 token 预算时智能地保留与当前查询相关的信息。

    

    对话式和工具调用型大语言模型（LLM）智能体的上下文窗口会同时从多个方向被填满。随着会话的进行，智能体会不断积累用户与助手的对话轮次、来自持久化记忆库的条目，以及通常占比最大的部分——工具调用的原始输出，例如文件读取、搜索结果和 API 响应。一旦累积上下文超过模型的 token 预算，框架就必须决定保留哪些内容。目前的主流机制是按最近性截断，有时辅以周期性摘要。这种机制对主题不敏感：会话早期确立的事实仅仅因为“太旧”就被丢弃，即使当前用户询问的恰恰就是那个事实；相反，冗长但与当前无关的近期内容却被保留下来。那些必须跨多轮回忆信息的智能体——这正是记忆功能的核心场景——恰恰是最近性截断失效之处。现有的替代方案位于……（原文摘要在此处截断）

    arXiv:2606.20047v2 Announce Type: replace  Abstract: Conversational and tool-using LLM agents operate over a context window that fills from several directions simultaneously. As a session proceeds, the agent accumulates user and assistant turns, entries drawn from a persistent memory store, and often largest of all, the verbatim outputs of tool calls such as file reads, search results, and API responses. Once the cumulative context exceeds the model's token budget, the framework must decide what to keep.   The prevailing mechanism is recency truncation, sometimes paired with periodic summarization. This is topic-blind: a fact established early in a session is discarded simply because it is old, even when the current user query is about exactly that fact; conversely, verbose but irrelevant recent material is retained. Agents that must recall information across many turns, the defining case for memory, are precisely where recency truncation fails.   Existing alternatives sit outside the 
    
[^28]: 稠密检索器中的位置偏差是固有存在的还是从数据中学习到的？

    Is Position Bias in Dense Retrievers Built In-or Learned from Data?

    [https://arxiv.org/abs/2605.26578](https://arxiv.org/abs/2605.26578)

    研究发现稠密检索器的位置偏差主要源于训练数据中证据位置的分布，而非模型架构本身，通过位置均衡的训练数据可将位置敏感性降低57%至87%。

    

    稠密检索器表现出位置偏差，倾向于查询相关信息出现在文档开头附近的文档，而当相关信息出现在较后位置时检索性能会下降。虽然先前关于稠密检索器位置偏差的研究主要集中在架构层面的解释上，我们则研究了训练数据中证据的位置分布如何影响检索层面的偏差方向。为了验证这一点，我们构建了合成的位置定向训练集，其中查询相关证据出现在文档的开头、中间或结尾，并在位置偏斜和均衡的训练分布下微调了八个架构各异的预训练模型。在排序层面，我们在所考察的所有模型中观察到了强烈的方向性模式：位置偏斜的训练分布会使模型偏好对应位置的证据。位置均衡的训练在位置感知基准上将位置敏感性降低了57%至87%。

    arXiv:2605.26578v2 Announce Type: replace  Abstract: Dense retrievers exhibit positional bias, favoring documents whose query-relevant information appears near the beginning and degrading retrieval performance when the information appears later. While prior work on positional bias in dense retrievers has largely focused on architectural explanations, we study how the positional distribution of evidence in training data affects retrieval-level bias direction. To test this, we construct synthetic position-targeted training sets in which query-relevant evidence appears at the beginning, middle, or end of documents, and fine-tune eight architecturally diverse pretrained models under position-skewed and balanced training distributions. At the ranking level, we observe a strong directional pattern across the examined models: skewed training distributions favor evidence at the corresponding positions. Position-balanced training reduces positional sensitivity by 57--87\% on position-aware benc
    
[^29]: 基于大语言模型的分布式多智能体系统中长期记忆的成本与精度

    Cost and Accuracy of Long-Term Memory in Distributed Multi-Agent Systems Based on Large Language Models

    [https://arxiv.org/abs/2601.07978](https://arxiv.org/abs/2601.07978)

    本文构建了一个独立可复现的云边环境测试平台，首次从系统级成本（CPU时间、内存、磁盘I/O、网络）角度对mem0、Graphiti、cognee三种长期记忆框架与RAG及全上下文基线在分布式多智能体系统中的精度和资源开销进行了系统对比评估。

    

    arXiv:2601.07978v5 公告类型：替换。摘要：长期记忆（LTM）是新兴的智能体互联网（IoA）中基于大语言模型（LLM）的智能体的基础，其中分布式多智能体系统（DMAS）跨越云和边缘网络。现有的评估通常由框架提供者发布，侧重于令牌使用量和延迟，很少考虑系统级成本或在DMAS中的部署情况。本文通过一个独立、可复现的测试平台弥补了这些空白，该平台在模拟的云边环境中评估精度、延迟、CPU时间、峰值内存、磁盘I/O和网络使用量。研究比较了三个获得风险投资资助的框架——涵盖向量、图和混合架构的mem0、Graphiti和cognee——以及检索增强生成（RAG）和全上下文两种基线方法，在LoCoMo基准上于无约束和有约束的网络场景下进行测试。结果呈现出两个集群：mem0、RAG和全上下文方法达到77%至81%的精度，而Graphiti和cognee……（原文摘要在此处被截断）

    arXiv:2601.07978v5 Announce Type: replace  Abstract: Long-term memory (LTM) is fundamental to large language model (LLM)-based agents in the emerging Internet of Agents (IoA), where distributed multi-agent systems (DMAS) span cloud and edge networks. Existing evaluations are typically published by framework providers and focus on token usage and latency, rarely accounting for system-level cost or deployment in DMAS. These gaps are addressed with an independent reproducible testbed that evaluates accuracy, latency, CPU time, peak RAM, disk I/O and network usage in a simulated cloud-edge environment. Three venture capital-funded frameworks spanning vector, graph, and hybrid architectures, namely mem0, Graphiti, and cognee, are compared alongside retrieval-augmented generation (RAG) and full-context baselines on the LoCoMo benchmark under unconstrained and constrained network scenarios. Two clusters emerge: mem0, RAG, and full-context reach 77% to 81% accuracy, while Graphiti and cognee r
    
[^30]: 因果-反事实RAG：将因果-反事实推理融入检索增强生成

    Causal-Counterfactual RAG: The Integration of Causal-Counterfactual Reasoning into RAG

    [https://arxiv.org/abs/2509.14435](https://arxiv.org/abs/2509.14435)

    该论文提出因果-反事实RAG框架，通过将显式因果图与反事实推理融入检索增强生成过程，解决了传统RAG系统因文本分块破坏上下文完整性和过度依赖语义相似性检索而导致的回答浅显不准确的问题。

    

    大型语言模型（LLMs）通过整合大规模预训练知识，变革了自然语言处理（NLP）领域，支持了多样化的应用。然而，其静态知识限制了对信息进行动态推理的能力，尤其是在知识密集型领域。检索增强生成（RAG）通过将检索机制与生成模型相结合以提升上下文理解能力，从而应对这一挑战。传统RAG系统由于文本分块破坏了上下文完整性，并且过度依赖语义相似性进行检索，往往导致回答浅显且不够准确。我们提出了因果-反事实RAG（Causal-Counterfactual RAG），这是一个新颖的框架，它将表示因果关系的显式因果图融入检索过程，并引入基于因果结构的反事实推理。与传统方法不同，我们的框架不仅评估……

    arXiv:2509.14435v3 Announce Type: replace  Abstract: Large language models (LLMs) have transformed natural language processing (NLP), enabling diverse applications by integrating large-scale pre-trained knowledge. However, their static knowledge limits dynamic reasoning over external information, especially in knowledge-intensive domains. Retrieval-Augmented Generation (RAG) addresses this challenge by combining retrieval mechanisms with generative modeling to improve contextual understanding. Traditional RAG systems suffer from disrupted contextual integrity due to text chunking and over-reliance on semantic similarity for retrieval, often resulting in shallow and less accurate responses. We propose Causal-Counterfactual RAG, a novel framework that integrates explicit causal graphs representing cause-effect relationships into the retrieval process and incorporates counterfactual reasoning grounded on the causal structure. Unlike conventional methods, our framework evaluates not only d
    
[^31]: TC-RAG：图灵完备的RAG在医疗大语言模型系统上的案例研究

    TC-RAG:Turing-Complete RAG's Case study on Medical LLM Systems

    [https://arxiv.org/abs/2408.09199](https://arxiv.org/abs/2408.09199)

    提出TC-RAG框架，通过引入图灵完备系统管理状态变量，并利用具备自适应检索、推理和规划能力的记忆栈系统，实现检索过程的受控终止并减少错误知识累积，从而显著提升医疗大语言模型系统知识检索的效率与准确性。

    

    在提升领域特定大语言模型（LLMs）能力的过程中，检索增强生成（RAG）成为一种有前景的解决方案，能够缓解幻觉、知识过时以及在高度专业化查询中专业知识有限等问题。然而，现有的RAG方法存在不足，因为它们忽略了系统状态变量，而这些变量对于实现自适应控制、检索终止以及系统收敛至关重要。在本文中，我们通过严格的证明提出了TC-RAG，这是一个新颖的框架，通过引入图灵完备系统来管理状态变量，从而应对上述挑战，实现更高效、更准确的知识检索。通过利用具备自适应检索、推理和规划能力的记忆栈系统，TC-RAG不仅确保了检索过程的受控终止，还通过Push和Pop操作减轻了错误知识的累积。

    arXiv:2408.09199v2 Announce Type: replace  Abstract: In the pursuit of enhancing domain-specific Large Language Models (LLMs), Retrieval-Augmented Generation (RAG) emerges as a promising solution to mitigate issues such as hallucinations, outdated knowledge, and limited expertise in highly specialized queries. However, existing approaches to RAG fall short by neglecting system state variables, which are crucial for ensuring adaptive control, retrieval halting, and system convergence. In this paper, we introduce the TC-RAG through rigorous proof, a novel framework that addresses these challenges by incorporating a Turing Complete System to manage state variables, thereby enabling more efficient and accurate knowledge retrieval. By leveraging a memory stack system with adaptive retrieval, reasoning, and planning capabilities, TC-RAG not only ensures the controlled halting of retrieval processes but also mitigates the accumulation of erroneous knowledge via Push and Pop actions. In the ca
    
[^32]: 多任务深度推荐系统：综述

    Multi-Task Deep Recommender Systems: A Survey

    [https://arxiv.org/abs/2302.03525](https://arxiv.org/abs/2302.03525)

    本综述填补了推荐领域多任务学习系统性综述的空白，全面回顾了多任务深度推荐系统（MTDRS），并从任务关系（并行、级联、辅助与主任务）和方法论（参数共享、优化、训练机制）两个角度提出了系统的分类体系。

    

    多任务学习（MTL）旨在通过统一模型学习相关任务，利用任务间的共享知识实现各任务间的相互提升。由于需要同时兼顾性能与效率的多任务预测需求，MTL成为推荐系统领域的重要研究课题。尽管MTL已得到充分的研究与发展，但推荐领域仍缺乏对其的系统性综述。为填补这一空白，本综述对现有的多任务深度推荐系统（MTDRS）进行了全面回顾。具体而言，本文首先给出了MTDRS的问题定义，并将其与其他相关领域进行了比较；随后描述了MTDRS的发展历程，并从任务关系与方法论两个角度引入分类体系。具体来说，任务关系被划分为并行式、级联式以及辅助任务与主任务结合式，而方法论则被归类为参数共享、优化和训练机制等类别。

    arXiv:2302.03525v3 Announce Type: replace  Abstract: Multi-task learning (MTL) aims at learning related tasks in a unified model to achieve mutual improvement among tasks considering their shared knowledge. It is an important topic in recommendation due to the demand for multi-task prediction considering performance and efficiency. Although MTL has been well studied and developed, there is still a lack of systematic review in the recommendation community. To fill the gap, we provide a comprehensive review of existing multi-task deep recommender systems (MTDRS) in this survey. To be specific, the problem definition of MTDRS is first given, and it is compared with other related areas. Next, the development of MTDRS is depicted and the taxonomy is introduced from the task relation and methodology aspects. Specifically, the task relation is categorized into parallel, cascaded, and auxiliary with main, while the methodology is grouped into parameter sharing, optimization, and training mecha
    

