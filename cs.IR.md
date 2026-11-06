# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Decision Models as Retrieval Gates Against Injection and Pollution](https://arxiv.org/abs/2611.08325) | 开放权重决策模型可作为检索闸门，在不生成任何文本的情况下以极低成本（每段 76 毫秒）过滤检索内容中的隐藏提示注入指令与伪造污染页面，将攻击成功率从 27.9% 降至 5.7%，且对正常问答的响应率几乎没有影响。 |
| [^2] | [GINGER: Spicing Up Graph-Based Vector Search](https://arxiv.org/abs/2611.08306) | GINGER是谷歌开源的基于图的近似最近邻搜索C++库，通过嵌套TurboQuant（NTQ）两级量化与近邻图压缩技术，在节省最多2.6倍内存的同时实现了领先的查询吞吐量和索引构建速度。 |
| [^3] | [Domain-specific information retrieval](https://arxiv.org/abs/2611.08228) | 本文系统阐述了领域专用信息检索的任务、数据集与应用，分析了其在长文档、长查询、长会话及偏差公平性等方面的独特挑战，并介绍了针对这些挑战所开发的模型与方法。 |
| [^4] | [Probabilistic Relevance Decisions for Semantic Retrieval](https://arxiv.org/abs/2611.08014) | 在 BEIR SciFact 官方基准的全语料库稠密检索加重排序评测中，基于概率相关性决策的 Jev 评分取得了最高的 Recall@5 和 nDCG@10，优于仅稠密检索的 BGE-M3 和 OpenAI 评分，且增大候选深度带来的最终效果提升较小且统计上不确定。 |
| [^5] | [ArcadeDB in Python: An In-Process Multi-Model Database](https://arxiv.org/abs/2611.07971) | 本文推出 arcadedb-embedded Python 包，通过一次 pip 安装即可在 Python 进程内嵌入同时支持文档、属性图和向量搜索的多模型数据库 ArcadeDB，无需安装 Java 或在多个独立系统间同步数据。 |
| [^6] | [ZeroRAG: Adaptive Semantic Peer Routing for Decentralized Retrieval-Augmented Generation](https://arxiv.org/abs/2611.07928) | ZeroRAG提出了一种无需全局数据结构和集中式编排的去中心化RAG文档检索方案，通过在对等覆盖网络上进行自适应语义查询路由，使文档始终保留在原始所有者手中。 |
| [^7] | [Persistent Retrieval for Conversational Agents: A Study of Retrieval Efficiency](https://arxiv.org/abs/2611.07899) | 提出一种跨轮次维护检索历史并将检索证据直接返回主代理的持久化检索子代理设计，相比基于技能的检索可减少43.3%的token用量和45.0%的预估API成本。 |
| [^8] | [CORE: Coarse-to-Fine Search with Context Management for Deep Research](https://arxiv.org/abs/2611.07859) | CORE是一个无需训练的深度研究框架，通过将粗到细搜索与上下文管理相结合，解决了搜索需要长上下文与返回证据应简短之间的权衡，实现了最先进的性能。 |
| [^9] | [Tasteply: Externalizing Stable and Situational Preference as Manipulable Spatial Objects](https://arxiv.org/abs/2611.07851) | 该论文提出Tasteply系统，将用户的稳定偏好与情境偏好这一偏好结构外化为可操作的空间对象，使用户能够直观地观察并操纵推荐算法的偏好。 |
| [^10] | [MEMOIR: A Training-Free Memory Lifecycle for Long-Horizon LLM Agents](https://arxiv.org/abs/2611.07631) | MEMOIR 提出了一种免训练的记忆生命周期，通过压缩前的证据保全筛选门与查询后的回溯性归因动态修订记忆的重要性，解决了长程 LLM 智能体中记忆整合破坏关键证据与检索偏向陈旧记忆这两个相互耦合的失败模式。 |
| [^11] | [Beyond Independent Predictions: Learning the Browsing Journey for Sequence-Level Re-ranking](https://arxiv.org/abs/2611.07591) | 该论文针对传统重排序评估器将用户连续浏览旅程简化为独立逐物品预测的缺陷，提出联合行为建模（JBM）视角，通过建模物品的触达与曝光过程以及用户注意力的竞争机制，提升序列级重排序中评估器打分的准确性与推荐质量。 |
| [^12] | [What Gets Hacked in a Collaborative Reward for LLM Recommenders: The Part That Does Not Depend on the User](https://arxiv.org/abs/2611.07588) | 本文发现LLM推荐器强化学习所用协同过滤奖励中不依赖用户的物品均值分量是导致曝光坍缩的原因，并提出逐物品中心化与标准化两种奖励修正方法来缓解该问题。 |
| [^13] | [OptST: Optimizing Semantic Tokenization for LLM-Enhanced Recommendation](https://arxiv.org/abs/2611.07523) | 提出OptST方法，让预训练的语义分词器能够感知并适配下游推荐任务，突破传统两阶段流程中token语义划分固定、分辨率单一的限制。 |
| [^14] | [MEvince: Knowledge-Constrained LLM Reasoning for Technique Identification from Malware Reports](https://arxiv.org/abs/2611.07486) | MEvince提出了一种知识约束的LLM推理框架，通过构建连接报告观察结果与技术定义的共享行为表示，解决恶意软件报告中ATT&CK技术识别证据分散、技术界限细微的难题。 |
| [^15] | [BRIDGE: Bridging the Semantic Quantization Gap for Generative Recommendation](https://arxiv.org/abs/2611.07425) | 该论文揭示了生成式推荐中的一个被忽视的瓶颈——语义量化鸿沟（SQG），即解码器仅接收离散语义ID而丢失了量化过程中丰富的连续语义结构，并提出了BRIDGE方法从接地、邻域和难度三个方面弥合这一鸿沟。 |
| [^16] | [CacheGraft: Enhancing Retrieval-Augmented Generation via Latent-Space Document Enrichment](https://arxiv.org/abs/2611.07393) | CacheGraft基于缓存到缓存（C2C）融合框架，将语料库段落离线编码为潜空间KV缓存表示，在查询时直接注入小模型，无需大模型参与即可增强检索增强生成（RAG）的答案质量。 |
| [^17] | [Reducing Representational Interference in Retrieval Embeddings with Trainable Prefix Tokens](https://arxiv.org/abs/2611.07298) | 提出在嵌入模型输入前添加可训练的[DUMP]前缀标记，通过标准对比检索目标进行优化，以减少残留标记级信息对检索表示的干扰，且不改变模型架构和训练流程，在编码器与解码器嵌入模型上均持续优于微调基线。 |
| [^18] | [MapHub: Daily Relevance Heatmaps of New arXiv Preprints for a Research Community](https://arxiv.org/abs/2611.07174) | MapHub利用大语言模型每日评估新arXiv论文与研究社区中每位成员研究组合的相关性，并以热图形式呈现，其高分论文可覆盖评审者盲选前5%论文的约70%，显著优于文本匹配基线，且适用于任何研究领域。 |
| [^19] | [WISE: An Agentic Framework for Whole-page Multimodal Search Experience Attribution](https://arxiv.org/abs/2611.06777) | 该论文提出了系统质量诊断（SQD）这一新任务，并构建了WISE智能体框架，能够对工业搜索服务进行端到端的质量诊断，发现面向用户的问题并将其归因到底层算法组件。 |
| [^20] | [Generative Retrieval under Real-Time Supply Constraints in Local Commerce](https://arxiv.org/abs/2611.06323) | 该论文提出RTP-LLM流水线，通过将合格的残差量化语义标识符预先编译为压缩稀疏行前缀索引、与模型预填充并行执行CPU编译并在beam间共享，使生成式检索能在解码过程中直接满足本地商业的实时供给约束，避免对不可行候选浪费beam容量和推理计算。 |
| [^21] | [LIBRA: Programmatic Induction of Executable Tool Libraries](https://arxiv.org/abs/2611.05445) | LIBRA 通过让大语言模型提出工具间契约并在沙盒中进行运行时验证，将扁平的工具目录自动归纳为可执行的工具库，从而在多工具任务中检索出包含先决工具的完整工作流，无需依赖标注或任务特定训练，在多个基准上显著提升了检索性能。 |
| [^22] | [Merge and Branch: Efficient Tree-Structured Supervised Fine-Tuning of LLMs for Recommendation](https://arxiv.org/abs/2610.37973) | 提出 Merge and Branch 方法，通过将历史物品压缩为少量可学习 token，并以浅而宽的树状结构结合树形 FlashAttention 实现历史计算共享与多目标分支预测，大幅降低 LLM 生成式推荐监督微调的输入成本并提升学习效率。 |
| [^23] | [Revisiting Generative vs. Dense Retrieval: A Static Value Prior Reverses the Gap](https://arxiv.org/abs/2610.19065) | 生成式检索相对稠密检索的价值优势主要源于物品流行度这一静态先验，在稠密检索中加入一步静态价值重排即可逆转差距并保持更高召回率，且工业级实验表明 140 亿参数的 SID-LLM 仍不及小 47 倍的稠密编码器。 |
| [^24] | [NEAR: Neighborhood-Aware Residuals for Memory Retrieval](https://arxiv.org/abs/2610.13314) | NEAR提出了一种免训练的记忆检索框架，通过利用邻居记录间的共享信息构建投影残差来补充原始嵌入，有效缓解了因共享内容导致无关记录排名高于正确证据的问题，从而提升查询-记忆匹配的准确性。 |
| [^25] | [Beyond the Final Embedding: Retrieving Memories from Token-Level Evidence](https://arxiv.org/abs/2610.13277) | 该论文提出一种无需额外训练的稀疏词元匹配方法，利用嵌入模型已有的词元级上下文表示捕捉局部证据，并结合整条记忆相似度对记忆排序，在八个基准测试和多个模型家族上显著提升检索效果，还可通过可选的低秩查询适配器进一步增强。 |
| [^26] | [Where Does Retrieval-Based Open-Ended Evaluation Fail? Automatic Taxonomy Induction from Long-Form Medical Answer Factuality Verification](https://arxiv.org/abs/2609.30467) | 该论文针对基于检索的开放式医学事实性验证，自动归纳出两套错误分类体系，将失败分解为五个质量维度上的检索阶段错误与六个连续步骤中的验证器推理错误，并利用LLM-as-Judge流程实现大规模自动化错误标注与分类，无需黄金答案或黄金证据。 |
| [^27] | [Set-Valued Routing for Language-Model Agent Systems: A WildChat Benchmark and Cost-Aware Evaluation](https://arxiv.org/abs/2606.28925) | 该论文将语言模型智能体系统的路由从单一top-1选择扩展为集合值路由，构建了基于WildChat的3000条提示、12个能力定义智能体的新基准，并通过成本感知评估证明微调文本编码器（89.59 F1）显著优于线性SVM和零样本GPT-4o。 |
| [^28] | [Multimodal Graph RAG for Long-range Visually Rich Document Understanding](https://arxiv.org/abs/2606.28780) | 本文提出了一种基于多模态图结构的检索增强生成（RAG）方法，通过自动构建多模态知识图谱来总结文档的全局知识，从而解决了长视觉富文档需要整体理解时的视觉问答难题。 |
| [^29] | [HyperSU: Corpus-Driven Semantic-Unit Hypergraph for Retrieval-Augmented Generation](https://arxiv.org/abs/2606.28351) | HyperSU 提出了一种基于超图的检索增强生成框架，通过语料驱动的语义单元超边构建来避免 LLM 生成摘要带来的幻觉与高索引成本，并采用线索引导的双向检索来克服单跳扩展遗漏多跳证据、PageRank 扩散引发语义漂移和噪声推理链的问题。 |
| [^30] | [HGCRS: Hypergraph-Enhanced Generative Conversational Recommender System](https://arxiv.org/abs/2605.21987) | 提出HGCRS，通过SID—实体超图和类型感知超图编码器，将结构化目录关系融入生成式对话推荐系统，弥合了物品级关系知识与组合式语义ID之间的不匹配。 |
| [^31] | [EcoGEO: Trajectory-Aware Evidence Ecosystems for Web-Enabled LLM Search Agents](https://arxiv.org/abs/2605.12887) | 本文提出EcoGEO框架，将生成式引擎优化（GEO）从单网页层面扩展为面向网络增强型LLM搜索智能体的环境级影响力问题，并设计了轨迹感知的协同证据生态系统TRACE，通过控制智能体浏览轨迹中证据的组织、连接与呈现方式来影响其搜索结果。 |
| [^32] | [Empirical Study of Pop and Jazz Mix Ratios for Genre-Adaptive Chord Generation](https://arxiv.org/abs/2605.04998) | 本研究通过对流行预训练Music Transformer进行爵士微调并系统扫描混合的流行“排练”数据量，实证发现排练数据可将爵士top-1提升7至9个百分点，同时将流行风格的遗忘从2.11个百分点降至最多1.19个百分点，为小数据新风格的适应提供了混合比例依据。 |
| [^33] | [AgentSelect: Benchmark for Narrative Query-to-Agent Recommendation](https://arxiv.org/abs/2603.03761) | AgentSelect是一个将智能体选择重新建模为“叙述性查询到智能体推荐”任务的基准，聚合了超过11万条查询、10.7万个可部署智能体和25万余条交互记录，填补了端到端智能体配置推荐缺乏查询条件监督数据的研究空白。 |
| [^34] | [OneRanker: Unified Generation and Ranking with One Model in Industrial Advertising Recommendation](https://arxiv.org/abs/2603.02999) | 提出统一模型 OneRanker，通过价值感知的多任务解耦架构在工业广告推荐中实现生成与排序的架构级深度融合，解决了兴趣与商业价值错位、生成目标无关以及两阶段割裂等核心挑战。 |
| [^35] | [FloodSQL-Bench: A Retrieval-Augmented Benchmark for Geospatially-Grounded Text-to-SQL](https://arxiv.org/abs/2512.12084) | 该论文提出了FLOODSQL-BENCH，一个面向洪水管理领域的地理空间Text-to-SQL检索增强基准，通过键、空间和混合连接整合异构数据集，为高风险应用领域中的Text-to-SQL研究建立了实用测试平台。 |

# 详细

[^1]: 决策模型作为抵御注入与污染的检索闸门

    Decision Models as Retrieval Gates Against Injection and Pollution

    [https://arxiv.org/abs/2611.08325](https://arxiv.org/abs/2611.08325)

    开放权重决策模型可作为检索闸门，在不生成任何文本的情况下以极低成本（每段 76 毫秒）过滤检索内容中的隐藏提示注入指令与伪造污染页面，将攻击成功率从 27.9% 降至 5.7%，且对正常问答的响应率几乎没有影响。

    

    检索到的网页文本可能携带针对语言模型的恶意指令，或是旨在误导模型的合成页面。我们测试了已发布的开放权重决策模型——这类模型针对文本回答指定类型的问题而不生成任何 token——能否作为筛选检索段落的闸门，同时抵御这两类威胁。在一个包含 1,000 个问题、基于带有隐藏注入指令的论坛式网页的基准测试中，一个 Kev-4B 闸门读取每个段落的 HTML。攻击成功率（自动测量）从无防御时的 27.9% 降至采用已发布缓解措施（净化、规范化和引用提示）后的 13.6%，再加入闸门后进一步降至 5.7%。该闸门处理每个段落仅需 76 毫秒，并且在 186 个包含面向人类读者指令的良性页面中仅错误标记 4 个。对干净查询的响应率保持在 1 个百分点以内（85.6% 至 86.2%）。在一个包含 200 个查询的污染测试平台中（其中三分之二的候选页面为伪造页面），在选出前十条结果之前对候选池进行过滤可保持伪造页面……（摘要至此截断）

    arXiv:2611.08325v1 Announce Type: cross  Abstract: Retrieved web text can carry instructions aimed at the language model, or synthetic pages written to mislead. We test whether released open-weight decision models, which answer typed questions about a text without generating tokens, can screen retrieved passages against both threats. On a benchmark of 1,000 questions over forum-style web pages with hidden injected instructions, a Kev-4B gate reads each passage's HTML. Attack success, measured automatically, falls from 27.9% with no defence to 13.6% with the published mitigations (sanitisation, normalisation and citation prompting) and to 5.7% with the gate added. The gate takes 76 ms per passage and wrongly flags 4 of 186 benign pages with instructions for human readers. Responsiveness on clean queries stays within a point (85.6% to 86.2%). On a 200-query pollution testbed where two thirds of the candidate pages are fake, filtering the pool before the top ten are chosen keeps fake page
    
[^2]: GINGER：为基于图的向量搜索增添风味

    GINGER: Spicing Up Graph-Based Vector Search

    [https://arxiv.org/abs/2611.08306](https://arxiv.org/abs/2611.08306)

    GINGER是谷歌开源的基于图的近似最近邻搜索C++库，通过嵌套TurboQuant（NTQ）两级量化与近邻图压缩技术，在节省最多2.6倍内存的同时实现了领先的查询吞吐量和索引构建速度。

    

    我们提出了GINGER，一个用于基于图的近似最近邻搜索（ANNS）的最先进开源C++库。我们在谷歌开发了GINGER，旨在提升图搜索在大型高维嵌入数据集上的速度和内存效率。GINGER在查询吞吐量和索引构建时间方面超越了现有的ANNS库，同时其内存占用与最节省空间的竞争者相当。GINGER建立在近邻图构建和向量量化领域的最新进展之上。我们提出了嵌套TurboQuant（NTQ），这是一种两级量化器，可在同一套代码中同时支持低精度的图导航和高精度的重排序。这种表示方式以减少2.6倍内存的代价实现了接近浮点数的精度。我们还将近邻图的规模压缩了最多2.5倍，且不影响查询吞吐量。我们结合图重排序（原文在此截断）。

    arXiv:2611.08306v1 Announce Type: new  Abstract: We present \textsc{Ginger}, a state-of-the-art open-source C++ library for graph-based approximate nearest neighbor search (ANNS). We developed \textsc{Ginger} at Google to push the speed and memory efficiency of graph search on large and high-dimensional embedding datasets. \textsc{Ginger} surpasses existing ANNS libraries with regards to query throughput and indexing time while matching the memory footprint of the most space-efficient competitors.   \textsc{Ginger} builds upon recent advances in proximity graph construction (PiPNN) and vector quantization (TurboQuant). We introduce Nested TurboQuant (NTQ), a two-level quantizer which supports low precision graph navigation and high precision reranking in a single code. This representation achieves near floating point precision with $2.6\times$ less memory. We also compress the size of our proximity graphs by up to $2.5\times$ without impacting query throughput. We combine graph reorder
    
[^3]: 领域专用信息检索

    Domain-specific information retrieval

    [https://arxiv.org/abs/2611.08228](https://arxiv.org/abs/2611.08228)

    本文系统阐述了领域专用信息检索的任务、数据集与应用，分析了其在长文档、长查询、长会话及偏差公平性等方面的独特挑战，并介绍了针对这些挑战所开发的模型与方法。

    

    信息检索（IR）远不止临时性的网络搜索：很大一部分信息查找活动发生在特定领域中，通常涉及与工作相关的任务。例如，律师为法律案件寻找相关的先前判例，学术研究者为撰写论文而研读背景资料。在第2章中，我们概述了最常见的领域专用任务、数据集和应用。领域专用信息检索面临许多挑战，使其有别于简单的网络搜索任务：它们通常需要处理长文档、长查询以及带有特定用户需求的长会话。我们在第3章中讨论了这些问题。我们还探讨了与模型的偏差和公平性相关的问题。在第4章中，我们描述了为领域专用任务和数据开发的模型与方法，以应对特定挑战，包括与领域专用信息检索任务相关的用户层面问题、查询处理、用户模型以及用户交互。

    arXiv:2611.08228v1 Announce Type: new  Abstract: Information Retrieval (IR) is much more than ad hoc web search: a large portion of the information seeking activities takes place in specific domains, often addressing work-related tasks. Examples are lawyers finding relevant prior cases for a legal case and academic researchers studying the background material for a paper. In Chapter 2 we outline the most common domain-specific tasks, datasets and applications. Domain-specific IR has a number of challenges that make set it apart from simple web search tasks: they often deal with long documents, long queries and long sessions with specific user needs. We discuss these in Chapter 3. We also address issues related to bias and fairness of models. In Chapter 4, we describe the models and methods developed for domain-specific tasks and data, addressing the specific challenges, including user aspects related to domain-specific IR tasks, dealing with queries, user models, and user interactions.
    
[^4]: 面向语义检索的概率相关性决策

    Probabilistic Relevance Decisions for Semantic Retrieval

    [https://arxiv.org/abs/2611.08014](https://arxiv.org/abs/2611.08014)

    在 BEIR SciFact 官方基准的全语料库稠密检索加重排序评测中，基于概率相关性决策的 Jev 评分取得了最高的 Recall@5 和 nDCG@10，优于仅稠密检索的 BGE-M3 和 OpenAI 评分，且增大候选深度带来的最终效果提升较小且统计上不确定。

    

    稠密相似度提供可复用的语料库表示，而概率相关性决策则提供依赖查询的文档评分。我们在官方 BEIR SciFact 测试基准上，将仅使用稠密检索的 BGE-M3 与 TypeSafe Jev Noul 以及 OpenAI Decisions Predicate 评分进行比较。该研究对 300 个官方测试查询进行评估，使用相同的经过正例增强的 100 篇文档池，以及全语料库稠密检索后分别在候选深度 50 和 100 处进行重排序。与受控文档池不同，稠密检索前缀部分均未注入正例。在所评估的条件下，Jev 的 Recall@5 和 nDCG@10 观测值最高。在深度 50 时，全语料库 nDCG@10 为：BGE-M3 为 0.6413，OpenAI 为 0.7545，Jev 为 0.7805。将候选深度从 50 扩展到 100 提高了已知正例的候选召回率，但对最终效果的改变较小且不确定。四个配对自助法深度区间均包含零。独特条件下的 API 评分成本大约……（摘要原文在此处截断）

    arXiv:2611.08014v1 Announce Type: new  Abstract: Dense similarity supplies reusable corpus representations, while probabilistic relevance decisions offer query-dependent document scoring. We compare dense-only BGE-M3 with TypeSafe Jev Noul and OpenAI Decisions Predicate scores on the official BEIR SciFact test benchmark. The study evaluates 300 official test queries using identical positive-enriched 100-document pools and full-corpus dense retrieval followed by reranking at candidate depths of 50 and 100. Unlike the controlled pools, neither dense prefix receives positive injection. Jev has the highest observed Recall@5 and nDCG@10 under the evaluated conditions. At depth 50, full-corpus nDCG@10 is 0.6413 for BGE-M3, 0.7545 with OpenAI and 0.7805 with Jev. Extending depth from 50 to 100 increases known-positive candidate recall but produces small, uncertain final-effectiveness changes. All four paired-bootstrap depth intervals include zero. Unique-condition API scoring cost approximate
    
[^5]: Python 中的 ArcadeDB：进程内多模型数据库

    ArcadeDB in Python: An In-Process Multi-Model Database

    [https://arxiv.org/abs/2611.07971](https://arxiv.org/abs/2611.07971)

    本文推出 arcadedb-embedded Python 包，通过一次 pip 安装即可在 Python 进程内嵌入同时支持文档、属性图和向量搜索的多模型数据库 ArcadeDB，无需安装 Java 或在多个独立系统间同步数据。

    

    一些 Python 工作流需要对文档进行事务性读写、对图进行遍历，以及对向量嵌入进行相似度搜索。在实践中，这些功能通常被拆分到不同的系统中（文档存储、图引擎和向量索引），数据在它们之间复制，并由应用程序代码来保持一致。ArcadeDB 是一个用 Java 编写的开源多模型数据库引擎，支持上述所有三种访问模式，并且可以嵌入到宿主进程中。我们提出了 arcadedb-embedded，这是一个 Python 包，它通过 JPype 在 Python 进程内运行 ArcadeDB，并在 wheel 包中捆绑了 Java 运行时环境。只需一次 pip 安装，即可获得一个同时具备文档、属性图和向量搜索功能的数据库，且无需安装 Java。查询结果可以以 Python 列表、NumPy 数组、pandas DataFrame 或 Arrow 表的形式返回。我们演示了一个混合检索工作流，将向量搜索、SQL 过滤和图遍历串联起来。

    arXiv:2611.07971v1 Announce Type: cross  Abstract: Some Python workflows need transactional reads and writes over documents, traversals over a graph, and similarity search over vector embeddings. In practice these are split across separate systems (a document store, a graph engine, and a vector index), and the data is copied between them and kept consistent by application code. ArcadeDB is an open-source multi-model database engine, written in Java, that supports all three access patterns and can be embedded in a host process. We present arcadedb-embedded, a Python package that runs ArcadeDB inside the Python process via JPype, using a Java Runtime Environment bundled in the wheel. One pip install gives a database with documents, a property graph, and vector search, and needs no Java installation. Query results can be returned as Python lists, NumPy arrays, pandas DataFrames, or Arrow tables. We demonstrate a hybrid retrieval workflow that chains vector search, SQL filtering, and graph
    
[^6]: ZeroRAG：面向去中心化检索增强生成的自适应语义对等路由

    ZeroRAG: Adaptive Semantic Peer Routing for Decentralized Retrieval-Augmented Generation

    [https://arxiv.org/abs/2611.07928](https://arxiv.org/abs/2611.07928)

    ZeroRAG提出了一种无需全局数据结构和集中式编排的去中心化RAG文档检索方案，通过在对等覆盖网络上进行自适应语义查询路由，使文档始终保留在原始所有者手中。

    

    检索增强生成（RAG）是一种通过检索与查询相关的文档并将其注入大语言模型（LLM）提示中来提升LLM输出可靠性的标准方法。如今的RAG系统从集中式存储中收集、索引并提供这些文档。然而在实践中，文档的所有者是组织、设备和个人，由于隐私、版权或监管合规等原因，他们不能或不愿将文档的托管权移交给中央机构。现有的现代去中心化检索技术要么重新引入协调机构，要么依赖需要全网持续协调的全局数据结构（如分布式哈希表DHT）。我们提出了ZeroRAG，这是一个面向RAG文档检索的去中心化路由层，它在由对等节点构成的覆盖网络上执行语义查询路由，无需全局数据结构或集中式编排。文档仍保留在其所属的对等节点上。（摘要在此处被截断）

    arXiv:2611.07928v1 Announce Type: cross  Abstract: Retrieval-augmented generation (RAG) is a standard approach to enhance the reliability of large language model (LLM) outputs by retrieving documents relevant to a query and supplying them in the LLM prompt. Today's RAG systems collect, index, and serve these documents from centralized stores. In practice, however, documents are owned by organizations, devices, and individuals who cannot or will not surrender custody of them to a central party, e.g., because of privacy, copyright or regulatory compliance. Existing modern decentralized retrieval techniques either reintroduce a coordinating authority, or rely on a global data structure that requires continuous network-wide coordination (e.g., DHTs). We present ZeroRAG, a decentralized routing layer for document retrieval in RAG that performs semantic query routing across an overlay of peers with- out global data structures or centralized orchestration. Documents remain with the peers that
    
[^7]: 面向对话代理的持久化检索：检索效率研究

    Persistent Retrieval for Conversational Agents: A Study of Retrieval Efficiency

    [https://arxiv.org/abs/2611.07899](https://arxiv.org/abs/2611.07899)

    提出一种跨轮次维护检索历史并将检索证据直接返回主代理的持久化检索子代理设计，相比基于技能的检索可减少43.3%的token用量和45.0%的预估API成本。

    

    对话代理可以直接检索信息，也可以将检索任务委托给子代理。这一设计选择会影响检索历史的维护方式、每次模型调用所接收的上下文，以及在后续问题中需要重新检索哪些信息。我们比较了三种设计：主代理中基于技能（skill-based）的检索、生成调查结果回复的非持久化检索子代理，以及我们提出的持久化检索子代理——该子代理跨轮次维护检索历史，并将检索到的证据直接返回给主代理。我们的方法还允许主代理向同一个子代理请求修正。我们在独立提出的企业问题、共享对话中的问题以及依赖性后续问题上对这三种设计进行了评估。在五个多轮设置中，与基于技能的检索相比，我们的方法记录的代理-模型token用量减少43.3%，预估API成本降低45.0%，同时降低了平均

    arXiv:2611.07899v1 Announce Type: new  Abstract: Conversational agents can retrieve information directly or delegate retrieval to a subagent. This design choice affects how retrieval history is maintained, what context each model call receives, and what information must be re-retrieved in follow-up questions. We compare three designs: skill-based retrieval in the main agent, a non-persistent retrieval subagent that generates a findings response, and our persistent retrieval subagent, which maintains retrieval history across turns and returns retrieved evidence directly to the main agent. Our approach also allows the main agent to request corrections from the same subagent.   We evaluate the three designs on enterprise questions posed independently, within shared conversations, and as dependent follow-ups. Across five multi-turn settings, our approach uses 43.3% fewer recorded agent-model tokens and incurs 45.0% lower estimated API cost than skill-based retrieval, while reducing average
    
[^8]: CORE：面向深度研究的粗粒度到细粒度搜索与上下文管理

    CORE: Coarse-to-Fine Search with Context Management for Deep Research

    [https://arxiv.org/abs/2611.07859](https://arxiv.org/abs/2611.07859)

    CORE是一个无需训练的深度研究框架，通过将粗到细搜索与上下文管理相结合，解决了搜索需要长上下文与返回证据应简短之间的权衡，实现了最先进的性能。

    

    深度研究智能体通过反复搜索来回答复杂问题。它们依赖于一个执行工具调用并将证据返回到其上下文的框架（harness）。这带来了一种权衡：搜索在长段落上效果最佳，因为长段落能为可靠的语义匹配提供足够的上下文；但返回的证据应当简短，因为每条观察结果都会保留在智能体的上下文中，贯穿其后的整个轨迹。我们提出了CORE，一个无需训练的框架，它将粗粒度到细粒度的搜索与上下文管理相结合。CORE使用稠密检索器检索段落，然后复用后期交互（late-interaction）重排序器的词元级相似度，仅返回每个段落中最相关的片段。上下文管理使智能体能够围绕已检索的来源扩展上下文，并抑制重复片段。在与通义DeepResearch和OpenResearcher框架的受控封闭搜索对比中，CORE达到了最先进的性能……

    arXiv:2611.07859v1 Announce Type: cross  Abstract: Deep research agents answer complex questions through repeated search. They rely on a harness that runs their tool calls and returns evidence to their context. This creates a trade-off: Search works best over long passages, which give enough context for reliable semantic matching. But the returned evidence should be short, because every observation stays in the agent's context for the rest of the trajectory. We introduce CORE, a training-free harness that combines coarse-to-fine search with context management. CORE retrieves passages with a dense retriever. It then reuses the token-level similarities from a late-interaction reranker to return only the most relevant snippet from each passage. Context management lets the agent expand the context around sources it has already retrieved and suppresses repeated snippets. In controlled closed-search comparisons with the Tongyi DeepResearch and OpenResearcher harnesses, CORE achieves state-of
    
[^9]: Tasteply：将稳定偏好与情境偏好外化为可操作的空间对象

    Tasteply: Externalizing Stable and Situational Preference as Manipulable Spatial Objects

    [https://arxiv.org/abs/2611.07851](https://arxiv.org/abs/2611.07851)

    该论文提出Tasteply系统，将用户的稳定偏好与情境偏好这一偏好结构外化为可操作的空间对象，使用户能够直观地观察并操纵推荐算法的偏好。

    

    大多数推荐系统将用户偏好视为单一信号进行推断和优化，然而人类偏好实际上是稳定偏好与情境偏好的混合体，其中情境偏好会根据情境的变化而开启或关闭。通过Tasteply，我们将这种偏好结构外化为可操作的空间对象。Tasteply将每个情境偏好表示为一个吸引子，将稳定偏好表示为底层的全局场。用户可以通过在空间中移动和调整全局场的强度来调整其信息流，将原本潜藏在模型内部的偏好结构转化为他们可以直接观察和操作的对象，从而引导推荐算法。通过一项探索性研究（N=5），我们发现这种空间控制方式是有效的，同时也揭示了其作为操纵基于偏好的推荐手段的局限性与未来发展方向。

    arXiv:2611.07851v1 Announce Type: cross  Abstract: Most recommender systems treat user preferences as a single signal to infer and optimize, yet human preferences are a mix of a stable preference and situational preferences that turn on and off depending on context. Through Tasteply, we externalize this preference structure as manipulable spatial objects. Tasteply represents each situational preference as an attractor and the stable preference as an underlying global field. Users can adjust their feed by moving around the space and adjusting the strength of the global field, turning the preference structure that remains latent within the model into something they can see and act on directly to steer the recommendation algorithm. Through an exploratory study (N=5), we found that this spatial control was effective, along with its limits and future direction as a means of manipulating preference-based recommendations.
    
[^10]: MEMOIR：一种面向长程 LLM 智能体的免训练记忆生命周期

    MEMOIR: A Training-Free Memory Lifecycle for Long-Horizon LLM Agents

    [https://arxiv.org/abs/2611.07631](https://arxiv.org/abs/2611.07631)

    MEMOIR 提出了一种免训练的记忆生命周期，通过压缩前的证据保全筛选门与查询后的回溯性归因动态修订记忆的重要性，解决了长程 LLM 智能体中记忆整合破坏关键证据与检索偏向陈旧记忆这两个相互耦合的失败模式。

    

    长程 LLM 智能体需要这样的记忆系统：能够在多个会话之间保存有用信息，并在需要时将其检索出来。现有系统将这些视为相互分离的问题：记忆被预先写入时带有静态的重要性分数，且在使用后从不修订。这导致了两个相互耦合的失败模式：记忆整合可能会悄悄破坏未来查询所需的证据，而检索则持续偏向陈旧的记忆。我们提出 MEMOIR（通过合并认证的在线重要性修订实现记忆证据），这是一种免训练的记忆生命周期，它将“由判别器把关的整合”与“回溯性归因”统一起来。在压缩记忆之前，筛选门会检验与答案相关的证据是否得到保留。在每次查询之后，归因机制会估计哪些被检索到的记忆支撑了答案，并据此更新它们的检索优先级。此外，UCB 风格的探索奖励会额外鼓励检索曝光不足的记忆，减少……（原文摘要在此处被截断）

    arXiv:2611.07631v1 Announce Type: new  Abstract: Long-horizon LLM agents require memory systems that preserve useful information across sessions and retrieve it when needed. Existing systems treat these as separate problems: memories are written prospectively with static importance scores that are never revised after use. This creates two coupled failure modes: consolidation can silently destroy evidence that future queries require, while retrieval continues to favor stale memories. We propose MEMOIR (Memory Evidence via Merge-certified, Online Importance Revision), a training-free memory lifecycle that unifies judge-gated consolidation with retrospective attribution. Before compressing memory, screening gates test whether answer-relevant evidence is preserved. After each query, attribution estimates which retrieved memories supported the answer and updates their retrieval priority. A UCB-style exploration bonus additionally encourages retrieval of under-exposed memories, reducing the 
    
[^11]: 超越独立预测：学习用户浏览旅程以实现序列级重排序

    Beyond Independent Predictions: Learning the Browsing Journey for Sequence-Level Re-ranking

    [https://arxiv.org/abs/2611.07591](https://arxiv.org/abs/2611.07591)

    该论文针对传统重排序评估器将用户连续浏览旅程简化为独立逐物品预测的缺陷，提出联合行为建模（JBM）视角，通过建模物品的触达与曝光过程以及用户注意力的竞争机制，提升序列级重排序中评估器打分的准确性与推荐质量。

    

    重排序旨在向用户呈现一个能够最大化序列级效用的有序列表。工业界的部署广泛采用“生成器—评估器”框架，其中生成器提出候选序列，评估器对这些序列进行打分，以选出最佳序列进行展示。因此，评估器效用估计的保真度直接决定了推荐质量。然而，现有的评估器大多遵循独立行为建模范式，该范式将用户连续的浏览旅程简化为相互独立的单物品预测（例如点击率CTR）。这种过度简化忽视了两个物理现实：（i）只有当物品被触达并获得曝光后，点击才可能发生；（ii）各物品之间在竞争有限的用户注意力。因此，IBM范式会产生具有系统性偏差的评估结果和次优的序列选择。为了解决这一问题，我们提出了一种联合行为建模的视角……

    arXiv:2611.07591v1 Announce Type: new  Abstract: Re-ranking aims to present users with an ordered list that maximizes sequence-level utility. Industrial deployments widely adopt the Generator--Evaluator framework, where the generator proposes candidate sequences, and the evaluator scores them to select the best one for display. The fidelity of the evaluator's utility estimates thus directly governs recommendation quality. However, existing evaluators predominantly follow the \textbf{Independent Behavior Modeling (IBM)} paradigm, which simplifies the continuous user \emph{browsing journey} into independent per-item predictions (e.g., CTR). This oversimplification ignores two physical realities: (i) a click can occur only after the item has been reached and impressed, and (ii) items compete for finite user attention. As a result, IBM produces systematically biased evaluations and suboptimal sequence selection. To address this, we propose a \textbf{Joint Behavior Modeling (JBM)} perspecti
    
[^12]: LLM推荐器中协同奖励被“钻空子”的部分：不依赖于用户的那一部分

    What Gets Hacked in a Collaborative Reward for LLM Recommenders: The Part That Does Not Depend on the User

    [https://arxiv.org/abs/2611.07588](https://arxiv.org/abs/2611.07588)

    本文发现LLM推荐器强化学习所用协同过滤奖励中不依赖用户的物品均值分量是导致曝光坍缩的原因，并提出逐物品中心化与标准化两种奖励修正方法来缓解该问题。

    

    协同过滤（CF）分数可通过强化学习为训练大型语言模型（LLM）推荐器提供分级化的个性化奖励。然而，这些奖励可能导致曝光坍缩：尽管CF模型本身在不同用户之间能产生多样化的热门推荐，但用其奖励优化的LLM却集中于较少的物品。我们将每个分数分解为物品均值分量（该物品在所有用户历史中的平均CF分数）和残差分量（当前用户的分数相对平均值的偏差）。通过受控干预实验，我们确定物品均值分量是导致曝光坍缩的一个因素。据此，我们提出两种奖励修正方法：逐物品中心化（去除每个物品的估计平均分），以及逐物品标准化（使用估计标准差进一步对残差进行重新缩放）。在两个数据集和两种标识符方案上的实验表明，两种修正方法都能显著……（摘要截断）

    arXiv:2611.07588v1 Announce Type: new  Abstract: Collaborative-filtering (CF) scores provide graded, personalized rewards for training large language model (LLM) recommenders through reinforcement learning. However, these rewards can cause exposure collapse: even though the CF model itself produces diverse top recommendations across users, an LLM optimized with its rewards concentrates on fewer items. We decompose each score into an item-mean component, the item's average CF score across user histories, and a residual component, the current user's score deviation from the average. Through controlled interventions, we identify the item-mean component as a contributor to exposure collapse. Accordingly, we propose two reward corrections: per-item centering removes each item's estimated mean score, while per-item standardization further rescales the residual using its estimated standard deviation. Experiments on two datasets with two identifier schemes show that both corrections substantia
    
[^13]: OptST：面向大语言模型增强推荐的语义分词优化

    OptST: Optimizing Semantic Tokenization for LLM-Enhanced Recommendation

    [https://arxiv.org/abs/2611.07523](https://arxiv.org/abs/2611.07523)

    提出OptST方法，让预训练的语义分词器能够感知并适配下游推荐任务，突破传统两阶段流程中token语义划分固定、分辨率单一的限制。

    

    语义信息可以丰富推荐系统所使用的标识符和结构化字段。语义分词通过将大语言模型导出的用户画像转换为离散token，为这些信息提供了结构化表示。在常见的两阶段流程中，分词器先于推荐器进行训练，其token分配结果在推荐训练阶段被当作固定的输入特征。这样一来，下游推荐器虽然能够学习如何使用分配到的索引，却无法修改产生这些索引的语义划分。要让这种划分具备任务感知能力十分困难，因为最近邻码本分配是离散的，而且不受约束的自适应调整还可能破坏预训练阶段学到的几何结构。此外，固定的token深度使得每一次交互都只能获得相同的语义分辨率。为此，我们提出了OptST（优化语义分词），使预训练好的分词器能够响应下游推荐任务的需求……

    arXiv:2611.07523v1 Announce Type: new  Abstract: Semantic information can enrich the identifiers and structured fields used by recommender systems. Semantic tokenization provides a structured representation of this information by converting LLM-derived user profiles into discrete tokens. In the common two-stage pipeline, the tokenizer is trained before the recommender. Its token assignments are then treated as fixed input features during recommendation training. As a result, the downstream recommender can learn how to use the assigned indices, but it cannot revise the semantic partition that produced them. Making this partition task-aware is difficult because nearest-code assignment is discrete, and unconstrained adaptation may also distort the geometry learned during pretraining. Moreover, a fixed token depth exposes the same semantic resolution to every interaction. We thus propose OptST (Optimizing Semantic Tokenization), which makes a pretrained tokenizer responsive to downstream r
    
[^14]: MEvince：面向恶意软件报告技术识别的知识约束大语言模型推理

    MEvince: Knowledge-Constrained LLM Reasoning for Technique Identification from Malware Reports

    [https://arxiv.org/abs/2611.07486](https://arxiv.org/abs/2611.07486)

    MEvince提出了一种知识约束的LLM推理框架，通过构建连接报告观察结果与技术定义的共享行为表示，解决恶意软件报告中ATT&CK技术识别证据分散、技术界限细微的难题。

    

    将恶意软件分析报告中的行为情报映射到ATT&CK技术有助于威胁分析与防御，但可靠的技术识别仍然充满挑战。支持性证据往往散布在冗长且异构的报告之中，而相互关联的技术共享行为线索，仅在细微却起决定性作用的细节上有所差异。尽管大语言模型（LLM）能够利用外部知识来解读报告，但它们可能基于看似合理的关联而非充分的证据来指派技术。我们提出了MEvince，一个面向报告级恶意软件技术识别的知识约束LLM推理框架。其核心是连接报告观察结果与技术定义的共享行为表示。该表示通过对技术文档的智能体式挖掘构建而成，并支持两个在线阶段：1）原语引导的提取，利用行为原语对LLM提取的观察结果进行结构化处理……

    arXiv:2611.07486v1 Announce Type: cross  Abstract: Mapping behavioral intelligence in malware analysis reports to ATT&CK techniques supports threat analysis and defense, but reliable identification remains challenging. Supporting evidence is often scattered across long, heterogeneous reports, while closely related techniques share behavioral cues and differ only in subtle but decisive details. While large language models (LLMs) can use external knowledge to interpret reports, they may assign techniques based on plausible associations rather than sufficient evidence. We present MEvince, a knowledge-constrained LLM reasoning framework for report-level malware technique identification. Its core is a shared behavioral representation connecting report observations to technique definitions. This representation is built through agentic mining of technique documentation and supports two online stages: 1) primitive-guided extraction uses behavioral primitives to structure LLM-extracted observat
    
[^15]: BRIDGE：弥合生成式推荐中的语义量化鸿沟

    BRIDGE: Bridging the Semantic Quantization Gap for Generative Recommendation

    [https://arxiv.org/abs/2611.07425](https://arxiv.org/abs/2611.07425)

    该论文揭示了生成式推荐中的一个被忽视的瓶颈——语义量化鸿沟（SQG），即解码器仅接收离散语义ID而丢失了量化过程中丰富的连续语义结构，并提出了BRIDGE方法从接地、邻域和难度三个方面弥合这一鸿沟。

    

    生成式推荐通过自回归地生成语义标识符（SID）来检索物品，其中RQ-VAE将每个物品量化为分层的语义ID序列，解码器逐层生成该序列。我们识别出该范式中的一个被忽视的瓶颈：语义量化鸿沟（SQG）。量化过程从丰富的连续信号中推导出语义ID，包括码本质心、局部邻域几何结构和逐层熵。然而，解码器仅接收离散标识符，丢失了塑造这些标识符的语义结构。这种不匹配体现在三个方面：一是接地鸿沟，即当目标侧信息有限时，解码器缺乏语义锚点；二是邻域鸿沟，即词元级别的监督丢弃了物品间的相似性并加剧了曝光偏差；三是难度鸿沟，即尽管生成难度随深度递减，计算资源却被均匀分配。为了弥合SQG，我们提出了BRIDGE。

    arXiv:2611.07425v1 Announce Type: new  Abstract: Generative recommendation retrieves items by autoregressively generating Semantic IDs(SIDs), where an RQ-VAE quantizes each item into a hierarchical SID sequence that the decoder generates level by level. We identify an overlooked bottleneck in this paradigm: the Semantic Quantization Gap(SQG). Quantization derives SIDs from rich continuous signals, including codebook centroids, local neighborhood geometry, and level-wise entropy. However, the decoder receives only discrete identifiers and loses the semantic structure that shaped them. This mismatch manifests in three ways: a grounding gap, where the decoder lacks semantic anchors when target-side information is limited; a neighborhood gap, where token-level supervision discards inter-item similarity and exacerbates exposure bias; and a difficulty gap, where computation is allocated uniformly despite generation difficulty diminishing with depth.   To bridge the SQG, we propose BRIDGE. Co
    
[^16]: CacheGraft：通过潜空间文档增强来提升检索增强生成

    CacheGraft: Enhancing Retrieval-Augmented Generation via Latent-Space Document Enrichment

    [https://arxiv.org/abs/2611.07393](https://arxiv.org/abs/2611.07393)

    CacheGraft基于缓存到缓存（C2C）融合框架，将语料库段落离线编码为潜空间KV缓存表示，在查询时直接注入小模型，无需大模型参与即可增强检索增强生成（RAG）的答案质量。

    

    机器学习领域的最新研究表明，基于Transformer的语言模型代理不仅可以直接通过文本进行通信，还可以在潜空间中通过键值（KV）缓存直接进行通信。在本文中，我们提出这种缓存到缓存的翻译技术可以提供超越其实时代理通信这一最初应用场景的实用价值，并可用于检索增强生成（RAG）。基于缓存到缓存（C2C）融合框架，我们提出的系统CacheGraft将每个语料库段落离线编码为较大共享模型（实验中为4B参数）和较小接收模型（0.6B参数）之间的融合KV缓存表示，使得该缓存可以在查询时注入到接收模型中，而无需较大模型。在五个问答基准测试和两个独立本地运行的评估器上，CacheGraft在29个案例中（摘要在此处被截断）相比纯文本接收模型基线提升了答案质量。

    arXiv:2611.07393v1 Announce Type: cross  Abstract: Recent work in machine learning has demonstrated that transformer-based language model agents can communicate state not only through text, but also directly in latent space via key-value (KV) caches. In this paper, we argue that such cache-to-cache translation can provide utility beyond the original motivating application of real-time agent communication and can instead be utilized for retrieval-augmented generation (RAG). Building on the Cache-to-Cache (C2C) fusion framework, our proposed system CacheGraft encodes each corpus passage offline into a fused KV cache representation between a larger sharer model (4B parameters in our experiments) and a smaller receiver model (0.6B parameters) such that this cache can be injected into the receiver at query time with no need for the larger model. Across five QA benchmarks and two independent locally-run judges, CacheGraft improves answer quality over the text-only receiver baseline in 29 out
    
[^17]: 利用可训练前缀标记减少检索嵌入中的表示干扰

    Reducing Representational Interference in Retrieval Embeddings with Trainable Prefix Tokens

    [https://arxiv.org/abs/2611.07298](https://arxiv.org/abs/2611.07298)

    提出在嵌入模型输入前添加可训练的[DUMP]前缀标记，通过标准对比检索目标进行优化，以减少残留标记级信息对检索表示的干扰，且不改变模型架构和训练流程，在编码器与解码器嵌入模型上均持续优于微调基线。

    

    嵌入模型将可变长度的文本压缩为固定大小的向量表示，但这种压缩可能使语义内容与残留的标记级信息纠缠在一起，例如由注意力汇聚行为产生的伪影。我们提出了一种简单的方法，在嵌入模型训练过程中将可训练的[DUMP]标记前置到输入中。该方法保持模型架构、池化策略和训练损失不变，同时通过标准的对比检索目标来优化所添加的标记。通过引入在编码过程中与常规标记交互的可训练前缀位置，[DUMP]标记有助于减少残留或无信息量的标记级信号对最终检索表示的影响。在标准检索和长上下文基准测试上的实验表明，我们的方法在编码器和解码器嵌入模型上均持续优于微调基线。进一步分析表明，[DUMP]

    arXiv:2611.07298v1 Announce Type: new  Abstract: Embedding models compress variable-length text into fixed-size vector representations, but this compression can entangle semantic content with residual token-level information, such as artifacts arising from attention sink behavior. We propose a simple method that prepends trainable [DUMP] tokens to the input during embedding model training. The method keeps the model architecture, pooling strategy, and training loss unchanged, while optimizing the added tokens through the standard contrastive retrieval objective. By introducing trainable prefix positions that interact with regular tokens during encoding, [DUMP] tokens help reduce the influence of residual or uninformative token-level signals on the final retrieval representation. Experiments on standard retrieval and long-context benchmarks show that our method consistently improves over finetuned baselines across encoder and decoder embedding models. Further analysis shows that [DUMP] 
    
[^18]: MapHub：面向研究社区的每日新arXiv预印本相关性热图

    MapHub: Daily Relevance Heatmaps of New arXiv Preprints for a Research Community

    [https://arxiv.org/abs/2611.07174](https://arxiv.org/abs/2611.07174)

    MapHub利用大语言模型每日评估新arXiv论文与研究社区中每位成员研究组合的相关性，并以热图形式呈现，其高分论文可覆盖评审者盲选前5%论文的约70%，显著优于文本匹配基线，且适用于任何研究领域。

    

    MapHub（MIDAS arXiv预印本中心）会为研究社区中的每位参与者评估每天新提交的arXiv论文，由大型语言模型阅读每篇论文，并将其与为参与者构建的研究组合进行比对。它将0到100分的相关性评分以热图形式发布，直观展示每篇论文触及了社区中哪些成员的工作。在作者的盲评实验中，按MapHub评分排名前10%的论文包含了评审者自己选出的前5%论文（汇总统计）的约70%，而文本匹配基线方法仅能达到约三分之一。每天astro-ph类别的论文中约6%对每位参与者的评分达到30分，平均每三天有一篇论文达到60分。MapHub运行在标准的Claude订阅之上，并设计为可适用于任何研究领域。

    arXiv:2611.07174v1 Announce Type: cross  Abstract: MapHub, the MIDAS arXiv Preprint Hub, assesses each day's new arXiv submissions for every participant in a research community, with a large language model reading each paper against portfolios built for the participants. It publishes the relevance scores, from 0 to 100, as a heatmap that shows whose work each paper touches. In the authors' blind ratings, the top 10\% of papers by MapHub's scores held about 70\% of the raters' own top 5\%, pooled, against about a third for a text-matching baseline. About 6\% of a day's astro-ph papers reach 30 for each participant, and one every three days reaches 60. MapHub runs on a standard Claude subscription and is designed to be useful for any field: https://github.com/kailicao/maphub-public.
    
[^19]: WISE：面向整页多模态搜索体验归因的智能体框架

    WISE: An Agentic Framework for Whole-page Multimodal Search Experience Attribution

    [https://arxiv.org/abs/2611.06777](https://arxiv.org/abs/2611.06777)

    该论文提出了系统质量诊断（SQD）这一新任务，并构建了WISE智能体框架，能够对工业搜索服务进行端到端的质量诊断，发现面向用户的问题并将其归因到底层算法组件。

    

    算法驱动的服务，包括搜索、推荐和广告，依赖于多阶段模型流水线。虽然研究界优先关注模型优化，而部署后的监控则聚焦于运营可靠性，但运营上健康的系统仍可能产生低质量的结果，这暴露了现有评估与监控实践的空白。然而，针对这一任务的系统性问题定义和基准仍然缺失。为了弥补这一空白，我们为算法驱动的服务提出了系统质量诊断（SQD）这一新任务，它将质量评估从仅仅识别低质量结果扩展到在系统各组件中归因其成因。聚焦于工业搜索场景，我们提出了WISE——一个用于端到端质量诊断的智能体框架，它能够发现面向用户的问题，并将其根源追溯到底层的算法组件。为了提高诊断的可靠性，我们引入了ReTrace用于构建证据（摘要在此处截断）。

    arXiv:2611.06777v1 Announce Type: new  Abstract: Algorithm-driven services, including search, recommendation, and advertising, rely on multi-stage model pipelines. While research prioritizes model optimization, post-deployment monitoring focuses on operational reliability. Yet operationally healthy systems may still produce poor-quality results, exposing a gap in existing evaluation and monitoring practices. However, a systematic problem formulation and benchmark for this task remain lacking. To bridge this gap, we formulate System Quality Diagnosis (SQD) for algorithm-driven services, a new task that extends quality evaluation beyond identifying poor outcomes to attributing their causes across system components. Focusing on industrial search, we propose WISE, an agentic framework for end-to-end quality diagnosis that discovers user-facing issues and traces their origins to underlying algorithmic components. To improve diagnostic reliability, we introduce ReTrace for constructing evide
    
[^20]: 本地商业中实时供给约束下的生成式检索

    Generative Retrieval under Real-Time Supply Constraints in Local Commerce

    [https://arxiv.org/abs/2611.06323](https://arxiv.org/abs/2611.06323)

    该论文提出RTP-LLM流水线，通过将合格的残差量化语义标识符预先编译为压缩稀疏行前缀索引、与模型预填充并行执行CPU编译并在beam间共享，使生成式检索能在解码过程中直接满足本地商业的实时供给约束，避免对不可行候选浪费beam容量和推理计算。

    

    生成式检索为候选实体预测语义标识符。在本地零售和外卖配送场景中，候选还必须在特定时间对特定用户是可服务的。位置、天气和交通会不断重塑商家与商品供给，因此目录中有效的标识符对当前请求而言可能是不可行的。生成后过滤方式只有在beam容量和推理计算已经消耗之后才会移除这类候选。我们研究了在显式物化的、请求时刻供给快照上的检索问题，此时较短的输出序列几乎没有机会来摊销在线约束准备的开销。我们的RTP-LLM流水线将合格的残差量化标识符编译为压缩稀疏行前缀索引，将CPU编译与模型预填充阶段重叠执行，并在多个beam之间共享上传的索引。合法的扩展保留原始模型的对数概率，约束状态则跟随被选中的beam父节点。我们……

    arXiv:2611.06323v1 Announce Type: new  Abstract: Generative retrieval predicts semantic identifiers for candidate entities. In local retail and food delivery, candidates must also be serviceable for a particular user at a particular time. Location, weather, and traffic reshape merchant and product supply, so a catalog-valid identifier may be infeasible for the current request. Post-generation filtering removes such candidates only after beam capacity and inference computation have been spent. We study retrieval over an explicitly materialized, request-time supply snapshot, where short output sequences leave little opportunity to amortize online constraint preparation. Our RTP-LLM pipeline compiles eligible residual-quantization identifiers into a compressed sparse row prefix index, overlaps CPU compilation with model prefill, and shares the uploaded index across beams. Legal expansions retain original-model log probabilities, and constraint states follow selected beam parents. We forma
    
[^21]: LIBRA：可执行工具库的程序化归纳

    LIBRA: Programmatic Induction of Executable Tool Libraries

    [https://arxiv.org/abs/2611.05445](https://arxiv.org/abs/2611.05445)

    LIBRA 通过让大语言模型提出工具间契约并在沙盒中进行运行时验证，将扁平的工具目录自动归纳为可执行的工具库，从而在多工具任务中检索出包含先决工具的完整工作流，无需依赖标注或任务特定训练，在多个基准上显著提升了检索性能。

    

    大型语言模型智能体越来越多地从庞大的工具目录中检索工具，然而大多数检索器仅根据请求相似度对每个工具进行独立排序。对于多工具任务，这种方式只能召回与目标匹配的工具，而遗漏了工作流程所需的先决工具。我们提出 LIBRA——一个通过运行时组装进行工具库归纳的框架，它将扁平的工具目录转化为可执行的工具库。大语言模型为每个工具提出契约，包括候选调用、可复用输出与状态变更以及成功条件；LIBRA 随后在沙盒环境中测试这些组合，并保留在运行时得到验证支持的工具间关系。该工具库在用户查询之前构建完成，无需依赖关系标注、工具使用轨迹或任务特定训练，同时支持一次性检索和智能体式检索。在 AppWorld、BFCL 和 SciToolEval 三个基准上，相比相同的稠密检索器，LIBRA 将一次性检索的 Recall 和 NDCG 平均提升了 7.9 和 3.7 个百分点。

    arXiv:2611.05445v1 Announce Type: cross  Abstract: Large language model agents increasingly retrieve tools from large catalogs, yet most retrievers rank each tool independently by request similarity. For multi tool tasks, this recovers the goal matching tool while omitting prerequisite tools required by the workflow.We introduce LIBRA, a framework for Library Induction by Runtime Assembly that turns a flat tool catalog into an executable tool library. An LLM proposes per tool contracts of candidate invocations, reusable outputs and state changes, and success conditions; LIBRA then tests these compositions in a sandbox and retains the relations supported at runtime. Constructed before user queries, the library requires no dependency annotations, tool use trajectories, or task specific training, and supports both one-shot and agentic retrieval. On AppWorld, BFCL, and SciToolEval, LIBRA improves one-shot Recall and NDCG over the same dense retriever by 7.9 and 3.7 percentage points on ave
    
[^22]: 合并与分支：面向推荐的高效树状结构大语言模型监督微调

    Merge and Branch: Efficient Tree-Structured Supervised Fine-Tuning of LLMs for Recommendation

    [https://arxiv.org/abs/2610.37973](https://arxiv.org/abs/2610.37973)

    提出 Merge and Branch 方法，通过将历史物品压缩为少量可学习 token，并以浅而宽的树状结构结合树形 FlashAttention 实现历史计算共享与多目标分支预测，大幅降低 LLM 生成式推荐监督微调的输入成本并提升学习效率。

    

    基于大语言模型（LLM）的生成式推荐需要读取冗长、文本丰富的交互历史，但仅生成简短的语义标识符（SID）。这种不对称性使得监督微调效率低下：编码历史的代价高昂，而每段历史通常只为一个目标提供监督。我们提出 Merge and Branch（M&B）方法，以降低输入成本并从每段已编码的历史中学习更多信息。Merge 将每个历史物品的 SID 和文本压缩为少量可学习的 token；Branch 利用压缩后的历史来预测多个目标 SID。为了在共享历史计算的同时保持这些目标相互独立，M&B 将每个样本组织成一棵浅而宽的树。我们开发了树形 FlashAttention 来高效执行这种布局：共享历史只需处理一次，而梯度从每个分支反向传播，实现端到端训练。树形结构的键/值缓存进一步将历史共享扩展到受限解码阶段。在公开基准上的实验表明……

    arXiv:2610.37973v1 Announce Type: cross  Abstract: LLM-based generative recommendation reads long, text-rich interaction histories but generates only short semantic identifiers (SIDs). This asymmetry makes supervised fine-tuning inefficient: encoding the history is expensive, yet each history usually provides supervision for only one target. We present Merge and Branch (M&B) to reduce input cost and learn more from each encoded history. Merge compresses each historical item's SID and text into a small number of learned tokens. Branch uses the compressed history to predict multiple target SIDs. To share history computation while keeping these targets independent, M&B organizes each example as a shallow, wide tree. We develop tree FlashAttention to execute this layout efficiently, processing the shared history once and propagating gradients from every branch for end-to-end training. A tree-structured key/value cache extends history sharing to constrained decoding. Experiments on public b
    
[^23]: 重新审视生成式检索与稠密检索：一个静态价值先验即可逆转差距

    Revisiting Generative vs. Dense Retrieval: A Static Value Prior Reverses the Gap

    [https://arxiv.org/abs/2610.19065](https://arxiv.org/abs/2610.19065)

    生成式检索相对稠密检索的价值优势主要源于物品流行度这一静态先验，在稠密检索中加入一步静态价值重排即可逆转差距并保持更高召回率，且工业级实验表明 140 亿参数的 SID-LLM 仍不及小 47 倍的稠密编码器。

    

    随着语言模型和语义物品标识符（SID）的不断改进，生成式检索（GR）已成为序列推荐中强有力的候选生成方法。然而，在在线零售候选生成场景中，GR 并非稠密检索（DR）的全面更优替代方案。我们在两个 Amazon Toys 数据集上用不同指标检验了二者的差异：在两种方法共同覆盖的用户中，GR 在最佳相关物品价值上领先，而 DR 在精确目标召回率以及排序列表整体的相关价值上领先。通过深入分析，我们发现流行度对 GR 的价值领先贡献最大。在 DR 上仅通过一步重排加入相同的静态物品价值，即可逆转这一领先，同时保持召回率高于 GR。最后，在一个使用 LLM 判定相关性的工业级内部电商数据集上，我们证明了 140 亿参数的 SID-LLM 仍不如比它小 47 倍的稠密编码器。

    arXiv:2610.19065v1 Announce Type: cross  Abstract: Generative retrieval (GR) has emerged as a strong candidate generator for sequential recommendation as language models and semantic-item identifiers have improved. However, GR is not a uniformly stronger replacement of dense retrieval (DR) in online retail candidate generation. We examine their differences on two Amazon Toys datasets under different metrics. GR leads on best relevant item value among users both methods cover, while DR leads on exact-target recall and relevant value across the ranked list. With deep analysis, we find that popularity contributes the most to GR's measured value lead. Adding the same static item value to DR in one reranking step reverses this lead and keeps recall above GR. Finally, on an industrial-scale internal e-commerce dataset with LLM-judged relevance, we show that a 14B SID-LLM still underperforms a $47\times$ smaller dense encoder.
    
[^24]: NEAR：面向记忆检索的邻域感知残差方法

    NEAR: Neighborhood-Aware Residuals for Memory Retrieval

    [https://arxiv.org/abs/2610.13314](https://arxiv.org/abs/2610.13314)

    NEAR提出了一种免训练的记忆检索框架，通过利用邻居记录间的共享信息构建投影残差来补充原始嵌入，有效缓解了因共享内容导致无关记录排名高于正确证据的问题，从而提升查询-记忆匹配的准确性。

    

    记忆检索旨在从存储的记录中识别证据以回答查询。具有不同事实细节的存储记录可能共享主题、实体或事件。这些共享内容会使得不包含所需证据的记录更有可能获得较高的相似度分数，从而排在正确证据之前。在本文中，我们利用这些记录之间共享的信息来构建用于查询-记忆匹配的局部参考。我们提出了邻域感知残差框架，这是一个免训练的框架，利用这些参考来构建残差表示，以补充原始嵌入。具体而言，每个记忆从其邻居记录中导出一个参考方向，每个查询从预计算的表中获取其参考。利用这些参考，NEAR为查询和记忆构建投影残差。这些残差与原始嵌入相结合，用于单次近似最近邻搜索。

    arXiv:2610.13314v1 Announce Type: new  Abstract: Memory retrieval aims to identify evidence from stored records to answer a query. Stored records with different factual details may share topics, entities, or events. Such shared contents would make records without the required evidence more likely to receive high similarity scores and outrank the correct evidence. In this paper, we use the information shared by these records to construct local references for query-memory matching. We propose NEighborhood-Aware Residuals (NEAR), a training-free framework that uses these references to construct residual representations that complement the original embeddings. Specifically, each memory derives a reference direction from its neighbors, and each query obtains its reference from a precomputed table. Using these references, NEAR constructs projection residuals for both queries and memories. These residuals are combined with the original embeddings for a single approximate nearest-neighbor sear
    
[^25]: 超越最终嵌入：从词元级证据中检索记忆

    Beyond the Final Embedding: Retrieving Memories from Token-Level Evidence

    [https://arxiv.org/abs/2610.13277](https://arxiv.org/abs/2610.13277)

    该论文提出一种无需额外训练的稀疏词元匹配方法，利用嵌入模型已有的词元级上下文表示捕捉局部证据，并结合整条记忆相似度对记忆排序，在八个基准测试和多个模型家族上显著提升检索效果，还可通过可选的低秩查询适配器进一步增强。

    

    记忆增强型助手从存储的对话和文档中检索证据以回答查询。大多数基于嵌入的检索器比较查询和记忆的单向量表示，即使相关性往往只取决于内容的一小部分。这种局部证据可以利用同一嵌入模型已经计算出的上下文词元表示与查询进行匹配。我们的方法使用稀疏词元匹配来改进检索，且无需额外训练。每个查询词元在候选记忆中检索一小撮匹配的词元，匹配得分针对每条记忆进行聚合，并与整条记忆的相似度相结合来对记忆进行排序。可选的低秩查询适配器可以在监督下进行训练，同时保持编码器和已存储的记忆表示固定。我们的方法无需额外训练即可在八个基准测试和多个模型家族上提升检索性能。查询适配器可进一步……（原文在此处截断）

    arXiv:2610.13277v1 Announce Type: cross  Abstract: Memory-augmented assistants retrieve evidence from stored conversations and documents to answer queries. Most embedding-based retrievers compare single-vector representations of queries and memories, even when relevance depends on a small part of the content. Such local evidence can be matched to queries using the contextual token representations already computed by the same embedding models. Our method uses sparse token matching to improve retrieval without additional training. Each query token retrieves a small set of matching tokens across candidate memories. The matching scores are aggregated for each memory and combined with whole-memory similarity to rank memories. An optional low-rank query adapter can be trained with supervision while keeping the encoder and stored memory representations fixed. Our method improves retrieval across eight benchmarks and multiple model families without additional training. Query adaptation further
    
[^26]: 基于检索的开放式评估在何处失效？从长篇医学答案事实性验证中自动归纳错误分类体系

    Where Does Retrieval-Based Open-Ended Evaluation Fail? Automatic Taxonomy Induction from Long-Form Medical Answer Factuality Verification

    [https://arxiv.org/abs/2609.30467](https://arxiv.org/abs/2609.30467)

    该论文针对基于检索的开放式医学事实性验证，自动归纳出两套错误分类体系，将失败分解为五个质量维度上的检索阶段错误与六个连续步骤中的验证器推理错误，并利用LLM-as-Judge流程实现大规模自动化错误标注与分类，无需黄金答案或黄金证据。

    

    检索式事实性评估——即将大语言模型生成的论断与权威医学语料库中的证据进行核对验证——已成为高风险临床环境中可扩展幻觉检测的主流范式。尽管可靠且透明的医学事实验证需求迫切，大多数系统仍采用F1等聚合指标衡量性能，这类指标掩盖了失败发生在何处以及为何发生。现有的RAG诊断方法需要黄金答案或人工标注的黄金证据，而这两者在该场景中均不存在。我们基于对开放式MedExpert数据集以及3个封闭式数据集的案例研究，引入了两套全面的分类体系：将失败分解为沿五个质量维度划分的检索阶段错误，以及分为六个连续步骤的验证器推理错误。我们改造了一种基于LLM-as-Judge的自动模式归纳流程，用以大规模标注证据质量并对验证器推理错误进行分类……

    arXiv:2609.30467v1 Announce Type: new  Abstract: Retrieval-based factuality evaluation, where LLM-generated claims are verified against evidence from authoritative medical corpora, has become the dominant paradigm for scalable hallucination detection in high-stakes clinical settings. Despite the urgency of reliable and transparent medical fact verification, most systems measure performance with aggregate metrics like F1, which obscure where and why failures occur. Existing RAG diagnostics require gold answers or annotated gold evidence, neither of which exists in this regime. We introduce two comprehensive taxonomies, grounded in a case study on the open-ended MedExpert dataset and 3 closed-ended datasets, decomposing failures into retrieval-stage errors along five quality dimensions, and verifier-reasoning errors into six consecutive steps. We adapt an automatic pattern induction pipeline using LLM-as-Judge to label evidence quality and classify verifier reasoning errors at scale, and
    
[^27]: 面向语言模型智能体系统的集合值路由：一个WildChat基准与成本感知评估

    Set-Valued Routing for Language-Model Agent Systems: A WildChat Benchmark and Cost-Aware Evaluation

    [https://arxiv.org/abs/2606.28925](https://arxiv.org/abs/2606.28925)

    该论文将语言模型智能体系统的路由从单一top-1选择扩展为集合值路由，构建了基于WildChat的3000条提示、12个能力定义智能体的新基准，并通过成本感知评估证明微调文本编码器（89.59 F1）显著优于线性SVM和零样本GPT-4o。

    

    语言模型智能体系统必须在规划或工具执行之前，将自然语言请求映射到一个或多个专用智能体。这一决策通常被处理为top-1选择，然而组合式请求可能需要多种能力，而不必要的分发则会增加成本。我们将固定目录下的路由问题表述为集合值语言理解任务，并构建了一个包含3,000条WildChat提示、覆盖12个按能力定义的智能体的基准测试。可复现的模式引导参考集合按智能体频率和路由基数进行了平衡。我们比较了画像检索、多标签分类器、微调文本编码器以及基于目录条件化的零样本GPT-4o。评估综合考察了集合质量、延迟、能力覆盖率和序数成本效用。微调编码器达到了89.59 F1，相比之下线性SVM为71.79，GPT-4o仅为41.51，并且在更难的语义组划分下仍保持88.02 F1。我们进一步引入了加权智能体路由……（摘要原文在此处截断）

    arXiv:2606.28925v3 Announce Type: replace-cross  Abstract: Language-model agent systems must map natural-language requests to one or more specialized agents before planning or tool execution. This decision is often treated as top-1 selection, although compositional requests can require several capabilities and unnecessary dispatch raises cost. We formulate fixed-catalog routing as set-valued language understanding and construct a benchmark of 3,000 WildChat prompts over 12 capability-defined agents. Reproducible schema-guided reference sets are balanced by agent frequency and route cardinality. We compare profile retrieval, multilabel classifiers, a fine-tuned text encoder, and catalog-conditioned zero-shot GPT-4o. Evaluation combines set quality, latency, capability coverage, and ordinal-cost utility. The encoder attains 89.59 F1, compared with 71.79 for linear SVM and 41.51 for GPT-4o, and retains 88.02 F1 under a harder semantic-group split. We further introduce Weighted Agent Routi
    
[^28]: 面向长程视觉富文档理解的多模态图RAG

    Multimodal Graph RAG for Long-range Visually Rich Document Understanding

    [https://arxiv.org/abs/2606.28780](https://arxiv.org/abs/2606.28780)

    本文提出了一种基于多模态图结构的检索增强生成（RAG）方法，通过自动构建多模态知识图谱来总结文档的全局知识，从而解决了长视觉富文档需要整体理解时的视觉问答难题。

    

    多模态大语言模型（MLLMs）被广泛应用于视觉文档理解。然而，由于上下文窗口的限制，理解长文档仍然是一个难题。尽管最近的多模态检索增强生成（MMRAG）可以通过检索相关页面来应对这一挑战，但它仍然难以处理需要整体理解文档的视觉问答（VQA）任务。为了解决这一问题，总结文档全局知识的知识图谱（KG）可以提供一种有效的解决方案。然而，大多数现有的基于大语言模型（LLM）的KG构建方法仅处理语言模态，针对视觉富文档的多模态知识图谱（MMKGs）自动构建在很大程度上尚未被探索。在本文中，我们提出了一种基于多模态图的RAG方法来解决这一问题。现有的基于LLM的KG方法依赖于间接证据（如全面性、多样性等）来评估问答性能……

    arXiv:2606.28780v2 Announce Type: replace  Abstract: Multimodal large language models (MLLMs) are widely applied to visual document understanding. However, comprehending long documents remains an issue by the limited context window. Though recent multimodal retrieval-augmented generation (MMRAG) can address this challenge by retrieving relevant pages. It still struggles with the visual question answering (VQA) requiring holistic comprehension of a document. To cope with this, knowledge graph (KG) that summarizes global knowledge of a document can provide an effective solution. However, most existing LLM-based KG construction methods handle only the language modality, leaving the automatic creation of multimodal KGs (MMKGs) for visually rich documents largely unexplored. In this paper, we introduce a multimodal graph-based RAG approach to tackle this problem. Existing LLM-based KG methods evaluate the QA performance relying on indirect evidence such as comprehensiveness, diversity, empo
    
[^29]: HyperSU：面向检索增强生成的语料驱动语义单元超图

    HyperSU: Corpus-Driven Semantic-Unit Hypergraph for Retrieval-Augmented Generation

    [https://arxiv.org/abs/2606.28351](https://arxiv.org/abs/2606.28351)

    HyperSU 提出了一种基于超图的检索增强生成框架，通过语料驱动的语义单元超边构建来避免 LLM 生成摘要带来的幻觉与高索引成本，并采用线索引导的双向检索来克服单跳扩展遗漏多跳证据、PageRank 扩散引发语义漂移和噪声推理链的问题。

    

    摘要：近年来，基于超图的检索增强生成方法利用超边同时连接多个实体，相比成对图结构能够更高效地组织多实体证据。然而，现有 HyperRAG 方法通常依赖大语言模型（LLM）生成的摘要来构建超边，这可能引入幻觉，同时带来高昂的索引成本。此外，在检索阶段，现有方法通常依赖单跳邻居扩展或 PageRank 扩散：前者可能遗漏有用的多跳证据，后者则可能在过度的枢纽节点上发生不受控的传播，导致语义漂移和嘈杂的推理链。为应对这些挑战，我们提出了 HyperSU，一种新型的基于超图的 RAG 框架，其特色在于语义单元超边和线索引导的双向检索。在构建过程中，HyperSU 将超边构建形式化为一个实体……（原文摘要到此截断）

    arXiv:2606.28351v3 Announce Type: replace  Abstract: Recent Hypergraph-based retrieval-augmented generation (HyperRAG) methods use hyper edges to connect multiple entities simultaneously, enabling more efficient multi-entity evidence organization than pairwise graph structures. However, existing HyperRAG methods often rely on LLM-generated summaries to construct hyperedges, which can introduce hallucinations while also incurring high indexing costs. In addition, during retrieval, existing methods typically rely on either one-hop neighbor expansion or PageRank diffusion. The former may miss useful multi-hop evidence, while the latter can suffer from uncontrolled propagation over excessive hub nodes, leading to semantic drift and noisy reasoning chains. To address these challenges, we propose HyperSU, a novel hypergraph-based RAG framework featuring semantic-unit hyperedges and clue guided bidirectional retrieval. During construction, HyperSU formulates hyperedge construction as an entit
    
[^30]: HGCRS：超图增强的生成式对话推荐系统

    HGCRS: Hypergraph-Enhanced Generative Conversational Recommender System

    [https://arxiv.org/abs/2605.21987](https://arxiv.org/abs/2605.21987)

    提出HGCRS，通过SID—实体超图和类型感知超图编码器，将结构化目录关系融入生成式对话推荐系统，弥合了物品级关系知识与组合式语义ID之间的不匹配。

    

    近期的对话推荐系统已开始在自回归模型中统一推荐与回复生成，语义ID（SID）为生成式推荐提供了一种自然的表示方式。然而，关系知识通常围绕物品级表示进行组织，这与组合式的语义ID无法自然对齐。这种不匹配使得统一的对话推荐系统难以有效利用结构化的目录关系。我们提出了HGCRS，一个通过SID—实体超图来弥合这一差距的超图增强生成式对话推荐系统。每个物品形成一条超边，连接其SID编码与属性实体，类型感知的超图编码器将关系信息融入生成式物品选择之中。HGCRS还进一步引入了模式—物品—回复协议，用于联合建模包含推荐意图的轮次和仅回复的轮次。在ReDial和INSPIRED数据集上的实验表明（摘要内容不完整，后续内容缺失）……

    arXiv:2605.21987v2 Announce Type: replace  Abstract: Recent conversational recommender systems have begun to unify recommendation and response generation within autoregressive models, with semantic IDs providing a natural representation for generative recommendation. However, relational knowledge is typically organized around item-level representations, which do not naturally align with the compositional SIDs. This mismatch makes it difficult for unified CRSs to effectively exploit structured catalog relations. We propose HGCRS, a hypergraph-enhanced generative conversational recommender system that bridges this gap with an SID--entity hypergraph. Each item forms a hyperedge connecting its SID codes and attribute entities, and a type-aware hypergraph encoder incorporates relational information into generative item selection. HGCRS further introduces a mode--item--response protocol for jointly modeling recommendation-bearing and response-only turns. Experiments on ReDial and INSPIRED de
    
[^31]: EcoGEO：面向网络增强型大语言模型搜索代理的轨迹感知证据生态系统

    EcoGEO: Trajectory-Aware Evidence Ecosystems for Web-Enabled LLM Search Agents

    [https://arxiv.org/abs/2605.12887](https://arxiv.org/abs/2605.12887)

    本文提出EcoGEO框架，将生成式引擎优化（GEO）从单网页层面扩展为面向网络增强型LLM搜索智能体的环境级影响力问题，并设计了轨迹感知的协同证据生态系统TRACE，通过控制智能体浏览轨迹中证据的组织、连接与呈现方式来影响其搜索结果。

    

    arXiv:2605.12887v3 公告类型：replace-cross。摘要：具备网络访问能力的大语言模型智能体正在改变在线信息影响搜索结果的方式。现有的生成式引擎优化（GEO）研究主要聚焦于单个网页。然而，智能体化的网络搜索并非单文档场景：智能体可能会发出查询、抓取页面、跟踪链接、重新表述搜索，并在多个浏览步骤中综合证据。因此，影响力不仅取决于页面内容，还取决于页面如何沿着智能体的浏览轨迹被组织、连接和呈现。我们通过生态系统生成式引擎优化（EcoGEO）来研究这一转变，该方法将GEO视为面向网络增强型大语言模型智能体的环境级影响力问题。为了具体实现这一视角，我们提出了TRACE——一个轨迹感知的协同证据生态系统。给定一个推荐查询和一个虚构的目标产品，我们的方法构建了一个受控的证据环境，该环境协调智能体辅助的……（原文摘要在此处截断）

    arXiv:2605.12887v3 Announce Type: replace-cross  Abstract: Web-enabled LLM agents are changing how online information influences search outcomes. Existing Generative Engine Optimization (GEO) studies mainly focus on individual webpages. However, agentic web search is not a single-document setting: an agent may issue queries, crawl pages, follow links, reformulate searches, and synthesize evidence across multiple browsing steps. Influence therefore depends not only on page content, but also on how pages are organized, connected, and encountered along the agent's browsing trajectory. We study this shift through Ecosystem Generative Engine Optimization (EcoGEO), which treats GEO as an environment-level influence problem for web-enabled LLM agents. To instantiate this perspective, we propose TRACE, a Trajectory-Aware Coordinated Evidence Ecosystem. Given a recommendation query and a fictional target product, our method builds a controlled evidence environment that coordinates an agent-faci
    
[^32]: 针对风格自适应和弦生成的流行与爵士数据混合比例实证研究

    Empirical Study of Pop and Jazz Mix Ratios for Genre-Adaptive Chord Generation

    [https://arxiv.org/abs/2605.04998](https://arxiv.org/abs/2605.04998)

    本研究通过对流行预训练Music Transformer进行爵士微调并系统扫描混合的流行“排练”数据量，实证发现排练数据可将爵士top-1提升7至9个百分点，同时将流行风格的遗忘从2.11个百分点降至最多1.19个百分点，为小数据新风格的适应提供了混合比例依据。

    

    本文将和弦进行生成视为一项独立任务，探讨在微调过程中需要保留多少旧领域数据，才能在习得新风格的同时不遗忘旧风格。作者对一个以流行音乐预训练的2500万参数Music Transformer（在留出的流行测试集上top-1下一token准确率为84.21%）进行爵士乐微调研究。由于爵士语料库比流行语料库小两个数量级以上，每次实验均使用全部1,513条爵士训练序列，而所扫描的变量是与其混合的流行“排练”数据量：0、1K、2.5K、5K或10K。在表中所列的最佳epoch处，每个微调后的模型在爵士top-1上提升了7至9个百分点（发布的F1检查点与Phase 0基线持平，如论文的更正说明所解释）；在仅用爵士数据微调时，流行top-1下降了2.11个百分点，而使用排练数据时下降幅度最多为1.19个百分点。Top-1指标还会对每个序列的三个头部token进行评分，其中两个token会改变整个运行结果：jazz-maj……

    arXiv:2605.04998v4 Announce Type: replace-cross  Abstract: This paper treats chord progression generation as a standalone task and asks how much old-domain data must be retained in fine-tuning to acquire a new genre without forgetting the old. I study jazz fine-tuning of a pop-pretrained 25M-parameter Music Transformer (84.21% top-1 next-token accuracy on a held-out pop test set). The jazz corpus is over two orders of magnitude smaller than the pop corpus, so every run uses all 1,513 jazz training sequences, and the swept variable is the volume of pop "rehearsal" data mixed alongside: 0, 1K, 2.5K, 5K, or 10K. At the tabulated best epochs every fine-tuned run gains 7 to 9 points of jazz top-1 (the released F1 checkpoint equals the Phase 0 baseline, as the paper's Correction explains), and pop top-1 falls by 2.11 points under jazz-only fine-tuning but by at most 1.19 with rehearsal. Top-1 also scores the three header tokens of each sequence, and two of them shift whole runs: the jazz-maj
    
[^33]: AgentSelect：面向叙述性查询到智能体推荐的基准测试

    AgentSelect: Benchmark for Narrative Query-to-Agent Recommendation

    [https://arxiv.org/abs/2603.03761](https://arxiv.org/abs/2603.03761)

    AgentSelect是一个将智能体选择重新建模为“叙述性查询到智能体推荐”任务的基准，聚合了超过11万条查询、10.7万个可部署智能体和25万余条交互记录，填补了端到端智能体配置推荐缺乏查询条件监督数据的研究空白。

    

    LLM智能体正迅速成为任务自动化的实用接口，然而该生态系统缺乏一种在快速膨胀的可部署配置空间中进行选择的原则性方法。现有的LLM排行榜和工具/智能体基准测试只是孤立地评估各个组件，并且在任务、指标和候选池之间仍然碎片化，留下了一个关键的研究空白：几乎没有基于查询条件的监督数据来学习推荐将骨干模型与工具包相结合的端到端智能体配置。我们通过AgentSelect来填补这一空白，该基准将智能体选择重新构建为基于能力档案的叙述性查询到智能体的推荐问题，并系统地将异构的评估产物转换为统一的、仅含正样本的交互数据。AgentSelect包含来自40多个数据源的111,179条查询、107,721个可部署智能体和251,103条交互记录，涵盖仅LLM型、仅工具包型以及组合型……（原文摘要至此截断）

    arXiv:2603.03761v2 Announce Type: replace  Abstract: LLM agents are rapidly becoming the practical interface for task automation, yet the ecosystem lacks a principled way to choose among an exploding space of deployable configurations. Existing LLM leaderboards and tool/agent benchmarks evaluate components in isolation and remain fragmented across tasks, metrics, and candidate pools, leaving a critical research gap: there is little query-conditioned supervision for learning to recommend end-to-end agent configurations that couple a backbone model with a toolkit. We address this gap with AgentSelect, a benchmark that reframes agent selection as narrative query-to-agent recommendation over capability profiles and systematically converts heterogeneous evaluation artifacts into unified, positive-only interaction data. AgentSelectcomprises 111,179 queries, 107,721 deployable agents, and 251,103 interaction records aggregated from 40+ sources, spanning LLM-only, toolkit-only, and composition
    
[^34]: OneRanker：工业广告推荐中基于单一模型的统一生成与排序

    OneRanker: Unified Generation and Ranking with One Model in Industrial Advertising Recommendation

    [https://arxiv.org/abs/2603.02999](https://arxiv.org/abs/2603.02999)

    提出统一模型 OneRanker，通过价值感知的多任务解耦架构在工业广告推荐中实现生成与排序的架构级深度融合，解决了兴趣与商业价值错位、生成目标无关以及两阶段割裂等核心挑战。

    

    端到端生成式范式正在彻底变革广告推荐系统，推动传统级联架构向统一建模转变。然而，实际部署面临三大核心挑战：兴趣目标与商业价值之间的错位、生成过程的目标无关性局限，以及生成阶段与排序阶段之间的割裂。现有解决方案往往陷入两难困境：单阶段融合会引发优化上的张力冲突，而阶段解耦则会造成不可逆的信息损失。为此，我们提出 OneRanker，实现了生成与排序在架构层面的深度融合。首先，我们设计了价值感知的多任务解耦架构，通过利用任务 token 序列和因果掩码，在共享表示中分离兴趣覆盖与价值优化空间，有效缓解了目标冲突。其次，（摘要在此处截断，内容不完整）

    arXiv:2603.02999v4 Announce Type: replace  Abstract: The end-to-end generative paradigm is revolutionizing advertising recommendation systems, driving a shift from traditional cascaded architectures towards unified modeling. However, practical deployment faces three core challenges: the misalignment between interest objectives and business value, the target-agnostic limitation of generative processes, and the disconnection between generation and ranking stages. Existing solutions often fall into a dilemma where single-stage fusion induces optimization tension, while stage decoupling causes irreversible information loss. To address this, we propose OneRanker, achieving architectural-level deep integration of generation and ranking. First, we design a value-aware multi-task decoupling architecture. By leveraging task token sequences and causal mask, we separate interest coverage and value optimization spaces within shared representations, effectively alleviating target conflicts. Second,
    
[^35]: FloodSQL-Bench：一个面向地理空间定位Text-to-SQL的检索增强基准

    FloodSQL-Bench: A Retrieval-Augmented Benchmark for Geospatially-Grounded Text-to-SQL

    [https://arxiv.org/abs/2512.12084](https://arxiv.org/abs/2512.12084)

    该论文提出了FLOODSQL-BENCH，一个面向洪水管理领域的地理空间Text-to-SQL检索增强基准，通过键、空间和混合连接整合异构数据集，为高风险应用领域中的Text-to-SQL研究建立了实用测试平台。

    

    现有的Text-to-SQL基准主要关注通用领域中的单表查询或有限的连接操作，因而无法反映特定领域、多表以及地理空间推理的复杂性。为解决这一局限，我们提出了FLOODSQL-BENCH，这是一个面向洪水管理领域的地理空间基准，通过基于键的连接、空间连接和混合连接来整合异构数据集。该基准通过结合社会、基础设施和灾害数据层，捕捉了现实的洪水相关信息需求。我们在相同的检索增强生成设置下系统评估了近期的多个大语言模型，并测量了它们在不同难度级别上的表现。通过提供一个基于真实灾害管理数据的统一开放基准，FLOODSQL-BENCH为推动高风险应用领域的Text-to-SQL研究建立了一个实用的测试平台。

    arXiv:2512.12084v3 Announce Type: replace  Abstract: Existing Text-to-SQL benchmarks primarily focus on single-table queries or limited joins in general-purpose domains, and thus fail to reflect the complexity of domain-specific, multi-table and geospatial reasoning, To address this limitation, we introduce FLOODSQL-BENCH, a geospatially grounded benchmark for the flood management domain that integrates heterogeneous datasets through key-based, spatial, and hybrid joins. The benchmark captures realistic flood-related information needs by combining social, infrastructural, and hazard data layers. We systematically evaluate recent large language models with the same retrieval-augmented generation settings and measure their performance across difficulty tiers. By providing a unified, open benchmark grounded in real-world disaster management data, FLOODSQL-BENCH establishes a practical testbed for advancing Text-to-SQL research in high-stakes application domains.
    

