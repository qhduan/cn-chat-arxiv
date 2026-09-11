# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Generative Late-Interaction Embeddings For Visual Document Retrieval](https://arxiv.org/abs/2609.11808) | 该论文发现视觉文档检索中的后期交互向量精确位于单位球面上并集中在内在维度仅五到六的低维流形附近，据此提出将k-means质心归一化到球面以修正MaxSim分数的系统性低估，并利用页面流形自由度少的特点，从少量向量生成式地重建完整向量集，从而在严格存储预算下保持检索精度。 |
| [^2] | [RAG-Safety-Bench: Reliable Evaluation of Retrieval-Augmented LLM Safety](https://arxiv.org/abs/2609.11758) | 提出了RAG-Safety-Bench基准测试，通过消除检索器质量的干扰并将问题分解为四种条件（非RAG、含答案文档、相关无答案文档、随机文档），可靠地评估检索增强生成对大语言模型安全性的影响。 |
| [^3] | [Your Retriever Already Knows: Distribution-Shape QPP for RAG Retrieval Sufficiency](https://arxiv.org/abs/2609.11646) | 本文提出由24个非词汇特征（以分布形状特征为主）构成的GeneralQPP方法，能以每条查询仅2毫秒的速度、0.856的AUROC判断RAG检索是否充分，性能超越经典QPP方法并大幅领先LLM判别器（快约3000倍），为隐私敏感的本地化部署场景提供了高效的检索充分性评估方案。 |
| [^4] | [FedHUR: Learning Hierarchical Utility-Guided Client Relations for Personalized Federated Recommendation](https://arxiv.org/abs/2609.11632) | 该论文提出了FedHUR框架，通过学习层次化的效用引导客户端关系来优化联邦推荐中的个性化聚合，克服了传统单一全局关系无法捕捉用户关系层次性以及无法直接衡量聚合效用的局限。 |
| [^5] | [TimelyRAG: Semantic-Temporal Hybrid Retrieval for Time-Critical Question Answering in Overlapping-Evolving Documents](https://arxiv.org/abs/2609.11572) | 提出TimelyRAG框架，通过在排序中融入时间距离信息，解决法律等重叠演化文档中版本选择的难题，并发布首个此类基准TimelyQABench，nDCG@10最高提升28.6%。 |
| [^6] | [ReGround: Grounding Reviewer Comments in Multimodal Evidence](https://arxiv.org/abs/2609.11460) | 提出大规模审稿意见定位数据集ReGround，将3,656篇论文中的10,267条审稿意见链接到16,274条证据，并发现证据类型推断是主要瓶颈、多模态证据能提供纯文本检索遗漏的补充信号。 |
| [^7] | [SWRouter: Similarity-Contractive Window Routing for Multi-Turn Large Language Model Conversations](https://arxiv.org/abs/2609.11414) | 提出SWRouter，通过基于相似性的上下文分割机制与双指标评估框架，解决多轮对话中大语言模型路由面临的上下文信息丢失混淆及评估指标耦合两大难题。 |
| [^8] | [VikingRAG: Accurate and Token-efficient Retrieval-augmented Generation over Structured Documents](https://arxiv.org/abs/2609.11390) | VikingRAG通过将代理式多轮检索轨迹物化为可复用的经验边，并引入自适应升级策略在证据充分时采用单轮经验增强检索，在保持高准确率的同时大幅降低了结构化文档检索增强生成的令牌成本。 |
| [^9] | [REVA: Reusable Evidence View Aggregation for Context-Efficient RAG Serving](https://arxiv.org/abs/2609.11209) | REVA通过挖掘目标生成器的历史注意力轨迹，将历史查询-文档-模型交互聚合为可复用的证据视图分数库，从而在不依赖辅助模型和在线压缩的情况下实现上下文高效且低开销的RAG服务。 |
| [^10] | [Agentic Share-of-Search: A Multi-Agent AI System for Competitive Decision-Making in LLM-Mediated E-Commerce](https://arxiv.org/abs/2609.11190) | 本文提出一个多智能体AI系统，以“智能体化搜索份额”为决策目标，自动化测量卖家在AI购物助手中的竞争可见性并诊断根因，实验表明其诊断能力显著优于随机水平。 |
| [^11] | [Benchmark Radar: A Living Database and Search Engine for AI Benchmarks and Evaluation](https://arxiv.org/abs/2609.11115) | 本文提出Benchmark Radar，一个每日自动发现并整合AI基准论文、数据集和代码的动态数据库与搜索引擎，为LLM评估、智能体、编程、推理和安全等领域提供可检索的基准目录、来源引用和分数历史。 |
| [^12] | [UniRec: Cross-stage Multi-Task Fusion with Preference Alignment for Cascaded Recommender Systems](https://arxiv.org/abs/2609.11052) | UniRec提出了一种统一的跨阶段推荐融合模型，通过共享嵌入和单一计算图联合训练实现跨阶段梯度传播，并结合双轴偏好对齐机制，解决级联推荐系统中预排序与排序阶段的跨阶段不一致问题。 |
| [^13] | [Project Qualia: Recovering Experiential Music Structure from Session Co-occurrence Data](https://arxiv.org/abs/2609.10862) | 该研究基于Last.fm的12.9亿次真实收听记录训练歌曲嵌入模型（Song2Vec），并提出艺术家残差方法以消除艺术家身份的影响，从而从收听会话共现数据中恢复超越流派和元数据的歌曲体验相似性结构。 |
| [^14] | [Following the Preference, Missing the Optimum: Compliance Without Optimization in AI Housing Recommendation](https://arxiv.org/abs/2609.10856) | 该研究构建了基于纽约市真实房源的可验证基准，发现AI住房推荐虽然能遵守租房者的显性约束，却经常推荐被严格支配的次优房源，从而错失池中更便宜、通勤更快且面积不小的更优选择。 |
| [^15] | [When Synthetic Data Hurts: On Catastrophic Forgetting in Skill Retrieval for LLM Agents](https://arxiv.org/abs/2609.10750) | 研究发现合成数据微调虽能提升LLM智能体的分布内技能检索效果，但会导致对真实和分布外数据的灾难性遗忘，而借鉴持续学习的微调方法（如LwF、EWC等）既能缓解遗忘，又能将分布内检索性能提升13.98%。 |
| [^16] | [ICEGR: An Intent-Coherent End-to-End Generative Retrieval Framework for E-commerce Search](https://arxiv.org/abs/2608.29652) | 提出ICEGR框架，通过在生成式检索的语义ID构建、监督微调和偏好优化等整个训练流程中一致融入查询意图，解决电商搜索中查询意图不一致的问题，从而提升低曝光商品的检索效果和查询-商品相关性。 |
| [^17] | [ITER: Interaction-Aware Retrieval for Agentic Search](https://arxiv.org/abs/2608.27912) | ITER 是一种交互感知的密集检索器，通过结合主问题、先前子查询以及智能体轨迹学习信号进行训练，在多个智能体骨干上持续优于现有的智能体轨迹训练检索器 LRAT。 |
| [^18] | [ECLASS-Augmented Semantic Product Search for Electronic Components](https://arxiv.org/abs/2604.19664) | 该论文提出将ECLASS标准的层次化语义融入LLM辅助的密集检索与重排序框架，显著提升了工业电子元器件的语义搜索效果，Hit_Rate@5达到94.3%，远超BM25基线的31.4%。 |
| [^19] | [LLMAR: A Tuning-Free Recommendation Framework for Sparse and Text-Rich Industrial Domains](https://arxiv.org/abs/2604.16379) | LLMAR提出了一种免调优的推荐框架，通过LLM推理驱动的标注将行为历史转化为结构化语义动机，并结合反思循环机制进行自我纠错，从而在数据稀疏、文本丰富的工业B2B场景中无需训练即可实现高效的推荐。 |
| [^20] | [MisEdu-RAG: A Misconception-Aware Dual-Hypergraph RAG for Novice Math Teachers](https://arxiv.org/abs/2604.04036) | 提出了面向新手数学教师的误解感知双超图RAG框架MisEdu-RAG，通过将教学知识组织为概念超图、将真实学生错误案例组织为实例超图，并采用两阶段检索，生成基于证据且更具可操作性的教学反馈。 |
| [^21] | [SumRank: Aligning Summarization Models for Long-Document Listwise Reranking](https://arxiv.org/abs/2603.24204) | 提出 SumRank 摘要模型，通过与下游列表式重排序对齐的三阶段训练流程（冷启动SFT、RL数据构建和强化学习对齐），将长文档压缩为排序友好的简洁摘要，从而提升长文档重排序的效果与效率。 |
| [^22] | [OpenResearcher: A Fully Open Pipeline for Long-Horizon Deep Research Trajectory Synthesis](https://arxiv.org/abs/2603.20278) | OpenResearcher提出了一个完全开源、可复现的离线轨迹合成流水线，在1500万文档语料库上合成超过97K条长程深度研究轨迹，微调后的模型在BrowseComp-Plus上相比基础模型提升34个百分点。 |
| [^23] | [Dynamic Feature-Embedding Communication via Codebook Distillation for Federated Recommendation](https://arxiv.org/abs/2601.18570) | 该论文提出RQFedRec，通过残差量化将物品表示为语义ID形式的共享潜在特征嵌入，并利用码本蒸馏实现动态特征嵌入通信，从而降低联邦推荐的通信成本并提升跨物品泛化能力和对噪声反馈的鲁棒性。 |
| [^24] | [MLLMRec: A Preference Reasoning Paradigm with Graph Refinement for Multimodal Recommendation](https://arxiv.org/abs/2508.15304) | MLLMRec利用多模态大语言模型将物品图像转化为高质量语义描述并细化含噪的物品-物品图结构，解决了多模态推荐中用户表示初始化受噪声污染及物品图存在噪声边的问题。 |

# 详细

[^1]: 用于视觉文档检索的生成式后期交互嵌入

    Generative Late-Interaction Embeddings For Visual Document Retrieval

    [https://arxiv.org/abs/2609.11808](https://arxiv.org/abs/2609.11808)

    该论文发现视觉文档检索中的后期交互向量精确位于单位球面上并集中在内在维度仅五到六的低维流形附近，据此提出将k-means质心归一化到球面以修正MaxSim分数的系统性低估，并利用页面流形自由度少的特点，从少量向量生成式地重建完整向量集，从而在严格存储预算下保持检索精度。

    

    后期交互检索是视觉文档搜索领域最先进的技术，但其准确性以存储开销为代价。现有的压缩方法保留每页N~1,000个向量中的子集或局部平均值。然而，在严格的存储预算下，这些方法的性能会急剧下降，而替代方案则需要重新训练编码器。通过对三个编码器的性能退化现象进行研究，我们发现了两个一致的特性：向量精确地位于单位球面上，并集中在内在维度为五到六的流形附近。这种几何结构带来了两个重要见解。首先，标准的k-means质心落在球体内部，导致MaxSim分数被系统性低估。将质心归一化到球面上是一种无需成本的修正方法，相比原始质心可带来高达+0.093 nDCG@5的性能提升。其次，由于页面流形的自由度很少，完整的向量集可以仅从少量向量重新生成。

    arXiv:2609.11808v1 Announce Type: new  Abstract: Late-interaction retrieval is the state-of-the-art for visual document search, but it pays for its accuracy in storage. Existing compression methods retain a subset or local average of the N~1,000 vectors per page. Under aggressive storage budgets, however, these methods degrade sharply, and alternatives require retraining the encoder. Investigating this degradation across three encoders, we found two consistent properties: the vectors lie exactly on the unit sphere and concentrate near a manifold of intrinsic dimension five to six. This geometry yields two insights. First, standard k-means centroids fall inside the sphere, causing systematic underestimation of MaxSim scores. Normalizing them to the surface is a free correction worth up to +0.093 nDCG@5 over raw centroids. Second, because the page manifold has few degrees of freedom, the full set of vectors can be regenerated from only a few. To this end, we introduce Generative Late-Int
    
[^2]: RAG-Safety-Bench：检索增强大语言模型安全性的可靠评估

    RAG-Safety-Bench: Reliable Evaluation of Retrieval-Augmented LLM Safety

    [https://arxiv.org/abs/2609.11758](https://arxiv.org/abs/2609.11758)

    提出了RAG-Safety-Bench基准测试，通过消除检索器质量的干扰并将问题分解为四种条件（非RAG、含答案文档、相关无答案文档、随机文档），可靠地评估检索增强生成对大语言模型安全性的影响。

    

    允许大语言模型（LLM）从一组可信文档中检索信息可以提高可靠性并减少幻觉。然而，最近的研究表明，当被要求生成有害或危险内容时，检索增强生成（RAG）可能对生成响应的整体安全性产生意想不到的副作用。随着越来越多的终端用户转向使用RAG将企业文档和知识库整合到基于LLM的系统中，人们需要更清晰地理解导致这一结果的机制。我们提出了RAG-Safety-Bench，这是一个用于衡量RAG对LLM模型安全影响的基准测试。通过消除检索器质量带来的干扰效应，并将问题清晰地划分为四种条件——非RAG、使用包含有害请求答案的oracle文档的RAG、使用与有害请求相关但不包含具体答案的文档的RAG、以及使用随机文档的RAG（原文在此处被截断）……

    arXiv:2609.11758v1 Announce Type: new  Abstract: Allowing large language models (LLMs) to retrieve information from a set of trusted documents can increase reliability and reduce hallucination. However, recent work has demonstrated that retrieval-augmented generation (RAG) can have unintended side effects on the overall safety of the generated responses, when prompted for harmful or dangerous content. A clearer understanding of the mechanisms leading to this result is needed, as increasing numbers of end users turn to RAG to incorporate corporate documents and knowledge bases into LLM-based systems. We introduce RAG-Safety-Bench, a benchmark to measure the safety impact of RAG on LLM models. By removing the confounding effect of retriever quality, and cleanly separating the problem into four conditions -- non-RAG, RAG with an oracle document containing the answer to the harmful request, RAG with documents related to the harmful request but without the specific answer, and RAG with rand
    
[^3]: 你的检索器早已知晓：用于RAG检索充分性判断的分布形状查询性能预测

    Your Retriever Already Knows: Distribution-Shape QPP for RAG Retrieval Sufficiency

    [https://arxiv.org/abs/2609.11646](https://arxiv.org/abs/2609.11646)

    本文提出由24个非词汇特征（以分布形状特征为主）构成的GeneralQPP方法，能以每条查询仅2毫秒的速度、0.856的AUROC判断RAG检索是否充分，性能超越经典QPP方法并大幅领先LLM判别器（快约3000倍），为隐私敏感的本地化部署场景提供了高效的检索充分性评估方案。

    

    标准的检索增强生成（RAG）流水线在推理时通常无法提供可靠的信号来判断检索是否成功；面对模糊或超出范围的查询时，生成过程可能因此产生幻觉。受捷克一家核监管机构的部署需求启发——由于数据敏感性而无法使用第三方LLM API——我们比较了三种用于评估RAG检索充分性的查询性能预测（QPP）范式：基于分数的特征、基于内容的LLM判别器，以及混合方法。在八个ViDoRe视觉领域（14,514条查询）上，我们的24个非词汇特征（GeneralQPP；包含15个分布形状特征、5个查询表面特征和4个全局特征）以每条查询仅2毫秒的速度达到加权平均AUROC 0.856，超越了经典QPP文献特征池（Classic Full，0.835），并远超本地部署的Qwen3.5 LLM判别器（0.649，差距+0.207；每条查询速度约快3000倍且成本更低）。将LLM判别作为单一特征加入（混合方法）在ViDoRe上可与S1持平（0.863），同时取得了具有统计学显著性的……

    arXiv:2609.11646v1 Announce Type: new  Abstract: Standard Retrieval-Augmented Generation (RAG) pipelines often provide no reliable inference-time signal of whether retrieval succeeded; on ambiguous or out-of-scope queries, generation may then hallucinate. Motivated by a Czech nuclear-regulator deployment where data sensitivity precludes third-party LLM APIs, we compare three Query Performance Prediction (QPP) paradigms for retrieval sufficiency in RAG: score-based features, a content-based LLM judge, and a hybrid. On the eight ViDoRe vision domains (14,514 queries), our 24 non-lexical features (GeneralQPP; 15 distribution-shape, 5 query-surface, 4 global) reach a weighted-average AUROC of 0.856 at 2 ms per query, ahead of a classic-QPP literature pool (Classic Full, 0.835) and well above a local Qwen3.5 LLM judge (0.649, +0.207 gap; $\sim$3000$\times$ faster and cheaper per query). Adding the LLM judgment as one feature (hybrid) matches S1 on ViDoRe (0.863) but gains a statistically si
    
[^4]: FedHUR：面向个性化联邦推荐的层次化效用引导客户端关系学习

    FedHUR: Learning Hierarchical Utility-Guided Client Relations for Personalized Federated Recommendation

    [https://arxiv.org/abs/2609.11632](https://arxiv.org/abs/2609.11632)

    该论文提出了FedHUR框架，通过学习层次化的效用引导客户端关系来优化联邦推荐中的个性化聚合，克服了传统单一全局关系无法捕捉用户关系层次性以及无法直接衡量聚合效用的局限。

    

    联邦推荐能够在保持用户交互数据存储于本地客户端的同时实现协作式模型训练。联邦推荐中的一个核心问题是如何跨客户端聚合有用信息以实现个性化推荐。现有的个性化聚合方法通常基于预定义的参数假设（如参数相似性或互补性）来构建客户端关系，并利用这些关系来确定聚合权重。然而，这类方法构建的是单一的全局关系，不足以捕捉推荐场景中用户关系的层次性与多粒度特性。此外，这些预定义的关系无法直接反映相关客户端在聚合后能否真正提升预测性能。为了解决这些局限性，我们提出了FedHUR，一个用于学习层次化效用引导客户端关系的联邦推荐框架。FedHUR采用……

    arXiv:2609.11632v1 Announce Type: new  Abstract: Federated recommendation enables collaborative model training while keeping user interaction data on local clients. A central problem in federated recommendation is how to aggregate useful information across clients for personalized recommendation. Existing personalized aggregation methods usually construct client relations from predefined parameter-based assumptions, such as parameter similarity or complementarity, and use these relations to determine aggregation weights. However, such methods construct a single global relation, which is insufficient to capture the hierarchical and multi-granularity nature of user relations in recommendation. Moreover, these predefined relations cannot directly reflect whether the related clients can improve prediction performance after aggregation. To address these limitations, we propose FedHUR, a federated recommendation framework for learning hierarchical utility-guided client relations. FedHUR take
    
[^5]: TimelyRAG：面向重叠演化文档中时间敏感问答的语义-时间混合检索

    TimelyRAG: Semantic-Temporal Hybrid Retrieval for Time-Critical Question Answering in Overlapping-Evolving Documents

    [https://arxiv.org/abs/2609.11572](https://arxiv.org/abs/2609.11572)

    提出TimelyRAG框架，通过在排序中融入时间距离信息，解决法律等重叠演化文档中版本选择的难题，并发布首个此类基准TimelyQABench，nDCG@10最高提升28.6%。

    

    尽管大型语言模型（LLM）和检索增强生成（RAG）已经推动了开放域问答（QA）的发展，但当文档通过修订不断演化时，它们仍然不可靠。现有的时间敏感检索方法仅针对离散演化环境，即每次更新都是独立的快照。然而，法律、政策和法规通常在重叠演化环境中运行，其中修订会覆盖较早的条款，同时保留大部分内容，从而在不同版本之间产生强烈的语义重叠。我们提出了TimelyRAG，这是一个与检索器无关的框架，它将时间距离纳入排序中，以使查询与版本合适的文档相匹配。我们还介绍了TimelyQABench，这是首个针对法规密集领域且具有重叠演化挑战的基准数据集。实验显示出一致的提升，nDCG@10最高提升28.6%，突出了时间推理对于可靠问答的重要性。

    arXiv:2609.11572v1 Announce Type: new  Abstract: Although large language models (LLMs) and retrieval-augmented generation (RAG) have advanced open-domain question answering (QA), they remain unreliable when documents evolve through amendments. Existing time-sensitive retrieval methods address only the disjoint-evolving environment, where each update is an independent snapshot. However, laws, policies, and regulations often operate in overlapping-evolving environments, where amendments override earlier clauses while preserving most content, creating strong semantic overlap across versions. We propose TimelyRAG, a retriever-agnostic framework that incorporates temporal distance into ranking to align queries with version-appropriate documents. We also introduce TimelyQABench, the first benchmark for regulation-heavy domains with overlapping-evolving challenges. Experiments show consistent gains, up to +28.6% in nDCG@10, highlighting the importance of temporal reasoning for reliable QA ove
    
[^6]: ReGround：将审稿意见定位到多模态证据

    ReGround: Grounding Reviewer Comments in Multimodal Evidence

    [https://arxiv.org/abs/2609.11460](https://arxiv.org/abs/2609.11460)

    提出大规模审稿意见定位数据集ReGround，将3,656篇论文中的10,267条审稿意见链接到16,274条证据，并发现证据类型推断是主要瓶颈、多模态证据能提供纯文本检索遗漏的补充信号。

    

    审稿意见通常与被审论文的特定部分相关联，但由于论文是长篇多模态文档，将这些意见定位到相应的潜在证据十分困难。现有的基准测试无法涵盖这一场景，且主要关注显式的信息查询类请求。我们提出了ReGround，这是一个用于审稿意见定位任务的大规模数据集，它将3,656篇论文原始匿名投稿中的10,267条审稿意见与16,274条证据相链接。我们的构建基于一个简单的观察：作者的回复中通常包含对投稿内容中用于回应审稿意见部分的显式引用，这为数据集提供了高精度的标注来源。我们将定位任务形式化为一个检索任务，并评估了多种检索方法。结果表明，对论文全部内容进行检索的效果较差，证据类型的推断是主要瓶颈，而多模态证据能够提供仅依靠文本会被遗漏的补充信号。

    arXiv:2609.11460v1 Announce Type: new  Abstract: Reviewer comments naturally relate to specific parts of the reviewed paper, yet grounding these comments to the underlying evidence is difficult due to long multimodal documents. Existing benchmarks do not capture this setting and largely focus on explicit, information-seeking queries. We introduce ReGround, a large-scale dataset for reviewer comment grounding that links 10,267 reviewer comments to 16,274 evidence in the original anonymous submission of 3,656 papers. We build on a simple observation: author rebuttals often include explicit references to content of the submission used to address reviewer comments, providing a high-precision annotation source. We cast grounding as a retrieval task and evaluate a wide range of retrieval methods. Results show that retrieval over the entire paper content performs poorly, evidence-type inference is a major bottleneck, and multimodal evidence provides complementary signals that text alone misse
    
[^7]: SWRouter：面向多轮大语言模型对话的相似性收缩窗口路由

    SWRouter: Similarity-Contractive Window Routing for Multi-Turn Large Language Model Conversations

    [https://arxiv.org/abs/2609.11414](https://arxiv.org/abs/2609.11414)

    提出SWRouter，通过基于相似性的上下文分割机制与双指标评估框架，解决多轮对话中大语言模型路由面临的上下文信息丢失混淆及评估指标耦合两大难题。

    

    大语言模型各具互补优势，这促使研究者开发路由方法，将每个查询分派给最合适的模型。尽管现有路由器在单轮设置中效果良好，但它们无法直接迁移到多轮对话场景——在多轮对话中，路由性能严重依赖于历史上下文如何被分割、保留并融入当前提示词。这带来了两个根本性挑战：一是在上下文构建过程中防止信息丢失与信息混淆，二是在评估路由质量时避免将模型选择与提示词构建质量混为一谈。在本文中，我们提出了SWRouter，一种面向多轮大语言模型路由的相似性收缩窗口路由器。SWRouter将基于相似性的上下文分割机制用于提示词构建，并结合双指标评估框架，将构建准确性与路由器性能解耦。在多个（数据集上的实验……摘要此处截断）

    arXiv:2609.11414v1 Announce Type: new  Abstract: Large language models exhibit complementary strengths, motivating routing methods that dispatch each query to the most suitable model. Although existing routers are effective in single-turn settings, they do not directly transfer to multi-turn dialogue, where routing performance critically depends on how historical context is segmented, retained, and incorporated into the current prompt. This introduces two fundamental challenges: preventing information loss and information confusion during context construction, and evaluating routing quality without conflating model selection with prompt construction quality. In this paper, we propose SWRouter, a Similarity-Contractive Window Router for multi-turn large language model routing. SWRouter combines a similarity-based context segmentation mechanism for prompt construction with a dual-metric evaluation framework that decouples construction accuracy from router performance. Experiments on mult
    
[^8]: VikingRAG：面向结构化文档的准确且令牌高效的检索增强生成

    VikingRAG: Accurate and Token-efficient Retrieval-augmented Generation over Structured Documents

    [https://arxiv.org/abs/2609.11390](https://arxiv.org/abs/2609.11390)

    VikingRAG通过将代理式多轮检索轨迹物化为可复用的经验边，并引入自适应升级策略在证据充分时采用单轮经验增强检索，在保持高准确率的同时大幅降低了结构化文档检索增强生成的令牌成本。

    

    最先进的检索增强生成（RAG）方法利用文档结构来获取充分的证据，但往往会产生高昂的令牌成本。为了在不损害高RAG准确性的前提下减少结构上下文令牌，我们提出了VikingRAG，一个目录感知的语义数据管理系统，它紧密集成语义与结构访问，以支持结构上下文高效、证据缺口驱动的多轮检索。为了进一步降低多轮交互的令牌开销，我们将代理式多轮检索轨迹物化为经验边，并对相似查询复用这些边，避免重复的多轮探索。此外，为了在无需代理式多轮检索时减少令牌成本，我们引入了一种自适应升级策略，当证据充分时从单轮经验增强检索中直接回答，仅在必要时才调用代理式多轮检索。

    arXiv:2609.11390v1 Announce Type: cross  Abstract: State-of-the-art retrieval-augmented generation (RAG) methods exploit document structures to acquire sufficient evidence, but often incur substantial token costs. To reduce structural-context tokens without compromising high RAG accuracy, we present {\sf VikingRAG}, a directory-aware semantic data management system that tightly integrates semantic and structural access to support structural-context-efficient, evidence-gap-driven multi-round retrieval. To further reduce token overhead of multi-round interaction, we materialize agentic multi-round retrieval traces as experience edges, and reuse these edges for similar queries, avoiding repeated multi-round exploration. To additionally reduce token costs when agentic multi-round retrieval is unnecessary, we introduce an adaptive escalation strategy that answers from one-round experience-augmented retrieval when the evidence is sufficient, and invokes agentic multi-round retrieval only oth
    
[^9]: REVA：面向上下文高效RAG服务的可复用证据视图聚合

    REVA: Reusable Evidence View Aggregation for Context-Efficient RAG Serving

    [https://arxiv.org/abs/2609.11209](https://arxiv.org/abs/2609.11209)

    REVA通过挖掘目标生成器的历史注意力轨迹，将历史查询-文档-模型交互聚合为可复用的证据视图分数库，从而在不依赖辅助模型和在线压缩的情况下实现上下文高效且低开销的RAG服务。

    

    检索增强生成（RAG）通过以检索到的文档为条件进行生成，提升了知识密集型大语言模型（LLM）应用的表现，但更长的上下文会增加延迟、键值（KV）缓存内存以及token成本。检索后压缩可以降低这一成本，然而现有的压缩器通常针对每个查询独立运行，依赖辅助模型或重写操作，并引入在线开销，这可能抵消较短提示词带来的收益。我们从数据挖掘的视角重新审视RAG压缩，将历史的查询-文档-模型交互聚合为可复用的证据视图。我们首先证明，现代压缩器相比简单的截断方法收益并不稳定，且可能带来显著的推理时延迟。随后我们提出可复用证据视图聚合（REVA），这是一个将目标生成器的历史注意力轨迹挖掘为以文档为键、与预算无关的分数存储库的框架。REVA将

    arXiv:2609.11209v1 Announce Type: cross  Abstract: Retrieval-augmented generation (RAG) improves knowledge-intensive large language model (LLM) applications by conditioning generation on retrieved documents, but longer contexts increase latency, key-value (KV) cache memory, and token cost. Post-retrieval compression can reduce this cost, yet existing compressors often operate independently for each query, rely on auxiliary models or rewriting, and introduce online overhead that can offset the benefit of shorter prompts. We revisit RAG compression from a data-mining perspective by aggregating historical query--document--model interactions into reusable evidence views. We first show that modern compressors have unstable gains over simple truncation and can add substantial inference-time latency. We then propose Reusable Evidence View Aggregation (REVA), a framework that mines the target generator's historical attention traces into a document-keyed, budget-agnostic score store. REVA maps 
    
[^10]: 智能体化搜索份额：面向LLM介导电商竞争决策的多智能体AI系统

    Agentic Share-of-Search: A Multi-Agent AI System for Competitive Decision-Making in LLM-Mediated E-Commerce

    [https://arxiv.org/abs/2609.11190](https://arxiv.org/abs/2609.11190)

    本文提出一个多智能体AI系统，以“智能体化搜索份额”为决策目标，自动化测量卖家在AI购物助手中的竞争可见性并诊断根因，实验表明其诊断能力显著优于随机水平。

    

    AI购物助手正日益改变消费者的商品发现方式，这迫切需要支持卖方竞争决策的工具。我们提出了一个多智能体AI系统，用于自动化测量LLM介导电商中的竞争可见性并进行根因诊断。该系统引入了“智能体化搜索份额”作为决策目标，在各大主流AI平台上部署查询智能体，并使用基于ReAct的诊断智能体来推荐按优先级排序的商品化干预措施。一项100次试验的消融研究（作为该原型的可行性评估）显示，该智能体在39%的试验中成功恢复了被消融的信号（95% CI: 30.0% - 48.8%，是随机概率的5.5倍），而在高相关性消融试验中这一比例上升至63.9%。

    arXiv:2609.11190v1 Announce Type: cross  Abstract: AI shopping assistants increasingly redirect consumer discovery, creating an urgent need for tools that support seller-side competitive decision-making. We present a multi-agent AI system that automates competitive visibility measurement and root cause diagnosis in LLM-mediated ecommerce. The system introduces Agentic Share-of-Search (ASoS) as the decision target, deploys query agents across leading AI platforms, and uses a ReAct-based diagnostic agent to recommend prioritized merchandising interventions. A 100-trial ablation study, presented as a feasibility evaluation of this prototype, shows the agent recovers the ablated signal in 39% of trials (95% CI: 30.0% - 48.8%, 5.5x over chance), rising to 63.9% among high-correlation ablations.
    
[^11]: 基准雷达：面向AI基准与评估的动态数据库和搜索引擎

    Benchmark Radar: A Living Database and Search Engine for AI Benchmarks and Evaluation

    [https://arxiv.org/abs/2609.11115](https://arxiv.org/abs/2609.11115)

    本文提出Benchmark Radar，一个每日自动发现并整合AI基准论文、数据集和代码的动态数据库与搜索引擎，为LLM评估、智能体、编程、推理和安全等领域提供可检索的基准目录、来源引用和分数历史。

    

    基准研究人员和大语言模型（LLM）及其他AI系统的开发者需要找到相关的评估方法、定位其基准数据集和代码，并理解已报告分数背后的设置。我们提出了Benchmark Radar（基准雷达），这是一个用于检索和发现AI基准的动态数据库和搜索引擎，涵盖LLM评估、智能体和工具使用基准、编程、推理、安全性以及特定领域评估。该系统将每日发现的基准论文、代码仓库、数据集和发布版本与可搜索的基准目录、模型卡和技术报告中的提及以及分数历史相结合。它保留了源身份信息和引用，以便读者可以查看候选基准及其评估证据。每日发现基于37个来源：13个直接连接器和24个第一方研究与工程信息源。该目录包含来自4个基准目录的1,283条源记录

    arXiv:2609.11115v1 Announce Type: cross  Abstract: Benchmark researchers and developers of large language models (LLMs) and other AI systems need to find relevant evaluations, locate their benchmark datasets and code, and understand the settings behind reported scores. We present Benchmark Radar, a living database and search engine for retrieval and discovery of AI benchmarks, covering LLM evaluation, agentic and tool-use benchmarks, coding, reasoning, safety, and domain-specific evaluations. The system combines daily discovery of benchmark papers, repositories, datasets, and releases with a searchable benchmark catalog, mentions in model cards and technical reports, and score histories. It retains source identities and citations so readers can inspect candidate benchmarks and their evaluation evidence. Daily discovery draws on 37 sources: 13 direct connectors and 24 first-party research and engineering feeds. The catalog contains 1,283 source records drawn from 4 benchmark catalogs an
    
[^12]: UniRec：面向级联推荐系统的跨阶段多任务融合与偏好对齐

    UniRec: Cross-stage Multi-Task Fusion with Preference Alignment for Cascaded Recommender Systems

    [https://arxiv.org/abs/2609.11052](https://arxiv.org/abs/2609.11052)

    UniRec提出了一种统一的跨阶段推荐融合模型，通过共享嵌入和单一计算图联合训练实现跨阶段梯度传播，并结合双轴偏好对齐机制，解决级联推荐系统中预排序与排序阶段的跨阶段不一致问题。

    

    工业推荐系统采用具有不同目标、特征空间和延迟约束的级联阶段。分别优化预排序和排序可能导致跨阶段不一致：上游模型可能会过滤掉下游排序器所偏好的物品，而独立调优的下游融合可能会抵消上游的改进效果。现有多任务融合方法主要关注排序阶段内的多目标融合，而跨阶段方法通常只是在上游排序中添加下游分数因子。对两个阶段融合模块的联合优化在很大程度上仍未被探索。我们提出了UniRec，一个统一的跨阶段推荐融合模型。首先，两个融合代理部分共享输入嵌入，并在单一计算图中进行训练，因此来自任一阶段的梯度都会通过共享表示进行传播并影响另一个阶段。其次，我们引入了双轴偏好对齐

    arXiv:2609.11052v1 Announce Type: new  Abstract: Industrial recommender systems use cascaded stages with different objectives, feature spaces, and latency constraints. Optimizing pre-ranking and ranking separately can create cross-stage inconsistency: upstream models may filter out items preferred by downstream rankers, and independently tuned downstream fusion can offset upstream improvements. Existing multi-task fusion methods focus on multi-objective fusion within the ranking stage, and cross-stage methods typically only add a downstream score factor to upstream ranking. Joint optimization of fusion modules across both stages remains largely unexplored.   We propose UniRec, a Unified Cross-stage Recommendation Fusion model. First, the two fusion agents partially share input embeddings and are trained in a single computation graph, so gradients from either stage propagate through the shared representation and influence the other. Second, we introduce a dual-axis preference alignment 
    
[^13]: 感受质项目（Project Qualia）：从会话共现数据中恢复体验性音乐结构

    Project Qualia: Recovering Experiential Music Structure from Session Co-occurrence Data

    [https://arxiv.org/abs/2609.10862](https://arxiv.org/abs/2609.10862)

    该研究基于Last.fm的12.9亿次真实收听记录训练歌曲嵌入模型（Song2Vec），并提出艺术家残差方法以消除艺术家身份的影响，从而从收听会话共现数据中恢复超越流派和元数据的歌曲体验相似性结构。

    

    本报告呈现了“感受质项目”的研究成果。该项目是一项持续进行的研究，旨在探究歌曲之间的体验相似性——这种结构无法被流派或元数据分类体系所捕捉——能否从真实的收听行为中恢复出来。我们构建了一个大规模的收听会话数据集，包含通过Last.fm API从9,396名用户收集的12.9亿次收听记录，并经预处理流程缩减为涵盖2,860万个会话的5.316亿条训练记录。在该语料库上，我们训练了一个skip-gram Word2Vec模型（Song2Vec），将每个会话视为一个句子，每首曲目视为一个词元。正如预期的那样，所得到的嵌入空间主要由艺术家身份主导，这是会话中连续播放同一艺术家歌曲所导致的结果。为了检验一种更微妙的、与艺术家无关的信号，我们开发了一种艺术家残差方法：从每首曲目的嵌入向量中减去其所属艺术家的质心，并评估剩余部分是否保留了……（摘要原文此处截断）

    arXiv:2609.10862v1 Announce Type: cross  Abstract: This report presents results from Project Qualia, an ongoing effort to determine whether experiential similarity between songs, a structure not captured by genre or metadata taxonomies, can be recovered from real listening behavior. We constructed a large-scale dataset of listening sessions, comprising 1.29 billion scrobbles collected from 9,396 users via the Last.fm API and reduced through a preprocessing pipeline to 531.6 million training scrobbles across 28.6 million sessions. On this corpus, we trained a skip-gram Word2Vec model (Song2Vec), treating each session as a sentence and each track as a token. As anticipated, the resulting embedding space was dominated by artist identity, a consequence of single-artist runs within sessions. To test for a subtler, artist-independent signal, we developed an artist-residual procedure: subtracting each artist's centroid from its tracks' embeddings and evaluating whether the remainder retained 
    
[^14]: 顺从偏好，错失最优：AI住房推荐中的“合规却非优化”现象

    Following the Preference, Missing the Optimum: Compliance Without Optimization in AI Housing Recommendation

    [https://arxiv.org/abs/2609.10856](https://arxiv.org/abs/2609.10856)

    该研究构建了基于纽约市真实房源的可验证基准，发现AI住房推荐虽然能遵守租房者的显性约束，却经常推荐被严格支配的次优房源，从而错失池中更便宜、通勤更快且面积不小的更优选择。

    

    大型语言模型正在成为消费者搜索的首要入口，而其所处领域往往利益重大且法律规则明确。现有审计表明，模型会根据感知到的用户身份来引导找房者，但由于缺乏一个可枚举的房源清单来评估遗漏，尚无研究能说明当推荐系统忽略某个合适选项时用户究竟损失了什么。我们以可验证的真实标准对AI住房推荐进行了审计。针对纽约市的150个合成租房者场景中的每一个，我们构建了一个包含120个真实房源的池子，这些房源的租金、卧室数量以及基于GTFS计算的公共交通通勤时间均已知；我们计算了满足租房者明示约束的精确集合，并推导出其帕累托前沿。主要评估结果不假设任何效用函数：如果同一房源池中存在比推荐房源更便宜、通勤更快且卧室数量不少于它的房源，则该推荐即被视为严格被支配。在来自两家供应商的三个模型共计9,945次调用中，模型的合规性表现……（摘要在此处截断）

    arXiv:2609.10856v1 Announce Type: cross  Abstract: Large language models are becoming the first point of contact for consumer search in domains where the stakes are material and the law is explicit. Existing audits show that models steer housing seekers by perceived identity, but none can say what a user loses when a recommender overlooks a suitable option, for want of an enumerated inventory to score omissions against. We audit AI housing recommendation against a verifiable ground truth. For each of 150 synthetic renter scenarios in New York City we build a pool of 120 real listings with known rent, bedrooms and GTFS-computed transit commute, compute the exact set satisfying the renter's stated constraints, and derive its Pareto frontier. The primary outcome assumes no utility function: a recommendation is strictly dominated if the same pool holds a listing cheaper, faster to commute from and no smaller in bedrooms. Across 9,945 calls to three models from two vendors, compliance is ne
    
[^15]: 当合成数据有害时：论LLM智能体技能检索中的灾难性遗忘

    When Synthetic Data Hurts: On Catastrophic Forgetting in Skill Retrieval for LLM Agents

    [https://arxiv.org/abs/2609.10750](https://arxiv.org/abs/2609.10750)

    研究发现合成数据微调虽能提升LLM智能体的分布内技能检索效果，但会导致对真实和分布外数据的灾难性遗忘，而借鉴持续学习的微调方法（如LwF、EWC等）既能缓解遗忘，又能将分布内检索性能提升13.98%。

    

    LLM智能体越来越依赖在运行时检索的外部技能，这使得从大型技能库中选择技能成为一个关键挑战。我们提出了一个覆盖34,396个技能的生产级技能路由器，并开展了一项使用有限真实监督和合成数据的大规模技能检索研究。我们发现，合成数据微调虽然能提升分布内检索效果，但会导致对真实数据和分布外（OOD）数据的灾难性遗忘。我们评估了几种受持续学习启发的遗忘缓解微调方法，包括嵌入锚点正则化、无遗忘学习（LwF）、弹性权重巩固（EWC）和L2初始化。结果表明，这些方法不仅能保持OOD技能检索的性能，还能将0.6B Qwen检索器和重排器在合成分布内技能上的检索性能提升13.98%。我们的研究结果提供了一个实用的基准和鲁棒的方案。

    arXiv:2609.10750v1 Announce Type: cross  Abstract: LLM agents increasingly rely on external skills retrieved at runtime, making skill selection from large repositories a critical challenge. We present a production skill router over 34,396 skills and a large-scale study of skill retrieval using limited real supervision and synthetic data. We found that the synthetic-data fine-tuning improves in-distribution retrieval but it causes catastrophic forgetting on real and out-of-distribution (OOD) data. We evaluate several forgetting mitigation fine-tuning approaches inspired by continual learning, including embedding-anchor regularization, Learning without Forgetting (LwF), Elastic Weight Consolidation (EWC), and L2-initialization. The results show that these approaches not only retain the performance on OOD skills retrieval but also improve the retrieval on synthetic in-distribution skills by 13.98\% for 0.6B Qwen retriever and reranker. Our results provide a practical benchmark and a robus
    
[^16]: ICEGR：面向电商搜索的意图连贯端到端生成式检索框架

    ICEGR: An Intent-Coherent End-to-End Generative Retrieval Framework for E-commerce Search

    [https://arxiv.org/abs/2608.29652](https://arxiv.org/abs/2608.29652)

    提出ICEGR框架，通过在生成式检索的语义ID构建、监督微调和偏好优化等整个训练流程中一致融入查询意图，解决电商搜索中查询意图不一致的问题，从而提升低曝光商品的检索效果和查询-商品相关性。

    

    生成式检索（GR）在电商搜索中前景广阔，但现有方法难以在整个训练流程中保持查询意图的一致性。首先，基于静态商品信息的语义ID（SID）构建方式限制了SID编码商品-意图关联的能力。其次，尽管监督微调（SFT）能够学习整个商品目录中的商品-SID映射，但由于查询到SID的训练仅依赖在线日志，低曝光商品仍然缺乏真实的查询意图监督，导致这些商品的检索性能不佳。第三，面向业务的偏好优化可能偏向热门或高价值商品，而非最匹配查询意图的商品，从而削弱了查询与商品之间的相关性。为解决这些问题，我们提出了ICEGR，一个面向电商搜索的意图连贯端到端生成式检索框架，它在整个GR训练流程中一致地融合查询意图（原文摘要在此处截断）。

    arXiv:2608.29652v1 Announce Type: new  Abstract: Generative Retrieval (GR) is promising for e-commerce search, yet existing methods struggle to maintain query-intent consistency throughout the training pipeline. First, semantic ID (SID) construction based on static product information limits the ability of SIDs to encode product-intent associations. Second, although supervised fine-tuning (SFT) learns product-SID mappings across the catalog, low-exposure products still lack real query-intent supervision because query-to-SID training relies solely on online logs, resulting in poor retrieval performance for these products. Third, business-oriented preference optimization may favor popular or high-value products over those that best match the query intent, weakening query-product relevance. To address these issues, we propose ICEGR, an Intent-Coherent End-to-End Generative Retrieval Framework for E-commerce Search that integrates query intent consistently throughout the GR training pipeli
    
[^17]: ITER：面向智能体搜索的交互感知检索

    ITER: Interaction-Aware Retrieval for Agentic Search

    [https://arxiv.org/abs/2608.27912](https://arxiv.org/abs/2608.27912)

    ITER 是一种交互感知的密集检索器，通过结合主问题、先前子查询以及智能体轨迹学习信号进行训练，在多个智能体骨干上持续优于现有的智能体轨迹训练检索器 LRAT。

    

    深度研究智能体通过迭代的搜索步骤序列来回答复杂的用户问题，其中智能体自主构建子查询以检索每个阶段所需的证据。然而，现有的检索器训练通常仅依赖当前步骤的子查询及其对应的搜索结果作为训练信号，导致从先前交互中积累的信息在很大程度上未被充分利用。我们提出了 ITER，这是一种利用智能体轨迹学习信号训练的智能体交互感知密集检索器。ITER 在表示每个查询时，不仅纳入当前子查询，还纳入主问题和先前的子查询，并使用从智能体交互中导出的轨迹相对学习信号进行训练。在来自三个模型家族的六个智能体骨干上，ITER 持续优于现有的基于智能体轨迹训练的密集检索器 LRAT，取得了平均改进……（摘要在此处被截断）

    arXiv:2608.27912v1 Announce Type: new  Abstract: Deep-research agents answer complex user questions through an iterative sequence of search steps, where the agent autonomously formulates sub-queries to retrieve the evidence needed at each stage. However, existing retriever training typically relies only on the sub-query and its corresponding search results at the current step as training signals, leaving the information accumulated from previous interactions largely underutilized.   We introduce iter, an agent interaction-aware dense retriever trained using agent trajectory learning signals. iter represents each query by incorporating not only the current sub-query, but also the main question and preceding sub-queries, and is trained using trajectory-relative learning signals derived from the agent's interactions.   Across six agent backbones from three model families, iter consistently outperforms the existing agent-trajectory-trained dense retriever, LRAT, achieving an average improv
    
[^18]: 基于ECLASS增强的电子元器件语义产品搜索

    ECLASS-Augmented Semantic Product Search for Electronic Components

    [https://arxiv.org/abs/2604.19664](https://arxiv.org/abs/2604.19664)

    该论文提出将ECLASS标准的层次化语义融入LLM辅助的密集检索与重排序框架，显著提升了工业电子元器件的语义搜索效果，Hit_Rate@5达到94.3%，远超BM25基线的31.4%。

    

    对工业产品数据的高效语义访问是工厂自动化和新兴的基于大语言模型（LLM）的智能体工作流的关键推动因素，在这些场景中，人类工程师和自主智能体都需要从高度结构化的产品目录中识别合适的元器件。然而，自然语言查询与以属性为中心的产品描述之间的词汇不匹配限制了传统检索方法（如BM25）的有效性。在本工作中，我们对LLM辅助的密集检索在工业电子元器件语义产品搜索中的应用进行了系统性评估，并研究了将ECLASS标准中的层次化语义整合到基于嵌入向量的检索方法中。我们的结果表明，结合重排序的密集检索方法显著优于经典词法方法和基础模型网络搜索基线。特别是，所提出的方法实现了94.3%的Hit_Rate@5，而BM25仅为31.4%。

    arXiv:2604.19664v2 Announce Type: replace  Abstract: Efficient semantic access to industrial product data is a key enabler for factory automation and emerging LLM-based agent workflows, where both human engineers and autonomous agents must identify suitable components from highly structured catalogs. However, the vocabulary mismatch between natural-language queries and attribute-centric product descriptions limits the effectiveness of traditional retrieval approaches, e.g., BM25. In this work, we present a systematic evaluation of LLM-assisted dense retrieval for semantic product search on industrial electronic components, and investigate the integration of hierarchical semantics from the ECLASS standard into embedding-based retrieval. Our results show that dense retrieval combined with re-ranking substantially outperforms classical lexical methods and foundation model web-search baselines. In particular, the proposed approach achieves a Hit_Rate@5 of 94.3 %, compared to 31.4 % for BM2
    
[^19]: LLMAR：面向稀疏且富文本工业领域的免调优推荐框架

    LLMAR: A Tuning-Free Recommendation Framework for Sparse and Text-Rich Industrial Domains

    [https://arxiv.org/abs/2604.16379](https://arxiv.org/abs/2604.16379)

    LLMAR提出了一种免调优的推荐框架，通过LLM推理驱动的标注将行为历史转化为结构化语义动机，并结合反思循环机制进行自我纠错，从而在数据稀疏、文本丰富的工业B2B场景中无需训练即可实现高效的推荐。

    

    工业B2B应用（如建筑工地风险预测、物资采购）面临极端的数据稀疏性，但同时又具有丰富的文本交互。在这类环境中，传统的基于ID的协同过滤因缺乏共现信号而失效，而微调标准大语言模型（LLM）会带来高昂的运营成本，且难以应对频繁的数据漂移。我们提出了LLMAR（LLM标注推荐），这是一个免调优的框架。该框架超越了简单的嵌入方法，系统地整合LLM推理来捕捉用户的“潜在动机”，而无需任何训练过程。我们引入了三个核心贡献：（1）推理驱动标注：利用LLM将行为历史转化为结构化的语义动机，实现基于推理的匹配，这是基于ID的方法无法达到的；（2）反思循环：一种自我纠错机制，通过细化生成的查询来减轻幻觉并解决……

    arXiv:2604.16379v3 Announce Type: replace-cross  Abstract: Industrial B2B applications (e.g., construction site risk prediction, material procurement) face extreme data sparsity yet feature rich textual interactions. In such environments, traditional ID-based collaborative filtering fails lacking co-occurrence signals, while fine-tuning standard Large Language Models (LLMs) incurs high operational costs and struggles with frequent data drift.   We propose LLMAR (LLM-Annotated Recommendation), a tuning-free framework. Moving beyond simple embeddings, LLMAR systematically integrates LLM reasoning to capture user "latent motives" without any training process. We introduce three core contributions: (1) Inference-Driven Annotation: uses LLMs to transform behavioral history into structured semantic motives, enabling reasoning-based matching unattainable by ID-based methods; (2) Reflection Loop: a self-correction mechanism that refines generated queries to mitigate hallucinations and resolve 
    
[^20]: MisEdu-RAG：面向新手数学教师的误解感知双超图检索增强生成框架

    MisEdu-RAG: A Misconception-Aware Dual-Hypergraph RAG for Novice Math Teachers

    [https://arxiv.org/abs/2604.04036](https://arxiv.org/abs/2604.04036)

    提出了面向新手数学教师的误解感知双超图RAG框架MisEdu-RAG，通过将教学知识组织为概念超图、将真实学生错误案例组织为实例超图，并采用两阶段检索，生成基于证据且更具可操作性的教学反馈。

    

    新手数学教师经常会遇到难以诊断和纠正的学生错误，其中误解尤为棘手，因为教师既要解释错在哪里，又要说明如何解决。尽管现有许多大型语言模型（LLM）平台可以辅助生成教学反馈，但这些LLM对教学知识和学生错误之间的关联较为松散，可能导致给出的指导对教师而言缺乏可操作性。为了弥补这一不足，我们提出了MisEdu-RAG，这是一个基于双超图的检索增强生成（RAG）框架，它将教学知识组织为概念超图，将真实的学生错误案例组织为实例超图。给定一个查询，MisEdu-RAG执行两阶段检索，从两个层级收集相互关联的证据，并基于检索到的案例和教学原则生成回答。我们在MisstepMath数据集上进行了评估，该数据集包含数学错误……（原文摘要在此处截断）

    arXiv:2604.04036v2 Announce Type: replace-cross  Abstract: Novice math teachers often encounter students' mistakes that are difficult to diagnose and remediate. Misconceptions are especially challenging because teachers must explain what went wrong and how to solve them. Although many existing large language model (LLM) platforms can assist in generating instructional feedback, these LLMs loosely connect pedagogical knowledge and student mistakes, which might make the guidance less actionable for teachers. To address this gap, we propose MisEdu-RAG, a dual-hypergraph-based retrieval-augmented generation (RAG) framework that organizes pedagogical knowledge as a concept hypergraph and real student mistake cases as an instance hypergraph. Given a query, MisEdu-RAG performs a two-stage retrieval to gather connected evidence from both layers and generates a response grounded in the retrieved cases and pedagogical principles. We evaluate on \textit{MisstepMath}, a dataset of math mistakes pa
    
[^21]: SumRank：面向长文档列表式重排序的摘要模型对齐

    SumRank: Aligning Summarization Models for Long-Document Listwise Reranking

    [https://arxiv.org/abs/2603.24204](https://arxiv.org/abs/2603.24204)

    提出 SumRank 摘要模型，通过与下游列表式重排序对齐的三阶段训练流程（冷启动SFT、RL数据构建和强化学习对齐），将长文档压缩为排序友好的简洁摘要，从而提升长文档重排序的效果与效率。

    

    大型语言模型（LLMs）在列表式段落重排序任务中展现出了卓越的性能。然而，由于上下文长度大幅增加，直接将其应用于长文档排序会同时带来效果和效率方面的问题。为应对这一挑战，我们提出了一个逐点式摘要模型 SumRank，该模型与下游的列表式重排序对齐，可在最终列表式重排序阶段之前将长文档压缩为简洁的、与排序对齐的摘要。为了获得摘要模型 SumRank，我们引入了一个三阶段训练流程，包括冷启动监督微调（SFT）、专门的强化学习数据构建以及通过强化学习进行的排序驱动对齐。这一范式使 SumRank 与下游排序目标保持一致，从而保留相关性信号。我们在来自 TREC 深度学习赛道（TREC DL 19-23）的五个基准数据集上进行了广泛的实验。

    arXiv:2603.24204v2 Announce Type: replace  Abstract: Large Language Models (LLMs) have demonstrated superior performance in listwise passage reranking task. However, directly applying them to rank long-form documents introduces both effectiveness and efficiency issues due to the substantially increased context length. To address this challenge, we propose a pointwise summarization model SumRank, aligned with downstream listwise reranking, to compress long-form documents into concise rank-aligned summaries before the final listwise reranking stage. To obtain our summarization model SumRank, we introduce a three-stage training pipeline comprising cold-start Supervised Fine-Tuning (SFT), specialized RL data construction, and rank-driven alignment via Reinforcement Learning. This paradigm aligns the SumRank with downstream ranking objectives to preserve relevance signals. We conduct extensive experiments on five benchmark datasets from the TREC Deep Learning tracks (TREC DL 19-23). Results
    
[^22]: OpenResearcher：面向长程深度研究轨迹合成的完全开源流水线

    OpenResearcher: A Fully Open Pipeline for Long-Horizon Deep Research Trajectory Synthesis

    [https://arxiv.org/abs/2603.20278](https://arxiv.org/abs/2603.20278)

    OpenResearcher提出了一个完全开源、可复现的离线轨迹合成流水线，在1500万文档语料库上合成超过97K条长程深度研究轨迹，微调后的模型在BrowseComp-Plus上相比基础模型提升34个百分点。

    

    训练深度研究智能体需要长程轨迹，这些轨迹交织着搜索、证据聚合与多步推理。然而，现有的数据收集流水线通常依赖专有的网络API，使得大规模轨迹合成成本高昂、不稳定且难以复现。我们提出了OpenResearcher，一个可复现的流水线，它将一次性的语料库构建与多轮轨迹合成解耦，并使用三个显式的浏览器原语（搜索、打开、查找）在1500万文档的语料库上完全离线地执行搜索与浏览循环。以GPT-OSS-120B作为教师模型，我们合成了超过97K条轨迹，其中包括大量包含100+次工具调用的长程尾部轨迹。在这些轨迹上对30B-A3B骨干模型进行监督微调后，在BrowseComp-Plus基准上达到54.8%的准确率，相比基础模型提升34.0个百分点，同时在BrowseComp、GAIA等基准上保持竞争力（原文此处截断）。

    arXiv:2603.20278v2 Announce Type: replace-cross  Abstract: Training deep research agents requires long-horizon trajectories that interleave search, evidence aggregation, and multi-step reasoning. However, existing data collection pipelines typically rely on proprietary web APIs, making large-scale trajectory synthesis costly, unstable, and difficult to reproduce. We present OpenResearcher, a reproducible pipeline that decouples one-time corpus bootstrapping from multi-turn trajectory synthesis and executes the search-and-browse loop entirely offline using three explicit browser primitives: search, open, and find, over a 15M-document corpus. Using GPT-OSS-120B as the teacher model, we synthesize over 97K trajectories, including a substantial long-horizon tail with 100+ tool calls. Supervised fine-tuning a 30B-A3B backbone on these trajectories achieves 54.8\% accuracy on BrowseComp-Plus, a +34.0 point improvement over the base model, while remaining competitive on BrowseComp, GAIA, and 
    
[^23]: 通过码本蒸馏实现动态特征嵌入通信的联邦推荐

    Dynamic Feature-Embedding Communication via Codebook Distillation for Federated Recommendation

    [https://arxiv.org/abs/2601.18570](https://arxiv.org/abs/2601.18570)

    该论文提出RQFedRec，通过残差量化将物品表示为语义ID形式的共享潜在特征嵌入，并利用码本蒸馏实现动态特征嵌入通信，从而降低联邦推荐的通信成本并提升跨物品泛化能力和对噪声反馈的鲁棒性。

    

    联邦推荐系统通常通过将用户参数保留在本地设备上来保护用户隐私，同时交换物品参数以进行协同模型训练。然而，这类物品参数通常独立地对物品进行建模，在效率和效果上都面临挑战，导致通信成本随物品空间增长，并限制了跨物品的泛化能力和对噪声反馈的鲁棒性。为解决这些局限，我们提出通过共享的潜在特征嵌入来建模物品并进行通信。残差量化为实例化这种通信提供了一种自然的方式，它用一串简短的离散码ID序列来表示每个物品，即语义ID。然而，由于1）私有且有偏差的历史交互，以及2）不断演化的协同信息，直接将集中式、静态的基于残差量化的推荐应用于联邦学习并非易事。我们提出了RQFedRec，

    arXiv:2601.18570v2 Announce Type: replace  Abstract: Federated recommendation systems commonly protect user privacy by keeping user parameters on local devices, while exchanging item parameters for collaborative model training. However, such item parameters usually model items independently and suffer from both efficiency and effectiveness challenges, making communication costs grow with the item space and limiting cross-item generalization and robustness to noisy feedback. To address these limitations, we propose to model items via shared latent feature embeddings for communication. Residual Quantization (RQ) provides a natural way to instantiate this communication by representing each item with a short sequence of discrete code IDs, i.e., Semantic IDs (SIDs). However, directly applying centralized and static RQ-based recommendation to federated learning is non-trivial due to 1) private and biased historical interactions and 2) evolving collaborative information. We propose RQFedRec, 
    
[^24]: MLLMRec：一种面向多模态推荐的基于图细化的偏好推理范式

    MLLMRec: A Preference Reasoning Paradigm with Graph Refinement for Multimodal Recommendation

    [https://arxiv.org/abs/2508.15304](https://arxiv.org/abs/2508.15304)

    MLLMRec利用多模态大语言模型将物品图像转化为高质量语义描述并细化含噪的物品-物品图结构，解决了多模态推荐中用户表示初始化受噪声污染及物品图存在噪声边的问题。

    

    多模态推荐将用户历史行为与物品的模态特征相结合，以捕捉有形的用户偏好，相比传统的基于ID的推荐系统展现出更优的性能。然而，现有方法在用户和物品的表示学习方面仍面临两个关键问题：（1）多模态用户表示的初始化要么对历史行为不敏感，要么被无关的模态噪声所污染；（2）广泛使用的基于KNN的物品-物品图包含大量低相似度的噪声边，且缺乏受众共现关系。为解决这些问题，我们提出了MLLMRec，一种面向多模态推荐的基于图细化的新型偏好推理范式。具体而言，一方面，首先利用多模态大语言模型（MLLM）将物品图像转换为高质量的语义描述，从而架起……

    arXiv:2508.15304v3 Announce Type: replace  Abstract: Multimodal recommendation combines the user historical behaviors with the modal features of items to capture the tangible user preferences, presenting superior performance compared to the conventional ID-based recommender systems. However, existing methods still encounter two key problems in the representation learning of users and items, respectively: (1) the initialization of multimodal user representations is either agnostic to historical behaviors or contaminated by irrelevant modal noise, and (2) the widely used KNN-based item-item graph contains noisy edges with low similarities and lacks audience co-occurrence relationships. To address such issues, we propose MLLMRec, a novel preference reasoning paradigm with graph refinement for multimodal recommendation. Specifically, on the one hand, the item images are first converted into high-quality semantic descriptions using a multimodal large language model (MLLM), thereby bridging 
    

