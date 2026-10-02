# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ScholarCatalyst: A Benchmark for Retrieving Papers That Inspire New Research](https://arxiv.org/abs/2610.02202) | 该论文构建了ScholarCatalyst基准，由207篇计算机科学论文的184位主导作者标注哪些先前文献激发或推进了其研究，据此提出基于作者判断的论文检索任务，并发现智能体搜索的表现并不优于简单的嵌入检索。 |
| [^2] | [Optimizing Effective Training Time for Large-Scale Recommendation Systems](https://arxiv.org/abs/2610.02057) | 本文提出以有效训练时间（ETT%）作为运维度量框架，系统性地分析并优化大规模推荐系统训练中的生命周期开销，通过覆盖整个训练栈的多项优化手段，显著提升数千GPU集群中原本仅占50-60%的有效训练时间占比。 |
| [^3] | [A Matryoshka Hierarchical RAG for Efficient Multi-Hop Question Answering](https://arxiv.org/abs/2610.01767) | MatRAG通过将文档聚类的语义层次与Matryoshka表示学习的嵌套维度对齐，以较低维度索引粗粒度层级，从而在降低索引与查询计算成本的同时保证多跳问答的检索质量。 |
| [^4] | [AgentWebRec: Compact Evidence Fusion over the Agent Web for Personalized Recommendation](https://arxiv.org/abs/2610.01705) | 该论文提出AgentWebRec，将智能体网络上的个性化推荐重新建模为有限证据预算下的任务时证据获取与融合问题，通过决定“询问什么”与“保留什么”来应对证据分散、查询受限和语义异构的挑战。 |
| [^5] | [From Rules to Neural Graphs: Scalable Structured Prediction for Patent Prior Art Search](https://arxiv.org/abs/2610.01553) | 该论文提出一种神经解析器，通过局部双仿射注意力直接从专利文本预测发明图，以低 3 倍的推理成本超越基于规则的解析器，实现了可处理超过 4 万词元超长文档的可扩展专利现有技术检索。 |
| [^6] | [Neither Black nor White: Balancing Semantic and Collaborative Signals with Graph-Informed Semantic IDs (GrIS)](https://arxiv.org/abs/2610.01533) | 该论文将生成式推荐中语义标识符的构建重新定义为一个递归聚类问题，提出统一框架 GrIS——在节点承载语义内容、边承载协同信号的图上进行层次化划分，并将 RQ-VAE、RQ-KMeans 等先前方法统一为图为空时的特殊情形。 |
| [^7] | [Learning to structure data from user-generated thematic corpora](https://arxiv.org/abs/2610.01463) | 该论文提出了一种完全自动化的迭代框架，利用大语言模型在无需预定义本体的情况下，从用户生成的主题语料库（如社交媒体社区）中发现并提取领域特定的属性模式，实现非结构化文本数据的结构化，并支持以可估算的精度损失使用较小模型进行值提取。 |
| [^8] | [Not All Is Lost: Repairing Lossy User Preference States of Personalization Encoders](https://arxiv.org/abs/2610.01270) | 提出REPAIR方法，通过在紧凑可学习的坐标空间中比较冻结编码器缓存的历史表示与当前偏好状态，选择性地聚合纠正性证据来修复有损的用户偏好状态，从而在多个数据集上显著提升所有十二个推荐主机的排序性能。 |
| [^9] | [Do Multilingual Encoders Produce Language-Consistent Semantic IDs?](https://arxiv.org/abs/2610.01139) | 该研究发现多语言编码器会将同一商品的不同语言版本嵌入到明显分离的位置，导致语义ID高度不一致（日语翻译仅7.7%保留英语原文的首个语义ID码），且没有证据表明量化器选择性地放大了翻译引起的位移，说明仅依靠多语言编码器不足以实现语言一致的语义ID。 |
| [^10] | [Madeleine: Learning Involuntary Recall for Conversational Memory from Simulated Lives](https://arxiv.org/abs/2610.01118) | Madeleine通过LLM人生模拟器离线学习记忆间的“非自主”关联（摊销化关联），在线阶段无需任何LLM调用、仅替换查询编码器即可接入任意向量记忆系统，以极低成本回忆起与当前话题不相似却至关重要的记忆，并在LoCoMo-Plus上取得最佳性能。 |
| [^11] | [JoinGR: Learning to Traverse Join Graphs for Table Retrieval](https://arxiv.org/abs/2610.01064) | JOINGR是一种连接感知的表格检索方法，它将数据库连接图作为检索空间，通过遍历连接边并聚合边贡献，能够识别出问题中未直接提及、但可通过连接关系推断出的所需表格。 |
| [^12] | [The Other Half of Workflow Portability: Evidence-Backed HPC Site Profiles with Agentic Discovery](https://arxiv.org/abs/2610.00971) | 该论文指出工作流可移植性缺失的“另一半”是站点本身的使用方式（资源形态、存储、网络权限与运行策略），并提出通过智能体自动发现、构建基于证据的HPC站点画像，使自动化部署工具能够将站点知识转化为部署决策，免去人工调整。 |
| [^13] | [RPTune: Learned Context Curation for LLM Catalog Search](https://arxiv.org/abs/2610.00964) | RPTune提出一个端到端框架，通过“学习式目录整理（商品排序与剪枝）”和“基于自动生成目录监督的LLM后训练”相互促进，为中小商户的上下文内商品目录搜索提供了优于多阶段检索的新方案。 |
| [^14] | [CANOPY: Adaptive-Granularity Evidence Compression for Multimodal RAG](https://arxiv.org/abs/2610.00923) | 提出CANOPY框架，通过将检索条目表示为层次结构并利用父级相对细化机制，在多模态RAG中实现无需LLM调用的自适应粒度证据压缩。 |
| [^15] | [TabJoinBench: A Benchmark for Joinable Table Discovery](https://arxiv.org/abs/2610.00817) | 本文提出TabJoinBench——一个覆盖语义型、关系型和混合型数据湖场景的可连接表发现基准，通过可组合扰动系统地引入结构、表示和语义变化并保持可靠真值，解决了以往方法特定基准难以进行可重现和公平比较的问题。 |
| [^16] | [Enterprise Representation Simplification (ERS): Reducing Representational Complexity for Enterprise AI](https://arxiv.org/abs/2610.00791) | 本文提出了企业表示简化（ERS）理念和企业表示复杂性（ERC）模型，后者通过对象、交互、行为和支持源四个维度以表示中立的方式量化比较企业表示的复杂性，从而为降低企业AI所面临的表示复杂性提供了框架。 |
| [^17] | [Comparison of Common Crawl News & GDELT](https://arxiv.org/abs/2610.00587) | 本文比较了 GDELT 和 Common Crawl News 两个新闻数据集，分析了它们各自的优势与局限，并揭示了两者在新闻来源获取渠道上的显著差异。 |
| [^18] | [A Shared Taste for Model-Written Text: The Generator-by-Selector Matrices of "AI-AI Bias" Show No Detectable Own-Model Premium](https://arxiv.org/abs/2610.00369) | 本研究重建了“AI-AI偏见”实验的三个5×5生成器×选择器矩阵并进行精确置换检验，发现语言模型选择器并不存在对自家模型生成文本的额外偏好，“自家模型溢价”在统计上不可检测，表明各模型对模型撰写文本的偏好是共享的而非自我偏爱。 |
| [^19] | [System Attribution in LLM Brand Recommendations: Single Responses Identify the System, Aggregated Brand Profiles Do Not Transfer](https://arxiv.org/abs/2610.00253) | 该研究发现，单条回复可通过简单的字符n-gram分类器以97.84%的准确率识别其来源的大语言模型系统，但由品牌推荐汇总而成的聚合品牌画像无法在不同系统间迁移使用。 |
| [^20] | [On-Device Commercial Intent Retrieval Under Size, Latency, and Privacy Constraints: A 3 MiB Retrieval System with Typed Egress Boundaries](https://arxiv.org/abs/2610.00170) | 该论文在3 MiB载荷、20ms p95延迟与零数据外泄三重约束下，构建了一个完全端侧运行的商业意图检索系统，通过从韩语句子变换器蒸馏并4比特量化的静态嵌入表，在6020叶节点的商业分类上取得75.0%的中类top-5准确率，并揭示了准确率在“查询是否包含叶节点名称子串”上的剧烈分化（83.5% 对 45.2%）。 |
| [^21] | [Ask a Language Model for Lottery Numbers: Concentration in Repeated Six-of-49 Outputs](https://arxiv.org/abs/2610.00052) | 语言模型在生成“49选6”随机彩票号码时输出高度集中，其多样性远低于真正的均匀随机抽样水平。 |
| [^22] | [Exploring Forum Post Retrieval with Generative Modeling](https://arxiv.org/abs/2609.38646) | 论文针对新界面 Facebook Forum 交互数据稀疏的问题，通过在 Facebook 群组互动数据上训练并复用从跨平台 Feed 数据学到的分层前缀式语义 ID，微调 30 亿参数指令语言模型直接从用户上下文生成语义 ID，实现了生成式的论坛帖子检索推荐。 |
| [^23] | [TAGGRAPH: Tag-Augmented Graphs for Graph Retrieval of Agent Persistent Histories](https://arxiv.org/abs/2609.38353) | 本文提出基于共享5W式对话记忆的受控评估框架，系统比较了标签增强图配置、AdaptiveGraph、BM25与OpenClaw在LLM智能体长期记忆检索上的表现，发现最优检索方法随基准和测试规模而变化，没有单一方法在所有设置下都占优。 |
| [^24] | [Route What Remains: A Meta-Modal Agent for Missing-Modality Candidate Reranking in Recommender Systems](https://arxiv.org/abs/2605.25007) | 提出元模态智能体（MMA），将缺失模态推荐中的候选重排序形式化为预算约束下的顺序证据获取问题，通过PPO学习自适应调用文本、图像和交互图工具进行稀疏重打分，在全部九组数据集与路径组合中均取得最优结果，NDCG@10最高提升10.0%。 |

# 详细

[^1]: ScholarCatalyst：一个用于检索能激发新研究的论文的基准测试

    ScholarCatalyst: A Benchmark for Retrieving Papers That Inspire New Research

    [https://arxiv.org/abs/2610.02202](https://arxiv.org/abs/2610.02202)

    该论文构建了ScholarCatalyst基准，由207篇计算机科学论文的184位主导作者标注哪些先前文献激发或推进了其研究，据此提出基于作者判断的论文检索任务，并发现智能体搜索的表现并不优于简单的嵌入检索。

    

    是什么造就了伟大的科学家？即使人工智能系统开始在开放性问题上取得进展，在感知一个新问题需要哪项埋藏于不断增长的科研档案库中的先前思想方面，科学家仍然远远领先于AI。为了研究这一技能，我们求助于那些亲身了解哪些早期工作推进了其已完成项目的研究人员，并以论文作为指向其中思想的指针。利用使作者标注可规模化的自动化流程，我们让207篇近期计算机科学论文的184位主导作者标注哪些候选论文确实推进或可能推进了他们的项目，且每个标注均附有详细理由，从而构建了ScholarCatalyst数据集。我们提出一个由作者提供判断的检索任务：给定初始研究问题，仅从项目开始时可获得的文献中检索这些论文。结果表明，智能体搜索尽管调用了同一个检索器，其表现并不优于嵌入检索（Recall@20分别为0.42与0.48）。

    arXiv:2610.02202v1 Announce Type: new  Abstract: What makes great scientists great? Even as AI systems start to make progress on open problems, scientists remain far ahead of them at sensing which prior idea, buried in an ever-growing archive of research, a new problem needs. To study this skill, we draw on researchers who know firsthand which earlier work advanced their completed projects, with papers serving as pointers to the ideas within. Using our automated pipeline that makes author annotation scalable, we build ScholarCatalyst by having 184 lead authors of 207 recent computer science papers label which candidates did or could have advanced their project, each with a detailed rationale. We introduce a retrieval task with author-provided judgments: given an initial research question, retrieve these papers from only the literature available when the project began. Agentic search does no better than embedding retrieval (0.42 vs. 0.48 Recall@20) despite calling that same retriever as
    
[^2]: 大规模推荐系统有效训练时间的优化

    Optimizing Effective Training Time for Large-Scale Recommendation Systems

    [https://arxiv.org/abs/2610.02057](https://arxiv.org/abs/2610.02057)

    本文提出以有效训练时间（ETT%）作为运维度量框架，系统性地分析并优化大规模推荐系统训练中的生命周期开销，通过覆盖整个训练栈的多项优化手段，显著提升数千GPU集群中原本仅占50-60%的有效训练时间占比。

    

    生命周期开销在大规模推荐系统训练集群中悄然消耗着加速器的计算容量。我们最大的推荐工作负载每天在数千个GPU上处理数百亿条训练样本。在本工作之前，其端到端总时长中仅有50-60%用于推进新数据上的训练。我们对这种生命周期开销进行了集群规模的系统性研究，并提出了一系列覆盖完整训练栈的优化方案。我们使用有效训练时间（ETT%）作为运维框架来度量损失的时间，将其定位到由不同团队独立负责的基础设施组件上，并揭示跨作业重启时被重复执行的工作。这一分析指导了诸多优化措施，如训练器初始化期间的通信消除与流水线重叠；动态形状处理、自动调优剪枝以及可复用的PyTorch 2编译缓存；异步检查点；独立模型发布；以及故障恢复成本的降低。我们评估了这些优化的效果。

    arXiv:2610.02057v1 Announce Type: new  Abstract: Lifecycle overhead silently consumes accelerator capacity across large-scale recommendation training fleets. Our largest recommendation workloads process tens of billions train- ing examples per day on thousands of GPUs. Before this work, only 50-60% of their end-to-end wall time advanced training on new data. We present a fleet-scale study of this lifecycle overhead and a set of optimizations spanning the full training stack. We use Effective Training Time (ETT%) as an operational framework to instrument lost time, localize it to independently owned infrastructure components, and expose work repeated across job restarts. This analysis guides optimizations like communication elimination and pipeline overlap during trainer initialization; dynamic-shape handling, autotuning pruning, and reusable Py- Torch 2 compilation caches; asynchronous checkpointing; stan- dalone model publishing; and reductions in recovery cost. We evaluate the optimi
    
[^3]: 一种用于高效多跳问答的Matryoshka分层检索增强生成框架

    A Matryoshka Hierarchical RAG for Efficient Multi-Hop Question Answering

    [https://arxiv.org/abs/2610.01767](https://arxiv.org/abs/2610.01767)

    MatRAG通过将文档聚类的语义层次与Matryoshka表示学习的嵌套维度对齐，以较低维度索引粗粒度层级，从而在降低索引与查询计算成本的同时保证多跳问答的检索质量。

    

    面向多跳问答的检索增强生成（RAG）系统必须在检索质量与计算成本之间取得平衡。这些成本既可能产生于索引阶段——例如使用昂贵知识图谱（KG）或大语言模型（LLM）来生成摘要——也可能产生于查询阶段——例如迭代式的LLM驱动检索。为了在保持检索质量的同时降低这些成本，我们提出了MatRAG，这是一个将RAG系统与Matryoshka表示学习（MRL）相结合的分层框架。MatRAG通过将聚类结构的语义层次与MRL的嵌套结构对齐，同时应对上述两类成本。具体而言，它将文档语料库组织成一个粒度逐渐粗化的聚类有向无环图（DAG），其中每一层由较低的Matryoshka维度进行索引。MatRAG将对DAG的迭代式自顶向下遍历与一种实体驱动的控制机制相结合……

    arXiv:2610.01767v1 Announce Type: cross  Abstract: Retrieval-Augmented Generation (RAG) systems for multi-hop Question Answering (QA) must balance retrieval quality with computational cost. This cost is incurred during indexing time, through the use of expensive Knowledge Graphs (KGs) or Large Language Models (LLMs) to generate summaries, or during querying, through iterative LLM-driven retrieval. To reduce it while maintaining retrieval quality, we present MatRAG, a hierarchical framework that combines RAG systems with Matryoshka Representation Learning (MRL). MatRAG addresses both kinds of cost by aligning the semantic hierarchy of a clustering structure with the nested structure of MRL. Specifically, it organizes the corpus of documents into a Directed Acyclic Graph (DAG) of clusters with progressively coarser granularity. Each level is indexed by a lower Matryoshka dimension. MatRAG pairs an iterative, top-down traversal of the DAG with an entity-driven mechanism that controls the 
    
[^4]: AgentWebRec：面向个性化推荐的智能体网络紧凑证据融合

    AgentWebRec: Compact Evidence Fusion over the Agent Web for Personalized Recommendation

    [https://arxiv.org/abs/2610.01705](https://arxiv.org/abs/2610.01705)

    该论文提出AgentWebRec，将智能体网络上的个性化推荐重新建模为有限证据预算下的任务时证据获取与融合问题，通过决定“询问什么”与“保留什么”来应对证据分散、查询受限和语义异构的挑战。

    

    基于大语言模型（LLM）的个人智能体正在成为用户语义的持久载体以及用户与推荐平台之间的中介，并在本地维护更丰富的用户知识。随着智能体之间的相互交互，传统的“用户—平台”关系演变为“用户—智能体网络—平台”的信息通路，使分布式的用户侧信息能够补充物品侧信息。然而，这一新通路对传统推荐提出了挑战：证据分散在彼此不透明的智能体之中，只能通过有界查询获取；其中仅有一小部分与当前的推荐决策相关；且不同智能体返回的响应在语义上是异构的。因此，作者将智能体网络上的推荐重新构建为一个在有限证据预算下的“任务时证据获取与融合”问题，通过决定“问什么”和“保留什么”，而非（学习……原文在此处截断）

    arXiv:2610.01705v1 Announce Type: new  Abstract: LLM-based personal agents are emerging as persistent carriers of user semantics and intermediaries between users and recommendation platforms, maintaining richer user knowledge locally. As agents interact with one another, the conventional \textit{User--Platform} relation evolves into a \textit{User--Agent Web--Platform} information pathway, enabling distributed user-side information to complement item-side information. This new pathway, however, defies conventional recommendation: evidence is scattered across mutually opaque agents and reachable only through bounded queries, only a small portion of it is relevant to the current recommendation decision, and the responses returned by different agents are semantically heterogeneous. We therefore recast recommendation over the agent web as a \emph{task-time evidence acquisition and fusion} problem under a finite evidence budget by deciding what to ask and what to keep, rather than learning 
    
[^5]: 从规则到神经图：面向专利现有技术检索的可扩展结构化预测

    From Rules to Neural Graphs: Scalable Structured Prediction for Patent Prior Art Search

    [https://arxiv.org/abs/2610.01553](https://arxiv.org/abs/2610.01553)

    该论文提出一种神经解析器，通过局部双仿射注意力直接从专利文本预测发明图，以低 3 倍的推理成本超越基于规则的解析器，实现了可处理超过 4 万词元超长文档的可扩展专利现有技术检索。

    

    专利检索需要处理通常超过数万个词元的文档。大多数神经检索方法在截断的输入上运行，限制了其有效性。基于图的检索通过将每项专利表示为结构化的发明图来解决这一问题，但构建这些图依赖于脆弱的基于规则的解析器。我们提出了神经解析器，它将依存句法分析中的双仿射注意力机制加以改造，直接从专利文本预测发明图。我们的局部双仿射注意力将成对评分限制在滑动窗口内，将复杂度从 O(n²) 降低到 O(n·w)。由于局部评分与全局评分共享相同的权重，该模型可以在短序列上训练，并在超过 40,000 词元的文档上直接部署而无需重新训练。该模型从 100 万份规则解析的文档中蒸馏而来，以低 3 倍的推理成本超越了其教师模型：神经图在短查询上将引用召回率提高了 0.5%。

    arXiv:2610.01553v1 Announce Type: new  Abstract: Patent search requires processing documents routinely exceeding tens of thousands of tokens. Most neural retrieval approaches operate on truncated inputs, limiting their effectiveness. Graph-based retrieval addresses this by representing each patent as a structured invention graph, but constructing these graphs relies on brittle rule-based parsers. We present the neural parser, which adapts biaffine attention from dependency parsing to predict invention graphs directly from patent text. Our local biaffine attention restricts pairwise scoring to a sliding window, reducing complexity from $O(n^2)$ to $O(n \cdot w)$. Since local and global scoring share the same weights, the model trains on short sequences and deploys on documents exceeding 40,000 tokens without retraining. Distilled from 1 million rule-parsed documents, it surpasses its teacher at 3$\times$ lower inference cost: neural graphs improve citation recall by 0.5% on short querie
    
[^6]: 非黑即白之外：基于图感知语义标识符（GrIS）平衡语义信号与协同信号

    Neither Black nor White: Balancing Semantic and Collaborative Signals with Graph-Informed Semantic IDs (GrIS)

    [https://arxiv.org/abs/2610.01533](https://arxiv.org/abs/2610.01533)

    该论文将生成式推荐中语义标识符的构建重新定义为一个递归聚类问题，提出统一框架 GrIS——在节点承载语义内容、边承载协同信号的图上进行层次化划分，并将 RQ-VAE、RQ-KMeans 等先前方法统一为图为空时的特殊情形。

    

    arXiv:2610.01533v1 公告类型：新 摘要：现有关于生成式推荐中语义标识符（Semantic IDs, SIDs）的工作将 SID 的构建视为一个表示学习问题：将物品编码到量化的潜空间中并读取离散编码。我们认为这种视角只是偶然的表象。SID 构建在本质上是一个递归聚类问题，而一旦以这种方式表述，聚类的自然对象便是一个图——其节点承载语义内容，其边承载协同信号；SID 的分配由此转化为一个层次化的图划分问题。这一重新定义产生了一个统一框架——图感知语义标识符（GrIS），它并非取代先前的方法，而是将其囊括其中。RQ-VAE 和 RQ-KMeans 可以被恢复为图为空这一特殊情形，从而揭示出仅基于内容的量化只是沿着两个此前被合并的设计轴（图构建与递归划分算法）所构成的更大设计空间中的一个角落。我们探索了两种截然不同的具体实现：RecDMoN，它执行层次化的……（原文摘要在此处截断）

    arXiv:2610.01533v1 Announce Type: new  Abstract: Existing work on Semantic IDs (SIDs) for generative recommendation treats SID construction as a representation learning problem: encode items into a quantised latent space and read off codes. We argue this view is incidental. SID construction is, at heart, a recursive clustering problem, and once stated this way the natural object to cluster is a graph whose nodes carry semantic content and whose edges carry collaborative signal; SID assignment becomes a hierarchical graph partition. This reframing yields a unified framework, Graph-Informed Semantic IDs (GrIS), that subsumes prior approaches rather than displacing them. RQ-VAE and RQ-KMeans are recovered as the special case where the graph is empty, exposing content-only quantisation as one corner of a larger design space along two so-far-collapsed axes: graph construction and recursive partition algorithm. We explore two contrasting instantiations: RecDMoN, which performs hierarchical a
    
[^7]: 从用户生成的主题语料库中学习数据结构化

    Learning to structure data from user-generated thematic corpora

    [https://arxiv.org/abs/2610.01463](https://arxiv.org/abs/2610.01463)

    该论文提出了一种完全自动化的迭代框架，利用大语言模型在无需预定义本体的情况下，从用户生成的主题语料库（如社交媒体社区）中发现并提取领域特定的属性模式，实现非结构化文本数据的结构化，并支持以可估算的精度损失使用较小模型进行值提取。

    

    主题语料库（如社交媒体社区）包含描述可被结构化数据的非结构化文本，例如社交媒体数据中提及的个人属性、行为和经历。提取结构化数据具有挑战性，因为相关属性往往是隐式的、依赖于领域的，且事先未知。我们提出了一个完全自动化的迭代框架，无需预定义本体即可发现和提取特定领域的属性模式。该框架利用大语言模型（LLM）归纳候选属性，依次合并语义重叠的属性，并分配结构类型，从而实现本体的创建，并用语料库中的值对其进行填充。该框架还支持使用较小的LLM进行值提取，与大型LLM相比，其精度损失可被估算。我们在5个健康相关的Reddit社区上对该框架进行了评估。

    arXiv:2610.01463v1 Announce Type: new  Abstract: Thematic corpora, such as social media communities, contain unstructured text describing data that could be made structured. These include, for example, personal attributes, behaviors, and experiences mentioned in social media data. Extracting structured data is challenging as relevant attributes are often implicit, domain-dependent, and unknown in advance. We propose a fully automated, iterative framework for discovering and extracting domain-specific attribute schemas without a predefined ontology. Using large language models (LLMs), the framework induces candidate attributes, sequentially consolidates semantically overlapping attributes, and assigns a structural type. These enable creating an ontology and populating it with values from the corpus. The framework also enables the use of smaller LLMs for value extraction with estimable accuracy loss compared to large LLMs. We evaluate the framework on 5 health-related Reddit communities.
    
[^8]: 并非一切皆失：修复个性化编码器的有损用户偏好状态

    Not All Is Lost: Repairing Lossy User Preference States of Personalization Encoders

    [https://arxiv.org/abs/2610.01270](https://arxiv.org/abs/2610.01270)

    提出REPAIR方法，通过在紧凑可学习的坐标空间中比较冻结编码器缓存的历史表示与当前偏好状态，选择性地聚合纠正性证据来修复有损的用户偏好状态，从而在多个数据集上显著提升所有十二个推荐主机的排序性能。

    

    个性化编码器将不断演化的交互历史压缩为偏好状态，用于对物品进行排序或为文本生成提供条件。仅基于该状态运行的任务头可能会遗漏冻结编码器缓存的各个时间步表示中仍留存的有用证据。我们研究了这种可恢复性差距，并提出REPAIR方法，它在一个紧凑的可学习坐标空间中将缓存的表示与当前偏好状态进行比较。它能够从长程历史、近期交互以及局部突发交互中解析出纠正性证据。随后，它选择哪些时间步上的哪些模式有贡献，并将其聚合的校正量在任务头之前加到偏好状态上。由编码器托管的修复机制复用现有前向计算中的表示，无需重新编码历史。在MovieLens、PENS、MIND和Amazon Reviews 2023数据集上，仅训练REPAIR即可提升全部十二个代表性推荐主机的MRR和nDCG@10指标。

    arXiv:2610.01270v1 Announce Type: cross  Abstract: Personalization encoders compress evolving interaction histories into preference states used to rank items or condition text generation. A task head operating only on this state can miss useful evidence that remains in the frozen encoder's cached representations for individual timesteps. We study this recoverability gap and propose REPAIR, which compares cached representations with the current preference state in a compact learned coordinate space. It resolves corrective evidence over extended history, recent interactions, and localized bursts. It then selects which patterns at which timesteps contribute and adds their aggregate correction to the state before the task head. Encoder-host repair reuses representations from the existing forward computation without re-encoding the history. Across MovieLens, PENS, MIND, and Amazon Reviews 2023, training only REPAIR improves MRR and nDCG@10 for all twelve representative recommendation hosts 
    
[^9]: 多语言编码器会产生语言一致的语义ID吗？

    Do Multilingual Encoders Produce Language-Consistent Semantic IDs?

    [https://arxiv.org/abs/2610.01139](https://arxiv.org/abs/2610.01139)

    该研究发现多语言编码器会将同一商品的不同语言版本嵌入到明显分离的位置，导致语义ID高度不一致（日语翻译仅7.7%保留英语原文的首个语义ID码），且没有证据表明量化器选择性地放大了翻译引起的位移，说明仅依靠多语言编码器不足以实现语言一致的语义ID。

    

    语义ID（Semantic IDs, SIDs）将商品嵌入压缩为离散码序列，用于生成式检索。我们探究一个问题：多语言编码器是否足以让同一商品的不同语言版本获得语言一致的语义ID。我们使用以英语、西班牙语和日语呈现的Amazon ESCI商品列表，测试了翻译版本是否与英语原文保持接近、残差量化对翻译引起的位移是否异常敏感，以及多语言或语言均衡的量化器拟合是否能提升语义ID的一致性。实验表明，Multilingual E5使翻译版本在嵌入空间中存在可测量的距离：在偏向英语的量化器拟合下，日语翻译仅在7.7%的情况下保留其英语原文对应的第一个语义ID码，而英语改写版本的这一比例为89.0%。距离匹配的面向商品的对照组产生了与翻译几乎相同的完整语义ID不匹配程度，没有证据表明量化器选择性地放大了翻译引起的位移（摘要在此处被截断）。

    arXiv:2610.01139v1 Announce Type: new  Abstract: Semantic IDs (SIDs) compress item embeddings into discrete code sequences used in generative retrieval. We ask whether a multilingual encoder is sufficient for different-language renderings of the same product to receive language-consistent SIDs. Using Amazon ESCI listings rendered in English, Spanish, and Japanese, we test whether translations remain close to their English source, whether residual quantization is unusually sensitive to translation-induced movement, and whether multilingual or language-balanced quantizer fitting improves SID agreement. Multilingual E5 places translations measurably apart: under an English-heavy fit, a Japanese translation preserves the first SID code of its English counterpart in only 7.7% of cases, compared with 89.0% for an English rewording. Distance-matched product-directed controls produce nearly the same full-SID mismatch as translation, providing no evidence that the quantizer selectively amplifie
    
[^10]: Madeleine：从模拟人生中学习对话记忆的非自主回忆

    Madeleine: Learning Involuntary Recall for Conversational Memory from Simulated Lives

    [https://arxiv.org/abs/2610.01118](https://arxiv.org/abs/2610.01118)

    Madeleine通过LLM人生模拟器离线学习记忆间的“非自主”关联（摊销化关联），在线阶段无需任何LLM调用、仅替换查询编码器即可接入任意向量记忆系统，以极低成本回忆起与当前话题不相似却至关重要的记忆，并在LoCoMo-Plus上取得最佳性能。

    

    长期对话助手必须在正确的时刻回忆起正确的记忆，然而最重要的记忆往往与用户当前所说的内容并不相似。现有系统通过让大语言模型（LLM）在写入或读取时进行推理来恢复这类关联，代价是每个记忆库需要数百到上千次LLM调用，且每次查询需消耗多达数千个上下文token。我们提出：关联是一种可学习的相关性，即在人类生活的展开过程中，记忆之间的逐点互信息。我们介绍Madeleine，它学习摊销化的关联：在离线阶段，一个LLM人生模拟器撰写模拟人生，其中的线索-触发对教会查询编码器在冻结的相似度之上学习残差关联；在在线阶段，它不调用任何LLM，仅需替换查询编码器即可接入任何向量记忆系统。在官方协议下的LoCoMo-Plus基准上，将Madeleine (I)接入HyperMem时达到66.6，是所有被评估系统中最高的。

    arXiv:2610.01118v1 Announce Type: cross  Abstract: A long-term conversational assistant must recall the right memory at the right moment, yet the memory that matters most is often not similar to what the user says now. Current systems recover such associations by letting an LLM reason at write or read time, at a cost of hundreds to over a thousand LLM calls per memory bank and up to several thousand context tokens per query. We argue that association is a learnable relevance: the pointwise mutual information of memories under how human lives unfold. We introduce Madeleine, which learns amortized association: offline, an LLM life simulator writes simulated lives, whose cue-trigger pairs teach a query encoder a residual association on top of frozen similarity; online, it calls no LLM and plugs into any vector memory by replacing only the query encoder. On LoCoMo-Plus under the official protocol, Madeleine (I) reaches 66.6 when plugged into HyperMem, the highest among all systems evaluate
    
[^11]: JoinGR：学习遍历连接图以实现表格检索

    JoinGR: Learning to Traverse Join Graphs for Table Retrieval

    [https://arxiv.org/abs/2610.01064](https://arxiv.org/abs/2610.01064)

    JOINGR是一种连接感知的表格检索方法，它将数据库连接图作为检索空间，通过遍历连接边并聚合边贡献，能够识别出问题中未直接提及、但可通过连接关系推断出的所需表格。

    

    在真实数据库上进行Text-to-SQL的前提是检索到正确的表格。稠密表格检索器独立地对模式元素进行排序，但这忽略了一个关键的证据来源：一些所需的表格并未在问题中被直接提及，只有通过与已相关表格之间的连接关系才能被识别出来。我们提出了JOINGR，一种连接感知的表格检索方法，它将数据库连接图视为检索空间。列被表示为图节点，而表内关系和外键关系则被表示为类型化边。给定一个问题，JOINGR会选择语义相似的锚点表格，使用查询条件化的评分器遍历连接边，并将由此产生的边贡献聚合为表格分数。该评分器是构建在冻结的查询、节点和边嵌入之上的轻量级MLP，并通过对黄金表格使用成对间隔损失进行训练。在BIRD和Spider数据集上，JOINGR表现出与现有方法相当的竞争力。

    arXiv:2610.01064v1 Announce Type: cross  Abstract: Retrieving the right tables is a prerequisite for Text-to-SQL over realistic databases. Dense table retrievers rank schema elements independently, but this ignores a key source of evidence: some required tables are not mentioned in the question and become identifiable only through their join relationships to already relevant tables. We introduce JOINGR, a join-aware table retrieval method that treats the database join graph as the retrieval space. Columns are represented as graph nodes, while intra-table and foreign-key relationships are represented as typed edges. Given a question, JOINGR selects semantically similar anchor tables, traverses join edges with a query-conditioned scorer, and aggregates the resulting edge deposits into table scores. The scorer is a lightweight MLP on top of frozen query, node, and edge embeddings, trained with a pairwise margin loss over gold tables. On BIRD and Spider datasets, JOINGR is competitive with
    
[^12]: 工作流可移植性的另一半：基于证据的HPC站点画像与智能体自动发现

    The Other Half of Workflow Portability: Evidence-Backed HPC Site Profiles with Agentic Discovery

    [https://arxiv.org/abs/2610.00971](https://arxiv.org/abs/2610.00971)

    该论文指出工作流可移植性缺失的“另一半”是站点本身的使用方式（资源形态、存储、网络权限与运行策略），并提出通过智能体自动发现、构建基于证据的HPC站点画像，使自动化部署工具能够将站点知识转化为部署决策，免去人工调整。

    

    将一个在高性能计算（HPC）站点上开发和测试的工作流迁移到另一个站点时，很少能够在没有反复试错的情况下顺利成功。包管理器可以重建软件环境，容器可以打包整个文件系统，而诸如backpacks之类的工作流规范则将工作流与其软件、数据和资源需求一同封装。这些方法解决了工作流可移植性的一半问题：即工作流需要什么。但没有一种方法描述了给定HPC站点应如何被使用，而这缺失的另一半正是即使是可移植的工作流，在每个新站点上仍需人工调整的原因。这一差距涵盖了站点的资源形态、存储配置、网络权限以及运行策略。这些信息可能显式地存在于批处理系统中，隐藏在文档的文字描述里，或深埋于路由器的配置之中，这使得自动化部署工具难以将站点知识转化为有效的部署决策。我们提出了HPC站点画像（此处摘要被截断）……

    arXiv:2610.00971v1 Announce Type: cross  Abstract: Moving a workflow developed and tested at one HPC site to another rarely succeeds without some amount of trial and error. Package managers rebuild software environments, containers ship whole filesystems, and workflow specifications such as backpacks package a workflow with its software, data, and resource requirements. These approaches address one half of workflow portability: what a workflow needs. But none describes how a given HPC site must be used, and that missing half is why even a portable workflow requires manual adjustment at each new site. That gap includes the site's resource shape, storage configuration, network permissions, and operating policies. This information may be explicit in the batch system, hidden in the prose of documentation, or buried deep within a router's configuration, making it difficult for an automated deployment tool to turn site knowledge into useful deployment decisions. We propose the HPC site profi
    
[^13]: RPTune：面向LLM商品目录搜索的学习式上下文整理方法

    RPTune: Learned Context Curation for LLM Catalog Search

    [https://arxiv.org/abs/2610.00964](https://arxiv.org/abs/2610.00964)

    RPTune提出一个端到端框架，通过“学习式目录整理（商品排序与剪枝）”和“基于自动生成目录监督的LLM后训练”相互促进，为中小商户的上下文内商品目录搜索提供了优于多阶段检索的新方案。

    

    对于商品目录能够完整容纳于长上下文LLM的中小商户（SMB）而言，全目录提示为多阶段检索提供了一种颇具吸引力的替代方案——后者主要是为拥有数百万商品的大型电商平台设计的。然而，将完整目录放入上下文窗口并不能确保模型能够有效利用它，因为LLM对长上下文的利用并不均匀。因此，我们从两个互补的问题入手研究上下文内目录搜索：（1）如何为LLM整理和呈现商品目录，以及（2）如何对LLM进行适配，使其能在整理后的上下文中进行商品选择。我们提出了RPTune，一个端到端框架，它将学习式目录整理与基于自动生成的、以目录为依据的监督信号的LLM后训练相结合。一个编码器-重组器结构的整理器在下游LLM反馈的引导下对商品进行排序和剪枝，而由此产生的整理后目录反过来又提升了LLM后训练的有效性。

    arXiv:2610.00964v1 Announce Type: new  Abstract: For small merchant businesses (SMBs) whose catalogs fit within a long-context LLM, full-catalog prompting offers a compelling alternative to multi-stage retrieval designed primarily for large marketplaces with millions of items. However, fitting the full catalog into the context window does not ensure that the model can use it effectively, since LLMs do not exploit long contexts uniformly. We therefore study in-context catalog search through two complementary questions: (1) how to curate and present catalogs to the LLM, and (2) how to adapt the LLM for product selection on curated contexts.   We propose RPTune, an end-to-end framework that couples learned catalog curation with LLM post-training using automatically generated, catalog-grounded supervision. An encoder-reorganizer curator orders and prunes products guided by downstream LLM feedback, while the resulting curated catalogs in turn improve the effectiveness of LLM post-training w
    
[^14]: CANOPY：面向多模态RAG的自适应粒度证据压缩

    CANOPY: Adaptive-Granularity Evidence Compression for Multimodal RAG

    [https://arxiv.org/abs/2610.00923](https://arxiv.org/abs/2610.00923)

    提出CANOPY框架，通过将检索条目表示为层次结构并利用父级相对细化机制，在多模态RAG中实现无需LLM调用的自适应粒度证据压缩。

    

    多模态RAG会检索文本、表格、图像和视频，但选择检索粒度并不能决定在每个条目内应保留多少上下文。粗粒度单元会包含无关内容，而统一采用细粒度选择则可能移除解释证据所需的上下文。现有压缩器通过针对特定模态的机制来处理这一权衡，缺乏一种共享的流程来在异构条目间逐区域地自适应调整保留范围。我们提出CANOPY（Canonical Projection over Hierarchy），一个用于自适应粒度检索后证据压缩的框架。CANOPY将检索到的条目表示为层次结构，并使用在黄金证据上微调的节点编码器来根据查询对各个区域进行评分。父级相对细化机制通过比较这些分数，在不同粒度上选择多个区域，而无需为节点级剪枝调用大语言模型。由于压缩无法恢复从未被检索到的证据……

    arXiv:2610.00923v1 Announce Type: new  Abstract: Multimodal RAG retrieves text, tables, images, and videos, but choosing a retrieval granularity does not determine how much context to retain within each item. Coarse units include irrelevant content, while uniformly fine selection can remove context needed to interpret the evidence. Existing compressors address this trade-off with modality-specific mechanisms, leaving open a shared procedure for adapting the retained extent region by region across heterogeneous items. We introduce CANOPY (Canonical Projection over Hierarchy), a framework for adaptive-granularity post-retrieval evidence compression. CANOPY represents retrieved items as hierarchies and uses a node encoder fine-tuned on gold evidence to score regions against the query. Parent-relative refinement compares these scores to select multiple regions at different granularities without LLM calls for node-level pruning. Because compression cannot recover evidence that was never ret
    
[^15]: TabJoinBench：可连接表发现的基准测试

    TabJoinBench: A Benchmark for Joinable Table Discovery

    [https://arxiv.org/abs/2610.00817](https://arxiv.org/abs/2610.00817)

    本文提出TabJoinBench——一个覆盖语义型、关系型和混合型数据湖场景的可连接表发现基准，通过可组合扰动系统地引入结构、表示和语义变化并保持可靠真值，解决了以往方法特定基准难以进行可重现和公平比较的问题。

    

    连接发现旨在从大型数据仓库中识别出能够用互补信息增强查询表的表，从而支持数据探索、特征工程和商业智能等下游任务。尽管已有众多连接发现方法被提出，但现有研究依赖于特定方法的基准构建，使得可重现的公平比较变得困难。我们提出了TabJoinBench，这是一个用于评估连接发现方法的基准测试，涵盖语义型、关系型和混合型数据湖场景。TabJoinBench采用特定于数据源的验证策略构建查询-候选对，并通过可组合的扰动系统地引入结构、表示和语义层面的变化，同时保持可靠的真值标准。我们评估了涵盖基于集合、基于特征和学习型方法的代表性连接发现方法，以及通用语言模型嵌入基线……（摘要截断）

    arXiv:2610.00817v1 Announce Type: cross  Abstract: Join discovery aims to identify tables from large data repositories that can augment a query table with complementary information, enabling downstream tasks such as data exploration, feature engineering, and business intelligence. Although numerous join discovery methods have been proposed, existing studies rely on method-specific benchmark construction, making reproducible and fair comparison difficult. We present TabJoinBench, a benchmark for evaluating join discovery methods across semantic, relational, and hybrid data lake scenarios. TabJoinBench constructs query-candidate pairs using source-specific validation strategies, systematically introduces structural, representation, and semantic changes through composable perturbations while preserving reliable ground truth. We evaluate representative join discovery methods spanning set-based, feature-based, and learned approaches, together with general-purpose language-model embedding ba
    
[^16]: 企业表示简化（ERS）：降低企业AI的表示复杂性

    Enterprise Representation Simplification (ERS): Reducing Representational Complexity for Enterprise AI

    [https://arxiv.org/abs/2610.00791](https://arxiv.org/abs/2610.00791)

    本文提出了企业表示简化（ERS）理念和企业表示复杂性（ERC）模型，后者通过对象、交互、行为和支持源四个维度以表示中立的方式量化比较企业表示的复杂性，从而为降低企业AI所面临的表示复杂性提供了框架。

    

    企业信息通过由应用程序、项目、技术、组织边界和本地需求所塑造的工件来表示。这些结构随时间不断累积，形成了必须由企业维护、并由信息消费者和AI系统解释的表示复杂性。本文提出了企业表示简化（ERS）的概念，即在定义的范围内减少不必要的表示复杂性，同时保留所需的信息；并提出了企业表示复杂性（ERC），这是一个表示中立的模型，用于跨不同表示状态比较复杂性。ERC通过四个维度来表征表示范围：表示对象、交互、行为和支持源。其中，对象、交互和行为构成相互依赖的类别，而支持源则表征表示的暴露程度。ERC被定义在表示（摘要截断于此）

    arXiv:2610.00791v1 Announce Type: new  Abstract: Enterprise information is represented through artifacts shaped by applications, projects, technologies, organizational boundaries, and local requirements. These structures accumulate over time, creating representational complexity that must be maintained by the enterprise and interpreted by information consumers and AI systems. This paper introduces Enterprise Representation Simplification (ERS) as reducing unnecessary representational complexity while preserving required information within a defined scope, and Enterprise Representation Complexity (ERC), a representation-neutral model for comparing complexity across representation states.   ERC characterizes representational extent through four dimensions: Representation Objects, Interactions, Behaviors, and Supporting Sources. Objects, Interactions, and Behaviors form dependent categories, while Supporting Sources characterize representation exposure. ERC is defined at representation an
    
[^17]: Common Crawl News 与 GDELT 的比较

    Comparison of Common Crawl News & GDELT

    [https://arxiv.org/abs/2610.00587](https://arxiv.org/abs/2610.00587)

    本文比较了 GDELT 和 Common Crawl News 两个新闻数据集，分析了它们各自的优势与局限，并揭示了两者在新闻来源获取渠道上的显著差异。

    

    全球新闻语料库对自然语言处理、知识图谱、大语言模型以及其他技术工作具有重要意义。此外，该语料库对于理解每天实时交互的人、地点、组织和事件也至关重要。本文比较了目前用于这些任务的两个新闻数据集，即全球事件、语言和语调数据库（GDELT）和 Common Crawl News。我们的研究重点分析了每个数据集的优势与局限性，并对它们的内容和覆盖范围进行了分析。值得注意的是，GDELT 依赖来自全球各地的广播、印刷媒体和网络新闻，而 Common Crawl 则专注于通过网络爬虫收集的来自世界各地的新闻网站。我们的分析显示，这两个数据集在新闻来源的获取渠道上存在显著差异。

    arXiv:2610.00587v1 Announce Type: new  Abstract: The corpus of worldwide news is important for natural language processing, knowledge graphs, large language models, and other technical efforts. Additionally, this corpus is important for understanding the people, places, organizations, and events that interact in real-time every day. This paper compares two news datasets used for these tasks today, namely the Global Database of Events, Language, and Tone (GDELT) and Common Crawl News. Our research highlights the strengths and limitations of each dataset, analyzing their content and coverage. Notably, while GDELT relies on broadcasts, prints, and web news from across the globe, Common Crawl focuses on news sites from around the world gathered through web crawling. Our analysis revealed considerable differences in where the two datasets gather their news sources.
    
[^18]: 对模型撰写文本的共同偏好：“AI-AI偏见”的生成器×选择器矩阵未显示可检测的自家模型溢价

    A Shared Taste for Model-Written Text: The Generator-by-Selector Matrices of "AI-AI Bias" Show No Detectable Own-Model Premium

    [https://arxiv.org/abs/2610.00369](https://arxiv.org/abs/2610.00369)

    本研究重建了“AI-AI偏见”实验的三个5×5生成器×选择器矩阵并进行精确置换检验，发现语言模型选择器并不存在对自家模型生成文本的额外偏好，“自家模型溢价”在统计上不可检测，表明各模型对模型撰写文本的偏好是共享的而非自我偏爱。

    

    Laurito等人（PNAS 2025）的研究表明，大型语言模型在同一产品、论文或电影的两段描述中进行选择时，比起人类撰写的描述，更偏好语言模型撰写的描述，且这种偏好程度远超人类评审。他们的设计将五个生成器与相同的五个模型作为选择器进行交叉，由此可以提出一个该论文未在标题中强调的第二个问题：选择器是否在生成器和选择器主效应所能预测之外，还额外偏好来自其自身模型的文本？我们根据作者公开仓库中的逐条目计数重建了三个5×5矩阵（21,828个有效试验；每个单元格均与已发表的数值一致），并拟合了一个包含自身模型项gamma的双向固定效应模型，通过对选择器的120种重新标记进行精确置换检验来检验该项。结果显示：产品上的溢价为+0.013（精确单侧p = 0.24），论文摘要上为-0.010（p = 0.74），电影上为+0.054（p = 0.07），合并后为+0.019（原文摘要在此处被截断）。

    arXiv:2610.00369v1 Announce Type: cross  Abstract: Laurito et al. (PNAS 2025) showed that large language models choosing between two descriptions of the same product, paper or film prefer the description written by a language model over the one written by a person, by a wide margin over what human judges do. Their design crosses five generators with the same five models as selectors, which permits a second question the paper does not headline: does a selector prefer text from its own model beyond what the generator and selector main effects predict? We rebuild the three 5x5 matrices from the per-item counts in the authors' public repository (21,828 valid trials; every cell matches the published value) and fit a two-way fixed-effects model with an own-model term gamma, tested by the exact permutation test over the 120 relabellings of the selectors. The premium is +0.013 on products (exact one-sided p = 0.24), -0.010 on paper abstracts (p = 0.74), +0.054 on films (p = 0.07) and +0.019 po
    
[^19]: 大语言模型品牌推荐中的系统归因：单条回复可识别系统，聚合品牌画像无法迁移

    System Attribution in LLM Brand Recommendations: Single Responses Identify the System, Aggregated Brand Profiles Do Not Transfer

    [https://arxiv.org/abs/2610.00253](https://arxiv.org/abs/2610.00253)

    该研究发现，单条回复可通过简单的字符n-gram分类器以97.84%的准确率识别其来源的大语言模型系统，但由品牌推荐汇总而成的聚合品牌画像无法在不同系统间迁移使用。

    

    对AI可见度的审计通常将已部署语言模型的品牌推荐汇总为针对每个系统的画像。我们检验这样的画像是否能够描述该系统：测试语料包含6,475条存储回复（其中6,324条可分析），收集于2025年12月至2026年2月，来自五个已部署端点，涵盖礼品推荐、企业声誉和品类归属三类查询。收集工具截断了大量答案：在1,024个token输出上限下，Gemini 3 Flash在品类归属查询中有83.1%的答案在句子中间被截断。在将每条回复截取为前800个字符后，一个按提示词属性进行交叉验证的字符n-gram分类器能够以97.84%的准确率将单条回复归因于GPT-5.2、Gemini 3 Flash、开启搜索的Gemini 3 Flash、Grok或Perplexity sonar-pro（5,028条回复，383个提示词，多数类占比31.5%，30个数据划分随机种子）。仅依赖文本长度会降至多数类水平，24个格式统计特征可达95.79%，而屏蔽品牌名称和...（原文摘要在此处截断）

    arXiv:2610.00253v1 Announce Type: new  Abstract: Audits of AI visibility summarise the brand recommendations of deployed language models into per-system profiles. We test whether such a profile describes the system on one corpus of 6,475 stored responses (6,324 analysable) collected between December 2025 and February 2026 from five deployed endpoints across gift-recommendation, corporate-reputation and category-ownership queries. The collection harness cut many answers short: 83.1% of Gemini 3 Flash answers in category ownership end mid-sentence under a 1,024-token output cap. With every answer cut to its first 800 characters, a character n-gram classifier cross-validated by prompt attributes one response to GPT-5.2, Gemini 3 Flash, Gemini 3 Flash with search, Grok or Perplexity sonar-pro with 97.84% accuracy (5,028 responses, 383 prompts, majority class 31.5%, 30 split seeds). Length alone falls to the majority rate, 24 formatting statistics reach 95.79%, and masking brand names and c
    
[^20]: 大小、延迟与隐私约束下的端侧商业意图检索：一个具有类型化出站边界的3 MiB检索系统

    On-Device Commercial Intent Retrieval Under Size, Latency, and Privacy Constraints: A 3 MiB Retrieval System with Typed Egress Boundaries

    [https://arxiv.org/abs/2610.00170](https://arxiv.org/abs/2610.00170)

    该论文在3 MiB载荷、20ms p95延迟与零数据外泄三重约束下，构建了一个完全端侧运行的商业意图检索系统，通过从韩语句子变换器蒸馏并4比特量化的静态嵌入表，在6020叶节点的商业分类上取得75.0%的中类top-5准确率，并揭示了准确率在“查询是否包含叶节点名称子串”上的剧烈分化（83.5% 对 45.2%）。

    

    我们研究完全在用户设备上运行的商业意图推断，其受到三项在工作开始前即已冻结的约束：下载载荷小于3 MiB、Tier-0推断在p95下低于20毫秒、且不向设备外发送任何原始文本、内容嵌入或稳定标识符。在这些约束下，我们在一个拥有6,020个叶节点的商业分类体系上构建了检索路径：一个从韩语句子变换器蒸馏而来、量化为4比特、无需推断运行时的静态嵌入表。我们的主要结果在于揭示该约束在何处付出精度代价：在由他人标注的真实韩语商业文本（22,900条AI-Hub购物评论）上，真实产品名称的中类top-5准确率为75.0%（随机排列基线为18.4%），但结果在一个可观测特征上出现分裂：包含某叶节点名称作为子串的查询达到83.5%，不含的仅为45.2%。一个通用的大196.6倍的教师模型似乎能定位这一差距（无锚点时+20.1个百分点，有锚点时+0.1个百分点）。该零结果（原文摘要在此处截断）……

    arXiv:2610.00170v1 Announce Type: new  Abstract: We study commercial intent inference that runs entirely on the user's device, under three constraints frozen before the work began: the downloaded payload under 3 MiB, Tier-0 inference under 20 ms at p95, and no raw text, content embedding, or stable identifier leaving the device. Under them we build a retrieval path over a 6,020-leaf commercial taxonomy: a static embedding table distilled from a Korean sentence transformer, quantized to 4 bits, no inference runtime.   Our main result is where that constraint costs accuracy. On real Korean commerce text labelled by others (22,900 AI-Hub shopping reviews), mid-category top-5 on real product names is 75.0% against an 18.4% permutation baseline, but splits on one observable: a query containing some leaf name as a substring scores 83.5%, one containing none 45.2%. A generic 196.6x larger teacher seemed to localize the gap (+20.1 pp without an anchor, +0.1 with). That null was two effects can
    
[^21]: 向语言模型询问彩票号码：重复“49选6”输出中的集中现象

    Ask a Language Model for Lottery Numbers: Concentration in Repeated Six-of-49 Outputs

    [https://arxiv.org/abs/2610.00052](https://arxiv.org/abs/2610.00052)

    语言模型在生成“49选6”随机彩票号码时输出高度集中，其多样性远低于真正的均匀随机抽样水平。

    

    我们在“从1-49中选取六个不同随机整数”的请求上评估了六种语言模型配置。在四组英文提示变体下共1,200次调用尝试中，有1,184次响应产生了有效彩票。号码频率的有效多样性介于9.9到18.0之间，而在相应样本量下，独立均匀“49选6”抽样的模拟第五百分位阈值为46.6-46.7。各系统产生了8至93种不同的无序彩票组合，其众数彩票占有效响应的22.5%至68.0%。两个存档的波兰乐透（Polish Lotto）抽样数据提供了实体彩票的对比基准，在更小的样本量下其有效多样性分别为41.1和41.4。这些结果表明，在所测试的部署设置下，语言模型的输出存在显著的集中现象。但本研究并未识别造成该现象的机制，也未考察其他提示、温度或工具配置下的性能表现。

    arXiv:2610.00052v1 Announce Type: cross  Abstract: We evaluate six language-model configurations on requests for six distinct random integers from 1-49. Across 1,200 attempted calls using four English prompt variants, 1,184 responses yielded valid tickets. Effective diversity of number frequencies ranged from 9.9 to 18.0, compared with simulated fifth-percentile thresholds of 46.6-46.7 under independent uniform six-of-49 sampling at the corresponding sample sizes. Systems produced 8-93 distinct unordered tickets, and their modal tickets accounted for 22.5-68.0% of valid responses. Two archived Polish Lotto samples provided a physical-lottery comparison, with effective diversities of 41.1 and 41.4 at smaller sample sizes. These results demonstrate substantial concentration under the tested deployment settings. They do not identify its mechanism or establish performance under other prompts, temperatures, or tool configurations.
    
[^22]: 探索基于生成式建模的论坛帖子检索

    Exploring Forum Post Retrieval with Generative Modeling

    [https://arxiv.org/abs/2609.38646](https://arxiv.org/abs/2609.38646)

    论文针对新界面 Facebook Forum 交互数据稀疏的问题，通过在 Facebook 群组互动数据上训练并复用从跨平台 Feed 数据学到的分层前缀式语义 ID，微调 30 亿参数指令语言模型直接从用户上下文生成语义 ID，实现了生成式的论坛帖子检索推荐。

    

    生成式推荐（GR）建立在生成式模型在语言和视觉领域成功的基础上，已成为基于嵌入的检索的一种替代方案。我们正在 Facebook Forum（一个面向 Facebook 群组中重度用户的独立应用）上探索 GR。由于 Forum 是一个全新的界面，其自身的交互数据过于稀疏，无法从零开始训练 GR 模型。我们通过两个维度的迁移来解决这一问题：一方面，我们在更广泛的 Facebook 群组互动语料库上训练，而不仅仅是 Forum 会话数据；另一方面，我们复用从跨平台 Facebook Feed 数据中学习到的分层、基于前缀的语义 ID（SID），而不是为 Forum 单独训练分词器。随后，我们对一个 30 亿参数的指令微调语言模型进行监督微调，使其能够直接从用户上下文生成 SID。我们系统地消融了实践中最关键的设计选择，包括 SID 的构建方式、用户历史记录的组成与长度等。

    arXiv:2609.38646v1 Announce Type: new  Abstract: Generative recommendation (GR) has emerged as an alternative to embedding-based retrieval, building on the success of generative models in language and vision. We are exploring GR on Facebook Forum, a standalone application for medium-to-heavy users of Facebook Groups. Because Forum is a new surface, its own interaction data are too sparse to train a GR model from scratch. We address this with transfer along two axes: we train on a broader corpus of Facebook Groups engagements rather than Forum sessions alone, and we reuse hierarchical, prefix-based semantic IDs (SIDs) learned from cross-platform Facebook Feed data instead of fitting a Forum-specific tokenizer. A 3B-parameter instruction-tuned language model is then supervised-fine-tuned to generate SIDs directly from user context. We systematically ablate the design choices that matter most in practice, including SID construction, the composition and length of user history, and the incl
    
[^23]: TAGGRAPH：用于智能体持久历史的图检索之标签增强图

    TAGGRAPH: Tag-Augmented Graphs for Graph Retrieval of Agent Persistent Histories

    [https://arxiv.org/abs/2609.38353](https://arxiv.org/abs/2609.38353)

    本文提出基于共享5W式对话记忆的受控评估框架，系统比较了标签增强图配置、AdaptiveGraph、BM25与OpenClaw在LLM智能体长期记忆检索上的表现，发现最优检索方法随基准和测试规模而变化，没有单一方法在所有设置下都占优。

    

    长期记忆使LLM智能体能够回忆过去的交互并在多个会话之间保持一致性，但由于记忆系统在表示、索引、检索和评估方面往往各不相同，因此难以相互比较。我们提出了一个基于共享的5W式对话记忆的受控评估框架。局部化的图配置在共同的基础图上进行遍历；AdaptiveGraph增加了时序边并使用个性化PageRank进行扩散。我们还在相同的抽取笔记上评估了BM25，并将OpenClaw作为使用原始输入的外部参照。检索排序在不同的记忆设置之间存在差异。在LongMemEval-S上，AdaptiveGraph是最强的图配置，MRR达到0.844，但BM25达到0.867，OpenClaw达到0.880。在ATANT Core上，局部化图遍历优于扩散和BM25，而在压力测试轮次中BM25领先。在测试范围内缩减LongMemEval-S的规模并未重现ATANT中的扩散惩罚，但最小的测试……（原文在此处截断）

    arXiv:2609.38353v1 Announce Type: cross  Abstract: Long-term memory lets LLM agents recall past interactions and remain consistent across sessions, but memory systems are hard to compare because they often vary in representation, indexing, retrieval, and evaluation. We present a controlled evaluation framework based on shared 5W-style conversational memories. Localized graph configurations traverse a common base graph; AdaptiveGraph adds chronological edges and Personalized PageRank diffusion. We also evaluate BM25 over the same extracted notes and OpenClaw as a raw-input external reference. Retrieval rankings vary across memory settings. On LongMemEval-S, AdaptiveGraph is the strongest graph configuration at 0.844 MRR, but BM25 reaches 0.867 and OpenClaw 0.880. On ATANT Core, localized graph traversal outperforms diffusion and BM25, whereas BM25 leads the stress rounds. Reducing LongMemEval-S within the tested range does not reproduce the ATANT diffusion penalty, but the smallest test
    
[^24]: 路由剩余证据：面向推荐系统中缺失模态候选重排序的元模态智能体

    Route What Remains: A Meta-Modal Agent for Missing-Modality Candidate Reranking in Recommender Systems

    [https://arxiv.org/abs/2605.25007](https://arxiv.org/abs/2605.25007)

    提出元模态智能体（MMA），将缺失模态推荐中的候选重排序形式化为预算约束下的顺序证据获取问题，通过PPO学习自适应调用文本、图像和交互图工具进行稀疏重打分，在全部九组数据集与路径组合中均取得最优结果，NDCG@10最高提升10.0%。

    

    缺失模态推荐系统通常通过重建缺失的表示来处理问题，但已有的观测证据可能并不足以确定缺失的内容。我们将候选重排序形式化为预算约束下的顺序证据获取问题。策略会查询文本、图像和交互图等工具，将空返回纳入观测历史，并对检索到的候选池进行稀疏重打分。我们的元模态智能体（MMA）使用PPO来优化最终的NDCG和工具成本，且无需显式获取路由可用性掩码或目标身份信息。当只有一条证据路径可用时，MMA-Auto相比最强的补全基线将NDCG@10提升了10.0%，相比使用相同Llama打分器的固定路由器提升了9.5%。在所有九个已报告的数据集与可用路径组合中，MMA-Auto均取得了最佳结果。此外，MMA-Auto还将失败调用减少了17.8个百分点，并平均减少1.1轮交互。

    arXiv:2605.25007v2 Announce Type: replace  Abstract: Missing-modality recommenders usually reconstruct absent representations, although the observed evidence may not determine the missing content. We formulate candidate reranking as budgeted sequential evidence acquisition. A policy queries text, image, and interaction-graph tools, incorporates \texttt{Null} returns into its observation history, and sparsely rescores a retrieved candidate pool. Our \textbf{Meta-Modal Agent} (MMA) uses PPO to optimize terminal NDCG and tool cost without explicit access to the route-availability mask or target identity. When only one evidence route is available, MMA-Auto improves NDCG@10 by $10.0$\% over the strongest completion baseline and by $9.5$\% over a fixed router with the same Llama scorer. It obtains the highest result in all nine reported combinations of dataset and available route against these comparators. MMA-Auto also reduces failed calls by 17.8 percentage points and uses 1.1 fewer turns 
    

