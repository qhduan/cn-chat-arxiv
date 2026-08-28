# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [misi: a Metric Inverted Sample Index](https://arxiv.org/abs/2608.27422) | 本文提出了一种通用度量空间上的倒排索引misi，通过线性大小的随机样本词汇表和idf加权共享邻居投票，实现了可扩展的近似最近邻搜索，并提供了理论上的召回保证。 |
| [^2] | [Scaling Graph Neural Networks for Friend Recommendation: Multi-Hash User Embeddings and Temporal Neighbor Sampling](https://arxiv.org/abs/2608.27413) | 本文提出了一种生产级社交图上的可扩展GNN好友推荐系统，通过多哈希嵌入将ID表大小缩减98%以上，并结合时间邻居采样提升排序质量。 |
| [^3] | [RATIO: A Benchmark for Retrieval Across Typed Ideation Operations in Scientific Literature](https://arxiv.org/abs/2608.27394) | RATIO基准首次定义了三种科学构思操作（Address、Broaden、Specify）的检索任务，并利用远距离监督扩展到大规模语料库，为科学文献的灵感检索提供了新范式。 |
| [^4] | [CorporateBench: Large-Scale Q&A Benchmarking with Temporal Knowledge Bases](https://arxiv.org/abs/2608.27391) | 企业基准（CB）是一个大规模、人工验证的多任务问答基准，通过超过23万份文档的时间知识库评估LLMs，揭示了在现实规模下性能显著下降的问题。 |
| [^5] | [Stageboost: Recommending Signals Based on Counterfactual Estimation](https://arxiv.org/abs/2608.27366) | 本文提出了一种两阶段XGBoost模型，通过反事实估计优化eBay商品页面的信号推荐，显著提升了高均价商品的转化率和整体购买额。 |
| [^6] | [Astar: Learning to Propose Evolution Directions for Self-Evolving Industrial AI Systems](https://arxiv.org/abs/2608.27287) | 本文提出Astar方法，通过从工业AI系统的迭代历史中训练专用模型，自动提出进化方向，以解决通用大语言模型在自进化系统中建议泛泛且不匹配的问题。 |
| [^7] | [ProRetrieval: Learning to Orchestrate Hybrid Search via Executable Program Synthesis](https://arxiv.org/abs/2608.27017) | ProRetrieval通过让语言模型合成混合DSL程序，将SQL操作与向量检索结合，实现灵活编排异构检索路径，优于固定组合的现有方法。 |
| [^8] | [Conversational Recommendation over Live E-Commerce Catalogues with Self-Refreshing Retrieval](https://arxiv.org/abs/2608.27006) | 本文提出一个基于自刷新检索的实时电商对话推荐系统，通过增量同步和专用功能处理动态目录，实现高效的多轮购物助手。 |
| [^9] | [Topology-Masked Unified Backbone for Joint Feature Interaction and Multi-Domain Sequence Modeling](https://arxiv.org/abs/2608.27005) | 本文提出MaskRec，一种拓扑掩码统一令牌交互架构，通过将异构特征和多域序列统一为令牌表示并引入记忆令牌，解决了CVR预测中特征交互与序列建模的分离问题。 |
| [^10] | [When Memory Takes Gradients: Collaborative Vector Memory for Agentic Recommender Systems](https://arxiv.org/abs/2608.26895) | 我们提出CoVeMem，通过将智能体推荐系统的记忆从文本转换为向量化协作核心，利用冻结的LightGCN状态和软标记检索，实现高效且协作感知的决策推理。 |
| [^11] | [Equal Ranking Quality, Different Decisions: Training Order-Consistent LLM Scorers](https://arxiv.org/abs/2608.26762) | 本文发现LLM评分器在排序质量相同时决策不一致，提出OC-SFT方法通过训练分数顺序无关性来保持排序质量并提升决策稳定性。 |
| [^12] | [STREAM: An Objective-Driven and Uncertainty-Aware Framework for Industrial Energy Data Acquisition](https://arxiv.org/abs/2608.26754) | 本文提出STREAM框架，通过目标驱动和不确定性感知的六阶段流程，确保工业能源数据采集满足能源绩效评估需求，并增强数据可追溯性。 |
| [^13] | [Beyond a Single Story: Meta-Reviewing Sparse and Incomplete User-generated Contents for Recommendation](https://arxiv.org/abs/2608.26728) | 提出了一种基于元评审概念的推荐方法，通过聚合邻居评论中的属性情感信息来有效应对用户生成内容的稀疏和不完整问题，从而提升推荐准确性。 |
| [^14] | [BLANC: Discovering Patent White Space via Changes in Normalized Pointwise Mutual Information Between Multi-View Clusters](https://arxiv.org/abs/2608.26685) | 本文提出BLANC方法，通过多视图神经主题建模和归一化逐点互信息（NPMI）的条件变化（ΔNPMI）自动定量检测专利空白区域，无需人工映射即可识别“全局已知但局部未探索”的技术组合。 |
| [^15] | [When Does Supervised Fine-Tuning Reduce Instruction Sensitivity?](https://arxiv.org/abs/2608.26661) | 监督微调在较小模型（1.7B和4B）中显著降低指令敏感性（降幅达54-71%），但在较大模型（8B）中效果不明显，表明模型规模是影响SFT降低指令敏感性的关键因素。 |
| [^16] | [PailitaoGR: Latent Think-with-Images for Generative Image Retrieval](https://arxiv.org/abs/2608.26658) | 本文提出帕利淘GR，一种通过潜在图像思考机制实现目标聚焦感知和选择性辅助证据利用的生成式图像检索方法，支持无需裁剪的缩放和无需OCR的阅读。 |
| [^17] | [hoBIT: A Profile-Aware Retrieval-Augmented Chatbot for University Academic Advising](https://arxiv.org/abs/2608.26604) | 本文提出proFILL方法，将规则型咨询机器人升级为档案感知的RAG系统，通过按需获取用户档案属性来提升大学学术咨询的准确性和适用性。 |
| [^18] | [Preference Flow Matching with Spectral Factorization for Micro-video Recommendation](https://arxiv.org/abs/2608.26579) | PrismRec通过谱分解将视频帧分离为静态和动态因子，并利用偏好流匹配生成过程，从而更精准地捕捉微视频推荐中的用户偏好。 |
| [^19] | [Case2Flow: Bridging Patient Cases and Guideline Flowcharts through Multimodal Retrieval](https://arxiv.org/abs/2608.26414) | 本文提出了Case2Flow任务和FlowAtlas语料库，用于从医学指南中为患者病例检索相关流程图，并揭示了现有多模态检索方法的系统性缺陷。 |
| [^20] | [Assessing the Downstream Utility of Evidence-Aware Retrieval in RAG](https://arxiv.org/abs/2608.26379) | 本研究评估证据感知检索信号在RAG中的下游实用性，发现其虽改变检索排名，但对训练、系统选择和答案质量预测的益处并不一致。 |
| [^21] | [A Reranker for Orchestrating Heterogeneous Speech and Text Retrievers](https://arxiv.org/abs/2608.26194) | 本文提出了一种名为STeReO的重排序器，它通过整合语音和文本检索器来聚合异构模态数据库，并利用自建数据集训练，从而在多模态检索中显著提升证据选择的准确性。 |
| [^22] | [Leveraging Large Language Models for Systematic Literature Review of Disease Spread Models](https://arxiv.org/abs/2608.26150) | 本研究开发了一种利用大型语言模型自动提取疾病传播模型论文信息的流水线，发现其准确率接近人工综述，并揭示LLM间一致性可作为输出质量的指示器。 |
| [^23] | [LLMs for Academic Workflows: An Evaluation of Literature Reviews Generated with Short and Long Context Windows of LLMs](https://arxiv.org/abs/2608.26145) | 本文评估了不同上下文窗口下LLMs生成的文献综述质量，发现它们需要人工监督，且长上下文虽能整合更多信息但加剧了重复和遗漏关键内容的问题。 |
| [^24] | [Agents Don't Paginate: First-Chunk Selection for LLM Tool Responses](https://arxiv.org/abs/2608.26130) | 该论文发现编码智能体从不使用分页获取额外工具响应块，并提出首块选择策略（基于0/1背包）以最大化首块中黄金项的包含率，实验表明现有方法在精度上存在不足。 |
| [^25] | [When Stale Constraints Go Unchecked: Budgeted Verification Failures in Inherited Agent Memory](https://arxiv.org/abs/2608.25553) | 该论文研究了在有限验证预算下，代理继承的过时约束未被检查导致验证失败的问题，并提出了通过重新分配验证槽位来减少这类错误的方法。 |
| [^26] | [DocPC: Document-Level Visual Retrieval via Representative Page Composition](https://arxiv.org/abs/2608.25434) | 本文提出了DocPC框架，通过代表性页面组合将多页文档编码为单一网格图像，大幅降低索引成本，并引入多正样本对比学习与稀疏列表优化，显著提升文档级视觉检索效率与性能。 |
| [^27] | [RetrievalFormer: A Dual-Encoder Transformer for Efficient Approximate Nearest Neighbor Retrieval and Cold-Item Recommendation](https://arxiv.org/abs/2608.24079) | 本文提出了一种双编码器变换器检索框架，能够仅基于特征对新物品进行高效近似最近邻检索，并保持索引对新物品开放，无需重新训练，同时展示了在冷物品推荐上的优越性能。 |
| [^28] | [ExecRubrics: Executable Tool-Augmented Rubrics for Verifiable and Efficient Long-Form Evaluation](https://arxiv.org/abs/2608.22559) | ExecRubrics通过将评分标准转化为可执行的Python函数，实现了可验证、高效且能捕捉复杂依赖关系的长篇评估，替代了昂贵的黑盒LLM评判器。 |
| [^29] | [Empowering Compact LLMs with Fusion of Layer-wise Exits for Recommendation](https://arxiv.org/abs/2608.17316) | 本文提出FLEXRec，一种通过融合紧凑型大语言模型多Transformer层退出点的得分分布来增强其表达能力，同时保持可扩展全语料库排名的判别式推荐框架。 |
| [^30] | [NRCD: An Open Database of Collegiate Running with Unified Performance Standardization](https://arxiv.org/abs/2608.14776) | 本文首次发布了大规模公开的大学跑步数据集NRCD，包含超过12.8万条标准化成绩，覆盖多个项目和广泛的时间范围，并附带详细的赛道和天气元数据，填补了该领域数据可获取性的空白。 |
| [^31] | [REPREC: Representation Driven Parameter-Efficient Recommendation System](https://arxiv.org/abs/2607.24845) | REPREC通过仅训练一个轻量级MLP注入器，将冻结序列编码器的用户表示映射为软令牌来条件化冻结的LLM，实现参数高效的序列推荐，同时保持预训练模型不变。 |
| [^32] | [Strategy-Aware Parameter-Efficient Adaptation for LLM-based Auto-Bidding](https://arxiv.org/abs/2607.24232) | 本文提出SAGE框架，通过参数高效的多模态对齐（包括时间语义位置嵌入和门控交叉注意力），在不进行昂贵微调的情况下，利用大语言模型实现策略感知的高效自动竞价。 |
| [^33] | [Drift-Adaptive ICU Intervention Prediction: Freezing the Physiological Encoder for Auditable Model Updating](https://arxiv.org/abs/2607.19020) | 该论文提出了一种双流架构，在模型更新时冻结生理编码器，仅更新治疗流，从而实现可审计的ICU干预预测，且性能与全参数更新相当。 |
| [^34] | [Planning over Matrix-Factorization MDPs for Candidate Generation](https://arxiv.org/abs/2607.02115) | 该论文提出将前K物品检索建模为基于矩阵分解后验的MDP，通过规划用户状态动态来改进推荐候选生成，并验证了动态感知规划在特定条件下优于静态检索。 |
| [^35] | [SHIFT: Semantic Harmonization via Index-side Feature Transformation for Multilingual Information Retrieval](https://arxiv.org/abs/2606.18801) | SHIFT提出了一种无训练、索引侧的特征变换方法，通过平行翻译对估计并校正语言偏移，有效缓解多语言信息检索中的语言偏见问题。 |
| [^36] | [When Should Queries Be Decomposed? A Stage-Aware Study of Query Decomposition for Multi-Condition Retrieval](https://arxiv.org/abs/2606.08577) | 本文提出一种阶段感知查询分解框架，在初始检索保留整体查询、重排序阶段使用子查询，从而显著提升多条件检索性能。 |
| [^37] | [MIMO: Multilingual Information Retrieval via Monolingual Objectives](https://arxiv.org/abs/2605.31171) | MIMO通过两阶段框架，利用英语教师模型作为锚点，结合知识蒸馏和跨语言对比学习，解决了多语言信息检索中语言聚类和性能下降的问题。 |
| [^38] | [Same Ranking, Different Winner: How Scoring Targets Shape LLM Memory Benchmarks](https://arxiv.org/abs/2605.24060) | 本文揭示了LLM记忆基准测试中评分目标选择的模糊性会显著影响排名结论，并提出TIAP审计方法，无需重跑检索即可评估不同目标对结果的影响。 |
| [^39] | [Category-based and Popularity-guided Video Game Recommendation: A Balance-oriented Framework](https://arxiv.org/abs/2604.14598) | 本文提出CPGRec框架，通过结合准确性驱动、多样性驱动和综合三个模块，利用物品类别和流行度信息，在视频游戏推荐中实现准确性与多样性的平衡。 |
| [^40] | [CPGRec+: A Balance-oriented Framework for Personalized Video Game Recommendations](https://arxiv.org/abs/2604.14586) | 本文提出了CPGRec+框架，通过偏好感知边缘重加权（PER）模块和利用大型语言模型能力，解决游戏推荐中准确性与多样性的权衡，并缓解过平滑问题。 |
| [^41] | [Generate to Accelerate: Improved Reranking via LLM-Generated Pivot Documents](https://arxiv.org/abs/2604.09492) | 本文提出利用大语言模型生成伪相关文档作为枢轴，替代传统依赖现有文档的重排序策略，从而减少计算开销并提升重排序效率。 |
| [^42] | [LLM-Specific Utility for Retrieval-Augmented Generation](https://arxiv.org/abs/2510.11358) | 本文首次形式化并实证了检索增强生成中证据的LLM特定效用，证明其具有模型依赖性和不可转移性，为优化RAG系统提供了新视角。 |
| [^43] | [SustainableQA: A Comprehensive Question Answering Dataset for Corporate Sustainability and EU Taxonomy Reporting](https://arxiv.org/abs/2508.03000) | 本文提出了SustainableQA，一个包含超过19.5万问答对的综合数据集及其可扩展生成流水线，通过自动化评估与精炼机制确保高质量，专门服务于企业可持续性和欧盟分类法报告中的精确数据提取任务。 |
| [^44] | [Refine-POI: Reinforcement Fine-Tuned Large Language Models for Next Point-of-Interest Recommendation](https://arxiv.org/abs/2506.21599) | Refine-POI通过拓扑感知的语义ID生成和强化微调，解决了LLM在POI推荐中的语义连续性和top-k排名不足问题。 |
| [^45] | [Unleashing the Power of LLMs in Dense Retrieval with Query Likelihood Modeling](https://arxiv.org/abs/2504.05216) | 本文提出LLM-QL模型，通过辅助的查询似然最大化任务增强大语言模型的稠密检索能力，利用生成优势改进对比学习。 |

# 详细

[^1]: misi：一种度量倒排样本索引

    misi: a Metric Inverted Sample Index

    [https://arxiv.org/abs/2608.27422](https://arxiv.org/abs/2608.27422)

    本文提出了一种通用度量空间上的倒排索引misi，通过线性大小的随机样本词汇表和idf加权共享邻居投票，实现了可扩展的近似最近邻搜索，并提供了理论上的召回保证。

    

    arXiv:2608.27422v1 公告类型：新 摘要：我们提出了misi，一种用于通用度量空间上近似最近邻搜索的倒排索引，其词汇表是数据库的随机样本，大小与$n$成正比。每个对象由其$k_b$个最近样本点表示，这些样本点通过一个可插拔的内部索引在样本上找到；查询通过一个基于idf权重的共享邻居投票来回答，随后对$C$个候选进行精确验证。该构建将NAPP索引从恒定数量的枢轴推广到线性大小的词汇表，这使发布列表在$n$增长时保持恒定的期望长度$\rho = k_b/\alpha$，并将索引转化为一种组合器：任何在$\alpha n$个点上具有高召回率的索引都能产生在$n$个点上的索引，适用于任何度量空间。一个概率模型给出了召回保证——在重叠间隙上，$k_b$随$n$对数增长即可，并带有索引自身估计的验证预算——以及一个匹配的限制：投票无法解决重叠差异。

    arXiv:2608.27422v1 Announce Type: new  Abstract: We present misi, an inverted index for approximate nearest-neighbor search over general metric spaces whose vocabulary is a random sample of the database, of size proportional to $n$. Each object is represented by its $k_b$ nearest sample points, found by a pluggable inner index over the sample; queries are answered by an idf-weighted shared-neighbor vote followed by exact verification of $C$ candidates. The construction generalizes the NAPP index from a constant number of pivots to a linear-size vocabulary, which keeps posting lists at constant expected length $\rho = k_b/\alpha$ as $n$ grows and turns the index into a combinator: any high-recall index on $\alpha n$ points yields an index on $n$ points, for any metric. A probabilistic model gives a recall guarantee -- $k_b$ logarithmic in $n$ over the overlap gap suffices, with a verification budget the index itself estimates -- and a matching limit: the vote cannot resolve overlap diff
    
[^2]: 扩展图神经网络用于好友推荐：多哈希用户嵌入与时间邻居采样

    Scaling Graph Neural Networks for Friend Recommendation: Multi-Hash User Embeddings and Temporal Neighbor Sampling

    [https://arxiv.org/abs/2608.27413](https://arxiv.org/abs/2608.27413)

    本文提出了一种生产级社交图上的可扩展GNN好友推荐系统，通过多哈希嵌入将ID表大小缩减98%以上，并结合时间邻居采样提升排序质量。

    

    arXiv:2608.27413v1 公告类型：交叉 摘要：好友推荐本质上是图结构化的：潜在连接的相关性取决于多跳社交上下文，而非仅凭用户属性。然而，在拥有数亿用户和数百亿边的生产规模社交图上部署消息传递GNN，需要解决众多建模和系统挑战。我们提出了一个用于生产社交图的可扩展端到端GNN排序系统，重点关注在该场景中至关重要的两个设计选择：多哈希ID嵌入和时间邻居采样。多哈希嵌入常用于高基数特征，但工业GNN系统通常要么忽略可训练ID，要么接受完整的嵌入表，而这对我们的图来说会超过200 GB。我们将多哈希作为主要节点表示，将ID嵌入表大小减少了超过98%，同时保持了排序质量。时间邻居采样i

    arXiv:2608.27413v1 Announce Type: cross  Abstract: Friend recommendation is inherently graph-structured: the relevance of a potential connection depends on multi-hop social context rather than user attributes alone. However, deploying message-passing GNNs on a production-scale social graph with hundreds of millions of users and tens of billions of edges requires addressing numerous modeling and systems challenges. We present a scalable end-to-end GNN ranking system for production social graphs, focusing on two design choices that are critical in this setting: multi-hash ID embeddings and temporal neighbor sampling. Multi-hash embeddings are common for high-cardinality features, but industrial GNN systems typically either ignore trainable IDs or accept full embedding tables, exceeding 200 GB for our graph. We integrate multi-hash as the primary node representation, reducing the ID-embedding table size by more than 98 percent while preserving ranking quality. Temporal neighbor sampling i
    
[^3]: RATIO：科学文献中跨类型构思操作检索的基准

    RATIO: A Benchmark for Retrieval Across Typed Ideation Operations in Scientific Literature

    [https://arxiv.org/abs/2608.27394](https://arxiv.org/abs/2608.27394)

    RATIO基准首次定义了三种科学构思操作（Address、Broaden、Specify）的检索任务，并利用远距离监督扩展到大规模语料库，为科学文献的灵感检索提供了新范式。

    

    arXiv:2608.27394v1 公告类型：新 摘要：检索到的科学文献可以为人与AI科学家提供灵感。灵感可以采取不同形式：先前的工作可能直接建议如何解决问题，或在不同抽象层次上指出方向——放大到更一般的视角或缩小到具体实现。我们引入RATIO（跨类型构思操作检索），这是一个大规模基准，其中相关性由三种操作定义，我们称之为构思动作：Address检索针对所提出问题的潜在方法，Broaden检索更一般的表述，Specify检索具体实例。RATIO是通过一种通用方法从CS文献中数百万篇全文科学论文构建而成，该方法将话语标记远距离监督——先前仅用于分类——扩展到语料库级检索，并结合了广泛的LLM和人工审核。实验表明，操作-

    arXiv:2608.27394v1 Announce Type: new  Abstract: Retrieved scientific literature can serve as inspiration for both human and AI scientists. Inspiration can take different forms: prior work may directly suggest how to address a problem, or surface directions at different levels of abstraction - zooming out to a more general view or zooming in to a concrete realization. We introduce RATIO (Retrieval Across Typed Ideation Operations), a large-scale benchmark in which relevance is defined by three operations which we name ideation moves: Address retrieves potential approaches for stated problems, Broaden retrieves more general formulations, and Specify retrieves concrete instantiations. RATIO is constructed from millions of full-text scientific papers across CS literature via a general recipe that extends discourse-marker distant supervision - previously used only for classification - to corpus-scale retrieval, combined with extensive LLM and human vetting. Experiments show that operation-
    
[^4]: 企业基准：基于时间知识库的大规模问答基准测试

    CorporateBench: Large-Scale Q&A Benchmarking with Temporal Knowledge Bases

    [https://arxiv.org/abs/2608.27391](https://arxiv.org/abs/2608.27391)

    企业基准（CB）是一个大规模、人工验证的多任务问答基准，通过超过23万份文档的时间知识库评估LLMs，揭示了在现实规模下性能显著下降的问题。

    

    arXiv:2608.27391v1 公告类型：新  摘要：大型语言模型（LLMs）越来越能够回答关于企业级文档集合的复杂问题。但评估很困难：公司不愿分享内部通信，而合成数据集往往过于简单。我们提出了企业基准（CB），一个经过人工验证的多任务问答基准，其规模接近LLMs在企业通信网络中遇到的条件，评估语料库超过230,000份文档。CB通过四个合成生成的公司（员工规模从12到10,000不等）评估LLMs在两个维度（信息提取和知识库查询）上的表现。每个语料库都从一个随时间演化的知识库中采样，描述一个一致的世界，确保即使在数十万份文档中也能保证跨文档的逻辑一致性。我们在CB上评估了五个LLMs，结果显示当输入规模接近实际尺度时，性能显著下降。CB为LLM开发者提供了一个...

    arXiv:2608.27391v1 Announce Type: new  Abstract: LLMs are increasingly able to answer complex questions about enterprise-scale document collections. But evaluation is hard: companies don't want to share internal communications, and synthetic datasets have been overly simple. We present CorporateBench (CB), a human-validated multi-task Q&A benchmark whose scale approaches the conditions LLMs encounter in corporate communication networks, with evaluation corpora surpassing 230,000 documents. CB evaluates LLMs across two dimensions (information extraction and knowledge base querying) through four synthetically generated firms ranging from 12 to 10,000 employees. Each corpus is sampled from a temporally evolving knowledge base describing a consistent world, guaranteeing cross-document logical consistency even across hundreds of thousands of documents. We evaluate five LLMs on CB, revealing increasingly poor performance as input size approaches realistic scales. CB provides LLM developers a
    
[^5]: Stageboost：基于反事实估计的信号推荐

    Stageboost: Recommending Signals Based on Counterfactual Estimation

    [https://arxiv.org/abs/2608.27366](https://arxiv.org/abs/2608.27366)

    本文提出了一种两阶段XGBoost模型，通过反事实估计优化eBay商品页面的信号推荐，显著提升了高均价商品的转化率和整体购买额。

    

    arXiv:2608.27366v1 公告类型：交叉 摘要：信号是显示在eBay商品详情（VI）页面上的简短文本或视觉片段，为用户提供关于所查看商品的额外上下文信息。展示这些信号的目的是促进智能购买并激励用户参与。在本文中，我们提出了一种基于两阶段XGBoost的模型，该模型能最优地填充VI页面上的信号。这种方法在整体GMB（总购买商品额）上实现了0.08%的提升，在零部件及配件类GMB上实现了0.58%的增长，这主要归因于在线实验中高平均价格商品转化率的提高。

    arXiv:2608.27366v1 Announce Type: cross  Abstract: Signals are short textual or visual snippets displayed on the eBay View-Item (VI) page, providing additional, contextual information for users about the viewed item. The aim of displaying these signals is to facilitate intelligent purchase and to incentivize engagement. In this paper, we present a 2 stage xgboost based model that optimally populates the VI page with signals. This approach has shown a 0.08% lift in overall GMB (Gross Merchandise Bought) and 0.58% increase in Parts and Accessories GMB, primarily due to increase in conversion of high average price items in online experimentation.
    
[^6]: Astar：学习为自进化工业AI系统提出进化方向

    Astar: Learning to Propose Evolution Directions for Self-Evolving Industrial AI Systems

    [https://arxiv.org/abs/2608.27287](https://arxiv.org/abs/2608.27287)

    本文提出Astar方法，通过从工业AI系统的迭代历史中训练专用模型，自动提出进化方向，以解决通用大语言模型在自进化系统中建议泛泛且不匹配的问题。

    

    arXiv:2608.27287v1 公告类型：新 摘要：现代AI系统通过持续迭代而进步：一个提出进化方向、实现代码、训练和评估的循环。虽然后三个阶段日益自动化，但起点——提出有效的进化方向——仍然是一个关键瓶颈，仍高度依赖资深专家。在这项工作中，我们探索AI是否能接管这一角色。我们发现，通用大语言模型，即使是先进的GPT-5.5，也只提供泛泛且不匹配的建议：所需的专业知识是通过经验积累而非明确编码的，因此难以直接注入。为此，我们提出了Astar，一种基于训练的方法，从工业系统丰富的迭代历史中学习一个专门的进化指导模型。然而，实现这一想法引发了四个挑战：稀疏监督、噪声数据、庞大的方向空间，以及代价高昂的验证。

    arXiv:2608.27287v1 Announce Type: new  Abstract: Modern AI systems advance through continuous iteration: a loop of proposing evolution directions, implementing code, training, and evaluation. While the latter three stages are increasingly automated, the starting point --- proposing effective evolution directions --- remains a critical bottleneck that still relies heavily on senior experts. In this work, we explore whether AI can take over this role. We find that general-purpose LLMs, even the advanced GPT-5.5, offer only generic and misaligned suggestions: the required expertise is accumulated through experience rather than explicitly codified, and thus hard to inject directly.   To this end, we propose Astar, a training-based approach that learns a specialized evolution-guiding model from the abundant iteration histories of industrial systems. Realizing this idea, however, raises four challenges: sparse supervision, noisy data, a vast direction space, and prohibitively expensive verif
    
[^7]: ProRetrieval：通过可执行程序合成学习编排混合检索

    ProRetrieval: Learning to Orchestrate Hybrid Search via Executable Program Synthesis

    [https://arxiv.org/abs/2608.27017](https://arxiv.org/abs/2608.27017)

    ProRetrieval通过让语言模型合成混合DSL程序，将SQL操作与向量检索结合，实现灵活编排异构检索路径，优于固定组合的现有方法。

    

    arXiv:2608.27017v1 公告类型：新 摘要：现实世界的检索常常通过任意布尔逻辑，将结构化约束与文本和图像上的语义意图组合起来。现有的混合流水线（如倒数排名融合或自查询检索器）仅支持固定形式的组合，而最近的强化学习检索器将语言模型训练为单一后端的查询生成器，将异构检索路径的编排排除在其动作空间之外。我们提出ProRetrieval，将语言模型重塑为检索编排器：给定自然语言查询，它合成一个可执行程序，该程序在混合领域特定语言（DSL）中交错使用结构化字段上的SQL操作符与文本和图像上的向量检索原语，其中SQL本身提供融合异构候选集的逻辑代数。我们使用GRPO和DAPO在分层四项奖励下训练Qwen3-4B，并在基于Amazo构建的两个新基准上进行评估。

    arXiv:2608.27017v1 Announce Type: new  Abstract: Real-world retrieval often composes structured constraints with semantic intents over text and images through arbitrary Boolean logic. Existing hybrid pipelines such as reciprocal rank fusion or self-querying retrievers admit only a fixed form of composition, while recent reinforcement-learning retrievers train the language model as a query generator for a single backend, leaving the orchestration of heterogeneous retrieval paths outside its action space. We propose ProRetrieval, which recasts the language model as a retrieval orchestrator: given a natural-language query, it synthesizes an executable program in a hybrid DSL interleaving SQL operators over structured fields with vector-retrieval primitives over text and images, with SQL itself providing the logical algebra that fuses heterogeneous candidate sets. We train Qwen3-4B with GRPO and DAPO under a hierarchical four-term reward, and evaluate on two new benchmarks built from Amazo
    
[^8]: 基于自刷新检索的实时电商目录对话推荐系统

    Conversational Recommendation over Live E-Commerce Catalogues with Self-Refreshing Retrieval

    [https://arxiv.org/abs/2608.27006](https://arxiv.org/abs/2608.27006)

    本文提出一个基于自刷新检索的实时电商对话推荐系统，通过增量同步和专用功能处理动态目录，实现高效的多轮购物助手。

    

    arXiv:2608.27006v1 公告类型：新 摘要：基于大型语言模型（LLMs）的对话推荐系统通常在静态、预索引的物品集合上进行评估，然而电商目录会随着产品的添加、移除、重新定价和补货而持续变化。我们提出一个商家无关的多轮对话购物助手，能够在这些实时目录上运行。其核心组件是一个自刷新检索器，它摄取商家产品馈送，丰富记录，并将它们同步到向量索引中。每次运行时，每项哈希标识哪些产品是新的、变更的、删除的或未变的，因此只处理差异部分，而不是重建整个目录。基于控制器的对话层消费此索引，仅使用LLM进行意图分类和偏好提取，而检索、重排序和多样性选择作为专用功能运行。我们的演示是一个WhatsApp购物助手，其中...

    arXiv:2608.27006v1 Announce Type: new  Abstract: Conversational recommender systems based on large language models (LLMs) are usually evaluated on static, pre-indexed item collections, yet e-commerce catalogues change continuously as products are added or removed, repriced, and restocked. We present a merchant-agnostic, multi-turn conversational shopping assistant that operates over such live catalogues. Its central component is a self-refreshing retriever that ingests a merchant product feed, enriches the records, and synchronizes them into a vector index. On each run, per-item hashes identify which products are new, changed, deleted, or unchanged, so only the delta is processed rather than rebuilding the whole catalogue. A controller-based dialogue layer consumes this index, using an LLM only for intent classification and preference elicitation while retrieval, reranking, and diversity selection run as dedicated functions. Our demonstration is a WhatsApp shopping assistant in which c
    
[^9]: 拓扑掩码统一骨干网络用于联合特征交互与多域序列建模

    Topology-Masked Unified Backbone for Joint Feature Interaction and Multi-Domain Sequence Modeling

    [https://arxiv.org/abs/2608.27005](https://arxiv.org/abs/2608.27005)

    本文提出MaskRec，一种拓扑掩码统一令牌交互架构，通过将异构特征和多域序列统一为令牌表示并引入记忆令牌，解决了CVR预测中特征交互与序列建模的分离问题。

    

    arXiv:2608.27005v1 公告类型：新  摘要：大规模点击后转化率（CVR）预测需要联合建模异构特征交互以及多域用户行为序列上的依赖关系。现有的工业排序模型通常使用独立模块分别处理这两个方面。近期的统一架构尝试将它们整合到单一框架中，但这种统一往往依赖于模块间的协调，并未在同一交互空间内完全组织所有信息源。为解决此问题，我们提出MaskRec，一种用于特征交互和多域序列建模的拓扑掩码统一令牌交互架构。MaskRec将异构特征、多域行为序列和上下文信号转换为统一令牌表示，并进一步引入可学习的全局记忆令牌和域级记忆令牌作为信息聚合节点。基于此统一令牌空间，该架构实现了全面的特征交互与序列建模。

    arXiv:2608.27005v1 Announce Type: new  Abstract: Large-scale post-click conversion rate (CVR) prediction requires jointly modeling heterogeneous feature interactions and dependencies over multi-domain user behavior sequences. Existing industrial ranking models usually handle these two aspects with separate modules. Recent unified architectures attempt to incorporate them into a single framework, but such unification often relies on coordination between modules and does not fully organize all information sources within the same interaction space. To address this problem, we propose MaskRec, a topology-masked unified token interaction architecture for feature interaction and multi-domain sequence modeling. MaskRec transforms heterogeneous features, multi-domain behavior sequences, and contextual signals into unified token representations, and further introduces learnable global memory tokens and domain-level memory tokens as information aggregation nodes. Based on this unified token spac
    
[^10]: 当记忆采用梯度：面向智能体推荐系统的协作向量记忆

    When Memory Takes Gradients: Collaborative Vector Memory for Agentic Recommender Systems

    [https://arxiv.org/abs/2608.26895](https://arxiv.org/abs/2608.26895)

    我们提出CoVeMem，通过将智能体推荐系统的记忆从文本转换为向量化协作核心，利用冻结的LightGCN状态和软标记检索，实现高效且协作感知的决策推理。

    

    arXiv:2608.26895v1 公告类型：交叉 公告摘要：智能体推荐系统将大型语言模型（LLM）的每个决策基于用户的持久记忆，而在现有智能体中，这种记忆是文本形式：由进一步的LLM调用编写和维护的叙述。文本以两种方式限制了这种记忆。它一次只能更新一次重写，因此利用完整的交互历史代价高昂；而且协作证据，即对整个目录的梯度相似性，无法在转化为句子时保留。我们提出了CoVeMem（协作向量记忆），它将智能体记忆的协作核心向量化。冻结的LightGCN用户和物品状态构成记忆库；在每个决策时，候选集本身检索最相关的历史状态，这些状态作为软标记连同轻量级文本简介进入LLM的上下文。通过对比对齐到物品语义锚点，随后与掩码候选进行列表式协同训练，教会模型进行推理。

    arXiv:2608.26895v1 Announce Type: cross  Abstract: Agentic recommender systems ground each decision of a large language model (LLM) in a persistent memory of the user, and in existing agents that memory is text: a narrative written and maintained by further LLM calls. Text limits this memory in two ways. It is updated one rewrite at a time, so exploiting the full interaction history is prohibitively expensive; and collaborative evidence, graded similarity over an entire catalog, does not survive translation into sentences. We propose CoVeMem (Collaborative Vector Memory), which vectorizes the collaborative core of the agent's memory. Frozen LightGCN user and item states form the memory bank; at each decision, the candidate set itself retrieves the most relevant historical states, which enter the LLM's context as soft tokens alongside a light textual profile. Contrastive alignment to item-semantic anchors, followed by listwise co-training with masked candidates, teaches the model to rea
    
[^11]: 同等排序质量，不同决策：训练顺序一致的LLM评分器

    Equal Ranking Quality, Different Decisions: Training Order-Consistent LLM Scorers

    [https://arxiv.org/abs/2608.26762](https://arxiv.org/abs/2608.26762)

    本文发现LLM评分器在排序质量相同时决策不一致，提出OC-SFT方法通过训练分数顺序无关性来保持排序质量并提升决策稳定性。

    

    arXiv:2608.26762v1 公告类型：新 摘要：重排序器、奖励模型和多文档问答评分器在一个LLM提示中为候选文档或响应打分，因此每个分数依赖于它们的顺序。这类评分器基于排序质量进行选择，但它们的分数决定了一个决策：分数阈值保留的内容、读者回答的内容或偏好模型选择的内容。然而，同等排序质量并不意味着同等决策：在段落重排序中，五个训练评分器在nDCG@10上相差0.010以内，但重新排序后保留集仅重叠0.66-0.84。一个已发表的重排序器在我们比较中取得最高保留集F1，但重叠率仍仅为0.667。我们测试的任何提示时间变化都无法消除这种顺序依赖性：唯一提升排序质量的改变使三个决策均未变化。顺序一致性SFT（OC-SFT）在权重中削弱了这种依赖性，训练候选分数不依赖于顺序。它保持了排序质量，并使训练评分器之间的所有决策稳定性指标均得到提升。

    arXiv:2608.26762v1 Announce Type: new  Abstract: Rerankers, reward models and multi-document QA scorers score candidate documents or responses in one LLM prompt, so each score depends on their order. Such scorers are selected on ranking quality, but their scores determine a decision: what a score threshold retains, a reader answers, or a preference model selects. However, equal ranking quality does not imply equal decisions: on passage reranking, five trained scorers within 0.010 nDCG@10 retain sets that overlap by only 0.66-0.84 when reordered. A published reranker takes the highest retained-set F1 in our comparison and still overlaps by only 0.667. No prompt-time change we test removes that order dependence: the only one that gains ranking quality leaves all three decisions unchanged. Order-consistency SFT (OC-SFT) attenuates it in the weights, training a candidate's score not to depend on the order. It holds ranking quality and leads every decision-stability measure among trained sc
    
[^12]: STREAM：面向工业能源数据采集的目标驱动与不确定性感知框架

    STREAM: An Objective-Driven and Uncertainty-Aware Framework for Industrial Energy Data Acquisition

    [https://arxiv.org/abs/2608.26754](https://arxiv.org/abs/2608.26754)

    本文提出STREAM框架，通过目标驱动和不确定性感知的六阶段流程，确保工业能源数据采集满足能源绩效评估需求，并增强数据可追溯性。

    

    工业能源管理需要将能源使用与设备状态、生产批次、物料流和工艺条件联系起来的数据集。然而，传统的采集工作流程通常强调连接性和存储，而未验证可访问的信号是否满足既定能源绩效评估的要求。本文提出了STREAM，一个目标驱动且不确定性感知的框架，包括目标规范、技术要求、资源映射、源提取、归档元数据和数据库迁移。STREAM是核心工作流程：目标到数据的可追溯性是其端到端输出，而测量、时间、上下文和处理不确定性在六个阶段中均得到评估。与原始概念性的STREAM序列相比，本文增加了阶段级工件、最小证据门、源适用性规则、元数据模板和更新机制。

    arXiv:2608.26754v1 Announce Type: new  Abstract: Industrial energy management requires datasets that connect energy use with equipment states, production batches, material flows, and process conditions. However, conventional acquisition workflows commonly emphasize connectivity and storage without verifying whether accessible signals satisfy the requirements of a defined energy-performance assessment. This paper presents STREAM, an objective-driven and uncertainty-aware framework comprising Specification of Objectives, Technical Requirements, Resource Mapping, Extraction from Sources, Archival Metadata, and Migration to Database. STREAM is the central workflow: objective-to-data traceability is its end-to-end output, while measurement, temporal, contextual, and processing uncertainty are assessed across all six stages. Compared with the original conceptual STREAM sequence, this paper adds stage-level artifacts, minimum-evidence gates, source-suitability rules, a metadata template, an u
    
[^13]: 超越单一故事：面向稀疏和不完整用户生成内容的元评审推荐方法

    Beyond a Single Story: Meta-Reviewing Sparse and Incomplete User-generated Contents for Recommendation

    [https://arxiv.org/abs/2608.26728](https://arxiv.org/abs/2608.26728)

    提出了一种基于元评审概念的推荐方法，通过聚合邻居评论中的属性情感信息来有效应对用户生成内容的稀疏和不完整问题，从而提升推荐准确性。

    

    数据稀疏性一直是推荐系统中的长期挑战，而对于依赖用户生成内容（UGC）如文本评论的方法来说，这一问题更为严重，因为这类内容虽能捕捉细粒度偏好，但需要用户付出更多努力来产生。因此，UGC表现出（1）缺失评论，即交互缺乏任何评论，和（2）不完整评论，即现有评论仅覆盖相关属性的子集。现有方法往往忽视这些UGC特有的问题，导致准确性下降。受学术同行评审中元评审的启发，我们提出了MOSAIC（Meta-review On Sparse And Incomplete user-generated Content），该方法通过聚合邻居用户评论中的属性-情感证据，为每个目标用户构建一个元评审。多门混合专家（MMoE）架构联合优化评分预测和元评审属性-情感预测，而注意力模块则进一步增强了信息整合能力。

    arXiv:2608.26728v1 Announce Type: new  Abstract: Data sparsity remains a long-standing challenge in recommender systems, and it becomes more severe for methods relying on user-generated content (UGC) such as textual reviews, which capture fine-grained preferences but require more user efforts to produce. As a result, UGC exhibits (1) missing reviews, where interactions lack any review, and (2) incomplete reviews, where available reviews cover only a subset of relevant attributes. Existing approaches often overlook these UGC-specific issues, leading to degraded accuracy. Motivated by meta-review in academic peer review, we propose MOSAIC (Meta-review On Sparse And Incomplete user-generated Content), which constructs a meta-review for each target user by aggregating attribute-sentiment evidence from neighbor users' reviews. A multi-gate mixture-of-experts (MMoE) architecture jointly optimizes rating prediction and meta-review attribute-sentiment prediction, while an attention module pers
    
[^14]: BLANC：通过多视图聚类间归一化逐点互信息的变化发现专利空白区域

    BLANC: Discovering Patent White Space via Changes in Normalized Pointwise Mutual Information Between Multi-View Clusters

    [https://arxiv.org/abs/2608.26685](https://arxiv.org/abs/2608.26685)

    本文提出BLANC方法，通过多视图神经主题建模和归一化逐点互信息（NPMI）的条件变化（ΔNPMI）自动定量检测专利空白区域，无需人工映射即可识别“全局已知但局部未探索”的技术组合。

    

    摘要：识别空白区域——即专利版图中未被探索但具有潜在价值的区域——对于战略性研发规划至关重要，然而现有方法依赖于人工专利映射或应用单视图聚类，且缺乏定量间隙检测。我们提出了BLANC（通过NPMI条件化的空白版图分析），这是一个三阶段流水线，结合了（1）沿三个语义维度（应用/用途、新颖性、创造性步骤）的多视图神经主题建模；（2）使用归一化逐点互信息（NPMI）来量化跨维度聚类关联；（3）条件检测，当语料库按用户指定的关键词过滤时，标记NPMI下降的组合。这种下降通过一个新指标$\Delta$NPMI捕获，该指标识别“全局已建立、局部未探索”的组合。由于空白区域没有真实基准，我们在两个公开的USPTO语料库上评估BLANC——机器学习/人工智能（5,41...（原文截断）

    arXiv:2608.26685v1 Announce Type: new  Abstract: Identifying white space --- the unexplored but potentially valuable regions of a patent landscape --- is essential for strategic R&D planning, yet existing methods rely on manual patent mapping or apply single-view clustering without quantitative gap detection. We propose BLANC (Blank Landscape Analysis through NPMI Conditioning), a three-phase pipeline combining (1) multi-view neural topic modeling along three semantic dimensions (application/use, novelty, inventive step); (2) Normalized Pointwise Mutual Information (NPMI) to quantify cross-dimensional cluster association; and (3) conditional detection that flags combinations whose NPMI drops when the corpus is filtered by a user-specified keyword. The drop is captured by a new metric, $\Delta$NPMI, which identifies combinations "established globally, unexplored locally." Because white space has no ground truth, we evaluate BLANC on two public USPTO corpora --- machine learning/AI (5,41
    
[^15]: 监督微调何时降低指令敏感性？

    When Does Supervised Fine-Tuning Reduce Instruction Sensitivity?

    [https://arxiv.org/abs/2608.26661](https://arxiv.org/abs/2608.26661)

    监督微调在较小模型（1.7B和4B）中显著降低指令敏感性（降幅达54-71%），但在较大模型（8B）中效果不明显，表明模型规模是影响SFT降低指令敏感性的关键因素。

    

    arXiv:2608.26661v1 公告类型：新 摘要：大型语言模型在相同任务指令的不同表述下可能表现出显著的性能差异，但目前尚不清楚常规的任务特定监督微调（SFT）如何改变这种指令敏感性。我们通过评估固定模型检查点在多种改写指令下的表现来研究这一问题，并将指令敏感性定义为任务性能在这些指令下的标准差。我们使用Qwen3模型在1.7B、4B和8B规模上对MS MARCO进行了受控规模分析，并辅以Mistral-7B和Gemma-2-9B的针对性跨家族检查。在SFT之前，指令敏感性随Qwen3模型规模增大而急剧降低。在1.7B和4B规模下，SFT始终能降低训练指令下的敏感性，降幅约为54-71%。在8B规模下，个体敏感性变化在统计上与零无显著差异，但训练指令间的配对对比显示存在显著差异。

    arXiv:2608.26661v1 Announce Type: new  Abstract: Large language models can exhibit substantial performance variation across alternative formulations of the same task instruction, yet it remains unclear how conventional task-specific supervised fine-tuning (SFT) changes this instruction sensitivity. We study this question by evaluating fixed model checkpoints under multiple paraphrased instructions and defining instruction sensitivity as the standard deviation of task performance across them. We conduct a controlled scale analysis with Qwen3 models at 1.7B, 4B, and 8B on MS MARCO, together with targeted cross-family checks using Mistral-7B and Gemma-2-9B. Before SFT, instruction sensitivity decreases sharply with Qwen3 model scale. At 1.7B and 4B, SFT consistently reduces sensitivity across training instructions, with reductions of approximately 54--71%. At 8B, individual sensitivity changes are not statistically distinguishable from zero, but paired contrasts between training instructi
    
[^16]: 帕利淘GR：用于生成式图像检索的潜在图像思考方法

    PailitaoGR: Latent Think-with-Images for Generative Image Retrieval

    [https://arxiv.org/abs/2608.26658](https://arxiv.org/abs/2608.26658)

    本文提出帕利淘GR，一种通过潜在图像思考机制实现目标聚焦感知和选择性辅助证据利用的生成式图像检索方法，支持无需裁剪的缩放和无需OCR的阅读。

    

    arXiv:2608.26658v1 公告类型：交叉 摘要：生成式检索通过直接生成产品语义标识符（SIDs）展现出强大性能。然而，将这一范式扩展到图像搜索并非易事，因为真实世界的查询图像包含多样信息，包括搜索目标、有用的辅助证据以及无关的视觉内容。这要求模型能够识别并聚焦于搜索目标，同时有选择地利用辅助证据。在本文中，我们提出了**帕利淘GR**，一种用于生成式图像检索的*潜在图像思考*方法，它将目标聚焦感知和选择性辅助证据利用内化到生成式检索模型中，实现了*无需裁剪的缩放*和*无需OCR的阅读*。具体来说，我们设计了一种目标聚焦感知机制，用于识别并增强搜索目标的视觉标记，该机制由目标增强器和一个学习式模块组成。

    arXiv:2608.26658v1 Announce Type: cross  Abstract: Generative retrieval has demonstrated strong performance by directly generating product semantic identifiers (SIDs).   Extending this paradigm to image search, however, is nontrivial because real-world query images contain diverse information, including the search target, useful auxiliary evidence, and irrelevant visual content.   This requires the model to identify and focus on the search target while selectively utilizing auxiliary evidence. In this paper, we propose \textbf{PailitaoGR}, a \emph{Latent Think-with-Images} method for generative image retrieval, which internalizes target-focused perception and selective auxiliary-evidence utilization into a the generative retrieval model, enabling \textit{Zooming without Cropping} and \textit{Reading without OCR}. Specifically, we design a target-focused perception mechanism that identifies and enhances visual tokens of the search target, consisting of a target Enhancer and a learning s
    
[^17]: hoBIT：面向大学学术咨询的档案感知检索增强聊天机器人

    hoBIT: A Profile-Aware Retrieval-Augmented Chatbot for University Academic Advising

    [https://arxiv.org/abs/2608.26604](https://arxiv.org/abs/2608.26604)

    本文提出proFILL方法，将规则型咨询机器人升级为档案感知的RAG系统，通过按需获取用户档案属性来提升大学学术咨询的准确性和适用性。

    

    在高校学术咨询中，相同的问题可能因学生的院系、入学批次和学位项目而需要不同的答案，这导致对档案不敏感的检索器可能提供看似合理但不适用的证据。我们提出了proFILL方法，将我们学院当前基于规则的咨询聊天机器人hoBIT转变为档案感知的检索增强生成（RAG）系统。proFILL不需要预先获取完整的用户档案，而是根据查询意图和初始检索到的证据，逐步获取每次查询所需的档案属性，并利用这些属性在档案感知索引上进行条件化检索。大量实验和人类偏好研究表明，proFILL优于多种RAG基线方法，受到目标用户的青睐，并且在开源权重模型上依然有效，适合经济高效的本地部署。

    arXiv:2608.26604v1 Announce Type: cross  Abstract: In university academic advising, identical questions can require different answers depending on a student's department, admission cohort, and degree program, causing profile-blind retrievers to surface plausible but inapplicable evidence. We present proFILL, a method for transforming hoBIT, our college's current rule-based advising chatbot, into a profile-aware retrieval-augmented generation (RAG) system. Rather than requiring a complete user profile upfront, proFILL progressively acquires only the profile attributes needed for each query, guided by both the query intent and the initially retrieved evidence, and uses them to condition retrieval over a profile-aware index. Extensive experiments and a human preference study show that proFILL outperforms diverse RAG baselines, is preferred by target users, and remains effective with open-weight models for cost-effective on-premise deployment.
    
[^18]: 基于谱分解的偏好流匹配用于微视频推荐

    Preference Flow Matching with Spectral Factorization for Micro-video Recommendation

    [https://arxiv.org/abs/2608.26579](https://arxiv.org/abs/2608.26579)

    PrismRec通过谱分解将视频帧分离为静态和动态因子，并利用偏好流匹配生成过程，从而更精准地捕捉微视频推荐中的用户偏好。

    

    微视频推荐旨在从历史交互和多模态视频内容中推断用户偏好，从而识别下一个感兴趣的视频。然而，现有方法将帧序列压缩为单一的整体表示，纠缠了共同塑造用户偏好的稳定视觉语义和动态演变特征。同时，基于扩散和流匹配的推荐器仅以粗略的行为上下文为条件来生成过程，将内部时间结构排除在偏好形成之外。因此，我们提出了PrismRec，一个用于微视频推荐的基于谱分解的偏好流匹配框架。类似于棱镜将白光分散为其组成光谱，PrismRec设计了谱语义分解（SSF），通过先验引导的可学习频率从帧级表示中推导出互补的静态语义和动态因子。

    arXiv:2608.26579v1 Announce Type: new  Abstract: Micro-video recommendation aims to infer user preferences from historical interactions and multimodal video content, thereby identifying the next video of interest. However, prevailing methods compress frame sequences into a single holistic representation, entangling the stable visual semantics and the evolving dynamics that jointly shape user preferences. Meanwhile, diffusion- and flow matching-based recommenders condition their generation process solely on coarse behavioral context, leaving its internal temporal structure outside preference formation. We therefore propose PrismRec, a Preference Flow Matching framework with Spectral Factorization for Micro-video Recommendation. Analogous to a prism that disperses white light into its constituent spectrum, PrismRec devises Spectral Semantic Factorization (SSF) to derive complementary static semantic and dynamic factors from frame-level representations via a prior-guided learnable frequen
    
[^19]: Case2Flow：通过多模态检索连接患者病例与指南流程图

    Case2Flow: Bridging Patient Cases and Guideline Flowcharts through Multimodal Retrieval

    [https://arxiv.org/abs/2608.26414](https://arxiv.org/abs/2608.26414)

    本文提出了Case2Flow任务和FlowAtlas语料库，用于从医学指南中为患者病例检索相关流程图，并揭示了现有多模态检索方法的系统性缺陷。

    

    arXiv:2608.26414v1 公告类型：新 摘要：医学指南编码了丰富的、基于证据的决策逻辑，但临床医生所需的特定决策工件在指南中难以定位，更不用说跨涵盖疑似疾病和治疗的指南了。尽管指南文本段落已支持端到端的问答，但流程图在决策支持中仍未被充分利用，尽管它们能够编码可操作的临床路径。因此，我们引入了Case2Flow，一个旨在从指南文档集合中为给定患者病例检索最相关指南流程图的任务。为支持该任务，我们构建了FlowAtlas，一个从2,080份医学指南中提取的202个流程图的精选语料库，以及一个合成1,911个对齐病例-流程图对的流水线。我们对多模态检索方法的评估揭示了系统性失败模式，包括过度依赖关键词以及由无信息性背景引发的虚假标记-补丁匹配。

    arXiv:2608.26414v1 Announce Type: new  Abstract: Medical guidelines encode rich, evidence-based decision logic, yet the specific decision artifact a clinician needs is hard to locate within a guideline, let alone across guidelines covering plausible diseases and treatments. While guideline passages have supported end-to-end question answering, flowcharts remain largely underused in decision support despite their ability to encode actionable clinical pathways. We therefore introduce Case2Flow, a task designed to retrieve the most relevant guideline flowchart for a given patient case from a collection of guideline documents. To support it, we construct FlowAtlas, a curated corpus of 202 flowcharts extracted from 2,080 medical guidelines, together with a pipeline that synthesises 1,911 aligned case-flowchart pairs. Our evaluation of multimodal retrieval methods reveals systematic failure modes, including overreliance on keywords and spurious token-patch matches induced by uninformative ba
    
[^20]: 评估证据感知检索在RAG中的下游实用性

    Assessing the Downstream Utility of Evidence-Aware Retrieval in RAG

    [https://arxiv.org/abs/2608.26379](https://arxiv.org/abs/2608.26379)

    本研究评估证据感知检索信号在RAG中的下游实用性，发现其虽改变检索排名，但对训练、系统选择和答案质量预测的益处并不一致。

    

    arXiv:2608.26379v1 公告类型：交叉 摘要：检索增强生成（RAG）的检索评估日益设计为围绕检索到的段落是否包含能够支持生成的证据，而不仅仅是主题相关性。我们研究这种与下游证据需求的更紧密对齐是否也使检索评估对其所构建的决策更有用。在五个检索基准和一个端到端的TREC RAG 2025设置中，我们检查了一个答案支持信号在四个角色中的作用：比较检索器、指导检索训练和系统选择、预测下游答案质量，以及过滤提供给生成器的证据。该信号改变了检索排名，但其下游价值并不统一。它不能可靠地改进检索器训练；使用它进行系统选择的益处取决于生成器被指示如何使用检索到的证据；基于它的检索分数并不能稳健地预测答案质量。

    arXiv:2608.26379v1 Announce Type: cross  Abstract: Retrieval evaluation for retrieval-augmented generation (RAG) is increasingly designed around whether retrieved passages contain evidence that can support generation, rather than topical relevance alone. We study whether this closer alignment with downstream evidence needs also makes retrieval evaluation more useful for the decisions built from it.   Across five retrieval benchmarks and an end-to-end TREC RAG 2025 setting, we examine an answer-support signal in four roles: comparing retrievers, guiding retrieval training and system selection, predicting downstream answer quality, and filtering the evidence supplied to a generator. The signal changes retrieval rankings, but its downstream value is not uniform. It does not reliably improve retriever training; the benefit of using it for system selection depends on how the generator is instructed to use the retrieved evidence; and retrieval scores based on it do not robustly predict answe
    
[^21]: 一个用于编排异构语音与文本检索器的重排序器

    A Reranker for Orchestrating Heterogeneous Speech and Text Retrievers

    [https://arxiv.org/abs/2608.26194](https://arxiv.org/abs/2608.26194)

    本文提出了一种名为STeReO的重排序器，它通过整合语音和文本检索器来聚合异构模态数据库，并利用自建数据集训练，从而在多模态检索中显著提升证据选择的准确性。

    

    检索增强生成（RAG）系统因其能够缓解大型语言模型（LLMs）中的幻觉现象而引起了广泛关注。尽管RAG的知识数据库日益多样化，包括语音和文本等多种模态，但针对此类多模态数据库场景的研究仍然有限。在本文中，我们提出了STeReO（语音与文本重排序编排器），一种基于语音和文本检索器的重排序器，用于聚合不同模态的数据库。为了解决缺乏专门训练数据的问题，我们首先构建了一个包含查询、混合模态证据及其相应相关性排名的数据集。然后，我们训练该重排序器，并在单模态和混合模态场景中评估其有效性。结果表明，所提出的算法擅长选择最相关的证据，从而显著改善下游任务。

    arXiv:2608.26194v1 Announce Type: cross  Abstract: Retrieval-Augmented Generation (RAG) systems have attracted significant interest for their ability to mitigate hallucinations in Large Language Models (LLMs). Although knowledge databases for RAG are increasingly diversifying to include various modalities such as speech and text, research on handling such multi-modal database scenarios remains limited. In this paper, we propose STeReO (Speech and Text Reranking Orchestrator), a reranker based on speech and text retrievers that aggregates disparate modality databases. To address the lack of specialized training data, we first curate a dataset comprising queries, mixed-modality evidence, and their corresponding relevance ranks. We then train the reranker and evaluate its effectiveness in both single-modality and mixed-modality scenarios. Our results demonstrate that the proposed algorithm excels at selecting the most relevant evidence, thereby significantly improving downstream question-
    
[^22]: 利用大型语言模型进行疾病传播模型系统文献综述

    Leveraging Large Language Models for Systematic Literature Review of Disease Spread Models

    [https://arxiv.org/abs/2608.26150](https://arxiv.org/abs/2608.26150)

    本研究开发了一种利用大型语言模型自动提取疾病传播模型论文信息的流水线，发现其准确率接近人工综述，并揭示LLM间一致性可作为输出质量的指示器。

    

    arXiv:2608.26150v1 公告类型：新 摘要：大型语言模型（LLMs）的最新进展为简化和潜在地自动化许多研究过程创造了新的机会，包括系统文献综述（SLRs）。本研究报告了一个LLM流水线的开发，用于从536篇同行评审的基于代理的建模论文中提取模型相关信息。我们将结果与人工进行的SLR结果进行了比较。我们的结果显示，GPT-4.1的论文级准确率约为77.95%，GPT-5.0约为81.67%。领域级准确率范围从32.40%到100.00%，其中更复杂或主观的领域表现可靠性较低。重要的是，我们发现LLMs之间的一致性可能是输出质量的潜在指标：低一致性可能表明幻觉，而高一致性结合低准确率可能指向人类数据集中的噪声或错误。总体而言，我们的研究为提示开发提供了实用见解，并强调了其潜力。

    arXiv:2608.26150v1 Announce Type: new  Abstract: Recent advancements in Large Language Models (LLMs) have created new opportunities to streamline and potentially automate many research processes, including systematic literature reviews (SLRs). This study reports an LLM pipeline development for extracting model-relevant information from 536 peer-reviewed agent-based modeling papers. We compare the results with those of a human-conducted SLR. Our results show paper-level accuracies of approximately 77.95% for GPT-4.1 and 81.67% for GPT-5.0. Field-level accuracy ranges from 32.40% to 100.00%, with more complex or subjective fields performing less reliably. Importantly, we find that agreement between LLMs is a potential indicator of output quality: low agreement may signal hallucinations, whereas high agreement combined with low accuracy may point to noise or errors in the human dataset. Overall, our study provides practical insights into prompt development and highlights both the potentia
    
[^23]: 大型语言模型在学术工作流中的应用：短与长上下文窗口下生成的文献综述评估

    LLMs for Academic Workflows: An Evaluation of Literature Reviews Generated with Short and Long Context Windows of LLMs

    [https://arxiv.org/abs/2608.26145](https://arxiv.org/abs/2608.26145)

    本文评估了不同上下文窗口下LLMs生成的文献综述质量，发现它们需要人工监督，且长上下文虽能整合更多信息但加剧了重复和遗漏关键内容的问题。

    

    摘要：我们的研究聚焦于评估在大型语言模型（LLMs）的短上下文和长上下文设置下生成的文献综述，以探究上下文窗口对AI生成文献综述质量的影响，以及AI在支持文献综述写作中的作用。基于来自Semantic Scholar和Arxiv的研究来源，我们生成了二十篇AI文献综述，并由两位研究人员在15个维度上进行了评估。我们的发现表明，AI生成的文献综述需要人工监督才能达到学术出版标准。随着上下文窗口的增加，LLMs能够整合更广泛的信息并保持长输入的一致性，但这也加剧了内容重复、遗漏关键工作以及倾向于描述性而非综合性等问题。我们的工作表明，AI生成的综述可以提供基础性概述，但其输出必须经过批判性评估和细化。

    arXiv:2608.26145v1 Announce Type: new  Abstract: Our research focuses on evaluating literature reviews generated in short and long context settings of large language models (LLMs) to investigate the impact of context window on the quality of AI-generated literature reviews and the role of AI in supporting literature review writing. Twenty AI-generated literature reviews based on research sources from Semantic Scholar and Arxiv were evaluated by two researchers across 15 dimensions. Our findings reveal that AI-generated literature reviews require human oversight to meet academic publishing standards. As context windows increase, LLMs can incorporate broader information and maintain coherence across longer inputs, but they also exacerbate issues such as content repetition, omission of critical work, and a tendency towards descriptiveness over synthesis. Our work shows that AI-generated reviews can provide foundational overviews, but their output must be critically evaluated and refined b
    
[^24]: 智能体不翻页：LLM工具响应的首块选择策略

    Agents Don't Paginate: First-Chunk Selection for LLM Tool Responses

    [https://arxiv.org/abs/2608.26130](https://arxiv.org/abs/2608.26130)

    该论文发现编码智能体从不使用分页获取额外工具响应块，并提出首块选择策略（基于0/1背包）以最大化首块中黄金项的包含率，实验表明现有方法在精度上存在不足。

    

    摘要：基于大型语言模型（LLM）构建的编码智能体，如Claude Code、Cursor、OpenAI Codex、GitHub Copilot和Aider，接收到的工具响应经常超出智能体每轮的令牌预算。标准的解决方案是分页，这在产生这些响应的所有协议中均可用；然而，在来自公共模型上下文协议中间件的会话日志语料库中，我们观察到没有智能体发起过获取第二块的请求。智能体读取的是首块，因此我们询问黄金项（智能体所需的那一项）在首块中出现的频率，即首块精确率$p_1$。在受控的离线基准测试中，我们将首块选择视为0/1背包问题，并在500个SWE-bench验证任务上比较了六种价值函数，然后通过单轮文件定位探针在五种语言模型（4,800次LLM调用；非端到端解析率测试）上测试$p_1$是否重要。两个预先注册的假设未成立。

    arXiv:2608.26130v1 Announce Type: new  Abstract: Coding agents built on large language models (LLMs), such as Claude Code, Cursor, OpenAI Codex, GitHub Copilot, and Aider, receive tool responses that routinely exceed the agent's per-turn token budget. The standard remedy, pagination, is available in every protocol that produced these responses; yet across the corpus of session logs from a public Model Context Protocol middleware we observed no agent-initiated requests for a second chunk. The first chunk is what the agent reads, so we ask how often the gold item (the one the agent needs) is placed first in it: the precision-at-1 rate $p_1$.   In a controlled offline benchmark we treat first-chunk selection as a 0/1 knapsack and compare six value functions on 500 SWE-bench Verified tasks, then test whether $p_1$ matters with a single-turn file-localisation probe on five language models (4,800 LLM calls; not an end-to-end resolve-rate test). Two pre-registered hypotheses did not hold and 
    
[^25]: 当过时约束未被检查：继承代理记忆中的预算验证失败

    When Stale Constraints Go Unchecked: Budgeted Verification Failures in Inherited Agent Memory

    [https://arxiv.org/abs/2608.25553](https://arxiv.org/abs/2608.25553)

    该论文研究了在有限验证预算下，代理继承的过时约束未被检查导致验证失败的问题，并提出了通过重新分配验证槽位来减少这类错误的方法。

    

    arXiv:2608.25553v1 公告类型：交叉 摘要：一个继承了整合记忆的代理可能继承了一个在写入时成立但已被更新的权威记录撤销的约束。在稀缺的验证预算下，代理能否恢复该撤销？如果不能，这种错误是否能在不增加支出的情况下避免？我们明确建模了替代关系——历史来源是不可变的；变化的是哪个记录是当前的——并设计性地分配了记忆的形式、世界状态（来源当前或已被替代）以及固定预算为两条记录的验证策略：代理自身的分配，或相同预算但将一个槽位重新分配给关键来源路径或随机记录。在声明约束的情况下，代理在大约五分之一的回合中检查了其来源路径；当该约束已被替代时，原生分配在主要运行、新措辞等中分别产生了77.3%、74.7%和74.7%的回合中的过时一致决策。

    arXiv:2608.25553v1 Announce Type: cross  Abstract: An agent that inherits a consolidated memory may inherit a constraint that was true when written and has since been withdrawn by a newer authoritative record. Under a scarce verification budget, does the agent recover the withdrawal, and if not, is the error avoidable without spending more? We model supersession explicitly -- historical provenance is immutable; what changes is which record is current -- and assign by design the memory's form, the world's state (source current or superseded), and the verification policy at a fixed budget of two records: the agent's own allocation, or the same budget with one slot re-assigned to the critical provenance path or to a random record. With a constraint stated, agents inspected its provenance path in about one episode in five; when that constraint had been superseded, native allocation produced stale-consistent decisions in 77.3%, 74.7% and 74.7% of episodes across a primary run, a fresh-wordi
    
[^26]: DocPC：通过代表性页面组合实现文档级视觉检索

    DocPC: Document-Level Visual Retrieval via Representative Page Composition

    [https://arxiv.org/abs/2608.25434](https://arxiv.org/abs/2608.25434)

    本文提出了DocPC框架，通过代表性页面组合将多页文档编码为单一网格图像，大幅降低索引成本，并引入多正样本对比学习与稀疏列表优化，显著提升文档级视觉检索效率与性能。

    

    视觉文档检索通过使用视觉语言模型对页面截图进行编码而取得了进展，绕过了OCR流程。然而，现有方法仍然以页面为中心，与需要完整文档检索的真实场景不符。一种简单的“先页面后文档”聚合方式面临线性索引成本，并且当相关性跨越多个页面时检索性能下降。我们提出了DocPC，一种基于代表性页面组合的文档级视觉检索框架：选择代表性页面并将它们组合成单个网格图像以进行文档级索引，将索引图像、向量和存储减少了10.1倍，端到端索引时间减少了约7.7倍。为了处理文档级别普遍存在的多正样本监督，我们将多正样本对比学习与稀疏调度的列表式优化相结合。我们还引入了DocViRe，一个具有多正样本相关性标注的基准。DocPC-ColQwen实现了最先进的性能。

    arXiv:2608.25434v1 Announce Type: new  Abstract: Visual document retrieval has advanced by encoding page screenshots with vision-language models, bypassing OCR pipelines. However, existing methods remain page-centric, misaligned with real-world scenarios requiring complete document retrieval. A naive page-then-document aggregation suffers from linear indexing cost and degraded retrieval when relevance spans multiple pages. We propose DocPC, a document-level visual retrieval framework based on Representative Page Composition: selecting representative pages and composing them into a single grid image for document-level indexing, reducing indexed images, vectors, and storage by 10.1x and end-to-end indexing time by roughly 7.7x. To handle multi-positive supervision prevalent at the document level, we combine multi-positive contrastive learning with sparsely scheduled listwise optimization. We also introduce DocViRe, a benchmark with multi-positive relevance annotations. DocPC-ColQwen achi
    
[^27]: 检索变换器：一种用于高效近似最近邻检索和冷物品推荐的双编码器变换器

    RetrievalFormer: A Dual-Encoder Transformer for Efficient Approximate Nearest Neighbor Retrieval and Cold-Item Recommendation

    [https://arxiv.org/abs/2608.24079](https://arxiv.org/abs/2608.24079)

    本文提出了一种双编码器变换器检索框架，能够仅基于特征对新物品进行高效近似最近邻检索，并保持索引对新物品开放，无需重新训练，同时展示了在冷物品推荐上的优越性能。

    

    摘要：arXiv:2608.24079v1 公告类型：交叉 摘要：共享的搜索与推荐索引必须仅从特征中评分新物品，因为搜索没有探索槽位。在一个覆盖同一目录中两个表面的公共日志中，$38.6\%$的保留查询-搜索印象显示了一个从未被展示或访问过的物品。对于用户冷启动参与，基于特征的塔式结构在无测量损失的情况下满足这一需求，与$99$个采样负样本相比（Recall@20为$0.9595$，而热启动为$0.9510$）。一个词汇基线达到了类似的同等水平，而完整目录检查在统计上仍不确定。因此，双编码器检索保持索引对新物品开放，不同于需要重新训练的ID-softmax推荐器。我们在推荐方面针对六个序列基线评估这种开放性，每个基线在修正目标上经过五轮重新训练和调优。一个float32时间戳错误曾为$19.7\%$的用户重新排序了留一法目标。在MovieLens-1M上，热启动准确性落后于强基线。

    arXiv:2608.24079v1 Announce Type: cross  Abstract: A shared search-and-recommendation index must score new items from features alone because search has no exploration slot. In a public log covering both surfaces over one catalog, $38.6\%$ of held-out query-search impressions show an item never previously shown or visited. For user-cold engagements, the feature-based tower serves this demand without measurable loss against $99$ sampled negatives ($0.9595$ Recall@20 versus $0.9510$ warm). A lexical baseline reaches similar parity, while a full-catalog check remains statistically undecided. Dual-encoder retrieval therefore keeps the index \emph{open} to new items, unlike an ID-softmax recommender that requires retraining. We price this openness on recommendation against six sequential baselines, each retrained and tuned through five rounds on corrected targets. A float32 timestamp bug had reordered leave-one-out targets for $19.7\%$ of users. On MovieLens-1M, warm accuracy trails the stro
    
[^28]: ExecRubrics：可执行工具增强的评分标准，用于可验证且高效的长篇评估

    ExecRubrics: Executable Tool-Augmented Rubrics for Verifiable and Efficient Long-Form Evaluation

    [https://arxiv.org/abs/2608.22559](https://arxiv.org/abs/2608.22559)

    ExecRubrics通过将评分标准转化为可执行的Python函数，实现了可验证、高效且能捕捉复杂依赖关系的长篇评估，替代了昂贵的黑盒LLM评判器。

    

    摘要：arXiv:2608.22559v1 公告类型：新 摘要：评分标准旨在通过将回答质量分解为可解释的准则，使语言模型评估透明化。然而，自然语言评分标准往往含糊不清，需要黑盒LLM评判器，并且通常假设准则通过线性加权和独立聚合，这限制了其捕捉依赖关系、替代方案、惩罚和覆盖条件的能力。我们提出ExecRubrics，一个将评分标准表示为紧凑可执行程序的框架。ExecRubrics将评估逻辑编码为可验证的Python评分函数，赋予自然语言评分标准意图一种操作语义：一个可检查、可执行和可编辑的固定决策程序。在三个长篇回答基准测试——HealthBench、HelpSteer和ArgQuality上，我们展示了ExecRubrics可以替代昂贵的黑盒评判器，在偏好排序中优于或匹配自然语言评分标准基线，具有最佳偏好性能。

    arXiv:2608.22559v1 Announce Type: new  Abstract: Rubrics aim to make language-model evaluation transparent by decomposing response quality into interpretable criteria. However, natural-language rubrics are often ambiguous, require black-box LLM judges, and typically assume criteria aggregate independently through linear weighted sums, limiting their ability to capture dependencies, alternatives, penalties, and override conditions. We propose ExecRubrics, a framework for representing rubrics as compact executable programs. ExecRubrics encodes evaluation logic as verifiable Python scoring functions, giving natural-language rubric intent an operational semantics: a fixed decision procedure that can be inspected, executed, and edited. On three long-form response benchmarks-HealthBench, HelpSteer, and ArgQuality-we show that ExecRubrics can substitute for expensive black-box judges in ranking preferred over dispreferred responses, matching or improving NL rubric baselines with best preferen
    
[^29]: 融合逐层退出机制以增强紧凑型大语言模型用于推荐系统

    Empowering Compact LLMs with Fusion of Layer-wise Exits for Recommendation

    [https://arxiv.org/abs/2608.17316](https://arxiv.org/abs/2608.17316)

    本文提出FLEXRec，一种通过融合紧凑型大语言模型多Transformer层退出点的得分分布来增强其表达能力，同时保持可扩展全语料库排名的判别式推荐框架。

    

    基于大语言模型的推荐系统（LLM-RSs）展现了卓越的能力，但在许多实际应用中计算成本过高，难以持续。紧凑型大语言模型提供了一种实用的替代方案，但其能力缩减通常需要推理或知识蒸馏方法，这增加了延迟或依赖更大模型。结合自回归生成，这些方法面临严重的可扩展性瓶颈。相比之下，判别式LLM-RSs通过嵌入相似性实现高效的全语料库排名，但紧凑型骨干网络在表达力和结构适应性方面仍然受限。我们提出了用于序列推荐的逐层退出融合框架（FLEXRec），这是一种判别式框架，在增强紧凑型大语言模型的同时保持可扩展的全语料库排名。FLEXRec在多个Transformer层插入预测头（即退出点），并自适应融合其得分分布。一种自适应...

    arXiv:2608.17316v1 Announce Type: new  Abstract: Large language model-based recommender systems (LLM-RSs) have demonstrated remarkable capabilities, but are computationally unsustainable for many real-world applications. Compact LLMs offer a practical alternative, yet their reduced capacity often requires reasoning or knowledge distillation methods that increase latency or depend on larger models. Combined with autoregressive generation, these approaches face severe scalability bottlenecks. In contrast, discriminative LLM-RSs enable efficient full-corpus ranking through embedding similarity, but compact backbones remain limited in expressiveness and structural adaptivity. We propose the Fusion of Layer-wise Exits for Sequential Recommendation (FLEXRec), a discriminative framework that enhances compact LLMs while retaining scalable full-corpus ranking. FLEXRec inserts prediction heads (i.e., exits) at multiple transformer layers and adaptively fuses their score distributions. An adaptiv
    
[^30]: NRCD：一个带有统一性能标准化的大学跑步开放数据库

    NRCD: An Open Database of Collegiate Running with Unified Performance Standardization

    [https://arxiv.org/abs/2608.14776](https://arxiv.org/abs/2608.14776)

    本文首次发布了大规模公开的大学跑步数据集NRCD，包含超过12.8万条标准化成绩，覆盖多个项目和广泛的时间范围，并附带详细的赛道和天气元数据，填补了该领域数据可获取性的空白。

    

    美国的大学跑步每年在越野和田径比赛中产生数千条比赛成绩，但目前还没有公开的大规模数据集可供研究。现有的网站如Athletic.net、MileSplit和TFRRS虽然提供成绩查询，但不支持批量下载，这限制了先前的研究仅能分析约500条成绩，且往往偏向于男性运动员。我们引入了国家跑步俱乐部数据库（NRCD），这是首个大规模公开的大学跑步数据集：包含来自28,913名运动员的128,963条认可成绩，覆盖1,336场比赛，涉及四个运动项目（越野、室内和室外田径、公路赛），其中女性占36.3%，时间跨度从2004年到2026年。在单一导出中，2023年8月及之后的比赛附带完整的赛道距离、海拔升降、比赛时天气和跑道场地元数据（97.7%的越野数据行包含天气信息）；早至2004年的赛季也包含在内。

    arXiv:2608.14776v1 Announce Type: new  Abstract: Collegiate running in the United States generates thousands of race results annually in cross country and track and field, yet no large-scale dataset has been publicly available for research. Existing websites such as Athletic.net, MileSplit, and TFRRS host results but do not support bulk download, restricting prior analyses to ~500 performances, often skewing studies toward male athletes. We introduce the National Running Club Database (NRCD), the first openly available collegiate running dataset at scale: 128,963 approved performances from 28,913 athletes across 1,336 meets in four sports (cross country (XC), indoor and outdoor track, and road races), 36.3% women, spanning 2004 through 2026. Within that single export, meets from August 2023 onward carry comprehensive course distance, elevation gain and loss, weather at race time, and track venue metadata (97.7% of XC rows with weather fields); earlier seasons back to 2004 are included 
    
[^31]: 基于表示驱动的参数高效推荐系统

    REPREC: Representation Driven Parameter-Efficient Recommendation System

    [https://arxiv.org/abs/2607.24845](https://arxiv.org/abs/2607.24845)

    REPREC通过仅训练一个轻量级MLP注入器，将冻结序列编码器的用户表示映射为软令牌来条件化冻结的LLM，实现参数高效的序列推荐，同时保持预训练模型不变。

    

    大型语言模型（LLMs）已被应用于序列推荐，通过将其表述为自然语言任务。先前的工作通过输入条件化或LLM微调，结合协同和序列信号来提高个性化。然而，现有方法通常依赖以下一种或多种方式：LLM微调、额外架构模块、表示蒸馏或对长交互历史的项目级条件化，这增加了训练复杂性和部署成本。我们提出REPREC，一种轻量级框架，通过轻量级用户表示对齐来重构基于LLM的序列推荐。REPREC将来自冻结序列编码器的固定大小用户嵌入，通过一个轻量级MLP注入器映射到一组学习的软令牌中，从而条件化冻结的LLM，保持两个预训练骨干不变，仅训练注入器。

    arXiv:2607.24845v3 Announce Type: replace-cross  Abstract: Large language models (LLMs) have been applied to sequential recommendation by formulating it as a natural language task. Previous work has improved personalization by incorporating collaborative and sequential signals through input conditioning or LLM fine-tuning. However, existing approaches often rely on one or more of the following: LLM fine-tuning, additional architectural modules, representation distillation, or item-level conditioning over long interaction histories, increasing training complexity and deployment cost. We propose REPREC, a lightweight framework that reformulates LLM-based sequential recommendation through lightweight user representation alignment. REPREC maps a fixed-size user embedding from a frozen sequential encoder into a small set of learned soft tokens through a lightweight MLP injector that conditions a frozen LLM, leaving both pretrained backbones unchanged while training only the injector. We con
    
[^32]: 基于策略感知的参数高效自适应方法用于大语言模型驱动的自动竞价

    Strategy-Aware Parameter-Efficient Adaptation for LLM-based Auto-Bidding

    [https://arxiv.org/abs/2607.24232](https://arxiv.org/abs/2607.24232)

    本文提出SAGE框架，通过参数高效的多模态对齐（包括时间语义位置嵌入和门控交叉注意力），在不进行昂贵微调的情况下，利用大语言模型实现策略感知的高效自动竞价。

    

    arXiv:2607.24232v2 公告类型：替换 摘要：广告竞价已从手动策略演变为更适合大规模、动态拍卖环境的自动竞价系统。尽管大语言模型（LLMs）的最新进展为自动竞价提供了强大的推理能力，但现有方法存在轨迹文本交互浅层化的问题，并且需要昂贵的微调，这阻碍了在多样约束下对预训练知识的高效利用。为解决这些挑战，我们提出了SAGE，一种新颖的、由LLMs引导的策略感知自动竞价框架，用于高效竞价。SAGE引入了一个参数高效的多模态对齐框架，用于约束下的LLM自动竞价。具体来说，SAGE包含三个关键组件：（i）位置增强模块采用时间语义位置嵌入，有效捕获内在动态和语义结构；（ii）文本对齐模块利用门控交叉注意力来对齐嵌入空间（原文截断）。

    arXiv:2607.24232v2 Announce Type: replace  Abstract: Advertising bidding has evolved from manual strategies to auto-bidding systems better adapted for large-scale, dynamic auction environments. While recent advances in Large Language Models (LLMs) offer strong reasoning for auto-bidding, existing methods suffer from shallow trajectory-text interactions and require costly fine-tuning, hindering the efficient use of pretrained knowledge under diverse constraints. To address these challenges, we propose SAGE, a novel Strategy-aware Auto-bidding framework Guided by LLMs for Efficient bidding. SAGE introduces a parameter-efficient multi-modal alignment framework for constrained auto-bidding with LLMs. Specifically, SAGE comprises three key components: (i) the position augmentation module adopts temporal-semantic positional embeddings to effectively capture the intrinsic dynamics and semantic structures; (ii) the text alignment module leverages gated cross-attention to align the embedding sp
    
[^33]: 漂移自适应的ICU干预预测：冻结生理编码器以实现可审计的模型更新

    Drift-Adaptive ICU Intervention Prediction: Freezing the Physiological Encoder for Auditable Model Updating

    [https://arxiv.org/abs/2607.19020](https://arxiv.org/abs/2607.19020)

    该论文提出了一种双流架构，在模型更新时冻结生理编码器，仅更新治疗流，从而实现可审计的ICU干预预测，且性能与全参数更新相当。

    

    arXiv:2607.19020v2 公告类型：替换交叉 摘要：随着治疗方案的发展，临床决策支持系统会性能下降，但更新已部署模型的障碍既在于治理也在于准确性：一旦重新训练触及所有参数，之后无人能说明更新作用在何处。我们提出了一种双流架构，将生理（LSTM）表示与治疗（MLP）表示分离。在双重分布/准确性触发条件下，更新仅限于治疗流和融合头，使生理编码器与源模型逐位保持一致。审计日志记录更新所依赖的治疗特征，证据检索将每个实例的PubMed查询与冻结的编码器耦合。我们在按三年时期划分的84,792个MIMIC-IV住院病例上进行了评估。该约束几乎无成本：选择性适应在整体判别力上与无约束的全适应相比无损失（平均AUROC 0.9316对比0.9249；在血管加压药方面领先，其他方面略落后）。

    arXiv:2607.19020v2 Announce Type: replace-cross  Abstract: Clinical decision support degrades as treatment protocols evolve, but the obstacle to updating a deployed model is governance as much as accuracy: once retraining touches every parameter, no one can say afterwards where the update acted. We propose a two-stream architecture separating physiological (LSTM) from treatment (MLP) representations. On a dual distributional/accuracy trigger, updates are confined to the treatment stream and fusion head, leaving the physiological encoder bitwise identical to the source model. Audit logs record which treatment features the update relied on, and evidence retrieval couples per-instance PubMed queries to the frozen encoder. We evaluate on 84,792 MIMIC-IV stays split by three-year era. The constraint proved close to free: selective adaptation cost nothing in aggregate discrimination against unconstrained full adaptation (mean AUROC 0.9316 vs. 0.9249; ahead on vasopressor, marginally behind o
    
[^34]: 基于矩阵分解MDP的候选生成规划

    Planning over Matrix-Factorization MDPs for Candidate Generation

    [https://arxiv.org/abs/2607.02115](https://arxiv.org/abs/2607.02115)

    该论文提出将前K物品检索建模为基于矩阵分解后验的MDP，通过规划用户状态动态来改进推荐候选生成，并验证了动态感知规划在特定条件下优于静态检索。

    

    摘要：对于推荐服务，我们将客户旅程视为一系列物品推荐的链条：一个有用的物品会改变用户的状态，从而影响接下来应检索的内容。标准矩阵分解检索忽略了这一点——它构建一个用户向量，并根据静态分数返回前K个物品，将它们视为独立项。我们提出了一个明确的问题：在折叠引入（fold-in）导致的用户状态动态中，何时值得进行规划？为回答此问题，我们提出将前K检索建模为基于隐式ALS后验$(A^{-1},u)$的MDP，其中动作是物品，转移是闭合形式的秩一折叠引入，轨迹奖励结合了相关性相似度和后验对齐项。在相同固定嵌入下，我们比较了静态检索、单步规划和水平K的MCTS，跨五个数据集和两种协议：每用户留最后n个划分和更严格的全局时间划分。动态感知规划...

    arXiv:2607.02115v2 Announce Type: replace  Abstract: For a recommender service, we view the customer journey as a chain of item recommendations: a useful item changes the user's state and therefore what should be retrieved next. Standard matrix-factorization retrieval ignores this -- it builds one user vector and returns the top-$K$ items by a static score, treating them as independent. We ask a narrow question: when is it worth planning over the user-state dynamics that fold-in induces? To answer it we propose casting top-$K$ retrieval as an MDP over the implicit-ALS posterior $(A^{-1},u)$, where an action is an item and the transition is a closed-form rank-one fold-in, and the trajectory reward combines a relevance similarity with a posterior-alignment term. Under the same fixed embeddings we compare static retrieval, one-step planning, and horizon-$K$ MCTS across five datasets and two protocols: a per-user leave-last-$n$ split and a stricter global time split. Dynamics-aware plannin
    
[^35]: 移位：通过索引侧特征变换实现多语言信息检索的语义协调

    SHIFT: Semantic Harmonization via Index-side Feature Transformation for Multilingual Information Retrieval

    [https://arxiv.org/abs/2606.18801](https://arxiv.org/abs/2606.18801)

    SHIFT提出了一种无训练、索引侧的特征变换方法，通过平行翻译对估计并校正语言偏移，有效缓解多语言信息检索中的语言偏见问题。

    

    arXiv:2606.18801v2 公告类型：替换交叉  摘要：随着大规模多语言语料库的迅速扩展，多语言信息检索（MLIR）已成为全球信息获取的关键技术。MLIR使用户能够通过单一语言查询，从多语言文本集合中检索语义相关的文档。然而，最近的多语言稠密检索模型往往表现出对与查询同语言文档的强烈偏好，导致严重的语言偏见，即使其他语言的文档包含更多语义相关信息，排名靠前的结果仍由特定语言的文档主导。为解决这一问题，我们提出了SHIFT，一种适用于索引阶段的无训练方法。具体来说，SHIFT利用平行翻译对来估计每个目标语言相对于源语言的相对语言向量，随后，SHIFT校正语言特定的偏移。

    arXiv:2606.18801v2 Announce Type: replace-cross  Abstract: With the rapid expansion of massive multilingual corpora, Multilingual Information Retrieval (MLIR) has emerged as a critical technology for global information access. MLIR enables users to retrieve semantically relevant documents from multilingual text collections using a single-language query. However, recent multilingual dense retrieval models often exhibit a strong preference for documents in the same language as the query. This leads to severe language bias, where top-ranked results are dominated by documents of specific languages, even when documents in other languages contain more semantically relevant information. To address this issue, we propose SHIFT, a training-free method applicable in the indexing stage. Specifically, SHIFT utilizes parallel translation pairs to estimate a relative language vector for each target language with respect to a source language. Subsequently, SHIFT corrects the language-specific offset 
    
[^36]: 查询何时应被分解？面向多条件检索的查询分解阶段感知研究

    When Should Queries Be Decomposed? A Stage-Aware Study of Query Decomposition for Multi-Condition Retrieval

    [https://arxiv.org/abs/2606.08577](https://arxiv.org/abs/2606.08577)

    本文提出一种阶段感知查询分解框架，在初始检索保留整体查询、重排序阶段使用子查询，从而显著提升多条件检索性能。

    

    多条件检索要求系统识别满足多个不同约束的文档，这超越了单纯的主题相关性。尽管查询分解作为一种直观的补救措施被广泛采用，但其在不同检索流水线阶段的有效性仍未被充分探索。在本文中，我们进行了一项阶段感知的实证研究，并揭示了一个显著且依赖阶段的效应：在初始检索阶段进行分解常因语义稀释而损害检索性能，但在重排序阶段却能通过实现更细粒度的约束验证而大幅提升效果。基于这些洞察，我们提出了一种原则性的阶段感知分解框架，该框架在初始检索阶段保留整体查询以维持全局语义上下文，而仅在重排序阶段使用子查询进行细粒度约束匹配。在MultiConIR和SSRB基准上的广泛评估表明，该方法具有显著优势。

    arXiv:2606.08577v2 Announce Type: replace  Abstract: Multi-condition retrieval requires systems to identify documents that satisfy multiple distinct constraints, moving beyond mere topical relevance. While query decomposition is widely adopted as an intuitive remedy, its effectiveness across different retrieval pipeline stages remains underexplored. In this paper, we conduct a stage-aware empirical study and uncover a stark, stage-dependent effect: decomposition during initial retrieval frequently harms retrieval performance due to semantic dilution, yet substantially improves reranking by enabling more fine-grained constraint verification. Motivated by these insights, we propose a principled Stage-Aware Decomposition framework that retains the monolithic query during initial retrieval to preserve global semantic context, while employing sub-queries exclusively during reranking for fine-grained constraint matching. Extensive evaluations on the MultiConIR and SSRB benchmarks demonstrate
    
[^37]: MIMO：通过单语目标实现多语言信息检索

    MIMO: Multilingual Information Retrieval via Monolingual Objectives

    [https://arxiv.org/abs/2605.31171](https://arxiv.org/abs/2605.31171)

    MIMO通过两阶段框架，利用英语教师模型作为锚点，结合知识蒸馏和跨语言对比学习，解决了多语言信息检索中语言聚类和性能下降的问题。

    

    arXiv:2605.31171v2 公告类型：替换-交叉 摘要：多语言信息检索（MLIR）反映了现实世界的搜索环境，其中查询和相关文档可能以不同语言出现在混合语言语料库中。然而，现有的嵌入模型主要针对多单语检索进行优化，其性能在MLIR设置中常常下降。此外，将传统对比学习直接应用于MLIR可能会加剧语言聚类，并在跨语言对齐和嵌入均匀性之间暴露出权衡问题。为解决这些限制，我们提出了MIMO：通过单语目标实现多语言信息检索，这是一个两阶段框架，使用高性能教师模型的稳定英语语义空间作为锚点。MIMO首先通过知识蒸馏初始化学生模型的跨语言对齐，然后联合优化蒸馏和跨语言对比学习以改善检索性能。

    arXiv:2605.31171v2 Announce Type: replace-cross  Abstract: Multilingual Information Retrieval (MLIR) reflects real-world search environments in which queries and relevant documents may appear in different languages within a mixed-language corpus. However, existing embedding models are primarily optimized for Multi-Monolingual retrieval and their performance often degrades in MLIR settings. Moreover, directly applying conventional contrastive learning to MLIR can exacerbate language clustering and expose a trade-off between cross-lingual alignment and embedding uniformity. To address these limitations, we propose MIMO: Multilingual Information Retrieval via Monolingual Objectives, a two-stage framework that uses a stable English semantic space from a high-performing teacher model as an anchor. MIMO first initializes the student model's cross-lingual alignment through knowledge distillation, and then jointly optimizes distillation and cross-lingual contrastive learning to improve retriev
    
[^38]: 相同排名，不同赢家：评分目标如何塑造LLM记忆基准测试

    Same Ranking, Different Winner: How Scoring Targets Shape LLM Memory Benchmarks

    [https://arxiv.org/abs/2605.24060](https://arxiv.org/abs/2605.24060)

    本文揭示了LLM记忆基准测试中评分目标选择的模糊性会显著影响排名结论，并提出TIAP审计方法，无需重跑检索即可评估不同目标对结果的影响。

    

    arXiv:2605.24060v2 公告类型：替换 摘要：对话记忆系统越来越多地将对话历史转换为事实、摘要、时间线及其他关联来源的派生内容，因此同一来源轮次可以与多个派生记忆共存于同一检索索引中。这引发了一个未被充分明确的评估问题：哪个存储形式应获得检索信用？我们表明，这种评分目标的选择常常被隐式处理，并可能实质性地改变基准测试结论。我们提出了TIAP，一种固定输出审计方法，在三种目标——原始、来源和规范——下对保存的排序输出进行重新评分，而无需重新运行检索。在LoCoMo和LongMemEval-S上，仅更改信用目标就改变了83.4%至94.0%共享查询的nDCG，翻转了Mem0和MemoryOS传输运行的目标排序，并逆转了解析器密度建议。一项1,902案例的语义审计进一步表明，放宽的来源关联信用仅在29.2%的情况下完全合理。

    arXiv:2605.24060v2 Announce Type: replace  Abstract: Conversational-memory systems increasingly transform dialogue history into facts, summaries, timelines, and other source-linked descendants, so a single source turn can coexist with several derived memories in the same retrieval index. This raises an underspecified evaluation question: which stored form should receive retrieval credit? We show that this scoring-target choice is often left implicit and can materially change benchmark conclusions. We present TIAP, a fixed-output audit that rescores saved ranked outputs under three targets -- Raw, Source, and Canonical -- without rerunning retrieval. On LoCoMo and LongMemEval-S, switching only the credited target changes nDCG on 83.4--94.0 percent of shared queries, flips target orderings on Mem0 and MemoryOS transfer runs, and reverses parser-density recommendations. A 1,902-case semantic audit further shows that relaxed source-linked credit is fully justified only 29.2 percent of the 
    
[^39]: 基于类别与流行度引导的视频游戏推荐：一种平衡导向框架

    Category-based and Popularity-guided Video Game Recommendation: A Balance-oriented Framework

    [https://arxiv.org/abs/2604.14598](https://arxiv.org/abs/2604.14598)

    本文提出CPGRec框架，通过结合准确性驱动、多样性驱动和综合三个模块，利用物品类别和流行度信息，在视频游戏推荐中实现准确性与多样性的平衡。

    

    arXiv:2604.14598v4 公告类型：替换 摘要：近年来，视频游戏行业经历了显著增长，为玩家提供了海量的游戏选择。这种选项激增催生了对专门针对视频游戏的推荐系统的需求。然而，当前的视频游戏推荐方法往往优先考虑准确性而非多样性，可能导致游戏建议缺乏变化。此外，现有的游戏推荐方法通常缺乏在游戏之间建立严格关联以提升准确性的能力。进一步地，许多现有的以多样性为重点的方法在邻居建模和信息传播过程中未能利用关键的物品信息，如物品类别和流行度。为应对这些挑战，我们提出了一种名为CPGRec的新框架，包含三个模块，即准确性驱动、多样性驱动和综合模块。第一个模块扩展了最先进的以准确性为重点的方法。

    arXiv:2604.14598v4 Announce Type: replace  Abstract: In recent years, the video game industry has experienced substantial growth, presenting players with a vast array of game choices. This surge in options has spurred the need for a specialized recommender system tailored for video games. However, current video game recommendation approaches tend to prioritize accuracy over diversity, potentially leading to unvaried game suggestions. In addition, the existing game recommendation methods commonly lack the ability to establish strict connections between games to enhance accuracy. Furthermore, many existing diversity-focused methods fail to leverage crucial item information, such as item category and popularity during neighbor modeling and message propagation. To address these challenges, we introduce a novel framework, called CPGRec, comprising three modules, namely accuracy-driven, diversity-driven, and comprehensive modules. The first module extends the state-of-the-art accuracy-focuse
    
[^40]: CPGRec+：一种面向个性化视频游戏推荐的平衡导向框架

    CPGRec+: A Balance-oriented Framework for Personalized Video Game Recommendations

    [https://arxiv.org/abs/2604.14586](https://arxiv.org/abs/2604.14586)

    本文提出了CPGRec+框架，通过偏好感知边缘重加权（PER）模块和利用大型语言模型能力，解决游戏推荐中准确性与多样性的权衡，并缓解过平滑问题。

    

    arXiv:2604.14586v4 公告类型：交叉替换 摘要：游戏行业的快速扩张要求推荐系统适应其动态格局。现有的基于图神经网络（GNN）的方法主要优先考虑准确性而非多样性，忽视了它们之间的固有权衡。为解决这一问题，我们先前提出了CPGRec，一种平衡导向的游戏推荐系统。然而，CPGRec未能考虑玩家-游戏交互中的关键差异，这些差异在反映玩家个人偏好方面具有不同重要性，并可能加剧基于GNN模型固有的过平滑问题。此外，现有方法未充分利用大型语言模型（LLMs）的推理能力和广泛知识来解决这些局限性。为弥补这一差距，我们提出了两个新模块。首先，偏好感知边缘重加权（PER）模块分配带符号的边缘权重，以定性区分显著的玩家兴趣和厌恶。

    arXiv:2604.14586v4 Announce Type: replace-cross  Abstract: The rapid expansion of gaming industry requires advanced recommender systems tailored to its dynamic landscape. Existing Graph Neural Network (GNN)-based methods primarily prioritize accuracy over diversity, overlooking their inherent trade-off. To address this, we previously proposed CPGRec, a balance-oriented gaming recommender system. However, CPGRec fails to account for critical disparities in player-game interactions, which carry varying significance in reflecting players' personal preferences and may exacerbate over-smoothness issues inherent in GNN-based models. Moreover, existing approaches underutilize the reasoning capabilities and extensive knowledge of large language models (LLMs) in addressing these limitations. To bridge this gap, we propose two new modules. First, Preference-informed Edge Reweighting (PER) module assigns signed edge weights to qualitatively distinguish significant player interests and disinterest
    
[^41]: 加速生成：通过大语言模型生成的枢轴文档改进重排序

    Generate to Accelerate: Improved Reranking via LLM-Generated Pivot Documents

    [https://arxiv.org/abs/2604.09492](https://arxiv.org/abs/2604.09492)

    本文提出利用大语言模型生成伪相关文档作为枢轴，替代传统依赖现有文档的重排序策略，从而减少计算开销并提升重排序效率。

    

    arXiv:2604.09492v3 公告类型：替换 摘要：减少重排序模型计算开销的常见方法包括识别用于重排序的候选文档集，或构建比较图以最小化冗余比较。对于逐点排序器，确定候选集通常涉及基于排名靠前文档的分数来估计查询相关的截断点。相比之下，列表式方法的比较图通常通过启发式方法推导，例如在滑动窗口内自下而上地传播局部比较，或通过基于枢轴策略自上而下地减少比较。在这项工作中，我们认为将这些过程限制在集合中已有的文档是不必要的。相反，我们提出利用大语言模型的生成能力来为给定查询合成一个伪相关文档。然后，我们调整现有的重排序方法，并提出一种新颖的并行重排序方法。

    arXiv:2604.09492v3 Announce Type: replace  Abstract: Common approaches to reduce the computational overhead of reranking models include identifying a candidate set of documents for reranking or constructing comparison graphs to minimize redundant comparisons. For pointwise rankers, determining a candidate set typically involves estimating a query-dependent cutoff based on the scores of the top-ranked documents. In contrast, comparison graphs for listwise approaches are often derived using heuristics, such as propagating local comparisons within sliding windows in a bottom-up fashion or reducing comparisons via pivot-based strategies in a top-down manner. In this work, we argue that restricting these processes to existing documents in the collection is unnecessary. Instead, we propose leveraging the generative capabilities of large language models to synthesize a pseudo-relevant document for a given query. We then adapt existing reranking approaches and also propose a novel parallel rer
    
[^42]: 面向检索增强生成的LLM特定效用

    LLM-Specific Utility for Retrieval-Augmented Generation

    [https://arxiv.org/abs/2510.11358](https://arxiv.org/abs/2510.11358)

    本文首次形式化并实证了检索增强生成中证据的LLM特定效用，证明其具有模型依赖性和不可转移性，为优化RAG系统提供了新视角。

    

    arXiv:2510.11358v3 公告类型：替换-交叉 摘要：检索增强生成（RAG）通常针对主题相关性进行优化，但其成功最终取决于检索到的段落是否有助于大型语言模型（LLM）生成正确且完整的答案。我们认为，这种效用往往是LLM特定的，而非普遍通用的，这归因于模型在知识、推理和利用证据能力方面的差异。我们将LLM特定效用形式化为，当提供某个段落时，目标LLM的性能相比无证据作答时的提升幅度。为系统研究LLM特定效用，我们构建了一个基准，针对四个LLM（Qwen3-8B/14B/32B和Llama 3.1-8B）在三个问答数据集（Natural Questions、TriviaQA和MS MARCO-FQA）上提供了LLM特定的黄金效用段落。我们的分析表明，效用段落具有模型依赖性和不可转移性：每个LLM在其自身的效用证据下表现最佳，而为其他模型优化的证据则表现不佳。

    arXiv:2510.11358v3 Announce Type: replace-cross  Abstract: Retrieval-augmented generation (RAG) is typically optimized for topical relevance, yet its success ultimately depends on whether retrieved passages are useful for a large language model (LLM) to generate correct and complete answers. We argue that such utility is often LLM-specific rather than universal, due to differences in models' knowledge, reasoning, and ability to leverage evidence. We formalize LLM-specific utility as the performance improvement of a target LLM when a passage is provided, compared to answering without evidence. To systematically study LLM-specific utility, we construct a benchmark of LLM-specific gold utilitarian passages for four LLMs (Qwen3-8B/14B/32B and Llama 3.1-8B) on three QA datasets (Natural Questions, TriviaQA, and MS MARCO-FQA). Our analysis shows that utilitarian passages are model-dependent and non-transferable: each LLM performs best with its own utilitarian evidence, while evidence optimiz
    
[^43]: SustainableQA：面向企业可持续性与欧盟分类法报告的综合问答数据集

    SustainableQA: A Comprehensive Question Answering Dataset for Corporate Sustainability and EU Taxonomy Reporting

    [https://arxiv.org/abs/2508.03000](https://arxiv.org/abs/2508.03000)

    本文提出了SustainableQA，一个包含超过19.5万问答对的综合数据集及其可扩展生成流水线，通过自动化评估与精炼机制确保高质量，专门服务于企业可持续性和欧盟分类法报告中的精确数据提取任务。

    

    arXiv:2508.03000v3 公告类型：替换 摘要：随着企业可持续性透明度需求的日益增长，特别是在欧盟分类法等新法规下，从大型非结构化企业报告中精确提取数据变得至关重要，而大型语言模型和检索增强生成（RAG）系统在此任务中需要高质量、领域特定的问答数据集。为解决这一问题，我们引入了SustainableQA，这是一个新颖的数据集和一个可扩展的流水线，通过整合语义分块分类、混合跨度提取流水线和专门的表格到段落转换，从企业可持续性和年度报告中生成全面的问答对。为确保高质量，生成过程之后会进行一项新颖的自动化评估与精炼流水线，系统性地验证每个问答对的忠实性和相关性，修复或丢弃低质量条目。最终形成了一个包含超过19.5万个高质量问答对的稳健数据集。

    arXiv:2508.03000v3 Announce Type: replace  Abstract: The growing demand for corporate sustainability transparency, particularly under new regulations like the EU Taxonomy, necessitates precise data extraction from large, unstructured corporate reports, a task for which Large Language Models and Retrieval-Augmented Generation (RAG) systems require high-quality, domain-specific question-answering datasets. To address this, we introduce SustainableQA, a novel dataset and a scalable pipeline that generates comprehensive QA pairs from corporate sustainability and annual reports by integrating semantic chunk classification, a hybrid span extraction pipeline, and a specialized table-to-paragraph transformation. To ensure high quality, the generation is followed by a novel automated assessment and refinement pipeline that systematically validates each QA pair for faithfulness and relevance, repairing or discarding low-quality entries. This results in a final, robust dataset of over 195,000 div
    
[^44]: Refine-POI：用于下一个兴趣点推荐的精炼微调大型语言模型

    Refine-POI: Reinforcement Fine-Tuned Large Language Models for Next Point-of-Interest Recommendation

    [https://arxiv.org/abs/2506.21599](https://arxiv.org/abs/2506.21599)

    Refine-POI通过拓扑感知的语义ID生成和强化微调，解决了LLM在POI推荐中的语义连续性和top-k排名不足问题。

    

    摘要：arXiv:2506.21599v5 公告类型：替换-交叉 摘要：推进大型语言模型（LLMs）用于下一个兴趣点（POI）推荐任务面临两个基本挑战：（i）尽管现有方法生成包含语义信息的语义ID，但其拓扑盲索引未能保持语义连续性，这意味着ID值的接近并不反映底层语义的一致性；（ii）基于监督微调（SFT）的方法将模型输出限制为top-1预测。这些方法遭受“答案固定”问题，并因监督稀缺而忽视了对top-k排名列表和推理的需求。我们提出Refine-POI，一个通过拓扑感知ID生成和强化微调来解决这些挑战的框架。首先，我们引入一种分层自组织映射（SOM）量化策略来生成语义ID，确保码本中坐标的接近性反映语义相似性。

    arXiv:2506.21599v5 Announce Type: replace-cross  Abstract: Advancing large language models (LLMs) for the next point-of-interest (POI) recommendation task faces two fundamental challenges: (i) although existing methods produce semantic IDs that incorporate semantic information, their topology-blind indexing fails to preserve semantic continuity, meaning that proximity in ID values does not mirror the coherence of the underlying semantics; and (ii) supervised fine-tuning (SFT)-based methods restrict model outputs to top-1 predictions. These approaches suffer from "answer fixation" and neglect the need for top-k ranked lists and reasoning due to the scarcity of supervision. We propose Refine-POI, a framework that addresses these challenges through topology-aware ID generation and reinforcement fine-tuning. First, we introduce a hierarchical self-organizing map (SOM) quantization strategy to generate semantic IDs, ensuring that coordinate proximity in the codebook reflects semantic simila
    
[^45]: 释放大语言模型在稠密检索中的潜力：基于查询似然建模

    Unleashing the Power of LLMs in Dense Retrieval with Query Likelihood Modeling

    [https://arxiv.org/abs/2504.05216](https://arxiv.org/abs/2504.05216)

    本文提出LLM-QL模型，通过辅助的查询似然最大化任务增强大语言模型的稠密检索能力，利用生成优势改进对比学习。

    

    稠密检索是信息检索（IR）中的关键任务，为后续的重新排序和增强生成等下游任务提供基础。近年来，大语言模型（LLMs）展现了令人印象深刻的语义理解能力，使其成为稠密检索研究者的关注焦点。尽管LLMs作为解码器风格的生成模型在语言生成方面表现出色，但由于缺乏对后续标记的关注，它们往往在建模全局信息方面有所不足。受经典基于词的语言建模方法在IR中的启发，特别是查询似然（QL）模型，我们旨在通过QL最大化来利用LLMs的生成优势。我们不采用QL估计来进行文档排序，而是提出一个辅助任务——QL最大化，以增强骨干网络，用于后续的检索器对比学习。我们介绍了我们的模型LLM-QL，它整合了...

    arXiv:2504.05216v4 Announce Type: replace-cross  Abstract: Dense retrieval is a crucial task in Information Retrieval (IR), serving as the basis for downstream tasks such as re-ranking and augmenting generation. Recently, large language models (LLMs) have demonstrated impressive semantic understanding capabilities, making them attractive to researchers focusing on dense retrieval. While LLMs, as decoder-style generative models, excel in language generation, they often fall short in modeling global information due to a lack of attention to subsequent tokens. Drawing inspiration from the classical word-based language modeling approach for IR, specifically the query likelihood (QL) model, we aim to leverage the generative strengths of LLMs through QL maximization. Rather than employing QL estimation for document ranking, we propose an auxiliary task of QL maximization to enhance the backbone for subsequent contrastive learning of the retriever. We introduce our model, LLM-QL, which incorp
    

