# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Creating SquashFS Images to Support HPC and AI Datasets](https://arxiv.org/abs/2610.20578) | 通过对十一种创建SquashFS数据集镜像的工作流程进行基准测试，为HPC用户提供了处理包含海量小文件的只读数据集的高效方案指导，以减轻并行文件系统元数据服务器的负担。 |
| [^2] | [CASI: Credit Assignment via Set Intervention for Agentic Retrieval](https://arxiv.org/abs/2610.20293) | 该论文提出CASI方法，通过基于快照的集合干预构建反事实信息并结合Shapley聚合，解决了智能体检索中多个联合生成的检索动作的集合内信用分配问题，从而提升策略优化效果。 |
| [^3] | [HEM-Med: A Hierarchical Agentic Memory Framework for Safe Multi-Center Medication Recommendation](https://arxiv.org/abs/2610.20191) | HEM-Med提出了一种分层智能体记忆框架，通过将可迁移的跨医院临床经验与医院特异性处方模式解耦，并纳入药物相互作用等多标签安全约束，实现安全的多中心用药推荐。 |
| [^4] | [XViewRank: Overlap-Conditioned Score Fusion for LLM-Augmented Sparse Retrieval](https://arxiv.org/abs/2610.20122) | 本文提出XViewRank，一种轻量级检索端方法，利用多个视图检索结果集之间的重叠度作为条件上下文来动态融合BM25分数，在TREC DL 2019/2020上优于倒数排序融合等固定融合方法。 |
| [^5] | [Reading the Neighborhood: Text-Attentive Graph Reranking for Automated Subject Indexing](https://arxiv.org/abs/2610.20048) | 该论文提出TAGHN，一种利用叙词表邻域语义关系的异构图重排序方法进行自动主题标引，在STW经济叙词表上宏F1达到0.380，超越了微调交叉编码器和固定词表极端多标签分类器。 |
| [^6] | [Twenty Days Later: A Pre-Registered Test of History-Based Forecasts of Brand Mentions in Gemini Answers](https://arxiv.org/abs/2610.19608) | 该论文通过预注册的重复实验证明，基于历史数据对Gemini回答中品牌提及率的预测在约20天后依然准确（Brier得分0.0281，接近8小时短期预测水平且优于品牌历史基线），表明大语言模型的品牌提及计数具有足够的时间稳定性，可供实际决策使用。 |
| [^7] | [Bridging the Topological Divide: Dual-Branch Graph Distillation with Semantic Priors for Robust Cold-Start Recommendation](https://arxiv.org/abs/2610.19578) | 提出DualGCN双分支图知识蒸馏框架，通过拓扑平滑的语义先验锚定教师网络的物品表示，并以解耦的两阶段训练将知识蒸馏给语义学生分支，从而弥合语义空间与协同空间之间的拓扑鸿沟，解决冷启动推荐中的表示坍塌问题。 |
| [^8] | [Retrieve What Helps, Only When It Helps: Agentic Reinforcement Learning for Selective Retrieval-Augmented Machine Translation](https://arxiv.org/abs/2610.19528) | 该论文提出SRAT智能体强化学习框架，让大语言模型在翻译中自主学习何时需要检索外部知识、检索什么证据以及如何整合，从而只在检索真正有助于翻译质量时才进行检索。 |
| [^9] | [AdCoEvo: A Co-Evolving Multi-Agent Advertising Delivery System](https://arxiv.org/abs/2610.19454) | 提出AdCoEvo——一种协同进化的多智能体广告投放系统，实现广告运营全流程自动化，并能适应动态市场变化。 |
| [^10] | [RCER: Reconstructing Cross-Evidence Relations for Iterative Retrieval-Augmented Generation](https://arxiv.org/abs/2610.19398) | 提出RCER迭代检索增强生成方法，通过轮间关系证据获取、推理上下文重构和跨证据组合推理，直接从积累的源段落中重构实体对应关系与推理依赖关系，从而更好地回答复杂问题。 |
| [^11] | [Plan or Retrieve First? A Scale Study of Tool Discovery Architectures](https://arxiv.org/abs/2610.19068) | 该研究系统比较了工具发现中“先检索后规划”与“先规划后检索”两种架构，首次揭示了随任务复杂度变化的交叉规律：单工具查询时先检索更优，多工具查询时先规划更佳。 |
| [^12] | [Revisiting Generative vs. Dense Retrieval: A Static Value Prior Reverses the Gap](https://arxiv.org/abs/2610.19065) | 生成式检索相对稠密检索的价值优势主要源于物品流行度这一静态先验，在稠密检索中加入一步静态价值重排即可逆转差距并保持更高召回率，且工业级实验表明 140 亿参数的 SID-LLM 仍不及小 47 倍的稠密编码器。 |
| [^13] | [Computed, not Memorized: Bit-Factorized Identifiers for Efficient Generative Retrieval](https://arxiv.org/abs/2610.17467) | 提出了一种将DocID分解为比特进行计算而非记忆的新型数字文档标识符设计，使生成式检索无需重新训练即可更好地泛化到未见过的文档。 |
| [^14] | [Data Curation and Model Souping: Unlocking Qwen3-0.6B for MS MARCO Passage Ranking](https://arxiv.org/abs/2610.14816) | 通过面向语料库的持续预训练、严格数据精选、难负例挖掘及Model Soup权重合并等以数据为中心的策略，轻量级Qwen3-0.6B模型无需复杂架构修改即可在MS MARCO段落排序任务上达到最先进性能。 |
| [^15] | [Self-Retrospection Distillation: Turning Post-hoc Experiences into Prior Foresight](https://arxiv.org/abs/2610.08077) | 该论文提出前瞻学习与自我回溯蒸馏方法，将已完成的轨迹作为特权后见之明，蒸馏为同一策略在交互前的轨迹无关的预见性预测，从而在组相对奖励信号消失时仍能从经验中有效学习。 |
| [^16] | [ARAFA: An LLM-Generated Arabic Fact-Checking Dataset](https://arxiv.org/abs/2609.25833) | 研究者利用大语言模型通过三步自动化流水线构建了包含18万余个标注声明-证据对的大规模阿拉伯语事实核查数据集Arafa，有效缓解了阿拉伯语事实核查资源稀缺的问题。 |
| [^17] | [Exploring LLMs and RAG for Plausible and Explainable Material Prediction of Vehicle Components](https://arxiv.org/abs/2609.18437) | 本研究探索利用大语言模型及RAG技术预测车辆部件的合理材料，发现标准LLM生成已显著优于先前工作，而RAG方法并未带来进一步提升，并揭示了RAG系统在超参数优化、领域语料库可用性和专家评估方面的现存挑战。 |
| [^18] | [The Dice Roll Method: A Standardized Protocol for Repeated-Query Auditing of Large Language Model Brand Recommendations](https://arxiv.org/abs/2609.04047) | 本文提出并形式化了“骰子法”——一个基于温度缩放核采样生成模型的可复用标准化协议，通过将响应方差分解为多个成分并提供完整的统计分析技术栈，为大语言模型品牌推荐的重复查询审计建立了系统方法。 |
| [^19] | [RATIO: A Benchmark for Retrieval Across Typed Ideation Operations in Scientific Literature](https://arxiv.org/abs/2608.27394) | RATIO基准首次定义了三种科学构思操作（Address、Broaden、Specify）的检索任务，并利用远距离监督扩展到大规模语料库，为科学文献的灵感检索提供了新范式。 |
| [^20] | [QASP: Query-Adaptive Robust Vector Search Policy](https://arxiv.org/abs/2607.29606) | QASP通过单次预先监督回归预测每个查询的完整召回率进展曲线，从而为任意召回率目标推导出自适应搜索策略，在跨目标、索引配置和数据集泛化的同时兼顾高召回率与低计算成本。 |
| [^21] | [Gauge-Invariant, Parameter-Insensitive Regularization for Potential Recovery from Flow on Directed Graphs](https://arxiv.org/abs/2607.13609) | 该论文提出用规范不变的图狄利克雷能量替代岭正则化，从有向图流数据中恢复势能，解决了岭正则化导致的排序反转问题，并实现了跨λ四个数量级的参数不敏感性。 |
| [^22] | [Retrieval-Augmented Generation to Support Railways Engineering Tasks: A Case Study](https://arxiv.org/abs/2607.01244) | 本文通过铁路领域的案例研究，展示了从设计到部署检索增强生成（RAG）系统以查询复杂技术法规的完整工业实践经验，为受监控行业提供了一种以人为中心、平衡技术与领域专业知识的LLM文档查询实施方案。 |
| [^23] | [From Clicks to Intent: Cross-Platform Session Embeddings with LLM-Distilled Taxonomy for Financial Services Recommendations](https://arxiv.org/abs/2606.26277) | 本文提出了一种利用大语言模型提炼的分类法来生成跨平台会话嵌入，从而在金融服务推荐中实现从网页点击到用户意图的预测，并支持定量推荐与定性理解的双重目标。 |
| [^24] | [When More Documents Hurt RAG: Measuring and Reducing Vector Search Dilution with Domain-Scoped Retrieval](https://arxiv.org/abs/2606.11350) | 论文揭示了大语料库下RAG的“向量搜索稀释”问题（precision@10下降63%），并基于200个验证查询、五个LLM骨干和六个基准语料库的MASDR-RAG测试平台证明，基于元数据的领域范围限定是缓解该问题的关键组件。 |
| [^25] | [Document-as-Image Representations Fall Short for Scientific Retrieval](https://arxiv.org/abs/2604.18508) | 本文提出基于 LaTeX 源文件构建的新基准 ArXivDoc，论证“文档即图像”的表示方法不适合科学文献检索任务，因为关键证据分布于文本、表格和图表等结构化源中。 |
| [^26] | [UniRec: Bridging the Expressive Gap between Generative and Discriminative Recommendation via Chain-of-Attribute](https://arxiv.org/abs/2604.12234) | 提出UniRec框架，通过贝叶斯定理证明生成式与判别式推荐在特征完整时可相互等价，并以属性链机制在SID解码前引入类目、卖家、品牌等结构化属性词元，恢复了物品侧特征交叉能力，从而弥合两类推荐范式的表达能力鸿沟。 |
| [^27] | [Relevance Supervision Can Teach Neural Retrievers to Prefer LLM-Generated Text](https://arxiv.org/abs/2604.06163) | 本文揭示了相关性监督数据本身存在的系统性表层伪影（如流畅度和词汇特异性差异）会教会神经检索器学习“捷径”，从而诱导其产生偏好LLM生成文本的来源偏见。 |
| [^28] | [GenState-AI: State-Aware Dataset for Text-to-Video Retrieval on AI-Generated Videos](https://arxiv.org/abs/2603.14426) | 该论文提出了首个面向AI生成视频的状态感知文本到视频检索基准GenState-AI，通过提示词控制视频变体与人工验证的状态变化查询，揭示了现有检索模型难以超越静态外观、真正捕捉状态变化与最终结果的问题。 |
| [^29] | [On the Impact of Graph Neural Networks in Recommender Systems: A Topological Perspective](https://arxiv.org/abs/2512.07384) | 本论文从拓扑学视角系统解释了图神经网络在推荐系统中优于传统协同过滤方法的根本原因，指出图结构特性与GNN架构设计的相互作用是理解其性能的关键，并建立了涵盖十一种代表性GNN推荐方法的形式化分类体系。 |
| [^30] | [Ground-Truth Subgraphs for Better Training and Evaluation of Knowledge Graph Augmented LLMs](https://arxiv.org/abs/2511.04473) | 提出了SynthKGQA框架，可从任意知识图谱生成包含完整真值事实的知识图谱问答数据集，既能更有效地评估知识图谱检索器，也能用于训练更好的知识图谱增强大语言模型，并基于Wikidata构建了测试零样本泛化能力的GTSQA数据集。 |

# 详细

[^1]: 创建SquashFS镜像以支持HPC和AI数据集

    Creating SquashFS Images to Support HPC and AI Datasets

    [https://arxiv.org/abs/2610.20578](https://arxiv.org/abs/2610.20578)

    通过对十一种创建SquashFS数据集镜像的工作流程进行基准测试，为HPC用户提供了处理包含海量小文件的只读数据集的高效方案指导，以减轻并行文件系统元数据服务器的负担。

    

    HPC用户经常在HPC系统上下载数据集用于分析和处理。当这些数据集由数十万（或更多）个小文件组成时，并行访问它们会给中央并行文件系统的元数据服务器带来巨大负担。如果数据集是只读的，一个有效的解决方案是将数据集压缩成SquashFS镜像。然而，存在多种可用于创建此类数据集镜像的工作流程，我们观察到其中一些比其他的更高效。为了给我们的用户提供更有效的指导，我们对十一种不同的工作流程进行了基准测试，这些工作流程用于创建包含数千到数百万个极小文件的数据集的SquashFS数据集镜像文件。我们考虑了数据是以tar.gz、zip还是文件和目录下载的形式到达、镜像创建是在中央文件系统上还是在本地计算节点存储上进行，以及SquashFS文件压缩的数量等因素。

    arXiv:2610.20578v1 Announce Type: new  Abstract: HPC users often download datasets on HPC systems for analysis and processing. When these datasets are comprised of hundreds of thousands (or more) of small files, accessing them in parallel can put a tremendous burden on the metadata servers of central parallel file systems. If the dataset is read-only, an effective solution is to compress the dataset into a SquashFS image. There are, however, a number of workflows that can be used to create such a dataset image, and we observed that some were more efficient than others. To provide more effective guidance for our userbase, we benchmarked eleven different workflows for creating SquashFS dataset image files of datasets with thousands to millions of very small files. We took into account whether the data arrived as tar.gz, zip, or files-and-directories downloads, whether the image creation was on the central file system or local compute node storage, and the number of SquashFS file compress
    
[^2]: CASI：基于集合干预的智能体检索信用分配

    CASI: Credit Assignment via Set Intervention for Agentic Retrieval

    [https://arxiv.org/abs/2610.20293](https://arxiv.org/abs/2610.20293)

    该论文提出CASI方法，通过基于快照的集合干预构建反事实信息并结合Shapley聚合，解决了智能体检索中多个联合生成的检索动作的集合内信用分配问题，从而提升策略优化效果。

    

    大型语言模型越来越多地联合生成多个检索动作，而下游系统通常只对最终结果集合提供一个奖励。广播这一奖励虽然能为期望的集合级目标提供有效的得分函数梯度估计器，但无法确定依赖于联盟结构的集合内归因。我们提出了CASI（Credit Assignment via Set Intervention for Agentic Retrieval，基于集合干预的智能体检索信用分配），它通过基于快照的集合干预来构建缺失的反事实信息：只需生成并检索一次，保持可重放的快照组件固定，并在共同条件下评估不同的替代子集。Shapley聚合将这些边际贡献转化为集合内信用，并通过有界零和优势整形将其融入策略优化。当 m >= 4 时，单例、留一和全集端点通常不足以确定Shapley分配；在集合值可加的情况下，额外的联盟……（原文摘要在此处截断）

    arXiv:2610.20293v1 Announce Type: cross  Abstract: Large language models increasingly generate multiple retrieval actions jointly, while downstream systems often provide only one reward for the resulting set. Broadcasting this reward provides a valid score-function gradient estimator for the expected set-level objective, but does not determine a coalition-dependent within-set attribution. We introduce CASI (Credit Assignment via Set Intervention for Agentic Retrieval), which constructs the missing counterfactual information through snapshot-grounded set intervention: generate and retrieve once, hold replayable snapshot components fixed, and evaluate alternative subsets under common conditions. Shapley aggregation converts these marginals into within-set credit, which bounded zero-sum advantage shaping incorporates into policy optimization. For m >= 4, singleton, leave-one-out, and full-set endpoints do not generally determine the Shapley allocation; under additive set values, extra coa
    
[^3]: HEM-Med：一种面向安全多中心用药推荐的分层智能体记忆框架

    HEM-Med: A Hierarchical Agentic Memory Framework for Safe Multi-Center Medication Recommendation

    [https://arxiv.org/abs/2610.20191](https://arxiv.org/abs/2610.20191)

    HEM-Med提出了一种分层智能体记忆框架，通过将可迁移的跨医院临床经验与医院特异性处方模式解耦，并纳入药物相互作用等多标签安全约束，实现安全的多中心用药推荐。

    

    用药推荐（MR）是临床决策支持的关键环节。现有的用药推荐方法大多针对单中心学习设计，即仅涉及一家独立医院；而在实践中，涉及不同规模医院的多中心用药推荐需求日益增长。在多中心场景下，大型医疗中心的丰富数据可以提升低资源中心的推荐性能。尽管已有一些针对该任务的初步尝试，但仍存在两个关键缺口。缺口一：现有的多中心方法未能将可迁移的跨医院临床经验与医院特异的处方模式解耦，限制了临床可行性和医生的接受度。缺口二：这些方法普遍忽视了用药推荐的多标签安全约束，包括药物-药物相互作用风险和本地处方集可行性，而这些对于安全的真实世界部署至关重要。为解决这些缺口，我们提出了HEM-Med，一种新颖的...

    arXiv:2610.20191v1 Announce Type: cross  Abstract: Medication recommendation (MR) is critical for clinical decision support. While existing MR methods are mostly designed for single-center learning, where only one independent hospital is involved, multi-center MR with hospitals of different scales has become increasingly demanding in practice. In multi center settings, abundant data from large medical centers can improve recommendation performance for low-resource centers. Despite a few preliminary attempts on this task, two critical gaps remain. Gap 1: Existing multi-center methods fail to disentangle transferable cross-hospital clinical experience from hospital-specific prescribing patterns, limiting clinical feasibility and physician acceptance. Gap 2: They commonly overlook multi label safety constraints of MR, including drug-drug interaction risks and local formulary feasibility, which are essential for safe real-world deployment. To address these gaps, we propose HEM-Med, a novel
    
[^4]: XViewRank：面向大语言模型增强稀疏检索的重叠条件化分数融合

    XViewRank: Overlap-Conditioned Score Fusion for LLM-Augmented Sparse Retrieval

    [https://arxiv.org/abs/2610.20122](https://arxiv.org/abs/2610.20122)

    本文提出XViewRank，一种轻量级检索端方法，利用多个视图检索结果集之间的重叠度作为条件上下文来动态融合BM25分数，在TREC DL 2019/2020上优于倒数排序融合等固定融合方法。

    

    大语言模型可以通过生成查询的替代视图来缓解稀疏检索中的词汇不匹配问题。然而，当这些视图被独立检索后，其排序列表可能是互补的、冗余的，或受到查询漂移的影响，使得固定的融合方式并非最优。我们提出了XViewRank，这是一种轻量级的检索端方法，用于有条件地融合来自原始查询、假设视图和后退视图的BM25结果。对于每个汇集的候选文档，XViewRank保留三个归一化的BM25分数以及检索结果集之间的三个查询级重叠度。一个小型非线性评分器将重叠度作为上下文，用以改变对视图特定分数的解读方式，而无需联合编码查询和段落文本。在匹配输入协议下，XViewRank在TREC DL 2019和2020两个数据集上的nDCG@10和MAP指标在数值上均优于倒数排序融合（RRF），并在DL19上取得了统计显著的MAP提升。

    arXiv:2610.20122v1 Announce Type: new  Abstract: Large language models can reduce vocabulary mismatch in sparse retrieval by generating alternative views of a query. Once these views are searched independently, however, their ranked lists may be complementary, redundant, or affected by query drift, making fixed fusion suboptimal. We present XViewRank, a lightweight retrieval-side method for conditionally fusing BM25 results from an original query, a hypothesis view, and a step-back view. For each pooled candidate, XViewRank retains three normalized BM25 scores and three query-level overlaps between the retrieved result sets. A small nonlinear scorer uses the overlaps as context that changes how the view-specific scores are interpreted, without jointly encoding query and passage text. Under a matched-input protocol, XViewRank improves nDCG@10 and MAP numerically over reciprocal rank fusion on both TREC DL 2019 and 2020, with a statistically significant MAP gain on DL19. Additional analy
    
[^5]: 阅读邻域：面向自动主题标引的文本注意力图重排序

    Reading the Neighborhood: Text-Attentive Graph Reranking for Automated Subject Indexing

    [https://arxiv.org/abs/2610.20048](https://arxiv.org/abs/2610.20048)

    该论文提出TAGHN，一种利用叙词表邻域语义关系的异构图重排序方法进行自动主题标引，在STW经济叙词表上宏F1达到0.380，超越了微调交叉编码器和固定词表极端多标签分类器。

    

    主题标引是将受控词表中的描述词分配给书目记录的任务，其特点是标签数量达数千个、每个标签的正样本稀少，以及大量极少使用的描述词构成长尾分布。现有的极端多标签分类器将词表视为扁平列表，忽略了叙词表所编码的语义关系。我们提出了TAGHN，一种异构图重排序器，它在由检索候选词构成的每文档子图上运行，并用叙词表邻域进行增强。候选描述词的相关性是通过其与文档的相似度，以及候选词的叙词表邻域与文档内容的匹配程度共同估计的。一个分解的评分头将可解释的图派生特征与检索信号相结合。在包含6,198个描述词的经济学叙词表STW上，TAGHN达到0.380的宏F1，相比之下微调的交叉编码器为0.346，固定词表的极端多标签分类器为0.278。

    arXiv:2610.20048v1 Announce Type: new  Abstract: Subject indexing assigns descriptors from a controlled vocabulary to bibliographic records, a task characterized by thousands of labels, few positive examples per label, and a long tail of rarely used descriptors. Existing extreme multi-label classifiers treat the vocabulary as a flat list, ignoring the semantic relations that thesauri encode. We present TAGHN, a heterogeneous graph reranker over a per-document subgraph of retrieved candidates augmented with the thesaurus neighborhood. The relevance of a candidate descriptor is estimated jointly from its similarity with the document and also the degree to which the candidate's thesaurus neighborhood matches the document content. A decomposed scoring head combines interpretable graph-derived features with retrieval signals. On STW, an economic thesaurus of 6,198 descriptors, TAGHN reaches 0.380 macro F1 against 0.346 for a fine-tuned cross-encoder and 0.278 for a fixed-vocabulary XMLC cla
    
[^6]: 二十天后：对Gemini回答中基于历史的品牌提及预测的预注册检验

    Twenty Days Later: A Pre-Registered Test of History-Based Forecasts of Brand Mentions in Gemini Answers

    [https://arxiv.org/abs/2610.19608](https://arxiv.org/abs/2610.19608)

    该论文通过预注册的重复实验证明，基于历史数据对Gemini回答中品牌提及率的预测在约20天后依然准确（Brier得分0.0281，接近8小时短期预测水平且优于品牌历史基线），表明大语言模型的品牌提及计数具有足够的时间稳定性，可供实际决策使用。

    

    arXiv:2610.19608v1 公告类型：交叉（cross）摘要：大型语言模型提及哪些品牌的计数，只有在其能维持足够长时间以供采取行动时才有用。我们通过一项预注册的重复实验来检验这一点。相同的162个请求（54个问题，涉及三个软件类别，分别以英语、爱沙尼亚语和芬兰语提出，每个问题包含12个候选品牌中的4个，重复三次）于2026年9月22日至23日在约十小时内分三波发送给gemini-3.1-pro-preview，并于2026年10月13日（即第三波之后20.3天）以完全相同的请求体再次发送。实验计划及其决策规则在首次调用之前被冻结、哈希并注册在OSF上。结果指标是每个答案与品牌的词汇提及标签，每波648个。主要预测——每个品牌-问题单元格在之前九个答案中的提及率——取得了Brier得分0.0281和对数损失0.126的成绩，相比之下，8小时波次的六答案预测Brier得分为0.0259，注册的上限为0.0559。它优于品牌历史基线（0.1……（原文摘要在此处截断）

    arXiv:2610.19608v1 Announce Type: cross  Abstract: Counts of which brands a large language model names are useful only if they hold long enough to act on. We test this with a pre-registered replication. The same 162 requests (54 questions in English, Estonian and Finnish about three software categories, four of twelve candidate brands per question, three repeats) were sent to gemini-3.1-pro-preview in three waves within about ten hours on 22 and 23 September 2026, and again on 13 October 2026, 20.3 days after the third wave, with identical request bodies. The plan and its decision rule were frozen, hashed and registered on OSF before the first call. The outcome is a lexical mention label per answer and brand, 648 per wave. The primary forecast, each brand-and-question cell's mention rate over the nine earlier answers, scored Brier 0.0281 and log loss 0.126, against Brier 0.0259 for the six-answer forecast of the 8-hour wave and a registered ceiling of 0.0559. It beat brand history (0.1
    
[^7]: 跨越拓扑鸿沟：基于语义先验的双分支图蒸馏实现鲁棒冷启动推荐

    Bridging the Topological Divide: Dual-Branch Graph Distillation with Semantic Priors for Robust Cold-Start Recommendation

    [https://arxiv.org/abs/2610.19578](https://arxiv.org/abs/2610.19578)

    提出DualGCN双分支图知识蒸馏框架，通过拓扑平滑的语义先验锚定教师网络的物品表示，并以解耦的两阶段训练将知识蒸馏给语义学生分支，从而弥合语义空间与协同空间之间的拓扑鸿沟，解决冷启动推荐中的表示坍塌问题。

    

    图协同过滤模型在推荐系统中取得了显著的成功。然而，由于未观测物品的拓扑隔离性，这些模型在严格的冷启动场景中固有地遭受灾难性的表示坍塌问题。虽然整合多模态语义特征提供了一种潜在的补救方案，但将孤立的语义属性直接映射到全局拓扑驱动的协同空间中，往往会引发表示退化和优化冲突。为了弥合这一根本性的结构鸿沟，我们提出了DualGCN——一个创新的双分支图知识蒸馏框架。DualGCN在解耦的两阶段训练范式下运行：首先构建一个鲁棒的协同教师网络，其中物品表示由拓扑平滑的语义先验锚定，有效重塑了潜在流形；随后，语义学生分支采用残差映射（摘要在此处截断）……

    arXiv:2610.19578v1 Announce Type: cross  Abstract: Graph Collaborative Filtering models have achieved remarkable success in recommendation systems. However, they inherently suffer from a catastrophic representation collapse in strict cold-start scenarios due to the topological isolation of unobserved items. While integrating multi-modal semantic features offers a potential remedy, directly mapping isolated semantic attributes into a globally topology-driven collaborative space frequently precipitates representation degradation and optimization conflicts. To bridge this fundamental structural divide, we propose DualGCN, an innovative Dual-Branch Graph Knowledge Distillation Framework. Operating under a decoupled two-stage training paradigm, DualGCN first constructs a robust collaborative teacher network where item representations are anchored by a topology-smoothed semantic prior, effectively reshaping the latent manifold. Subsequently, a semantic student branch employs a residual mappi
    
[^8]: 只在有帮助时检索有帮助的内容：面向选择性检索增强机器翻译的智能体强化学习

    Retrieve What Helps, Only When It Helps: Agentic Reinforcement Learning for Selective Retrieval-Augmented Machine Translation

    [https://arxiv.org/abs/2610.19528](https://arxiv.org/abs/2610.19528)

    该论文提出SRAT智能体强化学习框架，让大语言模型在翻译中自主学习何时需要检索外部知识、检索什么证据以及如何整合，从而只在检索真正有助于翻译质量时才进行检索。

    

    检索外部知识可以通过提供诸如习语解释和术语等补充信息来提升大语言模型（LLM）的翻译质量。然而，现有的检索增强机器翻译系统通常对所有输入统一应用预定义的检索策略。我们认为翻译处于一种“部分证据”机制中：LLM的参数化知识足以应对许多输入，而外部证据只对其中一部分输入有益，且通常是辅助性的，而非直接提供所需的译文。这带来了两个挑战：决定何时检索以及获取何种证据。我们提出了选择性检索增强翻译（SRAT），这是一个智能体式强化学习框架，通过优化最终翻译质量，学习何时直接翻译、何时进行检索、检索什么内容，以及如何整合这些决策。为了使证据检索与下游翻译质量对齐…

    arXiv:2610.19528v1 Announce Type: new  Abstract: Retrieving external knowledge can improve large language model (LLM) translation by providing supplementary information such as idiom interpretations and terminology. However, existing retrieval-augmented machine translation systems typically apply predefined retrieval strategies uniformly across all inputs. We argue that translation operates in a partial-evidence regime: LLM's parametric knowledge suffices for many inputs, while external evidence benefits a subset and usually supports rather than directly provides the desired translation. This creates two challenges: deciding when to retrieve and what evidence to acquire. We introduce Selective Retrieval-Augmented Translation (SRAT), an agentic reinforcement learning framework that learns when to translate directly or retrieve, what to retrieve, and how to integrate these decisions by optimizing final translation quality. To align evidence retrieval with downstream translation quality r
    
[^9]: AdCoEvo：一种协同进化的多智能体广告投放系统

    AdCoEvo: A Co-Evolving Multi-Agent Advertising Delivery System

    [https://arxiv.org/abs/2610.19454](https://arxiv.org/abs/2610.19454)

    提出AdCoEvo——一种协同进化的多智能体广告投放系统，实现广告运营全流程自动化，并能适应动态市场变化。

    

    数字广告是平台触达潜在用户、驱动业务增长的重要手段，而广告投放是连接营销目标与实际收益的核心环节。现有方法在竞价、预算分配等单个任务上实现了部分自动化，但整体协调仍依赖人工操作，难以支撑大规模运营；这些方法往往也难以达到最优性能，且个体经验难以转化为可复用的运营能力。此外，随着广告市场不断变化，广告投放系统需要及时更新策略才能在动态市场中保持有效性。为了实现广告运营全流程的自动化并提升对市场变化的适应能力，我们提出了AdCoEvo，一个协同进化的多智能体广告投放系统。具体而言，该系统将不同环节的投放决策相互连接，协同进化……

    arXiv:2610.19454v1 Announce Type: cross  Abstract: Digital advertising is an important means for platforms to reach potential users and drive business growth. Advertising delivery is the core link between marketing objectives and actual returns. Existing methods have achieved partial automation in individual tasks such as bidding and budget allocation, but they still rely on human operators for overall coordination, making it difficult to support large-scale operations. They also often fail to achieve optimal performance, and individual experience remains difficult to turn into reusable operational capabilities. In addition, as advertising markets change, advertising delivery systems need timely strategy updates to remain effective in dynamic markets. To automate the entire advertising operations process and improve adaptability to market changes, we propose AdCoEvo, a co-evolving multi-agent advertising delivery system. Specifically, the system connects delivery decisions across diffe
    
[^10]: RCER：面向迭代检索增强生成的跨证据关系重构

    RCER: Reconstructing Cross-Evidence Relations for Iterative Retrieval-Augmented Generation

    [https://arxiv.org/abs/2610.19398](https://arxiv.org/abs/2610.19398)

    提出RCER迭代检索增强生成方法，通过轮间关系证据获取、推理上下文重构和跨证据组合推理，直接从积累的源段落中重构实体对应关系与推理依赖关系，从而更好地回答复杂问题。

    

    迭代检索增强生成（RAG）通过从检索到的证据中积累知识来回答复杂问题，但随着新信息的出现，现有方法难以在检索到的段落之间重构实体对应关系和推理依赖关系。我们提出了RCER，这是一种迭代RAG方法，能够直接从积累的源段落中重构这些跨证据关系。轮间关系证据获取识别实体对应关系和关系依赖，以驱动动态问题分解和迭代检索。推理上下文重构通过联合选择互补段落并推导跨证据推断，独立构建连贯的推理上下文。跨证据组合推理利用这些推断建立实体对应关系，并在所选段落之间组合关系以推导答案。实验表明

    arXiv:2610.19398v1 Announce Type: new  Abstract: Iterative retrieval-augmented generation (RAG) accumulates knowledge from retrieved evidence to answer complex questions, but existing methods struggle to reconstruct entity correspondences and inferential dependencies across retrieved passages as new information emerges. We propose RCER, an iterative RAG method that reconstructs these cross-evidence relations directly from accumulated source passages. Inter-round relational evidence acquisition identifies entity correspondences and relational dependencies to drive dynamic problem decomposition and iterative retrieval. Reasoning context reconstruction independently constructs a coherent reasoning context through joint selection of complementary passages and derivation of cross-evidence inferences. Cross-evidence compositional reasoning leverages these inferences to establish entity correspondences and compose relations across selected passages for answer derivation. Experiments show that
    
[^11]: 先规划还是先检索？工具发现架构的规模化研究

    Plan or Retrieve First? A Scale Study of Tool Discovery Architectures

    [https://arxiv.org/abs/2610.19068](https://arxiv.org/abs/2610.19068)

    该研究系统比较了工具发现中“先检索后规划”与“先规划后检索”两种架构，首次揭示了随任务复杂度变化的交叉规律：单工具查询时先检索更优，多工具查询时先规划更佳。

    

    工具增强的大语言模型（LLM）智能体在规划执行前需要从大型注册表中发现相关API，然而有限的上下文窗口无法将所有工具定义注入单个提示词中。主流方法——我们称之为“先检索后规划”（Retrieve-then-Plan, RtP）——通过嵌入用户查询并利用近似最近邻搜索来检索相似工具。我们研究了另一种“先规划后检索”（Plan-then-Retrieve, PtR）范式，其中LLM首先生成抽象工具桩（描述所需能力的假设函数签名），然后将其作为针对注册表的检索查询。我们在ToolBench（包含46,984个唯一API端点、50个类别）上，对单工具（G1）、类别内多工具（G2）和跨类别多工具（G3）查询子集进行了评估，发现存在一个清晰的随复杂度变化的交叉现象：RtP在单工具查询上优于PtR，而PtR在多工具查询上优于RtP。这一交叉现象在所有检索预算（k=1至……）下均成立。

    arXiv:2610.19068v1 Announce Type: cross  Abstract: Tool-augmented LLM agents discover relevant APIs from large registries before planning execution, yet finite context windows prohibit injecting all tool definitions into a single prompt. The dominant approach, which we term Retrieve-then-Plan (RtP), embeds the user query and retrieves similar tools via approximate nearest-neighbor search. We study an alternative Plan-then-Retrieve (PtR) paradigm in which the LLM first generates Abstract Tool Stubs (hypothesized function signatures describing needed capabilities) which then serve as retrieval queries against the registry. Evaluating on ToolBench (46,984 unique API endpoints, 50 categories) across single-tool (G1), intra-category multi-tool (G2), and cross-category multi-tool (G3) query subsets, we find a clear complexity-dependent crossover: RtP outperforms PtR on single-tool queries while PtR outperforms RtP on multi-tool queries. This crossover holds at every retrieval budget (k=1 to 
    
[^12]: 重新审视生成式检索与稠密检索：一个静态价值先验即可逆转差距

    Revisiting Generative vs. Dense Retrieval: A Static Value Prior Reverses the Gap

    [https://arxiv.org/abs/2610.19065](https://arxiv.org/abs/2610.19065)

    生成式检索相对稠密检索的价值优势主要源于物品流行度这一静态先验，在稠密检索中加入一步静态价值重排即可逆转差距并保持更高召回率，且工业级实验表明 140 亿参数的 SID-LLM 仍不及小 47 倍的稠密编码器。

    

    随着语言模型和语义物品标识符（SID）的不断改进，生成式检索（GR）已成为序列推荐中强有力的候选生成方法。然而，在在线零售候选生成场景中，GR 并非稠密检索（DR）的全面更优替代方案。我们在两个 Amazon Toys 数据集上用不同指标检验了二者的差异：在两种方法共同覆盖的用户中，GR 在最佳相关物品价值上领先，而 DR 在精确目标召回率以及排序列表整体的相关价值上领先。通过深入分析，我们发现流行度对 GR 的价值领先贡献最大。在 DR 上仅通过一步重排加入相同的静态物品价值，即可逆转这一领先，同时保持召回率高于 GR。最后，在一个使用 LLM 判定相关性的工业级内部电商数据集上，我们证明了 140 亿参数的 SID-LLM 仍不如比它小 47 倍的稠密编码器。

    arXiv:2610.19065v1 Announce Type: cross  Abstract: Generative retrieval (GR) has emerged as a strong candidate generator for sequential recommendation as language models and semantic-item identifiers have improved. However, GR is not a uniformly stronger replacement of dense retrieval (DR) in online retail candidate generation. We examine their differences on two Amazon Toys datasets under different metrics. GR leads on best relevant item value among users both methods cover, while DR leads on exact-target recall and relevant value across the ranked list. With deep analysis, we find that popularity contributes the most to GR's measured value lead. Adding the same static item value to DR in one reranking step reverses this lead and keeps recall above GR. Finally, on an industrial-scale internal e-commerce dataset with LLM-judged relevance, we show that a 14B SID-LLM still underperforms a $47\times$ smaller dense encoder.
    
[^13]: 计算而非记忆：面向高效生成式检索的位分解标识符

    Computed, not Memorized: Bit-Factorized Identifiers for Efficient Generative Retrieval

    [https://arxiv.org/abs/2610.17467](https://arxiv.org/abs/2610.17467)

    提出了一种将DocID分解为比特进行计算而非记忆的新型数字文档标识符设计，使生成式检索无需重新训练即可更好地泛化到未见过的文档。

    

    生成式检索以其高效的端到端索引和检索而闻名。特别是，数字文档标识符（DocID）通常更加高效，因为它们是在小词表上更短且通常为固定长度的序列，在束搜索期间需要更少的解码步骤。然而，它们对未见过的文档泛化能力较差，必须在索引阶段的训练中学习得到。在本文中，我们提出了一种用于生成式检索的数字DocID设计，无需重新训练即可更好地适应未见过的文档。我们的DocID设计继承了语言模型的知识：每个DocID是一个被分解为比特的token序列，每个token的得分是其各比特得分之和。与现有的数字DocID不同——其token仅通过索引训练期间记忆的嵌入来承载含义——这里的每个比特都从模型的表示中获取其含义。因此，未见过的文档的DocID……

    arXiv:2610.17467v2 Announce Type: replace  Abstract: Generative retrieval is known for its efficient end-to-end indexing and retrieval. In particular, numeric Document Identifiers (DocIDs) are generally more efficient because they are shorter and generally fixed-length sequences over a small vocabulary, requiring fewer decoding steps during beam search. However, they tend to generalize poorly to unseen documents and must be learned during indexing-stage training. In this paper, we propose a numeric DocID design for generative retrieval that enables better adaptation to unseen documents without retraining. Our DocID design inherits the language model's knowledge: each DocID is a sequence of tokens factorized into bits, and each token is scored as the sum of its bits. Unlike existing numeric DocIDs, whose tokens carry meaning only through embeddings memorized during indexing training, each bit here derives its meaning from the model's representation. Therefore, the DocIDs of unseen docum
    
[^14]: 数据精选与模型融合：释放Qwen3-0.6B在MS MARCO段落排序任务中的潜力

    Data Curation and Model Souping: Unlocking Qwen3-0.6B for MS MARCO Passage Ranking

    [https://arxiv.org/abs/2610.14816](https://arxiv.org/abs/2610.14816)

    通过面向语料库的持续预训练、严格数据精选、难负例挖掘及Model Soup权重合并等以数据为中心的策略，轻量级Qwen3-0.6B模型无需复杂架构修改即可在MS MARCO段落排序任务上达到最先进性能。

    

    本报告研究了使用轻量级Qwen3-0.6B模型在MS MARCO基准测试上的段落检索与重排序问题。在三种实验设置下——BM25先检索后重排序任务、全语料库密集检索任务以及密集检索后重排序任务——结果表明，无需复杂的架构修改、自定义损失函数或交叉编码器蒸馏，即可实现具有竞争力的排序性能。相反，显著的性能提升源于以数据为中心的策略：面向语料库的持续预训练、严格的数据精选、难负例挖掘、训练与推理的分布对齐，以及通过Model Soup（模型汤）进行的权重合并。特别值得一提的是，Model Soup在保持单模型推理效率的同时，带来了显著的性能提升。这些发现突显了数据精选和权重合并技术如何能够将十亿参数以下的模型推向最先进的性能水平。

    arXiv:2610.14816v2 Announce Type: replace  Abstract: This report studies passage retrieval and re-ranking on the MS MARCO benchmark using lightweight Qwen3-0.6B models. Across three experimental settings --- BM25-then-reranking task, corpus-wide dense retrieval task, and dense retrieval then re-ranking task --- the results demonstrate that competitive ranking performance can be achieved without complex architectural modifications, custom loss functions, or cross-encoder distillation. Instead, substantial gains stem from data-centric strategies: corpus-targeted continued pre-training, rigorous data curation, hard-negative mining, training--inference distribution alignment, and weight merging via Model Soup. In particular, Model Soup delivers marked performance gains while preserving the inference efficiency of a single model. These findings highlight how data curation and weight merging can push sub-billion-parameter models to state-of-the-art performance. My models can be downloaded fr
    
[^15]: 自我回溯蒸馏：将事后经验转化为先验预见

    Self-Retrospection Distillation: Turning Post-hoc Experiences into Prior Foresight

    [https://arxiv.org/abs/2610.08077](https://arxiv.org/abs/2610.08077)

    该论文提出前瞻学习与自我回溯蒸馏方法，将已完成的轨迹作为特权后见之明，蒸馏为同一策略在交互前的轨迹无关的预见性预测，从而在组相对奖励信号消失时仍能从经验中有效学习。

    

    具有可验证奖励的强化学习（RLVR）主要通过交互后的标量结果奖励将智能体经验转化为学习信号。然而，对于组相对目标而言，当所有采样轨迹获得相同奖励时，这一信号便会消失，即使这些轨迹可能揭示了关于任务需求以及智能体如何失败的有用信息。我们提出了一个互补的问题：事后反思能否教会智能体在行动之前本可预见的东西？我们引入前瞻学习，利用事后经验来监督交互前视角下的预见性预测，并通过自我回溯蒸馏（SRD）加以实例化。直观地说，一条已完成的轨迹揭示了本会有用的知识和本应避免的陷阱；SRD将这种特权的后见之明蒸馏到同一策略的、不依赖轨迹的前瞻预测中。前瞻仅作为训练目标，无需成为……

    arXiv:2610.08077v1 Announce Type: new  Abstract: Reinforcement learning with verifiable rewards (RLVR) turns agent experience into learning signals primarily through scalar outcome rewards after interaction. For group-relative objectives, however, this signal vanishes when all rollouts receive the same reward, even though their trajectories may reveal useful information about what the task requires and how the agent fails. We ask a complementary question: can hindsight teach an agent what it could have anticipated before acting? We introduce prospective learning, which uses post-hoc experience to supervise foresight predictions from the pre-interaction view, and instantiate it with Self-Retrospection Distillation (SRD). Intuitively, a completed trajectory reveals knowledge that would have been useful and pitfalls that should be avoided; SRD distills this privileged hindsight into trajectory-blind foresight of the same policy. Foresight serves only as a training target and need not be e
    
[^16]: ARAFA：一个由大语言模型生成的阿拉伯语事实核查数据集

    ARAFA: An LLM-Generated Arabic Fact-Checking Dataset

    [https://arxiv.org/abs/2609.25833](https://arxiv.org/abs/2609.25833)

    研究者利用大语言模型通过三步自动化流水线构建了包含18万余个标注声明-证据对的大规模阿拉伯语事实核查数据集Arafa，有效缓解了阿拉伯语事实核查资源稀缺的问题。

    

    由于数据集和资源的稀缺，自动事实核查在阿拉伯语自然语言处理领域构成了重大挑战。在本文中，我们介绍了Arafa，一个新的大规模现代标准阿拉伯语事实核查数据集，该数据集通过一个利用大语言模型（LLM）的自动化框架构建而成。数据集的构建采用三步流水线：（1）从阿拉伯语维基百科页面生成带有支持性文本证据的声明；（2）通过声明变异生成带有反驳证据的具有挑战性的反事实声明；（3）通过自动验证步骤，检验生成的声明是被其伴随证据所支持或反驳，还是证据不足以判断声明的有效性。最终得到的数据集包含181,976个声明-证据对，标注为“支持”、“反驳”或“信息不足”。在样本上进行的人工评估……（原文摘要在此处截断）

    arXiv:2609.25833v1 Announce Type: new  Abstract: Automatic fact-checking poses a significant challenge in Arabic natural language processing due to the scarcity of datasets and resources. In this manuscript, we introduce Arafa, a new large-scale dataset for fact-checking in Modern Standard Arabic, constructed through an automated framework leveraging large language models (LLMs). The dataset was constructed through a three-step pipeline: (1) claim generation from Arabic Wikipedia pages with supporting textual evidence, (2) claim mutation to generate challenging counterfactual claims with refuting evidence, and (3) an automatic validation step to validate that the generated claims are either supported or refuted by their accompanying evidence, or if the evidence does not provide enough information to judge the validity of the claims. The resulting dataset comprises 181,976 claim-evidence pairs labeled as supported, refuted, or not enough information. Human evaluation carried out on a te
    
[^17]: 探索大语言模型与检索增强生成技术在车辆部件合理且可解释材料预测中的应用

    Exploring LLMs and RAG for Plausible and Explainable Material Prediction of Vehicle Components

    [https://arxiv.org/abs/2609.18437](https://arxiv.org/abs/2609.18437)

    本研究探索利用大语言模型及RAG技术预测车辆部件的合理材料，发现标准LLM生成已显著优于先前工作，而RAG方法并未带来进一步提升，并揭示了RAG系统在超参数优化、领域语料库可用性和专家评估方面的现存挑战。

    

    在这项工作中，我们探索大语言模型（LLM）能否在无需大量微调的情况下，准确预测并解释车辆部件（如刹车盘或喷油器）的合理材料。我们测试并评估了三种方法：标准生成式LLM基线、单次检索增强生成（RAG）方法以及迭代的验证链（CoVe）变体。在检索方面，我们使用经过领域过滤的维基百科语料库这一公开可用数据。由于该任务不存在金标准，我们开发了一个定制的基于网络的标注工具，支持结构化领域专家评估的关键功能。基于LLM的生成显著优于先前的工作，而所测试的RAG方法并未进一步超越该结果。我们的研究结果揭示了基于RAG的系统仍面临的挑战：超参数优化、高质量且合法可获取的领域语料库的可用性，以及专家评估研究的设计。

    arXiv:2609.18437v1 Announce Type: new  Abstract: In this work, we explore whether LLMs can accurately predict and explain plausible materials for vehicle components such as brake discs or fuel injectors without requiring extensive fine-tuning. We test and evaluate three approaches: a standard generative LLM baseline, a single-pass Retrieval-Augmented Generation (RAG) approach, and an iterative Chain-of-Verification (CoVe) variant. For retrieval, we rely on publicly available data using a domain-filtered Wikipedia corpus. Since no gold standard exists for this task, we develop a custom web-based annotation tool supporting crucial functions for structured domain expert evaluation. LLM-based generation substantially outperforms prior work, which is not further surpassed by the tested RAG approaches. Our results surface remaining challenges for RAG-based systems: hyperparameter optimization, the availability of high-quality, legally accessible domain corpora, and expert evaluation study de
    
[^18]: 骰子法：大语言模型品牌推荐重复查询审计的标准化协议

    The Dice Roll Method: A Standardized Protocol for Repeated-Query Auditing of Large Language Model Brand Recommendations

    [https://arxiv.org/abs/2609.04047](https://arxiv.org/abs/2609.04047)

    本文提出并形式化了“骰子法”——一个基于温度缩放核采样生成模型的可复用标准化协议，通过将响应方差分解为多个成分并提供完整的统计分析技术栈，为大语言模型品牌推荐的重复查询审计建立了系统方法。

    

    背景：研究人员越来越多地使用重复相同的提示词来审计大语言模型（LLM）品牌推荐中的随机变异，然而目前尚无用于设定迭代次数、选择稳定性指标或建立可靠性阈值的标准化协议。目标：我们将骰子法形式化为一个可复用的协议，用于LLM品牌推荐的重复查询审计，该协议建立在温度缩放核采样的生成模型基础之上。方法：将总响应方差分解为采样、提示措辞、运行间和模型版本等成分。该技术栈包括：以迭代作为重复测量的负二项混合模型；作为无分布效应量的Cliff's delta；保留依赖结构的自助法；基于模拟的统计功效分析；概化理论分解；以及对固定快照的漂移诊断。我们重新分析了五项品牌推荐审计研究：约190（摘要在此处截断）。

    arXiv:2609.04047v1 Announce Type: cross  Abstract: Background: Researchers increasingly use repeated identical prompts to audit stochastic variation in large language model (LLM) brand recommendations, yet no standardized protocol exists for setting iteration counts, selecting stability metrics, or establishing reliability thresholds. Objective: We formalize the Dice Roll Method as a reusable protocol for repeated-query auditing of LLM brand recommendations, grounded in a generative model of temperature-scaled nucleus sampling. Methods: Total response variance is decomposed into sampling, prompt-phrasing, run-to-run, and model-version components. The stack: a negative-binomial mixed model with iterations as repeated measures; Cliff's delta as the distribution-free effect size; dependence-preserving bootstrap; simulation-based power; a generalizability-theory decomposition; drift diagnostics on pinned snapshots. We reanalyse five brand-recommendation auditing studies: approximately 190,
    
[^19]: RATIO：科学文献中跨类型构思操作检索的基准

    RATIO: A Benchmark for Retrieval Across Typed Ideation Operations in Scientific Literature

    [https://arxiv.org/abs/2608.27394](https://arxiv.org/abs/2608.27394)

    RATIO基准首次定义了三种科学构思操作（Address、Broaden、Specify）的检索任务，并利用远距离监督扩展到大规模语料库，为科学文献的灵感检索提供了新范式。

    

    arXiv:2608.27394v1 公告类型：新 摘要：检索到的科学文献可以为人与AI科学家提供灵感。灵感可以采取不同形式：先前的工作可能直接建议如何解决问题，或在不同抽象层次上指出方向——放大到更一般的视角或缩小到具体实现。我们引入RATIO（跨类型构思操作检索），这是一个大规模基准，其中相关性由三种操作定义，我们称之为构思动作：Address检索针对所提出问题的潜在方法，Broaden检索更一般的表述，Specify检索具体实例。RATIO是通过一种通用方法从CS文献中数百万篇全文科学论文构建而成，该方法将话语标记远距离监督——先前仅用于分类——扩展到语料库级检索，并结合了广泛的LLM和人工审核。实验表明，操作-

    arXiv:2608.27394v1 Announce Type: new  Abstract: Retrieved scientific literature can serve as inspiration for both human and AI scientists. Inspiration can take different forms: prior work may directly suggest how to address a problem, or surface directions at different levels of abstraction - zooming out to a more general view or zooming in to a concrete realization. We introduce RATIO (Retrieval Across Typed Ideation Operations), a large-scale benchmark in which relevance is defined by three operations which we name ideation moves: Address retrieves potential approaches for stated problems, Broaden retrieves more general formulations, and Specify retrieves concrete instantiations. RATIO is constructed from millions of full-text scientific papers across CS literature via a general recipe that extends discourse-marker distant supervision - previously used only for classification - to corpus-scale retrieval, combined with extensive LLM and human vetting. Experiments show that operation-
    
[^20]: QASP：查询自适应的鲁棒向量搜索策略

    QASP: Query-Adaptive Robust Vector Search Policy

    [https://arxiv.org/abs/2607.29606](https://arxiv.org/abs/2607.29606)

    QASP通过单次预先监督回归预测每个查询的完整召回率进展曲线，从而为任意召回率目标推导出自适应搜索策略，在跨目标、索引配置和数据集泛化的同时兼顾高召回率与低计算成本。

    

    向量搜索的一个根本挑战是在最小化计算成本的同时实现持续的高召回率。固定的搜索参数会导致不同查询之间显著的性能差异，而传统的基于平均召回率的评估方式掩盖了这些逐查询的差异。我们提出了QASP（查询自适应鲁棒向量搜索策略），它通过一次预先的监督回归预测每个查询的完整召回率进展曲线，并据此为任意召回率目标推导出搜索策略；这避免了搜索过程中的迭代模型调用，也无需为每个目标单独设置预测器。通过利用尺度不变特征进行搜索前推理来预测归一化召回率值，QASP能够跨召回率目标、索引配置和数据集进行泛化。细粒度的进展预测还进一步支持一种轻量级的反应式补充机制，可根据预测值与观测值之间的偏差调整搜索深度，而无需额外的……（摘要原文在此处截断）

    arXiv:2607.29606v3 Announce Type: replace-cross  Abstract: A fundamental challenge of vector search is achieving consistently high recall while minimizing computational costs. Fixed search parameters cause significant performance variance across queries, and conventional evaluation on average recall masks these per-query disparities. We introduce QASP (Query-Adaptive robust vector Search Policy), which predicts the complete recall progression curve per query via a single upfront supervised regression, from which a search policy is derived for any recall target; this avoids iterative model invocations during search or separate predictors per target. By predicting normalized recall values with scale-invariant features and pre-search inference, QASP generalizes across recall targets, index configurations, and datasets. The fine-grained progress predictions further enable a lightweight reactive complement that adjusts search depth based on predicted-versus-observed deviations without addit
    
[^21]: 有向图上基于流恢复势能的规范不变、参数不敏感正则化方法

    Gauge-Invariant, Parameter-Insensitive Regularization for Potential Recovery from Flow on Directed Graphs

    [https://arxiv.org/abs/2607.13609](https://arxiv.org/abs/2607.13609)

    该论文提出用规范不变的图狄利克雷能量替代岭正则化，从有向图流数据中恢复势能，解决了岭正则化导致的排序反转问题，并实现了跨λ四个数量级的参数不敏感性。

    

    从有向图上观测到的流中恢复潜在势能（即带狄利克雷边界的离散泊松问题）是一个不适定问题，而标准的解决方案反而适得其反：岭正则化会向规范无意义的原点收缩，导致恢复的排序发生坍塌和反转（相对于植入的真实值，秩相关从+0.81变为-0.42）。规范不变的图狄利克雷能量消除了这一风险，并带来了参数不敏感性：估计值在λ的四个数量级范围内保持稳定，而岭正则化在所有λ>0的情况下都会反转排序。作者证明了约化求解是SPD（对称正定）的，并在岭正则化使其坍塌的地方精确保存了动态范围，还通过泊松残差仅从流中定位吸收边界。H^1半范数本身是经典的；新颖之处在于规范诊断、由此获得的参数不敏感性，以及表明结果对提取方法稳健的消融实验。在三个公开的点击流数据集上……（摘要被截断）

    arXiv:2607.13609v2 Announce Type: replace  Abstract: Recovering a latent potential from observed flow on a directed graph (a discrete Poisson problem with Dirichlet boundaries) is ill-posed, and the standard fix backfires: ridge regularization shrinks toward a gauge-meaningless origin, collapsing and reversing the recovered ordering ($+0.81\to-0.42$ rank correlation against a planted ground truth). The gauge-invariant graph Dirichlet energy removes the hazard and delivers parameter-insensitivity: the estimate is stable across four orders of magnitude in $\lambda$, whereas ridge inverts the ordering for every $\lambda>0$. We prove the reduced solve is SPD and preserves dynamic range exactly where ridge collapses it, and localize absorbing boundaries from flow alone via a Poisson residual. The $H^1$ seminorm is classical; what is new is the gauge diagnosis, the parameter-insensitivity it buys, and an ablation showing the result is robust to the extraction method. On three public clickstr
    
[^22]: 支持铁路工程任务的检索增强生成：一项案例研究

    Retrieval-Augmented Generation to Support Railways Engineering Tasks: A Case Study

    [https://arxiv.org/abs/2607.01244](https://arxiv.org/abs/2607.01244)

    本文通过铁路领域的案例研究，展示了从设计到部署检索增强生成（RAG）系统以查询复杂技术法规的完整工业实践经验，为受监控行业提供了一种以人为中心、平衡技术与领域专业知识的LLM文档查询实施方案。

    

    技术法规数量的不断增长及其日益复杂性，对所有受监管行业的从业者构成了重大挑战。本文描述了一个案例研究，涵盖从设计到部署的全过程，即构建一个用于查询铁路领域复杂技术法规的检索增强生成（RAG）系统。尽管该系统是为铁路行业开发的，但这一工业实践经验对于监管合规性以及从复杂文档中准确检索信息为基本需求的技术领域具有特殊价值。同时，它还提供了一种以人为中心的方法，用于在各受监控行业中实施基于大语言模型的技术文档查询服务，并实现了技术能力与领域专业知识之间的平衡。

    arXiv:2607.01244v2 Announce Type: replace  Abstract: The growing number and complexity of technical regulations represent an important challenge for all professionals in regulated industries. This paper describes a case study, from design to deployment, of building a Retrieval-Augmented Generation system for the consultation of complex technical regulations in the railway domain. Although developed for the railway sector, this testimony of an industrial experience is of particular value for technical domains where regulatory compliance and accurate information retrieval from complex documentation are essential requirements. It also constitutes a human-centered approach for implementing LLM-powered technical documentation consultation across various regulated industries, balancing technological capabilities with domain expertise.
    
[^23]: 从点击到意图：基于大语言模型提炼分类法的跨平台会话嵌入在金融服务推荐中的应用

    From Clicks to Intent: Cross-Platform Session Embeddings with LLM-Distilled Taxonomy for Financial Services Recommendations

    [https://arxiv.org/abs/2606.26277](https://arxiv.org/abs/2606.26277)

    本文提出了一种利用大语言模型提炼的分类法来生成跨平台会话嵌入，从而在金融服务推荐中实现从网页点击到用户意图的预测，并支持定量推荐与定性理解的双重目标。

    

    顺序用户行为建模在工业推荐系统中被广泛采用；然而，在金融服务领域仍存在显著差距，因为登录前的网页交互与认证后的应用内体验截然不同。具体而言，登录前的网页用户通常探索新产品，而登录后的应用用户则专注于账户服务。由于跨渠道实体解析的挑战（例如，将匿名网页会话匹配到已认证的移动账户），基于网页的意图信号在认证后个性化中仍未得到充分利用。现有捕获网页意图的方法通常是临时且狭窄的，缺乏灵活性来同时支持定量下游推荐和定性的大规模理解。在这项工作中，我们提出了一个可扩展且双用途的意图预测框架，用于基于网页的交互，并展示了其在个性化中的适用性。

    arXiv:2606.26277v1 Announce Type: cross  Abstract: Sequential user behavior modeling is widely adopted in industrial recommender systems; however, significant gaps remain in financial services, where pre-login web interactions and authenticated in-app experiences differ drastically. Specifically, pre-login web users typically explore new products, whereas logged-in app users focus on account servicing. Due to the challenge of cross-channel entity resolution (e.g., matching anonymous web sessions to authenticated mobile accounts), web-based intent signals remain underutilized for post-authentication personalization. Existing methods for capturing web-based intent are often ad-hoc and narrow, lacking the flexibility to support both quantitative downstream recommendations and qualitative understanding at scale. In this work, we propose a scalable and dual-purpose intent prediction framework for web-based interactions and demonstrate its applicability for personalization. Our approach tran
    
[^24]: 当更多文档反而损害RAG：通过领域范围检索测量并减少向量搜索稀释

    When More Documents Hurt RAG: Measuring and Reducing Vector Search Dilution with Domain-Scoped Retrieval

    [https://arxiv.org/abs/2606.11350](https://arxiv.org/abs/2606.11350)

    论文揭示了大语料库下RAG的“向量搜索稀释”问题（precision@10下降63%），并基于200个验证查询、五个LLM骨干和六个基准语料库的MASDR-RAG测试平台证明，基于元数据的领域范围限定是缓解该问题的关键组件。

    

    检索增强生成（RAG）在应用于大型异构文档集合时会出现性能退化。在RAG系统中，稠密相似度的区分性下降，导致top-k检索返回语义相似但上下文错误的文本块。我们将这种失败模式称为向量搜索稀释（vector search dilution）。即使在混合稠密+稀疏检索下，我们首先在实际部署的怀俄明州交通部（WYDOT）系统中观察到了这一现象；在对照实验中，语料库规模从54个文档增长到1,128个文档，使precision@10下降了63%，包含答案文档的召回率下降了23%。在其他六个州交通部（DOT）中也观察到了类似模式。为了诊断这一问题，我们引入了MASDR-RAG（面向RAG的元数据范围领域检索），作为一个受控测试平台，涵盖200个经人工验证的查询、五个LLM骨干模型、六个公开基准语料库和两种索引技术栈。在MASDR-RAG中，领域范围限定是起关键支撑作用的组件。我们发现硬分区

    arXiv:2606.11350v2 Announce Type: replace  Abstract: Retrieval-augmented generation (RAG) degrades when applied to large, heterogeneous document collections. In RAG systems, dense similarity becomes less discriminative, causing top-$k$ retrieval to return semantically similar but contextually incorrect chunks. We refer to this failure mode as vector search dilution. Even under hybrid dense+sparse retrieval, we first observed it in a deployed Wyoming Department of Transportation (WYDOT) system; in a controlled experiment, growing the corpus from 54 to 1,128 documents cut precision@10 by 63% and recall of the answer-bearing document by 23%. Similar patterns were observed in six other DOTs. To diagnose this issue, we introduce MASDR-RAG (Metadata-Scoped Domain Retrieval for RAG) as a controlled testbed over 200 human-validated queries, five LLM backbones, six public benchmark corpora, and two index stacks. In MASDR-RAG, domain scoping is the load-bearing component. We found hard partition
    
[^25]: 图像化文档表示在科学文献检索中的不足

    Document-as-Image Representations Fall Short for Scientific Retrieval

    [https://arxiv.org/abs/2604.18508](https://arxiv.org/abs/2604.18508)

    本文提出基于 LaTeX 源文件构建的新基准 ArXivDoc，论证“文档即图像”的表示方法不适合科学文献检索任务，因为关键证据分布于文本、表格和图表等结构化源中。

    

    许多近期的文档嵌入模型基于“文档即图像”的表示进行训练，即将渲染后的页面作为图像而非底层源文件进行嵌入。与此同时，现有的科学文档检索基准（如 ArXivQA 和 ViDoRe）将文档视为页面的图像，从而隐性地偏向了这类表示方法。在本工作中，我们认为这种范式并不适合文本丰富的多模态科学文档，因为在这类文档中，关键证据分布在包括文本、表格和图表在内的结构化源文件中。为了研究这一场景，我们提出了 ArXivDoc，一个基于科学论文底层 LaTeX 源文件构建的新基准。与 PDF 或基于图像的表示不同，LaTeX 提供了对结构化元素（如章节、表格、图表和公式）的直接访问，从而支持基于特定证据类型的受控查询构建。我们系统地比较了纯文本、基于图像的表示方法

    arXiv:2604.18508v2 Announce Type: replace-cross  Abstract: Many recent document embedding models are trained on document-as-image representations, embedding rendered pages as images rather than the underlying source. Meanwhile, existing benchmarks for scientific document retrieval, such as ArXivQA and ViDoRe, treat documents as images of pages, implicitly favoring such representations. In this work, we argue that this paradigm is not well-suited for text-rich multimodal scientific documents, where critical evidence is distributed across structured sources, including text, tables, and figures. To study this setting, we introduce ArXivDoc, a new benchmark constructed from the underlying LaTeX sources of scientific papers. Unlike PDF or image-based representations, LaTeX provides direct access to structured elements (e.g., sections, tables, figures, equations), enabling controlled query construction grounded in specific evidence types. We systematically compare text-only, image-based, and
    
[^26]: UniRec：通过属性链弥合生成式与判别式推荐之间的表达能力鸿沟

    UniRec: Bridging the Expressive Gap between Generative and Discriminative Recommendation via Chain-of-Attribute

    [https://arxiv.org/abs/2604.12234](https://arxiv.org/abs/2604.12234)

    提出UniRec框架，通过贝叶斯定理证明生成式与判别式推荐在特征完整时可相互等价，并以属性链机制在SID解码前引入类目、卖家、品牌等结构化属性词元，恢复了物品侧特征交叉能力，从而弥合两类推荐范式的表达能力鸿沟。

    

    生成式推荐（GR）将检索和排序重新框定为对语义标识符（SIDs）的自回归解码，将多阶段流程统一到单一模型中。然而，一个根本性的表达能力鸿沟依然存在：判别式模型通过直接访问特征对物品打分，从而实现显式的用户-物品交叉，而生成式推荐则在不具备物品侧信号的紧凑SID词元上进行解码。我们通过贝叶斯定理对这一差距进行了形式化：按 p(y|f,u) 排序等价于按 p(f|y,u) 排序，后者可在物品特征上进行自回归分解，这表明具有完整特征访问能力的生成式模型可与其判别式对应模型相媲美，任何实际差距都仅源于特征覆盖的不完整。我们提出了UniRec，其核心机制为属性链。CoA在解码SID之前，为每个SID序列前置结构化的属性词元：类目、卖家、品牌，从而恢复了判别式模型所具备的物品侧特征交叉能力。

    arXiv:2604.12234v5 Announce Type: replace  Abstract: Generative Recommendation (GR) reframes retrieval and ranking as autoregressive decoding over Semantic IDs (SIDs), unifying the multi-stage pipeline into a single model. Yet a fundamental expressive gap persists: discriminative models score items with direct feature access enabling explicit user-item crossing, whereas GR decodes over compact SID tokens without item-side signal. We formalize this via Bayes' theorem: ranking by p(y|f,u) is equivalent to ranking by p(f|y,u), which factorizes autoregressively over item features, showing that a generative model with full feature access matches its discriminative counterpart, with any practical gap stemming solely from incomplete feature coverage. We propose UniRec with Chain-of-Attribute (CoA) as its core mechanism. CoA prefixes each SID sequence with structured attribute tokens:category, seller, brand, before decoding the SID, recovering the item-side feature crossing that discriminative
    
[^27]: 相关性监督可以教会神经检索器偏好LLM生成的文本

    Relevance Supervision Can Teach Neural Retrievers to Prefer LLM-Generated Text

    [https://arxiv.org/abs/2604.06163](https://arxiv.org/abs/2604.06163)

    本文揭示了相关性监督数据本身存在的系统性表层伪影（如流畅度和词汇特异性差异）会教会神经检索器学习“捷径”，从而诱导其产生偏好LLM生成文本的来源偏见。

    

    近期研究表明，神经检索器常常表现出来源偏见，会将LLM生成的段落排在语义相似的人类撰写段落之上，这威胁到了现代信息访问系统的可靠性。虽然这种偏见通常被视为模型层面的固有属性，我们证明了相关性监督本身就能诱导这种偏见。实证结果显示，来源偏见在经过相关性监督训练的检索器中最为明显，且相同的编码器在经过相关性微调后会变得更加偏好LLM生成的文本。我们将这一效应追溯到监督检索数据中的表层伪影：标注的正例和负例在流畅度、词汇特异性等属性上存在系统性差异，而这些差异与LLM生成文本和人类撰写文本之间的差异高度相似。因此，检索器可以学会利用基于伪影的捷径来判断相关性。我们进一步从理论上证明，在对比学习下，能够可靠区分正例与负例的伪影……（原文摘要在此处截断）

    arXiv:2604.06163v2 Announce Type: replace  Abstract: Recent studies show that neural retrievers often exhibit source bias, ranking LLM-generated passages above semantically similar human-written ones, which threatens the reliability of modern information access systems. While this bias is often viewed as a model-level property, we show that relevance supervision itself can induce it. Empirically, source bias is clearest in relevance-supervised retrievers, and the same encoders become more LLM-preferring after relevance fine-tuning. We trace this effect to surface-level artifacts in supervised retrieval data: annotated positives and negatives differ systematically in properties such as fluency and lexical specificity, and these differences parallel those between LLM-generated and human-written passages. As a result, retrievers can learn artifact-based shortcuts for relevance. We further show theoretically that, under contrastive learning, artifacts that reliably separate positives from 
    
[^28]: GenState-AI：面向AI生成视频的文本到视频检索状态感知数据集

    GenState-AI: State-Aware Dataset for Text-to-Video Retrieval on AI-Generated Videos

    [https://arxiv.org/abs/2603.14426](https://arxiv.org/abs/2603.14426)

    该论文提出了首个面向AI生成视频的状态感知文本到视频检索基准GenState-AI，通过提示词控制视频变体与人工验证的状态变化查询，揭示了现有检索模型难以超越静态外观、真正捕捉状态变化与最终结果的问题。

    

    现有的文本到视频检索基准往往允许模型仅凭静态外观完成匹配，因此难以判断模型是否真正捕捉到了状态变化和最终结果。我们提出了GenState-AI，这是一个包含956个AI生成视频和195个经人工验证的状态变化查询的基准，涵盖玩具、卡通、家居和操作四个领域。其构建方法采用提示词控制的视频变体，分别针对相互冲突的时间/状态结果或语义替换。经过人工验证后，所有有效视频构成子集级检索池，而完整的查询特定三元组则支持诊断性的难负样本排序。我们使用标准Recall@K指标和基于三元组的一致性指标对VCF-Lik、Qwen3-VL和LanguageBind进行了评估。结果显示，Qwen3-VL（8B）在平均R@1和三元组一致性上表现最佳，而LanguageBind获得了最高的语义判别准确率。然而，更强的检索表现……（摘要在此处截断）

    arXiv:2603.14426v2 Announce Type: replace-cross  Abstract: Existing text-to-video retrieval benchmarks often permit matching from static appearance, making it difficult to determine whether models capture state changes and final outcomes. We introduce GenState-AI, a benchmark comprising 956 AI-generated videos and 195 verified state-change queries across Toy, Cartoon, Home, and Manipulation domains. Its construction uses prompt-controlled variants that target either conflicting temporal/state outcomes or semantic substitutions. Following human verification, all valid videos contribute to subset-level retrieval pools, while complete query-specific triplets support diagnostic hard-negative ordering. We evaluate VCF-Lik, Qwen3-VL, and LanguageBind using both standard Recall@K and triplet-based consistency metrics. Qwen3-VL (8B) achieves the best macro-average R@1 and triplet consistency, while LanguageBind obtains the highest semantic discrimination accuracy. However, stronger retrieval d
    
[^29]: 论图神经网络在推荐系统中的影响：一个拓扑学视角

    On the Impact of Graph Neural Networks in Recommender Systems: A Topological Perspective

    [https://arxiv.org/abs/2512.07384](https://arxiv.org/abs/2512.07384)

    本论文从拓扑学视角系统解释了图神经网络在推荐系统中优于传统协同过滤方法的根本原因，指出图结构特性与GNN架构设计的相互作用是理解其性能的关键，并建立了涵盖十一种代表性GNN推荐方法的形式化分类体系。

    

    在推荐系统中，用户与物品之间的交互可以被建模为一个二部图，其中用户节点和物品节点通过无向边相连。这种基于图的视角推动了图神经网络（GNN）的快速应用，GNN的表现通常优于协同过滤（CF）方法，例如潜在因子模型、深度神经网络和生成式策略。然而，尽管GNN在实证上取得了成功，其相对于其他协同过滤方法具有系统性优势的原因仍未被充分理解。本专著提出了一个以拓扑为中心的GNN推荐研究视角。我们认为，要全面理解这些模型的性能，必须考虑用户-物品图的结构特性及其与GNN架构设计之间的相互作用。为支持这一观点，我们引入了一个形式化的分类体系，提炼了十一种代表性GNN推荐方法中的共同建模模式。

    arXiv:2512.07384v2 Announce Type: replace  Abstract: In recommender systems, user-item interactions can be modeled as a bipartite graph, where user and item nodes are connected by undirected edges. This graph-based view has motivated the rapid adoption of graph neural networks (GNNs), which often outperform collaborative filtering (CF) methods such as latent factor models, deep neural networks, and generative strategies. Yet, despite their empirical success, the reasons why GNNs offer systematic advantages over other CF approaches remain only partially understood. This monograph advances a topology-centered perspective on GNN-based recommendation. We argue that a comprehensive understanding of these models' performance should consider the structural properties of user-item graphs and their interaction with GNN architectural design. To support this view, we introduce a formal taxonomy that distills common modeling patterns across eleven representative GNN-based recommendation approaches
    
[^30]: 用于更好训练和评估知识图谱增强大语言模型的真值子图

    Ground-Truth Subgraphs for Better Training and Evaluation of Knowledge Graph Augmented LLMs

    [https://arxiv.org/abs/2511.04473](https://arxiv.org/abs/2511.04473)

    提出了SynthKGQA框架，可从任意知识图谱生成包含完整真值事实的知识图谱问答数据集，既能更有效地评估知识图谱检索器，也能用于训练更好的知识图谱增强大语言模型，并基于Wikidata构建了测试零样本泛化能力的GTSQA数据集。

    

    从图结构知识库中检索信息是提高大语言模型事实性的一个有前景的方向。尽管已提出多种解决方案，但由于缺乏带有图检索真值目标的具有挑战性的问答数据集，方法之间的比较十分困难。我们提出了SynthKGQA，这是一个由大语言模型驱动的框架，可以从任何知识图谱生成高质量的知识图谱问答数据集，并提供知识图谱中用于推理问题的完整真值事实集合。我们展示了SynthKGQA生成的数据除了能够对知识图谱检索器进行更有信息量的基准测试外，还能用于训练更好的模型。我们将SynthKGQA应用于Wikidata，生成了GTSQA——一个旨在测试知识图谱检索器对未见过的图结构和关系类型的零样本泛化能力的新数据集，并在其上对流行的知识图谱增强大语言模型解决方案进行了基准测试。

    arXiv:2511.04473v3 Announce Type: replace-cross  Abstract: Retrieval of information from graph-structured knowledge bases represents a promising direction for improving the factuality of LLMs. While various solutions have been proposed, a comparison of methods is difficult due to the lack of challenging QA datasets with ground-truth targets for graph retrieval. We present SynthKGQA, an LLM-powered framework for generating high-quality Knowledge Graph Question Answering datasets from any Knowledge Graph, providing the full set of ground-truth facts in the KG to reason over questions. We show how, in addition to enabling more informative benchmarking of KG retrievers, the data produced with SynthKGQA also allows us to train better models.We apply SynthKGQA to Wikidata to generate GTSQA, a new dataset designed to test zero-shot generalization abilities of KG retrievers with respect to unseen graph structures and relation types, and benchmark popular solutions for KG-augmented LLMs on it.
    

