# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SpeakerMem-R1: Speaker-Centered Dual-Track Memory for Multi-Party Dialogue](https://arxiv.org/abs/2609.26780) | 提出SpeakerMem-R1，一种以说话人为中心的双轨记忆框架，通过存储带说话人标签的逐字消息和个人/群体层面的衍生状态，解决多方对话中的消息归属、关系理解与状态重建两大瓶颈。 |
| [^2] | [Discovery-Driven Integration of Disjoint Tables via Text](https://arxiv.org/abs/2609.26658) | 该论文提出LOKI架构，将“文本介导的连接路径发现”形式化为新任务，通过全局表-文本对比学习实现数据湖中缺乏连接属性的无关联表的细粒度集成。 |
| [^3] | [When Does Permutation Instability Generalize? Independent-View Validation for Listwise LLM Reranking](https://arxiv.org/abs/2609.26251) | 该研究通过精确的有限视图分解和独立视图对比实验发现，列表式LLM重排序中有限视图不稳定性分数对未见排列的“预测力”很大程度上源于共享测量造成的部分-整体关联伪影，在完全不相交的视图间相关性很弱或呈异质性。 |
| [^4] | [Which Reranking Conclusions Survive the Answer Interface? A Prospective Finite-Orbit Audit](https://arxiv.org/abs/2609.26250) | 该研究通过前瞻性审计发现，仅改变语言模型阅读器的答案界面（如标签绑定、词汇选择和选项顺序）不会实质性地改变BM25与BGE-v2-m3重排序对比的结论，六个确认性设置均未出现超过预设阈值的界面效应或排序反转。 |
| [^5] | [ABAI at COLIEE 2026 Task 1: Multi-Stage Retrieval with GraphRAG-Enhanced Meta-Learning, and a Post-Hoc Study of the Cross-Validation-to-Test Gap](https://arxiv.org/abs/2609.26237) | 本文提出了一种用于COLIEE 2026判例法检索任务的四阶段多阶段检索流程（多视角BM25、神经重排序、图特征与LightGBM元学习），并通过控制实验系统性地分析了官方测试集F1（0.177）远低于交叉验证结果（0.311）的原因。 |
| [^6] | [A Semantic Approach to the Academic Publishing Network: Document Vector Representations and Hybrid Structural-Semantic Fusion over OpenAlex Data](https://arxiv.org/abs/2609.26218) | 该论文在学术出版网络结构图分析的基础上引入语义层，利用SPECTER2引用感知文档嵌入和权重可调的结构-语义后期融合函数，实现了比TF-IDF基线更契合专家主题分类的文档表示，并提升了推荐效果。 |
| [^7] | [When Concealed Links Cannot Be Recovered: A Structural Identifiability Bound and Evaluation Pitfalls in Offshore Leak Networks](https://arxiv.org/abs/2609.26171) | 该论文提出了一个无分布假设的结构可辨识性界限，证明对于任何遵循图同构的链接预测方法（包括拓扑评分和消息传递图神经网络），离岸泄露网络中被隐藏的孤立节点与其结构孪生节点不可区分，其隐藏关系（如受益所有权）无法以高于随机猜测的概率被恢复，且恢复上限由节点的结构不可区分类大小决定。 |
| [^8] | [TailSpec-EASE: Knowledge-Graph-Regularized Linear Recommendation for Web Long-Tail Discovery](https://arxiv.org/abs/2609.26143) | 提出TailSpec-EASE，一种轻量级线性推荐模型，通过将关系感知的谱知识图谱先验注入局部闭式重构目标，并使先验强度随物品流行度自适应调整，在保持整体精度的同时显著改善长尾物品的推荐效果。 |
| [^9] | [CoVeR: Coverage-Based Routing of Verifier Calls in Agentic Retrieval](https://arxiv.org/abs/2609.26086) | CoVeR 利用冻结句子嵌入的覆盖度边际作为单一阈值门控，仅在证据状态模糊时才调用 LLM 验证器，从而在保持答案准确率几乎不变的前提下，大幅削减智能体检索中验证器反复处理证据的调用成本。 |
| [^10] | [Knowledge-as-Skill: A Structural Design for Autonomous Knowledge-Base Use by LLM Agents](https://arxiv.org/abs/2609.25991) | 提出“知识即技能”三层知识库组织方案——以 SKILL.md 为核心的发现层、逐目录 index.md 的导航层以及带 YAML 元数据的文档知识层——使知识库对 LLM 智能体而言可发现、可导航且自描述，从而支持智能体自主决定是否检索及如何使用知识。 |
| [^11] | [ARAFA: An LLM-Generated Arabic Fact-Checking Dataset](https://arxiv.org/abs/2609.25833) | 研究者利用大语言模型通过三步自动化流水线构建了包含18万余个标注声明-证据对的大规模阿拉伯语事实核查数据集Arafa，有效缓解了阿拉伯语事实核查资源稀缺的问题。 |
| [^12] | [Robust Fusion of Semantic and Behavioural Signals for LLM Reranking in Personalised Search](https://arxiv.org/abs/2609.25825) | 提出确定性双样本特征丢弃训练方法，使基于LLM的个性化搜索重排序系统能够鲁棒地融合行为统计特征（QSS）与语义信号，避免捷径学习，即使行为特征缺失时仍能保持良好的排序性能。 |
| [^13] | [From Offline Proxies to Online Decisions: A Layered Engagement Evaluation Framework for Conversational AI](https://arxiv.org/abs/2609.25408) | 该论文提出了一个分层参与度评估框架，将离线代理指标建模为三重对齐链条，并通过区间感知的决策一致性协议审计其与在线A/B实验结果的一致性，从而在不消耗线上流量的情况下预测对话式AI改动是否值得上线。 |
| [^14] | [ReFilter: Bridging Embeddings and LLM Filtering for Similar Mobile App Retrieval](https://arxiv.org/abs/2609.25306) | 提出ReFilter混合框架，先用嵌入检索语义相关的候选应用，再用大语言模型进行上下文过滤，以高精度识别功能相似的移动应用，F1分数达90%。 |
| [^15] | [GroundedGEO: Auditing the Evidence Gap in Generative Search Rankings](https://arxiv.org/abs/2609.25189) | 该论文提出证据配对基准与主张级重排序器GroundedGEO，首次揭示生成式搜索中“证据状态是主张-证据关系而非文本属性”的可识别性差距，并实证表明缺乏证据支持的内容丰富文本能显著操纵冻结排序器的排名。 |
| [^16] | [From Ranked Documents to Reliable Contexts: An Answer-Oriented Context Construct Framework for AI Search](https://arxiv.org/abs/2609.23354) | 该论文提出一个面向AI搜索的答案导向上下文构建三阶段框架（答案支持、内容可信度、上下文组织），将检索目标从传统的文档排序转变为为正确答案生成构建可靠上下文。 |
| [^17] | [Parameterized Dense-Sparse Fusion for Hybrid Retrieval: Tuning a Rank-Score Mix on BEIR SciFact with Qdrant](https://arxiv.org/abs/2609.22770) | 该论文提出了一种显式参数化的稠密-稀疏混合检索融合方法，通过在BEIR SciFact上网格搜索调整排名-分数混合参数（α=0.8, λ=0.75, κ=20），使nDCG@10达到0.753，优于稠密BGE和等权重RRF基线，同时证明这些参数具有数据集特异性。 |
| [^18] | [Benchmark Radar: A Living Database and Search Engine for AI Benchmarks and Evaluation](https://arxiv.org/abs/2609.11115) | 本文提出Benchmark Radar，一个每日自动发现并整合AI基准论文、数据集和代码的动态数据库与搜索引擎，为LLM评估、智能体、编程、推理和安全等领域提供可检索的基准目录、来源引用和分数历史。 |
| [^19] | [Plan Pointers and Record-Directive Form in Budgeted Verification of Inherited Agent Memory](https://arxiv.org/abs/2609.03450) | 该论文通过十二项注册研究发现，写入智能体记忆库的指令形式（准则、裸ID或指针）会以高度模型依赖的方式显著影响预算受限下的记录选择，长度匹配准则可带来35分的提升，但附加ID可能完全抵消准则的效果。 |
| [^20] | [Query-Side Attacks on GNN-Based KGQA: Tracing Failures from Entity Linking to Answer Generation](https://arxiv.org/abs/2608.25922) | 本文通过阶段隔离协议发现，基于GNN的知识图谱问答系统的主要脆弱性在于子图构建阶段，而非GNN推理阶段，这挑战了现有鲁棒性评估的假设。 |
| [^21] | [GreekBarRetrieval: A Benchmark for Greek Statutory Retrieval](https://arxiv.org/abs/2608.18752) | 本文提出了希腊法律条文检索基准GreekBarRetrieval，并通过实验发现LLM查询重构能显著提升BM25与密集检索的性能差距。 |
| [^22] | [WebArxiv: A Reproducible Benchmark for Evaluating Multimodal Web Agents on arXiv Tasks](https://arxiv.org/abs/2507.00938) | WebArxiv是一个基于arXiv静态快照构建的可复现基准，包含510个具有确定性答案的时间不变任务，用于评估多模态网络智能体在多约束论文检索、细粒度内容提取和跨论文比较等学术任务上的能力。 |

# 详细

[^1]: SpeakerMem-R1：面向多方对话的以说话人为中心的双轨记忆

    SpeakerMem-R1: Speaker-Centered Dual-Track Memory for Multi-Party Dialogue

    [https://arxiv.org/abs/2609.26780](https://arxiv.org/abs/2609.26780)

    提出SpeakerMem-R1，一种以说话人为中心的双轨记忆框架，通过存储带说话人标签的逐字消息和个人/群体层面的衍生状态，解决多方对话中的消息归属、关系理解与状态重建两大瓶颈。

    

    多方场景下的长期对话记忆不仅仅是从长期对话中检索相关内容：它必须区分谁说了什么、每句话涉及谁、个体之间如何看待彼此、哪些信息为群体所共享，以及状态如何随时间变化。最近针对多方对话基准的研究表明，现有的通用大语言模型记忆系统往往丢失人物与群体关系，或难以整合分布在成员、群体和时间中的线索。这些问题共同揭示了两个核心瓶颈：多方对话中的消息归属与关系理解，以及从交错历史中进行的状态重建。为解决这两个问题，我们提出了 SpeakerMem-R1：其双轨记忆存储带有说话人标签的逐字消息以及衍生状态，并将它们组织为个人层面和群体层面的视图，随后按实体、事件等方式结合两条轨道的证据（摘要在此处被截断）。

    arXiv:2609.26780v1 Announce Type: new  Abstract: Long-term conversational memory in multi-party settings requires more than retrieving relevant content from long-term conversations: it must distinguish who said what, whom each statement concerns, how individuals perceive one another, what information is shared by the group, and how states change over time. Recent studies on multi-party dialogue benchmarks show that existing general-purpose LLM memory systems tend to lose person and group relations or struggle to integrate clues distributed across members, groups, and time. Together, these issues reveal two core bottlenecks: message attribution and relational understanding in multi-party dialogue, and state reconstruction from interleaved histories. To address both, we propose $\textbf{SpeakerMem-R1}$: its dual-track memory stores speaker-labeled verbatim messages and derived states organized into person-level and group-level views, then combines evidence from both tracks by entity, eve
    
[^2]: 基于文本的发现式无关联表集成方法

    Discovery-Driven Integration of Disjoint Tables via Text

    [https://arxiv.org/abs/2609.26658](https://arxiv.org/abs/2609.26658)

    该论文提出LOKI架构，将“文本介导的连接路径发现”形式化为新任务，通过全局表-文本对比学习实现数据湖中缺乏连接属性的无关联表的细粒度集成。

    

    数据湖中异构数据集的集成是一个关键挑战，尤其是对于语义相关但缺乏显式连接属性的表。我们研究了“发现驱动的集成”问题，即在进行集成之前，必须先发现相关的数据源及其缺失的关系结构。在这一场景下，非结构化文本提供了连接原本互不相关的表的证据。其根本挑战在于以细粒度级别发现通过特定句子连接不同表中各行之间的关系。我们将该任务形式化为“文本介导的连接路径发现”，并提出了一种名为LOKI（知识集成的潜在空间优化）的水平双向交叉注意力架构，用于学习表行和句子的上下文表示。通过全局的表-文本对比学习目标，细粒度的行-句子关联得以自动涌现。

    arXiv:2609.26658v1 Announce Type: cross  Abstract: Integrating heterogeneous datasets within data lakes is a critical challenge, particularly for semantically related tables that lack the explicit attributes needed to be joined. We study Discovery-Driven Integration, where the relevant sources and their missing relational structure must be discovered before integration. In this setting, unstructured text provides the evidence that connects otherwise disjoint tables. The fundamental challenge is to discover the relationships at a fine-grained level that connect individual rows from different tables through specific sentences. We formalize this task as Text-Mediated Join Path Discovery and propose a horizontal bidirectional cross-attention architecture called LOKI Latent-space Optimization for Knowledge Integration) that learns contextualized representations of table rows and sentences. Through a global table-text contrastive objective, fine-grained row-sentence associations emerge witho
    
[^3]: 排列不稳定性何时能泛化？列表式大语言模型重排序的独立视图验证

    When Does Permutation Instability Generalize? Independent-View Validation for Listwise LLM Reranking

    [https://arxiv.org/abs/2609.26251](https://arxiv.org/abs/2609.26251)

    该研究通过精确的有限视图分解和独立视图对比实验发现，列表式LLM重排序中有限视图不稳定性分数对未见排列的“预测力”很大程度上源于共享测量造成的部分-整体关联伪影，在完全不相交的视图间相关性很弱或呈异质性。

    

    列表式语言模型重排序器在等价的候选排列之间常常出现不一致。因此，有限不稳定性诊断常被用来激励额外的采样、聚合或选择性计算。然而，与复用探测视图的验证统计量之间的关联，未必能分离出关于未见排列的预测信息——共享的测量会引发经典的“部分-整体”关联。我们研究了这一现象如何影响“有限视图不稳定性分数能够预测未见排列”这一论断。我们推导了精确的有限视图分解，并前瞻性地比较了复用零个、一个和两个视图的情形，其中包括一个完全不相交的四视图目标。该研究涵盖两个固定的7B模型家族和两个推荐数据集，使用受控列表进行带符号的离线分析，并使用未受干扰的检索器列表进行无目标的复制验证。在四个受控区块上，完全不相交视图间的相关性较弱或呈现异质性（-0.06...

    arXiv:2609.26251v1 Announce Type: new  Abstract: Listwise language-model rerankers often disagree across equivalent candidate permutations. Finite instability diagnostics are therefore used to motivate additional sampling, aggregation, or selective computation. But an association with a validation statistic that reuses the probe views need not isolate predictive information about unseen permutations. Shared measurements can induce classical part-whole association. We study how this affects claims that a finite-view instability score predicts unseen permutations. We derive the exact finite-view decomposition and prospectively compare zero, one, and two reused views, including a fully disjoint four-view target.   The study covers two pinned 7B model families and two recommendation datasets, with controlled lists for signed offline analysis and untouched retriever lists for target-free replication. On the four controlled blocks, fully disjoint correlations are weak or heterogeneous (-0.06
    
[^4]: 哪些重排序结论能在答案界面变化下依然成立？一项前瞻性有限轨道审计

    Which Reranking Conclusions Survive the Answer Interface? A Prospective Finite-Orbit Audit

    [https://arxiv.org/abs/2609.26250](https://arxiv.org/abs/2609.26250)

    该研究通过前瞻性审计发现，仅改变语言模型阅读器的答案界面（如标签绑定、词汇选择和选项顺序）不会实质性地改变BM25与BGE-v2-m3重排序对比的结论，六个确认性设置均未出现超过预设阈值的界面效应或排序反转。

    

    重排序器越来越多地通过下游语言模型的答案来进行评估。这引出了一个检索测量问题：如果仅改变阅读器的答案界面，我们对BM25与BGE-v2-m3的比较结论是否仍应保持一致？我们以前瞻性的方式审计了它们在RAGuard和FEVER数据集上使用四个阅读器时的声明配对效应。检索策略、证据、声明和上下文深度保持固定，而语义到标签的绑定方式、A/B与X/Y词汇表示以及选项顺序共同构成了八个任务等效的界面。我们探究估计出的检索策略效应、其排序关系或选择价值是否会因此发生变化。在六个确认性设置中，没有任何一个显示出超过预设0.015重要性阈值的、经统计认证的界面变异，也没有任何一个显示出经认证的BM25与BGE排序反转。选择器之间的分歧在某一环境中达到33.5%，然而八个环境中没有任何一个能够确立预设的重要性留出价值。

    arXiv:2609.26250v1 Announce Type: new  Abstract: Rerankers are increasingly evaluated through downstream language-model answers. This raises a retrieval-measurement question: if only the reader's answer interface changes, should we reach the same conclusion about BM25 versus BGE-v2-m3? We prospectively audit their claim-paired effect on RAGuard and FEVER with four readers. Retrieval policies, evidence, claims, and context depth remain fixed while semantic-to-label binding, A/B versus X/Y vocabulary, and option order form eight task-equivalent interfaces. We ask whether the estimated retrieval-policy effect, its ordering, or selection value changes. None of the six confirmatory settings showed statistically certified interface variation above the prespecified 0.015 materiality threshold, and none showed a certified reversal of the BM25-BGE ordering. Selector disagreement reaches 33.5% in one environment, yet none of eight environments establishes the prespecified material held-out value
    
[^5]: ABAI参加COLIEE 2026任务1：基于GraphRAG增强元学习的多阶段检索，以及交叉验证与测试差距的事后分析研究

    ABAI at COLIEE 2026 Task 1: Multi-Stage Retrieval with GraphRAG-Enhanced Meta-Learning, and a Post-Hoc Study of the Cross-Validation-to-Test Gap

    [https://arxiv.org/abs/2609.26237](https://arxiv.org/abs/2609.26237)

    本文提出了一种用于COLIEE 2026判例法检索任务的四阶段多阶段检索流程（多视角BM25、神经重排序、图特征与LightGBM元学习），并通过控制实验系统性地分析了官方测试集F1（0.177）远低于交叉验证结果（0.311）的原因。

    

    我们展示了ABAI参加COLIEE 2026任务1（判例法检索）的提交方案，并对该方案表现不佳的原因进行了控制性研究。该任务隐去了被引用的段落本身，这消除了检索器通常所依赖的大部分词汇重叠信息。为应对这一挑战，我们的流程采用四个独立训练的阶段：基于引文上下文窗口的多视角BM25结合倒数排名融合、神经重排序、基于实体社区和图注意力网络的图特征，以及基于34个特征的LightGBM元学习器。我们的最佳运行在官方测试集上达到F1=0.177，而交叉验证结果为0.311，我们将这一差距归因于召回率上限、时间分布偏移以及阈值校准失误。随后，我们对这三个假设逐一进行了检验。在无数据泄漏的协议下，阈值迁移仅造成0.007的F1损失，决策质量在各时间分位数上保持平稳，且官方测试查询与训练数据的距离并无显著差异……

    arXiv:2609.26237v1 Announce Type: cross  Abstract: We present the ABAI submission to COLIEE 2026 Task 1, case law retrieval, together with a controlled study of why it underperformed. The task suppresses the cited passages themselves, which removes much of the lexical overlap a retriever would rely on. Our pipeline answers this with four independently trained stages: multi-view BM25 over citation-context windows with reciprocal rank fusion, neural reranking, graph-based features from entity communities and a graph attention network, and a LightGBM meta-learner over 34 features. Our best run reached F1=0.177 on the official test set, against a cross-validated 0.311, and we attributed that gap to a recall ceiling, temporal distribution shift, and threshold miscalibration. We then tested all three. Under leakage-free protocols threshold transfer costs 0.007 F1, decision quality is flat across chronological quartiles, and the official test queries are not measurably farther from the traini
    
[^6]: 学术出版网络的语义化方法：基于OpenAlex数据的文档向量表示与结构-语义混合融合

    A Semantic Approach to the Academic Publishing Network: Document Vector Representations and Hybrid Structural-Semantic Fusion over OpenAlex Data

    [https://arxiv.org/abs/2609.26218](https://arxiv.org/abs/2609.26218)

    该论文在学术出版网络结构图分析的基础上引入语义层，利用SPECTER2引用感知文档嵌入和权重可调的结构-语义后期融合函数，实现了比TF-IDF基线更契合专家主题分类的文档表示，并提升了推荐效果。

    

    对学术出版网络的结构化图分析能够捕捉实体之间的拓扑关系，但无法感知作品的内容。在我们已有结构化方法的基础上，本工作通过引入一个语义层和参数化的结构-语义融合对其加以补充。我们采用基于引用信息的向量嵌入（SPECTER2）来表示科学文献，并将其存储在以稳定的OpenAlex ID为键的嵌入式向量数据库中，从而使其能够直接与图层相连接。我们定义了一个模块化的后期融合函数，将语义相似度（嵌入向量的余弦相似度）与结构相似度（书目耦合）结合起来，并引入一个可调权重alpha，其取值根据具体任务而定。在俄斯特拉发技术大学（VSB - Technical University of Ostrava）的语料库上，我们展示了两个结果：基于引用信息的嵌入与OpenAlex专家主题分类体系的一致性优于TF-IDF基线，并且在推荐应用场景中……（摘要原文在此处截断）

    arXiv:2609.26218v1 Announce Type: cross  Abstract: Structural graph analysis of the academic publishing network captures the topological relationships between entities but does not see the content of works. Building on our structural approach, this work complements it with a semantic layer and a parameterized structural-semantic fusion. We represent scientific documents by citation-informed vector embeddings (SPECTER2) and store them in an embedded vector database keyed by the stable OpenAlex ID, so that they connect directly to the graph layer. We define a modular late-fusion function that combines semantic similarity (cosine of embeddings) and structural similarity (bibliographic coupling) with a tunable weight alpha whose value is chosen according to the specific task. On the corpus of VSB - Technical University of Ostrava we show two things: citation-informed embeddings agree with the expert OpenAlex topical taxonomy better than a TF-IDF baseline, and in a recommendation use case t
    
[^7]: 当隐藏链接无法被恢复时：离岸泄露网络中的结构可辨识性界限与评估陷阱

    When Concealed Links Cannot Be Recovered: A Structural Identifiability Bound and Evaluation Pitfalls in Offshore Leak Networks

    [https://arxiv.org/abs/2609.26171](https://arxiv.org/abs/2609.26171)

    该论文提出了一个无分布假设的结构可辨识性界限，证明对于任何遵循图同构的链接预测方法（包括拓扑评分和消息传递图神经网络），离岸泄露网络中被隐藏的孤立节点与其结构孪生节点不可区分，其隐藏关系（如受益所有权）无法以高于随机猜测的概率被恢复，且恢复上限由节点的结构不可区分类大小决定。

    

    诸如“巴拿马文件”和“天堂文件”之类的泄露事件暴露了庞大的离岸实体网络，这引出了网络学习中一个显而易见的问题：这些结构刻意隐藏的关系——尤其是“谁实际受益拥有什么”——能否通过链接预测从泄露的公开部分中恢复出来？我们认为答案基本上是否定的，而那些得出相反结论的分析实际上测量了错误的对象。我们的主要结果是一个无分布假设的可辨识性界限：对于任何遵循图同构的恢复规则——涵盖所有基于拓扑的链接预测评分方法以及所有消息传递图神经网络——在观测图中处于孤立状态的被隐藏端点与其结构孪生节点是可互换的，因此其隐藏的边无法以高于随机猜测的概率被恢复。孤立只是最尖锐的情形；一般而言，恢复的上限由节点结构不可区分类的大小所决定。

    arXiv:2609.26171v1 Announce Type: new  Abstract: Leaks such as the Panama and Paradise Papers expose large networks of offshore entities, and they invite an obvious question for network learning. Can the relations these structures are built to hide---above all, who beneficially owns what---be recovered from the public part of the leak by link prediction? We argue that the answer is mostly no, and that the analyses which suggest otherwise are measuring the wrong thing. Our main result is a distribution-free identifiability bound. For any recovery rule that respects graph isomorphism, and that covers every topological link-prediction score together with every message-passing graph neural network, a concealed endpoint left isolated in the observed graph is interchangeable with its structural twins, so its hidden edge cannot be recovered above chance. Isolation is only the sharpest case. In general the ceiling on recovery is set by the size of a node's structural-indistinguishability class
    
[^8]: TailSpec-EASE：面向网络长尾发现的知识图谱正则化线性推荐

    TailSpec-EASE: Knowledge-Graph-Regularized Linear Recommendation for Web Long-Tail Discovery

    [https://arxiv.org/abs/2609.26143](https://arxiv.org/abs/2609.26143)

    提出TailSpec-EASE，一种轻量级线性推荐模型，通过将关系感知的谱知识图谱先验注入局部闭式重构目标，并使先验强度随物品流行度自适应调整，在保持整体精度的同时显著改善长尾物品的推荐效果。

    

    网络平台上的推荐系统往往过度服务热门物品而忽视长尾物品。物品侧的知识图谱（KG），通常以关联数据或RDF风格的网络资源形式存在，可以通过共享语义属性连接稀疏物品，从而缓解这一问题。许多具有竞争力的KG感知推荐器依赖图神经架构，而诸如EASE-R这样强大的浅层线性模型通常忽略侧信息，且其全局闭式版本可能变得不可行。我们提出了TailSpec-EASE，这是一种轻量级推荐器，它将关系感知的谱知识图谱先验注入到局部闭式重构目标中。先验强度随物品流行度自适应调整，从而为长尾物品提供更强的语义引导。在四个公开基准数据集以及涵盖经典方法、线性方法、图协同过滤、KG感知神经方法和分数级KG方法的广泛基线对比中，TailSpec-EASE在整体准确率、长尾性能等方面取得了有利的权衡。

    arXiv:2609.26143v1 Announce Type: cross  Abstract: Recommender systems on Web platforms tend to over-serve popular items and neglect the long tail. Item-side knowledge graphs (KGs), often available as linked data or RDF-style Web resources, can help by connecting sparse items through shared semantic attributes. Many competitive KG-aware recommenders rely on graph neural architectures, whereas strong shallow linear models such as EASE-R typically ignore side information and may become infeasible in their global closed-form version. We introduce TailSpec-EASE, a lightweight recommender that injects a relation-aware spectral KG prior into a local closed-form reconstruction objective. The prior strength adapts to item popularity, giving stronger semantic guidance to long-tail items. Across four public benchmarks and a broad set of classical, linear, graph-CF, KG-aware neural, and score-level KG baselines, TailSpec-EASE attains a favorable trade-off between overall accuracy, long-tail perfo
    
[^9]: CoVeR：智能体检索中基于覆盖度的验证器调用路由

    CoVeR: Coverage-Based Routing of Verifier Calls in Agentic Retrieval

    [https://arxiv.org/abs/2609.26086](https://arxiv.org/abs/2609.26086)

    CoVeR 利用冻结句子嵌入的覆盖度边际作为单一阈值门控，仅在证据状态模糊时才调用 LLM 验证器，从而在保持答案准确率几乎不变的前提下，大幅削减智能体检索中验证器反复处理证据的调用成本。

    

    智能体检索系统会发出一系列搜索查询，并且必须在每一步判断目前已收集的证据是否足以停止。将这一决策委托给 LLM 验证器或提示裁判可以使停止时机变得可靠，但验证器随后需要在每次检索步骤之后重新处理不断增长的证据，这是一笔巨大的重复成本。我们证明，其中大部分调用可以在不显著影响答案准确率的情况下被跳过：在冻结的句子嵌入覆盖度边际上设置单一阈值，即可检测出证据仍明显不完整的状态，而验证器只在模糊的剩余情况下被调用，我们将这一门控机制称为 CoVeR（基于覆盖度的验证器路由）。在三个多跳问答基准上（评估协议在全规模运行前已固定），采用 CoVeR 门控的智能体在答案准确率上与全预算智能体以及始终验证的基线相差不到一个 EM 点。它减少了 62……（原文摘要在此处截断）

    arXiv:2609.26086v1 Announce Type: new  Abstract: An agentic retrieval system issues a sequence of search queries and must decide, at each step, whether the evidence collected so far is enough to stop. Delegating that decision to an LLM verifier or a prompt judge makes stopping reliable, but the verifier then reprocesses the growing evidence after every retrieval step, a substantial repeated cost. We show that most of these calls can be skipped without materially changing answer accuracy: a single threshold on a frozen sentence-embedding coverage margin detects the states in which the evidence is still plainly incomplete, and the verifier is called only on the ambiguous remainder, a gate we call CoVeR (Coverage-based Verifier Routing). Across three multi-hop QA benchmarks, with the evaluation protocol fixed before the full-scale run, the CoVeR-gated agent matches the answer accuracy of both the full-budget agent and the always-verify baseline within a fraction of an EM point. It cuts 62
    
[^10]: 知识即技能：面向大语言模型智能体自主使用知识库的结构化设计

    Knowledge-as-Skill: A Structural Design for Autonomous Knowledge-Base Use by LLM Agents

    [https://arxiv.org/abs/2609.25991](https://arxiv.org/abs/2609.25991)

    提出“知识即技能”三层知识库组织方案——以 SKILL.md 为核心的发现层、逐目录 index.md 的导航层以及带 YAML 元数据的文档知识层——使知识库对 LLM 智能体而言可发现、可导航且自描述，从而支持智能体自主决定是否检索及如何使用知识。

    

    检索增强生成（RAG）使大语言模型（LLM）能够访问外部知识，但其传统的“检索—拼接—生成”流水线代替模型做出检索决策。随着工具使用和智能体循环日益可靠，智能体可以自行决定是否检索、查看什么内容以及何时停止。这一转变暴露出一个新的瓶颈：智能体可能并不了解知识库中包含什么。传统知识库将文档呈现为匿名的文本块，其关于范围、目的、来源或关系的信息十分有限。我们提出“知识即技能”，一种使知识库具备可发现性、可导航性和自描述性的组织方案。它包含三层：以 SKILL.md 为核心的发现层；每个目录配有一个 index.md 的导航层；以及包含带 YAML 前置元数据（用于标注主题、类型、来源和生命周期）文档的知识层。该设计遵循 Op（原文摘要在此处截断）

    arXiv:2609.25991v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) gives large language models (LLMs) access to external knowledge, but its conventional retrieve-concatenate-generate pipeline makes retrieval decisions on behalf of the model. As tool use and agent loops become more reliable, an agent can decide whether to retrieve, what to inspect, and when to stop. This shift exposes a new bottleneck: the agent may not know what a knowledge base contains. Traditional knowledge bases expose documents as anonymous text chunks with limited information about scope, purpose, provenance, or relations.   We propose Knowledge-as-Skill, an organization scheme that makes a knowledge base discoverable, navigable, and self-descriptive. It has three layers: a discovery layer centered on SKILL.md; a navigation layer with one index.md per directory; and a knowledge layer containing documents with YAML frontmatter for topic, type, provenance, and lifecycle. The design follows the Op
    
[^11]: ARAFA：一个由大语言模型生成的阿拉伯语事实核查数据集

    ARAFA: An LLM-Generated Arabic Fact-Checking Dataset

    [https://arxiv.org/abs/2609.25833](https://arxiv.org/abs/2609.25833)

    研究者利用大语言模型通过三步自动化流水线构建了包含18万余个标注声明-证据对的大规模阿拉伯语事实核查数据集Arafa，有效缓解了阿拉伯语事实核查资源稀缺的问题。

    

    由于数据集和资源的稀缺，自动事实核查在阿拉伯语自然语言处理领域构成了重大挑战。在本文中，我们介绍了Arafa，一个新的大规模现代标准阿拉伯语事实核查数据集，该数据集通过一个利用大语言模型（LLM）的自动化框架构建而成。数据集的构建采用三步流水线：（1）从阿拉伯语维基百科页面生成带有支持性文本证据的声明；（2）通过声明变异生成带有反驳证据的具有挑战性的反事实声明；（3）通过自动验证步骤，检验生成的声明是被其伴随证据所支持或反驳，还是证据不足以判断声明的有效性。最终得到的数据集包含181,976个声明-证据对，标注为“支持”、“反驳”或“信息不足”。在样本上进行的人工评估……（原文摘要在此处截断）

    arXiv:2609.25833v1 Announce Type: new  Abstract: Automatic fact-checking poses a significant challenge in Arabic natural language processing due to the scarcity of datasets and resources. In this manuscript, we introduce Arafa, a new large-scale dataset for fact-checking in Modern Standard Arabic, constructed through an automated framework leveraging large language models (LLMs). The dataset was constructed through a three-step pipeline: (1) claim generation from Arabic Wikipedia pages with supporting textual evidence, (2) claim mutation to generate challenging counterfactual claims with refuting evidence, and (3) an automatic validation step to validate that the generated claims are either supported or refuted by their accompanying evidence, or if the evidence does not provide enough information to judge the validity of the claims. The resulting dataset comprises 181,976 claim-evidence pairs labeled as supported, refuted, or not enough information. Human evaluation carried out on a te
    
[^12]: 个性化搜索中基于LLM重排序的语义与行为信号鲁棒融合

    Robust Fusion of Semantic and Behavioural Signals for LLM Reranking in Personalised Search

    [https://arxiv.org/abs/2609.25825](https://arxiv.org/abs/2609.25825)

    提出确定性双样本特征丢弃训练方法，使基于LLM的个性化搜索重排序系统能够鲁棒地融合行为统计特征（QSS）与语义信号，避免捷径学习，即使行为特征缺失时仍能保持良好的排序性能。

    

    个性化搜索需要在满足查询意图的同时融合用户上下文和历史交互信息。基于LLM的交叉编码器提供了统一的重排序接口，但将预测性行为统计特征注入其提示中可能导致捷径学习：模型过度依赖历史信号，而牺牲了能够泛化到稀疏或未见搜索场景的语义和用户上下文模式。我们在一个大规模音频流媒体平台的个性化搜索系统中研究这一问题，使用了查询切片统计（Query Slice Stats, QSS）——一种由交互数据衍生的行为特征，用于总结查询-候选对的历史成功情况。朴素的QSS注入在特征可用时能提升排序质量，但在特征缺失时会降低模型鲁棒性。我们通过确定性双样本特征丢弃训练来解决这一问题，即每个样本呈现两次：一次包含QSS，一次移除QSS。离线实验中，QSS注入将排序质量提升了13.3%（摘要在此处截断）。

    arXiv:2609.25825v1 Announce Type: new  Abstract: Personalised search must satisfy query intent while incorporating user context and historical interactions. LLM-based cross-encoders provide a single reranking interface, but injecting predictive behavioural statistics into their prompts can encourage shortcut learning: reliance on historical signals at the expense of semantic and user-context patterns that generalise to sparse or unseen searches.   We study this problem in the personalised search system of a large-scale audio streaming platform using Query Slice Stats (QSS), an interaction-derived behavioural feature summarising historical success for query-candidate pairs. Naive QSS injection improves ranking when the feature is available but reduces robustness when it is removed. We address this with deterministic dual-sample feature-dropout training, which presents each example once with QSS included and once with QSS removed.   Offline, QSS injection improves ranking quality by 13.3
    
[^13]: 从离线代理到在线决策：面向对话式AI的分层参与度评估框架

    From Offline Proxies to Online Decisions: A Layered Engagement Evaluation Framework for Conversational AI

    [https://arxiv.org/abs/2609.25408](https://arxiv.org/abs/2609.25408)

    该论文提出了一个分层参与度评估框架，将离线代理指标建模为三重对齐链条，并通过区间感知的决策一致性协议审计其与在线A/B实验结果的一致性，从而在不消耗线上流量的情况下预测对话式AI改动是否值得上线。

    

    在线A/B实验是评估用户参与度的决策标准，但流量和读出时间限制了可测试的对话式AI变更数量。我们探究一种无需处理组用户真实暴露即可计算的离线信号是否与这些实验的结果保持一致。我们贡献了一个可复用的构建与诊断清单，将离线代理视为由三个对齐环节组成的链条：行为标签与产品结果的对齐、学习到的分类器与候选助手行为的对齐、以及聚合后的离线信号与实验效应的对齐。配套的评估协议通过区间感知的决策一致性（比较离线与在线的置信区间而非点估计）以及实验内排序来审计整个复合系统。我们所评估的具体实例包括一个固定的评估套件（用于对候选行为进行评分）、一个训练用于预测会话/提示级别参与度的参与度分类器……

    arXiv:2609.25408v1 Announce Type: cross  Abstract: Online A/B experiments are the decision standard for user engagement, but traffic and readout time limit how many conversational-AI changes can be tested. We ask whether an offline signal designed to be computable without treatment-arm user exposure agrees with the outcomes of those experiments. We contribute a reusable construction and diagnosis checklist that treats an offline proxy as a chain of three alignments: behavioral label to product outcome, learned classifier to candidate-assistant behavior, and aggregated offline signal to experiment effect. A companion evaluation protocol audits the whole composite by interval-aware decision agreement, which compares offline and online confidence intervals instead of point estimates, and by within-experiment ranking. The instantiation we evaluate comprises a fixed evaluation suite on which candidate behavior is scored, an engagement classifier trained to predict session/prompt level engag
    
[^14]: ReFilter：融合嵌入与LLM过滤以检索相似的移动应用

    ReFilter: Bridging Embeddings and LLM Filtering for Similar Mobile App Retrieval

    [https://arxiv.org/abs/2609.25306](https://arxiv.org/abs/2609.25306)

    提出ReFilter混合框架，先用嵌入检索语义相关的候选应用，再用大语言模型进行上下文过滤，以高精度识别功能相似的移动应用，F1分数达90%。

    

    检索相似的移动应用程序对于研究人员、开发者和最终用户都至关重要。研究人员利用相似性检测来研究应用生态系统和趋势，开发者用于竞争对手分析，最终用户用于精准的应用推荐。现有方法依赖于基于嵌入的检索，虽然能够捕捉语义相似性，但无法识别功能相似的应用。据我们所知，由于评估大量应用对的计算成本过高，此前尚无工作将基于大语言模型（LLM）的过滤应用于该任务。为填补这一空白，我们提出了ReFilter——一个混合框架，它首先使用嵌入检索语义相关的候选应用，然后应用基于LLM的上下文过滤，以更高的精度识别真正功能相似的应用。这一设计兼顾了效率与准确性，在检索相似应用方面达到了90%的F1分数。

    arXiv:2609.25306v1 Announce Type: new  Abstract: Retrieving similar mobile applications (apps) is essential for researchers, developers, and end-users. Researchers use similarity detection to study app ecosystems and trends, developers for competitor analysis, and end-users for focused app recommendations. Existing approaches rely on embedding-based retrieval, which captures semantic similarity but fails to identify functionally similar apps. To our knowledge, no prior work has applied large language model (LLM)-based filtering to this task, due to the high computational cost of evaluating large numbers of app pairs. To address this gap, we propose ReFilter, a hybrid framework that first Retrieves semantically related candidate apps using embeddings and then applies LLM-based contextual Filtering to identify true functionally similar apps with higher precision. This design balances efficiency and accuracy, achieving an F1-score of 90% for retrieving similar apps. By improving the relev
    
[^15]: GroundedGEO：审计生成式搜索排名中的证据差距

    GroundedGEO: Auditing the Evidence Gap in Generative Search Rankings

    [https://arxiv.org/abs/2609.25189](https://arxiv.org/abs/2609.25189)

    该论文提出证据配对基准与主张级重排序器GroundedGEO，首次揭示生成式搜索中“证据状态是主张-证据关系而非文本属性”的可识别性差距，并实证表明缺乏证据支持的内容丰富文本能显著操纵冻结排序器的排名。

    

    生成式搜索系统为重大决策对产品和服务进行排名，而发布者可以低成本地让候选文本看起来与查询相关。然而，证据状态并非文本属性，而是主张与证据之间的关系：仅基于文本的排序器和防御机制无法将诚实详细的内容与捏造的细节区分开来，从而造成一个可识别性差距。我们通过一个证据配对基准（50个电子商务查询、1,950个案例）和一个主张级别的重排序器GroundedGEO来审计这一差距，该重排序器会对所提供证据包中缺乏支持但与查询相关的主张进行惩罚。匹配的丰富变体用于控制格式和内容量；证据包孪生实验在保持文本固定的情况下添加证明，而精简证据包则移除这些证明。在冻结的列表式排序器Qwen2.5-7B上，无支持的丰富变体相比干净候选者显示出显著的归一化排名提升（跨主张画像为+0.065至+0.092，经Holm校正），而有支持和中性对照组则没有；该效应依赖于具体模型（边际……

    arXiv:2609.25189v1 Announce Type: new  Abstract: Generative search systems rank products and services for consequential decisions, and publishers can cheaply make candidate text look relevant. Yet evidence status is not a text property but a claim-evidence relation: text-only rankers and defenses cannot separate honest detailed content from fabricated detail, creating an identifiability gap. We audit this gap with an evidence-paired benchmark (50 e-commerce queries, 1,950 cases) and a claim-level reranker, GroundedGEO, that penalizes query-relevant claims lacking support in a supplied packet. Matched rich variants control format and volume; packet twins add attestations at fixed text, while thinned packets withdraw them. On the frozen listwise ranker Qwen2.5-7B, unsupported-rich variants show significant normalized rank gain over clean candidates (+0.065 to +0.092 across claim profiles, Holm-corrected), while supported and neutral controls do not; the effect is model-dependent (margina
    
[^16]: 从排序文档到可靠上下文：面向AI搜索的答案导向上下文构建框架

    From Ranked Documents to Reliable Contexts: An Answer-Oriented Context Construct Framework for AI Search

    [https://arxiv.org/abs/2609.23354](https://arxiv.org/abs/2609.23354)

    该论文提出一个面向AI搜索的答案导向上下文构建三阶段框架（答案支持、内容可信度、上下文组织），将检索目标从传统的文档排序转变为为正确答案生成构建可靠上下文。

    

    传统网络搜索遵循一种面向人类的范式，即用户自行查看排序后的文档并综合信息。而在AI搜索中，检索到的文档转而作为生成模型的输入，这使得检索目标从以“搜索满意度”为导向的文档排序，转变为为正确答案生成构建可靠的上下文。我们将这一转变形式化为面向答案的上下文构建，并提出一个三阶段框架：(1) 答案支持——识别对答案生成有信息贡献的候选文档；(2) 内容可信度——从来源、时效和事实三个角度评估这些信息是否为正确答案提供了可靠依据；(3) 上下文组织——在有限的上下文预算下对保留信息进行选择、整合与结构化，以实现一致且稳健的生成。我们进一步开发了一个涵盖先验与后验优化的工业级工作流程……

    arXiv:2609.23354v1 Announce Type: new  Abstract: Traditional Web search follows a human-facing paradigm in which users inspect ranked documents and synthesize information themselves. In AI Search, retrieved documents instead serve as inputs to a generation model, shifting the retrieval objective from ranking documents by Search Satisfaction to constructing reliable context for correct answer generation. We formulate this shift as answer-oriented context construction through a three-stage framework: (1) Answer Support identifies candidate documents that contribute information to answer generation; (2) Content Trustworthiness assesses whether this information provides a reliable basis for correct answers from source, temporal, and factual perspectives; and (3) Context Organization selects, consolidates, and structures retained information under a finite context budget for consistent and robust generation. We further develop an industrial workflow spanning prior and posterior optimization
    
[^17]: 用于混合检索的参数化稠密-稀疏融合：基于Qdrant在BEIR SciFact上调整排名-分数混合

    Parameterized Dense-Sparse Fusion for Hybrid Retrieval: Tuning a Rank-Score Mix on BEIR SciFact with Qdrant

    [https://arxiv.org/abs/2609.22770](https://arxiv.org/abs/2609.22770)

    该论文提出了一种显式参数化的稠密-稀疏混合检索融合方法，通过在BEIR SciFact上网格搜索调整排名-分数混合参数（α=0.8, λ=0.75, κ=20），使nDCG@10达到0.753，优于稠密BGE和等权重RRF基线，同时证明这些参数具有数据集特异性。

    

    我们研究了一种参数化混合排序器，它将稠密嵌入列表与稀疏词法列表进行融合。该方法具有一个小的、显式的参数向量：稠密先验 α ∈ [0,1]、分数与排名混合 λ ∈ [0,1]、RRF平滑参数 κ > 0、可选的按查询调整 α 的列表几何系数，以及可以关闭稀疏搜索的路由器边距 τ。我们在SciFact训练集（809个查询）上对这些参数范围进行网格搜索，并在SciFact测试集（300个查询）上冻结所选数值。调整后的排名-分数混合（α = 0.8, λ = 0.75, κ = 20）达到了0.753 nDCG@10和0.889 recall@10，在该测试集上优于稠密BGE（0.742 / 0.871）和等权重RRF（0.707 nDCG@10）。列表条件化的 α 仅带来 +0.0006 nDCG 的提升；稀疏关闭路由器被同一训练集否定（任何跳过约50%查询的 τ 都会导致nDCG损失）。这些系数是特定于数据集的。等权重RRF与

    arXiv:2609.22770v1 Announce Type: new  Abstract: We study a parameterized hybrid ranker that fuses a dense embedding list and a sparse lexical list. The method has a small, explicit parameter vector: a dense prior $\alpha \in [0,1]$, a score-versus-rank mix $\lambda \in [0,1]$, an RRF smoothing parameter $\kappa > 0$, optional list-geometry coefficients that move $\alpha$ per query, and a router margin $\tau$ that can turn sparse search off. We grid-search those ranges on SciFact train (809 queries) and freeze the chosen values on SciFact test (300). The tuned rank-score mix ($\alpha = 0.8$, $\lambda = 0.75$, $\kappa = 20$) reaches 0.753 nDCG@10 and 0.889 recall@10, outperforming dense BGE (0.742 / 0.871) and equal-weight RRF (0.707 nDCG@10) on that test split. A list-conditioned $\alpha$ adds +0.0006 nDCG; a sparse-off router is rejected by the same train split (any $\tau$ that skipped approximately 50% of queries lost nDCG). These coefficients are dataset-specific. Equal RRF with the
    
[^18]: 基准雷达：面向AI基准与评估的动态数据库和搜索引擎

    Benchmark Radar: A Living Database and Search Engine for AI Benchmarks and Evaluation

    [https://arxiv.org/abs/2609.11115](https://arxiv.org/abs/2609.11115)

    本文提出Benchmark Radar，一个每日自动发现并整合AI基准论文、数据集和代码的动态数据库与搜索引擎，为LLM评估、智能体、编程、推理和安全等领域提供可检索的基准目录、来源引用和分数历史。

    

    基准研究人员和大语言模型（LLM）及其他AI系统的开发者需要找到相关的评估方法、定位其基准数据集和代码，并理解已报告分数背后的设置。我们提出了Benchmark Radar（基准雷达），这是一个用于检索和发现AI基准的动态数据库和搜索引擎，涵盖LLM评估、智能体和工具使用基准、编程、推理、安全性以及特定领域评估。该系统将每日发现的基准论文、代码仓库、数据集和发布版本与可搜索的基准目录、模型卡和技术报告中的提及以及分数历史相结合。它保留了源身份信息和引用，以便读者可以查看候选基准及其评估证据。每日发现基于37个来源：13个直接连接器和24个第一方研究与工程信息源。该目录包含来自4个基准目录的1,283条源记录

    arXiv:2609.11115v1 Announce Type: cross  Abstract: Benchmark researchers and developers of large language models (LLMs) and other AI systems need to find relevant evaluations, locate their benchmark datasets and code, and understand the settings behind reported scores. We present Benchmark Radar, a living database and search engine for retrieval and discovery of AI benchmarks, covering LLM evaluation, agentic and tool-use benchmarks, coding, reasoning, safety, and domain-specific evaluations. The system combines daily discovery of benchmark papers, repositories, datasets, and releases with a searchable benchmark catalog, mentions in model cards and technical reports, and score histories. It retains source identities and citations so readers can inspect candidate benchmarks and their evaluation evidence. Daily discovery draws on 37 sources: 13 direct connectors and 24 first-party research and engineering feeds. The catalog contains 1,283 source records drawn from 4 benchmark catalogs an
    
[^19]: 预算化继承式智能体记忆验证中的计划指针与记录指令形式

    Plan Pointers and Record-Directive Form in Budgeted Verification of Inherited Agent Memory

    [https://arxiv.org/abs/2609.03450](https://arxiv.org/abs/2609.03450)

    该论文通过十二项注册研究发现，写入智能体记忆库的指令形式（准则、裸ID或指针）会以高度模型依赖的方式显著影响预算受限下的记录选择，长度匹配准则可带来35分的提升，但附加ID可能完全抵消准则的效果。

    

    arXiv:2609.03450v1 公告类型：cross。摘要：一个继承了六条单行记忆的智能体在行动前最多只能拉取一条存档的源记录；写入存储中的指令可以引导这一选择：可以是指向该记录的指针、识别该记录的准则，或两者兼有。在同一仪器谱系上的十二项注册研究（共14,760次尝试）中，我们测量了每种指令形式下请求的去向。在六个直接提供商模型上，长度匹配的准则比裸ID高出+35.0个点 [+31.2, +38.8]（研究D）；而在九个模型的OpenRouter服务面板上，该对比未能通过注册的优越性规则（研究E）。在三个Claude模型上，附加ID会抵消准则的效果（Opus 5: 从40/40降至0/40；研究F-x）；六次字节匹配的编辑使每个精确字符串都产生了各自的效应（研究G），并且在每单元八十次运行的重跑中，三十个复现对比中有十五个处于误差范围内，十五个未获解决，没有一个超出范围（研究G'）。批准行（在Opus 5上+96.0个点）以及一个（摘要在此处截断）

    arXiv:2609.03450v1 Announce Type: cross  Abstract: An agent that inherits six one-line memories may pull at most one archived source record before acting; a directive written into the store can steer that choice: a pointer to the record, a criterion that identifies it, or both. Across twelve registered studies on one instrument lineage (14,760 attempts) we measured where the request goes under each form. On six direct-provider models a length-matched criterion exceeded a bare id by +35.0 points [+31.2, +38.8] (Study D); the contrast failed its registered superiority rule on a nine-model OpenRouter-served panel (Study E). Appending the id cancelled the criterion on three Claude models (Opus 5: 40/40 to 0/40; Study F-x); six byte-matched edits gave each exact string its own effect (Study G), and a re-run at eighty runs per cell left fifteen of thirty replication contrasts within the margin, fifteen unresolved and none beyond (Study G'). A ratification line (+96.0 points on Opus 5) and a 
    
[^20]: 基于GNN的知识图谱问答的查询侧攻击：从实体链接到答案生成的故障追踪

    Query-Side Attacks on GNN-Based KGQA: Tracing Failures from Entity Linking to Answer Generation

    [https://arxiv.org/abs/2608.25922](https://arxiv.org/abs/2608.25922)

    本文通过阶段隔离协议发现，基于GNN的知识图谱问答系统的主要脆弱性在于子图构建阶段，而非GNN推理阶段，这挑战了现有鲁棒性评估的假设。

    

    基于GNN的知识图谱问答（KGQA）流水线通过四个离散阶段处理查询：实体链接、子图检索、GNN推理和答案生成。标准的鲁棒性评估将阶段级故障合并为一个端到端指标，掩盖了脆弱性的来源和适当的缓解目标。我们研究了当流水线受到输入问题上的对抗性扰动时，哪个阶段失败，以及为什么失败。我们引入了一种阶段隔离协议，包含两种经过知识图谱验证的、保持答案的对抗性扰动：组合重构（CR）和关系同义词交换（RS）分别针对不同阶段，同时保持实体种子不变。在ComplexWebQuestions和WebQSP上的评估结果与主流假设相反：当子图完整时，GNN推理阶段保持接近基线的准确性，而子图构建则导致了大部分故障。

    arXiv:2608.25922v1 Announce Type: new  Abstract: GNN-based Knowledge Graph Question Answering (KGQA) pipelines process queries through four discrete stages: entity linking, subgraph retrieval, GNN reasoning, and answer generation. Standard robustness evaluations conflate stage-level failures into a single end-to-end metric, obscuring both the source of brittleness and the appropriate mitigation target. We ask which stage fails, and why, when the pipeline is subjected to adversarial perturbations on the input question. We introduce a stage-isolation protocol with two answer-preserving adversarial perturbations verified against the knowledge graph: Compositional Restructuring (CR) and Relation Synonym Swap (RS) target distinct stages while leaving entity seeds intact. Evaluated across ComplexWebQuestions and WebQSP, the results run counter to prevailing assumptions: the GNN reasoning stage retains near-baseline accuracy when the subgraph is intact, while subgraph construction accounts fo
    
[^21]: 希腊法律条文检索基准：GreekBarRetrieval

    GreekBarRetrieval: A Benchmark for Greek Statutory Retrieval

    [https://arxiv.org/abs/2608.18752](https://arxiv.org/abs/2608.18752)

    本文提出了希腊法律条文检索基准GreekBarRetrieval，并通过实验发现LLM查询重构能显著提升BM25与密集检索的性能差距。

    

    arXiv:2608.18752v1 公告类型：交叉 摘要：法定条文检索对于基于引用的法律问答系统是必要的，但在希腊语领域仍未得到充分探索。我们引入了GreekBarRetrieval，这是一个公开的检索基准，源自并补充了未包含检索部分的GreekBarBench。该新基准包含283个律师资格考试问题，每个问题附带其涉及案例的事实，以及6,308个可供检索的候选法定条文。问题和事实以日常语言表述，但需要映射到法条的正式术语及其抽象法律概念。另一个复杂之处在于，并非所有案例事实都与案例的每个问题相关。通过实验三种BM25变体和九种密集检索器，我们发现标准密集检索在Recall@100上远优于标准稀疏检索。然而，基于LLM的查询重构帮助BM25缩小了这一差距，同时提升了密集检索的性能。经过十轮...

    arXiv:2608.18752v1 Announce Type: cross  Abstract: Statutory retrieval is necessary for citation-grounded legal question answering, but remains underexplored for Greek. We introduce GreekBarRetrieval, a public retrieval benchmark derived from, and complementing GreekBarBench, which did not include retrieval. The new benchmark comprises 283 bar-exam questions, each accompanied by the facts of the case it refers to, and 6,308 candidate statutory articles to retrieve from. Questions and facts are stated in everyday language, but need to be mapped to the formal terminology of statutes and their abstract legal concepts. A further complication is that not all of the case facts are relevant to each question of a case. Experimenting with three BM25 variants and nine dense retrievers, we find that vanilla dense retrieval far outperforms vanilla sparse retrieval in Recall@100. However, LLM-based query reformulation helps BM25 close that gap, while also improving dense retrieval. With a ten-round
    
[^22]: WebArxiv：一个用于评估多模态网络智能体在arXiv任务上表现的可复现基准

    WebArxiv: A Reproducible Benchmark for Evaluating Multimodal Web Agents on arXiv Tasks

    [https://arxiv.org/abs/2507.00938](https://arxiv.org/abs/2507.00938)

    WebArxiv是一个基于arXiv静态快照构建的可复现基准，包含510个具有确定性答案的时间不变任务，用于评估多模态网络智能体在多约束论文检索、细粒度内容提取和跨论文比较等学术任务上的能力。

    

    arXiv:2507.00938v3 公告类型：replace  摘要：基础模型如今使自主智能体能够与真实网站进行交互，但现有的基准测试侧重于通用浏览，低估了面向研究的环境和学术发现工作流程，并且通常依赖于实时网站，而实时网站不断变化的内容和结构损害了可复现性。arXiv提供了一个真实、可复现、层次结构化、以信息为中心且不涉及隐私敏感交互的测试平台。我们提出了WebArxiv，这是一个静态快照基准，包含510个时间不变的任务，每个任务都有唯一确定的基准答案。其多样化、真实的学术任务超越了简单的信息查找和规则遵循，强调多约束论文检索、细粒度内容提取和跨论文比较。对一系列基于基础模型的网络智能体的评估表明，WebArxiv仍然具有挑战性。行为分析显示，智能体过度依赖于

    arXiv:2507.00938v3 Announce Type: replace  Abstract: Foundation models now enable autonomous agents to interact with real-world websites, but existing benchmarks emphasize general-purpose browsing, underrepresent research-oriented environments and scholarly discovery workflows, and often depend on live sites whose changing content and structure undermine reproducibility. arXiv provides a realistic, reproducible, hierarchically structured, information-centric testbed without privacy-sensitive interactions. We introduce WebArxiv, a static-snapshot benchmark comprising 510 time-invariant tasks, each with a unique deterministic ground truth. Its diverse, realistic scholarly tasks go beyond simple information lookup and rule following to emphasize multi-constraint paper retrieval, fine-grained content extraction, and cross-paper comparison. Evaluations of a range of foundation-model-based web agents show that WebArxiv remains challenging. Behavioral analysis reveals that agents over-rely on
    

