# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SAGE-PAIR: Source-Pretrained Routed Residual Adaptation for Sparse-Feedback Recommendation](https://arxiv.org/abs/2611.09728) | 提出SAGE-PAIR，通过在源时期预训练低维残差校正空间并以源用户画像为条件添加软路由专家机制，仅需0.05-1.0%的稀疏目标反馈即可校准残差状态而无需重新学习推荐器，有效解决了极端稀疏反馈下更新方向辨识度弱的问题。 |
| [^2] | [SPARSEUP: Towards Better Recipes for Training Modern Learned Sparse Retrievers](https://arxiv.org/abs/2611.09680) | 本文提出 SPARSEUP 训练方案，通过 logit 分布调控、逐位置扇出硬上限以及词汇表合并策略，让基于现代骨干网络的 Splade 类稀疏检索器实现更优的效率-效果权衡。 |
| [^3] | [ENGRAM: From Agent Experience to Weights through Verified Consolidation](https://arxiv.org/abs/2611.09472) | ENGRAM框架将训练准入与记忆释放分离，通过示范感知门控、印迹轨迹图上下文学习和验证门控，只有当经验被有效训练且更新后的模型在无需检索的情况下仍达到相对和绝对评分标准时，才将外部记忆释放移除。 |
| [^4] | [Orthogonal Norm Growth in Two-Tower Retrieval: A Popularity Mechanism for Unified Discovery](https://arxiv.org/abs/2611.09294) | 该论文揭示了双塔检索中仅余弦损失在特定充分条件下会导致物品向量范数随重复更新而正交增长，当搜索与推荐任务共享物品编码器并以原始内积服务时，这种增长会累积形成一种内在的流行度偏差机制。 |
| [^5] | [FondueQA: A Benchmark for Evaluating Parametric Compensation in Multi-Hop QA](https://arxiv.org/abs/2611.09274) | 提出FondueQA基准，通过将每个多跳事实唯一映射到指定证据文档并执行受控的文档级证据消融，评估大语言模型在不完整检索下利用内部参数化知识进行补偿的能力，同时避免基准污染问题。 |
| [^6] | [Structure and Lineage Evidence for Vietnamese Statutory Article Retrieval on DRiLL](https://arxiv.org/abs/2611.09200) | 该论文通过引入条款身份、同一法律支持以及引用与谱系证据等结构化特征，将DRiLL数据集上越南法律条文检索的官方F2指标从71.77%提升至72.91%。 |
| [^7] | [Large Language Model Consensus and Corpus-Adaptive Topic Modeling for Classifying and Mapping Patent Innovation](https://arxiv.org/abs/2611.09194) | 该研究提出了一个可审计的专利分析流水线，通过三个大语言模型的投票共识机制对专利的新颖性与创新类型进行分类，并结合随语料库规模自适应调参的PatentSBERTa嵌入聚类实现主题建模与解读。 |
| [^8] | [A Taxonomy-Driven Synthetic Dataset for Retrieval Effectiveness and Fairness Evaluation in Hiring](https://arxiv.org/abs/2611.09178) | 该论文提出了一个基于分类学驱动、完全合成的多语言HR数据集与基准，可同时评估检索有效性与算法公平性，揭示了通用基准在招聘场景中的不足，并为识别兼顾语义准确性与公平性的帕累托最优模型提供了定制化评估方案。 |
| [^9] | [PAGSL: Path-Augmenting Graph Structure Learning for explainable scholar recommendation](https://arxiv.org/abs/2611.09126) | 提出 PAGSL 模型，通过图增强激活潜在学术路径并利用基于路径的神经网络编码，结合带路径分数优化的掩码机制，实现高性能且解释简洁、个性化、有层次深度的可解释学者推荐。 |
| [^10] | [MV-IVF: Multi-Vector Retrieval via Multi-Vector Clustering](https://arxiv.org/abs/2611.09040) | 本文提出 MV-IVF，一种直接在多向量数据上构建的原生倒排文件索引，其核心是基于新的 Lloyd 风格算法 MV-Lloyd 在 Chamfer 目标下对点云聚类，从而以原生方式高效加速多向量检索。 |
| [^11] | [Rethinking Latent Reasoning for Generative Recommendation: Beyond Teacher-Generated CoT and Recurrent Hidden-State Feedback](https://arxiv.org/abs/2611.09006) | 提出PHiT方法，直接从物品历史中提取语义目标来学习思维令牌的中间状态，从而摆脱了对昂贵的教师生成思维链监督和专门的循环隐状态执行循环的依赖。 |
| [^12] | [SkinMind: Equipping Agentic RAG with Human-Inspired Graph Probing for Reliable Dermatology AI](https://arxiv.org/abs/2611.08935) | 提出了SkinMind多模态智能体RAG框架，通过八个层次化的类人图探索工具和编排器与图环境的迭代交互，实现渐进式披露的自主证据搜寻，从而提升皮肤病学AI在异构多模态来源中的可靠性。 |
| [^13] | [Generative Search and Recommendation at Shopify](https://arxiv.org/abs/2611.08804) | Shopify 提出 SID-LLM，将搜索与推荐统一建模为生成语义 ID 的任务，通过目录与行为数据联合预训练、强化学习、专家蒸馏以及多 token 预测结合轻量级排序器，实现了高质量、低延迟的检索与推荐。 |
| [^14] | [SteerGR: Steering Generative Recommendation via Sparse Autoencoders for Controllable Game Discovery](https://arxiv.org/abs/2611.08790) | 提出SteerGR框架，利用稀疏自编码器将人类可读的游玩概念与生成式推荐模型的内部特征关联，通过直接干预所选特征实现无需重训练的概念级可控游戏推荐。 |
| [^15] | [CascadeRec: Deployed Adaptive LLM Compute Allocation for Multi-Tenant Content Recommendation](https://arxiv.org/abs/2611.08566) | CascadeRec通过两级LLM级联与自适应路由，只将可能受益于额外算力的请求升级至昂贵的二级推理，在生产环境中将推理成本降低约10%，同时保持推荐质量且几乎不影响延迟。 |
| [^16] | [RedMall: Generative Recommendation for Large-Scale E-commerce](https://arxiv.org/abs/2611.08519) | RedMall通过对多模态表示进行残差量化构建层次化语义ID，并引入轻量级SID编码器显式建模逐层依赖关系，解决了大规模电商场景下生成式推荐中语义ID层次依赖被忽略及会话级推荐优化不足的问题。 |
| [^17] | [Improving Contextual Retrieval for Long Documents using Semantic Chunking and Hybrid Retrieval Strategies](https://arxiv.org/abs/2611.08498) | 该论文提出一种混合RAG框架，通过知识图谱结构化推理与稠密向量检索相结合，配合语义分块和分层检索机制，有效解决了长文档多跳推理中的上下文碎片化和检索精度问题。 |
| [^18] | [CacheGraft: Enhancing Retrieval-Augmented Generation via Latent-Space Document Enrichment](https://arxiv.org/abs/2611.07393) | CacheGraft基于缓存到缓存（C2C）融合框架，将语料库段落离线编码为潜空间KV缓存表示，在查询时直接注入小模型，无需大模型参与即可增强检索增强生成（RAG）的答案质量。 |
| [^19] | [Unified Heterogeneous Feature Interaction via Effective-Rank Scaling for Recommender Systems](https://arxiv.org/abs/2610.35318) | 提出了HiRank框架，通过有效秩缩放在统一主干中联合建模异构特征交互与门控序列注意力，解决了推荐系统中序列与非序列特征信息交换受限以及多场景负迁移的问题。 |
| [^20] | [PULSAR: Pooled Unified Late-Interaction Search and Retrieval for Enterprise Visual Document RAG](https://arxiv.org/abs/2608.28572) | PULSAR是一个部署在企业生产环境的视觉优先检索系统，通过池化两阶段晚期交互索引，在检索精度损失小于0.01的前提下将向量搜索延迟降低15.1倍、并发吞吐量提升约88倍。 |
| [^21] | [RACORN: Robust Predicate-Agnostic Filtered Vector Search under Extreme Selectivity](https://arxiv.org/abs/2607.00768) | RACORN-1 通过自适应搜索回退机制将过滤失败的节点作为“桥”引入候选队列，使谓词无关的过滤向量搜索在低于 1% 的极端选择性下仍能保持高召回率。 |
| [^22] | [Benchmarking Patent Embeddings: A Multi-Task Evaluation of 22 Models Across Retrieval, Classification, and Clustering](https://arxiv.org/abs/2605.24297) | 该研究对22个预训练嵌入模型（参数规模22M至12B）在专利检索、分类和聚类三大任务上进行了大规模基准测试，发现最优微调方案取决于下游任务类型——横截面配准方案最利于提升检索性能，而组合信号方案更适合分类与聚类任务。 |
| [^23] | [MCGI: Manifold-Consistent Graph Indexing for Billion-Scale Disk-Resident Vector Search](https://arxiv.org/abs/2601.01930) | 提出MCGI——一种利用局部内在维度（LID）感知数据几何的磁盘驻留图索引方法，能够自适应分配搜索开销，以缓解高维ANN搜索中的欧几里得-测地线失配问题。 |
| [^24] | [Beyond Overlap: Hierarchical Graph Abstraction for Out-of-Core Vector Search](https://arxiv.org/abs/2512.22838) | GANNS是一种层次化的核外近似最近邻搜索引擎，通过将各簇存储为SSD上互不相交的分区并利用内存中的图抽象在SSD访问前提供逻辑重叠，从而同时提升搜索准确性与I/O效率。 |
| [^25] | [Knowledge Graph-based Retrieval-Augmented Generation for Schema Matching](https://arxiv.org/abs/2501.08686) | 提出KG-RAG4SM方法，通过集成多种新颖检索机制与重排序技术从外部知识图谱中提取相关子图，无需重新训练即在模式匹配任务上显著超越现有基于LLM的最先进方法。 |

# 详细

[^1]: SAGE-PAIR：面向稀疏反馈推荐的源预训练路由残差自适应方法

    SAGE-PAIR: Source-Pretrained Routed Residual Adaptation for Sparse-Feedback Recommendation

    [https://arxiv.org/abs/2611.09728](https://arxiv.org/abs/2611.09728)

    提出SAGE-PAIR，通过在源时期预训练低维残差校正空间并以源用户画像为条件添加软路由专家机制，仅需0.05-1.0%的稀疏目标反馈即可校准残差状态而无需重新学习推荐器，有效解决了极端稀疏反馈下更新方向辨识度弱的问题。

    

    推荐系统常常在足够的结果数据可用于重新训练已部署模型之前，就进入了新的服务窗口。我们在0.05-1.0%目标反馈的范围内研究这一情形，该范围涵盖了我们的压力测试，同时使监督信号远低于源推荐器的自由度。在这种规模下，现有的微调和元学习方法仍然从极其稀疏的反馈中估计目标特定的更新，导致更新方向辨识度较弱。SAGE-PAIR则转而迁移一个由源数据支持的残差校正空间，并仅使用目标结果来校准该空间。我们提出SAGE-PAIR，这是一种分阶段适配器：它在相邻的源时期上预训练一个低维全局残差，并添加一个以源用户画像为条件的残差专家软路由器。目标反馈随后校准这一残差状态，而不是重新学习推荐器。在KuaiSAR-v2的锁定测试集划分上，SAGE-PAIR（Fu……

    arXiv:2611.09728v1 Announce Type: new  Abstract: Recommendation systems often enter a new serving window before enough outcomes are available to retrain the deployed model. We study this regime with 0.05-1.0% target feedback, a range that spans our stress tests while keeping supervision far below the degrees of freedom of the source recommender. At this scale, existing fine-tuning and meta-learning methods still estimate target-specific updates from extremely sparse feedback, leaving the update direction weakly identified. SAGE-PAIR instead transfers a source-supported residual correction space and only calibrates this space using target outcomes. We introduce SAGE-PAIR, a staged adapter that pretrains a low-dimensional global residual on an adjacent source period and adds a soft router over residual experts conditioned on source user profiles. Target feedback then calibrates this residual state rather than relearning the recommender. On a locked test split of KuaiSAR-v2, SAGE-PAIR (Fu
    
[^2]: SPARSEUP：面向现代学习型稀疏检索器训练的更优方案

    SPARSEUP: Towards Better Recipes for Training Modern Learned Sparse Retrievers

    [https://arxiv.org/abs/2611.09680](https://arxiv.org/abs/2611.09680)

    本文提出 SPARSEUP 训练方案，通过 logit 分布调控、逐位置扇出硬上限以及词汇表合并策略，让基于现代骨干网络的 Splade 类稀疏检索器实现更优的效率-效果权衡。

    

    学习型稀疏嵌入是稠密检索的一种强大替代方案，然而其发展已落后于单向量模型和迟交互模型。其应用受限的原因之一在于它们以难以训练而著称，通常需要非常严格的条件才能收敛。我们提出了一套在现代骨干网络和数据上训练 Splade 类编码器的完整方案，能够实现更优的效率与效果权衡。我们讨论了 logit 分布的作用及其对训练稳定性和编码器最终稀疏性的影响。我们还引入了一种新的逐位置扇出控制机制，即对每个位置可以激活的词汇维度数量设置硬性上限。此外，我们进一步应用了一种简单的词汇表合并策略，用于去除冗余词条，例如同一单词的大小写变体和词边界变体，这类冗余在基于区分大小写 BPE 的骨干网络中十分常见。我们将该方案应用于 LateOn 骨干网络和训练数据，由此得到了 SPARSEUP 模型。

    arXiv:2611.09680v1 Announce Type: new  Abstract: Learned sparse embeddings are a powerful alternative to dense retrieval, yet they have fallen behind single-vector and late interaction models. One reason for their limited adoption is that they are notoriously difficult to train, generally requiring very tight conditions to converge. We present a recipe for training Splade-like encoders on modern backbones and data that achieves a better efficiency-effectiveness trade-off. We discuss the role of logit distribution and its impact on the training stability and the final sparsity of the encoder. We also introduce a new per-position fan-out control - a hard cap on how many vocabulary dimensions each position may emit. We further apply a simple vocabulary merging strategy that removes redundant entries - such as case and word-boundary variants of the same word - which are common in cased BPE-based backbones. We apply our recipe to the LateOn backbone and training data, obtaining SPARSEUP, a 
    
[^3]: ENGRAM：通过验证式巩固将智能体经验融入模型权重

    ENGRAM: From Agent Experience to Weights through Verified Consolidation

    [https://arxiv.org/abs/2611.09472](https://arxiv.org/abs/2611.09472)

    ENGRAM框架将训练准入与记忆释放分离，通过示范感知门控、印迹轨迹图上下文学习和验证门控，只有当经验被有效训练且更新后的模型在无需检索的情况下仍达到相对和绝对评分标准时，才将外部记忆释放移除。

    

    语言模型智能体通过外部检索和参数学习来利用经验。将存储的经验巩固到模型权重中，既需要决定训练哪些经验，又需要决定何时其外部记录可以从检索中移除。仅基于学生模型掌握程度的准入可能会排除有用的教师示范，而训练后立即移除可能会丢弃模型仍然需要的指导。我们提出了ENGRAM，一个将训练准入与记忆释放相分离的框架。一个感知示范的门控基于学生掌握程度或足够多的成功教师示范来准入技能。学生模型从选定的动作轨迹中学习，并结合来自“印迹轨迹图”的上下文。随后，一个验证门控在保留的探针任务上，将不使用情节检索的更新后模型与使用检索的基础模型进行比较。记忆的释放要求经验参与了有效训练，并且得分同时满足相对和绝对标准。

    arXiv:2611.09472v1 Announce Type: cross  Abstract: Language-model agents use experience through external retrieval and parameter learning. Consolidating stored experience into model weights requires deciding both which experience to train on and when its external records can leave retrieval. Admission based solely on student mastery can exclude useful teacher demonstrations, while removal immediately after training can discard guidance that the model still needs. We introduce ENGRAM, a framework that separates training admission from memory release. A demonstration-aware gate admits skills based on student mastery or sufficient successful teacher demonstrations. The student learns from selected action traces with context from an Engram Trace Graph (ETG). A verification gate then compares the updated model without episodic retrieval against the base model with retrieval on reserved probes. Release requires valid training participation and scores meeting both relative and absolute criter
    
[^4]: 双塔检索中的正交范数增长：统一发现场景下的流行度机制

    Orthogonal Norm Growth in Two-Tower Retrieval: A Popularity Mechanism for Unified Discovery

    [https://arxiv.org/abs/2611.09294](https://arxiv.org/abs/2611.09294)

    该论文揭示了双塔检索中仅余弦损失在特定充分条件下会导致物品向量范数随重复更新而正交增长，当搜索与推荐任务共享物品编码器并以原始内积服务时，这种增长会累积形成一种内在的流行度偏差机制。

    

    统一搜索与推荐系统可能会从两个交互流（搜索流与推荐流）更新同一份物品向量。我们研究重复更新如何增加物品向量的范数，并在未归一化点积的检索方式下产生影响。仅使用余弦损失时，输出梯度垂直于当前物品向量——即与其恒定范数的球面相切。然而训练会改变编码器参数，因此实际产生的物品向量位移不一定保持切向。我们推导出使位移保持切向的充分条件；在此条件下，每一次非零更新都会使平方范数增加其自身长度的平方。当两个任务共享物品编码器时，来自两个任务的顺序更新可以在同一个范数上不断累积，而使用原始内积进行在线服务时，任一任务都可能暴露于这种范数增长之下。需要强调的是，这一条件性结果并非关于联合SGD的定理。受控消融实验对这些条件进行了检验。在MovieLens 32M和Amazon Books数据集上，累积的切向位移伴随着频率-范数关联现象出现。

    arXiv:2611.09294v1 Announce Type: new  Abstract: Unified search and recommendation systems may update the same item vectors from both interaction streams. We study how repeated updates can increase an item vector's norm and affect retrieval under an unnormalized dot product. A cosine-only loss makes the output gradient perpendicular to the current item vector - tangent to its constant-norm sphere. Training changes encoder parameters, however, so the realized item-vector movement need not remain tangent. We derive sufficient conditions under which it does; every nonzero update then increases squared norm by its squared length. When the tasks share an item encoder, sequential updates from both can accumulate in the same norm, which raw-inner-product serving can expose for either task. This conditional result is not a theorem about joint SGD. Controlled ablations test these conditions. On MovieLens 32M and Amazon Books, accumulated tangent movement accompanies frequency-norm association a
    
[^5]: FondueQA：用于评估多跳问答中参数化补偿能力的基准

    FondueQA: A Benchmark for Evaluating Parametric Compensation in Multi-Hop QA

    [https://arxiv.org/abs/2611.09274](https://arxiv.org/abs/2611.09274)

    提出FondueQA基准，通过将每个多跳事实唯一映射到指定证据文档并执行受控的文档级证据消融，评估大语言模型在不完整检索下利用内部参数化知识进行补偿的能力，同时避免基准污染问题。

    

    大型语言模型（LLM）展现出基于参数化知识的卓越性能，即其能够回忆起内化在模型权重中的事实与关联。然而，在现有的多跳问答基准上研究这一能力可能会受到基准污染的干扰，因为许多此类基准已公开发布很长时间，很可能已出现在现代预训练语料库中。因此，在证据缺失情况下给出的正确答案，可能反映的是模型的一般参数化知识，也可能是对基准特定实例的记忆。为了研究模型在检索证据不完整时如何利用内部知识，我们提出了FondueQA，其中每个所需的多跳事实都被唯一地定位到指定的证据文档。这使得我们可以进行受控的文档级证据消融，以研究在不完整检索条件下的参数化补偿。我们的实证评估表明，参数化补偿能力……

    arXiv:2611.09274v1 Announce Type: cross  Abstract: Large Language Models (LLMs) show remarkable performance based on parametric knowledge, i.e., their ability to recall facts and associations internalized in a model's weights. However, studying this capability on established multi-hop benchmarks may be complicated by benchmark contamination, as many of these benchmarks have been publicly available long enough to plausibly appear in modern pretraining corpora. As a result, correct answers under missing evidence may reflect either general parametric knowledge or memorization of benchmark-specific instances. To study how models use internal knowledge when retrieval evidence is incomplete, we introduce FondueQA, in which every required multi-hop fact is uniquely localized to a designated evidence document. This enables controlled document-level evidence ablations to study parametric compensation under incomplete retrieval. Our empirical evaluation reveals that parametric compensation stron
    
[^6]: 基于结构与谱系证据的越南法律条文检索研究（DRiLL数据集）

    Structure and Lineage Evidence for Vietnamese Statutory Article Retrieval on DRiLL

    [https://arxiv.org/abs/2611.09200](https://arxiv.org/abs/2611.09200)

    该论文通过引入条款身份、同一法律支持以及引用与谱系证据等结构化特征，将DRiLL数据集上越南法律条文检索的官方F2指标从71.77%提升至72.91%。

    

    我们在DRiLL数据集上研究越南法律条文检索，使用的特征包括条款身份与局部形式、同一法律内的支持信息，以及引用与谱系证据。在五个随机种子下，采用共享候选的selector、列表式Qwen证据和相同学习器设置，将官方F2指标从71.77%提升至72.91%，其中在双条文问题上的描述性提升最大。新增特征占所提出selector的总决策树分裂增益（以训练目标减少量衡量）的18.20%，主要来自同一法律和引用证据。案例分析揭示了基础条文与修正案、关联条文的恢复、干扰项的剔除以及失败情形。这些发现为在其他数据集、表示和学习器上的研究提供了启发，但并未确立单个特征的独立效应，也不依赖LightGBM、手工构造的分数或显式图结构。

    arXiv:2611.09200v1 Announce Type: cross  Abstract: We study Vietnamese statutory Article retrieval on DRiLL using features for provision identity and local form, same-law support, and citation and lineage evidence. Across five random seeds, selectors sharing candidates, listwise Qwen evidence, and learner settings improve official $F_2$ from 71.77% to 72.91%, with the largest descriptive gain on two-Article questions. Added features account for 18.20% of the proposed selector's total decision-tree split gain, measured as training-objective reduction, mainly from same-law and citation evidence. Cases reveal base provisions and amendments, companion recovery, distractor removal, and failures. These findings motivate research on other datasets, representations, and learners without establishing individual feature effects or requiring LightGBM, hand-crafted scores, or explicit graphs.
    
[^7]: 基于大语言模型共识与语料库自适应主题建模的专利创新分类与映射

    Large Language Model Consensus and Corpus-Adaptive Topic Modeling for Classifying and Mapping Patent Innovation

    [https://arxiv.org/abs/2611.09194](https://arxiv.org/abs/2611.09194)

    该研究提出了一个可审计的专利分析流水线，通过三个大语言模型的投票共识机制对专利的新颖性与创新类型进行分类，并结合随语料库规模自适应调参的PatentSBERTa嵌入聚类实现主题建模与解读。

    

    专利分析日益依赖于Transformer表示和大语言模型（LLMs），然而要将概念分类、主题形成与解读结合起来，需要对模型角色以及各阶段所保留证据作出明确的决策。在这项以设计为导向的研究中，我们提出并演示了一个将这些任务串联起来的专利分析流水线，并使每一项决策都可供审计。三个大语言模型分别按新颖性（突破性或渐进性）和创新类型（产品、流程、服务）对每件专利进行分类。突破性标签和创新类型标签均需获得两张支持票，而渐进性则作为新颖性标签的兜底选项。主题通过对PatentSBERTa嵌入进行聚类来发现，其参数范围根据语料库规模进行缩放。专利专用停用词表仅在使用基于类别的词频-逆文档频率（c-TF-IDF）表示主题时才被应用，并且第二个嵌入模型……（原文摘要在此处被截断）

    arXiv:2611.09194v1 Announce Type: cross  Abstract: Patent analytics increasingly relies on transformer representations and large language models (LLMs), yet linking conceptual classification, topic formation and interpretation requires explicit decisions about model roles and the evidence retained at each stage. In this design-oriented study, we propose and demonstrate a patent analytics pipeline that links these tasks and keeps each decision open to audit. Three LLMs separately classify each patent by novelty (radical or incremental) and innovation type (product, process, service). Radical and innovation-type labels each require two supporting votes, with Incremental used as the fallback novelty label. Topics are discovered by clustering PatentSBERTa embeddings, with parameter ranges scaled to corpus size. A patent-specific stopword list is applied only when topics are represented with class-based term frequency--inverse document frequency (c-TF-IDF), and a second embedding model supp
    
[^8]: 一个基于分类学驱动的合成数据集：用于招聘场景下的检索有效性与公平性评估

    A Taxonomy-Driven Synthetic Dataset for Retrieval Effectiveness and Fairness Evaluation in Hiring

    [https://arxiv.org/abs/2611.09178](https://arxiv.org/abs/2611.09178)

    该论文提出了一个基于分类学驱动、完全合成的多语言HR数据集与基准，可同时评估检索有效性与算法公平性，揭示了通用基准在招聘场景中的不足，并为识别兼顾语义准确性与公平性的帕累托最优模型提供了定制化评估方案。

    

    尽管机器学习招聘工具已无处不在，但隐私法规和商业敏感性的要求使得公开基准测试在法律上变得复杂。为应对这一挑战，我们引入了一个基于分类学驱动的完全合成的HR数据集和基准，可同时评估检索有效性和算法公平性。该多语言语料库基于66万份真实自由职业者档案构建，包含候选人档案、项目简介以及反事实人口统计孪生体。对最先进的嵌入模型的审计显示，其性能与通用MTEB分数存在差异，同时在具有语法性别的语言中也存在公平性差异。这表明通用基准不足以满足HR领域的需求，需要定制化的评估来识别在语义准确性与公平性之间取得平衡的帕累托最优模型。

    arXiv:2611.09178v1 Announce Type: cross  Abstract: While machine learning hiring tools are ubiquitous, the need for privacy regulations and commercial sensitivity makes public benchmarking legally complex. We address this challenge by introducing a taxonomy-driven, fully synthetic HR dataset and benchmark to simultaneously evaluate retrieval effectiveness and algorithmic fairness. Grounded in 660K real freelancer profiles, our multilingual corpus features candidate profiles, project briefs, and counterfactual demographic twins. Auditing state-of-the-art embedding models reveals performance divergence from general MTEB scores, alongside fairness disparities in grammatically gendered languages. This demonstrates that general benchmarks are insufficient for HR, requiring tailored evaluations to identify Pareto-optimal models that balance semantic accuracy and fairness.
    
[^9]: PAGSL：面向可解释学者推荐的路径增强图结构学习

    PAGSL: Path-Augmenting Graph Structure Learning for explainable scholar recommendation

    [https://arxiv.org/abs/2611.09126](https://arxiv.org/abs/2611.09126)

    提出 PAGSL 模型，通过图增强激活潜在学术路径并利用基于路径的神经网络编码，结合带路径分数优化的掩码机制，实现高性能且解释简洁、个性化、有层次深度的可解释学者推荐。

    

    可解释学者推荐旨在识别潜在感兴趣的学者，并为推荐结果提供合理的解释。然而，现有框架难以应对优化过程中产生的噪声和“枢纽节点偏差”，导致生成的解释过于琐碎，缺乏层次深度和个性化。为此，我们提出了一种新颖的可解释学者推荐模型——路径增强图结构学习（PAGSL）。PAGSL 将表示学习与语义路径知识相结合，在保证推荐性能的同时，从两个角度提供可解释的结果。具体而言，该模型通过图增强激活实体间潜在的学术路径，并采用基于路径的神经网络对这些路径进行编码以实现推荐。此外，还设计了一种带路径分数优化的掩码机制，用于生成简洁且信息丰富的解释。我们在三个学术数据集（Scholat、AM 等）上对 PAGSL 进行了评估。

    arXiv:2611.09126v1 Announce Type: new  Abstract: Explainable scholar recommendation aims to identify potential scholars of interest and provide reasonable explanations for the recommendations. However, existing frameworks struggle with optimization-induced noise and "hub node bias", yielding trivial explanations that lack hierarchical depth and personalization. To this end, we propose a novel explainable scholar recommendation model, Path-Augmenting Graph Structure Learning (named PAGSL). PAGSL combines representation learning and semantic path knowledge to ensure recommendation performance while providing interpretable results from two perspectives. Specifically, it activates latent academic paths among entities through graph augmentation and employs a path-based neural network to encode these paths for recommendation. A masking mechanism with path score optimization is designed to generate concise and informative explanations. We evaluate PAGSL on three academic datasets (Scholat, AM
    
[^10]: MV-IVF：通过多向量聚类实现的多向量检索

    MV-IVF: Multi-Vector Retrieval via Multi-Vector Clustering

    [https://arxiv.org/abs/2611.09040](https://arxiv.org/abs/2611.09040)

    本文提出 MV-IVF，一种直接在多向量数据上构建的原生倒排文件索引，其核心是基于新的 Lloyd 风格算法 MV-Lloyd 在 Chamfer 目标下对点云聚类，从而以原生方式高效加速多向量检索。

    

    多向量检索将每个查询和文档表示为一组嵌入向量，并通过细粒度的集合到集合交互（如 MaxSim，即 Chamfer 相似度）来评分相关性。这种表达能力相较于单向量表示提升了检索质量，但也使搜索成本大幅增加。现有系统主要通过代理表示或“过滤-重排”流水线来应对这一挑战，将候选生成简化为单向量搜索。我们研究了与之互补的方向：直接在多向量数据上构建原生的 IVF 风格索引。我们提出了 MV-IVF，一种面向多向量检索的倒排文件索引，它建立在 MV-Lloyd 之上——一种新的 Lloyd 风格算法，用于在 Chamfer 目标下对点云进行聚类。我们的关键观察是：一旦点云被分配到簇中，中心更新步骤就简化为对每个簇内展平的嵌入向量进行标准的单向量聚类。

    arXiv:2611.09040v1 Announce Type: new  Abstract: Multi-vector retrieval represents each query and document as a set of embeddings and scores relevance via fine-grained set-to-set interactions such as MaxSim (Chamfer similarity). This expressiveness improves retrieval quality over single-vector representations, but makes search substantially more expensive. Existing systems address this challenge primarily through proxy representations or filter-and-rerank pipelines that reduce candidate generation to single-vector search. We study the complementary direction of building a native IVF-style index directly over multi-vector data.   We introduce MV-IVF, an inverted-file index for multi-vector retrieval built on MV-Lloyd, a new Lloyd-style algorithm for clustering point clouds under a Chamfer objective. Our key observation is that once point clouds are assigned to clusters, the center-update step reduces to standard single-vector clustering over the flattened embeddings in each cluster. Thi
    
[^11]: 重新思考生成式推荐中的潜在推理：超越教师生成的思维链与循环隐状态反馈

    Rethinking Latent Reasoning for Generative Recommendation: Beyond Teacher-Generated CoT and Recurrent Hidden-State Feedback

    [https://arxiv.org/abs/2611.09006](https://arxiv.org/abs/2611.09006)

    提出PHiT方法，直接从物品历史中提取语义目标来学习思维令牌的中间状态，从而摆脱了对昂贵的教师生成思维链监督和专门的循环隐状态执行循环的依赖。

    

    生成式推荐中的潜在推理面临两个实际障碍：构建教师生成的思维链监督成本高昂，而循环隐状态反馈需要专门的执行循环。我们重新审视了监督来源和计算接口。一项步骤级诊断衡量了推荐思维链如何支持下一物品预测和轨迹连贯性。在所分析的轨迹中，历史分析是唯一在两个维度上都具有正平均贡献的步骤，而后面的偏好和意图步骤则具有负的平均答案贡献。受此发现启发，我们提出了PHiT（预填充的历史信息驱动思维令牌，Prefilled, History-Informed Thought Tokens），它从直接观察到的物品历史中提取语义目标，并据此学习中间状态。PHiT聚合缓存的稀疏物品编码来构建这些目标，并监督不同词汇思维令牌的上下文状态。（摘要在此处被截断）

    arXiv:2611.09006v1 Announce Type: new  Abstract: Latent reasoning in generative recommendation faces two practical barriers: constructing teacher-generated chain-of-thought (CoT) supervision is costly, and recurrent hidden-state feedback requires a specialized execution loop. We revisit both the supervision source and the computation interface. A step-level diagnosis measures how recommendation CoT supports next-item prediction and trace coherence. In the analyzed traces, history analysis is the only step with positive mean contributions on both dimensions, while later preference and intent steps have negative mean answer contributions. Guided by this finding, we propose PHiT (Prefilled, History-Informed Thought Tokens), which learns intermediate states from semantic targets extracted directly from observed item histories. PHiT aggregates cached sparse item encodings to construct these targets and supervises the contextual states of distinct vocabulary thought tokens. All thought token
    
[^12]: SkinMind：为智能体RAG配备类人图探索能力，实现可靠的皮肤病学AI

    SkinMind: Equipping Agentic RAG with Human-Inspired Graph Probing for Reliable Dermatology AI

    [https://arxiv.org/abs/2611.08935](https://arxiv.org/abs/2611.08935)

    提出了SkinMind多模态智能体RAG框架，通过八个层次化的类人图探索工具和编排器与图环境的迭代交互，实现渐进式披露的自主证据搜寻，从而提升皮肤病学AI在异构多模态来源中的可靠性。

    

    可靠的皮肤病学AI需要访问分布于异构文本和视觉来源中的可验证证据。然而，传统的检索增强生成（RAG）通常依赖于从扁平语料库中进行一次性检索，这限制了其适应检索粒度、探索多跳关系以及整合多模态临床证据的能力。我们提出了SkinMind，这是一个多模态智能体RAG框架，为策略模型配备类人的图探索能力以实现自主证据搜寻。SkinMind将检索组织为编排器与图环境之间的迭代交互。通过八个层次化的图探索工具，编排器逐步识别相关实体、检查节点级信息、遍历关系和路径，并检索源文本和视觉证据。这种渐进式披露设计使模型能够自适应地确定检索模态、探索（摘要在此处被截断）

    arXiv:2611.08935v1 Announce Type: new  Abstract: Reliable dermatology AI requires access to verifiable evidence distributed across heterogeneous textual and visual sources. However, conventional retrieval-augmented generation (RAG) typically relies on one-shot retrieval from flat corpora, limiting its ability to adapt retrieval granularity, explore multi-hop relations, and integrate multimodal clinical evidence. We introduce SkinMind, a multimodal agentic RAG framework that equips policy models with human-inspired graph probing for autonomous evidence seeking. SkinMind organizes retrieval as an iterative interaction between an orchestrator and a graph environment. Through eight hierarchical graph-probing tools, the orchestrator progressively identifies relevant entities, inspects node-level information, traverses relations and paths, and retrieves source-text and visual evidence. This progressive-disclosure design allows the model to adaptively determine the retrieval modality, explora
    
[^13]: Shopify 的生成式搜索与推荐

    Generative Search and Recommendation at Shopify

    [https://arxiv.org/abs/2611.08804](https://arxiv.org/abs/2611.08804)

    Shopify 提出 SID-LLM，将搜索与推荐统一建模为生成语义 ID 的任务，通过目录与行为数据联合预训练、强化学习、专家蒸馏以及多 token 预测结合轻量级排序器，实现了高质量、低延迟的检索与推荐。

    

    Shopify 的搜索和推荐系统需要在严格的延迟约束下，从不断变化的数十亿商品目录中检索相关商品。我们提出了 SID-LLM，将这两项任务都建模为生成语义 ID：即由内容派生的 token 组成的、用于表示商品的短序列。我们在商品目录数据和行为数据上对 SID-LLM 进行联合预训练，为搜索和推荐提供了共享的表示和训练目标。我们证明强化学习可以进一步提升经人工评判的搜索相关性。此外，通过从任务特定的专家模型进行蒸馏，相比联合监督微调能够提升推荐质量，同时在统计上未检测到搜索质量的损失。最后，为了高效服务，我们将多 token 预测与轻量级排序器相结合，在降低延迟的同时提升了相对于自回归基线的检索质量。我们既将 SID-LLM 作为独立检索器进行评估，也将其作为 Shopify 生产系统的组件进行评估。

    arXiv:2611.08804v1 Announce Type: new  Abstract: Search and recommendation at Shopify require retrieving relevant products from a changing catalog of billions of products under strict latency constraints. We present SID-LLM, which casts both tasks as generating semantic IDs: short sequences of content-derived tokens representing products. We jointly pre-train SID-LLM on catalog and behavioral data, giving search and recommendation a shared representation and training objective. We show that reinforcement learning further improves judged search relevance. Furthermore, distillation from task-specific experts improves recommendation quality over joint supervised fine-tuning without a statistically detected loss in search quality. Finally, for efficient serving, we pair multi-token prediction with a lightweight ranker, improving retrieval quality over our autoregressive baseline while reducing latency. We evaluate SID-LLM both as a standalone retriever and as a component of Shopify's produ
    
[^14]: SteerGR：基于稀疏自编码器的生成式推荐引导，实现可控的游戏发现

    SteerGR: Steering Generative Recommendation via Sparse Autoencoders for Controllable Game Discovery

    [https://arxiv.org/abs/2611.08790](https://arxiv.org/abs/2611.08790)

    提出SteerGR框架，利用稀疏自编码器将人类可读的游玩概念与生成式推荐模型的内部特征关联，通过直接干预所选特征实现无需重训练的概念级可控游戏推荐。

    

    生成式推荐旨在从用户交互历史中预测物品标识符，但对推荐物品属性的显式控制仍未得到充分探索。一个关键挑战是将人类可读的概念与内部特征关联起来，使得对这些特征的干预能够有效改变推荐的物品。为解决这一问题，我们提出了可引导生成式推荐器（SteerGR），这是一个利用稀疏自编码器在游戏发现中实现概念级控制的框架。SteerGR通过三个阶段进行：（i）概念定义与接地，使用大语言模型定义游玩概念；（ii）特征发现与选择，根据概念关联发现候选特征，并根据其引导效应选择特征；（iii）稀疏特征干预，通过放大或抑制所选特征来引导推荐，而无需重新训练推荐器。在两个工业级生成式推荐模型上进行了大量实验……

    arXiv:2611.08790v1 Announce Type: cross  Abstract: Generative recommendation aims to predict item identifiers from user interaction histories, but explicit control over the properties of recommended items remains underexplored. A key challenge is to connect human-readable concepts to internal features whose interventions effectively change the recommended items. To address this, we propose Steerable Generative Recommender (SteerGR), a framework that uses sparse autoencoders for concept-level control in game discovery. SteerGR proceeds through three stages: (i) Concept Definition and Grounding for defining play concepts with an LLM; (ii) Feature Discovery and Selection for discovering candidate features by their concept associations and selecting one by its steering effect; and (iii) Sparse Feature Intervention for amplifying or suppressing selected features to steer recommendations without retraining the recommender. Extensive experiments with two industrial generative recommendation m
    
[^15]: CascadeRec：面向多租户内容推荐的已部署自适应LLM算力分配方法

    CascadeRec: Deployed Adaptive LLM Compute Allocation for Multi-Tenant Content Recommendation

    [https://arxiv.org/abs/2611.08566](https://arxiv.org/abs/2611.08566)

    CascadeRec通过两级LLM级联与自适应路由，只将可能受益于额外算力的请求升级至昂贵的二级推理，在生产环境中将推理成本降低约10%，同时保持推荐质量且几乎不影响延迟。

    

    大语言模型（LLM）作为推荐系统的重排序器极具吸引力。但在每个请求上都运行昂贵模型的成本很高，而且对于交互历史稀疏的用户，这样做往往会降低推荐质量。我们提出了CascadeRec，一个两级LLM重排序级联系统，已部署于一家大型财务规划与分析（FP&A）SaaS提供商的生产环境中，在AWS EKS和GKE上为 企业租户提供内容推荐服务。轻量级的一级（Tier-1）模型对每个请求进行重排序；路由器仅将那些额外的推理时算力可能带来收益的请求进行升级，并将其发送给更昂贵的二级（Tier-2）流程（自洽性采样、多路径推理或更大的模型）。该系统已在生产环境A/B实验中运行超过一个月。与统一使用昂贵层级相比，CascadeRec将推理成本降低了约10%，延迟影响极小，且迄今为止未观察到质量回退。租户数据无法对外……（原文摘要至此截断）

    arXiv:2611.08566v1 Announce Type: new  Abstract: Large language models (LLMs) are attractive rerankers for recommendation. Running an expensive model on every request is costly, and for users with sparse interaction histories it often lowers quality. We present CascadeRec, a two-tier LLM reranking cascade deployed in production at a large financial planning and analysis (FP&A) SaaS provider. It serves content recommendations to enterprise tenants on AWS EKS and GKE. A lightweight Tier-1 model reranks every request. A router escalates only the requests where extra inference-time compute is likely to help, and sends them to a more expensive Tier-2 procedure (self-consistency sampling, multi-path reasoning, or a larger model). The system has run in a production A/B experiment for over a month. Compared with uniform use of the expensive tier, CascadeRec has reduced inference cost by about 10%, with minimal latency impact and no quality regression observed to date. Tenant data cannot be rel
    
[^16]: RedMall：面向大规模电商的生成式推荐

    RedMall: Generative Recommendation for Large-Scale E-commerce

    [https://arxiv.org/abs/2611.08519](https://arxiv.org/abs/2611.08519)

    RedMall通过对多模态表示进行残差量化构建层次化语义ID，并引入轻量级SID编码器显式建模逐层依赖关系，解决了大规模电商场景下生成式推荐中语义ID层次依赖被忽略及会话级推荐优化不足的问题。

    

    生成式推荐在工业推荐系统中已展现出可观的潜力。然而，现有生成式方法的性能受到以下几项挑战的制约：1) 语义ID（SID）被视为一维序列，忽略了不同层级之间的层次化依赖关系；2) 所使用的下一词元预测方法无法显式地优化会话级推荐；3) 尽管强化学习已被应用于将生成的推荐与用户偏好对齐，但现有方法中鲜有显式优化会话级推荐中正样本物品排序的。为解决上述问题，我们提出了RedMall，一个面向大规模电商的生成式推荐器。首先，RedMall通过对多模态表示进行残差量化来构建层次化语义ID（SID）。其次，RedMall引入了一个轻量级的SID编码器，以显式建模SID中的逐层依赖关系。

    arXiv:2611.08519v1 Announce Type: new  Abstract: Generative recommendation has shown considerable potential in industrial recommendation systems. However, the performance of existing generative methods is limited by several challenges. 1) Semantic IDs (SIDs) are treated as one-dimensional sequences, overlooking the hierarchical dependencies among different layers. 2) Next-token prediction used is unable to explicitly optimize session-wise recommendation. 3) Although reinforcement learning has been applied to align generated recommendations with user preferences, few existing methods explicitly optimize the ranking of positive items in session-wise recommendation. To address the above issues, we propose RedMall, a generative recommender designed for large-scale e-commerce. Firstly, RedMall constructs hierarchical SIDs through residual quantization over multimodal representations. Secondly, RedMall introduces a lightweight SID encoder to explicitly model layer-wise dependencies in SIDs. 
    
[^17]: 使用语义分块和混合检索策略改进长文档的上下文检索

    Improving Contextual Retrieval for Long Documents using Semantic Chunking and Hybrid Retrieval Strategies

    [https://arxiv.org/abs/2611.08498](https://arxiv.org/abs/2611.08498)

    该论文提出一种混合RAG框架，通过知识图谱结构化推理与稠密向量检索相结合，配合语义分块和分层检索机制，有效解决了长文档多跳推理中的上下文碎片化和检索精度问题。

    

    篇幅冗长且内容分散的文档中包含的海量信息使得查找相关答案变得困难，尤其是当答案需要连接来自多个来源的证据时。标准的检索增强生成（RAG）系统通常难以应对这种“多跳”推理，并且经常面临上下文碎片化的问题。在本文中，我们提出了一种混合RAG框架，该框架将知识图谱的结构化推理与稠密向量检索的广泛语义覆盖相结合。我们的方法采用多阶段流水线，利用基于transformer的摘要生成和语义分块来改善上下文保留。我们构建了一个动态知识图谱来捕获实体之间的显式关系，同时维护一个稠密向量索引。在推理过程中，系统使用分层机制，优先进行图遍历以保证精确性，并在需要时回退到向量检索。

    arXiv:2611.08498v1 Announce Type: cross  Abstract: The vast amount of information available in lengthy, distributed documents makes finding relevant answers difficult, especially when the answer requires connecting evidence from multiple sources. Standard Retrieval-Augmented Generation (RAG) systems often struggle with this type of "multi-hop" reasoning and frequently face issues with context fragmentation. In this paper, we propose a hybrid RAG framework that integrates the structured reasoning of knowledge graphs with the broad semantic coverage of dense vector retrieval. Our approach employs a multi-stage pipeline using transformer-based summarization and semantic chunking to improve context retention. We construct a dynamic knowledge graph to capture explicit relationships between entities while simultaneously maintaining a dense vector index. During inference, the system uses a hierarchical mechanism that prioritizes graph traversal for precision and falls back to vector retrieval
    
[^18]: CacheGraft：通过潜空间文档增强来提升检索增强生成

    CacheGraft: Enhancing Retrieval-Augmented Generation via Latent-Space Document Enrichment

    [https://arxiv.org/abs/2611.07393](https://arxiv.org/abs/2611.07393)

    CacheGraft基于缓存到缓存（C2C）融合框架，将语料库段落离线编码为潜空间KV缓存表示，在查询时直接注入小模型，无需大模型参与即可增强检索增强生成（RAG）的答案质量。

    

    机器学习领域的最新研究表明，基于Transformer的语言模型代理不仅可以直接通过文本进行通信，还可以在潜空间中通过键值（KV）缓存直接进行通信。在本文中，我们提出这种缓存到缓存的翻译技术可以提供超越其实时代理通信这一最初应用场景的实用价值，并可用于检索增强生成（RAG）。基于缓存到缓存（C2C）融合框架，我们提出的系统CacheGraft将每个语料库段落离线编码为较大共享模型（实验中为4B参数）和较小接收模型（0.6B参数）之间的融合KV缓存表示，使得该缓存可以在查询时注入到接收模型中，而无需较大模型。在五个问答基准测试和两个独立本地运行的评估器上，CacheGraft在29个案例中（摘要在此处被截断）相比纯文本接收模型基线提升了答案质量。

    arXiv:2611.07393v1 Announce Type: cross  Abstract: Recent work in machine learning has demonstrated that transformer-based language model agents can communicate state not only through text, but also directly in latent space via key-value (KV) caches. In this paper, we argue that such cache-to-cache translation can provide utility beyond the original motivating application of real-time agent communication and can instead be utilized for retrieval-augmented generation (RAG). Building on the Cache-to-Cache (C2C) fusion framework, our proposed system CacheGraft encodes each corpus passage offline into a fused KV cache representation between a larger sharer model (4B parameters in our experiments) and a smaller receiver model (0.6B parameters) such that this cache can be injected into the receiver at query time with no need for the larger model. Across five QA benchmarks and two independent locally-run judges, CacheGraft improves answer quality over the text-only receiver baseline in 29 out
    
[^19]: 基于有效秩缩放的推荐系统统一异构特征交互

    Unified Heterogeneous Feature Interaction via Effective-Rank Scaling for Recommender Systems

    [https://arxiv.org/abs/2610.35318](https://arxiv.org/abs/2610.35318)

    提出了HiRank框架，通过有效秩缩放在统一主干中联合建模异构特征交互与门控序列注意力，解决了推荐系统中序列与非序列特征信息交换受限以及多场景负迁移的问题。

    

    大规模工业级排序模型必须在严格的在线服务约束下，同时捕获异构特征交互、长期用户行为和多场景依赖关系。现有架构通常使用独立的模块来处理序列特征和非序列特征，限制了双向信息交换。此外，当序列信号和非序列信号表现出不同的有效秩时，直接对所有语义标记应用同构自注意力可能效率低下。同时，共享表征可能会忽略场景特定的模式，并在不同场景之间引发负迁移。在本文中，我们提出了HiRank，这是一个用于多场景推荐的统一框架，通过有效秩缩放联合建模异构特征交互。HiRank在统一的主干网络中将门控序列注意力与异构特征交互相结合，实现双向……（摘要在此处截断）

    arXiv:2610.35318v1 Announce Type: new  Abstract: Large-scale industrial ranking models must jointly capture heterogeneous feature interactions, long-term user behaviors, and multi-scenario dependencies under stringent online serving constraints. Existing architectures typically process sequential and non-sequential features with separate modules, limiting bidirectional information exchange. Directly applying homogeneous self-attention to all semantic tokens can also be inefficient when sequential and non-sequential signals exhibit different effective-ranks (eRanks). Meanwhile, shared representations can overlook scenario-specific patterns and induce negative transfer between scenarios. In this paper, we propose HiRank, a unified framework for multi-scenario recommendation that jointly models Heterogeneous feature Interactions via effective-Rank scaling. HiRank integrates gated sequential attention with heterogeneous feature interaction within a unified backbone, enabling bidirectional 
    
[^20]: PULSAR：面向企业级视觉文档RAG的池化统一晚期交互搜索与检索系统

    PULSAR: Pooled Unified Late-Interaction Search and Retrieval for Enterprise Visual Document RAG

    [https://arxiv.org/abs/2608.28572](https://arxiv.org/abs/2608.28572)

    PULSAR是一个部署在企业生产环境的视觉优先检索系统，通过池化两阶段晚期交互索引，在检索精度损失小于0.01的前提下将向量搜索延迟降低15.1倍、并发吞吐量提升约88倍。

    

    机构投资者需要在交易临近交割时每小时都在更新的视觉密集型路演材料、董事会资料包和尽职调查材料中进行搜索。OCR后接图表语言化的方法在这种规模下刷新成本高昂，且可能丢失图表细节。我们提出PULSAR，一个部署于穆巴达拉投资公司的生产级视觉优先检索系统。PULSAR使用冻结的ColPali风格骨干网络对页面图像进行索引，并采用池化的两阶段晚期交互索引：紧凑的页面摘要支持初始检索，随后在更精细的池化表示上执行精确的MaxSim重评分。在ViDoRe V3基准上，该设计相比未池化配置将中位向量搜索延迟降低了15.1倍，而NDCG@10和Recall@10的绝对损失小于0.01；生产环境的中位向量搜索延迟为156毫秒。在并发负载下，池化索引比未池化索引维持约88倍更高的QPS。事件驱动的摄取路径估计……

    arXiv:2608.28572v1 Announce Type: new  Abstract: Institutional investors search visually dense pitch decks, board packs, and diligence materials that change hourly near deal closing. OCR followed by figure verbalisation is costly to refresh at this scale and can lose chart detail. We present PULSAR, a production vision-first retrieval system deployed at Mubadala Investment Company. PULSAR indexes page images with a frozen ColPali-style backbone and uses a pooled two-stage late-interaction index: compact page summaries support initial retrieval, followed by exact MaxSim rescoring over a finer pooled representation. On ViDoRe V3, this design reduces median vector-search latency by 15.1 times against an unpooled configuration with less than 0.01 absolute NDCG@10 and Recall@10 loss; production median vector-search latency is 156 ms. Under concurrent load, the pooled index sustains approximately 88 times higher QPS than an unpooled index. The event-driven ingestion path is estimated to be a
    
[^21]: RACORN：极端选择性下鲁棒的谓词无关过滤向量搜索

    RACORN: Robust Predicate-Agnostic Filtered Vector Search under Extreme Selectivity

    [https://arxiv.org/abs/2607.00768](https://arxiv.org/abs/2607.00768)

    RACORN-1 通过自适应搜索回退机制将过滤失败的节点作为“桥”引入候选队列，使谓词无关的过滤向量搜索在低于 1% 的极端选择性下仍能保持高召回率。

    

    过滤向量搜索的目标是在满足谓词条件的向量中返回最近邻。谓词往往具有很强的选择性：Big ANN 过滤赛道的多数标签查询仅选择 YFCC 10M 数据集中不到 1% 的数据。谓词无关方法在不依赖元数据构建的单个 HNSW 图上进行搜索，因此可接受任意谓词，但当选择性低于几个百分点时它们便会失效。ACORN-1 仅从满足条件的邻居中提取候选，其召回率在选择性低于 1% 时急剧崩溃。HNSW 虽能继续搜索，但在选择性 s 下大约需要 ef/s 次扩展周期。我们提出 RACORN-1，作为 ACORN-1 的扩展，它保留了原有的图结构和“过滤优先”的遍历方式，并引入了自适应搜索回退机制。当某节点的两跳扩展未达到目标时，失败节点会作为“桥”进入候选队列（受每次扩展预算的限制），但永远不会进入结果队列。RACORN-1 能够覆盖 ACORN-1 无法处理的选择性范围：在 1% 以下的选择性区间，ACORN-1 在任何 ef 设置下都无法达到 0.9 的召回率。

    arXiv:2607.00768v2 Announce Type: replace-cross  Abstract: Filtered vector search returns the nearest neighbors among the vectors that pass a predicate. The predicate is often selective: most tag queries of the Big ANN filtered track select less than 1% of YFCC 10M. Predicate-agnostic methods search one HNSW graph built without metadata and so take any predicate, but below a few percent selectivity they break down. ACORN-1 draws candidates only from passing neighbors and its recall collapses below 1%. HNSW keeps searching but spends about ef/s expansion cycles at selectivity s. We present RACORN-1, an extension of ACORN-1 that keeps its graph and filter-first traversal and adds Adaptive Search Fallback. When the two-hop expansion of a node falls short of a target, failing nodes enter the candidate queue as bridges, under a per-expansion budget, but never enter the result queue. RACORN-1 serves the selectivities ACORN-1 cannot: from 1% down ACORN-1 no longer reaches recall 0.9 at any ef
    
[^22]: 专利嵌入基准测试：对22个模型在检索、分类与聚类任务上的多任务评估

    Benchmarking Patent Embeddings: A Multi-Task Evaluation of 22 Models Across Retrieval, Classification, and Clustering

    [https://arxiv.org/abs/2605.24297](https://arxiv.org/abs/2605.24297)

    该研究对22个预训练嵌入模型（参数规模22M至12B）在专利检索、分类和聚类三大任务上进行了大规模基准测试，发现最优微调方案取决于下游任务类型——横截面配准方案最利于提升检索性能，而组合信号方案更适合分类与聚类任务。

    

    关于从业者在实践中使用专利嵌入，会产生两个问题：（i）一种微调方案是否足以应对所有下游应用？（ii）在一个专利领域上进行微调，是否足以支持其他领域的下游应用？通过在三项任务（信息检索、分类和聚类）上评估22个预训练嵌入模型（参数规模从22M到12B），测试数据包括113,148件辅助技术领域的WIPO专利（46,069条引用查询）以及外部的DAPFAM数据集，我们发现有两项结果对主流观点提出了质疑。（i）最优微调方案取决于下游任务：横截面配准（方案R3）对检索性能的提升最大（nDCG@10提升7.1%），而组合信号方案（方案R4）更适合分类任务（F1提升7.1）和聚类任务（V-measure提升10.9）；匹配数据对照实验证实了训练数据的差异（原文在此处截断）。

    arXiv:2605.24297v3 Announce Type: replace  Abstract: Two questions regarding practitioners' use of patent embeddings arise: (i) Does one fine-tuning recipe suffice for all downstream applications? (ii) Is fine-tuning on one patent landscape sufficient for downstream application on other landscapes? By evaluating 22 pre-trained embedding models (ranging from 22M to 12B parameters) on three tasks -- information retrieval, classification, and clustering -- on 113,148 WIPO patents for assistive technology (46,069 citation queries) and on an external DAPFAM dataset, we find that two results cast doubt on the prevailing wisdom. (i) The optimal fine-tuning recipe depends on the downstream task: cross-sectional alignment (recipe R3) provides the largest improvements to retrieval performance (+7.1% nDCG@10), whereas a combined signal recipe (recipe R4) is better suited to classification (+7.1 F1) and clustering (+10.9 V-measure); a matched data control confirms that differences in training data
    
[^23]: MCGI：面向十亿级规模磁盘驻留向量搜索的流形一致图索引

    MCGI: Manifold-Consistent Graph Indexing for Billion-Scale Disk-Resident Vector Search

    [https://arxiv.org/abs/2601.01930](https://arxiv.org/abs/2601.01930)

    提出MCGI——一种利用局部内在维度（LID）感知数据几何的磁盘驻留图索引方法，能够自适应分配搜索开销，以缓解高维ANN搜索中的欧几里得-测地线失配问题。

    

    基于图的近似最近邻（ANN）搜索在高维空间中常因“欧几里得-测地线失配”（即贪婪路由偏离底层数据流形）而出现性能下降。为应对这一挑战，本文提出了流形一致图索引（MCGI），这是一种几何感知、可驻留磁盘的索引方法，利用局部内在维度（LID）根据数据的内在几何结构来分配图索引的搜索开销。具体而言，MCGI 从查询自身搜索已计算出的距离来估计该查询周围的 LID，并仅对候选列表进行一次尺寸调整：位于高 LID 区域的查询获得更长的候选列表，而低 LID 区域的查询则更早终止；该估计本身不产生任何额外的磁盘读取。理论分析表明，任何固定搜索列表的成功概率随……（原文摘要在此处被截断）

    arXiv:2601.01930v4 Announce Type: replace  Abstract: Graph-based Approximate Nearest Neighbor (ANN) search often suffers from performance degradation in high-dimensional spaces due to the Euclidean-Geodesic mismatch, where greedy routing diverges from the underlying data manifold. To address this challenge, this paper presents Manifold-Consistent Graph Indexing (MCGI), a geometry-aware and disk-resident indexing method that leverages Local Intrinsic Dimensionality (LID) to allocate the search effort of a graph index according to the intrinsic geometry of data. Specifically, MCGI estimates the LID around each query from the distances that its own search has already computed and resizes the candidate list of the query once, so that queries in high-LID regions receive longer lists while queries in low-LID regions terminate earlier; the estimate itself costs no additional disk reads. Theoretical analysis shows that the success probability of any fixed search list decays exponentially with 
    
[^24]: 超越重叠：面向核外向量搜索的层次图抽象

    Beyond Overlap: Hierarchical Graph Abstraction for Out-of-Core Vector Search

    [https://arxiv.org/abs/2512.22838](https://arxiv.org/abs/2512.22838)

    GANNS是一种层次化的核外近似最近邻搜索引擎，通过将各簇存储为SSD上互不相交的分区并利用内存中的图抽象在SSD访问前提供逻辑重叠，从而同时提升搜索准确性与I/O效率。

    

    arXiv:2512.22838v3 公告类型：replace-cross 摘要：在十亿规模下，近似最近邻搜索（ANNS）往往成为一个核外问题：完整的向量集合和索引结构超出了内存容量，使得查询性能主要受限于SSD访问以及DRAM与SSD之间的数据传输。现有系统难以在准确性与效率之间取得平衡：物理重叠方法将向量或索引条目复制到多个分区中，扩大了驻留在SSD上的索引规模并产生额外的I/O开销；而基于量化的方法虽然减少了内存占用，但其近似距离的可靠性较低，通常需要从SSD进行代价高昂的原始向量重排序以维持召回率。我们提出了GANNS（Graph-Abstraction ANNS，图抽象ANNS），一个层次化的核外ANNS引擎，能够同时提升准确性和I/O效率。GANNS将每个簇存储为SSD上互不相交的分区，并配备尺度感知的索引，同时通过一个驻留在内存中的图抽象在发生SSD访问之前提供逻辑重叠。

    arXiv:2512.22838v3 Announce Type: replace-cross  Abstract: At billion scale, approximate nearest neighbor search (ANNS) often becomes an out-of-core problem: the full vector collection and index structures exceed memory capacity, making query performance dominated by SSD accesses and DRAM-SSD data movement. Existing systems struggle to strike a balance between accuracy and efficiency: physical-overlap methods replicate vectors or index entries across partitions, enlarging the SSD-resident index and incurring extra I/O; quantization-based methods reduce memory usage, but their approximate distances are less reliable and often require costly raw-vector reranking from SSD to preserve recall.   We present GANNS (Graph-Abstraction ANNS), a hierarchical out-of-core ANNS engine that improves accuracy and I/O efficiency simultaneously. GANNS stores each cluster as a disjoint SSD partition with scale-aware indexes, while a memory-resident graph abstraction provides logical overlap before SSD ac
    
[^25]: 基于知识图谱的检索增强生成模式匹配方法

    Knowledge Graph-based Retrieval-Augmented Generation for Schema Matching

    [https://arxiv.org/abs/2501.08686](https://arxiv.org/abs/2501.08686)

    提出KG-RAG4SM方法，通过集成多种新颖检索机制与重排序技术从外部知识图谱中提取相关子图，无需重新训练即在模式匹配任务上显著超越现有基于LLM的最先进方法。

    

    传统的基于相似度的模式匹配方法由于缺乏常识和领域特定知识，无法解决复杂的、特定领域映射中的语义歧义和冲突。与此同时，缺乏足够背景知识的大语言模型（LLM）容易产生幻觉，进一步增加了模式匹配的难度。为了克服这些局限性，我们提出了KG-RAG4SM，一种基于知识图谱的模式匹配检索增强生成方法。该方法集成了新颖的检索机制，包括基于LLM的、基于向量的、基于图遍历的以及混合技术，并结合先进的重排序方案，从大型外部知识图谱中提取最相关的子图。我们的方法无需重新训练即可取得优越的结果，在MIMIC数据集上，相比基于LLM的最先进（SOTA）方法（如Jellyfish-8B），精确率提升了39.79%，F1分数提升了30.56%。

    arXiv:2501.08686v2 Announce Type: replace-cross  Abstract: Traditional similarity-based schema matching methods fail to resolve semantic ambiguities and conflicts in complex, domain-specific mappings due to a lack of commonsense and domain-specific knowledge. In parallel, large language models (LLMs) without sufficient background knowledge tend to hallucinate, further complicating schema matching. To overcome these limitations, we introduce KG-RAG4SM, a Knowledge Graph-based Retrieval-Augmented Generation for Schema Matching. This method integrates novel retrieval mechanisms, including LLM-based, vector-based, graph traversal-based, and hybrid techniques with advanced reranking schemes to extract the most relevant subgraphs from large external KGs. Our approach achieves superior results without requiring re-training, as evidenced by improvements of 39.79% in precision and 30.56% in F1 score over LLM-based state-of-the-art (SOTA) methods (e.g., Jellyfish-8B) on the MIMIC dataset, and en
    

