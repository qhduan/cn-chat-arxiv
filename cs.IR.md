# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [MRVQ: One Resident Index for Dimension- and Rate-Elastic Vector Search](https://arxiv.org/abs/2610.03651) | 提出 MRVQ（套娃残差向量量化），用单一常驻索引即可通过截断残差阶段降低码率、截断嵌入坐标降低维度，以比多索引方案低 1.89-22 倍的内存覆盖所有维度-码率组合的向量检索。 |
| [^2] | [TSGuard: A Real-Time Framework for Detecting and Imputing Missing Data in Streaming Time Series](https://arxiv.org/abs/2610.03147) | TSGuard是一个实时框架，将轻量级图感知时间序列填补模型与约束感知验证、回退估计相结合，实现流式时间序列中缺失数据的检测、填补与验证，并将其融入完整的数据质量闭环。 |
| [^3] | [Benchmarking Literature Retrieval for a Model Organism: A Dictyostelium Case Study](https://arxiv.org/abs/2610.03130) | 本文基于dictyBase构建了首个针对模式生物盘基网柄菌的文献检索基准，并发现交叉编码器重排序与基因注释驱动的查询扩展能够在小众生物医学检索中带来有选择性的性能提升。 |
| [^4] | [Query-aware routing for Cross-lingual performance gains in Encoders](https://arxiv.org/abs/2610.02875) | 该论文提出将仅作用于查询端的LoRA适配器与基于查询和索引语言的确定性路由相结合，在保留同语言性能和现有文档索引的同时，使英语、芬兰语、瑞典语六条跨语言检索方向的平均nDCG@10从0.241提升至0.291，相对提升20.9%。 |
| [^5] | [Learning Query Encoders Can Be Hard Even When Vector Retrieval Is Geometrically Easy](https://arxiv.org/abs/2610.02749) | 该研究发现单向量查询编码器的实际检索质量往往远低于冻结文档索引在几何上所能支持的上限，并从理论上证明了学习查询编码器在计算上可能是困难的。 |
| [^6] | [Asterism: Exploring and Synthesizing Scattered Observations into Literature-Grounded Hypotheses and Theories](https://arxiv.org/abs/2610.02673) | Asterism系统通过层次化本体从数百篇论文中提取概念-关系三元组，让研究者能够策划证据图并在不同粒度上聚合观察结果，从而在保留研究者选择与直觉的前提下，将分散的观察结果综合为基于文献的假设与理论。 |
| [^7] | [When History Misleads: Asymmetric Margin Supervision for Instruction-Guided LLM Generative Recommendation](https://arxiv.org/abs/2610.02600) | 提出AIMS（非对称干预引导间隔监督）方法，将移除单个历史事件的干预效果转化为排序监督信号，以解决交互历史误导LLM生成式推荐的问题。 |
| [^8] | [Adaptive Sparsity Optimization with Learnable Soft Top-K and Per-Term Thresholding for Efficient Retrieval](https://arxiv.org/abs/2610.02572) | 该论文提出了一种结合可学习软Top-K、逐词阈值化和FLOPs正则化的自适应稀疏优化方案，显著缩短了查询和文档向量的长度，在保持检索相关性的同时大幅降低了检索延迟和存储成本。 |
| [^9] | [On-Premises Multi-Course RAG Tutoring for Business Education: Hardware-Software Trade-offs in a Campus AI Tutor](https://arxiv.org/abs/2610.02510) | 本文提出了CourseChat——一个面向本科商业教育的本地部署多课程RAG辅导系统，通过模型对比测试与软硬件权衡评估，证明12B和7B级本地大语言模型能在课程数据不出校门的前提下满足课堂实时响应的速度要求。 |
| [^10] | [SOLO: Certified-Recall Metric Similarity Search with Scan-Only Sampled Inverted Lists](https://arxiv.org/abs/2610.02387) | SOLO是一种在度量空间中不依赖任何排序启发式的近似最近邻搜索索引，通过将查询路由到随机样本最近点并完整扫描倒排列表，实现了可直接从索引计算和认证的召回率，并满足召回率≈f(b·k_s)的等工作量定律。 |
| [^11] | [Learning to Route in Visual Space via Multi-Step Embedding Retrieval](https://arxiv.org/abs/2609.38743) | 该论文提出VHOP基准框架和VHOP-Router端到端训练流程（结合监督微调、在线模仿学习与强化学习），将标准嵌入模型改造为能直接在嵌入空间中完成多步视觉导航的检索工具，从而突破LLM智能体视觉搜索中单步检索的性能瓶颈。 |
| [^12] | [Auditable Long-Term Memory: A Deterministic Retrieval Chain Measured at 479/475 of 500 on LongMemEval-S](https://arxiv.org/abs/2609.38021) | 该论文提出一种以确定性检索链（混合候选检索、交叉编码器重排序、覆盖优先数据包编译）为核心、LLM仅作为可替换最终阅读器的可审计长期记忆系统，在LongMemEval-S上两次评测分别获得479/500和475/500，成绩跨越Chronos High的478/500，但差异不足以证明优越性或等效性。 |
| [^13] | [A Manifold-Aware Topic Modeling Approach via Rank-Based Prototypes](https://arxiv.org/abs/2609.29630) | MARETopic是一个无需训练的主题建模框架，通过将嵌入投影到低维流形并把主题发现转化为基于排序的原型选择，贪心选出邻域可覆盖语料库的真实文档作为主题原型，其MARETopic_Corr变体在类别最多的两个基准上Purity和NMI领先于神经与聚类主题模型。 |
| [^14] | [What Are You Listening to? Temporal Music Grounding for Audio-to-Text Large Language Models](https://arxiv.org/abs/2608.29480) | 本文提出了时序音乐定位新任务及具有精确符号-音频对齐的MusicGroundingBench基准，用于评估音频-语言模型能否将音乐查询定位到具体时间段，发现现有模型在此任务上仍面临挑战，而任务特定训练可带来显著提升。 |
| [^15] | [Min-Cost Flow Routing for Evidence Assembly in Long Multimodal Documents](https://arxiv.org/abs/2606.07235) | 该论文提出FlowReader，将长多模态文档问答中的证据选择建模为带容量限制的最小成本流问题，通过谱分解按比例分配证据预算并生成短证据链，无需语言模型规划即可实现全面覆盖，在VisDoMBench上取得最高宏观准确率68.9。 |
| [^16] | [More Efficient LLM Reranking with Whole-Pool, Setwise, Long-Context Language Models](https://arxiv.org/abs/2606.01782) | 提出全池集合式重排序方法DualEnd，通过从两端同时填充排序，仅需50次LLM比较即可完成100个候选的完整排序，比现有方法减少59.4%-88.8%的比较次数。 |
| [^17] | [Infinity Search: Approximate Vector Search with Projections on q-Metric Spaces](https://arxiv.org/abs/2506.06557) | 该论文提出将任意相异度函数投影到超度量空间并学习该投影的近似形式，在保持最近邻关系的同时实现最坏情况复杂度仅为树深度的近似向量搜索，并将该方法推广到更一般的q-度量空间。 |

# 详细

[^1]: MRVQ：面向维度与码率弹性向量检索的单一常驻索引

    MRVQ: One Resident Index for Dimension- and Rate-Elastic Vector Search

    [https://arxiv.org/abs/2610.03651](https://arxiv.org/abs/2610.03651)

    提出 MRVQ（套娃残差向量量化），用单一常驻索引即可通过截断残差阶段降低码率、截断嵌入坐标降低维度，以比多索引方案低 1.89-22 倍的内存覆盖所有维度-码率组合的向量检索。

    

    密集检索服务需要随着延迟、质量和内存预算的变化，在嵌入前缀维度和索引比特率之间进行切换。为每种码率单独调优一个量化器可以获得最佳质量，但此时检索层必须同时持有多个码流和量化器状态。我们提出了套娃残差向量量化，这是一种针对冻结嵌入的事后残差量化器。其最高码率的编码可以通过两种方式进行截断：丢弃残差阶段以降低码率，丢弃嵌入坐标以降低维度。因此，一个常驻的索引产物即可服务我们评估的每一个（维度， 码率）组合。在 FiQA 和 NFCorpus 数据集、四种嵌入家族以及 {4, 8, 16} 字节编码的设置下，MRVQ 是我们评估的所有设计中内存占用最低的。它比三个单独训练的 QINCo2 索引少占用 17.8-22.0 倍内存，比精简的共享模型最强基线少占用 1.89-2.02 倍。这种节省并非没有代价：按码率单独训练的 QINCo2 在 nDCG@10 上高出 0.026-0.107

    arXiv:2610.03651v1 Announce Type: new  Abstract: Dense-retrieval services must switch among embedding-prefix dimensions and index bit rates as latency, quality, and memory budgets change. Tuning a quantizer separately for each rate gives the best quality, but the retrieval tier then holds several code streams and quantizer states at once. We introduce Matryoshka Residual Vector Quantization (MRVQ), a post-hoc residual quantizer for frozen embeddings. Its maximum-rate code can be truncated two ways: dropping residual stages lowers the rate, and dropping embedding coordinates lowers the dimension. One resident artifact therefore serves every (dimension, rate) pair we evaluate. Across FiQA and NFCorpus, four embedding families, and {4, 8, 16}-byte codes, MRVQ is the lowest-RAM design we evaluate. It uses 17.8-22.0x less memory than three separately trained QINCo2 indices, and 1.89-2.02x less than a lean shared-model steelman. The saving is not free: per-rate QINCo2 is 0.026-0.107 nDCG@10 
    
[^2]: TSGuard：一个用于流式时间序列中缺失数据检测与填补的实时框架

    TSGuard: A Real-Time Framework for Detecting and Imputing Missing Data in Streaming Time Series

    [https://arxiv.org/abs/2610.03147](https://arxiv.org/abs/2610.03147)

    TSGuard是一个实时框架，将轻量级图感知时间序列填补模型与约束感知验证、回退估计相结合，实现流式时间序列中缺失数据的检测、填补与验证，并将其融入完整的数据质量闭环。

    

    流式传感器应用经常因故障、通信丢失或环境干扰而遭受观测数据延迟或缺失的困扰。尽管近期的填补方法能够有效利用时间和空间依赖性，但大多数方法要么假设可以离线访问未来的观测数据，要么只优先考虑吞吐量而不保证领域合理性。我们提出了TSGuard，这是一个用于监控、验证和填补流式时间序列中缺失值的实时演示系统。TSGuard将轻量级的图感知时间序列填补模型与约束感知验证、回退估计以及面向操作员的解释相结合。TSGuard并不将填补视为孤立的预测任务，而是将其集成到更广泛的数据质量闭环中：检测有问题的观测值、填补缺失值、根据物理和空间约束验证估计值，并将原始值保留为合理的……

    arXiv:2610.03147v1 Announce Type: cross  Abstract: Streaming sensor applications routinely suffer from delayed or missing observations caused by faults, communication losses, or environmental interference. Although recent imputation methods exploit temporal and spatial dependencies effectively, most either assume offline access to future observations or prioritize throughput without enforcing domain plausibility. We present TSGuard, a real-time demonstration system for monitoring, validating, and imputing missing values in streaming time series. TSGuard combines a lightweight graph-aware temporal imputation model with constraint-aware validation, fallback estimation, and operator-facing explanations.   Rather than treating imputation as an isolated prediction task, TSGuard integrates it into a broader data-quality loop: detect problematic observations, impute missing values, validate estimated against physical and spatial constraints, and either retain the original value as a plausible
    
[^3]: 针对模式生物的文献检索基准测试：以盘基网柄菌为案例研究

    Benchmarking Literature Retrieval for a Model Organism: A Dictyostelium Case Study

    [https://arxiv.org/abs/2610.03130](https://arxiv.org/abs/2610.03130)

    本文基于dictyBase构建了首个针对模式生物盘基网柄菌的文献检索基准，并发现交叉编码器重排序与基因注释驱动的查询扩展能够在小众生物医学检索中带来有选择性的性能提升。

    

    生物文献检索系统通常是基于广泛的生物医学语料库和通用搜索任务进行开发和评估的。然而，许多经过人工整理的知识库是在更狭窄的模式生物领域内运作的，在这些领域中文献稀疏且术语具有物种特异性。我们为盘基网柄菌构建了一个来自dictyBase的检索基准，盘基网柄菌是细胞生物学和发育生物学中的一种模式生物。该基准由数据管理员生成的生物学查询组成，这些查询与PubMed收录的文章相关联，并附带结构化的基因注释。利用该基准，我们研究了小众生物检索中的三个因素：交叉编码器重排序、基因感知的查询扩展，以及仅摘要检索与全文检索的对比。我们报告称，重排序和基因感知的查询扩展可以有选择性地提升检索效果：当模型本身非常适合进行生物证据匹配时，重排序最为有用，而经过整理的注释则有助于澄清紧凑的查询。

    arXiv:2610.03130v1 Announce Type: cross  Abstract: Biological literature retrieval systems are often developed and evaluated using broad biomedical corpora and general-purpose search tasks. However, many curated knowledge bases operate in narrower model-organism domains, where the literature is sparse and terminology is organism-specific. We introduce a retrieval benchmark from dictyBase for Dictyostelium, a model organism in cell and developmental biology. The benchmark consists of curator-generated biological queries linked to PubMed-indexed articles, together with structured gene annotations. Using this benchmark, we study three factors in niche biological retrieval: cross-encoder reranking, gene-aware query expansion, and abstract-only versus full-text retrieval. We report that reranking and gene-aware query expansion improve retrieval selectively: reranking is most useful when the model is well suited to biological evidence matching, whereas curated annotations help clarify compac
    
[^4]: 面向编码器跨语言性能提升的查询感知路由

    Query-aware routing for Cross-lingual performance gains in Encoders

    [https://arxiv.org/abs/2610.02875](https://arxiv.org/abs/2610.02875)

    该论文提出将仅作用于查询端的LoRA适配器与基于查询和索引语言的确定性路由相结合，在保留同语言性能和现有文档索引的同时，使英语、芬兰语、瑞典语六条跨语言检索方向的平均nDCG@10从0.241提升至0.291，相对提升20.9%。

    

    多语言编码器在查询与相关文档语言不同时，检索效果可能会下降，尽管其在同语言场景下表现强劲。我们研究了如何在保留编码器现有同语言性能和文档索引的前提下，提升芬兰语与瑞典语的跨语言检索效果。我们将仅作用于查询端、针对冻结文档嵌入训练的低秩适配器（LoRA），与基于查询语言和索引语言的确定性路由相结合：跨语言查询使用适配器，同语言查询则使用原始编码器。SampoTron（我们微调的低秩适配器与Nemotron-3-Embed-1B模型的组合）在一个采样的金融基准上，将六个英语、芬兰语和瑞典语方向的平均检索质量从nDCG@10的0.241提升至0.291，相对提升20.9%。所有六个跨语言方向均得到改善，并且路由保留了（原有同语言性能）。

    arXiv:2610.02875v1 Announce Type: cross  Abstract: Multilingual encoders can exhibit reduced retrieval effectiveness when queries and relevant documents differ in language, despite strong same-language performance. We investigate whether Finnish and Swedish cross-lingual retrieval can improve while preserving an encoder's existing same-language performance and document index. We combine a query-only low-rank adapter, trained against frozen document embeddings, with deterministic routing based on query and index languages. Cross-language queries use the adapter, while same-language queries use the original encoder. SampoTron, our fine-tuned low-rank (LoRA) adapter alongwith the Nemotron-3-Embed-1B model, improves average retrieval quality across six English, Finnish, and Swedish directions from 0.241 to 0.291 in normalized discounted cumulative gain (nDCG) at rank ten, a 20.9% relative gain on a sampled financial benchmark. All six cross-lingual directions improve, and routing preserves
    
[^5]: 即使向量检索在几何上很容易，学习查询编码器仍然可能很困难

    Learning Query Encoders Can Be Hard Even When Vector Retrieval Is Geometrically Easy

    [https://arxiv.org/abs/2610.02749](https://arxiv.org/abs/2610.02749)

    该研究发现单向量查询编码器的实际检索质量往往远低于冻结文档索引在几何上所能支持的上限，并从理论上证明了学习查询编码器在计算上可能是困难的。

    

    高效的向量检索需要同时满足两个条件：一是语料库的几何结构能够支持通过向量相似度检索到正确的文档，二是查询编码器能够将查询嵌入到嵌入空间中距离其目标文档较近的位置。近期的研究通过实现n个文档所有top-k答案集所需的最小嵌入维度这一视角来考察几何容量。我们研究了另一种几何容量的概念——冻结的文档索引所能达到的最大召回率——并探讨学习型查询编码器能否达到这一上限。在多个真实世界的检索基准上，我们发现单向量查询编码器的检索质量往往远低于文档索引所能支持的水平。受这一观察启发，我们给出了学习查询编码器在计算上可能非常困难的理论证据。特别地，我们构造了一个检索任务，该任务(1)存在一个能够实现完美召回率的查询编码器……

    arXiv:2610.02749v1 Announce Type: cross  Abstract: Efficient vector retrieval requires both a corpus geometry that supports retrieving the right documents through vector similarity, and a query encoder that can embed queries near their desired documents in the embedding space. Recent work has studied geometric capacity through the lens of the minimum embedding dimension needed to realize all top-$k$ answer sets of $n$ documents. We study a different notion of geometric capacity--the maximum recall achievable for a frozen document index--and explore whether learned query encoders can reach this ceiling. On several real-world retrieval benchmarks, we show that retrieval quality of single-vector query encoders often lies far below what the document indices can support.   Motivated by this observation, we give theoretical evidence that learning query encoders can be computationally hard. In particular, we construct a retrieval task that (1) admits a query encoder with perfect recall which 
    
[^6]: Asterism：探索并将分散的观察结果综合为基于文献的假设与理论

    Asterism: Exploring and Synthesizing Scattered Observations into Literature-Grounded Hypotheses and Theories

    [https://arxiv.org/abs/2610.02673](https://arxiv.org/abs/2610.02673)

    Asterism系统通过层次化本体从数百篇论文中提取概念-关系三元组，让研究者能够策划证据图并在不同粒度上聚合观察结果，从而在保留研究者选择与直觉的前提下，将分散的观察结果综合为基于文献的假设与理论。

    

    理论将许多独立的观察结果纳入一个包含新颖假设的统一框架。构建这样一个理论的研究者必须综合散布在众多论文中的观察结果，这些论文描述着相关概念，但往往使用不同的术语。而哪些概念最为重要，还取决于研究者自身的偏好和研究问题。近期的方法利用大语言模型（LLM）来规模化理论综合，但却将研究者的选择和直觉自动化抹除了。我们提出了Asterism，该系统从数百篇论文中提取观察结果，形成概念-关系三元组，并将概念统一在一个层次化本体中。研究者可以借助该本体来策划证据图，并在不同粒度级别上聚合观察结果，从而将理论构建聚焦于特定感兴趣的现象。在实际部署中（n=10），研究者从观察结果逐步推进到理论，并保留了符合自身偏好的概念和假设。在两个案例研究中，免疫学团队……（摘要原文截断）

    arXiv:2610.02673v1 Announce Type: cross  Abstract: A theory draws many independent observations into one framework with novel hypotheses. A researcher building such a theory must synthesize observations scattered across many papers, each describing related concepts but often in different terms. Which concepts matter most also depends on their preferences and research questions. Recent approaches scale theory synthesis with LLMs, but automate away choices and intuitions from researchers. We present Asterism, which extracts observations from hundreds of papers as concept-relation triples, with concepts unified in a hierarchical ontology. Researchers curate an evidence graph using the ontology and aggregate observations at different levels of granularity to focus theory formation on specific phenomena of interest. In a field deployment (n=10), researchers worked from observations to theories, and kept concepts and hypotheses fitting their preferences. In two case studies, teams of immunol
    
[^7]: 当历史误导时：面向指令引导LLM生成式推荐的非对称间隔监督

    When History Misleads: Asymmetric Margin Supervision for Instruction-Guided LLM Generative Recommendation

    [https://arxiv.org/abs/2610.02600](https://arxiv.org/abs/2610.02600)

    提出AIMS（非对称干预引导间隔监督）方法，将移除单个历史事件的干预效果转化为排序监督信号，以解决交互历史误导LLM生成式推荐的问题。

    

    在指令引导的生成式推荐中，基于大语言模型（LLM）的推荐系统需要平衡两个目标：响应用户的当前请求，以及与其交互历史中的偏好保持一致。当两者发生冲突时，历史事件可能会覆盖用户当前的请求。我们指出，将单个历史事件的影响转化为监督信号面临两个障碍。首先，对推荐结果影响最大的事件未必是支持目标物品的事件。其次，移除一个误导性事件虽然会提高目标物品的得分，但可能使竞争物品的得分提升更多，因此仅凭目标物品得分的提高并不能保证排序的改善。我们提出非对称干预引导间隔监督方法（AIMS），将移除单个历史事件的干预效果转化为排序监督。对于已被正确排序的训练请求，冻结的参考模型会识别出能够同时改善目标……（原文摘要在此处截断）

    arXiv:2610.02600v1 Announce Type: new  Abstract: In instruction-guided generative recommendation, LLM-based recommenders need to balance two goals: responding to the user's current request and aligning with the preferences in their interaction history. When the two conflict, history events can override the request. We show that turning the effect of individual history events into supervision faces two obstacles. First, the events that most influence a recommendation are not necessarily the ones that support the target item. Second, removing a misleading event can raise the target's score but a competing item's score even more, so a higher target score alone does not guarantee a better ranking. We propose Asymmetric Intervention-Guided Margin Supervision (AIMS), which converts the effect of removing individual history events into ranking supervision. For training requests already ranked correctly, a frozen reference model identifies request-specific deletions that improve both the targe
    
[^8]: 基于可学习软Top-K与逐词阈值化的自适应稀疏优化以实现高效检索

    Adaptive Sparsity Optimization with Learnable Soft Top-K and Per-Term Thresholding for Efficient Retrieval

    [https://arxiv.org/abs/2610.02572](https://arxiv.org/abs/2610.02572)

    该论文提出了一种结合可学习软Top-K、逐词阈值化和FLOPs正则化的自适应稀疏优化方案，显著缩短了查询和文档向量的长度，在保持检索相关性的同时大幅降低了检索延迟和存储成本。

    

    最近关于神经稀疏检索的研究通过利用大语言模型（LLM）进行语义词项扩展，展现出了强大的相关性表现。然而，学习得到的模型与以往的稀疏化技术结合使用时，仍然会产生过长的文档向量和查询向量，部分原因是LLM的词表规模庞大，这给检索的时间和空间效率带来了严峻挑战。本文提出了一种通过多种自适应策略协同作用来优化模型稀疏性的方案，包括可学习的软Top-K、逐词项阈值化以及FLOPs正则化，以提高查询向量和文档向量的稀疏度。在MS MARCO和BEIR数据集上使用Lion-SP模型的实验结果表明，所提出的方案能够通过显著降低查询和文档的平均长度而超越基线方法。我们的方案能够在保持高度竞争力的相关性的同时，实现更短的检索延迟和更低的存储成本。

    arXiv:2610.02572v1 Announce Type: new  Abstract: Recent work on neural sparse retrieval has demonstrated strong relevance by leveraging Large Language Models (LLMs) for semantic term expansion. However, learned models paired with previous sparsification techniques still yield overly long document and query vectors partly due to a large LLM vocabulary, imposing a serious challenge to retrieval time and space efficiency. This paper proposes a scheme for optimizing model sparsity through a synergy of adaptive strategies, including learnable soft top-K, per-term thresholding, and FLOPs regularization to increase the sparsity of query and document vectors. Experimental results with Lion-SP model on the MS MARCO and BEIR datasets demonstrate that the proposed scheme can outperform the baselines by significantly reducing the average query and document lengths. Our scheme can achieve much shorter retrieval latency and lower storage cost while maintaining highly competitive relevance.
    
[^9]: 面向商业教育的本地部署多课程RAG辅导系统：校园AI辅导员的软硬件权衡

    On-Premises Multi-Course RAG Tutoring for Business Education: Hardware-Software Trade-offs in a Campus AI Tutor

    [https://arxiv.org/abs/2610.02510](https://arxiv.org/abs/2610.02510)

    本文提出了CourseChat——一个面向本科商业教育的本地部署多课程RAG辅导系统，通过模型对比测试与软硬件权衡评估，证明12B和7B级本地大语言模型能在课程数据不出校门的前提下满足课堂实时响应的速度要求。

    

    基于检索增强生成（RAG）的校园AI辅导员必须使答案立足于指定的课程材料，同时将教科书和学生对话数据保留在机构自身的基础设施内。我们提出了CourseChat，这是一个面向本科商业教育的本地部署、多课程RAG辅导员系统，部署在校园Web网关之后，旨在嵌入Moodle中使用。六个相互隔离的课程班级（每个班级由其各自的课程参考号CRN标识）共享双边缘AI主机，其上运行着FastAPI服务、本地向量数据库以及由Ollama提供服务的本地大语言模型（LLM）。我们报告了两轮生成模型对比测试、一次独立的固定证据来源保真度比较实验，以及对话与测验审计结果。多个较大的模型未能通过课堂场景的速度门槛，但一个12B模型和一个7B替代方案通过了测试。另一个独立的混合专家模型候选方案在某些修正上有所改进，但同时引入了新的事实性错误和连贯性错误。因此我们……

    arXiv:2610.02510v1 Announce Type: new  Abstract: Campus AI tutors based on retrieval-augmented generation (RAG) must ground answers in assigned course materials while keeping textbooks and student dialogue on institutional infrastructure. We present CourseChat, an on-premises, multi-course RAG tutor for undergraduate business education, deployed behind a campus web gateway and intended for use embedded in Moodle. Six isolated course offerings, each keyed by its own course reference number (CRN), share twin-edge AI hosts running a FastAPI service, a local vector database, and a local large language model (LLM) served by Ollama. We report two generation-model bake-off rounds, a separate fixed-evidence source-fidelity comparison, and conversation and quiz audits. Several larger models failed the classroom speed gate, but a 12B model and a 7B alternative passed. A separate mixture-of-experts candidate improved some corrections while introducing new factual and continuity errors. We therefo
    
[^10]: SOLO：基于仅扫描采样倒排列表的认证召回率度量相似性搜索

    SOLO: Certified-Recall Metric Similarity Search with Scan-Only Sampled Inverted Lists

    [https://arxiv.org/abs/2610.02387](https://arxiv.org/abs/2610.02387)

    SOLO是一种在度量空间中不依赖任何排序启发式的近似最近邻搜索索引，通过将查询路由到随机样本最近点并完整扫描倒排列表，实现了可直接从索引计算和认证的召回率，并满足召回率≈f(b·k_s)的等工作量定律。

    

    我们提出了SOLO，这是一个面向一般度量空间近似最近邻搜索的索引，其服务路径不包含任何形式的排序启发式方法：查询被路由到数据库随机样本的 $k_s$ 个最近点，被触及的倒排列表中的每个对象都用真实距离进行评估。由于不需要任何对象在排序中超越其他对象，召回率等于一个可以从存储索引直接计算出的覆盖概率：只需对查询样本进行一次真值遍历，即可一次性认证所有操作点，而无需实际服务其中任何一个——这是一种召回率认证，而对于可导航图结构，无论付出多大代价都不存在类似的对象。整个索引就是一条递归规则——对集合进行采样，将每个对象发布到其 $b$ 个最近的样本点，拆分任何超出界限的列表，始终扫描叶子节点——并且其操作面遵循一个等工作量定律：召回率 ≈ f(b·k_s)，其水平是数据集的单标量特征。

    arXiv:2610.02387v1 Announce Type: cross  Abstract: We present SOLO, an index for approximate nearest-neighbor search in general metric spaces whose serving path contains no ranking heuristic of any kind: a query is routed to the $k_s$ nearest points of a random sample of the database, and every object in the touched posting lists is evaluated with the true distance. Because nothing must outrank anything, recall equals a coverage probability computable from the stored index: one ground-truth pass over a query sample certifies every operating point at once, without serving any of them -- a recall certificate, and for a navigable graph no analogous object exists at any price. The whole index is one recursive rule -- sample the collection, post each object to its $b$ nearest sample points, split any list that outgrows a bound, always scan the leaves -- and its operating surface obeys an equal-work law, recall $\approx f(b \cdot k_s)$, whose level is a one-scalar signature of the dataset. T
    
[^11]: 通过多步嵌入检索学习在视觉空间中进行路由

    Learning to Route in Visual Space via Multi-Step Embedding Retrieval

    [https://arxiv.org/abs/2609.38743](https://arxiv.org/abs/2609.38743)

    该论文提出VHOP基准框架和VHOP-Router端到端训练流程（结合监督微调、在线模仿学习与强化学习），将标准嵌入模型改造为能直接在嵌入空间中完成多步视觉导航的检索工具，从而突破LLM智能体视觉搜索中单步检索的性能瓶颈。

    

    LLM智能体依赖检索工具来访问外部知识，然而视觉智能体搜索仍然严重受限于标准的单步检索器。在现有流程中，智能体必须为每个中间步骤发出文本查询，当视觉线索难以用文字描述、或检索器无法在其前列结果中呈现必要的中间证据时，搜索就会陷入困境。我们假设，将跨越整个嵌入空间的多步导航直接交由检索工具完成，可以解决这一性能瓶颈。为了系统地研究这一问题，我们提出了VHOP——一个灵活的数据生成框架与基准，包含五个核心难度级别，同时测试视觉匹配与搜索规划能力。利用该框架，我们开发了VHOP-Router——一个端到端的训练流程，结合监督微调、在线模仿学习与强化学习，将标准嵌入模型转变为……

    arXiv:2609.38743v1 Announce Type: new  Abstract: LLM agents rely on retrieval tools to access external knowledge, yet visual agentic search remains severely bottlenecked by standard single-step retrievers. In current pipelines, the agent must issue text queries for every intermediate step, struggling when visual clues are difficult to describe or when the retriever fails to surface necessary intermediate evidence within its top results. We hypothesize that offloading multi-step navigation across the entire embedding space directly to the retrieval tool resolves this performance bottleneck. To study this systematically, we introduce VHOP, a flexible data generation framework and benchmark with five core difficulty levels testing both visual matching and search planning. Using this framework, we develop VHOP-Router, an end-to-end training pipeline---combining supervised fine-tuning, online imitation learning, and reinforcement learning---that transforms a standard embedding model into an
    
[^12]: 可审计的长期记忆：在LongMemEval-S上测得479/475（满分500）成绩的确定性检索链

    Auditable Long-Term Memory: A Deterministic Retrieval Chain Measured at 479/475 of 500 on LongMemEval-S

    [https://arxiv.org/abs/2609.38021](https://arxiv.org/abs/2609.38021)

    该论文提出一种以确定性检索链（混合候选检索、交叉编码器重排序、覆盖优先数据包编译）为核心、LLM仅作为可替换最终阅读器的可审计长期记忆系统，在LongMemEval-S上两次评测分别获得479/500和475/500，成绩跨越Chronos High的478/500，但差异不足以证明优越性或等效性。

    

    我们在LongMemEval-S上评估了一个可审计的长期记忆系统。其检索链采用混合候选检索、交叉编码器重排序、以覆盖优先的数据包编译以及确定性推理脚手架；大语言模型仅作为可替换的最终阅读器使用。该检索链在470个可回答问题中的468个上将所有金标准会话纳入候选池，并为其中462个生成金标准完整的数据包。使用通过未固定版本的CLI别名调用的Claude Opus阅读器，在GPT-4o评分下，两次500题评测分别获得479/500和475/500的分数。其中72个可回答的知识更新题目使用了经过实质性修改的评分提示词，该修改在官方提示词文本下的效果尚未被测量。这一对结果跨越了Chronos High已发表的478/500；由于阅读器生成方式、评分提示词、可能的数据版本差异，以及系统内部的方差，这些结果既不能确立优越性，也不能确立等效性。在同一数据包上使用grok-4.6-high阅读器的得分为476/474，而……

    arXiv:2609.38021v1 Announce Type: cross  Abstract: We evaluate an auditable long-term memory system on LongMemEval-S. Its retrieval chain uses hybrid candidate retrieval, cross-encoder reranking, coverage-first packet compilation, and deterministic reasoning scaffolds; an LLM is used only as a replaceable final reader. The chain places all gold sessions in the candidate pool for 468/470 answerable questions and produces gold-complete packets for 462/470. With a Claude Opus reader called through an unpinned CLI alias, two 500-question passes score 479/500 and 475/500 under GPT-4o. The 72 answerable knowledge-update rows used a substantively modified scoring prompt whose effect under the official text has not been measured. The pair straddles Chronos High's published 478/500; differences in reader generation, scoring prompt, and possibly data version, plus within-system variance, establish neither superiority nor equivalence. A grok-4.6-high reader on the same packets scores 476/474, whi
    
[^13]: 一种基于排序原型的流形感知主题建模方法

    A Manifold-Aware Topic Modeling Approach via Rank-Based Prototypes

    [https://arxiv.org/abs/2609.29630](https://arxiv.org/abs/2609.29630)

    MARETopic是一个无需训练的主题建模框架，通过将嵌入投影到低维流形并把主题发现转化为基于排序的原型选择，贪心选出邻域可覆盖语料库的真实文档作为主题原型，其MARETopic_Corr变体在类别最多的两个基准上Purity和NMI领先于神经与聚类主题模型。

    

    近期的主题模型利用预训练嵌入，但神经架构产生的潜在表示缺乏与具体文本的关联，而基于聚类的流水线只能在事后分配代表性文档，依赖于在 高维空间中因枢纽性和各向异性而失真的绝对距离。我们提出了MARETopic，这是一个无需训练的框架，将主题发现转化为基于排序的原型选择。在将嵌入投影到低维流形后，MARETopic构建编码序数邻域结构的排序列表。贪心算法精确选出K个范例文档（即真实的语料库文本），其邻域可覆盖整个语料库。两个变体共享这一准则。其中MARETopic_Corr利用查询性能预测器和秩相关性度量对候选进行评分，在类别最多的两个基准数据集上取得了Purity和NMI的最优结果，领先于神经主题模型和基于聚类的主题模型。

    arXiv:2609.29630v1 Announce Type: cross  Abstract: Recent topic models leverage pretrained embeddings, but neural architectures produce latent representations without grounding in specific texts, and clustering-based pipelines assign representative documents only post hoc, relying on absolute distances distorted by hubness and anisotropy in high-dimensional spaces. We introduce MARETopic, a training-free framework that casts topic discovery as rank-based prototype selection. After projecting embeddings onto a low-dimensional manifold, MARETopic builds ranked lists encoding ordinal neighborhood structure. A greedy algorithm selects exactly K exemplar documents, real corpus texts, whose neighborhoods cover the corpus. Two variants share this criterion. MARETopic$_\text{Corr}$ scores candidates with a query performance predictor and a rank correlation measure, leading Purity and NMI on the two benchmarks with the most categories, ahead of both neural and clustering-based topic models. MAR
    
[^14]: 你在听什么？面向音频到文本大语言模型的时序音乐定位

    What Are You Listening to? Temporal Music Grounding for Audio-to-Text Large Language Models

    [https://arxiv.org/abs/2608.29480](https://arxiv.org/abs/2608.29480)

    本文提出了时序音乐定位新任务及具有精确符号-音频对齐的MusicGroundingBench基准，用于评估音频-语言模型能否将音乐查询定位到具体时间段，发现现有模型在此任务上仍面临挑战，而任务特定训练可带来显著提升。

    

    大型音频-语言模型能够生成流畅且在音乐上看似合理的回答，然而这些回答是否真正基于音频输入往往仍不清楚。我们提出了时序音乐定位这一任务，在该任务中，模型需要返回与所查询的音符、音乐事件或音乐模式相对应的一个或多个时间段。为了评估这一能力，我们构建了MusicGroundingBench，这是一个受控基准套件，通过将算法生成的钢琴MIDI渲染为音频，从而获得精确的符号到音频的对齐。该套件包含两个子集：MGBench-3N，用于评估在最多包含三个音符的片段中的音符级定位能力；MGBench-2B，用于评估两小节片段中的结构化定位能力和短形式音乐理解能力。实验表明，时序音乐定位对当前的音频-语言模型而言仍然具有挑战性，而针对特定任务的训练则带来了显著的性能提升。我们还进一步报告了探索性证据。

    arXiv:2608.29480v1 Announce Type: cross  Abstract: Large audio-language models can produce fluent and musically plausible responses, yet it often remains unclear whether those responses are grounded in the audio input. We introduce temporal music grounding, a task in which a model returns one or more time spans corresponding to a queried musical note, event, or pattern. To evaluate this capability, we present MusicGroundingBench, a controlled benchmark suite built by rendering algorithmically generated piano MIDI to audio, yielding exact symbolic-to-audio alignment. The suite comprises two subsets: MGBench-3N, which evaluates note-level grounding in clips containing up to three notes, and MGBench-2B, which evaluates structured grounding and short-form music understanding in two-bar excerpts. Experiments show that temporal music grounding remains challenging for current audio-language models, whereas task-specific training yields substantial gains. We further report exploratory evidence
    
[^15]: 面向长多模态文档证据组装的最小成本流路由

    Min-Cost Flow Routing for Evidence Assembly in Long Multimodal Documents

    [https://arxiv.org/abs/2606.07235](https://arxiv.org/abs/2606.07235)

    该论文提出FlowReader，将长多模态文档问答中的证据选择建模为带容量限制的最小成本流问题，通过谱分解按比例分配证据预算并生成短证据链，无需语言模型规划即可实现全面覆盖，在VisDoMBench上取得最高宏观准确率68.9。

    

    回答关于长多模态文档的问题，需要在文本、表格、图表和幻灯片中的相关内容方面之间分配固定的证据预算，同时避免近乎重复的内容。我们提出了FlowReader，它将证据选择表述为多模态内容图上带容量限制的最小成本流问题。谱分解识别与查询相关内容的潜在方面，并根据其谱能量按比例在这些方面之间分配预算。这些容量限制在路由过程中强制实现方面覆盖，无需调用语言模型进行规划。查询条件化的成本优先选择相关且相互一致的证据链。对最优流进行分解可产生较短的证据链，由视觉-语言模型并行读取，再由推理模块进行协调。在VisDoMBench上使用Qwen3-VL-32B时，FlowReader取得了最高的宏观准确率（68.9），超越了最强的（摘要在此处截断）。

    arXiv:2606.07235v3 Announce Type: replace-cross  Abstract: Answering questions about long multimodal documents requires distributing a fixed evidence budget across relevant facets in text, tables, figures, and slides while avoiding near-duplicates. We present \flowreader, which formulates evidence selection as a single minimum-cost flow problem with capacity limits over a multimodal content graph. Spectral decomposition identifies latent aspects of query-relevant content and allocates the budget among them in proportion to their spectral energy. These capacity limits enforce aspect coverage during routing without requiring a language-model planning call. Query-conditioned costs prioritize chains of relevant, mutually consistent evidence. Decomposing the optimal flow produces short evidence chains, which a vision-language model reads in parallel and a reasoner reconciles. On VisDoMBench with Qwen3-VL-32B, \flowreader\ achieves the highest macro accuracy ($68.9$), surpassing the stronges
    
[^16]: 基于全池、集合式、长上下文语言模型的更高效LLM重排序

    More Efficient LLM Reranking with Whole-Pool, Setwise, Long-Context Language Models

    [https://arxiv.org/abs/2606.01782](https://arxiv.org/abs/2606.01782)

    提出全池集合式重排序方法DualEnd，通过从两端同时填充排序，仅需50次LLM比较即可完成100个候选的完整排序，比现有方法减少59.4%-88.8%的比较次数。

    

    基于LLM的重排序器通常通过重复的局部比较（列表式、成对式或逐点式）来产生排序，需要多次顺序模型调用。我们研究了当整个检索到的候选池能够放入上下文窗口时，长上下文LLM如何大幅减少这种计算量。我们引入了全池集合式重排序，其中每次比较都对整个候选池进行排序，并提出DualEnd方法，该方法联合选择预测为最相关和最不相关的候选。通过从两端填充排序，DualEnd仅需50次LLM比较即可构建100个候选的完整排序。在TREC DL19和DL20数据集上使用九个开源权重LLM的实验表明，与之前面向顶部的使用堆排序的窗口式集合排序相比，该方法减少了59.4%的比较次数；与使用冒泡排序的面向顶部的窗口式集合排序相比，减少了88.8%的比较次数——尽管这些基线方法仅针对前10名排序，而DualEnd针对的是完整排序。

    arXiv:2606.01782v2 Announce Type: replace  Abstract: LLM-based re-rankers produce a rankings through repeated local comparisons (listwise, pairwise or pointwise), requiring many sequential model calls. We study how long-context LLMs can drastically reduce this computation when the entire retrieved candidate pool fits within the context window. We introduce Whole-Pool Setwise re-ranking, where each comparison ranks all the entire candidate pool, and propose DualEnd, which jointly selects the candidates predicted to be most and least relevant. By filling the ranking from both ends, DualEnd constructs a complete ranking of 100 candidates in 50 LLM comparisons. Experiments with nine open-weight LLMs on TREC DL19 and DL20 show that this requires 59.4% fewer comparisons than previous top-oriented windowed Setwise with heapsort and 88.8% fewer than top-oriented windowed Setwise with bubblesort, even though those baselines target only the top-10 rankings while DualEnd targets the full ranking.
    
[^17]: Infinity搜索：基于q-度量空间投影的近似向量搜索

    Infinity Search: Approximate Vector Search with Projections on q-Metric Spaces

    [https://arxiv.org/abs/2506.06557](https://arxiv.org/abs/2506.06557)

    该论文提出将任意相异度函数投影到超度量空间并学习该投影的近似形式，在保持最近邻关系的同时实现最坏情况复杂度仅为树深度的近似向量搜索，并将该方法推广到更一般的q-度量空间。

    

    超度量空间（或称无穷度量空间）由一个相异度函数定义，该函数满足强三角不等式：三角形中任意一边不大于另外两边中的较大者。我们证明，在超度量空间中使用视点树进行搜索的最坏情况复杂度等于树的深度。由于所关注的数据集通常并非超度量，我们采用一种投影算子，将任意相异度函数变换到超度量空间中，同时保持最近邻关系不变。我们进一步学习该投影算子的近似形式，以高效计算查询点与数据集中点之间的超度量距离。随后，我们解决了一个更一般的问题，即考虑q-度量空间中的投影——其中三角形各边的q次幂小于另外两边的q次幂之和。注意到使用学习到的…

    arXiv:2506.06557v3 Announce Type: replace-cross  Abstract: An ultrametric space or infinity-metric space is defined by a dissimilarity function that satisfies a strong triangle inequality in which every side of a triangle is not larger than the larger of the other two. We show that search in ultrametric spaces with a vantage point tree has worst-case complexity equal to the depth of the tree. Since datasets of interest are not ultrametric in general, we employ a projection operator that transforms an arbitrary dissimilarity function into an ultrametric space while preserving nearest neighbors. We further learn an approximation of this projection operator to efficiently compute ultrametric distances between query points and points in the dataset. We proceed to solve a more general problem in which we consider projections in $q$-metric spaces -- in which triangle sides raised to the power of $q$ are smaller than the sum of the $q$-powers of the other two. Notice that the use of learned a
    

