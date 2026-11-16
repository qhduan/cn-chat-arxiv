# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [StreamEmbed: Real-Time Indexing from Pixels for Screenshot Retrieval](https://arxiv.org/abs/2611.18516) | StreamEmbed利用屏幕像素的冗余性，通过仅依据像素差异的空间token剪枝和跨截图时间KV缓存复用，实现增量式视觉嵌入，从而对不断到来的屏幕截图进行实时编码与索引，同时保持与全图嵌入相当的精度。 |
| [^2] | [MeReader: An Offline, Privacy-Preserving, Progress-Aware AI Assistant for Narrative eBook Reading](https://arxiv.org/abs/2611.18513) | MeReader是一个离线、隐私保护且进度感知的电子书阅读AI助手，通过混合向量相似度、BM25与摘要嵌入的检索架构，将内容检索与生成严格限制在读者当前阅读进度内以避免剧透，并证明2b-4b的中等规模本地模型可在可接受延迟下达到与更大模型相当的理解能力。 |
| [^3] | [RACE: Ranking-Uncertainty Aware Capacity-Adaptive Customization for Efficient Sequential Recommendation](https://arxiv.org/abs/2611.18244) | 提出RACE框架，通过捕获模型预测的排序不确定性并将其转化为主动的资源分配器，以不确定性感知的压缩比率动态调整模型结构和输入序列，从而解决序列推荐中统一压缩比率导致的容量-复杂度失配问题，实现资源受限环境下的高效推荐。 |
| [^4] | [Explanations that Transfer: Interpreting Medical Retrieval with Sparse AutoEncoders](https://arxiv.org/abs/2611.18179) | 本文提出将稀疏自编码器应用于医学信息检索，通过结合重建目标与InfoNCE对比损失，将不透明的稠密文本嵌入转化为语义可解释的表示，同时保持对新临床语料库的可迁移性。 |
| [^5] | [Who Counts as Exposed? Measurement Choices in an Open, Re-estimable Index of Occupational AI Exposure](https://arxiv.org/abs/2611.18168) | 本文构建了一个开放、可重新估计的任务级职业AI暴露度指数，通过系统改变上下文与校准等设计决策，揭示了测量方法的选择——尤其是如何评估依赖尚未建成的软件集成的暴露度——在很大程度上决定了哪些工人被判定为暴露于AI。 |
| [^6] | [RecMem: Recurrent Memory Compression for Long-Sequence Generative Recommendation](https://arxiv.org/abs/2611.17921) | RecMem提出一种循环记忆压缩框架，将长交互历史压缩为固定数量的记忆槽，使内存与计算开销不随历史长度增长，同时通过推荐监督与流一致性训练保证预测信息保真与不同处理顺序下的一致性。 |
| [^7] | [Towards High-Performance Multimodal Discovery: Evaluating Retrieval Efficacy and Operational Scalability in Digital Libraries](https://arxiv.org/abs/2611.17726) | 该论文提出将数字图书馆页面（同时涵盖文本与图像）转换为数值向量以克服传统倒排索引对视觉语义的“盲区”，并系统评估了这种多模态检索方法的检索效能与运行可扩展性。 |
| [^8] | [LEGIT: Live Eligibility Gating over Identifier Tries for Constrained Decoding of Heterogeneous Pages](https://arxiv.org/abs/2611.10357) | 提出 LEGIT 约束机制，通过对标识符字典树叶节点进行深度优先编号、把随请求和 token 动态变化的资格规则转化为高效的范围查找，使生成式推荐系统在整页解码时能精确且快速地保证每个语义 ID 真实、合规且不重复。 |
| [^9] | [Federated Vector Retrieval across Heterogeneous VectorDBs](https://arxiv.org/abs/2611.01514) | HVMap通过模糊对比损失学习不同嵌入空间之间的双向映射，在无需重新嵌入或迁移语料库的情况下，实现了跨异构向量数据库的高质量联邦检索。 |
| [^10] | [Prompt-Based Abstention Fails Under Misleading Context: A Controlled Study of Small Frozen RAG Models](https://arxiv.org/abs/2608.22228) | 本研究揭示了基于提示的弃权机制在RAG中无法有效应对误导性证据，模型在明确提示下仍会高比例回答误导性问题，并常复述错误实体，而思维链改善有限。 |
| [^11] | [Federated User Behavior Modeling for Privacy-Preserving LLM Recommendation](https://arxiv.org/abs/2604.14833) | 本文提出一种联邦用户行为建模方法，在不跨域共享用户身份与行为数据的前提下，解决跨域对齐、模态异质性和协同过滤信号融合等难题，实现基于大语言模型的隐私保护跨域推荐。 |
| [^12] | [CPGRec+: A Balance-oriented Framework for Personalized Video Game Recommendations](https://arxiv.org/abs/2604.14586) | 本文提出了CPGRec+框架，通过偏好感知边缘重加权（PER）模块和利用大型语言模型能力，解决游戏推荐中准确性与多样性的权衡，并缓解过平滑问题。 |
| [^13] | [Evaluating Chain-of-Thought Reasoning through Reusability and Verifiability](https://arxiv.org/abs/2602.17544) | 该论文提出用发送方-接收方框架评估模型间思维链推理传递的可靠性、有用性和安全性，并引入可验证性与可纠正性两个度量指标。 |
| [^14] | [ScrapeGraphAI-100k: Dataset for Schema-Constrained LLM Generation](https://arxiv.org/abs/2602.15189) | 本文发布了首个基于真实使用遥测数据构建的大规模模式约束抽取数据集 ScrapeGraphAI-100k，包含 93,695 条配对真实网页内容、提示词、模式与大语言模型响应及符合性标签的事件，覆盖 25,453 个独特模式，并揭示了模式复杂度增长带来的急剧失败阈值。 |
| [^15] | [Chi-Square Wavelet Graph Neural Networks for Heterogeneous Graph Anomaly Detection](https://arxiv.org/abs/2505.18934) | 提出基于新型卡方小波滤波器的谱图神经网络框架ChiGAD，通过多图卡方滤波捕获跨元路径异常信号、交互式元图卷积在维度对齐中保留高频信息，有效解决了异构图异常检测中的语义捕获与类别不平衡难题。 |

# 详细

[^1]: StreamEmbed：面向屏幕截图检索的从像素到索引的实时编码

    StreamEmbed: Real-Time Indexing from Pixels for Screenshot Retrieval

    [https://arxiv.org/abs/2611.18516](https://arxiv.org/abs/2611.18516)

    StreamEmbed利用屏幕像素的冗余性，通过仅依据像素差异的空间token剪枝和跨截图时间KV缓存复用，实现增量式视觉嵌入，从而对不断到来的屏幕截图进行实时编码与索引，同时保持与全图嵌入相当的精度。

    

    屏幕历史可以作为人类和智能体的可搜索视觉记忆，但与离线语料库不同，它必须在每张截图到达时即时进行编码和索引。基于OCR的索引方式速度快，但会丢失视觉信息；而视觉编码则需要超过两秒的时间。然而屏幕是增量变化的：大多数像素在单张截图内部以及跨截图之间都是重复的。这引出了一个问题：视觉编码器是否需要为每张新截图从头重新计算其表示？我们提出了StreamEmbed，一个利用这种冗余性的增量视觉嵌入系统。它仅由像素差异引导，结合了空间token剪枝（跳过与邻近块重复的块）和时间KV复用（对跨截图未变化的块复用缓存的键和值）。一个小的可学习适配器对剪枝和复用进行补偿，以保持与完整截图嵌入的一致性。在ScreenMemEval——我们的基准测试（包含针对18……的10,426个问题）——上……（摘要在此处截断）

    arXiv:2611.18516v1 Announce Type: cross  Abstract: Screen history can serve as searchable visual memory for people and agents, but unlike an offline corpus, it must be encoded and indexed as each screenshot arrives. OCR-based indexing is fast but loses visual information; visual encoding takes over two seconds. Yet screens change incrementally: most pixels repeat within and across screenshots. This raises a question: do visual encoders need to recompute their representations from scratch for every new screenshot? We present StreamEmbed, a system for incremental visual embedding that exploits this redundancy. Guided solely by pixel differences, it combines spatial token pruning, which skips blocks that repeat a neighbor, with temporal KV reuse, which reuses cached keys and values for blocks unchanged across screenshots. A small learned adapter compensates for pruning and reuse to preserve fidelity to full-screenshot embeddings. On ScreenMemEval, our benchmark of 10,426 questions over 18
    
[^2]: MeReader：面向叙事类电子书阅读的离线、隐私保护、进度感知AI助手

    MeReader: An Offline, Privacy-Preserving, Progress-Aware AI Assistant for Narrative eBook Reading

    [https://arxiv.org/abs/2611.18513](https://arxiv.org/abs/2611.18513)

    MeReader是一个离线、隐私保护且进度感知的电子书阅读AI助手，通过混合向量相似度、BM25与摘要嵌入的检索架构，将内容检索与生成严格限制在读者当前阅读进度内以避免剧透，并证明2b-4b的中等规模本地模型可在可接受延迟下达到与更大模型相当的理解能力。

    

    本文介绍了MeReader，这是一个隐私保护、离线且进度感知的AI助手，旨在作为消费级硬件上叙事阅读的上下文伴侣。与传统的基于大语言模型（LLM）的系统不同——后者以一次性处理全部文本的方式在线处理文档/书籍，从而存在叙事剧透的风险——MeReader将检索和生成都限制在读者在文本中的具体阅读位置。该系统集成了混合检索架构，结合向量相似度、BM25和摘要嵌入，以确保对读者当前上下文的高保真度。我们使用定量理解指标和基于LLM的定性独立评审，在经典英文书籍上对该系统进行了评估。结果确定了一个质量-延迟边界，表明中等规模的本地模型（2b-4b）可以在保持可接受（延迟）的同时，达到与更大模型（5b+）相当的理解能力。

    arXiv:2611.18513v1 Announce Type: cross  Abstract: This paper introduces MeReader, a privacy-preserving, offline and progress-aware AI assistant designed to function as a contextual companion for narrative reading on consumer hardware. Unlike conventional Large Language Model (LLM)-based systems that process documents/books online in an all-texts-at-once type of way, thereby risking narrative spoilers, MeReader confines both retrieval and generation to the reader's distinct position in the text. The system integrates a hybrid retrieval architecture, combining vector similarity, BM25, and summary embeddings, to ensure high fidelity to the reader's current context. We evaluate the system across classical English books using both quantitative comprehension metrics and qualitative LLM-based independent judging. The results identify a quality-latency frontier, showing that mid-sized local models (2b-4b) can achieve comprehension parity with larger counterparts (5b+) while maintaining accept
    
[^3]: RACE：面向高效序列推荐的排序不确定性感知容量自适应定制

    RACE: Ranking-Uncertainty Aware Capacity-Adaptive Customization for Efficient Sequential Recommendation

    [https://arxiv.org/abs/2611.18244](https://arxiv.org/abs/2611.18244)

    提出RACE框架，通过捕获模型预测的排序不确定性并将其转化为主动的资源分配器，以不确定性感知的压缩比率动态调整模型结构和输入序列，从而解决序列推荐中统一压缩比率导致的容量-复杂度失配问题，实现资源受限环境下的高效推荐。

    

    高效序列推荐已成为提供实时个性化服务的一种有前景的范式。为了在资源受限的环境下部署模型，模型压缩成为必要手段，然而现有方法通常采用统一的压缩比率，忽视了用户意图复杂性的异质性。这种“一刀切”的方式导致了根本性的“容量-复杂度失配”问题，即计算资源要么浪费在简单实例上，要么对复杂实例而言不足。为解决这一问题，我们提出了RACE（排序不确定性感知的容量自适应定制框架），该框架利用不确定性感知的压缩比率，动态调整模型结构和输入序列。为实现容量自适应推理，RACE捕获模型预测的内在不确定性，并将其转化为一种主动的资源分配器……

    arXiv:2611.18244v1 Announce Type: cross  Abstract: Efficient sequential recommendation has emerged as a promising paradigm for providing real-time personalized service. To deploy models under resource-constrained environment, model compression becomes a necessity, yet existing methods typically apply uniform compression ratios, overlooking the heterogeneity in user intent complexity. This "one-size-fits-all" approach leads to a fundamental Capacity-Complexity Mismatch, where computational resources are either wasted on simple instances or insufficient for complex ones. To address this, we propose RACE (Ranking-uncertainty Aware Capacity-Adaptive Customization for Efficient Sequential Recommendation), a framework that dynamically adapts both model structures and input sequences with uncertainty-aware compression ratios. To ensure capacity-adaptive inference, RACE captures the model's intrinsic uncertainty of its predictions and transforms it into a proactive resource allocator. Specific
    
[^4]: 可迁移的解释：用稀疏自编码器解读医学信息检索

    Explanations that Transfer: Interpreting Medical Retrieval with Sparse AutoEncoders

    [https://arxiv.org/abs/2611.18179](https://arxiv.org/abs/2611.18179)

    本文提出将稀疏自编码器应用于医学信息检索，通过结合重建目标与InfoNCE对比损失，将不透明的稠密文本嵌入转化为语义可解释的表示，同时保持对新临床语料库的可迁移性。

    

    回答用户查询的医学AI系统依赖于从大型文档集合中检索医学知识。在此背景下，在医学数据集上微调通用模型可以提升检索相关性。然而，现代检索系统所依赖的稠密嵌入在很大程度上仍是不透明的，难以理解其检索结果背后的机制。此外，针对特定医学数据集进行适配可能会失去对新临床语料库的可迁移性。在本工作中，我们通过将稀疏自编码器（SAEs）应用于医学信息检索来解决这些问题，将稠密文本嵌入转化为语义上可解释的表示。为此，我们的模型将重建目标与InfoNCE对比损失相结合，后者能够直接保留查询与文档之间的相关性。我们在六个医学检索数据集上（涵盖域内与域外场景）评估了三种嵌入模型。在医学语料库上训练后，我们的稀疏……

    arXiv:2611.18179v1 Announce Type: cross  Abstract: Medical AI systems that answer users' queries rely on retrieving medical knowledge from large document collections. In this context, finetuning general models on medical datasets can improve retrieval relevance. Yet the dense embeddings underlying modern retrieval systems remain largely opaque, making it difficult to understand the mechanics behind their results. Moreover, adapting to a specific medical dataset risks losing transferability to new clinical corpora. In this work, we tackle these issues by applying Sparse Autoencoders (SAEs) to medical information retrieval, transforming dense text embeddings into semantically interpretable representations. To do so, our model combines a reconstruction objective with an InfoNCE contrastive loss that directly preserves query-document relevance. We evaluate three embedding models across six medical retrieval datasets, both in-domain and out-of-domain. Trained on a medical corpus, our sparse
    
[^5]: 谁算作“被暴露”？开放、可重新估计的职业AI暴露度指数中的测量选择

    Who Counts as Exposed? Measurement Choices in an Open, Re-estimable Index of Occupational AI Exposure

    [https://arxiv.org/abs/2611.18168](https://arxiv.org/abs/2611.18168)

    本文构建了一个开放、可重新估计的任务级职业AI暴露度指数，通过系统改变上下文与校准等设计决策，揭示了测量方法的选择——尤其是如何评估依赖尚未建成的软件集成的暴露度——在很大程度上决定了哪些工人被判定为暴露于AI。

    

    职业AI暴露度指数被用作经济分析、劳动力规划和政策制定的输入，而这些指数中内嵌的选择决定了哪些工人被描述为暴露于AI。我们构建了一个开放、可重新估计的任务级指数，并通过改变其两个主要设计决策——上下文和校准——来展示结果在多大程度上取决于这些选择。对于O*NET中的18,796个职业-任务对，七个开放权重的推理模型在统一评分标准下对暴露度进行评估。每个模型分别在有或没有检索到的新闻与研究段落、工作场所描述符以及基于瓶颈的推理协议作为上下文的情况下进行评分，且该集成结果分别在有或没有针对观测到的Claude使用情况的单纯形约束校准下进行组合。与主流的2023年基准相比，最大的差异并不在于对AI能力的分歧，而在于对那些依赖于尚未建成的软件集成的暴露应给予多少权重——这是一个任何数据都无法裁决的选择。

    arXiv:2611.18168v1 Announce Type: new  Abstract: Occupational AI exposure indices are used as inputs to economic analysis, workforce planning, and policy, and the choices built into them decide which workers are described as exposed. We build an open, re-estimable task-level index and vary its two main design decisions, context and calibration, to show how much the answer depends on them. For each of the 18,796 occupation--task pairs in \onet, seven open-weight reasoning models rate exposure under a common rubric. Each model rates with or without a context of retrieved news and research passages, workplace descriptors, and a bottleneck-based reasoning protocol, and the ensemble is combined with or without simplex-constrained calibration to observed Claude usage. The largest difference from the dominant 2023 benchmark is not a disagreement about what AI can do but about how much credit to give exposure that depends on software integrations not yet built, a choice that no data can settle
    
[^6]: RecMem：面向长序列生成式推荐的循环记忆压缩

    RecMem: Recurrent Memory Compression for Long-Sequence Generative Recommendation

    [https://arxiv.org/abs/2611.17921](https://arxiv.org/abs/2611.17921)

    RecMem提出一种循环记忆压缩框架，将长交互历史压缩为固定数量的记忆槽，使内存与计算开销不随历史长度增长，同时通过推荐监督与流一致性训练保证预测信息保真与不同处理顺序下的一致性。

    

    生成式推荐受益于长交互历史，但直接处理这些历史的成本会随历史长度的增长而增加。我们提出RecMem，一个循环记忆压缩框架，其设计遵循三个目标：空间不变性、语义保真度和终身可演化性。RecMem采用分叉式编码器-解码器（共享底层）以及逐记忆槽的遗忘门控更新，将历史片段压缩到一组固定的记忆槽中。推荐监督训练使记忆保留具有预测价值的信息，而流一致性训练则对齐顺序处理与打包处理两种方式下的预测结果。通过构造，记忆状态和解码器上下文与历史长度无关；在收缩性和近似半群一致性条件下，我们进一步推导出这两种处理顺序之间状态差异的与时间视界无关的界限。在MerRec数据集上，RecMem仅使用16个记忆槽和216个token的解码器上下文……

    arXiv:2611.17921v1 Announce Type: new  Abstract: Generative recommendation benefits from long interaction histories, but the cost of processing them directly grows with history length. We propose RecMem, a recurrent memory compression framework guided by three desiderata: space invariance, semantic fidelity, and lifelong evolvability. RecMem compresses historical segments into a fixed set of memory slots using a bifurcated encoder-decoder with shared lower layers and slot-wise forget-gated updates. Recommendation supervision trains the memory to preserve predictive information, while stream-consistency training aligns predictions from sequential and packed processing. By construction, the memory state and decoder context are independent of history length; under contraction and approximate semigroup consistency, we further derive a horizon-independent bound on the state discrepancy between these two processing orders. On MerRec, RecMem uses 16 memory slots and a 216-token decoder contex
    
[^7]: 迈向高性能多模态发现：评估数字图书馆中的检索效能与运行可扩展性

    Towards High-Performance Multimodal Discovery: Evaluating Retrieval Efficacy and Operational Scalability in Digital Libraries

    [https://arxiv.org/abs/2611.17726](https://arxiv.org/abs/2611.17726)

    该论文提出将数字图书馆页面（同时涵盖文本与图像）转换为数值向量以克服传统倒排索引对视觉语义的“盲区”，并系统评估了这种多模态检索方法的检索效能与运行可扩展性。

    

    拥有数百万册数字化藏书的大型数字存储库是科学与学术发现的重要引擎。然而，随着馆藏规模的不断增长，查找特定的研究或教育材料仍然是一项重大挑战。文本的多模态特性进一步加剧了这一困难，即这些文本将密集的叙述内容与重要的说明性图表、注释示意图以及复杂的版面布局结合在一起。从历史上看，文献发现一直由以倒排索引（如 Lucene 或 Solr）实现的全文本搜索索引提供支持。虽然这些系统在处理文本字符串方面具有良好的可扩展性，但它们对视觉信息的语义细微差别在根本上是“视而不见”的。我们探索了通过将文本转换为数值向量来提升发现能力的好处。通过将页面图像表示为数值向量而非 OCR 提取的文本，并对页面上的文本和图像同时进行这种向量化处理，我们能够同时捕获文本和（视觉）……（摘要在此处截断）

    arXiv:2611.17726v1 Announce Type: new  Abstract: Large-scale digital repositories with million plus digitized volumes serve as vital engines for scientific and scholarly discovery. However, as collections grow, finding specific research or educational material remains a significant challenge. This difficulty is compounded by the multimodal nature of texts, that is, texts which combine dense narrative content with essential illustrative charts, annotated diagrams, and complex layouts. Historically, discovery has been supported by full-text search indexes implemented as inverted indexes (e.g., Lucene or Solr). While these systems scale well for textual strings, they are fundamentally "blind" to the semantic nuance of visual information. We explore the benefit of discovery through texts that have been converted to numerical vectors. By representing page images as numerical vectors rather than OCR-extracted text, and doing so for both text and images on a page, we capture both textual and 
    
[^8]: LEGIT：基于标识符字典树的实时资格门控，用于异构页面的受限解码

    LEGIT: Live Eligibility Gating over Identifier Tries for Constrained Decoding of Heterogeneous Pages

    [https://arxiv.org/abs/2611.10357](https://arxiv.org/abs/2611.10357)

    提出 LEGIT 约束机制，通过对标识符字典树叶节点进行深度优先编号、把随请求和 token 动态变化的资格规则转化为高效的范围查找，使生成式推荐系统在整页解码时能精确且快速地保证每个语义 ID 真实、合规且不重复。

    

    生成式推荐器用一段简短的 token 序列（即语义 ID，Semantic ID）来命名每个候选，并通过生成这些 token 来完成推荐。一些生成式推荐器，包括我们正在构建的 Uber Eats 搜索引擎，会在一次解码过程中写出整个结果页面，其中包含商店及其菜单项。这样的页面上，每个语义 ID 都必须指向一个真实的、且符合本次搜索资格的商店或条目（例如，能够配送到用户所在位置的商店），并且尚未出现在该页面上；此外，每个条目还必须属于其上方的商店。在商品目录上构建字典树可以保证标识符的真实性，但资格性与重复性约束会随着每次请求和每个 token 动态变化，而我们尝试过的语法约束解码器无法既精确又快速地执行这些不断变化的规则。我们将这一约束命名为 LEGIT，即“基于标识符字典树的实时资格门控”。对字典树叶节点进行深度优先编号，可以将每条规则转化为对一棵商店字典树和一棵条目字典树的范围查找。

    arXiv:2611.10357v2 Announce Type: replace-cross  Abstract: A generative recommender names each candidate with a short token sequence, a Semantic ID, and recommends by generating those tokens. Some generative recommenders, including the Uber Eats search engine we are building, write a whole results page, stores with their menu items, in one decoding pass. Every Semantic ID on such a page must name a real store or item that is eligible for the search, such as a store that delivers to the user's location, and that is not already on the page. Each item must also belong to the store above it. A trie over the catalog keeps identifiers real, but eligibility and repetition change with every request and token, and the grammar-constrained decoder we tried could not enforce these changing rules exactly and quickly. We call this constraint LEGIT, for live eligibility gating over identifier tries. Numbering trie leaves depth first turns each rule into a range lookup over one store trie and one item
    
[^9]: 跨异构向量数据库的联邦向量检索

    Federated Vector Retrieval across Heterogeneous VectorDBs

    [https://arxiv.org/abs/2611.01514](https://arxiv.org/abs/2611.01514)

    HVMap通过模糊对比损失学习不同嵌入空间之间的双向映射，在无需重新嵌入或迁移语料库的情况下，实现了跨异构向量数据库的高质量联邦检索。

    

    向量数据库如今已成为AI检索系统的核心基础设施，然而许多实际部署中需要对多个独立构建的向量数据库进行查询，而这些数据库采用了不同的嵌入模型构建。这带来了一个在异构向量数据库上进行联邦检索的系统难题：向量与相似度分数在不同嵌入空间之间无法直接比较，而通过全库重新嵌入和重新索引来彻底解决这种不兼容性，在运营上往往代价过高。因此，现有方法在检索质量、构建成本和查询吞吐量之间面临艰难的权衡。我们提出了HVMap，一个用于跨异构向量数据库联邦检索的轻量级兼容层。HVMap无需将数据库重建到统一空间，而是利用模糊对比损失学习嵌入空间之间的双向映射，从而在无需语料库迁移的情况下实现经过校准的跨数据库评分。

    arXiv:2611.01514v2 Announce Type: replace-cross  Abstract: Vector databases are now core infrastructure for AI retrieval systems, yet many real deployments must serve queries over multiple independently constructed vectorDBs built with different embedding models. This creates a systems problem of federated retrieval over heterogeneous vectorDBs: vectors and similarity scores are not directly comparable across embedding spaces, while fully resolving this incompatibility through corpus-wide re-embedding and re-indexing is often operationally prohibitive. Existing methods therefore face a difficult trade-off between retrieval quality, construction cost, and query throughput. We present HVMap, a lightweight compatibility layer for federated retrieval across heterogeneous vectorDBs. Instead of rebuilding databases into a unified space, HVMap learns bidirectional mappings between embedding spaces using a fuzzy contrastive loss, enabling calibrated cross-database scoring without corpus migrat
    
[^10]: 基于提示的弃权在误导性情境下失效：小型冻结RAG模型的受控研究

    Prompt-Based Abstention Fails Under Misleading Context: A Controlled Study of Small Frozen RAG Models

    [https://arxiv.org/abs/2608.22228](https://arxiv.org/abs/2608.22228)

    本研究揭示了基于提示的弃权机制在RAG中无法有效应对误导性证据，模型在明确提示下仍会高比例回答误导性问题，并常复述错误实体，而思维链改善有限。

    

    在检索增强生成（RAG）中，缺失证据与误导性证据并非同一问题，但基于提示的弃权机制却对二者一视同仁。模型在上下文缺失时会弃权，但在上下文具有误导性时却不会。我们引入了GRAB-RAG（检索增强生成的分级弃权基准），这是一个配对基准，在Natural Questions和HotpotQA中，针对四种上下文条件（支持性、退化、缺失和误导性）测试相同的问题。在误导性条件下，我们编辑一个黄金段落以支持错误答案，并将其置于其他检索到的段落中。我们在两个问答基准上，对三个小型冻结模型（3.8B至8B参数）测试了五种弃权策略。模型在证据缺失时可靠地弃权，但在明确的弃权提示下，仍回答了41.6%的误导性问题，其中63%的回答逐字复述了植入的错误实体。思维链几乎没有带来额外益处。一个生成器...

    arXiv:2608.22228v1 Announce Type: new  Abstract: Missing and misleading evidence are not the same problem in retrieval-augmented generation (RAG), but prompt-based abstention treats them alike. Models abstain when context is absent, not when it is misleading. We introduce GRAB-RAG (Graded Abstention Benchmark for Retrieval-Augmented Generation), a paired benchmark that tests the same questions across four context conditions (supportive, degraded, missing, and misleading) in Natural Questions and HotpotQA. In the misleading condition, we edit a gold passage to support a wrong answer and place it among other retrieved passages. We test five abstention policies on three small frozen models (3.8B--8B) across two QA benchmarks. Models abstain reliably when evidence is missing, but under explicit abstention prompting still answer 41.6% of misleading questions, with 63% of those answers echoing the planted wrong entity verbatim. Chain-of-thought provides little additional benefit. A generator
    
[^11]: 面向隐私保护大语言模型推荐的联邦用户行为建模

    Federated User Behavior Modeling for Privacy-Preserving LLM Recommendation

    [https://arxiv.org/abs/2604.14833](https://arxiv.org/abs/2604.14833)

    本文提出一种联邦用户行为建模方法，在不跨域共享用户身份与行为数据的前提下，解决跨域对齐、模态异质性和协同过滤信号融合等难题，实现基于大语言模型的隐私保护跨域推荐。

    

    大语言模型在推荐系统中已取得巨大成功。然而，用户数据的有限性和稀疏性往往限制了LLM有效建模用户行为模式的能力。为解决这一问题，现有研究通过开展跨域推荐任务探索了跨域解决方案。但以往方法通常假设域之间存在重叠且可以随时访问。目前没有任何LLM方法能够解决跨域推荐（CDR）场景中的隐私保护问题，即隐私保护跨域推荐（PPCDR）。使用LLM开展非重叠的PPCDR极具挑战性，原因在于：1）无法跨域共享用户身份或行为数据，阻碍了有效的跨域对齐；2）跨域数据模态的异质性使知识整合变得复杂；3）将传统推荐模型的协同过滤信号与LLM相融合十分困难，因为它们运行在截然不同……

    arXiv:2604.14833v2 Announce Type: replace  Abstract: Large Language Models have shown great success in recommender systems. However, the limited and sparse nature of user data often restricts the LLM's ability to effectively model behavior patterns. To address this, existing studies have explored cross-domain solutions by conducting Cross-Domain Recommendation tasks. But previous methods typically assume domains are overlapped and can be accessed readily. None of the LLM methods address the privacy-preserving issues in the CDR settings, that is, Privacy-Preserving Cross-Domain Recommendation. Conducting non-overlapping PPCDR with LLM is challenging since: 1)The inability to share user identity or behavioral data across domains impedes effective cross-domain alignment. 2)The heterogeneity of data modalities across domains complicates knowledge integration. 3)Fusing collaborative filtering signals from traditional recommendation models with LLMs is difficult, as they operate within disti
    
[^12]: CPGRec+：一种面向个性化视频游戏推荐的平衡导向框架

    CPGRec+: A Balance-oriented Framework for Personalized Video Game Recommendations

    [https://arxiv.org/abs/2604.14586](https://arxiv.org/abs/2604.14586)

    本文提出了CPGRec+框架，通过偏好感知边缘重加权（PER）模块和利用大型语言模型能力，解决游戏推荐中准确性与多样性的权衡，并缓解过平滑问题。

    

    arXiv:2604.14586v4 公告类型：交叉替换 摘要：游戏行业的快速扩张要求推荐系统适应其动态格局。现有的基于图神经网络（GNN）的方法主要优先考虑准确性而非多样性，忽视了它们之间的固有权衡。为解决这一问题，我们先前提出了CPGRec，一种平衡导向的游戏推荐系统。然而，CPGRec未能考虑玩家-游戏交互中的关键差异，这些差异在反映玩家个人偏好方面具有不同重要性，并可能加剧基于GNN模型固有的过平滑问题。此外，现有方法未充分利用大型语言模型（LLMs）的推理能力和广泛知识来解决这些局限性。为弥补这一差距，我们提出了两个新模块。首先，偏好感知边缘重加权（PER）模块分配带符号的边缘权重，以定性区分显著的玩家兴趣和厌恶。

    arXiv:2604.14586v4 Announce Type: replace-cross  Abstract: The rapid expansion of gaming industry requires advanced recommender systems tailored to its dynamic landscape. Existing Graph Neural Network (GNN)-based methods primarily prioritize accuracy over diversity, overlooking their inherent trade-off. To address this, we previously proposed CPGRec, a balance-oriented gaming recommender system. However, CPGRec fails to account for critical disparities in player-game interactions, which carry varying significance in reflecting players' personal preferences and may exacerbate over-smoothness issues inherent in GNN-based models. Moreover, existing approaches underutilize the reasoning capabilities and extensive knowledge of large language models (LLMs) in addressing these limitations. To bridge this gap, we propose two new modules. First, Preference-informed Edge Reweighting (PER) module assigns signed edge weights to qualitatively distinguish significant player interests and disinterest
    
[^13]: 通过可复用性与可验证性评估思维链推理

    Evaluating Chain-of-Thought Reasoning through Reusability and Verifiability

    [https://arxiv.org/abs/2602.17544](https://arxiv.org/abs/2602.17544)

    该论文提出用发送方-接收方框架评估模型间思维链推理传递的可靠性、有用性和安全性，并引入可验证性与可纠正性两个度量指标。

    

    AI应用日益由多个模型组合而成，其中一个模型基于另一个模型产生的输出进行操作。所传递的文本还可以携带发送方模型的推理过程。思维链（Chain-of-Thought）是我们在实验中使用的此类文本的实例。我们针对这种模型间推理传递提出三个问题：推理是否被可靠地传递？它对接收方是否有用？它是否安全？我们通过“发送方-接收方”实验设置来研究这三个问题：一个发送方LLM生成推理轨迹，一个独立的接收方LLM必须基于该轨迹执行操作。两个模型都不知道彼此的身份，因此发送方无法针对特定接收方定制其推理。可靠性关注接收方能否恢复出发送方所预期的结论，我们用“可验证性”来衡量它。有用性关注所传递的推理能否帮助接收方纠正自身的错误，我们用“可纠正性”来衡量它。安全性关注所传递的推理是否使接收方……

    arXiv:2602.17544v2 Announce Type: replace  Abstract: AI applications increasingly compose multiple models, where one model acts on the output produced by another model. The transferred text can also carry the reasoning of the sender model. Chain-of-thought is our experimental instance of such text. We ask three questions about such inter-model reasoning transfer. Is the reasoning transmitted reliably? Is it useful to the receiver? Is it safe? We study these three questions with a Sender-Receiver setup. A Sender LLM produces a reasoning trace. A separate Receiver LLM must act on it. Neither model knows the identity of the other. The Sender therefore cannot tailor its reasoning to a particular Receiver. Reliability asks whether the Receiver recovers the Sender's intended conclusion. We measure it with Verifiability. Usefulness asks whether transferred reasoning helps a Receiver correct its mistakes. We measure it with Correctability. Safety asks whether transferred reasoning makes a Rece
    
[^14]: ScrapeGraphAI-100k：面向模式约束的大语言模型生成数据集

    ScrapeGraphAI-100k: Dataset for Schema-Constrained LLM Generation

    [https://arxiv.org/abs/2602.15189](https://arxiv.org/abs/2602.15189)

    本文发布了首个基于真实使用遥测数据构建的大规模模式约束抽取数据集 ScrapeGraphAI-100k，包含 93,695 条配对真实网页内容、提示词、模式与大语言模型响应及符合性标签的事件，覆盖 25,453 个独特模式，并揭示了模式复杂度增长带来的急剧失败阈值。

    

    生成符合指定 JSON 模式（schema）的输出，是现代大语言模型中工具调用、结构化抽取和知识库构建的基础。尽管这一任务至关重要，但现有的公开数据集仍然规模小、多为合成数据或仅包含纯文本，且很少将真实网页内容与实践中使用的提示词和模式配对。我们提出了 ScrapeGraphAI-100k 数据集，包含 93,695 条模式约束抽取事件，这些事件通过 ScrapeGraphAI 的选择性退出（opt-out）遥测机制于 2025 年第二至第三季度收集，并从 900 万条原始事件中按模式去重和均衡而来。该语料库涵盖 25,453 个独特模式和 15 种可识别的语言，其中英语内容占检测内容的 89%。每个实例将转换为 Markdown 格式的网页内容与提示词、模式、大语言模型响应以及基于 jsonschema-rs 的逐样本结构符合性标签配对。我们刻画了整个语料库的结构多样性，并发现随着模式复杂度的增长，模型失败率存在急剧上升的临界阈值。作为案例研究，一个 1.（原文摘要在此处截断）

    arXiv:2602.15189v3 Announce Type: replace-cross  Abstract: Producing output that conforms to a specified JSON schema underlies tool use, structured extraction, and knowledge base construction in modern large language models. Despite this centrality, public datasets for the task remain small, synthetic, or text-only, and rarely pair real page content with the prompts and schemas used in practice. We introduce ScrapeGraphAI-100k, 93,695 schema-constrained extraction events collected via opt-out ScrapeGraphAI telemetry in Q2--Q3 2025, deduplicated and balanced by schema from 9M raw events. The corpus spans 25,453 unique schemas, across 15 named languages, with English covering 89% of detected content. Each instance pairs Markdown-converted page content with a prompt, schema, LLM response, and per-example jsonschema-rs structural conformance labels. We characterize structural diversity across the corpus and identify sharp failure thresholds as schema complexity grows. As a case study, a 1.
    
[^15]: 用于异构图异常检测的卡方小波图神经网络

    Chi-Square Wavelet Graph Neural Networks for Heterogeneous Graph Anomaly Detection

    [https://arxiv.org/abs/2505.18934](https://arxiv.org/abs/2505.18934)

    提出基于新型卡方小波滤波器的谱图神经网络框架ChiGAD，通过多图卡方滤波捕获跨元路径异常信号、交互式元图卷积在维度对齐中保留高频信息，有效解决了异构图异常检测中的语义捕获与类别不平衡难题。

    

    异构网络中的图异常检测由于节点和边的异质性而面临独特的挑战。现有的图神经网络（GNN）方法主要针对同构图异常检测，因此无法解决三个关键问题：（C1）跨不同元路径捕获异常信号和丰富语义；（C2）在异构信息网络（HIN）维度对齐中保留高频内容；（C3）在类别不平衡条件下从困难异常样本中有效学习。为克服这些问题，我们提出了ChiGAD，一种基于新型卡方滤波器的谱图神经网络框架，其灵感来源于小波在多个领域的有效性。具体而言，ChiGAD包含：（1）多图卡方滤波器，通过对每个元路径图应用专用卡方滤波器来捕获异常信息；（2）交互式元图卷积，在保留高频信息的同时对齐特征并融合异构消息。

    arXiv:2505.18934v3 Announce Type: replace-cross  Abstract: Graph Anomaly Detection (GAD) in heterogeneous networks presents unique challenges due to node and edge heterogeneity. Existing Graph Neural Network (GNN) methods primarily focus on homogeneous GAD and thus fail to address three key issues: (C1) Capturing abnormal signal and rich semantics across diverse meta-paths; (C2) Retaining high-frequency content in HIN dimension alignment; and (C3) Learning effectively from difficult anomaly samples with class imbalance. To overcome these, we propose ChiGAD, a spectral GNN framework based on a novel Chi-Square filter, inspired by the wavelet effectiveness in diverse domains. Specifically, ChiGAD consists of: (1) Multi-Graph Chi-Square Filter, which captures anomalous information via applying dedicated Chi-Square filters to each meta-path graph; (2) Interactive Meta-Graph Convolution, which aligns features while preserving high-frequency information and incorporates heterogeneous message
    

