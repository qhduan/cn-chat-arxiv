# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Retail Product Search: A Practical Approach at Target](https://arxiv.org/abs/2609.31498) | Target公司提出了一种结合词法搜索与向量搜索的混合式零售产品搜索系统，通过数据处理、嵌入训练、精度控制和多通道加权结果融合等实用方案，在保证低延迟的同时平衡相关性、收入与利润等多重目标。 |
| [^2] | [Enriching Sequential Recommendation with Graph Laplacian Positional Embeddings](https://arxiv.org/abs/2609.31253) | 该论文提出用基于物品共现图的拉普拉斯特征向量生成的冻结位置嵌入来替代SASRec中可学习的顺序位置嵌入，在不改变主干架构的情况下提升了推荐性能。 |
| [^3] | [AgentRecommender: LLM Agents Enable Customizable Recommender Systems on the User Side](https://arxiv.org/abs/2609.31166) | 提出AgentRecommender方法，利用LLM智能体的调查能力和内部知识，在无需额外数据的情况下灵活构建用户端推荐系统，使用户能够轻松创建符合自身偏好的定制化推荐系统。 |
| [^4] | [SPADE: Escaping the Popularity-Similarity Frontier to Measure Serendipitous Recommendations](https://arxiv.org/abs/2609.31164) | 提出SPADE评估指标，将物品映射到二维流行度-相似度空间并计算用户特定的帕累托前沿距离，从而同时考量流行度、相似度与用户实际相关性，有效度量意外性推荐并防止算法投机。 |
| [^5] | [CG-Probes: Recovering Guardrail Directions from Patient Query Embeddings](https://arxiv.org/abs/2609.31062) | 该论文提出CG-Probes方法，通过与肿瘤科医生合作定义医疗紧急度、心理紧急度和话题敏感性三个风险轴，并利用均值差方法从患者查询嵌入中恢复线性风险方向，从而为医疗AI助手构建有效的安全护栏。 |
| [^6] | [KuaFu: Compressing Long User Behavior into Understanding at Billion Scale](https://arxiv.org/abs/2609.31045) | 该论文提出KuaFu系统，在十亿用户规模上解决将长用户行为压缩为用户理解时序列过长与刷新吞吐量的双重工业瓶颈，并应对压缩过程可能引入的四类幻觉问题。 |
| [^7] | [QReason: Query-Focused Decoupled Chain-of-Thought for Efficient Passage Reranking](https://arxiv.org/abs/2609.30904) | QReason提出一种解耦框架，通过重写器仅生成一次面向排序的推理查询并在各滑动窗口中复用，避免重复的思维链推理，从而在保持复杂查询处理能力的同时大幅降低段落重排序的冗余与延迟。 |
| [^8] | [RecToolBench: Benchmarking Recommendation-Specific Tool Orchestration under Fuzzy User Intent](https://arxiv.org/abs/2609.30717) | 该论文提出了RecToolBench，一个基于MCP协议、用于评估推荐智能体在模糊用户意图下进行工具编排能力的基准测试，包含超过1,200个可执行任务，覆盖单工具调用、并行调用、顺序工具链和混合编排等多种复杂工具使用模式。 |
| [^9] | [Recommendation World Models for Future-State Control](https://arxiv.org/abs/2609.30711) | 提出UA-TWM效用锚定世界模型接口，使已训练的序列推荐排序器能够估计候选列表的未来后果并在效用约束下选择替代方案，从而同时提升推荐准确性与未来状态对齐。 |
| [^10] | [Component Benchmark: Hierarchical Model Profiling for Large-scale Recommendation Systems](https://arxiv.org/abs/2609.30656) | 本文提出了组件基准测试（CB）系统，以分层方式独立剖析大规模推荐系统中每个子模块的性能，并通过树状交互式可视化解决了现有工具无法将性能归因到具体子模块的问题。 |
| [^11] | [Epstein Files Engine: Agentic Search for Investigative Journalism](https://arxiv.org/abs/2609.30611) | 《纽约时报》开发的“爱泼斯坦文件引擎”通过大语言模型将记者问题转化为SQL查询，在三百万页司法部文件中检索带引用的可验证答案，助力100余名记者完成至少20篇报道，其核心创新Diff重复匹配方法放大了新颖性信号，并证明了新闻编辑室AI智能体应作为源材料接口而非自主写作者来发挥最大价值。 |
| [^12] | [Embedding Subspace Partitioning for Dynamic Multi-Objective Retrieval](https://arxiv.org/abs/2609.30601) | 提出嵌入子空间划分（ESP）框架，将嵌入分解为任务感知的子空间，并用权重可在服务时调节的子空间相似度加权和替代单一内积，使检索器无需重新训练即可动态适应多目标优先级的变化，同时缓解多目标联合优化中的目标干扰问题。 |
| [^13] | [T-RoPE: Time-Aware Rotary Position Embedding for Sequential Recommendation](https://arxiv.org/abs/2609.30576) | 提出T-RoPE，一种时间感知的旋转位置编码，通过基于时间戳的角度、可学习时间系数和多尺度频率等机制打破标准RoPE的时间平移不变性，使序列生成式推荐模型能够捕捉时间间隔、行为周期与季节性等关键时间信息。 |
| [^14] | [Nearest but Not Dearest: Shared Curator-Feedback Infrastructure for Content-Only Search and Recommendation](https://arxiv.org/abs/2609.30568) | 该论文提出将策展人反馈中的“声音失败”与“上下文失败”进行分解，并以此构建共享反馈基础设施，用于改进缺乏用户行为信号的纯内容搜索与推荐系统。 |
| [^15] | [REALMS: An AI-Assistant Conversational System for Real-Time Exact Audience Sizing over High-Dimensional Nested Profiles](https://arxiv.org/abs/2609.30547) | REALMS是一个部署在生产环境的对话式系统，结合嵌入向量检索与大语言模型NL2SQL技术，让营销人员用自然语言在数秒内对数百万级、上千属性的高维用户画像完成精确受众规模估算。 |
| [^16] | [AutoResearch at Production Scale: Failure Modes and a Multi-Agent Framework](https://arxiv.org/abs/2609.30541) | 该论文将AutoResearch范式应用于生产规模的推荐系统嵌入优化，在220多次实验中识别出基础设施脆弱、智能体记忆衰退、搜索方向停滞、迭代成本不对称和指标固化五种失效模式，并据此提出多智能体框架加以解决。 |
| [^17] | [Where Does Retrieval-Based Open-Ended Evaluation Fail? Automatic Taxonomy Induction from Long-Form Medical Answer Factuality Verification](https://arxiv.org/abs/2609.30467) | 该论文针对基于检索的开放式医学事实性验证，自动归纳出两套错误分类体系，将失败分解为五个质量维度上的检索阶段错误与六个连续步骤中的验证器推理错误，并利用LLM-as-Judge流程实现大规模自动化错误标注与分类，无需黄金答案或黄金证据。 |
| [^18] | [Bootstrapping Conversational Recommendation Agents At Spotify: Synthetic Data Generation and Self-Improvement Loops](https://arxiv.org/abs/2609.30297) | Spotify提出了一条多轮合成数据生成流水线与自我改进循环，通过基于方差的对比优化和编码智能体的迭代修复，在冷启动场景下自动优化对话式推荐智能体的规划与工具调用能力，使质量提升8%。 |
| [^19] | [SignTrace: Describe a Sign, Find the Word](https://arxiv.org/abs/2609.30295) | SignTrace 利用大语言模型增强的中国手语词典，结合动作提取、七路检索与候选重排序，让学习者仅凭日常语言描述的手部动作即可反向查找到对应手语词条及其含义，在 500 条查询的基准上达到 94.0% 的 Hit@1。 |
| [^20] | [MM-ContextFold: Context Folding for Multimodal Agentic Retrieval](https://arxiv.org/abs/2609.23121) | 提出了无需训练的 MM-ContextFold 框架，基于对约一万条轨迹的实证发现——当视觉信息被提取并文本化后原始图像变得冗余——通过“折叠”冗余视觉内容来解决多模态智能体检索中的上下文爆炸问题。 |
| [^21] | [High-probability guarantees for linear accessibility in feature superposition](https://arxiv.org/abs/2609.09556) | 该论文将特征叠加中的线性能及性建模为压缩感知问题，证明了充分维度只需线性规模（而非此前最坏情况的二次方限制）即可高概率地恢复同时激活的特征，从而量化了线性表示假设的几何约束并为稀疏自编码器和神经可解释性评估提供了理论框架。 |
| [^22] | [Bringing Agentic Search to Earth Observation Data Discovery](https://arxiv.org/abs/2607.02387) | 该论文提出了一个基于NASA地球观测知识图谱的智能体搜索框架用于地球科学数据发现，构建了包含47k查询-数据集对的开放基准NASA-EO-Bench，并通过微调神经评分器与BM25分数融合，将R@10和MRR提升至余弦基线的5倍以上。 |
| [^23] | [FlyAOC: Evaluating Agentic Ontology Curation of Drosophila Scientific Knowledge Bases](https://arxiv.org/abs/2602.09163) | FlyAOC是一个评估AI智能体从科学文献中进行端到端本体策展的基准，要求智能体在16,898篇果蝇论文中检索证据，并恢复策展人级别的结构化基因标注，涵盖功能术语、表达模式和历史同义词。 |
| [^24] | [On Function-Correcting Codes in the Lee Metric](https://arxiv.org/abs/2507.17654) | 本文将函数校正码的研究扩展到Lee度量下任意整数模环 Z_m（m≥2）上，通过引入不规则Lee距离码并刻画其最短可能长度，给出了最优冗余度的上下界。 |

# 详细

[^1]: 零售产品搜索：Target的实用方法

    Retail Product Search: A Practical Approach at Target

    [https://arxiv.org/abs/2609.31498](https://arxiv.org/abs/2609.31498)

    Target公司提出了一种结合词法搜索与向量搜索的混合式零售产品搜索系统，通过数据处理、嵌入训练、精度控制和多通道加权结果融合等实用方案，在保证低延迟的同时平衡相关性、收入与利润等多重目标。

    

    搜索是电子商务中最重要的功能之一，直接驱动着客户参与度和业务增长。一个好的产品搜索系统必须展示既相关又符合用户需求的结果。然而，零售搜索面临着独特的挑战：用户意图可能从精确匹配到开放式探索不等；搜索系统还必须平衡多个目标，如相关性、收入和利润，同时保持较低的响应时间。传统的基于关键词的方法在处理自然语言或语义查询时往往表现不佳。向量搜索有助于缓解这些问题，但可能会遗漏关键的意图信号或返回低精度的结果。在本文中，我们介绍了Target公司混合搜索系统的设计，该系统结合了词法搜索和向量搜索。我们描述了在数据处理、嵌入训练、最终结果集精度控制、多通道结果融合（我们对融合策略进行了比较并采用了加权方法）等方面的实践方案。

    arXiv:2609.31498v1 Announce Type: new  Abstract: Search is one of the most important features in e-commerce, directly driving customer engagement and business growth. A good product search system must show both relevant and desirable results. However, retail search presents unique challenges. User intent can range from exact matches to open-ended discovery. Search systems must also balance multiple goals, such as relevance, revenue, and profit, while keeping response times low. Traditional keyword-based methods often fall short in handling natural language or semantic queries. Vector search helps alleviate these issues, but it can miss key intent signals or return low-precision results. In this paper, we present the design of a hybrid search system at Target that combines lexical and vector search. We describe our approach to data processing, embedding training, precision control for the final result set, multi-channel result fusion (where we compared fusion strategies and adopted weig
    
[^2]: 用图拉普拉斯位置嵌入增强序列推荐

    Enriching Sequential Recommendation with Graph Laplacian Positional Embeddings

    [https://arxiv.org/abs/2609.31253](https://arxiv.org/abs/2609.31253)

    该论文提出用基于物品共现图的拉普拉斯特征向量生成的冻结位置嵌入来替代SASRec中可学习的顺序位置嵌入，在不改变主干架构的情况下提升了推荐性能。

    

    序列推荐模型通常依赖可学习的位置嵌入来编码用户交互的顺序。在这项工作中，我们探讨这种顺序信号是否可以被一种从物品空间导出的结构信号所替代。我们提出在SASRec中使用拉普拉斯位置嵌入：我们从训练交互数据中构建物品共现图，计算其对称归一化拉普拉斯矩阵的特征向量，并将其作为冻结的图导出位置嵌入。主干架构和训练目标保持不变。在四个公开的序列推荐基准数据集上的实验表明，这种简单的替换在大多数排序指标上提升了SASRec的性能，并且与强大的位置编码和时间编码基线相比仍具有竞争力。这些发现表明，物品-物品图结构可以成为序列推荐中标准顺序位置嵌入的有效替代品。

    arXiv:2609.31253v1 Announce Type: new  Abstract: Sequential recommenders typically rely on learnable positional embeddings to encode the order of user interactions. In this work, we ask whether this ordinal signal can be replaced by a structural one derived from the item space. We propose to use Laplacian positional embeddings in SASRec: we build an item co-occurrence graph from training interactions, compute eigenvectors of its symmetric normalized Laplacian, and use them as frozen graph-derived positional embeddings. The backbone architecture and training objective remain unchanged. Experiments on four public sequential-recommendation benchmarks show that this simple replacement improves SASRec performance on most ranking metrics and remains competitive with strong positional and temporal encoding baselines. These findings indicate that item-item graph structure can be an effective substitute for standard ordinal positional embeddings in sequential recommendation.
    
[^3]: AgentRecommender：基于LLM智能体的用户端可定制推荐系统

    AgentRecommender: LLM Agents Enable Customizable Recommender Systems on the User Side

    [https://arxiv.org/abs/2609.31166](https://arxiv.org/abs/2609.31166)

    提出AgentRecommender方法，利用LLM智能体的调查能力和内部知识，在无需额外数据的情况下灵活构建用户端推荐系统，使用户能够轻松创建符合自身偏好的定制化推荐系统。

    

    推荐系统传统上是为平台开发的。然而，这催生了许多可能有利于平台锁定用户但对用户造成困扰的现象，例如点击诱饵（标题党）、过滤气泡和虚假新闻的传播。最近，用户端推荐系统作为一种解决这一问题的新范式被提出。如果用户部署自己的推荐系统，他们就不再受制于平台的利益。然而，构建用户端推荐系统并非易事；特别是，为自己定制一个推荐系统需要额外的数据。我们提出了AgentRecommender，这是一种利用LLM智能体的调查能力和内部知识来灵活构建用户端推荐系统的方法，无需额外数据。AgentRecommender使用户能够轻松创建符合自身偏好的推荐系统。

    arXiv:2609.31166v1 Announce Type: cross  Abstract: Recommender systems have traditionally been developed for platforms. However, this has given rise to many phenomena that may be advantageous for platform lock-in but are a nuisance to users, such as clickbait, filter bubbles, and the spread of fake news. Recently, user-side recommender systems have been proposed as a new paradigm for solving this problem. If users deploy their own recommender systems, they are no longer at the mercy of the platform's interests. However, building a user-side recommender system is not trivial; in particular, customizing one for oneself requires additional data. We propose AgentRecommender, a method that leverages the investigation capability and internal knowledge of LLM agents to flexibly build user-side recommender systems without additional data. AgentRecommender allows users to easily create recommender systems tailored to their own preferences.
    
[^4]: SPADE：突破流行度-相似度边界以度量意外性推荐

    SPADE: Escaping the Popularity-Similarity Frontier to Measure Serendipitous Recommendations

    [https://arxiv.org/abs/2609.31164](https://arxiv.org/abs/2609.31164)

    提出SPADE评估指标，将物品映射到二维流行度-相似度空间并计算用户特定的帕累托前沿距离，从而同时考量流行度、相似度与用户实际相关性，有效度量意外性推荐并防止算法投机。

    

    推荐系统通过设计意外性（serendipity）来促进用户的主动探索，并打破可预测的消费循环。现有离线“准确性之外”的评估指标存在的问题是，它们往往只孤立地考察历史相似性或全局流行度。我们的目标是设计一个能够同时考察相似性、流行度和用户实际相关性的评估指标。为此，我们提出了SPADE（意外性帕累托距离评估，Serendipitous Pareto Distance Evaluation）。SPADE将所有物品映射到二维空间中，直接为每个用户计算由流行度最高且历史最相似的物品构成的帕累托前沿。最终的意外性得分通过严格针对测试集中被正确推荐的物品，计算它们到该边界的最小欧氏距离并取平均值而得到。在五个数据集和五种基线算法上的评估证实了SPADE的有效性；我们的结果表明，该指标成功防止了算法对“准确性之外”评估指标的投机利用。

    arXiv:2609.31164v1 Announce Type: cross  Abstract: Recommender systems engineer serendipity to foster active exploration and break predictable consumption cycles. The problem with existing offline beyond-accuracy metrics is that they often either isolate historical similarity or global popularity. We aim to design an evaluation metric that examines similarity, popularity, and actual user relevance. To achieve this, we introduce SPADE (Serendipitous Pareto Distance Evaluation). SPADE maps all items into a two-dimensional space to directly calculate a user-specific Pareto frontier of maximally popular and historically similar items. The final serendipity score is then computed by averaging the minimum Euclidean distance from this boundary strictly for the correctly recommended test-set items. Evaluating SPADE across five datasets and five baseline algorithms confirms its effectiveness; our results show that the metric successfully prevents algorithms from exploiting beyond-accuracy measu
    
[^5]: CG-Probes：从患者查询嵌入中恢复护栏方向

    CG-Probes: Recovering Guardrail Directions from Patient Query Embeddings

    [https://arxiv.org/abs/2609.31062](https://arxiv.org/abs/2609.31062)

    该论文提出CG-Probes方法，通过与肿瘤科医生合作定义医疗紧急度、心理紧急度和话题敏感性三个风险轴，并利用均值差方法从患者查询嵌入中恢复线性风险方向，从而为医疗AI助手构建有效的安全护栏。

    

    面向患者的AI助手有望为患者提供有价值的支持，但传入的查询可能带来医疗风险。为了构建护栏，我们与肿瘤科医生合作定义了三个有序的风险轴：医疗紧急程度、心理紧急程度和话题敏感性。我们提出临床护栏探针（CG-Probes）来从查询嵌入中测量风险。我们通过均值差方法在冻结嵌入器的归一化嵌入空间中对每个轴进行探测，将每个轴视为潜在的线性方向。为了训练探针，我们使用BERTopic对79,658条捷克语肿瘤学搜索查询进行聚类，并利用这些聚类通过少样本提示生成具有对比风险水平的查询对。我们在200条查询（90条真实，110条合成）上评估了该方法，每条查询由两名肿瘤科医生进行评分，并与两个开源权重LLM和一个前沿LLM进行对比。我们发现基于紧急程度的轴可以作为线性方向被恢复，且这些探针具有竞争力。

    arXiv:2609.31062v1 Announce Type: new  Abstract: Patient-facing AI assistants promise valuable support to patients, but incoming queries can pose medical risks. To create guardrails, we work with oncologists to define three ordinal risk axes: Medical Urgency, Psychological Urgency, and Topic Sensitivity. We propose Clinical Guardrail Probes (CG-Probes) to measure the risks from query embeddings. We probe for each axis in the normalized embedding space of frozen embedders via the difference-in-means method, treating each axis as a potential linear direction. To train the probes, we cluster 79,658 Czech oncology search queries with BERTopic and use these clusters to generate pairs of queries with contrastive risk levels via few-shot prompting. We evaluate the approach on 200 queries (90 real, 110 synthetic), each graded by two oncologists, against two open-weight LLMs and a frontier LLM. We find that urgency-based axes are recoverable as linear directions, and the probes are competitive 
    
[^6]: KuaFu：在十亿级规模上将长用户行为压缩为用户理解

    KuaFu: Compressing Long User Behavior into Understanding at Billion Scale

    [https://arxiv.org/abs/2609.31045](https://arxiv.org/abs/2609.31045)

    该论文提出KuaFu系统，在十亿用户规模上解决将长用户行为压缩为用户理解时序列过长与刷新吞吐量的双重工业瓶颈，并应对压缩过程可能引入的四类幻觉问题。

    

    对话式智能体、生成式推荐器和个性化广告都依赖一项核心能力：从原始行为中理解每一位用户。当前主流的工业实践是任务专用的：针对每个任务，从完整历史中提取相关子序列，并在其上训练专用模型。在生产环境中，这遇到了两个瓶颈。第一，即使经过过滤，单一任务的序列仍然极长：内容兴趣摘要需要读取每位用户的数百条内容，一旦序列化为提示文本就达数万个token。第二，用户画像需要例行刷新：每周覆盖十亿用户，总计约10万QPM，在固定的GPU预算下这设定了一个硬性的吞吐量下限。因此压缩是必不可少的，但截断或粗糙的压缩会悄然扭曲用户画像，引入四种幻觉类型（捏造、遗漏、日期错误归属、逻辑断裂），而由于无法评估压缩（原文在此处截断）……

    arXiv:2609.31045v1 Announce Type: cross  Abstract: Conversational agents, generative recommenders, and personalized advertising all rest on one capability: understanding each user from raw behavior. Prevailing industrial practice is task-specific: for each task, a relevant subsequence is extracted from the full history and a dedicated model trained on it. In production it hits two bottlenecks. First, even after filtering, a single-task sequence stays extremely long: content-interest summarization reads several hundred items per user, tens of thousands of tokens once serialized as prompt text. Second, profiles are refreshed routinely: a billion users weekly, roughly 100K QPM in aggregate, which under a fixed GPU budget sets a hard throughput floor. Compression is therefore mandatory, yet truncation or coarse compression can silently distort the profile, introducing four hallucination types (fabrication, omission, date misattribution, broken logic) that, with no way to evaluate the compr
    
[^7]: QReason：面向查询的解耦思维链实现高效段落重排序

    QReason: Query-Focused Decoupled Chain-of-Thought for Efficient Passage Reranking

    [https://arxiv.org/abs/2609.30904](https://arxiv.org/abs/2609.30904)

    QReason提出一种解耦框架，通过重写器仅生成一次面向排序的推理查询并在各滑动窗口中复用，避免重复的思维链推理，从而在保持复杂查询处理能力的同时大幅降低段落重排序的冗余与延迟。

    

    段落重排序在信息检索中发挥着至关重要的作用，它通过优化候选段落的排序来更好地反映其相关性。现有的采用思维链（CoT）推理的列表式LLM重排序器能够有效处理复杂查询，但由于滑动窗口策略会重复生成高度相似的思维链，导致大量冗余和高延迟问题。为解决这一问题，我们提出了QReason，一种将面向查询的推理与窗口特定的段落相关性评估相解耦的框架。具体而言，QReason引入了一个专门的重写器，它仅生成一次面向排序的推理查询，在捕捉查询核心意图的同时避免冗余推理，随后在所有窗口中将其与无推理的重排序器配合复用。该重写器通过两阶段过程进行训练：第一阶段使用监督微调，借助语义证据的相关段落引导来产生深度（摘要截断）。

    arXiv:2609.30904v1 Announce Type: new  Abstract: Passage reranking plays a crucial role in information retrieval by refining the ordering of candidate passages to better reflect relevance. Existing listwise LLM rerankers with Chain-of-Thought (CoT) reasoning can handle complex queries effectively, but they suffer from substantial redundancy and high latency due to sliding-window strategies, which repeatedly generate highly similar CoTs. To address this, we propose QReason, a decoupled framework that separates query-focused reasoning from window-specific passage relevance assessment. Specifically, QReason introduces a dedicated rewriter that generates a ranking-oriented reasoning query once, capturing the query's core intent while avoiding redundant reasoning, and then reuses it across all windows with a non-reasoning reranker. The rewriter is trained via a two-stage process that first uses supervised fine-tuning with relevant-passage guidance through semantic evidence to produce deeply
    
[^8]: RecToolBench：面向模糊用户意图的推荐专用工具编排基准测试

    RecToolBench: Benchmarking Recommendation-Specific Tool Orchestration under Fuzzy User Intent

    [https://arxiv.org/abs/2609.30717](https://arxiv.org/abs/2609.30717)

    该论文提出了RecToolBench，一个基于MCP协议、用于评估推荐智能体在模糊用户意图下进行工具编排能力的基准测试，包含超过1,200个可执行任务，覆盖单工具调用、并行调用、顺序工具链和混合编排等多种复杂工具使用模式。

    

    智能体化推荐系统的最新进展正在推动推荐系统从被动的过滤引擎转变为遵循指令的智能体，这些智能体利用外部工具来解析用户意图。然而，现有的基准测试通常假设用户意图是明确的、工具环境是简化的、或者函数调用是孤立的，这使得面向推荐场景的真实工具编排研究仍未得到充分探索。为了弥补这一空白，我们提出了RecToolBench，这是一个基于模型上下文协议（MCP）的基准测试，用于评估在使用工具的推荐智能体在模糊用户指令下的表现。RecToolBench包含超过1,200个可执行任务，覆盖三个推荐领域、13个MCP服务器和32个工具，涵盖单工具调用、并行工具调用、顺序工具链以及混合工具编排等多种模式。我们通过一个可扩展的“合成—模糊化—评判”流水线构建了RecToolBench，该流水线生成可执行的模糊推荐任务，并使用……（原文摘要在此处截断）评估智能体的执行轨迹。

    arXiv:2609.30717v1 Announce Type: new  Abstract: Recent advances in agentic recommender systems are shifting recommender systems from passive filtering engines to instruction-following agents that use external tools to resolve user intent. However, existing benchmarks often assume explicit user intent, simplified tool environments, or isolated function calls, leaving realistic tool orchestration for recommendation underexplored. To bridge this gap, we propose RecToolBench, a Model Context Protocol (MCP)-based benchmark for evaluating tool-using recommender agents under fuzzy user instructions. RecToolBench contains more than 1,200 executable tasks across three recommendation domains, 13 MCP servers, and 32 tools, spanning single-tool calls, parallel tool calls, sequential tool chains, and hybrid tool orchestration. We construct RecToolBench with a scalable synthesize--fuzzify--judge pipeline that generates executable fuzzy recommendation tasks, and evaluates agent trajectories using ru
    
[^9]: 用于未来状态控制的推荐世界模型

    Recommendation World Models for Future-State Control

    [https://arxiv.org/abs/2609.30711](https://arxiv.org/abs/2609.30711)

    提出UA-TWM效用锚定世界模型接口，使已训练的序列推荐排序器能够估计候选列表的未来后果并在效用约束下选择替代方案，从而同时提升推荐准确性与未来状态对齐。

    

    序列推荐优化的是对哪些物品进行排序，而每个展示的候选列表也会塑造后续的反馈和用户状态。我们研究一个已训练的排序器如何能够支持关于这些未来后果的决策。我们提出UA-TWM，一种以效用为锚点的世界模型接口，它构建邻近的列表动作，估计其与目标相关的后果，并在效用约束下选择替代方案。当没有合格的替代方案时，参考列表作为兜底选择。日志回放实例化将效用和目标增益估计与经校准的失败风险预测相结合；闭环实例化则使用一步状态-动作预测，并在观察到反馈后更新其决策。我们在MovieLens-25M和KuaiRand-Pure上评估了该接口在十二个序列骨干模型上的迁移能力，并在KuaiSim中进行了重复的目标导向交互评估。附加该接口提升了Recall@20、NDCG@20和未来状态对齐（原文摘要在此处被截断）。

    arXiv:2609.30711v1 Announce Type: new  Abstract: Sequential recommendation optimizes which items to rank, while each displayed slate also shapes subsequent feedback and user state. We study how a trained ranker can support decisions about these future consequences. We introduce UA-TWM, a utility-anchored world-model interface that constructs nearby slate actions, estimates their target-relevant consequences, and selects an alternative subject to utility constraints. The reference slate serves as a fallback when no alternative qualifies. A logged-replay instantiation combines utility and target-gain estimates with calibrated failure-risk prediction; a closed-loop instantiation uses one-step state-action prediction and updates its decisions after observed feedback. We evaluate transfer across twelve sequential backbones on MovieLens-25M and KuaiRand-Pure, and repeated target-directed interaction in KuaiSim. Attaching the interface improves Recall@20, NDCG@20, and future-state alignment f
    
[^10]: 组件基准测试：面向大规模推荐系统的分层模型性能剖析

    Component Benchmark: Hierarchical Model Profiling for Large-scale Recommendation Systems

    [https://arxiv.org/abs/2609.30656](https://arxiv.org/abs/2609.30656)

    本文提出了组件基准测试（CB）系统，以分层方式独立剖析大规模推荐系统中每个子模块的性能，并通过树状交互式可视化解决了现有工具无法将性能归因到具体子模块的问题。

    

    大规模推荐模型带来了独特且尚未被充分研究的性能剖析挑战。大多数推荐模型架构在结构上是异构的，混合了内存带宽受限的操作、小规模计算受限的稠密层、来自不规则类别特征的动态形状，以及低算术强度的操作。推荐模型随着建模工程师不断尝试各种组合而快速演进，而这些代码的编写往往缺乏对硬件执行特性的可见性。标准的性能剖析工具要么提供端到端的吞吐量，要么提供算子级别的追踪信息，但无法将性能归因到从业者所关注的子模块上。我们提出了组件基准测试，这是一种以分层方式独立刻画每个子模块性能的剖析系统，提供树状结构的交互式可视化，为机器学习从业者带来性能上的清晰洞察。其核心在于，CB 提供了一种简单而有效的（原文摘要在此处被截断）

    arXiv:2609.30656v1 Announce Type: new  Abstract: Large-scale recommendation models pose distinct, under-explored profiling challenges. Most recommendation model architectures are structurally heterogeneous, intermixing memory-bandwidth-bound operations, small compute-bound dense layers, dynamic shapes from jagged categorical features, and low-arithmetic-intensity operations. Recommendation models evolve rapidly as modeling engineers experiment with compositions, often written without visibility into hardware execution characteristics. Standard profiling tools offer either end-to-end throughput or operator-level traces, but cannot attribute performance to the submodules that practitioners reason about. We present Component Benchmark (CB), a profiling system that independently characterizes each submodule performance in a hierarchical manner, providing a tree-structured, interactive visualization that brings performance clarity to ML practitioners. At its core, CB provides a simple yet e
    
[^11]: 爱泼斯坦文件引擎：面向调查性新闻的智能体搜索

    Epstein Files Engine: Agentic Search for Investigative Journalism

    [https://arxiv.org/abs/2609.30611](https://arxiv.org/abs/2609.30611)

    《纽约时报》开发的“爱泼斯坦文件引擎”通过大语言模型将记者问题转化为SQL查询，在三百万页司法部文件中检索带引用的可验证答案，助力100余名记者完成至少20篇报道，其核心创新Diff重复匹配方法放大了新颖性信号，并证明了新闻编辑室AI智能体应作为源材料接口而非自主写作者来发挥最大价值。

    

    2026年1月30日，美国司法部发布了一组关于杰弗里·爱泼斯坦的多媒体资料合集，其中包括约三百万页PDF文档。我们介绍了“爱泼斯坦文件引擎”，这是《纽约时报》为调查这些文件而部署的人工智能智能体。该引擎将记者的问题转化为针对三个语料库的Google BigQuery SQL查询：爱泼斯坦相关文件发布、时报档案库以及外部的爱泼斯坦相关新闻标题。它使用大语言模型来规划查询，并返回带有丰富引用的答案，使记者能够验证并信赖这些结果。超过100名记者使用了该引擎，它为至少20篇已发表的报道做出了贡献。我们报告了记者们如何使用该引擎进行查询，并介绍了Diff——我们开发的文本与视觉重复匹配方法，该方法放大了新颖性信号，使引擎能够挖掘出真正的新信息。我们提出，新闻编辑室的智能体不应作为自主写作者，而应作为连接源材料的接口，这样才能最好地服务于新闻编辑室。

    arXiv:2609.30611v1 Announce Type: cross  Abstract: On Jan. 30, 2026, the U.S. Department of Justice released a mixed-media collection concerning Jeffrey Epstein, including about three million pages of PDFs. We describe the Epstein Files Engine, an A.I. agent The New York Times deployed to investigate the files. The Engine translated reporter questions into Google BigQuery SQL queries across three corpora: Epstein-related releases, the Times's archive and external, Epstein-related news headlines. It used an LLM to plan queries and returned citation-rich answers a reporter could verify and trust. More than 100 journalists used the Engine, and it contributed to at least 20 published stories. We report how reporters queried it and describe Diff, our text-and-visual duplicate matching method that amplified novelty signals and allowed the Engine to surface genuinely new information. We argue that newsroom agents serve newsrooms best not as autonomous writers, but as interfaces to source mate
    
[^12]: 面向动态多目标检索的嵌入子空间划分

    Embedding Subspace Partitioning for Dynamic Multi-Objective Retrieval

    [https://arxiv.org/abs/2609.30601](https://arxiv.org/abs/2609.30601)

    提出嵌入子空间划分（ESP）框架，将嵌入分解为任务感知的子空间，并用权重可在服务时调节的子空间相似度加权和替代单一内积，使检索器无需重新训练即可动态适应多目标优先级的变化，同时缓解多目标联合优化中的目标干扰问题。

    

    现代工业推荐系统必须在相互竞争的目标之间进行优化，平衡语义相关性与参与度、收入等业务指标。尽管双编码器凭借其高效性主导着大规模检索，但它们将这些异构信号压缩到单一的静态嵌入空间中。这种设计造成了根本性的局限：一旦训练完成，检索器在服务时若不重新训练就无法适应目标优先级的变化。此外，使用多目标损失进行联合优化常常会引起目标之间的相互干扰，导致次优的权衡。我们提出了嵌入子空间划分（Embedding Subspace Partitioning, ESP），这是一种检索框架，它将嵌入分解为任务感知的子空间，并用各子空间相似度的加权和取代单一的点积，其权重可在服务时进行调整。对于 Transformer 双编码器，ESP 使用模型原生的序列结束标记作为……（摘要在此处截断）

    arXiv:2609.30601v1 Announce Type: new  Abstract: Modern industrial recommender systems must optimize across competing objectives, balancing semantic relevance with business metrics such as engagement and revenue. While bi-encoders dominate large-scale retrieval due to their efficiency, they collapse these heterogeneous signals into a single static embedding space. This design creates a fundamental limitation: once trained, the retriever cannot adapt to shifting objective priorities at serving time without retraining. Moreover, joint optimization with multi-objective losses often induces interference between objectives, leading to suboptimal trade-offs. We propose Embedding Subspace Partitioning (ESP), a retrieval framework that decomposes the embedding into task-aware subspaces and replaces the single dot product with a weighted sum of per-subspace similarities, whose weights are tunable at serving time. For Transformer bi-encoders, ESP uses the model's native end-of-sequence token as 
    
[^13]: T-RoPE：面向序列推荐的时间感知旋转位置编码

    T-RoPE: Time-Aware Rotary Position Embedding for Sequential Recommendation

    [https://arxiv.org/abs/2609.30576](https://arxiv.org/abs/2609.30576)

    提出T-RoPE，一种时间感知的旋转位置编码，通过基于时间戳的角度、可学习时间系数和多尺度频率等机制打破标准RoPE的时间平移不变性，使序列生成式推荐模型能够捕捉时间间隔、行为周期与季节性等关键时间信息。

    

    大规模推荐系统日益采用大语言模型背后的序列生成式方案，将Transformer引入推荐领域，同时也沿用了为文本设计的组件，包括旋转位置编码。在语言模型中，RoPE对词元索引进行编码以实现相对位置推理，但在推荐场景中，交互索引仅记录事件顺序，无法反映流逝的时间、跨尺度的行为周期或日历相位。我们重新审视这一设计选择，提出T-RoPE，一种用于序列生成式推荐的时间感知RoPE，它以基于时间戳的角度、可学习的时间系数、多尺度频率库、偏移查询对齐以及非平稳的键旋转，替代仅基于索引的旋转。我们证明标准RoPE即使作用于时间戳，仍保持时间平移不变性，无法区分季节性上下文；而T-RoPE在保留RoPE的……

    arXiv:2609.30576v1 Announce Type: new  Abstract: Large-scale recommenders increasingly adopt the sequential generative recipe behind large language models, bringing the Transformer into recommendation along with design choices made for text, including Rotary Position Embedding (RoPE). In language models, RoPE encodes token indices for relative position reasoning, but in recommendation, an interaction index records only event order, saying nothing about elapsed time, behavioral cycles across scales, or calendar phase. We revisit this choice and propose T-RoPE, a time-aware RoPE for sequential generative recommendation that replaces index-only rotation with timestamp-based angles, learnable temporal coefficients, multiscale frequency banks, shifted query alignment, and non-stationary key rotation. We prove that standard RoPE, even on timestamps, remains time-translation invariant and cannot distinguish seasonal contexts, and that T-RoPE breaks this invariance while preserving the RoPE in
    
[^14]: 近在咫尺却非最爱：面向纯内容搜索与推荐的共享策展人反馈基础设施

    Nearest but Not Dearest: Shared Curator-Feedback Infrastructure for Content-Only Search and Recommendation

    [https://arxiv.org/abs/2609.30568](https://arxiv.org/abs/2609.30568)

    该论文提出将策展人反馈中的“声音失败”与“上下文失败”进行分解，并以此构建共享反馈基础设施，用于改进缺乏用户行为信号的纯内容搜索与推荐系统。

    

    arXiv:2609.30568v1 公告类型： 新论文 摘要： 一个已部署的B2B音乐发现平台同时提供查询驱动的搜索（文本提示、氛围标签）和种子驱动的推荐（种子曲目与艺术家电台），二者运行于同一个授权曲库、同一个LAION-CLAP音文联合嵌入空间、同一个候选生成过滤器和一个排序头之上——并且两条路径均不使用终端听众的行为信号。在这种纯内容场景中，策展人的判断是唯一可用的主要反馈信号，而离线余弦相似度对其预测效果很差：38%的余弦最近邻被策展人拒绝。这些拒绝揭示出一个清晰的划分：大多数（55%）是编码器可以解决的声音失败（风格、节奏、情绪不匹配），另有相当一部分（37%）是与波形本身正交的上下文失败（语言错误、节日内容、宗教内容、版权及歌词标记）。我们将这种“声音与上下文”的失败分解部署为反馈基础设施，将每种失败模式路由至相应的改进路径（论文摘要在此处截断）。

    arXiv:2609.30568v1 Announce Type: new  Abstract: A deployed B2B music-discovery platform serves both query-driven search (text prompts, vibe tags) and seed-driven recommendation (seed-track and artist stations) over one licensed catalog, one LAION-CLAP joint audio-text embedding space, one candidate-generation filter, and one ranking head -- and neither path consumes end-listener behavioral signal. In this content-only regime, curator judgment is the principal feedback signal available, and offline cosine similarity predicts it poorly: 38% of cosine-nearest neighbors are rejected by curators. The rejections reveal a clean partition: a majority (55%) are sound failures the encoder could address (style, tempo, mood mismatch), and a substantial minority (37%) are context failures orthogonal to the waveform (wrong language, holiday content, devotional content, rights and lyric flags). We deploy this sound-vs-context decomposition as feedback infrastructure, routing each failure mode to the
    
[^15]: REALMS：一种面向高维嵌套用户画像的实时精确受众规模估算AI助手对话系统

    REALMS: An AI-Assistant Conversational System for Real-Time Exact Audience Sizing over High-Dimensional Nested Profiles

    [https://arxiv.org/abs/2609.30547](https://arxiv.org/abs/2609.30547)

    REALMS是一个部署在生产环境的对话式系统，结合嵌入向量检索与大语言模型NL2SQL技术，让营销人员用自然语言在数秒内对数百万级、上千属性的高维用户画像完成精确受众规模估算。

    

    受众规模估算是数字营销中的关键组成部分，它能够实现精确的资源分配、营销活动规划和效果优化。传统方法（如骨架受众、抽样或预测建模）存在显著的延迟、估算误差，以及在高维画像数据上可扩展性差的问题。我们提出了REALMS（基于大语言模型多属性搜索的实时精确受众规模估算系统），这是一个部署在企业客户数据平台生产环境中的精确受众规模估算对话系统。REALMS使营销人员能够使用自然语言查询拥有数百万用户画像和数千个属性的海量画像库，并在几秒钟内获得精确的计数结果。该系统引入了三个关键组件：(1) 一种基于嵌入向量搜索的分类属性检索机制，无需手动配置即可动态识别相关的模式属性；(2) 一种由大语言模型驱动的自然语言转SQL（NL2SQL）组件……

    arXiv:2609.30547v1 Announce Type: new  Abstract: Audience sizing is a critical component of digital marketing. It enables precise resource allocation, campaign planning, and performance optimization. Traditional approaches using skeleton audiences, sampling, or predictive modeling suffer from significant delays, estimation errors, and poor scalability over high-dimensional profile data. We present REALMS (Real-time Exact Audience sizing via LLM-based Multi-attribute Search), a conversational system for exact audience sizing deployed in production on an enterprise customer data platform. REALMS enables marketers to query massive profile stores with millions of profiles and thousands of attributes using natural language and receive precise counts in seconds. The system introduces three key components: (1) a categorical attribute retrieval mechanism using embedding-based vector search to dynamically identify relevant schema attributes without manual configuration; (2) an LLM-powered NL2SQ
    
[^16]: 生产规模的AutoResearch：失效模式与多智能体框架

    AutoResearch at Production Scale: Failure Modes and a Multi-Agent Framework

    [https://arxiv.org/abs/2609.30541](https://arxiv.org/abs/2609.30541)

    该论文将AutoResearch范式应用于生产规模的推荐系统嵌入优化，在220多次实验中识别出基础设施脆弱、智能体记忆衰退、搜索方向停滞、迭代成本不对称和指标固化五种失效模式，并据此提出多智能体框架加以解决。

    

    arXiv:2609.30541v1 公告类型：cross 摘要：为生产级推荐流水线优化嵌入系统需要进行系统性探索，而在大规模场景下，这种探索会消耗不成比例的工程投入。我们应用 Andrej Karpathy 的 AutoResearch 范式——即由大型语言模型迭代地修改训练脚本，并保留那些能够提升留出标量指标的修改——来自动化这一探索过程。我们报告了在生产规模下运行该范式十二周的经验：其中每次迭代需要消耗数小时的多GPU算力，评估涉及相互竞争的多项准则，且整个实验活动跨越数周、涉及大量训练任务。在为一个图书推荐流水线独立开发的两套表征学习系统上，我们运行了220多次实验，观察到在原始设定中不存在的五种反复出现的失效模式：基础设施脆弱性、智能体记忆衰退、搜索方向停滞、迭代成本不对称以及指标固化。我们贡献了一个三原则……（摘要原文在此处截断）

    arXiv:2609.30541v1 Announce Type: cross  Abstract: Optimizing embedding systems for production recommendation pipelines demands systematic exploration that consumes disproportionate engineering effort at scale. We apply Andrej Karpathy's AutoResearch paradigm -- a large language model that iteratively edits a training script and retains modifications that improve a held-out scalar metric -- to automate this exploration. We report on twelve weeks of running this paradigm at production scale, where iterations consume hours of multi-GPU compute, evaluation involves competing criteria, and campaigns span weeks across many training jobs. Across two independently developed representation-learning systems for a book recommendation pipeline, we ran 220+ experiments and observed five recurring failure modes absent from the original setting: infrastructure fragility, agent memory decay, search-direction stagnation, iteration-cost asymmetry, and metric fixation. We contribute a three-principle sc
    
[^17]: 基于检索的开放式评估在何处失效？从长篇医学答案事实性验证中自动归纳错误分类体系

    Where Does Retrieval-Based Open-Ended Evaluation Fail? Automatic Taxonomy Induction from Long-Form Medical Answer Factuality Verification

    [https://arxiv.org/abs/2609.30467](https://arxiv.org/abs/2609.30467)

    该论文针对基于检索的开放式医学事实性验证，自动归纳出两套错误分类体系，将失败分解为五个质量维度上的检索阶段错误与六个连续步骤中的验证器推理错误，并利用LLM-as-Judge流程实现大规模自动化错误标注与分类，无需黄金答案或黄金证据。

    

    检索式事实性评估——即将大语言模型生成的论断与权威医学语料库中的证据进行核对验证——已成为高风险临床环境中可扩展幻觉检测的主流范式。尽管可靠且透明的医学事实验证需求迫切，大多数系统仍采用F1等聚合指标衡量性能，这类指标掩盖了失败发生在何处以及为何发生。现有的RAG诊断方法需要黄金答案或人工标注的黄金证据，而这两者在该场景中均不存在。我们基于对开放式MedExpert数据集以及3个封闭式数据集的案例研究，引入了两套全面的分类体系：将失败分解为沿五个质量维度划分的检索阶段错误，以及分为六个连续步骤的验证器推理错误。我们改造了一种基于LLM-as-Judge的自动模式归纳流程，用以大规模标注证据质量并对验证器推理错误进行分类……

    arXiv:2609.30467v1 Announce Type: new  Abstract: Retrieval-based factuality evaluation, where LLM-generated claims are verified against evidence from authoritative medical corpora, has become the dominant paradigm for scalable hallucination detection in high-stakes clinical settings. Despite the urgency of reliable and transparent medical fact verification, most systems measure performance with aggregate metrics like F1, which obscure where and why failures occur. Existing RAG diagnostics require gold answers or annotated gold evidence, neither of which exists in this regime. We introduce two comprehensive taxonomies, grounded in a case study on the open-ended MedExpert dataset and 3 closed-ended datasets, decomposing failures into retrieval-stage errors along five quality dimensions, and verifier-reasoning errors into six consecutive steps. We adapt an automatic pattern induction pipeline using LLM-as-Judge to label evidence quality and classify verifier reasoning errors at scale, and
    
[^18]: 在Spotify引导对话式推荐智能体：合成数据生成与自我改进循环

    Bootstrapping Conversational Recommendation Agents At Spotify: Synthetic Data Generation and Self-Improvement Loops

    [https://arxiv.org/abs/2609.30297](https://arxiv.org/abs/2609.30297)

    Spotify提出了一条多轮合成数据生成流水线与自我改进循环，通过基于方差的对比优化和编码智能体的迭代修复，在冷启动场景下自动优化对话式推荐智能体的规划与工具调用能力，使质量提升8%。

    

    对话式推荐智能体是内容发现的新范式，使用户能够通过自然语言表达复杂意图（例如：“推荐一些我没听过的意大利独立音乐人”）。构建此类智能体的核心挑战在于优化智能体规划——即决定如何选择、排序和调用工具——尤其是在真实用户交互尚不可用的冷启动场景中。为应对这一挑战，我们提出了一条多轮合成数据生成流水线和一种自我改进循环。合成数据流水线将单轮提示转化为逼真的多轮对话，从而支持上线前的系统性评估。自我改进循环将基于方差的对比优化与通过编码智能体进行的迭代改进相结合，能够自动识别并修复规划和工具使用中的错误。我们的方法在高度优化的基线之上将质量提升了+8%。

    arXiv:2609.30297v1 Announce Type: cross  Abstract: Conversational recommendation agents are a new paradigm for content discovery, enabling users to express complex intents through natural language (e.g., "recommend Italian indie artists I haven't heard before"). A central challenge in building such agents is optimizing agent planning -- deciding how to select, sequence, and invoke tools -- particularly in cold-start settings where real user interactions are not yet available. We introduce a pipeline for multi-turn synthetic data generation and a self-improvement loop to address this challenge. The synthetic data pipeline transforms single-turn prompts into realistic multi-turn conversations, enabling systematic evaluation before launch. The self-improvement loop combines variance-based contrastive optimization with iterative refinement through a coding agent, automatically identifying and fixing planning and tool-use errors. Our approach improves quality by +8% on top of a highly optim
    
[^19]: SignTrace：描述一个手势，找到对应的词

    SignTrace: Describe a Sign, Find the Word

    [https://arxiv.org/abs/2609.30295](https://arxiv.org/abs/2609.30295)

    SignTrace 利用大语言模型增强的中国手语词典，结合动作提取、七路检索与候选重排序，让学习者仅凭日常语言描述的手部动作即可反向查找到对应手语词条及其含义，在 500 条查询的基准上达到 94.0% 的 Hit@1。

    

    当学习者记得一个陌生手语的动作，却不知道其含义或正式的特征编码时，识别该手语十分困难。SignTrace 通过自然语言方式访问中国手语词典，解决了这一长期存在的反向查询问题。该系统集成了基于大语言模型的词典增强、动作提取、词典风格改写、七路检索以及候选重排序，覆盖 6,699 个词条。系统已部署进行用户试用，并收到了积极的非正式反馈。在一个由词典衍生、包含 500 条动作描述查询的基准测试上，系统取得了 94.0% 的 Hit@1、97.4% 的 Hit@9 以及 0.9540 的平均倒数排名。重排序将 Hit@1 从 71.8% 提升至 94.0%，组件分析显示了增强词条描述所作的贡献。在六个并发查询的情况下，查询处理的中位时间为 13.37 秒。通过将日常动作描述与已记录的手语动作及其含义关联起来……

    arXiv:2609.30295v1 Announce Type: cross  Abstract: Identifying an unfamiliar sign is difficult when a learner remembers its movement but does not know its meaning or formal feature codes. SignTrace addresses this longstanding reverse-lookup problem through natural-language access to a Chinese sign-language dictionary. The system integrates LLM-based dictionary enrichment, action extraction, dictionary-style rewriting, seven-channel retrieval, and candidate reranking over 6,699 entries. It has been deployed for user trials and has received positive informal feedback. Evaluation on a dictionary-derived benchmark of 500 movement-description queries yields 94.0% Hit@1, 97.4% Hit@9, and a mean reciprocal rank of 0.9540. Reranking increases Hit@1 from 71.8% to 94.0%, while component analyses show the contribution of enriched entry descriptions. Median query-processing time is 13.37 seconds with six concurrent queries. By connecting everyday movement descriptions to documented signs and meani
    
[^20]: MM-ContextFold：面向多模态智能体检索的上下文折叠

    MM-ContextFold: Context Folding for Multimodal Agentic Retrieval

    [https://arxiv.org/abs/2609.23121](https://arxiv.org/abs/2609.23121)

    提出了无需训练的 MM-ContextFold 框架，基于对约一万条轨迹的实证发现——当视觉信息被提取并文本化后原始图像变得冗余——通过“折叠”冗余视觉内容来解决多模态智能体检索中的上下文爆炸问题。

    

    多模态智能体检索要求智能体通过迭代调用外部工具来解决复杂的信息搜索任务。诸如 ReAct 等典型框架将原始多模态输入和不断累积的交互历史保持在单一且持续增长的上下文中，从而导致上下文爆炸问题。虽然现有方法通过压缩冗余文本来缓解这一问题，但针对如何有效管理 token 密集的视觉内容的策略在很大程度上仍未被充分探索。为了填补这一空白，我们首先对约 10,000 条轨迹进行了系统的实证研究。结果表明，随着视觉线索通过外部工具被逐步提取并文本化到上下文中，原始图像变得越来越冗余。持续保留图像与更高的输出熵相关，甚至会降低任务准确率。受这些发现的启发，我们提出了 MM-ContextFold，这是一个无需训练的框架，其……（摘要在此处被截断）

    arXiv:2609.23121v1 Announce Type: cross  Abstract: Multimodal Agentic Retrieval (MAR) requires agents to solve complex information-seeking tasks by iteratively invoking external tools. Typical frameworks such as ReAct maintain raw multimodal inputs and the accumulating interaction history in a single, ever-growing context, leading to the context explosion problem. While existing methods alleviate this issue by compressing redundant text, effective strategies for managing token-intensive visual content remain largely underexplored. To address this gap, we first conduct a systematic empirical study of approximately 10,000 trajectories. The results show that as visual cues are progressively extracted through external tools and textualized into the context, raw images become increasingly redundant. Continued image retention is associated with higher output entropy and can even degrade task accuracy. Motivated by these findings, we propose MM-ContextFold, a training-free framework that load
    
[^21]: 特征叠加中线性能及性的高概率保证

    High-probability guarantees for linear accessibility in feature superposition

    [https://arxiv.org/abs/2609.09556](https://arxiv.org/abs/2609.09556)

    该论文将特征叠加中的线性能及性建模为压缩感知问题，证明了充分维度只需线性规模（而非此前最坏情况的二次方限制）即可高概率地恢复同时激活的特征，从而量化了线性表示假设的几何约束并为稀疏自编码器和神经可解释性评估提供了理论框架。

    

    神经网络可以利用特征叠加来编码比维度数量更多的概念，但特征间的交叉干扰限制了同时激活特征的线性能及性。通过将线性能及性建模为一个压缩感知问题，我们在次高斯噪声下针对固定支撑集推导出高概率界，证明了充分维度以线性方式扩展（d=O_ε(k log m)），而非此前最坏情况下的二次方限制。随后，我们通过高斯尾近似在各系统参数下验证了这些界。这些结果量化了线性表示假设的几何约束，为评估稀疏自编码器、组合泛化和神经网络可解释性提供了一个框架。

    arXiv:2609.09556v1 Announce Type: cross  Abstract: Neural networks can leverage feature superposition to encode more concepts than dimensions, but cross-feature interference constrains the linear accessibility of simultaneously active features. By framing linear accessibility as a compressed sensing problem, we derive high-probability bounds for fixed supports under subgaussian noise, proving the sufficient dimension scales linearly ($d=O_{\varepsilon}(k \log m)$) rather than prior worst-case quadratic limits. We then validate these bounds across system parameters through Gaussian-tail approximations. These results quantify the geometric constraints of the linear representation hypothesis, providing a framework for evaluating sparse autoencoders, compositional generalization, and neural interpretability.
    
[^22]: 将智能体搜索引入地球观测数据发现

    Bringing Agentic Search to Earth Observation Data Discovery

    [https://arxiv.org/abs/2607.02387](https://arxiv.org/abs/2607.02387)

    该论文提出了一个基于NASA地球观测知识图谱的智能体搜索框架用于地球科学数据发现，构建了包含47k查询-数据集对的开放基准NASA-EO-Bench，并通过微调神经评分器与BM25分数融合，将R@10和MRR提升至余弦基线的5倍以上。

    

    NASA及其数据中心拥有数千个地球科学数据集，以及Worldview、Giovanni、科学发现引擎和Harmony等工具。即使是领域专家，要找到合适的数据或工具也十分困难。我们提出了一个面向地球科学数据发现的智能体搜索框架，该框架接收自然语言研究查询并返回匹配的数据集和工具。我们证明，在大语言模型时代，知识图谱（KG）的潜在价值可以通过智能体搜索得到显著放大。我们从NASA地球观测知识图谱（NASA EO-KG）中构建了NASA-EO-Bench，这是一个包含47k查询-数据集对（其中21k为基于任务的查询）的开放基准。在NASA-EO-Bench上微调的神经评分器优于余弦相似度和BM25基线。进一步通过分数融合将其与BM25结合，使Recall@10（R@10）和MRR均达到未适配余弦基线的5倍以上。在此监督流水线之上，零样本重排序阶段进一步提升了MRR……

    arXiv:2607.02387v2 Announce Type: replace  Abstract: NASA and its data centers hold thousands of geoscience datasets and tools like Worldview, Giovanni, the Science Discovery Engine, and Harmony. Finding the right one is hard even for domain experts. We present an agentic search framework for geoscience data discovery that takes a natural-language research query and returns matching datasets and tools. We demonstrate that, in the era of large language models, the latent value of knowledge graphs (KGs) can be substantially amplified through agentic search. From the NASA Earth Observation Knowledge Graph (NASA EO-KG) we derive NASA-EO-Bench, an open benchmark of 47k query-dataset pairs (21k task-based queries). A neural scorer fine-tuned on NASA-EO-Bench beats cosine and BM25 baselines. Further combining it with BM25 via score fusion raises both Recall@10 (R@10) and MRR to over 5x the unadapted cosine baseline. On top of this supervised pipeline, a zero-shot reranking stage lifts MRR by 
    
[^23]: FlyAOC：评估果蝇科学知识库的智能体本体策展

    FlyAOC: Evaluating Agentic Ontology Curation of Drosophila Scientific Knowledge Bases

    [https://arxiv.org/abs/2602.09163](https://arxiv.org/abs/2602.09163)

    FlyAOC是一个评估AI智能体从科学文献中进行端到端本体策展的基准，要求智能体在16,898篇果蝇论文中检索证据，并恢复策展人级别的结构化基因标注，涵盖功能术语、表达模式和历史同义词。

    

    科学知识库通过将原始文献中的研究发现策展为结构化、可查询的格式，从而加速科学发现，服务于人类研究者和新兴的AI系统。维护这些资源需要专家策展人检索论文、整合跨文档证据，并生成基于本体的标注。现有基准通常只评估孤立的子任务（如命名实体识别或关系抽取），因此无法捕捉这一端到端的工作流程。我们提出FlyAOC，用于评估AI智能体在科学文献上执行端到端智能体本体策展的能力。给定一个基因符号、一段简洁的FlyBase基因描述、一个包含16,898篇论文语料库的访问权限以及本体资源，智能体必须检索证据并尽可能多地恢复与策展人相关的结构化标注。输出涵盖标准化的功能术语、表达模式，以及连接数十年命名历史的历史同义词。

    arXiv:2602.09163v2 Announce Type: replace  Abstract: Scientific knowledge bases accelerate discovery by curating findings from primary literature into structured, queryable formats for both human researchers and emerging AI systems. Maintaining these resources requires expert curators to search papers, reconcile evidence across documents, and produce ontology-grounded annotations. Existing benchmarks usually evaluate isolated subtasks, such as named entity recognition or relation extraction, and therefore do not capture this end-to-end workflow. We present FlyAOC to evaluate AI agents on end-to-end agentic ontology curation from scientific literature. Given a gene symbol, a concise FlyBase gene description, access to a 16,898-paper corpus, and ontology resources, agents must search for evidence and recover as many curator-relevant structured annotations as possible. Outputs span standardized function terms, expression patterns, and historical synonyms linking decades of nomenclature. T
    
[^24]: 关于Lee度量下的函数校正码

    On Function-Correcting Codes in the Lee Metric

    [https://arxiv.org/abs/2507.17654](https://arxiv.org/abs/2507.17654)

    本文将函数校正码的研究扩展到Lee度量下任意整数模环 Z_m（m≥2）上，通过引入不规则Lee距离码并刻画其最短可能长度，给出了最优冗余度的上下界。

    

    函数校正码是一种编码框架，旨在最小化冗余的同时，确保即使在存在错误的情况下，编码数据的特定函数或计算也能被可靠地恢复。度量的选择在这类码的设计中至关重要，因为它决定了哪些计算需要被保护，以及错误如何被度量和纠正。Liu和Liu [6] 之前的工作使用齐次度量研究了 Z_{2^l}（l≥2）上的函数校正码，该度量在 Z_4 上与Lee度量一致。在本文中，我们将这一研究扩展到Lee度量下 Z_m 上的码（m为任意正整数且 m≥2），并旨在确定其最优冗余度。为实现这一目标，我们引入了不规则Lee距离码，并通过刻画此类码的最短可能长度，推导出最优冗余度的上界和下界。这些一般性的界随后被简化……

    arXiv:2507.17654v4 Announce Type: replace-cross  Abstract: Function-correcting codes are a coding framework designed to minimize redundancy while ensuring that specific functions or computations of encoded data can be reliably recovered, even in the presence of errors. The choice of metric is crucial in designing such codes, as it determines which computations must be protected and how errors are measured and corrected. Previous work by Liu and Liu [6] studied function-correcting codes over $\mathbb{Z}_{2^l},\ l\geq 2$ using the homogeneous metric, which coincides with the Lee metric over $\mathbb{Z}_4$. In this paper, we extend the study to codes over $\mathbb{Z}_m,$ for any positive integer $m\geq 2$ under the Lee metric and aim to determine their optimal redundancy. To achieve this, we introduce irregular Lee distance codes and derive upper and lower bounds on the optimal redundancy by characterizing the shortest possible length of such codes. These general bounds are then simplifie
    

