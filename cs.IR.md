# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Glyph: A Multi-Strategy Agentic System for Column Description and Sensitivity-Ontology Tagging of Enterprise Data Catalogs](https://arxiv.org/abs/2609.10430) | 本文提出Glyph生产级系统，将列描述生成与敏感性本体标注建模为协作式LLM智能体，通过基于管道源代码的主动检索增强生成和三种并行标注策略，解决企业数据目录的文档欠账问题。 |
| [^2] | [GANDR: Claim Auditing for Verifiable Legal Answer Generation](https://arxiv.org/abs/2609.10293) | GANDR是一个双智能体系统，由起草器生成结构化法律答案、批评者逐条对照引用来源审计论断，并配合要求每条引用必须命中检索结果的严格正确性标准，从而实现可逐条验证的法律答案生成。 |
| [^3] | [LiteRAG: Cost-Efficient Graph-Based Retrieval-Augmented Generation](https://arxiv.org/abs/2609.10239) | LiteRAG通过查询条件算法探索和推理链上下文构建取代昂贵的检索时LLM控制，在多跳问答质量上达到最优，同时将查询延迟降低100倍以上、成本降低99%以上、token使用量减少约14倍。 |
| [^4] | [The Answer Path and the Grounding Instruction in LLM Question Answering over Knowledge Graphs](https://arxiv.org/abs/2609.10237) | 本研究在六个大语言模型和两个知识图谱问答基准上系统变化四种提示设计选择，发现只有将答案路径纳入提示和接地指令的设计能显著影响答案准确率（非路径三元组可被无关内容替换而不影响效果，检索预算应全部用于保证召回率），而三元组的书写语法与排列顺序几乎没有作用。 |
| [^5] | [From Retrieval to Weights: Parametric Individualization of Small Language Models with Individual Text Corpora](https://arxiv.org/abs/2609.10155) | 该研究通过DoRA微调将515名参与者的个人搜索历史写入小语言模型权重，证明适配器能显著编码个人语料（个体化效应dz=1.27），但在通用知识测试中模型获得的是知识而非与个体的对齐。 |
| [^6] | [Guaranteeing Faithful Evidence Extraction in Speculative Retrieval-Augmented Generation](https://arxiv.org/abs/2609.10046) | 该论文提出约束混合解码（CHyD），将原本用于加速推理的投机解码架构重新用于强制执行硬解码约束，从而在投机式检索增强生成中保证抽取的证据逐忠实于检索到的上下文，满足安全关键领域对答案精确匹配的要求。 |
| [^7] | [Purchase Advice and Observable Buyer Responses in Real AI Conversations](https://arxiv.org/abs/2609.09878) | 该研究通过审计317个真实AI助手对话记录，分析了生成式AI在购买决策中的说服作用，发现77.6%的事件中助手提供购买建议，但只有26.9%的事件中能观察到用户后续购买反应。 |
| [^8] | [When Does Low-Bit Quantization Preserve the Decisions of Vector Search?](https://arxiv.org/abs/2609.09854) | 该论文在向量搜索算法实际使用的比较层面分析量化误差，给出了比较翻转概率的无分布分解、协方差感知的尾部界以及Vamana邻居选择的确定性耦合定理，从而解释了低比特量化何时能保留精确搜索的决策。 |
| [^9] | [Should I Be Polite to My LLM Relevance Judge? Tone as a Severity Operating-Point Shift](https://arxiv.org/abs/2609.09703) | 该研究发现提示语气对大语言模型相关性评判的影响，主要通过移动评判器的严重程度工作点（整体评分宽容度）而非提升判断质量实现，且效果高度依赖于具体模型。 |
| [^10] | [High-probability guarantees for linear accessibility in feature superposition](https://arxiv.org/abs/2609.09556) | 该论文将特征叠加中的线性能及性建模为压缩感知问题，证明了充分维度只需线性规模（而非此前最坏情况的二次方限制）即可高概率地恢复同时激活的特征，从而量化了线性表示假设的几何约束并为稀疏自编码器和神经可解释性评估提供了理论框架。 |
| [^11] | [Extracting Semantics from Cattle Reporting Categories for Data Interoperability and Findability](https://arxiv.org/abs/2609.09381) | 本研究采用自下而上的方法从牛群报告类别中提取语义，以解决牲畜数据缺乏元数据和标准的问题，从而提高数据的互操作性和可发现性。 |
| [^12] | [Better Together: Complementary Query Rewriting Under a Strong RAG Baseline](https://arxiv.org/abs/2609.05637) | 在强RAG检索基线下，单一查询改写策略收效有限，但联合多种互补的改写方法可大幅提升检索性能，企业数据上HIT@10提升12.5个百分点。 |
| [^13] | [DexterSQL: Deep Schema Exploration and Rule-based Correction for Text-to-SQL Generation](https://arxiv.org/abs/2608.11889) | DexterSQL通过深度模式探索、数据库无关规则挖掘和规则驱动修正三个创新组件，解决了非微调文本到SQL生成中模式信息粗糙、错误重复出现和条件处理不当的问题。 |
| [^14] | [LoopMemGR: From Behavior Logs to Evolving Memory for Generative Recommendation](https://arxiv.org/abs/2607.27647) | 本文提出LoopMemGR，一个闭环推荐经验记忆框架，在传统行为日志之外额外维护系统自身的推荐决策与反馈记忆，解决了生成式推荐中“只记用户行为、不记自身推荐”的不对称记忆问题，使偏好验证信号、反向证据和探索信息能够跨请求复用。 |
| [^15] | [On the Capacity of Distinguishable Synthetic Identity Generation under Face Verification](https://arxiv.org/abs/2604.10641) | 该论文首次在人脸验证框架下定义了合成身份生成的有限维容量这一理论概念，证明了确定性视角不变流水线下该容量等于可实现嵌入集上的球面码基数，并为随机身份条件嵌入分布推导了容量下界与中心分离条件。 |
| [^16] | [Query Brand Entity Linking in E-Commerce Search](https://arxiv.org/abs/2502.01555) | 该论文提出两种互补的品牌实体链接方法——级联流水线和极端多分类单阶段方法，在11种语言评估和在线实验中显著提升电商搜索的品牌召回率并保持高精确率，带来用户参与度的可衡量提升。 |

# 详细

[^1]: Glyph：一个用于企业数据目录列描述生成与敏感性本体标注的多策略智能体系统

    Glyph: A Multi-Strategy Agentic System for Column Description and Sensitivity-Ontology Tagging of Enterprise Data Catalogs

    [https://arxiv.org/abs/2609.10430](https://arxiv.org/abs/2609.10430)

    本文提出Glyph生产级系统，将列描述生成与敏感性本体标注建模为协作式LLM智能体，通过基于管道源代码的主动检索增强生成和三种并行标注策略，解决企业数据目录的文档欠账问题。

    

    企业数据湖中表的积累速度超过了人工管理员对其进行文档化和分类的速度，导致许多列缺失描述信息且未被分配治理标签。这种文档欠账削弱了数据发现、访问控制和法规合规能力。我们提出了Glyph，一个生产级系统，它将两个相互耦合的问题——列描述生成和用于数据分类的列类型标注——建模为由有状态图编排的协作式LLM智能体。描述器将生成过程建立在产生每一列的管道源代码之上，通过推理-行动工具循环（主动检索增强生成）按需从企业GitHub中检索相关代码。标注器通过并行运行三种互补策略（基于描述的标注器、基于业务线正则表达式的标注器，以及由微调对比编码器支持的元数据标注器），从受治理的包含275个叶子节点的数据分类本体中为列分配标签……

    arXiv:2609.10430v1 Announce Type: cross  Abstract: Enterprise data lakes accumulate tables faster than human stewards can document or classify them, leaving columns with missing descriptions and unassigned governance labels. This documentation debt undermines data discovery, access control, and regulatory compliance. We present Glyph, a production system that frames two coupled problems, column description generation and column type annotation for data classification, as cooperating LLM agents orchestrated as stateful graphs. The Descriptor grounds generation in the pipeline source code that produces each column, retrieved on demand from an enterprise GitHub via a reasoning--acting tool loop (active Retrieval-Augmented Generation). The Tagger assigns labels from a governed 275-leaf Data Classification Ontology by running three complementary strategies in parallel (a description tagger, a line-of-business regex tagger, and a metadata tagger backed by a fine-tuned contrastive encoder ove
    
[^2]: GANDR：面向可验证法律答案生成的论断审计

    GANDR: Claim Auditing for Verifiable Legal Answer Generation

    [https://arxiv.org/abs/2609.10293](https://arxiv.org/abs/2609.10293)

    GANDR是一个双智能体系统，由起草器生成结构化法律答案、批评者逐条对照引用来源审计论断，并配合要求每条引用必须命中检索结果的严格正确性标准，从而实现可逐条验证的法律答案生成。

    

    在法律实践等高风险领域，语言模型生成的答案只有在读者能够对照系统所引用的来源逐条验证每个论断时才有用。当前的有据生成流水线将答案作为一个整体进行评分，因此一个正确的结论可能建立在捏造的或匹配松散的引用之上，却仍能获得高分。弥合这一差距既需要一个为逐条论断验证而构建的系统，也需要一种能够衡量它的评估方法。我们提出了GANDR（Grounded ANswer DRafter，有据答案起草器），这是一个双智能体系统：起草器以结构化的法律推理格式撰写答案，而一个独立的批评者——拥有与人类验证者相同的视角——针对所引用的来源审计每个论断，并在每一轮输出逐条论断的审计轨迹。我们为其配备了一项严格的正确性标准，要求每条引用都能对应到检索器返回的某段文本。在一个包含185个条目的法律基准上，所有六个系统共享同一个骨干模型和一个检索源……（摘要在此处被截断）

    arXiv:2609.10293v1 Announce Type: new  Abstract: In high-stakes domains such as legal practice, a language-model answer is only useful to the extent that a reader can verify each claim against the source the system cites. Current grounded-generation pipelines score the answer as a whole, so a correct conclusion can rest on fabricated or loosely matched citations and still score well. Closing this gap requires both a system built for per-claim verification and an evaluation that measures it. We introduce GANDR (Grounded ANswer DRafter), a two-agent system in which a Drafter writes an answer in a structured legal-reasoning format and a separate Critic, with the same view as a human verifier, audits each claim against its cited source and emits a per-claim audit trace on every round. We pair it with a strict correctness criterion requiring every citation to resolve to a passage the retriever returned. On a 185-item legal benchmark where all six systems share one backbone, one retrieval su
    
[^3]: LiteRAG：低成本高效的基于图的检索增强生成

    LiteRAG: Cost-Efficient Graph-Based Retrieval-Augmented Generation

    [https://arxiv.org/abs/2609.10239](https://arxiv.org/abs/2609.10239)

    LiteRAG通过查询条件算法探索和推理链上下文构建取代昂贵的检索时LLM控制，在多跳问答质量上达到最优，同时将查询延迟降低100倍以上、成本降低99%以上、token使用量减少约14倍。

    

    基于图的检索可以提升多跳问答的效果，但现有方法往往在查询时产生高昂成本，并生成分散且过于庞大的上下文，从而降低生成效率。我们提出了LiteRAG，这是一种基于图的检索方法，它用基于查询条件的算法探索和推理链上下文构建，取代了昂贵的检索时LLM控制。在DistComp（一个针对分布式系统论文的多跳检索基准）上，LiteRAG在所评估的方法中获得了最高的整体质量得分（0.798），同时与GraphRAG Global和DRIFT相比，每次查询的延迟降低了100倍以上，成本降低了99%以上。在UltraDomain上，它在整体质量上与LinearRAG相当，但使用的token数量减少了约14倍。消融实验表明，LiteRAG的查询自适应阈值机制和社区感知的中心节点惩罚是其token效率提升的主要驱动因素。

    arXiv:2609.10239v1 Announce Type: cross  Abstract: Graph-based retrieval can improve multi-hop question answering, but existing approaches often incur high query-time costs and produce diffuse, oversized contexts that reduce generation efficiency. We present LiteRAG, a graph-based retrieval method that replaces expensive retrieval-time LLM control with query-conditioned algorithmic exploration and reasoning-chain context construction. On DistComp, a benchmark for multi-hop retrieval over distributed-systems papers, LiteRAG attains the highest overall quality among the evaluated methods (0.798) while reducing per-query latency by over 100$\times$ and cost by over 99% relative to GraphRAG Global and DRIFT. On UltraDomain, it matches LinearRAG on overall quality while using about 14$\times$ fewer tokens. An ablation study indicates that LiteRAG's query-adaptive thresholding and community-aware hub penalization are the main drivers of its token-efficiency gains.
    
[^4]: 知识图谱上大语言模型问答中的答案路径与接地指令

    The Answer Path and the Grounding Instruction in LLM Question Answering over Knowledge Graphs

    [https://arxiv.org/abs/2609.10237](https://arxiv.org/abs/2609.10237)

    本研究在六个大语言模型和两个知识图谱问答基准上系统变化四种提示设计选择，发现只有将答案路径纳入提示和接地指令的设计能显著影响答案准确率（非路径三元组可被无关内容替换而不影响效果，检索预算应全部用于保证召回率），而三元组的书写语法与排列顺序几乎没有作用。

    

    图检索增强生成流水线需要做出四种选择：将哪些三元组放入提示中、用什么语法书写它们、以什么顺序排列它们，以及用一句话告诉模型如何处理这些三元组。我们在六个大语言模型和两个知识图谱问答基准上对这四种选择进行了系统变化实验。结果表明，四种选择中有两种会影响答案表现，另外两种则几乎没有作用。第一种关键选择是答案路径——即推理出答案所需的三元组链——是否包含在提示中。在保持三元组总数不变的情况下，将所有不在路径链上的三元组替换为来自无关实体的材料，答案准确率仅变化+0.003 F1；而移除该路径链则会损失图检索所带来的绝大部分价值。因此，检索预算应当投入于召回率，在我们可测试的范围内，精确率并不能带来任何收益。本文中并不涉及检索器：子图来自黄金标准SPARQL，因此此处的精确率描述的是我们所构建的上下文质量，而非某个系统设置。第二种关键选择是接地指令……（摘要在此处截断）

    arXiv:2609.10237v1 Announce Type: new  Abstract: A graph retrieval-augmented generation pipeline chooses which triples to put in the prompt, a syntax to write them in, an order to write them in, and a sentence telling the model what to do with them. We vary all four over six large language models and two knowledge-graph question answering benchmarks. Two of the four choices move the answer and the other two are flat. The first is whether the answer path, the triples needed to reach the answer, is in the prompt at all. Holding the number of triples fixed and replacing every triple that is not on the chain with material from an unrelated entity changes answer accuracy by +0.003 F1, while removing the chain costs most of what the graph was worth. Retrieval budget belongs on recall, and precision in the range we can test buys nothing. There is no retriever here: subgraphs come from gold SPARQL, so precision describes the context we build, not a system setting. The second is the grounding i
    
[^5]: 从检索到权重：利用个人文本语料库实现小语言模型的参数化个体化

    From Retrieval to Weights: Parametric Individualization of Small Language Models with Individual Text Corpora

    [https://arxiv.org/abs/2609.10155](https://arxiv.org/abs/2609.10155)

    该研究通过DoRA微调将515名参与者的个人搜索历史写入小语言模型权重，证明适配器能显著编码个人语料（个体化效应dz=1.27），但在通用知识测试中模型获得的是知识而非与个体的对齐。

    

    我们从认知模拟的视角研究多选题问答中的情景记忆与语义记忆，方法是将个人文本语料库（ITC）中的文本融入检索增强生成和DoRA微调。我们通过网络爬取了515名回答36道多选题知识项的参与者的搜索历史，并分析了其中150名参与者的分层子样本。对于每位参与者，一个DoRA适配器将其个人文本语料库整合进一个小语言模型（SLM），该模型的基线正确率低于参与者群体的最低四分位数。该适配器可测量地将个人文本语料库写入权重之中：它对参与者自身留出文本的拟合优于其他参与者的文本（dz = 1.27），且这种个体化效应随个人文本语料库规模按秩次递增。然而，在通用知识测试中，适配器增加的是知识而非与个体的对齐：对数损失的匹配度有所提升，而在偏差下的匹配准确率……（原文摘要于此处截断）

    arXiv:2609.10155v1 Announce Type: new  Abstract: We approach a cognitive simulation perspective on episodic and semantic memory in multiple-choice question answering by incorporating text from individual text corpora (ITC) into retrieval-augmented generation and DoRA fine-tuning. We web-crawl the search histories of 515 participants who answered 36 multiple-choice knowledge items and analyze a stratified subsample of 150 participants. For each participant, one DoRA adapter consolidates their ITC into a small language model (SLM) whose baseline correctness falls below the participants' lowest quartile. The adapter measurably writes the ITC into the weights: it fits its own participant's held-out text better than other participants' texts (dz =1.27), an individuality effect that increases with ITC size in rank order. On the generalized knowledge test, however, the adapter adds knowledge rather than alignment with the individual: log-loss match improves, whereas match accuracy under a bia
    
[^6]: 在投机式检索增强生成中保证忠实的证据抽取

    Guaranteeing Faithful Evidence Extraction in Speculative Retrieval-Augmented Generation

    [https://arxiv.org/abs/2609.10046](https://arxiv.org/abs/2609.10046)

    该论文提出约束混合解码（CHyD），将原本用于加速推理的投机解码架构重新用于强制执行硬解码约束，从而在投机式检索增强生成中保证抽取的证据逐忠实于检索到的上下文，满足安全关键领域对答案精确匹配的要求。

    

    大型语言模型越来越多地被用作信息检索的接口，但它们仍然容易产生幻觉和忠实性错误，即生成的答案与检索到的证据相偏离。虽然检索增强生成（RAG）以及近期的混合或半抽取方法可以缓解这一问题，但它们并不能保证所引用或抽取的文本片段完全逐字来自检索到的上下文。这一局限性在安全关键领域可能造成严重后果，因为在这些领域中，答案必须与经过认证的文档完全一致。我们提出了约束混合解码，这是一种面向投机式RAG的新型“忠实性优先”范式。虽然传统的投机解码是为推理速度而优化的，但CHyD重新利用了这一架构，以便在抽取模式被正确触发时确保忠实的逐字证据抽取。我们的方法强制执行硬解码约束，将生成内容限制在……（摘要在此处截断）

    arXiv:2609.10046v1 Announce Type: new  Abstract: Large Language Models (LLMs) are increasingly used as interfaces for information retrieval, but they remain prone to hallucinations and faithfulness errors, in which the generated answers diverge from the retrieved evidence. While Retrieval-Augmented Generation (RAG) and recent hybrid or semi-extractive approaches mitigate this issue, they do not guarantee that quoted or extracted spans are verbatim from the retrieved context. This limitation can have severe consequences in safety-critical domains, where answers must exactly match certified documentation.   We introduce Constrained Hybrid Decoding (CHyD), a novel faithfulness-first paradigm for speculative RAG. While traditional speculative decoding is optimized for inference speed, CHyD repurposes this architecture to ensure faithful verbatim evidence extraction when the extraction mode is correctly triggered. Our approach enforces hard decoding constraints that restrict generation to c
    
[^7]: 真实AI对话中的购买建议与可观察的买家反应

    Purchase Advice and Observable Buyer Responses in Real AI Conversations

    [https://arxiv.org/abs/2609.09878](https://arxiv.org/abs/2609.09878)

    该研究通过审计317个真实AI助手对话记录，分析了生成式AI在购买决策中的说服作用，发现77.6%的事件中助手提供购买建议，但只有26.9%的事件中能观察到用户后续购买反应。

    

    生成式助手说服某人购买或说服其不购买的频率有多高？对话日志中包含推荐，但并不一定记录后续的决策。我们从Aiso的专有研究数据库中审计了317次历史交互，该数据库包含经授权、基于知情同意且已去标识化的与商业AI助手的对话记录。通过单智能体AI辅助筛选，我们识别出68条以购买为方向的记录；合并一条共享前缀的副本后，保留67个事件，时间跨度为2023年4月至2025年7月。在52个事件（77.6%）中，助手提供了候选选项、购买渠道或条件性偏好。其中一个事件包含对特定住宿候选方案的条件性转移推荐。没有任何事件被编码为建议放弃或推迟购买该类别的建议。在相同的购买相关任务中，只有18个事件（26.9%）包含用户后续发言，而23个事件（34.3%）包含……（原文截断）

    arXiv:2609.09878v1 Announce Type: new  Abstract: How often does a generative assistant persuade someone to buy, or persuade them not to buy? Conversation logs contain recommendations, but they do not necessarily record subsequent decisions. We audit 317 historical interactions from Aiso's proprietary research database of licensed, consent-based, de-identified conversations with commercially available AI assistants. Single-agent AI-assisted screening identifies 68 purchase-directed records; collapsing one shared-prefix copy yields 67 retained episodes, dated April 2023 to July 2025. Assistant responses provide candidate options, acquisition channels, or conditional preferences in 52 episodes (77.6%). One episode contains conditional redirection away from a named accommodation candidate. No episode is coded as advice to abandon or defer the purchase category. Only 18 episodes (26.9%) contain a subsequent user turn within the same purchase-related mission, compared with 23 (34.3%) that co
    
[^8]: 低比特量化何时能保留向量搜索的决策？

    When Does Low-Bit Quantization Preserve the Decisions of Vector Search?

    [https://arxiv.org/abs/2609.09854](https://arxiv.org/abs/2609.09854)

    该论文在向量搜索算法实际使用的比较层面分析量化误差，给出了比较翻转概率的无分布分解、协方差感知的尾部界以及Vamana邻居选择的确定性耦合定理，从而解释了低比特量化何时能保留精确搜索的决策。

    

    低比特量化在某些向量表示上能够实现高召回率，而在另一些向量表示上则会急剧失败，而平均失真和全局排序相关性无法解释这种差异。我们从排序和图剪枝算法实际使用的比较层面来研究量化向量搜索。我们的第一个结果是一个无分布分解：一次比较发生翻转的概率，由精确边际接近零的概率质量加上校准残差的尾部概率所界定。随后我们刻画了共享同一查询或图节点的残差之间的依赖性，并在联合矩生成函数代理下推导出考虑协方差的二阶矩恒等式和尾部界。对于冻结的候选排列，我们为Vamana邻居选择证明了一个确定性耦合定理：当且仅当所有候选级别的剪枝动作在冻结的精确状态上达成一致时，近似重放才恰好返回精确的邻居列表。

    arXiv:2609.09854v1 Announce Type: cross  Abstract: Low-bit quantization can achieve high recall on some vector representations and fail sharply on others, while average distortion and global rank correlation do not explain the difference. We study quantized vector search at the level of the comparisons consumed by ranking and graph-pruning algorithms. Our first result is a distribution-free decomposition: the probability that a comparison flips is bounded by the probability mass of exact margins near zero plus the tail probability of the calibrated residual. We then account for dependence between residuals that share a query or graph node, and derive covariance-aware second-moment identities and tail bounds under a joint MGF proxy. For a frozen candidate permutation, we prove a deterministic coupling theorem for Vamana neighbour selection: the approximate replay returns the exact neighbour list exactly when all candidate-level pruning actions agree on the frozen exact states. We connec
    
[^9]: 我应该对我的大语言模型相关性评判器有礼貌吗？语气作为严重程度工作点的偏移

    Should I Be Polite to My LLM Relevance Judge? Tone as a Severity Operating-Point Shift

    [https://arxiv.org/abs/2609.09703](https://arxiv.org/abs/2609.09703)

    该研究发现提示语气对大语言模型相关性评判的影响，主要通过移动评判器的严重程度工作点（整体评分宽容度）而非提升判断质量实现，且效果高度依赖于具体模型。

    

    大语言模型越来越多地被用作相关性评判器，但其标注结果可能随提示的表面形式而变化。我们在3,498个TREC DL19/DL20查询-段落对上研究了其中一个特征——语气，涵盖八个评判模型、五个经分类器校准的礼貌等级，以及每个等级的三种释义。研究效果高度依赖于具体模型：一个评判器表现出结构化的U型响应，而大多数模型仅显示微小变化。在语气改变一致性的情况下，结果更符合评判器严重程度工作点的偏移（即其整体评分宽容度的变化），而非判断质量的提升。当这种偏移使评判器趋近或远离人类标注者的严格程度时，一致性随之上升或下降。在查询不相交的交叉拟合中，这一预期关联得以保持（Spearman ρ = -0.683；精确模型块置换检验 p = 0.019）。语气对基于校准的一致性的影响大于对排序结果的影响：在32个模型-语气对比中……

    arXiv:2609.09703v1 Announce Type: new  Abstract: Large language models are increasingly used as relevance judges, yet their labels can shift with prompt surface form. We study one such feature -- tone -- on 3,498 TREC DL19/DL20 query-passage pairs, across eight judge models, five classifier-calibrated politeness levels, and three paraphrases per level. Effects are strongly model-dependent: one judge shows a structured U-shaped response, whereas most show only small changes. Where tone changes agreement, the results are more consistent with a shift in the judge's severity operating point -- its overall scoring leniency -- than with improved judgment. Agreement rises or falls as this shift moves the judge toward or away from human annotators' strictness. A query-disjoint cross-fit retains the expected association (Spearman $\rho = -0.683$; exact model-block permutation $p = 0.019$). Tone affects calibration-based agreement more than ranking outcomes: across 32 model-tone contrasts, the l
    
[^10]: 特征叠加中线性能及性的高概率保证

    High-probability guarantees for linear accessibility in feature superposition

    [https://arxiv.org/abs/2609.09556](https://arxiv.org/abs/2609.09556)

    该论文将特征叠加中的线性能及性建模为压缩感知问题，证明了充分维度只需线性规模（而非此前最坏情况的二次方限制）即可高概率地恢复同时激活的特征，从而量化了线性表示假设的几何约束并为稀疏自编码器和神经可解释性评估提供了理论框架。

    

    神经网络可以利用特征叠加来编码比维度数量更多的概念，但特征间的交叉干扰限制了同时激活特征的线性能及性。通过将线性能及性建模为一个压缩感知问题，我们在次高斯噪声下针对固定支撑集推导出高概率界，证明了充分维度以线性方式扩展（d=O_ε(k log m)），而非此前最坏情况下的二次方限制。随后，我们通过高斯尾近似在各系统参数下验证了这些界。这些结果量化了线性表示假设的几何约束，为评估稀疏自编码器、组合泛化和神经网络可解释性提供了一个框架。

    arXiv:2609.09556v1 Announce Type: cross  Abstract: Neural networks can leverage feature superposition to encode more concepts than dimensions, but cross-feature interference constrains the linear accessibility of simultaneously active features. By framing linear accessibility as a compressed sensing problem, we derive high-probability bounds for fixed supports under subgaussian noise, proving the sufficient dimension scales linearly ($d=O_{\varepsilon}(k \log m)$) rather than prior worst-case quadratic limits. We then validate these bounds across system parameters through Gaussian-tail approximations. These results quantify the geometric constraints of the linear representation hypothesis, providing a framework for evaluating sparse autoencoders, compositional generalization, and neural interpretability.
    
[^11]: 从牛群报告类别中提取语义以实现数据互操作性和可发现性

    Extracting Semantics from Cattle Reporting Categories for Data Interoperability and Findability

    [https://arxiv.org/abs/2609.09381](https://arxiv.org/abs/2609.09381)

    本研究采用自下而上的方法从牛群报告类别中提取语义，以解决牲畜数据缺乏元数据和标准的问题，从而提高数据的互操作性和可发现性。

    

    按年龄、性别和生产方式分类的牲畜种群数据是帮助理解全球健康问题的计算和模型的重要输入，然而这些数据分散在各个不同的来源中。弥合数据孤岛以提高数据的可发现性需要互操作性。提高数据可发现性和互操作性的传统方法包括为标准化元数据建立索引。然而，创建元数据既耗时又耗费资源，并且在牲畜等缺乏满足广泛用户群体需求的标准的领域中往往很困难。当元数据存在时，它们通常需要根据预先存在的词汇表、本体或叙词表进行标准化，这需要一种称为"交叉映射"的技术。为了克服缺乏元数据、缺乏标准以及现有解决方案资源密集等问题，本研究采用了一种自下而上的方法。

    arXiv:2609.09381v1 Announce Type: new  Abstract: Livestock population data disaggregated by age, sex, and production are important inputs to calculations and models that inform our understanding of global health, yet these data are fragmented across disparate sources. Bridging data siloes to improve the findability of data requires interoperability. Conventional approaches to improving the findability and interoperability of data include indexing standardized metadata. However, creating metadata is time and resource-intensive and is often difficult in domains such as livestock, which lack standards that address the needs of broad user groups. When metadata exist, they typically need to be standardized against a pre-existing vocabulary, ontology, or thesaurus, requiring a technique known as `crosswalking'. To overcome issues in the absence of metadata, the lack of standards, and the resource-intensive solutions that currently exist, this study uses a bottom-up approach. By leveraging re
    
[^12]: 合作更佳：强RAG基线下的互补性查询改写

    Better Together: Complementary Query Rewriting Under a Strong RAG Baseline

    [https://arxiv.org/abs/2609.05637](https://arxiv.org/abs/2609.05637)

    在强RAG检索基线下，单一查询改写策略收效有限，但联合多种互补的改写方法可大幅提升检索性能，企业数据上HIT@10提升12.5个百分点。

    

    arXiv:2609.05637v2 公告类型：替换 摘要：改进检索增强生成（RAG）的一种流行方法是将用户问题改写为多个变体并使用所有变体进行搜索。我们测试了在底层检索系统已经很强大的情况下，这种方法是否真的有效。在一个固定的、具有竞争力的流程（BGE密集检索、交叉编码器重排序和MMR多样化）下，我们在三个数据集（HotpotQA、AmbigNQ以及包含51.2万文档的EnterpriseRAG-Bench）上，通过三个随机种子和配对自举显著性检验，将四种查询改写策略（S1-S4）与两个强大的LLM基线（HyDE、Query2Doc）进行了比较。我们的核心发现是：单独使用改写策略充其量只能与强大基线持平，但组合多种方法能带来显著的超额收益，因为不同策略在不同类型的问题上各有短板。四种方法的事后联合（S1+S3+S4+HyDE）在企业数据上将HIT@10指标比基线提升了12.5个百分点（51.70对39.22），而五种方法的联合更是达到了52.9。

    arXiv:2609.05637v2 Announce Type: replace  Abstract: A popular way to improve Retrieval-Augmented Generation (RAG) is to rewrite the user's question into several variants and search with all of them. We test whether this actually helps once the underlying search is already strong. Under one fixed, competitive pipeline (BGE dense retrieval, cross-encoder reranking, and MMR diversification), we compare four query-rewriting strategies (S1-S4) against two strong LLM baselines (HyDE, Query2Doc) on three datasets (HotpotQA, AmbigNQ, and the 512K-document EnterpriseRAG-Bench) over three seeds with paired-bootstrap significance tests. Our headline result is that rewriting alone is at best competitive with a strong baseline, but combining methods yields outsized gains because different strategies fail on different questions. A post-hoc union of four methods (S1+S3+S4+HyDE) improves HIT@10 over the baseline by +12.5 points on enterprise data (51.70 vs 39.22), and a five-method union reaches 52.9
    
[^13]: DexterSQL：面向文本到SQL生成的深度模式探索与基于规则的修正

    DexterSQL: Deep Schema Exploration and Rule-based Correction for Text-to-SQL Generation

    [https://arxiv.org/abs/2608.11889](https://arxiv.org/abs/2608.11889)

    DexterSQL通过深度模式探索、数据库无关规则挖掘和规则驱动修正三个创新组件，解决了非微调文本到SQL生成中模式信息粗糙、错误重复出现和条件处理不当的问题。

    

    arXiv:2608.11889v1 公告类型：交叉 摘要：基于提示（即非微调）的文本到SQL方法，其中底层大语言模型参数不针对任务进行更改，面临三个问题：（i）依赖粗粒度的模式信息，这可能无法揭示区分模糊列所需的细粒度关系，（ii）未能捕捉重复出现的SQL生成失败，以及（iii）在复杂问题中遭受条件遗漏、幻觉或错位。本文开发了DexterSQL，一个基于提示/非微调的文本到SQL系统，通过三个新组件改进SQL生成：（i）深度模式探索器，识别模糊列，分析其单独和联合数据分布以揭示它们之间的关系及各自的不同作用，（ii）数据库无关的规则创建器，挖掘生成结果与目标结果之间的不匹配，（iii）规则驱动的修正器，应用这些规则来纠正SQL生成中的常见错误。

    arXiv:2608.11889v1 Announce Type: cross  Abstract: Prompting-based (\textit{i}.\textit{e}., non-fine-tuning) Text-to-SQL methods, where underlying large language model parameters are not changed for the task, face three problems: (\textit{i})~relying on coarse-grained schema information that may not reveal the fine-grained relationships needed to distinguish ambiguous columns, (\textit{ii})~not capturing recurring SQL-generation failures, and (\textit{iii})~suffering from omission, hallucination, or misplacement of conditions in complex questions.   This paper develops \textsc{DexterSQL}, a prompting/non-fine-tuning-based Text-to-SQL system that improves SQL generation with three novel components: (\textit{i})~\emph{deep schema explorator} that identifies ambiguous columns, analyzes their individual and joint data distributions to uncover their relationships and the distinct role of each, (\textit{ii})~\emph{database-agnostic rule creator} that mines mismatches between generated and go
    
[^14]: LoopMemGR：面向生成式推荐的从行为日志到演化记忆

    LoopMemGR: From Behavior Logs to Evolving Memory for Generative Recommendation

    [https://arxiv.org/abs/2607.27647](https://arxiv.org/abs/2607.27647)

    本文提出LoopMemGR，一个闭环推荐经验记忆框架，在传统行为日志之外额外维护系统自身的推荐决策与反馈记忆，解决了生成式推荐中“只记用户行为、不记自身推荐”的不对称记忆问题，使偏好验证信号、反向证据和探索信息能够跨请求复用。

    

    生成式推荐将下一个物品预测问题建模为在离散语义ID上的条件自回归生成，从而实现了对大规模物品空间的端到端推荐。然而，现有方法大多遵循“历史即上下文”的范式，即反复从用户行为历史中重建用户偏好，同时在每次请求结束后丢弃系统侧的推荐决策。这导致了一种不对称的记忆：系统记得用户做过什么，却不记得自己之前推荐过什么以及从反馈中学到了什么。因此，有价值的偏好验证信号、潜在的反向证据以及历史探索信息无法在请求之间直接复用。为解决这些局限，我们提出了LoopMemGR，一个面向生成式推荐的闭环推荐经验记忆框架。除了传统的行为日志之外，LoopMemGR还维护一个推荐……（摘要原文在此处截断）

    arXiv:2607.27647v2 Announce Type: replace  Abstract: Generative recommendation formulates next-item prediction as conditional autoregressive generation over discrete Semantic IDs, enabling end-to-end recommendation over large-scale item spaces. However, most existing methods follow a history-as-context paradigm that repeatedly reconstructs user preference from behavior history while discarding system-side recommendation decisions after each request. This creates an asymmetric memory: the system remembers what the user has done, but not what it has previously recommended or learned from the resulting feedback. Consequently, useful preference-validation signals, potential negative evidence, and historical exploration information cannot be directly reused across requests. To address these limitations, we propose LoopMemGR, a closed-loop recommendation experience memory framework for generative recommendation. In addition to the conventional behavior log, LoopMemGR maintains a recommendati
    
[^15]: 关于人脸验证下可区分合成身份生成容量的研究

    On the Capacity of Distinguishable Synthetic Identity Generation under Face Verification

    [https://arxiv.org/abs/2604.10641](https://arxiv.org/abs/2604.10641)

    该论文首次在人脸验证框架下定义了合成身份生成的有限维容量这一理论概念，证明了确定性视角不变流水线下该容量等于可实现嵌入集上的球面码基数，并为随机身份条件嵌入分布推导了容量下界与中心分离条件。

    

    合成人脸生成器可以产生许多名义上的身份，但名义数量并不能决定在指定的验证规则下有多少个身份是联合可区分的。我们定义了有限维容量为码本大小的上确界，该码本由不同的潜在身份编码组成，其诱导的身份条件嵌入分布满足每个身份的真接受约束和成对冒名者不匹配约束。对于确定性的视角不变流水线，固定码容量等于可实现嵌入集上的球面码基数，并且当每个球面方向均可实现时，该容量简化为经典的球面码基数。对于以至少 $1-\eta$ 的概率集中在角半径为 $\rho$ 的球冠内的随机身份条件嵌入分布，我们推导了充分的中心分离条件、完全角度表达能力下的球面码容量下界，以及泊（原文截断于此）

    arXiv:2604.10641v2 Announce Type: replace-cross  Abstract: Synthetic face generators can produce many nominal identities, but nominal count does not determine how many are jointly distinguishable under a specified verification rule. We define finite-dimensional capacity as the supremum of codebook sizes over distinct latent identity codes whose induced identity-conditional embedding distributions satisfy per-identity genuine acceptance and pairwise impostor non-match constraints. For deterministic view-invariant pipelines, fixed-code capacity equals the spherical-code cardinality over the realizable embedding set and reduces to the classical spherical-code cardinality when every sphere direction is realizable. For stochastic identity-conditional embedding distributions concentrated with probability at least $1-\eta$ in spherical caps of angular radius $\rho$, we derive a sufficient center-separation condition, spherical-code capacity lower bounds under full angular expressivity, and po
    
[^16]: 电商搜索中的查询品牌实体链接

    Query Brand Entity Linking in E-Commerce Search

    [https://arxiv.org/abs/2502.01555](https://arxiv.org/abs/2502.01555)

    该论文提出两种互补的品牌实体链接方法——级联流水线和极端多分类单阶段方法，在11种语言评估和在线实验中显著提升电商搜索的品牌召回率并保持高精确率，带来用户参与度的可衡量提升。

    

    将用户搜索查询与正确的品牌实体关联起来对于电商产品检索至关重要，但由于查询的简短性（平均仅三到四个词）、缺乏语法结构，以及拥有数十万个不同品牌的商品目录，这项任务仍然充满挑战。我们将其定义为品牌实体链接任务，并开发了两套互补且已大规模部署的解决方案：（1）级联流水线方法，首先通过序列标注检测品牌提及，然后针对品牌知识库进行消歧；（2）单阶段方法，将链接任务建模为极端多分类问题，直接将查询映射到品牌标识符。通过广泛的多语言评估（涵盖11种语言）和受控在线实验，我们证明了所提出的方法在保持高精确率的同时大幅提升了品牌召回率，并带来了可衡量的用户参与度提升。

    arXiv:2502.01555v3 Announce Type: replace  Abstract: Associating user search queries with the correct brand entity is critical for e-commerce product retrieval, yet remains challenging due to the brevity of queries (three to four words on average), their lack of grammatical structure, and a catalog of hundreds of thousands of distinct brands. We formulate this as a brand entity linking task and develop two complementary solutions deployed at scale: (1) a cascaded pipeline that first detects brand mentions via sequence labeling and then disambiguates against a brand knowledge base, and (2) a single-stage approach that frames linking as extreme multiclass classification, directly mapping queries to brand identifiers. Through extensive multilingual evaluation (11 languages) and a controlled online experiment, we demonstrate that the proposed methods substantially improve brand recall while maintaining high precision, leading to measurable gains in customer engagement.
    

