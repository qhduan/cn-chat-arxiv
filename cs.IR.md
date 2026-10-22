# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Think Harder Where It Matters: Item-Level Adaptive Recursion for Sequential Recommendation](https://arxiv.org/abs/2610.30128) | 提出LoopRec，一种针对序列推荐的物品级自适应递归方法，根据每个物品的实际计算需求动态分配递归深度，取代固定计算预算，从而既节省计算资源又提升整体推荐质量。 |
| [^2] | [What the Author Wrote, What the Agent Reads: Silent Cue Loss in Enterprise Document Ingestion](https://arxiv.org/abs/2610.30026) | 该研究揭示了企业文档摄取环节中所有主流转换器都会丢失作者通过文本格式编码的含义，导致来自两家供应商的六个大模型智能体在恢复作者重点标注要求的任务上性能从F1 0.690骤降至0.075-0.150，且这种失败表现为“自信地给出错误答案”而非“明显的不完整”。 |
| [^3] | [ERCLIP: Small-Data Adaptation of CLIP for Free-Length Image-Text Retrieval](https://arxiv.org/abs/2610.29939) | ERCLIP通过将长文本描述拆分为句子独立编码并融合，仅在训练阶段使用完整长文本进行对比学习，配合冻结CLIP保留损失维持零样本分类能力，仅用2万图文对就将长文本图文检索R@1提升6.5至11.2个百分点，以LongCLIP 2%的数据量弥补其62%至77%的性能差距。 |
| [^4] | [Real PMIDs, Wrong Papers: Harness Design for Biomedical Claim-Verification Agents](https://arxiv.org/abs/2610.29812) | 该论文通过受控消融实验发现，在基于实时PubMed API的生物医学论断验证任务中，让语言智能体自行构造查询的框架设计能将标准论文召回率提升三倍并取得最佳准确率（62%），而简单的单次检索仅能为6%的论断找到标准论文且无法提升准确率。 |
| [^5] | [SOAR: Social and Opinion-Aware Recommendations](https://arxiv.org/abs/2610.29556) | 提出了SOAR推荐模型，通过结合用户社交好友网络与文本评论的情感信息，实现更精准的个性化兴趣点推荐。 |
| [^6] | [pplx-embed-v2-late Technical Report](https://arxiv.org/abs/2610.29523) | Perplexity 推出的 pplx-embed-v2-late 包含 0.6B 和 9B 两个多语言多模态后期交互模型，通过 18B 教师模型的表示蒸馏训练，在文本、图像和视觉文档检索上达到同规模最先进水平，并借助共享嵌入空间支持“9B 建索引、0.6B 低延迟查询”的非对称部署。 |
| [^7] | [X-Embedding: Two-Stage Training for Multilingual Patent Prior-Art Retrieval](https://arxiv.org/abs/2610.29521) | 提出两阶段训练框架X-Embedding，先通过专利内部章节对齐与跨语言专利族对齐学习专利领域表示，再利用审查报告X类引用作为监督信号优化检索相关性，有效提升多语言专利现有技术检索中X类文献的识别能力。 |
| [^8] | [Beyond the Vote: When Do LLMs Add Value to Retrieval-Based Tabular Classification?](https://arxiv.org/abs/2610.29520) | 该研究在19个表格分类数据集上将大语言模型与距离加权最近邻投票进行系统比较，发现Distance-KNN整体平均准确率最高，LLM仅在检索邻居标签一致性低且测试样本距离较近或中等时才有优势。 |
| [^9] | [Directional Posterior Feedback Fusion for Query Reformulation in Dense Retrieval](https://arxiv.org/abs/2610.29464) | 该论文提出了方向性后验反馈融合方法，通过以方向性方式融合反馈证据而非简单的逐点聚合或插值，克服了现有嵌入空间查询重构方法会抑制仅由互补证据支持的方向上查询修正的缺陷，从而提升密集检索对简短、模糊查询的处理效果。 |
| [^10] | [Generative Recommenders Forget the Long Tail: Restoring Memorization in Retrieval and Ranking](https://arxiv.org/abs/2610.29298) | 提出免训练插件CAPE，利用物品共现记忆机制修复生成式推荐系统中长尾物品在检索与排序两个阶段被系统性遗忘的问题。 |
| [^11] | [Extending FAIR Principles with Observability for Trustworthy Federated Learning](https://arxiv.org/abs/2610.29252) | 提出FAIR-O框架，将FAIR原则扩展出可观测性维度，使每个参与者在每轮联邦学习中生成最小化的签名信任证据（可由本地TinyLLM辅助解释），在不迁移原始数据的前提下建立三级信任模型以实现可信联邦学习。 |
| [^12] | [Learning Marginal Viewpoint Utility for Budgeted Evidence Selection in RAG](https://arxiv.org/abs/2610.29234) | 提出轻量级排序器LMVU，通过学习候选证据为当前选择新增的边际视角覆盖来动态优先排序证据，从而在固定上下文预算下最大化多视角查询RAG的视角覆盖效果。 |
| [^13] | [Preference Is Not Permission: Personalizing the Auto-Purchase Line in Agentic Commerce](https://arxiv.org/abs/2610.29216) | 该论文提出将学习个人化的“自动购买边界”作为代理式商务中的新任务，并发现十六个指令微调模型普遍“越权”——即使商品匹配度低、购买依据缺失也会自动下单，与人类委托者的授权意愿存在系统性错位，且这一边界因人而异、需要个性化建模。 |
| [^14] | [Reasoning-Search-Augmented LLMs: An Operational Taxonomy, Benchmark Analysis and Controlled Evidence Synthesis](https://arxiv.org/abs/2610.29189) | 该论文提出了一个针对推理-搜索增强大语言模型（LLMs）的操作性分类体系，通过检索耦合、控制器设计、知识来源和接地机制四个架构维度系统性地分析将逐步推理与交互式搜索紧密耦合的框架，并提供了基准分析与受控证据综合。 |
| [^15] | [Progressive Learning from Content to Relevance to Personalization for CTR Prediction](https://arxiv.org/abs/2610.29138) | 提出FineSID渐进式学习框架，通过共享的细粒度语义ID表示统一内容分词、查询-物品相关性语义预训练和个性化后训练，有效解决多模态CTR预测中细粒度异构内容与ID预测器融合的难题。 |
| [^16] | [Our Own Jev: Typed Decisions in One Forward Pass, from an In-House Judge to Open LoRA Models](https://arxiv.org/abs/2610.29052) | 该论文提出用单次前向传播的“类型化决策”接口（直接读取logits或轻量指针头）替代生成式LLM裁判，其自研System 1裁判仅用一块GPU约3分钟、约0.06美元即可判定1,000个智能体编码会话，以约1%的成本取代每轮约1,000美元且标签一致性仅26.9%的前沿LLM裁判。 |
| [^17] | [Scriptorium: System Design and Implementation of a Multi-Human, Multi-AI Collaborative Digital Library Search Platform](https://arxiv.org/abs/2610.28929) | 该论文提出了Scriptorium，一个支持多人类与多AI协作的数字图书馆搜索平台，其核心创新在于将完全跨用户透明的搜索活动与由七个角色专门化生成式AI智能体组成的管理者-工作者架构相结合，并完整公开系统实现细节以支持复现与重用。 |
| [^18] | [Constrained Tagging for Span Labeling](https://arxiv.org/abs/2610.28922) | 提出了一种无需训练的受限标记方法，通过字节级词元trie和有限状态机约束模型逐字复现输入并内联标注片段，从而保证片段标注输出与输入位置精确对应，且几乎不产生失败。 |
| [^19] | [FutureLens: Longitudinal Topic Patterns in Scholarly Text and Future-Work References](https://arxiv.org/abs/2610.28656) | FutureLens 基于 2012–2022 年 5,655 篇 ACL 论文，系统考察了利用“未来工作”陈述追踪学术领域主题演变时的测量方法问题，并比较了 NMF 与 LDA 两种主题模型在稳定性、信息量与关键词多样性上的表现差异。 |
| [^20] | [Building Knowledge Graphs for Theses: A Survey](https://arxiv.org/abs/2610.28648) | 本文系统综述了2021至2025年间46项关于学位论文知识图谱构建的同行评审研究，提出了技术、模型与工具的结构化分类体系，并揭示了从自然语言处理抽取技术到大语言模型驱动增强的方法论演进。 |
| [^21] | [Controllable Generative Recommendation via Guided Token Refinement](https://arxiv.org/abs/2610.28642) | ReGen将生成式推荐从自回归的顺序解码转变为基于掩码填充的并行迭代token精炼，无需重新训练即可直接对高层语义token施加控制信号，同时保证解码候选映射到真实物品。 |
| [^22] | [Transaction Foundation Models for Retrieval-Augmented Generation over Financial Transaction Data: A Comparative Evaluation with Text Embeddings](https://arxiv.org/abs/2610.28641) | 本研究在IBM合成反洗钱交易数据集上构建了1,026个问题的问答基准，证明NVIDIA的2900万参数交易基础模型（TFM）在金融交易数据检索增强生成任务中的检索性能和端到端答案准确率均大幅超越同规模的通用文本嵌入模型。 |
| [^23] | [Can AI Chat Replace Traditional Search? Evidence from a RAG-based Live Events Experiment](https://arxiv.org/abs/2610.28639) | 通过一项预注册的现场活动搜索实验发现，基于RAG的纯AI聊天界面搜索表现最差，尤其在开放式探索场景中，表明生成式AI对话界面尚不能取代传统搜索方式。 |
| [^24] | [TAGR: Task-Adaptive Geo-Semantic Retrieval across Heterogeneous Urban Delivery Tasks](https://arxiv.org/abs/2610.28638) | 提出TAGR框架，通过任务自适应的地理语义检索机制，为城市配送中需求预测、ETA预测和下一站预测三类异构任务检索结构与粒度各异的历史证据，解决单一融合机制导致的匹配不当与负迁移问题，提升预测精度与资源分配效率。 |
| [^25] | [RESEARCHHUB: A Personalized Literature Discovery System for Biomedical Preprints](https://arxiv.org/abs/2610.28636) | 本文提出ResearchHub个性化文献发现系统，其核心检索器prism通过GPT-4o生成查询和混合训练目标适配生物医学预印本，在MedRxiv语料上将Recall@10较SPECTER2-Search提升12.4个百分点，并支持可控新颖性、多样性与课程式排序。 |
| [^26] | [DuoLM: Fast and Effective Personalized Recommendation with Language Models](https://arxiv.org/abs/2610.24481) | DuoLM框架通过编码器LLM离线学习紧凑物品表示、个性化排序器LLM融合用户历史与反馈，并借助虚拟token ID、共享历史计算和CUDA Graph重放等运行时加速技术，实现了与专家教师模型相当的推荐质量，同时大幅提升服务吞吐量。 |
| [^27] | [WeGR: Unified Generative Retrieval for Industrial Search Systems in WeChat](https://arxiv.org/abs/2610.16979) | WeGR提出统一生成式检索框架，通过融合协同信号且避免码本坍缩的CoCo-VAE分词器，并解决强化学习中精确SID匹配带来的奖励稀疏问题，显著提升工业搜索的检索质量。 |
| [^28] | [Token-Budgeted Escalation for Financial Document QA: Cost Is Predictable, Benefit Is the Bottleneck](https://arxiv.org/abs/2610.07760) | 本文将金融文档问答中的选择性升级形式化为令牌预算约束下的批次分配问题，发现额外检索调用的成本高度可预测，但准确识别哪些升级真正有益仍是主要瓶颈。 |
| [^29] | [Hierarchical Exponential-Gaussian Mixtures for Watch-Time Distribution Prediction](https://arxiv.org/abs/2608.23356) | 本文提出了一种分层指数-高斯混合模型，通过分层分解和正则化技术解决了现有EGMN模型的方差崩溃和组件冗余问题，从而提升了短视频推荐中观看时间预测的准确性。 |
| [^30] | [SIREN (Luring LLMs onto the Rocks): PAIR-Driven Preference Manipulation in Web-RAG Recommenders](https://arxiv.org/abs/2607.21951) | 本文提出SIREN方法，利用PAIR越狱循环和23种可解释编辑策略，在Web-RAG推荐系统中操纵LLM的排名输出，使目标实体升至首位。 |
| [^31] | [DualView: Adaptive Local-Global Fusion for Multi-Hop Document Reranking](https://arxiv.org/abs/2605.18767) | 提出了双视图级联重排序框架DualView，通过局部评分器（堆叠交叉注意力）与全局评分器（Transformer文档间依赖建模）的自适应门控动态融合，实现高效精准的多跳文档重排序。 |
| [^32] | [Controlling Output Rankings in LLM-Based Search via Generative Engine Optimization](https://arxiv.org/abs/2602.03608) | 提出了CORE优化方法，通过向搜索引擎返回的检索内容附加策略性设计的优化内容，来控制基于大语言模型的生成式搜索中的输出排名，从而提升小企业和独立创作者产品的可见性。 |
| [^33] | [PRISM: Prompt-Refined In-Context System Modelling for Financial Retrieval](https://arxiv.org/abs/2511.14130) | PRISM 提出一种无需训练的金融文档检索框架，通过组合提示工程、上下文学习与候选精炼三个可部署模块，将 FiQA-2018 的 Recall@100 从 0.7783 提升至 0.8326 并降低约 35% 的每查询成本，同时系统性地揭示了 k-倒数编码与多智能体协作等额外复杂性何时不再带来收益的边界条件。 |
| [^34] | [The Answer Lies Within: Self-Derived Rewards Enable Explainable Relation Extraction](https://arxiv.org/abs/2510.06198) | 该论文提出COGRE认知启发推理框架与HIT@DICT强化学习中间奖励策略，利用从正确预测中自动提取的信用词典派生奖励，引导大模型推理聚焦关系相关语义并与人类标注的抽象层次对齐，从而同时提升单样本关系抽取的准确率和解释质量。 |

# 详细

[^1]: 在关键之处深入思考：面向序列推荐的物品级自适应递归

    Think Harder Where It Matters: Item-Level Adaptive Recursion for Sequential Recommendation

    [https://arxiv.org/abs/2610.30128](https://arxiv.org/abs/2610.30128)

    提出LoopRec，一种针对序列推荐的物品级自适应递归方法，根据每个物品的实际计算需求动态分配递归深度，取代固定计算预算，从而既节省计算资源又提升整体推荐质量。

    

    现有的序列推荐模型通常为用户交互历史中的每个物品分配固定的计算预算，而不考虑该物品对最终预测的贡献。我们通过两项诊断性发现对这一事实上的设计提出挑战。（1）计算需求的不均匀性：同一序列中不同物品所需的计算量差异极大，大多数物品经过浅层处理即可收敛，而少数关键物品只有在完整深度下才能稳定。（2）固定分配的次优性：即使能够理想地获取目标物品信息，非均匀的深度分配也始终优于固定分配，这表明现有的均匀计算既在已收敛的物品上浪费了计算能力，又牺牲了整体推荐质量。这些观察促使计算方式从固定深度转向物品级自适应推理。为此，我们提出了LoopRec。LoopRec递归地应用一个权重共享的单一模块……（原文摘要在此处被截断）

    arXiv:2610.30128v1 Announce Type: cross  Abstract: Existing sequential recommenders usually allocate a fixed computational budget to every item in a user's interaction history, regardless of the item's contribution to the final prediction. We challenge this de facto design with two diagnostic findings. (1) Non-uniform computational need: items within the same sequence vary drastically in how much computation they require. Most converge after shallow processing while a critical minority only stabilizes at full depth. (2) Suboptimality of fixed allocation: given oracle access to the target item, a non-uniform depth assignment consistently outranks the fixed one, indicating that existing uniform computation both wastes capacity on already-converged items and sacrifices overall recommendation quality. These observations motivate a shift from fixed-depth computation to item-level adaptive reasoning. To this end, we propose LoopRec. LoopRec recursively applies a single weight-shared block an
    
[^2]: 作者所写与智能体所读：企业文档摄取中的静默线索丢失

    What the Author Wrote, What the Agent Reads: Silent Cue Loss in Enterprise Document Ingestion

    [https://arxiv.org/abs/2610.30026](https://arxiv.org/abs/2610.30026)

    该研究揭示了企业文档摄取环节中所有主流转换器都会丢失作者通过文本格式编码的含义，导致来自两家供应商的六个大模型智能体在恢复作者重点标注要求的任务上性能从F1 0.690骤降至0.075-0.150，且这种失败表现为“自信地给出错误答案”而非“明显的不完整”。

    

    企业智能体基于文档采取行动。在智能体能够遵循某项政策、回答合规问题或提交工单之前，某个摄取层必须先将Word文件或PDF转换为令牌（token）——而正是这一层面导致部署在真实环境中的智能体悄然丢失信息。我们证明，每一款主流转换器都会丢失一个层面的信息——即作者通过文本外观所编码的含义——并且这种丢失使智能体“自信地犯错”而非“明显地不完整”。我们基于76份真实政策文档构建了一项任务：恢复作者特别标出的要求，并采用一个无需LLM评判的确定性匹配器、依据文件自身的OOXML属性进行评分。对格式“视而不见”的摄取方式使微观F1从0.690骤降至0.150（docling）和0.075（纯文本），这一差距在来自两家供应商的六个回答模型中均得到复现——涵盖Opus、Sonnet和Haiku三个层级的五个Claude模型，以及OpenAI的GPT-5.6。无法看到线索的智能体只能靠猜测作答，而文档……

    arXiv:2610.30026v1 Announce Type: new  Abstract: Enterprise agents act on documents. Before an agent can follow a policy, answer a compliance question or file a ticket, some ingestion layer must turn a Word file or PDF into tokens - and that layer is where an agent deployed in the wild silently loses information. We show that one layer is lost by every mainstream converter - the meaning an author encodes in how text looks - and that the loss makes agents confidently wrong rather than visibly incomplete. We build a task from 76 real policy documents: recover the requirements the author singled out, scored against the file's own OOXML properties by a deterministic matcher with no LLM judge. Formatting-blind ingestion collapses from micro F1 0.690 to 0.150 (docling) and 0.075 (plain text), a gap that replicates across six answering models from two vendors - five Claude models spanning the Opus, Sonnet and Haiku tiers, and OpenAI's GPT-5.6. An agent that cannot see the cue guesses, and doc
    
[^3]: ERCLIP：面向自由长度图文检索的CLIP小数据适配方法

    ERCLIP: Small-Data Adaptation of CLIP for Free-Length Image-Text Retrieval

    [https://arxiv.org/abs/2610.29939](https://arxiv.org/abs/2610.29939)

    ERCLIP通过将长文本描述拆分为句子独立编码并融合，仅在训练阶段使用完整长文本进行对比学习，配合冻结CLIP保留损失维持零样本分类能力，仅用2万图文对就将长文本图文检索R@1提升6.5至11.2个百分点，以LongCLIP 2%的数据量弥补其62%至77%的性能差距。

    

    CLIP的77个token文本编码器会截断详细的图像描述。我们提出了ERCLIP，一种面向长文本描述图文检索的CLIP小数据适配方法。在训练过程中，长文本描述被拆分成句子，由CLIP文本编码器独立编码；句子嵌入通过均值池化或门控残差Transformer评分器进行融合，并拥有各自的对比损失。在推理阶段，ERCLIP是一个未经修改的CLIP双编码器：测试时的文本描述仍通过77个token的窗口读取，只有训练阶段才能看到完整的文本描述。我们发现这种微调会损害CLIP的零样本分类能力，尤其是当融合输出不是CLIP嵌入的组合时更为严重，因此我们在图像嵌入和短文本描述嵌入上添加了冻结CLIP保留损失。仅使用2万个图文对和一个epoch的训练，ERCLIP在长文本检索R@1指标上比CLIP提升了6.5至11.2个百分点，以LongCLIP仅2%的长文本数据量弥补了与其62%至77%的性能差距。

    arXiv:2610.29939v1 Announce Type: new  Abstract: CLIP's 77-token text encoder truncates detailed image descriptions. We present ERCLIP, a small-data adaptation of CLIP for long-caption image--text retrieval. During training, a long caption is split into sentences that the CLIP text encoder encodes independently; the sentence embeddings are fused by mean pooling or a gated residual Transformer scorer and receive their own contrastive loss. At inference ERCLIP is an unmodified CLIP dual encoder: test captions are still read through the 77-token window, and only training sees the full caption. We find that this fine-tuning erodes CLIP's zero-shot classification, most when the fusion output is not a combination of CLIP embeddings, and add a frozen-CLIP preservation loss on the image and short-caption embeddings. With 20k pairs and one epoch, ERCLIP improves long-caption R@1 over CLIP by 6.5--11.2 points, closing 62--77% of the gap to LongCLIP with 2% of its long captions. Preservation reco
    
[^4]: 真实的PMID，错误的论文：生物医学论断验证智能体的框架设计

    Real PMIDs, Wrong Papers: Harness Design for Biomedical Claim-Verification Agents

    [https://arxiv.org/abs/2610.29812](https://arxiv.org/abs/2610.29812)

    该论文通过受控消融实验发现，在基于实时PubMed API的生物医学论断验证任务中，让语言智能体自行构造查询的框架设计能将标准论文召回率提升三倍并取得最佳准确率（62%），而简单的单次检索仅能为6%的论断找到标准论文且无法提升准确率。

    

    文献检索语言智能体正在成为生命科学领域科研助手的标准组成部分，然而关于其框架中哪些部分真正重要，目前缺乏受控的实验证据。我们以实时PubMed API作为唯一工具，在生物医学论断验证任务上进行了受控消融实验。在固定骨干模型和输出格式的前提下，我们比较了四种框架：闭卷直接回答、以论断本身作为查询的单次检索、能够自行编写PubMed查询的ReAct式智能体，以及给定标准摘要的oracle（理想上界），每种框架均使用两个高效的前沿骨干模型。天真的单次检索仅能为6%的论断找到标准论文，并且只是将过度自信的错误转化为弃权，而没有提升准确率。让模型自行构造查询可将标准论文的召回率提升三倍，并产生最佳的判定结果（准确率为62%，而其他现实可行的框架为49-59%），尽管这些差异并不显著。

    arXiv:2610.29812v1 Announce Type: cross  Abstract: Literature-searching language agents are becoming a standard component of scientific assistants for the life sciences, yet there is little controlled evidence on which parts of their harness matter. We run a controlled ablation on biomedical claim verification (SciFact) with the live PubMed API as the only tool. Holding the backbone and output format fixed, we compare four harnesses: closed-book answering, single-shot retrieval using the claim as the query, a ReAct-style agent that writes its own PubMed queries, and an oracle given the gold abstracts, each with two efficient frontier backbones. Naive single-shot retrieval finds the gold paper for only 6% of claims and turns overconfident errors into abstentions without improving accuracy. Letting the model formulate its own queries triples gold recall and yields the best verdicts (62% accuracy vs. 49-59% for the other realistic harnesses), although these differences are not significant
    
[^5]: SOAR：社交与观点感知的推荐系统

    SOAR: Social and Opinion-Aware Recommendations

    [https://arxiv.org/abs/2610.29556](https://arxiv.org/abs/2610.29556)

    提出了SOAR推荐模型，通过结合用户社交好友网络与文本评论的情感信息，实现更精准的个性化兴趣点推荐。

    

    在线平台面临的一个持久挑战是推荐问题。无论是社区还是广告商，在分析复杂的用户行为、识别用户习惯和不断变化的偏好，以及提供与用户兴趣相关的及时推荐方面都存在困难。特别是兴趣点推荐还面临额外的挑战：识别影响用户行为的因素、捕捉用户倾向，并为每个个体生成准确、个性化的推荐。在本工作中，我们提出了SOAR，一个社交与观点感知的推荐模型，该模型结合了用户的社交好友网络和文本评论来评估某个地点是否适合特定用户，并特别关注共享评论中的情感信息。在真实世界数据集上的实验结果表明，将评论纳入推荐过程能够提高推荐的精确度。

    arXiv:2610.29556v1 Announce Type: new  Abstract: One of the enduring challenges in online platforms is recommendation. Both communities and advertisers face difficulties in analyzing complex user behavior, identifying habits and evolving preferences, and providing timely recommendations that are relevant to user interests. In particular, point of interest recommendation poses the additional challenge of identifying the factors that influence user behavior, capturing user tendencies, and generating accurate, personalized recommendations for each individual. In this work, we propose SOAR, a Social and Opinion Aware Recommendation model that incorporates both the user's friendship network and textual reviews to assess whether a location is suitable for a specific user, with a particular emphasis on the sentiment of shared reviews. Experimental results in real-world datasets demonstrate that incorporating reviews into the recommendation process improves the precision of the recommendation.
    
[^6]: pplx-embed-v2-late 技术报告

    pplx-embed-v2-late Technical Report

    [https://arxiv.org/abs/2610.29523](https://arxiv.org/abs/2610.29523)

    Perplexity 推出的 pplx-embed-v2-late 包含 0.6B 和 9B 两个多语言多模态后期交互模型，通过 18B 教师模型的表示蒸馏训练，在文本、图像和视觉文档检索上达到同规模最先进水平，并借助共享嵌入空间支持“9B 建索引、0.6B 低延迟查询”的非对称部署。

    

    在本报告中，我们介绍了 pplx-embed-v2-late，这是一对参数量分别为 0.6B 和 9B 的多语言、多模态后期交互（late-interaction）模型。两款模型在各自规模级别中，于一系列文本、图像和视觉文档检索基准测试上均取得了最先进的结果。在视觉文档检索任务上，pplx-embed-v2-late-0.6b 与比它大数倍的模型相比也具有竞争力。这些模型通过从一个 18B 参数的教师模型进行表示蒸馏而训练得到。因此，它们共享一个嵌入空间，支持非对称部署：可以使用 9B 模型对文档建立索引，并使用低延迟的 0.6B 模型进行查询。两款模型均可在 https://huggingface.co/collections/perplexity-ai/pplx-embed-v2 获取。

    arXiv:2610.29523v1 Announce Type: cross  Abstract: In this report, we introduce pplx-embed-v2-late, a pair of 0.6B and 9B multilingual and multimodal late-interaction models. Both models achieve state-of-the-art results within their respective size classes across a range of text, image, and visual document retrieval benchmarks. On visual document retrieval tasks, pplx-embed-v2-late-0.6b is competitive with models several times larger than itself. The models are trained via representation distillation from an 18B teacher. Consequently, they share an embedding space that supports asymmetric deployment: documents can be indexed with the 9B model and queried with the low-latency 0.6B model. Both models are available at https://huggingface.co/collections/perplexity-ai/pplx-embed-v2.
    
[^7]: X-Embedding：面向多语言专利现有技术检索的两阶段训练方法

    X-Embedding: Two-Stage Training for Multilingual Patent Prior-Art Retrieval

    [https://arxiv.org/abs/2610.29521](https://arxiv.org/abs/2610.29521)

    提出两阶段训练框架X-Embedding，先通过专利内部章节对齐与跨语言专利族对齐学习专利领域表示，再利用审查报告X类引用作为监督信号优化检索相关性，有效提升多语言专利现有技术检索中X类文献的识别能力。

    

    多语言专利现有技术检索需要识别对所要求保护的发明的新颖性或创造性（inventive step）特别相关的X类对比文献，而不仅仅是跨语言和跨文档视图的主题相似性匹配。我们提出了X-Embedding，这是一个两阶段框架：第一阶段学习专利领域的表示，第二阶段优化X类检索相关性。第一阶段通过对齐专利内部的互补章节以及不同语言专利族成员之间的对应章节，从专利结构和多语言表达中同时学习。第二阶段利用审查报告中的X类引用作为正样本，并从检索候选结果和A类引用中抽取负样本，来学习X类相关性。借助Matryoshka表示学习（MRL），单个模型即可输出64维和1024维的嵌入向量。在一个包含中文、英文和日文、并采用四种查询视图的通用数据集上进行评估，取得了89.09（原文在此处截断）的结果。

    arXiv:2610.29521v1 Announce Type: new  Abstract: Multilingual patent prior-art retrieval requires identifying X-category references that are particularly relevant to the novelty or inventive step of a claimed invention, beyond topical similarity across languages and document views. We present X-Embedding, a two-stage framework that first learns patent-domain representations and then optimizes X-category retrieval relevance. The first stage aligns complementary sections within patents and corresponding sections across different-language patent-family members, learning from both patent structure and multilingual expressions. The second stage learns X-category relevance using examination-report X citations as positives and negatives drawn from retrieval candidates and A-category citations. Matryoshka representation learning (MRL) enables 64- and 1024-dimensional embeddings within one model. Evaluation on a common Chinese, English, and Japanese collection with four query views yields 89.09
    
[^8]: 超越投票：大语言模型何时能为基于检索的表格分类增加价值？

    Beyond the Vote: When Do LLMs Add Value to Retrieval-Based Tabular Classification?

    [https://arxiv.org/abs/2610.29520](https://arxiv.org/abs/2610.29520)

    该研究在19个表格分类数据集上将大语言模型与距离加权最近邻投票进行系统比较，发现Distance-KNN整体平均准确率最高，LLM仅在检索邻居标签一致性低且测试样本距离较近或中等时才有优势。

    

    基于检索的上下文学习（ICL）使大语言模型（LLM）能够从相似示例中预测表格标签，但LLM推理是否比直接对相同示例进行投票更优仍不清楚。我们通过将四个通用LLM与距离加权最近邻投票（Distance-KNN）在19个表格分类数据集上使用完全相同的检索邻居进行比较来解答这一问题。在演示样本数量固定的情况下，扩大检索候选池可以提升所有预测器的准确率，但Distance-KNN在每个候选池规模下都取得了最高的平均准确率。更细致的分析显示，LLM的收益主要集中在检索邻居标签一致性较低的场景，且在不同模型之间存在差异。这些收益还进一步取决于测试查询在特征空间中的平均距离：在低标签一致性情况下，部分LLM在近邻和中等距离邻域中优于Distance-KNN，但……

    arXiv:2610.29520v1 Announce Type: cross  Abstract: Retrieval-based in-context learning (ICL) enables large language models (LLMs) to predict tabular labels from similar examples, but it remains unclear whether LLM inference improves on direct voting over those same examples. We address this question by comparing four general-purpose LLMs with distance-weighted nearest-neighbor voting (Distance-KNN) using identical retrieved neighbors across 19 tabular classification datasets. With the number of demonstrations fixed, enlarging the retrieval candidate pool improves accuracy for all predictors, yet Distance-KNN achieves the highest mean accuracy at every pool size. More detailed analyses reveal that LLM gains are concentrated where retrieved neighbors exhibit low label consensus and vary across models. These gains further depend on the average feature-space distance to the test query: under low label consensus, some LLMs outperform Distance-KNN in close and intermediate neighborhoods, but
    
[^9]: 面向密集检索中查询重构的方向性后验反馈融合

    Directional Posterior Feedback Fusion for Query Reformulation in Dense Retrieval

    [https://arxiv.org/abs/2610.29464](https://arxiv.org/abs/2610.29464)

    该论文提出了方向性后验反馈融合方法，通过以方向性方式融合反馈证据而非简单的逐点聚合或插值，克服了现有嵌入空间查询重构方法会抑制仅由互补证据支持的方向上查询修正的缺陷，从而提升密集检索对简短、模糊查询的处理效果。

    

    密集检索已成为现代信息检索的基础范式，支撑着跨广泛应用场景的高效语义搜索。然而，其有效性关键取决于查询嵌入能否充分捕捉用户的信息需求，而对于简短、模糊或信息不完整的查询而言，这一点往往难以实现。基于反馈的查询重构方法，如伪相关性反馈（PRF），通过纳入来自检索或生成内容的额外证据来缓解这一问题。然而，现有的嵌入空间方法通常对反馈向量进行聚合或插值，隐式地将其视为完整的目标表示。这种逐点融合方式可能会不必要地抑制查询沿单个反馈项所未表达方向的修正，即使这些方向得到了互补证据的支持。为解决这一局限，我们提出了方向性后验反馈融合（Directional Posterior Feedback Fusion）方法。

    arXiv:2610.29464v1 Announce Type: new  Abstract: Dense retrieval has become a fundamental paradigm for modern information retrieval, supporting efficient semantic search across a wide range of applications. Its effectiveness, however, relies critically on query embeddings that adequately capture users' information needs, which can be difficult for short, ambiguous, or underspecified queries. Feedback-based query reformulation approaches like pseudo-relevance feedback (PRF) alleviate this by incorporating additional evidence from retrieved or generated content. However, existing embedding-space approaches commonly aggregate or interpolate feedback vectors, implicitly treating them as complete target representations. Such pointwise fusion can unnecessarily suppress query corrections along directions not expressed by individual feedback items, even when these directions are supported by complementary evidence. To address this limitation, we propose Directional Posterior Feedback Fusion (D
    
[^10]: 生成式推荐系统遗忘了长尾：在检索与排序中恢复记忆能力

    Generative Recommenders Forget the Long Tail: Restoring Memorization in Retrieval and Ranking

    [https://arxiv.org/abs/2610.29298](https://arxiv.org/abs/2610.29298)

    提出免训练插件CAPE，利用物品共现记忆机制修复生成式推荐系统中长尾物品在检索与排序两个阶段被系统性遗忘的问题。

    

    生成式推荐系统用单一模型取代了传统的“先检索后排序”级联流程：通过对语义物品标识符进行束搜索来完成检索，并使用相同的似然函数进行排序。我们证明这种共享评分机制会导致模型在两个环节遗忘长尾物品。在检索阶段，束搜索依据标识符前缀子树的概率质量来保留候选前缀，因此长尾物品被过早剪枝，即使加宽束宽也只能召回少量长尾物品。在排序阶段，即使长尾物品被召回，似然函数也会将其排在头部物品之后，因此仅扩大检索范围几乎无法带来收益。这两种失效都源于物品级记忆能力（即直接回忆哪些物品共同出现的能力）的缺失——协同过滤具备这种能力，而共享前缀生成则缺乏。为此，我们提出CAPE（共现感知池扩展，Co-occurrence-Aware Pool Expansion），这是一种免训练的即插即用模块，能够在两个阶段恢复记忆能力：(i) 一个记忆通道用于检索由共现支持的长尾……（原文摘要在此处截断）

    arXiv:2610.29298v1 Announce Type: new  Abstract: Generative recommenders replace the retrieve-then-rank cascade with a single model: beam search over semantic item identifiers performs retrieval, and the same likelihood performs ranking. We show that this shared score makes them forget long-tail items twice. In retrieval, beam search keeps identifier prefixes by the probability mass of their subtrees, so tail items are pruned early, and a wider beam admits few more of them. In ranking, the likelihood places tail items below head items even when they are admitted, so expanding retrieval alone brings little or no gain. Both failures reflect missing item-level memorization (i.e., direct recall of which items co-occur), which collaborative filtering provides and shared-prefix generation lacks. To this end, we propose CAPE (Co-occurrence-Aware Pool Expansion), a training-free plug-in that restores memorization at both stages: (i) a memorization channel retrieves co-occurrence-supported tail
    
[^11]: 通过可观测性扩展FAIR原则以实现可信联邦学习

    Extending FAIR Principles with Observability for Trustworthy Federated Learning

    [https://arxiv.org/abs/2610.29252](https://arxiv.org/abs/2610.29252)

    提出FAIR-O框架，将FAIR原则扩展出可观测性维度，使每个参与者在每轮联邦学习中生成最小化的签名信任证据（可由本地TinyLLM辅助解释），在不迁移原始数据的前提下建立三级信任模型以实现可信联邦学习。

    

    联邦学习（FL）将数据保留在其源头，但仅靠数据的本地化并不能使联邦学习值得信赖。在缺乏足够可观测性的情况下，聚合器无法判断客户端服务是否健康、本地数据流水线是否可信，或合同约束在训练过程中是否得到妥善监控。我们的立场是：可信的联邦学习除了模型更新过程之外，还需要一个“证据平面”。每个参与者持续地将本地基础设施和数据总结为与每一轮联邦学习相关联的最小化的、经签名的信任证据，而原始数据和日志仍保留在本地。本地的TinyLLM（小型语言模型）可以协助解释可观测性信号、推导上下文证据，并向运维人员提供自然语言解释。我们通过FAIR-O实现了这一愿景，FAIR-O是FAIR原则的可观测性扩展。该方法不以基准测试联邦学习的准确率为重点，而是提出了一个三级信任模型以及一个双参与者的Python实现……

    arXiv:2610.29252v1 Announce Type: new  Abstract: Federated learning (FL) keeps data at its source, but locality alone does not make a federation trustworthy. Without sufficient observability, an aggregator cannot determine whether the client service was healthy, the local data pipeline was trustworthy, or contractual constraints were properly monitored during training. We take the position that trustworthy FL needs an evidence plane in addition to its model-update process. Each participant continuously summarizes local infrastructure, data into minimal, signed trust evidence associated with each FL round, while raw data, logs remain local. A local TinyLLM may assist in interpreting observability signals, deriving contextual evidence, and providing natural-language explanations to operators. We instantiate this vision through FAIR-O, an observability extension of FAIR principle. Rather than benchmarking FL accuracy, a three-level trust model has been proposed and a two-participant Pytho
    
[^12]: 面向RAG预算约束证据选择的边际视角效用学习

    Learning Marginal Viewpoint Utility for Budgeted Evidence Selection in RAG

    [https://arxiv.org/abs/2610.29234](https://arxiv.org/abs/2610.29234)

    提出轻量级排序器LMVU，通过学习候选证据为当前选择新增的边际视角覆盖来动态优先排序证据，从而在固定上下文预算下最大化多视角查询RAG的视角覆盖效果。

    

    在多视角查询上的检索增强生成（RAG）必须在固定的上下文预算下，从检索到的证据中保留不同的答案和立场。我们提出了LMVU，一个轻量级的排序器，它学习根据候选证据为当前选择所增加的边际视角覆盖来对其进行优先级排序。训练时使用每个基准测试开发集中的视角标注，在同一选择状态下对候选进行比较。在推理阶段，随着上下文的增长，LMVU反复对符合预算的候选进行评分，而无需访问这些标注。我们评估了两种恢复视角覆盖的形式：RVC₁在某个视角至少有一个支持来源单元时给予满分，而RVC₂在仅有一个支持来源单元时给予一半分数，在有两个或更多不同来源单元时给予满分。精确的oracle（理论最优解）用于在相同的证据池和预算下测量可实现的覆盖。在四个带视角标注的基准测试上，在1,02（摘要原文在此处截断）……

    arXiv:2610.29234v1 Announce Type: new  Abstract: Retrieval-augmented generation over multi-viewpoint queries must preserve distinct answers and positions from the retrieved evidence while operating under a fixed context budget. We introduce LMVU, a lightweight ranker that learns to prioritize candidates by the marginal viewpoint coverage they add to the current selection. Training uses viewpoint annotations from each benchmark's development split to compare candidates within the same selection state. At inference, LMVU repeatedly scores budget-feasible candidates as the context grows, without accessing these annotations. We evaluate two forms of recovered viewpoint coverage: RVC$_1$ assigns each viewpoint full credit for at least one supporting source unit, while RVC$_2$ assigns half credit for one and full credit for two or more distinct source units. Exact oracles measure achievable coverage under the same evidence pool and budget. Across four viewpoint-annotated benchmarks at a 1,02
    
[^13]: 偏好不等于许可：代理式商务中个性化自动购买边界的确定

    Preference Is Not Permission: Personalizing the Auto-Purchase Line in Agentic Commerce

    [https://arxiv.org/abs/2610.29216](https://arxiv.org/abs/2610.29216)

    该论文提出将学习个人化的“自动购买边界”作为代理式商务中的新任务，并发现十六个指令微调模型普遍“越权”——即使商品匹配度低、购买依据缺失也会自动下单，与人类委托者的授权意愿存在系统性错位，且这一边界因人而异、需要个性化建模。

    

    个性化购物基准关注的是哪件商品适合用户；我们则提出另一个问题：智能体可以在该商品上被允许承诺到什么程度，并将学习个人的“自动购买线”作为一项任务提出。这条边界以证据为条件且具有三值性：是否做出购买承诺是否恰当，取决于商品信息的充分性（购买依据 warrant）、可逆性、价格是否符合用户的规范以及紧迫性，而不仅仅取决于匹配程度；其自然输出为“不购买”、“先确认”或“自动购买”三个层级。在十六个指令微调模型上的实验表明，模型的自主承诺行为与人类授权参考存在系统性错位——在该参考中，十五名标注者扮演给定用户画像中的委托方，选择他们在无需确认的情况下所允许的最远行动：即便匹配度很低，即便模型自身的证据标志显示购买依据缺失，模型仍会做出购买承诺。这是一种越权行为。这条边界因人而异：这些评分者之间存在分歧，但分歧是结构化的。

    arXiv:2610.29216v1 Announce Type: cross  Abstract: Personalized shopping benchmarks ask which item fits the user; we ask how far an agent may commit on that item, and pose learning the individual's auto-purchase line as a task. The boundary is evidence-conditioned and three-valued: whether committing is appropriate depends on the sufficiency of product information (warrant), reversibility, price against the user's norm, and urgency, not only on fit; and the natural output is don't buy, confirm first, or buy automatically. Across sixteen instruction-tuned models, autonomous commitment is misaligned with a human authorization reference in which fifteen annotators, role-playing the delegating principal from a given profile, choose the furthest action they would allow without confirmation: models commit even when fit is low and when their own evidence flag reports the warrant missing. This is overreach. The line is individual: these raters disagree, yet the disagreement is structured and p
    
[^14]: 推理-搜索增强的大语言模型：一种操作分类体系、基准分析与受控证据综合

    Reasoning-Search-Augmented LLMs: An Operational Taxonomy, Benchmark Analysis and Controlled Evidence Synthesis

    [https://arxiv.org/abs/2610.29189](https://arxiv.org/abs/2610.29189)

    该论文提出了一个针对推理-搜索增强大语言模型（LLMs）的操作性分类体系，通过检索耦合、控制器设计、知识来源和接地机制四个架构维度系统性地分析将逐步推理与交互式搜索紧密耦合的框架，并提供了基准分析与受控证据综合。

    

    大语言模型（LLMs）展现出令人印象深刻的涌现推理能力，但它们仍受限于固定的模型参数和有限的上下文窗口。即使通过静态检索进行增强，它们也可能产生不受检索证据支持、甚至与检索证据相矛盾的陈述。这导致了幻觉问题、难以融入最新信息，以及无法正确解读或协调不同信息来源的失败。近期的工作通过将逐步推理与交互式搜索紧密耦合来解决这些局限性，将检索视为迭代式智能体循环中的一个动作，而非一次性的预处理步骤。本研究回顾了通过交错执行规划、查询、检索、整合、验证和停止等环节来实现此类推理-搜索循环的框架。我们通过四个架构维度对这些系统进行分析：检索耦合方式、控制器设计、知识来源与接地机制。

    arXiv:2610.29189v1 Announce Type: cross  Abstract: Large language models (LLMs) demonstrate impressive emergent reasoning capabilities, yet they remain constrained by fixed model parameters and finite context windows. Even when augmented with static retrieval, they can produce statements that are unsupported by, or even contradictory to, retrieved evidence. This leads to hallucinations, difficulty incorporating up-to-date information, and failures to correctly interpret or reconcile sources. Recent work addresses these limitations by tightly coupling step-by-step reasoning with interactive search, treating retrieval as an action within an iterative agentic loop rather than a one-shot preprocessing step. This study reviews frameworks that implement such reasoning-search loops by interleaving planning, querying, retrieval, integration, verification, and stopping. We analyze these systems through four architectural axes: retrieval coupling, controller design, knowledge source, and groundi
    
[^15]: 面向CTR预测的从内容到相关性再到个性化的渐进式学习

    Progressive Learning from Content to Relevance to Personalization for CTR Prediction

    [https://arxiv.org/abs/2610.29138](https://arxiv.org/abs/2610.29138)

    提出FineSID渐进式学习框架，通过共享的细粒度语义ID表示统一内容分词、查询-物品相关性语义预训练和个性化后训练，有效解决多模态CTR预测中细粒度异构内容与ID预测器融合的难题。

    

    多模态点击率（CTR）预测是搜索系统的核心，其排序取决于物品内容、查询相关性和用户偏好。现有方法通过稠密嵌入或粗粒度离散ID来融入内容信息，然而如何有效地将细粒度异构内容与基于ID的预测器相结合仍然具有挑战性。我们提出了FineSID，这是一个渐进式学习框架，通过共享的细粒度离散表示，统一了内容分词、查询-物品语义预训练和个性化后训练。每个阶段都优化一个明确的目标，同时保留对内容的token级访问。首先，预训练的多尺度视觉分词器和文本分词器将异构内容转换为细粒度的语义ID（SIDs），并将其与ID特征显式对齐。其次，在查询-物品相关性监督下进行全参数语义预训练，建立鲁棒的匹配先验。

    arXiv:2610.29138v1 Announce Type: new  Abstract: Multimodal click-through rate (CTR) prediction is central to search systems, where ranking depends on item content, query relevance, and user preference. Existing approaches incorporate content through dense embeddings or coarse discrete IDs, yet effectively integrating fine-grained heterogeneous content with ID-based predictors remains challenging. We propose FineSID, a progressive learning framework that unifies content tokenization, query--item semantic pretraining, and personalized post-training via a shared fine-grained discrete representation. Each stage optimizes an explicit objective while preserving token-level access to content. First, a pretrained multi-scale visual tokenizer and text tokenizers convert heterogeneous content into fine-grained Semantic IDs (SIDs), explicitly aligning them with the ID features. Second, full-parameter semantic pretraining under query--item relevance supervision establishes a robust matching prior
    
[^16]: 我们自己的Jev：单次前向传播中的类型化决策，从自研裁判到开放LoRA模型

    Our Own Jev: Typed Decisions in One Forward Pass, from an In-House Judge to Open LoRA Models

    [https://arxiv.org/abs/2610.29052](https://arxiv.org/abs/2610.29052)

    该论文提出用单次前向传播的“类型化决策”接口（直接读取logits或轻量指针头）替代生成式LLM裁判，其自研System 1裁判仅用一块GPU约3分钟、约0.06美元即可判定1,000个智能体编码会话，以约1%的成本取代每轮约1,000美元且标签一致性仅26.9%的前沿LLM裁判。

    

    类型化决策将证据、问题和一个固定的选项集映射为每个选项的概率，只需一次前向传播即可完成，方式是读取语言模型的logits或在其隐藏状态上附加一个小型头部，而不是生成并解析文本。商业系统Jev推广了这种System 1接口，开源复现也随之出现。我们报告了该模式的两种用途。第一部分描述了一个自研的System 1裁判，它为一个训练语料库筛选了约14.8万条智能体编码会话记录。它取代了一个前沿LLM裁判——后者每轮处理的成本约在1,000美元量级，且在两位裁判达成一致的会话中，其存储的标签与新做的双裁判共识仅有26.9%吻合；一个自托管的冻结骨干网络，配合秩为16的LoRA适配器和指针头，可在一次前向传播中回答三个类型化问题，在一块GPU上约三分钟内即可判定1,000个会话，按参考价格计算成本约0.06美元，大约仅为前沿模型成本的1%……

    arXiv:2610.29052v1 Announce Type: new  Abstract: A typed decision maps evidence, a question and a fixed set of options to a probability for every option, computed in one forward pass, by reading a language model's logits or a small head over its hidden states, instead of generating and parsing text. The commercial system Jev popularised this System~1 interface, and open reproductions have followed. We report two uses of the pattern. \emph{Part I} describes an in-house System~1 judge that triages about 148K agentic coding session records for a training corpus. It replaces a frontier LLM judge that cost on the order of \$1,000 per pass and whose stored labels matched a fresh two-judge consensus on only 26.9\% of the sessions where the two judges agreed; a self-hosted frozen backbone with a rank-16 LoRA adapter and a pointer head answers three typed questions in one forward pass, judges 1,000 sessions in about three minutes of one GPU, about \$0.06 at a reference rate and roughly 1\% of t
    
[^17]: Scriptorium：多人类、多AI协作的数字图书馆搜索平台的系统设计与实现

    Scriptorium: System Design and Implementation of a Multi-Human, Multi-AI Collaborative Digital Library Search Platform

    [https://arxiv.org/abs/2610.28929](https://arxiv.org/abs/2610.28929)

    该论文提出了Scriptorium，一个支持多人类与多AI协作的数字图书馆搜索平台，其核心创新在于将完全跨用户透明的搜索活动与由七个角色专门化生成式AI智能体组成的管理者-工作者架构相结合，并完整公开系统实现细节以支持复现与重用。

    

    本技术报告记录了Scriptorium的设计与实现，这是一个旨在支持多人类、多AI协作的数字图书馆搜索界面：实现了搜索活动的完全跨用户透明性，并结合了由七个角色专门化的生成式AI智能体组成的“管理者-工作者”（Manager-Worker）架构。一篇配套的简短演示论文描述了Scriptorium的交互设计；由于演示格式受限于几页篇幅，必然省略了大部分实现细节。本报告记录了Scriptorium的完整系统架构、数据模型、协作交互设计原理以及AI编排机制，包括智能体提示词原文、约束这些提示词的确定性防护机制、语言模型与检索配置，以及运行系统所需的步骤，以便其他研究人员能够检查、批评、复现和重用该设计。我们还讨论了该系统（原文在此处截断）。

    arXiv:2610.28929v1 Announce Type: cross  Abstract: This technical report documents the design and implementation of Scriptorium, a digital library search interface designed to support multi-human, multi-AI collaboration: full cross-user transparency of search activity combined with a Manager--Worker architecture of seven role-specialized generative AI agents. A companion short demonstration paper describes Scriptorium's interaction design; because a demonstration format is constrained to a few pages, it necessarily omits most implementation detail. This report documents Scriptorium's complete system architecture, data model, collaborative-interaction design rationale, and AI orchestration mechanism, including verbatim agent prompts, the deterministic guardrails that constrain them, the language model and retrieval configuration, and the steps needed to run the system, so that the design can be inspected, critiqued, reproduced, and reused by other researchers. We also discuss the system
    
[^18]: 面向片段标注的受限标记方法

    Constrained Tagging for Span Labeling

    [https://arxiv.org/abs/2610.28922](https://arxiv.org/abs/2610.28922)

    提出了一种无需训练的受限标记方法，通过字节级词元trie和有限状态机约束模型逐字复现输入并内联标注片段，从而保证片段标注输出与输入位置精确对应，且几乎不产生失败。

    

    生成式大语言模型无需任务特定训练即可对文本片段进行标注，但其输出并未与输入中的位置绑定：返回的片段可能不在输入中出现，或者出现多次。我们提出了“受限标记”（constrained tagging），这是一种无需训练的方法，让模型逐字复现输入，并用带标签的内联标记包裹每个片段。一个使用字节级词元trie与有限状态机的logits处理器，只允许那些逐字延续输入或构成有效标签的词元，因此输出能够精确复现输入，且标签来自给定集合，片段可以在任意字符边界处开始或结束，甚至可以在词元内部。两阶段解码使推理模型能够在约束生效之前先进行推理。在六个数据集和六个开源权重模型上，该保证在255,312次受限生成中仅有0.022%的失败，每次失败都是因为推理模型耗尽了词元预算或从未结束其推理过程，且该约束没有增加可测量的

    arXiv:2610.28922v1 Announce Type: new  Abstract: Generative LLMs can label spans without task-specific training, but their output is not tied to positions in the input: a returned span may be absent from the input or occur several times. We present constrained tagging, a training-free method in which the model reproduces the input and wraps each span in labeled inline markup. A logits processor using a byte-level token trie and a finite-state machine admits only tokens that continue the input verbatim or form a valid tag, so the output reproduces the input exactly with labels from the given set, and spans may begin or end at any character boundary, even inside a token. Two-phase decoding lets reasoning models reason before the constraint engages. On six datasets and six open-weight models, the guarantee holds in all but 0.022% of 255,312 constrained generations, each failure a reasoning model exhausting its token budget or never closing its reasoning, and the constraint adds no measura
    
[^19]: FutureLens：学术文本与未来工作参考文献中的纵向主题模式

    FutureLens: Longitudinal Topic Patterns in Scholarly Text and Future-Work References

    [https://arxiv.org/abs/2610.28656](https://arxiv.org/abs/2610.28656)

    FutureLens 基于 2012–2022 年 5,655 篇 ACL 论文，系统考察了利用“未来工作”陈述追踪学术领域主题演变时的测量方法问题，并比较了 NMF 与 LDA 两种主题模型在稳定性、信息量与关键词多样性上的表现差异。

    

    arXiv:2610.28656v1 公告类型：cross 摘要：未来工作陈述记录了研究者选择留待日后解决的问题。然而，要用它们来描述一个研究领域，需要将主题的演变与文档覆盖范围、主题分配以及学术写作风格的变化区分开来。FutureLens 在一个恢复的快照数据集上考察了这些测量问题，该数据集包含 2012 至 2022 年间发表的 5,655 篇 ACL 论文，其中 4,005 篇附有未来工作参考文本。我们比较了非负矩阵分解（NMF）与潜在狄利克雷分配（LDA）两种方法，在论文层面聚合主题出现情况，并考察了归一化流行度、探索性趋势统计、分配稳定性，以及输入文本派生的主题与参考文献之间的契合度。在不同的随机种子下，NMF 具有更高的平均归一化逐点互信息（0.1387 对比 0.1280）和更高的成对分配稳定性，而 LDA 则具有更大的关键词多样性（0.8900 对比 0.7750）。种子为零的结果显示流行度呈上升趋势……

    arXiv:2610.28656v1 Announce Type: cross  Abstract: Future-work statements offer a record of the questions researchers choose to leave open. Using them to describe a field, however, requires separating thematic change from changes in document coverage, topic assignment, and scholarly writing. FutureLens examines these measurement issues in a recovered snapshot of 5,655 ACL papers published during 2012--2022, including 4,005 with supplied future-work reference text. We compare non-negative matrix factorization (NMF) and latent Dirichlet allocation (LDA), aggregate topic presence at the paper level, and examine normalized prevalence, exploratory trend statistics, assignment stability, and agreement between input-derived text and references. Across available seeds, NMF has higher mean normalized pointwise mutual information (0.1387 versus 0.1280) and higher pairwise assignment stability, while LDA has greater keyword diversity (0.8900 versus 0.7750). Seed-zero results show increasing preva
    
[^20]: 学位论文知识图谱构建：综述研究

    Building Knowledge Graphs for Theses: A Survey

    [https://arxiv.org/abs/2610.28648](https://arxiv.org/abs/2610.28648)

    本文系统综述了2021至2025年间46项关于学位论文知识图谱构建的同行评审研究，提出了技术、模型与工具的结构化分类体系，并揭示了从自然语言处理抽取技术到大语言模型驱动增强的方法论演进。

    

    学术学位论文是科学知识的宝贵来源，但其语义层面的开发利用仍然有限。尽管学术知识图谱已在科学出版物的表示方面取得显著进展，但针对学术学位论文的知识图谱构建却受到相对较少的关注。本文对现有方法进行了系统性综述，并分析了它们在学位论文领域的潜在应用。我们分析了2021年至2025年间发表的46项同行评审研究，并将现有方法归纳为技术、模型和工具的结构化分类。该综述揭示了方法论的演进历程：从用于实体和关系抽取的自然语言处理技术，发展到本体工程、语义网技术，以及通过知识图谱嵌入、关联开放数据和大语言模型实现的AI驱动增强。这些发现为该领域的研究现状提供了整合性的视角。

    arXiv:2610.28648v1 Announce Type: cross  Abstract: Academic theses are a valuable source of scientific knowledge, yet their semantic exploitation remains limited. While Scholarly Knowledge Graphs have significantly advanced the representation of scientific publications, knowledge graph construction for academic theses has received comparatively limited attention. This paper provides a systematic review of existing approaches and analyzes their potential application for the thesis domain. We analyzed 46 peer-reviewed studies published between 2021 and 2025, and organised existing approaches into a structured classification of techniques, models, and tools. The review reveals a methodological evolution from natural language processing techniques for entity and relation extraction to ontology engineering, Semantic Web technologies, and AI-driven enrichment through knowledge graph embeddings, Linked Open Data, and Large Language Models. The findings provide a consolidated view of the curre
    
[^21]: 基于引导式Token精炼的可控生成式推荐

    Controllable Generative Recommendation via Guided Token Refinement

    [https://arxiv.org/abs/2610.28642](https://arxiv.org/abs/2610.28642)

    ReGen将生成式推荐从自回归的顺序解码转变为基于掩码填充的并行迭代token精炼，无需重新训练即可直接对高层语义token施加控制信号，同时保证解码候选映射到真实物品。

    

    基于离散语义ID（Semantic IDs）的生成式推荐已成为物品检索的一种强大范式。然而，现实世界中的系统经常需要实时可控性，以在准确性与多样性等目标之间取得平衡。现有的自回归方法严格地从左到右生成token，这种固定方向锁定了早期决策，无法进行整体规划，也使得控制变得困难。因此，直接干预往往会破坏语义结构，生成不存在的物品。为解决这一问题，我们提出了ReGen，一个从顺序解码转向并行迭代精炼的框架。ReGen将生成重新表述为掩码填充（mask-filling）任务，同时预测所有token而非逐个生成。这种全局视角使我们能够将控制信号直接作用于具有高层语义的token，而无需重新训练模型。为了确保解码出的候选能够映射到真实物品，我们引入了一种top-k……（摘要在此处截断）

    arXiv:2610.28642v1 Announce Type: new  Abstract: Generative recommendation over discrete Semantic IDs has become a powerful paradigm for item retrieval. Real-world systems, however, frequently require real-time controllability to balance accuracy against objectives such as diversity. Existing autoregressive methods generate tokens strictly from left to right. This fixed direction locks in early decisions, preventing holistic planning and making control difficult. Consequently, direct interventions often break the semantic structure, producing non-existent items. To address this, we propose ReGen, a framework that shifts from sequential decoding to parallel iterative refinement. ReGen reformulates generation as a mask-filling task, predicting all tokens simultaneously rather than one by one. This global view allows us to apply control signals directly to tokens with high-level semantics without retraining the model. To ensure that decoded candidates map to real items, we introduce a top
    
[^22]: 面向金融交易数据检索增强生成的交易基础模型：与文本嵌入的对比评估

    Transaction Foundation Models for Retrieval-Augmented Generation over Financial Transaction Data: A Comparative Evaluation with Text Embeddings

    [https://arxiv.org/abs/2610.28641](https://arxiv.org/abs/2610.28641)

    本研究在IBM合成反洗钱交易数据集上构建了1,026个问题的问答基准，证明NVIDIA的2900万参数交易基础模型（TFM）在金融交易数据检索增强生成任务中的检索性能和端到端答案准确率均大幅超越同规模的通用文本嵌入模型。

    

    面向特定领域的表格与交易基础模型日益被提出作为金融应用的即插即用表示学习器，但相对于现成的文本嵌入模型，它们在结构化数据检索增强生成（RAG）中的价值仍未得到充分检验。我们在IBM的合成反洗钱（AML）交易数据集上构建了一个确定性的、由模板生成的问答基准，包含1,026个问题并带有检索级别的金标准文档标注，并在大小和数据匹配的检索设置下，将NVIDIA的2900万参数交易基础模型（TFM）与五个通用及金融领域的文本嵌入模型进行比较，同时与人工聚合的摘要文本基线进行对比。TFM在检索性能（Recall@5：0.81 对比 0.25-0.39）和端到端答案准确率（66.7% 对比 27.5%）方面均显著优于所有大小匹配的通用嵌入模型。然而，一个简单的流水线……

    arXiv:2610.28641v1 Announce Type: new  Abstract: Domain-specific tabular and transaction foundation models are increasingly proposed as drop-in representation learners for financial applications, but their value relative to off-the-shelf text embedding models for retrieval-augmented generation (RAG) over structured data remains underexamined. We construct a deterministic, template-generated question-answering benchmark of 1,026 questions with retrieval-level gold-document annotations over IBM's synthetic Anti-Money-Laundering (AML) transaction dataset, and compare NVIDIA's 29M-parameter Transaction Foundation Model (TFM) against five general-purpose and financial-domain text embedding models under a size- and data-matched retrieval setup, as well as against a hand-aggregated summary-text baseline. TFM substantially outperforms every size-matched generic embedder on both retrieval (Recall@5: 0.81 vs. 0.25-0.39) and end-to-end answer accuracy (66.7% vs. 27.5%). However, a simple pipeline
    
[^23]: AI聊天能否取代传统搜索？来自基于RAG的现场活动实验的证据

    Can AI Chat Replace Traditional Search? Evidence from a RAG-based Live Events Experiment

    [https://arxiv.org/abs/2610.28639](https://arxiv.org/abs/2610.28639)

    通过一项预注册的现场活动搜索实验发现，基于RAG的纯AI聊天界面搜索表现最差，尤其在开放式探索场景中，表明生成式AI对话界面尚不能取代传统搜索方式。

    

    对话式AI工具正被越来越多地整合到在线搜索和浏览界面中。为了解此类工具如何影响用户行为和搜索结果，我们开展了一项预注册的、有激励机制的在线实验，参与者在从某大型现场活动平台抓取的数千个活动目录中进行搜索。我们比较了三种界面：经典的基于属性筛选的界面、生成式AI聊天界面以及两者兼备的混合界面。在两种AI条件中，AI输出均通过检索增强生成（RAG）进行限制。我们的主要结果是：相对于另外两种条件，聊天条件的表现最差。该效应在不同AI模型（gpt-4.1-mini、gpt-5.2和gpt-5.4）下均保持稳健，且主要由进行开放式探索的用户（相对于搜索特定歌手、场馆或运动队的用户）所驱动。此外，没有显著的差异……

    arXiv:2610.28639v1 Announce Type: new  Abstract: Conversational AI tools are being increasingly integrated into online search and browsing interfaces. To understand how such tools affect user behavior and search outcomes, we run a pre-registered, incentivized online experiment in which participants search a catalog of several thousand events scraped from a major live-events platform. We compare three interfaces: a classic, attribute-based filter interface (Filter), a GenAI chat interface (Chat), and a combined interface offering both (Hybrid). In both AI conditions, AI output is restricted via retrieval-augmented generation (RAG). Our main result is that, relative to the other two conditions, the Chat condition performs the worst. The effect is robust across AI models (gpt-4.1-mini, gpt-5.2 and gpt-5.4) and is driven mainly by users who perform open-ended exploration (as opposed to users searching for a specific artist, venue or sports team). Furthermore, there are no significant diffe
    
[^24]: TAGR：面向异构城市配送任务的任务自适应地理语义检索

    TAGR: Task-Adaptive Geo-Semantic Retrieval across Heterogeneous Urban Delivery Tasks

    [https://arxiv.org/abs/2610.28638](https://arxiv.org/abs/2610.28638)

    提出TAGR框架，通过任务自适应的地理语义检索机制，为城市配送中需求预测、ETA预测和下一站预测三类异构任务检索结构与粒度各异的历史证据，解决单一融合机制导致的匹配不当与负迁移问题，提升预测精度与资源分配效率。

    

    随着城市居民对高质量即时配送服务需求的不断提升，需求预测、预计到达时间（ETA）预测和下一站预测已成为城市配送系统中的关键任务。然而，这三项任务所依赖的历史证据在结构和粒度上存在显著差异。同时，它们的预测目标在输出形式上也各不相同，这使得难以通过单一的融合机制来整合检索到的证据，否则会造成匹配不当或负迁移。这些问题会增加预测误差，进而导致资源配置失衡——部分地区的需求得不到满足，而另一些地区则出现资源利用率不足的情况，最终降低整体效率和服务质量。为应对这些挑战，我们提出了TAGR，一个面向异构城市配送任务的任务自适应地理语义检索框架。TAGR构建了一个基于大语言模型（LLM）的地理语义共享……（原文摘要在此处截断）

    arXiv:2610.28638v1 Announce Type: cross  Abstract: As urban residents demand higher-quality instant delivery services, demand forecasting, estimated time of arrival prediction, and next-stop prediction have become critical tasks in urban delivery systems. However, the three tasks rely on historical evidence with substantially different structures and granularities. Meanwhile, their prediction targets differ in output form, making it difficult to integrate retrieved evidence through a single fusion mechanism without causing mismatch or negative transfer. These issues can increase prediction errors and subsequently lead to imbalanced resource allocation, where some regions experience unmet demand. In contrast, others suffer from underutilization, ultimately degrading overall efficiency and service quality. To address these challenges, we propose TAGR, a Task-Adaptive Geo-Semantic Retrieval framework across heterogeneous urban delivery tasks. TAGR constructs an LLM-based geo-semantic shar
    
[^25]: ResearchHub：面向生物医学预印本的个性化文献发现系统

    RESEARCHHUB: A Personalized Literature Discovery System for Biomedical Preprints

    [https://arxiv.org/abs/2610.28636](https://arxiv.org/abs/2610.28636)

    本文提出ResearchHub个性化文献发现系统，其核心检索器prism通过GPT-4o生成查询和混合训练目标适配生物医学预印本，在MedRxiv语料上将Recall@10较SPECTER2-Search提升12.4个百分点，并支持可控新颖性、多样性与课程式排序。

    

    我们提出了ResearchHub，一个面向生物医学预印本的交互式个性化文献发现系统。给定研究人员画像或自由格式的查询，ResearchHub从特定领域的MedRxiv语料库中检索相关论文，并通过可控的新颖性、多样性和课程式排序，帮助用户超越近似重复的推荐。系统的核心是prism，一个针对生物医学预印本适配的双编码器检索器，它使用GPT-4o生成的自然语言查询，以及结合交叉编码器软标签的混合式多负例排序加MarginMSE训练目标。在2023至2025年的44,937篇MedRxiv论文上，prism超越了强大的现成密集检索器，相比SPECTER2-Search将Recall@10提升了12.4个百分点。ResearchHub还进一步支持基于用户阅读历史的前沿检索、基于行列式点过程（DPP）的多样性重排序，以及从主流到专业领域的局部内在维度排序。

    arXiv:2610.28636v1 Announce Type: cross  Abstract: We present ResearchHub, an interactive system for personalized literature discovery over biomedical preprints. Given a researcher profile or a free-form query, ResearchHub retrieves relevant papers from a domain-specific MedRxiv corpus and helps users move beyond near-duplicate recommendations through controllable novelty, diversity, and curriculum ordering. At the core of the system is prism, a bi-encoder retriever adapted to biomedical preprints using GPT-4o-generated natural language queries and a hybrid Multiple Negatives Ranking plus MarginMSE objective with cross-encoder soft labels. On 44,937 MedRxiv papers from 2023--2025, prism outperforms strong off-the-shelf dense retrievers, improving Recall@10 over SPECTER2-Search by 12.4 percentage points. ResearchHub further supports frontier retrieval based on a user's reading history, DPP-based diversity re-ranking, Local Intrinsic Dimensionality ordering from mainstream to specialized
    
[^26]: DuoLM：基于语言模型的快速高效个性化推荐

    DuoLM: Fast and Effective Personalized Recommendation with Language Models

    [https://arxiv.org/abs/2610.24481](https://arxiv.org/abs/2610.24481)

    DuoLM框架通过编码器LLM离线学习紧凑物品表示、个性化排序器LLM融合用户历史与反馈，并借助虚拟token ID、共享历史计算和CUDA Graph重放等运行时加速技术，实现了与专家教师模型相当的推荐质量，同时大幅提升服务吞吐量。

    

    语言模型能够基于丰富的物品内容、用户历史和反馈进行个性化推荐。然而，使用全文输入评估大量候选物品会带来巨大的在线计算开销，并限制服务吞吐量。我们提出了DuoLM，一个快速且有效的个性化推荐框架，它将紧凑的推荐模型与高效的推理运行时相结合。其中，编码器LLM学习紧凑的物品表示以供离线复用，而个性化排序器LLM则将这些表示与用户历史和反馈相结合来完成推荐。我们通过推荐监督和教师蒸馏对两个模型进行端到端训练。运行时系统还通过用于嵌入访问的虚拟token ID、共享历史计算和CUDA Graph重放进一步加速推理。基准测试结果表明，我们的框架实现了与专家教师模型相当的推荐质量。在单个H100上，它可以提供（摘要在此处截断）

    arXiv:2610.24481v2 Announce Type: replace  Abstract: Language models can make personalized recommendations from rich item content, user history, and feedback. However, evaluating many candidates with full-text inputs incurs substantial online computation and limits serving throughput. We introduce DuoLM, a framework for fast and effective personalized recommendation that integrates a compact recommendation model with an efficient serving runtime. An encoder LLM learns compact item representations for offline reuse, while a personalized ranker LLM combines them with user history and feedback for recommendation. We train both models end to end with recommendation supervision and teacher distillation. The runtime further accelerates inference through virtual token IDs for embedding access, shared history computation, and CUDA Graph replay. Benchmark results show that our framework achieves recommendation quality comparable to that of the expert teacher model. On a single H100, it delivers
    
[^27]: WeGR：微信工业搜索系统的统一生成式检索

    WeGR: Unified Generative Retrieval for Industrial Search Systems in WeChat

    [https://arxiv.org/abs/2610.16979](https://arxiv.org/abs/2610.16979)

    WeGR提出统一生成式检索框架，通过融合协同信号且避免码本坍缩的CoCo-VAE分词器，并解决强化学习中精确SID匹配带来的奖励稀疏问题，显著提升工业搜索的检索质量。

    

    生成式检索（GR）通过在自回归框架内统一候选生成过程，为工业搜索提供了一种有前景的范式。然而，其有效性受到两个关键挑战的限制。首先，现有的语义ID（SID）分词器难以在融入协同信号的同时保持有效的码本利用率，限制了其区分物品和实现高质量检索的能力。其次，现有的强化学习（RL）方法在精确SID匹配下存在严重的奖励稀疏问题，导致许多采样轨迹缺乏有效的学习信号，阻碍了偏好优化。为了解决这些挑战，我们提出了WeGR，一个由两个互补组件构成的统一生成式检索框架。具体而言，协同无坍缩分词器在缓解码本坍缩的同时，融入了查询到物品和物品到物品的协同信号……

    arXiv:2610.16979v1 Announce Type: cross  Abstract: Generative retrieval (GR) offers a promising paradigm for industrial search by unifying candidate generation within an autoregressive framework. However, its effectiveness is limited by two key challenges. First, existing Semantic ID (SID) tokenizers struggle to incorporate collaborative signals while maintaining effective codebook utilization, limiting their ability to distinguish items and achieve high-quality retrieval. Second, existing reinforcement learning (RL) methods suffer from severe reward sparsity under exact SID matching, leaving many sampled trajectories without effective learning signals and hindering preference optimization. To address these challenges, we propose WeGR, a unified generative retrieval framework consisting of two complementary components. Specifically, the Collaborative and Collapse-free Tokenizer (CoCo-VAE) incorporates query-to-item and item-to-item collaborative signals while mitigating codebook collap
    
[^28]: 面向金融文档问答的令牌预算升级机制：成本可预测，收益才是瓶颈

    Token-Budgeted Escalation for Financial Document QA: Cost Is Predictable, Benefit Is the Bottleneck

    [https://arxiv.org/abs/2610.07760](https://arxiv.org/abs/2610.07760)

    本文将金融文档问答中的选择性升级形式化为令牌预算约束下的批次分配问题，发现额外检索调用的成本高度可预测，但准确识别哪些升级真正有益仍是主要瓶颈。

    

    检索增强生成系统可以将困难查询路由到更深层的上下文，但批量部署必须在各次调用之间分配共享的令牌预算，而这些调用的成本因查询而异。我们将选择性升级问题形式化为金融文档问答的有限批次分配问题。150个FinanceBench问题中的每一个首先获得top-1检索答案。预测器估计可选的top-5调用的裁定质量增益和令牌成本，分配器则根据预测的每令牌增益对调用进行优先级排序。在名义10%预算下，每令牌增益分配比仅增益排序将裁定质量提高了0.034（95%文档自助法置信区间 [0.001, 0.072]），同时比单次top-5检索节省46.6%的总令牌用量。额外调用的成本可以被准确预测（R方为0.93），而有益的升级仍然难以排序识别（AUROC为0.60）。这些结果表明，异构成本在紧（原文摘要在此处截断）

    arXiv:2610.07760v1 Announce Type: new  Abstract: Retrieval-augmented generation systems can route difficult queries to deeper context, but batch deployments must allocate a shared token budget across calls whose costs vary by query. We formulate selective escalation as finite-batch allocation for financial document question answering. Each of 150 FinanceBench questions first receives a top-1 retrieval answer. Predictors estimate the adjudication-quality gain and token cost of an optional top-5 call, and the allocator prioritizes calls by predicted gain per token. At the nominal 10% budget, gain-per-token allocation improves adjudication quality over gain-only ranking by 0.034 (95% document-bootstrap CI [0.001, 0.072]) while using 46.6% fewer total tokens than one-pass top-5 retrieval. Additional-call cost is accurately predictable (R-squared 0.93), whereas beneficial escalation remains difficult to rank (AUROC 0.60). These results show that heterogeneous cost is actionable under tight 
    
[^29]: 分层指数-高斯混合模型用于观看时间分布预测

    Hierarchical Exponential-Gaussian Mixtures for Watch-Time Distribution Prediction

    [https://arxiv.org/abs/2608.23356](https://arxiv.org/abs/2608.23356)

    本文提出了一种分层指数-高斯混合模型，通过分层分解和正则化技术解决了现有EGMN模型的方差崩溃和组件冗余问题，从而提升了短视频推荐中观看时间预测的准确性。

    

    准确的观看时间（WT）预测是短视频推荐的重要需求。然而，WT分布具有接近零膨胀、长尾和多模态的特点。最近的指数-高斯混合网络（EGMN）对完整的条件WT分布进行建模，而非单一的点估计，并达到了最先进的性能。我们的大规模复现研究表明，EGMN容易遭受方差崩溃、组件冗余和无效组件的问题。我们提出了一种分层指数-高斯混合（HEGM）模型，通过分层跳过观看分解、基于KL的方差正则化、结构化初始化、去除强制高斯偏移和熵正则化来解决这些失败模式。在公开和大规模工业数据集上，HEGM提高了排序准确性和阈值事件预测，同时保持了竞争力的点估计准确性，并显著提升了整体性能。

    arXiv:2608.23356v1 Announce Type: new  Abstract: Accurate watch-time (WT) prediction is an important requirement for short-video recommendations. Yet WT distributions are near-zero-inflated, long-tailed and multimodal. The recent Exponential-Gaussian Mixture Network (EGMN) models the full conditional WT distribution rather than a single point estimate and achieves state-of-the-art performance. Our large-scale reproduction study reveals that EGMN is vulnerable to variance collapse, component redundancy, and inactive components. We propose a Hierarchical Exponential-Gaussian Mixture (HEGM) model that addresses these failure modes through a hierarchical skip-watch decomposition, KL-based variance regularization, structured initialization, removing the forced Gaussian shift and the entropy regularizer. Across public and large-scale industrial datasets, HEGM improves ranking accuracy and threshold-event prediction, while maintaining competitive point-estimation accuracy and substantially im
    
[^30]: SIREN（将LLM引向险境）：Web-RAG推荐系统中的PAIR驱动偏好操纵

    SIREN (Luring LLMs onto the Rocks): PAIR-Driven Preference Manipulation in Web-RAG Recommenders

    [https://arxiv.org/abs/2607.21951](https://arxiv.org/abs/2607.21951)

    本文提出SIREN方法，利用PAIR越狱循环和23种可解释编辑策略，在Web-RAG推荐系统中操纵LLM的排名输出，使目标实体升至首位。

    

    本文研究了针对网页增强型大型语言模型（LLM）生成的排名推荐结果进行对抗性操纵的问题。当LLM通过检索并阅读实时网页来回答推荐查询时，它扮演了推荐系统的角色，而每个检索到的页面都可能成为攻击面。先前的研究探讨了虚构产品、检索投毒和排名提升等问题，但这些研究并未比较在周围源集不变的情况下，对已检索页面的不同编辑如何改变模型的最终排名。为填补这一空白，我们提出了SIREN，一种自动化攻击者-评判者方法，它将PAIR越狱循环适应于竞争性排名操纵，目标是使选定实体在LLM生成的推荐中升至第1位。SIREN使用Anthropic的网页工具检索并捕获网页，然后通过23种可解释的编辑分类法迭代修改检索到的源。

    arXiv:2607.21951v2 Announce Type: replace  Abstract: This paper investigates the adversarial manipulation of the ranked recommendations produced by web-augmented large language models (LLMs). When an LLM answers a recommendation query by retrieving and reading live webpages, it acts as a recommender, and each retrieved page becomes a potential attack surface. Prior work has examined fabricated products, retrieval poisoning, and rank promotion. However, these studies do not compare how different edits to an already retrieved page change the model's final ranking while the surrounding source set remains unchanged. To address this gap, we propose SIREN, an automated attacker--judge method that adapts the PAIR jailbreaking loop to competitive rank manipulation, with the goal of moving a chosen entity to rank~1 in an LLM-generated recommendation. SIREN retrieves and captures webpages using Anthropic's web tools, then iteratively edits a retrieved source using an interpretable taxonomy of 23
    
[^31]: DualView：面向多跳文档重排序的自适应局部-全局融合

    DualView: Adaptive Local-Global Fusion for Multi-Hop Document Reranking

    [https://arxiv.org/abs/2605.18767](https://arxiv.org/abs/2605.18767)

    提出了双视图级联重排序框架DualView，通过局部评分器（堆叠交叉注意力）与全局评分器（Transformer文档间依赖建模）的自适应门控动态融合，实现高效精准的多跳文档重排序。

    

    多跳问答需要聚合来自多个文档的信息，这是知识密集型应用的一项关键能力。一个根本性的挑战在于如何从检索候选中高效地识别出最小相关文档集，同时保持高召回率。我们提出了一种高效的双视图级联重排序框架，用于多跳文档重排序。该框架作为E5-base-v2候选结果之上的轻量级检索后阶段运行，其架构包括：(1) 局部评分器，采用堆叠交叉注意力实现细粒度的查询-文档相关性计算；(2) 全局评分器，通过基于Transformer的上下文聚合建模文档间依赖关系。这两种视图通过以查询语义为条件的自适应门控机制进行动态融合。在候选集固定重排序设置（使用离线缓存嵌入）下，我们的模型取得了有竞争力的结果，尤其在MuSiQue上表现突出。

    arXiv:2605.18767v2 Announce Type: replace  Abstract: Multi-hop question answering requires aggregating information from multiple documents, a critical capability for knowledge-intensive applications. A fundamental challenge lies in efficiently identifying the minimal relevant document set from retrieved candidates while maintaining high recall.   We present an efficient dual-view cascaded reranking framework for multi-hop document reranking. Operating as a lightweight post-retrieval stage over E5-base-v2 candidates, our architecture comprises: (1) a Local Scorer employing stacked cross-attention for fine-grained query-document relevance; and (2) a Global Scorer modeling inter-document dependencies via Transformer-based context aggregation. These views are dynamically fused through an Adaptive Gate conditioned on query semantics.   Under the fixed candidate set reranking setting with offline cached embeddings, our model achieves competitive results, particularly outstanding on MuSiQue w
    
[^32]: 通过生成式引擎优化控制基于大语言模型搜索的输出排名

    Controlling Output Rankings in LLM-Based Search via Generative Engine Optimization

    [https://arxiv.org/abs/2602.03608](https://arxiv.org/abs/2602.03608)

    提出了CORE优化方法，通过向搜索引擎返回的检索内容附加策略性设计的优化内容，来控制基于大语言模型的生成式搜索中的输出排名，从而提升小企业和独立创作者产品的可见性。

    

    随着大语言模型（LLM）的兴起，顾客搜索和选择产品的方式正在发生变化。基于大语言模型的搜索（即生成式引擎）直接向用户提供产品推荐，而不像传统在线搜索那样需要用户自己去探索各种选项。然而，这些推荐深受大语言模型初始检索顺序的影响，这限制了小型企业和独立创作者的可见性，使其处于不利地位。在本工作中，我们提出了CORE，一种用于基于大语言模型搜索的生成式引擎输出排名控制优化方法。由于大语言模型与搜索引擎的交互是黑盒的，CORE以搜索引擎返回的内容作为影响输出排名的主要手段。具体而言，CORE通过在检索到的内容后面附加策略性设计的优化内容，来引导相关内容的排名。

    arXiv:2602.03608v2 Announce Type: replace  Abstract: The way customers search for and choose products is changing with the rise of large language models (LLMs). LLM-based search, or generative engines, provides direct product recommendations to users, rather than traditional online search results that require users to explore options themselves. However, these recommendations are strongly influenced by the initial retrieval order of LLMs, which disadvantages small businesses and independent creators by limiting their visibility.   In this work, we propose CORE, an optimization method that \textbf{C}ontrols \textbf{O}utput \textbf{R}ankings in g\textbf{E}nerative Engines for LLM-based search. Since the LLM's interactions with the search engine are black-box, CORE targets the content returned by search engines as the primary means of influencing output rankings. Specifically, CORE optimizes retrieved content by appending strategically designed optimization content to steer the ranking of
    
[^33]: PRISM：面向金融检索的提示精炼式上下文系统建模

    PRISM: Prompt-Refined In-Context System Modelling for Financial Retrieval

    [https://arxiv.org/abs/2511.14130](https://arxiv.org/abs/2511.14130)

    PRISM 提出一种无需训练的金融文档检索框架，通过组合提示工程、上下文学习与候选精炼三个可部署模块，将 FiQA-2018 的 Recall@100 从 0.7783 提升至 0.8326 并降低约 35% 的每查询成本，同时系统性地揭示了 k-倒数编码与多智能体协作等额外复杂性何时不再带来收益的边界条件。

    

    将大语言模型（LLM）部署于金融信息检索需要准确性、成本效率和可复现性，然而大多数方法都需要代价高昂的微调。PRISM 是一个无需训练的框架，它由三个可部署的模块（提示工程、上下文学习和候选精炼模块）组合而成，并针对金融文档检索中的多智能体协作进行了边界条件研究。在 FiQA-2018 数据集上，堆叠式重排序与 k-倒数编码将 Recall@100 从 0.7783 提升至 0.8326。在 FinAgentBench 上，经 Cohere 精炼的 top-50 候选池带来了更高的观测端到端 NDCG@5（低于该评估设置下的最小可检测效应），同时将每条查询的成本削减约 35%。额外增加的复杂性并不总是划算的：当上游阶段已达到高召回率时，k-倒数编码便不再带来增益；而对于细粒度的文本块排序，多智能体协作也并非默认的最优选择。在保持提示、模型和示例一致的配对对比实验中……（原文摘要在此处截断）

    arXiv:2511.14130v3 Announce Type: replace-cross  Abstract: Deploying LLMs for financial information retrieval demands accuracy, cost efficiency, and reproducibility, yet most approaches require expensive fine-tuning. PRISM is a training-free framework that composes three deployable modules (prompt engineering, in-context learning, and a candidate refinement module) with a boundary-condition study of multi-agent coordination for financial document retrieval. On FiQA-2018, stacked reranking and $k$-reciprocal encoding raise Recall@100 from 0.7783 to 0.8326. On FinAgentBench, a Cohere-refined top-50 pool yields a higher observed end-to-end NDCG@5, below this evaluation's minimum detectable effect, while cutting per-query cost by about 35%. Added complexity does not always pay off: $k$-reciprocal encoding stops helping once the upstream stage is already high-recall, and multi-agent coordination is not a default for fine-grained chunk ranking. Matched pairs that hold prompt, model, and exem
    
[^34]: 答案蕴于其中：自我衍生奖励实现可解释关系抽取

    The Answer Lies Within: Self-Derived Rewards Enable Explainable Relation Extraction

    [https://arxiv.org/abs/2510.06198](https://arxiv.org/abs/2510.06198)

    该论文提出COGRE认知启发推理框架与HIT@DICT强化学习中间奖励策略，利用从正确预测中自动提取的信用词典派生奖励，引导大模型推理聚焦关系相关语义并与人类标注的抽象层次对齐，从而同时提升单样本关系抽取的准确率和解释质量。

    

    arXiv:2510.06198v4 公告类型：替换 摘要：尽管大型语言模型具有卓越的推理能力，但在没有预定义关系标签的情况下进行单样本（one-shot）关系抽取仍然存在困难。我们识别出两个问题：模型常常被无关的词元（token）误导，而非关注传达关系的语义；并且模型往往无法与人类标注者所期望的抽象层次保持一致。我们提出了一个新颖的框架，通过两个组件来弥合这一差距：(1) COGRE，一个受认知启发的推理框架，将关系抽取结构化为一系列模仿人类文本处理的过程；(2) HIT@DICT，一种强化学习中间奖励策略，通过对推理中与关系相关的短语给予奖励，促使推理与关系标签对齐。该奖励源自从正确预测中自动提取的信用词典（credit dictionary）。我们的实验表明，该框架通过解决上述问题，同时提升了准确率和解释质量。

    arXiv:2510.06198v4 Announce Type: replace  Abstract: Despite the remarkable reasoning capabilities of large language models, they still struggle with one-shot relation extraction without predefined relation labels. We identify two pitfalls: models are often misled by irrelevant tokens instead of relation-conveying semantics, and they often fail to align with the abstraction level human annotators expect. We introduce a novel framework that closes this gap with two components: (1) COGRE, a cognitively-inspired reasoning framework that structures RE into a series of processes mimicking human text-processing; and (2) HIT@DICT, a reinforcement learning intermediate reward strategy that encourages reasoning to align with relational labels by rewarding relation-relevant phrases in reasoning. The reward is derived on a credit dictionary automatically extracted from correct predictions. Our experiments show that our framework improves both accuracy and explanation quality by addressing these t
    

