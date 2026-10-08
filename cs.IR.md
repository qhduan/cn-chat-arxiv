# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Two-Level Softmax Sampling Done Right: Correcting Bias from Size Imbalance and Dispersion](https://arxiv.org/abs/2610.10483) | 本文揭示了双层softmax采样因忽略簇规模不均衡和簇内相似度离散度而产生的系统性采样偏差，并提出了S-2LS和SD-2LS两种修正方法，以几乎零额外计算开销实现了更优的softmax近似采样。 |
| [^2] | [CrossWeave: Bridging Perspectives Across Online Communities with a Dual-Pane Design](https://arxiv.org/abs/2610.10441) | 提出了AI驱动的桥接系统CrossWeave，通过双栏设计在用户浏览帖子时展示来自其他社区的相关多元观点，并借助内容展示与反应模拟帮助用户走出回音室、实现建设性的跨社区交流。 |
| [^3] | [Does Document Structure Help Dense Retrieval? A Placebo-Controlled Ablation of Four Mechanisms Across Two Corpora](https://arxiv.org/abs/2610.10170) | 该研究通过安慰剂对照的统一消融实验检验四种文档结构处理机制，发现结构组织带来的检索收益来自内容本身，而非前置到分块中的标记。 |
| [^4] | [Training with Missed Targets in Generative Recommendation: Separating Supervision from Probability Competition](https://arxiv.org/abs/2610.10124) | 该论文通过构建三个控制变量的匹配损失函数，首次将“为错失目标添加监督”与“两组目标的概率竞争”这两个效应解耦，发现概率竞争会损害返回物品的排序质量，从而解释了附加错失目标这一训练策略效果难以归因的原因。 |
| [^5] | [ExperienceIndex: Artifact-Grounded Memory](https://arxiv.org/abs/2610.10091) | 提出了 ExperienceIndex，一种让 AI 智能体基于先前推理轨迹捕获并复用工件特定经验知识的新型记忆层，可提升知识密集型任务的答案质量并降低在线成本。 |
| [^6] | [Inverting Multi-Vector Visual Document Indices](https://arxiv.org/abs/2610.09920) | 该论文揭示了多向量视觉文档索引的严重隐私风险：攻击者无需接触原始文档，仅凭存储的patch向量即可重建出页面图像，恢复近半数的文字和敏感信息，并可通过反演页面以98.4%的准确率定位源页面，作者同时评估了token池化等低成本防护手段的有效性。 |
| [^7] | [The Impact of Backbone Evolution on LLM-Based Relevance Assessments](https://arxiv.org/abs/2610.09820) | 研究发现，LLM骨干模型的版本更新并不能稳定提升相关性判断的质量，总体性能相当或提升也不代表判断稳定——早期版本做出的正确判断在新版本中可能丢失。 |
| [^8] | [SoccerNet-FoulRet: Retrieving Semantically Similar Soccer Foul Videos](https://arxiv.org/abs/2610.09742) | 本文提出了首个语义足球犯规检索基准 SoccerNet-FoulRet，将判罚相关性定义为基于裁判解读的先例检索任务，并发现现有最强零样本模型在该任务上 HitRate@10 不足 5%，表明该问题仍极具挑战性。 |
| [^9] | [Towards Explaining Query Expansion Performance in Information Retrieval](https://arxiv.org/abs/2610.09724) | 本研究提出理想扩展查询（IEQ）概念和基于Cohen's d的可分离性度量这两个互补视角，用以解释查询扩展技术在不同查询上性能差异的原因。 |
| [^10] | [Finding the Right Balance: Relevance and Diversity in LLM Retrieval](https://arxiv.org/abs/2610.09412) | 该研究提出一种查询自适应的检索多样化规则，仅当最近邻检索到的有效独立文档数低于查询证据需求时才启用多样化，从而在冗余候选池上提升多证据任务的表现，同时避免对干净候选池造成损害。 |
| [^11] | [From Chunks to Functional Evidence: Function-Aware Retrieval for EDA Documentation QA](https://arxiv.org/abs/2610.09361) | 该论文重新设计了RAG的基本检索单元，将异构耦合的EDA文档工件组织为以超边形式记录的“功能单元”，并通过编码器对齐查询与功能单元、结合统一重排序器，解决了EDA文档问答中查询与知识组织方式不匹配的检索瓶颈。 |
| [^12] | [Reading Position Is the Baseline to Beat: A Time-Ordered Evaluation of Personalised Highlight Prediction](https://arxiv.org/abs/2610.09262) | 该论文通过时序评估证明，在个性化高亮预测中，仅利用读者当前阅读位置（将第一条高亮之后的句子排序）这一无需任何其他读者数据的基线即可显著超越流行度与相似度方法，而经距离折扣的流行度则在两个预测目标上均取得最佳表现。 |
| [^13] | [Quantize by Drift: Label-Free Mixed-Precision Post-Training Quantization for Text Embedders](https://arxiv.org/abs/2610.09227) | 该论文提出以量化引起的输出嵌入漂移作为无标签的模块敏感度信号，用于文本嵌入器的混合精度训练后量化，该信号与检索质量高度相关（宏观Spearman达0.911），无需部署中难以获取的相关性标注。 |
| [^14] | [What Transfers from a VLM Teacher? Comparing Supervision Signals for Visual Document Retrieval](https://arxiv.org/abs/2610.09177) | 该论文发现，在视觉文档检索中，将VLM教师模型用于判定难负样本和分数蒸馏比用教师丰富正样本更有效，可将ViDoRe v2的nDCG@5从55.2提升至最高63.0。 |
| [^15] | [Building Navigable Graphs Without Search in Three Composable Stages](https://arxiv.org/abs/2610.09041) | 提出一种由候选池、收尾、脊柱三个可组合阶段构成、无需邻居搜索即可构建可导航图的框架，证明收尾（边选择）阶段决定图质量，且该收尾可与任意划分器组合，在六个百万至亿级点数的语料库上比PiPNN提升3–14%。 |
| [^16] | [From High Recall to High Utility: Dataset-Adaptive Post-Processing of LLM-Generated Customer Intents](https://arxiv.org/abs/2610.09039) | 提出一种数据集自适应的后处理架构，通过标准化、去重、语义向量化和自适应聚类，将LLM高召回提取的客户意图精简为稳定、可追溯且高效用的意图单元。 |
| [^17] | [BEACON-SP: Ontology-Grounded GraphRAG Framework for Clinical Suicide Risk Assessment](https://arxiv.org/abs/2610.09026) | BEACON-SP通过构建整合多种自杀理论的本体，并将其与患者知识图谱结合实现本体引导的多跳推理，为临床医生提供了一种基于GraphRAG的自杀风险评估决策支持框架。 |
| [^18] | [Trustworthy Domain-Specific AI for Structured Knowledge Retrieval and Reasoning](https://arxiv.org/abs/2610.08894) | 本论文提出了一种可解释的可扩展架构，通过Binary Bleed和HNMFk等创新方法将非结构化领域文本转化为结构化知识图谱与向量存储，并利用T-SRAG实现跨检索路径的动态查询路由与推理。 |
| [^19] | [Madeleine: Learning Involuntary Recall for Conversational Memory from Simulated Lives](https://arxiv.org/abs/2610.01118) | Madeleine通过LLM人生模拟器离线学习记忆间的“非自主”关联（摊销化关联），在线阶段无需任何LLM调用、仅替换查询编码器即可接入任意向量记忆系统，以极低成本回忆起与当前话题不相似却至关重要的记忆，并在LoCoMo-Plus上取得最佳性能。 |
| [^20] | [Self-Indexing Attention for Compression-Compatible Sparse Long-Context LLM Inference](https://arxiv.org/abs/2609.13205) | 提出了一种免训练的自索引注意力框架，利用共享的1比特符号索引在预填充和解码阶段统一实现高效token检索，同时兼容外部KV缓存压缩，在5%注意力密度下达到接近密集注意力的准确率，并获得高达6.1倍预填充和10.3倍解码的算子加速。 |
| [^21] | [Progressive Disclosure for LLM-Maintained Wiki Knowledge Bases: a Preregistered Ablation](https://arxiv.org/abs/2607.04576) | 本文通过一项预注册消融实验，在四个页面内容完全相同、仅访问结构不同的LLM维护知识库版本上，检验渐进式披露（先读简洁目录和摘要、再按需打开页面）能否降低智能体问答的成本。 |
| [^22] | [Listwise Explanation of Embedding-Based Rankings via Semantic Chunk Grouping](https://arxiv.org/abs/2606.27980) | 提出ChunkGroupSHAP方法，通过将跨文档语义相关的文本分块聚类为共享特征来解释稠密嵌入排序结果，在保持上下文证据的同时控制计算维度，并证明解释单元的粒度应与排序模型的表示方式相匹配。 |
| [^23] | [LoopFM: Learning frOm HistOrical RePresentations of Foundation Model for Recommendation](https://arxiv.org/abs/2605.29280) | LoopFM通过将基础模型的中间嵌入结构化为下游垂直模型的输入特征（如用户历史序列），开辟了高带宽知识传递通道，在无需实时FM推理的情况下实现了显著AUC提升，并与知识蒸馏形成互补。 |
| [^24] | [Retrieval-Augmented Generation Must Move Beyond Factual Grounding to Represent Diverse Opinions](https://arxiv.org/abs/2604.12138) | 本论文指出RAG系统因过度追求事实准确性而忽视观点多样性，提出了观点感知检索框架O-RAG，通过不确定性量化和基于Wasserstein距离的统一目标，将语料级情感分布的距离降低18-48%，从而更好地表征多元观点。 |
| [^25] | [Document Optimization for Black-Box Retrieval via Reinforcement Learning](https://arxiv.org/abs/2604.05087) | 提出DocOpt方法，通过GRPO强化学习以检索排序提升为奖励，直接训练LLM/VLM离线重写文档以优化黑盒检索器的检索效果，从而将昂贵的计算从延迟敏感的检索路径转移到离线阶段。 |
| [^26] | [Vectorizing the Trie: Efficient Constrained Decoding for LLM-based Generative Retrieval on Accelerators](https://arxiv.org/abs/2602.22647) | 提出 STATIC 方法，通过将前缀树展平为 CSR 稀疏矩阵，把不规则的树遍历转化为完全向量化的稀疏矩阵运算，从而在 TPU/GPU 上实现高效、可扩展的基于大语言模型生成式检索的约束解码。 |
| [^27] | [Hypergraph-Enhanced Dual Convolutional Network for Bundle Recommendation](https://arxiv.org/abs/2312.11018) | HED 通过构建包含用户-捆绑包、用户-物品、捆绑包-物品交互及用户内部、捆绑包内部关系的完整超图，并将全超图传播与用户-捆绑包双卷积分支耦合，在保留推荐特有信号的同时引入物品感知的高阶上下文，在 NetEase 和 Youshu 数据集上较最强基线显著提升了捆绑包推荐性能。 |

# 详细

[^1]: 正确的双层Softmax采样：修正来自簇规模不均衡与离散度的偏差

    Two-Level Softmax Sampling Done Right: Correcting Bias from Size Imbalance and Dispersion

    [https://arxiv.org/abs/2610.10483](https://arxiv.org/abs/2610.10483)

    本文揭示了双层softmax采样因忽略簇规模不均衡和簇内相似度离散度而产生的系统性采样偏差，并提出了S-2LS和SD-2LS两种修正方法，以几乎零额外计算开销实现了更优的softmax近似采样。

    

    从softmax分布中采样是机器学习中的一项基础操作，但其相对于项目数量的线性复杂度使得精确采样在大规模场景下难以实际应用。双层softmax（2LS）采样是一种流行的替代方案，可实现亚线性时间采样。该方法假设项目被划分为若干簇，2LS首先采样一个簇，然后在该簇内采样一个项目。在本文中，我们表明，尽管具有优势，2LS会引入系统性的不良采样偏差，这些偏差源于对簇的错误加权，即同时忽略了簇规模的不均衡性和簇内相似度的离散度。我们提出了两种采样方法：规模修正的2LS（S-2LS）以及规模与离散度修正的2LS（SD-2LS），它们修正了这些偏差，并以微乎其微甚至为零的额外计算开销提供了可证明更优的softmax近似。在五个大规模数据集上的深入实验验证了我们方法改进后的采样特性。

    arXiv:2610.10483v1 Announce Type: new  Abstract: Sampling from a softmax distribution is a fundamental operation in machine learning, but its linear complexity in the number of items makes exact sampling impractical at scale. Two-level softmax (2LS) sampling is a popular alternative enabling sublinear-time sampling. Assuming items are partitioned into clusters, 2LS first samples a cluster and then an item within it. In this paper, we show that, despite its advantages, 2LS introduces systematic and undesirable sampling biases, which arise from misweighting clusters by ignoring both cluster size imbalance and intra-cluster similarity dispersion. We propose two sampling methods, Size-Corrected 2LS (S-2LS) and Size- and Dispersion-Corrected 2LS (SD-2LS), which correct these biases and provide provably better softmax approximations with negligible to non-existent computational overhead. In-depth experiments on five large-scale datasets validate the improved sampling properties of our method
    
[^2]: CrossWeave：通过双栏设计连接在线社区间的多元视角

    CrossWeave: Bridging Perspectives Across Online Communities with a Dual-Pane Design

    [https://arxiv.org/abs/2610.10441](https://arxiv.org/abs/2610.10441)

    提出了AI驱动的桥接系统CrossWeave，通过双栏设计在用户浏览帖子时展示来自其他社区的相关多元观点，并借助内容展示与反应模拟帮助用户走出回音室、实现建设性的跨社区交流。

    

    社交媒体系统通常展示已经彼此熟悉的人之间的对话，这种设计可能显得可预测且片面。在公共讨论中，这种设计会缩小讨论范围、加剧群体分歧，并扭曲人们对舆论的认知。为了鼓励跨社区参与，我们提出了CrossWeave，这是一个AI驱动的桥接系统，用于增强标准的社交媒体信息流。当用户阅读帖子时，CrossWeave会在侧边栏中展示来自其他话题讨论的相关多元帖子，并突出它们之间的联系。该系统邀请用户走出回音室，探索更广泛的观点和论点，并“点击跨越”与帖子的作者展开互动。当用户这样做时，CrossWeave不仅会展示相关的历史内容，还会在用户撰写帖子时模拟可能的反应，从而促进建设性的发帖。

    arXiv:2610.10441v1 Announce Type: cross  Abstract: Social media systems typically display conversations among already familiar contributors, which can be predictable and one-sided. In civic discourse, this design narrows discussion, reinforces divides, and distorts the perception of public opinion. To encourage cross-community engagement, we present CrossWeave, an AI-powered bridging system that augments the standard social media feed. As the user reads a post, CrossWeave surfaces diverse relevant posts from other threads in a side pane and highlights the connections. Users are invited to venture out of their echo chamber, explore a broader range of views and arguments, and ``click across'' to engage with their authors. When they do, CrossWeave facilitates constructive posting, not only by showcasing relevant past content but also by simulating possible reactions as the user drafts a post.
    
[^3]: 文档结构有助于稠密检索吗？跨两个语料库对四种机制的安慰剂对照消融研究

    Does Document Structure Help Dense Retrieval? A Placebo-Controlled Ablation of Four Mechanisms Across Two Corpora

    [https://arxiv.org/abs/2610.10170](https://arxiv.org/abs/2610.10170)

    该研究通过安慰剂对照的统一消融实验检验四种文档结构处理机制，发现结构组织带来的检索收益来自内容本身，而非前置到分块中的标记。

    

    检索增强生成（RAG）系统日益依赖文档结构处理方法：结构对齐分块、LLM生成的分块上下文、标题路径元数据以及分层两阶段检索。已有研究分别在不同的语料库、嵌入模型和指标上支持这些方法，但都没有控制一个共同的混杂因素：任何前置到分块上的文本都会扰动其嵌入向量。我们提出了一种机制隔离的消融实验，在统一协议下测试所有四种处理方法，在各实验条件间匹配分块大小，并引入语义上无效的安慰剂——结构上有效但在文档间打乱顺序的标题路径。我们使用覆盖率感知的nDCG对检索效果进行评分，并通过基于文档聚类的自助法与Holm校正检验四个预注册的对比实验，实验在两个差异较大的语料库上进行：200篇维基百科特色文章（951个查询）和1,585篇QASPER论文（4,303个问题）。结构组织确实有帮助，且其原因在于内容本身而非标记（tokens）：结构对齐……

    arXiv:2610.10170v1 Announce Type: new  Abstract: Retrieval-augmented generation systems increasingly rely on document-structure treatments: structure-aligned chunking, LLM-generated chunk contexts, heading-path metadata, and hierarchical two-stage retrieval. Separate studies support each on different corpora, embedders, and metrics, and none control for a shared confound: any text prepended to a chunk perturbs its embedding. We present a mechanism-isolating ablation testing all four treatments under one protocol, matching chunk sizes across conditions and adding a semantically null placebo---heading paths that are structurally valid but shuffled across documents. We score retrieval with a coverage-aware nDCG and test four pre-registered contrasts via document-clustered bootstrap with Holm correction, on two distant corpora: 200 Wikipedia Featured Articles (951 queries) and 1,585 QASPER papers (4,303 questions). Organization helps, and the cause is content, not tokens: structure-aligned
    
[^4]: 生成式推荐中错失目标的训练：将监督与概率竞争相分离

    Training with Missed Targets in Generative Recommendation: Separating Supervision from Probability Competition

    [https://arxiv.org/abs/2610.10124](https://arxiv.org/abs/2610.10124)

    该论文通过构建三个控制变量的匹配损失函数，首次将“为错失目标添加监督”与“两组目标的概率竞争”这两个效应解耦，发现概率竞争会损害返回物品的排序质量，从而解释了附加错失目标这一训练策略效果难以归因的原因。

    

    生成式推荐器返回一个有限的候选集，并且在重排序之前可能会遗漏已观测到的目标。一种训练策略将这些被遗漏的目标附加到重排序器的训练列表中，尽管在推理时仍然只对原始候选进行排序。这一操作同时改变了已检索目标的权重、为附加目标增加了监督，并使两组目标争夺概率。因此，简单的附加/不附加对比实验无法解释返回物品排序的变化。我们构建了三个匹配的损失函数，在保持已检索目标权重不变的前提下，分别单独引入附加目标监督和组间竞争。其中中间的损失函数在两组内部进行训练但分别对它们归一化，从而防止仅用于训练的目标与推理候选产生竞争。在已发布的OneRec模型和本地训练的Amazon生成器上的实验表明，这种概率竞争可能会损害返回物品的排序。在四个预先设定的（实验中……摘要在此处截断）

    arXiv:2610.10124v1 Announce Type: cross  Abstract: Generative recommenders return a limited candidate set and may omit observed targets before reranking. A training strategy appends these missed targets to reranker training lists, although inference still ranks only original candidates. This operation simultaneously changes retrieved-target weight, adds supervision over appended targets, and makes the two groups compete for probability. An append/no-append comparison therefore cannot explain changes in returned-item rankings. We construct three matched losses that hold retrieved-target weight fixed while introducing appended-target supervision and group competition separately. The intermediate loss trains within both groups but normalizes them separately, preventing training-only targets from competing with inference candidates. Experiments with a released OneRec model and locally trained Amazon generators show that this competition can harm returned-item ranking. In four prespecified 
    
[^5]: ExperienceIndex：基于工件（Artifact）的记忆

    ExperienceIndex: Artifact-Grounded Memory

    [https://arxiv.org/abs/2610.10091](https://arxiv.org/abs/2610.10091)

    提出了 ExperienceIndex，一种让 AI 智能体基于先前推理轨迹捕获并复用工件特定经验知识的新型记忆层，可提升知识密集型任务的答案质量并降低在线成本。

    

    知识密集型任务需要通过推理共享的工件语料库（例如法院判例或科学文献）来回答大量问题。当人类与这些语料库交互时，会自然地积累关于工件的经验知识，从而能够快速识别出每个新任务的完整相关工件集合。然而，现有的人工智能智能体缺乏合适的记忆解决方案来构建或复用这种以工件为基础的经验，导致答案质量较低且在线成本较高。现有的记忆解决方案虽然能从先前的任务求解轨迹中提取和复用信息，但它们主要关注用户偏好、事实属性或抽象推理模式，而非持久的、特定于工件的知识。我们提出了 ExperienceIndex，这是一种面向 AI 智能体的新型经验层，它基于先前的推理轨迹来捕获和复用关于工件的知识。ExperienceIndex 存储两种互补的……

    arXiv:2610.10091v1 Announce Type: new  Abstract: Knowledge-intensive tasks require answering many questions by reasoning about a shared corpus of artifacts (e.g., court cases, or scientific literature). As humans interact with these corpora, they naturally accumulate experiential knowledge about artifacts, enabling them to quickly identify the complete set of relevant artifacts for each new task. However, existing AI agents lack appropriate memory solutions to build or reuse such artifact-grounded experience, leading to lower answer quality and higher online cost. Existing memory solutions extract and reuse information from prior task-solving traces, but they primarily focus on user preferences, factual attributes, or abstract reasoning patterns rather than persistent artifact-specific knowledge. We introduce ExperienceIndex, a novel experience layer for AI agents that captures and reuses knowledge about artifacts based on prior reasoning traces. ExperienceIndex stores two complementar
    
[^6]: 反演多向量视觉文档索引

    Inverting Multi-Vector Visual Document Indices

    [https://arxiv.org/abs/2610.09920](https://arxiv.org/abs/2610.09920)

    该论文揭示了多向量视觉文档索引的严重隐私风险：攻击者无需接触原始文档，仅凭存储的patch向量即可重建出页面图像，恢复近半数的文字和敏感信息，并可通过反演页面以98.4%的准确率定位源页面，作者同时评估了token池化等低成本防护手段的有效性。

    

    主流的多向量视觉文档检索器将每一页文档存储为大约一千个patch向量，这些向量通常保存在由第三方运营的向量数据库中。由于人们通常认为无法从向量中读出页面内容，这种索引往往被视为比原始页面本身敏感度更低。然而，由于索引按照光栅顺序为每个patch保留一个向量，且每个向量由预先训练用于阅读文档的视觉-语言模型计算得出，我们假设无论是谁运行或入侵了该存储库，都可以仅凭索引重建出整个页面。我们将这种反演问题构建为条件文档图像生成任务，并从向量中推断攻击所需的信息：编码器类型、页面形状，以及在向量顺序被打乱时的原始排列顺序。在ViDoRe v3基准测试中，从原始索引反演出的页面恢复了47%的单词和45%的敏感标记。当将这些反演出的页面作为查询对存储的索引进行检索时，它们在98.4%的情况下将源页面排在第一位。我们测试了两种低成本的防护措施，包括token池化……

    arXiv:2610.09920v1 Announce Type: cross  Abstract: Prevailing multi-vector visual document retrievers store each page as about a thousand patch vectors, often in vector databases run by a third party. Since no one can read a page from its vectors, this index is easily treated as less sensitive than the page. However, because the index keeps one vector per patch in raster order, and each vector is computed by a vision-language model pre-trained to read documents, we hypothesize that whoever runs or breaches the store can reproduce a page from its index alone. We frame inversion as conditional document image generation and infer from the vectors what the attack needs: the encoder, the page shape and, for shuffled vectors, their order. On the ViDoRe v3 benchmark, pages inverted from raw indices recover 47% of the words and 45% of the sensitive tokens. Used as queries against the stored indices, they rank their source page first 98.4% of the time. We test two cheap protections, token pooli
    
[^7]: 骨干模型演进对基于大语言模型的相关性评估的影响

    The Impact of Backbone Evolution on LLM-Based Relevance Assessments

    [https://arxiv.org/abs/2610.09820](https://arxiv.org/abs/2610.09820)

    研究发现，LLM骨干模型的版本更新并不能稳定提升相关性判断的质量，总体性能相当或提升也不代表判断稳定——早期版本做出的正确判断在新版本中可能丢失。

    

    大语言模型（LLM）正在快速演进，新模型往往具备更强的能力。这表明在基于LLM的相关性判断任务中，能力更强的模型在相同提示下应能与人类判断达成更高的一致性。我们通过研究骨干模型演进过程中基于LLM的相关性判断器的行为，对这一认知提出了挑战。在保持提示词固定不变的情况下，我们在商业模型（Gemini、GPT）和开放权重模型（Qwen、Llama）的连续版本上，评估了具有代表性的单提示方法（UMBRELA）和基于评分规则的提示方法（EXAM）。总体而言，我们没有发现一致的证据表明更新的模型版本能带来更好的相关性判断。至关重要的是，总体性能相似或有所提升并不意味着判断的稳定性：LLM骨干早期版本做出的正确判断，不一定会在后续版本中得到保留。我们进一步研究了这些判断退化的潜在驱动因素。我们的研究结果提醒人们不要盲目假设……

    arXiv:2610.09820v1 Announce Type: new  Abstract: LLMs are evolving rapidly, with newer models offering stronger capabilities. This suggests that in LLM-based relevance judging, more capable models will achieve higher agreement with human judgements under the same prompt. We challenge this understanding by investigating the behavior of LLM-based relevance judges under backbone evolution. Keeping the prompts fixed, we evaluate a representative single-prompt (UMBRELA) and a rubric-based prompt (EXAM) across sequential model versions of commercial (Gemini, GPT) and open-weight (Qwen, Llama) models. Overall, we find no consistent evidence that newer versions lead to better relevance judges. Crucially, similar or improved aggregate performance does not imply judgment stability: correct judgements made by an earlier version of an LLM backbone are not necessarily preserved by later versions. We investigate the potential drivers of these regressions. Our findings caution against the assumption 
    
[^8]: SoccerNet-FoulRet：检索语义相似的足球犯规视频

    SoccerNet-FoulRet: Retrieving Semantically Similar Soccer Foul Videos

    [https://arxiv.org/abs/2610.09742](https://arxiv.org/abs/2610.09742)

    本文提出了首个语义足球犯规检索基准 SoccerNet-FoulRet，将判罚相关性定义为基于裁判解读的先例检索任务，并发现现有最强零样本模型在该任务上 HitRate@10 不足 5%，表明该问题仍极具挑战性。

    

    职业足球比赛中的裁判判罚一直缺乏一致性，原因在于裁判难以将争议性犯规与类似的过往案例进行对比。我们将这一问题建模为检索任务，并提出了 SoccerNet-FoulRet——首个语义犯规检索基准。给定一个查询犯规视频，任务目标是检索出被判定为相关先例的过往犯规案例，且不受摄像机角度、球队或外观差异的影响。这与以往通过视觉相似性或共同事件来匹配片段的视频对视频检索不同，在本文中，相关性由裁判的判罚解读来定义。我们基于 SoccerNet-MVFoul 数据集构建了该基准，并在 693 个经人工验证的查询及类别相关性标签上，评估了零样本视频嵌入模型、视觉-语言嵌入模型以及一个针对任务微调的基线模型的检索能力。实验表明语义犯规检索仍然极具挑战性，最强的零样本模型在人工验证的先例上取得的 HitRate@10 不足 5%。

    arXiv:2610.09742v1 Announce Type: cross  Abstract: Refereeing decisions in professional soccer remain inconsistent because referees cannot easily compare a contentious foul against similar past cases. We cast this as a retrieval problem and introduce SoccerNet-FoulRet, the first benchmark for semantic foul retrieval. Given a query foul, the task is to retrieve past fouls judged to be relevant precedents, regardless of camera angle, teams, or appearance. This differs from prior video-to-video retrieval, which matches clips by visual similarity or a shared event. Here, relevance is defined by refereeing interpretation. We build the benchmark from the SoccerNet-MVFoul dataset and evaluate retrieval ability of zero-shot video and vision-language embedders together with a task-specific fine-tuned baseline on 693 human-verified queries and category-relevance labels. Semantic foul retrieval remains challenging. The strongest zero-shot model achieves under 5% HitRate@10 on human-verified prece
    
[^9]: 迈向解释信息检索中查询扩展性能

    Towards Explaining Query Expansion Performance in Information Retrieval

    [https://arxiv.org/abs/2610.09724](https://arxiv.org/abs/2610.09724)

    本研究提出理想扩展查询（IEQ）概念和基于Cohen's d的可分离性度量这两个互补视角，用以解释查询扩展技术在不同查询上性能差异的原因。

    

    查询扩展（QE）技术长期以来被广泛应用于信息检索（IR）中，以解决词汇不匹配问题。它们在现代检索系统中仍然具有重要价值，包括基于大语言模型（LLM）的检索系统。然而，没有任何单一的QE方法能够在所有查询上一致地优于其他方法。本工作试图通过两个互补的视角来解释QE性能的差异。第一个是理想扩展查询（IEQ）的概念——即一种假设性的查询，它能够在下游BM25检索模型中最大化检索效果。第二个是可分离性视角，它使用Cohen's d来量化在给定扩展查询下，相关文档与非相关文档被评分的区分程度。我们开发了一种可分离性度量以及逼近IEQ的实用公式，并研究这些因素与检索效果之间的关系。我们在TREC Robust数据集上进行了大量实验。

    arXiv:2610.09724v1 Announce Type: cross  Abstract: Query Expansion (QE) techniques have long been widely used in Information Retrieval (IR) to address the vocabulary mismatch problem. They remain relevant in modern retrieval systems, including those based on large language models (LLMs). However, no single QE method consistently outperforms others across all queries. This work seeks to explain the variation in QE performance through two complementary perspectives. The first is the concept of an Ideal Expanded Query (IEQ)--a hypothetical query that maximizes retrieval effectiveness with a downstream BM25 retrieval model. The second is a separability perspective, which quantifies how distinctly relevant and non-relevant documents are scored for a given expanded query using Cohen's (d). We develop a separability measure and practical formulations to approximate the IEQ and investigate how these factors relate to retrieval effectiveness. Extensive experiments on the TREC Robust collection,
    
[^10]: 找到恰当的平衡：LLM检索中的相关性与多样性

    Finding the Right Balance: Relevance and Diversity in LLM Retrieval

    [https://arxiv.org/abs/2610.09412](https://arxiv.org/abs/2610.09412)

    该研究提出一种查询自适应的检索多样化规则，仅当最近邻检索到的有效独立文档数低于查询证据需求时才启用多样化，从而在冗余候选池上提升多证据任务的表现，同时避免对干净候选池造成损害。

    

    检索多样化在检索增强生成（RAG）框架中被广泛应用，然而以往的研究对于它是否能改善检索效果和答案质量存在分歧。我们证明其有效性主要取决于候选池的冗余程度，其变化规律与查询所需的独立证据片段数量相一致。通过受控的近重复文本注入和生产环境风格的滚动重叠分块实验，我们发现多样化在干净的候选池上会损害相关性、证据覆盖率和答案质量，但当冗余导致最近邻检索反复选中重复段落时，多样化在多证据任务上会变得有益。因此，我们提出一种查询自适应规则：仅当最近邻top-k选择中有效独立文档的数量低于该查询的证据需求时才启用多样化。该规则可直接基于现有嵌入计算，能够获得大部分可实现的收益，并且具有良好的可迁移性。

    arXiv:2610.09412v1 Announce Type: cross  Abstract: Retrieval diversification is widely available in retrieval-augmented generation (RAG) frameworks, yet prior studies disagree on whether it improves retrieval and answer quality. We show that its effectiveness varies primarily with candidate-pool redundancy, in a pattern consistent with the number of distinct evidence pieces a query requires. Using controlled near-duplicate injection and production-style overlapping chunking, we find that diversification harms relevance, evidence coverage and answer quality on clean pools, but becomes beneficial on multi-evidence tasks when redundancy causes nearest-neighbor retrieval to select repeated passages. We therefore introduce a query-adaptive rule that diversifies only when the effective number of distinct documents in the nearest-neighbor top-$k$ selection falls below the query's evidence requirement. Computed from existing embeddings, the rule captures most of the achievable gain, transfers 
    
[^11]: 从文本块到功能性证据：面向EDA文档问答的功能感知检索

    From Chunks to Functional Evidence: Function-Aware Retrieval for EDA Documentation QA

    [https://arxiv.org/abs/2610.09361](https://arxiv.org/abs/2610.09361)

    该论文重新设计了RAG的基本检索单元，将异构耦合的EDA文档工件组织为以超边形式记录的“功能单元”，并通过编码器对齐查询与功能单元、结合统一重排序器，解决了EDA文档问答中查询与知识组织方式不匹配的检索瓶颈。

    

    检索增强生成（RAG）被广泛用于将回答锚定于文档之中。然而，对于复杂的技术文档而言，主要瓶颈往往不在于模型推理能力，而在于查询与知识为检索而组织的方式之间的不匹配。这种不匹配在电子设计自动化（EDA）文档中尤为突出，因为回答所需的信息分散在异构但紧密耦合的各类工件之中。因此，我们重新设计了RAG的基本检索单元。不同于在孤立文本块或二元关系上进行操作，我们将类型化的工件收集为EDA功能单元，每个单元被记录为一条超边，并带有指向其源文本块的链接。随后，我们训练一个编码器将查询与功能单元对齐，并将单元检索与直接的文本块检索相结合。在将选定的单元映射回其源文本块之后，一个统一的重排序器从中挑选出提供给生成器的证据。在新构建的（基准测试上……）

    arXiv:2610.09361v1 Announce Type: new  Abstract: Retrieval-Augmented Generation (RAG) is widely used to ground answers in documents. For complex technical documentation, however, the primary bottleneck is often not model reasoning but a mismatch between a query and the way knowledge is organized for retrieval. This mismatch is pronounced in Electronic Design Automation (EDA) documentation, where the information needed for an answer is scattered across heterogeneous yet tightly coupled artifacts. We therefore redesign the basic retrieval unit of RAG. Instead of operating on isolated chunks or binary relations, we collect typed artifacts into EDA functional units. Each unit is recorded as a hyperedge with links to its source chunks. We then train an encoder to align queries with functional units and combine unit retrieval with direct chunk retrieval. After mapping the selected units back to their sources, a unified reranker chooses the evidence given to the generator. On the newly constr
    
[^12]: 阅读位置才是需要击败的基线：个性化高亮预测的时序评估

    Reading Position Is the Baseline to Beat: A Time-Ordered Evaluation of Personalised Highlight Prediction

    [https://arxiv.org/abs/2610.09262](https://arxiv.org/abs/2610.09262)

    该论文通过时序评估证明，在个性化高亮预测中，仅利用读者当前阅读位置（将第一条高亮之后的句子排序）这一无需任何其他读者数据的基线即可显著超越流行度与相似度方法，而经距离折扣的流行度则在两个预测目标上均取得最佳表现。

    

    读者在页面上做出的第一条高亮，是阅读产品所拥有的成本最低的个性化信号。自然而然的方案是推荐相似的早期读者标注过的内容，并以流行度作为评判标准。我们认为，真正需要击败的基线是阅读位置。在一个社交高亮平台上的时序评估中（首次高亮后共7,343个读者-页面对，涉及1,511个页面），仅依据读者第一条高亮对其后的句子进行排序——完全不使用任何其他读者的数据——在47%的情况下即可将下一条高亮排进前五，而流行度方法为26%，两种相似度方法中较好的一个为29%。该基线的有效性取决于预测目标：在预测所有后续高亮时，该排序方法输给流行度；而按距最新高亮的距离进行折扣的流行度（在尝试的三种尺度中取平均精度最高者），在两个预测目标上都胜过流行度和两种相似度方法。在一项预先设定的比较中，两种相似度方法均……（原文在此处截断）

    arXiv:2610.09262v1 Announce Type: new  Abstract: A reader's first highlights on a page are the cheapest personal signal a reading product has. The natural plan is to suggest what similar earlier readers marked, and to judge the result against popularity. We argue that the baseline to beat is reading position. In a time-ordered evaluation on one social highlighting platform (7,343 reader-page pairs on 1,511 pages after one highlight), ranking the sentences just below a reader's first highlight, with no other reader's data, puts the next highlight in the top five 47% of the time, against 26% for popularity and 29% for the better of two similarity methods. The baseline depends on the target: over all later highlights that ranking loses to popularity, while popularity discounted by distance from the latest highlight, at the scale with the best average precision of three tried, beats popularity and both similarity methods on both targets. In a comparison specified in advance, neither simila
    
[^13]: 基于漂移的量化：面向文本嵌入器的无标签混合精度训练后量化

    Quantize by Drift: Label-Free Mixed-Precision Post-Training Quantization for Text Embedders

    [https://arxiv.org/abs/2610.09227](https://arxiv.org/abs/2610.09227)

    该论文提出以量化引起的输出嵌入漂移作为无标签的模块敏感度信号，用于文本嵌入器的混合精度训练后量化，该信号与检索质量高度相关（宏观Spearman达0.911），无需部署中难以获取的相关性标注。

    

    混合精度训练后量化需要一个逐模块的敏感度信号；对于文本嵌入器而言，最直观的信号——某个模块被量化时所损失的检索质量——需要部署场景中很少具备的相关性标注。我们测量了一种无标签的替代信号：量化引起的表示漂移，其获取方式是量化单个模块、重新编码语料库，并记录输出嵌入相对于其全精度位置的偏移距离。其独特之处在于所使用的观测量：即稠密检索器用于排序的部署态输出表示。在五个开发用嵌入器上，配置级漂移对采样得到的混合精度方案相对于留出集检索质量的排序达到了0.911的宏观Spearman相关系数；在可用范围内，该敏感度可跨校准语料库和检索域迁移；各模块的漂移在排序上可一致地组合，但在数值上则不然；而基于相关性导出的敏感度并未带来一致的价值提升。

    arXiv:2610.09227v1 Announce Type: cross  Abstract: Mixed-precision post-training quantization needs a per-module sensitivity signal; for a text embedder the obvious one -- the retrieval quality a module costs when quantized -- needs relevance labels that deployments rarely have. We measure a label-free substitute: quantization-induced representation drift, obtained by quantizing one module, re-encoding the corpus, and recording how far the output embeddings moved from their full-precision positions. What is specific is the observable: the deployed output representation a dense retriever ranks with. Across five development embedders, configuration-level drift orders sampled mixed-precision plans against held-out retrieval quality at a macro Spearman of 0.911, the sensitivity transports across calibration corpora and retrieval domains in the usable regime, module drifts compose rank-consistently but not numerically, and relevance-derived sensitivity adds no consistent value. The method i
    
[^14]: 从VLM教师模型中迁移什么？视觉文档检索中监督信号的对比研究

    What Transfers from a VLM Teacher? Comparing Supervision Signals for Visual Document Retrieval

    [https://arxiv.org/abs/2610.09177](https://arxiv.org/abs/2610.09177)

    该论文发现，在视觉文档检索中，将VLM教师模型用于判定难负样本和分数蒸馏比用教师丰富正样本更有效，可将ViDoRe v2的nDCG@5从55.2提升至最高63.0。

    

    视觉文档检索器通过对比学习进行训练：每个查询与一个标注为相关的页面（正样本）匹配，并与推定为不相关的页面（负样本）拉开距离。最近的方法将视觉语言模型（VLM）教师蒸馏到检索器中，通过丰富正样本来实现，例如迁移教师对正样本的注意力或对其的描述。我们探究教师模型是否更适合用在另一端，即对检索器挖掘出的候选样本进行负样本判定——而标注标签对此并未提供任何信息。在固定学生模型、数据、优化器和评估设置的条件下，教师判定的难负样本和分数蒸馏方法将ViDoRe v2的nDCG@5从55.2分别提升至62.6和63.0；本文改编的描述对齐方法提升了2.6个百分点，而注意力定位方法则没有可测量的提升。与在相同训练计算量下、从同一挖掘池中选择四个候选样本的无教师规则相比——其中最好的是当前系统所使用的正样本感知阈值——

    arXiv:2610.09177v1 Announce Type: new  Abstract: Visual document retrievers are trained contrastively: each query is matched to one page labelled relevant - the positive - and pushed away from negatives, pages presumed irrelevant. Recent methods distil a vision-language model (VLM) teacher into the retriever by enriching that positive, transferring the teacher's attention over it or a description of it. We ask whether the teacher is better spent on the other side, judging the candidates the retriever mines as negatives, which the label says nothing about. With student, data, optimizer and evaluation fixed, teacher-judged hard negatives and score distillation raise ViDoRe v2 nDCG@5 from 55.2 to 62.6 and 63.0; description alignment, as adapted here, gains 2.6 points and attention grounding nothing measurable. Against teacher-free rules that select four candidates from the same mined pool at identical training compute, the best of which is the positive-aware threshold current systems use,
    
[^15]: 无需搜索即可构建可导航图的三阶段可组合方法

    Building Navigable Graphs Without Search in Three Composable Stages

    [https://arxiv.org/abs/2610.09041](https://arxiv.org/abs/2610.09041)

    提出一种由候选池、收尾、脊柱三个可组合阶段构成、无需邻居搜索即可构建可导航图的框架，证明收尾（边选择）阶段决定图质量，且该收尾可与任意划分器组合，在六个百万至亿级点数的语料库上比PiPNN提升3–14%。

    

    可导航图可以在不搜索邻居的情况下构建：划分数据，评估每个部分内部的所有点对，并从候选集中为每个点选择边。我们给出了这样一种由三个可分离阶段组成的构建方法，并证明中间阶段决定了图的质量。候选池可以是任意一种每个点具有少量隶属关系的划分。收尾阶段将一个点的候选转化为出边；我们的方法维护一个有界堆，采用基于遮挡的剪枝并带有针对每个语料库的松弛量，同时追加反向边，仅在列表溢出处重新剪枝。脊柱是任意一个免于剪枝、能保证图从入口可达的边集；我们的方法采用随机样本上的半空间邻近边，可单调路由到每个采样点，并以生成树1/10至1/500的成本替代生成树。该收尾阶段可与任何划分器组合：在PiPNN自身的候选池上，它在六个规模从10^6到10^8个点的语料库上均胜过PiPNN的收尾方法，提升幅度为3%至14%（原文摘要在此处截断）。

    arXiv:2610.09041v1 Announce Type: cross  Abstract: Navigable graphs can be built without searching for neighbors: partition the data, evaluate every pair inside each part, and select each point's edges from the candidates. We give such a construction in three separable stages and show that the middle one decides the quality. The pool is any partition with a few memberships per point. The ending turns a point's candidates into out-edges; ours keeps a bounded heap, prunes by occlusion with a per-corpus slack, and appends reverse edges, re-pruning only where a list overflows. The spine is any edge set, exempt from the prune, that keeps the graph reachable from its entry; ours, half-space-proximal edges over a random sample, routes monotonically to every sampled point and replaces a spanning tree at 1/10 to 1/500 of its cost. The ending composes with any partitioner: on PiPNN's own candidate pool it beats PiPNN's ending on each of six corpora from $10^6$ to $10^8$ points, by 3 to 14% in di
    
[^16]: 从高召回到高效用：面向数据集自适应的LLM生成客户意图后处理

    From High Recall to High Utility: Dataset-Adaptive Post-Processing of LLM-Generated Customer Intents

    [https://arxiv.org/abs/2610.09039](https://arxiv.org/abs/2610.09039)

    提出一种数据集自适应的后处理架构，通过标准化、去重、语义向量化和自适应聚类，将LLM高召回提取的客户意图精简为稳定、可追溯且高效用的意图单元。

    

    大语言模型能够从异构的企业数据中提取有用的信号，但高召回率的提取往往产生重复、粒度不均、语义重叠或数量过多的输出，使得下游系统和人工审核人员难以有效使用。我们提出了一种为客户意图提取开发的数据集自适应后处理架构，将非结构化的客户语言转化为稳定、可追溯的意图单元。该方法将面向召回的提取与面向效用的精简相分离。源特定的预处理首先从多模态计划、稀疏的运营记录和结构化商机数据中分离出证据。随后，候选意图经过标准化和去重处理，可选择性地利用元数据信息进行增强以用于嵌入计算，在共享的语义向量空间中表示，并根据候选集的特征选择合适的聚类策略进行分组。

    arXiv:2610.09039v1 Announce Type: cross  Abstract: Large language models can extract useful signals from heterogeneous enterprise data, but high-recall extraction often produces outputs that are duplicated, uneven in granularity, semantically overlapping, or too numerous for downstream systems and human reviewers to use effectively. We present a dataset-adaptive post-processing architecture developed for Customer Intent Extraction (CIE), where unstructured customer language is transformed into stable, traceable intent units. The approach separates recall-oriented extraction from utility-oriented reduction. Source-specific preprocessing first isolates evidence from multimodal plans, sparse operational records, and structured opportunity data. Candidate intents are then standardized and deduplicated, optionally enriched with metadata for embedding computation, represented in a shared semantic vector space, and grouped using a clustering strategy selected according to the candidate set's 
    
[^17]: BEACON-SP：面向临床自杀风险评估的本体驱动GraphRAG框架

    BEACON-SP: Ontology-Grounded GraphRAG Framework for Clinical Suicide Risk Assessment

    [https://arxiv.org/abs/2610.09026](https://arxiv.org/abs/2610.09026)

    BEACON-SP通过构建整合多种自杀理论的本体，并将其与患者知识图谱结合实现本体引导的多跳推理，为临床医生提供了一种基于GraphRAG的自杀风险评估决策支持框架。

    

    我们提出BEACON-SP，一个基于本体的图检索增强生成（GraphRAG）框架，用于行为健康场景（如自杀预防）中面向临床医生的决策支持。在此类场景中，有效的评估需要整合异构的临床、行为、社会和时间维度证据。BEACON-SP将患者知识图谱与本体引导的检索相结合，支持跨诊断、药物、风险与保护因素、生活事件以及时间关系的多跳推理。该框架由一个全面的自杀预防本体驱动，该本体将三步理论、综合动机-意志模型以及自杀健康社会决定因素本体整合为患者风险因素的统一表示。我们构建了基于本体的患者知识图谱，并针对面向临床医生的问答任务对BEACON-SP进行了评估。与基于向量的检索增强方法相比……（摘要原文在此处截断）

    arXiv:2610.09026v1 Announce Type: cross  Abstract: We present BEACON-SP, an ontology-grounded Graph Retrieval-Augmented Generation (GraphRAG) framework for clinician-facing decision support in behavioral health settings such as suicide prevention, where effective assessment requires integrating heterogeneous clinical, behavioral, social, and temporal evidence. BEACON-SP combines patient knowledge graphs with ontology-guided retrieval to support multi-hop reasoning across diagnoses, medications, risk and protective factors, life events, and temporal relationships. The framework is enabled by a comprehensive suicide prevention ontology that integrates the Three-Step Theory, the Integrated Motivational-Volitional Model, and the Suicide Social Determinants of Health Ontology into a unified representation of patient risk factors. We construct ontology-grounded patient knowledge graphs and evaluate BEACON-SP for clinician-facing question answering. Compared with a vector-based retrieval-augm
    
[^18]: 面向结构化知识检索与推理的可信领域专用人工智能

    Trustworthy Domain-Specific AI for Structured Knowledge Retrieval and Reasoning

    [https://arxiv.org/abs/2610.08894](https://arxiv.org/abs/2610.08894)

    本论文提出了一种可解释的可扩展架构，通过Binary Bleed和HNMFk等创新方法将非结构化领域文本转化为结构化知识图谱与向量存储，并利用T-SRAG实现跨检索路径的动态查询路由与推理。

    

    本论文提出了一种可扩展的架构，用于将非结构化的领域专用文本转化为结构化知识，以支持检索与推理。该架构将半自动语料库整理、语义结构化、检索和推理整合为一个可解释的流水线。该研究引入了Binary Bleed，一种改进的二分搜索方法，可降低非负矩阵分解（NMF）的低秩搜索复杂度；以及具有自动潜在特征选择的层次化NMF（HNMFk），这是一种深度自适应的主题建模方法，可在领域专家指导下生成可解释的分类体系。这些表示填充了类型化知识图谱和语义对齐的向量存储库（包含提取出的潜在特征），并通过事件驱动的底层实现同步。张量结构化检索增强生成（T-SRAG）在检索路径之间动态路由查询。对比对齐将文档……

    arXiv:2610.08894v1 Announce Type: new  Abstract: This dissertation presents a scalable architecture for transforming unstructured, domain-specific text into structured knowledge for retrieval and reasoning. It integrates semi-automatic corpus curation, semantic structuring, retrieval, and inference into an interpretable pipeline.   The research introduces Binary Bleed, an adapted binary search method that reduces low-rank search complexity for Non-negative Matrix Factorization (NMF), and Hierarchical NMF with automatic latent feature selection (HNMFk), a depth-adaptive topic modeling method that produces interpretable taxonomies guided by subject matter experts. These representations populate a typed Knowledge Graph and a semantically aligned Vector Store containing extracted latent features, synchronized through an event-driven substrate.   Tensor-Structured Retrieval-Augmented Generation (T-SRAG) dynamically routes queries across retrieval paths. Contrastive alignment maps document a
    
[^19]: Madeleine：从模拟人生中学习对话记忆的非自主回忆

    Madeleine: Learning Involuntary Recall for Conversational Memory from Simulated Lives

    [https://arxiv.org/abs/2610.01118](https://arxiv.org/abs/2610.01118)

    Madeleine通过LLM人生模拟器离线学习记忆间的“非自主”关联（摊销化关联），在线阶段无需任何LLM调用、仅替换查询编码器即可接入任意向量记忆系统，以极低成本回忆起与当前话题不相似却至关重要的记忆，并在LoCoMo-Plus上取得最佳性能。

    

    长期对话助手必须在正确的时刻回忆起正确的记忆，然而最重要的记忆往往与用户当前所说的内容并不相似。现有系统通过让大语言模型（LLM）在写入或读取时进行推理来恢复这类关联，代价是每个记忆库需要数百到上千次LLM调用，且每次查询需消耗多达数千个上下文token。我们提出：关联是一种可学习的相关性，即在人类生活的展开过程中，记忆之间的逐点互信息。我们介绍Madeleine，它学习摊销化的关联：在离线阶段，一个LLM人生模拟器撰写模拟人生，其中的线索-触发对教会查询编码器在冻结的相似度之上学习残差关联；在在线阶段，它不调用任何LLM，仅需替换查询编码器即可接入任何向量记忆系统。在官方协议下的LoCoMo-Plus基准上，将Madeleine (I)接入HyperMem时达到66.6，是所有被评估系统中最高的。

    arXiv:2610.01118v1 Announce Type: cross  Abstract: A long-term conversational assistant must recall the right memory at the right moment, yet the memory that matters most is often not similar to what the user says now. Current systems recover such associations by letting an LLM reason at write or read time, at a cost of hundreds to over a thousand LLM calls per memory bank and up to several thousand context tokens per query. We argue that association is a learnable relevance: the pointwise mutual information of memories under how human lives unfold. We introduce Madeleine, which learns amortized association: offline, an LLM life simulator writes simulated lives, whose cue-trigger pairs teach a query encoder a residual association on top of frozen similarity; online, it calls no LLM and plugs into any vector memory by replacing only the query encoder. On LoCoMo-Plus under the official protocol, Madeleine (I) reaches 66.6 when plugged into HyperMem, the highest among all systems evaluate
    
[^20]: 面向压缩兼容的稀疏长上下文大语言模型推理的自索引注意力

    Self-Indexing Attention for Compression-Compatible Sparse Long-Context LLM Inference

    [https://arxiv.org/abs/2609.13205](https://arxiv.org/abs/2609.13205)

    提出了一种免训练的自索引注意力框架，利用共享的1比特符号索引在预填充和解码阶段统一实现高效token检索，同时兼容外部KV缓存压缩，在5%注意力密度下达到接近密集注意力的准确率，并获得高达6.1倍预填充和10.3倍解码的算子加速。

    

    稀疏长上下文推理需要在预填充（prefill）和解码（decode）两个阶段都进行高效的token检索。现有方法通常对这两个阶段采用不同的检索策略，导致单一检索表示无法在整个推理过程中被复用。我们提出了自索引注意力，这是一个基于共享变换域符号-幅值表示的免训练框架。键值符号提供了一个可复用的token级索引，用于分组的预填充选择和解码检索，同时该表示与外部KV缓存压缩保持兼容，无需单独的索引器元数据。这种1比特索引通过现代加速器广泛支持的按位运算实现高效检索。在5%注意力密度下，自索引注意力在LongBench和RULER基准上仍接近密集注意力的表现，并实现了高达6.1倍的预填充和10.3倍的解码注意力算子加速。与TurboQuant和DeepSeekV4-Flash结合的实验进一步验证了该方法的有效性。

    arXiv:2609.13205v1 Announce Type: cross  Abstract: Sparse long-context inference requires efficient token retrieval in both prefill and decode. Existing methods often use different retrieval strategies for the two stages, preventing one retrieval representation from being reused throughout inference. We propose Self-Indexing Attention, a training-free framework built on a shared transform-domain sign-magnitude representation. The key signs provide a reusable token-level index for grouped prefill selection and decode retrieval, while the same representation remains compatible with external KV-cache compression without separate indexer metadata. This 1-bit index enables efficient retrieval through bitwise operations widely supported by modern accelerators. At 5% attention density, Self-Indexing Attention remains close to dense attention on LongBench and RULER and achieves up to 6.1x prefill and 10.3x decode attention-operator speedups. Experiments with TurboQuant and DeepSeekV4-Flash fur
    
[^21]: 面向大语言模型维护的维基知识库的渐进式披露：一项预注册消融研究

    Progressive Disclosure for LLM-Maintained Wiki Knowledge Bases: a Preregistered Ablation

    [https://arxiv.org/abs/2607.04576](https://arxiv.org/abs/2607.04576)

    本文通过一项预注册消融实验，在四个页面内容完全相同、仅访问结构不同的LLM维护知识库版本上，检验渐进式披露（先读简洁目录和摘要、再按需打开页面）能否降低智能体问答的成本。

    

    大语言模型智能体现在经常从它们参与维护的知识库中回答问题。一种常见的直觉认为，渐进式披露应该能让这一过程更加节省成本：智能体不必加载一个庞大的索引，而是先阅读一份简洁的目录和每页一行的摘要，然后只打开它需要的页面。我们在一项预注册研究中检验了这一直觉，研究对象是一个由大语言模型维护的、包含709页的真实Markdown知识库。我们对其进行了渐进式披露的改造，并构建了四个仅在智能体访问页面方式上有所不同的版本。每个版本中的页面本身完全相同，因此任何差异都仅来自访问结构。每个版本都以三种方式进行测试：智能体遵循固定协议、自行选择路径，或被强制先加载目录。评分由来自不同模型家族的评审模型在盲测条件下，对照经过验证的参考答案进行。一项预备性试点研究改变了研究问题：一个能力较强的智能体从未加载该目（原文摘要在此处截断）。

    arXiv:2607.04576v2 Announce Type: replace  Abstract: LLM agents now often answer questions from knowledge bases they help maintain. A common intuition says progressive disclosure should make this cheaper. Instead of loading one large index, the agent reads a compact catalog and one-line page summaries, then opens only the pages it needs. We tested that intuition in a preregistered study on a real 709-page markdown knowledge base maintained by an LLM. We retrofitted it for progressive disclosure and built four versions that differ only in how the agent reaches the pages. The pages themselves are identical in every version, so any difference comes from the access structure alone. Each version was tested three ways, with the agent following a set protocol, choosing its own path, or made to load the catalog first. A judge from a different model family graded the answers blind against verified reference answers.   A preparatory pilot changed the question. A capable agent never loaded the la
    
[^22]: 基于语义分块分组的嵌入排序列表级解释方法

    Listwise Explanation of Embedding-Based Rankings via Semantic Chunk Grouping

    [https://arxiv.org/abs/2606.27980](https://arxiv.org/abs/2606.27980)

    提出ChunkGroupSHAP方法，通过将跨文档语义相关的文本分块聚类为共享特征来解释稠密嵌入排序结果，在保持上下文证据的同时控制计算维度，并证明解释单元的粒度应与排序模型的表示方式相匹配。

    

    稠密嵌入排序器通过上下文的句子级和段落级表示对文档进行评分，然而现有的列表级解释方法通常将排序结果归因于孤立的单词。我们研究了这种不匹配问题，并提出ChunkGroupSHAP——一种列表级Shapley解释方法，它将跨文档的语义相关分块聚类为共享特征，在保留上下文证据的同时，将KernelSHAP回归的维度限制在分组数量上。在MS MARCO、FinanceBench、AILACaseDocs和FinQA数据集上，结合E5系列排序器和BM25进行实验，结果表明原始分块在全部11个稠密排序器设置中的排序重构保真度均优于RankSHAP的词特征。最佳分块分组配置在其中8个设置上进一步超越了原始分块，其增量收益取决于分组范围；而在四个BM25设置中的三个里，词特征仍然表现最强。这些结果表明，解释单元应当与排序模型相匹配：上下文分块……（原文摘要截断）

    arXiv:2606.27980v2 Announce Type: replace  Abstract: Dense embedding rankers score documents through contextual sentence- and passage-level representations, yet listwise explanation methods often attribute rankings to isolated words. We study this mismatch and introduce ChunkGroupSHAP, a listwise Shapley method that clusters semantically related chunks across documents into shared features, preserving contextual evidence while bounding the KernelSHAP regression dimension by the group count. Across MS MARCO, FinanceBench, AILACaseDocs, and FinQA with E5-family rankers and BM25, raw chunks improve rank-reconstruction Fidelity over RankSHAP's word features in all 11 dense-ranker settings. The best chunk-group configuration further improves on raw chunks in eight of these settings, with the incremental benefit depending on grouping scope; word features remain strongest in three of four BM25 settings. These results show that explanation units should match the ranking model: contextual chunk
    
[^23]: LoopFM：从基础模型的历史表示中学习以用于推荐

    LoopFM: Learning frOm HistOrical RePresentations of Foundation Model for Recommendation

    [https://arxiv.org/abs/2605.29280](https://arxiv.org/abs/2605.29280)

    LoopFM通过将基础模型的中间嵌入结构化为下游垂直模型的输入特征（如用户历史序列），开辟了高带宽知识传递通道，在无需实时FM推理的情况下实现了显著AUC提升，并与知识蒸馏形成互补。

    

    知识蒸馏将大型基础模型（FM）的单一标量预测结果传递给紧凑的垂直模型（VM），由于单一标量无法传达大型FM所学习到的丰富中间知识，传递比率（即VM所能捕获的FM改进部分）会不断递减。为解决这一瓶颈，我们提出了LoopFM（从FM的历史表示中学习），该框架通过将FM的中间嵌入结构化为下游VM的输入特征（例如用户历史序列），开辟了一条高带宽的知识传递通道，且无需在服务阶段进行实时FM推理，也不需要FM与VM之间的架构耦合。我们为LoopFM提供了包含增益分解和传递比率分析的理论框架。在三个公开基准数据集上，LoopFM展现出显著的AUC提升（例如在TaobaoAd上提升超过6%），并展现出与知识蒸馏互补的知识传递能力。在工业界……

    arXiv:2605.29280v3 Announce Type: replace  Abstract: Knowledge distillation (KD) transfers a single scalar prediction from a large foundation model (FM) to compact vertical models (VMs), suffering from diminishing transfer ratio -- the fraction of FM improvement captured by the VM -- as a single scalar cannot convey the rich intermediate knowledge that larger FMs learn. To address this bottleneck, we propose LoopFM (Learning frOm HistOrical RePresentations of FM), a framework that opens a high-bandwidth transfer channel by structuring FM intermediate embeddings as input features (e.g., user history sequence) for downstream VMs, without requiring real-time FM inference at serving and architectural coupling between FM and VM. We provide a theoretical framework for LoopFM with a gain decomposition and transfer-ratio analysis. On three public benchmarks, LoopFM demonstrates strong AUC improvements (e.g., 6%+ on TaobaoAd) and complementary knowledge transfer capability with KD. On industria
    
[^24]: 检索增强生成必须超越事实依据，以表征多元观点

    Retrieval-Augmented Generation Must Move Beyond Factual Grounding to Represent Diverse Opinions

    [https://arxiv.org/abs/2604.12138](https://arxiv.org/abs/2604.12138)

    本论文指出RAG系统因过度追求事实准确性而忽视观点多样性，提出了观点感知检索框架O-RAG，通过不确定性量化和基于Wasserstein距离的统一目标，将语料级情感分布的距离降低18-48%，从而更好地表征多元观点。

    

    检索增强生成（RAG）系统建立在一个未经审视的假设之上——即查询存在正确答案，检索应当向其收敛。本立场论文指出，这造成了一种事实性偏差：RAG系统在优化时只注重降低认知不确定性，而忽视了观点丰富内容中固有的偶然不确定性。其后果超出了技术局限的范畴——包括少数声音被抹除的风险以及观点被操纵的风险。为解决这一问题，我们通过不确定性量化对观点感知检索进行了形式化，并利用Wasserstein距离推导出一个统一的目标函数。作为存在性证明，我们提出了观点感知RAG（O-RAG），该系统在索引之前利用大语言模型提取的、与实体关联的观点元数据来丰富文档内容。在电商卖家论坛和公开酒店评论的数据上，O-RAG将与语料级情感分布之间的Wasserstein距离降低了18%至48%，且人工评估结果（原文截断）……

    arXiv:2604.12138v5 Announce Type: replace-cross  Abstract: Retrieval-Augmented Generation (RAG) systems are built on an unexamined assumption - that queries have correct answers and retrieval should converge toward them. This position paper argues that this creates a factual bias where RAG systems optimize for reducing epistemic uncertainty while ignoring the aleatoric uncertainty, inherent in opinion-rich content. The consequences go beyond technical limitations- due to risk of minority voice erasure and risk of opinion manipulation. To address this, we formalize opinion-aware retrieval through uncertainty quantification and derive a unified objective using the Wasserstein distance. As an existence proof, we present Opinion-Aware RAG (O-RAG), which enriches documents with LLM-extracted, entity-linked opinion metadata before indexing. Across e-commerce seller forums and public hotel reviews, O-RAG reduces Wasserstein distance to corpus-level sentiment distributions by 18-48%, and human
    
[^25]: 基于强化学习的黑盒检索文档优化

    Document Optimization for Black-Box Retrieval via Reinforcement Learning

    [https://arxiv.org/abs/2604.05087](https://arxiv.org/abs/2604.05087)

    提出DocOpt方法，通过GRPO强化学习以检索排序提升为奖励，直接训练LLM/VLM离线重写文档以优化黑盒检索器的检索效果，从而将昂贵的计算从延迟敏感的检索路径转移到离线阶段。

    

    生成式大语言模型（LLM）越来越多地被用作检索流程中的推理时组件，执行诸如查询重写和文档重排序等任务。然而，这些在线方法将代价高昂的自回归计算直接置于对延迟敏感的检索路径上。我们探索另一条路径：利用LLM来改进文档本身，将其重写为更好的表示，从而把计算转移到离线阶段。然而，生成有用的文档重写并非易事：检索本质上是判别式的，因此有效的重写必须在检索器的相似度度量下，使文档比竞争候选文档更接近相关查询。为此，我们将文档转换表述为一个优化问题，直接训练LLM或VLM生成能够改善检索效果的重写。我们的方法DocOpt使用GRPO，以检索器排序的改进作为奖励，仅需对检索器进行黑盒访问。

    arXiv:2604.05087v4 Announce Type: replace  Abstract: Generative large language models (LLMs) are increasingly used as inference-time components in retrieval pipelines, for tasks such as query rewriting and document reranking. However, these online approaches place costly autoregressive computation directly on the latency-critical retrieval path. We explore an alternative axis: using LLMs to improve documents instead, rewriting them into better representations and shifting computation offline. Yet producing a useful document rewrite is not straightforward: retrieval is inherently discriminative, so an effective rewrite must make a document more similar to relevant queries than competing candidates under the retriever's notion of similarity. We therefore formulate document transformation as an optimization problem, directly training an LLM or VLM to produce rewrites that improve retrieval. Our approach, DocOpt, uses GRPO with retriever ranking improvements as rewards, requires only black
    
[^26]: 向量化字典树：面向加速器上基于大语言模型的生成式检索的高效约束解码

    Vectorizing the Trie: Efficient Constrained Decoding for LLM-based Generative Retrieval on Accelerators

    [https://arxiv.org/abs/2602.22647](https://arxiv.org/abs/2602.22647)

    提出 STATIC 方法，通过将前缀树展平为 CSR 稀疏矩阵，把不规则的树遍历转化为完全向量化的稀疏矩阵运算，从而在 TPU/GPU 上实现高效、可扩展的基于大语言模型生成式检索的约束解码。

    

    生成式检索已成为基于大语言模型（LLM）推荐系统中的一种强大范式。然而，工业级推荐系统通常需要根据业务逻辑将输出空间限制在受约束的物品子集内（例如强制内容时效性或商品类目），而标准的自回归解码无法原生支持这种约束。此外，现有的基于前缀树（Trie）的约束解码方法在硬件加速器（TPU/GPU）上会带来严重的延迟损失。在本工作中，我们提出了 STATIC（用于约束解码的稀疏转移矩阵加速字典树索引），这是一种高效且可扩展的约束解码技术，专为在 TPU/GPU 上实现高吞吐量的基于 LLM 的生成式检索而设计。通过将前缀树展平为静态压缩稀疏行（CSR）矩阵，我们将不规则的树遍历转化为完全向量化的稀疏矩阵运算，从而释放了大规模并行计算的能力。

    arXiv:2602.22647v3 Announce Type: replace-cross  Abstract: Generative retrieval has emerged as a powerful paradigm for LLM-based recommendation. However, industrial recommender systems often benefit from restricting the output space to a constrained subset of items based on business logic (e.g. enforcing content freshness or product category), which standard autoregressive decoding cannot natively support. Moreover, existing constrained decoding methods that make use of prefix trees (Tries) incur severe latency penalties on hardware accelerators (TPUs/GPUs). In this work, we introduce STATIC (Sparse Transition Matrix-Accelerated Trie Index for Constrained Decoding), an efficient and scalable constrained decoding technique designed specifically for high-throughput LLM-based generative retrieval on TPUs/GPUs. By flattening the prefix tree into a static Compressed Sparse Row (CSR) matrix, we transform irregular tree traversals into fully vectorized sparse matrix operations, unlocking mass
    
[^27]: 用于捆绑包推荐的超图增强双卷积网络

    Hypergraph-Enhanced Dual Convolutional Network for Bundle Recommendation

    [https://arxiv.org/abs/2312.11018](https://arxiv.org/abs/2312.11018)

    HED 通过构建包含用户-捆绑包、用户-物品、捆绑包-物品交互及用户内部、捆绑包内部关系的完整超图，并将全超图传播与用户-捆绑包双卷积分支耦合，在保留推荐特有信号的同时引入物品感知的高阶上下文，在 NetEase 和 Youshu 数据集上较最强基线显著提升了捆绑包推荐性能。

    

    arXiv:2312.11018v3 公告类型：replace-cross 摘要：捆绑包推荐是对相关物品的集合而非孤立物品进行排序。其核心挑战在于在不丢失捆绑包排序所需信号的前提下，连接用户偏好、物品交互与捆绑包构成。我们提出了超图增强双卷积神经网络（HED），该网络构建了一个完整的超图，其中包含用户-捆绑包、用户-物品和捆绑包-物品交互，以及用户内部和捆绑包内部关系。HED 将完整超图传播与用户-捆绑包分支相结合，使物品感知的高阶上下文能够为排序提供信息，同时保留推荐特有的信号。在 NetEase 数据集上，HED-128 在六项报告指标上较最强基线提升了 5.04%–6.97%；在 Youshu 数据集上，HED-64 提升了 1.87%–4.56%。消融实验结果支持了用户-捆绑包分支和类型内关系二者的贡献，敏感性分析确定了稳定的运行范围。

    arXiv:2312.11018v3 Announce Type: replace-cross  Abstract: Bundle recommendation ranks sets of related items rather than isolated items. Its central challenge is to connect user preferences, item interactions, and bundle composition without losing the signals needed to rank bundles. We propose Hypergraph-Enhanced Dual Convolutional Neural Network (HED), which constructs a complete hypergraph containing user--bundle, user--item, and bundle--item interactions together with intra-user and intra-bundle relations. HED couples complete-hypergraph propagation with a user--bundle branch, allowing item-aware higher-order context to inform ranking while preserving recommendation-specific signals. On NetEase, HED-128 improves over the strongest baseline by 5.04--6.97% across the six reported metrics; on Youshu, HED-64 improves by 1.87--4.56%. Ablation results support the contributions of both the user--bundle branch and intra-type relations, and sensitivity analyses identify stable operating rang
    

