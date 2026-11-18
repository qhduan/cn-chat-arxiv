# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CARE: Conversational Affective Response Retrieval for Internet Memes](https://arxiv.org/abs/2611.23005) | 提出了CARE对话式情感响应检索框架，将情感增强的CLIP检索器与基于Jina的重排序器相结合，利用情感描述、弱相关性标签和点赞参与度信号，从海量表情包中检索与对话语境和情感表达相匹配的响应表情包。 |
| [^2] | [When the tail is earned by its companions: measuring label dependence in long-tailed multi-label classification](https://arxiv.org/abs/2611.22859) | 该论文提出了可在训练前评估标签依赖性的标签内估计量、读取规则与统计量，揭示稀有标签的召回表现依赖频繁伴生标签而非模型真正学到稀有标签本身的现象。 |
| [^3] | [Music Similarity Learning in Sparse Latin Scenarios](https://arxiv.org/abs/2611.22846) | 该论文提出一种基于公开元数据的轻量级混合回归模型，结合内容过滤与项目协同过滤，在共现数据稀缺的拉丁音乐零样本评估中实现准确的相似度预测，无需音频信号处理或庞大的计算资源。 |
| [^4] | [Better Rankers, Worse Retrievers: The Hidden Cost of RL Post-Training in Generative Recommendation](https://arxiv.org/abs/2611.22838) | 本文揭示了强化学习后训练在生成式推荐中的隐性代价：由于每个用户只有一个真实标签，RL将模型采样的合理替代候选一律视为错误并给予负优势，导致模型在提升列表顶部排序准确率的同时显著降低了候选列表的召回率，成为更好的排序器却是更差的检索器。 |
| [^5] | [IntentGR: Enhancing Intent for End-to-End Personalized Generative Retrieval in Search](https://arxiv.org/abs/2611.22788) | 提出 IntentGR，通过在语义标识符分词和序列生成两个阶段显式引入查询意图与查询相关信息，弥补了端到端个性化生成式检索中意图引导不足的问题，使检索结果更好对齐用户即时搜索意图。 |
| [^6] | [Learning Cross-Domain Media Similarity from Amazon Reviews](https://arxiv.org/abs/2611.22756) | 提出一种监督学习方法，利用公开可得的亚马逊产品评论中的语义和时间特征，以客户评论历史中的产品共现作为训练目标，成功学习跨媒体领域（书籍与电影）的项目相似性，摆脱了对稀疏或不可获取用户数据的依赖。 |
| [^7] | [MIPR: A Benchmark for Reasoning-Intensive Multimodal Intent-to-Product Retrieval with Multi-Category Solutions](https://arxiv.org/abs/2611.22509) | 该论文提出了 MIPR 基准，首个针对需要推理、且一个意图可由多类别商品满足的多模态意图到商品检索任务，包含 3,522 条查询和 30,731 个商品，实验显示现有最优模型与理想上限仍有较大差距，凸显了该任务的挑战性。 |
| [^8] | [From Relevance to Utility: Rubric Alignment in Yuanbao Search](https://arxiv.org/abs/2611.22503) | 该论文提出“量规对齐搜索”框架，通过为每个查询构建从粗到细、包含加权正向与负面标准的量规，将整个搜索流水线统一对齐到面向LLM推理的文档集合效用，而非传统的单文档相关性排序。 |
| [^9] | [More Than a Score: Reinforcement Learning with Verbalized Feedback for Generative Recommendation](https://arxiv.org/abs/2611.22410) | 提出VerRec框架，通过用户模拟器生成个性化的语言化反馈来引导强化学习优化生成式推荐，克服了传统标量奖励无法指明候选物品与用户偏好冲突之处及改进方向的局限。 |
| [^10] | [Deal With It: On Better Schedules for Listwise Reranking](https://arxiv.org/abs/2611.22392) | 本文证明了在并行列表式重排序调度中，按检索器排名轮转“发牌”划分候选是均期望削减最少的最优分组方式，并提出简单高效的 DealRank 调度方案，在保持接近最优重排序质量的同时实现 1.1 至 1.4 倍的提速。 |
| [^11] | [RICoT: Recurrent Interleaved Chain-of-Thought for Page Stream Segmentation](https://arxiv.org/abs/2611.22382) | 该论文提出RICoT循环交错思维链提示框架，首次将页面流分割重新构想为对整个流的顺序处理过程，像读者一样在滑动窗口内逐页推理并携带已读内容的紧凑记录以控制成本，同时发布了包含205个真实金融页面流的FinancePSS-205基准数据集。 |
| [^12] | [NONSENSE: Non-Semantic Strings Expose Semanticity Gaps in Dense Retrieval](https://arxiv.org/abs/2611.22353) | 提出NONSENSE黑盒稀疏编辑探针，通过构造禁止复制查询词的明显非语义字符串，证明其稠密相似度得分常超过检索器最佳的查询相关段落，从而揭示了稠密检索中最近邻地位与真实语义含义相分离的语义性鸿沟。 |
| [^13] | [GuardAgent: A Memory-Augmented Agentic Framework for Adaptive Detection of Adversarial Promotions in LLM-Based Recommendation](https://arxiv.org/abs/2611.22274) | 该论文提出了首个针对基于大语言模型推荐系统的推广驱动攻击检测的系统性研究，设计了通过感知-行动-反思-记忆循环运作的智能体框架GuardAgent，以实现对攻击的自适应检测。 |
| [^14] | [Teaching Language Models to Narrow Boolean Patent Searches from Search Feedback](https://arxiv.org/abs/2611.22270) | 该论文利用日本专利局检索报告提炼专家策略，并提出行级“保留引用的收窄信用”学习方法对语言模型进行离线迭代优化，使其生成的布尔专利检索策略在保持引用召回率的同时大幅收窄候选集合，将包含引用文献的中位数集合规模从 768 篇降至 50 篇以下。 |
| [^15] | [Cluster-Guided Decoding with Trust-Region Constraints for Personalized Recommendation](https://arxiv.org/abs/2611.22163) | CG2R 通过学习语义簇层次结构并将簇级先验直接融入商品打分规则来塑造个性化推荐排序，且先验基于簇质心构建，因此对无历史记录的冷启动用户依然有效。 |
| [^16] | [SIMQ: A Single Graph-based Index for Multi-Vector Query](https://arxiv.org/abs/2611.22079) | SIMQ提出了一种基于单一图索引的多向量查询框架，通过数据感知的采样聚类和逐词元图导航，同时突破了现有近似方法在计算矩阵开销和召回稳定性方面的根本瓶颈。 |
| [^17] | [Beyond Bitwidth: Deployment-Constrained Mixed-Precision Quantization for Multilingual LLM Inference on NVIDIA Blackwell](https://arxiv.org/abs/2611.21974) | 该论文在NVIDIA B200上对Qwen3-8B进行了部署约束驱动的低比特与混合精度量化实测评估，发现INT2/INT3全面退化且模块恢复效应具有强非加性，最终提出一个NVFP4与INT8混合的配置，在通过全部预注册多语言与推理精度门槛的同时将权重内存压缩至5.917 GiB。 |
| [^18] | [Complexity and precision in quantum associative retrieval](https://arxiv.org/abs/2611.21941) | 该论文证明了量子联想检索相对于经典暴力搜索具有指数级复杂度优势、相对于经典神经网络具有多项式级容量优势，并能实现高效且高精度的模式检索。 |
| [^19] | [Urban Representation Learning for Identifying Potential Disparities in Crime Reporting](https://arxiv.org/abs/2611.21825) | 提出了一种自监督城市表征学习框架，通过邻居画像差异（NPD）指标量化各区域与其表征相似同伴区域的报告画像差异，从而识别开放犯罪数据中的潜在报告偏差。 |
| [^20] | [Evaluating AST-Derived Knowledge Graph Retrieval for AIOps Incident Diagnosis in Apache Kafka](https://arxiv.org/abs/2611.21625) | 本文在Apache Kafka事件诊断任务上系统评估了基于AST派生知识图谱的检索方法，发现仅向量检索已是强基线（Recall@20达0.956）且图增强方法难以显著超越，而分阶段图扩展可将重排序开销减少约52%。 |
| [^21] | [EE-Motion: An EEG-based Web Architecture for Live Affective Music Recommendation](https://arxiv.org/abs/2611.21032) | 该论文提出了EE-Motion，一个通过Web Bluetooth连接消费级EEG头带、实时将脑电信号映射为效价-唤醒情绪坐标，并借助情感适配器统一多种推荐后端、无需重新训练即可为匿名用户提供实时情感化音乐推荐的网页架构。 |
| [^22] | [RACE: Ranking-Uncertainty Aware Capacity-Adaptive Customization for Efficient Sequential Recommendation](https://arxiv.org/abs/2611.18244) | 提出RACE框架，通过捕获模型预测的排序不确定性并将其转化为主动的资源分配器，以不确定性感知的压缩比率动态调整模型结构和输入序列，从而解决序列推荐中统一压缩比率导致的容量-复杂度失配问题，实现资源受限环境下的高效推荐。 |
| [^23] | [Explanations that Transfer: Interpreting Medical Retrieval with Sparse AutoEncoders](https://arxiv.org/abs/2611.18179) | 本文提出将稀疏自编码器应用于医学信息检索，通过结合重建目标与InfoNCE对比损失，将不透明的稠密文本嵌入转化为语义可解释的表示，同时保持对新临床语料库的可迁移性。 |
| [^24] | [Lightfoot: Graph Retrieval-Augmented Generation with Exact, Model-Free Document Deletion](https://arxiv.org/abs/2611.11010) | Lightfoot 是一种图检索增强生成系统，它将图中的每个实体、链接、事实和同义词边都定义为逐段落提取结果的确定性函数，从而实现无需语言模型调用的检索以及作为数据库事务的精确文档删除（在 11,402 段语料库上删除 10% 仅需 14 秒且结果与全新索引完全一致），解决了现有 GraphRAG 系统的模型调用成本和文档删除不精确问题。 |
| [^25] | [LEGIT: Live Eligibility Gating over Identifier Tries for Constrained Decoding of Heterogeneous Pages](https://arxiv.org/abs/2611.10357) | 提出 LEGIT 约束机制，通过对标识符字典树叶节点进行深度优先编号、把随请求和 token 动态变化的资格规则转化为高效的范围查找，使生成式推荐系统在整页解码时能精确且快速地保证每个语义 ID 真实、合规且不重复。 |
| [^26] | [Enhancing Stock Market Information Retrieval with Retrieval-Augmented Generation](https://arxiv.org/abs/2611.03747) | 本文提出一种整合检索与生成方法的RAG系统，用于生成准确且相关的股票市场信息响应，并通过ROUGE指标验证了其有效性。 |
| [^27] | [REGER: Modeling inter-item Relations with Semantic IDs for Generative Recommendation](https://arxiv.org/abs/2610.37584) | 提出REGER统一框架，通过生成式语义检索预训练使GR模型继承物品间语义相似性，并将物品间关系显式融入语义预训练、历史编码和SID生成的全过程。 |
| [^28] | [Retrieving Complementary Evidence for Multi-Hop RAG with Semantic Gain Graphs](https://arxiv.org/abs/2610.15285) | 提出SG$^2$RAG框架，利用大语言模型离线评估文本块之间的非冗余语义增益并构建可复用的语义增益图，再通过个性化PageRank在线检索多跳RAG所需的互补证据，在多个基准数据集上取得最优性能。 |
| [^29] | [CodeHID: Learning an Addressable Hierarchical Code Index for Generative Code Retrieval](https://arxiv.org/abs/2608.24089) | CodeHID通过构建分层的可寻址代码索引，将代码检索从平面匹配转变为从粗到细的语义地址生成，从而提升对相似代码候选的区分能力。 |
| [^30] | [RecHarness: A Bandit-Routed Agentic Harness for Self-Evolving Recommender Systems](https://arxiv.org/abs/2607.29241) | 提出RecHarness框架，通过将优化过程解耦为“多臂老虎机路由器选择修改方向+LLM生成具体优化假设与代码修改”两个步骤，并借助跳跃盆地机制在局部优化停滞时触发结构性跳跃，实现推荐模型的自演化自动优化。 |
| [^31] | [Evidence Attribution in Visual Document Understanding without Coordinates or Region Labels](https://arxiv.org/abs/2607.24651) | 该研究提出以“模型逐字引用证据、由多模态检索器定位出处”的语言接口替代传统的坐标边界框接口，表明视觉文档理解中的归因幻觉部分源于坐标表达能力的局限。 |
| [^32] | [Beyond Retrieval: Learning Compact User Representations for Scalable LLM Personalization](https://arxiv.org/abs/2606.04547) | 提出TAP-PER框架，通过时序注意力前缀嵌入将用户偏好编码为紧凑的可学习表示，摆脱了检索式个性化对检索质量的依赖以及参数式个性化随用户规模增长的高昂存储成本，实现可扩展的大语言模型个性化。 |
| [^33] | [Efficient Generative Retrieval for E-commerce Search with Semantic Cluster IDs and Expert-Guided RL](https://arxiv.org/abs/2605.14434) | 该论文提出CQ-SID方法，通过类目感知对比学习与残差量化VAE将商品编码为层次化语义聚类ID以大幅降低束搜索复杂度，并引入专家引导的强化学习方法EG-GRPO，将生成式检索定位为工业级电商搜索召回阶段的高效补充方案。 |
| [^34] | [NewsReX: An Open-Source Multi-Framework for Neural News Recommendation](https://arxiv.org/abs/2508.21572) | NewsReX 是一个开源的多框架神经新闻推荐系统，通过 YAML 模型规范和 Hydra 配置系统统一了多种推荐模型，支持 PyTorch 与 JAX/Flax 两个后端间的单命令切换与数值一致性验证，解决了该领域结果复现与比较困难的问题。 |

# 详细

[^1]: CARE：面向网络表情包的对话式情感响应检索

    CARE: Conversational Affective Response Retrieval for Internet Memes

    [https://arxiv.org/abs/2611.23005](https://arxiv.org/abs/2611.23005)

    提出了CARE对话式情感响应检索框架，将情感增强的CLIP检索器与基于Jina的重排序器相结合，利用情感描述、弱相关性标签和点赞参与度信号，从海量表情包中检索与对话语境和情感表达相匹配的响应表情包。

    

    对话式表情包检索旨在支持在线交流中的响应选择，其相关性既取决于语境含义，也取决于情感表达。我们利用来自豆瓣和 Imgflip 的两个特定语言数据集来研究这一任务，数据集包含 5,583 个讨论话题下的 98,506 条表情包记录。会话线程中的从属关系提供了弱相关性标签，而点赞数则提供了隐式的排序反馈。自动提取的情感指标、面部表情指标以及跨模态一致性特征支持了对用户参与度的探索性分析。我们提出了 CARE——一个对话式情感响应检索框架，它将情感增强的 CLIP 检索器与基于 Jina 的重排序器相结合。检索器通过门控融合机制引入自由格式的情感描述，并从每个查询的多个正例样本中进行学习。重排序器将查询与图像的交互和情感表示相结合，并使用基于参与度的排序结果进行训练。

    arXiv:2611.23005v1 Announce Type: new  Abstract: Conversational meme retrieval supports response selection in online communication, where relevance depends on both contextual meaning and affective expression. We study this task using two language-specific collections from Douban and Imgflip, containing 98,506 meme records across 5,583 discussion topics. Thread membership supplies weak relevance labels, while likes provide implicit ranking feedback. Automatically extracted sentiment, facial-expression indicators, and cross-modal consistency features support an exploratory analysis of engagement. We propose CARE, a Conversational Affective Response Retrieval framework that combines an emotion-enhanced CLIP retriever with a Jina-based reranker. The retriever incorporates free-form emotion descriptors through gated fusion and learns from multiple positive examples per query. The reranker combines query--image interactions with emotion representations and is trained using engagement-based r
    
[^2]: 当尾部由其伴生标签挣得：测量长尾多标签分类中的标签依赖性

    When the tail is earned by its companions: measuring label dependence in long-tailed multi-label classification

    [https://arxiv.org/abs/2611.22859](https://arxiv.org/abs/2611.22859)

    该论文提出了可在训练前评估标签依赖性的标签内估计量、读取规则与统计量，揭示稀有标签的召回表现依赖频繁伴生标签而非模型真正学到稀有标签本身的现象。

    

    稀有标签很少单独出现，分类器可以从与其共现的频繁标签（即其“伴生标签”）中恢复出该稀有标签，而无需真正学到这个标签本身。现有的诊断方法都需要已训练好的模型：没有一种方法能在训练前判断某个语料库是否会出现这种失效，也无法揭示标准聚合指标所掩盖的问题。Macro-F1，以及任何基于逐标签混淆计数构建的度量，都不会因标签召回率在“有伴生标签出现”与“无伴生标签出现”两种情形之间的分配方式不同而改变。我们引入了刻画这种划分的标签内估计量，提出了一种读取规则（在该规则下，只有当无伴生召回率上升时，依赖性的下降才会被计入），并提出一个基于训练标签的统计量，它能够在训练之前界定报告召回率可能超过无伴生召回率的程度。在23个语料库、7种训练目标和五类标签相关机制上的实验表明，在定义了该估计量的全部六个语料库中，无伴生召回率均存在不足，长尾部分中高达18.4%的（召回）……

    arXiv:2611.22859v1 Announce Type: cross  Abstract: A rare label seldom occurs alone, and a classifier can recover it from the frequent labels it co-occurs with, its companions, without having learned the label itself. Existing diagnostics of this need a trained model: none says before training whether a corpus admits the failure, or what the standard aggregate hides. Macro-F1, like any measure built from per-label confusion counts, is unchanged by how a label's recall divides between occurrences with and without a companion. We introduce a within-label estimand of that division, a reading rule under which a fall in dependence counts only if companion-free recall rises, and a statistic of training labels that bounds, before training, how far reported recall can exceed companion-free recall. Across 23 corpora, 7 training objectives and five families of label-correlation mechanisms, companion-free recall falls short on all six corpora where the estimand is defined. Up to 18.4% of the tail
    
[^3]: 稀疏拉丁音乐场景下的音乐相似度学习

    Music Similarity Learning in Sparse Latin Scenarios

    [https://arxiv.org/abs/2611.22846](https://arxiv.org/abs/2611.22846)

    该论文提出一种基于公开元数据的轻量级混合回归模型，结合内容过滤与项目协同过滤，在共现数据稀缺的拉丁音乐零样本评估中实现准确的相似度预测，无需音频信号处理或庞大的计算资源。

    

    在这项工作中，我们提出了一种轻量级、可解释的方法，可以从公开可用的元数据中估计音乐项目之间的相似度。我们的混合回归模型结合了基于内容的过滤与经典的基于项目的协同过滤。我们通过对拉丁美洲音乐进行零样本评估来测试泛化能力，而拉丁音乐正是一种共现数据稀缺的音乐类型。对这些未见过的、稀疏的配对进行准确的相似度预测表明，我们的方法对影响许多现有系统的区域性数据差距具有鲁棒性，且无需深度多模态方法所需的音频信号处理或庞大的计算预算。

    arXiv:2611.22846v1 Announce Type: new  Abstract: In this work we propose a lightweight, interpretable approach that estimates item-item music similarity from publicly available metadata. Our hybrid regression model combines content-based filtering with the classic item-based collaborative filtering. We test generalization via a zero-shot evaluation on Latin American music, a genre for which co-occurrence data is scarce. Accurately predicting similarity for these unseen, sparse pairs shows that our approach is robust to the regional gap affecting many existing systems, without the audio signal processing or the computational budget that deep multimodal approaches would require.
    
[^4]: 更好的排序器，更差的检索器：生成式推荐中强化学习后训练的隐性代价

    Better Rankers, Worse Retrievers: The Hidden Cost of RL Post-Training in Generative Recommendation

    [https://arxiv.org/abs/2611.22838](https://arxiv.org/abs/2611.22838)

    本文揭示了强化学习后训练在生成式推荐中的隐性代价：由于每个用户只有一个真实标签，RL将模型采样的合理替代候选一律视为错误并给予负优势，导致模型在提升列表顶部排序准确率的同时显著降低了候选列表的召回率，成为更好的排序器却是更差的检索器。

    

    生成式推荐器通过语言模型解码用户下一个物品的语义ID，在监督微调（SFT）之后使用强化学习（RL）进行后训练已成为标准做法。相关增益大多是在前十项结果上报告的，然而在实际部署中，此类模型需要将数百个候选结果交给下游的排序器或奖励模型。我们揭示了强化学习后训练的一个隐性代价：它提升了列表顶部的准确率（小K值下的hit@K），但在我们研究的全部五个数据集上，都降低了候选列表的召回率（数百K值下的hit@K）。模型变成了更好的排序器，却成了更差的检索器。受控实验将这一性能损失追溯到强化学习为其采样候选（rollout）所分配的负优势。根本原因在于每个用户只记录了一个真实的下一个物品，因此强化学习将所有其他候选都视为错误。然而这些候选恰恰是模型最合理的替代选择，而且往往……（摘要原文在此处截断）

    arXiv:2611.22838v1 Announce Type: cross  Abstract: Generative recommenders decode the semantic ID of a user's next item with a language model, and post-training them with reinforcement learning (RL) after supervised fine-tuning (SFT) has become standard practice. The gains are mostly reported on the first ten results, yet in deployment such models hand hundreds of candidates to a downstream ranker or reward model. We show that RL post-training has a hidden cost: it improves accuracy at the top of the list (hit@K for small K) but, on all five datasets we study, lowers the recall of the candidate list (hit@K for K in the hundreds). The model becomes a better ranker and a worse retriever. Controlled experiments trace the loss to the negative advantages that RL assigns to the candidates it samples, the rollout. The root cause is that only one next item is logged per user, so RL treats every other candidate as wrong. Yet these candidates are the model's most plausible alternatives, and ofte
    
[^5]: IntentGR：增强意图以实现搜索中的端到端个性化生成式检索

    IntentGR: Enhancing Intent for End-to-End Personalized Generative Retrieval in Search

    [https://arxiv.org/abs/2611.22788](https://arxiv.org/abs/2611.22788)

    提出 IntentGR，通过在语义标识符分词和序列生成两个阶段显式引入查询意图与查询相关信息，弥补了端到端个性化生成式检索中意图引导不足的问题，使检索结果更好对齐用户即时搜索意图。

    

    搜索中的个性化生成式检索应当同时满足用户的即时搜索意图与长期偏好。近期主要在推荐场景中探索的端到端方法，将物品分词与序列生成统一在同一个模型内，使语义标识符（SIDs）能够面向下游生成目标进行优化。然而，这些方法通常仅将查询当作输入特征，未能显式地利用它来指导标识符学习或序列生成。这导致意图引导在分词阶段和生成阶段都比较薄弱，使得生成的语义标识符偏向高频词元模式，而非当前的搜索意图。在本工作中，我们发现将额外的查询相关信息融入端到端生成式检索框架十分重要。基于这一发现，我们提出了 IntentGR，一种意图增强的端到端……（原文摘要在此处截断）

    arXiv:2611.22788v1 Announce Type: new  Abstract: Personalized generative retrieval in search should jointly satisfy the user's immediate search intent and long-term preference. Recent end-to-end approaches, primarily explored in recommendation, unify item tokenization and sequence generation within a single model, allowing semantic identifiers (SIDs) to be optimized toward the downstream generation objective. However, these methods often treat the query merely as an input feature, without explicitly leveraging it to guide either identifier learning or sequence generation. This leaves intent guidance weak in both the tokenization and the generation stage, causing generated semantic identifiers to drift toward high-frequency token patterns rather than the current search intent. In this work, we find that incorporating additional query-related evidence into the framework of end-to-end generative retrieval is important. Based on this finding, we propose IntentGR, an intent-enhanced end-to-
    
[^6]: 从亚马逊评论中学习跨域媒体相似性

    Learning Cross-Domain Media Similarity from Amazon Reviews

    [https://arxiv.org/abs/2611.22756](https://arxiv.org/abs/2611.22756)

    提出一种监督学习方法，利用公开可得的亚马逊产品评论中的语义和时间特征，以客户评论历史中的产品共现作为训练目标，成功学习跨媒体领域（书籍与电影）的项目相似性，摆脱了对稀疏或不可获取用户数据的依赖。

    

    协同过滤是许多推荐算法的基本基础。它通过分析用户的历史记录，然后推荐与该历史中项目相似的项目。这种方法通常依赖用户数据来计算项目相似性，这些数据对商业系统来说是广泛可用的，但对于独立系统和研究来说通常是稀疏的或无法获取的，尤其是在跨多个媒体领域的情况下。在这项工作中，我们提出了一种使用监督学习来泛化项目之间相似性信息的方法。作为模型的输入，我们使用从公开可用的亚马逊产品评论中提取的语义和时间特征。作为训练目标，我们使用亚马逊客户评论历史中项目（产品）的共现情况。我们在两个媒体领域（书籍和电影）上评估了预测相似性的质量。实验结果证明了我们方法的鲁棒性，产生了积极的（效果）。

    arXiv:2611.22756v1 Announce Type: new  Abstract: Collaborative filtering is a basic foundation of many recommendation algorithms. It works by analyzing a user's history and then suggesting items similar to those in that history. This method typically relies on user data to compute item similarity, data that is widely available to commercial systems but is usually sparse or inaccessible for independent systems and research, especially across multiple media domains. In this work, we propose a method to generalize similarity information between items using supervised learning. As input to our model, we use semantic and temporal features extracted from Amazon product reviews, which are publicly available. As the training target, we use item (product) co-occurrence in the reviewing history of Amazon customers. We evaluate the quality of the predicted similarities across two media domains: books and movies. Experimental results demonstrate the robustness of our methodology, yielding positive
    
[^7]: MIPR：一个面向需要推理的多模态意图到商品检索（含多类别解决方案）的评测基准

    MIPR: A Benchmark for Reasoning-Intensive Multimodal Intent-to-Product Retrieval with Multi-Category Solutions

    [https://arxiv.org/abs/2611.22509](https://arxiv.org/abs/2611.22509)

    该论文提出了 MIPR 基准，首个针对需要推理、且一个意图可由多类别商品满足的多模态意图到商品检索任务，包含 3,522 条查询和 30,731 个商品，实验显示现有最优模型与理想上限仍有较大差距，凸显了该任务的挑战性。

    

    真实世界的电商查询往往只表达用户意图而不指明目标商品，且同一意图可能由多个类别的商品来满足。然而，现有多模态检索基准通常假设目标相对明确，对具有多类别解决方案的意图检索探索不足。为填补这一空白，我们提出了 MIPR，一个多模态意图到商品检索基准，包含 3,522 条多模态查询和 30,731 个商品的共享语料库。MIPR 具有推理密集型检索、多类别解决方案和细粒度相关性标签的特点。实验表明，最好的基于嵌入的检索器在完整查询集上仅达到 49.23% 的 nDCG@10，在具有多个解决方案类别的查询上达到 55.86% 的解决方案类别 recall@10。通过重排，这两个分数分别提升至 60.11% 和 67.00%，但与候选池的乐观上限之间仍存在显著差距。这些结果确立了……（原文在此截断）

    arXiv:2611.22509v1 Announce Type: new  Abstract: Real-world e-commerce queries often express user intent without naming a target product, and a single intent may be fulfilled by products from multiple categories. However, existing multimodal retrieval benchmarks typically assume a relatively well-defined target, leaving retrieval for intents with multi-category solutions less explored. To fill this gap, we introduce MIPR, a benchmark for Multimodal Intent-to-Product Retrieval comprising 3,522 multimodal queries and a shared corpus of 30,731 products. MIPR features reasoning-intensive retrieval, multi-category solutions, and fine-grained relevance labels. Experiments show that the best embedding-based retriever achieves 49.23% nDCG@10 on the full query set, and 55.86% solution-category recall@10 on queries with multiple solution categories. Reranking improves these scores to 60.11% and 67.00%, respectively, yet substantial gaps to the candidate-pool oracle remain. These results establis
    
[^8]: 从相关性到效用：元宝搜索中的量规对齐

    From Relevance to Utility: Rubric Alignment in Yuanbao Search

    [https://arxiv.org/abs/2611.22503](https://arxiv.org/abs/2611.22503)

    该论文提出“量规对齐搜索”框架，通过为每个查询构建从粗到细、包含加权正向与负面标准的量规，将整个搜索流水线统一对齐到面向LLM推理的文档集合效用，而非传统的单文档相关性排序。

    

    智能体搜索要求搜索引擎超越单个文档的相关性排序，转而构建为下游LLM推理优化的文档集合。先前的工作要么通过孤立地改进各个流水线阶段来解决这一问题，要么在经过相关性训练的上游模块之上应用基于强化学习的选择方法，导致整个技术栈未能对齐到面向推理的效用。我们提出“量规对齐搜索”，这是一个在单一监督面下统一传统搜索流水线的框架。对于每个查询，我们从粗到细地构建一个量规，明确规定文档集合必须满足的加权正向标准以及必须避免的负面标准。从这个量规中，我们推导出两个训练信号。首先，一个轻量级标注器将基于量规的判断蒸馏为逐文档效用标签，这些标签用于监督用于检索的对比式双编码器，以及一个通过四个量规对齐评分头训练的多任务排序器。

    arXiv:2611.22503v1 Announce Type: new  Abstract: Agentic search demands that a search engine move beyond per document relevance ranking to assemble document sets optimized for downstream LLM reasoning. Prior work addresses this either by improving individual pipeline stages in isolation or by applying RL based selection on top of relevance trained upstream modules, leaving the full stack unaligned to utility for reasoning. We propose rubric aligned search, a framework that unifies the traditional search pipeline under a single supervision surface. For each query, we derive a rubric from coarse to fine, specifying weighted positive criteria that the document set must satisfy and negative criteria that it must avoid. From this rubric we derive two training signals. First, a lightweight labeler distills rubric grounded judgments into per document utility labels, which supervise a contrastive dual encoder for retrieval and a multi task ranker trained with four rubric aligned scoring heads.
    
[^9]: 不仅仅是分数：基于语言化反馈的生成式推荐强化学习

    More Than a Score: Reinforcement Learning with Verbalized Feedback for Generative Recommendation

    [https://arxiv.org/abs/2611.22410](https://arxiv.org/abs/2611.22410)

    提出VerRec框架，通过用户模拟器生成个性化的语言化反馈来引导强化学习优化生成式推荐，克服了传统标量奖励无法指明候选物品与用户偏好冲突之处及改进方向的局限。

    

    大语言模型（LLM）正越来越多地被用于生成式推荐，即直接根据用户的交互历史生成推荐的物品。为了进一步提升推荐质量，强化学习（RL）已被应用于将这些模型与推荐目标进行对齐。然而，现有的强化学习方法依赖于标量奖励来隐式地引导探索。这些奖励无法说明候选物品的哪些方面与用户偏好存在冲突，以及下一个候选物品应如何改进。为填补这一空白，我们提出了VerRec，一个由语言化反馈引导的生成式推荐强化学习框架。具体而言，用户模拟器基于用户的历史偏好和候选物品画像生成个性化的语言化反馈。策略利用该反馈来生成经过改进的推荐结果。原始候选与改进后的候选共同构成一个扩展组，并通过组相对机制进行联合优化。

    arXiv:2611.22410v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used for generative recommendation, where recommended items are generated directly from users' interaction histories. To further improve recommendation quality, reinforcement learning (RL) has been applied to align these models with recommendation objectives. However, existing RL methods rely on scalar rewards to implicitly guide exploration. These rewards do not specify which aspects of a candidate conflict with the user's preferences or how the next candidate should differ. To address this gap, we propose VerRec, a verbalized feedback-guided RL framework for generative recommendation. Specifically, a user simulator generates personalized verbalized feedback based on the user's historical preferences and the candidate item profile. The policy uses this feedback to produce refined recommendations. Original and refined candidates form an expanded group and are jointly optimized with group rela
    
[^10]: Deal With It：论列表式重排序的更优调度方案

    Deal With It: On Better Schedules for Listwise Reranking

    [https://arxiv.org/abs/2611.22392](https://arxiv.org/abs/2611.22392)

    本文证明了在并行列表式重排序调度中，按检索器排名轮转“发牌”划分候选是均期望削减最少的最优分组方式，并提出简单高效的 DealRank 调度方案，在保持接近最优重排序质量的同时实现 1.1 至 1.4 倍的提速。

    

    列表式 LLM 重排序器无法在单次调用中可靠地对数百篇文档进行排序，因此需要依靠调度方案来协调多次调用。并行调度将候选文档划分成若干组，并从每组中按固定配额向下一轮推进。当某个组包含的高质量段落超过其配额时，部分段落会被削减，从而需要进行代价高昂的修复调用。我们证明了候选文档的划分方式决定了配额在何处可能成为约束，并且就期望削减数量而言，最优的划分方式是像发牌一样按检索器排名以轮转方式分发候选。尽管这种“发牌”策略在概念上并不新颖，但此前一直缺乏理论分析。我们证明，在一个简单的检索器模型下，没有任何配额相同的划分方式在平均意义上比发牌削减更少的真实前十条结果，而连续分块的削减量会随检索器性能的提升而增加。DealRank 是我们分析所允许的最简单的调度方案，在两轮设置中对六个 LLM 的平均表现与最优调度方案相差不超过 0.1 个 nDCG@10 点，且速度始终快 1.1 至 1.4 倍。

    arXiv:2611.22392v1 Announce Type: cross  Abstract: Listwise LLM rerankers cannot reliably rank hundreds of documents in a single call, and rely on schedules to orchestrate their calls. Parallel schedules split candidates into groups and advance a fixed quota from each. When a group holds more of the top passages than its quota, some are cut, and expensive repair calls become required. We show that the way candidates are split matters where such a quota can bind, and that, in expected cuts, the best split there deals them round-robin by retriever rank, like cards. Although not conceptually new, dealing has remained unanalyzed. We prove that, under a simple retriever model, no split with equal quotas cuts fewer of the true top ten on average, and that contiguous chunks cut more as the retriever improves. DealRank, the simplest schedule our analysis allows, is on average within 0.1 nDCG@10 points of the best schedules over six LLMs, in two rounds. It is consistently 1.1 to 1.4 times faste
    
[^11]: RICoT：用于页面流分割的循环交错思维链

    RICoT: Recurrent Interleaved Chain-of-Thought for Page Stream Segmentation

    [https://arxiv.org/abs/2611.22382](https://arxiv.org/abs/2611.22382)

    该论文提出RICoT循环交错思维链提示框架，首次将页面流分割重新构想为对整个流的顺序处理过程，像读者一样在滑动窗口内逐页推理并携带已读内容的紧凑记录以控制成本，同时发布了包含205个真实金融页面流的FinancePSS-205基准数据集。

    

    大语言模型（LLM）重新激发了人们对页面流分割（Page Stream Segmentation, PSS）的兴趣，该任务是将多页文件拆分为其组成的文档。然而，已发表的LLM方法仍然将该任务框架化为对一页或少数相邻页面的边界分类，而商业文档拆分API也不公开其方法。我们重新构想了面向LLM的PSS任务，将其视为对整个页面流的顺序处理过程，并提出了RICoT（循环交错思维链，Recurrent Interleaved Chain-of-Thought），这是一个像读者一样逐页处理文档的提示框架：在滑动窗口内，它对每一页进行推理并先做出边界决策再继续往下读；在窗口之间，它携带一个已读内容的紧凑记录，从而使每次调用的成本保持有界。我们还介绍了FinancePSS-205，一个包含205个真实世界、去标识化金融页面流的基准数据集，并将公开发布。在FinancePSS-205上，RICoT匹配或超越了成对（pairwise）和窗口化（windowed）提示方法的表现（摘要在此处被截断）。

    arXiv:2611.22382v1 Announce Type: cross  Abstract: Large language models (LLMs) have renewed interest in Page Stream Segmentation (PSS), splitting a multi-page file into its constituent documents. Yet published LLM methods still frame the task as boundary classification over one or a few neighboring pages, and commercial document-splitting APIs do not disclose their methods. We reimagine PSS for LLMs as a sequential process over the whole stream and propose RICoT (Recurrent Interleaved Chain-of-Thought), a prompting framework that works through the pages as a reader would: within a sliding window it reasons about each page and commits its boundary decision before moving on, and across windows it carries forward a compact record of what it has read, keeping per-call cost bounded. We also introduce FinancePSS-205, a benchmark of 205 real-world, de-identified financial page streams, to be released publicly. On FinancePSS-205, RICoT matches or outperforms pairwise and windowed prompting wi
    
[^12]: NONSENSE：非语义字符串揭示稠密检索中的语义性鸿沟

    NONSENSE: Non-Semantic Strings Expose Semanticity Gaps in Dense Retrieval

    [https://arxiv.org/abs/2611.22353](https://arxiv.org/abs/2611.22353)

    提出NONSENSE黑盒稀疏编辑探针，通过构造禁止复制查询词的明显非语义字符串，证明其稠密相似度得分常超过检索器最佳的查询相关段落，从而揭示了稠密检索中最近邻地位与真实语义含义相分离的语义性鸿沟。

    

    稠密检索是现代搜索、问答和检索增强生成技术的核心，其中高相似度通常被视为语义相关性的证据。我们检验了这一假设能被压榨到何种程度。NONSENSE 是一种黑盒稀疏编辑探针，它构造出表面上明显非语义的字符串，同时禁止复制查询词项，仅优化查询与该字符串之间的稠密相似度。在五个 BEIR 风格的数据集和三种流行的单向量检索器上，这些字符串的得分经常超过检索器自身最佳的已判定查询相关段落（QRP），并且常常在整个语料库中排名第一。这一结果表明，被评估的稠密检索器存在语义性鸿沟：最近邻地位可以与段落式的真实含义相分离。这些发现凸显了评估稠密检索器的重要性——不仅要看它们能否检索到相关文档，还要看它们的相似度分数有多么可靠。

    arXiv:2611.22353v1 Announce Type: cross  Abstract: Dense retrieval is central to modern search, question answering, and retrieval-augmented generation, where high similarity is often treated as evidence of semantic relevance. We examine how far this assumption can be stressed. NONSENSE is a black-box sparse-edit probe that constructs visibly non-semantic strings while forbidding query-term copying, optimizing only the dense similarity between a query and the string. Across five BEIR-style datasets and three popular single-vector retrievers, these strings frequently outscore the retriever's own best judged query-relevant passage (QRP), and often rank first in the full corpus. The result indicates a semanticity gap in the evaluated dense retrievers: nearest-neighbor status can be separated from passage-like meaning. These findings highlight the importance of evaluating dense retrievers not only by whether they retrieve relevant documents, but also by how reliably their similarity scores 
    
[^13]: GuardAgent：一种记忆增强的智能体框架，用于自适应检测基于大语言模型推荐系统中的对抗性推广攻击

    GuardAgent: A Memory-Augmented Agentic Framework for Adaptive Detection of Adversarial Promotions in LLM-Based Recommendation

    [https://arxiv.org/abs/2611.22274](https://arxiv.org/abs/2611.22274)

    该论文提出了首个针对基于大语言模型推荐系统的推广驱动攻击检测的系统性研究，设计了通过感知-行动-反思-记忆循环运作的智能体框架GuardAgent，以实现对攻击的自适应检测。

    

    在传统的推荐系统（RS）中，文本信息通常作为表示学习的辅助信号。相比之下，在基于大语言模型（LLM）的推荐系统中，文本信息通过语义理解和推理成为推荐生成的主要媒介。这一转变从根本上改变了攻击面，使得推广驱动的操纵行为更加可行、有效且更难被检测。然而，如何有效检测此类攻击在很大程度上仍处于未探索状态。为填补这一空白，我们首次针对基于LLM的推荐系统的推广驱动攻击检测开展了系统性研究，旨在保护下一代推荐平台。具体而言，我们提出了GuardAgent，这是一种新颖的由LLM驱动的智能体检测框架，它通过感知、行动、反思和记忆的循环机制运行。与传统的静态、一次性且不透明的LLM检测器不同，GuardAgent...

    arXiv:2611.22274v1 Announce Type: new  Abstract: In traditional recommender systems (RSs), textual information typically serves as auxiliary signals for representation learning. In contrast, in LLM-based RSs, textual information becomes the primary medium for recommendation generation through semantic understanding and reasoning. This shift fundamentally changes the attack surface, making promotion-driven manipulations more feasible, effective, and harder to detect. However, effectively detecting such attacks remains largely unexplored. To fill this gap, we present the first systematic study on detecting promotion-driven attacks against LLM-based RSs, aiming to safeguard next-generation recommendation platforms. Specifically, we propose GuardAgent, a novel LLM-powered agentic detection framework that operates through a loop of Perception, Action, Reflection and Memory. In contrast to conventional LLM-based detectors that function in a static, one-off, and opaque fashion, GuardAgent con
    
[^14]: 教语言模型利用检索反馈收窄布尔式专利检索

    Teaching Language Models to Narrow Boolean Patent Searches from Search Feedback

    [https://arxiv.org/abs/2611.22270](https://arxiv.org/abs/2611.22270)

    该论文利用日本专利局检索报告提炼专家策略，并提出行级“保留引用的收窄信用”学习方法对语言模型进行离线迭代优化，使其生成的布尔专利检索策略在保持引用召回率的同时大幅收窄候选集合，将包含引用文献的中位数集合规模从 768 篇降至 50 篇以下。

    

    专利现有技术检索人员逐行发布布尔查询，先扩大检索范围以找到相关文献，再逐步收窄，直到候选文献数量可供筛选。我们将 38,397 份日本专利局检索报告提炼为可执行的专家策略，重新执行这些策略，并通过引用召回率以及包含每条引用文献的最小文献集合来评估生成的策略。通过提示词驱动的模型能够写出有效且高召回率的策略，但会把引用文献留在规模过大、无法人工筛选的文献集合中。我们从打分的执行轨迹中以两种形式对生成器进行离线适配：一种是从筛选感知的策略分数中进行策略级学习；另一种是从“保留引用的收窄信用”中进行行级学习，即对那些将已找到的引用文献保留在更小集合中的查询行赋予更高权重。在共享的执行轨迹上，行级信用在筛选相关指标上优于策略级分数；经过迭代优化，它将包含引用文献的集合规模中位数从 768 降至 50 以下，并将更多引用文献置于少于 100 篇文献的集合中。

    arXiv:2611.22270v1 Announce Type: cross  Abstract: Patent prior-art searchers issue Boolean queries line by line, broadening to find relevant documents and narrowing until candidates can be screened. We distill 38,397 Japan Patent Office search reports into executable expert strategies, re-execute them, and evaluate generated strategies by citation recall and the smallest set containing each citation. Prompted models write valid, high-recall strategies but leave citations in sets too large to screen. We adapt a generator offline from scored rollouts in two forms: strategy-level learning from a screening-aware strategy score, and line-level learning from citation-preserving narrowing credit, which upweights the query lines keeping a found citation in a smaller set. On shared rollouts, line-level credit outperforms strategy-level scores on screening measures; iterated, it cuts the median citation-containing set from 768 to under 50 and places more citations in sets under 100 documents th
    
[^15]: 基于信任域约束的聚类引导解码个性化推荐方法

    Cluster-Guided Decoding with Trust-Region Constraints for Personalized Recommendation

    [https://arxiv.org/abs/2611.22163](https://arxiv.org/abs/2611.22163)

    CG2R 通过学习语义簇层次结构并将簇级先验直接融入商品打分规则来塑造个性化推荐排序，且先验基于簇质心构建，因此对无历史记录的冷启动用户依然有效。

    

    arXiv:2611.22163v1 公告类型：新论文。摘要：大多数推荐系统对目录中的每个商品单独打分，并未显式建模一条推荐属于目录中的哪个语义区域。而表达能力足够强、能够提供这种控制的模型，往往在全目录上进行在线推理时速度过慢。本文提出了聚类引导生成式重建（CG2R），该方法学习语义簇的层次结构，并使用簇级先验来引导个性化推荐中的商品排序。与现有仅将簇用于候选预筛选的系统不同，CG2R 将簇先验直接加入到商品打分规则中，使得学习到的簇结构能够直接塑造排序结果本身。由于该先验由簇质心而非逐商品统计量构成，即使对于没有任何历史记录的冷启动用户，先验依然有定义。在所评估的三个基准数据集上，CG2R 的表现优于经典方法、神经网络方法以及基于大语言模型的基线方法。

    arXiv:2611.22163v1 Announce Type: new  Abstract: Most recommenders score each item in the catalog separately, and do not explicitly model which semantic region of the catalog a recommendation belongs to. Models expressive enough to offer that control are often too slow to run over a full catalog for online inference. In this paper, we present Cluster-Guided Generative Reconstruction (CG2R), which learns a hierarchy of semantic clusters and uses a cluster-level prior to steer item ranking for personalized recommendations. Unlike existing systems that only use clusters to pre-select candidates, CG2R adds the cluster prior directly to the item scoring rule, so that the learned cluster structure shapes the ranking itself. Because the prior is assembled from cluster centroids rather than per-item statistics, it remains defined even for cold-start users when no history is available. Across the three benchmark datasets evaluated, CG2R outperforms classical, neural, and LLM-based baselines in 
    
[^16]: SIMQ：一种用于多向量查询的单一图索引

    SIMQ: A Single Graph-based Index for Multi-Vector Query

    [https://arxiv.org/abs/2611.22079](https://arxiv.org/abs/2611.22079)

    SIMQ提出了一种基于单一图索引的多向量查询框架，通过数据感知的采样聚类和逐词元图导航，同时突破了现有近似方法在计算矩阵开销和召回稳定性方面的根本瓶颈。

    

    诸如ColBERT等多向量检索模型通过将文档表示为词元级向量的集合并使用MaxSim进行评分，实现了最先进的检索效果，但其高昂的计算成本限制了其在大规模语料库上的可扩展性。现有的近似方法面临根本性瓶颈：质心交互矩阵方法需要在每次查询时计算完整矩阵，并依赖GPU加速的聚类；而数据无关的压缩方法会产生不稳定的召回率，需要反复补偿。我们提出了SIMQ（面向多向量查询的单一图索引），这是一个通过数据感知的采样聚类和逐词元图导航来衔接单向量图索引与多向量检索的框架，同时规避了上述所有瓶颈。SIMQ通过数据感知采样将所有词元向量组织成细粒度的簇，并在簇质心上构建一个独立于...（原文摘要在此处被截断）

    arXiv:2611.22079v1 Announce Type: new  Abstract: Multi-vector retrieval models such as ColBERT achieve state-of-the-art effectiveness by representing documents as sets of token-level vectors and scoring via MaxSim, but their high computational cost limits scalability to large corpora. Existing approximate methods face fundamental bottlenecks: centroid interaction matrix approaches require computing a full matrix at every query and rely on GPU-accelerated clustering, while data-oblivious compression methods yield unstable recall that requires repeated compensation. We propose SIMQ (Single graph Index for Multi-vector Query), a framework that bridges single-vector graph indices and multi-vector retrieval through data-aware sampling-based clustering and per-token graph navigation, simultaneously bypassing all aforementioned bottlenecks. SIMQ organizes all token vectors into fine-grained clusters via data-aware sampling, builds a single graph index over cluster centroids independently of P
    
[^17]: 超越位宽：面向NVIDIA Blackwell多语言大模型推理的部署约束混合精度量化

    Beyond Bitwidth: Deployment-Constrained Mixed-Precision Quantization for Multilingual LLM Inference on NVIDIA Blackwell

    [https://arxiv.org/abs/2611.21974](https://arxiv.org/abs/2611.21974)

    该论文在NVIDIA B200上对Qwen3-8B进行了部署约束驱动的低比特与混合精度量化实测评估，发现INT2/INT3全面退化且模块恢复效应具有强非加性，最终提出一个NVFP4与INT8混合的配置，在通过全部预注册多语言与推理精度门槛的同时将权重内存压缩至5.917 GiB。

    

    arXiv:2611.21974v1 公告类型： cross 摘要：低比特大模型量化只有在模型保持与服务相关的精度、能够通过受支持的GPU内核执行、并满足实测的内存和延迟限制时才有实用价值。我们在NVIDIA B200上使用vLLM研究Qwen3-8B，涵盖INT2/INT3、GPTQ、NVFP4以及混合精度配置。我们在全部144个融合线性模块上验证了原生打包执行与精度，并在预注册的门槛条件下评估韩语、英语、数值和推理质量。打包的INT2和INT3虽然可以执行，但所有七种测试的INT2网格在生成任务上均出现退化。孤立的模块恢复效应表现出强烈的非加性；韩语与推理任务的敏感性排名互不一致，且一个看似有前景的INT4/INT8开发候选方案在独立的最终评估中失败。一个包含16位激活、98个NVFP4仅权重量化模块、46个INT8模块和INT8输出头的配置，在未做任何改动的FINAL-81测试集（5,168个条目）上通过了全部门槛，权重加载内存为5.917 GiB……

    arXiv:2611.21974v1 Announce Type: cross  Abstract: Low-bit LLM quantization is useful only if a model preserves service-relevant accuracy, executes through supported GPU kernels, and meets measured memory and latency limits. We study Qwen3-8B on NVIDIA B200 with vLLM, including INT2/INT3, GPTQ, NVFP4, and mixed-precision configurations. We verify native packed execution and precision on all 144 fused linear modules and evaluate Korean, English, numerical, and reasoning quality under preregistered gates. Packed INT2 and INT3 execute, but all seven tested INT2 grids degenerate on generation. Isolated module-restoration effects are strongly non-additive; Korean and reasoning sensitivity rankings disagree, and a promising INT4/INT8 development candidate fails independent final evaluation. A configuration with 16-bit activations, 98 NVFP4 weight-only modules, 46 INT8 modules, and an INT8 output head passes all gates on the untouched FINAL-81 set (5,168 items): 5.917 GiB of weight-loading me
    
[^18]: 量子联想检索中的复杂度与精度

    Complexity and precision in quantum associative retrieval

    [https://arxiv.org/abs/2611.21941](https://arxiv.org/abs/2611.21941)

    该论文证明了量子联想检索相对于经典暴力搜索具有指数级复杂度优势、相对于经典神经网络具有多项式级容量优势，并能实现高效且高精度的模式检索。

    

    我们证明了高效且高精度的量子联想检索是可行的：相对于经典暴力搜索方法，它具有指数级的复杂度优势；对于任何以量子比特数多项式规模扩展的模式数量，相对于经典神经网络实现，它具有多项式级的容量优势。经典神经联想记忆和量子联想记忆都将模式存储在哈密顿量中；在经典情况下，动力学是向吸引子的不可逆弛豫，而在量子情况下，则是生成有效热分布的幺正变换，随后进行测量。

    arXiv:2611.21941v1 Announce Type: cross  Abstract: We show that efficient and high-precision quantum associative retrieval is possible with an exponential complexity advantage with respect to the classical brute force approach and with a polynomial capacity advantage with respect to classical neural network implementations for any number of patterns scaling polynomially in the number of qubits. Both classical neural and quantum associative memories store the patterns in a Hamiltonian; in the classical case the dynamics is irreversible relaxation toward attractors, in the quantum case it is unitary transformations generating an effective thermal distribution, followed by a measurement.
    
[^19]: 面向识别犯罪报告潜在差异的城市表征学习

    Urban Representation Learning for Identifying Potential Disparities in Crime Reporting

    [https://arxiv.org/abs/2611.21825](https://arxiv.org/abs/2611.21825)

    提出了一种自监督城市表征学习框架，通过邻居画像差异（NPD）指标量化各区域与其表征相似同伴区域的报告画像差异，从而识别开放犯罪数据中的潜在报告偏差。

    

    开放犯罪数据能够支持城市安全分析，但它们反映的是已报告的案件而非犯罪活动的完整分布，这可能给下游分析带来空间扭曲。我们提出了一种自监督的城市表征学习框架，仅利用带地理坐标的犯罪记录来识别相对于表征相似同伴区域的异常报告模式。该方法构建了互补的小时级、周级和静态报告画像，在地理自监督下学习特定尺度的表征，并将其整合到统一的潜在空间中。随后，利用邻居画像差异指标来量化每个区域的报告画像与其检索到的同伴区域画像之间的不一致程度。在墨西哥城超过180万条犯罪记录（聚合为1,058个H3区域）上的实验表明，所提出的表征方法达到了0.4052的平均轮廓系数，相比之下为0.2...

    arXiv:2611.21825v1 Announce Type: new  Abstract: Open crime data support urban safety analysis but reflect reported incidents rather than the complete distribution of criminal activity, potentially introducing spatial distortions in downstream analysis. We propose a self-supervised urban representation learning framework for identifying unusual reporting patterns relative to representation-similar peers using only georeferenced crime records. The method constructs complementary hourly, weekly, and static reporting profiles, learns scale-specific representations under geographic self-supervision, and integrates them into a unified latent space. Neighbor Profile Divergence (NPD) is then used to quantify disagreement between each region's reporting profile and those of its retrieved peers. Experiments on more than 1.8 million crime records from Mexico City, aggregated into 1,058 H3 regions, show that the proposed representation achieves a mean Silhouette score of 0.4052, compared with 0.2
    
[^20]: 评估基于抽象语法树（AST）的知识图谱检索在Apache Kafka AIOps事件诊断中的应用

    Evaluating AST-Derived Knowledge Graph Retrieval for AIOps Incident Diagnosis in Apache Kafka

    [https://arxiv.org/abs/2611.21625](https://arxiv.org/abs/2611.21625)

    本文在Apache Kafka事件诊断任务上系统评估了基于AST派生知识图谱的检索方法，发现仅向量检索已是强基线（Recall@20达0.956）且图增强方法难以显著超越，而分阶段图扩展可将重排序开销减少约52%。

    

    事件诊断需要从大型代码库中检索与源代码相关联的运维证据。我们使用从抽象语法树（AST）派生的所有权图谱，基于1,600条与源代码关联的日志记录以及315个经审计、涵盖103个独特目标的事件式查询基准，评估了面向Apache Kafka的图增强检索。仅向量检索是一个强基线，Recall@20达到0.956，而无论是静态还是置信度引导的分阶段AST扩展均未能显著提升其Recall@20。作为外部对照，我们评估了HippoRAG（一个已发表的基于文本派生图谱的检索系统）与其自身的ColBERTv2检索骨干：相比仅使用ColBERTv2（Recall@20为0.803），HippoRAG仅达到0.308；事后进行的文档集成敏感性分析将Recall@20提升至0.648，但仍未消除差距。相对于静态扩展，分阶段方法将交叉编码器的配对工作负载和实测重排序时间减少了约52%。（注：原文摘要末尾被截断）

    arXiv:2611.21625v1 Announce Type: cross  Abstract: Incident diagnosis requires retrieving source-linked operational evidence from large codebases. We evaluate graph-augmented retrieval for Apache Kafka using an abstract-syntax-tree (AST)-derived ownership graph over 1,600 source-linked log records and an audited benchmark of 315 incident-style queries covering 103 unique targets. Vector-only retrieval is a strong baseline at Recall@20 of 0.956, and neither static nor confidence-guided staged AST expansion significantly improves Recall@20 over it. As an external control we evaluate HippoRAG, a published text-derived graph-retrieval system, against its own ColBERTv2 retrieval backbone. Against ColBERTv2-only (Recall@20 0.803), HippoRAG attained 0.308; a post-hoc document-ensemble sensitivity raised Recall@20 to 0.648 but did not eliminate the deficit. Relative to static expansion, staging reduces cross-encoder pair workload and measured reranking time by about 52%. A manually evaluated 1
    
[^21]: EE-Motion：一种基于脑电图的实时情感化音乐推荐网页架构

    EE-Motion: An EEG-based Web Architecture for Live Affective Music Recommendation

    [https://arxiv.org/abs/2611.21032](https://arxiv.org/abs/2611.21032)

    该论文提出了EE-Motion，一个通过Web Bluetooth连接消费级EEG头带、实时将脑电信号映射为效价-唤醒情绪坐标，并借助情感适配器统一多种推荐后端、无需重新训练即可为匿名用户提供实时情感化音乐推荐的网页架构。

    

    音乐聆听与情绪本质上紧密相连，而情绪在一次聆听会话中会持续漂移。尽管现代浏览器已能够与可穿戴脑机接口交互，但现有推荐系统仍围绕离散的日志记录构建，缺乏整合连续生理信号的基础设施。此外，现有的情感感知模型依赖于静态用户画像和离线训练，无法在不重新训练模型的情况下服务匿名新用户。我们提出了一个同时解决这两个挑战的网页平台。该系统通过 Web Bluetooth 与消费级脑电（EEG）头带配对，实时采集脑电信号，将额叶 alpha 不对称性与唤醒指数映射到 Russell 情绪环形平面上的归一化效价-唤醒代理坐标。在这些坐标的驱动下，一个情感适配器将结构各异的推荐后端统一到共享的实时评分机制之下。通过对匿名访客进行参数化……（原文摘要在此处截断）

    arXiv:2611.21032v1 Announce Type: new  Abstract: Music listening is intrinsically tied to emotion, which drifts continuously within a single session. Although modern browsers interface with wearable brain-computer interfaces, recommendation systems remain structured around discrete logs and lack the infrastructure to integrate continuous physiological signals. Furthermore, existing affect-aware models rely on static user profiles and offline training, failing to serve anonymous newcomers without model retraining. We present a web platform that resolves both challenges. Pairing with a consumer electroencephalogram (EEG) headband over Web Bluetooth, the system captures live EEG, mapping frontal alpha asymmetry and arousal index to normalized valence-arousal proxy coordinates on Russell's circumplex plane. Driven by these coordinates, an affective adapter unifies structurally diverse recommendation backbones under a shared live scoring mechanism. By parameterizing anonymous visitors throu
    
[^22]: RACE：面向高效序列推荐的排序不确定性感知容量自适应定制

    RACE: Ranking-Uncertainty Aware Capacity-Adaptive Customization for Efficient Sequential Recommendation

    [https://arxiv.org/abs/2611.18244](https://arxiv.org/abs/2611.18244)

    提出RACE框架，通过捕获模型预测的排序不确定性并将其转化为主动的资源分配器，以不确定性感知的压缩比率动态调整模型结构和输入序列，从而解决序列推荐中统一压缩比率导致的容量-复杂度失配问题，实现资源受限环境下的高效推荐。

    

    高效序列推荐已成为提供实时个性化服务的一种有前景的范式。为了在资源受限的环境下部署模型，模型压缩成为必要手段，然而现有方法通常采用统一的压缩比率，忽视了用户意图复杂性的异质性。这种“一刀切”的方式导致了根本性的“容量-复杂度失配”问题，即计算资源要么浪费在简单实例上，要么对复杂实例而言不足。为解决这一问题，我们提出了RACE（排序不确定性感知的容量自适应定制框架），该框架利用不确定性感知的压缩比率，动态调整模型结构和输入序列。为实现容量自适应推理，RACE捕获模型预测的内在不确定性，并将其转化为一种主动的资源分配器……

    arXiv:2611.18244v1 Announce Type: cross  Abstract: Efficient sequential recommendation has emerged as a promising paradigm for providing real-time personalized service. To deploy models under resource-constrained environment, model compression becomes a necessity, yet existing methods typically apply uniform compression ratios, overlooking the heterogeneity in user intent complexity. This "one-size-fits-all" approach leads to a fundamental Capacity-Complexity Mismatch, where computational resources are either wasted on simple instances or insufficient for complex ones. To address this, we propose RACE (Ranking-uncertainty Aware Capacity-Adaptive Customization for Efficient Sequential Recommendation), a framework that dynamically adapts both model structures and input sequences with uncertainty-aware compression ratios. To ensure capacity-adaptive inference, RACE captures the model's intrinsic uncertainty of its predictions and transforms it into a proactive resource allocator. Specific
    
[^23]: 可迁移的解释：用稀疏自编码器解读医学信息检索

    Explanations that Transfer: Interpreting Medical Retrieval with Sparse AutoEncoders

    [https://arxiv.org/abs/2611.18179](https://arxiv.org/abs/2611.18179)

    本文提出将稀疏自编码器应用于医学信息检索，通过结合重建目标与InfoNCE对比损失，将不透明的稠密文本嵌入转化为语义可解释的表示，同时保持对新临床语料库的可迁移性。

    

    回答用户查询的医学AI系统依赖于从大型文档集合中检索医学知识。在此背景下，在医学数据集上微调通用模型可以提升检索相关性。然而，现代检索系统所依赖的稠密嵌入在很大程度上仍是不透明的，难以理解其检索结果背后的机制。此外，针对特定医学数据集进行适配可能会失去对新临床语料库的可迁移性。在本工作中，我们通过将稀疏自编码器（SAEs）应用于医学信息检索来解决这些问题，将稠密文本嵌入转化为语义上可解释的表示。为此，我们的模型将重建目标与InfoNCE对比损失相结合，后者能够直接保留查询与文档之间的相关性。我们在六个医学检索数据集上（涵盖域内与域外场景）评估了三种嵌入模型。在医学语料库上训练后，我们的稀疏……

    arXiv:2611.18179v1 Announce Type: cross  Abstract: Medical AI systems that answer users' queries rely on retrieving medical knowledge from large document collections. In this context, finetuning general models on medical datasets can improve retrieval relevance. Yet the dense embeddings underlying modern retrieval systems remain largely opaque, making it difficult to understand the mechanics behind their results. Moreover, adapting to a specific medical dataset risks losing transferability to new clinical corpora. In this work, we tackle these issues by applying Sparse Autoencoders (SAEs) to medical information retrieval, transforming dense text embeddings into semantically interpretable representations. To do so, our model combines a reconstruction objective with an InfoNCE contrastive loss that directly preserves query-document relevance. We evaluate three embedding models across six medical retrieval datasets, both in-domain and out-of-domain. Trained on a medical corpus, our sparse
    
[^24]: Lightfoot：具有精确、无模型文档删除功能的图检索增强生成

    Lightfoot: Graph Retrieval-Augmented Generation with Exact, Model-Free Document Deletion

    [https://arxiv.org/abs/2611.11010](https://arxiv.org/abs/2611.11010)

    Lightfoot 是一种图检索增强生成系统，它将图中的每个实体、链接、事实和同义词边都定义为逐段落提取结果的确定性函数，从而实现无需语言模型调用的检索以及作为数据库事务的精确文档删除（在 11,402 段语料库上删除 10% 仅需 14 秒且结果与全新索引完全一致），解决了现有 GraphRAG 系统的模型调用成本和文档删除不精确问题。

    

    基于图的检索增强生成（GraphRAG）提升了多跳问答能力，但已部署的系统还需承担研究对比中很少被测量的成本：查询时的模型调用、图的构建以及正确地删除文档。我们提出了 Lightfoot，这是一种 GraphRAG 系统，其中每个实体、链接、事实和同义词边都是携带块级溯源信息的逐段落提取结果的确定性函数。检索过程无需调用语言模型，索引过程每个段落仅需一次调用，而删除文档则是一个无需模型调用的数据库事务：在包含 11,402 个段落的语料库上删除 10% 的文档仅需 14 秒，之后得到的图与对剩余文档重新构建的新索引完全一致。在一个包含大量近似重复名称的语料库上，HippoRAG 2 的同义词列表上限导致其结果比重建少 565 条边，而 Lightfoot 始终保持精确。在 MuSiQue 和 2WikiMultiHopQA 的保留测试问题上，使用相同的提取、嵌入和答案模型……

    arXiv:2611.11010v2 Announce Type: replace  Abstract: Graph-based retrieval-augmented generation (GraphRAG) improves multi-hop question answering, but deployed systems pay costs that research comparisons rarely measure: model calls at query time, building the graph, and removing documents correctly. We present Lightfoot, a GraphRAG system in which every entity, link, fact and synonym edge is a deterministic function of per-passage extractions that carry chunk-level provenance. Retrieval makes no language-model call, indexing makes one per passage, and deleting a document is a database transaction with no model calls: 14 seconds for 10% of an 11,402-passage corpus, after which the graph is identical to a fresh index of the surviving documents. On a corpus of near-duplicate names, HippoRAG 2's capped synonym lists leave 565 fewer edges than a rebuild, while Lightfoot remains exact. Under identical extraction, embedding and answer models on held-out questions of MuSiQue, 2WikiMultiHopQA an
    
[^25]: LEGIT：基于标识符字典树的实时资格门控，用于异构页面的受限解码

    LEGIT: Live Eligibility Gating over Identifier Tries for Constrained Decoding of Heterogeneous Pages

    [https://arxiv.org/abs/2611.10357](https://arxiv.org/abs/2611.10357)

    提出 LEGIT 约束机制，通过对标识符字典树叶节点进行深度优先编号、把随请求和 token 动态变化的资格规则转化为高效的范围查找，使生成式推荐系统在整页解码时能精确且快速地保证每个语义 ID 真实、合规且不重复。

    

    生成式推荐器用一段简短的 token 序列（即语义 ID，Semantic ID）来命名每个候选，并通过生成这些 token 来完成推荐。一些生成式推荐器，包括我们正在构建的 Uber Eats 搜索引擎，会在一次解码过程中写出整个结果页面，其中包含商店及其菜单项。这样的页面上，每个语义 ID 都必须指向一个真实的、且符合本次搜索资格的商店或条目（例如，能够配送到用户所在位置的商店），并且尚未出现在该页面上；此外，每个条目还必须属于其上方的商店。在商品目录上构建字典树可以保证标识符的真实性，但资格性与重复性约束会随着每次请求和每个 token 动态变化，而我们尝试过的语法约束解码器无法既精确又快速地执行这些不断变化的规则。我们将这一约束命名为 LEGIT，即“基于标识符字典树的实时资格门控”。对字典树叶节点进行深度优先编号，可以将每条规则转化为对一棵商店字典树和一棵条目字典树的范围查找。

    arXiv:2611.10357v2 Announce Type: replace-cross  Abstract: A generative recommender names each candidate with a short token sequence, a Semantic ID, and recommends by generating those tokens. Some generative recommenders, including the Uber Eats search engine we are building, write a whole results page, stores with their menu items, in one decoding pass. Every Semantic ID on such a page must name a real store or item that is eligible for the search, such as a store that delivers to the user's location, and that is not already on the page. Each item must also belong to the store above it. A trie over the catalog keeps identifiers real, but eligibility and repetition change with every request and token, and the grammar-constrained decoder we tried could not enforce these changing rules exactly and quickly. We call this constraint LEGIT, for live eligibility gating over identifier tries. Numbering trie leaves depth first turns each rule into a range lookup over one store trie and one item
    
[^26]: 基于检索增强生成的股票市场信息检索优化

    Enhancing Stock Market Information Retrieval with Retrieval-Augmented Generation

    [https://arxiv.org/abs/2611.03747](https://arxiv.org/abs/2611.03747)

    本文提出一种整合检索与生成方法的RAG系统，用于生成准确且相关的股票市场信息响应，并通过ROUGE指标验证了其有效性。

    

    自然语言处理（NLP）已广泛应用于金融领域，实现了股票市场信息的自动化分析与检索。然而，传统的基于NLP的系统在动态金融环境中往往难以保持上下文准确性并生成相关响应。为应对这些挑战，检索增强生成（RAG）模型作为一种强大的解决方案应运而生，它将基于检索的方法与生成能力相结合，以提升响应质量。在本研究中，我们提出了一种旨在生成与股票市场信息相关响应的检索增强生成（RAG）系统。该系统整合了基于检索的方法和生成式方法，以提高金融文本生成的准确性和相关性。为评估其有效性，我们采用ROUGE指标（具体包括ROUGE-1、ROUGE-2和ROUGE-L）来衡量词汇和结构层面的相似性……

    arXiv:2611.03747v2 Announce Type: replace-cross  Abstract: Natural Language Processing (NLP) has been widely applied in financial domains, enabling automated analysis and retrieval of stock market information. However, traditional NLP-based systems often struggle with maintaining contextual accuracy and generating relevant responses in dynamic financial environments. To address these challenges, Retrieval-Augmented Generation (RAG) models have emerged as a powerful solution, combining retrieval-based methods with generative capabilities to enhance response quality. In this study, we present a Retrieval-Augmented Generation (RAG) system designed to generate responses related to stock market information. The system integrates retrieval-based and generative approaches to enhance the accuracy and relevance of financial text generation. To evaluate its effectiveness, we employ the ROUGE metric, specifically ROUGE-1, ROUGE-2, and ROUGE-L, to measure lexical and structural similarity between 
    
[^27]: REGER：基于语义ID建模物品间关系的生成式推荐

    REGER: Modeling inter-item Relations with Semantic IDs for Generative Recommendation

    [https://arxiv.org/abs/2610.37584](https://arxiv.org/abs/2610.37584)

    提出REGER统一框架，通过生成式语义检索预训练使GR模型继承物品间语义相似性，并将物品间关系显式融入语义预训练、历史编码和SID生成的全过程。

    

    个性化推荐依赖于从在线内容和用户交互中学习物品之间的语义关系和协同关系。生成式推荐（GR）通过自回归地生成物品的离散语义ID（SIDs），提供了一种有前景的范式。然而，随机初始化的SID词元嵌入以及从零开始训练的GR模型并不能直接继承物品固有的语义关系，而物品间的协同关系主要是通过词元级注意力机制和下一词元预测监督被隐式学习的。为了解决这一局限，我们提出了REGER，这是一个统一框架，将物品间关系显式地融入语义预训练、历史编码和SID生成之中。我们通过生成式语义检索对GR模型进行预训练，使其通过学习生成语义邻居的SID来继承物品间的语义相似性。我们将全局注意力与局部注意力相结合……（摘要原文在此处不完整）

    arXiv:2610.37584v1 Announce Type: cross  Abstract: Personalized recommendation depends on learning semantic and collaborative relations among items from online content and user interactions. Generative recommendation (GR) offers a promising paradigm by autoregressively generating discrete semantic IDs (SIDs) of items. However, randomly initialized SID token embeddings and GR models trained from scratch do not directly inherit the semantic relations inherent in items, while inter-item collaborative relations are learned largely implicitly through token-level attention and next-token supervision. To address this limitation, we propose REGER, a unified framework that explicitly incorporates inter-item relations into semantic pretraining, history encoding, and SID generation. We pretrain the GR model through generative semantic retrieval, enabling it to inherit inter-item semantic similarity by learning to generate semantic neighbors' SIDs. We combine global attention with local attention 
    
[^28]: 基于语义增益图的多跳RAG互补证据检索

    Retrieving Complementary Evidence for Multi-Hop RAG with Semantic Gain Graphs

    [https://arxiv.org/abs/2610.15285](https://arxiv.org/abs/2610.15285)

    提出SG$^2$RAG框架，利用大语言模型离线评估文本块之间的非冗余语义增益并构建可复用的语义增益图，再通过个性化PageRank在线检索多跳RAG所需的互补证据，在多个基准数据集上取得最优性能。

    

    多跳检索增强生成（RAG）需要找到那些证据组合起来有用的文档。查询相关性和图连通性有助于定位相关内容，但两者都无法直接衡量在给定另一个文档的情况下，一个文档所贡献的额外信息。我们提出了SG$^2$RAG，一个通过语义增益来刻画这种互补性的框架。在离线阶段，大语言模型在语义相似性所提出的邻域内，评估一个文本块在给定另一个文本块时所贡献的相关且非冗余的信息。这些有序的评估结果被聚合为对称的边权重，从而构建出一个可复用的语义增益图。在在线阶段，受控的个性化PageRank从与查询相关的种子节点出发，对超出初始检索集之外的互补证据进行排序。在MuSiQue、2Wiki和HotpotQA数据集上使用两种语言模型骨干网络的实验表明，SG$^2$RAG在平均EM、F1和召回率上取得了最佳表现。

    arXiv:2610.15285v2 Announce Type: replace  Abstract: Multi-hop retrieval-augmented generation (RAG) requires finding documents whose evidence is useful in combination. Query relevance and graph connectivity help locate related content, but neither directly measures the additional information one document contributes given another. We introduce SG$^2$RAG, a framework that represents this complementarity through semantic gain. Offline, a large language model assesses the relevant and nonredundant information contributed by one chunk given another within neighborhoods proposed by semantic similarity. These ordered assessments are aggregated into symmetric edge weights to construct a reusable Semantic Gain Graph. Online, controlled Personalized PageRank starts from query-relevant seeds and ranks complementary evidence beyond the initial retrieval set. Experiments on MuSiQue, 2Wiki, and HotpotQA with two language model backbones show that SG$^2$RAG achieves the best average EM, F1, and Reca
    
[^29]: CodeHID：学习可寻址的分层代码索引用于生成式代码检索

    CodeHID: Learning an Addressable Hierarchical Code Index for Generative Code Retrieval

    [https://arxiv.org/abs/2608.24089](https://arxiv.org/abs/2608.24089)

    CodeHID通过构建分层的可寻址代码索引，将代码检索从平面匹配转变为从粗到细的语义地址生成，从而提升对相似代码候选的区分能力。

    

    代码检索模型主要依赖一种平面匹配范式，将代码片段视为独立候选，这使得它们难以区分相似的代码候选。生成式检索通过构建代码库上的可学习索引提供了一种解决方案，引导检索器更好地理解代码候选的语义组织和寻址方式。然而，在代码检索任务中直接应用生成式检索可能面临标识符空间的操作问题，其前缀不对应于有意义的代码语义区域。在本文中，我们提出了CodeHID，一种生成式代码检索框架，将代码检索任务从平面候选匹配重构为从粗到细的语义地址生成。CodeHID依赖于两个核心组件。首先，伪邻居引导的DocID学习通过应用多级残差量化构建全局静态的分层索引。

    arXiv:2608.24089v1 Announce Type: new  Abstract: Code retrieval models have predominantly relied on a flat matching paradigm that treats code snippets as independent candidates, making them less capable of distinguishing similar code candidates. Generative retrieval offers a solution by constructing a learnable index over the code corpus, guiding the retriever to better understand how code candidates are semantically organized and addressed. However, naively applying generative retrieval in the code retrieval task may result in operating over an identifier space whose prefixes do not correspond to meaningful code-semantic regions. In this paper, we propose CodeHID, a generative code retrieval framework that reformulates the code retrieval task from flat candidate matching to coarse-to-fine semantic address generation. CodeHID relies on two core components. First, Pseudo-Neighbor Guided DocID Learning constructs a globally static hierarchical index by applying multi-level residual quant
    
[^30]: RecHarness：一种用于自演化推荐系统的多臂老虎机路由智能体框架

    RecHarness: A Bandit-Routed Agentic Harness for Self-Evolving Recommender Systems

    [https://arxiv.org/abs/2607.29241](https://arxiv.org/abs/2607.29241)

    提出RecHarness框架，通过将优化过程解耦为“多臂老虎机路由器选择修改方向+LLM生成具体优化假设与代码修改”两个步骤，并借助跳跃盆地机制在局部优化停滞时触发结构性跳跃，实现推荐模型的自演化自动优化。

    

    优化推荐模型仍然严重依赖于对架构、目标和训练策略的人工迭代。虽然基于大语言模型（LLM）的智能体可以自动化这一试错过程，但依赖LLM同时选择修改方向和制定具体假设，在有限的试验预算下可能导致不稳定的搜索。受上述挑战的启发，我们提出了RecHarness，一个用于自动化推荐模型优化的多臂老虎机路由智能体框架。RecHarness将优化过程解耦为两个步骤：多臂老虎机路由器根据历史验证反馈选择下一个修改方向，而LLM则在所选方向内生成具体的优化假设和可执行的代码修改。为了维持长期探索，RecHarness使用跳跃盆地（jump-basin）机制，在局部编辑陷入停滞时激活结构跳跃臂。在多个推荐任务、数据集和……（摘要截断）

    arXiv:2607.29241v2 Announce Type: replace-cross  Abstract: Optimizing recommender models remains heavily reliant on manual iteration over architectures, objectives, and training strategies. While LLM-based agents can automate this trial-and-error process, relying on LLMs to both select modification directions and formulate concrete hypotheses can lead to unstable search under limited trial budgets. Inspired by the above challenge, we propose RecHarness, a Bandit-Routed Agentic Harness for automated recommender model optimization. RecHarness decouples the optimization process into two steps: a bandit router selects the next modification direction according to historical validation feedback, while the LLM generates a concrete optimization hypothesis and executable code edit within the selected direction. To sustain long-horizon exploration, RecHarness uses a jump-basin mechanism to activate a structural-jump arm when local edits stagnate. Across multiple recommendation tasks, datasets, a
    
[^31]: 无需坐标或区域标签的视觉文档理解证据归因

    Evidence Attribution in Visual Document Understanding without Coordinates or Region Labels

    [https://arxiv.org/abs/2607.24651](https://arxiv.org/abs/2607.24651)

    该研究提出以“模型逐字引用证据、由多模态检索器定位出处”的语言接口替代传统的坐标边界框接口，表明视觉文档理解中的归因幻觉部分源于坐标表达能力的局限。

    

    可靠的视觉文档理解要求模型将每个答案归因于支持它的证据区域。近期的基准测试和系统通过坐标接口来表达这一步骤：模型输出标记文档中证据区域的边界框坐标。在此接口下，视觉语言模型即使答案正确，也常常无法识别正确的区域，这种失败被称为“归因幻觉”。我们开展了一项研究，探讨这种失败是否部分受限于模型通过坐标所能表达的内容。在一个经过验证的双语 CiteVQA 子集上，我们将坐标接口与语言接口进行对比：在语言接口中，模型仅输出文本，逐字引用其证据，随后由多模态检索器返回每条引用的位置，即布局解析器所提出的页面区域（表格和图片则通过其标题或注释进行引用）……（摘要在此处截断）

    arXiv:2607.24651v2 Announce Type: replace-cross  Abstract: Reliable visual document understanding requires a model to attribute each answer to the evidence regions that support it. Recent benchmarks and systems express this step through a coordinate interface: the model outputs the coordinates of bounding boxes that mark the evidence regions in the document. Under this interface, vision-language models often fail to identify the right regions even when the answer is correct, a failure known as Attribution Hallucination. We present a study that investigates whether this failure is partially limited by what the model can express through coordinates. On a verified bilingual CiteVQA subset, we compare the coordinate interface with a language interface in which the model outputs only text, quoting its evidence verbatim, and a multimodal retriever returns the location of each quote as a page region proposed by a layout parser (tables and figures are quoted through their captions or notes); t
    
[^32]: 超越检索：学习紧凑的用户表示以实现可扩展的大语言模型个性化

    Beyond Retrieval: Learning Compact User Representations for Scalable LLM Personalization

    [https://arxiv.org/abs/2606.04547](https://arxiv.org/abs/2606.04547)

    提出TAP-PER框架，通过时序注意力前缀嵌入将用户偏好编码为紧凑的可学习表示，摆脱了检索式个性化对检索质量的依赖以及参数式个性化随用户规模增长的高昂存储成本，实现可扩展的大语言模型个性化。

    

    个性化大语言模型需要在使模型行为适应个体用户的同时，保持鲁棒性和部署规模上的效率。现有方法通常在输入层面对大语言模型进行个性化，即通过检索用户历史或构建用户画像提示，或者在参数层面对其进行个性化，即为每个用户维护特定的参数高效模块。前者使个性化效果依赖于检索质量和提示设计，而后者会产生随用户规模增长而增加的存储和维护成本。为了解决这些局限，我们提出了TAP-PER（Temporal Attentive Prefix for PERsonalization，时序注意力前缀个性化），这是一个基于前缀的框架，它将用户偏好编码为可学习的表示，避免了将用户历史序列化到提示中，并用轻量级的用户状态前缀嵌入取代了繁重的每用户适配器模块。受个性化推荐系统的启发，TAP-PER将用户……（原文摘要在此处截断）

    arXiv:2606.04547v3 Announce Type: replace-cross  Abstract: Personalizing large language models requires adapting model behavior to individual users while preserving robustness and deployment-scale efficiency. Existing approaches typically personalize LLMs either at the input level, by retrieving user histories or constructing profile prompts, or at the parameter level, by maintaining user-specific parameter-efficient modules. The former makes personalization sensitive to retrieval quality and prompt design, whereas the latter incurs storage and maintenance costs that grow with the user population. To address these limitations, we propose TAP-PER (Temporal Attentive Prefix for PERsonalization), a prefix-based framework that encodes user preferences as learnable representations, avoiding the serialization of user histories into prompts and replacing heavy per-user adapters with lightweight user-state prefix embeddings. Inspired by personalized recommendation systems, TAP-PER decomposes u
    
[^33]: 基于语义聚类ID与专家引导强化学习的电商搜索高效生成式检索

    Efficient Generative Retrieval for E-commerce Search with Semantic Cluster IDs and Expert-Guided RL

    [https://arxiv.org/abs/2605.14434](https://arxiv.org/abs/2605.14434)

    该论文提出CQ-SID方法，通过类目感知对比学习与残差量化VAE将商品编码为层次化语义聚类ID以大幅降低束搜索复杂度，并引入专家引导的强化学习方法EG-GRPO，将生成式检索定位为工业级电商搜索召回阶段的高效补充方案。

    

    生成式检索通过将碎片化的多阶段检索过程统一为单一的端到端模型，提供了一种极具前景的替代方案。然而，面对庞大且动态变化的商品目录、严格的延迟要求，以及检索与下游排序目标对齐的需求，生成式检索在工业级电商搜索中的实际落地仍面临诸多挑战。在本工作中，我们提出了一个面向真实召回场景的检索框架，将生成式检索定位为召回阶段的补充手段，而非端到端的替代方案。我们的方法CQ-SID（类目与查询约束的语义ID）采用类目感知与查询-商品对比学习，并结合残差量化VAE，将商品编码为层次化的语义聚类标识符，从而显著降低了束搜索的复杂度。此外，我们还开发了EG-GRPO（专家引导的群体相对策略优化），这是一种强化学习方法……（原文摘要至此截断）

    arXiv:2605.14434v2 Announce Type: replace-cross  Abstract: Generative retrieval offers a promising alternative by unifying the fragmented multi-stage retrieval process into a single end-to-end model. However, its practical adoption in industrial e-commerce search remains challenging, given the massive and dynamic product catalogs, strict latency requirements, and the need to align retrieval with downstream ranking goals. In this work, we propose a retrieval framework tailored for real-world recall scenarios, positioning generative retrieval as a recall-stage supplement rather than an end-to-end replacement. Our method, CQ-SID (Category-and-Query constrained Semantic ID), employs category-aware and query-item contrastive learning along with Residual Quantized VAEs to encode items into hierarchical semantic cluster identifiers, significantly reducing beam search complexity. Additionally, we develop EG-GRPO (Expert-Guided Group Relative Policy Optimization), a reinforcement learning appro
    
[^34]: NewsReX：一个开源的神经新闻推荐多框架

    NewsReX: An Open-Source Multi-Framework for Neural News Recommendation

    [https://arxiv.org/abs/2508.21572](https://arxiv.org/abs/2508.21572)

    NewsReX 是一个开源的多框架神经新闻推荐系统，通过 YAML 模型规范和 Hydra 配置系统统一了多种推荐模型，支持 PyTorch 与 JAX/Flax 两个后端间的单命令切换与数值一致性验证，解决了该领域结果复现与比较困难的问题。

    

    由于代码库碎片化、预处理流程不一致，以及模型与深度学习框架之间的紧密耦合，神经新闻推荐研究中的结果复现与比较仍然十分困难。我们提出了 NewsReX，这是一个开源框架，它在两个机器学习后端（PyTorch 和 JAX/Flax）上统一了多种神经新闻推荐模型。模型通过结构化的 YAML 规范来定义，该规范涵盖模型架构、训练方案和数据契约，从而将模型定义与特定框架的实现解耦。基于 Hydra 的配置系统支持可组合的实验，只需一条命令即可完成框架切换。我们提供了跨框架一致性测试来验证各后端之间的数值一致性，使研究人员能够复现原始实现的结果，然后将实验迁移到他们偏好的框架上。我们在 MIND-small 数据集上报告了全面的实验。

    arXiv:2508.21572v2 Announce Type: replace  Abstract: Reproducing and comparing results in neural news recommendation research remains difficult due to fragmented codebases, inconsistent preprocessing pipelines, and tight coupling between models and deep learning frameworks. We present NewsReX, an open-source framework that unifies multiple neural news recommendation models across two ML backends (PyTorch and JAX/Flax). Models are defined through a structured YAML specification that captures architecture, training recipe, and data contract, decoupling model definition from framework-specific implementation. A Hydra-based configuration system enables composable experiments with single-command framework switching. We provide cross-framework parity tests to verify numerical consistency across backends, allowing researchers to reproduce results from the original implementations and then transfer experiments to their preferred framework. We report comprehensive experiments on the MIND-small 
    

