# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [REGER: Modeling inter-item Relations with Semantic IDs for Generative Recommendation](https://arxiv.org/abs/2610.37584) | 提出REGER统一框架，通过生成式语义检索预训练使GR模型继承物品间语义相似性，并将物品间关系显式融入语义预训练、历史编码和SID生成的全过程。 |
| [^2] | [LCM-Promo: An LLM-Based Uplift Model for Promotion Allocation](https://arxiv.org/abs/2610.37570) | LCM-Promo是一种在交易商务数据上微调的大语言模型促销分配增益模型，通过将单token输出解释为转化阈值概率来学习单调的促销-响应曲线，在生产A/B测试中将转化率提升1%并降低促销成本6%。 |
| [^3] | [Can Complementary Signals Bridge Similarity Islands? Manifold-Augmented Graph Embedding for Multimodal Recommendation](https://arxiv.org/abs/2610.37409) | 提出MAGE框架，将互补性建模解耦为离线几何扭曲和在线拓扑传播，利用Soft Conceptor机制提取互补感知的残差方向并注入物品表示，从而桥接多模态推荐中语义孤立物品形成的“相似性孤岛”。 |
| [^4] | [When Is Additional Retrieval Worthwhile? A Cost-Aware Study of Evidence Gain and Answer Reliability in RAG](https://arxiv.org/abs/2610.37358) | 该论文提出一种成本感知的检索路由策略，通过在路由前预估混合检索相对于稠密检索基线的问题级增量证据收益，仅在约四分之一的问题上触发更昂贵的混合检索，在保持相同证据Hit@10的同时显著降低检索成本。 |
| [^5] | [LESSRec: Level-Specific History Modeling for Semantic ID-Based Recommendation](https://arxiv.org/abs/2610.37286) | LESSRec通过按语义层级分别编码历史标记序列，并将预测时的注意力限制在目标标记所在层级，同时辅以按时间顺序排列的物品级语义序列，从而提升基于语义ID的推荐效果。 |
| [^6] | [Modeling Preference Dependencies with Directed Acyclic Graphs for Narrative-Driven Recommendation](https://arxiv.org/abs/2610.37280) | 提出偏好依赖图框架（PDG-F），利用有向无环图显式建模用户叙事中偏好之间的层级依赖关系，并通过有向路径检索和拓扑路径排序实现叙事驱动的推荐。 |
| [^7] | [GradCO: Behavior-Aware Gradient-Conflict Offloading for Text-Based Person Anomaly Search](https://arxiv.org/abs/2610.37267) | 该论文提出 GradCO 框架，通过行为感知混合专家（B-MoE）与冲突卸载学习（COL）等组件，解决文本行人异常搜索中同身份不同行为困难负样本导致的梯度冲突以及外观线索掩盖微弱行为证据的问题。 |
| [^8] | [TaskBridge: Technical Report for SIGIR 2026 eCommerce Challenge](https://arxiv.org/abs/2610.37231) | 提出TaskBridge框架，通过目录语义锚定、五候选统一改写和协议感知选择，将任务导向查询改写为适合商品检索的表达式，有效弥合查询与商品目录之间的词汇鸿沟。 |
| [^9] | [AutoRSE: Evaluating LLM Agents as Recommendation Engineers for Self-Evolving Recommender Systems](https://arxiv.org/abs/2610.37228) | 该论文提出了AutoRSE评估框架，通过覆盖六项工业级推荐任务的大规模实证研究发现，基于LLM的推荐工程智能体能将推荐性能平均提升16.1%，但目前主要充当工程优化者而非算法创新者。 |
| [^10] | [Lossless Compression of Proximity Graph Indexes for Efficient Approximate Nearest Neighbors Search](https://arxiv.org/abs/2610.37176) | 该论文提出一种专为HNSW实时路由设计的无损压缩技术，通过离线图置换诱导结构局部性，使SIMD加速的整数压缩器（如Stream VByte）能够在不牺牲查询延迟的前提下快速解压邻接列表，从而大幅降低图索引的内存占用。 |
| [^11] | [Coverage Under Contradiction: A Benchmark for Negation-Aware Retrieval](https://arxiv.org/abs/2610.37162) | 该论文提出了 CoverageBench-Negate 基准，通过为每个信息点构建否定版本并相应编辑支持文档，首次系统性地研究了检索器在语料库证据与其参数化知识相矛盾时的行为表现。 |
| [^12] | [Rethinking Positional Encoding and Interaction Granularity for CTR Prediction](https://arxiv.org/abs/2610.37140) | 提出TRIP模型，将目标感知的秩-时间位置编码与时间组交互建模相结合，揭示了大多数现有位置编码并不优于无位置编码，从而重新思考了CTR预测真正所需的位置编码与交互粒度。 |
| [^13] | [Tessera: Structure-Completed Evidence Units for Web Retrieval-Augmented Generation](https://arxiv.org/abs/2610.37080) | 提出无需训练的证据传递层Tessera，通过将网页HTML重建为结构补全的细粒度证据单元并解耦检索与阅读，解决了表格、信息框等结构化内容在RAG流水线中因分块而丢失证据的问题。 |
| [^14] | [Compote: Collapse-Resistant Multiple Hypothesis Learning For Sequential Recommendation](https://arxiv.org/abs/2610.37005) | COMPOTE提出了一种用于序列推荐的抗坍缩多假设学习方法，通过退火目标促使多个假设分工覆盖不同商品区域以表达用户多重意图，并从理论上证明其最优解满足最大熵量化原理且保持贝叶斯最优排序。 |
| [^15] | [SpectRAG: Weighted Chunk Graphs for Token-Budgeted Context Selection in Multi-Hop Retrieval-Augmented Generation](https://arxiv.org/abs/2610.36991) | SpectRAG提出了一种谱图增强检索框架，通过构建编码多种语义与结构关系的加权分块图，结合谱相关性传播与Token预算约束下的上下文选择，解决传统独立检索在多跳推理中证据冗余和不完整的问题。 |
| [^16] | [KL-MOPD: Discrepancy-Aware Multi-Objective On-Policy Distillation for Generative Recommendation](https://arxiv.org/abs/2610.36973) | 该论文提出KL-MOPD方法，通过差异感知的多目标在线策略蒸馏来缓解token级蒸馏难度不平衡与跨token教师不一致的问题，从而在生成式推荐中更好地协调点击、转化等多个推荐目标。 |
| [^17] | [PailitaoMMSearch2.0 Technical Report: Scaling Generative Retrieval to Unified End-to-End Multimodal Search](https://arxiv.org/abs/2610.36958) | 提出了拍立淘MMSearch，一个统一的端到端多模态生成式检索系统，可直接从原始查询图像和自由形式指令生成商品语义ID，并在数据、意图等三个维度上扩展，以支持开放式图文查询、异构购物意图和不同资源约束的在线场景。 |
| [^18] | [When Diffusion Recommenders Stop Sampling: Preserving Generative Diversity](https://arxiv.org/abs/2610.36954) | 该论文发现目标物品扩散推荐器存在严重采样坍塌问题（源于隐空间与去噪器联合学习以及轨迹内离散损失监督两种设计缺陷），并提出三阶段潜在扩散框架DARTS，通过解耦表示学习、生成建模与离散排序，在保持推荐准确性的同时保留了生成多样性。 |
| [^19] | [Do We Need an LLM to Know Why Sources Conflict? Decision Models for Knowledge-Conflict Typing in RAG](https://arxiv.org/abs/2610.36915) | 小型类型化决策模型在识别RAG中检索来源的冲突类型方面，可以以低4至39倍的成本媲美甚至超越Gemini 2.5 Flash等前沿大语言模型。 |
| [^20] | [Latent Residual Reasoning for Ranking in Generative Recommendation](https://arxiv.org/abs/2610.36890) | 该论文提出在生成式推荐排序中引入潜在残差推理，使基于ID的非文本推荐模型能够在推理时通过多步潜在计算提升对多候选物品的细粒度排序质量。 |
| [^21] | [DuTri: Dual-Branch Modeling with Triadic Factorization for Multimodal Multi-Scenario Recommendation](https://arxiv.org/abs/2610.36836) | 提出DuTri框架，通过协同分支与多模态分支的双分支建模结合三元分解，分别满足协同信息与多模态信息的不同建模需求，从而实现更优的多模态多场景推荐效果。 |
| [^22] | [The Vocabulary Is Already Here: Adapting Pretrained Interpretable Features for Generative Recommendation](https://arxiv.org/abs/2610.36763) | 该论文提出API框架，直接复用预训练稀疏自编码器的可解释特征索引作为生成式推荐的语义ID词表，无需额外的词表学习即可实现有效且可解释的推荐。 |
| [^23] | [Recall or Relearn? A Mixture of In-Context Experts for Recurring Concept Drift](https://arxiv.org/abs/2610.36585) | 提出无需训练的MICE方法，通过在检测到概念漂移时切分数据流并维护多个存储上下文，让表格基础模型在历史概念重现时能够自适应地在“回忆”旧上下文与“重学”新标注之间做出决策，从而更好地应对复发性概念漂移。 |
| [^24] | [Governed Sovereign RAG for Public Administration: Tripartite Epistemic Routing, Facet-Guided Clarification, and Deterministic Numeric Verification in Citizen Services](https://arxiv.org/abs/2610.36561) | 该论文提出并评估了一种部署于墨西哥特拉斯卡拉州官方门户的受管控主权RAG架构，通过三方认知路由门控、分面引导澄清和带SHA-256缓存失效的确定性数值验证三大支柱，解决政府公民服务中监管确定性、数据主权与查询模糊性之间的矛盾。 |
| [^25] | [CCP-DP: Cross-cluster context-guided phase dynamic programming for DNA storage sequence reconstruction](https://arxiv.org/abs/2610.36559) | 该论文提出CCP-DP方法，利用数据集层面的跨簇上下文信息补充簇内局部证据，通过相位动态规划在低覆盖率和强IDS噪声条件下显著提升DNA存储序列的精确重建率（95.36%–99.98%），性能优于现有方法。 |
| [^26] | [QESChunker: A Single Objective Unifies Overlapping and Non-Overlapping Chunking for RAG](https://arxiv.org/abs/2610.36557) | 提出QESChunker，以统一的问题-证据效用最大化目标将RAG中的非重叠与重叠分块统一为有监督组合优化问题，非重叠情形可通过动态规划在多项式时间内精确求解。 |
| [^27] | [Monitor Accuracy Isn't the Answer: A Decision-Value Framework for Evaluating AI Agent Restart Monitors](https://arxiv.org/abs/2610.36544) | 该论文提出一个决策价值框架，从分析上证明监控器的分类准确率本身无法决定重启AI智能体是否有益，并引入决策充分性准则和有限样本无模型决策审计两种技术，以决策价值的角度评估AI智能体重启监控器。 |
| [^28] | [ACOR: incremental evidence accumulation towards real-time DNA storage reconstruction](https://arxiv.org/abs/2610.36500) | ACOR是一种面向DNA存储的流式共识重建方法，通过在持久账本中增量记录k-mer转换与边界支持证据，并利用长度约束束搜索进行解码，实现了与测序并行的实时序列重建，恢复效果接近完全且具有竞争力。 |
| [^29] | [Cross-Day Compatibility of Daily Item Embeddings: An Empirical Study of Alignment, Scale Management, and CTR Adaptation](https://arxiv.org/abs/2610.36408) | 该实证研究系统比较了48种配置下每日物品嵌入的跨日兼容性管理方法，发现ID初始化与正交对齐等上游干预能显著提升嵌入对齐质量（余弦相似度最高提升93.4%）并改善多日CTR模型性能（GAUC最高提升2.6%）。 |
| [^30] | [MAREval: A Multi-Agent Framework for Evaluating Natural Language Recommendation Explanations](https://arxiv.org/abs/2610.36389) | MAREval提出了一种结构化多智能体框架，结合辩论链提示策略、主持人调节、仲裁者聚合与蒙特卡洛采样，实现了对推荐系统中自然语言解释的可扩展、一致且自动化的评估。 |
| [^31] | [From Generation to Scoring: Scaling LLM-Based Semantic Labeling at LinkedIn](https://arxiv.org/abs/2610.36383) | 该论文提出将基于大语言模型的语义标注从生成式答案列表重构为语义评分方法，通过单候选提示微调和多项评分技术，在LinkedIn的实际部署中实现了更高效、更稳健且对候选顺序不敏感的标注方案。 |
| [^32] | [Preserving rare-item recall under a storage budget: quantization, distillation, and the limits of teacher fidelity](https://arxiv.org/abs/2610.36267) | 在每行168字节的固定存储预算下，对300维教师模型评分进行秩300的4位低精度分解仅需教师存储的13%，即可在全部七个数据集-骨干组合中保留教师95-103%的稀有物品召回率，优于小型全精度模型和蒸馏学生模型，且教师保真度并不可靠地反映稀有物品召回的保留情况。 |
| [^33] | [BehaveRec: When Does Long-Term Behavioral History Help Sequential Recommendation? A Study with Session, History, and Community Signals](https://arxiv.org/abs/2610.36239) | 提出三信号框架BehaveRec，通过可学习门控融合短期会话、90天长期用户历史和基于行为嵌入相似度的社区信号，在1.37亿条序列的工业数据集上超越了更大容量的SASRec基线，并系统研究了长期历史信号何时真正对推荐有效。 |
| [^34] | [Counterfactual Profile Perturbations for Recommendation Explanations: A Reproducibility Study](https://arxiv.org/abs/2610.36208) | 本研究复现并扩展了LXR反事实推荐解释方法，证明其不仅对最高推荐项目有效，对其他项目同样有效，同时发现项目流行度等简单基线在某些任务中表现得出人意料地好。 |
| [^35] | [Is the education gap in generative AI use smaller than the internet's was at a similar level of use? Evidence from 16 European countries](https://arxiv.org/abs/2610.36181) | 该研究通过将16个欧洲国家2025年的生成式AI使用率与互联网使用率相当的年份匹配对比，发现生成式AI使用中的教育差距在大多数国家（13/16）小于互联网普及早期在相似使用水平时的差距，表明生成式AI的普及在教育维度上比互联网初期更为平等。 |
| [^36] | [When Connectivity Misleads: Guiding Agentic Search with Contrastive Evidence Paths](https://arxiv.org/abs/2610.36134) | 提出对比证据路径（CEP）方法，通过在查询条件化的文档图上依据最强单一路径评分，并利用背景访问分布进行校准来消除高连通性文档带来的结构偏差，从而引导搜索智能体发现真正有价值的证据而非高度连接的干扰节点。 |
| [^37] | [RELAY: One Query, Two Roles for Sequential Recommendation](https://arxiv.org/abs/2610.36000) | RELAY让单个查询同时承担语义偏好表示与行为记忆路由两种角色，并通过候选条件化门控融合两类证据，在单次前向传播中以低成本实现六个数据集上全面领先的推荐精度。 |
| [^38] | [Hypergraph-Enhanced Evidence Selection for Long-Context Multimodal Document QA](https://arxiv.org/abs/2610.35542) | 提出HyperDoc框架，通过构建文档超图显式捕获结构、逻辑与跨模态关系，在有限输入预算下实现预算感知的多模态证据选择，从而提升长上下文多模态文档问答的性能。 |
| [^39] | [Unified Heterogeneous Feature Interaction via Effective-Rank Scaling for Recommender Systems](https://arxiv.org/abs/2610.35318) | 提出了HiRank框架，通过有效秩缩放在统一主干中联合建模异构特征交互与门控序列注意力，解决了推荐系统中序列与非序列特征信息交换受限以及多场景负迁移的问题。 |
| [^40] | [Comparing GenAI to Machine Learning and Manual Review for eDiscovery](https://arxiv.org/abs/2610.16925) | 该研究指出此前声称生成式AI审查工具（88%召回率）优于机器学习工具（64%召回率）的结论因方法差异而不可比，实际分析发现生成式AI与人工审查效果相当，而采用持续主动学习（CAL）的机器学习模型仅需2,000条样本训练即可达到最先进效果并具备更高效率。 |
| [^41] | [Inherit4Rec: Parameter Inheritance for Efficient Scaling of Recommendation Models](https://arxiv.org/abs/2609.23111) | 本文提出Inherit4Rec参数继承框架，同时支持稠密到稠密（D2D）扩展和稠密到稀疏（D2S）转换，无需从头重训即可在动态演化的推荐数据下高效扩展推荐模型容量并避免性能骤降。 |
| [^42] | [Benchmark Radar: A Living Database and Search Engine for AI Benchmarks and Evaluation](https://arxiv.org/abs/2609.11115) | 本文提出Benchmark Radar，一个每日自动发现并整合AI基准论文、数据集和代码的动态数据库与搜索引擎，为LLM评估、智能体、编程、推理和安全等领域提供可检索的基准目录、来源引用和分数历史。 |
| [^43] | [Validating FKG.in: Soundness Assessment in LLM-Augmented Indian Food Knowledge](https://arxiv.org/abs/2608.29249) | 本文作为印度食品知识图谱FKG.in的一部分，提出了一种半自动化的健全性评估工作流程，通过结合形式文法、词汇检查、统计启发式、Set Transformer连贯性建模和检索验证的多阶段方法，识别并解决LLM从非正式烹饪来源提取和增强结构化食谱数据时的常见失败模式。 |
| [^44] | [Projecting BrowseComp-Plus onto ClimbMix: Toward More Realistic Corpora for Agentic Search](https://arxiv.org/abs/2608.20317) | 本文提出了一种投影流水线，将BrowseComp-Plus基准的问题迁移到大规模真实网页语料库ClimbMix中，从而构建更真实的智能体搜索评估环境，同时保持问题质量。 |
| [^45] | [When Do Pretrained Audio Representations Help Music Recommendation? A Study Across Ranking and Generative Interfaces](https://arxiv.org/abs/2608.06928) | 该研究发现预训练音频表征对音乐推荐的价值高度依赖于推荐界面——音乐预训练和音频-文本对齐编码器在基于内容检索和语义ID生成式推荐中带来明显增益，但在SASRec序列推荐中经交互适配后优势大幅缩小。 |
| [^46] | [TokenMinds: Pretrained User Tokens and Embeddings for User Understanding in Large Recommender Systems](https://arxiv.org/abs/2606.25147) | TokenMinds基于预训练LLM改造的编码器-解码器架构并采用双输出设计，将PLUM框架从基于SID的物品检索扩展到生成式序列用户建模，同时产出稠密用户嵌入与离散用户Token，突破了固定维度稠密嵌入的表征限制并捕捉用户行为的序列模式。 |
| [^47] | [LRanker: Scaling LLM Ranking to Millions of Candidates with Test-Time Scaling](https://arxiv.org/abs/2605.27810) | LRanker通过K-means候选聚合编码器与基于图的测试时扩展机制，将LLM排序高效扩展至数百万候选规模并提升排序质量。 |
| [^48] | [DeGenTWeb: A First Look at MGT-dominant Websites](https://arxiv.org/abs/2605.00087) | 提出了 DeGenTWeb 框架，通过将机器生成文本检测器适配到网页并聚合网站级检测结果，首次系统识别出以机器生成内容为主的网站，发现 Common Crawl 存档网站中 20.9% 和 Bing 搜索结果链接网站中 13.6% 属于机器生成文本主导型。 |
| [^49] | [Scoring-Rule Sensitivity in LLM Evaluation: Evidence from Indian Financial Regulatory Text](https://arxiv.org/abs/2604.19298) | 该研究构建了基于印度SEBI和RBI监管文本的专家标注基准IndiaFinBench，发现大语言模型的排名对评分规则高度敏感——严格字符串匹配与跨模型裁判两种评分机制之间甚至无显著的正向排名相关性——表明单一准确率数字是模型与评分规则的共同产物，不能可靠反映模型真实能力。 |
| [^50] | [LIME: Link-based User-item Interaction Modeling with Decoupled XOR Attention for Efficient Test Time Scaling](https://arxiv.org/abs/2510.18239) | LIME架构通过低秩链接嵌入解耦用户与候选交互、实现注意力权重预计算，并采用线性XOR注意力机制，从根本上降低了推荐系统推理时对候选集大小和用户序列长度的计算成本依赖。 |
| [^51] | [Role-Augmented Intent-Driven Generative Search Engine Optimization](https://arxiv.org/abs/2508.11158) | 提出RAID方法，通过从源内容推断检索意图并进行4W多角色深度反思，实现了在无需访问目标查询及其反馈的情况下进行生成式搜索引擎优化。 |

# 详细

[^1]: REGER：基于语义ID建模物品间关系的生成式推荐

    REGER: Modeling inter-item Relations with Semantic IDs for Generative Recommendation

    [https://arxiv.org/abs/2610.37584](https://arxiv.org/abs/2610.37584)

    提出REGER统一框架，通过生成式语义检索预训练使GR模型继承物品间语义相似性，并将物品间关系显式融入语义预训练、历史编码和SID生成的全过程。

    

    个性化推荐依赖于从在线内容和用户交互中学习物品之间的语义关系和协同关系。生成式推荐（GR）通过自回归地生成物品的离散语义ID（SIDs），提供了一种有前景的范式。然而，随机初始化的SID词元嵌入以及从零开始训练的GR模型并不能直接继承物品固有的语义关系，而物品间的协同关系主要是通过词元级注意力机制和下一词元预测监督被隐式学习的。为了解决这一局限，我们提出了REGER，这是一个统一框架，将物品间关系显式地融入语义预训练、历史编码和SID生成之中。我们通过生成式语义检索对GR模型进行预训练，使其通过学习生成语义邻居的SID来继承物品间的语义相似性。我们将全局注意力与局部注意力相结合……（摘要原文在此处不完整）

    arXiv:2610.37584v1 Announce Type: cross  Abstract: Personalized recommendation depends on learning semantic and collaborative relations among items from online content and user interactions. Generative recommendation (GR) offers a promising paradigm by autoregressively generating discrete semantic IDs (SIDs) of items. However, randomly initialized SID token embeddings and GR models trained from scratch do not directly inherit the semantic relations inherent in items, while inter-item collaborative relations are learned largely implicitly through token-level attention and next-token supervision. To address this limitation, we propose REGER, a unified framework that explicitly incorporates inter-item relations into semantic pretraining, history encoding, and SID generation. We pretrain the GR model through generative semantic retrieval, enabling it to inherit inter-item semantic similarity by learning to generate semantic neighbors' SIDs. We combine global attention with local attention 
    
[^2]: LCM-Promo：一种基于大语言模型的促销分配增益模型

    LCM-Promo: An LLM-Based Uplift Model for Promotion Allocation

    [https://arxiv.org/abs/2610.37570](https://arxiv.org/abs/2610.37570)

    LCM-Promo是一种在交易商务数据上微调的大语言模型促销分配增益模型，通过将单token输出解释为转化阈值概率来学习单调的促销-响应曲线，在生产A/B测试中将转化率提升1%并降低促销成本6%。

    

    我们提出了LCM-Promo，一种基于大语言模型（LLM）的促销分配增益模型。LCM-Promo是一个面向特定任务的大型商务模型（Large Commerce Model, LCM）：一个在交易型商务数据上微调的大语言模型，用于预测用户意图和促销响应。它通过将单token输出解释为转化阈值概率，从生产日志中学习单调的促销-响应曲线，同时以会话上下文（如日期和配送班次）作为条件。在一项针对流失用户召回的生产环境A/B测试中，与Segment-EMA生产基线相比，LCM-Promo将转化率提升了1%，并将促销成本降低了6%。

    arXiv:2610.37570v1 Announce Type: cross  Abstract: We present LCM-Promo, an LLM-based uplift model for promotion allocation. LCM-Promo is a task-specific Large Commerce Model (LCM): a finetuned LLM trained on transactional commerce data to predict user intent and promotion response. It learns monotonic promotion-response curves from production logs by interpreting single-token outputs as conversion-threshold probabilities, while conditioning on session context such as day and delivery shift. In a production A/B test on churn reactivation, LCM-Promo increased conversion by 1% and reduced promotion cost by 6% relative to the Segment-EMA production baseline.
    
[^3]: 互补信号能否桥接相似性孤岛？面向多模态推荐的流形增强图嵌入

    Can Complementary Signals Bridge Similarity Islands? Manifold-Augmented Graph Embedding for Multimodal Recommendation

    [https://arxiv.org/abs/2610.37409](https://arxiv.org/abs/2610.37409)

    提出MAGE框架，将互补性建模解耦为离线几何扭曲和在线拓扑传播，利用Soft Conceptor机制提取互补感知的残差方向并注入物品表示，从而桥接多模态推荐中语义孤立物品形成的“相似性孤岛”。

    

    多模态推荐系统（MMRS）利用丰富的物品特征来缓解数据稀疏问题，然而基于相似性驱动的图构建与传播往往更偏向语义相似的替代品，而非功能上相关的互补品。当行为关联将语义上相距较远的物品连接起来时，互补信号可能在几何上保持孤立，形成碎片化的“相似性孤岛”，并且在应用标准传播时产生语义漂移。为克服这些局限，我们提出了MAGE（流形增强图嵌入），这是一个用于个性化多模态排序的框架，它将互补性建模解耦为离线几何扭曲和在线拓扑传播两个阶段。离线阶段通过Soft Conceptor机制提取互补感知的残差方向，并在保持局部间隙的约束下将其注入物品表示中，在纳入共同购买证据的同时控制（摘要在此处截断）

    arXiv:2610.37409v1 Announce Type: new  Abstract: Multimodal Recommender Systems (MMRS) leverage rich item features to mitigate data sparsity, yet similarity-driven graph construction and propagation can favor semantically similar substitutes over functionally related complements. When behavioral associations connect semantically distant items, complementary signals can remain geometrically isolated, yielding fragmented "Similarity Islands" and semantic drift when standard propagation is applied. To overcome these limitations, we propose MAGE (Manifold-Augmented Graph Embedding), a framework for personalized multimodal ranking that decouples complementarity modeling into offline geometric warping and online topological propagation. The offline phase enhances item representations by extracting complement-aware residual directions via a Soft Conceptor mechanism and injecting them under a local gap-preserving constraint, incorporating co-purchase evidence while controlling the magnitude of
    
[^4]: 何时值得额外检索？RAG中证据增益与答案可靠性的成本感知研究

    When Is Additional Retrieval Worthwhile? A Cost-Aware Study of Evidence Gain and Answer Reliability in RAG

    [https://arxiv.org/abs/2610.37358](https://arxiv.org/abs/2610.37358)

    该论文提出一种成本感知的检索路由策略，通过在路由前预估混合检索相对于稠密检索基线的问题级增量证据收益，仅在约四分之一的问题上触发更昂贵的混合检索，在保持相同证据Hit@10的同时显著降低检索成本。

    

    检索增强生成（RAG）可以从更复杂的检索中受益，但对许多问题而言，额外的检索可能是不必要的，而且证据检索上的收益也不一定能转化为忠实或正确的答案。我们研究了一种成本感知的双动作检索策略，在“稠密检索加重排序”基线与“词汇-稠密混合检索器”之间进行选择。与简单地将基线检索归类为成功或失败不同，我们利用路由决策之前即可获得的特征，估计混合检索针对特定问题的增量证据检索收益。正则化线性模型与决策阈值通过按文档分组的交叉验证进行选择，并在留出集评估之前冻结。在60篇此前未见的QASPER训练集划分文章上（188个问题，其中169个可进行证据评估），该策略在46个问题（24.5%）上触发了混合检索，并获得了与……相同的证据Hit@10（摘要在此处截断）。

    arXiv:2610.37358v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) can benefit from more sophisticated retrieval, but extra retrieval may be unnecessary for many questions and gains in evidence retrieval may not translate into faithful or correct answers. We study a cost-aware, two-action retrieval policy that chooses between a dense-retrieval-plus-reranking baseline and a hybrid lexical--dense retriever. Rather than classifying baseline retrieval as simply successful or unsuccessful, we estimate the question-specific incremental evidence-retrieval benefit of hybrid retrieval using features available before the routing decision. A regularized linear model and decision threshold are selected using document-grouped cross-validation and frozen prior to held-out evaluation. On 60 previously unseen QASPER training-split articles (188 questions, 169 evidence-evaluable), the policy triggers hybrid retrieval on 46 questions (24.5\%) and obtains the same evidence Hit@10 as al
    
[^5]: LESSRec：面向基于语义ID推荐的层级特定历史建模

    LESSRec: Level-Specific History Modeling for Semantic ID-Based Recommendation

    [https://arxiv.org/abs/2610.37286](https://arxiv.org/abs/2610.37286)

    LESSRec通过按语义层级分别编码历史标记序列，并将预测时的注意力限制在目标标记所在层级，同时辅以按时间顺序排列的物品级语义序列，从而提升基于语义ID的推荐效果。

    

    基于语义ID的推荐通过残差量化将每个物品表示为一个语义标记序列。该范式以自回归方式从粗到细地生成下一个物品的标记，要求每次预测都关注所有语义层级的历史标记。然而，我们的推理时分析发现，与目标标记处于同一语义层级的历史标记往往更具信息量，而其他层级的历史标记可能引入干扰。受这些发现的启发，我们提出了LESSRec，它分别编码每个语义层级的历史标记序列，并在预测时将对历史标记的注意力限制在目标标记所在语义层级的序列上。为了补充这些层级特定的历史信息，LESSRec还显式提供了一个物品级语义序列，按时间顺序表示每个历史物品的整体语义。解码器通过访问层级对齐的标记……

    arXiv:2610.37286v1 Announce Type: new  Abstract: Semantic ID-based recommendation represents each item as a sequence of semantic tokens obtained through residual quantization. This paradigm generates the next item's tokens autoregressively from coarse to fine, entailing each prediction attending to historical tokens at all semantic levels. However, our inference-time analysis reveals that historical tokens at the target-token's semantic level tend to be more informative, whereas other levels can introduce interference. Motivated by these findings, we propose LESSRec that separately encodes the historical token sequence at each semantic level and restricts attention over historical tokens to the sequence at the target token's semantic level during prediction. To complement these level-specific histories, LESSRec explicitly provides an item-level semantic sequence that represents each historical item's overall semantics in chronological order. The decoder accesses the level-aligned token
    
[^6]: 基于有向无环图建模偏好依赖的叙事驱动推荐

    Modeling Preference Dependencies with Directed Acyclic Graphs for Narrative-Driven Recommendation

    [https://arxiv.org/abs/2610.37280](https://arxiv.org/abs/2610.37280)

    提出偏好依赖图框架（PDG-F），利用有向无环图显式建模用户叙事中偏好之间的层级依赖关系，并通过有向路径检索和拓扑路径排序实现叙事驱动的推荐。

    

    叙事驱动推荐允许用户通过自由文本表达多方面的需求，其中所陈述的偏好之间自然地相互限定和制约。现有方法通常将叙事视为扁平的关键词组合或无约束的生成式查询，无法捕捉复杂偏好如何以层级方式相互细化并赋予语境。我们提出了偏好依赖图框架（PDG-F），这是一个以显式偏好依赖为核心的叙事驱动推荐框架。具体而言，PDG-A（PDG智能体）首先解析非结构化请求，将用户偏好及其相互之间的限定关系组织成有向无环的偏好依赖图（PDG）。通过该图，经由有向路径检索从开放网络获取候选证据，最终推荐由拓扑路径排序确定。在三个涵盖英语等多种场景的多样化领域上进行了全面的实验……

    arXiv:2610.37280v1 Announce Type: cross  Abstract: Narrative-driven recommendation allows users to express multifaceted needs through free-form text where stated preferences naturally qualify and condition one another. Existing methods typically treat narratives as flat keyword combinations or unconstrained generative queries, failing to capture how complex preferences hierarchically refine and contextualize each other. We propose the Preference Dependency Graph Framework (PDG-F), a narrative-driven recommendation framework centered on explicit preference dependencies. Specifically, PDG-A (PDG-Agent) first parses unstructured requests to organize user preferences and their mutual qualifications into a directed acyclic Preference Dependency Graph (PDG). Through this graph, candidate evidence from the open Web is acquired via directed path retrieval, and final recommendations are determined by topological path ranking. Comprehensive experiments across three diverse domains spanning Engli
    
[^7]: GradCO：面向基于文本行人异常搜索的行为感知梯度冲突卸载方法

    GradCO: Behavior-Aware Gradient-Conflict Offloading for Text-Based Person Anomaly Search

    [https://arxiv.org/abs/2610.37267](https://arxiv.org/abs/2610.37267)

    该论文提出 GradCO 框架，通过行为感知混合专家（B-MoE）与冲突卸载学习（COL）等组件，解决文本行人异常搜索中同身份不同行为困难负样本导致的梯度冲突以及外观线索掩盖微弱行为证据的问题。

    

    基于文本的行人异常搜索（TPAS）利用自然语言描述中关于外观以及正常或异常行为的信息来检索目标行人。与传统的基于文本的行人检索不同，TPAS 必须在保留身份线索的同时区分细粒度行为。这带来了两个挑战：同身份、不同行为的困难负样本在细粒度对齐过程中会引起梯度冲突，而强烈的外观线索在聚合过程中可能掩盖微弱的行为证据。我们提出了 GradCO（行为感知梯度冲突卸载方法），由三个互补组件构成。行为感知混合专家（B-MoE）构建了外观、正常行为和异常行为三个流，其角色从检索目标中学习得到，无需专家特定的语义标签。冲突卸载学习（COL）在相似度计算中使用基于样本对条件的卸载锚点，以减轻主网络上的冲突梯度负担……（摘要原文在此处截断）

    arXiv:2610.37267v1 Announce Type: cross  Abstract: Text-based Person Anomaly Search (TPAS) retrieves a target pedestrian using natural-language descriptions of appearance and normal or anomalous behavior. Unlike conventional text-based person retrieval, TPAS must preserve identity cues while distinguishing fine-grained behaviors. It presents two challenges: same-identity, different-behavior hard negatives cause gradient conflict during granular alignment, while strong appearance cues can obscure weak behavioral evidence during aggregation. We propose GradCO (behavior-aware Gradient-Conflict Offloading), comprising three complementary components. Behavior-Aware Mixture-of-Experts (B-MoE) forms appearance, normal-behavior, and anomalous-behavior streams whose roles are learned from retrieval objectives without expert-specific semantic labels. Conflict Offload Learning (COL) uses pair-conditioned offload anchors in similarity computation to reduce the conflicting-gradient burden on the ma
    
[^8]: TaskBridge：SIGIR 2026电子商务挑战赛技术报告

    TaskBridge: Technical Report for SIGIR 2026 eCommerce Challenge

    [https://arxiv.org/abs/2610.37231](https://arxiv.org/abs/2610.37231)

    提出TaskBridge框架，通过目录语义锚定、五候选统一改写和协议感知选择，将任务导向查询改写为适合商品检索的表达式，有效弥合查询与商品目录之间的词汇鸿沟。

    

    SIGIR 2026电子商务挑战赛的查询改写任务评估将面向任务的查询改写为适合商品检索的表达式，随后进行基于BM25的检索。这些查询通常通过活动层面的语义隐含商品需求，与商品目录描述之间形成了词汇鸿沟。系统必须在保持任务意图的同时，根据自动和交互两种协议调整改写的数量。我们提出了TaskBridge，这是一个以商品目录为锚点的框架，包含目录语义锚定、统一的五候选改写和协议感知选择三个模块。该框架将查询锚定在目录词汇中，生成五个互补的改写版本，并利用BM25分数、查询-商品重排序和加权Jaccard重叠度，为自动检索选择一个查询，或为交互检索选择最多四个查询。选定的查询通过BM25独立执行，以生成所需六列格式的结果。

    arXiv:2610.37231v1 Announce Type: new  Abstract: The SIGIR 2026 eCommerce Challenge Query Reformulation Task evaluates the reformulation of task-oriented queries into expressions suitable for product retrieval, followed by BM25-based retrieval. These queries often imply product needs through activity-level semantics, creating a lexical gap with catalog descriptions. Systems must preserve task intent while adapting the number of reformulations to the Automatic and Interactive protocols. We propose TaskBridge, a catalog-anchored framework comprising Catalog Semantic Anchoring, Unified Five-Candidate Rewriting, and Protocol-Aware Selection. It grounds queries in catalog vocabulary, generates five complementary reformulations, and selects one query for Automatic retrieval or up to four for Interactive retrieval using BM25 scores, query-product reranking, and weighted Jaccard overlap. The selected queries are independently executed using BM25 to produce results in the required six-column fo
    
[^9]: AutoRSE：将大语言模型智能体评估为自演化推荐系统的推荐工程师

    AutoRSE: Evaluating LLM Agents as Recommendation Engineers for Self-Evolving Recommender Systems

    [https://arxiv.org/abs/2610.37228](https://arxiv.org/abs/2610.37228)

    该论文提出了AutoRSE评估框架，通过覆盖六项工业级推荐任务的大规模实证研究发现，基于LLM的推荐工程智能体能将推荐性能平均提升16.1%，但目前主要充当工程优化者而非算法创新者。

    

    推荐系统工程智能体（RSE-Agents）利用基于大语言模型的智能体自主改进推荐算法，正在成为一种有前景的新范式。然而，由于缺乏统一、标准化的评估框架以及系统性的轨迹级分析，其能力与局限性仍然知之甚少。为填补这一空白，我们提出了AutoRSE——一个覆盖六项代表性工业级推荐任务的评估框架，并对三个前沿大语言模型在三种广泛使用的智能体框架上开展了大规模实证研究。结果表明，RSE-Agents在六项任务中的五项上持续提升了骨干模型的性能，平均相对提升达16.1%，表明它们已经能够自主改进推荐性能。然而，轨迹级分析也揭示，当前的RSE-Agents主要扮演的是工程优化者的角色，而非算法……（摘要原文在此处截断）

    arXiv:2610.37228v1 Announce Type: new  Abstract: Recommender Systems Engineer Agents (RSE-Agents), which use LLM-based agents to autonomously improve recommendation algorithms, are emerging as a promising paradigm. However, their capabilities and limitations remain poorly understood due to the lack of a unified, standardized evaluation framework and systematic trajectory-level analysis. To address this gap, we introduce AutoRSE, an evaluation framework spanning six representative industrial-grade recommendation tasks, and conduct a large-scale empirical study of three frontier LLMs across three widely used agent harnesses. Our results show that RSE-Agents consistently enhance the performance of backbones on five of the six tasks, with an average relative improvement of 16.1%, indicating that they can already autonomously improve recommendation performance. However, trajectory-level analysis also reveals that current RSE-Agents function primarily as engineering optimizers rather than al
    
[^10]: 面向高效近似最近邻搜索的邻近图索引无损压缩

    Lossless Compression of Proximity Graph Indexes for Efficient Approximate Nearest Neighbors Search

    [https://arxiv.org/abs/2610.37176](https://arxiv.org/abs/2610.37176)

    该论文提出一种专为HNSW实时路由设计的无损压缩技术，通过离线图置换诱导结构局部性，使SIMD加速的整数压缩器（如Stream VByte）能够在不牺牲查询延迟的前提下快速解压邻接列表，从而大幅降低图索引的内存占用。

    

    基于图的近似最近邻（ANN）索引，如HNSW，是现代向量搜索的基石。然而，其巨大的内存占用造成了严重的基础设施瓶颈。虽然向量量化被广泛采用以减少存储数据向量所需的内存，但图的边结构日益成为总内存使用的主要部分。为在线搜索压缩这些邻接列表带来了一个关键的系统挑战：由于解压缩发生在查询处理过程中，即在图导航期间进行，因此内存节省不能以牺牲查询延迟为代价。本文提出了一种专门为实时HNSW路由定制的压缩技术。通过应用离线图置换来人为地诱导结构局部性，我们使SIMD加速的整数压缩器（如Stream VByte）能够进行快速的即时解压缩。在多样化的真实世界工作负载上的评估表明，我们显著……（原文摘要在此处被截断）

    arXiv:2610.37176v1 Announce Type: new  Abstract: Graph-based Approximate Nearest Neighbor (ANN) indexes, such as HNSW, are the backbone of modern vector search. However, their massive memory footprint creates a severe infrastructure bottleneck. While vector quantization is widely adopted to reduce the memory required to store data vectors, the graph's edge structure increasingly dominates total memory usage. Compressing these adjacency lists for online search presents a critical system challenge: because decompression occurs during query processing, i.e., while navigating the graph, memory savings cannot come at the expense of query latency. In this paper, we present a compression technique explicitly tailored for live HNSW routing. By applying offline graph permutations to artificially induce structural locality, we enable the use of SIMD-accelerated integer compressors, such as Stream VByte, for rapid on-the-fly decompression. Evaluated across diverse real-world workloads, we signifi
    
[^11]: 矛盾下的覆盖性评估：一个否定感知检索的基准

    Coverage Under Contradiction: A Benchmark for Negation-Aware Retrieval

    [https://arxiv.org/abs/2610.37162](https://arxiv.org/abs/2610.37162)

    该论文提出了 CoverageBench-Negate 基准，通过为每个信息点构建否定版本并相应编辑支持文档，首次系统性地研究了检索器在语料库证据与其参数化知识相矛盾时的行为表现。

    

    覆盖性评估基准用于量化检索到的证据是否覆盖了满足信息需求所需的各种信息片段。现有基准大多只考虑预训练检索组件可能已经掌握的事实性陈述，因此尚未解决一个开放性问题：当语料库证据与这些先验知识相矛盾时，检索器能否找到证据。我们研究了当检索器的参数化知识与语料库中的信息不一致时，检索器的行为表现。为此，我们开发了一种流程，通过受控的矛盾对基于信息点（nugget）的覆盖性基准进行增强。对于每个信息点，我们创建其否定版本，并编辑其支持文档使其支持该否定陈述，同时保留原有上下文。将该流程应用于衡量信息覆盖性的基准 CoverageBench，我们构建了 CoverageBench-Negate，涵盖六个数据集以及约 63,000 个“原始-编辑”证据对（每个信息点-文档链接对应一对）。我们还附带了丰富的构建元数据……

    arXiv:2610.37162v1 Announce Type: cross  Abstract: Coverage-evaluation benchmarks quantify whether retrieved evidence covers the different pieces of information needed to satisfy an information need. Existing benchmarks mostly consider factual claims that may already be known to pretrained retrieval components, leaving open whether retrievers find corpus evidence when it contradicts those priors. We study how retrievers behave when their parametric knowledge and the information in a corpus disagree. To do so, we develop a process to augment a nugget-based coverage benchmark with controlled contradictions. For each nugget, we create a negated counterpart and edit its supporting documents to support that negation while preserving the context. Applying this process to CoverageBench, a benchmark measuring information coverage, we create CoverageBench-Negate, spanning six datasets and roughly 63,000 original-edited evidence pairs (one per nugget-document link). We attach rich construction m
    
[^12]: 重新思考CTR预测中的位置编码与交互粒度

    Rethinking Positional Encoding and Interaction Granularity for CTR Prediction

    [https://arxiv.org/abs/2610.37140](https://arxiv.org/abs/2610.37140)

    提出TRIP模型，将目标感知的秩-时间位置编码与时间组交互建模相结合，揭示了大多数现有位置编码并不优于无位置编码，从而重新思考了CTR预测真正所需的位置编码与交互粒度。

    

    点击率（CTR）预测依靠历史行为和上下文特征来估计用户对目标物品的点击。现有基于Transformer的CTR模型通常采用基于索引或时间的位置编码以及token级别的因果交互。然而，我们观察到大多数现有位置编码的性能并不优于NoPE（无位置编码），而放宽目标相对时间组内的因果约束也能带来性能提升。这些观察引发了一个根本性问题：CTR预测究竟需要什么样的位置编码和交互粒度？为了回答这一问题，我们提出了TRIP，它将目标感知的秩-时间位置编码与时间组交互建模相结合。具体而言，因子化秩编码通过低秩因子分解表示每个行为在历史序列中的序数位置，而目标相对时间编码则刻画其与目标物品的时间距离

    arXiv:2610.37140v1 Announce Type: new  Abstract: Click-through rate (CTR) prediction relies on historical behaviors and contextual features to estimate clicks on a target item. Existing Transformer-based CTR models often adopt index-based or temporal positional encodings and token-level causal interaction. However, we observe that most existing positional encodings do not outperform NoPE, while relaxing causal constraints within target-relative temporal groups also yields gains. These observations motivate a fundamental question: what positional encoding and interaction granularity does CTR prediction actually require? To address this question, we propose \textbf{TRIP}, which combines target-aware rank--time positional encoding with temporal-group interaction modeling. Specifically, Factorized Rank Encoding represents each behavior's ordinal position within the historical sequence through a low-rank factorization, while Target-Relative Time Encoding characterizes its temporal distance 
    
[^13]: Tessera：面向网页检索增强生成的结构补全证据单元

    Tessera: Structure-Completed Evidence Units for Web Retrieval-Augmented Generation

    [https://arxiv.org/abs/2610.37080](https://arxiv.org/abs/2610.37080)

    提出无需训练的证据传递层Tessera，通过将网页HTML重建为结构补全的细粒度证据单元并解耦检索与阅读，解决了表格、信息框等结构化内容在RAG流水线中因分块而丢失证据的问题。

    

    网页检索增强生成（RAG）将页面切分为固定大小的文本块，或在近期的视觉流水线中切分为截图瓦片，检索到的条目同时充当索引、排序和阅读的单元。我们在完整的英文维基百科上追踪答案证据从源HTML到阅读器输入的全过程，发现这种耦合在文本和截图两类流水线中都会丢失证据：源页面中存在的证据往往无法完整送达阅读器，尤其是表格和信息框这类结构化内容，因为分块处理将其与赋予其意义的标题和上下文分离了。更强的抽取方法、上下文扩展以及更大的阅读器预算都无法消除这种损失。我们提出Tessera，一个无需训练的证据传递层，它将流水线检索的内容与阅读器读取的内容解耦。给定任何排序列表背后的页面，Tessera会将其HTML重建为细粒度的证据单元，每个单元都补全了其……（原文摘要在此处不完整）

    arXiv:2610.37080v1 Announce Type: new  Abstract: Web retrieval-augmented generation (RAG) splits a page into fixed-size text chunks or, in recent visual pipelines, screenshot tiles, and the retrieved item serves at once as the unit of indexing, ranking, and reading. We trace answer evidence from source HTML to reader input over the full English Wikipedia and find that this coupling loses evidence in text and screenshot pipelines alike: evidence present in the source page often fails to reach the reader intact, particularly structured content such as tables and infoboxes, which chunking separates from the headers and context that give it meaning. Stronger extraction, context expansion, and larger reader budgets leave the loss in place. We present Tessera, a training-free evidence delivery layer that decouples what a pipeline retrieves from what its reader reads. Given the pages behind any ranked list, Tessera rebuilds their HTML into fine-grained evidence units, each completed with its 
    
[^14]: Compote：面向序列推荐的抗坍缩多假设学习

    Compote: Collapse-Resistant Multiple Hypothesis Learning For Sequential Recommendation

    [https://arxiv.org/abs/2610.37005](https://arxiv.org/abs/2610.37005)

    COMPOTE提出了一种用于序列推荐的抗坍缩多假设学习方法，通过退火目标促使多个假设分工覆盖不同商品区域以表达用户多重意图，并从理论上证明其最优解满足最大熵量化原理且保持贝叶斯最优排序。

    

    下一项预测本质上是模糊的：单一的交互历史往往存在多种合理的后续延续，分布在不同的商品目录区域。标准的序列推荐模型将这些备选方案压缩为单一预测，从而混淆了用户意图。我们提出了COMPOTE（抗坍缩多假设学习），这是一种多假设Transformer，采用退火目标函数进行训练，该目标从均匀监督逐渐过渡到赢家通吃的分配方式。这一机制迫使各个假设专门化到不同的商品目录区域，而推理阶段则依赖高效的无归一化最大相似度规则。理论上，我们证明在总体情况下，最优解依据最大熵量化原理对条件下一项分布进行划分，同时汇集分数能保持贝叶斯最优排序。我们还证明了随假设数量呈对数级扩展的泛化边界……（原文摘要至此截断）

    arXiv:2610.37005v1 Announce Type: cross  Abstract: Next-item prediction is inherently ambiguous: a single interaction history often admits multiple plausible continuations across distinct catalogue regions. Standard sequential recommenders compress these alternatives into a single prediction, conflating user intents. We introduce COllapse-resistant Multiple hyPOThesis lEarning (COMPOTE), a multi-hypothesis transformer trained with an annealed objective that transitions from uniform supervision to a winner-takes-all assignment. This mechanism forces hypotheses to specialize into distinct catalogue regions, while inference relies on an efficient, normalizer-free max-similarity rule. Theoretically, we show that in the population case, the optimal solutions partition the conditional next-item distribution according to a max-entropy quantization principle, while pooled scores preserve the Bayes-optimal ranking. We also prove a generalization bound scaling logarithmically with the number of 
    
[^15]: SpectRAG：面向多跳检索增强生成中基于Token预算的上下文选择的加权分块图

    SpectRAG: Weighted Chunk Graphs for Token-Budgeted Context Selection in Multi-Hop Retrieval-Augmented Generation

    [https://arxiv.org/abs/2610.36991](https://arxiv.org/abs/2610.36991)

    SpectRAG提出了一种谱图增强检索框架，通过构建编码多种语义与结构关系的加权分块图，结合谱相关性传播与Token预算约束下的上下文选择，解决传统独立检索在多跳推理中证据冗余和不完整的问题。

    

    检索增强生成（RAG）系统通过检索外部证据并将生成过程以检索到的上下文为条件，从而为大型语言模型（LLM）提供事实依据。尽管稠密检索在局部语义匹配方面较为有效，但独立排序的检索可能返回冗余段落、遗漏跨分块的依赖关系，并为多跳推理提供不完整的证据。我们提出了SpectRAG——一种谱图增强检索框架，它结合了混合候选种子选择、加权分块图构建、基于图的扩展、谱相关性传播以及基于Token预算的上下文选择。文档分块由稠密语义嵌入和图坐标系表示，后者源自一个加权分块图，其边编码了语义相似性、文档相邻性、章节层次结构、共享实体、引用链接以及LLM生成的效用问题之间的相似性。在查询时，SpectRAG构建一个hi……（摘要原文在此截断）

    arXiv:2610.36991v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) systems ground large language models (LLMs) by retrieving external evidence and conditioning generation on the retrieved context. Although dense retrieval is effective for local semantic matching, independently ranked retrieval can return redundant passages, miss cross-chunk dependencies, and provide incomplete evidence for multi-hop reasoning. We propose SpectRAG, a spectral graph-augmented retrieval framework that combines hybrid candidate seeding, weighted chunk graph construction, graph-based expansion, spectral relevance propagation, and token-budgeted context selection. Document chunks are represented by dense semantic embeddings and graph-spectral coordinates derived from a weighted chunk graph whose edges encode semantic similarity, document adjacency, section hierarchy, shared entities, citation links, and similarity between LLM-generated utility questions. At query time, SpectRAG builds a hi
    
[^16]: KL-MOPD：面向生成式推荐的差异感知多目标在线策略蒸馏

    KL-MOPD: Discrepancy-Aware Multi-Objective On-Policy Distillation for Generative Recommendation

    [https://arxiv.org/abs/2610.36973](https://arxiv.org/abs/2610.36973)

    该论文提出KL-MOPD方法，通过差异感知的多目标在线策略蒸馏来缓解token级蒸馏难度不平衡与跨token教师不一致的问题，从而在生成式推荐中更好地协调点击、转化等多个推荐目标。

    

    协调多个推荐目标（例如点击与转化）对真实场景下的生成式推荐（GR）至关重要。然而，由于在GR中，一个正样本物品只有在每个解码步骤中其token被依次成功选中时才能被检索到，传统的物品级方法缺乏细粒度的token级监督来引导中间步骤。受其在基础模型中成功的启发，我们考虑将多目标在线策略蒸馏（MOPD）引入GR，在每个token处将两个单目标教师模型蒸馏到一个学生模型中。然而，我们发现，在原始的MOPD中：（i）教师模型在每个token处表现出不相等的蒸馏难度（即token级不平衡），（ii）更难蒸馏的教师模型会随token而变化（即跨token不一致性），这使得每个目标在某些步骤中学习不佳，并阻碍了两个目标的正样本物品检索，导致次优性能。为了解决这一问题……（原文摘要在此处截断）

    arXiv:2610.36973v1 Announce Type: new  Abstract: Coordinating multiple recommendation objectives (e.g., click and conversion) is critical for real-world generative recommendation (GR). However, since a positive item in GR can be retrieved only when its tokens are successively selected at every decoding step, traditional item-wise methods lack fine-grained token-wise supervision to guide intermediate steps. Motivated by its success in foundation models, we consider introducing multi-objective on-policy distillation (MOPD) into GR to distill two single-objective teachers into a student at each token. Yet, we find that in vanilla MOPD, (i) teachers exhibit unequal distillation difficulty at each token (i.e., token-level imbalance), and (ii) the harder-to-distill teacher varies across tokens (i.e., cross-token inconsistency), leaving each objective poorly learned at certain steps and hindering positive item retrieval for both objectives, leading to suboptimal performance.   To address this
    
[^17]: 拍立淘MMSearch2.0技术报告：将生成式检索扩展至统一的端到端多模态搜索

    PailitaoMMSearch2.0 Technical Report: Scaling Generative Retrieval to Unified End-to-End Multimodal Search

    [https://arxiv.org/abs/2610.36958](https://arxiv.org/abs/2610.36958)

    提出了拍立淘MMSearch，一个统一的端到端多模态生成式检索系统，可直接从原始查询图像和自由形式指令生成商品语义ID，并在数据、意图等三个维度上扩展，以支持开放式图文查询、异构购物意图和不同资源约束的在线场景。

    

    生成式检索模型在搜索和推荐系统中的应用日益广泛。然而，工业界的生成式检索模型通常专门针对固定的在线场景：它们从场景特定的交互中学习，仅支持狭窄的查询形式和检索目标。我们研究了如何消除这些耦合，构建一个统一的端到端多模态搜索系统，以支持开放式的图文查询、异构的购物意图以及具有不同资源约束的在线场景。我们提出了拍立淘MMSearch，它可以直接从原始查询图像和自由形式指令生成商品语义ID，并在三个维度上进行扩展。数据扩展使用可复用的工具集，可组合成针对特定意图和生命周期的工作流，用于冷启动合成、质量控制、真实流量细化和持续数据生产。意图扩展逐步（摘要原文在此处被截断）……

    arXiv:2610.36958v1 Announce Type: cross  Abstract: Generative retrieval models have been increasingly adopted in search and recommendation systems. However, industrial generative retrieval models are commonly specialized to a fixed online scenario: they learn from scenario-specific interactions, support a narrow set of query forms and retrieval objectives. We study how to remove these couplings and build a unified end-to-end multimodal search system for open-ended image--text queries, heterogeneous shopping intents, and online scenarios with different resource constraints. We present \textbf{PailitaoMMSearch}, which directly generates product Semantic IDs from a raw query image and free-form instruction and scales along three dimensions. \textbf{Data scaling} uses a reusable toolset that can be composed into intent- and lifecycle-specific workflows for cold-start synthesis, quality control, real-traffic refinement, and continuous data production. \textbf{Intent scaling} progressively e
    
[^18]: 当扩散推荐器停止采样时：保持生成多样性

    When Diffusion Recommenders Stop Sampling: Preserving Generative Diversity

    [https://arxiv.org/abs/2610.36954](https://arxiv.org/abs/2610.36954)

    该论文发现目标物品扩散推荐器存在严重采样坍塌问题（源于隐空间与去噪器联合学习以及轨迹内离散损失监督两种设计缺陷），并提出三阶段潜在扩散框架DARTS，通过解耦表示学习、生成建模与离散排序，在保持推荐准确性的同时保留了生成多样性。

    

    用户的交互历史可能存在多种合理的下一次交互，然而传统的序列推荐器只产生单一确定性排序。虽然扩散模型提供了一种自然的采样替代延续的方式，但我们发现现有的目标物品扩散推荐器往往遭受严重的采样坍塌问题，在多次独立采样时产生几乎相同的输出。我们识别出两种导致这种坍塌的普遍设计选择：隐空间与去噪器的联合学习，以及通过离散损失进行的轨迹内监督。为了解决这些失效模式，我们提出了DARTS（通过采样实现多样且准确的推荐），这是一个三阶段潜在扩散框架，通过在扩散训练之前学习并冻结隐空间，以及将排序监督延迟到最终解码阶段，显式地解耦了表示学习、生成建模和离散排序。

    arXiv:2610.36954v1 Announce Type: new  Abstract: A user history can admit multiple plausible next interactions, yet conventional sequential recommenders produce a single deterministic ranking. While diffusion models offer a natural way to sample alternative continuations, we show that existing target-item diffusion recommenders often suffer from severe sampling collapse, producing nearly identical outputs across independent draws. We identify two ubiquitous design choices that promote this collapse: joint learning of the latent space and denoiser, and in-trajectory supervision via discrete losses. To address these failure modes, we introduce DARTS (Diverse and Accurate Recommendation Through Sampling), a three-stage latent diffusion framework that explicitly decouples representation learning, generative modeling, and discrete ranking by learning and freezing the latent space before diffusion training and postponing ranking supervision to the final decoding stage. We further characteriz
    
[^19]: 我们需要大语言模型来知道来源为何冲突吗？面向RAG中知识冲突分类的决策模型

    Do We Need an LLM to Know Why Sources Conflict? Decision Models for Knowledge-Conflict Typing in RAG

    [https://arxiv.org/abs/2610.36915](https://arxiv.org/abs/2610.36915)

    小型类型化决策模型在识别RAG中检索来源的冲突类型方面，可以以低4至39倍的成本媲美甚至超越Gemini 2.5 Flash等前沿大语言模型。

    

    当为查询检索到的来源相互矛盾时，正确的回应取决于它们为何矛盾：一个存在多个有效答案的问题、一个有争议的话题、过时的信息和虚假的来源，各自需要不同的处理行为。Cattan等人（2025）表明，将冲突类型告知生成器可使其回应提升约24分，但即使对前沿大语言模型而言，预测冲突类型也很困难。我们研究类型化决策模型——在单次前向传播中返回固定选项的概率分布而非生成文本——能否快速且低成本地完成这一分类任务。我们使用CONFLICTS基准测试并原封不动地采用原论文的提示词，评估了由一个商业推理网关提供的全部11个决策模型（来自7家提供商），并在相同输入上与Gemini模型进行对比。直接提问时，决策模型达到50.0%至67.0%的准确率；11个模型中有10个达到或超过Gemini 2.5 Flash（论文所用模型，准确率61.3%），而成本低4至39倍。

    arXiv:2610.36915v1 Announce Type: new  Abstract: When the sources retrieved for a query disagree, the right response depends on why they disagree: a question with several valid answers, a contested topic, stale information and a false source each call for different behaviour. Cattan et al. (2025) showed that telling a generator the conflict type improves its responses by about 24 points, but that predicting the type is hard even for frontier LLMs. We study whether typed decision models, which return probabilities over fixed options in a single forward pass instead of generating text, can do this classification fast and cheaply. Using the CONFLICTS benchmark and the original paper's prompt verbatim, we evaluate all 11 decision models served by one commercial inference gateway, from 7 providers, against Gemini models on identical inputs. Asked directly, the decision models reach 50.0--67.0% accuracy; 10 of the 11 match or exceed Gemini 2.5 Flash, the paper's model (61.3%), at 4--39x lowe
    
[^20]: 生成式推荐排序中的潜在残差推理

    Latent Residual Reasoning for Ranking in Generative Recommendation

    [https://arxiv.org/abs/2610.36890](https://arxiv.org/abs/2610.36890)

    该论文提出在生成式推荐排序中引入潜在残差推理，使基于ID的非文本推荐模型能够在推理时通过多步潜在计算提升对多候选物品的细粒度排序质量。

    

    面向排序的生成式推荐（GR）模型通常采用具有很强实际效率的“快思考”直接预测范式。然而，排序需要对多个候选物品进行细粒度打分，而现有GR模型仍主要依赖一次性预测。这引出了一个问题：额外的推理时计算能否进一步提升排序质量。最近，受大语言模型（LLM）中思维链（CoT）推理的启发，一些工作开始将推理引入推荐系统。其中部分方法依赖LLM推理框架在文本空间中执行显式推理，但这类设计并不适用于部署更为广泛的非文本推荐模型。一个更相关的方向是将潜在推理引入基于ID的推荐模型。尽管近期工作已将潜在推理扩展到工业级排序场景，但如何使连续的推理步骤（摘要在此处截断，后续内容缺失）

    arXiv:2610.36890v1 Announce Type: new  Abstract: Generative recommendation (GR) models for ranking typically adopt a fast-thinking direct prediction paradigm with strong practical efficiency. However, ranking requires fine-grained scoring across multiple candidate items, while existing GR models still mainly rely on one-shot prediction. This raises the question whether additional inference-time computation can further improve ranking quality. Recently, inspired by Chain-of-Thought (CoT) reasoning in large language models (LLMs), some work has begun to introduce reasoning into recommendation systems. Some of these methods rely on LLM reasoning frameworks to perform explicit reasoning in textual space, but such designs are not suitable for more widely deployed non-textual recommendation models. A more relevant direction is to introduce latent reasoning into ID-based recommendation models. Although recent work has extended latent reasoning to industrial ranking, how successive reasoning s
    
[^21]: DuTri：面向多模态多场景推荐的双分支建模与三元分解方法

    DuTri: Dual-Branch Modeling with Triadic Factorization for Multimodal Multi-Scenario Recommendation

    [https://arxiv.org/abs/2610.36836](https://arxiv.org/abs/2610.36836)

    提出DuTri框架，通过协同分支与多模态分支的双分支建模结合三元分解，分别满足协同信息与多模态信息的不同建模需求，从而实现更优的多模态多场景推荐效果。

    

    随着推荐场景日益多样化、多模态信息愈发普及，如何有效建模跨场景的用户偏好与多模态语义已成为推荐系统中的关键挑战。在引入多模态信息时，现有多场景推荐（MSR）方法通常将文本和视觉表示当作通用的输入特征，直接送入其原有的参数共享架构中。然而，这种扩展方式可能无法充分满足协同信息与多模态信息各自不同的建模需求，也未必是最佳的建模范式。我们的对照实验表明，直接采用这些架构并非最优的建模方式。为此，我们提出了双分支建模与三元分解方法DuTri，这是一个面向多模态多场景推荐的框架，由一个协同分支和一个……（原文摘要在此处截断）

    arXiv:2610.36836v1 Announce Type: new  Abstract: As recommendation scenarios become increasingly diverse and multimodal information grows more prevalent, effectively modeling cross-scenario user preferences and multimodal semantics has become a critical challenge in recommender systems. When incorporating multimodal information, existing multi-scenario recommendation (MSR) methods typically treat textual and visual representations as generic input features and directly feed them into their original parameter-sharing architectures. However, this extension may not fully accommodate the different modeling requirements of collaborative and multimodal information and may not be the best modeling paradigm. Our controlled experiments show that directly adopting these architectures is not the optimal modeling approach. To this end, we propose Dual-Branch Modeling with Triadic Factorization (DuTri), a framework for multimodal multi-scenario recommendation comprising a collaborative branch and a
    
[^22]: 词表已然就绪：将预训练可解释特征适配于生成式推荐

    The Vocabulary Is Already Here: Adapting Pretrained Interpretable Features for Generative Recommendation

    [https://arxiv.org/abs/2610.36763](https://arxiv.org/abs/2610.36763)

    该论文提出API框架，直接复用预训练稀疏自编码器的可解释特征索引作为生成式推荐的语义ID词表，无需额外的词表学习即可实现有效且可解释的推荐。

    

    生成式推荐（GR）通过语义ID（SID）来预测下一个物品。现有方法通常将SID词表拟合到预训练语言模型（PLM）对物品描述的表示上，这带来了额外的训练成本，并使词表覆盖率依赖于拟合的领域。我们观察到，预训练的稀疏自编码器能够揭示具有人类可解释语义标注的PLM特征，使特征索引可以作为可复用的SID词表。然而，复用该词表需要在紧凑的token预算下保持物品之间的区分度，并学习在具有相同主导特征的候选物品中偏向目标物品。这还需要将特征可解释性从物品内容扩展到检索决策。为应对这些挑战，我们提出API框架，通过适配预训练可解释词表，在无需额外词表学习的情况下实现有效且可解释的生成式推荐。

    arXiv:2610.36763v1 Announce Type: cross  Abstract: Generative recommendation (GR) predicts the next item through its semantic ID (SID). Existing methods commonly fit SID vocabularies to pretrained language model (PLM) representations of item descriptions, incurring additional training costs and making vocabulary coverage dependent on the fitting domains. We observe that pretrained sparse autoencoders expose PLM features with human-interpretable semantic annotations, allowing feature indices to serve as a reusable SID vocabulary. However, reusing this vocabulary requires preserving item distinctions under compact token budgets and learning to favor target items over candidates with shared dominant features. It also requires extending feature interpretability from item content to retrieval decisions. To address these challenges, we propose API, a framework that Adapts the Pretrained Interpretable vocabulary for effective and explainable GR without additional vocabulary learning. Within A
    
[^23]: 回忆还是重学？用于复发性概念漂移的上下文专家混合方法

    Recall or Relearn? A Mixture of In-Context Experts for Recurring Concept Drift

    [https://arxiv.org/abs/2610.36585](https://arxiv.org/abs/2610.36585)

    提出无需训练的MICE方法，通过在检测到概念漂移时切分数据流并维护多个存储上下文，让表格基础模型在历史概念重现时能够自适应地在“回忆”旧上下文与“重学”新标注之间做出决策，从而更好地应对复发性概念漂移。

    

    表格基础模型（TFMs）正被应用于网络入侵检测等网络数据流场景，它们通过更新上下文中已标注的数据行来适应变化，而无需重新训练。当一个过去的概念在离开上下文后再次出现时，表格基础模型面临“回忆还是重学”的决策：是恢复存储的该概念的数据行，还是从新标注中重新学习。我们提出了MICE（上下文专家混合，Mixture of In-Context Experts），这是一个无需训练的学习器，它在检测到变化时将数据流切分为片段。当表格基础模型对某片段估计的额外误差在容忍度范围内时，它将该已关闭的片段合并到存储的上下文中，同时对存储上下文的数量施加容量限制。一个快速层级根据存储上下文和先进先出（FIFO）窗口在最近一批标注数据上预测的对数损失来对它们进行加权。我们证明，在纯存储上下文和边际条件下，该快速层级在期望精度上相对于重学习获得了标准化……

    arXiv:2610.36585v1 Announce Type: cross  Abstract: Tabular foundation models (TFMs) are being applied to Web data streams such as network intrusion detection, where they adapt by updating the labelled rows of their context without retraining. When a past concept returns after leaving the context, a TFM faces a recall-or-relearn decision between restoring stored rows of the concept and relearning it from new labels. We propose MICE (Mixture of In-Context Experts), a training-free learner that cuts the stream into segments at detected changes. It merges a closed segment into a stored context when the TFM's estimated additional error on the segment is within a tolerance, under a capacity limit on stored contexts. A fast level weights the stored contexts and first-in-first-out (FIFO) windows by the log-loss of their predictions on the last labelled batch. We show that, with pure stored contexts and a margin condition, this fast level gains over relearning, in expected accuracy, the normali
    
[^24]: 面向公共行政的受管控主权RAG：公民服务中的三方认知路由、分面引导澄清与确定性数值验证

    Governed Sovereign RAG for Public Administration: Tripartite Epistemic Routing, Facet-Guided Clarification, and Deterministic Numeric Verification in Citizen Services

    [https://arxiv.org/abs/2610.36561](https://arxiv.org/abs/2610.36561)

    该论文提出并评估了一种部署于墨西哥特拉斯卡拉州官方门户的受管控主权RAG架构，通过三方认知路由门控、分面引导澄清和带SHA-256缓存失效的确定性数值验证三大支柱，解决政府公民服务中监管确定性、数据主权与查询模糊性之间的矛盾。

    

    在政府服务窗口部署对话式助手面临多重张力：监管确定性（对捏造费用的零容忍）、禁止使用商业云的数据主权要求，以及大量欠明确或超出管辖范围的公民查询。我们提出并评估了一种受管控的主权检索增强生成（RAG）架构，部署于墨西哥特拉斯卡拉州官方门户网站（tlaxcaladigital.gob.mx），覆盖1,143项办事流程以及解耦的924个办事机构目录。该系统通过三大支柱治理交互：（1）两阶段三方认知门控（clarify_policy 和基于 AnyJev 的玻尔兹曼分布），在直接回答（SPECIFIC）、分面引导澄清（CLARIFY）与拒答（ABSTAIN）之间进行仲裁；（2）生成后的确定性数值验证器（verify_numeric_guardrail），结合 SHA-256 缓存失效机制，确保每个生成的费用或有效期数字均归属……（摘要在此处截断）

    arXiv:2610.36561v1 Announce Type: cross  Abstract: Deploying conversational assistants at government service desks faces a tension between regulatory certainty (zero tolerance for invented fees), data sovereignty prohibiting commercial clouds, and high rates of underspecified or out-of-jurisdiction citizen queries. We present and evaluate a governed, sovereign Retrieval-Augmented Generation (RAG) architecture deployed on the official portal of the State of Tlaxcala, Mexico (tlaxcaladigital.gob.mx), covering 1,143 procedures and a decoupled directory of 924 offices. The system governs interactions via three pillars: (1) a two-stage tripartite epistemic gate (clarify_policy and a Boltzmann distribution over AnyJev) arbitrating among answering (SPECIFIC), facet-guided clarification (CLARIFY), and abstaining (ABSTAIN); (2) a post-generation deterministic numeric verifier (verify_numeric_guardrail) with SHA-256 cache invalidation ensuring every generated fee or validity figure belongs to th
    
[^25]: CCP-DP：用于DNA存储序列重建的跨簇上下文引导相位动态规划

    CCP-DP: Cross-cluster context-guided phase dynamic programming for DNA storage sequence reconstruction

    [https://arxiv.org/abs/2610.36559](https://arxiv.org/abs/2610.36559)

    该论文提出CCP-DP方法，利用数据集层面的跨簇上下文信息补充簇内局部证据，通过相位动态规划在低覆盖率和强IDS噪声条件下显著提升DNA存储序列的精确重建率（95.36%–99.98%），性能优于现有方法。

    

    动机：DNA数据存储具有高信息密度和长期稳定性，但在有限的测序覆盖率和较强的插入、删除与替换（IDS）噪声下，精确的序列重建仍然具有挑战性。现有的重建方法主要依赖于单个读段簇内的局部证据。当测序覆盖率较低或由插入/缺失引起的相位偏移变得显著时，这类局部证据可能不足以区分高度接近的相互竞争的候选序列。结果：我们提出了跨簇上下文引导相位动态规划（CCP-DP），该方法利用数据集层面的跨簇信息来补充簇内局部证据，并有选择地对模糊的临时共识序列进行精修。在主要基准测试设置下，CCP-DP实现了95.36%–99.98%的精确重建率，并在多个数据集上优于现有方法。在5×覆盖率下，CCP-DP提高了精确重建……（原文摘要至此截断）

    arXiv:2610.36559v1 Announce Type: new  Abstract: Motivation: DNA data storage offers high information density and long-term stability, but accurate sequence reconstruction remains challenging under limited sequencing coverage and strong insertion, deletion, and substitution (IDS) noise. Existing reconstruction methods mainly rely on local evidence within individual read clusters. When sequencing coverage is low or indel-induced phase shifts become pronounced, such local evidence may be insufficient to distinguish closely competing candidate sequences.   Results: We present Cross-Cluster Context-Guided Phase Dynamic Programming (CCP-DP), which uses dataset-level cross-cluster information to complement cluster-local evidence and selectively refines ambiguous provisional consensuses. Under the main benchmark settings, CCP-DP achieved exact reconstruction rates of 95.36%-99.98% and outperformed existing methods on multiple datasets. At $5\times$ coverage, CCP-DP improved exact reconstructi
    
[^26]: QESChunker：单一目标统一RAG中的重叠与非重叠分块

    QESChunker: A Single Objective Unifies Overlapping and Non-Overlapping Chunking for RAG

    [https://arxiv.org/abs/2610.36557](https://arxiv.org/abs/2610.36557)

    提出QESChunker，以统一的问题-证据效用最大化目标将RAG中的非重叠与重叠分块统一为有监督组合优化问题，非重叠情形可通过动态规划在多项式时间内精确求解。

    

    分块是检索增强生成（RAG）中的一个基础预处理步骤，然而现有的大多数分块器主要依赖启发式方法。我们提出了问题-证据监督分块器（QESChunker），据我们所知，这是首个从问题-证据监督中构建静态分块索引的方法。QESChunker通过在人工标注或大语言模型生成的问题-证据对上进行优化来构建文档特定的分块索引，且无需访问留出的测试查询。我们通过一个统一的问题-证据效用最大化目标来表述非重叠和重叠索引构建，从而将分块转化为一个有监督的组合优化问题。在非重叠约束下，该目标可以在各个分块上进行分解，并可通过动态规划在多项式时间内精确优化。当允许重叠时，该目标变为单调次模的；其精确优化是NP难的，但我们的g…（原文摘要在此处截断）

    arXiv:2610.36557v1 Announce Type: cross  Abstract: Chunking is a fundamental preprocessing step in retrieval-augmented generation (RAG), yet most existing chunkers rely primarily on heuristics. We introduce the Question--Evidence-Supervised Chunker (QESChunker), to our knowledge the first method for constructing static chunk indices from question--evidence supervision. QESChunker constructs document-specific chunk indices by optimizing over human-annotated or LLM-generated question--evidence pairs, without access to held-out test queries. We formulate non-overlapping and overlapping index construction through a unified question--evidence utility-maximization objective, turning chunking into a supervised combinatorial optimization problem. Under the non-overlap constraint, the objective decomposes over chunks and can be optimized exactly in polynomial time by dynamic programming. When overlap is allowed, the objective becomes monotone submodular; exact optimization is NP-hard, but our g
    
[^27]: 监控准确率并非答案：评估AI智能体重启监控器的决策价值框架

    Monitor Accuracy Isn't the Answer: A Decision-Value Framework for Evaluating AI Agent Restart Monitors

    [https://arxiv.org/abs/2610.36544](https://arxiv.org/abs/2610.36544)

    该论文提出一个决策价值框架，从分析上证明监控器的分类准确率本身无法决定重启AI智能体是否有益，并引入决策充分性准则和有限样本无模型决策审计两种技术，以决策价值的角度评估AI智能体重启监控器。

    

    分类准确率以及真阳性率和假阳性率本身并不能决定重启AI智能体是否能提高成功率。利用一种针对零监控开销的独立干净重启的精确“警报-行动”恒等式，我们从分析上证明，一个固定90%准确率的监控器，仅因替换尝试的剩余预算成功概率不同，就能使成功率从损失五个百分点变为提升三十五个百分点。我们引入了两种以决策术语评估监控器和进度信号的技术。第一，决策充分性准则——Blackwell实验比较理论在二元行动上的特化版本——用于识别哪些进度表示或上下文压缩能够保留在指定行动之间做出最优选择所需的信息，而不依赖于轨迹重建的保真度。第二，有限样本、无模型的决策审计使用同时二项区间（原文在此处截断）……

    arXiv:2610.36544v1 Announce Type: new  Abstract: Classification accuracy and true-positive and false-positive rates alone do not determine whether restarting an AI agent improves success. Using an exact alarm-to-action identity for an independent clean restart with zero monitoring overhead, we show analytically that a monitor with fixed 90% accuracy can swing from a five-percentage-point loss to a 35-point gain in success rate, depending solely on the replacement attempt's residual-budget success probability. We introduce two techniques for evaluating monitors and progress signals in decision terms. First, a decision-sufficiency criterion, a binary-action specialization of Blackwell's comparison of experiments, identifies which progress representations or context compressions preserve the information needed to choose optimally between the specified actions, independent of trace reconstruction fidelity. Second, a finite-sample, model-free decision audit uses simultaneous binomial interv
    
[^28]: ACOR：面向实时DNA存储重建的增量证据积累方法

    ACOR: incremental evidence accumulation towards real-time DNA storage reconstruction

    [https://arxiv.org/abs/2610.36500](https://arxiv.org/abs/2610.36500)

    ACOR是一种面向DNA存储的流式共识重建方法，通过在持久账本中增量记录k-mer转换与边界支持证据，并利用长度约束束搜索进行解码，实现了与测序并行的实时序列重建，恢复效果接近完全且具有竞争力。

    

    DNA通过将数字信息编码到合成链中并通过测序进行检索，可提供高密度、持久的存储方式。序列重建是在下游解码之前，将每条链的多个含噪声副本组合起来的过程。传统的离线方法通常只重建已完成的读段集群，将重建过程与正在进行的读段收集分离开来。流式重建则提供了一种将证据处理与测序过程并行重叠的途径，从而减少下游解码前剩余的计算量。我们提出了ACOR，一种流式共识重建方法，它在持久账本中记录k-mer转换和边界支持信息，然后通过长度约束的束搜索以及集群闭合时基于优先级的选择来完成解码。在真实和模拟DNA存储数据集上的评估显示，ACOR具有有竞争力的恢复效果。在我们自己的短读长下一代测序（NGS）湿实验文库上，ACOR在实现接近完全恢复的同时，以第二次……

    arXiv:2610.36500v1 Announce Type: new  Abstract: DNA offers high-density, durable storage by encoding digital information into synthetic strands and retrieving it through sequencing. Sequence reconstruction combines noisy copies of each strand before downstream decoding. Conventional offline methods typically reconstruct completed read clusters, separating reconstruction from ongoing read collection. Streaming reconstruction offers a way to overlap evidence processing with sequencing and reduce the computation remaining before downstream decoding. We present ACOR, a streaming consensus reconstruction method that records $k$-mer transitions and boundary support in a persistent ledger, then decodes through length-constrained beam search and priority-based selection at cluster closure. Evaluation on real and simulated DNA-storage datasets showed competitive recovery. On our own short-read next-generation sequencing (NGS) wet-lab library, ACOR combined near-complete recovery with second-sc
    
[^29]: 每日物品嵌入的跨日兼容性：对齐、尺度管理与CTR适配的实证研究

    Cross-Day Compatibility of Daily Item Embeddings: An Empirical Study of Alignment, Scale Management, and CTR Adaptation

    [https://arxiv.org/abs/2610.36408](https://arxiv.org/abs/2610.36408)

    该实证研究系统比较了48种配置下每日物品嵌入的跨日兼容性管理方法，发现ID初始化与正交对齐等上游干预能显著提升嵌入对齐质量（余弦相似度最高提升93.4%）并改善多日CTR模型性能（GAUC最高提升2.6%）。

    

    每日物品共现嵌入与次日行为进行配对，且不存在未来行为泄露。因此，多日点击率（CTR）训练会混合同一物品在方向和范数上可能存在差异的多个嵌入版本。我们在初始化、继承、后处理、训练约束以及下游增强等方面，比较了Word2Vec与基于物品图的XSimGCL适配方法。实验库存共包含48种配置，其中32种为通用协议下的CTR评估。观测指标以相对对应Word2Vec基线（0%）的百分比变化形式报告，原始数值与确切数据集规模不予公开。研究得出三项发现。首先，多种上游干预措施能够同时提升兼容性与CTR表现。相对于普通Word2Vec，ID初始化和正交对齐使余弦相似度分别提升51.2%和93.4%，GAUC分别提升2.2%和2.6%。作为下游替代指标，检索与几何特性仍然不够充分。其次，热启动的表现……（原文摘要至此处被截断）

    arXiv:2610.36408v1 Announce Type: new  Abstract: Daily item co-occurrence embeddings are paired with next-day behavior without future-behavior leakage. Multi-day click-through rate (CTR) training therefore mixes versions whose directions and norms may differ for the same item. We compare Word2Vec and an item-graph XSimGCL adaptation across initialization, inheritance, post-processing, training constraints, and downstream augmentation. The inventory contains 48 configurations, including 32 common-protocol CTR evaluations. Observed metrics are reported as relative percentage changes from the corresponding Word2Vec baseline (0%), with raw values and exact dataset sizes withheld. Three findings emerge. First, multiple upstream interventions improve compatibility and CTR. ID-init and orthogonal alignment increase cosine by 51.2% and 93.4%, and GAUC by 2.2% and 2.6%, relative to ordinary Word2Vec. Retrieval and geometry remain insufficient downstream surrogates. Second, warm starts perform s
    
[^30]: MAREval：一个用于评估自然语言推荐解释的多智能体框架

    MAREval: A Multi-Agent Framework for Evaluating Natural Language Recommendation Explanations

    [https://arxiv.org/abs/2610.36389](https://arxiv.org/abs/2610.36389)

    MAREval提出了一种结构化多智能体框架，结合辩论链提示策略、主持人调节、仲裁者聚合与蒙特卡洛采样，实现了对推荐系统中自然语言解释的可扩展、一致且自动化的评估。

    

    arXiv:2610.36389v1 公告类型：新 摘要：评估推荐系统中的自然语言解释对于培养用户信任、透明度和参与度至关重要。然而，现有的评估方法（如人工评估）虽然准确，但资源消耗大，在现代推荐平台所需的规模下并不实用。此外，使用单智能体大语言模型的自动化方法存在提示词敏感性和输出不一致的问题。为应对这些挑战，我们提出了MAREval，一个使用大语言模型评估推荐解释的结构化多智能体框架。MAREval协调以下组件：（i）一个规划者智能体，采用新颖的辩论链提示策略来协调智能体角色并确保评估计划逻辑一致；（ii）一个主持人智能体，通过缓解提示漂移来调节讨论；（iii）一个仲裁者智能体，聚合多轮评估的输出；以及（iv）一种蒙特卡洛采样方法。

    arXiv:2610.36389v1 Announce Type: new  Abstract: Evaluating natural language explanations in recommender systems is essential for fostering user trust, transparency, and engagement. However, existing evaluation approaches like human evaluations, while accurate, are resource-intensive and impractical at the scale required by modern recommendation platforms. Also, automated methods using single-agent LLMs suffer from prompt sensitivity and inconsistent outputs. To address these challenges, we propose MAREval, a structured multi-agent framework for evaluating recommendation explanations using large language models. MAREval orchestrates (i) a planner agent that uses a novel Chain of Debate (CoD) prompting strategy to coordinate agent roles and enforce logically consistent evaluation plans; (ii) a moderator agent that regulates discussions by mitigating prompt drift; and (iii) an arbitrator agent that aggregates outputs from multiple evaluation rounds and (iv) a Monte Carlo sampling method,
    
[^31]: 从生成到评分：在LinkedIn扩展基于大语言模型的语义标注

    From Generation to Scoring: Scaling LLM-Based Semantic Labeling at LinkedIn

    [https://arxiv.org/abs/2610.36383](https://arxiv.org/abs/2610.36383)

    该论文提出将基于大语言模型的语义标注从生成式答案列表重构为语义评分方法，通过单候选提示微调和多项评分技术，在LinkedIn的实际部署中实现了更高效、更稳健且对候选顺序不敏感的标注方案。

    

    出于效率考虑，工业界的大型语言模型（LLM）标注器通常将多个候选标签打包到同一个提示中，并为每个候选生成一个是/否的答案，即使每个决策仅取决于实体和单个候选标签。将这些相互独立的决策作为一个答案列表生成会使它们相互耦合，把解码计算花费在列表上，并依赖列表格式的正确性。本文提出了一种端到端的方案，将标注器重新构建为语义评分，通过在单候选提示上进行微调，使其在仅预填充过程中将每个决策返回为一个答案token分数。我们还改进了多项评分方法，在一个序列中对数百个候选进行评分，并在候选之间屏蔽注意力。该方法已部署于LinkedIn的个人资料资格标注任务，当候选顺序被打乱时，语义评分最多仅改变0.01%的决策，其与参考标签的一致性达到打包生成方法需要约4.7倍计算量才能实现的水平。

    arXiv:2610.36383v1 Announce Type: cross  Abstract: For efficiency, industrial large language model (LLM) labelers often pack many candidate labels into one prompt and generate a YES/NO answer for each, even when each decision depends only on the entity and one candidate. Generating these independent decisions as one answer list couples them, spends decoding on the list, and relies on the list being well-formed. In this paper, we present an end-to-end recipe that reformulates the labeler as semantic scoring, fine-tuning it on single-candidate prompts to return each decision as one answer-token score from a prefill-only pass. We also adapt Multiple-Item Scoring (MIS) to score hundreds of candidates in one sequence with attention masked between them. Deployed for profile qualification labeling at LinkedIn, semantic scoring changes at most 0.01% of decisions when candidates are reordered and matches the agreement with reference labels that packed generation reaches only with about 4.7x as 
    
[^32]: 在存储预算下保持稀有物品召回率：量化、蒸馏与教师保真度的局限

    Preserving rare-item recall under a storage budget: quantization, distillation, and the limits of teacher fidelity

    [https://arxiv.org/abs/2610.36267](https://arxiv.org/abs/2610.36267)

    在每行168字节的固定存储预算下，对300维教师模型评分进行秩300的4位低精度分解仅需教师存储的13%，即可在全部七个数据集-骨干组合中保留教师95-103%的稀有物品召回率，优于小型全精度模型和蒸馏学生模型，且教师保真度并不可靠地反映稀有物品召回的保留情况。

    

    在固定的嵌入存储预算下，推荐系统可以使用小型全精度模型、蒸馏学生模型，或更高秩的低精度模型。我们在七个数据集-骨干组合中研究了哪种选择能够保持稀有物品的召回率，以及教师保真度能否反映这一点：这七个组合包括在Gowalla、Last.fm、Amazon Video Games、Steam、Goodbooks-10k和MovieLens-20M上的矩阵分解，以及在Amazon Video Games上的LightGCN。七个候选方案均符合每用户或物品行168字节的存储限制：40维贝叶斯个性化排序（BPR）、两个40维无偏知识蒸馏（UnKD）学生模型，以及对一个300维教师模型评分在秩40、80、155和300处进行分解并以32、16、8和4位存储的方案。在确定候选方案并完成验证性分析后，每个测试集仅评估一次。结果表明，秩300的4位分解仅使用教师模型13%的存储，就在所有组合中保留了教师模型95-103%的尾部recall@20和93-97%的NDCG@20。秩40的分解……

    arXiv:2610.36267v1 Announce Type: new  Abstract: Under a fixed embedding-storage budget, recommenders can use a small full-precision model, a distilled student, or a higher-rank low-precision model. We asked which choice preserves rare-item recall and whether teacher fidelity reflects it in seven dataset-backbone cells: matrix factorization on Gowalla, Last.fm, Amazon Video Games, Steam, Goodbooks-10k and MovieLens-20M, and LightGCN on Amazon Video Games. Seven candidates fit in 168 bytes per user or item row: 40-dimensional Bayesian personalized ranking (BPR), two 40-dimensional unbiased knowledge distillation (UnKD) students, and factorizations of a 300-dimensional teacher's scores at ranks 40, 80, 155 and 300, stored in 32, 16, 8 and 4 bits. Each test set was evaluated once after fixing candidates and confirmatory analyses. The rank-300 4-bit factorization used 13% of teacher storage and retained 95-103% of its tail recall@20 and 93-97% of its NDCG@20 in every cell. The rank-40 fact
    
[^33]: BehaveRec：长期行为历史何时有助于序列推荐？——基于会话、历史与社区信号的研究

    BehaveRec: When Does Long-Term Behavioral History Help Sequential Recommendation? A Study with Session, History, and Community Signals

    [https://arxiv.org/abs/2610.36239](https://arxiv.org/abs/2610.36239)

    提出三信号框架BehaveRec，通过可学习门控融合短期会话、90天长期用户历史和基于行为嵌入相似度的社区信号，在1.37亿条序列的工业数据集上超越了更大容量的SASRec基线，并系统研究了长期历史信号何时真正对推荐有效。

    

    序列推荐模型通常根据用户近期的会话交互来预测其下一个动作，这在短期信号较弱或模糊时会限制个性化效果。我们并不假设引入更长期的历史和社区信号总是有益，而是研究它们何时真正有效。我们提出了BehaveRec，这是一个三信号框架，结合了因果会话transformer、覆盖90天窗口的双向用户历史编码器，以及在用户-用户图上运行的GraphSAGE模块（该图基于行为嵌入相似度而非物品共现构建）；一个可学习的门控机制为每个用户自适应融合这三种信号。在一个包含1260万用户的1.37亿条序列的工业数据集上，BehaveRec将R@5相比SASRec提升了3.2%，而一个容量更大的SASRec基线（参数量多于BehaveRec）仍然落后2.5%，排除了模型容量作为解释。更重要的是，按历史长度进行的用户级分层分析揭示（摘要在此处截断）

    arXiv:2610.36239v1 Announce Type: new  Abstract: Sequential recommendation models typically predict a user's next action from recent session interactions, which limits personalization when short-term signals are weak or ambiguous. Rather than assuming that incorporating longer-term history and community signals is universally beneficial, we study when it actually helps. We propose BehaveRec, a three-signal framework combining a causal session transformer, a bidirectional User History Encoder over a 90-day window, and a GraphSAGE module over a user-user graph built from behavioral embedding similarity rather than item co-occurrence; a learned gate blends the three signals per user. On an industrial dataset of 137M sequences from 12.6M users, BehaveRec improves R@5 by +3.2% over SASRec, with a larger-capacity SASRec baseline (more parameters than BehaveRec) still trailing by +2.5%, ruling out capacity as the explanation. More importantly, a user-level stratification by history length rev
    
[^34]: 反事实用户画像扰动用于推荐解释：一项可复现性研究

    Counterfactual Profile Perturbations for Recommendation Explanations: A Reproducibility Study

    [https://arxiv.org/abs/2610.36208](https://arxiv.org/abs/2610.36208)

    本研究复现并扩展了LXR反事实推荐解释方法，证明其不仅对最高推荐项目有效，对其他项目同样有效，同时发现项目流行度等简单基线在某些任务中表现得出人意料地好。

    

    这篇可复现性论文探索了一种被称为“学习解释推荐”（LXR）的反事实解释技术。先前的研究表明，LXR能够可靠地识别出一些扰动，如果将这些扰动应用于用户的画像，会导致用户排名最高的项目不再被推荐，从而将项目的推荐解释为用户评分选择的函数。在本文中，我们以多种方式扩展了这项工作。我们将原始LXR研究的范围扩展到查看除最高推荐项目之外的其他项目的解释，发现LXR在这一版本的任务中同样有效。我们考虑了LXR的替代基线，包括项目流行度和潜在因子相似度，发现对于某些任务，这些替代方案的表现出人意料地好。我们还研究了项目流行度的影响，发现当推荐的项目不那么流行时，LXR能提供更好的解释质量（即更短的扰动）……

    arXiv:2610.36208v1 Announce Type: cross  Abstract: This reproducibility paper explores the counterfactual explanation technique known as Learning to eXplain Recommendations (LXR). Prior work has shown that LXR can reliably identify perturbations that, if applied to a user's profile, cause the user's top ranked item to no longer be recommended, thereby explaining the item's recommendation as a function of the user's rating choices. In this paper, we extend this work in multiple ways. We expand the scope of the original LXR study to look at explanations for items other than the most highly recommended item, finding that LXR is also effective at this version of the task. We consider alternate baselines for LXR, including item popularity and latent factor similarity, and we see that for some tasks these alternatives work surprisingly well. We also examine the impact of item popularity, finding that LXR offers better explanation quality (shorter perturbations) when less popular items are be
    
[^35]: 生成式AI使用中的教育差距是否小于互联网在相似使用水平时的差距？来自16个欧洲国家的证据

    Is the education gap in generative AI use smaller than the internet's was at a similar level of use? Evidence from 16 European countries

    [https://arxiv.org/abs/2610.36181](https://arxiv.org/abs/2610.36181)

    该研究通过将16个欧洲国家2025年的生成式AI使用率与互联网使用率相当的年份匹配对比，发现生成式AI使用中的教育差距在大多数国家（13/16）小于互联网普及早期在相似使用水平时的差距，表明生成式AI的普及在教育维度上比互联网初期更为平等。

    

    新技术通常首先被受过更多正规教育的人所采用。本文探讨的问题是：2025年生成式AI使用中的教育差距，与同一国家在整体互联网使用处于相似水平的那一年互联网的教育差距相比，是更小还是更大。研究使用了欧盟统计局关于16至74岁个体信息通信技术（ICT）使用的公开调查数据。将每个国家2025年过去三个月的生成式AI使用率，与该国整体互联网过去三个月使用率最接近（差距在5个百分点以内）的年份进行匹配，并以对数比率和百分点差异来衡量高学历（ISCED 5至8）与低学历（ISCED 0至2）人群之间的差距。35个国家中有16个可以匹配，其余大多数属于互联网的早期采用国家。在匹配的国家中，按对数比率计算，16个国家中有13个国家的生成式AI教育差距更小（中位数变化为-0.625，95%自助法置信区间为-0.969至-0.291）。

    arXiv:2610.36181v1 Announce Type: new  Abstract: New technologies are usually adopted first by people with more formal education. This paper asks whether the education gap in generative AI use in 2025 is smaller or larger than the internet's gap in the same country in the year when overall internet use was at a similar level. It uses public Eurostat survey data on ICT usage by individuals aged 16 to 74. Each country's 2025 rate of generative AI use in the last three months is matched to the internet year whose overall last-three-months use was closest, within 5 percentage points, and the gap between people with high (ISCED 5 to 8) and low (ISCED 0 to 2) completed education is measured as a log ratio and as a percentage-point difference. Sixteen of 35 countries could be matched; most of the others were early internet adopters. In the matched countries, the generative AI gap was smaller in 13 of 16 countries on the log ratio (median change -0.625, 95% bootstrap interval -0.969 to -0.291,
    
[^36]: 当连通性产生误导：利用对比证据路径引导智能体搜索

    When Connectivity Misleads: Guiding Agentic Search with Contrastive Evidence Paths

    [https://arxiv.org/abs/2610.36134](https://arxiv.org/abs/2610.36134)

    提出对比证据路径（CEP）方法，通过在查询条件化的文档图上依据最强单一路径评分，并利用背景访问分布进行校准来消除高连通性文档带来的结构偏差，从而引导搜索智能体发现真正有价值的证据而非高度连接的干扰节点。

    

    搜索智能体可以利用跨文档图通过多跳连接发现信息来源，并在有限的观察窗口内根据候选文档的排名和关系来选择后续行动。然而，某些文档之所以排名靠前，可能仅仅是因为其大量的连接使其更容易被访问到，而不一定是因为它们提供了更有价值的证据。诸如个性化 PageRank（PPR）等图扩散方法会聚合来自多个来源的传播贡献，导致高度连接的干扰节点挤占观察窗口。此外，扁平化的节点评分还会掩盖跨文档的证据链路，限制了持续探索。我们提出对比证据路径（CEP），该方法在查询条件化的文档图上根据候选文档的最强单一路径进行评分，并将这些评分与同一图的背景访问分布进行校准，以缓解结构偏差。CEP 根据校准后的分数对候选文档进行排序。

    arXiv:2610.36134v1 Announce Type: cross  Abstract: Search agents can use cross-document graphs to discover sources through multi-hop connections and choose subsequent actions based on candidate rankings and relationships within a limited observation window. However, documents may rank highly because their numerous connections make them easier to reach, without necessarily providing more valuable evidence. Graph diffusion methods such as Personalized PageRank (PPR) aggregate propagation contributions from multiple sources, allowing highly connected distractor nodes to crowd the observation window. Flat node scores also obscure cross-document evidence trails, limiting continued exploration. We propose Contrastive Evidence Paths (CEP), which scores candidates by their strongest individual paths on a query-conditioned document graph and calibrates these scores against the same graph's background visitation distribution to mitigate structural bias. CEP ranks candidates by the calibrated sco
    
[^37]: RELAY：一次查询，两种角色，用于序列推荐

    RELAY: One Query, Two Roles for Sequential Recommendation

    [https://arxiv.org/abs/2610.36000](https://arxiv.org/abs/2610.36000)

    RELAY让单个查询同时承担语义偏好表示与行为记忆路由两种角色，并通过候选条件化门控融合两类证据，在单次前向传播中以低成本实现六个数据集上全面领先的推荐精度。

    

    序列推荐必须将语义偏好与行为关联相结合，才能在低推理成本下提供准确的排序结果。共享历史选择将偏好表示直接与候选打分中使用的行为证据联系起来。我们提出了RELAY，一个围绕“一次查询、两种角色”构建的紧凑序列推荐模型：语义读出与行为记忆路由。单个可学习的查询既形成语义偏好表示，又复用其注意力将直接历史证据与转移证据路由到候选打分中。候选条件化的门控机制为每个候选项校准这两个记忆通道。RELAY使用单一目录似然对两种角色进行联合训练，并在一次前向传播中生成最终排序。在六个数据集上的全目录实验表明，RELAY在所有数据集上均取得了对比方法中最高的Recall@5和NDCG@5，相比此前最佳模型分别提升了14.22%和17...%。

    arXiv:2610.36000v1 Announce Type: new  Abstract: Sequential recommendation must combine semantic preferences with behavioral associations to deliver accurate rankings at low inference cost. Sharing history selection connects preference representation directly to the behavioral evidence used in candidate scoring. We present RELAY, a compact sequential recommender built around one query, two roles: semantic readout and behavioral-memory routing. A single learned query forms a semantic preference representation and reuses its attention to route direct-history and transition evidence into candidate scores. Candidate-conditioned gates calibrate the two memory channels for each candidate. RELAY trains both roles with a single catalog likelihood and produces the final ranking in one forward pass. Full-catalog experiments on six datasets show that RELAY achieves the highest Recall@5 and NDCG@5 among the compared methods on every dataset, improving over the previous best model by 14.22% and 17.
    
[^38]: 超图增强的长上下文多模态文档问答证据选择

    Hypergraph-Enhanced Evidence Selection for Long-Context Multimodal Document QA

    [https://arxiv.org/abs/2610.35542](https://arxiv.org/abs/2610.35542)

    提出HyperDoc框架，通过构建文档超图显式捕获结构、逻辑与跨模态关系，在有限输入预算下实现预算感知的多模态证据选择，从而提升长上下文多模态文档问答的性能。

    

    长上下文多模态文档问答已成为现代视觉语言模型（VLM）一项重要但具有挑战性的任务，因为它需要在有限的输入预算内，对分散在不同页面和不同模态中的证据进行推理。当前的流程采用检索增强生成（RAG）来为下游的视觉语言模型选择多模态证据。然而，现有系统主要依赖于基于固定粒度检索单元（如页面或文本块）的静态选择规则，导致检索过程中的信息丢失，并使下游推理性能不理想。为缓解这些问题，我们将核心任务形式化为对结构化文档证据的预算感知证据选择，并提出了HyperDoc——一个文档RAG框架，它在保留文档关系的同时构建紧凑且充分的证据。具体而言，HyperDoc构建了一个文档超图，以显式捕获结构、逻辑以及跨模态关系。

    arXiv:2610.35542v1 Announce Type: new  Abstract: Long-context multimodal document question answering (QA) has become an important yet challenging task for modern vision-language models (VLMs), as it requires reasoning over evidence dispersed across pages and modalities within limited input budgets. Current pipelines adopt retrieval-augmented generation (RAG) to select multimodal evidence for downstream VLMs. However, existing systems mainly rely on static selection rules over fixed-granularity retrieval units such as pages or blocks, resulting in information loss during retrieval and suboptimal performance in downstream reasoning. To alleviate these issues, we formulate the core task as budget-aware evidence selection over structured document evidence and propose HyperDoc, a document RAG framework that constructs compact yet sufficient evidence while preserving document relations. Specifically, HyperDoc constructs a document hypergraph to explicitly capture structural, logical, and cro
    
[^39]: 基于有效秩缩放的推荐系统统一异构特征交互

    Unified Heterogeneous Feature Interaction via Effective-Rank Scaling for Recommender Systems

    [https://arxiv.org/abs/2610.35318](https://arxiv.org/abs/2610.35318)

    提出了HiRank框架，通过有效秩缩放在统一主干中联合建模异构特征交互与门控序列注意力，解决了推荐系统中序列与非序列特征信息交换受限以及多场景负迁移的问题。

    

    大规模工业级排序模型必须在严格的在线服务约束下，同时捕获异构特征交互、长期用户行为和多场景依赖关系。现有架构通常使用独立的模块来处理序列特征和非序列特征，限制了双向信息交换。此外，当序列信号和非序列信号表现出不同的有效秩时，直接对所有语义标记应用同构自注意力可能效率低下。同时，共享表征可能会忽略场景特定的模式，并在不同场景之间引发负迁移。在本文中，我们提出了HiRank，这是一个用于多场景推荐的统一框架，通过有效秩缩放联合建模异构特征交互。HiRank在统一的主干网络中将门控序列注意力与异构特征交互相结合，实现双向……（摘要在此处截断）

    arXiv:2610.35318v1 Announce Type: new  Abstract: Large-scale industrial ranking models must jointly capture heterogeneous feature interactions, long-term user behaviors, and multi-scenario dependencies under stringent online serving constraints. Existing architectures typically process sequential and non-sequential features with separate modules, limiting bidirectional information exchange. Directly applying homogeneous self-attention to all semantic tokens can also be inefficient when sequential and non-sequential signals exhibit different effective-ranks (eRanks). Meanwhile, shared representations can overlook scenario-specific patterns and induce negative transfer between scenarios. In this paper, we propose HiRank, a unified framework for multi-scenario recommendation that jointly models Heterogeneous feature Interactions via effective-Rank scaling. HiRank integrates gated sequential attention with heterogeneous feature interaction within a unified backbone, enabling bidirectional 
    
[^40]: 比较生成式AI与机器学习和人工审查在电子证据开示中的应用

    Comparing GenAI to Machine Learning and Manual Review for eDiscovery

    [https://arxiv.org/abs/2610.16925](https://arxiv.org/abs/2610.16925)

    该研究指出此前声称生成式AI审查工具（88%召回率）优于机器学习工具（64%召回率）的结论因方法差异而不可比，实际分析发现生成式AI与人工审查效果相当，而采用持续主动学习（CAL）的机器学习模型仅需2,000条样本训练即可达到最先进效果并具备更高效率。

    

    Keeling等人[Keeling2026Generative]报告了Redgrave LLP最近开展的一项案例研究，他们声称Relativity的生成式AI审查工具获得了88%的召回率，而其机器学习工具Relativity Assisted Learning (RAL)仅达到64%的召回率。然而这些结果并不具有可比性，原因在于所采用的方法目标不同、评估者之间存在分歧，以及它们代表了不同的召回率-精确度权衡和成本-效益权衡。我们对Redgrave数据的分析表明，在考虑这些差异之后，aiR与人工审查的效果是相当的。此外，采用持续主动学习（CAL）的成熟机器学习模型，仅在2,000条初审人员相关性判定样本上进行训练，就能展现出最先进的有效性和更优越的效率。

    arXiv:2610.16925v1 Announce Type: cross  Abstract: Keeling et al. [Keeling2026Generative] report on a case study recently conducted by Redgrave LLP in which they assert that Relativity's Generative AI Review tool (aiR) achieved 88% recall while Relativity Assisted Learning (RAL), its machine-learning tool, achieved 64% recall. These results are not comparable because the methods employed have different objectives, because of inter-assessor disagreement, and because they represent different recall-precision and cost-benefit trade-offs. Our analysis of the Redgrave data indicates that, when these differences are taken into account, aiR and manual review are comparable. Furthermore, established machine-learning models employing Continuous Active Learning (CAL) show state-of-the-art effectiveness, and superior efficiency, when trained only on a sample of 2,000 first-pass-reviewer responsiveness determinations.
    
[^41]: Inherit4Rec：面向推荐模型高效扩展的参数继承方法

    Inherit4Rec: Parameter Inheritance for Efficient Scaling of Recommendation Models

    [https://arxiv.org/abs/2609.23111](https://arxiv.org/abs/2609.23111)

    本文提出Inherit4Rec参数继承框架，同时支持稠密到稠密（D2D）扩展和稠密到稀疏（D2S）转换，无需从头重训即可在动态演化的推荐数据下高效扩展推荐模型容量并避免性能骤降。

    

    扩展模型容量已成为克服工业推荐系统性能瓶颈的有效方法。然而，反复从零开始训练更大的稠密模型需要大量的数据和时间，而其不断增长的计算开销与工业系统严格的线上服务预算相冲突。参数继承为稠密模型的增长和稀疏化转换提供了一条有前景的路径，但现有方法主要面向静态语料库设计，在动态演化的推荐数据下可能出现剧烈的性能下降。为应对这些挑战，我们提出了Inherit4Rec，一个同时支持稠密到稠密（D2D）增长和稠密到稀疏（D2S）转换的参数继承框架。Inherit4Rec-D2D将混合增长与非对称训练相结合，在模型扩展时保持前向函数的稳定性并维持更新的连续性。Inherit4Rec-D2S通过共同激活……（摘要原文在此处截断）

    arXiv:2609.23111v1 Announce Type: new  Abstract: Scaling model capacity has emerged as an effective approach to overcoming performance bottlenecks in industrial recommender systems. However, repeatedly training larger dense models from scratch demands substantial data and time, while their growing computation conflicts with the strict serving budgets of industrial systems. Parameter inheritance provides a promising route for both dense model growth and sparse conversion, yet existing methods are primarily designed for static corpora and can suffer sharp performance drops under dynamically evolving recommendation data. To address these challenges, we propose Inherit4Rec, a parameter-inheritance framework that supports both Dense-to-Dense (D2D) growth and Dense-to-Sparse (D2S) conversion. Inherit4Rec-D2D combines hybrid growth with asymmetric training to preserve the forward function at expansion and maintain update continuity. Inherit4Rec-D2S constructs SMoE networks through co-activati
    
[^42]: 基准雷达：面向AI基准与评估的动态数据库和搜索引擎

    Benchmark Radar: A Living Database and Search Engine for AI Benchmarks and Evaluation

    [https://arxiv.org/abs/2609.11115](https://arxiv.org/abs/2609.11115)

    本文提出Benchmark Radar，一个每日自动发现并整合AI基准论文、数据集和代码的动态数据库与搜索引擎，为LLM评估、智能体、编程、推理和安全等领域提供可检索的基准目录、来源引用和分数历史。

    

    基准研究人员和大语言模型（LLM）及其他AI系统的开发者需要找到相关的评估方法、定位其基准数据集和代码，并理解已报告分数背后的设置。我们提出了Benchmark Radar（基准雷达），这是一个用于检索和发现AI基准的动态数据库和搜索引擎，涵盖LLM评估、智能体和工具使用基准、编程、推理、安全性以及特定领域评估。该系统将每日发现的基准论文、代码仓库、数据集和发布版本与可搜索的基准目录、模型卡和技术报告中的提及以及分数历史相结合。它保留了源身份信息和引用，以便读者可以查看候选基准及其评估证据。每日发现基于37个来源：13个直接连接器和24个第一方研究与工程信息源。该目录包含来自4个基准目录的1,283条源记录

    arXiv:2609.11115v1 Announce Type: cross  Abstract: Benchmark researchers and developers of large language models (LLMs) and other AI systems need to find relevant evaluations, locate their benchmark datasets and code, and understand the settings behind reported scores. We present Benchmark Radar, a living database and search engine for retrieval and discovery of AI benchmarks, covering LLM evaluation, agentic and tool-use benchmarks, coding, reasoning, safety, and domain-specific evaluations. The system combines daily discovery of benchmark papers, repositories, datasets, and releases with a searchable benchmark catalog, mentions in model cards and technical reports, and score histories. It retains source identities and citations so readers can inspect candidate benchmarks and their evaluation evidence. Daily discovery draws on 37 sources: 13 direct connectors and 24 first-party research and engineering feeds. The catalog contains 1,283 source records drawn from 4 benchmark catalogs an
    
[^43]: FKG.in的验证：LLM增强的印度食品知识中的健全性评估

    Validating FKG.in: Soundness Assessment in LLM-Augmented Indian Food Knowledge

    [https://arxiv.org/abs/2608.29249](https://arxiv.org/abs/2608.29249)

    本文作为印度食品知识图谱FKG.in的一部分，提出了一种半自动化的健全性评估工作流程，通过结合形式文法、词汇检查、统计启发式、Set Transformer连贯性建模和检索验证的多阶段方法，识别并解决LLM从非正式烹饪来源提取和增强结构化食谱数据时的常见失败模式。

    

    在线烹饪生态系统中，由大型语言模型（LLM）生成、修改或总结的食谱内容日益增多。虽然这些输出通常看似合理，但可能包含虚构的食材、被误述的用量或文化上不合常理的食材组合，从而限制了其在下游应用和知识图谱构建中的适用性。在本文中，我们提出了一种半自动化的健全性评估工作流程，用于验证由LLM从非正式烹饪来源中提取和增强的结构化食谱数据。该流程作为印度食品知识图谱FKG.in的一部分开发而成，通过结合形式文法、基于词汇的检查、统计启发式方法、基于Set Transformer的连贯性建模以及基于检索的验证等多阶段流程，识别并解决常见的失败模式，包括结构性不一致、语义和逻辑上的不连贯以及与源文本的偏差。

    arXiv:2608.29249v1 Announce Type: new  Abstract: The online culinary ecosystem is increasingly populated by recipe content generated, modified, or summarized by Large Language Models (LLMs). While often plausible, such outputs may contain hallucinated ingredients, misrepresented quantities, or culturally implausible combinations, limiting their suitability for downstream applications and knowledge graph construction. In this paper, we present a semi-automated soundness assessment workflow for validating structured recipe data extracted and augmented by LLMs from informal culinary sources. Developed as part of FKG.in, a knowledge graph of Indian food, the pipeline identifies and addresses common failure modes, including structural inconsistencies, semantic and logical incoherence, and deviations from the source text, through a multi-stage process combining formal grammars, vocabulary-based checks, statistical heuristics, Set Transformer-based coherence modeling, and retrieval-based veri
    
[^44]: 将BrowseComp-Plus映射到ClimbMix：朝向更真实的智能体搜索语料库

    Projecting BrowseComp-Plus onto ClimbMix: Toward More Realistic Corpora for Agentic Search

    [https://arxiv.org/abs/2608.20317](https://arxiv.org/abs/2608.20317)

    本文提出了一种投影流水线，将BrowseComp-Plus基准的问题迁移到大规模真实网页语料库ClimbMix中，从而构建更真实的智能体搜索评估环境，同时保持问题质量。

    

    BrowseComp-Plus基准通过用固定语料库替换不透明的网络搜索，解耦了智能体搜索的评估，从而可以将智能体的角色与检索器的角色分离。然而，该语料库仅包含约10万篇文档，且是由基准自身查询的支持文档加上挖掘的困难负样本组成的，因此证据和干扰项都是针对每个查询选择的。我们引入了$\text{BrowseComp-Plus}_{\text{CM}}$，它保留了BrowseComp-Plus的问题，但将其证据迁移到ClimbMix——一个由NVIDIA发布的、用于预训练语言模型的400B词元、553M文档的网页文本混合语料库，该语料库的构建未参考任何基准。我们的主要贡献是实现这一迁移的投影流水线：它将每个问题分解为原子推理步骤，并将每个步骤在新语料库中进行定位，仅当自动验证、独立智能体和人工检查（h）均通过时才保留问题。

    arXiv:2608.20317v1 Announce Type: new  Abstract: The BrowseComp-Plus benchmark disentangled the evaluation of agentic search by replacing opaque web search with a fixed corpus, so that an agent's role can be separated from the retriever's. That corpus, however, holds only about 100K documents and was assembled from the supporting documents of the benchmark's own queries plus mined hard negatives, so the evidence and the distractors were both selected per query. We introduce $\text{BrowseComp-Plus}_{\text{CM}}$, which keeps the BrowseComp-Plus questions but relocates their evidence to ClimbMix, a 400B-token, 553M-document mixture of web text released by NVIDIA for pre-training language models and built without reference to any benchmark. Our main contribution is the projection pipeline that makes this possible: it decomposes each question into atomic reasoning hops and grounds every hop in the new corpus, retaining a question only when automatic verification, an independent agent, and h
    
[^45]: 预训练音频表征何时有助于音乐推荐？跨排序与生成式界面的研究

    When Do Pretrained Audio Representations Help Music Recommendation? A Study Across Ranking and Generative Interfaces

    [https://arxiv.org/abs/2608.06928](https://arxiv.org/abs/2608.06928)

    该研究发现预训练音频表征对音乐推荐的价值高度依赖于推荐界面——音乐预训练和音频-文本对齐编码器在基于内容检索和语义ID生成式推荐中带来明显增益，但在SASRec序列推荐中经交互适配后优势大幅缩小。

    

    预训练音频编码器正越来越多地被用于为音乐推荐提供内容表征，但其实际下游价值会随所使用的推荐界面不同而发生显著变化。我们在六个预训练音频编码器和三种代表性场景中研究了这一效应：基于内容的检索、序列推荐以及基于语义ID（Semantic-ID）的生成式推荐。通过使用匹配的随机特征对照和仅ID对照，我们将绝对推荐性能与相对于无内容输入的提升区分开来。在两个音乐数据集和全目录评估中，音乐预训练和音频-文本对齐的编码器在直接使用其预训练几何结构或将其转换为语义ID时通常表现更强，而在SASRec中经过基于交互的适配后，这些差异变得小得多。我们进一步研究了在生成式推荐中应如何对音频表征进行离散化……

    arXiv:2608.06928v3 Announce Type: replace  Abstract: Pretrained audio encoders are increasingly used to provide content representations for music recommendation, but their downstream value can change substantially with the recommendation interface through which they are used. We study this effect across six pretrained audio encoders and three representative settings: content-based retrieval, sequential recommendation, and Semantic-ID-based generative recommendation. Using matched random-feature and ID-only controls, we distinguish absolute recommendation performance from gains over content-free inputs. Across two music datasets and a full-catalog evaluation, music-pretrained and audio-text-aligned encoders are generally stronger when their pretrained geometry is used directly or converted into Semantic IDs, while these differences become much smaller after interaction-based adaptation in SASRec. We further examine how audio representations should be discretized for generative recommend
    
[^46]: TokenMinds：面向大型推荐系统用户理解的预训练用户Token与嵌入

    TokenMinds: Pretrained User Tokens and Embeddings for User Understanding in Large Recommender Systems

    [https://arxiv.org/abs/2606.25147](https://arxiv.org/abs/2606.25147)

    TokenMinds基于预训练LLM改造的编码器-解码器架构并采用双输出设计，将PLUM框架从基于SID的物品检索扩展到生成式序列用户建模，同时产出稠密用户嵌入与离散用户Token，突破了固定维度稠密嵌入的表征限制并捕捉用户行为的序列模式。

    

    工业推荐系统中的用户建模通常产生稠密嵌入，但由于固定维度向量的限制，其表征能力受到约束。作为离散用户表示的一种新兴替代方案，由LLM生成的自然语言用户画像往往捕获的是语言语义，而非用户行为中的序列模式。同时，语义ID（SID）虽然对物品表示很有效，但现有的离散用户表示大多仍局限于量化已池化的用户向量、对历史观看行为进行聚类，或仅为候选检索预测粗粒度的主题聚类。我们提出TokenMinds——一个将PLUM框架从基于SID的物品检索扩展到生成式序列用户建模的工业系统。TokenMinds构建于由预训练LLM改造而来的编码器-解码器架构之上，引入了双输出设计：编码器输出一种稠密用户嵌入，用于捕获……（原文摘要在此处截断）

    arXiv:2606.25147v2 Announce Type: replace-cross  Abstract: User modeling in industrial recommender systems typically produces dense embeddings, which suffer from representational constraints due to fixed-dimensional vectors. As an emerging alternative for discrete user representation, LLM-generated natural-language user profiles often capture language semantics rather than sequential patterns in user behaviors. Meanwhile, Semantic ID (SID) is effective for item representation, yet existing discrete user representations remain largely confined to quantizing already-pooled user vectors, clustering historical watches, or predicting coarse topic clusters solely for candidate retrieval. We propose TokenMinds, an industrial system that extends the PLUM framework from SID-based item retrieval to generative sequential user modeling. Built on an encoder-decoder architecture adapted from pretrained LLMs, TokenMinds introduces a dual-output design: the encoder outputs a dense user embedding captu
    
[^47]: LRanker：通过测试时扩展将大语言模型排序扩展到数百万候选

    LRanker: Scaling LLM Ranking to Millions of Candidates with Test-Time Scaling

    [https://arxiv.org/abs/2605.27810](https://arxiv.org/abs/2605.27810)

    LRanker通过K-means候选聚合编码器与基于图的测试时扩展机制，将LLM排序高效扩展至数百万候选规模并提升排序质量。

    

    大语言模型（LLM）为排序任务提供了强大的语义表示能力，但当候选池包含数百万个项目时，在有限的上下文中处理候选文本、并反复调用LLM进行候选级别的比较，其代价可能十分高昂。我们提出了LRanker，一个通过测试时扩展将LLM排序扩展到数百万候选的框架。该框架结合了两个互补的组件：候选聚合编码器利用K-means聚类对候选嵌入进行离线汇总，并以紧凑的全局候选信息作为条件来引导LLM；随后排序通过缓存候选嵌入的内积分数来完成。基于图的测试时扩展机制将额外的推理计算分配给以分区为条件的查询嵌入，并对它们的排序分数进行集成。这将处理大规模候选池的评分能力与利用额外测试时计算提升排序质量分离开来。

    arXiv:2605.27810v2 Announce Type: replace  Abstract: Large language models (LLMs) offer strong semantic representations for ranking, but processing candidate texts within limited contexts and repeatedly invoking an LLM for candidate-level comparisons can be costly when candidate pools contain millions of items. We propose LRanker, a framework for scaling LLM ranking to millions of candidates with test-time scaling. The framework combines two complementary components. A candidate aggregation encoder summarizes offline candidate embeddings using K-means clustering and conditions the LLM on compact global candidate information; ranking then uses inner-product scores with cached candidate embeddings. A graph-based test-time scaling mechanism allocates additional inference computation to partition-conditioned query embeddings and ensembles their ranking scores. This separates the ability to score large candidate pools from the use of additional test-time computation to improve ranking quali
    
[^48]: DeGenTWeb：首次审视以机器生成文本为主的网站

    DeGenTWeb: A First Look at MGT-dominant Websites

    [https://arxiv.org/abs/2605.00087](https://arxiv.org/abs/2605.00087)

    提出了 DeGenTWeb 框架，通过将机器生成文本检测器适配到网页并聚合网站级检测结果，首次系统识别出以机器生成内容为主的网站，发现 Common Crawl 存档网站中 20.9% 和 Bing 搜索结果链接网站中 13.6% 属于机器生成文本主导型。

    

    许多新闻报道声称，来自大语言模型（LLM）的机器生成文本（MGT）正在接管网络。然而，这些说法通常并非基于具有代表性的网络样本，且其背后的方法往往不透明。因此，我们目前缺乏对网络上机器生成文本真实普及程度和特征的了解。我们提出了 DeGenTWeb，一个系统性识别“机器生成文本主导型网站”的方法，即那些信息性散文内容大部分由机器生成、几乎不包含人类输入的网站。我们展示了如何将机器生成文本检测器适配到网页上使用，以及如何汇总来自同一网站多个页面的检测结果，以实现准确的网站级分类。DeGenTWeb 仅将 0.12% 的人类撰写网站误标为机器生成文本主导型网站。我们发现，Common Crawl 在2026年1月至7月期间存档的网站中有 20.9%，以及2025年 Bing 搜索结果所链接的网站中有 13.6% 属于机器生成文本主导型网站。

    arXiv:2605.00087v2 Announce Type: replace-cross  Abstract: Many news reports have claimed that machine-generated text (MGT) from large language models (LLMs) is taking over the web. However, these claims are typically not based on a representative sample of the web, and the methodology underlying them is often opaque. Consequently, we lack an understanding of the true prevalence and characteristics of MGT on the web.   We describe DeGenTWeb, which systematically identifies MGT-dominant websites: sites whose informational prose content is mostly MGT, with little human input. We show how to adapt detectors of MGT for use on web pages, and how to aggregate detection results from multiple pages on a site for accurate site-level categorization. DeGenTWeb mislabels only 0.12% of human-authored sites as MGT-dominant. We find that 20.9% of the sites archived by Common Crawl between January and July 2026 and 13.6% of the sites linked from Bing's search results in 2025 are MGT-dominant. These fi
    
[^49]: 评分规则对大语言模型评估的敏感性：来自印度金融监管文本的证据

    Scoring-Rule Sensitivity in LLM Evaluation: Evidence from Indian Financial Regulatory Text

    [https://arxiv.org/abs/2604.19298](https://arxiv.org/abs/2604.19298)

    该研究构建了基于印度SEBI和RBI监管文本的专家标注基准IndiaFinBench，发现大语言模型的排名对评分规则高度敏感——严格字符串匹配与跨模型裁判两种评分机制之间甚至无显著的正向排名相关性——表明单一准确率数字是模型与评分规则的共同产物，不能可靠反映模型真实能力。

    

    一项基准排行榜报告的是单一的准确率数字，但这个数字是模型与评分规则的共同产物。我们探究当仅改变评分规则时会发生什么，并发现模型的排名对评分规则高度敏感。IndiaFinBench 是一个包含406个条目、由专家标注的基准，基于SEBI（印度证券交易委员会）和RBI（印度储备银行）的监管文本，涵盖监管解读、数值推理、矛盾检测和时间推理。我们在两种主要评分机制下对十二个大语言模型进行零样本评分——严格的四阶段字符串匹配，以及与任何被评估系统无共享模型家族的跨模型裁判——外加一种操作性复合机制。两种主要机制之间没有正向的排名对应关系（Spearman rho = -0.273，p = 0.39，不显著），且没有任何模型能在所有三种机制下保持其排名。严格准确率范围为75.12%–89.66%；仅裁判评分的准确率范围为87.19%–94.83%，且出现了不同的领先者

    arXiv:2604.19298v3 Announce Type: replace-cross  Abstract: A benchmark leaderboard reports one accuracy number, but that number is a joint product of a model and a scoring rule. We ask what changes when only the rule changes, and find that model ordering is highly sensitive to it. IndiaFinBench is a 406-item expert-annotated benchmark over SEBI and RBI regulatory text covering regulatory interpretation, numerical reasoning, contradiction detection, and temporal reasoning. We score twelve LLMs zero-shot under two primary regimes over the same predictions - strict four-stage string matching, and the verdicts of a cross-model judge sharing no model family with any evaluated system - plus one operational composite regime. The two primary regimes show no positive rank correspondence (Spearman rho = -0.273, p = 0.39, not significant), and no model holds its rank across all three regimes. Strict accuracy spans 75.12-89.66%; judge-only accuracy spans 87.19-94.83% with a different leader and tr
    
[^50]: LIME：基于链接的用户-物品交互建模与解耦XOR注意力机制，实现高效的测试时扩展

    LIME: Link-based User-item Interaction Modeling with Decoupled XOR Attention for Efficient Test Time Scaling

    [https://arxiv.org/abs/2510.18239](https://arxiv.org/abs/2510.18239)

    LIME架构通过低秩链接嵌入解耦用户与候选交互、实现注意力权重预计算，并采用线性XOR注意力机制，从根本上降低了推荐系统推理时对候选集大小和用户序列长度的计算成本依赖。

    

    扩展大型推荐系统需要在三个主要方向上取得进展：处理更长的用户历史、扩大候选集规模以及增加模型容量。尽管前景可观，但Transformer的计算成本随用户序列长度呈二次方增长，并随候选数量呈线性增长。这种权衡使得在推理阶段扩大候选集或增加序列长度的代价极其高昂，尽管这能带来显著的性能提升。我们提出了LIME，一种解决这一权衡的新型架构。通过两项关键创新，LIME从根本上降低了计算复杂度。第一，低秩“链接嵌入”通过解耦用户与候选之间的交互，实现注意力权重的预计算，使推理成本几乎与候选集大小无关。第二，线性注意力机制LIME-XOR降低了相对于用户序列长度的计算复杂度。

    arXiv:2510.18239v4 Announce Type: replace-cross  Abstract: Scaling large recommendation systems requires advancing three major frontiers: processing longer user histories, expanding candidate sets, and increasing model capacity. While promising, transformers' computational cost scales quadratically with the user sequence length and linearly with the number of candidates. This trade-off makes it prohibitively expensive to expand candidate sets or increase sequence length at inference, despite the significant performance improvements. We introduce \textbf{LIME}, a novel architecture that resolves this trade-off. Through two key innovations, LIME fundamentally reduces computational complexity. First, low-rank ``link embeddings" enable pre-computation of attention weights by decoupling user and candidate interactions, making the inference cost nearly independent of candidate set size. Second, a linear attention mechanism, \textbf{LIME-XOR}, reduces the complexity with respect to user seque
    
[^51]: 角色增强的意图驱动生成式搜索引擎优化

    Role-Augmented Intent-Driven Generative Search Engine Optimization

    [https://arxiv.org/abs/2508.11158](https://arxiv.org/abs/2508.11158)

    提出RAID方法，通过从源内容推断检索意图并进行4W多角色深度反思，实现了在无需访问目标查询及其反馈的情况下进行生成式搜索引擎优化。

    

    生成式搜索引擎（GSEs）将大语言模型（LLMs）与检索增强生成（RAG）相结合，正在重塑用户获取网络信息的方式，并使内容来源在生成回答中的可见度和影响力成为新的优化目标。现有的生成式搜索引擎优化（G-SEO）方法致力于提升内容可见度，但许多方法依赖于直接访问目标查询或由这些查询引出的GSE反馈。然而，网络内容通常在未来的用户查询尚未知晓时就已经被优化和发布，这使得在无法获取目标查询或其相关反馈的情况下进行G-SEO成为一个重要但尚未被充分探索的问题。为解决这一问题，我们提出了RAID，一种角色增强的意图驱动的G-SEO方法，它无需访问目标查询或依赖查询的GSE反馈即可运行。RAID从源内容中推断主要的检索意图，并通过对潜在查询场景进行4W多角色深度反思来细化该意图。

    arXiv:2508.11158v3 Announce Type: replace-cross  Abstract: Generative search engines (GSEs), which combine large language models (LLMs) with retrieval-augmented generation (RAG), are reshaping how users access Web information and making source visibility and influence in generated responses new optimization objectives. Existing generative search engine optimization (G-SEO) methods seek to improve content visibility, but many rely on direct access to target queries or GSE feedback elicited by those queries. However, Web content is typically optimized and published before future user queries are known, making G-SEO without access to target queries or their associated feedback an important yet underexplored problem. To address this, we propose RAID, a Role-Augmented Intent-Driven G-SEO method that operates without access to target queries or query-dependent GSE feedback. RAID infers a primary retrieval intent from source content, refines it through 4W multi-role deep reflection on prospec
    

