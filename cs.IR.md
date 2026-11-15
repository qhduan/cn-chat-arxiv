# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FERN: Cross-Benchmark Table Retrieval for Text-to-SQL with Frozen Retrievers and Learned Ranking](https://arxiv.org/abs/2611.14477) | FERN不使用任何生成式大语言模型，仅通过融合三个冻结检索器生成候选表格并用LightGBM学习排序器重排，在跨基准的Text-to-SQL表格检索任务上达到了与依赖Claude 3.5的RASL相近的召回率，且无需针对新基准重新训练。 |
| [^2] | [Point-in-Time Encoders for Earnings Calls](https://arxiv.org/abs/2611.14325) | 该论文提出按年度截止日期训练的“时点”文本编码器，以消除财报电话会议检索回测中的前视偏差，其小型专用模型在答案检索上全面优于通用检索器，并将收益预测的秩相关性提升约一半。 |
| [^3] | [StaRSE: Compact Russian Sentence Embeddings with a Sign-Coded Static Encoder](https://arxiv.org/abs/2611.14236) | StaRSE是一种紧凑的俄语静态句子编码器，通过将词元方向以符号位存储并去除Transformer推理，以仅11.3 MiB的模型大小和0.08毫秒的CPU延迟，在RuMTEB上实现了比最强静态基线高7.98分的检索性能。 |
| [^4] | [SHARP: Semantic Hierarchical Agentic Reasoning for Next POI Prediction with Dynamic Memory](https://arxiv.org/abs/2611.14220) | SHARP提出了一种具有动态记忆的分层智能体框架，通过类别级语义推理引导具体的POI预测，并利用轨迹记忆库在有限LLM上下文中保存和更新历史移动知识，解决了LLM语义推理与POI预测脱节、历史知识更新以及全局转移图抹平时间特定类别转移模式三大挑战。 |
| [^5] | [From Search to Decision: How Post-Training Shapes Information-Seeking Capabilities in Large Language Models](https://arxiv.org/abs/2611.14149) | 本文系统梳理了大语言模型的六项相互关联的信息寻求能力，综合分析了后训练方法通过学习信号、优化方式和信用分配机制培养这些能力的实证进展，并指出了检索效用预估等方面仍存在的开放挑战。 |
| [^6] | [Ontology-Guided Knowledge Graph Construction and Robust Evidence Retrieval for Indian Knowledge Systems](https://arxiv.org/abs/2611.14050) | 本文提出了一种面向印度知识体系的本体引导知识图谱构建与检索流水线，将大模型抽取的类型化实体和关系链接到原始文本证据，并通过语义接地、有界图遍历、词法锚点、关系向量搜索与段落重排序的组合检索，在 Dhara Bench 上取得优于单向量检索的证据检索效果，多跳问题 MRR@20 达到 0.8722。 |
| [^7] | [From Show Programs to Linked Data](https://arxiv.org/abs/2611.14037) | 提出一种结合视觉-语言模型、表演艺术 Linked Art 本体扩展与知识图谱约束语义标注的工作流，将阿维尼翁戏剧节节目单转化为结构化关联数据，实现超过98%的转录准确率并显著减少大语言模型幻觉。 |
| [^8] | [LUMIR: Decoupled Multimodal Representation Learning from Heterogeneous Data for Next-POI Recommendation](https://arxiv.org/abs/2611.14004) | 提出解耦多模态表示学习框架LUMIR，利用大语言模型将异构多模态数据统一到共享文本空间，并通过对比学习结合交互感知融合与序列信号约束进行训练，使学习到的表示与下游推荐模型解耦，提升下一兴趣点推荐中多模态表示的通用性和可重用性。 |
| [^9] | [Data Quality Challenges in Multimodal Tourism Recommender Systems](https://arxiv.org/abs/2611.13972) | 该论文聚焦旅游推荐场景，揭示了使用Yelp和Instagram等真实多模态数据时面临的虚假评论噪声、LLM幻觉内容、模态矛盾和人口偏差等数据质量挑战，并提出质量感知融合、受控LLM内容生成和用户群体差异化等未来改进方向。 |
| [^10] | [Controlled Knowledge Engineering for Taxonomy Induction with Tool-Grounded LLM Agents and Runtime-Adaptive Policies](https://arxiv.org/abs/2611.13966) | 本文提出TIGA架构，将分类体系归纳形式化为显式状态与受约束动作空间上的迭代决策过程，借助可执行的分类编辑工具、记录调用与理由的记忆机制以及运行时自适应策略，克服LLM自由生成中的实体幻觉与无效层次关系，实现完整、结构有效且语义连贯的自动分类体系构建。 |
| [^11] | [From Counterfactual Edits to Human Narratives: LLM-Driven Interpretations for Recommender System Explanations](https://arxiv.org/abs/2611.13914) | 该论文提出利用大语言模型作为中介，将推荐系统反事实解释中冗长且难以理解的用户交互历史编辑列表，总结为描述高层偏好模式的简洁自然语言解释，从而弥合生成有效反事实解释与向用户传达模型行为之间的差距。 |
| [^12] | [Auditing LLM Pseudo-Label Distillation for Review Informativeness at E-Commerce Scale](https://arxiv.org/abs/2611.13797) | 本研究审计了电商规模下LLM伪标签蒸馏的质量，发现类别前缀虽能提升教师保真度却降低人工编辑一致性，表明教师保真度不能等同于编辑对齐，且学生模型流水线处理速度比教师快50.2倍。 |
| [^13] | [IFRS Compliance Verification: An Open Multi-LLM Benchmark and Content-Conditioned Ensemble Classifier](https://arxiv.org/abs/2611.13595) | 本文构建了一个包含19,998个IFRS监管条款-披露配对、结合多LLM标签与8,837条人工标注的开放基准，并提出内容条件化的集成分类器框架，以克服单一模型准确率不足及简单集成中各模型系统性偏差导致性能下降的问题。 |
| [^14] | [Exploiting Feature-invariant Scaling Laws for Efficient A/B Experimentation in Recommendation Systems](https://arxiv.org/abs/2611.13539) | 该论文发现现代推荐模型具有特征不变的缩放定律特性，并提出一种结合统计推断的方法，能仅通过小规模A/B实验预测全规模实验效果（含置信区间），从而大幅降低特征变更评估的计算成本。 |
| [^15] | [A Technique to Model Product Information and User Reviews (MPIUR)](https://arxiv.org/abs/2611.13361) | 提出MPIUR多模态图推荐模型，将用户评论编码进异构图并对齐图文表示，在交互历史稀疏的场景下也能显著提升推荐性能（nDCG@10与Recall@10）。 |
| [^16] | [Activation-Guided Graph Retrieval for Cognitive Memory Reconstruction in Language Model Systems](https://arxiv.org/abs/2611.13127) | 该论文提出了一种激活条件化的记忆图检索架构，通过请求与目标线索激活授权记忆图、扇出衰减控制有界遍历、实体递归发射线索，并在生成前重构类型化证据图，从而为长寿命AI系统实现认知记忆重构。 |
| [^17] | [INTENT: Out-of-Sample, Out-of-Time, and Cold-Start Evaluation on User Event Sequences](https://arxiv.org/abs/2611.13069) | 提出了INTENT这一公开基准，根据用户、时期与目标项目是否在训练中出现，将用户事件序列上的下一动作预测模型划分到七种情形下评估，从而统一衡量其样本外、时间外与冷启动泛化能力。 |
| [^18] | [Beyond the Document: Compiled Context and Governed Agents for Any Work Product](https://arxiv.org/abs/2611.13055) | 该论文提出便携式智能体格式（PAF）包这一新交付单元，将工作成果连同编译上下文、证据、来源与交互规则一起封装，实验证明受治理的智能体阅读器在回答相关问题上的成功率（96%）显著优于传统的指令加原始文件检索方式（86%）。 |
| [^19] | [DocFS: File Systems as Agentic Reasoning Environments for Complex Multimodal Document Question Answering](https://arxiv.org/abs/2611.12957) | 该论文提出DocFS，将文件系统作为智能体推理环境，利用目录与文件的层级结构编码多模态文档的组织结构，并借助文件操作工具访问内容，从而提升智能体在复杂多模态文档问答中的表现。 |
| [^20] | [IntentTwin: Deconfounding Candidate Support and the Oracle-to-Deployment Gap in Offline Web Behavior Prediction](https://arxiv.org/abs/2611.12918) | 该论文揭示了离线网页行为预测基准中仅凭候选动作是否属于训练支持集即可“作弊”答对的混杂效应，提出支持可识别率（SIR）这一度量、支持均衡候选集构建方法以及免目标双路径提议器，从而真实刻画从理想评测条件到实际部署的性能差距。 |
| [^21] | [Combining Relevant Expert Knowledge Using Mixtures of Feedback Fuzzy Cognitive Maps](https://arxiv.org/abs/2611.05664) | 该论文将法律证据理论中的相关性定义扩展应用于神经符号系统，通过将专家的反馈模糊认知图混合为共同知识网络，并借助新的类贝叶斯定理对混合知识进行解混合，从而仅组合相关专家答案以提升系统的准确性和可扩展性。 |
| [^22] | [Explain, Then Forget: Recommendation Unlearning under Asymmetric Access Constraint](https://arxiv.org/abs/2610.40658) | 提出了CEU（基于因果解释的遗忘）框架，在不修改模型参数的情况下，通过因果解释机制实现了服务运营商仅能通过API访问模型的MLaaS环境下的推荐遗忘学习。 |
| [^23] | [SpectRAG: Weighted Chunk Graphs for Token-Budgeted Context Selection in Multi-Hop Retrieval-Augmented Generation](https://arxiv.org/abs/2610.36991) | SpectRAG提出了一种谱图增强检索框架，通过构建编码多种语义与结构关系的加权分块图，结合谱相关性传播与Token预算约束下的上下文选择，解决传统独立检索在多跳推理中证据冗余和不完整的问题。 |
| [^24] | [SAM-D2Q: Aligning Multimodal Doc2Query with Search Demand and Conversion for E-commerce](https://arxiv.org/abs/2609.04961) | 提出SAM-D2Q框架，通过任务自适应多模态监督微调与多模态数据增强，在布尔检索约束下将多模态Doc2Query与电商搜索需求和转化目标对齐，有效解决查询与商品标题之间的词汇不匹配问题。 |
| [^25] | [Controllable and Content-Based Recommendations](https://arxiv.org/abs/2607.20938) | 该论文提出了CCBR框架，通过文本瓶颈将协同过滤与直接从物品多模态内容（图像、音频、视频）推断出的文本用户画像相结合，在保持与标准潜在表示模型相当性能的同时，实现了可通过文本干预进行控制的可解释推荐。 |
| [^26] | [DOTRAG: Retrieval-Time Reasoning Along Paths](https://arxiv.org/abs/2605.18760) | DotRAG提出了一种轻量级、无需训练的GraphRAG检索算法，通过构建知识图谱、生成全图概览并利用图可达性索引，让LLM推理在查询时缩小搜索空间并高效恢复与任务相关的关系路径。 |
| [^27] | [Fairness-Aware Retrieval Optimization for Retrieval-Augmented Generation](https://arxiv.org/abs/2605.15790) | 该论文提出三阶段框架FARO，通过概率重排序器控制各位置的群体构成，并利用位置加权线性模型将检索形式化为相关性与输出偏差权衡的优化问题，实现top-k RAG中的公平性感知检索。 |
| [^28] | [ASPIRE: Make Spectral Graph Collaborative Filtering Great Again via Adaptive Filter Learning](https://arxiv.org/abs/2604.22549) | 本文提出ASPIRE框架，通过双层优化目标学习自适应图滤波器，克服了传统推荐目标偏差导致的“低频爆炸”问题，使谱图协同过滤达到与精心手工设计的滤波器相当的性能。 |
| [^29] | [OmniReview: A Large-scale Benchmark and LLM-enhanced Framework for Realistic Reviewer Recommendation](https://arxiv.org/abs/2602.08896) | 该论文提出了OmniReview，通过构建包含202,756条验证审稿记录的大规模基准数据集、三层分级评估框架以及大语言模型增强方法，解决了审稿人推荐中数据稀缺、评估指标过于简化以及嵌入方法信息瓶颈与可解释性不足的问题。 |

# 详细

[^1]: FERN：基于冻结检索器与学习排序的跨基准Text-to-SQL表格检索

    FERN: Cross-Benchmark Table Retrieval for Text-to-SQL with Frozen Retrievers and Learned Ranking

    [https://arxiv.org/abs/2611.14477](https://arxiv.org/abs/2611.14477)

    FERN不使用任何生成式大语言模型，仅通过融合三个冻结检索器生成候选表格并用LightGBM学习排序器重排，在跨基准的Text-to-SQL表格检索任务上达到了与依赖Claude 3.5的RASL相近的召回率，且无需针对新基准重新训练。

    

    在大型合并目录上运行的Text-to-SQL系统必须首先检索出问题所需的表格。我们在合并的BIRD、Spider和Fiben目录上，与RASL已发表的表格召回率进行了比较，RASL使用Claude 3.5模型来提取关键词并选择表格。FERN（无需生成式大语言模型的特征工程排序）则将三个冻结的检索器融合生成100个候选表格，用多达120个特征来描述每个候选，并使用仅在BIRD训练问题上拟合一次的LightGBM排序器对其进行重新排序。在未参与拟合的Spider数据集上，仅其候选池在15个表格处就达到了RASL已发表的召回率（98.0），且排序器将在前五名中包含所有标准表格的问题比例（complete@5）从83.9%提升至95.6%。在5个表格设置下，FERN在Spider上的召回率与RASL相差不到一个百分点（96.7对97.0），在带有证据提示的BIRD上也相差不到一个百分点（96.6对97.5），而在不带证据提示的BIRD上低8.2个百分点（89.3）。在121个真实BEAVER问题上……

    arXiv:2611.14477v1 Announce Type: cross  Abstract: Text-to-SQL systems over large pooled catalogs must first retrieve the tables a question needs. We compare against the published table recall of RASL, which uses Claude 3.5 models to extract keywords and select tables, on pooled BIRD, Spider, and Fiben catalogs. FERN (Feature-Engineered Ranking with No generative LLM) instead fuses three frozen retrievers into 100 candidates, describes each with up to 120 features, and reorders them with LightGBM rankers fitted once on BIRD's training questions. On Spider, where it was not fitted, its candidate pool alone reaches RASL's published recall at fifteen tables (98.0), and the ranker raises the share of questions with every gold table in the top five (complete@5) from 83.9% to 95.6%. At five tables, FERN's recall is within one point of RASL's on Spider (96.7 vs 97.0) and on BIRD with its evidence hints (96.6 vs 97.5), and 8.2 points below on BIRD without them (89.3). On 121 real BEAVER questi
    
[^2]: 财报电话会议的时点编码器

    Point-in-Time Encoders for Earnings Calls

    [https://arxiv.org/abs/2611.14325](https://arxiv.org/abs/2611.14325)

    该论文提出按年度截止日期训练的“时点”文本编码器，以消除财报电话会议检索回测中的前视偏差，其小型专用模型在答案检索上全面优于通用检索器，并将收益预测的秩相关性提升约一半。

    

    文本编码器将财报电话会议转换为向量，用于搜索、用于基于检索段落进行回答的语言模型，以及用于预测模型。实时检索器可以使用在所有可用文本上一次性训练的编码器（正如金融检索器通常的做法），但回测需要一个没有读过每个向量生成日期之后所写文本的编码器。为了解决这种前视偏差，我们从412,248个电话会议中，按照四种预训练方案训练了多个“版本”，截止日期从2010年到2024年逐年划分，每个版本只读取其截止日期当天或之前的文本。随后，我们在电话会议本身的问答对上对每个版本进行微调。仅用电话会议文本训练的版本（参数量3380万）在所有测试季度中，对留出电话会议的答案检索效果都优于我们测试的通用检索器。这些向量还将次日收益与基于过去收益和公司规模构建的预测之间的秩相关系数提高了约一半。当每个版本继续……（原文在此截断）

    arXiv:2611.14325v1 Announce Type: new  Abstract: Text encoders turn earnings calls into vectors for search, for language models that answer from retrieved passages, and for forecasting models. A live retriever can use an encoder trained once on all available text, as finance retrievers are, but a backtest needs an encoder that read no text written after the date of each vector. To address this look-ahead bias, from 412,248 calls we train vintages under four pretraining recipes, with yearly cutoffs from 2010 to 2024, and each vintage reads only text dated on or before its cutoff. We then fine-tune each vintage on the question-answer pairs of the calls themselves. The vintages we train on calls alone, with 33.8M parameters, retrieve answers on held-out calls better than the general retrievers we tested in every test quarter. Their vectors also raise by about half the rank correlation between next-day returns and forecasts built from past returns and firm size. When each vintage continues
    
[^3]: StaRSE：采用符号编码静态编码器的紧凑俄语句子嵌入

    StaRSE: Compact Russian Sentence Embeddings with a Sign-Coded Static Encoder

    [https://arxiv.org/abs/2611.14236](https://arxiv.org/abs/2611.14236)

    StaRSE是一种紧凑的俄语静态句子编码器，通过将词元方向以符号位存储并去除Transformer推理，以仅11.3 MiB的模型大小和0.08毫秒的CPU延迟，在RuMTEB上实现了比最强静态基线高7.98分的检索性能。

    

    Transformer句子编码器提供了强大的检索质量，但当需要在CPU上生成嵌入或需要频繁刷新嵌入时，其推理成本成为瓶颈。我们提出了StaRSE，这是一个紧凑的俄语静态句子编码器，它在部署阶段完全去除了Transformer推理：词元方向以符号位形式存储，词元范数单独存储，句子则通过对范数缩放后的符号方向进行平均池化来编码。StaRSE从ruBERT-base初始化，在俄语相邻句子对和任务形态语对上进行微调，并通过对比学习阶段进行后训练，该阶段直接优化符号投影的词元表。在RuMTEB基准上，StaRSE达到了51.16的任务平均值，比所包含的最强静态基线高出7.98分。打包后的模型仅占用11.3 MiB空间，每个句子的平均CPU延迟为0.08毫秒，而相比之下，16位参数的模型估计占用55.7 MiB空间，平均CPU延迟为1.43毫秒。

    arXiv:2611.14236v1 Announce Type: new  Abstract: Transformer sentence encoders provide strong retrieval quality, but their inference cost becomes a bottleneck when embeddings must be produced on CPUs or refreshed frequently. We introduce StaRSE, a compact Russian static sentence encoder that removes Transformer inference at deployment time: token directions are stored as sign bits, token norms are stored separately, and sentences are encoded by mean pooling over norm-scaled sign directions. StaRSE is initialized from ruBERT-base, fine-tuned on Russian adjacent-sentence and task-shaped pairs, and post-trained with a contrastive stage that directly optimizes the sign-projected token table. On RuMTEB, StaRSE reaches a task mean of $51.16$, $7.98$ points above the strongest included static baseline. The packaged model occupies $11.3$ MiB and has a mean CPU latency of $0.08$ ms per sentence, compared with an estimated $55.7$ MiB 16-bit parameter footprint and $1.43$ ms mean CPU latency for 
    
[^4]: SHARP：基于动态记忆的语义分层智能体推理用于下一个兴趣点（POI）预测

    SHARP: Semantic Hierarchical Agentic Reasoning for Next POI Prediction with Dynamic Memory

    [https://arxiv.org/abs/2611.14220](https://arxiv.org/abs/2611.14220)

    SHARP提出了一种具有动态记忆的分层智能体框架，通过类别级语义推理引导具体的POI预测，并利用轨迹记忆库在有限LLM上下文中保存和更新历史移动知识，解决了LLM语义推理与POI预测脱节、历史知识更新以及全局转移图抹平时间特定类别转移模式三大挑战。

    

    下一个兴趣点（POI）预测通过预判用户的下一个目的地来支持个性化位置服务。近期的利用大语言模型（LLM）进行语义推理、利用图结构进行群体移动性建模的方法改进了POI预测，但仍然存在三个挑战：如何将LLM的语义推理能力与具体的POI预测任务相结合仍具难度；用户移动模式会随时间演变，因此需要在有限的LLM上下文中保留并更新历史知识；同时，全局转移图会平滑掉特定时间段的类别转移模式。为了解决这些挑战，我们提出了SHARP（基于动态记忆的下一个兴趣点预测语义分层智能体推理），这是一个智能体分层框架，其中类别级推理指导具体的POI预测。轨迹记忆智能体维护一个动态轨迹记忆库，以保存早期的移动模式……（摘要被截断）

    arXiv:2611.14220v1 Announce Type: new  Abstract: Next point-of-interest (POI) prediction supports personalized location services by anticipating users' next destinations. Recent methods leverage large language models (LLMs) for semantic reasoning and graphs for collective mobility modeling, improving POI prediction. However, three challenges still remain. Bridging LLMs' semantic reasoning and specific POI prediction remains challenging. User mobility also evolves over time, making it necessary to retain and update historical knowledge within limited LLM contexts. Meanwhile, global transition graphs smooth out time-specific category transition patterns. To address these challenges, we propose Semantic Hierarchical Agentic Reasoning for Next POI Prediction with Dynamic Memory (SHARP), an agentic hierarchical framework in which category-level reasoning guides specific POI prediction. A trajectory memory agent maintains a dynamic trajectory memory bank that preserves earlier mobility patte
    
[^5]: 从搜索到决策：后训练如何塑造大语言模型的信息寻求能力

    From Search to Decision: How Post-Training Shapes Information-Seeking Capabilities in Large Language Models

    [https://arxiv.org/abs/2611.14149](https://arxiv.org/abs/2611.14149)

    本文系统梳理了大语言模型的六项相互关联的信息寻求能力，综合分析了后训练方法通过学习信号、优化方式和信用分配机制培养这些能力的实证进展，并指出了检索效用预估等方面仍存在的开放挑战。

    

    大语言模型（LLM）系统中的信息寻求日益涉及关于何时获取信息、在哪里搜索以及如何评估和使用证据的决策。近期的后训练方法已经改进了若干此类行为，但它们的目标和评估跨越不同的任务和系统设置。本文围绕六项相互关联的信息寻求能力组织相关研究，并考察用于培养这些能力的学习信号、优化方法、更新位置和信用分配机制。我们综合了在针对性行为、端到端性能、搜索控制和泛化方面的实证进展。现有研究报告了在检索路由、查询细化、多工具探索、基于证据的推理和资源感知搜索方面的收益，尽管不同能力的行为证据范围有所差异。我们识别了在估计预期检索效用、协调……（摘要原文在此处截断）等方面尚存的开放挑战。

    arXiv:2611.14149v1 Announce Type: new  Abstract: Information seeking in large language model (LLM) systems increasingly involves decisions about when to acquire information, where to search, and how to evaluate and use evidence. Recent post-training methods have improved several such behaviors, but their objectives and evaluations span different tasks and system settings. This paper organizes research around six interrelated information-seeking capabilities and examines the learning signals, optimization methods, update loci, and credit-assignment mechanisms used to develop them. We synthesize empirical progress in targeted behaviors, end-to-end performance, search control, and generalization. Existing studies report gains in retrieval routing, query refinement, multi-tool exploration, evidence-grounded reasoning, and resource-aware search, although the scope of behavioral evidence varies across capabilities. We identify open challenges in estimating prospective retrieval utility, coor
    
[^6]: 面向印度知识体系的本体引导知识图谱构建与鲁棒证据检索

    Ontology-Guided Knowledge Graph Construction and Robust Evidence Retrieval for Indian Knowledge Systems

    [https://arxiv.org/abs/2611.14050](https://arxiv.org/abs/2611.14050)

    本文提出了一种面向印度知识体系的本体引导知识图谱构建与检索流水线，将大模型抽取的类型化实体和关系链接到原始文本证据，并通过语义接地、有界图遍历、词法锚点、关系向量搜索与段落重排序的组合检索，在 Dhara Bench 上取得优于单向量检索的证据检索效果，多跳问题 MRR@20 达到 0.8722。

    

    从印度知识体系中检索证据面临音译变体、名称歧义以及关系分散于多个段落等问题的挑战。我们提出了一个本体引导的知识图谱（KG）流水线，该流水线将使用 Qwen3-32B 抽取的类型化实体和关系与原始文本证据相链接。检索方法结合了语义感知的语义接地、有界图遍历、词法锚点、关系向量搜索以及段落重排序。我们在来自 Dhara Bench 的 320 个问题上评估了单独的 KG、单向量检索（SV），以及 KG 与 SV 和实体笔记的并集，并在 BGE 和 Qwen 两种重排序器下保持各内部系统每题候选数量一致。在所评估的内部配置中，单独的 KG 在两种重排序器下均取得最高的 Precision@5，并在计数类、复合类和多跳类问题上取得最高的 MRR@20。在使用 Qwen 重排序器时，KG 在多跳问题上达到 0.8722 的 MRR@20，而经过完整重排序的 KG + SV 并集为 0.7622。

    arXiv:2611.14050v1 Announce Type: new  Abstract: Retrieving evidence from Indian Knowledge Systems is complicated by transliteration variants, ambiguous names, and relations distributed across passages. We present an ontology-guided knowledge graph (KG) pipeline that links typed entities and relations extracted with Qwen3-32B to original textual evidence. Retrieval combines sense-aware grounding, bounded graph traversal, lexical anchors, relationship-vector search, and passage reranking. We evaluate KG alone, single-vector retrieval (SV), and KG unions with SV and Entity Notes on 320 questions from Dhara Bench, matching the per-question candidate count across internal systems under BGE and Qwen rerankers. Standalone KG has the highest Precision@5 under both rerankers and the highest MRR@20 in Counting, Composite, and Multi-hop among the evaluated internal configurations. With Qwen, KG achieves Multi-hop MRR@20 of 0.8722, compared with 0.7622 for the fully reranked KG + SV union. The KG
    
[^7]: 从演出节目单到关联数据

    From Show Programs to Linked Data

    [https://arxiv.org/abs/2611.14037](https://arxiv.org/abs/2611.14037)

    提出一种结合视觉-语言模型、表演艺术 Linked Art 本体扩展与知识图谱约束语义标注的工作流，将阿维尼翁戏剧节节目单转化为结构化关联数据，实现超过98%的转录准确率并显著减少大语言模型幻觉。

    

    许多文化遗产机构收藏有大量的演出节目单。由于数量众多而无法逐一编目，这些节目单难以获取，且大部分尚未数字化。本文提出了一种工作流程，通过结合视觉-语言模型、面向表演艺术定制的 Linked Art 本体扩展，以及多种自动语义数据标注方法，将此类文档转化为结构化、可互操作的数据。以法国国家图书馆收藏的阿维尼翁戏剧节节目单（1947年至今）为案例研究，我们展示了视觉-语言模型如何对异构文化遗产文档图像实现超过98%的文本转录准确率，以及如何通过知识图谱约束大语言模型的生成，从而显著减少语义标注中的幻觉现象。我们认为，该工作流程为大规模、可互操作的表演艺术史研究开辟了新的可能性。

    arXiv:2611.14037v1 Announce Type: new  Abstract: Many heritage institutions hold extensive collections of show programs. Too numerous to be cataloged individually, they remain difficult to access and are still largely undigitized. This paper presents a workflow for transforming such documents into structured, interoperable data by combining vision-language models (VLMs), a custom extension of the Linked Art ontology for the performing arts, and different approaches for automatic semantic data annotation. Using the Festival d'Avignon programs (1947--present), preserved at the Biblioth{\`e}que nationale de France, as a case study, we demonstrate how VLMs achieve more than 98% text transcription accuracy from heterogeneous heritage document images and how constraining large language model generation through a knowledge graph significantly reduces hallucinations in semantic annotation. We argue that this workflow opens new possibilities for large-scale, interoperable performing arts histor
    
[^8]: LUMIR：面向下一兴趣点推荐的异构数据解耦多模态表示学习

    LUMIR: Decoupled Multimodal Representation Learning from Heterogeneous Data for Next-POI Recommendation

    [https://arxiv.org/abs/2611.14004](https://arxiv.org/abs/2611.14004)

    提出解耦多模态表示学习框架LUMIR，利用大语言模型将异构多模态数据统一到共享文本空间，并通过对比学习结合交互感知融合与序列信号约束进行训练，使学习到的表示与下游推荐模型解耦，提升下一兴趣点推荐中多模态表示的通用性和可重用性。

    

    随着从异构数据源收集的多模态旅游相关信息日益丰富，学习有效的表示以提升推荐性能受到了广泛关注。现有方法通常将多模态融合与特定下游模型的架构和优化目标紧密耦合，这限制了所学表示的通用性和可重用性。为解决这些局限性，我们提出了LUMIR，一个基于对比学习的解耦多模态表示学习框架。首先利用大语言模型将从异构数据源收集的多模态信息在语义上统一到共享的文本表示空间中，随后设计交互感知的融合机制来建模跨模态交互，同时将序列信号作为结构约束纳入以指导表示学习。为评估（摘要在此处截断）

    arXiv:2611.14004v1 Announce Type: cross  Abstract: As multimodal tourism-related information collected from heterogeneous data sources has become increasingly available, learning effective representations has attracted attention for improving recommendation performance. Existing approaches often tightly couple multimodal fusion with the architecture and optimization objective of a specific downstream model, which limits the generality and reusability of the learned representations. To address these limitations, we propose LUMIR, a decoupled multimodal representation learning framework based on contrastive learning. Multimodal information collected from heterogeneous data sources is first semantically unified into a shared textual representation space using large language models (LLMs). An interaction-aware fusion mechanism is then designed to model cross-modal interactions, while sequential signals are incorporated as a structural constraint to guide representation learning. To evaluat
    
[^9]: 多模态旅游推荐系统中的数据质量挑战

    Data Quality Challenges in Multimodal Tourism Recommender Systems

    [https://arxiv.org/abs/2611.13972](https://arxiv.org/abs/2611.13972)

    该论文聚焦旅游推荐场景，揭示了使用Yelp和Instagram等真实多模态数据时面临的虚假评论噪声、LLM幻觉内容、模态矛盾和人口偏差等数据质量挑战，并提出质量感知融合、受控LLM内容生成和用户群体差异化等未来改进方向。

    

    现代推荐系统日益考虑多样化的数据模态以提升模型性能。然而，底层多模态数据（例如来自社交媒体平台的数据）的质量迄今为止受到的关注有限。本文聚焦于旅游推荐领域，检验了在使用Yelp和Instagram数据集时遇到的现实世界挑战。这些问题包括虚假评论带来的噪声、大语言模型（LLM）产生的幻觉内容、模态间的矛盾以及人口统计学偏差。我们概述了未来的研究方向，例如质量感知融合、结构化和受控的基于LLM的内容生成以减少幻觉，以及用户群体差异化，以提高多模态推荐系统的鲁棒性和可靠性。

    arXiv:2611.13972v1 Announce Type: cross  Abstract: Modern recommender systems increasingly consider diverse data modalities to enhance model performance. However, the quality of the underlying multimodal data (e.g. data from social media platforms) has received limited attention so far. Focusing on tourism recommendation, we examine real-world challenges encountered when using datasets from Yelp and Instagram. Issues include noise from fake reviews, hallucinated content from LLMs, modality contradictions, and demographic bias. We outline future directions such as quality-aware fusion, structured and controlled LLM-based content generation to reduce hallucinations, and user group differentiation to improve the robustness and reliability of multimodal recommendation systems.
    
[^10]: 基于工具锚定LLM智能体与运行时自适应策略的分类体系归纳受控知识工程

    Controlled Knowledge Engineering for Taxonomy Induction with Tool-Grounded LLM Agents and Runtime-Adaptive Policies

    [https://arxiv.org/abs/2611.13966](https://arxiv.org/abs/2611.13966)

    本文提出TIGA架构，将分类体系归纳形式化为显式状态与受约束动作空间上的迭代决策过程，借助可执行的分类编辑工具、记录调用与理由的记忆机制以及运行时自适应策略，克服LLM自由生成中的实体幻觉与无效层次关系，实现完整、结构有效且语义连贯的自动分类体系构建。

    

    分类体系是一类结构化的知识表示，可支持语义搜索、推荐、产品目录组织以及检索增强生成。从固定的候选实体集合自动构建分类体系，所需的远不止看似合理的文本生成：所得的层次结构必须完整、结构有效且语义连贯。大型语言模型（LLM）在这项任务上前景可观，但迭代式的自由形式生成仍容易出现实体遗漏或幻觉、无效的层次关系以及不一致的组织原则等问题。我们提出了TIGA（Taxonomy Induction with Tool-Grounded Agents，基于工具锚定智能体的分类体系归纳），这是一种知识工程架构，它将分类体系归纳形式化为在显式状态与受约束动作空间上的迭代决策过程。TIGA结合了可执行的分类体系编辑工具、记录工具调用、理由与结果的记忆机制，以及运行时自适应的策略。

    arXiv:2611.13966v1 Announce Type: cross  Abstract: Taxonomies are structured knowledge representations that support semantic search, recommendation, product catalog organization, and retrieval-augmented generation. Their automated construction from a fixed set of candidate entities requires more than plausible text generation: the resulting hierarchy must be complete, structurally valid, and semantically coherent. Large Language Models (LLMs) are promising for this task, but iterative free-form generation remains vulnerable to omitted or hallucinated entities, invalid hierarchical relations, and inconsistent organizational principles. We introduce TIGA (Taxonomy Induction with Tool-Grounded Agents), a knowledge-engineering architecture that formulates taxonomy induction as an iterative decision process over an explicit state and constrained action space. TIGA combines executable taxonomy-editing tools, an memory that records tool calls, rationales, and outcomes, and a runtime-adaptive 
    
[^11]: 从反事实编辑到人类叙事：基于大语言模型的推荐系统解释诠释

    From Counterfactual Edits to Human Narratives: LLM-Driven Interpretations for Recommender System Explanations

    [https://arxiv.org/abs/2611.13914](https://arxiv.org/abs/2611.13914)

    该论文提出利用大语言模型作为中介，将推荐系统反事实解释中冗长且难以理解的用户交互历史编辑列表，总结为描述高层偏好模式的简洁自然语言解释，从而弥合生成有效反事实解释与向用户传达模型行为之间的差距。

    

    推荐系统日益影响着用户在网上发现和消费的内容，但其输出背后的决策逻辑往往仍不透明。反事实解释（CFX）方法提供了一种有效途径，可以展示用户交互历史中的哪些改变会改变推荐结果。然而，在推荐场景中，CFX 通常需要多次编辑，产生冗长的、低层次（细粒度）的编辑列表，用户难以理解。此外，大多数评估侧重于编辑数量，而在很大程度上忽略了生成的 CFX 对用户来说是否连贯且可操作。我们认为，生成有效的 CFX 与传达模型行为之间的这一差距，源于缺乏以用户为中心的抽象以及有限的定性评估。为弥合这一差距，我们研究将大语言模型作为中介，把原始的 CFX 交互编辑列表总结为描述更高层次偏好模式的简洁自然语言解释。

    arXiv:2611.13914v1 Announce Type: new  Abstract: Recommender systems increasingly influence what users discover and consume online, yet the decision logic behind their outputs often remains opaque. Counterfactual explanation (CFX) methods provide an effective way to show which changes to a user's interaction history would alter a recommendation. In recommendation settings, however, CFXs often require many edits, producing long, low-level lists that are hard for users to interpret. Moreover, most evaluations emphasize edit counts while largely overlooking whether the resulting CFXs are coherent and actionable for users. We argue that this gap between generating valid CFXs and communicating model behavior stems from the lack of user-centered abstraction and limited qualitative evaluation. To bridge this gap, we investigate large language models as mediators that summarize raw CFX interaction edit lists into concise natural language explanations describing higher-level preference patterns
    
[^12]: 电商规模下针对评论信息量的LLM伪标签蒸馏审计

    Auditing LLM Pseudo-Label Distillation for Review Informativeness at E-Commerce Scale

    [https://arxiv.org/abs/2611.13797](https://arxiv.org/abs/2611.13797)

    本研究审计了电商规模下LLM伪标签蒸馏的质量，发现类别前缀虽能提升教师保真度却降低人工编辑一致性，表明教师保真度不能等同于编辑对齐，且学生模型流水线处理速度比教师快50.2倍。

    

    我们在534万条土耳其电商评论上研究了从Gemini 3.1 Flash-Lite到紧凑编码器的硬标签蒸馏。研究使用严格的新SKU时间测试来衡量教师模型保真度；并由一位公司标注员进行独立的、对教师标签盲的审计，以衡量编辑一致性。H3类别前缀在完整的新SKU测试上略微提升了教师保真度。但在16,000条评论的时间顺序审计子集上，它反而降低了与Gemini和标注员的一致性，其中CORN的配对差异比交叉熵更为明显。因此，上下文效应取决于评估人群，仅凭教师保真度并不能确立编辑对齐。在所测量的离线配置下，单个L4学生模型流水线处理评论的速度比教师流水线快50.2倍。

    arXiv:2611.13797v1 Announce Type: new  Abstract: We study hard-label distillation from Gemini 3.1 Flash-Lite to compact encoders on 5.34 million Turkish marketplace reviews. A strict new-SKU temporal test measures teacher fidelity; a separate, teacher-label-blind audit by one company annotator measures editorial agreement. An H3 category prefix slightly improves teacher fidelity on the full new-SKU test. On the 16,000-review chronological audit subset, it instead reduces agreement with both Gemini and the annotator, with a clearer paired difference for CORN than for cross entropy. Thus the context effect depends on the evaluation population, and teacher fidelity alone does not establish editorial alignment. Under the measured offline configurations, a single-L4 student pipeline processes reviews 50.2 times faster than the teacher pipeline.
    
[^13]: 国际财务报告准则（IFRS）合规性验证：开放的多大语言模型基准与内容条件化集成分类器

    IFRS Compliance Verification: An Open Multi-LLM Benchmark and Content-Conditioned Ensemble Classifier

    [https://arxiv.org/abs/2611.13595](https://arxiv.org/abs/2611.13595)

    本文构建了一个包含19,998个IFRS监管条款-披露配对、结合多LLM标签与8,837条人工标注的开放基准，并提出内容条件化的集成分类器框架，以克服单一模型准确率不足及简单集成中各模型系统性偏差导致性能下降的问题。

    

    根据国际财务报告准则（IFRS）验证企业财务披露是一项核心审计任务，专家需要将披露内容与数百条细粒度的监管条款进行交叉比对：这一过程费力、成本高昂且容易出错，因此人们越来越有兴趣将大语言模型（LLM）用作合规助手。然而，单一模型的准确率对于高风险应用场景仍然不够，而简单的集成方法容易受到各模型系统性偏差的影响，可能导致总体准确率甚至低于最强的个体标注者。本研究通过一个多阶段框架来填补这一空白，该框架由一个新构建的基准数据集支持，该数据集包含19,998个IFRS监管条款-披露配对，其中包括11,161个多LLM一致同意的标签，以及在LLM意见不一致的条目上的8,837个人工标注。该框架结合了三阶段领域自适应的句子嵌入流水线（在SEC EDGAR文件上进行BERT掩码语言模型预训练、无监督……）※注：摘要原文在此处被截断。

    arXiv:2611.13595v1 Announce Type: new  Abstract: Verifying corporate financial disclosures against the International Financial Reporting Standards (IFRS) is a core auditing task in which experts cross-reference disclosures with hundreds of fine-grained regulatory paragraphs: a laborious, costly, and error-prone process that motivates growing interest in large language models (LLMs) as compliance assistants. Yet single-model accuracy remains insufficient for high-stakes use, and naive ensembling is vulnerable to systematic per-model biases that can drag aggregate accuracy below that of the strongest individual annotator. This gap is addressed with a multi-stage framework, supported by a new benchmark of 19,998 IFRS regulation-disclosure pairs comprising 11,161 unanimous multi-LLM labels and 8,837 human annotations on items where the LLMs disagreed. The framework couples a three-stage domain-adapted sentence embedding pipeline (BERT masked language modeling on SEC EDGAR filings, unsuperv
    
[^14]: 利用特征不变的缩放定律实现推荐系统中高效的A/B实验

    Exploiting Feature-invariant Scaling Laws for Efficient A/B Experimentation in Recommendation Systems

    [https://arxiv.org/abs/2611.13539](https://arxiv.org/abs/2611.13539)

    该论文发现现代推荐模型具有特征不变的缩放定律特性，并提出一种结合统计推断的方法，能仅通过小规模A/B实验预测全规模实验效果（含置信区间），从而大幅降低特征变更评估的计算成本。

    

    研究人员和工程师通过进行A/B实验来改进推荐系统，实验内容涉及新的模型架构、优化方法和数据特征。然而，针对每一次特征变更都运行全规模实验（例如使用在线数据进行全模型训练）的计算成本过高，而单个小规模实验的结果往往具有误导性。为了应对这一挑战，我们证明了现代推荐模型表现出特征不变的缩放特性：即使对基线模型应用不同的特征变更，模型性能的变化仍遵循可预测的缩放定律。我们提出了一种高效方法，利用缩放定律并结合统计推断来预测（并提供置信区间）推荐系统中A/B实验在全规模下的效果。这使得从业者能够仅通过小规模实验就高效地过滤掉中性或造成性能退化的特征变更。

    arXiv:2611.13539v1 Announce Type: cross  Abstract: Researchers and engineers improve recommendation systems by performing A/B experiments with new model architectures, optimization methods, and data features. However, running full-scale experiments (e.g., full-model training with online data) for every feature change is prohibitively computationally intensive, while the result from a single small-scale experiment is often misleading. To address this challenge, we show that modern recommendation models exhibit feature-invariant scaling: the change of a model's performance follows a predictable scaling-law even when different feature changes are applied to the baseline model. We introduce an efficient method that exploits scaling-laws and uses statistical inference to predict (with confidence intervals) the full-scale effect of an A/B experiment in recommendation systems. This enables practitioners to efficiently filter out neutral or regressive feature changes with only small-scale expe
    
[^15]: 一种对商品信息与用户评论建模的技术

    A Technique to Model Product Information and User Reviews (MPIUR)

    [https://arxiv.org/abs/2611.13361](https://arxiv.org/abs/2611.13361)

    提出MPIUR多模态图推荐模型，将用户评论编码进异构图并对齐图文表示，在交互历史稀疏的场景下也能显著提升推荐性能（nDCG@10与Recall@10）。

    

    现代电商平台天然提供三种互补的模态：文本形式的商品描述、商品图片以及用户评论。然而，现有的多模态推荐方法往往忽略评论，或者未能在一个统一框架内整合这三种模态。本文提出了MPIUR，这是一个基于图的多模态推荐模型，它在异构图中对评论进行编码，对齐图像与文本表示，并根据商品交互密度自适应地调整协同—语义对齐。在Amazon Reviews 2023的乐器和母婴两个品类上的实验表明，MPIUR在所评估的方法中于两个数据集上均取得了最佳的全局nDCG@10和Recall@10。它还在稀疏商品上获得了最佳的nDCG@10和Recall@10，这表明当交互历史有限时，联合利用评论与多模态商品内容尤为有效。

    arXiv:2611.13361v1 Announce Type: new  Abstract: Modern e-commerce platforms naturally provide three complementary modalities: textual item descriptions, item images, and user reviews. However, existing multimodal recommendation methods often omit reviews or do not integrate all three modalities within a unified framework. In this paper, we propose MPIUR, a multimodal graph-based recommendation model that encodes reviews in heterogeneous graphs, aligns image and text representations, and adapts collaborative--semantic alignment to item interaction density. Experiments on the Amazon Reviews 2023 Musical Instruments and Baby categories show that MPIUR achieves the best global nDCG@10 and Recall@10 among the evaluated methods on both datasets. It also obtains the best nDCG@10 and Recall@10 on sparse items, indicating that jointly exploiting reviews and multimodal item content is particularly useful when interaction histories are limited.
    
[^16]: 面向语言模型系统中认知记忆重构的激活引导图检索

    Activation-Guided Graph Retrieval for Cognitive Memory Reconstruction in Language Model Systems

    [https://arxiv.org/abs/2611.13127](https://arxiv.org/abs/2611.13127)

    该论文提出了一种激活条件化的记忆图检索架构，通过请求与目标线索激活授权记忆图、扇出衰减控制有界遍历、实体递归发射线索，并在生成前重构类型化证据图，从而为长寿命AI系统实现认知记忆重构。

    

    检索增强语言模型系统通常通过基于查询条件的相关性来选择上下文。我们认为，长寿命AI系统需要第二种算子：激活条件化的记忆可用性。我们规定了一种架构，其中请求与目标线索为授权的记忆图进行播种，激活与扇出衰减分配有界的遍历前沿，检索到的实体递归地发射线索，并在生成之前重构类型化证据图。该规范产生了关于候选选择、遍历控制、图正则化、证据重构和强化的五项架构性推论。我们推导了实现层面的成本估计，并定义了消融实验、失败条件以及可复现的评估协议。如果在同等预算下，一个更简单的图重排序器在因果与时间恢复任务上能够匹配完整遍历的效果，则该核心主张将被收窄或证伪。

    arXiv:2611.13127v1 Announce Type: new  Abstract: Retrieval-augmented language model systems generally select context by query-conditioned relevance. We argue that long-lived AI systems require a second operator: activation-conditioned memory availability. We specify an architecture in which request and goal cues seed an authorized memory graph, activation and fan attenuation allocate a bounded traversal frontier, retrieved entities recursively emit cues, and a typed evidence graph is reconstructed before generation. The specification yields five architectural consequences about candidate selection, traversal control, graph regularization, evidence reconstruction, and reinforcement. We derive implementation-level cost estimates and define ablations, failure conditions, and a reproducible evaluation protocol. The central claim is narrowed or falsified if a simpler graph reranker matches full traversal on causal and temporal recovery under equal budgets. This architecture-and-analysis pre
    
[^17]: INTENT：用户事件序列上的样本外、时间外与冷启动评估

    INTENT: Out-of-Sample, Out-of-Time, and Cold-Start Evaluation on User Event Sequences

    [https://arxiv.org/abs/2611.13069](https://arxiv.org/abs/2611.13069)

    提出了INTENT这一公开基准，根据用户、时期与目标项目是否在训练中出现，将用户事件序列上的下一动作预测模型划分到七种情形下评估，从而统一衡量其样本外、时间外与冷启动泛化能力。

    

    银行如今在客户事件的有序序列上训练基础模型，例如交易、应用会话和授权。部署后，这类模型需要在时间上向前预测，且预测对象往往是训练中未出现的客户和商家；然而每个模型都在各自的私有数据上评估，因而无法相互比较。针对此类序列的公共基准很少测试模型未见过的时期；推荐系统研究则通常只保留每个用户的最后几个事件作为测试。我们提出了INTENT（面向下一动作预测的交互、新颖性与迁移评估），这是一个公开基准，根据用户、时期和目标项目是否出现在训练中，将模型评估划分为七种情形进行打分。对59个公开数据集家族的审计最终保留了五个数据集；最常见的缺陷是缺少用于对事件排序的日历时间戳。由于没有任何日期落在所有五个数据集共同的时间跨度内，因此每个数据集都按照事先固定的统一规则在其各自的截止时间点进行切分。

    arXiv:2611.13069v1 Announce Type: new  Abstract: Banks now train foundation models on ordered sequences of customer events such as transactions, app sessions, and authorizations. Deployed, such a model predicts forward in time, often for customers and merchants absent from training, yet each is evaluated on private data, so no two can be compared. Public benchmarks for such sequences rarely test a period the model has not seen; recommendation studies usually hold out each user's last events instead. We present INTENT (INTEraction, Novelty, and Transfer evaluation for next-action prediction), a public benchmark that scores a model in seven cells defined by whether the user, the period, and the target item appeared in training. An audit of 59 public dataset families retains five datasets; the most common defect is the lack of a calendar timestamp that orders events. No date lies within the time span of all five datasets, so each is split at its own cutoff, chosen by one rule fixed before
    
[^18]: 超越文档：面向任意工作成果的编译上下文与受治理智能体

    Beyond the Document: Compiled Context and Governed Agents for Any Work Product

    [https://arxiv.org/abs/2611.13055](https://arxiv.org/abs/2611.13055)

    该论文提出便携式智能体格式（PAF）包这一新交付单元，将工作成果连同编译上下文、证据、来源与交互规则一起封装，实验证明受治理的智能体阅读器在回答相关问题上的成功率（96%）显著优于传统的指令加原始文件检索方式（86%）。

    

    文档保存了专业工作的可见成果，却往往丢弃了其背后的研究、计算、假设、证据和授权边界。我们研究了一种不同的交付单元：便携式智能体格式（Portable Agent Format, PAF）包，它将工作成果与编译后的上下文、证据、来源信息以及后续交互规则一同携带。为了检验这种上下文在产生它的过程之外是否依然有用，我们评估了四个新创建的包（均创建于托管阅读器和离线阅读器被固定之后）：一份投资提案、基金研究、财务报表分析和一份工程事故复盘。其中两个包由专用智能体生成，另外两个通过通用助手的打包技能生成。在400个冻结问题和12,000条评估回复中，托管的受治理阅读器在96%的问题上回答成功，而采用指令加原始文件检索的方式成功率为86%（经Holm校正的 p < 10^-7）。

    arXiv:2611.13055v1 Announce Type: cross  Abstract: A document preserves the visible result of professional work but often discards the research, calculations, assumptions, evidence, and authorization boundaries behind it. We study a different unit of delivery: a Portable Agent Format (PAF) package that carries a work product together with compiled context, evidence, provenance, and rules for subsequent interaction. To test whether this context remains useful beyond the process that produced it, we evaluate four new packages, created after the hosted and offline readers were fixed: an investment proposal, fund research, financial-statement analysis, and an engineering incident review. Two packages were produced by specialized agents and two through a general-purpose assistant's packaging skill. Across 400 frozen questions and 12,000 evaluated responses, a hosted governed reader succeeds on 96% of questions versus 86% with instructions plus raw-file retrieval (Holm-adjusted p < 10^-7). O
    
[^19]: DocFS：文件系统作为复杂多模态文档问答的智能体推理环境

    DocFS: File Systems as Agentic Reasoning Environments for Complex Multimodal Document Question Answering

    [https://arxiv.org/abs/2611.12957](https://arxiv.org/abs/2611.12957)

    该论文提出DocFS，将文件系统作为智能体推理环境，利用目录与文件的层级结构编码多模态文档的组织结构，并借助文件操作工具访问内容，从而提升智能体在复杂多模态文档问答中的表现。

    

    基于视觉语言模型（VLM）的智能体通过与环境的迭代交互来解决复杂任务，其有效性在很大程度上取决于环境如何暴露信息并提供作用于信息的工具。然而，为复杂的多模态文档问答构建有效的环境仍然具有挑战性：文档通常跨越多页，多模态内容分布在层级章节中，并通过跨模态关联相互链接，而现有方法将文档组织成结构化表示，仅解决了环境的信息层面，缺乏与结构对齐的工具，限制了智能体充分利用文档结构的能力。我们观察到，文件系统天然地将结构化组织与对齐的工具结合在一起：目录和文件编码文档结构，而文件操作提供了丰富的工具来访问文件内容。基于这一洞察，我们提出了DocFS，一个基于文件系统的智能体推理环境，用于复杂多模态文档问答。

    arXiv:2611.12957v1 Announce Type: new  Abstract: VLM-based agents solve complex tasks by iteratively interacting with environments, and their effectiveness depends heavily on how the environment exposes information and provides tools for acting on it. However, constructing an effective environment for complex multimodal document question answering remains challenging: documents usually span many pages, with multimodal contents distributed across hierarchical sections and linked by cross-modal associations, while existing methods organize documents into structured representations that address only the information side of the environment and lack tools aligned with the structure, limiting agents' ability to fully exploit it. We observe that file systems naturally combine structure organization with aligned tools: directories and files encode document structure, while file operations provide rich tools to access file contents. Building on this insight, we propose DocFS, a file-system-base
    
[^20]: IntentTwin：离线网页行为预测中候选支持去混杂与“理想评测到实际部署”的差距

    IntentTwin: Deconfounding Candidate Support and the Oracle-to-Deployment Gap in Offline Web Behavior Prediction

    [https://arxiv.org/abs/2611.12918](https://arxiv.org/abs/2611.12918)

    该论文揭示了离线网页行为预测基准中仅凭候选动作是否属于训练支持集即可“作弊”答对的混杂效应，提出支持可识别率（SIR）这一度量、支持均衡候选集构建方法以及免目标双路径提议器，从而真实刻画从理想评测条件到实际部署的性能差距。

    

    离线网页行为基准测试通常将示范动作插入一个有限候选集，然后测量模型的重排序性能。我们证明这种协议会造成“候选支持混杂”问题：当目标动作位于训练支持集之外、而所有干扰项均从该支持集中采样时，仅凭支持集归属信息就能识别出正确动作，完全无需建模用户行为。我们用“支持可识别率”（SIR）将这一效应形式化，构建了支持均衡的候选集，并评估了一种免目标的双路径候选提议器，该提议器将支持集检索与开放词表的DOM/语言模型提议相结合。在过滤后的OPeRA数据集（训练/验证/测试分别为4,282/469/992步）上，原始协议在目标为词表外（OOV）的样本上SIR达到1.0000，仅用支持集规则即可取得1.0000的OOV Exact@1。支持均衡化将SIR降至0.0000；此时最强的条件重排序器总体Exact@1为0.7214，在词表内（IV）样本上为0.7988，在OOV样本上为0.5668。免目标双路径系统达到P（摘要原文在此处截断）

    arXiv:2611.12918v1 Announce Type: new  Abstract: Offline web-behavior benchmarks frequently insert the demonstrated action into a finite candidate set and then measure reranking. We show that this protocol can create a candidate-support confound: when the target is outside training support while all distractors are sampled from that support, support membership identifies the gold action without modeling behavior. We formalize this effect with support-identifiability rate (SIR), construct support-balanced candidate sets, and evaluate a target-free dual-path proposer that combines support retrieval with open-vocabulary DOM/LM proposals. On filtered OPeRA (4,282/469/992 train/dev/test steps), the original protocol has SIR 1.0000 on target-OOV rows and a support-only rule reaches 1.0000 OOV Exact@1. Support balancing reduces SIR to 0.0000; the strongest conditional reranker obtains 0.7214 overall Exact@1, 0.7988 on IV rows, and 0.5668 on OOV rows. The target-free dual-path system reaches P
    
[^21]: 使用反馈模糊认知图混合体组合相关专家知识

    Combining Relevant Expert Knowledge Using Mixtures of Feedback Fuzzy Cognitive Maps

    [https://arxiv.org/abs/2611.05664](https://arxiv.org/abs/2611.05664)

    该论文将法律证据理论中的相关性定义扩展应用于神经符号系统，通过将专家的反馈模糊认知图混合为共同知识网络，并借助新的类贝叶斯定理对混合知识进行解混合，从而仅组合相关专家答案以提升系统的准确性和可扩展性。

    

    神经符号系统应当只组合来自专家智能体或其他知识源的相关答案。随着知识网络的不断增长并连接越来越多的知识源，针对给定输入查询仅使用相关专家对于准确性和可扩展性变得愈发重要。我们从证据法理论出发扩展了相关性的标准定义：在给定可用知识的条件下，当且仅当一个答案对于手头问题既具有证明力又具有实质性时，该答案才是相关的。我们将这一相关性方案应用于混合真实或人工专家的反馈模糊认知图（FCMs）。各专家FCM的有向因果图混合或组合成一个共同的知识网络。我们将专家的均衡极限环预测或答案与混合FCM的答案进行比较，然后通过它们之间的差异或变化来估计答案的证明力状态。一个新的类贝叶斯定理将知识混合物解混合为……（原文摘要在此处截断）

    arXiv:2611.05664v1 Announce Type: new  Abstract: A neuro-symbolic system should combine only relevant answers from expert agents or other knowledge sources. Using only relevant experts for a given input query becomes ever more important for accuracy and scalability as the knowledge network grows and connects ever more knowledge sources. We extend the standard definition of relevance from the legal theory of evidence: An answer is relevant if and only if it is both probative and material for the matter at hand given the knowledge available. We apply this relevancy scheme to mixing the feedback fuzzy cognitive maps (FCMs) of real or artificial experts. The expert FCM directed causal graphs mix or combine into a common knowledge network. We compare the equilibrium limit-cycle predictions or answers of the experts to the answers of the mixture FCM and then estimate the probative status of answers by their difference or variation. A new Bayes-like theorem de-mixes the knowledge mixture into
    
[^22]: 解释，然后遗忘：非对称访问约束下的推荐遗忘学习

    Explain, Then Forget: Recommendation Unlearning under Asymmetric Access Constraint

    [https://arxiv.org/abs/2610.40658](https://arxiv.org/abs/2610.40658)

    提出了CEU（基于因果解释的遗忘）框架，在不修改模型参数的情况下，通过因果解释机制实现了服务运营商仅能通过API访问模型的MLaaS环境下的推荐遗忘学习。

    

    推荐遗忘学习旨在消除那些不应再影响未来推荐的特定历史交互（如噪声点击或过时的用户偏好）所带来的影响。然而，在现代机器学习即服务架构中，执行推荐遗忘变得困难得多，因为推荐模型由模型提供商远程托管，服务运营商只能通过API进行访问。在这种设置下，服务运营商为最终用户提供服务，但无法访问或修改内部模型参数。因此，现有的遗忘方法难以应用，因为它们需要访问模型内部才能进行重新训练、微调或其他参数更新。为了应对这些挑战，我们提出了基于因果解释的遗忘方法（CEU），这是一种专为MLaaS环境量身定制的新型遗忘框架。CEU不修改内部模型权重，而是通过两阶段（摘要原文在此处截断）

    arXiv:2610.40658v1 Announce Type: new  Abstract: Recommendation unlearning aims to remove the influence of specific historical interactions that should no longer affect future recommendations, such as noisy clicks or outdated user preferences. However, performing recommendation unlearning becomes much harder in modern Machine-Learning-as-a-Service (MLaaS) architectures, where the recommendation model is remotely hosted by a model provider (MP) and accessed by service operators (SO) only through APIs. In this setting, a service operator serves end users but cannot access or modify internal model parameters. As a result, existing unlearning methods are difficult to apply because they require access to model internals for retraining, fine-tuning, or other parameter updates. To address these challenges, we propose Causal Explanation-based Unlearning (CEU), a novel unlearning framework tailored for MLaaS environments. Instead of modifying internal model weights, CEU operates through a two-s
    
[^23]: SpectRAG：面向多跳检索增强生成中基于Token预算的上下文选择的加权分块图

    SpectRAG: Weighted Chunk Graphs for Token-Budgeted Context Selection in Multi-Hop Retrieval-Augmented Generation

    [https://arxiv.org/abs/2610.36991](https://arxiv.org/abs/2610.36991)

    SpectRAG提出了一种谱图增强检索框架，通过构建编码多种语义与结构关系的加权分块图，结合谱相关性传播与Token预算约束下的上下文选择，解决传统独立检索在多跳推理中证据冗余和不完整的问题。

    

    检索增强生成（RAG）系统通过检索外部证据并将生成过程以检索到的上下文为条件，从而为大型语言模型（LLM）提供事实依据。尽管稠密检索在局部语义匹配方面较为有效，但独立排序的检索可能返回冗余段落、遗漏跨分块的依赖关系，并为多跳推理提供不完整的证据。我们提出了SpectRAG——一种谱图增强检索框架，它结合了混合候选种子选择、加权分块图构建、基于图的扩展、谱相关性传播以及基于Token预算的上下文选择。文档分块由稠密语义嵌入和图坐标系表示，后者源自一个加权分块图，其边编码了语义相似性、文档相邻性、章节层次结构、共享实体、引用链接以及LLM生成的效用问题之间的相似性。在查询时，SpectRAG构建一个hi……（摘要原文在此截断）

    arXiv:2610.36991v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) systems ground large language models (LLMs) by retrieving external evidence and conditioning generation on the retrieved context. Although dense retrieval is effective for local semantic matching, independently ranked retrieval can return redundant passages, miss cross-chunk dependencies, and provide incomplete evidence for multi-hop reasoning. We propose SpectRAG, a spectral graph-augmented retrieval framework that combines hybrid candidate seeding, weighted chunk graph construction, graph-based expansion, spectral relevance propagation, and token-budgeted context selection. Document chunks are represented by dense semantic embeddings and graph-spectral coordinates derived from a weighted chunk graph whose edges encode semantic similarity, document adjacency, section hierarchy, shared entities, citation links, and similarity between LLM-generated utility questions. At query time, SpectRAG builds a hi
    
[^24]: SAM-D2Q：面向电商场景的多模态Doc2Query与搜索需求及转化的对齐

    SAM-D2Q: Aligning Multimodal Doc2Query with Search Demand and Conversion for E-commerce

    [https://arxiv.org/abs/2609.04961](https://arxiv.org/abs/2609.04961)

    提出SAM-D2Q框架，通过任务自适应多模态监督微调与多模态数据增强，在布尔检索约束下将多模态Doc2Query与电商搜索需求和转化目标对齐，有效解决查询与商品标题之间的词汇不匹配问题。

    

    电商搜索常常面临用户查询与商家撰写的产品标题之间的词汇不匹配问题，因为简短的标题无法完全覆盖多样的用户表达方式或产品的视觉属性。尽管Doc2Query通过为文档生成伪查询进行扩展来缓解这一问题，但传统方法仅基于文本，且未针对电商业务目标进行优化。因此，这些方法可能产生语义上合理但商业上无效的扩展，并遗漏产品图像中包含的关键属性。为此，我们提出了电商搜索对齐的多模态Doc2Query（SAM-D2Q），这是一个在布尔检索约束下、面向电商搜索的业务对齐多模态文档扩展框架。SAM-D2Q包含三个阶段：（1）任务自适应的多模态监督微调，以增强对产品标题、图像和用户查询的视觉-语言理解能力；（2）多模态数据增强，以提升……（原文摘要不完整，到此截断）

    arXiv:2609.04961v1 Announce Type: new  Abstract: E-commerce search often suffers from vocabulary mismatch between user queries and merchant-authored product titles, since short titles cannot fully cover diverse user expressions or visual product attributes. Although Doc2Query alleviates this issue by generating pseudo-queries for document expansion, traditional methods are text-only and not optimized for e-commerce business objectives. As a result, they may produce semantically plausible but commercially ineffective expansions and miss key attributes present in product images. To this end, we propose E-commerce Search-Aligned Multimodal Doc2Query (SAM-D2Q), a business-aligned multimodal document expansion framework for e-commerce search under Boolean retrieval constraints. SAM-D2Q consists of three stages: (1) task-adapted multimodal supervised fine-tuning to enhance vision-language understanding of product titles, images, and user queries; (2) multimodal data augmentation to improve p
    
[^25]: 可控的基于内容的推荐系统

    Controllable and Content-Based Recommendations

    [https://arxiv.org/abs/2607.20938](https://arxiv.org/abs/2607.20938)

    该论文提出了CCBR框架，通过文本瓶颈将协同过滤与直接从物品多模态内容（图像、音频、视频）推断出的文本用户画像相结合，在保持与标准潜在表示模型相当性能的同时，实现了可通过文本干预进行控制的可解释推荐。

    

    传统推荐系统依赖于潜在（稠密）表示，这使得它们难以解释和控制。我们提出了可控且基于内容的推荐框架（CCBR），该框架基于文本用户画像表示来构建推荐。CCBR可嵌入协同过滤模型中，并通过文本瓶颈引入可控性。我们展示了CCBR支持基于文本和多模态的干预，允许用户将模型引导向他们偏好的方向。与现有的可控推荐系统不同，CCBR直接从物品内容（图像、音频或视频）推断文本摘要。在图像、音频和视频数据集上，我们证明了所提出的框架在获得与标准（潜在表示）模型相当的性能的同时，还能通过文本提供可控的模型摘要。该模型还优于TEARS——一个近期的可控推荐基线。

    arXiv:2607.20938v2 Announce Type: replace  Abstract: Traditional recommendation systems rely on latent (dense) representations, making them difficult to interpret and control. We propose the Controllable and Content-Based Recommendations (CCBR) framework, which builds its recommendations from textual user profile representations. CCBR plugs into collaborative filtering models and introduces controllability via text bottlenecks. We show that CCBR enables text-based and multimodal interventions, allowing users to steer the model towards the directions they prefer. Different from existing controllable recommendation systems, CCBR infers the text summaries directly from item contents (images, audio or video). Across image, audio, and video datasets, we demonstrate that the proposed framework obtains competitive model performance with standard (latent-representation) models while providing controllable model summaries via text. The model also outperforms TEARS, a recent baseline for control
    
[^26]: DOTRAG：检索时的路径推理

    DOTRAG: Retrieval-Time Reasoning Along Paths

    [https://arxiv.org/abs/2605.18760](https://arxiv.org/abs/2605.18760)

    DotRAG提出了一种轻量级、无需训练的GraphRAG检索算法，通过构建知识图谱、生成全图概览并利用图可达性索引，让LLM推理在查询时缩小搜索空间并高效恢复与任务相关的关系路径。

    

    图检索增强生成（GraphRAG）在复杂关系检索任务中面临重大挑战，尤其是在密集连接的知识图谱中，可能的路径数量会随着推理深度的增加而迅速增长。现有的GraphRAG检索算法要么依赖僵化的、预定义的启发式规则来识别相关的图上下文，要么采用成本高昂的基于大语言模型（LLM）的推理来引导图遍历。为了解决这些局限性，我们提出了DotRAG，这是一种轻量级、无需训练的GraphRAG检索算法，它利用大语言模型推理来缩小搜索空间，并恢复与任务相关的关系路径。在索引阶段，DotRAG构建实体-关系知识图谱，生成概括其内容的全图概览，并利用图可达性索引来避免代价高昂的在线图遍历。在查询阶段，LLM利用该概览来识别有潜力的图区域，并沿着路径进行检索时推理（摘要在此处被截断）。

    arXiv:2605.18760v2 Announce Type: replace-cross  Abstract: Graph Retrieval-Augmented Generation (GraphRAG) faces significant challenges in complex relational retrieval tasks, particularly in densely connected knowledge graphs where the number of possible paths grows rapidly with reasoning depth. Existing GraphRAG retrieval algorithms either rely on rigid, predefined heuristics to identify relevant graph context or employ costly LLM-based reasoning to guide graph traversal. To address these limitations, we propose DotRAG, a lightweight, training-free GraphRAG retrieval algorithm that uses large language model (LLM) reasoning to narrow the search space and recover task-relevant relational paths. During indexing, DotRAG constructs an entity-relation knowledge graph, generates a graph-wide overview summarizing its contents, and leverages graph reachability indexing to avoid expensive online graph traversals. At query time, the LLM uses this overview to identify promising graph regions and 
    
[^27]: 面向检索增强生成的公平性感知检索优化

    Fairness-Aware Retrieval Optimization for Retrieval-Augmented Generation

    [https://arxiv.org/abs/2605.15790](https://arxiv.org/abs/2605.15790)

    该论文提出三阶段框架FARO，通过概率重排序器控制各位置的群体构成，并利用位置加权线性模型将检索形式化为相关性与输出偏差权衡的优化问题，实现top-k RAG中的公平性感知检索。

    

    检索增强生成（RAG）通过外部知识为大语言模型（LLM）提供依据，但检索到的文档可能会系统性地使生成的答案偏向某一群体。在top-k设置中，由于不同上下文位置上的多个文档共同塑造输出，这种效应难以分析和控制。我们提出了一个用于top-k RAG公平性感知检索的三阶段框架。首先，一个概率重排序器可以设定检索上下文中每个位置的群体构成，从而在不修改检索器的情况下实现对检索偏差的可控扰动。其次，我们证明输出偏差可以很好地用位置级检索偏差的位置加权线性函数来近似，揭示了模型特有的位置敏感性。第三，我们利用该模型将检索形式化为一个在相关性与预测输出偏差之间进行权衡的优化问题，并提出了FARO这一可扩展的解决方案（摘要在此处截断）。

    arXiv:2605.15790v2 Announce Type: replace-cross  Abstract: Retrieval-Augmented Generation (RAG) grounds large language models (LLMs) in external knowledge, but the retrieved documents can systematically skew generated answers toward one group. This effect is difficult to analyze and control in top-k settings, where several documents at different context positions jointly shape the output. We present a three-stage framework for fairness-aware retrieval in top-k RAG. First, a probabilistic reranker sets the group composition of the retrieved context at each position, enabling controlled perturbation of retrieval bias without modifying the retriever. Second, we show that output bias is well approximated by a position-weighted linear function of position-wise retrieval bias, revealing model-specific positional sensitivities. Third, we use this model to formulate retrieval as an optimization problem that trades off relevance against predicted output bias, and propose FARO, a scalable soluti
    
[^28]: ASPIRE：通过自适应滤波器学习让谱图协同过滤重振雄风

    ASPIRE: Make Spectral Graph Collaborative Filtering Great Again via Adaptive Filter Learning

    [https://arxiv.org/abs/2604.22549](https://arxiv.org/abs/2604.22549)

    本文提出ASPIRE框架，通过双层优化目标学习自适应图滤波器，克服了传统推荐目标偏差导致的“低频爆炸”问题，使谱图协同过滤达到与精心手工设计的滤波器相当的性能。

    

    图滤波器设计是谱协同过滤的核心，然而大多数现有方法依赖于手动调整的超参数，而非完全可学习的滤波器。我们证明这一挑战源于传统推荐目标中存在的偏差，该偏差引发了一种被称为“低频爆炸”的谱现象，从而从根本上阻碍了图滤波器的有效学习。为克服这一局限性，我们提出了一种基于双层优化目标的新型自适应谱图协同过滤框架（ASPIRE）。在我们的理论分析指导下，我们对滤波器学习目标进行了解耦，进而在实践中带来了卓越的推荐性能、谱自适应性以及训练稳定性。大量实验表明，我们学习到的滤波器能够匹敌精心设计的任务特定滤波器的性能。此外，ASPIRE在基于大语言模型（LLM）的协同过滤中同样有效……

    arXiv:2604.22549v2 Announce Type: replace  Abstract: Graph filter design is central to spectral collaborative filtering, yet most existing methods rely on manually tuned hyperparameters rather than fully learnable filters. We show that this challenge stems from a bias in traditional recommendation objectives, which induces a spectral phenomenon termed low-frequency explosion, thereby fundamentally hindering the effective learning of graph filters. To overcome this limitation, we propose a novel adaptive spectral graph collaborative filtering framework (ASPIRE) based on a bi-level optimization objective. Guided by our theoretical analysis, we disentangle the filter learning objective, which in turn leads to excellent recommendation performance, spectral adaptivity, and training stability in practice. Extensive experiments show our learned filters match the performance of carefully engineered task-specific designs. Furthermore, ASPIRE is equally effective in LLM-powered collaborative fil
    
[^29]: OmniReview：面向现实审稿人推荐的大规模基准数据集与大语言模型增强框架

    OmniReview: A Large-scale Benchmark and LLM-enhanced Framework for Realistic Reviewer Recommendation

    [https://arxiv.org/abs/2602.08896](https://arxiv.org/abs/2602.08896)

    该论文提出了OmniReview，通过构建包含202,756条验证审稿记录的大规模基准数据集、三层分级评估框架以及大语言模型增强方法，解决了审稿人推荐中数据稀缺、评估指标过于简化以及嵌入方法信息瓶颈与可解释性不足的问题。

    

    学术同行评审仍然是学术验证的基石，然而该领域在数据和方法方面面临诸多挑战。从数据角度来看，现有研究受制于大规模、经过验证的基准数据集的稀缺，以及无法反映真实编辑工作流程的过度简化的评估指标。为弥补这一差距，我们提出了OmniReview，这是一个通过整合多源学术平台构建的综合数据集，经由消歧处理管道涵盖全面的学者档案，最终产出202,756条经过验证的审稿记录。基于这些数据，我们引入了一个三层分级评估框架，从召回层面到精确的专家识别层面来评估推荐效果。从方法角度来看，现有的基于嵌入的方法存在语义压缩导致的信息瓶颈以及可解释性有限的问题。为解决这些方法上的局限性，我们提出了Profi（原文摘要在此处截断）

    arXiv:2602.08896v2 Announce Type: replace-cross  Abstract: Academic peer review remains the cornerstone of scholarly validation, yet the field faces some challenges in data and methods. From the data perspective, existing research is hindered by the scarcity of large-scale, verified benchmarks and oversimplified evaluation metrics that fail to reflect real-world editorial workflows. To bridge this gap, we present OmniReview, a comprehensive dataset constructed by integrating multi-source academic platforms encompassing comprehensive scholarly profiles through the disambiguation pipeline, yielding 202, 756 verified review records. Based on this data, we introduce a three-tier hierarchical evaluaion framework to assess recommendations from recall to precise expert identification. From the method perspective, existing embedding-based approaches suffer from the information bottleneck of semantic compression and limited interpretability. To resolve these method limitations, we propose Profi
    

