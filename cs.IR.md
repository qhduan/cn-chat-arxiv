# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Does Your Agent's Memory Survive a Model Upgrade? A Controlled Study of Memory Portability](https://arxiv.org/abs/2609.05339) | 该对照研究发现智能体记忆的可移植性取决于存储格式：固定模式知识图在模型升级后准确率几乎不变，而由模型压缩生成的自然语言笔记与原模型高度耦合，迁移后性能会大幅不对称波动。 |
| [^2] | [Students' Perception of Big Data Engineering in Higher Education Curricula: Expectations, Interest and Ethical Implications](https://arxiv.org/abs/2609.05160) | 该研究通过对42名计算机科学和生物信息学硕士生的匿名调查发现，尽管学生背景各异，大多数人出于职业发展潜力与个人热情而希望学习大数据，并期望通过实践活动提升相关知识，同时关注大数据使用的伦理问题。 |
| [^3] | [Beyond Maintenance Manual Multimodal RAG: Suggesting What Tool](https://arxiv.org/abs/2609.05116) | 该论文提出MRAG-SWAT，一种对多模态检索增强生成（MRAG）流程的扩展框架，能够在检索飞机维修程序的同时自动推荐该任务所需的手动工具和专用工具，解决了现有检索系统无法提示工具需求的痛点。 |
| [^4] | [Embedding Surgery: Localized Updates for Adaptive Ranking Correction in Dense Retrieval](https://arxiv.org/abs/2609.05110) | 本文提出“嵌入手术”方法，通过在查询时对选定文档嵌入进行基于凸优化的局部最小更新，使密集检索系统能够根据编辑反馈、用户交互或大语言模型伪标签实现自适应排序校正，而无需重建静态索引。 |
| [^5] | [Beyond Co-purchase Relation: Evolution of Complementary Recommendations at Allegro](https://arxiv.org/abs/2609.05063) | 本文提出部署在Allegro电商平台的生产级检索框架AlleCompanion，通过数据级过滤启发式方法与类别约束的双塔架构（含类别适配器）将嘈杂的共同购买行为信号转化为精确的语义互补性，从而实现更精准的互补产品推荐。 |
| [^6] | [Repeated Queries Exhaust an LLM's Brand Recommendations but Not Its Sources](https://arxiv.org/abs/2609.05059) | 重复提问相同的购买问题时，不联网检索的大语言模型会不断涌现新的品牌推荐而难以饱和，启用检索的引擎则快速封闭品牌列表，但所有引擎引用的来源域名始终持续增加、远未收敛。 |
| [^7] | [Leveraging Low-Level Symbolic Competences for Unsupervised Grounding in Hallucination Detection](https://arxiv.org/abs/2609.05025) | 本文提出让LLM根据参考文档构建SQL数据库，并将其作为无监督幻觉检测的接地依据，通过这种神经符号化方法在RAGTruth和DiaHalu数据集上超越了直接预测并与最先进方法相媲美。 |
| [^8] | [A Tree-based RAG Framework for Evidence-Intensive QA via Adaptive Planning and Topology-Aware Evidence Gathering](https://arxiv.org/abs/2609.04981) | 本文提出APT-RAG框架，通过自适应规划动态扩展推理树结构，并结合拓扑感知的证据收集机制（兄弟节点证据复用、直接检索与子节点证据聚合），有效解决了证据密集型问答中结构僵化和证据整合困难的问题。 |
| [^9] | [SAM-D2Q: Aligning Multimodal Doc2Query with Search Demand and Conversion for E-commerce](https://arxiv.org/abs/2609.04961) | 提出SAM-D2Q框架，通过任务自适应多模态监督微调与多模态数据增强，在布尔检索约束下将多模态Doc2Query与电商搜索需求和转化目标对齐，有效解决查询与商品标题之间的词汇不匹配问题。 |
| [^10] | [AtomRec: Evolving Atomic Memory for Agentic Recommendation](https://arxiv.org/abs/2609.04882) | AtomRec 提出一种可演化的原子协同记忆机制，将用户和物品记忆表示为结构化原子单元并通过语义链接相连，在推荐时以多跳证据路径进行检索，从而保留细粒度偏好演变并提供可解释、有依据的推荐。 |
| [^11] | [Personalized Task Dependency Graphs for Mitigating Signal Erosion in Multi-Task Recommendation](https://arxiv.org/abs/2609.04862) | 提出个性化任务依赖图（PTDG），通过低秩近似为每个商品动态调整任务依赖路径强度，在保留因果约束的同时缓解多任务推荐中的信号侵蚀，提升稀疏深漏斗目标的表现。 |
| [^12] | [Inventory-Grounded Policy-Level Optimization for Training-Free AI Search](https://arxiv.org/abs/2609.04813) | IGPO提出了一种免训练的库存引导策略级优化方法，通过将决策策略与库存环境事实分离，使固定的AI搜索流水线无需重新训练或微调，就能适应频繁更新的产品目录。 |
| [^13] | [VizIt: A multi-view framework for exploring single-cell, spatial, and genetic data online](https://arxiv.org/abs/2609.04658) | VizIt是一个开源多视图框架，支持在线无缝探索单细胞、空间转录组、表观基因组和遗传数据，并通过帕金森病细胞图谱加以展示。 |
| [^14] | [CAGE: Coherence-Aware Graph Encoding for Retrieval-Augmented Generation](https://arxiv.org/abs/2609.04647) | CAGE提出了一种连贯性感知的图编码重排序框架，通过有向异构图建模检索段落间在领域相关性、抗噪性、信息联结和事实一致性四个维度的块间连贯性，显著提升了RAG系统在多跳问答任务上的检索与生成效果。 |
| [^15] | [Latent-Aligned Reasoning for Multimodal Recommendation](https://arxiv.org/abs/2609.04645) | 提出LARK两阶段潜空间推理框架，通过可学习token与冻结视觉编码器对齐及物品对比学习，解决多模态推荐中VLM多步推理导致的视觉与文本信号衰减（跨模态稀释）问题。 |
| [^16] | [MURAL: Multimodal Uncertainty-aware Recommendation via Adaptive edge Learning](https://arxiv.org/abs/2609.04574) | MURAL提出统一框架，通过可微分的自适应边学习器动态发现潜在物品关联并引入不确定性感知机制，解决了多模态推荐中静态图结构僵化和噪声模态信号融合导致的两大瓶颈。 |
| [^17] | [BioSync: Transformer-Based Cross-Modal Fusion for a Multimodal Physiological Digital Biomarker](https://arxiv.org/abs/2609.04504) | BioSync提出了一种基于Transformer多头自注意力的跨模态融合框架，将可穿戴设备采集的心脏、神经、行为和语音数据整合为符合BEST框架的连续型复合数字生物标志物——BioSync指数（BSI），在认知衰退检测的合成队列上达到0.928的AUC。 |
| [^18] | [Evaluation of Phonetic Encoding Algorithms on Transcription Datasets](https://arxiv.org/abs/2609.04391) | 提出了一种基于Hüllermeier-Rifqi指数的新型评估方案，通过计算语音编码与IPA真实转录之间的成对相似度差异，并结合碰撞率评估多种语音编码算法在多语言转录数据集上的性能。 |
| [^19] | [Corporate-Family Resolution Is Not a String-Matching Problem: A Public Benchmark Stratified by Name Visibility](https://arxiv.org/abs/2609.04269) | 该论文发布了CorpFam——一个基于美国联邦奖励记录中供应商自报母公司数据构建的企业家族消解公开基准，通过按名称可见性分层评估并报告分层召回率，揭示出最强匹配方法对名称不可见的企业家族对识别率仅4.2%，证明企业家族消解不能被简化为字符串匹配问题。 |
| [^20] | [SAGE: Semantic Attribute Graphs for Multi-Entity Visual Retrieval](https://arxiv.org/abs/2609.04255) | 提出无需训练的SAGE框架，通过将密集文档图像中的语义实体解析为带多向量嵌入的层次图并进行迭代实体级子图匹配，解决单向量编码导致的“语义稀释”问题，并发布了用于多实体细粒度视觉检索的DEAR数据集。 |
| [^21] | [UniCon: A Unified Context-Centric Modeling Paradigm for CTR Prediction](https://arxiv.org/abs/2609.03290) | 本文提出UniCon，一种统一的以上下文为中心的CTR预测建模范式，将用户历史行为与当前请求作为同质的上下文单元进行统一建模，克服传统异构信号划分的局限，提升扩展效率与预测质量。 |
| [^22] | [hLLM: Single Pass Decoding for Generative Reranking](https://arxiv.org/abs/2609.01807) | 提出hLLM，通过轻量自注意力头从LLM预填充隐状态读取项目-位置得分矩阵，并用匈牙利算法求最优二分匹配，在O(1)次前向传播内一次性解码全部N个序数，从而将生成式重排序的解码从逐token自回归生成变为常数次前向传播，且天然保证输出为有效排序。 |
| [^23] | [InsightToast: Proactive Information Retrieval & Glanceable Visualization in the Side Channel of Data-Rich Meetings](https://arxiv.org/abs/2608.31115) | InsightToast通过多智能体LLM与RAG管道实时监测会议话语并主动检索信息，以临时提示和一目了然图表的形式在会议侧信道中提供有据可依的洞察，从而避免任务切换对会议参与和决策的干扰。 |
| [^24] | [E-SENS: Exclusion-Sensitive Penalization for Negative-Constraint Retrieval](https://arxiv.org/abs/2608.30130) | E-SENS是一种无需训练的重排序方法，通过为被排除概念提取“陷阱查询”并从检索分数中减去其相似度，有效惩罚与用户排除概念相关的文档，从而提升检索系统对负向约束的遵守能力。 |
| [^25] | [An Event is Worth One Token: Event Tokenization for Industrial-scale LLM Recommendation](https://arxiv.org/abs/2608.25546) | 本文提出了一种以事件为中心的自回归推荐方法AMBER，通过将每个交互的完整时间快照压缩为事件令牌，显著提高了快照分辨率，从而增强了LLM推荐系统的性能和扩展性。 |
| [^26] | [Nepali Passport Question Answering: A Low-Resource Dataset for Public Service Applications](https://arxiv.org/abs/2603.13320) | 该研究构建了首个面向护照公共服务的尼泊尔语问答低资源数据集，通过微调Transformer嵌入模型并结合BM25进行混合检索，其中基于多语言E5嵌入的模型取得了最佳检索性能。 |
| [^27] | [Graph Foundation Models for Recommendation: A Comprehensive Survey](https://arxiv.org/abs/2502.08346) | 该综述首次全面梳理了图基础模型（GFM）在推荐系统中的应用，提出了现有方法的清晰分类体系，深入剖析了融合图神经网络与大语言模型优势的技术细节，并指出了该领域的关键挑战与未来研究方向。 |

# 详细

[^1]: 你的智能体记忆能在模型升级中幸存吗？一项关于记忆可移植性的对照研究

    Does Your Agent's Memory Survive a Model Upgrade? A Controlled Study of Memory Portability

    [https://arxiv.org/abs/2609.05339](https://arxiv.org/abs/2609.05339)

    该对照研究发现智能体记忆的可移植性取决于存储格式：固定模式知识图在模型升级后准确率几乎不变，而由模型压缩生成的自然语言笔记与原模型高度耦合，迁移后性能会大幅不对称波动。

    

    模型升级是常规操作，而记忆迁移却不是。即使智能体保留相同的记忆存储，仍可能发生遗忘：新模型可能以不同方式解读旧笔记，混合的嵌入版本可能破坏检索，且在缺乏原始证据时修复可能失败。我们在保留相同历史记录的前提下，比较了四种记忆存储方式：原文完整保留用于长上下文阅读（LC-RAW）、分块用于检索增强生成（RAG）、由模型压缩为自然语言笔记（NOTES）、或规范化为固定模式知识图（KG-fixed）。该研究使用了48个带有随机化答案代码的合成历史记录、精确评分机制，以及两个参数量低于100亿的开源权重模型。我们的测量结果表明，固定模式结构可以可靠地迁移——在写入器（模型）更换后，KG-fixed的准确率仅变化 $+0.0004 \pm 0.0020$。相反，压缩的NOTES表现出高度的模型耦合性，其准确率以不对称的方式变化，偏移量达 $+9.91$ 或 $-13

    arXiv:2609.05339v1 Announce Type: new  Abstract: Model upgrades are routine; memory migrations are not. An agent can keep the same memory store and still forget: a new model may interpret old notes differently, mixed embedding versions may break retrieval, and repair may fail without the original evidence. We compare memory as the same history is preserved verbatim for long-context reading (LC-RAW), divided into chunks for retrieval-augmented generation (RAG), compressed by a model into natural-language notes (NOTES), or normalized into a fixed-schema knowledge graph (KG-fixed). The study uses 48 synthetic histories with randomized answer codes, exact scoring, and two open-weight models with sub 10 billion parameters.   Our measurements show that fixed-schema structures transfer reliably, with KG-fixed accuracy changing by only $+0.0004 \pm 0.0020$ following a writer swap. Conversely, compressed NOTES exhibit high model coupling, with accuracy shifting asymmetrically by $+9.91$ or $-13
    
[^2]: 高等教育课程中学生对大数据工程的认知：期望、兴趣与伦理影响

    Students' Perception of Big Data Engineering in Higher Education Curricula: Expectations, Interest and Ethical Implications

    [https://arxiv.org/abs/2609.05160](https://arxiv.org/abs/2609.05160)

    该研究通过对42名计算机科学和生物信息学硕士生的匿名调查发现，尽管学生背景各异，大多数人出于职业发展潜力与个人热情而希望学习大数据，并期望通过实践活动提升相关知识，同时关注大数据使用的伦理问题。

    

    该研究调查了学生对融入硕士课程的大数据工程课程的兴趣和期望，以及使用大数据的伦理影响。研究对选修面向计算机科学和生物信息学硕士项目大数据课程的67名学生中的42名进行了匿名在线调查。研究采用主题分析法对调查结果进行分析和解读，突出了与学生期望、兴趣以及他们对从事大数据工作的伦理影响的看法相关的有趣方面。研究结论表明，尽管学生背景存在显著差异，但大多数学生出于与职业发展潜力相关的实际原因以及对该领域的热情等个人原因，对学习大数据感兴趣。学生表达的主要期望是通过实践活动来增强他们与大数据相关的知识。

    arXiv:2609.05160v1 Announce Type: cross  Abstract: The study investigates students' interest and expectations in a Big Data Engineering course integrated with a Master curricula, as well as ethical implications of using Big Data. An anonymous online survey was conducted with 42 of the 67 students enrolled in the Big Data course offered to Computer Science and Bioinformatics Master's programs. The responses were analyzed and interpreted using thematic analysis, highlighting interesting aspects related to students' expectations, interest, and their perspective of the ethical implications of working with Big Data. The study concludes that, even though there is significant difference in students' background, the majority are interested in learning Big Data, for practical and personal reasons related to the potential for career growth and their passion for the field. The main expectation expressed is related to enhancing their knowledge related to Big Data via practical activities. All stud
    
[^3]: 超越维修手册的多模态RAG：推荐所需工具

    Beyond Maintenance Manual Multimodal RAG: Suggesting What Tool

    [https://arxiv.org/abs/2609.05116](https://arxiv.org/abs/2609.05116)

    该论文提出MRAG-SWAT，一种对多模态检索增强生成（MRAG）流程的扩展框架，能够在检索飞机维修程序的同时自动推荐该任务所需的手动工具和专用工具，解决了现有检索系统无法提示工具需求的痛点。

    

    飞机技术人员几乎在执行每项任务时都必须查阅维修手册（MM），而在数百页的手册中定位相关程序仍然十分耗时。多模态检索增强生成（MRAG）被提出来解决这一问题，使技术人员能够通过自然语言查询检索维修程序及其附带的插图。然而，仅靠检索并不能告诉技术人员该任务需要哪些工具。维修手册只有在执行到相应步骤时才会标明专用工具，而且完全不说明手动工具的要求；为了选择手动工具，技术人员需要从图解零件目录（IPC）中查找硬件尺寸信息，并据此推断出正确的工具。因此，我们提出了MRAG-SWAT，这是MRAG流程的一种扩展，它在返回检索到的维修程序的同时，还返回所需的手动工具和专用工具。该框架已针对Ly（原文在此截断）...

    arXiv:2609.05116v1 Announce Type: cross  Abstract: Aircraft technicians are required to consult the maintenance manual (MM) for nearly every task, and locating the relevant procedure across hundreds of pages remains time-consuming. Multimodal retrieval augmented generation (MRAG) has been proposed to address this, allowing technicians to retrieve procedures, together with the accompanying figures, through natural-language queries. However, retrieval alone does not tell the technicians which tools the task requires. The MM identifies special tools only when the corresponding step is reached, and it does not state hand tool requirements at all; to select hand tools, technicians are required to find the hardware dimension from the illustrated parts catalog (IPC) and infer the right tool from it. We therefore propose MRAG-SWAT, an extension of the MRAG pipeline that returns the required hand tools and special tools alongside the retrieved procedure. The framework was implemented for the Ly
    
[^4]: 嵌入手术：面向密集检索自适应排序校正的局部化更新

    Embedding Surgery: Localized Updates for Adaptive Ranking Correction in Dense Retrieval

    [https://arxiv.org/abs/2609.05110](https://arxiv.org/abs/2609.05110)

    本文提出“嵌入手术”方法，通过在查询时对选定文档嵌入进行基于凸优化的局部最小更新，使密集检索系统能够根据编辑反馈、用户交互或大语言模型伪标签实现自适应排序校正，而无需重建静态索引。

    

    密集检索系统是现代搜索引擎、推荐平台以及检索增强生成流水线的核心组件。它们将文档和查询编码为稠密嵌入，通过向量相似度实现高效的语义搜索。然而，由于文档嵌入是在离线阶段计算并存储于静态索引中的，这些系统难以适应用户反馈或不断演变的搜索意图。为解决这一局限，我们提出了“嵌入手术”（embedding surgery），一种用于密集检索中自适应排序校正的轻量级方法。该方法在查询时对选定的文档嵌入施加局部化、最小化的更新，其依据可以是编辑反馈、用户交互或来自大语言模型的伪标签。我们将嵌入手术表述为一个凸优化问题，在满足排序约束的同时，最小化对受影响文档表示的修改。我们将嵌入手术集成……（原文在此截断）

    arXiv:2609.05110v1 Announce Type: new  Abstract: Dense retrieval systems are core components of modern search engines, recommendation platforms, and retrieval-augmented generation pipelines. They encode documents and queries into dense embeddings, enabling efficient semantic search via vector similarity. However, because document embeddings are computed offline and stored in static indexes, these systems struggle to adapt to user feedback or evolving search intent. To address this limitation, we introduce \emph{embedding surgery}, a lightweight approach for adaptive ranking correction in dense retrieval. The method applies localized, minimal updates to selected document embeddings at query time, guided by editorial feedback, user interactions, or pseudo-labels from large language models. We formulate embedding surgery as a convex optimization problem that enforces ranking constraints while minimizing modifications to the affected document representations. We integrate embedding surgery
    
[^5]: 超越共同购买关系：Allegro平台互补推荐的演进

    Beyond Co-purchase Relation: Evolution of Complementary Recommendations at Allegro

    [https://arxiv.org/abs/2609.05063](https://arxiv.org/abs/2609.05063)

    本文提出部署在Allegro电商平台的生产级检索框架AlleCompanion，通过数据级过滤启发式方法与类别约束的双塔架构（含类别适配器）将嘈杂的共同购买行为信号转化为精确的语义互补性，从而实现更精准的互补产品推荐。

    

    当客户将一台专业相机加入购物车时，系统应该推荐匹配的镜头、通用的三脚架，还是另一台相机机身？互补产品推荐对于构建完整的购物篮至关重要，然而标准模型往往无法区分那些仅仅是被一起购买的商品与真正能够协同使用的商品。在本文中，我们提出了AlleCompanion：一个部署在Allegro.com的生产级检索框架，它将嘈杂的行为信号转化为精确的语义兼容性。我们通过将数据级过滤启发式方法与类别约束的双塔架构相结合，来缓解大规模共同购买流量中固有的噪声。在该框架内，类别适配器在嵌入空间中引导模型，将候选商品约束在逻辑上互补的边界之内。由于大规模建模真实用户行为本身就十分困难，我们引入了ComCat，一个……

    arXiv:2609.05063v1 Announce Type: cross  Abstract: When a customer adds a professional camera to their cart, should the system suggest a matching lens, a generic tripod, or another camera body? Complementary Product Recommendation is vital for comprehensive basket building, yet standard models often fail to distinguish between items that are merely bought together and those that truly work together. In this paper, we present AlleCompanion: a production-scale retrieval framework deployed at Allegro.com that transforms noisy behavioural signals into precise semantic compatibility. We mitigate the intrinsic noise in large-scale co-purchase traffic by combining data-level filtering heuristics with a category-constrained Two Tower architecture. Within this framework, the Category Adapter guides the model in the embedding space, constraining candidates within logically complementary boundaries. Since modelling authentic user behaviour at scale is inherently difficult, we introduce ComCat, a 
    
[^6]: 重复提问会耗尽大语言模型的品牌推荐，却耗不尽其引用来源

    Repeated Queries Exhaust an LLM's Brand Recommendations but Not Its Sources

    [https://arxiv.org/abs/2609.05059](https://arxiv.org/abs/2609.05059)

    重复提问相同的购买问题时，不联网检索的大语言模型会不断涌现新的品牌推荐而难以饱和，启用检索的引擎则快速封闭品牌列表，但所有引擎引用的来源域名始终持续增加、远未收敛。

    

    重复提出相同的购买类问题是否会耗尽语言模型的品牌推荐，取决于其是否具备检索能力。在涵盖300个“问题-引擎”组合单元的实验中（50个问题、6个引擎、每个单元运行15次，并对1,470个经人工裁定的组织进行开放式抽取），五个不借助网络搜索作答的引擎在86-92%的单元中到第15次运行时仍在出现从未见过的品牌，其品牌储备中位数为15-31个组织；而唯一启用检索功能的引擎则封闭了其品牌列表（中位数8个组织，64%的单元仍在新增），这与此前四个深度实验单元中启用网络搜索的运行在第10次左右即趋于饱和的结果相符。在所有测试的时间范围内，被引用域名的累积量持续上升：四个深度实验单元在第24次运行时仍在新增域名，仅观测到Chao2下限估计值的59-84%，检索型引擎的广度单元中也有44%在第15次运行时仍在新增域名。单次运行仅能覆盖五次运行品牌集合的62-77%，且跨引擎来看，每个问题的中位数可引出38个组织……（摘要原文在此处截断）

    arXiv:2609.05059v1 Announce Type: cross  Abstract: Whether repeated identical buying questions exhaust a language model's brand recommendations depends on retrieval. Across 300 question-engine cells (50 questions, six engines, 15 runs each, open extraction over 1,470 adjudicated organizations), the five engines answering without web search were still adding never-seen brands at run 15 in 86-92% of cells, with median repertoires of 15-31 organizations; the one retrieval-enabled engine closed its list (median 8 organizations, 64% of cells still adding), matching four earlier deep cells where web-search runs saturated by run ten. Cited-domain accumulation keeps rising at every horizon tested: four deep cells were still adding domains at run 24 with 59-84% of the Chao2 lower-bound estimate observed, and 44% of the retrieval engine's breadth cells were still adding domains at run 15. A single run shows 62-77% of the five-run brand set, and across engines the median question draws 38 organiz
    
[^7]: 利用低层次符号能力实现无监督的幻觉检测接地

    Leveraging Low-Level Symbolic Competences for Unsupervised Grounding in Hallucination Detection

    [https://arxiv.org/abs/2609.05025](https://arxiv.org/abs/2609.05025)

    本文提出让LLM根据参考文档构建SQL数据库，并将其作为无监督幻觉检测的接地依据，通过这种神经符号化方法在RAGTruth和DiaHalu数据集上超越了直接预测并与最先进方法相媲美。

    

    幻觉——即语言模型生成事实上错误或缺乏来源支持的内容——是提示式和微调语言模型面临的重大挑战。由于大语言模型（LLM）的推理过程不透明，往往难以解释模型的输出为何可能不准确，因此检测幻觉十分困难。在本工作中，我们研究LLM能否使用一种替代性的、低层次的符号能力（如SQL）来在某个高层次任务中进行无监督的幻觉检测。为此，我们让LLM根据参考文档构建一个SQL数据库。该SQL数据库随后被用于在以数据库为依据的幻觉检测流程中，对参考内容和采样响应进行推理，从而提供一种神经符号化的检查。在RAGTruth和DiaHalu幻觉检测数据集上，我们发现我们的方法优于直接预测，并可与最先进的方法相媲美。

    arXiv:2609.05025v1 Announce Type: cross  Abstract: Hallucination-where a language model generates outputs that are factually incorrect or unsupported by the source-is a major challenge for both prompted and fine-tuned language models. Detecting hallucinations is difficult due to the opaque reasoning processes of LLMs, which often provide little insight into why a model's output may be inaccurate.   In this work, we investigate whether an LLM can use an alternative, low level, symbolic competence such as SQL for unsupervised hallucination detection in some high level task. For this, we make an LLM build an SQL database from reference documents. This SQL database is then used for reasoning over the reference and the sampled response in a hallucination detection pipeline that is grounded in the database, thereby providing a neurosymbolic checkup.   On RAGTruth and DiaHalu hallucination detection datasets, we find that our approach improves on direct prediction and competes with state-of-t
    
[^8]: 一种通过自适应规划与拓扑感知证据收集实现证据密集型问答的树状RAG框架

    A Tree-based RAG Framework for Evidence-Intensive QA via Adaptive Planning and Topology-Aware Evidence Gathering

    [https://arxiv.org/abs/2609.04981](https://arxiv.org/abs/2609.04981)

    本文提出APT-RAG框架，通过自适应规划动态扩展推理树结构，并结合拓扑感知的证据收集机制（兄弟节点证据复用、直接检索与子节点证据聚合），有效解决了证据密集型问答中结构僵化和证据整合困难的问题。

    

    近年来的结构化RAG方法利用树状或图状推理结构来改进多跳问答。然而，它们在证据密集型问答中面临关键局限——这类任务中回答一个问题需要综合散布在数十甚至数百份文档中的信息：一是结构僵化，限制了推理结构的自适应扩展；二是证据收集过程忽略拓扑结构，阻碍了跨不同推理节点的证据有效整合。为解决这些问题，我们提出了APT-RAG，一个自适应规划与拓扑感知证据收集的RAG框架。自适应规划根据问题依赖关系和证据需求动态扩展推理结构，而拓扑感知证据收集通过兄弟节点证据复用、直接检索以及从子节点聚合证据来提升证据覆盖率。我们进一步引入证据引导的批量答案生成，以显著减少……

    arXiv:2609.04981v1 Announce Type: new  Abstract: Recent structured RAG methods leverage tree- or graph-based reasoning structures to improve multi-hop QA. However, they face key limitations in evidence-intensive QA, where answering a question requires synthesizing information scattered across dozens or even hundreds of documents: structural rigidity, which limits adaptive reasoning expansion, and topology-ignorant evidence gathering, which prevents effective integration of evidence across different reasoning nodes. To address these issues, we propose APT-RAG, an Adaptive Planning and Topology-aware evidence gathering RAG framework. Adaptive planning dynamically expands the reasoning structure based on question dependencies and evidence requirements, while topology-aware evidence gathering improves evidence coverage through sibling evidence reuse, direct retrieval, and evidence aggregation from child nodes. We further introduce evidence-guided batched answer generation to reduce signifi
    
[^9]: SAM-D2Q：面向电商场景的多模态Doc2Query与搜索需求及转化的对齐

    SAM-D2Q: Aligning Multimodal Doc2Query with Search Demand and Conversion for E-commerce

    [https://arxiv.org/abs/2609.04961](https://arxiv.org/abs/2609.04961)

    提出SAM-D2Q框架，通过任务自适应多模态监督微调与多模态数据增强，在布尔检索约束下将多模态Doc2Query与电商搜索需求和转化目标对齐，有效解决查询与商品标题之间的词汇不匹配问题。

    

    电商搜索常常面临用户查询与商家撰写的产品标题之间的词汇不匹配问题，因为简短的标题无法完全覆盖多样的用户表达方式或产品的视觉属性。尽管Doc2Query通过为文档生成伪查询进行扩展来缓解这一问题，但传统方法仅基于文本，且未针对电商业务目标进行优化。因此，这些方法可能产生语义上合理但商业上无效的扩展，并遗漏产品图像中包含的关键属性。为此，我们提出了电商搜索对齐的多模态Doc2Query（SAM-D2Q），这是一个在布尔检索约束下、面向电商搜索的业务对齐多模态文档扩展框架。SAM-D2Q包含三个阶段：（1）任务自适应的多模态监督微调，以增强对产品标题、图像和用户查询的视觉-语言理解能力；（2）多模态数据增强，以提升……（原文摘要不完整，到此截断）

    arXiv:2609.04961v1 Announce Type: new  Abstract: E-commerce search often suffers from vocabulary mismatch between user queries and merchant-authored product titles, since short titles cannot fully cover diverse user expressions or visual product attributes. Although Doc2Query alleviates this issue by generating pseudo-queries for document expansion, traditional methods are text-only and not optimized for e-commerce business objectives. As a result, they may produce semantically plausible but commercially ineffective expansions and miss key attributes present in product images. To this end, we propose E-commerce Search-Aligned Multimodal Doc2Query (SAM-D2Q), a business-aligned multimodal document expansion framework for e-commerce search under Boolean retrieval constraints. SAM-D2Q consists of three stages: (1) task-adapted multimodal supervised fine-tuning to enhance vision-language understanding of product titles, images, and user queries; (2) multimodal data augmentation to improve p
    
[^10]: AtomRec：面向智能体推荐的可演化原子记忆

    AtomRec: Evolving Atomic Memory for Agentic Recommendation

    [https://arxiv.org/abs/2609.04882](https://arxiv.org/abs/2609.04882)

    AtomRec 提出一种可演化的原子协同记忆机制，将用户和物品记忆表示为结构化原子单元并通过语义链接相连，在推荐时以多跳证据路径进行检索，从而保留细粒度偏好演变并提供可解释、有依据的推荐。

    

    智能体推荐系统利用大语言模型来维护语义记忆并支持基于证据的推荐。然而，现有的记忆机制通常将用户和物品信息压缩为粗粒度的摘要，并通过标量的协同链接将其连接起来，导致随着用户兴趣的演变，难以保留细粒度的偏好阶段，也难以检索可解释的证据。我们提出 AtomRec，一种具有可演化原子协同记忆的智能体推荐系统。AtomRec 将用户和物品记忆表示为结构化的原子单元，在相关记忆之间构建语义链接，并在新交互到来时演化相关的历史字段。在推荐过程中，它以多跳证据路径的形式检索相互关联的记忆，而非孤立的邻居摘要，从而使协同信号能够支持有依据的排序。在四个公开基准数据集上的实验表明，AtomRec 持续优于……（原文此处截断）

    arXiv:2609.04882v1 Announce Type: new  Abstract: Agentic recommender systems use large language models to maintain semantic memory and support evidence-aware recommendation. However, existing memory mechanisms often compress user and item information into coarse summaries and connect them with scalar collaborative links, making it difficult to preserve fine-grained preference stages or retrieve interpretable evidence as user interests evolve. We propose \textsc{AtomRec}, an agentic recommender with evolving atomic collaborative memory. \textsc{AtomRec} represents user and item memories as structured atomic units, builds semantic links across related memories, and evolves related historical fields when new interactions arrive. During recommendation, it retrieves linked memories as multi-hop evidence paths rather than isolated neighbor summaries, allowing collaborative signals to support grounded ranking. Experiments on four public benchmarks show that \textsc{AtomRec} consistently outpe
    
[^11]: 用于缓解多任务推荐中信号侵蚀问题的个性化任务依赖图

    Personalized Task Dependency Graphs for Mitigating Signal Erosion in Multi-Task Recommendation

    [https://arxiv.org/abs/2609.04862](https://arxiv.org/abs/2609.04862)

    提出个性化任务依赖图（PTDG），通过低秩近似为每个商品动态调整任务依赖路径强度，在保留因果约束的同时缓解多任务推荐中的信号侵蚀，提升稀疏深漏斗目标的表现。

    

    优化多个转化目标是工业推荐系统中的核心挑战，但常受限于僵化架构中的信号侵蚀问题。现有多任务学习（MTL）方法通常在静态转化漏斗上强制执行统一的依赖强度，忽视了任务关联性会随商品特征而自然变化的特性。沿着这些固定链路进行分层消息传递会导致累积性的信号衰减，从而降低稀疏、深漏斗目标上的性能。为解决这一问题，我们提出了个性化任务依赖图。在遵循必要的物理因果约束（如点击 -> 支付）的同时，PTDG 通过低秩近似为每个商品动态地“重构”依赖路径的强度，以确保结构上的鲁棒性。我们实现了基于 GCN 的传播机制，并结合硬因果掩码来建立自适应的信息捷径。此外，我们还引入了自适应渐进……

    arXiv:2609.04862v1 Announce Type: new  Abstract: Optimizing multiple conversion objectives is a core challenge in industrial recommendation, often limited by signal erosion in rigid architectures. Existing Multi-Task Learning (MTL) methods typically enforce uniform dependency strengths across a static conversion funnel, overlooking how task correlations naturally vary based on item characteristics. Hierarchical message passing along these fixed chains leads to cumulative signal attenuation, which degrades performance on sparse, deep-funnel objectives. To address this, we propose the Personalized Task Dependency Graphs (PTDG). While respecting necessary physical causal constraints (e.g., Click -> Pay), PTDG dynamically "rewires" the intensity of dependency pathways for each item via low-rank approximation to ensure structural robustness. We implement a GCN-based propagation with hard causal masking to establish adaptive information shortcuts. Additionally, we introduce an Adaptive Progr
    
[^12]: 面向免训练AI搜索的库存引导策略级优化

    Inventory-Grounded Policy-Level Optimization for Training-Free AI Search

    [https://arxiv.org/abs/2609.04813](https://arxiv.org/abs/2609.04813)

    IGPO提出了一种免训练的库存引导策略级优化方法，通过将决策策略与库存环境事实分离，使固定的AI搜索流水线无需重新训练或微调，就能适应频繁更新的产品目录。

    

    arXiv:2609.04813v1 公告类型：新 摘要：在部署早期，AI搜索系统通常需要在频繁更新的产品目录上运行，因此可用商品及其属性不能被视为可以编码进固定提示词或策略中的稳定知识。微调、强化学习和静态提示词补丁都难以适用：标签稀缺，奖励随库存漂移，模型重新发布成本高昂，而提示词修复也会很快过时。我们提出了库存引导策略级优化（IGPO），这是一种面向固定AI搜索流水线的免训练方法。IGPO将策略与环境事实分离：它学习的是在运行时基于库存证据采取行动的策略指南，而非记忆可用商品。在线阶段，IGPO通过探测库存并构建库存画像来为每个查询提供事实依据，然后将相关的策略指南注入检索和选择提示词中。离线阶段，随机推演按查询分组——混合结果组……

    arXiv:2609.04813v1 Announce Type: new  Abstract: Early in deployment, an AI search system typically operates over a frequently updated product catalog, so the available items and their properties cannot be treated as stable knowledge that can be encoded in fixed prompts or strategies. Fine-tuning, reinforcement learning, and static prompt patches fit poorly: labels are scarce, rewards drift with inventory, model releases are costly, and prompt fixes quickly stale. We present Inventory-Grounded Policy-Level Optimization (IGPO), a training-free approach for fixed AI search pipelines. IGPO separates policy from environment facts: it learns Policy Guidelines for acting on runtime inventory evidence rather than memorizing available items. Online, IGPO grounds each query by probing the inventory and constructing an inventory portrait, then injects relevant Policy Guidelines into the retrieval and selection prompts. Offline, stochastic rollouts are grouped by query -- mixed outcome groups dir
    
[^13]: VizIt：一个用于在线探索单细胞、空间和遗传数据的多视图框架

    VizIt: A multi-view framework for exploring single-cell, spatial, and genetic data online

    [https://arxiv.org/abs/2609.04658](https://arxiv.org/abs/2609.04658)

    VizIt是一个开源多视图框架，支持在线无缝探索单细胞、空间转录组、表观基因组和遗传数据，并通过帕金森病细胞图谱加以展示。

    

    多组学研究日益需要从互补的生物学视角检视数据，然而交互式探索在不同数据模态和工具之间仍然处于碎片化状态。我们提出了VizIt，一个用于单细胞和空间转录组、表观基因组及遗传数据多视图探索的开源框架。VizIt连接了以基因、细胞类型、条件、空间、基因组区域和变异为中心的多种视图，实现了跨生物学视角的无缝导航。我们通过帕金森病细胞图谱展示了VizIt，这是一个可定制的交互式多组学资源。

    arXiv:2609.04658v1 Announce Type: new  Abstract: Multi-omic studies increasingly require data to be examined from complementary biological perspectives, yet interactive exploration remains fragmented across modalities and tools. We present VizIt, an open-source framework for multi-view exploration of single-cell and spatial transcriptomic, epigenomic and genetic data. VizIt connects gene-, cell type-, condition-, spatial-, genomic region- and variant-centered views, enabling seamless navigation across biological perspectives. We demonstrate VizIt through the Parkinson's Cell Atlas, a customizable interactive multi-omic resource.
    
[^14]: CAGE：面向检索增强生成的连贯性感知图编码

    CAGE: Coherence-Aware Graph Encoding for Retrieval-Augmented Generation

    [https://arxiv.org/abs/2609.04647](https://arxiv.org/abs/2609.04647)

    CAGE提出了一种连贯性感知的图编码重排序框架，通过有向异构图建模检索段落间在领域相关性、抗噪性、信息联结和事实一致性四个维度的块间连贯性，显著提升了RAG系统在多跳问答任务上的检索与生成效果。

    

    传统的检索增强生成系统独立地将每个段落与查询进行评分，由此组装的上下文集合虽然可能各自相关，但整体上缺乏连贯性。我们提出了连贯性感知图编码，这是一个重排序框架，从四个维度建模“块间连贯性”：领域内相关性、抗噪性、信息联结和事实一致性。我们的流程将检索到的段落转换为有向异构实体图，通过最小出度重加权放大事实锚点，通过关系图卷积网络编码结构模式，并将块间连贯性与查询相关性融合以进行最终排序。在四个多跳问答基准测试上的评估表明，CAGE在以桥梁型问题为主的数据集上的Recall@5指标与包括monoT5在内的强基线相当或更优，并持续提升下游精确匹配得分，证明了结构上连贯的（原文在此截断）

    arXiv:2609.04647v1 Announce Type: new  Abstract: Traditional Retrieval-Augmented Generation (RAG) systems score each passage independently against the query, assembling context sets that may be individually relevant yet collectively incoherent. We introduce Coherence-Aware Graph Encoding (CAGE), a reranking framework that models "between-chunk coherence" across four dimensions: Intra-Domain Relevance, Noise Resistance, Informational Bonding, and Factual Consistency. Our pipeline transforms retrieved passages into directed heterogeneous entity graphs, amplifies factual anchors via min-out-degree reweighting, encodes structural patterns through a Relational Graph Convolutional Network, and fuses inter-chunk coherence with query relevance for final ranking. Evaluated across four multi-hop benchmarks, CAGE matches or outperforms strong baselines including monoT5 in Recall@5 on bridge-dominated datasets and consistently improves downstream Exact Match, demonstrating that structurally cohere
    
[^15]: 面向多模态推荐的潜空间对齐推理

    Latent-Aligned Reasoning for Multimodal Recommendation

    [https://arxiv.org/abs/2609.04645](https://arxiv.org/abs/2609.04645)

    提出LARK两阶段潜空间推理框架，通过可学习token与冻结视觉编码器对齐及物品对比学习，解决多模态推荐中VLM多步推理导致的视觉与文本信号衰减（跨模态稀释）问题。

    

    多模态视觉-语言模型（VLMs）在跨模态理解方面展现出卓越的能力，但将其应用于推荐任务时仍存在一个根本性挑战：随着表示在多步推理过程中传播，视觉和文本信号会逐渐衰减——我们将这种现象称为跨模态稀释。为解决这一问题，我们提出了LARK（潜空间对齐推理框架），这是一个在单一VLM内具备互补对齐机制的两阶段潜空间推理框架。在第一阶段，可学习的潜空间token与多步思维链（CoT）推理交错进行，并与冻结的视觉编码器显式对齐，作为视觉检查点，在整个推理链中保留感知细节。在第二阶段，潜空间表示通过桥接MLP进行投影，并采用物品到物品的对比学习进行训练；为防止推理语义发生（摘要在此处截断）……

    arXiv:2609.04645v1 Announce Type: cross  Abstract: Multimodal Vision-Language Models (VLMs) have demonstrated remarkable capabilities in cross-modal understanding, yet a fundamental challenge persists when applying them to recommendation: as representations propagate through multi-step reasoning, both visual and textual signals progressively attenuate - a phenomenon we term cross-modal dilution. To address this, we propose LARK (Latent-Aligned Reasoning frameworK), a two-stage latent reasoning framework with complementary alignment mechanisms within a single VLM. In the first stage, learnable latent tokens are interleaved with multi-step chain-of-thought (CoT) reasoning and explicitly aligned with a frozen vision encoder, serving as visual checkpoints that preserve perceptual details throughout the reasoning chain. In the second stage, the latent representations are projected via a bridge MLP and trained with item-to-item contrastive learning; to prevent the reasoning semantics from fa
    
[^16]: MURAL：基于自适应边学习的多模态不确定性感知推荐

    MURAL: Multimodal Uncertainty-aware Recommendation via Adaptive edge Learning

    [https://arxiv.org/abs/2609.04574](https://arxiv.org/abs/2609.04574)

    MURAL提出统一框架，通过可微分的自适应边学习器动态发现潜在物品关联并引入不确定性感知机制，解决了多模态推荐中静态图结构僵化和噪声模态信号融合导致的两大瓶颈。

    

    多模态图神经网络通过内容特征增强稀疏的交互数据，已成为推荐系统的标准方法。然而，当前架构面临两个瓶颈：一是结构僵化，即依赖静态预计算的相似度图，无法适应不断演变的用户偏好；二是语义脆弱性，即噪声模态信号被不加区分地融合，从而扭曲了协同信号。我们提出MURAL（基于自适应边学习的多模态不确定性感知推荐），这是一个统一框架，将多模态推荐从固定的结构增强转变为动态拓扑发现。为解决结构僵化问题，自适应边学习器结合可微分的检索增强策略与近似最近邻搜索，发现既具备语义自适应性又具备计算可扩展性（O(NlogN)）的潜在物品-物品关联。为解决语义脆弱性问题，一个不确定性…

    arXiv:2609.04574v1 Announce Type: cross  Abstract: Multimodal Graph Neural Networks have become standard for recommendation by augmenting sparse interaction data with content features. Yet current architectures face two bottlenecks: structural rigidity, from a reliance on static precomputed similarity graphs that cannot adapt to evolving preferences; and semantic fragility, where noisy modality signals are indiscriminately fused, distorting the collaborative signal. We propose MURAL (Multimodal Uncertainty-aware Recommendation via Adaptive edge Learning), a unified framework that shifts multimodal recommendation from fixed structural augmentation to dynamic topology discovery. To address structural rigidity, an Adaptive Edge Learner combines a differentiable retrieval-augmented strategy with an approximate nearest neighbor search to discover latent item-item correlations that are both semantically adaptive and computationally scalable (O(NlogN)). To address semantic fragility, an Uncer
    
[^17]: BioSync：基于Transformer的跨模态融合方法，用于多模态生理数字生物标志物

    BioSync: Transformer-Based Cross-Modal Fusion for a Multimodal Physiological Digital Biomarker

    [https://arxiv.org/abs/2609.04504](https://arxiv.org/abs/2609.04504)

    BioSync提出了一种基于Transformer多头自注意力的跨模态融合框架，将可穿戴设备采集的心脏、神经、行为和语音数据整合为符合BEST框架的连续型复合数字生物标志物——BioSync指数（BSI），在认知衰退检测的合成队列上达到0.928的AUC。

    

    来自可穿戴和移动设备的心脏、神经、行为和语音测量数据只能提供对生理状态的局部且对噪声敏感的观测视角。BioSync将这些测量整合为BioSync指数（BSI），这是一个在BEST框架下定义的连续型复合数字生物标志物。该模型对模态令牌应用多头自注意力机制，并增加一个线性分支，其假设空间涵盖了标准的特征拼接方法。该架构的设计源于潜变量测量理论，以及联合观测可能包含单个模态无法获得的信息这一假设。研究者在两个基于文献构建的合成队列上评估了BioSync：一个是使用心率变异性（HRV）、脑电图（EEG）、体动记录仪和语音数据的四模态认知衰退队列；另一个是围绕公开的AI-READI可穿戴设备数据模式构建的代谢-自主神经队列。在认知衰退队列中，BioSync和特征拼接方法分别取得了0.928和0.926的AUC……

    arXiv:2609.04504v1 Announce Type: new  Abstract: Cardiac, neural, behavioral, and speech measurements from wearable and mobile devices provide partial, noise-sensitive views of physiological state. BioSync combines these measurements into the \textbf{BioSync Index (BSI)}, a continuous composite digital biomarker defined under the BEST framework. The model applies multi-head self-attention to modality tokens and adds a linear branch whose hypothesis class includes standard feature concatenation. This architecture is motivated by latent-variable measurement theory and by the possibility that joint observations contain information unavailable from individual modalities. We evaluated BioSync on two literature-informed synthetic cohorts: a four-modality cognitive-decline cohort using HRV, EEG, actigraphy, and speech, and a metabolic-autonomic cohort structured around the public AI-READI wearable schema. In the cognitive cohort, BioSync and concatenation obtained AUCs of 0.928 and 0.926, res
    
[^18]: 转录数据集上语音编码算法的评估

    Evaluation of Phonetic Encoding Algorithms on Transcription Datasets

    [https://arxiv.org/abs/2609.04391](https://arxiv.org/abs/2609.04391)

    提出了一种基于Hüllermeier-Rifqi指数的新型评估方案，通过计算语音编码与IPA真实转录之间的成对相似度差异，并结合碰撞率评估多种语音编码算法在多语言转录数据集上的性能。

    

    本工作提出了一种基于广义Rand指数变体（即Hüllermeier-Rifqi指数）的新型评估方案，用于评估语音编码算法与IPA（国际音标）标注的基于单词的转录结果的一致程度。为此，通过计算真实转录结果与相应语音编码之间的成对相似度值的绝对差来获得不一致分数，其中相似度值采用归一化编辑距离作为依赖排列的字符串度量进行计算。所得分数随后根据使用与所考虑编码器相同字母表的随机字符串生成器的分数进行调整。以此方式，在多语言转录数据集上评估了多种语音编码器，并基于碰撞率评估了它们的召回能力。

    arXiv:2609.04391v1 Announce Type: new  Abstract: In this work, a novel evaluation scheme built on a generalized variant of the Rand Index measure, namely, the H\"ullermeier-Rifqi Index, is proposed in order to assess how well phonetic encoding algorithms conform to word-based transcriptions in IPA (International Phonetic Alphabet) notation. For this objective, the discordance score is obtained by calculating the absolute difference between the pairwise similarity values of ground-truth transcriptions and those of corresponding phonetic encodings, which are computed using normalized edit distance as a permutation dependent string metric. The resulting score is subsequently adjusted with respect to that of a random string generator incorporating the same alphabet as the encoder under consideration. A wide range of phonetic encoders were evaluated as such on multi-lingual transcription datasets along with their recall capabilities based on the collision rate. The validity of the proposed 
    
[^19]: 企业家族消解不是一个字符串匹配问题：一个按名称可见性分层的公开基准

    Corporate-Family Resolution Is Not a String-Matching Problem: A Public Benchmark Stratified by Name Visibility

    [https://arxiv.org/abs/2609.04269](https://arxiv.org/abs/2609.04269)

    该论文发布了CorpFam——一个基于美国联邦奖励记录中供应商自报母公司数据构建的企业家族消解公开基准，通过按名称可见性分层评估并报告分层召回率，揭示出最强匹配方法对名称不可见的企业家族对识别率仅4.2%，证明企业家族消解不能被简化为字符串匹配问题。

    

    判断两条供应商记录是否属于同一个企业家族，是支出整合、信用风险敞口汇总和制裁筛查的前提条件。这项任务通常被当作实体匹配来处理，但两者本质不同：家族关联连接的是刻意不同的实体，而且相关证据往往不出现在任何一条记录中。我们提出了CorpFam，这是一个包含54,864个候选对、覆盖10,307个企业家族的公开基准，其数据源自6,638,350条美国联邦奖励记录，其中每个供应商都向政府登记处自行申报其最终母公司。候选对按名称可见性进行分层：即名称在归一化后是否完全相同、是否共享一个区分性词元，或完全不共享。由于各层的正例率介于10.2%到97.3%之间，我们报告按层计算的召回率（该指标不受基础率影响），而非F1分数（该指标会受基础率影响）。在5个匹配器中，表现最强的匹配器能恢复100.0%的名称相同候选对，但对名称不可见的候选对仅能恢复4.2%。

    arXiv:2609.04269v1 Announce Type: cross  Abstract: Deciding whether two supplier records belong to the same corporate family is a prerequisite for spend consolidation, credit exposure aggregation and sanctions screening. It is usually treated as entity matching, but the tasks differ: a family link connects records that are deliberately different entities, and the evidence often appears in neither record. We introduce CorpFam, a public benchmark of 54,864 candidate pairs over 10,307 corporate families, derived from 6,638,350 US federal award records in which every supplier self-reports its ultimate parent to a government registry. Pairs are stratified by name visibility: whether the names are identical after normalisation, share a distinctive token, or share none. Because strata have positive rates from 10.2% to 97.3%, we report per-stratum recall, base-rate invariant, rather than F1, which is not. The strongest of 5 matchers recovers 100.0% of identical pairs and 4.2% of invisible ones
    
[^20]: SAGE：面向多实体视觉检索的语义属性图

    SAGE: Semantic Attribute Graphs for Multi-Entity Visual Retrieval

    [https://arxiv.org/abs/2609.04255](https://arxiv.org/abs/2609.04255)

    提出无需训练的SAGE框架，通过将密集文档图像中的语义实体解析为带多向量嵌入的层次图并进行迭代实体级子图匹配，解决单向量编码导致的“语义稀释”问题，并发布了用于多实体细粒度视觉检索的DEAR数据集。

    

    密集文档图像通常包含许多细粒度的视觉和文本实体，其相关性取决于用户的查询。标准的视觉-语言检索器使用单个向量对裁剪区域进行编码，这可能会混合不同实体的信号，从而掩盖细粒度检索所需的证据。我们将这种失败模式称为“语义稀释”（Semantic Dilution），并定量地表明，它会随实体密度的增加而降低实体级检索的性能。为了缓解这一问题，我们提出了SAGE——一个无需训练的框架，它从密集文档图像中解析语义实体，将其表示为具有多向量嵌入的层次图节点，并通过迭代的实体级子图匹配来检索与查询相关的证据。我们还引入了DEAR数据集，该数据集包含源自产品详情页的1,055对查询-图像对，其中每个查询都需要从视觉密集的输入中检索并比较多个细粒度实体，涵盖四种问题类型。

    arXiv:2609.04255v1 Announce Type: new  Abstract: Dense document images often contain many fine-grained visual and textual entities whose relevance depends on a user query. Standard vision-language retrievers encode cropped regions with a single vector, which can mix distinct entity signals and obscure the evidence needed for fine-grained retrieval. We call this failure mode Semantic Dilution and quantitatively show that it degrades entity-level retrieval as a function of entity density. To mitigate it, we propose SAGE, a training-free framework that parses semantic entities from dense document images, represents them as hierarchical graph nodes with multi-vector embeddings, and retrieves query-relevant evidence through iterative entity-level subgraph matching. We also introduce DEAR, a dataset of 1,055 query--image pairs sourced from product detail pages, where each query requires retrieving and comparing multiple fine-grained entities from visually dense inputs across four question ty
    
[^21]: UniCon：一种面向CTR预测的统一上下文中心建模范式

    UniCon: A Unified Context-Centric Modeling Paradigm for CTR Prediction

    [https://arxiv.org/abs/2609.03290](https://arxiv.org/abs/2609.03290)

    本文提出UniCon，一种统一的以上下文为中心的CTR预测建模范式，将用户历史行为与当前请求作为同质的上下文单元进行统一建模，克服传统异构信号划分的局限，提升扩展效率与预测质量。

    

    统一建模已成为工业级点击率（CTR）预测的重要发展方向。现有方法通常在token级别统一序列信号与非序列信号，在共享骨干网络中建模它们的交互，并通过增大模型容量来改善扩展性。然而，这种划分源于传统的特征工程实践，与底层的决策过程并不一致。用户行为本质上是一系列同质的上下文单元；在输入组织的层面，历史行为与当前请求的区别仅在于其结果是已被观测的还是尚待预测的。将二者视为异构信号会掩盖用户决策上下文内部的结构依赖关系，从而同时限制了扩展效率和预测质量。这一局限在电商货架、瀑布流信息流等上下文丰富的场景中尤为突出。为解决这一问题……

    arXiv:2609.03290v1 Announce Type: new  Abstract: Unified modeling has become a major direction for industrial click-through rate (CTR) prediction. Existing approaches typically unify sequential and non-sequential signals at the token level, model their interactions in a shared backbone, and increase model capacity to improve scaling behavior. However, this division originates from legacy feature-engineering practice and is misaligned with the underlying decision process. User behavior is inherently a sequence of homogeneous context units; at the level of input organization, historical behavior and the current request differ only in whether their outcomes are observed or remain to be predicted. Treating them as heterogeneous signals obscures structural dependencies within the user's decision context, limiting both scaling efficiency and prediction quality. This limitation is particularly pronounced in context-rich scenarios such as e-commerce shelves and waterfall feeds. To address this
    
[^22]: hLLM：面向生成式重排序的单遍解码

    hLLM: Single Pass Decoding for Generative Reranking

    [https://arxiv.org/abs/2609.01807](https://arxiv.org/abs/2609.01807)

    提出hLLM，通过轻量自注意力头从LLM预填充隐状态读取项目-位置得分矩阵，并用匈牙利算法求最优二分匹配，在O(1)次前向传播内一次性解码全部N个序数，从而将生成式重排序的解码从逐token自回归生成变为常数次前向传播，且天然保证输出为有效排序。

    

    arXiv:2609.01807v1 公告类型： cross 摘要：大语言模型（LLM）实现了最先进的生成式排序质量，但其产生的排序结果必须经过解码，而自回归解码每生成一个token就需要一次顺序前向传播。我们观察到，排序器必须输出的token仅仅是N个序数值，用于按排序顺序命名各个项目，而这种狭窄的、具有置换结构的输出格式使得我们可以采用比从左到右生成高效得多的解码策略。我们提出了hLLM（匈牙利LLM），一种针对该输出格式专门设计的解码策略，它能够在O(1)次前向传播中解码全部N个序数。hLLM通过一个轻量级的自注意力头，从LLM预填充阶段的隐状态中读取一个N×K的项目-位置得分矩阵，然后利用匈牙利算法将该矩阵的最优二分匹配作为序数解码，从而在构造层面（而非通过事后修复）保证输出是一个有效的置换。通过对训练信号的系统性研究……

    arXiv:2609.01807v1 Announce Type: cross  Abstract: Large language models (LLMs) achieve state-of-the-art generative ranking quality, but the ranking they produce must be decoded, and autoregressive decoding spends one sequential forward pass per emitted token. We observe that the only tokens a ranker must emit are the $N$ ordinal values naming the items in ranked order, and that this narrow, permutation-structured output format admits decoding strategies which are much more efficient than left-to-right generation. We introduce hLLM (Hungarian LLM), a format-specialized decoding strategy that decodes all $N$ ordinals in $O(1)$ forward passes. hLLM reads an $N \times K$ item-position score matrix off the LLM's prefill hidden states with a lightweight self-attention head, then decodes the ordinals as the optimal bipartite assignment of that matrix via the Hungarian algorithm, yielding a valid permutation by construction rather than by repair. Through a systematic study of training signals
    
[^23]: InsightToast：数据密集型会议侧信道中的主动信息检索与一目了然的可视化

    InsightToast: Proactive Information Retrieval & Glanceable Visualization in the Side Channel of Data-Rich Meetings

    [https://arxiv.org/abs/2608.31115](https://arxiv.org/abs/2608.31115)

    InsightToast通过多智能体LLM与RAG管道实时监测会议话语并主动检索信息，以临时提示和一目了然图表的形式在会议侧信道中提供有据可依的洞察，从而避免任务切换对会议参与和决策的干扰。

    

    会议中缺失机构背景信息会阻碍有效参与。检索相关信息往往需要耗费大量精力的任务切换——这些信息通常分散在异构的内部与外部来源中——这种切换会破坏个人的专注力和集体的对话流畅性，在决策制定等高认知负荷任务中尤为有害。我们提出了InsightToast，这是一个混合主动性（mixed-initiative）应用，它实时监测会议中的口头话语，在话题和信息需求出现时及时识别，并通过一个集成了检索增强生成（RAG）的多智能体大语言模型（LLM）管道主动检索相关信息，生成有来源依据的洞察，以简洁文本和一目了然的交互式图表形式呈现，并通过外围界面以临时提示（toast）的形式在对话的侧信道中传递。为了展示其产生意外洞见的潜力，我们……

    arXiv:2608.31115v1 Announce Type: cross  Abstract: Missing institutional context during meetings can impede effective participation. Retrieving relevant information, often scattered across heterogeneous internal and external sources, requires costly task-switching that disrupts both individual focus and collective conversational flow, particularly detrimental during cognitively demanding tasks such as decision-making. We introduce InsightToast, a mixed-initiative application that monitors verbal discourse in real time, identifies topics and informational needs as they emerge, and proactively retrieves relevant information through a multi-agent large language model (LLM)-based pipeline integrating retrieval-augmented generation (RAG) to produce source-grounded insights as succinct text and glanceable interactive charts, delivered through a peripheral interface as ephemeral toasts in the conversation's side channel. To demonstrate the potential for yielding serendipitous insights, we sho
    
[^24]: E-SENS：面向负约束检索的排斥敏感惩罚方法

    E-SENS: Exclusion-Sensitive Penalization for Negative-Constraint Retrieval

    [https://arxiv.org/abs/2608.30130](https://arxiv.org/abs/2608.30130)

    E-SENS是一种无需训练的重排序方法，通过为被排除概念提取“陷阱查询”并从检索分数中减去其相似度，有效惩罚与用户排除概念相关的文档，从而提升检索系统对负向约束的遵守能力。

    

    检索增强语言模型在检索器提供用户明确排除概念的相关证据时，可能无法遵守负向约束。除了显式否定之外，查询还可能要求答案包含一个概念而排除另一个概念，或者要求实体属于某一类别但与密切相关的实例不同。由于被排除的概念仍然出现在查询文本中，稠密检索器可能会对与该概念相关的文档赋予高相似度，即使用户明确要求避开它。我们提出了E-SENS，一种面向否定敏感检索的无训练重排序方法。E-SENS为被排除的一方提取一个紧凑的“陷阱查询”，并从原始查询的检索分数中减去陷阱查询的相似度。在ExcluIR基准上，E-SENS在四个嵌入模型上展现出清晰的召回率-违规权衡，并在保持召回率的设置下有效减少了陷阱检索。

    arXiv:2608.30130v1 Announce Type: cross  Abstract: Retrieval-augmented language models can fail to respect negative constraints when the retriever supplies evidence about concepts the user explicitly excluded. Beyond explicit negation, queries may ask for answers that include one concept while excluding another, or for entities that belong to a category but differ from a closely related instance. Because the excluded concept still appears in the query text, dense retrievers may assign high similarity to documents about that concept even when the user asks to avoid it. We introduce E-SENS, a training-free reranking method for negation-sensitive retrieval. E-SENS extracts a compact trap query for the excluded side and subtracts trap-query similarity from the original-query retrieval score. On ExcluIR, E-SENS shows a clear recall-violation trade-off across four embedding models and reduces trap retrieval at recall-preserving settings.
    
[^25]: 一个事件值得一个令牌：面向工业规模LLM推荐的事件令牌化

    An Event is Worth One Token: Event Tokenization for Industrial-scale LLM Recommendation

    [https://arxiv.org/abs/2608.25546](https://arxiv.org/abs/2608.25546)

    本文提出了一种以事件为中心的自回归推荐方法AMBER，通过将每个交互的完整时间快照压缩为事件令牌，显著提高了快照分辨率，从而增强了LLM推荐系统的性能和扩展性。

    

    arXiv:2608.25546v1 公告类型：新 摘要：基于LLM的推荐系统已沿模型容量和序列长度进行扩展，但每个位置仅编码文本、语义ID或少量分类特征，忽略了每个事件中可用的丰富用户、物品、上下文和结果信号。在自回归建模下，这导致每个位置的查询能力弱，且由于每个位置成为下一个位置的上下文，这种退化在序列中累积。我们提出了一种以事件为中心的范式，通过完整的时间快照表示每次交互，并确定了一个新的扩展维度，我们称之为快照分辨率：每个事件编码的信息量。为了高效扩展快照分辨率，我们引入了AMBER（通过瓶颈事件表示的自回归建模），它将每个时间快照压缩为紧凑的事件令牌，这是一种新的LLM输入模态。该表示是端到端学习的，而事件令牌则预先计算并缓存以供服务使用。

    arXiv:2608.25546v1 Announce Type: new  Abstract: LLM-based recommendation has scaled along model capacity and sequence length, yet each position encodes only text, semantic IDs, or a few categorical features, discarding rich user, item, context, and outcome signals available at each event. Under autoregressive modeling, this yields weak queries at each position and, since each position becomes context for the next, the degradation compounds across the sequence. We propose an event-centric paradigm that represents each interaction by its full temporal snapshot, and identify a new scaling dimension we term snapshot resolution: the amount of information encoded per event. To efficiently scale snapshot resolution, we introduce AMBER (Autoregressive Modeling via Bottlenecked Event Representation), which compresses each temporal snapshot into a compact Event Token, a new LLM input modality. The representation is learned end-to-end, while Event Tokens are pre-computed and cached for serving, 
    
[^26]: 尼泊尔语护照问答：面向公共服务应用的低资源数据集

    Nepali Passport Question Answering: A Low-Resource Dataset for Public Service Applications

    [https://arxiv.org/abs/2603.13320](https://arxiv.org/abs/2603.13320)

    该研究构建了首个面向护照公共服务的尼泊尔语问答低资源数据集，通过微调Transformer嵌入模型并结合BM25进行混合检索，其中基于多语言E5嵌入的模型取得了最佳检索性能。

    

    尼泊尔语作为一种低资源语言，由于缺乏标注数据和计算语言学资源，在构建有效的信息检索系统方面面临重大挑战。在本研究中，我们试图通过构建成对结构的尼泊尔语问答数据集来弥补这一空白。我们专注于与护照服务相关的常见问题（FAQs），构建了用于训练和评估信息检索模型的数据集。在研究中，我们针对问答检索中的语义相似性任务对基于Transformer的嵌入模型进行了微调，并将微调后的模型与基线模型BM25进行了比较。此外，我们还实现了一种将微调模型与BM25相结合的混合检索方法，并评估了混合检索的性能。结果表明，微调的基于SBERT的模型优于BM25，而基于多语言E5嵌入的模型取得了最高的检索性能。

    arXiv:2603.13320v2 Announce Type: replace-cross  Abstract: Nepali, a low-resource language, faces significant challenges in building an effective information retrieval system due to the unavailability of annotated data and computational linguistic resources. In this study, we attempt to address this gap by preparing a pair-structured Nepali Question-Answer dataset. We focus on Frequently Asked Questions (FAQs) for passport-related services, building a data set for training and evaluation of IR models. In our study, we have fine-tuned transformer-based embedding models for semantic similarity in question-answer retrieval. The fine-tuned models were compared with the baseline BM25. In addition, we implement a hybrid retrieval approach, integrating fine-tuned models with BM25, and evaluate the performance of the hybrid retrieval. Our results show that the fine-tuned SBERT-based models outperform BM25, whereas multilingual E5 embedding-based models achieve the highest retrieval performance
    
[^27]: 用于推荐的图基础模型：一项全面综述

    Graph Foundation Models for Recommendation: A Comprehensive Survey

    [https://arxiv.org/abs/2502.08346](https://arxiv.org/abs/2502.08346)

    该综述首次全面梳理了图基础模型（GFM）在推荐系统中的应用，提出了现有方法的清晰分类体系，深入剖析了融合图神经网络与大语言模型优势的技术细节，并指出了该领域的关键挑战与未来研究方向。

    

    推荐系统（RS）是浏览海量在线信息的基础工具，深度学习的进步在提高排序准确性方面发挥着日益重要的作用。其中，图神经网络（GNN）擅长提取高阶结构信息，而大语言模型（LLM）则旨在处理和理解自然语言，这两种方法都因此高效且被广泛采用。近期的研究聚焦于图基础模型（GFM），它整合了GNN和LLM的优势，通过利用用户-物品关系的图结构以及文本理解能力，更高效地对复杂的推荐系统问题进行建模。在这篇综述中，我们通过引入当前方法的清晰分类体系、深入探讨方法论细节，并强调关键挑战与未来方向，对基于GFM的推荐系统技术进行了全面概述。

    arXiv:2502.08346v4 Announce Type: replace-cross  Abstract: Recommender systems (RS) serve as a fundamental tool for navigating the vast expanse of online information, with deep learning advancements playing an increasingly important role in improving ranking accuracy. Among these, graph neural networks (GNNs) excel at extracting higher-order structural information, while large language models (LLMs) are designed to process and comprehend natural language, making both approaches highly effective and widely adopted. Recent research has focused on graph foundation models (GFMs), which integrate the strengths of GNNs and LLMs to model complex RS problems more efficiently by leveraging the graph-based structure of user-item relationships alongside textual understanding. In this survey, we provide a comprehensive overview of GFM-based RS technologies by introducing a clear taxonomy of current approaches, diving into methodological details, and highlighting key challenges and future direction
    

