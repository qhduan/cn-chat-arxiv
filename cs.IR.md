# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Incremental Pooled LLM Evaluation for Cost-Effective Retrieval Model Selection](https://arxiv.org/abs/2609.02745) | 提出增量式池化LLM评估方法，通过LLM判断候选系统检索文档的并集并随新系统增量扩展文档池，实现低成本、可复用的检索模型对比评估，其排序结果与金标准评估高度一致。 |
| [^2] | [Recommender System as Slow and Fast Thinkers](https://arxiv.org/abs/2609.02671) | 提出了面向序列推荐的自适应快慢推理框架 DS-Frame，通过快速系统、慢速系统和可学习选择器在可控计算预算下动态路由样本，在五个真实数据集上持续提升推荐骨干模型性能，尤其在困难用户群体上收益显著。 |
| [^3] | [Training seeds and model-selection stability in recommender-system evaluation](https://arxiv.org/abs/2609.02499) | 推荐系统评估实验中，训练种子的变化往往会产生可检测的影响，仅报告单一随机种子的结果可能夸大评估结论的稳定性。 |
| [^4] | [ViSAR: Training-Free Adaptive-$k$ Retrieval for Visual Document Question Answering](https://arxiv.org/abs/2609.02486) | 提出了一种无需训练的自适应k值检索方法ViSAR，通过在嵌入空间中构建查询条件的页面级相似度矩阵来动态确定检索页面数量，在保持或提升答案准确性的同时将RAG延迟降低高达58.7%。 |
| [^5] | [MultiGhostBench: A Multilingual Benchmark for Long-Form LLM-Generated Text Attribution under Distribution Shifts](https://arxiv.org/abs/2609.02379) | 本文提出了MultiGhostBench多语言基准，包含五个最新大语言模型在六种语言下生成的928本约59K词的长篇书籍，用于评估领域、作者和语言偏移下的LLM文本归因，发现没有单一方法始终最优且分布偏移会导致性能下降。 |
| [^6] | [Adaptive Test-Time Inference for Text2Cypher with Trace Budgeting and Selective Refinement](https://arxiv.org/abs/2609.02324) | 本文提出针对Text2Cypher的自适应测试时推理方法，通过基于问题难度动态调整候选生成预算的自适应轨迹预算和仅在有益时才执行纠正的选择性执行引导精炼两种策略，在Gemma-2-9B和Qwen-2.5-7B上将平均生成预算减少30.7%，在降低计算开销的同时提升查询生成的可靠性。 |
| [^7] | [Counter-GEO-Bench: Evaluating Defenses Against Information-Distorting Generative Engine Optimization](https://arxiv.org/abs/2609.02316) | 提出了首个针对生成式引擎优化（GEO）攻击的防御基准Counter-GEO-Bench，通过将247个经人工验证的查询与信息保留型和信息扭曲型GEO改写配对来评估防御方法，并揭示现有三种主流防御方法最多仅能将攻击成功率相对降低5.7%。 |
| [^8] | [Genuine Information Needs of Social Scientists Looking for Data](https://arxiv.org/abs/2609.02303) | 本研究通过对72名社会科学研究人员进行在线调查，让他们以“向同事求助”的方式表达对研究数据的信息需求，并分析归纳出主题、元数据等需求类别，从而揭示了用户真实复杂的信息需求与数据检索系统能力之间的差距。 |
| [^9] | [Group-Aware Adaptive Retrieval for Evidence Navigation](https://arxiv.org/abs/2609.02188) | 提出了群组感知自适应检索框架 GAREN，通过将语料库图中的文档组织成语义连贯的群组并以组级别方式进行扩展导航，为多步推理检索提供超越文档级信号的指导，从而缓解召回受限问题。 |
| [^10] | [GenCAR: Generative Counterfactual Alignment with Risk-Controlled Selection for Out-of-Distribution Recommendation](https://arxiv.org/abs/2609.02162) | 提出GenCAR框架，将分布外推荐形式化为α-有效反事实推荐（α-VCR）问题，通过基于偏好的反事实监督与基于保形p值的Benjamini-Hochberg集合选择，在控制代理标签错误发现率（FDR）的同时提升推荐效用。 |
| [^11] | [Beyond Modality Harmony: Orthogonal Purification and Topology-Guided MoE for Conflict-Aware Multimodal Recommendation](https://arxiv.org/abs/2609.02152) | 该论文提出OrthoRec框架，通过协同引导正交净化（CGOP）在几何上解耦并过滤多模态特征中与协同拓扑冲突的噪声，结合拓扑引导的混合专家模型实现冲突感知的多模态推荐，突破了传统“模态和谐”假设的局限。 |
| [^12] | [A Power Law in Logarithm's Clothing: On the Scalability of Graph-Based Vector Search](https://arxiv.org/abs/2609.02143) | 该论文通过跨数据集规模的实测推翻了“搜索成本随数据规模呈多重对数增长”的通行说法，发现当数据规模相对内在维度较小时，基于图的向量搜索成本实际遵循次线性幂律（约按 $N^c$、$0<c<1$ 增长），只有规模足够大时才收敛到理论预言的对数式增长。 |
| [^13] | [Beyond Context Windows: Persistent Discovery Context for Data-Centric Agents](https://arxiv.org/abs/2609.02129) | 本文提出“持久化发现上下文”——一种存储并复用先前意图到对象映射的轻量级记忆层，能在多个结构化数据环境中持续提升数据中心智能体的检索质量，且在词汇稀疏领域甚至优于基于元数据的检索。 |
| [^14] | [SPAR: Enhancing Industrial-Scale Generative POI Recommendation via Real-World Spatial Perception](https://arxiv.org/abs/2609.02062) | 论文提出SPAR统一框架，通过三个协同阶段将真实的城市空间知识（距离、方向、可达性）显式注入兴趣空间，解决现有生成式POI推荐方法预测结果偏离用户实时位置的问题。 |
| [^15] | [GeoStore: Finding Small Storefronts in Large Scenes -- A Fine-Grained POI Localization Benchmark with Global-to-Local Asymmetric Matching](https://arxiv.org/abs/2609.02012) | 该论文提出了首个针对非对称、细粒度、开放集POI定位任务的基准GeoStore，揭示了为对称视觉地点识别设计的全局描述符方法在此任务上的系统性局限，并提出了全局到局部非对称匹配方法GLAM。 |
| [^16] | [Seed-Anchored Budget-Bounded Graph Rendering for Question Answering on Industry-Standard Power-Grid Information and Exchange Models](https://arxiv.org/abs/2609.02011) | 本文提出了一种确定性的种子锚定图渲染方法，在固定上下文预算内优先保留与查询相关的局部图证据，将电网CIM/CGMES模型上大语言模型问答的多跳证据保留率从0.12和0.00提升至全部保留，准确率从0.450提升至0.970。 |
| [^17] | [MERGED: Multimodal Entity Resolution via Generated Expert Reasoning Distillation](https://arxiv.org/abs/2609.01913) | MERGED框架让多个大型视觉-语言模型教师为商品对标注并阐述推理，将一致标注用于监督微调、分歧标注经元裁判转化为直接偏好优化的偏好对，从而把结构化推理蒸馏到无需人工标注的7B紧凑学生模型中，实现了生产规模下低成本、低延迟的多模态商品实体解析。 |
| [^18] | [ExecRetrieval: Measuring the Functional-Correctness Gap in Code-Embedding Retrieval](https://arxiv.org/abs/2609.01865) | 提出 ExecRetrieval 基准（939 个 Python 任务），通过在搜索池中植入与规范实现几乎相同、但经执行验证的有缺陷变体，首次衡量了代码嵌入检索在区分功能正确代码与错误代码上的差距。 |
| [^19] | [Cite or Decline: A Strict Course-Grounded Chatbot for STEM Lecture Videos](https://arxiv.org/abs/2609.01846) | 本文提出了VideoPoints平台一学期的实际部署，其检索增强聊天机器人严格基于课程材料回答问题并提供带时间戳的引用，在无证据时选择拒答，833条学生消息中实现了零课程边界越界，证明了严格课程约束设计的可行性。 |
| [^20] | [Index-Free Dynamic Edge Retrieval with Energy-Tail-Aware Partial Scans](https://arxiv.org/abs/2609.01820) | 提出了无索引的动态最大内积搜索方法ETAR，通过保留查询向量中能量最大的坐标进行部分扫描、以低精度表示估计相似度并校正尾部误差后重排序候选，在保持更新简单高效的同时显著降低了查询开销。 |
| [^21] | [hLLM: Single Pass Decoding for Generative Reranking](https://arxiv.org/abs/2609.01807) | 提出hLLM，通过轻量自注意力头从LLM预填充隐状态读取项目-位置得分矩阵，并用匈牙利算法求最优二分匹配，在O(1)次前向传播内一次性解码全部N个序数，从而将生成式重排序的解码从逐token自回归生成变为常数次前向传播，且天然保证输出为有效排序。 |
| [^22] | [KGVoyager: Knowledge Graph Agnostic Question Answering via Agentic Navigation](https://arxiv.org/abs/2609.01780) | KGVoyager提出了一种知识图谱无关的智能体架构，通过“思考-行动-观察”循环动态探索图谱结构与语义，仅需查询端点即可从自然语言问题生成SPARQL查询，在四个基准上将F1提升约8分，同时将成本与运行时间各降低约22%。 |
| [^23] | [NeoMME: A Single-Tower Multimodal-Native Multilingual Foundation Encoder for Efficient Fine-Tuning and Inference](https://arxiv.org/abs/2609.01657) | NeoMME是一系列从零预训练的单塔双向多模态多语言基础编码器（260M/800M参数），采用掩码离散扩散目标训练并支持16K token上下文，在视觉文档检索等下游任务上以更低的参数与计算开销实现高效微调和推理。 |
| [^24] | [From Feature Interaction to Feature Transport - A Unified Block for Scalable Recommendation Models](https://arxiv.org/abs/2609.01655) | 提出CRAFT统一模块，将非序列化特征转化为可靠性感知的上下文场，以残差位移和记忆保持信号主动控制意图信息在堆叠模块间的传输与演化，实现推荐模型从“特征交互”到“特征传输”的范式转变。 |
| [^25] | [MELON: A Large-Scale Dataset for Multi-Event Text-to-Long-Video Retrieval](https://arxiv.org/abs/2609.01654) | 该论文提出了MELON——首个面向多事件文本到长视频检索的大规模数据集，通过为每个视频标注多个事件区间及文本描述，并引入多事件感知损失来提升模型对全事件与部分事件匹配的区分能力。 |
| [^26] | [The Vocabulary Gap Is an Equity Gap: Register Mismatch in Retrieval Systems for Public-Benefits Access](https://arxiv.org/abs/2609.01645) | 该研究发现在公共福利资格检索系统中，用户用平实语言提问时的检索性能远低于用官方正式语体提问时（Recall@5从100%降至44%），表明文档与用户之间的词汇和语域鸿沟造成了实质性的公平性差距。 |
| [^27] | [Imagine Before Retrieval: Prospective Skill Retrieval for LLM Agents](https://arxiv.org/abs/2609.01642) | 提出受人类前瞻性认知启发的SkillDreamer框架，通过“先想象后检索”策略解决LLM智能体技能检索中任务查询与技能之间的失配（QSM）问题。 |
| [^28] | [GRAND-HC: Graph-Refined Author Name Disambiguation](https://arxiv.org/abs/2609.01636) | 提出端到端从头姓名消歧框架GRAND-HC，通过异构论文图、和谐对比学习和图优化距离矩阵，有效解决作者长尾分布导致的尾部作者过度合并问题。 |
| [^29] | [Graph Neural Team Recommendation: An Integrated Approach](https://arxiv.org/abs/2609.01631) | 该论文提出将团队推荐问题重新表述为专家协作图上的端到端链接预测任务，从而利用图神经网络捕捉团队内与跨团队的多跳协作关系及其技能间的复杂依赖，实现端到端优化。 |
| [^30] | [Marginal Expected Revenue for Jointly Ranking Auction and Fixed-Price Listings in E-Commerce Sponsored Search](https://arxiv.org/abs/2609.01628) | 该论文提出边际eCPM（meCPM）方法，将传统固定价格商品的期望收益估算框架扩展至价格动态演化的拍卖和“拍卖加一口价”（ABIN）商品，实现了电商赞助搜索中多种上架形式的联合排序。 |
| [^31] | [The Utility of LLMs in Recommender Systems Explanation Evaluation](https://arxiv.org/abs/2609.01627) | 本文研究了大语言模型作为“评判者”在推荐系统解释评估中的可靠性，旨在帮助为给定应用自动选择有效的解释方法。 |
| [^32] | [RecEvolve: A Knowledge-Driven Autonomous Agent System for Recommender Systems](https://arxiv.org/abs/2609.01622) | 本文提出知识驱动的自主智能体系统RecEvolve，将想法生成、代码实现、训练与评估的完整研究生命周期纳入持续自主闭环框架，首次在生产级大规模双塔召回模型上从零完成40余次自主训练迭代，突破隐藏架构瓶颈并取得NDCG约20%的相对提升，使线上用户满意度增长3.77%。 |
| [^33] | [When Literature Data Mislead Artificial Intelligence in Materials Discovery](https://arxiv.org/abs/2609.01621) | 该研究以固态电解质电导率数据为例，揭示了科学文献中普遍存在的文本-图表不匹配、单位不一致等看似合理却难以检测的错误，这些错误会以结构化标签噪声的形式污染AI训练数据库，甚至导致高达100倍的电导率误差。 |
| [^34] | [MGDiff: Multi-Interest Sequence Recommendation with Masking GNN-Guided Diffusion](https://arxiv.org/abs/2609.01619) | 提出了一种基于掩码GNN引导扩散的多兴趣序列推荐框架MGDiff，通过双层语义引导（DSG）和流行度感知引导（PAG）机制，在扩散过程中生成准确且无偏的用户兴趣表示，从而提升推荐精度。 |
| [^35] | [Multi-Agent Retrieval-Augmented Generation for Efficient Cloud Knowledge Base Search in Telecom SNOC Environment](https://arxiv.org/abs/2609.01618) | 本文提出了一个完全离线的多智能体RAG框架Athena，通过融合E5稠密检索、BM25稀疏检索和知识图谱扩展，并结合加权CombSUM融合与交叉编码器重排序，实现了电信SNOC环境中企业云知识库的高效精准检索。 |
| [^36] | [Hybrid Retrieval-Augmented Generation with Knowledge Graph Expansion, RRF Fusion, and Per-Chunk Grounded Evaluation for Enterprise Document Search](https://arxiv.org/abs/2609.01617) | DocuSearch 提出了一种混合检索增强生成系统，通过倒数排名融合（RRF）将稠密向量语义检索、BM25全文搜索与知识图谱邻居扩展三种互补证据源加权融合（权重分别为0.50、0.35、0.15），在电信网络运营生产环境中实现了更准确、有据可依的企业文档问答。 |
| [^37] | [Incident Memory: Training-Free Operational Memory through Sequential Pattern Mining and Velocity-Stratified Retrieval](https://arxiv.org/abs/2609.01616) | 该论文提出了一种无需模型训练的确定性运维记忆系统 Incident Memory，通过速度分层检索、指纹条件 PrefixSpan 序列挖掘和溯源感知的指标定义，从历史事件日志中提取有序处置手册，在 UCI ITSM 数据集上覆盖了 84.3% 的留存事故。 |
| [^38] | [Skim and Skip: Hierarchical Adaptive Inference for Efficient Multimodal Retrieval](https://arxiv.org/abs/2609.01613) | 提出层次化自适应推理框架SAS，通过词元级证据筛选与深度自适应推理两大机制，在保持检索性能的同时大幅降低通用多模态检索的推理成本。 |
| [^39] | [MESSY STREETS: A Benchmark for Geocoding Real-World Addresses](https://arxiv.org/abs/2609.01612) | MESSY STREETS是一个评估地理编码器处理真实杂乱网页地址的新基准，揭示了商业地理编码器的召回率比开源系统高出多达49个百分点，差距主要源于非规范地址的候选返回率差异。 |
| [^40] | [Making Revisions Understandable: A Survey of Edit Intentions, Methods, and Applications](https://arxiv.org/abs/2609.01610) | 这是首篇从编辑意图视角系统梳理文本修订研究的综述，为修订语料库、编辑意图分类体系、识别方法及写作辅助等下游应用提供了统一视图。 |
| [^41] | [Towards Effective Structured Context Modeling for Conversational Recommender Systems via Dual-node Monte Carlo Tree Search](https://arxiv.org/abs/2609.00618) | 提出DREAMS框架，通过双节点树结构（引导节点用蒙特卡洛树搜索探索对话动作以推断潜在偏好，利用节点用大语言模型将偏好状态精炼为结构化检索查询），显式建模对话式推荐系统中用户偏好的多轮演化。 |
| [^42] | [ICEGR: An Intent-Coherent End-to-End Generative Retrieval Framework for E-commerce Search](https://arxiv.org/abs/2608.29652) | 提出ICEGR框架，通过在生成式检索的语义ID构建、监督微调和偏好优化等整个训练流程中一致融入查询意图，解决电商搜索中查询意图不一致的问题，从而提升低曝光商品的检索效果和查询-商品相关性。 |
| [^43] | [A Storage-Retrieval Gap in Parametric Knowledge Graph Memory](https://arxiv.org/abs/2608.25489) | 该论文提出将知识图谱离线编译为LoRA适配器作为参数化知识层，在零查询上下文成本下实现事实知识泛化，但发现存储知识无法通过相似性检索恢复，揭示了参数化记忆中的存储-检索差距。 |
| [^44] | [Not Worth Another Token: Marginal Value Estimation for Efficient Deep Research Agents](https://arxiv.org/abs/2608.08389) | 该论文首次对深度研究智能体流水线中检索前、检索后和综合前三个阶段的上下文剪枝策略进行了系统性分阶段比较，发现剪枝时机比具体评分规则更关键，轻量级启发式方法可在几乎不损失质量的情况下减少多达73%的Token消耗。 |
| [^45] | [SABER-Math: Automated Benchmark for Information Retrieval Evaluation in Mathematics](https://arxiv.org/abs/2606.29894) | 该论文提出了首个无需专家标注、完全自动化的数学信息检索评估基准SABER-Math，它从28.3万道高中数学题出发自动构建具有挑战性的重排序任务，以克服现有基准无法捕捉细粒度数学相关性的问题。 |
| [^46] | [Beyond Retrieval: Learning Compact User Representations for Scalable LLM Personalization](https://arxiv.org/abs/2606.04547) | 提出TAP-PER框架，通过时序注意力前缀嵌入将用户偏好编码为紧凑的可学习表示，摆脱了检索式个性化对检索质量的依赖以及参数式个性化随用户规模增长的高昂存储成本，实现可扩展的大语言模型个性化。 |
| [^47] | [CaST-POI: Candidate-Conditioned Spatiotemporal Modeling for Next POI Recommendation](https://arxiv.org/abs/2604.20845) | 提出CaST-POI，通过在目标注意力中引入时间新近度和地理距离两个分桶偏置，使不同候选点以不同的注意力权重读取同一用户轨迹，从而改进下一兴趣点推荐。 |

# 详细

[^1]: 面向低成本检索模型选择的增量式池化LLM评估

    Incremental Pooled LLM Evaluation for Cost-Effective Retrieval Model Selection

    [https://arxiv.org/abs/2609.02745](https://arxiv.org/abs/2609.02745)

    提出增量式池化LLM评估方法，通过LLM判断候选系统检索文档的并集并随新系统增量扩展文档池，实现低成本、可复用的检索模型对比评估，其排序结果与金标准评估高度一致。

    

    为生产环境的RAG系统选择检索模型需要可靠的对比评估，但大规模获取相关性判断代价高昂，且随着新候选系统的出现难以重复进行。我们研究了池化LLM评估方法，即由LLM对当前候选系统集合所检索文档的并集进行判断，并随着新系统的引入，通过仅判断其贡献的新文档来增量扩展文档池。这些判断结果被重复利用，从而在共同基础上评估所有系统。我们在四个检索基准上验证了该方法，涵盖密集、稀疏和混合配置的11个系统，并将其部署用于比较一个金融新闻问答系统的62种检索配置。池化LLM排序在各数据集上与金标准评估高度相关，且在考虑qrels的bootstrap不确定性后，97%的系统两两排序得以保持。

    arXiv:2609.02745v1 Announce Type: cross  Abstract: Selecting a retrieval model for a production RAG system requires reliable comparative evaluation, but obtaining relevance judgments at scale is expensive and difficult to repeat as new candidate systems arrive. We study pooled LLM evaluation, in which an LLM judges the union of documents retrieved by the current set of candidate systems, and the pool is then expanded incrementally as new systems are introduced by judging only the new documents they contribute. These judgments are reused to evaluate all systems on a common basis. We validate this approach on four retrieval benchmarks with 11 systems spanning dense, sparse, and hybrid configurations, and deploy it to compare 62 retrieval configurations for a financial news QA system. Pooled LLM rankings correlate strongly with gold-standard evaluation across datasets, and 97% of pairwise system orderings are preserved once bootstrap uncertainty in the qrels is taken into account. In prod
    
[^2]: 推荐系统作为慢思考与快思考者

    Recommender System as Slow and Fast Thinkers

    [https://arxiv.org/abs/2609.02671](https://arxiv.org/abs/2609.02671)

    提出了面向序列推荐的自适应快慢推理框架 DS-Frame，通过快速系统、慢速系统和可学习选择器在可控计算预算下动态路由样本，在五个真实数据集上持续提升推荐骨干模型性能，尤其在困难用户群体上收益显著。

    

    序列推荐模型是现代个性化服务的基础，但其有效性在异构用户环境中差异显著。特别是，静态的单次前向推荐器通常在常见行为模式上表现良好，但在运营上具有挑战性的用户群体（如历史行为较长或物品偏好不够主流的用户）上性能会下降。为了解决这一局限，我们提出了 DS-Frame，一个用于序列推荐的自适应快-慢推理框架。DS-Frame 结合了用于高效常规预测的快速系统、用于迭代式潜在表示精炼的慢速系统，以及一个学习得到的选择器，在可控的计算预算下为每个样本分配路由。在五个真实世界数据集上的实验表明，DS-Frame 能够持续提升具有代表性的序列推荐骨干模型，在具有挑战性的用户群体上收益更大，并实现了有效的精度-效率权衡。

    arXiv:2609.02671v1 Announce Type: new  Abstract: Sequential recommendation models are foundational to modern personalized services, yet their effectiveness varies substantially across heterogeneous user environments. In particular, static one-pass recommenders often perform well on common behavior patterns but degrade on operationally challenging user groups, such as users with longer histories or less mainstream item profiles. To address this limitation, we propose \textsc{DS-Frame}, an adaptive fast--slow inference framework for sequential recommendation. \textsc{DS-Frame} combines a Fast System for efficient routine prediction, a Slow System for iterative latent refinement, and a learned selector that routes each sample under a controllable computation budget. Experiments on five real-world datasets show that \textsc{DS-Frame} consistently improves representative sequential recommendation backbones, with larger gains on challenging groups and effective accuracy--efficiency trade-off
    
[^3]: 推荐系统评估中的训练种子与模型选择稳定性

    Training seeds and model-selection stability in recommender-system evaluation

    [https://arxiv.org/abs/2609.02499](https://arxiv.org/abs/2609.02499)

    推荐系统评估实验中，训练种子的变化往往会产生可检测的影响，仅报告单一随机种子的结果可能夸大评估结论的稳定性。

    

    推荐系统实验通常依赖于单一的随机训练种子，并假设运行间的随机性对评估结论的影响有限。这一假设存在风险，因为训练种子可能影响多种与算法相关的机制，包括参数初始化、小批量数据排序、dropout、掩码、潜在采样以及训练时的负采样。我们通过固定数据划分并在不同超参数配置下改变训练种子来检验这一假设。我们在三个层面分析种子效应：用户级指标敏感性、基于验证集的模型选择以及推荐列表的一致性。结果表明，种子变化的影响通常是可检测的，其影响程度取决于各配置之间是否明显分离、验证集结果能否迁移到测试集，以及相似得分是否会产生相似的top-k列表。研究发现表明，仅报告单种子结果可能会夸大结论的稳定性。

    arXiv:2609.02499v1 Announce Type: cross  Abstract: Recommender-system experiments often rely on a single random training seed, assuming that run-to-run stochasticity has limited impact on evaluation conclusions. This assumption is risky, as a training seed may influence several algorithm-dependent mechanisms, including parameter initialization, mini-batch ordering, dropout, masking, latent sampling, and training-time negative sampling. We examine this assumption by fixing the data partition and varying the training seed across hyperparameter configurations. We analyze seed effects at three levels: user-level metric sensitivity, validation-based model selection and recommendation-list agreement. Results show that seed variation is often detectable. Its impact depends on whether configurations are clearly separated, whether validation results transfer to test, and whether similar scores lead to similar top-$k$ lists. Findings suggest that reporting single-seed results can overstate the s
    
[^4]: ViSAR：面向视觉文档问答的无需训练的自适应k值检索方法

    ViSAR: Training-Free Adaptive-$k$ Retrieval for Visual Document Question Answering

    [https://arxiv.org/abs/2609.02486](https://arxiv.org/abs/2609.02486)

    提出了一种无需训练的自适应k值检索方法ViSAR，通过在嵌入空间中构建查询条件的页面级相似度矩阵来动态确定检索页面数量，在保持或提升答案准确性的同时将RAG延迟降低高达58.7%。

    

    文档视觉问答通常利用检索增强生成技术，其中晚期交互编码器常被用于识别与用户查询相关的文档页面，然后由大型视觉-语言模型生成答案。现有方法通常无论查询复杂度如何都检索固定数量的前k个页面，这会增加大型视觉-语言模型的延迟，并可能降低答案的准确性。我们提出了ViSAR（视觉语义激活检索），这是一种面向晚期交互视觉文档检索的无需训练的自适应k值检索方法。ViSAR直接在嵌入空间中运行，构建以查询为条件的页面级相似度矩阵，突出与查询相关的语义，并动态确定需要检索的页面数量。在多个编码器和大型视觉-语言模型上的实验表明，ViSAR能够检索紧凑且适应查询的页面集合，将RAG延迟降低高达58.7%，同时保持或提升答案准确性。

    arXiv:2609.02486v1 Announce Type: cross  Abstract: Document Visual Question Answering (DocVQA) often leverages Retrieval-Augmented Generation (RAG), where late-interaction encoders are commonly used to identify document pages relevant to a user query, before answer generation by a Large Vision-Language Model (LVLM). Existing approaches typically retrieve a fixed top-$k$ number of pages regardless of query complexity, which increases LVLM latency and may degrade answer accuracy. We introduce ViSAR (Visual Semantic Activation Retrieval), a training-free adaptive-$k$ retrieval method for late-interaction visual document retrieval. ViSAR operates directly in the embedding space to construct a query-conditioned page-level similarity matrix that highlights query-relevant semantics and dynamically determines the number of pages to retrieve. Across multiple encoders and LVLMs, ViSAR retrieves compact, query-adapted page sets that reduce RAG latency by up to 58.7\%, while maintaining or improvi
    
[^5]: MultiGhostBench：面向分布偏移下长篇LLM生成文本归因的多语言基准

    MultiGhostBench: A Multilingual Benchmark for Long-Form LLM-Generated Text Attribution under Distribution Shifts

    [https://arxiv.org/abs/2609.02379](https://arxiv.org/abs/2609.02379)

    本文提出了MultiGhostBench多语言基准，包含五个最新大语言模型在六种语言下生成的928本约59K词的长篇书籍，用于评估领域、作者和语言偏移下的LLM文本归因，发现没有单一方法始终最优且分布偏移会导致性能下降。

    

    尽管现有的LLM作者归因研究已经取得了一定进展，但可用的基准仍然有限，通常只关注英语、受控环境或相对过时的模型，而少数多语言研究也仅考虑了相对较短的文本。我们提出了MultiGhostBench，这是一个多语言基准，包含由五个最新大语言模型生成的928本书，涵盖六种语言和三种文字系统，每本书的平均长度约为59,000词。该基准支持在领域、作者和语言偏移下的评估。对代表性作者归因方法的评估表明，没有单一方法能在所有设置下始终表现最佳，且在分布偏移下性能普遍下降。基于Transformer的检测器能够跨语言保留与生成器相关的信息，尽管迁移效果因语言对而异，而基于统计和指纹的检测器则更加依赖于具体语言。

    arXiv:2609.02379v1 Announce Type: cross  Abstract: While existing work on LLM authorship attribution (AA) has made progress, available benchmarks remain limited, often focusing on English, controlled settings, or relatively outdated models, with the few multilingual studies considering only relatively short texts. We introduce MultiGhostBench, a multilingual benchmark comprising 928 books generated by five recent LLMs across six languages and three scripts, with an average length of approximately 59K words per book. The benchmark supports evaluation under domain, author, and language shifts. Evaluation of representative AA methods shows that no single method consistently performs best across settings, and performance generally degrades under distribution shifts. Transformer-based detectors can retain generator-related information across languages, although transfer effectiveness varies by language pair, whereas statistical and fingerprint-based detectors are more language-dependent. We
    
[^6]: 基于轨迹预算与选择性精炼的Text2Cypher自适应测试时推理

    Adaptive Test-Time Inference for Text2Cypher with Trace Budgeting and Selective Refinement

    [https://arxiv.org/abs/2609.02324](https://arxiv.org/abs/2609.02324)

    本文提出针对Text2Cypher的自适应测试时推理方法，通过基于问题难度动态调整候选生成预算的自适应轨迹预算和仅在有益时才执行纠正的选择性执行引导精炼两种策略，在Gemma-2-9B和Qwen-2.5-7B上将平均生成预算减少30.7%，在降低计算开销的同时提升查询生成的可靠性。

    

    大语言模型已经为结构化数据库实现了自然语言接口，但生成的查询仍可能包含语法错误、违反数据库模式，或在执行时失败。测试时推理策略无需额外训练即可提高生成可靠性，但现有方法通常使用固定的推理预算和统一的精炼策略，导致对不同复杂度的问题进行了不必要的计算。在这项工作中，我们研究了Text2Cypher的自适应测试时推理，并引入了两种策略：自适应轨迹预算，根据问题难度动态调整候选生成预算；以及选择性执行引导精炼，仅在预期额外推理有益时才应用纠正。在Gemma-2-9B和Qwen-2.5-7B上的实验表明，自适应轨迹预算将平均生成预算减少了30.7%，并将墙钟时间（摘要原文在此处截断）

    arXiv:2609.02324v1 Announce Type: new  Abstract: Large language models have enabled natural language interfaces for structured databases, but generated queries may still contain syntactic errors, violate database schemas, or fail during execution. Test-time inference strategies improve generation reliability without additional training, but existing approaches often use fixed inference budgets and uniform refinement strategies, leading to unnecessary computation across questions with different complexity levels. In this work, we investigate adaptive test-time inference for Text2Cypher and introduce two strategies: adaptive trace budgeting, which dynamically adjusts the candidate generation budget based on question difficulty, and selective execution-guided refinement, which applies correction only when additional inference is expected to be beneficial. Experiments on Gemma-2-9B and Qwen-2.5-7B show that adaptive trace budgeting reduces the average generation budget by 30.7% and wall-cl
    
[^7]: Counter-GEO-Bench：评估针对信息扭曲型生成式引擎优化的防御方法

    Counter-GEO-Bench: Evaluating Defenses Against Information-Distorting Generative Engine Optimization

    [https://arxiv.org/abs/2609.02316](https://arxiv.org/abs/2609.02316)

    提出了首个针对生成式引擎优化（GEO）攻击的防御基准Counter-GEO-Bench，通过将247个经人工验证的查询与信息保留型和信息扭曲型GEO改写配对来评估防御方法，并揭示现有三种主流防御方法最多仅能将攻击成功率相对降低5.7%。

    

    生成式引擎优化（GEO）使内容生产者能够提高其网页在生成式搜索引擎中的可见度，但同样的技术也可能被用于传递针对性错误信息——当攻击者发布看似普通的GEO优化文档，这些文档被受害大语言模型（LLM）检索并合成为扭曲的答案时。目前尚无现有基准在受控条件下评估针对这一威胁的防御方法。因此，我们提出了Counter-GEO-Bench，这是一个防御基准，将247个经人工验证、质量把关的查询与信息保留型和信息扭曲型的GEO改写版本配对，并在三个受害LLM上从攻击成功率（ASR）、误报率和答案质量三个维度评估防御方法。在Counter-GEO-Bench上，三种现成的防御方法（Granite Guardian、Llama Guard 3和NeMo Self-Check Fact-Checking）最多仅能将攻击成功率相对降低5.7%，而Granite Guardian的降低效果并不显著……

    arXiv:2609.02316v1 Announce Type: cross  Abstract: Generative engine optimization (GEO) enables content producers to increase the visibility of their web pages in generative search engines, but the same techniques can deliver targeted misinformation when adversaries publish ordinary-looking GEO-optimized documents that victim large language models (LLMs) retrieve and synthesize into distorted answers. No existing benchmark evaluates defenses against this threat under controlled conditions. Therefore, we present Counter-GEO-Bench, a defense benchmark that pairs 247 human-verified, quality-gated queries with information-preserving and information-distorting GEO rewrites, and evaluates defenses on attack success rate (ASR), false positive rate, and answer quality across three victim LLMs. Under Counter-GEO-Bench, three off-the-shelf defenses (Granite Guardian, Llama Guard 3, and NeMo Self-Check Fact-Checking) reduce ASR by at most 5.7% relative, while Granite Guardian's reduction is not s
    
[^8]: 社会科学家寻找数据时的真实信息需求

    Genuine Information Needs of Social Scientists Looking for Data

    [https://arxiv.org/abs/2609.02303](https://arxiv.org/abs/2609.02303)

    本研究通过对72名社会科学研究人员进行在线调查，让他们以“向同事求助”的方式表达对研究数据的信息需求，并分析归纳出主题、元数据等需求类别，从而揭示了用户真实复杂的信息需求与数据检索系统能力之间的差距。

    

    发表研究数据被普遍期望能够增加其复用并激发新的研究。在社会科学领域，来自问卷调查、访谈、民意测验和统计数据的数据是研究的主要资源。在数据档案和在线存储库中收集和提供研究数据有着悠久的传统。研究人员使用这些系统来识别与其研究相关的数据。然而，特别是在数据检索方面，用户复杂的信息需求似乎与数据检索系统的能力发生冲突。而检索能力又在很大程度上取决于用于描述数据的元数据方案。在本研究中，我们对72名社会科学研究人员开展了在线调查，让他们像向同事求助一样表达自己对研究数据的个人信息需求。我们分析了这些信息需求，并将其不同的组成部分归入以下类别：主题、元数据……（原文摘要在此处被截断）

    arXiv:2609.02303v1 Announce Type: new  Abstract: Publishing research data is widely expected to increase its reuse and to inspire new research. In the social sciences, data from surveys, interviews, polls, and statistics are primary resources for research. There is a long tradition to collect and offer research data in data archives and online repositories. Researchers use these systems to identify data relevant to their research. However, especially in data search, users' complex information needs seem to collide with the capabilities of data search systems. The search capabilities, in turn, depend to a high degree upon the metadata schemes used to describe the data. In this research, we conducted an online survey with 72 social science researchers who expressed their individual information needs for research data like they would do when asking a colleague for help. We analyzed these information needs and attributed their different components to the categories: topic, metadata, and in
    
[^9]: 面向证据导航的群组感知自适应检索（GAREN）

    Group-Aware Adaptive Retrieval for Evidence Navigation

    [https://arxiv.org/abs/2609.02188](https://arxiv.org/abs/2609.02188)

    提出了群组感知自适应检索框架 GAREN，通过将语料库图中的文档组织成语义连贯的群组并以组级别方式进行扩展导航，为多步推理检索提供超越文档级信号的指导，从而缓解召回受限问题。

    

    推理密集型检索面向那些无法通过表层匹配识别相关性、因而需要多步推理的查询。由于相关文档很少出现在初始候选集中，检索系统面临召回受限问题。现有方法在语料库图上以文档级别迭代扩展候选池，孤立地检查每个邻居文档，导致搜索逐渐漂移至语料库的狭窄区域。为解决这一问题，我们提出了面向证据导航的群组感知自适应检索，通过组级别扩展来探索语料库图。GAREN 根据文档在语料库图中的连接关系，将其组织成语义连贯且相互可区分的群组。每个群组所包含的信息揭示了通过该群组扩展后可访问的内容，从而提供超越单个文档级信号的导航指导。在每次迭代中，GAREN 使用组级别的导航（摘要在此处截断）

    arXiv:2609.02188v1 Announce Type: new  Abstract: Reasoning-intensive retrieval addresses queries whose relevance cannot be identified by surface-level matching, thereby requiring multi-step reasoning. Because relevant documents rarely appear in the initial candidate set, retrieval systems suffer from the bounded recall problem. Existing methods iteratively expand a candidate pool at the document level over a corpus graph, examining each neighbor in isolation and drifting toward a narrow region of the corpus. To address this problem, we propose Group-Aware Adaptive Retrieval for Evidence Navigation (GAREN), which explores the corpus graph through group-level expansion. GAREN organizes documents into semantically coherent and distinguishable groups based on their connections in the corpus graph. The information in each group indicates what can be accessed by expanding through it, providing guidance beyond individual document-level signals. At each iteration, GAREN uses a group-level navi
    
[^10]: GenCAR：面向分布外推荐的生成式反事实对齐与风险可控选择

    GenCAR: Generative Counterfactual Alignment with Risk-Controlled Selection for Out-of-Distribution Recommendation

    [https://arxiv.org/abs/2609.02162](https://arxiv.org/abs/2609.02162)

    提出GenCAR框架，将分布外推荐形式化为α-有效反事实推荐（α-VCR）问题，通过基于偏好的反事实监督与基于保形p值的Benjamini-Hochberg集合选择，在控制代理标签错误发现率（FDR）的同时提升推荐效用。

    

    在分布偏移下提供有用的推荐，对于分布外（OOD）推荐中平衡效用与风险至关重要。然而，现有的大多数OOD方法在改进排序或构建反事实候选集时，并未对所服务集合的代理标签错误发现率（FDR）加以控制。在本工作中，我们将OOD服务形式化为α-有效反事实推荐（α-VCR）问题，以便在控制代理标签FDR的同时，保留从反事实监督中学习到的候选支持集，并提出GenCAR，该方法将基于偏好的反事实监督与经校准的集合选择相结合。具体而言，GenCAR在干预环境因素的同时固定稳定偏好表示，通过偏好锚点和信任半径过滤来约束离线大语言模型（LLM）的提议，并使用保形p值进行Benjamini-Hochberg选择。我们从理论上对（摘要在此处截断）……

    arXiv:2609.02162v1 Announce Type: cross  Abstract: Serving useful recommendations under distribution shift is crucial for balancing utility and risk in out-of-distribution (OOD) recommendation. However, most existing OOD methods improve ranking or construct counterfactual candidates without controlling the proxy-label false discovery rate (FDR) of the served set. In this work, we formulate OOD serving as the $\alpha$-Valid Counterfactual Recommendation ($\alpha$-VCR) problem to retain candidate support learned from counterfactual supervision while controlling proxy-label FDR, and propose GenCAR, which couples preference-grounded counterfactual supervision with calibrated set selection. In particular, GenCAR fixes the stable-preference representation while intervening on the environmental factor, grounds offline large language model proposals through preference anchors and trust-radius filtering, and uses conformal $p$-values for Benjamini--Hochberg selection. We theoretically bound con
    
[^11]: 超越模态和谐：面向冲突感知多模态推荐的正交净化与拓扑引导混合专家模型

    Beyond Modality Harmony: Orthogonal Purification and Topology-Guided MoE for Conflict-Aware Multimodal Recommendation

    [https://arxiv.org/abs/2609.02152](https://arxiv.org/abs/2609.02152)

    该论文提出OrthoRec框架，通过协同引导正交净化（CGOP）在几何上解耦并过滤多模态特征中与协同拓扑冲突的噪声，结合拓扑引导的混合专家模型实现冲突感知的多模态推荐，突破了传统“模态和谐”假设的局限。

    

    多模态推荐系统通常依赖于一个有缺陷的“模态和谐”假设，即假定多模态特征本质上是有益的，并与用户的协同交互模式严格对齐。然而，由于欺骗性的视觉标题党和语义错配，模态与拓扑结构之间的冲突在现实场景中普遍存在。盲目地融合这些带噪声的模态不可避免地会污染纯净的协同空间，导致严重的表示失真。为解决这一问题，我们提出了OrthoRec（正交净化与拓扑引导混合专家的冲突感知多模态推荐）。其核心是引入协同引导正交净化（CGOP），在几何上将多模态特征解耦为与纯协同锚点平行和正交的方向。通过采用能量保持归一化对正交噪声进行自适应截断，CGOP纠正了欺骗性信息带来的干扰……

    arXiv:2609.02152v1 Announce Type: cross  Abstract: Multimodal Recommender Systems (MRSs) typically rely on a flawed "modality harmony" assumption, presuming that multimodal features are inherently beneficial and strictly aligned with users' collaborative interaction patterns. However, modality-topology conflicts are ubiquitous in real-world scenarios due to deceptive visual clickbaits and mismatched semantics. Blindly integrating these noisy modalities inevitably pollutes the pristine collaborative space, causing severe representation distortion. To address this, we propose Orthogonal purification and topology-guided MoE for conflict-aware multimodal Recommendation (OrthoRec). At its core, OrthoRec introduces Collaborative-Guided Orthogonal Purification (CGOP), which geometrically decouples multimodal features into directions parallel and orthogonal to a pure collaborative anchor. By adaptively truncating the orthogonal noise with an energy-preserving normalization, CGOP rectifies dece
    
[^12]: 对数外衣下的幂律：论基于图的向量搜索的可扩展性

    A Power Law in Logarithm's Clothing: On the Scalability of Graph-Based Vector Search

    [https://arxiv.org/abs/2609.02143](https://arxiv.org/abs/2609.02143)

    该论文通过跨数据集规模的实测推翻了“搜索成本随数据规模呈多重对数增长”的通行说法，发现当数据规模相对内在维度较小时，基于图的向量搜索成本实际遵循次线性幂律（约按 $N^c$、$0<c<1$ 增长），只有规模足够大时才收敛到理论预言的对数式增长。

    

    大多数向量数据库依赖基于图的索引（尤其是 HNSW 和 Vamana）来进行近似最近邻搜索。随着嵌入模型的广泛采用，这些数据库存储的数据集迅速增长。在固定精度下，搜索成本如何随数据集规模扩展？流行的答案是多重对数增长。然而，这一论断仅在特殊条件下被证明过，对于实践中使用的索引则是未经证明的断言。它也基本未经检验：标准基准测试只在一个数据集规模下测量成本，而不是跨规模测量。我们对这一论断进行了检验。答案取决于规模本身。当数据集规模 $N$ 相对于数据的内在维度较小时，搜索成本按 $N^c$ 增长（其中 $c$ 是满足 $0<c<1$ 的常数）。我们将这种扩展规律称为次线性幂律。一旦 $N$ 足够大，增长放缓至亚多项式级别，与多重对数论断一致。次线性幂律出现在所有数据集上，主要持续到……

    arXiv:2609.02143v1 Announce Type: cross  Abstract: Most vector databases rely on graph-based indexes, notably HNSW and Vamana, for approximate nearest neighbor search. With embedding models widely adopted, the datasets these databases store grow rapidly. At a fixed accuracy, how does search cost scale with dataset size? The prevailing answer is poly-logarithmic growth. Yet the claim is proven only under special conditions and asserted without proof for the indexes used in practice. It is also largely untested: standard benchmarks measure cost at one dataset size, not across sizes. We put the claim to the test. The answer depends on the scale itself. While the dataset size $N$ is small relative to the data's intrinsic dimensionality, search cost grows as $N^c$ for a constant $0<1$. We call this scaling the Sublinear Power Law. Once $N$ is large enough, growth slows to subpolynomial, consistent with the poly-logarithmic claim. The Sublinear Power Law appears on every dataset, mostly up t
    
[^13]: 超越上下文窗口：面向数据中心智能体的持久化发现上下文

    Beyond Context Windows: Persistent Discovery Context for Data-Centric Agents

    [https://arxiv.org/abs/2609.02129](https://arxiv.org/abs/2609.02129)

    本文提出“持久化发现上下文”——一种存储并复用先前意图到对象映射的轻量级记忆层，能在多个结构化数据环境中持续提升数据中心智能体的检索质量，且在词汇稀疏领域甚至优于基于元数据的检索。

    

    数据中心智能体在规划或执行之前会反复执行一个发现步骤：识别与任务相关的数据对象。然而，成功的发现结果通常被丢弃而非被复用。我们提出了持久化发现上下文，这是一种轻量级记忆层，用于存储先前的意图到对象的映射关系，并将其复用于增强未来的检索。在三个结构化数据环境中，持久化发现上下文始终比仅基于元数据的搜索获得更高的检索质量，在使用自动生成的记忆时依然有效，并揭示了一种可复现的干扰失败模式。在词汇稀疏的领域中，仅基于记忆的检索甚至可以超越基于元数据的检索。这些发现表明，发现结果构成了数据中心智能体的一种有用的可复用上下文形式。

    arXiv:2609.02129v1 Announce Type: new  Abstract: Data-centric agents repeatedly perform a discovery step before planning or execution: identifying the data objects relevant to a task. Yet successful discovery outcomes are typically discarded rather than reused. We introduce persistent discovery context, a lightweight memory layer that stores prior intent-to-object mappings and reuses them to augment future retrieval. Across three structured data environments, persistent discovery context consistently improves retrieval quality over metadata-only search, remains effective with automatically generated memories, and exposes a reproducible interference failure mode. In lexically sparse domains, memory-only retrieval can even outperform metadata-based retrieval. These findings suggest that discovery outcomes constitute a useful form of reusable context for data-centric agents.
    
[^14]: SPAR：通过真实世界空间感知增强工业级生成式POI推荐

    SPAR: Enhancing Industrial-Scale Generative POI Recommendation via Real-World Spatial Perception

    [https://arxiv.org/abs/2609.02062](https://arxiv.org/abs/2609.02062)

    论文提出SPAR统一框架，通过三个协同阶段将真实的城市空间知识（距离、方向、可达性）显式注入兴趣空间，解决现有生成式POI推荐方法预测结果偏离用户实时位置的问题。

    

    生成式兴趣点（POI）推荐通过自回归方式生成目标POI的语义ID（SID），为基于位置的服务展现了巨大前景，因为只有用户能够实际到达的推荐才有价值。然而，现有方法在由行为序列和协同信号定义的兴趣空间中运作，地理信息仅作为SID的文本属性存在，缺乏显式机制来学习或保留城市地点之间在距离、方向和可达性上的关系；因此其预测虽然在行为上合理，却往往偏离用户的实时位置。我们认为此类服务需要将真实的城市空间知识注入兴趣空间，而非仅从行为中推断地理信息。为此，我们提出SPAR——一个统一框架，其三个协同阶段共同构建、培育并保留城市空间知识：(1) 在分词层面，Spatially-I（摘要原文在此处被截断）

    arXiv:2609.02062v1 Announce Type: new  Abstract: Generative Point-of-Interest (POI) recommendation, autoregressively generating a target POI's semantic ID (SID), holds great promise for Location-Based Services, where a recommendation helps only if the user can reach it. Yet, existing methods operate within an interest space defined by behavior sequences and collaborative signals, where geography enters only as a textual attribute of the SID, leaving no explicit mechanism to learn or preserve how urban places are related by distance, direction, and reachability; their predictions are thus behaviorally plausible yet far from the user's real-time location. We argue that such services require injecting real urban spatial knowledge into the interest space, rather than inferring geography from behavior alone. Hence, we propose SPAR, a unified framework whose three synergistic stages jointly construct, cultivate, and preserve urban spatial knowledge: (1) at the tokenization level, Spatially-I
    
[^15]: GeoStore：在大场景中寻找小型店面——一个采用全局到局部非对称匹配的细粒度POI定位基准

    GeoStore: Finding Small Storefronts in Large Scenes -- A Fine-Grained POI Localization Benchmark with Global-to-Local Asymmetric Matching

    [https://arxiv.org/abs/2609.02012](https://arxiv.org/abs/2609.02012)

    该论文提出了首个针对非对称、细粒度、开放集POI定位任务的基准GeoStore，揭示了为对称视觉地点识别设计的全局描述符方法在此任务上的系统性局限，并提出了全局到局部非对称匹配方法GLAM。

    

    兴趣点（POI）定位——即将用户的近距离店面照片与大规模带地理标记的街景图像进行匹配——是地图构建、POI验证和基于位置的服务的基础。其最接近的现有范式是视觉地点识别（VPR），它假设在同一场景、相近尺度下进行对称的全图匹配；而POI定位则必须在显著的采集域差距下，将目标充满画面的近距离查询图像，与同一POI仅占据偏离中心的小区域、且周围布满视觉相似店铺的宽幅参考图像进行匹配。我们提出了GeoStore，据我们所知，这是首个专门针对这种非对称、细粒度、开放集设定的基准，并表明为对称VPR调优的全局描述符方法在该基准上存在系统性局限，因为单一全局向量会稀释小目标。我们进一步提出了GLAM（全局到局部非对称匹配）

    arXiv:2609.02012v1 Announce Type: cross  Abstract: Point-of-interest (POI) localization -- matching a user's close-up storefront photograph against large-scale geo-tagged street-view imagery -- underpins map construction, POI verification, and location-based services. Its closest existing paradigm, visual place recognition (VPR), assumes symmetric, whole-image matching of the same scene at a comparable scale; POI localization instead must match a close-up query, in which the target fills the frame, against wide references in which the same POI occupies only a small, off-center region among visually similar shops, under a substantial capture-domain gap. We introduce GeoStore, to our knowledge the first benchmark dedicated to this asymmetric, fine-grained, open-set formulation, and show that global-descriptor methods tuned for symmetric VPR are systematically limited on it, since a single global vector dilutes the small target. We further propose GLAM (Global-to-Local Asymmetric Matching
    
[^16]: 面向工业标准电网信息与交换模型问答的种子锚定预算受限图渲染

    Seed-Anchored Budget-Bounded Graph Rendering for Question Answering on Industry-Standard Power-Grid Information and Exchange Models

    [https://arxiv.org/abs/2609.02011](https://arxiv.org/abs/2609.02011)

    本文提出了一种确定性的种子锚定图渲染方法，在固定上下文预算内优先保留与查询相关的局部图证据，将电网CIM/CGMES模型上大语言模型问答的多跳证据保留率从0.12和0.00提升至全部保留，准确率从0.450提升至0.970。

    

    基于电网模型的大语言模型问答必须遵守固定的上下文预算。我们提出了种子锚定图渲染，这是一种确定性方法，能够在不添加超出共享跳数上限和上下文预算之外的方法特定调优或学习参数的情况下，优先保留与查询相关的局部图证据。该方法提供了一个可检验的条件，在该条件下，预定义的种子局部含答案渲染单元能够在贪心构造的有界上下文前缀中得到保留。我们在通过公共电网模型交换标准（CGMES）进行交换的公共信息模型（CIM）网络模型上对该方法进行了评估。在两种预算受限的CGMES编码上，朴素的描述优先渲染虽然能对每个单跳项目保留局部证据，但对多跳项目的证据保留率仅为0.12和0.00，而种子锚定渲染则保留了全部此类证据。在一个预注册的、来自SmallGrid拓扑族的全新100题测试集上，准确率从0.450提升至0.970。

    arXiv:2609.02011v1 Announce Type: cross  Abstract: Large language model question answering over power-grid models must respect a fixed context budget. We introduce seed-anchored graph rendering, a deterministic method that prioritizes query-local graph evidence without adding method-specific tuned or learned parameters beyond the shared hop bound and context budget. The method provides a checkable condition under which predefined seed-local answer-bearing render units are preserved in a greedy bounded-context prefix. We evaluate the approach on Common Information Model (CIM) network models exchanged through the Common Grid Model Exchange Standard (CGMES). On two budget-binding CGMES encodings, naive descriptions-first rendering retains local evidence for every single-hop item but only 0.12 and 0.00 of multi-hop items, whereas seed-anchored rendering retains all such evidence. On a preregistered fresh 100-item bank from the SmallGrid topology family, accuracy rises from 0.450 to 0.970 u
    
[^17]: MERGED：通过生成的专家推理蒸馏实现的多模态实体解析

    MERGED: Multimodal Entity Resolution via Generated Expert Reasoning Distillation

    [https://arxiv.org/abs/2609.01913](https://arxiv.org/abs/2609.01913)

    MERGED框架让多个大型视觉-语言模型教师为商品对标注并阐述推理，将一致标注用于监督微调、分歧标注经元裁判转化为直接偏好优化的偏好对，从而把结构化推理蒸馏到无需人工标注的7B紧凑学生模型中，实现了生产规模下低成本、低延迟的多模态商品实体解析。

    

    在商品实体解析中，关系定义会随着业务需求不断演变，而传统上每次适应变化都需要缓慢且昂贵的人工标注，这些标注往往噪声较多且缺乏推理过程。通过零样本提示的大型视觉-语言模型（VLM）可以立即适应新定义，并提供人工标注所缺乏的推理，但其成本和延迟在生产规模下令人难以承受。我们提出了MERGED——一个蒸馏框架，它不仅传递标签，还将大型教师VLM的结构化推理转移到一个紧凑的70亿参数学生模型中，且无需任何人工标注。多个教师模型为每个商品对进行标注，并阐述其决策背后的推理：一致的标注对用于监督微调，而不一致的标注则由元裁判（meta-judge）裁定，转化为用于直接偏好优化（DPO）的偏好对。在多语言电商数据集上对照人工标注的真实标准进行评估……

    arXiv:2609.01913v1 Announce Type: new  Abstract: In product entity resolution, relationship definitions constantly evolve with business needs, yet adapting to each change traditionally requires slow, costly human annotation that is often noisy and carries no reasoning. Large vision-language models (VLMs) prompted zero-shot can adapt to a new definition immediately and supply the reasoning that human labels lack, but their cost and latency are prohibitive at production scale. We present MERGED, a distillation framework that transfers not just labels but structured reasoning from large teacher VLMs into a compact 7B-parameter student, requiring no human annotation. Multiple teachers label each product pair and articulate the reasoning behind their decision: agreement pairs supply supervised fine-tuning, while disagreements are resolved by a meta-judge into preference pairs for Direct Preference Optimization. Evaluated against human-labeled ground truth on a multilingual e-commerce datase
    
[^18]: ExecRetrieval：衡量代码嵌入检索中的功能正确性差距

    ExecRetrieval: Measuring the Functional-Correctness Gap in Code-Embedding Retrieval

    [https://arxiv.org/abs/2609.01865](https://arxiv.org/abs/2609.01865)

    提出 ExecRetrieval 基准（939 个 Python 任务），通过在搜索池中植入与规范实现几乎相同、但经执行验证的有缺陷变体，首次衡量了代码嵌入检索在区分功能正确代码与错误代码上的差距。

    

    基于嵌入的代码检索是编码智能体和检索增强代码生成的核心组件，在这些场景中，检索到功能正确的代码比检索到词汇上相似的代码更为重要。现有的代码检索基准并未在搜索池中植入受控的、经执行验证的、针对每个查询规范实现的单次编辑变体，因此“嵌入模型能否在检索场景中从功能上区分正确代码与近似克隆但不正确的代码”这一问题仍未得到解答。解决这一问题需要一个搜索池本身就包含相关反事实样本的基准——即与每个规范实现几乎完全相同、且经过执行验证的有缺陷变体——从而可以直接检验检索器的排序结果是否具备功能区分能力，而不仅仅是主题或身份上的重合。我们提出了 ExecRetrieval，包含 939 个 Python 任务，每个任务都配有一个经执行验证的规范实现，以及最多四个经执行验证的……

    arXiv:2609.01865v1 Announce Type: cross  Abstract: Embedding-based code retrieval is a core component of coding agents and retrieval-augmented code generation, where retrieving correct code matters more than retrieving lexically similar code. Existing code-retrieval benchmarks do not plant controlled, execution-verified single-edit variants of each query's canonical implementation in the search pool, leaving the question of whether embeddings can functionally discriminate correct from near-clone-but-incorrect code unanswered in a retrieval setting. Resolving this requires a benchmark whose search pool itself contains the relevant counterfactuals -- execution-verified buggy variants near-identical to each canonical -- so that a retriever's rank ordering can be directly tested for functional discrimination rather than topical or identity overlap. We introduce ExecRetrieval, 939 Python tasks each paired with one execution-verified canonical implementation and up to four execution-verified
    
[^19]: 引用或拒绝：面向STEM课程视频的严格课程约束聊天机器人

    Cite or Decline: A Strict Course-Grounded Chatbot for STEM Lecture Videos

    [https://arxiv.org/abs/2609.01846](https://arxiv.org/abs/2609.01846)

    本文提出了VideoPoints平台一学期的实际部署，其检索增强聊天机器人严格基于课程材料回答问题并提供带时间戳的引用，在无证据时选择拒答，833条学生消息中实现了零课程边界越界，证明了严格课程约束设计的可行性。

    

    录制的课程视频通常配备搜索和摘要功能，是标准的学习资源。然而，学生难以针对具体课程提问，也无法根据教师的授课内容验证答案。我们报告了VideoPoints平台为期一个学期的部署情况，该平台配备了一个检索增强聊天机器人，能够基于课程讲座材料回答问题并返回带时间戳的引用。该聊天机器人仅从当前课程中检索内容，利用章节摘要来指导转录文本的排序，并返回可点击的带时间戳引用。学生用它进行快速查询和考试复习。在833条消息中，70.5%的消息包含引用，没有任何一条跨越课程边界，当没有讲座证据匹配时，聊天机器人通常选择拒绝回答而非强行作答。在用户看来，引用功能是最持续有用的特性，而练习题生成则是最强烈的未满足需求。我们还在真实环境中对该设计进行了评估。

    arXiv:2609.01846v1 Announce Type: new  Abstract: Recorded lecture videos, often enhanced with search and summarization features, are a standard study resource. However, students cannot easily ask course specific questions or verify answers against an instructor's lecture. We report a semester-long deployment of VideoPoints platform with a retrieval-augmented chatbot that answers from course lecture materials and returns timestamped citations. The chatbot retrieves only from the active course, uses chapter summaries to guide transcript ranking, and returns clickable timestamped citations. Students used it for quick lookups and exam review. Across 833 messages, 70.5% included citations, none crossed a course boundary, and when no lecture evidence matched, the chatbot usually declined rather than answering. Among the users, citations were the most consistently useful feature, while practice-question generation was the strongest unmet request. We also evaluated the design on the real-world
    
[^20]: 基于能量尾部感知部分扫描的无索引动态边检索

    Index-Free Dynamic Edge Retrieval with Energy-Tail-Aware Partial Scans

    [https://arxiv.org/abs/2609.01820](https://arxiv.org/abs/2609.01820)

    提出了无索引的动态最大内积搜索方法ETAR，通过保留查询向量中能量最大的坐标进行部分扫描、以低精度表示估计相似度并校正尾部误差后重排序候选，在保持更新简单高效的同时显著降低了查询开销。

    

    动态最大内积搜索（MIPS）返回与查询向量点积最大的K个存储向量，同时允许数据集通过插入、替换和删除操作发生变化。对于边检索而言，挑战在于在不使更新代价过高的前提下实现高召回率和快速查询。全向量扫描使更新保持简单，但需要将每个查询与所有存储向量进行比较；而索引方法虽然降低了查询成本，却需要付出在更新过程中维护额外结构的代价。我们提出了ETAR，这是一种无索引方法，能够在保持简单更新的同时减少查询工作量。ETAR保留查询向量中平方值最大的坐标，直到它们覆盖了总平方能量的大部分，并将其余部分视为低能量尾部。该方法使用紧凑的低精度表示从保留的坐标估计相似度，对被跳过的坐标进行校正，并对固定数量的候选向量进行重排序。

    arXiv:2609.01820v1 Announce Type: new  Abstract: Dynamic maximum inner-product search (MIPS) returns the $K$ stored vectors with the largest dot products with a query while allowing the dataset to change through insertions, replacements, and deletions. For edge retrieval, the challenge is to achieve high recall and fast queries without making updates expensive. Full-vector scanning keeps updates simple but compares each query with every stored vector, while indexed methods reduce query cost at the expense of maintaining additional structures during updates. We propose ETAR, an index-free method that reduces query work while preserving simple updates. ETAR keeps the query coordinates with the largest squared values until they cover most of its total squared magnitude and treats the rest as a low-magnitude tail. It estimates similarity from the retained coordinates using a compact lower-precision representation, corrects for skipped coordinates, and reranks a fixed number of candidates u
    
[^21]: hLLM：面向生成式重排序的单遍解码

    hLLM: Single Pass Decoding for Generative Reranking

    [https://arxiv.org/abs/2609.01807](https://arxiv.org/abs/2609.01807)

    提出hLLM，通过轻量自注意力头从LLM预填充隐状态读取项目-位置得分矩阵，并用匈牙利算法求最优二分匹配，在O(1)次前向传播内一次性解码全部N个序数，从而将生成式重排序的解码从逐token自回归生成变为常数次前向传播，且天然保证输出为有效排序。

    

    arXiv:2609.01807v1 公告类型： cross 摘要：大语言模型（LLM）实现了最先进的生成式排序质量，但其产生的排序结果必须经过解码，而自回归解码每生成一个token就需要一次顺序前向传播。我们观察到，排序器必须输出的token仅仅是N个序数值，用于按排序顺序命名各个项目，而这种狭窄的、具有置换结构的输出格式使得我们可以采用比从左到右生成高效得多的解码策略。我们提出了hLLM（匈牙利LLM），一种针对该输出格式专门设计的解码策略，它能够在O(1)次前向传播中解码全部N个序数。hLLM通过一个轻量级的自注意力头，从LLM预填充阶段的隐状态中读取一个N×K的项目-位置得分矩阵，然后利用匈牙利算法将该矩阵的最优二分匹配作为序数解码，从而在构造层面（而非通过事后修复）保证输出是一个有效的置换。通过对训练信号的系统性研究……

    arXiv:2609.01807v1 Announce Type: cross  Abstract: Large language models (LLMs) achieve state-of-the-art generative ranking quality, but the ranking they produce must be decoded, and autoregressive decoding spends one sequential forward pass per emitted token. We observe that the only tokens a ranker must emit are the $N$ ordinal values naming the items in ranked order, and that this narrow, permutation-structured output format admits decoding strategies which are much more efficient than left-to-right generation. We introduce hLLM (Hungarian LLM), a format-specialized decoding strategy that decodes all $N$ ordinals in $O(1)$ forward passes. hLLM reads an $N \times K$ item-position score matrix off the LLM's prefill hidden states with a lightweight self-attention head, then decodes the ordinals as the optimal bipartite assignment of that matrix via the Hungarian algorithm, yielding a valid permutation by construction rather than by repair. Through a systematic study of training signals
    
[^22]: KGVoyager：基于智能体导航的知识图谱无关问答

    KGVoyager: Knowledge Graph Agnostic Question Answering via Agentic Navigation

    [https://arxiv.org/abs/2609.01780](https://arxiv.org/abs/2609.01780)

    KGVoyager提出了一种知识图谱无关的智能体架构，通过“思考-行动-观察”循环动态探索图谱结构与语义，仅需查询端点即可从自然语言问题生成SPARQL查询，在四个基准上将F1提升约8分，同时将成本与运行时间各降低约22%。

    

    在特定领域场景下，基于RDF图谱的知识图谱问答（KGQA）仍然充满挑战，因为形式化本体和人工整理的文本-SPARQL配对数据通常不可获得。我们提出了KGVoyager，这是一种知识图谱无关的智能体架构，能够从自然语言问题生成SPARQL查询，其方式是动态发现图谱的结构与语义，且仅需要底层图谱的一个查询端点。通过采用配备搜索、探索和执行工具的“思考-行动-观察”循环，KGVoyager能够将术语映射到图谱的IRI、发现图谱结构，并通过执行反馈不断改进查询——所有这些都不需要预先存在的本体或示例。与之前的最先进方法不同，KGVoyager仅需要一个轻量级的类索引，这使其能够应用于更多的真实世界端点。在四个基准测试中，KGVoyager将F1分数提升了约8个百分点，同时将成本和运行时间各降低了约22%。

    arXiv:2609.01780v1 Announce Type: new  Abstract: Knowledge Graph Question Answering (KGQA) over RDF graphs remains challenging in domain-specific settings, where formal ontologies and curated text-SPARQL pairs are often unavailable. We present KGVoyager, a KG-agnostic agentic architecture that generates SPARQL queries from natural language questions by dynamically discovering graph structure and semantics, requiring only a query endpoint of the underlying graph. Using a think-act-observe loop with search, exploration, and execution tools, KGVoyager maps terms to graph IRIs, uncovers structure, and refines queries through execution feedback - all without pre-existing ontologies or examples. Unlike the prior state of the art, KGVoyager requires only a lightweight class index which renders it applicable for far more real-world endpoints. Across four benchmarks, KGVoyager improves F1 by ~8 points while cutting cost and runtime by ~22% each.
    
[^23]: NeoMME：面向高效微调与推理的单塔多模态原生多语言基础编码器

    NeoMME: A Single-Tower Multimodal-Native Multilingual Foundation Encoder for Efficient Fine-Tuning and Inference

    [https://arxiv.org/abs/2609.01657](https://arxiv.org/abs/2609.01657)

    NeoMME是一系列从零预训练的单塔双向多模态多语言基础编码器（260M/800M参数），采用掩码离散扩散目标训练并支持16K token上下文，在视觉文档检索等下游任务上以更低的参数与计算开销实现高效微调和推理。

    

    多模态模型通常构建于为生成式视觉-语言建模设计的架构之上，典型做法是将单独预训练的视觉编码器与因果语言模型相结合。诸如ColPali之类的视觉文档检索器将这些模型重新用作编码器，从而为非生成式任务背负了视觉语言模型（VLM）的参数与计算开销。我们提出了NeoMME，这是一个包含260M和800M参数规模的多模态多语言双向编码器系列，它在单个双向Transformer编码器中同时处理多语言文本和原始图像块。两个模型均从零开始预训练，采用掩码离散扩散文本目标，并在多模态样本上以可见图像块作为条件。两者均支持16,384个token的上下文长度，足以编码最多两幅标准的4K超高清图像。为了展示其下游能力，我们使用联合训练的稠密检索头和后期交互头对NeoMME进行微调。在ViDoRe v3基准测试上，……（摘要在此处截断）

    arXiv:2609.01657v1 Announce Type: cross  Abstract: Multimodal models often build on architectures designed for generative vision-language modeling, typically combining separately pretrained vision encoders with causal language models. Visual document retrievers such as ColPali repurpose these models as encoders, carrying over the parameter and compute overhead of a VLM for a non-generative task.   We introduce NeoMME, a family of 260M and 800M-parameter Multimodal and Multilingual bidirectional Encoders that process multilingual text and raw image patches in a single bidirectional Transformer encoder. Both models are pretrained from scratch with a masked discrete-diffusion text objective, conditioned on visible image patches for multimodal examples. Both support a 16,384-token context, enough to encode up to two standard 4K UHD images.   To demonstrate its downstream capabilities, we fine-tune NeoMME with jointly trained dense and late-interaction heads. On the ViDoRe v3 benchmark, the
    
[^24]: 从特征交互到特征传输——面向可扩展推荐模型的统一模块

    From Feature Interaction to Feature Transport - A Unified Block for Scalable Recommendation Models

    [https://arxiv.org/abs/2609.01655](https://arxiv.org/abs/2609.01655)

    提出CRAFT统一模块，将非序列化特征转化为可靠性感知的上下文场，以残差位移和记忆保持信号主动控制意图信息在堆叠模块间的传输与演化，实现推荐模型从“特征交互”到“特征传输”的范式转变。

    

    统一推荐模型旨在联合建模非序列化的多字段特征和序列化的用户行为，但现有的以交互为中心的设计主要关注在各层内部混合异构token。我们认为，可扩展的统一推荐还需要控制意图信息在堆叠模块之间如何被携带、过滤和保留。受基于流的表示动力学启发，我们引入了“特征传输”这一视角，将深度统一推荐视为一个离散的、以上下文为条件的表示演化过程。我们提出了CRAFT（上下文残差自适应特征传输模块），它将非序列化特征总结为一个可靠性感知的上下文场，并利用该上下文场为意图表示和序列表示生成残差位移信号和记忆保持信号。通过这种方式，非序列化上下文成为表示演化的主动控制器，而不是……（摘要原文在此处截断）

    arXiv:2609.01655v1 Announce Type: cross  Abstract: Unified recommendation models aim to jointly model non-sequential multi-field features and sequential user behaviors, but existing interaction-centric designs mainly focus on mixing heterogeneous tokens within each layer. We argue that scalable unified recommendation also requires controlling how intent information is carried, filtered, and preserved across stacked blocks. Inspired by flow-based representation dynamics, we introduce feature transport, a view that treats deep unified recommendation as a discrete context-conditioned representation evolution process. We propose CRAFT, a Contextual Residual Adaptive Feature Transport block, which summarizes non-sequential features into a reliability-aware contextual field and uses it to generate residual displacement and memory-preserving signals for intent and sequence representations. In this way, non-sequential context acts as an active controller of representation evolution rather than
    
[^25]: MELON：一个面向多事件文本到长视频检索的大规模数据集

    MELON: A Large-Scale Dataset for Multi-Event Text-to-Long-Video Retrieval

    [https://arxiv.org/abs/2609.01654](https://arxiv.org/abs/2609.01654)

    该论文提出了MELON——首个面向多事件文本到长视频检索的大规模数据集，通过为每个视频标注多个事件区间及文本描述，并引入多事件感知损失来提升模型对全事件与部分事件匹配的区分能力。

    

    现有的文本-视频检索数据集主要由包含单一主要事件的短视频片段组成。虽然这些数据集适合衡量基础的视觉-语言对齐能力，但在捕捉真实世界的检索场景方面存在局限——长视频通常天然包含多个语义上不同的事件，且单个文本查询可能对应多个不连续的时间段。为了弥补这一差距，我们提出了MELON，这是首个旨在将文本-视频检索扩展至具有复杂多事件结构的长视频的大规模数据集。MELON为每个视频显式标注了多个事件区间及其对应的文本描述，使得在长且未经修剪的视频中进行多事件理解的训练和评估成为可能。此外，我们提出了一种多事件感知损失函数，鼓励模型区分全事件匹配与部分事件匹配，从而带来了显著的性能提升。

    arXiv:2609.01654v1 Announce Type: new  Abstract: Existing text-video retrieval datasets primarily consist of short-form clips containing a single dominant event. While suitable for measuring basic vision-language alignment, they are limited in capturing real-world retrieval scenarios, where long-form videos naturally contain multiple semantically distinct events and a single text query may correspond to several non-contiguous temporal segments. To bridge this gap, we introduce MELON, the first large-scale dataset designed to extend text-video retrieval to long-form videos featuring complex, multi-event structures. MELON explicitly annotates multiple event intervals per video along with their corresponding textual descriptions, enabling both training and evaluation of multi-event understanding in long, untrimmed videos. In addition, we propose a multi-event aware loss that encourages models to differentiate between full-event and partial-event matches, yielding substantial improvements 
    
[^26]: 词汇鸿沟即是公平鸿沟：公共福利获取检索系统中的语域不匹配问题

    The Vocabulary Gap Is an Equity Gap: Register Mismatch in Retrieval Systems for Public-Benefits Access

    [https://arxiv.org/abs/2609.01645](https://arxiv.org/abs/2609.01645)

    该研究发现在公共福利资格检索系统中，用户用平实语言提问时的检索性能远低于用官方正式语体提问时（Recall@5从100%降至44%），表明文档与用户之间的词汇和语域鸿沟造成了实质性的公平性差距。

    

    检索增强问答系统正被越来越多地用于帮助人们了解公共福利资格，然而这些系统检索所依据的文档是使用官方机构的正式语体撰写的，而实际用户往往用平实的、非正式的或非母语的英语进行提问。我们证明，这种语域不匹配会使一个高性能的检索系统变成一个不公平的系统。我们构建了一个受控基准，包含51条公开记录的联邦福利资格规则和25个信息需求，每个信息需求均以官方语体和平实用户语体两种方式表述，同时保持标准（gold）段落不变。在BM25、TF-IDF和词项图检索器上，正式语体的评估表现近乎完美（Recall@5为96-100%），但平实语体的检索性能大幅下降（Recall@5为36-44%）。对于BM25，Recall@1从84%降至16%，Recall@5从100%降至44%，即在完全相同的信息需求上产生了56个百分点的公平性差距。我们将这一机制归因于可测量的词汇鸿沟。

    arXiv:2609.01645v1 Announce Type: new  Abstract: Retrieval-augmented question answering is increasingly used to help people navigate public-benefits eligibility, yet the documents these systems retrieve from are written in agency register while intended users often ask questions in plain, informal, or non-native English. We show that this register mismatch can turn a high-performing retrieval system into an inequitable one. We construct a controlled benchmark of 51 publicly documented federal benefit-eligibility rules and 25 information needs, each phrased in both agency register and plain user register while keeping the gold passage fixed. Across BM25, TF-IDF, and a term-graph retriever, formal-register evaluation is nearly perfect (Recall@5 96-100%), but plain-register retrieval collapses (Recall@5 36-44%). For BM25, Recall@1 falls from 84% to 16% and Recall@5 from 100% to 44%, a 56-point equity gap on identical information needs. We trace the mechanism to a measurable vocabulary gap
    
[^27]: 先想象后检索：面向大语言模型智能体的前瞻性技能检索

    Imagine Before Retrieval: Prospective Skill Retrieval for LLM Agents

    [https://arxiv.org/abs/2609.01642](https://arxiv.org/abs/2609.01642)

    提出受人类前瞻性认知启发的SkillDreamer框架，通过“先想象后检索”策略解决LLM智能体技能检索中任务查询与技能之间的失配（QSM）问题。

    

    技能检索最近已成为一种有前景的范式，用于从技能库中识别理想的执行指南，从而使大语言模型（LLM）智能体具备完成指定任务所需的程序性知识。为此，大多数现有方法通过定制检索模型或重新配置检索流程，以优先选择与任务查询语义最相关的技能。然而，我们通过实证发现，任务查询和技能天然是从不同视角构建的，即目标导向和过程导向，这导致了一个尚未被充分研究的问题，称为“查询-技能失配”（Query–Skill Misalignment, QSM）。显然，在QSM的背景下，关联理想的技能是困难的甚至是不可能的，从而阻碍了智能体正确地执行任务。作为补救措施，受人类前瞻性认知的启发，我们提出了SkillDreamer，一种用于缓解这一负面影响的全新框架……

    arXiv:2609.01642v1 Announce Type: new  Abstract: Skill retrieval has recently emerged as a promising paradigm for identifying the desirable execution guidelines from the skill gallery, thus equipping large language model (LLM) agents with the procedural knowledge to accomplish the specified task. To this end, most existing methods customize the retrieval model or reconfigure the retrieval pipeline to prioritize skills that are most semantically relevant to the task query. However, we empirically reveal that task queries and skills are naturally formulated from different perspectives, namely, objective-oriented and procedural-oriented, leading to an under-explored problem termed Query--Skill Misalignment (QSM). Clearly, it is daunting and even impossible to associate the desirable skills in the context of QSM, thus hindering the agent from correctly executing the task. As a remedy, inspired by human prospective cognition, we propose SkillDreamer, a novel framework to alleviate the negat
    
[^28]: GRAND-HC：基于图优化的作者姓名消歧

    GRAND-HC: Graph-Refined Author Name Disambiguation

    [https://arxiv.org/abs/2609.01636](https://arxiv.org/abs/2609.01636)

    提出端到端从头姓名消歧框架GRAND-HC，通过异构论文图、和谐对比学习和图优化距离矩阵，有效解决作者长尾分布导致的尾部作者过度合并问题。

    

    从头姓名消歧旨在将共享同一歧义姓名的论文划分为对应不同真实作者的聚类。现有方法存在两个关键局限：（1）作者分布固有的长尾特性会使表示学习产生偏差，导致尾部作者被过度合并；（2）现有的聚类数目估计方法在处理长论文序列时不可靠，阻碍了大规模部署。我们提出了GRAND-HC，一个完整的端到端SND框架。我们通过合著、同机构和同发表场合关系构建异构论文图，并采用图注意力网络作为嵌入骨干。和谐对比学习动态地重新加权训练损失，以抑制模型对高产作者的过拟合，从而学习具有判别性的嵌入表示。图优化距离矩阵利用图拓扑结构来优化成对距离，进一步防止尾部作者被过度合并。

    arXiv:2609.01636v1 Announce Type: new  Abstract: From-Scratch Name Disambiguation (SND) groups papers sharing an ambiguous name into clusters of distinct real-world authors. Existing methods suffer from two critical limitations: (1) inherent long-tailed author distribution biases representation learning, causing over-merging of tail authors; (2) existing cluster number estimation methods are unreliable for long paper sequences, hindering large-scale deployment. We propose \textbf{GRAND-HC}, a complete end-to-end SND framework. We construct a heterogeneous paper graph via co-author, co-organization, and co-venue relations, using a graph attention network as the embedding backbone. \textbf{Harmony Contrastive Learning (HCL)} dynamically reweights training loss to suppress overfitting to prolific authors, learning discriminative embeddings. A \textbf{Graph-Refined Distance Matrix (GRDM)} leverages graph topology to optimize pairwise distances, further preventing tail author over-merging. 
    
[^29]: 图神经网络团队推荐：一种集成化方法

    Graph Neural Team Recommendation: An Integrated Approach

    [https://arxiv.org/abs/2609.01631](https://arxiv.org/abs/2609.01631)

    该论文提出将团队推荐问题重新表述为专家协作图上的端到端链接预测任务，从而利用图神经网络捕捉团队内与跨团队的多跳协作关系及其技能间的复杂依赖，实现端到端优化。

    

    团队推荐旨在为给定的一组所需技能选择最优的专家子集，使其能够组成一个几乎必然成功的协作团队。最先进的方法是神经多标签分类器，它们将技能的稠密向量表示转换为代表最优专家子集的稀疏出现向量。然而，这类方法忽略了专家协作图中编码的专家关系和结构信息，因此无法捕捉团队内专家之间及其相关技能之间的复杂相互依赖关系。此外，技能的稠密向量是分离预训练的，独立于底层神经分类器，从而阻碍了端到端优化。在本文中，我们提出将团队推荐问题重新表述为专家协作图中的端到端链接预测，以利用团队内和跨团队的多跳协作信息。

    arXiv:2609.01631v1 Announce Type: cross  Abstract: Team recommendation aims to select an optimal subset of experts who can form an almost surely successful collaborative team for a given set of required skills. State-of-the-art methods are neural multi-label classifiers that transfer dense vector representations of skills into a sparse occurrence vector representing the optimal subset of experts. Such methods, however, overlook experts' relational and structural information encoded in the expert collaboration graph and, thus, fall short of capturing complex inter-dependencies among experts and their associated skills within teams. Moreover, the skills' dense vectors are pretrained disjointly and independently of the underlying neural classifier, hence, preventing end-to-end optimization. In this paper, we propose to reformulate the team recommendation problem into end-to-end link predictions in the expert collaboration graph to consume multi-hop intra-team and cross-team collaborations
    
[^30]: 电子商务赞助搜索中联合排序拍卖与固定价格商品列表的边际期望收益

    Marginal Expected Revenue for Jointly Ranking Auction and Fixed-Price Listings in E-Commerce Sponsored Search

    [https://arxiv.org/abs/2609.01628](https://arxiv.org/abs/2609.01628)

    该论文提出边际eCPM（meCPM）方法，将传统固定价格商品的期望收益估算框架扩展至价格动态演化的拍卖和“拍卖加一口价”（ABIN）商品，实现了电商赞助搜索中多种上架形式的联合排序。

    

    电子商务搜索排序在向相互竞争的商品列表分配曝光位时，必须平衡多个目标——相关性、用户参与度和平台收入。对于固定价格商品，估算期望收益部分的方法已经非常成熟，但当市场库存中包含混合上架形式（如纯拍卖和混合式“拍卖加一口价”（Auction with Buy It Now, ABIN）商品）时，估算就变得具有挑战性，因为此类商品的价格是动态演化的，在排序时最终成交价值尚不可知。然而，拍卖和ABIN商品在eBay等平台的库存和交易量中占有相当可观的份额，并且是个人卖家以及价值不明确的独特商品常用的上架形式。我们将标准的千次曝光期望成本（eCPM）框架扩展到拍卖和ABIN商品，推导出边际eCPM（meCPM），以刻画对价格仍在演化中的商品多展示一次曝光所带来的增量价值。

    arXiv:2609.01628v1 Announce Type: cross  Abstract: E-commerce search ranking must balance multiple objectives--relevance, user engagement, and platform revenue--when allocating impression slots to competing listings. Estimating the expected revenue component is well understood for fixed-price items, but becomes challenging when marketplace inventory includes mixed listing formats such as pure auctions and hybrid "Auction with Buy It Now" (ABIN) items, where prices evolve dynamically and the final transaction value is unknown at ranking time. Yet auction and ABIN listings account for a meaningful share of inventory and transaction volume on platforms such as eBay, and are a popular format for individual sellers and for unique items with unclear value. We extend the standard Expected Cost-per-Mille (eCPM) framework to auction and ABIN listings by deriving a marginal eCPM (meCPM) that captures the incremental value of showing one more impression of an item whose price is still evolving. T
    
[^31]: 大语言模型在推荐系统解释评估中的效用

    The Utility of LLMs in Recommender Systems Explanation Evaluation

    [https://arxiv.org/abs/2609.01627](https://arxiv.org/abs/2609.01627)

    本文研究了大语言模型作为“评判者”在推荐系统解释评估中的可靠性，旨在帮助为给定应用自动选择有效的解释方法。

    

    解释在构建可信的推荐系统（RS）中起着至关重要的作用，然而如何选择一个好的解释方法却充满挑战。现有的解释方法众多，但关于哪种方法最适合哪种场景的指导却很少。现有的解释生成方法通常产生抽象的输出，需要进一步格式化才能变得对用户友好，而可选方案似乎无穷无尽。对所有可能的选项进行基于用户的评估通常是不可行的，而自动评估指标往往要么仅评估解释器的抽象输出，要么需要与真实标准（ground truth）进行比较，而真实标准通常是不可获得的。最近的研究表明，大语言模型（LLMs）可以充当解释评估的“评判者”，但其可靠性尚未得到深入探索。本文研究了大语言模型在为给定应用选择有效解释方法方面的效用。我们首先探索……

    arXiv:2609.01627v1 Announce Type: cross  Abstract: Explanations play a crucial role in creating trustworthy recommender systems (RS), yet choosing a good explanation method presents challenges. Many explanation methods exist, but little guidance exists on which is best for which setting. Existing explanation generation methods often produce abstract outputs that require further formatting to become user-friendly, with a seemingly endless pool of options. Running user-based evaluations of all possible options is usually unfeasible, while automated evaluation metrics often either assess only the explainer's abstract output or require comparison with a ground truth, which is generally unavailable. Recent studies have shown that large language models (LLMs) can serve as ``judges'' for explanation evaluation, but their reliability has not yet been thoroughly explored. This paper studies the utility of LLMs in selecting an effective explanation method for a given application. We first explor
    
[^32]: RecEvolve：面向推荐系统的知识驱动自主智能体系统

    RecEvolve: A Knowledge-Driven Autonomous Agent System for Recommender Systems

    [https://arxiv.org/abs/2609.01622](https://arxiv.org/abs/2609.01622)

    本文提出知识驱动的自主智能体系统RecEvolve，将想法生成、代码实现、训练与评估的完整研究生命周期纳入持续自主闭环框架，首次在生产级大规模双塔召回模型上从零完成40余次自主训练迭代，突破隐藏架构瓶颈并取得NDCG约20%的相对提升，使线上用户满意度增长3.77%。

    

    智能体AI（Agentic AI）的兴起催生了向自迭代系统的转变，为生产级推荐模型的自主优化开辟了新的前沿。本文展示了一个知识驱动的自主智能体系统的实证验证，该系统被直接部署在生产级大规模双塔召回模型上。通过将完整的研究生命周期——涵盖想法生成、代码实现、离线训练和指标评估——委托给一个持续闭环的自主框架，该智能体系统从零开始成功执行了40余次完整的自主训练运行。在严格的生产规模评估下执行这些运行时，该系统系统地攻克了最新生产模型中隐藏的架构瓶颈，实现了NDCG约20%的相对提升这一突破性进展，该提升直接转化为线上生产流量中用户满意度增长3.77%。此外，部署（摘要在此处被截断）

    arXiv:2609.01622v1 Announce Type: cross  Abstract: The rise of agentic AI has catalyzed a shift toward self-iterating systems, opening new frontiers for the autonomous optimization of production recommender models. This paper presents the empirical validation of a knowledge-driven autonomous agent system, deployed directly on a production large-scale Two-Tower retrieval model. By delegating the entire research lifecycle, spanning idea generation, code implementation, offline training, and metric evaluation, to a continuous closed-loop autonomous framework, the agent system executed over 40 completed autonomous training runs from scratch. Executing these runs under rigorous production-scale evaluations, the system systematically navigated hidden architectural bottlenecks on the latest production model to achieve a breakthrough ~20% relative improvement in NDCG, a gain that translated directly to a +3.77% increase in user satisfaction in live production traffic. Furthermore, the deployme
    
[^33]: 当文献数据在材料发现中误导人工智能时

    When Literature Data Mislead Artificial Intelligence in Materials Discovery

    [https://arxiv.org/abs/2609.01621](https://arxiv.org/abs/2609.01621)

    该研究以固态电解质电导率数据为例，揭示了科学文献中普遍存在的文本-图表不匹配、单位不一致等看似合理却难以检测的错误，这些错误会以结构化标签噪声的形式污染AI训练数据库，甚至导致高达100倍的电导率误差。

    

    人工智能（AI）日益将科学文献作为数据来源，用于构建数据库、训练预测模型并指导科学发现。然而，源自文献的数据集通常假设已报道的实验值具有内在一致性并可直接重复使用。本文以固态电解质（SE）电导率数据作为材料科学的代表性案例来分析这一假设。通过追踪从原始论文到策展数据集的数值来源，我们发现了反复出现的文本与图表不匹配、坐标轴标注含糊、单位不一致以及测量背景信息缺失等问题。这些偏差在数值上往往看似合理，因此难以通过常规预处理检测到，但它们会在数据库构建和机器学习再利用过程中以结构化标签噪声的形式传播。一个跨数据库的示例表明，模糊的报道方式可造成高达100倍的电导率误差。我们的分析重新定义了数据……（摘要在此处被截断）

    arXiv:2609.01621v1 Announce Type: cross  Abstract: Artificial intelligence (AI) increasingly treats scientific literature as a data source for building databases, training predictive models, and guiding discovery. Yet literature-derived datasets often assume that reported experimental values are internally consistent and directly reusable. Here, we analyze this assumption using solid electrolyte (SE) conductivity data as a representative materials-science case. By tracing values from source articles to curated datasets, we identify recurrent text-figure mismatches, ambiguous axis annotations, unit inconsistencies, and missing measurement context. These discrepancies are often numerically plausible and therefore difficult to detect through routine preprocessing, but they can propagate as structured label noise during database construction and machine-learning reuse. A cross-database example shows how ambiguous reporting can create a 100-fold conductivity error. Our analysis reframes dat
    
[^34]: MGDiff：基于掩码GNN引导扩散的多兴趣序列推荐

    MGDiff: Multi-Interest Sequence Recommendation with Masking GNN-Guided Diffusion

    [https://arxiv.org/abs/2609.01619](https://arxiv.org/abs/2609.01619)

    提出了一种基于掩码GNN引导扩散的多兴趣序列推荐框架MGDiff，通过双层语义引导（DSG）和流行度感知引导（PAG）机制，在扩散过程中生成准确且无偏的用户兴趣表示，从而提升推荐精度。

    

    我们提出了一种新颖的基于掩码GNN引导扩散模型（MGDiff）的多兴趣序列推荐框架，旨在扩散过程中生成准确、无偏的用户兴趣信息。首先，我们提出了一种语义增强的双层语义引导（DSG）框架，该框架将引导过程分解为两个协同阶段：提取潜在物品语义和解耦多维用户意图。我们设计了一个权重自适应掩码图神经网络，通过重建缺失链接来发现超越表面共现关系的深层物品关联，同时设计了一个动态多专家网络，将用户偏好投影到不同的语义子空间中以抑制无关干扰。这种分层设计产生了结构化引导，显著提高了扩散模型的生成精度。其次，我们提出了一种流行度感知引导（PAG）机制，该机制……（原文摘要在此处截断）

    arXiv:2609.01619v1 Announce Type: new  Abstract: We propose a novel Multi-Interest Sequence Recommendation Framework with \underline{M}asking \underline{G}NN-Guided \underline{Diff}usion Model (MGDiff), designed to generate accurate, bias-free user interest information during the diffusion process. First, we propose a semantics-enhanced Dual-layer Semantic Guidance (DSG) framework, which decomposes guidance into two synergistic stages: extracting latent item semantics and decoupling multidimensional user intent. We design a Weight-adaptive Masking Graph Neural Network reconstructs missing links to uncover deep item relationships beyond superficial co-occurrence, while a Dynamic Multi-Expert Network projects user preferences into distinct semantic subspaces to suppress irrelevant interference. This hierarchical design yields structured guidance that significantly improves the generation accuracy of diffusion models. Second, We propose a Popularity-Aware Guidance (PAG) mechanism that per
    
[^35]: 面向电信SNOC环境高效云知识库检索的多智能体检索增强生成

    Multi-Agent Retrieval-Augmented Generation for Efficient Cloud Knowledge Base Search in Telecom SNOC Environment

    [https://arxiv.org/abs/2609.01618](https://arxiv.org/abs/2609.01618)

    本文提出了一个完全离线的多智能体RAG框架Athena，通过融合E5稠密检索、BM25稀疏检索和知识图谱扩展，并结合加权CombSUM融合与交叉编码器重排序，实现了电信SNOC环境中企业云知识库的高效精准检索。

    

    电信服务与网络运营中心（SNOC）依赖大量的云文档集合，包括标准操作程序（SOP）、供应商技术手册、事件报告和配置指南，以维持不间断的网络运营。在关键事件发生期间，工程师必须快速检索到准确的信息，然而传统的基于关键词和单阶段的检索方法往往难以提供精确的结果。本文提出了 Athena for Cloud Knowledge Base，这是一个完全离线的多智能体检索增强生成（RAG）框架，专为 Vodafone Idea 的 SNOC 环境中的企业云文档搜索而设计。该系统在基于 LangGraph 的编排框架中集成了使用 E5 Large V2 嵌入的稠密检索、BM25 稀疏检索以及知识图谱扩展。检索到的候选结果通过加权 CombSUM 进行融合，随后进行交叉编码器重排序和最大边际相关性处理。

    arXiv:2609.01618v1 Announce Type: cross  Abstract: Telecom Service and Network Operations Centers (SNOCs) rely on large collections of cloud documents, including Standard Operating Procedures (SOPs), vendor technical manuals, incident reports, and configuration guides, to maintain uninterrupted network operations. During critical incidents, engineers must quickly retrieve accurate information, yet traditional keyword based and single stage retrieval approaches often struggle to provide precise results.   This paper presents Athena for Cloud Knowledge Base, a fully offline, multi agent Retrieval Augmented Generation (RAG) framework designed for enterprise cloud document search in Vodafone Idea's SNOC environment. The system integrates dense retrieval using E5 Large V2 embeddings, BM25 sparse retrieval, and Knowledge Graph expansion within a LangGraph based orchestration framework. Retrieved candidates are fused using Weighted CombSUM, followed by cross encoder reranking and Maximal Marg
    
[^36]: 面向企业文档搜索的混合检索增强生成：知识图谱扩展、RRF融合与分块接地评估

    Hybrid Retrieval-Augmented Generation with Knowledge Graph Expansion, RRF Fusion, and Per-Chunk Grounded Evaluation for Enterprise Document Search

    [https://arxiv.org/abs/2609.01617](https://arxiv.org/abs/2609.01617)

    DocuSearch 提出了一种混合检索增强生成系统，通过倒数排名融合（RRF）将稠密向量语义检索、BM25全文搜索与知识图谱邻居扩展三种互补证据源加权融合（权重分别为0.50、0.35、0.15），在电信网络运营生产环境中实现了更准确、有据可依的企业文档问答。

    

    从大型企业文档库中获取准确、有据可依的答案是一个难题。仅依靠稠密向量检索在处理混合技术术语、供应商特定缩写词、或需要跨多个不相邻章节进行推理的查询时，往往表现不佳。DocuSearch 正是为解决这一差距而构建的——一个在电信网络运营生产环境中开发和评估的离线多智能体文档智能系统。DocuSearch 不依赖单一检索信号，而是整合三种互补的证据来源：基于 Qdrant 向量库并使用 BGE-Large 嵌入的语义搜索、基于 SQLite FTS5 索引的 BM25 全文搜索，以及来自结构化边表的知识图谱邻居扩展。这三份排序列表通过倒数排名融合进行合并，其中向量搜索的信号权重为 0.50，BM25 为 0.35，知识图谱为 0.15，使用……

    arXiv:2609.01617v1 Announce Type: cross  Abstract: Getting accurate, grounded answers out of large enterprise document repositories is a difficult problem. Dense vector retrieval alone frequently performs poorly on queries that mix technical terminology, vendor-specific acronyms, or require reasoning across several non-adjacent sections. DocuSearch was built to address exactly this gap - an offline, multi-agent document intelligence system developed and evaluated in a production telecom network operations environment. Rather than relying on a single retrieval signal, DocuSearch pulls together three complementary sources of evidence: semantic search over a Qdrant vector store using BGE-Large embeddings, BM25 full text search over an SQLite FTS5 index, and Knowledge Graph neighbour expansion from a structured edge table. These three ranked lists are merged through Reciprocal Rank Fusion with signal weights of 0.50 for vector search, 0.35 for BM25, and 0.15 for the knowledge graph, using 
    
[^37]: 事件记忆：通过序列模式挖掘与速度分层检索实现免训练的运维记忆

    Incident Memory: Training-Free Operational Memory through Sequential Pattern Mining and Velocity-Stratified Retrieval

    [https://arxiv.org/abs/2609.01616](https://arxiv.org/abs/2609.01616)

    该论文提出了一种无需模型训练的确定性运维记忆系统 Incident Memory，通过速度分层检索、指纹条件 PrefixSpan 序列挖掘和溯源感知的指标定义，从历史事件日志中提取有序处置手册，在 UCI ITSM 数据集上覆盖了 84.3% 的留存事故。

    

    事件响应本质上是一个记忆问题：团队会不断积累工单、追踪日志、事后复盘和维基页面，但处理下一次事件所需的知识很少能以顺序、新鲜度和来源信息完整保留的方式存储下来。我们提出了 Incident Memory（事件记忆），这是一个无需模型训练即可积累运维知识的确定性系统。它结合了三项技术：(i) 速度分层检索，以不同速率对结构性、行为性、情境性和临时性事实进行老化处理；(ii) 基于指纹条件的 PrefixSpan 挖掘，从成功的调查过程中提取有序的处置手册；(iii) 溯源感知的指标定义，通过可执行的检查来发现相互冲突的定义。在包含 141,712 个事件、24,918 起事故的 UCI ITSM 事件日志上，Incident Memory 提取出 23,110 条有序轨迹，挖掘出 39 个处置手册，并覆盖了 6,934 个留存测试事故中的 84.3%。在具有已知真实标签的受控基准测试中，它达到了 99.2% 的有序……（摘要原文在此处截断）

    arXiv:2609.01616v1 Announce Type: new  Abstract: Incident response is a memory problem: teams accumulate tickets, traces, postmortems, and wiki pages, but the knowledge needed for the next incident is rarely stored with its order, freshness, and provenance intact. We present Incident Memory, a deterministic system that accumulates operational knowledge without model training. It combines (i) velocity-stratified retrieval, which ages structural, behavioral, contextual, and ephemeral facts at different rates; (ii) fingerprint-conditioned PrefixSpan mining, which extracts ordered playbooks from successful investigations; and (iii) provenance-aware metric definitions, which detect conflicting definitions through executable checks. On the UCI ITSM event log, containing 141,712 events across 24,918 incidents, Incident Memory extracts 23,110 ordered traces, mines 39 playbooks, and covers 84.3% of 6,934 held-out incidents. On controlled benchmarks with known ground truth, it achieves 99.2% ord
    
[^38]: 略读与跳过：面向高效多模态检索的层次化自适应推理

    Skim and Skip: Hierarchical Adaptive Inference for Efficient Multimodal Retrieval

    [https://arxiv.org/abs/2609.01613](https://arxiv.org/abs/2609.01613)

    提出层次化自适应推理框架SAS，通过词元级证据筛选与深度自适应推理两大机制，在保持检索性能的同时大幅降低通用多模态检索的推理成本。

    

    通用多模态检索（UMR）正越来越多地采用多模态大语言模型（MLLM）作为统一的嵌入骨干网络，但其强大的检索性能伴随着高昂的推理成本。现有方法通常依赖于均匀的密集推理，即所有输入词元都要经过整个模型的完整处理，并使用最后一层的 [EOS] 表示进行匹配。然而，这种范式忽视了多模态检索中的两种关键异质性：各词元对最终检索嵌入的贡献高度不均衡，且不同查询所需的推理深度存在显著差异。为此，我们提出了 Skim and Skip（SAS），一种面向高效多模态检索的层次化自适应推理框架。SAS 首先执行词元级证据筛选，仅保留与最终检索嵌入最相关的输入信息，随后执行深度自适应推理，以判断……（原文摘要在此处截断）

    arXiv:2609.01613v1 Announce Type: new  Abstract: Universal multimodal retrieval (UMR) increasingly adopts multimodal large language models (MLLMs) as unified embedding backbones, but their strong retrieval performance comes at substantial inference cost. Existing methods typically rely on uniformly dense inference, where all input tokens are processed through the entire model and matched using the final-layer [EOS] representation. However, this paradigm overlooks two key forms of heterogeneity in multimodal retrieval: token contributions to the final retrieval embedding are highly uneven, and different queries require markedly different amounts of inference depth. To address this, we propose Skim and Skip (SAS), a hierarchical adaptive inference framework for efficient multimodal retrieval. SAS first performs token-level evidence selection to preserve only the input information most relevant to the final retrieval embedding, and then performs depth-adaptive inference to determine wheth
    
[^39]: MESSY STREETS：一个用于真实世界地址地理编码的基准

    MESSY STREETS: A Benchmark for Geocoding Real-World Addresses

    [https://arxiv.org/abs/2609.01612](https://arxiv.org/abs/2609.01612)

    MESSY STREETS是一个评估地理编码器处理真实杂乱网页地址的新基准，揭示了商业地理编码器的召回率比开源系统高出多达49个百分点，差距主要源于非规范地址的候选返回率差异。

    

    我们介绍了MESSY STREETS，这是一个用于评估地理编码器处理逐字网页地址的基准，包含存在性验证以及对表面形式偏差的受控测量。与基于干净或合成扰动地址的传统基准不同，MESSY STREETS包含表面形式与规范表示不一致的地址，其组成部分可能缺失、重复、格式错误或不完整。该基准基于2024年12月的Web Data Commons语料库构建，参考位置由OpenAddresses或OpenStreetMap确定。性能最强的商业地理编码器在召回率方面比开源系统高出多达49个百分点。这一差距主要由各系统在非规范地址上候选返回率的差异造成；一旦返回了候选结果，各系统的位置精度大体相当。仅非规范表面形式一项就导致了多达25个百分点的召回率差异。

    arXiv:2609.01612v1 Announce Type: cross  Abstract: We introduce MESSY STREETS, a benchmark for evaluating geocoders on verbatim web addresses, with existence verification and controlled measurement of surface-form divergence. Unlike conventional benchmarks based on clean or synthetically perturbed addresses, MESSY STREETS contains addresses whose surface forms diverge from canonical representations and whose components may be missing, repeated, malformed, or incomplete. The benchmark is constructed from the December 2024 Web Data Commons corpus, with reference locations established from OpenAddresses or OpenStreetMap.   The strongest commercial geocoders outperform open-source systems by up to 49 percentage points in recall. This gap is driven primarily by differences in candidate return rates on non-canonical addresses; once a candidate is returned, positional accuracy is broadly comparable across systems. Non-canonical surface form alone accounts for up to 25 percentage points of rec
    
[^40]: 让修订变得可理解：编辑意图、方法与应用综述

    Making Revisions Understandable: A Survey of Edit Intentions, Methods, and Applications

    [https://arxiv.org/abs/2609.01610](https://arxiv.org/abs/2609.01610)

    这是首篇从编辑意图视角系统梳理文本修订研究的综述，为修订语料库、编辑意图分类体系、识别方法及写作辅助等下游应用提供了统一视图。

    

    文本修订是文档创作中的核心过程，体现了作者如何迭代地完善、重组和改进书面内容。随着维基百科和arXiv等平台大规模修订历史数据的日益可得，自然语言处理（NLP）研究已开始超越对“做了哪些修改”的建模，转向理解“为什么要做这些修改”，即背后的编辑意图。据我们所知，这是首篇从编辑意图视角对文本修订研究进行综合梳理的综述，为数据集、分类体系、识别方法及应用提供了统一视图。我们回顾了完整修订工作流程中的已有研究，包括修订语料库构建、编辑意图分类体系设计和编辑意图识别。我们进一步对代表性数据集和方法进行归类，总结了写作辅助和文档编辑摘要等下游应用，并指出了关键的开放研究方向。

    arXiv:2609.01610v1 Announce Type: new  Abstract: Text revision is a core process in document creation, capturing how authors iteratively refine, reorganize, and improve written content. With the increasing availability of large-scale revision histories from platforms such as Wikipedia and arXiv, NLP research has begun to move beyond modeling what changes are made to understanding why they are made, i.e., the underlying edit intentions. To our knowledge, this is the first survey that synthesizes text revision research through the lens of edit intentions, providing a unified view of datasets, taxonomies, identification methods, and applications. We review prior work across the full revision workflow, including revision corpus construction, edit intention taxonomy design, and edit intention identification. We further categorize representative datasets and methods, summarize downstream applications such as writing assistance and document edit summarization, and highlight key open research 
    
[^41]: 基于双节点蒙特卡洛树搜索的对话式推荐系统高效结构化上下文建模

    Towards Effective Structured Context Modeling for Conversational Recommender Systems via Dual-node Monte Carlo Tree Search

    [https://arxiv.org/abs/2609.00618](https://arxiv.org/abs/2609.00618)

    提出DREAMS框架，通过双节点树结构（引导节点用蒙特卡洛树搜索探索对话动作以推断潜在偏好，利用节点用大语言模型将偏好状态精炼为结构化检索查询），显式建模对话式推荐系统中用户偏好的多轮演化。

    

    我们研究了对话上下文建模在对话式推荐系统（CRS）用户偏好跟踪中的作用。为此，我们提出了DREAMS，这是一种新颖的树状结构上下文建模框架，能够显式地捕捉多轮交互过程中用户偏好的演化。DREAMS引入了两种专门的节点类型，以支持对话式推荐系统的两个基本目标：偏好引导与偏好利用。具体而言，引导节点利用蒙特卡洛树搜索（MCTS）策略性地探索对话动作并推断潜在的用户偏好，而利用节点则采用基于大语言模型（LLM）的精炼方法，将跟踪到的偏好状态转化为结构化的检索查询以用于推荐。在基准数据集上的大量实验证明了DREAMS及其设计的有效性。

    arXiv:2609.00618v1 Announce Type: cross  Abstract: We investigate the role of conversational context modeling in user preference tracking for Conversational Recommendation Systems (CRSs). In this regard, we propose DREAMS, a novel tree-structured context modeling framework that explicitly captures user preference evolution throughout multi-turn interactions. DREAMS introduces two specialized node types to support the two fundamental objectives of CRSs: preference elicitation and preference exploitation. Specifically, elicitation nodes leverage Monte Carlo Tree Search (MCTS) to strategically explore conversational actions and infer latent user preferences, while exploitation nodes employ LLM-based refinement to transform the tracked preference state into structured retrieval queries for recommendation. Extensive experiments on benchmark datasets demonstrate the effectiveness of DREAMS and its design.
    
[^42]: ICEGR：面向电商搜索的意图连贯端到端生成式检索框架

    ICEGR: An Intent-Coherent End-to-End Generative Retrieval Framework for E-commerce Search

    [https://arxiv.org/abs/2608.29652](https://arxiv.org/abs/2608.29652)

    提出ICEGR框架，通过在生成式检索的语义ID构建、监督微调和偏好优化等整个训练流程中一致融入查询意图，解决电商搜索中查询意图不一致的问题，从而提升低曝光商品的检索效果和查询-商品相关性。

    

    生成式检索（GR）在电商搜索中前景广阔，但现有方法难以在整个训练流程中保持查询意图的一致性。首先，基于静态商品信息的语义ID（SID）构建方式限制了SID编码商品-意图关联的能力。其次，尽管监督微调（SFT）能够学习整个商品目录中的商品-SID映射，但由于查询到SID的训练仅依赖在线日志，低曝光商品仍然缺乏真实的查询意图监督，导致这些商品的检索性能不佳。第三，面向业务的偏好优化可能偏向热门或高价值商品，而非最匹配查询意图的商品，从而削弱了查询与商品之间的相关性。为解决这些问题，我们提出了ICEGR，一个面向电商搜索的意图连贯端到端生成式检索框架，它在整个GR训练流程中一致地融合查询意图（原文摘要在此处截断）。

    arXiv:2608.29652v1 Announce Type: new  Abstract: Generative Retrieval (GR) is promising for e-commerce search, yet existing methods struggle to maintain query-intent consistency throughout the training pipeline. First, semantic ID (SID) construction based on static product information limits the ability of SIDs to encode product-intent associations. Second, although supervised fine-tuning (SFT) learns product-SID mappings across the catalog, low-exposure products still lack real query-intent supervision because query-to-SID training relies solely on online logs, resulting in poor retrieval performance for these products. Third, business-oriented preference optimization may favor popular or high-value products over those that best match the query intent, weakening query-product relevance. To address these issues, we propose ICEGR, an Intent-Coherent End-to-End Generative Retrieval Framework for E-commerce Search that integrates query intent consistently throughout the GR training pipeli
    
[^43]: 参数化知识图谱记忆中的存储-检索差距

    A Storage-Retrieval Gap in Parametric Knowledge Graph Memory

    [https://arxiv.org/abs/2608.25489](https://arxiv.org/abs/2608.25489)

    该论文提出将知识图谱离线编译为LoRA适配器作为参数化知识层，在零查询上下文成本下实现事实知识泛化，但发现存储知识无法通过相似性检索恢复，揭示了参数化记忆中的存储-检索差距。

    

    arXiv:2608.25489v1 公告类型：交叉 摘要：图检索增强生成在查询时将检索到的子图放入模型的上下文窗口中，每次调用都支付重复的令牌成本，并在每次调用时暴露源数据。我们研究了一种替代方案：将知识图谱离线编译为每个实体一个LoRA适配器的库，这些适配器作为参数化知识层，通过注入权重而非文本来查询，在查询时零上下文成本。在MetaQA数据集上，我们发现子图训练的适配器编码了上下文无关的事实知识，这些知识能泛化到未见问题：在单值关系上，适配器相对于几乎无法闭卷的基础模型（0.007）获得了+0.243的精确匹配分数提升，且只有正确的适配器能恢复这些知识（相对于基础模型的oracle差距为+0.283）。然而，存储的知识无法通过相似性恢复：在无子图的查询下，基于嵌入和权重空间几何的检索性能均不佳。

    arXiv:2608.25489v1 Announce Type: cross  Abstract: Graph retrieval-augmented generation places retrieved subgraphs into the model's context window at query time, paying a recurring token cost and exposing source data on every call. We study an alternative: compiling a knowledge graph offline into a bank of LoRA adapters, one per entity, that serve as a parametric knowledge layer queried by injecting weights rather than text, at zero query-time context cost. On the MetaQA dataset, we find that subgraph-trained adapters encode context-free factual knowledge that generalizes to unseen questions: on single-valued relations the adapter gains $+0.243$ exact-match score over a base model that is nearly blind closed-book ($0.007$), and only the correct adapter recovers this knowledge (an oracle gap of $+0.283$ over the base model). However, the stored knowledge is not recoverable by similarity: given a query with no subgraph, embedding-based and weight-space geometry retrieval both perform at 
    
[^44]: 不值得再多花一个Token：面向高效深度研究智能体的边际价值估计

    Not Worth Another Token: Marginal Value Estimation for Efficient Deep Research Agents

    [https://arxiv.org/abs/2608.08389](https://arxiv.org/abs/2608.08389)

    该论文首次对深度研究智能体流水线中检索前、检索后和综合前三个阶段的上下文剪枝策略进行了系统性分阶段比较，发现剪枝时机比具体评分规则更关键，轻量级启发式方法可在几乎不损失质量的情况下减少多达73%的Token消耗。

    

    长程研究智能体通过迭代式的检索、聚合与综合来解决开放式任务，但上下文会快速增长，而额外证据的边际价值往往不断下降。这会导致不必要的Token成本、更高的延迟，以及最终报告生成时更嘈杂的输入。我们研究了深度研究智能体中上下文管理的边际价值估计问题，并首次对整个流水线中的剪枝策略进行了系统性的分阶段比较。我们在检索前、检索后和综合前三个阶段评估了轻量级启发式准则和一个学习型价值模型。结果表明，剪枝的有效性更多取决于剪枝应用的阶段位置，而非具体的评分规则：早期剪枝带来最大的端到端节省，而后期剪枝主要用于优化最终综合阶段的上下文。轻量级启发式方法可在几乎不损失质量的情况下将Token使用量降低多达73%，学习型剪枝模型……（原文摘要在此处截断）

    arXiv:2608.08389v2 Announce Type: replace  Abstract: Long-horizon research agents solve open-ended tasks through iterative retrieval, aggregation, and synthesis, but context grows rapidly while the marginal value of additional evidence often declines. This leads to unnecessary token cost, higher latency, and noisier inputs for final report generation. We study marginal value estimation for context management in deep research agents and present the first systematic stage-aware comparison of pruning strategies across the pipeline. We evaluate lightweight heuristic criteria and a learned value model at pre-retrieval, post-retrieval, and pre-synthesis stages. Our results show that pruning effectiveness depends more on where pruning is applied than on the specific scoring rule: early pruning yields the largest end-to-end savings, while later pruning mainly refines the final synthesis context. Lightweight heuristics reduce token usage by up to 73% with little quality degradation, learned pru
    
[^45]: SABER-Math：面向数学信息检索评估的自动化基准

    SABER-Math: Automated Benchmark for Information Retrieval Evaluation in Mathematics

    [https://arxiv.org/abs/2606.29894](https://arxiv.org/abs/2606.29894)

    该论文提出了首个无需专家标注、完全自动化的数学信息检索评估基准SABER-Math，它从28.3万道高中数学题出发自动构建具有挑战性的重排序任务，以克服现有基准无法捕捉细粒度数学相关性的问题。

    

    随着智能体AI系统处理越来越复杂的数学任务，它们越来越依赖信息检索（IR）来搜索问题数据库、定理库和教育资源。然而，选择合适的检索器仍然很困难，因为无法直接将其对下游性能的影响隔离开来加以评估。另一方面，现有的检索专用基准往往无法捕捉细粒度的数学相关性，从而错误地惩罚相关文档。我们通过引入SABER-Math来填补这一空白，这是首个无需专家标注、完全自动化的数学信息检索评估基准。SABER-Math从28.3万道带解答的高中数学题目出发，通过三个步骤构建具有挑战性的重排序任务：(i) 首先，大语言模型为每道题目提取简洁的解题摘要和数学主题；(ii) 然后，利用基于本体主题和词汇解答相似性的方法为每个查询发现相关文档……（原文摘要在此处截断）

    arXiv:2606.29894v2 Announce Type: replace-cross  Abstract: As agentic AI systems tackle more complex mathematical tasks, they increasingly rely on information retrieval (IR) to search problem databases, theorem libraries, and educational resources. However, choosing the right retriever remains difficult, as it is infeasible to directly isolate its effect on downstream performance. On the other hand, existing retrieval-specific benchmarks often fail to capture fine-grained mathematical relevance, penalizing relevant documents. We address this gap by introducing SABER-Math, the first fully automated benchmark for evaluating mathematical IR without expert annotation. Starting from 283K high-school-level math problems with solutions, SABER-Math builds challenging reranking tasks in three steps: (i) first, LLMs extract concise solution summaries and mathematical topics for each problem; (ii) then, per-query relevant documents are discovered using ontology topic-based and lexical solutions-s
    
[^46]: 超越检索：学习紧凑的用户表示以实现可扩展的大语言模型个性化

    Beyond Retrieval: Learning Compact User Representations for Scalable LLM Personalization

    [https://arxiv.org/abs/2606.04547](https://arxiv.org/abs/2606.04547)

    提出TAP-PER框架，通过时序注意力前缀嵌入将用户偏好编码为紧凑的可学习表示，摆脱了检索式个性化对检索质量的依赖以及参数式个性化随用户规模增长的高昂存储成本，实现可扩展的大语言模型个性化。

    

    个性化大语言模型需要在使模型行为适应个体用户的同时，保持鲁棒性和部署规模上的效率。现有方法通常在输入层面对大语言模型进行个性化，即通过检索用户历史或构建用户画像提示，或者在参数层面对其进行个性化，即为每个用户维护特定的参数高效模块。前者使个性化效果依赖于检索质量和提示设计，而后者会产生随用户规模增长而增加的存储和维护成本。为了解决这些局限，我们提出了TAP-PER（Temporal Attentive Prefix for PERsonalization，时序注意力前缀个性化），这是一个基于前缀的框架，它将用户偏好编码为可学习的表示，避免了将用户历史序列化到提示中，并用轻量级的用户状态前缀嵌入取代了繁重的每用户适配器模块。受个性化推荐系统的启发，TAP-PER将用户……（原文摘要在此处截断）

    arXiv:2606.04547v3 Announce Type: replace-cross  Abstract: Personalizing large language models requires adapting model behavior to individual users while preserving robustness and deployment-scale efficiency. Existing approaches typically personalize LLMs either at the input level, by retrieving user histories or constructing profile prompts, or at the parameter level, by maintaining user-specific parameter-efficient modules. The former makes personalization sensitive to retrieval quality and prompt design, whereas the latter incurs storage and maintenance costs that grow with the user population. To address these limitations, we propose TAP-PER (Temporal Attentive Prefix for PERsonalization), a prefix-based framework that encodes user preferences as learnable representations, avoiding the serialization of user histories into prompts and replacing heavy per-user adapters with lightweight user-state prefix embeddings. Inspired by personalized recommendation systems, TAP-PER decomposes u
    
[^47]: CaST-POI：面向下一兴趣点推荐的候选条件化时空建模

    CaST-POI: Candidate-Conditioned Spatiotemporal Modeling for Next POI Recommendation

    [https://arxiv.org/abs/2604.20845](https://arxiv.org/abs/2604.20845)

    提出CaST-POI，通过在目标注意力中引入时间新近度和地理距离两个分桶偏置，使不同候选点以不同的注意力权重读取同一用户轨迹，从而改进下一兴趣点推荐。

    

    下一兴趣点（POI）推荐基于用户的签到历史对其可能前往的下一个位置进行排序。大多数最新的排序器将轨迹压缩为单一用户向量，并通过相同的表示对所有候选点进行评分，忽略了每个候选点都带有地理坐标这一事实，也忽略了历史访问的相关性取决于候选点所处的位置。来自点击率预测的目标注意力机制使用户表示以被评分的物品为条件，但其算子是为没有几何信息的网络物品设计的。本文提出了CaST-POI，一种候选条件化的POI排序器，它保留了标准的目标注意力读取器，并在注意力logits中加入了两个分桶偏置：一个针对每次历史访问的时间新近度，另一个针对该访问与被评分候选点之间的距离。因此，不同的候选点会以不同的注意力权重读取同一轨迹，且这些权重基于真实的地理距离。

    arXiv:2604.20845v2 Announce Type: replace-cross  Abstract: Next Point-of-Interest (POI) recommendation ranks a user's likely next location based on check-in history. Most recent rankers compress the trajectory into a single user vector and score every candidate through the same representation, ignoring that every candidate carries geographic coordinates and that the relevance of a past visit depends on where the candidate is located. Target attention from click-through-rate prediction conditions the user representation on the scored item, but its operator was developed for web items without geometry. This paper proposes CaST-POI, a candidate-conditioned POI ranker that keeps a standard target-attention reader and adds two bucketised biases to the attention logits: one for the recency of each past visit and another for its distance to the candidate being scored. Different candidates therefore read the same trajectory with different attention weights, grounded in real geographic distance
    

