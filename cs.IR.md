# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Predictable Failure in Multi-Hop Retrieval: Score-Distributional Confidence Scoring and Abstention](https://arxiv.org/abs/2609.22056) | 该论文证明多跳检索失败集中在结构上可预测的子群体中，并提出RegimeAbstain框架，通过融合多种ANN分数特征的检索置信度评分（RCS）实现有原则的弃权决策，从而可证明地减少高置信失败。 |
| [^2] | [AutoRecLab: Describe the Experiment, Get the Code!](https://arxiv.org/abs/2609.21863) | AutoRecLab是一个基于Python的自主推荐系统实验平台，能根据自然语言描述的研究想法自动推导实验需求、构建验证原型并迭代扩展为完整实验代码，演示中9次运行成功8次且每次成本仅约1美元。 |
| [^3] | [Do We Care About Personalization and Explainability? An Interview Study with News Recommendation Engineers](https://arxiv.org/abs/2609.21547) | 本研究通过对九家新闻机构15位工程师的访谈发现，由于用户追踪、编辑控制和资源限制等顾虑，个性化在新闻推荐中并非总是理想选择，而可解释性在实际生产系统中也鲜少被优先考虑。 |
| [^4] | [Adaptive Preference Modeling via Explicit Indirect Relational Learning for Personalized Fashion Matching](https://arxiv.org/abs/2609.21475) | 本文提出APCL框架，通过相关性引导的自适应聚合机制显式建模直接与间接关系信号，并结合功能视图对比学习策略，在稀疏多模态数据下实现个性化的时尚互补推荐。 |
| [^5] | [Auto-Bidding with Disentangled Advertiser Profiles and Train-Free Adaptation](https://arxiv.org/abs/2609.21308) | 提出ADAPT框架，通过解耦广告主画像同时建模共有与私有信息，并实现免训练的冷启动自适应，从而为自动出价提供更有效的个性化出价策略。 |
| [^6] | [Hybrid GPU-CPU Retrieval for Personalized Search at Ultra-Large Scale](https://arxiv.org/abs/2609.21281) | 该论文提出一种混合GPU-CPU协同服务系统，通过高深度GPU通路与高广度CPU通路的编排设计，解决了超大规模个性化搜索中个性化深度与库存覆盖广度难以兼得的“个性化-规模悖论”。 |
| [^7] | [Verify, Don't Trust: Agentic Model Development for Video Discovery Retrieval at Scale](https://arxiv.org/abs/2609.21257) | 提出了 EvoPilot——一种带人工把关的长周期在线自动研究方法，通过角色化智能体、持久化实验记录和确定性检查来保障结论的有效性，并在支撑视频发现产品 Video Deep Dive 的检索系统上完成了为期 37 天的大规模验证。 |
| [^8] | [MAGIC: Marginal-Guided Compression with Optimal Transport for Efficient Visual Document Retrieval](https://arxiv.org/abs/2609.21018) | 该论文提出MAGIC，一种基于边缘引导最优传输的无需训练事后压缩方法，通过使压缩目标与后期交互检索中稀疏、非均匀的补丁使用模式对齐，在保持检索质量的同时显著降低多向量视觉文档检索的索引存储和MaxSim评分开销。 |
| [^9] | [High-probability guarantees for linear accessibility in feature superposition](https://arxiv.org/abs/2609.09556) | 该论文将特征叠加中的线性能及性建模为压缩感知问题，证明了充分维度只需线性规模（而非此前最坏情况的二次方限制）即可高概率地恢复同时激活的特征，从而量化了线性表示假设的几何约束并为稀疏自编码器和神经可解释性评估提供了理论框架。 |
| [^10] | [IntHQ: Task-Interactive Hierarchical Query on Dual-Stream Representations for Generative Recommendation](https://arxiv.org/abs/2608.09634) | 提出IntHQ多任务生成式推荐框架，通过在双流表示上引入任务交互式分层查询机制，解决了现有多任务推荐系统中的源坍塌、关系坍塌和层级坍塌这三重固有问题。 |
| [^11] | [ANNLib: A Development Framework for Efficient Approximate Nearest Neighbor Search](https://arxiv.org/abs/2607.17582) | ANNLib通过将基于图的ANNS系统中的算法与数据结构组件解耦并独立优化，同时实现了高性能与灵活功能，支持过滤搜索、完全动态更新、快照历史查询和范围搜索等复杂场景。 |
| [^12] | [Recall Before Rerank: Benchmarking Deep Learning Models for Large-Scale Code-to-Code Retrieval](https://arxiv.org/abs/2606.27401) | 本文对大规模代码检索第一阶段召回中的深度学习模型进行了多语言、多数据集基准测试，揭示了其在TB级代码库上的精度与可扩展性局限，并提出基于LLM的代码规范化与查询重写方案显著提升检索精度。 |
| [^13] | [Transferable knowledge graphs with executable learned operators for algorithm design](https://arxiv.org/abs/2603.27922) | 提出生成式可执行算法知识图谱（GEAKG），将算法设计中的过程性知识表示为可迁移的图谱，使习得算子与边权重能够跨领域、跨数据集复用，无需为每个新领域重新构建知识。 |
| [^14] | [Beyond Final Answers: CRYSTAL Benchmark for Transparent Multimodal Reasoning Evaluation](https://arxiv.org/abs/2603.13099) | 提出CRYSTAL诊断基准，通过可验证的中间步骤及Match F1与Ordered Match F1两个新指标评估多模态大模型的透明推理能力，并揭示出仅靠最终答案准确率无法发现的系统性推理缺陷。 |
| [^15] | [IntTravel: A Real-World Dataset and Generative Framework for Integrated Multi-Task Travel Recommendation](https://arxiv.org/abs/2602.11664) | 该论文发布了首个基于高德地图大规模真实数据的综合出行推荐数据集IntTravel（含1.63亿用户、730万POI、41亿次交互），并提出了一个端到端的仅解码器生成式框架，实现覆盖出发时间、出行方式、目的地等多任务的一体化出行推荐。 |
| [^16] | [RankSteer: Can Pointwise LLM Rankers Be Calibrated at the Representation Level?](https://arxiv.org/abs/2602.03422) | RankSteer提出了一种推理时的激活引导框架，通过沿决策、证据和角色等多个方向进行投影干预来校准逐点LLM排序器的“校准鸿沟”，在无需更新模型权重或跨文档比较的情况下显著提升排序性能。 |
| [^17] | [Compass: General Filtered Search across Vector and Structured Data](https://arxiv.org/abs/2510.27141) | Compass 提出了一个统一框架，复用现有索引结构（HNSW、IVF、B+树）并通过协作式查询执行策略，在不设计新索引的情况下实现了支持任意合取、析取和范围谓词的向量与结构化数据通用过滤搜索。 |

# 详细

[^1]: 多跳检索中的可预测失败：基于分数分布的置信度评分与弃权机制

    Predictable Failure in Multi-Hop Retrieval: Score-Distributional Confidence Scoring and Abstention

    [https://arxiv.org/abs/2609.22056](https://arxiv.org/abs/2609.22056)

    该论文证明多跳检索失败集中在结构上可预测的子群体中，并提出RegimeAbstain框架，通过融合多种ANN分数特征的检索置信度评分（RCS）实现有原则的弃权决策，从而可证明地减少高置信失败。

    

    多跳检索的失败并非在所有查询上均匀分布：它们聚集在结构上可预测的子群体中。我们证明了两个形式化刻画这一结构的结果。第一（CWAR可约简性）：置信失败的减少当且仅当检索特征携带关于成功的互信息时才可实现，该条件在LLM评判流水线中得到满足，但在纯稠密检索设置中明显更弱，这解释了两种机制之间的AUC-AC差距。第二（特征机制互补性）：没有任何单一的ANN分数特征能在所有失败机制中取得最佳预测性能；主导特征因数据集而异（MuSiQue上为查询长度，HoVer上为hop-1集中度），并且一个构造性见证对表明每个特征在一种机制中是必要的，而在另一种机制中则无贡献。我们在RegimeAbstain中实例化这些原则，该系统计算检索置信度分数（RCS），即最多九个查询-ANN分数的逻辑函数……

    arXiv:2609.22056v1 Announce Type: cross  Abstract: Multi-hop retrieval failures are not uniformly distributed across queries: they cluster in structurally predictable subpopulations. We prove two results formalizing this structure. First (CWAR Reducibility): confident-failure reduction is achievable if and only if retrieval features carry mutual information about success, a condition satisfied by LLM-judge pipelines but substantially weaker in dense-only settings, explaining the AUC-AC gap between regimes. Second (Feature Regime Complementarity): no single ANN score feature achieves best predictive performance across all failure regimes; the dominant feature differs between datasets (query length on MuSiQue, hop-1 concentration on HoVer), and a constructive witness pair shows each is necessary in one regime and non-contributory in the other. We instantiate these principles in RegimeAbstain, which computes a Retrieval Confidence Score (RCS), a logistic function of up to nine query-ANN s
    
[^2]: AutoRecLab：描述实验，获取代码！

    AutoRecLab: Describe the Experiment, Get the Code!

    [https://arxiv.org/abs/2609.21863](https://arxiv.org/abs/2609.21863)

    AutoRecLab是一个基于Python的自主推荐系统实验平台，能根据自然语言描述的研究想法自动推导实验需求、构建验证原型并迭代扩展为完整实验代码，演示中9次运行成功8次且每次成本仅约1美元。

    

    实证评估是推荐系统研究的核心，但将实验设计转化为可执行代码仍然是一项手动且容易出错的任务。我们提出了AutoRecLab，一个基于Python的自主推荐系统实验室，能够从自然语言提示出发自动完成推荐系统实验。给定一个研究想法，AutoRecLab会推导出明确的实验需求，构建并验证原型，然后迭代地将其扩展为所需的完整实验。该工作流程结合了用于文档查询的检索增强生成（RAG）、静态类型验证以及执行引导的树搜索。在我们的演示中，AutoRecLab自主实现了一项显式到隐式反馈转换的研究。在跨六种算法和三个数据集的基线比较中，9次运行中有8次成功，使用GPT-5.4-mini时平均每次运行成本约为1美元。

    arXiv:2609.21863v1 Announce Type: new  Abstract: Empirical evaluation is central to recommender-systems (RecSys) research, but turning experimental designs into executable code remains a manual and error-prone task. We present AutoRecLab, a Python-based autonomous RecSys lab that automates RecSys experiments from natural-language prompts. Given a research idea, AutoRecLab derives explicit experiment requirements, builds and validates a prototype, and iteratively expands it into the requested full experiment. The workflow combines retrieval-augmented generation (RAG) for documentation lookup, static type verification, and execution-steered tree search. In our demonstration, AutoRecLab autonomously implements an explicit-to-implicit feedback conversion study. In a baseline comparison across six algorithms and three datasets, 8 of 9 runs succeed at an average cost of approx- imately $1 per run with GPT-5.4-mini.
    
[^3]: 我们是否在意个性化与可解释性？与新闻推荐工程师的访谈研究

    Do We Care About Personalization and Explainability? An Interview Study with News Recommendation Engineers

    [https://arxiv.org/abs/2609.21547](https://arxiv.org/abs/2609.21547)

    本研究通过对九家新闻机构15位工程师的访谈发现，由于用户追踪、编辑控制和资源限制等顾虑，个性化在新闻推荐中并非总是理想选择，而可解释性在实际生产系统中也鲜少被优先考虑。

    

    推荐系统可解释性的研究大多聚焦于最终用户，而忽视了构建和维护这些系统的人员的观点，以及模型调试等潜在应用场景。在本研究中，我们考察了新闻工程师及相关技术利益相关者在实践中如何看待和实施个性化与可解释性。我们在九家新闻机构中开展了15次半结构化访谈，涵盖公共和私营部门的不同地区，以探究塑造其技术方法的挑战与动机。研究结果表明，个性化对新闻机构而言并不总是一个直接可行或理想的选择，因为对用户追踪、编辑控制以及资源限制的顾虑往往限制了其应用。即使在生产环境中已部署个性化新闻推荐系统的机构中，可解释性也很少被优先考虑，日常运营需求往往占据主导地位。

    arXiv:2609.21547v1 Announce Type: cross  Abstract: Research on explainability in recommender systems largely centers on end users, overlooking the perspectives of those who build and maintain these systems and their potential use cases such as model debugging. In this study, we examine how news engineers and related technical stakeholders perceive and implement personalization and explainability in practice. We conducted 15 semi-structured interviews across nine news organizations, spanning diverse regions in both public and private sectors, to investigate the challenges and motivations shaping their approaches. Our findings reveal that personalization is not always a straightforward or desirable choice for news organizations, as concerns around user tracking, editorial control, and resource constraints often limit its adoption. Even among organizations implementing personalized news recommender systems in production, explainability is rarely prioritized, with day-to-day operational de
    
[^4]: 基于显式间接关系学习的自适应偏好建模：面向个性化时尚搭配

    Adaptive Preference Modeling via Explicit Indirect Relational Learning for Personalized Fashion Matching

    [https://arxiv.org/abs/2609.21475](https://arxiv.org/abs/2609.21475)

    本文提出APCL框架，通过相关性引导的自适应聚合机制显式建模直接与间接关系信号，并结合功能视图对比学习策略，在稀疏多模态数据下实现个性化的时尚互补推荐。

    

    个性化时尚互补推荐需要在稀疏和多模态数据条件下，联合建模用户偏好与物品兼容性。现有方法通常通过图传播隐式地捕获高阶关系信号，或依赖直接交互数据，这限制了它们显式建模间接偏好与兼容性关系的能力。为解决这一局限，我们提出了一种结合对比学习的自适应偏好框架（APCL），在统一的推荐架构中显式建模直接与间接关系信号。具体而言，APCL通过相关性引导的自适应聚合机制构建间接的用户-物品和物品-物品关系，并将其表示为专门的个性化视图和兼容性视图。为进一步改进表示学习，我们还引入了一种功能视图对比学习策略，用于对齐直接……

    arXiv:2609.21475v1 Announce Type: new  Abstract: Personalized fashion complementary recommendation requires jointly modeling user preferences and item compatibility under sparse and multimodal data conditions. Existing approaches often capture higher-order relational signals implicitly through graph propagation or rely on direct interaction data, limiting their ability to explicitly model indirect preference and compatibility relationships. To address this limitation, we propose an Adaptive Preference with Contrastive Learning framework (APCL) that explicitly models both direct and indirect relational signals within a unified recommendation architecture. Specifically, APCL constructs indirect user-item and item-item relationships through a correlation-guided adaptive aggregation mechanism and represents them as dedicated personalization and compatibility views. To improve representation learning, we further introduce a functional view contrastive learning strategy that aligns direct an
    
[^5]: 基于解耦广告主画像与免训练自适应的自动出价

    Auto-Bidding with Disentangled Advertiser Profiles and Train-Free Adaptation

    [https://arxiv.org/abs/2609.21308](https://arxiv.org/abs/2609.21308)

    提出ADAPT框架，通过解耦广告主画像同时建模共有与私有信息，并实现免训练的冷启动自适应，从而为自动出价提供更有效的个性化出价策略。

    

    自动出价是现代广告系统的关键组成部分，可为每个广告主提供个性化的出价策略。基于画像的方法通过刻画每个个体来实现个性化，已在推荐等领域被证明行之有效。然而，尽管广告主的出价行为多种多样，此类方法在自动出价中的应用仍然有限。一个主要原因在于，构建和利用广告主画像面临诸多挑战：提取纯净的画像并非易事，同时建模共有信息和私有信息十分困难，且画像更新与冷启动自适应也仍具挑战性。为解决这些问题，我们提出了ADAPT，一个具有解耦广告主画像和免训练自适应能力的自动出价框架。ADAPT引入了两阶段训练范式……（原文摘要在此处截断）

    arXiv:2609.21308v1 Announce Type: new  Abstract: Auto-bidding is a key component of modern advertising systems that provides a personalized bidding strategy for each advertiser. By characterizing each individual, profile-based methods achieve personalization and have proven effective in domains such as recommendation. However, despite the diverse bidding behavior of advertisers, their application to auto-bidding remains limited. A primary reason is that constructing and leveraging advertiser profiles face several challenges: extracting pure profiles is non-trivial, modeling common and private information simultaneously is difficult, and profile updating and cold-start adaptation remain challenging. To tackle these issues, we propose \textbf{ADAPT}, an \underline{\textbf{A}}uto-bidding framework with \underline{\textbf{D}}isentangled \underline{\textbf{A}}dvertiser \underline{\textbf{P}}rofiles and \underline{\textbf{T}}raining-free adaptation. ADAPT introduces a two-stage training para
    
[^6]: 超大规模个性化搜索的GPU-CPU混合检索

    Hybrid GPU-CPU Retrieval for Personalized Search at Ultra-Large Scale

    [https://arxiv.org/abs/2609.21281](https://arxiv.org/abs/2609.21281)

    该论文提出一种混合GPU-CPU协同服务系统，通过高深度GPU通路与高广度CPU通路的编排设计，解决了超大规模个性化搜索中个性化深度与库存覆盖广度难以兼得的“个性化-规模悖论”。

    

    在万亿级文档规模的用户生成内容上进行基于嵌入的检索，暴露了两种生产需求之间的尖锐冲突：对具有丰富用户意图的查询实现深度且富有表现力的个性化，以及在固定的延迟和资源预算下对海量库存的广泛覆盖。我们将这一矛盾特征化为“个性化-规模悖论”：在GPU内存中托管完整的推理库存资源消耗过高，而CPU计算无法在延迟关键路径上执行同样重交互的模型。我们提出了一种混合GPU-CPU协同服务系统，通过编排（而非新的模型类别）来解决这一悖论。高深度的GPU通路在约十亿文档规模的精选在线池上融合检索与交互预排序，而高广度的CPU通路则以轻量级个性化评分搜索一个规模约大二十倍的独立选择的在线库存。二者之一或两者……

    arXiv:2609.21281v1 Announce Type: cross  Abstract: Embedding-based retrieval on user-generated content at the trillion-document scale exposes a sharp conflict between two production demands: deep, expressive personalization for queries with rich user intent, and broad coverage of a massive inventory under fixed latency and resource budgets. We characterize this as the personalization-scale paradox: hosting the full serving inventory in GPU memory is too resource intensive, while CPU compute cannot execute the same interaction-heavy model on the latency-critical path.   We present a hybrid GPU-CPU co-serving system that resolves the paradox through orchestration rather than a new model class. A high-depth GPU pathway fuses retrieval and interaction pre-ranking over a curated online pool on the order of a billion documents, while a high-breadth CPU pathway searches an independently selected online inventory roughly twenty times larger with lightweight personalized scoring. Either or both
    
[^7]: 验证而非信任：面向大规模视频发现检索的智能体化模型开发

    Verify, Don't Trust: Agentic Model Development for Video Discovery Retrieval at Scale

    [https://arxiv.org/abs/2609.21257](https://arxiv.org/abs/2609.21257)

    提出了 EvoPilot——一种带人工把关的长周期在线自动研究方法，通过角色化智能体、持久化实验记录和确定性检查来保障结论的有效性，并在支撑视频发现产品 Video Deep Dive 的检索系统上完成了为期 37 天的大规模验证。

    

    大型语言模型（LLM）智能体能够提出、实现并评估模型变更。自动研究循环通过在自包含程序上进行分钟级的迭代展示了这种能力。而在线自动研究则涉及异步系统、长达数小时的变体，以及可能影响产品的长达数周的实验活动。当代码变更实际未生效、数据窗口发生泄漏、评估器语义发生漂移，或两个实验分支经过不同的服务链路时，一次已完成的运行仍可能支持一个无效的结论。我们提出了 EvoPilot，一种用于长周期在线自动研究的人工把关方法。角色特定的智能体通过版本化的领域技能和类型化适配器执行每一轮任务。持久化记录保存实验与失败信息；确定性检查强制执行已记录的经验教训。我们针对为 Video Deep Dive（VDD）提供支持的检索系统开展了一项为期 37 天的实验活动，VDD 是一种在用户打开搜索后用于发现后续视频的在线体验。

    arXiv:2609.21257v1 Announce Type: cross  Abstract: Large language model (LLM) agents can propose, implement, and evaluate model changes. Autoresearch loops demonstrate this capability through minutes-scale iterations on a self-contained program. Online autoresearch instead spans asynchronous systems, hours-long variants, and weeks-long campaigns that can influence a product. A completed run can still support an invalid conclusion when a code change is a no-op, data windows leak, evaluator semantics drift, or the two arms traverse different serving funnels. We present EvoPilot, a human-gated method for long-horizon online autoresearch. Role-specific agents execute each round through a versioned domain skill and typed adapter. Durable records preserve experiments and failures; deterministic checks enforce recorded lessons.   We study a 37-day campaign for the retrieval system that powers Video Deep Dive (VDD), an online experience for discovering follow-on videos after a user opens a see
    
[^8]: MAGIC：面向高效视觉文档检索的边缘引导最优传输压缩

    MAGIC: Marginal-Guided Compression with Optimal Transport for Efficient Visual Document Retrieval

    [https://arxiv.org/abs/2609.21018](https://arxiv.org/abs/2609.21018)

    该论文提出MAGIC，一种基于边缘引导最优传输的无需训练事后压缩方法，通过使压缩目标与后期交互检索中稀疏、非均匀的补丁使用模式对齐，在保持检索质量的同时显著降低多向量视觉文档检索的索引存储和MaxSim评分开销。

    

    近年来的视觉文档检索（VDR）系统（如ColPali）使用多向量页面嵌入，其中补丁级向量能够实现细粒度的证据匹配，但会带来巨大的索引存储和MaxSim评分开销。事后合并提供了一条实现高效VDR的实用途径，即无需重新训练检索器即可降低该成本，但其均匀的重构目标与后期交互检索所诱导的稀疏、非均匀补丁使用模式匹配不佳。在激进压缩的情况下，这种不匹配可能导致保留极少使用的补丁，同时将检索活动集中在过少的保留代表上。为了解决这种不匹配问题，我们提出了边缘引导最优传输压缩（MAGIC），这是一种无需训练的事后压缩器，可与冻结的多向量嵌入配合实现高效检索。MAGIC推导了一个由MaxSim导出的压缩代理目标，并通过双边缘熵正则化最优传输对其进行优化。

    arXiv:2609.21018v1 Announce Type: cross  Abstract: Recent visual document retrieval (VDR) systems such as ColPali use multi-vector page embeddings, in which patch-level vectors enable fine-grained evidence matching but incur substantial index storage and MaxSim scoring overhead. Post-hoc merging offers a practical route to efficient VDR by reducing this cost without retraining the retriever, but its uniform reconstruction objectives are poorly aligned with the sparse, non-uniform patch usage induced by late-interaction retrieval. Under aggressive compression, this misalignment can preserve rarely used patches while concentrating retrieval activity on too few retained representatives. To address this misalignment, we propose Marginal-Guided Compression with Optimal Transport (MAGIC), a training-free post-hoc compressor for efficient retrieval with frozen multi-vector embeddings. MAGIC derives a MaxSim-induced compression surrogate and optimizes it through a two-marginal entropic optimal
    
[^9]: 特征叠加中线性能及性的高概率保证

    High-probability guarantees for linear accessibility in feature superposition

    [https://arxiv.org/abs/2609.09556](https://arxiv.org/abs/2609.09556)

    该论文将特征叠加中的线性能及性建模为压缩感知问题，证明了充分维度只需线性规模（而非此前最坏情况的二次方限制）即可高概率地恢复同时激活的特征，从而量化了线性表示假设的几何约束并为稀疏自编码器和神经可解释性评估提供了理论框架。

    

    神经网络可以利用特征叠加来编码比维度数量更多的概念，但特征间的交叉干扰限制了同时激活特征的线性能及性。通过将线性能及性建模为一个压缩感知问题，我们在次高斯噪声下针对固定支撑集推导出高概率界，证明了充分维度以线性方式扩展（d=O_ε(k log m)），而非此前最坏情况下的二次方限制。随后，我们通过高斯尾近似在各系统参数下验证了这些界。这些结果量化了线性表示假设的几何约束，为评估稀疏自编码器、组合泛化和神经网络可解释性提供了一个框架。

    arXiv:2609.09556v1 Announce Type: cross  Abstract: Neural networks can leverage feature superposition to encode more concepts than dimensions, but cross-feature interference constrains the linear accessibility of simultaneously active features. By framing linear accessibility as a compressed sensing problem, we derive high-probability bounds for fixed supports under subgaussian noise, proving the sufficient dimension scales linearly ($d=O_{\varepsilon}(k \log m)$) rather than prior worst-case quadratic limits. We then validate these bounds across system parameters through Gaussian-tail approximations. These results quantify the geometric constraints of the linear representation hypothesis, providing a framework for evaluating sparse autoencoders, compositional generalization, and neural interpretability.
    
[^10]: IntHQ：面向生成式推荐的双流表示上任务交互式分层查询

    IntHQ: Task-Interactive Hierarchical Query on Dual-Stream Representations for Generative Recommendation

    [https://arxiv.org/abs/2608.09634](https://arxiv.org/abs/2608.09634)

    提出IntHQ多任务生成式推荐框架，通过在双流表示上引入任务交互式分层查询机制，解决了现有多任务推荐系统中的源坍塌、关系坍塌和层级坍塌这三重固有问题。

    

    基于异构数据的多任务学习是现代推荐系统的基础，而生成式模型正逐步成为下一代推荐系统的骨干架构。然而，将多任务学习融入生成式范式的研究在很大程度上仍处于空白。现有的多任务推荐系统（无论判别式还是生成式范式）都是从单一的任务无关表示中提取任务相关特征，并将各任务固定接入预定义的转化漏斗中。我们证明这种方案天然存在三重坍塌问题：源坍塌——任务特定信号注入过晚并在共享潜在空间中被稀释；关系坍塌——任务间的依赖关系要么被骨干网络隐式吸收，要么被预定义漏斗静态固定；层级坍塌——各任务依赖不同尺度的特征，且在训练阶段间会发生偏移。我们提出IntHQ，一个多任务生成式推荐框架（摘要截断于此）。

    arXiv:2608.09634v2 Announce Type: replace  Abstract: Multi-task learning over heterogeneous data is fundamental to modern recommendation, while generative models are emerging as the backbone of next-generation recommenders. However, the integration of multi-task learning into the generative paradigm remains largely unexplored. Existing multi-task recommenders, in both discriminative and generative paradigms, extract task-relevant features from a single task-agnostic representation and wire tasks into a predefined conversion funnel. We show that this scheme is inherently prone to a threefold collapse. Source collapse, where task-specific signals are injected late and diluted in the shared latent space. Relational collapse, where task dependencies are either implicitly absorbed by the backbone or statically fixed by predefined funnels. Hierarchical collapse, where tasks depend on features at different scales and shift across training stages. We propose IntHQ, a multi-task generative reco
    
[^11]: ANNLib：一个用于高效近似最近邻搜索的开发框架

    ANNLib: A Development Framework for Efficient Approximate Nearest Neighbor Search

    [https://arxiv.org/abs/2607.17582](https://arxiv.org/abs/2607.17582)

    ANNLib通过将基于图的ANNS系统中的算法与数据结构组件解耦并独立优化，同时实现了高性能与灵活功能，支持过滤搜索、完全动态更新、快照历史查询和范围搜索等复杂场景。

    

    近似最近邻搜索（ANNS）在现代深度学习流程中扮演着至关重要的角色。近年来，许多ANNS系统被提出，以提供广泛而灵活的功能或实现高性能。然而，同时实现这两者在本质上是困难的。我们提出ANNLib来填补这一空白。ANNLib是一个库，它基于流行的基于图的ANNS算法，提供了一个编程框架，使ANNS系统能够兼具高性能和灵活的功能。我们精心地将ANNS系统中的算法组件和数据结构组件解耦，并对它们进行独立优化。此外，我们将最先进的算法和数据结构作为模块集成到ANNLib中，同时也集成了我们的新设计。用户可以选择组件的组合，以高性能支持复杂的设置，例如过滤搜索、完全动态更新、基于快照的历史查询以及范围搜索。我们的实验……

    arXiv:2607.17582v2 Announce Type: replace  Abstract: Approximate Nearest Neighbor Search (ANNS) plays a pivotal role in modern deep learning pipelines. Recently, many ANNS systems have been proposed to provide broad, flexible functionalities or achieve high performance. However, it is inherently difficult to achieve both. We propose ANNLib to address this gap. ANNLib is a library that provides a programming framework to achieve high performance and flexible functionalities for ANNS systems, based on popular graph-based ANNS algorithms. We carefully decouple and independently optimize both the algorithm and the data structure components in an ANNS system. In addition, we integrate state-of-the-art algorithms and data structures as modules in ANNLib, as well as our new designs. Users can choose combinations of components to support sophisticated settings with high performance, such as filtered search, fully dynamic updates, historical queries on snapshots, and range searches. Our experim
    
[^12]: 先召回后重排：面向大规模代码到代码检索的深度学习模型基准测试

    Recall Before Rerank: Benchmarking Deep Learning Models for Large-Scale Code-to-Code Retrieval

    [https://arxiv.org/abs/2606.27401](https://arxiv.org/abs/2606.27401)

    本文对大规模代码检索第一阶段召回中的深度学习模型进行了多语言、多数据集基准测试，揭示了其在TB级代码库上的精度与可扩展性局限，并提出基于LLM的代码规范化与查询重写方案显著提升检索精度。

    

    语义代码搜索和克隆检测对于软件开发、维护和复用至关重要。本文评估了当代深度学习模型在大规模代码到代码搜索引擎中第一阶段召回的有效性、效率和可扩展性。在多种编程语言和数据集上的基准测试揭示了这些模型在TB级源代码集合上精度和可扩展性的关键局限。我们提出了基于大语言模型（LLM）的代码规范化与查询重写方案，为表现较差的模型带来了显著的精度提升。我们的结果对资源受限部署的可持续性以及当前代码专用大语言模型在跨数据集上鲁棒性的假设提出了质疑。最后，我们为构建可扩展、高效的代码检索系统提供了可操作的见解。

    arXiv:2606.27401v2 Announce Type: replace-cross  Abstract: Semantic code search and clone detection are essential for software development, maintenance, and reuse. This paper evaluates the effectiveness, efficiency, and scalability of contemporary deep learning models for first-stage recall in large-scale code-to-code search engines. Benchmarking across multiple programming languages and datasets reveals critical limits in the precision and scalability of these models on Terabyte-scale source-code collections. We present LLM-based code normalisation and query-rewriting schemes that yield significant gains in precision for lower-performing models. Our results question the sustainability of resource-constrained deployment and the assumed robustness of current code-specialised LLMs across datasets. We conclude with actionable insights for building scalable, efficient code-retrieval systems.
    
[^13]: 面向算法设计的可迁移知识图谱与可执行习得算子

    Transferable knowledge graphs with executable learned operators for algorithm design

    [https://arxiv.org/abs/2603.27922](https://arxiv.org/abs/2603.27922)

    提出生成式可执行算法知识图谱（GEAKG），将算法设计中的过程性知识表示为可迁移的图谱，使习得算子与边权重能够跨领域、跨数据集复用，无需为每个新领域重新构建知识。

    

    arXiv:2603.27922v2 公告类型：替换 摘要：算法设计中的过程性知识嵌入于源代码中，并且需要为每个新领域重新构建。我们提出生成式可执行算法知识图谱（GEAKG），这是一种将此类知识存储为生成式、可执行、可迁移图谱的表示方法：类型化节点保存经过验证的算子，边编码可容许的组合方式，习得的边权重记录有效的序列。仅需更改角色本体和绑定，同一引擎即可在不同领域间实例化该结构。我们将GEAKG作为一种表示机制而非最先进的优化器进行研究，探讨什么能够迁移以及何时迁移。层级消融实验按粒度定位迁移：在神经架构搜索家族内，习得的快照可在70个数据集对之间迁移——其权重在各数据集间保持相关性，且一个冻结快照在零部署token成本下仍与正则化进化保持竞争力；跨组合……

    arXiv:2603.27922v2 Announce Type: replace  Abstract: Procedural knowledge in algorithm design is embedded in source code and rebuilt for each new domain. We introduce Generative Executable Algorithm Knowledge Graphs (GEAKG), a representation in which this knowledge is stored as a generative, executable, transferable graph: typed nodes hold validated operators, edges encode admissible compositions, and learned edge weights record effective sequences. The same engine instantiates the structure across domains by changing only a role ontology (RoleSchema) and a binding. We study GEAKG as a representation mechanism rather than a state-of-the-art optimizer, asking what transfers and when. Layer ablations localize transfer by granularity: within a neural-architecture-search family the learned snapshot transfers across 70 dataset pairs - its weights stay correlated across datasets and one frozen snapshot remains competitive with Regularized Evolution at zero deployment-token cost; across combi
    
[^14]: 超越最终答案：用于透明多模态推理评估的CRYSTAL基准

    Beyond Final Answers: CRYSTAL Benchmark for Transparent Multimodal Reasoning Evaluation

    [https://arxiv.org/abs/2603.13099](https://arxiv.org/abs/2603.13099)

    提出CRYSTAL诊断基准，通过可验证的中间步骤及Match F1与Ordered Match F1两个新指标评估多模态大模型的透明推理能力，并揭示出仅靠最终答案准确率无法发现的系统性推理缺陷。

    

    我们提出了CRYSTAL（通过产出步骤、可追溯性和逻辑实现清晰推理），这是一个包含6,372个实例的诊断基准，通过可验证的中间步骤来评估多模态推理。我们提出两个互补的指标：Match F1，通过语义相似度匹配对步骤级精确率和召回率进行评分；以及Ordered Match F1，进一步惩罚无序的推理链。参考答案通过受德尔菲法启发的流程构建：四个独立的多模态大语言模型生成推理轨迹，随后通过语义聚类进行聚合，并经过人工质量关卡验证。对20个多模态大语言模型的评估（包括未参与基准构建的商业前沿系统）揭示了仅凭答案准确率无法察觉的系统性失败：普遍存在的“挑选性回答”现象（精确率远超召回率）、非单调的规模扩展权衡，以及无序推理——没有任何一个有竞争力的模型能保留超过（原文此处截断）。

    arXiv:2603.13099v3 Announce Type: replace  Abstract: We introduce CRYSTAL (Clear Reasoning via Yielded Steps, Traceability, and Logic), a diagnostic benchmark with 6,372 instances that evaluates multimodal reasoning through verifiable intermediate steps. We propose two complementary metrics: Match F1, which scores step-level precision and recall via semantic similarity matching, and Ordered Match F1, which further penalizes disordered reasoning chains. References are constructed through a Delphi-inspired pipeline in which four independent MLLMs generate trajectories, which are then aggregated via semantic clustering and validated through human quality gates. Evaluation of 20 MLLMs, including commercial frontier systems not used during benchmark construction, reveals systematic failures that are invisible to answer accuracy: universal cherry-picking (precision far exceeds recall), non-monotonic scaling trade-offs, and disordered reasoning in which no competitive model preserves more tha
    
[^15]: IntTravel：面向综合多任务出行推荐的真实世界数据集与生成式框架

    IntTravel: A Real-World Dataset and Generative Framework for Integrated Multi-Task Travel Recommendation

    [https://arxiv.org/abs/2602.11664](https://arxiv.org/abs/2602.11664)

    该论文发布了首个基于高德地图大规模真实数据的综合出行推荐数据集IntTravel（含1.63亿用户、730万POI、41亿次交互），并提出了一个端到端的仅解码器生成式框架，实现覆盖出发时间、出行方式、目的地等多任务的一体化出行推荐。

    

    下一兴趣点（POI）推荐对于现代出行和基于位置的服务至关重要。为了提供流畅的用户体验，模型必须整体理解一段旅程的多个组成部分：“何时出发”、“如何出行”、“去往何处”以及“沿途会产生什么需求”。然而，当前研究受限于碎片化的数据集，这些数据集仅关注下一POI推荐（“去往何处”），忽略了出发时间、出行方式以及旅途中的情境需求。此外，这些数据集规模的有限性也阻碍了对模型性能的准确评估。为弥合这一差距，我们推出了IntTravel，这是首个从高德地图收集的面向综合出行推荐的大规模公开数据集，包含来自1.63亿用户与730万个POI的41亿次交互。基于该数据集，我们提出了一个端到端的、仅解码器（decoder-only）的多任务推荐生成式框架。

    arXiv:2602.11664v2 Announce Type: replace  Abstract: Next Point of Interest (POI) recommendation is essential for modern mobility and location-based services. To provide a smooth user experience, models must understand several components of a journey holistically: "when to depart", "how to travel", "where to go", and "what needs arise via the route". However, current research is limited by fragmented datasets that focus merely on next POI recommendation ("where to go"), neglecting the departure time, travel mode, and situational requirements along the journey. Furthermore, the limited scale of these datasets impedes accurate evaluation of performance. To bridge this gap, we introduce IntTravel, the first large-scale public dataset collected from Amap for integrated travel recommendation, including 4.1 billion interactions from 163 million users with 7.3 million POIs. Built upon this dataset, we introduce an end-to-end, decoder-only generative framework for multi-task recommendation. It
    
[^16]: RankSteer：逐点式大语言模型排序器能否在表示层面被校准？

    RankSteer: Can Pointwise LLM Rankers Be Calibrated at the Representation Level?

    [https://arxiv.org/abs/2602.03422](https://arxiv.org/abs/2602.03422)

    RankSteer提出了一种推理时的激活引导框架，通过沿决策、证据和角色等多个方向进行投影干预来校准逐点LLM排序器的“校准鸿沟”，在无需更新模型权重或跨文档比较的情况下显著提升排序性能。

    

    大语言模型是强大的零样本逐点排序器，但其性能落后于成对式和列表式方法。除了缺失比较信号之外，我们还识别出一个“校准鸿沟”：隐藏状态中编码的与排序相关的信息未能被标量输出头完全捕获。我们提出了RankSteer，这是一种事后激活引导框架，在推理时通过基于投影的干预沿多个方向（决策、证据，以及可选的角色）来校准排序。该方法无需更新模型权重，也无需引入跨文档比较。我们在两个结构不同的逐点排序变体上实例化了RankSteer，并在三个骨干模型上的大多数TREC DL和BEIR数据集上观察到相对于各自基线的改进。这表明校准鸿沟是逐点排序器的一种普遍特性。我们额外的几何分析表明，引导通过集中（原文此处截断）……

    arXiv:2602.03422v2 Announce Type: replace  Abstract: Large language models (LLMs) are strong zero-shot pointwise rankers, but lag behind pairwise and listwise methods. Beyond missing comparative signals, we identify a \textit{calibration gap}: ranking-relevant information encoded in hidden states is not fully captured by the scalar output head. We propose RankSteer, a post-hoc activation-steering framework that calibrates ranking via projection-based interventions along multiple directions at inference time: decision, evidence, and, optionally, role. This is achieved without updating model weights or introducing cross-document comparisons. We instantiate RankSteer on two structurally distinct pointwise variants and observe improvements over their respective baselines on most TREC DL and BEIR datasets across three backbones. This suggests that the calibration gap is a general property of pointwise rankers. Our additional geometric analysis shows that steering improves ranking by concent
    
[^17]: Compass：跨向量数据与结构化数据的通用过滤搜索

    Compass: General Filtered Search across Vector and Structured Data

    [https://arxiv.org/abs/2510.27141](https://arxiv.org/abs/2510.27141)

    Compass 提出了一个统一框架，复用现有索引结构（HNSW、IVF、B+树）并通过协作式查询执行策略，在不设计新索引的情况下实现了支持任意合取、析取和范围谓词的向量与结构化数据通用过滤搜索。

    

    混合向量与关系型数据的日益普及，要求为结合高维向量搜索与复杂关系过滤的查询提供高效、通用的支持。然而，现有的过滤搜索方案从根本上受限于专用索引，这些索引限制了对任意过滤条件的支持，并阻碍了与通用数据库管理系统（DBMS）的集成。本工作提出了 Compass，一个统一框架，能够在不依赖新索引设计的情况下，实现对向量数据与结构化数据的通用过滤搜索。Compass 利用已有的成熟索引结构——例如针对向量属性的 HNSW 和 IVF，以及针对关系属性的 B+ 树——实现了一种有原则的协作式查询执行策略，在跨模态之间协调候选生成与谓词求值。独特之处在于，Compass 通过支持任意的合取、析取和范围谓词来保持通用性（摘要原文在此处被截断）。

    arXiv:2510.27141v5 Announce Type: replace-cross  Abstract: The increasing prevalence of hybrid vector and relational data necessitates efficient, general support for queries that combine high-dimensional vector search with complex relational filtering. However, existing filtered search solutions are fundamentally limited by specialized indices, which restrict arbitrary filtering and hinder integration with general-purpose DBMSs. This work introduces \textsc{Compass}, a unified framework that enables general filtered search across vector and structured data without relying on new index designs. Compass leverages established index structures -- such as HNSW and IVF for vector attributes, and B+-trees for relational attributes -- implementing a principled cooperative query execution strategy that coordinates candidate generation and predicate evaluation across modalities. Uniquely, Compass maintains generality by allowing arbitrary conjunctions, disjunctions, and range predicates, while e
    

