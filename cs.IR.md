# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Can Programs Replace LLM Judges? Executable Program Judges for Scalable and Efficient Relevance Labeling](https://arxiv.org/abs/2611.24849) | 提出ExecJudges方法，利用前沿LLM将评估指南转化为仅依赖词汇匹配和标准NLP库的可执行程序，从而取代昂贵的LLM裁判调用，实现快速、可解释、可复现的大规模相关性标注。 |
| [^2] | [Variance-Optimal Counterfactual Policy Design for Sample-Efficient Online A/B Testing](https://arxiv.org/abs/2611.24757) | 该论文提出一种方差最优的反事实策略设计方法，通过联合优化动作的采样方式，将流量引导至两种策略不一致的动作上，从而显著降低在线A/B测试的方差并提升样本效率。 |
| [^3] | [Task-Oriented Network Ranging via Joint Graph and Schedule Design](https://arxiv.org/abs/2611.24702) | 本文提出面向任务的稀疏信号复用网络测距框架S-SM-NR，通过联合设计稀疏任务图与均衡的发射-监听调度，以最小轮数完成网络定位所需的测距任务，从而显著缩短测距周期。 |
| [^4] | [X-split Argus: Learning Reusable History Tokens for Efficient CTR Prediction](https://arxiv.org/abs/2611.24283) | 提出X-split ARGUS端到端训练框架，将用户行为历史离线压缩为可复用令牌并结合教师蒸馏，在大幅降低在线计算与延迟成本的同时实现准确的CTR预测。 |
| [^5] | [MARS Technical Report](https://arxiv.org/abs/2611.24006) | MARS提出了一种基于微调Qwen3-0.6B的多阶段智能体推荐系统，通过近线智能体为每个用户动态生成七维流水线配置（涵盖召回配额、模型选择、融合权重等），将个性化从单阶段预测扩展到整条推荐流水线的编排，并以异步缓存机制确保级联流程不被LLM推理阻塞。 |
| [^6] | [ReflectRec: Cycle-Consistency Preference Verification for LLM-based Recommendation](https://arxiv.org/abs/2611.23969) | ReflectRec通过偏好-历史往返验证器对LLM生成的偏好笔记进行循环一致性校验，结合低质量偏好的条件性再生成与行为相似用户未来行为的前瞻检索，在MovieLens数据集上显著提升推荐性能。 |
| [^7] | [CONCORD-Rec: Catalog-Conditioned Modality Utility for Semantic-ID Diffusion Recommendation](https://arxiv.org/abs/2611.23710) | CONCORD-Rec提出了一种由反事实目录划分效用监督的可靠性条件化离散扩散推荐框架，能够区分真正降低目录级歧义的模态，从而避免缺失或冲突模态的误导性信息在语义ID去噪过程中传播。 |
| [^8] | [KnowMe Paper Radar: Lightweight Discovery of Promising Papers in the Age of Autoresearch](https://arxiv.org/abs/2611.23702) | KnowMe Paper Radar 是一个无需 LLM 推理的轻量级论文发现工具，它综合主题相关性、时效性、作者研究历史、合作路径和机构背景等多源证据生成个性化阅读优先级，帮助研究者快速筛选潜力论文，并将重复上下文准备速度提升三倍以上。 |
| [^9] | [VIBRA: Backward-Compatible Cross-Modal Retrieval with Consensus Barycenters](https://arxiv.org/abs/2611.23606) | VIBRA利用多个视图的共识重心来纠正单个模型的错误，并训练新编码器继承这一共识，实现了无需重新编码数十亿条目即可升级跨模态检索系统的向后兼容方案，在五个基准测试上取得最佳性能。 |
| [^10] | [From Docket Minutes to Case Trajectories: Building Analysis-Ready Criminal Court Data](https://arxiv.org/abs/2611.23537) | 本文提出一个面向新奥尔良刑事法院数据的数据摄取、协调与可视化仪表板系统，通过生成案件轨迹帮助专家和普通民众分析法院趋势、调查政策影响并实现公众监督。 |
| [^11] | [LusoClin: A Synthetic Clinical Note Dataset for Medical Information Retrieval in European Portuguese](https://arxiv.org/abs/2611.23454) | 该研究发布了LusoClin——首个公开的欧洲葡萄牙语合成临床记录数据集，通过QLoRA微调Llama 3.2生成并经BERT模型标注临床实体，为低资源语言下的隐私保护医学信息检索研究提供了宝贵资源。 |
| [^12] | [Prioritize, Don't Decide: What We Learned Auditing a Deployed Human-in-the-Loop CV Sorter](https://arxiv.org/abs/2611.23359) | 本文审计了一个部署在国家学术招聘平台上的人在环简历优先排序系统——委员会按其推荐顺序只需阅读一半简历即可覆盖 95% 的相关候选人；审计进一步发现该评测简历池仅凭文档长度即可排序，使精确率类指标失去参考意义，这一发现直接改变了产品设计。 |
| [^13] | [PGPO: Prototype-Guided Progressive Obfuscation for Privacy-Preserving LLM-Enhanced Recommendation](https://arxiv.org/abs/2611.23236) | 该论文提出PGPO方法，通过为高影响力原型词学习拓扑感知的混淆锚点并在静态语义图中渐进传播，在有效防御嵌入反演攻击保护隐私的同时保持物品间语义关系，实现接近上限的推荐性能。 |
| [^14] | [Explanations that Transfer: Interpreting Medical Retrieval with Sparse AutoEncoders](https://arxiv.org/abs/2611.18179) | 本文提出将稀疏自编码器应用于医学信息检索，通过结合重建目标与InfoNCE对比损失，将不透明的稠密文本嵌入转化为语义可解释的表示，同时保持对新临床语料库的可迁移性。 |
| [^15] | [Toward Agentic Retrieval-Augmented Generation for Enterprise Retail Workflows over Heterogeneous Data Sources](https://arxiv.org/abs/2610.25096) | 本文提出了RetailBridge零售基准和RetailRAG框架，通过自动模式简报、数据源路由计划串联和诊断性计划修订等机制，使智能体能够跨异构数据源识别、串联证据并生成有依据的答案，从而完成企业零售工作流任务，其性能在所有评估维度上全面超越现有代理式RAG基线。 |
| [^16] | [Recommendation Quality and the Concentration of Consumption: Experimental Evidence from Netflix](https://arxiv.org/abs/2608.21274) | 该研究表明，改进推荐算法可提升总消费并分散消费从热门到中等热门产品，而非极化消费，从而支持对中尾产品的投资。 |
| [^17] | [UniRank: Benchmarking Ranking Models for Unified Sequential Modeling and Feature Interaction](https://arxiv.org/abs/2607.19987) | UniRank提出了一个统一序列建模与特征交互的开放排序模型基准，通过标准化的评估流程和降低硬件门槛的高效PyTorch工具包，首次实现了15个代表性统一排序模型在公开数据集上的可复现比较。 |
| [^18] | [Know Before You Fetch: When Does Confidence-Based Retrieval-Budget Allocation Pay Off?](https://arxiv.org/abs/2606.29959) | 该研究在相同的每查询平均检索预算上限下，将基于阅读器置信度的自适应检索分配与最佳简单策略（固定深度或其随机混合）对比，发现置信度分配在极紧预算（每查询平均半个段落）下于全部七个设置中稳定胜出（准确率提升0.010–0.075），其价值在于配给稀缺预算和避免有害检索。 |
| [^19] | [LLM-Assisted Reranking to Operationalize Nuanced Objectives in Recommender Systems](https://arxiv.org/abs/2606.02883) | 本文基于真实YouTube轨迹实证检验了大语言模型辅助的重排序在提升个性化推荐的同时，是否会放大用户接触意识形态同质化、极端或阴谋论政治内容的风险。 |
| [^20] | [Learning Binary Codes with Spiking Neural Networks for Cross-Modal Hashing Retrieval](https://arxiv.org/abs/2606.00740) | 该论文提出了一种基于脉冲神经网络的无监督跨模态哈希框架，通过正负脉冲随时间积累的竞争来确定哈希比特，并让图像与文本通过时间交互共同生成哈希编码。 |
| [^21] | [PEARL: Personalized Streaming Video Understanding with Runtime-Defined Concepts](https://arxiv.org/abs/2603.20422) | 该论文提出个性化流式视频理解（PSVU）这一新任务及首个基准 PSVU-Bench，并给出即插即用基线 PEARL，使模型能够在流式视频播放过程中实时注册用户定义的新概念并回溯其历史出现时刻。 |

# 详细

[^1]: 程序能否取代大语言模型裁判？用于可扩展且高效相关性标注的可执行程序裁判

    Can Programs Replace LLM Judges? Executable Program Judges for Scalable and Efficient Relevance Labeling

    [https://arxiv.org/abs/2611.24849](https://arxiv.org/abs/2611.24849)

    提出ExecJudges方法，利用前沿LLM将评估指南转化为仅依赖词汇匹配和标准NLP库的可执行程序，从而取代昂贵的LLM裁判调用，实现快速、可解释、可复现的大规模相关性标注。

    

    大语言模型（LLM）裁判已成为为检索评估创建大规模相关性判断的常用方法。然而，每次判断都需要对前沿模型进行可能代价高昂的调用，且这些判断难以解释，并可能无法复现。在本文中，我们探讨LLM能否为每个主题编写一个可读的程序，以取代其对每个文档逐一进行判断的方式。为解决这些问题，我们提出了ExecJudges，一种用于自动相关性标注的方法，可生成快速、可检查且可扩展的裁判。具体而言，ExecJudges使用前沿LLM将评估指南转化为可执行程序，这些程序仅依赖词汇匹配、标准NLP库（如解析器）以及池统计信息。ExecJudges-L变体在可用时还会利用人工标注的示例。我们在七个TREC数据集合上评估了ExecJudges，通过将生成的标签与训练有素的NIST评估员、前沿LLM裁判以及（原文在此处截断）进行比较。

    arXiv:2611.24849v1 Announce Type: cross  Abstract: Large language model (LLM) judges have become a common way to create large-scale relevance judgments for retrieval evaluation. However, each judgment requires a potentially expensive call to a frontier model, and the judgments are hard to interpret and may not be reproducible. In this paper, we ask whether an LLM can write a readable program for each topic that replaces its per-document judgments. To address these issues, we propose ExecJudges, a method for generating fast, inspectable and scalable judges for automatic relevance labeling. Specifically, ExecJudges uses a frontier LLM to turn assessment guidelines into executable programs, which rely only on lexical matching, standard NLP libraries like parsers, and pool statistics. An ExecJudges-L variant also uses examples of human labels if available. We evaluate ExecJudges on seven TREC collections by comparing the produced labels with trained NIST assessors, frontier LLM judges, and
    
[^2]: 面向样本高效在线A/B测试的方差最优反事实策略设计

    Variance-Optimal Counterfactual Policy Design for Sample-Efficient Online A/B Testing

    [https://arxiv.org/abs/2611.24757](https://arxiv.org/abs/2611.24757)

    该论文提出一种方差最优的反事实策略设计方法，通过联合优化动作的采样方式，将流量引导至两种策略不一致的动作上，从而显著降低在线A/B测试的方差并提升样本效率。

    

    在线A/B测试是在部署前评估面向用户系统变更（例如新的推荐或排序策略）的标准方法。虽然标准A/B测试的均值差估计量是无偏的，但它通常存在高方差问题。当处理策略与生产策略以相近的概率选择相同的动作时，由此产生的观测数据会增加噪声，却几乎不提供关于两种策略差异的信息。因此，A/B测试往往需要大量流量和较长周期才能检测到较小的处理效应。现有的方差缩减方法主要是在数据收集方案固定之后改进估计量，而近年来的重叠感知设计仅优化两种策略之间的单一流量分配，因而无法将流量引导到两种策略不一致的动作上。在本文中，我们联合优化动作的采样方式……

    arXiv:2611.24757v1 Announce Type: new  Abstract: Online A/B testing is the standard approach for evaluating changes to user-facing systems, such as a new recommendation or ranking policy, before deployment. While the difference-in-means estimator of a standard A/B test is unbiased, it often suffers from high variance. When the treatment and production policies choose the same actions with similar probabilities, the resulting observations can add noise while providing little information about the difference between the policies. Consequently, A/B tests can require large amounts of traffic and long durations to detect small treatment effects. Existing variance-reduction methods mainly improve the estimator after the data-collection scheme has been fixed, and recent overlap-aware designs only optimize a single traffic split between the two policies; thereby, they cannot direct traffic to the actions on which the policies disagree.   In this paper, we jointly optimize how actions are sampl
    
[^3]: 基于图与调度联合设计的面向任务网络测距

    Task-Oriented Network Ranging via Joint Graph and Schedule Design

    [https://arxiv.org/abs/2611.24702](https://arxiv.org/abs/2611.24702)

    本文提出面向任务的稀疏信号复用网络测距框架S-SM-NR，通过联合设计稀疏任务图与均衡的发射-监听调度，以最小轮数完成网络定位所需的测距任务，从而显著缩短测距周期。

    

    测量节点间距离是网络定位的基础，而获取节点位置并不需要完全连通的距离图。现有的用于网络定位的全测距协议要求每个节点对都交换信号，这延长了测距周期。本文提出了稀疏信号复用网络测距，这是一种面向任务的框架，可根据网络定位所需的测距任务自适应调整发射-监听调度。首先，我们通过联合设计任务图与发射-监听调度建立了S-SM-NR框架。然后，我们分析了S-SM-NR所需的最小轮数，并建立了网络定位的连通性条件。随后，我们开发了一种闭式联合设计，用以构建稀疏任务图和均衡的发射-监听调度。特别地，该联合设计能够实现最小轮数……

    arXiv:2611.24702v1 Announce Type: cross  Abstract: Measuring inter-node distances is the basis for network localization, whereas acquiring node positions does not require a fully connected graph of distances. Existing complete-ranging protocols for network localization require every node pair to exchange signals, which prolongs the ranging cycle. In this paper, we propose sparse signal-multiplexing network ranging (S-SM-NR), a task-oriented framework that adapts the transmit-listen schedule to the ranging task required by network localization. First, we establish the S-SM-NR framework by formulating the joint design of the task graph and the transmit-listen schedule. Then, we analyze the minimum round count required by S-SM-NR and establish a connectivity condition for network localization. Subsequently, we develop a closed-form joint design that constructs the sparse task graph and the balanced transmit-listen schedule. In particular, the joint design can achieve the minimum round cou
    
[^4]: X-split Argus：学习可复用历史令牌以实现高效CTR预测

    X-split Argus: Learning Reusable History Tokens for Efficient CTR Prediction

    [https://arxiv.org/abs/2611.24283](https://arxiv.org/abs/2611.24283)

    提出X-split ARGUS端到端训练框架，将用户行为历史离线压缩为可复用令牌并结合教师蒸馏，在大幅降低在线计算与延迟成本的同时实现准确的CTR预测。

    

    准确的点击率（CTR）预测依赖于能够同时捕捉用户持久偏好和近期意图的用户表示，然而在生产环境的计算和延迟约束下，为每次排序请求处理庞大的行为历史序列代价高昂。我们提出X-split ARGUS，一个用于广告排序的框架，其历史压缩器和在线排序器进行端到端联合训练。压缩器将过往交互编码为一组紧凑的令牌，这些令牌离线计算并在各候选广告间复用；而在线排序器通过对这些令牌和近期事件表示进行交叉注意力来为广告打分。训练过程结合了针对点击反馈的自回归损失，以及从一个更大的教师模型对候选特定点击logits的蒸馏，该教师模型使用未压缩的历史对每个候选进行打分。压缩历史令牌的数量和近期历史的长度均可调整，以适应线上服务的计算和延迟预算。

    arXiv:2611.24283v1 Announce Type: new  Abstract: Accurate click-through rate (CTR) prediction depends on user representations that capture both persistent preferences and recent intent, yet processing extensive behavioral histories for every ranking request is costly under production compute and latency constraints. We present X-split ARGUS, a framework for ad ranking whose history compressor and online ranker are trained end to end. The compressor encodes past interactions into a compact set of tokens that are computed offline and reused across candidates, while the online ranker scores ads through cross-attention to these tokens and representations of recent events. Training combines an autoregressive loss for click feedback with distillation of candidate-specific click logits from a larger teacher that scores each candidate using the uncompressed history. The number of compressed history tokens and the length of recent history can be adjusted to fit serving compute and latency budge
    
[^5]: MARS技术报告

    MARS Technical Report

    [https://arxiv.org/abs/2611.24006](https://arxiv.org/abs/2611.24006)

    MARS提出了一种基于微调Qwen3-0.6B的多阶段智能体推荐系统，通过近线智能体为每个用户动态生成七维流水线配置（涵盖召回配额、模型选择、融合权重等），将个性化从单阶段预测扩展到整条推荐流水线的编排，并以异步缓存机制确保级联流程不被LLM推理阻塞。

    

    现代工业推荐系统采用多阶段级联架构——召回、粗排、精排和重排——将数百万候选缩小为个性化列表。尽管每个阶段都部署了复杂的模型，但在每个场景内，流水线级别的决策（如通道激活、配额、模型选择和分数融合）仍依赖工程师调优，且不针对用户个性化。本报告提出MARS（多阶段智能体推荐系统），将个性化从单阶段预测扩展到整个流水线的编排。基于微调的Qwen3-0.6B构建的近线智能体，从受限的动作空间中为每个用户生成配置——一个涵盖流水线模式、召回配额、模型选择、融合权重和重排器设置的七维动作。配置经过验证、缓存并异步消费，级联流程从不等待LLM推理。模式校验、回退机制和终止开关共同确保了服务的安全性。

    arXiv:2611.24006v1 Announce Type: new  Abstract: Modern industrial recommender systems use multi-stage cascades--retrieval, pre-ranking, ranking, and re-ranking--to narrow millions of candidates to a personalized list. While each stage deploys sophisticated models, pipeline-level decisions on channel activation, quotas, model selection, and score fusion remain engineer-tuned and user-agnostic within each scenario.   This report presents MARS (Multi-stage Agentic Recommendation System), extending personalization from individual-stage predictions to full pipeline orchestration. A near-line agent built on fine-tuned Qwen3-0.6B generates per-user configurations--a seven-dimensional action covering pipeline mode, retrieval quotas, model selection, fusion weights, and reranker settings--from a constrained action space. Configurations are validated, cached, and consumed asynchronously; the cascade never waits for LLM inference. Schema checks, fallbacks, and a kill switch ensure serving safety
    
[^6]: ReflectRec：面向基于大语言模型推荐的循环一致性偏好验证

    ReflectRec: Cycle-Consistency Preference Verification for LLM-based Recommendation

    [https://arxiv.org/abs/2611.23969](https://arxiv.org/abs/2611.23969)

    ReflectRec通过偏好-历史往返验证器对LLM生成的偏好笔记进行循环一致性校验，结合低质量偏好的条件性再生成与行为相似用户未来行为的前瞻检索，在MovieLens数据集上显著提升推荐性能。

    

    思维链（CoT）风格的大语言模型推荐器会生成一段自然语言偏好笔记，作为排序前的思维链前缀，但该笔记在未经验证的情况下就被直接信任，且排序器从不参考行为相似的用户。我们提出了ReflectRec，其核心是一个“偏好到历史”的往返验证器，该验证器的召回分数可作为偏好可恢复性的操作性信号，从而支持对低可恢复性偏好进行条件性再生成，并在多次再生成尝试中做出选择。围绕该验证器，我们构建了语义化的第一阶段提示、条件性再生成机制，以及“前瞻检索”，后者将行为相似用户的未来观看记录作为协同信号注入。在MovieLens-100K和MovieLens-1M数据集上，完整流水线在Hit@10指标上分别以+16.25和+4.46个百分点的优势超越CoT4Rec-ICL，且在所有ValScore分桶以及三种大语言模型骨干（Qwen3-30B、Qwen2.5-72B和Llama-3.3-70B）上均取得正向提升。我们还观察到类似的验证……

    arXiv:2611.23969v1 Announce Type: new  Abstract: CoT-style LLM recommenders generate a natural-language preference note as a chain-of-thought prefix to ranking, but the note is trusted without verification, and the ranker never consults behaviorally similar users. We propose ReflectRec, organized around a preference-to-history round-trip verifier whose recall score serves as an operational signal of preference-to-history recoverability, enabling conditional regeneration for low-recoverability preferences and selection among regeneration attempts. Around this verifier, we wire a semantic Stage-1 prompt, conditional regeneration, and Forward-Look retrieval that injects future watches of behaviorally similar users as a collaborative hint. On MovieLens-100K and MovieLens-1M, the full pipeline outperforms CoT4Rec-ICL by +16.25 / +4.46 pp Hit@10, with positive gains across all ValScore buckets and three LLM backbones (Qwen3-30B, Qwen2.5-72B, and Llama-3.3-70B). We also observe a similar veri
    
[^7]: CONCORD-Rec：面向语义ID扩散推荐的目录条件化模态效用方法

    CONCORD-Rec: Catalog-Conditioned Modality Utility for Semantic-ID Diffusion Recommendation

    [https://arxiv.org/abs/2611.23710](https://arxiv.org/abs/2611.23710)

    CONCORD-Rec提出了一种由反事实目录划分效用监督的可靠性条件化离散扩散推荐框架，能够区分真正降低目录级歧义的模态，从而避免缺失或冲突模态的误导性信息在语义ID去噪过程中传播。

    

    基于离散语义标识符的多模态生成式推荐需要区分那些仅仅改变token预测的历史模态与真正降低目录级歧义的模态。当视听、视觉或文本流缺失、退化或语义冲突时，均匀融合和无约束的token门控会在迭代去噪步骤中传播误导性内容。我们提出CONCORD-Rec，一种由反事实目录划分效用监督的可靠性条件化离散扩散推荐器。给定固定的分词器和部分掩码的SID状态，CONCORD-Rec在精确的可行目录支持上评估折外教师联盟，并在每个未解析位置上跨目标代码划分聚合物品后验分布。该方法并非直接蒸馏原始token概率，而是利用由此产生的反事实损失差异来监督可用性掩码的token门控（摘要在此处截断）。

    arXiv:2611.23710v1 Announce Type: new  Abstract: Multimodal generative recommendation with discrete semantic identifiers (SIDs) requires distinguishing historical modalities that merely alter token predictions from those that genuinely reduce catalog-level ambiguity. When audiovisual, visual, or textual streams are missing, degraded, or semantically conflicting, uniform fusion and unconstrained token gates can propagate misleading content across iterative denoising steps. We propose CONCORD-Rec, a reliability-conditioned discrete diffusion recommender supervised by counterfactual catalog-partition utility. Given a fixed tokenizer and a partially masked SID state, CONCORD-Rec evaluates out-of-fold teacher coalitions over the exact feasible catalog support and aggregates item posteriors across the target code partition at each unresolved position. Rather than distilling raw token probabilities directly, the resulting counterfactual loss differences supervise an availability-masked tokenw
    
[^8]: KnowMe 论文雷达：自动科研时代的轻量级潜力论文发现工具

    KnowMe Paper Radar: Lightweight Discovery of Promising Papers in the Age of Autoresearch

    [https://arxiv.org/abs/2611.23702](https://arxiv.org/abs/2611.23702)

    KnowMe Paper Radar 是一个无需 LLM 推理的轻量级论文发现工具，它综合主题相关性、时效性、作者研究历史、合作路径和机构背景等多源证据生成个性化阅读优先级，帮助研究者快速筛选潜力论文，并将重复上下文准备速度提升三倍以上。

    

    主题相似的论文数量不断增长，使得人们难以从中识别出能够提供有用研究见解或方法指导的新工作。我们提出了 KnowMe Paper Radar，这是一个轻量级的论文发现工具，它将多种证据来源综合成个性化的阅读优先级。用户可以指定主题、优先级、排除项和日期；系统会综合考量主题相关性、时效性、作者研究历史、合作路径以及特定研究方向上的机构背景。通过公开每个评分组成部分及其支持来源，KnowMe 帮助读者为其当前的研究问题挑选有前景的论文，并发现熟悉作者圈之外的相关工作。自动获取、版本合并和图缓存重用使这一工作流程保持便捷，在常规使用中无需大语言模型（LLM）推理。在 174,640 条获取记录上的实验显示，重复上下文准备的速度提升了三倍以上，同时……

    arXiv:2611.23702v1 Announce Type: new  Abstract: The growing volume of topically similar papers makes it difficult to identify new work that can offer useful research insights or methodological guidance. We present KnowMe Paper Radar, a lightweight discovery tool that combines multiple sources of evidence into personalized reading priorities. Users specify topics, priorities, exclusions and dates; the system brings together topical relevance, recency, author research histories, collaboration paths and direction-specific institutional context. By exposing each score component and its supporting sources, KnowMe helps readers select promising papers for their current questions and discover relevant work beyond familiar authors. Automatic acquisition, version merging and graph-cache reuse keep this workflow convenient without large language model (LLM) inference during ordinary use. Experiments on 174,640 acquired records show more than threefold faster repeated context preparation, while 
    
[^9]: VIBRA：基于共识重心的向后兼容跨模态检索

    VIBRA: Backward-Compatible Cross-Modal Retrieval with Consensus Barycenters

    [https://arxiv.org/abs/2611.23606](https://arxiv.org/abs/2611.23606)

    VIBRA利用多个视图的共识重心来纠正单个模型的错误，并训练新编码器继承这一共识，实现了无需重新编码数十亿条目即可升级跨模态检索系统的向后兼容方案，在五个基准测试上取得最佳性能。

    

    向后兼容检索使新编码器能够查询旧模型已编码的图库，从而无需对数十亿条目重新编码即可升级跨模态检索系统。现有的兼容方法要么拟合旧的查询而停留在旧模型的精度附近，要么对每个查询同时运行旧编码器和新编码器。我们提出VIBRA，它在旧图库上匹配相关性，用多个视图的重心来纠正单个模型的错误——该重心本身也是旧空间中的一个查询——并训练新编码器继承这一共识，使得部署时并不存在的模型仍然塑造着每个查询。我们证明了当各视图最困难的不相关条目互不相同时，重心恰好能比其视图的平均值获得优势，并刻画了单个编码器所能达到的这一增益的份额。在六个CLIP、MetaCLIP和SigLIP2骨干网络之间的五次升级所涉及的五个基准测试上，VIBRA表现最佳。

    arXiv:2611.23606v1 Announce Type: cross  Abstract: Backward-compatible retrieval enables a new encoder to query the gallery that an old model has already encoded, so that a cross-modal retrieval system can be upgraded without re-encoding billions of items. Existing compatible methods either fit the old query and stay near the accuracy of the old model, or run the old and the new encoder for every query. We propose VIBRA, which matches relevance on the old gallery, corrects the errors of a single model with the barycenter of several views, which is itself a query in the old space, and trains the new encoder to inherit this consensus, so that models absent at deployment still shape every query. We show that the barycenter gains margin over the average of its views exactly when their hardest irrelevant items differ, and we characterize the share of this gain that one encoder can reach. On five benchmarks with five upgrades among six CLIP, MetaCLIP and SigLIP2 backbones, VIBRA is the best 
    
[^10]: 从庭审笔录到案件轨迹：构建可分析就绪的刑事法院数据

    From Docket Minutes to Case Trajectories: Building Analysis-Ready Criminal Court Data

    [https://arxiv.org/abs/2611.23537](https://arxiv.org/abs/2611.23537)

    本文提出一个面向新奥尔良刑事法院数据的数据摄取、协调与可视化仪表板系统，通过生成案件轨迹帮助专家和普通民众分析法院趋势、调查政策影响并实现公众监督。

    

    刑事法院记录往往零碎、嘈杂且相互孤立，这使得法律专家和社区利益相关者都难以访问和分析法院数据以识别趋势并开展公众监督。在本文中，我们介绍了一个面向新奥尔良刑事法院庭审数据的数据摄取、协调与仪表板可视化系统，旨在同时服务专业人士和普通大众。我们的系统收集并协调分散的数据源，生成案件轨迹，并基于社区反馈提供直观的可视化技术，用于分析人口统计特征、时间趋势和具体案件事件。我们通过对新奥尔良近期短期趋势和政策影响的探索性分析来展示该系统，说明其如何支持对异常情况、政策变化和程序违规问题的调查。我们还展示了来自社区利益相关者小型形成性研究的反馈。

    arXiv:2611.23537v1 Announce Type: new  Abstract: Criminal court records are often fragmented, noisy, and siloed, making it challenging for both legal experts and community stakeholders to access and analyze court data to identify trends and provide public oversight. In this paper, we introduce a system for data ingestion, harmonization, and dashboard visualization designed for New Orleans criminal court proceedings data, aimed at both expert and lay audiences. Our system collects and harmonizes disparate data, generates case trajectories, and provides intuitive visualization techniques based on community feedback to analyze demographics, temporal trends, and specific case events. We demonstrate the system through an exploratory analysis of recent short-term trends and policy impacts within New Orleans, illustrating how it can support the investigation of anomalies, policy changes, and procedural irregularities. We also present feedback from a small formative study of community stakehol
    
[^11]: LusoClin：面向欧洲葡萄牙语医学信息检索的合成临床记录数据集

    LusoClin: A Synthetic Clinical Note Dataset for Medical Information Retrieval in European Portuguese

    [https://arxiv.org/abs/2611.23454](https://arxiv.org/abs/2611.23454)

    该研究发布了LusoClin——首个公开的欧洲葡萄牙语合成临床记录数据集，通过QLoRA微调Llama 3.2生成并经BERT模型标注临床实体，为低资源语言下的隐私保护医学信息检索研究提供了宝贵资源。

    

    获取大规模临床文本数据集对于推进医学信息检索研究至关重要，包括临床记录搜索、患者队列识别以及领域适配的检索模型。然而，受隐私法规限制，真实临床记录很少能够公开获取，这对欧洲葡萄牙语等低资源语言的研究进展造成了特别的限制。我们提出了LusoClin，一个公开发布的欧洲葡萄牙语合成临床记录数据集，旨在支持隐私保护的临床文本检索研究。LusoClin的构建流程包括：对真实临床数据进行预处理，使用量化低秩适应（QLoRA）方法微调开源的Llama 3.2（3B）指令模型，并应用基于提示的临床记录生成技术。为了便于语义索引和临床搜索，这些记录通过基于BERT的模型自动标注了临床实体。我们对LusoClin在实用性方面进行了评估（原文在此处截断）。

    arXiv:2611.23454v1 Announce Type: new  Abstract: Access to large-scale clinical text collections is crucial for advancing medical information retrieval research, including clinical note search, patient cohort identification, and domain-adapted retrieval models. However, real clinical notes are rarely available due to privacy regulations, which particularly limits progress for low-resource languages such as European Portuguese. We present LusoClin, a publicly released dataset of synthetic clinical notes in European Portuguese, created to support privacy-preserving research on clinical text retrieval. LusoClin was generated by pre-processing real clinical data, fine-tuning the open-source Llama 3.2 (3B) instruction-tuned model using Quantized Low-Rank Adaptation, and applying prompt-based clinical note generation. To facilitate semantic indexing and clinical search, the notes are automatically annotated with clinical entities using a BERT-based model. We evaluate LusoClin in terms of uti
    
[^12]: 优先排序，而非决策：审计一个已部署的人在环简历排序系统的收获

    Prioritize, Don't Decide: What We Learned Auditing a Deployed Human-in-the-Loop CV Sorter

    [https://arxiv.org/abs/2611.23359](https://arxiv.org/abs/2611.23359)

    本文审计了一个部署在国家学术招聘平台上的人在环简历优先排序系统——委员会按其推荐顺序只需阅读一半简历即可覆盖 95% 的相关候选人；审计进一步发现该评测简历池仅凭文档长度即可排序，使精确率类指标失去参考意义，这一发现直接改变了产品设计。

    

    一份学术职位空缺可以吸引数百份申请，而第一轮筛选会占用评审委员会的大部分时间。我们描述了一个部署在国家学术职位空缺平台上、并与该平台职位空缺数据库集成的简历优先级排序器。它为一堆简历提出阅读顺序，而将所有决策留给委员会；该系统将冻结的多语言 Transformer 编码器与一个会话级线性模型相结合，该模型在每次标注后重新训练，并在会话结束时被丢弃。它是有效的：在一个包含 300 份合成简历的池子中，按该工具推荐的顺序阅读的委员会在读完一半简历后即可接触到 95% 的相关候选人，而按任意顺序阅读则需要几乎读完全部简历。本文关注的是我们审计上述声明时的发现，而非系统本身。三个发现改变了产品。其中一个发现是：该简历池仅凭文档长度即可完成排序，这使得在其上使用精确率类指标变得毫无参考意义；重新生成……（原文在此处截断）

    arXiv:2611.23359v1 Announce Type: new  Abstract: A single academic vacancy can draw hundreds of applications, and the first screening round consumes most of a committee's time. We describe a CV Priority Sorter deployed at a national platform for academic vacancies and integrated with its vacancy database. It proposes a reading order for a pile of CVs and leaves every decision to the committee, pairing a frozen multilingual transformer encoder with a per-session linear model that retrains after each label and is discarded when the session ends. It works: on a pool of 300 synthetic CVs, a committee reading in the tool's order reaches 95% of the relevant candidates after half the pile, where an arbitrary order needs almost all of it. This paper is about what we found when we audited that claim, rather than about the system itself. Three findings changed the product. The pool turned out to be sortable by document length alone, which makes precision-style metrics on it uninformative; regene
    
[^13]: PGPO：面向隐私保护LLM增强推荐的原型引导渐进式混淆

    PGPO: Prototype-Guided Progressive Obfuscation for Privacy-Preserving LLM-Enhanced Recommendation

    [https://arxiv.org/abs/2611.23236](https://arxiv.org/abs/2611.23236)

    该论文提出PGPO方法，通过为高影响力原型词学习拓扑感知的混淆锚点并在静态语义图中渐进传播，在有效防御嵌入反演攻击保护隐私的同时保持物品间语义关系，实现接近上限的推荐性能。

    

    LLM增强的推荐系统将物品的辅助信息编码为语义嵌入，但这些嵌入可能通过反演攻击泄露敏感的文本属性。现有的输入侧防御依赖于可信收集或局部词元扰动，往往会破坏推荐所需的物品间语义拓扑结构。为解决这一问题，我们引入关系保真度作为关键目标，要求在实现隐私保护的语义偏移的同时，保留物品间的相对关系以维持推荐效用。基于这一原则，我们提出了原型引导的渐进式混淆（PGPO），该方法为高影响力的原型词学习拓扑感知的混淆锚点，并通过静态语义图进行传播。PGPO使用GRPO进行优化，采用个体差异性、结构一致性和语义约束三种奖励。在两个基准数据集上的实验表明，PGPO在实现强大反演攻击防御的同时，推荐性能接近上限。

    arXiv:2611.23236v1 Announce Type: cross  Abstract: LLM-enhanced recommender systems encode item side information into semantic embeddings, but these embeddings may leak sensitive textual attributes through inversion attacks. Existing input-side defenses rely on trusted collection or local token perturbation, often disrupting the inter-item semantic topology needed for recommendation. To address it, we introduce relational fidelity as the key objective, requiring semantic shift for privacy while preserving relative item relations for utility. Based on this principle, we propose prototype-guided progressive obfuscation (PGPO), which learns topology-aware obfuscation anchors for high-influence prototype words and propagates them through a static semantic graph. PGPO is optimized with GRPO using individual dissimilarity, structural consistency, and semantic constraint rewards. Experiments on two benchmarks show strong inversion defense with near-upper-bound recommendation performance. The 
    
[^14]: 可迁移的解释：用稀疏自编码器解读医学信息检索

    Explanations that Transfer: Interpreting Medical Retrieval with Sparse AutoEncoders

    [https://arxiv.org/abs/2611.18179](https://arxiv.org/abs/2611.18179)

    本文提出将稀疏自编码器应用于医学信息检索，通过结合重建目标与InfoNCE对比损失，将不透明的稠密文本嵌入转化为语义可解释的表示，同时保持对新临床语料库的可迁移性。

    

    回答用户查询的医学AI系统依赖于从大型文档集合中检索医学知识。在此背景下，在医学数据集上微调通用模型可以提升检索相关性。然而，现代检索系统所依赖的稠密嵌入在很大程度上仍是不透明的，难以理解其检索结果背后的机制。此外，针对特定医学数据集进行适配可能会失去对新临床语料库的可迁移性。在本工作中，我们通过将稀疏自编码器（SAEs）应用于医学信息检索来解决这些问题，将稠密文本嵌入转化为语义上可解释的表示。为此，我们的模型将重建目标与InfoNCE对比损失相结合，后者能够直接保留查询与文档之间的相关性。我们在六个医学检索数据集上（涵盖域内与域外场景）评估了三种嵌入模型。在医学语料库上训练后，我们的稀疏……

    arXiv:2611.18179v1 Announce Type: cross  Abstract: Medical AI systems that answer users' queries rely on retrieving medical knowledge from large document collections. In this context, finetuning general models on medical datasets can improve retrieval relevance. Yet the dense embeddings underlying modern retrieval systems remain largely opaque, making it difficult to understand the mechanics behind their results. Moreover, adapting to a specific medical dataset risks losing transferability to new clinical corpora. In this work, we tackle these issues by applying Sparse Autoencoders (SAEs) to medical information retrieval, transforming dense text embeddings into semantically interpretable representations. To do so, our model combines a reconstruction objective with an InfoNCE contrastive loss that directly preserves query-document relevance. We evaluate three embedding models across six medical retrieval datasets, both in-domain and out-of-domain. Trained on a medical corpus, our sparse
    
[^15]: 面向异构数据源上企业零售工作流的代理式检索增强生成研究

    Toward Agentic Retrieval-Augmented Generation for Enterprise Retail Workflows over Heterogeneous Data Sources

    [https://arxiv.org/abs/2610.25096](https://arxiv.org/abs/2610.25096)

    本文提出了RetailBridge零售基准和RetailRAG框架，通过自动模式简报、数据源路由计划串联和诊断性计划修订等机制，使智能体能够跨异构数据源识别、串联证据并生成有依据的答案，从而完成企业零售工作流任务，其性能在所有评估维度上全面超越现有代理式RAG基线。

    

    企业零售工作流要求智能体在异构数据源之间串联证据——从结构化的产品和订单记录到非结构化的咨询和评论——然而现有的代理式RAG基准很少涵盖这一挑战。为了实现对这类工作流的受控评估，我们提出了RetailBridge，这是一个基于角色驱动模拟构建的组合式零售基准。其问题通过跨异构数据源链接实体和中间答案来构建，要求智能体识别相关数据源、在数据源之间传递证据，并生成有依据的答案。为了系统地在这些零售工作流上评估大语言模型，我们开发了RetailRAG，这是一个具备零售感知能力的框架，配备了自动模式简报、基于数据源路由的计划串联以及诊断性计划修订功能。RetailRAG在所有评估维度上持续优于现有的代理式RAG基线，可作为评估的强力参考系统。

    arXiv:2610.25096v2 Announce Type: replace  Abstract: Enterprise retail workflows require agents to chain evidence across heterogeneous sources, from structured product and order records to unstructured inquiries and reviews, yet existing agentic RAG benchmarks rarely capture this challenge. To enable controlled evaluation of such workflows, we introduce RetailBridge, a compositional retail benchmark built from persona-driven simulations. Its questions are constructed by linking entities and intermediate answers across heterogeneous sources, requiring agents to identify the relevant sources, carry evidence between them, and produce a grounded answer. To systematically evaluate LLMs on these retail workflows, we develop RetailRAG, a retail-aware framework equipped with auto-schema briefing, source-routed plan chaining, and diagnostic plan revision. RetailRAG consistently outperforms existing agentic RAG baselines across all evaluation axes, serving as a strong reference system for assess
    
[^16]: 推荐质量与消费集中度：来自Netflix的实验证据

    Recommendation Quality and the Concentration of Consumption: Experimental Evidence from Netflix

    [https://arxiv.org/abs/2608.21274](https://arxiv.org/abs/2608.21274)

    该研究表明，改进推荐算法可提升总消费并分散消费从热门到中等热门产品，而非极化消费，从而支持对中尾产品的投资。

    

    摘要：arXiv:2608.21274v1 公告类型：交叉 摘要：我们针对Netflix推荐系统进行了一项涉及850万用户的实验，以衡量推荐技术的改进如何影响被消费的产品集合。改进提高了总消费量和用户对推荐的依赖，同时使推荐和消费从最受欢迎的标题（“超级明星”）扩散到数量更多的中等热门标题（“中尾”），对最冷门的标题（“长尾”）影响甚微。我们的结果挑战了推荐系统会使消费极化——即提高头部和尾部消费份额而牺牲中间部分——的观点，并表明随着算法改进和平台规模化，投资于中尾产品的回报会增长。

    arXiv:2608.21274v1 Announce Type: cross  Abstract: We study an experiment with 8.5 million users on Netflix's recommender system to measure how improvements in recommendation technology affect the set of products that get consumed. Improvements increase total consumption and users' reliance on recommendations while diffusing recommendations and consumption away from the most popular titles (``superstars") toward a larger number of moderately popular titles (``middle-tail"), with minimal effects on the most niche titles (``long-tail"). Our results challenge the notion that recommender systems polarize consumption -- raising the consumption shares of the head and tail at the expense of the middle -- and suggest that the returns to investing in middle-tail products grow as algorithms improve and platforms scale.
    
[^17]: UniRank：面向统一序列建模与特征交互的排序模型基准测试

    UniRank: Benchmarking Ranking Models for Unified Sequential Modeling and Feature Interaction

    [https://arxiv.org/abs/2607.19987](https://arxiv.org/abs/2607.19987)

    UniRank提出了一个统一序列建模与特征交互的开放排序模型基准，通过标准化的评估流程和降低硬件门槛的高效PyTorch工具包，首次实现了15个代表性统一排序模型在公开数据集上的可复现比较。

    

    排序是在线广告和推荐系统中的核心环节。现代排序模型日益将序列建模与特征交互相统一，然而许多研究进展依赖于专有数据、闭源实现和大规模工业基础设施。这种现状限制了可复现的模型比较，也阻碍了学术界对扩展规律、长序列建模和多任务排序的研究。为解决这些局限性，本文提出了UniRank，一个统一序列建模与特征交互的排序模型开放基准。UniRank采用按时间顺序的逐点自回归监督方式，标准化了跨反馈任务的评估流程，并提供了一个PyTorch工具包，包含分布式数据并行训练、算子优化、混合精度训练、注意力优化等效率提升技术，从而降低了硬件需求。我们在五个公开数据集上对15个具有代表性的统一排序模型进行了基准测试。

    arXiv:2607.19987v3 Announce Type: replace  Abstract: Ranking is a core stage in online advertising and recommender systems. Modern ranking models increasingly unify sequential modeling and feature interaction, yet many advances rely on proprietary data, closed implementations, and large-scale industrial infrastructure. This setting limits reproducible comparison and hinders academic study of scaling laws, long-sequence modeling, and multi-task ranking. To address these limitations, this paper proposes UniRank, an open benchmark for ranking models that unify sequential modeling and feature interaction. UniRank uses chronological pointwise autoregressive supervision, standardizes evaluation across feedback tasks, and provides a PyTorch toolkit with Distributed Data Parallel training, operator optimization, mixed-precision training, attention optimization, and other efficiency techniques that reduce hardware requirements. We benchmark 15 representative unified ranking models on five publi
    
[^18]: 先知而后取：基于置信度的检索预算分配何时才有收益？

    Know Before You Fetch: When Does Confidence-Based Retrieval-Budget Allocation Pay Off?

    [https://arxiv.org/abs/2606.29959](https://arxiv.org/abs/2606.29959)

    该研究在相同的每查询平均检索预算上限下，将基于阅读器置信度的自适应检索分配与最佳简单策略（固定深度或其随机混合）对比，发现置信度分配在极紧预算（每查询平均半个段落）下于全部七个设置中稳定胜出（准确率提升0.010–0.075），其价值在于配给稀缺预算和避免有害检索。

    

    自适应检索增强生成利用阅读器的信号（例如其在无检索的“闭卷”作答时的置信度）来决定是否检索以及检索多少。我们不是将这些方法与总是检索大量段落的做法相比，而是在每查询平均段落数相同上限的约束下，与最佳的简单策略进行比较：固定深度检索或固定深度的随机混合。各策略分配零个、一个或五个段落，并在保留问题集上进行评分。七个约含2,000个问题的实验设置组合了TriviaQA、Natural Questions与MS MARCO数据集、Qwen3-8B与Qwen3-32B阅读器，以及两种检索源：按问题的段落池和350万段落的维基百科索引。基于置信度的分配可能从三个方面获益。它能配给稀缺的预算：在每查询平均半个段落的上限下，它在全部七个设置中都优于最佳简单策略，准确率提升0.010至0.075。它能避免有害检索：在单个段落的上限下……

    arXiv:2606.29959v2 Announce Type: replace-cross  Abstract: Adaptive retrieval-augmented generation uses a signal from the reader, such as its confidence when answering without retrieval (closed-book), to decide whether and how much to retrieve. We compare such methods not with always retrieving many passages but with the best simple policy under the same cap on average passages per query: a fixed depth or a random mixture of fixed depths. Policies allocate zero, one or five passages and are scored on held-out questions. Seven settings of about 2,000 questions combine TriviaQA, Natural Questions and MS MARCO, Qwen3-8B and Qwen3-32B readers, and two retrieval sources: per-question passage pools and a 3.5-million-passage Wikipedia index. Allocation by confidence could gain in three ways. It rations a scarce budget: at a cap of half a passage per query it beats the best simple policy in all seven settings, by 0.010 to 0.075 in accuracy. It avoids harmful retrieval: at caps of one passage o
    
[^19]: 基于大语言模型辅助的重排序：在推荐系统中实现细致目标

    LLM-Assisted Reranking to Operationalize Nuanced Objectives in Recommender Systems

    [https://arxiv.org/abs/2606.02883](https://arxiv.org/abs/2606.02883)

    本文基于真实YouTube轨迹实证检验了大语言模型辅助的重排序在提升个性化推荐的同时，是否会放大用户接触意识形态同质化、极端或阴谋论政治内容的风险。

    

    推荐系统塑造了我们能看到的信息，引发了人们对信息茧房、激进化、极化和社会不平等的担忧。大语言模型（LLM）能够实现更强大的个性化，但这可能加剧上述动态，然而大多数推荐系统都只为参与度或狭窄的准确性指标进行调优，很少关注更广泛的社会影响。我们研究了大语言模型辅助的重排序在提升个性化的同时，是否会无意中放大用户接触意识形态上契合的、极端的或阴谋论式政治内容的风险——这一风险此前仅在理论上被提出，尚未得到实证刻画。利用真实的YouTube浏览轨迹，我们通过零样本提示对三个大语言模型（并与嵌入基线对比）对平台侧边栏候选内容进行重排序，并评估相关性、意识形态一致性与多样性，以及接触阴谋论或极端主义内容的情况。我们发现（i）在纯个性化指令下，重……（摘要原文在此处截断）

    arXiv:2606.02883v2 Announce Type: replace-cross  Abstract: Recommender systems shape what information we see, raising concerns about filter bubbles, radicalization, polarization, and social inequality. Large language models (LLMs) enable more powerful personalization that may intensify these dynamics, yet most recommenders are tuned for engagement or narrow accuracy metrics, with little attention to broader social effects. We investigate whether LLM-assisted reranking, while improving personalization, inadvertently amplifies exposure to ideologically congenial, extreme, or conspiratorial political content, a risk theorized but not yet empirically characterized. Using real YouTube trajectories, we rerank the platform's sidebar candidates by zero-shot prompting across three models, alongside an embedding baseline, and evaluate relevance, ideological alignment and diversity, and exposure to conspiratorial or extremist content. We find that (i) under a pure personalization instruction, rer
    
[^20]: 基于脉冲神经网络学习二进制编码的跨模态哈希检索

    Learning Binary Codes with Spiking Neural Networks for Cross-Modal Hashing Retrieval

    [https://arxiv.org/abs/2606.00740](https://arxiv.org/abs/2606.00740)

    该论文提出了一种基于脉冲神经网络的无监督跨模态哈希框架，通过正负脉冲随时间积累的竞争来确定哈希比特，并让图像与文本通过时间交互共同生成哈希编码。

    

    跨模态哈希（CMH）使用紧凑的二进制编码实现高效的图像-文本检索。大多数方法在连续空间中学习图像-文本关系，仅在最后阶段才对表示进行二值化，因此语义学习和比特决策在很大程度上是相互分离的。本工作针对哈希比特生成的两个关键问题展开研究：**（1）如何确定哈希比特？** 我们利用脉冲活动的离散特性，通过随时间积累的正负脉冲之间的竞争来确定每个比特，而不是通过单次放电事件或对连续表示的后续阈值化处理。**（2）如何生成这样的哈希比特？** 我们提出了一种用于跨模态哈希的无监督脉冲框架，其中图像和文本通过时间交互共同参与哈希比特的生成。此外，激活约束进一步防止了由静默通道对导致的缺乏依据的决策。

    arXiv:2606.00740v2 Announce Type: replace  Abstract: Cross-modal hashing (CMH) uses compact binary codes for efficient image-text retrieval. Most methods learn image-text relationships in a continuous space and binarize the representation only at the end. Semantic learning and bit decisions thus remain largely separate. This work addresses two key questions concerning hash-bit generation: \textbf{(1) How to determine hash bits?} We leverage the discrete nature of spiking activity and determine each bit through competition between positive and negative spikes accumulated over time, rather than through a single firing event or subsequent thresholding of continuous representations. \textbf{(2) How to generate such hash bits?} We propose an unsupervised spiking framework for cross-modal hashing, in which image and text jointly contribute to hash-bit generation through temporal interaction. An activation constraint further prevents unsupported decisions caused by silent channel pairs. Exten
    
[^21]: PEARL：基于运行时定义概念的个性化流式视频理解

    PEARL: Personalized Streaming Video Understanding with Runtime-Defined Concepts

    [https://arxiv.org/abs/2603.20422](https://arxiv.org/abs/2603.20422)

    该论文提出个性化流式视频理解（PSVU）这一新任务及首个基准 PSVU-Bench，并给出即插即用基线 PEARL，使模型能够在流式视频播放过程中实时注册用户定义的新概念并回溯其历史出现时刻。

    

    与人类能够从连续的视觉流中学习新概念不同，现有的个性化视觉语言模型（VLM）在训练过程中内化了一组预定义的概念，并且只能处理静态图像或离线视频。它们无法注册用户在流式传输中途定义的概念，也无法回想起该概念此前出现的时刻。我们将这一任务形式化为个性化流式视频理解，并引入 PSVU-Bench，这是该任务的首个基准，对 132 个视频进行了带时间戳问题的标注。其概念定义、实时与过去时间问答要求模型在任意时间戳注册概念、识别其当前状态（包括用户定义的动作），并检索其历史证据。我们提出了 PEARL，一个强大的即插即用基线模型，它将用户定义的概念与存档的视频片段分开存储，并利用概念描述对每个查询进行重写以进行检索……

    arXiv:2603.20422v2 Announce Type: replace-cross  Abstract: Unlike humans, who learn new concepts from a continuous visual stream, existing personalized vision-language models (VLMs) internalize a predefined concept set during training and operate on static images or offline videos. They cannot register a concept that a user defines in the middle of a stream, nor recall the earlier moments in which it appeared. We formalize this task as Personalized Streaming Video Understanding (PSVU) and introduce PSVU-Bench, the first benchmark for this task, which annotates 132 videos with timestamped questions. Its Concept-Definition, Real-Time and Past-Time QA ask a model to register concepts at arbitrary timestamps, recognize their current state, including user-defined actions, and retrieve their historical evidence. We propose PEARL, a strong plug-and-play baseline that stores user-defined concepts apart from archived video clips and rewrites each query with concept descriptions to retrieve the 
    

