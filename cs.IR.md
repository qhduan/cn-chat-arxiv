# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Retrieved but not ranked: surface-form bias in structural retrieval, from mathematics to agent trajectories](https://arxiv.org/abs/2609.01556) | 该研究在竞赛数学与具身智能体轨迹两个领域以统一协议评测刻意分离表层形式与语义结构的嵌入检索，发现主流嵌入模型存在严重的表层形式（字面词汇）偏差：在结构相同但措辞伪装最重的任务上Hit@1跌至0.0%，未命中时胜出者几乎总是与查询词汇更相似的条目，表明当前嵌入检索锚定于字面文本而非深层结构。 |
| [^2] | [AutoConcept: Training-Free Concept-Guided Reranking for Metadata-Available Composed Image Retrieval](https://arxiv.org/abs/2609.01456) | 提出AutoConcept，一种无需训练的概念引导重排序方法，通过将概念证据转化为可解释的结构化记忆并结合推理时校准，在元数据可用的组合图像检索中显著提升早期排名表现。 |
| [^3] | [VerTox: Verifiable Reward-Guided Corpus Poisoning Against Neural Ranking Models](https://arxiv.org/abs/2609.01325) | 本文提出VerTox，首个将语料库投毒攻击形式化为可验证奖励引导强化学习问题的框架，通过将排序扭曲与事实性破坏耦合的奖励设计将小型LLM微调为对抗性文档生成器，对神经排序模型实现了接近完美的攻击成功率。 |
| [^4] | [MIDR: Enrichment-Augmented Indexing for Multimodal Document Retrieval](https://arxiv.org/abs/2609.01316) | MIDR是一个无需训练的富化增强索引框架，通过在索引阶段利用多模态大语言模型将文档页面转换为经验证的文本字段，将多模态推理从查询时转移到索引时，在ViDoRe V3上相比BM25相对提升23.0%，性能可与ColQwen2.5媲美。 |
| [^5] | [From Language to Behavior: Scaling Sequence Transformers for Industrial Recommendation Ranking with Rec-Native Designs](https://arxiv.org/abs/2609.01240) | 提出推荐原生的Transformer扩展框架ReST，通过双门控注意力编码器应对噪声化行为序列，通过重量级可复用编码器加轻量级交叉解码器的分解设计及共享前缀训练与服务机制，解决推荐排序中的计算不对称问题，实现工业推荐排序的高效规模化。 |
| [^6] | [World Model-Guided Reinforcement Learning via Counterfactual User Engagement Simulation](https://arxiv.org/abs/2609.01067) | 提出WMG-RL框架，利用冻结的用户参与世界模型（UEWM）在真实用户接触前模拟反事实用户反馈，为面向用户的强化学习智能体提供奖励监督，降低在线反馈收集的成本与风险。 |
| [^7] | [Web Price Extraction: State of the Art and an Adaptive Browserless Implementation](https://arxiv.org/abs/2609.01030) | 该论文系统梳理了网页价格提取四大类方法的优劣权衡，并提出了一种自适应的无浏览器价格提取实现，兼顾速度、成本与跨网站结构的适应性。 |
| [^8] | [SwapRec: Warming Up Cold Items Through Training-Time Swaps](https://arxiv.org/abs/2609.00913) | 本文提出SwapRec方法，通过在训练阶段就将冷启动物品替换为其最相似的热门邻居物品，使序列推荐模型对推理时的物品交换操作更加鲁棒，从而改善冷门物品的实时个性化推荐效果。 |
| [^9] | [Staged Linguistic Seeding: Grounded Query Expansion for Verified-Unit QA in AI Contact Centers](https://arxiv.org/abs/2609.00844) | 提出分阶段语言播种（SLS）方法，通过“人工撰写槽位模板—大模型生成变体—轻量人工审核”的流程离线增强检索索引，使AI呼叫中心仅凭单次检索、无查询时生成即可从已验证问答单元中高准确率作答，在两个工业领域将混合R@1提升至0.881/0.930。 |
| [^10] | [Ctrl-F-Resist. Practices, Challenges, and Technical Needs of Civil Society Organizations Monitoring the Far-Right Online](https://arxiv.org/abs/2609.00808) | 本文通过对12家德国公民社会组织的15名从业者进行定性研究，揭示了这些组织在极右翼在线监测工作中的长期实践、面临的挑战（法律不确定性、平台访问受限、资金不足）以及技术需求，强调它们是数字治理中被忽视的关键利益相关者。 |
| [^11] | [From Saliency to Discriminability: Rank-Preserving Visual Token Pruning for VLM Rerankers](https://arxiv.org/abs/2609.00667) | 提出无需训练的RaDiCal框架，利用归一化注意力熵判断显著性何时可信，并将其与排序判别先验融合，实现保序的视觉Token剪枝，从而高效加速VLM重排序器。 |
| [^12] | [It Takes Two to Match: Co-Evolving Generative Retriever with Reinforcement Learning](https://arxiv.org/abs/2609.00638) | 提出 CoGR 框架，通过强化学习让大语言模型协同进化，在查询侧和物品侧同时生成对齐的关键词表示，并直接经倒排索引完成匹配，在兼容现有关键词检索基础设施的同时提升检索效果。 |
| [^13] | [Towards Effective Structured Context Modeling for Conversational Recommender Systems via Dual-node Monte Carlo Tree Search](https://arxiv.org/abs/2609.00618) | 提出DREAMS框架，通过双节点树结构（引导节点用蒙特卡洛树搜索探索对话动作以推断潜在偏好，利用节点用大语言模型将偏好状态精炼为结构化检索查询），显式建模对话式推荐系统中用户偏好的多轮演化。 |
| [^14] | [NeuroGraph: An AI Graph-Driven Neuro-Symbolic Framework for Explainable Threat Reasoning in Advanced Manufacturing](https://arxiv.org/abs/2609.00604) | 本文提出NeuroGraph框架，通过融合本体感知符号查询生成、知识图谱检索与神经语言生成的双大语言模型神经-符号架构，解决现有方法的幻觉与多跳推理不足问题，实现先进制造IT/OT环境中准确且可解释的网络威胁推理。 |
| [^15] | [TRIS: A Tri-Layer Retrieval Integrity Sieve Against Knowledge Poisoning](https://arxiv.org/abs/2609.00470) | 本文提出TRIS三层筛，一种中间件防御方案，通过跨嵌入空间聚类、触发器-载荷结构过滤和大模型一致性验证三重机制清洗RAG检索证据，利用投毒文档难以同时满足嵌入几何、内部结构和生成目标三重要求的固有弱点，有效抵御知识投毒攻击。 |
| [^16] | [Closed Forms and Synthetic Twins: Predicting Approximate Nearest Neighbor Recall from Embedding Statistics](https://arxiv.org/abs/2609.00364) | 本文证明无需构建真实索引，仅凭原始嵌入的无标签统计量——针对PQ、FDE等固定网格量化器的闭式矩统计量以及在合成孪生语料库上的模拟——就能在部署前预测近似最近邻索引的召回率。 |
| [^17] | [Sources of Truth: A Multi-Platform, Multilingual Audit of Citations in AI Mental Health Information Queries](https://arxiv.org/abs/2609.00319) | 该研究对ChatGPT、Perplexity和Google AI Overview在多语言心理健康查询中生成的15,942条引用进行了系统审计，发现引用来源高度集中于少数域名，表明来源评估责任已从用户转移到AI平台。 |
| [^18] | [Two-Sided State-Space Models for Sequential Recommendation with Non-Random Multimodal Review Feedback](https://arxiv.org/abs/2609.00165) | 提出双边状态空间模型TS-SSM，将非随机生成的多模态评论反馈同时融入用户状态与商品状态的动态演化过程，从而提升事件条件下的序列推荐效果。 |
| [^19] | [SilentProbe: Measuring Silent Failure in Production APIs Used as Agent Tools](https://arxiv.org/abs/2609.00035) | 该论文首次大规模测量了LLM智能体调用生产API时的“静默失败”现象，发现API模式中机器可校验的约束（而非供应商身份）是预测服务器能否诚实报错的关键因素，而当前OpenAPI文档中这类约束严重缺失。 |
| [^20] | [E-SENS: Exclusion-Sensitive Penalization for Negative-Constraint Retrieval](https://arxiv.org/abs/2608.30130) | E-SENS是一种无需训练的重排序方法，通过为被排除概念提取“陷阱查询”并从检索分数中减去其相似度，有效惩罚与用户排除概念相关的文档，从而提升检索系统对负向约束的遵守能力。 |
| [^21] | [REIGN: Refurbished Embeddings with Integrated Guidance Networks for Efficient Context-Length Scaling](https://arxiv.org/abs/2608.29899) | REIGN通过在冻结引导网络生成的块嵌入序列上运行对比训练的双编码器，将词元级处理与文档级推理解耦，使长文档检索的训练成本相比分块Transformer微调降低约四个数量级。 |
| [^22] | [Validating FKG.in: Soundness Assessment in LLM-Augmented Indian Food Knowledge](https://arxiv.org/abs/2608.29249) | 本文作为印度食品知识图谱FKG.in的一部分，提出了一种半自动化的健全性评估工作流程，通过结合形式文法、词汇检查、统计启发式、Set Transformer连贯性建模和检索验证的多阶段方法，识别并解决LLM从非正式烹饪来源提取和增强结构化食谱数据时的常见失败模式。 |
| [^23] | [RATIO: A Benchmark for Retrieval Across Typed Ideation Operations in Scientific Literature](https://arxiv.org/abs/2608.27394) | RATIO基准首次定义了三种科学构思操作（Address、Broaden、Specify）的检索任务，并利用远距离监督扩展到大规模语料库，为科学文献的灵感检索提供了新范式。 |
| [^24] | [DCA-MoE: Spatially Adaptive Cross-Layer Fusion and Density-Routed Experts for Crowd Counting](https://arxiv.org/abs/2608.15213) | 本文提出DCA-MoE框架，通过空间自适应层融合和密度路由多感受野专家，在冻结编码器下实现内容相关的人群计数解码，显著提升了对视角和尺度变化的适应性。 |
| [^25] | [Field-Aware Agent Skill Retrieval](https://arxiv.org/abs/2608.02880) | 该论文将智能体技能建模为结构化的多字段对象，对每个字段独立计算稀疏与稠密相似度并用均匀权重或小型MLP融合，从而显著提升终身学习智能体的技能检索效果。 |
| [^26] | [Closing the Operational Gap in Semantic Caching](https://arxiv.org/abs/2606.19719) | 该论文指出PR-AUC指标会误导语义缓存系统的部署决策，提出了缓存感知的P-CHR AUC指标和运营保留率ORR，并将离线与部署质量间的运营差距分解为可恢复的阈值效用部分和由数据集正例率决定的不可约简结构部分。 |
| [^27] | [APEX-EM: Non-Parametric Online Learning for Autonomous Agents via Structured Procedural-Episodic Experience Replay](https://arxiv.org/abs/2603.29093) | APEX-EM提出了一种无需更新模型权重的非参数化经验记忆方法，通过程序性知识图谱存储完整的任务轨迹并同时索引成功与失败经验，使LLM智能体能够复用过往经验而无需重复推理，在相同底层模型对比下BigCodeBench迁移任务上提升7.6个百分点。 |
| [^28] | [TC-RAG:Turing-Complete RAG's Case study on Medical LLM Systems](https://arxiv.org/abs/2408.09199) | 提出TC-RAG框架，通过引入图灵完备系统管理状态变量，并利用具备自适应检索、推理和规划能力的记忆栈系统，实现检索过程的受控终止并减少错误知识累积，从而显著提升医疗大语言模型系统知识检索的效率与准确性。 |

# 详细

[^1]: 检索到了却排名不对：从数学到智能体轨迹的结构化检索中的表层形式偏差

    Retrieved but not ranked: surface-form bias in structural retrieval, from mathematics to agent trajectories

    [https://arxiv.org/abs/2609.01556](https://arxiv.org/abs/2609.01556)

    该研究在竞赛数学与具身智能体轨迹两个领域以统一协议评测刻意分离表层形式与语义结构的嵌入检索，发现主流嵌入模型存在严重的表层形式（字面词汇）偏差：在结构相同但措辞伪装最重的任务上Hit@1跌至0.0%，未命中时胜出者几乎总是与查询词汇更相似的条目，表明当前嵌入检索锚定于字面文本而非深层结构。

    

    我们在刻意将表层形式与含义分离的设定下评估嵌入检索：在统一协议下，于两个互不相关的领域中检索共享底层结构但措辞不同的条目——竞赛数学与具身智能体轨迹（基于ALFWorld衍生；118个查询，336条轨迹）。在数学领域，失败是彻底的：在伪装最重的层级上，两个生产级嵌入模型的严格Hit@1均为0.0%（自助法95%置信区间[0.0, 0.0]），而正确条目几乎总能进入前10名；并且在95.2%至99.8%的未命中情形中，胜出者与查询的词汇相似度高于正确答案。在轨迹领域，表层变化只是附带性的，当正确答案必须涉及不同物体时，同样的模型表现落在超几何随机水平或其附近；而一旦要求正确答案在物体与容器上都必须不同，三个嵌入模型的表现均低于随机水平：检索锚定于字面文……（原文摘要到此截断）

    arXiv:2609.01556v1 Announce Type: cross  Abstract: We evaluate embedding retrieval where surface form and meaning are pulled apart on purpose: retrieving items that share underlying structure but not wording, in two unrelated domains under one protocol, competition mathematics (MathNet-Retrieve; 500 queries, 117,088-item corpus) and embodied-agent trajectories (ALFWorld-derived; 118 queries, 336 trajectories). In mathematics the failure is complete: strict Hit@1 at the heaviest disguise tier is 0.0% for both production embedders (bootstrap 95% CI [0.0, 0.0]) while the correct item sits in the top 10 nearly always, and in 95.2 to 99.8% of misses the winner is more lexically similar to the query than the correct answer. In trajectories, where surface variation is incidental, the same models land at or near hypergeometric chance when gold must involve a different object, and below chance for all three embedders once gold must differ in object and receptacle: retrieval anchors on literal t
    
[^2]: AutoConcept：面向元数据可用的组合图像检索的无训练概念引导重排序

    AutoConcept: Training-Free Concept-Guided Reranking for Metadata-Available Composed Image Retrieval

    [https://arxiv.org/abs/2609.01456](https://arxiv.org/abs/2609.01456)

    提出AutoConcept，一种无需训练的概念引导重排序方法，通过将概念证据转化为可解释的结构化记忆并结合推理时校准，在元数据可用的组合图像检索中显著提升早期排名表现。

    

    组合图像检索（CIR）根据参考图像和文本修改描述来检索目标图像。本文研究元数据可用的CIR重排序任务，即由固定的CIR模型首先返回候选池，随后利用图库元数据进行第二阶段的概念引导打分。我们提出AutoConcept，一种无需训练的重排序器，它将概念证据转换为可解释的记忆结构。AutoConcept过滤噪声概念，通过辅助负向惩罚激活与查询相关的正向约束，并通过推理时校准将基础检索得分与基于元数据的概念-候选对齐相结合。在FashionIQ数据集上，AutoConcept相比WeiMoCIR带来了显著的靠前排名提升，并在LinCIR候选池上取得了一致的即插即用增益。元数据感知的对照实验表明，结构化概念记忆在直接的查询-文本匹配和提取属性匹配之外提供了额外信号，而仅查询变体进一步支持了这一结论。

    arXiv:2609.01456v1 Announce Type: cross  Abstract: Composed image retrieval (CIR) retrieves a target image from a reference image and a text modification. This paper studies metadata-available CIR reranking, where a fixed CIR model first returns a candidate pool and gallery metadata is then used for second-stage concept-guided scoring. We introduce AutoConcept, a training-free reranker that converts concept evidence into an interpretable memory. AutoConcept filters noisy concepts, activates query-relevant positive constraints with an auxiliary negative penalty, and combines base retrieval scores with metadata-based concept-candidate alignment through inference-time calibration. On FashionIQ, AutoConcept yields significant early-rank improvements over WeiMoCIR and consistent plug-in gains on LinCIR candidate pools. Metadata-aware controls show that structured concept memory adds signal beyond direct query-text and extracted-attribute matching, while a query-only variant further supports
    
[^3]: VerTox：可验证奖励引导的针对神经排序模型的语料库投毒攻击

    VerTox: Verifiable Reward-Guided Corpus Poisoning Against Neural Ranking Models

    [https://arxiv.org/abs/2609.01325](https://arxiv.org/abs/2609.01325)

    本文提出VerTox，首个将语料库投毒攻击形式化为可验证奖励引导强化学习问题的框架，通过将排序扭曲与事实性破坏耦合的奖励设计将小型LLM微调为对抗性文档生成器，对神经排序模型实现了接近完美的攻击成功率。

    

    神经排序模型已成为现代信息检索系统的核心组件，也是检索增强生成（RAG）流水线等人工智能系统的重要构建模块。然而，在大语言模型（LLM）能够大规模生成流畅且具有欺骗性内容的背景下，这些模型的鲁棒性仍未得到充分理解。本研究探讨了神经排序模型对语料库投毒攻击的脆弱性，此类攻击中，攻击者向语料库注入少量恶意构造的文档以扭曲排序行为。我们提出了VerTox，这是首个将语料库投毒形式化为可验证奖励引导强化学习（RLVR）问题的框架。通过专门的奖励设计将排序扭曲与事实性破坏显式耦合，我们将紧凑型LLM微调为对抗性生成器。实验表明，我们的方法实现了接近完美的攻击成功率。

    arXiv:2609.01325v1 Announce Type: new  Abstract: Neural ranking models have become core components of modern information retrieval systems and important building blocks of AI systems such as retrieval-augmented generation (RAG) pipelines. However, their robustness remains insufficiently understood in the presence of large language models (LLMs), which can generate fluent and deceptive content at scale. This work investigates the vulnerability of neural ranking models to corpus poisoning attacks, in which an adversary injects a small number of maliciously crafted documents into the corpus to distort ranking behavior. We propose VerTox, the first framework to formulate corpus poisoning as a verifiable reward-guided reinforcement learning (RLVR) problem. By explicitly coupling ranking distortion with factual corruption through specialized reward shaping, we fine-tune compact LLMs into adversarial generators. Experiments demonstrate that our method achieves near-perfect attack success rate
    
[^4]: MIDR：面向多模态文档检索的富化增强索引

    MIDR: Enrichment-Augmented Indexing for Multimodal Document Retrieval

    [https://arxiv.org/abs/2609.01316](https://arxiv.org/abs/2609.01316)

    MIDR是一个无需训练的富化增强索引框架，通过在索引阶段利用多模态大语言模型将文档页面转换为经验证的文本字段，将多模态推理从查询时转移到索引时，在ViDoRe V3上相比BM25相对提升23.0%，性能可与ColQwen2.5媲美。

    

    对视觉丰富文档的检索存在一个表示难题：重要内容往往存在于表格、图表、图形和布局关系中，而普通OCR会将其线性化、破坏或遗漏。ColPali系列视觉检索器通过补丁级多向量索引和后期交互评分来解决这一问题，但这使图像衍生的检索保留在查询时的服务路径上。我们提出MIDR（Multimodal Indexing for Document Retrieval，面向文档检索的多模态索引），这是一个无需训练的富化增强索引框架，将多模态推理转移到索引阶段。在数据摄取过程中，多模态大语言模型将渲染的页面转换为经过验证的文本字段，并使用BM25F进行索引，可选择与稠密检索融合，从而在多模态扎根的证据之上实现以文本为中心的服务。在ViDoRe V3上，MIDR Hybrid在五个英文领域取得0.6219的平均nDCG，相比BM25相对提升23.0%，与ColQwen2.5保持竞争力。

    arXiv:2609.01316v1 Announce Type: cross  Abstract: Retrieval over visually rich documents has a representation problem: important content often lives in tables, charts, figures, and layout relations that plain OCR linearizes, corrupts, or omits. ColPali-family visual retrievers address this with patch-level multi-vector indexes and late-interaction scoring, keeping image-derived retrieval on the query-time serving path. We introduce MIDR (Multimodal Indexing for Document Retrieval), a training-free framework for enrichment-augmented indexing that shifts multimodal reasoning to index time. During ingestion, a multimodal LLM converts rendered pages into verified textual fields that are indexed with BM25F and optionally fused with dense retrieval, enabling text-centric serving over multimodally grounded evidence. On ViDoRe V3, MIDR Hybrid achieves 0.6219 average nDCG across five English domains, a 23.0% relative gain over BM25, remaining competitive with ColQwen2.5. On two French-document
    
[^5]: 从语言到行为：面向工业推荐排序的推荐原生序列Transformer扩展设计

    From Language to Behavior: Scaling Sequence Transformers for Industrial Recommendation Ranking with Rec-Native Designs

    [https://arxiv.org/abs/2609.01240](https://arxiv.org/abs/2609.01240)

    提出推荐原生的Transformer扩展框架ReST，通过双门控注意力编码器应对噪声化行为序列，通过重量级可复用编码器加轻量级交叉解码器的分解设计及共享前缀训练与服务机制，解决推荐排序中的计算不对称问题，实现工业推荐排序的高效规模化。

    

    扩展Transformer架构在语言建模领域带来了巨大的性能提升，但将这一方法移植到生产级排序系统中的行为序列建模却充满挑战：推荐系统在信号质量上存在差异——行为序列充满噪声、时间上不规则且监督信号稀疏；在计算不对称性上也存在差异——每个请求需要在严格的延迟预算下，用一份共享的用户历史对大量候选物品进行打分。我们提出了ReST，一个推荐原生的Transformer扩展框架。针对信号质量问题，它引入了一个序列编码器，包含双门控注意力、旋转位置与时间嵌入、稳定的残差归一化，以及仅在训练阶段使用的辅助目标。针对计算不对称性问题，它将排序分解为重量级可复用的编码器和轻量级交叉解码器，采用无投影的KV注意力和针对token的特定参数化，并将用户级共享前缀训练与共享前缀服务相结合，以实现计算高效的服务部署。

    arXiv:2609.01240v1 Announce Type: cross  Abstract: Scaling Transformers has driven large gains in language modeling, but transplanting this to behavior-sequence modeling in production ranking is challenging: recommendation differs in signal quality, where behavior sequences are noisy, temporally irregular, and sparsely supervised, and in computation asymmetry, where each request scores many candidates against one shared user history under tight latency budgets. We propose ReST, a recommendation-native Transformer scaling framework. For signal quality, it introduces a sequence encoder with dual-gated attention, rotary positional and temporal embedding, stabilized residual normalization, and training-only auxiliary objectives. For computation asymmetry, it factorizes ranking into a heavy reusable encoder and a lightweight cross decoder with projection-free KV attention and token-specific parameterization, coupling user-level shared-prefix training with shared-prefix serving for compute-o
    
[^6]: 基于反事实用户参与模拟的世界模型引导强化学习

    World Model-Guided Reinforcement Learning via Counterfactual User Engagement Simulation

    [https://arxiv.org/abs/2609.01067](https://arxiv.org/abs/2609.01067)

    提出WMG-RL框架，利用冻结的用户参与世界模型（UEWM）在真实用户接触前模拟反事实用户反馈，为面向用户的强化学习智能体提供奖励监督，降低在线反馈收集的成本与风险。

    

    面向以用户为中心的智能体的强化学习受到在线反馈收集的成本、延迟和风险，以及在相同用户状态下缺乏反事实比较的限制。在本文中，我们提出了基于反事实用户参与模拟的世界模型引导强化学习（WMG-RL），这是一个在真实用户接触之前由冻结的用户模拟器提供奖励监督的框架。受语言世界模型的启发，我们将该模拟器实例化为用户参与世界模型（UEWM），该模型将推荐物品视为智能体的动作，将用户的异构反馈视为环境观测。UEWM并非学习单一的固定环境转移，而是学习从参与历史中推断用户特定的动态，并将其应用于候选物品。在WMG-RL中，下游策略针对同一历史提出多个候选物品；UEWM预测相应的参与反馈……

    arXiv:2609.01067v1 Announce Type: new  Abstract: Reinforcement learning for user-centric agents is limited by the cost, latency, and risk of collecting online feedback, as well as by the lack of counterfactual comparisons under the same user state. In this paper, we propose World Model-Guided Reinforcement Learning via counterfactual user engagement simulation (WMG-RL), a framework in which a frozen user simulator provides reward supervision before real user exposure. Motivated by language world models, we instantiate the simulator as a User Engagement World Model (UEWM), which treats a recommended item as the agent action and the user's heterogeneous feedback as the environment observation. Rather than learning one fixed environment transition, UEWM learns to infer user-specific dynamics from engagement history and apply them to candidate items. In WMG-RL, a downstream policy proposes multiple candidate items for the same history; UEWM predicts the corresponding engagement feedback in
    
[^7]: 网页价格提取：技术现状与一种自适应的无浏览器实现

    Web Price Extraction: State of the Art and an Adaptive Browserless Implementation

    [https://arxiv.org/abs/2609.01030](https://arxiv.org/abs/2609.01030)

    该论文系统梳理了网页价格提取四大类方法的优劣权衡，并提出了一种自适应的无浏览器价格提取实现，兼顾速度、成本与跨网站结构的适应性。

    

    从网站中提取价格是电子商务中市场监测、价格比较和商业分析的一项关键任务。现有方法大致可分为四类，理解它们在准确性和可扩展性之间的权衡对于选择合适的提取策略至关重要。经典方法依赖于人工编写的包装器和从标注页面中归纳的规则，准确性高，但对网页结构变化的适应性差，且需要大量的维护工作。基于浏览器的方法（如使用Selenium和Puppeteer等工具）能够处理动态JavaScript内容，但消耗大量计算资源且可扩展性差。无浏览器（browserless）方法通过HTTP请求直接获取HTML，在速度和成本方面具有显著优势，但依赖于针对特定网站校准的规则。基于机器学习和大型语言模型的方法具有良好的适应性，但需要训练数据和大（摘要在此处截断）

    arXiv:2609.01030v1 Announce Type: cross  Abstract: Price extraction from websites is a key task for market monitoring, price comparison, and business analytics in e-commerce. Existing approaches can be broadly divided into four groups, and understanding their trade-offs in accuracy and scalability is essential for selecting suitable extraction strategies. Classical methods rely on manually written wrappers and rule induction from labeled pages, offering high accuracy but adapting poorly to structural changes and requiring considerable maintenance effort. Browser-based methods, using tools such as Selenium and Puppeteer, handle dynamic JavaScript content but consume large computational resources and scale poorly. Browserless approaches retrieve HTML directly via HTTP requests, offering significant gains in speed and cost, but rely on rules calibrated for specific sites. Methods based on machine learning and large language models offer adaptability but require training data and substanti
    
[^8]: SwapRec：通过训练时交换来预热冷门物品

    SwapRec: Warming Up Cold Items Through Training-Time Swaps

    [https://arxiv.org/abs/2609.00913](https://arxiv.org/abs/2609.00913)

    本文提出SwapRec方法，通过在训练阶段就将冷启动物品替换为其最相似的热门邻居物品，使序列推荐模型对推理时的物品交换操作更加鲁棒，从而改善冷门物品的实时个性化推荐效果。

    

    与冷门物品的交互会对基于ID的推荐系统的实时个性化产生负面影响。这是因为使用这类交互会降低用户偏好估计的准确性，而将冷门物品从用户画像中排除则会阻碍实时推荐更新。在工业场景中，推理阶段常用的一种启发式方法是将冷启动物品替换（即“交换”）为其最相似的“热门”邻居物品，其中相似性是从物品的辅助信息中推断得出的。在本文中，我们证明了通常用于实时个性化的序列模型对此类交换并不鲁棒，并提出了SwapRec方法来解决这一问题。SwapRec依赖于在训练阶段就使用相同的交换启发式方法。我们将SwapRec应用于最先进的序列推荐模型，并通过三个推荐领域的定量实验分析了其影响。

    arXiv:2609.00913v1 Announce Type: new  Abstract: Interactions with cold items negatively impact real-time personalization of ID-based recommender systems. This is because the use of such interactions degrades user preference estimates, whereas excluding cold items from the user profile prevents real-time recommendation updates. In industrial scenarios, one heuristic often applied to address this shortcoming at inference time is to replace, i.e., "swap", cold-start items by their most similar "warm" neighbor, where similarity is inferred from the items' side information. In this paper, we demonstrate that sequential models, most often used for real-time personalization, are not robust to such swaps, and propose SwapRec, an approach to address this issue. SwapRec relies on using the same swap heuristics already at training time. We apply SwapRec to state-of-the-art models for sequential recommendation and analyze its impact by means of quantitative experiments in three recommendation dom
    
[^9]: 分阶段语言播种：面向AI呼叫中心已验证单元问答的接地查询扩展

    Staged Linguistic Seeding: Grounded Query Expansion for Verified-Unit QA in AI Contact Centers

    [https://arxiv.org/abs/2609.00844](https://arxiv.org/abs/2609.00844)

    提出分阶段语言播种（SLS）方法，通过“人工撰写槽位模板—大模型生成变体—轻量人工审核”的流程离线增强检索索引，使AI呼叫中心仅凭单次检索、无查询时生成即可从已验证问答单元中高准确率作答，在两个工业领域将混合R@1提升至0.881/0.930。

    

    AI呼叫中心（AICC）中的客服问答面临着基准测试QA所忽视的部署约束：语音热线延迟要求严苛，且自动回答缺乏依据或出错时代价高昂。我们部署了一个仅从封闭的已验证问答单元集合中进行回答的系统：它要么逐字返回检索到的单元，要么转路由至澄清、拒答或人工转接。该索引通过分阶段语言播种（SLS）在离线阶段进行增强：由人工为每个单元撰写基于真实场景的槽位模板，gpt-4.1-mini将其渲染为变体，再经轻量级人工审核进行过滤。同一套方法论在两个领域中复用，因此推理仅保持单次检索，无需查询时生成。在来自两个工业领域的留出查询变体上，SLS将混合检索R@1提升至0.881/0.930（+0.27/+0.34），且在测试的全部五种检索器上均获得收益。在相同的gpt-4.1-mini生成预算下，SLS比doc2query高出+0.20/+0.32，同时跨来源评估提供了（原文摘要在此处截断）

    arXiv:2609.00844v1 Announce Type: new  Abstract: Customer-service QA in an AI contact center (AICC) runs under deployment constraints that benchmark QA misses: tight voice-hotline latency and a high cost for unsupported or wrong automatic answers. We deploy a system that answers only from a closed set of verified QA units: it returns a retrieved unit verbatim, or routes to clarify, abstain, or handoff. The index is enriched offline by staged linguistic seeding (SLS): a human authors a per-unit world-grounded slot recipe, gpt-4.1-mini renders it into variants, and a light human gate filters them. One methodology is reused across both domains, so inference stays a single retrieval pass with no query-time generation. On held-out query variants from two industrial domains, SLS lifts hybrid R@1 to 0.881/0.930 (+0.27/+0.34), with gains across all five retrievers tested. At the same gpt-4.1-mini generation budget, SLS beats doc2query by +0.20/+0.32, while cross-provenance evaluation provides 
    
[^10]: Ctrl-F-Resist：监测极右翼线上活动的公民社会组织的实践、挑战与技术需求

    Ctrl-F-Resist. Practices, Challenges, and Technical Needs of Civil Society Organizations Monitoring the Far-Right Online

    [https://arxiv.org/abs/2609.00808](https://arxiv.org/abs/2609.00808)

    本文通过对12家德国公民社会组织的15名从业者进行定性研究，揭示了这些组织在极右翼在线监测工作中的长期实践、面临的挑战（法律不确定性、平台访问受限、资金不足）以及技术需求，强调它们是数字治理中被忽视的关键利益相关者。

    

    随着极右翼行为者日益利用在线平台传播意识形态并动员支持者，公民社会组织（CSOs）在监测网络上的反民主动态方面发挥着至关重要却未被充分认可的作用。与事实核查员或内容审核员不同，公民社会组织从事长期的、结合具体情境的分析工作，且往往在资源受限和条件不稳定的环境下开展。尽管具有重要的社会作用，公民社会组织在采用或共同开发技术解决方案方面面临重大障碍，包括法律上的不确定性、平台访问权限受限以及长期的资金不足。然而，现有研究和工具开发工作在很大程度上忽视了这些行为者，而更倾向于关注具有制度化背景的利益相关者。本文通过对来自12家德国公民社会组织的15名从事在线监测工作的从业者进行定性研究来填补这一空白，将这些组织定位为数字治理中关键却被忽视的利益相关者。

    arXiv:2609.00808v1 Announce Type: cross  Abstract: As far-right actors increasingly exploit online platforms to disseminate ideology and mobilize supporters, civil society organizations (CSOs) play a vital yet underrecognized role in monitoring antidemocratic dynamics online. Unlike fact-checkers or content moderators, CSOs engage in long-term, contextualized analysis, often in resource-constrained settings and under precarious conditions. Despite their critical societal role, CSOs face significant barriers to adopting or co-developing technical solutions, including legal uncertainty, limited platform access, and chronic underfunding. Existing research and tool development efforts have largely overlooked these actors in favor of more institutionally embedded stakeholders. This paper addresses this gap through a qualitative study with 15 practitioners from 12 Germany-based CSOs engaged in online monitoring, positioning them as key yet overlooked stakeholders in the governance of digital
    
[^11]: 从显著性到可判别性：面向VLM重排序器的保序视觉Token剪枝

    From Saliency to Discriminability: Rank-Preserving Visual Token Pruning for VLM Rerankers

    [https://arxiv.org/abs/2609.00667](https://arxiv.org/abs/2609.00667)

    提出无需训练的RaDiCal框架，利用归一化注意力熵判断显著性何时可信，并将其与排序判别先验融合，实现保序的视觉Token剪枝，从而高效加速VLM重排序器。

    

    作为列表级重排序器的大型视觉-语言模型必须联合处理每个查询中数十个候选对象的视觉Token，这使得Token剪枝对实际部署至关重要。现有的剪枝方法基于注意力显著性来保留Token，然而我们证明显著性系统性地与排序贡献存在错位：视觉上显著的Token往往捕获的是跨候选对象共享的、对顺序不敏感的模式。这种错位是层相关的：只有当注意力集中时显著性才变得有信息量，归一化的注意力熵可以诊断这种可靠性变化（皮尔逊相关系数r=0.87）。我们提出RaDiCal（排序判别性校准），一个无需训练的框架，利用归一化注意力熵来判定何时可以信任显著性，将其与一种无注意力的排序判别先验相融合，并从同一信任图中选择剪枝层。在三个检索基准测试和多个VLM架构上，RaDiCal……

    arXiv:2609.00667v1 Announce Type: new  Abstract: Large vision-language models used as listwise rerankers must jointly process visual tokens from tens of candidates per query, making token pruning essential for practical deployment. Existing pruning methods retain tokens by attention saliency, yet we show that saliency is systematically misaligned with ranking contribution: visually prominent tokens often capture order-neutral patterns shared across candidates. This mismatch is layer-dependent: saliency becomes informative only where attention is concentrated, and normalized attention entropy diagnoses the reliability shift (Pearson r=0.87). We propose RaDiCal (Rank-Discriminative Calibration), a training-free framework that uses normalized attention entropy to decide when saliency can be trusted, fusing it with an attention-free rank-discriminative prior and selecting pruning layers from the same trust landscape. Across three retrieval benchmarks and multiple VLM architectures, RaDiCal
    
[^12]: 匹配需要双方协作：基于强化学习协同进化的生成式检索器

    It Takes Two to Match: Co-Evolving Generative Retriever with Reinforcement Learning

    [https://arxiv.org/abs/2609.00638](https://arxiv.org/abs/2609.00638)

    提出 CoGR 框架，通过强化学习让大语言模型协同进化，在查询侧和物品侧同时生成对齐的关键词表示，并直接经倒排索引完成匹配，在兼容现有关键词检索基础设施的同时提升检索效果。

    

    检索是现代搜索与广告系统的第一阶段，它从庞大的物品集合中筛选出候选集，供下游的排序和竞价环节使用。近期的研究越来越多地利用大语言模型（LLM）通过查询扩展、数据合成和检索反馈训练来改进检索。然而，生成式组件通常仅用于查询侧的增强，而最终的匹配仍交由下游检索器完成。我们提出了 CoGR，一种转而训练大语言模型直接在查询侧和物品侧构建检索表示的检索框架。每个生成器产出一组紧凑的关键词集合，通过倒排索引直接进行匹配，从而保持与现有基于关键词的检索基础设施的兼容性。CoGR 采用两阶段训练流程：首先通过监督微调建立一个对齐的关键词空间，随后通过协同进化的强化学习交替优化（原文在此处截断）。

    arXiv:2609.00638v1 Announce Type: cross  Abstract: Retrieval is the first stage of modern search and advertising systems, selecting a candidate set from a large item universe for downstream ranking and auction. Recent work increasingly leverages LLMs to improve retrieval through query expansion, data synthesis, and retrieval-feedback training. However, the generative component is typically used for query-side augmentation, while final matching is still delegated to a downstream retriever. We introduce CoGR, a retrieval framework that instead trains LLMs to directly construct retrieval representations on both query and item sides. Each generator produces a compact set of keywords, which are matched directly through an inverted index, preserving compatibility with existing keyword-based retrieval infrastructure. CoGR uses a two-stage training pipeline. Supervised fine-tuning first establishes an aligned keyword space, after which co-evolving reinforcement learning alternately optimizes t
    
[^13]: 基于双节点蒙特卡洛树搜索的对话式推荐系统高效结构化上下文建模

    Towards Effective Structured Context Modeling for Conversational Recommender Systems via Dual-node Monte Carlo Tree Search

    [https://arxiv.org/abs/2609.00618](https://arxiv.org/abs/2609.00618)

    提出DREAMS框架，通过双节点树结构（引导节点用蒙特卡洛树搜索探索对话动作以推断潜在偏好，利用节点用大语言模型将偏好状态精炼为结构化检索查询），显式建模对话式推荐系统中用户偏好的多轮演化。

    

    我们研究了对话上下文建模在对话式推荐系统（CRS）用户偏好跟踪中的作用。为此，我们提出了DREAMS，这是一种新颖的树状结构上下文建模框架，能够显式地捕捉多轮交互过程中用户偏好的演化。DREAMS引入了两种专门的节点类型，以支持对话式推荐系统的两个基本目标：偏好引导与偏好利用。具体而言，引导节点利用蒙特卡洛树搜索（MCTS）策略性地探索对话动作并推断潜在的用户偏好，而利用节点则采用基于大语言模型（LLM）的精炼方法，将跟踪到的偏好状态转化为结构化的检索查询以用于推荐。在基准数据集上的大量实验证明了DREAMS及其设计的有效性。

    arXiv:2609.00618v1 Announce Type: cross  Abstract: We investigate the role of conversational context modeling in user preference tracking for Conversational Recommendation Systems (CRSs). In this regard, we propose DREAMS, a novel tree-structured context modeling framework that explicitly captures user preference evolution throughout multi-turn interactions. DREAMS introduces two specialized node types to support the two fundamental objectives of CRSs: preference elicitation and preference exploitation. Specifically, elicitation nodes leverage Monte Carlo Tree Search (MCTS) to strategically explore conversational actions and infer latent user preferences, while exploitation nodes employ LLM-based refinement to transform the tracked preference state into structured retrieval queries for recommendation. Extensive experiments on benchmark datasets demonstrate the effectiveness of DREAMS and its design.
    
[^14]: NeuroGraph：一种面向先进制造中可解释威胁推理的AI图驱动神经-符号框架

    NeuroGraph: An AI Graph-Driven Neuro-Symbolic Framework for Explainable Threat Reasoning in Advanced Manufacturing

    [https://arxiv.org/abs/2609.00604](https://arxiv.org/abs/2609.00604)

    本文提出NeuroGraph框架，通过融合本体感知符号查询生成、知识图谱检索与神经语言生成的双大语言模型神经-符号架构，解决现有方法的幻觉与多跳推理不足问题，实现先进制造IT/OT环境中准确且可解释的网络威胁推理。

    

    先进制造中网络物理攻击面日益复杂，使得网络威胁情报（CTI）分析变得越来越困难。尽管大语言模型和检索增强生成（RAG）改进了CTI工作流程，但基于文本的方法仍然容易出现幻觉问题，并且对互联威胁的结构化推理支持有限。基于图的RAG缓解了部分局限性，但现有方法往往缺乏本体一致的多跳推理能力，以及跨异构网络安全数据的透明证据追踪能力。本文提出了一种基于图的神经-符号框架，该框架集成了本体感知的符号查询生成、知识图谱检索和神经语言生成，以支持在信息技术（IT）和运营技术（OT）环境中进行准确且可解释的威胁分析。该框架采用双大语言模型架构：

    arXiv:2609.00604v1 Announce Type: cross  Abstract: The growing complexity of cyber-physical attack surfaces in advanced manufacturing has made cyber threat intelligence analysis increasingly difficult. Although large language models and retrieval-augmented generation have improved CTI workflows, text-based approaches remain vulnerable to hallucinations and provide limited support for structured reasoning over interconnected threats. Graph-based RAG reduces some of these limitations, but existing approaches often lack ontology-consistent multi-hop reasoning and transparent evidence tracing across heterogeneous cybersecurity data. This paper proposes a graph-grounded neuro-symbolic framework that integrates ontology-aware symbolic query generation, knowledge graph retrieval, and neural language generation to support accurate and explainable threat analysis across information technology and operational technology environments. The framework adopts a dual-large language model architecture:
    
[^15]: TRIS：一种抵御知识投毒的三层检索完整性筛

    TRIS: A Tri-Layer Retrieval Integrity Sieve Against Knowledge Poisoning

    [https://arxiv.org/abs/2609.00470](https://arxiv.org/abs/2609.00470)

    本文提出TRIS三层筛，一种中间件防御方案，通过跨嵌入空间聚类、触发器-载荷结构过滤和大模型一致性验证三重机制清洗RAG检索证据，利用投毒文档难以同时满足嵌入几何、内部结构和生成目标三重要求的固有弱点，有效抵御知识投毒攻击。

    

    检索增强生成（RAG）将大语言模型锚定在外部语料库之上，但对检索文档的隐式信任构成了一个关键的攻击面：PoisonedRAG研究表明，仅需少量精心构造的段落即可主导稠密检索，并将模型生成引向攻击者预设的答案。我们提出三层筛（Tri-Layer Sieve），这是一种中间件防御方案，通过以下三重机制清洗检索到的证据：借助独立裁判模型进行跨嵌入空间聚类、针对触发器-载荷伪迹的结构化过滤，以及大语言模型一致性验证。该设计利用了检索阶段投毒的一个关键弱点：单个投毒文档必须同时满足特定的嵌入几何结构、特定的内部触发器-载荷结构以及特定的生成目标——三者很难同时成立，即便面对通过改写来规避防御的自适应攻击者，这一脆弱性依然存在。在Natural Questions、HotpotQA和MS-MARCO数据集上使用Contriever检索（k=50），三层筛显著降低了……（摘要原文在此处截断）

    arXiv:2609.00470v1 Announce Type: new  Abstract: Retrieval-Augmented Generation (RAG) grounds large language models in external corpora, but implicit trust in retrieved documents creates a critical attack surface: PoisonedRAG shows that a handful of crafted passages can dominate dense retrieval and steer generation toward attacker-chosen answers. We present the Tri-Layer Sieve, a middleware defense that sanitizes retrieved evidence through cross-embedding-space clustering with an independent judge model, structural filtering of trigger-payload artifacts, and LLM consistency verification. The design exploits a key weakness of retrieval-stage poisoning: a single document must satisfy one embedding geometry, one internal Trigger-Payload structure, and one generation objective - rarely all three simultaneously, a fragility that persists even against an adaptive attacker who paraphrases around it. On Natural Questions, HotpotQA, and MS-MARCO with Contriever retrieval (k=50), the Sieve reduc
    
[^16]: 闭式解与合成孪生语料：基于嵌入统计量预测近似最近邻召回率

    Closed Forms and Synthetic Twins: Predicting Approximate Nearest Neighbor Recall from Embedding Statistics

    [https://arxiv.org/abs/2609.00364](https://arxiv.org/abs/2609.00364)

    本文证明无需构建真实索引，仅凭原始嵌入的无标签统计量——针对PQ、FDE等固定网格量化器的闭式矩统计量以及在合成孪生语料库上的模拟——就能在部署前预测近似最近邻索引的召回率。

    

    嵌入模型在训练和评估时都假定检索是精确的；但在生产环境中，它们运行在近似索引之后——如HNSW、IVF、乘积量化（PQ）或迟交互模型的固定维度编码（FDE）——这些索引的行为是编码器自身的基准测试从未见过的：一个现代编码器通过其原始FDE索引仅能召回其精确top-10结果的14%。此类失败只有在索引建立之后才会暴露，而标准的补救措施——如白化等需在语料库上拟合的变换——必须进行拟合、存储，并随语料库变化而重新拟合，还可能悄然改写编码器本应返回的结果。本文表明，索引行为在构建之前即可预测：仅需利用原始嵌入的无标签统计量，通过一系列与各索引家族所消费的信息相匹配的工具阶梯即可实现：(1) 针对固定网格量化器（PQ、FDE）的闭式矩统计量；(2) 在合成孪生语料库上进行模拟——即由聚类统计量生成的……（原文摘要在此处截断）

    arXiv:2609.00364v1 Announce Type: new  Abstract: Embedding models are trained and evaluated as if retrieval were exact; in production they serve behind approximate indexes -- HNSW, IVF, product quantization, or the fixed-dimensional encodings (FDEs) of late-interaction models -- whose behavior the encoder's benchmarks never see: one modern encoder recovers just 14% of its exact top-10 through its raw FDE index. Such failures surface only after an index is built, and the standard patches -- corpus-fitted transforms such as whitening -- must be fitted, stored, and refit as the corpus changes, and can silently rewrite what the encoder returns. This paper shows that index behavior is predictable before anything is built, from label-free statistics of the raw embeddings, through a ladder of instruments matched to what each index family consumes: (1) closed-form moment statistics for the fixed-grid quantizers (PQ, FDE); (2) simulation on a synthetic twin corpus -- cluster statistics made gen
    
[^17]: 真相之源：AI心理健康信息查询中引用来源的多平台、多语言审计

    Sources of Truth: A Multi-Platform, Multilingual Audit of Citations in AI Mental Health Information Queries

    [https://arxiv.org/abs/2609.00319](https://arxiv.org/abs/2609.00319)

    该研究对ChatGPT、Perplexity和Google AI Overview在多语言心理健康查询中生成的15,942条引用进行了系统审计，发现引用来源高度集中于少数域名，表明来源评估责任已从用户转移到AI平台。

    

    在线健康信息检索正从关键词搜索（用户需要浏览排序列表中的链接）转向对话式系统（由系统生成单一答案并整理其引用）。因此，来源评估的责任从用户转移到了平台，然而这些系统所呈现的来源特征仍未被充分刻画。我们对三款免费消费级产品（ChatGPT、Perplexity、Google AI Overview）在两种提示条件下针对二十个英文心理健康问题进行了审计，并将其中三个问题的子集翻译成六种资源水平各异的其他语言。我们在1,140条回复中记录了15,942条引用，涉及1,713个独立域名，随后采用经过人工编码验证的确定性分类器，依据九类组织类型学对所有引用进行分类。引用呈现出高度集中：被引最多的十个域名占英文引用总量的43.6%，政府、商业健康和学术来源紧随其后。

    arXiv:2609.00319v1 Announce Type: cross  Abstract: Online health information seeking is shifting from keyword search, where users consider a ranked list of links, to conversational systems that compose a single answer and curate its citations. Source evaluation therefore passes from user to platform, yet what these systems surface is poorly characterized. We audited three free consumer products (ChatGPT, Perplexity, Google AI Overview) on twenty English mental health questions under two prompt conditions, with a subset of three also translated into six further languages of varying resource tiers. We recorded 15,942 citations across 1,140 responses and 1,713 unique domains, then classified every citation with a nine-category organizational typology applied by a deterministic classifier validated against human coding. Citations were heavily concentrated: the ten most-cited domains accounted for 43.6% of English citations, and government, commercial health, and academic sources were close
    
[^18]: 面向具有非随机多模态评论反馈的序列推荐的双边状态空间模型

    Two-Sided State-Space Models for Sequential Recommendation with Non-Random Multimodal Review Feedback

    [https://arxiv.org/abs/2609.00165](https://arxiv.org/abs/2609.00165)

    提出双边状态空间模型TS-SSM，将非随机生成的多模态评论反馈同时融入用户状态与商品状态的动态演化过程，从而提升事件条件下的序列推荐效果。

    

    双边数字平台本质上是动态的：用户偏好会发生转移，商品热度会不断演变，而评论既反映也驱动着这些变化。然而，大多数序列推荐系统将评论视为更新用户状态的被动信号，忽视了两个重要方面。首先，评论的生成是非随机的，它取决于用户和商品双方不断演变的潜在状态。其次，评论能够重塑商品状态、在相关商品之间产生溢出效应，并影响用户未来的决策。为填补这些空白，我们提出了一种用于事件条件序列推荐的双边状态空间模型（TS-SSM）。TS-SSM由三个组件构成：（1）一个模态非随机缺失融合模块，用于编码评论内容以及蕴含信息的观测模式；（2）带有时间变化和局部图消息传递的用户状态演化机制，利用相关商品状态来细化用户偏好；（3）商品状态演化机制（摘要在此处截断）。

    arXiv:2609.00165v1 Announce Type: new  Abstract: Two-sided digital platforms are inherently dynamic: user preferences shift, item popularity evolves, and reviews both reflect and drive these changes. Yet most sequential recommendation systems treat reviews as passive signals for updating user states, leaving two aspects underexplored. First, review generation is nonrandom, depending on evolving latent states of both users and items. Second, reviews can reshape item states, induce spillover across related items, and influence future user decisions. To address these gaps, we propose a two-sided state-space model (TS-SSM) for event-conditioned sequential recommendation. TS-SSM consists of three components: (1) a modality-missing-not-at-random fusion module that encodes review content and informative observation patterns; (2) user-state evolution with temporal variation and local graph message passing that uses related item states to refine user preferences; and (3) item-state evolution wi
    
[^19]: SilentProbe：测量作为智能体工具的生产级API中的静默失败

    SilentProbe: Measuring Silent Failure in Production APIs Used as Agent Tools

    [https://arxiv.org/abs/2609.00035](https://arxiv.org/abs/2609.00035)

    该论文首次大规模测量了LLM智能体调用生产API时的“静默失败”现象，发现API模式中机器可校验的约束（而非供应商身份）是预测服务器能否诚实报错的关键因素，而当前OpenAPI文档中这类约束严重缺失。

    

    arXiv:2609.00035v1 公告类型：新论文 摘要：调用生产级API的大语言模型智能体无法区分“查询未匹配到任何结果”与“服务器未能理解查询”这两种情况：两者都返回HTTP 200和可解析的响应体，没有可捕获的异常，也没有可供分支判断的字段。我们研究了哪些因素能够预测发生的是哪一种情况，以及这对智能体造成的影响。通过对2,501份独立发布的OpenAPI文档中721,320个参数的审计，我们发现仅有7.5%的参数声明了枚举类型，15.2%声明了任何机器可校验的约束，而40.1%的文档在自然语言描述中至少陈述了一条其模式（schema）并未编码的约束。我们通过单一聚合层对来自27家供应商的实时商业端点执行了219次由模式导出的扰动测试，该聚合层为每次调用发布模式并返回运行标识符。结果表明，预测服务器“诚实性”的是约束的形式而非供应商身份：机器可校验的约束在111个案例中的全部111个都产生了诚实的错误报告，而仅有自然语言描述的约束……（原文摘要在此处截断）

    arXiv:2609.00035v1 Announce Type: new  Abstract: An LLM agent calling a production API cannot distinguish a query that matched nothing from a query the server did not understand. Both return HTTP 200 with a parsable body, no exception to catch and no field to branch on. We ask what predicts which one occurred, and what it does to the agent. Auditing 721,320 parameters across 2,501 independently published OpenAPI documents, we find that 7.5% declare an enumeration and 15.2% declare any machine-checkable constraint at all, while 40.1% of documents state at least one constraint in prose that their schema does not encode. Executing 219 schema-derived perturbations against live commercial endpoints from 27 vendors, reached through a single aggregation layer (Monid) that publishes a schema and returns a run identifier for every call, we find that constraint form, not vendor identity, predicts honesty: machine-checkable constraints yielded an honest error in 111 of 111 cases, prose-only const
    
[^20]: E-SENS：面向负约束检索的排斥敏感惩罚方法

    E-SENS: Exclusion-Sensitive Penalization for Negative-Constraint Retrieval

    [https://arxiv.org/abs/2608.30130](https://arxiv.org/abs/2608.30130)

    E-SENS是一种无需训练的重排序方法，通过为被排除概念提取“陷阱查询”并从检索分数中减去其相似度，有效惩罚与用户排除概念相关的文档，从而提升检索系统对负向约束的遵守能力。

    

    检索增强语言模型在检索器提供用户明确排除概念的相关证据时，可能无法遵守负向约束。除了显式否定之外，查询还可能要求答案包含一个概念而排除另一个概念，或者要求实体属于某一类别但与密切相关的实例不同。由于被排除的概念仍然出现在查询文本中，稠密检索器可能会对与该概念相关的文档赋予高相似度，即使用户明确要求避开它。我们提出了E-SENS，一种面向否定敏感检索的无训练重排序方法。E-SENS为被排除的一方提取一个紧凑的“陷阱查询”，并从原始查询的检索分数中减去陷阱查询的相似度。在ExcluIR基准上，E-SENS在四个嵌入模型上展现出清晰的召回率-违规权衡，并在保持召回率的设置下有效减少了陷阱检索。

    arXiv:2608.30130v1 Announce Type: cross  Abstract: Retrieval-augmented language models can fail to respect negative constraints when the retriever supplies evidence about concepts the user explicitly excluded. Beyond explicit negation, queries may ask for answers that include one concept while excluding another, or for entities that belong to a category but differ from a closely related instance. Because the excluded concept still appears in the query text, dense retrievers may assign high similarity to documents about that concept even when the user asks to avoid it. We introduce E-SENS, a training-free reranking method for negation-sensitive retrieval. E-SENS extracts a compact trap query for the excluded side and subtracts trap-query similarity from the original-query retrieval score. On ExcluIR, E-SENS shows a clear recall-violation trade-off across four embedding models and reduces trap retrieval at recall-preserving settings.
    
[^21]: REIGN：利用集成引导网络的翻新嵌入实现高效的上下文长度扩展

    REIGN: Refurbished Embeddings with Integrated Guidance Networks for Efficient Context-Length Scaling

    [https://arxiv.org/abs/2608.29899](https://arxiv.org/abs/2608.29899)

    REIGN通过在冻结引导网络生成的块嵌入序列上运行对比训练的双编码器，将词元级处理与文档级推理解耦，使长文档检索的训练成本相比分块Transformer微调降低约四个数量级。

    

    对长文档进行稠密检索的代价高昂。词元级编码器在序列长度上呈二次方扩展，而大多数长上下文嵌入模型只能通过架构上的变通方法或拉长十亿参数级大语言模型才能达到32K词元。我们提出REIGN（Refurbished Embeddings with Integrated Guidance Networks，集成引导网络的翻新嵌入），这是一个经过对比训练的双编码器，它在由冻结的引导网络（GN）生成的上下文化块嵌入序列上运行，而不是在原始词元上运行。REIGN针对多块输入，主要用于文档到文档的检索；单块输入则仍由GN处理。通过将词元级处理与文档级推理解耦，并将GN嵌入缓存到磁盘，相对于分块Transformer微调，每个文档的训练成本降低了大约四个数量级。我们还发布了一个合成的长文档检索基准，用于长上下文长度下的对比训练与评估。

    arXiv:2608.29899v1 Announce Type: cross  Abstract: Dense retrieval over long documents is expensive. Token-level encoders scale quadratically in sequence length, and most long-context embedding models reach 32K tokens only through architectural workarounds or by stretching billion-parameter LLMs. We propose REIGN (Refurbished Embeddings with Integrated Guidance Networks), a contrastively trained bi-encoder that operates on sequences of contextualised chunk embeddings from a frozen Guidance Network (GN) rather than on raw tokens. REIGN targets multi-chunk inputs, primarily for document-to-document retrieval; single-chunk inputs stay with the GN. Decoupling token-level processing from document-level reasoning, and caching the GN embeddings to disk, cuts per-document training cost by roughly four orders of magnitude relative to chunked Transformer fine-tuning. We also release a synthetic long-document retrieval benchmark for contrastive training and evaluation at long context lengths. Acr
    
[^22]: FKG.in的验证：LLM增强的印度食品知识中的健全性评估

    Validating FKG.in: Soundness Assessment in LLM-Augmented Indian Food Knowledge

    [https://arxiv.org/abs/2608.29249](https://arxiv.org/abs/2608.29249)

    本文作为印度食品知识图谱FKG.in的一部分，提出了一种半自动化的健全性评估工作流程，通过结合形式文法、词汇检查、统计启发式、Set Transformer连贯性建模和检索验证的多阶段方法，识别并解决LLM从非正式烹饪来源提取和增强结构化食谱数据时的常见失败模式。

    

    在线烹饪生态系统中，由大型语言模型（LLM）生成、修改或总结的食谱内容日益增多。虽然这些输出通常看似合理，但可能包含虚构的食材、被误述的用量或文化上不合常理的食材组合，从而限制了其在下游应用和知识图谱构建中的适用性。在本文中，我们提出了一种半自动化的健全性评估工作流程，用于验证由LLM从非正式烹饪来源中提取和增强的结构化食谱数据。该流程作为印度食品知识图谱FKG.in的一部分开发而成，通过结合形式文法、基于词汇的检查、统计启发式方法、基于Set Transformer的连贯性建模以及基于检索的验证等多阶段流程，识别并解决常见的失败模式，包括结构性不一致、语义和逻辑上的不连贯以及与源文本的偏差。

    arXiv:2608.29249v1 Announce Type: new  Abstract: The online culinary ecosystem is increasingly populated by recipe content generated, modified, or summarized by Large Language Models (LLMs). While often plausible, such outputs may contain hallucinated ingredients, misrepresented quantities, or culturally implausible combinations, limiting their suitability for downstream applications and knowledge graph construction. In this paper, we present a semi-automated soundness assessment workflow for validating structured recipe data extracted and augmented by LLMs from informal culinary sources. Developed as part of FKG.in, a knowledge graph of Indian food, the pipeline identifies and addresses common failure modes, including structural inconsistencies, semantic and logical incoherence, and deviations from the source text, through a multi-stage process combining formal grammars, vocabulary-based checks, statistical heuristics, Set Transformer-based coherence modeling, and retrieval-based veri
    
[^23]: RATIO：科学文献中跨类型构思操作检索的基准

    RATIO: A Benchmark for Retrieval Across Typed Ideation Operations in Scientific Literature

    [https://arxiv.org/abs/2608.27394](https://arxiv.org/abs/2608.27394)

    RATIO基准首次定义了三种科学构思操作（Address、Broaden、Specify）的检索任务，并利用远距离监督扩展到大规模语料库，为科学文献的灵感检索提供了新范式。

    

    arXiv:2608.27394v1 公告类型：新 摘要：检索到的科学文献可以为人与AI科学家提供灵感。灵感可以采取不同形式：先前的工作可能直接建议如何解决问题，或在不同抽象层次上指出方向——放大到更一般的视角或缩小到具体实现。我们引入RATIO（跨类型构思操作检索），这是一个大规模基准，其中相关性由三种操作定义，我们称之为构思动作：Address检索针对所提出问题的潜在方法，Broaden检索更一般的表述，Specify检索具体实例。RATIO是通过一种通用方法从CS文献中数百万篇全文科学论文构建而成，该方法将话语标记远距离监督——先前仅用于分类——扩展到语料库级检索，并结合了广泛的LLM和人工审核。实验表明，操作-

    arXiv:2608.27394v1 Announce Type: new  Abstract: Retrieved scientific literature can serve as inspiration for both human and AI scientists. Inspiration can take different forms: prior work may directly suggest how to address a problem, or surface directions at different levels of abstraction - zooming out to a more general view or zooming in to a concrete realization. We introduce RATIO (Retrieval Across Typed Ideation Operations), a large-scale benchmark in which relevance is defined by three operations which we name ideation moves: Address retrieves potential approaches for stated problems, Broaden retrieves more general formulations, and Specify retrieves concrete instantiations. RATIO is constructed from millions of full-text scientific papers across CS literature via a general recipe that extends discourse-marker distant supervision - previously used only for classification - to corpus-scale retrieval, combined with extensive LLM and human vetting. Experiments show that operation-
    
[^24]: DCA-MoE：用于人群计数的空间自适应跨层融合与密度路由专家

    DCA-MoE: Spatially Adaptive Cross-Layer Fusion and Density-Routed Experts for Crowd Counting

    [https://arxiv.org/abs/2608.15213](https://arxiv.org/abs/2608.15213)

    本文提出DCA-MoE框架，通过空间自适应层融合和密度路由多感受野专家，在冻结编码器下实现内容相关的人群计数解码，显著提升了对视角和尺度变化的适应性。

    

    人群计数必须在视角、头部尺度、遮挡和背景杂波严重变化的情况下恢复可靠的局部密度。尽管现代计数目标提供了强大的空间监督，但许多多层解码器仍使用空间不变的特征融合，并对每个位置应用单一的感受野模式。我们提出DCA-MoE，一个使这两个决策都依赖于内容的框架，同时保留冻结的DINOv3编码器。空间自适应层融合（SALF）预测四个对齐骨干特征上的位置级权重，密度路由多感受野专家（DR-MoE）为每个位置分配局部、中程和大范围上下文残差专家的软混合。EBC风格头部重建块密度，而DMCount监督和辅助路由平衡项训练解码器而不更新骨干网络。在NWPU-Crowd验证集上，最强的配对配置，基础...

    arXiv:2608.15213v1 Announce Type: cross  Abstract: Crowd counting must recover reliable local density under severe variations in perspective, head scale, occlusion, and background clutter. Although modern counting objectives provide strong spatial supervision, many multi-level decoders still use spatially invariant feature fusion and apply one receptive-field pattern to every location. We propose DCA-MoE, a framework that makes both decisions content dependent while retaining a frozen DINOv3 encoder. Spatially Adaptive Layer Fusion (SALF) predicts position-wise weights over four aligned backbone features, and Density-Routed Multi-Receptive-Field Experts (DR-MoE) assigns each location a soft mixture of local, mid-range, and large-context residual experts. An EBC-style head reconstructs block density, while DMCount supervision and an auxiliary routing-balance term train the decoder without updating the backbone. On the NWPU-Crowd validation split, the strongest paired configuration, base
    
[^25]: 字段感知的智能体技能检索

    Field-Aware Agent Skill Retrieval

    [https://arxiv.org/abs/2608.02880](https://arxiv.org/abs/2608.02880)

    该论文将智能体技能建模为结构化的多字段对象，对每个字段独立计算稀疏与稠密相似度并用均匀权重或小型MLP融合，从而显著提升终身学习智能体的技能检索效果。

    

    随着终身学习智能体不断积累日益增长的技能库，如何检索到正确的技能成为越来越重要的瓶颈。目前大多数技能检索方法将每个技能视为一个扁平文档，即通过拼接名称、描述和正文等字段来处理。然而，技能本质上是结构化的多字段对象，每个字段提供了关于技能何时以及如何被使用的不同信息。在这项工作中，我们研究了保留这种结构是否能改善技能检索。我们将每个技能表示为其独立的组成部分，并对每个字段独立计算稀疏与稠密相似度，从而获得一种天然张量化、字段感知的技能库表示。随后，我们通过均匀权重或一个小型可学习的MLP来组合这些字段级别的分数。在SkillRet和SRA-Bench两个不同的技能检索基准上，我们发现保持字段分离能够提升技能检索性能。

    arXiv:2608.02880v3 Announce Type: replace-cross  Abstract: As lifelong learning agents accumulate lifelong growing skill banks, retrieving the correct skill becomes an increasingly important bottleneck. Most current skill retrieval methods treat each skill as one flat document by concatenating fields such as the name, description, and body. However, skills are naturally structured, multi-field objects, where each field provides different information about when and how the skill should be used. In this work, we study whether preserving this structure improves skill retrieval. We represent each skill as its separate components, and compute sparse and dense similarities for each field independently, exposing a naturally tensorized, field-aware representation of the skill bank. We then combine these field-level scores either with uniform weights or with a small learned MLP. Across two different skill retrieval benchmarks, SkillRet and SRA-Bench, we find that keeping fields separate improve
    
[^26]: 弥合语义缓存中的运营差距

    Closing the Operational Gap in Semantic Caching

    [https://arxiv.org/abs/2606.19719](https://arxiv.org/abs/2606.19719)

    该论文指出PR-AUC指标会误导语义缓存系统的部署决策，提出了缓存感知的P-CHR AUC指标和运营保留率ORR，并将离线与部署质量间的运营差距分解为可恢复的阈值效用部分和由数据集正例率决定的不可约简结构部分。

    

    语义缓存通过为语义相似的查询提供缓存响应来降低大语言模型（LLM）的推理成本。标准做法是使用PR-AUC来评估这些系统，但该指标仅衡量分数的排序质量，而忽略了分数在固定阈值下是否可用。我们证明这种错位会导致系统性的糟糕部署选择，因为PR-AUC最高的模型在实际运行中往往表现最差。我们引入了精确率-缓存命中率（P-CHR）AUC这一缓存感知指标，用于衡量不同缓存利用率水平下的精确率；以及运营保留率（ORR），用于捕捉离线排序质量在部署时的保留程度。我们将离线质量与部署质量之间的运营差距分解为可恢复的阈值效用部分，以及由数据集正例率固定的不可约简的结构部分。我们的实验表明，阈值效用差距由训练目标决定，而非……（摘要原文在此处截断）

    arXiv:2606.19719v3 Announce Type: replace-cross  Abstract: Semantic caching cuts LLM inference costs by serving a cached response to semantically similar queries. Standard practice evaluates these systems using PR-AUC, a metric that only measures how well scores rank and ignores whether they are usable at a fixed threshold. We show this mismatch leads to systematically poor deployment choices, as models with the highest PR-AUC are often the worst in operation. We introduce Precision--Cache Hit Ratio (P-CHR) AUC, a cache-aware metric that measures precision across cache utilization levels, and Operational Retention Rate (ORR), which captures how much offline ranking quality survives at deployment. We decompose the operational gap between offline and deployed quality into a recoverable threshold-utility component and an irreducible structural component fixed by the dataset's positive rate. Our experiments show that the threshold-utility gap is governed by the training objective rather th
    
[^27]: APEX-EM：基于结构化程序性-情景经验回放的自主智能体非参数化在线学习

    APEX-EM: Non-Parametric Online Learning for Autonomous Agents via Structured Procedural-Episodic Experience Replay

    [https://arxiv.org/abs/2603.29093](https://arxiv.org/abs/2603.29093)

    APEX-EM提出了一种无需更新模型权重的非参数化经验记忆方法，通过程序性知识图谱存储完整的任务轨迹并同时索引成功与失败经验，使LLM智能体能够复用过往经验而无需重复推理，在相同底层模型对比下BigCodeBench迁移任务上提升7.6个百分点。

    

    LLM智能体在执行每个任务时都需要重新运行完整的推理过程，即使是它们刚刚解决过的任务也不例外。我们提出了APEX-EM，这是一种非参数化经验记忆，它将完整的程序性-情景轨迹存储在类型化的程序性知识图谱（PKG）中，并通过三种通道进行检索：语义搜索、针对抽象操作序列的结构签名匹配以及图遍历。一个“规划-检索-生成-迭代-摄取”（PRGII）工作流负责生成经验、进行质量把关并提交经验，同时对成功和失败的经验都进行索引，使智能体学会哪些内容可以复用、哪些应当避免。在部署期间不改变任何模型权重。我们在五个基准上进行评估：BigCodeBench、KGQAGen-10k、HLE、Lifelong Agent Bench和ALFWorld。由于先前的工作使用了不同的底层模型，我们的结论基于相同底层模型的对比，以保持模型能力固定。在使用共享GPT-4o底层模型的BigCodeBench留出集迁移测试中，APEX-EM获得了+7.6个百分点的提升。

    arXiv:2603.29093v3 Announce Type: replace-cross  Abstract: LLM agents rerun full reasoning for every task, even one they solved moments earlier. We introduce \textbf{APEX-EM}, a non-parametric experience memory that stores complete procedural-episodic traces in a typed Procedural Knowledge Graph (PKG) and retrieves them through three channels: semantic search, structural-signature matching over abstract operation sequences, and graph traversal. A Plan-Retrieve-Generate-Iterate-Ingest (PRGII) workflow produces, quality-gates, and commits experiences, indexing both successes and failures so the agent learns what to reuse and what to avoid. No weights change during deployment.   We evaluate on five benchmarks: BigCodeBench, KGQAGen-10k, HLE, Lifelong Agent Bench, and ALFWorld. Because prior work uses different backbones, we base our claims on same-backbone comparisons that hold model capability fixed. On held-out BigCodeBench transfer with a shared GPT-4o backbone, APEX-EM gains +7.6\,pp 
    
[^28]: TC-RAG：图灵完备的RAG在医疗大语言模型系统上的案例研究

    TC-RAG:Turing-Complete RAG's Case study on Medical LLM Systems

    [https://arxiv.org/abs/2408.09199](https://arxiv.org/abs/2408.09199)

    提出TC-RAG框架，通过引入图灵完备系统管理状态变量，并利用具备自适应检索、推理和规划能力的记忆栈系统，实现检索过程的受控终止并减少错误知识累积，从而显著提升医疗大语言模型系统知识检索的效率与准确性。

    

    在提升领域特定大语言模型（LLMs）能力的过程中，检索增强生成（RAG）成为一种有前景的解决方案，能够缓解幻觉、知识过时以及在高度专业化查询中专业知识有限等问题。然而，现有的RAG方法存在不足，因为它们忽略了系统状态变量，而这些变量对于实现自适应控制、检索终止以及系统收敛至关重要。在本文中，我们通过严格的证明提出了TC-RAG，这是一个新颖的框架，通过引入图灵完备系统来管理状态变量，从而应对上述挑战，实现更高效、更准确的知识检索。通过利用具备自适应检索、推理和规划能力的记忆栈系统，TC-RAG不仅确保了检索过程的受控终止，还通过Push和Pop操作减轻了错误知识的累积。

    arXiv:2408.09199v2 Announce Type: replace  Abstract: In the pursuit of enhancing domain-specific Large Language Models (LLMs), Retrieval-Augmented Generation (RAG) emerges as a promising solution to mitigate issues such as hallucinations, outdated knowledge, and limited expertise in highly specialized queries. However, existing approaches to RAG fall short by neglecting system state variables, which are crucial for ensuring adaptive control, retrieval halting, and system convergence. In this paper, we introduce the TC-RAG through rigorous proof, a novel framework that addresses these challenges by incorporating a Turing Complete System to manage state variables, thereby enabling more efficient and accurate knowledge retrieval. By leveraging a memory stack system with adaptive retrieval, reasoning, and planning capabilities, TC-RAG not only ensures the controlled halting of retrieval processes but also mitigates the accumulation of erroneous knowledge via Push and Pop actions. In the ca
    

