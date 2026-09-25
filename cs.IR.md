# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Return or Revise? Learning When Revision Helps Retrieval-Augmented QA](https://arxiv.org/abs/2609.30087) | 本文提出“可恢复性”指标——通过在同一评判标准下同时评估草稿答案与其候选修订所得到的成对效果——并训练模型在修订前预测该指标，从而判断何时进行检索增强修订有益，在多个实验设置下均优于仅基于草稿置信度的决策方法。 |
| [^2] | [Advancing Model Research in AgentX: Long-Horizon Autonomy for Industrial Recommender Systems](https://arxiv.org/abs/2609.30001) | AgentX-Model是一个面向工业推荐系统的双智能体自主研究框架，通过研究智能体与模型智能体的协作，围绕复现、跟进、组合和诊断四种行动实现长程自主的持续性模型研究。 |
| [^3] | [From Interests to Semantic IDs: Retrieval-Grounded Credit Assignment for Generative Recommendation](https://arxiv.org/abs/2609.29983) | 提出基于检索的查询归因方法，为每条推理轨迹分配可追溯的兴趣级信用，从而解决生成式推荐训练中稀疏精确匹配SID奖励导致的信用分配缺口问题。 |
| [^4] | [Learning Better Reasoning for Generative Recommendation with Semantic IDs](https://arxiv.org/abs/2609.29973) | 提出Evo-Rec三阶段框架，使生成式推荐系统能够筛选有效的推理轨迹并从自身生成中逐步学习更优的推理，从而避免低质量推理误导物品生成并提升推荐性能。 |
| [^5] | [An Empirical Study of VLM Pipelines for Long-Document QA](https://arxiv.org/abs/2609.29933) | 该研究在两个长文档问答基准上系统评估了VLM的部署选择，发现六工具智能体流水线只有在回答模型足够大时才能超越静态页面输入，且其优势随基准和阅读器的不同而变化。 |
| [^6] | [Fair Feed Ranking for Participatory Budgeting](https://arxiv.org/abs/2609.29819) | 本文针对参与式预算平台提出FairFeed公平信息流排序设计，通过提升曝光不足提案的可见度来均衡分配曝光机会，并在基于慕尼黑2025年参与式预算的模拟中验证其优于热门、最新等常见排序方式。 |
| [^7] | [LSF-SR: Latent Semantic Fusion for Sequential Recommendation via Flow-based Conditional Variational Autoencoders](https://arxiv.org/abs/2609.29815) | 提出LSF-SR框架，通过带归一化流的条件变分自编码器融合物品ID嵌入与大语言模型生成的语义信号，从而提升序列推荐的推荐质量。 |
| [^8] | [SEEK: Skill-Routed Evaluation with Evolvable Knowledge for Industrial Search](https://arxiv.org/abs/2609.29803) | SEEK 通过将搜索评估标准外化为可演化的技能库，并针对每个查询-结果对动态路由相关技能进行列表级评估，从而解决了工业搜索自动评估中标准间干扰和规则更新需昂贵重训练的问题。 |
| [^9] | [C3M: Cross-Session Multimodal Memory Maintenance for Long-Horizon Tasks](https://arxiv.org/abs/2609.29735) | C3M提出了一种跨会话多模态记忆维护框架，通过关系感知更新在有界活动索引中整合安全冗余并保留互补与不兼容记录，并在查询时以预算化路由展开关联源证据，为长程任务提供了紧凑且保留来源信息的记忆组织。 |
| [^10] | [SALI: Shot-Aware Late Interaction for Cross-Shot Relation Matching in Text-to-Video Retrieval using Film-Grammar Knowledge](https://arxiv.org/abs/2609.29721) | SALI提出了一种镜头感知的后期交互方法，将查询及其提取的主语和宾语的文本嵌入与视频各镜头的视觉嵌入进行跨镜头匹配，并结合电影语法知识进行微调，显著提升了文本到视频检索中多镜头人物关系查询的准确率。 |
| [^11] | [Stochastic Semantic Evidence Graphs: Uncertainty Propagation and Governance for Agentic AI](https://arxiv.org/abs/2609.29703) | 提出随机语义证据图（SSEG）框架，通过分层随机有向无环图对智能体AI中证据、检索、提示、生成及决策映射各环节的不确定性进行建模与传播，实现终端误差的逐路径界定、来源溯源的Fréchet界传播以及治理触发的诊断。 |
| [^12] | [EvLink: Source-Grounded Evidence Linking for Graph RAG](https://arxiv.org/abs/2609.29695) | EvLink通过在段落间构建基于显式来源关系的关系锚定证据链接与端点对齐链接，并结合有界广度优先搜索和证据需求挖掘的两阶段检索策略，解决了GraphRAG中图可达性不等于证据支持的问题，为多跳推理提供紧凑且非冗余的证据集合。 |
| [^13] | [Ingest-Time Fact Compilation for Cost-Efficient and Reliable Question Answering over Revised Corpora](https://arxiv.org/abs/2609.29661) | 提出摄入时事实编译架构，在数据摄入或变更时一次性将修订、删除、生效日期与来源信任规则解析并编译为带溯源的类型化事实记录，使查询时仅需低成本模型读取编译结果，从而在含修订的语料库上实现成本更低、更可靠的问答。 |
| [^14] | [SmallReason-ColBERT: An Ultra-Small Late-Interaction Retriever for Reasoning Intensive Retrieval](https://arxiv.org/abs/2609.29652) | 该论文提出了仅32M参数的超小型后期交互检索器SmallReason-ColBERT，通过变长度对比预热、难负例对比打磨和逐查询词元重要性头三项技术，使边缘规模模型在推理密集型检索任务上大幅缩小了与150M+参数推理调优模型的性能差距。 |
| [^15] | [OBLIQ-IR: Training a Dense Retriever for Oblique Queries](https://arxiv.org/abs/2609.29649) | 提出 OBLIQ-IR 单向量稠密检索器，通过针对各机制的合成查询以及从冻结作者编码器进行 kNN 图蒸馏这一新型跨模型监督，显著提升了检索由潜在属性决定相关性的“斜向查询”的能力。 |
| [^16] | [An Exploratory Ablation of a Small MLA--SSM Hybrid Language Model](https://arxiv.org/abs/2609.29618) | 该消融研究表明，在小MLA-SSM混合语言模型中，SSM分支对性能的贡献大于MLA分支，且密集FFN混合模型以更少的峰值训练内存达到了与三值MoE混合模型相当的困惑度表现。 |
| [^17] | [Anatomy of a Decision: Uncertainty-aware Hierarchical Intent Learning via Flow Matching for Multimodal Recommendation](https://arxiv.org/abs/2609.29609) | 提出UHIFlow框架，利用条件流匹配量化视觉与文本模态的不确定性，并以此不确定性引导生成分层意图结构，动态适应用户的决策确定性，从而提升多模态推荐效果。 |
| [^18] | [CodeGraph: Open-Taxonomy Knowledge Graph for Source Code with Wikidata Grounding](https://arxiv.org/abs/2609.29474) | 该论文提出CodeGraph流水线，利用代码专用大语言模型对源代码进行开放分类法语义标注，并通过三阶段实体链接过程将其锚定到Wikidata，从而构建出首个面向源代码的开放分类法知识图谱。 |
| [^19] | [A Systematic Multi-Domain Evaluation of Document Retrievers](https://arxiv.org/abs/2609.29455) | 该论文对三大家族的33个文档检索器在七个信息检索数据集上进行了大规模系统性实证评估，在统一配置与计算预算下综合分析了检索质量、运行时间和失败点。 |
| [^20] | [Decoupled Learning and Selection in Slate Recommendation for Privacy and Stability Under Noisy Scores](https://arxiv.org/abs/2609.29453) | 该论文将组合推荐解耦为随机评分学习与确定性选择两个阶段，阐明了差分隐私保证经选择器端到端传递的成立条件，并提出了一种能在评分噪声扰动下保证推荐列表顺序不变的间隔证书。 |
| [^21] | [Asymmetric Dynamic Routing: Balancing Reasoning Depth and Computational Efficiency in Hypergraph RAG](https://arxiv.org/abs/2609.29282) | 提出非对称动态路由（ADR）框架，利用轻量级分类器根据查询意图在三种非对称拓扑遍历策略间动态分发查询，从而在超图RAG中同时兼顾推理深度与计算效率。 |
| [^22] | [ASIRF: An Agentic Framework for Context-Dependent Sensitive Information Redaction](https://arxiv.org/abs/2609.29191) | 该论文提出ASIRF智能体框架，通过在推理时从知识库检索领域特定的敏感信息定义，无需重新训练即可适应新领域进行敏感信息脱敏，在85%的模型-领域组合中召回率超越了OpenAI隐私过滤器。 |
| [^23] | [ScalarLens: Numerical Embeddings with Stable Coordinates and Contextual Responses for CTR Prediction](https://arxiv.org/abs/2609.29182) | 提出ScalarLens数值嵌入方法，利用单调局部网格仅从标量本身构建稳定坐标，再通过有界低秩动态生成上下文响应，使同一数值在不同上下文中获得不同解释，从而改进CTR预测。 |
| [^24] | [X-Rec Technical Report](https://arxiv.org/abs/2609.29180) | 提出X-Rec，一种基于流匹配在连续物品嵌入空间中直接学习推荐分布的生成式推荐方法，通过锚点条件化等三项关键设计，同时克服了U2I方法兴趣多样性不足与SID-AR方法量化误差及低吞吐量的缺陷。 |
| [^25] | [Claim-Gated Source-Risk Auditing for Generative Search](https://arxiv.org/abs/2609.29145) | 该论文提出针对生成式搜索的声明门控来源风险审计契约：只有当关系证据、答案采纳、重要性和披露全部被观测到时才判定遗漏已解决，否则保持未解决状态，并通过可执行的参考检查器在穷举合成测试中验证了该契约的完备性。 |
| [^26] | [Seek: Self-Evaluative Exploration for Knowledge Retrieval](https://arxiv.org/abs/2609.28980) | 提出无需训练的Seek框架，通过测试时的迭代式语料库探索与自我评估式分级相关性反馈，弥补单次检索无法找回遗漏文档的缺陷，在BRIGHT基准上相对BM25提升82%并超越所有训练基线。 |
| [^27] | [Cross-Country Code-Mixing for Generative Recommendation](https://arxiv.org/abs/2609.28972) | 提出CMRec框架，借鉴多语言NLP中的语码转换思想，通过学习跨国共享语义码本并进行上下文感知的代码混合，在数据层面（而非仅参数层面）实现跨国生成式推荐的知识迁移。 |
| [^28] | [Reinforcement Learning with Verifiable Rewards for Small Search Agents](https://arxiv.org/abs/2609.28765) | 该研究首次证明，无需蒸馏，仅用GRPO和维基百科搜索工具在MuSiQue上训练0.8B参数的小模型，即可成功应用可验证奖励强化学习于开放域问答，在七个基准上取得未训练基线3.8倍的精确匹配率（0.352）。 |
| [^29] | [The Fellowship of the Query: Learning Retrieval Actions](https://arxiv.org/abs/2609.28653) | 通过轨迹微调可以让小型语言模型有效学会检索增强问答中的“下一步动作”控制决策，宏F1分数远超零样本提示，且单个SLM可同时兼任控制器与答案生成器。 |
| [^30] | [OneTrans-V2: Unifying Retrieval, Pre-rank, and Fine-rank with One Transformer in Industrial Recommender](https://arxiv.org/abs/2609.28589) | OneTrans-V2用一个共享Transformer统一工业推荐系统的召回、粗排和精排三个级联阶段，实现一次用户序列编码、联合训练、精排到粗排的模型内知识蒸馏以及稀疏MoE扩展。 |
| [^31] | [Repeated Queries Exhaust an LLM's Brand Recommendations but Not Its Sources](https://arxiv.org/abs/2609.05059) | 重复提问相同的购买问题时，不联网检索的大语言模型会不断涌现新的品牌推荐而难以饱和，启用检索的引擎则快速封闭品牌列表，但所有引擎引用的来源域名始终持续增加、远未收敛。 |
| [^32] | [Effective Graph and Rank-based Contextual Embeddings for Textual and Multimedia Data](https://arxiv.org/abs/2608.29001) | 本文提出RaDE方法，利用基于排名的信息和代表性节点子集选择来实现图嵌入的维度可解释性，在降低计算成本的同时改进文本与多媒体数据的检索任务。 |
| [^33] | [Who Owns the AI Recommendation? A Multi-Industry Empirical Map of Brand Category Ownership Across Large Language Models](https://arxiv.org/abs/2606.23057) | 该研究通过对五个行业50个品牌、250个查询在三个大语言模型上跨越两个月的大规模实证测量，首次绘制了AI推荐中品牌类别归属的图谱，发现品牌收录率较为均衡、推荐份额随时间高度稳定，并识别出7.6%的“竞争真空”查询。 |
| [^34] | [DeGRe: Dense-supervised Generative Reranking for Recommendation](https://arxiv.org/abs/2605.25749) | 该论文提出DeGRe框架，通过引入密集监督信号来解决生成式重排序中的启发式标签偏差和稀疏奖励导致的信用分配难题。 |
| [^35] | [RQ-Reg: A Residual-Quantization-Based Framework for Continuous Value Prediction in Recommender Systems](https://arxiv.org/abs/2602.23012) | 该论文提出RQ-Reg框架，通过残差量化将观看时长、GMV等连续目标值分解为从粗到细的量化编码序列并进行自回归预测，以克服传统回归方法在建模复杂长尾分布时欠拟合或牺牲泛化性的缺陷。 |
| [^36] | [SPARQL-LLM: Real-Time SPARQL Query Generation from Natural Language Questions](https://arxiv.org/abs/2512.14277) | SPARQL-LLM是一种开源、与三元组存储无关、由轻量级元数据驱动的方法，能够从自然语言实时生成SPARQL查询，兼顾准确性、运行时和成本等指标，从而实现生产环境的实际部署。 |
| [^37] | [WebArxiv: A Reproducible Benchmark for Evaluating Multimodal Web Agents on arXiv Tasks](https://arxiv.org/abs/2507.00938) | WebArxiv是一个基于arXiv静态快照构建的可复现基准，包含510个具有确定性答案的时间不变任务，用于评估多模态网络智能体在多约束论文检索、细粒度内容提取和跨论文比较等学术任务上的能力。 |

# 详细

[^1]: 返回还是修订？学习修订何时有助于检索增强问答

    Return or Revise? Learning When Revision Helps Retrieval-Augmented QA

    [https://arxiv.org/abs/2609.30087](https://arxiv.org/abs/2609.30087)

    本文提出“可恢复性”指标——通过在同一评判标准下同时评估草稿答案与其候选修订所得到的成对效果——并训练模型在修订前预测该指标，从而判断何时进行检索增强修订有益，在多个实验设置下均优于仅基于草稿置信度的决策方法。

    

    我们考虑这样一个决策问题：在答案修订系统中，是直接返回已有的草稿答案，还是利用检索到的证据对其进行修订。草稿置信度估计的是当前答案是否正确，但这一决策需要估计某次特定修订所带来的效果。为了进行离线训练与评估，我们在同一正确性评判标准下对返回的草稿答案及其候选修订同时进行评分，这使得修复效果、潜在损害以及与最优答案之间的差距变得可观测。我们将这种成对效应称为“可恢复性”（recoverability），并训练策略在修订之前对其进行预测。在三个修订设置下共 25,870 个留出的开放域问题上，基于成对结果训练的评分器在全部九个 Llama 设置-随机种子组合中，其“准确率-修订率”曲线下面积均优于同等条件的草稿正确性评分器，并且在由开发集选定的阈值下平均提升 0.23–0.68 个准确率百分点，该差异仅在不同训练运行之间具有统计显著性。

    arXiv:2609.30087v1 Announce Type: new  Abstract: We consider the decision of whether to return an existing draft answer or revise it using retrieved evidence, as in answer-revision systems. Draft confidence estimates whether the current answer is correct, but the decision requires estimating the effect of a specified revision. For offline training and evaluation, we grade both the returned draft and its candidate revision under the same correctness judge, which makes repair, harm, and the gap to an oracle observable. We call this paired effect its recoverability, and we train policies to predict it before revision. On 25,870 held-out open-domain questions across three revision setups, a scorer trained on the paired outcome has greater area under the accuracy--revision-rate curve than a matched draft-correctness scorer in all nine Llama setup--seed fits, and gains 0.23--0.68 accuracy points on average at development-selected thresholds, a difference significant across training runs only
    
[^2]: AgentX中推进模型研究：面向工业推荐系统的长程自主性

    Advancing Model Research in AgentX: Long-Horizon Autonomy for Industrial Recommender Systems

    [https://arxiv.org/abs/2609.30001](https://arxiv.org/abs/2609.30001)

    AgentX-Model是一个面向工业推荐系统的双智能体自主研究框架，通过研究智能体与模型智能体的协作，围绕复现、跟进、组合和诊断四种行动实现长程自主的持续性模型研究。

    

    持续的工业推荐研究需要利用一次实验的结果来决定下一步的研究方向。我们提出了AgentX-Model，这是AgentX模型研究框架的下一代版本，它在由业务输入和预测任务定义的沙盒环境中将提案开发与模型实验连接起来。AgentX-Model采用由研究智能体和模型智能体组成的双智能体架构。研究智能体从论文和实验发现中制定经过独立评审的提案，而模型智能体进行多轮调查并返回代码、测量结果和未解决的问题。利用返回的结果，研究智能体选择一个起始实现并制定下一个研究问题，使后续实验能够建立在先前发现的基础上。我们将这种持续研究围绕四种行动来组织：复现、跟进、组合和诊断。

    arXiv:2609.30001v1 Announce Type: new  Abstract: Sustaining industrial recommendation research requires using the results of one experiment to decide what to investigate next. We present AgentX-Model, the next generation of AgentX's model research framework, which connects proposal development and model experimentation within sandboxes defined by business inputs and prediction tasks. AgentX-Model adopts a dual-agent architecture comprising a Research Agent and a Model Agent. The Research Agent develops independently reviewed proposals from papers and experimental findings, while the Model Agent conducts multi-round investigations and returns code, measurements, and unresolved questions. Using the returned results, the Research Agent selects a starting implementation and formulates the next research question, allowing subsequent experiments to build on earlier findings. We organize this continuing research around four actions: Reproduce, Follow-up, Composition, and Diagnose. The first t
    
[^3]: 从兴趣到语义ID：面向生成式推荐的基于检索的信用分配

    From Interests to Semantic IDs: Retrieval-Grounded Credit Assignment for Generative Recommendation

    [https://arxiv.org/abs/2609.29983](https://arxiv.org/abs/2609.29983)

    提出基于检索的查询归因方法，为每条推理轨迹分配可追溯的兴趣级信用，从而解决生成式推荐训练中稀疏精确匹配SID奖励导致的信用分配缺口问题。

    

    语义ID（SIDs）将目录中的每个物品编码为一段简短的token序列，使生成式推荐器能够以自回归方式预测下一个物品。推理增强的变体（一种日益普遍的扩展形式）会先生成一段文本推理轨迹，再通过束搜索解码出下一个物品的SID。这类推荐器通常在精确匹配SID奖励下采用组相对策略优化进行训练，而这种奖励在大规模目录中非常稀疏。由此产生两种失败模式：当一组rollout全部未命中目标时，该组的优势为零，不产生任何学习信号；而共享相同SID奖励的rollout无论其推理轨迹差异多大，都会获得完全相同的优势。在这两种情况下，奖励仅反映解码出的SID，而从不反映产生该SID的推理过程，从而造成信用分配缺口。我们通过基于检索的查询归因来弥合这一缺口：每条轨迹被结构化为一个历史摘要、一组兴趣假设以及……

    arXiv:2609.29983v1 Announce Type: cross  Abstract: Semantic IDs (SIDs) encode each catalog item as a short token sequence, enabling generative recommenders to predict the next item autoregressively. Reasoning-enhanced variants, an increasingly common extension, first generate a textual trace and then decode a next-item SID by beam search. Such recommenders are commonly trained with group-relative policy optimization under an exact-match SID reward, which is sparse in large catalogs. Two failure modes follow. When all rollouts in a group miss the target, the group yields zero advantage and no learning signal. Rollouts sharing the same SID reward receive identical advantages, however much their traces differ. In both cases the reward reflects only the decoded SID, never the reasoning that produced it. This creates a credit-assignment gap.   We address this gap with retrieval-grounded query attribution. Each trace is structured into a history summary, a set of interest hypotheses, and a f
    
[^4]: 基于语义ID学习生成式推荐的更优推理

    Learning Better Reasoning for Generative Recommendation with Semantic IDs

    [https://arxiv.org/abs/2609.29973](https://arxiv.org/abs/2609.29973)

    提出Evo-Rec三阶段框架，使生成式推荐系统能够筛选有效的推理轨迹并从自身生成中逐步学习更优的推理，从而避免低质量推理误导物品生成并提升推荐性能。

    

    生成式推荐将物品检索重新表述为序列生成任务，使统一模型能够直接从用户的交互历史中生成下一个物品。语义ID通过将每个物品表示为离散编码，使语义相关的物品之间能够共享知识，从而让这一范式变得高效且可扩展。最近的研究在语义ID生成之前引入显式推理，帮助模型总结用户兴趣并推断可能的偏好转变。然而，推理并不天然有益：不准确或缺乏信息量的推理可能会误导后续的物品生成，最终降低推荐性能。这引出了一个核心挑战：推荐系统如何选择并学习有效的推理轨迹，并从自身生成的内容中逐步演进出更好的推理？在这项工作中，我们提出了Evo-Rec，一个用于学习更优推理并进一步提升推荐性能的三阶段框架。

    arXiv:2609.29973v1 Announce Type: cross  Abstract: Generative recommendation reformulates item retrieval as sequence generation, allowing a unified model to directly generate the next item from a user's interaction history. Semantic IDs further make this paradigm effective and scalable by representing each item as discrete codes, enabling knowledge sharing among semantically related items. Recent studies introduce explicit reasoning before Semantic-ID generation, helping models summarize user interests and infer possible preference transitions. However, reasoning is not inherently beneficial: Inaccurate or uninformative reasoning may mislead subsequent item generation and ultimately degrade recommendation performance. This raises a central challenge: how can a recommender select and learn effective reasoning traces and progressively evolve toward better reasoning from its own generations? In this work, we propose Evo-Rec, a three-stage framework for learning better reasoning and furthe
    
[^5]: 面向长文档问答的视觉语言模型流水线实证研究

    An Empirical Study of VLM Pipelines for Long-Document QA

    [https://arxiv.org/abs/2609.29933](https://arxiv.org/abs/2609.29933)

    该研究在两个长文档问答基准上系统评估了VLM的部署选择，发现六工具智能体流水线只有在回答模型足够大时才能超越静态页面输入，且其优势随基准和阅读器的不同而变化。

    

    视觉语言模型（VLM）越来越多地被用于长文档处理，其输入将文本与图表、表格、插图和复杂版式结合在一起。部署这类模型意味着需要做出多项选择：如何将文档提供给模型、当仅发送部分页面时应使用哪种检索器，以及让模型以智能体方式运行还是作为静态流水线运行。我们在两个长文档问答基准上，使用前沿API模型和开源权重VLM研究了这些选择。首先，在MMLongBench-Doc上，我们提出的包含页面、表格、插图和搜索调用的六工具智能体只有在回答用VLM足够大时才能体现价值：使用Qwen3.5-4B和9B时它落后于静态页面输入，使用Qwen3.5-27B时持平，而使用Sonnet 4.5时则领先。在LongDocURL上，它在所有阅读器下均与静态输入持平或更优。它相对最强静态流水线的优势在MMLongBench-Doc上以前沿阅读器最为明显，而在LongDocURL上则缩小至噪声范围内。其次，检索……（摘要原文在此处截断）

    arXiv:2609.29933v1 Announce Type: cross  Abstract: Vision-Language Models (VLMs) are increasingly used for long-document processing, where the inputs combine text with charts, tables, figures, and complex layouts. Deploying them means choosing how to feed the document to the model, which retriever to use when only a subset of pages is sent, and whether to run the model agentically or as a static pipeline. We study these choices on two long-document QA benchmarks with both frontier API and open-weight VLMs. First, on MMLongBench-Doc our six-tool agent with page, table, figure, and search calls pays off only once the answering VLM is large enough: with Qwen3.5-4B and 9B it trails static page input, with Qwen3.5-27B it draws level, and with Sonnet 4.5 it leads. On LongDocURL it is level with or ahead of static input at every reader. Its lead over the strongest static pipeline is clearest with the frontier reader on MMLongBench-Doc and narrows to within noise on LongDocURL. Second, retriev
    
[^6]: 参与式预算中的公平信息流排序

    Fair Feed Ranking for Participatory Budgeting

    [https://arxiv.org/abs/2609.29819](https://arxiv.org/abs/2609.29819)

    本文针对参与式预算平台提出FairFeed公平信息流排序设计，通过提升曝光不足提案的可见度来均衡分配曝光机会，并在基于慕尼黑2025年参与式预算的模拟中验证其优于热门、最新等常见排序方式。

    

    在大规模参与式预算中，公民无法查看全部提案池，因此提案展示的顺序实际上构成了一种议程设置的权力。我们主张应将公平曝光视为一项民主设计目标。我们研究了被广泛部署的开源数字民主平台 Consul Democracy，并指出其提案信息流通常按热度、发布时间或评论活跃度进行排序。基于这一诊断，我们提出了 FairFeed——一种面向参与式预算的信息流排序设计，它利用透明声明的用户偏好，提升曝光不足提案的可见度，并引入一个限流的否决通道以支持众包式审查。我们在以慕尼黑2025年参与式预算流程为原型的模拟环境中对该设计进行评估，并与随机排序、最新排序和最多评论排序的信息流进行比较。在该模拟中，FairFeed 拓宽了提案被发现的可能性，使可见度在合格提案池中分布更为均衡，并增加了跨立场的支持……

    arXiv:2609.29819v1 Announce Type: cross  Abstract: In large-scale participatory budgeting, citizens cannot inspect the full proposal pool, so the order in which proposals are shown becomes a form of agenda-setting power. We argue that fair exposure should therefore be treated as a democratic-design goal. We study Consul Democracy, a widely deployed open-source digital-democracy platform, and show that its proposal feeds are typically ordered by popularity, recency, or comment activity. Building on this diagnosis, we propose FairFeed, a feed-ranking design for PB that uses transparently declared preferences, boosts under-exposed proposals, and admits a rate-limited reject channel for crowd-sourced vetting. We evaluate the design in a simulation anchored in Munich's 2025 PB process and compare it with random, newest, and most-commented feeds. In this simulation, FairFeed broadens proposal discovery, distributes visibility more evenly across the eligible pool, increases cross-cutting supp
    
[^7]: LSF-SR：基于流条件变分自编码器的序列推荐潜在语义融合

    LSF-SR: Latent Semantic Fusion for Sequential Recommendation via Flow-based Conditional Variational Autoencoders

    [https://arxiv.org/abs/2609.29815](https://arxiv.org/abs/2609.29815)

    提出LSF-SR框架，通过带归一化流的条件变分自编码器融合物品ID嵌入与大语言模型生成的语义信号，从而提升序列推荐的推荐质量。

    

    序列推荐旨在从用户的历史交互中预测其未来兴趣。尽管大语言模型（LLM）能够捕捉丰富的物品语义，但现有方法往往难以将协同信号与文本语义知识进行对齐。因此，所学到的物品表示无法同时捕捉两种信号的互补优势，导致推荐质量欠佳。为解决这一局限，我们提出了LSF-SR（基于流条件变分自编码器的序列推荐潜在语义融合），这是一个利用带归一化流的条件变分自编码器（CVAE）来融合物品ID嵌入和LLM生成语义信号的新型框架。LSF-SR的核心是一个由平面流或径向流增强的条件融合模块，该模块学习一个灵活的潜在空间，使具有相似语义特征的物品在其中聚集（摘要至此截断）。

    arXiv:2609.29815v1 Announce Type: new  Abstract: Sequential recommendation aims to predict users' future interests from their historical interactions. Although Large Language Models (LLMs) capture rich item semantics, existing methods often struggle to align collaborative signals with textual semantic knowledge. As a result, the learned item representations fail to capture the complementary strengths of both signals, leading to suboptimal recommendation quality. To address this limitation, we propose Latent Semantic Fusion for Sequential Recommendation via Flow-based Conditional Variational Autoencoders (LSF-SR), a novel framework that uses a Conditional Variational Autoencoder (CVAE) with Normalizing Flows to fuse item ID embeddings and LLM-generated semantic signals. At the core of LSF-SR is a conditional fusion module augmented with planar or radial flows. This module learns a flexible latent space that encourages items with similar semantic profiles to cluster together within the l
    
[^8]: SEEK：面向工业搜索的技能路由评估与可演化知识

    SEEK: Skill-Routed Evaluation with Evolvable Knowledge for Industrial Search

    [https://arxiv.org/abs/2609.29803](https://arxiv.org/abs/2609.29803)

    SEEK 通过将搜索评估标准外化为可演化的技能库，并针对每个查询-结果对动态路由相关技能进行列表级评估，从而解决了工业搜索自动评估中标准间干扰和规则更新需昂贵重训练的问题。

    

    搜索质量评估为工业搜索系统的开发与迭代提供了至关重要的监督和诊断信号。尽管大型语言模型（LLM）为人工评估提供了一种可扩展的替代方案，但可靠的自动评估仍然面临挑战：用户是在页面级别体验搜索结果的，而适用的评估标准是多维的且持续演化的。将所有评估标准打包进一个统一的提示词中会引入无关上下文和潜在的标准间干扰，而通过后训练将标准内化到模型中，则会使规则更新与代价高昂的模型重新训练周期紧密耦合。为解决这些问题，我们提出了技能路由评估与可演化知识框架（SEEK）。具体而言，SEEK 将特定的搜索评估标准外化为技能库，针对每个查询-结果列表对动态路由相关技能，并采用任务适配的列表级评估方式（摘要在此处截断）。

    arXiv:2609.29803v1 Announce Type: cross  Abstract: Search quality evaluation provides essential supervision and diagnostic signals for the development and iteration of industrial search systems. Although large language models (LLMs) offer a scalable alternative to manual assessment, reliable automatic evaluation remains challenging: users experience search results at the page level, while the applicable evaluation criteria are multi-dimensional and continuously evolving. Packing all evaluation criteria into a unified prompt introduces irrelevant context and potential criterion interference, whereas internalizing them through post-training tightly couples rule updates with costly model retraining cycles.   To address these issues, we propose Skill-routed Evaluation with Evolvable Knowledge (SEEK). Specifically, SEEK externalizes specific search evaluation criteria into a skill bank, dynamically routes relevant skills for each query-result list pair, and employs a task-adapted listwise e
    
[^9]: C3M：面向长程任务的跨会话多模态记忆维护

    C3M: Cross-Session Multimodal Memory Maintenance for Long-Horizon Tasks

    [https://arxiv.org/abs/2609.29735](https://arxiv.org/abs/2609.29735)

    C3M提出了一种跨会话多模态记忆维护框架，通过关系感知更新在有界活动索引中整合安全冗余并保留互补与不兼容记录，并在查询时以预算化路由展开关联源证据，为长程任务提供了紧凑且保留来源信息的记忆组织。

    

    长程任务需要在有限的、与查询无关的记忆预算下保存并在之后恢复跨会话证据。现有的压缩方法可能会丢弃细粒度的视觉线索，或会将语义相似但不兼容的观察混淆在一起。我们提出了C3M，这是一种跨会话多模态记忆组织方式，它在持久的源文本-图像证据之上维护一个有界的活动索引。基于关系的更新在整合安全冗余的同时，保留互补和不兼容的记录。在查询时，预算化的路由机制选择有用的索引页面，并在固定的读取器预算下展开其关联的源证据。这些机制共同为跨会话长程任务建立了一种紧凑的、保留来源信息的多模态记忆组织，保留了可靠下游推理所需的时间区分和源链接。代码可在 https://github.com/HuzhouNLP/C3M 获取。

    arXiv:2609.29735v1 Announce Type: new  Abstract: Long-horizon tasks require preserving and later recovering cross-session evidence under a bounded, query-blind memory budget. Existing compression can discard fine-grained visual cues or conflate semantically similar but incompatible observations. We present C3M, a cross-session multimodal memory organization that maintains a bounded active index over persistent source text-image evidence. Relation-aware updates consolidate safe redundancy while preserving complementary and incompatible records. At query time, budgeted routing selects useful index pages and expands their associated source evidence under a fixed reader budget. Together, these mechanisms establish a compact, provenance-preserving multimodal memory organization for cross-session long-horizon tasks, retaining temporal distinctions and source links required for reliable downstream reasoning. Code is available at https://github.com/HuzhouNLP/C3M.
    
[^10]: SALI：基于电影语法知识的镜头感知后期交互方法，用于文本到视频检索中的跨镜头关系匹配

    SALI: Shot-Aware Late Interaction for Cross-Shot Relation Matching in Text-to-Video Retrieval using Film-Grammar Knowledge

    [https://arxiv.org/abs/2609.29721](https://arxiv.org/abs/2609.29721)

    SALI提出了一种镜头感知的后期交互方法，将查询及其提取的主语和宾语的文本嵌入与视频各镜头的视觉嵌入进行跨镜头匹配，并结合电影语法知识进行微调，显著提升了文本到视频检索中多镜头人物关系查询的准确率。

    

    文本到视频检索通常使用单个嵌入向量来表示视频片段。这种嵌入常常丢失人物之间的重要关系。例如，“安娜与马克对峙”这样的互动通常以两人正反打镜头交替的方式拍摄（图1a）。无论是任何单个镜头，还是对片段内所有镜头取平均的嵌入，都无法捕捉这种关系。因此，我们提出了SALI（镜头感知后期交互）。该方法从单句查询中提取主语和宾语，并将查询、其主语和宾语的文本嵌入分别与视频片段的每个视觉镜头嵌入进行匹配。匹配算子采用贪婪最大值或最优传输。微调过程中的电影语法惩罚项带来了一个微小且一致的偏移。基于CLIP4Clip-meanP构建的SALI在Condensed Movies和ActivityNet数据集上保持了与原有方法相当的整体召回率，同时在多镜头关系查询上将R@1分别提升了3分和12分，在所有对比方法中提升幅度最大，并在MSR-VTT上也改善了此类查询的表现，其代价是……

    arXiv:2609.29721v1 Announce Type: cross  Abstract: Text-to-video retrieval usually represents a video clip by a single embedding. This embedding often loses important relations between people. E.g., an interaction "Anna confronts Mark" is regularly filmed as alternating shot and reverse shot of both (Fig. 1a). No single shot or averaged embedding over clip shots captures this relation. Thus, we propose SALI (Shot-Aware Late Interaction). It extracts the subject and object from a single-sentence query, and matches the query, its subject and object text embeddings against each visual shot embedding of a video clip. The matching operator is greedy max or optimal transport. A film-grammar penalty in fine-tuning adds a small, consistent shift. Built on CLIP4Clip-meanP, SALI keeps overall recall on par on Condensed Movies and ActivityNet while raising R@1 on multi-shot relation queries by 3 and 12 points, the most among all compared methods, and improves such queries on MSR-VTT at a cost of 
    
[^11]: 随机语义证据图：面向智能体AI的不确定性传播与治理

    Stochastic Semantic Evidence Graphs: Uncertainty Propagation and Governance for Agentic AI

    [https://arxiv.org/abs/2609.29703](https://arxiv.org/abs/2609.29703)

    提出随机语义证据图（SSEG）框架，通过分层随机有向无环图对智能体AI中证据、检索、提示、生成及决策映射各环节的不确定性进行建模与传播，实现终端误差的逐路径界定、来源溯源的Fréchet界传播以及治理触发的诊断。

    

    AI智能体评估通常只检查最终答案，但误差可能通过证据、检索、提示、生成或决策映射等环节引入。我们提出随机语义证据图（SSEG），这是一种分层随机有向无环图（DAG），其语言节点可扩展为自回归的词量子图，其可观测输出可以是完整短语上的概率分布。语义约简与校准均为可选操作。我们定义了图相对的局部缺陷与下游边影响，推导了终端误差的逐路径上界，并利用其逐节点分量来诊断治理触发条件。在来源溯源方面，该图保留了不确定的论断—段落关系，并传播精确的Fréchet界，而非假设各来源之间相互独立。在三种开放权重架构上，信息等价的变化会实质性改变完整短语的概率分布。一项受控实验在5,000个案例中未出现证书违规；交叉RAG与l…（摘要原文在此处截断）

    arXiv:2609.29703v1 Announce Type: new  Abstract: AI-agent evaluations usually inspect a final answer, yet error may enter through evidence, retrieval, prompting, generation or decision mapping. We introduce a stochastic semantic evidence graph (SSEG), a hierarchical stochastic DAG whose language node expands into an autoregressive token subgraph and whose observable output may be a law over complete phrases. Semantic reduction and calibration are optional. We define graph-relative local defects and downstream edge influences, derive a pathwise bound on terminal error and use its nodewise terms to diagnose governance triggers. For source provenance, the graph preserves uncertain claim--passage relations and propagates sharp Fr\'echet bounds rather than assuming independence across sources. Across three open-weight architectures, information-equivalent changes materially alter complete-phrase laws. A controlled experiment yields no certificate violations in 5,000 cases; crossed-RAG and l
    
[^12]: EvLink：面向图检索增强生成的基于来源锚定的证据链接

    EvLink: Source-Grounded Evidence Linking for Graph RAG

    [https://arxiv.org/abs/2609.29695](https://arxiv.org/abs/2609.29695)

    EvLink通过在段落间构建基于显式来源关系的关系锚定证据链接与端点对齐链接，并结合有界广度优先搜索和证据需求挖掘的两阶段检索策略，解决了GraphRAG中图可达性不等于证据支持的问题，为多跳推理提供紧凑且非冗余的证据集合。

    

    基于图的检索增强生成通过将语料库组织成结构化图来支持多跳推理。然而，图的可达性往往捕捉的是语义关联而非证据支持，因此一个可达的段落仍可能无法为所需的跨段落转换提供依据。我们提出了EvLink，这是一种证据链接检索器，它将段落保留为可检索的证据单元，并在段落之间构建有证据支持的转换。EvLink构建了两种类型的可靠链接：由显式来源关系支撑的关系锚定证据链接，以及作为来源受限后备的端点对齐链接。在检索方面，我们引入了两阶段检索策略：首先，在来源锚定的证据链接上进行有界广度优先搜索，以恢复基于相似度的方法所遗漏的桥接段落；然后，通过结合噪声或（noisy-OR）覆盖度细化的证据需求挖掘，选择紧凑且非冗余的证据集合。

    arXiv:2609.29695v1 Announce Type: new  Abstract: Graph-based Retrieval-Augmented Generation (GraphRAG) supports multi-hop reasoning by organizing corpora into structured graphs. However, graph reachability often captures semantic association rather than evidence support, so a reachable passage may still fail to justify a required cross-passage transition. We propose EvLink, an evidence-linking retriever that preserves passages as retrievable evidence units and builds evidence-supported transitions between them. EvLink constructs two types of reliable links: relation-grounded evidence links justified by explicit source relations, and endpoint-alignment links serving as sourcebounded fallbacks. For retrieval, we introduce a two-stage retrieval strategy. First, bounded breadth-first search over source-grounded evidence links recovers bridge passages missed by similarity-based methods. Then, evidenceneed mining with noisy-OR coverage refinement selects a compact, non-redundant evidence set
    
[^13]: 面向修订语料库的低成本可靠问答：摄入时事实编译方法

    Ingest-Time Fact Compilation for Cost-Efficient and Reliable Question Answering over Revised Corpora

    [https://arxiv.org/abs/2609.29661](https://arxiv.org/abs/2609.29661)

    提出摄入时事实编译架构，在数据摄入或变更时一次性将修订、删除、生效日期与来源信任规则解析并编译为带溯源的类型化事实记录，使查询时仅需低成本模型读取编译结果，从而在含修订的语料库上实现成本更低、更可靠的问答。

    

    大多数智能体式问答系统在最糟糕的时间完成了一部分重要的语义工作：即每次有人提问的时候。当语料库包含修订、草稿、撤销、删除以及具有不同权威级别的来源时，模型必须在每次读取时重建受治理的当前状态——然后丢弃这些工作，并在下一次查询时重复这一过程。这有点像一个数据库在每次有人读取时都重新构建物化视图。我们提出了“摄入时事实编译”这一架构，它在语料库数据被摄入或发生变更时完成这项工作：原始段落被改写为自包含的事实；管理修订、删除、生效日期和来源可信度的规则只解析一次；得到的状态以携带来源与修订溯源信息的类型化记录形式存储。在查询时，一个低成本的模型只需读取编译后的记录，而无需从噪声……（原文摘要到此被截断）

    arXiv:2609.29661v1 Announce Type: new  Abstract: Most agentic question answering (QA) systems do an important part of their semantic work at the worst possible time: every time someone asks a question. When a corpus contains revisions, drafts, revocations, deletions, and sources with different levels of authority, the model must reconstruct the governed current state on every read - then throw that work away and repeat it on the next query. This is a bit like a database that rebuilds a materialized view every time someone reads from it. We present ingest-time fact compilation, an architecture that performs this work when corpus data is ingested or changed. Raw passages are rephrased into self-contained facts; rules governing revisions, deletions, effective dates, and source trust are resolved once; and the resulting state is stored as typed records carrying source and revision provenance. At query time, an inexpensive model reads the compiled record instead of reconstructing it from no
    
[^14]: SmallReason-ColBERT：一种面向推理密集型检索的超小型后期交互检索器

    SmallReason-ColBERT: An Ultra-Small Late-Interaction Retriever for Reasoning Intensive Retrieval

    [https://arxiv.org/abs/2609.29652](https://arxiv.org/abs/2609.29652)

    该论文提出了仅32M参数的超小型后期交互检索器SmallReason-ColBERT，通过变长度对比预热、难负例对比打磨和逐查询词元重要性头三项技术，使边缘规模模型在推理密集型检索任务上大幅缩小了与150M+参数推理调优模型的性能差距。

    

    推理密集型检索对小型模型而言仍然十分困难。现有的紧凑型公开ColBERT模型通常在通用语料上训练，在BRIGHT基准上比经过推理调优的150M+参数基线模型低出数个nDCG@10点。然而，目前尚不存在边缘规模的公开推理调优ColBERT模型。我们提出了SmallReason-ColBERT，这是一个32M参数的后期交互检索器，通过三个组件大幅缩小了这一差距：在ReasonIR-VL数据上进行变长度对比预热、在合并的ReasonIR-HQ与BGE-Reasoner数据上进行难负例对比打磨，以及在冻结的基础模型之上训练的单层逐查询词元重要性头。该头采用非归一化加权MaxSim分数进行训练，并用其长度归一化形式进行评估。在受控重训练实验中，若将此训练目标替换为对称归一化分数，损失会陷入停滞，并损失3.59的nDCG@10。完整方案达到了21.41的平均分（原文摘要在此处截断）。

    arXiv:2609.29652v1 Announce Type: new  Abstract: Reasoning-intensive retrieval remains difficult for small models. Compact public ColBERTs are usually trained on general-purpose corpora and underperform reasoning-tuned 150M+ baselines on BRIGHT~\cite{bright} by several nDCG@10 points. However, no public reasoning-tuned ColBERT exists at edge scale. We introduce \textbf{SmallReason-ColBERT}, a 32M late-interaction retriever that closes much of this gap with three components: a varied-length contrastive warmup on ReasonIR-VL, a hard-negative contrastive polish on merged ReasonIR-HQ and BGE-Reasoner data, and a single-layer per-query-token importance head trained on top of the frozen base. The head is trained with an un-normalised weighted MaxSim score and evaluated with its length-normalised form. In a controlled re-training, replacing this training objective with the symmetric normalised score causes the loss to stall and costs $3.59$ nDCG@10. The full recipe reaches \textbf{21.41} mean
    
[^15]: OBLIQ-IR：训练面向斜向查询的稠密检索器

    OBLIQ-IR: Training a Dense Retriever for Oblique Queries

    [https://arxiv.org/abs/2609.29649](https://arxiv.org/abs/2609.29649)

    提出 OBLIQ-IR 单向量稠密检索器，通过针对各机制的合成查询以及从冻结作者编码器进行 kNN 图蒸馏这一新型跨模型监督，显著提升了检索由潜在属性决定相关性的“斜向查询”的能力。

    

    以 OBLIQ-Bench 为代表的斜向检索要求检索器找到那些相关性由某种潜在属性决定的文档——例如隐含立场、类比推理技巧、作者写作指纹或模糊的“话到嘴边”式回忆——而这些属性在文档表面几乎没有或完全没有显式表达。最先进的稠密编码器以及围绕前沿语言模型构建的智能体搜索流水线在这些任务上存在巨大的第一阶段瓶颈，而同样的语言模型在看到候选文档时却能可靠地验证相关性。为此我们提出 OBLIQ-IR，这是一个单向量稠密检索器，其训练混合了针对每种机制的合成查询与一种新形式的跨模型监督：从冻结的作者风格编码器进行 kNN 图蒸馏，从而将“风格重于主题”的归纳偏置迁移到学生模型中。经过微调的 3B 检索器在 Writing-Style 上达到 0.211 NDCG@10，在 Math 上达到 0.171，在 Twitter 上达到 0.177……

    arXiv:2609.29649v1 Announce Type: new  Abstract: Oblique retrieval, as exemplified by OBLIQ-Bench, asks a retriever to find documents whose relevance is determined by a latent attribute (an implicit stance, an analogous reasoning technique, an authorial fingerprint, or a vague tip-of-the-tongue recollection) that has little or no surface expression in the document. State-of-the-art dense encoders and agentic search pipelines built around frontier language models exhibit a large first-stage bottleneck on these tasks, while the same language models reliably verify relevance when shown candidates. We address this with OBLIQ-IR, a single-vector dense retriever whose training mixture combines per-mechanism synthetic queries with a new form of cross-model supervision: kNN-graph distillation from a frozen authorship encoder, which transfers a style-versus-topic inductive bias into the student. A 3B retriever fine-tuned reaches 0.211 NDCG@10 on Writing-Style, 0.171 on Math, 0.177 on Twitter, a
    
[^16]: 一个小型MLA-SSM混合语言模型的探索性消融研究

    An Exploratory Ablation of a Small MLA--SSM Hybrid Language Model

    [https://arxiv.org/abs/2609.29618](https://arxiv.org/abs/2609.29618)

    该消融研究表明，在小MLA-SSM混合语言模型中，SSM分支对性能的贡献大于MLA分支，且密集FFN混合模型以更少的峰值训练内存达到了与三值MoE混合模型相当的困惑度表现。

    

    我们报告了对TALH（Adaptive Latent Hybrid，自适应潜在混合模型）的一项探索性单种子消融实验。TALH是一个仅有解码器的语言模型，结合了并行的多头潜在注意力机制与自定义的循环状态空间分支。五个变体（每token估计活跃参数量在1.17亿至2.17亿之间）在FineWeb样本上从零开始训练，采用相同的优化步数和token数量。在这一特定设置下，移除SSM分支会导致验证困惑度的最大退化（仅MLA模型PPL为315），而移除MLA的影响则小得多（仅SSM模型PPL为239）。密集FFN混合模型取得了231的PPL，而测试的top-2三值MoE混合模型为240，但后者可节省3.87 GB的峰值训练内存。我们还保留了一项初步的Apple M3计时观察：在五个未优化的实现中，仅MLA模型在512至2,048个提示token范围内的首token生成时间曲线最为平坦，尽管密集Transformer的速度要快得多……

    arXiv:2609.29618v1 Announce Type: cross  Abstract: We report an exploratory, single-seed ablation of TALH (Adaptive Latent Hybrid), a decoder-only language model with parallel Multi-head Latent Attention (MLA) and a custom recurrent state-space (SSM) branch. Five variants, spanning 117--217M estimated active parameters per token, are trained from scratch on a FineWeb sample for the same number of optimisation steps and tokens. In this specific setup, removing the SSM branch gives the largest degradation in validation perplexity (MLA-only PPL 315), whereas removing MLA has a much smaller effect (SSM-only PPL 239). A dense-FFN hybrid obtains PPL 231, compared with 240 for the tested top-2 ternary-MoE hybrid, while using 3.87 GB less peak training memory. We also preserve a preliminary Apple M3 timing observation: among the five unoptimised implementations, MLA-only has the flattest measured time-to-first-token curve from 512 to 2,048 prompt tokens, although the dense Transformer is much 
    
[^17]: 解剖一个决策：通过流匹配进行不确定性感知的分层意图学习以实现多模态推荐

    Anatomy of a Decision: Uncertainty-aware Hierarchical Intent Learning via Flow Matching for Multimodal Recommendation

    [https://arxiv.org/abs/2609.29609](https://arxiv.org/abs/2609.29609)

    提出UHIFlow框架，利用条件流匹配量化视觉与文本模态的不确定性，并以此不确定性引导生成分层意图结构，动态适应用户的决策确定性，从而提升多模态推荐效果。

    

    arXiv:2609.29609v1 公告类型：新论文 摘要：建模潜在的用户意图对推荐系统至关重要，但现有方法难以应对用户兴趣固有的不确定性及其动态、分层的特性。当前方法通常依赖聚类或原型学习来发现一组静态的意图。然而，它们面临两个关键挑战：（1）忽视了多模态特征中固有的不确定性；（2）假设了一个静态且扁平的意图结构，无法适应用户不同的决策确定性。为解决这些局限性，我们提出了UHIFlow，一个基于流匹配的不确定性感知分层意图学习框架。首先，我们的跨模态不确定性协同建模（CUSM）模块利用条件流匹配来量化来自视觉和文本模态的不确定性，并对它们进行协同对齐。随后，不确定性引导的分层意图生成（UHIG）模块使用这种量化后的不确定性……

    arXiv:2609.29609v1 Announce Type: new  Abstract: Modeling the underlying user intent is crucial for recommendation, but existing methods struggle with the inherent uncertainty and the dynamic, hierarchical nature of user interests. Current approaches often rely on clustering or prototype learning to discover a static set of intents. However, they face two critical challenges: (1) they overlook the uncertainty inherent in multimodal features; and (2) they assume a static and flat intent structure, failing to adapt to a user's varying decision certainty. To address these limitations, we propose UHIFlow, an Uncertainty-aware Hierarchical Intent learning framework via Flow matching. First, our Cross-modal Uncertainty Synergistic Modeling (CUSM) module leverages conditional flow matching to quantify uncertainty from visual and textual modalities and synergistically align them. Subsequently, the Uncertainty-guided Hierarchical Intent Generation (UHIG) module uses this quantified uncertainty 
    
[^18]: CodeGraph：基于Wikidata实体锚定的开放分类法源代码知识图谱

    CodeGraph: Open-Taxonomy Knowledge Graph for Source Code with Wikidata Grounding

    [https://arxiv.org/abs/2609.29474](https://arxiv.org/abs/2609.29474)

    该论文提出CodeGraph流水线，利用代码专用大语言模型对源代码进行开放分类法语义标注，并通过三阶段实体链接过程将其锚定到Wikidata，从而构建出首个面向源代码的开放分类法知识图谱。

    

    GitHub和Software Heritage Archive等公共软件仓库存储了数十亿个文件，然而提取其中隐含的工程知识——即它们所实现的算法、所遵循的编程范式、所实例化的设计模式以及所服务的应用领域——仍然极具挑战性，因为现有的工具仅局限于句法和词法层面的分析。我们提出了一条利用代码专用大型语言模型构建源代码开放分类法语义标注的流水线。提取出的实体通过一个三阶段链接过程锚定到Wikidata：确定性的SPARQL阶段处理无歧义实体，深度研究智能体解析剩余的长尾实体，层级汇总阶段导入每个已解析Wikidata标识符的父级闭包。最终所得的标注被物化为一个面向源代码的开放分类法知识图谱。我们进一步引入了一个校准……

    arXiv:2609.29474v1 Announce Type: cross  Abstract: Public software repositories, like GitHub and Software Heritage Archive, store billions of files, yet extracting their implicit engineering knowledge ---i.e., the algorithms they implement, the paradigms they follow, the patterns they instantiate, and the application domains they serve--- remains challenging, as current tools are constrained to syntactic and token-level analysis. We present a pipeline for building an open-taxonomy semantic annotation of source code using a code-specialised Large Language Model. The extracted entities are grounded in Wikidata through a three-stage linking procedure: a deterministic SPARQL stage handles unambiguous entities, a Deep Research Agent resolves the residual long tail, and a hierarchy-rollup stage imports the parent-of closure of each resolved Wikidata identifier. The resulting annotations are materialised as a source-code-specific open-taxonomy knowledge graph. We further introduce a calibrate
    
[^19]: 文档检索器的系统性多领域评估

    A Systematic Multi-Domain Evaluation of Document Retrievers

    [https://arxiv.org/abs/2609.29455](https://arxiv.org/abs/2609.29455)

    该论文对三大家族的33个文档检索器在七个信息检索数据集上进行了大规模系统性实证评估，在统一配置与计算预算下综合分析了检索质量、运行时间和失败点。

    

    文档检索是许多现代AI系统的关键组成部分，直接影响其在下游任务中的有效性、鲁棒性和公平性。尽管近年来检索器的数量不断增长，但文献中的比较研究通常范围有限，或仅聚焦于单一的基准、领域或模型家族。这种碎片化使得人们难以就文档检索器的相对优势、劣势和权衡得出可靠的结论。为填补这一空白，我们对文档检索器进行了大规模的实证评估，涵盖三大家族（稀疏、稠密和基于扩展的），在七个信息检索数据集上评估了33个检索器，分析了检索质量、运行时间和失败点。我们没有对每个模型进行单独调优，而是以开箱即用的方式评估每个检索器，采用可根据其公开文档重建的配置以及统一的计算预算。

    arXiv:2609.29455v1 Announce Type: new  Abstract: Document retrieval is a crucial component of many modern AI systems, directly influencing their effectiveness, robustness, and fairness in downstream tasks. While recent years have seen a growing number of retrievers, comparative studies in the literature are typically limited in scope or focused on singular benchmarks, domains, or model families. This fragmentation makes it difficult to draw reliable conclusions about the relative strengths, weaknesses, and trade-offs of document retrievers. To address this gap, we conduct a large-scale empirical evaluation of document retrievers, covering three families (sparse, dense, and expansion-based) and evaluating 33 retrievers across seven IR datasets, analyzing retrieval quality, runtime, and failure points. Rather than tuning each model individually, we evaluate every retriever off the shelf, under the configuration reconstructable from its public documentation and a uniform compute budget. O
    
[^20]: 噪声评分下面向隐私与稳定性的组合推荐中学习与选择解耦方法

    Decoupled Learning and Selection in Slate Recommendation for Privacy and Stability Under Noisy Scores

    [https://arxiv.org/abs/2609.29453](https://arxiv.org/abs/2609.29453)

    该论文将组合推荐解耦为随机评分学习与确定性选择两个阶段，阐明了差分隐私保证经选择器端到端传递的成立条件，并提出了一种能在评分噪声扰动下保证推荐列表顺序不变的间隔证书。

    

    我们将组合推荐形式化为一个随机化的评分学习器之后接一个确定性的选择器。首先，通过后处理机制，适当界定范围的差分隐私保证可以传递到选择过程及其审计轨迹中。端到端的隐私保证仅在以下情况下成立：选择器的输入为公开或独立的信息、为先前已输出的私有数据、或已单独进行过隐私核算；若固定原始状态或候选信息，则只能获得条件性的隐私保证。其次，我们推导出一个基于日志的间隔证书：当评分扰动引起的目标函数变动为有界且小于最小贪心决策间隔的一半时，即可保证有序的组合推荐列表保持不变。受控的固定间隔测试显示出近线性的指数缩放特性，经验斜率为-0.220（95%置信区间为[-0.231, -0.210]），对比独立噪声下的参考斜率-1/4。在OULAD、MovieLens-25M和Amazon Musical Instruments数据集上的真实锚点实验表明，更大的锚点权重能够降低由评分噪声引起的……

    arXiv:2609.29453v1 Announce Type: new  Abstract: We formalize slate recommendation as a randomized score learner followed by deterministic selection. First, an appropriately scoped differential-privacy guarantee passes through selection and its audit trace by post-processing. End-to-end privacy holds only when selector inputs are public or independent, previous private outputs, or separately privacy-accounted; fixing raw state or candidate information instead yields only a conditional guarantee. Second, we derive a logged margin certificate: bounded score-induced objective movement below half the smallest greedy decision margin guarantees that the ordered slate is unchanged.   Controlled fixed-margin tests show near-linear exponent scaling, with an empirical slope of $-0.220$ (95% CI $[-0.231,-0.210]$) against the independent-noise reference $-1/4$. Real-anchor experiments on OULAD, MovieLens-25M, and Amazon Musical Instruments show that greater anchor weight reduces score-noise-induce
    
[^21]: 非对称动态路由：在超图RAG中平衡推理深度与计算效率

    Asymmetric Dynamic Routing: Balancing Reasoning Depth and Computational Efficiency in Hypergraph RAG

    [https://arxiv.org/abs/2609.29282](https://arxiv.org/abs/2609.29282)

    提出非对称动态路由（ADR）框架，利用轻量级分类器根据查询意图在三种非对称拓扑遍历策略间动态分发查询，从而在超图RAG中同时兼顾推理深度与计算效率。

    

    尽管基于图和超图的检索增强生成（RAG）显著缓解了大语言模型（LLM）的幻觉问题，但现有的基于结构的RAG系统通常采用静态遍历策略，而不考虑查询的复杂度。我们将这种“静态检索谬误”识别为导致简单查询产生计算冗余以及复杂推理任务出现认知上下文缺口的主要根源。为了平衡推理质量与推理效率，我们提出了非对称动态路由（ADR），这是一个在分层知识图谱上运行的意图条件检索框架。ADR采用轻量级结构化分类器，在三种非对称拓扑遍历算子之间动态分发查询：局部事实锚定、自下而上的邻接扩散和自上而下的洞察落地，这些算子共同实现了分层知识层之间的双向信息流。大量的……

    arXiv:2609.29282v1 Announce Type: new  Abstract: While graph-based and hypergraph-based Retrieval-Augmented Generation (RAG) significantly mitigate hallucinations in Large Language Models (LLMs), existing structure-based RAG systems typically adopt static traversal strategies regardless of the query complexity. We identify this ``static retrieval fallacy'' as a primary source of computational redundancy for simple queries and cognitive context gaps for complex reasoning tasks. To balance reasoning quality and inference efficiency, we propose Asymmetric Dynamic Routing (ADR), an intent-conditioned retrieval framework operating over hierarchical knowledge graphs. ADR employs a lightweight structured classifier to dynamically dispatch queries among three asymmetric topological traversal operators: localized fact anchoring, bottom-up adjacency diffusion, and top-down insight grounding, which collectively enable bidirectional information flow across hierarchical knowledge layers. Extensive 
    
[^22]: ASIRF：一个面向上下文相关敏感信息脱敏的智能体框架

    ASIRF: An Agentic Framework for Context-Dependent Sensitive Information Redaction

    [https://arxiv.org/abs/2609.29191](https://arxiv.org/abs/2609.29191)

    该论文提出ASIRF智能体框架，通过在推理时从知识库检索领域特定的敏感信息定义，无需重新训练即可适应新领域进行敏感信息脱敏，在85%的模型-领域组合中召回率超越了OpenAI隐私过滤器。

    

    敏感信息是由领域和意图定义的，而非一个通用类别，然而诸如隐私过滤器和命名实体识别器等脱敏系统在训练时便固定了分类体系，导致每进入一个新领域都需要重新训练。我们提出了ASIRF（智能体敏感信息脱敏框架），它在推理时根据输入所属领域从灵活的知识库中检索领域特定的定义，无需重新训练即可适应新领域。我们评估了两种架构——三次调用的多智能体流水线和单智能体变体——涵盖十个小型开源权重模型和八个数据集（包括分布外的虚构领域），并以OpenAI隐私过滤器（OPF）作为基于训练分类器的基线。每个领域仅需数十条专家撰写的定义且不使用任何训练数据，ASIRF在80个模型-领域组合中的68个（85%）上，召回率至少由两种架构之一超过了OPF

    arXiv:2609.29191v1 Announce Type: new  Abstract: Sensitive information is defined by domain and intent, not a universal category, yet redaction systems such as privacy filters and named-entity recognizers fix a taxonomy at training time, requiring retraining for each new domain. We introduce ASIRF (Agentic Sensitive Information Redaction Framework), which retrieves domain-specific definitions based on the input's domain from a flexible knowledge base at inference time, needing no retraining to adapt. Two architectures, a three-call multi-agent pipeline and a single-agent variant, are evaluated across ten small open-weight models and eight datasets, including out-of-distribution fictional domains, against the OpenAI Privacy Filter (OPF) as a trained-classifier baseline. With only a few dozen expert-authored definitions per domain and no training data, ASIRF's recall exceeds OPF's in 68 of 80 model-domain combinations (85 percent), by at least one of the two architectures, with shortfall
    
[^23]: ScalarLens：用于CTR预测的具有稳定坐标与上下文响应的数值嵌入

    ScalarLens: Numerical Embeddings with Stable Coordinates and Contextual Responses for CTR Prediction

    [https://arxiv.org/abs/2609.29182](https://arxiv.org/abs/2609.29182)

    提出ScalarLens数值嵌入方法，利用单调局部网格仅从标量本身构建稳定坐标，再通过有界低秩动态生成上下文响应，使同一数值在不同上下文中获得不同解释，从而改进CTR预测。

    

    用于点击率（CTR）预测的数值嵌入建立在一个方便但具有限制性的前提之上：一个标量只有一个表示。这一前提混淆了数值所处的位置与其对当前样本的含义。在Criteo验证集上，即使在去除加性主效应之后，同一个数值区间在不同类别与数值上下文中携带的残余点击证据仍具有相反的符号。生产流水线进一步加剧了这种不匹配，因为外部归一化的特征需要在训练与推理服务之间保持变换和统计信息的同步。我们提出了ScalarLens，这是一种数值嵌入方法，它在保留数值本身是什么的同时，适配该数值应如何被解释。单调局部网格仅根据焦点标量本身构建一个稳定坐标；随后，有界低秩动态在移动该坐标、替换类别词元以及改变CTR骨干网络的情况下，生成上下文响应。

    arXiv:2609.29182v1 Announce Type: new  Abstract: Numerical embeddings for click-through rate (CTR) prediction are built on a convenient but restrictive premise: a scalar has one representation. This premise conflates where a value lies with what it means for the current sample. On the Criteo validation split, the same numerical interval carries residual click evidence with opposite signs across categorical and numerical contexts, even after additive main effects are removed. Production pipelines compound this mismatch because externally normalized features require transformations and statistics to remain synchronized between training and serving. We introduce ScalarLens, a numerical embedding that preserves what a value is while adapting how it should be interpreted. A monotone local mesh constructs a stable coordinate from the focal scalar alone; bounded low-rank dynamics then produce a contextual response without moving that coordinate or replacing categorical tokens and the CTR back
    
[^24]: X-Rec 技术报告

    X-Rec Technical Report

    [https://arxiv.org/abs/2609.29180](https://arxiv.org/abs/2609.29180)

    提出X-Rec，一种基于流匹配在连续物品嵌入空间中直接学习推荐分布的生成式推荐方法，通过锚点条件化等三项关键设计，同时克服了U2I方法兴趣多样性不足与SID-AR方法量化误差及低吞吐量的缺陷。

    

    生成式建模的最新进展通过将推荐形式化为“下一物品生成”问题，重塑了推荐系统领域。现有的检索方法主要遵循两种范式：用户到物品方法使用一个或少数确定性嵌入来表示用户上下文，这限制了其捕捉多样化和多模式兴趣的能力；而基于语义ID的自回归（SID-AR）方法虽然能够建模更具表达力的分布，但存在量化误差和顺序解码吞吐量低的问题。为了解决这些局限性，我们提出了X-Rec，通过流匹配直接在连续的物品嵌入空间中学习推荐分布，并生成嵌入触发器用于近似最近邻检索。X-Rec 包含三项关键设计，使这一方法形式化既有效又高效。首先，我们引入锚点条件化，将生成过程分解为粗粒度语义区域……（原文摘要在此处截断）

    arXiv:2609.29180v1 Announce Type: new  Abstract: Recent advances in generative modeling have reshaped recommender systems by formulating recommendation as a next-item generation problem. Existing retrieval approaches primarily follow two paradigms: user-to-item (U2I) methods represent user context using one or a few deterministic embeddings, which limits the ability to capture diverse and multi-mode interests, while semantic-ID-based autoregressive (SID-AR) methods model more expressive distributions but suffer from quantization errors and the low throughput of sequential decoding. To address these limitations, we propose X-Rec to directly learn the recommendation distribution in the continuous item embedding space through flow matching and generate embedding triggers for approximate nearest neighbor retrieval. X-Rec incorporates three key designs to make this formulation effective and efficient. First, we introduce anchor conditioning to decompose generation into coarse semantic-regio
    
[^25]: 面向生成式搜索的声明门控式来源风险审计

    Claim-Gated Source-Risk Auditing for Generative Search

    [https://arxiv.org/abs/2609.29145](https://arxiv.org/abs/2609.29145)

    该论文提出针对生成式搜索的声明门控来源风险审计契约：只有当关系证据、答案采纳、重要性和披露全部被观测到时才判定遗漏已解决，否则保持未解决状态，并通过可执行的参考检查器在穷举合成测试中验证了该契约的完备性。

    

    生成式搜索的答案可能引用了有依据的段落，却遗漏了会改变其解释方式的来源关系。我们规定了一种针对“查询-来源-答案”三元组的声明门控审计方法。只有当关系证据、答案采纳、重要性和披露全部被观测到时，遗漏才被判定为“已解决”；证据不完整时保持“未解决”状态，而不是被默认当作独立性处理。该规范将此判定终点与引用支持和审查优先级区分开来，并将决策绑定到带版本号的证据片段上。一个参考检查器使记录契约变得可执行。在一个穷举式的合成测试套件上，该检查器复现了全部81种三态谓词组合，并拒绝了192条故意构造的畸形记录。公共防护基线和谓词消融实验将终点逻辑与缺失证据的处理区分开来，而受控转换则检验了支持分离与证据移除。这些是有限的契约符合性结果，而非检测器性能……（原文摘要截断）

    arXiv:2609.29145v1 Announce Type: new  Abstract: A generative search answer can cite a supported passage yet omit a source relationship that changes its interpretation. We specify a claim-gated audit of the query-source-answer tuple. An omission is resolved only when relationship evidence, answer adoption, materiality, and disclosure are all observed; incomplete evidence remains unresolved rather than being treated as independence. The specification separates this endpoint from citation support and review priority, and binds decisions to versioned evidence spans. A reference checker makes the record contract executable. On an exhaustive synthetic suite, it reproduces all 81 three-state predicate combinations and rejects 192 deliberately malformed records. Common-guard baselines and predicate ablations isolate endpoint logic from missing-evidence handling, while controlled transitions check support separation and evidence removal. These are finite contract-conformance results, not detec
    
[^26]: Seek：面向知识检索的自我评估式探索

    Seek: Self-Evaluative Exploration for Knowledge Retrieval

    [https://arxiv.org/abs/2609.28980](https://arxiv.org/abs/2609.28980)

    提出无需训练的Seek框架，通过测试时的迭代式语料库探索与自我评估式分级相关性反馈，弥补单次检索无法找回遗漏文档的缺陷，在BRIGHT基准上相对BM25提升82%并超越所有训练基线。

    

    基于大语言模型（LLM）的检索器和重排器在段落排序方面取得了显著进展，然而这两种范式都以单次交互的方式与语料库互动，并固定采纳由此产生的候选集，导致相关文档一旦被遗漏便无法再被找回。我们提出Seek（Self-Evaluative Exploration for Knowledge Retrieval），一个无需训练的框架，通过测试时的迭代式语料库交互来克服这一局限。在每一轮中，LLM基于累积的相关性反馈生成伪段落，检索器据此挖掘新的候选文档，并由专门的评估器给出分级相关性判断，以指导后续轮次的检索。在TREC Deep Learning数据集上，Seek在排序质量上可与经过训练的重排器相媲美，同时相比单次交互的BM25持续提升Recall@100。在以推理密集型著称的BRIGHT基准上，采用Qwen2.5-7B的Seek相对BM25实现了82%的相对提升，超越了所有经过训练的基线方法；采用GPT-4.1的Seek则达到了37.4的得分。

    arXiv:2609.28980v1 Announce Type: new  Abstract: LLM-based retrievers and rerankers have advanced passage ranking, yet both paradigms interact with the corpus in a single pass and commit to the resulting candidate set, leaving relevant documents permanently unrecoverable once missed. We introduce Seek, Self-Evaluative Exploration for Knowledge Retrieval, a training-free framework that addresses this limitation through iterative corpus interaction at test time. At each round, an LLM generates pseudo-passages conditioned on accumulated relevance feedback, a retriever surfaces fresh candidates, and a dedicated assessor assigns graded relevance judgments that guide subsequent rounds. On TREC Deep Learning, Seek matches trained rerankers in ranking quality while consistently improving Recall@100 over single-pass BM25. On the reasoning-intensive BRIGHT benchmark, Seek with Qwen2.5-7B achieves an 82% relative gain over BM25, surpassing all trained baselines, and Seek with GPT-4.1 reaches 37.4
    
[^27]: 面向生成式推荐的跨国代码混合方法

    Cross-Country Code-Mixing for Generative Recommendation

    [https://arxiv.org/abs/2609.28972](https://arxiv.org/abs/2609.28972)

    提出CMRec框架，借鉴多语言NLP中的语码转换思想，通过学习跨国共享语义码本并进行上下文感知的代码混合，在数据层面（而非仅参数层面）实现跨国生成式推荐的知识迁移。

    

    现代电商平台上的跨国推荐系统通常在各市场间部署相互隔离的用户与物品ID空间，这使得传统跨领域方法所依赖的共享锚点不复存在。生成式推荐（GR）通过将物品映射到共享的token空间并训练统一模型来缓解这一问题，但现有方法的行为序列仍严格限定于单一国家内部，因此知识迁移仅发生在参数层面，在数据层面依然缺失。受多语言自然语言处理中语码转换语料库的启发，我们提出CMRec——一个跨国生成式推荐框架，通过双重约束、上下文感知的代码混合，在数据层面注入跨国监督信号。CMRec首先从跨国家的多模态内容与行为共现中学习一个共享语义码本，然后利用该码本通过token级别的替换来合成混合国家的序列（摘要在此处截断）。

    arXiv:2609.28972v1 Announce Type: cross  Abstract: Cross-country recommendation on modern e-commerce platforms is typically deployed with disjoint user and item ID spaces across markets, removing the shared anchors that conventional cross-domain methods rely on. Generative recommendation (GR) mitigates this by mapping items into a shared token space and training a unified model, but existing approaches keep behavior sequences strictly country-specific, so knowledge transfer occurs only at the parameter level and remains absent at the data level. Inspired by code-switching corpora in multilingual natural language processing, we propose CMRec, a cross-country GR framework that injects cross-country supervision at the data level via dual-constrained, context-aware code-mixing. CMRec first learns a shared semantic codebook from multi-modal content and behavioral co-occurrence across countries. It then uses this codebook to synthesize mixed-country sequences via token-level substitutions th
    
[^28]: 面向小型搜索智能体的可验证奖励强化学习

    Reinforcement Learning with Verifiable Rewards for Small Search Agents

    [https://arxiv.org/abs/2609.28765](https://arxiv.org/abs/2609.28765)

    该研究首次证明，无需蒸馏，仅用GRPO和维基百科搜索工具在MuSiQue上训练0.8B参数的小模型，即可成功应用可验证奖励强化学习于开放域问答，在七个基准上取得未训练基线3.8倍的精确匹配率（0.352）。

    

    可验证奖励强化学习在奖励明确的问题（如数学和编程）上表现出色，但在奖励不那么明确的场景中是否同样有效仍是一个悬而未决的问题。“推理-搜索”方法将RLVR应用于开放域问答，其中检索为答案提供依据，与参考答案的匹配提供奖励信号。到目前为止，该方法仅在大型模型上得到验证，而在十亿参数以下的模型上只有借助更大教师模型蒸馏的先例。我们在一个小模型上测试了该方法：我们使用组相对策略优化（GRPO）和交错的维基百科搜索工具，在MuSiQue数据集上训练Qwen3.5-0.8B，仅在三种子奖励形式上进行变化（每种形式各三个随机种子），并在七个基准问答测试套件上评估每个保留检查点。结果表明该方法有效：最佳运行达到0.352的平均精确匹配率，而未训练基线仅为0.092，实现了3.8倍的提升，且无需任何蒸馏步骤。

    arXiv:2609.28765v1 Announce Type: new  Abstract: Reinforcement Learning with Verifiable Rewards (RLVR) performs well on problems with clear rewards, such as mathematics and coding, but whether it also works where the reward is less clear remains open. The reason-over-search recipe applies RLVR to open-domain question answering, where retrieval grounds the answer and a match against the reference supplies the reward. So far it has been demonstrated on large models, and below one billion parameters only with distillation from a larger teacher. We test the recipe on a small model. We train Qwen3.5-0.8B with Group Relative Policy Optimization (GRPO) and an interleaved Wikipedia-search tool on MuSiQue, varying only the reward across three shapes over three seeds each, and we evaluate every checkpoint held-out on a seven-benchmark question-answering suite. The recipe works: the best run reaches 0.352 average exact match against a 0.092 untrained floor, a 3.8-fold gain, with no distillation s
    
[^29]: 查询远征队：学习检索动作

    The Fellowship of the Query: Learning Retrieval Actions

    [https://arxiv.org/abs/2609.28653](https://arxiv.org/abs/2609.28653)

    通过轨迹微调可以让小型语言模型有效学会检索增强问答中的“下一步动作”控制决策，宏F1分数远超零样本提示，且单个SLM可同时兼任控制器与答案生成器。

    

    检索增强式问答需要对何时分解问题、搜索、重新表述、提取证据、综合事实、验证进度以及何时停止等控制决策。我们研究轨迹微调能否提升小型语言模型（SLM）作为“下一步动作控制器”的表现。我们还额外评估了一种低资源设置，即由单个SLM同时充当控制器和最终答案生成器。基于被采纳的教师搜索轨迹，我们构建了一个七分类动作预测任务，模型从当前轨迹状态预测下一个结构化的教师动作，并在多种SLM和超小型语言模型（xSLM）上评估了LoRA监督微调作为控制器的效果。在1,646个留出的动作示例上，基于13,194个动作训练的Granite 4.1 3B达到了0.6536的宏F1分数，而同一模型的零样本提示仅为0.1736，TF-IDF逻辑回归基线为0.5399。在端到端的控制器/生成器交换评估……（摘要在此处截断）

    arXiv:2609.28653v1 Announce Type: cross  Abstract: Retrieval-augmented question answering requires control decisions about when to decompose a question, search, reformulate, extract evidence, synthesize facts, verify progress, and stop. We study whether trajectory fine-tuning can improve small language models (SLMs) as next-action controllers. We additionally evaluate a low-resource setting in which a single SLM serves as both the controller and the final-answer generator. From accepted teacher search traces, we build a seven-way action-prediction task, where the model predicts the next structured teacher action from the current trajectory state, and evaluate LoRA-supervised fine-tuning across SLMs and xSLMs as controllers. On 1,646 held-out action examples, Granite 4.1 3B trained on 13,194 actions reaches macro-F1 0.6536, compared with 0.1736 for zero-shot prompting of the same model and 0.5399 for a TF-IDF logistic-regression baseline. In an end-to-end controller/generator swap evalu
    
[^30]: OneTrans-V2：在工业推荐系统中用一个Transformer统一召回、粗排与精排

    OneTrans-V2: Unifying Retrieval, Pre-rank, and Fine-rank with One Transformer in Industrial Recommender

    [https://arxiv.org/abs/2609.28589](https://arxiv.org/abs/2609.28589)

    OneTrans-V2用一个共享Transformer统一工业推荐系统的召回、粗排和精排三个级联阶段，实现一次用户序列编码、联合训练、精排到粗排的模型内知识蒸馏以及稀疏MoE扩展。

    

    工业推荐系统通常以召回、粗排和精排的级联形式运行，但这些阶段通常作为独立模型进行训练和部署，导致用户行为序列被重复编码、各阶段优化相互孤立，并造成重复的工程投入。在OneTrans模型层面统一的基础上，我们提出了OneTrans-V2，用一个Transformer统一整个级联流程。它将用户行为序列仅编码一次作为共享上下文，同时保留各阶段特定的候选特征与计算。联合训练使三个阶段相互促进，并实现了从精排到粗排的模型内知识蒸馏。我们通过稀疏混合专家模型扩展共享骨干网络，在激活计算量受限的情况下提升模型容量，并采用μP风格的参数化来稳定扩展过程。为了统一面向特定目标的各召回通道，我们引入了决策条件生成式召回。

    arXiv:2609.28589v1 Announce Type: new  Abstract: Industrial recommendation systems typically operate as a \emph{cascade} of retrieval, pre-rank, and fine-rank, but these stages are usually trained and served as separate models, causing repeated user-sequence encoding, isolated optimization, and duplicated engineering effort. Building on OneTrans' model-level unification, we present OneTrans-V2, one Transformer that unifies the entire cascade. It encodes the user behavior sequence once as a shared context while preserving stage-specific candidate features and computation. Joint training lets the three stages reinforce one another and enables in-model knowledge distillation from fine-rank to pre-rank. We scale the shared backbone with sparse mixture-of-experts (MoE), which increases capacity with bounded activated computation, and stabilize scaling with $\mu$P-style parameterization. To consolidate objective-specific retrieval channels, we introduce Decision-Conditioned Generative Retrie
    
[^31]: 重复提问会耗尽大语言模型的品牌推荐，却耗不尽其引用来源

    Repeated Queries Exhaust an LLM's Brand Recommendations but Not Its Sources

    [https://arxiv.org/abs/2609.05059](https://arxiv.org/abs/2609.05059)

    重复提问相同的购买问题时，不联网检索的大语言模型会不断涌现新的品牌推荐而难以饱和，启用检索的引擎则快速封闭品牌列表，但所有引擎引用的来源域名始终持续增加、远未收敛。

    

    重复提出相同的购买类问题是否会耗尽语言模型的品牌推荐，取决于其是否具备检索能力。在涵盖300个“问题-引擎”组合单元的实验中（50个问题、6个引擎、每个单元运行15次，并对1,470个经人工裁定的组织进行开放式抽取），五个不借助网络搜索作答的引擎在86-92%的单元中到第15次运行时仍在出现从未见过的品牌，其品牌储备中位数为15-31个组织；而唯一启用检索功能的引擎则封闭了其品牌列表（中位数8个组织，64%的单元仍在新增），这与此前四个深度实验单元中启用网络搜索的运行在第10次左右即趋于饱和的结果相符。在所有测试的时间范围内，被引用域名的累积量持续上升：四个深度实验单元在第24次运行时仍在新增域名，仅观测到Chao2下限估计值的59-84%，检索型引擎的广度单元中也有44%在第15次运行时仍在新增域名。单次运行仅能覆盖五次运行品牌集合的62-77%，且跨引擎来看，每个问题的中位数可引出38个组织……（摘要原文在此处截断）

    arXiv:2609.05059v1 Announce Type: cross  Abstract: Whether repeated identical buying questions exhaust a language model's brand recommendations depends on retrieval. Across 300 question-engine cells (50 questions, six engines, 15 runs each, open extraction over 1,470 adjudicated organizations), the five engines answering without web search were still adding never-seen brands at run 15 in 86-92% of cells, with median repertoires of 15-31 organizations; the one retrieval-enabled engine closed its list (median 8 organizations, 64% of cells still adding), matching four earlier deep cells where web-search runs saturated by run ten. Cited-domain accumulation keeps rising at every horizon tested: four deep cells were still adding domains at run 24 with 59-84% of the Chao2 lower-bound estimate observed, and 44% of the retrieval engine's breadth cells were still adding domains at run 15. A single run shows 62-77% of the five-run brand set, and across engines the median question draws 38 organiz
    
[^32]: 面向文本与多媒体数据的有效图与基于排名的上下文嵌入

    Effective Graph and Rank-based Contextual Embeddings for Textual and Multimedia Data

    [https://arxiv.org/abs/2608.29001](https://arxiv.org/abs/2608.29001)

    本文提出RaDE方法，利用基于排名的信息和代表性节点子集选择来实现图嵌入的维度可解释性，在降低计算成本的同时改进文本与多媒体数据的检索任务。

    

    在数据驱动的世界中，高效地组织和映射对象之间的关系至关重要。图是建模这些连接关系的强大工具，被广泛应用于社交网络、电信和生物学等领域。然而，基于图的方法通常面临较高的计算成本，尤其是在内存和空间使用方面。为解决这一问题，图嵌入技术（也称为网络表示学习）将图信息编码到低维表示中，同时保留结构特性。然而，传统方法缺乏可解释的维度。RaDE（Rank Diffusion Embedding，排名扩散嵌入）引入了一种使用基于排名信息的新方法，其关键步骤是选择一个具有代表性的节点子集，从而为其维度提供可解释性并改进检索任务。尽管潜力巨大，RaDE的原始提案并未充分探索代表性子集选择的有效性。

    arXiv:2608.29001v1 Announce Type: new  Abstract: In a data-driven world, efficiently organizing and mapping relationships between objects is crucial. Graphs are powerful tools for modeling these connections, being widely used in social networks, telecommunications, and biology. However, graph-based methods often face high computational costs, particularly in memory and space usage. To address this, graph embedding techniques, also referred to as Network Representation Learning, encode graph information into lower-dimensional representations while preserving structural aspects. Traditional methods, however, lack interpretable dimensions. RaDE (Rank Diffusion Embedding) introduces a new approach using rank-based information, with a key step being the selection of a representative subset of nodes to provide interpretability for its dimensions and improve retrieval tasks. Despite its potential, RaDE's original proposal did not fully explore the effectiveness of representative subset select
    
[^33]: 谁拥有AI的推荐？跨大语言模型品牌类别归属的多行业实证图谱

    Who Owns the AI Recommendation? A Multi-Industry Empirical Map of Brand Category Ownership Across Large Language Models

    [https://arxiv.org/abs/2606.23057](https://arxiv.org/abs/2606.23057)

    该研究通过对五个行业50个品牌、250个查询在三个大语言模型上跨越两个月的大规模实证测量，首次绘制了AI推荐中品牌类别归属的图谱，发现品牌收录率较为均衡、推荐份额随时间高度稳定，并识别出7.6%的“竞争真空”查询。

    

    这项探索性研究测量了五个行业、50个品牌、250个查询中的品牌收录情况，于2026年2月和9月各将每个查询五次提交给GPT-5.2、Gemini 3 Flash和Perplexity sonar-pro（分别获得3,614和3,750条评分答案）。类别收录率、推荐份额、竞争真空指数和共同提及不对称性等指标均有明确的分母定义。2月份同一行业内被采样品牌的收录率较为接近（平均基尼系数为0.30），而在250个查询中有204个查询的答案中至少80%提及了至少一个品牌。竞争真空出现在7.6%的查询中；初步的开放词汇模型解读表明，大多数真空现象反映了所采样的品牌列表本身。部分预先设定的9月复制实验显示，跨日期的推荐份额具有很强的相关性（Spearman 0.994），真空现象的普遍程度保持不变，一致性为60.8%（2月为57.2%）。这种描述性的规模关联持续存在。固定边际并不能解释……

    arXiv:2606.23057v2 Announce Type: replace-cross  Abstract: This exploratory study measures brand inclusion across five industries, 50 brands and 250 queries, each put five times to GPT-5.2, Gemini 3 Flash and Perplexity sonar-pro in February and September 2026 (3,614 and 3,750 scored answers). Category Inclusion Rate, Recommendation Share, Competitive Vacuum Index and Co-Mention Asymmetry have stated denominators. February inclusion rates sit close together across an industry's sampled brands (mean Gini 0.30), while at least one brand is named in 80% or more of answers to 204 of 250 queries. Vacuums occur in 7.6% of queries; provisional open-vocabulary model readings suggest most reflect the sampled brand list. The partially pre-specified September replication shows strong cross-date Recommendation Share correlation (Spearman 0.994), unchanged vacuum prevalence and agreement of 60.8% against February's 57.2%. The descriptive size association persists. Fixed margins do not account for a
    
[^34]: DeGRe：面向推荐的密集监督生成式重排序

    DeGRe: Dense-supervised Generative Reranking for Recommendation

    [https://arxiv.org/abs/2605.25749](https://arxiv.org/abs/2605.25749)

    该论文提出DeGRe框架，通过引入密集监督信号来解决生成式重排序中的启发式标签偏差和稀疏奖励导致的信用分配难题。

    

    在多阶段推荐系统中，重排序通过捕捉列表内部的上下文依赖关系来优化整体效用，但其核心挑战在于如何在指数级庞大的排列空间中探索最优序列。近期研究已转向端到端生成式框架，这类方法通常利用列表级奖励或偏好对齐来指导生成器的训练。然而，这些方法仍面临两个关键问题。第一是启发式标签偏差：现有方法往往基于简单规则构建训练目标，例如将用户点击过的物品提升至列表顶部，而忽略了列表上下文中的因果依赖关系。第二是信用分配问题：稀疏的列表级事后奖励无法直接指导序列生成过程中的中间步骤，导致优化方向模糊不清。为解决这些问题，我们提出了DeGRe（密集监督生成式重排序），一种……

    arXiv:2605.25749v2 Announce Type: replace-cross  Abstract: In multi-stage recommender systems, reranking optimizes overall utility by capturing intra-list contextual dependencies, yet its central challenge lies in exploring optimal sequences within an exponentially large permutation space. Recent studies have shifted towards end-to-end generative frameworks, which typically leverage list-wise rewards or preference alignment to guide generator training. However, these methods still face two critical issues. First is the heuristic label bias. Existing methods often construct training targets based on simple rules, such as promoting clicked items to the top, while ignoring causal dependencies within the list context. Second is the credit assignment problem. Sparse list-level posterior rewards fail to directly guide intermediate steps in sequence generation, leading to ambiguous optimization directions.   To address these issues, we propose DeGRe (Dense-supervised Generative Reranking), a 
    
[^35]: RQ-Reg：一种基于残差量化的推荐系统连续值预测框架

    RQ-Reg: A Residual-Quantization-Based Framework for Continuous Value Prediction in Recommender Systems

    [https://arxiv.org/abs/2602.23012](https://arxiv.org/abs/2602.23012)

    该论文提出RQ-Reg框架，通过残差量化将观看时长、GMV等连续目标值分解为从粗到细的量化编码序列并进行自回归预测，以克服传统回归方法在建模复杂长尾分布时欠拟合或牺牲泛化性的缺陷。

    

    预测观看时长和商品交易总额（GMV）等连续值是工业推荐系统中的核心问题。其固有难点在于目标信号具有高度复杂且呈长尾的分布，难以精确建模。现有的回归方法通常依赖于对目标分布的固定参数假设：过于简单的假设会对真实世界的数据欠拟合，而更复杂的假设则往往牺牲可扩展性与泛化能力。为解决这些局限性，我们提出了一种基于残差量化（RQ）的序列建模框架，将目标连续值分解为一系列量化编码，这些编码代表逐步精细化的近似值。模型以自回归的方式从粗粒度到细粒度地预测这些编码，每一步都细化上一步遗留的残差误差。为进一步提升质量……（摘要原文在此处截断）

    arXiv:2602.23012v2 Announce Type: replace-cross  Abstract: Predicting continuous values such as watch-time and gross merchandise value (GMV) is a core problem in industrial recommendation systems. Its inherent difficulty stems from the highly complex and long-tailed distributions of the target signals, which are hard to model accurately. Existing regression methods typically rely on fixed parametric assumptions on the target distribution: overly simple assumptions underfit real-world data, whereas more intricate ones tend to sacrifice scalability and generalization. To address these limitations, we propose a sequence modeling framework based on residual quantization (RQ), in which the target continuous value is decomposed into a sequence of quantization codes that represent progressively finer approximations. The model autoregressively predicts these codes from coarse to fine granularity, with each step refining the residual error left by the previous one. To further improve the qualit
    
[^36]: SPARQL-LLM：从自然语言问题实时生成SPARQL查询

    SPARQL-LLM: Real-Time SPARQL Query Generation from Natural Language Questions

    [https://arxiv.org/abs/2512.14277](https://arxiv.org/abs/2512.14277)

    SPARQL-LLM是一种开源、与三元组存储无关、由轻量级元数据驱动的方法，能够从自然语言实时生成SPARQL查询，兼顾准确性、运行时和成本等指标，从而实现生产环境的实际部署。

    

    大语言模型的出现正在推动新方法的涌现，这些方法有望更好地应对从自然语言生成结构化查询（如SPARQL查询）的挑战。然而，这些新方法大多只关注响应准确性，而忽略了其他评估标准，例如生成SPARQL查询的运行时间和成本。因此，它们往往不具备生产就绪性，也难以在真实世界的知识图谱上以良好的准确率进行部署。为了缓解这些问题，本文描述并系统评估了SPARQL-LLM，这是一种开源且与三元组存储无关的方法，由轻量级元数据驱动，能够从自然语言文本生成SPARQL查询。首先，我们描述了其架构，该架构由用于元数据索引、提示构建以及查询生成与执行的专用组件组成。然后，我们基于一项最先进的挑战对其进行了评估……

    arXiv:2512.14277v2 Announce Type: replace-cross  Abstract: The advent of large language models is contributing to the emergence of novel approaches that promise to better tackle the challenge of generating structured queries, such as SPARQL queries, from natural language. However, these new approaches mostly focus on response accuracy while ignoring other evaluation criteria, such as runtime and cost to generate SPARQL queries. Consequently, they are often not production-ready or easy to deploy over real-world knowledge graphs with good accuracy. To mitigate these issues, in this paper, we describe and systematically evaluate SPARQL-LLM, an open-source and triplestore-agnostic approach, powered by lightweight metadata, that generates SPARQL queries from natural language text. First, we describe its architecture, which consists of dedicated components for metadata indexing, prompt building, and query generation and execution. Then, we evaluate it based on a state-of-the-art challenge wi
    
[^37]: WebArxiv：一个用于评估多模态网络智能体在arXiv任务上表现的可复现基准

    WebArxiv: A Reproducible Benchmark for Evaluating Multimodal Web Agents on arXiv Tasks

    [https://arxiv.org/abs/2507.00938](https://arxiv.org/abs/2507.00938)

    WebArxiv是一个基于arXiv静态快照构建的可复现基准，包含510个具有确定性答案的时间不变任务，用于评估多模态网络智能体在多约束论文检索、细粒度内容提取和跨论文比较等学术任务上的能力。

    

    arXiv:2507.00938v3 公告类型：replace  摘要：基础模型如今使自主智能体能够与真实网站进行交互，但现有的基准测试侧重于通用浏览，低估了面向研究的环境和学术发现工作流程，并且通常依赖于实时网站，而实时网站不断变化的内容和结构损害了可复现性。arXiv提供了一个真实、可复现、层次结构化、以信息为中心且不涉及隐私敏感交互的测试平台。我们提出了WebArxiv，这是一个静态快照基准，包含510个时间不变的任务，每个任务都有唯一确定的基准答案。其多样化、真实的学术任务超越了简单的信息查找和规则遵循，强调多约束论文检索、细粒度内容提取和跨论文比较。对一系列基于基础模型的网络智能体的评估表明，WebArxiv仍然具有挑战性。行为分析显示，智能体过度依赖于

    arXiv:2507.00938v3 Announce Type: replace  Abstract: Foundation models now enable autonomous agents to interact with real-world websites, but existing benchmarks emphasize general-purpose browsing, underrepresent research-oriented environments and scholarly discovery workflows, and often depend on live sites whose changing content and structure undermine reproducibility. arXiv provides a realistic, reproducible, hierarchically structured, information-centric testbed without privacy-sensitive interactions. We introduce WebArxiv, a static-snapshot benchmark comprising 510 time-invariant tasks, each with a unique deterministic ground truth. Its diverse, realistic scholarly tasks go beyond simple information lookup and rule following to emphasize multi-constraint paper retrieval, fine-grained content extraction, and cross-paper comparison. Evaluations of a range of foundation-model-based web agents show that WebArxiv remains challenging. Behavioral analysis reveals that agents over-rely on
    

