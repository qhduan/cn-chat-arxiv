# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ECHO: Efficient Cascade with Homogeneous Operators for Distillation-Guided 100K-Sequence Modeling](https://arxiv.org/abs/2610.28569) | 提出ECHO级联框架，利用蒸馏选择器（DS）和目标感知细粒度检索器（TFR）两级结构，在10万长度用户行为序列上实现高效且保持物品级可寻址性的目标感知推荐排序建模。 |
| [^2] | [Evolving Semantic Indexes for Generative Retrieval](https://arxiv.org/abs/2610.28568) | 本文首次将LLM驱动的进化程序搜索应用于生成式检索，通过本地开源大模型自动变异语义标识符的各组件设计，并借助轻量代理解码器提供快速反馈，显著提升了检索性能。 |
| [^3] | [Multimodal Reranking Is a System One Decision](https://arxiv.org/abs/2610.28435) | 提出通用视觉语言模型可通过“系统一”决策方式充当多模态重排序器——用单个选择题的一次前向传播读取候选标签概率进行排序，无需专用重排序训练即可显著优于逐候选yes/no提问的方式。 |
| [^4] | [DeclineCraft: A RAG Abstention Benchmark for Jointly Measuring Combined Retrieval and Parametric Knowledge](https://arxiv.org/abs/2610.28340) | 该论文提出了DeclineCraft——一个基于Minecraft语料库截止日期的动态RAG弃答基准，发现单次检索能显著提升模型在不可回答问题上的正确弃答率，而允许模型自主多轮检索的迭代式RAG反而会降低弃答能力。 |
| [^5] | [AutoSim: A Self-Improving User Simulator for Conversational Search Evaluation](https://arxiv.org/abs/2610.28054) | AutoSim 是一个会话式搜索用户模拟器，它通过真实用户信号微调权重实现自我改进，模拟行为逼真到难以与真实用户区分，且其相关性判断与 NIST 高度一致，从而能够可靠地大规模评估会话式搜索系统。 |
| [^6] | [SPAR: Stepwise Personalized Autoregressive Recommendation with Prefix-Balanced Semantic IDs](https://arxiv.org/abs/2610.27898) | SPAR提出一个生成式推荐框架，通过PATH解码器从长期用户记忆中检索前缀相关证据以实现逐步个性化决策，并通过RQ-PIB模块在构建语义ID时平衡各分支的历史交互负载，从而显著提升推荐命中率。 |
| [^7] | [Robustness Analysis of Pareto-based Joint Evaluation of Fairness and Relevance in Recommender Systems](https://arxiv.org/abs/2610.27850) | 提出了一种名为Pareto前沿距离（DPFR）的新方法，通过基于用户-物品交互数据构建相关性与公平性得分的Pareto前沿，实现对推荐系统公平性和相关性的联合且平衡的评估。 |
| [^8] | [GARRET: Generative Feedback for Region Retrieval in Visual Documents](https://arxiv.org/abs/2610.27829) | GARRET通过可微的生成式反馈，仅利用（页面、问题、答案）三元组即可训练出视觉文档的区域级检索器，无需昂贵的区域标注。 |
| [^9] | [Factual Access States in Diffusion Language Models: From Mechanism to Knowledge Editing and Retrieval](https://arxiv.org/abs/2610.27580) | 该论文揭示了扩散语言模型在第一层注意力层就通过将关系措辞路由到可见实体位置来组装事实访问状态，这一机制与自回归模型的组装方式截然不同且发生更早，为知识编辑与检索提供了新的机制层面依据。 |
| [^10] | [Baseline and Incremental: User, Item, and Request Debiasing in Instagram Feed Ranking](https://arxiv.org/abs/2610.27423) | 该论文提出在Instagram信息流价值模型中同时保留原始预测（基线）并单独加权加入增量项，将观看者习惯、内容热门度和请求上下文三类独立部署的去偏统一为一个参数化算子，四臂消融实验证明基线加增量的双项设计在绝大多数指标上优于任何单项去偏方案。 |
| [^11] | [Conversational AI for Perceived Cognitive Decline Screening in Older Adults](https://arxiv.org/abs/2610.27344) | 该研究证明了基于聊天机器人的智能认知助手问答访谈可用于筛查老年人主观认知衰退，其中采用每类单示例的Gemini 3.1 Flash Lite在聚合问答流程下取得了最佳筛查性能。 |
| [^12] | [SweepANN: A GPU-Accelerated Large-K Approximate Vector Search for Drug Discovery](https://arxiv.org/abs/2610.27306) | SweepANN通过将压缩向量分区驻留在GPU上、利用单次矩阵乘法对整个查询批次批量评分并优化K相关候选的合并策略，解决了传统IVF和图索引方法在大K值（每个查询需50万候选、99%召回率）药物筛选场景下成本随K增长而失效的问题。 |
| [^13] | [FLINT: Fixed-budget Late Interaction](https://arxiv.org/abs/2610.27273) | FLINT 提出一种基于注意力池化的架构，将查询和文档对称地压缩为固定数量的嵌入向量，在大幅降低存储开销和检索时的向量比较次数的同时，保持了与多向量模型相当的检索效果。 |
| [^14] | [Neighborhood-Aware Unlearning for Generative Recommendation](https://arxiv.org/abs/2610.27232) | 本文揭示了生成式推荐中的机器遗忘会因语义标识符的共享而无意识地损害被遗忘物品邻域内的推荐性能，且该退化被全局指标掩盖，并提出邻域感知遗忘（NAU）方法，在交互级遗忘的同时对被遗忘物品邻域的概率质量进行显式控制。 |
| [^15] | [UIBK-Embedding: Effective Text Retrieval by Unifying Dense and Late-Interaction Retrieval](https://arxiv.org/abs/2610.27195) | 该论文提出单一模型 UIBK-Embedding-v1，通过将稠密嵌入作为带标签的额外向量写入词元向量列表，使标准的 MaxSim 计算能同时融合稠密检索与词元级晚期交互检索两种优势，且无需修改即可直接在现有晚期交互索引（如PLAID）中运行。 |
| [^16] | [Joint Constraint Aggregation for Qualifier-Sensitive Retrieval-Augmented QA](https://arxiv.org/abs/2610.27114) | 提出联合约束聚合方法，通过基于检索证据将候选答案与问题派生的统一要求进行联合比对，解决检索增强问答中因检索偏斜导致错误答案被选中的决策问题，并显著提升整体准确率。 |
| [^17] | [MultiCard: Multiple Attribute Cardinality Estimation via Deep Learning](https://arxiv.org/abs/2610.27019) | MultiCard 提出了一个统一的端到端深度学习基数估计器，通过 Transformer 与 LSTM 相结合来同时处理 LIKE 字符串谓词和数值过滤谓词，并能动态评估直方图估计在不同算子和数据倾斜条件下的准确性。 |
| [^18] | [The Bottleneck Is Retrieval, Not Reading: Diagnosing Evidence-Based Generation with RAG for Long Context Question Answering](https://arxiv.org/abs/2610.26951) | 该论文提出基于证据的生成（EBG）方法及 Ans+Evid 联合评估指标，通过构建多个多文档问答基准诊断发现：RAG 的瓶颈在于基于相似度的检索环节而非模型阅读，而这些检索局限正是长上下文模型所能规避的。 |
| [^19] | [GRaSP: Latency-Aware Optimization for Geo-Distributed Retrieval-Augmented Generation](https://arxiv.org/abs/2610.26938) | 该论文提出GRaSP，一种面向地理分布式检索增强生成的延迟感知重评分方法，通过将问题形式化为划分拟阵上的线性目标并用O(N log N)贪心算法最优求解，在无需重训练的情况下兼顾检索相关性、延迟与成本。 |
| [^20] | [Expert-Guided Rubrics as Dense Reward Signals for Reinforcement Learning in Non-Verifiable Domains](https://arxiv.org/abs/2610.26932) | 本文提出将少量固定的主题专家判断转化为基于评分标准的密集奖励信号，并结合检索指标解决GRPO在非可验证领域中因稀疏奖励导致优化停滞的问题。 |
| [^21] | [MOSS: A Production Framework for Monitoring and Diagnosing Semantic Failures in Search and Recommendation Systems](https://arxiv.org/abs/2610.26902) | MOSS是一个将LLM相关性评分嵌入持续监控循环的生产级框架，通过针对性采样和诊断性提示自动检测并归类推荐与搜索系统中的语义故障，已在大型零售商运行六个月并成功发现两个线上服务缺陷，还能离线量化修复效果（告警率相对降低76.4%）。 |
| [^22] | [WeGR: Unified Generative Retrieval for Industrial Search Systems in WeChat](https://arxiv.org/abs/2610.16979) | WeGR提出统一生成式检索框架，通过融合协同信号且避免码本坍缩的CoCo-VAE分词器，并解决强化学习中精确SID匹配带来的奖励稀疏问题，显著提升工业搜索的检索质量。 |
| [^23] | [KuaFu: Compressing Long User Behavior into Understanding at Billion Scale](https://arxiv.org/abs/2609.31045) | 该论文提出KuaFu系统，在十亿用户规模上解决将长用户行为压缩为用户理解时序列过长与刷新吞吐量的双重工业瓶颈，并应对压缩过程可能引入的四类幻觉问题。 |
| [^24] | [Simpler Logarithmic Approximation Algorithms for the Optimal Decision Tree and Adaptive Set Cover](https://arxiv.org/abs/2604.12036) | 该论文提出了一种带有简化分析的新贪心算法，为最优决策树问题同时实现了最坏情形代价低于 $3.65\log_2 n$ 和平均情形代价低于 $7.3\log_2 n$ 的对数近似保证。 |
| [^25] | [Deep learning enables urban change profiling through alignment of historical maps](https://arxiv.org/abs/2602.02154) | 提出了一种基于深度学习的模块化框架，通过空间地图对齐、多时相目标提取和变化估计，实现了从大规模历史地图集合中进行细粒度、定量的城市变化分析。 |
| [^26] | [From Crawled to Published: A Publisher-Controlled Web Architecture for an AI-Native Internet](https://arxiv.org/abs/2511.18354) | 提出AISTRAL架构，让出版商通过受策略控制的接口直接暴露语义单元，并借助类似DNS的语义解析器，使AI智能体能够在出版商可控的方式下进行内容的发现、检索与归属，从而解决AI时代网络的结构性缺陷。 |

# 详细

[^1]: ECHO：面向蒸馏引导的10万长度序列建模的同构算子高效级联框架

    ECHO: Efficient Cascade with Homogeneous Operators for Distillation-Guided 100K-Sequence Modeling

    [https://arxiv.org/abs/2610.28569](https://arxiv.org/abs/2610.28569)

    提出ECHO级联框架，利用蒸馏选择器（DS）和目标感知细粒度检索器（TFR）两级结构，在10万长度用户行为序列上实现高效且保持物品级可寻址性的目标感知推荐排序建模。

    

    对用户终身行为序列进行建模对于捕捉工业推荐系统中的长期偏好至关重要，然而10万长度（100K）的用户序列使得目标感知的排序计算代价高得难以承受。从原始序列中检索的基于搜索的方法会产生随序列长度增长且依赖候选物品的计算开销，而潜在压缩方法则以牺牲物品级的可寻址性为代价来降低计算量。我们提出ECHO（Efficient Cascade with Homogeneous Operators，同构算子高效级联）框架，它通过蒸馏选择器（DS）为每个用户保留一个原始事件记录子集，并通过目标感知细粒度检索器（TFR）为每个候选物品检索一个更小的目标相关记录集合，从而在效率与细粒度检索之间取得平衡。DS由具备候选感知能力的全历史教师模型通过逐目标蒸馏进行监督，实现了目标感知的用户级选择，而无需为每个候选执行全历史计算。TFR……

    arXiv:2610.28569v1 Announce Type: new  Abstract: Modeling lifetime user behavior sequences is essential for capturing long-range preferences in industrial recommendation, yet 100K-length user sequences make target-aware ranking prohibitively expensive. Search-based methods that retrieve from raw sequences incur candidate-dependent costs that grow with sequence length, whereas latent compression methods reduce computation at the expense of item-level addressability. We propose ECHO (Efficient Cascade with Homogeneous Operators), a framework that balances efficiency and fine-grained retrieval by retaining a subset of original event records per user via a Distilled Selector (DS) and retrieving a smaller set of target-relevant records per candidate via a Target-aware Fine-grained Retriever (TFR). DS is supervised by a candidate-aware full-history teacher via per-target distillation, enabling target-aware user-level selection without incurring full-history computation for every candidate. T
    
[^2]: 面向生成式检索的语义索引演化

    Evolving Semantic Indexes for Generative Retrieval

    [https://arxiv.org/abs/2610.28568](https://arxiv.org/abs/2610.28568)

    本文首次将LLM驱动的进化程序搜索应用于生成式检索，通过本地开源大模型自动变异语义标识符的各组件设计，并借助轻量代理解码器提供快速反馈，显著提升了检索性能。

    

    生成式检索（GR）将搜索与推荐任务转化为对离散语义标识符（SID）的自回归生成。SID的设计在很大程度上决定了GR的质量，但其离散且相互依赖的各个组件（SID构建、冲突处理、解码和训练）无法通过梯度进行端到端优化，目前主要依赖手工设计。我们提出了一个框架，通过LLM驱动的进化程序搜索来改进这些组件，据我们所知，这是该方法在GR领域的首次应用。一个本地部署的开源权重31B模型负责对可执行的GR组件进行变异，因此无需使用专有的LLM API；同时，一个100万参数的代理解码器提供快速反馈，替代对10亿参数目标生成器的反复训练。在NQ320K数据集上对21个SID程序的实验中，代理模型与目标模型的Recall@100表现出很强的排序相关性（Spearman ρ = 0.90）。在该代理的引导下，进化搜索在NQ320K、MSMARCO300K以及多个

    arXiv:2610.28568v1 Announce Type: new  Abstract: Generative retrieval (GR) casts search and recommendation as autoregressive generation of discrete semantic identifiers (SIDs). SID design largely determines GR quality, but its discrete, interdependent components (SID construction, collision handling, decoding, and training) cannot be optimized end-to-end with gradients and remain largely hand-designed. We introduce a framework that improves these components through LLM-driven evolutionary program search, to our knowledge the first application of this approach to GR. A locally served, open-weight 31B model mutates executable GR components, so no proprietary LLM API is used, and a 1M-parameter proxy decoder provides fast feedback in place of repeatedly training the 1B target generator. Across 21 SID programs on NQ320K, proxy and target Recall@100 show strong rank correlation (Spearman $\rho = 0.90$). Guided by this proxy, evolution improves the seed GR on NQ320K, MSMARCO300K, and a multi
    
[^3]: 多模态重排序是一种“系统一”决策

    Multimodal Reranking Is a System One Decision

    [https://arxiv.org/abs/2610.28435](https://arxiv.org/abs/2610.28435)

    提出通用视觉语言模型可通过“系统一”决策方式充当多模态重排序器——用单个选择题的一次前向传播读取候选标签概率进行排序，无需专用重排序训练即可显著优于逐候选yes/no提问的方式。

    

    多模态重排序器通常是专用模型：即在检索数据上训练一个视觉语言模型（VLM），用于对查询的候选对象进行评分。然而，对选项进行评分并非重排序所独有。“系统一”决策模型（如Jev）将一个状态和一个类型化的问题映射到每个选项的一个概率上，而无需生成文本，并且通用模型只需极少甚至无需训练就能做出此类决策。重排序正是这样一种决策：查询及其候选对象构成状态，候选对象则构成选项。因此，我们认为通用的VLM可以通过“系统一”决策充当多模态重排序器，并且决策微调可以进一步提升其重排序效果。我们向VLM针对第一阶段检索器返回的前16个候选对象提出一个选择题，并通过一次前向传播读取各候选标签的概率来对其进行排序。在MMEB-V2图像检索任务上，这种选择题方式比逐个候选对象进行yes/no提问的方式在hit@1指标上高出5.4到7.3个百分点……

    arXiv:2610.28435v1 Announce Type: new  Abstract: Multimodal rerankers are usually dedicated models: a vision-language model (VLM) is trained on retrieval data to score the candidates of a query. Scoring options is not specific to reranking. System One decision models such as Jev map a state and a typed question to one probability per option without generating text, and general models can make such decisions with little or no training. Reranking is one such decision, with the query and its candidates as the state and the candidates as the options. We therefore argue that a general VLM can serve as a multimodal reranker through System One decisions, and that decision tuning can improve its reranking. We ask a VLM one choice question over the top 16 candidates of a first-stage retriever and rank the candidates by the probabilities of their labels, read from one forward pass. On MMEB-V2 image retrieval, this choice scores 5.4 to 7.3 hit@1 above a yes/no question per candidate, and the yes/
    
[^4]: DeclineCraft：一个用于联合衡量检索知识与参数知识组合的RAG弃答基准

    DeclineCraft: A RAG Abstention Benchmark for Jointly Measuring Combined Retrieval and Parametric Knowledge

    [https://arxiv.org/abs/2610.28340](https://arxiv.org/abs/2610.28340)

    该论文提出了DeclineCraft——一个基于Minecraft语料库截止日期的动态RAG弃答基准，发现单次检索能显著提升模型在不可回答问题上的正确弃答率，而允许模型自主多轮检索的迭代式RAG反而会降低弃答能力。

    

    可靠的检索增强生成（RAG）需要模型能够识别何时检索到的文档和参数知识都无法支持答案。我们提出了DeclineCraft，一个动态弃答基准，其不可回答的问题针对语料库截止日期之后Minecraft游戏中新增的功能特性，并对可能存在的参数化知识暴露进行了审计。较晚的语料库截止日期还可支持对未来模型的评估。我们在27个大语言模型上测量了正确弃答率（CAR；即模型在不可回答问题上承认知识不足的频率）、可回答问题尝试率以及答案准确率。我们发现：（1）与无检索相比，单次检索在40个配置（模型与思考模式设置）中的38个里显著提高了CAR；（2）迭代式RAG——即模型对自身上下文进行自我评估并可请求最多两轮额外检索——在40个配置中的30个里相比单次RAG显著降低了CAR（平均46.4%），而答案准确率几乎没有变化。

    arXiv:2610.28340v1 Announce Type: new  Abstract: Reliable retrieval-augmented generation (RAG) requires recognizing when neither retrieved documents nor parametric knowledge supports an answer. We introduce DeclineCraft, a dynamic abstention benchmark whose unanswerable questions target Minecraft feature additions after a corpus cut-off, with audits of possible parametric exposure. Later cut-offs support evaluation of future models. Across 27 LLMs, we measure Correct Abstention Rate (CAR; how often models acknowledge insufficient knowledge on unanswerable questions), Answerable Attempt Rate and Answer Accuracy. We find that (1) single-shot retrieval significantly raises CAR in 38 of 40 configurations (model and thinking setting) versus no retrieval. (2) Iterative RAG, in which the model self-assesses its context and can request up to two further retrieval rounds, significantly lowers CAR versus single-shot RAG in 30 of 40 configurations (mean 46.4%), with little change in Answer Accura
    
[^5]: AutoSim：一个用于会话式搜索评估的自我改进用户模拟器

    AutoSim: A Self-Improving User Simulator for Conversational Search Evaluation

    [https://arxiv.org/abs/2610.28054](https://arxiv.org/abs/2610.28054)

    AutoSim 是一个会话式搜索用户模拟器，它通过真实用户信号微调权重实现自我改进，模拟行为逼真到难以与真实用户区分，且其相关性判断与 NIST 高度一致，从而能够可靠地大规模评估会话式搜索系统。

    

    用户模拟器有望以人类研究无法企及的规模来评估会话式搜索系统，但其前提是模拟行为必须足够真实可信。我们研究了 AutoSim——一个会话式搜索用户模拟器，其相关性判断在三个 TREC iKAT 年份中对一组受控检索系统的排序与 NIST 保持一致（Kendall's τ 分别为 0.66、0.91 和 0.91；所有 p ≤ 0.002），且这种一致性在移除其人格设定后依然保持。AutoSim 利用来自真实用户的保留信号进行自我改进，我们发现对其权重进行微调能使其表现得像真实用户，从而更准确地估计系统对真实用户的服务效果。要达到这种效果，仅靠提示工程是不够的：在区分真实与模拟用户会话时，我们尝试的所有提示方法都无法使保留集上的真实-模拟分类器的 AUC 降至 0.85 以下，而微调权重则将其降至 0.45，此时无论是……（原文在此处截断）

    arXiv:2610.28054v1 Announce Type: new  Abstract: User simulators promise to evaluate conversational search systems at a scale human studies cannot reach, but only if their behavior is realistic enough to trust. We study AutoSim, a conversational-search user simulator whose relevance judgments rank a controlled pool of retrieval systems in agreement with NIST across three TREC iKAT years (Kendall's $\tau$ of $0.66$, $0.91$, and $0.91$; all $p{\le}0.002$), an agreement that survives removing its persona. AutoSim self-improves against a held-out signal from real users, and we find that fine-tuning its weights makes it behave like real users, which in turn gives a more accurate estimate of how well a system serves them. Reaching that behavior takes more than prompting: when discriminating between real and simulated user sessions, no prompt we tried lowers a held-out real-versus-simulated classifier below AUC $0.85$, whereas fine-tuning the weights drives it to $0.45$, where neither the cla
    
[^6]: SPAR：基于前缀平衡语义ID的逐步个性化自回归推荐

    SPAR: Stepwise Personalized Autoregressive Recommendation with Prefix-Balanced Semantic IDs

    [https://arxiv.org/abs/2610.27898](https://arxiv.org/abs/2610.27898)

    SPAR提出一个生成式推荐框架，通过PATH解码器从长期用户记忆中检索前缀相关证据以实现逐步个性化决策，并通过RQ-PIB模块在构建语义ID时平衡各分支的历史交互负载，从而显著提升推荐命中率。

    

    基于语义ID（SID）的生成式推荐通过连续的token级决策来检索物品，因此在每个解码步骤实现个性化至关重要。然而，现有方法对依赖前缀的用户证据以及候选分支之间不均衡的历史交互负载考虑不足，这可能导致不可逆的分支选择错误，并偏向热门路径。我们提出SPAR，一个同时改进逐步个性化与SID构建的生成式推荐框架。其PATH解码器从压缩的长期用户记忆中检索与前缀相关的证据，并将其与近期行为、用户画像和事件上下文相结合，以指导每个token的决策。其RQ-PIB模块在每个父前缀下的兄弟分支之间平衡历史交互负载，同时保持语义结构。在大规模优酷生产数据上的实验表明，平均相对HitRate提升达2%（摘要在此处被截断）。

    arXiv:2610.27898v1 Announce Type: new  Abstract: Semantic ID (SID)-based generative recommendation retrieves items through successive token-level decisions, making personalization at each decoding step essential. However, existing methods insufficiently account for prefix-dependent user evidence and imbalanced historical interaction loads among candidate branches. This can cause irreversible branch-selection errors and favor popular paths. We propose SPAR, a generative recommendation framework that jointly improves stepwise personalization and SID construction. Its PATH Decoder retrieves prefix-relevant evidence from compressed long-term user memories and combines it with recent behavior, user profiles, and event context to guide each token decision. Its RQ-PIB module balances historical interaction loads among sibling branches under each parent prefix while preserving semantic structure. Experiments on large-scale Youku production data show an average relative HitRate improvement of 2
    
[^7]: 基于Pareto的推荐系统中公平性与相关性联合评估的鲁棒性分析

    Robustness Analysis of Pareto-based Joint Evaluation of Fairness and Relevance in Recommender Systems

    [https://arxiv.org/abs/2610.27850](https://arxiv.org/abs/2610.27850)

    提出了一种名为Pareto前沿距离（DPFR）的新方法，通过基于用户-物品交互数据构建相关性与公平性得分的Pareto前沿，实现对推荐系统公平性和相关性的联合且平衡的评估。

    

    推荐系统（RSs）可以从多个方面进行评估，其中两个方面是推荐相关性和公平性。这两个方面既可以分别独立评估，即对每个方面使用一个度量指标；也可以联合评估，即使用单一的“相关性感知”公平性度量来量化相对于相关性的公平性。然而，这两种方法都无法同时对相关性和公平性进行评估：第一种方法可能导致两种不同的模型排名，每种排名对应一个评估方面；第二种方法通常对这两个方面的评估是不平衡的。为了弥补这一空白，我们提出了Pareto前沿距离（DPFR），这是一种联合评估推荐系统公平性和相关性的新方法。DPFR基于现有度量指标得到的相关性和公平性得分构建的Pareto前沿。该前沿由用户-物品交互数据生成，代表了两个方面的各种Pareto最优权衡。

    arXiv:2610.27850v1 Announce Type: new  Abstract: Recommender Systems (RSs) can be evaluated for many aspects, two of them being recommendation relevance and fairness. These two aspects can be evaluated either in isolation, using a measure for each aspect; or jointly with a single`relevance-aware' fairness measure that quantifies fairness with respect to relevance. Yet, neither approach provides a simultaneous evaluation of both relevance and fairness: the first approach may result in two different rankings of models, each corresponding to an evaluation aspect; the second approach often provides an imbalanced evaluation of the two aspects. To address this gap, we propose Distance to Pareto Frontier (DPFR), a new approach to jointly evaluate RS fairness and relevance. DPFR is based on a Pareto frontier of relevance and fairness scores from existing measures. The frontier is generated from user-item interaction data and represents various Pareto-optimal trade-offs between the two aspects.
    
[^8]: GARRET：面向视觉文档区域检索的生成式反馈

    GARRET: Generative Feedback for Region Retrieval in Visual Documents

    [https://arxiv.org/abs/2610.27829](https://arxiv.org/abs/2610.27829)

    GARRET通过可微的生成式反馈，仅利用（页面、问题、答案）三元组即可训练出视觉文档的区域级检索器，无需昂贵的区域标注。

    

    在视觉内容丰富的文档上进行多模态检索增强生成，需要比整页更细粒度的证据。现有的视觉检索器通常索引或返回完整页面，而训练区域级检索器往往需要代价高昂的查询-区域标注或带有噪声的模型生成标签。我们提出GARRET，一个通过可微生成式反馈、从（页面、问题、答案）三元组训练的区域检索框架。一个共享的上下文保持检索器生成页面和区域的多向量表示。在训练过程中，其检索得分定义了候选区域裁剪图块上的软路由分布。一个辅助视觉语言模型分别关注每个图块，利用路由权重聚合图块特定的上下文，并在下一词预测目标下先预测问题再预测答案。通过路由分布回传的梯度为查询条件分配信用（摘要在此处被截断）。

    arXiv:2610.27829v1 Announce Type: cross  Abstract: Multimodal retrieval-augmented generation over visually rich documents requires evidence at a finer granularity than entire pages. Existing visual retrievers typically index or return full pages, whereas training region-level retrievers often requires costly query-to-region annotations or noisy model-generated labels. We introduce GARRET, a region-retrieval framework trained from (page, question, answer) triplets through differentiable generative feedback. A shared context-preserving retriever produces page and region multi-vector representations. During training, their retrieval scores define a soft routing distribution over candidate region crops. An auxiliary vision-language model attends to each crop separately, aggregates crop-specific contexts using the routing weights, and predicts the question followed by the answer under a next-token prediction objective. Gradients through the routing distribution assign query-conditioned cred
    
[^9]: 扩散语言模型中的事实访问状态：从机制到知识编辑与检索

    Factual Access States in Diffusion Language Models: From Mechanism to Knowledge Editing and Retrieval

    [https://arxiv.org/abs/2610.27580](https://arxiv.org/abs/2610.27580)

    该论文揭示了扩散语言模型在第一层注意力层就通过将关系措辞路由到可见实体位置来组装事实访问状态，这一机制与自回归模型的组装方式截然不同且发生更早，为知识编辑与检索提供了新的机制层面依据。

    

    事实回忆本质上是一种查找：网络内部的某种表示充当键，而该键的构成决定了哪些查询能够检索到该事实、一次编辑必须覆盖哪些内容、以及检索器能够读取到什么。我们将其称为“事实访问状态”。在自回归（AR）模型中，主语位置对属性的丰富过程在很大程度上独立于关系，而针对关系的选择发生在较晚的读出位置。在掩码扩散语言模型（DLMs）中，我们发现该状态的组装方式不同，且时间上早得多：在第一层注意力层中，关系的措辞被路由到可见实体的位置上，随后早期的前馈层基于这一融合表示查找事实。因此，该状态会随关系的表面形式和查询方向而变化，这意味着关于同一事实的查询占据的是状态空间中的一个区域而非一个点，而对事实的操作能否成功则取决于（摘要在此处截断，未完）

    arXiv:2610.27580v1 Announce Type: cross  Abstract: Factual recall is a lookup: some representation inside the network serves as the key, and what that key is made of decides which queries can retrieve the fact, what an edit must cover, and what a retriever can read. We call it the factual access state. In autoregressive (AR) models the subject position enriches attributes largely independently of the relation, and relation-specific selection happens later, at the readout position. In masked diffusion language models (DLMs) we find the state is assembled differently, and much earlier: in the first attention layer the wording of the relation is routed into the position of whichever entity is visible and the early feed-forward layers look the fact up on this fused representation. The state therefore varies with the relation's surface form and with the query direction, so the queries about one fact occupy a region of state space rather than a point, and an operation on the fact succeeds on
    
[^10]: 基线与增量：Instagram信息流排序中的用户、物品与请求去偏

    Baseline and Incremental: User, Item, and Request Debiasing in Instagram Feed Ranking

    [https://arxiv.org/abs/2610.27423](https://arxiv.org/abs/2610.27423)

    该论文提出在Instagram信息流价值模型中同时保留原始预测（基线）并单独加权加入增量项，将观看者习惯、内容热门度和请求上下文三类独立部署的去偏统一为一个参数化算子，四臂消融实验证明基线加增量的双项设计在绝大多数指标上优于任何单项去偏方案。

    

    信息流排序器使用价值模型为候选内容打分，该模型是各项互动预测的加权和。每个预测都混淆了两个部分：基线——即该观看者、该内容或该请求本来就会产生的互动量——以及该候选内容超出基线的增量部分。现有文献大多采取移除基线的做法。我们主张，多信号价值模型应当保留原始预测，并将增量部分作为一个单独加权的项加入模型。在Instagram信息流中，三个独立部署的去偏方案族（分别针对观看者习惯、内容热门度和请求上下文）实际上可统一为一个参数化算子，并通过一套设计流程在该算子中选取一个具体的点。媒体类型（格式）偏差则转而由同一流程所选定的去偏标签来处理。在一项针对已部署信息流项的受控四臂消融实验（无任何项、仅含其一、两者兼具）中，在生产权重下的点估计上，双项方案在13个受评判的（项，指标）对中有11个优于两个单项方案。在26个直接……

    arXiv:2610.27423v1 Announce Type: cross  Abstract: A feed ranker scores candidates with a value model, a weighted sum of engagement predictions. Each prediction conflates a baseline, what this viewer, item or request would produce anyway, with the incremental amount by which this candidate exceeds it. The literature mostly removes the baseline. We argue that a multi-signal value model should keep the raw prediction and add the incremental part as a separately weighted term. On Instagram Feed, three debiasing families deployed independently, for viewer habit, item popularity and request context, are one parameterised operator, and a design procedure picks a point in it. Media-type (format) bias goes instead to a debiased label chosen by the same procedure. In a controlled four-arm ablation of deployed Feed terms (neither term, each alone, both), the two-term arm beats both single-term arms at production weights on point estimates in 11 of 13 judged (term, metric) pairs. Of the 26 direct
    
[^11]: 面向老年人主观认知衰退筛查的对话式人工智能

    Conversational AI for Perceived Cognitive Decline Screening in Older Adults

    [https://arxiv.org/abs/2610.27344](https://arxiv.org/abs/2610.27344)

    该研究证明了基于聊天机器人的智能认知助手问答访谈可用于筛查老年人主观认知衰退，其中采用每类单示例的Gemini 3.1 Flash Lite在聚合问答流程下取得了最佳筛查性能。

    

    由于标准神经心理学评估对资源的高需求，老年人认知衰退的早期检测仍然具有挑战性。基于聊天机器人的智能认知助手（ICAs）通过引导生成富含语言、语义和自传体线索的开放式回答，提供了一种可扩展的替代方案。本研究评估了智能认知助手与老年人进行问答（Q/A）访谈能否筛查参与者自报的主观认知衰退及相关认知、功能和人格结果。我们在全局问答、单问答置信度平均和聚合问答三种流程下，比较了传统机器学习、冻结transformer嵌入、微调transformer以及大语言模型（LLM）推理策略。在基于文本的方法中，Gemini 3.1 Flash Lite（每类仅一个示例、聚合问答流程）取得了最高的留出测试集性能（准确率0.742、平衡准确率0.739、宏F1值0.728）。

    arXiv:2610.27344v1 Announce Type: new  Abstract: Early detection of cognitive decline in older adults remains challenging due to the resource demands of standard neuropsychological assessments. Chatbot-based Intelligent Cognitive Assistants (ICAs) provide a scalable alternative by eliciting open-ended responses rich in linguistic, semantic, and autobiographical cues. This study evaluates whether ICA question-answer (Q/A) interviews with older adults can screen for participant-reported perceived cognitive decline and related cognitive, functional, and personality outcomes. We compare traditional machine learning, frozen transformer embeddings, fine-tuned transformers, and large language model (LLM) reasoning strategies across Global Q/A, Single Q/A confidence averaging, and Aggregated Q/A pipelines. Among text-based methods, Gemini 3.1 Flash Lite (1-example-per-class, Aggregated Q/A) achieved the highest held-out test performance (0.742 accuracy, 0.739 balanced accuracy, 0.728 macro-F1)
    
[^12]: SweepANN：面向药物发现的GPU加速大K值近似向量搜索

    SweepANN: A GPU-Accelerated Large-K Approximate Vector Search for Drug Discovery

    [https://arxiv.org/abs/2610.27306](https://arxiv.org/abs/2610.27306)

    SweepANN通过将压缩向量分区驻留在GPU上、利用单次矩阵乘法对整个查询批次批量评分并优化K相关候选的合并策略，解决了传统IVF和图索引方法在大K值（每个查询需50万候选、99%召回率）药物筛选场景下成本随K增长而失效的问题。

    

    DrugCLIP将小分子和蛋白质口袋置于同一向量空间中，把全基因组范围的药物发现转化为基于向量邻近性的大K值检索任务。一次筛选活动会针对一个超过5亿分子的化合物文库预先固定超过2万个蛋白质口袋。每个口袋需保留K = 50万个候选结果（约占整个集合的0.1%），而典型的ANN基准测试中K仅为100。由于漏掉的候选永远不会被实际测试，因此相对于精确搜索，平均召回率必须达到99%。如此巨大的答案规模导致IVF索引必须探测每一个包含结果的列表，图搜索则必须追踪K个以上的候选，使得计算成本随K线性增长。在我们的CPU和GPU平台上，这两类索引家族都无法在保留其相对暴力扫描优势的前提下达到目标。SweepANN将压缩后的向量划分到多个GPU的显存中，并逐一扫描每个驻留分区。一次矩阵乘法即可为整个查询批次对每个数据块进行评分，每个块仅需读取一次。系统采用多种执行方案来完成与K相关的候选选择与合并操作，每种方案都能在压缩表示下返回精确的top-K结果。

    arXiv:2610.27306v1 Announce Type: cross  Abstract: DrugCLIP places small molecules and protein pockets in one space, turning genome-wide drug discovery into large-K retrieval by vector proximity. A campaign fixes over 20K pockets in advance for one library of over 500M molecules. Each pocket keeps K = 500K candidates, 0.1% of the collection, versus 100 in ANN benchmarks. Missed hits never get tested, so mean recall must reach 99% relative to exact search. Large answers make IVF probe every list holding a result and graph search track K or more candidates, so costs grow with K. On our CPUs and GPUs, neither index family reaches the target without losing most of its edge over scanning.   SweepANN partitions compressed vectors across GPUs and scans each resident partition. One matrix multiplication scores each block for a whole query batch, reading it once. Several plans perform the K-dependent selection and merging of each query's K answers, each returning an exact top-K under compressed
    
[^13]: FLINT：固定预算的后期交互

    FLINT: Fixed-budget Late Interaction

    [https://arxiv.org/abs/2610.27273](https://arxiv.org/abs/2610.27273)

    FLINT 提出一种基于注意力池化的架构，将查询和文档对称地压缩为固定数量的嵌入向量，在大幅降低存储开销和检索时的向量比较次数的同时，保持了与多向量模型相当的检索效果。

    

    arXiv:2610.27273v1 公告类型： cross 摘要：以 ColBERT 为代表的多向量检索模型通过在词元级别匹配查询和文档，往往在同等规模下优于单向量模型。然而，为每个词元保留一个向量会显著增加存储需求，并且与单向量双编码器相比会降低检索速度，限制了其可扩展性。我们提出了 FLINT（Fixed-budget Late INTeraction，固定预算后期交互），这是一种利用基于注意力的池化机制，以固定数量的嵌入向量来表示每个查询和文档的架构。与仅对文档表示进行池化而保留词元级查询的架构不同，FLINT 对查询和文档两侧对称地应用池化，从而同时限制了文档存储量以及检索过程中所需的向量比较次数。FLINT 在 BEIR 上取得了与最接近的基线相当的排序效果，并在包含更长、更复杂查询的多个基准测试上超越了它们，包括 BrowseComp+、CRUMB 和 T

    arXiv:2610.27273v1 Announce Type: cross  Abstract: Multi-vector retrieval models such as ColBERT often outperform single-vector models of comparable scale by matching queries and documents at the token level. However, retaining a vector for every token substantially increases storage requirements and slows retrieval compared to single-vector bi-encoders, limiting their scalability. We introduce FLINT (Fixed-budget Late INTeraction), an architecture that uses attention-based pooling to represent each query and document with a fixed number of embedding vectors. Unlike architectures that pool only document representations while retaining token-level queries, FLINT applies pooling symmetrically to both sides, limiting document storage and the number of vector comparisons needed during retrieval. FLINT achieves comparable ranking effectiveness to the closest baselines on BEIR and outperforms them on several benchmarks with longer and more complex queries, including BrowseComp+, CRUMB, and T
    
[^14]: 面向生成式推荐的邻域感知遗忘

    Neighborhood-Aware Unlearning for Generative Recommendation

    [https://arxiv.org/abs/2610.27232](https://arxiv.org/abs/2610.27232)

    本文揭示了生成式推荐中的机器遗忘会因语义标识符的共享而无意识地损害被遗忘物品邻域内的推荐性能，且该退化被全局指标掩盖，并提出邻域感知遗忘（NAU）方法，在交互级遗忘的同时对被遗忘物品邻域的概率质量进行显式控制。

    

    生成式推荐通过自回归方式生成语义标识符（SID）来预测物品，而这些标识符的词元在语义相关的物品之间是共享的。随着这一范式逐渐被采用，如何在其框架下支持机器遗忘已成为一个紧迫的问题。当训练数据中的交互需要被移除时——例如注入的垃圾信息，或用户不再想要的物品的交互——这种共享机制会使遗忘更新在无意中影响到与被遗忘交互相邻的物品的预测结果。我们证明这种影响具有局部结构性，并且会被全局指标所掩盖：遗忘操作可能使整个测试集上的推荐效用几乎保持不变，同时却在被遗忘交互的SID所在邻域上造成显著的性能退化。我们提出了邻域感知遗忘（Neighborhood-Aware Unlearning, NAU），它在交互级别的遗忘基础上，增加了对分配给被遗忘物品邻域的概率质量的显式控制。在三个数据集上的实验……（摘要原文在此处截断）

    arXiv:2610.27232v1 Announce Type: cross  Abstract: Generative recommendation predicts items by autoregressively generating semantic identifiers (SIDs), whose tokens are shared across semantically related items. With this paradigm gaining adoption, the question of how to support unlearning in it has become pressing. If training interactions must later be removed, such as injected spam or interactions with items a user no longer wants, this sharing lets an unlearning update unintentionally affect the predictions for the interactions' neighbors. We show that this effect is locally structured and hidden by global metrics: unlearning can leave recommendation utility on the whole test set nearly unchanged while substantially degrading it on neighborhoods of the forgotten interactions' SIDs. We propose Neighborhood-Aware Unlearning (NAU), which augments interaction-level unlearning with explicit control over the probability mass assigned to the forgotten items' neighborhoods. Across three dat
    
[^15]: UIBK-Embedding：通过统一稠密检索与晚期交互检索实现高效文本检索

    UIBK-Embedding: Effective Text Retrieval by Unifying Dense and Late-Interaction Retrieval

    [https://arxiv.org/abs/2610.27195](https://arxiv.org/abs/2610.27195)

    该论文提出单一模型 UIBK-Embedding-v1，通过将稠密嵌入作为带标签的额外向量写入词元向量列表，使标准的 MaxSim 计算能同时融合稠密检索与词元级晚期交互检索两种优势，且无需修改即可直接在现有晚期交互索引（如PLAID）中运行。

    

    稠密双编码器和晚期交互检索器通常被视为两种备选方案，但它们会在不同的查询上失败：单个向量捕捉文本的主题内容，而词元级匹配则能找到文本中确切的实体和短语。目前若要同时使用两者，需要两个模型、两个索引以及一个分数融合步骤。我们提出了 UIBK-Embedding-v1，一个拥有1.79亿参数的英文编码器，它能通过一次前向传播同时生成两种表示，并使用一个未经修改的 MaxSim 对其进行打分。稠密嵌入被作为三个额外向量写入词元向量列表中，同时一个一维标签防止两类向量相互匹配，从而使普通的 MaxSim 等于词元级 MaxSim 加上加权的稠密余弦相似度。因此，该模型无需任何改动即可在标准晚期交互索引中运行；我们使用 PLAID 对其进行了测试。共享的18层主干网络、每个头独立的4层顶层，以及将交叉编码器蒸馏到融合分数中的方法，使两个头合并为一个检索器。

    arXiv:2610.27195v1 Announce Type: new  Abstract: Dense bi-encoders and late-interaction retrievers are usually treated as alternatives, yet they fail on different queries: one vector captures what a text is about, token-level matching finds its exact entities and phrases. Using both today means two models, two indexes and a score-fusion step. We present UIBK-Embedding-v1, a 179M-parameter English encoder that produces both representations in one pass and scores them with one unmodified MaxSim. The dense embedding is written into the list of token vectors as three extra vectors, and a one-dimensional tag keeps the two kinds of vectors from matching each other, so that plain MaxSim equals the token-level MaxSim plus a weighted dense cosine. The model therefore runs in a standard late-interaction index without changes; we test it with PLAID. A shared 18-layer trunk with a separate 4-layer top per head, and a cross-encoder distilled into the fused score, turn the two heads into one retriev
    
[^16]: 面向限定条件敏感的检索增强问答的联合约束聚合方法

    Joint Constraint Aggregation for Qualifier-Sensitive Retrieval-Augmented QA

    [https://arxiv.org/abs/2610.27114](https://arxiv.org/abs/2610.27114)

    提出联合约束聚合方法，通过基于检索证据将候选答案与问题派生的统一要求进行联合比对，解决检索增强问答中因检索偏斜导致错误答案被选中的决策问题，并显著提升整体准确率。

    

    检索增强问答即使当正确答案出现在检索到的证据中时也可能失败。这种情况可能发生在：某个竞争性答案只满足问题要求中的一部分（而非全部），却在检索到的文档中被频繁提及，从而使其看起来像是得到最多支持的答案。为了检验这种检索偏斜是否真的会影响模型决策，我们直接对其进行操控：增加偏向竞争性答案的段落的数量，会使模型更倾向于选择该答案。这表明存在一个仅凭候选答案层面的支持度无法解决的决策问题。因此，我们提出了联合约束聚合（JCA），它利用检索到的证据，将候选答案与由问题派生的相同要求进行对比。在三种评估设置和三种指令微调骨干模型上，JCA 提升了整体准确率，并且当检索偏向竞争性答案时，其优势会进一步扩大。这些结果……

    arXiv:2610.27114v1 Announce Type: cross  Abstract: Retrieval-augmented QA can fail even when the correct answer is in the retrieved evidence. This can happen when a competing answer satisfies some, but not all, of the question's requirements and is mentioned frequently in the retrieved documents, which can make it appear to be the best-supported answer. To test whether this retrieval skew actually affects the decision, we manipulate it directly: increasing the prevalence of passages that favor the competing answer makes the model more likely to select it. This points to a decision problem that candidate-level support alone cannot resolve. We therefore introduce Joint Constraint Aggregation (JCA), which compares candidates against the same question-derived requirements using the retrieved evidence. Across three evaluation settings and three instruction-tuned backbones, JCA improves overall accuracy, and its advantage grows when retrieval favors a competing answer. Together, these result
    
[^17]: MultiCard：基于深度学习的多属性基数估计

    MultiCard: Multiple Attribute Cardinality Estimation via Deep Learning

    [https://arxiv.org/abs/2610.27019](https://arxiv.org/abs/2610.27019)

    MultiCard 提出了一个统一的端到端深度学习基数估计器，通过 Transformer 与 LSTM 相结合来同时处理 LIKE 字符串谓词和数值过滤谓词，并能动态评估直方图估计在不同算子和数据倾斜条件下的准确性。

    

    基数估计对查询优化至关重要，但现有的学习型估计器大多只关注单一类型的谓词，例如字符串 LIKE 子句或数值范围过滤条件。在实际应用中，分析型工作负载往往同时包含这两类谓词，而且字符串估计器的准确率通常会随着模式变长而下降。我们提出了 MultiCard，一个全面的学习型基数估计器，它在单一的端到端架构中统一处理 LIKE 谓词和数值过滤条件。通过将用于处理长而稀疏模式的 Transformer 与用于处理短而高频模式的 LSTM 相结合，MultiCard 实现了鲁棒的模式建模。与此同时，一个数值编码器整合了数据目录中的直方图和结构化谓词特征，使模型能够在不同算子和数据倾斜条件下动态评估直方图估计的准确性。最后一个 Transformer 层捕捉异构谓词之间的交互，然后经过多……（原文摘要在此处截断）

    arXiv:2610.27019v1 Announce Type: cross  Abstract: Cardinality estimation is essential for query optimization, but most existing learned estimators focus on a single predicate type, such as string LIKE clauses or numerical range filters. In practice, analytical workloads often combine both types, while the accuracy of string estimators tends to decrease with longer patterns. We introduce MultiCard, a comprehensive learned cardinality estimator that unifies LIKE predicates and numerical filters within a single end-to-end architecture. By combining a Transformer for long-range sparse patterns with an LSTM for short, frequent patterns, MultiCard achieves robust pattern modeling. In parallel, a numerical encoder integrates catalog histograms and structural predicate features, enabling the model to dynamically assess histogram estimation accuracy across different operators and data skew conditions. A final Transformer layer captures interactions among heterogeneous predicates before a multi
    
[^18]: 瓶颈在于检索而非阅读：用基于证据的生成诊断面向长上下文问答的RAG

    The Bottleneck Is Retrieval, Not Reading: Diagnosing Evidence-Based Generation with RAG for Long Context Question Answering

    [https://arxiv.org/abs/2610.26951](https://arxiv.org/abs/2610.26951)

    该论文提出基于证据的生成（EBG）方法及 Ans+Evid 联合评估指标，通过构建多个多文档问答基准诊断发现：RAG 的瓶颈在于基于相似度的检索环节而非模型阅读，而这些检索局限正是长上下文模型所能规避的。

    

    检索增强生成（RAG）是将大语言模型与外部语料库相结合的主流范式，但其基于相似度的分块流水线存在结构性盲区：它无法预判问题需要哪些段落，并且会丢弃能够解决跨段落引用的文档级上下文。我们提出了基于证据的生成（EBG），要求模型在返回答案的同时提供行级引用，并设计了联合的Ans+Evid指标，可以同时评估答案的正确性与证据支撑度。我们将两个语料库重新构建为基于证据的多文档问答基准——MultiHop-RAG 和 QASPER-MP（QASPER 的多论文改编版本，带有真实场景中的短论文提示），并配以受控的合成两跳“大海捞针”测试，以及覆盖1万至100万字符的证据增强版 NoLiMa。在十个专有前沿大模型和三个开源权重大模型上的实验中，EBG 分离出了长上下文模型可以规避的两个 RAG 局限性：相似性检索……（摘要在此处截断）

    arXiv:2610.26951v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) is the dominant paradigm for grounding large language models in external corpora, but its similarity-driven, chunked pipeline has structural blind spots: it cannot anticipate which passages a question requires and discards document-level context that resolves cross-passage references. We introduce evidence-based generation (EBG), requiring models to return an answer with line-level citations, and a joint Ans+Evid metric that scores correctness and grounding simultaneously. We reformulate two corpora into evidence-grounded multi-document QA benchmarks- MultiHop-RAG and QASPER-MP, a multi-paper adaptation of QASPER with realistic short paper cues, and pair them with a controlled synthetic two-hop haystack and an evidence-augmented NoLiMa spanning 10K-1M characters. Across ten proprietary frontier and three open-weight LLMs, EBG isolates two RAG limitations that long-context models avoid: similarity retr
    
[^19]: GRaSP：面向地理分布式检索增强生成的延迟感知优化

    GRaSP: Latency-Aware Optimization for Geo-Distributed Retrieval-Augmented Generation

    [https://arxiv.org/abs/2610.26938](https://arxiv.org/abs/2610.26938)

    该论文提出GRaSP，一种面向地理分布式检索增强生成的延迟感知重评分方法，通过将问题形式化为划分拟阵上的线性目标并用O(N log N)贪心算法最优求解，在无需重训练的情况下兼顾检索相关性、延迟与成本。

    

    检索增强生成（RAG）日益被部署在地理分布式的云和边缘基础设施上，在这些环境中，广域网延迟、跨区域出口流量成本以及数据驻留规则与检索质量相互制约。传统流水线仅依据语义相似度进行排序，而忽略了内容所在的地理位置，从而推高了尾延迟和成本。我们将地理RAG选择问题形式化为GRaSP，这是一个即插即用的重评分层，通过延迟对相关性进行衰减，并在基数预算和各区域上限约束下添加可选的成本惩罚，且无需重新训练。我们首先确定了该问题的计算复杂度：基础形式化是划分拟阵上的线性目标函数，可以通过O(N log N)的贪心算法最优求解，而非像背包问题框架所暗示的那样是NP难问题；只有在一附加出口预算约束下问题才会变难。在模拟四区域广域网的五节点CloudLab测试床上，基于Natural Questions语料库（268万个段落）的实验中，GRaSP的收益……

    arXiv:2610.26938v1 Announce Type: cross  Abstract: Retrieval-Augmented Generation (RAG) is increasingly served across geo-distributed cloud and edge infrastructures, where wide-area latency, cross-region egress cost, and data-residency rules compete with retrieval quality. Conventional pipelines rank by semantic similarity alone and ignore where content lives, inflating tail latency and cost. We formalize the Geo-RAG Selection Problem (GRaSP), a drop-in re-scoring layer that decays relevance by latency and adds an optional cost penalty under a cardinality budget and per-region caps, with no retraining. We first settle its complexity: the base formulation is a linear objective over a partition matroid, solved optimally by an O(N log N) greedy algorithm, rather than NP-hard as a knapsack framing suggests, with hardness entering only under an additive egress budget. On a five-node CloudLab testbed emulating a four-region WAN over the Natural Questions corpus (2.68M passages), GRaSP's bene
    
[^20]: 专家引导的评分标准作为非可验证领域强化学习的密集奖励信号

    Expert-Guided Rubrics as Dense Reward Signals for Reinforcement Learning in Non-Verifiable Domains

    [https://arxiv.org/abs/2610.26932](https://arxiv.org/abs/2610.26932)

    本文提出将少量固定的主题专家判断转化为基于评分标准的密集奖励信号，并结合检索指标解决GRPO在非可验证领域中因稀疏奖励导致优化停滞的问题。

    

    arXiv:2610.26932v1 公告类型：交叉 摘要：大语言模型（LLM）正越来越多地被部署为金融服务等非可验证领域中的自主智能体，在这些领域中，它们必须解释用户意图、重写查询，并在特定领域约束下检索信息。尽管强化学习（RL）可以通过行动的后果来训练此类智能体，但近期的成功主要依赖于具有自动计算二元奖励的可验证任务。在非可验证的检索任务中，评估查询质量需要稀缺的主题专家（SME）判断，而这种判断本质上是分级的而非二元的。在使用组相对策略优化（GRPO）时，当组内所有采样输出都获得相同结果时，稀疏奖励无法产生有用的学习信号，导致优化停滞。我们提出了一种方法，将少量固定的SME判断投入转化为密集奖励，并将其与基于检索的指标相结合，形成一种混合的（摘要在此处被截断）

    arXiv:2610.26932v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly deployed as autonomous agents in non-verifiable domains such as financial services, where they must interpret user intent, rewrite queries, and retrieve information under domain-specific constraints. Although reinforcement learning (RL) can train such agents from the consequences of their actions, recent successes largely rely on verifiable tasks with automatically computed binary rewards. In non-verifiable retrieval tasks, assessing query quality requires scarce subject-matter expert (SME) judgment that is inherently graded rather than binary. With Group Relative Policy Optimization (GRPO), sparse rewards yield no useful learning signal when all sampled outputs within a group receive the same outcome, causing optimization to stall. We introduce a method that converts a small, fixed investment of SME judgment into a dense reward and combines it with retrieval-based metrics to form a hybrid,
    
[^21]: MOSS：用于监控和诊断搜索与推荐系统中语义故障的生产级框架

    MOSS: A Production Framework for Monitoring and Diagnosing Semantic Failures in Search and Recommendation Systems

    [https://arxiv.org/abs/2610.26902](https://arxiv.org/abs/2610.26902)

    MOSS是一个将LLM相关性评分嵌入持续监控循环的生产级框架，通过针对性采样和诊断性提示自动检测并归类推荐与搜索系统中的语义故障，已在大型零售商运行六个月并成功发现两个线上服务缺陷，还能离线量化修复效果（告警率相对降低76.4%）。

    

    在生产环境的推荐和搜索系统中检测语义故障十分困难：行为指标滞后数天，人工审核无法规模化，而且大多数LLM评估框架是为离线固定数据集设计的，而非用于持续的运行监控。我们提出了Model Sanity Service（MOSS），这是一个填补该运维空白的生产级框架，它将基于LLM的相关性评分嵌入到周期性监控循环中，结合针对性采样、策略感知提示、结构化告警，以及能将故障归类为可操作根因的诊断性后续提示。MOSS已在一家美国大型零售商的生产环境中运行超过六个月，覆盖推荐、搜索和优惠等多个业务面，并已发现了两个得到确认的服务缺陷。针对其中一个回填混合缺陷，MOSS离线量化了所提出修复方案的影响，显示告警率相对降低了76.4%。（原文摘要在此处被截断）

    arXiv:2610.26902v1 Announce Type: cross  Abstract: Detecting semantic failures in production recommendation and search systems is hard: behavioral metrics lag by days, manual review does not scale, and most LLM evaluation frameworks are designed for offline, fixed datasets rather than continuous operational monitoring. We present Model Sanity Service (MOSS), a production framework that addresses this operational gap by embedding LLM-based relevance scoring into a recurring monitoring loop with targeted sampling, strategy-aware prompting, structured alerting, and diagnostic follow-up prompting that categorizes failures into actionable root causes.   MOSS has been running in production at a large U.S. retailer for over six months across recommendation, search, and offer surfaces, and has surfaced two confirmed serving defects. For one of these, a backfill blending defect, MOSS quantified the impact of the proposed fix offline as a 76.4\% relative reduction in alert rate. On a 498-pair hu
    
[^22]: WeGR：微信工业搜索系统的统一生成式检索

    WeGR: Unified Generative Retrieval for Industrial Search Systems in WeChat

    [https://arxiv.org/abs/2610.16979](https://arxiv.org/abs/2610.16979)

    WeGR提出统一生成式检索框架，通过融合协同信号且避免码本坍缩的CoCo-VAE分词器，并解决强化学习中精确SID匹配带来的奖励稀疏问题，显著提升工业搜索的检索质量。

    

    生成式检索（GR）通过在自回归框架内统一候选生成过程，为工业搜索提供了一种有前景的范式。然而，其有效性受到两个关键挑战的限制。首先，现有的语义ID（SID）分词器难以在融入协同信号的同时保持有效的码本利用率，限制了其区分物品和实现高质量检索的能力。其次，现有的强化学习（RL）方法在精确SID匹配下存在严重的奖励稀疏问题，导致许多采样轨迹缺乏有效的学习信号，阻碍了偏好优化。为了解决这些挑战，我们提出了WeGR，一个由两个互补组件构成的统一生成式检索框架。具体而言，协同无坍缩分词器在缓解码本坍缩的同时，融入了查询到物品和物品到物品的协同信号……

    arXiv:2610.16979v1 Announce Type: cross  Abstract: Generative retrieval (GR) offers a promising paradigm for industrial search by unifying candidate generation within an autoregressive framework. However, its effectiveness is limited by two key challenges. First, existing Semantic ID (SID) tokenizers struggle to incorporate collaborative signals while maintaining effective codebook utilization, limiting their ability to distinguish items and achieve high-quality retrieval. Second, existing reinforcement learning (RL) methods suffer from severe reward sparsity under exact SID matching, leaving many sampled trajectories without effective learning signals and hindering preference optimization. To address these challenges, we propose WeGR, a unified generative retrieval framework consisting of two complementary components. Specifically, the Collaborative and Collapse-free Tokenizer (CoCo-VAE) incorporates query-to-item and item-to-item collaborative signals while mitigating codebook collap
    
[^23]: KuaFu：在十亿级规模上将长用户行为压缩为用户理解

    KuaFu: Compressing Long User Behavior into Understanding at Billion Scale

    [https://arxiv.org/abs/2609.31045](https://arxiv.org/abs/2609.31045)

    该论文提出KuaFu系统，在十亿用户规模上解决将长用户行为压缩为用户理解时序列过长与刷新吞吐量的双重工业瓶颈，并应对压缩过程可能引入的四类幻觉问题。

    

    对话式智能体、生成式推荐器和个性化广告都依赖一项核心能力：从原始行为中理解每一位用户。当前主流的工业实践是任务专用的：针对每个任务，从完整历史中提取相关子序列，并在其上训练专用模型。在生产环境中，这遇到了两个瓶颈。第一，即使经过过滤，单一任务的序列仍然极长：内容兴趣摘要需要读取每位用户的数百条内容，一旦序列化为提示文本就达数万个token。第二，用户画像需要例行刷新：每周覆盖十亿用户，总计约10万QPM，在固定的GPU预算下这设定了一个硬性的吞吐量下限。因此压缩是必不可少的，但截断或粗糙的压缩会悄然扭曲用户画像，引入四种幻觉类型（捏造、遗漏、日期错误归属、逻辑断裂），而由于无法评估压缩（原文在此处截断）……

    arXiv:2609.31045v1 Announce Type: cross  Abstract: Conversational agents, generative recommenders, and personalized advertising all rest on one capability: understanding each user from raw behavior. Prevailing industrial practice is task-specific: for each task, a relevant subsequence is extracted from the full history and a dedicated model trained on it. In production it hits two bottlenecks. First, even after filtering, a single-task sequence stays extremely long: content-interest summarization reads several hundred items per user, tens of thousands of tokens once serialized as prompt text. Second, profiles are refreshed routinely: a billion users weekly, roughly 100K QPM in aggregate, which under a fixed GPU budget sets a hard throughput floor. Compression is therefore mandatory, yet truncation or coarse compression can silently distort the profile, introducing four hallucination types (fabrication, omission, date misattribution, broken logic) that, with no way to evaluate the compr
    
[^24]: 更简单的最优决策树与自适应集合覆盖的对数近似算法

    Simpler Logarithmic Approximation Algorithms for the Optimal Decision Tree and Adaptive Set Cover

    [https://arxiv.org/abs/2604.12036](https://arxiv.org/abs/2604.12036)

    该论文提出了一种带有简化分析的新贪心算法，为最优决策树问题同时实现了最坏情形代价低于 $3.65\log_2 n$ 和平均情形代价低于 $7.3\log_2 n$ 的对数近似保证。

    

    我们研究在平均代价和最坏情形代价两种准则下，构造能够从给定基集中识别未知假设的决策树问题。最优决策树问题在文献中已被广泛研究，针对每种准则均已有 $\mathcal{O}(\log n)$ 近似保证的结果，其中也包括同时对两种准则的保证（此处 $n$ 为假设的数量）[ICML]。尽管现有的大多数算法采用了简单的贪心策略，但它们的分析依赖于复杂的论证。由于 ODT 是一个具有广泛应用的基础问题，我们希望开发既有效又易于理解的算法。我们提出了一种新的贪心算法并给出了简化分析，该算法同时实现了最坏情形代价下低于 $3.65\log_2 n$、平均情形代价下低于 $7.3\log_2 n$ 的近似因子。

    arXiv:2604.12036v4 Announce Type: replace-cross  Abstract: We study the problem of constructing a decision tree that identifies an unknown hypothesis from a given ground set, under both average- and worst-case cost. The Optimal Decision Tree (ODT) problem has been extensively studied in the literature, and $\mathcal{O}(\log n)$-approximation guarantees are known for each criterion, including a simultaneous guarantee for both (here $n$ is the number of hypotheses) [ICML]. Although most of the existing algorithm use simple greedy strategies, their analyses rely on intricate arguments. Since ODT is a fundamental problem with wide-ranging applications, we are interested in developing algorithms that are both effective and easy to understand.   We present a new greedy algorithm with a simplified analysis that simultaneously achieves approximation factors below $3.65\log_2 n$ for worst-case cost and $7.3\log_2 n$ for average-case cost. Our algorithm for the Optimal Decision Tree behaves gree
    
[^25]: 深度学习通过历史地图对齐实现城市变化特征分析

    Deep learning enables urban change profiling through alignment of historical maps

    [https://arxiv.org/abs/2602.02154](https://arxiv.org/abs/2602.02154)

    提出了一种基于深度学习的模块化框架，通过空间地图对齐、多时相目标提取和变化估计，实现了从大规模历史地图集合中进行细粒度、定量的城市变化分析。

    

    在现代对地观测技术出现之前，历史地图为长期城市变迁提供了独特的记录，也是观察城市不断演变的特征的一个视角。然而，由于空间错位、制图差异以及文档质量退化等问题，从历史地图系列中提取一致且细粒度的变化信息仍然具有挑战性，这使得大多数分析只能局限于小规模或定性的方法。我们提出了一种基于深度学习的框架，用于从大规模历史地图集合中进行细粒度的城市变化分析，该框架采用模块化设计，集成了空间地图对齐、多时相目标提取和变化估计。该框架将历史地图分析从临时的视觉比较转向系统的、定量的城市变化特征描述。实验证明了所提出的对齐和目标提取方法的鲁棒性能。

    arXiv:2602.02154v2 Announce Type: replace-cross  Abstract: Prior to modern Earth observation technologies, historical maps provide a unique record of long-term urban transformation and offer a lens on the evolving identity of cities. However, extracting consistent and fine-grained change information from historical map series remains challenging due to spatial misalignment, cartographic variation, and degrading document quality, limiting most analyses to small-scale or qualitative approaches. We propose a deep-learning-based framework for fine-grained urban change analysis from large collections of historical maps, built on a modular design that integrates spatial map alignment, multi-temporal object extraction, and change estimation. This framework shifts the analysis of historical maps from ad hoc visual comparison toward systematic, quantitative characterization of urban change. Experiments demonstrate the robust performance of the proposed alignment and object extraction methods. A
    
[^26]: 从爬取到发布：面向AI原生互联网的出版商可控网络架构

    From Crawled to Published: A Publisher-Controlled Web Architecture for an AI-Native Internet

    [https://arxiv.org/abs/2511.18354](https://arxiv.org/abs/2511.18354)

    提出AISTRAL架构，让出版商通过受策略控制的接口直接暴露语义单元，并借助类似DNS的语义解析器，使AI智能体能够在出版商可控的方式下进行内容的发现、检索与归属，从而解决AI时代网络的结构性缺陷。

    

    AI智能体而非人类正在成为网络的主要消费者，然而网络所传输的内容仍然是为人类浏览而组织的，出版商没有原生的方式来暴露其中的语义单元。因此，内容访问由少数AI提供商中介，它们单方面地对内容进行爬取、分块和索引，使出版商几乎无法控制其内容如何被表示、检索或归属。我们认为这是一个结构性问题，而非更好的分块或检索算法所能解决的：网络缺乏一个由出版商控制的语义发现与检索的原生层。我们提出AISTRAL，一个提供该层的AI原生互联网架构。出版商通过其自主控制的、受策略治理的显式接口暴露语义上有意义的单元（如段落、论断、表格），而一个网络原生的语义解析器——类似于DNS，但操作对象是含义而非位置——使智能体能够发现并（此处摘要被截断）

    arXiv:2511.18354v2 Announce Type: replace-cross  Abstract: AI agents, not humans, are becoming the web's primary consumers, yet the web still ships content structured for human browsing, with no native way for publishers to expose the semantic units within it. Access is therefore mediated by a handful of AI providers that crawl, chunk, and index content unilaterally, leaving publishers little control over how their content is represented, retrieved, or attributed. We argue this is a structural problem, not one better chunking or retrieval algorithms can solve. The web lacks a native layer for publisher-controlled semantic discovery and retrieval. We propose AISTRAL, an AI-Native Internet architecture that supplies this layer. Publishers expose semantically meaningful units (e.g., passages, claims, tables) through explicit, policy-governed interfaces they control, and a web-native semantic resolver, analogous to DNS but operating over meaning rather than location, lets agents discover r
    

