# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Autonomous Research for Open-Ended Problems: A Case Study on Telecom Ticket Retrieval](https://arxiv.org/abs/2609.13073) | 本文通过电信工单检索这一工业级案例研究，探讨了基于大语言模型的自主研究在开放式机器学习问题中的应用，发现其在狭窄的超参数优化中表现出色，但在处理具有表示、架构和训练数据生成自由度的开放式问题时仍存在局限性。 |
| [^2] | [MIMA: Multi-Interest Recommendation via Multi-Positive Exclusive Assignment](https://arxiv.org/abs/2609.12842) | 针对多兴趣推荐中的兴趣坍塌问题，本文提出MIMA框架，通过将同一请求中共现物品组成多正样本集合并独占分配给不同兴趣来提供更充分的监督，同时建模用户对各兴趣的激活强度，使不同兴趣通道的得分在推理时可比。 |
| [^3] | [Cognition on Graph: Navigating Massive Knowledge Space via Cognitive Cycles and Bidirectional Graph-Text Synergy](https://arxiv.org/abs/2609.12791) | 提出了CoG框架，一个受认知启发、无需训练的自适应知识探索方法，通过持续的“计划-探索-反思”认知循环和图-文本深度双向协同，实现海量异构知识库中的复杂推理导航。 |
| [^4] | [A Historical Corpus Is Not a Historical System: Auditing Hindsight Leakage in Stateful Data Discovery](https://arxiv.org/abs/2609.12766) | 该论文形式化了时间点（PIT）发现回放审计协议，揭示仅冻结历史语料库而保留未来交互记忆的“后见之明泄漏”会系统性虚高有状态数据发现系统的检索性能评估。 |
| [^5] | [Learning the Lake: Reliable Experience for Adaptive Data Product Discovery](https://arxiv.org/abs/2609.12754) | 该论文提出 SafeLake 方法，将操作性熟悉度与独立校准的产品证据相分离，利用演化发现记忆安全地缩减数据湖中数据产品发现的搜索工作量，并在部分基准上提升产品召回率。 |
| [^6] | [Personalized and Trust-Aware Health Recommendation Policies for a Construction Workplace](https://arxiv.org/abs/2609.12679) | 本文提出了一种信任感知的个性化健康推荐策略模型，其中工人健康随时间演变、信任受健康与推荐动态影响并进而影响推荐依从性，并通过健康触发阈值和推荐频率来刻画最优推荐策略。 |
| [^7] | [Preference-Drift-Aware Subsequence Learning and Hierarchical Context Fusion for Long-Sequence Generative Recommendation](https://arxiv.org/abs/2609.12556) | 该论文提出偏好漂移感知的子序列学习与层次化上下文融合方法，通过建模用户偏好的动态变化和跨子序列依赖关系，克服了长序列生成式推荐中全序列建模计算成本高且易受噪声干扰、目标感知检索上下文不完整的局限。 |
| [^8] | [OneLA: Scaling Linear-Attention Decoding to Large Beams in Generative Recommendation](https://arxiv.org/abs/2609.12399) | OneLA通过用单个共享的提示状态和紧凑的仅追加转换记录来表示所有束状态，将线性注意力解码高效扩展至生成式推荐的大束搜索场景，显著降低了内存和流量开销。 |
| [^9] | [ChronicleRec: Pre-training Temporally Anchored Tokens for Lifelong User Modeling](https://arxiv.org/abs/2609.12375) | ChronicleRec提出一种预训练-迁移框架，将超长用户行为序列一次性压缩为按时间顺序排列的时序锚定Token，通过新近感知的多粒度合并在保留近期行为细节的同时粗化远期行为，解决了工业推荐中长序列建模的计算开销大与时序结构缺失的问题。 |
| [^10] | [Recommendation Retrievers Need Verifiers: Universal Generative Reranking for Sequential Recommendations](https://arxiv.org/abs/2609.12270) | 本文提出一种轻量级生成式验证器，通过项目标识符token的似然性对候选进行评分，无需重新训练或替换检索器即可将检索列表深处的相关候选提升到实际消费的短名单中，从而提升序列推荐的Recall@$k$。 |
| [^11] | [EAR: Entity-Aware Partitioning Approach for Retrieval-Augmented Generation Development](https://arxiv.org/abs/2609.12268) | EAR提出了一种实体感知的语料库分割方法，通过从问题和选项中提取锚点、检索语料库中的局部窗口，在改进多项选择题问答的同时减少检索内容长度。 |
| [^12] | [Extracting Dataset Mentions in Forced Displacement and FCV Documents: A Weakly Supervised Framework with LLM-Based Label Refinement](https://arxiv.org/abs/2609.12107) | 该论文提出一种弱监督框架，利用在通用文献上训练的轻量级模型生成候选数据集提及，再由前沿大语言模型进行上下文审查与标签精炼，从而在无需大规模人工标注语料的情况下实现强迫流离失所和FCV领域文档中的数据集引用自动提取。 |
| [^13] | [Cortex: Content Analysis Support Software, a Resource for Qualitative Research](https://arxiv.org/abs/2609.11970) | 本文基于巴丹方法论并结合文献研究与半结构化访谈识别研究者实际需求，开发了面向学术研究者的内容分析支持网络应用Cortex，为定性研究提供高效的数据组织与分类工具。 |
| [^14] | [InitGen: Candidate Generation for Interaction Initiation in Intelligent Assistants](https://arxiv.org/abs/2609.11953) | InitGen 是一个部署于 OPPO 小布助手的候选生成框架，它联合生成一组候选查询，并通过基于用户活跃度和下游排序得分的加权偏好优化，解决了用户反馈不完整且难以归因到单个查询的难题。 |
| [^15] | [MemRetriever: Learning to Search, Reflect, and Retrieve from Long-Term Memory](https://arxiv.org/abs/2609.11951) | MemRetriever提出了一种智能体式长期记忆检索模型，将记忆访问建模为多步搜索过程，通过自适应选择并行搜索、串行搜索以及反思去噪来迭代检索与过滤证据，直到证据足以支持下游回答，从而克服传统静态top-k检索的局限。 |
| [^16] | [Who Are We Recommending To? Recommender Systems in the Agentic Web](https://arxiv.org/abs/2609.11945) | 该立场论文提出推荐范式正在分化：在可委托场景（如日常购物、旅行）中，推荐的主要接收者正从人类转向AI智能体，需要新的优化目标、交互协议和评估标准，而在体验性场景中人类仍是核心接收者。 |
| [^17] | [PinDCO: Whole-Page Aware Dynamic Creative Optimization at Scale](https://arxiv.org/abs/2609.11943) | Pinterest提出的PinDCO生产级动态创意优化系统，核心创新在于创意组件融合网络（CCFN），通过为图像、标题、布局等各创意组件配置专用塔式网络并融合评分，在严格延迟与成本约束下实现大规模、整页感知的广告创意与受众高效匹配。 |
| [^18] | [Position: Recommender Systems Should Move Beyond Platform-Centric Ranking toward Personal Agent-Mediated Recommendation](https://arxiv.org/abs/2609.11942) | 本文提出推荐系统应从平台中心的物品排序范式转向“个人代理中介推荐”（PAMR），即由用户侧的个人代理代表用户在分布式来源中发现、过滤、聚合和治理推荐证据，将控制重心从平台排序转移到用户侧证据中介。 |
| [^19] | [Guaranteeing Faithful Evidence Extraction in Speculative Retrieval-Augmented Generation](https://arxiv.org/abs/2609.10046) | 该论文提出约束混合解码（CHyD），将原本用于加速推理的投机解码架构重新用于强制执行硬解码约束，从而在投机式检索增强生成中保证抽取的证据逐忠实于检索到的上下文，满足安全关键领域对答案精确匹配的要求。 |
| [^20] | [Right Family, Wrong Skill: Benchmarking Risk Exposure in Agent Skill Retrieval](https://arxiv.org/abs/2606.10388) | 本文提出了SameCapRisk-Bench基准，专门研究智能体技能检索中“找到正确技能族但暴露错误同族技能”的风险暴露失败，并提供了1,190个技能-风险单元和1,686个评估查询案例。 |
| [^21] | [CoHyDE: Iterative Co-Training of LLM Rewriter & Dense Encoder for Tool Retrieval](https://arxiv.org/abs/2605.29271) | 提出CoHyDE方法，通过迭代协同训练将LLM查询重写器与稠密编码器作为一个共同演化的整体系统，融合编码器微调与HyDE查询扩展二者的优势，显著提升LLM智能体在大型API目录上的工具检索性能。 |
| [^22] | [Total Recall QA: A Verifiable Evaluation Suite for Deep Research Agents](https://arxiv.org/abs/2603.18516) | 该论文提出了Total Recall QA评估套件，通过基于结构化知识库构建具有唯一答案且可精确验证的查询，首次为深度研究智能体提供了满足全部评估要求的可验证评估框架。 |
| [^23] | [Membership Inference Attacks on Recommender System: A Survey](https://arxiv.org/abs/2509.11080) | 本文是一篇关于推荐系统成员推断攻击的综述，指出传统成员推断攻击因后验概率不可见而不适用于推荐系统，并系统分析了此类攻击对用户隐私造成的泄露风险。 |

# 详细

[^1]: 开放式问题的自主研究：电信工单检索案例研究

    Autonomous Research for Open-Ended Problems: A Case Study on Telecom Ticket Retrieval

    [https://arxiv.org/abs/2609.13073](https://arxiv.org/abs/2609.13073)

    本文通过电信工单检索这一工业级案例研究，探讨了基于大语言模型的自主研究在开放式机器学习问题中的应用，发现其在狭窄的超参数优化中表现出色，但在处理具有表示、架构和训练数据生成自由度的开放式问题时仍存在局限性。

    

    基于大语言模型（LLM）的系统在问题解决和编程能力方面的最新突破推动了“AI for Science”范式的进展，有望在机器学习（ML）研究中取代人类角色。然而，尽管已有多个完全自主的端到端机器学习研究框架被提出，它们的成功实施通常仅限于搜索空间狭窄的问题，如语言建模或生物医学机器学习基准测试。在本文中，我们通过一个案例研究探索了如何将自主研究应用于解决开放式、工业级的机器学习问题：电信工单检索，这是一个在表示、架构和训练数据生成方面具有自由度的开放式任务。我们发现，使用商业和开源智能体对开放式问题进行自主研究既展现出前景也存在局限性：虽然自主研究在狭窄的超参数优化中表现出色，但它在……（原文在此处截断）

    arXiv:2609.13073v1 Announce Type: cross  Abstract: Recent breakthroughs in LLM-based systems and their abilities in problem solving and coding have allowed progress in the AI for Science paradigm, potentially replacing human roles in machine learning (ML) research. However, while several frameworks of fully autonomous end-to-end ML research have been proposed, successful implementations of them are often limited to problems with narrow search spaces, like language modeling or biomedical ML benchmarks. In this paper, we explore how autonomous research can be adapted to solve open-ended, industry-grade ML problems, by considering a case study: telecom ticket retrieval, an open-ended task with degrees of freedom in representation, architecture, and training data generation. We discover that autonomous research for open-ended problems with commercial and open-source agents shows both promise and limitations: while autonomous research can excel in narrow hyperparameter optimization, it lack
    
[^2]: MIMA：基于多正样本独占分配的多兴趣推荐

    MIMA: Multi-Interest Recommendation via Multi-Positive Exclusive Assignment

    [https://arxiv.org/abs/2609.12842](https://arxiv.org/abs/2609.12842)

    针对多兴趣推荐中的兴趣坍塌问题，本文提出MIMA框架，通过将同一请求中共现物品组成多正样本集合并独占分配给不同兴趣来提供更充分的监督，同时建模用户对各兴趣的激活强度，使不同兴趣通道的得分在推理时可比。

    

    多兴趣推荐通过多个兴趣向量来表示每个用户，以实现细粒度的候选匹配，但它常常面临兴趣坍塌问题，即学习到的兴趣收敛为相似的表示。我们指出主流的单正样本范式是造成这一问题的重要因素之一。由于每个实例只提供一个正样本物品，各兴趣被独立优化，可能导致同一个最佳匹配兴趣被反复更新以拟合不同的正样本，而其他兴趣则监督不足。此外，现有方法很少建模用户对每个兴趣的激活强度，导致在推理阶段不同兴趣通道的得分不可比。为解决这些问题，我们提出MIMA，一个基于多正样本独占分配构建的多兴趣推荐框架。MIMA将同一请求中共现的物品分组为一个正样本集合，生成互补的……

    arXiv:2609.12842v1 Announce Type: new  Abstract: Multi-interest recommendation represents each user with multiple interest vectors for fine-grained candidate matching, yet it often suffers from interest collapse, where the learned interests converge to similar representations. We highlight the prevailing single-positive paradigm as one important factor behind this issue. Since each instance provides only one positive item, intents are optimized independently, potentially causing the same best-matching interest to be repeatedly updated toward different positives while leaving the others under-supervised. Moreover, existing methods rarely model how strongly a user activates each interest, leaving scores from different interest channels incomparable at inference. To address these problems, we propose MIMA, a Multi-Interest recommendation framework built on Multi-positive exclusive Assignment. MIMA groups items co-occurring within the same request into a positive set, generates complementa
    
[^3]: 图上认知：通过认知循环与图-文本双向协同导航海量知识空间

    Cognition on Graph: Navigating Massive Knowledge Space via Cognitive Cycles and Bidirectional Graph-Text Synergy

    [https://arxiv.org/abs/2609.12791](https://arxiv.org/abs/2609.12791)

    提出了CoG框架，一个受认知启发、无需训练的自适应知识探索方法，通过持续的“计划-探索-反思”认知循环和图-文本深度双向协同，实现海量异构知识库中的复杂推理导航。

    

    检索增强生成（RAG）已使大型语言模型（LLM）能够处理知识密集型任务。然而，在全局、异构的知识库（大规模知识图谱和文本语料库）中进行导航以完成复杂推理仍然是一个挑战。现有方法通常采用反应式、图驱动的探索策略，盲目遵循图拓扑结构，而无法适应问题上下文或不断演化的探索进度，并且缺乏图与文本之间的深度双向协同。为了解决这些局限性，我们提出了CoG（Cognition on Graph），一个受认知启发、无需训练的自适应知识探索框架。受人类解决问题方式的启发，CoG执行持续的“计划-探索-反思”循环，主动制定调查计划，执行双源检索，并动态反思进度以调整策略。至关重要的是，它建立了深度的双向图与文本协同机制。

    arXiv:2609.12791v1 Announce Type: new  Abstract: Retrieval-Augmented Generation (RAG) has empowered Large Language Models (LLMs) to tackle knowledge-intensive tasks. However, navigating global, heterogeneous knowledge bases (large-scale knowledge graphs and text corpora) for complex reasoning remains a challenge. Existing methods typically employ reactive, graph-driven exploration strategies, which blindly follow graph topology without adapting to the question context or evolving exploration progress, and lack deep bidirectional synergy between graph and text. To address these limitations, we propose CoG (Cognition on Graph), a cognitive-inspired, training-free framework for adaptive knowledge exploration. Drawing inspiration from human problem-solving, CoG performs a continuous plan-explore-reflect cycle, where it proactively formulates investigation plans, performs dual-source retrieval, and dynamically reflects on progress to adjust strategies. Crucially, it establishes deep bidirec
    
[^4]: 历史语料库并非历史系统：审计有状态数据发现中的后见之明泄漏

    A Historical Corpus Is Not a Historical System: Auditing Hindsight Leakage in Stateful Data Discovery

    [https://arxiv.org/abs/2609.12766](https://arxiv.org/abs/2609.12766)

    该论文形式化了时间点（PIT）发现回放审计协议，揭示仅冻结历史语料库而保留未来交互记忆的“后见之明泄漏”会系统性虚高有状态数据发现系统的检索性能评估。

    

    离线回放应当估计发现系统在历史时间点上所能检索到的内容，然而冻结语料库会使交互记忆处于不受约束的状态。我们通过历史状态 $(D_t, \theta_t, M_{<i})$ 形式化了时间点（PIT）发现，并引入了一种仅改变记忆可用性的成对回放方法。该协议从仅行为的轨迹中构建 PIT 视图和全流 Future 视图，并使用时间违反率对选定条目进行审计。在三个表格-文本领域、两种流式机制、两个检索器和五个随机种子（共216,000行）上的实验显示，Future 视图使 Asset Recall@100 虚高了2.62-5.24个百分点；全部12个成对区间均排除了零差异。在使用仅行为轨迹记忆时，PIT 的表现逊于无记忆的 Stateless 条件；而 Future 掩盖了其中32.7-48.4%的危害。对于模拟的正反馈缓存，PIT 比 Stateless 提升了4.65-18.96个百分点，而 Future 又额外提升了4.11-9.58个百分点。在五个带时间戳的 FreshStack 主题上，（摘要在此处截断）

    arXiv:2609.12766v1 Announce Type: new  Abstract: Offline replay should estimate what a discovery system could retrieve at a historical point, yet freezing the corpus leaves interaction memory unconstrained. We formalize point-in-time (PIT) discovery through historical state $(D_t, \theta_t, M_{< i})$ and introduce a paired replay that changes only memory availability. The protocol constructs PIT and full-stream Future views from behavior-only traces and audits selected entries with a Temporal Violation Rate. Across three table-text domains, two stream regimes, two retrievers, and five seeds (216,000 rows), Future inflated Asset Recall@100 by 2.62-5.24 points; all 12 paired intervals excluded zero. With behavior-only trace memory, PIT underperformed the no-memory Stateless condition; Future masked 32.7-48.4% of that harm. For a simulated positive-feedback cache, PIT added 4.65-18.96 points over Stateless while Future added another 4.11-9.58 points. On five timestamped FreshStack topics,
    
[^5]: 学习数据湖：面向自适应数据产品发现的可靠经验

    Learning the Lake: Reliable Experience for Adaptive Data Product Discovery

    [https://arxiv.org/abs/2609.12754](https://arxiv.org/abs/2609.12754)

    该论文提出 SafeLake 方法，将操作性熟悉度与独立校准的产品证据相分离，利用演化发现记忆安全地缩减数据湖中数据产品发现的搜索工作量，并在部分基准上提升产品召回率。

    

    arXiv:2609.12754v1 公告类型：新论文 摘要：数据产品发现在工作负载重复访问相关产品和区域时，仍需搜索整个数据湖。重复性允许收缩搜索范围，但相似性不能作为选择搜索路径的依据，因为遗漏任何一个资产都会使合取型产品失效。我们研究在何种情况下服务经验能够安全地减少这项工作。演化发现记忆在固定区域索引之上记录带来源标签的查询—产品—区域证据。SafeLake 将决定搜索范围大小的操作性熟悉度，与决定搜索位置的独立校准产品证据分离开来。固定探针比较使 SafeLake 与仅熟悉度方法之间的自适应预算保持恒定。在 TAT-QA 数据集上，产品引导使产品召回率提升 0.072；ConvFinQA 上已解析映射无增益，而 HybridQA 敏感性分析表明在 Full R@100 上仅熟悉度方法更优。仅轨迹、缺失以及虚假反馈暴露了映射引导的边界条件，而范围审计一致性……

    arXiv:2609.12754v1 Announce Type: new  Abstract: Data-product discovery searches a full lake even when workloads revisit related products and regions. Repetition permits contracted search, but similarity cannot justify a route because one omitted asset invalidates a conjunctive product. We study when serving experience can safely reduce this work. Evolving Discovery Memory records source-labelled query--product--region evidence above a fixed regional index. SafeLake separates operational familiarity, which determines how much to search, from independently calibrated product evidence, which determines where to search. The fixed-probe comparison holds the adaptive budget constant between SafeLake and Familiarity-only. On TAT-QA, product steering raises Product Recall by 0.072; ConvFinQA shows no resolved map gain, while the HybridQA sensitivity favors Familiarity-only in Full R@100. Trace-only, missing, and false feedback expose boundaries on map steering, while scope-audit agreement can
    
[^6]: 面向建筑工地的个性化且信任感知的健康推荐策略

    Personalized and Trust-Aware Health Recommendation Policies for a Construction Workplace

    [https://arxiv.org/abs/2609.12679](https://arxiv.org/abs/2609.12679)

    本文提出了一种信任感知的个性化健康推荐策略模型，其中工人健康随时间演变、信任受健康与推荐动态影响并进而影响推荐依从性，并通过健康触发阈值和推荐频率来刻画最优推荐策略。

    

    建筑工人面临诸如疲劳、热应激以及其他高体力消耗工作条件等工作场所风险，这些风险可能对他们的健康和安全产生负面影响。尽管监测这些风险很重要，但及时且个性化的健康干预同样必要，以帮助防止对工人福祉和生产力的负面影响。为此，本文提出了一个模型，用于捕捉信任感知的健康推荐系统与健康和信任敏感度各不相同的工人之间的交互。具体而言，在我们提出的动态模型中，工人的健康状态随时间演变，工人的信任受健康和推荐动态的共同影响，而信任反过来又会影响其对未来推荐的依从性。基于该模型，我们对推荐策略进行了刻画，包括基于健康状态的推荐触发阈值和推荐频率。我们采用基于模型的短时程控制和……

    arXiv:2609.12679v1 Announce Type: new  Abstract: Construction workers face workplace risks such as fatigue, heat stress, and other physically demanding conditions that can negatively affect their health and safety. Although monitoring these risks is important, timely and personalized health interventions are also needed to help prevent negative impacts on workers' well-being and productivity. To this end, in this paper, we propose a model to capture the interactions between a trust-aware health recommender system and workers who differ in health and trust sensitivity. Specifically, in our proposed dynamic model, worker health evolves over time, worker trust is affected by both health and recommendation dynamics, and trust in turn affects compliance with future recommendations. Given this model, we characterize the recommender policy, including a health-based recommendation triggering threshold and the recommendation frequency. We do so using both model-based short-horizon control and m
    
[^7]: 面向长序列生成式推荐的偏好漂移感知子序列学习与层次化上下文融合方法

    Preference-Drift-Aware Subsequence Learning and Hierarchical Context Fusion for Long-Sequence Generative Recommendation

    [https://arxiv.org/abs/2609.12556](https://arxiv.org/abs/2609.12556)

    该论文提出偏好漂移感知的子序列学习与层次化上下文融合方法，通过建模用户偏好的动态变化和跨子序列依赖关系，克服了长序列生成式推荐中全序列建模计算成本高且易受噪声干扰、目标感知检索上下文不完整的局限。

    

    长序列生成式推荐方法通过自回归地建模用户交互序列来生成下一物品的表示。现有方法大致分为两类：高效的全序列建模和目标感知的上下文检索。我们的实验揭示，随着序列长度的增加，前者的计算成本持续增长，而其准确率提升迅速饱和，甚至因噪声而退化；后者虽然缩短了输入序列，却容易受到语义一致但偏好不一致的噪声以及不完整上下文的影响。这两种范式在处理历史信息时都忽略了用户偏好的动态变化和跨子序列的依赖关系，从而限制了准确性和效率。为解决这些问题，我们提出了一种面向长序列生成式推荐的偏好漂移感知子序列学习与层次化上下文融合方法（注：原文摘要在此处截断）。

    arXiv:2609.12556v1 Announce Type: new  Abstract: Long-sequence generative recommendation methods autoregressively model the user's interaction sequence to generate the next-item representation. Existing methods generally fall into two categories: efficient full-sequence modeling and target-aware context retrieval. Our experiments reveal that as the sequence length increases, the former incurs steadily growing computational cost while its accuracy gains quickly saturate and even degrade due to noise; the latter, though shortening the input sequence, is susceptible to noise that is semantically consistent yet preference-inconsistent, as well as to incomplete contexts. Both paradigms ignore the dynamic changes of user preferences and the cross-subsequence dependencies when handling historical information, thereby limiting accuracy and efficiency. To address these issues, we propose a preference-drift-aware subsequence learning and hierarchical context fusion for long-sequence generative r
    
[^8]: OneLA：将线性注意力解码扩展至生成式推荐中的大束搜索

    OneLA: Scaling Linear-Attention Decoding to Large Beams in Generative Recommendation

    [https://arxiv.org/abs/2609.12399](https://arxiv.org/abs/2609.12399)

    OneLA通过用单个共享的提示状态和紧凑的仅追加转换记录来表示所有束状态，将线性注意力解码高效扩展至生成式推荐的大束搜索场景，显著降低了内存和流量开销。

    

    生成式推荐（GR）依赖于大束解码来生成数百个候选项，这为循环线性注意力带来了新的扩展挑战。现有的线性注意力服务系统要么为每个束物化完整的循环状态，要么反复重放共享历史，从而产生巨大的内存和流量开销。为解决这一问题，我们提出了OneLA，一个利用GR工作负载中共享提示和短发散后缀特性的线性注意力解码框架。具体而言，OneLA使用单个由提示派生的共享状态以及紧凑的、仅追加的发散转换记录来表示所有束状态。基于这种表示，OneLA在每个解码步骤中仅计算所需的状态信息，而无需为每个束重建完整的循环状态。此外，OneLA采用轻量级的祖先索引来追踪构成每个束历史的转换记录，允许……

    arXiv:2609.12399v1 Announce Type: cross  Abstract: Generative recommendation (GR) relies on large-beam decoding to generate hundreds of candidate items, creating a new scaling challenge for recurrent linear attention. Existing linear attention serving systems either materialize a full recurrent state for every beam or repeatedly replay shared history, incurring substantial memory and traffic overhead. To address this, we present OneLA, a linear-attention decoding framework that exploits the shared prompt and short divergent suffixes of GR workloads. Specifically, OneLA represents all beam states using a single shared prompt-derived state and compact, append-only records of their divergent transitions. Using this representation, OneLA computes only the state information required at each decoding step, without reconstructing a full recurrent state for every beam. Furthermore, OneLA uses a lightweight ancestry index to track the transition records that make up each beam's history, allowin
    
[^9]: ChronicleRec：面向终身用户建模的时序锚定Token预训练

    ChronicleRec: Pre-training Temporally Anchored Tokens for Lifelong User Modeling

    [https://arxiv.org/abs/2609.12375](https://arxiv.org/abs/2609.12375)

    ChronicleRec提出一种预训练-迁移框架，将超长用户行为序列一次性压缩为按时间顺序排列的时序锚定Token，通过新近感知的多粒度合并在保留近期行为细节的同时粗化远期行为，解决了工业推荐中长序列建模的计算开销大与时序结构缺失的问题。

    

    建模超长用户行为序列对工业推荐和在线广告至关重要，然而直接将数千条历史行为输入排序模型在计算上难以承受，而截断则会丢失长程信号。现有的终身兴趣方法为每个候选物品检索与目标相关的行为，将长序列建模与候选打分耦合在一起，并带来重复的在线计算成本。近期的目标无关压缩方法虽然支持缓存用户摘要，但通常在序列末尾附加查询Token并使用双向编码，产生无序且冗余的摘要，忽视了时序结构。我们提出ChronicleRec，这是一个预训练-迁移框架，可将超长行为序列一次性压缩为按时间顺序排列的Chronicle Token集合。ChronicleRec采用新近感知的多粒度合并策略，在保留近期行为的同时对远期行为进行粗化压缩。

    arXiv:2609.12375v1 Announce Type: new  Abstract: Modeling ultra-long user behavior sequences is crucial for industrial recommendation and online advertising, yet directly feeding thousands of historical actions into ranking models is computationally prohibitive, while truncation discards long-range signals. Existing lifelong-interest methods retrieve target-relevant behaviors for each candidate, coupling long-sequence modeling with candidate scoring and repeated online cost. Recent target-independent compression methods enable cached user summaries, but often append query tokens at the sequence end and use bidirectional encoding, producing unordered and redundant summaries that overlook temporal structure. We propose ChronicleRec, a pre-train-and-transfer framework that compresses an ultra-long behavior sequence once into a chronologically ordered set of Chronicle Tokens. ChronicleRec applies a recency-aware multi-granularity merge, preserving recent behaviors while coarsening distant 
    
[^10]: 推荐检索器需要验证器：面向序列推荐的通用生成式重排序

    Recommendation Retrievers Need Verifiers: Universal Generative Reranking for Sequential Recommendations

    [https://arxiv.org/abs/2609.12270](https://arxiv.org/abs/2609.12270)

    本文提出一种轻量级生成式验证器，通过项目标识符token的似然性对候选进行评分，无需重新训练或替换检索器即可将检索列表深处的相关候选提升到实际消费的短名单中，从而提升序列推荐的Recall@$k$。

    

    多阶段系统中的第一阶段推荐器会生成一个经过排序的候选列表，其中有限的前缀部分会被传递给下游的排序器。由于每个被传递的项目都需要经过更昂贵的排序阶段处理，因此这份短名单不能任意扩大。因此，第一阶段的目标是在被传递的前缀内实现对相关项目的高覆盖率，通常以Recall@$k$来衡量。相关项目可能存在于检索列表的更深处，但却不在实际被消费的较短前缀之中。本文研究了事后验证方法，在不重新训练或替换检索器的前提下，将此类候选提升到实际消费的短名单中。我们为检索模型引入了一种轻量级的生成式验证器。给定检索器状态和候选项目，验证器通过该项目标识符token的似然性对其进行评分。该验证器通过next-token交叉熵进行事后训练，无需采样负样本……

    arXiv:2609.12270v1 Announce Type: new  Abstract: First-stage recommenders in multi-stage systems produce a ranked candidate list from which a limited prefix is forwarded to downstream rankers. Because each forwarded item must be processed by more expensive ranking stages, this shortlist cannot be arbitrarily large. The first-stage objective is therefore high coverage of relevant items within the forwarded prefix, commonly measured by Recall@$k$. A relevant item may be available deeper in the retrieved list but absent from the shorter prefix that is actually consumed. This paper studies post-hoc verification for promoting such candidates into the consumed shortlist without retraining or replacing the retriever.   We introduce a lightweight generative verifier for retrieval models. Given a retriever state and a candidate item, the verifier scores the item through the likelihood of its identifier tokens. It is trained post hoc with next-token cross entropy, requires no sampled negatives o
    
[^11]: EAR：面向检索增强生成开发的实体感知分割方法

    EAR: Entity-Aware Partitioning Approach for Retrieval-Augmented Generation Development

    [https://arxiv.org/abs/2609.12268](https://arxiv.org/abs/2609.12268)

    EAR提出了一种实体感知的语料库分割方法，通过从问题和选项中提取锚点、检索语料库中的局部窗口，在改进多项选择题问答的同时减少检索内容长度。

    

    检索增强生成（RAG）可以改进知识密集型问答，但第一个设计选择往往容易被忽视：源语料库应如何被分割为可检索单元？固定大小的分块常常返回与问题关系仅为隐式的长段落。我们提出了EAR，一种面向多项选择题问答（MCQA）的实体感知分割方法。EAR从问题、答案选项和语料库中提取规范化的表面锚点；检索语料库中匹配锚点周围的局部窗口；并可通过抽取式摘要附加更大的父级段落。我们在一个经过清洗的MMLU风格子集上评估EAR，该子集由自动语料库支持启发式方法筛选出153个问题，并使用经过去污染的公共教科书文本。在与Mistral、Gemma和DeepSeek进行的相同协议top-k=3和top-k=8扫描实验中，EAR实体窗口方法减少了检索词的数量……

    arXiv:2609.12268v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) can improve knowledge-intensive question answering, but the first design choice is easy to overlook: how should the source corpus be partitioned into retrievable units? Fixed-size chunks often return long passages whose relation to the question is only implicit. We introduce EAR, an Entity-Aware Partitioning approach for multiple-choice question answering (MCQA). EAR extracts normalized surface anchors from the question, answer options, and corpus; retrieves local windows around matching corpus anchors; and can attach a larger parent passage through an extractive summary. We evaluate EAR on a cleaned Massive Multitask Language Understanding (MMLU)-style subset of 153 questions selected by an automatic corpus-support heuristic and using decontaminated public textbook text. Across same-protocol top-k = 3 and top-k = 8 sweeps with Mistral, Gemma, and DeepSeek, EAR entity-window reduces retrieved words by
    
[^12]: 在强迫流离失所与脆弱、冲突和暴力（FCV）文档中提取数据集提及：一种基于大语言模型标签精炼的弱监督框架

    Extracting Dataset Mentions in Forced Displacement and FCV Documents: A Weakly Supervised Framework with LLM-Based Label Refinement

    [https://arxiv.org/abs/2609.12107](https://arxiv.org/abs/2609.12107)

    该论文提出一种弱监督框架，利用在通用文献上训练的轻量级模型生成候选数据集提及，再由前沿大语言模型进行上下文审查与标签精炼，从而在无需大规模人工标注语料的情况下实现强迫流离失所和FCV领域文档中的数据集引用自动提取。

    

    发展和人道主义组织生产并支持调查、行政登记册和其他数据资源，以为研究、政策和运营提供信息，但系统地识别这些数据集在何处被引用仍然困难。此类引用分散在研究论文、项目文件、人道主义报告和其他非结构化文本中，这限制了追踪数据使用以及识别数据可用性或传播方面潜在缺口的能力。我们提出了一个弱监督框架，用于将数据集提取适配到强迫流离失所和脆弱、冲突与暴力（FCV）文档，而无需首先构建大型人工标注训练语料库。一个在通用研究文献上训练的轻量级模型从无标注的领域文档中生成候选数据集提及，然后由前沿大语言模型（LLM）在上下文中进行审查，验证或拒绝候选提及并纠正……

    arXiv:2609.12107v1 Announce Type: new  Abstract: Development and humanitarian organizations produce and support surveys, administrative registries, and other data resources to inform research, policy, and operations, yet systematically identifying where these datasets are referenced remains difficult. Such references are dispersed across research papers, project documents, humanitarian reports, and other unstructured text, limiting both the ability to trace data use and to identify potential gaps in data availability or dissemination. We present a weakly supervised framework for adapting dataset extraction to forced displacement and Fragile, Conflict, and Violence (FCV) documents without first constructing a large manually labeled training corpus. A lightweight model trained on general research literature generates candidate dataset mentions from unlabeled domain documents, which a frontier large language model (LLM) reviews in context, validating or rejecting candidates and correcting
    
[^13]: Cortex：内容分析支持软件——定性研究的资源

    Cortex: Content Analysis Support Software, a Resource for Qualitative Research

    [https://arxiv.org/abs/2609.11970](https://arxiv.org/abs/2609.11970)

    本文基于巴丹方法论并结合文献研究与半结构化访谈识别研究者实际需求，开发了面向学术研究者的内容分析支持网络应用Cortex，为定性研究提供高效的数据组织与分类工具。

    

    定性研究在人文与社会科学中被广泛应用，其特点在于通过对意义和语境的解读来深入理解现象。在各类定性数据分析方法中，内容分析作为一种成熟的技术尤为突出，它能够对文本内容进行系统的描述与阐释。然而，随着数据量的增加，数据组织、阅读和分类所需的时间成为一项重大挑战，可能延误研究进程。因此，本工作旨在基于巴丹（Bardin）方法论，开发一款面向学术研究者的内容分析支持网络应用程序。该研究采用混合方法，将有关内容分析的文献研究与对四位资深研究者的半结构化访谈相结合，以识别实际需求和具体要求。基于这些输入，Cortex软件得以开发。

    arXiv:2609.11970v1 Announce Type: cross  Abstract: Qualitative research is widely used in the human and social sciences, characterized by a deep understanding of phenomena through the interpretation of meanings and contexts. Among qualitative data analysis methods, content analysis stands out as a consolidated technique, which allows for the systematic description and interpretation of textual contents. However, as data volume increases, the time required for organization, reading, and categorization becomes a significant challenge, potentially delaying research development. Therefore, this work aimed to develop a web application to support content analysis, based on Bardin's methodology, targeted at academic researchers. The methodology adopted a mixed approach, combining bibliographic research on content analysis with semi-structured interviews with four experienced researchers, aiming to identify real needs and requirements. Based on these inputs, the Cortex software was developed t
    
[^14]: InitGen：智能助手中交互发起的候选生成

    InitGen: Candidate Generation for Interaction Initiation in Intelligent Assistants

    [https://arxiv.org/abs/2609.11953](https://arxiv.org/abs/2609.11953)

    InitGen 是一个部署于 OPPO 小布助手的候选生成框架，它联合生成一组候选查询，并通过基于用户活跃度和下游排序得分的加权偏好优化，解决了用户反馈不完整且难以归因到单个查询的难题。

    

    交互发起是指在用户打开智能助手、尚未表达当前会话任何意图时，向其呈现多个候选查询。在生产环境中，候选生成需要结合动态上下文，并在严格的延迟预算内生成全部候选。此外，从用户反馈中学习也很困难，因为生成器通常产生的候选数量多于最终展示的数量。经过下游的过滤与排序，只有一部分候选会展示给用户，因此观察到的反馈是不完整的，无法可靠地归因到单个查询上。我们提出了 InitGen，一个候选生成框架，已部署于 OPPO 小布助手的交互发起流程中。InitGen 联合生成一组候选查询，并通过加权偏好优化使生成的候选集合与用户反馈对齐，其中样本权重来源于用户活跃度和下游排序得分。

    arXiv:2609.11953v1 Announce Type: new  Abstract: Interaction initiation refers to presenting multiple candidate queries when a user opens an intelligent assistant before expressing any intent for the current session. In production, candidate generation incorporates dynamic context and produces all candidates within a strict latency budget. Learning from user feedback is also difficult since the generator usually produces more candidates than are finally displayed. After downstream filtering and ranking, only a subset is exposed to users, so the observed feedback is partial and cannot be reliably assigned to individual queries. We present InitGen, a framework for candidate generation that is deployed in the interaction initiation pipeline of OPPO's Xiaobu Assistant. InitGen generates a set of candidate queries jointly and aligns the generated set with user feedback through weighted preference optimization. The sample weights are derived from user activity and downstream ranking scores. 
    
[^15]: MemRetriever：学习搜索、反思并从长期记忆中检索

    MemRetriever: Learning to Search, Reflect, and Retrieve from Long-Term Memory

    [https://arxiv.org/abs/2609.11951](https://arxiv.org/abs/2609.11951)

    MemRetriever提出了一种智能体式长期记忆检索模型，将记忆访问建模为多步搜索过程，通过自适应选择并行搜索、串行搜索以及反思去噪来迭代检索与过滤证据，直到证据足以支持下游回答，从而克服传统静态top-k检索的局限。

    

    长期记忆能够支持个性化智能体，但其价值取决于能否在合适的时机检索到正确的证据。大多数记忆系统采用静态的top-k检索方式：发起一次查询，返回固定数量的记忆，并直接传递给下游模型。这种方式可能会遗漏分布在多个会话中的证据，引入无关内容，并浪费上下文，尤其是对于多跳、时间性和知识更新类问题。我们提出了MemRetriever，一种将记忆访问视为多步搜索过程的智能体检索模型。在每一步中，MemRetriever会对当前证据进行推理，并选择并行搜索以进行广泛探索、串行搜索以进行有针对性的补充，或通过反思与去噪进行过滤和证据评估。当保留的证据足以支持下游回答时，它会停止检索。我们构建了ReAct风格的搜索-记忆轨迹用于监督热启动训练，并进一步（摘要在此处被截断）

    arXiv:2609.11951v1 Announce Type: new  Abstract: Long-term memory enables personalized agents, but its value depends on retrieving the right evidence at the right time. Most memory systems use static top-k retrieval: they issue one query, return a fixed number of memories, and pass them directly to a downstream model. This approach can miss evidence distributed across sessions, introduce irrelevant content, and waste context, especially for multi-hop, temporal, and knowledge-update questions. We present MemRetriever, an agentic retrieval model that treats memory access as a multi-step search process. At each step, MemRetriever reasons over the current evidence and selects parallel search for broad exploration, serial search for targeted completion, or reflection and denoising for filtering and evidence assessment. It stops when the retained evidence is sufficient for downstream answering. We construct ReAct-style search-memory trajectories for supervised warm-start training and further
    
[^16]: 我们在向谁推荐？智能体网络中的推荐系统

    Who Are We Recommending To? Recommender Systems in the Agentic Web

    [https://arxiv.org/abs/2609.11945](https://arxiv.org/abs/2609.11945)

    该立场论文提出推荐范式正在分化：在可委托场景（如日常购物、旅行）中，推荐的主要接收者正从人类转向AI智能体，需要新的优化目标、交互协议和评估标准，而在体验性场景中人类仍是核心接收者。

    

    二十年来，推荐系统的设计一直基于这样一个假设：由人类直接消费每一条推荐——接收、理解并据此采取行动。然而，由大语言模型驱动的AI智能体的出现正在挑战这一假设。在新兴的“智能体网络”中，自主智能体日益代表用户行事，例如浏览、比较、谈判和执行交易，这引出了一个核心问题：推荐的接收者究竟是谁？在这篇立场论文中，我们论证推荐范式正在经历一场分化。在可委托场景中，例如日常购物、旅行以及受限的交易任务，推荐的主要操作性消费者正从人类转向智能体，这需要全新的优化目标、交互协议和评估标准。而在体验性场景中，例如娱乐、艺术以及其他主观性或高风险的（摘要原文在此处截断）

    arXiv:2609.11945v1 Announce Type: new  Abstract: For two decades, recommender systems have been designed under the assumption that a human directly consumes each recommendation: receiving, interpreting, and acting upon it. The emergence of AI agents powered by large language models challenges this assumption. In the emerging Agentic Web [ 28 ], autonomous agents increasingly act on behalf of users, e.g., browsing, comparing, negotiating, and executing transactions, raising a central question: who is the receiver of a recommendation? In this position paper, we argue that the recommendation paradigm is undergoing a bifurcation. In delegable contexts, such as routine purchases, travel, and constrained transactional tasks, the primary operational consumer of recommendations is shifting from the human to the agent, requiring new optimization objectives, interaction protocols, and evaluation criteria. In experiential contexts, such as entertainment, art, and other subjective or high-stakes c
    
[^17]: PinDCO：面向整页感知的大规模动态创意优化

    PinDCO: Whole-Page Aware Dynamic Creative Optimization at Scale

    [https://arxiv.org/abs/2609.11943](https://arxiv.org/abs/2609.11943)

    Pinterest提出的PinDCO生产级动态创意优化系统，核心创新在于创意组件融合网络（CCFN），通过为图像、标题、布局等各创意组件配置专用塔式网络并融合评分，在严格延迟与成本约束下实现大规模、整页感知的广告创意与受众高效匹配。

    

    生成式人工智能的最新进展极大地加速了高质量广告创意的制作，显著增加了每个广告活动中的候选变体数量。这一转变提升了业界对可扩展的动态创意优化（DCO）系统的需求，此类系统需在严格的延迟和成本约束下将创意匹配给最相关的受众。我们提出了PinDCO，一个部署于Pinterest（一个十亿级规模的视觉发现平台）的生产级DCO系统，用于广告创意的检索与选择。PinDCO围绕创意组件融合网络（CCFN）构建，该网络通过为每个创意组件（如图像、标题、布局）设置专用的塔式网络进行建模来执行动态创意评分，并使用组件专属的超参数以应对不同的建模复杂度。各组件的表示经过融合后，在广告级预测的条件下预测创意级评分，我们还通过一种探索机制来提升训练数据质量……（摘要在此处截断）

    arXiv:2609.11943v1 Announce Type: cross  Abstract: Recent advances in generative AI have substantially accelerated the creation of high-quality ad creatives, dramatically expanding the number of candidate variants per campaign. This shift increases the need for scalable dynamic creative optimization (DCO) systems that can match creatives to the most relevant audiences under stringent latency and cost constraints. We present PinDCO, a production DCO system for ad creative retrieval and selection on Pinterest, a billion-scale visual discovery platform. PinDCO is built around a Creative Component Fusion Network (CCFN) that performs dynamic creative scoring by modeling each creative component (e.g., image, title, layout) with a dedicated tower, using component-specific hyperparameters to account for differing modeling complexity. The component representations are fused to predict a creative-level score conditioned on the ad-level prediction, and we improve training data quality via an expl
    
[^18]: 立场论文：推荐系统应超越以平台为中心的排序，迈向个人代理中介的推荐

    Position: Recommender Systems Should Move Beyond Platform-Centric Ranking toward Personal Agent-Mediated Recommendation

    [https://arxiv.org/abs/2609.11942](https://arxiv.org/abs/2609.11942)

    本文提出推荐系统应从平台中心的物品排序范式转向“个人代理中介推荐”（PAMR），即由用户侧的个人代理代表用户在分布式来源中发现、过滤、聚合和治理推荐证据，将控制重心从平台排序转移到用户侧证据中介。

    

    推荐系统通常被构建为排序系统：平台观察用户、构建候选集，并代表用户选择物品。这种框架掩盖了更深层的控制权分配，即平台还决定候选物品的准入、证据边界、解释方式，以及从用户需求到推荐输出的路径。我们认为，推荐的下一个瓶颈不仅在于偏好建模，更在于对证据获取与披露的控制权。我们提倡“个人代理中介推荐”（PAMR）这一范式：由一个面向用户的个人代理代表用户，在分布式来源中发现、过滤、聚合和治理推荐证据。其核心转变不是从一种排序模型换成另一种排序模型，而是从平台侧的物品排序转向用户侧的证据中介。作为一篇立场论文，我们将PAMR定义为一种新的推荐范式，并确立了其边界标准……

    arXiv:2609.11942v1 Announce Type: new  Abstract: Recommender systems are usually framed as ranking systems: platforms observe users, construct candidate sets, and select items on their behalf. This framing hides a deeper allocation of control, in which platforms also determine candidate access, evidence boundaries, explanations, and the path from user need to recommended output. We argue that the next bottleneck in recommendation is not only preference modeling, but control over evidence acquisition and disclosure. We argue for \textbf{Personal Agent-Mediated Recommendation} (PAMR), a paradigm in which a user-facing personal agent represents the user in discovering, filtering, aggregating, and governing recommendation evidence across distributed sources. The central shift is not simply from one ranking model to another, but from platform-side item ranking to user-side evidence mediation. As a position paper, we define PAMR as a new recommendation paradigm, establish its boundary criter
    
[^19]: 在投机式检索增强生成中保证忠实的证据抽取

    Guaranteeing Faithful Evidence Extraction in Speculative Retrieval-Augmented Generation

    [https://arxiv.org/abs/2609.10046](https://arxiv.org/abs/2609.10046)

    该论文提出约束混合解码（CHyD），将原本用于加速推理的投机解码架构重新用于强制执行硬解码约束，从而在投机式检索增强生成中保证抽取的证据逐忠实于检索到的上下文，满足安全关键领域对答案精确匹配的要求。

    

    大型语言模型越来越多地被用作信息检索的接口，但它们仍然容易产生幻觉和忠实性错误，即生成的答案与检索到的证据相偏离。虽然检索增强生成（RAG）以及近期的混合或半抽取方法可以缓解这一问题，但它们并不能保证所引用或抽取的文本片段完全逐字来自检索到的上下文。这一局限性在安全关键领域可能造成严重后果，因为在这些领域中，答案必须与经过认证的文档完全一致。我们提出了约束混合解码，这是一种面向投机式RAG的新型“忠实性优先”范式。虽然传统的投机解码是为推理速度而优化的，但CHyD重新利用了这一架构，以便在抽取模式被正确触发时确保忠实的逐字证据抽取。我们的方法强制执行硬解码约束，将生成内容限制在……（摘要在此处截断）

    arXiv:2609.10046v1 Announce Type: new  Abstract: Large Language Models (LLMs) are increasingly used as interfaces for information retrieval, but they remain prone to hallucinations and faithfulness errors, in which the generated answers diverge from the retrieved evidence. While Retrieval-Augmented Generation (RAG) and recent hybrid or semi-extractive approaches mitigate this issue, they do not guarantee that quoted or extracted spans are verbatim from the retrieved context. This limitation can have severe consequences in safety-critical domains, where answers must exactly match certified documentation.   We introduce Constrained Hybrid Decoding (CHyD), a novel faithfulness-first paradigm for speculative RAG. While traditional speculative decoding is optimized for inference speed, CHyD repurposes this architecture to ensure faithful verbatim evidence extraction when the extraction mode is correctly triggered. Our approach enforces hard decoding constraints that restrict generation to c
    
[^20]: 正确的技能族，错误的技能项：智能体技能检索中的风险暴露基准测试

    Right Family, Wrong Skill: Benchmarking Risk Exposure in Agent Skill Retrieval

    [https://arxiv.org/abs/2606.10388](https://arxiv.org/abs/2606.10388)

    本文提出了SameCapRisk-Bench基准，专门研究智能体技能检索中“找到正确技能族但暴露错误同族技能”的风险暴露失败，并提供了1,190个技能-风险单元和1,686个评估查询案例。

    

    arXiv:2606.10388v2 公告类型：替换-交叉 摘要：智能体技能库正成为可路由的软件资产：检索到的技能可以为智能体提供指令、脚本、资源绑定和执行假设。这使得检索失败比广泛的不相关性更具特异性。系统可能找到正确的技能族，却暴露了错误的同族代表性技能。我们将这种失败研究为同族风险暴露检索。每个基准单元将一个有用的技能与一个查询特定的风险孪生技能配对，该孪生技能共享技能族但在执行控制契约（如所需资源、前置条件、过程或工件）上有所不同。我们引入了SameCapRisk-Bench，一个可审计的基准测试，包含1,190个技能-风险单元和1,686个评估查询案例：694个在公共库压力下的标记孪生单元，以及496个硬角色翻转单元，其中相同的两个技能在配对查询中交换有用/风险角色。发布记录包括资格标准、排除规则、查询生成协议和验证程序。

    arXiv:2606.10388v2 Announce Type: replace-cross  Abstract: Agent skill libraries are becoming routable software assets: a retrieved skill can contribute instructions, scripts, resource bindings, and execution assumptions to an agent. This makes retrieval failures more specific than broad irrelevance. A system can find the right capability family yet expose the wrong same-capability representative. We study this failure as same-capability risk-exposure retrieval. Each benchmark unit pairs a helpful skill with a query-specific risky sibling that shares the capability family but differs on an execution-controlling contract, such as the required resource, precondition, procedure, or artifact. We introduce SameCapRisk-Bench, an auditable benchmark with 1,190 skill-risk units and 1,686 evaluation query cases: 694 marked-sibling units under public library pressure and 496 hard role-flip units where the same two skills swap helpful/risky roles across paired queries. The release records admissi
    
[^21]: CoHyDE：用于工具检索的LLM重写器与稠密编码器的迭代协同训练

    CoHyDE: Iterative Co-Training of LLM Rewriter & Dense Encoder for Tool Retrieval

    [https://arxiv.org/abs/2605.29271](https://arxiv.org/abs/2605.29271)

    提出CoHyDE方法，通过迭代协同训练将LLM查询重写器与稠密编码器作为一个共同演化的整体系统，融合编码器微调与HyDE查询扩展二者的优势，显著提升LLM智能体在大型API目录上的工具检索性能。

    

    针对大型API目录的工具检索是LLM智能体的核心瓶颈：用户查询以口语化且往往不完整的语言到达，而目录使用技术性的API词汇，任何固定的编码器都无法独自弥合这一鸿沟。两种主流训练方法——对比式编码器微调和基于冻结LLM的HyDE式查询扩展——从相反的两端解决这一问题，并以互补的方式失效：微调后的编码器在查询的表面形式已与目录匹配时表现出色，但在不匹配时性能崩溃；而零样本HyDE对不完整查询更为鲁棒，却会生成不感知目录内容的假设性描述，当查询格式良好时反而会降低检索效果。我们提出CoHyDE，一种将稠密编码器和LLM重写器作为单一协同演化系统进行训练的迭代方法：编码器使用InfoNCE在生成的目录风格假设性描述上重新训练……

    arXiv:2605.29271v2 Announce Type: replace-cross  Abstract: Tool retrieval over large API catalogs is a core bottleneck for LLM agents: user queries arrive in colloquial, often underspecified language, while the catalog uses technical API vocabulary that no fixed encoder can bridge on its own. The two dominant training approaches, contrastive encoder fine-tuning and HyDE-style query expansion with a frozen LLM, address this problem from opposite ends and fail in complementary directions: the fine-tuned encoder excels when the query's surface form already matches the catalog but collapses when it does not, while zero-shot HyDE is more robust to underspecified queries yet generates catalog-unaware hypothetical descriptions that degrade retrieval when queries are well-formed. We introduce CoHyDE, an iterative procedure that trains the dense encoder and the LLM rewriter as a single co-evolving system: the encoder is retrained with InfoNCE on catalog-style hypothetical descriptions produced 
    
[^22]: Total Recall QA：一个用于深度研究智能体的可验证评估套件

    Total Recall QA: A Verifiable Evaluation Suite for Deep Research Agents

    [https://arxiv.org/abs/2603.18516](https://arxiv.org/abs/2603.18516)

    该论文提出了Total Recall QA评估套件，通过基于结构化知识库构建具有唯一答案且可精确验证的查询，首次为深度研究智能体提供了满足全部评估要求的可验证评估框架。

    

    深度研究智能体是基于大语言模型（LLM）的系统，旨在对大型开放域数据源执行多步骤的信息检索与推理，通过综合多个信息来源的内容来回答复杂问题。鉴于该任务本身的复杂性，尽管近期已有诸多相关努力，深度研究智能体的评估仍然存在根本性的挑战。本文列出了评估深度研究智能体所需满足的一系列必要要求和可选属性，并观察到现有基准测试并不能满足所有这些要求。受此前关于TREC Total Recall Tracks研究的启发，我们提出了完全召回问答（Total Recall Question Answering）任务，并开发了一个满足上述标准的深度研究智能体评估框架。我们的框架构建了具有唯一答案的完全召回式查询，其精确的评估结果和相关性判断源自与结构化知识库相配对的数据。

    arXiv:2603.18516v2 Announce Type: replace  Abstract: Deep research agents have emerged as LLM-based systems designed to perform multi-step information seeking and reasoning over large, open-domain sources to answer complex questions by synthesizing information from multiple information sources. Given the complexity of the task and despite various recent efforts, evaluation of deep research agents remains fundamentally challenging. This paper identifies a list of requirements and optional properties for evaluating deep research agents. We observe that existing benchmarks do not satisfy all identified requirements. Inspired by prior research on TREC Total Recall Tracks, we introduce the task of Total Recall Question Answering and develop a framework for deep research agents evaluation that satisfies the identified criteria. Our framework constructs single-answer, total recall queries with precise evaluation and relevance judgments derived from a structured knowledge base paired with a te
    
[^23]: 推荐系统上的成员推断攻击：综述

    Membership Inference Attacks on Recommender System: A Survey

    [https://arxiv.org/abs/2509.11080](https://arxiv.org/abs/2509.11080)

    本文是一篇关于推荐系统成员推断攻击的综述，指出传统成员推断攻击因后验概率不可见而不适用于推荐系统，并系统分析了此类攻击对用户隐私造成的泄露风险。

    

    推荐系统已被广泛应用于各种应用场景，包括电子商务、金融、医疗保健和社交媒体，其在塑造用户行为和决策方面的影响力日益增强，凸显了其在各个领域中不断增长的重要性。然而，最近的研究表明，推荐系统容易受到成员推断攻击的威胁，这类攻击旨在推断用户的交互记录是否被用于训练目标模型。针对推荐系统模型的成员推断攻击可以直接导致隐私泄露。例如，通过识别某条购买记录已被用于训练与特定用户相关联的推荐系统，攻击者可以推断出该用户的特殊癖好。近年来，成员推断攻击已被证明在其他机器学习任务上是有效的，例如分类模型和自然语言处理。然而，由于后验概率不可见，传统的成员推断攻击并不适合推荐系统。尽管……

    arXiv:2509.11080v4 Announce Type: replace  Abstract: Recommender systems (RecSys) have been widely applied to various applications, including E-commerce, finance, healthcare, social media and have become increasingly influential in shaping user behavior and decision-making, highlighting their growing impact in various domains. However, recent studies have shown that RecSys are vulnerable to membership inference attacks (MIAs), which aim to infer whether user interaction record was used to train a target model or not. MIAs on RecSys models can directly lead to a privacy breach. For example, via identifying the fact that a purchase record that has been used to train a RecSys associated with a specific user, an attacker can infer that user's special quirks. In recent years, MIAs have been shown to be effective on other ML tasks, e.g., classification models and natural language processing. However, traditional MIAs are ill-suited for RecSys due to the unseen posterior probability. Although
    

