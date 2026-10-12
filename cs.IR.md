# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Can Language Models Play the Wikipedia Game?](https://arxiv.org/abs/2610.13948) | 该论文提出以维基百科游戏作为信息检索基准，通过OpenAlex构建大规模本地维基百科页面图与论文引用图，在多语言和多学术数据集上系统比较句子嵌入模型与大语言模型的导航检索能力，发现没有任何模型能显著超越人类表现。 |
| [^2] | [RATED: Reliability-Aware AI Agent Tool Retrieval via Execution Dynamics](https://arxiv.org/abs/2610.13889) | RATED通过冻结双编码器进行语义匹配，并用轻量级上下文老虎机模块根据实时执行反馈重排序候选工具，无需模型重训即可快速适应运行时工具故障，同时提出含5,172个真实工具的EcoAgent基准，实现52.2%的相对MRR提升。 |
| [^3] | [Tracing and Taming Hallucinated Terms via Subsequence-Association Filtering and Breadth-Aware Selection for LLM Query Expansion](https://arxiv.org/abs/2610.13810) | 提出无需训练、与检索器无关的查询扩展方法SAFE-QE，基于幻觉的子序列关联解释，通过语料库条件下的跨上下文可复现性评分追踪并过滤幻觉词项，并以广度感知的原则性标准决定保留词项的数量。 |
| [^4] | [SPARQL-LLM in the TEXT2SPARQL 2026 Challenge](https://arxiv.org/abs/2610.13648) | SPARQL-LLM是一个基于大语言模型的开源知识图谱问答系统，在TEXT2SPARQL 2026挑战赛中八种方法里排名第一，准确率比第二名高出59%，并支持英、德、西三语问答。 |
| [^5] | [Beyond Temporal Proximity: Explanatory World Models for Credit Assignment in Reinforcement Learning](https://arxiv.org/abs/2610.13504) | 该论文提出解释性世界模型（EWM）框架，用语言模型生成的因果解释取代传统的时间邻近性来进行强化学习中的信用分配，通过保守分层算子递归重分配奖励并保持回合总回报不变，从而提升生成式推荐系统中长时程决策的样本效率。 |
| [^6] | [Web$hell: A Unified Interface for Agentic Web Access](https://arxiv.org/abs/2610.13416) | 提出Web$hell——一个统一搜索增强、浏览器与文件系统等多种访问模式的智能体网页访问接口，使智能体能够在单条任务轨迹内同时完成信息收集、浏览器操作和证据回溯。 |
| [^7] | [NEAR: Neighborhood-Aware Residuals for Memory Retrieval](https://arxiv.org/abs/2610.13314) | NEAR提出了一种免训练的记忆检索框架，通过利用邻居记录间的共享信息构建投影残差来补充原始嵌入，有效缓解了因共享内容导致无关记录排名高于正确证据的问题，从而提升查询-记忆匹配的准确性。 |
| [^8] | [PRIME: Pre-Retrieval Index Modeling Enhancement for Structured Document Retrieval-Real](https://arxiv.org/abs/2610.13296) | PRIME提出了一个索引侧优化框架，通过LLM驱动的语义重构与确定性的元数据合成及分层检索两条互补轨道，提升结构化文档检索质量并降低嵌入计算成本。 |
| [^9] | [Beyond the Final Embedding: Retrieving Memories from Token-Level Evidence](https://arxiv.org/abs/2610.13277) | 该论文提出一种无需额外训练的稀疏词元匹配方法，利用嵌入模型已有的词元级上下文表示捕捉局部证据，并结合整条记忆相似度对记忆排序，在八个基准测试和多个模型家族上显著提升检索效果，还可通过可选的低秩查询适配器进一步增强。 |
| [^10] | [Locate, Judge, Read: Role-Specialized Tool Agents over a Knowledge-Point Index for Open-Domain Multimodal Document QA](https://arxiv.org/abs/2610.13128) | 提出按职能分离的KnowForge流水线，通过知识点索引、仅能搜索的定位器、强制第二跳检索的桥接判断器和仅能阅读的深度阅读器等角色专门化工具代理，在开放域多模态文档问答上显著超越了现有检索增强方法。 |
| [^11] | [GeopoliticalQA: Benchmarking Retrieval and Generation over Intelligence Reports](https://arxiv.org/abs/2610.12617) | 该论文构建了基于415份开源地缘政治与战略情报报告、包含5,525个问答对的问答基准GeopoliticalQA，并通过评估15个检索系统和六个开放权重LLM发现，模型在文档级检索上表现相近，而更具挑战性的块级检索质量并不取决于嵌入模型的规模。 |
| [^12] | [Claim-Gated Source-Risk Auditing for Generative Search](https://arxiv.org/abs/2609.29145) | 该论文提出针对生成式搜索的声明门控来源风险审计契约：只有当关系证据、答案采纳、重要性和披露全部被观测到时才判定遗漏已解决，否则保持未解决状态，并通过可执行的参考检查器在穷举合成测试中验证了该契约的完备性。 |
| [^13] | [Trust, but Validate the Instrument: Auditing AI-Generated RTL Verification Plans on Authored Security-Regression Proxies](https://arxiv.org/abs/2609.19844) | 论文提出可审计框架SecTB-RTL，通过自建硬件安全回归测试发现AI生成的RTL验证计划虽被提供商接受但几乎全部无法通过生产语义验证，证明提供商模式的接受并不能等同于执行的有效性。 |
| [^14] | [Imagine Before Retrieval: Prospective Skill Retrieval for LLM Agents](https://arxiv.org/abs/2609.01642) | 提出受人类前瞻性认知启发的SkillDreamer框架，通过“先想象后检索”策略解决LLM智能体技能检索中任务查询与技能之间的失配（QSM）问题。 |
| [^15] | [UxSID: Semantic-Aware User Interests Modeling for Ultra-Long Sequence](https://arxiv.org/abs/2605.09040) | UxSID 通过语义 ID 与双层注意力机制构建语义组共享兴趣记忆，为超长用户行为序列建模提供兼顾效率与效果的第三条路径，实现 SOTA 性能并在大规模广告 A/B 测试中带来 0.337% 的收入提升。 |
| [^16] | [Retrieval-Augmented Generation for Predicting Cellular Responses to Gene Perturbation](https://arxiv.org/abs/2603.07233) | 提出了PT-RAG，一个即插即用的两阶段检索增强生成模块，通过GenePT语义检索与可微分Gumbel-Softmax选择器动态获取相关的扰动上下文，从而改进单细胞基因扰动响应的预测。 |
| [^17] | [Access Paths for Efficient Ordering with Large Language Models](https://arxiv.org/abs/2509.00303) | 提出了LLM ORDER BY语义算子及预算感知优化器，通过动态选择近优访问路径，在所有基准测试中实现了与最佳静态方法相当或更优的排序准确率。 |
| [^18] | [CRM: Retrieval Model with Controllable Condition](https://arxiv.org/abs/2412.13844) | 提出可控检索模型（CRM），将观看时长作为用户塔的条件输入，使模型在推理阶段无需重新训练即可通过指定期望观看时长来引导检索，同时保持物品向量可预先计算以支持高效的 ANN 检索。 |

# 详细

[^1]: 语言模型能玩维基百科游戏吗？

    Can Language Models Play the Wikipedia Game?

    [https://arxiv.org/abs/2610.13948](https://arxiv.org/abs/2610.13948)

    该论文提出以维基百科游戏作为信息检索基准，通过OpenAlex构建大规模本地维基百科页面图与论文引用图，在多语言和多学术数据集上系统比较句子嵌入模型与大语言模型的导航检索能力，发现没有任何模型能显著超越人类表现。

    

    语言模型日益用于支持或自动化从过于庞大而无法建立索引的文档集合中进行信息检索的任务。当该任务需要专业的语言学或科学知识时，难度会显著增加。我们使用维基百科游戏来测试语言模型在类似信息检索任务上的表现。在该游戏中，玩家必须仅通过文章之间的链接到达目标文章。我们在五种维基百科语言和五个学术文献数据集上比较了句子嵌入模型与大型语言模型（LLM）。我们利用OpenAlex构建了规模非常大的本地维基百科页面图和论文引用图，并提供了构建和导航这些图的代码。这使得我们无需依赖第三方API即可研究语言模型的导航能力。我们将这些模型与人类表现和TF-IDF基准进行了比较。维基百科游戏的形式能够在不同技术、语言和领域之间实现稳健的比较。没有模型显著超越人类，尽管表现最好的模……

    arXiv:2610.13948v1 Announce Type: new  Abstract: Language models increasingly support or automate information retrieval from document collections too large to index. This task is difficult when it requires specialist linguistic or scientific knowledge. We used the Wikipedia Game to test language models on an analogous information retrieval task. In the game, players must reach a target article using only article links. We compared sentence embedding models and LLMs across five Wikipedia languages and five academic-literature datasets. We construct very large local graphs of Wikipedia pages and of paper citations using OpenAlex and provide code to build and navigate them. This allows the study of language models' navigation capabilities without relying on third-party APIs. We compared the models with human and TF-IDF benchmarks. The Wikipedia Game format enables robust comparisons across techniques, languages, and domains. No model significantly outperformed humans, although the best mo
    
[^2]: RATED：基于执行动态的可靠性感知AI智能体工具检索

    RATED: Reliability-Aware AI Agent Tool Retrieval via Execution Dynamics

    [https://arxiv.org/abs/2610.13889](https://arxiv.org/abs/2610.13889)

    RATED通过冻结双编码器进行语义匹配，并用轻量级上下文老虎机模块根据实时执行反馈重排序候选工具，无需模型重训即可快速适应运行时工具故障，同时提出含5,172个真实工具的EcoAgent基准，实现52.2%的相对MRR提升。

    

    随着AI智能体日益依赖外部工具，准确的工具检索对于实现可靠行为至关重要。现有方法仅优化语义相关性，却在很大程度上忽略了执行期间工具的可用性。为填补这一空白，我们提出了RATED（基于执行动态的可靠性感知工具检索），这是一个将鲁棒语义匹配与动态可靠性自适应相结合的检索框架。RATED采用经对比微调的冻结双编码器进行语义匹配，并利用一个轻量级的上下文多臂老虎机模块，基于实时执行反馈对候选工具进行重排序。这一设计使系统能够快速适应运行时工具故障，而无需昂贵的模型重新训练。我们还引入了EcoAgent基准，包含5,172个真实世界工具和超过21万个查询-工具对，用于离线检索与动态可靠性评估。实验结果表明，RATED相比ToolLLM API实现了52.2%的相对MRR提升。

    arXiv:2610.13889v1 Announce Type: new  Abstract: As AI agents increasingly rely on external tools, accurate retrieval is essential for reliable behavior. Existing methods optimize semantic relevance but largely ignore tool availability during execution. To address this gap, we propose RATED (Reliability-Aware Tool Retrieval via Execution Dynamics), a retrieval framework that combines robust semantic matching with dynamic reliability adaptation. RATED employs a frozen dual-encoder with contrastive fine-tuning for semantic matching, and a lightweight contextual bandit module that re-ranks candidates based on real-time execution feedback. This design enables rapid adaptation to runtime tool failures without expensive model retraining. We introduce EcoAgent, a benchmark of 5,172 real-world tools and over 210,000 query-tool pairs for offline retrieval and dynamic reliability evaluation. Experimental results demonstrate that RATED achieves a 52.2% relative MRR improvement over the ToolLLM AP
    
[^3]: 通过子序列关联过滤与广度感知选择来追踪和驯服大语言模型查询扩展中的幻觉词项

    Tracing and Taming Hallucinated Terms via Subsequence-Association Filtering and Breadth-Aware Selection for LLM Query Expansion

    [https://arxiv.org/abs/2610.13810](https://arxiv.org/abs/2610.13810)

    提出无需训练、与检索器无关的查询扩展方法SAFE-QE，基于幻觉的子序列关联解释，通过语料库条件下的跨上下文可复现性评分追踪并过滤幻觉词项，并以广度感知的原则性标准决定保留词项的数量。

    

    大语言模型（LLM）目前已成为查询扩展词项的标准来源，但它们会产生幻觉。它们会注入不存在的实体和偏离主题的词项，这些词项增加了排序噪声，甚至可能损害强大的检索器。现有的补救措施，如语料库接地、相互验证和偏好对齐，将幻觉词项视为不加区分的干扰因素，仅通过一致性启发式规则进行过滤，既没有对某个词项为何会出现的机制性解释，也没有关于应保留多少词项的原则性标准。我们提出了子序列关联过滤、广度感知的查询扩展方法（SAFE-QE），这是一种无需训练、与检索器无关的方法，其建立在对幻觉的子序列关联解释之上。我们根据每个候选扩展词项在语料库条件下的跨上下文可复现性对其进行评分——那些在随机化查询上下文中反复出现、却在伪相关文档集中缺乏支持的词项，会被标记为主导性幻觉的产物（摘要在此处截断）。

    arXiv:2610.13810v1 Announce Type: cross  Abstract: Large language models (LLMs) are now a standard source of query expansion terms but they hallucinate. They inject non existent entities and off-topic terms that add ranking noise and can hurt strong retrievers. Existing remedies such as corpus grounding, mutual verification and preference alignment treat hallucinated terms as an undifferentiated nuisance to be filtered by agreement heuristics, without a mechanistic account of why a given term appears or a principled criterion for how many terms to keep. We propose Subsequence-Association Filtered, breadth-aware query expansion (SAFE-QE), a training-free, retriever agnostic method that draws on the subsequence-association account of hallucination. We score each candidate expansion term by its corpus conditioned cross-context reproducibility, terms that reappear under randomised query contexts yet lack support in the pseudo-relevant set are flagged as products of dominant hallucinatory a
    
[^4]: SPARQL-LLM 在 TEXT2SPARQL 2026 挑战赛中的表现

    SPARQL-LLM in the TEXT2SPARQL 2026 Challenge

    [https://arxiv.org/abs/2610.13648](https://arxiv.org/abs/2610.13648)

    SPARQL-LLM是一个基于大语言模型的开源知识图谱问答系统，在TEXT2SPARQL 2026挑战赛中八种方法里排名第一，准确率比第二名高出59%，并支持英、德、西三语问答。

    

    通过SPARQL查询以知识图谱（KG）形式表示的分布式数据仍然是一项困难的任务，需要熟练掌握查询语法、端点规范和模式细节。知识图谱问答（KGQA）系统通过自动将自然语言翻译成SPARQL来解决这个问题，而大语言模型（LLM）的最新进展使这种翻译越来越可行。在本文中，我们在TEXT2SPARQL 2026挑战赛的背景下介绍了我们的开源KGQA系统SPARQL-LLM。在该挑战赛中，SPARQL-LLM在所有数据集上八种竞争方法中排名第一，比第二名方法的准确率高出59%，并支持英语、德语和西班牙语的多语言问答。SPARQL-LLM已在 https://github.com/sib-swiss/sparql-llm 公开发布，并已在 https://www.expasy.org/chat 上部署于去中心化生物信息学知识图谱的生产环境中。

    arXiv:2610.13648v1 Announce Type: new  Abstract: Querying distributed data represented as Knowledge Graphs (KGs) via SPARQL remains a difficult task, requiring fluency in query syntax, endpoint specifications, and schema details. Knowledge Graph Question Answering (KGQA) systems address this by automatically translating natural language into SPARQL, and recent advances in Large Language Models (LLMs) have made this translation increasingly feasible. In this paper, we describe SPARQL-LLM, our open-source KGQA system, in the context of the TEXT2SPARQL 2026 challenge. In this challenge, SPARQL-LLM ranks first overall among eight competing approaches across all datasets, achieving up to 59% higher accuracy than the second-best approach and supporting multilingual question answering in English, German, and Spanish. SPARQL-LLM is publicly released at https://github.com/sib-swiss/sparql-llm and deployed in production over decentralized bioinformatics KGs at https://www.expasy.org/chat.
    
[^5]: 超越时间邻近性：面向强化学习中信用分配的解释性世界模型

    Beyond Temporal Proximity: Explanatory World Models for Credit Assignment in Reinforcement Learning

    [https://arxiv.org/abs/2610.13504](https://arxiv.org/abs/2610.13504)

    该论文提出解释性世界模型（EWM）框架，用语言模型生成的因果解释取代传统的时间邻近性来进行强化学习中的信用分配，通过保守分层算子递归重分配奖励并保持回合总回报不变，从而提升生成式推荐系统中长时程决策的样本效率。

    

    arXiv:2610.13504v1 公告类型：交叉 摘要：生成式推荐系统越来越多地采用强化学习（RL）来优化那些可能在决策发生数百甚至数千次交互之后才显现的结果。标准的强化学习依据时间邻近性而非显式相关性来传播奖励信号，这削弱了分配给那些距离遥远但至关重要的动作的信号，并成为样本效率低下的一个根源。我们提出了解释性世界模型，这是一个归因框架，利用解释来识别可能对最终已实现结果有贡献的早期事件。我们用一个前沿语言模型来实例化EWM，并通过一个保守的分层算子将其生成的解释用作学习信号，该算子在保持未折扣回合回报的前提下递归地重新分配奖励。为了探究什么样的解释才是有用的，我们受Halpern的实际因果理论、Lombrozo的解释简洁性理论以及De……（原文摘要在此处截断）

    arXiv:2610.13504v1 Announce Type: cross  Abstract: Generative Recommenders increasingly use Reinforcement Learning (RL) to optimize outcomes that may appear hundreds or thousands of interactions after a decision. Standard RL propagates reward according to temporal proximity rather than explicit relevance, attenuating the signal assigned to distant but important actions and creating a source of sample inefficiency. We introduce Explanatory World Models (EWM), an attribution framework that uses explanations to identify the earlier events that plausibly contributed to a realized outcome. We instantiate EWM with a frontier language model and use its explanations as learning signals using a conservative hierarchical operator that recursively redistributes reward while preserving the undiscounted episode return. To investigate what constitutes a useful explanation, we operationalize three complementary variants inspired by Halpern's actual causation, Lombrozo's explanatory simplicity, and De
    
[^6]: Web$hell：智能体网页访问的统一接口

    Web$hell: A Unified Interface for Agentic Web Access

    [https://arxiv.org/abs/2610.13416](https://arxiv.org/abs/2610.13416)

    提出Web$hell——一个统一搜索增强、浏览器与文件系统等多种访问模式的智能体网页访问接口，使智能体能够在单条任务轨迹内同时完成信息收集、浏览器操作和证据回溯。

    

    大型语言模型（LLM）智能体正在成为开放网页的直接用户。随着它们承担越来越复杂、信息密集的任务，有效的网页访问正成为智能体问题求解的核心能力。现有方法通过不同的访问模式推进了智能体网页访问，包括搜索增强型智能体、基于浏览器的智能体，以及近期的基于文件系统的智能体。然而，随着用户期望智能体完成长时程、端到端的任务，它们越来越需要在同一条任务轨迹内进行广泛的信息收集、执行浏览器操作，并重新审视已获取的证据。我们的初步分析表明，现有访问模式难以满足这些需求：它们主要只支持信息收集或浏览器交互二者之一，而将两者能力结合起来往往伴随巨大的成本或性能折衷。为此，我们提出了Web$hell，一个面向智能体网页访问的统一接口。

    arXiv:2610.13416v1 Announce Type: new  Abstract: Large language model (LLM) agents are becoming direct users of the open web. As they take on increasingly complex and information-intensive tasks, effective web access is becoming a core capability for agentic problem solving. Existing approaches have advanced agentic web access through different access models, including search-augmented agents, browser-based agents, and, more recently, filesystem-based agents. Nevertheless, as users expect agents to complete long-horizon, end-to-end tasks, they increasingly need to conduct broad information gathering, perform browser actions, and revisit acquired evidence within a single trajectory. Yet our preliminary analyses reveal that existing access models struggle to meet these demands. They primarily support either information gathering or browser interaction, while combining their capabilities can involve substantial costs or performance trade-offs. To this end, we present Web\$hell, an interfa
    
[^7]: NEAR：面向记忆检索的邻域感知残差方法

    NEAR: Neighborhood-Aware Residuals for Memory Retrieval

    [https://arxiv.org/abs/2610.13314](https://arxiv.org/abs/2610.13314)

    NEAR提出了一种免训练的记忆检索框架，通过利用邻居记录间的共享信息构建投影残差来补充原始嵌入，有效缓解了因共享内容导致无关记录排名高于正确证据的问题，从而提升查询-记忆匹配的准确性。

    

    记忆检索旨在从存储的记录中识别证据以回答查询。具有不同事实细节的存储记录可能共享主题、实体或事件。这些共享内容会使得不包含所需证据的记录更有可能获得较高的相似度分数，从而排在正确证据之前。在本文中，我们利用这些记录之间共享的信息来构建用于查询-记忆匹配的局部参考。我们提出了邻域感知残差框架，这是一个免训练的框架，利用这些参考来构建残差表示，以补充原始嵌入。具体而言，每个记忆从其邻居记录中导出一个参考方向，每个查询从预计算的表中获取其参考。利用这些参考，NEAR为查询和记忆构建投影残差。这些残差与原始嵌入相结合，用于单次近似最近邻搜索。

    arXiv:2610.13314v1 Announce Type: new  Abstract: Memory retrieval aims to identify evidence from stored records to answer a query. Stored records with different factual details may share topics, entities, or events. Such shared contents would make records without the required evidence more likely to receive high similarity scores and outrank the correct evidence. In this paper, we use the information shared by these records to construct local references for query-memory matching. We propose NEighborhood-Aware Residuals (NEAR), a training-free framework that uses these references to construct residual representations that complement the original embeddings. Specifically, each memory derives a reference direction from its neighbors, and each query obtains its reference from a precomputed table. Using these references, NEAR constructs projection residuals for both queries and memories. These residuals are combined with the original embeddings for a single approximate nearest-neighbor sear
    
[^8]: PRIME：面向结构化文档检索的检索前索引建模增强

    PRIME: Pre-Retrieval Index Modeling Enhancement for Structured Document Retrieval-Real

    [https://arxiv.org/abs/2610.13296](https://arxiv.org/abs/2610.13296)

    PRIME提出了一个索引侧优化框架，通过LLM驱动的语义重构与确定性的元数据合成及分层检索两条互补轨道，提升结构化文档检索质量并降低嵌入计算成本。

    

    对外部知识库的高精度检索对于LLM智能体的高效运行至关重要。然而，现有的信息检索（IR）研究主要优化查询侧的检索器（如ColBERT、BGE）和生成器（如Self-RAG、GraphRAG），而将语料库侧的表示简化为分块处理，这使得索引侧质量成为一个未被充分探索的优化杠杆。我们提出了检索前索引建模与增强（PRIME），这是一个索引侧优化框架，建立在两条互补的技术轨道上：一条是由LLM驱动的语义重构（SR）轨道，将长文档转换为结构化、信息密集的章节；另一条是确定性的元数据合成（MS）与分层检索（HR）轨道，提取语义锚点和标题结构，实现由粗到细的章节检索。我们的评估表明，确定性元数据在检索质量上可与结构感知分块相媲美，同时削减了嵌入计算成本。

    arXiv:2610.13296v1 Announce Type: cross  Abstract: High-precision retrieval over external knowledge bases is essential for LLM agents to operate effectively. However, existing Information Retrieval (IR) research primarily optimizes query-side retrievers (e.g., ColBERT, BGE) and generators (e.g., Self-RAG, GraphRAG), while reducing corpus-side representation to chunking. This leaves index-side quality an underexplored lever. We propose Pre-Retrieval Index Modeling & Enhancement (PRIME), an index-side optimization framework built on two complementary tracks: an LLM-driven Semantic Restructuring (SR) track that transforms long documents into structured, information-dense sections, and a deterministic Metadata Synthesis (MS) and Hierarchical Retrieval (HR) track that extracts semantic anchors and heading structure for coarse-to-fine section retrieval. Our evaluations show that deterministic metadata matches structure-aware chunking's retrieval quality while cutting embedding computation by
    
[^9]: 超越最终嵌入：从词元级证据中检索记忆

    Beyond the Final Embedding: Retrieving Memories from Token-Level Evidence

    [https://arxiv.org/abs/2610.13277](https://arxiv.org/abs/2610.13277)

    该论文提出一种无需额外训练的稀疏词元匹配方法，利用嵌入模型已有的词元级上下文表示捕捉局部证据，并结合整条记忆相似度对记忆排序，在八个基准测试和多个模型家族上显著提升检索效果，还可通过可选的低秩查询适配器进一步增强。

    

    记忆增强型助手从存储的对话和文档中检索证据以回答查询。大多数基于嵌入的检索器比较查询和记忆的单向量表示，即使相关性往往只取决于内容的一小部分。这种局部证据可以利用同一嵌入模型已经计算出的上下文词元表示与查询进行匹配。我们的方法使用稀疏词元匹配来改进检索，且无需额外训练。每个查询词元在候选记忆中检索一小撮匹配的词元，匹配得分针对每条记忆进行聚合，并与整条记忆的相似度相结合来对记忆进行排序。可选的低秩查询适配器可以在监督下进行训练，同时保持编码器和已存储的记忆表示固定。我们的方法无需额外训练即可在八个基准测试和多个模型家族上提升检索性能。查询适配器可进一步……（原文在此处截断）

    arXiv:2610.13277v1 Announce Type: cross  Abstract: Memory-augmented assistants retrieve evidence from stored conversations and documents to answer queries. Most embedding-based retrievers compare single-vector representations of queries and memories, even when relevance depends on a small part of the content. Such local evidence can be matched to queries using the contextual token representations already computed by the same embedding models. Our method uses sparse token matching to improve retrieval without additional training. Each query token retrieves a small set of matching tokens across candidate memories. The matching scores are aggregated for each memory and combined with whole-memory similarity to rank memories. An optional low-rank query adapter can be trained with supervision while keeping the encoder and stored memory representations fixed. Our method improves retrieval across eight benchmarks and multiple model families without additional training. Query adaptation further
    
[^10]: 定位、判断、阅读：基于知识点索引的角色专门化工具代理用于开放域多模态文档问答

    Locate, Judge, Read: Role-Specialized Tool Agents over a Knowledge-Point Index for Open-Domain Multimodal Document QA

    [https://arxiv.org/abs/2610.13128](https://arxiv.org/abs/2610.13128)

    提出按职能分离的KnowForge流水线，通过知识点索引、仅能搜索的定位器、强制第二跳检索的桥接判断器和仅能阅读的深度阅读器等角色专门化工具代理，在开放域多模态文档问答上显著超越了现有检索增强方法。

    

    对大型异构文档集合进行问答需要先找到正确的段落再进行阅读，然而当前的检索增强代理将两类工具都交给同一个模型并让其自行调度。我们提出了KnowForge，一个按职能分离的流水线：一个混合检索的知识点索引；一个只能执行搜索的定位器代理；一个由代码调度的桥接判断器，强制执行缺失的第二跳检索；一个类型化、大小受限的证据池，用于合并每次搜索的排序结果；一个只能阅读文本、图像和表格的深度阅读器代理；以及一个带有格式检查器的综合器。通过自托管的Qwen3.6-35B-A3B（激活参数3B），KnowForge在完整的开放域M3DocVQA（2,441个问题、3,368个PDF）上达到F1 47.5，比已发布的M3DocRAG系统高出11.0，比在同一索引和模型上进行一次性检索回答高出7.1；使用DeepSeek-V4-Flash时达到53.5。在MultiHop-RAG上达到73.5的准确率，高于最强（摘要在此处截断）

    arXiv:2610.13128v1 Announce Type: new  Abstract: Question answering over large heterogeneous document collections requires first finding the right passages and then reading them, yet current retrieval-augmented agents give one model both kinds of tools and let it schedule itself. We present KnowForge, a pipeline separated by job: a hybrid-retrieval knowledge-point index; a Locator agent that can only search; a code-scheduled bridge judge that forces the missing second hop; a typed, size-bounded evidence pool that merges every search's ranking; a Deep Reader agent that can only read text, images and tables; and a synthesizer with a format checker. With a self-hosted Qwen3.6-35B-A3B (3B active), KnowForge reaches F1 47.5 on the full open-domain M3DocVQA (2,441 questions, 3,368 PDFs), +11.0 over the published M3DocRAG system and +7.1 over one-shot retrieve-and-answer on the same index and model, and 53.5 with DeepSeek-V4-Flash. On MultiHop-RAG it reaches 73.5 accuracy, above the strongest
    
[^11]: GeopoliticalQA：面向情报报告的检索与生成基准测试

    GeopoliticalQA: Benchmarking Retrieval and Generation over Intelligence Reports

    [https://arxiv.org/abs/2610.12617](https://arxiv.org/abs/2610.12617)

    该论文构建了基于415份开源地缘政治与战略情报报告、包含5,525个问答对的问答基准GeopoliticalQA，并通过评估15个检索系统和六个开放权重LLM发现，模型在文档级检索上表现相近，而更具挑战性的块级检索质量并不取决于嵌入模型的规模。

    

    地缘政治与战略情报报告是专家知识的丰富来源，在当前政治环境下价值日益凸显，但其异质性和分析复杂性给检索增强问答带来了挑战。我们提出了一个基于415份开源地缘政治与战略情报报告的有据可依问答基准，包含5,525个基于文档的问答对，涵盖事实性、推理性、比较性和风险导向型问题。我们提出了一种文档感知的生成流水线，在生成问题前先对报告进行画像分析，并通过专家标注和LLM-as-judge协议验证了数据集质量。利用该基准，我们评估了15个检索系统和六个开放权重LLM用于RAG。我们发现各模型在文档级检索上表现相近，而在更具挑战性的块级检索上，检索质量并非由嵌入模型规模驱动：bge

    arXiv:2610.12617v1 Announce Type: new  Abstract: Geopolitical and strategic intelligence reports constitute a rich source of expert knowledge and have increasing value in the current political climate, but their heterogeneity and analytical complexity pose challenges for retrieval-augmented question answering. We introduce a benchmark for grounded question answering over 415 open-source geopolitical and strategic intelligence reports, comprising 5,525 document-grounded question-answer pairs spanning factual, inferential, comparative, and risk-oriented questions. We propose a document-aware generation pipeline that profiles reports before generating questions and validate dataset quality through expert annotation and an LLM-as-judge protocol. Using the benchmark, we evaluate 15 retrieval systems and six open-weight LLMs for RAG. We find that models perform similarly on document-level retrieval, while at the harder chunk level retrieval quality is not driven by embedding-model scale: bge
    
[^12]: 面向生成式搜索的声明门控式来源风险审计

    Claim-Gated Source-Risk Auditing for Generative Search

    [https://arxiv.org/abs/2609.29145](https://arxiv.org/abs/2609.29145)

    该论文提出针对生成式搜索的声明门控来源风险审计契约：只有当关系证据、答案采纳、重要性和披露全部被观测到时才判定遗漏已解决，否则保持未解决状态，并通过可执行的参考检查器在穷举合成测试中验证了该契约的完备性。

    

    生成式搜索的答案可能引用了有依据的段落，却遗漏了会改变其解释方式的来源关系。我们规定了一种针对“查询-来源-答案”三元组的声明门控审计方法。只有当关系证据、答案采纳、重要性和披露全部被观测到时，遗漏才被判定为“已解决”；证据不完整时保持“未解决”状态，而不是被默认当作独立性处理。该规范将此判定终点与引用支持和审查优先级区分开来，并将决策绑定到带版本号的证据片段上。一个参考检查器使记录契约变得可执行。在一个穷举式的合成测试套件上，该检查器复现了全部81种三态谓词组合，并拒绝了192条故意构造的畸形记录。公共防护基线和谓词消融实验将终点逻辑与缺失证据的处理区分开来，而受控转换则检验了支持分离与证据移除。这些是有限的契约符合性结果，而非检测器性能……（原文摘要截断）

    arXiv:2609.29145v1 Announce Type: new  Abstract: A generative search answer can cite a supported passage yet omit a source relationship that changes its interpretation. We specify a claim-gated audit of the query-source-answer tuple. An omission is resolved only when relationship evidence, answer adoption, materiality, and disclosure are all observed; incomplete evidence remains unresolved rather than being treated as independence. The specification separates this endpoint from citation support and review priority, and binds decisions to versioned evidence spans. A reference checker makes the record contract executable. On an exhaustive synthetic suite, it reproduces all 81 three-state predicate combinations and rejects 192 deliberately malformed records. Common-guard baselines and predicate ablations isolate endpoint logic from missing-evidence handling, while controlled transitions check support separation and evidence removal. These are finite contract-conformance results, not detec
    
[^13]: 信任，但要验证工具：在自建安全回归代理上审计AI生成的RTL验证计划

    Trust, but Validate the Instrument: Auditing AI-Generated RTL Verification Plans on Authored Security-Regression Proxies

    [https://arxiv.org/abs/2609.19844](https://arxiv.org/abs/2609.19844)

    论文提出可审计框架SecTB-RTL，通过自建硬件安全回归测试发现AI生成的RTL验证计划虽被提供商接受但几乎全部无法通过生产语义验证，证明提供商模式的接受并不能等同于执行的有效性。

    

    AI生成的RTL验证计划可能满足提供商的模式，但在与可信执行的边界处失败。我们提出了SecTB-RTL，一个涵盖31个任务和124个自建硬件安全回归测试的可审计框架。确定性非AI基线在递增的资源限制下分别杀死了36、75和78个突变体。第一次确认性运行（C1-R2）在模型执行之前就失败了，因为提供商拒绝了其响应模式。在未查看结果的情况下进行仅模式修复后，单独冻结的后续运行（C1-R3）完成了1,860次调用。提供商接受了1,857个响应，但只有九个通过了生产语义验证器。生成规则和执行规则并不匹配。因此，我们将该运行保留为工具验证事件，不报告任何提示效果估计。这一事件表明，提供商或模式的接受并不能证明执行的有效性。编译和覆盖率仅是诊断指标。

    arXiv:2609.19844v1 Announce Type: cross  Abstract: AI-generated RTL verification plans can satisfy a provider schema yet fail at the boundary to trusted execution. We present SecTB-RTL, an auditable framework covering 31 tasks and 124 authored hardware-security regressions. A deterministic non-AI baseline killed 36, 75, and 78 mutants at increasing resource limits. The first confirmatory run (C1-R2) failed before model execution because the provider rejected its response schema. After a schema-only repair made without viewing outcomes, a separately frozen follow-up run (C1-R3) completed 1,860 calls. The provider accepted 1,857 responses, but only nine passed the production semantic validator. The generation and execution rules did not match. We therefore preserve the run as an instrument-validation incident and report no prompt-effect estimate. This incident shows that provider or schema acceptance does not establish execution validity. Compilation and coverage are only diagnostics; th
    
[^14]: 先想象后检索：面向大语言模型智能体的前瞻性技能检索

    Imagine Before Retrieval: Prospective Skill Retrieval for LLM Agents

    [https://arxiv.org/abs/2609.01642](https://arxiv.org/abs/2609.01642)

    提出受人类前瞻性认知启发的SkillDreamer框架，通过“先想象后检索”策略解决LLM智能体技能检索中任务查询与技能之间的失配（QSM）问题。

    

    技能检索最近已成为一种有前景的范式，用于从技能库中识别理想的执行指南，从而使大语言模型（LLM）智能体具备完成指定任务所需的程序性知识。为此，大多数现有方法通过定制检索模型或重新配置检索流程，以优先选择与任务查询语义最相关的技能。然而，我们通过实证发现，任务查询和技能天然是从不同视角构建的，即目标导向和过程导向，这导致了一个尚未被充分研究的问题，称为“查询-技能失配”（Query–Skill Misalignment, QSM）。显然，在QSM的背景下，关联理想的技能是困难的甚至是不可能的，从而阻碍了智能体正确地执行任务。作为补救措施，受人类前瞻性认知的启发，我们提出了SkillDreamer，一种用于缓解这一负面影响的全新框架……

    arXiv:2609.01642v1 Announce Type: new  Abstract: Skill retrieval has recently emerged as a promising paradigm for identifying the desirable execution guidelines from the skill gallery, thus equipping large language model (LLM) agents with the procedural knowledge to accomplish the specified task. To this end, most existing methods customize the retrieval model or reconfigure the retrieval pipeline to prioritize skills that are most semantically relevant to the task query. However, we empirically reveal that task queries and skills are naturally formulated from different perspectives, namely, objective-oriented and procedural-oriented, leading to an under-explored problem termed Query--Skill Misalignment (QSM). Clearly, it is daunting and even impossible to associate the desirable skills in the context of QSM, thus hindering the agent from correctly executing the task. As a remedy, inspired by human prospective cognition, we propose SkillDreamer, a novel framework to alleviate the negat
    
[^15]: UxSID：面向超长序列的语义感知用户兴趣建模

    UxSID: Semantic-Aware User Interests Modeling for Ultra-Long Sequence

    [https://arxiv.org/abs/2605.09040](https://arxiv.org/abs/2605.09040)

    UxSID 通过语义 ID 与双层注意力机制构建语义组共享兴趣记忆，为超长用户行为序列建模提供兼顾效率与效果的第三条路径，实现 SOTA 性能并在大规模广告 A/B 测试中带来 0.337% 的收入提升。

    

    对超长用户行为序列进行建模需要在效率与效果之间进行艰难的权衡。现有范式要么依赖特定物品的检索，要么采用与物品无关的压缩方式，我们提出 UxSID 框架，探索第三条路径：语义组共享兴趣记忆。通过利用语义 ID（SIDs）和双层注意力策略，UxSID 无需承担特定物品模型的沉重开销，即可捕捉目标感知的用户偏好。这一端到端架构在计算简洁性与语义感知之间取得平衡，实现了最先进的性能，并在大规模广告 A/B 测试中带来 0.337% 的收入提升。

    arXiv:2605.09040v4 Announce Type: replace  Abstract: Modeling ultra-long user sequences involves a difficult trade-off between efficiency and effectiveness. While current paradigms rely on either item-specific search or item-agnostic compression, we propose UxSID, a framework exploring a third path: semantic-group shared interest memory. By utilizing Semantic IDs (SIDs) and a dual-level attention strategy, UxSID captures target-aware preferences without the heavy cost of item-specific models. This end-to-end architecture balances computational parsimony with semantic awareness, achieving state-of-the-art performance and a 0.337% revenue lift in large-scale advertising A/B test.
    
[^16]: 用于预测细胞对基因扰动响应的检索增强生成

    Retrieval-Augmented Generation for Predicting Cellular Responses to Gene Perturbation

    [https://arxiv.org/abs/2603.07233](https://arxiv.org/abs/2603.07233)

    提出了PT-RAG，一个即插即用的两阶段检索增强生成模块，通过GenePT语义检索与可微分Gumbel-Softmax选择器动态获取相关的扰动上下文，从而改进单细胞基因扰动响应的预测。

    

    预测基因扰动的转录响应是功能基因组学和治疗发现的基础。最近的深度学习模型在单细胞扰动响应预测方面展现出了前景，但它们通常孤立地生成每个响应，而没有显式地利用相关的扰动信息。我们提出了PT-RAG（扰动感知的两阶段检索增强生成），这是一个用于生成式细胞扰动响应的即插即用检索与条件化模块。PT-RAG通过学习访问相关的扰动上下文来增强现有的扰动-响应骨干模型。关键挑战在于，在这种设置下相关性并非固定不变：功能相关的基因在不同细胞类型中可能引发不同的效应。PT-RAG通过两阶段检索机制解决这一问题：首先基于GenePT的语义检索识别K个候选扰动，随后由一个可微分的Gumbel-Softmax选择器…

    arXiv:2603.07233v2 Announce Type: replace  Abstract: Predicting transcriptional responses to genetic perturbations is fundamental to functional genomics and therapeutic discovery. Recent deep learning models have shown promise in single-cell perturbation response prediction, but they typically generate each response in isolation, without explicitly leveraging related perturbations. We introduce PT-RAG (Perturbation-aware Two-stage Retrieval-Augmented Generation), a plug-in retrieval-and-conditioning module for generative cellular perturbation response. PT-RAG augments an existing perturbation-response backbone with learned access to related perturbation contexts. The key challenge is that relevance is not fixed in this setting: functionally related genes may elicit different effects across cell types. PT-RAG addresses this with a two-stage retrieval mechanism: GenePT-based semantic retrieval first identifies K candidate perturbations, after which a differentiable Gumbel-Softmax selecto
    
[^17]: 面向大语言模型高效排序的访问路径

    Access Paths for Efficient Ordering with Large Language Models

    [https://arxiv.org/abs/2509.00303](https://arxiv.org/abs/2509.00303)

    提出了LLM ORDER BY语义算子及预算感知优化器，通过动态选择近优访问路径，在所有基准测试中实现了与最佳静态方法相当或更优的排序准确率。

    

    在这项工作中，我们提出了 \texttt{LLM ORDER BY} 语义算子作为一种逻辑抽象，并对其物理实现进行了系统性研究。首先，我们对现有的语义排序算法提出了若干改进，并引入了一种语义感知的外部归并排序算法。我们的大量评估表明，没有任何单一实现在所有数据集上都能达到普遍最优。从评估中，我们观察到基于比较的算法在排序成本与排序质量之间存在普遍的扩展关系。基于这些洞察，我们设计了一个预算感知的优化器，该优化器利用启发式规则、LLM-as-Judge 评估和共识聚合来动态选择 LLM ORDER BY 的近优访问路径。在我们的大量评估中，我们的优化器在所有基准测试中始终实现了与最佳静态方法相当或更优的排序准确率。我们相信，这

    arXiv:2509.00303v4 Announce Type: replace-cross  Abstract: In this work, we present the \texttt{LLM ORDER BY} semantic operator as a logical abstraction and conduct a systematic study of its physical implementations. First, we propose several improvements to existing semantic sorting algorithms and introduce a semantic-aware external merge sort algorithm. Our extensive evaluation reveals that no single implementation offers universal optimality on all datasets. From our evaluations, we observe a general scaling relationship between sorting cost and the ordering quality for comparison-based algorithms. Building on these insights, we design a budget-aware optimizer that utilizes heuristic rules, LLM-as-Judge evaluation, and consensus aggregation to dynamically select the near-optimal access path for LLM ORDER BY. In our extensive evaluations, our optimizer consistently achieves ranking accuracy on par with or superior to the best static methods across all benchmarks. We believe that this
    
[^18]: CRM：具备可控条件的检索模型

    CRM: Retrieval Model with Controllable Condition

    [https://arxiv.org/abs/2412.13844](https://arxiv.org/abs/2412.13844)

    提出可控检索模型（CRM），将观看时长作为用户塔的条件输入，使模型在推理阶段无需重新训练即可通过指定期望观看时长来引导检索，同时保持物品向量可预先计算以支持高效的 ANN 检索。

    

    面向推荐的检索模型需要在高效搜索大规模物品语料库的同时，兼顾观看时长等结果指标。我们提出了可控检索模型，这是一个以结果为条件的检索框架，它将观看时长作为用户塔的输入。在训练阶段，CRM 学习在给定观测到的观看时长的条件下预测下一个物品；在推理阶段，我们输入期望的观看时长来引导检索，而无需重新训练。由于物品塔与该条件无关，物品向量仍然可以被预先计算并建立索引，用于近似最近邻（ANN）搜索。我们将 CRM 同时实现为基于 DNN 的双塔模型和 Decision Transformer 两种形式。其中 Transformer 将物品 token 与剩余观看时长条件 token 交错排列，并使用目标观看时长作为数值提示来进行下一物品预测。对于每个在线请求，我们在用户近期历史记录的平均观看时长与最大观看时长之间进行选择。

    arXiv:2412.13844v2 Announce Type: replace-cross  Abstract: Retrieval models for recommendation need to account for outcomes such as watch time while searching large item corpora efficiently. We present the Controllable Retrieval Model (CRM), an outcome-conditioned retrieval framework that adds watch time as an input to the user tower. During training, CRM learns to predict the next item given its observed watch time; at inference, we supply a desired watch time to steer retrieval without retraining. The item tower is independent of the condition, so item embeddings can still be precomputed and indexed for approximate nearest neighbor (ANN) search. We implement CRM as both a DNN-based two-tower model and a Decision Transformer. The Transformer interleaves item tokens with watch-time-to-go condition tokens, using the target watch time as a numerical prompt for next-item prediction. For each online request, we choose between the average and maximum watch times in the user's recent history
    

