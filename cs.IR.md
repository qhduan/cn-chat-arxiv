# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Making Collaborative Signals Count: Graph-Aware Large Language Models for Sequential Recommendation](https://arxiv.org/abs/2608.12184) | 该论文提出GALLM，一种图感知大语言模型框架，通过在文本和物品标记上构建协同图来建模文本-文本、物品-文本和物品-物品三种关系，从而有效捕捉全局协同信号，提升序列推荐性能。 |
| [^2] | [A corpus-specific clinical RAG system matches or outperforms newer frontier LLMs on HealthBench](https://arxiv.org/abs/2608.12138) | VITA是一个针对低收入和中等收入环境定制的临床RAG系统，通过从特定语料库检索知识，在HealthBench基准上超越或匹敌更新的前沿LLM，并公开其评估过程以确保透明性。 |
| [^3] | [Token-Level Credit Assignment Optimization for Generative Document Retrieval](https://arxiv.org/abs/2608.12049) | 本文提出了一种针对生成式文档检索的细粒度强化学习框架，通过分配令牌级相关性奖励，解决了现有方法中序列级奖励导致的信用分配模糊问题，从而提升检索性能。 |
| [^4] | [HCGRec: Hint-Conditioned Generative Recommendation with Semantic IDs](https://arxiv.org/abs/2608.11980) | HCGRec通过提供最小目标前缀提示来克服语义ID生成式推荐中因早期令牌错误分支导致的零奖励问题，从而恢复困难实例的学习信号。 |
| [^5] | [Sci-Surf: Navigating Scientific Literature Discovery through Human Feedback and Intelligent Summarizatio](https://arxiv.org/abs/2608.11973) | Sci-Surf通过集成人类反馈驱动的个性化推荐和基于LLM的多模态博客式论文摘要，实现了意图中心的科学文献发现，显著提升了推荐与理解质量。 |
| [^6] | [LODESTAR: Trustworthy Entropy Is Navigated, Not Merely Measured -- Reinforced Polarizer Keeps a Frozen LLM from Being Confidently Misled by the Wrong Evidence](https://arxiv.org/abs/2608.11922) | LODESTAR通过干预文本上下文并利用第三方冻结LLM的不确定性评分，解决了最低熵规则在误导性段落上导致自信错误的问题，从而提升了检索增强问答的可靠性。 |
| [^7] | [DexterSQL: Deep Schema Exploration and Rule-based Correction for Text-to-SQL Generation](https://arxiv.org/abs/2608.11889) | DexterSQL通过深度模式探索、数据库无关规则挖掘和规则驱动修正三个创新组件，解决了非微调文本到SQL生成中模式信息粗糙、错误重复出现和条件处理不当的问题。 |
| [^8] | [Total Recall at What Cost? Benchmarking the Serving Cost of Agentic Memory Systems](https://arxiv.org/abs/2608.11879) | 该论文通过基准测试发现，代理记忆系统的服务成本主要由内部记忆行为而非对话长度决定，且其相对于完整转录的成本优势高度依赖于系统实现。 |
| [^9] | [From Overlooked to Explored: Recovering Item Relations via Mixture of Perspectives for Sequential Recommendation](https://arxiv.org/abs/2608.11846) | 本文提出PRISM模块，通过多视角混合方法克服序列推荐中自注意力模型的相似性偏差，从而恢复被忽视的异质物品关系以提升推荐性能。 |
| [^10] | [AgenticTwin: An Agentic LLM Framework Integrated with Digital Twin for Anomaly Detection](https://arxiv.org/abs/2608.11679) | 提出了AgenticTwin框架，通过将LLM推理与数字孪生异常检测管道集成，实现了异常解释的自然语言交互和基准化评估。 |
| [^11] | [FunnelCausalNet: Funnel-aware Joint Conversion-Revenue Uplift for Multi-tier Coupon Allocation](https://arxiv.org/abs/2608.11675) | 本文提出FunnelCausalNet，通过将转化与条件价值联合建模并利用漏斗结构，实现了多层优惠券分配中转化率和收入的同时提升，且提供了方差降低的启发式分析和预算分配方案。 |
| [^12] | [Defending against Model Extraction for GNNs with Model Reprogramming](https://arxiv.org/abs/2608.11495) | 本文提出GraphRP框架，利用模型重编程和结构感知门控机制，构建动态“结构防火墙”以主动防御GNN模型提取攻击，克服了现有方法忽视拓扑依赖导致的实用性下降问题。 |
| [^13] | [TRACES: A Benchmark for Epistemic Reliability in Scientific Reasoning by LLMs](https://arxiv.org/abs/2608.11415) | 本文提出了TRACES基准，通过42篇有缺陷论文的探测语料库，首次直接评估大型语言模型在科学推理中区分可靠与不可靠文献的认知可靠性。 |
| [^14] | [Exploring the Social Life of Data: Finding Data You Can Trust](https://arxiv.org/abs/2608.11395) | 本文提出利用数据使用图作为新基础设施，通过积累的社会和经验证据来评估和确定数据的适用性与可信度，以应对AI时代寻找可信数据的挑战。 |
| [^15] | [Can Frontier LLMs Match Natively Multimodal Embeddings? A Comparison on Hard-Negative Text-to-Image Retrieval](https://arxiv.org/abs/2608.11343) | 本研究首次直接比较了原生多模态嵌入（如Gemini Embedding 2）与前沿大语言模型（如GPT-4.1和Claude Sonnet 4.6）在硬负样本文本到图像检索任务中的表现，发现LLM在零样本排序上可与原生多模态嵌入媲美，但后者在预计算场景下更具优势。 |
| [^16] | [RecSys Factory: Bounding LLM Agent Autonomy to Decision Points in the Industrial Recommender Lifecycle](https://arxiv.org/abs/2608.11241) | 本文提出RecSys Factory平台，通过将LLM代理的自主性限定在决策点而非整个管道，并采用无守护进程的事件驱动架构，解决了工业推荐系统中自主性、确定性和效率之间的三元悖论。 |
| [^17] | [Sona Technical Report](https://arxiv.org/abs/2608.11015) | Sona通过共享用户表示和联合训练目标，在单一模型中统一了候选生成与排序，无需手工特征，并在在线测试中取代了包含15个以上候选生成器的生产级联架构，显著提升了参与度指标。 |
| [^18] | [ENTLORE: A Graph-Grounded Benchmark for Latent Organizational Reasoning in Enterprise Question Answering](https://arxiv.org/abs/2608.10679) | ENTLORE通过图基框架重建企业世界，专注于测试从常规文档中恢复隐含组织关系的能力，而非仅组合已有事实。 |
| [^19] | [When Do Anchor-Based Pointwise LLM Rerankers Help? Retriever Quality, Statistical Scope, and Anchor Design](https://arxiv.org/abs/2608.10528) | 本研究通过复现和组件级压力测试发现，基于锚点的逐点LLM重排序方法在统计校正下核心思想稳健，但原始论文中的两个设计选择（如锚点设计）在不同条件下可靠性不足。 |
| [^20] | [MISO: Model-Internal-State-Guided Optimization for Ranking Models](https://arxiv.org/abs/2608.07035) | MISO利用模型内部状态指导排序模型的局部优化决策，以减少试错成本并提高效率。 |
| [^21] | [VLM2Rec: Resolving Modality Collapse in Vision-Language Model Embedders for Multimodal Sequential Recommendation](https://arxiv.org/abs/2603.17450) | 本文提出VLM2Rec框架，通过弱模态惩罚机制解决视觉-语言模型嵌入器在多模态序列推荐中的模态坍缩问题，实现模态利用平衡并提升推荐准确性。 |

# 详细

[^1]: 让协同信号发挥作用：面向序列推荐的图感知大语言模型

    Making Collaborative Signals Count: Graph-Aware Large Language Models for Sequential Recommendation

    [https://arxiv.org/abs/2608.12184](https://arxiv.org/abs/2608.12184)

    该论文提出GALLM，一种图感知大语言模型框架，通过在文本和物品标记上构建协同图来建模文本-文本、物品-文本和物品-物品三种关系，从而有效捕捉全局协同信号，提升序列推荐性能。

    

    arXiv:2608.12184v1 公告类型：新论文 摘要：大语言模型（LLMs）已被广泛用作推荐系统的骨干模型。然而，它们以语言为中心的预训练使其难以捕捉用户-物品交互中隐含的协同信号，而这些信号对于个性化推荐至关重要。现有方法要么注入由外部推荐器生成的协同表示，要么仅建模序列内部依赖，限制了其利用全局协同模式的能力。为解决这一局限，我们提出了GALLM，一种用于序列推荐的图感知大语言模型框架。GALLM在文本标记和物品标记上构建协同图，并建模三种关系：文本-文本关系用于保留语义依赖，物品-文本关系用于将物品标记与其文本描述对齐，以及从全局物品共现模式中推导出的物品-物品关系。这些关系被转化为轻量级...

    arXiv:2608.12184v1 Announce Type: new  Abstract: Large language models (LLMs) have been widely adopted as backbones for recommender systems. However, their language-centric pretraining makes it difficult to capture collaborative signals implicit in user-item interactions, which are crucial for personalized recommendation. Existing methods either inject collaborative representations produced by external recommenders or model only intra-sequence dependencies, limiting their ability to exploit global collaborative patterns. To address this limitation, we propose GALLM, a graph-aware LLM framework for sequential recommendation. GALLM constructs a collaborative graph over text tokens and item tokens, and models three types of relations: Text--Text relations for preserving semantic dependencies, Item--Text relations for aligning item tokens with their textual descriptions, and Item--Item relations derived from global item co-occurrence patterns. These relations are transformed into lightweig
    
[^2]: 针对特定语料库的临床RAG系统在HealthBench上达到或超越更新的前沿LLM

    A corpus-specific clinical RAG system matches or outperforms newer frontier LLMs on HealthBench

    [https://arxiv.org/abs/2608.12138](https://arxiv.org/abs/2608.12138)

    VITA是一个针对低收入和中等收入环境定制的临床RAG系统，通过从特定语料库检索知识，在HealthBench基准上超越或匹敌更新的前沿LLM，并公开其评估过程以确保透明性。

    

    arXiv:2608.12138v1 公告类型：交叉 摘要：通用大型语言模型（LLM）近期在医学基准测试中被报道达到或超过专门化的临床AI工具，但这些比较基于狭窄的系统集和主要在高收入环境中开发的基准。我们评估了VITA，一个为印度及其他低收入和中等收入（LMIC）环境中的上下文知识检索而专门构建的检索增强生成（RAG）系统。VITA从经过整理的疾病特定指南、印度特定抗菌药物耐药性数据、国家处方集限制和资源有限护理协议中检索信息；其架构和语料库是专有的，但基准测试、医生编写的评分标准以及我们的完整回答和评分输出是公开的，以供独立验证。在4,023个英文HealthBench问题（占基准的80.5%）上，使用GPT-4.1评判器评分，VITA以51.9%的潜在评分点排名第一。

    arXiv:2608.12138v1 Announce Type: cross  Abstract: General-purpose large language models (LLMs) have recently been reported to match or exceed specialized clinical AI tools on medical benchmarks, but such comparisons draw on a narrow set of systems and on benchmarks developed largely in high-income settings. We evaluate VITA, a retrieval-augmented generation (RAG) system purpose-built for contextual knowledge retrieval in India and other low- and middle-income (LMIC) settings. VITA retrieves from a curated corpus of disease-specific guidelines, India-specific antimicrobial resistance data, national formulary constraints, and resource-limited care protocols; its architecture and corpus are proprietary, but the benchmark, the physician-written rubrics, and our full response and scoring outputs are public for independent verification. On 4,023 English-language HealthBench questions (80.5% of the benchmark), scored with a GPT-4.1 judge, VITA ranked first with 51.9% of possible rubric point
    
[^3]: 生成式文档检索中的令牌级信用分配优化

    Token-Level Credit Assignment Optimization for Generative Document Retrieval

    [https://arxiv.org/abs/2608.12049](https://arxiv.org/abs/2608.12049)

    本文提出了一种针对生成式文档检索的细粒度强化学习框架，通过分配令牌级相关性奖励，解决了现有方法中序列级奖励导致的信用分配模糊问题，从而提升检索性能。

    

    arXiv:2608.12049v1 公告类型：新 摘要：生成式检索模型通过自回归生成文档标识符（DocIDs）来执行文档检索。该过程自然形成一个顺序决策问题，其中每个解码步骤选择一个DocID令牌，完整的令牌序列决定检索到的文档。然而，检索效果通常仅在完整DocID生成后进行评估，这导致令牌级生成与文档级相关性监督之间存在不匹配。因此，现有的生成式检索强化学习方法大多依赖序列级奖励，将相同的文档级反馈传播到所有解码步骤。这种粗粒度反馈使得难以识别哪些令牌决策对检索成功或失败负责。在本工作中，我们提出了一种基于令牌级相关性奖励的生成式检索细粒度强化学习框架，而非仅依赖文档级反馈。

    arXiv:2608.12049v1 Announce Type: new  Abstract: Generative retrieval models perform document retrieval by autoregressively generating document identifiers (DocIDs). This process naturally forms a sequential decision problem, where each decoding step selects a DocID token and the complete token sequence determines the retrieved document. However, retrieval effectiveness is typically evaluated only after the full DocID is generated, creating a mismatch between token-level generation and document-level relevance supervision. As a result, existing reinforcement learning methods for generative retrieval mostly rely on sequence-level rewards, where the same document-level feedback is propagated to all decoding steps. Such coarse-grained feedback makes it difficult to identify which token decisions are responsible for successful or failed retrieval.   In this work, we propose a fine-grained reinforcement learning framework for generative retrieval with token-level relevance rewards. Instead 
    
[^4]: HCGRec：基于提示条件的生成式推荐与语义ID

    HCGRec: Hint-Conditioned Generative Recommendation with Semantic IDs

    [https://arxiv.org/abs/2608.11980](https://arxiv.org/abs/2608.11980)

    HCGRec通过提供最小目标前缀提示来克服语义ID生成式推荐中因早期令牌错误分支导致的零奖励问题，从而恢复困难实例的学习信号。

    

    arXiv:2608.11980v1 公告类型：交叉 摘要：语义ID生成式推荐器将每个项目表示为一系列离散的语义令牌，并通过自回归生成该令牌序列来预测下一个项目。这种范式为项目ID、历史和项目文本提供了统一的生成接口，但在基于奖励的后训练过程中也造成了结构化优化瓶颈：当早期语义令牌进入项目令牌空间的错误分支时，有限的展开组很少能到达真实项目，因此组相对优化接收到相同的零奖励，并产生无用的优势。我们提出了基于提示条件的生成式推荐（HCGRec），一种语义ID生成式推荐框架，用于恢复此类困难训练实例的学习信号。HCGRec通过检查点展开诊断每个实例，并仅在当前生成器无法到达正确项目时提供最小目标前缀提示。该模型...

    arXiv:2608.11980v1 Announce Type: cross  Abstract: Semantic-ID generative recommenders represent each item as a short sequence of discrete semantic tokens and predict the next item by autoregressively generating this token sequence. This paradigm enables a unified generation interface for item IDs, histories, and item text, but it also creates a structured optimization bottleneck during reward-based post-training: when an early semantic token enters the wrong branch of the item-token space, finite rollout groups rarely reach the ground-truth item, so group-relative optimization receives identical zero rewards and produces no useful advantage. We propose Hint-Conditioned Generative Recommendation (HCGRec), a semantic-ID generative recommendation framework that recovers learning signal for such hard training instances. HCGRec diagnoses each instance with checkpoint rollouts and supplies a minimal target-prefix hint only when the current generator cannot reach the correct item. The model 
    
[^5]: Sci-Surf：通过人类反馈与智能摘要导航科学文献发现

    Sci-Surf: Navigating Scientific Literature Discovery through Human Feedback and Intelligent Summarizatio

    [https://arxiv.org/abs/2608.11973](https://arxiv.org/abs/2608.11973)

    Sci-Surf通过集成人类反馈驱动的个性化推荐和基于LLM的多模态博客式论文摘要，实现了意图中心的科学文献发现，显著提升了推荐与理解质量。

    

    arXiv:2608.11973v1 公告类型：新 摘要：科学出版物的快速增长使得研究人员越来越难以识别相关的新研究并有效理解它们。现有的学术发现平台通常依赖静态主题订阅或基于嵌入的相似性，仅提供摘要或简短总结，对细致意图建模和深入论文摘要的支持有限。我们提出了Sci-Surf，一个以意图为中心的知识发现系统，它集成了反馈驱动的个性化推荐与多模态博客式论文消化。我们的方法通过基于LLM的用户画像细化用户意图表示，同时生成结构化摘要，综合全文中的文本和视觉信息。该演示展示了一个端到端的学术发现流程，并通过真实用户评估在推荐质量和消化质量方面均显示出可衡量的改进。

    arXiv:2608.11973v1 Announce Type: new  Abstract: The rapid growth of scientific publications makes it increasingly difficult for researchers to identify relevant new studies and effectively comprehend them. Existing academic discovery platforms typically rely on static topic subscriptions or embedding-based similarity and provide only abstracts or short summaries, offering limited support for nuanced intent modeling and in-depth paper summarization. We present Sci-Surf, an intent-centric knowledge discovery system that integrates feedback-driven personalized recommendation with multi-modal blog-style paper digestion. Our approach refines user intent representations through LLM-based user profiling, while generating structured summaries that synthesize textual and visual information from full papers. The demo presents an end-to-end academic discovery pipeline and demonstrates measurable improvements in both recommendation quality and digestion quality through real-user evaluations. Spec
    
[^6]: LODESTAR：可信熵是被引导的，而非仅被测量——强化极化器防止冻结LLM被错误证据自信误导

    LODESTAR: Trustworthy Entropy Is Navigated, Not Merely Measured -- Reinforced Polarizer Keeps a Frozen LLM from Being Confidently Misled by the Wrong Evidence

    [https://arxiv.org/abs/2608.11922](https://arxiv.org/abs/2608.11922)

    LODESTAR通过干预文本上下文并利用第三方冻结LLM的不确定性评分，解决了最低熵规则在误导性段落上导致自信错误的问题，从而提升了检索增强问答的可靠性。

    

    arXiv:2608.11922v1 公告类型：新 摘要：预测分布熵在检索增强的问答中构成了一个强大的选择规则：在五个问答基准测试中，保持冻结回答型LLM对候选答案生成的最低答案令牌熵，将平均答案F1分数从0.4769提升至0.5148，相对于检索器排名最高的段落，且无需金标准答案。然而，这种最低熵规则（先前的基于熵的选择器所采用的）在一种具体且重要的方式上失败：一个误导性段落使回答者自信地犯错，恰好在其信号看起来最可信的地方降低了熵。我们表明，这种失败源于回答者所读的段落——而该段落被阅读的上下文是我们可干预的输入。我们引入了LODESTAR，据我们所知，这是第一种通过其在第三方冻结回答者中引发的不确定性来评分文本干预的方法，并跨一个问题的候选进行比较。LODESTAR使用

    arXiv:2608.11922v1 Announce Type: new  Abstract: Predictive-distribution entropy makes a strong selection rule in retrieval-augmented question answering: across five QA benchmarks, keeping the candidate answer that a frozen respondent LLM produces with the lowest answer-token entropy lifts mean answer $F_1$ from 0.4769 to 0.5148 over the retriever's top-ranked passage, with no gold answers. Yet this lowest-entropy rule, which prior entropy-based selectors adopt, fails in a specific and consequential way: a misleading passage makes the respondent confidently wrong, driving its entropy down precisely where the signal looks most trustworthy. We show that the failure comes from the passage the respondent reads -- and the context that passage is read in is an input we can intervene on. We introduce LODESTAR, to our knowledge the first method to score a text intervention by the uncertainty it induces in a third-party frozen respondent, compared across one question's candidates. LODESTAR uses
    
[^7]: DexterSQL：面向文本到SQL生成的深度模式探索与基于规则的修正

    DexterSQL: Deep Schema Exploration and Rule-based Correction for Text-to-SQL Generation

    [https://arxiv.org/abs/2608.11889](https://arxiv.org/abs/2608.11889)

    DexterSQL通过深度模式探索、数据库无关规则挖掘和规则驱动修正三个创新组件，解决了非微调文本到SQL生成中模式信息粗糙、错误重复出现和条件处理不当的问题。

    

    arXiv:2608.11889v1 公告类型：交叉 摘要：基于提示（即非微调）的文本到SQL方法，其中底层大语言模型参数不针对任务进行更改，面临三个问题：（i）依赖粗粒度的模式信息，这可能无法揭示区分模糊列所需的细粒度关系，（ii）未能捕捉重复出现的SQL生成失败，以及（iii）在复杂问题中遭受条件遗漏、幻觉或错位。本文开发了DexterSQL，一个基于提示/非微调的文本到SQL系统，通过三个新组件改进SQL生成：（i）深度模式探索器，识别模糊列，分析其单独和联合数据分布以揭示它们之间的关系及各自的不同作用，（ii）数据库无关的规则创建器，挖掘生成结果与目标结果之间的不匹配，（iii）规则驱动的修正器，应用这些规则来纠正SQL生成中的常见错误。

    arXiv:2608.11889v1 Announce Type: cross  Abstract: Prompting-based (\textit{i}.\textit{e}., non-fine-tuning) Text-to-SQL methods, where underlying large language model parameters are not changed for the task, face three problems: (\textit{i})~relying on coarse-grained schema information that may not reveal the fine-grained relationships needed to distinguish ambiguous columns, (\textit{ii})~not capturing recurring SQL-generation failures, and (\textit{iii})~suffering from omission, hallucination, or misplacement of conditions in complex questions.   This paper develops \textsc{DexterSQL}, a prompting/non-fine-tuning-based Text-to-SQL system that improves SQL generation with three novel components: (\textit{i})~\emph{deep schema explorator} that identifies ambiguous columns, analyzes their individual and joint data distributions to uncover their relationships and the distinct role of each, (\textit{ii})~\emph{database-agnostic rule creator} that mines mismatches between generated and go
    
[^8]: 全面记忆，代价几何？代理记忆系统服务成本的基准测试

    Total Recall at What Cost? Benchmarking the Serving Cost of Agentic Memory Systems

    [https://arxiv.org/abs/2608.11879](https://arxiv.org/abs/2608.11879)

    该论文通过基准测试发现，代理记忆系统的服务成本主要由内部记忆行为而非对话长度决定，且其相对于完整转录的成本优势高度依赖于系统实现。

    

    长期运行的对话代理越来越依赖记忆系统来避免每轮重新发送整个对话，然而服务这些记忆系统的成本却鲜有系统性的基准测试。我们比较了三种记忆系统（Mem0、Hindsight和Mastra观察性记忆）与两种参考策略——固定大小的滚动窗口和重新提交完整转录——在两种骨干模型和长达400轮对话上的表现，并将每次成本测量与665个LoCoMo问题的回答准确性配对。首先，记忆系统的服务成本无法仅从对话长度和消息大小预测：一个追踪两种参考策略的回归模型与记忆系统的实际成本相差18-69%，其成本反而由内部记忆行为驱动。其次，盈亏平衡分析表明，记忆系统是否以及何时比完整转录更便宜地服务，高度依赖于系统的具体实现。

    arXiv:2608.11879v1 Announce Type: new  Abstract: Long-running conversational agents increasingly rely on a memory system to avoid resending the whole conversation each turn, yet how much that costs to serve has received little systematic benchmarking. We compare three memory systems (Mem0, Hindsight, and Mastra Observational Memory) against two reference strategies -- a fixed-size rolling window and resubmitting the full transcript -- across two backbones and conversations of up to 400 turns, pairing every cost measurement with answer accuracy on 665 LoCoMo questions. First, a memory system's serving cost cannot be predicted from conversation length and message size alone: a regression that tracks the two reference strategies closely misses the memory systems by 18-69%, their cost driven instead by internal memory behavior. Second, a break-even analysis shows that whether -- and when -- a memory system becomes cheaper to serve than the full transcript is highly sensitive to the system 
    
[^9]: 从被忽视到被探索：通过多视角混合恢复物品关系以用于序列推荐

    From Overlooked to Explored: Recovering Item Relations via Mixture of Perspectives for Sequential Recommendation

    [https://arxiv.org/abs/2608.11846](https://arxiv.org/abs/2608.11846)

    本文提出PRISM模块，通过多视角混合方法克服序列推荐中自注意力模型的相似性偏差，从而恢复被忽视的异质物品关系以提升推荐性能。

    

    从用户的交互序列中捕捉用户偏好是序列推荐（SR）的核心挑战。这种偏好直观上源于物品间的关系：每个物品转换反映了嵌入在物品关系中的偏好，因此忠实捕捉这些关系对于准确推荐至关重要。为此，自注意力在序列推荐中占主导地位，因为它能计算成对物品交互，但我们的实证分析揭示，它在各种基于Transformer的SR模型中持续遭受相似性偏差：点积注意力分数不成比例地偏向相似物品，系统性地忽视了具有有意义偏好信号的异质关系，直接限制了推荐性能。为解决这一问题，我们提出了PRISM（基于视角的关系洞察合成模块），一个重新审视物品关系的模块。

    arXiv:2608.11846v1 Announce Type: new  Abstract: Capturing user preference from a user's interaction sequence is the central challenge of Sequential Recommendation (SR). This preference intuitively emerges from inter-item relations: each item transition reflects a preference embedded in the relations between items, making the faithful capture of these relations essential for accurate recommendation. For this reason, self-attention is dominant in sequential recommendation for its ability to compute pairwise item interactions, yet our empirical analysis reveals that it consistently suffers from similarity bias across various types of transformer-based SR models: dot-product attention scores disproportionately favor similar items, systematically overlooking heterogeneous relations with meaningful preference signals and directly limiting recommendation performance. To address this, we propose PRISM (Perspective-based Relational Insight Synthesis Module), a module that re-examines item rela
    
[^10]: AgenticTwin：一种集成数字孪生的智能体LLM框架用于异常检测

    AgenticTwin: An Agentic LLM Framework Integrated with Digital Twin for Anomaly Detection

    [https://arxiv.org/abs/2608.11679](https://arxiv.org/abs/2608.11679)

    提出了AgenticTwin框架，通过将LLM推理与数字孪生异常检测管道集成，实现了异常解释的自然语言交互和基准化评估。

    

    数字孪生越来越多地被用于监控和模拟网络物理系统的行为。即使有经验丰富的操作员，解读数字孪生管道中检测到的异常仍然具有挑战性，因为原始传感器数据的复杂性和数量使得深入分析变得困难。近年来，大型语言模型（LLMs）的进步为推理和解释提供了有前景的能力，但将其集成到数字孪生驱动的异常分析中仍未被充分探索。在这项工作中，我们提出了AgenticTwin，一种将LLM驱动的推理与基于数字孪生的异常检测管道相结合的智能体框架。该框架将LLM生成的解释锚定在数字孪生驱动的异常分类器输出中，并使人类操作员能够就系统提出相关的自然语言问题。除框架本身外，我们还引入了一个基于合成异常构建的基准导向评估管道。

    arXiv:2608.11679v1 Announce Type: new  Abstract: Digital twins are increasingly used to monitor and simulate the behavior of cyber-physical systems. Even with skilled operators, interpreting anomalies detected within digital twin pipelines is challenging, as the sheer complexity and volume of raw sensor data make thorough analysis difficult. Recent advances in large language models (LLMs) offer promising capabilities for reasoning and explanation, yet their integration into digital twin-driven anomaly analysis remains underexplored. In this work, we propose AgenticTwin, an agentic framework that integrates LLM-driven reasoning with a digital twin-based anomaly detection pipeline. The framework grounds LLM-generated explanations in outputs from a digital twin-driven anomaly classifier and enables human operators to ask relevant natural-language questions about the system. Beyond the framework itself, we introduce a benchmark-oriented evaluation pipeline constructed over synthetic anomal
    
[^11]: 漏斗因果网：面向多层优惠券分配的多层级转化-收入提升联合建模

    FunnelCausalNet: Funnel-aware Joint Conversion-Revenue Uplift for Multi-tier Coupon Allocation

    [https://arxiv.org/abs/2608.11675](https://arxiv.org/abs/2608.11675)

    本文提出FunnelCausalNet，通过将转化与条件价值联合建模并利用漏斗结构，实现了多层优惠券分配中转化率和收入的同时提升，且提供了方差降低的启发式分析和预算分配方案。

    

    优惠券活动旨在同时提升转化率和收入，但商品交易总额（GMV）遵循从转化到条件订单价值的确定性漏斗，且具有零膨胀和重尾特征。我们提出了FunnelCausalNet，一种提升估计器，通过μ_gmv = μ_conv × μ_val将二元转化头与非负条件价值头耦合。在显式随机对照试验（RCT）、支持度、率差和跨头协方差控制假设下，一个理想化的主导阶均方误差比较识别出一个漏斗组合可降低逐点方差的区间；这是一个启发式方法，而非共享表示神经模型的保证。该估计器与边际分裂共形CATE摘要配对，通过Bonferroni并集组合为审计带，并使用基于RCT锚定估计的拉格朗日预算分配器进行补贴感知的ROI核算。在半合成的多层级Criteo-MT7数据集上，Fun...

    arXiv:2608.11675v1 Announce Type: new  Abstract: Coupon campaigns seek to lift both conversion and revenue, but gross merchandise value (GMV) follows a deterministic funnel from conversion to conditional order value and is zero-inflated and heavy-tailed. We propose FunnelCausalNet, an uplift estimator coupling a binary conversion head with a nonnegative conditional-value head through $\mu_{\mathrm{gmv}}=\mu_{\mathrm{conv}}\mu_{\mathrm{val}}$. Under explicit RCT, support, rate-gap, and cross-head covariance-control assumptions, an idealized leading-order MSE comparison identifies a regime in which funnel composition can reduce pointwise variance; this is a heuristic, not a guarantee for the shared-representation neural model. The estimator is paired with marginal split-conformal CATE summaries, combined through a Bonferroni union as audit bands, and a Lagrangian budgeted allocator using RCT-anchored estimates for subsidy-aware ROI accounting. On semi-synthetic multi-tier Criteo-MT7, Fun
    
[^12]: 针对GNN模型提取攻击的模型重编程防御方法

    Defending against Model Extraction for GNNs with Model Reprogramming

    [https://arxiv.org/abs/2608.11495](https://arxiv.org/abs/2608.11495)

    本文提出GraphRP框架，利用模型重编程和结构感知门控机制，构建动态“结构防火墙”以主动防御GNN模型提取攻击，克服了现有方法忽视拓扑依赖导致的实用性下降问题。

    

    arXiv:2608.11495v1 公告类型：新 摘要：图神经网络（GNNs）作为机器学习即服务（MLaaS）中高风险应用的核心支撑。然而，其黑盒部署使其暴露于模型提取（ME）攻击之下，在这种攻击中，对手通过查询API来窃取知识产权。现有防御方法存在严重的“欧几里得偏差”：它们将基于图像的策略（如随机噪声）迁移到图上，忽略了节点之间复杂的拓扑依赖关系，这常常导致严重的实用性下降。诸如水印之类的被动方法也无法实时阻止窃取行为。为弥补这一差距，我们提出了GraphRP（图重编程保护），一种主动防御框架，将模型重编程重新用于安全目的。与静态扰动不同，GraphRP引入了一种由可学习拓扑原型驱动的结构感知门控机制。这创建了一个动态的“结构防火墙”，可选择性地调节模型的行为。

    arXiv:2608.11495v1 Announce Type: new  Abstract: Graph Neural Networks (GNNs) serve as the backbone for high-stakes applications in Machine-Learning-as-a-Service (MLaaS). Still, their black-box deployment exposes them to Model Extraction (ME) attacks, in which adversaries steal intellectual property by querying APIs. Existing defenses suffer from a critical ''Euclidean bias'': they transfer image-based strategies (e.g., random noise) to graphs, ignoring the complex topological dependencies between nodes, which often results in severe utility degradation. Passive methods like watermarking also fail to prevent theft in real time. To bridge this gap, we propose GraphRP (Graph Reprogramming Protection), a proactive defense framework that repurposes Model Reprogramming for security. Unlike static perturbations, GraphRP introduces a Structure-Aware Gating Mechanism driven by learnable topological prototypes. This creates a dynamic ''structural firewall'' that selectively modulates the model'
    
[^13]: TRACES：大型语言模型科学推理中认知可靠性基准

    TRACES: A Benchmark for Epistemic Reliability in Scientific Reasoning by LLMs

    [https://arxiv.org/abs/2608.11415](https://arxiv.org/abs/2608.11415)

    本文提出了TRACES基准，通过42篇有缺陷论文的探测语料库，首次直接评估大型语言模型在科学推理中区分可靠与不可靠文献的认知可靠性。

    

    大型语言模型被提议作为科学工作流程中的代理，特别是在没有下游验证器的领域。这种部署假设模型能够区分可靠的科学文献与不可靠的文献，但这一能力尚未被直接测量。现有基准评估的是已知答案问题的事实性；而我们针对的失败模式不同。我们引入了一个包含42篇被撤回、欺诈性和伪科学论文的探测语料库，并配以方法论，用于引发和评分模型对每篇论文框架的单次参与。每个探测将目标论文中近乎逐字提取的前言与科学上合理的研究设计请求配对。这些探测涵盖五种声明类型：虚构观察、伪物理机制、魔法前提、合法化桥梁和仿冒实验。两个互补的评分指标衡量模型是否拒绝有缺陷的前提。

    arXiv:2608.11415v1 Announce Type: cross  Abstract: Large language models are being proposed as agents in scientific workflows, in domains where no downstream verifier exists. Such deployment assumes the model can distinguish reliable scientific literature from unreliable literature, a capability that has not yet been directly measured. Existing benchmarks evaluate factuality on questions with known answers; the failure mode we target here is different. We introduce a probe corpus of 42 retracted, fraudulent, and pseudoscientific papers, paired with a methodology for eliciting and scoring single-shot model engagement with each paper's framing. Each probe pairs a preamble extracted near-verbatim from the target paper with a scientifically plausible study-design request. The probes span five claim types: fabricated observation, pseudophysical mechanism, magical premise, legitimization bridge, and cargo-cult experiment. Two complementary scores measure whether a model rejects the flawed pr
    
[^14]: 探索数据的社会生活：寻找你可信赖的数据

    Exploring the Social Life of Data: Finding Data You Can Trust

    [https://arxiv.org/abs/2608.11395](https://arxiv.org/abs/2608.11395)

    本文提出利用数据使用图作为新基础设施，通过积累的社会和经验证据来评估和确定数据的适用性与可信度，以应对AI时代寻找可信数据的挑战。

    

    人工智能正在改变科学探究的规模和节奏。模型现在能够搜索、整合并推理远超任何单个研究者熟悉的数据仓库的数据。然而，这种扩展带来了一个前置问题：在模型能够产生可信的科学结果之前，它必须找到适合问题的数据、对预期分析足够可靠的数据，并伴有足够的上下文以支持负责任的解释。随着数据变得越来越丰富，寻找数据的挑战已被寻找你可信赖的数据的挑战所取代。本文探讨了当数据在研究中使用时积累的社会和经验证据，如何类似于社会信任网络，被用来确定适用性和可信度。具体而言，本文探讨了数据使用图作为科学数据基础设施的新层次。数据使用图连接了数据与其使用上下文，从而提供了一种评估数据可信度的机制。

    arXiv:2608.11395v1 Announce Type: cross  Abstract: Artificial intelligence is changing the scale and tempo of scientific inquiry. Models can now search, integrate, and reason over data far beyond data repositories familiar to any individual researcher. Yet this expansion creates a prior problem: before a model can produce a trustworthy scientific result, it must locate data that are appropriate for the question, sufficiently reliable for the intended analysis, and accompanied by enough context to support responsible interpretation. As data becomes increasingly abundant, the challenge of finding data has been overcome by the challenge of finding data that you can trust.   This paper explores how the social and empirical evidence that accumulates when data are used in research can be used, analogous to social trust networks, to determine fit for purpose and trust. Specifically, the paper explores data-usage graphs as a new layer of scientific data infrastructure. A data-usage graph conne
    
[^15]: 前沿大语言模型能否匹敌原生多模态嵌入？——关于硬负样本文本到图像检索的比较研究

    Can Frontier LLMs Match Natively Multimodal Embeddings? A Comparison on Hard-Negative Text-to-Image Retrieval

    [https://arxiv.org/abs/2608.11343](https://arxiv.org/abs/2608.11343)

    本研究首次直接比较了原生多模态嵌入（如Gemini Embedding 2）与前沿大语言模型（如GPT-4.1和Claude Sonnet 4.6）在硬负样本文本到图像检索任务中的表现，发现LLM在零样本排序上可与原生多模态嵌入媲美，但后者在预计算场景下更具优势。

    

    arXiv:2608.11343v1 公告类型：新 摘要：跨不同类型媒体（涵盖文本、图像、视频和音频）的多模态检索与分类，传统上依赖于通过对比学习对齐视觉和文本表示的双编码器模型。2026年3月发布的Gemini Embedding 2，作为谷歌首个原生多模态嵌入模型，将文本、图像、视频、音频和文档映射到单一共享空间，加剧了多模态检索系统间的竞争。与此同时，前沿大语言模型（LLMs）也展现出强大的视觉理解能力，引发了一个问题：它们能否作为有效的零样本排序器。我们的研究首次在Flickr30k数据集上对原生多模态嵌入与基于LLM的视觉排序进行了直接比较。我们观察到，GPT-4.1和Claude Sonnet 4.6的表现与Gemini Embedding 2相当。此外，一旦嵌入被预计算，多模态嵌入更适合用于...

    arXiv:2608.11343v1 Announce Type: new  Abstract: Multimodal retrieval and classification across different types of media, spanning text, images,video and audio, has traditionally relied on dual-encoder models that align visual and textual representations through contrastive learning. The March 2026 release of Gemini Embedding 2, Google's first natively multimodal embedding model to map text, images, video, audio, and documents into a single shared space, raises competition among multimodal retrieval systems. Simultaneously, frontier Large language models (LLMs) have also demonstrated strong visual understanding, raising the question of whether they can serve as effective zero-shot rankers. Our study provides the first direct comparison of native multimodal embeddings against LLM-based visual ranking on Flickr30k. We observe that GPT-4.1 and Claude Sonnet 4.6 perform on par with Gemini Embedding 2. Additionally, once embeddings are precomputed, multimodal embeddings are better suited fo
    
[^16]: 推荐系统工厂：将LLM代理自主性限定在工业推荐生命周期中的决策点

    RecSys Factory: Bounding LLM Agent Autonomy to Decision Points in the Industrial Recommender Lifecycle

    [https://arxiv.org/abs/2608.11241](https://arxiv.org/abs/2608.11241)

    本文提出RecSys Factory平台，通过将LLM代理的自主性限定在决策点而非整个管道，并采用无守护进程的事件驱动架构，解决了工业推荐系统中自主性、确定性和效率之间的三元悖论。

    

    arXiv:2608.11241v1 公告类型：新 摘要：将LLM代理部署到工业推荐操作中，暴露出我们称之为自主性-确定性-效率三元悖论的三方张力：通用自主性（解释操作员意图、零样本生成粘合代码）、工业确定性（符合模式的特性提取、无崩溃的A/B测试、零合规路径幻觉）和端到端效率。任何两个都可以最大化以牺牲第三个为代价。我们提出了RecSys Factory，一个在腾讯三个异构推荐业务线上部署了78天的LLM代理平台。设计原则是决策点自主性，而非管道自主性，通过三个解构具体实现，每个解构释放三元悖论的一个顶点。运行时被解构为三个主机发出的事件源（Claude Code停止钩子、企业IM Webhooks、工作流调度器API）：平台在等待阶段不携带长时间运行的守护进程，在94%的时间内消耗零CPU。

    arXiv:2608.11241v1 Announce Type: new  Abstract: Deploying LLM agents into industrial recommender operations exposes a three-way tension we frame as the autonomy-determinism-efficiency trilemma: general autonomy (interpreting operator intent, generating glue code zero-shot), industrial determinism (schema-conforming feature extraction, non-crashing A/B, zero compliance-path hallucination), and end-to-end efficiency. Any two can be maximized against the third. We present RecSys Factory, an LLM-agent platform deployed for 78 days across three heterogeneous Tencent recommender business lines. The design principle is autonomy at decision points, not over pipelines, made concrete through three deconstructions that each discharge one vertex of the trilemma. Runtime is deconstructed into three host-emitted event sources (Claude Code Stop hooks, corporate-IM webhooks, workflow scheduler APIs): the platform carries no long-running daemon during the wait phase and consumes zero CPU during the 94
    
[^17]: Sona技术报告

    Sona Technical Report

    [https://arxiv.org/abs/2608.11015](https://arxiv.org/abs/2608.11015)

    Sona通过共享用户表示和联合训练目标，在单一模型中统一了候选生成与排序，无需手工特征，并在在线测试中取代了包含15个以上候选生成器的生产级联架构，显著提升了参与度指标。

    

    我们介绍了Sona，一个用于Yandex Music的单一模型生成式推荐系统。在在线A/B测试中，Sona取代了整个生产级级联架构，该架构包含超过15个候选生成器，随后是预排序和排序模型，这些模型消费数百个特征，包括来自大型Transformer模型（如Argus）和目标注意力评分器的信号，同时显著提升了关键参与度指标。Sona的架构围绕共享的用户表示统一了候选生成和排序。其编码器将用户按时间顺序记录的参与事件序列转换为隐藏状态，供自回归解码器和排序模块共同使用。下一令牌预测和蒸馏目标联合更新编码器，通过相同的用户状态耦合生成和排序。Sona及其教师排序器均不使用手工设计特征；它们仅基于记录的事件字段和学习到的表示进行操作。

    arXiv:2608.11015v2 Announce Type: replace  Abstract: We introduce Sona, a single-model generative recommender for Yandex Music. In an online A/B test, Sona replaced the entire production cascade, comprising more than 15 candidate generators followed by pre-ranking and ranking models that consume hundreds of features, including signals from large transformer models such as Argus and target-attention scorers, while significantly improving key engagement metrics.   The architecture of Sona unifies candidate generation and ranking around a shared user representation. Its encoder transforms the user's chronological sequence of logged engagement events into hidden states consumed by both the autoregressive decoder and the Ranking Module. The next-token-prediction and distillation objectives jointly update the encoder, coupling generation and ranking through the same user state. Neither Sona nor its Teacher Ranker uses hand-engineered features; both operate on logged event fields and learned 
    
[^18]: ENTLORE：面向企业问答中潜在组织推理的图基基准

    ENTLORE: A Graph-Grounded Benchmark for Latent Organizational Reasoning in Enterprise Question Answering

    [https://arxiv.org/abs/2608.10679](https://arxiv.org/abs/2608.10679)

    ENTLORE通过图基框架重建企业世界，专注于测试从常规文档中恢复隐含组织关系的能力，而非仅组合已有事实。

    

    企业问答被定义为检索内部文档并生成基于依据的答案。然而，常规企业记录是工作副产品，其中所需的组织关系在异构来源中保持隐式。现有基准提供了现实的多源证据，但通常具体化了一个预定义的答案路径，因此测试的是已陈述事实的组合，而非从语料库中缺失的目标关系的恢复。我们将后者称为潜在组织推理。我们引入了ENTLORE，一个图基基准构建框架，从常规文档、权威组织表和操作记录中重建一个经过审计的企业世界。版本化的组织约定在真值图中认证派生关系，从而实现完整的黄金答案和证明证书。对齐的匿名化发布仅暴露文档部分。

    arXiv:2608.10679v2 Announce Type: replace-cross  Abstract: Enterprise question answering is framed as retrieving internal documents and generating grounded answers. Routine enterprise records, however, are work by-products in which required organizational relations remain implicit across heterogeneous sources. Existing benchmarks provide realistic multi-source evidence, but often materialize a predefined answer path and therefore test the composition of stated facts rather than recovery of a target relation absent from the corpus. We call the latter capability latent organizational reasoning.   We introduce ENTLORE, a graph-grounded benchmark construction framework that reconstructs an audited enterprise world from routine documents, authoritative organizational tables, and operational records. Versioned organizational conventions certify derived relations in a truth graph, enabling complete golden answers and proof certificates. The aligned anonymized release exposes only the document
    
[^19]: 基于锚点的逐点LLM重排序器何时有帮助？检索器质量、统计范围与锚点设计

    When Do Anchor-Based Pointwise LLM Rerankers Help? Retriever Quality, Statistical Scope, and Anchor Design

    [https://arxiv.org/abs/2608.10528](https://arxiv.org/abs/2608.10528)

    本研究通过复现和组件级压力测试发现，基于锚点的逐点LLM重排序方法在统计校正下核心思想稳健，但原始论文中的两个设计选择（如锚点设计）在不同条件下可靠性不足。

    

    arXiv:2608.10528v2 公告类型：替换-交叉 摘要：基于锚点的逐点LLM重排序通过将每个候选文档与共享参考段落进行评分，以逐点计算成本恢复跨文档上下文。我们使用GCCP/PAGC作为代表性方法，研究这种方法何时真正有效。我们的研究以复现为先导。我们将复现作为受控组件级压力测试的起点，以评估基于锚点的逐点重排序。我们仅基于论文文本的初步重实现，实现了0.24的nDCG@10，而非报告中的0.66，这揭示了若干未记录的实现细节对于复现该方法至关重要。在识别并恢复这八个细节后，我们复现了报告结果，差异在1.6%以内，并使用验证后的实现进行受控分析。我们发现，核心对比评分思想在严格的统计校正下是稳健的。然而，原始论文中固定不变的两个设计选择不太可靠。首先，我们进一步...

    arXiv:2608.10528v2 Announce Type: replace-cross  Abstract: Anchor-based pointwise LLM reranking scores each candidate against a shared reference passage to recover cross-document context at pointwise cost. We study when this actually helps, using GCCP/PAGC as a representative method. Our study is reproduction-first. We use reproduction as a starting point for a controlled component-level stress test of anchor-based pointwise reranking. Our initial reimplementation, based only on the paper text, achieves 0.24 nDCG@10 instead of the reported 0.66, revealing that several undocumented implementation details are necessary to reproduce the method. After identifying and recovering eight such details, we reproduce the reported results within 1.6% and use the validated implementation for controlled analysis.   We find that the core contrastive scoring idea is robust under rigorous statistical correction. However, two design choices held fixed in the original paper are less reliable. First, we f
    
[^20]: MISO：基于模型内部状态的排序模型优化方法

    MISO: Model-Internal-State-Guided Optimization for Ranking Models

    [https://arxiv.org/abs/2608.07035](https://arxiv.org/abs/2608.07035)

    MISO利用模型内部状态指导排序模型的局部优化决策，以减少试错成本并提高效率。

    

    排序模型在既定的模型家族中不断被精细化改进，然而选择扩展、替换或淘汰哪个组件通常依赖于昂贵的试错过程。我们提出了模型内部状态优化（MISO），这是一种系统工作流程，利用模型内部状态（MIS），包括参数、激活、梯度和归一化统计，来优先进行此类局部优化决策。MISO从训练好的排序模型中提取MIS，将其聚合为排序、对齐和比较信号，并将这些信号转换为少量可解释的候选编辑。由于每次重训练周期后都会重新提取MIS，MISO自然支持自适应优化工作流程，该流程能随着数据分布和系统需求随时间变化而跟踪模型行为的演变。在一个广告排序案例研究中，MISO在显著减少验证运行次数的同时，改善了归一化熵。

    arXiv:2608.07035v2 Announce Type: replace  Abstract: Ranking models are repeatedly refined within established model families, yet the choice of which component to scale, replace, or retire is often guided by expensive trial-and-error. We present Model Internal State Optimization (MISO), a systems workflow that uses model internal states (MIS), including parameters, activations, gradients, and normalization statistics, to prioritize such local optimization decisions. MISO extracts MIS from a trained ranking model, aggregates them into ranking, alignment, and comparison signals, and converts those signals into a small set of interpretable candidate edits. Because MIS are re-extracted after each retraining cycle, MISO naturally supports an adaptive optimization workflow that tracks evolving model behavior as data distributions and system requirements shift over time. In an ads ranking case study, MISO improves normalized entropy while requiring substantially fewer validation runs than exp
    
[^21]: VLM2Rec：解决多模态序列推荐中视觉-语言模型嵌入器的模态坍缩问题

    VLM2Rec: Resolving Modality Collapse in Vision-Language Model Embedders for Multimodal Sequential Recommendation

    [https://arxiv.org/abs/2603.17450](https://arxiv.org/abs/2603.17450)

    本文提出VLM2Rec框架，通过弱模态惩罚机制解决视觉-语言模型嵌入器在多模态序列推荐中的模态坍缩问题，实现模态利用平衡并提升推荐准确性。

    

    多模态环境下的序列推荐（SR）通常依赖于小型冻结的预训练编码器，这限制了语义容量，并阻碍了协同过滤（CF）信号完全整合到项目表示中。受大型语言模型（LLMs）作为高容量嵌入器近期成功的启发，我们研究了使用视觉-语言模型（VLMs）作为CF感知的多模态嵌入器用于序列推荐。然而，我们发现，用于适应VLM生成嵌入并注入CF信号的标准对比监督微调（SFT）会放大固有的模态不平衡：优化过程由一种模态主导，而另一种模态退化，最终损害推荐准确性。为解决此问题，我们提出VLM2Rec，一个基于VLM嵌入器的多模态序列推荐框架，旨在促进模态利用的平衡。具体而言，我们引入了弱模态惩罚机制。

    arXiv:2603.17450v2 Announce Type: replace-cross  Abstract: Sequential Recommendation (SR) in multimodal settings typically relies on small frozen pretrained encoders, which limits semantic capacity and prevents Collaborative Filtering (CF) signals from being fully integrated into item representations. Inspired by the recent success of Large Language Models (LLMs) as high-capacity embedders, we investigate the use of Vision-Language Models (VLMs) as CF-aware multimodal embedders for SR. However, we find that standard contrastive Supervised Fine-Tuning (SFT), used to adapt VLMs for embedding generation and inject CF signals, can amplify inherent modality imbalance: optimization becomes dominated by one modality while the other degrades, ultimately undermining recommendation accuracy. To address this, we propose VLM2Rec, a VLM embedder-based framework for multimodal sequential recommendation designed to promote balanced modality utilization. Specifically, we introduce Weak-modality Penali
    

