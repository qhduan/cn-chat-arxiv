# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Lexplorer: Navigating the Complexity of Legal Document Landscapes](https://arxiv.org/abs/2609.17366) | 本文提出了基于用户意图分类体系的灵活法律文献探索界面 Lexplorer，通过区分单篇、少量和大量文献的文本与数据视图，支持与互联法律文本集合的情境敏感交互，并在欧盟法背景下通过20位法律学者的评估验证了其有效性。 |
| [^2] | [Diagnosing the Fact-Grounding Gap in Multi-Hop Question Answering](https://arxiv.org/abs/2609.17043) | 该研究首次将多跳问答的失败分解为检索失败和提取失败两种模式，揭示出近一半的失败源于“事实接地差距”——即文档已被检索到但无法提取所需事实，且这一缺陷无法通过任何检索改进手段解决。 |
| [^3] | [Efficient Swing Computation for Retrieval in Large-Scale Recommender Systems](https://arxiv.org/abs/2609.16850) | 本文提出了ASC和K-ASC两种新颖高效的算法，用于大规模推荐系统中的近似Swing查询和top-K Swing查询，在提供严格理论保证的同时，解决了现有方法计算代价过高或依赖截断启发式导致质量不佳的问题。 |
| [^4] | [RegRet: Enhancing Region-Level Retrieval in Large Multimodal Models](https://arxiv.org/abs/2609.16847) | RegRet是一个基于大型多模态模型的区域级检索框架，通过引入区域感知编码器和包含局部描述生成与区域对比学习的多阶段训练流程，在不牺牲全局检索性能的前提下显著增强了区域级表示能力。 |
| [^5] | [Can We Do Interpretable NLI with Graphs Based on Atomic Propositions?](https://arxiv.org/abs/2609.16814) | 本文提出一种完全基于图的可解释自然语言推理流水线，将句子分解为原子命题并转换为ConceptNet三元组构建图表示后在SNLI上达到89.7%的准确率，仅比同条件的文本模型低1.9个百分点，证明了可解释的图表示方法在NLI任务中的可行性。 |
| [^6] | [LSREP: A Longitudinal State-Replay Protocol for Evaluating Conversational Memory, with ICE v2 as an Audited Local-First Architecture](https://arxiv.org/abs/2609.16730) | 该论文提出了LSREP纵向评估协议来研究对话记忆的动态演化，并以本地优先记忆中间件ICE v2为案例，证明其在质量与向量RAG相当的同时显著减少检索片段并揭示基线在高密度场景下的灾难性失败。 |
| [^7] | [Quantifying Organizational Environmental Action from Web Data and Large Language Models](https://arxiv.org/abs/2609.16627) | 本文提出了一个可扩展的计算框架，利用网络爬取与大语言模型将组织网页中的非结构化文本转化为结构化的环境行动度量，并通过美国4,964个犹太会众的全国性数据库比较了三种检测方法的效果。 |
| [^8] | [AURA: Agentic Diagnosis and Refinement for Production Recommender Systems at Scale](https://arxiv.org/abs/2609.16625) | 本文提出AURA，一个端到端的AI智能体系统，能够对大规模生产级推荐系统进行定性评估，提供超越传统汇总指标的可操作诊断与改进方案。 |
| [^9] | [Measuring Decision-Scale Use in Tool-Augmented LLMs: A Contrastive Urban Benchmark](https://arxiv.org/abs/2609.16607) | 该论文提出了URBANCONTRASTIVEQA基准，揭示了工具增强型大语言模型在判断城市活动异常程度时倾向于依赖原始数量而非相对本地历史基线的偏差，并证明在工具接口中提供本地基线信息可提升模型的基线相对比较能力。 |
| [^10] | [ReliGRec: Reliability-Oriented LLM-Based Generative Recommendation via User-Risk-Aware Prompt Routing](https://arxiv.org/abs/2609.16560) | 提出ReliGRec弱监督框架，通过从评论反馈信号中提取用户级弱风险代理标签，并结合用户风险感知的提示路由机制，将可靠性适配引入基于LLM的生成式推荐系统。 |
| [^11] | [Predicting Partial Answer Quality and Utility in Agentic Retrieval-Augmented Generation](https://arxiv.org/abs/2609.16453) | 本文提出了一个轨迹内探测框架，通过在每次检索-推理迭代后强制智能体模型生成中间答案，来度量智能体RAG过程中部分答案的质量及其跨迭代的效用变化，从而揭示模型答案状态在生成过程中的演变。 |
| [^12] | [PCap: Personalized Retrieval-Stage Diversity Capping in Facebook Marketplace](https://arxiv.org/abs/2609.16452) | 本文提出 PCap 框架，在 Facebook Marketplace 检索阶段通过基于香农熵的个性化用户分桶和类别上限控制，结合自动化在线参数优化方法，显著提升了推荐多样性和用户互动体验。 |
| [^13] | [Balancing Trial and Reorder: A Hybrid Sequential Transformer-GBDT Ranker for On-Demand Delivery](https://arxiv.org/abs/2609.16407) | 该论文提出了部署在Wolt的统一商家排序系统UVR，通过双向Transformer编码器进行序列用户建模并结合GBDT排序器，利用标签平滑和尝试偏置样本加权来平衡新商家探索与复购排序质量，以单一模型替代四个独立排序模型，使离线尝试MRR提升12%至30%。 |
| [^14] | [Where Post-Training Quantization Breaks Text Embedders: A Measured Map Across Four Embedder Families](https://arxiv.org/abs/2609.16391) | 该论文通过对四个架构家族、五个嵌入模型在不同位宽与分组大小下的系统性量化实验，证明LLM量化中通用的经验法则（保护嵌入表、按模块敏感度分配比特、优先排序感知目标）无法直接迁移到检索嵌入模型上，每一项启发式建议都需要针对嵌入器重新审视。 |
| [^15] | [Evaluating Brand Retrieval and Ranking in Large Language Model Recommendations](https://arxiv.org/abs/2609.16304) | 本文提出了一个独立于模型输出、基于重复采样的LLM品牌推荐评估框架，通过品牌推荐概率（BRP@k）和平均倒数排名（MRR@k）两个指标衡量推荐的普遍性与显著性，并发现LLM会大量遗漏知名品牌，其推荐显著性未必与传统品牌流行度一致。 |
| [^16] | [P3Rec: Distilling Prior--Posterior Preference Reasoning for LLM-based Recommendation](https://arxiv.org/abs/2609.13993) | 提出P3Rec框架，通过联合蒸馏大语言模型中互补的先验偏好（稳定长期兴趣）与后验偏好（目标相关细粒度兴趣）推理知识，克服单一视角蒸馏的局限性，提升轻量级推荐器的性能。 |
| [^17] | [Pre-retrieval Query Clustering for Adaptive Top-k Document Retrieval in RAG Systems](https://arxiv.org/abs/2609.13489) | 该论文提出了一种检索前查询聚类框架，通过离线对查询进行嵌入空间聚类并为每个聚类总结推荐检索深度，在运行时为RAG系统提供查询自适应的top-k检索深度，解决了固定top-k检索导致的过度检索或检索不足问题。 |
| [^18] | [Same Problem, Different Field: Cross-Domain Solution Import via Domain-Stripped Computational Fingerprints](https://arxiv.org/abs/2609.07595) | 该论文提出一种去除领域和方法名称信息的计算指纹方法，能够跨领域识别解决相同底层计算问题的论文，使其他领域的成熟专用求解器得以导入复用，跨领域检索平均精度从0.222大幅提升至0.557。 |
| [^19] | [omni-macos: On-Device Omni-Modal Search on Apple Silicon](https://arxiv.org/abs/2608.05543) | omni-macos在苹果设备上实现全模态本地搜索，通过内存预算管理和高效编码策略，确保所有数据不出设备。 |
| [^20] | [Attention Calibration for Position-Fair Dense Retrieval](https://arxiv.org/abs/2606.02737) | 该论文提出一种带强度系数的注意力校准方法，缓解密集检索中嵌入表示偏向文本前部内容的位置偏差问题，同时将校准内存开销从数 GiB 降至 1 MiB 以下，显著提升相关片段位于文本后部时的检索性能。 |
| [^21] | [SciNLP: A Domain-Specific Benchmark for Full-Text Scientific Entity and Relation Extraction in NLP](https://arxiv.org/abs/2509.07801) | SciNLP是首个提供NLP领域全文实体与关系标注的基准数据集，包含60篇人工标注的全文论文，涵盖6,429个实体和1,649个关系，填补了该领域全文标注数据集的空白。 |
| [^22] | [Revisiting Self-Attentive Sequential Recommendation Beyond the LLM Paradigm](https://arxiv.org/abs/2504.09596) | 本文指出推荐系统（熵增目标）与语言建模（熵减目标）在本质上追求相反的目标，这种数据特性——行为数据局部规律而全局异质，语言局部多样而全局收敛——的根本差异才是推荐系统无法复现语言模型缩放规律的原因，并据此超越LLM范式重新审视自注意力序列推荐。 |

# 详细

[^1]: Lexplorer：探索法律文献版图的复杂性

    Lexplorer: Navigating the Complexity of Legal Document Landscapes

    [https://arxiv.org/abs/2609.17366](https://arxiv.org/abs/2609.17366)

    本文提出了基于用户意图分类体系的灵活法律文献探索界面 Lexplorer，通过区分单篇、少量和大量文献的文本与数据视图，支持与互联法律文本集合的情境敏感交互，并在欧盟法背景下通过20位法律学者的评估验证了其有效性。

    

    随着技术和社会创新带来新的监管挑战，法律系统的复杂性日益增长——这增加了对能够与法律文献集合进行有效交互的界面的需求。通过对法律学者（n=15）的访谈，我们发现支持法律工作需要超越以检索为中心的法律信息系统范式。因此，我们提出了 Lexplorer，这是一个基于捕捉用户意图的分类体系构建的灵活界面，用于探索、导航和分析法律文献。Lexplorer 区分了单篇、少量和大量文献的文本视图和数据视图，支持用户与不断演变的互联法律文本集合进行情境敏感的交互，从而促进法律领域的自适应意义建构。我们在欧盟法的背景下与法律学者（n=20）共同评估了 Lexplorer，验证了我们提炼的需求、意图分类体系和原型设计。

    arXiv:2609.17366v1 Announce Type: cross  Abstract: As technological and social innovations create novel regulatory challenges, legal systems grow in complexity - increasing the need for interfaces that enable effective interactions with legal document collections. Through interviews with legal scholars (n=15), we find that supporting legal work requires going beyond retrieval-centered legal-information-system paradigms. Hence, we propose Lexplorer, a flexible interface for exploring, navigating, and analyzing legal documents, based on a taxonomy capturing user intents. Distinguishing text and data views for one, few, and many documents, Lexplorer enables context-sensitive interactions with evolving collections of interconnected legal texts, facilitating Adaptive Meaning Construction in law. We evaluate Lexplorer with legal scholars (n=20) in the context of European Union law, validating our elicited requirements, intent taxonomy, and prototype design. Resulting from a close collaborati
    
[^2]: 诊断多跳问答中的事实接地差距

    Diagnosing the Fact-Grounding Gap in Multi-Hop Question Answering

    [https://arxiv.org/abs/2609.17043](https://arxiv.org/abs/2609.17043)

    该研究首次将多跳问答的失败分解为检索失败和提取失败两种模式，揭示出近一半的失败源于“事实接地差距”——即文档已被检索到但无法提取所需事实，且这一缺陷无法通过任何检索改进手段解决。

    

    多跳问答需要结合来自多个文档的信息来回答复杂问题。这些系统的能力日益增强，然而当它们失败时，错误通常被归因于没有找到正确的文档。但这种解释在单个推理步骤层面是否成立，在很大程度上尚未被检验。我们在三个标准的多跳问答基准上对此进行了研究，发现失败可分解为两种不同的模式：检索失败，即所需的段落未被检索到；以及提取失败，即段落已被检索到，但无法从中提取出所需的事实——我们将这一现象称为“事实接地差距”。提取失败占每跳缺陷的近一半，且对标准检索指标是不可见的。我们测试的所有检索干预措施都无法解决这类失败，从而为仅依靠检索改进的效果设定了上限。该差距的严重程度在不同（摘要在此处截断）

    arXiv:2609.17043v1 Announce Type: cross  Abstract: Multi-hop question answering requires combining information from multiple documents to answer complex questions. These systems have grown increasingly capable, yet when they fail, the error is typically attributed to not finding the right documents. Whether this holds at the level of individual reasoning steps remains largely unexamined. We investigate this across three standard multi-hop QA benchmarks and find that failures decompose into two distinct modes: retrieval failures, where the needed passage was not retrieved, and extraction failures, where the passage was retrieved but the needed fact could not be extracted - a phenomenon we term the fact-grounding gap. Extraction failures account for nearly half of all per-hop deficiencies and are invisible to standard retrieval metrics. They remain unresolved by every retrieval intervention we test, establishing a ceiling for retrieval-only improvements. The gap's severity varies across 
    
[^3]: 大规模推荐系统中面向检索的高效Swing计算

    Efficient Swing Computation for Retrieval in Large-Scale Recommender Systems

    [https://arxiv.org/abs/2609.16850](https://arxiv.org/abs/2609.16850)

    本文提出了ASC和K-ASC两种新颖高效的算法，用于大规模推荐系统中的近似Swing查询和top-K Swing查询，在提供严格理论保证的同时，解决了现有方法计算代价过高或依赖截断启发式导致质量不佳的问题。

    

    给定一个用户-物品图 $G$、一个查询物品 $v_q$ 和一个目标物品 $v_t$，物品对 $(v_q, v_t)$ 的 Swing 分数 $sw(v_q, v_t)$ 利用用户-物品-用户交互结构来评估二者的相似度。该度量在物品到物品（i2i）检索任务中被证明非常有效，并在工业级推荐系统中得到广泛应用。然而，现有的 Swing 分数计算方案要么由于物品度数的二次时间复杂度而代价过高，要么依赖截断启发式方法导致质量不佳，使其在拥有数十亿交互的图上难以实用。在本文中，我们提出了 ASC 和 $K$-ASC，这是两种新颖且高效的算法，用于解决近似 Swing 查询和 top-$K$ Swing 查询的上述问题。具体而言，这些算法在概率相对误差和可加误差方面提供了严格的理论保证。

    arXiv:2609.16850v1 Announce Type: new  Abstract: Given a user-item graph $G$, a query item $v_q$ and a target item $v_t$, the Swing score $sw(v_q, v_t)$ of the item pair $(v_q, v_t)$ leverages the user-item-user interaction structure to evaluate their similarity. This measure is found to be highly effective in item-to-item (i2i) retrieval task and finds extensive applications in industrial-scale recommender systems. However, existing solutions towards computing Swing scores are either prohibitively expensive due to their quadratic time complexity w.r.t. the item degree, or rely on truncation heuristics that yield unsatisfactory quality, rendering them impractical particularly on graphs with billions of interactions.   In this paper, we present ASC and $K$-ASC, two novel and efficient algorithms for approximate and top-$K$ Swing queries, to address the aforementioned limitations. Specifically, these algorithms provide rigorous theoretical guarantees in probabilistic relative and additiv
    
[^4]: RegRet：增强大型多模态模型中的区域级检索

    RegRet: Enhancing Region-Level Retrieval in Large Multimodal Models

    [https://arxiv.org/abs/2609.16847](https://arxiv.org/abs/2609.16847)

    RegRet是一个基于大型多模态模型的区域级检索框架，通过引入区域感知编码器和包含局部描述生成与区域对比学习的多阶段训练流程，在不牺牲全局检索性能的前提下显著增强了区域级表示能力。

    

    区域级检索旨在将用户指定的图像区域与相关区域或文本描述进行对齐，在电商产品搜索和RAG（检索增强生成）等实际应用中发挥着至关重要的作用。尽管近年来大型多模态模型（LMMs）在多模态检索方面取得了显著进展，但它们主要专注于全局级任务，难以捕获有效的区域级表示。为了弥补这一差距，我们提出了RegRet，一个基于LMM的区域级检索框架，它能够在不损害整体全局检索性能的前提下增强区域表示。其核心在于，RegRet集成了一个区域感知编码器来捕获详细的区域特征，同时将其与全局背景上下文进行平衡。为了进一步增强表示的细粒度理解能力和判别性，我们设计了一个多阶段训练流程，其中包括详细的局部描述生成和区域对比学习。

    arXiv:2609.16847v1 Announce Type: cross  Abstract: Region-level retrieval aims to align user-specified image regions with relevant regions or textual descriptions, playing a crucial role in realworld applications such as e-commerce product search and RAG. Although recent Large Multimodal Models (LMMs) have made significant strides in multimodal retrieval, they primarily focus on global-level tasks and struggle to capture effective region-level representations. To bridge this gap, we present RegRet, an LMM-based Region-level Retrieval framework that enhances the regional representations without compromising overall global retrieval performance. At its core, RegRet integrates a Region-Aware Encoder to capture detailed regional features while balancing them with the global background context. To further enhance the fine-grained understanding and discriminability of representations, we design a multi-stage training pipeline that includes detailed localized captioning and regional contrasti
    
[^5]: 我们能否基于原子命题构建的图来实现可解释的自然语言推理？

    Can We Do Interpretable NLI with Graphs Based on Atomic Propositions?

    [https://arxiv.org/abs/2609.16814](https://arxiv.org/abs/2609.16814)

    本文提出一种完全基于图的可解释自然语言推理流水线，将句子分解为原子命题并转换为ConceptNet三元组构建图表示后在SNLI上达到89.7%的准确率，仅比同条件的文本模型低1.9个百分点，证明了可解释的图表示方法在NLI任务中的可行性。

    

    尽管基于大语言模型（LLM）的自然语言推理（NLI）系统达到了很高的准确率，但其决策过程缺乏可审计的结构。本文探讨了是否可以仅使用可解释的、基于图的证据表示来执行自然语言推理。我们引入了一个完全基于图的流水线，其中分类器从不直接处理输入文本。取而代之的是，句子被分解为原子命题，通过受限解码转换为ConceptNet三元组，并为每个文本对表示为三张图：前提图、假设图以及检索到的ConceptNet子图。随后将这些图输入到一个经过微调的8亿参数语言模型中。在SNLI数据集上，我们的流水线达到了89.7%的准确率，仅比以相同方式训练的基于文本的模型低1.9个百分点。在ANLI上，它在R2和R3轮次上与已发表的RoBERTa-large性能相当（50%准确率），但在R1上落后16个百分点。

    arXiv:2609.16814v1 Announce Type: new  Abstract: While Large Language Model (LLM)-based Natural Language Inference (NLI) systems achieve high accuracy, their decision-making processes lack auditable structures. This paper explores whether NLI can be performed using only interpretable, graph-based representations of evidence. We introduce a fully graph-based pipeline where the classifier never directly processes the input text. Instead, sentences are decomposed into atomic propositions, converted into ConceptNet triples via constrained decoding, and represented as three graphs per pair: premise, hypothesis, and a retrieved ConceptNet subgraph. These graphs are then fed into a fine-tuned 0.8-billion-parameter language model. On the SNLI dataset, our pipeline achieves 89.7% accuracy, just 1.9 points below an identically trained text-based model. On ANLI, it matches the published performance of RoBERTa-large on rounds R2 and R3 (50% accuracy) but trails by 16 points on R1, resulting in an 
    
[^6]: LSREP：一种用于评估对话记忆的纵向状态回放协议，以及作为经审计的本地优先架构的ICE v2

    LSREP: A Longitudinal State-Replay Protocol for Evaluating Conversational Memory, with ICE v2 as an Audited Local-First Architecture

    [https://arxiv.org/abs/2609.16730](https://arxiv.org/abs/2609.16730)

    该论文提出了LSREP纵向评估协议来研究对话记忆的动态演化，并以本地优先记忆中间件ICE v2为案例，证明其在质量与向量RAG相当的同时显著减少检索片段并揭示基线在高密度场景下的灾难性失败。

    

    对话记忆在使用过程中会发生变化，因此仅靠端点问答无法确定持久状态如何积累、老化或纳入修订。我们提出了LSREP，这是一种纵向状态回放评估协议，结合了有序回放、显式生命周期调度、重复探测、演化的参考答案以及机制保真度检查。其架构案例研究是ICE v2，这是一种具有类型化存储、检索融合和动态上下文预算的本地优先记忆中间件。该私有的单用户实例包含1,985轮对话、219个不同探测和跨52个检查点的1,211个探测检查点观察。在三个普通密度数据集上，ICE v2与向量RAG的平均质量差异接近于零，同时选择的片段减少了32%，但估计提示令牌多使用了6.6%。第四个高密度数据集暴露了无预算基线的灾难性失败。保真度审计限制了归因：

    arXiv:2609.16730v1 Announce Type: new  Abstract: Conversational memory changes during use, so endpoint question answering alone cannot establish how a persistent state accumulates, ages, or incorporates revisions. We introduce LSREP, a Longitudinal State-Replay Evaluation Protocol combining ordered replay, explicit lifecycle schedules, repeated probes, evolving reference answers, and mechanism-fidelity checks. Its architectural case study is ICE v2, a local-first memory middleware with typed stores, retrieval fusion, and dynamic context budgets. The private, single-user instantiation contains 1,985 turns, 219 distinct probes, and 1,211 probe-checkpoint observations across 52 checkpoints. On three ordinary-density datasets, ICE v2 has a near-zero mean quality difference from vector-RAG while selecting 32% fewer fragments but using 6.6% more estimated prompt tokens. A fourth, dense dataset exposes catastrophic failures of the unbudgeted baseline. The fidelity audit limits attribution: pr
    
[^7]: 基于网络数据与大语言模型的组织环境行动量化研究

    Quantifying Organizational Environmental Action from Web Data and Large Language Models

    [https://arxiv.org/abs/2609.16627](https://arxiv.org/abs/2609.16627)

    本文提出了一个可扩展的计算框架，利用网络爬取与大语言模型将组织网页中的非结构化文本转化为结构化的环境行动度量，并通过美国4,964个犹太会众的全国性数据库比较了三种检测方法的效果。

    

    从公开可用的网络内容中量化组织的环境行动仍然是一个具有挑战性的环境数据科学问题，因为相关信息可能分散在多个网页中，且主要以非结构化文本形式呈现。我们提出了一个可扩展的计算框架，用于将组织的网络内容转化为环境行动的结构化度量，并以美国的犹太会众为例展示了该方法。我们通过整合多个地理空间、知识库、目录以及人工审核的数据来源，构建了一个包含4,964个会众的全国性数据库。其中，2,657个会众拥有可成功爬取的活跃网站，生成了包含154,454个网页的语料库。我们比较了三种检测环境行动的方法：关键词检索后进行大语言模型（LLM）分类、语义向量检索后进行LLM分类，以及直接……

    arXiv:2609.16627v1 Announce Type: new  Abstract: Quantifying organizational environmental action from publicly available web content remains a challenging environmental data science problem because relevant information can be dispersed across multiple webpages and is primarily communicated through unstructured text. We present a scalable computational framework for transforming organizational web content into structured measures of environmental action and demonstrate the approach using Jewish congregations in the United States. We constructed a national database of 4,964 congregations by integrating multiple geospatial, knowledge-base, directory, and manually reviewed sources. Of these, 2,657 had active websites that were successfully crawled, producing a corpus of 154,454 webpages. We compared three approaches for detecting environmental actions: keyword retrieval followed by large language model (LLM) classification, semantic vector retrieval followed by LLM classification, and dire
    
[^8]: AURA：面向大规模生产级推荐系统的智能体诊断与优化

    AURA: Agentic Diagnosis and Refinement for Production Recommender Systems at Scale

    [https://arxiv.org/abs/2609.16625](https://arxiv.org/abs/2609.16625)

    本文提出AURA，一个端到端的AI智能体系统，能够对大规模生产级推荐系统进行定性评估，提供超越传统汇总指标的可操作诊断与改进方案。

    

    推荐系统为何以及如何辜负其所服务的用户？通常情况下，从业者只能依靠利益相关团队的反馈、领域专业知识以及数据分析洞察相结合的方式来改进他们的算法。然而，推荐结果对终端用户在何处以及如何表现良好或不佳的细微差别，很难从汇总的定量指标中辨别出来。这些指标只能提供一个高层次且不完整的画面，要进一步细化推荐质量及其模式，则需要结合领域理解和客观性进行大规模的推理。我们针对这一复杂的难题，描述了一种利用最新AI智能体技术进展、为生产级推荐系统提供可操作诊断与改进的方法及实现。我们提出了AURA（推荐算法的智能体理解与优化），这是一个端到端的智能体系统，能够对生产推荐系统执行定性评估……

    arXiv:2609.16625v1 Announce Type: cross  Abstract: How and why does a recommender system fail the users it serves? Oftentimes, practitioners are left to improve their algorithms based on a combination of feedback from stakeholder teams, domain expertise, and insights from data analyses. Yet the nuances of how and where recommendations perform well or poorly for end users are difficult to discern from aggregate quantitative metrics. Whereas these metrics provide a high-level and incomplete picture, further granularity into the quality of recommendations and their patterns requires reasoning with domain understanding and objectivity, at scale. We contemplate this complex conundrum and describe a method and implementation that uses the latest AI agentic advances to provide actionable diagnoses and improvements for production recommender systems. We present AURA (Agentic Understanding and Refinement of recommender Algorithms), an end-to-end agentic system that performs qualitative evaluati
    
[^9]: 衡量工具增强型大语言模型中决策尺度的使用：一个对比性城市基准

    Measuring Decision-Scale Use in Tool-Augmented LLMs: A Contrastive Urban Benchmark

    [https://arxiv.org/abs/2609.16607](https://arxiv.org/abs/2609.16607)

    该论文提出了URBANCONTRASTIVEQA基准，揭示了工具增强型大语言模型在判断城市活动异常程度时倾向于依赖原始数量而非相对本地历史基线的偏差，并证明在工具接口中提供本地基线信息可提升模型的基线相对比较能力。

    

    城市决策支持通常关注的是某个特定地点的活动是否异常偏高或偏低，而不是哪个地点的原始数量更大。安静社区里的二十次接单可能比机场的一百八十次接单更异常。我们提出了URBANCONTRASTIVEQA，这是一个测试工具增强型语言模型能否进行这种基线相对比较的基准。每个条目将来自纽约、芝加哥和西雅图公共出行数据的两个城市情境进行配对，并以当前活动偏离该地点历史基线的程度进行标注。我们在五种工具输出格式下评估了六个指令微调模型。在仅有原始计数的情况下，模型往往选择更大的数字，即使该数字对于其所在区域来说异常程度更低。服务器计算的基线分数和序数标签提高了准确率，但提升幅度因模型而异。对于异构的城市数据流，工具接口需要暴露本地基线信息，而不仅仅是活动量。我们发布了配对题库、标签和评分数据。

    arXiv:2609.16607v1 Announce Type: new  Abstract: Urban decision-support often asks whether activity is unusually high or low for a specific place, not which place has the larger raw count. Twenty pickups in a quiet neighborhood can be more abnormal than 180 at an airport. We introduce URBANCONTRASTIVEQA, a benchmark that asks whether tool-augmented language models can make this baseline-relative comparison. Each item pairs two urban situations from public mobility data in NYC, Chicago, and Seattle, labeled by how far current activity deviates from that place's historical baseline. We evaluate six instruction-tuned models under five tool-output formats. With only raw counts, models often pick the larger number even when it is less abnormal for its zone. Server-computed baseline scores and ordinal labels raise accuracy, but gains vary by model. For heterogeneous urban feeds, tool interfaces need to expose local baselines, not just activity volumes. We release the pair bank, labels, scori
    
[^10]: ReliGRec：通过用户风险感知提示路由实现的可靠性导向的基于大语言模型（LLM）的生成式推荐

    ReliGRec: Reliability-Oriented LLM-Based Generative Recommendation via User-Risk-Aware Prompt Routing

    [https://arxiv.org/abs/2609.16560](https://arxiv.org/abs/2609.16560)

    提出ReliGRec弱监督框架，通过从评论反馈信号中提取用户级弱风险代理标签，并结合用户风险感知的提示路由机制，将可靠性适配引入基于LLM的生成式推荐系统。

    

    现实世界推荐系统中的用户行为是异构的。虽然一些用户表现出连贯一致的偏好，但其他用户则表现出突然的兴趣转变、爆发式的交互、过度重复，或与协同邻域（collaborative neighborhoods）不一致的行为。这种偏差可能源于良性的变化或操纵行为（包括水军攻击/shilling攻击），但仅凭这些并不能证明存在恶意意图。现有的鲁棒推荐系统通过训练时的重新加权或图聚合来利用用户风险信号，而在基于大语言模型的生成式推荐中，使生成过程适应估计出的用户级弱风险仍是一个未被充分探索的方向。我们提出了ReliGRec（可靠性导向的生成式推荐），这是一个弱监督框架，其名称表达的是其设计目标而非一个监督式的可靠性变量。ReliGRec从评论反馈信号中为一部分用户推导出用户级弱风险代理标签，并表示序列行为与协同（注：摘要原文在此处被截断）

    arXiv:2609.16560v1 Announce Type: new  Abstract: User behavior in real-world recommender systems is heterogeneous. While some users exhibit coherent preferences, others show abrupt interest shifts, bursty interactions, excessive repetition, or inconsistency with collaborative neighborhoods. Such deviations may arise from benign variation or manipulation, including shilling attacks, but do not alone establish malicious intent. Existing robust recommenders exploit user-risk signals through training-time reweighting or graph aggregation, whereas adapting generation to estimated user-level weak risk remains underexplored in LLM-based generative recommendation. We propose ReliGRec (Reliability-oriented Generative Recommendation), a weakly supervised framework whose name denotes its design goal rather than a supervised reliability variable. ReliGRec derives user-level weak-risk proxy labels from review-feedback signals for a subset of users and represents sequential behavior and collaborativ
    
[^11]: 智能体检索增强生成中部分答案质量与效用的预测

    Predicting Partial Answer Quality and Utility in Agentic Retrieval-Augmented Generation

    [https://arxiv.org/abs/2609.16453](https://arxiv.org/abs/2609.16453)

    本文提出了一个轨迹内探测框架，通过在每次检索-推理迭代后强制智能体模型生成中间答案，来度量智能体RAG过程中部分答案的质量及其跨迭代的效用变化，从而揭示模型答案状态在生成过程中的演变。

    

    智能体检索增强生成已成为多跳问答领域一种有前景的范式，其中推理模型迭代地向检索器发出查询，并将新检索到的上下文融入后续的推理步骤。虽然这种迭代过程可以提升最终答案的质量，但当前对智能体RAG的评估主要聚焦于端到端的结果，对模型在生成过程中答案状态如何变化的可见性有限。在这项工作中，我们引入了一个轨迹内探测框架来研究智能体RAG中的中间答案状态。具体而言，在每次检索-推理迭代之后，我们强制智能体模型停止推理，并基于其当前状态生成一个中间答案。这使我们能够定义两个迭代级别的度量：每次迭代时的部分答案质量，以及部分效用，即各迭代之间部分答案质量的变化。我们对……的分析（摘要在此处被截断）

    arXiv:2609.16453v1 Announce Type: new  Abstract: Agentic Retrieval-Augmented Generation (RAG) has become a promising paradigm for multi-hop question answering, where a reasoning model iteratively issues queries to a retriever and incorporates newly retrieved context into subsequent reasoning steps. While this iterative process can improve final answer quality, current evaluations of agentic RAG largely focus on end-to-end outcomes and provide limited visibility into how a model's answer state changes during generation. In this work, we introduce an in-trajectory probing framework to study intermediate answer states in agentic RAG. Specifically, after each retrieval-reasoning iteration, we force an agentic model to stop reasoning and generate an intermediate answer based on its current state. This allows us to define two iteration-level measures: partial answer quality at each iteration, and partial utility as the change in partial answer quality across iterations. Our analysis across m
    
[^12]: PCap：Facebook Marketplace 中检索阶段的个性化多样性上限控制

    PCap: Personalized Retrieval-Stage Diversity Capping in Facebook Marketplace

    [https://arxiv.org/abs/2609.16452](https://arxiv.org/abs/2609.16452)

    本文提出 PCap 框架，在 Facebook Marketplace 检索阶段通过基于香农熵的个性化用户分桶和类别上限控制，结合自动化在线参数优化方法，显著提升了推荐多样性和用户互动体验。

    

    我们提出了一个个性化上限控制框架（PCap），通过在检索阶段引入用户级别的多样性约束来改善 Facebook Marketplace 的多样性。PCap 使用基于香农熵的评分来建模个体多样性偏好，将用户划分为不同的多样性分桶，并在多源候选检索过程中应用个性化的类别上限。为了探索各分桶上限所构成的高维参数空间，我们采用了一种称为参数调优序列的自动化在线优化方法。大规模在线实验表明，PCap 显著提升了用户浏览体验，这体现在互动指标上。这项工作为将个性化多样性整合到工业级检索系统中提供了实用的见解。

    arXiv:2609.16452v1 Announce Type: new  Abstract: We propose a personalized capping framework (PCap) to improve the diversity in Facebook Marketplace by introducing user-level diversity constraints at the retrieval stage. PCap models individual diversity preferences using Shannon entropy-based scoring, segments users into diversity buckets, and applies personalized category caps during multi-source candidate retrieval. To navigate the high-dimensional parameter space of per-bucket caps, we leverage an automated online optimization method called Parameter Tuning Sequence. Large-scale online experiments demonstrate that PCap significantly improves users' browsing experience shown in engagement metrics. This work provides practical insights into integrating personalized diversity into industrial retrieval systems.
    
[^13]: 平衡尝试与复购：面向即时配送的混合序列Transformer-GBDT排序器

    Balancing Trial and Reorder: A Hybrid Sequential Transformer-GBDT Ranker for On-Demand Delivery

    [https://arxiv.org/abs/2609.16407](https://arxiv.org/abs/2609.16407)

    该论文提出了部署在Wolt的统一商家排序系统UVR，通过双向Transformer编码器进行序列用户建模并结合GBDT排序器，利用标签平滑和尝试偏置样本加权来平衡新商家探索与复购排序质量，以单一模型替代四个独立排序模型，使离线尝试MRR提升12%至30%。

    

    在配送平台上，个性化的商家排序极大地影响用户发现和下单的内容。与纯数字领域不同，候选商家具有本地属性，并受实时可用性和配送运营的约束。其中一个核心的建模矛盾在于：既要推广新商家以供用户尝试，又要在具有复购意图的会话中保持排序质量。我们提出了通用商家排序器，这是部署在Wolt的生产系统，它将用于序列用户建模的双向Transformer编码器与整合了上下文、用户和商家特征的GBDT排序器相结合。UVR在一个国家的所有商家和业务领域上进行训练，同时在推理时强制执行本地配送约束，它用一个统一的系统替代了之前四个独立的排序模型（三个用于餐厅，一个用于零售）。标签平滑和偏向尝试的样本加权引导模型关注新商家，使离线尝试MRR相比生产系统提升了12%至30%。

    arXiv:2609.16407v1 Announce Type: cross  Abstract: On a delivery platform, personalized store ranking greatly influences what users find and order. Unlike digital-only domains, candidate stores are local and bound by real-time availability and delivery operations. One central modeling tension is between surfacing new stores for trial and preserving ranking quality for sessions with reorder intent. We present Universal Venue Ranker (UVR), a production system deployed at Wolt that pairs a bidirectional transformer encoder for sequential user modeling with a GBDT ranker integrating contextual, user, and store features. Trained across all stores and domains of a country while enforcing local delivery constraints at inference, UVR replaces four previously separate ranking models (three for restaurants, one for retail) with a single unified system. Label smoothing and trial-biased sample weighting steer the model toward new stores, lifting offline trial MRR by +12% to +30% over production wh
    
[^14]: 后训练量化在何处破坏文本嵌入模型：跨四个嵌入器家族的实测图谱

    Where Post-Training Quantization Breaks Text Embedders: A Measured Map Across Four Embedder Families

    [https://arxiv.org/abs/2609.16391](https://arxiv.org/abs/2609.16391)

    该论文通过对四个架构家族、五个嵌入模型在不同位宽与分组大小下的系统性量化实验，证明LLM量化中通用的经验法则（保护嵌入表、按模块敏感度分配比特、优先排序感知目标）无法直接迁移到检索嵌入模型上，每一项启发式建议都需要针对嵌入器重新审视。

    

    仅权重的后训练量化是缩小检索嵌入模型最廉价的方法，而流传的应用建议——保护嵌入表、按模块敏感度分配比特、优先选择排序感知目标而非权重重建——几乎原封不动地被沿用到LLM量化中。我们直接在检索嵌入模型上检验这些建议，对来自四个架构家族的五个检查点，在位宽和分组大小的网格上进行了量化，并在每个位宽下分别隔离测试嵌入块、注意力块和前馈块。结果显示，每一条启发式规则都无法按原样迁移。嵌入表在任何家族中都从未成为主导性的独立保护优先项，尽管它在其中几个家族中是最大的张量。模块敏感度也不能作为一种可迁移的排序而存在：在INT4/g16下，模块之间的差异太小而无法据此进行比特分配；在INT3下，排序则变得依赖于具体家族，联合损伤停止……

    arXiv:2609.16391v1 Announce Type: cross  Abstract: Weight-only post-training quantization is the cheapest way to shrink a retrieval embedder, and the received advice for applying it -- protect the embedding table, allocate bits by module sensitivity, prefer a ranking-aware objective over weight reconstruction -- was carried into LLM quantization largely intact. We test that advice on retrieval embedders directly, quantizing five checkpoints from four architecture families across a grid of bit widths and group sizes, and isolating the embedding, attention and feed-forward blocks at each width.   Every heuristic fails to transfer as stated. The embedding table never emerges as the dominant isolated protection priority in any family, despite being the largest tensor in several of them. Module sensitivity does not survive as a transferable ordering: at INT4/g16 the spread between modules is too small to allocate against, at INT3 the ordering becomes family-dependent and joint damage stops 
    
[^15]: 评估大型语言模型推荐中的品牌检索与排名

    Evaluating Brand Retrieval and Ranking in Large Language Model Recommendations

    [https://arxiv.org/abs/2609.16304](https://arxiv.org/abs/2609.16304)

    本文提出了一个独立于模型输出、基于重复采样的LLM品牌推荐评估框架，通过品牌推荐概率（BRP@k）和平均倒数排名（MRR@k）两个指标衡量推荐的普遍性与显著性，并发现LLM会大量遗漏知名品牌，其推荐显著性未必与传统品牌流行度一致。

    

    大型语言模型（LLM）正被越来越多地用于产品推荐，但评估其推荐结果所面临的挑战与传统信息检索和推荐系统有所不同。LLM可以在没有明确候选集的情况下生成推荐，且对同一查询的重复响应可能产生不同的品牌和排名。我们引入了一个评估开放式LLM品牌推荐的框架，该框架独立于模型输出定义竞争集合，并通过重复采样来估计推荐的普遍性和显著性。我们使用品牌推荐概率（BRP@$k$）和平均倒数排名（MRR@$k$）来操作化这些概念，并将该框架应用于跨五个产品类别的六个大型语言模型。仅基于类别的查询显示，知名品牌被大量遗漏，且推荐显著性遵循传统品牌知名度的证据有限。

    arXiv:2609.16304v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used for product recommendation, but evaluating their recommendations presents challenges that differ from conventional information retrieval and recommender systems. LLMs can generate recommendations without an explicit candidate set, and repeated responses to the same query can produce different brands and rankings. We introduce a framework for evaluating open-ended LLM brand recommendations that defines the competitive set independently of model outputs and estimates recommendation prevalence and prominence through repeated sampling. We operationalize these constructs using Brand Recommendation Probability (BRP@$k$) and Mean Reciprocal Rank (MRR@$k$), and apply the framework to six LLMs across five product categories. Category-only queries reveal substantial omission of established brands and limited evidence that recommendation prominence follows conventional brand popularity. Instead, pr
    
[^16]: P3Rec：为基于大语言模型的推荐蒸馏先验-后验偏好推理

    P3Rec: Distilling Prior--Posterior Preference Reasoning for LLM-based Recommendation

    [https://arxiv.org/abs/2609.13993](https://arxiv.org/abs/2609.13993)

    提出P3Rec框架，通过联合蒸馏大语言模型中互补的先验偏好（稳定长期兴趣）与后验偏好（目标相关细粒度兴趣）推理知识，克服单一视角蒸馏的局限性，提升轻量级推荐器的性能。

    

    大语言模型（LLM）展现出强大的语义理解和偏好推理能力，为推荐系统中的用户建模提供了新机遇。现有的LLM-as-Enhancer方法通常将大语言模型提炼的偏好知识蒸馏到轻量级推荐器中，以避免昂贵的在线LLM推理。然而，这些方法往往仅从单一视角构建蒸馏知识。先验偏好能够捕捉用户稳定且一致的长期兴趣，但对当前决策提供的指导有限；而后验偏好揭示了与目标相关的细粒度兴趣，却可能过度依赖目标线索。为了解决这些局限性，我们提出了P3Rec框架，该框架联合提取并内化互补的先验偏好和后验偏好推理知识。具体而言，P3Rec首先从用户侧推导出与目标无关的先验偏好以及以目标为条件的后验偏好……

    arXiv:2609.13993v1 Announce Type: new  Abstract: Large language models (LLMs) exhibit strong semantic understanding and preference reasoning capabilities, offering new opportunities for user modeling in recommender systems. Existing LLM-as-Enhancer methods typically distill LLM-derived preference knowledge into lightweight recommenders to avoid costly online LLM inference. However, they often construct distillation knowledge from only one perspective. Prior preference captures users' stable and consistent interests but provides limited guidance for the current decision, whereas posterior preference reveals target-relevant fine-grained interests but may rely excessively on target clues. To address these limitations, we propose P$^3$Rec, a framework that jointly extracts and internalizes complementary prior and posterior preference reasoning knowledge. Specifically, P$^3$Rec first derives target-agnostic prior preferences and target-conditioned posterior preferences from the user side, w
    
[^17]: 面向RAG系统自适应Top-k文档检索的检索前查询聚类方法

    Pre-retrieval Query Clustering for Adaptive Top-k Document Retrieval in RAG Systems

    [https://arxiv.org/abs/2609.13489](https://arxiv.org/abs/2609.13489)

    该论文提出了一种检索前查询聚类框架，通过离线对查询进行嵌入空间聚类并为每个聚类总结推荐检索深度，在运行时为RAG系统提供查询自适应的top-k检索深度，解决了固定top-k检索导致的过度检索或检索不足问题。

    

    RAG系统通常检索固定数量的文档（top-k）来支撑生成，但这种静态方法十分脆弱：简单查询会导致过度检索（增加噪声和成本），而复杂查询则检索不足，导致召回失败并级联为错误答案。出于“需要检索多少文档才能可靠地回答任意查询”这一问题的动机，我们提出了一个实用的、通用的查询自适应检索深度框架。在离线阶段，我们通过在默认检索器下测量NDCG来估计每个查询的检索难度，并从NDCG-k曲线中推导出查询特定的“饱和”点k*。由于在线计算这些信号的成本高昂，我们在嵌入空间中对大量查询进行聚类，并采用均值加方差规则为每个聚类总结出一个以高覆盖率（例如约95%）为目标的推荐检索深度。在运行时，系统为传入的查询分配（相应的聚类推荐检索深度，从而实现自适应检索）。

    arXiv:2609.13489v1 Announce Type: new  Abstract: RAG systems commonly retrieve a fixed number of documents (top-k) to ground generation, but this static approach is brittle: simple queries suffer over-retrieval (adding noise and cost) while complex queries are under-retrieved, causing recall failures that cascade into incorrect answers. Motivated by the question of how many documents must be retrieved to answer an arbitrary query reliably, we propose a practical, general framework for query-adaptive retrieval depth. Offline, we estimate per-query retrieval difficulty by measuring NDCG under the default retriever and deriving a query-specific "saturation" point k* from the NDCG-k curve. Because computing these signals online is expensive, we cluster a large set of queries in embedding space and summarize each cluster with a recommended retrieval depth that targets high coverage (e.g., \textasciitilde{}95\%) using a mean-plus-variance rule. At runtime, the system assigns an incoming quer
    
[^18]: 同一问题，不同领域：通过去除领域信息的计算指纹实现跨领域解决方案导入

    Same Problem, Different Field: Cross-Domain Solution Import via Domain-Stripped Computational Fingerprints

    [https://arxiv.org/abs/2609.07595](https://arxiv.org/abs/2609.07595)

    该论文提出一种去除领域和方法名称信息的计算指纹方法，能够跨领域识别解决相同底层计算问题的论文，使其他领域的成熟专用求解器得以导入复用，跨领域检索平均精度从0.222大幅提升至0.557。

    

    arXiv:2609.07595v2 公告类型：replace-cross 摘要：同一个底层计算问题在互不相关的领域中以不同的名称被解决：递归贝叶斯状态估计在控制领域被称为“卡尔曼滤波器”，在药代动力学中被称为“贝叶斯预测”，在地球科学中被称为“数据同化”。基于主题和引用的科学嵌入无法识别这种共享的问题。我们将每篇论文一次性提炼为去除领域和方法名称的多面计算指纹——即一个自由文本机制骨架加上受控的计算侧面。我们在此基础上定义了一种可调节、可选择侧面的相似度度量。其目标是解决方案导入：发现解决相同问题的跨领域论文对，从而可以将某个领域的定制实现替换为另一领域的标准专用求解器。在一个涵盖109篇论文、18个方法族的基准测试中，机制骨架将跨领域检索的平均精度从基于摘要的0.222提升至0.513，而完整的指纹则达到0.557。

    arXiv:2609.07595v2 Announce Type: replace-cross  Abstract: The same underlying computational problem is solved across unrelated fields under different names: recursive Bayesian state estimation appears as a "Kalman filter" in control, "Bayesian forecasting" in pharmacokinetics, and "data assimilation" in geoscience. Topical and citation-based scientific embeddings cannot see this shared problem. We distill each paper once into a domain- and method-name-stripped faceted computational fingerprint, a free-text mechanism skeleton plus controlled computational facets. We define a tunable, facet-selectable similarity over it. The goal is solution import: surface cross-field pairs solving the same problem, so a bespoke implementation can be swapped for another field's standard, specialized solver. On a benchmark of 18 method families across 109 papers, the skeleton lifts cross-domain retrieval average precision over the abstract from 0.222 to 0.513, and the whole fingerprint reaches 0.557. St
    
[^19]: omni-macos：苹果芯片上的设备端全模态搜索

    omni-macos: On-Device Omni-Modal Search on Apple Silicon

    [https://arxiv.org/abs/2608.05543](https://arxiv.org/abs/2608.05543)

    omni-macos在苹果设备上实现全模态本地搜索，通过内存预算管理和高效编码策略，确保所有数据不出设备。

    

    一个将文本、代码、文档、图像、音频和视频嵌入同一表示空间的搜索引擎，必须运行其编码器并将索引存储在某处，而几乎所有为此构建的组件都假设存在服务器。我们提出了omni-macos，它在其编码器、索引和存储都在已持有文件的Mac上运行，因此没有索引文件、键入的查询或向量会离开设备。它在用户设定的内存预算内，同时运行后台索引器和交互式搜索框：它仅重新编码编辑更改的块，在用户输入时将较小的单元交给GPU，从索引的一位副本回答查询并进行精确重新评分，并将该预算传播到使用统一内存的分配器。我们在五台Mac上进行了测量，其加速器宽度跨度八倍，内存跨度三十二倍，每台都对已持有的文件进行索引。

    arXiv:2608.05543v3 Announce Type: replace  Abstract: A search engine that embeds text, code, documents, images, audio and video into the same representation space has to run its encoder and keep its index somewhere, and almost every component built for the purpose assumes a server. We present omni-macos, which runs its encoder, index and store on the Mac that already holds the files, so no indexed file, no typed query and no vector ever leaves the machine. It keeps a background indexer and an interactive search box inside one memory budget the user sets: it re-encodes only the chunks an edit changes, hands the GPU smaller units while the user is typing, answers queries from a one-bit replica of the index with exact rescoring, and propagates that budget to the allocators that draw on unified memory. We measure on five Macs spanning an eightfold range of accelerator width and a thirty-twofold range of memory, each indexing the files it already holds.
    
[^20]: 面向位置公平密集检索的注意力校准

    Attention Calibration for Position-Fair Dense Retrieval

    [https://arxiv.org/abs/2606.02737](https://arxiv.org/abs/2606.02737)

    该论文提出一种带强度系数的注意力校准方法，缓解密集检索中嵌入表示偏向文本前部内容的位置偏差问题，同时将校准内存开销从数 GiB 降至 1 MiB 以下，显著提升相关片段位于文本后部时的检索性能。

    

    密集检索将一段文本压缩为单个向量，但这种压缩存在位置偏差：文本前部的内容主导嵌入表示，当相关片段出现在较后位置时，检索性能会下降。先前的工作提出了一种推理时方法，通过均衡池化标记（pooling token）在文本各片段上的注意力来抵消这种偏差。然而，该方法存在以下问题：（i）它以固定强度重新分配注意力；（ii）尽管在不同层和不同架构之间存在显著差异，它仍强制将池化标记对自身的注意力固定为统一的段落级质量；（iii）其对检索效果的影响尚未得到评估。我们引入了一个强度系数，在未校准与完全均衡的注意力之间进行插值调节，并提供一种高效实现，将校准的峰值内存开销从 5-7 GiB 降低至 1 MiB 以下。在三个嵌入模型和两种池化方案上的实验表明，适度的校准能够带来更好的检索效果。

    arXiv:2606.02737v2 Announce Type: replace-cross  Abstract: Dense retrieval compresses a passage into a single vector, but this compression is positionally skewed: early content dominates the embedding, and retrieval degrades when the relevant span appears later. Prior work proposed an inference-time method that counteracts this skew by equalizing the pooling token's attention across passage segments. However, (i) it redistributes attention at a fixed strength, (ii) it forces the pooling token's attention to itself to a fixed basket-level mass despite substantial variation across layers and architectures, and (iii) its effect on retrieval has not been evaluated. We introduce a strength coefficient that interpolates between uncalibrated and fully equalized attention, together with an efficient implementation that reduces peak calibration memory overhead from 5-7 GiB to under 1 MiB. Across three embedding models and two pooling schemes, moderate calibration provides a better retrieval tra
    
[^21]: SciNLP：面向自然语言处理领域的全文科学实体与关系抽取专用基准数据集

    SciNLP: A Domain-Specific Benchmark for Full-Text Scientific Entity and Relation Extraction in NLP

    [https://arxiv.org/abs/2509.07801](https://arxiv.org/abs/2509.07801)

    SciNLP是首个提供NLP领域全文实体与关系标注的基准数据集，包含60篇人工标注的全文论文，涵盖6,429个实体和1,649个关系，填补了该领域全文标注数据集的空白。

    

    从科学文献中进行结构化信息抽取对于捕捉专业领域的核心概念和新兴趋势至关重要。尽管现有数据集有助于模型开发，但由于领域复杂性和科学文本标注的高成本，大多数数据集仅关注出版物的特定章节。为了解决这一局限性，我们推出了SciNLP——一个专门用于自然语言处理（NLP）领域全文实体与关系抽取的基准数据集。该数据集包含60篇人工标注的NLP领域全文论文，涵盖6,429个实体和1,649个关系。与现有研究相比，SciNLP是首个提供NLP领域实体及其关系全文标注的数据集。为了验证SciNLP的有效性，我们与类似数据集进行了对比实验，并评估了最先进的监督模型在该数据集上的性能。结果揭示了不同的

    arXiv:2509.07801v5 Announce Type: replace  Abstract: Structured information extraction from scientific literature is crucial for capturing core concepts and emerging trends in specialized fields. While existing datasets aid model development, most focus on specific publication sections due to domain complexity and the high cost of annotating scientific texts. To address this limitation, we introduce SciNLP - a specialized benchmark for full-text entity and relation extraction in the Natural Language Processing (NLP) domain. The dataset comprises 60 manually annotated full-text NLP publications, covering 6,429 entities and 1,649 relation. Compared to existing research, SciNLP is the first dataset providing full-text annotations of entities and their relationships in the NLP domain. To validate the effectiveness of SciNLP, we conducted comparative experiments with similar datasets and evaluated the performance of state-of-the-art supervised models on this dataset. Results reveal varying 
    
[^22]: 超越大型语言模型范式：重新审视自注意力序列推荐

    Revisiting Self-Attentive Sequential Recommendation Beyond the LLM Paradigm

    [https://arxiv.org/abs/2504.09596](https://arxiv.org/abs/2504.09596)

    本文指出推荐系统（熵增目标）与语言建模（熵减目标）在本质上追求相反的目标，这种数据特性——行为数据局部规律而全局异质，语言局部多样而全局收敛——的根本差异才是推荐系统无法复现语言模型缩放规律的原因，并据此超越LLM范式重新审视自注意力序列推荐。

    

    序列推荐几乎在Transformer问世之初便采用了它：SASRec于2018年将解码器移植到下一物品预测任务中，距《Attention is All You Need》发表仅一年，此后该范式一直借鉴语言建模的方法。这两个任务看起来几乎完全相同——都以因果自注意力机制处理整数ID序列——但它们追求的目标却截然相反。推荐系统致力于让更多用户接触更多物品，这是一个熵增目标；而语言模型则致力于将问题的多种表述收敛到唯一答案上，这是一个熵减目标。我们认为，正是这种目标上的根本差异，而非工程层面的努力不足，导致推荐系统无法复现语言模型所展现的清晰缩放规律：行为数据局部规律但全局异质，如同一个赌场；而语言局部多样但全局收敛，如同一座图书馆。以SASRec为切入点，我们重新审视自注意力范式作为比较

    arXiv:2504.09596v2 Announce Type: replace  Abstract: Sequential recommendation adopted the Transformer almost as soon as it appeared: SASRec ported the decoder to next-item prediction in 2018, a year after Attention is All You Need, and the paradigm has borrowed from language modeling ever since. The two tasks look nearly identical, both consume integer-ID sequences with causal self-attention, yet they pursue opposite ends. A recommender works to bring more users into contact with more items, an entropy-increasing goal; a language model works to converge many phrasings of a question onto one answer, an entropy-decreasing one. We argue this difference, not engineering effort, is why recommendation has not reproduced the clean scaling that language models enjoy: behavioral data is locally regular yet globally heterogeneous, a casino, whereas language is locally diverse yet globally convergent, a library. Taking SASRec as an entry point, we revisit the self-attentive paradigm as a compara
    

