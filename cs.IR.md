# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Systematic Study of Semantic ID Spaces for Generative Information Retrieval](https://arxiv.org/abs/2610.08732) | 本文针对生成式信息检索中的核心问题“什么样的DocID才是好的DocID”，首次提出统一框架将乘积量化、残差量化及其混合变体纳入单一设计空间，系统研究了定义有效数值DocID的属性、指标与权衡，从而摆脱对昂贵下游评估的依赖，支持系统性分析与快速迭代。 |
| [^2] | [Disentangling Paradigm, Identifier, and Decoding in Generative Retrieval](https://arxiv.org/abs/2610.08716) | 该论文通过控制变量实验首次解耦了生成式检索中范式、标识符和解码三个因素，发现仅解码方式就能使扩散检索器的Hit@1波动6.6至13.7个百分点，并提出单次评分解码方法，使模型一次性读取掩码标识符并按编码概率为文档打分，从而大幅提升扩散检索性能。 |
| [^3] | [UNREAL: Unifying Retrieval and Long-Context with a Single Model](https://arxiv.org/abs/2610.08463) | UNREAL提出了一种模型原生的证据选择框架，直接从冻结LLM的内部表示中推导检索查询，以不到50万可训练参数统一了语料库检索与长上下文推理，并在多个基准上大幅超越最先进的检索-重排序系统。 |
| [^4] | [Agentic AutoRAG: RAG Pipeline Optimization through Reasoning-Driven Agents](https://arxiv.org/abs/2610.08452) | 该论文提出Agentic AutoRAG，一种利用LLM智能体进行多目标RAG超参数优化的方法，其核心创新在于通过诊断器将每次失败归因于检索或生成阶段，从而让优化器能够推理配置失败的原因并智能地指导后续搜索。 |
| [^5] | [Seeing the Context: Enhancing Recommender Systems with Image-Derived Contextual Signals](https://arxiv.org/abs/2610.08407) | 该论文提出从图像中提取物理、社交和模态三类情境信息的新表示，并通过ICE-Fuse流水线将其融合到情境感知推荐系统中，证明图像情境信号与评论等现有信号形成互补，能够提升推荐效果。 |
| [^6] | [Aligning Performance with Contribution: Towards Contribution-Aware Fair Recommendation](https://arxiv.org/abs/2610.08245) | 论文提出“贡献-性能公平性”这一全新公平视角，要求推荐系统使用户获得的推荐性能与其对模型学习的估计贡献相匹配，从而激励用户持续贡献信息并构建可持续的推荐生态。 |
| [^7] | [Behavior-Mining, Generative Conversations, and Collaborative Advisory: the Future of Travel and Tourism Recommender Systems](https://arxiv.org/abs/2610.08232) | 本文剖析了旅游推荐系统（TTRSs）至今未能广泛普及的三大局限——训练数据集过时稀疏、算法过度追求预测准确性而忽视新颖性与情境相关性、以及未满足旅行者具体需求，并展望了由行为挖掘、生成式对话与协同咨询驱动的旅游推荐系统未来发展方向。 |
| [^8] | [Confidence-Ordering Reversal under Contextual Priors in Neural Decoding](https://arxiv.org/abs/2610.08229) | 论文揭示神经解码中上下文先验引发的“置信度排序反转”现象：当正确候选者初始排名较低时，融合后更大的置信度差距反而预示修复可能性更低，导致初始排名20开外的错误占融合后错误的46.6%。 |
| [^9] | [Adapting Generative Recommenders for Multi-Turn Interaction](https://arxiv.org/abs/2610.08136) | 提出INTEGER框架，通过可学习的路由标记、历史重新锚定和行为回放与指令数据预演机制，将生成式推荐器扩展为多轮交互系统，使用户能在对话中纠正推荐意图，同时不牺牲推荐准确率。 |
| [^10] | [Self-Retrospection Distillation: Turning Post-hoc Experiences into Prior Foresight](https://arxiv.org/abs/2610.08077) | 该论文提出前瞻学习与自我回溯蒸馏方法，将已完成的轨迹作为特权后见之明，蒸馏为同一策略在交互前的轨迹无关的预见性预测，从而在组相对奖励信号消失时仍能从经验中有效学习。 |
| [^11] | [From Delivery to Stateful Exploration: Rethinking the Index for Agentic Search](https://arxiv.org/abs/2610.07960) | 提出IndexAct接口，将候选集精炼与文本检视分离，让智能体通过对倒排索引执行集合运算来操作持久化候选集并获取统计反馈而非直接阅读文本段落。 |
| [^12] | [ShanLiangRen: A Nutrition Agent for Personalized Daily Meal Planning](https://arxiv.org/abs/2610.07886) | 该论文提出了个性化全量化多目标膳食规划问题（MDP），并开发了营养智能体ShanLiangRen，通过将用户需求转化为约束规划实例、以精确检索增强生成缩小候选空间，并结合帕累托原则引导的精细化方法，生成兼顾个性化约束与多维营养目标的每日膳食方案。 |
| [^13] | [Contrastive Learning for Aspect Representation towards Explainable Recommendation](https://arxiv.org/abs/2610.07761) | 该论文提出CLARER推荐模型，通过Transformer编码器和对比学习从评论中提取方面特征，并与评分信息融合，同时提升了推荐的准确性与可解释性。 |
| [^14] | [Token-Budgeted Escalation for Financial Document QA: Cost Is Predictable, Benefit Is the Bottleneck](https://arxiv.org/abs/2610.07760) | 本文将金融文档问答中的选择性升级形式化为令牌预算约束下的批次分配问题，发现额外检索调用的成本高度可预测，但准确识别哪些升级真正有益仍是主要瓶颈。 |
| [^15] | [Learning to Retrieve via Reinforcement Learning in Embedding Space](https://arxiv.org/abs/2610.07731) | 该论文提出RELER强化学习框架，通过从vMF分布采样嵌入动作、结合RLOO基线的REINFORCE算法以及减少采样噪声的条件均值投影（CMP）技术，使现有嵌入模型能够直接在嵌入空间中学习检索并对齐任务特定的奖励。 |
| [^16] | [DBRAG: Multi-Table Retrieval-Augmented Generation for Complex Database Queries](https://arxiv.org/abs/2610.07622) | DBRAG 提出了一个面向多表问答的检索增强生成框架，通过离线表格索引检索候选表、用查询相关行丰富摘要并借助 LLM 重排序，再由程序辅助推理器对完整表格内容执行操作，从而高效处理相关信息分布于多个关系表中的复杂数据库查询。 |
| [^17] | [A Study of Prior Case Retrieval Using Lexical, Semantic, and Rhetorical Role Information in Indian Legal Documents](https://arxiv.org/abs/2610.07437) | 本研究针对印度法律先前案例检索任务进行了三种检索系统设计的实证评估，发现修辞角色的组合检索优于单一角色或全文检索、法条相似性不具区分性，并揭示了内部验证MRR高达0.97的模型在官方评估中MRR仅0.14的泛化失效问题。 |
| [^18] | [Rethinking Semantic ID Construction for Generative Recommendation: SimHash with Parallel Decoding and Semantic Alignment](https://arxiv.org/abs/2610.07402) | 该论文提出FLASH两阶段框架，揭示了SimHash等哈希方法的性能差距源于与自回归解码的结构性不匹配而非哈希本身的局限，并通过并行解码和显式语义对齐使免训练的SimHash在生成式推荐中达到最先进性能。 |
| [^19] | [WildMatch: Weakly Supervised Image Matcher Adaptation for Wildlife Re-Identification](https://arxiv.org/abs/2610.07384) | 提出WildMatch，仅利用个体身份标签（无需关键点或几何对应标注）对预训练关键点匹配器进行弱监督适配，以解决相机陷阱图像中野生动物个体重识别的难题。 |
| [^20] | [The Right Memory in the Wrong Context: Verifying Retrieval Admissibility in Long-Term Agent Memory](https://arxiv.org/abs/2610.07309) | 该论文提出了一个检索可采性验证框架，通过将记忆-查询对标注为“可采纳、不可采纳或未决”三种状态，并在提示词暴露层面追踪记忆ID及其与目标级泄露的关联，从而检测长期记忆智能体在错误语境下检索出“正确的记忆”这一安全隐患。 |
| [^21] | [RAGFlip: Measuring Query-Level Negative Flips in Retriever Upgrades](https://arxiv.org/abs/2610.07266) | 该论文提出RAGFlip框架来测量检索器升级中被聚合指标掩盖的查询级负向翻转，发现尽管新检索器整体表现更优，仍会丢失8.6-37.5%的BM25原本能正确检索的查询。 |
| [^22] | [CroissantMiner: Automated Extraction and Validation of Croissant Metadata for ML Datasets](https://arxiv.org/abs/2610.07132) | 该论文提出了首个针对Croissant元数据提取的端到端评估基准（包含602篇论文的金/银双级标注），并发现单次提取方法在各类模型骨干上始终优于四种智能体架构。 |
| [^23] | [Beyond Successor Accuracy: State Retention for Recursive Self-Improvement in Recommendation](https://arxiv.org/abs/2610.07105) | 该论文提出“分布式进展”概念和跨代优势（CGA）度量，揭示推荐递归自改进中更新前后的模型可能保留互补的排序决策，并利用无需标签的排序分离统计量预测应保留旧模型还是新模型，实验显示最优保留策略因推荐架构而异。 |
| [^24] | [Smart Content Ingestion for Generative AI Workloads](https://arxiv.org/abs/2610.07091) | 本文提出智能内容摄取的理念，指出在生成式AI时代，由于企业知识以PDF、电子表格等异构格式承载多种信息模态，内容提取已演进为AI生命周期中独立且不可替代的关键阶段，其错误无法被下游检索或重排序组件修复。 |
| [^25] | [Beyond Refusal Patterns: Safe-Role Internalization for Robust and Generalizable LLM Safety Alignment](https://arxiv.org/abs/2610.07023) | 提出SSRFT（监督安全角色微调）框架，首次将LLM安全对齐重新表述为对预定义安全角色的内化，通过构建SRQA数据集使模型内化安全价值观与原则，从而以更少的攻击特定监督实现更鲁棒、可泛化的安全对齐，并缓解过度拒绝问题。 |
| [^26] | [Tree Navigation Without LLM Summaries: A Matched-Cost Study of Hierarchical Retrieval for Long-Document QA](https://arxiv.org/abs/2610.06902) | 该论文提出NavTree，证明长文档问答中RAPTOR式摘要树的主要收益来自树的导航结构而非LLM生成的摘要内容，该方法在索引阶段零LLM调用，仅用确定性平衡线段树作为导航支架即可实现有效的分层检索。 |
| [^27] | [Diff-SQL: SQL Efficiency Optimization via Patch Generation and Constraint Alignment](https://arxiv.org/abs/2610.06857) | Diff-SQL提出了一种两阶段框架，先以统一差异补丁的形式生成针对性的SQL优化编辑，再通过在线策略强化学习在可执行性与语义等价性约束下对结果进行修正，从而避免了端到端大模型重写SQL的目标错位问题，实现可靠的SQL效率优化。 |
| [^28] | [Do We Still Need Gazetteers in the Era of LLMs? Chaining Retrieval with a Spatial Neuro-Symbolic Index](https://arxiv.org/abs/2610.05028) | 本文通过空间-语义索引实验评估了文本编码器的表示能否替代地名辞典作为可靠的空间语义索引，并据此提出将检索与空间神经符号索引相链合的方案，回应了“大语言模型时代是否还需要地名辞典”这一问题。 |
| [^29] | [SOLO: Certified-Recall Metric Similarity Search with Scan-Only Sampled Inverted Lists](https://arxiv.org/abs/2610.02387) | SOLO是一种在度量空间中不依赖任何排序启发式的近似最近邻搜索索引，通过将查询路由到随机样本最近点并完整扫描倒排列表，实现了可直接从索引计算和认证的召回率，并满足召回率≈f(b·k_s)的等工作量定律。 |
| [^30] | [AX is the New AEO](https://arxiv.org/abs/2609.34951) | 该论文提出“代理体验（AX）”——即AI智能体能否顺利抓取并阅读企业自身网站——正在取代答案引擎优化（AEO）成为决定AI推荐结果的关键因素，并通过超过3.7万次智能体买家旅程实验加以验证。 |
| [^31] | [Calibrated Uncertainty for Informative Path Planning in Aquatic Environmental Monitoring](https://arxiv.org/abs/2609.34577) | 用校准良好的深度集成替代高斯过程可为水域环境监测的信息路径规划提供更可靠的不确定性估计，在原油泄漏场景模拟中将归一化重构误差降低83%。 |
| [^32] | [More Efficient LLM Reranking with Whole-Pool, Setwise, Long-Context Language Models](https://arxiv.org/abs/2606.01782) | 提出全池集合式重排序方法DualEnd，通过从两端同时填充排序，仅需50次LLM比较即可完成100个候选的完整排序，比现有方法减少59.4%-88.8%的比较次数。 |
| [^33] | [KadiAssistant: A conversational AI Agent for information retrieval in Kadi4Mat](https://arxiv.org/abs/2605.18850) | 本文提出了KadiAssistant，一个集成于Kadi研究数据生态系统中、以隐私保护为设计核心的对话式AI智能体，使研究人员能够高效访问、聚合和综合异构且隐私敏感的研究数据信息。 |
| [^34] | [Enhancing High-order Interaction Awareness in LLM-based Recommender Model](https://arxiv.org/abs/2409.19979) | 本文提出增强型LLM推荐器ELMRec，通过增强全词嵌入使大语言模型无需图预训练即可理解用户-项目高阶交互，并针对LLM偏好早期交互而忽略近期交互的问题提出重排序方案，在直接推荐和序列推荐中均超越现有最先进方法。 |
| [^35] | [Hypergraph-Enhanced Dual Convolutional Network for Bundle Recommendation](https://arxiv.org/abs/2312.11018) | HED 通过构建包含用户-捆绑包、用户-物品、捆绑包-物品交互及用户内部、捆绑包内部关系的完整超图，并将全超图传播与用户-捆绑包双卷积分支耦合，在保留推荐特有信号的同时引入物品感知的高阶上下文，在 NetEase 和 Youshu 数据集上较最强基线显著提升了捆绑包推荐性能。 |

# 详细

[^1]: 生成式信息检索中语义ID空间的系统性研究

    A Systematic Study of Semantic ID Spaces for Generative Information Retrieval

    [https://arxiv.org/abs/2610.08732](https://arxiv.org/abs/2610.08732)

    本文针对生成式信息检索中的核心问题“什么样的DocID才是好的DocID”，首次提出统一框架将乘积量化、残差量化及其混合变体纳入单一设计空间，系统研究了定义有效数值DocID的属性、指标与权衡，从而摆脱对昂贵下游评估的依赖，支持系统性分析与快速迭代。

    

    生成式信息检索（GIR）已成为一种变革性范式，它将文档检索从传统的“检索-排序”工作流程转变为序列到序列的生成方式，由模型直接预测文档标识符。尽管这些DocID的语义设计对性能至关重要，但一个根本性问题仍未得到充分探索：什么样的DocID才是好的DocID？当前方法严重依赖计算代价高昂的下游评估，这阻碍了系统性分析和快速迭代。在这项工作中，我们通过全面研究定义有效数值DocID的属性、指标与权衡来应对这一挑战。具体而言，我们的贡献有三方面：首先，我们提出了一个统一框架，将乘积量化（PQ）、残差量化（RQ）及其混合变体统一到单一设计空间中，这使我们能够系统性地……（摘要原文在此处截断）

    arXiv:2610.08732v1 Announce Type: cross  Abstract: Generative Information Retrieval (GIR) has emerged as a transformative paradigm, shifting document retrieval from a traditional "retrieve-and-rank" workflow to sequence-to-sequence generation, where a model directly predicts document identifiers (DocIDs). While the semantic design of these DocIDs is known to be critical for performance, a fundamental question remains under-explored: what makes a good DocID? Current approaches rely heavily on computationally expensive downstream evaluations, hindering systematic analysis and rapid iteration. In this work, we address this challenge by presenting a comprehensive study on the properties, metrics, and trade-offs that define effective numerical DocIDs. Specifically, our contributions are threefold: First, we propose a unified framework that unifies Product Quantization (PQ) and Residual Quantization (RQ), and their hybrid variants within a single design space. This enables us to systematical
    
[^2]: 解耦生成式检索中的范式、标识符与解码方法

    Disentangling Paradigm, Identifier, and Decoding in Generative Retrieval

    [https://arxiv.org/abs/2610.08716](https://arxiv.org/abs/2610.08716)

    该论文通过控制变量实验首次解耦了生成式检索中范式、标识符和解码三个因素，发现仅解码方式就能使扩散检索器的Hit@1波动6.6至13.7个百分点，并提出单次评分解码方法，使模型一次性读取掩码标识符并按编码概率为文档打分，从而大幅提升扩散检索性能。

    

    生成式检索训练语言模型来生成相关文档的标识符。近期工作用扩散模型取代自回归解码器，但同时改变了标识符、训练方案和解码方式，因此性能差异无法归因于范式本身。在NQ320K和MS300K数据集上，我们使用残差量化、乘积量化和随机标识符训练了自回归、掩码扩散和块扩散模型。在固定标识符长度和训练预算的条件下，我们用多种方式对每个模型进行解码。仅解码方式一项就能使扩散模型的Hit@1指标变动6.6至13.7个百分点。我们的参考扩散解码方法“生成-匹配”先生成一个标识符，然后检索语料库中最接近的标识符，而生成标识符完全正确的比例在NQ320K查询中仅为14-21%。我们测试了一种单次评分方法来解码扩散检索器：模型一次性读取完全掩码的标识符，每个文档根据其编码的概率进行评分。

    arXiv:2610.08716v1 Announce Type: cross  Abstract: Generative retrieval trains a language model to generate the identifier of a relevant document. Recent work replaces the autoregressive decoder with diffusion, but changes identifiers, training recipe and decoding at once, so differences cannot be credited to the paradigm. On NQ320K and MS300K, we train autoregressive, masked-diffusion and block-diffusion models with residual-quantised, product-quantised and random identifiers. With identifier length and training budget fixed, we decode each model in several ways. Decoding alone moves a diffusion model's Hit@1 by 6.6 to 13.7 points. Our reference diffusion decoding, generate-and-match, generates an identifier, then retrieves the closest corpus identifiers. The generated identifier is right for 14-21% of NQ320K queries. We test one-pass scoring to decode diffusion retrievers: the model reads a fully masked identifier once, and each document is scored by its codes' probabilities. It matc
    
[^3]: UNREAL：用单一模型统一检索与长上下文

    UNREAL: Unifying Retrieval and Long-Context with a Single Model

    [https://arxiv.org/abs/2610.08463](https://arxiv.org/abs/2610.08463)

    UNREAL提出了一种模型原生的证据选择框架，直接从冻结LLM的内部表示中推导检索查询，以不到50万可训练参数统一了语料库检索与长上下文推理，并在多个基准上大幅超越最先进的检索-重排序系统。

    

    长上下文推理和检索增强生成（RAG）在截然不同的尺度上处理证据选择问题，范围从单个长提示词到整个语料库。我们探究是否存在一种单一的模型内部机制能够覆盖这一范围并完成证据选择。我们提出了UNREAL（用单一模型统一检索与长上下文），一个模型原生的证据选择框架，可同时覆盖语料库检索和长上下文推理。UNREAL对文本块进行编码，并直接从冻结的大语言模型（LLM）的内部表示中推导检索查询。它仅增加不到50万个可训练参数，且完全不改动骨干模型。在一个包含30亿标记、2100万文本块的维基百科索引上，全部四种稠密型和混合型UNREAL骨干模型均超越了最先进的检索器-重排序器系统。其中最佳模型将HotpotQA上的召回率从49.1%提升至73.2%，将2WikiMultiHopQA上的召回率从31.7%提升至60.1%，将MuSiQue上的召回率从8.8%提升至14.4%。当应用于长上下文任务时，同样的选择机制……（摘要到此截断）

    arXiv:2610.08463v1 Announce Type: new  Abstract: Long-context inference and Retrieval-Augmented Generation (RAG) handle evidence selection at vastly different scales, from a single long prompt to an entire corpus. We ask whether a single model-internal mechanism can select evidence across this range. We introduce UNifying REtrieval And Long-Context with a Single Model (UNREAL), a model-native evidence selection framework to span corpus retrieval and long-context inference. UNREAL encodes chunks and derives retrieval queries directly from the frozen LLM's internal representations. It adds fewer than 500K trainable parameters and leaves the backbone unchanged. On a 3B-token, 21M-chunk Wikipedia index, all four dense and hybrid UNREAL backbones outperform state-of-the-art retriever-reranker systems. The best model raises recall from 49.1% to 73.2% on HotpotQA, from 31.7% to 60.1% on 2WikiMultiHopQA, and from 8.8% to 14.4% on MuSiQue. Applied to long-context tasks, the same selection mecha
    
[^4]: Agentic AutoRAG：通过推理驱动的智能体进行RAG流程优化

    Agentic AutoRAG: RAG Pipeline Optimization through Reasoning-Driven Agents

    [https://arxiv.org/abs/2610.08452](https://arxiv.org/abs/2610.08452)

    该论文提出Agentic AutoRAG，一种利用LLM智能体进行多目标RAG超参数优化的方法，其核心创新在于通过诊断器将每次失败归因于检索或生成阶段，从而让优化器能够推理配置失败的原因并智能地指导后续搜索。

    

    检索增强生成（RAG）是一种被广泛使用的方法，用于将大语言模型（LLM）扎根于外部知识。然而，配置一个RAG流程是一个代价高昂的超参数优化问题，涉及众多相互关联的选择，从分块和嵌入模型到重排序和生成。现有的优化器，从贪心搜索到贝叶斯优化，都将每次试验简化为一个汇总分数进行搜索，而不会建模某个配置为何会有那样的表现，尽管检索到的文本块其实已经提供了关于每次失败究竟发生在检索阶段还是检索之后的证据。我们提出了Agentic AutoRAG，一个用于多目标RAG超参数优化的LLM智能体优化器，具备检索与生成之间的失败归因能力。它提出候选配置，并在从语料库构建的固定考题上进行评分：每次试验后，一个诊断器会将每个失败的问题归因于检索阶段或生成阶段，而一个提议器则基于……（原文摘要在此处被截断，后续内容未能提供）

    arXiv:2610.08452v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) is a widely used approach for grounding large language models (LLMs) in external knowledge. However, configuring a pipeline is an expensive hyperparameter optimization problem over many interacting choices, from chunking and embedding model to reranking and generation. Existing optimizers, from greedy search to Bayesian optimization, reduce each trial to an aggregate score and search without modeling why a configuration performed as it did, even though the retrieved chunks already provide evidence about whether each failure occurred during retrieval or after it. We introduce Agentic AutoRAG, an LLM-agent optimizer for multi-objective RAG hyperparameter optimization with retrieval-versus-generation failure attribution. It proposes configurations scored on a frozen exam from the corpus: after each trial a Diagnoser attributes each failed question to retrieval or generation, and a Proposer, grounded in a
    
[^5]: 看见情境：利用图像衍生的情境信号增强推荐系统

    Seeing the Context: Enhancing Recommender Systems with Image-Derived Contextual Signals

    [https://arxiv.org/abs/2610.08407](https://arxiv.org/abs/2610.08407)

    该论文提出从图像中提取物理、社交和模态三类情境信息的新表示，并通过ICE-Fuse流水线将其融合到情境感知推荐系统中，证明图像情境信号与评论等现有信号形成互补，能够提升推荐效果。

    

    情境信息捕捉用户与物品交互时的具体背景，是推荐系统的核心。以往的研究从位置、时间或评论中获取情境信息，但从未从图像中获取；多模态推荐系统主要利用图像来丰富物品或用户的表示，而非识别情境。我们提出了一种从图像中衍生的新型情境表示，涵盖通过视觉-语言模型学习的物理、社交和模态三个类别。我们引入了ICE-Fuse，这是一个用于评估该表示的流水线，它将这三个类别融合并集成到情境感知推荐系统中，使用TripAdvisor数据和评论感知图对比学习作为推荐算法。图像情境单独使用时并不优于现有的情境信号，但将两者结合时能够带来提升，这表明图像包含了互补信息。语义分析表明，图像衍生的情境与评论衍生的情境捕捉了交互的不同方面。

    arXiv:2610.08407v1 Announce Type: new  Abstract: Contextual information, capturing the circumstances of a user-item interaction, is central to recommender systems. Prior work draws context from location, time, or reviews, but not images; multimodal recommender systems mainly use images to enrich item or user representations, not identify situational context. We propose a new representation of context derived from images, spanning physical, social, and modal categories learned via a vision-language model. We introduce ICE-Fuse, a pipeline for evaluating this representation that fuses these categories and integrates them into a context-aware recommender system, using TripAdvisor data and Review-aware Graph Contrastive Learning as the recommendation algorithm. Image context does not outperform established signals standalone, but improves them combined, indicating complementary information. Semantic analysis shows image- and review-derived context capture distinct aspects of the interactio
    
[^6]: 使性能与贡献相匹配：迈向贡献感知的公平推荐

    Aligning Performance with Contribution: Towards Contribution-Aware Fair Recommendation

    [https://arxiv.org/abs/2610.08245](https://arxiv.org/abs/2610.08245)

    论文提出“贡献-性能公平性”这一全新公平视角，要求推荐系统使用户获得的推荐性能与其对模型学习的估计贡献相匹配，从而激励用户持续贡献信息并构建可持续的推荐生态。

    

    现有关于推荐系统中用户公平性的研究已经发展出了多样化的目标。然而，这些研究对一个独特的分配视角关注有限：用户对模型学习的贡献是否应该体现在他们所获得的推荐收益中。我们认为，除了现有的公平性保护之外，一个公平的系统还应考虑用户的估计贡献与其所获得的推荐性能之间的一致性。这种一致性可以激励用户持续且富有信息量的参与，从而支持一个可持续的推荐生态系统。为此，我们提出了贡献-性能公平性这一全新的公平性视角，它要求推荐性能在跨用户群体之间与估计贡献保持一致，并在同一群体中贡献相当的用户之间保持公平。为将这一视角具体化，我们引入了贡献-性能（原文摘要至此截断）。

    arXiv:2610.08245v1 Announce Type: new  Abstract: Existing research on user fairness in recommender systems has developed diverse objectives. However, it has paid limited attention to a distinct distributive perspective: whether users' contributions to model learning should be reflected in the recommendation benefits they receive. We argue that, in addition to existing fairness protections, a fair system may account for the alignment between users' estimated contributions and the recommendation performance they receive. Such alignment can incentivize sustained and informative engagement, thereby supporting a sustainable recommendation ecosystem. To this end, we propose Contribution-Performance Fairness, a novel fairness perspective which requires recommendation performance to be aligned with estimated contribution across user groups and to remain equitable among users with comparable contributions within a same group. To instantiate this perspective, we introduce the Contribution-Perfor
    
[^7]: 行为挖掘、生成式对话与协同咨询：旅游与旅游业推荐系统的未来

    Behavior-Mining, Generative Conversations, and Collaborative Advisory: the Future of Travel and Tourism Recommender Systems

    [https://arxiv.org/abs/2610.08232](https://arxiv.org/abs/2610.08232)

    本文剖析了旅游推荐系统（TTRSs）至今未能广泛普及的三大局限——训练数据集过时稀疏、算法过度追求预测准确性而忽视新颖性与情境相关性、以及未满足旅行者具体需求，并展望了由行为挖掘、生成式对话与协同咨询驱动的旅游推荐系统未来发展方向。

    

    自电子商务早期应用以来，旅游与旅游业一直是推荐系统设计的试验田：这类工具帮助旅行者选择目的地、航班和住宿，并将它们组合成完整行程。从基于案例的推理到强化学习，各种数据驱动的推荐技术已被调整以适应旅行者的需求。研究界已经开发出形式多样的旅游推荐系统（TTRSs）原型，这些系统依赖于具体情境、面向多方利益相关者，并且近来还开始应对过度旅游等可持续性问题。尽管有这些长期的努力，TTRSs至今仍未得到广泛应用。我们认为有三个局限可以解释这一现象：用于训练和验证TTRSs的数据集过时且稀疏，算法过度优先考虑预测准确性而忽视了新颖性和情境相关性等领域特定维度，以及未能满足旅行者的具体需求。

    arXiv:2610.08232v1 Announce Type: new  Abstract: Since the early adoption of e-commerce, travel and tourism has been a lab for the design of recommender systems: tools that help travelers choose destinations, flights, accommodations, and combine them into itineraries. Data-driven recommendation techniques, ranging from case-based reasoning to reinforcement learning, have been adapted to travelers' needs. The research community has produced multifaceted prototypes of travel and tourism recommender systems (TTRSs), which are context-dependent, multistakeholder-oriented, and more recently, addressing sustainability issues, such as overtourism. Despite this enduring work, TTRSs are not widespread yet. We argue that three limitations can explain this: outdated and sparse data sets used to train and validate TTRSs, algorithms that prioritize prediction accuracy over domain-specific dimensions such as novelty and contextual relevance, and a failure to address the specific needs of travelers. 
    
[^8]: 神经解码中上下文先验下的置信度排序反转

    Confidence-Ordering Reversal under Contextual Priors in Neural Decoding

    [https://arxiv.org/abs/2610.08229](https://arxiv.org/abs/2610.08229)

    论文揭示神经解码中上下文先验引发的“置信度排序反转”现象：当正确候选者初始排名较低时，融合后更大的置信度差距反而预示修复可能性更低，导致初始排名20开外的错误占融合后错误的46.6%。

    

    上下文先验通过重塑候选分数来改进从神经信号到语言的解码。然而，置信度是从同一重塑后的分数中读取的，因此先验遗留下的错误可能在准确率没有任何变化来揭示它们的情况下变得更加自信。我们研究先验如何在MEG-MASC和MOUS数据集的语音检索中塑造置信度，使用局部解码分数、通过加性浅融合结合的上下文先验，以及融合后的前两名差距作为置信度。在最初错误的预测中，我们发现了一种置信度排序反转现象：当正确候选者最初位于局部排名靠前位置时，更大的差距使修复更有可能发生；但当正确候选者最初排名较低时，更大的差距反而使修复更不可能。在MEG-MASC上，汇总的判断正确性AUROC为0.87，但区分修复与残留错误的AUROC从初始排名2-3时的0.70降至排名21-50时的0.39。初始排名超过20（位于反转区域内）的错误占融合后所有错误的46.6%。我们提出……（原文摘要在此处截断）

    arXiv:2610.08229v1 Announce Type: new  Abstract: Contextual priors improve neural-to-language decoding by reshaping candidate scores. However, confidence is read from the same reshaped scores, so the errors a prior leaves behind can become more confident with no change in accuracy to reveal it. We study how a prior shapes confidence in speech retrieval on MEG-MASC and MOUS using local decoding scores, a contextual prior combined by additive shallow fusion, and the fused top-two margin as confidence. Among initially incorrect predictions, we find a confidence-ordering reversal: a larger margin makes a repair more likely when the correct candidate starts near the top of the local ranking, but less likely when it starts lower. On MEG-MASC, pooled correctness AUROC is 0.87, yet AUROC separating repairs from residual errors falls from 0.70 at initial ranks 2-3 to 0.39 at ranks 21-50. Errors starting beyond rank 20, inside the reversed region, make up 46.6% of all post-fusion errors. We prop
    
[^9]: 面向多轮交互的生成式推荐器适配

    Adapting Generative Recommenders for Multi-Turn Interaction

    [https://arxiv.org/abs/2610.08136](https://arxiv.org/abs/2610.08136)

    提出INTEGER框架，通过可学习的路由标记、历史重新锚定和行为回放与指令数据预演机制，将生成式推荐器扩展为多轮交互系统，使用户能在对话中纠正推荐意图，同时不牺牲推荐准确率。

    

    生成式推荐器从用户的交互历史中解码物品，但当推荐偏离用户当前意图时，用户没有任何途径进行纠正。添加对话能力是一个自然的选择，因为物品和词语共享相同的输出空间，然而训练模型进行对话可能会覆盖模型所依赖的历史到物品的映射。我们提出INTEGER（**交互式生成式推荐**，**INTE**ractive **GE**nerative **R**ecommendation），将生成式推荐扩展至多轮交互：引入一个可学习的路由标记，让模型自主决定何时进行推荐；通过历史重新锚定机制，使每个物品的生成同时以过去的行为和对话内容为条件；并采用行为回放与指令数据预演，防止模型在适配过程中发生遗忘。由此，用户可以在对话中直接对推荐结果给出反馈，同时推荐仍扎根于行为历史，推荐准确率不会为对话流畅性而妥协。在Amazon Beauty和Toys数据集上，INTEGER达到或超过……

    arXiv:2610.08136v1 Announce Type: new  Abstract: Generative recommenders decode items from a user's interaction history, but offer no way for users to correct a recommendation that misses their current intent. Adding conversation is natural since items and words share same output space, yet training the model to converse may overwrite the history-to-item mapping it relies on. We introduce INTEGER (**INTE**ractive **GE**nerative **R**ecommendation), which extends generative recommendation to multi-turn interaction with a learned routing token that lets the model decide when to recommend, history re-anchoring that conditions each item on both past behavior and the dialogue, and behavioral replay with instruction-data rehearsal that prevents forgetting during adaptation. Users can thus give feedback on recommendations within the dialogue, while recommendations stay grounded in behavioral history and accuracy is not traded for fluency. On Amazon Beauty and Toys, INTEGER matches or exceeds 
    
[^10]: 自我回溯蒸馏：将事后经验转化为先验预见

    Self-Retrospection Distillation: Turning Post-hoc Experiences into Prior Foresight

    [https://arxiv.org/abs/2610.08077](https://arxiv.org/abs/2610.08077)

    该论文提出前瞻学习与自我回溯蒸馏方法，将已完成的轨迹作为特权后见之明，蒸馏为同一策略在交互前的轨迹无关的预见性预测，从而在组相对奖励信号消失时仍能从经验中有效学习。

    

    具有可验证奖励的强化学习（RLVR）主要通过交互后的标量结果奖励将智能体经验转化为学习信号。然而，对于组相对目标而言，当所有采样轨迹获得相同奖励时，这一信号便会消失，即使这些轨迹可能揭示了关于任务需求以及智能体如何失败的有用信息。我们提出了一个互补的问题：事后反思能否教会智能体在行动之前本可预见的东西？我们引入前瞻学习，利用事后经验来监督交互前视角下的预见性预测，并通过自我回溯蒸馏（SRD）加以实例化。直观地说，一条已完成的轨迹揭示了本会有用的知识和本应避免的陷阱；SRD将这种特权的后见之明蒸馏到同一策略的、不依赖轨迹的前瞻预测中。前瞻仅作为训练目标，无需成为……

    arXiv:2610.08077v1 Announce Type: new  Abstract: Reinforcement learning with verifiable rewards (RLVR) turns agent experience into learning signals primarily through scalar outcome rewards after interaction. For group-relative objectives, however, this signal vanishes when all rollouts receive the same reward, even though their trajectories may reveal useful information about what the task requires and how the agent fails. We ask a complementary question: can hindsight teach an agent what it could have anticipated before acting? We introduce prospective learning, which uses post-hoc experience to supervise foresight predictions from the pre-interaction view, and instantiate it with Self-Retrospection Distillation (SRD). Intuitively, a completed trajectory reveals knowledge that would have been useful and pitfalls that should be avoided; SRD distills this privileged hindsight into trajectory-blind foresight of the same policy. Foresight serves only as a training target and need not be e
    
[^11]: 从投递到有状态探索：重新思考智能体搜索中的索引

    From Delivery to Stateful Exploration: Rethinking the Index for Agentic Search

    [https://arxiv.org/abs/2610.07960](https://arxiv.org/abs/2610.07960)

    提出IndexAct接口，将候选集精炼与文本检视分离，让智能体通过对倒排索引执行集合运算来操作持久化候选集并获取统计反馈而非直接阅读文本段落。

    

    智能体搜索的最新进展使大型语言模型（LLM）智能体能够对语料库探索实现更精细的控制。然而，即使关于候选集的反馈足以支持下一步决策，搜索接口仍常常返回匹配的文本段落，这将候选集的精炼与源文本的暴露耦合在一起。我们提出了IndexAct，一个面向索引原生语料库交互的接口，它将候选集的精炼与文本检视分离开来。智能体通过倒排索引上的词法条件和集合运算来构建并操作持久化的候选集，接收可复用的状态引用和统计信息（如候选数量），而非匹配的文本段落。这种反馈引导进一步的精炼，而单独请求的文本段落则提供新的线索或证据，为后续对保留候选集的操作提供参考。在涵盖智能体搜索和多跳问答的五个基准上的实验表明

    arXiv:2610.07960v1 Announce Type: new  Abstract: Recent advances in agentic search have given large language model (LLM) agents finer control over corpus exploration. However, search interfaces often return matching passages even when feedback about the candidate set would suffice for the next decision, coupling candidate refinement with source-text exposure. We propose IndexAct, an interface for Index-Native Corpus Interaction that separates candidate-set refinement from text inspection. Agents construct and manipulate persistent candidate sets through lexical conditions and set operations over an inverted index, receiving reusable state references and statistics such as candidate counts rather than matching passages. This feedback guides further refinement, while separately requested passages provide new clues or evidence that can inform subsequent operations on retained candidate sets. Experiments on five benchmarks spanning agentic search and multi-hop question answering show that 
    
[^12]: ShanLiangRen：一个用于个性化每日膳食规划的营养智能体

    ShanLiangRen: A Nutrition Agent for Personalized Daily Meal Planning

    [https://arxiv.org/abs/2610.07886](https://arxiv.org/abs/2610.07886)

    该论文提出了个性化全量化多目标膳食规划问题（MDP），并开发了营养智能体ShanLiangRen，通过将用户需求转化为约束规划实例、以精确检索增强生成缩小候选空间，并结合帕累托原则引导的精细化方法，生成兼顾个性化约束与多维营养目标的每日膳食方案。

    

    膳食营养规划在慢性病管理和保持身体健康方面发挥着重要作用。在实际应用中，它必须同时满足个性化约束和合理的多维营养目标。这两个方面常常相互冲突，且用户约束会随着反馈不断演变，导致通用指南与可执行方案之间存在巨大差距。为了弥合这一差距，我们首先提出了个性化全量化多目标膳食规划问题（MDP）。为了解决MDP，我们开发了一个营养智能体ShanLiangRen。该系统首先将膳食规范、营养数据、用户属性和自然语言需求转化为个性化的约束规划实例；然后采用精确的检索增强生成方法，从大规模的食材与食谱空间中缩小可行候选集；最后，采用受帕累托原则指导的精细化方法……

    arXiv:2610.07886v1 Announce Type: new  Abstract: Dietary nutrition planning plays an important role in chronic disease management and maintaining a healthy body. In applications, it must simultaneously satisfy personalized constraints and reasonable multidimensional nutritional goals. These two aspects often conflict, and user constraints evolve with feedback, resulting in a substantial gap between generic guidelines and executable plans. To bridge this gap, we first propose the personalized fully quantified multiobjective dietary planning problem (MDP). To tackle MDP, we develop a nutrition agent, ShanLiangRen. The system first transforms dietary specifications, nutrient data, user attributes and natural language requirements into an individualized constrained planning instance. It then employs an exact retrieval-augmented generation method to shrink the feasible candidate set from a large scale ingredient and recipe space. Finally, it adopts a refinement guided by Pareto principles, 
    
[^13]: 面向可解释推荐的方面表示对比学习方法

    Contrastive Learning for Aspect Representation towards Explainable Recommendation

    [https://arxiv.org/abs/2610.07761](https://arxiv.org/abs/2610.07761)

    该论文提出CLARER推荐模型，通过Transformer编码器和对比学习从评论中提取方面特征，并与评分信息融合，同时提升了推荐的准确性与可解释性。

    

    在这项工作中，我们提出了一种新颖的推荐模型 CLARER（Contrastive Learning for Aspect Representation towards Explainable Recommendation，面向可解释推荐的方面表示对比学习），该模型将从文本评论中学习到的方面特征与评分信息相融合，以提升推荐的准确性和可解释性。我们提出的框架通过结合基于评分的特征和来自评论的基于方面的特征来学习用户和物品的表示。具体而言，基于评分的特征通过多层感知机（MLP）模型学习，而特定方面的评论表示则使用 Transformer 编码器捕获语义信息，并通过对比学习更好地区分用户偏好。为了提供解释，我们训练了一个 Transformer 解码器，将来自评分特征和方面特征的最终用户与物品表示作为上下文。在三个基准数据集上的实验结果表明，我们的模型

    arXiv:2610.07761v1 Announce Type: cross  Abstract: In this work, we propose a novel recommendation model, CLARER (Contrastive Learning for Aspect Representation towards Explainable Recommendation) that integrates aspect features learned from textual reviews with rating information to improve the accuracy and explainability of recommendations. Our proposed framework learns user and item representations by combining rating-based features and aspect-based features from reviews. Specifically, rating-based features are learned through a multi-layer perceptron (MLP) model, while aspect-specific review representations are learned using a transformer encoder to capture the semantic information and contrastive learning to better distinguish user preferences. To provide explanations, we train a transformer decoder, using the final representations of users and items from both rating and aspect-based features as context. Experimental results in three benchmark data sets demonstrate that our model 
    
[^14]: 面向金融文档问答的令牌预算升级机制：成本可预测，收益才是瓶颈

    Token-Budgeted Escalation for Financial Document QA: Cost Is Predictable, Benefit Is the Bottleneck

    [https://arxiv.org/abs/2610.07760](https://arxiv.org/abs/2610.07760)

    本文将金融文档问答中的选择性升级形式化为令牌预算约束下的批次分配问题，发现额外检索调用的成本高度可预测，但准确识别哪些升级真正有益仍是主要瓶颈。

    

    检索增强生成系统可以将困难查询路由到更深层的上下文，但批量部署必须在各次调用之间分配共享的令牌预算，而这些调用的成本因查询而异。我们将选择性升级问题形式化为金融文档问答的有限批次分配问题。150个FinanceBench问题中的每一个首先获得top-1检索答案。预测器估计可选的top-5调用的裁定质量增益和令牌成本，分配器则根据预测的每令牌增益对调用进行优先级排序。在名义10%预算下，每令牌增益分配比仅增益排序将裁定质量提高了0.034（95%文档自助法置信区间 [0.001, 0.072]），同时比单次top-5检索节省46.6%的总令牌用量。额外调用的成本可以被准确预测（R方为0.93），而有益的升级仍然难以排序识别（AUROC为0.60）。这些结果表明，异构成本在紧（原文摘要在此处截断）

    arXiv:2610.07760v1 Announce Type: new  Abstract: Retrieval-augmented generation systems can route difficult queries to deeper context, but batch deployments must allocate a shared token budget across calls whose costs vary by query. We formulate selective escalation as finite-batch allocation for financial document question answering. Each of 150 FinanceBench questions first receives a top-1 retrieval answer. Predictors estimate the adjudication-quality gain and token cost of an optional top-5 call, and the allocator prioritizes calls by predicted gain per token. At the nominal 10% budget, gain-per-token allocation improves adjudication quality over gain-only ranking by 0.034 (95% document-bootstrap CI [0.001, 0.072]) while using 46.6% fewer total tokens than one-pass top-5 retrieval. Additional-call cost is accurately predictable (R-squared 0.93), whereas beneficial escalation remains difficult to rank (AUROC 0.60). These results show that heterogeneous cost is actionable under tight 
    
[^15]: 在嵌入空间中通过强化学习学习检索

    Learning to Retrieve via Reinforcement Learning in Embedding Space

    [https://arxiv.org/abs/2610.07731](https://arxiv.org/abs/2610.07731)

    该论文提出RELER强化学习框架，通过从vMF分布采样嵌入动作、结合RLOO基线的REINFORCE算法以及减少采样噪声的条件均值投影（CMP）技术，使现有嵌入模型能够直接在嵌入空间中学习检索并对齐任务特定的奖励。

    

    密集检索模型通常使用对比目标进行训练，这类目标虽然能学习有效的表示，但无法直接优化检索指标或下游任务性能。为了解决这一问题，我们提出了RELER（面向检索的强化学习），这是一个强化学习框架，能够使现有的嵌入模型直接在嵌入空间中学习检索，并与任务特定的奖励对齐。我们通过以下方式训练RELER：从以归一化编码器输出为中心的von Mises-Fisher（vMF）分布中采样单位长度的查询和文档嵌入动作，将由此产生的检索或下游结果评分作为奖励，并使用留一法基线（RLOO）通过REINFORCE算法更新编码器。由于在高维嵌入空间中进行探索容易受到采样噪声的影响，我们进一步提出了条件均值投影（CMP），将每个采样的嵌入投影到低维子空间上……

    arXiv:2610.07731v1 Announce Type: cross  Abstract: Dense retrieval models are typically trained with contrastive objectives that learn effective representations but do not directly optimize retrieval metrics or downstream task performance. To address this problem, we introduce RELER (REinforcement LEarning for Retrieval), a reinforcement learning framework that enables existing embedding models to learn to retrieve directly in embedding space and align to task-specific rewards. We train RELER by sampling unit-length query and document embedding actions from von Mises-Fisher (vMF) distributions centered on normalized encoder outputs, scoring the resulting retrieval or downstream outcomes as rewards, and updating the encoder with REINFORCE using a leave-one-out baseline (RLOO). As exploration in the high-dimensional embedding space is prone to sampling noise, we further propose conditional-mean projection (CMP), which projects each sampled embedding onto the low-dimensional subspace span
    
[^16]: DBRAG：面向复杂数据库查询的多表检索增强生成

    DBRAG: Multi-Table Retrieval-Augmented Generation for Complex Database Queries

    [https://arxiv.org/abs/2610.07622](https://arxiv.org/abs/2610.07622)

    DBRAG 提出了一个面向多表问答的检索增强生成框架，通过离线表格索引检索候选表、用查询相关行丰富摘要并借助 LLM 重排序，再由程序辅助推理器对完整表格内容执行操作，从而高效处理相关信息分布于多个关系表中的复杂数据库查询。

    

    大型语言模型的最新进展为结构化数据推理带来了新的能力，尤其是通过能够分析表格的程序辅助工具。然而，许多现有方法仅针对单表场景，或者假设相关表格已经被预先给定。在实际应用中，用户往往会对整个数据库提出复杂的数据探索查询，而相关信息可能分布在多个关系表中。在这项工作中，我们提出了 DBRAG，一个专为多表问答设计的检索增强生成框架。DBRAG 首先利用离线表格索引检索候选表格，通过与查询相关的行来丰富表格摘要，并使用大语言模型对候选表格进行重排序。随后，一个程序辅助推理器选择所需的表格并对其完整内容执行操作，从而保持初始提示上下文的紧凑。在 Spider、GeoQuery 和 ATIS 数据集上的实验……

    arXiv:2610.07622v1 Announce Type: new  Abstract: Recent advancements in large language models have introduced new capabilities for reasoning over structured data, particularly through program-aided tools that can analyze tables. However, many existing methods address single-table scenarios or assume that the relevant tables are already provided. In practice, users often issue complex data exploration queries over entire databases, where relevant information may be distributed across multiple relations. In this work, we introduce DBRAG, a retrieval-augmented generation framework tailored for multi-table question answering. DBRAG first retrieves candidate tables using an offline table index, enriches their summaries with query-relevant rows, and uses an LLM to rerank the candidates. A program-aided reasoner then selects the required tables and executes operations over their full contents, keeping the initial prompt context compact. Experiments on the Spider, GeoQuery, and ATIS datasets u
    
[^17]: 印度法律文档中使用词汇、语义和修辞角色信息进行先前案例检索的研究

    A Study of Prior Case Retrieval Using Lexical, Semantic, and Rhetorical Role Information in Indian Legal Documents

    [https://arxiv.org/abs/2610.07437](https://arxiv.org/abs/2610.07437)

    本研究针对印度法律先前案例检索任务进行了三种检索系统设计的实证评估，发现修辞角色的组合检索优于单一角色或全文检索、法条相似性不具区分性，并揭示了内部验证MRR高达0.97的模型在官方评估中MRR仅0.14的泛化失效问题。

    

    在印度法律判决中检索先前案例的问题，涉及如何从单纯的词汇相似性中区分出相关的法律事实，因为与查询判决共享同一法条的先前案例不一定相关。本文对IL PCR（印度法律先前案例检索）任务的三种连续设计的检索系统进行了实证评估。我们证明了修辞角色（事实、判决理由、先例、论据、法条）的组合优于单独使用各个角色或执行全文检索；法条相似性不具区分性；使用LLM生成训练数据的法律蕴含重排序器在官方评估中的表现远差于其内部验证得分。两个排序器的内部MRR高达0.97，但在官方评估中表现不佳（MRR低至0.14）。因此，我们提出了一种新的设计……

    arXiv:2610.07437v1 Announce Type: new  Abstract: For retrieving prior cases in Indian legal judgments, the problem involves distinguishing relevant legal facts from mere lexical similarities because a prior case that shares a statute with the query judgment is not necessarily relevant. In this paper, we describe an empirical evaluation of three consecutive designs of retrieval systems for the IL PCR(Indian Legal Prior Case Retrieval) task. We demonstrate that the combination of the rhetorical roles (Fact, Ratio Of The Decision, Precedent, Argument, Statute) is better than either using individual roles or performing full text retrieval, that statute similarity is not discriminative, and that a legal entailment reranker with training data produced by an LLM is much worse in terms of official evaluation than its internal validation score. Two rankers, with excellent internal MRR up to 0.97, performed poorly in terms of official evaluation (MRR as low as 0.14). Thus, we propose a new desig
    
[^18]: 重新思考生成式推荐中的语义ID构建：融合并行解码与语义对齐的SimHash方法

    Rethinking Semantic ID Construction for Generative Recommendation: SimHash with Parallel Decoding and Semantic Alignment

    [https://arxiv.org/abs/2610.07402](https://arxiv.org/abs/2610.07402)

    该论文提出FLASH两阶段框架，揭示了SimHash等哈希方法的性能差距源于与自回归解码的结构性不匹配而非哈希本身的局限，并通过并行解码和显式语义对齐使免训练的SimHash在生成式推荐中达到最先进性能。

    

    基于语义ID的生成式推荐将每个物品表示为离散token序列，从而实现对物品语义的结构化建模。一个关键挑战是构建既具有语义表达能力又具备计算高效性的语义ID。尽管近期方法更倾向于复杂的可学习量化技术，但基于简单哈希的方法（如SimHash）被普遍认为在本质上性能较差。在这项工作中，我们挑战了这一共识，通过研究表明，表观上的性能差距并非源于哈希方法的固有局限性，而是源于其与自回归解码之间的结构性不匹配，以及刚性离散化过程中不可避免的信息损失。基于这一洞察，我们提出了FLASH，一个两阶段框架，通过并行解码和显式语义对齐来重新激活无需训练的SimHash分词方法。尽管方法简单，FLASH在多个基准上实现了最先进的性能。

    arXiv:2610.07402v1 Announce Type: new  Abstract: Semantic ID-based generative recommendation represents each item as a sequence of discrete tokens, enabling structured modeling of item semantics. A critical challenge is constructing semantic IDs that are both semantically expressive and computationally efficient. While recent approaches favor complex learned quantization, simple hashing-based methods such as SimHash are widely regarded as fundamentally inferior. In this work, we challenge this consensus by showing that the apparent performance gap does not stem from inherent limitations of hashing, but rather from a structural mismatch with autoregressive decoding, coupled with the inevitable information loss during rigid discretization. Based on this insight, we propose FLASH, a two-stage framework that revitalizes training-free SimHash tokenization through parallel decoding and explicit semantic alignment. Despite its simplicity, FLASH achieves state-of-the-art performance across mul
    
[^19]: WildMatch：面向野生动物重识别的弱监督图像匹配器自适应

    WildMatch: Weakly Supervised Image Matcher Adaptation for Wildlife Re-Identification

    [https://arxiv.org/abs/2610.07384](https://arxiv.org/abs/2610.07384)

    提出WildMatch，仅利用个体身份标签（无需关键点或几何对应标注）对预训练关键点匹配器进行弱监督适配，以解决相机陷阱图像中野生动物个体重识别的难题。

    

    从相机陷阱图像中进行动物个体重识别是非侵入式野生动物监测的核心实例检索问题：给定一张查询图像，需要从已知动物的参考集合中检索出正确的个体。这要求计算机视觉模型能够识别毛皮、皮肤或其他视觉标记中独特的局部图案。现有方法要么将问题当作分类任务来学习全局嵌入，这需要每个个体有大量标注图像，同时在很大程度上忽略了局部证据；要么直接采用现成的、与领域无关的通用图像匹配器。尽管这类匹配器在大型且多样化的图像集合上进行了预训练，但由于可用的野生动物数据集规模小且缺乏对应级别的标注，将其适配到野生动物图像上具有挑战性。我们研究了预训练关键点匹配器的弱监督适配，仅使用个体身份标签，而无需关键点级别或几何对应的真值标注。我们挖掘……

    arXiv:2610.07384v1 Announce Type: cross  Abstract: Individual animal re-identification from camera-trap imagery is an instance retrieval problem central to non-invasive wildlife monitoring: a query image must retrieve the correct individual from a reference set of known animals. This requires computer vision models to recognize distinctive local patterns in fur, skin, or other visual markings. Current approaches either learn global embeddings as a classification problem, requiring many labeled images per individual while largely ignoring local evidence, or apply off-the-shelf, domain-agnostic image matchers. Although such matchers are pretrained on large and diverse image collections, adapting them to wildlife imagery is challenging because available datasets are small and lack correspondence-level annotations. We study weakly supervised adaptation of a pretrained keypoint matcher using only identity labels, without keypoint-level or geometric correspondence ground truth. We mine infor
    
[^20]: 《正确的记忆，错误的语境：验证长期智能体记忆中的检索可采性》

    The Right Memory in the Wrong Context: Verifying Retrieval Admissibility in Long-Term Agent Memory

    [https://arxiv.org/abs/2610.07309](https://arxiv.org/abs/2610.07309)

    该论文提出了一个检索可采性验证框架，通过将记忆-查询对标注为“可采纳、不可采纳或未决”三种状态，并在提示词暴露层面追踪记忆ID及其与目标级泄露的关联，从而检测长期记忆智能体在错误语境下检索出“正确的记忆”这一安全隐患。

    

    长期记忆智能体可能检索到与当前请求相关但不可采纳的信息，原因在于这些信息属于其他主体、违反了政策，或反映了不兼容的生命周期状态。召回率与最终答案准确率无法揭示这一问题：一条检索路径可能因缺失必要证据而显得安全，而正确的答案也可能是在经历过不可采纳的提示词暴露之后给出的。我们提出了一个检索可采性验证框架，该框架为每个记忆-查询对赋予三种状态之一（可采纳、不可采纳或未决），在匹配的必要证据召回率下比较不同检索路径并为未决情况设定界限，并通过提示词暴露追踪记忆ID，同时将暴露情况与目标级泄露相关联。我们在相互独立、未合并的总体上评估该框架的各个阶段。对两个公开长期记忆基准 RHELM 和 MemOps 的冻结排名进行的事后 top-20 重分析共覆盖 3,767 个查询。所有已发布的锚点均落在真实……（摘要原文在此处截断）

    arXiv:2610.07309v1 Announce Type: new  Abstract: Long-term-memory agents can retrieve relevant information that is inadmissible for the current request because it belongs to another principal, violates policy, or reflects an incompatible lifecycle state. Recall and final-answer accuracy do not reveal this: a route can appear safe by missing required evidence, while a correct answer may follow inadmissible prompt exposure. We introduce a retrieval-admissibility verification framework that assigns each memory-query pair one of three statuses (admissible, inadmissible, or unresolved), compares routes at matched required-evidence recall with bounds for unresolved cases, and tracks memory IDs through prompt exposure while linking exposure to target-level disclosure. We evaluate its stages on separate, non-pooled populations. A post-hoc top-20 reanalysis of frozen rankings from two public long-term-memory benchmarks, RHELM and MemOps, covers 3,767 queries. All released anchors lie within tru
    
[^21]: RAGFlip：测量检索器升级中的查询级负向翻转

    RAGFlip: Measuring Query-Level Negative Flips in Retriever Upgrades

    [https://arxiv.org/abs/2610.07266](https://arxiv.org/abs/2610.07266)

    该论文提出RAGFlip框架来测量检索器升级中被聚合指标掩盖的查询级负向翻转，发现尽管新检索器整体表现更优，仍会丢失8.6-37.5%的BM25原本能正确检索的查询。

    

    检索器升级通常采用聚合指标进行评估，这可能会掩盖先前检索器已经能够正确处理的查询上出现的性能退化。我们将这类退化研究为“负向翻转”：即BM25能够检索到被判定为相关的段落，而替换后的检索器却无法检索到的查询。我们在三个BEIR数据集上评估了BGE-large、E5-large-v2和SPLADE：Natural Questions、HotpotQA和FiQA，涵盖了五个检索深度。所有替换检索器均提升了整体检索覆盖率。负向翻转在所有设置中都会发生，且因语料库、检索器和深度的不同而有显著差异。在k=1时，在所评估的设置中，8.6-37.5%的BM25成功案例丢失。负向翻转率在较大深度的设置中较低，其中BM25支持的查询队列在每个深度单独定义。这些比率使用any-relevant支持标签。在HotpotQA的k=10下，若要求所有正相关的qrel段落均被检索到，负向翻转率将上升至12.3-17.3%。简单的固定……

    arXiv:2610.07266v1 Announce Type: new  Abstract: Retriever upgrades are typically evaluated using aggregate metrics, which can hide regressions on queries the previous retriever already served correctly. We study these regressions as negative flips: queries for which BM25 retrieves a judged relevant passage and the replacement does not. We evaluate BGE-large, E5-large-v2, and SPLADE on three BEIR collections: Natural Questions, HotpotQA, and FiQA, across five retrieval depths. All replacements improve overall retrieval coverage. Negative flips occur in every setting and vary substantially by corpus, retriever, and depth. At k=1, 8.6-37.5% of BM25 successes are lost across the evaluated settings. Negative-flip rates are lower in the larger-depth settings, where the BM25-supported cohort is defined separately at each depth. These rates use the any-relevant support label. On HotpotQA at k=10, requiring every positive qrel passage raises the negative-flip rate to 12.3-17.3%. Simple fixed-b
    
[^22]: CroissantMiner：面向机器学习数据集的Croissant元数据自动提取与验证

    CroissantMiner: Automated Extraction and Validation of Croissant Metadata for ML Datasets

    [https://arxiv.org/abs/2610.07132](https://arxiv.org/abs/2610.07132)

    该论文提出了首个针对Croissant元数据提取的端到端评估基准（包含602篇论文的金/银双级标注），并发现单次提取方法在各类模型骨干上始终优于四种智能体架构。

    

    Croissant已成为机器可读数据集元数据的标准，然而填充其字段仍然是一项劳动密集型工作，需要仔细阅读数据集随附的文档。我们提出了首个能够对照社区标准模式进行端到端元数据提取评估的基准。该基准包含602篇论文，其中102篇带有经人工验证的金标准标注，500篇带有大语言模型生成的银标准标注，覆盖完整的Croissant模式，包括核心字段和负责任AI（RAI）字段。基于该基准，我们在一个两层评估框架下评估了一系列提取系统，涵盖前沿模型、开源权重模型和智能体架构，该框架结合了基于规则的评分与经人工审核选定的LLM裁判。我们发现，单次提取始终优于我们评估的四种智能体架构：在不同骨干模型上，这些分解式变体取得了较低的……（原文在此处截断）

    arXiv:2610.07132v1 Announce Type: cross  Abstract: Croissant has emerged as a standard for machine-readable dataset metadata, yet populating its fields remains labor-intensive and requires careful reading of accompanying dataset documentation. We present the first benchmark enabling end-to-end evaluation of metadata extraction aligned with a community-standard schema. The benchmark comprises 602 papers, including 102 with human-validated gold annotations and 500 with LLM-generated silver annotations, covering the full Croissant schema with both core and Responsible AI (RAI) fields. Using this benchmark, we evaluate a range of extraction systems spanning frontier models, open-weight models, and agentic architectures, under a two-tier evaluation framework that combines rule-based scoring with an LLM judge selected via human audit. We find that single-pass extraction consistently outperforms the four agentic architectures we evaluate: across backbones, these decomposed variants achieve lo
    
[^23]: 超越后继模型精度：推荐递归自改进中的状态保持

    Beyond Successor Accuracy: State Retention for Recursive Self-Improvement in Recommendation

    [https://arxiv.org/abs/2610.07105](https://arxiv.org/abs/2610.07105)

    该论文提出“分布式进展”概念和跨代优势（CGA）度量，揭示推荐递归自改进中更新前后的模型可能保留互补的排序决策，并利用无需标签的排序分离统计量预测应保留旧模型还是新模型，实验显示最优保留策略因推荐架构而异。

    

    推荐递归自改进将推荐器的输出反馈到后续训练中。仅通过每一轮的最新模型来评估该轮，隐含假设了后继模型能够巩固更新，但实际上更新前与更新后的模型可能保留互补的排序决策。我们将这种现象称为“分布式进展”，并使用跨代优势来量化它——这是一种在跨代模型对与代内模型对之间进行的边际匹配对比。一种排序分离统计量在选择时无需标签，能够预测应保留哪个模型家族。在四个数据集和三种序列推荐编码器上的实验表明，首选的保留机制因架构而异：跨代配对有利于GRU4Rec和SASRec，而FMLP最初倾向于代内配对，并在第二次更新后转向跨代配对。排序分离统计量在12/12的首次更新和5/6的第二次更新案例中成功选择了更强的模型家族。

    arXiv:2610.07105v1 Announce Type: cross  Abstract: Recommendation recursive self-improvement (Rec-RSI) feeds recommender outputs into subsequent training. Evaluating each round solely through its latest model assumes that the successor consolidates the update, although pre- and post-update models may retain complementary ranking decisions. We term this \emph{distributed progress} and quantify it using cross-generation advantage (CGA), a marginally matched contrast between cross- and within-generation model pairs. A rank-separation statistic, label-free at selection time, predicts which family to retain. Across four datasets and three sequential recommendation encoders, the preferred retention regime varies by architecture: cross-generation pairing benefits GRU4Rec and SASRec, whereas FMLP initially favors within-generation pairing and shifts toward cross-generation pairing after a second update. Rank separation selects the stronger family in 12/12 first-update and 5/6 second-update dat
    
[^24]: 面向生成式AI工作负载的智能内容摄取

    Smart Content Ingestion for Generative AI Workloads

    [https://arxiv.org/abs/2610.07091](https://arxiv.org/abs/2610.07091)

    本文提出智能内容摄取的理念，指出在生成式AI时代，由于企业知识以PDF、电子表格等异构格式承载多种信息模态，内容提取已演进为AI生命周期中独立且不可替代的关键阶段，其错误无法被下游检索或重排序组件修复。

    

    机器学习的演进逐步改变了智能在AI系统中的所在位置。在传统机器学习中，任务、数据表示、标签和模型架构紧密耦合，因此数据准备是狭窄的、受模式约束且过程可见的。生成式AI将模型与任何单一任务解耦：一个基础模型服务于开放式的下游任务，而模型端获得的通用性在数据端则对应着高度的异构性，因为企业知识是以人们日常使用的格式（PDF、演示文稿、电子表格、扫描文档、表单、表格、图表和混合布局文件）编写的，这些格式同时承载着文本、视觉、几何和结构信息。语言模型或检索器无法对在此接口处被错误表示的信息进行可靠推理，因此内容提取本身成为了一个独立的生命周期阶段，其产生的错误无法被任何下游检索器或重排序器所修复。

    arXiv:2610.07091v1 Announce Type: new  Abstract: The evolution of machine learning has progressively changed where intelligence resides in an AI system. In conventional machine learning the task, data representation, labels and model architecture were tightly coupled, so data preparation was narrow, schema-bound and visible. Generative AI decouples the model from any single task: one foundation model serves open-ended downstream tasks, and the generality gained on the model side is matched by heterogeneity on the data side, because enterprise knowledge is authored in the formats people use (PDF, presentations, spreadsheets, scanned documents, forms, tables, diagrams and mixed-layout files) that carry textual, visual, geometric and structural information at once. A language model or retriever cannot reason reliably over information misrepresented at this interface, so content extraction becomes a lifecycle stage in its own right whose errors no downstream retriever or re-ranker can repa
    
[^25]: 超越拒绝模式：通过安全角色内化实现鲁棒且可泛化的大语言模型安全对齐

    Beyond Refusal Patterns: Safe-Role Internalization for Robust and Generalizable LLM Safety Alignment

    [https://arxiv.org/abs/2610.07023](https://arxiv.org/abs/2610.07023)

    提出SSRFT（监督安全角色微调）框架，首次将LLM安全对齐重新表述为对预定义安全角色的内化，通过构建SRQA数据集使模型内化安全价值观与原则，从而以更少的攻击特定监督实现更鲁棒、可泛化的安全对齐，并缓解过度拒绝问题。

    

    大型语言模型（LLM）已展现出卓越的能力，但仍易受越狱攻击的影响，这类攻击会诱使其产生有害或不安全的输出。现有的安全对齐方法，包括监督微调（SFT）和基于人类反馈的强化学习（RLHF），通常需要大量针对特定攻击的监督数据和计算资源，同时仍容易陷入浅层安全对齐和过度拒绝的问题。为应对这些挑战，我们提出了SSRFT（监督安全角色微调），这是首个将安全对齐重新表述为对预定义安全角色进行内化的框架。SSRFT基于心理测量问题、少量越狱提示词以及安全角色描述构建了安全角色问答（SRQA）数据集。通过合成、验证角色一致的回复并将其扩展至多样化场景，使模型能够内化以安全为导向的价值观和原则，而非显式的……

    arXiv:2610.07023v1 Announce Type: new  Abstract: Large Language Models (LLMs) have achieved remarkable capabilities but remain vulnerable to jailbreak attacks that elicit harmful or unsafe outputs. Existing safety alignment approaches, including Supervised Fine-Tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF), often require substantial attack-specific supervision and computational resources, while remaining susceptible to shallow safety alignment and over-refusal. To address these challenges, we introduce SSRFT(Supervised Safe-Role Fine-Tuning), the first framework that reformulates safety alignment as the internalization of a predefined safe role. SSRFT constructs a Safe-Role Question-Answer (SRQA) dataset from psychometric questions, limited jailbreak prompts, and a safe-role description. Role-consistent responses are synthesized, validated, and expanded into diverse scenarios, enabling models to internalize safety-oriented values and principles rather than explicit
    
[^26]: 无需LLM摘要的树导航：面向长文档问答的分层检索等成本研究

    Tree Navigation Without LLM Summaries: A Matched-Cost Study of Hierarchical Retrieval for Long-Document QA

    [https://arxiv.org/abs/2610.06902](https://arxiv.org/abs/2610.06902)

    该论文提出NavTree，证明长文档问答中RAPTOR式摘要树的主要收益来自树的导航结构而非LLM生成的摘要内容，该方法在索引阶段零LLM调用，仅用确定性平衡线段树作为导航支架即可实现有效的分层检索。

    

    检索增强生成通过外部上下文为语言模型提供依据，但对于长文档，扁平的top-k检索可能会聚集在单一区域，从而遗漏互补证据。RAPTOR风格的摘要树通过在索引时递归聚类文本块并使用语言模型为每个聚类生成摘要，然后在查询时将摘要节点与原始文本块一起排序，来解决这一问题。我们证明，在长文档问答中，摘要树的主要收益可以来自导航本身，而非生成的摘要内容。我们提出了NavTree，一种仅使用叶子节点的检索器，它在文本块之上构建确定性的平衡线段树（索引阶段零语言模型调用），并将树纯粹用作导航支架：一种混合词法与稠密向量的前沿遍历方法，以检索得到的顶部叶子节点为锚点，从根节点向下行进，仅向阅读器输出叶子文本块。在与扁平检索器和抽取式重新实现的等成本评估中……

    arXiv:2610.06902v1 Announce Type: new  Abstract: Retrieval-augmented generation grounds language models in external context, but for long documents flat top-$k$ retrieval can cluster on a single region and miss complementary evidence. RAPTOR-style summary trees address this by recursively clustering chunks and using a language model to summarize each cluster at indexing time, then ranking summary nodes alongside raw chunks at query time. We show the main benefit of summary trees in long-document QA can come from navigation rather than the generated summary content. We introduce NavTree, a leaves-only retriever that builds a deterministic balanced segment tree over chunks (zero language-model calls at indexing) and uses the tree purely as a navigation scaffold: a hybrid lexical-and-dense frontier walk, anchored on top retrieved leaves, descends from the root and emits only leaf chunks to the reader. On a matched-cost evaluation against flat retrievers and an extractive re-implementation
    
[^27]: Diff-SQL：基于补丁生成与约束对齐的SQL效率优化

    Diff-SQL: SQL Efficiency Optimization via Patch Generation and Constraint Alignment

    [https://arxiv.org/abs/2610.06857](https://arxiv.org/abs/2610.06857)

    Diff-SQL提出了一种两阶段框架，先以统一差异补丁的形式生成针对性的SQL优化编辑，再通过在线策略强化学习在可执行性与语义等价性约束下对结果进行修正，从而避免了端到端大模型重写SQL的目标错位问题，实现可靠的SQL效率优化。

    

    SQL效率优化旨在将慢查询转换为语义等价但执行更快的查询语句。然而，使用大语言模型以端到端方式直接优化SQL往往会引发目标错位问题，这在优化与正确性之间造成了根本性的矛盾，使得直接进行完整SQL重写对于面向执行的数据库应用而言并不可靠。为解决这一问题，我们提出了Diff-SQL，这是一个将面向效率的优化与约束感知的对齐相解耦的两阶段框架。第一阶段识别优化机会，并以统一差异补丁（unified diff patch）的形式提出针对性的编辑建议；第二阶段则通过在线策略强化学习进行训练，在可执行性和语义等价性约束下对输出结果进行修正。为训练和评估Diff-SQL，我们构建了一个自动化流水线，从StackOverflow中挖掘优化知识，并构建了Slow-（摘要在此处截断）

    arXiv:2610.06857v1 Announce Type: cross  Abstract: SQL efficiency optimization aims to transform slow queries into semantically equivalent but faster alternatives. However, directly optimizing SQL with large language models in an end-to-end fashion often induces Objective Misalignment which creates a fundamental tension between optimization and correctness, making direct full SQL rewriting unreliable for execution-facing database applications. To address this problem, we propose Diff-SQL, a two-stage framework that decouples efficiency-oriented optimization from constraint-aware alignment. The first stage identifies optimization opportunities and proposes targeted edits in the form of a unified diff patch, while the second stage is trained with on-policy reinforcement learning to revise outputs under executability and semantic-equivalence constraints. To train and evaluate Diff-SQL, we construct an automated pipeline that mines optimization knowledge from StackOverflow and builds Slow-
    
[^28]: 在大语言模型时代，我们还需要地名辞典吗？将检索与空间神经符号索引相链合

    Do We Still Need Gazetteers in the Era of LLMs? Chaining Retrieval with a Spatial Neuro-Symbolic Index

    [https://arxiv.org/abs/2610.05028](https://arxiv.org/abs/2610.05028)

    本文通过空间-语义索引实验评估了文本编码器的表示能否替代地名辞典作为可靠的空间语义索引，并据此提出将检索与空间神经符号索引相链合的方案，回应了“大语言模型时代是否还需要地名辞典”这一问题。

    

    地理信息检索任务要求系统能够为下游应用解释含糊不清的地名。传统上，地名消解依赖于地名辞典来提供地点实体和空间关系的显式索引。近年来，无地名辞典的方法试图减少对手工构造搜索的依赖：稠密检索利用文本编码器捕获丰富的上下文信息，从而突破词汇检索的局限。然而，文本编码器隐含地假设其学到的表示可以充当可靠的空间-语义索引。在本文中，我们通过一个空间-语义索引实验设置来评估这一假设：给定一个带有上下文的地名提及，我们检索由来自地名辞典知识图谱的文本所表示的对应地名辞典实体。我们在两种检索策略下对五个冻结的文本编码器进行基准测试：基于实体表示的暴力最近邻检索，……（摘要内容不完整，此处截断）

    arXiv:2610.05028v2 Announce Type: replace  Abstract: Geographic information retrieval (GeoIR) tasks require systems to interpret ambiguous toponyms for downstream applications. Traditionally, toponym resolution relies on gazetteers to provide an explicit index of place entities and spatial relationships. Recently, gazetteer-free approaches seek to reduce dependence on handcrafted searches: dense retrieval utilizes text encoders to capture rich context, moving beyond the limitations of lexical search. However, text encoders implicitly assume that learned representations can function as reliable spatial-semantic indexes. In this paper, we evaluate this assumption through a spatial-semantic indexing setup: given a contextualized toponym mention, we retrieve the corresponding gazetteer entity represented by text derived from a gazetteer knowledge graph. We benchmark five frozen text encoders under two retrieval strategies: brute-force nearest-neighbor retrieval over entity representations,
    
[^29]: SOLO：基于仅扫描采样倒排列表的认证召回率度量相似性搜索

    SOLO: Certified-Recall Metric Similarity Search with Scan-Only Sampled Inverted Lists

    [https://arxiv.org/abs/2610.02387](https://arxiv.org/abs/2610.02387)

    SOLO是一种在度量空间中不依赖任何排序启发式的近似最近邻搜索索引，通过将查询路由到随机样本最近点并完整扫描倒排列表，实现了可直接从索引计算和认证的召回率，并满足召回率≈f(b·k_s)的等工作量定律。

    

    我们提出了SOLO，这是一个面向一般度量空间近似最近邻搜索的索引，其服务路径不包含任何形式的排序启发式方法：查询被路由到数据库随机样本的 $k_s$ 个最近点，被触及的倒排列表中的每个对象都用真实距离进行评估。由于不需要任何对象在排序中超越其他对象，召回率等于一个可以从存储索引直接计算出的覆盖概率：只需对查询样本进行一次真值遍历，即可一次性认证所有操作点，而无需实际服务其中任何一个——这是一种召回率认证，而对于可导航图结构，无论付出多大代价都不存在类似的对象。整个索引就是一条递归规则——对集合进行采样，将每个对象发布到其 $b$ 个最近的样本点，拆分任何超出界限的列表，始终扫描叶子节点——并且其操作面遵循一个等工作量定律：召回率 ≈ f(b·k_s)，其水平是数据集的单标量特征。

    arXiv:2610.02387v1 Announce Type: cross  Abstract: We present SOLO, an index for approximate nearest-neighbor search in general metric spaces whose serving path contains no ranking heuristic of any kind: a query is routed to the $k_s$ nearest points of a random sample of the database, and every object in the touched posting lists is evaluated with the true distance. Because nothing must outrank anything, recall equals a coverage probability computable from the stored index: one ground-truth pass over a query sample certifies every operating point at once, without serving any of them -- a recall certificate, and for a navigable graph no analogous object exists at any price. The whole index is one recursive rule -- sample the collection, post each object to its $b$ nearest sample points, split any list that outgrows a bound, always scan the leaves -- and its operating surface obeys an equal-work law, recall $\approx f(b \cdot k_s)$, whose level is a one-scalar signature of the dataset. T
    
[^30]: AX 是新的 AEO

    AX is the New AEO

    [https://arxiv.org/abs/2609.34951](https://arxiv.org/abs/2609.34951)

    该论文提出“代理体验（AX）”——即AI智能体能否顺利抓取并阅读企业自身网站——正在取代答案引擎优化（AEO）成为决定AI推荐结果的关键因素，并通过超过3.7万次智能体买家旅程实验加以验证。

    

    2023年，AI模型依靠训练数据回答问题，数据耗尽时便产生幻觉，因此企业被告知要预先植入相关知识。此后，模型的训练知识已让位于实时网络搜索，相关建议也随之转移：答案引擎优化（AEO）如今建议企业在论坛帖子、清单文章和站外引用中散布“面包屑”信息，以便AI引擎更容易将它们展示和推荐出来。但仅仅被展示出来已经不够了：智能体会打开搜索结果、阅读内容后才做决策，一个买家问题会驱使它经历多轮搜索与抓取。在这一深入挖掘环节中，决定成败的是智能体能否成功抓取并阅读企业自己的网站——这就是代理体验（AX）。我们主张，AX 是新的 AEO。我们在四个独立测试框架上，针对1,056家真实企业开展了37,927次智能体旅程实验，每次旅程都是一个关于某企业的买家问题，并在知名度、模型先验知识以及（原文在此截断）……等方面进行了匹配。

    arXiv:2609.34951v2 Announce Type: replace  Abstract: In 2023, AI models answered from training data and hallucinated when it ran out, and businesses were told to seed that knowledge. Models' training knowledge has since given way to live web search, and the advice followed it there: answer-engine optimization, or AEO, now tells businesses to scatter breadcrumbs across forum threads, listicles, and off-site citations, so AI engines are likelier to surface and recommend them. But being surfaced is no longer enough: an agent opens the results and reads them before deciding, and one buyer question sends it through several rounds of search and fetch. What decides the outcome at this drill-down step is whether the agent can fetch and read the business's own site: agent experience (AX). We argue that AX is the new AEO. We run 37,927 agent journeys, each a buyer question about a business, across four independent harnesses over 1,056 real businesses, matched on fame, prior model knowledge, and 
    
[^31]: 水域环境监测中信息路径规划的校准不确定性

    Calibrated Uncertainty for Informative Path Planning in Aquatic Environmental Monitoring

    [https://arxiv.org/abs/2609.34577](https://arxiv.org/abs/2609.34577)

    用校准良好的深度集成替代高斯过程可为水域环境监测的信息路径规划提供更可靠的不确定性估计，在原油泄漏场景模拟中将归一化重构误差降低83%。

    

    arXiv:2609.34577v2 公告类型：替换 摘要：标量场重构的信息路径规划利用预测不确定性来引导感知载体前往信息量最大的位置。高斯过程（Gaussian Process）能够提供这一信号，但其平稳各向同性核对于诸如原油泄漏等非均质现象存在模型设定偏差，产生校准不良的估计，从而降低规划效果。我们研究了用校准良好的深度集成（Deep Ensemble）替代高斯过程是否能改善路径规划结果，以及不确定性质量是否与规划算法的选择存在交互作用。五种策略（$\epsilon$-贪心、价值贪心、不确定性贪心、蒙特卡洛树搜索和滚动时域定向）共享同一个基于物理原油泄漏模拟训练的深度集成主干。在留出的随机泄漏场景上，深度集成相对于高斯过程基线将归一化重构误差降低了83%。至关重要的是，校准良好的（摘要在此处截断）

    arXiv:2609.34577v2 Announce Type: replace  Abstract: Informative Path Planning for scalar field reconstruction uses predictive uncertainty to direct sensing vehicles toward maximally informative locations. Gaussian Processes provide this signal but their stationary isotropic kernels are misspecified for non-homogeneous phenomena such as oil spills, producing miscalibrated estimates that degrade planning. We investigate whether replacing the Gaussian Process with a well-calibrated Deep Ensemble improves path planning outcomes, and whether uncertainty quality interacts with the choice of planning algorithm. Five strategies ($\epsilon$-Greedy, Value Greedy, Uncertainty Greedy, Monte Carlo Tree Search, and Receding Horizon Orienteering) share a common Deep Ensemble backbone trained on physics-based oil spill simulations. On held-out stochastic spill scenarios, the Deep Ensemble reduces normalised reconstruction error by $83\%$ relative to the Gaussian Process baseline. Crucially, well-cali
    
[^32]: 基于全池、集合式、长上下文语言模型的更高效LLM重排序

    More Efficient LLM Reranking with Whole-Pool, Setwise, Long-Context Language Models

    [https://arxiv.org/abs/2606.01782](https://arxiv.org/abs/2606.01782)

    提出全池集合式重排序方法DualEnd，通过从两端同时填充排序，仅需50次LLM比较即可完成100个候选的完整排序，比现有方法减少59.4%-88.8%的比较次数。

    

    基于LLM的重排序器通常通过重复的局部比较（列表式、成对式或逐点式）来产生排序，需要多次顺序模型调用。我们研究了当整个检索到的候选池能够放入上下文窗口时，长上下文LLM如何大幅减少这种计算量。我们引入了全池集合式重排序，其中每次比较都对整个候选池进行排序，并提出DualEnd方法，该方法联合选择预测为最相关和最不相关的候选。通过从两端填充排序，DualEnd仅需50次LLM比较即可构建100个候选的完整排序。在TREC DL19和DL20数据集上使用九个开源权重LLM的实验表明，与之前面向顶部的使用堆排序的窗口式集合排序相比，该方法减少了59.4%的比较次数；与使用冒泡排序的面向顶部的窗口式集合排序相比，减少了88.8%的比较次数——尽管这些基线方法仅针对前10名排序，而DualEnd针对的是完整排序。

    arXiv:2606.01782v2 Announce Type: replace  Abstract: LLM-based re-rankers produce a rankings through repeated local comparisons (listwise, pairwise or pointwise), requiring many sequential model calls. We study how long-context LLMs can drastically reduce this computation when the entire retrieved candidate pool fits within the context window. We introduce Whole-Pool Setwise re-ranking, where each comparison ranks all the entire candidate pool, and propose DualEnd, which jointly selects the candidates predicted to be most and least relevant. By filling the ranking from both ends, DualEnd constructs a complete ranking of 100 candidates in 50 LLM comparisons. Experiments with nine open-weight LLMs on TREC DL19 and DL20 show that this requires 59.4% fewer comparisons than previous top-oriented windowed Setwise with heapsort and 88.8% fewer than top-oriented windowed Setwise with bubblesort, even though those baselines target only the top-10 rankings while DualEnd targets the full ranking.
    
[^33]: KadiAssistant：一个用于Kadi4Mat信息检索的对话式AI智能体

    KadiAssistant: A conversational AI Agent for information retrieval in Kadi4Mat

    [https://arxiv.org/abs/2605.18850](https://arxiv.org/abs/2605.18850)

    本文提出了KadiAssistant，一个集成于Kadi研究数据生态系统中、以隐私保护为设计核心的对话式AI智能体，使研究人员能够高效访问、聚合和综合异构且隐私敏感的研究数据信息。

    

    我们介绍了KadiAssistant，一个采用隐私设计理念并集成于Kadi研究数据生态系统中的AI助手，使研究人员能够高效地访问、聚合和综合来自异构且涉及隐私敏感的研究数据的信息。材料科学等跨学科领域汇集了各自拥有不同术语和标准的学科。虽然这种融合推动了创新，但也使得知识的连接和获取日益困难，因为数据分散在各个学科、组织和个人之间。例如，电池研究结合了电化学测量、材料表征数据、基于物理的模拟以及制造参数，而每个部分都使用不同的格式、词汇和标准。通过Kadi4Mat等研究数据平台高效地存储和共享此类异构数据，需要领域知识、技术专长以及对相关工具的熟悉（摘要原文在此处被截断）。

    arXiv:2605.18850v2 Announce Type: replace-cross  Abstract: We introduce KadiAssistant, a privacy-by-design AI assistant integrated into the Kadi research data ecosystem, enabling researchers to efficiently access, aggregate, and synthesize information from heterogeneous, privacy-sensitive research data. Interdisciplinary fields such as materials science bring together disciplines with their own terminology and standards. While this convergence fuels innovation, it also makes it increasingly difficult to connect and access knowledge, as data are distributed across disciplines, organizations, and individuals. For example, battery research combines electrochemical measurements, materials characterization data, physics-based simulations, and manufacturing parameters, each using different formats, vocabularies, and standards. Efficiently storing and sharing such heterogeneous data via research data platforms, such as Kadi4Mat, demands domain knowledge, technical expertise, and familiarity w
    
[^34]: 增强基于大语言模型的推荐模型中的高阶交互感知能力

    Enhancing High-order Interaction Awareness in LLM-based Recommender Model

    [https://arxiv.org/abs/2409.19979](https://arxiv.org/abs/2409.19979)

    本文提出增强型LLM推荐器ELMRec，通过增强全词嵌入使大语言模型无需图预训练即可理解用户-项目高阶交互，并针对LLM偏好早期交互而忽略近期交互的问题提出重排序方案，在直接推荐和序列推荐中均超越现有最先进方法。

    

    大语言模型（LLM）通过将推荐任务转化为文本生成任务，已在推荐任务中展现出卓越的推理能力。然而，现有方法要么忽略了用户-项目的高阶交互，要么对其建模效果不佳。为此，本文提出了一种增强型的大语言模型推荐器（ELMRec）。我们增强了全词嵌入，从而大幅提升大语言模型对图构建的交互信息的理解能力，且无需图预训练。这一发现可能启发人们通过全词嵌入将丰富的知识图谱融入基于大语言模型的推荐器中。我们还发现，大语言模型在推荐项目时往往基于用户的早期交互而非近期交互，并据此提出了一种重排序解决方案。我们的ELMRec在直接推荐和序列推荐中均优于最先进（SOTA）的方法。

    arXiv:2409.19979v4 Announce Type: replace-cross  Abstract: Large language models (LLMs) have demonstrated prominent reasoning capabilities in recommendation tasks by transforming them into text-generation tasks. However, existing approaches either disregard or ineffectively model the user-item high-order interactions. To this end, this paper presents an enhanced LLM-based recommender (ELMRec). We enhance whole-word embeddings to substantially enhance LLMs' interpretation of graph-constructed interactions for recommendations, without requiring graph pre-training. This finding may inspire endeavors to incorporate rich knowledge graphs into LLM-based recommenders via whole-word embedding. We also found that LLMs often recommend items based on users' earlier interactions rather than recent ones, and present a reranking solution. Our ELMRec outperforms state-of-the-art (SOTA) methods in both direct and sequential recommendations.
    
[^35]: 用于捆绑包推荐的超图增强双卷积网络

    Hypergraph-Enhanced Dual Convolutional Network for Bundle Recommendation

    [https://arxiv.org/abs/2312.11018](https://arxiv.org/abs/2312.11018)

    HED 通过构建包含用户-捆绑包、用户-物品、捆绑包-物品交互及用户内部、捆绑包内部关系的完整超图，并将全超图传播与用户-捆绑包双卷积分支耦合，在保留推荐特有信号的同时引入物品感知的高阶上下文，在 NetEase 和 Youshu 数据集上较最强基线显著提升了捆绑包推荐性能。

    

    arXiv:2312.11018v3 公告类型：replace-cross 摘要：捆绑包推荐是对相关物品的集合而非孤立物品进行排序。其核心挑战在于在不丢失捆绑包排序所需信号的前提下，连接用户偏好、物品交互与捆绑包构成。我们提出了超图增强双卷积神经网络（HED），该网络构建了一个完整的超图，其中包含用户-捆绑包、用户-物品和捆绑包-物品交互，以及用户内部和捆绑包内部关系。HED 将完整超图传播与用户-捆绑包分支相结合，使物品感知的高阶上下文能够为排序提供信息，同时保留推荐特有的信号。在 NetEase 数据集上，HED-128 在六项报告指标上较最强基线提升了 5.04%–6.97%；在 Youshu 数据集上，HED-64 提升了 1.87%–4.56%。消融实验结果支持了用户-捆绑包分支和类型内关系二者的贡献，敏感性分析确定了稳定的运行范围。

    arXiv:2312.11018v3 Announce Type: replace-cross  Abstract: Bundle recommendation ranks sets of related items rather than isolated items. Its central challenge is to connect user preferences, item interactions, and bundle composition without losing the signals needed to rank bundles. We propose Hypergraph-Enhanced Dual Convolutional Neural Network (HED), which constructs a complete hypergraph containing user--bundle, user--item, and bundle--item interactions together with intra-user and intra-bundle relations. HED couples complete-hypergraph propagation with a user--bundle branch, allowing item-aware higher-order context to inform ranking while preserving recommendation-specific signals. On NetEase, HED-128 improves over the strongest baseline by 5.04--6.97% across the six reported metrics; on Youshu, HED-64 improves by 1.87--4.56%. Ablation results support the contributions of both the user--bundle branch and intra-type relations, and sensitivity analyses identify stable operating rang
    

