# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Balancing hierarchical representation and quantization: A performance trade-off in text embeddings](https://arxiv.org/abs/2611.06859) | 该研究通过每任务168种配置的大规模基准实验，首次系统评估了层次化嵌入表示与量化的交互作用，发现8-bit量化几乎不损害文本嵌入性能（最多损失1.5个百分点），而二值量化在所有任务类别中均显著退化，为文本嵌入的效率优化提供了实证权衡依据。 |
| [^2] | [WISE: An Agentic Framework for Whole-page Multimodal Search Experience Attribution](https://arxiv.org/abs/2611.06777) | 该论文提出了系统质量诊断（SQD）这一新任务，并构建了WISE智能体框架，能够对工业搜索服务进行端到端的质量诊断，发现面向用户的问题并将其归因到底层算法组件。 |
| [^3] | [Alternatives to Shallow Autoencoders for Collaborative Filtering](https://arxiv.org/abs/2611.06610) | 该论文针对EASE模型仅依赖物品-物品关系而限制细粒度用户关系捕捉的问题，提出了利用用户-用户信息的替代模型，以及结合用户-用户和物品-物品邻接分布的混合模型，从而提升协同过滤的推荐准确性。 |
| [^4] | [Deliberate Latent Reasoning for Sequential Recommendation](https://arxiv.org/abs/2611.06521) | 本文提出DESRec框架，通过为不同用户历史个性化选择潜在推理的深度，克服了现有方法固定推理步数导致的“简单历史过度思考、复杂历史思考不足”问题，从而学习更具信息量的推理状态并提升序列推荐的用户建模效果。 |
| [^5] | [Signed but Steerable: Measuring and Closing the Text-Vector Integrity Gap in Offline Retrieval-Augmented Generation](https://arxiv.org/abs/2611.06498) | 本文揭示了离线RAG签名知识包中“签名只证明发布者身份、无法保证向量与文本一致”这一安全鸿沟，提出了量化攻击者可劫持查询比例的“边际定律”，并给出了测量与弥合该完整性差距的验证方案。 |
| [^6] | [ArcRAG: Agentic Compositional Retrieval over Knowledge Graphs](https://arxiv.org/abs/2611.06481) | ArcRAG通过在推理时归纳查询局部的图谱接口并规划可修复的多跳遍历，实现了无需任何构建特定调整即可跨不同图谱构建方式保持有效的智能体式组合检索。 |
| [^7] | [PMRecBench: A Benchmark for Personalization in Recommendation Agents under Multimodal Queries](https://arxiv.org/abs/2611.06349) | 提出了首个评估多模态查询下推荐智能体个性化能力的基准PMRecBench，通过将510个来源于真实用户行为模式的图文查询与用户真实历史配对，填补了多模态推荐评估中缺失个性化维度的空白。 |
| [^8] | [Generative Retrieval under Real-Time Supply Constraints in Local Commerce](https://arxiv.org/abs/2611.06323) | 该论文提出RTP-LLM流水线，通过将合格的残差量化语义标识符预先编译为压缩稀疏行前缀索引、与模型预填充并行执行CPU编译并在beam间共享，使生成式检索能在解码过程中直接满足本地商业的实时供给约束，避免对不可行候选浪费beam容量和推理计算。 |
| [^9] | [Do Search-Augmented Recommenders Use the Latest Search Episode?](https://arxiv.org/abs/2611.06044) | 提出FocalSwap审计框架，通过在固定目标与候选集的条件下替换最新搜索或全部搜索输入，检验搜索增强推荐系统对最新搜索会话的实际依赖程度，发现其估计收益随对照方式的选择而变化。 |
| [^10] | [From Waste to Discovery: Recycling High-Scored Unexposed Candidates in Taobao Feed Recommendation](https://arxiv.org/abs/2611.05997) | 论文提出无需训练的召回阶段系统URED，通过回收工业推荐流水线中通常被丢弃的高分未曝光候选物品并在召回阶段重新注入，在淘宝信息流推荐中高效帮助用户发现新兴趣，避免了传统曝光提升策略因冷启动候选被排序器高估而损害参与度的问题。 |
| [^11] | [One Basket, Many Experts: Decorrelated Re-ranking for Financial Asset Recommendation](https://arxiv.org/abs/2611.05952) | 提出DSER重排序方法，通过用收益去相关的替代资产替换专家推荐列表中的部分内容，并借助单一可调参数m，在用户相关性与投资组合风险调整后收益之间实现灵活权衡。 |
| [^12] | [Optimal Re-Ranking Depth](https://arxiv.org/abs/2611.05908) | 该论文利用基于大语言模型的相关性判断，首次在逐查询层面揭示了许多查询存在明确的最优重排序深度（超过该深度检索效果会停滞或下降），并将重排序深度定义为查询特有属性，研究其能否被预先预测。 |
| [^13] | [Weaver: Weaving Preference Evidence through Incremental Reasoning for Personalized Retrieval-Augmented Generation](https://arxiv.org/abs/2611.05900) | Weaver 框架通过耦合意图依赖图与记忆条件检索轨迹图的双层检索与推理图，以增量方式将记忆条件检索轨迹与组合式偏好推理相编织，解决了个性化检索增强生成中偏好证据分散且跨情境易混淆的问题。 |
| [^14] | [SciToT: Multimodal Tip-of-the-Tongue Retrieval for Scientific Papers](https://arxiv.org/abs/2611.05829) | 该论文提出了首个面向科学论文的多模态“话到嘴边”（ToT）检索基准SciToT，通过从文本和视觉元素中提取记忆线索并用大语言模型合成模拟人类不完整记忆的查询，填补了科学文献重新查找任务中缺乏评测基准的空白。 |
| [^15] | [Item Representations Matter: Bridging Hard Semantic IDs and Soft Prototype Mixtures for Generative Retrieval](https://arxiv.org/abs/2611.05799) | 提出CAP-Mix流式推荐器，将每个物品表示为共享原型的软混合加上物品专属向量，在不引入离散瓶颈的前提下结合了硬语义ID的共享结构与连续表示的无损优势。 |
| [^16] | [Toward a Compositional Knowledge Architecture for a Shared Organizational Cortex](https://arxiv.org/abs/2611.05753) | 提出由信号、领域、主张、维度四层组成的组合式知识架构，使组织解释连同其依据与立场持久保存，并通过真值维护网络随证据变化动态修订，从而显著提升检索性能。 |
| [^17] | [Turbocharging PiPNN for Proximity and $k$-NN Graph Building](https://arxiv.org/abs/2611.05690) | 本文提出优化版PiPNN II及其GPU实现GPiPNN，将基于图的近似最近邻索引构建速度最高提升8倍（CPU），并比CAGRA快最高50倍，还可直接高效构建高质量k-NN图。 |
| [^18] | [Combining Relevant Expert Knowledge Using Mixtures of Feedback Fuzzy Cognitive Maps](https://arxiv.org/abs/2611.05664) | 该论文将法律证据理论中的相关性定义扩展应用于神经符号系统，通过将专家的反馈模糊认知图混合为共同知识网络，并借助新的类贝叶斯定理对混合知识进行解混合，从而仅组合相关专家答案以提升系统的准确性和可扩展性。 |
| [^19] | [Data-Efficient Scaling-Law Discovery for Recommendation Systems via Coupled Hyperparameter Scaling](https://arxiv.org/abs/2611.05643) | 该论文提出通过缩放层将低维超超参数映射到完整架构空间，并结合缩放感知代理模型与帕累托前沿选择，仅需少量实验即可高效发现任意推荐排序模型的缩放定律。 |
| [^20] | [RAGCompass: Generalizable Retriever Routing via Reranking](https://arxiv.org/abs/2611.05596) | RAGCompass 通过对比微调重排序器直接读取不同检索器返回的段落集合来实现查询路由，从而可泛化到未见过的检索器和新语料库，将答案 F1 相对提升 28%。 |
| [^21] | [LIBRA: Programmatic Induction of Executable Tool Libraries](https://arxiv.org/abs/2611.05445) | LIBRA 通过让大语言模型提出工具间契约并在沙盒中进行运行时验证，将扁平的工具目录自动归纳为可执行的工具库，从而在多工具任务中检索出包含先决工具的完整工作流，无需依赖标注或任务特定训练，在多个基准上显著提升了检索性能。 |
| [^22] | [Factorized MLLM Embeddings for Composed Image Retrieval](https://arxiv.org/abs/2611.05193) | 提出FactEmbed方法，通过因子化潜在变量和中间目标想象目标将组合查询理解与判别性嵌入构建解耦，从而更充分地利用MLLM的组合理解能力以提升组合图像检索性能。 |
| [^23] | [TAGR: Temporally Adaptive Generative Recommendation for Industrial Live-Streaming Advertising](https://arxiv.org/abs/2608.24034) | TAGR通过三层时间自适应机制（令牌级动态ID刷新、意图级多尺度建模、偏好级在线稳定对齐）解决了直播广告推荐中内容快速变化导致的时效性问题。 |
| [^24] | [Trustworthy RAG: An Evaluation Agent for Detecting Misinformation and Knowledge Poisoning in Generative AI Systems](https://arxiv.org/abs/2608.21095) | 本文提出了一种结合NLI事实核查与五信号投毒检测的评估代理，并引入信任指数，在TruthfulQA上实现了高准确率与精确率，有效缓解了RAG系统中的知识投毒风险。 |
| [^25] | [When Is Complex Chunking Worth It? A Multi-Objective Evaluation of Chunking Methods at Scale](https://arxiv.org/abs/2608.16586) | 本文通过多目标评估发现，复杂分块方法在检索效果和系统成本上并不总是优于简单方法，最佳策略依赖于具体应用场景。 |
| [^26] | [Safety, or Just Capability? A Validity Audit of Agent-Safety Benchmarks](https://arxiv.org/abs/2607.28685) | 本文对四个主流智能体安全基准进行效度审计，发现其评估指标存在系统性缺陷（如恒判为正的策略即可获得高分）、不同基准对同一批模型的排名互相矛盾，且安全基准之间或安全与能力之间表观的负相关很可能只是小样本导致的统计伪象。 |
| [^27] | [IMFuse: Instance-Aware Multi-Layer Fusion for LLM-Enhanced Sequential Recommendation](https://arxiv.org/abs/2607.27002) | 提出IMFuse实例感知多层融合策略，自适应地为每个物品融合LLM多个层的嵌入语义，克服了仅依赖最后一层嵌入存在谱坍缩、信息不足的问题，从而增强序列推荐性能。 |
| [^28] | [Driving Video Retrieval for Complex Queries with Structured Grounding](https://arxiv.org/abs/2606.09109) | 提出STRIVE-D数据校准检索框架，通过弱标注视频评估并自适应调整查询规则，将校准后的规则分数与视觉-语言及关键词检索信号融合，从而在驾驶视频中可靠检索加塞、急刹等复杂动态事件。 |
| [^29] | [Algorithmic Authority and the Clinical Standard of Care](https://arxiv.org/abs/2606.00044) | 本文借助“代码即法律”框架论证临床AI系统已成为事实上的医疗监管，并提出将AI与医生视为单一责任诊断实体的辩证诊疗标准，从而对AI幻觉与人类认知失误这两类失败模式进行统一治理。 |
| [^30] | [Space efficient implementation of hypergraph dualization in the D-basis algorithm](https://arxiv.org/abs/2512.06988) | 提出了一种名为 Small Space 的 D-基算法新实现，借助反向搜索的超图对偶化方法在不存储全部蕴涵关系的情况下逐步计算属性频率，大幅降低了数据分析中的内存占用。 |
| [^31] | [When Reviews Are Scarce: Structure-Aware Review Imputation for Recommendation in Realistic Sparse Review Settings](https://arxiv.org/abs/2508.01128) | 提出TWISTER统一框架，通过在文本边图上联合建模语义与结构信号来合成缺失评论，从而在用户很少写评论的现实稀疏场景下提升评论感知推荐系统的性能。 |

# 详细

[^1]: 平衡层次化表示与量化：文本嵌入中的性能权衡

    Balancing hierarchical representation and quantization: A performance trade-off in text embeddings

    [https://arxiv.org/abs/2611.06859](https://arxiv.org/abs/2611.06859)

    该研究通过每任务168种配置的大规模基准实验，首次系统评估了层次化嵌入表示与量化的交互作用，发现8-bit量化几乎不损害文本嵌入性能（最多损失1.5个百分点），而二值量化在所有任务类别中均显著退化，为文本嵌入的效率优化提供了实证权衡依据。

    

    arXiv:2611.06859v1 公告类型： new 摘要： 层次化文本嵌入模型提供可变长度的表示，而量化则降低数值精度；两者都旨在提高计算效率。本工作是一项实证基准研究，探讨这两种相互正交且已被广泛确立的技术之间的交互作用。我们使用四个基于Transformer的嵌入模型，在大规模文本嵌入基准（MTEB）的38个任务上进行评估，涵盖一维和二维层次化表示，并结合32位、8位和二值量化，每个任务共168种配置。我们分别报告每个效率维度：每个向量的理论存储开销和端到端评估时间。在Friedman检验及Nemenyi事后分析下，int8量化在配对分类和重排序任务中与float32在统计上无显著差异，总体性能损失最多为1.5个百分点；而二值量化在所有任务类别中均显著更差。将嵌入（原文在此处截断）……

    arXiv:2611.06859v1 Announce Type: new  Abstract: Hierarchical text embedding models offer variable-length representations, while quantization reduces numerical precision; both aim to improve computational efficiency. This work is an empirical benchmark study of how these two orthogonal, already-established techniques interact. We evaluate 38 tasks from the Massive Text Embedding Benchmark (MTEB) using four transformer-based embedding models, covering 1D and 2D hierarchical representations combined with 32-bit, 8-bit, and binary quantization, for 168 configurations per task. We report each efficiency dimension separately: theoretical per-vector storage and end-to-end evaluation time. Under a Friedman test with Nemenyi post-hoc analysis, int8 quantization is statistically indistinguishable from float32 for pair classification and re-ranking and costs at most 1.5 percentage points overall, whereas binary quantization is significantly worse across every task category. Halving the embedding
    
[^2]: WISE：面向整页多模态搜索体验归因的智能体框架

    WISE: An Agentic Framework for Whole-page Multimodal Search Experience Attribution

    [https://arxiv.org/abs/2611.06777](https://arxiv.org/abs/2611.06777)

    该论文提出了系统质量诊断（SQD）这一新任务，并构建了WISE智能体框架，能够对工业搜索服务进行端到端的质量诊断，发现面向用户的问题并将其归因到底层算法组件。

    

    算法驱动的服务，包括搜索、推荐和广告，依赖于多阶段模型流水线。虽然研究界优先关注模型优化，而部署后的监控则聚焦于运营可靠性，但运营上健康的系统仍可能产生低质量的结果，这暴露了现有评估与监控实践的空白。然而，针对这一任务的系统性问题定义和基准仍然缺失。为了弥补这一空白，我们为算法驱动的服务提出了系统质量诊断（SQD）这一新任务，它将质量评估从仅仅识别低质量结果扩展到在系统各组件中归因其成因。聚焦于工业搜索场景，我们提出了WISE——一个用于端到端质量诊断的智能体框架，它能够发现面向用户的问题，并将其根源追溯到底层的算法组件。为了提高诊断的可靠性，我们引入了ReTrace用于构建证据（摘要在此处截断）。

    arXiv:2611.06777v1 Announce Type: new  Abstract: Algorithm-driven services, including search, recommendation, and advertising, rely on multi-stage model pipelines. While research prioritizes model optimization, post-deployment monitoring focuses on operational reliability. Yet operationally healthy systems may still produce poor-quality results, exposing a gap in existing evaluation and monitoring practices. However, a systematic problem formulation and benchmark for this task remain lacking. To bridge this gap, we formulate System Quality Diagnosis (SQD) for algorithm-driven services, a new task that extends quality evaluation beyond identifying poor outcomes to attributing their causes across system components. Focusing on industrial search, we propose WISE, an agentic framework for end-to-end quality diagnosis that discovers user-facing issues and traces their origins to underlying algorithmic components. To improve diagnostic reliability, we introduce ReTrace for constructing evide
    
[^3]: 协同过滤中浅层自编码器的替代方案

    Alternatives to Shallow Autoencoders for Collaborative Filtering

    [https://arxiv.org/abs/2611.06610](https://arxiv.org/abs/2611.06610)

    该论文针对EASE模型仅依赖物品-物品关系而限制细粒度用户关系捕捉的问题，提出了利用用户-用户信息的替代模型，以及结合用户-用户和物品-物品邻接分布的混合模型，从而提升协同过滤的推荐准确性。

    

    协同过滤（CF）是推荐系统的基石，在许多现代应用中发挥着重要作用。CF利用用户-物品交互数据来发现用户的未来偏好。尽管深度学习模型在协同过滤中展现出了潜力，但面向稀疏数据的极简浅层自编码器（EASE）因其闭式解所带来的出色排序准确性而备受关注。EASE主要依靠物品之间的关系来拟合一个满秩的高维线性映射。我们假设这种设计限制了其捕捉相似但不完全等价的细粒度用户关系的能力，从而限制了其推荐准确性。本文提出了一种基于EASE的替代公式，该公式利用了用户-用户之间的信息。此外，我们提出了一个同时结合用户-用户和物品-物品邻接分布的混合模型。我们的实验表明，所提出的模型表现更优。

    arXiv:2611.06610v1 Announce Type: new  Abstract: Collaborative filtering (CF) is a cornerstone of recommender systems and plays a relevant role in many modern applications. CF uses user-item interaction data to discover future preferences. Although deep learning models have shown promise in CF, Embarrassingly Shallow Autoencoders for Sparse Data (EASE) has gained attention for its outstanding ranking accuracy provided by its closed-form solution. EASE relies primarily on relationships among items to fit a full-rank high-dimensional linear mapping. We hypothesize that this design limits its capacity to capture similar but not equivalent fine-grained user relationships, consequently limiting its recommendation accuracy. This paper introduces an alternative formulation based on EASE that takes advantage of user-user information. Furthermore, we propose a hybrid model that combines both user-user and item-item adjacency distributions.   Our experiments reveal that the proposed models outpe
    
[^4]: 面向序列推荐的深思式潜在推理

    Deliberate Latent Reasoning for Sequential Recommendation

    [https://arxiv.org/abs/2611.06521](https://arxiv.org/abs/2611.06521)

    本文提出DESRec框架，通过为不同用户历史个性化选择潜在推理的深度，克服了现有方法固定推理步数导致的“简单历史过度思考、复杂历史思考不足”问题，从而学习更具信息量的推理状态并提升序列推荐的用户建模效果。

    

    序列推荐器广泛应用于现实场景中，基于用户交互来建模用户偏好和行为模式。序列推荐的最新进展通过潜在推理来增强基于Transformer的模型，通过自回归生成中间推理向量来优化用户表示。然而，为多样化的用户历史探索最优推理深度仍然具有挑战性：现有方法通常对所有用户采用固定数量的推理步骤。这种“一刀切”策略常常导致次优的表示学习，因为在简单历史上过度思考或在复杂历史上思考不足。为解决这一问题，我们提出了DESRec，一个深思式潜在推理框架，它执行个性化的潜在推理以学习更具信息量的推理状态，并增强序列推荐中的用户建模。具体而言，它设计了一个推理步骤选择（原文摘要在此处截断）。

    arXiv:2611.06521v1 Announce Type: new  Abstract: Sequential recommenders are widely applied in real-world scenarios to model user preferences and behavior patterns based on user interactions. Recent advances in sequential recommendation enhance transformer-based models with latent reasoning to refine user representations by autoregressively generating intermediate reasoning vectors. However, exploring the optimal reasoning depth for diverse user histories remains challenging: existing methods typically employ a fixed number of reasoning steps for all users. This ``one-size-fits-all'' strategy often leads to suboptimal representation learning due to over-thinking in simple histories or under-thinking in complex ones. To address this issue, we propose DESRec, a deliberate latent reasoning framework that performs personalized latent reasoning to learn more informative reasoning states and enhance user modeling for sequential recommendation. Specifically, it designs a reasoning step select
    
[^5]: 已签名却仍可被操纵：测量并弥合离线检索增强生成中的文本-向量完整性鸿沟

    Signed but Steerable: Measuring and Closing the Text-Vector Integrity Gap in Offline Retrieval-Augmented Generation

    [https://arxiv.org/abs/2611.06498](https://arxiv.org/abs/2611.06498)

    本文揭示了离线RAG签名知识包中“签名只证明发布者身份、无法保证向量与文本一致”这一安全鸿沟，提出了量化攻击者可劫持查询比例的“边际定律”，并给出了测量与弥合该完整性差距的验证方案。

    

    在边缘设备上运行的离线检索增强生成（RAG）通过携带文本和预计算嵌入向量的签名知识包进行更新。签名只能证明知识包由谁发布，却不能证明其中的向量确实编码了对应的文本。我们研究了这种“文本-向量完整性鸿沟”。一个会重新计算向量并在余弦容差范围内接受它们的验证器，会将可预测比例的查询拱手让给控制向量的攻击者：劫持率等于该容差处角度检索边际的分布（即“边际定律”）。在五个语料库和两个嵌入模型上，检索边际都很小，因此处于浮点噪声级别的理想化容差会使最多5.5%的查询暴露于攻击之下（排除完全重复文档后，任何语料库中最多为1.2%），而根据我们对小型嵌入模型的CPU INT8-FP32对比校准的阈值（取诚实余弦的经验0.1%分位数处）则会使98-100%的查询暴露于攻击之下；在树莓派上……（原文摘要在此处被截断）

    arXiv:2611.06498v1 Announce Type: cross  Abstract: Offline retrieval-augmented generation (RAG) on edge devices is updated through signed knowledge packs that carry text and precomputed embedding vectors. A signature proves who published a pack, not that its vectors encode its text. We study this text-vector integrity gap. A verifier that recomputes vectors and accepts them within a cosine tolerance concedes a predictable share of queries to an attacker who controls the vectors: the hijack rate equals the distribution of angular retrieval margins at the tolerance (the margin law). Margins are small in five corpora and two embedding models, so an idealised tolerance at the level of floating-point noise leaves up to 5.5% of queries exposed (at most 1.2% in any corpus once exact duplicate documents are excluded), and a threshold calibrated from our CPU INT8-FP32 comparison of a small embedding model (at the empirical 0.1% quantile of honest cosines) leaves 98-100% exposed; on a Raspberry 
    
[^6]: ArcRAG：基于知识图谱的智能体式组合检索

    ArcRAG: Agentic Compositional Retrieval over Knowledge Graphs

    [https://arxiv.org/abs/2611.06481](https://arxiv.org/abs/2611.06481)

    ArcRAG通过在推理时归纳查询局部的图谱接口并规划可修复的多跳遍历，实现了无需任何构建特定调整即可跨不同图谱构建方式保持有效的智能体式组合检索。

    

    基于图的检索增强生成（GraphRAG）通过在从语料库构建的知识图谱上进行检索来回答问题。然而，检索策略通常围绕其自身构建流程所产生的结构来设计，这些策略在构建方式改变后是否依然有效却很少被测试。我们研究了这种跨构建的可移植性：即一个不依赖特定构建方式的模式、关系名称或重新调优的检索流程，在图谱以不同方式构建时是否仍能保持有效。我们提出了ArcRAG，它在推理时归纳出一个查询局部的图谱接口（即从问题所命名的实体出发可达的节点标签、关系描述符和属性键），据此规划多跳遍历，并以Cypher语言执行和修复这些遍历。我们将MuSiQue、HotpotQA和TriviaQA语料库分别构建为模式引导的图谱（类型化节点、封闭关系集合、来源溯源）和开放信息抽取图谱（摘要在此处被截断）

    arXiv:2611.06481v1 Announce Type: cross  Abstract: Graph-based retrieval-augmented generation (GraphRAG) answers questions by retrieving over a knowledge graph built from a corpus. Yet retrieval strategies are designed around the structure their own construction pipeline produces, and whether they survive a change of construction is rarely tested. We study this cross-construction portability: whether one retrieval procedure, with no construction-specific schema, relation names, or retuning, stays effective when the graph is built differently. We introduce ArcRAG, which induces a query-local graph interface at inference time (the node labels, relation descriptors, and property keys reachable from the entities a question names), plans multi-hop traversals against it, and executes and repairs them as Cypher. We build each of the MuSiQue, HotpotQA, and TriviaQA corpora as a schema-guided graph (typed nodes, a closed relation set, source provenance) and as an open-information-extraction gra
    
[^7]: PMRecBench：面向多模态查询下推荐智能体个性化的基准测试

    PMRecBench: A Benchmark for Personalization in Recommendation Agents under Multimodal Queries

    [https://arxiv.org/abs/2611.06349](https://arxiv.org/abs/2611.06349)

    提出了首个评估多模态查询下推荐智能体个性化能力的基准PMRecBench，通过将510个来源于真实用户行为模式的图文查询与用户真实历史配对，填补了多模态推荐评估中缺失个性化维度的空白。

    

    在购物和娱乐类Web平台上，多模态查询下的推荐智能体旨在提供满足用户视觉指定需求、并反映从用户历史中推断出的偏好的个性化推荐。然而，现有的个性化推荐基准仅使用纯文本查询，而多模态基准只评估查询匹配，并未基于真实用户历史来评估个性化能力。我们提出了个性化多模态推荐基准PMRecBench，这是首个评估多模态查询下推荐智能体个性化能力的基准。该基准将涵盖服装和电影领域的510个经过质量控制的图文查询与相应用户的真实历史记录相配对。PMRecBench具有两个关键设计：(1) 查询来源于真实用户记录中一致的行为模式，参考图像和目标均与实际交互相关联；(2) 具有挑战性的诊断性设计

    arXiv:2611.06349v1 Announce Type: new  Abstract: On Web platforms for shopping and entertainment, recommendation agents under multimodal queries aim to deliver personalized recommendations that satisfy users' visually specified requirements and reflect preferences inferred from user histories. However, existing personalized recommendation benchmarks use text-only queries, while multimodal ones assess query matching without evaluating personalization based on real user histories. We introduce the Personalized Multimodal Recommendation Benchmark (PMRecBench), the first benchmark to evaluate the personalization of recommendation agents under multimodal queries. It pairs 510 quality-controlled image-text queries across clothing and movies with the corresponding users' real histories. PMRecBench features two key designs: (1) queries derived from consistent behavioral patterns in real user records, with reference images and targets linked to actual interactions; and (2) challenging diagnosti
    
[^8]: 本地商业中实时供给约束下的生成式检索

    Generative Retrieval under Real-Time Supply Constraints in Local Commerce

    [https://arxiv.org/abs/2611.06323](https://arxiv.org/abs/2611.06323)

    该论文提出RTP-LLM流水线，通过将合格的残差量化语义标识符预先编译为压缩稀疏行前缀索引、与模型预填充并行执行CPU编译并在beam间共享，使生成式检索能在解码过程中直接满足本地商业的实时供给约束，避免对不可行候选浪费beam容量和推理计算。

    

    生成式检索为候选实体预测语义标识符。在本地零售和外卖配送场景中，候选还必须在特定时间对特定用户是可服务的。位置、天气和交通会不断重塑商家与商品供给，因此目录中有效的标识符对当前请求而言可能是不可行的。生成后过滤方式只有在beam容量和推理计算已经消耗之后才会移除这类候选。我们研究了在显式物化的、请求时刻供给快照上的检索问题，此时较短的输出序列几乎没有机会来摊销在线约束准备的开销。我们的RTP-LLM流水线将合格的残差量化标识符编译为压缩稀疏行前缀索引，将CPU编译与模型预填充阶段重叠执行，并在多个beam之间共享上传的索引。合法的扩展保留原始模型的对数概率，约束状态则跟随被选中的beam父节点。我们……

    arXiv:2611.06323v1 Announce Type: new  Abstract: Generative retrieval predicts semantic identifiers for candidate entities. In local retail and food delivery, candidates must also be serviceable for a particular user at a particular time. Location, weather, and traffic reshape merchant and product supply, so a catalog-valid identifier may be infeasible for the current request. Post-generation filtering removes such candidates only after beam capacity and inference computation have been spent. We study retrieval over an explicitly materialized, request-time supply snapshot, where short output sequences leave little opportunity to amortize online constraint preparation. Our RTP-LLM pipeline compiles eligible residual-quantization identifiers into a compressed sparse row prefix index, overlaps CPU compilation with model prefill, and shares the uploaded index across beams. Legal expansions retain original-model log probabilities, and constraint states follow selected beam parents. We forma
    
[^9]: 搜索增强推荐系统是否使用了最新的搜索会话？

    Do Search-Augmented Recommenders Use the Latest Search Episode?

    [https://arxiv.org/abs/2611.06044](https://arxiv.org/abs/2611.06044)

    提出FocalSwap审计框架，通过在固定目标与候选集的条件下替换最新搜索或全部搜索输入，检验搜索增强推荐系统对最新搜索会话的实际依赖程度，发现其估计收益随对照方式的选择而变化。

    

    搜索增强推荐系统可以利用最新的搜索会话以及更早的搜索记录。我们提出FocalSwap来测试这些输入对下一次推荐的贡献。对于每一条推荐，FocalSwap在保持目标、候选集、搜索前历史和评分器不变的情况下，替换最新的搜索槽位或所有搜索输入。它将中性掩码与匹配的已观测搜索替换进行对比，记录排名变化、相对于各对照组的RR增益，以及对真实焦点搜索的偏好。植入信号实验和GenSAR的当前查询任务证实，该审计方法能够检测到已知的输入依赖。我们在KuaiSAR和Qilin数据集上评估行为与词汇证据，并使用基于公开的SESRec和UniSAR实现所训练的推荐模型检查点进行实验。在三个独立训练的UniSAR检查点上，估计的全源收益取决于对照组的选择。在相同的66,139行数据上，将所有搜索输入替换为中性ID…（原文摘要在此处被截断）

    arXiv:2611.06044v1 Announce Type: cross  Abstract: Search-augmented recommenders can use the latest search episode and earlier searches. We introduce FocalSwap to test how these inputs contribute to the next recommendation. For each recommendation, it changes the latest search slot or all search inputs while holding the target, candidates, pre-search history, and scorer fixed. It compares neutral masks with matched observed-search replacements, recording rank changes, RR gains relative to each control, and preference for the true focal search. A planted signal and GenSAR's current-query task confirm that the audit detects known input dependence. We evaluate behavioral and lexical evidence on KuaiSAR and Qilin, and recommendation checkpoints trained using the released SESRec and UniSAR implementations.   Across three independently trained UniSAR checkpoints, the estimated all-source benefit depends on the control. On the same 66,139 rows, replacing all search inputs with a neutral ID yi
    
[^10]: 从废料到发现：在淘宝信息流推荐中回收高分未曝光候选物品

    From Waste to Discovery: Recycling High-Scored Unexposed Candidates in Taobao Feed Recommendation

    [https://arxiv.org/abs/2611.05997](https://arxiv.org/abs/2611.05997)

    论文提出无需训练的召回阶段系统URED，通过回收工业推荐流水线中通常被丢弃的高分未曝光候选物品并在召回阶段重新注入，在淘宝信息流推荐中高效帮助用户发现新兴趣，避免了传统曝光提升策略因冷启动候选被排序器高估而损害参与度的问题。

    

    帮助用户发现新兴趣对于工业推荐系统的长期健康发展至关重要，然而常见的补救措施——提升探索不足（探索性）内容的曝光——在生产环境中往往适得其反。我们提出了URED（Unexposed Recycling for Efficient Discovery，未曝光回收高效发现），这是一个无需训练的召回阶段系统，已在阿里巴巴的淘宝信息流推荐中为千万级日活跃用户服务超过六个月。该系统基于一个经生产环境验证的洞察：朴素的曝光提升之所以失败，是因为它注入了冷启动、信息贫乏的候选物品，而排序器会系统性地高估这些物品，不仅无法带来发现收益，还会损害用户参与度。那些得分很高但从未曝光的物品（工业流水线通常会丢弃的信号）构成了一个既相关又可被可靠预测的候选池。在30天的时间窗口内，URED聚合这些物品，并在召回阶段通过一个涵盖离线存储、分布式聚合和低延迟服务的轻量级流水线将其重新注入。

    arXiv:2611.05997v1 Announce Type: new  Abstract: Helping users discover new interests is vital to the long-term health of industrial recommender systems, yet the usual remedy, boosting exposure of under-explored (discovery) content, often backfires in production. We report URED (Unexposed Recycling for Efficient Discovery), a training-free recall-stage system serving Taobao feed recommendation at Alibaba for over six months to tens of millions of daily active users. It rests on a production-validated insight: naive boosting fails because it injects cold, information-poor candidates that rankers systematically over-estimate, buying no discovery gain while risking engagement. Items scored highly but never exposed (signals industrial pipelines routinely discard) form a pool that is both relevant and reliably predictable. Over a 30-day horizon URED aggregates them and re-injects at recall time through a lightweight pipeline spanning offline storage, distributed aggregation, and low-latency
    
[^11]: 一篮子多专家：面向金融资产推荐的去相关重排序方法

    One Basket, Many Experts: Decorrelated Re-ranking for Financial Asset Recommendation

    [https://arxiv.org/abs/2611.05952](https://arxiv.org/abs/2611.05952)

    提出DSER重排序方法，通过用收益去相关的替代资产替换专家推荐列表中的部分内容，并借助单一可调参数m，在用户相关性与投资组合风险调整后收益之间实现灵活权衡。

    

    针对散户投资者的金融资产推荐通常以相关性为优化目标，即推荐结果与每位用户实际购买行为的匹配程度。而从服务提供商的角度来看，推荐的价值还取决于投资组合的表现：即所推荐资产（投资篮子）所带来的风险调整后收益，这直接驱动用户参与度和费用收入。这两个目标很少能同时实现，因为最匹配用户行为的推荐往往集中在少数热门资产上，而这些资产的收益高度相关，从而限制了提供商所能提供的风险调整后收益上限。我们提出了去相关卫星专家重排序方法（DSER），该方法保留专家跟踪列表中最相关的推荐，并用收益去相关的替代资产替换其余推荐。被替换的槽位数量是一个由提供商设定的单一整数 m（一个可调节的旋钮），因此相关性与绩效之间的权衡……

    arXiv:2611.05952v1 Announce Type: new  Abstract: Financial asset recommendation for retail investors is typically optimized for relevance, that is, how well a recommendation matches what each user will buy. From the service provider's point of view, the value of a recommendation also depends on portfolio performance: the risk-adjusted return the recommended assets (the basket) deliver, which drives user engagement and fee revenue. The two goals are rarely achieved jointly, because the recommendations that best match user behavior concentrate on a few popular assets whose returns are highly correlated, which caps the risk-adjusted return a provider can offer. We propose Decorrelated-Satellite Expert Re-ranking (\textbf{DSER}), which retains the most relevant recommendations from an expert-following list and replaces the rest with return-decorrelated alternatives. The number of replaced slots is a single integer~$m$ (a knob) that the provider sets, so the relevance--performance trade-off
    
[^12]: 最优重排序深度

    Optimal Re-Ranking Depth

    [https://arxiv.org/abs/2611.05908](https://arxiv.org/abs/2611.05908)

    该论文利用基于大语言模型的相关性判断，首次在逐查询层面揭示了许多查询存在明确的最优重排序深度（超过该深度检索效果会停滞或下降），并将重排序深度定义为查询特有属性，研究其能否被预先预测。

    

    二级神经排序器通常采用固定的重排序深度，其假设是随着深度的增加，检索效果会趋于饱和。先前的研究已经质疑了增加重排序深度会带来线性性能提升这一假设，并进一步表明最优重排序深度在不同查询之间存在很大差异。过去，由于所需人工标注的规模庞大，在方法上很难研究这类现象。随着基于大语言模型（LLM）的相关性判断技术的出现，我们现在可以更容易地开展此类研究，在本研究中用于精确确定每个查询各自的最优重排序深度。通过在典型的重排序流程中使用密集的基于LLM的相关性判断，我们证明了许多查询都存在一个明确的最优重排序深度，超过该深度后，检索效果会停滞甚至下降。我们将重排序深度表述为一种查询特有的属性，并研究其是否可以被预先预测。

    arXiv:2611.05908v1 Announce Type: new  Abstract: Second-stage neural rankers are commonly applied with a fixed re-ranking depth, assuming that retrieval effectiveness saturates as depth increases. Prior research has questioned the assumption that increasing the re-ranking depth yields linear performance gains, further suggesting that optimal re-ranking depth varies considerably from query to query. In the past, studying such phenomena was methodologically difficult given the scale of manual annotation required. With the advent of LLM-based relevance judgments we can now more easily undertake such studies, in this case to pinpoint the optimal re-ranking depth on a per-query basis. Using dense LLM-based relevance judgments over a typical re-ranking pipeline, we show that many queries exhibit a well-defined optimal re-ranking depth, beyond which effectiveness stagnates or degrades. We formulate re-ranking depth as a query-specific property and study whether it can be predicted a priori fr
    
[^13]: Weaver：通过增量推理编织偏好证据以实现个性化检索增强生成

    Weaver: Weaving Preference Evidence through Incremental Reasoning for Personalized Retrieval-Augmented Generation

    [https://arxiv.org/abs/2611.05900](https://arxiv.org/abs/2611.05900)

    Weaver 框架通过耦合意图依赖图与记忆条件检索轨迹图的双层检索与推理图，以增量方式将记忆条件检索轨迹与组合式偏好推理相编织，解决了个性化检索增强生成中偏好证据分散且跨情境易混淆的问题。

    

    个性化检索增强生成需要从长期的交互历史中提取并应用用户偏好。这项任务具有挑战性，因为偏好证据分散在过去的对话中，并且与特定的时间和情境上下文紧密绑定。因此，孤立的片段很少能为响应提供完整的支持，而在不同条件下表达的偏好也容易被混淆。为了解决这些局限性，我们提出了 Weaver，这是一个以增量方式将记忆条件下的检索轨迹与组合式偏好推理相编织的框架，用于生成个性化响应。Weaver 运行在一个双层检索与推理图之上，该图将上层的意图依赖图与下层的记忆条件检索轨迹图耦合在一起。依赖信息缺口追踪用于刻画回答当前请求所需的信息及其前置依赖关系……

    arXiv:2611.05900v1 Announce Type: new  Abstract: Personalized retrieval-augmented generation requires extracting and applying user preferences from extended interaction histories. This task is challenging because preference evidence is scattered across past conversations and remains bound to specific temporal and situational contexts. Consequently, isolated fragments rarely provide complete support for a response, while preferences expressed under different conditions are easily conflated. To address these limitations, we propose Weaver, a framework that incrementally weaves memory-conditioned retrieval trajectories with compositional preference reasoning for personalized responses. Weaver operates on a Dual-Layer Retrieval and Reasoning Graph that couples an upper Intent Dependency Graph with a lower Memory-Conditioned Retrieval Trajectory Graph. Dependency Information Gap Tracking represents the information needed to answer the current request and its prerequisite dependencies in the
    
[^14]: SciToT：面向科学论文的多模态“话到嘴边”（Tip-of-the-Tongue）检索

    SciToT: Multimodal Tip-of-the-Tongue Retrieval for Scientific Papers

    [https://arxiv.org/abs/2611.05829](https://arxiv.org/abs/2611.05829)

    该论文提出了首个面向科学论文的多模态“话到嘴边”（ToT）检索基准SciToT，通过从文本和视觉元素中提取记忆线索并用大语言模型合成模拟人类不完整记忆的查询，填补了科学文献重新查找任务中缺乏评测基准的空白。

    

    研究人员经常难以重新找到某篇科学论文，尽管他们记得论文内容的某些方面，却无法回忆起可靠的标识信息，例如标题或作者姓名。这类重新查找任务被称为“话到嘴边”检索，对于科学文献而言尤为困难，因为人们对科学文献的记忆可能同时涉及文本信息和视觉元素，包括插图、示意图和绘图。然而，现有的ToT检索基准并未涵盖科学搜索场景。我们提出了一个面向科学ToT检索的多模态基准，包含涵盖五个领域的108,512篇开放获取科学文献以及11,499条合成生成的查询。为构建该基准，我们设计了一个可扩展的查询引导流水线，该流水线从源文档中独立提取文本记忆线索和视觉记忆线索，并利用大语言模型生成类似于人类不完整记忆的查询。我们对……（原文摘要在此处截断）

    arXiv:2611.05829v1 Announce Type: new  Abstract: Researchers frequently encounter difficulties re-finding scientific papers when they remember aspects of the content but cannot recall reliable identifiers, such as titles or author names. This type of re-finding task, known as tip-of-the-tongue (ToT) retrieval, is particularly challenging for scientific documents, where recollections may involve textual information and visual elements, including figures, diagrams, and plots. Existing ToT retrieval benchmarks do not address scientific search.   We introduce a multimodal benchmark for scientific ToT retrieval, comprising 108,512 open-access scientific documents across five domains and 11,499 synthetically generated queries. To construct the benchmark, we propose a scalable query elicitation pipeline that independently extracts textual and visual memory cues from source documents and uses large language models to generate queries resembling incomplete human recollections. We characterize s
    
[^15]: 物品表示至关重要：连接硬语义ID与软原型混合以实现生成式检索

    Item Representations Matter: Bridging Hard Semantic IDs and Soft Prototype Mixtures for Generative Retrieval

    [https://arxiv.org/abs/2611.05799](https://arxiv.org/abs/2611.05799)

    提出CAP-Mix流式推荐器，将每个物品表示为共享原型的软混合加上物品专属向量，在不引入离散瓶颈的前提下结合了硬语义ID的共享结构与连续表示的无损优势。

    

    现代推荐系统依赖生成式检索技术来预测下一个物品，既可以将其表示为离散编码，也可以表示为连续向量。一方面，语义ID（SIDs）为离散模型提供了共享参数和预训练的语义结构，但将物品离散化为硬编码会丢失原始内容嵌入中的部分信息。另一方面，连续模型避免了量化，但通常使用扁平的物品查找表，物品之间缺乏显式的参数绑定。我们探究连续推荐器能否在不引入离散瓶颈的情况下引入共享结构。我们提出了CAP-Mix，一种流式推荐器，它将每个物品表示为共享原型的软混合再加上一个物品专属向量。分配、原型和物品向量进行联合训练，并在推理时保留软表示。预训练信息是可选的：SID的归属概率可以作为先验。

    arXiv:2611.05799v1 Announce Type: cross  Abstract: Modern recommender systems rely on generative retrieval techniques to predict the next item either as discrete codes or as a continuous vector. On the one hand, Semantic IDs (SIDs) give discrete models shared parameters and pretrained semantic structure, but discretizing items into hard codes discards some information from the original content embeddings. On the other hand, continuous models avoid quantization, yet typically use flat item lookup tables with no explicit parameter tying across items. We ask whether a continuous recommender can introduce shared structure without making it a discrete bottleneck. We propose CAP-Mix, a flow recommender that represents each item as a soft mixture of shared prototypes plus an item-specific vector. The assignments, prototypes, and item vectors are trained jointly, and the soft representation is retained at inference. Pretrained information is optional: SID responsibilities can serve as a prior 
    
[^16]: 面向共享组织皮质的组合式知识架构

    Toward a Compositional Knowledge Architecture for a Shared Organizational Cortex

    [https://arxiv.org/abs/2611.05753](https://arxiv.org/abs/2611.05753)

    提出由信号、领域、主张、维度四层组成的组合式知识架构，使组织解释连同其依据与立场持久保存，并通过真值维护网络随证据变化动态修订，从而显著提升检索性能。

    

    组织在对话、文档、决策和人际关系中积累证据，但其行动依赖于解释——即由多项观察组合而成、建立在假设之上、并随环境变化而不断修订的判断。现有的检索技术、知识图谱和智能体记忆系统只保存了记录，却丢失了解释。我们提出一个四层架构——信号、领域、主张与维度，其中解释连同其依据与立场被持久化保存；一个真值维护网络随证据变化修订这些立场；经过验证且带有版本管理的维度将解释定位于命名轴线上。我们将这一成果称为共享组织皮质。在一项注册检索实验中，维度词项带来 0.028 的 nDCG@10 提升（p = 0.002）。在系统级对比中，完整检索配置相比仅信号检索将 nDCG@10 提升 0.081，相比图增强的仅信号检索提升 0.053。

    arXiv:2611.05753v1 Announce Type: new  Abstract: Organizations accumulate evidence in conversations, documents, decisions and relationships, but they act on interpretations: judgements assembled from several observations, resting on assumptions, and revised as circumstances change. Existing retrieval, knowledge graphs and agent memory keep the records and lose the interpretation. We propose a four-layer architecture, Signals, Domains, Claims and Dimensions, in which interpretations persist with their justifications and standing, a truth-maintenance network revises that standing as evidence changes, and validated, versioned Dimensions position interpretations on named axes. We call the result a shared organizational cortex. In a registered retrieval experiment, the Dimension term gains 0.028 nDCG@10 (p = 0.002). In system-level comparisons, the full retrieval configuration improves nDCG@10 by 0.081 over Signals-only and 0.053 over graph-augmented Signals-only retrieval (both reported p 
    
[^17]: 为邻近图与k近邻图构建提速的PiPNN全面优化

    Turbocharging PiPNN for Proximity and $k$-NN Graph Building

    [https://arxiv.org/abs/2611.05690](https://arxiv.org/abs/2611.05690)

    本文提出优化版PiPNN II及其GPU实现GPiPNN，将基于图的近似最近邻索引构建速度最高提升8倍（CPU），并比CAGRA快最高50倍，还可直接高效构建高质量k-NN图。

    

    基于图的近似最近邻搜索（ANNS）索引（如HNSW和Vamana）提供了最先进的查询性能，但代价是漫长的索引构建时间。最近提出的PiPNN方法通过一种基于分区的算法，利用稠密矩阵乘法和top-K聚合内核，大幅加速了索引构建。本文充分释放了PiPNN硬件友好算法设计的潜力。我们提出PiPNN II，对PiPNN方案的多个步骤进行了优化，使基于图的ANNS索引构建速度提升高达8倍。此外，我们提出了GPiPNN这一GPU实现，其构建的ANNS索引质量相当，但速度比Jasper快高达43.7倍，比CAGRA快高达50倍。我们还将PiPNN II和GPiPNN扩展为可直接计算高质量k-NN图，为许多现有系统中用于聚类、异常检测等任务的方法带来了巨大的加速。

    arXiv:2611.05690v1 Announce Type: cross  Abstract: Graph-based Approximate Nearest Neighbor Search (ANNS) indexes such as HNSW and Vamana deliver state-of-the-art query performance at the cost of large indexing times. The recently introduced PiPNN method drastically accelerated index construction by way of a partitioning-based algorithm leveraging dense matrix multiplication and top-$K$ aggregation kernels.   In this paper, we unlock the full potential of PiPNN's hardware-friendly algorithm design. We introduce PiPNN II, which optimizes multiple steps of the PiPNN recipe, giving up to $8\times$ speedups for graph-based ANNS index construction. In addition, we present GPiPNN, a GPU implementation which builds comparable ANNS indices up to $43.7\times$ faster than Jasper and up to $50\times$ faster than CAGRA. We also extend PiPNN II and GPiPNN to compute high-quality $k$-NN graphs directly, which unlocks massive speedups over methods used in many existing systems for clustering, anomaly
    
[^18]: 使用反馈模糊认知图混合体组合相关专家知识

    Combining Relevant Expert Knowledge Using Mixtures of Feedback Fuzzy Cognitive Maps

    [https://arxiv.org/abs/2611.05664](https://arxiv.org/abs/2611.05664)

    该论文将法律证据理论中的相关性定义扩展应用于神经符号系统，通过将专家的反馈模糊认知图混合为共同知识网络，并借助新的类贝叶斯定理对混合知识进行解混合，从而仅组合相关专家答案以提升系统的准确性和可扩展性。

    

    神经符号系统应当只组合来自专家智能体或其他知识源的相关答案。随着知识网络的不断增长并连接越来越多的知识源，针对给定输入查询仅使用相关专家对于准确性和可扩展性变得愈发重要。我们从证据法理论出发扩展了相关性的标准定义：在给定可用知识的条件下，当且仅当一个答案对于手头问题既具有证明力又具有实质性时，该答案才是相关的。我们将这一相关性方案应用于混合真实或人工专家的反馈模糊认知图（FCMs）。各专家FCM的有向因果图混合或组合成一个共同的知识网络。我们将专家的均衡极限环预测或答案与混合FCM的答案进行比较，然后通过它们之间的差异或变化来估计答案的证明力状态。一个新的类贝叶斯定理将知识混合物解混合为……（原文摘要在此处截断）

    arXiv:2611.05664v1 Announce Type: new  Abstract: A neuro-symbolic system should combine only relevant answers from expert agents or other knowledge sources. Using only relevant experts for a given input query becomes ever more important for accuracy and scalability as the knowledge network grows and connects ever more knowledge sources. We extend the standard definition of relevance from the legal theory of evidence: An answer is relevant if and only if it is both probative and material for the matter at hand given the knowledge available. We apply this relevancy scheme to mixing the feedback fuzzy cognitive maps (FCMs) of real or artificial experts. The expert FCM directed causal graphs mix or combine into a common knowledge network. We compare the equilibrium limit-cycle predictions or answers of the experts to the answers of the mixture FCM and then estimate the probative status of answers by their difference or variation. A new Bayes-like theorem de-mixes the knowledge mixture into
    
[^19]: 通过耦合超参数缩放实现推荐系统数据高效的缩放定律发现

    Data-Efficient Scaling-Law Discovery for Recommendation Systems via Coupled Hyperparameter Scaling

    [https://arxiv.org/abs/2611.05643](https://arxiv.org/abs/2611.05643)

    该论文提出通过缩放层将低维超超参数映射到完整架构空间，并结合缩放感知代理模型与帕累托前沿选择，仅需少量实验即可高效发现任意推荐排序模型的缩放定律。

    

    大型Web平台的排序与推荐模型与大语言模型类似，同样受益于计算量的增加，其缩放行为大致呈幂律形式。然而，排序模型具有异构架构，这使得其缩放定律的确定比大语言模型更为复杂。我们研究如何高效地发现任意排序模型的缩放定律。我们引入了一个缩放层，它将底层的架构参数划分为语义上连贯的组，并定义这些组之间如何协同变化，从而建立起从低维“超超参数”空间到完整架构空间的映射。在此基础上，Axia拟合缩放感知的代理模型，基于后验的采集准则选择配置，并返回帕累托前沿，从中验证每个模型的具体工作点——用少量信息量丰富的实验取代穷举式扫描。一个面向模型集群的评估协议

    arXiv:2611.05643v1 Announce Type: cross  Abstract: Large-scale ranking and recommendation models of Web platforms benefit from increased compute much as large language models do, and their scaling behavior is broadly power-law in form. Yet ranking models have heterogeneous architectures which makes their scaling law less straightforward than that of LLMs. We address how to discover the scaling law of an arbitrary ranking model efficiently. We introduce a scaling layer, which groups primitive architecture parameters into semantically coherent groups and defines how they co-vary, inducing a mapping from a low-dimensional space of hyper-hyperparameters into the full architecture space. On top of it, Axia fits scaling-informed surrogates, selects configurations by a posterior-based acquisition criterion, and returns a Pareto frontier from which per-model operating points are validated -- replacing exhaustive sweeps with a small number of informative runs. A fleet-level evaluation protocol,
    
[^20]: RAGCompass：通过重排序实现可泛化的检索器路由

    RAGCompass: Generalizable Retriever Routing via Reranking

    [https://arxiv.org/abs/2611.05596](https://arxiv.org/abs/2611.05596)

    RAGCompass 通过对比微调重排序器直接读取不同检索器返回的段落集合来实现查询路由，从而可泛化到未见过的检索器和新语料库，将答案 F1 相对提升 28%。

    

    在检索增强生成（RAG）中存在多种不同的检索器，但没有一种检索器能在所有查询上都表现最佳。因此，关键问题是：给定一个查询以及不同检索器检索到的段落，哪一种最适合该查询？现有的路由器通常基于查询特定特征和固定的检索器池进行预测；这类路由器对于未见过的检索器无法给出输出，且在设计上天生无法应对新语料库。为填补这些空白，RAGCompass 直接读取检索到的段落集合本身：来自陌生检索器或新语料库的段落仍可与其他段落直接比较，因此在仅依赖查询特征的路由会失效的场景下，路由依然有效。我们针对每个证据集所产生的答案，对重排序器进行对比微调，这贡献了绝大部分性能提升；一个小型的比较模块带来了适度的额外改进，且这两个组件都不包含检索器专属参数。RAGCompass 将答案 F1 相对提升了 28%

    arXiv:2611.05596v1 Announce Type: new  Abstract: Different retrievers exist for retrieval-augmented generation, yet none performs best for all queries. Therefore, the key question is: given a query and retrieved passages from different retrievers, which one is most suited for that query? Existing routers usually make predictions from query-specific features and a fixed pool of retrievers; such routers have no output for an unseen retriever and are incapable by design on a new corpus. To fill these gaps, RAGCompass reads the retrieved passage sets themselves: passages from an unfamiliar retriever or corpus stay directly comparable to other passages, so routing stays well designed where a query-only map would fail. We contrastively fine-tune a reranker against the answer each evidence set produces, which accounts for most of the gain; a small comparison head adds a moderate improvement, and neither component has retriever-specific parameters. RAGCompass improves answer F1 by 28% relative
    
[^21]: LIBRA：可执行工具库的程序化归纳

    LIBRA: Programmatic Induction of Executable Tool Libraries

    [https://arxiv.org/abs/2611.05445](https://arxiv.org/abs/2611.05445)

    LIBRA 通过让大语言模型提出工具间契约并在沙盒中进行运行时验证，将扁平的工具目录自动归纳为可执行的工具库，从而在多工具任务中检索出包含先决工具的完整工作流，无需依赖标注或任务特定训练，在多个基准上显著提升了检索性能。

    

    大型语言模型智能体越来越多地从庞大的工具目录中检索工具，然而大多数检索器仅根据请求相似度对每个工具进行独立排序。对于多工具任务，这种方式只能召回与目标匹配的工具，而遗漏了工作流程所需的先决工具。我们提出 LIBRA——一个通过运行时组装进行工具库归纳的框架，它将扁平的工具目录转化为可执行的工具库。大语言模型为每个工具提出契约，包括候选调用、可复用输出与状态变更以及成功条件；LIBRA 随后在沙盒环境中测试这些组合，并保留在运行时得到验证支持的工具间关系。该工具库在用户查询之前构建完成，无需依赖关系标注、工具使用轨迹或任务特定训练，同时支持一次性检索和智能体式检索。在 AppWorld、BFCL 和 SciToolEval 三个基准上，相比相同的稠密检索器，LIBRA 将一次性检索的 Recall 和 NDCG 平均提升了 7.9 和 3.7 个百分点。

    arXiv:2611.05445v1 Announce Type: cross  Abstract: Large language model agents increasingly retrieve tools from large catalogs, yet most retrievers rank each tool independently by request similarity. For multi tool tasks, this recovers the goal matching tool while omitting prerequisite tools required by the workflow.We introduce LIBRA, a framework for Library Induction by Runtime Assembly that turns a flat tool catalog into an executable tool library. An LLM proposes per tool contracts of candidate invocations, reusable outputs and state changes, and success conditions; LIBRA then tests these compositions in a sandbox and retains the relations supported at runtime. Constructed before user queries, the library requires no dependency annotations, tool use trajectories, or task specific training, and supports both one-shot and agentic retrieval. On AppWorld, BFCL, and SciToolEval, LIBRA improves one-shot Recall and NDCG over the same dense retriever by 7.9 and 3.7 percentage points on ave
    
[^22]: 用于组合图像检索的因子化多模态大语言模型嵌入

    Factorized MLLM Embeddings for Composed Image Retrieval

    [https://arxiv.org/abs/2611.05193](https://arxiv.org/abs/2611.05193)

    提出FactEmbed方法，通过因子化潜在变量和中间目标想象目标将组合查询理解与判别性嵌入构建解耦，从而更充分地利用MLLM的组合理解能力以提升组合图像检索性能。

    

    组合图像检索旨在根据参考图像和修改文本检索目标图像。尽管近期利用多模态大语言模型（MLLM）的方法已取得显著进展，我们认为MLLM的组合理解能力仍未得到充分利用。这源于现有方法仅通过对比监督将组合查询直接映射到检索嵌入中，而未显式解析预期的视觉修改。在本工作中，我们提出了因子化MLLM嵌入，该方法将组合查询理解与判别性嵌入的构建解耦开来。FactEmbed不是在没有内部推理的情况下直接将组合查询投影为单一嵌入，而是采用一组因子化的潜在变量，通过中间目标想象目标来模拟预期的视觉修改。该过程促使这些潜在变量落地到目标（摘要在此处截断）……

    arXiv:2611.05193v2 Announce Type: replace-cross  Abstract: Composed Image Retrieval (CIR) aims to retrieve a target image given a reference image and a modification text. Although recent approaches leveraging Multimodal Large Language Models (MLLMs) have shown significant progress, we argue that the MLLM's compositional understanding remains underutilized. This stems from directly mapping composed queries into a retrieval embedding via contrastive supervision alone, without explicitly resolving the intended visual modifications. In this work, we introduce Factorized MLLM Embeddings (FactEmbed), which decouples compositional query comprehension from building discriminative embeddings. Rather than directly projecting the composed query into a single embedding without internal reasoning, FactEmbed employs a set of factorized latents that simulate the intended visual modifications via an intermediate target imagination objective. This process encourages the latents to ground into target se
    
[^23]: TAGR：面向工业直播广告的时间自适应生成式推荐

    TAGR: Temporally Adaptive Generative Recommendation for Industrial Live-Streaming Advertising

    [https://arxiv.org/abs/2608.24034](https://arxiv.org/abs/2608.24034)

    TAGR通过三层时间自适应机制（令牌级动态ID刷新、意图级多尺度建模、偏好级在线稳定对齐）解决了直播广告推荐中内容快速变化导致的时效性问题。

    

    直播广告是短视频和电商平台重要的变现渠道，其中快速变化的直播内容、推广产品和用户反馈对推荐模型提出了很强的时效性要求。现有为静态领域设计的生成式推荐器在三个层面存在不足：静态语义ID（SID）无法跟踪不断演变的直播广告；单尺度行为建模忽略了意图转移；偏好优化在新鲜的在策略反馈与训练稳定性之间冲突。我们提出TAGR，一种在三个层面具有时间自适应性的生成式推荐框架：直播广告令牌化、用户意图建模和偏好对齐。在令牌层面，直播语义协作ID（LSID）基于当前直播场景和推广产品定期刷新每个活跃广告的SID，同时保留稳定的分层令牌词汇表用于自回归生成。在意图层面...

    arXiv:2608.24034v1 Announce Type: new  Abstract: Live-streaming advertising is an important monetization channel on short-video and e-commerce platforms, where rapidly changing live content, promoted products, and user feedback impose strong freshness requirements on recommendation models. Existing generative recommenders designed for static domains fail at three levels: static semantic IDs (SID) cannot track evolving live ads; single-scale behavior modeling misses shifting intent; preference optimization conflicts between fresh on-policy feedback and training stability. We propose TAGR, a generative recommendation framework with temporal adaptation at three levels: live-ad tokenization, user intent modeling, and preference alignment. At the token level, Live Semantic-Collaborative ID (LSID) periodically refreshes each active ad's SID based on its current live scene and promoted products, while retaining a stable hierarchical token vocabulary for autoregressive generation. At the inten
    
[^24]: 可信RAG：用于检测生成式AI系统中错误信息与知识投毒的评估代理

    Trustworthy RAG: An Evaluation Agent for Detecting Misinformation and Knowledge Poisoning in Generative AI Systems

    [https://arxiv.org/abs/2608.21095](https://arxiv.org/abs/2608.21095)

    本文提出了一种结合NLI事实核查与五信号投毒检测的评估代理，并引入信任指数，在TruthfulQA上实现了高准确率与精确率，有效缓解了RAG系统中的知识投毒风险。

    

    检索增强生成（RAG）将大型语言模型（LLM）的输出锚定在外部知识上，但RAG系统通常信任其检索到的任何内容，从而造成“安全-可靠性差距”：高语义相关性并不保证事实真实性。攻击者利用这一漏洞进行知识投毒，插入恶意文档以引发定向错误信息。我们提出了一种评估代理，这是一种中间件，结合了自然语言推理（NLI）事实核查、具有相关性加权聚合的五信号投毒检测器，以及一个信任指数 T = 0.4 F + 0.35 C + 0.25 (1 - P )，并针对高污染情境采用非线性阻尼器。在TruthfulQA上使用Llama 3.3 70B时，该代理达到了91%的准确率和100%的精确率，对指令注入的召回率为100%，而就地编辑（如实体替换）仍难以检测。在三个LLM上，信任指数保持判别性，接收者操作特征（ROC）曲线表现良好。

    arXiv:2608.21095v1 Announce Type: cross  Abstract: Retrieval-Augmented Generation (RAG) grounds Large Language Model (LLM) outputs in external knowledge, but RAG systems usually trust whatever they retrieve, creating a Security-Reliability Gap: high semantic relevance does not guarantee factual truth. Adversaries exploit this through knowledge poisoning, inserting malicious documents to cause targeted misinformation. We propose an Evaluation Agent, middleware that combines Natural Language Inference (NLI) factual verification, a five-signal poison detector with relevance-weighted aggregation, and a Trust Index T = 0.4 F + 0.35 C + 0.25 (1 - P ) with a non-linear dampener for high-contamination contexts. On TruthfulQA with Llama 3.3 70B, the agent reaches 91% accuracy and 100% precision, with 100% recall on instruction injection, while in-place edits, such as entity swaps, remain hard to detect. Across three LLMs the Trust Index stays discriminative, with a Receiver Operating Characteri
    
[^25]: 复杂分块何时值得？多目标评估大规模分块方法

    When Is Complex Chunking Worth It? A Multi-Objective Evaluation of Chunking Methods at Scale

    [https://arxiv.org/abs/2608.16586](https://arxiv.org/abs/2608.16586)

    本文通过多目标评估发现，复杂分块方法在检索效果和系统成本上并不总是优于简单方法，最佳策略依赖于具体应用场景。

    

    摘要：密集检索通常在基准测试中用单个嵌入表示每个文档，尽管实际检索系统往往索引需要分块的长文档。在这些场景中，选择的分块方法不仅影响检索质量，还影响索引吞吐量、查询延迟和内存使用。以往对分块策略的比较主要集中在检索性能上，对操作权衡的探索不足。为解决这些问题，我们在两个可扩展语料库、三种嵌入模型和多种语料库规模上评估了八种代表性分块策略，同时测量了检索效果和系统级成本。我们的结果表明，计算成本高的方法很少能比简单分块提供一致的改进。相反，最佳策略取决于嵌入模型、数据集、语料库规模和目标检索指标。具有相似性能的方法...

    arXiv:2608.16586v1 Announce Type: new  Abstract: Dense retrieval is commonly evaluated on benchmarks that represent each document with a single embedding, even though real-world retrieval systems often index long documents that require chunking. In these settings, the chosen chunking method not only affects retrieval quality, but also indexing throughput, query latency, and memory usage. Prior comparisons of chunking strategies have mainly focused on retrieval performance, leaving operational trade-offs underexplored. To address these issues, we evaluate eight representative chunking strategies across two scalable corpora, three embedding models, and multiple corpus sizes, measuring both retrieval effectiveness and system-level costs. Our results show that computationally expensive methods rarely provide consistent gains over simpler chunking. Instead, the best performing strategy depends on the embedding model, dataset, corpus size, and target retrieval metric. Methods with similar pe
    
[^26]: 安全，还是仅仅能力？智能体安全基准的效度审计

    Safety, or Just Capability? A Validity Audit of Agent-Safety Benchmarks

    [https://arxiv.org/abs/2607.28685](https://arxiv.org/abs/2607.28685)

    本文对四个主流智能体安全基准进行效度审计，发现其评估指标存在系统性缺陷（如恒判为正的策略即可获得高分）、不同基准对同一批模型的排名互相矛盾，且安全基准之间或安全与能力之间表观的负相关很可能只是小样本导致的统计伪象。

    

    智能体安全基准各自测量的是不同的行为，但它们的分数却经常被交替引用为智能体的“安全性”。我们将其中四个基准视为待验证的测量工具，在多达22个模型上使用各基准的官方实现和作者提供的评分器进行运行，并在统一协议下自行测量MMLU和GPQA作为能力综合指标。首要问题在于评估指标本身：在任何以 $F_1$ 评分的二值轨迹判断基准上，“恒判为正”的策略都能达到 $F_1 = 2\pi/(1+\pi)$；在R-Judge上该值为0.690，高于21个实际具有区分能力的模型中的五个。进一步地，三个覆盖面较广的基准对同一批18个模型给出了不同的排名，而这种不一致背后的权衡其实是小样本面板造成的伪象：R-Judge特异性与AgentHarm安全性在 $n=7$ 时相关系数为 $-0.64$，而在 $n=18$ 时仅为 $+0.02$，且约四分之一的随机规模为7的子集在该值附近达到 $|\rho| \geq 0.5$。

    arXiv:2607.28685v2 Announce Type: replace  Abstract: Agent-safety benchmarks measure different behaviors, and their scores get quoted interchangeably as an agent's safety. We treat four of them (R-Judge, InjecAgent, AgentHarm, AgentDojo) as measurements to be validated, running each under its official implementation and author-provided scorer on up to 22 models, with MMLU and GPQA measured by us under one protocol as a capability composite. The metric is the first problem. On any binary trace-judgment benchmark scored by $F_1$, an ``always positive'' policy attains $F_1 = 2\pi/(1+\pi)$; on R-Judge that is $0.690$, above five of the 21 models that actually discriminate. The three broad-coverage benchmarks then rank the same 18 models differently, and the trade-off behind that disagreement is a small-panel artifact: R-Judge specificity against AgentHarm safety correlates $-0.64$ at $n{=}7$ and $+0.02$ at $n{=}18$, and a quarter of random size-7 subsets reach $|\rho| \geq 0.5$ around that
    
[^27]: IMFuse：面向LLM增强序列推荐的实例感知多层融合

    IMFuse: Instance-Aware Multi-Layer Fusion for LLM-Enhanced Sequential Recommendation

    [https://arxiv.org/abs/2607.27002](https://arxiv.org/abs/2607.27002)

    提出IMFuse实例感知多层融合策略，自适应地为每个物品融合LLM多个层的嵌入语义，克服了仅依赖最后一层嵌入存在谱坍缩、信息不足的问题，从而增强序列推荐性能。

    

    利用大型语言模型（LLM）以文本语义丰富物品嵌入，已成为增强序列推荐（SR）的主流范式。然而，现有方法通常仅依赖LLM的最后一层嵌入，其他层中编码的语义在很大程度上未被利用。通过实证分析，我们发现了三个关键结论：1）来自LLM更深层的物品嵌入会出现日益严重的谱坍缩（spectral collapse），这降低了其信息丰富度并阻碍了优化过程；2）LLM的中间层编码了超出最后一层的互补性物品语义；3）不同物品呈现出异构的层级语义模式。这些发现表明，仅使用最后一层是不够的，应当为每个物品自适应地利用多层的LLM语义。为此，我们提出了IMFuse，一种新颖的实例感知多层融合策略，用于LLM增强的序列推荐。

    arXiv:2607.27002v2 Announce Type: replace  Abstract: Leveraging large language models (LLMs) to enrich item embeddings with textual semantics has become a prevailing paradigm for enhancing sequential recommendation (SR). However, existing methods typically rely solely on the final-layer LLM embeddings, leaving the semantics encoded in other layers largely unexploited. Through an empirical analysis, we uncover three key findings: 1) item embeddings from deeper LLM layers suffer from increasingly severe spectral collapse, which reduces their information richness and hinders the optimization; 2) intermediate LLM layers encode complementary item semantics beyond the final layer; 3) different items exhibit heterogeneous layer-wise semantic patterns. These findings suggest that the final layer alone is insufficient and that multi-layer LLM semantics should be exploited adaptively for each item.   To this end, we propose IMFuse, a novel Instance-Aware MultiLayer Fusion strategy for LLM-enhanc
    
[^28]: 基于结构化定位的驾驶视频复杂查询检索

    Driving Video Retrieval for Complex Queries with Structured Grounding

    [https://arxiv.org/abs/2606.09109](https://arxiv.org/abs/2606.09109)

    提出STRIVE-D数据校准检索框架，通过弱标注视频评估并自适应调整查询规则，将校准后的规则分数与视觉-语言及关键词检索信号融合，从而在驾驶视频中可靠检索加塞、急刹等复杂动态事件。

    

    大规模视频检索是自动驾驶数据整理与安全验证的核心任务，用户不仅希望检索到场景，还希望找到诸如加塞、急刹车等动态事件。现有的视觉-语言检索与基于关键词的检索方法常常会遗漏这些事件，原因在于相关运动可能未在文本中被明确描述，或无法通过词汇重叠被捕获。基于规则的检索能够更直接地编码这类事件，但它较为脆弱：当生成或手写的规则与真实驾驶数据的假设不匹配时往往会失效。我们提出了STRIVE-D，一个面向驾驶视频的数据校准检索框架。该框架利用弱标注的域内视频来估计查询规则何时可靠，对与观测数据不匹配的规则进行自适应调整，并将校准后的规则分数与视觉-语言检索及关键词检索信号相融合。在三个驾驶基准上的实验（包括新发布的Dr...（摘要原文在此处被截断）

    arXiv:2606.09109v3 Announce Type: replace-cross  Abstract: Video retrieval at scale is central to data curation and safety validation in autonomous driving, where users want to find not only scenes but also dynamic events such as cut-ins and hard braking. Existing vision-language and keyword-based retrieval methods often miss these events because the relevant motion may not be explicitly described in text or captured by lexical overlap. Rule-based retrieval can encode such events more directly, but it is brittle: generated or hand-written rules often fail when their assumptions do not match real driving data. We propose STRIVE-D, a data-calibrated retrieval framework for driving videos. It uses weakly labeled in-domain videos to estimate when a query rule is reliable, adapt rules that mismatch observed data, and fuse calibrated rule scores with vision-language and keyword-based retrieval signals. Across three driving benchmarks, including newly released human-annotated event data on Dr
    
[^29]: 算法权威与临床诊疗标准

    Algorithmic Authority and the Clinical Standard of Care

    [https://arxiv.org/abs/2606.00044](https://arxiv.org/abs/2606.00044)

    本文借助“代码即法律”框架论证临床AI系统已成为事实上的医疗监管，并提出将AI与医生视为单一责任诊断实体的辩证诊疗标准，从而对AI幻觉与人类认知失误这两类失败模式进行统一治理。

    

    将人工智能整合入临床医学，在算法概率推理与专家医生的经验直觉之间造成了根本性的张力。本文运用劳伦斯·莱斯格的“代码即法律”框架，论证临床AI系统的架构已经在发挥着事实上的医疗监管功能，重塑了责任归属与诊疗标准。本文将AI“幻觉”重新框定为与已被充分记录的人类认知失误（如确认偏误和过早的诊断闭合）在结构上相类似的现象，并指出这两种失败模式都需要统一的治理回应。因此，本文提出一种辩证的诊疗标准，将整合的“AI-医生”二元体视为单一的责任诊断主体，要求在健全的数据治理和患者隐私框架内，实现算法精确性与人类解释权威的综合。

    arXiv:2606.00044v1 Announce Type: cross  Abstract: The integration of artificial intelligence into clinical medicine creates a fundamental tension between algorithmic probabilistic reasoning and the experiential intuition of expert physicians; applying Lawrence Lessig's \enquote{Code is Law} framework, I argue that the architecture of clinical AI systems already functions as de facto medical regulation, reshaping liability and the standard of care. Reframing AI \enquote{hallucination} as structurally analogous to well-documented human cognitive failures such as confirmation bias and premature diagnostic closure, I show that both failure modes demand a unified governance response. I therefore propose a dialectical standard of care that treats the integrated AI-physician dyad as the singular responsible diagnostic entity, mandating the synthesis of algorithmic precision with human interpretive authority within robust data governance and patient privacy frameworks.
    
[^30]: D-基算法中超图对偶化的空间高效实现

    Space efficient implementation of hypergraph dualization in the D-basis algorithm

    [https://arxiv.org/abs/2512.06988](https://arxiv.org/abs/2512.06988)

    提出了一种名为 Small Space 的 D-基算法新实现，借助反向搜索的超图对偶化方法在不存储全部蕴涵关系的情况下逐步计算属性频率，大幅降低了数据分析中的内存占用。

    

    我们提出了一种名为 Small Space 的 D-基（D-basis）算法新实现，可显著降低该算法在数据分析应用中的内存使用量。先前的实现会输出在输入二值表属性集上成立的全部蕴涵关系；而在新版本中，唯一的输出是与固定后件属性相对应的、出现在 D-基蕴涵前件中的属性频率。在过去十年中，在应用 D-基的数据集分析中，这类频率（而非蕴涵关系本身）已成为主要关注点。D-基采用超图对偶化算法，而一种称为反向搜索的对偶化实现允许逐步计算这些频率，而无需存储所有已发现的蕴涵关系。我们通过对比实验证明了 Small Space 实现的有效性。

    arXiv:2512.06988v2 Announce Type: replace-cross  Abstract: We present a new implementation of the $D$-basis algorithm called the Small Space which considerably reduces the algorithm's memory usage for data analysis applications. The previous implementation delivers the complete set of implications that hold on the set of attributes of an input binary table. In the new version, the only output is the frequencies of attributes that appear in the antecedents of implications from the $D$-basis, with a fixed consequent attribute. Such frequencies, rather than the implications themselves, became the primary focus in analysis of datasets where the $D$-basis has been applied over the last decade. The $D$-basis employs a hypergraph dualization algorithm, and a dualization implementation known as Reverse Search allows for the gradual computation of frequencies without the need for storing all discovered implications. We demonstrate the effectiveness of the Small Space implementation by comparing
    
[^31]: 当评论稀缺时：现实稀疏评论场景下感知结构的评论补全推荐方法

    When Reviews Are Scarce: Structure-Aware Review Imputation for Recommendation in Realistic Sparse Review Settings

    [https://arxiv.org/abs/2508.01128](https://arxiv.org/abs/2508.01128)

    提出TWISTER统一框架，通过在文本边图上联合建模语义与结构信号来合成缺失评论，从而在用户很少写评论的现实稀疏场景下提升评论感知推荐系统的性能。

    

    文本评论为推荐系统提供了细粒度的偏好信号和更强的可解释性。然而，在现实场景中，用户很少留下评论，导致严重的稀疏性，削弱了部分可观测条件下评论感知推荐器的有效性。一个自然的补救方法是合成缺失的评论，以恢复这些推荐器所依赖的文本信号。然而，传统的补全技术，如矩阵补全和基于大语言模型的增强方法，要么在将文本嵌入为向量时丢失上下文语义，要么忽视了用户-物品图上的结构依赖关系。为了解决这些缺陷，我们提出了TWISTER（ToWards Imputation on Sparsity with Textual Edge Graph Representation），这是一个用于稀疏文本推荐的统一框架，通过联合建模语义信号和结构信号来合成缺失评论。具体而言，我们将用户-物品交互表示为……

    arXiv:2508.01128v2 Announce Type: replace-cross  Abstract: Textual reviews enrich recommender systems with fine-grained preference signals and enhanced explainability. However, in real-world scenarios, users rarely leave reviews, resulting in severe sparsity that undermines the effectiveness of review-aware recommenders under partial observability. A natural remedy is to synthesize the missing reviews to restore the textual signal these recommenders rely on. Yet, conventional imputation techniques, such as matrix completion and LLM-based augmentation, either lose contextualized semantics by embedding texts into vectors or overlook structural dependencies on the user-item graph. To address these shortcomings, we propose TWISTER (ToWards Imputation on Sparsity with Textual Edge Graph Representation), a unified framework for sparse-text recommendation that synthesizes missing reviews by jointly modeling semantic and structural signals. Specifically, we represent user-item interactions as 
    

