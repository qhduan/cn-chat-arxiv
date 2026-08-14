# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Structure then Query: Enabling Precise Analytical Queries over Unstructured Documents](https://arxiv.org/abs/2608.13384) | 本文提出了AnnoIndex系统，通过自动生成分层注释模式并提取结构化值，将非结构化文档转化为可查询的结构化索引，从而克服了传统向量模糊匹配的局限性，实现了精确的分析查询。 |
| [^2] | [When Should Multi-Round RAG Stop? Structured Stopping Judgments and Retrieval Reduction in Search-R1](https://arxiv.org/abs/2608.13237) | 本文提出了一种基于结构化充分性与差距判断的轻量级停止策略，在不改变Search-R1核心组件的情况下，通过冻结判断器有效减少多轮RAG的检索次数，同时仅带来极小的答案精确度损失。 |
| [^3] | [GEM: A Generative Embedding Model Bridging Reasoning and Retrieval](https://arxiv.org/abs/2608.13200) | GEM通过统一生成与嵌入，在检索前显式推理用户意图，显著提升了推理密集型任务的检索性能。 |
| [^4] | [RAGSieve: Self-Referenced Local Contrast for Knowledge-Poison Detection in Retrieval-Augmented Generation](https://arxiv.org/abs/2608.13010) | RAGSieve通过自参照局部对比机制，无需外部参考即可高效检测检索增强生成中的知识投毒攻击，显著优于现有方法。 |
| [^5] | [EviReform: Evidence-Guided Query Reformulation for Multi-Hop Graph Retrieval](https://arxiv.org/abs/2608.13006) | EviReform通过将检索请求修订与图证据聚合分离，并利用残差查询和归一化信号传播，显著提升了多跳图检索的性能，在多个基准上超越了现有最强方法。 |
| [^6] | [HybridRAG-BN: A Retrieval-Augmented Framework with Fine-Tuned Verification for Bangla KBQA](https://arxiv.org/abs/2608.13004) | 该论文提出了HybridRAG-BN，一个结合混合检索、生成模型与微调验证的检索增强框架，专门针对孟加拉语知识库问答，通过后处理策略显著提升了低资源语言下的答案准确性与鲁棒性。 |
| [^7] | [Generative Universal Multimodal Retrieval with Dual-role Identifiers](https://arxiv.org/abs/2608.12987) | 本文提出DrIG框架，通过双角色标识符统一处理多模态检索任务，同时解决解码脆弱性、单模态限制和精度不足的问题。 |
| [^8] | [STAR: Structured Tokenization and Target-Aware Interest Representation for PCVR Prediction](https://arxiv.org/abs/2608.12986) | STAR通过结构化分词和目标感知兴趣表示，结合高基数信号恢复与对比学习，解决了PCVR预测中特征异质性和训练-推理不一致性问题。 |
| [^9] | [DTAMLP: Denoise Time-aware MLP for Session-based Recommendation](https://arxiv.org/abs/2608.12975) | 本文提出DTAMLP模型，通过即插即用的权重融合模块减少会话推荐中的零星点击噪声，并解释频域滤波的效用，从而在几乎不改变架构的情况下提升推荐准确率。 |
| [^10] | [FSGR: Mitigating Token Frequency Bias for Fair SID-Based Generative Recommendation](https://arxiv.org/abs/2608.12845) | 本文提出FSGR方法，针对基于SID的生成式推荐中因令牌频率偏差导致的公平性问题，通过缓解高频令牌过度预测和低频令牌预测不足，实现物品类别间的公平曝光。 |
| [^11] | [Query Translation vs. Cross-Lingual Embeddings for Sinhala-Tamil E-Government Information Retrieval](https://arxiv.org/abs/2608.12820) | 本文通过对比查询翻译和跨语言嵌入方法，发现BGE-M3在僧伽罗语-泰米尔语到英语的电子政务信息检索中取得了最高性能（Recall@15达96.2%和95.6%），显著优于单语基线。 |
| [^12] | [A Comprehensive Empirical Evaluation of Vector Database Systems for Approximate Nearest Neighbor Search: Performance, Quality, and Resource Trade-offs](https://arxiv.org/abs/2608.12812) | 该论文首次对七个主流向量数据库系统在六个数据集上进行了系统性的多维度实证比较，揭示了检索质量、查询性能和资源消耗之间的关键权衡，为实际系统选型提供了全面参考。 |
| [^13] | [CRAFT: LLM-Based Iterative Refinement for Temporal Reasoning over Clinical Narratives](https://arxiv.org/abs/2608.12779) | CRAFT提出了一种结合生成器和约束验证器的大语言模型迭代优化框架，能够从锚点稀疏的临床叙事中自动重建结构化症状时间线，并在新基准MedTempo上显著优于现有方法。 |
| [^14] | [DrEM: Dual-Side Robust Ensemble Ranking from Noisy User Preference Predictions in Video Recommendation](https://arxiv.org/abs/2608.12778) | 本文提出DrEM方法，首次从监督侧和特征侧双侧处理视频推荐集成排序中的用户偏好预测噪声，通过鲁棒学习机制提升排序稳定性。 |
| [^15] | [Knowledge Synthesis Review Framework: Task-Level Benchmarking of LLM-Based Systems for Multi-Source Evidence Synthesis](https://arxiv.org/abs/2608.12741) | 该论文提出了一个名为KSR的人机协作框架，将证据综合分解为四个任务并逐项基准测试LLM系统，确保在专家验证下选择最优系统，以提高多源证据综合的可靠性和效率。 |
| [^16] | [Attribute-Conditioned Multimodal Slot Factorization for Controllable Fashion Retrieval](https://arxiv.org/abs/2608.12570) | 本文提出MM-slotgate多模态槽位编码器，将时尚检索嵌入分解为四个可独立控制的属性槽位，并通过文本-图像门控机制实现属性级别的可控检索，在H&M数据集上取得了显著性能提升。 |
| [^17] | [Test-Time Optimization of Query Embeddings with Ranking Aware Reward Maximization](https://arxiv.org/abs/2608.12569) | 提出TTT-Embed框架，通过测试时优化冻结模型输出嵌入空间中的轻量级向量，从排名奖励中蒸馏知识，无需权重访问或索引修改，实现奖励的可重用与特异性平衡。 |
| [^18] | [MASCOT: Model-Aware Submodular Coverage for Composite-Attribute Text-to-Image Retrieval](https://arxiv.org/abs/2608.12532) | MASCOT提出了一种模型感知的子模覆盖方法，替代流形排斥，以在复合属性文本到图像检索中实现精确的多样化控制，并克服早期排名召回率的退化问题。 |
| [^19] | [MindMemOS: A Portable and Self-Evolving Memory Operating Layer for AI Agents](https://arxiv.org/abs/2608.12428) | 本文提出MindMemOS，一种可移植且自进化的记忆操作系统层，通过统一实体属性时间结构、验证驱动的进化搜索和“梦境”巩固机制，实现AI代理记忆的自适应建模、模式发现和持续技能进化。 |
| [^20] | [Sci-Surf: Navigating Scientific Literature Discovery through Human Feedback and Intelligent Summarizatio](https://arxiv.org/abs/2608.11973) | Sci-Surf通过集成人类反馈驱动的个性化推荐和基于LLM的多模态博客式论文摘要，实现了意图中心的科学文献发现，显著提升了推荐与理解质量。 |
| [^21] | [Do LLM Recommenders Know When They're Hallucinating? Auditing Confidence Calibration in Catalog Faithfulness](https://arxiv.org/abs/2608.10008) | 本文首次联合审计了多个LLM推荐器的幻觉率和置信度校准，发现目录成员资格测量方法显著影响结果，并验证了更准确的评估工具，揭示了推荐器在输出目录外项目时过度自信的问题。 |
| [^22] | [DREAM Technical Report](https://arxiv.org/abs/2608.09408) | DREAM通过添加一个感知感知、可编排且可审计的策略层，在不替换现有流水线的情况下，解决了工业推荐系统中会话级意图转换未被充分处理的问题，并显著减少了报告量。 |
| [^23] | [Teacher Retains Full Tokens, Student Merges Efficiently: TM20K for E-Commerce Sequence Modeling in Ad Recommendation](https://arxiv.org/abs/2608.07055) | 本文提出一种结合全Transformer和两阶段知识蒸馏的框架，在保留完整令牌的教师模型与高效融合的学生模型之间实现平衡，从而在电商广告推荐中高效处理超长行为序列。 |
| [^24] | [omni-macos: On-Device Omni-Modal Search on Apple Silicon](https://arxiv.org/abs/2608.05543) | omni-macos在苹果设备上实现全模态本地搜索，通过内存预算管理和高效编码策略，确保所有数据不出设备。 |
| [^25] | [One prompt is not enough: Instruction Sensitivity Undermines Embedding Model Evaluation](https://arxiv.org/abs/2605.22544) | 研究表明，指令嵌入模型的单提示评估不足以反映性能，因为提示措辞的敏感性会导致排名不稳定，基准测试需引入提示鲁棒性。 |
| [^26] | [DiffGRM: Diffusion-based Generative Recommendation Model](https://arxiv.org/abs/2510.21805) | DiffGRM通过引入掩蔽离散扩散模型替代自回归解码器，解决了语义ID的因果路径限制和数字间训练不平衡问题，提升了生成式推荐的准确性和鲁棒性。 |
| [^27] | [How Significant Are the Real Performance Gains? An Unbiased Evaluation Framework for GraphRAG](https://arxiv.org/abs/2506.06331) | 本文提出了一种无偏评估框架，通过图文本接地问题生成和消除LLM评估偏差，发现现有GraphRAG方法的实际性能提升远低于先前报告的水平。 |

# 详细

[^1]: 先结构化再查询：实现对非结构化文档的精确分析查询

    Structure then Query: Enabling Precise Analytical Queries over Unstructured Documents

    [https://arxiv.org/abs/2608.13384](https://arxiv.org/abs/2608.13384)

    本文提出了AnnoIndex系统，通过自动生成分层注释模式并提取结构化值，将非结构化文档转化为可查询的结构化索引，从而克服了传统向量模糊匹配的局限性，实现了精确的分析查询。

    

    非结构化文档构成了企业和网络数据的主体。随着大型语言模型（LLMs）的快速发展，研究人员开始构建像操作数据库一样分析非结构化文本数据的数据系统。然而，由于主流检索方法仍依赖于基于向量相似性的模糊匹配，精确获取信息并进行结构化分析和推理仍然是一个重大挑战。为了解决这些局限性，AnnoIndex引入了两个核心基础组件。第一个是注释索引。该系统使用一个名为SchemaLoop的模块，从原始语料库中自动创建分层注释模式，然后使用轻量级语言模型提取特定值。它将分散的非结构化文本转化为物化的、结构化的索引，从而实现低成本的过滤和查询。注释索引避免了向量黑盒匹配的缺陷。

    arXiv:2608.13384v1 Announce Type: new  Abstract: Unstructured documents constitute the majority of enterprise and web data. With the rapid development of large language models(LLMs), researchers have started to build data systems that analyze unstructured textual documents like operating on databases. However, because mainstream retrieval methods still relies on fuzzy matching based on vector similarity, accurately obtaining information and performing structured analysis and reasoning remains a major challenge. To address these limitations, AnnoIndex introduces two core fundamental components. The first is Annotation Index. The system uses a module called SchemaLoop to automatically create hierarchical annotation schemas from the raw corpus, and then uses lightweight language model to extract specific values. It turns scattered unstructured text into a materialized, structured index that enables low-cost filtering and querying. The annotation index avoids the black-box matching of vect
    
[^2]: 多轮RAG何时应停止？Search-R1中的结构化停止判断与检索减少

    When Should Multi-Round RAG Stop? Structured Stopping Judgments and Retrieval Reduction in Search-R1

    [https://arxiv.org/abs/2608.13237](https://arxiv.org/abs/2608.13237)

    本文提出了一种基于结构化充分性与差距判断的轻量级停止策略，在不改变Search-R1核心组件的情况下，通过冻结判断器有效减少多轮RAG的检索次数，同时仅带来极小的答案精确度损失。

    

    arXiv:2608.13237v1 公告类型：交叉 摘要：多轮检索增强生成（RAG）必须在证据累积过程中决定何时停止搜索。由于部署的策略由每条轨迹上的第一个STOP决定，这是一个顺序选择问题，而非独立的状态分类任务。我们将S2G-RAG的结构化充分性与差距判断适配到冻结的Search-R1流程中，并在来自900个不重叠HotpotQA问题的3,009个状态上训练了一个Qwen3.5-2B判断器。Search-R1的推理器、检索器、语料库、提示词和搜索预算保持不变，而判断器检查点和停止阈值在分组验证上选择，并在确认性评估前冻结。在确认性测试集上，所得策略相对于原生Search-R1将检索调用减少了77次（3.70%），而官方精确匹配下降了0.625个百分点。因此，训练后的S2G风格结构化判断器在广泛保持答案质量的同时减少了检索。

    arXiv:2608.13237v1 Announce Type: cross  Abstract: Multi-round retrieval-augmented generation (RAG) must decide when to stop searching as evidence accumulates. Because the deployed policy is determined by the first STOP on each trajectory, this is a sequential selection problem rather than an independent state-classification task. We adapt S2G-RAG's structured sufficiency-and-gap judgment to a frozen Search-R1 pipeline and train a Qwen3.5-2B judge on 3,009 states from 900 disjoint HotpotQA questions. Search-R1's reasoner, retriever, corpus, prompt, and search budget remain unchanged, while the judge checkpoint and stopping threshold are selected on grouped validation and frozen before confirmatory evaluation. On the confirmatory test set, the resulting policy reduces retrieval calls by 77 (3.70\%) relative to Native Search-R1, while Official Exact Match decreases by 0.625 percentage points. Thus, the trained S2G-style structured judge reduces retrieval while broadly preserving answer a
    
[^3]: GEM：一种弥合推理与检索的生成式嵌入模型

    GEM: A Generative Embedding Model Bridging Reasoning and Retrieval

    [https://arxiv.org/abs/2608.13200](https://arxiv.org/abs/2608.13200)

    GEM通过统一生成与嵌入，在检索前显式推理用户意图，显著提升了推理密集型任务的检索性能。

    

    arXiv:2608.13200v1 公告类型：交叉 摘要：现代大型语言模型在推理和指令遵循方面表现出色，使用户能够表达复杂多样的信息需求。然而，传统检索器主要依赖于查询和文档之间的表面匹配，导致用户表达需求的方式与检索器解读方式之间的差距日益扩大。在本文中，我们提出了GEM，一种生成式嵌入模型，通过显式推理用户意图和相关性标准，利用自身知识增强检索。GEM在单一模型中统一了生成和嵌入功能：它首先对查询进行推理，然后附加一个嵌入标记，以编码用于检索的增强上下文。在推理密集型和指令遵循型检索任务上的评估表明，GEM展示了其推理增强检索的有效性，优于其非推理变体，并匹配了使用更大模型的基线性能。此外，GEM的生成式嵌入能力使其能够灵活适应多样化的查询表达。

    arXiv:2608.13200v1 Announce Type: cross  Abstract: Modern LLMs excel at reasoning and instruction following, enabling users to express complex and diverse information needs. However, conventional retrievers largely rely on surface-level matching between queries and documents, resulting in a growing gap between how users express their needs and how retrievers interpret them. In this paper, we present GEM, a generative embedding model that augments retrieval through its own knowledge by explicitly reasoning about user intent and relevance criteria. GEM unifies generation and embedding within a single model: it first reasons over the query, then appends an embedding token to encode the enriched context for retrieval. \zhili{Evaluated on reasoning-intensive and instruction-following retrieval tasks, GEM demonstrates the effectiveness of its reasoning-augmented retrieval, outperforming its non-reasoning variant and matching baselines using substantially larger models.} Furthermore, GEM's ge
    
[^4]: RAGSieve：检索增强生成中知识投毒检测的自参照局部对比方法

    RAGSieve: Self-Referenced Local Contrast for Knowledge-Poison Detection in Retrieval-Augmented Generation

    [https://arxiv.org/abs/2608.13010](https://arxiv.org/abs/2608.13010)

    RAGSieve通过自参照局部对比机制，无需外部参考即可高效检测检索增强生成中的知识投毒攻击，显著优于现有方法。

    

    检索增强生成将外部语料库视为推理证据，使得注入的文档能够推广攻击者选择的主张。现有检测器依赖于可信参考、特定攻击痕迹或对语料库拓扑结构敏感的全局阈值。我们提出RAGSieve，一种自参照检测框架，其从被检测系统自身构建参考。RAGSieve-Query（RSQ）执行查询局部对比，将前五名候选与同一检索中的第6至20名进行评分比较，以检测答案锚点集中度和载体转换。RAGSieve-Graph（RSG）执行语料库局部对比，将每个文档的语义相似但词汇不同的邻居与其局部基线进行比较，以在查询到来前检测协调密度。在三个问答数据集和六种投毒构造上，RSQ实现了95.2%的AUROC，并在5%干净文档移除下检测出82.2%的投毒，而GMTP分别为81.1%/52.5%。

    arXiv:2608.13010v1 Announce Type: new  Abstract: Retrieval-augmented generation treats an external corpus as inference evidence, allowing injected documents to promote attacker-chosen claims. Existing detectors depend on trusted references, specific attack artifacts, or global thresholds sensitive to corpus topology. We present RAGSieve, a self-referenced detection framework that constructs its reference from the inspected system. RAGSieve-Query (RSQ) performs query-local contrast, scoring top-five candidates against ranks 6-20 of the same retrieval to detect answer-anchor concentration and carrier transitions. RAGSieve-Graph (RSG) performs corpus-local contrast, comparing each document's semantically similar but lexically distinct neighbors with its local baseline to detect coordinated density before queries arrive. Across three QA datasets and six poisoning constructions, RSQ achieves 95.2% AUROC and detects 82.2% of poison at 5% clean-document removal, versus 81.1%/52.5% for GMTP. R
    
[^5]: EviReform：面向多跳图检索的证据引导查询重构

    EviReform: Evidence-Guided Query Reformulation for Multi-Hop Graph Retrieval

    [https://arxiv.org/abs/2608.13006](https://arxiv.org/abs/2608.13006)

    EviReform通过将检索请求修订与图证据聚合分离，并利用残差查询和归一化信号传播，显著提升了多跳图检索的性能，在多个基准上超越了现有最强方法。

    

    多跳检索必须恢复能够共同提供充分证据的段落。初始段落通常解决了问题中隐含的实体或关系，使得缺失的证据只有在检索开始后才更容易描述。图检索通过存储的语料库结构改善了对相关证据的访问，但其检索信号通常来源于原始问题。因此，即使观察到的段落提供了更直接的语义线索，互补证据也必须通过存储的关系来获取。我们引入了EviReform，它将检索请求的修订与图内证据的聚合分离开来。检索到的源段落为未解决的信息需求制定残差查询。原始和残差检索信号分别归一化、组合，并在共享实体的命题之间传播。在2WikiMultiHopQA、HotpotQA和MuSiQue上，EviReform超越了最强的基线方法。

    arXiv:2608.13006v1 Announce Type: new  Abstract: Multi-hop retrieval must recover passages that provide sufficient evidence together. An initial passage often resolves an entity or relation implicit in the question, making the missing evidence easier to describe only after retrieval begins. Graph retrieval improves access to related evidence through stored corpus structure, but its retrieval signal is commonly derived from the original question. Complementary evidence must then be reached through stored relations even when an observed passage provides a more direct semantic cue. We introduce EviReform, which separates revising the retrieval request from aggregating evidence in the graph. Retrieved source passages formulate residual queries for the unresolved information need. The original and residual retrieval signals are normalized separately, combined, and propagated between propositions that share entities. On 2WikiMultiHopQA, HotpotQA, and MuSiQue, EviReform exceeds the strongest 
    
[^6]: HybridRAG-BN：一种带有微调验证的孟加拉语知识库问答检索增强框架

    HybridRAG-BN: A Retrieval-Augmented Framework with Fine-Tuned Verification for Bangla KBQA

    [https://arxiv.org/abs/2608.13004](https://arxiv.org/abs/2608.13004)

    该论文提出了HybridRAG-BN，一个结合混合检索、生成模型与微调验证的检索增强框架，专门针对孟加拉语知识库问答，通过后处理策略显著提升了低资源语言下的答案准确性与鲁棒性。

    

    arXiv:2608.13004v1 公告类型：新论文  摘要：知识库问答（KBQA）系统依赖于有效的检索和推理机制，以从外部知识源生成准确答案。然而，对于孟加拉语等低资源语言，开发可靠的KBQA系统仍面临挑战，原因包括检索相关研究有限、语言资源稀缺，以及难以将生成的回答锚定在外部知识上。在本工作中，我们提出了HybridRAG-BN，一种面向孟加拉语KBQA的检索增强框架，它集成了使用BM25和BGE-M3的混合检索、使用GGUF版本的Gemma-4-31B-Instruct进行答案生成，以及一个LoRA微调的Gemma-4-31B-Instruct模型用于答案验证和细化。为进一步提升鲁棒性，该框架包含一个后处理阶段，通过回退答案替换和DuckDuckGo辅助检索来处理未解决的情况。实验结果表明了该框架的有效性。

    arXiv:2608.13004v1 Announce Type: new  Abstract: Knowledge-base question answering (KBQA) systems rely on effective retrieval and reasoning mechanisms to generate accurate answers from external knowledge sources. However, developing reliable KBQA systems for low-resource languages such as Bangla remains challenging due to limited retrieval-focused research, scarce language resources, and difficulties in grounding generated responses in external knowledge. In this work, we propose HybridRAG-BN, a retrieval-augmented framework for Bangla KBQA that integrates hybrid retrieval using BM25 and BGE-M3, answer generation using the GGUF version of Gemma-4-31B-Instruct, and a LoRA-fine-tuned Gemma-4-31B-Instruct model for answer verification and refinement. To further improve robustness, the framework incorporates a post-processing stage that addresses unresolved cases through fallback answer replacement and DuckDuckGo-assisted retrieval. Experimental results demonstrate the effectiveness of the
    
[^7]: 双角色标识符的生成式通用多模态检索

    Generative Universal Multimodal Retrieval with Dual-role Identifiers

    [https://arxiv.org/abs/2608.12987](https://arxiv.org/abs/2608.12987)

    本文提出DrIG框架，通过双角色标识符统一处理多模态检索任务，同时解决解码脆弱性、单模态限制和精度不足的问题。

    

    生成式信息检索（GIR）通过训练生成器直接生成相关项目的标识符，已成为传统“索引-检索-排序”检索流程的一种有吸引力的替代方案。尽管其前景广阔，但仍存在若干未解决的挑战。首先，受限的左到右解码容易受到前缀级错误和局部最优的影响。其次，先前大多数GIR研究仍主要局限于单模态，导致跨文本、图像及混合图像-文本项目的指令感知检索探索不足。第三，尽管基于离散标识符的GIR提供了更高的效率，但其检索精度仍落后于先进的基于稠密向量的检索方法。基于这些挑战，我们提出了DrIG，一种新颖的生成式框架，用于通用多模态检索，其特点是双角色标识符，支持跨多种模态和领域的多样化检索任务。每个候选项目被赋予双重角色，既作为检索目标又作为上下文标识符，从而增强检索的灵活性和准确性。

    arXiv:2608.12987v1 Announce Type: cross  Abstract: Generative information retrieval (GIR) has emerged as a compelling alternative to the conventional index-retrieve-then-rank retrieval pipeline by training a generator to produce the identifiers of relevant items directly. Despite its promise, a number of open challenges still remain. First, constrained left-to-right decoding is vulnerable to prefix-level errors and local optima. Second, most prior GIR research remains largely unimodal, leaving instruction-aware retrieval across text, image, and mixed image-text items underexplored. Third, although discrete identifier-based GIR offers higher efficiency, its retrieval accuracy still lags behind that of the cutting-edge dense-vector-based retrieval methods. Motivated by these challenges, we propose DrIG, a novel Generative framework for universal multimodal retrieval featuring Dual-role Identifiers, which supports diverse retrieval tasks across multiple modalities and domains. Each candid
    
[^8]: STAR：面向PCVR预测的结构化分词与目标感知兴趣表示

    STAR: Structured Tokenization and Target-Aware Interest Representation for PCVR Prediction

    [https://arxiv.org/abs/2608.12986](https://arxiv.org/abs/2608.12986)

    STAR通过结构化分词和目标感知兴趣表示，结合高基数信号恢复与对比学习，解决了PCVR预测中特征异质性和训练-推理不一致性问题。

    

    摘要：arXiv:2608.12986v1 公告类型：新 摘要：点击后转化率（PCVR）预测是工业推荐系统中的核心排序任务。现代排序模型必须联合捕捉异质的非序列特征、多行为用户序列以及目标物品感知的用户兴趣，同时保持对高基数稀疏特征、缺失值和训练-推理不一致性的鲁棒性。在本文中，我们提出了STAR（结构化分词与目标感知兴趣表示），这是一个面向KDD Cup 2026腾讯UniRec挑战赛的实用框架。STAR在HyFormer风格的多序列骨干之上，结合了结构化特征分词与目标感知兴趣表示。它引入了高基数信号恢复、显式用户-物品交互分词、目标感知序列解码，以及受InfoNCE启发的加权用户-物品对比辅助目标。我们通过重构特征来对齐训练和推理流程。

    arXiv:2608.12986v1 Announce Type: new  Abstract: Post-click conversion rate (PCVR) prediction is a core ranking task in industrial recommender systems. Modern ranking models must jointly capture heterogeneous non-sequential features, multi-behavior user sequences, and target-item-aware user interests, while remaining robust to high-cardinality sparse features, missing values, and train-inference inconsistencies. In this paper, we present STAR (Structured Tokenization and Target-Aware Interest Representation), a practical framework for the KDD Cup 2026 Tencent UniRec Challenge. STAR combines structured feature tokenization with target-aware interest representation on top of a HyFormer-style multi-sequence backbone. It introduces high-cardinality signal recovery, explicit user-item interaction tokens, target-aware sequence decoding, and a weighted user-item contrastive auxiliary objective inspired by InfoNCE. We further align the training and inference pipelines by reconstructing feature
    
[^9]: DTAMLP：面向会话推荐的去噪时间感知MLP

    DTAMLP: Denoise Time-aware MLP for Session-based Recommendation

    [https://arxiv.org/abs/2608.12975](https://arxiv.org/abs/2608.12975)

    本文提出DTAMLP模型，通过即插即用的权重融合模块减少会话推荐中的零星点击噪声，并解释频域滤波的效用，从而在几乎不改变架构的情况下提升推荐准确率。

    

    arXiv:2608.12975v1 公告类型：交叉 摘要：本文报告了关于会话推荐（SBR）的两个实证发现，并将它们统一到一个单一模型DTAMLP中。首先，现有的时间感知和基于图神经网络（GNN）的模型（如TiSASRec、SR-GNN）将每次点击的时间间隔视为同等信息量，尽管极短的停留时间往往反映的是携带很少偏好信号的意外点击——我们称之为零星噪声现象。我们展示了一个轻量级、即插即用的权重融合模块，它将模型的注意力权重与一个阈值限制的时间间隔权重混合，可以插入到这些模型中，几乎无需架构更改，并带来一致的准确率提升；我们认为这是本工作中最直接可验证的贡献。其次，我们重新审视了FMLP-Rec中一个未被充分解释的观察结果，即对物品嵌入进行可学习的频域滤波能提高准确率，并提供了一个可能的解释：时域行为混合了多种纠缠的心理因素。

    arXiv:2608.12975v1 Announce Type: cross  Abstract: This paper reports two empirical findings on session-based recommendation (SBR), unified in a single model, DTAMLP. First, existing time-aware and GNN-based models (e.g., TiSASRec, SR-GNN) treat every click-time interval as equally informative, even though very short dwell times often reflect accidental clicks carrying little preference signal -- a phenomenon we call sporadic noise. We show that a lightweight, plug-and-play weight fusion module, blending a model's attention weight with a threshold-capped time-interval weight, can be inserted into such models with almost no architectural change and yields a consistent accuracy gain; we view this as the most directly verifiable contribution of this work. Second, we revisit an under-explained observation from FMLP-Rec, where a learnable frequency-domain filter on item embeddings improves accuracy, and offer a possible explanation: time-domain behavior mixes several entangled psychological
    
[^10]: FSGR：缓解基于SID的生成式推荐中的令牌频率偏差以实现公平性

    FSGR: Mitigating Token Frequency Bias for Fair SID-Based Generative Recommendation

    [https://arxiv.org/abs/2608.12845](https://arxiv.org/abs/2608.12845)

    本文提出FSGR方法，针对基于SID的生成式推荐中因令牌频率偏差导致的公平性问题，通过缓解高频令牌过度预测和低频令牌预测不足，实现物品类别间的公平曝光。

    

    基于语义ID（SID）的生成式推荐近期取得了显著成功。然而，现有方法存在一个此前被忽视的公平性问题，我们称之为“令牌频率偏差”，即高频SID令牌被系统性过度预测，而低频SID令牌则被预测不足。这种偏差源于SID构建过程中不平衡语义码本的影响，以及推荐训练中流行度偏差与最大似然估计目标共同作用，导致物品类别间的曝光不公平。现有SID方法主要侧重于提升码本质量，却忽视了令牌频率不平衡对下游推荐公平性的影响，而直接应用LLM去偏方法于基于SID的推荐时，由于SID令牌的层次语义，往往效果不佳。为解决此问题，我们提出了FSGR。

    arXiv:2608.12845v1 Announce Type: cross  Abstract: Semantic ID (SID)-based generative recommendation has recently achieved remarkable success. However, existing methods suffer from a previously overlooked fairness issue, which we term \textbf{Token Frequency Bias}, where high-frequency SID tokens are systematically over-predicted while low-frequency SID tokens are under-predicted. This bias originates from the combined effects of imbalanced semantic codebooks during SID construction, and popularity bias together with the maximum likelihood estimation objective during recommendation training, resulting in unfair exposure across item categories. Existing SID methods mainly focus on improving codebook quality and overlook the impact of token frequency imbalance on downstream recommendation fairness, while LLM debiasing methods often yield suboptimal results when directly applied to SID-based recommendation, due to the hierarchical semantics of SID tokens. To address this issue, we propose
    
[^11]: 查询翻译与跨语言嵌入在僧伽罗语-泰米尔语电子政务信息检索中的对比研究

    Query Translation vs. Cross-Lingual Embeddings for Sinhala-Tamil E-Government Information Retrieval

    [https://arxiv.org/abs/2608.12820](https://arxiv.org/abs/2608.12820)

    本文通过对比查询翻译和跨语言嵌入方法，发现BGE-M3在僧伽罗语-泰米尔语到英语的电子政务信息检索中取得了最高性能（Recall@15达96.2%和95.6%），显著优于单语基线。

    

    arXiv:2608.12820v1 公告类型：新 摘要：本文对使用僧伽罗语和泰米尔语查询检索英语政府信息的跨语言信息检索（CLIR）方法进行了比较评估。研究调查了两种CLIR范式：查询翻译（QT），采用Google Translate、NLLB和mBART50；以及跨语言嵌入（CLE），使用LaBSE、多语言E5和BGE-M3，并以单语英语检索作为基线。实验在一个人工验证的基准数据集上进行，该数据集包含500个僧伽罗语、泰米尔语和英语问答对，源自斯里兰卡政府信息中心（GIC）的1,699个分段上下文。检索性能通过Recall@k（k = 1, 3, 5, 10, 15）进行评估。单语检索表现不佳（Recall@15 <10%），而所有CLIR方法均显著提高了检索准确性。其中，BGE-M3取得了最高的Recall@15，在僧伽罗语-英语和泰米尔语-英语检索中分别达到96.2%和95.6%，优于其他方法。

    arXiv:2608.12820v1 Announce Type: new  Abstract: This paper presents a comparative evaluation of cross-lingual information retrieval (CLIR) methods for retrieving English government information using Sinhala and Tamil queries. Two CLIR paradigms are investigated: Query Translation (QT), employing Google Translate, NLLB, and mBART50, and Cross-Lingual Embeddings (CLE), using LaBSE, multilingual E5, and BGE-M3, with monolingual English retrieval as the baseline. Experiments are conducted on a human-verified benchmark comprising 500 Sinhala, Tamil, and English question-answer pairs derived from 1,699 segmented contexts from Sri Lanka's Government Information Center (GIC). Retrieval performance is evaluated using Recall@k (k = 1, 3, 5, 10, 15). Monolingual retrieval performs poorly (Recall@15 <10%), whereas all CLIR approaches substantially improve retrieval accuracy. Among them, BGE-M3 achieves the highest Recall@15, reaching 96.2% for Sinhala-English and 95.6% for Tamil-English, outperfo
    
[^12]: 向量数据库系统用于近似最近邻搜索的全面实证评估：性能、质量与资源权衡

    A Comprehensive Empirical Evaluation of Vector Database Systems for Approximate Nearest Neighbor Search: Performance, Quality, and Resource Trade-offs

    [https://arxiv.org/abs/2608.12812](https://arxiv.org/abs/2608.12812)

    该论文首次对七个主流向量数据库系统在六个数据集上进行了系统性的多维度实证比较，揭示了检索质量、查询性能和资源消耗之间的关键权衡，为实际系统选型提供了全面参考。

    

    向量数据库已成为现代人工智能应用的关键基础设施，特别是检索增强生成（RAG）、语义搜索和推荐系统。尽管其重要性日益增长，但在联合评估检索质量、查询延迟、吞吐量和资源利用方面，仍然缺乏全面且可复现的基准测试。我们对七个主流向量数据库系统进行了系统性实证评估：FAISS、Qdrant、Milvus、Weaviate、Chroma、pgvector 和 LanceDB。我们的方法涵盖六个多样化数据集，从经典计算机视觉描述符（SIFT、GIST）到基于Transformer的文本嵌入（MS MARCO、GloVe），包含超过400万个向量，维度从96到960不等。我们测量了15个指标，涵盖检索质量（Recall@K、Precision@K、MRR、NDCG@K、Hit Rate@K）、查询性能（延迟百分位数、QPS、冷启动延迟）以及资源利用等方面。

    arXiv:2608.12812v1 Announce Type: new  Abstract: Vector databases have emerged as critical infrastructure for modern artificial intelligence applications, particularly retrieval-augmented generation (RAG), semantic search, and recommendation systems. Despite their growing importance, there remains a significant gap in comprehensive, reproducible benchmarks that jointly evaluate retrieval quality, query latency, throughput, and resource utilization. We present a systematic empirical evaluation of seven prominent vector database systems: FAISS, Qdrant, Milvus, Weaviate, Chroma, pgvector, and LanceDB. Our methodology spans six diverse datasets, from classical computer-vision descriptors (SIFT, GIST) to transformer-based text embeddings (MS MARCO, GloVe), encompassing over 4 million vectors at dimensionalities from 96 to 960. We measure 15 metrics spanning retrieval quality (Recall@K, Precision@K, MRR, NDCG@K, Hit Rate@K), query performance (latency percentiles, QPS, cold-start latency), a
    
[^13]: CRAFT：基于大语言模型的临床叙事时间推理迭代优化框架

    CRAFT: LLM-Based Iterative Refinement for Temporal Reasoning over Clinical Narratives

    [https://arxiv.org/abs/2608.12779](https://arxiv.org/abs/2608.12779)

    CRAFT提出了一种结合生成器和约束验证器的大语言模型迭代优化框架，能够从锚点稀疏的临床叙事中自动重建结构化症状时间线，并在新基准MedTempo上显著优于现有方法。

    

    理解临床叙事中症状的时间进展对于疾病监测、安全监测和因果关系评估至关重要。然而，临床叙事很少提供明确的时间锚点。当前的时间信息推理方法主要集中于对多访视、时间戳丰富的记录进行成对关系分类，而忽略了从锚点稀疏的个体报告中重建结构化症状轨迹的问题。我们提出了CRAFT，这是一个大语言模型框架，它将生成器与基于约束的验证器配对，通过有针对性的反馈迭代地生成并优化分阶段的症状时间线。我们在MedTempo上进行了评估，这是一个包含5,347份跨越三种COVID-19疫苗类型的疫苗不良事件叙事的新基准，其中3,166份报告具有专家验证的时间阶段注释。跨四个大语言模型骨干的实验表明，CRAFT在所有评估指标上均显著优于现有基线，展示了其在锚点稀疏的临床叙事中进行稳健时间推理的能力。

    arXiv:2608.12779v1 Announce Type: cross  Abstract: Understanding the temporal progression of symptoms in clinical narratives is critical for disease monitoring, safety surveillance, and causality assessment. Clinical narratives, however, rarely provide explicit temporal anchors. Current approaches to temporal information reasoning focus predominantly on pairwise relation classification across multi-visit and timestamp-rich records, leaving the reconstruction of structured symptom trajectories from individual anchor-sparse reports largely unaddressed. We propose CRAFT, an LLM framework that pairs a generator with a constraint-based verifier to iteratively produce and refine stage-wise symptom timelines through targeted feedback. We conduct evaluation on MedTempo, a new benchmark of 5,347 vaccine adverse-event narratives spanning three COVID-19 vaccine types, with expert-validated temporal stage annotations for 3,166 reports. Experiments across four LLM backbones demonstrate that CRAFT c
    
[^14]: DrEM：视频推荐中基于噪声用户偏好预测的双侧鲁棒集成排序

    DrEM: Dual-Side Robust Ensemble Ranking from Noisy User Preference Predictions in Video Recommendation

    [https://arxiv.org/abs/2608.12778](https://arxiv.org/abs/2608.12778)

    本文提出DrEM方法，首次从监督侧和特征侧双侧处理视频推荐集成排序中的用户偏好预测噪声，通过鲁棒学习机制提升排序稳定性。

    

    工业视频推荐系统通常采用多阶段架构。在集成排序阶段，来自上游多任务模型的多维用户偏好预测（pxtrs）被融合成一个统一的排序分数，以反映用户满意度。由于用户的真实满意度难以直接观察，集成排序模型通常将pxtrs同时用作输入特征和构建代理偏好的来源。然而，作为上游预测模型的输出，pxtrs不可避免地包含预测噪声，这些噪声会在下游学习中从两侧传播。在监督侧，噪声pxtrs可能翻转代理偏好并引入错误梯度。在特征侧，pxtr噪声可能通过模型输入传播并破坏排序分数的稳定性。现有的集成排序方法通常将pxtrs视为可靠信号，忽视了这种预测噪声。为解决此问题，我们提出了DrEM。

    arXiv:2608.12778v1 Announce Type: new  Abstract: Industrial video recommendation systems typically adopt a multi-stage architecture. At the ensemble ranking stage, multi-dimensional user preference predictions (pxtrs) from an upstream multi-task model are fused into a unified ranking score to reflect user satisfaction. Since users' true satisfaction is difficult to observe directly, ensemble ranking models commonly use pxtrs both as input features and as a source for constructing proxy preferences. However, as outputs of an upstream prediction model, pxtrs inevitably contain prediction noise, which propagates to downstream learning across two sides. On the supervision side, noisy pxtrs may flip proxy preferences and introduce erroneous gradients. On the feature side, pxtr noise may propagate through model inputs and destabilize ranking scores. Existing ensemble ranking methods typically treat pxtrs as reliable signals and overlook such prediction noise. To address this, we propose DrEM
    
[^15]: 知识综合综述框架：基于任务级基准测试的LLM系统用于多源证据综合

    Knowledge Synthesis Review Framework: Task-Level Benchmarking of LLM-Based Systems for Multi-Source Evidence Synthesis

    [https://arxiv.org/abs/2608.12741](https://arxiv.org/abs/2608.12741)

    该论文提出了一个名为KSR的人机协作框架，将证据综合分解为四个任务并逐项基准测试LLM系统，确保在专家验证下选择最优系统，以提高多源证据综合的可靠性和效率。

    

    在快速发展的领域中，证据分散于学术研究、行业报告、政策文件和媒体来源，这些来源在质量、结构和目的上各不相同，使得及时综合变得困难。大型语言模型（LLMs）可能加速这一工作，但它们在综述的不同认知任务中的可靠性仍不确定。我们引入了知识综合综述（KSR），这是一个人在环框架，将证据综合分解为筛选、提取、分析和综合，针对专家参考标准对基于LLM的系统进行每个任务的基准测试，并在持续专家验证下将每个任务路由到表现最佳的系统。我们在一个包含244篇文档的基准子集上评估了GPT-5、Claude Sonnet 4、Gemini 2.5 Pro和NotebookLM，该子集来自一个涵盖四种来源类型的1893篇关于AI和工作的语料库，与具有高评分者间信度（92.2%一致性，kappa）的金标准进行对比。

    arXiv:2608.12741v1 Announce Type: new  Abstract: Evidence in rapidly evolving fields is fragmented across academic studies, industry reports, policy documents, and media sources that differ in quality, structure, and purpose, making timely synthesis difficult. Large language models (LLMs) may accelerate this work, but their reliability across the distinct cognitive tasks of a review remains uncertain. We introduce the Knowledge Synthesis Review (KSR), a human-in-the-loop framework that decomposes evidence synthesis into screening, extraction, analysis, and synthesis, benchmarks LLM-based systems on each task against expert reference standards, and routes each task to the best-performing system under continuous expert validation. We evaluated GPT-5, Claude Sonnet 4, Gemini 2.5 Pro, and NotebookLM on a 244-document benchmark subset drawn from a 1,893-document corpus on AI and work spanning four source types, against a gold standard with high inter-rater reliability (92.2% agreement, kapp
    
[^16]: 属性条件化的多模态槽位分解用于可控时尚检索

    Attribute-Conditioned Multimodal Slot Factorization for Controllable Fashion Retrieval

    [https://arxiv.org/abs/2608.12570](https://arxiv.org/abs/2608.12570)

    本文提出MM-slotgate多模态槽位编码器，将时尚检索嵌入分解为四个可独立控制的属性槽位，并通过文本-图像门控机制实现属性级别的可控检索，在H&M数据集上取得了显著性能提升。

    

    时尚检索通常需要同时满足多个属性，如类别、颜色、图案和人群特征。整体嵌入将这些信号混合到一个单一向量中，使得在检索时难以进行特定属性的控制。许多现有的语义ID方法提供离散的物品编码，但这些编码通常被优化为物品级或残差地址，并未暴露具名且可独立控制的属性槽位。我们引入了MM-slotgate，一种多模态槽位编码器，将Fashion-CLIP文本和图像嵌入分解为四个具名属性槽位。每个槽位学习自己的文本-图像门控，因此视觉锚定的属性（如颜色和图案）可以更多依赖图像证据，而分类导向的属性（如类别和人群特征）则保持更多文本驱动。在H&M数据集上，使用组合的槽位相似度和槽位逻辑检索评分，MM-slotgate实现了0.7566的宏约束性指标。

    arXiv:2608.12570v1 Announce Type: cross  Abstract: Fashion retrieval often requires satisfying multiple attributes at once, such as category, color, pattern, and demographic. Monolithic embeddings mix these signals into a single vector, making attribute-specific control difficult at retrieval time. Many existing semantic-ID methods provide discrete item codes, but these codes are typically optimized as item-level or residual addresses and do not expose named, independently controllable attribute slots.   We introduce MM-slotgate, a multimodal slot encoder that factorizes Fashion-CLIP text and image embeddings into four named attribute slots. Each slot learns its own text-image gate, so visually grounded attributes such as color and pattern can rely more on image evidence, while taxonomy-oriented attributes such as category and demographic can remain more text-driven.   On H&M, using a combined slot-similarity and slot-logit retrieval score, MM-slotgate achieves 0.7566 macro ConstraintS
    
[^17]: 测试时查询嵌入优化与排名感知奖励最大化

    Test-Time Optimization of Query Embeddings with Ranking Aware Reward Maximization

    [https://arxiv.org/abs/2608.12569](https://arxiv.org/abs/2608.12569)

    提出TTT-Embed框架，通过测试时优化冻结模型输出嵌入空间中的轻量级向量，从排名奖励中蒸馏知识，无需权重访问或索引修改，实现奖励的可重用与特异性平衡。

    

    摘要：arXiv:2608.12569v1 公告类型：新 摘要：稠密检索器使用冻结编码器与预计算索引之间的向量相似性对文档进行排序。虽然测试时来自重排序器或LLM评判者的排名奖励可以改善结果，但现有方法在单次查询后会丢弃此信号。更新检索器的权重可使奖励可重用，但这需要参数访问权限，对于闭源模型不可用，且计算代价高昂。我们提出TTT-Embed（测试时嵌入调优），一种将排名奖励蒸馏到冻结模型输出嵌入空间中的轻量级学习向量的框架。该向量仅从检索器自身候选文档的标量排名分数优化，无需访问模型权重、真实标签或修改索引。单个范围参数控制奖励重用（全局、任务或查询），在可重用性与特异性之间实现原则性权衡。

    arXiv:2608.12569v1 Announce Type: new  Abstract: Dense retrievers rank documents using vector similarity between a frozen encoder and a precomputed index. While test-time ranking rewards from a reranker or LLM judge can improve results, existing methods discard this signal after a single query. Updating the retriever's weights makes rewards reusable, but this requires parameter access, which is unavailable for closed-source models, and is computationally prohibitive. We propose TTT-Embed (Test-Time Tuning of Embeddings), a framework that distills ranking rewards into a lightweight, learned vector within the output embedding space of a frozen model. This vector is optimized purely from scalar ranking scores assigned to the retriever's own candidate documents, requiring no access to model weights, ground-truth labels, or modifications to index. A single scope parameter controls rewards reuse (global, task, or query), enabling a principled trade-off between reusability and specificity und
    
[^18]: MASCOT：面向复合属性文本到图像检索的模型感知子模覆盖

    MASCOT: Model-Aware Submodular Coverage for Composite-Attribute Text-to-Image Retrieval

    [https://arxiv.org/abs/2608.12532](https://arxiv.org/abs/2608.12532)

    MASCOT提出了一种模型感知的子模覆盖方法，替代流形排斥，以在复合属性文本到图像检索中实现精确的多样化控制，并克服早期排名召回率的退化问题。

    

    arXiv:2608.12532v1 公告类型：交叉 摘要：视觉语言模型（VLMs）在检索语义相关图像方面非常有效。然而，在实践中，仅靠相关性往往是不够的。系统还必须在地理和时间等复合属性上实现结果多样化（RD），而这一任务中的精确控制仍然具有挑战性。当前的重新排序方法，如多源行列式点过程（MS-DPP），通过基于相似性表示的流形排斥来解决这一问题。尽管这种策略在广泛探索中有效，但它暴露了流形模型的一个关键限制：当面对离散元数据的多样性减少任务时，它们在早期排名召回率上会遭受显著退化。为弥合这一差距，我们引入了MASCOT（复合属性文本到图像检索的模型感知子模覆盖）。MASCOT不依赖流形排斥，而是将多属性多样性表述为资源分配问题。

    arXiv:2608.12532v1 Announce Type: cross  Abstract: Vision-Language Models (VLMs) are highly effective in retrieving semantically relevant images. However, in practice, relevance alone is often insufficient. Systems must also achieve Result Diversification (RD) across composite attributes such as geography and time, a task for which precise control remains challenging. Current re-ranking methods, such as Multi-Source Determinantal Point Processes (MS-DPP), address this using manifold-based repulsion over similarity representations. Although this strategy is effective for broad exploration, it exposes a key limitation in manifold-based models: when subjected to diversity-decrease tasks on discrete metadata, they suffer substantial degradation in early-rank recall.   To bridge this gap, we introduce MASCOT (Model-Aware Submodular Coverage for Composite-Attribute Text-to-Image Retrieval). Instead of relying on manifold repulsion, MASCOT formulates multi-attribute diversity as a resource al
    
[^19]: MindMemOS：一种面向AI代理的可移植且自进化的记忆操作系统层

    MindMemOS: A Portable and Self-Evolving Memory Operating Layer for AI Agents

    [https://arxiv.org/abs/2608.12428](https://arxiv.org/abs/2608.12428)

    本文提出MindMemOS，一种可移植且自进化的记忆操作系统层，通过统一实体属性时间结构、验证驱动的进化搜索和“梦境”巩固机制，实现AI代理记忆的自适应建模、模式发现和持续技能进化。

    

    记忆是AI代理的核心组件，使其能够积累经验、保持个性化并在长期交互中适应。然而，现有记忆系统在开发后往往保持固定，限制了其通过持续使用来适应记忆模型、组织策略和程序性知识的能力。我们提出了MindMemOS，一种可移植且自进化的记忆操作系统层，它使用统一的实体属性时间结构来组织开放世界信息。MindMemOS支持场景自适应记忆建模、高阶模式发现、自主记忆细化和持续技能进化。其MindMemEvolve算法采用验证驱动的进化搜索来优化目标场景的记忆模式，而“梦境”通过合并冗余记录和解决冲突来巩固积累的记忆。此外，隐式纠正反馈作为人在回路中的信号。

    arXiv:2608.12428v1 Announce Type: new  Abstract: Memory is a core component of AI agents, enabling them to accumulate experience, maintain personalization, and adapt over long-term interactions. However, existing memory systems often remain fixed after development, limiting their ability to adapt their memory models, organization strategies, and procedural knowledge through continued use. We present MindMemOS, a portable and self-evolving memory operating layer that organizes open-world information using a unified entity property timestructure. MindMemOS supports scenario-adaptive memory modeling, higher-order pattern discovery, autonomous memory refinement, and continuous skill evolution. Its MindMemEvolve algorithm employs validation-driven evolutionary search to optimize memory schemas for target scenarios, whiledreaming consolidates accumulated memories by merging redundant records and resolving conflicts. In addition, implicit corrective feedback serves as a human-in-the-loop sign
    
[^20]: Sci-Surf：通过人类反馈与智能摘要导航科学文献发现

    Sci-Surf: Navigating Scientific Literature Discovery through Human Feedback and Intelligent Summarizatio

    [https://arxiv.org/abs/2608.11973](https://arxiv.org/abs/2608.11973)

    Sci-Surf通过集成人类反馈驱动的个性化推荐和基于LLM的多模态博客式论文摘要，实现了意图中心的科学文献发现，显著提升了推荐与理解质量。

    

    arXiv:2608.11973v1 公告类型：新 摘要：科学出版物的快速增长使得研究人员越来越难以识别相关的新研究并有效理解它们。现有的学术发现平台通常依赖静态主题订阅或基于嵌入的相似性，仅提供摘要或简短总结，对细致意图建模和深入论文摘要的支持有限。我们提出了Sci-Surf，一个以意图为中心的知识发现系统，它集成了反馈驱动的个性化推荐与多模态博客式论文消化。我们的方法通过基于LLM的用户画像细化用户意图表示，同时生成结构化摘要，综合全文中的文本和视觉信息。该演示展示了一个端到端的学术发现流程，并通过真实用户评估在推荐质量和消化质量方面均显示出可衡量的改进。

    arXiv:2608.11973v1 Announce Type: new  Abstract: The rapid growth of scientific publications makes it increasingly difficult for researchers to identify relevant new studies and effectively comprehend them. Existing academic discovery platforms typically rely on static topic subscriptions or embedding-based similarity and provide only abstracts or short summaries, offering limited support for nuanced intent modeling and in-depth paper summarization. We present Sci-Surf, an intent-centric knowledge discovery system that integrates feedback-driven personalized recommendation with multi-modal blog-style paper digestion. Our approach refines user intent representations through LLM-based user profiling, while generating structured summaries that synthesize textual and visual information from full papers. The demo presents an end-to-end academic discovery pipeline and demonstrates measurable improvements in both recommendation quality and digestion quality through real-user evaluations. Spec
    
[^21]: 大语言模型推荐器是否知道自己在幻觉？目录忠实度中的置信度校准审计

    Do LLM Recommenders Know When They're Hallucinating? Auditing Confidence Calibration in Catalog Faithfulness

    [https://arxiv.org/abs/2608.10008](https://arxiv.org/abs/2608.10008)

    本文首次联合审计了多个LLM推荐器的幻觉率和置信度校准，发现目录成员资格测量方法显著影响结果，并验证了更准确的评估工具，揭示了推荐器在输出目录外项目时过度自信的问题。

    

    arXiv:2608.10008v2 公告类型：交叉替换 摘要：用于Top-K项目推荐的大语言模型（LLM）推荐器经常输出目标目录之外的标题。先前的审计仅报告了二元的域外率，但没有人询问模型是否自知。我们联合审计了来自四个独立供应商（Mistral Large、Llama-3.3-70B、GPT-OSS-120B、Claude Sonnet 4.6）的四个零样本LLM推荐器的幻觉率（OOD@10）和口头置信度校准（ECE、Brier、可靠性），这些系统未经过接地或微调，并跨越三个目录（MovieLens-25M、Amazon Reviews 2023 Toys、Yelp Open Dataset），按项目流行度分层。衡量目录成员资格本身是难点：在相同输出上，报告的比率会因使用的字符串匹配器而改变一个数量级，且F1无法区分候选方案。我们针对201个人类判断验证了该工具，并选择净偏差，其中采用的工具偏差为-0.040，而常见模糊规则的偏差为+0.144。幻觉率随后为...

    arXiv:2608.10008v2 Announce Type: replace-cross  Abstract: LLM recommenders for top-K item suggestion regularly emit titles outside the target catalog. Prior audits report a binary out-of-domain rate; none ask whether the model knew. We jointly audit hallucination rate (OOD@10) and verbalized-confidence calibration (ECE, Brier, reliability) for four zero-shot LLM recommenders from four independent vendors (Mistral Large, Llama-3.3-70B, GPT-OSS-120B, Claude Sonnet 4.6), not grounded or fine-tuned systems, across three catalogs (MovieLens-25M, Amazon Reviews 2023 Toys, Yelp Open Dataset), stratified by item popularity. Measuring catalog membership is itself the hard part: on identical outputs the reported rate moves by an order of magnitude with the string matcher used, and F1 cannot separate the candidates. We validate the instrument against 201 human judgments and select on net bias, where the adopted one is off by -0.040 against +0.144 for the common fuzzy rule. Hallucination is then 
    
[^22]: DREAM技术报告

    DREAM Technical Report

    [https://arxiv.org/abs/2608.09408](https://arxiv.org/abs/2608.09408)

    DREAM通过添加一个感知感知、可编排且可审计的策略层，在不替换现有流水线的情况下，解决了工业推荐系统中会话级意图转换未被充分处理的问题，并显著减少了报告量。

    

    摘要：arXiv:2608.09408v3 公告类型：替换 摘要：工业推荐系统通常使用级联的检索、排序和重排序流水线。尽管这些流水线效率高，但它们将信息和目标分散在多个模块中，依赖僵化的规则，并且对实时意图的感知有限，导致浏览、比较和购买之间的会话级转换未得到充分解决。我们提出了DREAM（开发具有代理方法的推荐引擎），这是一种自主优化控制架构，在现有流水线之上添加了一个感知感知、可编排且可审计的策略层，而无需替换它们。DREAM有两个核心组件。首先，一个三层意图引擎将设备端信号融合成结构化的L0/L1/L2意图表示；其边缘-云触发链将报告量减少至约8.7%。其次，一个元引擎使用元模型进行分层的M1到M2到M3推理：意图总结、由策略备忘录指导的策略规划。

    arXiv:2608.09408v3 Announce Type: replace  Abstract: Industrial recommender systems commonly use cascaded retrieval, ranking, and re-ranking pipelines. Although efficient, these pipelines fragment information and objectives across modules, rely on rigid rules, and have limited awareness of real-time intent, leaving session-level shifts among browsing, comparison, and purchase insufficiently addressed. We present DREAM (Developing Recommender Engine with Agentic Methods), an autonomous optimization control architecture that adds a perception-aware, orchestrable, and auditable policy layer atop existing pipelines without replacing them. DREAM has two core components. First, a three-tier Intent Engine fuses on-device signals into structured L0/L1/L2 intent representations; its edge-cloud trigger chain reduces reporting volume to approximately 8.7%. Second, a Meta Engine uses a MetaModel for layered M1-to-M2-to-M3 reasoning: intent summarization, strategy planning informed by Strategy Memo
    
[^23]: 教师保留全部令牌，学生高效融合：TM20K在广告推荐中的电商序列建模

    Teacher Retains Full Tokens, Student Merges Efficiently: TM20K for E-Commerce Sequence Modeling in Ad Recommendation

    [https://arxiv.org/abs/2608.07055](https://arxiv.org/abs/2608.07055)

    本文提出一种结合全Transformer和两阶段知识蒸馏的框架，在保留完整令牌的教师模型与高效融合的学生模型之间实现平衡，从而在电商广告推荐中高效处理超长行为序列。

    

    arXiv:2608.07055v2 公告类型：替换 摘要：受益于超长行为序列建模，现有的推荐系统通过同时考虑用户的长期和短期兴趣，为用户带来了更好的体验。然而，扩展的序列长度给训练效率和推理吞吐量带来了巨大负担。以往的方法通常采用基于搜索或聚类的压缩方式处理超长序列，但以损失细粒度信息为代价，或者依赖各种轻量级目标注意力结构，却无法充分提取序列特征。在本文中，我们通过全Transformer建模并辅以两阶段知识蒸馏框架，平衡了超长序列建模的有效性和效率。首先，教师和学生模型均采用全注意力机制，而非纯目标序列注意力，以实现有效的序列扩展。对于学生模型，我们提出了几种简单而有效的策略。

    arXiv:2608.07055v2 Announce Type: replace  Abstract: Benefiting from ultra-long behavior sequence modeling, existing recommender systems bring users a better experience via simultaneously considering their long-term and short-term interests. Nevertheless, extended sequence lengths introduce substantial burdens on training efficiency and serving throughput. Prior approaches typically utilize search-based or cluster-based compression on ultra-long sequences at the cost of fine-grained information, or rely on various lightweight target attention structures incapable of sufficient sequential feature extraction. In this paper, we balance the effectiveness and efficiency for ultra-long sequence modeling via full transformer modeling accompanied with a two-stage knowledge distillation framework. First, both teacher and student models take the full attention mechanism rather than pure target-sequence attention for effective sequence scaling. For student models, we propose several simple yet we
    
[^24]: omni-macos：苹果芯片上的设备端全模态搜索

    omni-macos: On-Device Omni-Modal Search on Apple Silicon

    [https://arxiv.org/abs/2608.05543](https://arxiv.org/abs/2608.05543)

    omni-macos在苹果设备上实现全模态本地搜索，通过内存预算管理和高效编码策略，确保所有数据不出设备。

    

    一个将文本、代码、文档、图像、音频和视频嵌入同一表示空间的搜索引擎，必须运行其编码器并将索引存储在某处，而几乎所有为此构建的组件都假设存在服务器。我们提出了omni-macos，它在其编码器、索引和存储都在已持有文件的Mac上运行，因此没有索引文件、键入的查询或向量会离开设备。它在用户设定的内存预算内，同时运行后台索引器和交互式搜索框：它仅重新编码编辑更改的块，在用户输入时将较小的单元交给GPU，从索引的一位副本回答查询并进行精确重新评分，并将该预算传播到使用统一内存的分配器。我们在五台Mac上进行了测量，其加速器宽度跨度八倍，内存跨度三十二倍，每台都对已持有的文件进行索引。

    arXiv:2608.05543v3 Announce Type: replace  Abstract: A search engine that embeds text, code, documents, images, audio and video into the same representation space has to run its encoder and keep its index somewhere, and almost every component built for the purpose assumes a server. We present omni-macos, which runs its encoder, index and store on the Mac that already holds the files, so no indexed file, no typed query and no vector ever leaves the machine. It keeps a background indexer and an interactive search box inside one memory budget the user sets: it re-encodes only the chunks an edit changes, hands the GPU smaller units while the user is typing, answers queries from a one-bit replica of the index with exact rescoring, and propagates that budget to the allocators that draw on unified memory. We measure on five Macs spanning an eightfold range of accelerator width and a thirty-twofold range of memory, each indexing the files it already holds.
    
[^25]: 一个提示不够：指令敏感性削弱嵌入模型评估

    One prompt is not enough: Instruction Sensitivity Undermines Embedding Model Evaluation

    [https://arxiv.org/abs/2605.22544](https://arxiv.org/abs/2605.22544)

    研究表明，指令嵌入模型的单提示评估不足以反映性能，因为提示措辞的敏感性会导致排名不稳定，基准测试需引入提示鲁棒性。

    

    arXiv:2605.22544v2 公告类型：替换 摘要：指令嵌入模型已成为最先进模型中的常见类型，然而它们仅使用每个任务一个提示进行评估。这种单点评估忽略了一个主要问题，即基于指令的方法的敏感性：对指令措辞的敏感度。我们对6个嵌入模型和11个数据集进行了提示敏感性的实证研究。我们表明，报告得分无法代表在合理提示下得分的分布。默认提示可能系统性地低估或高估性能。此外，我们显示排行榜排名对提示选择不稳健：开发者可以通过有利地选择提示来提高排名，而在对抗性提示选择下，任何模型都可以被提升到第一名。我们的发现表明，单提示评估不足以用于指令调优的嵌入模型，基准测试应纳入提示鲁棒性，要么通过评估...

    arXiv:2605.22544v2 Announce Type: replace  Abstract: Instruction embedding models have become common among state-of-the-art models, however are evaluated using a single prompt per task. The single-point evaluation ignores a main problem of the instruction-based approach namely: sensitivity to the phrasing of the instruction. We present an empirical study of prompt sensitivity across 6 embedding models and 11 datasets. We show that reported scores misrepresent the distribution of scores over plausible prompts. The default prompt can both systematically understate or overstate performance. Furthermore, we show that the leaderboard ranking is not robust to prompt selection: a developer improve their rank by favorably selecting prompts, and under adversarial prompt selection any model can be promoted to first place. Our findings suggest that single-prompt evaluation is insufficient for instruction-tuned embedding models and that benchmarks should incorporate prompt robustness, either by ev
    
[^26]: DiffGRM：基于扩散的生成式推荐模型

    DiffGRM: Diffusion-based Generative Recommendation Model

    [https://arxiv.org/abs/2510.21805](https://arxiv.org/abs/2510.21805)

    DiffGRM通过引入掩蔽离散扩散模型替代自回归解码器，解决了语义ID的因果路径限制和数字间训练不平衡问题，提升了生成式推荐的准确性和鲁棒性。

    

    生成式推荐（GR）是一种新兴范式，它通过分词器将每个项目表示为n位语义ID（SID），并根据用户历史记录自回归生成其SID来预测下一个项目。然而，SID的两个结构特性使得自回归模型（ARM）不太适用。首先，项目内一致性：n位数字共同指定一个项目，但从左到右的因果性仅在前缀条件下训练每位数字，阻碍了双向跨数字证据，将监督压缩到单一因果路径。其次，数字间异质性：数字在语义粒度和可预测性上有所不同，而统一的下一标记目标对所有数字赋予相同权重，导致对简单数字的过度训练和对困难数字的训练不足。为解决这两个问题，我们提出了DiffGRM，一种基于扩散的GR模型，用掩蔽离散扩散模型（MDM）替代自回归解码器，从而实现双向...

    arXiv:2510.21805v2 Announce Type: replace-cross  Abstract: Generative recommendation (GR) is an emerging paradigm that represents each item via a tokenizer as an n-digit semantic ID (SID) and predicts the next item by autoregressively generating its SID conditioned on the user's history. However, two structural properties of SIDs make ARMs ill-suited. First, intra-item consistency: the n digits jointly specify one item, yet the left-to-right causality trains each digit only under its prefix and blocks bidirectional cross-digit evidence, collapsing supervision to a single causal path. Second, inter-digit heterogeneity: digits differ in semantic granularity and predictability, while the uniform next-token objective assigns equal weight to all digits, overtraining easy digits and undertraining hard digits. To address these two issues, we propose DiffGRM, a diffusion-based GR model that replaces the autoregressive decoder with a masked discrete diffusion model (MDM), thereby enabling bidir
    
[^27]: 实际性能提升有多大？GraphRAG的无偏评估框架

    How Significant Are the Real Performance Gains? An Unbiased Evaluation Framework for GraphRAG

    [https://arxiv.org/abs/2506.06331](https://arxiv.org/abs/2506.06331)

    本文提出了一种无偏评估框架，通过图文本接地问题生成和消除LLM评估偏差，发现现有GraphRAG方法的实际性能提升远低于先前报告的水平。

    

    arXiv:2506.06331v2 公告类型：替换-交叉 摘要：通过从知识图谱中检索上下文，基于图的检索增强生成（GraphRAG）增强了大型语言模型（LLMs）为用户问题生成高质量答案的能力。已提出许多GraphRAG方法，并报告了在答案质量方面令人鼓舞的性能。然而，我们观察到当前GraphRAG的答案评估框架存在两个关键缺陷，即不相关问题和评估偏差，这可能导致对性能的有偏甚至错误的结论。为解决这两个缺陷，我们提出了一种无偏评估框架，该框架使用图-文本-接地问题生成来产生与底层数据集更相关的问题，并采用无偏评估程序来消除基于LLM的答案评估中的偏差。我们将我们的无偏框架应用于评估3种代表性的GraphRAG方法，并发现它们的性能提升比先前报告的要温和得多。

    arXiv:2506.06331v2 Announce Type: replace-cross  Abstract: By retrieving contexts from knowledge graphs, graph-based retrieval-augmented generation (GraphRAG) enhances large language models (LLMs) to generate quality answers for user questions. Many GraphRAG methods have been proposed and reported inspiring performance in answer quality. However, we observe that the current answer evaluation framework for GraphRAG has two critical flaws, i.e., unrelated questions and evaluation biases, which may lead to biased or even wrong conclusions on performance. To tackle the two flaws, we propose an unbiased evaluation framework that uses graph-text-grounded question generation to produce questions that are more related to the underlying dataset and an unbiased evaluation procedure to eliminate the biases in LLM-based answer assessment. We apply our unbiased framework to evaluate 3 representative GraphRAG methods and find that their performance gains are much more moderate than reported previous
    

