# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SAG-Mem-1.0 Technical Report](https://arxiv.org/abs/2611.33124) | SAG-Mem 提出了一种构建记忆时无需模型推理的对话记忆框架，通过保留完整对话分块，并在查询时利用 BM25 检索、证据评估与词汇扩展进行多轮检索，有效发现与问题相似度低但具互补性的证据。 |
| [^2] | [Delegent: A Multi-Agent Harness That Reduces Cost without Sacrificing Performance](https://arxiv.org/abs/2611.33098) | Delegent是一个基于Claude Code的多智能体框架，让前沿模型始终主导任务执行、仅将简单子任务委托给可在本地硬件运行的27B小型语言模型，从而无需训练路由器即可在不牺牲性能的情况下大幅降低智能体运行成本。 |
| [^3] | [Towards Realistic Evaluation of Uncertainty Quantification for LLM-Based Recommender Systems](https://arxiv.org/abs/2611.33036) | 该论文提出了一个固定排序基准，在统一的输出契约下将八种异构的不确定性量化信号作为记录级监控器进行评估，从而首次实现了对基于LLM的推荐系统中不确定性量化方法的公平、现实的可比性评估。 |
| [^4] | [SOARS: System One Assisted Reranking for Agentic Search](https://arxiv.org/abs/2611.32917) | SOARS是一个无需训练的智能体搜索重排序框架，利用快速“系统一”模型在单次调用中完成相关性估计与候选重排序，在六个大模型骨干上显著提升答案准确率和证据召回率，同时大幅降低搜索调用次数、延迟和推理成本。 |
| [^5] | [Speculative Decoding for Semantic-ID-Based Generative Recommendations](https://arxiv.org/abs/2611.32855) | SIDECAR通过近似的草稿候选与树注意力机制，在单次模型调用中对整个草稿打分，从而在认证精确top-$K$检索结果的同时，大幅降低基于语义ID的生成式推荐的推理开销。 |
| [^6] | [A Text-to-SQL Benchmark for Official Statistics](https://arxiv.org/abs/2611.32646) | 本文提出了LOCuST——首个面向官方统计场景的text-to-SQL基准，将荷兰统计局两千余个多语言宽表与语义知识图谱相结合，以应对现有基准无法模拟编码式列名和外部元数据等官方统计特点的问题。 |
| [^7] | [LazUnirec: Unifying Retrieval and Ranking in Multi-scenario Industrial Recommendation](https://arxiv.org/abs/2611.32590) | LazUnirec通过在Transformer主干上构建任务共享与任务私有参数来缓解梯度冲突，实现了多场景工业推荐中检索与排序的统一建模。 |
| [^8] | [See the Answer, Learn to Embed: Learning Universal Multimodal Embeddings from Hindsight](https://arxiv.org/abs/2611.32439) | 该论文提出HEME方法，在训练时将匹配的候选目标作为“后见之明”上下文来学习基于查询的表示，使模型在推理时无需额外推理计算即可生成高质量的通用多模态嵌入。 |
| [^9] | [Unjamming Deferrals: Supporting Students Through JAMN Bot, a Mobile RAG-Grounded Policy Assistant](https://arxiv.org/abs/2611.32266) | 本文提出了集成JAMN Bot的移动应用Schedulater，通过基于Ollama自托管Llama 3.2 3B的RAG管道，为大学缓考流程提供隐私保护的、基于政策的智能问答支持。 |
| [^10] | [OmniEVIE: Evidence-Vector-Informed Embedding for Multi-modal Retrieval](https://arxiv.org/abs/2611.32203) | 提出 OmniEVIE——一个 45 亿参数的多向量嵌入模型，通过单一压缩索引和三阶段检索统一支持文本、图像、视觉文档、视频和工具的多模态检索，在极低存储成本下实现证据保留与细粒度命中定位。 |
| [^11] | [LARF: Learning-aligned Reasoning Framework for Search Relevance](https://arxiv.org/abs/2611.32125) | 该论文提出了三阶段框架LARF，通过对齐推理学习解决搜索相关性估计中领域判断依据监督不足、相邻有序相关性标签间缺乏边际分离、以及思维链推理无法用于低延迟在线排序的难题。 |
| [^12] | [User-Adaptive Fusion of Multiple Foundation Models for Recommender Systems](https://arxiv.org/abs/2611.31951) | 本文提出用户自适应融合方法UF2O，通过在得分层面为每个用户自适应地组合多个基础模型的推荐得分，无需对齐表示空间即可融合异构基础模型，从而显著提升推荐效果。 |
| [^13] | [MELODI: Similarity-Identity Decoupling for Lightweight Ranking](https://arxiv.org/abs/2611.31940) | 本文提出相似性-身份解耦框架 MELODI，证明了相似度算子线性条件下“先池化后相似”与“先相似后池化”等价，从而以更高效的轻量级设计在公开及工业数据集上达到或超越现有排序基线。 |
| [^14] | [RELAY: Learning Compatible Representations One Encoder at a Time](https://arxiv.org/abs/2611.31882) | RELAY 通过“先训练带投影的新查询编码器、再冻结后训练新文档编码器”的两阶段方法，使新旧文档嵌入在增量式重编码过程中保持兼容，用同一查询向量即可同时评价新旧两版文档，显著提升文本到图像检索的性能。 |
| [^15] | [Provenance Loss and Repair in Multilingual Compaction](https://arxiv.org/abs/2611.31668) | 该论文发现多语言对话压缩会因语法性别等语言差异而丢失属性的来源信息（无法区分用户明确声明与语法默认），并提出一种无需训练、覆盖写作标注、丢弃前似然验证和阅读信任三个环节的修复方法，大幅恢复被压缩丢弃的已声明属性召回率。 |
| [^16] | [Provably Near-Optimal PQ in the Extended Zador Model](https://arxiv.org/abs/2611.31579) | 本文提出EZPQ算法，通过证明SVD是最优预处理旋转、构建低比特率下准确的扩展Zador代价估计器、以及基于一维装箱的全局比特分配分割方法，实现了具有可证明近似保证且实际性能接近最优的乘积量化。 |
| [^17] | [ModernELECTRA: Replaced-Token Detection with Boundary-Window Reconstruction for Compact Encoders that Classify and Embed](https://arxiv.org/abs/2611.31495) | ModernELECTRA提出了一种预训练方案，在保留替换词元检测（RTD）主目标的同时，引入轻量级的边界窗口重建辅助任务，使紧凑编码器既能擅长分类任务，又能胜任检索与语义相似度等嵌入任务。 |
| [^18] | [A dataset of clinically similar image pairs from citation-linked case reports for medical image retrieval](https://arxiv.org/abs/2611.31465) | 该研究构建了包含86,144对临床相似图像的PMC-PaIR数据集，通过跟随引文并利用大语言模型生成和回答分层问题来匹配病例图像，为医学图像检索系统的训练与评估提供了高质量的大规模数据资源。 |
| [^19] | [Small Learned Projections for Semantic-Reversal-Aware Retrieval with Spectral Acceleration: A Four-Domain Empirical Study](https://arxiv.org/abs/2611.31442) | 该论文提出一种小型学习投影，在法律、医疗、专利、金融四个领域实证中能有效阻止对抗性“陷阱”文档进入检索前10名，同时保留93-99%的正确释义文档，并借助边界条件化的谱图检索加速器实现大规模检索加速。 |
| [^20] | [Query Obfuscation with Embedding-Optimized Noise (EON): A Representational Blindspot in Agentic Retrieval](https://arxiv.org/abs/2611.31344) | 该论文发现，仅利用查询自身嵌入表示优化生成的非语义噪声（EON），在无法访问候选库、标签或反馈的情况下即可保留原查询的检索结果与工具路由，揭示了智能体检索系统可被恶意查询利用以规避安全过滤的表征盲区。 |
| [^21] | [Reliability of Language-Model Judges for Quantitative Scientific Question Answering: Evidence from Perovskite Photovoltaics](https://arxiv.org/abs/2611.31269) | 本文构建了由45,441条钙钛矿光伏实验记录确定性生成真实答案的定量基准，系统评估语言模型评判器的可靠性，发现配备参考答案的前沿评判器Claude Opus可近乎完美地复现真实评分并精确保留架构排名。 |
| [^22] | [CACO: Coverage-Aware Coordinated Optimization for Continuous Generative Retrieval](https://arxiv.org/abs/2611.25738) | CACO是一种轻量级自监督后训练框架，通过覆盖感知的无标签目标将多个采样查询协调至互补的物品覆盖，在几乎不增加推理延迟的情况下显著提升连续生成式检索的召回率。 |
| [^23] | [The Delayed Countdown Method for Validating Review-Oriented AI Workflows in eDiscovery](https://arxiv.org/abs/2610.31311) | 本文提出延迟倒计时方法，为电子取证中审查导向型AI工作流的有效性验证提供了一种简单实用、易于集成、兼容任意停止规则并能给出有效置信区间的评估方案。 |
| [^24] | [GEM: A Generative Embedding Model Bridging Reasoning and Retrieval](https://arxiv.org/abs/2608.13200) | GEM通过统一生成与嵌入，在检索前显式推理用户意图，显著提升了推理密集型任务的检索性能。 |
| [^25] | [EXCISE: Query-Side Exclusion for Late-Interaction Retrieval](https://arxiv.org/abs/2608.05497) | 提出EXCISE方法，通过仅在查询端训练两个轻量模块（共150万参数）来检测被排除的主题并对候选短名单重新嵌入，配合无参数的降权规则，在保持检索索引完全冻结的前提下，大幅提升迟交互检索器处理“X但不包括Z”类排除查询的能力。 |
| [^26] | [MMEB-V3: Measuring the Performance Gaps of Omni-Modality Embedding Models](https://arxiv.org/abs/2604.23321) | 本文提出覆盖文本、图像、视频、音频及智能体场景的全模态嵌入基准MMEB-V3，并构建跨模态语义等价元组数据集OmniSET，首次系统评估全模态嵌入模型性能，实现了语义相似性与模态效应的解耦分析。 |
| [^27] | [The Reasoning Bottleneck in Graph-RAG: Structured Prompting and Context Compression for Multi-Hop QA](https://arxiv.org/abs/2603.14045) | 该论文发现 Graph-RAG 的真正瓶颈在于推理而非检索，并提出 SPARQL 思维链提示与免 LLM 调用的图游走上下文压缩两项方法，显著提升多跳问答准确率并将上下文压缩约 60%。 |
| [^28] | [Diagnosing and Repairing Citation Failures in Generative Engine Optimization](https://arxiv.org/abs/2603.09296) | 本文提出首个引用失败模式分类体系与智能体系统AgentGEO，通过诊断文档未被AI生成回答引用的具体原因并施加针对性修复，显著提升生成式引擎优化中的引用效果。 |

# 详细

[^1]: SAG-Mem-1.0 技术报告

    SAG-Mem-1.0 Technical Report

    [https://arxiv.org/abs/2611.33124](https://arxiv.org/abs/2611.33124)

    SAG-Mem 提出了一种构建记忆时无需模型推理的对话记忆框架，通过保留完整对话分块，并在查询时利用 BM25 检索、证据评估与词汇扩展进行多轮检索，有效发现与问题相似度低但具互补性的证据。

    

    长期对话记忆需要检索散落在各个对话会话中的互补事实，包括与原始问题相似度有限的中间证据。现有方法通常依赖派生的记忆表示或持久化的关系结构，这会引入构建成本，并可能丢失源对话的细节。我们提出了 SAG-Mem，这是一个对话记忆框架，它将完整的对话消息保存在互不重叠的分块中，并通过查询时的词汇扩展来发现互补证据。记忆构建过程无需任何模型推理，仅使用词汇索引，而不需要基于大语言模型（LLM）的信息抽取、摘要生成或语义关系构建。在查询阶段，BM25 首先检索初始候选结果，随后证据评估器识别出有用的分块，并利用所选证据中的词汇线索引导额外一轮检索。检索到的对话内容还会进一步标注日历解释信息（原文在此处截断）。

    arXiv:2611.33124v1 Announce Type: new  Abstract: Long-term conversational memory requires retrieving complementary facts scattered across dialogue sessions, including intermediate evidence with limited similarity to the original question. Existing approaches often rely on derived memory representations or persistent relational structures, introducing construction costs and potentially losing source details. We present SAG-Mem, a conversational memory framework that preserves complete dialogue messages in non-overlapping chunks and discovers complementary evidence through query-time lexical expansion. Memory construction requires no model inference, using a lexical index without LLM-based extraction, summarization, or semantic relation construction. At query time, BM25 retrieves initial candidates, an evidence evaluator identifies useful chunks, and lexical cues from selected evidence guide an additional retrieval round. Retrieved dialogue is further annotated with calendar interpretati
    
[^2]: Delegent：一个在不牺牲性能的前提下降低成本的多智能体框架

    Delegent: A Multi-Agent Harness That Reduces Cost without Sacrificing Performance

    [https://arxiv.org/abs/2611.33098](https://arxiv.org/abs/2611.33098)

    Delegent是一个基于Claude Code的多智能体框架，让前沿模型始终主导任务执行、仅将简单子任务委托给可在本地硬件运行的27B小型语言模型，从而无需训练路由器即可在不牺牲性能的情况下大幅降低智能体运行成本。

    

    基于前沿语言模型构建的智能体成本高昂。在每一步中，前沿模型都会重新读取其全部上下文并执行每一个子任务，而不考虑任务本身的难度。本文探讨了一个简单的论点：通过将简单的子任务委托给可以在用户自有硬件上运行的小型语言模型（SLM），可以大幅削减这部分成本。为了验证这一论点，我们构建了Delegent，这是一个基于Claude Code构建的简单而有效的智能体框架。在该框架中，前沿模型始终主导整个任务，但会将聚焦的子任务（如调查代码、运行测试和搜索语料库）交给SLM子智能体处理。现有方法（如级联和模型路由器）通常选择由哪个模型执行整个任务，而在Delegent中，前沿模型始终主导执行过程，仅将简单的子任务委托给SLM。无需训练任何路由器或模型，因此两个模型都可以自由替换。借助一个27B FP8的SLM，Delegent降低了……

    arXiv:2611.33098v1 Announce Type: new  Abstract: Agents built on frontier language models are expensive. At every step, the frontier model re-reads its entire context and executes every subtask regardless of its difficulty. This paper explores the simple thesis that much of this cost can be removed by delegating easy subtasks to a small language model, or SLM, that runs on hardware one already owns. To test it, we build Delegent, a simple and effective agent harness built on Claude Code, in which the frontier model stays in charge of every task but hands focused subtasks, such as investigating code, running tests, and searching a corpus, to SLM subagents. Existing approaches such as cascades and model routers usually choose which model executes the entire task, whereas in Delegent, the frontier model always leads the execution and delegates easy subtasks to the SLM. No router or model is trained, so either model can be swapped freely. With the help of a 27B FP8 SLM, Delegent reduces th
    
[^3]: 迈向基于大语言模型的推荐系统中不确定性量化的现实评估

    Towards Realistic Evaluation of Uncertainty Quantification for LLM-Based Recommender Systems

    [https://arxiv.org/abs/2611.33036](https://arxiv.org/abs/2611.33036)

    该论文提出了一个固定排序基准，在统一的输出契约下将八种异构的不确定性量化信号作为记录级监控器进行评估，从而首次实现了对基于LLM的推荐系统中不确定性量化方法的公平、现实的可比性评估。

    

    大语言模型（LLM）为推荐系统提供了灵活的接口。然而，其给出的排序结果需要可靠的实例级不确定性监控。在这种场景下，比较不确定性量化（UQ）方法变得具有挑战性，因为LLM排序器缺乏标准的物品分类头。其打分可能依赖于选项标签、提示词、生成过程以及可用的用户和物品证据。现有方法从不同的来源提取不确定性，并将其应用于推荐过程的不同阶段。一些方法评估已生成的排序结果，而另一些方法则利用不确定性来改变排序本身，这使得直接比较变得困难。本文引入了一个固定排序基准，在共享的输出契约下，将八种异构的UQ信号作为记录级监控器进行评估。我们在四个推荐数据集和两个LLM骨干网络上，在共享的…（原文在此处截断）

    arXiv:2611.33036v1 Announce Type: cross  Abstract: Large language models (LLMs) provide a flexible interface for recommendation systems. However, their issued rankings require reliable instance-level uncertainty monitoring. In this scenario, comparing uncertainty quantification (UQ) methods becomes challenging because LLM rankers lack a standard item-classification head. Their scores may depend on option labels, prompts, generation procedures, and the available user and item evidence. Existing methods derive uncertainty from different sources and apply it at different stages of the recommendation process. Some methods evaluate an issued ranking, whereas others use uncertainty to change the ranking itself, making direct comparison difficult. Here we introduce a fixed-ranking benchmark which evaluates eight heterogeneous UQ signals as record-level monitors under a shared output contract. We evaluate these signals across four recommendation datasets and two LLM backbones under a shared fi
    
[^4]: SOARS：面向智能体搜索的系统一（System One）辅助重排序

    SOARS: System One Assisted Reranking for Agentic Search

    [https://arxiv.org/abs/2611.32917](https://arxiv.org/abs/2611.32917)

    SOARS是一个无需训练的智能体搜索重排序框架，利用快速“系统一”模型在单次调用中完成相关性估计与候选重排序，在六个大模型骨干上显著提升答案准确率和证据召回率，同时大幅降低搜索调用次数、延迟和推理成本。

    

    深度搜索智能体通过迭代检索与推理来满足复杂的信息需求，其中检索到的证据的相关性会直接影响下游任务的性能。现有的重排序器虽然能够提升检索效果，但在整个搜索过程中被频繁调用时会引入大量的延迟和成本。我们提出了SOARS，这是一个无需训练的框架，利用“系统一”模型进行快速且经过校准的相关性估计。在每个搜索步骤中，SOARS从长文档中筛选出与查询相关的段落，并在单次调用中完成候选结果的重排序，同时综合考虑原始问题和智能体生成的子查询。在BrowseComp-Plus和WebShaper数据集上的六个大语言模型骨干上，SOARS持续提升答案准确率和证据召回率，同时减少搜索调用次数、端到端延迟和推理成本。在BrowseComp-Plus上，它将GPT-5.4 Mini的准确率从73.9%提升至84.7%，同时将搜索调用次数从23次降至14次，……

    arXiv:2611.32917v1 Announce Type: cross  Abstract: Deep search agents address complex information needs through iterative retrieval and reasoning, where the relevance of retrieved evidence influences downstream performance. Existing rerankers improve retrieval effectiveness but introduce substantial latency and cost when invoked throughout the search trajectory. We introduce SOARS, a training-free framework leveraging a System One model for fast, calibrated relevance estimation. At each search step, SOARS selects query-relevant passages from long documents and reranks candidates in a single call, jointly considering the original question and agent-generated sub-query. Across six LLM backbones on BrowseComp-Plus and WebShaper, SOARS consistently improves answer accuracy and evidence recall while reducing search calls, end-to-end latency, and inference cost. On BrowseComp-Plus, it improves GPT-5.4 Mini's accuracy from 73.9% to 84.7% while reducing search calls from 23 to 14, demonstratin
    
[^5]: 面向基于语义ID的生成式推荐的投机解码

    Speculative Decoding for Semantic-ID-Based Generative Recommendations

    [https://arxiv.org/abs/2611.32855](https://arxiv.org/abs/2611.32855)

    SIDECAR通过近似的草稿候选与树注意力机制，在单次模型调用中对整个草稿打分，从而在认证精确top-$K$检索结果的同时，大幅降低基于语义ID的生成式推荐的推理开销。

    

    采用语义ID（SID）的生成式推荐系统将每个物品表示为一个由 $L$ 个离散词元组成的短序列，并利用序列模型根据用户历史预测下一个物品的词元。工业级部署通常使用此类模型，通过受限束搜索为每个请求检索多个候选物品。然而，这会带来每请求 $L$ 次顺序模型调用的固定开销，且当SID生成器的分布变得不够尖锐时准确率会下降，同时无法保证输出与精确top-$K$结果的接近程度。因此，生产系统无法检测到单个请求层面的检索质量退化。我们提出SIDECAR，它利用近似的草稿候选和一次树注意力机制，在内存允许的情况下通过单次模型调用对整个草稿进行打分。这一过程为每个未探索的SID前缀的概率设定上界，并在草稿包含精确top-$K$结果时对其进行认证。否则，则回退到分支定界策略进行扩展……（原文在此处截断）

    arXiv:2611.32855v1 Announce Type: new  Abstract: Generative recommender systems with Semantic IDs (SIDs) represent each item as a short sequence of $L$ discrete tokens and use a sequence model to predict the next item's tokens from a user's history. Industry-scale deployments typically use these models to retrieve multiple candidates per request with constrained beam search. However, this incurs a fixed cost of $L$ sequential model calls per request, and accuracy degrades as the SID generator becomes less sharp, without a guarantee on how close the output is to the exact top-$K$. Production systems therefore cannot detect degraded retrieval for individual requests.   We propose SIDECAR, which uses approximate draft candidates and a tree-attention pass to score the entire draft in a single model call, memory permitting. This pass bounds the probability of every unexplored SID prefix and certifies the exact top-$K$ if the draft contains them. Otherwise, a branch-and-bound fallback expand
    
[^6]: 面向官方统计数据的Text-to-SQL基准测试

    A Text-to-SQL Benchmark for Official Statistics

    [https://arxiv.org/abs/2611.32646](https://arxiv.org/abs/2611.32646)

    本文提出了LOCuST——首个面向官方统计场景的text-to-SQL基准，将荷兰统计局两千余个多语言宽表与语义知识图谱相结合，以应对现有基准无法模拟编码式列名和外部元数据等官方统计特点的问题。

    

    官方统计是包括政策制定者和企业在内的各类主体的重要数据集。对于官方统计的提供者而言，提升其数据的可访问性、治理和披露水平至关重要，而基于text-to-SQL方法构建的问答系统可以改善这一点。然而，现有的text-to-SQL基准测试并不符合这一应用场景。官方统计以开放域目录的形式发布，包含数千个宽表，其列名和取值均为编码，其含义记录在独立的元数据中，而非关系模式内。为了填补这一空白，我们提出了LOCuST，这是一个text-to-SQL基准，它将来自荷兰统计局的超过两千个英语和荷兰语表格与语义知识图谱相配对。该知识图谱通过标签、定义、层级结构和计量单位来描述表格、度量指标和维度编码，是系统完成该任务所需的外部知识。

    arXiv:2611.32646v1 Announce Type: new  Abstract: Official statistics are critical datasets for a wide variety of actors, including policymakers and businesses. For providers of official statistics, improving accessibility, governance, and disclosure of their data is of key importance. This can be improved by providing question-answering systems based on text-to-SQL approaches. However, existing text-to-SQL benchmarks do not resemble this setting. Official statistics are published as open-domain catalogs of thousands of wide tables whose columns and values are codes, and whose meaning is recorded in separate metadata rather than in a relational schema. To address this gap, we introduce LOCuST, a text-to-SQL benchmark that pairs over two thousand English and Dutch tables from Statistics Netherlands with a semantic knowledge graph. The graph describes the tables, measures, and dimension codes through labels, definitions, hierarchies, and units, and is the external knowledge a system needs
    
[^7]: LazUnirec：在多场景工业推荐中统一检索与排序

    LazUnirec: Unifying Retrieval and Ranking in Multi-scenario Industrial Recommendation

    [https://arxiv.org/abs/2611.32590](https://arxiv.org/abs/2611.32590)

    LazUnirec通过在Transformer主干上构建任务共享与任务私有参数来缓解梯度冲突，实现了多场景工业推荐中检索与排序的统一建模。

    

    现代工业推荐系统通常将检索和排序划分为两个独立的级联阶段，这导致了优化目标不一致、用户序列的冗余编码以及计算资源的浪费。以往的研究尝试使用Transformer同时建模检索和排序任务，但这些研究仍面临两个主要局限：(1) 由于检索和排序共享单一的Transformer主干网络，当两个任务的梯度方向不一致时会产生梯度冲突；(2) 当梯度冲突发生时，模型无法确定可靠的梯度更新方向。为此，我们提出了一种面向多场景工业推荐的统一检索与排序方法，称为LazUnirec。具体而言，我们在Transformer主干网络上构建任务共享参数与任务私有参数，以缓解梯度冲突。其中，对于……

    arXiv:2611.32590v1 Announce Type: new  Abstract: Modern industrial recommendation systems typically divide retrieval and ranking into two separate cascaded stages, leading to inconsistent optimization objectives, redundant encoding of user sequences, and a waste of computational resources. Previous efforts have investigated using Transformers to model both retrieval and ranking simultaneously. However, these studies still face two major limitations: (1) Since retrieval and ranking share a single Transformer backbone, the gradient conflict arises when the gradient directions of the two tasks are inconsistent. (2) When gradient conflict occurs, the model cannot determine a reliable direction for gradient updates. To this end, we propose a Unified Retrieval and Ranking approach for Multi-scenario Industrial Recommendation, termed as LazUnirec. Specifically, we construct task-shared and task-private parameters on the Transformer backbone to mitigate the gradient conflict. Wherein, for the 
    
[^8]: 看见答案，学会嵌入：从后见之明中学习通用多模态嵌入

    See the Answer, Learn to Embed: Learning Universal Multimodal Embeddings from Hindsight

    [https://arxiv.org/abs/2611.32439](https://arxiv.org/abs/2611.32439)

    该论文提出HEME方法，在训练时将匹配的候选目标作为“后见之明”上下文来学习基于查询的表示，使模型在推理时无需额外推理计算即可生成高质量的通用多模态嵌入。

    

    通用多模态嵌入（UME）模型将异构的多模态输入映射到一个共享的嵌入空间中，其中成对相似度可支持多种下游任务。近期的推理驱动的UME方法使用中间推理为嵌入生成额外上下文，但通常会在推理阶段引入额外的计算开销。在本文中，我们探索将匹配的候选（即配对的目标）作为一种替代性的上下文来源，研究模型能否在训练时“看见答案”，而在推理时无需它也能“学会嵌入”。然而，直接以这种特权上下文为条件，可能会引入仅凭查询本身无法获得的候选特定信息，或者导致以候选为条件的查询表示向候选嵌入坍缩。为解决这一问题，我们提出了后见之明增强的多模态嵌入（HEME），它从这种候选上下文中学习基于查询的后见之明表示。HEME

    arXiv:2611.32439v1 Announce Type: cross  Abstract: Universal Multimodal Embedding (UME) models map heterogeneous multimodal inputs into a shared embedding space, where pairwise similarity supports diverse downstream tasks. Recent reasoning-driven UME approaches use intermediate reasoning to generate additional context for embedding, but often incur extra computation at inference. In this paper, we explore the matched candidate (i.e., the paired target) as an alternative source of context, asking whether a model can "see the answer" during training and "learn to embed" without it at inference. However, direct conditioning on this privileged context can introduce candidate-specific information unavailable from the query alone or cause the candidate-conditioned query representation to collapse toward the candidate embedding. To address this, we introduce Hindsight-Enhanced Multimodal Embeddings (HEME), which learns query-grounded hindsight representations from such candidate context. HEME
    
[^9]: 解除缓考阻塞：通过JAMN Bot——基于移动端RAG的政策助手支持学生

    Unjamming Deferrals: Supporting Students Through JAMN Bot, a Mobile RAG-Grounded Policy Assistant

    [https://arxiv.org/abs/2611.32266](https://arxiv.org/abs/2611.32266)

    本文提出了集成JAMN Bot的移动应用Schedulater，通过基于Ollama自托管Llama 3.2 3B的RAG管道，为大学缓考流程提供隐私保护的、基于政策的智能问答支持。

    

    在一些教育机构中，诸如考试缓考申请流程之类的工作流程可能缓慢、不便且不一致，这产生了实施新型数字系统的需求。本文提出了Schedulater，一款用于管理大学考试缓考申请的移动应用。JAMN Bot是集成到Schedulater中的由大语言模型（LLM）驱动的聊天机器人，提供上下文感知的支持，帮助用户理解缓考流程。该系统采用基于Ollama（Llama 3.2 3B）的检索增强生成（RAG）管道进行自托管推理，能够在不牺牲隐私的情况下提供基于政策的回答。我们使用一个包含43个项目的基准测试来评估该系统，涵盖缓考、隐私、排期、跨政策以及对抗性和超范围问题，并报告了人类评估者之间的评分者间一致性以及检索命中率指标。随后，我们采用以GPT-5.4作为评判的LLM-as-a-judge方法，揭示了……

    arXiv:2611.32266v1 Announce Type: new  Abstract: In some educational institutions, workflows such as the exam deferral process can be slow, inconvenient, and inconsistent, which creates the need to implement novel digital systems. This paper presents Schedulater, a mobile app for managing university exam deferrals. JAMN Bot is a Large Language Model (LLM)-powered chatbot integrated into Schedulater that provides context-aware support to help users understand the deferral process. The system features a Retrieval-Augmented Generation (RAG) pipeline using Ollama (Llama 3.2 3B) for self-hosted inference which enables policy-grounded responses without sacrificing privacy. We evaluate the system using a 43-item benchmark spanning Exam Deferral, Privacy, Scheduling, Cross-Policy, and Adversarial & Out-of-Scope questions, and report inter-rater agreement among our human evaluators alongside retrieval hit-rate metrics. Subsequently, we utilize an LLM-as-a-judge approach with GPT-5.4, revealing 
    
[^10]: OmniEVIE：证据向量感知的多模态检索嵌入模型

    OmniEVIE: Evidence-Vector-Informed Embedding for Multi-modal Retrieval

    [https://arxiv.org/abs/2611.32203](https://arxiv.org/abs/2611.32203)

    提出 OmniEVIE——一个 45 亿参数的多向量嵌入模型，通过单一压缩索引和三阶段检索统一支持文本、图像、视觉文档、视频和工具的多模态检索，在极低存储成本下实现证据保留与细粒度命中定位。

    

    多模态检索需要匹配密集页面、细粒度实体、有序视频以及智能体的工具和记忆。单向量嵌入会丢失局部证据信息，交叉编码器需要对每个召回项进行在线重读，而未压缩的晚期交互存储开销过大难以部署。我们提出了 OmniEVIE，一个 4.5B 参数的多向量嵌入模型，统一覆盖文本、图像、视觉文档、视频和工具，采用单一压缩索引和三阶段检索器。每条记录最多存储 128 个维度为 64 的 int8 向量（9,472 字节，即每百万条记录仅 8.82 GiB），提供均值向量、1 比特符号位和 int8 多向量，分别服务于粗排、中排和精排，同时将每个命中结果精确定位到页面区域或视频区间。训练、合并、索引和检索进行协同设计：所有任务共享统一的检索格式，三级证据校验消除假负例，动态采样依据基线-教师差距，候选压缩采用 ST-HAC……

    arXiv:2611.32203v1 Announce Type: cross  Abstract: Multimodal retrieval must match dense pages, fine-grained entities, ordered video, and agent tools and memory. Single-vector embeddings lose local evidence, cross-encoders reread every recalled item online, and uncompressed late interaction is too large to serve. We present OmniEVIE, a 4.5B multi-vector embedding model covering text, images, visual documents, video, and tools with one compressed index and a three-stage retriever. Each record stores at most 128 int8 vectors of dimension 64 (9,472 bytes, or 8.82 GiB per million records), providing a mean vector, 1-bit signs, and int8 multi-vectors for coarse, middle, and final search while localizing each hit to a page region or video interval. Training, merging, indexing, and retrieval are co-designed: all tasks share one retrieval format, a three-grade evidence check removes false negatives, dynamic sampling follows the baseline-teacher gap, candidates are compressed with ST-HAC during
    
[^11]: LARF：面向搜索相关性的学习对齐推理框架

    LARF: Learning-aligned Reasoning Framework for Search Relevance

    [https://arxiv.org/abs/2611.32125](https://arxiv.org/abs/2611.32125)

    该论文提出了三阶段框架LARF，通过对齐推理学习解决搜索相关性估计中领域判断依据监督不足、相邻有序相关性标签间缺乏边际分离、以及思维链推理无法用于低延迟在线排序的难题。

    

    查询-文档相关性估计是AI原生搜索系统的核心组成部分，模型需要在复杂的多维度标准下评估检索到的文档是否满足用户的信息需求。然而，直接将大语言模型（LLM）应用于该任务仍具挑战性：传统的监督微调（SFT）对领域特定的判断依据提供的监督有限，而标准的轨迹级强化学习（RL）目标并未显式地在相邻的有序相关性标签之间强制实现边际分离。此外，尽管这类推理在训练阶段可能有用，但思维链推理对于延迟敏感的在线排序来说并不实用。这些挑战使得模糊的决策边界难以被有效解决。我们提出了一个三阶段框架LARF（面向搜索相关性的学习对齐推理框架）：它首先在...上进行冷启动SFT（原文在此处截断，后续内容缺失）。

    arXiv:2611.32125v1 Announce Type: new  Abstract: Query-document relevance estimation is a core component of AI-native search systems, where a model must assess whether retrieved documents satisfy users' information needs under complex, multi-dimensional criteria. However, directly applying large language models (LLMs) to this task remains challenging because conventional supervised fine-tuning (SFT) provides limited supervision for domain-specific judgment rationales, while standard trajectory-level reinforcement learning (RL) objectives do not explicitly enforce margin separation among adjacent ordinal relevance labels. Moreover, although such reasoning can be useful during training, chain-of-thought (CoT) reasoning is impractical for latency-sensitive online ranking. These challenges make it difficult to resolve ambiguous decision boundaries. We propose a three-stage framework LARF (Learning-Aligned Reasoning Framework for Search Relevance): it first performs cold-start SFT on strate
    
[^12]: 面向推荐系统的用户自适应多基础模型融合

    User-Adaptive Fusion of Multiple Foundation Models for Recommender Systems

    [https://arxiv.org/abs/2611.31951](https://arxiv.org/abs/2611.31951)

    本文提出用户自适应融合方法UF2O，通过在得分层面为每个用户自适应地组合多个基础模型的推荐得分，无需对齐表示空间即可融合异构基础模型，从而显著提升推荐效果。

    

    推荐系统能否通过融合多个基础模型（模态内及跨模态）而受益？虽然不同的基础模型可能捕捉到物品元数据中互补的不同方面，但我们发现，简单地将它们的表示组合起来，相比仅使用单个基础模型只能带来微弱的提升。我们提出了一个关键观察：受益于不同基础模型的用户群体是各不相同的。受此启发，我们提出了UF2O，一种面向推荐的用户自适应多基础模型融合方法。简而言之，UF2O通过为每个基础模型维护独立的推荐视图，并自适应地为每个用户组合各模型的得分，从而将每个基础模型的贡献建模为用户特定的。由于在得分层面进行融合，UF2O还能够融合异构的基础模型，而无需直接对齐它们的表示空间。在四个数据集上联合融合四个基础模型的大量实验表明，UF2O相比单一基础模型和朴素融合方法取得了显著的性能提升。

    arXiv:2611.31951v1 Announce Type: new  Abstract: Can recommender systems benefit from combining a collection of foundation models within and across modalities? While different foundation models may capture complementary aspects of item metadata, we find that naively combining their representations yields only marginal gains over using a single foundation model. We make a key observation: the users who benefit vary across foundation models. Motivated by this observation, we introduce UF2O, a user-adaptive approach for fusing multiple foundation models for recommendation. In a nutshell, UF2O models the contribution of each foundation model as user-specific by maintaining separate recommendation views and adaptively combining their scores for each user. By performing fusion at the score level, UF2O can also fuse heterogeneous foundation models without directly aligning their representation spaces. Extensive experiments that jointly combine four foundation models on four datasets show that
    
[^13]: MELODI：面向轻量级排序的相似性-身份解耦

    MELODI: Similarity-Identity Decoupling for Lightweight Ranking

    [https://arxiv.org/abs/2611.31940](https://arxiv.org/abs/2611.31940)

    本文提出相似性-身份解耦框架 MELODI，证明了相似度算子线性条件下“先池化后相似”与“先相似后池化”等价，从而以更高效的轻量级设计在公开及工业数据集上达到或超越现有排序基线。

    

    对于延迟受限的候选排序任务，我们提出了一种相似性-身份解耦框架，将目标-历史兼容性与仅目标信息分离开来。我们研究了计算池化相似度的两种顺序——先相似后池化与先池化后相似，并证明对于每个固定的目标，相似度算子在其历史参数上的线性性是两种顺序等价的充分条件，从而为计算成本更低的先池化后相似形式提供了理论依据。我们将该框架实例化为解耦身份与相似性的多头高效轻量算子（MELODI），其相似性分支将目标条件化的多头历史池化与一个对其池化历史输入呈线性的单一无残差交叉层相结合，而其身份分支则依赖于具体问题设定。在公开基准数据集上，MELODI 达到或超越了所评估的目标感知与目标无关基线。在两个大规模工业数据集上，它的……

    arXiv:2611.31940v1 Announce Type: cross  Abstract: For latency-constrained candidate ranking, we propose a similarity-identity decoupling framework that separates target-history compatibility from target-only information. We study two orders for computing pooled similarity, similarity-then-pool and pool-then-similarity, and identify linearity of the similarity operator in its history argument, for each fixed target, as sufficient for their equivalence, justifying the cheaper pool-then-similarity form. We instantiate the framework as multi-head efficient lightweight operator for decoupling identity and similarity (MELODI), whose similarity branch combines target-conditioned multi-head history pooling with a single residual-free cross layer that is linear in its pooled-history input and whose identity branch is problem-dependent. On public benchmark datasets, MELODI matches or surpasses the evaluated target-aware and target-agnostic baselines. On two large-scale industrial datasets, its 
    
[^14]: RELAY：一次训练一个编码器以学习兼容的表示

    RELAY: Learning Compatible Representations One Encoder at a Time

    [https://arxiv.org/abs/2611.31882](https://arxiv.org/abs/2611.31882)

    RELAY 通过“先训练带投影的新查询编码器、再冻结后训练新文档编码器”的两阶段方法，使新旧文档嵌入在增量式重编码过程中保持兼容，用同一查询向量即可同时评价新旧两版文档，显著提升文本到图像检索的性能。

    

    在增量式文档库重编码过程中，新旧文档嵌入会同时共存。RELAY 先针对原始文档库训练一个带有学习投影的新查询编码器，然后将两者冻结，再训练新的文档编码器。在整个单次重编码过程中，同一个查询向量可以对两个版本的文档进行评分。回填完成后，可选的查询细化步骤会移除该投影并复用已存储的文档向量。在 COCO 和 Flickr30K 上的三次文本到图像检索系统升级实验中，RELAY 相比原始系统在未改动文档库上的 Recall@1 提升了 8.66-10.23 个百分点。在直接重编码方面，无论是在原生评分还是匹配校准条件下，RELAY 在 Qwen-to-Qwen 升级中按 Recall@1 曲线下面积排名第一，在 CLIP-to-Qwen 升级中排名第二。在嵌入固定的情况下，针对每个文档版本单独校准评分可以减少回填期间 Recall@1 的下降，并改变哪种方法能取得最高的回填表现。

    arXiv:2611.31882v1 Announce Type: cross  Abstract: Old and new document embeddings coexist during incremental gallery re-encoding. RELAY trains a new query encoder with a learned projection against the original gallery, then freezes both to train the new document encoder. The same query vector scores both document versions throughout a single re-encoding pass. After backfill, optional query refinement drops the projection and reuses the stored document vectors. Across three text-to-image retrieval upgrades on COCO and Flickr30K, RELAY improves Recall@1 on untouched galleries by 8.66-10.23 percentage points over the original systems. For direct re-encoding, RELAY ranks first in both Qwen-to-Qwen upgrades and second in CLIP-to-Qwen by area under the Recall@1 curve, under both native scoring and matched calibration. With embeddings held fixed, calibrating scores separately for each document version can reduce drops in Recall@1 during backfill and change which method achieves the highest b
    
[^15]: 多语言压缩中的来源信息丢失与修复

    Provenance Loss and Repair in Multilingual Compaction

    [https://arxiv.org/abs/2611.31668](https://arxiv.org/abs/2611.31668)

    该论文发现多语言对话压缩会因语法性别等语言差异而丢失属性的来源信息（无法区分用户明确声明与语法默认），并提出一种无需训练、覆盖写作标注、丢弃前似然验证和阅读信任三个环节的修复方法，大幅恢复被压缩丢弃的已声明属性召回率。

    

    arXiv:2611.31668v1 公告类型： cross 摘要：会话型助手会将较早的对话消息压缩成简短笔记并丢弃原始消息。当笔记所用的语言需要作出英语中不作区分的区分时，例如西班牙语形容词的性属变化（agotado vs. agotada），省略该属性或明确声明该属性的消息可能产生相同的笔记。在原始消息被丢弃后，后续对话无法再区分这两种情况：将词形变化按字面理解会断言一个用户从未提供过的属性，而将其视为常规语法而忽略则会丢弃用户确实提供过的属性。我们提出了一种无需训练的修复方法，涵盖写作、丢弃前验证和阅读三个环节：写入者用成对形式（如 agotado/a）标注未声明的属性，丢弃前的似然检查将缺乏依据的单式形式改写为成对形式，而阅读者则将剩余的单式形式视为已声明的事实。在五个 Qwen3.5 和 Gemma-4 模型上，消息丢弃使问答（QA）中对已声明属性的召回率从 90% 降至 34%；我们的修复方法……

    arXiv:2611.31668v1 Announce Type: cross  Abstract: Conversational assistants compact older messages into short notes and evict the originals. When the note's language requires a distinction that English leaves open, such as Spanish adjective gender (agotado vs. agotada), messages that omit or explicitly state that attribute can yield the same note. After eviction, later turns cannot tell them apart: taking the inflected form literally asserts an attribute the user never gave, whereas discounting it as routine grammar discards one the user did give. We propose a training-free repair across writing, pre-eviction verification and reading. The writer marks unstated attributes with paired forms (e.g., agotado/a), a likelihood check before eviction rewrites unsupported single forms into paired ones, and the reader trusts remaining single forms as stated facts. On five Qwen3.5 and Gemma-4 models, eviction cuts recall of stated attributes from 90% to 34% in question answering (QA); our repair 
    
[^16]: 扩展Zador模型中可证明近似最优的乘积量化

    Provably Near-Optimal PQ in the Extended Zador Model

    [https://arxiv.org/abs/2611.31579](https://arxiv.org/abs/2611.31579)

    本文提出EZPQ算法，通过证明SVD是最优预处理旋转、构建低比特率下准确的扩展Zador代价估计器、以及基于一维装箱的全局比特分配分割方法，实现了具有可证明近似保证且实际性能接近最优的乘积量化。

    

    本文研究了针对高斯数据的乘积量化（PQ）优化问题，提出了三项关键贡献： 证明了SVD是PQ的最优预处理旋转； 提出了一种新的扩展Zador（EZ）估计器，用于估计$k$-means量化代价，在低比特率和低维度下依然准确； 提出了一种基于全局比特分配的一维装箱的新型分割算法。最终得到一个具有强近似保证的简单算法（EZPQ）。在高斯EZ代价模型下，它达到了与OPQ相当的经验精度，同时能够提供针对每个实例的失真下界证书。在典型的真实嵌入谱上，每坐标2比特的EZPQ在该模型中通常可证明处于最优值1%以内。此外，相比EDEN、E-RaBitQ和TurboQuant等所有基于随机旋转的算法，它始终能够取得更低的重构误差。

    arXiv:2611.31579v1 Announce Type: new  Abstract: This paper investigates Product Quantization (PQ) Optimization with Gaussian data. It offers three key contributions (i) a proof that SVD is the optimal preconditioning rotation for PQ; (ii) a new Extended Zador (EZ) estimator of the $k$-means quantization cost that is accurate in low bit rates and low dimensions; and (iii) a novel segmentation algorithm based on 1-d bin packing of global bit allocations. The result is a simple algorithm (EZPQ) with strong approximation guarantees. It matches OPQ's empirical accuracy while providing a per-instance distortion lower bound certificate under the Gaussian EZ cost model. On typical real embedding spectra, EZPQ with 2 bits per coordinate is often certifiably within 1% of the optimum in this model. It also consistently achieves a lower reconstruction error than all random-rotation-based algorithms such as EDEN, E-RaBitQ, and TurboQuant.
    
[^17]: ModernELECTRA：面向分类与嵌入的紧凑编码器的替换词元检测与边界窗口重建

    ModernELECTRA: Replaced-Token Detection with Boundary-Window Reconstruction for Compact Encoders that Classify and Embed

    [https://arxiv.org/abs/2611.31495](https://arxiv.org/abs/2611.31495)

    ModernELECTRA提出了一种预训练方案，在保留替换词元检测（RTD）主目标的同时，引入轻量级的边界窗口重建辅助任务，使紧凑编码器既能擅长分类任务，又能胜任检索与语义相似度等嵌入任务。

    

    arXiv:2611.31495v1 公告类型：新论文 摘要：ELECTRA式的替换词元检测（RTD）是一种用于预训练文本编码器的高样本效率训练目标，采用RTD训练的模型（如DeBERTaV3）在分类基准测试中表现优异。然而，众所周知，这类模型作为骨干网络在检索和语义相似度等嵌入任务上表现不佳：二元的“原始词元 vs. 替换词元”信号并未要求最终层保留稠密表示所依赖的词元身份和上下文内容。我们提出了ModernELECTRA，这是一种预训练方案，它将RTD保留为主要目标，并增加了一个轻量级辅助任务——边界窗口重建。对于每个词元，我们采样一个严格包含该词元的3至12个词元的窗口，仅从判别器在窗口两个边界处的隐藏状态来重建该词元的原始身份。该任务在训练时只需额外添加一个小型前馈网络（FFN），训练结束后即被丢弃，但它能迫使每个最终层状态携带信息……

    arXiv:2611.31495v1 Announce Type: new  Abstract: ELECTRA-style replaced-token detection (RTD) is a sample-efficient objective for pre-training text encoders, and RTD-trained models such as DeBERTaV3 are strong on classification benchmarks. The same models, however, are known to be poor backbones for embedding tasks such as retrieval and semantic similarity: the binary "original vs. replaced" signal does not ask the final layer to retain the token identity and contextual content that dense representations rely on. We introduce ModernELECTRA, a pre-training recipe that keeps RTD as the main objective and adds a lightweight auxiliary task, boundary-window reconstruction. For every token, we sample a window of 3-12 tokens that strictly contains it and reconstruct the token's original identity only from the discriminator's hidden states at the two window boundaries. The task costs one small FFN at training time and is discarded afterwards, yet it forces every final-layer state to carry info
    
[^18]: 基于引文关联病例报告的临床相似图像对数据集：用于医学图像检索

    A dataset of clinically similar image pairs from citation-linked case reports for medical image retrieval

    [https://arxiv.org/abs/2611.31465](https://arxiv.org/abs/2611.31465)

    该研究构建了包含86,144对临床相似图像的PMC-PaIR数据集，通过跟随引文并利用大语言模型生成和回答分层问题来匹配病例图像，为医学图像检索系统的训练与评估提供了高质量的大规模数据资源。

    

    检索与给定患者临床相似的患者图像可以支持诊断、教学和研究。训练和评估检索系统需要知道不同患者的哪些图像显示相同的发现，而现有数据集规模较小或范围狭窄。我们构建了PMC-PaIR数据集，包含来自56,013篇开放获取的PubMed Central病例报告的86,144对临床相似图像，这些病例报告通常记录罕见或具有挑战性的病例表现。我们开发了一种跟随引文的方法，因为引文通常链接相似患者的病例报告。为了确定两个关联病例在临床上的共同之处，该方法提示大语言模型生成并回答分层问题（从检查模态到诊断和发现），并使用答案通过说明文字匹配图像。专家认为这些图像对具有很高的准确性，且这种问题引导的方法找到的正确图像对数量是直接匹配方法的两倍。在PMC上微调BiomedCLIP……

    arXiv:2611.31465v1 Announce Type: cross  Abstract: Retrieving images of patients clinically similar to a given patient can support diagnosis, teaching and research. Training and evaluating retrieval systems requires knowing which images of different patients show the same finding, and existing datasets are small or narrow in scope. We construct PMC-PaIR, a dataset of 86,144 pairs of clinically similar images from 56,013 open-access PubMed Central case reports, which often document rare or challenging presentations. We develop an approach that follows citations, which often link case reports of similar patients. To determine what two linked cases share clinically, it prompts large language models to generate and answer hierarchical questions, from modality to diagnosis and findings, and to match images by caption using the answers. Experts find the pairs highly accurate, and this question-guided approach finds twice as many correct pairs as direct matching. Fine-tuning BiomedCLIP on PMC
    
[^19]: 用于语义反转感知检索的小型学习投影及谱加速：一项四领域实证研究

    Small Learned Projections for Semantic-Reversal-Aware Retrieval with Spectral Acceleration: A Four-Domain Empirical Study

    [https://arxiv.org/abs/2611.31442](https://arxiv.org/abs/2611.31442)

    该论文提出一种小型学习投影，在法律、医疗、专利、金融四个领域实证中能有效阻止对抗性“陷阱”文档进入检索前10名，同时保留93-99%的正确释义文档，并借助边界条件化的谱图检索加速器实现大规模检索加速。

    

    稠密嵌入检索器可能在对抗性场景下失效：当查询本应由文档A回答，却被一个与A仅有一处微小改变含义编辑的“近复制”文档B（即“陷阱”）反转时，在5,000文档池上，E5-Large-v2的原始余弦相似度会将B排入82-100%查询的前10名。我们研究两件事。第一，一个小型学习投影 W: R^1024 → R^d（d ∈ {11, 32}），通过确定性方案在一组声明的案例上训练，以抑制陷阱：对于声明的案例，在四个独立领域（法律、医疗、专利、金融）以及高达100,000的文档池的全部十二个（领域，池大小）设置中，HN@10 = 0.0000（前10名中不含陷阱），同时在精确最近邻搜索下，93-99%的案例中正确的释义文档得以保留。第二，一个边界条件化的加速器，用于在该规模下服务该投影：一种对谱图检索的加速方法（SpectRAG），以边界……替代密集特征分解。（摘要在此处截断）

    arXiv:2611.31442v1 Announce Type: new  Abstract: Dense-embedding retrievers can fail adversarially: when a query is answered by a document A and reversed by a near-copy B that differs from A by a small meaning-changing edit (a "trap"), raw cosine over E5-Large-v2 places B in the top-10 for 82-100% of queries at a pool of 5,000. We study two things. First, a small learned projection W: R^1024 -> R^d (d in {11, 32}), trained by a deterministic recipe on a declared set of cases, that suppresses the trap: for the declared cases, HN@10 = 0.0000 (no trap in the top-10) in all twelve (domain, pool-size) settings across four independent domains (legal, medical, patent, financial) and pools up to 100,000, while the correct paraphrase is retained for 93-99% of cases, under exact nearest-neighbour search. Second, a boundary-conditioned accelerator that serves the projection at that scale: an acceleration of spectral graph retrieval (SpectRAG) that replaces a dense eigendecomposition with a bounda
    
[^20]: 基于嵌入优化噪声（EON）的查询混淆：智能体检索中的表征盲区

    Query Obfuscation with Embedding-Optimized Noise (EON): A Representational Blindspot in Agentic Retrieval

    [https://arxiv.org/abs/2611.31344](https://arxiv.org/abs/2611.31344)

    该论文发现，仅利用查询自身嵌入表示优化生成的非语义噪声（EON），在无法访问候选库、标签或反馈的情况下即可保留原查询的检索结果与工具路由，揭示了智能体检索系统可被恶意查询利用以规避安全过滤的表征盲区。

    

    在现代智能体与语义检索系统中，稠密嵌入模型日益决定着针对某个查询应检索何种意图、工具、文档、记忆或图像。我们在这一流程中发现了一种表征失效模式：一个有意义的查询可以被明显非语义的内容所替换，同时保留其大部分检索结果——即使候选库对优化器是隐藏的。我们提出了嵌入优化噪声（Embedding-Optimized Noise, EON），它仅使用自然查询自身的表示来构建，无需访问候选嵌入、相关性标签、排序、检索反馈或护栏决策。尽管这一目标对语料库是完全盲目的，EON 仍能在多个稠密嵌入模型家族中频繁地保留意图与工具路由、基准文档搜索、情景式智能体记忆检索以及跨模态搜索。至关重要的是，这种脱节带来了严重的安全隐患：对于恶意查询，EON 能够可靠地规避输入（安全过滤）。

    arXiv:2611.31344v1 Announce Type: new  Abstract: In modern agentic and semantic retrieval systems, dense embedding models increasingly determine which intent, tool, document, memory, or image to retrieve for a query. We identify a representational failure mode in this pipeline: a meaningful query can be replaced by visibly nonsemantic content while preserving much of its retrieval outcome, even when the candidate bank is hidden from the optimizer. We introduce Embedding-Optimized Noise (EON), constructed using only the natural query representation, without access to candidate embeddings, relevance labels, rankings, retrieval feedback, or guardrail decisions. Despite this corpus-blind objective, EONs frequently preserve intent and tool routing, benchmark document search, episodic agent-memory retrieval, and cross-modal search across multiple dense embedding families. Crucially, this disconnect carries serious security implications: for malicious queries, an EON can reliably evade input 
    
[^21]: 语言模型评判器在定量科学问答中的可靠性：来自钙钛矿光伏的证据

    Reliability of Language-Model Judges for Quantitative Scientific Question Answering: Evidence from Perovskite Photovoltaics

    [https://arxiv.org/abs/2611.31269](https://arxiv.org/abs/2611.31269)

    本文构建了由45,441条钙钛矿光伏实验记录确定性生成真实答案的定量基准，系统评估语言模型评判器的可靠性，发现配备参考答案的前沿评判器Claude Opus可近乎完美地复现真实评分并精确保留架构排名。

    

    评估检索增强语言模型日益依赖自动化模型评判器，但其在定量科学综合任务上的可靠性在很大程度上尚未被测量。本文在一个钙钛矿光伏定量基准上评估自动化评分，其中全部755个真实答案均由45,441条实验器件记录确定性计算得出。我们评估了一个前沿评判器（Claude Opus 5.5）以及两个本地开源权重评判器（Qwen3-14B和Qwen3-8B），针对来自六种检索架构的4,530个自由文本答案，在三种模式下进行测试：参考增强（提供计算所得的参考答案）、证据增强（提供检索段落和查询日志）以及无辅助（仅提供问题和答案）。在配备参考答案的情况下，Claude Opus近乎完美地复现真实评分（Cohen's κ = 0.982 ± 0.003），并完整保留了架构排名（τ = 1.00）。取消参考答案会导致评估……

    arXiv:2611.31269v1 Announce Type: cross  Abstract: Evaluating retrieval-augmented language models increasingly relies on automated model judges, yet their reliability on quantitative scientific synthesis remains largely unmeasured. Here, we evaluate automated grading on a quantitative benchmark for perovskite photovoltaics, where all 755 ground-truth answers are deterministically computed from 45,441 experimental device records. We evaluate a frontier judge (Claude Opus 5.5) alongside two local open-weight judges (Qwen3-14B and Qwen3-8B) across 4,530 free-text answers from six retrieval architectures under three regimes: reference-augmented (with the computed key), evidence-augmented (with retrieved passages and query logs), and unassisted (question and answer alone). Equipped with the reference key, Claude Opus reproduces ground truth near-perfectly (Cohen's $\kappa = 0.982 \pm 0.003$) and preserves exact architectural rankings ($\tau = 1.00$). Withholding the reference causes evaluat
    
[^22]: CACO：面向连续生成式检索的覆盖感知协同优化

    CACO: Coverage-Aware Coordinated Optimization for Continuous Generative Retrieval

    [https://arxiv.org/abs/2611.25738](https://arxiv.org/abs/2611.25738)

    CACO是一种轻量级自监督后训练框架，通过覆盖感知的无标签目标将多个采样查询协调至互补的物品覆盖，在几乎不增加推理延迟的情况下显著提升连续生成式检索的召回率。

    

    基于流匹配和扩散模型的连续生成式检索器在物品嵌入空间中采样多个查询，以在固定检索预算下进行并行的近似最近邻（ANN）搜索。然而，优化分布多样性或目标层面的兴趣分离忽略了集合层面的检索几何结构：采样出的查询坍缩到相互重叠的ANN邻域中，而离散的top-K查找和单目标监督又阻碍了端到端的覆盖优化。我们提出了CACO（覆盖感知协同优化），这是一个轻量级的自监督后训练框架，它将来自冻结的流匹配和扩散检索器的查询协调至互补的物品覆盖，且额外推理延迟几乎可以忽略。CACO利用跨采样参考查询的经验检索频率作为相关性代理，构建了一个无标签的覆盖目标，在所有30个骨干模型-数据集设置中都提升了召回率。

    arXiv:2611.25738v1 Announce Type: new  Abstract: Continuous generative retrievers powered by flow matching and diffusion sample multiple queries in item embedding space for parallel approximate nearest-neighbor (ANN) search under fixed retrieval budgets. However, optimizing distribution diversity or target-level interest separation ignores set-level retrieval geometry: sampled queries collapse onto overlapping ANN neighborhoods, while discrete top-K lookup and single-target supervision block end-to-end coverage optimization. We introduce CACO (Coverage-Aware Coordinated Optimization), a lightweight self-supervised post-training framework that coordinates queries from frozen flow-matching and diffusion retrievers toward complementary item coverage with negligible additional inference latency. CACO uses empirical retrieval frequencies across sampled reference queries as a relevance proxy, formulating a label-free coverage objective that improves recall across all 30 backbone-dataset sett
    
[^23]: 用于验证电子取证中审查导向型AI工作流的延迟倒计时方法

    The Delayed Countdown Method for Validating Review-Oriented AI Workflows in eDiscovery

    [https://arxiv.org/abs/2610.31311](https://arxiv.org/abs/2610.31311)

    本文提出延迟倒计时方法，为电子取证中审查导向型AI工作流的有效性验证提供了一种简单实用、易于集成、兼容任意停止规则并能给出有效置信区间的评估方案。

    

    面向审查的工作流已成为电子取证中AI辅助文档审查的主流方法。然而，如何验证审查导向型工作流的有效性，一直是法律领域技术辅助审查（TAR）中一个令人困惑且充满争议的话题。我们分析了这一争议的根源，并为审查导向型工作流提出了一种简单而实用的评估方法。我们的延迟倒计时方法易于融入现有的审查导向型工作流和软件中，兼容任意停止规则，并能产生关于有效性的有效频率学派置信区间。我们阐释了该方法的概念基础，提供了示例计算，讨论了对逻辑上不可能的召回率值的处理，并就降低审查成本的后续研究提出了建议。

    arXiv:2610.31311v1 Announce Type: new  Abstract: Review-oriented workflows have become the dominant approach to AI-assisted document review in electronic discovery (eDiscovery). How to validate the effectiveness of review-oriented workflows has, however, remained a confusing and controversial topic in technology-assisted review (TAR) in the law. We examine the sources of this controversy, and propose a simple and practical evaluation approach for review-oriented workflows. Our Delayed Countdown method is easily incorporated into existing review-oriented workflows and software, is compatible with arbitrary stopping rules, and produces valid frequentist confidence intervals on effectiveness. We explain the conceptual foundations of the method, provide example computations, discuss the handling of logically impossible recall values, and suggest future work on decreasing review costs.
    
[^24]: GEM：一种弥合推理与检索的生成式嵌入模型

    GEM: A Generative Embedding Model Bridging Reasoning and Retrieval

    [https://arxiv.org/abs/2608.13200](https://arxiv.org/abs/2608.13200)

    GEM通过统一生成与嵌入，在检索前显式推理用户意图，显著提升了推理密集型任务的检索性能。

    

    arXiv:2608.13200v1 公告类型：交叉 摘要：现代大型语言模型在推理和指令遵循方面表现出色，使用户能够表达复杂多样的信息需求。然而，传统检索器主要依赖于查询和文档之间的表面匹配，导致用户表达需求的方式与检索器解读方式之间的差距日益扩大。在本文中，我们提出了GEM，一种生成式嵌入模型，通过显式推理用户意图和相关性标准，利用自身知识增强检索。GEM在单一模型中统一了生成和嵌入功能：它首先对查询进行推理，然后附加一个嵌入标记，以编码用于检索的增强上下文。在推理密集型和指令遵循型检索任务上的评估表明，GEM展示了其推理增强检索的有效性，优于其非推理变体，并匹配了使用更大模型的基线性能。此外，GEM的生成式嵌入能力使其能够灵活适应多样化的查询表达。

    arXiv:2608.13200v1 Announce Type: cross  Abstract: Modern LLMs excel at reasoning and instruction following, enabling users to express complex and diverse information needs. However, conventional retrievers largely rely on surface-level matching between queries and documents, resulting in a growing gap between how users express their needs and how retrievers interpret them. In this paper, we present GEM, a generative embedding model that augments retrieval through its own knowledge by explicitly reasoning about user intent and relevance criteria. GEM unifies generation and embedding within a single model: it first reasons over the query, then appends an embedding token to encode the enriched context for retrieval. \zhili{Evaluated on reasoning-intensive and instruction-following retrieval tasks, GEM demonstrates the effectiveness of its reasoning-augmented retrieval, outperforming its non-reasoning variant and matching baselines using substantially larger models.} Furthermore, GEM's ge
    
[^25]: EXCISE：面向迟交互检索的查询端排除方法

    EXCISE: Query-Side Exclusion for Late-Interaction Retrieval

    [https://arxiv.org/abs/2608.05497](https://arxiv.org/abs/2608.05497)

    提出EXCISE方法，通过仅在查询端训练两个轻量模块（共150万参数）来检测被排除的主题并对候选短名单重新嵌入，配合无参数的降权规则，在保持检索索引完全冻结的前提下，大幅提升迟交互检索器处理“X但不包括Z”类排除查询的能力。

    

    迟交互检索器在处理排除查询时表现不佳。当用户提出"X但不包括Z"这类查询时，加性的MaxSim分数会奖励那些命名Z的词元，从而提升而非抑制关于Z的文档——我们将这种失败称为“排除反转”。我们证明，在冻结的词元嵌入上测试的所有方法都无法恢复这一约束；其困难在于识别被排除的主题，而这仅取决于查询本身。因此，EXCISE将学习限制在查询路径上，并保持索引冻结。两个查询端模块（总计150万参数）负责检测被排除的主题并对100篇文档的候选短名单重新嵌入，再由一个无参数规则降低匹配该主题的候选文档的排名。在六个数据集合和三个骨干模型上，EXCISE在每个数据集合上都优于各骨干模型的冻结基线和微调基线。它将ExcluIR的success@10从0.10-0.22提升至0.84-0.85，将布尔NOT准确率从0.25-0.29提升至0.84-0.92。

    arXiv:2608.05497v2 Announce Type: replace  Abstract: Late-interaction retrievers handle exclusion queries poorly. When a user asks for "X but not Z", the additive MaxSim score rewards the tokens that name Z, so documents about Z are promoted rather than suppressed, a failure we call exclusion inversion. We show that no method we tested on the frozen token embeddings recovers this constraint; the difficulty lies in identifying the excluded topic, which depends on the query alone. EXCISE therefore confines learning to the query path and leaves the index frozen. Two query-side modules (1.5M parameters in total) detect the excluded topic and re-embed a 100-document shortlist, and a parameter-free rule demotes the candidates that match it. Across six collections and three backbones, EXCISE outperforms each backbone's frozen and fine-tuned baselines on every collection. It raises ExcluIR success@10 from 0.10-0.22 to 0.84-0.85 and Boolean NOT accuracy from 0.25-0.29 to 0.84-0.92. Pooled over 
    
[^26]: MMEB-V3：衡量全模态嵌入模型的性能差距

    MMEB-V3: Measuring the Performance Gaps of Omni-Modality Embedding Models

    [https://arxiv.org/abs/2604.23321](https://arxiv.org/abs/2604.23321)

    本文提出覆盖文本、图像、视频、音频及智能体场景的全模态嵌入基准MMEB-V3，并构建跨模态语义等价元组数据集OmniSET，首次系统评估全模态嵌入模型性能，实现了语义相似性与模态效应的解耦分析。

    

    多模态嵌入模型旨在将文本、图像、视频和音频等异构输入映射到共享的语义空间中。然而，现有的方法和基准在很大程度上仍局限于部分模态的覆盖，难以系统地评估全模态表示学习。在这项工作中，我们朝着全模态设置迈出了一步。我们引入了MMEB-V3，这是一个全面的基准，用于评估跨越文本、图像、视频、音频以及以智能体为中心的场景的嵌入。为了实现更细粒度的诊断，我们进一步构建了OmniSET（全模态语义等价元组），其中语义等价的实例在不同模态间进行表示，使我们能够将语义相似性与模态效应解耦。通过在MMEB-V3上的实验，我们对全模态嵌入进行了系统分析，并识别出三个关键发现：(1) 模型往往无法检索到预期的……（原文摘要在此处截断）

    arXiv:2604.23321v3 Announce Type: replace  Abstract: Multimodal embedding models aim to map heterogeneous inputs, such as text, images, videos, and audio, into a shared semantic space. However, existing methods and benchmarks remain largely limited to partial modality coverage, making it difficult to systematically evaluate full-modality representation learning. In this work, we take a step toward the full-modality setting. We introduce MMEB-V3, a comprehensive benchmark that evaluates embeddings across text, image, video, audio, as well as agent-centric scenarios. To enable more fine-grained diagnosis, we further construct OmniSET (Omni-modality Semantic Equivalence Tuples), where semantically equivalent instances are represented across modalities, allowing us to disentangle semantic similarity from modality effects. Through experiments on MMEB-V3, we conduct a systematic analysis of full-modality embeddings and identify three key findings: (1) models often fail to retrieve the intend
    
[^27]: Graph-RAG 中的推理瓶颈：面向多跳问答的结构化提示与上下文压缩

    The Reasoning Bottleneck in Graph-RAG: Structured Prompting and Context Compression for Multi-Hop QA

    [https://arxiv.org/abs/2603.14045](https://arxiv.org/abs/2603.14045)

    该论文发现 Graph-RAG 的真正瓶颈在于推理而非检索，并提出 SPARQL 思维链提示与免 LLM 调用的图游走上下文压缩两项方法，显著提升多跳问答准确率并将上下文压缩约 60%。

    

    Graph-RAG 系统通过将文档索引到知识图谱中来实现强大的多跳问答能力，但强大的检索并不保证能产生正确的答案。我们在三个多跳问答基准（HotpotQA、MuSiQue、2WikiMultiHopQA）上评估了领先的 Graph-RAG 系统 KET-RAG，发现 77% 至 91% 的问题在检索到的上下文中已包含标准答案，但准确率仅为 35% 至 78%，且 73% 至 84% 的错误源于推理失败。我们提出了两项增强方法：（i）SPARQL 思维链提示，将问题分解为与实体-关系上下文对齐的三元组模式查询；（ii）图游走压缩，通过知识图谱遍历将上下文压缩约 60%，且无需任何大语言模型调用。SPARQL 思维链提示将准确率提升了 2 至 14 个百分点；在较小模型上与结构化提示配合使用时，图游走压缩平均可额外带来 6 个百分点的提升。令人惊讶的是，我们发现通过问题类型路由，……

    arXiv:2603.14045v3 Announce Type: replace-cross  Abstract: Graph-RAG systems achieve strong multi-hop question answering by indexing documents into knowledge graphs, but strong retrieval does not guarantee strong answers. Evaluating KET-RAG, a leading Graph-RAG system, on three multi-hop QA benchmarks (HotpotQA, MuSiQue, 2WikiMultiHopQA), we find that 77% to 91% of questions have the gold answer in the retrieved context, yet accuracy is only 35% to 78%, and 73% to 84% of errors are reasoning failures. We propose two augmentations: (i) SPARQL chain-of-thought prompting, which decomposes questions into triple-pattern queries aligned with the entity-relationship context, and (ii) graph-walk compression, which compresses the context by ~60% via knowledge-graph traversal with no LLM calls. SPARQL CoT improves accuracy by +2 to +14 pp; graph-walk compression adds +6 pp on average when paired with structured prompting on smaller models. Surprisingly, we show that, with question-type routing, 
    
[^28]: 生成式引擎优化中引用失败的诊断与修复

    Diagnosing and Repairing Citation Failures in Generative Engine Optimization

    [https://arxiv.org/abs/2603.09296](https://arxiv.org/abs/2603.09296)

    本文提出首个引用失败模式分类体系与智能体系统AgentGEO，通过诊断文档未被AI生成回答引用的具体原因并施加针对性修复，显著提升生成式引擎优化中的引用效果。

    

    生成式引擎优化旨在提升内容在AI生成回答中的可见性。然而，现有方法衡量的是贡献度（即文档对回答的影响程度），而非引用——真正将流量引回内容创作者的机制。此外，这些方法统一套用通用的改写规则，无法诊断出单个文档为何未被引用。本文提出了一种面向GEO的诊断方法，探究文档为何未能被引用并进行针对性干预。我们构建了一个统一框架，包括：（1）首个覆盖引用流程不同阶段的引用失败模式分类体系；（2）AgentGEO——一个智能体系统，利用该分类体系诊断失败原因，从相应的工具库中选择针对性修复方案，并迭代优化直至成功获得引用；（3）一个以文档为中心的基准测试，用于评估优化效果能否泛化到未见过的查询。

    arXiv:2603.09296v2 Announce Type: replace-cross  Abstract: Generative Engine Optimization (GEO) aims to improve content visibility in AI-generated responses. However, existing methods measure contribution (how much a document influences a response) rather than citation, the mechanism that actually drives traffic back to creators. Also, these methods apply generic rewriting rules uniformly, failing to diagnose why individual documents are not cited. This paper introduces a diagnostic approach to GEO that asks why a document fails to be cited and intervenes accordingly. We develop a unified framework comprising: (1) the first taxonomy of citation failure modes spanning different stages of a citation pipeline; (2) AgentGEO, an agentic system that diagnoses failures using this taxonomy, selects targeted repairs from a corresponding tool library, and iterates until citation is achieved; and (3) a document-centric benchmark evaluating whether optimizations generalize across held-out queries.
    

