# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SURF: Subtractive Updates for Recommender Forgetting](https://arxiv.org/abs/2609.18695) | 提出轻量级框架SURF，通过在嵌入空间中定位待遗忘物品的邻域、在局部子集上训练辅助模型并在推理时进行减法更新，高效实现序列推荐系统的近似机器遗忘。 |
| [^2] | [SEEK: Secure and Efficient Encrypted Keyword Search For Privacy-Preserving Messaging Protocols](https://arxiv.org/abs/2609.18459) | SEEK结合同态加密与安全两方计算，提出了一种实用的加密关键词搜索协议，可将长消息的发送方加密与上传开销降低两个数量级，并将相关计算速度提升最高5.47倍。 |
| [^3] | [Exploring LLMs and RAG for Plausible and Explainable Material Prediction of Vehicle Components](https://arxiv.org/abs/2609.18437) | 本研究探索利用大语言模型及RAG技术预测车辆部件的合理材料，发现标准LLM生成已显著优于先前工作，而RAG方法并未带来进一步提升，并揭示了RAG系统在超参数优化、领域语料库可用性和专家评估方面的现存挑战。 |
| [^4] | [Understanding AI Provider Recommendations in Local Service Markets](https://arxiv.org/abs/2609.18341) | 该研究首次系统审计了AI助手在本地服务市场（如医疗、金融顾问）中的服务提供者推荐质量，发现无搜索时模型大量捏造推荐（仅4-11%对应真实服务提供者），而开启网络搜索后推荐准确率大幅提升至64-71%。 |
| [^5] | [One-Step Retrieval Framework for Real-Time Sponsored Search Ads Using Hierarchical Text Representations](https://arxiv.org/abs/2609.18296) | 提出ANGLE框架，利用LLM生成的分层文本表示将生成、判别与排序统一为单步实时检索过程，克服了传统多阶段级联架构的模块目标不一致问题以及基于SID的生成方法泛化性差、解码低效的缺陷。 |
| [^6] | [Quanta: A Self-Contained Python Library for Hybrid Retrieval over Quantised Embeddings, Lexical Indexes, and Knowledge Graphs](https://arxiv.org/abs/2609.18248) | Quanta是一个开源Python库，将4位量化嵌入的稠密向量搜索、BM25全文检索和知识图谱遍历统一到单一检索API中，并采用加权倒数排名融合（而非有问题的分数归一化）来组合多路检索信号。 |
| [^7] | [Single-Token Expected-Value Scoring for Cold-Start Candidate Ranking](https://arxiv.org/abs/2609.18188) | 该论文提出单Token期望值评分方法，将候选人-职位相关性建模为对等级Token的有序分类，通过读取首Token概率分布的期望值获得相关性评分，从而在标签数据有限的冷启动招聘排序场景中实现稳定准确的候选人排序。 |
| [^8] | [Time-Aligned Evolving Concept Graphs for Scientific Relation Forecasting](https://arxiv.org/abs/2609.18163) | 该论文提出一种时间对齐的演化概念图框架，将带日期的论文作为共享更新事件来联合建模语义与结构演化，使科学关系预测的平均关系AUPRC相比冻结上下文提升了16.6%。 |
| [^9] | [PageRecall: Measuring Page Selection in Literature-Grounded Question Answering](https://arxiv.org/abs/2609.18154) | 该论文发现文献问答系统的证据锚定瓶颈在检索而非阅读——页面选择器仅以 52.6% 的召回率将黄金页面呈现给模型，而模型拿到正确页面后引用准确率高达 94%，且页面缺失时往往静默失败，因此提出放弃页面选择、直接将整篇检索到的论文放入模型上下文的解决思路。 |
| [^10] | [LIGE-GR: A Smooth Leap from Ranking to Generative Recommendation in the LLM Era](https://arxiv.org/abs/2609.18148) | 该论文提出LIGE-GR框架，通过列表级生成与评估的方法，解决了将LLM范式中的序列级生成优化融入推荐系统以及避免整体替换成熟工业系统的两大挑战，实现了从传统排序推荐到生成式推荐的平滑跃迁。 |
| [^11] | [DUPAR: Dual-Path Conversational Retrieval via Speech Retriever with Cross-Turn Evidence Caching](https://arxiv.org/abs/2609.18042) | DUPAR提出了双路径对话检索框架，通过快速路径的跨轮证据缓存与慢速路径的全索引检索互补协作，在保持接近文本检索精度的同时实现3.75倍的查询端加速。 |
| [^12] | [How Calibration Content Shapes Attention-Based Reranking](https://arxiv.org/abs/2609.17764) | 该论文揭示了注意力重排序器中的空查询校准在面对包含详细指令的提示时会错误地移除相关信号，并提出了一种无需训练的插值空校准方法，通过控制进入空基线的指令内容比例，恢复了指令密集型任务上的重排序性能。 |
| [^13] | [One Size Does Not Fit All! Dynamic Retriever and Generator Selection for RAG](https://arxiv.org/abs/2609.17709) | 提出查询自适应框架DRAG，通过动态选择检索器-生成器组合来应对查询复杂度的差异，揭示检索与生成复杂度的收益均呈非单调递减特性，从而实现RAG系统计算资源的高效分配。 |
| [^14] | [Scaling Articulated Rationales for MLLM-based Recommendation](https://arxiv.org/abs/2609.17639) | 该论文提出工业级框架SARA，从2.4亿快手直播用户中挖掘并筛选用户的自然语言偏好理由（AUR），构建高质量数据集并将其转化为可扩展的推荐信号，弥补了隐式行为信号只能揭示用户行为而无法揭示偏好原因的不足。 |
| [^15] | [Beyond Static RAG: An Adaptive, Tri-Metric Routing Framework for Efficient Long-Context Inference on Commodity GPUs](https://arxiv.org/abs/2609.17564) | 本文提出一种无需训练的三指标路由框架，利用空间复杂度、句法密度和类符-形符比三个CPU侧信号，结合显存余量等硬件物理指标，在原始、神经压缩和词法三种流水线间动态选择，解决了消费级GPU上RAG部署中的“压缩悖论”问题。 |
| [^16] | [Can We Do Interpretable NLI with Graphs Based on Atomic Propositions?](https://arxiv.org/abs/2609.16814) | 本文提出一种完全基于图的可解释自然语言推理流水线，将句子分解为原子命题并转换为ConceptNet三元组构建图表示后在SNLI上达到89.7%的准确率，仅比同条件的文本模型低1.9个百分点，证明了可解释的图表示方法在NLI任务中的可行性。 |
| [^17] | [P3Rec: Distilling Prior--Posterior Preference Reasoning for LLM-based Recommendation](https://arxiv.org/abs/2609.13993) | 提出P3Rec框架，通过联合蒸馏大语言模型中互补的先验偏好（稳定长期兴趣）与后验偏好（目标相关细粒度兴趣）推理知识，克服单一视角蒸馏的局限性，提升轻量级推荐器的性能。 |
| [^18] | [Which Histories Matter for Time Series Forecasting? Learning Predictive Relevance with Future Supervision](https://arxiv.org/abs/2608.23221) | 该论文提出了一种通过未来监督学习预测相关性来重新排名时间序列检索候选的方法，显著提升了模式检索性能。 |
| [^19] | [From Overlooked to Explored: Recovering Item Relations via Mixture of Perspectives for Sequential Recommendation](https://arxiv.org/abs/2608.11846) | 本文提出PRISM模块，通过多视角混合方法克服序列推荐中自注意力模型的相似性偏差，从而恢复被忽视的异质物品关系以提升推荐性能。 |
| [^20] | [Bumblebee: Interleaved Mixed-Layer Building Blocks for Large-Scale Recommendation Systems](https://arxiv.org/abs/2607.24804) | 提出Bumblebee推荐架构，通过交错可堆叠的混合层构建模块将序列建模与特征交互两种范式有机融合，实现了特征模态的早期与反复混合，从而提升大规模推荐系统的表达能力。 |
| [^21] | [Time-Aware Diffusion based on Preference Disentanglement for Generative Recommendation](https://arxiv.org/abs/2606.01670) | 该论文提出TDPM框架，将用户偏好解耦并通过时间感知的扩散机制显式建模时间演化的偏好影响，从而克服了现有扩散推荐模型对历史物品统一处理的局限性。 |
| [^22] | [An Industrial-Scale Sequential Recommender for LinkedIn Feed Ranking](https://arxiv.org/abs/2602.12354) | 领英推出基于Transformer的序列推荐模型Feed SR取代原有DCNv2排序器，在12亿会员规模下成功部署，在线A/B测试中显著提升用户参与度（使用时长+2.10%，互动行为+3.52%）。 |
| [^23] | [Scalable Music Cover Retrieval Using Lyrics-Aligned Audio Embeddings](https://arxiv.org/abs/2601.11262) | 该论文提出利用歌词作为翻唱歌曲间的强不变量，通过歌词对齐的音频嵌入实现高效可扩展的音乐翻唱检索，从而避免现有方法对复杂音频处理流程和高计算资源的需求。 |
| [^24] | [Seeing Through the MiRAGE: Evaluating Multimodal Retrieval Augmented Generation](https://arxiv.org/abs/2510.24870) | MiRAGE是一个以论断为中心的多模态检索增强生成评估框架，通过InfoF1和CiteF1指标评估事实性、信息覆盖度与引用完整性，在文本任务上优于现有RAG评估指标，且是唯一能够推广到多模态来源的方法。 |
| [^25] | [On Predicting Post-Click Conversion Rate via Counterfactual Inference](https://arxiv.org/abs/2510.04816) | 该论文提出了一种以因果性为指导原则的反事实推断方法，通过为未点击样本反事实地生成转化标签，从而缓解点击样本稀疏性问题并改进点击后转化率预测。 |
| [^26] | [Unleash LLMs Potential for Sequential Recommendation by Coordinating Dual Dynamic Index Mechanism](https://arxiv.org/abs/2409.09253) | 该论文提出了首个采用双重动态索引机制的端到端大语言模型序列推荐系统ED²，将索引生成与序列推荐统一到单一LLM主干流水线中，同时解决了语义信息与协同信息整合不足以及高阶用户-物品交互模式利用不充分的问题。 |
| [^27] | [The Death of Schema Linking? Text-to-SQL in the Age of Well-Reasoned Language Models](https://arxiv.org/abs/2408.07702) | 最新的大语言模型即使面对大量不相关模式元素也能准确利用所需信息，因此当数据库模式能放入上下文窗口时，Text-to-SQL流程可完全省去模式链接步骤，转而采用增强、选择和纠正等技术来提升查询生成的准确性。 |
| [^28] | [Attribution in Scientific Literature: New Benchmark and Methods](https://arxiv.org/abs/2405.02228) | 该论文提出了科学引用归因基准REASONS以及由弃答率与幻觉率构成的双指标评估框架，系统评估了大语言模型在不同证据条件下的引用归因能力，发现高级RAG虽能降低幻觉率但牺牲了弃答能力，而对抗性元数据会使多个系统的幻觉率超过85%。 |

# 详细

[^1]: SURF：用于推荐系统遗忘的减法更新方法

    SURF: Subtractive Updates for Recommender Forgetting

    [https://arxiv.org/abs/2609.18695](https://arxiv.org/abs/2609.18695)

    提出轻量级框架SURF，通过在嵌入空间中定位待遗忘物品的邻域、在局部子集上训练辅助模型并在推理时进行减法更新，高效实现序列推荐系统的近似机器遗忘。

    

    对用户隐私日益增长的需求以及对GDPR等法规的合规要求，使得机器遗忘成为现代推荐系统的基本需求。然而，序列推荐系统（SRS）由于其依赖于时间交互模式，给机器遗忘带来了独特的挑战。现有方法要么需要计算上极其昂贵的完全重训练，要么无法考虑用户行为的序列特性。我们提出了SURF（Subtractive Updates for Recommender Forgetting），一个用于序列推荐系统中近似机器遗忘的轻量级框架。SURF分三个阶段运行：（i）在嵌入空间中识别待遗忘物品的邻域，（ii）在这个紧凑的局部子集上训练辅助模型，以及（iii）在推理时从原始模型中减去辅助模型的评分。在7个数据集上与五个基线方法的对比实验表明，SURF实现了高效的遗忘效果。

    arXiv:2609.18695v1 Announce Type: new  Abstract: The increasing demand for user privacy and compliance with regulations such as GDPR has made machine unlearning a fundamental requirement for modern recommender systems. However, Sequential Recommender Systems (SRS) pose unique challenges for unlearning due to their reliance on temporal interaction patterns. Existing approaches either require computationally prohibitive full retraining or fail to account for the sequential nature of user behavior. We propose SURF (Subtractive Updates for Recommender Forgetting), a lightweight framework for approximate machine unlearning in SRS. SURF operates in three stages: (i) identifying the neighborhood of the item to forget in the embedding space, (ii) training an auxiliary model on this compact local subset, and (iii) subtracting the auxiliary model's scores from the original model at inference time. Experiments against five baselines on 7 datasets show that SURF achieves unlearning effectiveness c
    
[^2]: SEEK：面向隐私保护消息协议的安全高效加密关键词搜索

    SEEK: Secure and Efficient Encrypted Keyword Search For Privacy-Preserving Messaging Protocols

    [https://arxiv.org/abs/2609.18459](https://arxiv.org/abs/2609.18459)

    SEEK结合同态加密与安全两方计算，提出了一种实用的加密关键词搜索协议，可将长消息的发送方加密与上传开销降低两个数量级，并将相关计算速度提升最高5.47倍。

    

    加密通信能够保护敏感用户数据，但也可能助长有害或非法的信息交流，从而在检测危险消息与保护终端用户隐私之间形成权衡。为解决这一问题，我们提出了SEEK，一种实用且高效的面向隐私保护消息传递的加密关键词搜索协议，它结合了同态加密与安全两方计算（2PC）。SEEK首先将消息分割为具有最小充分重叠的密文片段，然后使用加密关键词陷阱门对片段进行同态相关运算。对于长消息，该设计相比最先进的基线方法可将发送方的加密与上传开销降低多达两个数量级。它支持ASCII大小写不敏感匹配，每个片段仅需一个固定大小的加密陷阱门和一次同态乘法，相比最强的基于片段化的基线方法可获得高达5.47倍的相关计算加速。

    arXiv:2609.18459v1 Announce Type: cross  Abstract: Encrypted communication protects sensitive user data but can facilitate harmful or unlawful exchanges, creating a trade-off between detecting dangerous messages and preserving end-user privacy. To address this, we propose SEEK, a practical and efficient encrypted keyword-search protocol for privacy-preserving messaging that combines homomorphic encryption with secure two-party computation (2PC). SEEK first partitions messages into ciphertext fragments with the minimum sufficient overlap, then homomorphically correlates them using encrypted keyword trapdoors. For long messages, this design can reduce sender-side encryption and upload overhead by up to two orders of magnitude over state-of-the-art baselines. It supports ASCII case-insensitive matching with one fixed-size encrypted trapdoor and one homomorphic multiplication per fragment, yielding up to 5.47x faster correlation computation than the strongest fragmentation-based baselines.
    
[^3]: 探索大语言模型与检索增强生成技术在车辆部件合理且可解释材料预测中的应用

    Exploring LLMs and RAG for Plausible and Explainable Material Prediction of Vehicle Components

    [https://arxiv.org/abs/2609.18437](https://arxiv.org/abs/2609.18437)

    本研究探索利用大语言模型及RAG技术预测车辆部件的合理材料，发现标准LLM生成已显著优于先前工作，而RAG方法并未带来进一步提升，并揭示了RAG系统在超参数优化、领域语料库可用性和专家评估方面的现存挑战。

    

    在这项工作中，我们探索大语言模型（LLM）能否在无需大量微调的情况下，准确预测并解释车辆部件（如刹车盘或喷油器）的合理材料。我们测试并评估了三种方法：标准生成式LLM基线、单次检索增强生成（RAG）方法以及迭代的验证链（CoVe）变体。在检索方面，我们使用经过领域过滤的维基百科语料库这一公开可用数据。由于该任务不存在金标准，我们开发了一个定制的基于网络的标注工具，支持结构化领域专家评估的关键功能。基于LLM的生成显著优于先前的工作，而所测试的RAG方法并未进一步超越该结果。我们的研究结果揭示了基于RAG的系统仍面临的挑战：超参数优化、高质量且合法可获取的领域语料库的可用性，以及专家评估研究的设计。

    arXiv:2609.18437v1 Announce Type: new  Abstract: In this work, we explore whether LLMs can accurately predict and explain plausible materials for vehicle components such as brake discs or fuel injectors without requiring extensive fine-tuning. We test and evaluate three approaches: a standard generative LLM baseline, a single-pass Retrieval-Augmented Generation (RAG) approach, and an iterative Chain-of-Verification (CoVe) variant. For retrieval, we rely on publicly available data using a domain-filtered Wikipedia corpus. Since no gold standard exists for this task, we develop a custom web-based annotation tool supporting crucial functions for structured domain expert evaluation. LLM-based generation substantially outperforms prior work, which is not further surpassed by the tested RAG approaches. Our results surface remaining challenges for RAG-based systems: hyperparameter optimization, the availability of high-quality, legally accessible domain corpora, and expert evaluation study de
    
[^4]: 理解本地服务市场中的AI服务提供者推荐

    Understanding AI Provider Recommendations in Local Service Markets

    [https://arxiv.org/abs/2609.18341](https://arxiv.org/abs/2609.18341)

    该研究首次系统审计了AI助手在本地服务市场（如医疗、金融顾问）中的服务提供者推荐质量，发现无搜索时模型大量捏造推荐（仅4-11%对应真实服务提供者），而开启网络搜索后推荐准确率大幅提升至64-71%。

    

    当某人询问AI助手应该看哪位医生、或将积蓄托付给哪家公司时，得到的答案是一种推荐。我们对美国100个最大都市区内四个有官方注册记录支持的服务领域中的AI服务提供者推荐进行了审计，将每一条推荐与该领域的官方注册记录（Medicare临床医生与机构记录，以及SEC投资顾问披露信息）进行比对，并在三种条件下进行测试：开源权重模型、不带网络搜索的专有模型，以及带搜索功能的同一专有模型。在没有搜索的情况下，两种模型在网页覆盖薄弱的领域中大量捏造推荐。开源权重模型推荐的医生中仅有4%、专有模型推荐的医生中仅有11%能在被查询的城市中匹配到真实临床医生，且开源模型的匹配仅是名字上的巧合：其匹配到的临床医生作为初级保健医生的可能性并不比从注册记录中随机抽取的名字更高。在有搜索的情况下，64-71%的推荐……

    arXiv:2609.18341v1 Announce Type: cross  Abstract: When someone asks an AI assistant which doctor to see or which firm to trust with their savings, the answer is a referral. We audit AI provider recommendations in four registry-backed service domains across the 100 largest U.S. metropolitan areas, matching every recommendation against the official registry for its domain (Medicare clinician and facility records, and SEC adviser disclosures), under three conditions: an open-weight model, a proprietary model without web search, and the same proprietary model with search. Without search, both models largely fabricate recommendations in the domains the web covers thinly. Only 4% of the open-weight model's recommended doctors and 11% of the proprietary model's match a clinician in the queried city, and the open-weight matches are name coincidences: its matched clinicians are no likelier to be primary-care doctors than names drawn at random from the registry. With search, 64-71% of recommend
    
[^5]: 基于分层文本表示的实时赞助搜索广告单步检索框架

    One-Step Retrieval Framework for Real-Time Sponsored Search Ads Using Hierarchical Text Representations

    [https://arxiv.org/abs/2609.18296](https://arxiv.org/abs/2609.18296)

    提出ANGLE框架，利用LLM生成的分层文本表示将生成、判别与排序统一为单步实时检索过程，克服了传统多阶段级联架构的模块目标不一致问题以及基于SID的生成方法泛化性差、解码低效的缺陷。

    

    传统检索系统通常采用多阶段级联架构（MCA），其中每个模块独立优化，导致目标不一致，并使高潜力的候选广告被过早淘汰。近期基于大语言模型（LLM）的生成方法提供了端到端的解决方案，但它们使用离散语义标识符（SID）来检索广告，这些标识符并未被基础LLM学习到，且需要在监督微调（SFT）阶段记忆大量SID到广告的映射，导致对未见广告的泛化能力有限、维护和更新成本高昂。此外，SID与广告之间的一对一映射导致解码效率低下；这些方法还依赖小型奖励模型（如pCTR）进行相关性判断和排序，限制了LLM全面评估广告商业价值的能力。为应对这些挑战，我们提出了统一生成-判别-排序实时检索框架ANGLE。ANGLE使用LLM生成的分层文本表示（摘要内容在此处截断）。

    arXiv:2609.18296v1 Announce Type: new  Abstract: Traditional retrieval systems typically use multi-stage cascading architectures (MCA), where each module is optimized independently, leading to inconsistent objectives and the premature elimination of high-potential candidates. Recent LLM-based generation methods offer end-to-end solutions but use discrete semantic identifiers (SIDs) to retrieve ads, which are not learned by the base LLM and require memorization of numerous SID-to-ad mappings during SFT, suffering from limited generalization to unseen ads, high maintenance and update costs. The one-to-one mapping between SIDs and advertisements leads to inefficient decoding. Moreover, these methods rely on a small reward model (e.g. pctr) for relevance and ranking, limiting the LLM's ability to fully assess ads' commercial value. To address these challenges, we propose A uNified Generation-discriminative-ranking reaL-time rEtrieval (ANGLE) framework. ANGLE uses LLM-generated hierarchical
    
[^6]: Quanta：一个用于量化嵌入、词法索引和知识图谱混合检索的自包含Python库

    Quanta: A Self-Contained Python Library for Hybrid Retrieval over Quantised Embeddings, Lexical Indexes, and Knowledge Graphs

    [https://arxiv.org/abs/2609.18248](https://arxiv.org/abs/2609.18248)

    Quanta是一个开源Python库，将4位量化嵌入的稠密向量搜索、BM25全文检索和知识图谱遍历统一到单一检索API中，并采用加权倒数排名融合（而非有问题的分数归一化）来组合多路检索信号。

    

    先进的检索增强生成（RAG）流水线通常由三到四个独立运行的系统组装而成：近似最近邻索引、全文搜索引擎、图数据库和关系型文档存储。每个系统都带来各自的部署界面、配置模型和故障模式，而将它们绑定在一起的集成逻辑在每个项目中都需要重新编写。在本工作中，我们提出了Quanta——一个开源Python库，它将基于4位量化嵌入的稠密向量搜索、BM25全文检索和知识图谱遍历统一在单一检索API之后。Quanta做出了两项设计承诺，使其区别于现有的混合检索技术栈。首先，信号通过加权倒数排名融合（weighted reciprocal rank fusion）进行组合，而不是通过将异构分数归一化到共享范围——我们认为后者是有问题的（ill-posed），因为此类归一化依赖于具体查询。

    arXiv:2609.18248v1 Announce Type: cross  Abstract: An advanced retrieval-augmented generation pipeline is typically assembled from three or four independently operated systems: an approximate nearest-neighbour index, a full-text search engine, a graph database, and a relational document store. Each contributes its own deployment surface, configuration model, and failure modes, and the integration logic that binds them is written anew in every project. In this work, we present \textsc{Quanta}, an open-source Python library, which unifies dense vector search over 4-bit quantised embeddings, BM25 full-text retrieval, and knowledge-graph traversal behind a single retrieval API. Quanta makes two design commitments, which distinguish it from existing hybrid retrieval stacks. First, signals are combined by \emph{weighted reciprocal rank fusion} rather than by normalising heterogeneous scores onto a shared range, which we argue is ill-posed because such normalisations are query-dependent. Seco
    
[^7]: 冷启动候选人排序的单Token期望值评分方法

    Single-Token Expected-Value Scoring for Cold-Start Candidate Ranking

    [https://arxiv.org/abs/2609.18188](https://arxiv.org/abs/2609.18188)

    该论文提出单Token期望值评分方法，将候选人-职位相关性建模为对等级Token的有序分类，通过读取首Token概率分布的期望值获得相关性评分，从而在标签数据有限的冷启动招聘排序场景中实现稳定准确的候选人排序。

    

    AI辅助的人才搜寻简化了候选人审查流程，减轻了招聘人员手动筛选的管理负担。然而，将语言模型部署为生产环境的排序器仍然充满挑战。零样本大语言模型（LLM）可能产生不稳定、非确定性的评分且排序准确性较低；而传统的深度神经网络排序器需要数百万条记录的交互数据，这是低流量、细分领域的人才搜寻平台无法产生的。实际可用的是几十万条有序相关性标签——按排序器训练的标准来看规模较小，但当预训练语言模型已经编码了该任务所依赖的一般世界知识时，这个规模已经足够。我们提出了单Token期望值评分方法，这是一种排序原语，将候选人-职位相关性建模为对等级Token {1, ..., 5} 的有序分类，并将相关性评分读取为首Token概率分布的期望值。因为……（摘要在此处截断）

    arXiv:2609.18188v1 Announce Type: new  Abstract: AI-assisted sourcing streamlines candidate review, reducing the administrative burden of manual screening for recruiters. However, deploying language models as production rankers remains challenging. Zero-shot Large Language Models (LLMs) may produce unstable, non-deterministic scores and rank less accurately, while conventional deep neural rankers require millions of logged interactions that a low-traffic, niche sourcing platform does not produce. What is available instead is a few hundred thousand ordinal relevance labels -- small by ranker-training standards, but sufficient when a pretrained language model already encodes the general world knowledge the task depends on.   We present single-token expected-value scoring, a ranking primitive that casts candidate-job relevance as an ordinal classification over the grade tokens {1, ..., 5} and reads the relevance score as the expectation of the first-token probability distribution. Because
    
[^8]: 用于科学关系预测的时间对齐演化概念图

    Time-Aligned Evolving Concept Graphs for Scientific Relation Forecasting

    [https://arxiv.org/abs/2609.18163](https://arxiv.org/abs/2609.18163)

    该论文提出一种时间对齐的演化概念图框架，将带日期的论文作为共享更新事件来联合建模语义与结构演化，使科学关系预测的平均关系AUPRC相比冻结上下文提升了16.6%。

    

    预测科学关系可以在有前景的连接出现之前识别它们，从而指导科学发现。现有方法通常将概念语义和图结构分开建模，或者在粗糙的历史快照上总结语义，导致语义表示可能与快速演化的图证据不一致。我们提出了一种时间对齐的演化概念图框架，联合建模语义和结构演化。其核心思想是将带日期的论文视为共享更新事件，在每个预测时间点从相同的发表历史中重建语义状态和结构状态。对级别融合将这些状态结合起来，用于预测首次共现、关系形成和条件关系类型。在保持架构和训练设置不变的情况下，随着图更新同步刷新上下文，相比冻结上下文使平均关系AUPRC提升了16.6%。在基于187,848篇论文和270,687个概念构建的图上进行的实验表明……

    arXiv:2609.18163v1 Announce Type: new  Abstract: Forecasting scientific relations can guide discovery by identifying promising connections before they emerge. Existing approaches often model concept semantics and graph structure separately or summarize semantics over coarse historical snapshots, leaving semantic representations potentially misaligned with rapidly evolving graph evidence. We propose a time-aligned evolving concept graph framework that jointly models semantic and structural evolution. Its core idea is to treat dated papers as shared update events, reconstructing semantic and structural states from the same publication history through each prediction time. Pair-level fusion combines these states to forecast first co-occurrence, relation formation, and conditional relation type. Holding architecture and training fixed, refreshing context alongside graph updates improves mean relation AUPRC by 16.6% over frozen context. On a graph built from 187,848 papers with 270,687 conc
    
[^9]: PageRecall：衡量文献问答中的页面选择

    PageRecall: Measuring Page Selection in Literature-Grounded Question Answering

    [https://arxiv.org/abs/2609.18154](https://arxiv.org/abs/2609.18154)

    该论文发现文献问答系统的证据锚定瓶颈在检索而非阅读——页面选择器仅以 52.6% 的召回率将黄金页面呈现给模型，而模型拿到正确页面后引用准确率高达 94%，且页面缺失时往往静默失败，因此提出放弃页面选择、直接将整篇检索到的论文放入模型上下文的解决思路。

    

    我们描述了我们为 LitTraceQA（GroundLM @ EMNLP 2026）构建的系统：给定一个研究问题，从包含 27,487 篇论文的文献池中检索相关论文，引用答案所在的页面及对应的表格或图片，并按要求的格式作答。我们的主要发现是，证据锚定受限于检索环节，而非阅读环节。页面选择器仅有约一半的概率将标注者标注的页面（我们称之为“黄金页面”）呈现在负责定位证据的模型面前（黄金页面召回率为 52.6%）；而该模型在拿到正确页面的情况下，在其发出的 48 个定位结果中有 45 次引用了正确的页面（准确率达 94%）。当页面缺失时，模型很少如实说明：在这 45 个案例中，有 14 次返回空结果，24 次返回错误页面，仅 7 次返回正确页面，因此该流水线“静默失败”的频率几乎是“显性失败”的两倍。既然失败的根源在于正确的页面从未被呈现出来，解决办法就是不再进行选择：每篇检索到的论文都能完整放入模型的上下文中，因此我们……

    arXiv:2609.18154v1 Announce Type: cross  Abstract: We describe our system for LitTraceQA (GroundLM @ EMNLP 2026): given a research question, retrieve the relevant papers from a pool of 27,487, cite the page and the table or figure where the answer lives, and answer in a requested format. Our main finding is that evidence grounding is limited by retrieval, not by reading. The page selector put the annotator's page, which we call the gold page, in front of the model that locates evidence only about half the time (52.6% gold-page recall), while that model, given the page, cited the right one in 45 of the 48 locators it emitted (94%). When the page was missing it rarely said so: of 45 such cases it returned nothing 14 times, a wrong page 24 times, and a correct page 7 times, so the pipeline failed quietly almost twice as often as it failed visibly. Since the failure was that the right page was never shown, the fix is to stop choosing: each retrieved paper fits in the model's context, so we
    
[^10]: LIGE-GR：大模型时代从排序到生成式推荐的平滑跃迁

    LIGE-GR: A Smooth Leap from Ranking to Generative Recommendation in the LLM Era

    [https://arxiv.org/abs/2609.18148](https://arxiv.org/abs/2609.18148)

    该论文提出LIGE-GR框架，通过列表级生成与评估的方法，解决了将LLM范式中的序列级生成优化融入推荐系统以及避免整体替换成熟工业系统的两大挑战，实现了从传统排序推荐到生成式推荐的平滑跃迁。

    

    arXiv:2609.18148v1 公告类型：新论文 摘要：大语言模型（LLMs）的卓越成功为下一代推荐系统提供了重要启示。从结构上看，推荐与语言生成存在相似之处：两者都旨在生成一个能够优化用户体验的有序序列。然而，如何将LLM范式的精髓精准地融入成熟的工业推荐系统，仍然是一个开放性问题。这里存在两个挑战：首先，目前尚不清楚如何将LLM范式中的序列级生成与优化引入推荐任务；其次，现实中的推荐系统是成熟的系统，它们多年来围绕特定产品、业务约束、服务基础设施和组织架构进行了深度定制化迭代，整体替换这类系统在技术上往往风险很高，在组织层面也具有破坏性。在本文中，我们提出了LIGE-GR，一种列表级的生成与评估方法（摘要内容在此处截断）。

    arXiv:2609.18148v1 Announce Type: new  Abstract: The remarkable success of large language models (LLMs) has provided important inspiration for the next generation of recommender systems. Structurally, recommendation and language generation share a similarity: both aim to produce an ordered sequence that optimizes the user's experience. However, how to precisely absorb the essence of the LLM paradigm into mature industrial recommender systems remains an open problem.   There are two challenges. First, it is unclear how to incorporate sequence-level generation and optimization from the LLM paradigm into recommendation. Second, real-world recommender systems are mature systems that have been iteratively customized for years around specific products, business constraints, serving infrastructure, and organizational ownership. Replacing such systems wholesale is often technically risky and organizationally disruptive.   In this paper, we propose LIGE-GR, a listwise generation and evaluation 
    
[^11]: DUPAR：基于语音检索器与跨轮证据缓存的双路径对话检索

    DUPAR: Dual-Path Conversational Retrieval via Speech Retriever with Cross-Turn Evidence Caching

    [https://arxiv.org/abs/2609.18042](https://arxiv.org/abs/2609.18042)

    DUPAR提出了双路径对话检索框架，通过快速路径的跨轮证据缓存与慢速路径的全索引检索互补协作，在保持接近文本检索精度的同时实现3.75倍的查询端加速。

    

    基于外部知识的语音助手通常使用自动语音识别（ASR）将语音查询转写为文本后，再从文本知识库中检索证据。这种级联方式会增加延迟并传播识别错误，而直接语音检索则容易受到跨模态不对齐的影响。为解决这些局限性，我们提出了DUPAR，一个具有互补的慢速和快速路径的对话检索框架。快速路径使用与冻结的BGE-M3文本嵌入对齐的任务自适应音频编码器来搜索跨轮证据缓存。当缓存置信度不足时，慢速路径融合使用音频和ASR转写嵌入的全索引检索，并且所选证据通过一跳图扩展刷新下一轮的证据缓存。在一个特定领域的知识库上，我们训练的音频编码器在干净语音上的检索精度接近文本检索，同时查询端相比（原文截断）实现了3.75倍的加速。

    arXiv:2609.18042v1 Announce Type: new  Abstract: Voice assistants grounded in external knowledge typically use automatic speech recognition (ASR) to transcribe speech queries before retrieving evidence from textual knowledge bases. This cascade adds latency and propagates recognition errors, whereas direct speech retrieval is vulnerable to cross-modal misalignment. To address these limitations, we propose DUPAR, a conversational retrieval framework with complementary slow and fast paths. The fast path uses a task-adapted audio encoder aligned with frozen BGE-M3 text embeddings to search a cross-turn evidence cache. When cache confidence is insufficient, the slow path fuses full-index retrieval using audio and ASR-transcript embeddings, and the selected evidence refreshes the next-turn evidence cache through one-hop graph expansion. On a domain-specific knowledge base, our trained audio encoder approaches text-retrieval accuracy on clean speech with a 3.75$\times$ query-side speedup ove
    
[^12]: 校准内容如何影响基于注意力的重排序

    How Calibration Content Shapes Attention-Based Reranking

    [https://arxiv.org/abs/2609.17764](https://arxiv.org/abs/2609.17764)

    该论文揭示了注意力重排序器中的空查询校准在面对包含详细指令的提示时会错误地移除相关信号，并提出了一种无需训练的插值空校准方法，通过控制进入空基线的指令内容比例，恢复了指令密集型任务上的重排序性能。

    

    基于注意力的重排序器通过聚合查询到文档的注意力，并减去一次空查询校准过程来对文档进行评分，以消除位置和结构偏差。尽管被广泛使用，这种校准假设空查询过程会从每个文档中移除无关信号。我们表明，现代提示内容（例如约束、指令、角色设定和示例演示）在进入评分读出时可能会违反这一假设，使空查询过程变得具有相关性感知而非真正的“空”。我们发现，当校准应用于包含较长、更详细指令的提示时尤其有害，因为空查询步骤会移除相关信号。基于这些发现，我们提出插值空校准，这是一种无需训练的修改方法，可控制多少指令内容进入空基线。它在标准校准失效的指令密集型任务上恢复了基于注意力的重排序性能，同时……（摘要截断）

    arXiv:2609.17764v1 Announce Type: new  Abstract: Attention-based rerankers score documents by aggregating query-to-document attention and subtracting a null-query calibration pass to remove positional and structural bias. Although widely used, this calibration assumes that the null pass removes irrelevant signal from each document. We show that modern prompt content, e.g. constraints, instructions, personas, and demonstrations can violate this assumption when it enters the scoring readout, making the null pass relevance-aware rather than null. We find that calibration is especially harmful when applied to prompts containing longer, more detailed instructions as the null-pass step removes relevant signal. Based on these findings, we propose interpolated null calibration, a training-free modification that controls how much of the instruction content enters the null baseline. It recovers attention-based reranking performance on instruction-heavy tasks where standard calibration fails, whi
    
[^13]: 一种尺寸并不适合所有情况！面向RAG的动态检索器与生成器选择

    One Size Does Not Fit All! Dynamic Retriever and Generator Selection for RAG

    [https://arxiv.org/abs/2609.17709](https://arxiv.org/abs/2609.17709)

    提出查询自适应框架DRAG，通过动态选择检索器-生成器组合来应对查询复杂度的差异，揭示检索与生成复杂度的收益均呈非单调递减特性，从而实现RAG系统计算资源的高效分配。

    

    检索增强生成（RAG）系统通常在所有查询中使用固定的检索器和生成器配置，尽管不同查询的复杂度和信息需求存在显著差异，这导致了计算资源的低效分配。虽然检索自适应性和生成自适应性已被分别独立研究，但两者对端到端RAG性能的联合影响仍未得到充分探索。我们系统地分析了检索器和生成器复杂度在事实型问题和多跳问答（QA）任务之间的交互作用，包括桥接推理和组合推理任务。我们的分析表明，更强的检索通常比增加生成投入带来更大的收益，但两者均表现出收益递减且非单调的特性，这表明更高复杂度的配置并非在所有查询上都一致更优。基于这些发现，我们提出了DRAG，一个用于选择检索器-生成器组合的查询自适应框架。

    arXiv:2609.17709v1 Announce Type: cross  Abstract: Retrieval-Augmented Generation (RAG) systems typically employ fixed retriever and generator configurations across queries, despite substantial differences in query complexity and information needs, leading to inefficient allocation of computational resources. While retrieval and generation adaptivity have been studied independently, their joint effect on end-to-end RAG performance remains underexplored. We systematically analyze how retriever and generator complexity interacts across factoid and multi-hop question answering (QA), including bridge and composition reasoning tasks. Our analysis shows that stronger retrieval generally yields larger gains than increased generation effort, but both exhibit diminishing and non-monotonic returns, indicating that higher-complexity configurations are not uniformly better across queries. Motivated by these findings, we introduce DRAG, a query-adaptive framework for selecting retriever-generator c
    
[^14]: 基于多模态大语言模型推荐的显性化用户理由规模化

    Scaling Articulated Rationales for MLLM-based Recommendation

    [https://arxiv.org/abs/2609.17639](https://arxiv.org/abs/2609.17639)

    该论文提出工业级框架SARA，从2.4亿快手直播用户中挖掘并筛选用户的自然语言偏好理由（AUR），构建高质量数据集并将其转化为可扩展的推荐信号，弥补了隐式行为信号只能揭示用户行为而无法揭示偏好原因的不足。

    

    现代推荐系统主要从点击、观看时长和负反馈等隐式行为中推断用户偏好，但这些信号只揭示用户做了什么，而非用户为什么喜欢或不喜欢某些内容。本工作研究显性化用户理由，即用户以自然语言对其偏好做出的解释，将其作为一类新的、具有极性感知和原因级别的推荐文本信号。尽管AUR具有潜在价值，但由于其天然稀疏、质量往往较低，且仅覆盖一小部分物品，因此难以在工业系统中使用。我们提出了SARA（Scaling Articulated Rationales），一个将稀疏AUR转化为可扩展推荐信号的工业级框架。SARA首先构建了一个数据引擎，从2.4亿快手直播用户中引出并筛选AUR，生成了SARA-HQ——一个经过质量控制且以作者为中心的理由数据集。随后，它将一个通用的多模态大语言模型对齐至……（摘要在此处被截断）

    arXiv:2609.17639v1 Announce Type: cross  Abstract: Modern recommendation systems largely infer user preferences from implicit behaviors such as clicks, watch time, and negative feedback, but these signals reveal what users do rather than why they like or dislike content. This work studies articulated user rationales (AURs), i.e., users' natural-language explanations of their preferences, as a new class of polarity-aware and reason-level textual signals for recommendation. Despite their potential value, AURs are difficult to use in industrial systems because they are naturally sparse, often low-quality, and only cover a small fraction of items. We present SARA (Scaling Articulated Rationales), an industrial framework that turns sparse AURs into scalable recommendation signals. SARA first builds a data engine that elicits and curates AURs from 240M Kuaishou Live users, producing SARA-HQ, a quality-controlled and author-centric rationale dataset. It then aligns a general-purpose MLLM into
    
[^15]: 超越静态RAG：一种面向消费级GPU高效长上下文推理的自适应三指标路由框架

    Beyond Static RAG: An Adaptive, Tri-Metric Routing Framework for Efficient Long-Context Inference on Commodity GPUs

    [https://arxiv.org/abs/2609.17564](https://arxiv.org/abs/2609.17564)

    本文提出一种无需训练的三指标路由框架，利用空间复杂度、句法密度和类符-形符比三个CPU侧信号，结合显存余量等硬件物理指标，在原始、神经压缩和词法三种流水线间动态选择，解决了消费级GPU上RAG部署中的“压缩悖论”问题。

    

    在诸如NVIDIA T4（16 GB显存）等消费级GPU上部署检索增强生成（RAG）会暴露出一种我们称之为“压缩悖论”的实际失败模式：神经提示压缩会带来键值（KV）缓存竞争和预处理延迟，其开销可能超过生成阶段所节省的时间；而跳过压缩则可能在处理长上下文时导致内存溢出（OOM）故障。我们识别了在紧张内存预算下同时部署基于vLLM的大语言模型和基于PyTorch的压缩器时的两种不同失败机制，并提出了三指标路由器——一种确定性、无需训练的策略，可在原始、神经（LLMLingua-2）和词法（BM25）三种流水线之间进行选择。该路由器使用三个CPU侧信号：空间复杂度（$L$）、句法密度（$\rho_{key}$）和类符-形符比（TTR）。与以往仅基于语义的自适应方法不同，我们的调度信号是硬件物理层面的，基于显存余量和延迟交叉点。阈值设定……

    arXiv:2609.17564v1 Announce Type: new  Abstract: Deploying retrieval-augmented generation (RAG) on commodity GPUs such as the NVIDIA T4 (16 GB VRAM) exposes a practical failure mode we call the Compression Paradox: neural prompt compression can add key-value (KV) cache contention and preprocessing latency that outweigh generation-time savings, while skipping compression can cause out-of-memory (OOM) failures on long contexts. We identify two distinct failure mechanisms when a vLLM-served LLM and a PyTorch-based compressor are co-deployed under tight memory budgets, and introduce the Tri-Metric Router, a deterministic, training-free policy that selects among Raw, Neural (LLMLingua-2), and Lexical (BM25) pipelines. The router uses three CPU-side signals: spatial complexity ($L$), syntactic density ($\rho_{key}$), and type-token ratio (TTR). Unlike prior semantic-only adaptation, our dispatch signal is hardware-physical, based on VRAM headroom and a latency crossover point. Thresholds are
    
[^16]: 我们能否基于原子命题构建的图来实现可解释的自然语言推理？

    Can We Do Interpretable NLI with Graphs Based on Atomic Propositions?

    [https://arxiv.org/abs/2609.16814](https://arxiv.org/abs/2609.16814)

    本文提出一种完全基于图的可解释自然语言推理流水线，将句子分解为原子命题并转换为ConceptNet三元组构建图表示后在SNLI上达到89.7%的准确率，仅比同条件的文本模型低1.9个百分点，证明了可解释的图表示方法在NLI任务中的可行性。

    

    尽管基于大语言模型（LLM）的自然语言推理（NLI）系统达到了很高的准确率，但其决策过程缺乏可审计的结构。本文探讨了是否可以仅使用可解释的、基于图的证据表示来执行自然语言推理。我们引入了一个完全基于图的流水线，其中分类器从不直接处理输入文本。取而代之的是，句子被分解为原子命题，通过受限解码转换为ConceptNet三元组，并为每个文本对表示为三张图：前提图、假设图以及检索到的ConceptNet子图。随后将这些图输入到一个经过微调的8亿参数语言模型中。在SNLI数据集上，我们的流水线达到了89.7%的准确率，仅比以相同方式训练的基于文本的模型低1.9个百分点。在ANLI上，它在R2和R3轮次上与已发表的RoBERTa-large性能相当（50%准确率），但在R1上落后16个百分点。

    arXiv:2609.16814v1 Announce Type: new  Abstract: While Large Language Model (LLM)-based Natural Language Inference (NLI) systems achieve high accuracy, their decision-making processes lack auditable structures. This paper explores whether NLI can be performed using only interpretable, graph-based representations of evidence. We introduce a fully graph-based pipeline where the classifier never directly processes the input text. Instead, sentences are decomposed into atomic propositions, converted into ConceptNet triples via constrained decoding, and represented as three graphs per pair: premise, hypothesis, and a retrieved ConceptNet subgraph. These graphs are then fed into a fine-tuned 0.8-billion-parameter language model. On the SNLI dataset, our pipeline achieves 89.7% accuracy, just 1.9 points below an identically trained text-based model. On ANLI, it matches the published performance of RoBERTa-large on rounds R2 and R3 (50% accuracy) but trails by 16 points on R1, resulting in an 
    
[^17]: P3Rec：为基于大语言模型的推荐蒸馏先验-后验偏好推理

    P3Rec: Distilling Prior--Posterior Preference Reasoning for LLM-based Recommendation

    [https://arxiv.org/abs/2609.13993](https://arxiv.org/abs/2609.13993)

    提出P3Rec框架，通过联合蒸馏大语言模型中互补的先验偏好（稳定长期兴趣）与后验偏好（目标相关细粒度兴趣）推理知识，克服单一视角蒸馏的局限性，提升轻量级推荐器的性能。

    

    大语言模型（LLM）展现出强大的语义理解和偏好推理能力，为推荐系统中的用户建模提供了新机遇。现有的LLM-as-Enhancer方法通常将大语言模型提炼的偏好知识蒸馏到轻量级推荐器中，以避免昂贵的在线LLM推理。然而，这些方法往往仅从单一视角构建蒸馏知识。先验偏好能够捕捉用户稳定且一致的长期兴趣，但对当前决策提供的指导有限；而后验偏好揭示了与目标相关的细粒度兴趣，却可能过度依赖目标线索。为了解决这些局限性，我们提出了P3Rec框架，该框架联合提取并内化互补的先验偏好和后验偏好推理知识。具体而言，P3Rec首先从用户侧推导出与目标无关的先验偏好以及以目标为条件的后验偏好……

    arXiv:2609.13993v1 Announce Type: new  Abstract: Large language models (LLMs) exhibit strong semantic understanding and preference reasoning capabilities, offering new opportunities for user modeling in recommender systems. Existing LLM-as-Enhancer methods typically distill LLM-derived preference knowledge into lightweight recommenders to avoid costly online LLM inference. However, they often construct distillation knowledge from only one perspective. Prior preference captures users' stable and consistent interests but provides limited guidance for the current decision, whereas posterior preference reveals target-relevant fine-grained interests but may rely excessively on target clues. To address these limitations, we propose P$^3$Rec, a framework that jointly extracts and internalizes complementary prior and posterior preference reasoning knowledge. Specifically, P$^3$Rec first derives target-agnostic prior preferences and target-conditioned posterior preferences from the user side, w
    
[^18]: 哪些历史数据对时间序列预测重要？通过未来监督学习预测相关性

    Which Histories Matter for Time Series Forecasting? Learning Predictive Relevance with Future Supervision

    [https://arxiv.org/abs/2608.23221](https://arxiv.org/abs/2608.23221)

    该论文提出了一种通过未来监督学习预测相关性来重新排名时间序列检索候选的方法，显著提升了模式检索性能。

    

    arXiv:2608.23221v1 公告类型：新 摘要：时间序列预测中的历史检索通常将过去相似性作为有用性的代理。我们提出了一个不同的问题：对于某个查询，哪些历史示例应该被预期会产生影响？我们将预测相关性定义为在推理时信息条件下预期未来效用，仅在训练期间使用实现未来作为特权监督。一个归一化模式检索器首先形成一个粗略的候选集，然后一个轻量级残差多层感知器（MLP）学习一个列表式未来兼容性目标，同时保持推理时评分严格仅基于过去。我们的方法保留了基于相似性的候选生成，但通过更预测性的相关性标准重新排名其候选。最优相关性分解为候选级效用和查询特定兼容性，这启发了候选先验和打乱未来控制。在六个基准上，重排名器改进了模式检索，同时揭示了...

    arXiv:2608.23221v1 Announce Type: new  Abstract: Historical retrieval for time-series prediction commonly treats past similarity as a proxy for usefulness. We ask a different question: which historical examples should be expected to matter for a query? We define predictive relevance as expected future utility conditioned on inference-time information, using realized futures only during training as privileged supervision. A normalized-pattern retriever first forms a coarse candidate set, and a lightweight residual multilayer perceptron (MLP) learns a listwise future-compatibility target while keeping inference-time scoring strictly past-only. Our method retains similarity-based candidate generation but reranks its candidates by a more predictive relevance criterion. Optimal relevance decomposes into candidate-level utility and query-specific compatibility, motivating Candidate-Prior and Shuffled-Future controls. Across six benchmarks, the reranker improves Pattern retrieval while reveal
    
[^19]: 从被忽视到被探索：通过多视角混合恢复物品关系以用于序列推荐

    From Overlooked to Explored: Recovering Item Relations via Mixture of Perspectives for Sequential Recommendation

    [https://arxiv.org/abs/2608.11846](https://arxiv.org/abs/2608.11846)

    本文提出PRISM模块，通过多视角混合方法克服序列推荐中自注意力模型的相似性偏差，从而恢复被忽视的异质物品关系以提升推荐性能。

    

    从用户的交互序列中捕捉用户偏好是序列推荐（SR）的核心挑战。这种偏好直观上源于物品间的关系：每个物品转换反映了嵌入在物品关系中的偏好，因此忠实捕捉这些关系对于准确推荐至关重要。为此，自注意力在序列推荐中占主导地位，因为它能计算成对物品交互，但我们的实证分析揭示，它在各种基于Transformer的SR模型中持续遭受相似性偏差：点积注意力分数不成比例地偏向相似物品，系统性地忽视了具有有意义偏好信号的异质关系，直接限制了推荐性能。为解决这一问题，我们提出了PRISM（基于视角的关系洞察合成模块），一个重新审视物品关系的模块。

    arXiv:2608.11846v1 Announce Type: new  Abstract: Capturing user preference from a user's interaction sequence is the central challenge of Sequential Recommendation (SR). This preference intuitively emerges from inter-item relations: each item transition reflects a preference embedded in the relations between items, making the faithful capture of these relations essential for accurate recommendation. For this reason, self-attention is dominant in sequential recommendation for its ability to compute pairwise item interactions, yet our empirical analysis reveals that it consistently suffers from similarity bias across various types of transformer-based SR models: dot-product attention scores disproportionately favor similar items, systematically overlooking heterogeneous relations with meaningful preference signals and directly limiting recommendation performance. To address this, we propose PRISM (Perspective-based Relational Insight Synthesis Module), a module that re-examines item rela
    
[^20]: Bumblebee：面向大规模推荐系统的交错混合层构建模块

    Bumblebee: Interleaved Mixed-Layer Building Blocks for Large-Scale Recommendation Systems

    [https://arxiv.org/abs/2607.24804](https://arxiv.org/abs/2607.24804)

    提出Bumblebee推荐架构，通过交错可堆叠的混合层构建模块将序列建模与特征交互两种范式有机融合，实现了特征模态的早期与反复混合，从而提升大规模推荐系统的表达能力。

    

    推荐系统在过去几年中经历了重大变革。从传统的特征交互模块到生成式下一动作预测的转变，推动了个性化内容的发展边界。这些发展主要沿着两条独立的轨道演进：一方面是序列建模方法，另一方面是特征交互方法。在本文中，我们提出了Bumblebee，这是一种推荐架构，通过交错的可堆叠块设计解决了这两个方向之间缺乏交互的问题。每个块实现了一个微管道层结构，将序列个性化、基于注意力的编码和特征交叉组合成一个自包含的单元。每个块都会生成两种特征模态的联合表示，该表示被序列中的下一个块所消费。这种机制鼓励模态的早期和重复混合，并丰富了下游（模型的表达能力）。

    arXiv:2607.24804v3 Announce Type: replace-cross  Abstract: Recommendation systems have undergone significant transformations in the past years. The transition from traditional feature interaction modules to generative next-action prediction has pushed the boundaries of personalized content. Developments have largely evolved along two separate tracks. Sequence modeling approaches on the one hand and feature interaction methods on the other. In this paper, we introduce Bumblebee, a recommendation architecture that addresses the lack of interaction between the two directions through an interleaved, stackable block design. Each block implements a micro-pipeline of layers combining sequence personalization, attention-based encoding, and feature crossing into a self-contained unit. Every block produces a joint representation of both feature modalities which is consumed by the next block in the sequence. This mechanism encourages early and repeated mixture of modalities and enriches downstrea
    
[^21]: 基于偏好解耦的时间感知扩散生成式推荐

    Time-Aware Diffusion based on Preference Disentanglement for Generative Recommendation

    [https://arxiv.org/abs/2606.01670](https://arxiv.org/abs/2606.01670)

    该论文提出TDPM框架，将用户偏好解耦并通过时间感知的扩散机制显式建模时间演化的偏好影响，从而克服了现有扩散推荐模型对历史物品统一处理的局限性。

    

    近年来，生成式推荐器（GRs）通过用语义索引（SIDs）取代传统的物品ID，已成为一种变革性的推荐范式。得益于扩散模型卓越的生成能力，一些开创性工作开始探索以扩散架构为骨干来构建生成式推荐器。然而，现有基于扩散的生成式推荐器存在一个致命的局限性：扩散过程均匀地应用于历史交互中的所有物品。相比之下，用户偏好由多方面的时间演化因素所塑造，因此在时间维度上呈现出非平稳分布。为弥补这一不足，本研究提出了一种新颖的生成式推荐框架TDPM，通过在SID令牌上设计时间感知的扩散机制。具体而言，TDPM显式地将时间演化的用户偏好所产生的影响融入扩散过程中。详细来说，用户偏好被解耦为（i）周期性（摘要在此处截断）

    arXiv:2606.01670v2 Announce Type: replace-cross  Abstract: Recently, Generative Recommenders (GRs) have emerged as a transformative recommendation paradigm by replacing traditional item IDs with semantic indices (SIDs). Owing to the exceptional generative capabilities of diffusion models, a few pioneering works explore developing GRs with diffusion architectures as the backbone. However, a fatal limitation of existing diffusion-based GRs is that the diffusion process applies uniformly to all items within the historical interactions. In contrast, the user preference is shaped by multifaceted time-evolving factors and thus exhibits a non-stationary distribution in the temporal aspect. To bridge this gap, this study proposes a novel GR framework, named TDPM, by designing the time-aware diffusion on SID tokens. Specifically, TDPM explicitly integrates the impact of time-evolving user preferences into the diffusion process. In detail, the user preference is disentangled into (i) the period 
    
[^22]: 面向领英信息流排序的工业级序列推荐系统

    An Industrial-Scale Sequential Recommender for LinkedIn Feed Ranking

    [https://arxiv.org/abs/2602.12354](https://arxiv.org/abs/2602.12354)

    领英推出基于Transformer的序列推荐模型Feed SR取代原有DCNv2排序器，在12亿会员规模下成功部署，在线A/B测试中显著提升用户参与度（使用时长+2.10%，互动行为+3.52%）。

    

    领英信息流使全球专业人士能够大规模地发现相关内容、建立联系并分享知识。我们提出了Feed序列推荐系统，这是一个基于Transformer的序列排序模型，用于领英信息流，它取代了基于DCNv2的排序器，并满足了严格的生产约束条件。我们详细介绍了在12亿会员规模上实现部署的建模选择、训练技术和服务优化。Feed SR已为领英信息流的大部分流量服务超过三个月，在与现有生产模型的在线A/B测试中显示出显著的会员参与度提升（使用时长+2.10%，点赞、评论或转发+3.52%）。我们还描述了在部署其他序列模型和基于大语言模型的排序架构方面的经验，以及为什么Feed SR提供了在线指标与生产效率的最佳组合。

    arXiv:2602.12354v3 Announce Type: replace  Abstract: LinkedIn Feed enables professionals worldwide to discover relevant content, build connections, and share knowledge at scale. We present Feed Sequential Recommender (Feed SR), a transformer-based sequential ranking model for LinkedIn Feed that replaces a DCNv2-based ranker and meets strict production constraints. We detail the modeling choices, training techniques, and serving optimizations that enable deployment at a scale of 1.2 billion members. Feed SR has been serving the majority of LinkedIn's Feed traffic for over three months and shows significant improvements in member engagement (+2.10% time spent, +3.52% like, comments, or reshares) in online A/B tests compared to the existing production model. We also describe our deployment experience with alternative sequential and LLM-based ranking architectures and why Feed SR provided the best combination of online metrics and production efficiency.
    
[^23]: 使用歌词对齐音频嵌入的可扩展音乐翻唱检索

    Scalable Music Cover Retrieval Using Lyrics-Aligned Audio Embeddings

    [https://arxiv.org/abs/2601.11262](https://arxiv.org/abs/2601.11262)

    该论文提出利用歌词作为翻唱歌曲间的强不变量，通过歌词对齐的音频嵌入实现高效可扩展的音乐翻唱检索，从而避免现有方法对复杂音频处理流程和高计算资源的需求。

    

    音乐翻唱检索，也称为版本识别，旨在识别同一底层音乐作品的不同演绎版本，这一任务对于曲目目录管理、版权执法和音乐检索至关重要。最先进的方法主要集中于和声与旋律特征，采用日益复杂的音频处理流程，这些流程被设计为对翻唱版本间常常差异巨大的音乐属性保持不变性。尽管有效，但这些方法需要大量的训练时间和计算资源。相比之下，歌词是翻唱版本间的一个强不变量，但其应用一直受到从复调音频中准确且高效提取歌词这一难题的限制。早期方法依赖于简单的框架，从而限制了下游性能，而较新的系统虽然能提供更强的结果，但需要将大型模型集成到复杂的多模态架构中。我们提出了LIVI（Lyr（摘要在此处截断）

    arXiv:2601.11262v2 Announce Type: replace-cross  Abstract: Music Cover Retrieval, also known as Version Identification, aims to recognize distinct renditions of the same underlying musical work, a task central to catalog management, copyright enforcement, and music retrieval. State-of-the-art approaches have largely focused on harmonic and melodic features, employing increasingly complex audio pipelines designed to be invariant to musical attributes that often vary widely across covers. While effective, these methods demand substantial training time and computational resources. By contrast, lyrics constitute a strong invariant across covers, though their use has been limited by the difficulty of extracting them accurately and efficiently from polyphonic audio. Early methods relied on simple frameworks that limited downstream performance, while more recent systems deliver stronger results but require large models integrated within complex multimodal architectures. We introduce LIVI (Lyr
    
[^24]: 看穿MiRAGE：多模态检索增强生成的评估

    Seeing Through the MiRAGE: Evaluating Multimodal Retrieval Augmented Generation

    [https://arxiv.org/abs/2510.24870](https://arxiv.org/abs/2510.24870)

    MiRAGE是一个以论断为中心的多模态检索增强生成评估框架，通过InfoF1和CiteF1指标评估事实性、信息覆盖度与引用完整性，在文本任务上优于现有RAG评估指标，且是唯一能够推广到多模态来源的方法。

    

    我们提出了MiRAGE，一个针对多模态来源检索增强生成（RAG）的评估框架。随着视听媒体日益成为网络上普遍的信息来源，RAG系统必须将此类媒体整合到生成过程中。然而，现有的RAG评估方法主要以文本为中心，难以直接迁移到多模态场景。MiRAGE是一种以论断为中心的多模态RAG评估方法，由两部分组成：InfoF1用于评估事实性与信息覆盖度，CiteF1用于评估引用的支持性与完整性。研究表明，当由人类使用时，MiRAGE与输出质量的外部判断高度一致。此外，我们还介绍了MiRAGE的自动实现版本，并将其与三个著名的以文本为中心的RAG指标的多模态变体——ALCE、ARGUE和RAGAS——进行比较，发现MiRAGE在文本任务上优于这三个指标，并且是唯一能够推广到多模态来源的评估方法。

    arXiv:2510.24870v3 Announce Type: replace  Abstract: We introduce MiRAGE, an evaluation framework for retrieval-augmented generation (RAG) from multimodal sources. As audiovisual media becomes a more prevalent source of information online, RAG systems must integrate such media into generation. Yet, existing evaluation methods for RAG are largely text-centric and do not readily transfer to multimodal settings. MiRAGE is a claim-centric approach to multimodal RAG evaluation, consisting of InfoF1, which assesses factuality and information coverage, and CiteF1, which assesses citation support and completeness. We show that, when applied by humans, MiRAGE strongly aligns with extrinsic judgments of output quality. We additionally introduce an automatic implementation of MiRAGE and compare it to multimodal variants of three prominent text-centric RAG metrics---ALCE, ARGUE, and RAGAS---finding that MiRAGE outperforms all three on text while being the only one to generalize to multimodal sourc
    
[^25]: 基于反事实推断的点击后转化率预测研究

    On Predicting Post-Click Conversion Rate via Counterfactual Inference

    [https://arxiv.org/abs/2510.04816](https://arxiv.org/abs/2510.04816)

    该论文提出了一种以因果性为指导原则的反事实推断方法，通过为未点击样本反事实地生成转化标签，从而缓解点击样本稀疏性问题并改进点击后转化率预测。

    

    准确预测转化率（CVR）在在线广告系统和电子商务等各类推荐领域中至关重要。这些系统利用由曝光、点击和转化构成的用户交互日志。由于转化只能在点击之后才能确定，CVR预测模型通常仅基于点击样本进行训练。然而，点击样本的稀疏性导致需要收集大量日志才能有效训练模型。近期的研究通过设计利用未点击样本的框架来解决这一问题。虽然这些框架旨在减少由点击样本与未点击样本之间差异所引起的偏差，但它们往往依赖于启发式方法。在此背景下，我们提出一种方法，以因果性为指导原则，通过反事实方式为未点击样本生成转化标签，试图回答“如果用户点击了，他们是否会转化？”这一问题。

    arXiv:2510.04816v1 Announce Type: cross  Abstract: Accurately predicting conversion rate (CVR) is essential in various recommendation domains such as online advertising systems and e-commerce. These systems utilize user interaction logs, which consist of exposures, clicks, and conversions. CVR prediction models are typically trained solely based on clicked samples, as conversions can only be determined following clicks. However, the sparsity of clicked instances necessitates the collection of a substantial amount of logs for effective model training. Recent works address this issue by devising frameworks that leverage non-clicked samples. While these frameworks aim to reduce biases caused by the discrepancy between clicked and non-clicked samples, they often rely on heuristics. Against this background, we propose a method to counterfactually generate conversion labels for non-clicked samples by using causality as a guiding principle, attempting to answer the question, "Would the user h
    
[^26]: 通过协调双重动态索引机制释放大语言模型在序列推荐中的潜力

    Unleash LLMs Potential for Sequential Recommendation by Coordinating Dual Dynamic Index Mechanism

    [https://arxiv.org/abs/2409.09253](https://arxiv.org/abs/2409.09253)

    该论文提出了首个采用双重动态索引机制的端到端大语言模型序列推荐系统ED²，将索引生成与序列推荐统一到单一LLM主干流水线中，同时解决了语义信息与协同信息整合不足以及高阶用户-物品交互模式利用不充分的问题。

    

    由于在语义理解和逻辑推理方面具有前所未有的能力，大语言模型（LLMs）在开发下一代序列推荐系统（RSs）方面展现出了巨大的潜力。然而，现有的基于LLM的序列推荐系统大多将索引生成与序列推荐相分离，导致语义信息与协同信息之间的整合不足。另一方面，对用户相关信息的忽视阻碍了基于LLM的序列推荐系统利用高阶用户-物品交互模式。在本文中，我们提出了端到端双重动态（ED²）推荐器，这是首个采用双重动态索引机制的基于LLM的序列推荐系统，旨在同时解决上述局限性。双重动态索引机制不仅能够将索引生成和序列推荐整合到统一的以LLM为主干的流水线中，还能使其……

    arXiv:2409.09253v2 Announce Type: replace-cross  Abstract: Owing to the unprecedented capability in semantic understanding and logical reasoning, large language models (LLMs) have shown fantastic potential in developing next-generation sequential recommender systems (RSs). However, existing LLM-based sequential RSs mostly separate index generation from sequential recommendation, leading to insufficient integration between semantic information and collaborative information. On the other hand, the neglect of user-related information hinders LLM-based sequential RSs from exploiting high-order user-item interaction patterns. In this paper, we propose the End-to-End Dual Dynamic (ED$^2$) recommender, the first LLM-based sequential RS which adopts dual dynamic index mechanism, targeting resolving the above limitations simultaneously. The dual dynamic index mechanism can not only assembly index generation and sequential recommendation into a unified LLM-backbone pipeline, but also make it pra
    
[^27]: 模式链接的消亡？善于推理的语言模型时代的Text-to-SQL

    The Death of Schema Linking? Text-to-SQL in the Age of Well-Reasoned Language Models

    [https://arxiv.org/abs/2408.07702](https://arxiv.org/abs/2408.07702)

    最新的大语言模型即使面对大量不相关模式元素也能准确利用所需信息，因此当数据库模式能放入上下文窗口时，Text-to-SQL流程可完全省去模式链接步骤，转而采用增强、选择和纠正等技术来提升查询生成的准确性。

    

    模式链接（Schema Linking）是Text-to-SQL流程中的关键步骤。其目标是为用户的查询检索目标数据库中相关的表和列，同时忽略不相关的部分。然而，不完善的模式链接常常会排除生成准确查询所需的列。在这项工作中，我们在使用最新一代大语言模型（LLMs）时重新审视了模式链接。我们通过实证发现，较新的模型即使在大量不相关元素存在的情况下，也善于在生成过程中利用相关的模式元素。因此，当模式能够完全容纳在模型的上下文窗口内时，我们的Text-to-SQL流程完全放弃模式链接，以最大限度地减少因过滤掉所需模式元素而导致的问题。此外，我们不采用过滤上下文信息的方式，而是强调增强、选择和纠正等技术，并采用这些技术来提高我们Text-to-SQL系统的准确性。

    arXiv:2408.07702v2 Announce Type: cross  Abstract: Schema linking is a crucial step in Text-to-SQL pipelines. Its goal is to retrieve the relevant tables and columns of a target database for a user's query while disregarding irrelevant ones. However, imperfect schema linking can often exclude required columns needed for accurate query generation. In this work, we revisit schema linking when using the latest generation of large language models (LLMs). We find empirically that newer models are adept at utilizing relevant schema elements during generation even in the presence of large numbers of irrelevant ones. As such, our Text-to-SQL pipeline entirely forgoes schema linking in cases where the schema fits within the model's context window in order to minimize issues due to filtering required schema elements. Furthermore, instead of filtering contextual information, we highlight techniques such as augmentation, selection, and correction, and adopt them to improve the accuracy of our Text
    
[^28]: 科学文献中的引用归因：新基准与方法

    Attribution in Scientific Literature: New Benchmark and Methods

    [https://arxiv.org/abs/2405.02228](https://arxiv.org/abs/2405.02228)

    该论文提出了科学引用归因基准REASONS以及由弃答率与幻觉率构成的双指标评估框架，系统评估了大语言模型在不同证据条件下的引用归因能力，发现高级RAG虽能降低幻觉率但牺牲了弃答能力，而对抗性元数据会使多个系统的幻觉率超过85%。

    

    大型语言模型越来越多地生成带有引用支持的回答，然而引用幻觉仍然是可信科学信息获取面临的主要挑战。我们提出了REASONS，这是一个包含12,723个句子级引用实例的基准，涵盖12个arXiv学科类别，旨在评估不同证据条件下的科学引用归因。我们提出了一个由弃答率和幻觉率组成的双指标框架，用于刻画可靠性与响应性之间的权衡。通过作者归因和标题归因任务，我们在零上下文、元数据增强、级联元数据增强提示（CMP）、检索增强和对抗等多种设置下评估了专有和开源大语言模型。高级RAG相对于朴素RAG降低了幻觉率（65.4%对比87.6%），但使弃答率从5.0%降至0%。在对抗性元数据下，多个系统的幻觉率超过85%。

    arXiv:2405.02228v4 Announce Type: replace-cross  Abstract: Large language models (LLMs) increasingly generate citation-backed responses, yet citation hallucination remains a major challenge for trustworthy scientific information access. We introduce REASONS, a benchmark of 12,723 sentence-level citation instances spanning 12 arXiv subject categories, designed to evaluate scientific citation attribution under varying evidence conditions. We propose a dual-metric framework consisting of Abstention Rate (AR) and Hallucination Rate (HR) to characterize the trade-off between reliability and responsiveness. Using author-attribution and title-attribution tasks, we evaluate proprietary and open-source LLMs under zero-context, metadata-augmented, cascaded metadata-augmented prompting (CMP), retrieval-augmented, and adversarial settings. Advanced RAG lowers HR relative to Naive RAG (65.4% vs. 87.6%) but reduces AR from 5.0% to 0%. Under adversarial metadata, several systems exceed 85% HR, while 
    

