# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Variants of Beam Search for DiskANN](https://arxiv.org/abs/2611.26609) | 本文统一并推广了贪心搜索与自适应束搜索在 α-可达图上的理论保证，证明了自适应束搜索在任意度量下可实现 (α+1)/(α(γ+1)-1) 的近似因子，填补了贪心搜索在可导航图上无保证的空白。 |
| [^2] | [Cost-Aware Agentic Feature Optimization System for Fleet-Scale Ranking Models](https://arxiv.org/abs/2611.26531) | 该论文提出了一个端到端的LLM智能体系统，在跨模型共享与逐模型设定的资源约束下，为大规模排序模型集群自动化地完成假设生成、交叉特征生成与筛选、多方法重要性排序融合以及异步试验调优，实现成本感知的跨模型特征优化。 |
| [^3] | [The Window Lever: Longer Observation Windows Outweigh the Observed Effects of Data, Capacity, and Architecture in Customer Behavior Forecasting](https://arxiv.org/abs/2611.26521) | 在长时程客户行为预测中，将观察窗口从8周延长到16周所带来的性能提升超过了增加数据量、扩大模型容量或采用新架构的效果，且匹配测试表明这一增益甚至被原始比较所低估。 |
| [^4] | [Democratizing Cross-Project Knowledge for Transforming Water Governance and Climate Resilience: an Agentic AI Ecosystem](https://arxiv.org/abs/2611.26398) | 本文提出了一个智能体AI生态系统，将三个欧盟项目的异构文档整合为领域感知知识库，结合领域微调嵌入模型与ReAct式智能体检索增强生成流水线，实现跨项目知识的自适应访问，以服务水治理与气候韧性决策。 |
| [^5] | [From Sessions to Serving: Integrating LLM-Based Embeddings into Retrieval for Production Content Recommendation](https://arxiv.org/abs/2611.26096) | 本文提出LSER生产级内容推荐系统，将预训练大语言模型生成的会话级和历史级用户嵌入作为辅助特征集成到检索阶段用户塔中，并通过近线异步服务路径保持嵌入实时更新。 |
| [^6] | [A Task-Level Bias Taxonomy for Historical Image Retrieval](https://arxiv.org/abs/2611.26019) | 本文针对历史图像检索场景提出了一种任务级的偏差分类法，系统梳理了AI系统在历史图像搜索与检索中可能引入偏差的各类因素。 |
| [^7] | [LINEAGE: A Benchmark for Empirical Lineage Retrieval in AI Scientific Literature](https://arxiv.org/abs/2611.25845) | 提出了Lineage基准，用于在AI科学文献中检索实证谱系关系（即先前论文的方法被后续论文明确用作实证比较基线），并开发了Lineage-Constructor工具来大规模构建谱系图，为结构性检索评估提供真值。 |
| [^8] | [CACO: Coverage-Aware Coordinated Optimization for Continuous Generative Retrieval](https://arxiv.org/abs/2611.25738) | CACO是一种轻量级自监督后训练框架，通过覆盖感知的无标签目标将多个采样查询协调至互补的物品覆盖，在几乎不增加推理延迟的情况下显著提升连续生成式检索的召回率。 |
| [^9] | [Can LLMs Resolve Entity Conflicts in Domain Specific NLU Pipelines?](https://arxiv.org/abs/2611.25733) | 本研究通过比较基于规则的冲突解决方法与两种基于大语言模型的方法（多轮提示和LoRA微调的Llama3.1-8B-Instruct），验证了大语言模型在解决领域特定NLU流水线中实体冲突问题上的可行性。 |
| [^10] | [STARK: Moving towards Personalized Agents That Truly Learn and Remember](https://arxiv.org/abs/2611.25705) | STARK提出了一种双图记忆架构，通过情景图记录用户交互事件、洞察图提取并连接行为模式，使个性化智能体能够真正学习和记住用户的行为规律。 |
| [^11] | [Exploring Semantic Table Search in Data Lakes](https://arxiv.org/abs/2611.25691) | 该论文提出了Thetis++系统，将关键词搜索与基于命名实体的语义搜索相结合，通过融合多种排名组合技术，更有效地在数据湖中发现相关表格数据集。 |
| [^12] | [DNA: Unlocking Depth Scaling in CTR Models](https://arxiv.org/abs/2611.25659) | 提出DNA（双流枢纽注意力）方法，通过让用户行为序列随模型深度逐层演化更新，而非每层重复投影同一固定序列，突破了CTR模型中交叉注意力堆叠的深度扩展瓶颈。 |
| [^13] | [Uncontrolled source titles: When Nature, Science, and PNAS meet Scopus](https://arxiv.org/abs/2611.25627) | Scopus数据库中来源刊名缺乏规范控制，导致Nature、Science和PNAS三大期刊的检索结果碎片化且不一致，而Web of Science中不存在此类问题，研究并揭示了刊名变异源于上游数据商继承、自动解析和OCR错误及验证不足等机制。 |
| [^14] | [Generative News Recommendation Reproduces Popularity: A Memorization-Based Analysis](https://arxiv.org/abs/2611.25606) | 该研究利用约2000万条日本新闻浏览会话数据，首次从训练数据记忆化的角度揭示生成式新闻推荐再现流行内容的机制——被频繁点击的新闻标题更容易被模型记住，并在更多样化的用户上下文中被生成，从而导致信息多样性下降。 |
| [^15] | [AutoRAGBench: A Benchmark for LLM Agents Automating RAG System Development](https://arxiv.org/abs/2611.25582) | 本文提出了首个基准测试AutoRAGBench，通过九个任务评估LLM智能体在无起始代码和预设策略的条件下，于开放代码空间中自主从零构建并迭代优化RAG系统的能力。 |
| [^16] | [Learning Generative Ranking Policies for Whole-Page Optimization](https://arxiv.org/abs/2611.25527) | 提出一种局部页面级训练目标，可精确拆分为逐位置的训练信号，使生成式排序策略在整页优化中动态生成更高分的页面，同时克服了现有方法中统一页面级信号和固定模仿目标的双重局限。 |
| [^17] | [VideoTrail: Interactive Video Foraging Through Hierarchical Visualization](https://arxiv.org/abs/2611.25377) | VideoTrail是一个基于信息觅食理论的AI浏览器扩展，通过多模态自适应的层次化可视化将视频中的语义与结构线索呈现为“信息气味”，帮助用户高效定位关键时刻并支持目标导向的视频信息搜寻。 |
| [^18] | [LQ-Rec: Late Quantization for Sequential Recommendation](https://arxiv.org/abs/2611.25256) | 该论文提出“后期量化”这一非自回归范式，将序列推荐中生成式检索的离散化限制在物品输出端，从而突破自回归语义ID的结构性扩展限制，并通过加性误差分解与成对排序定理为量化引起的排序损失提供了以往方法所不具备的理论保证。 |
| [^19] | [Retrieving Patristic References and Modelling Topics in Bullinger's Reformation Correspondence](https://arxiv.org/abs/2611.25247) | 本研究首次将针对拉丁语圣经互文性微调的检索模型迁移至布林格宗教改革书信中的教父引用检测任务，通过比较词汇、稠密与混合检索方法，发现在圣经复用上微调的拉丁语编码器在双编码器中表现最佳，而 BM25 在显式引用检测上更具优势。 |
| [^20] | [LSP-RAG: Latent Scoring Perspectives for Knowledge Graph Retrieval-Augmented Generation](https://arxiv.org/abs/2611.25229) | 提出LSP-RAG框架，通过多视角提示评分增强知识图谱检索增强生成中的逐跳知识检索，在不依赖大规模语言模型的情况下降低计算成本并提升多跳场景下的检索质量。 |
| [^21] | [FinVeriQA: Financial Research Agents for Question Answering: Routing, Retrieval, MCP Tool Use, and Provenance-Aware Evidence Verification](https://arxiv.org/abs/2611.25071) | 该论文提出FinVeriQA——一个基于SEC文件与实时市场数据的开源多智能体金融问答系统，其核心创新在于用紧凑的专用决策模型Laya进行问题路由与建议过滤，配合Python精确计算、MCP工具检索与溯源感知的证据验证，实验证明小型决策模型的路由性能可显著超越大型语言模型，并在决策准确性与推理速度之间取得最佳权衡。 |
| [^22] | [Variance-Optimal Counterfactual Policy Design for Sample-Efficient Online A/B Testing](https://arxiv.org/abs/2611.24757) | 该论文提出一种方差最优的反事实策略设计方法，通过联合优化动作的采样方式，将流量引导至两种策略不一致的动作上，从而显著降低在线A/B测试的方差并提升样本效率。 |
| [^23] | [IntentFlow: An Intent-Driven Real-Time Multimodal Search System with Image-Text Queries at Xiaohongshu](https://arxiv.org/abs/2611.20448) | IntentFlow是小红书部署的意图驱动实时多模态搜索系统，通过层次化意图分类体系和意图特定的相关性标准，在百亿级索引上实现了图文查询场景下候选检索与重排序的一致性对齐。 |
| [^24] | [Keep it Simple! A Multiple-Choice Task can Benefit Generative Recommendation](https://arxiv.org/abs/2611.19320) | 该论文提出只需在生成式推荐的学习目标中加入一个简单的多项选择题辅助任务，即可显式引入负样本和与BPR理念一致的相对排序信号，从而在不依赖复杂架构和大量调参的情况下有效提升推荐效果。 |
| [^25] | [When Is Additional Retrieval Worthwhile? A Cost-Aware Study of Evidence Gain and Answer Reliability in RAG](https://arxiv.org/abs/2610.37358) | 该论文提出一种成本感知的检索路由策略，通过在路由前预估混合检索相对于稠密检索基线的问题级增量证据收益，仅在约四分之一的问题上触发更昂贵的混合检索，在保持相同证据Hit@10的同时显著降低检索成本。 |
| [^26] | [The Missing Complement: State-Conditioned Minimal Sufficient Evidence for Coding Agents](https://arxiv.org/abs/2609.20050) | 该论文提出了状态条件化最小充分证据恢复这一新问题并构建了SERBench基准，同时提出MSS-Complement方法，将证据获取从排序转变为集合构建，为编码代理的决策恢复紧凑且充分的证据组合。 |
| [^27] | [Stop Removing Stopwords: How an Inherited Preprocessing Default Distorts Legal Text-as-Data](https://arxiv.org/abs/2609.19153) | 本研究通过穷尽式单词消融实验，首次直接针对下游分类目标验证停用词去除这一沿袭自信息检索时代、从未被验证过的预处理默认设置，揭示其可能扭曲实证法学中基于TF-IDF和线性分类器的文本数据分析结果。 |
| [^28] | [From transcription to semantic corpus analysis: unsupervised learning of sentence representations for ancient languages](https://arxiv.org/abs/2607.24542) | 本文提出两种完全无监督的方法（TSDAE 和无监督 SimCSE），仅用原始句子将面向古代语言的专用词元级语言模型转化为语料库专属的句子编码器，并在拉丁语和古希腊语教父文献中经专家验证的圣经文本复用识别任务上验证了其有效性。 |
| [^29] | [Beyond Dimensionality: Domain Shift and Catastrophic Forgetting in Single- vs. Multi-Vector Retrieval](https://arxiv.org/abs/2603.29519) | 该研究反驳了嵌入维度是单向量检索模型失效主因的观点，实证表明领域偏移以及嵌入相似性与任务相关性之间的错位才是导致检索质量下降的主导因素。 |
| [^30] | [Can Explanations Improve Recommendations? Evidence from Prediction-Informed Explanations](https://arxiv.org/abs/2502.16759) | 该论文提出RecPIE框架，将自然语言解释生成嵌入推荐学习循环，通过预测引导解释、解释反馈优化预测的交替训练机制，同时提升推荐的准确性与可解释性。 |

# 详细

[^1]: DiskANN 的束搜索变体

    Variants of Beam Search for DiskANN

    [https://arxiv.org/abs/2611.26609](https://arxiv.org/abs/2611.26609)

    本文统一并推广了贪心搜索与自适应束搜索在 α-可达图上的理论保证，证明了自适应束搜索在任意度量下可实现 (α+1)/(α(γ+1)-1) 的近似因子，填补了贪心搜索在可导航图上无保证的空白。

    

    基于图的数据结构（如 DiskANN、HNSW 和 NSG）构成了大规模近似最近邻（ANN）搜索的最先进技术。Indyk 和 Xu 引入了 α-可达图的概念，其中包括由 DiskANN 的慢预处理构建方法所构建的图，并证明了贪心搜索可以实现 (α+1)/(α-1) 的近似因子。与此同时，Al-Jazzazi 等人独立提出了带松弛参数 γ > 0 的自适应束搜索（ABS），证明了在可导航图（α ≈ 1，此时贪心搜索不提供任何保证）上可实现 2/γ 的近似。在本工作中，我们统一并推广了这些结果，证明了在 α-可达图上使用参数 γ 的 ABS 在任意度量下可实现 (α+1)/(α(γ+1)-1) 的近似。此外，对于在 ℓ₂ 度量下的排序 α-可达图（由 Gollapudi 等人引入），我们得到了……

    arXiv:2611.26609v1 Announce Type: cross  Abstract: Graph-based data structures such as DiskANN, HNSW, and NSG constitute the state of the art for large-scale approximate nearest-neighbor (ANN) search. Indyk and Xu introduced the notion of $\alpha$-reachable graphs, which include graphs built by DiskANN's slow-preprocessing construction, and proved that greedy search achieves an approximation factor of $\tfrac{\alpha+1}{\alpha-1}$. Independently, Al-Jazzazi et al. proposed adaptive beam search (ABS) with a relaxation parameter $\gamma > 0$, showing a $\tfrac{2}{\gamma}$ approximation on navigable graphs ($\alpha \approx 1$), where greedy search admits no guarantees. In this work, we unify and generalize these results by proving that ABS with parameter $\gamma$ on $\alpha$-reachable graphs achieves an approximation of $\tfrac{\alpha+1}{\alpha(\gamma+1)-1}$ for any metric. Moreover, on sorted $\alpha$-reachable graphs (introduced by Gollapudi et al.) under the $\ell_2$ metric, we obtain t
    
[^2]: 面向集群规模排序模型的成本感知智能体特征优化系统

    Cost-Aware Agentic Feature Optimization System for Fleet-Scale Ranking Models

    [https://arxiv.org/abs/2611.26531](https://arxiv.org/abs/2611.26531)

    该论文提出了一个端到端的LLM智能体系统，在跨模型共享与逐模型设定的资源约束下，为大规模排序模型集群自动化地完成假设生成、交叉特征生成与筛选、多方法重要性排序融合以及异步试验调优，实现成本感知的跨模型特征优化。

    

    大型网络平台运行着大规模的排序模型集群，每个模型都从包含数万个特征的共享特征池中选取特征。我们提出了一种端到端的智能体系统，在跨模型共享的资源约束和逐模型设定的资源约束（分别为特征管线成本与服务吞吐量QPS）下，对整个模型集群的特征进行优化。在该系统中，LLM智能体负责提出假设，确定性工具负责执行实验并验证假设。对于每个模型，LLM从组合式候选空间中生成并初筛交叉特征，随后采用多种重要性方法对所有特征进行排序。特征融合将上述多个排序混合为单一选择，对于多任务模型则利用跨任务的帕累托前沿。接着，LLM研究员智能体在资源上限约束下通过10至25次异步试验对特征选择进行调优，并由修剪步骤确保管线成本保持在预算之内。在生产模型上，该系统……（原文摘要在此处截断）

    arXiv:2611.26531v1 Announce Type: cross  Abstract: Large Web platforms operate large fleets of ranking models, each selecting features from a shared pool of tens of thousands. We present an end-to-end agentic system that optimizes features across the fleet under resource constraints shared across models and those set per model, such as feature pipeline cost and serving throughput (QPS), respectively. In this system, LLM agents propose hypotheses, and deterministic tools execute experiments and verify them. For each model, an LLM generates and shortlists cross features from a combinatorial candidate space, and all features are then ranked by several importance methods. Feature blending mixes these rankings into a single selection, using Pareto fronts across tasks for multi-task models. An LLM Researcher agent then tunes the feature selection in 10--25 asynchronous trials under resource caps, and a trimming step keeps the pipeline cost within budget. Across production models, the system 
    
[^3]: 窗口杠杆：在客户行为预测中，更长的观察窗口的影响超过数据量、模型容量与架构的效果

    The Window Lever: Longer Observation Windows Outweigh the Observed Effects of Data, Capacity, and Architecture in Customer Behavior Forecasting

    [https://arxiv.org/abs/2611.26521](https://arxiv.org/abs/2611.26521)

    在长时程客户行为预测中，将观察窗口从8周延长到16周所带来的性能提升超过了增加数据量、扩大模型容量或采用新架构的效果，且匹配测试表明这一增益甚至被原始比较所低估。

    

    在我们关于长时程客户行为预测的实验中，更长的观察窗口所产生的效果超过了更多数据、更大模型容量和更新架构所带来的效应。所谓窗口，是指模型在预测目标周之前能观察到的历史周数。在100万客户规模下，将窗口从8周延长到16周，在所有模型族中都带来了 +0.070 到 0.107 的对数空间 R² 原始提升。匹配测试队列（在每个窗口下对相同的客户-周进行评分）表明，这种原始比较实际上低估了其涵盖的每个模型族的增益。这一方向性结论在五个公开数据集上均成立，其中四个数据集的时间跨度超过一年（岭回归和我们的transformer在全部五个数据集上进行了验证）。在我们的内部数据上，更长的窗口允许纳入最多多10%的、更“冷”（历史更少）的训练客户，而匹配方法无法消除这一效应。我们的设计并未将窗口长度与日历覆盖范围分离开，因为每个测试锚点（即进行预测的周）都落在同一个八周区间内。在训练期锚点上，每周岭回归……（原文在此处截断）

    arXiv:2611.26521v1 Announce Type: new  Abstract: In our experiments on long-horizon customer-behavior forecasting, a longer observation window outweighed the observed effects of more data, more capacity and newer architectures. The window is the weeks of history a model sees before the week it forecasts from. Going from an 8- to a 16-week window adds a raw +0.070 to 0.107 log-space $R^2$ in every family at 1M. Matched test cohorts, which score the same customer-weeks at every window, show this raw comparison understates the gain in every family they cover. The direction holds on five public datasets, four spanning over a year (ridge and our transformer on all five). On our internal data, longer windows admit up to 10% more and colder training customers, which matching does not fix. Our design does not separate window length from calendar coverage, because every test anchor (the week a forecast is made from) falls in one eight-week band. On training-period anchors, the weekly ridge prob
    
[^4]: 民主化跨项目知识以变革水治理与气候韧性：一个智能体AI生态系统

    Democratizing Cross-Project Knowledge for Transforming Water Governance and Climate Resilience: an Agentic AI Ecosystem

    [https://arxiv.org/abs/2611.26398](https://arxiv.org/abs/2611.26398)

    本文提出了一个智能体AI生态系统，将三个欧盟项目的异构文档整合为领域感知知识库，结合领域微调嵌入模型与ReAct式智能体检索增强生成流水线，实现跨项目知识的自适应访问，以服务水治理与气候韧性决策。

    

    欧盟资助的项目以交付成果、报告和案例研究的形式产生了大量证据。然而，这些知识通常分散在不同项目和文档结构之中，使得决策时难以定位相关且具有情境的信息。本文提出了一个智能体AI生态系统，它将异构的项目文档整合为一个领域感知的知识库，并通过智能体检索增强生成流水线提供自适应访问。来自三个欧盟资助项目（IMPETUS、NATALIE和InnWater）的文档被重构为层次化章节树，并使用结构化元数据进行索引。检索由两个在领域数据（水和气候）上微调的领域特定嵌入模型提供支持。在该知识库之上，一个ReAct风格的控制器通过在领域集合之间分配检索预算并积累证据，迭代地编排检索动作。

    arXiv:2611.26398v1 Announce Type: cross  Abstract: EU-funded projects produce extensive evidence in the form of deliverables, reports, and case studies. However, this knowledge is typically scattered across projects and document structures, making it difficult to locate relevant, contextual information when making decisions. This paper introduces an Agentic AI Ecosystem that consolidates heterogeneous project documentation into a domain-aware knowledge base and provides adaptive access through an agentic Retrieval-Augmented Generation pipeline. Documents from three EU-funded projects (IMPETUS, NATALIE, and InnWater) are reconstructed into hierarchical section trees and indexed with structured metadata. Retrieval is supported by two domain-specific embedding models fine-tuned on domain data (water and climate). On top of this knowledge base, a ReAct-style controller iteratively orchestrates retrieval actions by allocating retrieval budget across domain collections and accumulating evide
    
[^5]: 从会话到服务：将基于大语言模型的嵌入集成到生产级内容推荐的检索阶段

    From Sessions to Serving: Integrating LLM-Based Embeddings into Retrieval for Production Content Recommendation

    [https://arxiv.org/abs/2611.26096](https://arxiv.org/abs/2611.26096)

    本文提出LSER生产级内容推荐系统，将预训练大语言模型生成的会话级和历史级用户嵌入作为辅助特征集成到检索阶段用户塔中，并通过近线异步服务路径保持嵌入实时更新。

    

    大型市场网站上的个性化内容推荐必须将异构的候选内容（例如客户故事、产品页面、行业页面和学习内容）与稀疏且快速变化的用户行为相协调，这使得检索阶段不仅要服务于下游排序，还需更好地表示用户兴趣，其要求日益提高。我们提出了LSER（LLM Session Embedding Retrieval，基于大语言模型的会话嵌入检索），这是一个面向autodesk.com的生产级内容推荐系统，它从预训练大语言模型（LLM）中提取会话级和历史级嵌入，并将其作为辅助用户侧特征添加到检索阶段的用户塔中，与现有的已学习序列编码器并存。为了在实时行为下保持用户侧嵌入的新鲜度，我们构建了一条近线服务路径，异步刷新主内容推荐服务器的序列向量以及LLM导出的用户向量。我们验证了LLM特征的增量价值（摘要在此处截断）

    arXiv:2611.26096v1 Announce Type: cross  Abstract: Personalized content recommendation on large marketplace websites must reconcile heterogeneous candidate content (e.g., customer stories, product pages, industry pages, and learning content) with sparse, fast-changing user behavior, placing increasing demands on the retrieval stage to represent user interest well, not just the downstream ranking stage. We present LSER (LLM Session Embedding Retrieval), a production content recommendation system for autodesk.com that derives session- and history-level embeddings from a pretrained large language model (LLM) and adds them as auxiliary user-side features in the retrieval-stage user tower alongside existing learned sequence encoders. To keep user-side embeddings fresh under real-time behavior, we build a nearline serving path that asynchronously refreshes sequential and LLM-derived user vectors for our main content recommendation server. We validate the incremental value of the LLM features
    
[^6]: 面向历史图像检索的任务级偏差分类法

    A Task-Level Bias Taxonomy for Historical Image Retrieval

    [https://arxiv.org/abs/2611.26019](https://arxiv.org/abs/2611.26019)

    本文针对历史图像检索场景提出了一种任务级的偏差分类法，系统梳理了AI系统在历史图像搜索与检索中可能引入偏差的各类因素。

    

    近几十年来，画廊、图书馆、档案馆和博物馆（GLAM机构）已经数字化了大量文化遗产资料，包括历史图像。让这些馆藏能够被公众访问的一大挑战在于实现有效的搜索与发现。许多历史图像包含不完整、稀疏或缺失的描述性元数据。因此，机器学习（ML）方法正被越来越多地用于生成或丰富元数据，以提升这类馆藏的可搜索性。然而，基于人工智能（AI）的系统可能会重现或放大现有偏差，尤其是在应用于历史图像时。诸如训练模型所用数据与实际应用的历史图像之间的差异、图像质量退化、单色照片，以及历史上存在偏见或不完整的描述等因素，都可能降低模型性能，并在搜索与检索中引入偏差。

    arXiv:2611.26019v1 Announce Type: new  Abstract: In recent decades, galleries, libraries, archives, and museums (GLAM institutions) have digitised vast amounts of cultural heritage material, including historical imagery. A major challenge in making these collections publicly accessible is enabling effective search and discovery. Many historical images contain incomplete, sparse, or no descriptive metadata. Consequently, machine learning (ML) methods are increasingly being used to generate or enrich metadata to improve the searchability of such collections. However, artificial intelligence (AI)-based systems can reproduce or amplify existing biases, particularly when applied to historical imagery. Factors such as differences between the data used to train the models and the historical images on which they are applied, degraded image quality, monochrome photographs, and historically biased or incomplete descriptions can reduce model performance and introduce bias into search and retrieva
    
[^7]: LINEAGE：AI科学文献中实证谱系检索的基准

    LINEAGE: A Benchmark for Empirical Lineage Retrieval in AI Scientific Literature

    [https://arxiv.org/abs/2611.25845](https://arxiv.org/abs/2611.25845)

    提出了Lineage基准，用于在AI科学文献中检索实证谱系关系（即先前论文的方法被后续论文明确用作实证比较基线），并开发了Lineage-Constructor工具来大规模构建谱系图，为结构性检索评估提供真值。

    

    当前的研究智能体正从单纯的学术搜索迈向文献综合与实验规划。这些工作流日益依赖文献检索系统，检索系统既可以作为独立系统运行，也可以作为研究智能体的组成部分。然而，现有的科学文献检索基准主要使用主题相关性或引用链接作为监督信号来评估这些方法，无法捕捉哪些先前的方法被用作实证比较。为填补这一空白，我们提出了Lineage，一个面向AI科学文献中实证谱系检索的基准。实证谱系关系将一篇先前论文与一篇明确将其方法用作实证比较基线的源论文关联起来。这些关系构成谱系图，其多跳邻域为结构性评估提供了检索真值。为了大规模构建这些图谱，我们开发了Lineage-Constructor，它将异构的（摘要在此处被截断）

    arXiv:2611.25845v1 Announce Type: cross  Abstract: Current research agents are moving beyond academic search toward literature synthesis and experiment planning. These workflows increasingly rely on literature retrieval systems, which serve as standalone systems or as components of research agents. However, existing scientific literature retrieval benchmarks mainly evaluate these methods using topical relevance or citation links as supervision, which cannot capture which prior methods are used for empirical comparison. To address this gap, we introduce Lineage, a benchmark for empirical lineage retrieval in AI scientific literature. An empirical lineage relation links a prior paper to a source paper that explicitly uses its method as an empirical comparison baseline. These relations form lineage graphs whose multi-hop neighborhoods provide retrieval ground truth for structural evaluation. To construct these graphs at scale, we develop Lineage-Constructor, which converts heterogeneous p
    
[^8]: CACO：面向连续生成式检索的覆盖感知协同优化

    CACO: Coverage-Aware Coordinated Optimization for Continuous Generative Retrieval

    [https://arxiv.org/abs/2611.25738](https://arxiv.org/abs/2611.25738)

    CACO是一种轻量级自监督后训练框架，通过覆盖感知的无标签目标将多个采样查询协调至互补的物品覆盖，在几乎不增加推理延迟的情况下显著提升连续生成式检索的召回率。

    

    基于流匹配和扩散模型的连续生成式检索器在物品嵌入空间中采样多个查询，以在固定检索预算下进行并行的近似最近邻（ANN）搜索。然而，优化分布多样性或目标层面的兴趣分离忽略了集合层面的检索几何结构：采样出的查询坍缩到相互重叠的ANN邻域中，而离散的top-K查找和单目标监督又阻碍了端到端的覆盖优化。我们提出了CACO（覆盖感知协同优化），这是一个轻量级的自监督后训练框架，它将来自冻结的流匹配和扩散检索器的查询协调至互补的物品覆盖，且额外推理延迟几乎可以忽略。CACO利用跨采样参考查询的经验检索频率作为相关性代理，构建了一个无标签的覆盖目标，在所有30个骨干模型-数据集设置中都提升了召回率。

    arXiv:2611.25738v1 Announce Type: new  Abstract: Continuous generative retrievers powered by flow matching and diffusion sample multiple queries in item embedding space for parallel approximate nearest-neighbor (ANN) search under fixed retrieval budgets. However, optimizing distribution diversity or target-level interest separation ignores set-level retrieval geometry: sampled queries collapse onto overlapping ANN neighborhoods, while discrete top-K lookup and single-target supervision block end-to-end coverage optimization. We introduce CACO (Coverage-Aware Coordinated Optimization), a lightweight self-supervised post-training framework that coordinates queries from frozen flow-matching and diffusion retrievers toward complementary item coverage with negligible additional inference latency. CACO uses empirical retrieval frequencies across sampled reference queries as a relevance proxy, formulating a label-free coverage objective that improves recall across all 30 backbone-dataset sett
    
[^9]: 大语言模型能否解决领域特定自然语言理解流水线中的实体冲突？

    Can LLMs Resolve Entity Conflicts in Domain Specific NLU Pipelines?

    [https://arxiv.org/abs/2611.25733](https://arxiv.org/abs/2611.25733)

    本研究通过比较基于规则的冲突解决方法与两种基于大语言模型的方法（多轮提示和LoRA微调的Llama3.1-8B-Instruct），验证了大语言模型在解决领域特定NLU流水线中实体冲突问题上的可行性。

    

    解决实体冲突是基于流水线的自然语言理解（NLU）架构中的一个关键步骤，尤其是当多个提取器提供重叠或冲突的输出时。传统的基于规则的实体冲突解决方法往往缺乏灵活性且难以扩展。在这项工作中，我们旨在研究是否可以利用大语言模型的广泛知识来解决领域特定NLU流水线中的实体冲突。我们对传统的基于规则的冲突解决方法与两种基于大语言模型的方法进行了实验比较：一种是使用Llama3.1-8B-Instruct模型的多轮提示方法，另一种是经过LoRA微调的Llama3.1-8B-Instruct模型。评估在两种设置下进行：（1）对大语言模型匹配现有NLU流水线输出的能力进行广泛评估；（2）在更受控的场景中评估大语言模型在人工标注数据上的准确率。结果表明，虽然多轮提示具有易于适配和行为快速调整的优势……

    arXiv:2611.25733v1 Announce Type: cross  Abstract: Resolving entity conflicts is a critical step in Pipeline-based NLU architectures, especially when multiple extractors provide overlapping or conflicting outputs. Traditional rule-based entity conflict resolution methods often lack flexibility and are unscalable. In this work, we aim to study if the broad knowledge of LLMs can be leveraged to resolve entity conflicts in domain specific NLU pipelines. We present an experimental comparison between a traditional rule-based conflict resolution method, and two LLM-based approaches: a multi-shot prompting method using the Llama3.1-8B-Instruct model and a LoRA fine-tuned Llama3.1-8B-Instruct model. The evaluation was conducted under two setups: (1) a broad assessment of LLM's capability to match existing NLU pipeline output, and (2) a more controlled scenario assessing the LLM accuracy on human-labeled data. Results show that while multi-shot prompting offers ease of adaptation and rapid beha
    
[^10]: STARK：迈向真正会学习与记忆的个性化智能体

    STARK: Moving towards Personalized Agents That Truly Learn and Remember

    [https://arxiv.org/abs/2611.25705](https://arxiv.org/abs/2611.25705)

    STARK提出了一种双图记忆架构，通过情景图记录用户交互事件、洞察图提取并连接行为模式，使个性化智能体能够真正学习和记住用户的行为规律。

    

    长期记忆对于在长时间内与用户交互的个性化智能体系统至关重要，然而当前的架构只能捕获“发生了什么”，而将“为什么会发生”的推理留给底层大语言模型事后推导。这些事件不仅能够传达用户经历了什么，还能传达这些经历如何塑造了用户的行为模式。我们提出了STARK，一种双图记忆架构，它将这些事件不仅视为节点，还视为行为洞察之间的边。STARK维护一个情景图，将交互记录为在时间、因果和语义上相互关联的情景；以及一个洞察图，用于表示由大语言模型提取的行为信号，并根据支持这些信号的事件数量进行评分，同时通过引发这些洞察的事件将它们相互连接。在推理时，行为知识可以直接从由事件作为边所支撑的洞察节点中检索获得。

    arXiv:2611.25705v1 Announce Type: cross  Abstract: Long-term memory is essential for personalized agentic systems that interact with users over long time periods, yet current architectures only capture what happened while leaving the reasoning of why it happened to be derived later by underlying LLMs.%, and silo knowledge within individual users. These events can, however, convey not only what a user experienced but also how that experience shaped their behavioral patterns. We introduce STARK, a dual-graph memory architecture which treats these events not just as nodes but also as edges between behavioral insights. STARK maintains an Episodic Graph which records interactions as temporally, causally, and semantically linked episodes; and an Insight Graph which represents LLM-extracted behavioral signals scored by how many events support them, connected by the very events that led to them. At inference, behavioral knowledge is retrieved directly from insight nodes grounded by event-as-ed
    
[^11]: 探索数据湖中的语义表格搜索

    Exploring Semantic Table Search in Data Lakes

    [https://arxiv.org/abs/2611.25691](https://arxiv.org/abs/2611.25691)

    该论文提出了Thetis++系统，将关键词搜索与基于命名实体的语义搜索相结合，通过融合多种排名组合技术，更有效地在数据湖中发现相关表格数据集。

    

    数据湖中相关数据集的发现仍然是一个关键挑战。联合搜索和连接搜索是在增强现有表格时发现数据的有效技术。而在探索性搜索中，关键词搜索通常被用来找到与某些搜索词相匹配的相对较小的高相关性表格集合。在这一背景下，语义搜索是一种新技术，它利用了作为查询的示例元组以及目标表格中命名实体的存在。因此，关键词搜索和实体搜索能够找到多样化的表格集合。在这项工作中，我们提出了 Thetis++，它将关键词搜索和语义表格搜索相结合，从而发现同时融合语义信号与精确关键词匹配的相关表格。整合多种范式和信息来源使我们能够结合不同方法的优势。因此，我们研究了多种用于组合排名的替代技术并进行评估。

    arXiv:2611.25691v1 Announce Type: new  Abstract: Discovery of relevant datasets within data lakes remains a key challenge. Union and join search are effective techniques for discovering data when augmenting an existing table. In exploratory search, instead, keyword search is commonly adopted to find a relatively small set of highly relevant tables matching some search terms. In this context, semantic search is a new technique that exploits the presence of named entities in exemplar tuples used as queries and in target tables. Hence, keyword search and entity search are able to find a diverse set of tables. In this work, we propose \texttt{Thetis++}, which combines keyword and semantic table search to discover relevant tables that combine the semantic signal with the presence of exact keyword matches. Incorporating multiple paradigms and sources of information allow us to combine the best of multiple worlds. Therefore, we study many alternative techniques for combining rankings and eval
    
[^12]: DNA：解锁CTR模型中的深度扩展

    DNA: Unlocking Depth Scaling in CTR Models

    [https://arxiv.org/abs/2611.25659](https://arxiv.org/abs/2611.25659)

    提出DNA（双流枢纽注意力）方法，通过让用户行为序列随模型深度逐层演化更新，而非每层重复投影同一固定序列，突破了CTR模型中交叉注意力堆叠的深度扩展瓶颈。

    

    长用户行为序列是点击率（CTR）预测中最强的信号之一。交叉注意力使这些序列在工业排序系统中变得实用，因为其计算量随序列长度呈线性增长。然而，堆叠交叉注意力层带来的收益逐渐递减：在淘宝搜索场景中，将交叉注意力堆叠层从16层增加到28层，仅带来0.02个百分点（pt）的AUC提升。我们认为键值侧是瓶颈所在。每一层都将相同的行为序列投影为键和值，因此增加深度只是让模型对固定序列进行更多次读取。我们提出DNA（Dual-stream Nexus Attention，双流枢纽注意力），使序列本身随深度演化。在每一层中，请求表示（搜索查询、用户特征和场景）首先读取序列；随后每个行为token关注更新后的请求表示，并利用其读取到的内容更新自身，从而使下一层读取到一个由……塑造的序列（摘要在此处截断）。

    arXiv:2611.25659v1 Announce Type: new  Abstract: Long user behavior sequences are among the strongest signals for click-through rate (CTR) prediction. Cross-attention makes these sequences practical for industrial ranking by keeping compute linear in sequence length. Yet the gains from stacking cross-attention layers taper off: on Taobao search, growing a cross-attention stack from 16 to 28 layers adds only 0.02 percentage points (pt) of AUC. We argue that the key-value side is the bottleneck. Each layer projects the same behavior sequence into keys and values, so adding depth gives the model more reads of a fixed sequence. We propose DNA (Dual-stream Nexus Attention), which lets the sequence itself evolve with depth. At each layer, the request representation (the search query, user features and scene) first reads the sequence. Each behavior token then attends to the updated request representation and uses what it reads to update itself, so the next layer reads a sequence shaped by the
    
[^13]: 非受控的来源刊名：当Nature、Science和PNAS遇上Scopus

    Uncontrolled source titles: When Nature, Science, and PNAS meet Scopus

    [https://arxiv.org/abs/2611.25627](https://arxiv.org/abs/2611.25627)

    Scopus数据库中来源刊名缺乏规范控制，导致Nature、Science和PNAS三大期刊的检索结果碎片化且不一致，而Web of Science中不存在此类问题，研究并揭示了刊名变异源于上游数据商继承、自动解析和OCR错误及验证不足等机制。

    

    诸如Scopus和Web of Science等文献数据库在文献发现、系统性综述、文献计量分析和科研评估中发挥着关键作用，然而其可靠性取决于对文献数据质量的有效控制。本研究通过在三种旗舰期刊——Nature、Science和PNAS——上比较五种可能被采用的期刊级检索策略，考察了Scopus中来源刊名缺乏规范控制的问题。检索结果之间的巨大差异表明来源刊名字段存在持续的碎片化现象。利用Web of Science核心合集进行的对比分析显示，相同期刊并未观察到类似程度的来源刊名非受控变异。研究进一步探讨了可能的成因机制，包括从（不同的）上游数据提供商继承下来的不同刊名形式、自动解析及OCR相关的错误痕迹，以及交叉验证不足。

    arXiv:2611.25627v1 Announce Type: cross  Abstract: Bibliographic databases such as Scopus and Web of Science play a critical role in literature discovery, systematic reviews, bibliometric analyses, and research evaluation, yet their reliability depends on effective control of the quality of the bibliographic data. This study examines the problem of uncontrolled source titles in Scopus by comparing five possibly adopted journal-level retrieval strategies across three flagship journals-Nature, Science, and PNAS. Substantial discrepancies across retrieval results indicate persistent fragmentation in the source title field. A comparative analysis using the Web of Science Core Collection shows that no comparable degree of uncontrolled source-title variation is observed for the same journals. The study further explores possible mechanisms, including inheritance of different title forms from (different) upstream providers, automated parsing and OCR-related artifacts, and insufficient cross-va
    
[^14]: 生成式新闻推荐再现流行度：基于记忆化的分析

    Generative News Recommendation Reproduces Popularity: A Memorization-Based Analysis

    [https://arxiv.org/abs/2611.25606](https://arxiv.org/abs/2611.25606)

    该研究利用约2000万条日本新闻浏览会话数据，首次从训练数据记忆化的角度揭示生成式新闻推荐再现流行内容的机制——被频繁点击的新闻标题更容易被模型记住，并在更多样化的用户上下文中被生成，从而导致信息多样性下降。

    

    基于文本的生成式推荐器能够通过直接生成物品标题或其他文本表示，超越对现有物品进行排序。在新闻领域，这引发了一个社会层面的担忧：模型可能反复推送本已热门的文章，从而缩小用户接触信息的多样性。我们将训练数据记忆化作为导致这种集中现象的机制之一加以研究。利用来自日本某新闻服务的约2000万条浏览会话数据，我们对一个经过指令微调的Llama 3 8B模型进行持续训练，使其生成会话中的下一篇新闻标题。研究发现，生成标题的分布比浏览日志中实际观察到的标题分布更加集中、多样性更低。频繁重复出现的目标标题表现出更强的逐字记忆与近似记忆，而记忆化程度越高的标题，会在越广泛的提示（上下文）中被生成。这些发现支持一条……

    arXiv:2611.25606v1 Announce Type: cross  Abstract: Text-based generative recommenders can go beyond ranking existing items by directly generating item titles or other textual representations. In the news domain, this raises a societal concern that models may repeatedly surface already popular articles and thereby narrow the diversity of information to which users are exposed. We investigate training-data memorization as one mechanism behind this concentration. Using approximately 20 million browsing sessions from a Japanese news service, we continually trained an instruction-tuned Llama 3 8B model to generate the next article title in a session. The generated-title distribution was more concentrated and less diverse than the distribution of titles observed in the browsing logs. Frequently repeated target titles exhibited greater verbatim and approximate memorization, and more strongly memorized titles were generated in response to a wider range of prompts. These findings support a path
    
[^15]: AutoRAGBench：用于自动化RAG系统开发的LLM智能体基准测试

    AutoRAGBench: A Benchmark for LLM Agents Automating RAG System Development

    [https://arxiv.org/abs/2611.25582](https://arxiv.org/abs/2611.25582)

    本文提出了首个基准测试AutoRAGBench，通过九个任务评估LLM智能体在无起始代码和预设策略的条件下，于开放代码空间中自主从零构建并迭代优化RAG系统的能力。

    

    检索增强生成（RAG）是一种成熟的范式，通过从外部语料库检索的信息来增强大语言模型（LLM）。然而，开发有效的RAG系统需要大量的工程工作和迭代实验，因为有效的设计选择因任务和领域而异。随着LLM智能体能力的快速进步，我们提出这样一个问题：这些智能体能够多好地自主构建RAG系统，以及它们能够通过迭代多有效地改进这些系统。在本文中，我们提出了AutoRAGBench，这是第一个评估LLM智能体在开放代码空间中从零开始开发RAG系统能力的基准测试。它包含九个RAG系统开发任务。对于每个任务，智能体在没有起始代码或预设策略的情况下，在单个H800 GPU上六小时的预算内从零构建RAG系统。它们分析任务数据、实现RAG流水线，并基于实验结果迭代地优化其设计……

    arXiv:2611.25582v1 Announce Type: cross  Abstract: Retrieval-augmented generation (RAG) is a well-established paradigm for augmenting large language models (LLMs) with information retrieved from external corpora. However, developing effective RAG systems requires substantial engineering effort and iterative experimentation, as effective design choices vary across tasks and domains. With rapid advances in LLM agent capabilities, we ask how well these agents can autonomously build RAG systems and how effectively they can improve them through iteration. In this paper, we propose AutoRAGBench, the first benchmark to evaluate how well LLM agents develop RAG systems from scratch in an open code space. It comprises nine RAG system development tasks. For each task, agents build RAG systems from scratch without starter code or prescribed strategies, under a six-hour budget on a single H800 GPU. They analyze task data, implement RAG pipelines, and iteratively refine their designs based on experi
    
[^16]: 面向整页优化的生成式排序策略学习

    Learning Generative Ranking Policies for Whole-Page Optimization

    [https://arxiv.org/abs/2611.25527](https://arxiv.org/abs/2611.25527)

    提出一种局部页面级训练目标，可精确拆分为逐位置的训练信号，使生成式排序策略在整页优化中动态生成更高分的页面，同时克服了现有方法中统一页面级信号和固定模仿目标的双重局限。

    

    推荐页面是作为一个整体被消费的：页面上的物品既相互争夺注意力又彼此互补，因此最优页面并非简单地由各自最优的物品并排放置而成。我们研究整页优化问题，即联合地选择和排列页面上的物品以最大化页面整体的得分。可能的页面数量呈组合式增长，因此好的页面必须通过生成而非枚举的方式获得。一些生成式重排方法对采样页面中的每个选择都使用相同的页面级训练信号；另一些方法则模仿在训练前构建的目标页面，而这些目标页面在生成器不断改进的过程中基本保持固定不变。我们用一个局部页面级目标同时解决了上述两个局限，该目标反复地将当前生成器向得分更高的页面推进。这一目标可以精确地拆分为每个位置一个训练目标，每个目标通过完整页面的得分来比较同一部分页面下的候选物品……（摘要在此处被截断）

    arXiv:2611.25527v1 Announce Type: cross  Abstract: A recommendation page is consumed as a whole: the items on it compete for attention and complement one another, so the best page is not simply the individually best items placed side by side. We study whole-page optimization, which selects and orders the items on a page jointly to maximize the score of the page. The number of possible pages grows combinatorially, so good pages must be generated rather than enumerated. Some generative re-ranking methods use the same page-level training signal for every choice in a sampled page. Others imitate target pages built before training, which remain largely fixed as the generator improves. We address both limitations with one local page-level objective, which repeatedly moves the current generator toward pages with higher score. This objective splits exactly into one training target per position, and each target compares candidates at the same partial page by the scores of the complete pages the
    
[^17]: VideoTrail：通过层次化可视化实现交互式视频信息觅食

    VideoTrail: Interactive Video Foraging Through Hierarchical Visualization

    [https://arxiv.org/abs/2611.25377](https://arxiv.org/abs/2611.25377)

    VideoTrail是一个基于信息觅食理论的AI浏览器扩展，通过多模态自适应的层次化可视化将视频中的语义与结构线索呈现为“信息气味”，帮助用户高效定位关键时刻并支持目标导向的视频信息搜寻。

    

    人们常常通过观看视频来查找信息，但定位相关片段可能既困难又耗时。现有的策略（如拖动进度条快进浏览）主要依赖视觉扫描和线性搜索，对目标导向式导航的支持有限。我们提出了VideoTrail，这是一个AI驱动的浏览器扩展，通过层次化可视化支持视频中的信息搜寻。受信息觅食理论的启发，VideoTrail将语义和结构线索呈现为“信息气味”，构建出一个可探索的视频片段路线图。该系统是多模态且自适应的，会根据视频内容优先呈现视觉或文本线索。在一项用户研究（N=16）中，VideoTrail帮助参与者高效地定位关键时刻，并且相比传统时间线实现了新的导航策略，凸显了揭示自适应视频结构如何能够增强目标导向式的观看体验。

    arXiv:2611.25377v1 Announce Type: cross  Abstract: People often watch videos to find information, but locating relevant moments can be difficult and time-consuming. Existing strategies like scrubbing rely primarily on visual scanning and linear search, providing limited support for goal-directed navigation. We present VideoTrail, an AI-powered browser extension that supports information seeking in videos through a hierarchical visualization. Inspired by information foraging theory, VideoTrail surfaces semantic and structural cues as information scent, creating an explorable roadmap of segments. The system is multimodal and adaptive, prioritizing visual or textual scent based on the video's content. In a user study (N=16), VideoTrail helped participants efficiently locate key moments and enabled new navigation strategies compared to traditional timelines, highlighting how exposing an adaptive video structure can enhance goal-directed viewing.
    
[^18]: LQ-Rec：面向序列推荐的后期量化

    LQ-Rec: Late Quantization for Sequential Recommendation

    [https://arxiv.org/abs/2611.25256](https://arxiv.org/abs/2611.25256)

    该论文提出“后期量化”这一非自回归范式，将序列推荐中生成式检索的离散化限制在物品输出端，从而突破自回归语义ID的结构性扩展限制，并通过加性误差分解与成对排序定理为量化引起的排序损失提供了以往方法所不具备的理论保证。

    

    面向序列推荐的生成式检索（GR）将每个物品压缩为一个简短的语义ID，并训练自回归解码器根据用户历史生成下一物品的编码。尽管自回归GR在冷启动处理和存储方面颇具吸引力，但它存在结构性的扩展限制：通常，更长的语义ID虽然在原则上可以减少量化误差，却会降低排序质量，且内存占用随语义ID长度呈二次增长。我们提出了后期量化，这是一种非自回归范式，它将离散化限制在流程的输出（物品）端。编码器和评分规则在连续空间中运行，而物品则以语义ID的形式进行存储和匹配，这些语义ID由端到端训练的残差量化器生成。这种架构转变使得加性误差分解与成对排序定理成为可能，二者共同界定了由量化引起的排序下降，这是以往语义ID方法所缺乏的结构性保证。

    arXiv:2611.25256v1 Announce Type: new  Abstract: Generative retrieval (GR) for sequential recommendation compresses each item into a short semantic ID and trains an autoregressive decoder to generate next-item codes from a user history. While appealing for cold-start handling and storage, au- toregressive GR suffers a structural scaling limit: typically, longer semantic IDs reduce quantization error in principle but degrade ranking quality and quadratic memory usage against semantic ID length. We propose Late Quantization (LQ), a non-autoregressive paradigm that confines discretization to the output (item) side of the pipeline. The encoder and scoring rule operate in continuous space, while items are stored and matched as semantic IDs produced by an end-to-end-trained residual quantizer. This architectural shift permits an additive error decomposition and a pairwise ordering theorem that jointly bound the quantization- induced ranking drop, a structural guarantee absent in prior semant
    
[^19]: 在布林格的宗教改革书信中检索教父文献引用并建模主题

    Retrieving Patristic References and Modelling Topics in Bullinger's Reformation Correspondence

    [https://arxiv.org/abs/2611.25247](https://arxiv.org/abs/2611.25247)

    本研究首次将针对拉丁语圣经互文性微调的检索模型迁移至布林格宗教改革书信中的教父引用检测任务，通过比较词汇、稠密与混合检索方法，发现在圣经复用上微调的拉丁语编码器在双编码器中表现最佳，而 BM25 在显式引用检测上更具优势。

    

    海因里希·布林格（1504-1575）是瑞士宗教改革家、苏黎世主教，他的书信记录了宗教改革时期的政治与宗教辩论，其中包括大量援引教父作为权威来源的内容。如何通过计算方法检测此类引用在很大程度上尚未被探索。我们研究了用于教父引用检测的稠密检索与稀疏检索方法，以及以元数据信息驱动的主题建模作为所讨论话题和思想的代理。此前关于拉丁语互文性的研究已针对各自目标领域对稠密检索器进行了微调，我们测试了这些模型向布林格书信的迁移效果。我们在显式引用和隐式引用上比较了词汇方法（BM25）、稠密方法（双编码器）和混合方法。在圣经文本复用上微调的拉丁语编码器在双编码器中表现最佳（显式引用的 Recall@20 为 82%，隐式引用为 16%），而 BM25 在显式引用上优于稠密方法。

    arXiv:2611.25247v1 Announce Type: cross  Abstract: The correspondence of Heinrich Bullinger (1504-1575), Swiss Reformer and Antistes of Zurich, documents the political and religious debates of the Reformation, including extensive appeals to the Church Fathers as authoritative sources. Detecting such references computationally remains largely unexplored. We investigate dense and sparse retrieval for patristic reference detection, and metadata-informed topic modelling as a proxy for the topics and ideas discussed. Prior work on Latin intertextuality has fine-tuned dense retrievers for their respective target domain, and we test how well these models transfer to Bullinger's correspondence. We compare lexical (BM25), dense (bi-encoders), and hybrid methods on explicit and implicit references. A Latin encoder fine-tuned on biblical reuse performs best among the bi-encoders (82%, Recall@20 on explicit and 16% on implicit references), while BM25 outperforms dense approaches on explicit citati
    
[^20]: LSP-RAG：面向知识图谱检索增强生成的潜在评分视角

    LSP-RAG: Latent Scoring Perspectives for Knowledge Graph Retrieval-Augmented Generation

    [https://arxiv.org/abs/2611.25229](https://arxiv.org/abs/2611.25229)

    提出LSP-RAG框架，通过多视角提示评分增强知识图谱检索增强生成中的逐跳知识检索，在不依赖大规模语言模型的情况下降低计算成本并提升多跳场景下的检索质量。

    

    大语言模型（LLMs）在开放域问答中展现了强大的能力，但由于训练语料的不完整性，它们在事实准确性和多跳推理方面常常表现不佳。一个有前景的解决方案是知识图谱检索增强生成（KG-RAG），它通过从外部知识图谱（KGs）中检索结构化知识来补充大语言模型。然而，现有的KG-RAG方法要么依赖大规模语言模型（如ChatGPT）来引导检索过程，导致高昂的计算成本；要么在使用轻量级语言模型时检索质量受限，尤其是在多跳场景下。我们提出了LSP-RAG（面向知识图谱检索增强生成的潜在评分视角），这是一种新颖的KG-RAG框架，通过多视角提示评分来增强逐跳的知识图谱知识检索。每个潜在评分视角由一个……（原文摘要到此截断）

    arXiv:2611.25229v1 Announce Type: new  Abstract: Large Language Models (LLMs) have demonstrated strong capabilities in open-domain question answering, but often struggle with factual accuracy and multi-hop reasoning due to the incompleteness of the training corpora. A promising solution is Knowledge Graph Retrieval-Augmented Generation (KG-RAG), which supplements LLMs with structured knowledge retrieved from external knowledge graphs (KGs). However, existing KG-RAG methods either rely on large-scale language models (e.g., ChatGPT) to guide the retrieval process, which leads to high computational costs, or suffer from limited retrieval quality when using lightweight language models, particularly under multi-hop scenarios. We propose LSP-RAG (Latent Scoring Perspectives for Retrieval-Augmented Generation over Knowledge Graphs), a novel KG-RAG framework that enhances hop-wise KG knowledge retrieval through multi-perspective prompted scoring. Each latent scoring perspective is guided by a 
    
[^21]: FinVeriQA：面向问答的金融研究智能体：路由、检索、MCP工具使用与溯源感知的证据验证

    FinVeriQA: Financial Research Agents for Question Answering: Routing, Retrieval, MCP Tool Use, and Provenance-Aware Evidence Verification

    [https://arxiv.org/abs/2611.25071](https://arxiv.org/abs/2611.25071)

    该论文提出FinVeriQA——一个基于SEC文件与实时市场数据的开源多智能体金融问答系统，其核心创新在于用紧凑的专用决策模型Laya进行问题路由与建议过滤，配合Python精确计算、MCP工具检索与溯源感知的证据验证，实验证明小型决策模型的路由性能可显著超越大型语言模型，并在决策准确性与推理速度之间取得最佳权衡。

    

    大语言模型（LLM）能够写出流畅的金融分析，但会算错比率、引用过时的价格数据，并且高置信度地将问题路由到错误的工具。我们探讨在金融问答助手系统中，哪个组件应该承担哪项任务，并通过FinVeriQA——一个基于SEC文件与实时市场数据的开源多智能体研究原型——来研究这一问题：一个紧凑的类型化决策模型（Laya）负责路由每个问题并筛选出投资建议类请求，Python负责计算每一个数字，检索与工具负责获取证据，Mistral-7B撰写带引用的回答，一个代码验证器检查引用的ID是否存在以及数字是否与计算出的笔记相匹配，并通过模型上下文协议（MCP）服务器将整条流水线对外暴露。在实时实验中，Laya的路由表现优于Mistral-7B（0.85 vs. 0.52；总体0.86 vs. 0.68，McNemar检验 p = 0.0003）；一个12B参数的Mistral决策能力并未更优，而DeepSeek-R1-8B决策表现更好（0.79 vs. 0.68，p = 0.0192），但速度慢约9.9倍。

    arXiv:2611.25071v1 Announce Type: cross  Abstract: Large language models (LLMs) write fluent financial analyses but miscompute ratios, recall stale prices, and route questions to the wrong tool with high confidence. We ask which component should do each job in a financial question-answering assistant, and study it with FinVeriQA, an open-source multi-agent research prototype over SEC filings and live market data: a compact typed-decision model (Laya) routes each question and screens out advice requests, Python computes every number, retrieval and tools fetch evidence, Mistral-7B writes a cited answer, a code validator checks that cited IDs exist and that numbers match computed notes, and a Model Context Protocol (MCP) server exposes the pipeline. In live experiments, Laya routed better than Mistral-7B (0.85 vs. 0.52; overall 0.86 vs. 0.68, McNemar p = 0.0003); a 12B Mistral decided no better, and DeepSeek-R1-8B decided better (0.79 vs. 0.68, p = 0.0192) but about 9.9 times more slowly.
    
[^22]: 面向样本高效在线A/B测试的方差最优反事实策略设计

    Variance-Optimal Counterfactual Policy Design for Sample-Efficient Online A/B Testing

    [https://arxiv.org/abs/2611.24757](https://arxiv.org/abs/2611.24757)

    该论文提出一种方差最优的反事实策略设计方法，通过联合优化动作的采样方式，将流量引导至两种策略不一致的动作上，从而显著降低在线A/B测试的方差并提升样本效率。

    

    在线A/B测试是在部署前评估面向用户系统变更（例如新的推荐或排序策略）的标准方法。虽然标准A/B测试的均值差估计量是无偏的，但它通常存在高方差问题。当处理策略与生产策略以相近的概率选择相同的动作时，由此产生的观测数据会增加噪声，却几乎不提供关于两种策略差异的信息。因此，A/B测试往往需要大量流量和较长周期才能检测到较小的处理效应。现有的方差缩减方法主要是在数据收集方案固定之后改进估计量，而近年来的重叠感知设计仅优化两种策略之间的单一流量分配，因而无法将流量引导到两种策略不一致的动作上。在本文中，我们联合优化动作的采样方式……

    arXiv:2611.24757v1 Announce Type: new  Abstract: Online A/B testing is the standard approach for evaluating changes to user-facing systems, such as a new recommendation or ranking policy, before deployment. While the difference-in-means estimator of a standard A/B test is unbiased, it often suffers from high variance. When the treatment and production policies choose the same actions with similar probabilities, the resulting observations can add noise while providing little information about the difference between the policies. Consequently, A/B tests can require large amounts of traffic and long durations to detect small treatment effects. Existing variance-reduction methods mainly improve the estimator after the data-collection scheme has been fixed, and recent overlap-aware designs only optimize a single traffic split between the two policies; thereby, they cannot direct traffic to the actions on which the policies disagree.   In this paper, we jointly optimize how actions are sampl
    
[^23]: IntentFlow：小红书中基于意图驱动的图文查询实时多模态搜索系统

    IntentFlow: An Intent-Driven Real-Time Multimodal Search System with Image-Text Queries at Xiaohongshu

    [https://arxiv.org/abs/2611.20448](https://arxiv.org/abs/2611.20448)

    IntentFlow是小红书部署的意图驱动实时多模态搜索系统，通过层次化意图分类体系和意图特定的相关性标准，在百亿级索引上实现了图文查询场景下候选检索与重排序的一致性对齐。

    

    基于图文结合查询的多模态搜索帮助用户在用户生成内容（UGC）社区中查找笔记，同时也是多模态搜索代理的关键搜索工具。对于此类查询，查询与笔记之间的相关性取决于用户的检索意图，无法仅通过视觉或主题相似性来判定。简单地对视觉搜索结果进行重排序无法完全解决这一挑战，因为符合用户意图的笔记可能在初始候选集中就已经缺失。这促使我们进行显式的意图建模，以便在一致的相关性标准下对齐候选检索与重排序两个环节。我们提出了IntentFlow，一个部署在小红书的意图驱动实时多模态搜索系统，其索引规模达到百亿级。IntentFlow通过层次化的意图分类体系来表示检索目标，并使用针对特定意图的相关性标准来指导数据构建、表示学习和重排序。一个离线代理（agent）……（原文摘要在此处被截断）

    arXiv:2611.20448v1 Announce Type: new  Abstract: Multimodal search with image-text queries helps users find notes in user-generated content (UGC) communities and serves as a key search tool for multimodal search agents. For such queries, query-note relevance depends on the user's retrieval intent and cannot be determined by visual or topical similarity alone. Simply reranking visual search results cannot fully address this challenge, as notes relevant to the user's intent may be missing from the initial candidate set. This motivates explicit intent modeling to align candidate retrieval and reranking under consistent relevance criteria. We present IntentFlow, an intent-driven real-time multimodal search system deployed at Xiaohongshu over an index on the order of ten billion entries. IntentFlow represents retrieval objectives with a hierarchical intent taxonomy and uses intent-specific relevance criteria to guide data construction, representation learning, and reranking. An offline agen
    
[^24]: 保持简单！一项多项选择题任务可以有益于生成式推荐

    Keep it Simple! A Multiple-Choice Task can Benefit Generative Recommendation

    [https://arxiv.org/abs/2611.19320](https://arxiv.org/abs/2611.19320)

    该论文提出只需在生成式推荐的学习目标中加入一个简单的多项选择题辅助任务，即可显式引入负样本和与BPR理念一致的相对排序信号，从而在不依赖复杂架构和大量调参的情况下有效提升推荐效果。

    

    序列推荐领域近期对生成式检索的兴趣日益增长。其中最流行的方法之一是自回归地生成物品的语义标识符（SIDs）作为推荐结果。现有的研究工作主要依赖于语义对齐、复杂的架构设计以及大量的超参数调优。然而，生成模型的传统学习框架可能存在基础性的局限。具体而言，大多数生成模型直接预测目标物品，而没有显式地学习负样本组合和相对排序信号。这与贝叶斯个性化排序（BPR）的理念并不契合，后者强调正负样本之间相对排序的重要性。在本文中，我们证明只需在学习目标中加入一个简单的多项选择题辅助任务，就能有效解决这一问题。通过极小的改动，我们引入了

    arXiv:2611.19320v1 Announce Type: new  Abstract: Sequential recommendation has recently witnessed growing interest in generative retrieval. One of the most popular approaches is to autoregressively generate the semantic identifiers (SIDs) of items as recommendations. Existing efforts mainly rely on semantic alignment, sophisticated architectural designs, and extensive hyperparameter tuning. However, the conventional learning framework for generative models could have foundational limitations. Specifically, most generative models directly predict target items without explicitly learning negative compositions and relative ranking signals. This is not well aligned with the philosophy of Bayesian personalized ranking (BPR), which emphasizes the importance of relative rankings between positive and negative samples.   In this paper, we show that adding a simple multiple-choice auxiliary task into the learning objective can effectively address this issue. With minimal adaptation, we introduce
    
[^25]: 何时值得额外检索？RAG中证据增益与答案可靠性的成本感知研究

    When Is Additional Retrieval Worthwhile? A Cost-Aware Study of Evidence Gain and Answer Reliability in RAG

    [https://arxiv.org/abs/2610.37358](https://arxiv.org/abs/2610.37358)

    该论文提出一种成本感知的检索路由策略，通过在路由前预估混合检索相对于稠密检索基线的问题级增量证据收益，仅在约四分之一的问题上触发更昂贵的混合检索，在保持相同证据Hit@10的同时显著降低检索成本。

    

    检索增强生成（RAG）可以从更复杂的检索中受益，但对许多问题而言，额外的检索可能是不必要的，而且证据检索上的收益也不一定能转化为忠实或正确的答案。我们研究了一种成本感知的双动作检索策略，在“稠密检索加重排序”基线与“词汇-稠密混合检索器”之间进行选择。与简单地将基线检索归类为成功或失败不同，我们利用路由决策之前即可获得的特征，估计混合检索针对特定问题的增量证据检索收益。正则化线性模型与决策阈值通过按文档分组的交叉验证进行选择，并在留出集评估之前冻结。在60篇此前未见的QASPER训练集划分文章上（188个问题，其中169个可进行证据评估），该策略在46个问题（24.5%）上触发了混合检索，并获得了与……相同的证据Hit@10（摘要在此处截断）。

    arXiv:2610.37358v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) can benefit from more sophisticated retrieval, but extra retrieval may be unnecessary for many questions and gains in evidence retrieval may not translate into faithful or correct answers. We study a cost-aware, two-action retrieval policy that chooses between a dense-retrieval-plus-reranking baseline and a hybrid lexical--dense retriever. Rather than classifying baseline retrieval as simply successful or unsuccessful, we estimate the question-specific incremental evidence-retrieval benefit of hybrid retrieval using features available before the routing decision. A regularized linear model and decision threshold are selected using document-grouped cross-validation and frozen prior to held-out evaluation. On 60 previously unseen QASPER training-split articles (188 questions, 169 evidence-evaluable), the policy triggers hybrid retrieval on 46 questions (24.5\%) and obtains the same evidence Hit@10 as al
    
[^26]: 缺失的补充：面向编码代理的状态条件化最小充分证据

    The Missing Complement: State-Conditioned Minimal Sufficient Evidence for Coding Agents

    [https://arxiv.org/abs/2609.20050](https://arxiv.org/abs/2609.20050)

    该论文提出了状态条件化最小充分证据恢复这一新问题并构建了SERBench基准，同时提出MSS-Complement方法，将证据获取从排序转变为集合构建，为编码代理的决策恢复紧凑且充分的证据组合。

    

    一个处理问题进行到一半的编码代理已经读过检索器排名最高的很多内容。相关性是按段落评分的，但充分性属于集合层面：一个排序器可能用某个所需事实的多个变体填满其预算，却使决策仍然缺乏支持。我们提出了状态条件化最小充分证据恢复问题：给定一个捕获的代理状态，恢复一个紧凑的证据组合，以提供其下一个决策仍然缺乏的支持。SERBench在来自45个代码仓库的500个保留状态上对此进行测量，记录代理已经看到的内容，并且只对覆盖当前决策被标注所需的每一个事实的集合给予认可。MSS-Complement将证据获取视为集合构建而非排序。三次语义调用提出一个联合充分的集合，搜索其缺失的内容，并在6,144个token内返回4-8个完整的源单元。一个仅在校准数据上固定的配置，为其中73.0%的状态恢复了完整的证据集合。

    arXiv:2609.20050v1 Announce Type: cross  Abstract: A coding agent halfway through an issue has already read much of what a retriever ranks highest. Relevance is scored per passage, but sufficiency belongs to the set: a ranker can fill its budget with variants of one required fact and leave the decision unsupported. We formulate state-conditioned minimal sufficient evidence recovery: given a captured agent state, recover a compact evidence combination that supplies the support its next decision still lacks. SERBench measures this on 500 held-out states from 45 repositories, recording what the agent has seen and crediting only sets that cover every fact the current decision was annotated to require. MSS-Complement treats acquisition as set construction, not ranking. Three semantic calls propose a jointly sufficient set, search for what it lacks, and return 4-8 intact source units within 6,144 tokens. One configuration, fixed on calibration data, recovers a complete set for 73.0% of those
    
[^27]: 停止去除停用词：一项沿袭的预处理默认设置如何扭曲法律文本即数据研究

    Stop Removing Stopwords: How an Inherited Preprocessing Default Distorts Legal Text-as-Data

    [https://arxiv.org/abs/2609.19153](https://arxiv.org/abs/2609.19153)

    本研究通过穷尽式单词消融实验，首次直接针对下游分类目标验证停用词去除这一沿袭自信息检索时代、从未被验证过的预处理默认设置，揭示其可能扭曲实证法学中基于TF-IDF和线性分类器的文本数据分析结果。

    

    实证法学研究日益将司法文本视为数据，其中许多研究仍依赖稀疏、可解释的处理流程——TF-IDF特征和线性分类器——因为文本特征往往是研究对象本身，而不仅仅是实现预测的手段。然而，这些流程继承了一系列源自二十世纪中叶信息检索领域的预处理默认设置，这些设置从未针对分类准确度进行过验证，其中最根深蒂固的是停用词去除。本研究引入了一种穷尽式的单词消融方法，直接根据下游目标衡量预处理步骤的效果，并将其应用于停用词去除这一最难撼动的案例。通过将最高法院数据库的标签与Caselaw Access Project的判决意见文本相匹配，该研究考察了两个涵盖F1提升空间的双分类任务：意识形态方向（不去除停用词的基线F1约0.68）和宪法与非宪法法律类型（约0.92），涉及7,66

    arXiv:2609.19153v1 Announce Type: new  Abstract: Empirical legal scholarship increasingly treats judicial text as data, and much of it still runs on sparse, interpretable pipelines -- TF-IDF features and linear classifiers -- because the textual feature is often the object of study, not merely a means to a prediction. Yet these pipelines inherit a chain of preprocessing defaults from mid-century information retrieval that were never validated against classification accuracy, the most entrenched being stopword removal. This study introduces an exhaustive single-word ablation that measures a preprocessing step's effect directly against the downstream objective, and applies it to stopword removal as the hardest case to dislodge. Matching Supreme Court Database labels to Caselaw Access Project opinion texts, it examines two binary tasks that bracket F1 headroom, ideological direction (no-removal baseline F1 ~ 0.68) and constitutional versus non-constitutional law type (~ 0.92), across 7,66
    
[^28]: 从转录到语义语料库分析：面向古代语言的无监督句子表示学习

    From transcription to semantic corpus analysis: unsupervised learning of sentence representations for ancient languages

    [https://arxiv.org/abs/2607.24542](https://arxiv.org/abs/2607.24542)

    本文提出两种完全无监督的方法（TSDAE 和无监督 SimCSE），仅用原始句子将面向古代语言的专用词元级语言模型转化为语料库专属的句子编码器，并在拉丁语和古希腊语教父文献中经专家验证的圣经文本复用识别任务上验证了其有效性。

    

    自动文本识别（ATR）如今为数字人文领域提供了大量非结构化、异构且常含噪声的古代语言文本。下游的语义分析——如文本复用识别、对齐和语义检索——依赖于句子嵌入，然而现有方法向古代语言的迁移效果不佳：通用多语言编码器表现欠佳，专用语言模型产生各向异性的表示空间，且标注的相似度数据十分稀缺。我们研究了两种完全无监督的策略——TSDAE 和无监督 SimCSE（以下简称 CSE）——它们仅利用原始句子，将专用的词元级语言模型适配为面向特定语料库的句子编码器。我们在拉丁语和古希腊语教父文献中的圣经文本复用任务上对这两种方法进行了验证：语料包含奥古斯丁、哲罗姆和阿塔纳修斯著作中 2,936 个经专家验证含有圣经复用的句子，并与《武加大译本》、《七十士译本》以及 Nestle-Aland 版本文本进行匹配。

    arXiv:2607.24542v2 Announce Type: replace  Abstract: Automatic Text Recognition (ATR) now supplies digital humanities with large volumes of unstruc- tured, heterogeneous, and often noisy text in ancient languages. Downstream semantic analyses- text reuse identification, alignment, and semantic search-rely on sentence embeddings, yet ex- isting methods transfer poorly to ancient languages: generic multilingual encoders underperform, specialized language models yield anisotropic representation spaces, and labeled similarity data is scarce. We study two fully unsupervised strategies-TSDAE and unsupervised SimCSE (hence- forth CSE)-that adapt a specialized token-level language model into a corpus-specific sentence encoder using only raw sentences. We validate them on biblical reuse in patristic literature in Latin and Ancient Greek: 2,936 sentences carrying expert-verified reuses in works of Augustine, Jerome, and Athanasius, matched against the Vulgate, the Septuagint, and the Nestle--Ala
    
[^29]: 超越维度：单向量与多向量检索中的领域偏移与灾难性遗忘

    Beyond Dimensionality: Domain Shift and Catastrophic Forgetting in Single- vs. Multi-Vector Retrieval

    [https://arxiv.org/abs/2603.29519](https://arxiv.org/abs/2603.29519)

    该研究反驳了嵌入维度是单向量检索模型失效主因的观点，实证表明领域偏移以及嵌入相似性与任务相关性之间的错位才是导致检索质量下降的主导因素。

    

    Weller 等人最近的工作引入了自然化基准 LIMIT，并表明广泛使用的单向量嵌入模型在检索质量上出现显著下降，他们将此主要归因于有限的嵌入维度。在我们的工作中，我们重新审视了这一解释，并认为仅凭维度不太可能解释所观察到的失效现象；事实上，Alon 等人的结果意味着 2k+1 维足以在固定的查询和文档集合上进行 top-k 检索。基于此，我们研究了导致性能不佳的其他驱动因素，包括分词、语言结构和领域偏移。我们的实证分析表明，分词伪影和语言相似性仅带来适度的改进，而领域偏移以及嵌入相似性与任务特定相关性之间的错位才是主导因素。虽然微调可以部分缓解这些问题并改善……

    arXiv:2603.29519v2 Announce Type: replace  Abstract: Recent work by Weller et al. introduced the naturalistic benchmark LIMIT and showed that widely used single-vector embedding models suffer substantial degradation in retrieval quality, attributing this primarily to limited embedding dimensionality. In our work, we revisit this explanation and argue that dimensionality alone is unlikely to account for the observed failures; in fact, results of Alon et al. imply that $2k+1$ dimensions suffice for top-$k$ retrieval over a fixed set of queries and documents. Motivated by this, we investigate alternative drivers of poor performance, including tokenization, linguistic structure, and domain shift. Our empirical analysis shows that tokenization artifacts and linguistic similarity yield only modest improvements, whereas domain shift and misalignment between embedding similarity and task-specific relevance are the dominant factors. While finetuning partially mitigates these issues and improves
    
[^30]: 解释能否改善推荐效果？来自预测引导式解释的证据

    Can Explanations Improve Recommendations? Evidence from Prediction-Informed Explanations

    [https://arxiv.org/abs/2502.16759](https://arxiv.org/abs/2502.16759)

    该论文提出RecPIE框架，将自然语言解释生成嵌入推荐学习循环，通过预测引导解释、解释反馈优化预测的交替训练机制，同时提升推荐的准确性与可解释性。

    

    推荐系统是数字平台的核心，但它们面临准确性与可解释性之间的根本性权衡：黑盒模型虽然性能强大，却缺乏建立用户信任与采纳所需的解释能力，而现有的可解释AI方法往往只能在事后生成解释，或者以牺牲准确性为代价。我们挑战了这一传统观点，提出当解释被设计为系统的内在组成部分并与预测保持一致时，可以同时提升可解释性和准确性。我们提出了RecPIE（基于预测引导解释的推荐框架），这是一个联合优化推荐预测与自然语言解释的框架。RecPIE将解释生成嵌入到学习循环中：预测引导解释的生成（预测引导式解释），解释再通过交替训练反馈回来以优化后续预测（解释引导式预测）。

    arXiv:2502.16759v4 Announce Type: replace  Abstract: Recommender systems are central to digital platforms, yet they face a fundamental trade-off between accuracy and explainability: Black-box models achieve strong performance but lack interpretability needed for trust and adoption, while existing explainable AI approaches often generate explanations post hoc or sacrifice accuracy. We challenge this view, proposing that explanations, when designed as an integral component of the system and aligned with predictions, can improve \emph{both} interpretability and accuracy. We introduce RecPIE (Recommendation with Prediction-Informed Explanations), a framework that jointly optimizes recommendation predictions and natural-language explanations. RecPIE embeds explanation generation into the learning loop: predictions guide explanation generation (prediction-informed explanations), which are fed back to refine subsequent predictions (explanation-informed predictions) via alternating training. T
    

