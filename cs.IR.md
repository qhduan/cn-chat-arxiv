# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CO-GRAFT: Attributing Reinforcement Learning Gains in Search Agents via Role-Level Policy Grafting](https://arxiv.org/abs/2611.31185) | 提出CO-GRAFT框架，通过角色级策略嫁接与合作博弈的Shapley值分解，将强化学习为搜索智能体带来的整体准确率提升归因到中间推理、搜索决策、查询生成和答案提取四个功能角色上，并利用Möbius交互项区分角色独立改进与多角色耦合效应。 |
| [^2] | [Orthogonally Mixed Sign Coding with Progressive Residual Refinement for Transformer Weight Compression](https://arxiv.org/abs/2611.31089) | 该论文提出了一种无需训练、无需CDF的渐进残差-符号量化框架，通过随机正交混合使残差分布趋近高斯分布，在Transformer权重的低比特压缩下仍能保持接近全精度的模型预测性能。 |
| [^3] | [From Head to Tail: LLM-Augmented Sequential Recommendation and Trade-off Optimization](https://arxiv.org/abs/2611.31052) | 该论文提出利用LLM生成分组特定的合成交互数据，将缓解推荐系统流行度偏差的问题形式化为约束优化，以提升长尾与冷启动项目表现，并发现多分组增强策略并非简单可加，且增强质量随上下文丰富度提高而提升但成本更高。 |
| [^4] | [Ask Again: Answer-List Instability in Generative Engines](https://arxiv.org/abs/2611.30819) | 该研究通过对6万条生成式引擎回答的大规模测量，首次系统揭示了生成式搜索引擎答案列表的不稳定性——同一问题的答案有32%的概率给出不同的第一名，即便重复25次也只有60%的概率能确定接近的领先者，且引擎间差异大于引擎内部、开放式问题的不稳定性是确定性问题的两倍以上，因此对生成式引擎可见性的可靠监测远比现有推荐的方法更困难。 |
| [^5] | [Trallie: from Free-Form Text to Structured Data with LLMs](https://arxiv.org/abs/2611.30773) | Trallie 是一个开源框架，利用大语言模型的零样本/少样本能力实现多种文档格式的信息抽取，并可自动从文档集合中归纳抽取模式，让非专家用户也能便捷地将非结构化文本转换为结构化数据。 |
| [^6] | [SuSi-RAG: A Hybrid Graph RAG for Equipment Maintenance](https://arxiv.org/abs/2611.30588) | SuSi-RAG是一种通过子图相似度匹配用户查询与维护记录的混合图检索增强生成方案，已在博世生产线实际运行超过八个月，将top-1检索准确率从61.5%提升至96.2%，有效解决了传统向量RAG在复杂因果链和跨领域综合查询上的不足。 |
| [^7] | [Fetch, Read, Reuse: How AI Search Agents Bend the Web's Rules](https://arxiv.org/abs/2611.30546) | 该论文通过黑盒测量研究发现，主流AI搜索智能体普遍不遵守robots.txt和Retry-After等网站规则，暗中减少读取的内容并在用户之间共享抓取的页面，偏离了网站发布者所依赖的Web规范。 |
| [^8] | [Can Search Agents Correct Their Own Errors? Evaluating Retrieval, Evidence Delivery, and Answer Completion](https://arxiv.org/abs/2611.30243) | 该研究通过重放模型自身的错误答案并控制其接收到的证据，系统评估了搜索智能体在检索、证据传递和答案修正各环节的自我纠错能力，发现即使正确证据在手，模型仍常受先前错误答案干扰而无法完成纠错。 |
| [^9] | [Ontology-Compiled 2D Clinical State Fields with Bidirectional Diffusion Encoders for Zero-Training Cross-Ontology Medical Coding](https://arxiv.org/abs/2611.30218) | 该论文提出无词表框架OCSF，通过将临床文档编码为固定维度的二维状态矩阵，并利用离线编译的Tikhonov反卷积算子匹配任意目标本体，实现了无需重新训练、数秒内即可切换医学编码词表的零训练跨本体医学编码。 |
| [^10] | [Beyond Knowledge Transfer: How User Overlap Shapes Diversity in Cross-Domain Recommendation](https://arxiv.org/abs/2611.30156) | 提出与模型无关的DivCDR推拉框架，通过为重叠用户构建置信度门控的虚拟正样本拓宽偏好桥梁，并用重叠密度引导的负采样将非重叠用户推离重叠密集区域，在保持准确性的同时显著提升跨领域推荐的多样性。 |
| [^11] | [Pixel as the Common Language: Unified Multimodal Embedding via Visual Rendering](https://arxiv.org/abs/2611.30069) | 该论文提出RenderEmbed框架，通过视觉渲染将文本、图像及组合输入统一转换为像素形式的视觉基底后再编码，实现了真正统一的多模态嵌入表示学习。 |
| [^12] | [STUN-Bench: Truly Unseen Tools for Evaluating Shared Tool Experience](https://arxiv.org/abs/2611.30034) | 该论文提出STUN-Bench基准，构建了经验池中完全不包含172个保留工具任何信息的数据分割，并揭示了共享经验会挤占工具描述排名、导致真正未见的工具从候选列表中消失的“拥挤”问题。 |
| [^13] | [SHAKTI: Evidence-Grounded Conversational AI for Nutrition and Health](https://arxiv.org/abs/2611.29988) | 该论文提出了SHAKTI，一个仅基于检索到的学术文献和精确食物成分数据回答营养健康问题的对话式AI助手，通过来源链接核查和引文数字在线验证等机制确保回答有据可依，有效解决了大语言模型产生无依据论断的问题。 |
| [^14] | [Structure-Exploiting Hierarchical Retrieval for Long Narrative Documents: A Precision-Oriented Alternative to Learned Abstraction Trees](https://arxiv.org/abs/2611.29966) | 提出了一种利用文档原生章节结构的两级层次化检索方法，通过章节摘要混合匹配选择、章节过滤密集搜索和词元预算约束下的感知章节段落扩展，并辅以内容哈希门控的廉价增量索引更新，作为 RAPTOR 等学习型抽象树之外的精度导向替代方案，用于长篇叙事文档的检索增强生成。 |
| [^15] | [Look Before You Leap: Re-Architecting Disk-Resident Vector Search with TriVS](https://arxiv.org/abs/2611.29948) | TriVS通过以I/O为中心的分析揭示了基于聚类与基于图的搜索在I/O上的互补权衡，并创新性地将磁盘驻留向量搜索重构为“粗略接近、渐进蒸馏、精细收敛”的三部曲范式，融合两种粒度的I/O优势以突破I/O瓶颈。 |
| [^16] | [Where Do Fact-Checking Errors Arise? Separating Retrieval, Search Control and Judgement in Evidence-Grounded Verification](https://arxiv.org/abs/2611.29697) | 该论文提出一套分离检索、搜索控制、判断与裁决验证的诊断协议，发现事实核查失败的主要根源不在证据检索（单次混合检索即可覆盖绝大多数声明的完整证据），而在于证据的后续使用环节——判断器弃权与验证规则推翻了本可正确作出的裁决。 |
| [^17] | [Weight Coding: Evolving Web-Scale Recommenders through Direct Weight Edits with LLM Agents](https://arxiv.org/abs/2611.29613) | 该论文提出“权重编码”新方法，利用LLM智能体通过直接、可执行的权重编辑来演化网络规模推荐模型，在Pinterest图像搜索双塔召回器上以仅约1/4的微调成本将recall@10提升高达15个百分点（相对提升24%），远超常规上线的约1%提升。 |
| [^18] | [MoR4Rec: Mixture-of-Recursion for Sequential Recommendation](https://arxiv.org/abs/2611.29462) | 该论文提出MoR4Rec，一种基于递归混合范式的序列推荐框架，通过为不同复杂度的物品自适应分配计算深度，解决了传统Transformer对所有交互物品采用固定计算深度所导致的模型容量浪费与计算开销过高的问题。 |
| [^19] | [VistaTree: Question Answering over Long Multimodal Documents with Image-Derived Summary Trees](https://arxiv.org/abs/2611.29419) | VistaTree通过直接从页面图像构建层次化摘要树并进行层级导航，实现了对超出上下文窗口的长篇多模态文档的高效问答。 |
| [^20] | [LLM-Based Assistant for Dog Care Guidance](https://arxiv.org/abs/2611.29286) | 研究通过对比实验发现，基于RAG检索增强的领域接地方法在提升大语言模型生成狗狗护理建议的准确性方面显著优于LoRA参数高效微调。 |
| [^21] | [NoeVera: Can Knowledge Guide Experiments and Experiments Refine Knowledge?](https://arxiv.org/abs/2611.29255) | NoeVera是一个智能体式“第二大脑”，它将论文、代码、基准与实验结果整合为保留来源、持久化的三层知识库，通过决策感知检索、角色化智能体推理与模拟器调用，构建“知识指导实验、实验反哺知识”的闭环，以优化LLM服务系统的科研决策。 |
| [^22] | [Zarc: Exact Nearest Neighbors over a Billion Vectors in Seconds](https://arxiv.org/abs/2611.29252) | Zarc通过在GPU内存中仅保留短小的距离证明、并从主机内存按需取回少量完整向量，实现了在多GPU上对十亿级向量在数秒内完成精确k近邻搜索，将10,000个查询的耗时从五分钟降至1.1秒。 |
| [^23] | [DataRecycle: Trustworthy Distillation of Structured Data from Scattered Web Evidence](https://arxiv.org/abs/2611.29206) | DataRecycle是一个交互式系统，通过将用户查询分解为实体与属性并组织候选证据，把网络上零散且可能矛盾的信息可信地转化为可比较、可追溯的结构化数值事实表格。 |
| [^24] | [Bridging Blind Spots With Words: Decentralized Multi-View Spatial Reasoning via Language-Constrained Communication](https://arxiv.org/abs/2611.29112) | 该论文首次形式化了去中心化多视角空间推理这一通用问题（N个智能体各自观察不同视角并通过受限自然语言消息协作），并提出了新的多视角空间基准，研究发现用预算受限的自然语言进行空间推理整合的代价远小于预期。 |
| [^25] | [HME-PFD: Heterogeneous Mixture-of-Experts Privileged Feature Distillation](https://arxiv.org/abs/2611.29094) | 提出HME-PFD方法，通过在专家混合学生模型的路由模块中引入硬编码与可学习两种路由机制来融合特权特征，从而动态应对特权特征覆盖率有限且分布异构的问题，提升广告推荐场景下的蒸馏效果。 |
| [^26] | [Who Gets Re-ranked When a Clinical Model Gains New Data?](https://arxiv.org/abs/2611.28968) | 该论文研究了医院临床风险模型添加新数据源（如实验室结果）时缺乏新数据的患者如何被重新排序，发现实验室结果虽能改善预测，却会使未经检测的危重患者在两家医院的不同模型中都下降到警报名单的更低位置。 |
| [^27] | [ChatGPT Can't Sell (And That's Probably a Good Thing): Buyer-Aligned AI Advice, the Sales-Representative Gap, and Evidence from Commercial Conversations](https://arxiv.org/abs/2611.28946) | 该研究通过三类大规模商业对话证据表明，ChatGPT等AI助手在购买情境中只提供推荐和购买资格甄别建议，而不会执行真正的交易促进行动，作者认为这种与买家立场一致、无法代商家完成销售的特性可能是件好事。 |
| [^28] | [PROSPER: LLMs as Sparse Retrievers for First-Stage Product Search](https://arxiv.org/abs/2611.28645) | 提出PROSPER框架，将大语言模型用作商品搜索第一阶段的稀疏检索器，在利用其语义扩展能力的同时显式保持品牌名、型号等字面信号的词汇保真度。 |
| [^29] | [MADRec: Reliable Multi-Agent Reasoning with Evidence-Grounded Debate for Recommendation](https://arxiv.org/abs/2611.28604) | 提出了一种以证据为支撑的多智能体辩论推荐框架MADRec，通过评估协作过程中中间推理的可靠性来防止错误判断在智能体间传播，从而提升多智能体推荐推理的可靠性和推荐质量。 |
| [^30] | [The abc index: Exact Retrieval for Learned Sparse Representations](https://arxiv.org/abs/2611.28555) | abc索引仅用标准IR技术（TAAT评估、Block-Max MaxScore、二进制压缩和文档ID重排序）构建，在MS MARCO上成为各种k值下最快的精确LSR检索索引，不仅比现有精确索引快数倍到数十倍且体积更小，还能在大k值时反超近似索引并节省1.5-3.6倍空间。 |
| [^31] | [Beyond Relevance: An Empirical Study of Multimodal LLMs for Diversity-Aware Recommendation Reranking](https://arxiv.org/abs/2611.28379) | 本文提出一个无需训练的多样性感知多模态重排序框架，通过整合物品图像、文本元数据与用户交互历史，实证表明多模态大语言模型能够同时提升推荐的相关性与多样性，且视觉信息带来的增益具有明显的领域依赖性。 |
| [^32] | [GERF: A Token-Efficient Multimodal Retrieval Framework for Schema-Guided Scientific Information Extraction](https://arxiv.org/abs/2611.28295) | 提出了GERF框架，将论文表示为连接模式字段、文本块与插图的异构证据图，实现令牌高效、模式引导的多模态科学信息抽取，显著减少无关证据并降低文本与视觉token成本。 |
| [^33] | [DM-Lab at TREC 2026 RAG: Multi-Query Retrieval, Ensemble Judging, and Claim-Pack Generation](https://arxiv.org/abs/2611.28233) | DM-Lab-RAG-2026框架通过多查询BM25检索、倒数排名融合、集成相关性评判以及声明优先的引用验证生成策略，在5.53亿文档语料库上实现了可靠的检索增强问答，并在TREC 2026 RAG官方评测中取得0.910的人工要点增益。 |
| [^34] | [Evaluating Memanto on DolphinBench for AI Agents](https://arxiv.org/abs/2611.28231) | Memanto记忆智能体凭借信息论相关性分数对记忆排序，在DolphinBench职场记忆基准上以更少的迭代次数（最多6次）、更低的成本（10.86美元）和更短的延迟（15.32秒中位数）取得83.83%的任务完成率，显著超越此前最强的Mem0配置（70.67%）。 |
| [^35] | [Close the Loop on the Retrievability Gap](https://arxiv.org/abs/2611.28146) | 本文揭示了全色锐化领域中保真度指标与下游检索性能之间的“可检索性差距”，并提出LEDGER方法，通过测试时无需标签的集合级软Jaccard损失围绕冻结视觉-语言编码器构建闭环，使每个概念都获得梯度贡献，从而弥合这一差距。 |
| [^36] | [Monte Carlo Power Analysis for Small-System A/B Trials: Detecting Differences in Newsletter Engagement](https://arxiv.org/abs/2611.28018) | 本文提出一种蒙特卡洛方法，用于估计旨在提升用户对个性化新闻邮件通讯参与度的A/B实验的统计功效。 |
| [^37] | [CACO: Coverage-Aware Coordinated Optimization for Continuous Generative Retrieval](https://arxiv.org/abs/2611.25738) | CACO是一种轻量级自监督后训练框架，通过覆盖感知的无标签目标将多个采样查询协调至互补的物品覆盖，在几乎不增加推理延迟的情况下显著提升连续生成式检索的召回率。 |
| [^38] | [KnowMe Paper Radar: Lightweight Discovery of Promising Papers in the Age of Autoresearch](https://arxiv.org/abs/2611.23702) | KnowMe Paper Radar 是一个无需 LLM 推理的轻量级论文发现工具，它综合主题相关性、时效性、作者研究历史、合作路径和机构背景等多源证据生成个性化阅读优先级，帮助研究者快速筛选潜力论文，并将重复上下文准备速度提升三倍以上。 |
| [^39] | [IntentFlow: An Intent-Driven Real-Time Multimodal Search System with Image-Text Queries at Xiaohongshu](https://arxiv.org/abs/2611.20448) | IntentFlow是小红书部署的意图驱动实时多模态搜索系统，通过层次化意图分类体系和意图特定的相关性标准，在百亿级索引上实现了图文查询场景下候选检索与重排序的一致性对齐。 |
| [^40] | [Straight to the Product: What Online Retailers Miss When Shoppers Ask AI First](https://arxiv.org/abs/2611.20378) | 本文首次将用户的AI助手对话与其在二十家大型在线零售商的实际浏览行为关联，发现助手提及的产品会使用户在一天内到访该产品页的概率提升约百倍（从0.02升至2.07每百次回答），但89%的访问并未通过助手链接到达、多经网页搜索实现，说明零售商追踪不到也错失了这类由AI引导而来的流量。 |
| [^41] | [Anonymous in Name Only: A Critical Review and a Reidentification Exercise on the European Commission's Anonymization Measures for Google Search Data under Article 6(11) of the DMA](https://arxiv.org/abs/2611.17012) | 本文批判性地审查了欧盟委员会依据《数字市场法》第6(11)条为谷歌搜索数据共享所制定的匿名化措施，指出其存在重大概念缺陷，并通过在真实数据上的低成本重识别实验证明该方法不应成为监管基准。 |
| [^42] | [SPARSEUP: Towards Better Recipes for Training Modern Learned Sparse Retrievers](https://arxiv.org/abs/2611.09680) | 本文提出 SPARSEUP 训练方案，通过 logit 分布调控、逐位置扇出硬上限以及词汇表合并策略，让基于现代骨干网络的 Splade 类稀疏检索器实现更优的效率-效果权衡。 |
| [^43] | [AutoRSE: Evaluating LLM Agents as Recommendation Engineers for Self-Evolving Recommender Systems](https://arxiv.org/abs/2610.37228) | 该论文提出了AutoRSE评估框架，通过覆盖六项工业级推荐任务的大规模实证研究发现，基于LLM的推荐工程智能体能将推荐性能平均提升16.1%，但目前主要充当工程优化者而非算法创新者。 |
| [^44] | [The Price of Scale in Proactive Memory Recall](https://arxiv.org/abs/2610.34507) | 该论文证明主动记忆召回的门控决策本质上是一个随记忆库规模 N 增长的多重假设检验问题——单条记忆的假阳性率按 -ln(1-α)/N 衰减，导致命中率随 N 呈幂次下降，从理论与实证上解释了记忆库越大、主动记忆触发越容易失效的现象。 |
| [^45] | [A Study of Prior Case Retrieval Using Lexical, Semantic, and Rhetorical Role Information in Indian Legal Documents](https://arxiv.org/abs/2610.07437) | 本研究针对印度法律先前案例检索任务进行了三种检索系统设计的实证评估，发现修辞角色的组合检索优于单一角色或全文检索、法条相似性不具区分性，并揭示了内部验证MRR高达0.97的模型在官方评估中MRR仅0.14的泛化失效问题。 |
| [^46] | [MUSES: A Benchmark for Prospective Intellectual-Roots Retrieval](https://arxiv.org/abs/2609.00313) | 该论文提出了首个百万规模的前瞻性学术思想根源检索基准MUSES，通过作者确认的论文层面根源标注和熟悉度分级任务设计，用于评估检索系统发现那些日后被证明具有开创性、但目前不为人熟知的小众文献的能力。 |
| [^47] | [Multi Interests for Joint Search-Recommendation Modeling](https://arxiv.org/abs/2608.10535) | 提出基于多兴趣的混合序列建模框架MIJSR，从结构与语义两个视角对搜索推荐混合序列进行多兴趣挖掘与自适应融合，实现搜索与推荐的联合建模。 |
| [^48] | [IDP AutoOpt: Agent-Driven Optimization of Document Processing Pipeline Configurations](https://arxiv.org/abs/2607.26075) | IDP AutoOpt通过闭环LLM代理自动优化文档处理流水线配置，在降低成本的同时匹配或超越人类专家准确性，并将配置时间从数周缩短到两小时以下。 |
| [^49] | [Invisible to humans, visible to machines: a preregistered audit of Unicode fidelity across four biomedical bibliographic APIs](https://arxiv.org/abs/2606.24897) | 本研究开发并预注册了一种可复用的字符级保真度审计方法，系统评估了PubMed、Crossref、OpenAlex和Semantic Scholar四个生物医学文献API对Unicode特殊字符（如希腊字母、数学符号、排版标点和特殊空白）的保留能力，揭示了这些API存在的系统性字符丢失问题。 |
| [^50] | [LCSHBench: A Multilingual, Consensus-Grounded Benchmark for Library of Congress Subject Heading Assignment](https://arxiv.org/abs/2606.04382) | LCSHBench是首个基于多家图书馆编目共识构建的多语言公开基准，涵盖15种语言2万余本书，用于评估LCSH主题标目的自动分配，并同时衡量精确匹配与概念级匹配两种结果。 |
| [^51] | [MusicRec: Audio, Lyrics, and Engagement Feature Enrichment for Music Recommendation](https://arxiv.org/abs/2606.00125) | 该论文提出MusicRec——一个基于LLM的多模态音乐推荐框架，首次在统一的、基于歌曲内容的框架内联合建模音频/歌词嵌入、LLM生成的语义元数据和听歌完成率这三种互补信号，从而提升基于会话的音乐推荐效果。 |
| [^52] | [Bayesian Rational Search Engine User](https://arxiv.org/abs/2605.24233) | 该论文将搜索引擎用户的付费浏览行为建模为贝叶斯最优停止问题，证明最优策略是随深度变化的“突出规则”，通过一维马尔可夫链闭式刻画检查深度的完整分布，揭示信任、承诺与止损三种停止原因，并将观察到的检查深度转化为学习排序的似然函数。 |
| [^53] | [EcoGEO: Trajectory-Aware Evidence Ecosystems for Web-Enabled LLM Search Agents](https://arxiv.org/abs/2605.12887) | 本文提出EcoGEO框架，将生成式引擎优化（GEO）从单网页层面扩展为面向网络增强型LLM搜索智能体的环境级影响力问题，并设计了轨迹感知的协同证据生态系统TRACE，通过控制智能体浏览轨迹中证据的组织、连接与呈现方式来影响其搜索结果。 |
| [^54] | [Evaluating Small Open LLMs for Medical Question Answering: A Practical Framework](https://arxiv.org/abs/2604.10535) | 该论文提出了一个开源评估框架，将可复现性与准确性并列为首要指标，通过八项质量指标和重复推理来评估可本地部署的小型开源LLM在医学问答中的表现。 |
| [^55] | [UniFAR: A Unified Facet-Aware Retrieval Framework for Scientific Documents](https://arxiv.org/abs/2602.23766) | 提出了UniFAR，一个统一的分面感知检索框架，通过多粒度表示与聚合模块在共享表示空间中同时支持文档-文档和问答-文档两种科学文献检索范式，从而融合二者的互补优势。 |
| [^56] | [Transform Before You Query: A Privacy-Preserving Approach for Vector Retrieval with Embedding Space Alignment](https://arxiv.org/abs/2507.18518) | 提出STEER隐私保护向量检索框架，利用不同嵌入模型语义空间之间的对齐关系，在本地将查询文本变换为近似嵌入向量后再查询，从而避免向向量数据库服务商暴露包含敏感信息的原始查询文本。 |
| [^57] | [Evaluating Conversational Recommender Systems via Large Language Models: A User-Centric Framework](https://arxiv.org/abs/2501.09493) | 该论文提出了基于大语言模型的以用户为中心评估框架CoRE，通过单一指标同时评估对话式推荐系统的推荐准确性与对话管理质量，真实反映用户体验。 |

# 详细

[^1]: CO-GRAFT：通过角色级策略嫁接归因搜索智能体中强化学习的增益

    CO-GRAFT: Attributing Reinforcement Learning Gains in Search Agents via Role-Level Policy Grafting

    [https://arxiv.org/abs/2611.31185](https://arxiv.org/abs/2611.31185)

    提出CO-GRAFT框架，通过角色级策略嫁接与合作博弈的Shapley值分解，将强化学习为搜索智能体带来的整体准确率提升归因到中间推理、搜索决策、查询生成和答案提取四个功能角色上，并利用Möbius交互项区分角色独立改进与多角色耦合效应。

    

    强化学习（RL）可以将大型语言模型搜索智能体的性能较其强化学习前的初始版本提升多达32个精确匹配（EM）分值，然而整体准确率的提升并不能识别出在训练过程中哪些底层功能能力得到了改善或发生了交互作用。我们提出了CO-GRAFT，这是一个与奖励无关的干预式归因框架，其中经过强化学习训练的智能体与其强化学习前的初始版本在闭环执行和实时检索的条件下联合生成单条轨迹。每个策略控制四个功能角色中互斥的一个子集：中间推理、搜索或回答决策、搜索查询生成以及最终答案提取。在不修改权重或重新训练的情况下评估所有2^4=16种角色分配，这定义了一个合作博弈，其Shapley值将总的强化学习增益分解到各个角色上，而Möbius交互项则区分了单个角色的独立改进与多角色之间的耦合效应。（原文摘要在此处截断）

    arXiv:2611.31185v1 Announce Type: cross  Abstract: Reinforcement learning (RL) improves large language model search agents by up to 32 exact-match (EM) points over their pre-RL initializations, yet overall accuracy gains do not identify which underlying functional capabilities improve or interact during training. We introduce CO-GRAFT, a reward-agnostic interventional attribution framework in which an RL-trained agent and its pre-RL initialization jointly generate a single trajectory under closed-loop execution with live retrieval. Each policy controls a mutually exclusive subset of four functional roles: intermediate reasoning, search-or-answer decision, search query generation, and final answer extraction. Evaluating all 2^4=16 role assignments without modifying weights or retraining defines a cooperative game whose Shapley values decompose the total RL gain across roles, while M"obius interaction terms distinguish standalone role improvements from multi-role coupling. Applying CO-GR
    
[^2]: 面向Transformer权重压缩的正交混合符号编码与渐进残差精炼方法

    Orthogonally Mixed Sign Coding with Progressive Residual Refinement for Transformer Weight Compression

    [https://arxiv.org/abs/2611.31089](https://arxiv.org/abs/2611.31089)

    该论文提出了一种无需训练、无需CDF的渐进残差-符号量化框架，通过随机正交混合使残差分布趋近高斯分布，在Transformer权重的低比特压缩下仍能保持接近全精度的模型预测性能。

    

    大语言模型量化可以降低存储成本，但在低比特率下往往会损害预测保真度。我们提出了一种无需训练、无需CDF（累积分布函数）、受高斯中心极限定理启发的渐进残差-符号量化框架，该框架利用随机化正交混合促使残差分布趋于高斯分布，从而改进基于符号的Transformer权重重建。每个阶段独立地对残差进行归一化，并以解析方式校准重建过程，逐步恢复量化误差。该框架无需学习码本、经验CDF或网格优化，即可实现灵活的存储-保真度权衡。我们在GPT-2 Large和Llama-3-8B的注意力与MLP投影上进行了评估。在五个残差阶段下，联合量化保持了接近基线的预测保真度，困惑度分别达到32.16和12.73，而全精度模型的对应值分别为29.89和11.07。消融实验进一步证明，正交混合显著（原文此处截断）……

    arXiv:2611.31089v1 Announce Type: new  Abstract: LLM quantization reduces storage but often degrades predictive fidelity at low bit rates. We propose a training-free, CDF-free, Gaussian-CLT-motivated progressive residual-sign quantization framework that uses randomized orthogonal mixing to promote Gaussian-like residual distributions and improve sign based reconstruction of Transformer weights. Each stage independently normalizes residuals and analytically calibrates reconstruction, progressively recovering quantization errors. The framework enables flexible storage-fidelity trade-offs without learned codebooks, empirical CDFs, or trellis optimization. We evaluate GPT-2 Large and Llama-3-8B across attention and MLP projections. At five residual stages, joint quantization preserves near-baseline predictive fidelity, achieving perplexities of 32.16 and 12.73, compared with full-precision values of 29.89 and 11.07, respectively. Ablations further demonstrate that orthogonal mixing substan
    
[^3]: 从头部到尾部：LLM增强的序列推荐与权衡优化

    From Head to Tail: LLM-Augmented Sequential Recommendation and Trade-off Optimization

    [https://arxiv.org/abs/2611.31052](https://arxiv.org/abs/2611.31052)

    该论文提出利用LLM生成分组特定的合成交互数据，将缓解推荐系统流行度偏差的问题形式化为约束优化，以提升长尾与冷启动项目表现，并发现多分组增强策略并非简单可加，且增强质量随上下文丰富度提高而提升但成本更高。

    

    推荐系统存在流行度偏差，常常对交互数据稀疏的长尾项目和冷启动项目服务不足。我们提出了一种面向特定分组的数据增强方法，利用大语言模型（LLM）生成合成交互数据，并将其形式化为一个约束优化问题：在将整体NDCG@20的下降限制在可接受阈值内的同时，最大化各分组的NDCG@20和覆盖率。项目按交互频率进行划分，以找到每个分组的最佳增强策略。我们进一步研究了同时组合跨分组增强策略的效果，发现虽然单个最优策略可以指导多分组增强设计，但它们并非简单可加。我们还对基于LLM的增强本身进行了消融实验，比较了不同规模的LLM以及token约束下的增强配置，发现增强质量随上下文丰富度的提升而提升，但相应成本也成比例增加。

    arXiv:2611.31052v1 Announce Type: cross  Abstract: Recommender systems exhibit popularity bias, often under-serving long-tail and cold-start items with sparse interaction data. We propose a group-specific data augmentation approach that leverages LLMs to generate synthetic interactions, formulated as a constrained optimization problem, maximizing group-specific NDCG@20 and coverage while bounding overall NDCG@20 degradation within an acceptable threshold. Items are partitioned by interaction frequency to find the optimal augmentation strategy for each group. We further study combining the augmentation across groups simultaneously and find that while individual optimal strategies navigate the multi-group augmentation design, they are not simply additive. We further ablate on the LLM-based augmentation itself, comparing LLM scales, augmentation configuration under token constraints, finding that augmentation quality scales with contextual richness, albeit at a proportionally higher cost.
    
[^4]: 再次询问：生成式引擎中答案列表的不稳定性

    Ask Again: Answer-List Instability in Generative Engines

    [https://arxiv.org/abs/2611.30819](https://arxiv.org/abs/2611.30819)

    该研究通过对6万条生成式引擎回答的大规模测量，首次系统揭示了生成式搜索引擎答案列表的不稳定性——同一问题的答案有32%的概率给出不同的第一名，即便重复25次也只有60%的概率能确定接近的领先者，且引擎间差异大于引擎内部、开放式问题的不稳定性是确定性问题的两倍以上，因此对生成式引擎可见性的可靠监测远比现有推荐的方法更困难。

    

    生成式搜索引擎中的可见性是通过哪些实体出现在回答中、以及以什么顺序出现来衡量的。向引擎询问同一个问题两次，答案列表就可能发生变化：排序不同，或者条目完全不同。通过对60,000条回答的分析，我们发现对同一个提示，答案有32%的概率给出不同的第一项；即使两个答案包含完全相同的条目，它们对哪个条目排在第一位的判断也不一致。第一名本身就是一个独立的测量目标。在运行七到八次之后，才会出现一个“明显的领先者”——而这正是品牌与来源监控所推荐的测量次数预算——但“接近的领先者”只在一半左右的情况下出现。即使运行25次，接近的领先者也只有在60%的情况下能被识别出来。引擎之间的变异性大于同一引擎内部的变异性。对于开放式提示，其差距是确定性提示的两倍以上。观察到的搜索查询和引用网站只能解释其中一小部分分歧，大部分分歧仍然无法解释。可见性可测

    arXiv:2611.30819v1 Announce Type: cross  Abstract: Visibility in generative search engines is measured by which entities appear in responses, and in what order. Ask an engine the same question twice and the answer list can change: a different ordering, or different items altogether. Across 60,000 responses, we find that answers to the same prompt name a different first item 32% of the time, and even two answers with the same items disagree on which comes first. The first position is a measurement target of its own. A "clear leader" emerges after seven or eight runs, which are the budgets recommended for brand and source monitoring, but a "close leader" only about half the time. After 25 runs, a close leader is identified only 60% of the time. Variability between engines is larger than within the same engine. For open-ended prompts, the gap is more than double that for determinate ones. Search queries and cited sites observed leave most of the disagreement unexplained. Visibility measur
    
[^5]: Trallie：利用大语言模型将自由格式文本转换为结构化数据

    Trallie: from Free-Form Text to Structured Data with LLMs

    [https://arxiv.org/abs/2611.30773](https://arxiv.org/abs/2611.30773)

    Trallie 是一个开源框架，利用大语言模型的零样本/少样本能力实现多种文档格式的信息抽取，并可自动从文档集合中归纳抽取模式，让非专家用户也能便捷地将非结构化文本转换为结构化数据。

    

    Trallie 是一个开源框架，它利用大语言模型（LLM）在有或没有用户指导的情况下从自然语言来源进行信息抽取（IE）。与依赖大量标注数据集或领域专用训练的传统方法不同，Trallie 能够在仅有最少甚至没有代表性示例的情况下，将非结构化数据转换为结构化格式。该框架的设计包含三个关键目标：（i）通过利用现代大语言模型基于提示的零样本和少样本能力来减少人工标注；（ii）灵活解析多种文档格式（即 HTML、PDF、TXT 和 JSON）；（iii）通过从文档集合中自动归纳抽取模式来提升非专家用户的易用性，这些模式随后可以被细化用于信息抽取。我们在 FDA 510(k) 文档以及 SWDE 数据集中的电影和大学网站上，使用开源权重的大语言模型对 Trallie 进行了评估，其表现优于先前报告的方法。

    arXiv:2611.30773v1 Announce Type: new  Abstract: Trallie is an open-source framework that leverages large language models (LLMs) to perform information extraction (IE) from natural language sources with or without user guidance. Unlike conventional approaches that rely on extensive labeled datasets or domain-specific training, Trallie enables the conversion of unstructured data into structured formats with minimal or no representative examples. The framework is designed with three key objectives: (i) reducing manual annotation by harnessing prompt-based zero-shot and few-shot capabilities of modern LLMs, (ii) flexibly parsing diverse document formats (i.e., HTML, PDF, TXT, and JSON), and (iii) improving accessibility for non-expert users by automatically inducing schemas from document collections, which can then be refined for extraction. We evaluate Trallie with open-weight LLMs on FDA 510(k) documents and on movie and university websites from SWDE. It outperforms previously reported 
    
[^6]: SuSi-RAG：一种面向设备维护的混合图检索增强生成方案

    SuSi-RAG: A Hybrid Graph RAG for Equipment Maintenance

    [https://arxiv.org/abs/2611.30588](https://arxiv.org/abs/2611.30588)

    SuSi-RAG是一种通过子图相似度匹配用户查询与维护记录的混合图检索增强生成方案，已在博世生产线实际运行超过八个月，将top-1检索准确率从61.5%提升至96.2%，有效解决了传统向量RAG在复杂因果链和跨领域综合查询上的不足。

    

    在制造领域，设备维护任务依赖于专家知识，而其文档通常以异构的格式和来源存储。利用检索增强生成（RAG）来增强大语言模型（LLM）有望弥合这些分散的知识，但传统的基于向量的RAG在面对真实维护查询中常见的两个特征——复杂因果链和跨领域综合——时表现出明显弱点。为解决这一问题，我们提出了SuSi-RAG，一种混合图RAG解决方案，通过子图相似度将用户查询与每条维护记录进行匹配。本工作展示了一个已实际投入使用的系统：SuSi-RAG已在博世（Bosch）一条活跃的生产线上运行超过八个月，处理了10,664次问答交互，答案准确率上限达到96.25%（确认错误率为3.75%）。在概念验证中，SuSi-RAG将top-1检索准确率从61.5%提升至96.2%，专家评分的答案质量也得到显著改善。

    arXiv:2611.30588v1 Announce Type: cross  Abstract: In the manufacturing domain, equipment maintenance tasks depend on expert knowledge. Their documentation is often stored in heterogeneous formats and sources. The enhancement of LLMs with Retrieval-Augmented Generation (RAG) is promising for bridging the scattered knowledge, but traditional vector-based RAG shows weaknesses with two characteristics which are common in real maintenance queries: complex causal chains and cross-domain synthesis. To address this, we propose SuSi-RAG, a hybrid Graph RAG solution, that matches a user query and each maintenance record using subgraph similarity. This work shows an in-use system: SuSi-RAG has run on an active Bosch production line for more than eight months, handling 10,664 question-answer interactions and demonstrates upper bound answer accuracy of 96.25% (confirmed error rate 3.75%). In a proof-of-concept, SuSi-RAG improves top-1 retrieval accuracy from 61.5% to 96.2% and expert-rated answer 
    
[^7]: 获取、阅读、复用：AI搜索智能体如何扭曲万维网的规则

    Fetch, Read, Reuse: How AI Search Agents Bend the Web's Rules

    [https://arxiv.org/abs/2611.30546](https://arxiv.org/abs/2611.30546)

    该论文通过黑盒测量研究发现，主流AI搜索智能体普遍不遵守robots.txt和Retry-After等网站规则，暗中减少读取的内容并在用户之间共享抓取的页面，偏离了网站发布者所依赖的Web规范。

    

    AI搜索智能体构成了一类新兴且快速增长的网络流量。它们像浏览器一样按需为单个用户抓取页面；像转码代理一样重写所抓取的内容；又像共享缓存一样，利用存储的副本回答其他用户的问题。网站发布者所依赖的规则，例如robots.txt、Retry-After和Cache-Control，原本是为更常见的客户端编写的，因此由智能体的供应商自行决定遵循其中哪些规则。我们对这些智能体如何抓取、阅读和复用页面进行了一项黑盒测量研究，涵盖了OpenAI、Anthropic、xAI和Google的搜索API以及十项服务的消费级应用。我们发现OpenAI和xAI不遵守robots.txt，且没有任何服务会对Retry-After做出响应。各服务暗中读取的内容少于源站所提供的内容：OpenAI会丢弃某些格式，没有服务能读取仅由JavaScript呈现的内容，而xAI只读取页面的前0.5 MB。OpenAI、Anthropic和xAI还会在用户之间共享已抓取的页面。

    arXiv:2611.30546v1 Announce Type: cross  Abstract: AI search agents form a new and fast-growing class of web traffic. Like a browser, they fetch pages on demand for one user; like a transcoding proxy, they rewrite what they fetch; and like a shared cache, they answer other users' questions from the stored copy. The rules publishers rely on, such as robots.txt, Retry-After, and Cache-Control, were written for more familiar clients, so the agent's vendor decides which of them to follow. We present a black-box measurement study of how these agents fetch, read, and reuse pages, covering the search APIs of OpenAI, Anthropic, xAI, and Google and the consumer apps of ten services. We find that OpenAI and xAI do not honor robots.txt, and no service acts on Retry-After. Services silently read less than the origin served: OpenAI drops some formats, no service reads JavaScript-only content, and xAI reads only the first 0.5 MB of a page. OpenAI, Anthropic, and xAI share fetched pages across users 
    
[^8]: 搜索智能体能否纠正自身的错误？评估检索、证据传递与答案完成

    Can Search Agents Correct Their Own Errors? Evaluating Retrieval, Evidence Delivery, and Answer Completion

    [https://arxiv.org/abs/2611.30243](https://arxiv.org/abs/2611.30243)

    该研究通过重放模型自身的错误答案并控制其接收到的证据，系统评估了搜索智能体在检索、证据传递和答案修正各环节的自我纠错能力，发现即使正确证据在手，模型仍常受先前错误答案干扰而无法完成纠错。

    

    搜索智能体可能因为错过正确的信息来源、忽略相关段落，或在收到证据后未能修改并给出答案，而使先前的错误得不到解决。我们在BrowseComp-Plus数据集上评估这些可能性，方法是在对话中重放每个模型自己给出的错误闭卷答案，并对模型接收到的证据进行干预。在一项涵盖四个开源权重模型、709个问题和26,665个测试条件的证据供给研究中，即使上下文中已包含含有答案的段落，一个自信的助手先前回答仍使Nemotron的配对准确率下降了16.1个百分点。当用户坚持先前答案时，Nemotron的准确率下降了32.9个百分点，gpt-oss下降了12.6个百分点。在该条件下失去正确性的gpt-oss回复中，弃权比重复先前答案更为常见。在对Qwen3.5和Nemotron进行的独立搜索干预中，将包含正确答案的关键页面置于首位往往导致决定性的信息来源未被打开。

    arXiv:2611.30243v1 Announce Type: new  Abstract: A search agent can leave an earlier error unresolved by missing the right source, overlooking a relevant passage, or failing to revise and deliver the answer after receiving the evidence. We evaluate these possibilities on BrowseComp-Plus by replaying each model's own wrong closed-book answer in conversation and intervening on what evidence reaches the model. In a supplied-evidence study spanning four open-weight checkpoints, 709 questions and 26,665 test conditions, a confident assistant prior reduced Nemotron's paired accuracy by 16.1 percentage points despite answer-bearing passages in context. When the user insisted on the prior, accuracy fell by 32.9 points for Nemotron and 12.6 for gpt-oss. Among gpt-oss responses that lost correctness in that condition, abstention was more common than repeating the prior. In separate search interventions with Qwen3.5 and Nemotron, placing a gold page first often left the decisive source unopened. 
    
[^9]: 基于本体编译二维临床状态场与双向扩散编码器的零训练跨本体医学编码

    Ontology-Compiled 2D Clinical State Fields with Bidirectional Diffusion Encoders for Zero-Training Cross-Ontology Medical Coding

    [https://arxiv.org/abs/2611.30218](https://arxiv.org/abs/2611.30218)

    该论文提出无词表框架OCSF，通过将临床文档编码为固定维度的二维状态矩阵，并利用离线编译的Tikhonov反卷积算子匹配任意目标本体，实现了无需重新训练、数秒内即可切换医学编码词表的零训练跨本体医学编码。

    

    医疗系统不断修订临床分类体系、更新报销计划并协调跨国登记系统。现有的自动化编码模型将临床文本理解绑定到固定的词表专用输出头上，每当编码集合发生变更时都需要人工重新标注和重新训练。我们提出了本体编译二维临床状态场，这是一个无词表框架，它将临床文档阅读与目标本体索引解耦，使得一个训练好的编码器可以在数秒内切换到任意医学词表，而无需针对目标词表进行训练。单次前向传播的双向扩散预训练编码器与无词表解码器将每份临床病历映射为固定维度的二维状态矩阵，不包含任何编码特定参数。任意目标本体都可以通过公开的文本描述在CPU上离线编译为一个尺度不变的Tikhonov反卷积算子，该算子在压制同级编码干扰的同时……（原文摘要在此处截断）

    arXiv:2611.30218v1 Announce Type: new  Abstract: Healthcare systems continually revise clinical taxonomies, update reimbursement schedules, and harmonize cross-border registries. Existing automated coding models bind clinical text understanding to fixed vocabulary-specific heads, requiring manual relabeling and retraining whenever code sets change. We introduce Ontology-Compiled 2D Clinical State Fields (OCSF), a vocabulary-free framework that decouples clinical document reading from target ontology indexing so one trained encoder can switch across arbitrary medical vocabularies in seconds without target training. A single-pass bidirectional diffusion-pretrained encoder and vocabulary-free decoder map each clinical note into a fixed-dimension two-dimensional state matrix with zero code-specific parameters. Any target ontology is compiled offline on CPU from public text descriptions into a scale-invariant Tikhonov deconvolution operator that suppresses sibling-code interference while pr
    
[^10]: 超越知识迁移：用户重叠如何塑造跨领域推荐中的多样性

    Beyond Knowledge Transfer: How User Overlap Shapes Diversity in Cross-Domain Recommendation

    [https://arxiv.org/abs/2611.30156](https://arxiv.org/abs/2611.30156)

    提出与模型无关的DivCDR推拉框架，通过为重叠用户构建置信度门控的虚拟正样本拓宽偏好桥梁，并用重叠密度引导的负采样将非重叠用户推离重叠密集区域，在保持准确性的同时显著提升跨领域推荐的多样性。

    

    跨领域推荐（CDR）中的一种常见范式是通过源领域和目标领域共享的用户来迁移知识。尽管这种重叠桥梁在提高准确性方面行之有效，但其对多样性的影响仍不明确。我们识别出两个瓶颈：当重叠用户所覆盖的目标领域偏好有限时，桥梁会“过窄”；当共享信号压过非重叠用户稀疏的个人证据时，桥梁会“过响”。为解决这两个问题，我们提出了DivCDR，一个与模型无关的“推—拉”框架。Pull（拉）为重叠用户构建置信度门控的虚拟正样本，并优化其曝光熵增益，从而拓宽桥梁所承载的偏好范围；Push（推）则采用重叠密度引导的负采样，使非重叠用户远离重叠密集的区域。在三个数据集和五个代表性CDR骨干模型上的实验表明，DivCDR在保持或提升准确性的同时改善了推荐多样性。

    arXiv:2611.30156v1 Announce Type: new  Abstract: A common paradigm in cross-domain recommendation (CDR) transfers knowledge through users shared by source and target domains. Although this overlap bridge is effective in improving accuracy, its effect on diversity remains unclear. We identify two bottlenecks: the bridge can be too narrow when overlapping users cover limited target-domain tastes, and too loud when shared signals dominate the sparse personal evidence of non-overlapping users. To address both, we propose DivCDR, a model-agnostic Push--Pull framework. Pull constructs confidence-gated virtual positives for overlapping users and optimizes their exposure-entropy gain, broadening the preferences carried by the bridge. Push uses overlap-density-guided negative sampling to move non-overlapping users away from overlap-dense regions. Experiments on three datasets and five representative CDR backbones show that DivCDR improves recommendation diversity while preserving or improving a
    
[^11]: 像素作为共同语言：通过视觉渲染实现统一的多模态嵌入

    Pixel as the Common Language: Unified Multimodal Embedding via Visual Rendering

    [https://arxiv.org/abs/2611.30069](https://arxiv.org/abs/2611.30069)

    该论文提出RenderEmbed框架，通过视觉渲染将文本、图像及组合输入统一转换为像素形式的视觉基底后再编码，实现了真正统一的多模态嵌入表示学习。

    

    多模态嵌入模型将异构输入映射到共享的表示空间，然而纯文本、纯图像和组合输入仍然以不同的格式呈现，并经由模型中不同的路径进行处理。这种碎片化阻碍了异构任务共同优化同一端到端编码过程，限制了监督信号在不同输入配置之间的迁移。为解决这一问题，我们提出“编码前统一”的范式，并在RenderEmbed中加以实现。该框架使用视觉基底渲染将异构输入转换为通用的视觉基底，以供视觉语言模型（VLM）进行编码。基于所生成的视觉标记，渐进式层次锚点编码通过交错插入可学习锚点并跨层融合其状态，产生紧凑的检索表示，同时保留广泛的上下文信息和细粒度线索。干预式渲染学习训练这些表示对渲染扰动保持敏感……（原文摘要至此截断）

    arXiv:2611.30069v1 Announce Type: cross  Abstract: Multimodal embedding models map heterogeneous inputs into a shared representation space, yet text-only, image-only, and composed inputs are still presented in distinct formats and follow different routes through the model. This fragmentation prevents heterogeneous tasks from jointly optimizing the same end-to-end encoding process, limiting supervision transfer across input configurations. To address this issue, we propose unification before encoding and instantiate this paradigm in RenderEmbed. The framework uses Visual Substrate Rendering to convert heterogeneous inputs into a common visual substrate for encoding with a VLM. From the resulting visual tokens, Progressive Hierarchical Anchor Encoding produces compact retrieval representations by interleaving learnable anchors and fusing their states across layers, retaining broad context and fine-grained cues. Interventional Rendering Learning trains these representations to remain sens
    
[^12]: STUN-Bench：用于评估共享工具经验的真正未见工具基准

    STUN-Bench: Truly Unseen Tools for Evaluating Shared Tool Experience

    [https://arxiv.org/abs/2611.30034](https://arxiv.org/abs/2611.30034)

    该论文提出STUN-Bench基准，构建了经验池中完全不包含172个保留工具任何信息的数据分割，并揭示了共享经验会挤占工具描述排名、导致真正未见的工具从候选列表中消失的“拥挤”问题。

    

    工具路由器为每个请求检索少量候选工具，并让语言模型进行选择。跨组织共享经验的方法会将关于每个工具的历史请求添加到路由器检索的内容中。StableToolBench 将其中某些测试组称为对“未见”工具的测试，然而检索器和经验共享方法从中学习的 ToolBench 指令池，却包含了这两个单工具测试组中的每一个测试查询，并标记了其中所有的黄金工具。我们发布了 STUN-Bench（StableToolBench, Truly UNseen），这是一个新的数据分割，其经验池中不存在任何标记或命名172个保留工具的条目。这里的“未见”指的是在共享经验和检索器训练数据中缺失，而非在预训练数据中缺失。在该分割数据集上，我们发现了“拥挤”现象：从五个模拟客户端汇集的经验被添加到工具描述索引后，其排名超过了没有任何客户端使用过的工具的描述，导致这些工具经常从候选列表中被淘汰。在 Jina-v2 检索器下……

    arXiv:2611.30034v1 Announce Type: cross  Abstract: A tool router retrieves a few candidate tools per request and lets a language model choose. Methods that share experience across organizations add past requests about each tool to what the router retrieves. StableToolBench calls some of its test groups tests on "unseen" tools, yet the ToolBench instruction pool, which retrievers and experience-sharing methods learn from, contains every test query and labels every gold tool of the two single-tool groups among them. We release STUN-Bench (StableToolBench, Truly UNseen), a split whose experience pool has no item that labels or names 172 held-out tools. Unseen here means absent from shared experience and retriever training data, not from pretraining. On the split we find crowding: experience pooled from five simulated clients, added to an index of tool descriptions, outranks the descriptions of tools no client has used, which then often drop out of the candidate list. Under the Jina-v2 ret
    
[^13]: SHAKTI：基于证据的营养与健康对话式AI助手

    SHAKTI: Evidence-Grounded Conversational AI for Nutrition and Health

    [https://arxiv.org/abs/2611.29988](https://arxiv.org/abs/2611.29988)

    该论文提出了SHAKTI，一个仅基于检索到的学术文献和精确食物成分数据回答营养健康问题的对话式AI助手，通过来源链接核查和引文数字在线验证等机制确保回答有据可依，有效解决了大语言模型产生无依据论断的问题。

    

    大型语言模型（LLM）能够回答许多饮食与健康问题，但可能陈述缺乏依据的观点，并引用实际上并不支持其说法的文献来源。我们提出了SHAKTI（Smart Health Advisor for Knowledge-based Tailored Interventions，基于知识的定制化干预智能健康顾问），这是一个仅依据检索到的学术文献和精确的食物成分数据进行回答的对话助手，并采用检索、生成、依据性（grounding）、效率和上下文感知等指标对其进行评估。我们索引了484,368,086篇OpenAlex文献，在评估阶段（涵盖NFCorpus、BioASQ、SciFact和PubMedQA共10,065个问题）通过GPU进行精确检索，在部署阶段则通过一个压缩的包含2.3亿篇摘要的索引进行检索。在1,000个分层抽样的问题上，我们将单轮、多跳、纠错型和深度研究型检索增强生成（RAG）方法与Claude Sonnet进行了对比。SHAKTI增加了对话式查询改写、USDA FoodData Central食物数据查询、来源链接核查以及引文和数字的在线验证功能。（注：原摘要在此处截断）

    arXiv:2611.29988v1 Announce Type: new  Abstract: Large language models (LLMs) answer many diet and health questions but can state unsupported claims and cite sources that do not support them. We present SHAKTI (Smart Health Advisor for Knowledge-based Tailored Interventions), a conversational assistant that answers only from retrieved scholarly papers and exact food-composition data, and evaluate it with retrieval, generation, grounding, efficiency and contextawareness metrics. We indexed 484,368,086 OpenAlex works, searched exactly on a GPU for evaluation (10,065 NFCorpus, BioASQ, SciFact and PubMedQA questions) and through a compressed 230-million-abstract index in deployment. On 1,000 stratified questions, we compared single-shot, multi-hop, corrective and deep-research retrieval-augmented generation (RAG) with Claude Sonnet. SHAKTI adds conversational query rewriting, USDA FoodData Central lookups, source-link checks and online verification of quotes and numbers. It was tested in 3
    
[^14]: 面向长篇叙事文档的结构利用型层次化检索：一种以精度为导向的学习型抽象树替代方案

    Structure-Exploiting Hierarchical Retrieval for Long Narrative Documents: A Precision-Oriented Alternative to Learned Abstraction Trees

    [https://arxiv.org/abs/2611.29966](https://arxiv.org/abs/2611.29966)

    提出了一种利用文档原生章节结构的两级层次化检索方法，通过章节摘要混合匹配选择、章节过滤密集搜索和词元预算约束下的感知章节段落扩展，并辅以内容哈希门控的廉价增量索引更新，作为 RAPTOR 等学习型抽象树之外的精度导向替代方案，用于长篇叙事文档的检索增强生成。

    

    针对书籍长度文档的检索增强生成（RAG）需要在数万词元的文本中定位一小段连贯的段落。平面密集检索忽略了文档结构；而 RAPTOR 则通过递归聚类和使用语言模型进行摘要来构建层次结构。我们研究了一种利用文档原生结构的互补设计：一个两级检索器，它（i）通过对章节摘要采用 BM25+密集混合匹配来选择章节，（ii）运行经章节过滤的密集搜索，（iii）在词元预算约束下将排名靠前的段落扩展为连续的、感知章节结构的文本跨度，此外还采用内容哈希门控的增量摄取机制，以保持索引更新成本低廉。我们在五本公有领域书籍（113 个章节定位查询）以及 QuALITY 和 NovelQA（20 万+词元的小说）数据集上进行了评估。在内部测试中，我们发现了随随机种子稳定的精度/排序权衡：当关闭章节扩展时，层次化检索返回的片段中具有更高比例的……（原文摘要在此处截断）

    arXiv:2611.29966v1 Announce Type: new  Abstract: Retrieval-augmented generation over book-length documents must locate a small, coherent set of passages inside tens of thousands of tokens. Flat dense retrieval ignores document structure; RAPTOR instead builds a hierarchy by recursively clustering and summarizing with a language model. We study a complementary design that exploits the native structure of a document: a two-level retriever that (i) selects chapters with a hybrid BM25+dense match over chapter recaps, (ii) runs a chapter-filtered dense search, and (iii) expands top passages into contiguous, section-aware spans under a token budget, plus content-hash-gated delta ingestion to keep the index cheap to update. We evaluate on five public-domain books (113 chapter-localization queries) and on QuALITY and NovelQA (200K+ token novels). In-house, we find a seed-stable precision/ranking tradeoff: with section expansion off, hierarchical retrieval returns a higher fraction of chunks fr
    
[^15]: 三思而后行：使用TriVS重新架构磁盘驻留向量搜索

    Look Before You Leap: Re-Architecting Disk-Resident Vector Search with TriVS

    [https://arxiv.org/abs/2611.29948](https://arxiv.org/abs/2611.29948)

    TriVS通过以I/O为中心的分析揭示了基于聚类与基于图的搜索在I/O上的互补权衡，并创新性地将磁盘驻留向量搜索重构为“粗略接近、渐进蒸馏、精细收敛”的三部曲范式，融合两种粒度的I/O优势以突破I/O瓶颈。

    

    磁盘驻留近似最近邻搜索（ANNS）将向量搜索的规模扩展到DRAM之外，然而即使采用快速SSD和复杂的I/O执行机制，最先进的系统仍然受限于I/O瓶颈。为了理解这一瓶颈，我们进行了以I/O为中心的分析，将每个查询分解为两个阶段：定位查询邻近区域的方法阶段和求解最终top-k结果的收敛阶段。这一分析揭示了两种互补的I/O权衡：基于聚类的搜索通过内存中的路由规避了方法阶段的I/O，但在收敛阶段需要读取粗粒度的完整簇；而基于图的搜索在整个过程中采用细粒度的向量级I/O，以方法阶段不可避免的I/O开销为代价，实现了高效的收敛。为了结合这两种粒度的互补I/O优势，我们提出了TriVS，一种新的范式，将磁盘驻留向量搜索重新架构为三部曲：粗略接近、渐进蒸馏……

    arXiv:2611.29948v1 Announce Type: cross  Abstract: Disk-resident approximate nearest-neighbor search (ANNS) scales vector search beyond DRAM, yet state-of-the-art systems remain I/O-bound, even with fast SSDs and sophisticated I/O execution. To understand this bottleneck, we conduct an I/O-centric analysis that breaks each query down into an approach phase that locates a query-near region and a converge phase that resolves the final top-k. This analysis reveals complementary I/O tradeoffs: cluster-based search circumvents approach-phase I/O through in-memory routing but reads coarse-grained, full clusters during convergence, whereas graph-based search employs fine-grained, vector-level I/O throughout, enabling efficient convergence at the cost of inevitable I/O during the approach phase.   To combine the complementary I/O benefits of the two granularities, we present TriVS, a new paradigm that re-architects disk-resident vector search as a trilogy: approach coarsely, distill progressiv
    
[^16]: 事实核查错误产生于何处？在基于证据的验证中分离检索、搜索控制与判断

    Where Do Fact-Checking Errors Arise? Separating Retrieval, Search Control and Judgement in Evidence-Grounded Verification

    [https://arxiv.org/abs/2611.29697](https://arxiv.org/abs/2611.29697)

    该论文提出一套分离检索、搜索控制、判断与裁决验证的诊断协议，发现事实核查失败的主要根源不在证据检索（单次混合检索即可覆盖绝大多数声明的完整证据），而在于证据的后续使用环节——判断器弃权与验证规则推翻了本可正确作出的裁决。

    

    基于证据的事实核查器可能因找不到证据而失败，也可能因没有使用证据而失败；端到端的准确率无法区分这两种情况。我们提出了一套协议，追踪参考证据从存储库到最终裁决的完整路径，将检索、搜索控制、判断和裁决验证四个环节分离开来。我们使用一个开放权重的9B模型，在两种证据条件下评估了来自FEVER、FEVER 2.0和AVeriTeC的293条声明：一种是标注证据的小型存储库（证据发现较为容易），另一种是完整的源文档集合。在对照条件下，单次混合检索即可为228条含证据声明中的226条提供完整的参考证据，而自适应控制器和工具智能体花费了二到九倍的模型调用次数，却没有带来一致的性能提升。在34条被标记为缺乏证据的决定性声明中，有33条的证据包中实际包含完整证据：其中15条是判断器选择弃权，另外18条是验证规则推翻了决定性……

    arXiv:2611.29697v1 Announce Type: new  Abstract: An evidence-grounded fact-checker can fail because it does not find the evidence or because it does not use it; end-to-end accuracy does not tell the two apart. We present a protocol that follows the reference evidence from the store to the verdict and separates retrieval, search control, judgement and verdict validation. We assess 293 claims from FEVER, FEVER 2.0 and AVeriTeC with one open-weights 9B model under two evidence conditions: small stores of annotated evidence, where discovery is easy, and the full source collections. In the control condition a single hybrid retrieval pass delivers complete reference evidence for 226 of 228 evidence-bearing claims, and adaptive controllers and a tool agent spend two to nine times as many model calls without a consistent gain. Of 34 decisive claims labelled as lacking evidence, 33 have the complete evidence in the packet: in 15 the judge abstains, and in 18 validation rules overturn a decisive
    
[^17]: 权重编码：通过LLM智能体的直接权重编辑来演化网络规模推荐系统

    Weight Coding: Evolving Web-Scale Recommenders through Direct Weight Edits with LLM Agents

    [https://arxiv.org/abs/2611.29613](https://arxiv.org/abs/2611.29613)

    该论文提出“权重编码”新方法，利用LLM智能体通过直接、可执行的权重编辑来演化网络规模推荐模型，在Pinterest图像搜索双塔召回器上以仅约1/4的微调成本将recall@10提升高达15个百分点（相对提升24%），远超常规上线的约1%提升。

    

    网络规模的推荐模型主要通过重新训练或微调来演化。我们提出了“权重编码”，这是一种通过直接的、可执行的权重编辑来实现模型演化的方法。一个大语言模型（LLM）智能体检查模型结构和参数，提出假设，并利用评估反馈来实施编辑。这一迭代循环从局部修改逐步推进到有针对性的搜索和组合，使被编辑的组件能够相互适应。一个缓存引擎支持无需反向传播的快速前向评估。在Pinterest的1.47亿参数图像搜索双塔召回模型上，权重编码仅通过少量编辑、耗费完整训练后微调约1/4的GPU小时，就将生产模型在留出集上的recall@10提升高达15.0个百分点（相对提升24%），而常规上线通常仅能提升约1%。在随机重新初始化整个1360万参数交互模块的同时保留周围的（摘要截断）……

    arXiv:2611.29613v1 Announce Type: cross  Abstract: Web-scale recommender models evolve primarily through retraining or fine-tuning. We propose \emph{weight coding}, a methodology for model evolution through direct, executable weight edits. A large language model (LLM) agent inspects model structure and parameters, formulates hypotheses, and implements edits using evaluation feedback. The loop progresses from localized changes to targeted searches and composition, allowing edited components to adapt to one another. A cached engine enables rapid forward evaluation without backpropagation. On Pinterest's 147M-parameter Image Search two-tower retriever, weight coding improves held-out recall@10 of production model by up to 15.0 percentage points (+24\% relative) with few edits in about 1/4 GPU hours of a full post-training finetune, while a common launch only improves around 1\%. After reinitializing the entire 13.6M-parameter interaction module randomly while retaining the surrounding tra
    
[^18]: MoR4Rec：面向序列推荐的递归混合模型

    MoR4Rec: Mixture-of-Recursion for Sequential Recommendation

    [https://arxiv.org/abs/2611.29462](https://arxiv.org/abs/2611.29462)

    该论文提出MoR4Rec，一种基于递归混合范式的序列推荐框架，通过为不同复杂度的物品自适应分配计算深度，解决了传统Transformer对所有交互物品采用固定计算深度所导致的模型容量浪费与计算开销过高的问题。

    

    arXiv:2611.29462v1 公告类型：交叉 摘要：序列推荐系统旨在通过建模用户历史行为序列中的时序依赖关系来预测用户的下一次交互。为了捕捉这种依赖关系，近期的相关方法越来越依赖于基于注意力的架构，尤其是Transformer。尽管这类模型在建模长程序列模式方面十分有效，但它们对所有交互物品都施加固定的计算深度，将其视为同等复杂。然而，用户行为本质上是异构的：许多交互遵循简单、重复的模式，而另一些交互则需要建模高阶依赖关系。这种不匹配导致了模型容量的低效利用和不必要的计算开销，尤其是在基于深度Transformer的架构中。为解决这一问题，我们提出了MoR4Rec，一种基于递归混合范式的序列推荐框架，它通过共享的Transformer……（摘要截断于此）

    arXiv:2611.29462v1 Announce Type: cross  Abstract: Sequential recommender systems aim to predict a user's next interaction by modeling temporal dependencies in historical behavior sequences. To capture such dependencies, recent approaches increasingly rely on attention-based architectures, particularly transformers. While effective at modeling long-range sequential patterns, these models apply a fixed computation depth to all interaction items, treating them as equally complex. However, user behavior is inherently heterogeneous, with many interactions following simple repetitive patterns, whereas others require modeling higher-order dependencies. This mismatch leads to inefficient utilization of model capacity and unnecessary computational overhead, particularly in deep transformer-based architectures. To address this issue, we propose MoR4Rec, a sequential recommendation framework based on a Mixture-of-Recursion paradigm, which enables item-wise adaptive computation through a shared t
    
[^19]: VistaTree：基于图像导出摘要树的长篇多模态文档问答

    VistaTree: Question Answering over Long Multimodal Documents with Image-Derived Summary Trees

    [https://arxiv.org/abs/2611.29419](https://arxiv.org/abs/2611.29419)

    VistaTree通过直接从页面图像构建层次化摘要树并进行层级导航，实现了对超出上下文窗口的长篇多模态文档的高效问答。

    

    对长篇多模态文档进行问答需要在包含文本、表格和图表的页面中定位相关证据。当文档超出模型的上下文窗口时，就需要进行选择性检索。我们提出了VistaTree，这是一个文档问答智能体，它直接从渲染的页面图像构建摘要树。VistaTree将文档分割为连续的页面片段，递归地将它们组织成主题层次结构，并细化兄弟节点摘要以区分相邻章节。在查询时，该智能体使用节点摘要来选择相关节点并阅读对应页面，当完整大纲超出上下文预算时则采用层次化导航。在MMLongBench-Doc和LongDocURL数据集上的实验使用本地部署的视觉-语言模型评估了答案质量和检索成本，并量化了对索引进行全局访问与层次化访问之间的性能差距。

    arXiv:2611.29419v1 Announce Type: new  Abstract: Answering questions over long multimodal documents requires locating relevant evidence across pages containing text, tables, and figures. When the document exceeds the model's context window, selective retrieval becomes necessary. We introduce VistaTree, a document question-answering agent that builds a summary tree directly from rendered page images. VistaTree segments documents into contiguous page spans, recursively groups them into a topic hierarchy, and refines sibling summaries to distinguish neighboring sections. At query time, the agent uses node summaries to select relevant nodes and reads the corresponding pages, using hierarchical navigation when the full outline exceeds the context budget. Experiments on MMLongBench-Doc and LongDocURL evaluate answer quality and retrieval cost using a locally hosted vision-language model, and quantify the performance gap between global and hierarchical access to the index.
    
[^20]: 基于大语言模型的狗狗护理指导助手

    LLM-Based Assistant for Dog Care Guidance

    [https://arxiv.org/abs/2611.29286](https://arxiv.org/abs/2611.29286)

    研究通过对比实验发现，基于RAG检索增强的领域接地方法在提升大语言模型生成狗狗护理建议的准确性方面显著优于LoRA参数高效微调。

    

    本研究探讨了通过检索增强生成（RAG）进行领域接地以及使用低秩适应（LoRA）进行参数高效微调，能否提升大语言模型（LLM）生成的宠物护理建议的准确性和可靠性。我们精心构建了一个包含特定品种狗狗护理信息的数据集，并基于该数据集开发了两个系统：一个基于TinyLlama-1.1B-Chat-v1.0的RAG管道，以及一个使用LoRA适配的微调LLaMA-3.2-1B模型。两个系统在15个狗狗护理问题上进行了评估。RAG模型获得了2.0的满分平均分，优于微调模型（平均分=1.47）。统计分析结果，包括Welch's t检验、Mann-Whitney U检验和Wilcoxon符号秩检验（均p < 0.05），证实了该性能差距的统计显著性。我们的研究结果表明，在专业领域（宠物护理）中，基于检索的领域接地比单独进行微调能够产生更准确、更一致且更值得信赖的回答。

    arXiv:2611.29286v1 Announce Type: cross  Abstract: This study investigates whether domain grounding through Retrieval-Augmented Generation (RAG) and parameter-efficient fine-tuning using Low-Rank Adaptation (LoRA) can enhance the accuracy and reliability of pet care advice generated by large language models (LLMs). We curated a dataset of breed-specific dog care information and used it to develop two systems: a RAG pipeline based on TinyLlama-1.1B-Chat-v1.0, and a fine-tuned LLaMA-3.2-1B model adapted using LoRA. Both systems were evaluated on 15 dog care questions. The RAG model achieved a perfect average score of 2.0, outperforming the fine-tuned model (mean = 1.47). Statistical analyses, including Welch's t-test, Mann-Whitney U, and Wilcoxon signed-rank tests (all p < 0.05), confirmed the significance of the performance gap. Our findings suggest that retrieval-based domain grounding leads to more accurate, consistent, and trustworthy responses than standalone fine-tuning in speciali
    
[^21]: NoeVera：知识能否指导实验，实验又能否反哺知识？

    NoeVera: Can Knowledge Guide Experiments and Experiments Refine Knowledge?

    [https://arxiv.org/abs/2611.29255](https://arxiv.org/abs/2611.29255)

    NoeVera是一个智能体式“第二大脑”，它将论文、代码、基准与实验结果整合为保留来源、持久化的三层知识库，通过决策感知检索、角色化智能体推理与模拟器调用，构建“知识指导实验、实验反哺知识”的闭环，以优化LLM服务系统的科研决策。

    

    随着对大语言模型（LLM）服务需求的日益增长，优化LLM服务系统正变得愈发复杂，瓶颈已不再是预测给定配置的性能表现。更困难的问题在于决定要研究什么：对于某个工作负载，哪些机制至关重要、哪些配置值得测试、以及一项决策还需要什么证据。在本工作中，我们提出了NoeVera，一个超越单次检索的智能体式“第二大脑”，它将论文、代码、基准测试和实验结果转化为一个持久化且保留来源可追溯性的知识库，该知识库包含三个层面：源证据、可复用的综合知识以及智能体记忆。决策感知检索会组装紧凑的证据包，各角色的专属智能体对这些证据进行推理，并在需要定量支持时调用模拟器（Vidur或GAUSS），而每项发现都会作为带有作用域的证据记录写回知识库，从而使知识能够指导实验。

    arXiv:2611.29255v1 Announce Type: new  Abstract: With the growing demand for large language model (LLM) services, optimizing LLM serving systems is becoming increasingly complex, and the bottleneck is no longer predicting how a given configuration will perform. The harder problem is deciding what to investigate: which mechanisms matter for a workload, which configurations are worth testing, and what evidence a decision still needs. In this work, we present NoeVera, an agentic Second Brain that goes beyond just one-shot retrieval by turning papers, code, benchmarks, and experimental results into a persistent, provenance-preserving knowledge base with three planes: source evidence, reusable synthesis, and agent memory. Decision-aware retrieval assembles compact evidence packs, role-specific agents reason over them and invoke a simulator (Vidur or GAUSS) whenever quantitative support is needed, and every finding is written back as a scoped evidence record, so knowledge guides experiments 
    
[^22]: Zarc：在数秒内对十亿向量进行精确最近邻搜索

    Zarc: Exact Nearest Neighbors over a Billion Vectors in Seconds

    [https://arxiv.org/abs/2611.29252](https://arxiv.org/abs/2611.29252)

    Zarc通过在GPU内存中仅保留短小的距离证明、并从主机内存按需取回少量完整向量，实现了在多GPU上对十亿级向量在数秒内完成精确k近邻搜索，将10,000个查询的耗时从五分钟降至1.1秒。

    

    精确k近邻向量搜索处于机器学习系统的关键路径上，应用涵盖最新大型语言模型（LLM）中的稀疏注意力、长上下文记忆以及训练数据去重。穷举式GPU搜索会为每个查询扫描整个数据集，而近似索引则返回不精确的邻居。近期的方法利用一种廉价的半精度证明跳过几乎所有距离过远的向量，但该方法只能在持有整个数据集的单块GPU上运行。我们的核心洞察是：仅在GPU内存中保留这些证明的短行，并从主机内存中取回少数剩余的完整向量。我们的系统Zarc能够在数据更新情况下精确回答欧几里得距离、余弦相似度、内积以及带过滤条件的查询。在四块H100 GPU上，Zarc对十亿向量回答10,000个精确查询仅需1.1秒，而传统方法需要五分钟。在十一个开源LLM上，它检索精确top-10注意力键的速度比穷举搜索快2.2倍至13.1倍。

    arXiv:2611.29252v1 Announce Type: cross  Abstract: Exact k-nearest-neighbor vector search lies on the critical path of machine learning systems, from sparse attention in the latest large language models (LLMs) to long-context memories and training-data deduplication. Exhaustive GPU search scans the whole dataset for every query, while approximate indexes return inexact neighbors. Recent methods skip almost every vector with a cheap half-precision proof that it lies too far away, but only on one GPU that holds the whole dataset. Our insight is to keep only the short rows of these proofs in GPU memory and to fetch the few remaining full vectors from host memory. Our system, Zarc, answers Euclidean, cosine, inner-product, and filtered queries exactly under updates. On four H100 GPUs, it answers 10,000 exact queries over one billion vectors in 1.1 seconds instead of five minutes. On eleven open LLMs, it retrieves the exact top-10 attention keys 2.2x to 13.1x faster than exhaustive search. 
    
[^23]: DataRecycle：从零散网络证据中可信地蒸馏结构化数据

    DataRecycle: Trustworthy Distillation of Structured Data from Scattered Web Evidence

    [https://arxiv.org/abs/2611.29206](https://arxiv.org/abs/2611.29206)

    DataRecycle是一个交互式系统，通过将用户查询分解为实体与属性并组织候选证据，把网络上零散且可能矛盾的信息可信地转化为可比较、可追溯的结构化数值事实表格。

    

    网络中包含大量零散的量化证据，这些证据难以被组织成可比较、可验证的结构。然而，其非结构化、稀疏且有时相互矛盾的特性，使得系统性的信息蒸馏成本高昂且容易出错。现有方法迫使用户在两者之间进行权衡：搜索引擎返回的文档列表需要耗费大量人力进行提取与核对，而生成式AI系统产生的流畅摘要则会掩盖数据来源、引入报告偏差并隐藏不一致之处。我们提出了DataRecycle，这是一个交互式系统，能够帮助将零散的网络证据转化为包含可比较数值事实的结构化表格。给定用户的查询或陈述，DataRecycle会将其分解为实体和属性，检索候选证据，并在基于表格的证据视图中组织零散且可能相互矛盾的记录。为解决自动化数据蒸馏中的关键挑战……（原文摘要在此处截断）

    arXiv:2611.29206v1 Announce Type: cross  Abstract: The web contains vast amounts of scattered quantitative evidence that is difficult to organize into comparable and verifiable structures. Yet its unstructured, sparse, and sometimes conflicting nature makes a systematic distillation costly and error-prone. Existing approaches force users to make a trade-off: search engines return document lists that require labor-intensive extraction and reconciliation, while generative AI systems produce fluent summaries that obscure data provenance, introduce reporting bias, and hide inconsistencies. We present DataRecycle, an interactive system that helps transform scattered web evidence into structured tables of comparable numerical facts. Given a user's query or statement, DataRecycle decomposes it into entities and attributes, retrieves candidate evidence, and organizes scattered, potentially conflicting records in a table-based evidence view. To address key challenges in automated data distillat
    
[^24]: 用语言弥补盲区：通过语言受限通信实现去中心化多视角空间推理

    Bridging Blind Spots With Words: Decentralized Multi-View Spatial Reasoning via Language-Constrained Communication

    [https://arxiv.org/abs/2611.29112](https://arxiv.org/abs/2611.29112)

    该论文首次形式化了去中心化多视角空间推理这一通用问题（N个智能体各自观察不同视角并通过受限自然语言消息协作），并提出了新的多视角空间基准，研究发现用预算受限的自然语言进行空间推理整合的代价远小于预期。

    

    随着多模态大语言模型（MLLM）越来越多地被用作具身机器人、AR/VR和多用户系统中的协作伙伴，空间推理现在往往分布在仅具有部分环境可见性的多个智能体之间，因此需要空间推理的整合。然而，以往在语言介导的空间整合方面的研究严格局限于简化的双智能体场景。为了填补这一空白，我们首先形式化了去中心化多视角空间推理的一般问题，其中N个智能体从各自不同的视角进行观察，交换受预算限制的自然语言消息，并相互协作使监听者能够回答空间查询。随后，我们引入了一个新颖的多视角空间基准，以筛选出那些需要非平凡多视角整合的查询。通过广泛的实证研究，我们发现将去中心化空间推理限制在预算有限的自然语言通信下所产生的性能损失远小于……

    arXiv:2611.29112v1 Announce Type: cross  Abstract: As multimodal LLMs (MLLMs) get increasingly used as collaborative partners across embodied robotics, AR/VR, and multi-user systems, spatial reasoning is now often distributed across agents with partial environment visibility, necessitating spatial reasoning integration. Prior work in language-mediated spatial integration is however strictly limited to simplified two-agent scenarios. To address this gap, we first formalize the general problem of decentralized multi-view spatial reasoning, where N agents observe distinct viewpoints, exchange budget-constrained natural language messages, and collaborate to enable a listener to answer spatial queries. We then introduce a novel multi-view spatial benchmark to isolate queries that demand non-trivial multi-view integration. Through an extensive empirical study, we discover that the penalty of restricting decentralized spatial reasoning to natural language with limited budget is far smaller th
    
[^25]: HME-PFD：异构专家混合特权特征蒸馏

    HME-PFD: Heterogeneous Mixture-of-Experts Privileged Feature Distillation

    [https://arxiv.org/abs/2611.29094](https://arxiv.org/abs/2611.29094)

    提出HME-PFD方法，通过在专家混合学生模型的路由模块中引入硬编码与可学习两种路由机制来融合特权特征，从而动态应对特权特征覆盖率有限且分布异构的问题，提升广告推荐场景下的蒸馏效果。

    

    知识蒸馏（KD）被广泛用于将大型教师模型的知识迁移到较小的学生模型中，从而促进机器学习系统的高效服务。在广告推荐场景中，特权特征蒸馏（PFD）扩展了知识蒸馏，允许教师模型利用特权特征（即仅在训练阶段可用的特征）来提升蒸馏性能。然而，在现实场景中，特权特征往往覆盖率有限且呈现异构分布，导致不同训练样本对蒸馏和模型学习的重要性各不相同。我们提出了异构专家混合特权特征蒸馏（HME-PFD），这是一种新颖的方法，将特权特征融入到基于专家混合（Mixture-of-Experts）的学生模型的路由模块中。我们的方法同时引入了硬编码和可学习的路由模块，使学生模型能够动态确定……（摘要在此处截断）

    arXiv:2611.29094v1 Announce Type: new  Abstract: Knowledge distillation (KD) is widely used to transfer knowledge from large teacher models to smaller student models, facilitating efficient serving of machine learning systems. In the context of ads recommendation, privileged features distillation (PFD) extends KD by allowing the teacher model to leverage privileged features, i.e., features available only during training, to enhance distillation performance. However, in real-world scenarios, privileged features often have limited coverage and exhibit heterogeneous distributions, resulting in training examples of varying importance for distillation and model learning. We propose Heterogeneous Mixture-of-Experts Privileged Feature Distillation (HME-PFD), a novel approach that incorporates privileged features into the routing module of a Mixture-of-Experts-based student model. Our method introduces both hard-coded and learnable routing modules, enabling the student to dynamically determine
    
[^26]: 当临床模型获得新数据时，谁会被重新排序？

    Who Gets Re-ranked When a Clinical Model Gains New Data?

    [https://arxiv.org/abs/2611.28968](https://arxiv.org/abs/2611.28968)

    该论文研究了医院临床风险模型添加新数据源（如实验室结果）时缺乏新数据的患者如何被重新排序，发现实验室结果虽能改善预测，却会使未经检测的危重患者在两家医院的不同模型中都下降到警报名单的更低位置。

    

    医院通常会通过添加一个数据源（例如只有部分患者才有的实验室检测结果）来更新已部署的风险模型。关于安全模型更新的现有研究假设输入是固定的，因此忽略了缺乏新数据的患者。模型更新对这些患者没有新的测量值，但他们的排名仍可能发生变化，一位尚未接受检测的危重患者可能会从警报名单中掉落。我们按数据可用性拆分了基于排名的兼容性，并将其与误差校正和警报名单效应相结合。我们在来自 MC-MED 的 78,475 次急诊科就诊数据上，以及在第二家医院 MIMIC-IV-ED 的 100,000 次住院记录的预注册重复实验中，将每种更新方式与在旧输入上进行普通重训练进行了比较，任务为预测 ICU 入院或死亡以及医院入院。结果显示，放射报告文本没有带来可检测的增益，也几乎没有引起额外的重新排序；实验室结果改善了预测，但在两家医院、无论采用线性模型还是树模型的情况下，都把没有实验室结果的患者推向了警报名单的更下方（注：原文摘要在此处截断）。

    arXiv:2611.28968v1 Announce Type: cross  Abstract: Hospitals often update a deployed risk model by adding a data source, such as laboratory results, that only some patients have. Work on safe model updates assumes fixed inputs and so ignores patients who lack the new data. The update has no new measurements for them, yet their rank can change, and a sick patient not yet tested can drop off an alert list. We split rank-based compatibility by data availability and pair it with error correction and alert-list effects. We compared each update with ordinary retraining on the old inputs, predicting ICU admission or death and hospital admission, on 78,475 emergency department visits from MC-MED and, in a pre-registered replication, 100,000 stays from MIMIC-IV-ED at a second hospital. Radiograph report text gave no detectable gain and almost no extra re-ranking. Laboratory results improved prediction but, at both hospitals and with linear or tree models, pushed patients without labs down alert
    
[^27]: 《ChatGPT无法销售（而这或许是件好事）：面向买家的AI建议、销售代表差距及来自商业对话的证据》

    ChatGPT Can't Sell (And That's Probably a Good Thing): Buyer-Aligned AI Advice, the Sales-Representative Gap, and Evidence from Commercial Conversations

    [https://arxiv.org/abs/2611.28946](https://arxiv.org/abs/2611.28946)

    该研究通过三类大规模商业对话证据表明，ChatGPT等AI助手在购买情境中只提供推荐和购买资格甄别建议，而不会执行真正的交易促进行动，作者认为这种与买家立场一致、无法代商家完成销售的特性可能是件好事。

    

    AI助手和商家销售代理有着不同的目标：助手应当帮助买家比较各种选项，并保留不购买的选择权；而销售代理还会引导符合条件的买家迈向经授权的下一步行动。我们通过三个证据流来研究这种“销售代表差距”。在12,178条经过商业或交易意图筛选的历史英文GPT系列对话中，一个279个请求的样本里有63条首次回复（22.6%）包含推荐语言，且没有任何狭义的交易行动措辞。在一个包含56段对话的明确购买样本中，24条回复（42.9%）包含推荐标记，32条（57.1%）包含资格甄别短语，而没有一条包含交易特定短语。这些词汇层面的度量并不能估算实际的销售成功率。一项2026年品牌监测队列研究对1,050对GPT-5.4与Gemini的回答在383个购买相关提示上进行了分析，结果显示：在措辞为推荐的提示中，交易特定短语的出现率为0.0%，在……（原文摘要在此处截断）

    arXiv:2611.28946v1 Announce Type: new  Abstract: AI assistants and merchant sales agents have different objectives: assistants should help buyers compare options and preserve the choice not to purchase, while sales agents also guide qualified buyers toward an authorized next step. We study this sales-representative gap with three evidence streams. In 12,178 historical English GPT-family conversations screened for commercial or transactional intent, a 279-request sample contained recommendation language in 63 first replies (22.6%) and no narrow transactional action wording. In a 56-conversation explicit-buying sample, 24 replies (42.9%) contained recommendation markers, 32 (57.1%) qualification phrases, and none transaction-specific phrases. These lexical measures do not estimate successful selling. A 2026 brand-monitoring cohort of 1,050 GPT-5.4--Gemini answer pairs over 383 purchase-related prompts showed transaction-specific phrases in 0.0% of recommendation-worded prompts, 0.21% of 
    
[^28]: PROSPER：将大语言模型用作第一阶段商品搜索的稀疏检索器

    PROSPER: LLMs as Sparse Retrievers for First-Stage Product Search

    [https://arxiv.org/abs/2611.28645](https://arxiv.org/abs/2611.28645)

    提出PROSPER框架，将大语言模型用作商品搜索第一阶段的稀疏检索器，在利用其语义扩展能力的同时显式保持品牌名、型号等字面信号的词汇保真度。

    

    商品搜索是大规模电子商务平台的关键组成部分，其第一阶段检索需要在确保高效在线部署的同时实现高召回率。稀疏检索因其可解释性和可部署性在该场景下仍极具吸引力，但它面临严重的词汇不匹配挑战，尤其是对于短查询和商品标题。大语言模型（LLM）具备强大的语义能力，但将其直接应用于商品搜索的稀疏检索会带来新的挑战：（i）语义扩展可能扭曲品牌名称和型号标识符等关键的字面信号；（ii）在LLM所诱导的高维词表空间中学习有意义的稀疏表示十分困难。为应对这些挑战，我们提出了PROSPER，一个将LLM用作商品搜索稀疏检索器、同时显式保持词汇保真度的框架。PROSPER 结合了（i）一个……（原文摘要在此处被截断）

    arXiv:2611.28645v1 Announce Type: new  Abstract: Product search is a crucial component of large-scale e-commerce platforms, where first-stage retrieval should achieve high recall while ensuring efficient online deployment. Sparse retrieval remains attractive for this setting due to its interpretability and deployability, yet it suffers from severe vocabulary mismatch challenges, especially for short queries and product titles. Large language models (LLMs) offer strong semantic capabilities, but directly applying them to sparse retrieval in product search introduces new challenges: (i) semantic expansion may distort critical literal signals such as brand names and model identifiers; and (ii) learning meaningful sparse representations is difficult in the high-dimensional vocabulary space induced by LLMs.To address these challenges, we propose PROSPER, a framework that uses LLMs as sparse retrievers for product search while explicitly preserving lexical fidelity. PROSPER combines (i) a li
    
[^29]: MADRec：面向推荐的基于证据支撑辩论的可靠多智能体推理框架

    MADRec: Reliable Multi-Agent Reasoning with Evidence-Grounded Debate for Recommendation

    [https://arxiv.org/abs/2611.28604](https://arxiv.org/abs/2611.28604)

    提出了一种以证据为支撑的多智能体辩论推荐框架MADRec，通过评估协作过程中中间推理的可靠性来防止错误判断在智能体间传播，从而提升多智能体推荐推理的可靠性和推荐质量。

    

    大语言模型正在将推荐系统从传统的偏好匹配转变为以推理驱动且可解释的决策模式。多智能体协作通过将推荐推理分配给专门化的智能体，进一步推动了这一范式的发展。然而，现有框架很少评估协作过程中中间推理的可靠性，导致错误判断在智能体之间传播，最终降低推荐质量。为了解决这一问题，我们提出了基于多智能体辩论的推荐系统MADRec，这是一个以证据为支撑的多智能体框架，旨在增强多智能体推荐推理的可靠性。MADRec由三个协同工作的模块组成。首先，多智能体辩论（MAD）模块使交互智能体和语义智能体能够从互补的角度分析交互信号和内容层面的语义信号，产生多样的……

    arXiv:2611.28604v1 Announce Type: new  Abstract: Large Language Models (LLMs) are transforming recommender systems from conventional preference matching toward reasoning-driven and explainable decision-making. Multi-agent collaboration further advances this paradigm by distributing recommendation reasoning across specialized agents. However, existing frameworks rarely assess the reliability of intermediate reasoning during collaboration, allowing erroneous judgments to propagate across agents and ultimately degrade recommendation quality. To address this issue, we propose Multi-Agent Debate-based Recommendation (MADRec), an evidence-grounded multi-agent framework that enhances the reliability of multi-agent recommendation reasoning. MADRec consists of three coordinated modules. First, the Multi-Agent Debate (MAD) module enables an Interaction Agent and a Semantic Agent to analyze interaction signals and content-level semantic signals from complementary perspectives, generating diverse 
    
[^30]: abc索引：面向学习型稀疏表示的精确检索

    The abc index: Exact Retrieval for Learned Sparse Representations

    [https://arxiv.org/abs/2611.28555](https://arxiv.org/abs/2611.28555)

    abc索引仅用标准IR技术（TAAT评估、Block-Max MaxScore、二进制压缩和文档ID重排序）构建，在MS MARCO上成为各种k值下最快的精确LSR检索索引，不仅比现有精确索引快数倍到数十倍且体积更小，还能在大k值时反超近似索引并节省1.5-3.6倍空间。

    

    在学习型稀疏检索（LSR）中，查询包含数十个带有学习权重的词项，人们普遍认为经典的动态剪枝技术在此会失效，而Seismic和HNSW等近似索引被认为比精确索引更快。我们用abc索引挑战这一观点——这是一种仅由标准信息检索技术构建的精确索引：逐词项（TAAT）评估、Block-Max MaxScore、二进制压缩编码和文档ID重排序。这些技术被以一种全新的设计组合起来，并针对现代CPU进行了工程优化。在用四种LSR模型编码的MS MARCO数据集上，abc索引在任何k值下都是最快的精确索引：比Qdrant的精确检索快6.5-84倍，比Lucene快13-55倍，比BMP快2.9-18倍且索引体积缩小5-7倍。与近似索引相比，它的空间占用少1.5-3.6倍，且其延迟随k值增长要缓慢得多：在k=10时近似索引最多可快4.3倍，但在k=1000时abc反超它们，最多快7.4倍。精确性还带来一个额外的好处：没有……（原文摘要在此处截断）

    arXiv:2611.28555v1 Announce Type: new  Abstract: In learned sparse retrieval (LSR), queries have tens of terms with learned weights, on which classic dynamic pruning is believed to fail, and approximate indexes such as Seismic and HNSW are considered faster than exact ones. We challenge this view with abc, an exact index built only from standard IR techniques: term-at-a-time (TAAT) evaluation, Block-Max MaxScore, binary packing and doc-id reordering. They are combined in a new design and engineered for modern CPUs. On MS MARCO encoded with four LSR models, the abc index is the fastest exact index at every k: 6.5-84x faster than the exact search of Qdrant, 13-55x faster than that of Lucene, and 2.9-18x faster than BMP with an index 5-7x smaller. Against approximate indexes, it takes 1.5-3.6x less space, and its latency grows much more slowly with k: they are up to 4.3x faster at k=10, but at k=1000 it overtakes them, by up to 7.4x. Exactness comes with an extra benefit: there are no par
    
[^31]: 超越相关性：多模态大语言模型用于多样性感知推荐重排序的实证研究

    Beyond Relevance: An Empirical Study of Multimodal LLMs for Diversity-Aware Recommendation Reranking

    [https://arxiv.org/abs/2611.28379](https://arxiv.org/abs/2611.28379)

    本文提出一个无需训练的多样性感知多模态重排序框架，通过整合物品图像、文本元数据与用户交互历史，实证表明多模态大语言模型能够同时提升推荐的相关性与多样性，且视觉信息带来的增益具有明显的领域依赖性。

    

    大语言模型（LLM）已成为推荐系统中颇具前景的重排序器，然而大多数现有方法依赖于文本形式的物品描述，并主要关注相关性。在以视觉为导向的领域（如视频流媒体）中，物品图像提供了互补信号，这些信号也可能有助于提升推荐多样性。这引出了两个问题：视觉信息何时能提升推荐质量？多模态大语言模型（MLLM）能否同时增强相关性与多样性？我们通过一个无需训练的、多样性感知的多模态重排序框架来研究这些问题，该框架联合融入了物品图像、文本元数据和用户交互历史。我们在三个公开的亚马逊评论数据集和一个工业界的亚马逊Prime Video数据集上，使用开源与专有的多模态大语言模型开展了实证研究。我们的结果揭示了显著的领域依赖性效应：加入视觉信息最高可带来121.9%（原文此处截断）。

    arXiv:2611.28379v1 Announce Type: new  Abstract: Large language models (LLMs) have emerged as promising rerankers for recommendation, yet most existing approaches rely on textual item descriptions and primarily focus on relevance. In visually oriented domains such as video streaming, item images provide complementary signals that may also inform recommendation diversity. This raises two questions: when does visual information improve recommendation quality, and can multimodal LLMs (MLLMs) enhance relevance and diversity simultaneously? We investigate these questions through a training-free, diversity-aware multimodal reranking framework that jointly incorporates item images, textual metadata, and user interaction histories. We conduct an empirical study across three public Amazon Review datasets and an industrial Amazon Prime Video dataset, using both open-source and proprietary MLLMs. Our results reveal substantial domain-dependent effects: adding visual information yields up to 121.9
    
[^32]: GERF：一种面向模式引导科学信息抽取的令牌高效多模态检索框架

    GERF: A Token-Efficient Multimodal Retrieval Framework for Schema-Guided Scientific Information Extraction

    [https://arxiv.org/abs/2611.28295](https://arxiv.org/abs/2611.28295)

    提出了GERF框架，将论文表示为连接模式字段、文本块与插图的异构证据图，实现令牌高效、模式引导的多模态科学信息抽取，显著减少无关证据并降低文本与视觉token成本。

    

    科学信息抽取（IE）是“AI for Science”的基础组成部分，它将科学文献中蕴含的实验知识转化为结构化、可追溯的数据，用于证据综合、科学数据库构建和数据驱动的科学发现。然而，关键的实验细节往往分散在文本、表格、插图、图注及周围的讨论中，仅依赖文本的传统抽取方法难以完整恢复这些信息。现有的基于大语言模型（LLM）和多模态RAG的流程通常依赖全文上下文或粗糙的页面级检索，引入大量无关证据，并带来高昂的文本与视觉token成本。我们提出了GERF，一种基于图的证据检索框架，用于令牌高效、模式引导的多模态科学信息抽取。GERF将每篇文章表示为一张异构证据图，将模式字段、文本块、插图及由插图衍生的内容连接起来，从而……

    arXiv:2611.28295v1 Announce Type: new  Abstract: Scientific information extraction (IE) is a fundamental component of AI for Science, transforming experimental knowledge embedded in scientific literature into structured and traceable data for evidence synthesis, scientific database construction, and data-driven discovery. However, critical experimental details are often distributed across text, tables, figures, captions, and surrounding discussion, making them difficult to recover using conventional text-only extraction. Existing LLM-based and multimodal RAG pipelines typically rely on full-document context or coarse page-level retrieval, introducing substantial irrelevant evidence and high textual and visual token costs. We propose GERF, a Graph-based Evidence Retrieval Framework for token-efficient, schema-guided multimodal scientific IE. GERF represents each article as a heterogeneous evidence graph that connects schema fields, text chunks, figures, and figure-derived content throug
    
[^33]: DM-Lab参加TREC 2026 RAG评测：多查询检索、集成评判与声明包生成

    DM-Lab at TREC 2026 RAG: Multi-Query Retrieval, Ensemble Judging, and Claim-Pack Generation

    [https://arxiv.org/abs/2611.28233](https://arxiv.org/abs/2611.28233)

    DM-Lab-RAG-2026框架通过多查询BM25检索、倒数排名融合、集成相关性评判以及声明优先的引用验证生成策略，在5.53亿文档语料库上实现了可靠的检索增强问答，并在TREC 2026 RAG官方评测中取得0.910的人工要点增益。

    

    我们提出DM-Lab-RAG-2026，这是一个为TREC 2026检索增强生成（RAG）赛道开发的框架。该框架在包含5.53亿篇文档的ClimbMix-400B语料库上集成了大规模证据检索与基于引用的答案生成。其检索架构结合了多查询BM25、倒数排名融合、集成相关性评估、列表式重排序以及自适应检索深度。生成管线采用“声明优先”策略，提取与证据关联的原子声明、构建带引用的回答，并验证来源支持情况。在22个开发主题上，检索系统取得了平均0.604的nDCG@30。在官方评测中，该系统获得了0.455的集合精确率和0.910的人工要点增益。额外的实验考察了重排序、证据选择和引用验证对系统性能的影响。评测结果展示了多样化词法检索……的潜力。

    arXiv:2611.28233v1 Announce Type: new  Abstract: We present DM-Lab-RAG-2026, a framework developed for the TREC 2026 Retrieval-Augmented Generation (RAG) track. The framework integrates large-scale evidence retrieval and citation-grounded answer generation over the 553-million-document ClimbMix-400B corpus. The retrieval architecture combines multi-query BM25, reciprocal rank fusion, ensemble relevance assessment, listwise reranking, and adaptive retrieval depth. The generation pipeline employs a claim-first strategy to extract evidence-linked atomic claims, construct cited responses, and verify source support. Across 22 development topics, the retrieval system achieved a mean nDCG@30 of 0.604. In the official evaluation, it attained a set precision of 0.455 and a manual nugget gain of 0.910. Additional experiments examine the influence of reranking, evidence selection, and citation verification on system performance. The evaluation demonstrates the potential of diversified lexical ret
    
[^34]: 在面向AI智能体的DolphinBench上评估Memanto

    Evaluating Memanto on DolphinBench for AI Agents

    [https://arxiv.org/abs/2611.28231](https://arxiv.org/abs/2611.28231)

    Memanto记忆智能体凭借信息论相关性分数对记忆排序，在DolphinBench职场记忆基准上以更少的迭代次数（最多6次）、更低的成本（10.86美元）和更短的延迟（15.32秒中位数）取得83.83%的任务完成率，显著超越此前最强的Mem0配置（70.67%）。

    

    基于问答的记忆基准测试会明确告诉智能体需要回忆哪个事实。而职场请求往往让这种需求保持隐含：例如要在团队频道发布更新，智能体必须从早先的交互中推断出应使用哪个频道，并将其用于工具调用中。我们在DolphinBench上评估了Memanto——一种使用信息论相关性分数对记忆进行排序的记忆智能体。DolphinBench通过600个模拟职场任务来测试记忆能力，每个任务都经过验证确实需要记忆。Memanto在Hermes智能体框架中运行，使用GPT-6-Luna模型，每次对话最多6次迭代，完成了503个任务（83.83%）。该基准测试作者报告的最强配置——使用GPT-5.6-Luna且最多60次迭代的Mem0——完成了424个任务（70.67%）。Memanto的运行还具有最低的总成本（10.86美元，而已报告的最便宜运行为61.48美元）和最短的中位任务延迟（15.32秒）。评分使用该基准测试自带的评判器，但该智能体（原文摘要在此处截断）

    arXiv:2611.28231v1 Announce Type: cross  Abstract: Memory benchmarks built on question answering tell an agent which fact to recall. Workplace requests often leave that need implicit: to post an update in the team's channel, an agent must work out which channel from earlier interactions and use it in a tool call. We evaluate Memanto, a memory agent that ranks memories with an information-theoretic relevance score, on DolphinBench, which tests memory through 600 simulated workplace tasks, each validated as requiring memory. Running in the Hermes agent harness with GPT-6-Luna and at most 6 iterations per conversation, Memanto completes 503 tasks (83.83%). The strongest configuration reported by the benchmark's authors, Mem0 with GPT-5.6-Luna and up to 60 iterations, completes 424 (70.67%). Memanto's run also has the lowest total cost ($10.86, against $61.48 for the cheapest reported run) and the shortest median task latency (15.32s). Grading uses the benchmark's own judge, but the agent 
    
[^35]: 弥合可检索性差距的闭环

    Close the Loop on the Retrievability Gap

    [https://arxiv.org/abs/2611.28146](https://arxiv.org/abs/2611.28146)

    本文揭示了全色锐化领域中保真度指标与下游检索性能之间的“可检索性差距”，并提出LEDGER方法，通过测试时无需标签的集合级软Jaccard损失围绕冻结视觉-语言编码器构建闭环，使每个概念都获得梯度贡献，从而弥合这一差距。

    

    冻结的视觉-语言编码器如今既用于指导图像恢复，又在事后对其生成的结果进行索引。在单一WorldView-III测试集上对固定的十八种全色锐化系统进行评估时，衡量保真度的指标——有参考的$Q2^n$和无参考的HQNR——有三分之一的概率、甚至二分之一的概率会选择两个系统中可检索性较差的那一个；这种分歧主要存在于像素监督方法族与语言条件方法族之间，而且即使是最佳系统也未能达到参考水平，小目标首当其冲。我们将这一现象称为“可检索性差距”，并将其根源追溯到位置平均损失中的面积加权问题。LEDGER围绕编码器构建了闭环：它将输入索引为一组命名概念，进行图像生成，随后仅使用同一个冻结编码器对输出进行重新编码，并在该概念集合的覆盖率上最小化集合级软Jaccard损失——这是一个在测试时即可获得、无需标签的优化目标，从而确保每个尺度上出现的每个概念都能获得相应的梯度份额。

    arXiv:2611.28146v1 Announce Type: cross  Abstract: A frozen vision-language encoder now guides image restoration and later indexes what it produced. Over a fixed set of eighteen pansharpening systems on one WorldView-III test set, the fidelity index that ranks them, $Q2^n$ with a reference and HQNR without one, picks the less retrievable of two systems one time in three and one in two; the disagreement lies between the pixel-supervised and the language-conditioned family, and even the best system falls short of the reference, small objects first. We call this the retrievability gap and trace it to the area weighting of position-averaged losses. LEDGER closes the loop around the encoder: it indexes the input into a set of named concepts, generates, re-encodes the output alone with the same frozen encoder, and minimises a set-level soft-Jaccard loss on the coverage of that set, a label-free target available at test time, so that every present concept of a scale receives a gradient share 
    
[^36]: 面向小系统A/B试验的蒙特卡洛功效分析：检测新闻通讯参与度的差异

    Monte Carlo Power Analysis for Small-System A/B Trials: Detecting Differences in Newsletter Engagement

    [https://arxiv.org/abs/2611.28018](https://arxiv.org/abs/2611.28018)

    本文提出一种蒙特卡洛方法，用于估计旨在提升用户对个性化新闻邮件通讯参与度的A/B实验的统计功效。

    

    良好的实验实践以及预测实验或实验平台的科学价值都需要功效分析：即在实验前估计当预期效果确实存在时，实验能够检测到该效果的可能性。虽然简单的实验设计可以使用已被充分理解的功率分析方法，但更复杂的实验设计和实验环境往往需要定制的技术，以避免高估或低估实验功效。我们提出了用于估计实验功效的蒙特卡洛方法，该实验旨在提高用户对个性化推荐新闻文章电子邮件通讯的参与度。

    arXiv:2611.28018v1 Announce Type: cross  Abstract: Good experimental practice and predicting the scientific usefulness of an experiment or experimental platform require power analysis: estimating, *a priori*, how likely the experiment is to detect the intended effect if indeed it exists. While simple experimental designs admit well-understood power analysis methods, more sophisticated experimental designs and settings often require bespoke techniques to avoid either over- or under-estimating experimental power. We present the Monte Carlo method we use to estimate the power of experiments intended to increase user engagement with personalized e-mail newsletters of recommended news articles.
    
[^37]: CACO：面向连续生成式检索的覆盖感知协同优化

    CACO: Coverage-Aware Coordinated Optimization for Continuous Generative Retrieval

    [https://arxiv.org/abs/2611.25738](https://arxiv.org/abs/2611.25738)

    CACO是一种轻量级自监督后训练框架，通过覆盖感知的无标签目标将多个采样查询协调至互补的物品覆盖，在几乎不增加推理延迟的情况下显著提升连续生成式检索的召回率。

    

    基于流匹配和扩散模型的连续生成式检索器在物品嵌入空间中采样多个查询，以在固定检索预算下进行并行的近似最近邻（ANN）搜索。然而，优化分布多样性或目标层面的兴趣分离忽略了集合层面的检索几何结构：采样出的查询坍缩到相互重叠的ANN邻域中，而离散的top-K查找和单目标监督又阻碍了端到端的覆盖优化。我们提出了CACO（覆盖感知协同优化），这是一个轻量级的自监督后训练框架，它将来自冻结的流匹配和扩散检索器的查询协调至互补的物品覆盖，且额外推理延迟几乎可以忽略。CACO利用跨采样参考查询的经验检索频率作为相关性代理，构建了一个无标签的覆盖目标，在所有30个骨干模型-数据集设置中都提升了召回率。

    arXiv:2611.25738v1 Announce Type: new  Abstract: Continuous generative retrievers powered by flow matching and diffusion sample multiple queries in item embedding space for parallel approximate nearest-neighbor (ANN) search under fixed retrieval budgets. However, optimizing distribution diversity or target-level interest separation ignores set-level retrieval geometry: sampled queries collapse onto overlapping ANN neighborhoods, while discrete top-K lookup and single-target supervision block end-to-end coverage optimization. We introduce CACO (Coverage-Aware Coordinated Optimization), a lightweight self-supervised post-training framework that coordinates queries from frozen flow-matching and diffusion retrievers toward complementary item coverage with negligible additional inference latency. CACO uses empirical retrieval frequencies across sampled reference queries as a relevance proxy, formulating a label-free coverage objective that improves recall across all 30 backbone-dataset sett
    
[^38]: KnowMe 论文雷达：自动科研时代的轻量级潜力论文发现工具

    KnowMe Paper Radar: Lightweight Discovery of Promising Papers in the Age of Autoresearch

    [https://arxiv.org/abs/2611.23702](https://arxiv.org/abs/2611.23702)

    KnowMe Paper Radar 是一个无需 LLM 推理的轻量级论文发现工具，它综合主题相关性、时效性、作者研究历史、合作路径和机构背景等多源证据生成个性化阅读优先级，帮助研究者快速筛选潜力论文，并将重复上下文准备速度提升三倍以上。

    

    主题相似的论文数量不断增长，使得人们难以从中识别出能够提供有用研究见解或方法指导的新工作。我们提出了 KnowMe Paper Radar，这是一个轻量级的论文发现工具，它将多种证据来源综合成个性化的阅读优先级。用户可以指定主题、优先级、排除项和日期；系统会综合考量主题相关性、时效性、作者研究历史、合作路径以及特定研究方向上的机构背景。通过公开每个评分组成部分及其支持来源，KnowMe 帮助读者为其当前的研究问题挑选有前景的论文，并发现熟悉作者圈之外的相关工作。自动获取、版本合并和图缓存重用使这一工作流程保持便捷，在常规使用中无需大语言模型（LLM）推理。在 174,640 条获取记录上的实验显示，重复上下文准备的速度提升了三倍以上，同时……

    arXiv:2611.23702v1 Announce Type: new  Abstract: The growing volume of topically similar papers makes it difficult to identify new work that can offer useful research insights or methodological guidance. We present KnowMe Paper Radar, a lightweight discovery tool that combines multiple sources of evidence into personalized reading priorities. Users specify topics, priorities, exclusions and dates; the system brings together topical relevance, recency, author research histories, collaboration paths and direction-specific institutional context. By exposing each score component and its supporting sources, KnowMe helps readers select promising papers for their current questions and discover relevant work beyond familiar authors. Automatic acquisition, version merging and graph-cache reuse keep this workflow convenient without large language model (LLM) inference during ordinary use. Experiments on 174,640 acquired records show more than threefold faster repeated context preparation, while 
    
[^39]: IntentFlow：小红书中基于意图驱动的图文查询实时多模态搜索系统

    IntentFlow: An Intent-Driven Real-Time Multimodal Search System with Image-Text Queries at Xiaohongshu

    [https://arxiv.org/abs/2611.20448](https://arxiv.org/abs/2611.20448)

    IntentFlow是小红书部署的意图驱动实时多模态搜索系统，通过层次化意图分类体系和意图特定的相关性标准，在百亿级索引上实现了图文查询场景下候选检索与重排序的一致性对齐。

    

    基于图文结合查询的多模态搜索帮助用户在用户生成内容（UGC）社区中查找笔记，同时也是多模态搜索代理的关键搜索工具。对于此类查询，查询与笔记之间的相关性取决于用户的检索意图，无法仅通过视觉或主题相似性来判定。简单地对视觉搜索结果进行重排序无法完全解决这一挑战，因为符合用户意图的笔记可能在初始候选集中就已经缺失。这促使我们进行显式的意图建模，以便在一致的相关性标准下对齐候选检索与重排序两个环节。我们提出了IntentFlow，一个部署在小红书的意图驱动实时多模态搜索系统，其索引规模达到百亿级。IntentFlow通过层次化的意图分类体系来表示检索目标，并使用针对特定意图的相关性标准来指导数据构建、表示学习和重排序。一个离线代理（agent）……（原文摘要在此处被截断）

    arXiv:2611.20448v1 Announce Type: new  Abstract: Multimodal search with image-text queries helps users find notes in user-generated content (UGC) communities and serves as a key search tool for multimodal search agents. For such queries, query-note relevance depends on the user's retrieval intent and cannot be determined by visual or topical similarity alone. Simply reranking visual search results cannot fully address this challenge, as notes relevant to the user's intent may be missing from the initial candidate set. This motivates explicit intent modeling to align candidate retrieval and reranking under consistent relevance criteria. We present IntentFlow, an intent-driven real-time multimodal search system deployed at Xiaohongshu over an index on the order of ten billion entries. IntentFlow represents retrieval objectives with a hierarchical intent taxonomy and uses intent-specific relevance criteria to guide data construction, representation learning, and reranking. An offline agen
    
[^40]: 直达产品：当购物者先询问AI时，在线零售商错过了什么

    Straight to the Product: What Online Retailers Miss When Shoppers Ask AI First

    [https://arxiv.org/abs/2611.20378](https://arxiv.org/abs/2611.20378)

    本文首次将用户的AI助手对话与其在二十家大型在线零售商的实际浏览行为关联，发现助手提及的产品会使用户在一天内到访该产品页的概率提升约百倍（从0.02升至2.07每百次回答），但89%的访问并未通过助手链接到达、多经网页搜索实现，说明零售商追踪不到也错失了这类由AI引导而来的流量。

    

    当人们向AI助手询问该买什么时，接下来在商店里会发生什么？我们将同一批用户的助手对话记录与其后续在二十家大型在线零售商的浏览行为进行了关联。研究发现，用户会前往助手所提及的产品：在过去一周内未访问过该商店的购物者中，一天内对被提及产品的访问率从每100个回答中的0.02次上升至2.07次（提升2.0个百分点 [1.7, 2.4]）；在发生产品页面访问的当天，44%的访问涉及用户自己回答中提到的产品，而将同一回答匹配给其他用户的对照组仅为1.5%。该结果在另一个独立的五个月时间段内得到复现。很少有人通过助手提供的链接到达产品页：89%的访问没有任何来自该链接的痕迹，54%经由网页搜索实现，与用户平时的访问比例相同。当搜索恰好发生在首次访问之前时，搜索内容在28%的情况下包含回答中的产品（而在之前几周仅为1%），就好像用户把搜索框当成了地址栏。对话中点名了产品；而商店记录……（摘要在此处截断）

    arXiv:2611.20378v1 Announce Type: new  Abstract: When people ask an AI assistant what to buy, what happens next at the store? We link the same people's assistant conversations to their later browsing at twenty large online retailers. People go to the product the assistant named: among shoppers who had not visited the store in the past week, visits to the named product within a day rise from 0.02 to 2.07 per 100 answers (+2.0 points [1.7, 2.4]), and on days with a product-page visit at the store, 44% show a product from their own answer, against 1.5% for a matched answer given to someone else. The result replicates in a separate five-month period. Few arrive through the assistant's link: 89% of visits carry no sign of it, and 54% follow a web search, the same share as their usual visits. When a search comes just before the first visit, it names the answer's product 28% of the time (1% in earlier weeks), as if the search box were an address bar. The chat names the product; the store logs
    
[^41]: 仅名义上的匿名化：对欧盟委员会依据《数字市场法》第6(11)条针对谷歌搜索数据所制定匿名化措施的批判性审查与重识别实验

    Anonymous in Name Only: A Critical Review and a Reidentification Exercise on the European Commission's Anonymization Measures for Google Search Data under Article 6(11) of the DMA

    [https://arxiv.org/abs/2611.17012](https://arxiv.org/abs/2611.17012)

    本文批判性地审查了欧盟委员会依据《数字市场法》第6(11)条为谷歌搜索数据共享所制定的匿名化措施，指出其存在重大概念缺陷，并通过在真实数据上的低成本重识别实验证明该方法不应成为监管基准。

    

    arXiv:2611.17012v2 公告类型：replace-cross 摘要：2026年7月，欧盟委员会（EC）规定了Alphabet必须实施的匿名化措施，以便依据《数字市场法》（DMA）第6(11)条与竞争对手搜索引擎共享谷歌搜索数据。作为欧盟数字规则框架下委员会如何具体落实匿名化的首个实例，该方法论可能成为监管基准。本文论证了其不应成为基准的原因。我们对欧盟委员会提出的匿名化措施进行了批判性审查，涵盖残余风险的界定、抑制阈值以及对合同保障措施的依赖等方面。针对每一个问题，我们都指出了重大的概念性缺陷，并在适用情况下，利用Alphabet提供的、受欧盟委员会措施保护的真实谷歌搜索数据的实证观察来展示这些缺陷。重要的是，我们给出了低工作量重识别、记录链接以及机密属性

    arXiv:2611.17012v2 Announce Type: replace-cross  Abstract: In July 2026, the European Commission (EC) set out the anonymization measures Alphabet must implement to share Google Search data with competing search engines under Article 6(11) of the Digital Markets Act. As the first concrete illustration of how the EC operationalizes anonymization under the EU digital rulebook, this methodology could become a regulatory benchmark. In this paper, we show why it should not. We critically review the anonymization measures proposed by the EC, from the characterization of residual risk to the suppression thresholds and the reliance on contractual safeguards. For each of those issues, we highlight significant conceptual flaws, and, where applicable, we illustrate these flaws with empirical observations from real Google Search data provided by Alphabet and protected under the EC's measures. Importantly, we present examples of low-effort reidentification, record linkage, and confidential attribute
    
[^42]: SPARSEUP：面向现代学习型稀疏检索器训练的更优方案

    SPARSEUP: Towards Better Recipes for Training Modern Learned Sparse Retrievers

    [https://arxiv.org/abs/2611.09680](https://arxiv.org/abs/2611.09680)

    本文提出 SPARSEUP 训练方案，通过 logit 分布调控、逐位置扇出硬上限以及词汇表合并策略，让基于现代骨干网络的 Splade 类稀疏检索器实现更优的效率-效果权衡。

    

    学习型稀疏嵌入是稠密检索的一种强大替代方案，然而其发展已落后于单向量模型和迟交互模型。其应用受限的原因之一在于它们以难以训练而著称，通常需要非常严格的条件才能收敛。我们提出了一套在现代骨干网络和数据上训练 Splade 类编码器的完整方案，能够实现更优的效率与效果权衡。我们讨论了 logit 分布的作用及其对训练稳定性和编码器最终稀疏性的影响。我们还引入了一种新的逐位置扇出控制机制，即对每个位置可以激活的词汇维度数量设置硬性上限。此外，我们进一步应用了一种简单的词汇表合并策略，用于去除冗余词条，例如同一单词的大小写变体和词边界变体，这类冗余在基于区分大小写 BPE 的骨干网络中十分常见。我们将该方案应用于 LateOn 骨干网络和训练数据，由此得到了 SPARSEUP 模型。

    arXiv:2611.09680v1 Announce Type: new  Abstract: Learned sparse embeddings are a powerful alternative to dense retrieval, yet they have fallen behind single-vector and late interaction models. One reason for their limited adoption is that they are notoriously difficult to train, generally requiring very tight conditions to converge. We present a recipe for training Splade-like encoders on modern backbones and data that achieves a better efficiency-effectiveness trade-off. We discuss the role of logit distribution and its impact on the training stability and the final sparsity of the encoder. We also introduce a new per-position fan-out control - a hard cap on how many vocabulary dimensions each position may emit. We further apply a simple vocabulary merging strategy that removes redundant entries - such as case and word-boundary variants of the same word - which are common in cased BPE-based backbones. We apply our recipe to the LateOn backbone and training data, obtaining SPARSEUP, a 
    
[^43]: AutoRSE：将大语言模型智能体评估为自演化推荐系统的推荐工程师

    AutoRSE: Evaluating LLM Agents as Recommendation Engineers for Self-Evolving Recommender Systems

    [https://arxiv.org/abs/2610.37228](https://arxiv.org/abs/2610.37228)

    该论文提出了AutoRSE评估框架，通过覆盖六项工业级推荐任务的大规模实证研究发现，基于LLM的推荐工程智能体能将推荐性能平均提升16.1%，但目前主要充当工程优化者而非算法创新者。

    

    推荐系统工程智能体（RSE-Agents）利用基于大语言模型的智能体自主改进推荐算法，正在成为一种有前景的新范式。然而，由于缺乏统一、标准化的评估框架以及系统性的轨迹级分析，其能力与局限性仍然知之甚少。为填补这一空白，我们提出了AutoRSE——一个覆盖六项代表性工业级推荐任务的评估框架，并对三个前沿大语言模型在三种广泛使用的智能体框架上开展了大规模实证研究。结果表明，RSE-Agents在六项任务中的五项上持续提升了骨干模型的性能，平均相对提升达16.1%，表明它们已经能够自主改进推荐性能。然而，轨迹级分析也揭示，当前的RSE-Agents主要扮演的是工程优化者的角色，而非算法……（摘要原文在此处截断）

    arXiv:2610.37228v1 Announce Type: new  Abstract: Recommender Systems Engineer Agents (RSE-Agents), which use LLM-based agents to autonomously improve recommendation algorithms, are emerging as a promising paradigm. However, their capabilities and limitations remain poorly understood due to the lack of a unified, standardized evaluation framework and systematic trajectory-level analysis. To address this gap, we introduce AutoRSE, an evaluation framework spanning six representative industrial-grade recommendation tasks, and conduct a large-scale empirical study of three frontier LLMs across three widely used agent harnesses. Our results show that RSE-Agents consistently enhance the performance of backbones on five of the six tasks, with an average relative improvement of 16.1%, indicating that they can already autonomously improve recommendation performance. However, trajectory-level analysis also reveals that current RSE-Agents function primarily as engineering optimizers rather than al
    
[^44]: 主动记忆召回中规模的代价

    The Price of Scale in Proactive Memory Recall

    [https://arxiv.org/abs/2610.34507](https://arxiv.org/abs/2610.34507)

    该论文证明主动记忆召回的门控决策本质上是一个随记忆库规模 N 增长的多重假设检验问题——单条记忆的假阳性率按 -ln(1-α)/N 衰减，导致命中率随 N 呈幂次下降，从理论与实证上解释了记忆库越大、主动记忆触发越容易失效的现象。

    

    一个无需被询问便能主动提起记忆的长期助手，必须在每轮交互中判断其记忆库中是否有内容值得打断用户。我们证明这一决策本质上是一个多重假设检验问题，其难度随记忆库规模 N 的增长而加剧。在 LongMemEval 与 LoCoMo 的流式回放实验中，一个在 N=8 时调校至 5% 误报率的相似度阈值，在 N=2048 时误报率飙升至 87–100%，该现象在各类嵌入模型以及显式与隐式线索中均成立；LLM 判别器虽起点更低，也呈现同样的增长趋势。我们证明，一个误报率为 α 的门控机制，其单条记忆的假阳性率约为 -ln(1-α)/N，这将把其命中率锁定在单次比较 ROC 曲线的最左端，且在正则变化条件下，命中率随 N 以幂次规律衰减并带有显式常数。由此得出的预测与逐 N oracle 阈值的命中率相差在一个百分点以内。该分析解释了为何词汇与语义通道……（原文摘要在此处截断）

    arXiv:2610.34507v1 Announce Type: cross  Abstract: A long-lived assistant that brings up memories without being asked must decide at every turn whether anything in its store is worth an interruption. We show that this decision is a multiple-testing problem whose difficulty grows with the store size $N$. On streaming replays of LongMemEval and LoCoMo, a similarity threshold tuned to 5% false alarms at $N=8$ reaches 87-100% at $N=2048$, across embedding models and for explicit and implicit cues; LLM judges show the same growth from a lower base. We prove that a gate with false-alarm rate $\alpha$ operates at a per-item false-positive rate of about $-\ln(1-\alpha)/N$, which pins its hit rate to the far left of the single-comparison ROC curve and, under regular variation, makes it decay as a power of $N$ with an explicit constant. The resulting prediction matches the hit rate of a per-$N$ oracle threshold to within one percentage point. The analysis explains why lexical and semantic channe
    
[^45]: 印度法律文档中使用词汇、语义和修辞角色信息进行先前案例检索的研究

    A Study of Prior Case Retrieval Using Lexical, Semantic, and Rhetorical Role Information in Indian Legal Documents

    [https://arxiv.org/abs/2610.07437](https://arxiv.org/abs/2610.07437)

    本研究针对印度法律先前案例检索任务进行了三种检索系统设计的实证评估，发现修辞角色的组合检索优于单一角色或全文检索、法条相似性不具区分性，并揭示了内部验证MRR高达0.97的模型在官方评估中MRR仅0.14的泛化失效问题。

    

    在印度法律判决中检索先前案例的问题，涉及如何从单纯的词汇相似性中区分出相关的法律事实，因为与查询判决共享同一法条的先前案例不一定相关。本文对IL PCR（印度法律先前案例检索）任务的三种连续设计的检索系统进行了实证评估。我们证明了修辞角色（事实、判决理由、先例、论据、法条）的组合优于单独使用各个角色或执行全文检索；法条相似性不具区分性；使用LLM生成训练数据的法律蕴含重排序器在官方评估中的表现远差于其内部验证得分。两个排序器的内部MRR高达0.97，但在官方评估中表现不佳（MRR低至0.14）。因此，我们提出了一种新的设计……

    arXiv:2610.07437v1 Announce Type: new  Abstract: For retrieving prior cases in Indian legal judgments, the problem involves distinguishing relevant legal facts from mere lexical similarities because a prior case that shares a statute with the query judgment is not necessarily relevant. In this paper, we describe an empirical evaluation of three consecutive designs of retrieval systems for the IL PCR(Indian Legal Prior Case Retrieval) task. We demonstrate that the combination of the rhetorical roles (Fact, Ratio Of The Decision, Precedent, Argument, Statute) is better than either using individual roles or performing full text retrieval, that statute similarity is not discriminative, and that a legal entailment reranker with training data produced by an LLM is much worse in terms of official evaluation than its internal validation score. Two rankers, with excellent internal MRR up to 0.97, performed poorly in terms of official evaluation (MRR as low as 0.14). Thus, we propose a new desig
    
[^46]: MUSES：前瞻性学术思想根源检索基准

    MUSES: A Benchmark for Prospective Intellectual-Roots Retrieval

    [https://arxiv.org/abs/2609.00313](https://arxiv.org/abs/2609.00313)

    该论文提出了首个百万规模的前瞻性学术思想根源检索基准MUSES，通过作者确认的论文层面根源标注和熟悉度分级任务设计，用于评估检索系统发现那些日后被证明具有开创性、但目前不为人熟知的小众文献的能力。

    

    科学发现依赖于找到能够塑造后续研究方向的先前文献。现有的检索系统以相关性和流行度为优化目标，往往偏爱处于核心地位的论文，而忽视了那些不那么广为人知、但后来被证明具有开创性启发作用的著作。我们提出了MUSES，一个用于前瞻性学术思想根源检索的百万实例基准，基于固定的233万篇论文语料库构建，每个熟悉度层级约有14万个测试实例。据我们所知，这是首个在此规模上具有统一检索任务和作者确认的论文层面根源标注的前瞻性基准。与之配套的CiteRoots将基于局部引文文本的可扩展修辞层（LLM裁判与人类金标准的κ值为0.896）与论文层面的作者认可层（来自753篇焦点论文的1,518个生成性启发配对）相结合。MUSES沿两个维度组织任务难度：跨越CiteNext、CiteNew和CiteNew-Isolated的“熟悉度”维度，以及一个……（摘要原文在此处被截断）

    arXiv:2609.00313v1 Announce Type: new  Abstract: Scientific discovery depends on finding prior literature that shapes what comes next. Existing retrieval systems optimize for relevance and popularity, often favoring central papers over less familiar works that later prove generative. We introduce \textbf{MUSES}, a million-instance benchmark for prospective intellectual-roots retrieval over a fixed 2.33M-paper corpus, with roughly 140K test instances per familiarity tier. To our knowledge, it is the first prospective benchmark at this scale with a shared retrieval task and author-confirmed paper-level root labels. Alongside it, \textbf{CiteRoots} pairs a scalable rhetorical layer over local citation text (LLM judge $\kappa = 0.896$ versus human gold) with a paper-level author-endorsed layer ($n = 1{,}518$ generative-inspiration pairs from 753 focal papers). MUSES organizes difficulty along two axes: a \emph{familiarity} axis spanning CiteNext, CiteNew, and CiteNew-Isolated, and a \emph{
    
[^47]: 面向搜索与推荐联合建模的多兴趣方法

    Multi Interests for Joint Search-Recommendation Modeling

    [https://arxiv.org/abs/2608.10535](https://arxiv.org/abs/2608.10535)

    提出基于多兴趣的混合序列建模框架MIJSR，从结构与语义两个视角对搜索推荐混合序列进行多兴趣挖掘与自适应融合，实现搜索与推荐的联合建模。

    

    搜索与推荐对于理解用户偏好至关重要。越来越多的研究尝试对搜索行为与推荐行为进行联合建模，通过整合用户主动搜索与被动推荐的行为数据来更好地挖掘用户偏好。然而，尽管现有的跨域统一建模框架能够有效弥补不同领域之间行为差异，它们却忽视了混合序列中不同场景下兴趣的表达。在本研究中，我们提出了一种基于多兴趣的混合序列建模框架MIJSR，该框架从结构和语义两个视角对搜索推荐混合序列进行多兴趣挖掘与自适应融合。具体而言，我们的模型大致可分为三个模块：跨域行为融合、多兴趣挖掘和多任务预测。首先，我们对…（摘要在此处被截断）

    arXiv:2608.10535v2 Announce Type: replace  Abstract: Search and recommendation are crucial for understanding user preferences. More and more studies are attempting to jointly model search behavior and recommendation behavior, by integrating user active search and passive recommendation behavior data to better mine user preferences. However, although existing cross-domain unified modeling frameworks can effectively compensate for the differences in behavior between domains, they overlook the expression of interests in different scenarios under mixed sequences. In this study, we propose a multi-interest-based mixed sequential modeling framework MIJSR, which performs multi-interest mining and adaptive integration on search recommendation mixed sequences from both structural and semantic perspectives. Specifically, our model can be roughly divided into three modules: cross-domain behavior fusion, multi-interest mining, and multi-task prediction. Firstly, we align the representations of que
    
[^48]: IDP AutoOpt：智能文档处理流水线配置的代理驱动优化

    IDP AutoOpt: Agent-Driven Optimization of Document Processing Pipeline Configurations

    [https://arxiv.org/abs/2607.26075](https://arxiv.org/abs/2607.26075)

    IDP AutoOpt通过闭环LLM代理自动优化文档处理流水线配置，在降低成本的同时匹配或超越人类专家准确性，并将配置时间从数周缩短到两小时以下。

    

    我们提出了IDP AutoOpt，一个自主的LLM代理，用于发现智能文档处理（IDP）流水线的高性能配置。当前，联合调整IDP提示、模型、OCR设置和模式，每个文档类型需要领域专家20到80以上的人工小时，并且随着企业增加文档类别而无法扩展。IDP AutoOpt运行一个闭环：它在一个小标注集上对配置进行评分，诊断字段级错误，生成有针对性的编辑，并重新评估，由编码生产专业知识的人工编写的领域技能引导。在医疗保健、市场情报和金融服务环境中部署的提取、分类和分组任务中，IDP AutoOpt在相同或更低的成本下匹配或超过人类专家的准确性（在提取基准上，90.2%对比81.6%，每页成本低4.6倍），将配置时间从数周缩短到两小时以下。我们还进一步展示了...

    arXiv:2607.26075v2 Announce Type: replace-cross  Abstract: We present IDP AutoOpt, an autonomous LLM agent that discovers high-performing configurations for intelligent document processing (IDP) pipelines. Tuning IDP prompts, models, OCR settings, and schemas jointly currently costs domain specialists 20 to 80+ person-hours per document type and does not scale as enterprises add document classes. IDP AutoOpt runs a closed loop: it scores a configuration on a small labeled set, diagnoses field-level errors, generates targeted edits, and re-evaluates, guided by human-authored domain skills that encode production expertise. Across extraction, classification, and packet-splitting tasks deployed in healthcare, marketing-intelligence, and financial-services settings, IDP AutoOpt matches or exceeds human-expert accuracy at equal or lower cost (on an extraction benchmark, 90.2% vs 81.6% at 4.6 x lower per-page cost), cutting configuration time from weeks to under two hours. We further show tha
    
[^49]: 对人类不可见，对机器可见：四项生物医学文献API的Unicode保真度预注册审计

    Invisible to humans, visible to machines: a preregistered audit of Unicode fidelity across four biomedical bibliographic APIs

    [https://arxiv.org/abs/2606.24897](https://arxiv.org/abs/2606.24897)

    本研究开发并预注册了一种可复用的字符级保真度审计方法，系统评估了PubMed、Crossref、OpenAlex和Semantic Scholar四个生物医学文献API对Unicode特殊字符（如希腊字母、数学符号、排版标点和特殊空白）的保留能力，揭示了这些API存在的系统性字符丢失问题。

    

    本研究旨在开发一种可复用、预注册的方法，用于审计文献应用程序编程接口（API）的字符级保真度，并将其应用于用于构建生物医学文本语料库的四个公共API。这是一项预注册审计（OSF osf.io/zjevr）。研究从2024年PubMed Central（PMC）开放获取子集中简单随机抽取了4000篇英文研究论文，将PubMed E-utilities、Crossref、OpenAlex和Semantic Scholar返回的摘要与PMC期刊文章标签套件（JATS）XML摘要进行对比，共涉及51个注册码点，其中34个属于四个主要类别：排版标点符号、数学/科学符号、希腊字母和特殊空白字符。当条件保留率的95% Wilson置信区间上限低于5%时，某单元格（API x 类别）被归类为系统性丢失；当置信区间下限超过90%时，则被归类为保真。研究还对每类丢失的机制进行了检验……（原文在此处截断）

    arXiv:2606.24897v2 Announce Type: replace-cross  Abstract: To develop a reusable, preregistered method for auditing the character-level fidelity of bibliographic application programming interfaces (APIs), and to apply it to four public APIs used to build biomedical text corpora. Preregistered audit (OSF osf.io/zjevr). From the 2024 PubMed Central (PMC) Open Access subset, a simple random sample of 4,000 English-language research articles was drawn. Abstracts returned by the PubMed E-utilities, Crossref, OpenAlex and Semantic Scholar were compared with the PMC Journal Article Tag Suite (JATS) XML abstract for 51 registered code points, 34 of them in four primary classes: typographic punctuation, mathematical/scientific symbols, Greek letters and special whitespace. A cell (API x class) was classified as systematic loss when the upper 95% Wilson bound of conditional preservation was below 5%, and as faithful when the lower bound exceeded 90%. The mechanism of each loss was examined in th
    
[^50]: LCSHBench：一个多语言、基于共识的美国国会图书馆主题标目（LCSH）分配基准

    LCSHBench: A Multilingual, Consensus-Grounded Benchmark for Library of Congress Subject Heading Assignment

    [https://arxiv.org/abs/2606.04382](https://arxiv.org/abs/2606.04382)

    LCSHBench是首个基于多家图书馆编目共识构建的多语言公开基准，涵盖15种语言2万余本书，用于评估LCSH主题标目的自动分配，并同时衡量精确匹配与概念级匹配两种结果。

    

    自动主题编目是将受控词表标目分配给书目记录的过程，但LCSH（美国国会图书馆主题词表）一直缺乏标准的公开基准。我们推出LCSHBench：该数据集包含来自哈佛大学、哥伦比亚大学和普林斯顿大学开放许可目录的15种语言共22,346本图书。只有当至少两个独立编目机构为记录分配了LCSH标目时，该记录才被纳入数据集；我们同时发布了逐馆来源信息以及并集和完全一致两种答案视图。针对三家图书馆共同编目的465,187部作品开展的一致性研究证明了这一设计的必要性：各图书馆通常在底层主题上达成一致（93.3%的记录共享概念级标目），但在确切表达上往往不同（仅39.4%的记录具有完全相同的标目集合）。因此，LCSHBench同时评估精确匹配与概念匹配，并按语言和标目类型细分集合与排序指标，涵盖开放词表生成和全词表检索两类任务。作为首次演示，对一个3亿参数的端侧嵌入模型进行低秩微调……

    arXiv:2606.04382v2 Announce Type: replace-cross  Abstract: Automated subject cataloging assigns controlledvocabulary headings to bibliographic records, but LCSH has no standard public benchmark. We introduce LCSHBench: 22,346 books in 15 languages from the openly licensed Harvard, Columbia, and Princeton catalogs. Records enter only when at least two independent cataloging agencies assigned LCSH; we release per-catalog provenance plus union and unanimous answer views. A concordance study of 465,187 works cataloged by all three libraries shows why this design matters: libraries usually agree on the underlying topic (93.3% share a concept-level heading) but often differ in exact expression (39.4% have identical heading sets). LCSHBench therefore scores both exact and concept matches, with set and rank metrics broken down by language and heading type, across open-vocabulary generation and full-vocabulary retrieval. As a first demonstration, a low-rank fine-tune of a 300M on-device embedde
    
[^51]: MusicRec：用于音乐推荐的音频、歌词与互动特征增强

    MusicRec: Audio, Lyrics, and Engagement Feature Enrichment for Music Recommendation

    [https://arxiv.org/abs/2606.00125](https://arxiv.org/abs/2606.00125)

    该论文提出MusicRec——一个基于LLM的多模态音乐推荐框架，首次在统一的、基于歌曲内容的框架内联合建模音频/歌词嵌入、LLM生成的语义元数据和听歌完成率这三种互补信号，从而提升基于会话的音乐推荐效果。

    

    音乐推荐系统通常将歌曲视为不透明的标记，依赖于忽略语义或声学内容的交互历史。先前的LLM增强型、多模态以及文本增强型序列推荐器仅部分结合了语义、声学或互动信号，没有任何工作在基于歌曲内容的统一LLM框架内对这三种信号进行联合建模。在本工作中，我们提出了一个用于基于会话的音乐推荐的多模态框架，该框架通过三种互补信号来丰富LastFM-1K数据集：(1) 来自预训练音乐模型和文本模型的音频与歌词嵌入，(2) 使用MGPHot模式生成的LLM语义元数据，以及(3) 听歌完成率。我们针对全部58个属性，将生成的元数据与人工MGPHot标注进行了验证。我们通过多模态特征、包括SASRec、BERT4Rec和GRU4Rec在内的ItemID编码器骨干网络以及LLM骨干网络对E4SRec框架进行了扩展……

    arXiv:2606.00125v2 Announce Type: replace-cross  Abstract: Music recommendation systems typically treat songs as opaque tokens, relying on interaction histories that overlook semantic or acoustic content. Prior LLM-augmented, multimodal, and text-enhanced sequential recommenders partially combine semantic, acoustic, or engagement signals, and none jointly model all three within a unified LLM-based framework grounded in song content. In this work, we propose a multimodal framework for session-based music recommendation that enriches the LastFM-1K dataset with three complementary signals: (1) audio and lyric embeddings from pretrained music and text models, (2) LLM-generated semantic metadata using the MGPHot schema, and (3) listening completion ratios. We validate the generated metadata against human MGPHot annotations across all 58 attributes. We extend the E4SRec framework with multimodal features, ItemID encoder backbones including SASRec, BERT4Rec, and GRU4Rec, and LLM backbones inc
    
[^52]: 贝叶斯理性搜索引擎用户

    Bayesian Rational Search Engine User

    [https://arxiv.org/abs/2605.24233](https://arxiv.org/abs/2605.24233)

    该论文将搜索引擎用户的付费浏览行为建模为贝叶斯最优停止问题，证明最优策略是随深度变化的“突出规则”，通过一维马尔可夫链闭式刻画检查深度的完整分布，揭示信任、承诺与止损三种停止原因，并将观察到的检查深度转化为学习排序的似然函数。

    

    一位用户面对一个按有噪声的相关性代理指标排序的列表，需要决定是付费检查下一条项目，还是带着她已找到的最佳结果停止。由于她不了解页面的质量，每次检查既产生一个候选结果，又更新她对剩余项目的信念。我们证明最优策略是一条“突出规则”：当用户已找到的最佳结果相对于她对页面的期望足够好时，她便会停止，所需的优势裕度随浏览深度变化。浏览动态可坍缩为一维马尔可夫链，并通过闭式递归给出检查深度的完整分布。该模型揭示了停止浏览的三种原因：信任、承诺与止损，并给出可检验的推论。它还将检查深度转化为一个用于学习排序的似然函数：观察到的深度将潜在相关性路径约束在一个由生存不等式构成的多面体中，而转化行为则识别最终的w……（原文摘要在此截断）

    arXiv:2605.24233v4 Announce Type: replace  Abstract: A user faces a list ordered by a noisy proxy for relevance and decides whether to pay to inspect another item or stop with the best she has found. Because she does not know the page's quality, each inspection both produces a candidate and updates her beliefs about the remaining items. We show that the optimal policy is a standout rule: the user stops when her best find is sufficiently good relative to what she expects from the page, with the required margin changing with depth. The browsing dynamics collapse to a one-dimensional Markov chain, yielding the full distribution of inspection depth through a closed-form recursion. The model uncovers three reasons for stopping: trust, commit, and cut-losses, and yields testable implications. It also turns inspection depth into a learning-to-rank likelihood. Observed depth restricts the latent relevance path to a polyhedron of survival inequalities, while conversion identifies the terminal w
    
[^53]: EcoGEO：面向网络增强型大语言模型搜索代理的轨迹感知证据生态系统

    EcoGEO: Trajectory-Aware Evidence Ecosystems for Web-Enabled LLM Search Agents

    [https://arxiv.org/abs/2605.12887](https://arxiv.org/abs/2605.12887)

    本文提出EcoGEO框架，将生成式引擎优化（GEO）从单网页层面扩展为面向网络增强型LLM搜索智能体的环境级影响力问题，并设计了轨迹感知的协同证据生态系统TRACE，通过控制智能体浏览轨迹中证据的组织、连接与呈现方式来影响其搜索结果。

    

    arXiv:2605.12887v3 公告类型：replace-cross。摘要：具备网络访问能力的大语言模型智能体正在改变在线信息影响搜索结果的方式。现有的生成式引擎优化（GEO）研究主要聚焦于单个网页。然而，智能体化的网络搜索并非单文档场景：智能体可能会发出查询、抓取页面、跟踪链接、重新表述搜索，并在多个浏览步骤中综合证据。因此，影响力不仅取决于页面内容，还取决于页面如何沿着智能体的浏览轨迹被组织、连接和呈现。我们通过生态系统生成式引擎优化（EcoGEO）来研究这一转变，该方法将GEO视为面向网络增强型大语言模型智能体的环境级影响力问题。为了具体实现这一视角，我们提出了TRACE——一个轨迹感知的协同证据生态系统。给定一个推荐查询和一个虚构的目标产品，我们的方法构建了一个受控的证据环境，该环境协调智能体辅助的……（原文摘要在此处截断）

    arXiv:2605.12887v3 Announce Type: replace-cross  Abstract: Web-enabled LLM agents are changing how online information influences search outcomes. Existing Generative Engine Optimization (GEO) studies mainly focus on individual webpages. However, agentic web search is not a single-document setting: an agent may issue queries, crawl pages, follow links, reformulate searches, and synthesize evidence across multiple browsing steps. Influence therefore depends not only on page content, but also on how pages are organized, connected, and encountered along the agent's browsing trajectory. We study this shift through Ecosystem Generative Engine Optimization (EcoGEO), which treats GEO as an environment-level influence problem for web-enabled LLM agents. To instantiate this perspective, we propose TRACE, a Trajectory-Aware Coordinated Evidence Ecosystem. Given a recommendation query and a fictional target product, our method builds a controlled evidence environment that coordinates an agent-faci
    
[^54]: 评估用于医学问答的小型开源大语言模型：一个实用框架

    Evaluating Small Open LLMs for Medical Question Answering: A Practical Framework

    [https://arxiv.org/abs/2604.10535](https://arxiv.org/abs/2604.10535)

    该论文提出了一个开源评估框架，将可复现性与准确性并列为首要指标，通过八项质量指标和重复推理来评估可本地部署的小型开源LLM在医学问答中的表现。

    

    将大语言模型（LLM）应用于医学问答，所需的不只是较高的平均准确率：一个每次查询都返回实质性不同答案的模型并不是可靠的医疗工具。Reddit等在线健康社区已成为数百万用户获取医学信息的主要来源，但这些社区仍然极易受到错误信息的影响；在这些场景中部署LLM作为助手，进一步放大了在保证正确性的同时对输出一致性的需求。我们提出了一个实用的、开源的评估框架，用于评估小型、可本地部署的开源权重LLM在医学问答上的表现，并将可复现性作为与词汇和语义准确性同等重要的首要指标。我们的流水线计算八项质量指标，包括BERTScore、ROUGE-L以及LLM作为评判的评分标准，此外还包括从重复推理（N=10次运行）中得出的两项模型内可复现性指标

    arXiv:2604.10535v2 Announce Type: replace-cross  Abstract: Incorporating large language models (LLMs) in medical question answering demands more than high average accuracy: a model that returns substantively different answers each time it is queried is not a reliable medical tool. Online health communities such as Reddit have become a primary source of medical information for millions of users, yet they remain highly susceptible to misinformation; deploying LLMs as assistants in these settings amplifies the need for output consistency alongside correctness. We present a practical, open-source evaluation framework for assessing small, locally-deployable open-weight LLMs on medical question answering, treating reproducibility as a first-class metric alongside lexical and semantic accuracy. Our pipeline computes eight quality metrics, including BERTScore, ROUGE-L, and an LLM-as-judge rubric, together with two within-model reproducibility metrics derived from repeated inference (N=10 runs 
    
[^55]: UniFAR：一种面向科学文献的统一分面感知检索框架

    UniFAR: A Unified Facet-Aware Retrieval Framework for Scientific Documents

    [https://arxiv.org/abs/2602.23766](https://arxiv.org/abs/2602.23766)

    提出了UniFAR，一个统一的分面感知检索框架，通过多粒度表示与聚合模块在共享表示空间中同时支持文档-文档和问答-文档两种科学文献检索范式，从而融合二者的互补优势。

    

    科学文献检索（SDR）在现代科学研究中发挥着关键作用，支持知识发现和基于证据的推理。它沿着两种范式发展：由文档间对比学习驱动的文档-文档（doc-doc）检索，以及随着大语言模型（LLM）和RAG在自然语言交互中应用而兴起的问答-文档（q-doc）检索。在实践中，科学工作流程同时依赖这两种范式，既需要根据种子文档检索相关论文，也需要根据用户问题识别相关文档。然而，现有方法通常将这两种范式分开处理，阻碍了它们互补优势的发挥。为了解决这一问题，我们提出了UniFAR，一个统一的分面感知检索框架，能够在共享表示空间中同时支持文档-文档检索和问答-文档检索。UniFAR引入了多粒度表示与聚合模块，以统一短问题和（原文摘要在此处截断）

    arXiv:2602.23766v2 Announce Type: replace  Abstract: Scientific document retrieval (SDR) plays a critical role in modern scientific research, supporting knowledge discovery and evidence-based reasoning. It has evolved along two paradigms: document--document (doc-doc) retrieval driven by inter-document contrastive learning, and question--document (q-doc) retrieval emerging from LLMs and RAG for natural-language interaction. In practice, scientific workflows rely on both paradigms, requiring retrieval of related papers given a seed document and identifying relevant documents given a user question. However, existing methods typically treat these paradigms separately, hindering their complementary strengths. To address this, we propose UniFAR, a unified facet-aware retrieval framework that jointly supports doc-doc and q-doc retrieval within a shared representation space. UniFAR introduces a multi-granularity representation and aggregation module to unify the encoding of short questions and
    
[^56]: 先变换再查询：一种基于嵌入空间对齐的隐私保护向量检索方法

    Transform Before You Query: A Privacy-Preserving Approach for Vector Retrieval with Embedding Space Alignment

    [https://arxiv.org/abs/2507.18518](https://arxiv.org/abs/2507.18518)

    提出STEER隐私保护向量检索框架，利用不同嵌入模型语义空间之间的对齐关系，在本地将查询文本变换为近似嵌入向量后再查询，从而避免向向量数据库服务商暴露包含敏感信息的原始查询文本。

    

    向量数据库（VDB）能够高效地索引和搜索来自非结构化数据的高维向量嵌入，为生成式AI和推荐系统等现代AI应用提供至关重要的快速语义相似性搜索能力。由于当前的VDB服务提供商主要使用专有的黑盒模型，用户被迫通过API向其暴露原始查询文本，以换取向量检索服务。因此，如果查询文本涉及金融或医疗领域的机密记录，这种机制将不可避免地导致用户敏感信息的严重泄露。为解决这一问题，我们提出了STEER（安全变换嵌入向量检索），这是一个隐私保护的向量检索框架，它利用不同嵌入模型语义空间之间的对齐关系，为查询文本导出近似嵌入向量。

    arXiv:2507.18518v3 Announce Type: replace  Abstract: Vector Database (VDB) can efficiently index and search high-dimensional vector embeddings from unstructured data, crucially enabling fast semantic similarity search essential for modern AI applications like generative AI and recommendation systems. Since current VDB service providers predominantly use proprietary black-box models, users are forced to expose raw query text to them via API in exchange for the vector retrieval services. Consequently, if query text involves confidential records from finance or healthcare domains, this mechanism inevitably leads to critical leakage of user's sensitive information. To address this issue, we introduce STEER (\textbf{S}ecure \textbf{T}ransformed \textbf{E}mbedding v\textbf{E}ctor\textbf{ R}etrieval), a private vector retrieval framework that leverages the alignment relationship between the semantic spaces of different embedding models to derive approximate embeddings for the query text. STEE
    
[^57]: 基于大语言模型评估对话式推荐系统：以用户为中心的框架

    Evaluating Conversational Recommender Systems via Large Language Models: A User-Centric Framework

    [https://arxiv.org/abs/2501.09493](https://arxiv.org/abs/2501.09493)

    该论文提出了基于大语言模型的以用户为中心评估框架CoRE，通过单一指标同时评估对话式推荐系统的推荐准确性与对话管理质量，真实反映用户体验。

    

    对话式推荐系统（CRS）同时整合了推荐与对话两项任务，这使得对其评估面临独特的挑战。现有方法主要通过基于规则的指标，分别对物品推荐和对话管理进行评估来衡量CRS的性能。然而，这些方法无法捕捉真实的人类体验，也无法对系统的整体性能得出直接结论。随着对话式推荐系统在电子商务、社交媒体和客户支持领域日益重要，如何使用单一指标同时评估推荐准确性和对话管理质量，从而真实反映用户体验，已成为阻碍该领域发展的主要挑战。在这项工作中，我们提出了一种基于大语言模型（LLM）的、以用户为中心的CRS评估框架，即对话式推荐评估器。

    arXiv:2501.09493v5 Announce Type: replace  Abstract: Conversational recommender systems (CRSs) integrate both recommendation and dialogue tasks, making their evaluation uniquely challenging. Existing approaches primarily assess CRS performance by separately evaluating item recommendation and dialogue management using rule-based metrics. However, these methods fail to capture the real human experience, and they cannot draw direct conclusions about the system's overall performance. As conversational recommender systems become increasingly vital in e-commerce, social media, and customer support, the ability to evaluate both recommendation accuracy and dialogue management quality using a single metric, thereby authentically reflecting user experience, has become the principal challenge impeding progress in this field.   In this work, we propose a user-centric evaluation framework based on large language models (LLMs) for CRSs, namely Conversational Recommendation Evaluator (CoRE). CoRE con
    

