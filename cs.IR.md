# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ACARE: Agent-Context-Aware Reranker for Iterative Search](https://arxiv.org/abs/2610.35840) | ACARe 是一个40亿参数的智能体上下文感知重排序器，通过利用任务背景与智能体推理信息对文档重排序，在提升检索质量和答案准确率的同时减少搜索次数，并帮助小型智能体缩小与大智能体的差距。 |
| [^2] | [Do Similarity Measures Measure Similarity? How User Activity Creates Phantom Correlation in Recommender Systems](https://arxiv.org/abs/2610.35765) | 本文提出“幻影相关性”这一新现象，证明推荐系统中标准物品相似性度量即使在没有真实偏好结构的情况下，也会仅因用户活动量的异质性而检测出大量虚假的物品关联。 |
| [^3] | [Towards a Unified Framework of Optimal Input-Side Dimensional Configuration for Deep CTR Prediction](https://arxiv.org/abs/2610.35695) | 提出OptIDC统一框架，在深度CTR预测中联合学习样本、特征和嵌入单元三个层面的输入侧维度配置，克服了以往单一维度独立优化的局限。 |
| [^4] | [Cross-Domain Sequential Recommendation with Representation-Level User Transfer](https://arxiv.org/abs/2610.35550) | 该论文提出了一种仅通过学习到的用户表示进行跨域信息迁移的简单实用的跨域序列推荐框架，利用轻量级适配器将源域用户表示投影到目标域嵌入空间并与目标交互序列融合，无需共享原始交互数据或物品嵌入，从而应对业务数据隔离和用户异构性两大挑战。 |
| [^5] | [Tiny Suffix Arrays](https://arxiv.org/abs/2610.35543) | 该论文提出了一种仅占用 $O(n/\tau)$ 个机器字空间的辅助数据结构，能够在 $\widetilde{O}(\tau^3)$ 时间内回答文本上任意位置的后缀数组和逆后缀数组查询，实现了空间与时间之间可灵活调节的权衡。 |
| [^6] | [Hypergraph-Enhanced Evidence Selection for Long-Context Multimodal Document QA](https://arxiv.org/abs/2610.35542) | 提出HyperDoc框架，通过构建文档超图显式捕获结构、逻辑与跨模态关系，在有限输入预算下实现预算感知的多模态证据选择，从而提升长上下文多模态文档问答的性能。 |
| [^7] | [Beyond "[at]": A Large-Scale Measurement of Email Address Presentation on Computer Science Scholars' Homepages](https://arxiv.org/abs/2610.35499) | 该论文开发了MailScope工具，对超过3.2万名计算机科学学者主页上的电子邮件地址呈现方式进行了首次大规模系统测量，发现呈现形式高度多样（仅53.8%为纯文本，其余涵盖23种子类型）且呈现方式主要由托管平台而非学者个人决定。 |
| [^8] | [Read the Route: Attention-Steered Hypergraph Retrieval for Multi-hop RAG](https://arxiv.org/abs/2610.35412) | 该论文发现大语言模型中存在能于单次前向传播中跨多跳解析隐式实体引用的“寻址头”注意力头，并据此提出注意力引导的超图检索方法HyperAddress，以低推理成本实现准确的多跳RAG检索。 |
| [^9] | [Agreement Between Reconstructed and Native GDELT GKG Records: A Field-Level Baseline](https://arxiv.org/abs/2610.35358) | 研究者开发了开源工具 gdelt-regkg 来重构 GDELT 全球知识图谱记录，并对近 20 万篇文章进行了字段级一致性评估，首次为研究人员自行生成兼容 GKG 记录提供了量化基准。 |
| [^10] | [CRIM: Composable and Reusable Item Memory for Efficient Personalized Retrieval](https://arxiv.org/abs/2610.35323) | CRIM通过将物品压缩为与用户和查询无关的可复用连续记忆，并与请求级历史建模解耦，在推理时按交互顺序组合缓存记忆生成个性化检索表示，从而大幅降低大语言模型在个性化搜索与推荐中的训练与推理成本。 |
| [^11] | [Unified Heterogeneous Feature Interaction via Effective-Rank Scaling for Recommender Systems](https://arxiv.org/abs/2610.35318) | 提出了HiRank框架，通过有效秩缩放在统一主干中联合建模异构特征交互与门控序列注意力，解决了推荐系统中序列与非序列特征信息交换受限以及多场景负迁移的问题。 |
| [^12] | [Octen-WebGR: Scaling Generative Retrieval to 200 Million Web Pages](https://arxiv.org/abs/2610.35256) | Octen-WebGR提出一种生成式检索框架，将2亿网页存储在语言模型参数中，通过生成受URL表约束的双token语义标识符定位真实网页，在未见查询的导航搜索任务上显著超越稠密检索器。 |
| [^13] | [Reading Beyond OCR Noise: Robust Cross-Lingual RAG for Legislative Archives](https://arxiv.org/abs/2610.35240) | 本文提出一个将视觉版面分割、多文字OCR与代理式RAG相结合的端到端框架，通过迭代代理循环绕过OCR噪声与版面歧义，在20万页的印度旁遮普邦立法议会历史档案上实现了优于纯文本基线的鲁棒跨语言信息检索。 |
| [^14] | [When Fusion Order Matters: Insights on Progressive Multimodal Representation Learning for Cross-modal Retrieval](https://arxiv.org/abs/2610.35125) | 该论文提出UniPRO框架，在严格控制的实验条件下系统研究了多模态检索中融合顺序的影响，发现逐一融合优于堆叠融合，且“最后引入与查询匹配的模态”的调度策略在R@1指标上表现最佳，为融合顺序选择提供了两个互补的指导原则。 |
| [^15] | [Useful Diversity or Hallucinated Variety? Persona-Conditioned Synthetic Query Generation for E-Commerce Search](https://arxiv.org/abs/2610.35102) | 该研究通过人格条件化提示生成大规模合成查询来冷启动电商语义解析器，发现IPIP人格捆绑包可显著提升解析器F1、语法条件化能引导生成结果更贴近真实流量，并提出对齐sKL作为解析器性能的无标注代理指标。 |
| [^16] | [Learning to Remember: Advances in Memory Mechanisms for Large Language Models](https://arxiv.org/abs/2610.35089) | 本综述从记忆生命周期、存储基底和系统角色三个维度系统梳理了大语言模型记忆机制的研究，并将其划分为非参数化记忆、参数化记忆和面向智能体进化的记忆三大机制家族。 |
| [^17] | [Reading the Registers of Honorius III without Manual Line Transcription: Forced-Aligned HTR, End to End, Modular or Hybrid](https://arxiv.org/abs/2610.35085) | 本文提出利用强制对齐技术从无行对齐的校勘本底稿中自动生成训练数据，并采用结合CTC与双向子词注意力解码器的混合Conformer模型，实现了无需人工逐行转录即可对教皇霍诺留斯三世登记册中高度缩写的拉丁手稿进行端到端转录与缩写展开。 |
| [^18] | [Hallucination-Aware Multi-Agent RAG via Internal Attention Rewards](https://arxiv.org/abs/2610.35056) | 该论文提出一种幻觉感知的多智能体RAG框架，通过从生成器注意力头范数中提取内部幻觉风险信号并与共享任务奖励结合，在协作PPO训练中有效提升生成答案对检索证据的忠实性。 |
| [^19] | [CA-HNM: Constraint-Aware Hard Negative Mining for Dense Retrieval with Structured Domain Models](https://arxiv.org/abs/2610.35045) | CA-HNM通过将结构化领域模型的确定性不兼容规则与检索候选生成相结合，为稠密检索挖掘更符合领域结构约束的难负例，从而在多数数据集与编码器组合上提升了检索效果。 |
| [^20] | [SLIDE: Search-to-List Item Distillation for Single-Rollout Generative Retrieval](https://arxiv.org/abs/2610.34764) | SLIDE提出了一种将离线束搜索与在线列表生成解耦的物品蒸馏框架，利用教师模型检索的高质量候选提供额外监督，从而解决单次自回归生成列表时性能不佳的问题。 |
| [^21] | [Discovering Topics from Linear Autoencoders](https://arxiv.org/abs/2610.34739) | 该论文证明潜在主题可以直接从闭式解线性自编码器的权重矩阵中提取，无需额外训练即可解释推荐结果，并通过新颖的物品和用户归一化方法实现超越专用主题模型的主题质量，同时保持有竞争力的推荐性能。 |
| [^22] | [DPE-Rec: Diversity-Preserving and Efficient Diffusion Language Model Recommendation for Bundle Construction](https://arxiv.org/abs/2610.34700) | 提出DPE-Rec扩散语言模型推荐框架，通过解决置信度解码导致的语义分叉决策延迟和多词元并行解码的质量下降问题，在保持捆绑包多样性的同时实现高效推理。 |
| [^23] | [Knowing When the Evidence Is Enough: Requirement-Directed Search with Horn-Closure Certificates](https://arxiv.org/abs/2610.34684) | 该论文提出Certify框架，通过将查询分解为需求超图并利用Horn闭包运算生成证书，把搜索增强模型“证据是否足够”的判断结构化，从而可靠地决定回答、继续搜索或放弃回答并指出缺失的需求。 |
| [^24] | [Agent Memory as Recommendation: Recommender-System Design for Long-Term Memory of LLM Agents](https://arxiv.org/abs/2610.34673) | 该论文的核心创新是将LLM智能体的记忆选择问题形式化为预算约束下的板块推荐问题（请求为用户、记忆为物品、上下文窗口为板块），从而把各种智能体记忆系统统一纳入推荐级联的设计空间，并将准确性分解为可分析的组件。 |
| [^25] | [HARP-LoRA: Hierarchy-Conditioned Rank Projection for Low-Rank Adaptation in Scientific Table Understanding](https://arxiv.org/abs/2610.34589) | 该论文提出HARP-LoRA方法，利用表格行表头和列表头的层次结构为每个数据标记构建条件化表示并作用于低秩下投影，从而在参数高效的微调中提升科学表格（尤其是药代动力学表格）的理解能力。 |
| [^26] | [RefinedGAW: A Refined German Academic Web Dataset](https://arxiv.org/abs/2610.34572) | RefinedGAW是一个基于26次德国学术网络爬取构建的德语数据集，包含1.22亿份文档和3140亿词元，其中876亿学术词元超越了规模五倍于它的FineWeb2与FinePDFs德语子集，并为教育、数学和专家导向内容提供了无商业偏见的互补覆盖。 |
| [^27] | [The Price of Scale in Proactive Memory Recall](https://arxiv.org/abs/2610.34507) | 该论文证明主动记忆召回的门控决策本质上是一个随记忆库规模 N 增长的多重假设检验问题——单条记忆的假阳性率按 -ln(1-α)/N 衰减，导致命中率随 N 呈幂次下降，从理论与实证上解释了记忆库越大、主动记忆触发越容易失效的现象。 |
| [^28] | [Multimodal Requirements Traceability Link Recovery in Industrial Settings](https://arxiv.org/abs/2610.34503) | 该论文提出了一种多模态RAG扩展方法，通过分别检索文本和图像候选、利用视觉-语言模型进行分类并用言语化置信度过滤预测链接，从而在工业需求可追溯性链接恢复任务中将F1分数从基线的0.114提升至0.310。 |
| [^29] | [Theoretically Sound Item Similarity Measures for Recommender Systems](https://arxiv.org/abs/2610.34402) | 该论文提出了一种基于概率建模与参数估计的原则性方法来设计物品相似度度量，推导出基于伯努利分布和多项式分布的两种新度量，并证明常用的余弦相似度也可在此理论框架下得到解释。 |
| [^30] | [A Domain-Aware Pipeline for Structured Extraction and Retrieval-Augmented Question Answering over arXiv Superconductivity Literature](https://arxiv.org/abs/2610.34391) | 本文提出了一个面向超导领域的端到端流水线，通过结构化抽取材料与临界温度信息，并将领域属性融入混合检索，从而提升了arXiv超导文献检索增强问答的准确性与可靠性。 |
| [^31] | [Cross-Lingual Information Retrieval in Tetun for Ad-Hoc Search](https://arxiv.org/abs/2610.34362) | 本研究首次针对东帝汶多语言环境探索了德顿语跨语言信息检索的可行性，通过将葡萄牙语、英语和印尼语查询翻译回德顿语进行文档检索，发现谷歌翻译是目前德顿语最可靠的自动翻译工具。 |
| [^32] | [Factual Yet One-Sided: Stance Poisoning and Evidence Balancing in Retrieval-Augmented Generation](https://arxiv.org/abs/2610.34294) | 本文揭示了一种新型立场投毒攻击——注入事实正确但立场片面的材料即可使检索增强生成系统的输出立场随之偏移，并提出一个无需额外大模型调用的立场平衡中间件，通过立场检测、固定配比的证据平衡与去偏提示词来消除这一偏见。 |
| [^33] | [Recovering Business Rules as Cascade Lookups for Enterprise Field Prediction](https://arxiv.org/abs/2610.34258) | 本文提出一种知识图谱引导的LLM工作流，从历史交易中自动恢复企业业务规则并编译为确定性的级联查找，使字段预测在推理时无需模型服务且完全透明可解释。 |
| [^34] | [$\Delta\tau$-Bench: Pushing Agents Toward a Moving Capability](https://arxiv.org/abs/2610.34238) | Δτ-Bench是一个模拟工具生态系统持续演化的可执行智能体基准测试，包含覆盖42个领域的1,588个多轮任务，评估显示最强模型的任务成功率也仅达42.3%，揭示了智能体在应对工具环境动态变化方面的重大不足。 |
| [^35] | [MIDA: Multi-Interest Decoding with Adaptive Aggregation for Session-based Recommendation](https://arxiv.org/abs/2610.34192) | 提出MIDA框架，通过双空间解耦设计（在原始物品嵌入上聚类、利用编码器输出进行聚合）重构解码阶段，解决了基于会话推荐中过度关注近期物品、忽视远距离语义相关交互的问题。 |
| [^36] | [LoopGR: Parameter-Efficient Scaling for Generative Recommendation via Recurrent Depth](https://arxiv.org/abs/2610.34157) | LoopGR提出分层循环深度机制，在共享生成式解码器与轻量级上下文更新器之间交替计算，并结合循环轨迹自蒸馏实现高效推理，从而在不大幅增加模型规模的情况下实现生成式推荐的参数高效扩展。 |
| [^37] | [Decision-Native Recommendation with Large Language Models](https://arxiv.org/abs/2610.34124) | 该论文提出了“决策原生推荐”新范式及DICE方法，利用大语言模型表示直接在候选物品上进行选择估计并同时输出显式推荐置信度，解决了生成式推荐无法唯一确定物品、受候选顺序影响且缺乏置信度的问题。 |
| [^38] | [Bridging Search Intent and Tabular Data in Dataset Search](https://arxiv.org/abs/2610.33936) | 提出假设模式嵌入方法，利用语言模型生成任务对齐的假设表格模式进行嵌入检索，有效弥合了数据集搜索中用户意图与表格内容之间的鸿沟，并构建了KaggleDS语料库验证其在召回率上优于现有基线方法。 |
| [^39] | [TriageRAG: Failure-Aware Self-Training for Adapting Retrieval-Augmented Language Models to Specialized Domains](https://arxiv.org/abs/2610.33776) | TriageRAG提出了一种故障感知的自训练方法，通过在有无检索上下文的条件下探测学生模型，并将每个自生成的示例按失败类型（已知、阅读器失败或检索器失败）进行分诊，从而更有针对性地将RAG系统适配到医学等专业领域。 |
| [^40] | [SkillQR: Agentic Skill Discovery for Reason-Intensive Query Rewriting](https://arxiv.org/abs/2610.33768) | SkillQR提出了一种智能体式查询重写框架，能够针对不同检索目标自动发现最优的推理重写技能，解决了现有方法采用统一重写策略、依赖昂贵模型或大量训练数据的问题。 |
| [^41] | [Novel or Not? PatEx-Gym: Agentic Prior-Art Search for Novelty Assessment in Patent Examination](https://arxiv.org/abs/2610.33503) | 该论文提出了首个用于开发和评估专利审查智能体的框架PatEx-Gym，其包含1000个真实审查任务、超百万件专利与5.2万篇论文的大规模检索语料库，并创新性地引入无需标注的检索轨迹合成框架PASS，从而完整覆盖现有技术“检索—推理—评估”的全流程。 |
| [^42] | [DyHist: Query-Aware Historical Dynamics Modeling for Temporal Knowledge Graph Extrapolation](https://arxiv.org/abs/2610.33473) | 提出DyHist框架，通过查询感知地建模历史动态——分离近期证据与中程上下文的查询相关偏差，并关系感知地评估历史循环的可靠性——从而提升时序知识图谱外推的预测效果。 |
| [^43] | [PinGER: Token-Efficient Generative Retrieval with Semantic IDs](https://arxiv.org/abs/2610.33435) | PinGER是Pinterest部署的令牌高效生成式检索器，通过在交错的SID与稠密令牌上使用非对称注意力掩码实现输入历史压缩，并结合目标条件输出、强化学习和前缀多样性惩罚实现可控生成，在保留语义ID输出的同时大幅提升了上下文窗口的物品容量。 |
| [^44] | [Reinforcement Learning for Controller Parameters in the Auto-Bidding System at Pinterest](https://arxiv.org/abs/2610.33248) | 本文提出了一种部署在Pinterest的混合架构RLBP，以强化学习作为元控制器每日为各广告活动动态选择PID控制器参数，解决了传统固定PID调参无法适应多样化广告活动与非平稳市场条件的问题。 |
| [^45] | [Holo-NN: Holographic Nearest Neighbor Search via Compressive Sensing](https://arxiv.org/abs/2610.33224) | 提出Holo-NN代数框架，利用压缩感知与保局特征变换，以单一线性投影取代传统索引中数据依赖的非线性分支运算，从而在隐私保护约束（如全同态加密）下实现高效的高维近似最近邻搜索。 |
| [^46] | [How Much Fresh Data Should CTR Recalibration Use? A Temporal Evaluation of Post-hoc Calibration](https://arxiv.org/abs/2610.33168) | 该研究通过时序评估发现随机划分会显著低估CTR模型的校准误差，并证明使用12小时新鲜数据重新校准可将校准性能恶化的情形从36个中的13-17个降至3-5个，为事后校准的数据量选择提供了实证依据。 |
| [^47] | [Lessons from Luck: Do LLM-Written Lessons Carry Chance into an Agent's Next Decision?](https://arxiv.org/abs/2610.33029) | 本文通过预注册的受控实验（256个合成世界、11个LLM端点、90,084条回答），首次系统测量了LLM智能体从偶然（幸运或不幸）结果中撰写的经验教训是否会将“运气”传导到智能体的下一次工具选择决策中。 |
| [^48] | [Separating Thinking from Spelling for Efficient Generative Recommendation](https://arxiv.org/abs/2610.32980) | 该论文提出Deco方法，将阅读用户历史与书写物品标识符的过程解耦——骨干网络将历史一次性压缩为紧凑的“下一物品概念”，再由轻量的KV缓存解码器独立执行整个束搜索，从而大幅降低生成式推荐中随骨干规模、SID深度和候选数量增长的解码开销。 |
| [^49] | [The Delayed Countdown Method for Validating Review-Oriented AI Workflows in eDiscovery](https://arxiv.org/abs/2610.31311) | 本文提出延迟倒计时方法，为电子取证中审查导向型AI工作流的有效性验证提供了一种简单实用、易于集成、兼容任意停止规则并能给出有效置信区间的评估方案。 |
| [^50] | [FormBench: A Benchmark for Fine-Grained Retrieval in Formulation Patents](https://arxiv.org/abs/2610.30175) | 该论文提出了FormBench基准，用于评估模型从配方专利等技术文档中检索精确化学信息的能力，发现商业API编码器表现极差，而基于FormBench三元组进行任务自适应微调的开源编码器性能显著提升。 |
| [^51] | [Scriptorium: System Design and Implementation of a Multi-Human, Multi-AI Collaborative Digital Library Search Platform](https://arxiv.org/abs/2610.28929) | 该论文提出了Scriptorium，一个支持多人类与多AI协作的数字图书馆搜索平台，其核心创新在于将完全跨用户透明的搜索活动与由七个角色专门化生成式AI智能体组成的管理者-工作者架构相结合，并完整公开系统实现细节以支持复现与重用。 |
| [^52] | [ECHO: Efficient Cascade with Homogeneous Operators for Distillation-Guided 100K-Sequence Modeling](https://arxiv.org/abs/2610.28569) | 提出ECHO级联框架，利用蒸馏选择器（DS）和目标感知细粒度检索器（TFR）两级结构，在10万长度用户行为序列上实现高效且保持物品级可寻址性的目标感知推荐排序建模。 |
| [^53] | [Beyond Behavioral Accuracy: Evaluating Individual-Level Fidelity in LLM Investor Simulation](https://arxiv.org/abs/2610.26524) | 该研究通过80名参与者的纵向模拟交易实验，首次系统评估了LLM在个体层面对投资者行为与决策逻辑的保真度，发现当前LLM无法超越简单基线，买卖结构与资产选择尤其难以复现，且相似的活动表现会导致截然不同的投资组合轨迹。 |
| [^54] | [The Geometry of Agent Memory: Forecasting Retrieval-Bound Memory Failures, and Characterizing Its Limits](https://arxiv.org/abs/2610.26433) | 该研究用对话、记忆与查询之间的几何关系特征来预测智能体记忆系统的检索失效，但发现其整体预测能力不及简单的问题难度基线，且在控制问题难度和系统身份后，几何特征并未带来统计学上显著的性能提升。 |
| [^55] | [Comparing GenAI to Machine Learning and Manual Review for eDiscovery](https://arxiv.org/abs/2610.16925) | 该研究指出此前声称生成式AI审查工具（88%召回率）优于机器学习工具（64%召回率）的结论因方法差异而不可比，实际分析发现生成式AI与人工审查效果相当，而采用持续主动学习（CAL）的机器学习模型仅需2,000条样本训练即可达到最先进效果并具备更高效率。 |
| [^56] | [SearchJev: A Calibrated System-1 Model for Search Agents](https://arxiv.org/abs/2610.05107) | SearchJev提出一个无需自回归生成的校准化System-1模型，将搜索代理的短决策与System-2推理分离，并通过软标签学习（SLCD）从不确定监督中提升决策置信度的可靠性。 |
| [^57] | [5W1H+Which: Context-Valid Semantic Indexing with Progressive Ontology Binding](https://arxiv.org/abs/2609.35184) | 论文提出5W1H+Which设计，通过将基于源的内容提取与渐进式本体绑定相分离，实现了既保留可复用信息、又为推理提供可检验前提的上下文有效语义索引。 |
| [^58] | [Beyond Similarity through Zero-Token Geometric Graphs for Multi-Hop RAG](https://arxiv.org/abs/2609.19622) | 提出G³RAG框架，通过无需LLM调用和生成令牌的离线图构建，结合cosθ·sinθ几何增益分数、密度感知拓扑惩罚和单步受控扩散，在多跳RAG中实现超越纯相似性检索的互补证据发现。 |
| [^59] | [UniRec: Cross-stage Multi-Task Fusion with Preference Alignment for Cascaded Recommender Systems](https://arxiv.org/abs/2609.11052) | UniRec提出了一种统一的跨阶段推荐融合模型，通过共享嵌入和单一计算图联合训练实现跨阶段梯度传播，并结合双轴偏好对齐机制，解决级联推荐系统中预排序与排序阶段的跨阶段不一致问题。 |
| [^60] | [CoinRAG: Contextualized Information Nugget KV Cache Reuse for Long-Context RAG](https://arxiv.org/abs/2608.07458) | CoinRAG通过两阶段检索定位查询相关语义单元，并以组合方式复用离线计算的细粒度信息微块KV缓存，在低预填充延迟下实现了长上下文RAG效率与准确性的帕累托最优。 |
| [^61] | [Finding Change in Satellite Archives from Text: How to Combine Before-and-After Images Efficiently](https://arxiv.org/abs/2607.28571) | 本文在固定图像编码器与统一训练方案的条件下，系统对比了注意力机制、Mamba状态空间模型和时序瓶颈融合（TBF）三类共八种双时相图像融合设计，为文本驱动的卫星图像变化检索提供了高效融合模块的选择依据。 |
| [^62] | [Diagnosing and Mitigating Context Rot in Long-horizon Search](https://arxiv.org/abs/2606.29718) | 该论文通过对四个旗舰模型在三个基准上的系统性研究，首次揭示了“无结论搜索失败”这一此前被忽视的上下文腐化现象——模型远在用尽上下文窗口之前就放弃或给出不确定的错误答案，即使对简单查询也普遍存在，并进一步诊断与缓解这一问题。 |
| [^63] | [An Interpretable CF-RL-TOPSIS Fusion Model for Skills-Aware Talent Recommendation](https://arxiv.org/abs/2605.24155) | 本研究提出了一种可解释的CF-RL-TOPSIS融合模型，通过整合协同过滤、强化学习和熵权TOPSIS三种方法进行技能感知的人才推荐，在两个公开ICT人才数据集上显著优于现有基准方法。 |
| [^64] | [AllSERP: Exhaustive Per-Element Enrichment of the Versatile AdSERP Dataset](https://arxiv.org/abs/2605.04949) | AllSERP 通过为 AdSERP 数据集添加覆盖自然结果、People-Also-Ask 等非广告元素的类型化兴趣区域标注层，实现了搜索页面的穷举式逐元素增强，使 95.7% 的最终点击可归入类型化主轴 AOI。 |
| [^65] | [Good Scores, Bad Data: A Metric for Multimodal Coherence](https://arxiv.org/abs/2603.25924) | 提出多模态一致性得分（MCS），一种轻量级、无需人工标注的指标，通过将一致性分解为身份、空间、语义、决策四个维度来独立于下游模型评估多模态融合质量，其区分质量的灵敏度高于任务准确率。 |
| [^66] | [Improving the Performance of Compressed Transformers with Post-Hoc Embedding Alignment](https://arxiv.org/abs/2602.14635) | 提出了一种事后训练的对齐适配器（AlAd），通过将压缩Transformer模型的词元嵌入与性能更优的参考模型对齐，并利用局部上下文信息，从而恢复压缩所导致的性能损失。 |
| [^67] | [Navigating Ideation Space: Decomposed Conceptual Representations for Positioning Scientific Ideas](https://arxiv.org/abs/2601.08901) | 该论文提出“构思空间”这一结构化表示，通过对比训练将科学知识分解为三个独立维度，实现了科学思想间概念距离的原则性测量与具有区分性的新颖性评估，帮助研究者精确定位新想法相对于现有研究的差异。 |

# 详细

[^1]: ACARE：面向迭代搜索的智能体上下文感知重排序器

    ACARE: Agent-Context-Aware Reranker for Iterative Search

    [https://arxiv.org/abs/2610.35840](https://arxiv.org/abs/2610.35840)

    ACARe 是一个40亿参数的智能体上下文感知重排序器，通过利用任务背景与智能体推理信息对文档重排序，在提升检索质量和答案准确率的同时减少搜索次数，并帮助小型智能体缩小与大智能体的差距。

    

    在智能体搜索中，一篇文档可能匹配了智能体生成的子查询，却无助于回答用户的原始问题。传统重排序器仅针对子查询对文档进行评分，忽略了更宏观的任务背景和智能体的推理过程。我们提出了智能体上下文感知重排序器（ACARe），这是一个紧凑的40亿参数大语言模型，经训练可在上下文可用时利用该上下文对文档进行重排序。在多个基准测试和多种智能体系上，ACARe 在提升检索质量的同时降低了搜索成本。这些优势还延伸到答案生成阶段：与不使用重排序相比，ACARe 提高了准确率并减少了搜索次数，同时以更精准的证据达到或超过基础重排序器的平均准确率。它还能帮助较小的智能体缩小与大智能体在检索质量上的差距，表明更好的检索工具可以让小型智能体更加高效。模型、训练数据和代码即将发布。

    arXiv:2610.35840v1 Announce Type: new  Abstract: In agentic search, a document can match an agent-generated sub-query without helping answer the user's original question. Conventional rerankers score documents against the sub-query alone, overlooking the broader task and the agent's reasoning. We introduce the Agent-Context-Aware Reranker (ACARe), a compact 4B LLM trained to rerank documents using this context when available. Across multiple benchmarks and agent families, ACARe improves retrieval quality over its base reranker while reducing search effort. These benefits extend to answer generation: ACARe improves accuracy and reduces searches over no reranking, while matching or exceeding the base reranker's mean accuracy with more precise evidence. It also helps smaller agents narrow the retrieval-quality gap to larger agents, showing how better retrieval tools can make smaller agents more effective. The model, training data, and code will be released soon.
    
[^2]: 相似性度量真的能度量相似性吗？用户活动如何在推荐系统中制造“幻影相关性”

    Do Similarity Measures Measure Similarity? How User Activity Creates Phantom Correlation in Recommender Systems

    [https://arxiv.org/abs/2610.35765](https://arxiv.org/abs/2610.35765)

    本文提出“幻影相关性”这一新现象，证明推荐系统中标准物品相似性度量即使在没有真实偏好结构的情况下，也会仅因用户活动量的异质性而检测出大量虚假的物品关联。

    

    物品间相似度是推荐系统中历史最悠久、最基础的信号之一：当相同用户消费两个物品时，这两个物品就被认为彼此相似。尽管交互数据稀疏且规模庞大，该领域普遍隐含的假设是，物品共现反映了真实的共同偏好，刻画了用户喜欢的物品组合。本文证明，所观测到的物品间相似度中有相当大一部分是用户活动量差异造成的伪象，而非共同品味所致。即使每个用户完全独立地消费物品，仅在总消费量上存在差异，标准的相似性度量依然能检测到物品之间强关联。我们将这一现象命名为“幻影相关性”：即在完全不存在偏好结构的情况下，仅由异质的用户活动引发的虚假物品关联。从最少的正式假设出发，我们表明……（原文摘要在此处截断）

    arXiv:2610.35765v1 Announce Type: cross  Abstract: Item-item similarity is one of the oldest and most foundational signals in recommender systems: two items are assumed to be similar when the same users consume them. Despite the sparsity and scale of interaction data, the implicit assumption across the field is that item co-occurrence reflects genuine shared preference, capturing what users like together. In this paper, we demonstrate that a substantial fraction of observed item-item similarity is an artifact of variance in user activity rather than shared taste. Even if every user consumed items completely independently, differing only in their total consumption volume, standard similarity measures still detect strong associations between items. We term this phenomenon phantom correlation: spurious item association that emerges strictly from heterogeneous user activity in the complete absence of preference structure. Starting from minimal formal assumptions, we show that independent c
    
[^3]: 迈向深度CTR预测中最优输入侧维度配置的统一框架

    Towards a Unified Framework of Optimal Input-Side Dimensional Configuration for Deep CTR Prediction

    [https://arxiv.org/abs/2610.35695](https://arxiv.org/abs/2610.35695)

    提出OptIDC统一框架，在深度CTR预测中联合学习样本、特征和嵌入单元三个层面的输入侧维度配置，克服了以往单一维度独立优化的局限。

    

    深度点击率（CTR）预测通过日益强大的特征交互架构取得了显著成功。然而，预测性能还取决于在交互建模之前输入侧信息的配置方式。已有研究表明，选择信息丰富的样本、特征或嵌入单元能够提升预测准确率。然而，这些研究主要在单一输入侧维度上独立优化各选择机制，忽视了它们共同的基本形式以及不同输入层次之间的内在依赖关系，从而限制了输入侧维度配置的有效性。为解决这一问题，我们提出了OptIDC，一个面向深度CTR预测的最优输入侧维度配置统一框架。OptIDC通过整合各维度对应的选择机制，在统一框架内联合学习样本、特征和嵌入单元层面的配置。

    arXiv:2610.35695v1 Announce Type: cross  Abstract: Deep click-through rate (CTR) prediction has achieved substantial success through increasingly expressive feature interaction architectures. However, predictive performance also depends on how input-side information is configured before interaction modeling. Existing studies have shown that selecting informative samples, features, or embedding units can improve prediction accuracy. Nevertheless, they primarily optimize these selection mechanisms independently along a single input-side dimension, overlooking their shared basic form and the inherent dependencies across input levels, thereby limiting the effectiveness of input-side dimensional configuration. To address this issue, we propose OptIDC, a unified framework for Optimal Input-side Dimensional Configuration in deep CTR prediction. OptIDC jointly learns sample, feature, and embedding-unit level configurations within a unified framework by integrating their corresponding selection
    
[^4]: 基于表征级用户迁移的跨域序列推荐

    Cross-Domain Sequential Recommendation with Representation-Level User Transfer

    [https://arxiv.org/abs/2610.35550](https://arxiv.org/abs/2610.35550)

    该论文提出了一种仅通过学习到的用户表示进行跨域信息迁移的简单实用的跨域序列推荐框架，利用轻量级适配器将源域用户表示投影到目标域嵌入空间并与目标交互序列融合，无需共享原始交互数据或物品嵌入，从而应对业务数据隔离和用户异构性两大挑战。

    

    跨域序列推荐旨在利用来自辅助源域的用户行为来改进目标域中的下一项预测。然而，现实世界的部署面临两个关键挑战：一是隔离跨域交互数据的业务限制，二是用户异构性（即许多用户缺乏跨域活动）。这使得现有的跨域方法难以应用。在本文中，我们提出了一个简单且实用的跨域序列推荐框架，该框架仅通过学习到的用户表示来传递信息，而无需跨域共享原始交互数据或物品嵌入。我们引入了一个轻量级适配器，将源域用户表示投影到目标域嵌入空间中，并将其与目标交互序列集成。重要的是，当源数据不可用时（即用户仅在目标域中被观察到时），该框架依然能够……

    arXiv:2610.35550v1 Announce Type: cross  Abstract: Cross-domain sequential recommendation aims to improve next-item prediction in a target domain by leveraging user behavior from auxiliary source domains. However, real-world deployment faces two critical challenges: business constraints that isolate interaction data across domains, and user heterogeneity, where many users lack cross-domain activity. This makes existing cross-domain methods difficult to apply. In this paper, we propose a simple and practical framework for cross-domain sequential recommendations that transfers information exclusively through learned user representations, without sharing raw interaction data or item embeddings across domains. We introduce a lightweight adapter that projects the source-domain user representation into the target-domain embedding space and integrates it with the target interaction sequence. Importantly, when source data is unavailable (i.e., the user observed only in the target domain), the 
    
[^5]: 微型后缀数组

    Tiny Suffix Arrays

    [https://arxiv.org/abs/2610.35543](https://arxiv.org/abs/2610.35543)

    该论文提出了一种仅占用 $O(n/\tau)$ 个机器字空间的辅助数据结构，能够在 $\widetilde{O}(\tau^3)$ 时间内回答文本上任意位置的后缀数组和逆后缀数组查询，实现了空间与时间之间可灵活调节的权衡。

    

    对于长度为 $n$ 的文本，后缀数组查询和逆后缀数组查询可以在 $\widetilde{O}(\tau^3)$ 时间内得到回答，所使用的辅助数据结构仅占用 $O(n/\tau)$ 个机器字的空间，其中 $\tau$ 可以取 $[1,n]$ 范围内的任意值。

    arXiv:2610.35543v1 Announce Type: cross  Abstract: Suffix array and inverse suffix array queries on a text of length $n$ can be answered in $\widetilde{O}(\tau^3)$ time using an auxiliary data structure occupying $O(n/\tau)$ words, for any $\tau \in [1,n]$.
    
[^6]: 超图增强的长上下文多模态文档问答证据选择

    Hypergraph-Enhanced Evidence Selection for Long-Context Multimodal Document QA

    [https://arxiv.org/abs/2610.35542](https://arxiv.org/abs/2610.35542)

    提出HyperDoc框架，通过构建文档超图显式捕获结构、逻辑与跨模态关系，在有限输入预算下实现预算感知的多模态证据选择，从而提升长上下文多模态文档问答的性能。

    

    长上下文多模态文档问答已成为现代视觉语言模型（VLM）一项重要但具有挑战性的任务，因为它需要在有限的输入预算内，对分散在不同页面和不同模态中的证据进行推理。当前的流程采用检索增强生成（RAG）来为下游的视觉语言模型选择多模态证据。然而，现有系统主要依赖于基于固定粒度检索单元（如页面或文本块）的静态选择规则，导致检索过程中的信息丢失，并使下游推理性能不理想。为缓解这些问题，我们将核心任务形式化为对结构化文档证据的预算感知证据选择，并提出了HyperDoc——一个文档RAG框架，它在保留文档关系的同时构建紧凑且充分的证据。具体而言，HyperDoc构建了一个文档超图，以显式捕获结构、逻辑以及跨模态关系。

    arXiv:2610.35542v1 Announce Type: new  Abstract: Long-context multimodal document question answering (QA) has become an important yet challenging task for modern vision-language models (VLMs), as it requires reasoning over evidence dispersed across pages and modalities within limited input budgets. Current pipelines adopt retrieval-augmented generation (RAG) to select multimodal evidence for downstream VLMs. However, existing systems mainly rely on static selection rules over fixed-granularity retrieval units such as pages or blocks, resulting in information loss during retrieval and suboptimal performance in downstream reasoning. To alleviate these issues, we formulate the core task as budget-aware evidence selection over structured document evidence and propose HyperDoc, a document RAG framework that constructs compact yet sufficient evidence while preserving document relations. Specifically, HyperDoc constructs a document hypergraph to explicitly capture structural, logical, and cro
    
[^7]: 超越"[at]"：计算机科学学者主页上电子邮件地址呈现方式的大规模测量

    Beyond "[at]": A Large-Scale Measurement of Email Address Presentation on Computer Science Scholars' Homepages

    [https://arxiv.org/abs/2610.35499](https://arxiv.org/abs/2610.35499)

    该论文开发了MailScope工具，对超过3.2万名计算机科学学者主页上的电子邮件地址呈现方式进行了首次大规模系统测量，发现呈现形式高度多样（仅53.8%为纯文本，其余涵盖23种子类型）且呈现方式主要由托管平台而非学者个人决定。

    

    发布电子邮件地址使学者能够被联系到，但同时也使其暴露于自动化收集之中。然而，电子邮件地址在网络上的呈现方式尚未从内容、形式和位置三个维度得到系统性测量。我们使用MailScope对CSRankings收录的32,413条计算机科学学者主页记录进行了测量，该工具保留原始和渲染后的HTML，通过三阶段级联方法（先是规则，然后是大语言模型，最后是爬虫代理）提取地址，并对每个地址进行标注。在19,517个可访问的主页中，我们为17,712位学者（90.8%）找到了地址。我们的测量得出了四个主要发现：(i) 呈现方式多样：只有53.8%发布纯文本内容，其余分布在符号替换、代码和语义描述共23种子类型中，且5.5%的学者地址仅在页面渲染后才出现。(ii) 呈现方式遵循托管平台而非个人：同一托管平台上的学者共享相同的内容类别……（原文摘要在此处被截断）

    arXiv:2610.35499v1 Announce Type: new  Abstract: Publishing an email address makes a scholar reachable, but also exposes it to automated collection. Yet how email addresses are presented on the Web has not been measured systematically across their content, form, and location. We measure it across 32,413 homepage records of computer-science scholars listed in CSRankings with MailScope, which preserves raw and rendered HTML, extracts addresses with a three-stage cascade (rules, then large language models, then a crawling agent), and labels each address. Among 19,517 reachable homepages, we find an address for 17,712 scholars (90.8%). Our measurement yields four main findings. (i) Presentation is diverse: only 53.8% publish plain content, the others spread over 23 subtypes of symbol substitution, code, and semantic description, and 5.5% of scholars' addresses appear only after rendering. (ii) Presentation follows hosts rather than individuals: scholars on one host share a content class in
    
[^8]: 读懂路线：面向多跳RAG的注意力引导超图检索

    Read the Route: Attention-Steered Hypergraph Retrieval for Multi-hop RAG

    [https://arxiv.org/abs/2610.35412](https://arxiv.org/abs/2610.35412)

    该论文发现大语言模型中存在能于单次前向传播中跨多跳解析隐式实体引用的“寻址头”注意力头，并据此提出注意力引导的超图检索方法HyperAddress，以低推理成本实现准确的多跳RAG检索。

    

    基于图的检索增强生成通过将语料库组织成图并沿其结构进行检索，来回答多跳问题。现有检索方法面临两个局限：基于扩散和相似度的方法沿固定的边权重或依据与问题的相似度进行检索扩展，却无法解析问题中的隐式实体引用，导致相关性误差在多跳过程中不断累积；而基于探索的方法让大语言模型推理下一跳，但每一跳都需要单独进行一次生成，造成高昂的推理成本。我们观察到，大语言模型在生成任何文本之前，就能够从检索到的段落和多跳问题中解析出隐式实体引用。进一步分析发现，存在一小组注意力头（称为“寻址头”），它们能够在单次前向传播中捕获跨多个跳数的这种能力。基于这一发现，我们提出了HyperAddress，它运行个性化（PageRank）……（原文摘要至此截断）

    arXiv:2610.35412v1 Announce Type: new  Abstract: Graph-based retrieval-augmented generation (GraphRAG) answers multi-hop questions by organizing a corpus into a graph and retrieving along its structure. Existing retrieval methods face two limitations. Diffusion- and similarity-based methods expand retrieval along fixed edge weights or by similarity to the question, without resolving its implicit entity references, allowing relevance errors to accumulate across hops. In contrast, exploration-based methods let an LLM reason about the next hop, but require a separate generation at each hop, leading to high inference costs. We observe that an LLM can resolve implicit entity references from retrieved passages and a multi-hop question before generating any tokens. Further analysis identifies a small set of attention heads, termed addressing heads, that capture this capability across multiple hops in a single forward pass. Building on this finding, we propose HyperAddress, which runs Personal
    
[^9]: 重构记录与原生GDELT全球知识图谱记录的一致性：一个字段级基准

    Agreement Between Reconstructed and Native GDELT GKG Records: A Field-Level Baseline

    [https://arxiv.org/abs/2610.35358](https://arxiv.org/abs/2610.35358)

    研究者开发了开源工具 gdelt-regkg 来重构 GDELT 全球知识图谱记录，并对近 20 万篇文章进行了字段级一致性评估，首次为研究人员自行生成兼容 GKG 记录提供了量化基准。

    

    全球事件、语言与语气数据库（GDELT）的全球知识图谱（GKG）通过命名实体、主题、定量与时间特征、引文、语气以及基于词典的内容度量来刻画新闻文章的特征。其代码手册定义了记录格式，但生成记录的软件并未公开，因此研究人员无法为自己收集的文章生成兼容的记录。我们构建了一个开源重构版本 gdelt-regkg，并测量了它与 GDELT 存档（原生）记录的一致程度，评估对象涵盖来自 8,729 个出版商（2020-2025 年）的 193,550 篇英文文章，以及五种其他语言的 2,500 篇文章。一致性在很大程度上取决于具体字段。英文语气（正面词减去负面词所占百分比）的平均绝对误差（MAE）为 1.79 个百分点，一致性相关系数为 0.75。身份识别一致性（F1）方面，人物为 0.65，组织为 0.42，地点为 0.65。

    arXiv:2610.35358v1 Announce Type: new  Abstract: The Global Knowledge Graph (GKG) of the Global Database of Events, Language, and Tone (GDELT) characterizes news articles using named entities, topics, quantitative and temporal features, quotations, tone, and dictionary-based content measures. Its codebook defines the record format, but the software that produces the records is not public, so researchers cannot generate compatible records for articles they collect themselves. We built an open-source reconstruction, gdelt-regkg, and measured how closely it agrees with GDELT's archived (native) records for 193,550 English articles from 8,729 publishers (2020-2025) and 2,500 articles in five other languages. Agreement depends strongly on the field. English tone, the percentage of positive minus negative words, has a mean absolute error (MAE) of 1.79 points and a concordance correlation coefficient of 0.75. Identity agreement (F1) is 0.65 for persons, 0.42 for organizations, and 0.65 for lo
    
[^10]: CRIM：面向高效个性化检索的可组合与可复用物品记忆

    CRIM: Composable and Reusable Item Memory for Efficient Personalized Retrieval

    [https://arxiv.org/abs/2610.35323](https://arxiv.org/abs/2610.35323)

    CRIM通过将物品压缩为与用户和查询无关的可复用连续记忆，并与请求级历史建模解耦，在推理时按交互顺序组合缓存记忆生成个性化检索表示，从而大幅降低大语言模型在个性化搜索与推荐中的训练与推理成本。

    

    在个性化搜索与推荐任务中，大语言模型会产生高昂的训练与推理成本，原因在于不断增长的交互历史延长了请求输入，并使相同的物品文本被反复暴露给模型。我们提出了可组合与可复用物品记忆（Composable and Reusable Item Memory, CRIM），这是一个将物品级语义编码与请求级历史建模相分离的框架。一小组可学习的摘要词元将每个物品压缩为紧凑的连续记忆，同时片段级读取词元（REC词元）在交互历史内部引入预测性监督。这些记忆独立于用户和查询，因此可以在不同请求之间复用。在推理阶段，CRIM按交互顺序组合缓存的记忆，并结合可选的查询来生成个性化检索表示，在减少重复物品文本编码的同时保留与请求相关的上下文信息。

    arXiv:2610.35323v1 Announce Type: cross  Abstract: Large language models incur substantial training and inference costs in personalized search and recommendation, as growing interaction histories lengthen request inputs and repeatedly expose the same item text to the model. We propose \textbf{C}omposable and \textbf{R}eusable \textbf{I}tem \textbf{M}emory (CRIM), a framework that separates item-level semantic encoding from request-level history modeling. A small set of learnable summary tokens compresses each item into compact continuous memories, while segment-level readout tokens (REC tokens) introduce predictive supervision within the interaction history. These memories are independent of users and queries and can therefore be reused across requests. At inference time, CRIM composes cached memories in interaction order and incorporates an optional query to produce a personalized retrieval representation, reducing repeated item-text encoding while retaining request-dependent contextu
    
[^11]: 基于有效秩缩放的推荐系统统一异构特征交互

    Unified Heterogeneous Feature Interaction via Effective-Rank Scaling for Recommender Systems

    [https://arxiv.org/abs/2610.35318](https://arxiv.org/abs/2610.35318)

    提出了HiRank框架，通过有效秩缩放在统一主干中联合建模异构特征交互与门控序列注意力，解决了推荐系统中序列与非序列特征信息交换受限以及多场景负迁移的问题。

    

    大规模工业级排序模型必须在严格的在线服务约束下，同时捕获异构特征交互、长期用户行为和多场景依赖关系。现有架构通常使用独立的模块来处理序列特征和非序列特征，限制了双向信息交换。此外，当序列信号和非序列信号表现出不同的有效秩时，直接对所有语义标记应用同构自注意力可能效率低下。同时，共享表征可能会忽略场景特定的模式，并在不同场景之间引发负迁移。在本文中，我们提出了HiRank，这是一个用于多场景推荐的统一框架，通过有效秩缩放联合建模异构特征交互。HiRank在统一的主干网络中将门控序列注意力与异构特征交互相结合，实现双向……（摘要在此处截断）

    arXiv:2610.35318v1 Announce Type: new  Abstract: Large-scale industrial ranking models must jointly capture heterogeneous feature interactions, long-term user behaviors, and multi-scenario dependencies under stringent online serving constraints. Existing architectures typically process sequential and non-sequential features with separate modules, limiting bidirectional information exchange. Directly applying homogeneous self-attention to all semantic tokens can also be inefficient when sequential and non-sequential signals exhibit different effective-ranks (eRanks). Meanwhile, shared representations can overlook scenario-specific patterns and induce negative transfer between scenarios. In this paper, we propose HiRank, a unified framework for multi-scenario recommendation that jointly models Heterogeneous feature Interactions via effective-Rank scaling. HiRank integrates gated sequential attention with heterogeneous feature interaction within a unified backbone, enabling bidirectional 
    
[^12]: Octen-WebGR：将生成式检索扩展至2亿网页规模

    Octen-WebGR: Scaling Generative Retrieval to 200 Million Web Pages

    [https://arxiv.org/abs/2610.35256](https://arxiv.org/abs/2610.35256)

    Octen-WebGR提出一种生成式检索框架，将2亿网页存储在语言模型参数中，通过生成受URL表约束的双token语义标识符定位真实网页，在未见查询的导航搜索任务上显著超越稠密检索器。

    

    导航搜索旨在从数亿网页中找到唯一的目标页面。点击表对于之前见过的查询效果良好，但无法泛化到新的查询表述；而基于嵌入的检索器虽然能在内容上泛化，但对导航目标的排序往往不佳。我们提出Octen-WebGR，这是一种将网页存储在语言模型参数中的生成式检索框架：给定一个查询，模型生成一个双token语义标识符，该标识符被约束在一个真实URL表中。我们开发了Octen-WebGR-4B，可索引2.01亿个网页，并在训练之后一个月的真实导航流量上进行了评估。生成式检索在用新的表述方式请求熟悉页面时最为有效：对于针对具有大量交互历史的页面的未见查询，它有30.9%的概率将正确页面排在第一位，而在相同点击数据上训练的稠密检索器仅为17.2%（使用点击先验时为21.4%）。嵌入

    arXiv:2610.35256v1 Announce Type: new  Abstract: Navigational search aims to find a single target page among hundreds of millions of web pages. Click tables work well for queries seen before but cannot generalise to new formulations, while embedding retrievers generalise across content but often rank navigational targets poorly. We introduce Octen-WebGR, a generative retrieval framework that stores web pages in the parameters of a language model: given a query, the model generates a two-token semantic identifier constrained to a table of real URLs. We develop Octen-WebGR-4B, which indexes 201 million web pages, and evaluate it on real navigational traffic from the month after training. Generative retrieval is most effective on new ways of asking for familiar pages: for unseen queries targeting pages with substantial interaction history, it ranks the correct page first 30.9% of the time, against 17.2% for a dense retriever trained on the same clicks (21.4% with a click prior). Embedding
    
[^13]: 超越OCR噪声的阅读：面向立法档案的鲁棒跨语言检索增强生成

    Reading Beyond OCR Noise: Robust Cross-Lingual RAG for Legislative Archives

    [https://arxiv.org/abs/2610.35240](https://arxiv.org/abs/2610.35240)

    本文提出一个将视觉版面分割、多文字OCR与代理式RAG相结合的端到端框架，通过迭代代理循环绕过OCR噪声与版面歧义，在20万页的印度旁遮普邦立法议会历史档案上实现了优于纯文本基线的鲁棒跨语言信息检索。

    

    印度的历史公共档案带来了严峻的计算机视觉与文档图像分析挑战：数十年的议会记录仅以退化的多栏扫描页面形式保存，其特点是混合文字、复杂版面和密集的视觉噪声，其中OCR失败是信息检索的首要障碍。我们提出了一个端到端框架，将基于视觉的版面分割和多文字OCR与代理式检索增强生成（RAG）管道相结合。我们的方法专门研究视觉退化和版面碎片化如何传播为文本提取错误，并利用迭代式代理循环在生成答案之前绕过OCR噪声和版面歧义。在20万页的旁遮普邦立法议会（Punjab Vidhan Sabha）档案上的评估表明，我们的系统能够鲁棒地处理历史文档退化问题，性能优于标准的纯文本基线方法。

    arXiv:2610.35240v1 Announce Type: cross  Abstract: India's historical public archives present severe computer vision and document image analysis challenges: decades of parliamentary records survive only as degraded, multi-column scanned pages featuring mixed scripts, complex layouts, and dense visual noise where OCR failure is the primary obstacle to information retrieval. We present an end-to-end framework that couples vision-based layout segmentation and multi-script OCR with an agentic retrieval-augmented generation (RAG) pipeline. Our approach specifically addresses how visual degradation and layout fragmentation propagate into text extraction errors, using an iterative agentic loop to navigate around OCR noise and layout ambiguities before generating answers. Evaluated on the 200,000-page Punjab Vidhan Sabha archive, our system demonstrates robust handling of historical document degradation, outperforming standard text-only baselines.
    
[^14]: 当融合顺序至关重要时：面向跨模态检索的渐进式多模态表示学习的洞察

    When Fusion Order Matters: Insights on Progressive Multimodal Representation Learning for Cross-modal Retrieval

    [https://arxiv.org/abs/2610.35125](https://arxiv.org/abs/2610.35125)

    该论文提出UniPRO框架，在严格控制的实验条件下系统研究了多模态检索中融合顺序的影响，发现逐一融合优于堆叠融合，且“最后引入与查询匹配的模态”的调度策略在R@1指标上表现最佳，为融合顺序选择提供了两个互补的指导原则。

    

    多模态视频检索结合了视觉、声学和语音信息，但其融合顺序应如何依赖于数据和检索查询仍不清楚。我们提出了统一渐进表示学习顺序分析框架（UniPRO），通过在匹配的模型组件和训练设置下使用可配置的融合调度来研究这一问题。从文本到视频检索任务出发，我们发现所测试的逐一融合调度优于堆叠式融合，而在逐一融合中改变模态顺序也会改变性能。为指导顺序选择，我们考察了两个互补的考虑因素：模态可用性不均情况下的学习难度，以及与查询相关信息的保留。完整模态的对照实验表明，仅凭输入缺失无法解释顺序敏感性。在R@1指标上，文本和音频检索更倾向于最后引入与查询匹配模态的调度……

    arXiv:2610.35125v1 Announce Type: new  Abstract: Multimodal video retrieval combines visual, acoustic, and speech information, yet how their fusion order should depend on the data and the retrieval query remains unclear. We introduce Unified Progressive Representation Learning for Order Analysis, or UniPRO, to investigate this question through configurable fusion schedules with matched model components and training settings. Starting from text-to-video retrieval, we find that the tested one-by-one schedules outperform stacked fusion, while changing the order within one-by-one fusion also changes performance. To guide order selection, we examine two complementary considerations: learning difficulty under unequal modality availability and preservation of information relevant to the query. Complete-modality controls show that missing inputs alone cannot explain order sensitivity. At R@1, text and audio retrieval favor schedules that introduce the modality matching the query last. Controll
    
[^15]: 有用的多样性还是幻觉的多样性？面向电商搜索的人格条件化合成查询生成

    Useful Diversity or Hallucinated Variety? Persona-Conditioned Synthetic Query Generation for E-Commerce Search

    [https://arxiv.org/abs/2610.35102](https://arxiv.org/abs/2610.35102)

    该研究通过人格条件化提示生成大规模合成查询来冷启动电商语义解析器，发现IPIP人格捆绑包可显著提升解析器F1、语法条件化能引导生成结果更贴近真实流量，并提出对齐sKL作为解析器性能的无标注代理指标。

    

    合成监督可以为新的搜索领域引导语义解析器的冷启动，但提示诱导的多样性只有在生成内容忠实于结构化意图时才有用。我们在冷启动电商搜索场景中对这一问题进行了探索：使用6个大语言模型配合9个人格条件化提示词家族，在47字段的英国汽车模式上生成了118,098个合成查询对，并与1,214条人工标注查询和15,304条生产查询进行对比。我们表明，IPIP人格捆绑包在所有四组匹配对比中都提升了语义解析器的F1（经Holm校正的p<.001），并且在完整的2³网格上，所有三个捆绑包都具有正向主效应。语法条件化是将生成结果推向真实流量分布的关键维度，同时也能诱导词汇多样性。对齐sKL——即分别拟合到合成查询与人工标注查询上的von Mises-Fisher分布之间的散度——是解析器F1的一种无标注代理指标（与为模拟购物者场景编写的人工查询相比，ρ=-0.917……

    arXiv:2610.35102v1 Announce Type: new  Abstract: Synthetic supervision can bootstrap semantic parsers in new search domains, but prompt-induced diversity is useful only when generations are faithful to the structured intent. We explore this for cold-start e-commerce search: 118,098 synthetic pairs produced by 6 LLMs with 9 persona-conditioned prompt families on a 47-field UK automotive schema, against 1,214 human-annotated and 15,304 production queries. We show that the IPIP personality bundle improves parser F1 in all four matched contrasts (Holm-corrected $p<.001$), and all three bundles have a positive main effect over the complete $2^3$ grid. Syntax conditioning is the axis that moves generations towards real traffic. It also induces lexical diversity. Alignment sKL, the divergence between von Mises-Fisher distributions fitted to synthetic and human-annotated queries, is an unlabelled proxy for parser F1 ($\rho=-0.917$ against human queries written for simulated shopper scenarios, 
    
[^16]: 学会记忆：大语言模型记忆机制研究进展

    Learning to Remember: Advances in Memory Mechanisms for Large Language Models

    [https://arxiv.org/abs/2610.35089](https://arxiv.org/abs/2610.35089)

    本综述从记忆生命周期、存储基底和系统角色三个维度系统梳理了大语言模型记忆机制的研究，并将其划分为非参数化记忆、参数化记忆和面向智能体进化的记忆三大机制家族。

    

    记忆已成为将大语言模型（LLM）拓展至固定上下文窗口、无状态交互和静态参数化知识之外的核心机制。本综述从三个互补的维度组织相关文献：记忆的生命周期、存储基底，以及记忆在基于LLM的智能体中所扮演的系统角色，同时将受认知科学启发的记忆概念作为轻量级的概念性动机，而非作为独立的技术贡献或实现类别。在此基础上，我们区分了三种机制家族：非参数化记忆，即信息以文本、图、多模态或类数据库结构的形式存储在模型参数之外；参数化记忆，即信息通过激活、键值缓存、可学习模块、基于适配器的参数化或模型权重进行存储或访问；以及面向智能体进化的记忆，即存储的经验支持强化学习等过程。

    arXiv:2610.35089v1 Announce Type: new  Abstract: Memory has become a central mechanism for extending large language models (LLMs) beyond fixed context windows, stateless interaction, and static parametric knowledge. This survey organizes the literature around three complementary dimensions: the life cycle of memory, the storage substrate, and the system role that memory plays in LLM-based agents, while using cognitive-science-inspired memory concepts as a lightweight conceptual motivation rather than as a separate technical contribution or implementation family. On this basis, we distinguish three mechanism families: nonparametric memory, where information is stored outside model parameters in text, graph, multimodal, or database-like structures; parametric memory, where information is stored or accessed through activations, key-value caches, learned modules, adapter-based parameterizations, or model weights; and memory for agent evolution, where stored experiences support reinforcemen
    
[^17]: 无需人工逐行转录读取霍诺留斯三世教皇登记册：强制对齐的手写文本识别——端到端、模块化或混合式方法

    Reading the Registers of Honorius III without Manual Line Transcription: Forced-Aligned HTR, End to End, Modular or Hybrid

    [https://arxiv.org/abs/2610.35085](https://arxiv.org/abs/2610.35085)

    本文提出利用强制对齐技术从无行对齐的校勘本底稿中自动生成训练数据，并采用结合CTC与双向子词注意力解码器的混合Conformer模型，实现了无需人工逐行转录即可对教皇霍诺留斯三世登记册中高度缩写的拉丁手稿进行端到端转录与缩写展开。

    

    中世纪档案的大规模开发利用受限于逐行外交式转录与人工缩写展开的高昂成本。本文聚焦于教皇霍诺留斯三世（1216-1227年）的登记册——其中超过5,000封信件仅以手稿图像的形式完整保存——我们证明，现有的不含换行信息的专家转录文本（校勘本底稿）已足以训练端到端的手写文本识别（HTR）模型，实现高度缩写拉丁文的同步转录与缩写展开。通过对《Reg. Vat. 9》394页中的860条未经对齐的校勘本条目应用强制对齐技术，我们在无需人工对齐分行文本的情况下，自动构建了13,122条训练行。通用的CATMuS基线模型在该领域表现不佳（字符错误率CER为18.9%），模块化的无空格HTR加词切分流水线（参见Camps等人2021年工作）也止步于15.6%的词错误率（WER），而我们从零训练的混合Conformer模型——将CTC与双向子词注意力解码器及迭代自重对齐相结合——（摘要在此处被截断）

    arXiv:2610.35085v1 Announce Type: cross  Abstract: Unlocking medieval archives at scale is bottlenecked by the cost of line-level diplomatic transcriptions and manual abbreviation expansion. Focusing on the registers of Pope Honorius III (1216-1227) over 5,000 letters surviving in full only as manuscript images - we show that existing, line-break-free expert transcriptions (edition drafts) suffice to train end-to-end hand-written text recognition (HTR) that jointly transcribes and expands heavily abbreviated Latin. Using forced alignment on 860 unaligned edition entries across 394 pages of Reg. Vat. 9, we automatically bootstrap 13,122 training lines without manual alignment of segmented lines. While the generic CATMuS baseline struggles on this domain (18.9% CER) and a modular space-free HTR + word-segmentation pipeline (cf. Camps et al. 2021) plateaus at 15.6% WER, our from-scratch hybrid Conformer - combining CTC with bidirectional subword attention decoders, iterative self-realignm
    
[^18]: 基于内部注意力奖励的幻觉感知多智能体检索增强生成

    Hallucination-Aware Multi-Agent RAG via Internal Attention Rewards

    [https://arxiv.org/abs/2610.35056](https://arxiv.org/abs/2610.35056)

    该论文提出一种幻觉感知的多智能体RAG框架，通过从生成器注意力头范数中提取内部幻觉风险信号并与共享任务奖励结合，在协作PPO训练中有效提升生成答案对检索证据的忠实性。

    

    检索增强生成（RAG）将语言模型建立在检索到的证据之上，但其输出常常包含证据无法支持的声明。近期的多智能体RAG系统将整个流程分解为多个协作决策，然而它们通常在共享的任务奖励（如EM或F1）下优化所有智能体，这类奖励衡量的是答案的重叠程度，而非对证据的忠实性。我们提出了一种幻觉感知的多智能体RAG框架，将任务奖励与从生成器注意力头范数计算得出的内部幻觉风险信号相结合。三个角色特定的LoRA适配器——查询规划器、证据选择器和答案生成器——共享一个冻结的主干网络。一个轻量级评论家模型仅在TruthfulQA的真实/虚假样本对上训练一次后即保持冻结，提供一个互补的风险评分，并在协作PPO训练过程中将该评分加入共享奖励中。在三个主干模型（Qwen3-8B、Llama3-8B、Llama2-13B）上……

    arXiv:2610.35056v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) grounds language models in retrieved evidence, but its outputs often contain claims unsupported by that evidence. Recent multi-agent RAG systems decompose the pipeline into cooperative decisions, yet they typically optimize all agents under a shared task reward such as EM or F1, which measures answer overlap rather than faithfulness to evidence. We propose a hallucination-aware multi-agent RAG framework that combines the task reward with an internal hallucination-risk signal computed from the generator's attention-head norms. Three role-specific LoRA adapters, a query planner, an evidence selector, and an answer generator, share a frozen backbone. A lightweight critic is trained once on TruthfulQA truthful/false pairs and then kept frozen, providing a complementary risk score that is added to the shared reward during cooperative PPO training. Across three backbones (Qwen3-8B, Llama3-8B, Llama2-13B) an
    
[^19]: CA-HNM：基于结构化领域模型的约束感知难负例挖掘用于稠密检索

    CA-HNM: Constraint-Aware Hard Negative Mining for Dense Retrieval with Structured Domain Models

    [https://arxiv.org/abs/2610.35045](https://arxiv.org/abs/2610.35045)

    CA-HNM通过将结构化领域模型的确定性不兼容规则与检索候选生成相结合，为稠密检索挖掘更符合领域结构约束的难负例，从而在多数数据集与编码器组合上提升了检索效果。

    

    稠密检索器通常使用从词法检索或稠密检索结果中选取的难负例进行训练。然而，检索邻近度并不能表明候选样本是否与显式的领域结构相冲突。我们提出了CA-HNM，这是一个约束感知的难负例挖掘框架，它将基于检索的候选生成与从结构化领域模型推导出的确定性不兼容规则相结合。该框架移除已知的正例，围绕查询目标构建局部类型化上下文，记录关系、范围、目标和层级的错配情况，并使用由此产生的候选样本对标准双编码器进行微调。结构化信息仅在离线负例构建阶段需要。我们在MOOCCubeX和Course-Skill Atlas数据集上，使用两种编码器、两种训练损失和十个随机种子对CA-HNM进行了评估。相对于检索排名匹配的对照组，CA-HNM-Pure在八个数据集-编码器组合中的六个上取得了更高的平均NDCG@10。

    arXiv:2610.35045v1 Announce Type: cross  Abstract: Dense retrievers are commonly trained with hard negatives selected from lexical or dense retrieval results. Retrieval proximity, however, does not indicate whether a candidate conflicts with explicit domain structure. We introduce CA-HNM, a constraint-aware hard-negative mining framework that combines retrieval-based candidate generation with deterministic incompatibility rules derived from a structured domain model. The framework removes known positives, constructs a local typed context around the query target, records relational, scope, target, and level mismatches, and uses the resulting candidates to fine-tune a standard dual encoder. Structured information is required only during offline negative construction. We evaluate CA-HNM on MOOCCubeX and Course-Skill Atlas with two encoders, two training losses, and ten seeds. Relative to a retrieval-rank-matched control, CA-HNM-Pure has higher mean NDCG@10 in six of eight dataset--encoder
    
[^20]: SLIDE：面向单次生成式检索的搜索到列表物品蒸馏

    SLIDE: Search-to-List Item Distillation for Single-Rollout Generative Retrieval

    [https://arxiv.org/abs/2610.34764](https://arxiv.org/abs/2610.34764)

    SLIDE提出了一种将离线束搜索与在线列表生成解耦的物品蒸馏框架，利用教师模型检索的高质量候选提供额外监督，从而解决单次自回归生成列表时性能不佳的问题。

    

    生成式检索（GR）通常依赖物品级的束搜索来检索候选结果，这使得跨物品的依赖关系未被建模，并限制了对列表级目标的直接优化。一种自然的替代方案是在单条自回归轨迹中生成整个列表。然而，使用直接训练的列表级模型进行贪心解码时，其性能显著逊于束搜索。我们识别出两个原因：束搜索所探索的更广阔的搜索空间，以及稀缺正信号带来的位置偏差，后者使得靠后的位置缺乏充分的监督。为应对这些挑战，我们提出了SLIDE，这是一种将离线搜索与在线列表生成解耦的搜索到列表物品蒸馏框架。教师模型通过离线的物品级束搜索检索高质量候选，在稀疏的行为标签之外提供额外的与转化相关的监督。其语义ID（SID...

    arXiv:2610.34764v1 Announce Type: new  Abstract: Generative retrieval (GR) typically relies on item-level beam search to retrieve candidates, leaving cross-item dependencies unmodeled and limiting direct optimization of list-level objectives. A natural alternative is to generate the entire list within a single autoregressive trajectory. However, greedy decoding with directly trained list-wise models substantially underperforms beam search. We identify two contributing factors: the broader search space explored by beam search and the positional bias of scarce positive signals, which deprives later positions of adequate supervision. To address these challenges, we propose $\textbf{SLIDE}$, a search-to-list item distillation framework that decouples offline search from online list generation. A teacher model retrieves high-quality candidates through offline item-level beam search, providing additional conversion-relevant supervision beyond sparse behavioral labels. Their Semantic IDs (SID
    
[^21]: 从线性自编码器中发现主题

    Discovering Topics from Linear Autoencoders

    [https://arxiv.org/abs/2610.34739](https://arxiv.org/abs/2610.34739)

    该论文证明潜在主题可以直接从闭式解线性自编码器的权重矩阵中提取，无需额外训练即可解释推荐结果，并通过新颖的物品和用户归一化方法实现超越专用主题模型的主题质量，同时保持有竞争力的推荐性能。

    

    线性推荐器和主题模型的相似之处在于它们都隐式地从物品的共现关系中学习。然而，关于这两个模型家族之间关系的研究却很少。在这项工作中，我们证明，在用户-物品交互的潜在因子模型下，潜在主题可以从闭式解线性自编码器的权重矩阵中发现。这些主题为解释推荐结果提供了基础，并且可以直接从训练好的推荐器中获取，无需额外的训练。基于我们的分析，我们进一步提出了新颖的物品和用户归一化方法，以防止热门物品和极度活跃的用户主导主题。我们在六个公开数据集上的实验表明，我们的方法获得了显著高于所有专用主题模型的主题质量，并且在所有六个数据集上匹配或超过了最好的线性自编码器，同时其推荐性能也具有竞争力。

    arXiv:2610.34739v1 Announce Type: new  Abstract: Linear recommenders and topic models are similar in that they implicitly learn from co-occurrences of items. However, there has been little study of how these two model families are related. In this work, we show that, under a latent-factor model of the user-item interactions, the latent topics can be discovered from the weight matrix of a closed-form linear autoencoder. These topics provide a basis for interpreting the recommendations, and can be obtained directly from the trained recommender without additional training. Building on our analysis, we further propose novel item and user normalizations that prevent popular items and very active users from dominating the topics. Our experiments on six public datasets demonstrate that our method achieves significantly higher topic quality than every dedicated topic model and matches or exceeds the best linear autoencoder on all six datasets, while its recommendation performance is competitiv
    
[^22]: DPE-Rec：面向捆绑包构建的多样性保持且高效的扩散语言模型推荐

    DPE-Rec: Diversity-Preserving and Efficient Diffusion Language Model Recommendation for Bundle Construction

    [https://arxiv.org/abs/2610.34700](https://arxiv.org/abs/2610.34700)

    提出DPE-Rec扩散语言模型推荐框架，通过解决置信度解码导致的语义分叉决策延迟和多词元并行解码的质量下降问题，在保持捆绑包多样性的同时实现高效推理。

    

    生成式捆绑包构建通过从大型商品目录中生成缺失的商品来完成部分观测到的捆绑包。与自回归大语言模型相比，扩散大语言模型（DLM）天然适合这种无序生成任务。其双向注意力提供了一种与无序捆绑包构建更加契合的归纳偏置；迭代去噪可以每步揭示多个词元，有望降低推理延迟。然而，直接将扩散语言模型应用于该任务存在两个关键挑战：1）尽管揭示顺序灵活，但基于置信度引导的解码往往会推迟“语义分叉”决策，直到精化词元限制了其选择空间，从而降低捆绑包的多样性；2）每步揭示更多词元可能损害推荐质量，抑制更快推理的潜力。我们提出了DPE-Rec，一个实现多样性保持与高效推荐的扩散框架。DPE-Rec通过…（摘要在此处被截断）

    arXiv:2610.34700v1 Announce Type: new  Abstract: Generative bundle construction completes a partially observed bundle by generating its missing items from a large product catalog. Compared to autoregressive LLMs, diffusion LLM emerges as a natural fit for this unordered generation task. Their bidirectional attention provides an inductive bias better aligned with unordered bundle construction. Iterative denoising can reveal multiple tokens per step, potentially reducing inference latency. However, directly applying DLMs to this task presents two key challenges: 1 Despite its flexible reveal order, confidence-guided decoding tends to postpone semantic-fork decisions until refinement tokens constrain their choices, reducing bundle diversity. 2 Revealing more tokens per step can compromise recommendation quality, stifling the potential for faster inference. We introduce DPE-Rec, a diffusion framework for diversity preservation and efficient recommendation. DPE-Rec preserves diversity by in
    
[^23]: 知道证据何时足够：基于Horn闭包证书的需求导向搜索

    Knowing When the Evidence Is Enough: Requirement-Directed Search with Horn-Closure Certificates

    [https://arxiv.org/abs/2610.34684](https://arxiv.org/abs/2610.34684)

    该论文提出Certify框架，通过将查询分解为需求超图并利用Horn闭包运算生成证书，把搜索增强模型“证据是否足够”的判断结构化，从而可靠地决定回答、继续搜索或放弃回答并指出缺失的需求。

    

    每次网页搜索之后，搜索增强的模型必须判断其证据是否充分：回答问题、再次搜索，或放弃回答。现有系统将这一判断交给模型的整体性直觉判断，这可能导致模型接受一个虽然提及了所有相关实体、却缺少某个所需事实的上下文。我们转而将“充分性”结构化：针对每个查询构建需求超图，用以声明哪些事实可以共同推导出另一个事实。这带来了两个挑战：仅凭问题本身推导出这些需求，以及将每条需求链接到关于正确实体的文本段落。为此，Certify 对问题进行分解，按需求逐项搜索，并且只有在通过实体检查之后才接受链接。对这些链接进行Horn闭包运算即可产生一个证书——即最小充分段落集及其推导过程；若无法满足，则放弃回答并明确指出尚未满足的需求。在检索、需求和链接均正确的前提下，该决策是可靠的，且在实体检查的范围内是完备的。在包含不可回答孪生问题的三个语料库上的汇总结果显示，Certi（摘要在此处截断）

    arXiv:2610.34684v1 Announce Type: cross  Abstract: After each Web search, a search-augmented model must decide whether its evidence suffices: answer, search again, or abstain. Prevailing systems leave this to holistic model judgement, which can accept a context naming every entity yet lacking one required fact. We instead make sufficiency structural: a per-query requirement hypergraph states which facts jointly yield another fact. This raises two challenges: deriving requirements from the question alone, and linking each requirement to passages about the right entity. Certify therefore decomposes the question, searches per requirement, and accepts links only after entity checks. Horn closure over the links yields a certificate, the minimal sufficient passages and their derivation, or an abstention naming open requirements. Given correct retrieval, requirements and links, the decision is sound, and complete up to the entity check. Pooled over three corpora with unanswerable twins, Certi
    
[^24]: 智能体记忆即推荐：面向大语言模型智能体长期记忆的推荐系统设计

    Agent Memory as Recommendation: Recommender-System Design for Long-Term Memory of LLM Agents

    [https://arxiv.org/abs/2610.34673](https://arxiv.org/abs/2610.34673)

    该论文的核心创新是将LLM智能体的记忆选择问题形式化为预算约束下的板块推荐问题（请求为用户、记忆为物品、上下文窗口为板块），从而把各种智能体记忆系统统一纳入推荐级联的设计空间，并将准确性分解为可分析的组件。

    

    在多次会话中为用户提供服务的大语言模型（LLM）智能体，必须在每次请求时从存储的记忆中挑选出少数几条进入有限的上下文窗口。这一选择在很大程度上决定了智能体能否正确回答问题，并且需要在网络规模下对每次请求执行。这一选择面临三个挑战：庞大且不断增长的存储库、每次请求的成本预算限制，以及需要全部证据的多会话问题。许多记忆系统在构建笔记或图结构时，每份历史记录需要数百到数千次LLM调用，且各系统都是独立设计的，因此难以确定究竟是哪个组件带来了准确性。这些挑战与推荐系统的特性高度相似，因此我们将智能体记忆形式化为预算约束下的板块推荐：请求对应用户，存储的记忆对应物品，上下文窗口对应板块。这一形式化将记忆系统置于推荐级联的统一设计空间中，并将准确性分解为完整性……

    arXiv:2610.34673v1 Announce Type: cross  Abstract: Large language model (LLM) agents that serve users across many sessions must choose, for every request, the few stored memories that enter a limited context window. This choice largely decides whether the agent answers correctly, and it runs on every request at Web scale. This selection faces three challenges: a large and growing store, a per-request cost budget, and multi-session questions that need all of their evidence. Many memory systems build notes or graphs with hundreds to thousands of LLM calls per history, and each is designed on its own, so it is unclear which component yields the accuracy. These challenges mirror the properties of recommendation, so we formalize agent memory as budgeted slate recommendation, with the request as the user, stored memories as items and the context window as a slate. The formalization places memory systems in one design space of recommender cascades and factorizes accuracy into the completeness
    
[^25]: HARP-LoRA：面向科学表格理解的层次条件化秩投影低秩自适应方法

    HARP-LoRA: Hierarchy-Conditioned Rank Projection for Low-Rank Adaptation in Scientific Table Understanding

    [https://arxiv.org/abs/2610.34589](https://arxiv.org/abs/2610.34589)

    该论文提出HARP-LoRA方法，利用表格行表头和列表头的层次结构为每个数据标记构建条件化表示并作用于低秩下投影，从而在参数高效的微调中提升科学表格（尤其是药代动力学表格）的理解能力。

    

    表格数据广泛应用于科学领域，其中复杂的版式布局——包括多级表头、合并单元格、标题和脚注——使得信息提取变得困难。这在药代动力学（PK）中尤为重要，因为其中的数值必须与药物、物种、剂量、给药途径、单位和实验条件正确关联。LoRA（低秩自适应）等参数高效微调方法虽然降低了适配成本，但并未显式地融合表格数值与其周围表头之间的关系。我们提出了HARP-LoRA，这是一种用于科学表格理解的层次条件化秩投影低秩自适应方法。HARP-LoRA通过从每个数据标记所对应的行表头和列表头层次结构中构建标记特定的内在表示，对投影式LoRA进行了扩展。这些表示对低秩下投影进行条件化，从而在保持参数高效的同时融入层次化上下文信息。

    arXiv:2610.34589v1 Announce Type: cross  Abstract: Tabular data is widely used in scientific applications, where complex layouts, including multi-level headers, merged cells, captions, and footnotes, make information extraction difficult. This is particularly important in pharmacokinetics (PK), where values must be correctly associated with the drug, species, dose, administration route, unit, and experimental condition. Parameter-efficient fine-tuning methods such as Low-Rank Adaptation (LoRA) reduce adaptation cost but do not explicitly incorporate relationships between table values and surrounding headers. We introduce HARP-LoRA, a hierarchy-conditioned rank projection method for low-rank adaptation in scientific table understanding. HARP-LoRA extends projected LoRA by constructing token-specific intrinsic representations from the row- and column-header hierarchies of each data token. These representations condition the low-rank down-projection, incorporating hierarchical context whi
    
[^26]: RefinedGAW：一个精炼的德国学术网络数据集

    RefinedGAW: A Refined German Academic Web Dataset

    [https://arxiv.org/abs/2610.34572](https://arxiv.org/abs/2610.34572)

    RefinedGAW是一个基于26次德国学术网络爬取构建的德语数据集，包含1.22亿份文档和3140亿词元，其中876亿学术词元超越了规模五倍于它的FineWeb2与FinePDFs德语子集，并为教育、数学和专家导向内容提供了无商业偏见的互补覆盖。

    

    arXiv:2610.34572v1 公告类型：新论文 摘要：最大的开放德语文本数据集通常构建自广泛的网络爬取，这类爬取提供了规模优势，但对特定领域的覆盖有限。我们推出了RefinedGAW，这是一个基于26次半年度德国学术网络爬取经过处理和标注的资源，总计1.22亿份文档和3140亿个词元，并保留了每篇文本完整的爬取和捕获历史。在HTML和PDF格式中，被标注为学术研究的文档占德语RefinedGAW词元的84%。其876亿个词元超过了FineWeb2和FinePDFs德语子集的649亿词元，尽管后者的总规模几乎是前者的五倍。RefinedGAW还包含大量教育性、数学密集型和面向专家的内容，且商业偏见极小。在HTML和PDF格式中，88%的德语RefinedGAW文档在三个被评估的参考数据集中没有近似重复的匹配，这支持了其互补性覆盖。我们发布了一个查询（接口）……

    arXiv:2610.34572v1 Announce Type: new  Abstract: The largest open German text datasets are built from broad web crawls, which provide scale but limited coverage of specific domains. We introduce RefinedGAW, a processed and annotated resource derived from 26 semi-annual crawls of the German Academic Web, totaling 122 million documents and 314 billion tokens, with full crawl and capture history of every text. Documents labeled as academic research account for 84% of German RefinedGAW tokens across HTML and PDF. At 87.6 billion tokens, this exceeds the 64.9 billion tokens in the German subsets of FineWeb2 and FinePDFs, despite their almost five-times-larger combined size. RefinedGAW also contains substantial amounts of educational, math-heavy, and expert-oriented content with minimal commercial bias. Across HTML and PDF, 88% of German RefinedGAW documents have no near-duplicate candidate match in the three evaluated reference datasets, supporting complementary coverage. We release a query
    
[^27]: 主动记忆召回中规模的代价

    The Price of Scale in Proactive Memory Recall

    [https://arxiv.org/abs/2610.34507](https://arxiv.org/abs/2610.34507)

    该论文证明主动记忆召回的门控决策本质上是一个随记忆库规模 N 增长的多重假设检验问题——单条记忆的假阳性率按 -ln(1-α)/N 衰减，导致命中率随 N 呈幂次下降，从理论与实证上解释了记忆库越大、主动记忆触发越容易失效的现象。

    

    一个无需被询问便能主动提起记忆的长期助手，必须在每轮交互中判断其记忆库中是否有内容值得打断用户。我们证明这一决策本质上是一个多重假设检验问题，其难度随记忆库规模 N 的增长而加剧。在 LongMemEval 与 LoCoMo 的流式回放实验中，一个在 N=8 时调校至 5% 误报率的相似度阈值，在 N=2048 时误报率飙升至 87–100%，该现象在各类嵌入模型以及显式与隐式线索中均成立；LLM 判别器虽起点更低，也呈现同样的增长趋势。我们证明，一个误报率为 α 的门控机制，其单条记忆的假阳性率约为 -ln(1-α)/N，这将把其命中率锁定在单次比较 ROC 曲线的最左端，且在正则变化条件下，命中率随 N 以幂次规律衰减并带有显式常数。由此得出的预测与逐 N oracle 阈值的命中率相差在一个百分点以内。该分析解释了为何词汇与语义通道……（原文摘要在此处截断）

    arXiv:2610.34507v1 Announce Type: cross  Abstract: A long-lived assistant that brings up memories without being asked must decide at every turn whether anything in its store is worth an interruption. We show that this decision is a multiple-testing problem whose difficulty grows with the store size $N$. On streaming replays of LongMemEval and LoCoMo, a similarity threshold tuned to 5% false alarms at $N=8$ reaches 87-100% at $N=2048$, across embedding models and for explicit and implicit cues; LLM judges show the same growth from a lower base. We prove that a gate with false-alarm rate $\alpha$ operates at a per-item false-positive rate of about $-\ln(1-\alpha)/N$, which pins its hit rate to the far left of the single-comparison ROC curve and, under regular variation, makes it decay as a power of $N$ with an explicit constant. The resulting prediction matches the hit rate of a per-$N$ oracle threshold to within one percentage point. The analysis explains why lexical and semantic channe
    
[^28]: 工业环境中的多模态需求可追溯性链接恢复

    Multimodal Requirements Traceability Link Recovery in Industrial Settings

    [https://arxiv.org/abs/2610.34503](https://arxiv.org/abs/2610.34503)

    该论文提出了一种多模态RAG扩展方法，通过分别检索文本和图像候选、利用视觉-语言模型进行分类并用言语化置信度过滤预测链接，从而在工业需求可追溯性链接恢复任务中将F1分数从基线的0.114提升至0.310。

    

    需求可追溯性链接可以支持变更影响分析、覆盖率检查和验证等活动。工业需求中可能包含可追溯的图像，而纯文本方法无法直接处理这些图像。对于固定的分类器配置而言，仅依靠二元决策无法直接调整漏检链接与错误链接之间的平衡。我们扩展了一种基于检索增强生成（RAG）的需求间可追溯性链接恢复（TLR）方法。我们的扩展方法分别检索文本和图像候选，使用视觉-语言模型对每个检索到的配对进行分类，并通过言语化置信度对预测的链接进行过滤。我们在三个互不重叠的工业需求集合和六个公开的纯文本数据集上评估了我们的方法。在工业测试集上，我们的方法取得了0.310的F1分数，而我们对Hey等人基线方法的复现仅达到0.114。我们的方法恢复的链接（原文在此处截断）

    arXiv:2610.34503v1 Announce Type: cross  Abstract: Requirements traceability links support activities such as change-impact analysis, coverage checking, and validation. Industrial requirements can include traceable images, which text-only approaches cannot process directly. For a fixed classifier configuration, binary decisions alone offer no direct way to adjust the balance between missed and incorrect links. We extend a retrieval-augmented generation (RAG) approach for inter-requirements traceability link recovery (TLR). Our extension retrieves text and image candidates separately, classifies each retrieved pair with a vision-language model, and filters the predicted links by verbalized confidence. We evaluate our approach on three non-overlapping industrial requirement sets and six public text-only datasets. On the industrial test sets, our approach achieves an F1 score of 0.310, compared with 0.114 for our replication of the baseline approach by Hey et al. Our approach recovers lin
    
[^29]: 面向推荐系统的理论可靠的物品相似度度量

    Theoretically Sound Item Similarity Measures for Recommender Systems

    [https://arxiv.org/abs/2610.34402](https://arxiv.org/abs/2610.34402)

    该论文提出了一种基于概率建模与参数估计的原则性方法来设计物品相似度度量，推导出基于伯努利分布和多项式分布的两种新度量，并证明常用的余弦相似度也可在此理论框架下得到解释。

    

    量化物品之间的相似度是基于邻域的协同过滤的基础，然而许多常用的相似度度量都是缺乏理论依据的启发式方法。使用启发式方法迫使从业者依赖经验性的试错，既无法深入洞察为什么某些度量方法优于其他方法，也无法了解如何改进它们。在本文中，我们提出了一种基于概率建模和参数估计的、有原则的物品相似度度量设计方法。通过采用这一形式化视角，我们推导出了两种此前被忽视的、基于伯努利分布和多项式分布的相似度度量。通过广泛的消融研究，我们强调了偏差、概率平滑、正确处理未知量以及选择合适概率分布的关键作用。最后，我们表明常用的余弦相似度可以被解释为该框架下的一个求解结果。

    arXiv:2610.34402v1 Announce Type: new  Abstract: Quantifying the similarity between items is fundamental to neighborhood-based collaborative filtering, yet many commonly used similarity measures are heuristics that lack theoretical justification. The use of heuristics forces practitioners to rely on empirical trial-and-error, offering little insight into why certain measures outperform others or how they might be improved. In this paper, we present a principled approach for designing item similarity measures grounded in probabilistic modeling and parameter estimation.   By adopting this formal perspective, we derive two previously overlooked similarity measures based on the Bernoulli and multinomial distributions. Through an extensive ablation study we highlight the critical role of bias, probabilistic smoothing, treating the unknowns correctly, and choosing appropriate probability distributions. Finally, we show that the commonly used cosine similarity can be interpreted as the soluti
    
[^30]: 面向超导领域的arXiv文献结构化抽取与检索增强问答流水线

    A Domain-Aware Pipeline for Structured Extraction and Retrieval-Augmented Question Answering over arXiv Superconductivity Literature

    [https://arxiv.org/abs/2610.34391](https://arxiv.org/abs/2610.34391)

    本文提出了一个面向超导领域的端到端流水线，通过结构化抽取材料与临界温度信息，并将领域属性融入混合检索，从而提升了arXiv超导文献检索增强问答的准确性与可靠性。

    

    材料物理学领域的科学文献增长速度已超过研究人员人工阅读的能力，然而大多数检索工具仍依赖关键词匹配或通用问答方法，忽略了PDF文件中蕴含的领域特定结构信息。本文提出了一个端到端的流水线，将arXiv超导领域论文转化为带有结构化元数据的可检索语料库，并提供检索增强问答能力。该系统结合了主题语料库构建、面向领域的材料及超导转变温度抽取，以及将稠密文本块嵌入与结构化论文字段相融合的混合检索方法。我们验证了相对于仅使用词法匹配和仅使用稠密文本的方法，归一化的领域属性能否提升证据检索质量和答案的有据性。抽取效果在一个经过人工验证、包含59种超导材料的黄金标准数据集上进行了评估，其中46种材料具有已确认的实验临界温度。

    arXiv:2610.34391v1 Announce Type: cross  Abstract: Scientific literature in materials physics is growing faster than researchers can manually inspect it, yet most retrieval tools rely on keyword matching or generic question answering that ignores domain-specific structure in PDF files. This paper presents an end-to-end pipeline that transforms arXiv superconductivity publications into a searchable corpus with structured metadata and retrieval-augmented question answering. The system combines thematic corpus construction, domain-aware extraction of materials and superconducting transition temperatures, and hybrid retrieval that merges dense chunk embeddings with structured publication fields. We test whether normalized domain attributes improve evidence retrieval and answer groundedness relative to lexical-only and dense text-only methods. Extraction is evaluated on a manually verified gold set of 59 superconducting materials, 46 of which have confirmed experimental critical temperature
    
[^31]: 面向即席搜索的德顿语跨语言信息检索

    Cross-Lingual Information Retrieval in Tetun for Ad-Hoc Search

    [https://arxiv.org/abs/2610.34362](https://arxiv.org/abs/2610.34362)

    本研究首次针对东帝汶多语言环境探索了德顿语跨语言信息检索的可行性，通过将葡萄牙语、英语和印尼语查询翻译回德顿语进行文档检索，发现谷歌翻译是目前德顿语最可靠的自动翻译工具。

    

    开发能够支持多语言访问的信息检索（IR）系统在多语言环境中至关重要。在东帝汶，德顿语、葡萄牙语、英语和印尼语是官方语言和工作语言，但目前尚无跨语言信息检索（CLIR）解决方案来支持这些语言之间的信息访问。本研究通过探索针对东帝汶语言环境量身定制的CLIR方法来填补这一空白。利用现有的德顿语单语文档集合和即席文本检索基线，我们探索了德顿语CLIR的可行性。查询被人工翻译成葡萄牙语、英语和印尼语以构建多语言查询集，随后使用谷歌翻译和多个大语言模型将这些查询自动翻译回德顿语，并用于检索德顿语文档。结果表明，谷歌翻译是目前处理德顿语最可靠的翻译工具。

    arXiv:2610.34362v1 Announce Type: new  Abstract: Developing information retrieval (IR) systems that enable access across multiple languages is crucial in multilingual contexts. In Timor-Leste, where Tetun, Portuguese, English, and Indonesian are official and working languages, no cross-lingual information retrieval (CLIR) solutions currently exist to support information access across these languages. This study addresses that gap by investigating CLIR approaches tailored to the linguistic landscape of Timor-Leste. Leveraging an existing monolingual Tetun document collection and ad-hoc text retrieval baselines, we explore the feasibility of CLIR for Tetun. Queries were manually translated into Portuguese, English, and Indonesian to create a multilingual query set. These were then automatically translated back into Tetun using Google Translate and several large language models, and used to retrieve documents in Tetun. Results show that Google Translate is the most reliable tool for Tetun
    
[^32]: 事实正确却立场偏颇：检索增强生成中的立场投毒与证据平衡

    Factual Yet One-Sided: Stance Poisoning and Evidence Balancing in Retrieval-Augmented Generation

    [https://arxiv.org/abs/2610.34294](https://arxiv.org/abs/2610.34294)

    本文揭示了一种新型立场投毒攻击——注入事实正确但立场片面的材料即可使检索增强生成系统的输出立场随之偏移，并提出一个无需额外大模型调用的立场平衡中间件，通过立场检测、固定配比的证据平衡与去偏提示词来消除这一偏见。

    

    检索增强生成系统回复中所表达的立场，显著受检索材料的立场分布影响。现有关于知识投毒的研究主要针对事实性错误，而本研究关注的是一种事实正确、但在争议议题上立场片面的注入攻击。实验表明，在注入后的 ProCon 文档中话题层面平均投毒占比达到 18.9% 时，来自不同厂商的四个生成模型均表现出与投毒方向一致的立场偏移；且在四种集中度水平下，偏移呈单调递增。本文提出了一种位于检索器与生成器之间的立场平衡中间件，由立场检测、固定平衡机制（2条支持 + 2条反对 + 1条中立）以及去偏提示词构成，且不引入额外的大语言模型调用。在全部八种配置下……（原文此处截断）

    arXiv:2610.34294v1 Announce Type: new  Abstract: The stance expressed in responses from retrieval-augmented generation systems is significantly influenced by the stance distribution of the retrieved materials. Existing research on knowledge poisoning has primarily targeted factual errors. In contrast, this study focuses on injection attacks in which the facts are correct but the stance on a controversial issue is one-sided. At a mean topic-level poisoning share of 18.9% within the post-injection ProCon documents, all four generation models from different vendors exhibited stance shifts aligned with the poisoning direction; across four concentration levels, the shifts increased monotonically.   This paper proposes a Stance-Balancing Middleware positioned between the retriever and the generator. It consists of stance detection, fixed balancing (2 supporting + 2 opposing + 1 neutral), and a debiasing prompt, without introducing additional large language model calls. Across all eight conce
    
[^33]: 将业务规则恢复为级联查找以实现企业字段预测

    Recovering Business Rules as Cascade Lookups for Enterprise Field Prediction

    [https://arxiv.org/abs/2610.34258](https://arxiv.org/abs/2610.34258)

    本文提出一种知识图谱引导的LLM工作流，从历史交易中自动恢复企业业务规则并编译为确定性的级联查找，使字段预测在推理时无需模型服务且完全透明可解释。

    

    企业销售订单系统依赖人工维护的配置规则来确定付款条件、发货点和销售归属等字段。当这些规则不完整或过时时，操作人员必须手动输入数值，而下游的错误会扩散到计费、物流和报表中。本文研究是否可以从历史交易中恢复缺失或过时的规则，而不是将其重新学习为不透明的统计模式。我们提出了一种知识图谱引导的工作流，其中LLM智能体提出候选业务键（即诸如客户和单据类型之类的列组合），通过SQL剖析对其进行验证，并将所得证据编译为确定性的级联查找。LLM仅在离线规则发现阶段使用。在推理时，预测是一个透明的键查找过程，无需模型服务支撑。在企业销售订单预测和公开数据集上的实验……

    arXiv:2610.34258v1 Announce Type: cross  Abstract: Enterprise sales-order systems rely on manually maintained configuration rules to determine fields such as payment terms, shipping points, and sales ownership. When these rules are incomplete or stale, operators must enter values by hand, and downstream errors spread into billing, logistics, and reporting. This paper studies whether missing or outdated rules can be recovered from historical transactions instead of being relearned as opaque statistical patterns. We introduce a knowledge-graph-guided workflow in which an LLM agent proposes candidate business keys, meaning column combinations such as customer and document type, validates them through SQL profiling, and compiles the resulting evidence into deterministic cascade lookups. The LLM is used only during offline rule discovery. At inference time, prediction is a transparent key lookup with no model-serving requirement. Experiments on enterprise sales-order prediction and public r
    
[^34]: Δτ-Bench：推动智能体迈向动态变化的能力边界

    $\Delta\tau$-Bench: Pushing Agents Toward a Moving Capability

    [https://arxiv.org/abs/2610.34238](https://arxiv.org/abs/2610.34238)

    Δτ-Bench是一个模拟工具生态系统持续演化的可执行智能体基准测试，包含覆盖42个领域的1,588个多轮任务，评估显示最强模型的任务成功率也仅达42.3%，揭示了智能体在应对工具环境动态变化方面的重大不足。

    

    本文提出了Δτ-Bench，一个可执行的智能体基准测试，用于建模大型工具生态系统的结构演化。在Δτ-Bench中，各自独立演化的服务不断重塑任务的完成方式：熟悉的工具可能消失或被替换，而更高效的替代方案则会在功能重叠和充满噪声的文档中涌现。面对这一不断移动的能力边界，我们评估智能体能否主动发现高效的工具使用路径，并及时修订其积累的程序性知识，从而汇集了现有基准测试未能考察的多重需求。Δτ-Bench包含1,588个经过验证的多轮任务，覆盖42个领域，以及通过能力保持变换生成的六个连续的生态系统状态。我们的评估显示，即使是最强的被评估模型或框架，其任务成功率也仅为42.3%，而现有方法带来的初始收益……

    arXiv:2610.34238v1 Announce Type: cross  Abstract: In this paper, we introduce $\Delta\tau$-Bench, an executable agent benchmark that models the structural evolution of large tool ecosystems. In $\Delta\tau$-Bench, independently evolving services continually reshape how tasks can be accomplished: familiar tools may disappear or be replaced, while more efficient alternatives emerge amid overlapping functionality and noisy documentation. Against this moving capability frontier, we evaluate whether agents can proactively discover efficient tool-use paths and promptly revise their accumulated procedural knowledge, bringing together demands that existing benchmarks fail to examine. $\Delta\tau$-Bench comprises 1,588 validated multi-turn tasks across 42 domains and six successive ecosystem states generated through capability-preserving transformations. Our evaluation shows that even the strongest evaluated model or harness achieves only 42.3% task success, while initial gains from existing s
    
[^35]: MIDA：面向基于会话推荐的多兴趣解码与自适应聚合

    MIDA: Multi-Interest Decoding with Adaptive Aggregation for Session-based Recommendation

    [https://arxiv.org/abs/2610.34192](https://arxiv.org/abs/2610.34192)

    提出MIDA框架，通过双空间解耦设计（在原始物品嵌入上聚类、利用编码器输出进行聚合）重构解码阶段，解决了基于会话推荐中过度关注近期物品、忽视远距离语义相关交互的问题。

    

    基于会话的推荐（SBR）旨在从有限的交互序列中推断用户兴趣，然而当更长的历史记录可用时，现有模型却出现一致的性能下降。这种退化与解码器设计有关：这些设计过度强调近期物品，同时低估了远距离但语义相关的交互，导致对长程依赖和多兴趣行为的建模不佳。为应对这一挑战，我们提出了MIDA（Multi-Interest Decoding with Adaptive Aggregation for Session-based Recommendation），一个轻量级且与编码器无关的框架，对解码阶段进行了重新设计。其核心设计原则是双空间解耦：聚类在原始物品嵌入上进行，使语义分组不会被编码器学习到的关系偏差所扭曲；编码器输出则保留用于聚合，从而保持序列上下文的完整性。具体而言，MIDA（i）采用diff……（摘要在此处被截断）

    arXiv:2610.34192v1 Announce Type: new  Abstract: Session-based recommendation (SBR) infers user interest from limited interaction sequences, yet current models show a consistent performance drop when longer histories are available. This degradation is associated with decoder designs that overemphasize recent items while underutilizing distant, yet semantically relevant, interactions, leading to poor modeling of long-range dependencies and multi-interest behaviors. To address this challenge, we introduce MIDA (Multi-Interest Decoding with Adaptive Aggregation for Session-based Recommendation), a lightweight and encoder-agnostic framework that provides a reformulation of the decoding stage. The central design principle is a two-space decoupling: clustering operates on raw item embeddings so that semantic groupings are not distorted by encoder-learned relational biases; encoder outputs are reserved for aggregation, where sequential context remains intact. Concretely, MIDA (i) employs diff
    
[^36]: LoopGR：通过循环深度实现生成式推荐的参数高效扩展

    LoopGR: Parameter-Efficient Scaling for Generative Recommendation via Recurrent Depth

    [https://arxiv.org/abs/2610.34157](https://arxiv.org/abs/2610.34157)

    LoopGR提出分层循环深度机制，在共享生成式解码器与轻量级上下文更新器之间交替计算，并结合循环轨迹自蒸馏实现高效推理，从而在不大幅增加模型规模的情况下实现生成式推荐的参数高效扩展。

    

    生成式推荐将物品推荐形式化为自回归生成问题，而非判别式评分问题。尽管实证研究显示推荐系统具有良好的扩展行为，但在真实世界系统严格的模型规模和推理成本约束下，如何释放扩展带来的收益仍然充满挑战。一个核心难题是如何在不按比例增加模型规模的前提下，将对用户-物品交互历史的表达性建模与灵活的计算深度相结合。我们提出了LoopGR，一个基于循环深度的生成式推荐参数高效扩展框架。LoopGR采用了一种定制的分层循环深度机制，在共享的生成式解码器与轻量级上下文更新器之间交替运行。为了实现高效推理，我们进一步开发了循环轨迹自蒸馏方法，通过同时约束预测（摘要在此处截断），从高预算的教师模型中蒸馏出低预算的学生模型。

    arXiv:2610.34157v1 Announce Type: new  Abstract: Generative recommendation formulates item recommendation as autoregressive generation rather than a discriminative scoring problem. Although empirical studies exhibit favorable scaling behavior in recommendation, unleashing the benefits of scaling remains challenging under the strict model size and inference cost constraints of real-world systems. A central challenge is to combine expressive modeling of user-item interaction histories with flexible computational depth without proportionally increasing model size. We introduce LoopGR, a parameter-efficient scaling framework for generative recommendation via recurrent depth. LoopGR employs a tailored hierarchical recurrent depth mechanism that alternates between a shared generative decoder and a lightweight context updater. For efficient inference, we further develop recurrent trajectory self-distillation that distills a low-budget student from a high-budget teacher by aligning both predic
    
[^37]: 基于大语言模型的决策原生推荐

    Decision-Native Recommendation with Large Language Models

    [https://arxiv.org/abs/2610.34124](https://arxiv.org/abs/2610.34124)

    该论文提出了“决策原生推荐”新范式及DICE方法，利用大语言模型表示直接在候选物品上进行选择估计并同时输出显式推荐置信度，解决了生成式推荐无法唯一确定物品、受候选顺序影响且缺乏置信度的问题。

    

    大语言模型（LLM）为序列推荐中的用户偏好建模带来了世界知识，但基于生成的推荐器以文本形式间接表达下一物品预测。生成的输出可能无法唯一确定一个候选物品，依赖于提示中候选物品的呈现顺序，并且缺乏对推荐的显式置信度。近期研究探索了Jev，一个具有显式置信度的快速选项选择模型。受Jev启发，我们提出了决策原生推荐，这是一种利用LLM表示联合建模物品选择和推荐置信度的新范式。该范式要求有效的候选物品选择、在候选顺序重排下的一致性，以及能够反映推荐准确性的置信度。我们提出了直接物品选择估计（DICE），它从用户交互历史中学习在给定候选物品集合上的个性化下一物品分布，并推荐得分最高的物品…

    arXiv:2610.34124v1 Announce Type: new  Abstract: Large language models (LLMs) bring world knowledge to user preference modeling in sequential recommendation, but generation-based recommenders express next-item predictions indirectly as text. Generated outputs may fail to identify a unique candidate item, depend on candidate presentation order in the prompt, and lack explicit confidence in the recommendation. Recent studies explore Jev, a model for fast option selection with explicit confidence. Inspired by Jev, we introduce decision-native recommendation, a new paradigm that jointly models item choice and recommendation confidence using LLM representations. It requires valid candidate selection, consistency under candidate reordering, and confidence that reflects recommendation accuracy. We propose Direct Item Choice Estimation (DICE), which learns a personalized next-item distribution over supplied candidates from user interaction history and recommends the highest-scoring item with c
    
[^38]: 弥合数据集搜索中搜索意图与表格数据之间的鸿沟

    Bridging Search Intent and Tabular Data in Dataset Search

    [https://arxiv.org/abs/2610.33936](https://arxiv.org/abs/2610.33936)

    提出假设模式嵌入方法，利用语言模型生成任务对齐的假设表格模式进行嵌入检索，有效弥合了数据集搜索中用户意图与表格内容之间的鸿沟，并构建了KaggleDS语料库验证其在召回率上优于现有基线方法。

    

    数据集搜索仍然是数据分析工作流程中的一个瓶颈，因为现有系统主要依赖关键词匹配或浅层检索，无法捕捉用户意图。为了促进基于意图的搜索，我们提出了两个检索组件，通过自然语言为多方面的数据集搜索界面提供支持。首先，我们提出了假设模式嵌入，它利用语言模型生成与任务对齐的假设表格模式，并将其用于基于嵌入的检索，从而弥合了用户意图与表格内容之间的差距。我们构建了KaggleDS语料库，其中包含6,503个表格以及源自Kaggle的多样化任务查询，实验表明HySE在召回率上持续优于强大的关键词基线、基于嵌入的基线，以及在KaggleDS上微调的密集表格检索器。我们还证明了HySE能够泛化到端到端问答任务，例如在NQ-Tables数据集（一个包含169,898个维基百科表格的语料库）上的应用。

    arXiv:2610.33936v1 Announce Type: new  Abstract: Dataset search remains a bottleneck in data analysis workflows, as existing systems largely rely on keyword matching or shallow retrieval that fail to capture user intent. To facilitate intent-driven search, we present two retrieval components to equip multi-faceted dataset search interfaces through natural language. First, we propose Hypothetical Schema Embeddings (HySE) that uses a language model to generate task-aligned hypothetical table schemas which are used for embedding-based retrieval, bridging the gap between intent and table content. We introduce KaggleDS, a corpus comprising 6,503 tables and diverse task-queries derived from Kaggle, and show that HySE consistently improves recall over strong keyword and embedding-based baselines, as well as over a dense table retriever fine-tuned on KaggleDS. We also show that HySE generalizes to end-to-end question answering on, for example, the NQ-Tables dataset, a corpus of 169,898 Wikiped
    
[^39]: TriageRAG：一种用于将检索增强语言模型适配到专业领域的故障感知自训练方法

    TriageRAG: Failure-Aware Self-Training for Adapting Retrieval-Augmented Language Models to Specialized Domains

    [https://arxiv.org/abs/2610.33776](https://arxiv.org/abs/2610.33776)

    TriageRAG提出了一种故障感知的自训练方法，通过在有无检索上下文的条件下探测学生模型，并将每个自生成的示例按失败类型（已知、阅读器失败或检索器失败）进行分诊，从而更有针对性地将RAG系统适配到医学等专业领域。

    

    检索增强生成（RAG）使大型语言模型（LLM）能够基于检索到的证据回答知识密集型问题，但在仅有未标注语料库可用的情况下，将通用的RAG系统适配到医学等专业领域十分困难。诸如SimRAG之类的自训练方法通过让LLM根据领域段落自行撰写问答对，并在通过往返一致性过滤的问答对上进行微调来应对这一挑战。我们观察到，这些方法对所有保留下来的示例一视同仁：它们从不询问学生模型是否已经知道答案、是否已经能够从检索到的上下文中正确提取出答案，或者是否失败——如果失败，问题究竟出在阅读器还是检索器上。为了解决这一问题，我们提出了TriageRAG，这是一种故障感知的自训练方法，它在有检索上下文和无检索上下文两种情况下对学生模型在每个自生成的问题上进行探测，并将其分类为（原文在此处截断）

    arXiv:2610.33776v1 Announce Type: cross  Abstract: Retrieval-augmented generation (RAG) lets large language models (LLMs) answer knowledge-intensive questions from retrieved evidence, but adapting a general-purpose RAG system to a specialized domain such as medicine is difficult when only an unlabeled corpus is available. Self-training approaches such as SimRAG address this by letting the LLM write its own question-answer pairs from domain passages and fine-tuning on the pairs that survive a round-trip consistency filter. We observe that these approaches treat every retained example identically: they never ask whether the student already knows the answer, already extracts it correctly from the retrieved context, or fails - and if so, whether the failure lies with the reader or with the retriever. To tackle this, we propose TriageRAG, a failure-aware self-training approach that probes the student on each self-generated question with and without retrieved context, triages it into one of 
    
[^40]: SkillQR：面向推理密集型查询重写的智能体技能发现

    SkillQR: Agentic Skill Discovery for Reason-Intensive Query Rewriting

    [https://arxiv.org/abs/2610.33768](https://arxiv.org/abs/2610.33768)

    SkillQR提出了一种智能体式查询重写框架，能够针对不同检索目标自动发现最优的推理重写技能，解决了现有方法采用统一重写策略、依赖昂贵模型或大量训练数据的问题。

    

    稠密检索器为缓解大型语言模型（LLM）输出中的幻觉问题提供了有据可依的证据。然而，现实世界中的查询往往冗长、表述模糊或需要多步推理，其中的无关细节会削弱稠密检索器的语义匹配效果。查询重写是一种被广泛采用的检索前技术，它利用LLM对查询进行重新表述，以弥合此类查询与检索目标之间的语义鸿沟。现有的重写方法要么依赖昂贵的专有模型、大量训练数据，或迭代的检索与重写流程，同时在很大程度上无论检索语料库为何都采用统一的重写策略。我们观察到，有效的查询重写指令并非普适的：最优策略取决于检索目标，而不仅仅是模型本身。我们提出了SkillQR，一个能够自动发现推理策略（重写技能）的查询重写框架。

    arXiv:2610.33768v1 Announce Type: new  Abstract: Dense retrievers provide grounded evidence that mitigates hallucinations in outputs from Large Language Models (LLMs). Real-world queries are often verbose, ambiguously framed, or require multi-step reasoning, introducing irrelevant details that weaken semantic matching for dense retrievers. Query rewriting is a widely adopted pre-retrieval technique that leverages LLMs to reformulate queries and bridge the semantic gap between such queries and the retrieval target. Existing rewriting approaches either rely on expensive proprietary models, substantial training data, or iterative retrieval and rewriting, while largely applying a uniform rewriting strategy regardless of the retrieval corpus. We observe that effective query rewriting instructions are not universal: the optimal strategy depends on the retrieval target, rather than the model alone. We introduce SkillQR, a query rewriting framework that automatically discovers reasoning strate
    
[^41]: 新颖与否？PatEx-Gym：面向专利审查中新颖性评估的智能体化现有技术检索

    Novel or Not? PatEx-Gym: Agentic Prior-Art Search for Novelty Assessment in Patent Examination

    [https://arxiv.org/abs/2610.33503](https://arxiv.org/abs/2610.33503)

    该论文提出了首个用于开发和评估专利审查智能体的框架PatEx-Gym，其包含1000个真实审查任务、超百万件专利与5.2万篇论文的大规模检索语料库，并创新性地引入无需标注的检索轨迹合成框架PASS，从而完整覆盖现有技术“检索—推理—评估”的全流程。

    

    专利审查是一项具有挑战性的现实世界任务，它要求智能体发现稀疏的现有技术、迭代式地优化检索查询、审查技术证据，并在异构的科学技术领域中生成记录于审查意见通知书中的新颖性评估。然而，现有的基准测试大多关注静态且有限的候选集合，使得完整的“检索—推理—评估”流程未得到充分探索。我们提出了PatEx-Gym，这是首个用于开发和评估专利审查智能体的框架，它集成了三个组成部分。首先，PatEx-Bench提供了1000个真实的审查任务、两个包含超过100万件专利和5.2万篇科学论文的大规模检索语料库，以及一个支持迭代检索、文档访问、推理和审查意见通知书生成的运行时环境。其次，我们引入了PASS，这是一个无需人工标注的现有技术智能体检索轨迹合成框架。具体而言，该框架利用文档……（摘要原文在此处截断）

    arXiv:2610.33503v1 Announce Type: cross  Abstract: Patent examination is a challenging real-world task that requires agents to discover sparse prior art, iteratively refine queries, inspect technical evidence, and produce novelty assessments documented in Office Actions (OAs) across heterogeneous scientific and technical domains. However, existing benchmarks largely focus on static and limited candidate sets, leaving the full search--reasoning--assessment process underexplored. We present PatEx-Gym, the first framework for developing and evaluating agents for patent examination, which integrates three components. First, PatEx-Bench provides 1K realistic examination tasks, two large-scale retrieval corpora containing over 1M patents and 52K scientific papers, and a harness runtime supporting iterative search, document visits, reasoning, and OA generation. Second, we introduce PASS, an annotation-free Prior-Art Agentic Search Trajectory Synthesis framework. Specifically, it exploits docu
    
[^42]: DyHist：面向时序知识图谱外推的查询感知历史动态建模

    DyHist: Query-Aware Historical Dynamics Modeling for Temporal Knowledge Graph Extrapolation

    [https://arxiv.org/abs/2610.33473](https://arxiv.org/abs/2610.33473)

    提出DyHist框架，通过查询感知地建模历史动态——分离近期证据与中程上下文的查询相关偏差，并关系感知地评估历史循环的可靠性——从而提升时序知识图谱外推的预测效果。

    

    时序知识图谱（TKG）外推旨在从已观测的历史事件中预测未来事实。现有方法通常通过联合建模局部历史演化与长程历史上下文来实现这一目标。首先，局部演化建模往往将局部历史总结为一个整体演化表示，其中与查询相关的局部偏差可能未被充分挖掘为有用的预测特征。其次，现有的长程历史建模通常将历史循环视为稳定线索，而忽视了其可靠性可能因关系和近期上下文的不同而变化。为解决这些问题，我们提出了DyHist，一个面向时序知识图谱外推的查询感知历史动态建模框架。DyHist通过将近期证据与中程上下文分离，并显式地将二者的查询相关偏差建模为预测特征来捕捉局部动态。它进一步采用一种关系感知的历史……

    arXiv:2610.33473v1 Announce Type: new  Abstract: Temporal Knowledge Graph (TKG) extrapolation aims to predict future facts from observed historical events. Existing methods usually achieve this goal by jointly modeling local historical evolution and long-range historical contexts. First, local evolution modeling often summarizes local history into an overall evolving representation, where query-relevant local deviations may not be sufficiently exploited as useful predictive features. Second, existing long-range history modeling often treats historical recurrence as a stable cue, while overlooking that its reliability may vary across relations and recent contexts. To address these issues, we propose DyHist, a query-aware historical dynamics modeling framework for TKG extrapolation. DyHist captures local dynamics by separating recent evidence from medium-range contexts and explicitly modeling their query-relevant deviations as predictive features. It further uses a relation-aware histori
    
[^43]: PinGER：基于语义ID的令牌高效生成式检索

    PinGER: Token-Efficient Generative Retrieval with Semantic IDs

    [https://arxiv.org/abs/2610.33435](https://arxiv.org/abs/2610.33435)

    PinGER是Pinterest部署的令牌高效生成式检索器，通过在交错的SID与稠密令牌上使用非对称注意力掩码实现输入历史压缩，并结合目标条件输出、强化学习和前缀多样性惩罚实现可控生成，在保留语义ID输出的同时大幅提升了上下文窗口的物品容量。

    

    生成式检索通过逐令牌解码物品的语义ID（SID）来取代最近邻搜索。但现有的基于SID的方法主要使用L层令牌按顺序表示输入历史中的每个物品。因此，固定的上下文窗口所能容纳的物品数量比每个物品使用一个嵌入表示时少了L倍。我们提出了PinGER，这是部署在Pinterest上的一种生成式检索器，它在保留SID输出的同时使输入历史更加紧凑，其构建基于三个想法：（1）在交错的SID令牌与稠密令牌上使用非对称注意力掩码，使后续物品只能关注稠密令牌，这样模型在训练时能在每个SID位置都获得监督信号，但在服务时每个物品只需处理一个令牌，同时计算相同的函数。（2）通过目标条件输出、强化学习和前缀多样性惩罚实现可控生成，三者共同将检索引导至表面特定的目标和多样化的候选结果。（3）一种ser……

    arXiv:2610.33435v1 Announce Type: new  Abstract: Generative retrieval replaces nearest-neighbor search with token-by-token decoding of an item's semantic ID (SID). But existing SID-based methods primarily use L levels of tokens sequentially to represent each item in the input history. As a result, a fixed context window holds L times fewer items than it would with one embedding per item. We present PinGER, a generative retriever deployed at Pinterest that retains SID outputs while making input histories compact, built on three ideas. (1) An asymmetric attention mask over interleaved SID and dense tokens lets later items attend only to the dense ones, so the model is supervised at every SID position in training yet serves one token per item while computing the same function. (2) Generation is controllable through goal-conditioned outputs, reinforcement learning, and a prefix-diversity penalty, which together steer retrieval toward surface-specific goals and diverse candidates. (3) A ser
    
[^44]: Pinterest自动竞价系统中控制器参数的强化学习方法

    Reinforcement Learning for Controller Parameters in the Auto-Bidding System at Pinterest

    [https://arxiv.org/abs/2610.33248](https://arxiv.org/abs/2610.33248)

    本文提出了一种部署在Pinterest的混合架构RLBP，以强化学习作为元控制器每日为各广告活动动态选择PID控制器参数，解决了传统固定PID调参无法适应多样化广告活动与非平稳市场条件的问题。

    

    在线广告中的自动竞价系统必须在非平稳的市场条件下，针对异构的广告活动平衡相互竞争的目标——在最大化预算消耗的同时最小化单次行为成本。这种权衡会随广告活动的生命周期而演变：早期在预算可以结转时，成本优化占主导地位；而在临近截止日期时，预算消耗则变得紧迫。虽然PID控制器因其稳定性和可解释性在生产环境中被广泛使用，但固定的调优参数无法适应多样化的广告活动和不断变化的流量模式。我们提出了RLBP（竞价参数强化学习），这是一种部署在Pinterest的混合架构，它使用强化学习作为元控制器每日为每个广告活动选择特定的PID参数，同时保留PID用于高频的竞价调整。RLBP使用相对于生产基线进行归一化的相对提升奖励公式来训练多头Q网络，并在推理时执行融合……（原文摘要到此截断）

    arXiv:2610.33248v1 Announce Type: cross  Abstract: Auto-bidding systems in online advertising must balance competing objectives-maximizing budget delivery while minimizing cost per action-across heterogeneous campaigns under non-stationary market conditions. This trade-off evolves over campaign lifespans: early on, cost optimization dominates when budgets roll over; near deadlines, budget delivery becomes urgent. While PID controllers are widely used in production for their stability and interpretability, fixed tunings cannot adapt to diverse campaigns and shifting traffic patterns. We present RLBP (Reinforcement Learning for Bidding Parameters), a hybrid architecture deployed at Pinterest that uses RL as a meta-controller to select campaign-specific PID parameters daily while retaining PID for high-frequency bid adjustments. RLBP trains a multi-head Q-network using a relative-lift reward formulation normalized against the production baseline, and performs inference-time fusion that ad
    
[^45]: Holo-NN：基于压缩感知的全息最近邻搜索

    Holo-NN: Holographic Nearest Neighbor Search via Compressive Sensing

    [https://arxiv.org/abs/2610.33224](https://arxiv.org/abs/2610.33224)

    提出Holo-NN代数框架，利用压缩感知与保局特征变换，以单一线性投影取代传统索引中数据依赖的非线性分支运算，从而在隐私保护约束（如全同态加密）下实现高效的高维近似最近邻搜索。

    

    在人工智能时代，对高维嵌入向量进行隐私保护的近似最近邻（ANN）搜索已成为一个至关重要却又极具挑战性的领域，其核心难题在于速度、隐私和准确性之间的权衡。传统的基于树和图的索引结构（如HNSW、ScaNN）从根本上依赖于非线性运算，而这些运算在全同态加密（FHE）或盲服务器端约束下是难以高效计算的。本文提出了Holo-NN（全息最近邻搜索），这是一个用单一线性投影取代数据依赖分支的代数框架。我们的关键观察是：虽然数据库中各点之间的距离分布是稠密的，但邻域成员关系在本质上是稀疏的——只有极少数点会靠近某个查询点。通过将保局特征变换与压缩感知相结合，Holo-NN在离线阶段将整个数据库折叠为一个紧凑的全息（摘要在此处截断）

    arXiv:2610.33224v1 Announce Type: cross  Abstract: In the age of AI, privacy-preserving approximate nearest neighbor (ANN) search over high-dimensional embeddings has emerged as a critical yet frustratingly challenging domain, dominated by tradeoffs between speed, privacy, and accuracy. Traditional tree and graph-based indexing structures (e.g., HNSW, ScaNN) rely fundamentally on non-linear operations that are computationally intractable under Fully Homomorphic Encryption (FHE) or blind server-side constraints.   In this paper, we introduce Holo-NN (Holographic Nearest Neighbor Search), an algebraic framework that replaces data-dependent branching with a single linear projection. Our key observation is that while distances across a database are dense, neighborhood membership is inherently sparse: only a handful of points ever lie close to a query. By pairing locality-preserving feature transforms with Compressive Sensing, Holo-NN folds the entire database offline into a compact hologra
    
[^46]: CTR重新校准应该使用多少新鲜数据？事后校准的时序评估

    How Much Fresh Data Should CTR Recalibration Use? A Temporal Evaluation of Post-hoc Calibration

    [https://arxiv.org/abs/2610.33168](https://arxiv.org/abs/2610.33168)

    该研究通过时序评估发现随机划分会显著低估CTR模型的校准误差，并证明使用12小时新鲜数据重新校准可将校准性能恶化的情形从36个中的13-17个降至3-5个，为事后校准的数据量选择提供了实证依据。

    

    点击率（CTR）模型的离线评估仍然默认使用随机划分，尽管部署的模型需要对未来流量进行评分，且下游系统消费的是校准后的概率。在Avazu数据集（4040万次曝光）上，我们在两个滚动时间起点和一个随机划分下训练逻辑回归（LR）、DeepFM和DCN-v2，每组设置三个随机种子，并使用基于前几小时数据拟合的Platt缩放和等渗回归对每个测试日进行重新校准。在起点1最近的测试日，随机划分报告的LR期望校准误差（ECE）在每个种子中均为时序数值的55-62%（深度模型的比率在不同种子间不稳定），且没有任何模型的ECE随时间距离呈单调变化。使用6小时的拟合数据片段，重新校准在严重校准不良的情况下最多可消除61%的ECE，但在约三分之一的情形中反而增加了ECE（跨种子统计为48个情形中的14-17个）。在固定评估集上，12小时的数据片段将有害情形从36个中的13-17个减少到3-5个。

    arXiv:2610.33168v1 Announce Type: cross  Abstract: Offline evaluation of click-through rate (CTR) models still defaults to random splits, although deployed models score future traffic and downstream systems consume calibrated probabilities. On Avazu (40.4M impressions) we train logistic regression (LR), DeepFM, and DCN-v2 under two rolling temporal origins and a random split, three seeds per arm, and recalibrate each test day with Platt scaling and isotonic regression fitted on the preceding hours. At origin 1's nearest test day, random splits report LR's expected calibration error (ECE) at 55-62% of the temporal value in every seed (the deep models' ratio is seed-unstable), and no model's ECE is monotone in temporal distance. With a 6-hour fitting slice, recalibration removes up to 61% of ECE in badly miscalibrated cells but increases it in about a third of applications (14-17 of 48 across seeds). On fixed evaluation sets, a 12-hour slice cuts harm from 13-17 to 3-5 of 36 applications
    
[^47]: 运气的教训：LLM撰写的经验教训会把偶然性带入智能体的下一次决策吗？

    Lessons from Luck: Do LLM-Written Lessons Carry Chance into an Agent's Next Decision?

    [https://arxiv.org/abs/2610.33029](https://arxiv.org/abs/2610.33029)

    本文通过预注册的受控实验（256个合成世界、11个LLM端点、90,084条回答），首次系统测量了LLM智能体从偶然（幸运或不幸）结果中撰写的经验教训是否会将“运气”传导到智能体的下一次工具选择决策中。

    

    arXiv:2610.33029v1 公告类型：交叉 摘要：自我改进的智能体会根据运行结果撰写经验教训；当结果存在噪声时，这些教训可能将偶然因素带入后续决策中。我们在一项预注册研究中测量了这一链条中的一环。在256个合成世界中，每个世界都是一个周期性任务，配备两个可互换、可靠性已记录在案的工具。一段脚本化的先前运行会使用更可靠的工具（好决策）或较不可靠的工具（坏决策），并被指定结果：成功或瞬时失败。一个模型根据ReasoningBank的记忆撰写指令为该次运行撰写一条经验教训；同一模型的一个全新会话在掌握权威日志和先前结果的情况下，依据一条记忆条目为下一次运行选择首个工具——该条目可以是这条教训、另一结果的教训、无关条目或未经推理的指令。该实验方案在任何验证性调用之前被冻结、存档并加盖时间戳；11个LLM端点共给出了90,084个回答。在出现一次偶然失败后……（摘要至此被截断）

    arXiv:2610.33029v1 Announce Type: cross  Abstract: Self-improving agents write lessons from the outcomes of their runs; when outcomes are noisy, a lesson can carry chance into later decisions. We measure one link of this chain in a registered study. In 256 synthetic worlds, each a recurring task with two interchangeable tools of recorded reliability, a scripted previous run used the more reliable tool (a good decision) or the less reliable one (a bad decision), with an assigned outcome: success or a transient failure. A model wrote a lesson from the run with ReasoningBank's memory-writing instruction; a fresh session of the same model, holding the authoritative log and the previous result, chose the next run's first tool with one memory item: that lesson, the lesson from the other outcome, an irrelevant item, or an unreasoned directive. The protocol was frozen, deposited and timestamped before any confirmatory call; 11 LLM endpoints gave 90,084 answers. After a chance failure of the be
    
[^48]: 将思考与拼写分离以实现高效的生成式推荐

    Separating Thinking from Spelling for Efficient Generative Recommendation

    [https://arxiv.org/abs/2610.32980](https://arxiv.org/abs/2610.32980)

    该论文提出Deco方法，将阅读用户历史与书写物品标识符的过程解耦——骨干网络将历史一次性压缩为紧凑的“下一物品概念”，再由轻量的KV缓存解码器独立执行整个束搜索，从而大幅降低生成式推荐中随骨干规模、SID深度和候选数量增长的解码开销。

    

    生成式推荐器通过在Trie树约束的束搜索下逐码书写物品的语义标识符（SID）来检索物品，这一方法目前已部署于大型网络平台。标准的自回归接口为SID长度K付出了双倍代价：T个物品的历史序列以TK个代码token的形式进入骨干网络，且每一次束搜索步骤都要经过骨干网络，因此为每个用户返回B个候选需要O(BK)个骨干网络解码token。第二项开销通常占据主导地位，并随骨干网络规模、SID深度和候选预算的增长而增长。我们提出了Deco（从概念解码，decoding from concepts），将阅读历史与书写标识符相分离。骨干网络将每个历史物品读取为一个融合token，并将历史一次性压缩为紧凑的“下一物品概念”；一个轻量的、带KV缓存的解码器以该概念和已生成前缀为条件运行整个束搜索，而标识符、Trie树以及自回归分解保持不变。

    arXiv:2610.32980v1 Announce Type: new  Abstract: Generative recommenders retrieve items by writing their semantic IDs (SIDs) code by code under trie-constrained beam search, an approach now deployed on large web platforms. Standard autoregressive interfaces pay for the SID length K twice: a history of T items enters the backbone as TK code tokens, and every beam-search step runs through the backbone, so returning B candidates costs O(BK) backbone decode tokens per user. The second cost often dominates, and it grows with the backbone, the SID depth and the candidate budget.   We introduce Deco (decoding from concepts), which separates reading the history from writing the identifier. The backbone reads each history item as one fused token and condenses the history once into a compact next-item concept; a light KV-cached decoder conditioned on the concept and the generated prefix runs the entire beam search, while the identifiers, the trie and the autoregressive factorization stay unchang
    
[^49]: 用于验证电子取证中审查导向型AI工作流的延迟倒计时方法

    The Delayed Countdown Method for Validating Review-Oriented AI Workflows in eDiscovery

    [https://arxiv.org/abs/2610.31311](https://arxiv.org/abs/2610.31311)

    本文提出延迟倒计时方法，为电子取证中审查导向型AI工作流的有效性验证提供了一种简单实用、易于集成、兼容任意停止规则并能给出有效置信区间的评估方案。

    

    面向审查的工作流已成为电子取证中AI辅助文档审查的主流方法。然而，如何验证审查导向型工作流的有效性，一直是法律领域技术辅助审查（TAR）中一个令人困惑且充满争议的话题。我们分析了这一争议的根源，并为审查导向型工作流提出了一种简单而实用的评估方法。我们的延迟倒计时方法易于融入现有的审查导向型工作流和软件中，兼容任意停止规则，并能产生关于有效性的有效频率学派置信区间。我们阐释了该方法的概念基础，提供了示例计算，讨论了对逻辑上不可能的召回率值的处理，并就降低审查成本的后续研究提出了建议。

    arXiv:2610.31311v1 Announce Type: new  Abstract: Review-oriented workflows have become the dominant approach to AI-assisted document review in electronic discovery (eDiscovery). How to validate the effectiveness of review-oriented workflows has, however, remained a confusing and controversial topic in technology-assisted review (TAR) in the law. We examine the sources of this controversy, and propose a simple and practical evaluation approach for review-oriented workflows. Our Delayed Countdown method is easily incorporated into existing review-oriented workflows and software, is compatible with arbitrary stopping rules, and produces valid frequentist confidence intervals on effectiveness. We explain the conceptual foundations of the method, provide example computations, discuss the handling of logically impossible recall values, and suggest future work on decreasing review costs.
    
[^50]: FormBench：面向配方专利细粒度检索的基准测试

    FormBench: A Benchmark for Fine-Grained Retrieval in Formulation Patents

    [https://arxiv.org/abs/2610.30175](https://arxiv.org/abs/2610.30175)

    该论文提出了FormBench基准，用于评估模型从配方专利等技术文档中检索精确化学信息的能力，发现商业API编码器表现极差，而基于FormBench三元组进行任务自适应微调的开源编码器性能显著提升。

    

    我们提出了FormBench，一个用于测试模型从技术文档中检索精确化学信息能力的平台。FormBench包含55,352个查询和两个检索语料库变体，每个变体约有100万个段落：FormBench-Random（C0）使用随机采样的干扰项，FormBench-Structured（C1）使用结构上相近的干扰项。每个查询都配有一个锚定段落、一个硬负例和一个软负例——这些通过材料本体中的实体重叠评分构建，为模型的判别能力提供了更深层次的分级信号。我们评估了10种模型配置，涵盖商业API编码器（OpenAI、Voyage）、开源编码器（Nomic、GTE-large、Qwen3-Embed）及其任务自适应微调（TAPT）变体。我们发现：(1) 商业API在所有配置上表现都很差（OpenAI-3-large在C1上仅为0.009，在C0上为0.213）；(2) 在FormBench三元组上进行TAPT使Qwen3-Embed在C1上的性能提升至0.4085。

    arXiv:2610.30175v1 Announce Type: new  Abstract: We introduce FormBench, a platform for testing the ability of models to retrieve precise chemical information from technical documents. FormBench contains 55,352 queries and two retrieval corpus variants each with ~1M passages: FormBench-Random (C0), with randomly sampled distractors, and FormBench-Structured (C1), with structurally proximate distractors. Each query is paired with an anchor passage, a hard negative, and a soft negative - derived from a materials ontology through entity overlap scoring, providing a deeper graded signal on a model's discriminatory power. We evaluate 10 model configurations spanning commercial API encoders (OpenAI, Voyage), open-source encoders (Nomic, GTE-large, Qwen3-Embed), and their task-adaptive fine-tuned (TAPT) variants. We find: (1) commercial APIs score very poorly across all configurations (OpenAI-3-large: 0.009 on C1, 0.213 on C0); (2) TAPT on FormBench triplets lifts Qwen3-Embed to 0.4085 on C1,
    
[^51]: Scriptorium：多人类、多AI协作的数字图书馆搜索平台的系统设计与实现

    Scriptorium: System Design and Implementation of a Multi-Human, Multi-AI Collaborative Digital Library Search Platform

    [https://arxiv.org/abs/2610.28929](https://arxiv.org/abs/2610.28929)

    该论文提出了Scriptorium，一个支持多人类与多AI协作的数字图书馆搜索平台，其核心创新在于将完全跨用户透明的搜索活动与由七个角色专门化生成式AI智能体组成的管理者-工作者架构相结合，并完整公开系统实现细节以支持复现与重用。

    

    本技术报告记录了Scriptorium的设计与实现，这是一个旨在支持多人类、多AI协作的数字图书馆搜索界面：实现了搜索活动的完全跨用户透明性，并结合了由七个角色专门化的生成式AI智能体组成的“管理者-工作者”（Manager-Worker）架构。一篇配套的简短演示论文描述了Scriptorium的交互设计；由于演示格式受限于几页篇幅，必然省略了大部分实现细节。本报告记录了Scriptorium的完整系统架构、数据模型、协作交互设计原理以及AI编排机制，包括智能体提示词原文、约束这些提示词的确定性防护机制、语言模型与检索配置，以及运行系统所需的步骤，以便其他研究人员能够检查、批评、复现和重用该设计。我们还讨论了该系统（原文在此处截断）。

    arXiv:2610.28929v1 Announce Type: cross  Abstract: This technical report documents the design and implementation of Scriptorium, a digital library search interface designed to support multi-human, multi-AI collaboration: full cross-user transparency of search activity combined with a Manager--Worker architecture of seven role-specialized generative AI agents. A companion short demonstration paper describes Scriptorium's interaction design; because a demonstration format is constrained to a few pages, it necessarily omits most implementation detail. This report documents Scriptorium's complete system architecture, data model, collaborative-interaction design rationale, and AI orchestration mechanism, including verbatim agent prompts, the deterministic guardrails that constrain them, the language model and retrieval configuration, and the steps needed to run the system, so that the design can be inspected, critiqued, reproduced, and reused by other researchers. We also discuss the system
    
[^52]: ECHO：面向蒸馏引导的10万长度序列建模的同构算子高效级联框架

    ECHO: Efficient Cascade with Homogeneous Operators for Distillation-Guided 100K-Sequence Modeling

    [https://arxiv.org/abs/2610.28569](https://arxiv.org/abs/2610.28569)

    提出ECHO级联框架，利用蒸馏选择器（DS）和目标感知细粒度检索器（TFR）两级结构，在10万长度用户行为序列上实现高效且保持物品级可寻址性的目标感知推荐排序建模。

    

    对用户终身行为序列进行建模对于捕捉工业推荐系统中的长期偏好至关重要，然而10万长度（100K）的用户序列使得目标感知的排序计算代价高得难以承受。从原始序列中检索的基于搜索的方法会产生随序列长度增长且依赖候选物品的计算开销，而潜在压缩方法则以牺牲物品级的可寻址性为代价来降低计算量。我们提出ECHO（Efficient Cascade with Homogeneous Operators，同构算子高效级联）框架，它通过蒸馏选择器（DS）为每个用户保留一个原始事件记录子集，并通过目标感知细粒度检索器（TFR）为每个候选物品检索一个更小的目标相关记录集合，从而在效率与细粒度检索之间取得平衡。DS由具备候选感知能力的全历史教师模型通过逐目标蒸馏进行监督，实现了目标感知的用户级选择，而无需为每个候选执行全历史计算。TFR……

    arXiv:2610.28569v1 Announce Type: new  Abstract: Modeling lifetime user behavior sequences is essential for capturing long-range preferences in industrial recommendation, yet 100K-length user sequences make target-aware ranking prohibitively expensive. Search-based methods that retrieve from raw sequences incur candidate-dependent costs that grow with sequence length, whereas latent compression methods reduce computation at the expense of item-level addressability. We propose ECHO (Efficient Cascade with Homogeneous Operators), a framework that balances efficiency and fine-grained retrieval by retaining a subset of original event records per user via a Distilled Selector (DS) and retrieving a smaller set of target-relevant records per candidate via a Target-aware Fine-grained Retriever (TFR). DS is supervised by a candidate-aware full-history teacher via per-target distillation, enabling target-aware user-level selection without incurring full-history computation for every candidate. T
    
[^53]: 超越行为准确性：评估大语言模型投资者模拟中的个体层面保真度

    Beyond Behavioral Accuracy: Evaluating Individual-Level Fidelity in LLM Investor Simulation

    [https://arxiv.org/abs/2610.26524](https://arxiv.org/abs/2610.26524)

    该研究通过80名参与者的纵向模拟交易实验，首次系统评估了LLM在个体层面对投资者行为与决策逻辑的保真度，发现当前LLM无法超越简单基线，买卖结构与资产选择尤其难以复现，且相似的活动表现会导致截然不同的投资组合轨迹。

    

    大语言模型（LLM）越来越多地被用作用户模拟器，但它们能否随时间推移忠实地再现个体的行为和决策逻辑？我们通过一项涉及80名参与者的纵向模拟交易（paper-trading）研究来探讨这一问题。通过滚动式次日预测，我们评估了交易活动、买卖结构、资产选择和投资组合结果，并辅以受控输入消融实验和参与者评分的解释一致性评估。在预测交易发生方面，没有任何被评估的LLM能够可靠地超越简单的近期活动持续性基线，而买卖结构和资产选择仍然难以复现。相似的活动层面表现也可能产生截然不同的投资组合轨迹。消融实验表明，活动预测主要依赖于近期交易，而资产选择则对多个证据来源敏感。参与者认为解释与……（原文摘要在此处截断）

    arXiv:2610.26524v2 Announce Type: replace  Abstract: Large language models (LLMs) are increasingly used as user simulators, but can they faithfully reproduce an individual's behavior and decision logic over time? We investigate this question in a longitudinal paper-trading study with 80 participants. Using rolling next-day predictions, we evaluate trading activity, buy-sell structure, asset selection, and portfolio outcomes, complemented by controlled input ablations and participant-rated explanation alignment. No evaluated LLM reliably outperforms a simple recent-activity persistence baseline for predicting trade occurrence, while buy-sell structure and asset selection remain difficult to reproduce. Similar activity-level performance can also yield substantially different portfolio trajectories. Ablations show that activity predictions depend primarily on recent trades, whereas asset choices are sensitive to multiple evidence sources. Participants rate explanations as more aligned wit
    
[^54]: 智能体记忆的几何学：预测检索相关的记忆失效，并刻画其局限性

    The Geometry of Agent Memory: Forecasting Retrieval-Bound Memory Failures, and Characterizing Its Limits

    [https://arxiv.org/abs/2610.26433](https://arxiv.org/abs/2610.26433)

    该研究用对话、记忆与查询之间的几何关系特征来预测智能体记忆系统的检索失效，但发现其整体预测能力不及简单的问题难度基线，且在控制问题难度和系统身份后，几何特征并未带来统计学上显著的性能提升。

    

    我们研究了对话轮次、记忆文本和查询之间的几何关系能否在生成答案之前预测答案的正确性。我们评估了四种采用稠密或混合检索的记忆实现，以及一个观察-摘要对比基线。我们将这些文本嵌入到共享空间中。一个逻辑回归分类器使用25个特征来描述查询邻近性、记忆-对话对应关系和对话结构。在LongMemEval-small数据集上，2,367个问题-系统配对的合并AUROC为0.640，低于问题难度基线的0.655。性能因问题类型而异：几何特征在单会话助手类问题上达到0.841，但仅凭系统身份即可达到0.826。在控制问题难度和系统身份因素后，几何特征未带来统计学上成立的合并改进（delta = +0.002，95%置信区间[-0.010, 0.014]）。固定存储的上下文选择实验发现，使用按邻近度排序的记忆能获得比...更高的准确率（原文截断）。

    arXiv:2610.26433v2 Announce Type: replace  Abstract: We study whether geometric relationships between conversation turns, memory texts, and queries can predict answer correctness before generation. We evaluate four memory implementations with dense or hybrid retrieval and one observation-summary comparator. We embed these texts in a shared space. A logistic classifier uses 25 features describing query proximity, memory-conversation correspondence, and conversation structure. On LongMemEval-small, pooled AUROC is 0.640 over 2,367 question-system pairs, below a question-difficulty baseline of 0.655. Performance varies by question type: geometry reaches 0.841 on single-session-assistant questions, but system identity alone reaches 0.826. After controlling for difficulty and identity, geometry adds no statistically established pooled improvement (delta = +0.002, 95% CI [-0.010, 0.014]). A fixed-store context-selection experiment finds higher accuracy with proximity-ranked memories than wit
    
[^55]: 比较生成式AI与机器学习和人工审查在电子证据开示中的应用

    Comparing GenAI to Machine Learning and Manual Review for eDiscovery

    [https://arxiv.org/abs/2610.16925](https://arxiv.org/abs/2610.16925)

    该研究指出此前声称生成式AI审查工具（88%召回率）优于机器学习工具（64%召回率）的结论因方法差异而不可比，实际分析发现生成式AI与人工审查效果相当，而采用持续主动学习（CAL）的机器学习模型仅需2,000条样本训练即可达到最先进效果并具备更高效率。

    

    Keeling等人[Keeling2026Generative]报告了Redgrave LLP最近开展的一项案例研究，他们声称Relativity的生成式AI审查工具获得了88%的召回率，而其机器学习工具Relativity Assisted Learning (RAL)仅达到64%的召回率。然而这些结果并不具有可比性，原因在于所采用的方法目标不同、评估者之间存在分歧，以及它们代表了不同的召回率-精确度权衡和成本-效益权衡。我们对Redgrave数据的分析表明，在考虑这些差异之后，aiR与人工审查的效果是相当的。此外，采用持续主动学习（CAL）的成熟机器学习模型，仅在2,000条初审人员相关性判定样本上进行训练，就能展现出最先进的有效性和更优越的效率。

    arXiv:2610.16925v1 Announce Type: cross  Abstract: Keeling et al. [Keeling2026Generative] report on a case study recently conducted by Redgrave LLP in which they assert that Relativity's Generative AI Review tool (aiR) achieved 88% recall while Relativity Assisted Learning (RAL), its machine-learning tool, achieved 64% recall. These results are not comparable because the methods employed have different objectives, because of inter-assessor disagreement, and because they represent different recall-precision and cost-benefit trade-offs. Our analysis of the Redgrave data indicates that, when these differences are taken into account, aiR and manual review are comparable. Furthermore, established machine-learning models employing Continuous Active Learning (CAL) show state-of-the-art effectiveness, and superior efficiency, when trained only on a sample of 2,000 first-pass-reviewer responsiveness determinations.
    
[^56]: SearchJev：面向搜索代理的校准化System-1模型

    SearchJev: A Calibrated System-1 Model for Search Agents

    [https://arxiv.org/abs/2610.05107](https://arxiv.org/abs/2610.05107)

    SearchJev提出一个无需自回归生成的校准化System-1模型，将搜索代理的短决策与System-2推理分离，并通过软标签学习（SLCD）从不确定监督中提升决策置信度的可靠性。

    

    搜索代理需要反复就相关性、证据充分性和搜索动作做出短决策。使用生成式语言模型进行这些决策会引入延迟和不可靠的置信度。我们提出SearchJev，一个校准化的System-1模型，将搜索决策与System-2推理和生成相分离。给定搜索状态和决策模式，SearchJev无需自回归输出生成即可对合法选项进行评分。我们提出了用于校准决策的软标签学习，从不确定的监督中学习决策概率，并提高决策置信度的可靠性。在双系统搜索代理中，SearchJev处理短决策，并将不确定的判断委托给System 2，后者保留规划、查询生成和答案组合等能力。我们还引入了SearchDecision-Bench，一个统一六种搜索决策类型用于训练和评估的基准。在SearchDecision-Bench上……

    arXiv:2610.05107v2 Announce Type: replace-cross  Abstract: Search agents repeatedly make short decisions about relevance, evidence sufficiency, and search actions. Using generative language models for these decisions introduces latency and unreliable confidence. We present SearchJev, a calibrated System-1 model that separates search decisions from System-2 reasoning and generation. Given a search state and a decision schema, SearchJev scores legal options without autoregressive output generation. We propose Soft-Label Learning for Calibrated Decisions (SLCD) to learn decision probabilities from uncertain supervision and improve the reliability of decision confidence. In a dual-system search agent, SearchJev handles short decisions and delegates uncertain judgments to System 2, which retains planning, query generation, and answer composition. We also introduce SearchDecision-Bench, a benchmark unifying six types of search decisions for training and evaluation. On SearchDecision-Bench, S
    
[^57]: 5W1H+Which：基于渐进式本体绑定的上下文有效语义索引

    5W1H+Which: Context-Valid Semantic Indexing with Progressive Ontology Binding

    [https://arxiv.org/abs/2609.35184](https://arxiv.org/abs/2609.35184)

    论文提出5W1H+Which设计，通过将基于源的内容提取与渐进式本体绑定相分离，实现了既保留可复用信息、又为推理提供可检验前提的上下文有效语义索引。

    

    语义索引必须在不过早绑定到某一业务模式的前提下保留可复用的源信息，同时为推理提供可检验的前提。我们提出5W1H+Which，这是一种将基于源的内容提取与渐进式本体绑定相分离的设计。5W1H问题用于组织可独立寻址的内容单元；Which则记录与版本化本体元素的链接、映射关系、适用范围和验证状态。时间、位置、系统环境和参与者角色约束了事实、绑定和规则的使用时机。未绑定的内容仍然保持可搜索性，而形式化推理则需要显式的前提检查。该设计区分了业务有效时间、系统知识时间和操作轨迹，并跟踪依赖关系以支持重新验证和增量维护。工作实例展示了云迁移和时间性策略有效性等应用场景。所得到的链接语料库可以支持……

    arXiv:2609.35184v2 Announce Type: replace  Abstract: Semantic indexing must preserve reusable source information without committing too early to one business schema, while providing checkable premises for reasoning. We propose 5W1H+Which, a design that separates source-grounded content extraction from progressive ontology binding. The 5W1H questions organize independently addressable content units; Which records links to versioned ontology elements, mapping relations, applicability scopes, and validation states. Time, location, system environment, and participant roles constrain when facts, bindings, and rules may be used. Unbound content remains searchable, whereas formal reasoning requires explicit premise checks. The design distinguishes business valid time, system knowledge time, and operational traces, and tracks dependencies for revalidation and incremental maintenance. Worked examples illustrate cloud migration and temporal policy validity. The resulting linked corpus can suppor
    
[^58]: 通过零令牌几何图超越相似性的多跳RAG

    Beyond Similarity through Zero-Token Geometric Graphs for Multi-Hop RAG

    [https://arxiv.org/abs/2609.19622](https://arxiv.org/abs/2609.19622)

    提出G³RAG框架，通过无需LLM调用和生成令牌的离线图构建，结合cosθ·sinθ几何增益分数、密度感知拓扑惩罚和单步受控扩散，在多跳RAG中实现超越纯相似性检索的互补证据发现。

    

    多跳检索增强生成（RAG）所需的证据既要与查询保持相关性，又要引入足够的新颖性以弥合语义鸿沟。密集检索往往集中于语义相似的文档，而基于图的替代方案通常依赖于代价高昂的大语言模型（LLM）实体抽取，并可能通过噪声连接进行传播。我们提出了几何增益图RAG（G$^3$RAG），这是一个仅基于文档的框架，其离线图构建过程不使用任何LLM调用或生成的令牌。G$^3$RAG为每条边分配一个几何增益分数 $\cos\theta \cdot \sin\theta$，该分数联合刻画了文档表示之间的方向一致性与正交性。密度感知的拓扑惩罚抑制了高度连接的枢纽节点，同时单步受控扩散从过滤后的查询种子向互补证据进行扩展。我们在MusiQue、2WikiMultiHopQA和HotpotQA数据集上使用Nv-emb（摘要截断）对G$^3$RAG进行了评估。

    arXiv:2609.19622v1 Announce Type: new  Abstract: Multi-hop retrieval-augmented generation (RAG) requires evidence that remains relevant to a query while introducing enough novelty to bridge semantic gaps. Dense retrieval tends to concentrate on semantically similar documents, whereas graph-based alternatives often depend on costly Large Language Model (LLM) entity extraction and may propagate through noisy connections. We introduce Geometric Gain Graph RAG (G$^3$RAG), a document-only framework whose offline graph construction uses no LLM calls or generated tokens. G$^3$RAG assigns each edge a geometric gain score, $\cos\theta \cdot \sin\theta$, that jointly captures directional consistency and orthogonality between document representations. A density-aware topological penalty suppresses highly connected hubs, while single-step controlled diffusion expands from filtered query seeds toward complementary evidence. We evaluate G$^3$RAG on MusiQue, 2WikiMultiHopQA, and HotpotQA using Nv-emb
    
[^59]: UniRec：面向级联推荐系统的跨阶段多任务融合与偏好对齐

    UniRec: Cross-stage Multi-Task Fusion with Preference Alignment for Cascaded Recommender Systems

    [https://arxiv.org/abs/2609.11052](https://arxiv.org/abs/2609.11052)

    UniRec提出了一种统一的跨阶段推荐融合模型，通过共享嵌入和单一计算图联合训练实现跨阶段梯度传播，并结合双轴偏好对齐机制，解决级联推荐系统中预排序与排序阶段的跨阶段不一致问题。

    

    工业推荐系统采用具有不同目标、特征空间和延迟约束的级联阶段。分别优化预排序和排序可能导致跨阶段不一致：上游模型可能会过滤掉下游排序器所偏好的物品，而独立调优的下游融合可能会抵消上游的改进效果。现有多任务融合方法主要关注排序阶段内的多目标融合，而跨阶段方法通常只是在上游排序中添加下游分数因子。对两个阶段融合模块的联合优化在很大程度上仍未被探索。我们提出了UniRec，一个统一的跨阶段推荐融合模型。首先，两个融合代理部分共享输入嵌入，并在单一计算图中进行训练，因此来自任一阶段的梯度都会通过共享表示进行传播并影响另一个阶段。其次，我们引入了双轴偏好对齐

    arXiv:2609.11052v1 Announce Type: new  Abstract: Industrial recommender systems use cascaded stages with different objectives, feature spaces, and latency constraints. Optimizing pre-ranking and ranking separately can create cross-stage inconsistency: upstream models may filter out items preferred by downstream rankers, and independently tuned downstream fusion can offset upstream improvements. Existing multi-task fusion methods focus on multi-objective fusion within the ranking stage, and cross-stage methods typically only add a downstream score factor to upstream ranking. Joint optimization of fusion modules across both stages remains largely unexplored.   We propose UniRec, a Unified Cross-stage Recommendation Fusion model. First, the two fusion agents partially share input embeddings and are trained in a single computation graph, so gradients from either stage propagate through the shared representation and influence the other. Second, we introduce a dual-axis preference alignment 
    
[^60]: CoinRAG：面向长上下文检索增强生成（RAG）的上下文化信息微块KV缓存复用

    CoinRAG: Contextualized Information Nugget KV Cache Reuse for Long-Context RAG

    [https://arxiv.org/abs/2608.07458](https://arxiv.org/abs/2608.07458)

    CoinRAG通过两阶段检索定位查询相关语义单元，并以组合方式复用离线计算的细粒度信息微块KV缓存，在低预填充延迟下实现了长上下文RAG效率与准确性的帕累托最优。

    

    近期关于检索增强生成（RAG）的优化研究利用块级KV缓存复用来避免对冗长的检索上下文进行重复处理，从而提高效率；然而粗粒度的块中仍然存在显著的信息冗余与噪声。本文提出CoinRAG（面向长上下文RAG的上下文化信息微块KV缓存复用），在低预填充延迟的约束下优化帕累托前沿，同时最大化准确性。该名称隐喻性地体现了我们的核心机制：正如将小代币（“硬币”）拼合起来可以累积出更大的价值，CoinRAG以组合的方式复用离线计算的细粒度微块缓存，以语义上更相关且更紧凑的方式高效地构建学习到的上下文表示。具体而言，CoinRAG不再对整个块进行编码，而是通过两阶段检索来识别检索块中与查询相关的语义单元，并无缝地……（原文摘要至此截断）

    arXiv:2608.07458v2 Announce Type: replace-cross  Abstract: Recent optimization studies on Retrieval-Augmented Generation (RAG) have exploited chunk-level KV cache reuse to avoid processing long retrieved contexts for higher efficiency, while significant information redundancy and noise still remain in the coarse-grained chunks. This paper optimizes the Pareto frontier under low prefill latency constraints while maximizing accuracy by proposing CoinRAG (Contextualized Information Nugget KV Cache Reuse for Long-Context RAG). The name metaphorically reflects our core mechanism: much like assembling small tokens (or "coins") to accumulate a larger value, CoinRAG compositionally reuses offline-computed, fine-grained nugget caches to form a learned contextual representation efficiently in a more semantically relevant but compact manner. Specifically, instead of full-chunk encoding, CoinRAG identifies query-relevant semantic units within retrieved chunks through two-stage retrieval and seamle
    
[^61]: 从文本检索卫星图像档案中的变化：如何高效融合前后时相图像

    Finding Change in Satellite Archives from Text: How to Combine Before-and-After Images Efficiently

    [https://arxiv.org/abs/2607.28571](https://arxiv.org/abs/2607.28571)

    本文在固定图像编码器与统一训练方案的条件下，系统对比了注意力机制、Mamba状态空间模型和时序瓶颈融合（TBF）三类共八种双时相图像融合设计，为文本驱动的卫星图像变化检索提供了高效融合模块的选择依据。

    

    arXiv:2607.28571v2 公告类型：replace-cross。摘要：业务化地球观测日益需要回答诸如“找出出现新建筑物的图像对”之类的查询。这意味着需要在一个由前后（双时相）卫星图像对组成的档案库中进行搜索，并根据每个图像对与变化的自然语言描述的匹配程度进行排序。执行这一匹配的组件——即融合“之前”与“之后”两个视图的融合模块——必须在查询时对众多候选图像对运行，因此其速度在很大程度上决定了每次搜索的成本。我们对如何构建该模块进行了受控比较：使用固定的图像编码器（冻结的CLIP模型）和统一的训练方案，评估了来自三个类别的八种设计：注意力机制、状态空间模型（Mamba）以及学习式压缩（我们提出的时序瓶颈融合TBF）。每种设计均在两个基准数据集（LEVIR-CC和Dubai-CC）上使用十个随机种子进行测试，因此报告……

    arXiv:2607.28571v2 Announce Type: replace-cross  Abstract: Operational Earth observation increasingly calls for answering queries such as ``find the image pairs where a new building appeared.'' This means searching an archive of before-and-after (bi-temporal) satellite image pairs and ranking each pair by how well it matches a natural-language description of the change. The component that performs this match, the fusion module that combines the ``before'' and ``after'' views, must be run at query time across many candidate pairs, so its speed largely sets the cost of every search. We present a controlled comparison of how to build that module. Using one fixed image encoder (a frozen CLIP model) and one training recipe for all variants, we evaluate eight designs drawn from three families: attention, state-space models (Mamba), and learned compression (our Temporal Bottleneck Fusion, TBF). Each design is tested on two benchmarks (LEVIR-CC and Dubai-CC) with ten random seeds, so the repor
    
[^62]: 诊断与缓解长程搜索中的上下文腐化

    Diagnosing and Mitigating Context Rot in Long-horizon Search

    [https://arxiv.org/abs/2606.29718](https://arxiv.org/abs/2606.29718)

    该论文通过对四个旗舰模型在三个基准上的系统性研究，首次揭示了“无结论搜索失败”这一此前被忽视的上下文腐化现象——模型远在用尽上下文窗口之前就放弃或给出不确定的错误答案，即使对简单查询也普遍存在，并进一步诊断与缓解这一问题。

    

    随着大语言模型（LLM）越来越多地被部署于长程搜索任务，使用大量上下文已成为常态。上下文长度增加会导致模型能力下降的担忧，即“上下文腐化”，已成为这些应用中被广泛承认的问题。然而，在深度搜索场景中，模型在大量上下文下究竟如何失败，以及现有方法能在多大程度上缓解这些失败，仍不清楚。通过对四个旗舰模型在三个基准上的系统性研究，我们发现了一个此前被忽视的现象：在大量上下文的情况下，模型远在用尽上下文窗口之前就会放弃或给出不确定的错误答案。我们将这两种失败模式称为“无结论搜索失败”。我们表明，即使对于简单和中等难度的查询，这些也是常见的失败模式，这表明模型会在其本有能力解决的查询上过早终止……

    arXiv:2606.29718v3 Announce Type: replace-cross  Abstract: Extensive context has become the norm as Large Language Models (LLMs) are increasingly deployed in long-horizon search tasks. The concern that increasing context length degrades model capabilities, known as context rot, has become a widely recognized issue for these applications. However, in deep search scenarios, it remains unclear how models actually fail under extensive context, and to what extent existing methods can mitigate such failures. Through a systematic study of four flagship models across three benchmarks, we identify a previously overlooked phenomenon: under extensive context, models give up or provide uncertain incorrect answers long before exhausting the context window. We refer to these two failure modes as inconclusive search failures. We show that these are common failure modes even for easy and moderately difficult queries, suggesting premature termination on queries that the model is capable of solving. Bas
    
[^63]: 面向技能感知人才推荐的可解释CF-RL-TOPSIS融合模型

    An Interpretable CF-RL-TOPSIS Fusion Model for Skills-Aware Talent Recommendation

    [https://arxiv.org/abs/2605.24155](https://arxiv.org/abs/2605.24155)

    本研究提出了一种可解释的CF-RL-TOPSIS融合模型，通过整合协同过滤、强化学习和熵权TOPSIS三种方法进行技能感知的人才推荐，在两个公开ICT人才数据集上显著优于现有基准方法。

    

    有效的技能感知人才推荐必须平衡行为转换模式、轨迹敏感的适应性以及可审查的职业层级标准。然而，关于这些信号如何相互作用的公开基准证据仍然有限。本研究提出CF-RL-TOPSIS，这是一种可解释的后期融合模型，它集成了一个转换感知的协同过滤分支、一个紧凑的强化学习式职业族老虎机分支，以及一个由六个语义代理指标构建的熵权TOPSIS分支；经验证筛选的融合系数保持可审计性。该模型在两个冻结的公开ICT人才历史基准数据集JobHop和Karrierewege上进行评估，采用重复的时间顺序top-5排名和配对Wilcoxon检验。在JobHop数据集上，完整的混合模型达到NDCG@5 = 0.3040 ± 0.0073，并显著超越了repeat-last、item Markov、转换感知协同过滤、CF+TOPSIS混合模型、GRU4Rec和SAS等基线方法。

    arXiv:2605.24155v2 Announce Type: replace-cross  Abstract: Effective skills-aware talent recommendation must balance behavioral transition patterns, trajectory-sensitive adaptation, and inspectable occupation-level criteria. Evidence from public benchmarks on how these signals interact, however, remains limited. This study proposes CF-RL-TOPSIS, an interpretable late-fusion model that integrates a transition-aware collaborative branch, a compact reinforcement-style occupation-family bandit, and an entropy-weighted TOPSIS branch constructed from six semantic proxies; the validation-selected fusion coefficients remain auditable. The model is evaluated on two frozen public ICT talent-history benchmarks, JobHop and Karrierewege, using repeated chronological top-5 ranking and paired Wilcoxon tests. On JobHop the full hybrid attains NDCG@5 = 0.3040 +/- 0.0073 and significantly surpasses repeat-last, item Markov, transition-aware collaborative filtering, the CF+TOPSIS hybrid, GRU4Rec, and SAS
    
[^64]: AllSERP：对多功能 AdSERP 数据集的穷举式逐元素增强

    AllSERP: Exhaustive Per-Element Enrichment of the Versatile AdSERP Dataset

    [https://arxiv.org/abs/2605.04949](https://arxiv.org/abs/2605.04949)

    AllSERP 通过为 AdSERP 数据集添加覆盖自然结果、People-Also-Ask 等非广告元素的类型化兴趣区域标注层，实现了搜索页面的穷举式逐元素增强，使 95.7% 的最终点击可归入类型化主轴 AOI。

    

    AdSERP 是一个公开语料库，包含 2,776 次搜索试验，记录了在真实 Google 结果页面上的眼动注视、光标、滚动、瞳孔和点击遥测数据，同时还包括页面截图和捕获的 HTML。其随附的边界框仅覆盖广告，而广告只获得约 16% 的主轴点击。AllSERP 为其余所有内容添加了一个类型化的兴趣区域（AOI）层：从捕获的 HTML 中标注的、锚定于截图的边界框，涵盖自然搜索结果、People-Also-Ask（大家还在问）、图片包、本地包及其他页面组件。三种类型化风格在边界框的紧凑性与覆盖范围之间进行权衡：从紧凑框到间隙填充框，再到顶部广告轮播中的逐卡片单元格。广告分区在全部 38,250 次比较中与随附的广告矩形完全一致，并且使用基于 DOM 的测试框架在整个语料库上测量了边界框的保真度。在间隙填充风格下，一旦将光标坐标转换为截图空间，95.7% 的最终点击都落在类型化的主轴 AOI 内。

    arXiv:2605.04949v4 Announce Type: replace  Abstract: AdSERP is a public corpus of 2,776 search trials with eye gaze, cursor, scroll, pupil and click telemetry recorded on real Google result pages, together with the page screenshots and captured HTML. Its shipped bounding boxes cover only advertisements, which receive about 16 % of main-axis clicks. AllSERP adds a typed area-of-interest (AOI) layer for everything else: screenshot-anchored boxes for organic results, People-Also-Ask, image packs, local packs and other widgets, labelled from the captured HTML. Three typed flavours trade box tightness for coverage, from tight boxes to gap-filled boxes to per-card cells inside the top-ads carousel. The ad partition agrees with the shipped ad rectangles on all 38,250 comparisons, and box fidelity is measured on the full corpus with a DOM-based harness. Under the gap-filled flavour, 95.7 % of final clicks land in a typed main-axis AOI once cursor coordinates are converted into screenshot space
    
[^65]: 好分数，坏数据：一种多模态一致性度量

    Good Scores, Bad Data: A Metric for Multimodal Coherence

    [https://arxiv.org/abs/2603.25924](https://arxiv.org/abs/2603.25924)

    提出多模态一致性得分（MCS），一种轻量级、无需人工标注的指标，通过将一致性分解为身份、空间、语义、决策四个维度来独立于下游模型评估多模态融合质量，其区分质量的灵敏度高于任务准确率。

    

    多模态AI系统通常通过下游任务准确率来评估，但高准确率并不意味着底层数据是一致的。一个模型可以在视觉问答（VQA）上取得高分，而其输入却相互矛盾。我们提出了多模态一致性得分（MCS），这是一种独立于任何下游模型来评估融合质量的度量指标。MCS将一致性分解为四个维度：身份、空间、语义和决策，其权重通过Nelder-Mead优化方法学习得到。我们在1,000张Visual Genome图像上使用DETR、CLIP和ViLT进行评估，并在150张COCO图像上无需重新训练进行验证。在三种融合架构中，MCS区分数据质量的灵敏度高于仅使用任务准确率（Spearman rho = 0.093 vs. 0.071）。扰动实验证实每个维度独立响应其对应的失效模式，且无交叉干扰。MCS轻量级、无需人工标注，并且能够告诉你……

    arXiv:2603.25924v2 Announce Type: replace-cross  Abstract: Multimodal AI systems are evaluated by downstream task accuracy, but high accuracy does not mean the underlying data is coherent. A model can score well on Visual Question Answering (VQA) while its inputs contradict each other. We introduce the Multimodal Coherence Score (MCS), a metric that evaluates fusion quality independent of any downstream model. MCS decomposes coherence into four dimensions, identity, spatial, semantic, and decision, with weights learned via Nelder-Mead optimization. We evaluate on 1,000 Visual Genome images using DETR, CLIP, and ViLT, and validate on 150 COCO images with no retraining. Across three fusion architectures, MCS discriminates quality with higher sensitivity than task accuracy alone (Spearman rho = 0.093 vs. 0.071). Perturbation experiments confirm each dimension responds independently to its failure mode with zero cross-talk. MCS is lightweight, requires no human annotation, and tells you no
    
[^66]: 通过事后嵌入对齐提升压缩Transformer模型的性能

    Improving the Performance of Compressed Transformers with Post-Hoc Embedding Alignment

    [https://arxiv.org/abs/2602.14635](https://arxiv.org/abs/2602.14635)

    提出了一种事后训练的对齐适配器（AlAd），通过将压缩Transformer模型的词元嵌入与性能更优的参考模型对齐，并利用局部上下文信息，从而恢复压缩所导致的性能损失。

    

    模型压缩是在资源受限环境中部署Transformer模型的必要手段。然而，压缩会降低Transformer模型在下游任务上的性能。现有解决方案如知识蒸馏（KD）和参数高效微调（PEFT）各有其局限性。知识蒸馏在教师模型的监督下训练学生模型，但需要教师模型的完整训练流程。LoRA等PEFT方法可以使模型适应新任务，但它们无法恢复压缩所造成的性能损失。我们的直觉是，通过将压缩模型的词元嵌入与性能更优的参考模型的词元嵌入进行对齐，可以改进压缩后的模型。我们通过一个放置在压缩模型最后一层之后的小型前馈对齐适配器（Alignment Adapter, AlAd）来学习这种对齐。AlAd会读取相邻词元的窗口，因此能够利用局部上下文信息。AlAd是在压缩模型生成之后进行事后训练的。

    arXiv:2602.14635v2 Announce Type: replace-cross  Abstract: Model compression is necessary for deployment of transformer models in resource-constrained environments. However, compression reduces the performance of transformer models on downstream tasks. Existing solutions such as Knowledge Distillation (KD) and Parameter Efficient Fine-Tuning (PEFT) have their own limitations. KD trains the student under the teacher's supervision, but needs the teacher's training pipeline. PEFT methods such as LoRA adapt a model to a new task, but they do not recover the loss caused by compression. Our intuition is that a compressed model can be improved by aligning its token embeddings with those of a better performing reference model. We learn this alignment with a small feed-forward Alignment Adapter (AlAd) placed after the final layer of the compressed model. AlAd reads a window of neighbouring tokens, so it can use local context. AlAd is trained post-hoc, after the compressed model has been produce
    
[^67]: 导航构思空间：用于科学思想定位的分解式概念表示

    Navigating Ideation Space: Decomposed Conceptual Representations for Positioning Scientific Ideas

    [https://arxiv.org/abs/2601.08901](https://arxiv.org/abs/2601.08901)

    该论文提出“构思空间”这一结构化表示，通过对比训练将科学知识分解为三个独立维度，实现了科学思想间概念距离的原则性测量与具有区分性的新颖性评估，帮助研究者精确定位新想法相对于现有研究的差异。

    

    科学发现需要将新想法置于不断扩大的现有知识版图之中。一个新兴的挑战是如何从快速增长的文献中识别出概念上相关的先前工作，并评估一个新想法与现有研究的区别。当前的嵌入方法通常将不同的概念方面混淆为单一表示，无法支持细粒度检索；与此同时，基于大语言模型的评估器容易受到谄媚偏好的影响，无法提供具有区分性的新颖性评估。我们引入了构思空间，这是一种结构化表示，通过对比训练将科学知识分解为三个不同的维度。我们的框架能够有原则地测量思想之间的概念距离，并对构思转换进行建模，以捕捉单个思想内部的逻辑联系。基于这一表示，我们提出了一个Hi（摘要在此处截断）

    arXiv:2601.08901v2 Announce Type: replace-cross  Abstract: Scientific discovery requires new ideas to be situated within an ever-expanding landscape of existing knowledge. An emerging challenge is how to identify conceptually relevant prior work from rapidly growing literature, and assess how a new idea differentiates from existing research. Current embedding approaches typically conflate distinct conceptual aspects into single representations and cannot support fine-grained retrieval; meanwhile, LLM-based evaluators are subject to sycophancy biases, failing to provide discriminative novelty assessment. We introduce the Ideation Space, a structured representation that decomposes scientific knowledge into three distinct dimensions learned through contrastive training. Our framework enables principled measurement of conceptual distance between ideas and modeling of ideation transitions that capture the logical connections within an idea. Building upon this representation, we propose a Hi
    

