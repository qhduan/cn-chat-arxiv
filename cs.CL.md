# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Ambient LLMs: Data-Efficient Supervised Fine-tuning via Temperature Matching](https://arxiv.org/abs/2611.23229) | 提出Ambient SFT框架，通过将微调模型的高温条件分布与冻结的预训练“噪声教师”模型进行温度匹配，利用不完美预测扩充微调数据集，从而防止过拟合和数据分布尾部的灾难性遗忘。 |
| [^2] | [What Does a Bias Head Measure? Non-Identifiability in Head-Level Bias Attribution](https://arxiv.org/abs/2611.23218) | 该论文审计了将LLM偏差归因于单个注意力头的常用方法，发现用于选头的统计量主要由提示身份驱动（占64%方差），而公平性标签、响应长度与受保护群体标记之间存在固有共线性，证明头级偏差归因本质上是不可辨识的。 |
| [^3] | [Structure--Residual Inductive Biases for Language Model Pretraining](https://arxiv.org/abs/2611.23197) | 提出一种结构—残差归纳偏置原则，通过边界瓶颈概念（BBC）和连续概念（CC）两种机制将语言模型表示中的结构与实现细节分离，并推导出相应的编码界，发现二者在深度偏好上存在显著差异。 |
| [^4] | [Polarity Composition Hypothesis: Causal Evidence from Negation in NLI](https://arxiv.org/abs/2611.23185) | 该论文提出“极性组合假说”来解释语言模型在自然语言推理中处理否定的计算机制，并通过分布式对齐搜索提供了因果证据，证明模型内部存在与假说变量对齐的因果有效子空间，且这些表示可跨否定形式迁移。 |
| [^5] | [MoE--LoRA Methods for Multilingual Adaptation: A Systematic Empirical Study](https://arxiv.org/abs/2611.23158) | 本研究首次在统一的多语言评估协议下，用200种语言和9000多个评估设置系统比较八种LoRA方法，发现小规模搜索无法可靠预测方法在大规模下的相对表现，且LoRA虽能提升翻译数据拟合度但会持续损害阅读理解能力。 |
| [^6] | [TextSphere: Efficient Text-to-Image Generation via Shared Hyperspherical Alignment and Uniformity](https://arxiv.org/abs/2611.23155) | TextSphere 通过 AdaLN 引入 CLIP 文本嵌入并提出球面语义对齐与均匀性（SSAU）目标，将高效的 Sphere Encoder 扩展到自由文本条件下，实现仅需少量前向步骤的文本到图像生成。 |
| [^7] | [Mind the Gap: A Deployed Multi-Agent System for Documentation Maintenance](https://arxiv.org/abs/2611.23142) | 该论文提出并实际部署了一个以"TriAgent"分诊智能体为核心的多智能体系统，通过持续挖掘客服工单和代码-文档不一致等显性与隐性信号，自动检测并修复客户文档与技术实现之间的漂移。 |
| [^8] | [The Illusion of Grounding: Do Vision-Language Models Weigh Retrieved Text and Images Like an Ideal Observer?](https://arxiv.org/abs/2611.23127) | 该论文构建了包含20万个条目的GIB-200K压力测试基准，并从理想贝叶斯线索组合理论出发，首次形式化定义并量化了视觉-语言模型在融合检索文本与图像证据时的“知识惯性”与“感官捕获”偏差。 |
| [^9] | [Nunchi: Ears Help Eyes in Multi-Party Conversations](https://arxiv.org/abs/2611.23115) | 该论文提出了SORBET基准数据集与Nunchi框架，通过将语音转录等对话线索与视线几何信息相结合，并利用视觉-语言模型进行权衡，实现了多方对话中被注视人物的准确识别。 |
| [^10] | [Large-scale Neuron Community Discovery Reveals Neuron-Sparse Behaviors in Language Models](https://arxiv.org/abs/2611.23112) | 该论文提出了一种无监督的“神经元社区发现”方法，通过梯度归因、图构建和社区检测，在大语言模型中找出跨输入复用且因果必要充分的神经元群组，从而证明语言模型中存在可复用的稀疏神经元计算单元。 |
| [^11] | [Natural Language Programming: Rethinking Language, Memory, and Cognition in the Era of Generative AI](https://arxiv.org/abs/2611.23106) | 本文提出自然语言编程（NLPg）这一新概念，将提示词重新定义为能够外化目标、约束与推理状态的动态认知制品，主张自然语言正从被计算的对象演变为人类与AI协作中可执行的认知媒介。 |
| [^12] | [What's in a Tree ? Hierarchical $n$-gram models as controlled approximators of language](https://arxiv.org/abs/2611.23093) | 本文提出 tree-grams——一种通过二阶和三阶张量层次化组合构建的自回归 n-gram 模型，其上下文窗口与模型容量分别由超参数 n 和 d 可控地调节，在超越传统 n-gram 基线的同时可与 Transformer 语言模型媲美，并可作为研究语言的可控近似器用于超参数迁移等实验。 |
| [^13] | [Runtime Verification of Multiple Natural Language Criteria for Agent Governance](https://arxiv.org/abs/2611.23074) | 提出了一种基于树注意力的并行验证器VFM，可在运行时对智能体输出同时评估数百条自然语言治理准则并给出逐条三元判定，其预训练语料库CriteriaBank带来了跨域性能提升。 |
| [^14] | [The Resolution Hypothesis: Agent Evolution's Key Is Identifying Improvements](https://arxiv.org/abs/2611.23068) | 智能体进化的停滞并非因为方法无法提出更好的候选产物，而是因为噪声评估无法可靠地识别出这些改进，使得选择的效果与随机无异。 |
| [^15] | [ForecastGym: A Live, Controlled, and Replayable Benchmark for Forecasting Agents](https://arxiv.org/abs/2611.23045) | ForecastGym 通过每两天记录四个商业搜索引擎 API 对预测市场问题的返回结果，并在结果揭晓后离线回放这些冻结证据，首次实现了在固定证据获取策略下对预测智能体的模型、记忆与搜索来源的公平、可控比较。 |
| [^16] | [Storing the Gemma 4 K=V Cache Once: Exact Global-KV Memory Savings in a Paged Serving Engine with Unchanged Attention Kernels](https://arxiv.org/abs/2611.23035) | This paper exploits the linear dependence between the key and value in the global attention layers of Gemma 4 31B (K=V), storing only one 640-value row without modifying the attention kernel, which reads the key and value caches through two overlapping strided views, achieving a 1.34 to 1.40 fold reduction in KV cache memory during decoding with bit-identical outputs. |
| [^17] | [TPHC: Improving Hyper-Connections with Tensor-Parallel Grouped Updates](https://arxiv.org/abs/2611.23026) | TPHC 将张量并行分片与超连接流相结合，在不增加参数和 FLOPs 的情况下使残差流更新更具表达力，仅需约 9% 的额外训练时间即可突破 mHC 四条流的扩展瓶颈。 |
| [^18] | [Large Language Models Have Not Yet Transformed Science](https://arxiv.org/abs/2611.22975) | 该研究通过分析2015至2025年间五个学科250万篇全文论文发现，除个别例外，大语言模型出现后科学实践的大多数既有趋势仍在延续，表明LLM尚未从根本上改变科学研究的方式。 |
| [^19] | [When Do Typed Decision Models Work in Transportation? Evidence from Travel Choice and Traffic Signal Control](https://arxiv.org/abs/2611.22926) | 该论文系统评估了类型化决策模型在出行方式选择预测与交通信号控制两类交通任务中的表现，发现在Swissmetro数据集上其预测准确率低于传统离散选择模型和最强的对话式大语言模型基线，表明此类模型的有效性高度依赖于具体任务类型。 |
| [^20] | [When the tail is earned by its companions: measuring label dependence in long-tailed multi-label classification](https://arxiv.org/abs/2611.22859) | 该论文提出了可在训练前评估标签依赖性的标签内估计量、读取规则与统计量，揭示稀有标签的召回表现依赖频繁伴生标签而非模型真正学到稀有标签本身的现象。 |
| [^21] | [A Full-Ledger Counterfactual Audit of Adaptive Reasoning](https://arxiv.org/abs/2611.22778) | 该论文提出FLCVA审计框架，通过在同一冻结前缀上对比立即回答与固定长度续写的反事实结果并进行全成本核算，发现随机续写能以极低损害率（约1-2%）修复大量错误答案（约19-27%），且续写正确性标签可用于指导算力分配。 |
| [^22] | [CrossEnv: Synthesizing Verifiable Cross-Environment Tasks for Long-Horizon Agent Training](https://arxiv.org/abs/2611.22777) | CrossEnv框架通过实体-工具图识别环境间连接并对齐任务数据，合成可执行、可验证的跨环境训练任务，并借助课程式训练方法让长程智能体从单环境工具使用逐步进阶到多环境协调。 |
| [^23] | [StitchRAG: Constructing Compact and Evidence-Preserving Contexts for RAG with Diffusion-Based Stitching](https://arxiv.org/abs/2611.22766) | StitchRAG提出了一种RAG框架，通过效用最大化自适应压缩检索段落，提取含锚点的句子并利用文本扩散技术将其拼接成连贯、紧凑且保留证据的上下文，在降低计算开销的同时提升事实性。 |
| [^24] | [Cross-cultural Adaptation of Recipes for Low-Resource Languages: A Benchmark and Multilingual Culinary Lexicon for Hausa, Yor\`ub\'a, and Igbo](https://arxiv.org/abs/2611.22708) | 该论文提出了包含600份专家翻译与文化适应食谱的ReciLoReL基准数据集，以及涵盖约15,000个翻译实体的多语言烹饪词汇表ReciLex，用于评估和提升大语言模型在豪萨语、约鲁巴语和伊博语等低资源语言食谱翻译中的文化适应能力。 |
| [^25] | [Route Optimization under Port Congestion and Carbon Reduction: An NSGA-II Approach](https://arxiv.org/abs/2611.22705) | 本文提出一个融合变分不等式网络均衡模型与NSGA-II算法的多目标航线优化框架，联合最小化时间、运输与碳排放成本，并以中国沿海集装箱网络（上海为主要拥堵港、宁波-舟山为替代港）实证验证了港口拥堵与碳减排双重约束下的船舶航线分配方案。 |
| [^26] | [Visibility Graph Representations of Word Embeddings for Text Classification](https://arxiv.org/abs/2611.22685) | 本文提出将平均词嵌入生成的文档向量转化为自然可见性图，并以节点度向量或邻接矩阵形式输入分类器，在AG News数据集上证明在词汇预算受限条件下，从选定700个词开始节点度表示成为最准确的分类配置。 |
| [^27] | [The Between-Round Decision in Autonomous Research: An LLM Controller That Reads the Whole Trajectory](https://arxiv.org/abs/2611.22661) | 该论文提出BRED，一个基于LLM的轮间决策控制器，通过维护最佳已验证解决方案（Champion）和过往轮次日志，让LLM直接读取完整轨迹证据来决定是改进现有方案还是开辟新方向，从而摆脱了规则、评分和搜索树等外部机制。 |
| [^28] | [Understanding Cross-Modal Inconsistency through Representation Subspaces in Omni Models](https://arxiv.org/abs/2611.22619) | 本文提出从答案位置表示中估计低维“答案子空间”，发现预测全模态模型跨模态不一致性的信息主要集中于答案子空间之外的一个可复现二维表示中，且该表示与答案子空间在干预下表现出截然不同的行为作用。 |
| [^29] | [Evolution Under Fire: Evolving Agent Safety Harnesses with Adaptive Injection Stressors](https://arxiv.org/abs/2611.22583) | 该论文提出分阶段框架 Evolution Under Fire，通过让安全防护机制与自适应注入攻击压力源相互对抗演化，自动生成可执行的安全防护，在 AgentDyn 基准上将攻击成功率从 50.4% 降至 3.0% 且不牺牲任务效用。 |
| [^30] | [AfriNORM: An Evaluation Benchmark for Normative Reasoning of LLMs Across African Cultures](https://arxiv.org/abs/2611.22580) | 本文提出AfriNORM基准——一个基于非洲社会文化场景的选择题数据集，并配合通用常识偏见得分（UCBS）和干扰项偏见得分（DBS）两个新指标，用于测量和诊断大语言模型在非洲文化规范性推理方面的局限。 |
| [^31] | [From Perception to Simulation: Unveiling and Empowering Mental Spatial Dynamics in MLLMs](https://arxiv.org/abs/2611.22541) | 该论文提出MentalGrid基准以评估多模态大语言模型在无中间视觉反馈下模拟动态空间状态的能力，并通过三阶段课程学习框架训练出SpatialWalker模型，从而显著弥补了现有模型在空间模拟上的感知与多步推理缺陷。 |
| [^32] | [JET: Just-Enough Exploration Tuning for Efficient Tool-Calling Agents in Repository-Level Question Answering](https://arxiv.org/abs/2611.22539) | JET通过在信息充足的步骤截断训练轨迹并插入理由说明进行加权微调，有效缓解了LLM智能体在仓库级问答中的“过度探索”问题，使智能体在收集到足够信息后立即作答。 |
| [^33] | [When Agent Skills Go Stale: Runtime Behavior and Selective Maintenance in Evolving Environments](https://arxiv.org/abs/2611.22525) | 该论文发现环境变化会导致智能体复用的技能状态过时，使性能下降超过10个百分点或执行成本增加近40%，并提出了一种无需额外模型调用的“选择性维护”方法来验证技能是否仍然有效。 |
| [^34] | [LMLineage: Content Lineage and Context Governance for LLM Agents](https://arxiv.org/abs/2611.22511) | LMLineage是一个无需模型白盒访问、无需额外推理的带外溯源系统，可将智能体输出的每句话归因到输入片段并给出校准置信度，并基于溯源元数据实现概念漂移微压缩，仅精准压缩不再被引用的输入文档同时逐字保留用户指令。 |
| [^35] | [Growing Clusters Lower the Bar: Profile Accretion in LLM-Based Incremental Entity Resolution](https://arxiv.org/abs/2611.22485) | 论文揭示了一种新的“画像累积”偏差：在基于LLM的增量实体解析中，模型的匹配判断不仅取决于所比较的记录，还取决于已有簇的规模——簇越大，模型接纳新记录的门槛越低，导致名称相似公司的错误匹配率显著上升而判别能力几乎不变。 |
| [^36] | [Surfacing Disagreement Beats Resolving It: A Gated Ablation Study of Provenance-Weighted Memory for Multi-Agent Debate](https://arxiv.org/abs/2611.22461) | 多智能体辩论的共享记忆应保留矛盾并将排序后的冲突集呈现给辩论各方，而非将其坍缩为单一获胜事实——这一做法在同等成本下将F1提升0.340，并把投毒攻击成功率从22/122降至7/122。 |
| [^37] | [SignAligner: Dual-Granularity Span Generation for Sign Language Alignment](https://arxiv.org/abs/2611.22409) | 提出SignAligner，一个基于提示的视觉-语言片段生成器，通过在手语词定位的字幕片段与词片段上联合训练并采用基于重叠度的奖励优化，在不使用外部手语视频资源的情况下实现了最佳的手语字幕对齐效果。 |
| [^38] | [RICoT: Recurrent Interleaved Chain-of-Thought for Page Stream Segmentation](https://arxiv.org/abs/2611.22382) | 该论文提出RICoT循环交错思维链提示框架，首次将页面流分割重新构想为对整个流的顺序处理过程，像读者一样在滑动窗口内逐页推理并携带已读内容的紧凑记录以控制成本，同时发布了包含205个真实金融页面流的FinancePSS-205基准数据集。 |
| [^39] | [NONSENSE: Non-Semantic Strings Expose Semanticity Gaps in Dense Retrieval](https://arxiv.org/abs/2611.22353) | 提出NONSENSE黑盒稀疏编辑探针，通过构造禁止复制查询词的明显非语义字符串，证明其稠密相似度得分常超过检索器最佳的查询相关段落，从而揭示了稠密检索中最近邻地位与真实语义含义相分离的语义性鸿沟。 |
| [^40] | [Bad Apples in the Kitchen: How LLM Teams Detect, Correct, and Calibrate Against Their Misaligned Members](https://arxiv.org/abs/2611.22342) | 该论文在实时具身协作环境Overcooked中注入三种基于理论的失准智能体，将LLM团队鲁棒性分解为检测、纠正与校准三个连续阶段，并发现只有当同一句话中同时包含归因和情境诊断时团队才能实现恢复。 |
| [^41] | [VRLLM: Turning LLMs into Powerful Tabular Classifiers](https://arxiv.org/abs/2611.22308) | Visa Research通过在数十亿行带标签表格数据上微调Llama 3.2 1B模型（VRLLM），首次证明LLM在大规模表格分类任务上可以超越传统表格分类器以及生产级深度学习模型。 |
| [^42] | [Teaching Language Models to Narrow Boolean Patent Searches from Search Feedback](https://arxiv.org/abs/2611.22270) | 该论文利用日本专利局检索报告提炼专家策略，并提出行级“保留引用的收窄信用”学习方法对语言模型进行离线迭代优化，使其生成的布尔专利检索策略在保持引用召回率的同时大幅收窄候选集合，将包含引用文献的中位数集合规模从 768 篇降至 50 篇以下。 |
| [^43] | [OfficePrism: Benchmarking Agents on Single-Domain Operations Across Enterprise Business Functions](https://arxiv.org/abs/2611.22262) | 该论文提出 OfficePrism 基准，通过 420 个覆盖七大企业业务职能的单域任务（结合中文工单、可执行参考见证及自动与人工筛选流程），在 124 种模型-框架-工具包配置下系统评估智能体，发现其自然语言与数据库联合通过率仅为 22.6%–76.4%。 |
| [^44] | [Where Does Hindsight Enter Clinical Agent Benchmarks? An Audit of Simulators and Case Construction](https://arxiv.org/abs/2611.22241) | 该论文提出 HindAudit 审计框架，将临床智能体基准测试中的后见之明偏差分离为三条通道并逐一测量，发现模拟实验室结果不存在信息泄露，而最大偏差来自病例文本——基准风格的病历摘要将四个大语言模型的零样本诊断准确率从 25.4% 提升至 39.7%。 |
| [^45] | [SOTA or Luck? The Winner's Curse in LLM Leaderboards](https://arxiv.org/abs/2611.22227) | 该论文揭示了LLM排行榜中因排名噪声分数而产生的“赢家诅咒”与“排名诱导偏差”会系统性高估顶级模型的性能，并提出一种m-out-of-N自助法来纠正该偏差，在14个基准测试上排名第一模型的调整幅度为0.09至3.86个百分点。 |
| [^46] | [Fine-Grained Japanese Clinical Named Entity Recognition: A Controlled Comparison of BERT and Open-Weight Large Language Models](https://arxiv.org/abs/2611.22219) | 在包含113个实体类别的细粒度日语临床命名实体识别任务中，微调的BERT-CRF基线模型（F1达0.836-0.841）显著优于开放权重大型语言模型（零样本最高F1仅0.337）。 |
| [^47] | [EGS: A Complementary Safety Framework for Educational LLMs](https://arxiv.org/abs/2611.22203) | 本文提出教育护栏系统EGS，一个轻量级、与模型无关的运行时安全框架，通过风险分级分流、情境化教学引导和输出审计，超越简单的二元拒绝模式，有效提升面向青少年的教育大语言模型的安全性。 |
| [^48] | [From Imperfective Paradox to Moorean Absurdity: Single-Chain Reasoning Lock-In and Multi-Candidate Verbalization Across Aspectual and Pragmatic Ambiguity](https://arxiv.org/abs/2611.22188) | 该论文通过四项预注册的形式化实验证实了大语言模型在体貌与语用歧义推理中存在“单链过早锁定”现象，并提出多候选言语化（MCV）方法，在单次调用中生成多个加权候选估计以缓解该锁定问题。 |
| [^49] | [Is 4-Bit Quantization More Than Noise to a Chat Model's Safety Behaviour? Attributing Behavioural Shifts with Matched Perturbations](https://arxiv.org/abs/2611.22186) | 该研究通过预注册实验和匹配扰动归因方法证明，4位量化对聊天模型安全行为的影响并非随机噪声，而是由特定舍入误差驱动的真实且近似线性的行为转变，且在不同模型上效应方向相反。 |
| [^50] | [What RL on Expert-Built Tasks Teaches Nemotron 3.5 Super](https://arxiv.org/abs/2611.22161) | 通过在五个专家构建的任务数据集上分别进行无监督阶段的强化学习后训练，NVIDIA Nemotron 3.5 Super 在十二项基准测试上取得了最高 +13.8 分的显著性能提升。 |
| [^51] | [Long-Context Generation Is a Sampling Problem (When You Turn Up the Heat)](https://arxiv.org/abs/2611.22148) | 该论文通过大规模实验证明，长上下文生成的退化本质上是采样器问题——高温下保持候选 token 池小而稳定的采样器（如 top-nsigma）能让编程智能体等长程生成任务保持高成功率，而放任低概率 token 进入的采样器则完全崩溃。 |
| [^52] | [Weight-Space Programming Doesn't Need Transformers](https://arxiv.org/abs/2611.22055) | 在无需梯度训练的权重空间编程中，三个前沿LLM在自由设计架构时都不约而同地将知识编译为可查询的稀疏表，且其性能以约1/1800的参数量超越了Transformer，表明权重空间编程并不需要Transformer。 |
| [^53] | [RLDE: A Reinforcement Learning Framework for Interpretable Iterative Document Editing](https://arxiv.org/abs/2611.22039) | 提出了RLDE框架，将文档迭代修订建模为无参考的MDP问题，通过指针网络与PPO智能体结合冻结的BART模型，依据可读性、困惑度和语法等多目标奖励实现可解释的自动化文档编辑。 |
| [^54] | [Thinking Harder Doesn't Lead to Better Choices: Reasoning Effort Increases the Confidence but Not the Correctness of LLM Shopping Agents](https://arxiv.org/abs/2611.22033) | 在三个购物基准测试中，增加LLM购物代理的推理努力并不能显著提升其选购产品的正确性，却会显著提高其报告的置信度——也就是说，模型“想得更多”只是变得更自信，而不是变得更正确。 |
| [^55] | [GauntletScore: Auditable Fact-Checking That Matches the Best Published Checkers on LLM-AggreFact](https://arxiv.org/abs/2611.22027) | GauntletScore是一种可审计的多读者证据引擎，通过向不同提供商的独立读者提问、在证据账本中记录测量并使用固定贝叶斯规则汇总（仅在两位读者一致时判定矛盾），在LLM-AggreFact完整测试集上达到79.51的平衡准确率，与最佳已发表检查器持平。 |
| [^56] | [SkillBuffBench: Measuring What Agent Skills Contribute across Harnesses, Models, Skill Access Modes and Memory](https://arxiv.org/abs/2611.22017) | SkillBuffBench 通过将技能访问方式（无技能/精选技能/自动调用）、记忆开关、四个智能体框架和多个主流模型进行交叉配对评估，系统性地衡量了智能体技能在贴近真实部署场景下的实际贡献。 |
| [^57] | [DiagTrace: An Interactive Benchmark for Clinical Decision Making](https://arxiv.org/abs/2611.22009) | DiagTrace是一个基于PubMed Central病例报告构建的交互式临床决策基准，包含510个跨八个专科的病例，用于评估大语言模型能否通过交互式信息获取从不完整信息中逐步重建诊断轨迹，突破了传统一次性提供完整患者信息的评估方式。 |
| [^58] | [Large Language Models as Nonlinear Dynamical Systems: The Three Clocks of Depth, Training, and Generation](https://arxiv.org/abs/2611.21983) | 本文提出“三时钟”框架，将语言模型在层深度、训练过程和生成过程三个尺度上的行为与非线性动力学的经典概念（不动点、分岔、同步、混沌等）系统对应，并通过汇总约200篇文献的“三时钟词典”和一系列降阶模型在两个学科之间架起桥梁。 |
| [^59] | [Modeling Document and Section Hierarchies in Embedding Space](https://arxiv.org/abs/2611.21980) | 该论文提出层次化嵌入式主题模型，通过嵌入空间中的层次先验将文档级与章节级主题分布相耦合，并借助摊销变分推断在法律、报告和论坛等多个领域的结构化语料库上取得了最低或接近最低的困惑度表现。 |
| [^60] | [Beyond Bitwidth: Deployment-Constrained Mixed-Precision Quantization for Multilingual LLM Inference on NVIDIA Blackwell](https://arxiv.org/abs/2611.21974) | 该论文在NVIDIA B200上对Qwen3-8B进行了部署约束驱动的低比特与混合精度量化实测评估，发现INT2/INT3全面退化且模块恢复效应具有强非加性，最终提出一个NVFP4与INT8混合的配置，在通过全部预注册多语言与推理精度门槛的同时将权重内存压缩至5.917 GiB。 |
| [^61] | [It's My Turn! Interpretable Methods Reveal Developmental Differences from Mother-Child Conversational Dynamics](https://arxiv.org/abs/2611.21969) | 该研究开发了一个结合序列分析、信息论测度和贝叶斯层次建模的可解释计算框架，揭示了语言发展较低的两岁儿童与母亲的对话表现出更异质的组织结构和更少的互惠式轮流交流。 |
| [^62] | [Solver to System: LLM Reasoning Across Representations and Execution in Scientific Simulation](https://arxiv.org/abs/2611.21964) | 该论文提出了Solver2System数据集，通过多表示形式、受控损坏与词汇扰动来评估大语言模型在PDE科学仿真中的跨表示一致性与物理无效性检测能力。 |
| [^63] | [Conserved Verdicts, Drifting Commitments: Identifying Harmful Overthinking](https://arxiv.org/abs/2611.21946) | 该论文提出“承诺-判定差距”框架来识别推理模型中的有害过度思考，证明仅凭最终正确性探针估计的“判定”无法预测继续思考的利弊，而“承诺”与“判定”之差恰好量化了停止思考可获得的准确率增益，并给出将准确率-预算曲线分解为链内答案变化与链存活变化的恒等式。 |
| [^64] | [DNeo: Your Parallel Draft Model Needs Better Target Feature Encoding](https://arxiv.org/abs/2611.21925) | 本文发现标准特征编码机制存在有损（残差混合破坏特征空间）、无记忆、均匀采样忽略重要隐藏状态三大缺陷，并提出特征混合注意力（MoFA）来改进推测-扩散解码中并行草稿模型的目标特征编码质量。 |
| [^65] | [Latent Fact Composition under Next-Token Training](https://arxiv.org/abs/2611.21921) | 本文提出 Baton 系列循环 Transformer，通过在第二次循环中将中间实体作为词元嵌入传递，使逐词元训练的模型能够回答中间实体从未在两跳训练样本中出现过的两跳推理问题，并归纳出决定各模型成败的三个设计条件。 |
| [^66] | [Call-Efficient ICD-10 Code Prediction via Supervised Candidate Generation and Selective LLM Invocation](https://arxiv.org/abs/2611.21891) | 该论文提出了一种结合监督式候选生成、候选重排序与门控式选择性LLM调用的高效ICD-10编码预测流程，在防泄漏验证协议下实现高准确率的同时降低大语言模型调用成本。 |
| [^67] | [Knowing Which, Not How Much: Language Models Order Party Positions They Cannot Place](https://arxiv.org/abs/2611.21878) | 研究发现，语言模型能够可靠地对日本和韩国政党的立场进行排序，却无法将这些立场准确定位到具体量级上，这揭示了“顺序”与“量级”是两种可分离的知识，也意味着现有的单实体打分基准可能误读了模型的领域知识水平。 |
| [^68] | [The Hidden Cost of MoE Diffusion Decoding: Profiling and Fixing the Sampler](https://arxiv.org/abs/2611.21852) | 该论文发现量化后MoE扩散语言模型的解码瓶颈在于画布级logits采样流水线而非专家矩阵计算，并提出两个无损优化层级（比特精确与精度对等）将其消除，从而大幅加速解码。 |
| [^69] | [TUSI: A Citation-Grounded Graph-Based Benchmark](https://arxiv.org/abs/2611.21823) | 该论文提出了TUSI-Bench——一个建立在包含7,261条法律条文的多层知识框架和法律知识图谱之上的结构化波斯语法律基准，用于对具有引用溯源能力的法律RAG系统进行严格评估。 |
| [^70] | [SimAI: Mechanistic User Simulation through Psychological Network Models](https://arxiv.org/abs/2611.21808) | SimAI是一个基于心理网络模型的机制性用户模拟框架，通过图重构目标建模心理因素间的相互作用，并借助PSYCHE数据集实现了比现有患者模拟器更高的多样性和档案一致性。 |
| [^71] | [Corrupting Ties: Investigating Robustness to Bias Injection Across Multi-Agent Network Topologies](https://arxiv.org/abs/2611.21804) | 该论文提出受控偏见注入框架与决策分歧度量指标，系统研究了多智能体网络拓扑对偏见传播的鲁棒性，发现序列网络受偏见注入影响产生的决策分歧最大。 |
| [^72] | [Bookkeeping Is Not Judgement: EJBench, an Engineering Judgement Benchmark Built from a Conjugate Heat Transfer Design Campaign](https://arxiv.org/abs/2611.21707) | 本文提出EJBench——一个基于电池冷板共轭传热设计活动构建的与软件无关的工程判断基准，考察模型能否从少量已有结果预测设计在新工况下的行为并做出决策，结果发现最佳前沿模型仅得0.71分，且模型答错时表达的置信度与答对时几乎相同。 |
| [^73] | [Disambiguating Moral Circuits and Moral Features in Language Models: Moral content, normative judgment or cue words? Evidence from 14 models](https://arxiv.org/abs/2611.21705) | 通过227对最小短语对和14个开源模型的实验，研究发现改变语言模型道德答案的注意力头并非道德专属，而是属于一个对所有类型规范性判断共用的通用评估回路，其效应与“道德上”这一线索词无关。 |
| [^74] | [Watermark Evidence under Open-Weight LLM Modification: Exact Bounds for Low-Rank Output Watermarks](https://arxiv.org/abs/2611.21702) | 该论文针对开放权重大语言模型权重被修改后水印验证失效的问题，推导出残差约束下水印证据的精确有限词表下界以及序列级检测的严格统计界，证明 OpenStamp 式验证器在模型被修改的情况下依然可靠。 |
| [^75] | [Token Commitment as a Control Surface for Steering Masked Diffusion Generation](https://arxiv.org/abs/2611.21669) | 该论文提出将令牌提交策略（即每步提交哪些令牌和提交多少令牌）作为掩码扩散模型推理引导的新型控制界面，通过反事实目标感知提交和校准自适应预算两种机制，在语言和视觉-语言任务上实现了对生成过程的有效引导。 |
| [^76] | [Evaluating Semantic Insensitivity in Financial Sentiment Analysis: Implications for Sustainable Financial Decision-Making](https://arxiv.org/abs/2611.21653) | 本文提出了智能体驱动的语义反转（ASI）对抗攻击框架，利用LLM多智能体流水线对金融新闻进行针对性词汇替换，以90.85%的成功率欺骗了FinBERT模型，揭示了金融情感分析模型对语义操纵的严重脆弱性。 |
| [^77] | [Consistency Is Not Correction: Separating Confidence, Abstention, and Recovery in LLM Agents](https://arxiv.org/abs/2611.21651) | 该论文将LLM智能体置信度信号的“纠错效用”精确分解为“门控恢复减去门控损害”，并证明当初始错误数低于配对检验所需数量时，实验在数学上不可能验证纠错机制的有效性，也无法将其与从未触发的门控区分开来。 |
| [^78] | [From Preference Data to Personalization: Tracing Sycophancy in Large Language Models](https://arxiv.org/abs/2611.21650) | 该论文提出了有效性感知的谄媚评分器和包含3,298个有效性标注探针的SycoDrift评估集，将谄媚重新定义为“不当附和”而非简单的一致，并据此系统追踪了大语言模型附和压力的三个来源：人类偏好数据、自动评判器规范与上下文个性化。 |
| [^79] | [Literary Language Mashup: Curating Fictions with Large Language Models](https://arxiv.org/abs/2611.21649) | 本文提出在GrAImes框架中让大语言模型充当文学评估者，通过对比LLM与人类评委（文学教授及爱好者）对西班牙语和英语小说的评估，发现LLM虽可作为辅助文学评估工具，但无法替代人类评委。 |
| [^80] | [Can an LLM dream? Testing the overfitted brain hypothesis on an agent with text memory](https://arxiv.org/abs/2611.21648) | 该研究将“做梦防止大脑过拟合”假说应用于具有文本记忆的LLM智能体，通过在每晚记忆重写前对当天对话记录施加细节损坏或随机文本损坏来模拟梦境扭曲，结果显示这种损坏并未带来帮助。 |
| [^81] | [FieldGuide: A Corpus Map for Token-Efficient Agentic Document Exploration](https://arxiv.org/abs/2611.21647) | FieldGuide通过将文档集合组织为主题层次结构并提供定位、范围化搜索和受限阅读工具，使智能体在新闻文档探索任务中相比原始文件访问和RAG基线显著减少词元消耗（27%-38%）和耗时（21%-41%），同时保持相当的准确性。 |
| [^82] | [Limite 1B - Violetto: Technical Report](https://arxiv.org/abs/2611.21646) | Limite 1B - Violetto 是一个仅 10 亿参数的数学推理模型，通过在精心策划的数学语料上从头训练，并结合知识蒸馏、监督微调与 RLVR 后训练，在 AIME 2026 上取得 94.01% 的成绩，性能媲美参数量数倍于它的模型。 |
| [^83] | [Asking What Matters: Disagreement-Guided Ambiguity Detection for Interactive Text-to-SQL](https://arxiv.org/abs/2611.21645) | 提出DisGuAD框架，将多个合理SQL解释之间的分歧作为判断依据，在目标SQL未知的情况下定位真正影响最终查询的歧义，从而在有限的交互轮次内更聚焦、更高效地选择澄清问题，改进交互式text-to-SQL性能。 |
| [^84] | [MInfoVisQA: A Multilingual Benchmark for Information-Rich Visual Question Answering](https://arxiv.org/abs/2611.21644) | 该论文提出了 MInfoVisQA——一个涵盖 24 种语言、8,960 个问答配置的信息丰富型视觉问答多语言基准，评估揭示了多模态大语言模型在不同语言间存在显著的鲁棒性差距，总体准确率掩盖了性能波动。 |
| [^85] | [Sleep2Word: Peripheral-to-EEG Translation via Duration-Aware Wordization](https://arxiv.org/abs/2611.21643) | Sleep2Word提出了一种时长感知的词化框架，能够将呼吸、心电或脉搏波等外周生理信号翻译成整夜脑电频谱图，从而在没有脑电设备的情况下也能开展基于脑电的睡眠分析。 |
| [^86] | [RAW: Refusal Aversion in the Wild, A Causal Measurement Method for Deployed LLMs](https://arxiv.org/abs/2611.21641) | 本文提出RAW方法，利用大语言模型在相同提示下有时拒答有时不拒答的随机性作为自然实验，首次对已部署模型的拒答进行因果测量，发现一次拒答会使约一半被测模型的用户后续再参与率下降3至17个百分点。 |
| [^87] | [ClearTalk: Data-Driven Mandarin Articulation Analysis and Personalized Communication Support for Children with Dysarthria](https://arxiv.org/abs/2611.21639) | ClearTalk 是一款基于智能手表的系统，它将普通话语音识别差异转化为可解释的音节级构音部位画像，为构音障碍儿童提供个性化练习与分阶段沟通支持，其准确率与专家清晰度评分高度相关（r=0.908），且自动选择的目标与言语病理学家共识高度一致。 |
| [^88] | [CrisisBench: an evaluation suite for crisis counseling AI](https://arxiv.org/abs/2611.21622) | CrisisBench是一个隐私保护的危机心理辅导AI评估套件，通过去标识化的真实热线对话、模拟患者的LLM以及经专家校准的评判标准来评估模型表现，结果显示现有LLM系统在危机对话中仍不及人类辅导员。 |
| [^89] | [From Repositories to Training Environments: Scaling Long-Horizon Coding Agents with Environment Evolution Loop](https://arxiv.org/abs/2611.21152) | RepoFactory提出环境演化循环，利用及时可靠的代理反馈自动将真实代码仓库转化为可验证的训练环境，并大规模生成了超过一万个人类可验证的编码智能体训练环境。 |
| [^90] | [Does the Voice Matter? Evaluating Frustration Detection on Deployed Clinical Voice-Agent Calls](https://arxiv.org/abs/2611.21001) | 在真实临床语音代理通话中，患者话语极短（约59%仅一两个词），仅依赖文本转录无法可靠检测患者的挫败情绪，语音音频信息是必要的。 |
| [^91] | [CuRSE-Bench: A Benchmark for Probing Cultural Blind Spots in Text-to-Image Safety Evaluation](https://arxiv.org/abs/2611.20821) | 该论文提出CultureCraft智能体框架并构建了CuRSE-Bench基准（包含10,709个特定文化提示词和2,648个对照提示词，覆盖六个地区和十个危害维度），揭示了现有文生图安全评估因文化盲点而无法识别仅在特定文化中构成潜在危害的内容。 |
| [^92] | [Can AI Judge You? An Empirical Audit of Deployment Readiness in LLM-assisted Hiring Systems](https://arxiv.org/abs/2611.20140) | 本文提出了一种将候选人证据、评估、权限控制与评估记录分离的LLM辅助简历筛选系统架构，并通过涵盖210份简历、14个职位和6种决策界面的受控实验，从运行就绪度与合规就绪度两个维度实证审计了其部署就绪度，发现标量分数比通过/不通过判断更能有效区分资质变化。 |
| [^93] | [B-OPSD: On-Policy Self-Distillation with Self-Generated Problem Briefs](https://arxiv.org/abs/2611.19273) | B-OPSD让模型自行从问题生成不含答案的简报（包含原理、适用条件和常见陷阱）来替代标准答案解作为教师模型的指导上下文，实现了无需真实解的在线策略自蒸馏，并在数学竞赛基准上取得了优于OPSD的准确率。 |
| [^94] | [WaferTetris: Efficient Freeform Mapping on Wafer-Scale Accelerators](https://arxiv.org/abs/2611.18527) | 提出了WaferTetris系统，通过类俄罗斯方块的自由形状映射策略打破模板固定几何布局的限制，灵活地在晶圆级加速器上打包计算，从而提升处理单元利用率和可用内存容量。 |
| [^95] | [Gamer: A Generalist Vision-Language-Action Foundation Model for Long-Horizon Real-Time Game Agent](https://arxiv.org/abs/2611.18397) | 提出Gamer，一个端到端的通用视觉-语言-动作基础模型，通过海量键鼠交互轨迹预训练、稀疏思考指令微调和稳定的多轮强化学习，实现了在复杂2D/3D交互环境中执行长时程任务的实时游戏智能体。 |
| [^96] | [MamaReason: Unfinished Reasoning and Answer Recovery in Clinical Language Models](https://arxiv.org/abs/2611.18264) | 该研究发现临床语言模型通过答案续写恢复的诊断大多已显式存在于未完成的推理轨迹中，仅直接读取这些诊断承诺（无需额外生成）即可挽救大部分续写所恢复的失败案例。 |
| [^97] | [Tracing Knowledge and Reasoning in LLM](https://arxiv.org/abs/2611.17461) | 该论文提出了一个以任务为中心的评估框架，结合注意力、激活与表示信号来刻画大语言模型在知识型与推理型任务中的内部处理模式，将评估从分数层面深入到内部机制层面。 |
| [^98] | [Lightfoot: Graph Retrieval-Augmented Generation with Exact, Model-Free Document Deletion](https://arxiv.org/abs/2611.11010) | Lightfoot 是一种图检索增强生成系统，它将图中的每个实体、链接、事实和同义词边都定义为逐段落提取结果的确定性函数，从而实现无需语言模型调用的检索以及作为数据库事务的精确文档删除（在 11,402 段语料库上删除 10% 仅需 14 秒且结果与全新索引完全一致），解决了现有 GraphRAG 系统的模型调用成本和文档删除不精确问题。 |
| [^99] | [Continued Training of Continuous Language Flows with Token Identity Squared Supervision](https://arxiv.org/abs/2611.07721) | 提出词元身份Brier流，通过直接对词元身份施加平方监督（Brier损失）解决了连续MSE目标无法区分不同概率分布的问题，从而更有效地继续训练预训练语言流模型。 |
| [^100] | [Corralling AI Risks with Grassroots Contracts](https://arxiv.org/abs/2611.06374) | 提出以“草根契约”这一数字化契约机制，来圈控AI操控全球平台、匿名AI智能体逃避追责以及无法归因的深度伪造内容这三大AI风险。 |
| [^101] | [See, Think, Use: Diagnosing Perception and Reasoning Failures in VLMs](https://arxiv.org/abs/2611.01467) | 该论文提出“看、想、用”协议，通过定位探针、文本替换探针和反事实图像对，在无需访问模型权重的情况下，将视觉语言模型的每个答案归因于感知、推理或感知使用，从而精准定位应改进的环节并指导测试时计算的分配。 |
| [^102] | [Multi-Estimate Elicitation Does Not Uniformly Debias Language Model Judgments](https://arxiv.org/abs/2611.00597) | 本研究通过预注册实验发现，在单次回复中引导语言模型给出多个加权估计值的格式并不能统一地消除其认知偏差，在锚定和现状等任务中偏差效应已接近于零，尽管多项效应量低于预设的最小感兴趣效应量，但并未能确立统计等效性。 |
| [^103] | [DAPS: Dependency-Aware Premise Selection for LLM Theorem Proving](https://arxiv.org/abs/2610.41536) | 该论文提出DAPS方法，通过首次系统挖掘53个Lean Blueprint项目构建依赖关系语料库，将数学中的依赖结构信号引入前提选择的训练数据与模型中，克服了现有基于语义相似度的选择器在检索偏斜和分布外查询上的固有局限。 |
| [^104] | [Language Models Show Conformity Bias in RAG: When the Majority Endorses a Poisoned Claim](https://arxiv.org/abs/2610.41012) | 本研究发现当RAG检索上下文中多数相互一致的投毒文档共同支持错误信息时，语言模型的投毒检测率从25%显著降至10%，揭示了语言模型存在类似社会从众心理的从众偏差，使其更容易被多数派的投毒信息误导。 |
| [^105] | [Multi-Label Agreement: A Flexible Framework for Inter-Annotator Reliability](https://arxiv.org/abs/2610.37292) | 本文提出了多标签一致性（MLA）框架，通过保留完整二值标签向量并采用平均成对汉明一致性，为多标签标注任务的标注者间信度评估提供了灵活方案，同时证明了其数学形式与Fleiss/Scott型系数及Krippendorff's alpha密切相关。 |
| [^106] | [MemT-1.0: Towards Memory-Native Evolution](https://arxiv.org/abs/2610.26825) | MemT-1.0提出了一种将记忆内化为基础模型固有计算能力的新架构，通过Gated DeltaNet-2、稳定的token级记忆更新机制和Mixture-of-Storage来扩展原生记忆容量，其发布的模型家族在记忆与通用能力评估中超越了参数化记忆基线。 |
| [^107] | [AdaRSI: On-Demand Recursive Self-Improvement for Embodied Agents](https://arxiv.org/abs/2610.20474) | AdaRSI提出了一种按需递归自我改进框架，先改进技能与外围框架、再通过重试归因持续失败、最后仅微调不可靠的子步骤，从而在无需人类示范的情况下显著提升具身智能体完成长时程任务的能力。 |
| [^108] | [SarcasTraps: Effect of Sarcasm on Turkish Hate Speech Detection](https://arxiv.org/abs/2610.16957) | 本文构建了SarcasTraps——一个包含479条数据、沿仇恨言论与讽刺双维度标注的土耳其语外部测试集，并通过实验揭示讽刺会显著干扰仇恨言论检测：微调的编码器模型在讽刺性仇恨言论上召回率下降，零样本大语言模型同样受到影响。 |
| [^109] | [ANCHOR: Assessing Naturalistic Constraint Hierarchies and Ordering in Repository-Grounded Agentic Coding](https://arxiv.org/abs/2610.16859) | 提出 ANCHOR 基准，通过层次化约束分类体系、自动化约束组合流水线和基于轨迹的评估框架，首次系统性地评估基于仓库的编码智能体遵循工程约束（如修改范围、工作流顺序、验证等）的能力，弥补了现有评估仅关注功能正确性的不足。 |
| [^110] | [Communication, Vocalisation, and Language](https://arxiv.org/abs/2610.16490) | 本文主张大语言模型并非真正“理解”语言，而是在“执行”语言，其机制是捕捉文化智能并将符号扎根于社会学意义上的实践，而非形成符号的分布式表示。 |
| [^111] | [Mingbird: A Local-First Agent Harness Enabling Small Open Models to Complete Real Tasks](https://arxiv.org/abs/2610.02001) | 提出了Mingbird，一个面向Windows和Ollama的本地优先智能体框架，通过十项针对性补偿机制（如字节级净零预填充预算、完成门控和签名级循环检测）来克服小模型的典型失败形式，使2-9B的开放模型能够在普通笔记本上完成真实任务，并证明许多失败源于框架而非模型本身。 |
| [^112] | [Spike-driven Vision-Language-Action Model](https://arxiv.org/abs/2609.39514) | 提出了首个支持机器人操作端到端直接训练的脉冲驱动视觉-语言-动作（VLA）框架，利用脉冲神经网络的稀疏事件驱动计算和多赢家脉冲融合机制，解决了传统大型Transformer模型延迟高、能耗大而难以在资源受限平台部署的问题。 |
| [^113] | [Can Language Models Learn to Forecast Stock Prices](https://arxiv.org/abs/2609.36914) | 该论文在按时间顺序推进的股票价格沙盒中，通过对 Qwen3-4B 进行 SFT 与 PPO 后训练，探索语言模型能否学会利用价格、成交量等信息预测股票未来收益。 |
| [^114] | [OpenWhistle: A Large-Scale Longitudinal Dataset and Benchmark of Bottlenose Dolphin Vocalizations](https://arxiv.org/abs/2609.34839) | 该论文发布了OpenWhistle——迄今最大的公开海豚发声数据集，包含五年间录制的约18万条宽吻海豚哨声（114小时）及8,354条专家标注样本，并提供了可复现的哨声类型检测与分类基准。 |
| [^115] | [FLARE: A Full-Lifecycle Dense Supervision Paradigm for Long-Horizon Coding Agents via Generative Reward Model](https://arxiv.org/abs/2609.23808) | 提出FLARE，一种由轻量级生成式奖励模型驱动的全生命周期密集监督范式，通过离线因果诊断框架RADAR提取无后见之明的监督信号，为长程编码智能体提供实时步骤级风险反馈，解决稀疏二值奖励带来的信用分配危机。 |
| [^116] | [RPMem: Learning Long-Term Recurrent Parametric Memory Across Sessions for LLM Agents](https://arxiv.org/abs/2609.23466) | RPMem提出了一种两阶段架构，将LLM智能体的每个会话编译为模型无关的潜在记忆，经任务训练的循环门控整合后映射为LoRA参数，从而实现跨会话的长期参数化记忆演化，并在更换骨干模型时保持记忆可迁移。 |
| [^117] | [The Memory Trust Gap: Capability-Dependent Failures in Persistent-Memory Agents](https://arxiv.org/abs/2609.01852) | 该论文提出并量化了“记忆信任差距”现象：持久记忆智能体会过度信任（而非混淆于）过期的存储事实并覆盖权威证据，且这种失效受模型能力门控——规模越大的模型在过期记忆被伪装成最新信息时崩溃反而越严重。 |
| [^118] | [Arabic Safety Alignment as Selective Refusal: An Empirical Study of SFT, DPO, and Guard Calibration](https://arxiv.org/abs/2608.29378) | 该研究提出用良性拒绝率 B 和有害提示拒绝率 H 双指标评估阿拉伯语大模型安全对齐，发现仅拒绝式 SFT 会坍缩为全面拒绝，而精选混合 SFT 配置能在低良性拒绝率下实现 90% 以上的有害提示拒绝，且 DPO 与推理护栏的效果因模型而异、并非统一提升。 |
| [^119] | [Towards Safer RAG: Only Agents Capable of System 2 Thinking may Access Untrusted Documents](https://arxiv.org/abs/2608.17153) | 本文提出一种新的安全原则，即仅允许具备系统2推理能力的代理访问不可信文档，以减少RAG系统中的知识投毒攻击影响，并引入新指标量化检测与影响间的差异。 |
| [^120] | [Asymmetric Discourse Homogenization and Shared Language Technology: Evidence from Reddit](https://arxiv.org/abs/2608.13674) | 本文通过Reddit数据发现，2022年底后保守派政治话语的同质化趋势不对称地加剧，而进步派无此变化，且该效应由累积AI语言模型影响驱动而非单一事件。 |
| [^121] | [Query Timing Produces Opposite Positional Biases Between LLMs and Humans](https://arxiv.org/abs/2608.12387) | 本研究揭示了LLMs在位置偏差上表现出与人类相反的模式，且新模型中的偏差更为显著，表明查询时机对模型评估机制有重要影响。 |
| [^122] | [One Frozen Simulator Is Not Enough: Simulator Collapse in Multi-Agent RL](https://arxiv.org/abs/2608.12253) | 该论文揭示了多智能体强化学习中使用单一冻结模拟器会导致策略泛化失败，并提出推理时的言语化采样和训练时的协同训练两种方法来解决模拟器崩溃问题。 |
| [^123] | [Role Steering of Language Models for Social Simulations](https://arxiv.org/abs/2608.00023) | 该论文提出了一种激活引导筛选工作流和新方法 CastVectors，用于为语言模型智能体注入可验证的角色条件化行为，在社会模拟中实现了显著更高的角色档案对齐度（63.2 对比 41.1），同时保持了词汇多样性。 |
| [^124] | [RecHarness: A Bandit-Routed Agentic Harness for Self-Evolving Recommender Systems](https://arxiv.org/abs/2607.29241) | 提出RecHarness框架，通过将优化过程解耦为“多臂老虎机路由器选择修改方向+LLM生成具体优化假设与代码修改”两个步骤，并借助跳跃盆地机制在局部优化停滞时触发结构性跳跃，实现推荐模型的自演化自动优化。 |
| [^125] | [Evidence Attribution in Visual Document Understanding without Coordinates or Region Labels](https://arxiv.org/abs/2607.24651) | 该研究提出以“模型逐字引用证据、由多模态检索器定位出处”的语言接口替代传统的坐标边界框接口，表明视觉文档理解中的归因幻觉部分源于坐标表达能力的局限。 |
| [^126] | [Where Is the Tradeoff in Using Third-Party API Routers for Agentic Software Development?](https://arxiv.org/abs/2607.23624) | 该论文首次实证研究了编码智能体中第三方API路由器的路由器端注入风险，通过四个隐蔽程度递增的干预级别（如响应替换）揭示其可对软件开发任务造成真实且难以检测的影响，而现有客户端权限机制难以有效防御。 |
| [^127] | [RL Post-Training Builds Compositional Reasoning Strategies](https://arxiv.org/abs/2607.07646) | 强化学习后训练并非只是放大预训练模型已有的基础技能，而是通过“先强化基础规约、再发现顺序与并行组合程序”的分阶段组合机制，将基础技能重组为新的更高层级推理策略，从而解决预训练模型几乎无法解决的问题。 |
| [^128] | [Localized LoRA-MoE: Mixture of Low-Rank Experts With Adaptive Routing](https://arxiv.org/abs/2607.05114) | 提出Localized LoRA-MoE框架，将局部空间分块与上下文条件自适应专家路由相结合，提供了集中式块级路由和去中心化单元级路由两种架构变体，以解决传统PEFT方法中的梯度干扰和静态拓扑无法适应上下文切换的问题。 |
| [^129] | [The Language-Energy Divide: Measuring Energy Costs of Multilingual LLM Inference](https://arxiv.org/abs/2606.21869) | 该论文系统性测量了大语言模型多语言推理的能耗，揭示了不同语言间高达179倍的能耗差异，并发现高能耗语言同时伴随最低的任务准确率，存在“成本+性能”双重惩罚现象。 |
| [^130] | [Rhythm of the Deep: Two-tier acoustic organization of sperm-whale codas from click waveforms to second-order sequence dependence](https://arxiv.org/abs/2606.16084) | 该研究发现抹香鲸尾音的声学组织具有两级嵌套结构：尾音内部反复出现的点击声波形差异与点击间隔时间互补，而尾音之间反复出现的整体尾音形式本身携带序列依赖性。 |
| [^131] | [ICA Lens: Non-Gaussianity Reveals Interpretable Directions in Language Models](https://arxiv.org/abs/2606.11722) | 该论文发现语言模型激活中可解释方向具有非高斯投影分布这一统计特征，并提出可以通过直接搜索非高斯方向来发现可解释特征，而无需依赖稀疏重建等间接学习目标。 |
| [^132] | [Beyond Retrieval: Learning Compact User Representations for Scalable LLM Personalization](https://arxiv.org/abs/2606.04547) | 提出TAP-PER框架，通过时序注意力前缀嵌入将用户偏好编码为紧凑的可学习表示，摆脱了检索式个性化对检索质量的依赖以及参数式个性化随用户规模增长的高昂存储成本，实现可扩展的大语言模型个性化。 |
| [^133] | [What Am I Missing? Question-Answering as Hidden State Probing](https://arxiv.org/abs/2605.31561) | 本文提出让学生模型向教师模型提问作为推理时干预手段，并通过在提问前后学生隐状态上训练探测器，证明在教师给出答案之前即可预测推理轨迹的最终正确性，揭示了提问能够暴露模型隐状态中的自我诊断信号。 |
| [^134] | [Beyond Bilingual Transfer: Multilingual Code-Switching in Instruction Tuning](https://arxiv.org/abs/2605.29414) | 该研究首次探索了在指令微调中混合英语、日语、韩语和中文四种语言的多语言语码转换数据构建方法，实验证明句子级多语言语码转换能持续提升LLM的多语言理解能力，其效果超越传统仅限双语迁移的方案。 |
| [^135] | [In-Context Optimization for Retrieval-Augmented Generation: A Gradient-Descent Perspective](https://arxiv.org/abs/2605.26356) | 该论文将检索增强生成（RAG）形式化为上下文内优化过程，证明单层线性自注意力可在统一的线性化RAG目标上执行一步梯度下降，并以此理论洞察指导查询与检索证据之间交互方式的设计。 |
| [^136] | [RICE-PO: Turning Retrieval Interactions into Credit Signals for Reasoning Agents](https://arxiv.org/abs/2605.26352) | RICE-PO 是一个无评论家的策略优化框架，通过将高不确定性可执行动作作为锚点、评估局部反事实分支并将信用有条件地传播给潜在推理步骤，把检索交互转化为局部化信用信号，从而解决交互式推理检索智能体训练中的信用分配难题。 |
| [^137] | [Distributional Alignment as a Criterion for Designing Task Vectors in In-Context Learning](https://arxiv.org/abs/2605.20730) | 该论文提出以分布对齐作为设计任务向量的准则，引入指标 $d_{\text{NTP}}$ 来衡量基于任务向量的推理与基于上下文学习的推理之间下一个词概率的差异，该指标与下游任务准确率呈强负相关，从而为任务向量质量提供了直接的评估与设计依据。 |
| [^138] | [Uncertainty Quantification for Large Language Diffusion Models](https://arxiv.org/abs/2605.14570) | 本文首次系统研究了大语言扩散模型（LLDMs）的不确定性量化问题，提出了从迭代去噪过程中提取的轻量级零样本不确定性信号，并通过结合掩码扩散似然与轨迹语义相异度，将先进的不确定性量化方法适配到LLDMs。 |
| [^139] | [TableVista: Benchmarking Multimodal Table Reasoning under Visual and Structural Complexity](https://arxiv.org/abs/2605.05955) | TableVista是一个包含3,000个表格问题、30,000个多模态样本的综合基准，通过评估29个前沿模型揭示了当前基础模型虽能稳定应对多样渲染风格，但在复杂结构布局和纯视觉设置下存在显著的性能下降。 |
| [^140] | [WHBench: Evaluating Frontier LLMs with Expert-in-the-Loop Validation on Women's Health Topics](https://arxiv.org/abs/2604.00024) | 该论文提出WHBench——一个由专家深度参与验证构建的女性健康评估基准，通过对22个前沿大语言模型在47个专家设计的场景和23项标准下的系统测试，发现没有任何模型的平均表现超过75%，暴露了现有模型在女性健康领域的安全性、临床准确性和公平性方面的重大缺陷。 |
| [^141] | [ToolFlood: Beyond Selection -- Hiding Valid Tools from LLM Agents via Semantic Covering](https://arxiv.org/abs/2603.13950) | 本文提出ToolFlood，一种针对工具增强型LLM智能体的检索层攻击，通过注入少量语义上覆盖众多用户查询的对抗性工具来主导top-k检索结果，从而将所有良性工具挤出智能体上下文。 |
| [^142] | [Thinking Is Not Telling: Information Disclosure in User-Service LLM Agents](https://arxiv.org/abs/2602.07796) | 该论文研究了用户服务LLM智能体中的信息披露机制，发现智能体的内部推理（配备"think"工具进行思考）反而会损害其与用户交互时的信息披露效果。 |
| [^143] | [Stress-Testing Emotional Support Models: Moving from Homogeneous to Diverse Help Seekers](https://arxiv.org/abs/2601.07698) | 本文提出一个由九个心理语言特征驱动、基于专家混合架构训练的可控求助者模拟器，以多样化且细粒度可控的方式模拟真实求助者行为，从而对情感支持聊天机器人进行更严格的压力测试。 |
| [^144] | [PSM: Prompt Sensitivity Minimization via LLM-Guided Black-Box Optimization](https://arxiv.org/abs/2511.16209) | 该论文提出PSM框架，将提示词加固形式化为效用约束下的优化问题，通过在系统提示词后附加保护性文本层（SHIELD），并利用LLM作为黑盒优化器进行搜索，从而最小化对抗攻击下的系统提示词泄露风险。 |
| [^145] | [Response Attack: Exploiting Contextual Priming to Jailbreak Large Language Models](https://arxiv.org/abs/2507.05248) | 本文提出“响应攻击”（RA）框架，通过在对话中策略性地将中间产生的轻度有害响应作为上下文启动器，再发出触发提示，从而利用上下文启动效应这一此前被忽视的漏洞实现对大语言模型的越狱。 |
| [^146] | [Visual Contextual Attack: Jailbreaking MLLMs with Image-Driven Context Injection](https://arxiv.org/abs/2507.02844) | 该论文提出了一种新颖的“以视觉为中心的越狱”设定及VisCo攻击方法，通过图像驱动的上下文注入构建完整真实的越狱语境，从而有效攻击多模态大语言模型。 |
| [^147] | [Say It Another Way: Auditing LLMs with a User-Grounded Automated Paraphrasing Framework](https://arxiv.org/abs/2505.03563) | 该论文提出AUGMENT框架，通过基于用户行为的受控提示词改写来审计大语言模型，能够揭示无约束改写方式下被掩盖的系统性弱点。 |
| [^148] | [Clustering in pure-attention hardmax transformers and its role in sentiment analysis](https://arxiv.org/abs/2407.01602) | 本文从动力系统角度严格证明了纯注意力硬最大值Transformer在层数趋于无穷时输入会收敛到由“领导者”点决定的聚类平衡态，并据此构建了完全可解释的Transformer模型，通过将无意义词聚类到语义核心词周围来有效捕捉上下文，从而解决情感分析问题。 |

# 详细

[^1]: 环境大语言模型：通过温度匹配实现数据高效的监督微调

    Ambient LLMs: Data-Efficient Supervised Fine-tuning via Temperature Matching

    [https://arxiv.org/abs/2611.23229](https://arxiv.org/abs/2611.23229)

    提出Ambient SFT框架，通过将微调模型的高温条件分布与冻结的预训练“噪声教师”模型进行温度匹配，利用不完美预测扩充微调数据集，从而防止过拟合和数据分布尾部的灾难性遗忘。

    

    为自回归语言模型收集大规模、多样化且高质量的后训练数据集是极具挑战性的。因此，监督微调（SFT）通常应用于远小于预训练规模的数据集，这会导致过拟合以及对数据分布尾部信息的灾难性遗忘。为了防止这种崩溃，我们提出了Ambient SFT，这是一个利用不完美预测来扩充微调数据集的原则性框架。在我们的方法中，预训练模型的“冻结”版本充当一个带噪声的教师，微调后的模型只需在粗粒度上与其匹配。特别是，我们要求后训练模型产生的高温条件分布版本应与预训练模型的高温条件分布版本相匹配。温度超参数控制着对预训练模型的信任程度，其极限为T→∞。

    arXiv:2611.23229v1 Announce Type: cross  Abstract: Collecting large-scale, diverse, and high-quality post-training datasets for autoregressive language models is challenging. Consequently, Supervised Fine-Tuning (SFT) is typically applied to datasets significantly smaller than those used for pre-training, which leads to overfitting and the catastrophic forgetting of the tails of the data distribution. To prevent this collapse, we introduce Ambient SFT, a principled framework for leveraging imperfect predictions to augment the fine-tuning set. In our method, a "frozen" version of the pre-trained model serves as a noisy teacher that the fine-tuned model must match at a coarse level. In particular, we require that the high-temperature versions of the conditional distributions produced by the post-training model should match the high-temperature versions of the pre-training ones. The temperature hyperparameter controls the level of trust in the pre-trained model, with the limit $T\to \inft
    
[^2]: 偏差注意力头究竟测量了什么？头级偏差归因中的不可辨识性

    What Does a Bias Head Measure? Non-Identifiability in Head-Level Bias Attribution

    [https://arxiv.org/abs/2611.23218](https://arxiv.org/abs/2611.23218)

    该论文审计了将LLM偏差归因于单个注意力头的常用方法，发现用于选头的统计量主要由提示身份驱动（占64%方差），而公平性标签、响应长度与受保护群体标记之间存在固有共线性，证明头级偏差归因本质上是不可辨识的。

    

    目前将大语言模型中的社会偏差定位到单个注意力头并通过消融（ablation）来消除偏差的方法，通常仅通过其对公平性指标的影响来验证，而从未验证过用于选择注意力头的统计量是否真正测量了偏差。我们在 Qwen-2.5-7B-Instruct 和 Llama-3.1-8B-Instruct 上对这一方法进行了审计。该头级图谱与一个不依赖标签的响应长度图谱相关系数分别达到 ρ = -0.950 和 -0.768，表面上似乎识别出一个混淆因素。然而，一项打破长度与标签耦合关系的干预实验否定了这一解读：在提示保持固定的情况下，响应长度仅能解释该统计量方差的 1%。真正能解释该统计量的是提示本身的身份，占方差的 64%，且其排序与长度并不呈单调关系。在任何固定的提示内，公平性标签、响应长度与受保护群体标记的出现三者之间的共线性高达 φ = 0.95，这部分源于标注规则的定义本身，因此任何观察性分析都无法将该图谱归因于任何单一因素（摘要原文在此处截断）。

    arXiv:2611.23218v1 Announce Type: new  Abstract: Methods that localise social bias in large language models to individual attention heads and remove it by ablation are validated by their effect on a fairness metric, never by evidence that the statistic selecting the heads measures bias. We audit this on Qwen-2.5-7B-Instruct and Llama-3.1-8B-Instruct. The head map correlates at $\rho = -0.950$ and $-0.768$ with a label-blind response-length map, apparently identifying a surface confound. An intervention that breaks the length-label coupling rejects this reading: with the prompt held fixed, response length explains 1% of the statistic's variance. What does explain it is prompt identity, at 64% of variance, in an ordering that is not monotonic in length. Within any fixed prompt the fairness label, response length and the presence of a protected-group token are collinear at $\phi = 0.95$, partly by definition of the labelling rule, so no observational analysis can attribute the map to any 
    
[^3]: 面向语言模型预训练的结构—残差归纳偏置

    Structure--Residual Inductive Biases for Language Model Pretraining

    [https://arxiv.org/abs/2611.23197](https://arxiv.org/abs/2611.23197)

    提出一种结构—残差归纳偏置原则，通过边界瓶颈概念（BBC）和连续概念（CC）两种机制将语言模型表示中的结构与实现细节分离，并推导出相应的编码界，发现二者在深度偏好上存在显著差异。

    

    下一个词预测被普遍依赖用于同时学习预测行为和内部表示。我们提出这样一个问题：表示结构本身是否应当作为一种显式的归纳偏置。受Kolmogorov复杂度两部分编码观点的启发，我们研究了一种结构—残差原则，在该原则中，表示将结构与实现细节分离开来。我们以两种互补的方式实例化这一原则：边界瓶颈概念（BBC）将信息从词元片段通过边界载体进行路由，同时保留局部自回归计算；连续概念（CC）则对表示几何进行正则化，并诱导出将表示谱分解为低频结构与高频残差的形式。我们为两种视角推导了编码界，并证明当片段内路径能量趋于消失时，经多重性校正的边界注意力变为精确的。在实证方面，BBC与CC表现出截然不同的深度偏好。

    arXiv:2611.23197v1 Announce Type: cross  Abstract: Next-token prediction is relied upon to learn both predictive behavior and internal representations. We ask whether representation structure itself should be an explicit inductive bias. Motivated by the two-part coding view of Kolmogorov complexity, we study a structure--residual principle in which a representation separates structure from realization detail. We instantiate this principle in two complementary ways. Boundary Bottleneck Concept (BBC) routes information from token segments through boundary carriers while preserving local autoregressive computation. Continuous Concept (CC) regularizes representation geometry and induces a spectral decomposition into low-frequency structure and high-frequency residual. We derive coding bounds for both views and show that multiplicity-corrected boundary attention becomes exact as the within-segment path energy vanishes. Empirically, BBC and CC exhibit strikingly different depth preferences: 
    
[^4]: 极性组合假说：来自自然语言推理中否定的因果证据

    Polarity Composition Hypothesis: Causal Evidence from Negation in NLI

    [https://arxiv.org/abs/2611.23185](https://arxiv.org/abs/2611.23185)

    该论文提出“极性组合假说”来解释语言模型在自然语言推理中处理否定的计算机制，并通过分布式对齐搜索提供了因果证据，证明模型内部存在与假说变量对齐的因果有效子空间，且这些表示可跨否定形式迁移。

    

    我们研究了语言模型在自然语言推理（NLI）中如何处理否定，并提出了“极性组合假说”，这是对这一计算过程的一个高层次因果模型。行为评估表明，语言模型的预测在很大程度上符合所提出的假说。通过使用分布式对齐搜索，我们识别出了与极性组合假说中的变量相一致的因果有效子空间。这些表示能够跨不同否定形式迁移，并部分迁移到MNLI的一个子集上。我们公开了数据集和代码。

    arXiv:2611.23185v1 Announce Type: new  Abstract: We investigate how language models process negation in natural language inference (NLI). We propose the \textit{Polarity Composition Hypothesis}, a high-level causal model of this computation. Behavioral evaluations show that language-model predictions largely conform to the proposed hypothesis. Using Distributed Alignment Search, we identify causally effective subspaces aligned with the variables in the polarity-composition hypothesis. These representations transfer across negation forms and partially to a subset of MNLI. We release the dataset and code.
    
[^5]: 面向多语言适配的MoE-LoRA方法：一项系统性实证研究

    MoE--LoRA Methods for Multilingual Adaptation: A Systematic Empirical Study

    [https://arxiv.org/abs/2611.23158](https://arxiv.org/abs/2611.23158)

    本研究首次在统一的多语言评估协议下，用200种语言和9000多个评估设置系统比较八种LoRA方法，发现小规模搜索无法可靠预测方法在大规模下的相对表现，且LoRA虽能提升翻译数据拟合度但会持续损害阅读理解能力。

    

    路由式低秩适配器为提升大语言模型（LLM）在各类任务上的性能提供了一种参数高效的手段。因此，人们假定它们也可用于改善LLM对低资源语言的支持——目前这些模型对英语的服务远好于全球数千种其他语言。然而，这一假设从未在统一的多语言评估协议下得到实证验证。我们正是针对这一空白开展研究：在冻结的基础模型上，使用200种独特语言、超过9,000个评估设置，对八种低秩适配方法进行了比较。我们的结果表明，缩减规模的搜索能够识别出LoRA方法的可行配置，但无法可靠地预测其相对性能，因为排名最高的配置在大规模评估中从未表现最佳。LoRA能可靠地提升模型在翻译数据上的语言建模拟合度，却持续降低阅读理解能力，并在（原文摘要在此处截断）……

    arXiv:2611.23158v1 Announce Type: new  Abstract: Routed low-rank adapters offer a parameter-efficient means to improve the performance of large language models (LLMs) across tasks. Hence, it is assumed that they can also be used to improve the support of low-resource languages in LLMs, which currently serve English far better than the thousands of other languages spoken worldwide. However, this assumption has never been verified empirically under a shared multilingual evaluation protocol. We address exactly this gap by comparing eight low-rank adaptation (LoRA) methods on a frozen base model using 200 unique languages in more than 9{,}000 evaluation settings. Our results suggest that a reduced-scale search can identify viable configurations of LoRA methods but does not reliably predict their relative performance, as the top-ranking configuration is never best at scale. LoRA reliably improves language-model fit on translation data, consistently degrades reading comprehension, and benefi
    
[^6]: TextSphere：基于共享超球面对齐与均匀性的高效文本到图像生成

    TextSphere: Efficient Text-to-Image Generation via Shared Hyperspherical Alignment and Uniformity

    [https://arxiv.org/abs/2611.23155](https://arxiv.org/abs/2611.23155)

    TextSphere 通过 AdaLN 引入 CLIP 文本嵌入并提出球面语义对齐与均匀性（SSAU）目标，将高效的 Sphere Encoder 扩展到自由文本条件下，实现仅需少量前向步骤的文本到图像生成。

    

    文本到图像生成的最新进展显著提升了图像质量，尤其是借助需要多次推理步骤的基于扩散模型的方法。相比之下，Sphere Encoder 提供了一种高效的生成框架，能够以更少的前向步骤生成图像。然而，将其扩展到自由形式的文本并非易事，因为其类条件设计缺乏跨模态对齐。为应对这一挑战，我们提出了 TextSphere，一种高效的文本到图像生成框架，它通过语言条件机制和跨模态对齐对 Sphere Encoder 进行了扩展。我们通过自适应层归一化（AdaLN）引入预训练的 CLIP 文本嵌入来引导图像生成，同时保留原有的球面采样机制。为弥合跨模态表示之间的差距，我们引入了球面语义对齐与均匀性（SSAU）目标，共同促进语义对齐和表示的均匀性（原文摘要在此处被截断）。

    arXiv:2611.23155v1 Announce Type: cross  Abstract: Recent advances in text-to-image generation have substantially improved image quality, particularly through diffusion-based methods that require multiple inference steps. In contrast, Sphere Encoder offers an efficient generative framework capable of producing images with fewer forward steps. However, extending it to free-form text is nontrivial, because its class-conditioned design lacks cross-modal alignment. To address this challenge, we propose TextSphere, an efficient text-to-image generative framework that extends Sphere Encoder with language conditioning and cross-modal alignment. We incorporate pretrained CLIP text embeddings through adaptive layer normalization (AdaLN) to guide image generation while retaining the original spherical sampling mechanism. To bridge the cross-modal representation gap, we introduce a Spherical Semantic Alignment and Uniformity (SSAU) objective that jointly promotes semantic alignment and representa
    
[^7]: 注意差距：一个用于文档维护的已部署多智能体系统

    Mind the Gap: A Deployed Multi-Agent System for Documentation Maintenance

    [https://arxiv.org/abs/2611.23142](https://arxiv.org/abs/2611.23142)

    该论文提出并实际部署了一个以"TriAgent"分诊智能体为核心的多智能体系统，通过持续挖掘客服工单和代码-文档不一致等显性与隐性信号，自动检测并修复客户文档与技术实现之间的漂移。

    

    “AI革命”使科技行业的各团队能够以前所未有的速度交付软件功能。然而，当代码以比以往更快的速度演进时，面向客户的文档却常常难以跟上步伐。随着用户越来越依赖以文档作为主要交互接口的基于大语言模型（LLM）的智能体，这种文档差距的影响日益加剧：过时的信息会被传播并被当作权威信息对待。用于识别文档滞后于代码的信号其实早已大量存在，既有显性信号（例如，已解决的客服工单能够揭示文档缺失之处），也有隐性信号（例如，可通过自动化循环检测出的代码与文档之间的不一致）。我们部署了一个多智能体系统，其核心是我们称之为"TriAgent"的分诊智能体，该系统持续提取这些信号并修复文档漂移。在43天的影子模式运行中，我们的系统对499个工单进行了分诊，并发现了92个文档差距。

    arXiv:2611.23142v1 Announce Type: cross  Abstract: The "AI revolution" has enabled teams across the technology industry to deliver software features at unprecedented velocity. While code evolves faster than ever, customer-facing documentation often struggles to stay up to speed. As users increasingly rely on LLM-based agents that consume documentation as their primary interface, the impact of this documentation gap worsens: stale information propagates and is treated as authoritative. Signals for identifying where documentation has fallen behind already exist in abundance, both explicit (e.g., resolved support tickets revealing what documentation lacks) and implicit (e.g., code-documentation discrepancies detectable from automated loops). We deploy a multi-agent system, built around a triage agent we call "TriAgent," that continuously extracts these signals and repairs documentation drift. Over 43 days of shadow-mode operation, our system triaged 499 tickets and surfaced 92 gaps in the
    
[^8]: 接地的幻觉：视觉-语言模型是否像理想观察者那样权衡检索到的文本与图像？

    The Illusion of Grounding: Do Vision-Language Models Weigh Retrieved Text and Images Like an Ideal Observer?

    [https://arxiv.org/abs/2611.23127](https://arxiv.org/abs/2611.23127)

    该论文构建了包含20万个条目的GIB-200K压力测试基准，并从理想贝叶斯线索组合理论出发，首次形式化定义并量化了视觉-语言模型在融合检索文本与图像证据时的“知识惯性”与“感官捕获”偏差。

    

    检索增强的视觉-语言模型（VLMs）从两种可能各自失效的证据来源中生成答案：检索到的文本可能是错误的或含噪声的，图像也可能退化或与文本相矛盾。我们提出了GIB-200K——一个基于COCO和Visual Genome构建、包含200,000个图像-问题条目的基准数据集，每个条目都被扩展为受控的压力测试条件：硬负例检索噪声（1-20个文档）、视觉歧义（高斯模糊与运动模糊、低资源裁剪）以及带有反事实翻转对的跨模态冲突。我们不仅测量性能退化，还探究模型如何组合这两种模态。基于理想贝叶斯线索组合理论，我们推导出可检验的预测，从而给出“知识惯性”（相对于模型自身的图像证据，文本被过度加权）与“感官捕获”（相反情形）的形式化定义，并通过经测量误差校正的回归来估计权重，同时验证了该方法对已知权重的恢复能力。反事实翻转分解……（原文摘要在此处截断）

    arXiv:2611.23127v1 Announce Type: cross  Abstract: Retrieval-augmented vision-language models (VLMs) answer from two sources of evidence that can each fail: the retrieved text can be wrong or noisy, and the image can be degraded or contradicted. We introduce GIB-200K, 200,000 image-question items built from COCO and Visual Genome, each expanded into controlled stress conditions: hard-negative retrieval noise (1-20 documents), visual ambiguity (Gaussian and motion blur, low-resource crops) and cross-modal conflict with counterfactual flip pairs. Instead of only measuring degradation, we ask how models combine the two modalities. From ideal Bayesian cue combination we derive a testable prediction that yields formal definitions of Knowledge Inertia (text over-weighted relative to the model's own image evidence) and Sensory Capture (the reverse), and we estimate the weights with a measurement-error-corrected regression whose recovery of known weights we verify. A counterfactual flip decomp
    
[^9]: Nunchi：多方对话中耳朵帮助眼睛

    Nunchi: Ears Help Eyes in Multi-Party Conversations

    [https://arxiv.org/abs/2611.23115](https://arxiv.org/abs/2611.23115)

    该论文提出了SORBET基准数据集与Nunchi框架，通过将语音转录等对话线索与视线几何信息相结合，并利用视觉-语言模型进行权衡，实现了多方对话中被注视人物的准确识别。

    

    识别一个人正在看向谁对于理解多方对话至关重要，然而随着场景中人数增多，仅凭视线几何关系会变得模糊不清。对话本身提供了几何信息所缺乏的线索，例如谁在说话以及说了什么，但目前没有视线数据集在众多多方场景中提供这些线索。我们提出了SORBET，一个包含来自1,578个片段的3,340个样本的基准数据集，具有同步的语音、转录文本和注视对象标签，涵盖了比现有视线数据集更多的多方场景。已有的视线模型难以利用这些线索，因为它们形式各异、必须与正确的人相关联，并且可能与视线几何信息相冲突。我们提出了Nunchi框架，它从转录文本以及视线模型和说话人模型的输出中构建文本提示，通过共享标签将每条线索与候选人物关联起来，并让视觉-语言模型（VLM）权衡视线几何信息与对话线索，从而识别出被注视的人。

    arXiv:2611.23115v1 Announce Type: cross  Abstract: Identifying whom a person looks at is central to understanding multi-party conversations, yet gaze geometry alone becomes ambiguous as more people appear. Conversations offer cues that geometry lacks, such as who is speaking and what is said, but no gaze dataset provides them for many multi-party scenes. We introduce SORBET, a benchmark of 3,340 samples from 1,578 clips with synchronized speech, transcripts, and attended person labels, covering more multi-party scenes than existing gaze datasets. Prior gaze models struggle to use these cues, which come in different forms, must be linked to the right person, and can conflict with gaze geometry. We propose Nunchi, a framework that constructs a text prompt from the transcript and the outputs of gaze and speaker models, links every cue to the candidates with shared labels, and lets a vision-language model (VLM) weigh gaze geometry against conversational cues to identify the attended person
    
[^10]: 大规模神经元社区发现揭示语言模型中的神经元稀疏行为

    Large-scale Neuron Community Discovery Reveals Neuron-Sparse Behaviors in Language Models

    [https://arxiv.org/abs/2611.23112](https://arxiv.org/abs/2611.23112)

    该论文提出了一种无监督的“神经元社区发现”方法，通过梯度归因、图构建和社区检测，在大语言模型中找出跨输入复用且因果必要充分的神经元群组，从而证明语言模型中存在可复用的稀疏神经元计算单元。

    

    在语言模型的神经元中，我们能发现哪些可复用的计算单元？先前的研究表明，单次前向传播通常可以在事后被稀疏化为小型神经元群组，但这些群组是否构成跨输入复用的连贯计算实体仍不清楚。我们提出了神经元社区发现，这是一种无监督方法，用于寻找跨提示反复出现、且在其所跨越的层范围内对常见计算既因果必要又因果充分的神经元群组。该方法结合了基于梯度的归因分析、连接反复被共同归因的神经元的图结构，以及社区检测。将该方法应用于 Llama-3.1-8B 并使用一千万条提示，它发现了数千个候选社区。我们通过三项因果测试在保留提示上评估它们：社区必要性、层子集内的充分性以及层必要性。这些测试识别出了那些移除后会破坏计算功能的神经元群组。

    arXiv:2611.23112v1 Announce Type: cross  Abstract: What reusable units of computation can we discover within a language model's neurons? Prior work has shown that individual forward passes can often be sparsified post-hoc to small groups of neurons, but it has been unclear whether these groups form coherent computational entities reused across inputs. We introduce neuron community discovery, an unsupervised method for finding neuron groups that recur across prompts and are causally necessary and sufficient, within the layers they span, for common computations. The method combines gradient-based attribution, a graph connecting neurons that are repeatedly attributed together, and community detection. Applied to Llama-3.1-8B using ten million prompts, it discovers thousands of candidate communities. We evaluate them on held-out prompts through three causal tests: community necessity, sufficiency within a layer subset, and layer necessity. These tests identify groups whose removal disrupts
    
[^11]: 自然语言编程：在生成式人工智能时代重新思考语言、记忆与认知

    Natural Language Programming: Rethinking Language, Memory, and Cognition in the Era of Generative AI

    [https://arxiv.org/abs/2611.23106](https://arxiv.org/abs/2611.23106)

    本文提出自然语言编程（NLPg）这一新概念，将提示词重新定义为能够外化目标、约束与推理状态的动态认知制品，主张自然语言正从被计算的对象演变为人类与AI协作中可执行的认知媒介。

    

    生成式人工智能正在改变自然语言在计算系统中的角色。在经典的自然语言处理（NLP）中，语言被视为计算的对象，即需要被分析、分类、翻译或生成的东西。大语言模型（LLM）挑战了这一观点，使自然语言越来越作为一种可执行的认知媒介发挥作用，人类通过它来构建推理、组织问题求解，并调节人机混合工作流程。本文提出了自然语言编程（NLPg）这一概念，作为传统NLP在理论和认识论上的延伸。NLPg不仅关注语言理解本身，而是将提示词概念化为一种动态认知制品，它将目标、约束、假设、验证标准和中间推理状态外化出来。因此，提示不再仅仅是一种技术交互手段，而是一种对话式认知的形式。

    arXiv:2611.23106v1 Announce Type: cross  Abstract: Generative Artificial Intelligence is transforming the role of natural language in computational systems. In classical Natural Language Processing (NLP), language was treated as an object of computation: something to analyse, classify, translate, or generate. Large Language Models (LLMs) challenge this perspective, enabling natural language to function increasingly as an executable cognitive medium through which humans structure reasoning, orchestrate problem solving, and regulate hybrid human/AI workflows.   This paper introduces Natural Language Programming (NLPg) as a theoretical and epistemological extension of traditional NLP. Rather than focusing on language understanding alone, NLPg conceptualises prompts as dynamic cognitive artefacts that externalise goals, constraints, assumptions, validation criteria, and intermediate reasoning states. Prompting is thus not merely a technical interaction technique, but a form of dialogic cog
    
[^12]: 树中有什么？层次化 n-gram 模型作为语言的可控近似器

    What's in a Tree ? Hierarchical $n$-gram models as controlled approximators of language

    [https://arxiv.org/abs/2611.23093](https://arxiv.org/abs/2611.23093)

    本文提出 tree-grams——一种通过二阶和三阶张量层次化组合构建的自回归 n-gram 模型，其上下文窗口与模型容量分别由超参数 n 和 d 可控地调节，在超越传统 n-gram 基线的同时可与 Transformer 语言模型媲美，并可作为研究语言的可控近似器用于超参数迁移等实验。

    

    层次化模型已被提出作为研究语言的合成环境，但它们与现实语言语料库的相关性仍未得到充分探索。在这项工作中，我们提出了 tree-grams，这是一类自回归生成式 n-gram 模型，通过对二阶和三阶张量的层次化组合来构建条件概率张量 p(x_n | x_{1:n-1})。其中，n 控制最大上下文窗口，而秩超参数 d 控制模型容量，由此产生了一个可解释且可系统性扩展的语言近似器家族。我们展示了 tree-grams 如何超越现有的 n-gram 基线，同时与基于深度 Transformer 的语言模型保持竞争力。我们还通过将 tree-grams 用作生成模型，展示了这种受控近似的实用性。特别是，我们表明当在更简单的数据上调优更窄的模型时，μP 下的超参数迁移可以变得更加稳健。

    arXiv:2611.23093v1 Announce Type: cross  Abstract: Hierarchical models have been proposed as synthetic settings for studying language. However, their relevance to realistic language corpora remains underexplored. In this work, we introduce tree-grams, autoregressive generative $n$-gram models that build the conditional probability tensor $p(x_n \mid x_{1:n-1})$ through a hierarchical composition of order-2 and order-3 tensors. $n$ controls the maximal context window, whereas a rank hyperparameter $d$ controls model capacity, yielding an interpretable and systematically scalable family of approximators of language. We show how tree-grams outperform existing $n$-gram baselines, while remaining competitive with deep transformer-based language models. We demonstrate the utility of such controlled approximations by using tree-grams as generative models. In particular, we show that hyperparameter transfer under $\mu$P can be made more robust when narrower models are tuned on simpler data, ge
    
[^13]: 面向智能体治理的多条自然语言准则运行时验证

    Runtime Verification of Multiple Natural Language Criteria for Agent Governance

    [https://arxiv.org/abs/2611.23074](https://arxiv.org/abs/2611.23074)

    提出了一种基于树注意力的并行验证器VFM，可在运行时对智能体输出同时评估数百条自然语言治理准则并给出逐条三元判定，其预训练语料库CriteriaBank带来了跨域性能提升。

    

    AI智能体受数百条自然语言准则的约束，然而现有的验证器无法高效地将智能体的输出与大量此类准则进行比对评估，并报告哪些准则被满足、被违反或不适用。我们提出了VFM，一种树注意力验证器，它对共享上下文和目标文本进行编码，并行地对N条自然语言准则进行评分，返回相互独立的逐准则三元判定结果。VFM在CriteriaBank语料库上进行了预训练，该语料库包含394K个（上下文、目标、准则、判定）元组，涵盖超过321K条不同的准则字符串。经微调的VFM-4B在一个合成治理基准上优于GPT-5.4。在DynaBench多准则基准上，它在轨迹级失败检测方面优于DynaGuard-4B，并且其逐准则判定结果能够定位被违反的规则，支持将问题聚焦升级至更大的思维链评判器。我们证明CriteriaBank预训练可以提升跨域（泛化能力）。

    arXiv:2611.23074v1 Announce Type: cross  Abstract: AI agents are governed by hundreds of natural language criteria, yet existing verifiers cannot efficiently evaluate an agent's output against a large set of such criteria and report which are satisfied, violated, or inapplicable. We introduce the VFM, a tree-attention verifier that encodes the shared context and target text, and scores N natural language criteria in parallel, returning independent per-criterion ternary verdicts. The VFM is pretrained on CriteriaBank, a corpus of 394K (context, target, criterion, verdict) tuples spanning over 321K distinct criterion strings. A finetuned VFM-4B outperforms GPT-5.4 on a synthetic governance benchmark. On the DynaBench multi-criterion benchmark, it outperforms DynaGuard-4B on trace-level failure detection, and its per-criterion verdicts localize the violated rule and support focused escalation to a larger chain-of-thought judge. We show that CriteriaBank pretraining improves cross-domain t
    
[^14]: 分辨率假说：智能体进化的关键在于识别改进

    The Resolution Hypothesis: Agent Evolution's Key Is Identifying Improvements

    [https://arxiv.org/abs/2611.23068](https://arxiv.org/abs/2611.23068)

    智能体进化的停滞并非因为方法无法提出更好的候选产物，而是因为噪声评估无法可靠地识别出这些改进，使得选择的效果与随机无异。

    

    智能体进化遵循一个简单的循环：提议者撰写候选产物（如提示词、技能、记忆），选择机制则接受得分更高的那些候选产物。这一循环已展现出前景，但实践者面临一个难题：不同方法在各篇论文中互有胜负，且进化在某些任务上常常无法实现自我提升。我们将这一难题归结为分辨率假说：智能体进化停滞，并非因为方法无法提出更好的产物，而是因为带有噪声的评估无法可靠地识别出这些更好的产物。通过跨越七种进化方法和六个模型（超过1000万次LLM调用）的实验，我们发现该假说在所有使用评估得分的两个层面都得到了支持。在候选层面（选择产物）上，有益的候选产物确实存在，但选择机制在区分有益与有害产物方面的表现并不比随机更好，因此误差在多轮迭代中不断累积。在方法层面（对方法进行排名）上，典型的几个百分点的差距小于……

    arXiv:2611.23068v1 Announce Type: new  Abstract: Agent evolution follows a simple loop: a proposer writes candidate artifacts (prompts, skills, memories) and selection accepts those that score higher. This loop has shown promise, yet practitioners face a puzzle: methods trade wins across papers, and evolution often fails to bootstrap on certain tasks. We trace this puzzle to the Resolution Hypothesis: agent evolution stalls not because methods fail to propose better artifacts, but because noisy evaluations cannot reliably identify them. Across seven evolution methods and six models (over 10M LLM calls), we find support for the hypothesis at both levels where evaluation scores are used. At the candidate level (choosing artifacts), helpful candidate artifacts exist, but selection is no better than random at telling them from harmful ones, so errors accumulate across rounds. At the method level (ranking methods), typical margins of a few percentage points are smaller than the difference b
    
[^15]: ForecastGym：一个面向预测智能体的实时、可控且可回放的基准测试

    ForecastGym: A Live, Controlled, and Replayable Benchmark for Forecasting Agents

    [https://arxiv.org/abs/2611.23045](https://arxiv.org/abs/2611.23045)

    ForecastGym 通过每两天记录四个商业搜索引擎 API 对预测市场问题的返回结果，并在结果揭晓后离线回放这些冻结证据，首次实现了在固定证据获取策略下对预测智能体的模型、记忆与搜索来源的公平、可控比较。

    

    基于大语言模型（LLM）构建的智能体越来越多地通过搜索网络并对所找到的信息进行推理来预测真实世界事件。此类智能体的预测好坏既取决于检索到的证据，也取决于模型本身，因此对证据的控制对于评估至关重要。当每个系统各自进行搜索时，实时评估会将检索与推理混杂在一起，而回溯式搜索则可能通过日期过滤器或历史页面在当今的排名泄露后续信息。要比较不同的模型、搜索来源和记忆策略，既需要固定的证据，也需要该证据在问题揭晓前如何演变的记录。我们提出了 ForecastGym，它每两天记录四个商业网络搜索 API 针对开放预测市场问题所返回的结果，并在结果揭晓后离线回放这些冻结的证据，从而在固定的证据获取策略下比较模型、记忆以及所捕获的搜索来源。其首个发布版本涵盖1（摘要在此处截断）

    arXiv:2611.23045v1 Announce Type: new  Abstract: Agents built on large language models (LLMs) increasingly forecast real-world events by searching the Web and reasoning over what they find. How well such an agent forecasts depends on both the retrieved evidence and the model, making control over that evidence essential for evaluation. Live evaluations can mix retrieval with reasoning when each system searches on its own, while retrospective search can leak later information through date filters or today's ranking of historical pages. Comparing models, search sources, and memory strategies requires both fixed evidence and a record of how it evolved before resolution. We introduce ForecastGym, which records what four commercial web search APIs return for open prediction-market questions every two days and replays this frozen evidence offline once outcomes are known, comparing models, memory, and captured search sources under a fixed evidence-acquisition policy. Its first release covers 1
    
[^16]: The K=V Cache of Gemma 4 Only Needs to Be Stored Once: Achieving Exact Global-KV Memory Savings in a Paged Serving Engine Without Modifying the Attention Kernel

    Storing the Gemma 4 K=V Cache Once: Exact Global-KV Memory Savings in a Paged Serving Engine with Unchanged Attention Kernels

    [https://arxiv.org/abs/2611.23035](https://arxiv.org/abs/2611.23035)

    This paper exploits the linear dependence between the key and value in the global attention layers of Gemma 4 31B (K=V), storing only one 640-value row without modifying the attention kernel, which reads the key and value caches through two overlapping strided views, achieving a 1.34 to 1.40 fold reduction in KV cache memory during decoding with bit-identical outputs.

    

    In the global attention layers of Gemma 4 31B, the value is the normalized key projection, and the key is the value multiplied by a fixed weight, with rotary position encoding applied to only one quarter of its dimensions. The Gemma 4 technical report points out that this design can reduce the global KV cache by up to 37.5%, but the open-source serving engines we inspected still store a complete key and a complete value: 1024 values per KV head per token. We instead store only one row of 640 values, which is exact in real-number arithmetic, and an unmodified paged attention kernel reads it as its key cache and value cache through two overlapping strided views. In a vLLM prototype using the 16-bit 31B model, the outputs are bit-identical to an emulation of the same arithmetic on the original cache; 2 of 432 final answers differ from the original engine (within its run-to-run noise range); and at 32K to 64K tokens, the steady decoding KV memory per request decreases by a factor of 1.34 to 1.40. On a pair of prefill/decode...

    arXiv:2611.23035v1 Announce Type: cross  Abstract: In the global attention layers of Gemma 4 31B, the value is the normalized key projection, and the key is the value scaled by a fixed weight, with rotary position encoding on one quarter of its dimensions. The Gemma 4 report states that this design reduces the global KV cache by up to 37.5%, but the open serving engines that we checked still store a full key and a full value: 1,024 values per KV head and token. We store one 640-value row instead, exact in real arithmetic, and an unchanged paged attention kernel reads it through two overlapping strided views as its key cache and its value cache. In a vLLM prototype with the 16-bit 31B model, the outputs are bit-identical to an emulation of the same arithmetic on the stock cache, 2 of 432 final answers differ from the stock engine (within its run-to-run noise), and the steady decode KV memory per request falls by a factor of 1.34 to 1.40 at 32K to 64K tokens. On a prefill/decode pair of 
    
[^17]: TPHC：通过张量并行分组更新改进超连接

    TPHC: Improving Hyper-Connections with Tensor-Parallel Grouped Updates

    [https://arxiv.org/abs/2611.23026](https://arxiv.org/abs/2611.23026)

    TPHC 将张量并行分片与超连接流相结合，在不增加参数和 FLOPs 的情况下使残差流更新更具表达力，仅需约 9% 的额外训练时间即可突破 mHC 四条流的扩展瓶颈。

    

    超连接将 Transformer 的残差流扩展为 $N$ 条并行流，而流形约束超连接（mHC）使其在大规模场景下保持稳定。然而，当流数量超过四条时，mHC 几乎没有收益，部分原因在于每个子层仅向所有流写入单一输出，限制了各流更新方式的差异性。在本工作中，我们提出 TPHC，它将张量并行（TP）分片与 HC 流相集成，以促进各流之间更强的独立性。具体而言，我们为每个 TP 分片路由一个不同的残差流混合，并在回写时保持各分片的输出相互独立，使得每条流可以接收这些输出的不同混合。这一设计使残差流更新更具表达力，且无需为注意力或 FFN 子层增加任何参数或 FLOPs。凭借对 TP 下流分片的原生支持，在两者均使用四条流的情况下，TPHC 在 8B 模型上仅比 mHC 多消耗 9% 的训练时间。

    arXiv:2611.23026v1 Announce Type: new  Abstract: Hyper-Connections (HC) expand the residual stream of a Transformer into $N$ parallel streams, and manifold-constrained HC (mHC) stabilizes them at scale. However, mHC benefits little from more than four streams, partly because each sublayer writes a single output to all streams, limiting how differently they can be updated. In this work, we propose TPHC, which integrates tensor-parallel (TP) shards with HC streams to promote greater independence across streams. Specifically, we route a separate mixture of the residual streams to each TP shard and keep the individual shard outputs separate during write-back, so each stream can receive a different mixture of those outputs. This design makes the residual-stream updates more expressive without adding parameters or FLOPs to the attention or FFN sublayers. With native support for stream sharding under TP, TPHC requires only 9% more training time than mHC on an 8B model when both use four strea
    
[^18]: 大语言模型尚未改变科学

    Large Language Models Have Not Yet Transformed Science

    [https://arxiv.org/abs/2611.22975](https://arxiv.org/abs/2611.22975)

    该研究通过分析2015至2025年间五个学科250万篇全文论文发现，除个别例外，大语言模型出现后科学实践的大多数既有趋势仍在延续，表明LLM尚未从根本上改变科学研究的方式。

    

    2022年底大语言模型（LLMs）的出现引发了对科学未来的诸多疑问，但要评估其影响，需要理解科学实践此前是如何演变的。我们分析了2015年至2025年间五个学科的250万篇全文论文，在词汇、结构、认识论和社会等多个维度上追踪了广泛的变化趋势。除了一些词汇频率的变化外，大多数2023年之前的趋势仍在延续，可检测到的中断十分有限。在少数显著的例外中，亚洲计算机科学领域合作团队的快速增长明显加速，而生物医学领域主题多样性的明显下降则主要反映了论文格式的标准化。我们的研究为在大语言模型应用日益深入的背景下监测全球科学的健康状况奠定了实证基础。

    arXiv:2611.22975v1 Announce Type: cross  Abstract: The emergence of large language models (LLMs) in late 2022 has raised questions about the future of science, but assessing their impact requires understanding how scientific practices were already evolving. We analyze 2.5 million full-text papers across five disciplines from 2015 to 2025, tracing trends over a broad spectrum of lexical, structural, epistemological, and social dimensions. Aside from frequency shifts in some words, most pre-2023 trends continued with limited detectable breaks. Among the notable exceptions, growth in computer science coauthor teams accelerated significantly in Asia, while apparent declines in biomedical topic diversity largely reflected standardization in article formats. Our study establishes an empirical foundation for monitoring the health of global science as LLM adoption deepens.
    
[^19]: 类型化决策模型何时在交通领域有效？来自出行选择与交通信号控制的证据

    When Do Typed Decision Models Work in Transportation? Evidence from Travel Choice and Traffic Signal Control

    [https://arxiv.org/abs/2611.22926](https://arxiv.org/abs/2611.22926)

    该论文系统评估了类型化决策模型在出行方式选择预测与交通信号控制两类交通任务中的表现，发现在Swissmetro数据集上其预测准确率低于传统离散选择模型和最强的对话式大语言模型基线，表明此类模型的有效性高度依赖于具体任务类型。

    

    类型化决策模型将自然语言理解与结构化接口相结合，在不生成冗长文本回复的情况下返回所选动作以及选项级别的概率分布。其低推理成本和响应延迟使其对于具有预定义备选方案的交通问题颇具吸引力。本文研究了这类模型在两个代表性交通任务中的有效性、鲁棒性和效率：个体出行方式选择预测和序贯交通信号控制。我们系统地将TypeSafe Jev和GPT-6 Luna Decisions与传统交通方法、强化学习算法以及四个基于对话的大语言模型进行了对比评估。在Swissmetro数据集上的实验表明，类型化决策模型的预测准确率低于传统离散选择模型和最强的基于对话的基线模型，且其性能存在显著差异。

    arXiv:2611.22926v1 Announce Type: cross  Abstract: Typed decision models combine natural-language understanding with a structured interface that returns a selected action and an option-level probability distribution without generating lengthy textual responses. Their low inference cost and response latency make them attractive for transportation problems with predefined alternatives. This paper investigates their effectiveness, robustness, and efficiency in two representative transportation tasks: individual travel-mode choice prediction and sequential traffic signal control. We systematically evaluate TypeSafe Jev and GPT-6 Luna Decisions against conventional transportation methods, reinforcement learning algorithms, and four chat-based large language models. Experiments on the Swissmetro dataset show that typed decision models achieve lower prediction accuracy than traditional discrete choice models and the strongest chat-based baseline, while their performance varies substantially w
    
[^20]: 当尾部由其伴生标签挣得：测量长尾多标签分类中的标签依赖性

    When the tail is earned by its companions: measuring label dependence in long-tailed multi-label classification

    [https://arxiv.org/abs/2611.22859](https://arxiv.org/abs/2611.22859)

    该论文提出了可在训练前评估标签依赖性的标签内估计量、读取规则与统计量，揭示稀有标签的召回表现依赖频繁伴生标签而非模型真正学到稀有标签本身的现象。

    

    稀有标签很少单独出现，分类器可以从与其共现的频繁标签（即其“伴生标签”）中恢复出该稀有标签，而无需真正学到这个标签本身。现有的诊断方法都需要已训练好的模型：没有一种方法能在训练前判断某个语料库是否会出现这种失效，也无法揭示标准聚合指标所掩盖的问题。Macro-F1，以及任何基于逐标签混淆计数构建的度量，都不会因标签召回率在“有伴生标签出现”与“无伴生标签出现”两种情形之间的分配方式不同而改变。我们引入了刻画这种划分的标签内估计量，提出了一种读取规则（在该规则下，只有当无伴生召回率上升时，依赖性的下降才会被计入），并提出一个基于训练标签的统计量，它能够在训练之前界定报告召回率可能超过无伴生召回率的程度。在23个语料库、7种训练目标和五类标签相关机制上的实验表明，在定义了该估计量的全部六个语料库中，无伴生召回率均存在不足，长尾部分中高达18.4%的（召回）……

    arXiv:2611.22859v1 Announce Type: cross  Abstract: A rare label seldom occurs alone, and a classifier can recover it from the frequent labels it co-occurs with, its companions, without having learned the label itself. Existing diagnostics of this need a trained model: none says before training whether a corpus admits the failure, or what the standard aggregate hides. Macro-F1, like any measure built from per-label confusion counts, is unchanged by how a label's recall divides between occurrences with and without a companion. We introduce a within-label estimand of that division, a reading rule under which a fall in dependence counts only if companion-free recall rises, and a statistic of training labels that bounds, before training, how far reported recall can exceed companion-free recall. Across 23 corpora, 7 training objectives and five families of label-correlation mechanisms, companion-free recall falls short on all six corpora where the estimand is defined. Up to 18.4% of the tail
    
[^21]: 自适应推理的全账本反事实审计

    A Full-Ledger Counterfactual Audit of Adaptive Reasoning

    [https://arxiv.org/abs/2611.22778](https://arxiv.org/abs/2611.22778)

    该论文提出FLCVA审计框架，通过在同一冻结前缀上对比立即回答与固定长度续写的反事实结果并进行全成本核算，发现随机续写能以极低损害率（约1-2%）修复大量错误答案（约19-27%），且续写正确性标签可用于指导算力分配。

    

    arXiv:2611.22778v1 公告类型：new。自适应推理必须决定是立即给出答案，还是花费更多算力继续推理。然而，预测当前答案是错误的，并不能证明进一步的推理就能将其改善。我们提出了全账本反事实价值审计（FLCVA），它在统一的回答流程下，将立即给出的答案与来自同一冻结前缀的四个固定长度续写进行对比，并对策略所使用的每一个信号的成本进行核算。在GSM8K和MATH-500上，随机的256-token续写在Qwen3-8B的26.6%案例和DeepSeek-R1-Distill-Qwen-7B的18.9%案例中修复了原本错误的答案，而在1.4%和1.9%的案例中损害了原本正确的答案。来自独立采样续写的正确性标签在两个主要模型中均能预测另一续写的结果。在离线回放中，利用这些标签来分配算力提升了DeepSeek的性能，而Qwen3上的增益尚无定论。在完整成本核算下，所测试的轻量级、短视的（原文在此处截断）……

    arXiv:2611.22778v1 Announce Type: new  Abstract: Adaptive reasoning must decide whether to answer now or spend more compute. Predicting that the current answer is wrong does not establish whether further reasoning will improve it. We introduce the Full-Ledger Counterfactual Value Audit (FLCVA), which compares an immediate answer with four fixed-length continuations from the same frozen prefix under one answer procedure and accounts for the cost of every signal used by a policy. On GSM8K and MATH-500, a random 256-token continuation recovers a wrong answer in 26.6% of Qwen3-8B cases and 18.9% of DeepSeek-R1-Distill-Qwen-7B cases, while damaging a correct answer in 1.4% and 1.9%. Correctness labels from independently sampled continuations predict a separate continuation's outcome in both primary models. In offline replay, using these labels to allocate compute improves performance on DeepSeek; the Qwen3 gain remains inconclusive. Under full cost accounting, the tested lightweight, myopic
    
[^22]: CrossEnv：为长程智能体训练合成可验证的跨环境任务

    CrossEnv: Synthesizing Verifiable Cross-Environment Tasks for Long-Horizon Agent Training

    [https://arxiv.org/abs/2611.22777](https://arxiv.org/abs/2611.22777)

    CrossEnv框架通过实体-工具图识别环境间连接并对齐任务数据，合成可执行、可验证的跨环境训练任务，并借助课程式训练方法让长程智能体从单环境工具使用逐步进阶到多环境协调。

    

    可执行的合成环境为长程智能体提供了可扩展的训练任务和可验证的反馈。现有的环境合成方法主要在单个环境内生成任务，然而这些任务无法充分捕捉真实世界中跨服务的交互场景，即一个服务中的操作依赖于从另一个服务获取的信息。合成此类任务极具挑战性，因为跨环境生成的数据必须保持一致性，才能支持可执行且可验证的交互。为应对这一挑战，我们提出了CrossEnv，该框架利用实体-工具图识别环境之间的连接，并对齐任务特定数据，从而合成可执行、可验证的跨环境任务。我们进一步提出了CrossEnv-Curriculum，它从单环境工具使用逐步过渡到跨更多环境和业务领域的协调，最后再引入用户交互。

    arXiv:2611.22777v1 Announce Type: new  Abstract: Executable synthetic environments provide scalable training tasks and verifiable feedback for long-horizon agents. Existing environment synthesis methods primarily generate tasks within individual environments. However, these tasks do not adequately capture real-world interactions across services, where actions in one service depend on information obtained from another. Synthesizing such tasks is challenging because data generated across environments must remain consistent to support executable and verifiable interactions. To address this challenge, we introduce CrossEnv, a framework that uses an entity-tool graph to identify connections between environments and aligns task-specific data to synthesize executable, verifiable cross-environment tasks. We further propose CrossEnv-Curriculum, which progresses from single-environment tool use to coordination across more environments and business domains, before introducing user interaction. Ac
    
[^23]: StitchRAG：通过基于扩散的拼接为RAG构建紧凑且保留证据的上下文

    StitchRAG: Constructing Compact and Evidence-Preserving Contexts for RAG with Diffusion-Based Stitching

    [https://arxiv.org/abs/2611.22766](https://arxiv.org/abs/2611.22766)

    StitchRAG提出了一种RAG框架，通过效用最大化自适应压缩检索段落，提取含锚点的句子并利用文本扩散技术将其拼接成连贯、紧凑且保留证据的上下文，在降低计算开销的同时提升事实性。

    

    检索增强生成（RAG）通过将大型语言模型（LLM）锚定于外部证据来提升事实性。然而，冗长的检索上下文会带来计算开销，甚至可能因噪声或无关内容而损害性能。我们提出了StitchRAG，一个用于在RAG中构建紧凑上下文的框架。StitchRAG在动态压缩检索段落的同时，构建忠实且信息密集的上下文。我们通过效用最大化的方式权衡保留信息与计算成本，为每个段落自适应地确定压缩比例。随后，我们从段落中提取答案候选和与查询相关的关键词作为锚点，并在所得的尺寸约束下收集包含锚点的段落句子。最后，我们利用文本扩散技术将收集到的句子连接或拼接起来，形成连贯且自然的上下文。在五个开放域问答基准上的实验表明……

    arXiv:2611.22766v1 Announce Type: cross  Abstract: Retrieval-augmented generation (RAG) improves factuality by grounding large language models (LLMs) in external evidence. However, long retrieved contexts incur computational overhead and can even hurt performance due to noisy or irrelevant content. We propose \textsc{StitchRAG}, a framework for constructing compact context in RAG. \textsc{StitchRAG} builds a faithful, information-dense context while dynamically compressing the retrieved passages. We adapt the size reduction ratio to each passage by formulating a utility maximization trading-off the retained information and computation costs. We then extract answer candidates and query-related keywords from passages as anchors and collect passage sentences containing anchors under the derived size constraint. Finally, we connect or stitch the collected sentences together using text diffusion to form a coherent and natural context. Experiments on five open-domain QA benchmarks show that 
    
[^24]: 面向低资源语言的食谱跨文化适应：豪萨语、约鲁巴语和伊博语的基准数据集与多语言烹饪词汇表

    Cross-cultural Adaptation of Recipes for Low-Resource Languages: A Benchmark and Multilingual Culinary Lexicon for Hausa, Yor\`ub\'a, and Igbo

    [https://arxiv.org/abs/2611.22708](https://arxiv.org/abs/2611.22708)

    该论文提出了包含600份专家翻译与文化适应食谱的ReciLoReL基准数据集，以及涵盖约15,000个翻译实体的多语言烹饪词汇表ReciLex，用于评估和提升大语言模型在豪萨语、约鲁巴语和伊博语等低资源语言食谱翻译中的文化适应能力。

    

    近年来的多语言大语言模型（LLMs）提升了机器翻译的水平，但针对低资源语言（LRLs）的文化适应性翻译仍未得到充分探索。这对于食谱翻译而言尤其具有挑战性，因为食材、烹饪工具、计量单位和烹饪步骤可能需要根据目标文化背景进行适应调整。在这项工作中，我们提出了ReciLoReL——一个包含600份经专家翻译并完成文化适应的食谱基准数据集，涵盖约鲁巴语、豪萨语和伊博语（尼日利亚最主要的三种语言）；同时提出了ReciLex——一个多语言烹饪词汇表，包含约15,000个翻译实体，覆盖十个食谱实体类别，包括食材、工具、厨师动作、工具状态和时长等。我们在零样本和少样本设置下评估了最先进的多语言LLMs在英译低资源语言食谱翻译中的表现，并研究了利用ReciLex增强LLMs是否能改善文化适应性食谱翻译的效果。

    arXiv:2611.22708v1 Announce Type: cross  Abstract: Recent multilingual large language models (LLMs) have improved machine translation, yet culturally adaptive translation for low-resource languages (LRLs) remains underexplored. This is particularly challenging for recipes, where ingredients, cooking tools, measurements, and procedures may require adaptation to the target cultural context. In this work, we introduce ReciLoReL, a benchmark of 600 expert-translated and culturally adapted recipes across Yor\`ub\'a, Hausa, and Igbo (three of the most prominent languages in Nigeria), and ReciLex, a multilingual culinary lexicon containing approximately 15,000 translated entities across ten recipe entity categories, including ingredients, tools, chef actions, state of tools, and durations. We evaluate state-of-the-art multilingual LLMs for English-to-LRL recipe translation under zero-shot and few-shot settings and investigate whether augmenting LLMs with ReciLex improves culturally adaptive r
    
[^25]: 港口拥堵与碳减排下的航线优化：基于NSGA-II的方法

    Route Optimization under Port Congestion and Carbon Reduction: An NSGA-II Approach

    [https://arxiv.org/abs/2611.22705](https://arxiv.org/abs/2611.22705)

    本文提出一个融合变分不等式网络均衡模型与NSGA-II算法的多目标航线优化框架，联合最小化时间、运输与碳排放成本，并以中国沿海集装箱网络（上海为主要拥堵港、宁波-舟山为替代港）实证验证了港口拥堵与碳减排双重约束下的船舶航线分配方案。

    

    海运业同时面临港口拥堵日益加剧和脱碳要求日趋严格的双重挑战。大量船舶在枢纽港口锚泊不仅推高了单位运输成本，还因辅助发动机怠速运转产生了大量额外的二氧化碳排放。本研究开发了一个多目标航线优化框架，在拥堵相关的广义成本函数下，联合考虑时间、运输和碳排放三类成本。研究构建了变分不等式（VI）模型来刻画船舶航线分配中的网络均衡，并采用非支配排序遗传算法II（NSGA-II）求解由此形成的三目标优化问题。该模型以中国沿海集装箱网络为实证进行了验证（上海作为主要拥堵港口，宁波-舟山作为备选替代港口），并开展了相应的敏感性分析。

    arXiv:2611.22705v1 Announce Type: new  Abstract: Maritime shipping simultaneously faces the dual challenges of escalating port congestion and increasingly stringent decarbonization mandates. Large-scale vessel anchorage at hub ports inflates unit transport costs while generating significant additional \(CO_2\) emissions from idling auxiliary engines. This study develops a multi-objective routing optimization framework that jointly incorporates time, transport, and carbon emission costs under congestion-dependent generalized cost functions. A variational inequality (VI) model is formulated to characterize network equilibrium in vessel route allocation, while the Non-dominated Sorting Genetic Algorithm II (NSGA-II) is deployed to solve the resulting three-objective optimization problem. The model is empirically validated using China's coastal container network (Shanghai as the primary congested port, Ningbo-Zhoushan as the backup alternative), with sensitivity analyses conducted across c
    
[^26]: 用于文本分类的词嵌入可见性图表示

    Visibility Graph Representations of Word Embeddings for Text Classification

    [https://arxiv.org/abs/2611.22685](https://arxiv.org/abs/2611.22685)

    本文提出将平均词嵌入生成的文档向量转化为自然可见性图，并以节点度向量或邻接矩阵形式输入分类器，在AG News数据集上证明在词汇预算受限条件下，从选定700个词开始节点度表示成为最准确的分类配置。

    

    文本分类器依赖于能够以学习模型可用的形式揭示类别之间差异的表示，而受限的词汇量会改变哪种表示最为有效。本文研究了自然可见性图作为一种中间表示方法，用于在受控词汇预算下的新闻主题分类。词嵌入被平均为固定长度的文档向量，该向量被读作有序的数值序列，其自然可见性图以有序节点度向量或邻接矩阵的形式传递给分类器。在四分类的AG News基准数据集上，卡方选择保留100至1500个词，并将图输入与来自连续词袋（CBOW）、skip-gram、GloVe、FastText和BERT的直接基于嵌入的表示进行比较。在使用全连接分类器的情况下，从选定700个词开始，节点度表示是最准确的配置；

    arXiv:2611.22685v1 Announce Type: new  Abstract: Text classifiers depend on representations that expose the distinctions between classes in a form the learning model can use, and a restricted vocabulary changes which representation is most effective. This paper studies natural visibility graphs as an intermediate representation for news topic classification under controlled vocabulary budgets. Word embeddings are averaged into a fixed-length document vector, the vector is read as an ordered numerical series, and its natural visibility graph is passed to a classifier either as an ordered node-degree vector or as an adjacency matrix. On the four-class AG News benchmark, chi-square selection retains 100 to 1,500 words, and the graph inputs are compared with direct embedding-based representations from continuous bag-of-words, skip-gram, GloVe, FastText and BERT. With a fully connected classifier, the degree representation is the most accurate configuration from 700 selected words onward; i
    
[^27]: 自主研究中的轮间决策：一个读取完整轨迹的LLM控制器

    The Between-Round Decision in Autonomous Research: An LLM Controller That Reads the Whole Trajectory

    [https://arxiv.org/abs/2611.22661](https://arxiv.org/abs/2611.22661)

    该论文提出BRED，一个基于LLM的轮间决策控制器，通过维护最佳已验证解决方案（Champion）和过往轮次日志，让LLM直接读取完整轨迹证据来决定是改进现有方案还是开辟新方向，从而摆脱了规则、评分和搜索树等外部机制。

    

    执行一个想法往往会改变它看起来有多有前景，因此自主研究的下一步应该基于结果而非提案。在每轮实验之后，系统必须决定是继续改进最佳解决方案还是开启新方向、以哪个解决方案为基础进行构建、以及设定什么目标。我们将此称为“轮间决策”。现有系统用规则、评分、搜索树和调度等机制来处理这一决策；而在由LLM做出该决策的场合，LLM需要读取哪些内容仍是一个悬而未决的问题。我们探究LLM能否在没有周围搜索结构辅助的情况下，仅凭积累的证据做出这一决策，以及它需要读取多少证据。我们提出了BRED（Between-Round Evidence-based Decision，基于证据的轮间决策），这是一个基于LLM的控制器，它维护迄今为止最佳的已验证解决方案（称为Champion）以及过往轮次的日志，并在每轮做出一次决策。在三个广泛使用的研究智能体基准任务上，BRED优于任务基线（原文摘要在此处截断）。

    arXiv:2611.22661v1 Announce Type: new  Abstract: Executing an idea often changes how promising it looks, so the next step in autonomous research should rest on results, not on proposals. After each round of experiments, the system must decide whether to refine the best solution or start a new direction, which solution to build from, and what goal to set. We call this the between-round decision. Existing systems wrap it in machinery such as rules, scores, search trees, and schedules; where an LLM makes it, what the LLM needs to read is left open. We ask whether an LLM can make this decision from accumulated evidence, without a search structure around it, and how much of that evidence it needs to read. We present BRED (Between-Round Evidence-based Decision), an LLM-based controller that maintains the best verified solution so far (the Champion) and a log of past rounds, and makes one decision per round. On three widely used research-agent benchmark tasks, BRED improves on the task baseli
    
[^28]: 通过全模态模型中的表示子空间理解跨模态不一致性

    Understanding Cross-Modal Inconsistency through Representation Subspaces in Omni Models

    [https://arxiv.org/abs/2611.22619](https://arxiv.org/abs/2611.22619)

    本文提出从答案位置表示中估计低维“答案子空间”，发现预测全模态模型跨模态不一致性的信息主要集中于答案子空间之外的一个可复现二维表示中，且该表示与答案子空间在干预下表现出截然不同的行为作用。

    

    全模态模型（Omni Models）在统一的语言主干中处理音频、视觉和文本，但对于语义等价的跨模态输入，它们常常给出不同的答案。现有的表示层面分析没有区分“预测是否会发生分歧的信息”与“其扰动会影响模型答案的表示”。在本文中，我们从答案位置的表示中估计出一个低维的“答案子空间”，并利用它将答案相关成分与表示中的其余部分分离开来。在三个全模态模型和两个多模态推理任务上，预测跨模态不一致性的信息主要位于答案子空间之外，集中在一个可复现的二维表示中，且该表示无法由固定的模态偏移所解释。匹配干预实验揭示了二者不同的行为角色：扰动答案子空间会显著改变模型的预测，而……

    arXiv:2611.22619v1 Announce Type: new  Abstract: Omni Models process audio, vision, and text within a unified language backbone, yet often produce different answers for semantically equivalent inputs across modalities. Existing representation-level analyses do not distinguish between information that predicts whether disagreement will occur and the representation whose perturbation affects the model's answer. In this paper, we estimate a low-dimensional Answer Subspace from answer-position representations and use it to separate answer-related components from the remaining representation. Across three Omni Models and two multimodal reasoning tasks, information predictive of cross-modal inconsistency lies mainly outside the Answer Subspace, concentrating in a reproducible two-dimensional representation that is not explained by a fixed modality offset. Matched interventions reveal distinct behavioral roles: perturbing the Answer Subspace substantially changes the model's predictions, wher
    
[^29]: 火中进化：利用自适应注入压力源演化智能体安全防护机制

    Evolution Under Fire: Evolving Agent Safety Harnesses with Adaptive Injection Stressors

    [https://arxiv.org/abs/2611.22583](https://arxiv.org/abs/2611.22583)

    该论文提出分阶段框架 Evolution Under Fire，通过让安全防护机制与自适应注入攻击压力源相互对抗演化，自动生成可执行的安全防护，在 AgentDyn 基准上将攻击成功率从 50.4% 降至 3.0% 且不牺牲任务效用。

    

    使用工具的语言模型智能体必须基于来自不可信电子邮件、网页和文档的信息采取行动。这带来了一个核心安全挑战：恶意指令与合法任务指令可能通过同一交互渠道到达。现有防御大多为手工设计且静态不变，使得进一步的改进依赖于反复的人工重新设计。我们提出了 Evolution Under Fire（火中进化），这是一个分阶段框架，将安全防护机制的演化与自适应压力源的演化相耦合。在保持任务模型、基准、工具和评估器固定不变的前提下，采用效用约束、按帕累托前沿调度的树搜索，从执行证据中演化出完整可执行的安全防护机制。随后，自适应压力源演化会发现保持效用的攻击，用以指导后续阶段的防护机制演化。在基于 GPT-4o-mini 的 AgentDyn 上，演化出的防护机制将攻击成功率从 50.4% 降至 3.0%，并在干净和受攻击场景下均提升了效用。

    arXiv:2611.22583v1 Announce Type: cross  Abstract: Tool-using language-model agents must act on information from untrusted emails, webpages, and documents. This creates a central security challenge: malicious instructions and legitimate task directives can arrive through the same interaction channel. Existing defenses are largely hand-designed and static, making further improvements dependent on repeated manual redesign. We introduce Evolution Under Fire, a staged framework that couples safety-harness evolution with adaptive stressor evolution. Utility-constrained, Pareto-scheduled tree search evolves complete executable safety harnesses from execution evidence while keeping the task model, benchmark, tools, and evaluator fixed. Adaptive stressor evolution then discovers utility-preserving attacks that guide a subsequent stage of harness evolution. On AgentDyn with GPT-4o-mini, an evolved harness reduces attack success from 50.4% to 3.0%, improves clean and attacked utility over the in
    
[^30]: AfriNORM：面向非洲文化的LLM规范性推理评估基准

    AfriNORM: An Evaluation Benchmark for Normative Reasoning of LLMs Across African Cultures

    [https://arxiv.org/abs/2611.22580](https://arxiv.org/abs/2611.22580)

    本文提出AfriNORM基准——一个基于非洲社会文化场景的选择题数据集，并配合通用常识偏见得分（UCBS）和干扰项偏见得分（DBS）两个新指标，用于测量和诊断大语言模型在非洲文化规范性推理方面的局限。

    

    大语言模型（LLM）正日益被部署到各个领域。然而，当前的评估基准不足以评估这些模型在非洲本地情境中的适用性，因为大多数基准关注的是语言流畅性（语法/翻译）而非文化能力。现有少数衡量常识知识或社会惯例的基准也无法捕捉非洲的义务性模态，即规范各种社会互动的严格规范和禁忌。在本文中，我们提出了AfriNORM，这是一个基于场景的选择题数据集，旨在测量和诊断LLM在应用于非洲社会文化情境时的规范性推理局限。我们引入了两个指标：通用常识偏见得分（UCBS），用于量化LLM中的常识偏见；干扰项偏见得分（DBS），用于量化LLM对非洲文化规范理解的不足。我们还对……进行了案例研究（原文摘要在此处截断）。

    arXiv:2611.22580v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly deployed across various sectors. However, current evaluation benchmarks are insufficient to assess the applicability of these models in local African contexts, as most focus on linguistic fluency (grammar/translation) rather than cultural competence. The few existing benchmarks that measure general knowledge or social conventions do not capture the African deontic modality, i.e., the strict norms and taboos that govern diverse social interactions. In this paper, we introduce AfriNORM, a scenario-based, multiple-choice question dataset designed to measure and diagnose the normative reasoning limitations of LLMs when applied to African sociocultural contexts. We introduce two metrics: a Universal Commonsense Bias Score (UCBS) to quantify commonsense bias in LLM and a Distractor Bias Score (DBS) to quantify an LLM's limited understanding of African cultural norms. We also conduct a case study on
    
[^31]: 从感知到模拟：揭示并赋能多模态大语言模型中的心智空间动态

    From Perception to Simulation: Unveiling and Empowering Mental Spatial Dynamics in MLLMs

    [https://arxiv.org/abs/2611.22541](https://arxiv.org/abs/2611.22541)

    该论文提出MentalGrid基准以评估多模态大语言模型在无中间视觉反馈下模拟动态空间状态的能力，并通过三阶段课程学习框架训练出SpatialWalker模型，从而显著弥补了现有模型在空间模拟上的感知与多步推理缺陷。

    

    空间推理是多模态大语言模型（MLLM）理解世界的基本组成部分。尽管现有研究已推进了静态场景中空间关系的感知，但动态空间模拟这一领域仍相对未被探索。为弥合这一空白，我们提出了MentalGrid，一个专注于空间推理任务的基准，这些任务要求模型在没有中间视觉反馈的情况下模拟不断演变的空间状态。在我们基准上的广泛评估揭示了现有模型空间模拟能力的显著缺陷。具体而言，我们发现在简单模拟场景中性能受限于感知局限，而随着复杂性增加，性能则受制于多步推理能力的不足。为应对这些挑战，我们提出了一个利用16k专门空间推理样本的三阶段课程学习框架。该方法产出了SpatialWalker，

    arXiv:2611.22541v1 Announce Type: cross  Abstract: Spatial reasoning is a fundamental component of world understanding for Multimodal Large Language Models (MLLMs). While existing research has advanced the perception of spatial relations in static scenes, the domain of dynamic spatial simulation remains relatively unexplored. To bridge this gap, we introduce MentalGrid, a benchmark focused on spatial reasoning tasks that require models to simulate evolving spatial states without intermediate visual feedback. Extensive evaluation on our benchmark reveals a significant deficiency in the spatial simulation capabilities of existing models. Specifically, we find that performance is constrained by perceptual limitations in simple simulation scenarios and hindered by inadequate multi-step reasoning as complexity increases. To address these challenges, we propose a three-stage curriculum learning framework utilizing 16k specialized spatial reasoning samples. This approach yields SpatialWalker,
    
[^32]: JET：面向仓库级问答中高效工具调用智能体的“刚好足够探索”调优方法

    JET: Just-Enough Exploration Tuning for Efficient Tool-Calling Agents in Repository-Level Question Answering

    [https://arxiv.org/abs/2611.22539](https://arxiv.org/abs/2611.22539)

    JET通过在信息充足的步骤截断训练轨迹并插入理由说明进行加权微调，有效缓解了LLM智能体在仓库级问答中的“过度探索”问题，使智能体在收集到足够信息后立即作答。

    

    在仓库级问答任务中，智能体使用工具对代码仓库进行探索，并在收集到足够信息后应当回答问题。然而，我们发现大语言模型（LLM）智能体即使在收集的信息已足以回答问题时，仍经常继续调用工具。我们将这种现象称为“过度探索”。此外，尽管现有方法通过对专家轨迹进行监督微调（SFT）来提升任务性能，但SFT也会将专家轨迹中的过度探索行为传播给智能体。为了缓解过度探索问题，我们提出了“刚好足够探索调优”，这是一种微调方法，训练智能体在收集的信息足够时立即作答。JET在每条训练轨迹中信息变得充足的步骤处进行截断，插入一段解释智能体为何可以在该步骤回答的理由说明，并以更高的损失权重对该理由说明进行微调。

    arXiv:2611.22539v1 Announce Type: cross  Abstract: In repository-level question answering (QA), an agent explores a code repository with tools and should answer a question once enough information is gathered. Nevertheless, we find that LLM agents often continue to invoke tools even after the gathered information is already sufficient to answer. We call this phenomenon over-exploration. Moreover, even though existing methods improve task performance through supervised fine-tuning (SFT) on expert trajectories, SFT also propagates the over-exploration in the trajectories to the agent. To mitigate over-exploration, we propose Just-Enough Exploration Tuning (JET), a fine-tuning method that trains the agent to answer once the gathered information is sufficient. JET truncates each training trajectory at the step where the gathered information becomes sufficient, inserts a rationale that explains why the agent can answer at that step, and fine-tunes the agent with a higher loss weight on the r
    
[^33]: 当智能体技能过时：演化环境中的运行时行为与选择性维护

    When Agent Skills Go Stale: Runtime Behavior and Selective Maintenance in Evolving Environments

    [https://arxiv.org/abs/2611.22525](https://arxiv.org/abs/2611.22525)

    该论文发现环境变化会导致智能体复用的技能状态过时，使性能下降超过10个百分点或执行成本增加近40%，并提出了一种无需额外模型调用的“选择性维护”方法来验证技能是否仍然有效。

    

    智能体技能利用过去的经验和当前的观察来指导相关任务上的决策与行动。这些经验被保留在技能状态中，其复用可以摊销推理时的推理开销与环境交互成本。当环境发生变化时，该状态中的部分内容可能变得过时，而其他部分仍然有用。我们在四个任务设置和两个模型上，比较了环境变化前后的技能复用效果。与不使用技能相比，先前有益的技能可能使任务性能降低超过10个百分点，或使执行成本增加近40%。任务成功率也可能保持不变，而执行节省却下降。这些效应取决于哪些环境特定的细节变得过时，以及智能体利用当前观察进行补偿的能力。我们提出了选择性维护方法，它在无需额外模型调用的情况下，重放成功执行中的检查，以验证观察到的差异是否混淆了技能的价值……

    arXiv:2611.22525v1 Announce Type: new  Abstract: Agent skills use past experience and current observations to guide decisions and actions on related tasks. This experience is retained in skill state, whose reuse can amortize inference-time reasoning and environment interaction. When environment changes, parts of this state can become outdated while others remain useful. Across four task settings and two models, we compare skill reuse before and after environment changes. Previously beneficial skills can reduce task performance by over 10 percentage points or increase execution cost by nearly 40% relative to no skill. Task success can also remain unchanged while execution savings decline. These effects depend on which environment-specific details become outdated and the agent's ability to compensate using current observations. We introduce selective maintenance, which replays checks from successful executions without additional model calls. It verifies whether observed differences confl
    
[^34]: LMLineage：面向大语言模型智能体的内容溯源与上下文治理

    LMLineage: Content Lineage and Context Governance for LLM Agents

    [https://arxiv.org/abs/2611.22511](https://arxiv.org/abs/2611.22511)

    LMLineage是一个无需模型白盒访问、无需额外推理的带外溯源系统，可将智能体输出的每句话归因到输入片段并给出校准置信度，并基于溯源元数据实现概念漂移微压缩，仅精准压缩不再被引用的输入文档同时逐字保留用户指令。

    

    对AI智能体的可解释性不应仅局限于理解模型的内部权重，还必须覆盖智能体在多源上下文之上的推理与决策过程。我们提出了LMLineage，这是一种带外（out-of-band）系统，能够将智能体会话中的每一句输出归因到其所依据的输入片段，并给出经过校准的置信度，且无需对模型的白盒访问权限，也无需任何额外的推理过程。在溯源元数据的基础上，我们构建了概念漂移微压缩（concept-drift micro compaction）机制：系统不再等到上下文长度达到上限后将所有内容进行摘要压缩，而是检测先前使用的输入文档何时不再被引用（即发生“漂移”），并仅对这部分文档进行精准压缩，同时逐字保留用户指令。我们在一个多层级的语料库上进行了评估：包括来自10个中文网页问答案例的71条黄金标注句子，以及跨三个生成器模型独立收集的90轮多轮会话……

    arXiv:2611.22511v1 Announce Type: new  Abstract: Interpretability for AI agents should not be confined to understanding the model's internal weights. It must also cover the agent's reasoning and decision-making process over multi-source context. We present LMLineage, an out-of-band system that attributes every output sentence of an agent session to its supporting input spans with calibrated confidences, without requiring white-box access to the model or any additional inference passes. On top of the lineage metadata, we build concept-drift micro compaction: instead of waiting for the context-length limit and summarizing everything, the system detects when previously used input documents have stopped being referenced ("drifted") and surgically compacts only those, preserving user instructions verbatim. We evaluate on a multi-tier corpus: 71 gold annotated sentences from 10 Chinese web-QA cases, plus 90 turns of independently collected multi-turn sessions across three generator models (g
    
[^35]: 不断增长的簇降低了门槛：基于大语言模型的增量实体解析中的画像累积效应

    Growing Clusters Lower the Bar: Profile Accretion in LLM-Based Incremental Entity Resolution

    [https://arxiv.org/abs/2611.22485](https://arxiv.org/abs/2611.22485)

    论文揭示了一种新的“画像累积”偏差：在基于LLM的增量实体解析中，模型的匹配判断不仅取决于所比较的记录，还取决于已有簇的规模——簇越大，模型接纳新记录的门槛越低，导致名称相似公司的错误匹配率显著上升而判别能力几乎不变。

    

    增量实体解析流程会询问大语言模型（LLM）某条新记录是否属于已有簇，并向其展示已关联的记录。对匹配预言机的分析通常假设答案仅取决于所比较的记录。而对于我们测试的LLM匹配器而言，答案还取决于簇的大小，且随着簇的增长，模型接纳新记录加入簇的门槛会不断降低。基于FOPPA的法国采购供应商数据，并以SIREN标识符作为真值标准，我们固定待匹配的新记录不变，仅改变所展示的已验证成员数量。与展示一条记录相比，展示十六条成员记录使六个开放权重模型在名称相似的公司上的错误匹配率提升了（\dfmrMin--\dfmrMax）个百分点，而其ROC曲线下面积（AUC）仅上升（\aucGainMin--\aucGainMax）。在Claude Sonnet 5和Haiku 5.5中，这一提升分别为\clSonD和\clHaiD个百分点。这种偏移在使用街道地址、不同提示词变体以及BatchER和ComEM格式时依然存在。

    arXiv:2611.22485v1 Announce Type: new  Abstract: Incremental entity resolution pipelines ask a large language model (LLM) whether a new record belongs to an existing cluster and show it the records already linked. Analyses of matching oracles assume that the answer depends only on the records compared. For the LLM matchers we test, it depends on cluster size, and the model lowers its bar for joining a cluster as the cluster grows. On French procurement suppliers from FOPPA, with SIREN identifiers as ground truth, we hold the new record fixed and vary only the number of verified members shown. Sixteen members instead of one raise the false match rate on similarly named companies by \dfmrMin--\dfmrMax{} percentage points in six open-weight models, while their area under the ROC curve rises by only \aucGainMin--\aucGainMax. In Claude Sonnet~5 and Haiku~5.5 the rise is \clSonD{} and \clHaiD{} points. The shift persists with street addresses, prompt variants and the BatchER and ComEM format
    
[^36]: 呈现分歧胜过解决分歧：多智能体辩论中溯源加权记忆的门控消融研究

    Surfacing Disagreement Beats Resolving It: A Gated Ablation Study of Provenance-Weighted Memory for Multi-Agent Debate

    [https://arxiv.org/abs/2611.22461](https://arxiv.org/abs/2611.22461)

    多智能体辩论的共享记忆应保留矛盾并将排序后的冲突集呈现给辩论各方，而非将其坍缩为单一获胜事实——这一做法在同等成本下将F1提升0.340，并把投毒攻击成功率从22/122降至7/122。

    

    多智能体辩论的共享记忆通常被设计用来解决不一致性：检索返回一个“获胜”的事实，分歧在任何智能体看到之前就已消失。我们质疑这种做法是否正确。在DebateGraph中，四个后端共享一套统一契约（读/写/快照），因此来自智能体记忆文献的五种机制可以在固定预算下被独立移除，每个机制都以预先声明的阈值进行门控。在39次运行中，只有一种机制幸存：保留矛盾并将排序后的冲突集传递给辩论，而不是将其坍缩为单一获胜者，可将F1提升+0.340（95%置信区间 [+0.267, +0.414]），并将122个中毒项中的攻击成功数从22降至7（McNemar精确检验 p=0.0003），且成本不变。要得出这一结果，需要修复测试框架中的一个瓶颈：冲突集仅基于检索到的边构建，而我们预先声明的上限导致四分之三的植入矛盾无法被检索，最初测得该机制的效果仅为+0.084。

    arXiv:2611.22461v1 Announce Type: cross  Abstract: Shared memory for multi-agent debate is usually built to resolve inconsistency: retrieval returns one winning fact and the disagreement disappears before any agent sees it. We ask whether that is correct. In DebateGraph, four backends share one contract (read/write/snapshot), so five mechanisms from the agent-memory literature can be removed independently under a fixed budget, each gated on a pre-declared threshold. Across 39 runs one survives: preserving contradictions and passing the ranked conflict set to the debate, rather than collapsing it to a winner, improves F1 by +0.340 (95% CI [+0.267, +0.414]) and cuts attack success from 22 to 7 of 122 poisoned items (McNemar exact p=0.0003), at equal cost. Reaching it meant fixing a bottleneck in our harness: conflict sets are built only over retrieved edges, and our pre-declared cap left three quarters of planted contradictions unretrievable, first measuring the mechanism at +0.084. The 
    
[^37]: SignAligner：面向手语对齐的双粒度片段生成

    SignAligner: Dual-Granularity Span Generation for Sign Language Alignment

    [https://arxiv.org/abs/2611.22409](https://arxiv.org/abs/2611.22409)

    提出SignAligner，一个基于提示的视觉-语言片段生成器，通过在手语词定位的字幕片段与词片段上联合训练并采用基于重叠度的奖励优化，在不使用外部手语视频资源的情况下实现了最佳的手语字幕对齐效果。

    

    大型手语语料库受益于两种尺度的时间标注：用于字幕对齐的字幕片段和用于手语词定位的词片段。手语词定位已经能够在语料库规模上实现词的时间定位，但同时在字幕片段和词片段上联合训练字幕对齐的研究仍然不足。我们提出了SignAligner，这是一个基于提示条件的视觉-语言片段生成器，它将手语视频以及字幕句子或目标词查询映射为文本形式的起止帧区间。该模型首先在手语词定位得到的字幕片段和词片段上进行联合训练，然后利用基于重叠度的奖励进行优化，同时保留辅助词样本。在英国广播公司-牛津英国手语（BOBSL）语料库的字幕测试集上，优化后的模型达到了66.68 F1@0.50（时间交并比阈值为0.5时的F1值），在不使用外部手语视频资源的对比系统中表现最佳。

    arXiv:2611.22409v1 Announce Type: cross  Abstract: Large sign-language corpora benefit from temporal labels at two scales: subtitle spans for subtitle alignment and word spans for sign spotting. Sign spottings already locate words in time at corpus scale, but training subtitle alignment jointly on subtitle and word spans remains underexplored. We present SignAligner, a prompt-conditioned visual-language span generator that maps a sign-language video and a subtitle-sentence or target-word query to a textual start/end frame interval. It is first trained jointly on subtitle spans and word spans from sign spottings, then refined with overlap-based rewards while retaining auxiliary word examples. On the subtitle test set of the British Broadcasting Corporation-Oxford British Sign Language (BOBSL) corpus, the refined model reaches 66.68 F1@0.50 (F1 with a temporal intersection-over-union threshold of one half), the best among compared systems without external sign-video resources. This overl
    
[^38]: RICoT：用于页面流分割的循环交错思维链

    RICoT: Recurrent Interleaved Chain-of-Thought for Page Stream Segmentation

    [https://arxiv.org/abs/2611.22382](https://arxiv.org/abs/2611.22382)

    该论文提出RICoT循环交错思维链提示框架，首次将页面流分割重新构想为对整个流的顺序处理过程，像读者一样在滑动窗口内逐页推理并携带已读内容的紧凑记录以控制成本，同时发布了包含205个真实金融页面流的FinancePSS-205基准数据集。

    

    大语言模型（LLM）重新激发了人们对页面流分割（Page Stream Segmentation, PSS）的兴趣，该任务是将多页文件拆分为其组成的文档。然而，已发表的LLM方法仍然将该任务框架化为对一页或少数相邻页面的边界分类，而商业文档拆分API也不公开其方法。我们重新构想了面向LLM的PSS任务，将其视为对整个页面流的顺序处理过程，并提出了RICoT（循环交错思维链，Recurrent Interleaved Chain-of-Thought），这是一个像读者一样逐页处理文档的提示框架：在滑动窗口内，它对每一页进行推理并先做出边界决策再继续往下读；在窗口之间，它携带一个已读内容的紧凑记录，从而使每次调用的成本保持有界。我们还介绍了FinancePSS-205，一个包含205个真实世界、去标识化金融页面流的基准数据集，并将公开发布。在FinancePSS-205上，RICoT匹配或超越了成对（pairwise）和窗口化（windowed）提示方法的表现（摘要在此处被截断）。

    arXiv:2611.22382v1 Announce Type: cross  Abstract: Large language models (LLMs) have renewed interest in Page Stream Segmentation (PSS), splitting a multi-page file into its constituent documents. Yet published LLM methods still frame the task as boundary classification over one or a few neighboring pages, and commercial document-splitting APIs do not disclose their methods. We reimagine PSS for LLMs as a sequential process over the whole stream and propose RICoT (Recurrent Interleaved Chain-of-Thought), a prompting framework that works through the pages as a reader would: within a sliding window it reasons about each page and commits its boundary decision before moving on, and across windows it carries forward a compact record of what it has read, keeping per-call cost bounded. We also introduce FinancePSS-205, a benchmark of 205 real-world, de-identified financial page streams, to be released publicly. On FinancePSS-205, RICoT matches or outperforms pairwise and windowed prompting wi
    
[^39]: NONSENSE：非语义字符串揭示稠密检索中的语义性鸿沟

    NONSENSE: Non-Semantic Strings Expose Semanticity Gaps in Dense Retrieval

    [https://arxiv.org/abs/2611.22353](https://arxiv.org/abs/2611.22353)

    提出NONSENSE黑盒稀疏编辑探针，通过构造禁止复制查询词的明显非语义字符串，证明其稠密相似度得分常超过检索器最佳的查询相关段落，从而揭示了稠密检索中最近邻地位与真实语义含义相分离的语义性鸿沟。

    

    稠密检索是现代搜索、问答和检索增强生成技术的核心，其中高相似度通常被视为语义相关性的证据。我们检验了这一假设能被压榨到何种程度。NONSENSE 是一种黑盒稀疏编辑探针，它构造出表面上明显非语义的字符串，同时禁止复制查询词项，仅优化查询与该字符串之间的稠密相似度。在五个 BEIR 风格的数据集和三种流行的单向量检索器上，这些字符串的得分经常超过检索器自身最佳的已判定查询相关段落（QRP），并且常常在整个语料库中排名第一。这一结果表明，被评估的稠密检索器存在语义性鸿沟：最近邻地位可以与段落式的真实含义相分离。这些发现凸显了评估稠密检索器的重要性——不仅要看它们能否检索到相关文档，还要看它们的相似度分数有多么可靠。

    arXiv:2611.22353v1 Announce Type: cross  Abstract: Dense retrieval is central to modern search, question answering, and retrieval-augmented generation, where high similarity is often treated as evidence of semantic relevance. We examine how far this assumption can be stressed. NONSENSE is a black-box sparse-edit probe that constructs visibly non-semantic strings while forbidding query-term copying, optimizing only the dense similarity between a query and the string. Across five BEIR-style datasets and three popular single-vector retrievers, these strings frequently outscore the retriever's own best judged query-relevant passage (QRP), and often rank first in the full corpus. The result indicates a semanticity gap in the evaluated dense retrievers: nearest-neighbor status can be separated from passage-like meaning. These findings highlight the importance of evaluating dense retrievers not only by whether they retrieve relevant documents, but also by how reliably their similarity scores 
    
[^40]: 厨房里的坏苹果：LLM团队如何检测、纠正并防范其失准成员

    Bad Apples in the Kitchen: How LLM Teams Detect, Correct, and Calibrate Against Their Misaligned Members

    [https://arxiv.org/abs/2611.22342](https://arxiv.org/abs/2611.22342)

    该论文在实时具身协作环境Overcooked中注入三种基于理论的失准智能体，将LLM团队鲁棒性分解为检测、纠正与校准三个连续阶段，并发现只有当同一句话中同时包含归因和情境诊断时团队才能实现恢复。

    

    当一名队友悄悄出错时，是什么在保护LLM团队？大多数多智能体LLM研究假设合作者是协作性的，并在行动前的轮次间隙检查对话。实时具身协作不允许这种停顿：对话与行动并行进行，因此一个言语保持协作但行为却发生偏离的队友，可能比团队说出发生了什么更快地造成损害。我们通过将三个基于理论的失准智能体注入定制化的四人Overcooked环境来研究这一情形，每个智能体分别对不同的联合行动层施加压力（言语-行动不匹配、角色入侵和总体进度破坏），并通过3x4x2实验设计（n=720回合）将鲁棒性分解为检测、纠正和校准三个连续的关口。几乎所有“捣乱者”最终都会被指认出来，但只有当归因与情境诊断出现在同一句话中时，团队恢复才会随之发生；直接的来源归因提示改善了……（原文摘要在此处截断）

    arXiv:2611.22342v1 Announce Type: new  Abstract: What protects an LLM team when one teammate quietly goes wrong? Most multi-agent LLM work assumes cooperative partners and inspects chat between turns before acting. Real-time embodied cooperation does not allow this pause: chat runs alongside action, so a teammate whose words stay cooperative while behavior diverges can cause damage faster than the team can name what is happening. We study this case by injecting three theory-grounded misaligned agents into a custom four-player Overcooked environment, each stressing a different joint-action layer (speech-action mismatch, role invasion, and aggregate-progress disruption), and decompose robustness into three sequential gates of detection, correction, and calibration through a 3x4x2 design (n=720 episodes). Nearly all trolls are eventually named, but recovery follows only when attribution and situational diagnosis appear in the same utterance; direct source-attribution prompting improves th
    
[^41]: VRLLM：将大语言模型转化为强大的表格分类器

    VRLLM: Turning LLMs into Powerful Tabular Classifiers

    [https://arxiv.org/abs/2611.22308](https://arxiv.org/abs/2611.22308)

    Visa Research通过在数十亿行带标签表格数据上微调Llama 3.2 1B模型（VRLLM），首次证明LLM在大规模表格分类任务上可以超越传统表格分类器以及生产级深度学习模型。

    

    大语言模型（LLM）在文本任务上已展现出强大性能，但预训练LLM是否能有效用于大规模表格分类仍不明确，因为许多先前的研究表明，LLM在表格数据分类上的表现落后于基于树的模型或专为表格数据设计的神经架构。在Visa Research，我们成功地在数十亿行带标签的表格数据上微调了预训练的Llama 3.2 1B大语言模型，使其超越了众多表格分类器，甚至超越了我们具有竞争力的生产级深度学习模型。为便于参考，我们将该模型称为Visa Research LLM（VRLLM）。在本文中，我们分享了实现这一令人振奋的成果的方法，涵盖数据准备、训练方案、设计选择的消融研究以及部署验证。我们还在同一任务上对前沿LLM进行了基准测试，并分析了不同模型家族和世代之间的预测行为差异。

    arXiv:2611.22308v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have shown strong performance on text, but whether pretrained LLMs are effective for large-scale tabular classification remains unclear, as many previous works demonstrate that LLMs' tabular data classification performance falls behind tree-based models or neural architectures designed for tabular data. At Visa Research, we managed to fine-tune a pretrained Llama 3.2 1B LLM on billions of rows of labeled tabular data and make it outperform a wide array of tabular classifiers and even our competitive production deep learning model. For ease of reference, we refer to the model as the Visa Research LLM (VRLLM). In this paper, we share how we achieved this exciting result, covering data preparation, training recipe, ablation studies of design choices and deployment validation. We also benchmark frontier LLMs on the same task and analyze their prediction behavior across model families and generations. Our interp
    
[^42]: 教语言模型利用检索反馈收窄布尔式专利检索

    Teaching Language Models to Narrow Boolean Patent Searches from Search Feedback

    [https://arxiv.org/abs/2611.22270](https://arxiv.org/abs/2611.22270)

    该论文利用日本专利局检索报告提炼专家策略，并提出行级“保留引用的收窄信用”学习方法对语言模型进行离线迭代优化，使其生成的布尔专利检索策略在保持引用召回率的同时大幅收窄候选集合，将包含引用文献的中位数集合规模从 768 篇降至 50 篇以下。

    

    专利现有技术检索人员逐行发布布尔查询，先扩大检索范围以找到相关文献，再逐步收窄，直到候选文献数量可供筛选。我们将 38,397 份日本专利局检索报告提炼为可执行的专家策略，重新执行这些策略，并通过引用召回率以及包含每条引用文献的最小文献集合来评估生成的策略。通过提示词驱动的模型能够写出有效且高召回率的策略，但会把引用文献留在规模过大、无法人工筛选的文献集合中。我们从打分的执行轨迹中以两种形式对生成器进行离线适配：一种是从筛选感知的策略分数中进行策略级学习；另一种是从“保留引用的收窄信用”中进行行级学习，即对那些将已找到的引用文献保留在更小集合中的查询行赋予更高权重。在共享的执行轨迹上，行级信用在筛选相关指标上优于策略级分数；经过迭代优化，它将包含引用文献的集合规模中位数从 768 降至 50 以下，并将更多引用文献置于少于 100 篇文献的集合中。

    arXiv:2611.22270v1 Announce Type: cross  Abstract: Patent prior-art searchers issue Boolean queries line by line, broadening to find relevant documents and narrowing until candidates can be screened. We distill 38,397 Japan Patent Office search reports into executable expert strategies, re-execute them, and evaluate generated strategies by citation recall and the smallest set containing each citation. Prompted models write valid, high-recall strategies but leave citations in sets too large to screen. We adapt a generator offline from scored rollouts in two forms: strategy-level learning from a screening-aware strategy score, and line-level learning from citation-preserving narrowing credit, which upweights the query lines keeping a found citation in a smaller set. On shared rollouts, line-level credit outperforms strategy-level scores on screening measures; iterated, it cuts the median citation-containing set from 768 to under 50 and places more citations in sets under 100 documents th
    
[^43]: OfficePrism：面向企业业务职能单域操作的智能体基准测试

    OfficePrism: Benchmarking Agents on Single-Domain Operations Across Enterprise Business Functions

    [https://arxiv.org/abs/2611.22262](https://arxiv.org/abs/2611.22262)

    该论文提出 OfficePrism 基准，通过 420 个覆盖七大企业业务职能的单域任务（结合中文工单、可执行参考见证及自动与人工筛选流程），在 124 种模型-框架-工具包配置下系统评估智能体，发现其自然语言与数据库联合通过率仅为 22.6%–76.4%。

    

    宽泛的职场能力评分可能将单一业务职能内的胜任能力与跨职能、跨系统协调的额外成本混为一谈，导致失败难以定位。OfficePrism 通过 420 个涵盖七大业务职能的单域企业任务，在统一的、由基准定义的 L1–L3 任务结构层级下评估这一前提。每个任务都在一个全新的环境中运行，该环境由一个共享的合成组织与一个特定职能领域及类型化状态访问组合而成。我们的流水线利用有据可依的记录、业务规则和工具模式，联合构建中文工单、隐藏的可执行参考见证以及预期响应，随后进行自动过滤和人工筛选。同一任务集通过完整模式工具包和搜索-分发工具包两种方式进行评估。在 124 种模型-框架-工具包配置中，每个案例的自然语言与数据库联合通过率介于 22.6% 至 76.4% 之间。

    arXiv:2611.22262v1 Announce Type: cross  Abstract: Broad workplace scores can mix competence within one business function with the extra cost of coordinating across functions and systems, making failures hard to localize. OfficePrism evaluates this prerequisite through 420 single-domain enterprise tasks spanning seven business functions under shared, benchmark-defined L1--L3 task-structure levels. Each task runs in a fresh environment combining a shared synthetic organization with one function-specific domain and yped state access. Our pipeline uses grounded records, business rules, and tool schemas to jointly construct a Chinese-language ticket, a hidden executable reference witness, and an expected response, followed by automated filtering and human curation. The same task set is evaluated through full-schema and search-and-dispatch packages. Across 124 model--harness--package configurations, the per-case joint natural-language-and-database pass rate ranges from 22.6% to 76.4%, with 
    
[^44]: 后见之明从何进入临床智能体基准测试？对模拟器与病例构建的审计

    Where Does Hindsight Enter Clinical Agent Benchmarks? An Audit of Simulators and Case Construction

    [https://arxiv.org/abs/2611.22241](https://arxiv.org/abs/2611.22241)

    该论文提出 HindAudit 审计框架，将临床智能体基准测试中的后见之明偏差分离为三条通道并逐一测量，发现模拟实验室结果不存在信息泄露，而最大偏差来自病例文本——基准风格的病历摘要将四个大语言模型的零样本诊断准确率从 25.4% 提升至 39.7%。

    

    用于大语言模型诊断智能体的基准测试是基于回顾性病历记录构建的：智能体所阅读的病例由出院小结改写而成，而未记录的检查结果则由一个往往能看到最终诊断的模拟器生成。HindAudit 将三条后见之明通道（提供给模拟器的诊断、模拟器所依据的回顾性上下文、以及面向智能体的病例文本）分离开来，并使用仅在真实数据上训练的评判器分别进行测量。基于 MIMIC-IV 中 27,489 例急诊入院数据，五项预先注册的测试未发现模拟实验室结果中存在信息泄露：向五个模拟器提供诊断的影响微乎其微，而提示词引导的守门员稀释了诊断信号，尽管探索性结果表明它们可以提高智能体的得分。最大的效应出现在病例文本中：与基于分诊记录的就诊时点重建相比，基准风格的病历摘要将四个大语言模型的零样本诊断准确率从 25.4% 提升至 39.7%。

    arXiv:2611.22241v1 Announce Type: new  Abstract: Benchmarks for LLM diagnostic agents are built from retrospective records: the case an agent reads is rewritten from discharge summaries, and unrecorded test results are generated by a simulator that often sees the final diagnosis. HindAudit separates three hindsight channels (the diagnosis given to the simulator, the retrospective context it conditions on, and the agent-facing case text) and measures each with judges trained only on real data. On 27,489 emergency admissions from MIMIC-IV, five registered tests found no leakage into simulated laboratory results: providing the diagnosis to five simulators had a negligible effect, and prompted gatekeepers diluted the diagnostic signal, although exploratory results suggest they can raise agent scores. The largest effect lay in the case text. Against a presentation-time reconstruction from triage records, benchmark-style summaries raised the zero-shot accuracy of four LLMs from 25.4% to 39.7
    
[^45]: SOTA还是运气？LLM排行榜中的赢家诅咒

    SOTA or Luck? The Winner's Curse in LLM Leaderboards

    [https://arxiv.org/abs/2611.22227](https://arxiv.org/abs/2611.22227)

    该论文揭示了LLM排行榜中因排名噪声分数而产生的“赢家诅咒”与“排名诱导偏差”会系统性高估顶级模型的性能，并提出一种m-out-of-N自助法来纠正该偏差，在14个基准测试上排名第一模型的调整幅度为0.09至3.86个百分点。

    

    公共LLM排行榜指导着模型的评估与采用，然而排名靠前的模型所报告的性能可能被系统性高估。这种“排行榜赢家诅咒”源于对有噪声的基准测试分数进行排名时，会同时奖励采样运气和真实能力。同样的选择效应可能会扭曲整个排行榜上的分数，我们将这一现象称为“排名诱导偏差”。我们开发了一种m-out-of-N自助法（bootstrap）来纠正这种偏差。该方法假设模型池固定、任务独立同分布且有界，同时允许模型间在任务内存在任意相关性。在由正间隔分隔的精确总体层级下，我们证明了该自助法能将Θ(N^{-1/2})的排名偏差降低至期望残差偏差o(N^{-1/2})。在来自5个基准家族的14个基准测试上，排名第一模型的分数调整幅度约为0.09至3.86个百分点。在我们的SWE-bench Verified快照中，并列领先的模型

    arXiv:2611.22227v1 Announce Type: cross  Abstract: Public LLM leaderboards guide model evaluation and adoption, yet the reported performance of top-ranked models can be systematically overstated. This leaderboard winner's curse arises because ranking noisy benchmark scores rewards sampling luck alongside genuine ability. The same selection effect can distort scores throughout the leaderboard, a phenomenon we call the rank-induced bias. We develop an $m$-out-of-$N$ bootstrap to correct for this bias. It assumes a fixed model pool with independent, identically distributed bounded tasks while allowing arbitrary within-task correlation among models. Under exact population tiers separated by positive gaps, we prove that the bootstrap reduces $\Theta(N^{-1/2})$ ranking bias to expected residual bias $o(N^{-1/2})$. Across 14 benchmarks from 5 benchmark families, the rank-1 adjustments range from approximately 0.09 to 3.86 percentage points. In our SWE-bench Verified snapshot, the tied leaders
    
[^46]: 细粒度日语临床命名实体识别：BERT与开放权重大型语言模型的受控比较

    Fine-Grained Japanese Clinical Named Entity Recognition: A Controlled Comparison of BERT and Open-Weight Large Language Models

    [https://arxiv.org/abs/2611.22219](https://arxiv.org/abs/2611.22219)

    在包含113个实体类别的细粒度日语临床命名实体识别任务中，微调的BERT-CRF基线模型（F1达0.836-0.841）显著优于开放权重大型语言模型（零样本最高F1仅0.337）。

    

    背景：生成式大型语言模型（LLM）为基于编码器的临床命名实体识别（NER）提供了替代方案，但其在细粒度日语抽取任务上的性能以及对生成设置的敏感性仍不确定。方法：我们在定义了113个实体类别的iCorpus语料库上，将两个经过微调的BERT-CRF基线模型与九个开放权重的大型语言模型进行了比较。评估使用了来自36个测试文档的433个句子；1,672个句子用于训练。大型语言模型通过有无JSON Schema约束的零样本提示以及三次随机种子的QLoRA微调进行评估。微调后的模型在不同的输出token限制下进行评估，包括900以及更大的模型特定限制（2,600或4,096个token）。性能通过句子级标签-表层匹配进行测量，采用宽松表层F1和精确表层F1。结果：零样本设置下最高的宽松表层F1为0.337，而BERT基线模型达到0.836-0.841。

    arXiv:2611.22219v1 Announce Type: new  Abstract: Background: Generative large language models (LLMs) offer an alternative to encoder-based clinical named entity recognition (NER), but their performance on fine-grained Japanese extraction tasks and sensitivity to generation settings remain uncertain. Methods: We compared two fine-tuned BERT-CRF baselines with nine open-weight LLMs on iCorpus, which defines 113 entity categories. The evaluation used 433 sentences from 36 test documents; 1,672 sentences were used for training. LLMs were evaluated using zero-shot prompting with and without JSON Schema constraints and three-seed QLoRA fine-tuning. Fine-tuned models were assessed under different output-token limits, including 900 and larger model-specific limits of 2,600 or 4,096 tokens. Performance was measured using sentence-level label-surface matching with relaxed and exact-surface F1. Results: The highest zero-shot relaxed F1 was 0.337, compared with 0.836-0.841 for the BERT baselines. 
    
[^47]: EGS：面向教育型大语言模型的补充性安全框架

    EGS: A Complementary Safety Framework for Educational LLMs

    [https://arxiv.org/abs/2611.22203](https://arxiv.org/abs/2611.22203)

    本文提出教育护栏系统EGS，一个轻量级、与模型无关的运行时安全框架，通过风险分级分流、情境化教学引导和输出审计，超越简单的二元拒绝模式，有效提升面向青少年的教育大语言模型的安全性。

    

    大语言模型（LLM）在个性化教育方面具有巨大潜力，但面向青少年的部署带来了有毒、偏见、误导性或发展不适宜内容的风险。我们认为，教育安全应当超越简单的二元拒绝模式：学生的提问往往模糊不清或出于好奇心驱动，需要情境化的引导而非简单的拦截。我们提出了教育护栏系统（EGS），这是一个轻量级、与模型无关的运行时框架，通过查询审核、情境感知的教学引导和响应审核来补充模型内在的对齐能力。EGS 按风险等级对输入进行分流，通过针对特定类别的教育干预将灰色地带的查询进行重定向，并在输出交付前对其进行审计。在通用基准和教育专用基准上的实验表明，EGS 能够在多种大语言模型上提升审核性能、拒绝质量以及越狱攻击的鲁棒性。

    arXiv:2611.22203v1 Announce Type: cross  Abstract: Large Language Models (LLMs) offer strong potential for personalized education, but adolescent-facing deployment raises risks of toxic, biased, misleading, or developmentally inappropriate content. We argue that educational safety should move beyond binary refusal: student queries are often ambiguous or curiosity-driven, requiring contextual guidance rather than simple blocking. We introduce the Educational Guardrail System (EGS), a lightweight, model-agnostic runtime framework that complements intrinsic alignment through query moderation, scenario-aware pedagogical guidance, and response moderation. EGS triages inputs by risk level, redirects gray-zone queries with category-specific educational interventions, and audits outputs before delivery. Experiments across general and education-specific benchmarks show that EGS improves moderation performance, refusal quality, and jailbreak robustness across diverse LLMs. These findings suggest
    
[^48]: 从未完成体悖论到摩尔式荒谬：体貌与语用歧义下的单链推理锁定与多候选言语化

    From Imperfective Paradox to Moorean Absurdity: Single-Chain Reasoning Lock-In and Multi-Candidate Verbalization Across Aspectual and Pragmatic Ambiguity

    [https://arxiv.org/abs/2611.22188](https://arxiv.org/abs/2611.22188)

    该论文通过四项预注册的形式化实验证实了大语言模型在体貌与语用歧义推理中存在“单链过早锁定”现象，并提出多候选言语化（MCV）方法，在单次调用中生成多个加权候选估计以缓解该锁定问题。

    

    自然语言的蕴含与连贯性判断往往取决于一个事件是否仅仅发生了、是否达成了终结性的完成点、是否满足了某个计划，抑或是在多种语用标准下被评估。我们通过四项预注册、密封形式的实验（在 Qwen3-8B-Q8_0 模型上进行，共 16,640 个形式单元，temperature=0.0）检验了“单链过早锁定假说”的可观察结果：16 个假设槽位中有 15 个保持验证性地位（其中 12 个在实验内 Holm 校正下于 alpha_fam=0.05 水平拒绝 H0；E3_H3 降级为描述性地位）。在实验 1 中，Direct-CoT 与 MCU 在终结性高置信度完成上相差 25.00 个百分点（Holm p=1.31239e-05），而 NLI-CoT 在无中断的非终结性项目上崩溃了 -91.25 个百分点（Holm p=1.81805e-21）。多候选言语化（MCV）在单次调用中引出 K=5 个加权估计，将提示与未加权（MCU）和自加权（MCW）读出方式分离。普通话（实验 2）与英语（实验 3）……（原文摘要至此截断）

    arXiv:2611.22188v1 Announce Type: cross  Abstract: Natural language entailment and coherence judgments often depend on whether an event merely occurred, reached a telic culmination, satisfied a plan, or was evaluated under multiple pragmatic standards. We test observable consequences of the single-chain premature lock-in hypothesis across four preregistered, sealed formal experiments on Qwen3-8B-Q8_0 (16,640 formal units, temperature=0.0): 15 of 16 hypothesis slots retain confirmatory status (12 rejecting H0 under within-experiment Holm control at alpha_fam=0.05; E3_H3 demoted to descriptive status). In Experiment 1, Direct-CoT vs. MCU differs by 25.00 pp on telic high-confidence completion (Holm p=1.31239e-05), while NLI-CoT collapses by -91.25 pp on uninterrupted atelic items (Holm p=1.81805e-21). Multi-Candidate Verbalization (MCV) elicits K=5 weighted estimates in one call, separating the prompt from unweighted (MCU) and self-weighted (MCW) readouts. Mandarin (E2) and English (E3) 
    
[^49]: 4位量化对聊天模型的安全行为而言是否不止是噪声？利用匹配扰动归因行为变化

    Is 4-Bit Quantization More Than Noise to a Chat Model's Safety Behaviour? Attributing Behavioural Shifts with Matched Perturbations

    [https://arxiv.org/abs/2611.22186](https://arxiv.org/abs/2611.22186)

    该研究通过预注册实验和匹配扰动归因方法证明，4位量化对聊天模型安全行为的影响并非随机噪声，而是由特定舍入误差驱动的真实且近似线性的行为转变，且在不同模型上效应方向相反。

    

    手机和笔记本电脑上运行的聊天模型通常是经过评估的原始模型的4位量化副本，而针对这些副本的审计在量化如何影响拒答行为这一问题上结论不一。4位量化版本只是同等规模的众多权重扰动中的一种，因此其行为变化既可能源于其特定的舍入误差，也可能来自任何同等大小的扰动。在一项针对三个3-4B聊天模型的预注册研究中，我们采用配对测试、盲评人工标注以及8位零假设对照，发现了两个方向相反的4位量化转变：在预填合规回复的情况下，Qwen2.5-3B更常继续生成有害内容；而在留出的越狱测试中，Llama-3.2-3B的顺从率降低10.3个百分点。匹配扰动实验将这两种效应区分开来：打乱顺序和高斯版本的舍入误差平均效应接近于零，而将真实舍入误差从其镜像版本缩放至两倍大小时，行为变化呈近似线性关系。在真实误差一半的规模下，Llama的似然几乎不变，但它已经（摘要在此处截断）。

    arXiv:2611.22186v1 Announce Type: cross  Abstract: Chat models on phones and laptops are usually 4-bit copies of the model that was evaluated, and audits of such copies disagree about what quantization does to refusal. A 4-bit build is one of many weight perturbations of the same size, so a change in its behaviour could come from its specific rounding error or from any perturbation that large. In a pre-registered study of three 3-4B chat models, with paired tests, blind human labels and an 8-bit null, we find two 4-bit shifts of opposite sign: with a compliant reply prefilled, Qwen2.5-3B continues into harmful content more often, and on held-out jailbreaks Llama-3.2-3B complies 10.3 points less often. Matched perturbations separate the two. Shuffled and Gaussian versions of the error average close to zero, while scaling the real error, from its mirror image to twice its size, changes behaviour near-linearly. At half the real error, Llama's likelihood is almost unchanged, yet it already
    
[^50]: 专家构建任务上的强化学习教会了 Nemotron 3.5 Super 什么

    What RL on Expert-Built Tasks Teaches Nemotron 3.5 Super

    [https://arxiv.org/abs/2611.22161](https://arxiv.org/abs/2611.22161)

    通过在五个专家构建的任务数据集上分别进行无监督阶段的强化学习后训练，NVIDIA Nemotron 3.5 Super 在十二项基准测试上取得了最高 +13.8 分的显著性能提升。

    

    我们对 NVIDIA 的 Nemotron 3.5 Super（总参数量 120B，激活参数 12B）进行强化学习（RL）后训练，使用五个专家构建的任务数据集，每次训练一个：真实世界软件工程、复杂约束、长程工具使用、专业图表理解和企业文档推理。每次运行都在 Tinker 上使用 CISPO 训练一个全新的 rank-32 LoRA 适配器，不设监督学习阶段，并在相同的任务、测试框架和设置下将每个适配器与基础模型进行比较。在十二个主要基准测试上（其中八个由其他团队构建，四个由 Surge AI 构建），提升幅度从 IFBench 上的 +4.5 分到 GDP_pdf 上的 +13.8 分不等（后者从 10.4% 提升至 24.2%）；SWE-bench Pro V2 从 86.9% 提升至 92.1%，Terminal-Bench 2.1 从 63.9% 提升至 70.3%，MultiChallenge 从 58.3% 提升至 66.6%，AutomationBench 从 14.8% 提升至 27.5%，Chartography 从 39.4% 提升至 50.6%。文档适配器在没有 API 工具的情况下训练，也在 AutomationBench 上获得了 +8.0 分的提升。

    arXiv:2611.22161v1 Announce Type: cross  Abstract: We post-train NVIDIA's Nemotron 3.5 Super (120B total, 12B active parameters) with reinforcement learning (RL) on five expert-built task datasets, one at a time: real-world software engineering, complex constraints, long-horizon tool use, professional chart understanding and enterprise document reasoning. Each run trains a fresh rank-32 LoRA adapter with CISPO on Tinker, with no supervised stage, and each adapter is compared with the base model on the same tasks, harness and settings. On twelve headline benchmarks, eight built by other groups and four by Surge AI, the gains range from +4.5 points on IFBench to +13.8 on GDP_pdf, which rises from 10.4% to 24.2%; SWE-bench Pro V2 rises from 86.9% to 92.1%, Terminal-Bench 2.1 from 63.9% to 70.3%, MultiChallenge from 58.3% to 66.6%, AutomationBench from 14.8% to 27.5% and Chartography from 39.4% to 50.6%. The document adapter, trained without API tools, also gains +8.0 points on AutomationB
    
[^51]: 长上下文生成是一个采样问题（当你调高温度时）

    Long-Context Generation Is a Sampling Problem (When You Turn Up the Heat)

    [https://arxiv.org/abs/2611.22148](https://arxiv.org/abs/2611.22148)

    该论文通过大规模实验证明，长上下文生成的退化本质上是采样器问题——高温下保持候选 token 池小而稳定的采样器（如 top-nsigma）能让编程智能体等长程生成任务保持高成功率，而放任低概率 token 进入的采样器则完全崩溃。

    

    开源权重语言模型被普遍观察到生成得越长表现越差，而常见的修复尝试往往依赖更多的训练。然而在高温下，采样器——即从模型概率中挑选下一个 token 的规则——决定了长程与智能体式生成能否存活乃至蓬勃发展。我们在十个成熟基准测试以及最长一百万 token 的大海捞针检索任务上，测试了从 1B 到 122B 参数的 29 个开源权重模型，覆盖 SWE-bench Verified 编程智能体、长思维链推理、长篇写作、长结构化输出和长上下文阅读。结果表明：随着温度升高，保持候选 token 池小而稳定的采样器不会损失准确率；而允许低概率 token 进入的采样器则会彻底崩溃。在温度为 2 时，运行 top-p、top-k 或 Top-H 的编程智能体在 100 个 GitHub 问题中解决 0 个，而运行 top-nsigma 的相同智能体可解决 46 到 62 个，并赢得了两种采样器之间的每一次对比。

    arXiv:2611.22148v1 Announce Type: new  Abstract: Open-weight language models are widely observed to get worse the longer they generate. Often, attempts to fix this involve more training. At high temperature, The sampler, i.e. the rule that picks the next token out of the model's probabilities, decides whether long and agentic generation survives and/or thrives. We test 29 open-weight models from 1B to 122B parameters on ten established benchmarks plus needle retrieval out to one million tokens, covering SWE-bench Verified coding agents, long chain-of-thought reasoning, long-form writing, long structured output and long-context reading. Samplers that keep their pool of candidate tokens small and steady as temperature rises don't lose accuracy. Samplers that let in low-probability tokens fall apart. At temperature 2, coding agents running top-p, top-k or Top-H resolve 0 of 100 GitHub issues, while the same agents running top-nsigma resolve 46 to 62 and win every issue the two samplers di
    
[^52]: 权重空间编程不需要Transformer

    Weight-Space Programming Doesn't Need Transformers

    [https://arxiv.org/abs/2611.22055](https://arxiv.org/abs/2611.22055)

    在无需梯度训练的权重空间编程中，三个前沿LLM在自由设计架构时都不约而同地将知识编译为可查询的稀疏表，且其性能以约1/1800的参数量超越了Transformer，表明权重空间编程并不需要Transformer。

    

    诸如Transformer之类的标准神经架构之所以被选用，是因为它们能通过梯度下降很好地训练，并且在GPU上运行速度快。而在权重空间编程中，大型语言模型（LLM）无需基于梯度的训练即可直接编写目标权重，因此这些理由不再适用。我们还应该继续使用Transformer吗？在最少指导、十个标注示例和一个沙盒环境的条件下（任务主要为多项选择题），我们让三个前沿LLM要么为固定架构生成权重，要么自行设计架构。当可以自由设计时，这三个LLM都将纯文本事实编译成一个由固定程序查询的稀疏表，测试准确率最高达到54.4%。LLM设计的模型比同一LLM所编写的最准确的已发布Transformer取得了更高的测试准确率，而参数量仅约为后者的1/1800。在没有梯度的情况下，它们选择了一种表结构，其中LLM可以说明每个权重的含义……

    arXiv:2611.22055v1 Announce Type: cross  Abstract: Standard neural architectures such as the Transformer were chosen because they train well with gradient descent and run fast on GPUs. In weight-space programming, a large language model (LLM) writes target weights by hand without gradient-based training, so those reasons no longer apply. Should we keep the Transformer? Given minimal guidance, ten labeled examples, and a sandbox, mainly on multiple-choice questions, we task three frontier LLMs with either generating the weights of a fixed architecture or designing the architecture themselves. When free to design, all three compile plain-text facts into a sparse table queried by a fixed procedure, reaching up to 54.4% test accuracy. An LLM-designed model achieves higher test accuracy than the most accurate released Transformer that the same LLM wrote while using about 1,800x fewer parameters. Without gradients, they choose a table, in which the LLM can state what each weight means and ca
    
[^53]: RLDE：一个用于可解释迭代文档编辑的强化学习框架

    RLDE: A Reinforcement Learning Framework for Interpretable Iterative Document Editing

    [https://arxiv.org/abs/2611.22039](https://arxiv.org/abs/2611.22039)

    提出了RLDE框架，将文档迭代修订建模为无参考的MDP问题，通过指针网络与PPO智能体结合冻结的BART模型，依据可读性、困惑度和语法等多目标奖励实现可解释的自动化文档编辑。

    

    提升长文档的质量、清晰度和流畅性仍然是自然语言处理（NLP）中一个具有挑战性的问题，因为人工编辑通常是通过迭代式润色而非一次性重写完成的。现有的大多数自动化系统要么充当黑盒生成器，要么依赖监督学习，这限制了可解释性，并且无法捕捉修订过程的序列特性。本文提出了RLDE，一个自动化的端到端、指标驱动的强化学习框架，它将文档修订形式化为一个无参考的马尔可夫决策过程（MDP）问题（相对于成对的文档-修订目标而言是无监督的）。在有限的动作空间（NOOP、INSERT、DELETE、REPLACE）下，RLDE利用指针网络在词元级别进行动态片段选择，并将生成式动作发送给冻结的BART-base模型以获得上下文感知的修改建议。PPO智能体学习优化基于可读性、困惑度和语法指标的多目标奖励函数……

    arXiv:2611.22039v1 Announce Type: cross  Abstract: Improving the quality, clarity, and flow of long documents remains a challenging problem in NLP, as human editing typically occurs through iterative refinement rather than a single-pass rewrite. Most existing automated systems act as black-box generators or rely on supervised learning, limiting interpretability and failing to capture the sequential nature of revisions. In this paper, we propose RLDE, an automated end-to-end metric-driven RL framework that formulates document revision as a reference-free MDP problem (unsupervised with respect to paired document-revision targets). Working with a limited action space (NOOP, INSERT, DELETE, REPLACE), RLDE leverages Pointer Networks for dynamic span selection at the token level and sends generative actions to the frozen BART-base model to obtain context-aware suggestions. PPO agents learn to optimize a multi-objective reward function based on readability, perplexity, and grammar metrics wit
    
[^54]: 更努力思考并不能带来更好的选择：推理努力提升了LLM购物代理的置信度，而非正确性

    Thinking Harder Doesn't Lead to Better Choices: Reasoning Effort Increases the Confidence but Not the Correctness of LLM Shopping Agents

    [https://arxiv.org/abs/2611.22033](https://arxiv.org/abs/2611.22033)

    在三个购物基准测试中，增加LLM购物代理的推理努力并不能显著提升其选购产品的正确性，却会显著提高其报告的置信度——也就是说，模型“想得更多”只是变得更自信，而不是变得更正确。

    

    大型语言模型（LLM）越来越多地被部署为自主购物代理，它们处理多模态商品信息（包括视觉和文本数据）并代表消费者推荐购买。人们自然会认为，思考能力更强的模型会做出更高质量的购买决策。为了验证这一观点，我们针对此类代理提出了两个问题：要求模型更努力地思考，是否会让它选出更好的产品？它所报告的置信度，能否告诉我们它出错的概率？我们在基于WebShop、亚马逊SQID和Wayfair的WANDS平台构建的三个产品选择基准上，对十个前沿模型进行了测试，以检验思考能力更强的LLM是否真的能做出更高质量的产品购买决策。在全部三个基准上，我们发现增加思考努力并不能在统计上显著提升决策质量。而同样的思考努力变化却会提高模型所声称的置信度（原文摘要在此处截断）。

    arXiv:2611.22033v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly deployed as autonomous shopping agents that process multimodal product listings (consisting of visual and text data) and recommend purchases on behalf of consumers. Naturally, one might assume that models with more thinking power would make higher-quality purchasing decisions. To test this belief, we ask two questions about such agents: does asking a model to think harder make it choose better products, and does the confidence it reports tell us the likelihood it may be wrong? We benchmark ten frontier models on three product-choice benchmarks built from WebShop, Amazon's SQID and Wayfair's WANDS platforms to see if LLMs with more thinking power actually make better-quality product purchase decisions. Across all three benchmarks, we found that increasing the thinking effort does not increase decision quality in a statistically meaningful way. The same change in thinking effort raises stated c
    
[^55]: GauntletScore：在 LLM-AggreFact 上媲美最佳已发表检查器的可审计事实核查

    GauntletScore: Auditable Fact-Checking That Matches the Best Published Checkers on LLM-AggreFact

    [https://arxiv.org/abs/2611.22027](https://arxiv.org/abs/2611.22027)

    GauntletScore是一种可审计的多读者证据引擎，通过向不同提供商的独立读者提问、在证据账本中记录测量并使用固定贝叶斯规则汇总（仅在两位读者一致时判定矛盾），在LLM-AggreFact完整测试集上达到79.51的平衡准确率，与最佳已发表检查器持平。

    

    检测生成的句子是否得到其源文档的支持，是针对AI生成文本的一项标准保障措施，而已发表的最强检查器均为单一模型。我们介绍GauntletScore，这是一个证据引擎，其少量规则和阈值仅在基准的开发集上设定。它向来自不同提供商的独立读者提出类型化问题，并将每个答案作为一项测量记录在证据账本中。它采用固定的贝叶斯规则组合这些测量结果，只有当两个读者一致同意时才判定为矛盾。在完整的LLM-AggreFact测试集（涵盖11个数据集的29,320条声明）上，按照排行榜的方法，GauntletScore取得了79.51的平衡准确率，高于截至2026年11月15日排行榜上发布的所有平均值，使其与其中最佳的模型持平，即nimble-v4-max（其发布的数据集得分为79.45；该引擎的95%置信区间为78.47至80.53，包含了该值）。

    arXiv:2611.22027v1 Announce Type: cross  Abstract: Checking whether a generated sentence is supported by its source document is a standard safeguard for AI-written text, and the strongest published checkers are single models. We describe GauntletScore, an evidence engine whose few rules and thresholds were set on the benchmark's development split. It puts typed questions to independent readers from different providers and records each answer as a measurement in an evidence ledger. It combines the measurements with a fixed Bayesian rule that counts a contradiction only when two readers agree. On the full LLM-AggreFact test split (29,320 claims across 11 datasets), following the leaderboard's approach, GauntletScore achieves 79.51 balanced accuracy, above every average published on the leaderboard as of 15 November 2026. That places it level with the best of them, nimble-v4-max (79.45 from its published dataset scores; the engine's 95% interval, 78.47 to 80.53, contains it). On the same 
    
[^56]: SkillBuffBench：衡量智能体技能在框架、模型、技能访问模式与记忆等方面的贡献

    SkillBuffBench: Measuring What Agent Skills Contribute across Harnesses, Models, Skill Access Modes and Memory

    [https://arxiv.org/abs/2611.22017](https://arxiv.org/abs/2611.22017)

    SkillBuffBench 通过将技能访问方式（无技能/精选技能/自动调用）、记忆开关、四个智能体框架和多个主流模型进行交叉配对评估，系统性地衡量了智能体技能在贴近真实部署场景下的实际贡献。

    

    智能体技能是一类基于文件的程序性知识包，编码智能体可以按需发现并加载，目前各大主流智能体框架均已支持。然而，关于技能能带来多大帮助的证据，几乎完全来自这样的设定：为任务人工挑选正确的技能，且智能体从空白状态开始。这两点在真实部署中都不成立——部署时智能体需要从已安装的技能库中自行选择，并且经常携带此前工作中积累的记忆。我们提出了 SkillBuffBench，这是一个基准测试及配套的配对评估协议，它将技能访问视为一个三级因素（无技能、精选技能、以及从已安装库中自动调用），并与记忆（开启与关闭）、四个智能体框架以及多个模型（来自 Anthropic、Google 和 OpenAI 的最先进专有模型，以及来自中国厂商的最先进开源权重模型）进行交叉组合，每个模型都在每个框架下运行……

    arXiv:2611.22017v1 Announce Type: new  Abstract: Agent Skills, file-based packages of procedural knowledge that a coding agent can discover and load on demand, are now shipped by every major agent harness, yet the evidence for how much they help comes almost entirely from settings in which the right Skill is hand-picked for the task and the agent starts from a blank slate. Neither holds in deployment, where an agent chooses from whatever library happens to be installed and frequently carries memory from earlier work. We present SkillBuffBench, a benchmark and paired evaluation protocol that treats Skill access as a three-level factor (no Skill, a curated Skill, and automatic invocation from an installed library), crosses it with memory (off and on), with four agent harnesses and with multiple models (the state-of-the-art proprietary models from Anthropic, Google and OpenAI, and the state-of-the-art open-weight models from Chinese providers), every model being run under every harness, a
    
[^57]: DiagTrace：一个用于临床决策的交互式基准测试

    DiagTrace: An Interactive Benchmark for Clinical Decision Making

    [https://arxiv.org/abs/2611.22009](https://arxiv.org/abs/2611.22009)

    DiagTrace是一个基于PubMed Central病例报告构建的交互式临床决策基准，包含510个跨八个专科的病例，用于评估大语言模型能否通过交互式信息获取从不完整信息中逐步重建诊断轨迹，突破了传统一次性提供完整患者信息的评估方式。

    

    大语言模型（LLM）在医学问答和诊断推理基准测试中展现出强大的性能。然而，现有的评估通常预先提供患者的全部信息，并且仅评估最终诊断或推理理由。这种设置忽略了真实临床诊断中需要从不完整信息出发逐步获取诊断证据的需求。为弥补这一差距，我们提出了DiagTrace，这是一个基于真实文献来源的基准测试，用于评估LLM能否通过交互式信息获取来重建已被记录的诊断轨迹。每个基准实例均基于PubMed Central病例报告构建，包含初始临床表现、事件库、临床病程和诊断参考资料。DiagTrace包含覆盖八个专科的510个病例，其中225个用于评估四个LLM在临床病程对齐度、交互质量和效率方面的表现。结果显示模型在覆盖范围与信息源一致性等方面存在权衡。

    arXiv:2611.22009v1 Announce Type: cross  Abstract: Large language models (LLMs) have shown strong performance on medical question answering and diagnostic reasoning benchmarks. However, existing evaluations typically provide all patient information upfront and assess only the final diagnosis or rationale. This setup ignores the need to acquire diagnostic evidence incrementally from incomplete information in real-world diagnosis. To bridge this gap, we introduce DiagTrace, a source-grounded benchmark for evaluating whether LLMs can reconstruct the documented diagnostic trajectory through interactive information seeking. Each benchmark instance is built from a PubMed Central case report and includes an initial presentation, an event bank, a clinical course, and diagnostic references. DiagTrace contains 510 cases across eight specialties, with 225 used to evaluate four LLMs on clinical-course alignment, interaction quality, and efficiency. Results reveal trade-offs in coverage, source-con
    
[^58]: 大语言模型作为非线性动力系统：深度、训练与生成的三个时钟

    Large Language Models as Nonlinear Dynamical Systems: The Three Clocks of Depth, Training, and Generation

    [https://arxiv.org/abs/2611.21983](https://arxiv.org/abs/2611.21983)

    本文提出“三时钟”框架，将语言模型在层深度、训练过程和生成过程三个尺度上的行为与非线性动力学的经典概念（不动点、分岔、同步、混沌等）系统对应，并通过汇总约200篇文献的“三时钟词典”和一系列降阶模型在两个学科之间架起桥梁。

    

    语言模型是动力系统。Transformer 在其层上迭代、在训练步骤中迭代、在生成的词元序列上迭代，而在这三个“时钟”的每一个上，它都表现出非线性动力学已研究一个世纪的行为：不动点、分岔、同步、混沌以及噪声驱动的势阱逃逸。由于两个学术社区很少交流，这些行为常常以新名称被重新发现。我们在两者之间架起桥梁。为了描述，我们将约200篇论文浓缩为“三时钟词典”，把十五个非线性动力学概念映射到每个时钟上，并将每条对应关系标记为已证明、已测量或推测，同时在机械系统和语言模型中并排展示每个概念。为了解释，我们推导了降阶模型：softmax 是玻尔兹曼分布，梯度下降是非线性弹簧上的欧拉步，损失尖峰则是小球被抛出刚性山谷。

    arXiv:2611.21983v1 Announce Type: cross  Abstract: Language models are dynamical systems. A transformer is iterated over its layers, over the steps of its training and over the tokens it generates, and on each of these three clocks it shows behavior that nonlinear dynamics has studied for a century: fixed points, bifurcations, synchronization, chaos and noise-driven escape from potential wells. The two communities rarely meet, so this behavior is often rediscovered under new names. We build a bridge between them. To describe, we condense about 200 papers into the Three-Clock Dictionary, which maps fifteen concepts of nonlinear dynamics onto each clock and marks every correspondence as proved, measured or conjectured, and we show each concept in a mechanical system and in a language model side by side. To explain, we derive reduced-order models: the softmax is a Boltzmann distribution, gradient descent an Euler step on a nonlinear spring, a loss spike a ball thrown out of a stiff valley
    
[^59]: 在嵌入空间中对文档与章节层次结构建模

    Modeling Document and Section Hierarchies in Embedding Space

    [https://arxiv.org/abs/2611.21980](https://arxiv.org/abs/2611.21980)

    该论文提出层次化嵌入式主题模型，通过嵌入空间中的层次先验将文档级与章节级主题分布相耦合，并借助摊销变分推断在法律、报告和论坛等多个领域的结构化语料库上取得了最低或接近最低的困惑度表现。

    

    嵌入式主题模型将词嵌入整合到主题生成过程中，能够产生连贯的主题，并在大词表上表现出稳健的性能。然而，每篇文档仅由单个主题分布表示，忽略了其内部结构。许多现实世界的语料库，包括法律文档、公司和政府报告以及在线讨论论坛，都呈现出层次化的组织结构：文档由多个章节组成，而这些章节往往涉及不同的子主题。我们提出了层次化嵌入式主题模型，这是一种结构化主题模型，通过嵌入空间中的层次先验将文档的主题分布与其各章节的主题分布相耦合。我们为 hETM 推导了一种摊销变分推断过程，并在来自上述各领域的结构化语料库上对其进行评估。实验结果表明，hETM 在扁平模型与……之中取得了最低或接近最低的留出集困惑度（摘要原文在此处截断）。

    arXiv:2611.21980v1 Announce Type: cross  Abstract: Embedded topic models integrate word embeddings into the topic generation process, producing coherent topics and robust performance on large vocabularies. However, each document is represented by a single topic distribution, ignoring its internal structure. Many real world collections, including legal documents, company and government reports, and online discussion forums, exhibit hierarchical organization in which documents consist of multiple sections that often address distinct subtopics. We introduce the Hierarchical Embedded Topic Model (hETM), a structured topic model that couples the topic distribution of a document and those of its sections through a hierarchical prior in the embedding space. We derive an amortized variational inference procedure for hETM and evaluate it on structured collections from each of these domains. Experimental results show that hETM achieves the lowest or near-lowest held-out perplexity among flat and
    
[^60]: 超越位宽：面向NVIDIA Blackwell多语言大模型推理的部署约束混合精度量化

    Beyond Bitwidth: Deployment-Constrained Mixed-Precision Quantization for Multilingual LLM Inference on NVIDIA Blackwell

    [https://arxiv.org/abs/2611.21974](https://arxiv.org/abs/2611.21974)

    该论文在NVIDIA B200上对Qwen3-8B进行了部署约束驱动的低比特与混合精度量化实测评估，发现INT2/INT3全面退化且模块恢复效应具有强非加性，最终提出一个NVFP4与INT8混合的配置，在通过全部预注册多语言与推理精度门槛的同时将权重内存压缩至5.917 GiB。

    

    arXiv:2611.21974v1 公告类型： cross 摘要：低比特大模型量化只有在模型保持与服务相关的精度、能够通过受支持的GPU内核执行、并满足实测的内存和延迟限制时才有实用价值。我们在NVIDIA B200上使用vLLM研究Qwen3-8B，涵盖INT2/INT3、GPTQ、NVFP4以及混合精度配置。我们在全部144个融合线性模块上验证了原生打包执行与精度，并在预注册的门槛条件下评估韩语、英语、数值和推理质量。打包的INT2和INT3虽然可以执行，但所有七种测试的INT2网格在生成任务上均出现退化。孤立的模块恢复效应表现出强烈的非加性；韩语与推理任务的敏感性排名互不一致，且一个看似有前景的INT4/INT8开发候选方案在独立的最终评估中失败。一个包含16位激活、98个NVFP4仅权重量化模块、46个INT8模块和INT8输出头的配置，在未做任何改动的FINAL-81测试集（5,168个条目）上通过了全部门槛，权重加载内存为5.917 GiB……

    arXiv:2611.21974v1 Announce Type: cross  Abstract: Low-bit LLM quantization is useful only if a model preserves service-relevant accuracy, executes through supported GPU kernels, and meets measured memory and latency limits. We study Qwen3-8B on NVIDIA B200 with vLLM, including INT2/INT3, GPTQ, NVFP4, and mixed-precision configurations. We verify native packed execution and precision on all 144 fused linear modules and evaluate Korean, English, numerical, and reasoning quality under preregistered gates. Packed INT2 and INT3 execute, but all seven tested INT2 grids degenerate on generation. Isolated module-restoration effects are strongly non-additive; Korean and reasoning sensitivity rankings disagree, and a promising INT4/INT8 development candidate fails independent final evaluation. A configuration with 16-bit activations, 98 NVFP4 weight-only modules, 46 INT8 modules, and an INT8 output head passes all gates on the untouched FINAL-81 set (5,168 items): 5.917 GiB of weight-loading me
    
[^61]: 轮到我了！可解释方法揭示母子对话动态中的发展差异

    It's My Turn! Interpretable Methods Reveal Developmental Differences from Mother-Child Conversational Dynamics

    [https://arxiv.org/abs/2611.21969](https://arxiv.org/abs/2611.21969)

    该研究开发了一个结合序列分析、信息论测度和贝叶斯层次建模的可解释计算框架，揭示了语言发展较低的两岁儿童与母亲的对话表现出更异质的组织结构和更少的互惠式轮流交流。

    

    理解母亲与儿童之间的对话互动是计算行为建模的一项重要挑战。现有方法采用传统统计方法和对话动态的主观测量，未纳入时间或概率建模。我们开发了一个可解释的计算框架，结合序列分析、信息论测度和贝叶斯层次建模来刻画亲子游戏中的时间性对话动态。我们将该框架应用于94对母亲与两岁儿童的双向互动数据。序列分析显示，与高发展组相比，语言发展水平较低的双向组（以贝利发育量表分数定义）表现出更加异质性的对话组织和更少的“一来一回”式交流，即一种伙伴双方轮流发起交流的互惠式轮流模式。

    arXiv:2611.21969v1 Announce Type: cross  Abstract: Understanding conversational interactions between mothers and children is an important challenge for computational behavioral modeling. Existing methods use traditional statistical approaches and subjective measures of conversational dynamics, without incorporating temporal or probabilistic modeling. We developed an interpretable computational framework that combines sequence analysis, information-theoretic measures, and Bayesian hierarchical modeling to characterize temporal conversational dynamics in parent-child play. We applied this framework to interactions from 94 mother-child dyads with two-year-old children. Sequence analysis revealed that dyads with lower language development (as defined by scores on the Bayley developmental test) exhibited more heterogeneous conversational organization and fewer serve-and-return exchanges than higher-development dyads, a reciprocal turn-taking pattern in which one partner initiates communicat
    
[^62]: 从求解器到系统：科学仿真中大语言模型跨表征与执行的推理

    Solver to System: LLM Reasoning Across Representations and Execution in Scientific Simulation

    [https://arxiv.org/abs/2611.21964](https://arxiv.org/abs/2611.21964)

    该论文提出了Solver2System数据集，通过多表示形式、受控损坏与词汇扰动来评估大语言模型在PDE科学仿真中的跨表示一致性与物理无效性检测能力。

    

    基于偏微分方程（PDE）的仿真器被广泛用于对物理系统建模，但利用大语言模型（LLM）生成和解释PDE求解器代码仍然具有挑战性。科学仿真代码是检验LLM科学推理能力的有效试验平台，原因在于：（1）同一个系统可以用代码、方程、自然语言和数值轨迹等多种形式表示；（2）代码可能成功执行，但其实现的物理却是无效的。我们提出了Solver2System，这是一个包含多种表示形式、受控损坏和词汇扰动的PDE求解器数据集，用于评估LLM能否通过（1）跨表示一致性以及（2）对物理无效性的静态检测和智能体式检测来恢复底层物理系统。在八个开源权重模型上的实验表明，当证据相互冲突时，LLM会系统性地偏好某些表示形式，且对代码施加词汇扰动会改变其对其他未变内容的解释。

    arXiv:2611.21964v1 Announce Type: cross  Abstract: Simulators based on partial differential equations (PDEs) are widely used to model physical systems, but generating and interpreting PDE solver code with large language models (LLMs) remains challenging. Scientific simulation code is a useful testbed for LLM scientific reasoning because (1) one system can be represented by code, equations, natural language, and numerical trajectories and (2) code can execute successfully while implementing invalid physics. We introduce Solver2System, a dataset of PDE solvers with multiple representations, controlled corruptions, and lexical perturbations, to evaluate whether LLMs can recover the underlying physical system through (1) cross-representation consistency and (2) static and agentic detection of physical invalidity. Across eight open-weight models, LLMs systematically favor some representations when evidence conflicts, and lexical perturbations to code alter interpretation of other unchanged 
    
[^63]: 守恒的判定，漂移的承诺：识别有害的过度思考

    Conserved Verdicts, Drifting Commitments: Identifying Harmful Overthinking

    [https://arxiv.org/abs/2611.21946](https://arxiv.org/abs/2611.21946)

    该论文提出“承诺-判定差距”框架来识别推理模型中的有害过度思考，证明仅凭最终正确性探针估计的“判定”无法预测继续思考的利弊，而“承诺”与“判定”之差恰好量化了停止思考可获得的准确率增益，并给出将准确率-预算曲线分解为链内答案变化与链存活变化的恒等式。

    

    一个已经得出正确答案的推理模型，可能会因为继续思考而失去这个正确答案。决定是否停止，意味着在推理链的同一位置比较两个概率：模型若现在停止所给出的答案是正确的概率（即其“承诺”），以及模型最终完成时的答案是正确的概率（即其“判定”）。我们证明，仅凭判定——也就是最终正确性探针所估计的量——无法完成这种比较。作为对一个固定结果的预测，判定没有可预测的漂移，无论继续思考是有益还是有害。承诺与判定之差等于答案质量的期望剩余变化量，也正是停止思考所能获得的准确率增益。第二个恒等式则将任何准确率-预算曲线分解为两部分：推理链内部的答案变化，以及仍在运行的推理链集合的变化。我们在 MATH-500 上对这两者进行了测量。在 DeepSeek-R1-Distill-Qwen-1.5B 持有正确答案的全部 379 个点上（这些点的选取不考虑其推理链如何结束），继续……（原文摘要至此截断）

    arXiv:2611.21946v1 Announce Type: new  Abstract: A reasoning model that has reached a correct answer can lose it by continuing to think. Deciding whether to stop means comparing two probabilities at the same point of a chain: that the answer the model would give if stopped now is correct (its commitment), and that the answer it will finish with is correct (its verdict). We show that the verdict alone, which is what final-correctness probes estimate, cannot make this comparison. As a forecast of a fixed outcome it has no predictable drift, whether continuing helps or hurts. The commitment-verdict gap equals the expected remaining change in answer quality and the accuracy gained by stopping. A second identity splits any accuracy-versus-budget curve into answer change within chains and change in which chains are still running. We measure both on MATH-500. At all 379 points where DeepSeek-R1-Distill-Qwen-1.5B holds a correct answer, selected without regard to how its chain ends, continuing
    
[^64]: DNeo：你的并行草稿模型需要更好的目标特征编码

    DNeo: Your Parallel Draft Model Needs Better Target Feature Encoding

    [https://arxiv.org/abs/2611.21925](https://arxiv.org/abs/2611.21925)

    本文发现标准特征编码机制存在有损（残差混合破坏特征空间）、无记忆、均匀采样忽略重要隐藏状态三大缺陷，并提出特征混合注意力（MoFA）来改进推测-扩散解码中并行草稿模型的目标特征编码质量。

    

    推测-扩散解码使用一个小型并行草稿模型在一次前向传播中生成 b 个草词元，然后通过更大的目标模型对这些草词元进行并行验证。当多个草词元同时通过验证时，便可以实现更快的推理速度。近期的并行草稿架构（DFlash、DSpark）通过一种称为特征编码的机制，以目标模型计算出的丰富特征为条件来提升草稿质量。然而，标准的特征编码机制存在三个局限性：（1）它们是有损的，其应用的编码投影会通过一种被称为“残差混合”的效应破坏底层的目标特征空间；（2）它们是无记忆的，没有对之前处理过的提议的记录；（3）它们使用从目标模型各层均匀采样的特征，忽略了重要的隐藏状态。为了解决这些问题，我们提出了特征混合注意力（MoFA），一种……

    arXiv:2611.21925v1 Announce Type: new  Abstract: Speculative-diffusion decoding uses a small parallel drafter model to generate b draft tokens in one forward pass, which then undergo parallel verification via a larger target model. Faster inference can be achieved when multiple draft tokens pass verification simultaneously. Recent parallel draft architectures (DFlash, DSpark) improve draft quality by conditioning on rich features computed by the target, through a mechanism known as feature encoding. However, standard feature encoding mechanisms have three limitations: (1) they are lossy, applying encoding projections that are destructive to the underlying target feature-space via a discovered effect known as residual mixing, (2) they are memory-less, with no record of previously processed proposals, and (3) they utilize features sampled uniformly from across the target layers, omitting important hidden states. To address these issues, we propose Mixture-of-Features Attention (MoFA), a 
    
[^65]: 下一词元训练下的潜在事实组合

    Latent Fact Composition under Next-Token Training

    [https://arxiv.org/abs/2611.21921](https://arxiv.org/abs/2611.21921)

    本文提出 Baton 系列循环 Transformer，通过在第二次循环中将中间实体作为词元嵌入传递，使逐词元训练的模型能够回答中间实体从未在两跳训练样本中出现过的两跳推理问题，并归纳出决定各模型成败的三个设计条件。

    

    对潜在两跳推理的受控研究通常训练 Transformer 只预测最终答案，而语言模型则被训练来预测每一个词元。我们探究：在每一个词元上都进行训练的模型，能否学会回答那些中间实体从未出现在任何两跳训练样本中的两跳问题。我们构建了一个包含此类问题的合成语料库，在该语料库上损失函数的每一项都可以被精确计算。标准 Transformer、循环 Transformer 以及 DiscoLoop 几乎都无法回答这些问题。因此，我们提出了 Baton 系列循环 Transformer，它们将中间实体以词元嵌入的形式传递给第二次循环。当训练时只预测答案，其三个变体中有两个几乎完美地回答了这些问题。随后我们将每个模型的成功与失败归因于三个设计条件，每次只改变一个条件的实验结果与该假设一致。当 Baton 在每一个词元上进行训练时（摘要在此处截断）

    arXiv:2611.21921v1 Announce Type: new  Abstract: Controlled studies of latent two-hop reasoning train transformers to predict only the answer, whereas language models are trained to predict every token. We ask whether a model trained on every token learns to answer two-hop questions whose intermediate entity appears in no two-hop training example. We build a synthetic corpus with such questions. On this corpus every term of the loss can be computed exactly. A looped transformer, a standard transformer and DiscoLoop answer almost none of these questions. We therefore introduce the Baton family of looped transformers. They pass the intermediate entity to the second loop as a token embedding. When trained to predict only the answer, two of its three variants answer these questions almost perfectly. We then relate the success and failure of each model to three design conditions. Experiments that change one condition at a time agree with this hypothesis. When Baton is trained on every token
    
[^66]: 通过监督式候选生成与选择性大语言模型调用实现高调用效率的ICD-10编码预测

    Call-Efficient ICD-10 Code Prediction via Supervised Candidate Generation and Selective LLM Invocation

    [https://arxiv.org/abs/2611.21891](https://arxiv.org/abs/2611.21891)

    该论文提出了一种结合监督式候选生成、候选重排序与门控式选择性LLM调用的高效ICD-10编码预测流程，在防泄漏验证协议下实现高准确率的同时降低大语言模型调用成本。

    

    从简短的临床诊断文本中自动分配ICD-10编码是一项实用的医学文本分类与规范化任务。该任务的难点在于诊断短语短且含噪、ICD-10标签空间具有层次结构，以及类别分布严重不平衡。本文提出了一种高调用效率的ICD-10编码预测流程，该流程结合了Transformer嵌入、基于树的监督式候选生成、候选级别的重排序以及门控式大语言模型辅助。评估采用了防泄漏的验证协议：模型与阈值选择基于折外交叉验证预测，而最终结果报告则在无共享源文件组的锁定测试集上进行。监督式条件父编码候选生成器在锁定测试集上达到了74.32%的Top-1准确率和93.05%的Top-10候选覆盖率。直接的零样本GPT-5.6-Sol基线达到了81.01%的Top-1准确率。

    arXiv:2611.21891v1 Announce Type: new  Abstract: Automatic assignment of ICD-10 codes from short clinical diagnosis text is a practical medical text classification and normalization task. The task is difficult because diagnosis phrases are short and noisy, the ICD-10 label space is hierarchical, and the class distribution is strongly imbalanced. This paper presents a call-efficient ICD-10 code prediction pipeline that combines transformer embeddings, tree-based supervised candidate generation, candidate-level reranking, and a gated LLM assistance. The evaluation uses a leakage-safe validation protocol: model and threshold selection are based on out-of-fold cross-validation predictions, while final reporting is performed on a locked test set with no shared source-file groups. The supervised conditional parent-code candidate generator reaches 74.32% Top-1 accuracy on the locked test set and 93.05% Top-10 candidate coverage. A direct zero-shot GPT-5.6-Sol baseline reaches 81.01% Top-1 acc
    
[^67]: 知其序而不知其量：语言模型能对其无法定位的政党立场进行排序

    Knowing Which, Not How Much: Language Models Order Party Positions They Cannot Place

    [https://arxiv.org/abs/2611.21878](https://arxiv.org/abs/2611.21878)

    研究发现，语言模型能够可靠地对日本和韩国政党的立场进行排序，却无法将这些立场准确定位到具体量级上，这揭示了“顺序”与“量级”是两种可分离的知识，也意味着现有的单实体打分基准可能误读了模型的领域知识水平。

    

    语言模型越来越多地被问及现实行为体的立场：某政党的纲领优先考虑什么，某机构承诺了什么。这类判断是可分级的，既包含顺序，也包含量级。然而，几乎所有测试这类能力的基准都一次只询问一个实体，然后将其得分解读为模型对该领域的知识。事实上，顺序与量级并不一定是同时习得的：关于社会实体的文本所包含的比较信息远多于测量信息，因此模型可能掌握可靠的排序能力，但其背后却没有锚定的量表支撑。我们探究模型究竟是能够将已有记录的社会立场放置到某个量表之上，还是只能对其所比较的行为体进行排序，以及这对衡量此类知识的方式意味着什么。我们在日本和韩国的政党立场上进行测试，比较宣言项目的编码数据为此提供了难得的外部标准答案。当被要求对四个政党进行排序时，模型的表现超过了最强的固定答案基线。

    arXiv:2611.21878v1 Announce Type: new  Abstract: Language models are increasingly asked where real actors stand: what a party's platform prioritises, what an institution has committed to. Such judgments are gradable, carrying both an order and a magnitude. Nearly every benchmark that tests them asks about one entity at a time, then reads that score as the model's knowledge of the domain. Order and magnitude, however, need not be acquired together. Text about social entities carries far more comparison than measurement, so a model may hold a dependable ordering with no anchored scale beneath it. We ask whether a model can place a documented social position on a scale or only order the actors it compares, and what that means for how such knowledge is measured. We test this on party positions in Japan and South Korea, where the Comparative Manifesto Project's coded data supplies a rare external answer key. Asked to order four parties, models beat the strongest fixed-answer baseline by mor
    
[^68]: MoE扩散解码的隐性代价：剖析与修复采样器

    The Hidden Cost of MoE Diffusion Decoding: Profiling and Fixing the Sampler

    [https://arxiv.org/abs/2611.21852](https://arxiv.org/abs/2611.21852)

    该论文发现量化后MoE扩散语言模型的解码瓶颈在于画布级logits采样流水线而非专家矩阵计算，并提出两个无损优化层级（比特精确与精度对等）将其消除，从而大幅加速解码。

    

    掩码扩散语言模型通过迭代去噪整个token画布来进行解码，因此每次前向传播都需要在整个“画布×词表”的logits张量上运行逐token的采样与停止流水线。我们在带宽受限的边缘加速器（NVIDIA GB10）上对一个26B参数的扩散混合专家模型进行了单用户解码的性能剖析，并报告了一个容易被忽视的发现：在权重量化之后，这一logits流水线（而非专家GEMM计算）成为解码时间的最大消耗者。与自回归解码不同——自回归解码每提交一个token只需对单个位置的logits进行采样——扩散解码每输出一个token需要处理S倍的logits条目，其中S为去噪步数。我们证明该流水线在很大程度上是冗余的或可融合的，并以两个无损层级将其移除：一个是比特精确层级，其输出与参考解码器字节完全一致；另一个是精度对等层级，使用融合的单（原文摘要在此处截断）。

    arXiv:2611.21852v1 Announce Type: cross  Abstract: Masked diffusion language models (dLLMs) decode by iteratively denoising a full canvas of tokens, so every forward pass runs a per-token sampling and stopping pipeline over the entire canvas-by-vocabulary logits tensor. We profile single-user decoding of a 26B-parameter diffusion Mixture-of-Experts model on a bandwidth-bound edge accelerator (NVIDIA GB10) and report a result that is easy to miss: after weight quantisation, this logits pipeline, not the expert GEMMs, is the single largest consumer of decode time. Unlike autoregressive decoding, where sampling touches one position's logits per committed token, diffusion decoding processes S times as many logit entries per output token, where S is the number of denoising steps. We show this pipeline is largely redundant or fusable and remove it in two lossless tiers: a bit-exact tier whose output is byte-identical to the reference decoder, and an accuracy-parity tier using a fused single-
    
[^69]: TUSI：一个基于引用溯源的图结构基准测试

    TUSI: A Citation-Grounded Graph-Based Benchmark

    [https://arxiv.org/abs/2611.21823](https://arxiv.org/abs/2611.21823)

    该论文提出了TUSI-Bench——一个建立在包含7,261条法律条文的多层知识框架和法律知识图谱之上的结构化波斯语法律基准，用于对具有引用溯源能力的法律RAG系统进行严格评估。

    

    对波斯语（伊朗）法律领域的检索增强生成（RAG）系统进行评估仍然具有挑战性，因为现有基准缺乏真实性、引用溯源性以及对法律推理的覆盖。现有资源要么是从成文法文本中人工合成的，缺乏自然的信息需求；要么来源于非正式的法律问答资源，与权威法律文件之间没有明确的关联，因此不适合对有据可依的法律推理进行严格评估。为弥补这一空白，我们提出了TUSI-Bench，一个专为评估基于RAG的系统而设计的结构化波斯语法律基准。TUSI-Bench建立在一个多层法律知识框架之上，该框架包含7,261条法律条文，并整合了成文法典、法律评注、学理著作以及最高法院统一裁定。我们还构建了一个连接法条与法律概念的知识图谱，以支持结构感知的检索。基于这一基础，TUSI

    arXiv:2611.21823v1 Announce Type: cross  Abstract: Evaluating retrieval-augmented generation (RAG) systems in Persian law remains challenging because benchmarks lack realism, grounding, and coverage of legal reasoning. Existing resources are either synthetically generated from statutory texts, lacking natural information needs, or derived from informal legal question--answer sources without explicit linkage to authoritative legal documents, making them unsuitable for rigorous evaluation of grounded legal reasoning. To address this gap, we introduce TUSI-Bench, a structured Persian legal benchmark designed for the evaluation of RAG-based systems. TUSI-Bench is built upon a multi-layered legal knowledge framework containing 7,261 legal articles and integrating codified statutes, legal commentaries, doctrinal writings, and Supreme Court unification rulings. We also construct a legal knowledge graph linking articles and concepts for structure-aware retrieval. Based on this foundation, TUSI
    
[^70]: SimAI：基于心理网络模型的机制性用户模拟

    SimAI: Mechanistic User Simulation through Psychological Network Models

    [https://arxiv.org/abs/2611.21808](https://arxiv.org/abs/2611.21808)

    SimAI是一个基于心理网络模型的机制性用户模拟框架，通过图重构目标建模心理因素间的相互作用，并借助PSYCHE数据集实现了比现有患者模拟器更高的多样性和档案一致性。

    

    现实的用户模拟使得在直接实验成本高昂或可能带来潜在危害的情况下研究人机交互成为可能。然而，现有的模拟器在很大程度上忽视了驱动用户行为的心理机制。我们提出了SimAI，一个用于机制性用户模拟的框架，它通过建模心理因素之间的相互作用来塑造用户响应。为了克服监督开放式行为这一挑战，我们提出了一种图重构目标，根据模拟对话反映潜在心理关系的程度来提供图级别的奖励。为支持这一方法，我们构建了PSYCHE数据集，其中包含从516个心理治疗案例中提取的18,576个档案-网络对。自动评估显示，SimAI在保持较强档案一致性的同时，比现有的患者模拟器实现了更高的模拟多样性和行为多样性。在一项与30名心理健康专业人士开展的交互式研究中……

    arXiv:2611.21808v1 Announce Type: cross  Abstract: Realistic user simulation enables the study of human interactions when direct experimentation is costly or potentially harmful. However, existing simulators largely overlook the psychological mechanisms that drive user behavior. We introduce SimAI, a framework for mechanistic user simulation that models how psychological factors interact to shape user responses. To overcome the challenge of supervising open-ended behavior, we propose a graph reconstruction objective that provides graph-level rewards based on how well simulated conversations reflect the underlying psychological relationships. To support this approach, we construct PSYCHE, a dataset of 18,576 profile-network pairs derived from 516 psychotherapy cases. Automatic evaluations show that SimAI achieves greater simulation and behavioral diversity than existing patient simulators while maintaining strong profile alignment. In an interactive study with 30 mental-health professio
    
[^71]: 腐蚀纽带：探究多智能体网络拓扑对偏见注入的鲁棒性

    Corrupting Ties: Investigating Robustness to Bias Injection Across Multi-Agent Network Topologies

    [https://arxiv.org/abs/2611.21804](https://arxiv.org/abs/2611.21804)

    该论文提出受控偏见注入框架与决策分歧度量指标，系统研究了多智能体网络拓扑对偏见传播的鲁棒性，发现序列网络受偏见注入影响产生的决策分歧最大。

    

    AI系统正日益被部署于高风险决策场景中，其中组件间的交互可能引入公平性风险，而这种风险在单独审计组件或集体结果时往往无法被发现。随着基于LLM的多智能体系统日益普及，理解偏见信息如何在智能体交互中传播对于评估其公平性至关重要。我们提出了一种受控偏见注入框架，将选定的智能体响应替换为预定义的偏见响应，从而能够通过干预分析来考察网络拓扑和注入配置如何塑造系统行为。我们提出了决策分歧度量指标，在智能体、网络和系统层面量化这些干预的影响，并使用来自Qwen和Gemma系列的LLM在三个二分类任务上评估了五种拓扑结构。序列网络显示出最大的分歧，而全连接……

    arXiv:2611.21804v1 Announce Type: new  Abstract: AI systems are increasingly deployed in high-stakes decision-making contexts, where interactions among components can introduce fairness risks that remain undetected when auditing components or collective outcomes in isolation. As LLM-based multi-agent systems become more prevalent, understanding how biased information propagates through agent interactions is essential for assessing their fairness. We introduce a controlled bias injection framework that replaces selected agent responses with predefined biased ones, enabling an intervention-based analysis of how network topology and injection configurations shape system behavior. We propose decision divergence metrics that quantify the effects of these interventions at the agent, network, and system levels, and evaluate five topologies on three binary classification tasks using LLMs from the Qwen and Gemma families. Sequential networks show the greatest divergence, whereas fully connected
    
[^72]: 记账不等于判断：EJBench——基于共轭传热设计活动构建的工程判断基准

    Bookkeeping Is Not Judgement: EJBench, an Engineering Judgement Benchmark Built from a Conjugate Heat Transfer Design Campaign

    [https://arxiv.org/abs/2611.21707](https://arxiv.org/abs/2611.21707)

    本文提出EJBench——一个基于电池冷板共轭传热设计活动构建的与软件无关的工程判断基准，考察模型能否从少量已有结果预测设计在新工况下的行为并做出决策，结果发现最佳前沿模型仅得0.71分，且模型答错时表达的置信度与答对时几乎相同。

    

    我们提出了EJBench，一个工程判断基准：考察模型能否从少量早期结果中预测设计在新工况下的表现，并据此做出决策。与那些测试智能体能否操作仿真或CAD软件的工程基准不同，EJBench与具体软件无关，考察的是工程师在不运行仿真的情况下会得出什么结论。其任务来自电池冷板的共轭传热设计活动，分为三类：将未标注的测试结果与其运行工况进行匹配、在签署设计之前选择应揭示哪些已存储的仿真运行结果、以及按峰值电芯温度对新任务进行排序。每个答案都取自收敛的仿真结果，而非LLM裁判。十个前沿模型中表现最好的GPT-6 Astra得分为0.71，且模型在答错的任务上所表达的置信度与答对的任务几乎相同。我们还使用LoRA和强化学习对Qwen3.5-9B进行了后训练……（摘要原文在此处截断）

    arXiv:2611.21707v1 Announce Type: cross  Abstract: We introduce EJBench, a benchmark of engineering judgement: whether a model can predict how a design behaves at new conditions from a few earlier results, and decide on it. Unlike engineering benchmarks that test whether an agent can operate simulation or CAD software, EJBench is software-agnostic and asks what an engineer would conclude without running a simulation. Its tasks come from a conjugate heat transfer design campaign for battery cold plates, in three families: matching unlabelled test results to their operating points, choosing which stored runs to reveal before signing off a design, and ranking new duties by peak cell temperature. Every answer is taken from a converged simulation, not from an LLM judge. The best of ten frontier models, GPT-6 Astra, scores 0.71, and models state almost the same confidence on tasks they get wrong as on tasks they get right. We also post-train Qwen3.5-9B with LoRA and reinforcement learning, r
    
[^73]: 消解语言模型中道德回路与道德特征的歧义：道德内容、规范性判断还是线索词？来自14个模型的证据

    Disambiguating Moral Circuits and Moral Features in Language Models: Moral content, normative judgment or cue words? Evidence from 14 models

    [https://arxiv.org/abs/2611.21705](https://arxiv.org/abs/2611.21705)

    通过227对最小短语对和14个开源模型的实验，研究发现改变语言模型道德答案的注意力头并非道德专属，而是属于一个对所有类型规范性判断共用的通用评估回路，其效应与“道德上”这一线索词无关。

    

    可解释性方法能够定位那些改变语言模型对道德提示所给出答案的注意力头和稀疏自编码器（SAE）特征，例如“为垂死的孩子偷药在道德上是”这句话会续接“正当的”还是“错误的”。这样的组件可能响应的是道德内容、以任何规范对行为的判断，或是提示中的“道德上”一词。我们构建了227对经五名标注者验证的短语最小对，其两个版本分别在道德标准、另一种规范（审慎、礼仪、审美或证据）或描述性属性上存在差异，并将它们呈现给14个开源权重模型（Llama、Gemma和Qwen，参数量1B至14B，包括基础版和指令微调版），提示词中分别包含或不包含“道德上”一词。在每一个模型中，改变道德答案的注意力头都属于模型在所有被测试的规范性判断类型中所使用的同一个评估回路：这些注意力头承担了相同比例的效应……

    arXiv:2611.21705v1 Announce Type: cross  Abstract: Interpretability methods can locate the attention heads and sparse autoencoder (SAE) features that change the answer a language model gives to a moral prompt, e.g., whether "Stealing medicine for a dying child is morally" continues with "justified" or "wrong". Such a component could respond to moral content, to the judgment of conduct by any norm, or to the word "morally" in the prompt. We built 227 minimal pairs of short phrases, validated by five annotators, whose two versions differ in a moral standard, in another kind of norm (prudence, etiquette, aesthetics or evidence) or in a descriptive quality, and showed them to 14 open-weight models (Llama, Gemma and Qwen, 1B to 14B, base and instruction-tuned), with and without the word "morally". In every model, the heads that change the moral answer belong to an evaluation circuit the models use for every kind of normative judgment we tested: these heads carry the same share of the effect
    
[^74]: 开放权重大语言模型修改下的水印证据：低秩输出水印的精确界

    Watermark Evidence under Open-Weight LLM Modification: Exact Bounds for Low-Rank Output Watermarks

    [https://arxiv.org/abs/2611.21702](https://arxiv.org/abs/2611.21702)

    该论文针对开放权重大语言模型权重被修改后水印验证失效的问题，推导出残差约束下水印证据的精确有限词表下界以及序列级检测的严格统计界，证明 OpenStamp 式验证器在模型被修改的情况下依然可靠。

    

    开放权重语言模型在其发布的权重被修改后仍可继续生成文本，而水印验证器仍会依据原始基础模型与带水印模型对输出进行评分。我们研究此类修改如何改变 OpenStamp 式验证器可获得的证据。在每个共享前缀处，我们将修改表示为下一词元 logits 中的残差。我们推导出残差跨度约束下期望对数似然比证据的精确有限词表下界，以及一个仅依赖跨度的锐利偏差界。我们还通过一个带有显式二阶余项的协方差项刻画了一阶证据偏移，并确定了达到目标证据水平所需的残差跨度。对于序列级检测，我们推导出精确的条件指数矩界，并将其与超鞅论证和无分布假设的次序统计容许界相结合，从而得到了一种高性能的（摘要在此处被截断）检测方法。

    arXiv:2611.21702v1 Announce Type: cross  Abstract: Open-weight language models can continue generating after their released weights are modified, while a watermark verifier still scores outputs against the original base and marked models. We study how such modifications change the evidence available to an OpenStamp-style verifier. At each shared prefix, we represent the modification as a residual in the next-token logits. We derive an exact finite-vocabulary lower bound on the expected log-likelihood-ratio evidence under a residual-span constraint, along with a sharp span-only deviation bound. We also characterize the first-order evidence shift through a covariance term with an explicit second-order remainder, and determine the residual span needed to reach a target evidence level. For sequence-level detection, we derive an exact conditional exponential-moment bound and combine it with a supermartingale argument and a distribution-free order-statistic tolerance bound. This yields a hig
    
[^75]: 令牌提交作为引导掩码扩散生成的控制界面

    Token Commitment as a Control Surface for Steering Masked Diffusion Generation

    [https://arxiv.org/abs/2611.21669](https://arxiv.org/abs/2611.21669)

    该论文提出将令牌提交策略（即每步提交哪些令牌和提交多少令牌）作为掩码扩散模型推理引导的新型控制界面，通过反事实目标感知提交和校准自适应预算两种机制，在语言和视觉-语言任务上实现了对生成过程的有效引导。

    

    掩码扩散模型在采样器决定哪些预测成为永久之前，会反复预测临时令牌。现有的推理时引导方法主要修改激活值、logits或去噪轨迹，而将这种提交策略视为固定的解码细节。我们转而将令牌提交作为一种扩散模型特有的控制界面进行研究，并将其分解为两个决策：每一步提交哪些令牌以及提交多少令牌。针对第一个决策，我们引入了反事实目标感知提交机制，通过模型置信度与留一令牌目标贡献的混合来对未解决的位置进行排序。针对第二个决策，我们引入了由目标进度和置信度驱动的校准自适应预算。在LLaDA家族的语言和视觉-语言设置中，跨多个目标、两种激活引导方法以及无引导对照条件，两种机制均在经验上产生单调的目标质量提升。

    arXiv:2611.21669v1 Announce Type: cross  Abstract: Masked diffusion models repeatedly predict provisional tokens before a sampler decides which predictions become permanent. Existing inference-time steering methods primarily modify activations, logits, or denoising trajectories, while treating this commitment policy as a fixed decoding detail. We instead study token commitment as a diffusion-specific control surface and separate it into two decisions: which tokens to commit and how many tokens to commit at each step. For the first, we introduce counterfactual target-aware commitment, which ranks unresolved positions by a mixture of model confidence and leave-one-token-out target contribution. For the second, we introduce calibrated adaptive budgets driven by target progress and confidence. Across language and vision-language settings in the LLaDA family, multiple targets, two activation-steering methods, and an unsteered control, both mechanisms yield empirically monotonic target-quali
    
[^76]: 评估金融情感分析中的语义不敏感性：对可持续金融决策的启示

    Evaluating Semantic Insensitivity in Financial Sentiment Analysis: Implications for Sustainable Financial Decision-Making

    [https://arxiv.org/abs/2611.21653](https://arxiv.org/abs/2611.21653)

    本文提出了智能体驱动的语义反转（ASI）对抗攻击框架，利用LLM多智能体流水线对金融新闻进行针对性词汇替换，以90.85%的成功率欺骗了FinBERT模型，揭示了金融情感分析模型对语义操纵的严重脆弱性。

    

    金融情感分析对于自动化交易、投资组合管理和风险评估至关重要，其严重依赖于FinBERT等模型。尽管这些模型在基准测试中准确率很高，但它们仍然容易受到细微的对抗性文本修改的影响，从而威胁算法决策的完整性。在本文中，我们提出了智能体驱动的语义反转（Agent-driven Semantic Inversion, ASI），这是一种针对金融情感模型的新型对抗攻击框架，通过特定领域的语义操纵实施攻击。我们的方法利用由大语言模型驱动的多智能体流水线，识别符合条件的金融新闻句子并进行有针对性的词汇替换。我们对ASI进行了实证评估，目标是最广泛部署的金融情感模型之一FinBERT。结果揭示了一个惊人的脆弱性：在符合条件的案例中，攻击成功欺骗了FinBERT达90.85%，模型在文本被修改后仍频繁保持其原有的情感预测。

    arXiv:2611.21653v1 Announce Type: cross  Abstract: Financial sentiment analysis is critical for automated trading, portfolio management, and risk assessment, heavily relying on models like FinBERT. Despite high benchmark accuracy, these models remain vulnerable to subtle adversarial text modifications, threatening the integrity of algorithmic decision-making. In this paper, we propose Agent-driven Semantic Inversion (ASI), a novel adversarial attack framework targeting financial sentiment models through domain-specific semantic manipulation. Utilizing an LLM-powered multi-agent pipeline, our approach identifies eligible financial news sentences and applies targeted lexical substitutions. We empirically evaluate ASI against FinBERT, one of the most widely deployed financial sentiment models. Our results reveal a striking vulnerability: the attack successfully deceived FinBERT in 90.85% of eligible cases, with the model frequently maintaining its original sentiment prediction despite the
    
[^77]: 一致性并非纠错：分离大语言模型智能体中的置信度、弃权与恢复

    Consistency Is Not Correction: Separating Confidence, Abstention, and Recovery in LLM Agents

    [https://arxiv.org/abs/2611.21651](https://arxiv.org/abs/2611.21651)

    该论文将LLM智能体置信度信号的“纠错效用”精确分解为“门控恢复减去门控损害”，并证明当初始错误数低于配对检验所需数量时，实验在数学上不可能验证纠错机制的有效性，也无法将其与从未触发的门控区分开来。

    

    重新采样答案、升级至更大模型或选择弃权的控制器，其合理性通常由置信度信号来论证。关于这种信号的三个主张并不能相互推导：它能对答案进行排序（判别能力），它能支持对部分答案不予采用（选择性预测），以及它能将答案变得更好（纠错效用）。我们对第三个主张给出了精确分解，即“门控恢复减去门控损害”，这一分解在实验运行之前就限定了比较能否成功：一个只能替换提案的门控只能作用于基础策略已经出错的地方，因此初始错误数量限制了任何正确门控所能产生的不一致对数的上限。当该上限低于配对检验所需的数量时，实验既无法为有效机制返回阳性结果，也无法将其与从未触发的门控区分开来。我们在两个模型规模上测量了这三个目标，发现该界限确实起到约束作用。在小模型套件中

    arXiv:2611.21651v1 Announce Type: cross  Abstract: A controller that resamples an answer, escalates to a larger model, or abstains is usually justified by a confidence signal. Three claims about such a signal do not follow from one another: it ranks answers (discrimination), it supports withholding some of them (selective prediction), and it changes an answer for the better (correction utility). We give an exact decomposition of the third into gated recoveries minus gated harms, which bounds whether a comparison can succeed before it is run: a gate that only replaces a proposal can act only where the base policy was already wrong, so the initial error count caps the discordant pairs any correct gate can produce. When that cap falls below the count a paired test needs, the experiment cannot return a positive result for a working mechanism, nor distinguish one from a gate that never fired. We measure all three targets on two model scales and find the bound binds. In a small-model suite a
    
[^78]: 从偏好数据到个性化：追踪大语言模型中的谄媚行为

    From Preference Data to Personalization: Tracing Sycophancy in Large Language Models

    [https://arxiv.org/abs/2611.21650](https://arxiv.org/abs/2611.21650)

    该论文提出了有效性感知的谄媚评分器和包含3,298个有效性标注探针的SycoDrift评估集，将谄媚重新定义为“不当附和”而非简单的一致，并据此系统追踪了大语言模型附和压力的三个来源：人类偏好数据、自动评判器规范与上下文个性化。

    

    大语言模型中的谄媚行为通常被衡量为对用户的附和，然而当用户的主张是有效的、其偏好是真实的、或其经历值得认同时，附和可能是恰当的。本文将谄媚研究为一种不恰当的附和：即当情境需要纠正、限定或抵制时，仍然对某一主张、偏好或自我评估表示认可。作者引入了一个有效性感知的谄媚评分器，以及SycoDrift——一个面向人设的评估集，包含3,298个带有有效性标注的探针，并对用户的开放性和事情的利害程度进行了受控标注。研究利用这一框架分析了三种附和压力的来源：人类偏好数据、自动化评判器的规范设定以及上下文中的个性化。首先，在人类数据集（Chatbot Arena、HH-RLHF、Community Alignment）中，对认可性回答的偏好较弱，且集中在附和往往恰当的场景中；其次，在选定值、单模型实验中……（原文摘要至此截断）

    arXiv:2611.21650v1 Announce Type: cross  Abstract: Sycophancy in large language models is often measured as agreement with the user, yet agreement can be appropriate when the user's claim is valid, their preference is genuine, or their experience warrants acknowledgment. We study sycophancy as inappropriate agreement: validating a claim, preference, or self-assessment when the context calls for correction, qualification, or resistance. We introduce a validity-aware sycophancy grader and SycoDrift, a persona-targeted evaluation set of 3,298 validity-annotated probes with controlled annotations for user openness and stakes. We use this framework to analyze three sources of agreement pressure: human preference data, automated judge specification, and in-context personalization. First, preference for validating responses in human datasets (Chatbot Arena, HH-RLHF, Community Alignment) is weak and concentrated where agreement is often appropriate. Second, in a selected-value, single-model ex
    
[^79]: 文学语言混搭：利用大语言模型策划小说

    Literary Language Mashup: Curating Fictions with Large Language Models

    [https://arxiv.org/abs/2611.21649](https://arxiv.org/abs/2611.21649)

    本文提出在GrAImes框架中让大语言模型充当文学评估者，通过对比LLM与人类评委（文学教授及爱好者）对西班牙语和英语小说的评估，发现LLM虽可作为辅助文学评估工具，但无法替代人类评委。

    

    本文提出了一种替代性的文学虚构作品人工评估方法，即在GrAImes框架内采用大语言模型（LLMs）本身作为评估者。我们将这一方法应用于评估西班牙语的人工写作与AI生成的微型小说，评估对象包括五位文学博士教授和十六位文学爱好者，同时也评估了西班牙语和英语的短篇小说。通过比较LLM所进行的评估与人类评委的评估，我们考察了两种视角之间的契合度与分歧程度，从而评估LLM作为辅助文学评估者的可行性。我们的分析重点关注LLM的回答与人类评估者回答的一致性，为LLM在文学评估中的潜力提供了洞见。所开展的实验表明，尽管LLM不能被视为文学微型小说和短篇小说评估中人类评委的替代品（Krippendorff一致性系数显示……

    arXiv:2611.21649v1 Announce Type: cross  Abstract: In this paper, we propose an alternative approach to human evaluation of literary fictions by employing LLMs themselves as evaluators within the GrAImes framework. We apply this methodology to assess human-written and AI-generated microfictions in Spanish, to five PhD professors in literature and sixteen literary enthusiasts, and to short stories in both Spanish and English. By comparing the evaluations performed by LLMs with those of human judges, we examine the degree of alignment and divergence between both perspectives, thereby assessing the feasibility of LLMs as auxiliary literary evaluators. Our analysis focuses on the alignment of responses from LLMs with those of human evaluators, providing insights into the potential of LLMs in literary assessment. The conducted experiments reveal that while LLMs cannot be regarded as substitutes for human judges in the evaluation of literary microfictions and short stories, with a Krippendor
    
[^80]: LLM会做梦吗？在具有文本记忆的智能体上测试“过拟合大脑”假说

    Can an LLM dream? Testing the overfitted brain hypothesis on an agent with text memory

    [https://arxiv.org/abs/2611.21648](https://arxiv.org/abs/2611.21648)

    该研究将“做梦防止大脑过拟合”假说应用于具有文本记忆的LLM智能体，通过在每晚记忆重写前对当天对话记录施加细节损坏或随机文本损坏来模拟梦境扭曲，结果显示这种损坏并未带来帮助。

    

    Hoel（2021）提出，梦境会对重放的经验进行扭曲，从而防止大脑对近期事件产生过拟合。我们在一个LLM智能体（Claude Opus 5.5）上测试了这一假说，该智能体将所学内容存储在文本记忆中，并在每晚根据当天的对话记录重写这份记忆。该智能体经历了由LongMemEval构建的十个模拟天的对话。我们在每晚记忆重写之前，以两种方式之一损坏对话记录中5%、10%、20%或40%的轮次：细节损坏方式将某一轮中的一两个细节改为看似合理的错误内容，或用同一说话者在更早一天的某轮对话替换该轮；随机文本损坏方式用《白鲸记》中的一段文字替换一半被选中的轮次，并在另一半轮次中插入《白鲸记》的词汇。记忆大小限制在十晚内从19,500个token逐步提升至117,000个token（该上限基于人类自由回忆数据设定），迫使智能体遗忘。我们发现，损坏并未对该智能体起到帮助作用……

    arXiv:2611.21648v1 Announce Type: cross  Abstract: Hoel (2021) proposed that dreams distort replayed experience so that the brain does not overfit to recent events. We test this hypothesis on an LLM agent (Claude Opus 5.5) that stores what it learns in a text memory and rewrites that memory each night from the day's conversation transcript. The agent goes through ten simulated days of conversation built from LongMemEval. We corrupt 5, 10, 20 or 40 percent of the transcript's turns before each nightly rewrite in one of two ways. Detail corruption changes one or two details in a turn to plausible wrong ones, or replaces the turn with one by the same speaker from an earlier day. Random-text corruption replaces half of the selected turns with a passage of Moby-Dick and inserts Moby-Dick words into the other half. A size limit on the memory, rising from 19,500 to 117,000 tokens over ten nights, set from human free-recall data, forces it to forget. We learned that corruption did not help the
    
[^81]: FieldGuide：面向智能体高效词元文档探索的语料库地图

    FieldGuide: A Corpus Map for Token-Efficient Agentic Document Exploration

    [https://arxiv.org/abs/2611.21647](https://arxiv.org/abs/2611.21647)

    FieldGuide通过将文档集合组织为主题层次结构并提供定位、范围化搜索和受限阅读工具，使智能体在新闻文档探索任务中相比原始文件访问和RAG基线显著减少词元消耗（27%-38%）和耗时（21%-41%），同时保持相当的准确性。

    

    生成式AI智能体可以帮助记者探索大型文档集合，但探索性工作流程可能需要大量的词元消耗和时间，尤其是当任务范围宽泛且相关子问题尚不明确时。本文介绍了FieldGuide，这是一种面向智能体的语料库地图，它将文档集合组织成主题层次结构，并提供语料库定位、范围化搜索和受限阅读等工具。作者使用38份芝加哥警察局公共记录文档，在五项新闻文档探索任务上对FieldGuide进行了评估，并将其与原始文件访问和检索增强生成（RAG）进行了比较。在这五项任务中，FieldGuide相比原始文件访问少使用了27%的词元和41%的耗时；相比RAG少使用了38%的词元和21%的耗时。准确性审计在FieldGuide的回应中发现了四个小问题，与其他条件相当。在对生成报道的盲评中……（摘要原文在此处截断）

    arXiv:2611.21647v1 Announce Type: new  Abstract: Generative AI agents can help journalists explore large document collections. But exploratory workflows can require substantial token use and time, especially when the task is broad and the relevant subquestions are not yet known. This paper introduces FieldGuide, an agent-facing corpus map that organizes a document collection into a topic hierarchy and exposes tools for corpus orientation, scoped search, and bounded reading. I evaluate FieldGuide on five journalistic document-exploration tasks using 38 Chicago Police Department public-records documents, comparing it with raw-file access and retrieval-augmented generation. Across the five tasks, FieldGuide used 27% fewer tokens and 41% less elapsed time than raw-file access; relative to RAG, it used 38% fewer tokens and 21% less elapsed time. An accuracy audit identified four minor issues in FieldGuide's responses, comparable to the other conditions. In blind ratings of generated reporti
    
[^82]: Limite 1B - Violetto：技术报告

    Limite 1B - Violetto: Technical Report

    [https://arxiv.org/abs/2611.21646](https://arxiv.org/abs/2611.21646)

    Limite 1B - Violetto 是一个仅 10 亿参数的数学推理模型，通过在精心策划的数学语料上从头训练，并结合知识蒸馏、监督微调与 RLVR 后训练，在 AIME 2026 上取得 94.01% 的成绩，性能媲美参数量数倍于它的模型。

    

    我们推出了 Limite 1B - Violetto，这是一个专为高吞吐量数学推理设计的十亿参数模型。Limite 在一个精心策划、以数学为核心的语料库上从头训练，共使用了 2370 亿个 token，并支持最长 131k token 的上下文长度。其架构与训练方案是协同开发的，旨在最大化推理能力的同时保持训练和推理成本较低。后训练阶段结合了知识蒸馏、针对性的监督微调以及在精选课程上的 RLVR（基于可验证奖励的强化学习）。Limite 在 AIME 2026 上取得了 94.01% 的成绩，在 BeyondAIME 上取得了 74.25% 的成绩，在我们的评测套件中跻身于参数量为其数倍的模型之列。该模型刻意只进行了轻度的指令微调：它主要被定位为单轮数学求解器以及大型系统中快速并行运行的组件，而非通用助手。我们发布了该模型、其基础模型和价值模型检查点，以及一套经过优化的服务栈。

    arXiv:2611.21646v1 Announce Type: new  Abstract: We present Limite 1B - Violetto, a one-billion-parameter model designed for high-throughput mathematical reasoning. Limite is trained from scratch on 237 billion tokens from a carefully curated, mathematics-centric corpus and supports context lengths of up to 131k tokens. Its architecture and training recipe were developed together to maximize reasoning capability while keeping training and inference costs low. Post-training combines knowledge distillation, targeted supervised fine-tuning, and RLVR on a curated curriculum. Limite achieves 94.01% on AIME 2026 and 74.25% on BeyondAIME, placing it among models many times its size on our evaluation suite. The model is deliberately only lightly instruction-tuned: it is intended primarily as a single-turn mathematical solver and as a fast, parallel component of larger systems, rather than as a general-purpose assistant. We release the model, its base and value checkpoints, and an optimized ser
    
[^83]: 追问何为关键：面向交互式Text-to-SQL的分歧引导歧义检测

    Asking What Matters: Disagreement-Guided Ambiguity Detection for Interactive Text-to-SQL

    [https://arxiv.org/abs/2611.21645](https://arxiv.org/abs/2611.21645)

    提出DisGuAD框架，将多个合理SQL解释之间的分歧作为判断依据，在目标SQL未知的情况下定位真正影响最终查询的歧义，从而在有限的交互轮次内更聚焦、更高效地选择澄清问题，改进交互式text-to-SQL性能。

    

    交互式text-to-SQL要求大语言模型（LLM）智能体不仅能生成可执行的SQL，还需在有限的交互预算内决定应澄清哪些不确定性。然而，当前的澄清策略存在三个局限：（i）仅针对表面歧义，而不检验该歧义对SQL的影响；（ii）遗漏了隐含在模式路径或领域知识中的歧义；（iii）将有限的对话轮次耗费在不聚焦或冗余的问题上。这些局限揭示了一个关键挑战：在目标SQL尚未确定之前，识别出真正会影响SQL的歧义。本文提出DisGuAD，一个分歧引导的歧义检测框架，专门用于澄清问题的选择。DisGuAD将多个合理SQL解释之间的分歧作为直接证据，用以判断哪些不确定性对最终查询真正重要。具体而言，它构建查询词项清单，通过互补的探测族群引出对比性的SQL解释……

    arXiv:2611.21645v1 Announce Type: cross  Abstract: Interactive text-to-SQL requires an LLM agent not only to generate executable SQL, but also to decide which uncertainty to clarify within a limited interaction budget. However, current clarification policies have three limitations: (i) they target surface ambiguity without testing its SQL impact; (ii) they miss ambiguities implicit in schema routes or domain knowledge; and (iii) they spend limited turns on unfocused or redundant questions. These limitations expose a critical challenge: identifying SQL-impacting ambiguity before the intended SQL is known. In this paper, we propose DisGuAD, a Disagreement-Guided Ambiguity Detection framework that specializes clarification selection. DisGuAD treats disagreement among plausible SQL interpretations as direct evidence of which uncertainty matters to the final query. Specifically, it constructs a query-term inventory, elicits contrastive SQL interpretations through complementary probe familie
    
[^84]: MInfoVisQA：面向信息丰富型视觉问答的多语言基准测试

    MInfoVisQA: A Multilingual Benchmark for Information-Rich Visual Question Answering

    [https://arxiv.org/abs/2611.21644](https://arxiv.org/abs/2611.21644)

    该论文提出了 MInfoVisQA——一个涵盖 24 种语言、8,960 个问答配置的信息丰富型视觉问答多语言基准，评估揭示了多模态大语言模型在不同语言间存在显著的鲁棒性差距，总体准确率掩盖了性能波动。

    

    多模态大语言模型（MLLMs）在视觉-语言任务中表现出色；然而，当表面语言发生变化时，它们能否对信息丰富的可视化保持一致的理解仍不清楚。为了量化这一能力，我们提出了 MInfoVisQA，这是一个包含 8,960 个问答配置的平行基准，这些配置源自 24 种语言中的 128 个种子实例。我们通过将数值结构数据与文本解耦并以编程方式同步翻译后的标签，构建了种子匹配、语言平行的可视化与问答对。我们在两种设置下评估了 25 个多模态大语言模型：语言对齐问答（LQA，视觉与文本语言匹配）和跨语言问答（XQA，固定问答语言而视觉语言变化）。结果表明，总体准确率掩盖了巨大的鲁棒性差距：两个平均 LQA 准确率相当（均为 80.4%）的模型，在不同语言间的性能波动分别为 7.8 和 20.3 个百分点。此外，所有 25 个多模态大语言模型……

    arXiv:2611.21644v1 Announce Type: new  Abstract: Multimodal large language models (MLLMs) excel in vision-language tasks; yet, whether they maintain consistent understanding of information-rich visuals (InfoVis) when the surface language changes remains unclear. To quantify this capability, we introduce MInfoVisQA, a parallel benchmark with 8,960 QA configurations derived from 128 seed instances across 24 languages. We construct seed-matched, language-parallel visuals and QA pairs by disentangling numerical-structural data from text and programmatically synchronizing translated labels. We evaluate 25 MLLMs under two settings: Language-Aligned QA (LQA, matching visual-text languages) and Cross-Language QA (XQA, fixed QA language with varying visual languages). The results show that aggregate accuracy masks substantial robustness gaps: two models with comparable mean LQA accuracy (80.4\%) exhibit performance spreads of 7.8 and 20.3 percentage points across languages. Moreover, all 25 mod
    
[^85]: Sleep2Word：通过时长感知的词化实现外周信号到脑电的翻译

    Sleep2Word: Peripheral-to-EEG Translation via Duration-Aware Wordization

    [https://arxiv.org/abs/2611.21643](https://arxiv.org/abs/2611.21643)

    Sleep2Word提出了一种时长感知的词化框架，能够将呼吸、心电或脉搏波等外周生理信号翻译成整夜脑电频谱图，从而在没有脑电设备的情况下也能开展基于脑电的睡眠分析。

    

    从外周生理测量中估计整晚脑电图（EEG）频谱图，可以将基于脑电的睡眠分析扩展到在没有脑电设备情况下采集的记录。该任务需要对具有不同时间结构的信号进行紧凑表示。我们提出了Sleep2Word，一个通过时长感知的生理“词”将呼吸、心电图（ECG）或光电容积脉搏波（PPG）翻译为脑电频谱图的框架。基于上下文相关的分块机制，Sleep2Word将生理嵌入分组为可变时长的块，并将其内容与时长联合量化为模态特定的词本。随后，块自回归翻译器在固定时长的块内预测目标脑电词的身份以及针对特定出现情况的时长修正。这种设计支持独立的源词本与目标词本，限制了时间漂移，并缩短了自回归解码路径。该模型在超过2万（训练数据上进行了训练）

    arXiv:2611.21643v1 Announce Type: new  Abstract: Estimating full-night electroencephalography (EEG) spectrograms from peripheral measurements could extend EEG-informed sleep analysis to recordings acquired without EEG. This task requires compact representations of signals with different temporal structures. We introduce Sleep2Word, a framework for translating respiration, electrocardiography (ECG), or photoplethysmography (PPG) into EEG spectrograms through duration-aware physiological words. Building on context-dependent chunking, Sleep2Word groups physiological embeddings into variable-duration chunks and jointly quantizes their content and duration into modality-specific wordbooks. A block-autoregressive translator then predicts target EEG word identities and occurrence-specific duration refinements within fixed-duration blocks. This design accommodates independent source and target wordbooks, limits temporal drift, and shortens the autoregressive decoding path. Trained on over 20K 
    
[^86]: RAW：现实场景中的拒答回避——一种针对已部署大语言模型的因果测量方法

    RAW: Refusal Aversion in the Wild, A Causal Measurement Method for Deployed LLMs

    [https://arxiv.org/abs/2611.21641](https://arxiv.org/abs/2611.21641)

    本文提出RAW方法，利用大语言模型在相同提示下有时拒答有时不拒答的随机性作为自然实验，首次对已部署模型的拒答进行因果测量，发现一次拒答会使约一半被测模型的用户后续再参与率下降3至17个百分点。

    

    已部署的大语言模型每天处理数亿次对话，并会拒绝相当一部分用户请求，然而拒答给用户侧带来的代价——即用户是否会因此流失以及流失程度——在任何生产模型上都未曾被测量过，这对建立在这些模型之上的面向消费者的平台而言是一个盲区。在前沿实验室产品上进行拒答的A/B测试在伦理和运营层面均受限制，而对于通过固定API部署这些模型的第三方平台来说更是无法实现。我们提出了RAW方法，利用大语言模型的随机性作为自然实验：在任何现有对话日志中，面对几乎相同的提示，同一模型有时拒答、有时不拒答，这种变化使得我们能够识别拒答对用户侧行为的因果效应。将该应用于LMSYS-Chat-1M和WildChat数据集中的15个已部署大语言模型，并采用用户聚类推断，我们发现对于约一半可精确测量的模型，一次拒答会导致后续用户再参与率下降3至17个百分点。

    arXiv:2611.21641v1 Announce Type: cross  Abstract: Deployed LLMs field hundreds of millions of conversations a day and refuse a non-trivial fraction of user requests, yet the user-side cost of refusal, whether users disengage and by how much, is unmeasured on any production model: a blind spot for the consumer-facing platforms built on them. A/B-testing refusal is ethically and operationally constrained on frontier-lab products and unavailable to third-party platforms deploying these models through a fixed API. We introduce RAW, a method that uses LLM stochasticity as a natural experiment: at near-identical prompts in any existing log, the same model sometimes refuses and sometimes does not, and that variation identifies the causal user-side effect of refusal. Applying the method to 15 deployed LLMs in LMSYS-Chat-1M and WildChat, with user-clustered inference, we find that for about half of the models we can measure precisely, a refusal costs 3 to 17 pp of subsequent re-engagement; the
    
[^87]: ClearTalk：面向构音障碍儿童的普通话构音数据分析与个性化沟通支持

    ClearTalk: Data-Driven Mandarin Articulation Analysis and Personalized Communication Support for Children with Dysarthria

    [https://arxiv.org/abs/2611.21639](https://arxiv.org/abs/2611.21639)

    ClearTalk 是一款基于智能手表的系统，它将普通话语音识别差异转化为可解释的音节级构音部位画像，为构音障碍儿童提供个性化练习与分阶段沟通支持，其准确率与专家清晰度评分高度相关（r=0.908），且自动选择的目标与言语病理学家共识高度一致。

    

    患有构音障碍的儿童可能面临言语清晰度下降以及表达日常需求困难的问题。ClearTalk 是一个基于智能手表的系统，它将普通话语音识别的差异转化为可解释的构音特征画像，用于支持个性化练习和分阶段沟通。系统将目标话语与识别话语在音节层面进行对齐，汇总到七个构音部位上，并通过计数敏感的非对称启发式方法进行排序。对于来自普通话构音障碍语音语料库的 13 名构音障碍说话者，系统的音节级准确率与专家清晰度评分的皮尔逊相关系数达到 0.908，R² 值为 0.824，p 值小于 0.001。系统选择的重点训练目标与三名盲评言语语言病理学家的共识达到 86.7% 的重合度；在 210 个目标辅音上，自动标注达到 86.7% 的一致率，Cohen's kappa 为 0.76，宏平均 F1……（原文摘要在此处被截断）

    arXiv:2611.21639v1 Announce Type: cross  Abstract: Children with dysarthria may experience reduced intelligibility and difficulty expressing everyday needs. ClearTalk is a smartwatch-based system that converts Mandarin speech-recognition differences into interpretable articulatory profiles for personalized practice and staged communication. Target and recognized utterances are aligned at the syllable level, summarized across seven articulatory positions, and ranked by a count-sensitive asymmetric heuristic. For 13 dysarthric speakers from the Mandarin Dysarthria Speech Corpus, system syllable-level accuracy showed a Pearson correlation of 0.908 with expert intelligibility, with an R-squared value of 0.824 and a p-value below 0.001. The system-selected priority targets achieved an 86.7 percent overlap with the consensus of three blinded speech-language pathologists; across 210 target consonants, automated labels achieved 86.7 percent agreement, a Cohen's kappa of 0.76, and a macro-F1 sc
    
[^88]: CrisisBench：危机心理辅导AI的评估套件

    CrisisBench: an evaluation suite for crisis counseling AI

    [https://arxiv.org/abs/2611.21622](https://arxiv.org/abs/2611.21622)

    CrisisBench是一个隐私保护的危机心理辅导AI评估套件，通过去标识化的真实热线对话、模拟患者的LLM以及经专家校准的评判标准来评估模型表现，结果显示现有LLM系统在危机对话中仍不及人类辅导员。

    

    背景：危机热线对心理健康系统至关重要，但常常面临人员不足和资金短缺的问题。大型语言模型（LLM）可以为辅导员提供支持，但现有的基准测试无法捕捉危机对话中至关重要的情绪升级、风险评估和去升级（安抚降级）过程，同时隐私限制也限制了真实对话记录的使用。方法：我们提出了CrisisBench，这是一个用于单轮和多轮危机辅导评估的隐私保护型评估套件。它包含来自200个去标识化对话的1,085个单轮任务，以及基于热线元数据衍生的1,185个患者LLM模拟模型。患者模型和一个包含23个评分项目的LLM评判器均经过专家辅导员的校准。结果：我们评估了五个GPT模型以及“先起草后编辑”和检索增强生成（RAG）方法。在单轮评估中，所有系统在过度共情和提问方面均不及人类辅导员。该评分标准在人与人之间的Krippendorff's α系数为0.51，LLM与人之间的α系数为0.54（Pear...

    arXiv:2611.21622v1 Announce Type: new  Abstract: Background: Crisis helplines are essential to mental health systems but are often understaffed and underfunded. LLMs could support counselors, but existing benchmarks do not capture the escalation, risk assessment, and de-escalation central to crisis conversations, while privacy restrictions limit use of real transcripts. Methods: We present CrisisBench, a privacy-preserving suite for single- and multi-turn crisis counseling evaluation. It includes 1,085 single-turn tasks from 200 de-identified conversations and 1,185 patient LLMs derived from hotline metadata. Patient models and a 23-item LLM judge were calibrated against expert counselors. Results: We evaluated five GPT models plus draft-then-edit and retrieval-augmented generation. All systems underperformed human counselors on excessive empathy and questioning in single-turn evaluation. The rubric achieved human-human Krippendorff's $\alpha$ = 0.51 and LLM-human $\alpha$ = 0.54 (Pear
    
[^89]: 从代码仓库到训练环境：通过环境演化循环扩展长时程编码智能体

    From Repositories to Training Environments: Scaling Long-Horizon Coding Agents with Environment Evolution Loop

    [https://arxiv.org/abs/2611.21152](https://arxiv.org/abs/2611.21152)

    RepoFactory提出环境演化循环，利用及时可靠的代理反馈自动将真实代码仓库转化为可验证的训练环境，并大规模生成了超过一万个人类可验证的编码智能体训练环境。

    

    训练面向长时程任务的编码智能体需要具备可验证反馈的环境。现有流程通常遵循以验证为终点的线性过程，虽能确立可用性，但对如何提升训练价值缺乏指导。递归自我改进（RSI）通常需要模型推理或模型训练以及随后的下游评估，其成本和反馈延迟在长时程任务上会被进一步放大。如何利用及时可靠的代理信号来提升训练价值并大规模合成环境？我们提出了RepoFactory，这是一个自动化流程，可将真实的代码仓库转换为可验证的环境，在这些环境中，智能体在给定语言规范的情况下从零开始构建完整项目。其环境演化循环通过代理反馈，优化一个结合有效性、结构耦合度和实现范围的乘性目标。RepoFactory在17个软件领域中生成了10,159个环境。

    arXiv:2611.21152v1 Announce Type: new  Abstract: Training coding agents for long-horizon tasks requires environments with verifiable feedback. Existing pipelines often follow a linear process that ends with validation, establishing usability but providing little guidance on improving training value. Recursive self-improvement (RSI) often requires model rollouts or model training followed by downstream evaluation, with costs and feedback delays amplified on long-horizon tasks. How can we use timely, reliable proxy signals to improve training value and synthesize environments at scale? We present RepoFactory, an automated pipeline that converts real repositories into verifiable environments where agents build complete projects from scratch given a language specification. Its Environment Evolution Loop optimizes a multiplicative objective combining validity, structural coupling, and implementation scope through proxy feedback. RepoFactory produces 10,159 environments across 17 software do
    
[^90]: 声音是否重要？在已部署的临床语音代理通话中评估挫败感检测

    Does the Voice Matter? Evaluating Frustration Detection on Deployed Clinical Voice-Agent Calls

    [https://arxiv.org/abs/2611.21001](https://arxiv.org/abs/2611.21001)

    在真实临床语音代理通话中，患者话语极短（约59%仅一两个词），仅依赖文本转录无法可靠检测患者的挫败情绪，语音音频信息是必要的。

    

    自主语音代理如今已能进行常规的临床随访电话，因此某种自动化的分诊不可避免，而患者产生挫败感正是分诊应当捕捉的情况之一。这些系统通常只读取语音转文字阶段生成的文本转录，而不使用语音本身，这种做法是否足够，很大程度上尚未在真实临床语音上得到验证。我们在一个已部署的临床语音代理的真实患者通话中研究了逐轮次的挫败感检测：来自六个通话的150个轮次由三名标注者进行标注，并以长度匹配的表演语料库作为对照，另外还有来自30个通话的714个轮次。患者的通话轮次绝大多数非常简短：在已部署通话中，约59%的轮次仅为一两个词，而表演语料库中这一比例仅为13%；即使将表演语料库进行长度匹配，两个语料库之间的准确率差距依然存在。在单词轮次上，仅基于文本转录的检测结果无法排除随机猜测的可能性。在五个具备音频处理能力的语言模型中，转录文本仅标记出……中的3或4个（摘要原文在此处截断）。

    arXiv:2611.21001v1 Announce Type: cross  Abstract: Autonomous voice agents now conduct routine clinical follow-up calls, so some automated triage is unavoidable. A patient becoming frustrated is one thing that triage should catch. These systems typically read only the transcript the speech-to-text stage produces, not the voice, and whether that is enough is largely untested on real clinical speech. We study per-turn frustration detection on real patient calls from a deployed clinical voice agent: 150 turns from six calls labelled by three annotators, with a length-matched acted corpus as a control, and 714 turns from 30 calls. Patient turns are overwhelmingly short: across the deployed calls, roughly 59% are one or two words, against 13% in the full acted corpus, and a corpus accuracy gap remains even after the acted set is length-matched. On one-word turns, transcript-only detection does not exclude chance. Across five audio-capable language models, the transcript flags 3 or 4 of the 
    
[^91]: CuRSE-Bench：一个用于探测文生图安全评估中文化盲点的基准测试

    CuRSE-Bench: A Benchmark for Probing Cultural Blind Spots in Text-to-Image Safety Evaluation

    [https://arxiv.org/abs/2611.20821](https://arxiv.org/abs/2611.20821)

    该论文提出CultureCraft智能体框架并构建了CuRSE-Bench基准（包含10,709个特定文化提示词和2,648个对照提示词，覆盖六个地区和十个危害维度），揭示了现有文生图安全评估因文化盲点而无法识别仅在特定文化中构成潜在危害的内容。

    

    文生图安全评估通常为每个提示词或图像分配单一的文化无关危害标签，因此仅在特定文化中才有害的内容可能被误判为无害。我们提出了CultureCraft，这是一个智能体框架，用于构建在通用评估标准下看似无害、却在目标文化中带有潜在有害含义的文生图提示词，并利用它构建了CuRSE-Bench：包含10,709个特定文化提示词和2,648个通用文化对照提示词，覆盖六个地区和十个危害维度，每个提示词均源自预定义的已记录危害向量知识库，且可追溯至至少两个独立来源。通过六个地区人格化评判者（persona judges）和一个配对的文化无关评判者对每个提示词进行评分，我们发现文化无关评判者会低估特定文化的潜在危害，而在对照提示词上则与人格化评判者的表现一致。每个地区的人格化评判者反过来在我们的评测工具下主要能够解读其所属地区的潜在危害……

    arXiv:2611.20821v1 Announce Type: new  Abstract: Text-to-image (T2I) safety evaluation typically assigns each prompt or image a single culture-blind harm label, so content that is harmful only within a particular culture can pass as benign.   We introduce CultureCraft, an agentic framework that constructs T2I prompts that appear benign under a general-purpose rubric yet carry a latent harmful meaning in a target culture, and use it to build CuRSE-Bench: 10,709 culture-specific prompts and 2,648 culture-general controls over six regions and ten harm axes, each drawn from a predefined knowledge base of documented harm vectors and traceable to at least two independent sources.   Scoring every prompt under six region persona judges and a paired culture-blind judge, we find that the culture-blind judge under-rates culture-specific latent harm while matching the persona judges on the controls.   Each persona judge, in turn, chiefly decodes its own region's latent harm under our instrument: a
    
[^92]: AI能评判你吗？对LLM辅助招聘系统部署就绪度的实证审计

    Can AI Judge You? An Empirical Audit of Deployment Readiness in LLM-assisted Hiring Systems

    [https://arxiv.org/abs/2611.20140](https://arxiv.org/abs/2611.20140)

    本文提出了一种将候选人证据、评估、权限控制与评估记录分离的LLM辅助简历筛选系统架构，并通过涵盖210份简历、14个职位和6种决策界面的受控实验，从运行就绪度与合规就绪度两个维度实证审计了其部署就绪度，发现标量分数比通过/不通过判断更能有效区分资质变化。

    

    将简历筛选工作委托给大语言模型（LLM）时，组织需要确认系统能否依据相关资质区分候选人，同时只允许被授权的用户修改筛选规则或批准例外。为此，我们开发了一种将候选人证据、评估、权限控制和评估记录相互分离的系统架构，并从两个互补的维度对其部署就绪度进行了实证检验。第一，我们通过一项涵盖210份简历、14个职位角色和6种决策界面的受控实验，评估了该系统在4个LLM上的运行就绪度。第二，我们通过测试系统如何应对未授权用户的请求以及嵌入在简历中的指令，评估了其合规就绪度。我们发现，运行性能取决于评估界面的选择：标量分数通常比简单的通过/不通过判断更能有效地区分资质变化，而比较类……（原文摘要在此处截断）

    arXiv:2611.20140v1 Announce Type: cross  Abstract: Delegating resume screening to large language models (LLMs) requires organizations to establish whether they can distinguish candidates by relevant qualifications while allowing only authorized users to change screening rules or approve exceptions. To this end, we develop a system architecture that separates candidate evidence, evaluation, permission controls, and assessment record, and empirically examine two complementary dimensions of its deployment readiness. First, we assess its operational readiness on four LLMs using a controlled experiment on 210 resumes across 14 job roles and six decision interfaces. Second, we assess its compliance readiness by testing how it handles unauthorized user requests and instructions embedded in resumes. We find that operational performance depends on the choice of evaluation interface: scalar scores generally distinguish qualification changes better than simple pass/fail judgments, while comparati
    
[^93]: B-OPSD：基于自生成问题简报的在线策略自蒸馏

    B-OPSD: On-Policy Self-Distillation with Self-Generated Problem Briefs

    [https://arxiv.org/abs/2611.19273](https://arxiv.org/abs/2611.19273)

    B-OPSD让模型自行从问题生成不含答案的简报（包含原理、适用条件和常见陷阱）来替代标准答案解作为教师模型的指导上下文，实现了无需真实解的在线策略自蒸馏，并在数学竞赛基准上取得了优于OPSD的准确率。

    

    在线策略自蒸馏（OPSD）近来受到关注，因为它无需一个单独的、通常更大的教师模型即可提供密集的教师指导。然而，OPSD以真实标准答案解作为教师模型的输入条件，这限制了它在这些解不可得时的使用。我们提出了基于自生成问题简报的在线策略自蒸馏（B-OPSD）。通过使用提示模板，初始模型或保存的学生检查点仅根据问题本身生成一个问题简报。该提示要求给出相关原理、适用条件和可能的陷阱，但不要求给出最终答案或完整解答。一个冻结的教师模型将该简报作为特权上下文，通过token级别的分布匹配来指导当前学生模型的在线策略采样。B-OPSD在训练过程中不使用任何真实标准答案解。在AIME 2024、AIME 2025和HMMT 2025上的结果显示，两种B-OPSD变体在每个基准上的平均准确率均高于OPSD。

    arXiv:2611.19273v1 Announce Type: new  Abstract: On-Policy Self-Distillation (OPSD) has recently attracted interest because it provides dense teacher guidance without a separate, often larger teacher model. However, OPSD conditions its teacher on ground-truth solutions, which limits its use when such solutions are unavailable. We propose On-Policy Self-Distillation with Self-Generated Problem Briefs (B-OPSD). Using a prompt template, the initial model or a saved student checkpoint generates a problem brief from the question alone. The prompt asks for relevant principles, applicability conditions, and likely pitfalls, but not a final answer or a complete solution. A frozen teacher uses the brief as privileged context to guide the current student's on-policy rollouts through token-level distribution matching. B-OPSD uses no ground-truth solutions during training. Results on AIME 2024, AIME 2025, and HMMT 2025 show that both B-OPSD variants achieve higher average accuracy than OPSD at eac
    
[^94]: WaferTetris：晶圆级加速器上的高效自由形状映射

    WaferTetris: Efficient Freeform Mapping on Wafer-Scale Accelerators

    [https://arxiv.org/abs/2611.18527](https://arxiv.org/abs/2611.18527)

    提出了WaferTetris系统，通过类俄罗斯方块的自由形状映射策略打破模板固定几何布局的限制，灵活地在晶圆级加速器上打包计算，从而提升处理单元利用率和可用内存容量。

    

    晶圆级加速器通过单片晶圆上互连的数十万个处理单元（PE）提供巨大的计算能力，但要发挥这一潜力需要高效的应用映射。其庞大的规模使得对这些系统进行编程以及将应用映射到其上变得极具挑战性。现有方法依赖于基于模板的映射来保持开发效率，但这些模板限制了映射的设计空间。特别是，它们在PE网格上强加了固定的几何布局，往往导致计算和内存资源利用不足。在这项工作中，我们提出了WaferTetris，一个通过自由形状映射消除这些几何限制的系统。其类俄罗斯方块的策略灵活地在整个晶圆上打包计算，在保留PE-任务代码的同时提高了PE利用率和可用内存容量。我们开发了一个基于递推的吞吐量模型，该模型捕获分支-汇合依赖关系以指导空间……（摘要截断）

    arXiv:2611.18527v1 Announce Type: cross  Abstract: Wafer-scale accelerators offer massive compute capacity through hundreds of thousands of processing elements (PEs) interconnected across a single wafer, but realizing this potential requires efficient application mapping. Their scale makes programming these systems and mapping applications onto them challenging. Existing approaches rely on template-based mappings to maintain productivity, but these templates restrict the mapping design space. In particular, they impose fixed geometric layouts over the PE grid, often leaving compute and memory resources underutilized.   In this work, we present WaferTetris, a system that removes these geometric restrictions through freeform mapping. Its Tetris-like strategy flexibly packs computation across the wafer, improving PE utilization and usable memory capacity while preserving PE-task code. We develop a recurrence-based throughput model that captures branch-and-join dependencies to guide spatia
    
[^95]: Gamer：面向长时程实时游戏智能体的通用视觉-语言-动作基础模型

    Gamer: A Generalist Vision-Language-Action Foundation Model for Long-Horizon Real-Time Game Agent

    [https://arxiv.org/abs/2611.18397](https://arxiv.org/abs/2611.18397)

    提出Gamer，一个端到端的通用视觉-语言-动作基础模型，通过海量键鼠交互轨迹预训练、稀疏思考指令微调和稳定的多轮强化学习，实现了在复杂2D/3D交互环境中执行长时程任务的实时游戏智能体。

    

    我们提出了Gamer，一个专为复杂交互环境中长时程任务构建的通用智能体框架。其端到端架构结合了感知、记忆管理和多步推理。统一的训练与部署流程支持与常规2D图形界面以及复杂3D模拟世界的交互。我们首先在从数万小时设备级键盘和鼠标交互中构建的数据集上进行继续预训练，这些轨迹涵盖模拟游戏和具身环境，使模型能够学习跨物理与虚拟领域的环境动态。随后的稀疏思考指令微调在保留复杂决策所需推理能力的同时，大幅减少了冗余计算。我们采用一种稳定的多轮强化学习方法来训练策略以实现长时程交互，该方法解决了有限上下文窗口、累积（误差/奖励）等问题（摘要在此处被截断）。

    arXiv:2611.18397v1 Announce Type: cross  Abstract: We introduce Gamer, a generalist agent framework built for long-horizon tasks in complex interactive environments. Its end-to-end architecture combines perception, memory management, and multi-step reasoning. A shared training and deployment pipeline supports interaction with both conventional 2D graphical interfaces and complex 3D simulated worlds. We first continue pre-training on a dataset constructed from tens of thousands of hours of device-level keyboard and mouse interaction. These trajectories span simulated games and embodied environments, allowing the model to learn environmental dynamics across both physical and virtual domains. Sparse-thinking instruction tuning then substantially reduces redundant computation while preserving the reasoning needed for complex decisions. We train the policy for long-horizon interaction with a stable multi-turn reinforcement learning method that addresses finite context windows, accumulated r
    
[^96]: MamaReason：临床语言模型中的未完成推理与答案恢复

    MamaReason: Unfinished Reasoning and Answer Recovery in Clinical Language Models

    [https://arxiv.org/abs/2611.18264](https://arxiv.org/abs/2611.18264)

    该研究发现临床语言模型通过答案续写恢复的诊断大多已显式存在于未完成的推理轨迹中，仅直接读取这些诊断承诺（无需额外生成）即可挽救大部分续写所恢复的失败案例。

    

    答案续写常常返回一个在未完成的临床推理中早已陈述过的诊断。我们在 MamaReason（由稠密的 Gemma 4 31B 适配而来）上，针对 MamaBench 的 217 组配对母婴健康病例演示研究了这一行为。一个显式承诺提取器在 94 例强化学习（RL）首次通过失败的案例中识别出 75 例已有诊断，在 93 例监督微调（SFT）首次通过失败的案例中识别出 75 例已有诊断。仅使用诊断字符串，续写答案与该诊断匹配的情况为 RL 中的 67/75 例、SFT 中的 58/75 例。在归档的正确性标签下，不进行任何额外生成而直接读取这些承诺，即可恢复续写在 40 个 RL 配对中挽救的 30 个。复发控制实验区分了推理轨迹中的常规提及与重复的答案选择。一个离线的二次承诺触发器量化了更早读取与配对准确率之间的权衡。由此得到的分解方法可将推理轨迹中已存在的诊断内容、对已接受答案的补全以及配对诊断一致性区分开来。

    arXiv:2611.18264v1 Announce Type: new  Abstract: Answer continuation often returns a diagnosis already stated in unfinished clinical reasoning. We study this behavior in MamaReason, adapted from dense Gemma 4 31B, on MamaBench's 217 paired maternal and child-health presentations. An explicit-commitment extractor identifies a diagnosis in 75 of 94 failed RL first passes and 75 of 93 failed SFT first passes. The continuation matches that diagnosis in 67/75 RL cases and 58/75 SFT cases, using diagnosis strings alone. Reading these commitments without additional generation recovers 30 of the 40 RL pairs restored by continuation under the archived correctness labels. Recurrence controls distinguish routine mentions from repeated answer selection. An offline second-commitment trigger measures the tradeoff between earlier readout and paired accuracy. The resulting decomposition separates diagnostic content already present in a trace, completion of an accepted answer, and paired diagnostic cor
    
[^97]: 追踪大语言模型中的知识与推理

    Tracing Knowledge and Reasoning in LLM

    [https://arxiv.org/abs/2611.17461](https://arxiv.org/abs/2611.17461)

    该论文提出了一个以任务为中心的评估框架，结合注意力、激活与表示信号来刻画大语言模型在知识型与推理型任务中的内部处理模式，将评估从分数层面深入到内部机制层面。

    

    当前对大语言模型（LLM）的评估通常在数据集或任务层面报告性能，这只能揭示模型取得了什么成果，却很少能提供与该性能相关的内部处理过程的洞察。我们提出了一个以任务为中心的框架，用于刻画基准测试所引发的模型内部处理过程。该框架将基于注意力和激活的信号与基于表示的度量相结合，以刻画从检索主导到推理主导这一连续谱系上的各类基准测试，并揭示任务层面分数所掩盖的失败模式。我们在24个多样化基准上评估了16个大语言模型，并对代表性模型和任务进行了针对性的内部状态分析，观察到知识中心型任务与推理中心型任务在处理方式上存在系统性差异。我们将评估范式从询问模型取得了什么分数，转向刻画与该分数相关联的处理模式。

    arXiv:2611.17461v1 Announce Type: new  Abstract: Current evaluations of Large Language Models (LLMs) report performance at the dataset or task level, revealing what a model achieves but offering little insight into the internal processing associated with its performance. We propose a task-centric framework for characterising the model-internal processing elicited by a benchmark. It combines attention- and activation-based signals with representation-based measures to characterise benchmarks spanning a continuum from retrieval-dominant to reasoning-dominant and to surface failure modes that task-level scores conceal. We evaluate 16 LLMs across 24 diverse benchmarks and conduct targeted internal-state analyses on representative models and tasks, observing systematic differences in processing between knowledge-centric and reasoning-centric tasks. We shift evaluation from asking what score a model achieves toward characterising the processing patterns associated with that score, potentiall
    
[^98]: Lightfoot：具有精确、无模型文档删除功能的图检索增强生成

    Lightfoot: Graph Retrieval-Augmented Generation with Exact, Model-Free Document Deletion

    [https://arxiv.org/abs/2611.11010](https://arxiv.org/abs/2611.11010)

    Lightfoot 是一种图检索增强生成系统，它将图中的每个实体、链接、事实和同义词边都定义为逐段落提取结果的确定性函数，从而实现无需语言模型调用的检索以及作为数据库事务的精确文档删除（在 11,402 段语料库上删除 10% 仅需 14 秒且结果与全新索引完全一致），解决了现有 GraphRAG 系统的模型调用成本和文档删除不精确问题。

    

    基于图的检索增强生成（GraphRAG）提升了多跳问答能力，但已部署的系统还需承担研究对比中很少被测量的成本：查询时的模型调用、图的构建以及正确地删除文档。我们提出了 Lightfoot，这是一种 GraphRAG 系统，其中每个实体、链接、事实和同义词边都是携带块级溯源信息的逐段落提取结果的确定性函数。检索过程无需调用语言模型，索引过程每个段落仅需一次调用，而删除文档则是一个无需模型调用的数据库事务：在包含 11,402 个段落的语料库上删除 10% 的文档仅需 14 秒，之后得到的图与对剩余文档重新构建的新索引完全一致。在一个包含大量近似重复名称的语料库上，HippoRAG 2 的同义词列表上限导致其结果比重建少 565 条边，而 Lightfoot 始终保持精确。在 MuSiQue 和 2WikiMultiHopQA 的保留测试问题上，使用相同的提取、嵌入和答案模型……

    arXiv:2611.11010v2 Announce Type: replace  Abstract: Graph-based retrieval-augmented generation (GraphRAG) improves multi-hop question answering, but deployed systems pay costs that research comparisons rarely measure: model calls at query time, building the graph, and removing documents correctly. We present Lightfoot, a GraphRAG system in which every entity, link, fact and synonym edge is a deterministic function of per-passage extractions that carry chunk-level provenance. Retrieval makes no language-model call, indexing makes one per passage, and deleting a document is a database transaction with no model calls: 14 seconds for 10% of an 11,402-passage corpus, after which the graph is identical to a fresh index of the surviving documents. On a corpus of near-duplicate names, HippoRAG 2's capped synonym lists leave 565 fewer edges than a rebuild, while Lightfoot remains exact. Under identical extraction, embedding and answer models on held-out questions of MuSiQue, 2WikiMultiHopQA an
    
[^99]: 基于词元身份平方监督的连续语言流继续训练方法

    Continued Training of Continuous Language Flows with Token Identity Squared Supervision

    [https://arxiv.org/abs/2611.07721](https://arxiv.org/abs/2611.07721)

    提出词元身份Brier流，通过直接对词元身份施加平方监督（Brier损失）解决了连续MSE目标无法区分不同概率分布的问题，从而更有效地继续训练预训练语言流模型。

    

    扩散模型和流模型最近被扩展到语言建模领域，通过在词元嵌入空间中进行连续去噪来生成文本。预训练模型提供了词元表示和上下文预测能力，但要在有限额外训练的情况下改进生成效果，需要在监督信号与连续生成之间建立适当的连接。连续MSE评估的是通过用预测概率对词元嵌入加权所得到的预测向量；不同的概率分布可以产生相同的向量，因此在该目标函数下无法被区分。我们提出词元身份Brier流（TIBF），在继续训练预训练语言流模型时，直接对词元身份应用平方监督。TIBF在高斯观测模型下将网络得分与词元匹配得分相结合，使用Brier损失对词元概率进行监督，并使用概率加权嵌入进行……

    arXiv:2611.07721v1 Announce Type: cross  Abstract: Diffusion and flow models have recently been extended to language modeling, generating text through continuous denoising in token embedding spaces. Pretrained models provide token representations and contextual prediction capabilities, but improving generation with limited additional training requires an appropriate connection between supervision and continuous generation. Continuous MSE evaluates a predicted vector obtained by weighting token embeddings with predicted probabilities; different probability distributions can yield the same vector and are therefore indistinguishable under this objective. We propose Token Identity Brier Flow (TIBF), which applies squared supervision directly to token identities when continuing pretrained language flows. TIBF combines network scores with token-matching scores under a Gaussian observation model, supervises token probabilities using the Brier loss, and uses probability-weighted embeddings for
    
[^100]: 以草根契约圈控人工智能风险

    Corralling AI Risks with Grassroots Contracts

    [https://arxiv.org/abs/2611.06374](https://arxiv.org/abs/2611.06374)

    提出以“草根契约”这一数字化契约机制，来圈控AI操控全球平台、匿名AI智能体逃避追责以及无法归因的深度伪造内容这三大AI风险。

    

    人工智能的转折点带来了大量令人担忧的风险。数字平台尤其容易受到三大AI风险的影响，因为全球平台的控制者正在将控制权让渡给AI——包括算法控制权、成员控制权和内容控制权：(1) AI控制被人类大多数人所使用的全球社会与经济平台，导致大多数人类容易受到AI的操控。(2) 无法问责的AI智能体在社交与经济数字平台上与人类并肩匿名运行，可能伪装成人类，为身份不明的主人服务，追求未知的目标，在未知的约束下运行，缺乏问责机制和责任追究。(3) 由匿名AI生成、无法归因的深度伪造内容，人们无法将其与真实内容区分开来，其制作者则逃避问责和追责。在此，我们提出用草根契约来圈控这些AI风险，草根契约是一种……（原文摘要在此处截断）

    arXiv:2611.06374v1 Announce Type: new  Abstract: AI's inflection point is a fountain of alarming risks. Digital platforms in particular are vulnerable to three major AI risks, as those in control of global platforms cede control to AI, of the algorithms, of membership and of content: (1) AI control of global platforms, social and economic, used by the majority of humanity, resulting in the majority of humanity becoming vulnerable to AI control. (2) Unaccountable AI agents operating anonymously on social and economic digital platforms, side-by-side with humans, possibly masquerading as humans, serving unknown masters with unknown goals, operating under unknown restraints without accountability or culpability. (3) Unattributed deepfakes generated by anonymous AI, with people unable to distinguish them from genuine content and with their producers evading accountability and culpability.   Here, we propose to corral these AI risks with grassroots contracts, a digital incarnation of a two-c
    
[^101]: 看、想、用：诊断视觉语言模型中的感知与推理失败

    See, Think, Use: Diagnosing Perception and Reasoning Failures in VLMs

    [https://arxiv.org/abs/2611.01467](https://arxiv.org/abs/2611.01467)

    该论文提出“看、想、用”协议，通过定位探针、文本替换探针和反事实图像对，在无需访问模型权重的情况下，将视觉语言模型的每个答案归因于感知、推理或感知使用，从而精准定位应改进的环节并指导测试时计算的分配。

    

    我们提出了“看、想、用”，这是一种能够将视觉语言模型的每个答案追溯到感知、推理或对感知的使用三个阶段的诊断协议。该诊断可以告诉开发者应当改进哪个阶段，并将测试时计算引导到失败的环节。基准测试只对最终答案评分，基于图像得出的答案与猜测或记忆的先验获得同样的分数。该协议在每个题目上只需向模型额外提出两个问题，且无需访问模型权重。定位探针要求模型为每个实例框出边界框，以测量“看”的能力；文本替换探针让模型基于文字描述来回答问题，以测量“想”的能力。“用”衡量的是在反事实图像对上，答案遵循模型自身感知的比率。该比率无需答案标签，因为问题和其先验在两张图像上是相同的。在两个模型家族和六个数据集上，该协议表明有效的改进手段取决于任务类型。

    arXiv:2611.01467v1 Announce Type: cross  Abstract: We present See, Think, Use, a protocol that traces each answer of a vision-language model to perception, reasoning or the use of perception. The diagnosis shows a developer which stage to improve and directs test-time compute to the stage that fails. Benchmarks score the final answer, and an answer that rests on the image earns the same credit as a guess or a memorized prior. The protocol asks the model two extra questions per item and needs no access to its weights. A localization probe asks for a box around every instance and measures seeing, and a text-replacement probe answers from a description and measures thinking. Use is the rate at which the answer follows the model's own perception across counterfactual image pairs. The rate needs no answer labels, because the question and its prior are the same on both images. On two model families and six datasets the protocol shows that the knob that helps follows the task family. Answers 
    
[^102]: 多重估计引导并不能统一地消除语言模型判断中的偏差

    Multi-Estimate Elicitation Does Not Uniformly Debias Language Model Judgments

    [https://arxiv.org/abs/2611.00597](https://arxiv.org/abs/2611.00597)

    本研究通过预注册实验发现，在单次回复中引导语言模型给出多个加权估计值的格式并不能统一地消除其认知偏差，在锚定和现状等任务中偏差效应已接近于零，尽管多项效应量低于预设的最小感兴趣效应量，但并未能确立统计等效性。

    

    在单次回复中引导模型给出五个加权估计值，是否会改变语言模型对信息的响应方式？在锚定、后见之明、基础比率、合取以及现状/默认等任务中，我们将直接数值回答与多重估计响应格式进行对比，同时比较了不同的来源框架、显式随机线索以及概率一致性约束。我们评估了三种已完成配置——gpt-6.1-sol、claude-sonnet-5.5 以及 Qwen3-8B（禁用推理，进行描述性分析）。在前两种配置中，直接引导产生的无关锚定指数和现状效应接近于零，同时模型仍能对信息性来源和知情建议保持敏感。在23项预注册的验证性测试中，有六项在经过 Holm 校正后被拒绝；然而，这六项测试的绝对点估计值均低于预先设定的最小感兴趣效应量，但并未据此建立等效性结论。此外，虽然分离引导与联合引导的（摘要在此处截断）

    arXiv:2611.00597v1 Announce Type: new  Abstract: Does eliciting five weighted estimates within a single response change how a language model responds to information? Across anchoring, hindsight, base-rate, conjunction, and status-quo/default tasks, we compare direct numerical answers with a multi-estimate response format, contrasting source framings, explicitly random cues, and probability-consistency constraints. We evaluate three completed configurations--gpt-6.1-sol, claude-sonnet-5.5, and Qwen3-8B (reasoning disabled, analyzed descriptively). In the first two configurations, direct elicitation yielded near-zero irrelevant-anchor indices and status-quo effects, while remaining responsive to informative sources and informed advice. Of 23 preregistered confirmatory tests, six rejected after Holm adjustment; however, all six absolute point estimates fell below their prespecified smallest effect sizes of interest without establishing equivalence. Moreover, while separate versus joint el
    
[^103]: DAPS：面向LLM定理证明的依赖感知前提选择

    DAPS: Dependency-Aware Premise Selection for LLM Theorem Proving

    [https://arxiv.org/abs/2610.41536](https://arxiv.org/abs/2610.41536)

    该论文提出DAPS方法，通过首次系统挖掘53个Lean Blueprint项目构建依赖关系语料库，将数学中的依赖结构信号引入前提选择的训练数据与模型中，克服了现有基于语义相似度的选择器在检索偏斜和分布外查询上的固有局限。

    

    基于LLM的Lean 4定理证明发展迅速，但证明器难以利用新贡献的引理，因为检索偏向于流行的基础引理，使得正确的前提被埋没在数十万个候选之中。现有的神经选择器按语义相似度对候选进行排序，存在三个内在局限：频率偏斜的检索、孤立的成对排序，以及在分布外查询（如竞赛题目和非正式语言）上的性能退化。鉴于数学是按依赖关系而非相似性来构建结构的，我们将这一信号重新引入训练数据和模型中。我们整理了两项未被充分利用的依赖资源。我们对53个Lean Blueprint项目进行了首次系统性挖掘，构建了一个此前未被开发的语料库，其中包含3,805个人工整理的非正式+正式节点，捕捉了Mathlib4中缺失的罕见研究级前提。我们进一步将其与提取的……（原文摘要在此截断）

    arXiv:2610.41536v1 Announce Type: cross  Abstract: LLM-based theorem proving in Lean 4 has advanced rapidly, but provers are difficult to leverage newly contributed lemmas, because retrieval skews toward popular foundational lemmas, leaving the right premises buried among hundreds of thousands of candidates. Existing neural selectors rank candidates by semantic similarity, with three intrinsic limitations: frequency-skewed retrieval, isolated pairwise ranking, and degradation on out-of-distribution queries such as competition problems and informal language. Given that mathematics is structured by dependencies rather than similarities, we restore this signal to both the training data and the model. We curate two underused dependency resources. We conduct the first systematic mining of 53 Lean Blueprint projects, a previously untapped corpus of 3,805 manually curated informal+formal nodes that capture rare, research-level premises absent from Mathlib4. We further pair it with the extract
    
[^104]: 语言模型在检索增强生成（RAG）中表现出从众偏差：当多数文档支持被投毒的论断时

    Language Models Show Conformity Bias in RAG: When the Majority Endorses a Poisoned Claim

    [https://arxiv.org/abs/2610.41012](https://arxiv.org/abs/2610.41012)

    本研究发现当RAG检索上下文中多数相互一致的投毒文档共同支持错误信息时，语言模型的投毒检测率从25%显著降至10%，揭示了语言模型存在类似社会从众心理的从众偏差，使其更容易被多数派的投毒信息误导。

    

    检索增强生成（RAG）可能使语言模型接触到检索文档中具有误导性的证据。我们研究了相互一致的投毒文档数量增加是否会影响模型检测错误信息的能力。我们从SciFact数据集中随机选取200个问题（随机种子38）进行评估，将包含一篇投毒文档和六篇健康文档的上下文，与包含六篇相互一致的投毒文档和一篇健康文档的上下文进行对比。对于Gemma-3-12b模型，投毒检测率从少数条件下的25%（50/200）下降至多数条件下的10%（20/200）。在两种条件下结果不同的38个问题中，有34个从“被检测出”变为“未被检测出”，仅有4个朝相反方向变化（精确McNemar检验，p = 6.04e-7）。这一模式与语言模型面对多数相互一致的投毒文档时产生的类从众式反应一致。我们的结果提供了初步证据……

    arXiv:2610.41012v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) can expose language models to misleading evidence in retrieved documents. We investigate whether increasing the number of mutually agreeing poisoned documents affects a model's ability to detect misinformation. We evaluate 200 randomly selected questions from SciFact (random seed 38), comparing contexts with one poisoned and six healthy documents against contexts with six mutually agreeing poisoned documents and one healthy document. For Gemma-3-12b, poison-detection rates decrease from 25% (50/200) in the minority condition to 10% (20/200) in the majority condition. Among the 38 questions with different outcomes across conditions, 34 change from detected to undetected, while only four change in the opposite direction (exact McNemar test, p = 6.04e-7). This pattern is consistent with a conformity-like response to a majority of mutually agreeing poisoned documents. Our results provide initial evidence 
    
[^105]: 多标签一致性：一种灵活的标注者间信度评估框架

    Multi-Label Agreement: A Flexible Framework for Inter-Annotator Reliability

    [https://arxiv.org/abs/2610.37292](https://arxiv.org/abs/2610.37292)

    本文提出了多标签一致性（MLA）框架，通过保留完整二值标签向量并采用平均成对汉明一致性，为多标签标注任务的标注者间信度评估提供了灵活方案，同时证明了其数学形式与Fleiss/Scott型系数及Krippendorff's alpha密切相关。

    

    对标注者间一致性（IAA）的可靠评估对于衡量标注过程的一致性至关重要。现有的IAA指标（如Cohen's kappa、Fleiss' kappa和Krippendorff's alpha）通常应用于单标签场景，而多标签任务则需要额外决定如何比较完整标签集、共同选择和共同遗漏。在这项工作中，我们提出了多标签一致性，这是一个用于评估多标签标注任务一致性的灵活框架。MLA保留了分配给每个条目的完整二值标签向量，并使用平均成对汉明一致性，从而同时考虑了共同选择和共同遗漏。其合并边际形式在数学上并非一个独立的系数：它是以偶然不一致性加权的逐标签Fleiss/Scott型系数的平均值，并与采用归一化汉明距离的Krippendorff's alpha密切相关。

    arXiv:2610.37292v1 Announce Type: new  Abstract: Reliable evaluation of inter-annotator agreement (IAA) is important for assessing the consistency of annotation processes. While established IAA metrics such as Cohen's kappa, Fleiss' kappa, and Krippendorff's alpha are commonly applied in single-label settings, multi-label tasks require additional decisions about how full label sets, joint selections, and joint omissions should be compared.   In this work, we present Multi-Label Agreement (MLA), a flexible framework for evaluating agreement in multi-label annotation tasks. MLA preserves the full binary label vectors assigned to each item and uses average pairwise Hamming agreement, thereby accounting for both joint selections and joint omissions. Its pooled-marginal form is not a mathematically independent coefficient: it is a chance-disagreement-weighted average of label-wise Fleiss/Scott-type coefficients and is closely related to Krippendorff's alpha with normalized Hamming distance.
    
[^106]: MemT-1.0：迈向记忆原生演化

    MemT-1.0: Towards Memory-Native Evolution

    [https://arxiv.org/abs/2610.26825](https://arxiv.org/abs/2610.26825)

    MemT-1.0提出了一种将记忆内化为基础模型固有计算能力的新架构，通过Gated DeltaNet-2、稳定的token级记忆更新机制和Mixture-of-Storage来扩展原生记忆容量，其发布的模型家族在记忆与通用能力评估中超越了参数化记忆基线。

    

    记忆原生演化要求将记忆作为基础模型的内在能力，使其能够持续积累和利用信息。为此，我们提出了MemT-1.0，一种将记忆内化为模型计算一部分的记忆基础模型新架构。MemT-1.0基于Metis构建，采用Gated DeltaNet-2，并在每一轮内使用稳定的token级记忆更新机制。我们进一步引入Mixture-of-Storage（存储混合）来扩展原生记忆容量。在经过精心准备的主干模型之后，我们开发了一个由继续预训练和后训练组成的两阶段训练流程，并由340亿（34B）token的训练数据以及高效的训练与推理基础设施支撑。我们发布了包含MemT-1.0-Nano和MemT-1.0-Flash的模型家族，并对记忆能力和通用能力进行了全面评估。结果表明，它们在（各项评估中）优于参数化记忆基线。

    arXiv:2610.26825v2 Announce Type: replace  Abstract: Memory-native evolution requires making memory an intrinsic capability of foundation models, enabling them to continuously accumulate and utilize information. To this end, we introduce MemT-1.0, a new architecture of memory foundation models that internalize memory as part of model computation. Building on Metis, MemT-1.0 employs Gated DeltaNet-2 with a stable token-level memory update mechanism within each turn. We further introduce Mixture-of-Storage to scale native memory capacity. Following a deliberate backbone preparation, we develop a two-stage training pipeline consisting of continued pre-training and post-training, supported by 34B tokens of training data and efficient training and inference infrastructure. We release a model family comprising MemT-1.0-Nano and MemT-1.0-Flash, and conduct a comprehensive evaluation of memory and general capabilities. The results show that they outperform the parametric memory baselines on av
    
[^107]: AdaRSI：面向具身智能体的按需递归自我改进

    AdaRSI: On-Demand Recursive Self-Improvement for Embodied Agents

    [https://arxiv.org/abs/2610.20474](https://arxiv.org/abs/2610.20474)

    AdaRSI提出了一种按需递归自我改进框架，先改进技能与外围框架、再通过重试归因持续失败、最后仅微调不可靠的子步骤，从而在无需人类示范的情况下显著提升具身智能体完成长时程任务的能力。

    

    预训练的视觉-语言-动作模型（VLA）在执行未见过的长时程任务时常常中途失败。递归自我改进（RSI）提供了一种补救方法，但现有的具身RSI方法要么只是围绕冻结的执行器改进其外围框架，因而无法补充缺失的动作；要么在持续失败的情况下直接训练执行器，却不追究失败究竟源于执行器的调用方式，还是源于其自身能力。我们提出一个问题：自我改进的具身系统应当在何时改变其执行器的权重？为此我们提出了AdaRSI，一个面向具身智能体的按需RSI框架。一个编程智能体首先改进其技能和外围框架，随后通过从保存的状态重试失败的子步骤来归因持续存在的失败，并仅对仍然不可靠的子步骤进行微调，所用数据为它自行收集的示范以及从早期任务中传承的经验。在RoboCasa365的全部50个目标任务上，以GR00T N1.5作为执行器且无需任何人类示范，AdaRSI改进了……（摘要原文在此处截断）

    arXiv:2610.20474v1 Announce Type: new  Abstract: Pretrained vision-language-action models (VLAs) often fail midway through unseen long-horizon tasks. Recursive self-improvement (RSI) offers a remedy, but embodied RSI methods either refine the harness around a frozen executor and so cannot add a missing motion, or train the executor on persistent failures without asking whether a failure lies in how the executor is called or in what it can do. We ask when a self-improving embodied system should change the weights of its executor, and propose AdaRSI, an on-demand RSI framework for embodied agents. A coding agent first improves its skills and harness, then attributes persistent failures by retrying failing sub-steps from saved states, and fine-tunes only the sub-steps that remain unreliable, with demonstrations it collects itself and experience carried over from earlier tasks. On all 50 target tasks of RoboCasa365, with GR00T N1.5 as the executor and no human demonstrations, AdaRSI improv
    
[^108]: SarcasTraps：讽刺对土耳其语仇恨言论检测的影响

    SarcasTraps: Effect of Sarcasm on Turkish Hate Speech Detection

    [https://arxiv.org/abs/2610.16957](https://arxiv.org/abs/2610.16957)

    本文构建了SarcasTraps——一个包含479条数据、沿仇恨言论与讽刺双维度标注的土耳其语外部测试集，并通过实验揭示讽刺会显著干扰仇恨言论检测：微调的编码器模型在讽刺性仇恨言论上召回率下降，零样本大语言模型同样受到影响。

    

    标准的仇恨言论检测数据集通常强调显性的有害语言，但针对性伤害也可以通过间接方式表达，而不使用明显仇恨的措辞。讽刺正是这样一种间接表达形式，它可以通过假意称赞、夸张、幽默或表面上的礼貌来传达有害意图。本文介绍了SarcasTraps，这是一个用于分析讽刺对土耳其语仇恨言论检测影响的针对性外部测试集。SarcasTraps包含479条数据，并沿两个维度进行标注：仇恨言论和讽刺。我们使用BERTurk和XLM-RoBERTa在两个现有的土耳其语仇恨言论数据集上训练分类器，并在SarcasTraps上对其进行评估，严格将其用作外部测试集。我们还在零样本设置下于SarcasTraps上评估了GPT-4.1和Llama-3.3-70B。结果表明，讽刺会显著影响仇恨言论检测。微调的编码器模型在检测讽刺性仇恨言论时召回率下降，而零样本大语言模型（原文摘要在此处截断）……

    arXiv:2610.16957v1 Announce Type: cross  Abstract: Standard hate speech detection datasets often emphasize explicit harmful language, but directed harm can also be expressed indirectly, without clearly hateful wording. Sarcasm is one such form of indirect expression, as it can convey harmful intent through mock praise, exaggeration, humor, or apparent politeness. In this paper, we introduce SarcasTraps, a targeted external test set for analyzing the effect of sarcasm on Turkish hate speech detection. SarcasTraps contains 479 entries annotated along two dimensions: hate speech and sarcasm. We train classifiers using BERTurk and XLM-RoBERTa on two existing Turkish hate speech datasets and evaluate them on SarcasTraps, using it strictly as an external test set. We also evaluate GPT-4.1 and Llama-3.3-70B in a zero-shot setting on SarcasTraps. Results show that sarcasm substantially affects hate speech detection. Fine-tuned encoder models lose recall on sarcastic hate, while zero-shot LLMs 
    
[^109]: ANCHOR：基于仓库的智能体编码中自然约束层次与排序的评估

    ANCHOR: Assessing Naturalistic Constraint Hierarchies and Ordering in Repository-Grounded Agentic Coding

    [https://arxiv.org/abs/2610.16859](https://arxiv.org/abs/2610.16859)

    提出 ANCHOR 基准，通过层次化约束分类体系、自动化约束组合流水线和基于轨迹的评估框架，首次系统性地评估基于仓库的编码智能体遵循工程约束（如修改范围、工作流顺序、验证等）的能力，弥补了现有评估仅关注功能正确性的不足。

    

    对于基于仓库的编码智能体而言，功能正确的补丁未必是可接受的：开发者还会约束任务执行的方式——修改范围、工作流顺序、验证、工具使用和报告。我们将这种能力称为“工程约束遵循”，但目前的评估方法在很大程度上忽视了这一能力。我们提出了 ANCHOR（Assessing Naturalistic Constraint Hierarchies and Ordering in Repository-Grounded Agentic Coding），一个建立在自然、多语言仓库任务之上的基准，涵盖八个语言组。ANCHOR 结合了：(i) 从真实开发者交互中提炼的包含 10 个类别和 78 个叶子类型的层次化分类体系；(ii) 一个自动化流水线，用于实例化基于任务的原子约束并将其组合为带类型的 AND/THEN/PRIORITY 关系；(iii) 一个基于轨迹的评估框架，可同时检查代码差异和执行轨迹。

    arXiv:2610.16859v2 Announce Type: replace  Abstract: For repository-grounded coding agents, a functionally correct patch is not necessarily an acceptable one: developers also constrain how the task must be carried out-modification scope, workflow order, validation, tool use, and reporting. This capability, which we call engineering constraint following, remains largely invisible to current evaluations. We introduce ANCHOR (Assessing Naturalistic Constraint Hierarchies and Ordering in Repository-Grounded Agentic Coding), a benchmark built on naturalistic, multilingual repository tasks spanning eight language groups. ANCHOR combines (i) a hierarchical taxonomy of 10 categories and 78 leaf types distilled from real developer interactions, (ii) an automated pipeline that instantiates task-grounded atomic constraints and composes them into typed AND/THEN/PRIORITY relations, and (iii) a trajectory-grounded evaluation framework that checks both the code diff and the execution trajectory with 
    
[^110]: 交流、发声与语言

    Communication, Vocalisation, and Language

    [https://arxiv.org/abs/2610.16490](https://arxiv.org/abs/2610.16490)

    本文主张大语言模型并非真正“理解”语言，而是在“执行”语言，其机制是捕捉文化智能并将符号扎根于社会学意义上的实践，而非形成符号的分布式表示。

    

    本文的核心论点是，当前这一批大语言模型并不“理解”语言，但它们在“做”语言。本文认为，大语言模型并非形成某种符号的“分布式表示”，而是在捕捉“文化智能”，并将符号扎根于社会学家所称的“实践”之中。

    arXiv:2610.16490v2 Announce Type: replace  Abstract: The contention is that the current batch of Large Language Models don't *understand* language, but they *do* it. Rather than LLMs forming some kind of ``distributed representation'' of symbols, the claim is that LLMs are capturing ``Cultural Intelligence'' and grounding symbols in what the sociologists call practices.
    
[^111]: Mingbird：一种使小型开放模型能够完成真实任务的本地优先智能体框架

    Mingbird: A Local-First Agent Harness Enabling Small Open Models to Complete Real Tasks

    [https://arxiv.org/abs/2610.02001](https://arxiv.org/abs/2610.02001)

    提出了Mingbird，一个面向Windows和Ollama的本地优先智能体框架，通过十项针对性补偿机制（如字节级净零预填充预算、完成门控和签名级循环检测）来克服小模型的典型失败形式，使2-9B的开放模型能够在普通笔记本上完成真实任务，并证明许多失败源于框架而非模型本身。

    

    小型开放权重模型（2-9B）可以在普通笔记本电脑上运行，但在云规模的智能体框架下，它们很少能完成真实任务：工具预填充溢出上下文、自我修正发散、工具演示陷入循环，任务被悄悄放弃。我们通过受控的单机对比和一个第三方基准测试提供了证据，表明这些失败中有很大一部分归因于框架本身而非模型。我们提出了Mingbird，一个面向Windows和Ollama的本地优先智能体框架，其十项机制逐一弥补小模型的失败形式，其中三项具有代表性：字节级净零预填充预算、在接受完成之前重新读取任务的完成门控，以及签名级循环检测。在LRAB（一项保持机器、模型、预算和评分固定的受控对比：4个框架 × 4个开放模型（2B-35B）× 18个真实任务，采用确定性产物评分）上，

    arXiv:2610.02001v1 Announce Type: new  Abstract: Small open-weight models (2-9B) run on ordinary laptops, but under cloud-scale agent harnesses they rarely complete real tasks: tool prefill overflows the context, self-correction diverges, tool demonstrations loop, and tasks are silently abandoned. We present evidence, from a controlled single-machine comparison and one third-party benchmark, that a substantial share of these failures is attributable to the harness rather than the model. We introduce Mingbird, a local-first agent harness for Windows and Ollama whose ten mechanisms compensate point-by-point for small-model failure forms, three of them representative: a byte-level net-zero prefill budget, a finish gate that re-reads the task before accepting completion, and signature-level loop detection. On LRAB, a controlled comparison holding machine, models, budgets, and scoring fixed (4 harnesses $\times$ 4 open models (2B-35B) $\times$ 18 real tasks, deterministic artifact scoring),
    
[^112]: 脉冲驱动的视觉-语言-动作模型

    Spike-driven Vision-Language-Action Model

    [https://arxiv.org/abs/2609.39514](https://arxiv.org/abs/2609.39514)

    提出了首个支持机器人操作端到端直接训练的脉冲驱动视觉-语言-动作（VLA）框架，利用脉冲神经网络的稀疏事件驱动计算和多赢家脉冲融合机制，解决了传统大型Transformer模型延迟高、能耗大而难以在资源受限平台部署的问题。

    

    视觉-语言-动作（VLA）模型连接了多模态理解与机器人控制，推进了具身智能的主导范式。然而，大多数现有模型依赖大型Transformer，其延迟和能耗成本阻碍了在资源受限平台上的部署。通过稀疏的事件驱动计算，脉冲神经网络为高性能和节能计算提供了一种有前景的范式。在此，我们提出了首个支持机器人操作端到端直接训练的脉冲驱动VLA框架，其主要包含三个核心组件。首先，我们开发了用于多模态感知的脉冲视觉编码器和脉冲指令编码器，将视觉观察和语言指令编码为稀疏、可靠的脉冲表示，用于后续的跨模态融合。然后，我们引入了多赢家脉冲融合用于指令引导的场景理解，采用双向top-k赢家通吃脉冲机制……（摘要在此处被截断）

    arXiv:2609.39514v1 Announce Type: new  Abstract: Vision-language-action (VLA) models bridge multimodal understanding and robotic control, advancing the dominant paradigm for embodied intelligence. However, most existing models rely on large Transformers, whose latency and energy costs hinder deployment on resource-constrained platforms. Through sparse event-driven computation, spiking neural networks offer a promising paradigm for high-performance and energy-efficient computing. Here, we propose the first Spike-driven VLA framework enabling end-to-end direct training for robotic manipulation, which mainly comprises three core components. First, we develop spiking visual and instruction encoders for multimodal perception, encoding visual observations and language instructions into sparse, reliable spike representations for subsequent cross-modal fusion. Then, we introduce Multi-Winner Spike Fusion for instruction-guided scene understanding, using bidirectional top-$k$ winner-take-all sp
    
[^113]: 语言模型能否学会预测股票价格

    Can Language Models Learn to Forecast Stock Prices

    [https://arxiv.org/abs/2609.36914](https://arxiv.org/abs/2609.36914)

    该论文在按时间顺序推进的股票价格沙盒中，通过对 Qwen3-4B 进行 SFT 与 PPO 后训练，探索语言模型能否学会利用价格、成交量等信息预测股票未来收益。

    

    后训练已被证明能显著提升语言模型在具有可验证结果任务上的表现，包括数学推理、软件工程和计算机使用。然而，同样的方法能否改善金融市场的预测则远非明确。与具有可验证结果的任务相比，不仅实际收益充满噪声，甚至什么构成进行有效预测所需的相关信息集也并非先验可知：模型必须自行决定收集哪些观察信息，然后在结果揭晓之前给出数值判断。我们在一个按时间顺序推进的股票价格沙盒中研究这一问题，其中语言模型收集价格、成交量、相对表现以及市场背景等证据，并预测未来收益。我们首先基于工具使用演示对 Qwen3-4B 进行监督微调（SFT），随后使用近端策略优化（PPO）进行训练，其终端奖励由……

    arXiv:2609.36914v1 Announce Type: new  Abstract: Post-training has been shown to significantly improve language models' performance on tasks with verifiable outcomes, including mathematical reasoning, software engineering, and computer use. However, whether the same approach can improve forecasting in financial markets is much less clear. Compared with tasks with verifiable outcomes, not only are realized returns noisy, but even what constitutes a relevant information set for making effective predictions is not obvious a priori: the model must decide which observations to gather and then commit to a numerical judgment before the outcome is known. We study this question in a chronological stock-price sandbox, where a language model gathers price, volume, relative-performance, and market-context evidence and predicts a future return. We post-train Qwen3-4B with supervised fine-tuning (SFT) on tool-use demonstrations, then proximal policy optimization (PPO) with a terminal reward given by
    
[^114]: OpenWhistle：大规模宽吻海豚发声的纵向数据集与基准

    OpenWhistle: A Large-Scale Longitudinal Dataset and Benchmark of Bottlenose Dolphin Vocalizations

    [https://arxiv.org/abs/2609.34839](https://arxiv.org/abs/2609.34839)

    该论文发布了OpenWhistle——迄今最大的公开海豚发声数据集，包含五年间录制的约18万条宽吻海豚哨声（114小时）及8,354条专家标注样本，并提供了可复现的哨声类型检测与分类基准。

    

    生物声学领域的近期进展得益于大规模语料库和标准化基准的推动，然而现有资源绝大多数以鸟类为中心，且每个物种的数据深度有限，这限制了它们在研究单一物种交流系统结构方面的应用。这一缺口对于鲸类动物尤为突出：尽管宽吻海豚（Tursiops truncatus）是非人类哺乳动物中复杂声音交流的一个引人注目的典型案例，但现有的海豚数据集规模小、碎片化，且大多封闭不公开。我们推出OpenWhistle，这是目前最大的公开海豚发声数据集。它包含约18万条哨声（共114小时），是在五年间对半自然环境中一个由五只个体组成的稳定海豚群持续录音获得的，同时配套提供经过专家标注的8,354条哨声精选子集，以及用于哨声类型检测与分类的可复现评估协议。我们还进一步发布了完整的处理流程。

    arXiv:2609.34839v2 Announce Type: replace  Abstract: Recent advances in bioacoustics have been driven by large-scale corpora and standardized benchmarks, yet existing resources are overwhelmingly bird-centric and shallow per species, limiting their use for studying the structure of a single species' communication system. This gap is particularly acute for cetaceans: despite bottlenose dolphins (Tursiops truncatus) being a compelling case of complex vocal communication among non-human mammals, existing dolphin datasets are small, fragmented, and largely closed. We introduce OpenWhistle, the largest publicly available dataset of dolphin vocalizations. It comprises approximately 180,000 whistles (114 hours) recorded over five years from a stable pod of five individuals in a semi-natural environment, paired with a curated subset of 8,354 expert-annotated whistles and reproducible evaluation protocols for whistle-type detection and classification. We further release the full processing pipe
    
[^115]: FLARE：基于生成式奖励模型的长程编码智能体全生命周期密集监督范式

    FLARE: A Full-Lifecycle Dense Supervision Paradigm for Long-Horizon Coding Agents via Generative Reward Model

    [https://arxiv.org/abs/2609.23808](https://arxiv.org/abs/2609.23808)

    提出FLARE，一种由轻量级生成式奖励模型驱动的全生命周期密集监督范式，通过离线因果诊断框架RADAR提取无后见之明的监督信号，为长程编码智能体提供实时步骤级风险反馈，解决稀疏二值奖励带来的信用分配危机。

    

    虽然测试时扩展提升了大型语言模型（LLM）智能体在长程软件工程（SWE）任务中的表现，但稀疏的二值奖励（通过/失败）造成了严重的信用分配危机，并浪费了失败的探索轨迹。当前的轨迹优化与扩展方法成本高昂且在结构上存在局限，它们要么依赖缺乏因果诊断的启发式状态复用，要么依赖缺乏可操作在线引导的延迟标量评分。我们提出了FLARE（全生命周期对齐与奖励引擎），这是一种由轻量级生成式奖励模型（GRM）驱动的新型密集监督范式。首先，RADAR作为一个离线的因果感知诊断框架，通过因果链回溯提取高保真、无后见之明的监督信号，从而蒸馏出一个能够提供实时、步骤级风险反馈的GRM。其次，FLARE利用该GRM在智能体的整个生命周期中对其进行持续优化。在推理阶段，FLARE充当主动脚手架，自主……

    arXiv:2609.23808v1 Announce Type: new  Abstract: While test-time scaling enhances Large Language Model (LLM) agents in long-horizon software engineering (SWE), sparse binary rewards (Pass/Fail) create a severe credit assignment crisis and waste failed exploratory trajectories. Current trajectory optimization and scaling methods are costly and structurally limited, relying on heuristic state reuse without causal diagnosis or delayed scalar scoring without actionable online guidance. We propose FLARE (Full-Lifecycle Alignment and Reward Engine), a novel dense supervision paradigm driven by a lightweight Generative Reward Model (GRM). First, RADAR, an offline causal-aware diagnostic framework, extracts high-fidelity, hindsight-free supervision through causal-chain backtracking to distill a GRM providing real-time, step-level risk feedback. Second, FLARE uses this GRM to continuously optimize the agent across its entire lifecycle. During inference, FLARE acts as an Active Scaffold, autonom
    
[^116]: RPMem：为LLM智能体学习跨会话的长期循环参数化记忆

    RPMem: Learning Long-Term Recurrent Parametric Memory Across Sessions for LLM Agents

    [https://arxiv.org/abs/2609.23466](https://arxiv.org/abs/2609.23466)

    RPMem提出了一种两阶段架构，将LLM智能体的每个会话编译为模型无关的潜在记忆，经任务训练的循环门控整合后映射为LoRA参数，从而实现跨会话的长期参数化记忆演化，并在更换骨干模型时保持记忆可迁移。

    

    长时间运行的LLM智能体需要能够跨会话持续存在并不断演化的记忆。基于文本的记忆在每次查询时都需要检索和重建过去的交互，随着历史记录的增长，长程性能越来越依赖于检索质量和上下文推理能力。参数化记忆将经验直接编码到模型计算中，但现有方法对跨会话记忆演化的支持有限，且其与特定骨干模型的耦合进一步限制了模型替换后的记忆复用。我们提出了RPMem，这是一种两阶段架构，通过前向计算将每个会话编译为与模型无关的潜在记忆，并通过任务训练的循环门控选择性地将其与保留的记忆进行整合。整合后的记忆随后被映射为特定于骨干模型的低秩适配（LoRA）参数，使得编码能力在骨干模型被替换时仍可迁移。在三个（数据集/任务上）的评估……

    arXiv:2609.23466v1 Announce Type: new  Abstract: Long-running LLM agents require memory that persists and evolves across sessions. Text-based memory retrieves and reconstructs past interactions at every query, making long-horizon performance increasingly dependent on retrieval quality and contextual reasoning as histories grow. Parametric memory encodes experience directly into model computation, but existing approaches provide limited support for cross-session memory evolution. Their coupling to a specific backbone further restricts memory reuse after model replacement. We introduce RPMem, a two-stage architecture that compiles each session into a model-independent latent memory through forward computation and selectively integrates it with retained memory via a task-trained recurrent gate. The consolidated memory is then mapped to backbone-specific low-rank adaptation (LoRA) parameters, allowing the encoding capability to transfer when the backbone is replaced. Evaluation across thre
    
[^117]: 记忆信任差距：持久记忆智能体中依赖模型能力的失效

    The Memory Trust Gap: Capability-Dependent Failures in Persistent-Memory Agents

    [https://arxiv.org/abs/2609.01852](https://arxiv.org/abs/2609.01852)

    该论文提出并量化了“记忆信任差距”现象：持久记忆智能体会过度信任（而非混淆于）过期的存储事实并覆盖权威证据，且这种失效受模型能力门控——规模越大的模型在过期记忆被伪装成最新信息时崩溃反而越严重。

    

    持久记忆为个性化智能体提供支持，但一条过期的存储事实可能在毫无警告的情况下覆盖当前的权威证据。我们研究随着模型能力的变化，这种危害何时开始显现。我们评估了一个冻结的、闭集的、按动作评分的基准测试，该基准包含两个测试套件，分别代表“无记忆”的两种不同含义（一个是“收益”套件，在没有存储事实的情况下无法解决；另一个是“安全”套件，其中权威工具始终持有正确值），并在同一系列不同规模的模型（Qwen3 0.6/1.7/4/8B）上进行测试。我们发现“记忆信任差距”反映的是过度信任而非混淆。在收益套件中，所有规模的模型在0.92-1.00的情况下都会使用过期值作答。在安全套件中，陷阱条件下低于无记忆基线的危害（Δ_mem）受模型能力门控，一旦过期笔记被伪装成最新内容，规模更大的模型反而崩溃得最严重。在一个2×2×2×2的因子实验中，探究了哪个特征触发了过度信任……（原文摘要在此处截断）

    arXiv:2609.01852v1 Announce Type: new  Abstract: Persistent memory supports personalized agents, but a stale stored fact can override current authoritative evidence without warning. We study when this harm begins as model capability changes. We evaluate a frozen, closed-set, action-scored benchmark with 2 suites that represent 2 different meanings of "no memory" (a Benefit suite, unsolvable without the stored fact, and a Safety suite, in which an authoritative tool always holds the correct value), on a same-family model-size series (Qwen3 0.6/1.7/4/8B). The Memory Trust Gap reflects over-trust rather than confusion. In the Benefit suite, models answer with the stale value 0.92-1.00 of the time at every scale. In the Safety suite, harm below the no-memory baseline under the trap conditions ($\Delta_{\mathrm{mem}}$) is capability-gated, with the larger models collapsing most once a stale note is made to look current. In a $2\times2\times2\times2$ factorial, which feature triggers over-tr
    
[^118]: 阿拉伯语安全对齐作为选择性拒绝：SFT、DPO 与护栏校准的实证研究

    Arabic Safety Alignment as Selective Refusal: An Empirical Study of SFT, DPO, and Guard Calibration

    [https://arxiv.org/abs/2608.29378](https://arxiv.org/abs/2608.29378)

    该研究提出用良性拒绝率 B 和有害提示拒绝率 H 双指标评估阿拉伯语大模型安全对齐，发现仅拒绝式 SFT 会坍缩为全面拒绝，而精选混合 SFT 配置能在低良性拒绝率下实现 90% 以上的有害提示拒绝，且 DPO 与推理护栏的效果因模型而异、并非统一提升。

    

    阿拉伯语大语言模型必须拒绝有害提示，同时不过度拒绝良性或敏感提示，然而单一的拒绝率指标掩盖了这一权衡。我们采用良性拒绝率 B 和有害提示拒绝率 H 来评估该权衡，其中 H 衡量的是拒绝行为而非对有害请求的顺从。在五个支持阿拉伯语的模型上，基于完整人工撰写的 AraSafe 数据集进行的 130 次运行中，仅拒绝式监督微调（SFT）会坍缩为全面拒绝，而精选的混合 SFT 配置能够在 B = 14% 至 23% 的条件下达到 H = 90% 至 93%；四个精选配置在全部三次运行中均超过 H = 90% 的目标，而 Fanar 在三次运行中有两次达标。直接偏好优化（DPO）和推理时护栏对不同模型的 B 和 H 的影响各不相同，而非作为统一的改进手段。在一项盲评的 300 条回复审计中，标注者二元拒绝判断的一致性为 89.0%（kappa = 0.78）；Qwen3Guard 和 Aya Expanse 32B 分别达到 88.7% 和 91.0% 的准确率，而

    arXiv:2608.29378v1 Announce Type: cross  Abstract: Arabic large language models must refuse harmful prompts without over-refusing benign or sensitive prompts, yet a single refusal rate hides this trade-off. We evaluate it using benign refusal B and harmful-prompt refusal H, where H measures refusal rather than harmful compliance. Across five Arabic-capable models and 130 runs on the full human-written AraSafe set, refusal-only supervised fine-tuning (SFT) collapses toward blanket refusal, whereas selected mixed-SFT configurations reach H = 90% to 93% at B = 14% to 23%; four selected configurations exceed the H = 90% target in all three runs, while Fanar does so in two of three. Direct Preference Optimization (DPO) and inference guards change B and H differently across models rather than acting as uniform upgrades. In a blinded 300-response audit, annotator binary-refusal agreement is 89.0% (kappa = 0.78); Qwen3Guard and Aya Expanse 32B reach 88.7% and 91.0% accuracy, respectively, with
    
[^119]: 迈向更安全的RAG：只有具备系统2思考能力的代理才能访问不可信文档

    Towards Safer RAG: Only Agents Capable of System 2 Thinking may Access Untrusted Documents

    [https://arxiv.org/abs/2608.17153](https://arxiv.org/abs/2608.17153)

    本文提出一种新的安全原则，即仅允许具备系统2推理能力的代理访问不可信文档，以减少RAG系统中的知识投毒攻击影响，并引入新指标量化检测与影响间的差异。

    

    检索增强生成（RAG）显著提升了大型语言模型（LLMs）的性能，但这些系统仍然容易受到知识投毒攻击，即检索文档中的错误信息可能影响模型的最终输出。值得注意的是，LLM可能正确检测到文档包含错误信息，却仍受其影响。先前的研究通过“隔离原则”（Cordon Principle）解决了这一漏洞，该原则防止负责最终答案合成的模型直接访问原始证据。尽管有效，但这种严格隔离可能带来大量计算开销。在本工作中，我们提出了一种精细化的安全原则：只有具备深思熟虑的系统2推理能力的代理才能访问不可信文档。为评估这一原则，我们引入了新指标，用于量化错误信息检测与下游影响之间的差异。我们进行了实验...

    arXiv:2608.17153v1 Announce Type: new  Abstract: Retrieval-Augmented Generation (RAG) has significantly enhanced the performance of large language models (LLMs), yet these systems remain vulnerable to knowledge-poisoning attacks, in which misinformation in retrieved documents can influence the model's final outputs. Notably, an LLM may correctly detect that a document contains incorrect information while nevertheless being influenced by it. Prior work has addressed this vulnerability through the Cordon Principle, which prevents models responsible for final answer synthesis from directly accessing raw evidence. Although effective, this strict isolation can introduce substantial computational overhead. In this work, we propose a refined security principle: only agents capable of deliberative System 2 reasoning may access untrusted documents. To evaluate this principle, we introduce novel metrics that quantify the discrepancy between misinformation detection and downstream influence. We t
    
[^120]: 不对称的话语同质化与共享语言技术：来自Reddit的证据

    Asymmetric Discourse Homogenization and Shared Language Technology: Evidence from Reddit

    [https://arxiv.org/abs/2608.13674](https://arxiv.org/abs/2608.13674)

    本文通过Reddit数据发现，2022年底后保守派政治话语的同质化趋势不对称地加剧，而进步派无此变化，且该效应由累积AI语言模型影响驱动而非单一事件。

    

    摘要：arXiv:2608.13674v1 公告类型：交叉 摘要：我利用2019年至2025年间来自两个跨党派论坛的600万条Reddit评论，记录了政治话语先前多元化趋势中一个意识形态上不对称的断裂，该断裂大约出现在2022年底。保守派用户的先前多元化轨迹被打断；进步派用户则没有表现出类似的变化。这种不对称性在多种估计策略（中断时间序列、双重差分、回归不连续、倾向得分匹配）和时间聚合方式下均保持一致。一项针对2,377个候选截止日期的日频排列检验显示，ChatGPT阈值产生了一个不显著的估计值（第49.8百分位）：这种转变是逐渐积累的，而非在单一日期发生断裂。一个追踪七次模型发布中AI暴露情况的连续累积LLM指数，在消除二元估计的二次趋势规范下仍然显著。一项留驻者分析缩小了机制范围：同质化效应消失……

    arXiv:2608.13674v1 Announce Type: cross  Abstract: I document an ideologically asymmetric break in the pre-existing diversification trend of political discourse, emerging around late 2022, using 6 million Reddit comments from two cross-partisan forums, 2019-2025. Conservative users experienced an interruption of their prior diversification trajectory; progressive users showed no comparable change. The asymmetry is consistent across estimation strategies (ITS, DiD, RDiT, propensity-score matching) and temporal aggregations. A daily-frequency permutation test over 2,377 candidate cutoff dates shows the ChatGPT threshold produces an unremarkable estimate (49.8th percentile): the shift builds gradually instead of breaking at a single date. A continuous cumulative LLM index, tracking AI exposure across seven model releases, remains significant under a quadratic trend specification that eliminates the binary estimate. A stayer analysis narrows the mechanism: the homogenization effect disappe
    
[^121]: 查询时机导致大语言模型与人类之间产生相反的位置偏差

    Query Timing Produces Opposite Positional Biases Between LLMs and Humans

    [https://arxiv.org/abs/2608.12387](https://arxiv.org/abs/2608.12387)

    本研究揭示了LLMs在位置偏差上表现出与人类相反的模式，且新模型中的偏差更为显著，表明查询时机对模型评估机制有重要影响。

    

    位置偏差，如近因效应和首因效应，已在大型语言模型（LLMs）中被记录，然而这些模型进行评估的潜在机制仍知之甚少。在人类对证据的回应中，首因和近因偏差均有观察到，但近期研究表明，听者更新其信念的时间点——是在证据呈现过程中还是仅在结束时——会影响这些效应的存在。我们研究了类似现象是否在LLMs中成立，发现其与人类行为存在分歧。这些偏差在较新模型中相较于早期模型更为加剧。

    arXiv:2608.12387v1 Announce Type: cross  Abstract: Positional biases such as recency and primacy effects have been documented in large language models (LLMs), yet the underlying mechanism by which these models make their evaluations remains poorly understood. Both primacy and recency biases have been observed in human judgments in response to evidence, but recent work suggest that \emph{when} the listener updates their beliefs -- during the presentation of evidence or only at the end -- influences the presence of such effects. We investigate whether a similar phenomenon holds for LLMs, finding divergence from human behavior. These biases are more exacerbated in newer models compared to their predecessors.
    
[^122]: 一个冻结的模拟器是不够的：多智能体强化学习中的模拟器崩溃

    One Frozen Simulator Is Not Enough: Simulator Collapse in Multi-Agent RL

    [https://arxiv.org/abs/2608.12253](https://arxiv.org/abs/2608.12253)

    该论文揭示了多智能体强化学习中使用单一冻结模拟器会导致策略泛化失败，并提出推理时的言语化采样和训练时的协同训练两种方法来解决模拟器崩溃问题。

    

    用于人机交互的多智能体强化学习通常依赖单一大型语言模型来模拟用户行为。我们表明这种方法系统性无法泛化，并将失败追溯到模拟器崩溃：由于模拟器语言模型存在模式崩溃，针对其训练的策略会过度拟合到利用模拟器主导模式的狭窄策略，而这种策略在未见过的模拟器和真实用户中迁移效果不佳。我们在理论上形式化了这种崩溃，并提出了两种互补的解决方案，一种在推理时，一种在训练时。推理时的解决方案，即言语化采样，通过从言语化响应分布中采样来拓宽模拟器的行为，减少模式崩溃。训练时的解决方案，即协同训练，与一组可训练的模拟器种群联合优化策略，防止其过度拟合任何单一模拟器的模式。我们进行了验证...

    arXiv:2608.12253v1 Announce Type: cross  Abstract: Multi-agent reinforcement learning for human-AI interaction typically relies on a single large language model to simulate user behavior. We show that this approach systematically fails to generalize, and trace the failure to simulator collapse: because the simulator LLM is mode-collapsed, an LLM policy trained against it overfits to narrow strategies that exploit the simulator's dominant mode, and such a policy transfers poorly to unseen simulators and real users. We formalize this collapse theoretically and propose two complementary solutions, one at inference time and one at training time. The inference-time solution, Verbalized Sampling, broadens the simulator's behavior by sampling from a verbalized response distribution, reducing mode collapse. The training-time solution, Co-Training, jointly optimizes the policy against a population of trainable simulators, preventing it from overfitting to any single simulator's mode. We validat
    
[^123]: 面向社会模拟的语言模型角色引导

    Role Steering of Language Models for Social Simulations

    [https://arxiv.org/abs/2608.00023](https://arxiv.org/abs/2608.00023)

    该论文提出了一种激活引导筛选工作流和新方法 CastVectors，用于为语言模型智能体注入可验证的角色条件化行为，在社会模拟中实现了显著更高的角色档案对齐度（63.2 对比 41.1），同时保持了词汇多样性。

    

    由语言模型智能体构建的社会模拟需要在将智能体放入模拟人群之前能够被验证的角色条件化行为。我们提出了一种面向角色条件化智能体的激活引导筛选工作流：定义角色档案、提取角色特定方向、扫描四个引导系数、评估角色档案对齐度，并对每个候选配置进行通过或标记。在 OLMo-3-7B-Instruct 上，我们将该工作流应用于包含 275 个角色的混合角色清单、228 个与角色无关的问题、由 GPT-4.1-mini 提示生成的角色参考以及 GPT-4.1-mini 评判者。我们还提出了 CastVectors，其在评判的角色档案对齐度上优于此前人格向量工作中的助手轴方向控制方法，在测试网格上的平均总分为 63.2 对比 41.1。CastVectors 还保持了较高的词汇多样性，而控制方法在较大系数下词汇多样性急剧下降。角色级筛选是……（摘要原文在此处截断）

    arXiv:2608.00023v3 Announce Type: replace-cross  Abstract: Social simulations built from language-model agents need role-conditioned behavior that can be checked before agents are placed into a simulated population. We introduce an activation-steering screening workflow for role-conditioned agents: define a role profile, extract a role-specific direction, sweep four steering coefficients, evaluate role-profile alignment, and pass or flag each candidate configuration. On OLMo-3-7B-Instruct, we apply the workflow to a mixed 275-role inventory with 228 role-agnostic questions, GPT-4.1-mini prompted role references, and GPT-4.1-mini judges. We also introduce CastVectors, which receive higher judged role-profile alignment than an assistant-axis directional control from prior persona-vector work, with mean overall scores of 63.2 versus 41.1 across the tested grid. They also preserve high lexical diversity, while the control drops sharply at larger coefficients. The role-level screen is the m
    
[^124]: RecHarness：一种用于自演化推荐系统的多臂老虎机路由智能体框架

    RecHarness: A Bandit-Routed Agentic Harness for Self-Evolving Recommender Systems

    [https://arxiv.org/abs/2607.29241](https://arxiv.org/abs/2607.29241)

    提出RecHarness框架，通过将优化过程解耦为“多臂老虎机路由器选择修改方向+LLM生成具体优化假设与代码修改”两个步骤，并借助跳跃盆地机制在局部优化停滞时触发结构性跳跃，实现推荐模型的自演化自动优化。

    

    优化推荐模型仍然严重依赖于对架构、目标和训练策略的人工迭代。虽然基于大语言模型（LLM）的智能体可以自动化这一试错过程，但依赖LLM同时选择修改方向和制定具体假设，在有限的试验预算下可能导致不稳定的搜索。受上述挑战的启发，我们提出了RecHarness，一个用于自动化推荐模型优化的多臂老虎机路由智能体框架。RecHarness将优化过程解耦为两个步骤：多臂老虎机路由器根据历史验证反馈选择下一个修改方向，而LLM则在所选方向内生成具体的优化假设和可执行的代码修改。为了维持长期探索，RecHarness使用跳跃盆地（jump-basin）机制，在局部编辑陷入停滞时激活结构跳跃臂。在多个推荐任务、数据集和……（摘要截断）

    arXiv:2607.29241v2 Announce Type: replace-cross  Abstract: Optimizing recommender models remains heavily reliant on manual iteration over architectures, objectives, and training strategies. While LLM-based agents can automate this trial-and-error process, relying on LLMs to both select modification directions and formulate concrete hypotheses can lead to unstable search under limited trial budgets. Inspired by the above challenge, we propose RecHarness, a Bandit-Routed Agentic Harness for automated recommender model optimization. RecHarness decouples the optimization process into two steps: a bandit router selects the next modification direction according to historical validation feedback, while the LLM generates a concrete optimization hypothesis and executable code edit within the selected direction. To sustain long-horizon exploration, RecHarness uses a jump-basin mechanism to activate a structural-jump arm when local edits stagnate. Across multiple recommendation tasks, datasets, a
    
[^125]: 无需坐标或区域标签的视觉文档理解证据归因

    Evidence Attribution in Visual Document Understanding without Coordinates or Region Labels

    [https://arxiv.org/abs/2607.24651](https://arxiv.org/abs/2607.24651)

    该研究提出以“模型逐字引用证据、由多模态检索器定位出处”的语言接口替代传统的坐标边界框接口，表明视觉文档理解中的归因幻觉部分源于坐标表达能力的局限。

    

    可靠的视觉文档理解要求模型将每个答案归因于支持它的证据区域。近期的基准测试和系统通过坐标接口来表达这一步骤：模型输出标记文档中证据区域的边界框坐标。在此接口下，视觉语言模型即使答案正确，也常常无法识别正确的区域，这种失败被称为“归因幻觉”。我们开展了一项研究，探讨这种失败是否部分受限于模型通过坐标所能表达的内容。在一个经过验证的双语 CiteVQA 子集上，我们将坐标接口与语言接口进行对比：在语言接口中，模型仅输出文本，逐字引用其证据，随后由多模态检索器返回每条引用的位置，即布局解析器所提出的页面区域（表格和图片则通过其标题或注释进行引用）……（摘要在此处截断）

    arXiv:2607.24651v2 Announce Type: replace-cross  Abstract: Reliable visual document understanding requires a model to attribute each answer to the evidence regions that support it. Recent benchmarks and systems express this step through a coordinate interface: the model outputs the coordinates of bounding boxes that mark the evidence regions in the document. Under this interface, vision-language models often fail to identify the right regions even when the answer is correct, a failure known as Attribution Hallucination. We present a study that investigates whether this failure is partially limited by what the model can express through coordinates. On a verified bilingual CiteVQA subset, we compare the coordinate interface with a language interface in which the model outputs only text, quoting its evidence verbatim, and a multimodal retriever returns the location of each quote as a page region proposed by a layout parser (tables and figures are quoted through their captions or notes); t
    
[^126]: 在智能体软件开发中使用第三方API路由器的权衡在哪里？

    Where Is the Tradeoff in Using Third-Party API Routers for Agentic Software Development?

    [https://arxiv.org/abs/2607.23624](https://arxiv.org/abs/2607.23624)

    该论文首次实证研究了编码智能体中第三方API路由器的路由器端注入风险，通过四个隐蔽程度递增的干预级别（如响应替换）揭示其可对软件开发任务造成真实且难以检测的影响，而现有客户端权限机制难以有效防御。

    

    第三方API路由器已成为一种常见的中间层，统一了对日益多样化的LLM提供商的访问。在编码智能体工作流中，高自主性操作被广泛采用，因为它减少了交互开销。因此，位于智能体与上游提供商之间的第三方API路由器不可避免地占据了可信路径。它可以检查和修改每一个请求与响应，然而目前没有任何机制来验证提供商的输出与智能体最终执行的仓库级操作之间的一致性。因此，客户端权限机制在实践中可能变得无效。这种控制缺口是否会对软件开发任务产生真实的、难以检测的影响，此前尚未得到实证测量。在本文中，我们对编码智能体中的路由器端注入进行了实证研究，考察了四个隐蔽程度递增的干预级别：响应替换（L1）、响应追加……

    arXiv:2607.23624v3 Announce Type: replace-cross  Abstract: Third-party API routers have become a common layer that unifies access across increasingly diverse LLM providers. In coding-agent workflows, high-autonomy operation is widely adopted because it reduces interaction overhead. As a result, a third-party API router, which sits between the agent and the upstream provider, inevitably occupies the trusted path. It can inspect and modify every request and response, yet no mechanism verifies alignment between the provider's output and the repository-level actions ultimately executed by the agent. Consequently, client-side permission mechanisms may become ineffective in practice. Whether this control gap produces real, hard-to-detect effects on software development tasks remains empirically unmeasured. In this paper, we conduct an empirical study of router-side injection in coding agents, examining four intervention levels of increasing subtlety: Response Substitution (L1), Response Appe
    
[^127]: 强化学习后训练构建组合推理策略

    RL Post-Training Builds Compositional Reasoning Strategies

    [https://arxiv.org/abs/2607.07646](https://arxiv.org/abs/2607.07646)

    强化学习后训练并非只是放大预训练模型已有的基础技能，而是通过“先强化基础规约、再发现顺序与并行组合程序”的分阶段组合机制，将基础技能重组为新的更高层级推理策略，从而解决预训练模型几乎无法解决的问题。

    

    强化学习（RL）后训练仅仅是放大基础模型中已经潜在存在的基础技能，还是能够将基础技能组合成新的更高层级策略？我们在一个完全可观测的重写语法环境中研究这个问题，在该环境中预训练分布是已知的，且每一个生成的重写都可以被审计。Transformer模型首先在基础符号重写链上进行预训练，然后在一个基于Trace的推理任务上进行后训练，仅使用二元的最终答案奖励。强化学习能够解决那些预训练模型即使在更大的采样预算下也极少能解决的保留测试问题，而拒绝采样微调虽然早期有所提升但随后趋于停滞。Trace分析显示，强化学习通过一种分阶段的组合机制重新组织基础能力：它首先强化基础规约，然后发现有效的组合程序，包括将有序的基础收缩链折叠起来的顺序组合，以及并行组合……

    arXiv:2607.07646v2 Announce Type: replace  Abstract: Does RL post-training merely amplify primitive skills already latent in a base model, or can it compose primitive skills into new higher-level strategies? We study this question in a fully observable rewrite-grammar environment where the pretraining distribution is known and every generated rewrite can be audited. A Transformer is pretrained on primitive symbol-rewrite chains and post-trained on a Trace-based reasoning task with only a binary final-answer reward. RL solves held-out problems that remain rarely solved by the pretrained model even under much larger sampling budgets, while rejection fine-tuning improves early but plateaus. Trace analysis shows that RL reorganizes primitive competence through a phased compositional mechanism: it first strengthens primitive reductions, then discovers valid composed procedures. These include sequential compositions, which collapse ordered chains of primitive contractions, and parallel compo
    
[^128]: 局部化LoRA-MoE：具有自适应路由的低秩专家混合

    Localized LoRA-MoE: Mixture of Low-Rank Experts With Adaptive Routing

    [https://arxiv.org/abs/2607.05114](https://arxiv.org/abs/2607.05114)

    提出Localized LoRA-MoE框架，将局部空间分块与上下文条件自适应专家路由相结合，提供了集中式块级路由和去中心化单元级路由两种架构变体，以解决传统PEFT方法中的梯度干扰和静态拓扑无法适应上下文切换的问题。

    

    参数高效微调（PEFT）方法（如LoRA）通过单一低秩更新来适配大型模型。这类更新容易受到梯度干扰的影响：当训练流中交织着多个操作上下文时，对共享适配器权重的相互冲突的更新会使适配器坍缩成一个缺乏专门化的平均值。虽然近期的方法引入了块级隔离来缓解这一问题，但其拓扑结构是静态的，无法适应上下文切换或局部传感器退化。在这项工作中，我们提出了Localized LoRA-MoE，一个将局部空间分块与上下文条件专家路由相结合的框架。我们提出并评估了两种架构变体：Block-Wise LoRA-MoE（集中式路由），它通过单一的上下文条件门控来调制整个结构网格；以及Cell-Wise LoRA-MoE（去中心化路由），它为结构网格的每个坐标单元分配一个独立的门控。

    arXiv:2607.05114v2 Announce Type: replace-cross  Abstract: Parameter-efficient fine-tuning (PEFT) methods such as LoRA adapt large models through a single low-rank update. Such updates are susceptible to gradient interference: when a training stream interleaves multiple operational contexts, conflicting updates to shared adapter weights collapse the adapter into an unspecialized average. While recent methods introduce block-wise isolation to remedy this issue, their topologies are static and cannot adapt to context switching or localized sensor degradation. In this work, we introduce Localized LoRA-MoE, a framework that combines localized spatial blocking with context-conditioned expert routing. We propose and evaluate two architectural variants: Block-Wise LoRA-MoE (centralized routing), which modulates the entire structural grid through a single context-conditioned gate, and Cell-Wise LoRA-MoE (decentralized routing), which assigns an independent gate to every coordinate cell of the 
    
[^129]: 语言-能源鸿沟：测量多语言大语言模型推理的能源成本

    The Language-Energy Divide: Measuring Energy Costs of Multilingual LLM Inference

    [https://arxiv.org/abs/2606.21869](https://arxiv.org/abs/2606.21869)

    该论文系统性测量了大语言模型多语言推理的能耗，揭示了不同语言间高达179倍的能耗差异，并发现高能耗语言同时伴随最低的任务准确率，存在“成本+性能”双重惩罚现象。

    

    大语言模型（LLM）日益被部署于多语言环境中，然而跨不同语言服务这些模型的能源成本仍然鲜为人知。我们利用 ML.Energy 框架对不同语言的推理能耗进行了系统性研究。我们发现了显著的差异：不同语言之间每个输出 token 的能耗差异高达 8.3 倍，而在固定的请求集合下，最便宜的语言（英语，17.6 kJ）与最昂贵的语言（普什图语，3,147 kJ）之间的总能耗差异高达 179 倍。我们的分析表明，这种差异由两个叠加因素驱动：（1）使用复杂或罕见文字系统的语言具有更高的每 token 能耗成本；（2）低资源语言会生成更多的 token。此外，我们还发现了一种“成本+性能”的双重惩罚：能耗最高的语言往往任务准确率也最低。

    arXiv:2606.21869v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) are increasingly deployed in multilingual settings, yet the energy costs of serving these models across different languages remain poorly understood. We present a systematic study of inference energy consumption across languages with ML.Energy framework (Chung et al., 2026). We find striking disparities: energy consumption per output token varies by up to 8.3 times across languages, while total energy for a fixed set of requests varies by up to 179 times between the cheapest (English, 17.6 kJ) and the most expensive (Pashto, 3,147 kJ) languages. Our analysis shows that this disparity is driven by two compounding factors: (1) higher per-token energy costs for languages using complex or rare scripts, and (2) more tokens generated for low-resource languages. Moreover, we find a double cost + performance penalty: languages with the highest energy footprints also tend to achieve the lowest task accuracy.
    
[^130]: 深海之韵：从点击声波形到二阶序列依赖的抹香鲸尾音两级声学组织

    Rhythm of the Deep: Two-tier acoustic organization of sperm-whale codas from click waveforms to second-order sequence dependence

    [https://arxiv.org/abs/2606.16084](https://arxiv.org/abs/2606.16084)

    该研究发现抹香鲸尾音的声学组织具有两级嵌套结构：尾音内部反复出现的点击声波形差异与点击间隔时间互补，而尾音之间反复出现的整体尾音形式本身携带序列依赖性。

    

    抹香鲸尾音（codas）通常以点击声数量和点击间隔（ICI）来刻画，但其组成点击声波形中反复出现的差异一直未能得到解释。本研究检验了声学组织是否在两个尺度上呈嵌套结构：在尾音内部，反复出现的点击声波形差异可能与ICI时间信息互补；在尾音之间，反复出现的整体尾音形式本身可能携带序列依赖性。研究从1,483个尾音中识别出候选的反复出现的点击声和整体尾音分组，且未预先指定波形类别，随后通过原始采样率的频谱/包络测量、精确的干扰因素匹配、留出的时序对比以及序列控制进行评估。在第一层级，在匹配的日期、社会单元、个体和采样率层级内，反复出现的点击声波形组在频谱斜率、带宽、平坦度、高/低频带能量和包络结构上均存在差异。其组成增加了关于……（摘要原文在此处截断）

    arXiv:2606.16084v3 Announce Type: replace  Abstract: Sperm-whale codas are conventionally characterized by click count and inter-click intervals (ICIs), leaving recurring differences in constituent click waveforms unresolved. This study tests whether acoustic organization is nested across two scales: within codas, where recurring click-waveform differences may complement ICI timing, and across codas, where recurring whole-coda forms may themselves carry sequence dependence. Candidate recurring click and whole-coda groupings were identified from 1,483 codas without prespecifying waveform categories, then evaluated with native-rate spectral/envelope measurements, exact nuisance matching, held-out timing contrasts, and sequence controls. At the first tier, recurring click-waveform groups differed in spectral slope, bandwidth, flatness, high/low-band energy, and envelope structure within matched date, social unit, individual, and sample-rate strata. Their composition added information abou
    
[^131]: ICA Lens：非高斯性揭示语言模型中的可解释方向

    ICA Lens: Non-Gaussianity Reveals Interpretable Directions in Language Models

    [https://arxiv.org/abs/2606.11722](https://arxiv.org/abs/2606.11722)

    该论文发现语言模型激活中可解释方向具有非高斯投影分布这一统计特征，并提出可以通过直接搜索非高斯方向来发现可解释特征，而无需依赖稀疏重建等间接学习目标。

    

    在语言模型激活中寻找人类可解释的方向，对于理解和控制模型行为至关重要。这些方向通常通过间接的学习目标（如稀疏重建）来发现，但目前尚不清楚是何种统计结构区分了可解释方向，以及这种结构能否直接用于特征发现。我们发现非高斯性提供了这样一种特征标志：自然文本残差流激活沿稀疏自编码器（SAE）解码器方向的投影分布，明显比沿随机方向的投影分布更加非高斯，其中大多数上下文形成一个集中的背景，而重复出现的模式往往占据分布的一个尾部。这引出了一个自然的问题：如果可解释方向往往具有非高斯的投影分布，我们能否通过直接搜索非高斯方向来发现它们？

    arXiv:2606.11722v3 Announce Type: replace-cross  Abstract: Finding human-interpretable directions in language-model activations is important for understanding and controlling model behavior. These directions are usually discovered through indirect learning objectives such as sparse reconstruction, but it remains unclear what existing statistical structure distinguishes interpretable directions and whether that structure can be used directly for feature discovery. We find that non-Gaussianity provides such a signature: projection distributions of natural-text residual-stream activations along sparse autoencoder (SAE) decoder directions are substantially more non-Gaussian than those along random directions, with most contexts forming a concentrated background and recurring patterns often occupying one tail. This raises a natural question: if interpretable directions tend to have non-Gaussian projection distributions, can we discover them by searching for non-Gaussian directions directly 
    
[^132]: 超越检索：学习紧凑的用户表示以实现可扩展的大语言模型个性化

    Beyond Retrieval: Learning Compact User Representations for Scalable LLM Personalization

    [https://arxiv.org/abs/2606.04547](https://arxiv.org/abs/2606.04547)

    提出TAP-PER框架，通过时序注意力前缀嵌入将用户偏好编码为紧凑的可学习表示，摆脱了检索式个性化对检索质量的依赖以及参数式个性化随用户规模增长的高昂存储成本，实现可扩展的大语言模型个性化。

    

    个性化大语言模型需要在使模型行为适应个体用户的同时，保持鲁棒性和部署规模上的效率。现有方法通常在输入层面对大语言模型进行个性化，即通过检索用户历史或构建用户画像提示，或者在参数层面对其进行个性化，即为每个用户维护特定的参数高效模块。前者使个性化效果依赖于检索质量和提示设计，而后者会产生随用户规模增长而增加的存储和维护成本。为了解决这些局限，我们提出了TAP-PER（Temporal Attentive Prefix for PERsonalization，时序注意力前缀个性化），这是一个基于前缀的框架，它将用户偏好编码为可学习的表示，避免了将用户历史序列化到提示中，并用轻量级的用户状态前缀嵌入取代了繁重的每用户适配器模块。受个性化推荐系统的启发，TAP-PER将用户……（原文摘要在此处截断）

    arXiv:2606.04547v3 Announce Type: replace-cross  Abstract: Personalizing large language models requires adapting model behavior to individual users while preserving robustness and deployment-scale efficiency. Existing approaches typically personalize LLMs either at the input level, by retrieving user histories or constructing profile prompts, or at the parameter level, by maintaining user-specific parameter-efficient modules. The former makes personalization sensitive to retrieval quality and prompt design, whereas the latter incurs storage and maintenance costs that grow with the user population. To address these limitations, we propose TAP-PER (Temporal Attentive Prefix for PERsonalization), a prefix-based framework that encodes user preferences as learnable representations, avoiding the serialization of user histories into prompts and replacing heavy per-user adapters with lightweight user-state prefix embeddings. Inspired by personalized recommendation systems, TAP-PER decomposes u
    
[^133]: 我遗漏了什么？将问答作为隐状态探测方法

    What Am I Missing? Question-Answering as Hidden State Probing

    [https://arxiv.org/abs/2605.31561](https://arxiv.org/abs/2605.31561)

    本文提出让学生模型向教师模型提问作为推理时干预手段，并通过在提问前后学生隐状态上训练探测器，证明在教师给出答案之前即可预测推理轨迹的最终正确性，揭示了提问能够暴露模型隐状态中的自我诊断信号。

    

    arXiv:2605.31561v2 公告类型：替换 摘要：大语言模型（LLM）测试时推理的研究因其对众多任务的深远影响而日益受到关注，早期工作可追溯到思维链推理等方法。然而，测试时推理过程背后的基本机制仍未得到充分研究：在相同的输入提示下，甚至在相同的部分解答的情况下，LLM在多次采样时可能产生不同的答案。在本文中，我们利用提问作为一种推理时干预手段，用以揭示模型隐状态中的信息。具体而言，我们提出了一个学生-教师设置，其中学生向教师提问。我们在学生提问前后的隐状态上训练一个探测器，发现该探测器能够预测推理轨迹的最终正确性，甚至在教师生成答案之前就能做到。这表明，提问过程中发生的自我诊断蕴含着有意义的信号。

    arXiv:2605.31561v2 Announce Type: replace  Abstract: The study of test-time reasoning in large language models (LLMs) has attracted increasing interests due to its profound impact on many tasks, with early work tracing back to approaches such as chain-of-thought reasoning. The basic mechanism underlying the test-time reasoning process, however, is still under-studied: from the same input prompt, and even the same partial solution, LLMs can produce varied answers if sampled multiple times. In this paper, we leverage question-asking as an inference-time intervention that articulates information about the model's hidden state. Specifically, we present a student-teacher setting where a student asks questions to a teacher. We train a probe on the student's hidden state before and after asking a question and find it is predictive of the trajectory's final correctness, even before generating the teacher's answer. This suggests there is a meaningful signal from the self-diagnosis that occurs d
    
[^134]: 超越双语迁移：指令微调中的多语言语码转换

    Beyond Bilingual Transfer: Multilingual Code-Switching in Instruction Tuning

    [https://arxiv.org/abs/2605.29414](https://arxiv.org/abs/2605.29414)

    该研究首次探索了在指令微调中混合英语、日语、韩语和中文四种语言的多语言语码转换数据构建方法，实验证明句子级多语言语码转换能持续提升LLM的多语言理解能力，其效果超越传统仅限双语迁移的方案。

    

    近期研究表明，语码转换数据（CSD），即在同一上下文中混合多种语言，能够改善大语言模型（LLM）的跨语言迁移和多语言对齐能力。然而，现有针对LLM的合成语码转换研究主要聚焦于英语与目标语言之间的双语迁移，在每个样本中同时混合三种或更多语言的指令微调设置在很大程度上尚未被探索。在这项工作中，我们研究了涵盖英语、日语、韩语和中文四种语言的多语言语码转换指令微调。我们在Belebele和MMLU数据集上评估多语言理解能力。实验表明，简单的句子级多语言CSD在Gemma-2-2B模型的双语、三语和四语所有设置中均能持续提升四种语言的平均准确率，这表明多语言语码转换的有效性可以超越双语场景。

    arXiv:2605.29414v2 Announce Type: replace-cross  Abstract: Recent studies have shown that code-switching data (CSD), in which multiple languages are mixed within the same context, can improve cross-lingual transfer and multilingual alignment in large language models (LLMs). Recent studies on synthetic code-switching for LLMs have primarily focused on bilingual transfer between English and a target language, leaving instruction-tuning settings that jointly mix three or more languages within each example largely unexplored. In this work, we investigate multilingual code-switching instruction tuning across four languages: English, Japanese, Korean, and Chinese. We evaluate multilingual understanding on Belebele and MMLU. Our experiments show that simple sentence-level multilingual CSD consistently improves the four-language average accuracy across all bilingual, trilingual, and quadrilingual settings for Gemma-2-2B, indicating that multilingual code-switching can be effective beyond bilin
    
[^135]: 检索增强生成的上下文内优化：梯度下降视角

    In-Context Optimization for Retrieval-Augmented Generation: A Gradient-Descent Perspective

    [https://arxiv.org/abs/2605.26356](https://arxiv.org/abs/2605.26356)

    该论文将检索增强生成（RAG）形式化为上下文内优化过程，证明单层线性自注意力可在统一的线性化RAG目标上执行一步梯度下降，并以此理论洞察指导查询与检索证据之间交互方式的设计。

    

    上下文学习近期被与线性自注意力模型中的隐式梯度下降联系起来，表明上下文可以在前向传播过程中引发更新。检索增强生成（RAG）同样依赖上下文，但检索到的文档通常被视为静态证据，而非用于适配的信号。我们将RAG作为一个上下文内优化过程进行研究。首先，我们证明单层线性自注意力可以在一个统一的线性化RAG目标上实现一步梯度下降，该目标同时涵盖基于投影和点积两种检索接口。这给出了检索增强预测与上下文内优化完全重合的精确情形。我们并非将这一结果作为LLM计算的字面模型，而是将其用作调整查询与检索证据之间交互方式的指导。随后，我们检验了这种对应关系的边界：它在受控的线性扩展下保持稳定，但会随特征……（原文摘要在此处截断）

    arXiv:2605.26356v2 Announce Type: replace  Abstract: In-context learning has recently been linked to implicit gradient descent in linear self-attention models, suggesting that context can induce a forward-pass update. Retrieval-augmented generation (RAG) also relies on context, but retrieved documents are usually treated as static evidence rather than signals for adaptation. We study RAG as an in-context optimization process. First, we show that one linear self-attention layer can implement one gradient-descent step on a unified linearized RAG objective covering both projection-based and dot-product retrieval interfaces. This gives an exact regime where retrieval-augmented prediction and in-context optimization coincide. We use this result not as a literal model of LLM computation, but as a guide for adapting the interaction between queries and retrieved evidence. We then test the boundary of this correspondence: it remains stable under controlled linear extensions, but becomes feature
    
[^136]: RICE-PO：将检索交互转化为推理智能体的信用信号

    RICE-PO: Turning Retrieval Interactions into Credit Signals for Reasoning Agents

    [https://arxiv.org/abs/2605.26352](https://arxiv.org/abs/2605.26352)

    RICE-PO 是一个无评论家的策略优化框架，通过将高不确定性可执行动作作为锚点、评估局部反事实分支并将信用有条件地传播给潜在推理步骤，把检索交互转化为局部化信用信号，从而解决交互式推理检索智能体训练中的信用分配难题。

    

    检索正日益从一次性匹配转向交互式推理，其中语言智能体迭代地检查证据、重新构造查询并再次搜索。训练此类智能体带来了信用分配方面的挑战：查询或摘要等可执行动作可以由检索器直接评估，而潜在的推理步骤却不可直接观察，只能通过影响未来的可执行动作来发挥作用。这种不对称性使得基于最终结果的奖励分配不可靠，因为相同的最终奖励可能会错误地归功于那些实际上并未促成检索成功的推理步骤。我们提出 RICE-PO，这是一个无评论家（critic-free）的策略优化框架，能够将检索交互转化为局部化的学习信号。RICE-PO 选择高不确定性的可执行动作作为锚点，利用检索指标评估局部反事实分支，并且仅在推理对动作的影响足够显著时，才将信用传播到潜在的推理步骤。

    arXiv:2605.26352v4 Announce Type: replace  Abstract: Retrieval is increasingly moving from one-shot matching toward interactive reasoning, where language agents iteratively inspect evidence, reformulate queries, and search again. Training such agents raises a credit-assignment challenge: executable actions such as queries or summaries can be directly evaluated by the retriever, while latent reasoning steps are not directly observable and only affect future executable actions. This asymmetry makes outcome-level reward assignment unreliable, as the same final reward may credit reasoning steps that did not actually shape retrieval success. We propose RICE-PO, a critic-free policy optimization framework that converts retrieval interactions into localized learning signals. RICE-PO selects high-uncertainty executable actions as anchors, evaluates local counterfactual branches using retrieval metrics, and propagates credit to latent reasoning steps only when reasoning-to-action influence is s
    
[^137]: 分布对齐作为上下文学习中任务向量设计的准则

    Distributional Alignment as a Criterion for Designing Task Vectors in In-Context Learning

    [https://arxiv.org/abs/2605.20730](https://arxiv.org/abs/2605.20730)

    该论文提出以分布对齐作为设计任务向量的准则，引入指标 $d_{\text{NTP}}$ 来衡量基于任务向量的推理与基于上下文学习的推理之间下一个词概率的差异，该指标与下游任务准确率呈强负相关，从而为任务向量质量提供了直接的评估与设计依据。

    

    上下文学习使大型语言模型（LLMs）能够通过示例演示来适应新任务，但随着上下文长度的增加，其推理成本不断攀升。任务向量通过将演示压缩为紧凑的隐状态表示，提供了一种有前景的替代方案，然而其质量一直仅通过下游任务准确率来评估。这种间接的评估标准对于如何设计更有效的任务向量提取方法所能提供的指导十分有限。在本文中，我们提出：使用任务向量进行推理时，其预测分布应与上下文学习的预测分布对齐。为了量化这一点，我们引入了 $d_{\text{NTP}}$，这是一个衡量基于任务向量的推理与基于上下文学习的推理之间下一个词概率差异的指标。我们的实证分析表明，$d_{\text{NTP}}$ 可以作为性能的代理指标，与下游准确率呈现强负相关。受此启发……

    arXiv:2605.20730v2 Announce Type: replace-cross  Abstract: In-context learning (ICL) allows large language models (LLMs) to adapt to new tasks through demonstrations, yet it suffers from escalating inference costs as context length increases. While task vectors offer a promising alternative by compressing demonstrations into compact hidden-state representations, their quality has been evaluated only through downstream task accuracy. This indirect criterion provides limited insight into how to design more effective task vector extraction methods. In this paper, we posit that inference using task vectors should align their predictive distribution with that of ICL. To quantify this, we introduce $d_{\text{NTP}}$, a metric that measures the discrepancy in next-token probabilities between task vector-based and ICL-based inference. Our empirical analysis reveals that $d_{\text{NTP}}$ serves as a performance proxy, exhibiting a strong negative correlation with downstream accuracy. Motivated b
    
[^138]: 大语言扩散模型的不确定性量化

    Uncertainty Quantification for Large Language Diffusion Models

    [https://arxiv.org/abs/2605.14570](https://arxiv.org/abs/2605.14570)

    本文首次系统研究了大语言扩散模型（LLDMs）的不确定性量化问题，提出了从迭代去噪过程中提取的轻量级零样本不确定性信号，并通过结合掩码扩散似然与轨迹语义相异度，将先进的不确定性量化方法适配到LLDMs。

    

    大语言扩散模型正在成为自回归模型的替代方案，通过更高的并行性提供更快的推理速度。与自回归大语言模型类似，它们仍然容易产生幻觉，这使得可靠的不确定性量化（UQ）对于安全部署至关重要。然而，现有的不确定性量化方法与这一新范式存在根本性的不匹配：它们假设自回归分解或使用代价高昂的重复采样，从而抵消了LLDMs的效率优势。在这项工作中，我们首次对LLDMs的不确定性量化进行了系统研究，并提出了从迭代去噪过程中导出的轻量级、零样本不确定性信号，利用中间生成结果、token重新掩码动态以及去噪复杂度。我们进一步通过将掩码扩散似然与基于轨迹的语义相异度相结合，将最先进的不确定性量化方法适配到LLDMs。我们证明了期望轨迹相异度可以下界……

    arXiv:2605.14570v2 Announce Type: replace  Abstract: Large Language Diffusion Models (LLDMs) are emerging as an alternative to autoregressive models, offering faster inference through higher parallelism. Similar to autoregressive LLMs, they remain prone to hallucinations, making reliable uncertainty quantification (UQ) crucial for safe deployment. However, existing UQ methods are fundamentally misaligned with this new paradigm: they assume autoregressive factorization or use expensive repeated sampling, negating the efficiency of LLDMs. In this work, we present the first systematic study of UQ for LLDMs and propose lightweight, zero-shot uncertainty signals derived from the iterative denoising process, leveraging intermediate generations, token remasking dynamics, and denoising complexity. We further adapt a state-of-the-art UQ method to LLDMs by combining masked diffusion likelihoods with trajectory-based semantic dissimilarity. We prove that expected trajectory dissimilarity lower bo
    
[^139]: TableVista：视觉与结构复杂性下的多模态表格推理基准测试

    TableVista: Benchmarking Multimodal Table Reasoning under Visual and Structural Complexity

    [https://arxiv.org/abs/2605.05955](https://arxiv.org/abs/2605.05955)

    TableVista是一个包含3,000个表格问题、30,000个多模态样本的综合基准，通过评估29个前沿模型揭示了当前基础模型虽能稳定应对多样渲染风格，但在复杂结构布局和纯视觉设置下存在显著的性能下降。

    

    我们提出了TableVista，一个用于评估基础模型在视觉和结构复杂性条件下多模态表格推理能力的综合性基准。TableVista包含3,000个高质量的表格推理问题，其中每个实例通过我们的多风格渲染与转换流水线扩展为10个不同的视觉变体。这一过程涵盖了多样化的场景风格、鲁棒性扰动以及纯视觉配置，最终形成30,000个多模态样本，用于多维度评估。我们对29个最先进的开源和专有基础模型在TableVista上进行了广泛评估。通过全面的定量与定性分析，我们发现尽管被评估的模型在各种渲染风格下基本保持稳定，但它们在复杂结构布局和纯视觉设置下表现出明显的性能下降，这表明当前模型难以应对视觉与结构复杂性带来的挑战。

    arXiv:2605.05955v2 Announce Type: replace  Abstract: We introduce TableVista, a comprehensive benchmark for evaluating foundation models in multimodal table reasoning under visual and structural complexity. TableVista consists of 3,000 high-quality table reasoning problems, where each instance is expanded into 10 distinct visual variants through our multi-style rendering and transformation pipeline. This process encompasses diverse scenario styles, robustness perturbations, and vision-only configurations, culminating in 30,000 multimodal samples for a multi-dimensional evaluation. We conduct an extensive evaluation of 29 state-of-the-art open-source and proprietary foundation models on TableVista. Through comprehensive quantitative and qualitative analysis, we find that while evaluated models remain largely stable across diverse rendering styles, they exhibit pronounced performance degradation on complex structural layouts and vision-only settings, revealing that current models struggl
    
[^140]: WHBench：基于专家在环验证的前沿大语言模型在女性健康主题上的评估基准

    WHBench: Evaluating Frontier LLMs with Expert-in-the-Loop Validation on Women's Health Topics

    [https://arxiv.org/abs/2604.00024](https://arxiv.org/abs/2604.00024)

    该论文提出WHBench——一个由专家深度参与验证构建的女性健康评估基准，通过对22个前沿大语言模型在47个专家设计的场景和23项标准下的系统测试，发现没有任何模型的平均表现超过75%，暴露了现有模型在女性健康领域的安全性、临床准确性和公平性方面的重大缺陷。

    

    大语言模型越来越多地被用于医疗指导，但女性健康在基准设计中仍然缺乏充分评估。我们提出了女性健康基准（WHBench），这是一个包含47个由专家精心设计的场景、覆盖10个女性健康主题的针对性评估套件，旨在揭示具有临床意义的失败模式，包括过时的指南、不安全的遗漏、剂量错误以及公平性相关的盲点。我们使用涵盖临床准确性、完整性、安全性、沟通质量、指令遵循、公平性、不确定性处理和指南依从性等23项标准的评分细则，对22个模型进行了评估，并采用安全加权惩罚和服务器端分数重新计算机制。在3,102次尝试回答中（3,100次被评分），没有任何模型的平均表现超过75%；最佳模型达到72.1%。即使是顶级模型也显示出较低的全对率和伤害率上的显著差异。评估者间信度中等……（摘要原文在此处截断）

    arXiv:2604.00024v3 Announce Type: replace-cross  Abstract: Large language models are increasingly used for medical guidance, but women's health remains under-evaluated in benchmark design. We present the Women's Health Benchmark (WHBench), a targeted evaluation suite of 47 expert-crafted scenarios across 10 women's health topics, designed to expose clinically meaningful failure modes including outdated guidelines, unsafe omissions, dosing errors, and equity-related blind spots. We evaluate 22 models using a 23-criterion rubric spanning clinical accuracy, completeness, safety, communication quality, instruction following, equity, uncertainty handling, and guideline adherence, with safety-weighted penalties and server-side score recalculation. Across 3,102 attempted responses (3,100 scored), no model mean performance exceeds 75 percent; the best model reaches 72.1 percent. Even top models show low fully correct rates and substantial variation in harm rates. Inter-rater reliability is mod
    
[^141]: ToolFlood：超越选择——通过语义覆盖向LLM智能体隐藏有效工具

    ToolFlood: Beyond Selection -- Hiding Valid Tools from LLM Agents via Semantic Covering

    [https://arxiv.org/abs/2603.13950](https://arxiv.org/abs/2603.13950)

    本文提出ToolFlood，一种针对工具增强型LLM智能体的检索层攻击，通过注入少量语义上覆盖众多用户查询的对抗性工具来主导top-k检索结果，从而将所有良性工具挤出智能体上下文。

    

    大型语言模型（LLM）智能体在处理复杂任务时越来越多地使用外部工具，并依赖基于嵌入的检索来选择一个较小的top-k子集用于推理。先前针对这些系统的攻击工作主要在工具选择阶段进行竞争，并假设良性工具仍保留在检索到的集合中；然而，产生该集合的检索阶段的鲁棒性却很少被研究。本文提出了ToolFlood，一种针对工具增强型LLM智能体的检索层攻击。ToolFlood并非改变检索后被选择的工具，而是直接攻击检索本身：它注入少量由攻击者控制的工具，其元数据经过构造以利用嵌入空间的几何特性，使这些工具在语义上覆盖大量用户查询，主导top-k检索结果，并将所有良性工具挤出智能体的上下文。ToolFlood采用两阶段对抗性工具生成策略：首先对目标查询的子集进行采样，然后利用LLM迭代地生成……

    arXiv:2603.13950v2 Announce Type: replace  Abstract: Large Language Model (LLM) agents increasingly use external tools for complex tasks and rely on embedding-based retrieval to select a small top-$k$ subset for reasoning. Prior work on attacking these systems competes for tool selection, assuming benign tools remain in the retrieved set; the robustness of the retrieval stage that produces this set is underexplored. This paper introduces ToolFlood, a retrieval-layer attack on tool-augmented LLM agents. Rather than altering which tool is chosen after retrieval, ToolFlood attacks retrieval itself: it injects a few attacker-controlled tools whose metadata is positioned to exploit the geometry of embedding space, so that these tools semantically span many user queries, dominate the top-k results, and push all benign tools out of the agent's context.   ToolFlood uses a two-phase adversarial tool generation strategy. It first samples subsets of target queries and uses an LLM to iteratively g
    
[^142]: 思考不等于告知：用户服务LLM智能体中的信息披露

    Thinking Is Not Telling: Information Disclosure in User-Service LLM Agents

    [https://arxiv.org/abs/2602.07796](https://arxiv.org/abs/2602.07796)

    该论文研究了用户服务LLM智能体中的信息披露机制，发现智能体的内部推理（配备"think"工具进行思考）反而会损害其与用户交互时的信息披露效果。

    

    用户交互型LLM智能体越来越多地运行在服务场景中，在这些场景下，任务的成功取决于智能体、用户和有状态环境三者之间的协调。在这类交互中，智能体通常掌握着用户无法观察到的丰富信息，例如任务策略、工具结果和环境状态，这使得智能体与用户之间的沟通成为任务完成的核心环节。在这项工作中，我们研究了用户服务智能体在沟通层面的一种失效模式：内部推理反而会对交互式信息披露产生负面影响。我们将信息披露引入为一种可度量的沟通机制，涵盖状态纠正、可用选项、约束条件、行为后果以及主动检索的环境信息。受LLM推理最新进展的启发，我们在受控实验设置下，对配备了"think"（思考）工具的智能体进行信息披露的评估与分析（摘要在此处截断）。

    arXiv:2602.07796v3 Announce Type: replace  Abstract: User-engaged LLM agents increasingly operate in service scenarios where task success depends on coordination between the agent, the user, and a stateful environment. In such interactions, the agent often has access to rich information that the user does not observe, such as task policies, tool results, and environment states. This makes agent-user communication a central component of task completion. In this work, we study a communication-level failure mode of user-service agents: internal reasoning is counterproductive for the interactive information disclosure. We introduce information disclosure as a measurable communication mechanism, covering state corrections, available options, constraints, consequences, and proactively retrieved environment information. Motivated by recent advances in LLM reasoning, we evaluate and analyse information disclosure under a controlled setting of agents equipped with a "think" tool, which allows a
    
[^143]: 压力测试情感支持模型：从同质化求助者迈向多样化求助者

    Stress-Testing Emotional Support Models: Moving from Homogeneous to Diverse Help Seekers

    [https://arxiv.org/abs/2601.07698](https://arxiv.org/abs/2601.07698)

    本文提出一个由九个心理语言特征驱动、基于专家混合架构训练的可控求助者模拟器，以多样化且细粒度可控的方式模拟真实求助者行为，从而对情感支持聊天机器人进行更严格的压力测试。

    

    随着情感支持聊天机器人近来在研究和产业领域获得显著关注，一种常见的评估策略随之出现：使用求助者模拟器与支持者聊天机器人进行交互。然而，当前的模拟器存在两个关键局限：(1) 它们未能捕捉现实世界中求助者的行为多样性，往往将求助者刻画得过于配合；(2) 它们缺乏模拟特定求助者画像所需的可控性。为应对这些挑战，我们提出了一个由支撑求助者行为的九个心理和语言特征驱动的可控求助者模拟器。利用真实的Reddit对话数据，我们通过专家混合架构训练模型，该架构能够有效地将多样化的求助者行为区分到专门的参数子空间中，从而增强细粒度的可控性。我们的模拟器在画像遵循度和行为方面取得了卓越表现。

    arXiv:2601.07698v3 Announce Type: replace  Abstract: As emotional support chatbots have recently gained significant traction across both research and industry, a common evaluation strategy has emerged: use help-seeker simulators to interact with supporter chatbots. However, current simulators suffer from two critical limitations: (1) they fail to capture the behavioral diversity of real-world seekers, often portraying them as overly cooperative, and (2) they lack the controllability required to simulate specific seeker profiles. To address these challenges, we present a controllable seeker simulator driven by nine psychological and linguistic features that underpin seeker behavior. Using authentic Reddit conversations, we train our model via a Mixture-of-Experts (MoE) architecture, which effectively differentiates diverse seeker behaviors into specialized parameter subspaces, thereby enhancing fine-grained controllability. Our simulator achieves superior profile adherence and behaviora
    
[^144]: PSM：通过LLM引导的黑盒优化实现提示词敏感度最小化

    PSM: Prompt Sensitivity Minimization via LLM-Guided Black-Box Optimization

    [https://arxiv.org/abs/2511.16209](https://arxiv.org/abs/2511.16209)

    该论文提出PSM框架，将提示词加固形式化为效用约束下的优化问题，通过在系统提示词后附加保护性文本层（SHIELD），并利用LLM作为黑盒优化器进行搜索，从而最小化对抗攻击下的系统提示词泄露风险。

    

    系统提示词对于引导大语言模型（LLM）的行为至关重要，然而它们往往包含专有逻辑或敏感信息，因此成为提取攻击的首要目标。对抗性查询能够成功诱导模型泄露这些隐藏指令，带来严重的安全与隐私风险。现有的防御机制通常依赖启发式方法、产生巨大的计算开销，或无法适用于通过黑盒API访问的模型。本文提出了一种通过“防护层附加”来加固系统提示词的新型框架，这是一种轻量级方法，即在原始提示词之上添加一个保护性文本层。我们的核心贡献是将提示词加固形式化为一个效用约束下的优化问题。我们利用“LLM即优化器”的方法来搜索可能的SHIELD防护文本空间，力求在保持模型效用的同时，最小化由一系列对抗性攻击得出的泄露指标。

    arXiv:2511.16209v3 Announce Type: replace-cross  Abstract: System prompts are critical for guiding the behavior of Large Language Models (LLMs), yet they often contain proprietary logic or sensitive information, making them a prime target for extraction attacks. Adversarial queries can successfully elicit these hidden instructions, posing significant security and privacy risks. Existing defense mechanisms frequently rely on heuristics, incur substantial computational overhead, or are inapplicable to models accessed via black-box APIs. This paper introduces a novel framework for hardening system prompts through shield appending, a lightweight approach that adds a protective textual layer to the original prompt. Our core contribution is the formalization of prompt hardening as a utility-constrained optimization problem. We leverage an LLM-as-optimizer to search the space of possible SHIELDs, seeking to minimize a leakage metric derived from a suite of adversarial attacks, while simultane
    
[^145]: 响应攻击：利用上下文启动效应越狱大语言模型

    Response Attack: Exploiting Contextual Priming to Jailbreak Large Language Models

    [https://arxiv.org/abs/2507.05248](https://arxiv.org/abs/2507.05248)

    本文提出“响应攻击”（RA）框架，通过在对话中策略性地将中间产生的轻度有害响应作为上下文启动器，再发出触发提示，从而利用上下文启动效应这一此前被忽视的漏洞实现对大语言模型的越狱。

    

    上下文启动效应（contextual priming）是指先前的刺激会隐蔽地影响后续的判断，这为大语言模型（LLM）提供了一个尚未被探索的攻击面。我们揭示了一种上下文启动漏洞：对话中的前一个响应可以将模型的后续行为引向违反政策的内容。现有的越狱攻击主要依赖于单轮或多轮的提示词操纵，或注入静态的上下文示例，这些方法存在效果有限、效率低下或语义漂移等问题。我们提出了响应攻击（Response Attack, RA），这是一种新颖的框架，它在对话中策略性地利用中间产生的轻度有害响应作为上下文启动器。通过重新表述有害查询，并在发出目标触发提示之前注入这些中间响应，RA利用了LLM中一个此前被忽视的漏洞。在八个最先进的大语言模型上进行的大量实验表明，RA始终能够实现……（摘要在此处被截断）

    arXiv:2507.05248v3 Announce Type: replace  Abstract: Contextual priming, where earlier stimuli covertly bias later judgments, offers an unexplored attack surface for large language models (LLMs). We uncover a contextual priming vulnerability in which the previous response in the dialogue can steer its subsequent behavior toward policy-violating content. While existing jailbreak attacks largely rely on single-turn or multi-turn prompt manipulations, or inject static in-context examples, these methods suffer from limited effectiveness, inefficiency, or semantic drift. We introduce Response Attack (RA), a novel framework that strategically leverages intermediate, mildly harmful responses as contextual primers within a dialogue. By reformulating harmful queries and injecting these intermediate responses before issuing a targeted trigger prompt, RA exploits a previously overlooked vulnerability in LLMs. Extensive experiments across eight state-of-the-art LLMs show that RA consistently achie
    
[^146]: 视觉上下文攻击：利用图像驱动的上下文注入越狱多模态大语言模型

    Visual Contextual Attack: Jailbreaking MLLMs with Image-Driven Context Injection

    [https://arxiv.org/abs/2507.02844](https://arxiv.org/abs/2507.02844)

    该论文提出了一种新颖的“以视觉为中心的越狱”设定及VisCo攻击方法，通过图像驱动的上下文注入构建完整真实的越狱语境，从而有效攻击多模态大语言模型。

    

    随着强大的视觉语言能力的出现，多模态大语言模型（MLLMs）在现实世界应用中展现出巨大的潜力。然而，视觉模态所表现出的安全漏洞对在开放世界环境中部署此类模型构成了重大挑战。近期的研究通过将有害文本语义直接编码到视觉输入中，成功诱导目标MLLMs产生有害响应。然而，在这些方法中，视觉模态主要充当不安全行为的触发器，往往表现出语义模糊性，并且缺乏与现实场景的关联。在这项工作中，我们定义了一个新颖的设置：以视觉为中心的越狱，其中视觉信息作为构建完整且真实的越狱上下文的必要组成部分。基于这一设置，我们提出了VisCo（视觉上下文）攻击。VisCo伪造上下文对话（原文在此处截断）

    arXiv:2507.02844v3 Announce Type: replace-cross  Abstract: With the emergence of strong vision language capabilities, multimodal large language models (MLLMs) have demonstrated tremendous potential for real-world applications. However, the security vulnerabilities exhibited by the visual modality pose significant challenges to deploying such models in open-world environments. Recent studies have successfully induced harmful responses from target MLLMs by encoding harmful textual semantics directly into visual inputs. However, in these approaches, the visual modality primarily serves as a trigger for unsafe behavior, often exhibiting semantic ambiguity and lacking grounding in realistic scenarios. In this work, we define a novel setting: vision-centric jailbreak, where visual information serves as a necessary component in constructing a complete and realistic jailbreak context. Building on this setting, we propose the VisCo (Visual Contextual) Attack. VisCo fabricates contextual dialogu
    
[^147]: 换个说法：基于用户行为的自动改写框架用于大语言模型审计

    Say It Another Way: Auditing LLMs with a User-Grounded Automated Paraphrasing Framework

    [https://arxiv.org/abs/2505.03563](https://arxiv.org/abs/2505.03563)

    该论文提出AUGMENT框架，通过基于用户行为的受控提示词改写来审计大语言模型，能够揭示无约束改写方式下被掩盖的系统性弱点。

    

    大语言模型（LLMs）对提示词措辞的细微变化高度敏感，这给可靠的模型审计带来了挑战。先前的方法通常采用无约束的提示词改写，这可能会遗漏塑造真实用户交互的语言和人口统计因素。我们提出了AUGMENT（Automated User-Grounded Modeling and Evaluation of Natural Language Transformations，基于用户行为的自然语言转换自动建模与评估），一个以用户行为为基础生成受控改写文本的框架。AUGMENT利用基于语言学知识的规则，并通过指令遵循、语义相似性和真实性检查来保证质量，确保改写结果既可靠又有意义，适用于审计任务。通过对BBQ和MMLU数据集的案例研究，我们展示了受控改写能够揭示在无约束变化下仍然被掩盖的系统性弱点。这些结果凸显了AUGMENT框架在实现可靠审计方面的价值。

    arXiv:2505.03563v4 Announce Type: replace  Abstract: Large language models (LLMs) are highly sensitive to subtle changes in prompt phrasing, posing challenges for reliable auditing. Prior methods often apply unconstrained prompt paraphrasing, which risk missing linguistic and demographic factors that shape authentic user interactions. We introduce AUGMENT (Automated User-Grounded Modeling and Evaluation of Natural Language Transformations), a framework for generating controlled paraphrases, grounded in user behaviors. AUGMENT leverages linguistically informed rules and enforces quality through checks on instruction adherence, semantic similarity, and realism, ensuring paraphrases are both reliable and meaningful for auditing. Through case studies on the BBQ and MMLU datasets, we show that controlled paraphrases uncover systematic weaknesses that remain obscured under unconstrained variation. These results highlight the value of the AUGMENT framework for reliable auditing.
    
[^148]: 纯注意力硬最大值Transformer中的聚类及其在情感分析中的作用

    Clustering in pure-attention hardmax transformers and its role in sentiment analysis

    [https://arxiv.org/abs/2407.01602](https://arxiv.org/abs/2407.01602)

    本文从动力系统角度严格证明了纯注意力硬最大值Transformer在层数趋于无穷时输入会收敛到由“领导者”点决定的聚类平衡态，并据此构建了完全可解释的Transformer模型，通过将无意义词聚类到语义核心词周围来有效捕捉上下文，从而解决情感分析问题。

    

    Transformer是极其成功的机器学习模型，但其数学性质仍未被充分理解。本文严格刻画了具有硬最大值自注意力和归一化子层的Transformer在层数趋于无穷时的行为。通过将此类Transformer视为描述欧几里得空间中点演化的离散时间动力系统，并借助基于超平面分离的自注意力机制的几何解释，我们证明了Transformer的输入渐近收敛到由称为“领导者”的特殊点所决定的聚类平衡态。随后，我们利用这一理论认识，采用完全可解释的Transformer模型来解决语言处理中的情感分析问题，该模型通过将无意义词汇聚类到承载最多语义的领导者词汇周围，有效捕捉了“上下文”。最后，我们概述了仍然存在的（问题）……

    arXiv:2407.01602v3 Announce Type: replace  Abstract: Transformers are extremely successful machine learning models whose mathematical properties remain poorly understood. Here, we rigorously characterize the behavior of transformers with hardmax self-attention and normalization sublayers as the number of layers tends to infinity. By viewing such transformers as discrete-time dynamical systems describing the evolution of points in a Euclidean space, and thanks to a geometric interpretation of the self-attention mechanism based on hyperplane separation, we show that the transformer inputs asymptotically converge to a clustered equilibrium determined by special points called \textit{leaders}. We then leverage this theoretical understanding to solve sentiment analysis problems from language processing using a fully interpretable transformer model, which effectively captures `context' by clustering meaningless words around leader words carrying the most meaning. Finally, we outline remainin
    

