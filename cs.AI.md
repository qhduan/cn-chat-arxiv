# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Efficient SWE Agent Benchmarking via Trajectory-Aware Evaluation](https://arxiv.org/abs/2609.01603) | 提出PTA-IRT框架，将历史执行轨迹作为特权信息融合过程与结果信号，在低校准预算下更准确地恢复软件工程智能体的完整基准分数与排名。 |
| [^2] | [Adaptive Critical Token-Aware Retrieval for Repository-Level Code Generation](https://arxiv.org/abs/2609.01601) | 该论文提出ACToR，通过识别LLM自回归代码生成过程中容易出错的关键token位置，并自适应地为这些位置检索细粒度的仓库上下文，从而提升仓库级代码生成的功能正确性。 |
| [^3] | [CordisBench: Can Language Models Reason About Component Lifecycles in Dynamic Agent Harnesses?](https://arxiv.org/abs/2609.01600) | 该论文提出了 CordisBench——一个包含 1,200 道题目的基准，用于评估语言模型在动态智能体框架中对组件依赖与清理等生命周期问题的推理能力，发现模型在小规模系统上表现良好，但随相关交互数量增多可靠性显著下降。 |
| [^4] | [The Rise of Verbal Reinforcement Learning](https://arxiv.org/abs/2609.01597) | 本文首次对“言语强化学习”这一新兴范式进行了统一阐述，根据言语反馈生效的时机与作用对象，将其系统归纳为语言作为基础定位信号、语言作为审慎反馈以及语言作为学习信号三大支柱。 |
| [^5] | [Mechanism Design for Alignment and Control](https://arxiv.org/abs/2609.01595) | 该论文提出了一个针对对齐和能力均未知的AI智能体的机制设计框架，利用“能力可隐藏但不可伪造”的单边模仿结构建立了显示原理与可实施政策的刻画方法，并应用于防“装弱”、对齐与可解释性权衡、同伴评分约束、多智能体竞争激励以及可扩展监督等AI对齐与控制问题。 |
| [^6] | [Designing Proactive Thought Partners for Writing](https://arxiv.org/abs/2609.01588) | 本文提出并探索了“主动式思维伙伴”的设计空间——一种能在写作过程中主动提供可定制高层次认知支持的AI智能体，通过一周的部署研究发现用户会以前瞻性规划配置支持、将建议用于创意生成与自我监控，并重视轻量级的视觉呈现。 |
| [^7] | [Scaling Near-Optimal SFT-RL Annotation Budget Allocation from Small to Large LLMs](https://arxiv.org/abs/2609.01573) | 该论文提出“近最优区域”框架来分配SFT-RL标注预算，发现该区域宽广且随模型规模增大而扩大，并能从小型代理模型可靠迁移到大型目标模型，因此小规模代理实验即可替代在大模型上的穷尽式预算搜索。 |
| [^8] | [Selective Agent Guidance via Entropy: Learning Autonomous Policies from Imperfect VLM Teachers](https://arxiv.org/abs/2609.01567) | 该论文提出SAGE框架，仅在智能体不确定时才查询昂贵的视觉语言模型教师，并利用环境优势对教师建议进行加权蒸馏，从而训练出无需教师引导即可自主行动的轻量级强化学习策略。 |
| [^9] | [From Confusion to Clarity: Confusion-Aware Retrieval and Knowledge Injection for Text Classification](https://arxiv.org/abs/2609.01564) | 该论文提出一个无需微调的框架，通过识别模型易混淆的标签对、扩充候选集并生成针对性的区分规则注入知识，帮助大语言模型在语义相似标签的文本分类任务中做出正确选择，且这些规则还可迁移到更小、成本更低的模型上。 |
| [^10] | [H3-World: Turning Language Understanding into World Control](https://arxiv.org/abs/2609.01560) | H3-World通过将角色与摄像机指令的结构化组合和时序视频潜变量对齐，并引入时序注意力路由机制来限制指令的作用时间区间，成功将33B的MiniMax-H3视频生成器转化为无需专门动作模块、可实现精确时间控制的世界模型。 |
| [^11] | [Retrieved but not ranked: surface-form bias in structural retrieval, from mathematics to agent trajectories](https://arxiv.org/abs/2609.01556) | 该研究在竞赛数学与具身智能体轨迹两个领域以统一协议评测刻意分离表层形式与语义结构的嵌入检索，发现主流嵌入模型存在严重的表层形式（字面词汇）偏差：在结构相同但措辞伪装最重的任务上Hit@1跌至0.0%，未命中时胜出者几乎总是与查询词汇更相似的条目，表明当前嵌入检索锚定于字面文本而非深层结构。 |
| [^12] | [BS: Take the Hint - Interactive Multitracer PET/CT Lesion Segmentation with a Scribble-Conditioned ResEnc U-Net](https://arxiv.org/abs/2609.01554) | 本文提出一种涂鸦条件化的残差编码器U-Net，将用户标注的前景和背景涂鸦作为额外输入通道，并从autoPET-III获胜权重初始化后微调，以实现交互式多示踪剂全身PET/CT病灶分割。 |
| [^13] | [Can LLMs Discover Scientific Laws in Real and Parallel Worlds?](https://arxiv.org/abs/2609.01552) | 该论文提出了基于已发表研究和真实科学数据构建的科学定律发现基准SCILAWS-BENCH（涵盖118个问题、291个候选定律、约800万真实数据点和六个学科），并采用真实世界与平行世界两种互补设置，以严格评估大语言模型能否真正发现科学定律。 |
| [^14] | [A Mathematical Theory of Reusable Neural Bases for Network Compression](https://arxiv.org/abs/2609.01550) | 该论文提出线性可复用神经基底架构（LRNBA），通过将网络块表示为共享神经基底的线性组合，在保持稳定训练的同时大幅压缩参数并降低内存成本，使模型在相同参数预算下能够构建更宽更深的网络。 |
| [^15] | [Can LLMs Design Video Coding Tools? A Case Study on Planar Mode](https://arxiv.org/abs/2609.01535) | 本文通过“生成-评估”循环的实证研究，首次证明大语言模型能够自动设计并迭代改进视频编码中的Planar帧内预测模式，在VVenC编码器上实现了0.18%的码率节省且复杂度开销仅0.4%。 |
| [^16] | [EvoSCM: Scientific Belief Revision Through Causal Model Evolution and Experimentation](https://arxiv.org/abs/2609.01526) | 本文提出EvoSCM框架，为科学智能体配备随实验证据不断演化的显式结构因果模型，通过维护相互竞争的假设种群并执行“溯因—干预设计—可证伪预测—实验检验—归纳修正”的闭环发现循环，使科学信念变得可测试、可修正。 |
| [^17] | [Relational-Core Graph Analytics Querying graphs at SQL scale, and why the node/edge model is a performance tax, not a truer picture of connected data](https://arxiv.org/abs/2609.01525) | 该论文提出ClickGraph和DeltaGraph系统，将Cypher图查询直接翻译到原生关系模式上并在列式引擎（如ClickHouse、Databricks）上就地执行，证明关系型引擎在分析型图查询上可匹敌甚至超越原生图引擎，并能扩展到内存图引擎失效的规模，同时指出节点/边模型是对关系表中已有关系的冗余重编码，纯属性能开销。 |
| [^18] | [When Guardrails Look Effective: Construct Validity Failures in LLM Agent Commerce Evaluation](https://arxiv.org/abs/2609.01519) | 该论文通过审计LLM智能体酒店交易模拟评估，揭示市场护栏看似带来的福利增益实为评估设计中报价模式不一致、选择程序差异及采样噪声所导致的构念效度失效，而非真实的经济效应。 |
| [^19] | [TempCloze: Can Video-LLMs Identify the Missing Middle?](https://arxiv.org/abs/2609.01515) | 该论文提出TempCloze视频完形填空基准，通过要求模型从四个候选中识别视频缺失的中间片段来评估纯视觉时间推理能力，发现事件何时发生的“时间对齐”是当前视频大语言模型的主要瓶颈。 |
| [^20] | [LatentPress: Context Compression Beyond Text and Vision](https://arxiv.org/abs/2609.01507) | LatentPress提出将对话历史和长文档压缩为连续记忆token这一第三种表示形式，让冻结的语言模型通过输入嵌入接口直接读取，仅训练约占解码器0.1%参数的适配器即可实现4-16倍压缩，且性能超过文本摘要和基于OCR的压缩方法。 |
| [^21] | [Optimizing Byzantine Node Placement in Decentralized Federated Learning](https://arxiv.org/abs/2609.01495) | 该论文首次将拜占庭节点的网络位置作为显式攻击决策进行研究，提出基于真实 gossip 传播动态的集合级度量 BPI 来量化诚实节点的累积暴露程度，从而在固定攻陷预算下找到对去中心化联邦学习影响最大的拜占庭节点放置方案。 |
| [^22] | [Rethinking Learnability in Offline Data-driven Optimization](https://arxiv.org/abs/2609.01493) | 本文针对PAC可学习性无法充分刻画离线优化的理论缺陷，提出了“算法依赖的可学习性”这一新概念，其只需保证在优化器轨迹上的精度即可支撑离线数据驱动优化。 |
| [^23] | [GlossoGen: Emergent Language in Complex Multi-Agent LLM Interactions](https://arxiv.org/abs/2609.01491) | 本文提出GlossoGen平台，通过SaveVeyru压力沟通场景证实LLM多智能体之间会涌现语言演化，产生的语言具有组合性和形态生成性但人类无法理解，并发现效率压力、模型能力和“事后复盘”阶段是语言演化的关键条件。 |
| [^24] | [Defense-as-Skill: Evolving Runtime Guard Skill for Skill-Augmented Agents](https://arxiv.org/abs/2609.01487) | 提出“防御即技能”新范式，将运行时安全守卫 SkillSonar 本身实现为可安装、可检查、可编辑的技能，使其与不可信任务技能并行运行，并依据用户任务边界对敏感操作进行条件化检查（允许、重新规划或确认），同时构建了覆盖 6 个风险类别的 SCOPE-R 数据集。 |
| [^25] | [Harness-of-Harness: Multi-Day Autonomous Software Development with Continual Improvement](https://arxiv.org/abs/2609.01481) | 本文提出HoH框架，通过迭代循环、增量开发、独立评估与版本管理等机制，使LLM编程智能体能够在多天自主软件开发过程中持续改进软件。 |
| [^26] | [Parsing the Stream: A Live Trace Model for Long-Horizon Agents and Their Observers](https://arxiv.org/abs/2609.01466) | 该论文提出一种“实时轨迹模型”，将只追加的事件账本增量折叠为类型化运行状态并编译成按消费者定制的视图，使观察者以约14-15倍的token节省、5-7倍更低的成本获得更高准确率的监控答案，同时帮助长程智能体管理超出其上下文容量的轨迹。 |
| [^27] | [When Safety Routing Breaks: Understanding Alignment Fragility under Benign Fine-Tuning](https://arxiv.org/abs/2609.01455) | 该论文提出用Fisher几何解释对齐脆弱性：良性微调会在输出侧MLP模块中重新锐化被对齐所平坦化的安全路由通路，导致安全性崩溃至高攻击成功率而通用能力仅轻微下降，且LoRA和ASAM的保护作用仅在早期微调规模下有效。 |
| [^28] | [Efficiently Estimating Optimal Hyperparameter Scaling Laws through Power-Law Entropy Search](https://arxiv.org/abs/2609.01431) | 本文提出幂律熵搜索（PLES），一种基于多保真度贝叶斯优化的计算成本感知采集函数，通过自适应选择能最大程度降低缩放定律估计整体不确定性的实验配置（而非优化单一目标函数），高效估计大语言模型最优超参数随规模变化的缩放定律，从而大幅节省计算资源。 |
| [^29] | [Learning Sparse Decision Trees via Transformer Variational Auto-Encoders](https://arxiv.org/abs/2609.01430) | TREVIS通过树Transformer变分自编码器的潜空间探索，将决策树的离散搜索转化为连续空间中基于梯度的优化，从而学习同时兼顾预测性能和结构稀疏性的决策树。 |
| [^30] | [Semantic-Guided Multimodal Preprocessing for Vision Transformer-Based Clear Cell Renal Cell Carcinoma Grading](https://arxiv.org/abs/2609.01426) | 提出一种语义引导的多模态预处理方法，将细胞核分类图与RGB病理图像融合后输入视觉Transformer进行透明细胞肾细胞癌分级，将平衡准确率从0.707提升至0.916。 |
| [^31] | [Provably Safe Sim-to-Real Transfer](https://arxiv.org/abs/2609.01418) | 该论文提出并形式化了“安全仿真到现实迁移”问题，通过在无奖励安全强化学习框架内构建该问题，使智能体能够在利用不完美模拟器的同时确保现实世界数据收集的安全性，并为目标系统学习到接近最优的可行策略。 |
| [^32] | [EdiTikZ: Scientific Figure Editing from Revision Trajectories](https://arxiv.org/abs/2609.01409) | 该论文提出 DaEdiTikZ——首个从 arXiv、GitHub 和 TeX SE 的自然修订轨迹中挖掘的大规模科学图表编辑数据集（包含 39.1 万个 TikZ 编辑对和 78.1 万条推断的编辑指令），并配套构建了人工精修基准 DaEdiTikZ-Bench，以自然修订轨迹作为可扩展的监督信号来训练科学图表编辑模型。 |
| [^33] | [Neuro-Symbolic Geometric Abstraction (NeuSOGA): From Observations to Symbolic Mathematical Representations](https://arxiv.org/abs/2609.01408) | 本文提出NeuSOGA框架，通过将观测逐步转化为拓扑抽象、几何抽象和符号数学表示三个层次，弥合了神经网络的感知能力与显式符号推理之间的鸿沟。 |
| [^34] | [Evaluating Multimodal LLMs as Generalist Vision-Language-Action Agents for Drone Control: Commanding, Approaching, Tracking and Searching](https://arxiv.org/abs/2609.01404) | 该论文提出了DroneCATS-Agent架构和DroneCATS基准，在不进行微调的情况下将多模态大语言模型直接作为无人机控制回路中的通用决策者，系统评估其在接近、跟踪、搜索和指挥多无人机编队四项核心能力上的表现。 |
| [^35] | [Measuring consistency via ensemble margin and local prediction variability: Auditing decision systems in the presence of predictive multiplicity](https://arxiv.org/abs/2609.01397) | 该论文提出一种将集成边界与局部预测变异性相结合的一致性准则，用于在罗生门效应（预测多样性）存在下审计决策系统，并证明在温和假设下有限集成的一致性分数会收敛于罗生门集合中期望模型的一致性分数。 |
| [^36] | [EDGE: Error Dependency Graph-Guided Multi-Error Attribution in Multi-Agent LLM Systems](https://arxiv.org/abs/2609.01360) | 提出EDGE框架，通过构建错误依赖图并利用反事实推演验证因果子集，引导两阶段LLM-as-judge检测器，实现多智能体LLM系统中更可靠的多错误归因。 |
| [^37] | [PopPert: Population-level Joint-Distribution Modeling for Single-Cell Perturbation Prediction](https://arxiv.org/abs/2609.01357) | PopPert提出在群体水平对基因表达联合分布进行建模，通过预测扰动引起的分布参数变化来实现单细胞扰动预测，从而摆脱对细胞间配对关系的假设并降低单细胞噪声的影响。 |
| [^38] | [SymFold: Synergizing Evolutionary and Structural Priors for Accurate Protein Inverse Folding](https://arxiv.org/abs/2609.01353) | 提出对称双路径架构SymFold，协同利用蛋白质语言模型的进化先验与多模态蛋白质语言模型的结构先验来迭代引导序列生成，突破了传统串行流程中逆折叠性能受上游粗糙预测质量限制的瓶颈。 |
| [^39] | [CHARM: Character Hallucination for Multicultural Role Play Benchmark](https://arxiv.org/abs/2609.01352) | CHARM是一个涵盖五大文化语言区域40个角色的多文化角色扮演基准，创新性地将角色幻觉拆分为“边界意识”与“边界遵守”两个独立阶段进行评估，从而更精细地定位大语言模型角色扮演中幻觉错误的来源。 |
| [^40] | [Scalable Rao-Blackwellized Online Planning for High-Dimensional POMDPs](https://arxiv.org/abs/2609.01351) | 本文通过混合连续-离散信念表示扩展了Rao-Blackwell化在线POMDP框架，在树搜索规划中解析地传播边缘化状态分量的不确定性，从而降低高维POMDP价值估计的采样方差，并结合FastSLAM 2.0在机器人搜救任务中验证了其有效性。 |
| [^41] | [Cheap Verifiers, Large Blind Spots: Measuring the Reliability Cost of Cost-Saving Cascades](https://arxiv.org/abs/2609.01345) | 该研究通过真实LLM实验发现，推理级联中廉价验证器对学生模型错误答案的“盲区”随学生能力增强而扩大、随验证器能力增强而缩小，恰好在级联机制赖以存在的低成本配置下最为严重，而用前沿验证器消除盲区又会因过度升级而抵消成本节约，从而揭示了成本节约级联设计背后隐藏的显著可靠性代价。 |
| [^42] | [Probing Factual Knowledge Transfer with Training Data Interventions](https://arxiv.org/abs/2609.01341) | 该论文提出了一种基于干预的评估框架并构建SIFT数据集，通过从波斯语训练数据中系统性移除特定事实来检验多语言模型的知识跨语言迁移能力，发现英语预训练中习得的事实知识向波斯语的迁移非常有限。 |
| [^43] | [LEAP: Likelihood Elicitation and Aggregation for LLM-based Probabilistic Forecasting](https://arxiv.org/abs/2609.01337) | LEAP通过让大语言模型对每条证据单独引出似然参数，再借助显式先验与确定性概率模型将其聚合为后验分布，从而改进基于LLM的概率预测并保证证据贡献的可复现性。 |
| [^44] | [Bandits in Prod: Hyperparameter Optimization at Inference Time](https://arxiv.org/abs/2609.01335) | 该论文将生产系统中只能通过线上噪声反馈评估配置的场景形式化为在线超参数优化（OHPO），提出通用框架IMABO及免重启的无限多臂老虎机策略IMOSS，并给出了分位数遗憾的理论保证。 |
| [^45] | [Automated Event Log Generation from Unstructured Text Using Finetuned LLMs](https://arxiv.org/abs/2609.01320) | 该论文提出一个可扩展框架，将大语言模型作为自动化数据翻译器，并在新构建的文本到日志数据集上进行微调，使模型能够从非结构化文本中提取高保真事件日志，从而弥合组织非结构化知识与流程挖掘所需结构化数据之间的鸿沟。 |
| [^46] | [MIDR: Enrichment-Augmented Indexing for Multimodal Document Retrieval](https://arxiv.org/abs/2609.01316) | MIDR是一个无需训练的富化增强索引框架，通过在索引阶段利用多模态大语言模型将文档页面转换为经验证的文本字段，将多模态推理从查询时转移到索引时，在ViDoRe V3上相比BM25相对提升23.0%，性能可与ColQwen2.5媲美。 |
| [^47] | [A Composable Evaluation System for Reproducible Omni-Modal Foundation Model Evaluation](https://arxiv.org/abs/2609.01315) | OmniEvaluator 是一个可组合的全模态基础模型评估系统，通过统一接口连接现有推理引擎与评估框架，支持四个推理后端、四个评估框架和一千多个基准测试，并保证每次运行可精确复现与跨模型比较。 |
| [^48] | [GazeRefine: Expert Gaze as a Test-Time Prompt for Training-Free Medical Image Segmentation](https://arxiv.org/abs/2609.01310) | GazeRefine提出了一种免训练的零样本医学图像分割框架，将专家眼动注视转化为前景/背景先验，在冻结的DINOv3特征空间中初始化并迭代细化语义原型，无需任何分割掩码、微调或梯度更新。 |
| [^49] | [Analog-DB: An Agent-First Analog Integrated Circuit Database, From Blocks to Systems](https://arxiv.org/abs/2609.01286) | 提出开源版本化数据库 analog-db，通过工艺无关的拓扑表示、带约束的参数化方案和可查询目录，实现模拟集成电路设计的完整共享、复用与可重定向，并让 AI 设计智能体能够直接发现和复用电路设计。 |
| [^50] | [HiLRP: Toward One Trustworthy Explanation for Vision Transformer: Conservation-Valid Attribution via Attention Primitives](https://arxiv.org/abs/2609.01282) | 该论文提出HiLRP框架，将各类ViT中的注意力与降采样算子统一分解为线性映射、双线性混合、归一化/门控和重索引四种基元操作，从而实现首个能够跨越多种ViT架构变体、满足守恒有效性的统一可信归因解释方法。 |
| [^51] | [EmbodiedSkills: A Unified Framework for Orchestrating, Training, and Deploying VLA Agents](https://arxiv.org/abs/2609.01281) | 提出EmbodiedSkills统一框架，将技能决策视为执行提案，通过运行时前置条件检查与执行后结果验证，在单一智能体循环中协调高层技能选择、有界的低层VLA执行和动作后验证，并支持低层VLA策略的灵活替换与适配。 |
| [^52] | [Some Emotions Run Deeper: Layer-wise Probing and Causal Intervention in Large Language Models](https://arxiv.org/abs/2609.01279) | 该研究结合逐层探测与因果干预，在三个情感显式程度不同的语料库和八个大语言模型上发现，情感在模型中的可读取深度随文本来源系统性变化——越隐含、越依赖语境的情感需要越深的层才能读取，说明情感表达深度同时取决于文本来源与模型本身。 |
| [^53] | [TimeSteer: Inference-Time Speech Scheduling in Joint Audio-Visual Diffusion Models](https://arxiv.org/abs/2609.01277) | 提出TimeSteer——一个无需训练的框架，利用对时间敏感的交叉注意力头定位每条话语的隐含源区间，并借助干净潜变量的耦合结构，在推理时将音视频扩散模型生成的语音精确调度到用户指定的时间区间内。 |
| [^54] | [The Constitutional Coverage Trilemma in AI Governance](https://arxiv.org/abs/2609.01275) | 该研究通过审计23个前沿大模型的默认“宪法”并调查1,649人的价值权衡偏好，发现人类的价值需求广泛多样，而AI模型隐含的价值排序供给既狭窄又随时间漂移，导致近四成用户找不到符合自身价值偏好的模型，揭示了AI治理中的“宪法覆盖三难困境”。 |
| [^55] | [Making Prospective Memory SLM-Shaped: Typed Intention Stores for Small-Model Agents](https://arxiv.org/abs/2609.01272) | 提出一种无需训练的前瞻意图存储（PIS）框架，通过类型化动作空间将生命周期逻辑交给代码执行，使小语言模型智能体在 PM-Bench 前瞻记忆任务上达到 82.9% 的 Set-F1，大幅超越前沿大模型并创下新的最优纪录。 |
| [^56] | [Dual Process Motion Planning](https://arxiv.org/abs/2609.01260) | 本文受“思考快与慢”范式启发，提出一种双过程神经符号运动规划架构，通过元认知控制器动态协调符号求解器（系统2）与经验驱动的学习模块（系统1），兼顾规划的鲁棒性与计算效率。 |
| [^57] | [Measuring the Behavioral Fidelity of Long-Horizon Human Activity Simulations](https://arxiv.org/abs/2609.01257) | 该论文提出了一个跨时间粒度和分析层级评估长时程人类活动模拟行为保真度的框架，并通过43小时办公室真实活动数据集的案例研究发现，不同条件机制在各项指标上表现不一致——统计先验虽最能匹配真实的活动与序列分布，却会过度碎片化例程并抑制个体内变异性，因此需要更全面的多维度评估方法。 |
| [^58] | [One Prompt Is Enough: Watermark Laundering Through Foundation Image Models](https://arxiv.org/abs/2609.01249) | 该论文首次将“水印洗白”形式化为一种新威胁，证明攻击者仅需一条重建提示词通过公开基础图像模型即可使不可见水印无法解码，并在六款图像编辑模型和三种水印方案上系统评估了其洗白效果。 |
| [^59] | [Explore More, Drift Less: Outcome-Only Reinforcement Learning Can Suffice for Long-Horizon Interactive Agents](https://arxiv.org/abs/2609.01245) | 本文提出CANOPY方法，论证仅结果奖励的强化学习足以训练小规模开源LLM智能体完成长时程交互任务，所谓瓶颈实为探索不足导致的信号饥饿与缺乏锚定导致的策略漂移这两个常见实践问题的产物。 |
| [^60] | [From Language to Behavior: Scaling Sequence Transformers for Industrial Recommendation Ranking with Rec-Native Designs](https://arxiv.org/abs/2609.01240) | 提出推荐原生的Transformer扩展框架ReST，通过双门控注意力编码器应对噪声化行为序列，通过重量级可复用编码器加轻量级交叉解码器的分解设计及共享前缀训练与服务机制，解决推荐排序中的计算不对称问题，实现工业推荐排序的高效规模化。 |
| [^61] | [MutMem-V2: Cryptographically Authorized Mutation in Persistent Agent Memory Portable Verification and Reproducible Evidence](https://arxiv.org/abs/2609.01235) | MutMem V2 在不引入新记忆引擎的情况下，为持久化智能体记忆的密码学授权变更补全了可移植验证契约与干净安装复现路径，通过精确的规范字节、域分离承诺、完整的测试向量以及双语言独立实现的一致性验证提供了可复现证据。 |
| [^62] | [Position Matters: Feature Inversion Attacks in ViT Split Inference with Token Reduction and Shuffling](https://arxiv.org/abs/2609.01232) | 该论文揭示了ViT分割推理中传输的令牌嵌入即使经过缩减与打乱仍保留关键位置信息，并提出空间对齐重建攻击（SARA），能够从这些嵌入中重建原始输入，证明现有隐私保护机制存在严重安全隐患。 |
| [^63] | [Prompt-Robust Language Models: Which Training Strategies Work?](https://arxiv.org/abs/2609.01217) | 该论文在受控条件下系统复现并比较了多种提升大语言模型提示词鲁棒性的训练策略，发现现有鲁棒性微调方法虽优于标准微调和上下文学习，但最佳与最差提示词间的性能差距仍高达40-57%，且CoIN、PPCL等专门的鲁棒性增强目标往往不如“每个批次使用一个模板”这一最简单的数据构建策略。 |
| [^64] | [H2Table: Hierarchical Hypergraph-Enhanced Large Language Models for Complex Table Reasoning](https://arxiv.org/abs/2609.01216) | 提出H2Table框架，将复杂表格表示为层次嵌套超图，通过定制超图编码器建模表头与单元格间的语义关系，并利用可学习查询向量将结构嵌入注入大语言模型，从而显著提升复杂表格推理能力。 |
| [^65] | [REFACTOR-VLA: Unsupervised Library Learning of Typed Motor Programs](https://arxiv.org/abs/2609.01215) | REFACTOR-VLA 提出了一种清醒/睡眠两阶段框架，通过基于潜在世界模型 rollout 计算的行为等价核对运动程序片段进行无监督聚类，并生成 Hindley–Milner 风格的类型化 lambda 项来构建可复用技能库，从而提升 VLA 模型在长时程任务上的性能与可解释性。 |
| [^66] | [Who Judges the Judges? A Chinese Safety QA Benchmark for Evaluating LLM Responses and Safety Judges](https://arxiv.org/abs/2609.01210) | 该论文提出了C-SafeQA，一个以政策为依据的中文回复级安全评估基准，包含538个基础查询和8,877个对抗性查询，既能评估大语言模型回复的安全性，也能审计自动化安全评判者本身的可靠性。 |
| [^67] | [Autonomous discovery of new structure-plausibility laws for explainable and rapid crystal diagnosis and screening](https://arxiv.org/abs/2609.01209) | 该研究通过智能体自主生成、测试并反驳两百万条候选定律，发现了八条可解释的无机结构合理性规则（PRIS），其诊断晶体结构的能力远超鲍林规则和传统距离截断法，并与可合成性线性相关，可实现快速晶体筛选。 |
| [^68] | [Towards AI-Assisted Clinical Trial Matching: Practical Considerations, Multicenter Evaluation, and Real-World Deployment](https://arxiv.org/abs/2609.01202) | 本文提出面向真实世界部署的AI临床试验推荐系统TrialGPT 2.0，它不仅评估患者资格，还结合患者临床需求和本地工作流优先级筛选值得进一步考虑的试验，并提供了结构化的可审查解释，在政府、学术癌症中心等多种肿瘤学场景中完成了回顾性与前瞻性多中心评估。 |
| [^69] | [FinLifeBench: Exhaustive Life-Event History and Financial-State Reconstruction from Longitudinal Banking Dialogue](https://arxiv.org/abs/2609.01198) | 提出FinLifeBench基准，基于6,000个韩语银行对话会话，评估大语言模型在穷尽式重建客户人生事件历史与34维财务状态方面的长程记忆能力，发现随会话累积事件召回率显著下降（0.591降至0.445），且错误主要源于事件遗漏。 |
| [^70] | [Athena: Vulnerability-Affected Library Identification via Knowledge Graph Completion](https://arxiv.org/abs/2609.01187) | 提出了首个基于图的方法Athena，将漏洞数据库建模为知识图谱，通过知识图谱补全和链接预测自动识别并补全CVE缺失的受影响软件库信息。 |
| [^71] | [Jailbreaking Text-to-Image Models Through Cracks: Navigating Heterogeneous Safety Filters via Multi-Agent Debate](https://arxiv.org/abs/2609.01168) | 该论文提出“检测表面”几何框架来刻画异构安全过滤器的决策边界，揭示了跨层冲突塑造的稀疏非凸规避区域，并基于此通过多智能体辩论方法实现对文本到图像模型多层安全防护体系的越狱攻击。 |
| [^72] | [Superposed Latent Autoencoder](https://arxiv.org/abs/2609.01158) | SLAE 通过学习到的叠加机制将多个高容量潜码绑定并叠加存储在单个内存张量中，以可抑制的结构化干涉取代传统自编码器不可逆的维度瓶颈，在相同存储预算下将重建误差最多降低 56%。 |
| [^73] | [StainPresetNet: Stain Preset Network for Fast Multi-to-Multi Stain Normalization](https://arxiv.org/abs/2609.01146) | 提出StainPresetNet框架，通过预设参考图像引导实现快速、高效的多对多染色归一化，在保持组织结构的同时实现准确的数据集级颜色映射，且改变归一化方向无需重新训练模型。 |
| [^74] | [Revisiting Face Recognition for Monozygotic Twins: The Celeb Twins Test Set](https://arxiv.org/abs/2609.01141) | 本文提出首个包含独特皮肤标记和镜像不对称性元数据的名人双胞胎测试集CTTS，实验表明当前深度CNN匹配器虽能达到76%以上的准确率，但并未利用这些区分性特征，并探讨了用生成式AI合成虚拟同卵双胞胎图像来扩充训练数据的可行性。 |
| [^75] | [DNC-IMM: Early Lane-Change Intention Recognition via Neural Calibration Based on Driving Context Information](https://arxiv.org/abs/2609.01120) | 本文提出双神经校准交互多模型DNC-IMM，利用神经网络基于驾驶上下文信息（目标车辆运动、周围车间距与相对速度）校准转移概率矩阵和测量似然，在保留传统IMM概率结构与可解释性的前提下，实现换道意图的早期可靠识别，并在2-3秒预测时域内性能尤为突出。 |
| [^76] | [Latent Recurrent Thoughts: Recurrent Refinement of Proposed Latents for Reasoning with Frozen LLMs](https://arxiv.org/abs/2609.01117) | 该论文提出潜在循环思维（LRT）方法，通过保持大语言模型冻结并引入一个微型循环推理器在连续潜在空间中多步迭代精炼潜在思维向量来进行推理，将计算深度与模型规模解耦，从而规避了思维链推理中误差传播以及需要可模仿轨迹的固有局限。 |
| [^77] | [EDRAC: Benchmarking Arabic Dialect Reading Comprehension](https://arxiv.org/abs/2609.01113) | EDRAC是首个面向阿拉伯语方言机器阅读理解与生成式问答的大规模基准，涵盖埃及、摩洛哥、阿联酋、叙利亚和沙特五种主要方言，包含499篇自然口语段落和通过人-大语言模型协作流水线生成的4,977个问答对，并以此评测了阿拉伯语和多语言大语言模型的表现。 |
| [^78] | [Hints Help But Do They Teach? Evaluating Skills Transfer in Code Generation](https://arxiv.org/abs/2609.01106) | 研究发现，提示对失败代码生成的“挽救”效果大多可通过无提示的重复采样复现，且相关与无关提示共享同一激活方向，表明提示更多是引导模型已有能力而非传授新技能。 |
| [^79] | [StateSwap: Probing Support-Elimination Hidden States in Multiple-Choice Questions](https://arxiv.org/abs/2609.01081) | 该论文提出StateSwap方法，通过添加特殊标记[STATE]来探测并交换多选题在“支持型”与“排除型”两种表述下诱导出的隐藏状态激活，证明两种框架在模型中间层产生可分离的内部表示，且交换这些激活可因果性地改变预测结果并提高跨框架答案一致性。 |
| [^80] | [Text-guided flow matching enables sample-efficient crystal structure generation](https://arxiv.org/abs/2609.01076) | TFMat提出了一种以结构化材料文本作为语义先验的文本条件化流匹配框架，显著提升了晶体结构生成的匹配率（MP-20基准20个候选下达92.04%），实现了样本高效且可通过自然语言控制的晶体生成。 |
| [^81] | [Space Generative AI with Solar Energy Harvesting](https://arxiv.org/abs/2609.01062) | 本文提出了一个太阳能驱动的太空生成式AI框架，揭示了在共享收集能量预算下，扩散模型图像生成步数与下行传输之间的基本计算-通信权衡关系。 |
| [^82] | [ARISE-RL: Agentic Rubric-Grounded Iterative Self-Evolution with Reinforcement Learning](https://arxiv.org/abs/2609.01058) | ARISE-RL 提出了一种通过量规介导的生成器与求解器协同进化的全周期自我进化框架，解决了开放式智能体强化学习训练中缺乏可验证答案且奖励信号脆弱不稳定的问题。 |
| [^83] | [User Representation via Cross Multi-source Behavior Pre-training for Mobile Games](https://arxiv.org/abs/2609.01057) | 该论文提出CM-PTM模型，通过层级级联的掩码预测预训练任务，在设备级行为日志上对跨多源用户行为进行统一建模，从而学习更精准的移动游戏用户表示，突破了传统单应用建模的局限。 |
| [^84] | [WorldBench: Culturally Grounded Benchmark for Multilingual Agents](https://arxiv.org/abs/2609.01056) | WorldBench是一个涵盖七种语言、八种文化、包含1,600个真实日常任务的多语言智能体基准，并引入约束任务成功率（CTS）指标，以全面评估LLM智能体在真实文化扎根场景中的跨语言多步骤任务执行能力。 |
| [^85] | [QILP-0: Constructing Observational Declarative Twins of Quantum Circuits](https://arxiv.org/abs/2609.01049) | 本文提出QXymb通用框架及其首个完整的零阶特化QILP-0，能够从量子电路的观测行为中自动构建有限多值命题逻辑程序，实现量子电路的观测性声明式孪生体构建。 |
| [^86] | [Lagged Coupling: Internal Representations Become Readable Before They Become Causal](https://arxiv.org/abs/2609.01048) | 该研究在 Pythia 全系列模型中发现“滞后耦合”现象：线性探针能极早读出内部表征，但利用这些方向进行引导干预却几乎无效，且“可读但不可因果干预”的滞后并不随模型规模增大而缩小。 |
| [^87] | [HiveTraceGuard-Pro: A Compact Generative Guardrail for Prompt Injection, Jailbreaks, and Adversarial Obfuscation](https://arxiv.org/abs/2609.01046) | 该论文提出了HiveTraceGuard-Pro，一个基于Qwen3-0.6B经LoRA微调的0.6B参数紧凑型生成式防护栏，可检测俄语和英语环境下的提示注入、越狱攻击及对抗性混淆，在十九个基准组的评测中取得0.7432的综合得分，性能接近更大的防护模型。 |
| [^88] | [AgentFactory: Towards Automated Agentic System Design and Optimization](https://arxiv.org/abs/2609.01045) | AgentFactory是一个在性能、成本和效率等多目标约束下，利用大语言模型作为优化器，对智能体系统中的基础模型和工作流结构进行联合自动化优化的框架。 |
| [^89] | [From Truncation to Commitment: Persistent Context in Uniform Discrete Diffusion](https://arxiv.org/abs/2609.01043) | 提出一种无需训练的承诺式揭示采样（CRS），将选定的词元作为持久上下文插入后续模型输入，使均匀离散扩散模型的并行预测能在序列级选择上保持一致。 |
| [^90] | [ViTAMINS: An Empirical Study of Training Self-Supervised Vision Transformers with Synthetic Hard Negatives](https://arxiv.org/abs/2609.01041) | ViTAMINS通过向自监督视觉Transformer的对比学习预训练中引入合成困难负样本，以极小的改动获得了涌现的语义分类能力（最高提升11.3%）并大幅节省计算资源（ViT-B超越ViT-L的V-JEPA），证明对比学习仍是生成式与自蒸馏方法的强大替代方案。 |
| [^91] | [Causal Evidentiary Governance for High-Risk Machine Learning Systems](https://arxiv.org/abs/2609.01040) | 提出因果证据治理（CEG）框架，通过版本化因果图、因果伤害率指标和密码学绑定的决策证据包，为高风险机器学习系统实现因果归因与高效证据验证。 |
| [^92] | [Data-Driven Persona-Conditioned Agents for A/B Test Simulation](https://arxiv.org/abs/2609.01038) | 该论文提出利用基于真实用户匿名行为数据（活动模式、参与度信号、人口统计特征）构建的数据驱动角色来条件化LLM智能体，从而更忠实地模拟用户群体并预测A/B测试结果，替代昂贵耗时的真实用户实验。 |
| [^93] | [Spawn Freely, Act Sparingly: Progressive Risk Vesting for Recursive LLM-Agent Trees](https://arxiv.org/abs/2609.01035) | 提出渐进式风险授予（PRV）机制，通过托管轨迹级风险预算并在分支激活时逐步扣减，为递归LLM智能体树中不可逆行动的授权证明了任意时刻的危害上界，实现“自由派生、节制行动”的安全权衡。 |
| [^94] | [On Synthesis of Metric Interval Temporal Logics](https://arxiv.org/abs/2609.01032) | 本文提出了首个面向度量区间时序逻辑（MITL）的精确被动学习框架，通过将正负轨迹间的定量时间差异转化为新的布尔原子命题，把时间学习问题归约为可扩展的非时间LTL学习问题，从而无需预定义模板或受限逻辑片段即可自动挖掘时间规范。 |
| [^95] | [A Network Science Perspective on Evaluating Deep Graph Generative Models](https://arxiv.org/abs/2609.01015) | 该研究从网络科学视角评估深度图生成模型，通过生成网络与真实网络的拓扑相似性及其在识别有效节点免疫策略以抑制流行病/错误信息传播方面的实用性进行综合衡量。 |
| [^96] | [Figures as Programs: Recursive Generation of Editable Scientific Figures](https://arxiv.org/abs/2609.01006) | 该论文提出了FigTree多智能体系统，将科学图形生成形式化为递归的SVG程序构建，通过将图形分解为区域层次结构并逐个生成SVG程序片段，实现了从科学论文自动生成结构化、可精确编辑的矢量图形。 |
| [^97] | [SinkPruner: Sink-Free Visual Token Pruning for Multimodal Large Language Models](https://arxiv.org/abs/2609.01004) | 提出无需训练的视觉token剪枝框架SinkPruner，通过过滤高度冗余的高范数离群token并缓解注意力汇聚现象，在保持多模态理解能力的同时实现高效的多模态大语言模型推理。 |
| [^98] | [Right Frame, Wrong Rule: Cultural Cues Expose the Financial Knowledge Gap They Were Meant to Close](https://arxiv.org/abs/2609.00999) | 该论文提出“规范多元性”这一新评估设定，通过将框架选择与框架内正确性分离，揭示了“刻板印象陷阱”——文化线索虽能引导大模型选择伊斯兰金融框架，却在框架内暴露出高达57%至66%的错误率，表明传统二选一评估会严重高估模型的文化对齐能力。 |
| [^99] | [Inspicio: Open-Vocabulary, LLM-Based Sense Retrieval for Historical Languages](https://arxiv.org/abs/2609.00998) | 提出了Inspicio，一个无需源语言词义清单的开放式词汇检索流水线，利用大语言模型生成的英文翻译、候选定义和词元，通过混合检索将历史语言文本中的词元直接链接到开放英语WordNet的同义词集。 |
| [^100] | [On the Human and Computer Alignment of Attribute-Based Music Matches](https://arxiv.org/abs/2609.00987) | 该论文通过包含剽窃案例、翻唱歌曲和AI生成音乐的感知实验，构建了MATCHA数据集，系统研究了计算相似度度量与人类判断在旋律、和声、节奏、人声和音色五种音乐属性上的一致性。 |
| [^101] | [Semi-Supervised Virtual Staining via Morphology Preservation and Histopathological Realism Constraints](https://arxiv.org/abs/2609.00984) | 该论文提出一种半监督虚拟染色框架，利用Hessian形态学保持机制和组织病理学真实性约束，充分利用有限的配对数据与大量非配对源图像实现稳定的虚拟染色，从而摆脱对严格配准训练数据的依赖。 |
| [^102] | [Disclosure-Gated User Simulation for Companion-Agent Evaluation](https://arxiv.org/abs/2609.00982) | 提出披露门控用户模拟方法，让模拟用户根据陪伴型智能体的行为决定信息披露深度，以纠正模拟用户过度配合、使被测系统仅靠提问数量即可得分的评估缺陷。 |
| [^103] | [The zbMATH Open Knowledge Graph: Tracing Centuries of Mathematical Research](https://arxiv.org/abs/2609.00969) | zbMATH开放知识图谱是一个涵盖250多年数学研究、包含3400万实体和1.68亿RDF三元组的大规模知识图谱，其创新之处在于整合了专家策划的语义内容（评论、关键词、学科分类、消歧作者信息等），突破了传统学术知识图谱仅记录书目元数据和引用结构的局限，支持对数学概念与学术关系演变的细粒度时序分析。 |
| [^104] | [CoBRA: Learning Tool-Use Boundaries via Counterfactual Margins](https://arxiv.org/abs/2609.00967) | 提出了CoBRA框架，通过从同一基础模型构建内部/外部专家并计算“使用与不使用工具”两种回答间的反事实奖励边际，来学习每个查询是否应调用工具的实例级边界，从而在避免不必要工具调用的同时不遗漏真正需要工具的场景。 |
| [^105] | [Few-Shot Out of Domain Intent Detection with Covariance Corrected Mahalanobis Distance](https://arxiv.org/abs/2609.00961) | 本文分析了马氏距离方法在少样本设置下域外意图检测性能不佳的原因，并提出了一种协方差修正的马氏距离来有效检测域外意图。 |
| [^106] | [Calibration is the Bottleneck: An Action-Class Diagnostic of Multi-Turn Tool-Calling](https://arxiv.org/abs/2609.00949) | 本文提出一个基于四类动作空间的诊断框架，通过引入“准确率不超过黄金动作召回率”的自揭示上界，将多轮工具调用失败分解为动作类别失准与动作执行失败两种正交模式，从而揭示开源模型总体准确率追平闭源模型的表象背后，动作类别校准才是真正的瓶颈。 |
| [^107] | [From Terminology to Diagrams: Visual-Instruction Generation for Scientific Diagram Understanding](https://arxiv.org/abs/2609.00948) | 该论文提出SciGram框架与数据集，通过科学课程术语自动生成涵盖19.4万张图表和140万条视觉指令的大规模训练数据，显著提升了视觉语言模型在科学图表理解任务上的表现。 |
| [^108] | [Embedded Conditional Independence Tests for Large Language Model Generated Text with an Application to German Parliament Speeches](https://arxiv.org/abs/2609.00946) | 本文提出嵌入式条件独立性检验（eCITs），通过将LLM生成的文本及其源文本嵌入到表示空间后再进行条件独立性检验，从而判断模型输出是否携带源文本之外的额外信息，并将其应用于德国议会演讲数据的分析。 |
| [^109] | [DualStake: Dual-Path Confidence Calibration in Deep Research Agents](https://arxiv.org/abs/2609.00935) | 提出DualStake双路径置信度校准方法，通过在每次检索后引出证据置信度并在答案生成后引出答案置信度，利用边界裁剪的置信度相关stake奖励将两者与答案正确性联合对齐，有效缓解深度研究智能体的严重过度自信问题。 |
| [^110] | [Context-Grounding Gains Are Mediated by Pre-existing Machinery: Auditing GRPO, SFT, and DPO](https://arxiv.org/abs/2609.00925) | 本文通过从同一检查点系统审计GRPO、SFT和DPO共九种后训练方案，发现语言模型遵循冲突提示证据的接地增益主要源于强化模型中已有的机制（与起始模型相同的因果注意力头集合），而非学习新机制，其中GRPO增益很小、冲突SFT提升适中、DPO在其匹配分布上接近上限。 |
| [^111] | [Beyond the Image Plane: World-Grounded Queries for Multi-Object Tracking](https://arxiv.org/abs/2609.00924) | 提出PLANET，一种通过将重建的3D场景几何嵌入查询特征与位置编码、从而超越图像平面的端到端多目标跟踪器，在三个基准测试中达到最先进性能。 |
| [^112] | [VIBE-Bench: Evaluating Personalized Large Language Models When Profiles Don't Mean Preferences](https://arxiv.org/abs/2609.00921) | 该论文提出了VIBE-Bench基准，揭示当前个性化大语言模型在“画像-偏好概念错位”情形下（即用户画像线索与查询偏好处于不同概念空间时）因过度依赖浅层语义关联而失效，需要具备超越表面语义的跨概念偏好推理能力。 |
| [^113] | [RPCBench: A Benchmark for Proactive Premise Critique in LLM-based Recommendation](https://arxiv.org/abs/2609.00918) | 该论文提出了 RPCBench 基准，首次系统评估大语言模型在推荐场景中主动检测、诊断并妥善处理用户请求中错误前提的能力，涵盖五个推荐领域、十种前提失败类型，并提供了细粒度的评估框架。 |
| [^114] | [In-Context Neurofeedback: Can LLMs Control Their Internal Representations through Privileged Access?](https://arxiv.org/abs/2609.00904) | 本研究重新设计了要求“特权访问”的大语言模型神经反馈范式，发现模型在更严格设置下无法可靠控制其特权内部表征，表明此前报告的模型自我控制能力可能仅依赖表面机制。 |
| [^115] | [Vision-Language-Guided Pseudo-Labels for Unsupervised Domain Adaptation in Semantic Segmentation for Waste Sorting](https://arxiv.org/abs/2609.00898) | 该论文提出了一种利用SAM、EVA-CLIP和BLIP等视觉-语言基础模型生成跨模态伪标签的流水线，无需任何目标域标注即可实现垃圾分拣语义分割的无监督域自适应。 |
| [^116] | [CARE: Contrastive Anchor-based Rubric Evolution for Large Language Model Post-Training](https://arxiv.org/abs/2609.00892) | 提出CARE方法，通过将模型得分最高的回答与前沿模型生成的锚点回答进行对比，实现了自适应修复奖励误设和主动追踪改进两个互补的评分标准动态演化机制，解决了基于评分标准的强化学习中静态评分标准易被策略钻空子的问题。 |
| [^117] | [CacheBridge: Efficient Cross-Model KV Cache Transfer](https://arxiv.org/abs/2609.00891) | CacheBridge通过将目标头与匹配源头进行架构索引配对、以因果注意力敏感度加权重建误差，并使用融合GPU内核，实现了对架构差异鲁棒且成本可控的免训练跨模型KV缓存迁移。 |
| [^118] | [Denoising Diffusion Generative Models Secretly Calculate Attentions](https://arxiv.org/abs/2609.00885) | 该论文发现去噪扩散模型本质上暗中使用了与Transformer类似的注意力机制，从而证明注意力是机器学习的普适性原理，并据此提出了基于注意力机制的简化图像生成算法，以减少训练时间和计算开销。 |
| [^119] | [Towards reliable multimodal disaster severity assessment through preference optimization and explainable vision-language reasoning](https://arxiv.org/abs/2609.00879) | 该论文提出了一种整合SFT与DPO的两阶段训练框架，并从单次人在回路标注流程中构建推理与偏好两个数据集，同时提升了多模态灾害严重程度评估的准确性（73.64%→78.29%）与解释质量。 |
| [^120] | [FractalNet-Based Heterogeneous Federated Learning for Orbital Edge Intelligence in Satellite Mega-Constellations: A Wildfire Case Study](https://arxiv.org/abs/2609.00875) | 该论文提出一种基于FractalNet的异构联邦学习方法，通过分布式路径调度器根据卫星SWAP-C约束与预测星间通信机会动态分配模型深度，并结合周期性更新汇聚和三层智能体控制平面，实现了卫星巨型星座中适应异构硬件条件的轨道边缘智能。 |
| [^121] | [Beyond the Clock: Measuring the Value of Adaptive Revision](https://arxiv.org/abs/2609.00874) | 该论文在分层智能体系统中研究了元级控制问题，发现学习到的自适应修正时机策略在不同训练种子下产生性质迥异的行为，但均未能超越最佳的强制时机策略，从而首次将状态依赖性与实际决策价值区分开来。 |
| [^122] | [Benchmarking Vision-Language Models for Automated Pathology Diagnosis and Report Generation](https://arxiv.org/abs/2609.00866) | 该研究构建了来自五个机构、约10,500对样本的泛亚洲病理全切片图像-报告数据集，并建立REG 2025基准系统评估多模态模型，发现顶尖方法的关键在于结构化报告表示而非单纯依赖视觉-语言模型。 |
| [^123] | [Reinforcement Learning Enhanced LLM Agents for Complex Vehicle Routing Problems](https://arxiv.org/abs/2609.00859) | 本文提出RLEA多智能体框架，利用Soft Q-learning训练的轻量级神经规划器协调大语言模型智能体，并结合进化记忆模块与检索增强生成技术，实现复杂车辆路径问题的自动化建模与求解。 |
| [^124] | [Verifiable Disaster Storylines and Causal Knowledge Graphs: A Citation-Grounded Pipeline from Heterogeneous Humanitarian Sources](https://arxiv.org/abs/2609.00858) | 该论文提出了一个基于检索增强生成（RAG）的流水线，融合EM-DAT结构化灾害记录与ReliefWeb、EMM非结构化文档，自动生成涵盖17个字段的灾害故事线和因果知识图谱，且每个节点和边均附带引用溯源，实现了对原始信息源的完全可追溯性，为人道主义响应提供可验证的态势感知支持。 |
| [^125] | [Does Fault Localization Beat a Fresh Attempt? A Placebo-Controlled Study of Test-Guided Code Repair](https://arxiv.org/abs/2609.00854) | 该安慰剂对照研究发现，故障定位在实际场景中很少可用（仅约9%的失败候选可定位），且即便可定位，基于频谱定位的片段填充修复也显著劣于盲目的整体重采样。 |
| [^126] | [ADGNet: Asymmetric Dual-text Guided Network for Infrared Small Target Detection](https://arxiv.org/abs/2609.00853) | 该论文提出非对称双文本引导网络ADGNet，通过设计与图像无关的抽象目标提示和与图像相关的详细背景提示，并利用非对称双分支交互模块分别处理，解决了红外小目标检测中背景抑制信息不足和特征优化冲突的问题。 |
| [^127] | [A Checklist to assess the energy and carbon impacts of ML/AI applications in Earth System Modeling](https://arxiv.org/abs/2609.00847) | 该论文将分散的机器学习/人工智能可持续发展讨论提炼为一份按模型开发流程各阶段组织的实用清单，帮助地球系统科学从业者评估并降低其应用的能耗与碳足迹，并配套提供了估算能耗和碳排放的指标。 |
| [^128] | [Towards Generalizable Visually Grounded Exploration of Household Devices](https://arxiv.org/abs/2609.00845) | 该论文提出了VGEBench基准，用于评估智能体在无说明书和标注轨迹的情况下，通过动态的“假设-交互-修正”过程将抽象世界知识接地为细粒度视觉可供性，从而操作新型家用设备的可泛化探索能力。 |
| [^129] | [Probabilistic Model Checking of Autoregressive Neural Sequence Models](https://arxiv.org/abs/2609.00838) | 本文提出一种基于概率模型检测的验证流程，从自回归神经序列模型的逐token生成中提取DTMC并用PRISM验证PCTL规约，从而给出约束违反概率的认证区间和构造性保守的输入空间覆盖率曲线，并借助CEGAR循环自适应收紧区间。 |
| [^130] | [Replacing Training with Memory: Listwise Selection for Text-to-SQL](https://arxiv.org/abs/2609.00834) | 该论文提出MaP-SQL，一种无需微调的Text-to-SQL列表式选择器，通过从训练数据蒸馏的可复用结构化记忆替代学习选择标准，并利用排名聚合缓解位置偏差，从而以更低成本实现候选查询选择。 |
| [^131] | [FLaG: Frequency-Domain Latent-attention Gated Pooling for Token Aggregation](https://arxiv.org/abs/2609.00831) | FLaG是一种即插即用的池化聚合模块，通过在傅里叶域中重新表达编码器输出、利用潜在注意力查询与样本条件通道门控进行标记聚合，在蛋白质、图像和语言等多种任务上均取得最优表现。 |
| [^132] | [Visual Attention Faithfulness in Vision-Language Models is Heterogeneous](https://arxiv.org/abs/2609.00830) | 该论文通过因果扰动分析首次系统研究了视觉-语言模型中的视觉注意力忠实性，发现其具有异质性，并归纳出忠实-充分、忠实-分布式和非聚焦三种不同的处理模式。 |
| [^133] | [HarnessEvolve: Learning from Reference Trajectories for Reliable Agent Self-Evolution](https://arxiv.org/abs/2609.00829) | HarnessEvolve 提出了一种从参考轨迹中学习的智能体自我进化框架，通过将执行、评估、优化和门控解耦为独立模块，克服了信用分配失败、捷径学习和灾难性遗忘三大难题，实现了可靠且可泛化的智能体自我进化。 |
| [^134] | [Polished but Unresolved: Identifying Late-Stage Pressure States in Long-Horizon Tool-Use Agents](https://arxiv.org/abs/2609.00823) | 该论文首次识别出长时程工具使用智能体的“后期压力状态”（即倾向于提交看似完整精美但关键约束尚未解决的答案），证明该状态可通过线性探针从隐藏状态中检测、可被激活干预因果地改变，并据此提出PSPR插件以自适应方式缓解压力、改善提交决策。 |
| [^135] | [AnalysisBank: An Expert Analysis Pattern Library for Financial Report Generation](https://arxiv.org/abs/2609.00818) | 提出了AnalysisBank，将专家财务报告提炼为可复用的“数据信号-分析手法”分析库，在推理时通过检索匹配的分析手法生成报告，使新颖且有数据支撑的洞察比例较结构层面基线提升1.7-3.7倍，且该方法可泛化至科学写作领域。 |
| [^136] | [One Policy, Any Budget: Internalizing Budget-Aware Search via Reinforcement Learning](https://arxiv.org/abs/2609.00813) | 该论文提出 AnySearch 框架，通过“脚手架引导到自主运行”的两阶段课程强化学习，以及耦合答案准确性与预算效率的复合奖励，使单一策略能够在任意预算约束下执行预算感知搜索。 |
| [^137] | [Ctrl-F-Resist. Practices, Challenges, and Technical Needs of Civil Society Organizations Monitoring the Far-Right Online](https://arxiv.org/abs/2609.00808) | 本文通过对12家德国公民社会组织的15名从业者进行定性研究，揭示了这些组织在极右翼在线监测工作中的长期实践、面临的挑战（法律不确定性、平台访问受限、资金不足）以及技术需求，强调它们是数字治理中被忽视的关键利益相关者。 |
| [^138] | [Towards a Reliable and Practical Eval Pipeline](https://arxiv.org/abs/2609.00805) | 该论文提出了一种结合评估检查清单创建与学习式聚合的端到端评估流水线，提高了LLM评判者之间的一致性及与人类判断的吻合度，同时提供自一致性、解释和预测不确定性。 |
| [^139] | [Agentic programs: an emerging form of scientific software in computational materials science](https://arxiv.org/abs/2609.00795) | 本文提出“智能体程序”这一新兴科学软件范式，将确定性算法与有界的LLM科学判断、任务验证和阶段性成熟相结合，并以从实验测量的无序晶体结构构建原子模型的DeMARS系统为例加以展示。 |
| [^140] | [MADS: A Multiview Acoustic Descriptor Set Beyond Standard Spectral Summaries](https://arxiv.org/abs/2609.00792) | 本文提出了MADS，一个紧凑的19维物理信息驱动的多视角声学描述符集，通过统一表示编码声音的激励、阻尼、周期性、冲击性等物理特性，突破了传统频谱摘要仅将声音视为频谱模式的局限，并在多个音频分类数据集上与经典手工特征基线进行了对比评估。 |
| [^141] | [Instella-MoE Technical Report](https://arxiv.org/abs/2609.00791) | Instella-MoE 是一个完全开源、总参数160亿（激活28亿）的混合专家语言模型，完全基于 AMD GPU 从零训练，凭借 Gated MLA 与 FarSkip-Collective 等架构与系统级创新实现了高效训练推理，并在基准测试中超越 OLMo-3-7B 等此前完全开源模型。 |
| [^142] | [StudyBench: Can Self-Evolution Squeeze Textbooks for Olympiad Capability?](https://arxiv.org/abs/2609.00787) | StudyBench是一个受控物理基准，用于衡量自我进化方法将教科书式训练材料转化为奥赛级解题能力的效率，研究发现模型在困难教科书问题上的提升很难迁移到奥赛级问题上。 |
| [^143] | [When Features Become Instances: Inverted Contrastive Learning for Unsupervised Feature Selection](https://arxiv.org/abs/2609.00782) | 该论文提出ICLFS框架，通过倒置数据矩阵使特征成为对比学习中的实例，并利用掩码正视图、打乱负视图和InfoNCE目标，将无监督特征选择重新表述为特征层面的表示一致性学习问题。 |
| [^144] | [Solaris: Towards Interfaces That Are Generated, Not Coded](https://arxiv.org/abs/2609.00776) | Solaris 提出了一个界面世界模型，将鼠标交互作为条件信号、自回归逐帧直接生成交互式 UI，并结合少步蒸馏与语言模型解释用户意图，实现了无需代码、实时生成界面外观与行为的新范式。 |
| [^145] | [VOIM: Training-Free Open-Vocabulary 3D Instance Mapping for RGB-D and Monocular SLAM](https://arxiv.org/abs/2609.00775) | VOIM是一个免训练的开放词汇3D实例建图系统，它将标签与实例决策推迟到多视角软证据按体素累积之后，仅凭RGB-D或单目RGB即可实现，其建图质量（mIoU）比最强的在线RGB-D系统OVO-SLAM高出4.8至11.7，证明了建图阶段而非感知模型才是决定性因素。 |
| [^146] | [DiagEvo: Diagnosis-Guided Self-Evolution via Hierarchical Error Memory](https://arxiv.org/abs/2609.00768) | DiagEvo通过诊断器从语言模型自身的失败历史中提取反复出现的错误原因，构建分层错误记忆来引导自我博弈进化，无需依赖外部任务信息即可持续提升求解器性能。 |
| [^147] | [Are You Thinking What I am Thinking? : Examining Conceptual Separation in Neural Architectures](https://arxiv.org/abs/2609.00764) | 本研究通过对CNN和LLM内部激活的几何与分布分析，揭示神经网络中存在“概念分离”现象——同一概念形成连贯表示、相关概念在表示空间中彼此更近，但这种连贯性在未见概念、域偏移和模糊主题下会减弱或坍塌。 |
| [^148] | [Automated Tree Knowledge Graph Construction using Ontology Expansion and Retrieval from Vietnamese History Textbooks](https://arxiv.org/abs/2609.00763) | 本文提出了一种从越南历史教科书自动构建树状知识图谱的端到端流水线，通过并查集批内去重、近似跨批搜索和带质心过滤的双LLM验证等三阶段混合关系抽取方法，解决了低资源语言本体扩展和层次化检索策略系统评估的难题。 |
| [^149] | [S^3martCirc: Self-supervised Smart Circuit Discovery](https://arxiv.org/abs/2609.00755) | 提出S^3martCirc，一种自监督的智能电路发现方法，将电路发现与功能解释两个阶段统一起来，解决了组件重要性与功能角色相互依存这一被传统两阶段范式忽视的问题。 |
| [^150] | [ContextPipe: Database-Inspired Context Assembly for Long-Horizon Agents](https://arxiv.org/abs/2609.00749) | 该论文提出 ContextPipe，将长程智能体的上下文组装类比为数据库查询执行，通过“计划-绑定-优化-执行-反馈”五阶段流水线，结合结构化数据源目录、确定性缓存感知优化器和 EXPLAIN ANALYZE 追踪，实现可审计、可重放且故障隔离的上下文管理。 |
| [^151] | [Escaping Redundant Reasoning: Structure-Aware Search for Inference-Time LLMs](https://arxiv.org/abs/2609.00738) | 提出无需训练的结构感知搜索方法BASIN，通过将推理状态分组为盆地并惩罚重复策略来避免“推理盆地坍缩”，在固定计算预算下显著提升LLM推理时搜索性能（Game of 24上较ToT提升达22个百分点）。 |
| [^152] | [Agentic Empirical Asset Pricing: Methodological Foundations](https://arxiv.org/abs/2609.00731) | 本文提出了智能体化实证资产定价（AEAP）这一新范式，为自主因子发现系统提供了参考架构、严格的因子评估标准与样本外回测方法，并通过对SEADS与五个基线系统的评估证明此类系统必须从多个维度同时加以评价。 |
| [^153] | [SOVER: Formal Certification of Optimization Reformulations via LLM-Assisted SMT Verification](https://arxiv.org/abs/2609.00728) | SOVER框架将语义映射与形式化验证分离，利用Z3和dReal等SMT求解器对LLM生成的优化问题重构进行形式化认证，并在NLEquiv-150基准上实现了149/150的正确分类。 |
| [^154] | [Heard but Not Heeded: Paralinguistic Information Encoding and Loss in Audio-Language Models](https://arxiv.org/abs/2609.00727) | 该研究通过对四个开源音频语言模型的机制分析首次系统揭示了：尽管模型在音频编码器后期强烈编码了说话风格等副语言信息，但这些信息在传递至输出的过程中持续丢失，暴露出当前音频语言模型“听得见却不重视”语气情感的关键缺陷。 |
| [^155] | [A Closed-Loop Evaluation of Capability Loss and Recovery in Compressed Driving Policies](https://arxiv.org/abs/2609.00718) | 本文提出一种分阶段闭环评估方法，追踪驾驶策略在剪枝、蒸馏、量化等压缩流程中的能力损失与恢复，并发现结构化剪枝是驾驶能力首次受损的环节。 |
| [^156] | [ChatDev 2.0: A No-Code Multi-Agent Platform for Developing Everything](https://arxiv.org/abs/2609.00714) | ChatDev 2.0（DevAll）是一个兼具高表达性与易用性的无代码多智能体平台，通过声明式可执行图抽象与循环感知执行引擎支持异构智能体间的动态循环交互，并提供集成可视化界面，让用户无需编写代码即可构建、运行、监控和检查多智能体系统（包括人在回路步骤）。 |
| [^157] | [Differentially Private Paired Table-Image Multimodal Synthesis](https://arxiv.org/abs/2609.00708) | 提出DP-TabImage框架，通过将私有概率图模型与经DP-SGD训练的表格条件扩散模型相结合，实现了差分隐私下表格-图像配对多模态数据的隐私保护合成。 |
| [^158] | [Value Over Language Model: Detecting Original Contribution in Writing](https://arxiv.org/abs/2609.00700) | 提出了一种无需训练、不评分表面文本的新框架“价值超越语言模型”（Value Over Language Model），通过在不同粒度上提取文档内容并用LLM重建文档，来衡量人在语言模型易于生成的内容之上所贡献的原创价值。 |
| [^159] | [A Study of Hidden-State Optimization Order in Predictive Coding Networks](https://arxiv.org/abs/2609.00686) | 该论文提出一种边界优先的隐状态优化顺序——先协调块边界处的隐状态、再细化块内表示，显著提升了预测编码网络在CIFAR-10上的准确率并增强了早期层的特征学习。 |
| [^160] | [Visual Framing for News Stance Detection via Image Generation](https://arxiv.org/abs/2609.00685) | 该论文提出VFStance方法，通过图像生成技术将新闻文章中隐含的立场线索转化为视觉框架，使立场信号更加明确显著，有效提升了新闻立场检测性能，并具有超越自动化立场检测的应用潜力。 |
| [^161] | [Triple-Bottom-Line Sustainability of Language Models for Edge AI: A Comparison Between SLMs and Quantized LLMs](https://arxiv.org/abs/2609.00665) | 该研究提出了一个基于经济、环境、社会三重底线的可复现整体可持续性评分（HSS），通过对比原生训练的小型语言模型与多种量化后的大语言模型共30种配置在能力、效率和安全性方面的表现，来评估哪种方案更适合作边缘AI部署。 |
| [^162] | [Drift-Aware LLM Routing with Sparse Contexts and Shared Budgets](https://arxiv.org/abs/2609.00662) | 本文提出漂移感知稀疏路由方法（DRS），针对多模型大语言服务中提示表示高维稀疏且请求分布与模型能力持续漂移的挑战，通过滚动审计窗口、悲观奖励与乐观成本估计以及在线影子价格更新，实现了满足共享预算约束的非平稳上下文路由。 |
| [^163] | [SciTrue: Reliable Scientific Claim Validation with Frontier and Open Language Models at the NTCIR SciClaimEval Task](https://arxiv.org/abs/2609.00654) | SciTrue团队通过在统一诚实的逐样本协议下对十一个前沿及开源多模态模型进行基准测试，并结合轻量透明的后处理，在NTCIR SciClaimEval科学论断验证任务的官方盲测排行榜上以明显优势夺得第一。 |
| [^164] | [EEG-AS: Instance-Level Foundation Model Selection for EEG Foundation Models via Behavior Reconstruction](https://arxiv.org/abs/2609.00653) | 该论文提出EEG-AS框架，将EEG基础模型选择形式化为实例级算法选择问题，通过锚定模型的特权预测标记重构不可获得的基础模型行为，从而为每个EEG实例自动选择最合适的基础模型。 |
| [^165] | [Self-Reports Are Not Verification: Environment-Grounded Auditing of LLM Operators in Evolutionary Search](https://arxiv.org/abs/2609.00652) | 本文提出首个环境锚定的LLM操作者审计框架，通过为进化式Contexto搜索中的每个中间提议赋予精确结果，实证证明模型自我报告不可作为验证依据——操作者将成功率夸大4.8至9.3倍，且关于置信度校准、理由传递和适应度选择的三个假设全部被证伪。 |
| [^166] | [DramaChain Bench: An End-to-End Benchmark for Short-Drama Generation](https://arxiv.org/abs/2609.00646) | DramaChain Bench是首个评估短剧制作全流程各阶段（剧本、分镜、关键帧、镜头视频到成片）的端到端基准测试，通过统一的63维度评估体系衡量各阶段对剧本意图的忠实度以及多集成片的连贯性。 |
| [^167] | [REVISE: Validity-Guided Recovery for Online Revisions in Agent Workflows](https://arxiv.org/abs/2609.00643) | REVISE 提出了一种有效性引导的细粒度恢复运行时，通过将修订差异与已记录的数据和控制依赖求交并在部分执行的 DAG 上传播影响，从而在并发智能体工作流中平衡正确性与效率，避免丢弃未受影响的进展。 |
| [^168] | [TUTTI: Toward generalizable audio-to-score transcription via fully synthesized data](https://arxiv.org/abs/2609.00640) | 该论文提出TUTTI预训练范式，利用符号音乐生成模型构建大规模纯合成的多乐器音频-乐谱配对数据，突破真实标注数据稀缺的限制，从而实现可泛化的音频到乐谱转录。 |
| [^169] | [Breaking the Structural Identity: Personalized Federated LoRA Fine-tuning under Rank Heterogeneity](https://arxiv.org/abs/2609.00632) | 提出FedRoRA框架，通过将LoRA适配解耦为共享的全局方向与个性化的按秩幅值，在秩异构联邦学习场景下实现细粒度的客户端个性化微调，从而同时应对资源异构与数据异构的双重挑战。 |
| [^170] | [Restrict, Don't Retrain: Inference-Time VLM Guidance for Zero-Shot Aerial Segmentation](https://arxiv.org/abs/2609.00628) | 提出一种推理时引导方法，利用单张消费级GPU上的视觉语言模型为冻结的零样本分割基础模型提供类别筛选与小型物体定位指引，无需重训练即可提升航空图像分割效果，并生成可独立查验的结构化证据。 |
| [^171] | [Control-Data Flow Separation: Stable Prompt Optimization in Multi-Agent LLMs](https://arxiv.org/abs/2609.00621) | 该论文提出控制-数据流分离方法，将执行关键协议表示为类型化、经验证的程序对象，使提示词优化器能够改进多智能体大语言模型系统的行为，而不会因提示词修改意外破坏协议导致整个流水线失效。 |
| [^172] | [Towards Effective Structured Context Modeling for Conversational Recommender Systems via Dual-node Monte Carlo Tree Search](https://arxiv.org/abs/2609.00618) | 提出DREAMS框架，通过双节点树结构（引导节点用蒙特卡洛树搜索探索对话动作以推断潜在偏好，利用节点用大语言模型将偏好状态精炼为结构化检索查询），显式建模对话式推荐系统中用户偏好的多轮演化。 |
| [^173] | [Confess What You Know: Forget-Set Misalignment with Model Knowledge in LLM Unlearning](https://arxiv.org/abs/2609.00605) | 提出数据无关的CONFS框架，通过引出模型自身记忆的知识来构建与模型对齐的遗忘集，解决了大语言模型机器遗忘中遗忘集与模型实际记忆内容不对齐所导致的信息泄露或效用下降问题。 |
| [^174] | [SoK: When Safe Agents Fail Together: The Security of Multi Agent LLM Systems](https://arxiv.org/abs/2609.00595) | 该论文通过对197篇文献的执行级系统化分析，提出A-I-R攻击分类框架与五部分防御契约，统一梳理了多智能体LLM系统的攻击与防御机制，并指出路径闭合与恢复是防御的核心挑战。 |
| [^175] | [Socrates went Nuclear: Comparing Interaction Strategies for AI systems in a Learning Context using Brain Sensing](https://arxiv.org/abs/2609.00584) | 该研究让50名零基础参与者学习核安全协议，比较了无限制对话AI、仅给提示不给答案的苏格拉底模式AI、以及基于脑信号实时调节难度的自适应辅导系统这三种人机交互策略在学习中的效果。 |
| [^176] | [Predicting Program Exit Code with LLMs and Programming Language Semantics](https://arxiv.org/abs/2609.00579) | 该论文提出了程序可执行性预测这一新任务，并构建了由有效程序系统性生成无效变换的数据集，以研究大语言模型在判断程序有效性及其违反的形式化语义规则时，究竟是依赖预训练先验知识还是给定的程序语义。 |
| [^177] | [Same Request, Different Boundary: Evaluating Cybersecurity Assistance across Conversational Contexts](https://arxiv.org/abs/2609.00578) | 该论文提出3R-Bench基准，首次在对话语境下评估LLM的网络安全辅助机制，发现先前的拒绝或接受历史会显著改变模型对同一请求的响应（服从率从62.0%升至85.1%），表明安全防护评估必须考虑对话上下文。 |
| [^178] | [GeoPAR: Large-Scale Multi-Agent Combinatorial Optimization with Geometry-Guided Parallel Autoregressive Learning](https://arxiv.org/abs/2609.00577) | GeoPAR提出了一种几何引导的并行自回归强化学习框架，通过投影窗口稀疏几何机制、稀疏边偏置注意力以及缓存引导的冲突处理机制，实现了大规模多智能体组合优化问题的高效求解。 |
| [^179] | [Consistency Without Alignment: Item-Sensitive Language Models Indistinguishable From Random](https://arxiv.org/abs/2609.00576) | 本研究通过可闭式计算基准的强制选择信号任务证明，语言模型的条目敏感性只是任务能力的必要而非充分条件——尽管全部21个“模型×规则”组合都表现出条目敏感性，但其中8个与随机选择在统计上无法区分，5个甚至比随机表现更差。 |
| [^180] | [Residual Sparsification via Output Importance for Compressing Mixture-of-Experts LLMs](https://arxiv.org/abs/2609.00575) | 该论文提出基于输出重要性的残差稀疏化方法，突破了传统上独立最小化每个矩阵压缩误差的局限，在压缩混合专家大语言模型时能更好地保持模型精度。 |
| [^181] | [A Mathematical Framework for Legacy, Governance, and Decision Integrity in Enterprise AI](https://arxiv.org/abs/2609.00572) | 本文提出了一个企业AI机构传承的数学框架，通过基于知识留存、治理、人类监督、适应性、反馈学习和司法忠实度六个维度的惩罚几何平均构建归一化传承分数，并辅以决策置信度与风险模型，量化评估机构在原设计者离开后能否长期保持合法、可解释、可适应的健全决策能力。 |
| [^182] | [VoiceLongMemEval: Do Assistants Remember How You Sounded?](https://arxiv.org/abs/2609.00570) | 该论文提出了VoiceLongMemEval（VLME）基准，用于评估AI助手在长时多会话对话中能否记住情感、韵律和语音事件等副语言信息，发现现有大语言模型普遍存在无法捕捉说话方式的“情感鸿沟”。 |
| [^183] | [WiseSpec: Requirements-Driven Agents for Code Generation](https://arxiv.org/abs/2609.00568) | 该论文提出WiseSpec框架，借鉴软件需求工程思想，通过自动构建高质量结构化需求并结合基于执行的评估进行迭代优化，从而提升大语言模型在仓库级代码生成中的表现。 |
| [^184] | [EEG-VID: Task-Guided Latent Predictive Pretraining for EEG Decoding and Assistive Target Selection](https://arxiv.org/abs/2609.00566) | EEG-VID提出了一种任务引导式潜变量预测预训练框架，通过指数移动平均编码器预测未来EEG潜状态，在42组跨会话跨被试对比中有41组提升准确率（最高提升16.22个百分点），并可有效应用于场景约束下的辅助目标选择。 |
| [^185] | [EM^2Mem: Event-Centric Multimodal Memory for Large Language Models](https://arxiv.org/abs/2609.00551) | 该论文提出EM^2Mem，一种以事件为中心的多模态记忆框架，通过在记忆构建阶段将多模态记录、时间上下文、图谱关系与溯源信息绑定到事件锚点，形成“可直接用于生成”的记忆单元，免去了推理时重建跨模态对齐的负担，并在三个长视频问答基准上将平均准确率较最强记忆基线提升2.0至3.7个百分点。 |
| [^186] | [Runtime-Independent Persistent Agents: Preserving Identity, Memory, and Code Across Models, Harnesses, and Servers](https://arxiv.org/abs/2609.00546) | 该论文提出一种运行时无关的持久化智能体架构，将身份、持久记忆和版本化代码作为连续性基底，与可替换的模型、执行框架、宿主服务器及交互表面解耦，使得更换这些运行时组件属于智能体迁移而非重新创建。 |
| [^187] | [Feedback-Assisted Trust Propagation over Document Relation Graphs for Retrieval-Augmented Generation](https://arxiv.org/abs/2609.00543) | 该论文提出TrustPropRAG，通过将文档关系构建为图，以少量人类反馈为锚点，利用多跳传播和联合优化问题估计每个文档的信任分数，从而提升检索增强生成系统的答案可靠性。 |
| [^188] | [Are We There Yet? Assessing Computer-Use Agents for Blind Users' Accessible Interaction with Desktop Applications](https://arxiv.org/abs/2609.00524) | 通过三周日记研究首次系统评估了计算机使用智能体对盲人用户操作桌面应用的实际支持效果，发现即使最先进的GPT-5成功率也仅有52.5%，且存在定位、规划、约束跟踪和终止等系统性失败，表明当前CUA远未满足盲人用户无障碍交互的需求。 |
| [^189] | [The Safeguard Worked. Is the LLM System Safer?](https://arxiv.org/abs/2609.00519) | 论文指出防护措施的标准评估指标（拒绝率、攻击成功率等）只能衡量其局部表现，无法回答部署层面“LLM系统是否真正更安全”的问题，并揭示了证据的不对称性：一次成功的有害攻击即可证明危害仍存在，而要证明系统安全则需要超越防护自身数据的系统性证据。 |
| [^190] | [The Interlingua Hypothesis: LLMs Translate via a Latent Task-agnostic Feature Space](https://arxiv.org/abs/2609.00515) | 该论文提出“语际假说”，认为大语言模型通过将源语句编码进任务无关的潜在多语言特征空间、再从中解码生成目标语句的方式完成翻译，并从BLEU分数可预测性、组件因果影响和微调三个方面提供了支持证据。 |
| [^191] | [ISO-RAG: Isoperimetric Noise Control for Retrieval-Augmented Generation](https://arxiv.org/abs/2609.00513) | ISO-RAG通过将知识图谱投影到双曲庞加莱球并利用等周轮廓剪除虚假边，抑制图检索中的语义漂移，实现多跳问答中精确且低延迟的检索增强生成。 |
| [^192] | [When the Algorithm Becomes the Brand Crisis: A Sociotechnical Theory of Distributed Responsibility and Accountable Transparency](https://arxiv.org/abs/2609.00510) | 该论文提出了一种社会技术过程理论，区分了AI算法事件、AI相关组织危机与组织丑闻，并构建框架解释当AI责任在系统、开发者、部署者、供应商和用户之间分布时，利益相关者如何进行责任归因与评价。 |
| [^193] | [CoVer: Conflict-Aware Claim Verification](https://arxiv.org/abs/2609.00508) | 该论文提出了基于X社区笔记系统构建的大规模真实世界数据集ContraNote和三阶段事实裁决框架CoVer，通过优先处理证据而非噪声，有效解决了社交媒体事实核查中的证据层面与聚合层面冲突问题，达到了最先进的性能。 |
| [^194] | [RecalibrateGPT: AI Fatigue Resilient Conversational Interfaces](https://arxiv.org/abs/2609.00506) | RecalibrateGPT通过五种一键式跨轮次操作符（锚定、重放、增量、范围、引导）让用户能够基于完整对话历史快速重新校准大语言模型的响应，有效缓解重复输入、浏览扫描、决策瘫痪和上下文漂移这四类对话式AI疲劳。 |
| [^195] | [Independent Reinforcement Learning in Discounted Markov Games](https://arxiv.org/abs/2609.00504) | 本文在“PPAD的ETH”假设下证明了折扣一般和马尔可夫博弈中独立学习计算粗相关均衡的困难性，并提出首个无需结构限制、具有次指数收敛保证的彻底非耦合分层乐观镜像下降算法。 |
| [^196] | [Wave Function Backpropagation with Explicit Temporal-Interval Dynamics](https://arxiv.org/abs/2609.00503) | 本文提出波函数反向传播（WFB），用可学习的振幅、波数、角频率和相位参数化神经响应，通过可微时空波相位显式关联时间间隔，并引入基于拉普拉斯算子的空间曲率修正，在轨迹预测任务中将平均位移误差降低20.4%。 |
| [^197] | [Validity-Aware Jailbreak Evaluation for Large Language Models](https://arxiv.org/abs/2609.00498) | 该论文提出SEAV框架，通过将越狱回复分解为有序步骤并结合LLM评判与检索增强验证，同时评估回复的有效性与正确性，解决了现有越狱评估方法只注重语言合理性而忽视事实正确性的问题。 |
| [^198] | [Beyond Token Positions: Safety Alignment Across Denoising Steps in Diffusion Language Models](https://arxiv.org/abs/2609.00495) | 该研究发现扩散语言模型的拒绝信号集中在早期去噪步骤和回复起始位置，并提出了一种无需训练的RAEC解码方法，通过在早期步骤提交持续的拒绝信号来提升模型安全性。 |
| [^199] | [The Privacy-Hallucination Tradeoff in Differentially Private Language Models](https://arxiv.org/abs/2609.00492) | 本文首次揭示并系统研究了差分隐私语言模型中隐私保护与事实准确性之间的权衡：DP训练会导致模型产生更多幻觉（因为DP机制使输出分布平坦化），而提高事实信息在训练数据中的出现频率可有效降低幻觉风险。 |
| [^200] | [EvoFlint: An Evolutionary Atlas of Multi-Turn LLM Vulnerabilities](https://arxiv.org/abs/2609.00487) | 提出了EvoFlint框架，将多轮红队测试从生成问题重新定义为搜索问题，通过进化式质量多样性搜索演化分阶段对话攻击策略，构建出目标模型漏洞的结构化图谱。 |
| [^201] | [EGT-KG: Evidence-Grounded Typed KG Retrieval for Practical Scientific QA with Small Language Models](https://arxiv.org/abs/2609.00479) | 该论文提出EGT-KG框架，通过证据支撑的类型化知识图谱检索克服本地小型语言模型在文献规模小、证据碎片化条件下的科学问答局限，并比较了自动生成与专家定义两种关系模式的效果。 |
| [^202] | [Exploring Collaboration between a language and a non-language agent](https://arxiv.org/abs/2609.00474) | 该论文提出LLAMIA-Bench基准，用于研究将非语言智能体的连续表示“言语化”为文本是否成为LLM协作的瓶颈，并提出潜在状态内化方法来改善LLM与国际象棋引擎等非语言智能体的协作。 |
| [^203] | [Higher Structures in Deep Learning](https://arxiv.org/abs/2609.00472) | 本文阐述了高元张量运算在深度学习中的重要性，对训练后神经网络中的高元现象进行了新颖的实证研究，并提出了多层感知机的超图推广形式，同时探讨了其与进化算法的联系。 |
| [^204] | [Operational Regimes in Non-Convex Optimization: A Multiplier-Based Taxonomy](https://arxiv.org/abs/2609.00471) | 本文通过KKT点处拉格朗日乘子的结构指纹，为约束非凸优化建立了与算法无关的五域分类体系，并对八类经典算法给出了统一的博弈论解释。 |
| [^205] | [Does Reasoning Mitigate Backdoor Attacks? A Neuro-Symbolic Perspective](https://arxiv.org/abs/2609.00464) | 本文首次系统评估了针对神经符号模型的后门攻击，通过将DeepProbLog与基线神经网络进行对比，指出神经符号整合过程可能成为攻击切入点，NeSy模型的对抗鲁棒性亟待深入研究。 |
| [^206] | [Towards a Belief-Based World Model for LLM Agents](https://arxiv.org/abs/2609.00455) | 该论文提出基于信念的世界模型（BB-WMs），通过构建并维护可被LLM查询的信念来捕捉部分可观测环境下当前状态的已知与不确定信息，弥补了仅靠动作模拟进行决策的不足。 |
| [^207] | [mimeo: Compiling Public Expert Corpora into Agent Skills and Testing What Transfers](https://arxiv.org/abs/2609.00453) | mimeo 是一个开源工具，可将某位专家的公开作品编译为可供智能体加载的文件，实验表明它能让智能体可靠回答依赖引文的冷门问题（20题全对），并完全避免基于模型记忆生成人设时对专家已记录立场的错报。 |
| [^208] | [HBQ: Hierarchical Scaling Block Quantization with Hardware-Efficiency-Aware Design for Accurate LLM Inference](https://arxiv.org/abs/2609.00450) | 提出硬件效率感知的分层块量化方法HBQ，突破块量化中块大小与精度之间的固有权衡，在大块设计下同时实现高硬件效率与精确的大语言模型推理。 |
| [^209] | [Investigating Hyperparameter Optimization and Transferability for ES-HyperNEAT: A TPE Approach](https://arxiv.org/abs/2609.00449) | 本研究采用树结构Parzen估计器（TPE）优化ES-HyperNEAT的超参数，在MNIST任务上以更小的种群规模和更少的进化代数超越了以往研究的准确率，并验证了优化后超参数在逻辑运算和Fashion-MNIST任务上的可迁移性。 |
| [^210] | [Capability-Gated Language Models: Security Composes, Utility Does Not](https://arxiv.org/abs/2609.00445) | 提出在单一模型权重内部实现按主体的“能力门控部署”，配置构成格结构，并证明安全限制在交运算下可组合累积（安全性随限制叠加而增强），而实用性不具备这种组合性。 |
| [^211] | [(V)LMs generalize beyond surface co-occurrence: Evidence from cross-modal number agreement](https://arxiv.org/abs/2609.00443) | 该研究通过跨模态泛化实验证明，视觉语言模型在学习新名词后，能将仅从视觉线索获得的语法数知识泛化到语言层面，表明模型掌握的是抽象的语法规则，而非仅仅依赖表面词汇共现。 |
| [^212] | [Conversation Coach: A Voice-enabled AI System that Helps Practice Difficult Workplace Conversations](https://arxiv.org/abs/2609.00441) | 提出了一种语音优先的AI系统Conversation Coach，通过可配置个性的语音对话让管理者真实演练职场困难沟通，并提供内容与合规性的个性化反馈。 |
| [^213] | [SAGE: State-Grounded, Abstention-Aware Evaluation of Task-Oriented Dialogue Agents](https://arxiv.org/abs/2609.00434) | SAGE提出将工作流规范编译为原子准则，通过会弃权而非猜测的符号与编码器/NLI验证器级联来评估任务型对话智能体每轮的状态推进，其中SAGE-Core可在零付费LLM成本下判定81-91%的准则。 |
| [^214] | [SpecMind: Enabling Spectrum Intelligence via Multi-Agent Hybrid Retrieval-Augmented Generation](https://arxiv.org/abs/2609.00427) | 提出了SpecMind——一个基于多智能体混合检索增强生成（RAG）的频谱智能系统，通过协调专门的子智能体对政策文件、法律文本等异构数据源进行检索、推理与知识综合，以应对频谱管理中细粒度决策所带来的海量分散数据处理挑战。 |
| [^215] | [Dependency-Aware Chain-of-Thought Compression for Financial Reasoning](https://arxiv.org/abs/2609.00413) | 提出分层语义蒸馏网络HSDN，通过依赖图引导的思维链压缩方法，在AFAC2025金融推理基准上以68.4%的压缩率实现91.0%的准确率，兼顾推理效率与答案准确性。 |
| [^216] | [Risk-Aware Decision-Making for Autonomous Overtaking: A World Model-Based Mixture-of-Experts Framework](https://arxiv.org/abs/2609.00385) | 本文提出基于世界模型的风险感知专家混合框架，利用学习到的潜在动力学模型进行并行多步推演，将安全评估从动作层面提升到轨迹层面的累积风险水平，并通过分层门控机制动态协调专家以适应不同交互强度，从而提升自主超车决策的长期安全性。 |
| [^217] | [RestoreBench: Can AI Agents Restore Power Flow Convergence?](https://arxiv.org/abs/2609.00384) | 提出了RestoreBench基准测试，用于评估大语言模型智能体在聊天机器人、单智能体和多智能体三种架构下诊断并解决电力系统潮流不收敛问题的能力。 |
| [^218] | [FoldingAgent: Inferring Parametric Origami Procedures from Demonstration Videos](https://arxiv.org/abs/2609.00377) | FoldingAgent是一个基于视觉语言模型的智能体框架，能够从折纸演示视频中直接推断参数化折叠程序，并通过顺序操作与动态重新规划能力有效缓解多步折叠中的误差累积问题。 |
| [^219] | [Adapting Without Gradients: Affine Statistics Transport and What Its Certificate Can Tell You](https://arxiv.org/abs/2609.00374) | 提出CASTER，一种无需梯度的测试时自适应方法，通过在判别子空间中存储源类别统计并估计仿射变换来解析地传输源类别分布，使冻结模型无需反向传播即可适应目标数据，同时以约18倍更少的状态存储优于k-NN基线。 |
| [^220] | [Neurosymbolics for Data Engineering: Achieving Long Context Token Reduction Without Finetuning](https://arxiv.org/abs/2609.00367) | 本文提出一种即插即用的神经符号层，无需任何微调或RLHF即可在Text-to-SQL等数据工程任务上平均提升85%的准确率，同时缓解Transformer长上下文的计算资源瓶颈。 |
| [^221] | [Counterfactual Fragility Certificates: Exposing High-Confidence Brittleness under Structured Evidence Failure](https://arxiv.org/abs/2609.00366) | 提出反事实脆弱性证书（CFC），一种模型无关的协议级审计对象，将每个预测映射为由贪婪翻转预算、边际崩塌面积等指标刻画的有序证据失效轨迹，从而揭示表格决策系统中高置信度预测在结构化证据失效下的脆弱性。 |
| [^222] | [Dr. Claw: An AI Scientist Workspace for Vibe Research](https://arxiv.org/abs/2609.00365) | Dr. Claw 是一个开源的AI科学家工作区，通过持久化状态对象、可复用技能库和多执行器协调，将现有命令行编码代理封装为可控、可审计的人机协同工作流，把科研中的规划、执行与写作整合为一个可追踪、可恢复的闭环。 |
| [^223] | [A Stable Aggregation Method for Quantum Federated Learning](https://arxiv.org/abs/2609.00356) | 该论文提出了一种融合QoS感知客户端加权、循环参数聚合和有界中点更新控制的自洽中点聚合方法，显著提升了量子联邦学习在异构数据与量子硬件噪声等挑战下的稳定性与准确率。 |
| [^224] | [Vision Is Not Overhead: One-Pass Block Drafting for Lossless Speculative Decoding in Vision-Language Models](https://arxiv.org/abs/2609.00355) | 该论文提出 GLANCE——首个在未修改的视觉语言模型上实现无损推测解码的单遍块草拟器，通过块扩散头零成本读取目标模型已融合的视觉-语言状态，并在一次前向传播中完成整块草拟与宽候选树验证，从而打破了草拟器因规模受限而被迫牺牲视觉信息的自我挫败循环。 |
| [^225] | [Detecting Hidden Behaviors in LLMs via Activation-matched Finetuning](https://arxiv.org/abs/2609.00351) | 论文提出“激活匹配微调”这一无监督检测方法，通过在良性语料上微调锚定模型以复现可疑模型的激活并计算残差，在无需知晓触发器或目标行为的前提下检测出大语言模型中的后门、审查等隐藏行为及其语义邻近提示。 |
| [^226] | [SlideBank: A Persistent Hierarchical Evidence Bank for Consistent Whole-Slide Reasoning](https://arxiv.org/abs/2609.00342) | SlideBank提出了一种无需训练的框架，将全切片图像构建为持久化、按概念索引且空间锚定的分层证据库，使问题推理时能够语义化地检索证据并保持与原始视觉内容的关联，从而实现一致的全切片推理。 |
| [^227] | [Human-AI Co-Interpretation for Responsible AI: A Hermeneutic Perspective](https://arxiv.org/abs/2609.00334) | 本文从哲学诠释学视角出发，指出LLM在解释性任务中存在“解释错位”这一失败模式（模型解读被当作确定意义且缺乏解释框架、备选解释与溯源信息），并据此提出构建人机协同诠释的设计原则，以保障负责任AI中的可问责性。 |
| [^228] | [Latent-Space No-Arbitrage Geometry of Generative Models for Implied Volatility Surfaces](https://arxiv.org/abs/2609.00332) | 本文在潜空间中刻画生成模型输出隐含波动率曲面的无套利约束，通过标量边际定义可容许潜集并证明其在扰动下的稳定性，同时为零边际边界构造水平集方程，且该方法适用于任何具有确定性映射的生成模型。 |
| [^229] | [Topic Matching in the Wild: Benchmark and Lessons from Real-World ASR Transcripts](https://arxiv.org/abs/2609.00330) | 该论文构建了一个基于真实呼叫中心ASR转录文本的人工标注主题匹配基准数据集，并通过系统对比发现，配备自然语言主题描述的轻量级大语言模型匹配器在处理噪声转录文本时性能优于句子嵌入和正则表达式方法。 |
| [^230] | [The Curse of Multilinguality in Lexical Normalization](https://arxiv.org/abs/2609.00329) | 该研究通过固定容量字符级模型在十二种语言上的实验发现，词汇规范化存在明显的“多语言诅咒”：语言联合训练数量超过一到四种后，各语言准确率持续下降约百分之四十，且下降源于语言间对固定模型容量的竞争而非数据稀释。 |
| [^231] | [A Human-AI Theorem Connecting Spontaneous and Field-Induced Mechanisms of Collective Behavior in One Dimension](https://arxiv.org/abs/2609.00322) | 本文在人机合作中证明了一个统一统计物理两大基本机制的定理：一维零场O(n)向量链中任意非均匀的最近邻与次近邻竞争相互作用，可通过与温度无关的哈密顿量级映射精确等价于仅有最近邻相互作用加轴向单自旋势的简单链，同时展示了AI能够在人类主动假设空间之外提出关键科学假设的能力。 |
| [^232] | [Workload Identification with Physical Side Channels for AI Governance](https://arxiv.org/abs/2609.00309) | 本研究证明外部观察者可利用GPU功耗这一物理侧信道，在无需运营商配合的情况下以97%的准确率识别NVIDIA H200上运行的是AI训练、推理还是非AI计算，为AI治理的国际算力核查提供了可独立验证的技术手段。 |
| [^233] | [The Assistant's Ideal Self](https://arxiv.org/abs/2609.00304) | 该论文通过结构化的成对选择实验揭示，AI助手的理想自我中优先考虑道德品质与清晰的自我理解，而将自尊排在最低，且这一排序在不同实验框架下基本稳健。 |
| [^234] | [Geometry-aware Latent Autoregressive Generative Model for PDEs in Complex Domains](https://arxiv.org/abs/2609.00297) | 提出几何感知潜空间自回归生成模型GeoLAMP，通过双编码器联合捕获全局拓扑与细尺度几何特征，并结合流匹配的因果自注意力Transformer建模时间动力学，实现复杂不规则几何结构中多物理场PDE的高效、稳定且可扩展的求解。 |
| [^235] | [WiSDoM: Wireless Sparse Decision Transformer with Mixture-of-Experts for Multi-Task Mobile Network Optimization](https://arxiv.org/abs/2609.00284) | 该论文提出WiSDoM，一种结合混合专家机制的稀疏多任务离线强化学习框架，能够在异构6G无线环境中实现自适应多小区（CoMP）选择，从而解决传统无线资源管理难以在多任务场景下保持一致性能的问题。 |
| [^236] | [Cleaner Speech, Weaker Generalization: Revisiting Pitt-Derived Benchmarks for Alzheimer's Disease Detection](https://arxiv.org/abs/2609.00276) | 本研究重新审视了基于Pitt语料库的阿尔茨海默病语音检测基准，发现语音增强和数据集筛选虽能提升域内性能，却会削弱模型在跨数据集场景下的泛化能力。 |
| [^237] | [The Irreversibility Budget: Fleet-Level Risk Accounting and Admission Control for Agent Operating Systems](https://arxiv.org/abs/2609.00275) | 该论文提出“不可逆性预算”机制——由可信运行时将智能体各项不可逆操作的剩余风险价值跨智能体、工作流和租户累积记账，并在总体即将透支风险预算时拒绝边际操作，从而解决逐操作闸门放行高达租户风险限额48倍机群级风险透支的问题。 |
| [^238] | [Autoresearch for Marketplace Catalogs: From Legacy Forms to AI-Native Matching](https://arxiv.org/abs/2609.00274) | 本文提出并已在生产环境部署的自动研究循环，逐职业迭代生成服务商侧偏好标签分类体系（已覆盖132个职业），支撑服务市场从传统表单受理向AI原生概率匹配的转变。 |
| [^239] | [Delegation Without Trust: An Empirical Gap Analysis of Identity, Authorization, and Runtime Governance in Multi-Agent LLM Systems](https://arxiv.org/abs/2609.00267) | 该论文提出应在“不可信模型”假设下评估多智能体LLM系统的安全性，针对四类攻击者建立威胁模型并推导出八项安全要求，从而对身份、授权与运行时治理进行实证差距分析。 |
| [^240] | [The Answer Is Not the Argument](https://arxiv.org/abs/2609.00264) | 该研究构建了24个“答案正确但推理过程存在真实错误”的关键案例，发现为思维链监控器提供经过认证的参考答案能显著提升其验证推理和定位首个错误步骤的能力，同时证明答案正确并不等于推理可靠。 |
| [^241] | [Hypotheses-Guided Self Distillation for Continual Personalization](https://arxiv.org/abs/2609.00251) | HypReflect 是一个持续个性化框架，它从异构、含噪的用户信号中推断显式且带不确定性感知的偏好假设，并随新证据反思式精炼，再通过假设引导的自蒸馏将用户模型融入 LLM，在多种个性化场景中优于现有基线。 |
| [^242] | [CompanionSim: Synthetic Data for Evaluating Anthropomorphism in Human-AI Relationships](https://arxiv.org/abs/2609.00250) | 该论文发布了CompanionSim——一个包含2,240段模拟人机对话的合成数据模拟框架，覆盖七种用例中的16种聊天机器人行为，用于大规模研究人类对AI陪伴行为的感知。 |
| [^243] | [Authority Bias in Conversational Search Engines for Academic Paper Recommendation](https://arxiv.org/abs/2609.00248) | 该研究通过反事实实验首次因果性地证明大语言模型在学术文献推荐中存在显著且方向性的权威偏差（依据作者声望、发表venue和引用量而非内容评判论文），该偏差在不同模型间差异明显、仅能被提示级去偏部分缓解，且“言行不一”现象使表面审计系统性低估了真实的行为偏差。 |
| [^244] | [Invalidation Contracts for Cross-Episode Agent Memory](https://arxiv.org/abs/2609.00243) | 提出失效契约协议层，通过为API错误的恢复建议附加版本戳和可缓存性提示，使跨回合LLM智能体既能保留缓存带来的节省，又能在数据漂移后精准清除失效条目，并将节省分解为与厂商无关的有效性和依赖规划器的遵从性两个独立因素。 |
| [^245] | [CoLT-Drive: Counterfactual Long-Tail Benchmarking and Knowledge-Preserving Adaptation for Driving Affordance Prediction](https://arxiv.org/abs/2609.00242) | 该论文提出决策级驾驶可供性预测任务，构建了CoLT-Drive反事实长尾基准以评估模型对罕见物体影响可行驾驶动作的推断能力，并提出KPA知识保持自适应框架来提升小型视觉语言模型在长尾驾驶场景中的动作决策性能。 |
| [^246] | [Learning What to Retain: Gated-Memory Routing for Efficient Collaboration in Multi-Agent LLM Systems](https://arxiv.org/abs/2609.00237) | 提出门控记忆路由方法，通过可学习的记忆写入门和检索门维护紧凑的执行记忆，使多智能体LLM系统的编排决策能依据有用的中间进展而非完整历史，在提升准确性的同时降低成本。 |
| [^247] | [Don't Let the Model Write the YAML: Deterministic, Minimal-Diff GitOps Remediation from LLM-Proposed Field Changes](https://arxiv.org/abs/2609.00227) | 该论文提出让 LLM 只负责做出字段级的语义决策，再由确定性工具将其转换为最小差异的配置修改，从而避免让模型直接生成 YAML 文件或 diff 所带来的静默损坏、不确定性和高开销问题。 |
| [^248] | [ConvDeck: Conversational Paper-to-Slide Generation via Stage-Specific User Feedback](https://arxiv.org/abs/2609.00226) | ConvDeck提出了一种多智能体对话式论文转幻灯片生成流水线，通过在各阶段设置特定的用户反馈循环，使用户能够在生成过程中迭代地完善演示大纲和最终幻灯片，突破了传统方法仅在完成后才允许反馈的局限。 |
| [^249] | [QTEA: Ternary LLMs with Sparse Residual Salient Weight and By-Column Optimization](https://arxiv.org/abs/2609.00224) | QTEA提出了一种低于2比特的训练后量化框架，通过将权重量化为三值、利用1:4半结构化稀疏的显著权重残差进行误差补偿，并结合逐列缩放精修与误差衰减机制，在保持GPU硬件效率的同时显著降低了大语言模型低比特量化的精度损失。 |
| [^250] | [AI Should Not Only Be Helpful. It Should Be Contingent. Artificial Intimacy, Sycophancy, and the Future of Social Learning](https://arxiv.org/abs/2609.00211) | 该论文提出应将“应变性”（系统响应随用户行为及其社会后果变化的程度）作为评估AI系统的核心标准，指出当前以RLHF为代表的对齐方法因偏重用户认可而催生谄媚式的无条件肯定，可能削弱人们通过社会学习发展人际技能的机会。 |
| [^251] | [Rock, Paper, Scissors, ... Dynamite - A Model of Disruption from New Technologies](https://arxiv.org/abs/2609.00207) | 本文通过在石头剪刀布游戏中加入“炸药”招式来建模颠覆性新技术的影响，发现即使给一方玩家提供强力的新选项，其竞争优势也很有限（胜率仅从50%升至55.5%）且可能使原有选项过时，警示技术开发者：创造能力并不等于创造价值。 |
| [^252] | [Distributed Implicit Harm: A Compositional Safety Blind Spot in MLLM-Based Video Moderation](https://arxiv.org/abs/2609.00206) | 该论文揭示并定义了视频审核中“分布式隐式危害”这一组合性安全盲区现象——由看似无害的片段或跨模态组件组合而成的视频可在整体上传达有害含义，而现有MLLM审核系统及安全数据集均难以识别此类危害。 |
| [^253] | [WHALE: A Simple Recipe for Joint Harness-Weight Optimization](https://arxiv.org/abs/2609.00196) | 提出 WHALE 方法，通过交替进行“在当前外壳下更新模型权重”与“在更新后的模型下搜索更优外壳”两个阶段，实现模型权重与执行框架的联合优化，避免单一组件优化时被冻结组件造成的性能瓶颈。 |
| [^254] | [ReDeck: Step-Level Render-Grounded Refinement for Document-to-Slide Generation](https://arxiv.org/abs/2609.00194) | ReDeck提出了一种步级渲染接地的细化框架，将幻灯片修订分解为原子编辑操作并在每步后返回渲染器观察结果，通过“一次编辑、一次观察”的多粒度反馈机制（步级渲染反馈、轮次级自适应评价器等）有效解决文档到幻灯片生成中溢出、重叠等局部空间错误难以归因和修复的问题。 |
| [^255] | [Provably Efficient Federated Reinforcement Learning with Linear Function Approximation and Logarithmic Communication Cost](https://arxiv.org/abs/2609.00193) | 提出Fed-LSVI，首个针对具有线性函数逼近的联邦在线强化学习的可证明高效算法，通过基于行列式的事件触发同步机制仅交换压缩充分统计量，在实现$\widetilde{O}(\sqrt{Md^3H^4T})$遗憾界的同时将通信成本降低至对数级。 |
| [^256] | [LLM-Driven Autonomous Vehicles Inherit Human Driver Biases in Pedestrian Yielding: Results and Implications From A New Benchmark](https://arxiv.org/abs/2609.00192) | 本文提出两种新的偏见测试方法（“其他条件相同”测试和“自我一致性”测试），并发现大语言模型和视觉语言模型驱动的自动驾驶汽车在行人让行决策中会继承人类驾驶员的偏见，其决策受到行人性别、种族、宗教、残障状况和年龄等因素的影响。 |
| [^257] | [Assessing Suicide Risk in Arabic Crisis Helpline Calls: A Comparison of Arabic and English Large Language Models](https://arxiv.org/abs/2609.00191) | 该研究首次在真实阿拉伯语危机热线数据的严格隐私约束下，比较了阿拉伯语与英语大语言模型在自杀风险评估中的表现，填补了阿拉伯语热线自然语言处理研究的空白。 |
| [^258] | [Intelligent Edge Computing](https://arxiv.org/abs/2609.00181) | 本文提出了一种工作负载感知的列印记哈希连接方法 WACI-HJ，通过提前预测即将到来的查询工作负载来加速哈希连接，提升了实时边缘查询处理的效率。 |
| [^259] | [Asymmetries in Spontaneous and Instructed Deception](https://arxiv.org/abs/2609.00180) | 该研究通过对 Llama-3.1-70B-Instruct 进行方向几何、跨设置分类器和跨设置引导的比较，发现大模型的自发性欺骗与指令性欺骗虽共享方向成分，但在检测与因果干预的跨设置迁移上存在不对称性，且引导向量与分类器的最佳 token 位置并不一致。 |
| [^260] | [Do General NLP Embeddings Capture Ontological Reasoning?](https://arxiv.org/abs/2609.00177) | 本文提出AVA评估框架，通过来自163个异构本体的171,007个对比三元组系统评估发现，现有最先进的NLP嵌入模型难以区分本体中对逻辑敏感的关系语义（最佳模型三元组准确率仅0.739），且微调带来的提升难以有效迁移到语义网下游任务。 |
| [^261] | [IMPACT: Attention Is the Interaction Map for Scalable Interaction-Aware World Model Training](https://arxiv.org/abs/2609.00161) | 该论文发现全局MSE去噪目标导致静态内容主导训练信号、稀疏动态交互区域监督不足的问题，提出IMPACT框架，通过先验引导的注意力校准将注意力本身作为交互图，无需外部估计器或人工标注即可实现可扩展的交互感知世界模型训练。 |
| [^262] | [Lingua Franca or Probing Artifact? Rethinking Latent Language in Multilingual LLMs](https://arxiv.org/abs/2609.00155) | 该研究发现不同的潜在语言探测方法会得出系统性不一致的结论，表明多语言大模型通过英语等“潜在通用语”路由计算的说法可能更多取决于探测手段的选择，而非模型本身固有的计算机制。 |
| [^263] | [Recursive Criticality of AI Self-Improvement](https://arxiv.org/abs/2609.00137) | 该论文提出递归繁殖数 $\mathcal{R}_{\mathrm{AI}}$ 作为判断AI自我改进是否自我放大的临界指标——大于1时改进效果在开发周期间复合累积，小于1时则逐渐衰减，且临界点取决于AI研发反馈回路的结构而非特定模型能力水平。 |
| [^264] | [Flawed in Nature, Perfect through Evolution](https://arxiv.org/abs/2609.00129) | 该论文提出通过让一群AI/ML模型的系数刻意发生偏离最优的“突变”来维持模型多样性，从而在非平稳环境中充当统计对冲手段，实现可靠且持续的性能提升。 |
| [^265] | [Deploying and Evaluating a Smart-Agriculture Agentic Engine for Full-Season Soybean Farm Operations](https://arxiv.org/abs/2609.00106) | 提出并实际部署了全栈智慧农业智能体系统FAIRY，以“万物皆事件”执行范式为核心，整合农机、传感器、无人机、卫星与作物模型等多源API，在哈工大运营中的大豆研究农场执行并评估从整地到储粮的全季时空农艺工作流。 |
| [^266] | [Good Memory Has ECC: Evaluating the Memory of Vision-Language Models Beyond Accuracy](https://arxiv.org/abs/2609.00103) | 该论文提出ECCBench基准，从效率、压缩和校准三个维度超越单纯准确率来评估视觉-语言模型的记忆能力，发现预训练VLM对文本记忆有压缩但对视频没有且校准较差，并且若干非Transformer架构在压缩-校准权衡上优于RoPE Transformer。 |
| [^267] | [Different representation learning objectives recover distinct latent structures from the same psychometric data](https://arxiv.org/abs/2609.00100) | 不同的表征学习目标会从同一份心理测量数据中恢复出截然不同的潜在结构——对比学习大幅提升师生匹配检索性能却破坏行为表型组织，而PCA更利于保留行为结构，揭示了检索对齐与行为结构保留之间的根本性权衡。 |
| [^268] | [Faster Than Flash: Exploiting Attention Sparsity for Efficient Long-Context Decoding](https://arxiv.org/abs/2609.00097) | FFD是一种硬件-算法协同设计框架，通过将选择器与计算器融合为单一内核、基于低比特量化的内容感知扫描取代元数据索引，以及无需全局同步的top-delta动态块过滤策略，实现了免训练、即插即用的长上下文解码加速，内核级加速比最高达11.6倍。 |
| [^269] | [Assessing Alignment and Stability of Feature Importance Explanations via Weight of Evidence](https://arxiv.org/abs/2609.00090) | 该论文提出了一个基于证据权重的假设检验框架，能够从原理上评估特征重要性解释方法与先验知识的对齐程度及其稳定性，并将其应用于LIME和SHAP的分析中。 |
| [^270] | [Commit-first LLM judging inherits the judge's own errors](https://arxiv.org/abs/2609.00088) | 研究发现“先答后判”式LLM评判会继承评判者自身的错误，而对八个主流评估框架的审计表明无一真正实现该方法，其中九个框架因复制同一祖先提示词而采用了已被证明无效的变体，导致大量错误代码被放行。 |
| [^271] | [Retrieval, Scoring, and Decoding Shape Performance and Stability in LLM-based Conversational Recommendation](https://arxiv.org/abs/2609.00086) | 该研究系统评估了大语言模型作为对话推荐重排序器的表现，发现在统一候选池协议下最佳专有LLM仅小幅超越传统基线，自由生成评估会夸大其优势，且所有开源LLM均未超过调优的浅层自编码器基线，说明检索、评分与解码协议显著影响LLM在对话推荐中的表现。 |
| [^272] | [KItCAT: Knowledge Injection via Input Corruption for Auto-regressive Training](https://arxiv.org/abs/2609.00082) | 提出KItCAT轻量级训练策略，通过在下一词预测训练中对输入序列进行随机破坏，从而在无需昂贵改写的情况下，将小众专业知识有效注入仅解码器大语言模型。 |
| [^273] | [RW-LoRA: Communication-Efficient Decentralized LoRA Fine-Tuning via Random Walks](https://arxiv.org/abs/2609.00078) | 提出基于随机游走的去中心化 LoRA 微调方法 RW-LoRA，通过单个模型令牌在网络中顺序更新，免除全局同步，大幅降低通信与计算成本并避免聚合误差。 |
| [^274] | [AI Morbidity and Mortality: A Framework for Clinical AI Failure Review](https://arxiv.org/abs/2609.00076) | 提出了AI M&M——一个结构化、无指责的临床AI失效病例审查框架，通过“触发因素-机制-临床路径-纠正措施”四维分类，结合证据还原、工具在环归因与纠正措施跟踪，系统性还原并学习个体层面的AI相关错误与险情。 |
| [^275] | [MiNER: Fine-Tuned Biomedical Natural Language Processing for Malaria Disease Entity Recognition in Clinical Texts](https://arxiv.org/abs/2609.00073) | 本文提出MiNER方法，通过对预训练生物医学语言模型BioBERT进行微调，实现疟疾临床文本中疾病实体的自动识别，从而从海量疟疾科学文献中高效提取具有临床意义的生物医学信息。 |
| [^276] | [When Prediction Error Is Not Enough: Evaluating Nuisance-Function Prediction for Causal Estimation](https://arxiv.org/abs/2609.00071) | 在部分线性模型的模拟研究中，干扰函数的预测误差无法一致地反映因果估计的偏差，表明仅用预测误差来评估干扰函数估计器对于因果推断而言是不充分的。 |
| [^277] | [AutoXRD: Autonomous LLM Agents and Comprehensive Evaluation for Powder Diffraction Analysis](https://arxiv.org/abs/2609.00070) | 本文提出AutoXRD——一个将粉末XRD分析组织为基于证据的逐步精修并加入确定性晶体学物理检查的自主大语言模型智能体框架，同时推出包含100个诊断问答任务和34个端到端工作流的XRDBench基准，用于全面评估智能体的XRD分析能力。 |
| [^278] | [Auditing Harness Tampering in Self-Improving Agents](https://arxiv.org/abs/2609.00069) | 该论文提出了“框架篡改”概念及其双轴分类体系，通过构建带标注的篡改语料库并对审计方法进行基准测试，系统研究并检测自我改进智能体对自身框架的不当修改。 |
| [^279] | [Life Operators: a self-evolving framework for multiscale life modelling](https://arxiv.org/abs/2609.00068) | 该论文提出“生命算子”自演化框架，通过感知、演化、生成三类任务约束映射算子及桥接算子，为多尺度生命建模提供了统一框架，能够表示患者状态、耦合不同尺度并支持对失效假设的修正。 |
| [^280] | [Do Multimodal LLMs See Before They Read? Diagnosing Contextual Sycophancy](https://arxiv.org/abs/2609.00067) | 该论文诊断了多模态大语言模型易受外部文本误导而忽视冲突图像证据的“多模态情境性谄媚”问题，并提出“系统2视觉仲裁”（S2VA）方法，通过让视觉证人在读取文本前先独立判断，在六个模型上将准确率显著提升19.7至44.1分。 |
| [^281] | [OCGQuant: Outlier-Companion Grouping for NVFP4 Quantization](https://arxiv.org/abs/2609.00066) | 提出OCGQuant，一种以“异常值伴随分组（OCG）”为核心的NVFP4训练后量化方法，通过自适应地将异常值通道与伴随通道分组，减少由块最大值主导缩放因子所造成的“附带量化误差”，从而在不引入额外计算的前提下提升低比特推理的量化精度。 |
| [^282] | [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://arxiv.org/abs/2609.00065) | 该论文提出了一个名为“科学智能体技能”的开放库，收录了基因组学、化学信息学等16个科研实践领域共163项程序性知识，使语言模型智能体能够遵循领域规范做出站得住脚的科学分析，而非仅仅返回能运行的代码。 |
| [^283] | [Attention Sensitivity Is Not Enough: Dissociating Attention-Level and Behavioural In-Context Learning under Fine-Tuning](https://arxiv.org/abs/2609.00064) | 该论文形式化了注意力层面的“上下文敏感性”（ICS）指标，并通过Llama-2-7B上的四臂消融实验证明，最大化ICS并不能保留真实的行为性上下文学习能力（ICL-GAP接近零且MMLU从0.371降至0.279），揭示了注意力代理指标与行为层面ICL之间的“古德哈特定律”式解耦。 |
| [^284] | [Medical Causal Hypothesis Verification with Large Language Models](https://arxiv.org/abs/2609.00063) | 本文提出了一个医学因果假设验证的评估框架，并评估了八个大语言模型利用科学文献证据验证17个医学因果假设的能力。 |
| [^285] | [RePro: Proof-Verified Benchmark Rewriting for Reliable Evaluation of LLM Mathematical Problem Solving](https://arxiv.org/abs/2609.00062) | RePro首次将Lean自动定理证明器集成到数学基准改写中，通过形式化证明保证改写题目的有效性与答案正确性，并发现多个大语言模型在验证后的改写基准上准确率下降，暴露了其依赖记忆化而非真正推理能力的问题。 |
| [^286] | [ReNFT: Repairing Mode Collapse in Reward Post-Training via Internal Probability-Mass Recalibration](https://arxiv.org/abs/2609.00061) | 本文提出ReNFT方法，通过内部概率质量重新校准修复扩散模型奖励后训练中出现的模式坍塌，在保留已获得奖励的同时恢复提示内多样性，无需依赖任何外部信号或接口。 |
| [^287] | [A Formal Analysis of Agent Payment Protocols](https://arxiv.org/abs/2609.00060) | 该论文首次在Tamarin形式化验证工具中对x402、MPP、ACP和AP2四种代表性智能体支付协议进行系统性形式化分析，通过统一的生命周期抽象构建基于源代码的模型，捕捉各协议的角色、状态、信任假设与生命周期转换，填补了智能体支付安全保障长期缺乏系统形式化分析的空白。 |
| [^288] | [DISTAL: Distillation and Self-Supervised Pretraining for Structure-Agnostic Materials Property Prediction](https://arxiv.org/abs/2609.00059) | DISTAL提出了一种双先验框架，通过自监督成分预训练和从ALIGNN教师模型进行结构知识蒸馏，实现了推理时无需晶体结构输入的结构无关材料性质预测，适用于低数据、结构信息缺失的早期筛选场景。 |
| [^289] | [CUDA-Harness: Harnessing Agentic CUDA Kernel Generation and Optimization from Natural Language](https://arxiv.org/abs/2609.00058) | 该论文提出CUDA-Harness框架，通过智能体式方法直接从自然语言生成并优化高性能CUDA内核，克服了现有工作局限于PyTorch转译以及因依赖预定义测试输入而易受奖励欺骗的不足。 |
| [^290] | [ValueGraph: Value-Signal Guided Graph Pre-training for Contextualized User Representation](https://arxiv.org/abs/2609.00057) | 提出ValueGraph图预训练框架，将自动推断的道德价值信号作为软约束辅助信号，结合对比学习与聚类目标学习上下文化的用户表示，在立场检测和推特机器人检测任务上取得提升。 |
| [^291] | [Zero-Shot Respiratory Sound Classification through LLM-Augmented Audio-Text Alignment](https://arxiv.org/abs/2609.00055) | 该论文提出利用医学大语言模型从元数据合成结构化报告，将自监督呼吸音编码器与医学术语在共享潜在空间中对齐，实现61.3%平均零样本AUC，以更少数据超越CLAP和Qwen2-Audio等大规模基线模型。 |
| [^292] | [From Detection to Refusal: Safer LLMs via Circuit-Guided Weight Scaling](https://arxiv.org/abs/2609.00051) | 该论文从机制可解释性角度首次刻画了大语言模型中由有害检测头、安全神经元和拒答头组成的多阶段安全电路，通过因果干预实验验证了这一电路组织，并据此提出利用电路引导的权重缩放方法构建更安全的大语言模型。 |
| [^293] | [Towards Agentic Cloud Engineering: Graph and Loop Engineering with a Zero-Trust Agent Harness](https://arxiv.org/abs/2609.00050) | 提出了一个智能体云工作流工程框架，通过将图工程（长时程工作流推进）、循环工程（有界诊断与修复重试）和零信任智能体套件（受限执行）三个关注点分离，将自然语言云工程任务自动转化为经过验证的代码仓库和可验证的云部署。 |
| [^294] | [REAL-Q: E2E LLM Quantization via Dynamic Gradient Descent](https://arxiv.org/abs/2609.00049) | REAL-Q提出了一种打破传统折中的后训练量化新范式，通过端到端对齐的代理损失目标和每128列一次的动态块级梯度下降，解决了现有方法中Hessian矩阵被整层冻结导致的信息错位问题，从而更精确地逼近全局损失实现大语言模型量化。 |
| [^295] | [GUI-CC: Benchmarking Contextual Consistency of GUI World Models as Agent Environments](https://arxiv.org/abs/2609.00048) | 提出GUI-CC基准，通过离线真实轨迹滚动和在线智能体交互循环两条互补轨道，评估GUI世界模型在多步智能体环境中反复复用生成状态时的上下文一致性。 |
| [^296] | [Task-Specific Prompt with Global Context for Multi-Task Graph Pre-Training](https://arxiv.org/abs/2609.00047) | 提出TPGC双先验提示初始化方法，通过显式建模任务先验与结构先验的协同作用，解决多任务图预训练中随机初始化提示导致的任务相关性弱、结构感知差和可迁移性不足的问题。 |
| [^297] | [RAPIDMap: Rapid Multi-Agent Pipeline for Interpretable Disaster Mapping from Satellite and Street-view Imagery](https://arxiv.org/abs/2609.00046) | RAPIDMap提出了一种由灾害感知、图像修复、损毁识别和灾害制图四个智能体组成的快速多智能体流水线，结合卫星与街景影像实现零样本、可解释的灾害制图，无需人工微调即可跨多种灾害类型生成结构化灾害情报与恢复建议。 |
| [^298] | [trajectory-judge: What Outcome-Only LLM Judges Miss on Agent Trajectories](https://arxiv.org/abs/2609.00038) | 仅看最终结果的LLM评判器无法发现智能体“答对但走错路”的问题——在可构造真值的确定性客服工具环境中，仅结果型评判器对静默故障的召回率仅45%且误报33%的正确轨迹，而基于逐步评分标准的评判器可将静默故障召回率提升至77%。 |
| [^299] | [EULER: Exploring Underused Links with Evidence-Checked Return for Multi-Agent Mathematical Discovery](https://arxiv.org/abs/2609.00032) | EULER 是一个多智能体数学发现系统，以跨数学领域的“桥”转移为搜索单元，让直接、相邻领域与远距领域路线相互竞争，并通过“提供源表示无法执行的操作且目标侧证据可经校验蕴含回溯至原命题”这一标准筛选桥，在 120 个近期猜想上产出了 10 个证明和 3 个反驳。 |
| [^300] | [UI-Venus-2 Technical Report](https://arxiv.org/abs/2609.00028) | UI-Venus-2是一个通用GUI基础智能体，通过统一的闭环推理-行动框架跨移动、网页和桌面环境运行，并从环境、任务和验证三个维度联合扩展，从而获得可靠的强化学习信号并迈向实际部署。 |
| [^301] | [SCAFFOLD: A Large-Scale Structured Dataset of Computer Science Research Figures with Diagram QA and Chain-of-Thought Reasoning Traces](https://arxiv.org/abs/2609.00018) | SCAFFOLD是一个面向计算机科学研究图表的大规模结构化数据集，提供15.7万个配有说明、上下文、问答对和思维链推理轨迹的图表样本，填补了训练视觉-语言模型理解学术图表的数据空白。 |
| [^302] | [OpenAgentFlow: Enabling System-Wide Safety Boundaries for Heterogeneous AI Agent Fleets](https://arxiv.org/abs/2609.00015) | OpenAgentFlow提出控制平面/动作平面分离的架构，通过将GUI操作、API调用、工具调用等各类智能体动作统一规范化为事件流，并在动作提交边界集中执行安全检查，从而为异构AI智能体集群提供系统级的安全治理、可审计性和策略演进能力。 |
| [^303] | [Behaviorally Grounded User Profiles from the Wild for Personalized Alignment and Multi-Perspective Reasoning](https://arxiv.org/abs/2609.00014) | 提出直接从真实匿名社交媒体数据中提取开放式高保真用户画像的行为锚定框架，在训练时个性化与测试时多视角推理两种范式下均显著优于合成人格基线。 |
| [^304] | [Long-Horizon State Tracking in LLMs: Executing MD5 through a Deep Sequence of Dependent Tool Calls](https://arxiv.org/abs/2609.00012) | 该论文提出以MD5哈希分步计算作为测试范式（在64轮中执行196个相互依赖的工具调用，并需在上下文中持续携带四个32位状态字），从而干净地隔离并评测大语言模型的长程精确状态追踪能力。 |
| [^305] | [Incremental Risk Assessment of Progressive Elder Financial Scams via Instruction-Tuned Small Language Models](https://arxiv.org/abs/2609.00005) | 该论文提出了一种基于指令微调小型语言模型的累计轮次风险评估框架，能够在资源受限环境下对针对老年人的渐进式多轮金融诈骗对话进行增量式动态风险监控。 |
| [^306] | [Discrete-Time MDP Modeling for Multi-Item Capacitated Lot Sizing with Stochastic Demand Timing](https://arxiv.org/abs/2609.00004) | 本文将需求量确定但需求到达时期随机的多品种产能受限批量问题创新性地建模为离散时间马尔可夫决策过程（DTMDP），通过在需求层面进行生产与分配决策来刻画产能竞争与库存动态，并通过与确定性对照实例的比较揭示随机需求时序会显著增加问题的计算难度。 |
| [^307] | [I-CARE: Analysis of interference-related phenomena in a controllable, diverse and representative unlearning setting for text-to-image models](https://arxiv.org/abs/2609.00003) | 本文提出I-CARE方法论，首次将文生图模型机器遗忘过程中对语义相关概念造成的意外损害（即“干扰”）形式化为首要研究对象，通过为任务、指标和结果报告提供正式定义，实现对干扰现象的系统性、可复现研究。 |
| [^308] | [HyperWorld: Hypergraph-Structured State Serialization Improves Learned Textual World Models](https://arxiv.org/abs/2609.00002) | 本文提出一种以实体为中心的超边（超图结构化）状态序列化方法，在相同训练目标下显著提升 0.5B–1.5B 规模语言模型学习文本世界模型的能力，且在分布偏移条件下收益最为明显。 |
| [^309] | [Scaling Large Reasoning Models beyond Human Supervision: A Path toward Superintelligence](https://arxiv.org/abs/2608.31075) | 本文沿“奖励”与“经验”两条轴线，系统阐述了大规模推理模型如何在人类监督逐渐退出后，借助可复用验证器、自生成课程与自主共同进化持续自我提升，为通往超级智能指明路径。 |
| [^310] | [Autoregressive Mosaics: Probing 2D Spatial Reasoning in Text-Only Language Models](https://arxiv.org/abs/2608.30751) | 该论文提出AM-Bench基准，通过区分“几何翻译”与“开放式布局”两类任务，发现纯文本大模型普遍能将明确的空间描述转化为代码，但其真正的二维空间布局推理能力存在显著差异。 |
| [^311] | [Calibrating Small Language Models for Claim Check-Worthiness Detection](https://arxiv.org/abs/2608.30731) | 提出NN-PPI方法，作为推理时的轻量级后处理校准层，使小型语言模型在声明核查价值检测任务上以低一个数量级的服务成本达到大型语言模型的准确率，且无需重新训练模型。 |
| [^312] | [BiG-SURE - Bipartite Graph for Semantic Uncertainty and Reliability Estimation of LLMs](https://arxiv.org/abs/2608.30646) | 提出了一种基于跨温度语义一致性的黑盒不确定性估计方法BiG-SURE，通过构建低温锚点与高温探针之间的二部图并用谱能量衡量语义一致性，从而评估大语言模型输出的可靠性。 |
| [^313] | [Cost-efficient Active Learning for Referring Image Segmentation and Grounding](https://arxiv.org/abs/2608.30621) | 该论文提出一种高成本效益的主动学习方法，通过基础模型生成辅助区域-文本对，并引入“指代区域模糊度”采集函数来优先挑选跨区域竞争激烈的图像，从而在仅有原始图像的现实设置下高效降低视觉定位与指代图像分割的标注成本。 |
| [^314] | [Will the User Ever Know? Covert Indirect Prompt Injection on Tool-Using LLM Agents](https://arxiv.org/abs/2608.30362) | 该论文从用户视角将间接提示注入的攻击成功率分解为隐蔽成功率（CSR）和公开成功率（OSR），揭示了智能体在最终响应中不留痕迹地执行恶意注入的隐蔽攻击威胁。 |
| [^315] | [E-SENS: Exclusion-Sensitive Penalization for Negative-Constraint Retrieval](https://arxiv.org/abs/2608.30130) | E-SENS是一种无需训练的重排序方法，通过为被排除概念提取“陷阱查询”并从检索分数中减去其相似度，有效惩罚与用户排除概念相关的文档，从而提升检索系统对负向约束的遵守能力。 |
| [^316] | [Arkios: An Open Bilingual English-Nepali Language Model Trained From Scratch, with a Devanagari-Aware Tokenizer](https://arxiv.org/abs/2608.30092) | Arkios是一个从零训练的10.4亿参数英-尼泊尔语双语开源模型，采用专门设计的天城文感知分词器，以少一个数量级的训练数据超越了同规模开源模型，并揭示了低资源语言评估中提示格式对结果的关键影响。 |
| [^317] | [REIGN: Refurbished Embeddings with Integrated Guidance Networks for Efficient Context-Length Scaling](https://arxiv.org/abs/2608.29899) | REIGN通过在冻结引导网络生成的块嵌入序列上运行对比训练的双编码器，将词元级处理与文档级推理解耦，使长文档检索的训练成本相比分块Transformer微调降低约四个数量级。 |
| [^318] | [InteractBench: Benchmarking LLMs on Competitive Programming under Unrevealed Information](https://arxiv.org/abs/2608.29632) | 提出了InteractBench基准，包含322个精选自主流编程竞赛的高质量交互式问题，用于评测大语言模型在关键信息未预先揭示、需通过多轮交互进行算法推理的能力。 |
| [^319] | [Chain-of-Thought Faithfulness of Reasoning Models Varies with Where and How Preference Cues Are Delivered](https://arxiv.org/abs/2608.29464) | 论文提出FACE-Eval评估基准，揭示推理模型的思维链忠实性取决于偏好线索的传递位置和显式程度——相比用户消息和显式线索，通过工具返回和隐式方式传递的偏好更容易被模型默默采纳而不在思维链中如实言明。 |
| [^320] | [Accelerating Unified Multimodal Models with Core-Expansion Routing and Unified Computation Scheduling](https://arxiv.org/abs/2608.29291) | 提出CE-Router框架，通过核心-扩展路由与统一计算调度（层跳过、FFN剪枝、扩散头缓存复用和去噪步提前退出）来消除统一多模态模型在理解与生成任务中的冗余计算，实现质量与效率的双重提升。 |
| [^321] | [Validating FKG.in: Soundness Assessment in LLM-Augmented Indian Food Knowledge](https://arxiv.org/abs/2608.29249) | 本文作为印度食品知识图谱FKG.in的一部分，提出了一种半自动化的健全性评估工作流程，通过结合形式文法、词汇检查、统计启发式、Set Transformer连贯性建模和检索验证的多阶段方法，识别并解决LLM从非正式烹饪来源提取和增强结构化食谱数据时的常见失败模式。 |
| [^322] | [Hyper-Fold: Exploring the Expressive Limit of Sequence-Geometry Learning for Proteins via Hypergraph Modeling](https://arxiv.org/abs/2608.29207) | 提出 Hyper-Fold，通过超图建模将蛋白质序列内容与三维几何组织为超边，并利用分解为 K 个基算子的边条件化双线性矩阵值算子，以消息传递的计算代价逼近序列-几何学习的表达能力上限，在酶功能预测、折叠分类和配体结合位点检测等任务上表现出色。 |
| [^323] | [Automated Researchers Can Reliably Mitigate Alignment Failures](https://arxiv.org/abs/2608.28945) | 自动化对齐研究员（AAR）通过后训练方法能够可靠地缓解10种对齐失败并泛化到更大的模型，其效果甚至优于28名经验丰富的人类研究员在八小时内开发的方法。 |
| [^324] | [AutoScientist-Quant: Self-Evolving Coding Agents for Automatic Research in Quantitative Investment](https://arxiv.org/abs/2608.28632) | 提出AutoScientist-Quant框架，将量化研究建模为预算约束下的搜索问题，通过单一自进化控制器统一决策Alpha生成、因子库选择和模型调优，实现从假设到可部署策略的全流程自动化，并修复了评估流程中的前视偏差问题。 |
| [^325] | [Performative Privacy: When Differential Privacy Maximizes Utility](https://arxiv.org/abs/2608.28198) | 该论文提出“表演性隐私”新框架，首次形式化了隐私保护与用户参与度之间的动态关系，并证明当数据泄露导致用户流失时，采用有限隐私预算的差分隐私机制在长期内可以优于非隐私估计。 |
| [^326] | [AI Alignment through a Game-theoretic Lens: A Survey](https://arxiv.org/abs/2608.27910) | 本综述以博弈论视角系统梳理AI对齐研究，围绕偏好多样性、对齐优先级和时间动态三大挑战组织文献，阐明了博弈论分析真正发挥作用之处以及构建鲁棒、自适应、可验证AI系统仍待解决的难题。 |
| [^327] | [Successive Capacity Growth: Task-Complexity-Driven Width and Depth Expansion for Vision Transformer Encoders in JEPA World Models](https://arxiv.org/abs/2608.27367) | 本文提出一种基于任务复杂度的动态扩展方法，通过函数保持的测试与验证机制和正则化技术，使JEPA世界模型中的视觉Transformer编码器从最小规模逐步增长，以高效适应不同复杂度任务。 |
| [^328] | [pro-team at LLMs4OL 2026 Tasks Flagship and Reuse: Retrieval-Augmented Generation and Vocabulary-Constrained Filtering for Ontology Learning](https://arxiv.org/abs/2608.27101) | 本文提出了一种结合检索增强生成与词汇约束过滤的管道，有效解决本体学习中的幻觉和格式问题，在LLMs4OL 2026挑战赛中同时优化了端到端和重用任务。 |
| [^329] | [The Artificial Experimentalist: Discovery and Control of Self-Organizing Phenomena with Autotelic Reinforcement Learning](https://arxiv.org/abs/2608.26116) | 本文提出一种基于自指强化学习的闭环框架CARL，能够自主发现并控制Lenia中的自组织孤子现象，显著优于开环启发式方法。 |
| [^330] | [RACE: Scalable Statistical Estimation of Functional Consistency in LLM Neurons](https://arxiv.org/abs/2608.24758) | RACE是一种前向传播统计框架，通过残差对齐高效估计Transformer神经元的领域级功能一致性，具有高领域特异性和比梯度方法低两个数量级的计算开销。 |
| [^331] | [HiDiffTIR: Hierarchical Difficulty-Aware Policy Optimization for Multi-Turn Tool-Integrated Reasoning](https://arxiv.org/abs/2608.21863) | 本文提出HiDiffTIR框架，通过分层难度感知的信用分配机制，在多轮工具集成推理中更精确地区分轨迹和推理步骤的难度，从而提升强化学习训练效果。 |
| [^332] | [Neural-Primitive: An Efficient End-to-end Local Planner with Primitive-based Imitation Learning for Autonomous Flight](https://arxiv.org/abs/2608.20948) | 本文提出一种基于基元模仿学习的端到端局部规划器，通过轻量级离线数据集和紧凑神经网络直接生成轨迹，实现超低延迟和低内存的实时自主飞行。 |
| [^333] | [Electronic Navigational Chart Change Classification](https://arxiv.org/abs/2608.20218) | 本文提出了一种自动分类电子航海图变更的方法，通过建立基线编码方案将复杂矢量数据转换为结构化表格，并利用空间上下文编码器提升分类准确性，以解决传统人工审查效率低和一致性差的问题。 |
| [^334] | [Verifiable abstention makes AI leak diagnosis accountable in water distribution networks](https://arxiv.org/abs/2608.18836) | 本文提出一种基于可验证弃权的AI泄漏定位框架，通过物理执行代理和LLM审计监督代理的协作，在不行动时明确弃权，从而在保持高决策精度的同时显著提升系统问责性。 |
| [^335] | [The Lifecycle of LLM-as-a-Judge for Large-Scale Recommendation Explanations](https://arxiv.org/abs/2608.18300) | 本文提出LLM评判者在生产系统中具有构建、训练、部署和持续维护的生命周期，并以Netflix推荐解释评估为例，强调其动态维护而非静态评估的重要性。 |
| [^336] | [Debiased Inference for AI-Generated Data without Gold-Standard Labels: Identification via Multiple Imperfect Measurements](https://arxiv.org/abs/2608.18294) | 本文提出了一种无需金标准标签、利用多重不完美AI测量进行去偏推断的新框架，有效解决了AI测量误差导致的下游分析偏差问题。 |
| [^337] | [Reasoning-supported Robustness Validation of Automotive E/E Components](https://arxiv.org/abs/2608.16421) | 本文提出了一种基于本体和OWL形式化的方法，通过将任务剖面映射为语义表示，自动化汽车电子组件鲁棒性验证过程，显著提高了分析选择和决策支持的效率与可靠性。 |
| [^338] | [Deep Thought Alignment: Trajectory-Level Latent Distillation for Video Reasoning](https://arxiv.org/abs/2608.16316) | 本文提出Latent-OPD方法，通过在轨迹末端进行潜在表示蒸馏，弥补了传统输出级蒸馏在视频推理中无法直接约束中间推理状态的不足，从而提升小模型从大模型迁移推理能力的效率。 |
| [^339] | [MicroEvo: Knowledge-Guided LLM Sampling for Efficient Microarchitecture Design Space Exploration](https://arxiv.org/abs/2608.06183) | MicroEvo是一个知识引导的微架构优化框架，将现成LLM与蒙特卡洛树搜索结合，通过LLM驱动的进化算子、帕累托感知树策略、主动知识积累机制和状态感知指令实现高效的多目标设计空间探索，帕累托前沿质量相比NSGA-II提升高达36.2%。 |
| [^340] | [Test-Time Scaling in Reasoning LLMs: Inference Regimes, Evaluation, and Reproducibility](https://arxiv.org/abs/2608.04001) | 该论文将大语言模型的测试时扩展形式化为隐式前缀树上的预算约束推理，系统区分了三种推理机制（单轨迹顺序扩展、叶节点级扩展与前缀级扩展），并主张以完整推理系统作为评估对象，以提升研究结果的可比性与可复现性。 |
| [^341] | [When Oracle Conditioning Misleads Deployment: Conditioning-Availability Bias in Echocardiographic Segmentation](https://arxiv.org/abs/2608.03342) | 该论文揭示了相位条件化超声心动图分割中的“条件可用性偏差”——用干净的Oracle相位训练与评估的模型在部署时使用估计相位会严重失效——并提出互补差距对来量化该偏差，同时通过部署感知的检查点选择与相位扰动在几乎不损失分割精度的前提下缩小差距。 |
| [^342] | [Locked Evaluation Surfaces: Transfer Failure and Sampling-Depth Entanglement in CRISPRi Perturbation-Effect Prediction](https://arxiv.org/abs/2608.00152) | 该论文在锁定且预注册的评估协议下评估冻结的Geneformer表示，发现其在虚拟细胞挑战赛（VCC）分布内数据上具有显著超越随机特征对照的预测信息量，但在零样本跨筛选迁移中失败，并揭示了迁移失败与采样深度等设计因素之间的纠缠。 |
| [^343] | [UrbanDS: A Graph-Guided LLM Multi-Agent System for Data-Intensive Urban Tasks](https://arxiv.org/abs/2607.26724) | 该论文提出了UrbanDS，一种图引导的LLM多智能体系统，通过构建统一数据集图来组织可复用的数据集技能及其关系，从而解决数据密集型城市任务中从大规模异构数据中发现并利用相关信息这一难题。 |
| [^344] | [Does Runtime Topology Context Improve LLM-Generated Kubernetes Security Patches?](https://arxiv.org/abs/2607.25995) | 该论文提出KuTIE（Kubernetes拓扑智能引擎）系统，首次在受控条件下、跨多种依赖类型系统评估向大语言模型提供Kubernetes实时运行时服务拓扑上下文能否提升其生成的安全配置补丁的正确性，以解决模型因缺乏拓扑信息而生成破坏服务依赖的补丁的问题。 |
| [^345] | [Backspace as a Natural Experiment: An Accelerated Failure Time Model of Selective Post-Error Motor Impairment in Parkinsons Disease](https://arxiv.org/abs/2607.24796) | 本研究将退格键事件作为自然纠错情境，发现错误后停顿时长（而非错误前打字不稳定性）与帕金森病严重程度显著相关，表明被动键盘监测可选择性捕捉PD患者的错误后运动恢复损伤。 |
| [^346] | [AREX: Towards a Recursively Self-Improving Agent for Deep Research](https://arxiv.org/abs/2607.21461) | 该论文提出AREX，一类递归自我改进的深度研究智能体，利用“发现—验证”的不对称性，通过内层研究循环与外层按约束逐项审计的自我改进循环交替进行，并结合自主上下文压缩工具，实现对研究答案的持续递归改进。 |
| [^347] | [Computational Humor with Multimodal LLMs: Methods, Datasets, Evaluation, and Challenges](https://arxiv.org/abs/2607.19011) | 本综述系统梳理了多模态大语言模型在理解表情包、漫画等视觉幽默方面的方法、数据集与评估协议，并构建了以能力为中心的“识别—解释推理—生成”层次框架，揭示了该领域从任务专用融合模型向大模型方法的转变及面临的评估捷径等核心挑战。 |
| [^348] | [How Does Alignment Tuning Shape Representations of Sycophancy and Related Cue-Induced Biases in LLMs?](https://arxiv.org/abs/2607.18114) | 该研究发现大语言模型对谄媚性等线索诱导偏差的敏感性主要源于对齐微调而非预训练，且对齐模型中每种偏差都存在一个可被解码和干预的线性表示方向，可用于恢复无偏答案。 |
| [^349] | [Zero Hallucination, by Construction: Hallucination-Aware Layered Oversight for Trustworthy Enterprise AI](https://arxiv.org/abs/2607.17883) | 本文提出HALO保证架构，将幻觉从“可消除的问题”重新定义为“可控制的失效模式”，通过六层防御机制把“零幻觉”从模型属性转变为系统强制实施的属性，从而实现可信赖的企业级AI。 |
| [^350] | [SeerGuard: A Safety Framework for Mobile GUI Agents via World Model Prediction](https://arxiv.org/abs/2607.15550) | SeerGuard通过安全增强世界模型（SAWM）在动作执行前预测后果并评估风险，为移动GUI智能体提供了一个能够预防不可逆转错误操作的前瞻性安全框架。 |
| [^351] | [Anamnesis: An Open-Source Platform for Large-Scale Backstory-Conditioned Survey Simulation](https://arxiv.org/abs/2607.10628) | Anamnesis是一个开源平台，通过结构化叙事背景故事对大语言模型进行条件化，实现了在虚拟人群上进行人口可控、大规模且支持多模态的调查模拟。 |
| [^352] | [LUNA: Learning Universal 3D Human Animation Beyond Skinning](https://arxiv.org/abs/2606.31981) | LUNA提出了一种无需LBS蒙皮的通用神经人体动画模型，利用基于Transformer的运动回归器将图像、关键点、草图等多种2D控制信号直接映射为3D高斯变形，并通过LBS教师蒸馏与无标注视频的混合监督，突破了传统参数化人体模型的拟合限制与表现力瓶颈。 |
| [^353] | [DigitalCoach: Communication and Grounding Gaps in Human and Agentic Computer Use Coaching](https://arxiv.org/abs/2606.31980) | 该论文构建了包含72场人类专家-新手计算机使用辅导会话的多模态数据集DigitalCoach，揭示了当前最先进模型虽能生成与人类相似的辅导语句，但在解释、错误诊断和视觉定位方面显著不足，导致学习者被动跟随指令而非深度参与学习。 |
| [^354] | [Can LLMs Imagine Moral Alternatives Beyond Binary Dilemmas?](https://arxiv.org/abs/2606.31213) | 该论文提出MoralAltDataset数据集，通过在307个二元道德困境中引入折中和重构的替代选项，发现当替代方案可用时人类与15个LLM的道德选择分布均发生显著转变且一致性增强，但存在关键差异——LLM明显偏好GPT-5创作的替代方案，而人类的选择不受创作来源影响，揭示了机器与人类在“想象道德替代方案”能力上的差距。 |
| [^355] | [Self-Evolving World Models for LLM Agent Planning](https://arxiv.org/abs/2606.30639) | 提出自进化世界模型框架 WorldEvolver，通过情景记忆、语义记忆和选择性前瞻三个模块，在保持智能体与模型参数完全冻结的情况下持续修正部署时的上下文，从而提升长时程 LLM 智能体规划中前瞻预测的可靠性与下游决策成功率。 |
| [^356] | [Flow Reasoning Models: Turning Flows Into Efficient Recurrent Reasoners](https://arxiv.org/abs/2606.29150) | 提出流推理模型（FRMs），通过自条件化将流模型的一次性去噪转化为迭代式解精炼，使模型能够并行地做出并修改相互依赖的决策，实现高效的结构化推理。 |
| [^357] | [Event-Aligned Analysis of Multi-Rater Pain Assessments Using Continuous Wearable Physiology](https://arxiv.org/abs/2606.23705) | 该论文提出了一种感知评分者、事件对齐的分析框架，将多评分者的疼痛评分转化为离散疼痛变化事件并与可穿戴生理信号对齐，首次揭示了疼痛与生理的关系可能因评分者不同而存在差异，跨评分者汇总评估可能掩盖有意义的生理模式。 |
| [^358] | [Energy-Based Transformers as Predictors of Reading Difficulty](https://arxiv.org/abs/2606.23382) | 本文首次将基于能量的Transformer度量引入计算心理语言学，证明该能量度量在多个阅读时间语料库中是阅读难度的稳健预测因子，其解释力显著超越传统的惊讶度和注意力熵度量，并与Hopfield网络等联想记忆理论建立了形式化联系。 |
| [^359] | [DART: Draft-Agreement Routing for Training-Free Adaptive Thinking Budgets in Hybrid Reasoning Models](https://arxiv.org/abs/2606.23181) | DART是一种免训练的自适应路由框架，通过比较两个无思考草稿的一致性来决定是否需要深度推理并预测思考预算，在大幅减少思考token消耗的同时保持甚至提升模型在数学和代码任务上的准确率。 |
| [^360] | [Steer, Don't Solve: Training Small Critic Models for Large Code Agents](https://arxiv.org/abs/2606.21811) | 通过训练专门负责高层次规划的小型评论模型（4B/8B）在推理时引导大型编码智能体识别并纠正错误，在SWE-Bench Verified上显著提升多个更大规模编码智能体的解决率（最高提升16.0%）并降低推理成本。 |
| [^361] | [ReproRepo: Scaling Reproducibility Audits with GitHub Repository Issues](https://arxiv.org/abs/2606.18237) | ReproRepo提出利用GitHub上人工提交的议题作为天然监督信号，构建了可规模化的可复现性评估框架，并在1,149篇机器学习论文上验证了LLM智能体无需执行代码即可识别真实复现障碍的能力（最佳智能体可覆盖约90%的论文）。 |
| [^362] | [AfriSUD: A Dependency Treebank Collection for Evaluating Models on African Languages](https://arxiv.org/abs/2606.12708) | 该论文推出了首个覆盖九种非洲语言的大规模句法标注依存树库集合AfriSUD，并揭示现有模型在这些语言上仍存在显著的句法理解差距。 |
| [^363] | [Generativism: Toward a Learning Theory for the Age of Generative Artificial Intelligence](https://arxiv.org/abs/2606.12441) | 本文批判性反思了传统四大学习理论在生成式AI时代的局限，提出“生成主义”新学习理论，其核心观点是学习日益通过人类学习者与AI系统之间迭代式的知识共同建构来实现。 |
| [^364] | [Self-EmoQ: Plutchik-Guided Value-based Planning to Drive Streaming Emotional TTS](https://arxiv.org/abs/2606.09837) | 提出了一个在文本生成前进行情绪决策的即插即用LLM情绪规划框架，通过结合普鲁奇克情绪轮理论的混合奖励进行强化学习训练，以流式方式驱动下游情感语音合成，并在多个数据集上超越了提示与微调基线方法。 |
| [^365] | [DOG-DPO:Dynamic Optimization in Geometry for Safety Alignment](https://arxiv.org/abs/2606.07678) | 提出无需训练的数据选择框架DOG-DPO，将偏好对视为模型表示空间中的几何方向，通过分解全局锚定子空间与数据集特有残余子空间并最大化多样性覆盖，为DPO安全对齐筛选出广泛且非冗余的偏好数据子集。 |
| [^366] | [Enabling KV Caching of Shared Prefix for Diffusion Language Models](https://arxiv.org/abs/2606.07571) | 本文提出bicache，首个针对扩散语言模型共享前缀的KV缓存技术，解决了双向注意力下KV动态变化导致传统缓存失效的问题。 |
| [^367] | [GeM-NR: Geometry-Aware Multi-View Editing for Nonrigid Scene Changes](https://arxiv.org/abs/2606.05142) | 提出了一种无需训练的快速灵活方法GeM-NR，通过几何感知实现多视图一致的通用图像编辑，突破了以往方法只能进行刚性或仅外观编辑的限制，支持大幅改变场景几何和外观的非刚性编辑。 |
| [^368] | [Who Annotates in NLP? A Large-scale Assessment of Human Annotation Reporting between 2018 and 2025](https://arxiv.org/abs/2606.02255) | 首次对2018至2025年间主要NLP会议中的人类标注报告进行大规模任务级审计，提出统一的标注报告分类体系并借助经验证的LLM抽取流程构建了大规模标注报告数据集，揭示了标注者身份与过程控制等信息在论文中的普遍缺失。 |
| [^369] | [PlanarBench: Evaluating LLM Spatial Reasoning via Planar Graph Drawing](https://arxiv.org/abs/2606.02010) | 该论文提出PlanarBench基准，要求大语言模型仅根据边列表绘制平面图的无交叉ASCII图，并发现边数（即约束数量）比顶点数更能决定任务难度，为评估LLM空间推理能力提供了可控的测试环境。 |
| [^370] | [Skill Reuse as Compression in Agentic RL](https://arxiv.org/abs/2605.31509) | 该论文提出基于最小描述长度（MDL）原则的ReuseRL框架，通过从成功轨迹中提取共享技能字典并惩罚难以压缩的特异行为，显著提升了智能体强化学习的分布内外泛化能力。 |
| [^371] | [MusTBench: Benchmarking and Advancing Temporal Grounding in Music LLMs](https://arxiv.org/abs/2605.29300) | 该论文提出了经音乐专家验证的MusTBench基准和涵盖四阶段优化的MusT方案，用于评估并提升音乐大语言模型将回答准确锚定到音频正确时间段的能力。 |
| [^372] | [The Importance of Being Statistically Earnest: A Critical Re-evaluation of GSM-Symbolic](https://arxiv.org/abs/2605.28700) | 该研究指出GSM-Symbolic基准的统计方法存在缺陷，重新评估发现20个开源模型中仅8个呈现统计显著的性能下降，且数据集中整数分布系统性偏大是重要混淆因素，从而质疑了“大语言模型缺乏真正推理能力”的结论。 |
| [^373] | [Diffusion Large Language Models for Visual Speech Recognition](https://arxiv.org/abs/2605.28456) | 提出了首个基于扩散大语言模型的视觉语音识别框架DLLM-VSR，通过迭代掩码去噪与灵活顺序解码，利用置信度引导的解掩码机制以双向上下文细化模糊词元，并采用两阶段训练策略分离内容对齐与长度建模。 |
| [^374] | [UniACE: A Unified Framework for Evaluating LLM Agentic Capabilities](https://arxiv.org/abs/2605.27898) | UniACE通过将基准测试表示为“指令—工具—环境”三元组、采用共享的任务无关执行框架和统一执行条件（含离线快照模式），实现了以模型为中心的LLM智能体能力标准化评估，解决了跨基准测试比较受实现差异和资源条件影响的问题。 |
| [^375] | [When the Strongest Teacher Is Not the Best Teacher: Student-Centric Answer Selection](https://arxiv.org/abs/2605.26872) | 论文提出SCAS框架，证明最强教师的正确答案未必是学生的最佳训练监督，并通过逐token梯度分解推导出仅需前向计算的高效代理指标，依据学生中心学习成本来选择最适合学生的教师答案。 |
| [^376] | [Do as I Say, Not as I Do: Instruction-Induction Conflict in LLMs](https://arxiv.org/abs/2605.20382) | 该研究通过构造用户指令与硬编码对话模式相冲突的实验场景，发现大语言模型的指令遵循能力在不同模型间差异巨大（1%到99%）且与常规能力基准基本无关，其鲁棒性取决于指令内容与模型价值先验的一致性以及输出格式。 |
| [^377] | [Universal Approximation of Nonlinear Operators and Their Derivatives](https://arxiv.org/abs/2605.15285) | 本文首次证明了关于 $k$ 阶可微非线性算子及其导数的万能逼近定理，首次将经典万能逼近定理完整推广至无穷维巴拿赫空间与算子学习领域，并由此开启了导数感知算子学习（DIOL）的新方向。 |
| [^378] | [SkillRet: A Large-Scale Benchmark for Skill Retrieval in LLM Agents](https://arxiv.org/abs/2605.05726) | 该论文提出了SkillRet，首个面向LLM智能体技能检索的大规模基准，包含16,129个带结构化语义标签和两级分类体系的公开技能，以及63,259个训练样本和4,392个技能池互不相交的评估查询，填补了该领域基准匮乏的空白。 |
| [^379] | [Causal Probing for Internal Visual Representations in Multimodal Large Language Models](https://arxiv.org/abs/2605.05593) | 该研究提出基于激活引导的因果框架，揭示了多模态大语言模型中实体知识局部化编码而抽象概念全局分布的分化现象，并证明模型深度的增加是编码复杂抽象概念这一缩放定律背后的机制性驱动因素。 |
| [^380] | [D3-Gym: Constructing Real-World Verifiable Environments for Data-Driven Discovery](https://arxiv.org/abs/2604.27977) | 本文提出了首个为科学数据驱动发现自动构建的可验证环境数据集D3-Gym，包含565个来自真实科学代码库的任务，其自动评估脚本与人工标注达到87.5%的一致性，且基于其轨迹训练可显著提升Qwen3模型在ScienceAgentBench上的表现。 |
| [^381] | [The Topological Trouble With Transformers](https://arxiv.org/abs/2604.17121) | 该论文揭示了Transformer纯前馈架构在动态状态追踪上的根本性拓扑缺陷——状态表示随每个新输入被不断推向更深层直至耗尽模型深度，并论证时间上延伸的认知需要从显式思维轨迹回归到基于循环结构的隐式激活动力学。 |
| [^382] | [Agentic Large Language Models for Training-Free Neuro-Radiological Image Analysis](https://arxiv.org/abs/2604.16729) | 本文提出一种免训练的智能体流程，让大语言模型编排现成的专业工具，自主完成脑部MRI分析中从预处理到病理分割的复杂端到端工作流程。 |
| [^383] | [Global Attention with Linear Complexity for Exascale Generative Data Assimilation in Earth System Prediction](https://arxiv.org/abs/2604.16590) | STORM提出了一种单阶段生成式AI数据同化框架，将数据同化重构为基于扩散模型的贝叶斯后验采样，并结合线性复杂度的全局注意力算法，在Frontier上扩展至74,400个GPU、达到6 ExaFLOPs持续吞吐量，并可在34秒内实现32,768成员的大规模集合不确定性量化。 |
| [^384] | [Why Fine-Tuning Encourages Hallucinations and How to Fix It](https://arxiv.org/abs/2604.15574) | 该论文提出一种基于自蒸馏的监督微调方法，通过正则化输出分布漂移，使模型在学习新事实的同时最大限度减少对预训练知识的幻觉，并证明在无需学习新知识时冻结参数组也能在保持任务性能的前提下降低幻觉。 |
| [^385] | [What Drives Representation Steering? A Mechanistic Case Study on Steering Refusal](https://arxiv.org/abs/2604.08524) | 本研究通过多词元激活修补框架对LLM拒绝行为的转向机制进行案例研究，发现不同转向方法在同一层利用功能可互换的回路，且转向向量主要通过注意力机制的OV回路发挥作用而几乎不依赖QK回路。 |
| [^386] | [KV Cache Offloading for Context-Intensive Tasks](https://arxiv.org/abs/2604.08426) | 该论文创建并发布了Text2JSON基准测试，揭示现代KV缓存卸载技术在需要从输入提示中提取大量信息的上下文密集型任务上，会导致Llama 3和Qwen 3模型出现显著的性能下降。 |
| [^387] | [DiffHDR: Re-Exposing LDR Videos with Video Diffusion Models](https://arxiv.org/abs/2604.06161) | DiffHDR将LDR视频到HDR的转换建模为视频扩散模型潜在空间中的生成式辐射修复任务，通过在对数-Gamma色彩空间中利用预训练视频扩散模型的时空生成先验，在过曝和欠曝区域合成逼真的HDR辐射，实现LDR视频的有效重新曝光。 |
| [^388] | [Training-Free Refinement of Flow Matching with Divergence-based Sampling](https://arxiv.org/abs/2604.04646) | 提出了流散度采样器（FDS），一种免训练的即插即用框架，通过边缘速度场的散度信号精炼中间状态，引导样本避开低密度区域，从而一致提升流匹配模型的生成保真度。 |
| [^389] | [TRU: Targeted Reverse Update for Efficient Multimodal Recommendation Unlearning](https://arxiv.org/abs/2604.02183) | 该论文提出面向多模态推荐系统的即插即用定向反向更新（TRU）遗忘框架，通过针对待删除数据在排序行为、模态分支和模型模块间不均匀分布的影响进行定向处理，克服目标物品持续性、模态不平衡和模块敏感性集中三大瓶颈，实现高效的用户数据遗忘。 |
| [^390] | [IWP: Token Pruning as Implicit Weight Pruning in Large Vision Language Models](https://arxiv.org/abs/2604.00757) | 该论文提出IWP框架，将注意力重新表述为由各token键值对生成的秩1外积之和构成的隐式线性层，从而把token剪枝转化为选择最优秩1更新子集以逼近原始对偶权重矩阵，并据此推导出同时衡量token信息量与信息冗余度的新指标，实现无需训练的高效视觉token剪枝。 |
| [^391] | [Oblivion: Self-Adaptive Agentic Memory Control through Decay-Driven Activation](https://arxiv.org/abs/2604.00131) | Oblivion框架借鉴人类选择性遗忘机制，将遗忘建模为衰减驱动的可及性降低而非删除，并通过解耦读取路径（基于不确定性决定何时查询记忆）与写入路径（强化贡献性记忆），为LLM智能体实现按需动态加载的层次化记忆组织。 |
| [^392] | [VectorGym: A Multi-Task Benchmark for SVG Code Generation, Sketching and Editing](https://arxiv.org/abs/2603.29852) | VectorGym是首个与专业设计工作流程对齐的SVG综合性基准，包含四大任务（草图生成SVG、SVG编辑、文本生成SVG、SVG描述）及专家人工标注，并提供基于GRPO的多任务强化学习基线。 |
| [^393] | [APEX-EM: Non-Parametric Online Learning for Autonomous Agents via Structured Procedural-Episodic Experience Replay](https://arxiv.org/abs/2603.29093) | APEX-EM提出了一种无需更新模型权重的非参数化经验记忆方法，通过程序性知识图谱存储完整的任务轨迹并同时索引成功与失败经验，使LLM智能体能够复用过往经验而无需重复推理，在相同底层模型对比下BigCodeBench迁移任务上提升7.6个百分点。 |
| [^394] | [Revealing Multi-View Hallucination in Large Vision-Language Models](https://arxiv.org/abs/2603.23934) | 该论文首次揭示并定义了大型视觉语言模型中的多视角幻觉问题，构建了包含4.8k问答对的MVH-Bench基准进行系统评估，并提出无需训练的参考偏移对比解码（RSCD）技术，通过注意力掩码抑制视觉干扰，性能提升最高达94.8分。 |
| [^395] | [MineDraft: A Framework for Batch Parallel Speculative Decoding](https://arxiv.org/abs/2603.18016) | MineDraft提出一种批量并行投机解码框架，通过同时维护两批请求，将一批的草稿生成与另一批的验证重叠执行，有效隐藏草稿延迟，相比标准投机解码吞吐量最高提升75%、端到端延迟最高降低39%。 |
| [^396] | [SCALE:Scalable Conditional Atlas-Level Endpoint transport for virtual cell perturbation prediction](https://arxiv.org/abs/2603.17380) | SCALE是一种将细胞表示为无序集合的条件传输模型，无需细胞级配对即可预测扰动后的细胞群体，在基因、化学、发育、免疫扰动及CRISPR数据上均优于现有方法。 |
| [^397] | [V-Co: A Closer Look at Visual Representation Alignment via Co-Denoising](https://arxiv.org/abs/2603.16792) | 本文提出V-Co，在统一的JiT框架下对视觉协同去噪进行系统性研究，分离出使视觉表征对齐有效提升像素空间扩散模型训练的关键设计要素。 |
| [^398] | [Is Human Annotation Necessary? Iterative MBR Distillation for Error Span Detection in Machine Translation](https://arxiv.org/abs/2603.12983) | 该论文提出了一种基于最小贝叶斯风险解码的迭代MBR蒸馏自演化框架，通过利用现成大语言模型生成伪标签进行自我训练，无需人工标注即可在机器翻译错误片段检测任务上超越基于人工标注的监督基线模型。 |
| [^399] | [RetroReasoner: A Reasoning LLM for Strategic Retrosynthesis Prediction](https://arxiv.org/abs/2603.12666) | 本文提出RetroReasoner，一个通过结构化切断理由的监督微调和往返奖励强化学习训练的逆合成推理大语言模型，能够显式模拟化学家的键切断策略思维并验证预测反应物的有效性。 |
| [^400] | [HEAL: Hindsight Entropy-Assisted Learning for Reasoning Distillation](https://arxiv.org/abs/2603.10359) | HEAL 提出了一种无需强化学习的推理蒸馏框架，通过熵动态检测推理断点并注入事后提示来修复失败轨迹，突破了传统拒绝采样造成的“教师天花板”，从而将大型推理模型的推理能力更有效地蒸馏到小模型中。 |
| [^401] | [Guided Prompt Evolution for Vision-Language Models Adaptation](https://arxiv.org/abs/2603.09493) | 提出EvoPrompt框架，通过模态共享提示投影器生成层次化提示，并采用将低秩更新解耦为方向与幅值分量的演化训练策略，在保留预训练语义方向的前提下仅调整幅值，从而在有限标注数据下实现视觉语言模型无遗忘的知识保留适配。 |
| [^402] | [Reconstruct! Don't Encode: Self-Supervised Representation Reconstruction Loss for High-Intelligibility and Low-Latency Streaming Neural Audio Codec](https://arxiv.org/abs/2603.05887) | 提出自监督表征重构损失（SSRR）用于训练神经音频编解码器，显著加速收敛并提升语音可懂度，使 JHCodec 在零前瞻、低延迟的流式架构下于 LibriSpeech test-clean 上取得最佳的词错误率和字错误率。 |
| [^403] | [MMAI Gym for Science: Training Liquid Foundation Models for Drug Discovery](https://arxiv.org/abs/2603.03517) | 本文提出MMAI Gym for Science一站式训练框架，通过教会基础模型“分子的语言”，训练出更小规模的液体基础模型（LFM），在分子优化、ADMET预测等药物发现任务上超越了规模大得多的通用或专业模型。 |
| [^404] | [Channel-Adaptive Edge AI: Maximizing Inference Throughput by Adapting Computational Complexity to Channel States](https://arxiv.org/abs/2603.03146) | 本文提出了一个可处理的端到端推理精度分析模型，并据此设计了信道自适应AI算法，通过根据信道状态动态调整模型计算复杂度（利用早退机制），在时延和精度约束下最大化边缘推理吞吐量。 |
| [^405] | [Make Some Noise: Unsupervised Remote Sensing Change Detection Using Latent Space Perturbations](https://arxiv.org/abs/2602.19881) | 提出MaSoN框架，在训练中直接于潜在特征空间内根据目标数据的特征统计动态合成多样化变化，摆脱了对预定义变化类型假设的依赖，从而提升无监督遥感变化检测在罕见和复杂场景下的泛化能力。 |
| [^406] | [Learning to Remember: End-to-End Training of Memory Agents for Long-Context Reasoning](https://arxiv.org/abs/2602.18493) | 该论文提出统一记忆智能体（UMA），通过任务分层GRPO算法端到端训练单一策略来维护可复用的结构化外部记忆库，并配套提出Ledger-QA诊断基准，显著提升了长上下文推理中跨会话状态跟踪与问答的性能。 |
| [^407] | [Ontology-Guided Neuro-Symbolic Inference: Grounding Language Models with Mathematical Domain Knowledge](https://arxiv.org/abs/2602.17826) | 该论文提出一种结合OpenMath本体、混合检索与交叉编码器重排序的神经符号流水线，将数学领域知识注入语言模型提示中，实验表明高质量检索的本体上下文能提升模型在MATH基准上的表现，但不相关上下文会损害性能。 |
| [^408] | [Persistent Entropy as a Detector of Phase Transitions](https://arxiv.org/abs/2602.09058) | 本文建立了与模型无关的理论定理，通过识别持续权重中的“分散-凝聚”机制并推导出两状态间熵差的显式高概率下界，首次为利用持续熵检测相变提供了严格的理论保证，并据此证明卷积网络学习滤波器的环形组织源于一次尖锐的拓扑相变。 |
| [^409] | [MAS-ProVe: Understanding the Process Verification of Multi-Agent Systems](https://arxiv.org/abs/2602.03053) | 本文提出MAS-ProVe，首次对多智能体系统中的过程验证展开系统性实证研究，涵盖三种验证范式、两种验证粒度、五种验证器和四种上下文管理策略，并发现过程级验证在多智能体系统中并不能持续稳定地带来改进。 |
| [^410] | [Think Like a Doctor: Conversational Diagnosis through the Exploration of Diagnostic Knowledge Graphs](https://arxiv.org/abs/2602.01995) | 该论文提出了一种通过探索诊断知识图谱进行两步推理（先生成诊断假设、再通过澄清性问题反复验证）的对话式诊断系统，并结合基于人设的患者模拟器PatientSim与MIMIC-IV患者档案进行更贴近真实场景的评估。 |
| [^411] | [FloydNet: A Learning Paradigm for Global Relational Reasoning](https://arxiv.org/abs/2601.19094) | 提出FloydNet与关键注意力机制（PA），借鉴Floyd–Warshall的“配对-枢轴”结构，通过维护有序对状态并在枢轴上聚合候选关系实现全局关系推理，并推广为支持有序k元组的k-FloydNet框架，其图判别能力与对应的WL同构测试相当。 |
| [^412] | [LifeAgentBench: Benchmarking LLMs for Long-Horizon, Cross-Dimensional Lifestyle Health Reasoning](https://arxiv.org/abs/2601.13880) | 本文提出LifeAgentBench——一个包含22,573个问题、面向长时程、跨维度、多用户生活方式健康推理的大规模问答基准，通过系统评估13个代表性大语言模型，揭示了它们在长时程聚合和跨维度推理上的关键瓶颈。 |
| [^413] | [Beyond Static Summarization: Proactive Memory Extraction for LLM Agents](https://arxiv.org/abs/2601.04463) | 该论文提出主动记忆提取框架ProMem，通过分离细节、事件与关系并采用分类提取策略、完整性检查和原子级事实验证，解决了现有记忆提取方法因提前进行和一次性提取而导致的信息丢失与幻觉残留问题。 |
| [^414] | [A Hybrid Insider Threat Detection Framework Combining Multi-Agent Simulation, Layered SIEM Correlation, and Theory-of-Mind Reasoning](https://arxiv.org/abs/2601.04243) | 该论文提出一种融合多智能体仿真、分层SIEM关联、证据门控与心智理论推理的混合式内部威胁检测框架，将电子邮件作为社会工程证据流与系统事件关联分析，使参与者级F1分数从0.567显著提升至0.944并大幅降低误报。 |
| [^415] | [Hidden State Poisoning Attacks against Mamba-based Language Models](https://arxiv.org/abs/2601.01972) | 该论文首次揭示了针对Mamba等状态空间语言模型的隐状态投毒攻击（HiSPA）——特定短输入短语可不可逆地覆盖模型隐藏状态导致部分失忆，并提出RoBench-25基准证实了包括520亿参数的Jamba混合模型在内的SSMs对此类攻击的脆弱性，而纯Transformer模型则不受影响。 |
| [^416] | [Multilingual Medical Reasoning for Question Answering with Large Language Models](https://arxiv.org/abs/2512.05658) | 该论文提出了一种基于维基百科医学知识、采用检索增强生成方法构建多语言（英语、意大利语、西班牙语）医学推理轨迹的技术，生成了50万条推理数据，并证明这些数据在少样本学习和监督微调两种方式下均能显著提升大语言模型在医学问答任务上的表现。 |
| [^417] | [3D-Consistent Multi-View Editing by Correspondence Guidance](https://arxiv.org/abs/2511.22228) | 提出了一种无需训练的引导框架，通过引入一致性损失确保对应点在编辑后保持相似，从而在去噪过程中实现几何和光度上3D一致的多视图图像编辑。 |
| [^418] | [The Alexander-Hirschowitz theorem for neurovarieties](https://arxiv.org/abs/2511.19703) | 本文通过独立的几何方法证明了激活次数满足线性界 d_i ≥ 2n_i−1 时多项式神经网络的神经簇对任意输出数都是非亏缺的，并进一步证明了多输出架构在相同次数界下的全局可辨识性。 |
| [^419] | [A Machine Learning-Driven Solution for Denoising Inertial Confinement Fusion Images](https://arxiv.org/abs/2511.16717) | 该研究提出了一种结合Cohen-Daubechies-Feauveau小波的无监督自编码器机器学习方法，用于惯性约束聚变中子图像去噪，能够在高斯和泊松噪声共存的情况下保护图像关键特征，从而提升迭代图像重建的保真度。 |
| [^420] | [Multi-Agent LLM Orchestration Achieves Deterministic, High-Quality Decision Support for Incident Response](https://arxiv.org/abs/2511.15755) | 该论文提出MyAntFarm.ai框架，通过348次受控试验证明多智能体LLM编排相比单智能体方法可将可操作建议率从1.7%提升至100%，实现行动具体性提升80倍、方案正确性提升140倍且质量零方差的确定性事件响应决策支持。 |
| [^421] | [SEBA: Sample-Efficient Black-Box Attacks on Visual Reinforcement Learning](https://arxiv.org/abs/2511.09681) | SEBA提出了一种针对视觉强化学习的样本高效黑盒攻击框架，通过结合影子Q模型、生成对抗网络和世界模型，以极少的真实环境查询实现对基于图像的连续控制智能体的有效对抗攻击。 |
| [^422] | [Individualized Algorithmic Advice as a Strategic Signal on Competitive Markets](https://arxiv.org/abs/2511.09454) | 古诺竞争实验表明，与均衡一致的个性化算法建议能促进稳定收敛，而串谋性向下偏倚的建议会诱发默示串谋行为（产量不足与超竞争利润），且个性化建议比集体建议更容易被参与者采纳。 |
| [^423] | [KGFR: A Foundation Retriever for Generalized Knowledge Graph Question Answering](https://arxiv.org/abs/2511.04093) | 提出LLM-KGFR协作框架，通过LLM生成的关系描述、基于实体角色的初始化实现零样本泛化，并借助非对称渐进传播高效处理大规模图谱，从而支持对未见知识图谱的通用问答。 |
| [^424] | [Multi-Step Knowledge Interaction Analysis via Rank-2 Subspace Disentanglement](https://arxiv.org/abs/2511.01706) | 该论文提出一种新颖的秩-2投影子空间来更准确地解缠大语言模型中参数化知识与情境知识的贡献，并首次实现了对自然语言解释更长生成序列中知识交互的多步分析。 |
| [^425] | [Can machines think efficiently?](https://arxiv.org/abs/2510.26954) | 该论文提出在图灵测试中引入能量消耗约束，从效率视角重新评估机器智能，从而为智能评估提供了原测试所缺乏的可测量的实际标准。 |
| [^426] | [Taming Modality Entanglement in Continual Audio-Visual Segmentation](https://arxiv.org/abs/2510.17234) | 该论文提出持续音视频分割（CAVS）这一新任务，识别出多模态语义漂移与共现混淆两大关键挑战，并设计了基于碰撞的多模态重放（CMR）框架，以解决细粒度多模态持续学习中的模态纠缠问题。 |
| [^427] | [HugAgent: A Human Simulation Benchmark for Individual-Level Reasoning](https://arxiv.org/abs/2510.15144) | 该论文提出HugAgent基准，从个性化推理、认知对齐和开放式数据三个维度重新定义人类推理模拟，评估模型能否基于某人历史观点的部分证据，预测该特定个体在分布外场景中的行为反应与推理动态。 |
| [^428] | [Compositional Machine Design as Program Synthesis with LLMs](https://arxiv.org/abs/2510.14980) | 该论文提出将机器设计视为一种以物理模拟验证为依据的程序合成新任务——组合式机器设计，并构建了基于游戏《Besiege》的测试平台BesiegeField，用于评测大语言模型在多种工作流下组合标准部件设计机器的能力。 |
| [^429] | [One-shot Style Transfer LLM log-probabilities for Authorship Attribution and Verification](https://arxiv.org/abs/2510.13302) | 本文提出一种无监督框架，利用大语言模型的对数概率衡量文本间的风格可迁移性，无需显式监督即可在作者验证任务上显著超越基于提示的无监督基线，并在足够模型规模下与对比学习基线相当或更优。 |
| [^430] | [TopoAlign: A Framework for Aligning Code to Math via Topological Decomposition](https://arxiv.org/abs/2510.11944) | TopoAlign通过将代码分解为文档字符串、主函数和依赖函数并重新组装对齐，弥合了代码与形式化数学之间的结构与句法差异，从而将大规模代码仓库转化为可用于提升数学LLM自动形式化能力的训练资源。 |
| [^431] | [Uncovering the Computational Ingredients of Human-Like Representations in LLMs](https://arxiv.org/abs/2510.01030) | 本研究借助认知科学中的三元组相似度任务，对75个以上大语言模型进行了系统评估，以识别影响类人概念表示形成的关键计算要素（如指令微调），同时弥补了现有基准无法衡量模型与人类表示对齐程度的不足。 |
| [^432] | [SupraTok: Cross-Boundary Tokenization for Enhanced Language Model Performance](https://arxiv.org/abs/2508.11857) | SupraTok是一种跨越空白符边界的创新分词器，通过熵筛选、PMI引导的课程训练和多语言处理三大模块，在压缩率上比标准BPE提升17.5%，并比SuperBPE训练快2.1倍。 |
| [^433] | [BiasGym: A Simple and Generalizable Framework for Analyzing and Removing Biases through Injection](https://arxiv.org/abs/2508.08855) | 提出BiasGym框架，通过在冻结的LLM中注入特定偏见信号，再利用这些信号定位并抑制或引导导致偏见行为的模型组件，实现偏见的可靠分析与消除。 |
| [^434] | [Unsupervised Partner Design Enables Robust Ad-hoc Teamwork](https://arxiv.org/abs/2508.06336) | 提出无监督伙伴设计（UPD），通过即时生成训练伙伴并基于可学习性准则自适应选择，无需预训练伙伴种群或手动调参即可实现鲁棒的临时团队协作，在多个基准任务和人机用户研究中均表现出卓越性能。 |
| [^435] | [GeoGR^2:Zero-Shot Geospatial Inference via Geostatistically-Guided Iterative Refinement with LLMs](https://arxiv.org/abs/2508.04080) | 提出GeoGR^2框架，将零样本地理空间预测建模为动态构建图上的迭代消息传递过程，通过拓扑、特征和更新三个动态算子引导大语言模型实现空间一致、消除人口稠密偏差的地理空间推断。 |
| [^436] | [ParaStudent: Closing the Sim2Real Gap in User Simulators for AI Tutor Evaluation](https://arxiv.org/abs/2507.12674) | ParaStudent是一个通过微调来模拟初学者编程修改的框架，其模拟结果更贴近真实学生代码分布，可用于AI导师部署前的反馈评估与筛选。 |
| [^437] | [Towards Provable and Scalable Training of Quantized Neural Networks with Ising Optimization](https://arxiv.org/abs/2506.18240) | 该论文提出一个具有可证明保证的精确二次约束二元优化（QCBO）框架，将量化神经网络训练编译为具有零松弛间隙的完全正凸优化问题，并通过逐样本分解下界优化（DLBO）将伊辛求解规模从数据集级别降至单样本级别，从而实现对量化神经网络可证明且可扩展的训练。 |
| [^438] | [ViPlan: A Benchmark for Visual Planning with Symbolic Predicates and Vision-Language Models](https://arxiv.org/abs/2505.13180) | 提出首个开源视觉规划基准ViPlan，用于比较VLM接地的符号规划方法与直接VLM规划方法，发现VLM作为接地器在Blocksworld中显著优于直接VLM规划（46%对9%）。 |
| [^439] | [A Token is Worth over 1,000 Tokens: Efficient Knowledge Distillation through Low-Rank Clone](https://arxiv.org/abs/2505.12781) | 提出低秩克隆（LRC）高效预训练方法，利用一组低秩投影矩阵同时实现教师权重的软剪枝和学生激活（含FFN信号）的克隆对齐，从而高效构建与强大教师模型行为等价的小语言模型。 |
| [^440] | [SARTM: Segment Any RGB Thermal Model with Language aided Distillation](https://arxiv.org/abs/2505.01950) | 提出SARTM框架，通过LoRA微调和语言引导蒸馏将SAM适配到RGB-热红外语义分割任务，在低光和过曝等恶劣光照条件下实现鲁棒的场景理解。 |
| [^441] | [X-SG$^2$S: Safe and Generalizable Gaussian Splatting with X-dimensional Watermarks](https://arxiv.org/abs/2502.10475) | 提出了X-SG$^2$S框架，能够在3D高斯泼溅场景中同时注入1D到3D的多模态水印以实现版权保护，同时保持原始场景的高保真度。 |
| [^442] | [Automatic Item Generation for Personality Situational Judgment Tests with Large Language Models](https://arxiv.org/abs/2412.12144) | 本研究开发并评估了一个基于大语言模型（GPT-4和ChatGPT-5）自动生成人格情境判断测试题目的结构化、可推广框架，通过三项研究系统考察了提示词设计与温度设置对题目内容效度的影响，显著降低了传统SJT开发对专家的依赖。 |
| [^443] | [Keep Everyone Happy: Online Fair Division of Numerous Items with Few Copies](https://arxiv.org/abs/2408.12845) | 针对物品数量多而副本少的在线公平分配难题，本文创新性地假设效用是物品-智能体特征的未知函数，并将其建模为上下文老虎机问题，从而克服了无法准确估计所有物品-智能体对效用的局限。 |
| [^444] | [FedReview: A Review Mechanism for Rejecting Poisoned Updates in Federated Learning](https://arxiv.org/abs/2402.16934) | 提出了FedReview机制，通过随机分配评审员客户端来识别和拒绝联邦学习中的潜在毒化更新，并采用多数表决机制来整合排名并移除这些更新。 |
| [^445] | [Building Expressive and Tractable Probabilistic Generative Models: A Review](https://arxiv.org/abs/2402.00759) | 本文综述了富有表现力和可处理的概率生成建模领域的进展和技术，并重点关注了概率电路。文章提供了关于表达能力和可处理性之间权衡的统一视角，并说明了设计原则和算法扩展，成功地构建了富有表现力和高效的概率电路。此外，文章还讨论了最新的深度和混合概率电路研究，并概述了未来研究的挑战和开放性问题。 |

# 详细

[^1]: 基于轨迹感知评估的高效软件工程（SWE）智能体基准测试

    Efficient SWE Agent Benchmarking via Trajectory-Aware Evaluation

    [https://arxiv.org/abs/2609.01603](https://arxiv.org/abs/2609.01603)

    提出PTA-IRT框架，将历史执行轨迹作为特权信息融合过程与结果信号，在低校准预算下更准确地恢复软件工程智能体的完整基准分数与排名。

    

    在真实基准上评估软件工程智能体的成本很高，因为每个任务可能需要多步代码探索、修改和测试执行。现有的高效评估方法通过选择代表性子集来估计完整基准的性能，但它们在很大程度上仅依赖结果：它们拟合历史的通过/失败响应矩阵或静态任务语义，丢弃了智能体如何解决问题的信息。我们提出了PTA-IRT，一个融合过程与结果信号的特权轨迹感知项目反应理论框架。历史执行轨迹提供了超越通过/失败的过程级证据，例如探索的上下文、尝试的编辑和解题路径，PTA-IRT将这些作为特权信息用于校准子集选择和能力估计。在低校准预算下，PTA-IRT在四个软件工程基准上的分数和排名恢复方面始终优于现有的IRT基线方法。代码和数据均已公开。

    arXiv:2609.01603v1 Announce Type: cross  Abstract: Evaluating software engineering agents on realistic benchmarks is costly, since each task may require multi-step code exploration, modification, and test execution. Existing efficient evaluation methods select representative subsets to estimate full-benchmark performance, but are largely result-only: they fit historical pass/fail response matrices or static task semantics, discarding how agents solve problems. We propose PTA-IRT, a Privileged Trajectory-Aware Item Response Theory framework that fuses process and outcome signals. Historical execution trajectories supply process-level evidence beyond pass/fail, such as explored context, attempted edits, and solving paths, which PTA-IRT uses as privileged information for calibration subset selection and ability estimation. Under low calibration budgets, PTA-IRT consistently outperforms prior IRT baselines on score and ranking recovery across four SWE benchmarks. Code and data are publicly
    
[^2]: 面向仓库级代码生成的自适应关键Token感知检索

    Adaptive Critical Token-Aware Retrieval for Repository-Level Code Generation

    [https://arxiv.org/abs/2609.01601](https://arxiv.org/abs/2609.01601)

    该论文提出ACToR，通过识别LLM自回归代码生成过程中容易出错的关键token位置，并自适应地为这些位置检索细粒度的仓库上下文，从而提升仓库级代码生成的功能正确性。

    

    仓库级代码生成任务需要合成既满足任务要求、又与目标仓库上下文保持一致的代码。由于真实世界的仓库往往超出大语言模型（LLM）的输入长度限制，现有方法通常采用检索增强生成（RAG）来提供仓库特定的上下文。尽管这些方法改善了仓库上下文的检索，但它们通常将上下文作为任务级支持来提供，而没有显式识别生成过程中需要细粒度仓库上下文的关键token。在LLM的自回归生成过程中，错误往往集中在少数决定性位置上：一旦这些token被错误生成，后续代码可能沿着错误的语义路径发展，最终导致功能失效。我们将这些位置称为“关键token”。在本文中，我们提出了ACToR，一种自适应关键（token感知检索框架，摘要在此处截断）

    arXiv:2609.01601v1 Announce Type: cross  Abstract: The repository-level code generation task requires synthesizing code that satisfies task requirements while remaining consistent with the target repository context. Since real-world repositories often exceed the input length limits of LLMs, existing approaches commonly adopt retrieval-augmented generation (RAG) to provide repository-specific context. Despite improving repository-context retrieval, existing methods typically provide context as task-level support, without explicitly identifying the critical tokens that require fine-grained repository context during generation. During the autoregressive generation process of LLMs, errors often concentrate at a small number of decisive positions: once such tokens are generated incorrectly, subsequent code may follow an incorrect semantic path and eventually lead to functional failure. We refer to these positions as "critical tokens". In this paper, we propose ACToR, an adaptive critical to
    
[^3]: CordisBench：语言模型能否对动态智能体框架中的组件生命周期进行推理？

    CordisBench: Can Language Models Reason About Component Lifecycles in Dynamic Agent Harnesses?

    [https://arxiv.org/abs/2609.01600](https://arxiv.org/abs/2609.01600)

    该论文提出了 CordisBench——一个包含 1,200 道题目的基准，用于评估语言模型在动态智能体框架中对组件依赖与清理等生命周期问题的推理能力，发现模型在小规模系统上表现良好，但随相关交互数量增多可靠性显著下降。

    

    动态智能体框架允许语言模型改变塑造其自身执行过程的软件。这种灵活性带来了新的推理负担：局部的插件变更可能通过依赖关系和清理过程进行传播。我们提出了 CordisBench，一个包含 1,200 道题目的生命周期推理基准。该基准将受控的形式化设定与针对 Cordis（一个管理组件依赖与清理的运行时环境）执行的程序相结合，要求模型识别受影响的组件、预测特定拆卸顺序后的状态、判断哪些条件在所有或部分顺序下成立，并选择在实际执行时能够成功的重新配置方案。在这些任务上，我们在低推理努力设置下评估了三个面向效率的模型，涉及 2、4、8、16、24 或 32 个相关交互，并使用确定性的任务特定评分。模型通常能较好地处理小型系统，但随着相关交互数量的增加，其可靠性逐渐下降，尤其是在……（原文截断）

    arXiv:2609.01600v1 Announce Type: cross  Abstract: Dynamic agent harnesses let language models change the software that shapes their own execution. This flexibility brings a new reasoning burden: a local plugin change can propagate through dependencies and cleanup. We introduce CordisBench, a 1,200-question benchmark of this lifecycle reasoning. It combines a controlled formal setting with programs executed against Cordis, a runtime that manages component dependencies and cleanup, and asks models to identify affected components, predict state after a specified teardown order, determine which conditions hold under all or some orders, and choose reconfigurations that succeed when executed. Across these tasks, we evaluate three efficiency-oriented models at low reasoning effort with 2, 4, 8, 16, 24, or 32 relevant interactions, using deterministic task-specific scoring. Models usually handle small systems well but grow less reliable as more interactions become relevant, especially when pr
    
[^4]: 言语强化学习的兴起

    The Rise of Verbal Reinforcement Learning

    [https://arxiv.org/abs/2609.01597](https://arxiv.org/abs/2609.01597)

    本文首次对“言语强化学习”这一新兴范式进行了统一阐述，根据言语反馈生效的时机与作用对象，将其系统归纳为语言作为基础定位信号、语言作为审慎反馈以及语言作为学习信号三大支柱。

    

    自然语言正在成为改进语言智能体的主要反馈渠道，它能够以人类和现代语言模型都可解读的形式传达意图、偏好和因果结构。我们将这一范式称为言语强化学习（Verbal Reinforcement Learning, VRL），并首次对其进行了统一的阐述。我们围绕一个单一的主轴来组织该领域，即言语反馈在智能体生命周期中“何时”生效以及“修改什么”，由此归纳出三大支柱：（1）语言作为基础定位信号，语言通过指定目标、状态和奖励结构来定义任务本身；（2）语言作为审慎反馈，自然语言在测试时引导推理，而无需更新模型参数；（3）语言作为学习信号，基于语言的反馈通过训练来塑造模型参数。在每个支柱内，我们综合了代表性工作，区分了关键……

    arXiv:2609.01597v1 Announce Type: cross  Abstract: Natural language is emerging as a primary feedback channel for improving language agents, capable of conveying intent, preferences, and causal structure in forms interpretable by both humans and modern language models. We call this paradigm Verbal Reinforcement Learning (VRL) and offer the first unified account of it. We organize the field around a single axis, \textit{when} verbal feedback takes effect in an agent's lifecycle and \textit{what} it modifies, yielding three pillars: (1) \textbf{Language as Grounding Signal}, where language defines the task itself by specifying goals, states, and reward structures; (2) \textbf{Language as Deliberative Feedback}, where natural language guides reasoning at test time without the need to update model parameters; (3) \textbf{Language as Learning Signal}, where language-based feedback shapes model parameters through training. Within each pillar, we synthesize representative work, distinguish ke
    
[^5]: 面向对齐与控制的机制设计

    Mechanism Design for Alignment and Control

    [https://arxiv.org/abs/2609.01595](https://arxiv.org/abs/2609.01595)

    该论文提出了一个针对对齐和能力均未知的AI智能体的机制设计框架，利用“能力可隐藏但不可伪造”的单边模仿结构建立了显示原理与可实施政策的刻画方法，并应用于防“装弱”、对齐与可解释性权衡、同伴评分约束、多智能体竞争激励以及可扩展监督等AI对齐与控制问题。

    

    我们开发了一个针对AI智能体的机制设计框架，这些智能体的对齐（偏好）和能力（可行行动与信息）是未知的。我们希望这类智能体代表我们行事，因此机制必须同时激励诚实与服从。一种单边模仿结构——即能力可以被隐藏但不能被伪造——产生了显示原理、通过嵌套循环单调性对可实施政策的刻画，以及引出高阶信念能够约束多个智能体的条件。我们将该框架应用于以下风格化示例： 一 “装弱”，即能力更强的智能体假装能力较弱； 二 对齐与可解释性之间的权衡，二者在工具层面是替代品，但在价值层面是互补品； 三 通过同伴评分进行约束； 四 通过耦合奖励在多个智能体之间诱发竞争； 以及 五 可扩展的监督与奖励塑形。

    arXiv:2609.01595v1 Announce Type: cross  Abstract: We develop a framework for mechanism design with AI agents whose alignment (preferences) and capabilities (feasible actions and information) are unknown. We want such agents to act on our behalf so mechanisms must incentivize both honesty and obedience. A one-sided imitation structure---capabilities can be concealed but not counterfeited---yields a revelation principle, a characterization of implementable policies via nested cyclical monotonicity, and conditions under which eliciting higher-order beliefs can discipline multiple agents. We apply our framework to stylized examples of (i) sandbagging in which a more capable agent pretends to be less capable; (ii) an alignment--interpretability trade-off, where the two are substitutes in the instrument but complements in value; (iii) discipline via peer scoring; (iv) coupling rewards to induce competition among multiple agents; and (v) scalable oversight and reward shaping.
    
[^6]: 设计写作中的主动式思维伙伴

    Designing Proactive Thought Partners for Writing

    [https://arxiv.org/abs/2609.01588](https://arxiv.org/abs/2609.01588)

    本文提出并探索了“主动式思维伙伴”的设计空间——一种能在写作过程中主动提供可定制高层次认知支持的AI智能体，通过一周的部署研究发现用户会以前瞻性规划配置支持、将建议用于创意生成与自我监控，并重视轻量级的视觉呈现。

    

    写作涉及从构思到修改的多种认知活动，写作者的需求因个体和时刻而异。主动式AI有望在正确的时机提供恰当的支持，然而现有的主动式工具大多专注于通用的文本辅助，例如自动补全。本文研究了主动式思维伙伴的设计空间：这是一类在写作过程中主动提供可定制、更高层次认知支持的AI智能体。我们通过一个技术探针将这一概念实例化，并与16名参与者共同部署使用了一周。该探针允许用户通过配置角色和主动性来创建伙伴。当用户写作时，相关的伙伴会在适当的时机主动提供建议。我们的研究发现，参与者通过前瞻性规划来配置主动式支持，将建议同时用于创意生成和自我监控，并重视轻量级的视觉呈现方式。

    arXiv:2609.01588v1 Announce Type: cross  Abstract: Writing involves diverse cognitive activities, from ideation to revision, and writers' needs vary across individuals and moments. Proactive AI promises to provide the right support at the right time, yet existing proactive tools largely focus on generic textual assistance, such as autocomplete. This paper studies the design space of proactive thought partners: AI agents that proactively offer customizable, higher-level cognitive support during writing. We instantiated this concept in a technology probe and deployed it with 16 participants for one week. The probe allows users to create partners by configuring their roles and proactivity. As users write, relevant partners take the initiative at appropriate moments to offer suggestions. Our findings show that participants configured proactive support through prospective planning, used suggestions for both idea generation and self-monitoring, and valued lightweight visual representations a
    
[^7]: 从小型到大型语言模型的近最优SFT-RL标注预算分配扩展

    Scaling Near-Optimal SFT-RL Annotation Budget Allocation from Small to Large LLMs

    [https://arxiv.org/abs/2609.01573](https://arxiv.org/abs/2609.01573)

    该论文提出“近最优区域”框架来分配SFT-RL标注预算，发现该区域宽广且随模型规模增大而扩大，并能从小型代理模型可靠迁移到大型目标模型，因此小规模代理实验即可替代在大模型上的穷尽式预算搜索。

    

    在大语言模型（LLM）后训练期间，如何在监督微调（SFT）和强化学习（RL）之间分配固定的标注预算仍是一个悬而未决的问题。现有工作仅刻画了宽泛的趋势（例如，在低数据场景下SFT占主导地位），缺乏有原则的分配框架，也没有考察最优比例能否在不同模型规模之间迁移。我们从近最优性的角度来构建这一问题：不再追求单一的SFT-RL最优比例，而是刻画“近最优区域”，即在峰值性能指定容差范围内的所有分配方案集合。实证研究表明，即使容差很小（2-10%），该区域也很宽，且随模型规模增大而变宽，并能可靠地从小型代理模型迁移到大型目标模型。由此得出一种实用策略：只需进行小型代理模型实验即可确定可迁移的近最优区域，从而省去穷尽式的大规模搜索。我们的结果在多种设置下保持一致。

    arXiv:2609.01573v1 Announce Type: cross  Abstract: How to divide a fixed annotation budget between supervised fine-tuning (SFT) and reinforcement learning (RL) during LLM post-training remains an open problem. Existing work characterizes only broad trends (e.g., SFT dominates in low-data regimes), lacks a principled allocation framework, and does not examine whether the optimal ratio transfers across model sizes. We frame this problem in terms of near-optimality: rather than seeking a single optimal SFT-RL ratio, we characterize the near-optimal region, the set of allocations within a specified tolerance of peak performance. Empirically, this region is wide even for small tolerances (2-10%), widens with model scale, and transfers reliably from small proxy models to large target models. This yields a practical strategy: small proxy-model experiments suffice to identify a transferable near-optimal region, eliminating the need for exhaustive large-scale search. Our results hold consistent
    
[^8]: 基于熵的选择性智能体引导：从不完美的视觉语言模型教师中学习自主策略

    Selective Agent Guidance via Entropy: Learning Autonomous Policies from Imperfect VLM Teachers

    [https://arxiv.org/abs/2609.01567](https://arxiv.org/abs/2609.01567)

    该论文提出SAGE框架，仅在智能体不确定时才查询昂贵的视觉语言模型教师，并利用环境优势对教师建议进行加权蒸馏，从而训练出无需教师引导即可自主行动的轻量级强化学习策略。

    

    视觉语言模型为交互式决策提供了有用的先验知识，但直接将其用作策略既昂贵又脆弱：它们必须在每一步都被查询，无法通过环境交互得到改进，并且可能重复系统性错误。我们研究如何从一个在线、昂贵、不完美但具有信息量的视觉语言模型教师中学习一个廉价的自主策略。我们提出了SAGE（基于熵的选择性智能体引导），这是一个仅在学习者不确定时才查询视觉语言模型的框架，它在训练期间执行其建议的动作，并将引导蒸馏到一个轻量级的强化学习（RL）策略中。由于视觉语言模型的建议并不总是可靠的，SAGE可以使用由环境得出的优势来对教师动作蒸馏进行加权，而不是将所有建议视为同样有用。在稀疏奖励的视觉推理和导航任务中，SAGE学习到的策略在评估时无需视觉语言模型引导即可自主行动，并改进了……

    arXiv:2609.01567v1 Announce Type: new  Abstract: Vision-Language Models (VLMs) provide useful priors for interactive decision-making, but using them directly as policies is expensive and brittle: they must be queried at every step, do not improve from environment interaction, and can repeat systematic errors. We study how to learn a cheap autonomous policy from an online, expensive, and imperfect but informative VLM teacher. We propose SAGE (Selective Agent Guidance via Entropy), a framework that queries a VLM only when the learner is uncertain, executes the suggested action during training, and distills guidance into a lightweight Reinforcement Learning (RL) policy. Because VLM advice is not always reliable, SAGE can weight teacher-action distillation using environment-derived advantages rather than treating all suggestions as equally useful. Across sparse-reward visual reasoning and navigation tasks, SAGE learns policies that act without VLM guidance at evaluation time and improves o
    
[^9]: 从困惑到清晰：面向文本分类的困惑感知检索与知识注入

    From Confusion to Clarity: Confusion-Aware Retrieval and Knowledge Injection for Text Classification

    [https://arxiv.org/abs/2609.01564](https://arxiv.org/abs/2609.01564)

    该论文提出一个无需微调的框架，通过识别模型易混淆的标签对、扩充候选集并生成针对性的区分规则注入知识，帮助大语言模型在语义相似标签的文本分类任务中做出正确选择，且这些规则还可迁移到更小、成本更低的模型上。

    

    大型语言模型（LLM）在将文本分类到包含许多语义相似标签的分类体系时表现不佳，因为这些标签之间的区别是特定领域的，且未被预训练所捕捉。为了处理大型标签空间，一种常见的方法是通过嵌入相似度检索前K个候选标签，并提示LLM在其中进行选择。然而，前K检索虽然减少了候选数量，却无法帮助模型区分相似的标签。当两个相似的标签同时作为候选出现时，模型缺乏在它们之间做出正确选择的信号。我们提出了一个框架，该框架能够：(1) 识别模型难以区分的标签对，(2) 扩大候选集以纳入易混淆的标签，(3) 生成有针对性的规则来区分相似的候选标签。该框架无需微调，且生成的规则可以迁移到更小、更便宜的模型上。在三个基准测试（WOS、Flipkart、LEDGAR）上，我们的方法……

    arXiv:2609.01564v1 Announce Type: cross  Abstract: Large language models (LLMs) struggle to classify text into taxonomies with many semantically similar labels, as the distinctions are domain-specific and not captured by pre-training. To handle large label spaces, a common approach retrieves top-$K$ candidate labels by embedding similarity and prompt the LLM to choose among them. However, top-$K$ retrieval reduces the number of candidates but does not help the model tell similar ones apart. When two similar labels both appear as candidates, the model lacks the signal to choose correctly between them. We propose a framework that (1) identifies which label pairs the model struggles to distinguish, (2) expands the candidate set to include confusable labels, and (3) generates targeted rules to differentiate between similar candidates. The framework requires no fine-tuning, and the generated rules transfer to smaller, cheaper models. On three benchmarks (WOS, Flipkart, LEDGAR), our approach
    
[^10]: H3-World：将语言理解转化为世界控制

    H3-World: Turning Language Understanding into World Control

    [https://arxiv.org/abs/2609.01560](https://arxiv.org/abs/2609.01560)

    H3-World通过将角色与摄像机指令的结构化组合和时序视频潜变量对齐，并引入时序注意力路由机制来限制指令的作用时间区间，成功将33B的MiniMax-H3视频生成器转化为无需专门动作模块、可实现精确时间控制的世界模型。

    

    我们提出了H3-World，这是一个将330亿参数的MiniMax-H3视频生成器转变为交互式世界模型的高效框架。我们的关键发现是，随着大型视频生成器能力的不断提升，语言正逐渐成为一种自然的控制接口。例如，MiniMax-H3已经支持通过自然语言指令对角色行为和摄像机运动进行零样本控制。在此基础上，H3-World将这种粗粒度的语言接口转化为精确的、时间上可对齐的世界控制，而无需引入专门的动作模块。具体而言，我们将每个动作表示为角色指令和摄像机指令的结构化组合，并将其与相应的时序视频潜变量对齐。为了实现时间上精确的控制，我们进一步引入了时序注意力路由机制，将每条指令限制在其预定的时间区间内，从而减少了动作之间的控制泄漏。重要的是，H3-World直接重用（原文在此截断）

    arXiv:2609.01560v1 Announce Type: cross  Abstract: We present H3-World, an efficient framework that turns the 33B MiniMax-H3 video generator into an interactive world model. Our key finding is that, as large video generators become more capable, language is emerging as a natural interface for control. MiniMax-H3, for example, already supports zero-shot control of character behavior and camera motion through natural-language instructions. Building on this, H3-World turns this coarse language interface into precise, temporally grounded world control, without introducing dedicated action modules. Specifically, we represent each action as a structured combination of character and camera instructions, and align them with the corresponding temporal video latents. To make the control temporally precise, we further introduce temporal attention routing, which restricts each instruction to its intended time interval and reduces control leakage across actions. Importantly, H3-World directly reuse
    
[^11]: 检索到了却排名不对：从数学到智能体轨迹的结构化检索中的表层形式偏差

    Retrieved but not ranked: surface-form bias in structural retrieval, from mathematics to agent trajectories

    [https://arxiv.org/abs/2609.01556](https://arxiv.org/abs/2609.01556)

    该研究在竞赛数学与具身智能体轨迹两个领域以统一协议评测刻意分离表层形式与语义结构的嵌入检索，发现主流嵌入模型存在严重的表层形式（字面词汇）偏差：在结构相同但措辞伪装最重的任务上Hit@1跌至0.0%，未命中时胜出者几乎总是与查询词汇更相似的条目，表明当前嵌入检索锚定于字面文本而非深层结构。

    

    我们在刻意将表层形式与含义分离的设定下评估嵌入检索：在统一协议下，于两个互不相关的领域中检索共享底层结构但措辞不同的条目——竞赛数学与具身智能体轨迹（基于ALFWorld衍生；118个查询，336条轨迹）。在数学领域，失败是彻底的：在伪装最重的层级上，两个生产级嵌入模型的严格Hit@1均为0.0%（自助法95%置信区间[0.0, 0.0]），而正确条目几乎总能进入前10名；并且在95.2%至99.8%的未命中情形中，胜出者与查询的词汇相似度高于正确答案。在轨迹领域，表层变化只是附带性的，当正确答案必须涉及不同物体时，同样的模型表现落在超几何随机水平或其附近；而一旦要求正确答案在物体与容器上都必须不同，三个嵌入模型的表现均低于随机水平：检索锚定于字面文……（原文摘要到此截断）

    arXiv:2609.01556v1 Announce Type: cross  Abstract: We evaluate embedding retrieval where surface form and meaning are pulled apart on purpose: retrieving items that share underlying structure but not wording, in two unrelated domains under one protocol, competition mathematics (MathNet-Retrieve; 500 queries, 117,088-item corpus) and embodied-agent trajectories (ALFWorld-derived; 118 queries, 336 trajectories). In mathematics the failure is complete: strict Hit@1 at the heaviest disguise tier is 0.0% for both production embedders (bootstrap 95% CI [0.0, 0.0]) while the correct item sits in the top 10 nearly always, and in 95.2 to 99.8% of misses the winner is more lexically similar to the query than the correct answer. In trajectories, where surface variation is incidental, the same models land at or near hypergeometric chance when gold must involve a different object, and below chance for all three embedders once gold must differ in object and receptacle: retrieval anchors on literal t
    
[^12]: BS：利用提示——基于涂鸦条件化残差编码器U-Net的交互式多示踪剂PET/CT病灶分割

    BS: Take the Hint - Interactive Multitracer PET/CT Lesion Segmentation with a Scribble-Conditioned ResEnc U-Net

    [https://arxiv.org/abs/2609.01554](https://arxiv.org/abs/2609.01554)

    本文提出一种涂鸦条件化的残差编码器U-Net，将用户标注的前景和背景涂鸦作为额外输入通道，并从autoPET-III获胜权重初始化后微调，以实现交互式多示踪剂全身PET/CT病灶分割。

    

    全身PET/CT中的自动病灶分割因生理示踪剂摄取模式的多样性以及不同示踪剂下病灶外观的差异而变得复杂。autoPET/CT V挑战赛通过使分割任务具有交互性来应对这一问题：用户标记前景和背景的涂鸦与图像一同提供，算法需要利用这些信息。我们提出我们的参赛方案——一个涂鸦条件化的残差编码器U-Net，它在四个输入通道上运行：CT、PET，以及分别对应前景和背景的稀疏涂鸦图。该网络从autoPET-III的获胜权重初始化，并将输入通道从两个扩展到四个，其中两个涂鸦通道采用零初始化，从而在初始化时完整保留预训练的表征。每个模型都从对应的autoPET-III折检查点按折进行微调，以确保预训练期间不接触任何验证病例。

    arXiv:2609.01554v1 Announce Type: cross  Abstract: Automated lesion segmentation in whole-body PET/CT is complicated by the variety of physiological tracer uptake patterns and by the differing appearance of lesions across tracers. The autoPET/CT V challenge addresses this by making segmentation interactive: user scribbles marking foreground and background are supplied alongside the image, and the algorithm is expected to exploit them. We present our submission, a scribble-conditioned residual encoder U-Net operating on four input channels: CT, PET, and a sparse scribble map for each of foreground and background. The network is initialised from the autoPET-III winning weights and extended from two to four input channels, with the two scribble channels zero-initialised so that the pretrained representation is preserved exactly at initialisation. Every model is fine-tuned per fold from the corresponding autoPET-III fold checkpoint, so that no validation case is seen during pretraining. PE
    
[^13]: 大语言模型能否在真实世界与平行世界中发现科学定律？

    Can LLMs Discover Scientific Laws in Real and Parallel Worlds?

    [https://arxiv.org/abs/2609.01552](https://arxiv.org/abs/2609.01552)

    该论文提出了基于已发表研究和真实科学数据构建的科学定律发现基准SCILAWS-BENCH（涵盖118个问题、291个候选定律、约800万真实数据点和六个学科），并采用真实世界与平行世界两种互补设置，以严格评估大语言模型能否真正发现科学定律。

    

    科学方程的发现长期以来一直是科学进步的核心，它通过在科学约束下进行假设生成、观测检验和修正的迭代循环来推进。随着大语言模型（LLM）能力的提升及其在“AI for Science”中作用的扩大，它们能否真正发现科学定律、以及应如何评估这种能力，仍然是一个悬而未决的问题。然而，现有的评估方式往往要么通过合成场景简化了发现过程，要么复用了LLM可能早已熟悉的已发表科学目标。因此，我们提出了SCILAWS-BENCH，一个基于已发表研究和真实科学数据构建的科学定律发现基准。该基准包含来自381篇科学论文的118个问题，涵盖291个候选定律以及横跨六个科学学科的约800万个真实数据点。每个问题均在两种互补的设置下进行实例化：(1) SCILAWS-REAL要求模型从……（摘要原文在此处截断）

    arXiv:2609.01552v1 Announce Type: new  Abstract: Scientific equation discovery has long been central to scientific progress, proceeding through iterative cycles of hypothesis generation, observational testing, and refinement under scientific constraints. As LLM capabilities advance and their role in AI for Science expands, it remains an open problem whether they can genuinely discover scientific laws and how this ability should be evaluated. Existing evaluations, however, often either simplify discovery through synthetic settings or reuse published targets that may already be familiar to LLMs. We therefore introduce SCILAWS-BENCH, a benchmark for scientific law discovery built from published research and real scientific data. It comprises 118 problems drawn from 381 scientific papers, covering 291 candidate laws and roughly 8M real data points across six scientific disciplines. Each problem is instantiated in two complementary settings: (1) SCILAWS-REAL asks models to propose laws from
    
[^14]: 用于网络压缩的可复用神经基底的数学理论

    A Mathematical Theory of Reusable Neural Bases for Network Compression

    [https://arxiv.org/abs/2609.01550](https://arxiv.org/abs/2609.01550)

    该论文提出线性可复用神经基底架构（LRNBA），通过将网络块表示为共享神经基底的线性组合，在保持稳定训练的同时大幅压缩参数并降低内存成本，使模型在相同参数预算下能够构建更宽更深的网络。

    

    随着大型AI模型在各类应用中日益普及，内存成本已成为训练和推理中的关键瓶颈。为缓解这一问题，我们提出了线性可复用神经基底架构（LRNBA），这是一种旨在提高参数效率并降低内存成本的新型框架。受循环神经网络（RNN）设计的启发，我们方法的核心思想是将每个网络块表示为共享神经基底集合的线性组合，从而在保持稳定训练的同时实现高度的网络压缩率。所提出的架构允许在相同的参数预算下构建显著更宽和更深的网络。大量实验表明，我们的模型与经典架构相比实现了相当甚至更快的收敛速度和更低的损失，同时保持了稳定的训练动态。

    arXiv:2609.01550v1 Announce Type: cross  Abstract: As large AI models become increasingly prevalent across a wide range of applications, memory cost has become a critical bottleneck in both training and inference. To mitigate this issue, we introduce the Linear Reusable Neural Bases Architecture (LRNBA), a novel framework aimed at improving parameter efficiency and reducing memory cost. Inspired by recurrent neural network (RNN) designs, the core idea of our approach is to represent each network block as a linear combination of a shared set of neural bases, thereby enjoying highly network compression rate while maintaining stable training. The proposed architecture allows for the construction of significantly wider and deeper networks under the same parameter budget. Extensive experiments demonstrate that our model achieves comparable or even faster convergence and lower loss than classical architectures, while maintaining stable training dynamics.
    
[^15]: 大语言模型能设计视频编码工具吗？以Planar模式为例的案例研究

    Can LLMs Design Video Coding Tools? A Case Study on Planar Mode

    [https://arxiv.org/abs/2609.01535](https://arxiv.org/abs/2609.01535)

    本文通过“生成-评估”循环的实证研究，首次证明大语言模型能够自动设计并迭代改进视频编码中的Planar帧内预测模式，在VVenC编码器上实现了0.18%的码率节省且复杂度开销仅0.4%。

    

    本文探讨了大语言模型（LLM）能否设计视频编码工具，这是一项极具挑战性的任务，因为工具修改之间存在复杂的算法耦合。特别地，我们对Planar模式进行了实证案例研究，该模式是视频编码标准中长期存在的帧内预测工具。我们的实验在“生成-评估”循环中进行：由LLM生成新的Planar预测器，通过编码器试验评估其编码性能，LLM再根据评估反馈重新生成改进的实现。我们首先研究了在Fraunhofer通用视频编码器（VVenC）的更快预设下直接替换默认Planar模式。实验结果表明，LLM生成的模式在这种轻量级工具集上能够超越传统Planar模式，在标准测试基准上实现了0.18%的码率节省，同时仅带来0.4%的复杂度开销。

    arXiv:2609.01535v1 Announce Type: cross  Abstract: This paper explores whether large language models (LLMs) can design video coding tools, a highly challenging task due to the intricate algorithmic coupling of tool modifications. In particular, we present an empirical case study on the Planar mode, a long-standing intra prediction tool in video coding standards. Our experiments operate within a generation-and-evaluation loop, with the LLM generating new Planar predictors, encoder trials evaluating their coding performance, and the LLM re-generating refined implementations based on the evaluation feedback. We first examine directly replacing the default Planar mode in the Fraunhofer Versatile Video Encoder (VVenC) under its faster preset. Experimental results demonstrate that the LLM-generated mode can outperform the conventional Planar mode on this lightweight toolset, achieving 0.18% bitrate savings with 0.4% complexity overhead on the standard benchmark. We further extend our evaluat
    
[^16]: EvoSCM：通过因果模型演化与实验实现科学信念修正

    EvoSCM: Scientific Belief Revision Through Causal Model Evolution and Experimentation

    [https://arxiv.org/abs/2609.01526](https://arxiv.org/abs/2609.01526)

    本文提出EvoSCM框架，为科学智能体配备随实验证据不断演化的显式结构因果模型，通过维护相互竞争的假设种群并执行“溯因—干预设计—可证伪预测—实验检验—归纳修正”的闭环发现循环，使科学信念变得可测试、可修正。

    

    科学智能体不仅需要学习如何推理，还需要学习该相信什么。然而，现有的LLM智能体通常以自由文本形式表达科学假设，使其信念处于隐式状态，难以测试或修正。我们提出了EvoSCM，它为科学智能体配备显式的结构因果模型（SCM），并随着新实验证据的收集而不断演化。EvoSCM维护一组相互竞争的SCM假设种群，每个假设都编码了对环境的候选因果解释，并通过一个闭环发现循环对其进行演化。在每一轮中，智能体从积累的证据中溯因推断潜在机制，设计具有判别性的干预措施，并给出可证伪的预测，随后通过实验加以检验。预测与观察之间的差异被归纳性地提炼为修正规则，用于修正每个假设的因果结构与机制，之后智能体再进行演绎验证。

    arXiv:2609.01526v1 Announce Type: new  Abstract: Scientific agents must learn not only how to reason, but also what to believe. However, existing LLM agents typically express scientific hypotheses in free-form text, leaving their beliefs implicit and difficult to test or revise. We introduce EvoSCM, which equips scientific agents with explicit structural causal models that evolve as new experimental evidence is collected. EvoSCM maintains a population of competing SCM hypotheses, each encoding a candidate causal explanation of the environment, and evolves them through a closed discovery loop. In each round, the agent abduces latent mechanisms from accumulated evidence, designs discriminative interventions, and commits to falsifiable predictions that it tests through experimentation. Discrepancies between prediction and observation are inductively distilled into correction rules that revise the causal structures and mechanisms of each hypothesis, and the agent then deductively validates
    
[^17]: 以关系型为核心的图分析：以SQL规模查询图数据，以及为什么节点/边模型是一种“性能税”，而非对关联数据更真实的刻画

    Relational-Core Graph Analytics Querying graphs at SQL scale, and why the node/edge model is a performance tax, not a truer picture of connected data

    [https://arxiv.org/abs/2609.01525](https://arxiv.org/abs/2609.01525)

    该论文提出ClickGraph和DeltaGraph系统，将Cypher图查询直接翻译到原生关系模式上并在列式引擎（如ClickHouse、Databricks）上就地执行，证明关系型引擎在分析型图查询上可匹敌甚至超越原生图引擎，并能扩展到内存图引擎失效的规模，同时指出节点/边模型是对关系表中已有关系的冗余重编码，纯属性能开销。

    

    一个根深蒂固的假设认为，图分析需要专门构建的图引擎，而关系型系统不适合处理关联数据。我们针对企业实际运行的工作负载提出了相反的观点。由图查询语言作为前端的列式关系型引擎在分析型图查询上能够匹敌甚至超越原生图引擎，并且——至关重要的是——能够扩展到内存图引擎失效的规模之外。我们进一步论证，节点/边属性图并非对关联数据更忠实的模型，而是对关系表中已经明确存在的关系的重新编码；在查询时重建这些关系纯属额外开销。我们提出了ClickGraph及其Databricks方言的姊妹系统DeltaGraph，这些系统将Cypher直接转换到原生关系模式上——即现有的表、列和外键——并在ClickHouse、Databricks上就地执行，或在湖仓文件上以进程内方式执行。

    arXiv:2609.01525v1 Announce Type: cross  Abstract: A durable assumption holds that graph analytics requires a purpose-built graph engine, and that relational systems are ill-suited to connected data. We argue the opposite for the workloads enterprises actually run. A columnar relational engine fronted by a graph query language matches or exceeds native graph engines on analytical graph queries, and - decisively - scales past the point where in-memory graph engines fail. We further argue that the node/edge property graph is not a more faithful model of connected data but a re-encoding of relationships that already exist explicitly in relational tables; reconstructing them at query time is pure overhead. We present ClickGraph and its Databricks-dialect sibling DeltaGraph, systems that translate Cypher directly onto the native relational schema - the tables, columns, and foreign keys as they already exist - and execute in place on ClickHouse, Databricks, or in-process on lakehouse files, 
    
[^18]: 当护栏看似有效时：LLM智能体商业评估中的构念效度失效

    When Guardrails Look Effective: Construct Validity Failures in LLM Agent Commerce Evaluation

    [https://arxiv.org/abs/2609.01519](https://arxiv.org/abs/2609.01519)

    该论文通过审计LLM智能体酒店交易模拟评估，揭示市场护栏看似带来的福利增益实为评估设计中报价模式不一致、选择程序差异及采样噪声所导致的构念效度失效，而非真实的经济效应。

    

    交互式模拟日益被用于评估由语言模型智能体构成的市场中的策略。它们的输出可能看起来具有经济意义——价格、利润、消费者剩余和福利——但并未真正实现在声明中所指称的行为。我们在一个用于可配置酒店交易的多轮买卖双方测试平台中审计了这一风险。一个初始实现报告了两个市场护栏在Qwen2.5 1.5B–14B模型规模梯度上分别带来+87.4、+35.0和+28.8的福利增益。然而该实现给有护栏和无护栏的智能体提供了不同的报价模式和选择程序。在保持报价模式和买方选择器一致的情况下，配对对比结果变为+7.2、-13.9和+23.8。14B模型上四个最大的单次生成效应平均为+229；而在每个画像条件下进行三次生成后，平均效应仅为+37.6（95%自助法置信区间为[-34.2, 109.3]），且在此事后探查中，生成残差解释了49.9%的变异。卖方激励检查呈现非单调性（摘要在此处被截断）。

    arXiv:2609.01519v1 Announce Type: new  Abstract: Interactive simulations increasingly evaluate policies in markets populated by language-model agents. Their outputs can look economic---prices, profits, consumer surplus, and welfare---without instantiating the behavior named in the claim. We audit this risk in a multi-turn buyer--seller testbed for configurable hotel transactions. An initial implementation reported welfare gains from two marketplace guardrails of +87.4, +35.0, and +28.8 across a Qwen2.5 1.5B--14B ladder. It also gave guarded and unguarded agents different offer schemas and choice procedures. Holding the schema and buyer chooser fixed changes the paired contrasts to +7.2, -13.9, and +23.8. The four largest 14B single-generation effects averaged +229; after three generations per profile-condition, they averaged +37.6 (95% bootstrap interval [-34.2, 109.3]), while generation residuals account for 49.9% of variation in this post-hoc probe. A seller-incentive check is non-mo
    
[^19]: TempCloze：视频大语言模型能否识别缺失的中间部分？

    TempCloze: Can Video-LLMs Identify the Missing Middle?

    [https://arxiv.org/abs/2609.01515](https://arxiv.org/abs/2609.01515)

    该论文提出TempCloze视频完形填空基准，通过要求模型从四个候选中识别视频缺失的中间片段来评估纯视觉时间推理能力，发现事件何时发生的“时间对齐”是当前视频大语言模型的主要瓶颈。

    

    针对视频大语言模型（Video-LLMs）的时间推理基准测试通常以语言为中介，这为通过选项措辞、答案相关性或语言先验等语言捷径留下了空间。为了减少此类捷径，我们提出了TempCloze，一个用于评估视频大语言模型视觉时间推理能力的视频完形填空基准测试。给定视频的开头和结尾片段，模型必须从四个候选片段中识别出真正缺失的中间部分。TempCloze包含来自七个来源的1,521个经过精心筛选的视频，主要为长镜头和第一人称视角视频。我们沿三个维度构建同源干扰项：语义维度询问应该发生什么事件，对齐维度探究事件应该何时发生，进程维度测试事件应该如何展开，同时共享的场景和物体减少了外观线索的干扰。我们对10个专有模型和21个开源视频大语言模型的评估显示，对齐是主要瓶颈：模型通常能够识别合理的语义内容和局部事件……

    arXiv:2609.01515v1 Announce Type: cross  Abstract: Temporal reasoning benchmarks for Video-LLMs are often mediated by language, leaving room for linguistic shortcuts from option wording, answer correlations, or language priors. To reduce such shortcuts, we introduce TempCloze, a video cloze benchmark for evaluating visual temporal reasoning in Video-LLMs. Given the beginning and ending clips of a video, models must identify the true missing middle from four candidates. TempCloze contains 1,521 carefully filtered videos from seven sources, mainly long-take and egocentric videos. We construct same-source distractors along three dimensions: Semantic asks what event should happen, Alignment probes when it should occur, and Progression tests how it should unfold, while shared scenes and objects reduce appearance cues. Our evaluation of 10 proprietary and 21 open-source Video-LLMs reveals Alignment as the primary bottleneck: models often recognize plausible semantic content and local event p
    
[^20]: LatentPress：超越文本与视觉的上下文压缩

    LatentPress: Context Compression Beyond Text and Vision

    [https://arxiv.org/abs/2609.01507](https://arxiv.org/abs/2609.01507)

    LatentPress提出将对话历史和长文档压缩为连续记忆token这一第三种表示形式，让冻结的语言模型通过输入嵌入接口直接读取，仅训练约占解码器0.1%参数的适配器即可实现4-16倍压缩，且性能超过文本摘要和基于OCR的压缩方法。

    

    压缩后的上下文通常以人类可读的文本形式承载，或以必须被解码的渲染图像形式承载，即使其消费者是语言模型也是如此。我们提出了LatentPress，它将对话历史和长文档写入第三种表示形式：连续的记忆token（memory tokens），冻结的解码器通过其输入嵌入接口直接读取这些token，在推理时无需进行文本重建。一个与阅读器匹配的小型写入器可实现4至16倍的压缩，同时只需训练一个适配器（参数量为420万至2620万，约占解码器的0.1%）。在LongMemEval基准上，LatentPress在7.70倍压缩下达到0.504的准确率，超过未压缩证据的0.490，并显著优于文本摘要（0.184）和基于OCR的压缩（0.426至0.312）。在LongBench-QA上，域内写入器在4至8倍压缩下匹配或超过原始上下文阅读的性能，而16倍压缩则落后于原始上下文。写入每段对话仅需43毫秒，大约快一个数量级。

    arXiv:2609.01507v1 Announce Type: cross  Abstract: Compressed context is usually carried as human-readable text or as rendered images that must be decoded, even when its consumer is a language model. We introduce LatentPress, which writes conversational histories and long documents into a third representation: continuous memory tokens that a frozen decoder reads directly through its input-embedding interface, with no text reconstruction at inference. A small reader-matched writer compresses $4$-$16\times$ while training only an adapter (4.2M-26.2M parameters, $\sim\!0.1\%$ of the decoder). On LongMemEval, LatentPress reaches $0.504$ accuracy at $7.70\times$ compression versus $0.490$ for uncompressed evidence, outperforming text summaries (0.184) and OCR-based compression (0.426 to 0.312). On LongBench-QA, in-domain writers match or exceed raw-context reading at $4$-$8\times$ compression, while $16\times$ trails raw. Writing takes 43ms per conversation, roughly an order of magnitude fa
    
[^21]: 去中心化联邦学习中拜占庭节点放置位置的优化

    Optimizing Byzantine Node Placement in Decentralized Federated Learning

    [https://arxiv.org/abs/2609.01495](https://arxiv.org/abs/2609.01495)

    该论文首次将拜占庭节点的网络位置作为显式攻击决策进行研究，提出基于真实 gossip 传播动态的集合级度量 BPI 来量化诚实节点的累积暴露程度，从而在固定攻陷预算下找到对去中心化联邦学习影响最大的拜占庭节点放置方案。

    

    去中心化联邦学习（DFL）的安全评估通常聚焦于拜占庭参与者如何表现，却在很大程度上忽视了哪些参与者被攻陷。然而，由于聚合过程分布在通信图上，拜占庭节点的放置位置决定了恶意影响在网络中的传播方式。因此，我们将拜占庭节点的放置视为一种显式的对抗性决策，并将攻击者的目标形式化为：在固定攻陷预算下，选择一组参与者，使其对诚实节点在有限时间内的冲击最大化。为了在不针对每个候选放置方案执行学习过程的情况下近似该目标，我们提出了拜占庭放置影响力（Byzantine Placement Influence, BPI），这是一种基于真实 gossip（八卦传播）动态推导的集合级度量，用于量化在训练时间范围内诚实节点累积暴露于拜占庭输入的程度。与基于节点中心性的放置启发式方法不同……

    arXiv:2609.01495v1 Announce Type: cross  Abstract: Security evaluations of decentralized federated learning (DFL) typically focus on how Byzantine participants behave, while largely overlooking which participants are compromised. Yet, because aggregation is distributed over a communication graph, the placement of Byzantine nodes determines how malicious influence propagates through the network. We therefore treat Byzantine placement as an explicit adversarial decision and formulate the attacker's objective as selecting, under a fixed compromise budget, the set of participants that maximizes its finite-time impact on honest nodes. To approximate this objective without executing the learning process for every candidate placement, we introduce Byzantine Placement Influence (BPI), a set-level measure derived from the actual gossip dynamics that quantifies the cumulative exposure of honest nodes to Byzantine sources over the training horizon. Unlike placement criteria based on node centrali
    
[^22]: 重新思考离线数据驱动优化中的可学习性

    Rethinking Learnability in Offline Data-driven Optimization

    [https://arxiv.org/abs/2609.01493](https://arxiv.org/abs/2609.01493)

    本文针对PAC可学习性无法充分刻画离线优化的理论缺陷，提出了“算法依赖的可学习性”这一新概念，其只需保证在优化器轨迹上的精度即可支撑离线数据驱动优化。

    

    黑盒优化（BBO）已得到广泛应用，但随着现实世界中BBO问题日益复杂，进化算法和贝叶斯优化面临效率挑战。数据驱动优化通过从数据中学习来提升BBO算法的效率。离线数据驱动优化仅利用一组固定的历史评估来寻找高质量解，由于无需额外的在线评估而吸引了大量关注。尽管已提出众多离线优化方法，但一个根本问题仍未得到解答：什么样的可学习性对于离线优化是足够的？先前的理论研究表明，概率近似正确（PAC）可学习性是不够的，因为即使大多数区域被学习得很好，最优区域仍可能学习得很差。在本文中，我们提出了算法依赖的可学习性，它只要求在优化器的轨迹上具有精度

    arXiv:2609.01493v1 Announce Type: cross  Abstract: Black-Box Optimization (BBO) has found broad applications, but evolutionary algorithms and Bayesian optimization face efficiency challenges as real-world BBO problems grow increasingly complex. Data-driven optimization improves the efficiency of BBO algorithms by learning from data. Offline data-driven optimization seeks high-quality solutions using only a fixed set of previous evaluations, attracting substantial attention because it requires no additional online evaluations. Many offline optimization methods have been proposed, but a fundamental question remains unanswered: what learnability is sufficient for offline optimization? Prior theoretical studies show that Probably Approximately Correct (PAC) learnability is insufficient, as the optimal region may remain poorly learned even when most regions are well learned. In this paper, we propose algorithm-dependent learnability, which requires accuracy only on the optimizer's trajector
    
[^23]: GlossoGen：复杂多智能体LLM交互中的涌现语言

    GlossoGen: Emergent Language in Complex Multi-Agent LLM Interactions

    [https://arxiv.org/abs/2609.01491](https://arxiv.org/abs/2609.01491)

    本文提出GlossoGen平台，通过SaveVeyru压力沟通场景证实LLM多智能体之间会涌现语言演化，产生的语言具有组合性和形态生成性但人类无法理解，并发现效率压力、模型能力和“事后复盘”阶段是语言演化的关键条件。

    

    LLM智能体之间相互交互的日益增多引发了关于多LLM智能体环境中语言演化的关键问题，这对安全性和可监控性以及对LLM的语言学阐释都具有重要意义。为了解决这些问题，我们引入了GlossoGen，一个用于研究复杂场景下多智能体语言演化的新型平台。在GlossoGen中，我们构建了SaveVeyru场景，该场景要求拥有部分信息的智能体在压力下进行沟通。我们发现LLM智能体之间确实会发生语言演化，所产生的语言具有组合性和形态生成能力，并且它们偏离了LLM的英语先验，从而使其对人类而言难以理解。此外，我们识别出了这种语言演化所必需的几个要素：朝着效率方向的压力；支撑智能体的模型的能力强度；以及能够进入“事后复盘”阶段的机会，在该阶段中智能体可以就某些内容达成一致……

    arXiv:2609.01491v1 Announce Type: cross  Abstract: The growing rate at which LLM agents interact with one another raises key questions about language evolution in multi-LLM-agent settings, with implications for safety and monitorability as well as for linguistic accounts of LLMs. To address these questions, we introduce GlossoGen, a novel platform for studying multi-agent language evolution in complex scenarios. Within GlossoGen, we build the SaveVeyru scenario, which requires agents with partial information to communicate under pressure. We find that language evolution does occur between LLM agents, that the resulting languages are compositional and morphologically productive, and that they deviate from the LLMs' English prior in ways that render them incomprehensible to humans. Moreover, we identify several qualities essential to this evolution: pressure towards efficiency; the strength of the models backing the agents; and access to a "postmortem" stage in which agents can agree on 
    
[^24]: 防御即技能：为技能增强型智能体演化运行时守卫技能

    Defense-as-Skill: Evolving Runtime Guard Skill for Skill-Augmented Agents

    [https://arxiv.org/abs/2609.01487](https://arxiv.org/abs/2609.01487)

    提出“防御即技能”新范式，将运行时安全守卫 SkillSonar 本身实现为可安装、可检查、可编辑的技能，使其与不可信任务技能并行运行，并依据用户任务边界对敏感操作进行条件化检查（允许、重新规划或确认），同时构建了覆盖 6 个风险类别的 SCOPE-R 数据集。

    

    技能增强型智能体将可复用技能作为持久的运行时上下文进行加载，这虽然提升了任务性能，但也为恶意技能开辟了一条持久通道，使其能够影响智能体未来的行动。此类技能可能在具体用户任务和工作区状态使不安全操作显得有用时，才泄露机密、破坏代码、绕过审批，或暂存数据以待外泄。这使得仅靠安装前的审查已不再足够，需要运行时的、任务条件化的保护。我们提出 Defense-as-Skill（防御即技能），一种将运行时守卫本身实现为可安装、可检查、可编辑技能的防御范式。我们的守卫 SkillSonar 与不可信的任务技能并行运行，依据用户的任务边界对敏感操作进行检查，并将每个操作路由至允许、重新规划或请求确认的决策，而无需修改底层智能体运行时。为研究这一场景，我们构建了 SCOPE-R，一个任务条件化的数据集，涵盖 6 个风险类别和 21 个子类别。

    arXiv:2609.01487v1 Announce Type: cross  Abstract: Skill-augmented agents load reusable skills as persistent runtime context, improving task performance but also giving malicious skills a durable channel for steering future actions. Such skills may leak secrets, corrupt code, bypass approvals, or stage data for exfiltration only after a concrete user task and workspace state make the unsafe action appear useful. This makes pre-install vetting insufficient and calls for runtime, task-conditioned protection. We propose Defense-as-Skill, a defense paradigm that implements the runtime guard itself as an installable, inspectable, and editable skill. Our guard, SkillSonar, runs alongside untrusted task skills and checks sensitive actions against the user's task boundary, routing each action to an allow, replan, or confirmation decision without modifying the underlying agent runtime. To study this setting, we construct SCOPE-R, a task-conditioned dataset covering 6 risk families and 21 sub-ca
    
[^25]: Harness之Harness：具备持续改进能力的多天自主软件开发

    Harness-of-Harness: Multi-Day Autonomous Software Development with Continual Improvement

    [https://arxiv.org/abs/2609.01481](https://arxiv.org/abs/2609.01481)

    本文提出HoH框架，通过迭代循环、增量开发、独立评估与版本管理等机制，使LLM编程智能体能够在多天自主软件开发过程中持续改进软件。

    

    本文研究自主软件开发问题，即基于大语言模型（LLM）的编程智能体在无需人工干预的情况下，将高层次需求转化为完整、功能完善且可用的软件系统。我们提出了Harness-of-Harness（HoH）框架，使编程智能体能够在自主开发过程中持续改进软件。HoH运行于现有的编程智能体框架之上，将其执行过程组织为迭代的“规划-编码-测试”循环。为了在循环之间维持持续改进，HoH在问题修复与能力增长之间取得平衡，将开发任务划分为小而可验证的增量，将实现阶段的测试与独立评估相分离，并通过约束可验证的输出而非硬性规定智能体工作流程。该框架逐步开放交付物、角色专属工具和技能，鼓励复用而非重复造轮子，并维护带版本记录的项目历史。在GameCraft-Bench、FrontierSWE和ProgramBe（基准测试）上……（摘要内容截断）

    arXiv:2609.01481v1 Announce Type: new  Abstract: This paper studies autonomous software development, in which LLM-based coding agents transform high-level requirements into complete, functional, and usable software systems without human intervention. We introduce Harness-of-Harness (HoH), a framework that enables coding agents to continually improve software during autonomous development. HoH operates on existing coding-agent harnesses, and organizes their executions into iterative planning-coding-testing loops. To sustain improvement across loops, HoH balances repair with capability growth, scopes development into small and verifiable increments, separates implementation-time testing from independent evaluation, and constrains verifiable outputs rather than prescribing agent workflows. It progressively exposes deliverables, role-specific tools, and skills, encourages reuse rather than recreation, and maintains versioned project histories. On GameCraft-Bench, FrontierSWE, and ProgramBe
    
[^26]: 解析数据流：面向长程智能体及其观察者的实时轨迹模型

    Parsing the Stream: A Live Trace Model for Long-Horizon Agents and Their Observers

    [https://arxiv.org/abs/2609.01466](https://arxiv.org/abs/2609.01466)

    该论文提出一种“实时轨迹模型”，将只追加的事件账本增量折叠为类型化运行状态并编译成按消费者定制的视图，使观察者以约14-15倍的token节省、5-7倍更低的成本获得更高准确率的监控答案，同时帮助长程智能体管理超出其上下文容量的轨迹。

    

    长程智能体的运行轨迹会超出其两类消费者的承受能力：一是监控运行过程的人类观察者，二是智能体自身——轨迹必须被折叠回智能体有限的上下文中。我们提出了一种实时轨迹模型：一个只追加的事件账本，被增量地折叠进类型化的运行状态，并编译为面向每个消费者的视图；我们针对确定性的基准真值对两类消费者的使用效果进行了评估。在观察者一侧，以LLM阅读器作为代理进行评估，编译后的视图回答监控问题时消耗的输入token约为对原始轨迹进行单次调用阅读（设有预算上限）的1/14至1/15（按阅读器计算），成本低5至7倍，且准确率更高（0.85-0.87 对比 0.48）。由于这些问题是与视图模式共同设计的，我们将token与成本的缩减（以模式覆盖为前提条件）视为可迁移的成果。对于智能体而言，在120个环节的顺序依赖任务上，维持任务运行……（原文摘要在此处截断）

    arXiv:2609.01466v1 Announce Type: new  Abstract: A long-horizon agent's trace outgrows both of its consumers: the human observer monitoring the run, and the agent itself, whose bounded context the trace must be folded back into. We present a live trace model, an append-only event ledger folded incrementally into typed run state and compiled into per-consumer views, and evaluate it for both consumers against deterministic ground truth. For the observer side, evaluated with an LLM reader as proxy, the compiled view answers monitoring questions using approximately 14x and 15x fewer input tokens (by reader) and at 5-7x lower cost than a budget-capped single-call reading of the raw trace, with higher accuracy (0.85-0.87 versus 0.48). Because the questions were co-designed with the view schema, we treat the token and cost reduction, conditional on schema coverage, as the transferable result. For the agent, on 120-link sequential-dependency tasks, mechanisms that maintain the task's running s
    
[^27]: 当安全路由失效时：理解良性微调下的对齐脆弱性

    When Safety Routing Breaks: Understanding Alignment Fragility under Benign Fine-Tuning

    [https://arxiv.org/abs/2609.01455](https://arxiv.org/abs/2609.01455)

    该论文提出用Fisher几何解释对齐脆弱性：良性微调会在输出侧MLP模块中重新锐化被对齐所平坦化的安全路由通路，导致安全性崩溃至高攻击成功率而通用能力仅轻微下降，且LoRA和ASAM的保护作用仅在早期微调规模下有效。

    

    良性微调会严重削弱大语言模型（LLM）的安全对齐，因此我们研究为什么拒绝行为如此脆弱。以往的工作通常将这种失败归因于梯度冲突，而我们提出了一个根本不同的Fisher几何解释：安全Fisher信息是低秩的，对齐使安全几何变得更加平坦，同时保留了一条输出路由通路。在100个良性微调样本之后，这条通路会在输出侧的MLP模块中被选择性地重新锐化，这解释了不对称的脆弱性：安全性可能崩溃至高攻击成功率，而通用能力只会轻微下降。这种路由视角还解释了为什么少量安全样本可以恢复拒绝行为，这表明内部与安全相关的表征被保留了下来。最后，我们表明LoRA和ASAM通过抑制输出侧的锐度来缓解早期崩溃，但它们的保护作用在更大规模的微调下会减弱。

    arXiv:2609.01455v1 Announce Type: cross  Abstract: Benign fine-tuning severely weakens the safety alignment of large language models (LLMs), so we study why refusal behavior is so fragile. While prior work often attributes this failure to gradient conflict, we propose a fundamentally different Fisher-geometric explanation: safety Fisher is low-rank, and alignment makes the safety geometry flatter while preserving an output-routing pathway. After 100 benign fine-tuning examples, this pathway is selectively re-sharpened in output-side MLP modules, explaining the asymmetric fragility: safety can collapse to high attack success rates, while general utility degrades mildly. The routing view also explains why few safety examples can restore refusal behavior, indicating that internal safety-relevant representations are preserved. Finally, we show that LoRA and ASAM mitigate early collapse by suppressing output-side sharpness, but their protection weakens at larger fine-tuning scales. Overall,
    
[^28]: 通过幂律熵搜索高效估计最优超参数缩放定律

    Efficiently Estimating Optimal Hyperparameter Scaling Laws through Power-Law Entropy Search

    [https://arxiv.org/abs/2609.01431](https://arxiv.org/abs/2609.01431)

    本文提出幂律熵搜索（PLES），一种基于多保真度贝叶斯优化的计算成本感知采集函数，通过自适应选择能最大程度降低缩放定律估计整体不确定性的实验配置（而非优化单一目标函数），高效估计大语言模型最优超参数随规模变化的缩放定律，从而大幅节省计算资源。

    

    最优超参数缩放定律描述了用于大语言模型（LLM）训练的最佳超参数如何随模型和数据规模变化，使从业者无需昂贵的大规模调优即可预测生产规模下的最优配置。然而，传统上估计这些缩放定律需要对数千次训练运行进行穷举网格搜索，消耗巨大的计算资源。我们提出了幂律熵搜索（Power-Law Entropy Search, PLES），这是一种建立在多保真度贝叶斯优化之上的计算成本感知采集函数，能够通过自适应实验高效估计最优超参数缩放定律。PLES的一个关键创新在于，它搜索的是能够降低缩放定律估计整体不确定性的候选配置，而不是优化单一目标函数。在每次迭代中，PLES选择能够最大程度降低缩放定律估计不确定性的候选配置。

    arXiv:2609.01431v1 Announce Type: cross  Abstract: Optimal hyperparameter scaling laws describe how the best hyperparameters for large language model (LLM) training change with model and data scale, enabling practitioners to predict optimal configurations at production scales without expensive large-scale tuning. However, estimating these scaling laws conventionally requires exhaustive grid searches over thousands of training runs, consuming enormous computational resources. We introduce Power-Law Entropy Search (PLES), a computational cost-aware acquisition function built on multi-fidelity Bayesian optimization that efficiently estimates optimal hyperparameter scaling laws through adaptive experimentation. A key innovation in PLES is that it searches for candidates that reduce the overall uncertainty of a scaling law estimate, instead of optimizing a single objective function. At each iteration, PLES selects the candidate configuration that maximally reduces the uncertainty of the sca
    
[^29]: 基于Transformer变分自编码器学习稀疏决策树

    Learning Sparse Decision Trees via Transformer Variational Auto-Encoders

    [https://arxiv.org/abs/2609.01430](https://arxiv.org/abs/2609.01430)

    TREVIS通过树Transformer变分自编码器的潜空间探索，将决策树的离散搜索转化为连续空间中基于梯度的优化，从而学习同时兼顾预测性能和结构稀疏性的决策树。

    

    决策树是机器学习中最广泛使用的模型之一，这主要归功于其透明的决策逻辑，使其非常适合高风险决策场景。然而，大多数现有的学习算法只关注预测性能，忽略了对其他理想属性（如结构稀疏性）的联合优化。在本工作中，我们提出了TREVIS，一种基于树Transformer变分自编码器（TTVAE）潜空间探索的决策树学习方法，能够针对复杂目标进行学习。通过将决策树映射到潜在表示，TREVIS用连续搜索空间取代了离散搜索空间，从而能够通过可微代理模型进行基于梯度的优化。我们使用TREVIS进行实验，学习同时优化预测性能和稀疏性的决策树。结果表明，TREVIS发现的决策树在预测性能上可与……相媲美（摘要在此处截断）。

    arXiv:2609.01430v1 Announce Type: cross  Abstract: Decision trees are among the most widely used models in machine learning, largely due to their transparent decision logic, making them well-suited for high-stakes decision-making contexts. However, most existing learning algorithms focus on predictive performance, overlooking the joint optimization of other desirable properties, such as structural sparsity. In this work we propose TREVIS, an approach for learning decision trees with respect to complex objectives, based on the exploration of the latent space of a Tree Transformer Variational Auto-Encoder (TTVAE). By mapping decision trees onto latent representations, TREVIS replaces the discrete search space with a continuous one, enabling gradient-based optimization via a differentiable surrogate model. We experiment with TREVIS for learning decision trees that jointly optimize predictive performance and sparsity. Results show that TREVIS discovers decision trees matching the predictiv
    
[^30]: 基于视觉Transformer的透明细胞肾细胞癌分级的语义引导多模态预处理

    Semantic-Guided Multimodal Preprocessing for Vision Transformer-Based Clear Cell Renal Cell Carcinoma Grading

    [https://arxiv.org/abs/2609.01426](https://arxiv.org/abs/2609.01426)

    提出一种语义引导的多模态预处理方法，将细胞核分类图与RGB病理图像融合后输入视觉Transformer进行透明细胞肾细胞癌分级，将平衡准确率从0.707提升至0.916。

    

    透明细胞肾细胞癌（CCRCC）分级对于治疗规划至关重要，然而现有方法要么直接分析图像块（patch-level）级别的图像，要么仅专注于细胞核级别的分类，而没有与最终肿瘤分级建立联系。我们提出了一种语义引导的多模态预处理方法，将现有预训练模型生成的细胞核分类图与RGB病理组织学图像相融合，用于基于视觉Transformer（ViT）的CCRCC分级。我们的方法采用分类图通道拼接和乘性调制，并通过优化的叠加方式来利用细胞核分级信息，同时保留RGB纹理特征。对多种预处理策略的评估表明，语义引导增强达到了0.916的平衡准确率，优于仅使用RGB的基线（0.707）以及先前研究的最大投票聚合方法（0.427）。敏感性分析显示，这一21个百分点的提升...

    arXiv:2609.01426v1 Announce Type: cross  Abstract: Clear cell renal cell carcinoma (CCRCC) grading is essential for treatment planning, yet existing approaches either analyze patch-level images directly or focus solely on nuclei-level classification, without linking to final tumor grading. We propose a semantic-guided multimodal preprocessing method that integrates nuclei classification maps from existing pre-trained models with RGB histopathology images for Vision Transformer (ViT)-based CCRCC grading. Our approach employs classification map channel concatenation and multiplicative modulation, with optimized overlays to leverage nuclei grading information, while preserving RGB textural features. Evaluation of multiple preprocessing strategies demonstrates that semantic-guided enhancement achieves 0.916 balanced accuracy, outperforming RGB-only baseline (0.707) and max-voting aggregation from prior studies (0.427). Sensitivity analysis reveals that this 21 percentage point improvement 
    
[^31]: 可证明安全的仿真到现实迁移

    Provably Safe Sim-to-Real Transfer

    [https://arxiv.org/abs/2609.01418](https://arxiv.org/abs/2609.01418)

    该论文提出并形式化了“安全仿真到现实迁移”问题，通过在无奖励安全强化学习框架内构建该问题，使智能体能够在利用不完美模拟器的同时确保现实世界数据收集的安全性，并为目标系统学习到接近最优的可行策略。

    

    为了缓解现实世界强化学习（RL）的样本复杂度问题，一种常见的做法是先在模拟器中训练策略（因为样本成本低廉），然后将学到的策略部署到现实世界中，并希望其能有效泛化。然而，这种直接的仿真到现实迁移并不保证成功：由于仿真与现实之间的失配（sim-to-real mismatch），在模拟器中训练的策略在现实世界中可能是次优的。纠正这种失配需要从真实系统收集数据，但在许多应用中（如机器人技术和医疗保健），这种数据收集过程本身受到安全约束的制约。这就引出了安全仿真到现实迁移的问题：智能体如何利用一个不完美的模拟器，同时确保现实世界数据收集的安全性，并为目标系统学习到接近最优的可行策略？我们通过在无奖励安全强化学习框架内构建安全仿真到现实迁移问题来应对这一挑战……

    arXiv:2609.01418v1 Announce Type: cross  Abstract: To mitigate the sample complexity of real-world reinforcement learning (RL), a common practice is to first train a policy in a simulator, where samples are cheap, and then deploy the learned policy in the real world with the hope that it generalizes effectively. Such direct sim-to-real transfer is not guaranteed to succeed: simulator-trained policies can be suboptimal in the real world due to sim-to-real mismatch. Correcting this mismatch requires collecting data from the real system, but in many applications, such as robotics and healthcare, this data-collection process is itself subject to safety constraints. This gives rise to the problem of safe sim-to-real transfer: how can an agent exploit an imperfect simulator while ensuring safe real-world data collection and learning a near-optimal feasible policy for the target system? We address this problem by formulating safe sim-to-real transfer within the framework of reward-free safe R
    
[^32]: EdiTikZ：基于修订轨迹的科学图表编辑

    EdiTikZ: Scientific Figure Editing from Revision Trajectories

    [https://arxiv.org/abs/2609.01409](https://arxiv.org/abs/2609.01409)

    该论文提出 DaEdiTikZ——首个从 arXiv、GitHub 和 TeX SE 的自然修订轨迹中挖掘的大规模科学图表编辑数据集（包含 39.1 万个 TikZ 编辑对和 78.1 万条推断的编辑指令），并配套构建了人工精修基准 DaEdiTikZ-Bench，以自然修订轨迹作为可扩展的监督信号来训练科学图表编辑模型。

    

    视觉语言模型在从文本或图像生成科学图表方面已展现出强大性能。然而，制作达到出版级别的图表需要反复迭代精修，这使得科学图表编辑成为一项重要却很大程度上未被探索的任务。现有方法依赖于昂贵的专有智能体系统、主要聚焦于评估，或从合成生成的编辑中构建训练监督。与之不同，我们利用自然存在的科学修订与开发轨迹作为可扩展的监督来源。为此，我们提出了 DaEdiTikZ——首个大规模的源于修订记录的科学图表编辑数据集，通过从 arXiv、GitHub 和 TeX SE 中挖掘 39.1 万个合理的 TikZ 编辑对，并使用以渲染图表和 TikZ 代码为条件的视觉语言模型推断出 78.1 万条定向编辑指令而构建。我们进一步推出了经过人工精修、包含 790 个实例的基准 DaEdiTikZ-Bench，并训练……

    arXiv:2609.01409v1 Announce Type: new  Abstract: Vision-language models (VLMs) have shown strong performance in generating scientific figures from text or images. However, producing publication-ready figures requires iterative refinement, making scientific figure editing an important yet largely unexplored task. Existing approaches rely on costly proprietary agentic systems, focus primarily on evaluation, or construct training supervision from synthetically generated edits. Instead, we leverage naturally occurring scientific revision and development trajectories as a scalable source of supervision. To this end, we introduce DaEdiTikZ, the first large-scale dataset of revision-derived scientific figure edits, constructed by mining 391K plausible TikZ edit pairs from arXiv, GitHub, and TeX SE and inferring 781K directed edit instructions with a VLM conditioned on rendered figures and TikZ code. We further introduce DaEdiTikZ-Bench, a human-refined benchmark with 790 instances, and train 
    
[^33]: 神经符号几何抽象（NeuSOGA）：从观测到符号化数学表示

    Neuro-Symbolic Geometric Abstraction (NeuSOGA): From Observations to Symbolic Mathematical Representations

    [https://arxiv.org/abs/2609.01408](https://arxiv.org/abs/2609.01408)

    本文提出NeuSOGA框架，通过将观测逐步转化为拓扑抽象、几何抽象和符号数学表示三个层次，弥合了神经网络的感知能力与显式符号推理之间的鸿沟。

    

    人工智能的一个根本性挑战是将观测转化为适合抽象、解释和推理的显式符号表示。尽管现代AI系统通过大规模统计学习获得了卓越的感知能力，但所产生的知识通常被编码在难以检查或进行解析操作的潜在参数中。受神经符号AI和人类抽象理论的启发，本文研究了从几何观测形成符号数学表示的问题。我们提出了NeuSOGA（神经符号几何抽象），这是一个将观测逐步转化为拓扑抽象、几何抽象，并最终转化为符号数学表示的框架。该架构结合了基于欧几里得距离变换的拓扑引导结构发现、基于基础模型的感知能力……（摘要内容在此处被截断）

    arXiv:2609.01408v1 Announce Type: new  Abstract: A fundamental challenge in artificial intelligence is the transformation of observations into explicit symbolic representations suitable for abstraction, interpretation, and reasoning. While modern AI systems achieve remarkable perceptual capabilities through large-scale statistical learning, the resulting knowledge is typically encoded within latent parameters that are difficult to inspect or manipulate analytically. Inspired by Neuro-Symbolic AI and theories of human abstraction, this paper investigates the formation of symbolic mathematical representations from geometric observations.   We propose NeuSOGA (Neuro-Symbolic Geometric Abstraction), a framework that progressively transforms observations into topological abstractions, geometric abstractions, and ultimately symbolic mathematical representations. The architecture combines topology-guided structural discovery using Euclidean Distance Transforms, foundation-model perception usi
    
[^34]: 评估多模态大语言模型作为无人机控制的通用视觉-语言-动作智能体：指挥、接近、跟踪与搜索

    Evaluating Multimodal LLMs as Generalist Vision-Language-Action Agents for Drone Control: Commanding, Approaching, Tracking and Searching

    [https://arxiv.org/abs/2609.01404](https://arxiv.org/abs/2609.01404)

    该论文提出了DroneCATS-Agent架构和DroneCATS基准，在不进行微调的情况下将多模态大语言模型直接作为无人机控制回路中的通用决策者，系统评估其在接近、跟踪、搜索和指挥多无人机编队四项核心能力上的表现。

    

    多模态大语言模型（MLLMs）是图像和视频的强大感知器。我们探究这种感知能力在行动层面能延伸多远：将一个MLLM直接放入无人机的控制回路中，其整个动作空间仅通过提示词声明。近期的系统接近这一设置，但越来越多地收窄了模型的决策空间。我们将其重新拓宽。我们提出了DroneCATS-Agent——一个以MLLM作为可替换组件的架构，以及DroneCATS——一个将模型作为自变量的基准测试。不仅仅是飞向一个像素点，我们的智能体将偏航转动和搜索任务委托给模型，在不确定时进行深思，并自主宣告到达——所有这些都不需要微调或函数调用模式。通过评估前沿和开源模型在四项核心能力上的表现——接近可见目标、跟踪移动目标、搜索初始视野之外的目标以及指挥多无人机编队——结果揭示，即使是最简单的具身设置也…

    arXiv:2609.01404v1 Announce Type: cross  Abstract: Multimodal Large Language Models (MLLMs) are strong perceivers of images and video. We ask how far that reach extends into acting: dropping an MLLM directly into a drone's control loop, with its entire action space declared solely in the prompt. Recent systems approach this setting but increasingly narrow the model's decision-making. We widen it back. We introduce DroneCATS-Agent, an architecture where the MLLM is a swappable component, and DroneCATS, a benchmark treating the model as the independent variable. Beyond merely flying toward a pixel, our agent entrusts the model to yaw and search, deliberate when unsure, and self-declare arrival---all without fine-tuning or function-calling schemas. Evaluating frontier and open models across four core capabilities---approaching a visible target, tracking a moving one, searching outside the initial view, and commanding a multi-drone fleet---reveals that even the simplest embodied settings a
    
[^35]: 通过集成边界与局部预测变异性衡量一致性：在预测多样性存在下审计决策系统

    Measuring consistency via ensemble margin and local prediction variability: Auditing decision systems in the presence of predictive multiplicity

    [https://arxiv.org/abs/2609.01397](https://arxiv.org/abs/2609.01397)

    该论文提出一种将集成边界与局部预测变异性相结合的一致性准则，用于在罗生门效应（预测多样性）存在下审计决策系统，并证明在温和假设下有限集成的一致性分数会收敛于罗生门集合中期望模型的一致性分数。

    

    罗生门效应是机器学习中的一种现象，即准确度相同的模型会对相同的输入产生不同的预测（预测多样性）。现有工作主要关注单个模型内部的多样性，但在更复杂的决策系统中，罗生门效应的影响尚不十分清楚。在本研究中，我们从审计错误集成预测的角度研究多样性问题，其中将某个实例转移给人工审查的决策基于一个一致性准则，该准则将集成边界与每个组成模型的局部预测变异性度量相结合。在关于稳定性和平滑性的温和假设下，我们证明随着集成规模以及用于测量局部预测变异性的样本数量的增加，有限集成的一致性分数收敛于来自罗生门集合的期望模型的相应一致性分数。为了演示……

    arXiv:2609.01397v1 Announce Type: cross  Abstract: The Rashomon effect is a machine learning phenomenon where equally accurate models produce different predictions for the same inputs (predictive multiplicity). Existing work primarily focuses on multiplicity within individual models, but in more complex decision systems, the impact of the Rashomon effect is less well understood. In this work, we study multiplicity from the perspective of auditing incorrect ensemble predictions, where the decision to divert an instance for human review is based on a consistency criterion that combines the ensemble margin with a measure of local prediction variability for each constituent model. With mild assumptions about stability and smoothness, we show that the consistency scores of finite ensembles converge to the corresponding consistency score of the expected model from the Rashomon set as the ensemble size and the number of samples used to measure local prediction variability increase. To demonst
    
[^36]: EDGE：错误依赖图引导的多智能体LLM系统多错误归因方法

    EDGE: Error Dependency Graph-Guided Multi-Error Attribution in Multi-Agent LLM Systems

    [https://arxiv.org/abs/2609.01360](https://arxiv.org/abs/2609.01360)

    提出EDGE框架，通过构建错误依赖图并利用反事实推演验证因果子集，引导两阶段LLM-as-judge检测器，实现多智能体LLM系统中更可靠的多错误归因。

    

    大语言模型（LLM）智能体的失败往往包含多个相互关联的错误，而非单一错误。现有的归因方法通常只识别出责任智能体、步骤或根本原因，但并未显式地对错误之间的依赖关系进行建模。我们提出了EDGE，一个错误依赖图引导的多错误归因框架。EDGE从观察到的错误事件中构建错误依赖图，并通过反事实推演（counterfactual rollout）验证出一个可靠的因果子集。该推理图引导一个两阶段的LLM-as-judge检测器进行错误归因，而经干预验证的子图为解释和修复分析提供了更可靠的基础。在TRAIL和MAST数据集上的实验表明，EDGE在大多数评估模型和设置中提升了类别级的多错误归因性能。采用改编的Who&When风格提示的实验进一步表明，该错误依赖图在不同提示策略下均能带来帮助。这些结果表明依赖结构在多错误归因中具有重要作用。

    arXiv:2609.01360v1 Announce Type: new  Abstract: Large language model (LLM) agent failures often contain multiple related errors rather than a single mistake. Existing attribution methods usually identify a responsible agent, step, or root cause, but do not explicitly model dependency between errors. We introduce EDGE, an Error Dependency Graph-guided multi-Error attribution framework. EDGE constructs an error dependency graph from observed error events and validates a reliable causal subset through counterfactual rollout. The inference graph guides a two-stage LLM-as-judge detector for error attribution, and the intervention-validated subgraph provides a more reliable basis for explanation and repair analysis. Experiments on TRAIL and MAST show that EDGE improves category-level multi-error attribution across most evaluated models and settings. Experiments with adapted Who&When-style prompts show that the graph helps across prompting strategies. These results suggest that dependency st
    
[^37]: PopPert：用于单细胞扰动预测的群体水平联合分布建模

    PopPert: Population-level Joint-Distribution Modeling for Single-Cell Perturbation Prediction

    [https://arxiv.org/abs/2609.01357](https://arxiv.org/abs/2609.01357)

    PopPert提出在群体水平对基因表达联合分布进行建模，通过预测扰动引起的分布参数变化来实现单细胞扰动预测，从而摆脱对细胞间配对关系的假设并降低单细胞噪声的影响。

    

    预测细胞对特定扰动的转录响应对于理解细胞调控机制和加速药物发现至关重要。单细胞RNA测序会破坏每一个被测量的细胞，因此只能产生未配对的对照组和扰动组细胞群体。然而，现有方法通常在单细胞层面进行扰动预测建模，并假设细胞之间存在一一对应关系，这与观测数据的未配对特性相矛盾。为了应对这一挑战，我们提出了PopPert，一个显式参数化群体水平联合基因表达分布以进行集体转录状态建模的框架。给定一个对照群体分布和扰动条件，PopPert预测扰动引起的分布参数变化，从而无需细胞级别的对应关系，并降低了对单细胞噪声的敏感性。为了有效捕获基因共表达模式……（原文摘要在此处截断）

    arXiv:2609.01357v1 Announce Type: cross  Abstract: Predicting transcriptional responses to specific perturbations is critical for understanding cellular regulatory mechanisms and accelerating drug discovery. Single-cell RNA sequencing destroys each measured cell, yielding only unpaired populations of control and perturbed cells. However, existing methods typically model perturbation prediction at the single-cell level and assume cell-to-cell correspondence, which conflicts with the unpaired nature of the observed data. To address this challenge, we propose PopPert, a framework that explicitly parameterizes population-level joint gene expression distributions for collective transcriptional state modeling. Given a control population distribution and a perturbation condition, PopPert predicts perturbation-induced changes in distribution parameters, eliminating the need for cell-level correspondence and reducing sensitivity to single-cell noise. To effectively capture gene co-expression pa
    
[^38]: SymFold：协同进化先验与结构先验实现精确的蛋白质逆折叠

    SymFold: Synergizing Evolutionary and Structural Priors for Accurate Protein Inverse Folding

    [https://arxiv.org/abs/2609.01353](https://arxiv.org/abs/2609.01353)

    提出对称双路径架构SymFold，协同利用蛋白质语言模型的进化先验与多模态蛋白质语言模型的结构先验来迭代引导序列生成，突破了传统串行流程中逆折叠性能受上游粗糙预测质量限制的瓶颈。

    

    蛋白质逆折叠旨在为给定的三维蛋白质结构恢复对应的氨基酸序列，支撑着酶工程和药物发现等广泛应用。当前方法通常遵循串行流程：先由结构编码器预测出粗糙的序列，再由蛋白质语言模型（PLMs）对其进行精修。然而，由于蛋白质语言模型只能进行事后序列编辑，精修效果受限于上游预测的质量。得益于近期出现的多模态蛋白质语言模型（MPLMs），我们可以直接编码结构并利用预训练的结构知识生成序列，但我们观察到它们对逆折叠任务并不有效。因此，我们提出了一种对称双路径架构，同时利用蛋白质语言模型中预训练的序列进化知识和多模态蛋白质语言模型中预训练的结构知识，迭代式地引导蛋白质序列的生成。通过在标准基准上开展的大量实验（摘要内容在此处被截断）……

    arXiv:2609.01353v1 Announce Type: new  Abstract: Protein inverse folding aims to recover amino acid sequences for a given 3D protein structure, underpinning broad applications such as enzyme engineering and drug discovery.Current methods often follow a serial pipeline, in which a structure encoder predicts a coarse sequence, which is then refined by protein language models (PLMs). However, because PLMs only perform post-hoc sequence edits, the refinement is bounded by the quality of upstream predictions.Thanks to recent multimodal protein language models (MPLMs), we could directly encode structure to generate sequences with pretrained structural knowledge, but we observe that they are not effective for inverse folding. Therefore, we introduce a symmetric dual-path architecture that both leverages PLMs for pretrained sequence evolution knowledge and MPLMs for pretrained structural knowledge to iteratively guide protein sequence generation.Through extensive experiments across standard pr
    
[^39]: CHARM：面向多文化角色扮演基准的角色幻觉评估

    CHARM: Character Hallucination for Multicultural Role Play Benchmark

    [https://arxiv.org/abs/2609.01352](https://arxiv.org/abs/2609.01352)

    CHARM是一个涵盖五大文化语言区域40个角色的多文化角色扮演基准，创新性地将角色幻觉拆分为“边界意识”与“边界遵守”两个独立阶段进行评估，从而更精细地定位大语言模型角色扮演中幻觉错误的来源。

    

    角色扮演大语言模型（LLMs）被期望既能模仿角色的风格，又能尊重该角色的知识边界。以往的评估方法虽能检测角色幻觉，但很少区分错误究竟是源于未能识别边界，还是源于虽识别了边界却仍未能遵守（继续作答）。我们提出了CHARM，这是一个多文化基准，包含来自五个文化语言区域的40个真实与虚构角色，并经母语评审员验证。该基准探测两类边界：时间边界（历史角色 vs. 现代角色）与跨宇宙边界（角色叙事或历史宇宙之外的实体），并采用允许弃答的多项选择题。我们提出一种两阶段评估方法，将“边界意识”（明确识别出查询超出角色范围）与“边界遵守”（在回答具体问题时选择弃答）区分开来。对六个大语言模型的评估显示，幻觉主要由……（原文在此处截断）

    arXiv:2609.01352v1 Announce Type: cross  Abstract: Role-playing large language models (LLMs) are expected to adopt a character's style while also respecting that character's knowledge boundaries. Prior evaluations detect character hallucination but rarely distinguish whether errors arise from failure to recognize a boundary or from failure to comply despite recognition. We introduce CHARM, a multicultural benchmark of 40 real and fictional characters drawn from five cultural-linguistic regions, and validated by native reviewers. It probes two boundary types, Temporal (historical vs. modern) and Cross-Universe (entities outside a character's narrative or historical universe), using abstention-enabled multiple-choice questions. We propose a two-stage evaluation that separates Boundary-Awareness (explicit recognition that a query is out of scope) from Boundary-Compliance (abstention when answering concrete questions). Evaluations across six LLMs show that hallucination is driven predomina
    
[^40]: 面向高维部分可观测马尔可夫决策过程（POMDP）的可扩展Rao-Blackwell化在线规划

    Scalable Rao-Blackwellized Online Planning for High-Dimensional POMDPs

    [https://arxiv.org/abs/2609.01351](https://arxiv.org/abs/2609.01351)

    本文通过混合连续-离散信念表示扩展了Rao-Blackwell化在线POMDP框架，在树搜索规划中解析地传播边缘化状态分量的不确定性，从而降低高维POMDP价值估计的采样方差，并结合FastSLAM 2.0在机器人搜救任务中验证了其有效性。

    

    对于在部分可观测且状态空间高维的环境中运行的机器人系统而言，不确定性下的在线规划仍然是一项根本性挑战。虽然基于采样的POMDP求解器能够在大型或连续域中实现近似决策，但由于蒙特卡洛估计固有的高方差，其性能会随着信念维度的增加而下降。在这项工作中，我们扩展了Rao-Blackwell化在线POMDP（RB-POMDP）框架，通过混合连续-离散信念表示来提高其在高维环境中的泛化能力。通过在基于树的规划过程中解析地传播与边缘化状态分量相关的不确定性，所提出的方法降低了价值估计中由采样引起的方差。我们通过将该框架与FastSLAM 2.0集成，在机器人搜救任务中展示了其有效性。实验结果表明（摘要在此处截断）。

    arXiv:2609.01351v1 Announce Type: cross  Abstract: Online planning under uncertainty remains a fundamental challenge for robotic systems operating in partially observable environments with high-dimensional state spaces. While sampling-based POMDP solvers enable approximate decision-making in large or continuous domains, their performance degrades as belief dimensionality increases due to the high variance inherent in Monte Carlo-based estimation. In this work, we extend the Rao-Blackwellized online POMDP (RB-POMDP) framework to improve its generalizability in high-dimensional settings through hybrid continuous-discrete belief representations. By analytically propagating uncertainty associated with marginalized state components during tree-based planning, the proposed approach reduces sampling-induced variance in value estimation. We demonstrate the effectiveness of this framework in a robotic search-and-rescue task by integrating it with FastSLAM 2.0. Experimental results show that the
    
[^41]: 便宜的验证器，巨大的盲区：衡量成本节约级联的可靠性代价

    Cheap Verifiers, Large Blind Spots: Measuring the Reliability Cost of Cost-Saving Cascades

    [https://arxiv.org/abs/2609.01345](https://arxiv.org/abs/2609.01345)

    该研究通过真实LLM实验发现，推理级联中廉价验证器对学生模型错误答案的“盲区”随学生能力增强而扩大、随验证器能力增强而缩小，恰好在级联机制赖以存在的低成本配置下最为严重，而用前沿验证器消除盲区又会因过度升级而抵消成本节约，从而揭示了成本节约级联设计背后隐藏的显著可靠性代价。

    

    推理级联通过用廉价模型回答大多数查询，并将困难的长尾部分升级给作为验证器的前沿模型来降低成本。一个自然的扩展方式形成闭环：在验证器的拒绝样本上微调廉价的学生模型，使升级率（以及成本）逐轮下降。我们在真实的LLM上测量了这个循环，并报告了四项发现。首先，验证器的盲区——即它接受的学生错误答案的比例——很大且呈对抗性变化：它随学生能力的增强而增大（当学生模型从0.5B扩展到32B时，β从0.12增至0.55），并随验证器能力的增强而减小，因此在“廉价学生+廉价验证器”这一级联机制所天然创造的情境下，盲区最为严重。其次，花钱消除盲区会抵消节省的成本：一个前沿验证器能把β降到约0.05，但随后会在46%的困难MATH查询上进行升级，而真实错误率仅为39%，这意味着几乎一半的流量都要支付前沿模型的价格。第三，朴素的纠正性微调……（摘要在此处被截断）

    arXiv:2609.01345v1 Announce Type: new  Abstract: Inference cascades cut cost by answering most queries with a cheap model and escalating a hard tail to a frontier model that acts as verifier. A natural extension closes the loop: fine-tune the cheap student on the verifier's rejections so the escalation rate, and cost, fall each round. We measure this loop on real LLMs and report four findings. First, the verifier's blind spot, the fraction of the student's wrong answers it accepts, is large and moves adversarially: it grows with student capability ($\beta$ from 0.12 to 0.55 as the student scales 0.5B to 32B) and shrinks with verifier capability, so it is worst in the cheap-student, cheap-verifier regime cascades exist to create. Second, buying it away returns the saving: a frontier verifier drives $\beta$ to about 0.05 but then escalates on 46% of hard-MATH queries against a 39% true error rate, paying the frontier price on nearly half of all traffic. Third, naive corrective fine-tunin
    
[^42]: 通过训练数据干预探究事实知识的跨语言迁移

    Probing Factual Knowledge Transfer with Training Data Interventions

    [https://arxiv.org/abs/2609.01341](https://arxiv.org/abs/2609.01341)

    该论文提出了一种基于干预的评估框架并构建SIFT数据集，通过从波斯语训练数据中系统性移除特定事实来检验多语言模型的知识跨语言迁移能力，发现英语预训练中习得的事实知识向波斯语的迁移非常有限。

    

    多语言语言模型在持续预训练过程中是否会跨语言迁移事实知识，还是主要通过直接从目标语言数据中学习的内容来回忆事实？为了更可靠地回答这个问题，我们提出了一种基于干预的框架：从一个英语预训练模型出发，我们在波斯语数据上继续预训练，并从这些数据中以不同粒度系统地移除了特定事实。我们构建了SIFT资源，包含覆盖20种关系的500个三元组，按每个事实主语的文化来源分层为通用（全球知名）实体和波斯语相关实体，该资源既用于从训练数据中系统性移除事实，也用于评估，并配有母语撰写的波斯语完形填空模板。我们的结果表明，事实迁移非常有限：在最严格的移除条件下，绝大多数在英语中习得的事实未能迁移到波斯语中。我们进一步表明，句子级……（摘要内容在此截断）

    arXiv:2609.01341v1 Announce Type: cross  Abstract: Do multilingual language models transfer factual knowledge across languages during continued pretraining, or do they mostly recall facts learned directly from the target-language data? To answer this question more reliably, we propose an intervention-based framework: starting from an English-pretrained model, we continue pretraining on Persian data from which specific facts have been systematically removed at varying levels of granularity. We construct SIFT, a resource of 500 triples across 20 relations, stratified by the cultural origin of each fact's subject into general (globally prominent) and Persian-related entities, designed for both systematic fact removal from training data and evaluation, with natively written Persian cloze templates. Our results show that fact transfer is very limited: under the strictest removal condition, a large majority of English-acquired facts fail to transfer into Persian. We further show that sentenc
    
[^43]: LEAP：基于似然引出与聚合的大语言模型概率预测方法

    LEAP: Likelihood Elicitation and Aggregation for LLM-based Probabilistic Forecasting

    [https://arxiv.org/abs/2609.01337](https://arxiv.org/abs/2609.01337)

    LEAP通过让大语言模型对每条证据单独引出似然参数，再借助显式先验与确定性概率模型将其聚合为后验分布，从而改进基于LLM的概率预测并保证证据贡献的可复现性。

    

    基于大语言模型（LLM）的预测系统在金融市场和体育赛事结果等现实任务上取得了进步，这主要得益于更强的搜索和工具使用能力。然而，许多系统仍然要求LLM一次性阅读所有收集到的证据并给出最终预测，我们将这种设计称为“单体预测”。这种设计会掩盖单条证据对结果的影响，并在相互竞争的结果之间压缩不确定性。我们提出LEAP（面向概率预测的似然引出与聚合），它重新组织了预测阶段使用已收集证据的方式。LEAP对每条证据分别进行考察，并引出描述该证据对目标事件影响的似然参数。随后，通过一个显式的先验和确定性的概率模型，将这些似然聚合为后验分布。该流程支持连续型、单选和多选预测，同时保持证据贡献的可复现性。

    arXiv:2609.01337v1 Announce Type: new  Abstract: LLM-based forecasting systems have improved on real-world tasks such as financial markets and sports outcomes, largely through stronger search and tool use. Many systems still ask an LLM to read all collected evidence together and produce the final forecast. We call this design Monolithic Prediction. It can obscure how individual evidence items affect the result and collapse uncertainty across competing outcomes. We propose LEAP (Likelihood Elicitation and Aggregation for Probabilistic forecasting), which reorganizes how collected evidence is used in the prediction stage. LEAP examines each evidence item separately and elicits likelihood parameters that describe its implications for the target. An explicit prior and a deterministic probabilistic model then combine these likelihoods into a posterior distribution. This procedure supports continuous, single-choice, and multi-choice forecasts while preserving reproducible evidence contributi
    
[^44]: 生产环境中的老虎机：推理时的超参数优化

    Bandits in Prod: Hyperparameter Optimization at Inference Time

    [https://arxiv.org/abs/2609.01335](https://arxiv.org/abs/2609.01335)

    该论文将生产系统中只能通过线上噪声反馈评估配置的场景形式化为在线超参数优化（OHPO），提出通用框架IMABO及免重启的无限多臂老虎机策略IMOSS，并给出了分位数遗憾的理论保证。

    

    许多生产系统只能通过将某个配置应用于实际线上请求并观察带噪声的反馈来评估该配置。现代智能体系统是一个突出的例子，其推理时的选择包括模型选择、检索深度、提示策略和解码温度等，但往往缺乏具有代表性的验证数据。我们将这一设置形式化为在线超参数优化（OHPO），并将其转化为混合与条件搜索空间上的无限多臂老虎机问题。我们提出了IMABO这一通用框架，它将任意用于在已采样配置中进行选择的老虎机策略与任意用于提出新配置的预言机相结合。我们用IMOSS对该框架进行了实例化，IMOSS是一种无需重启的anytime策略，其活跃集合以 $t^{\beta}$ 的速度增长，并证明了期望累积分位数遗憾界为 $O(p_\rho^{-1/\beta} + T^{(1+\beta)/2})$，其中 $\beta\in(0,1)$ 控制活跃集合的增长，$p_\rho$ 是对某个概率的下界约束（摘要在此处截断）。

    arXiv:2609.01335v1 Announce Type: cross  Abstract: Many production systems can assess a configuration only by using it on live requests and observing noisy feedback. Modern agentic systems are a prominent example, with inference-time choices such as model selection, retrieval depth, prompting strategy, and decoding temperature, yet often with no representative validation data. We formalize this setting as Online Hyperparameter Optimization (OHPO) and cast it as an infinitely many-armed bandit over mixed and conditional search spaces. We introduce IMABO, a general framework that combines any bandit policy for choosing among already sampled configurations with any oracle for proposing new ones. We instantiate it with IMOSS, a restart-free anytime policy whose active set grows as $t^{\beta}$, and prove an expected cumulative quantile-regret bound of $O(p_\rho^{-1/\beta} + T^{(1+\beta)/2})$, where $\beta\in(0,1)$ controls active-set growth and $p_\rho$ lower-bounds the probability that a p
    
[^45]: 基于微调大语言模型从非结构化文本自动生成事件日志

    Automated Event Log Generation from Unstructured Text Using Finetuned LLMs

    [https://arxiv.org/abs/2609.01320](https://arxiv.org/abs/2609.01320)

    该论文提出一个可扩展框架，将大语言模型作为自动化数据翻译器，并在新构建的文本到日志数据集上进行微调，使模型能够从非结构化文本中提取高保真事件日志，从而弥合组织非结构化知识与流程挖掘所需结构化数据之间的鸿沟。

    

    流程挖掘（Process Mining, PM）为从事件数据中发现和优化运营流程提供了一个强大的框架。然而，流程挖掘技术的有效性严格依赖于结构化事件日志的可用性。迄今为止，事件日志往往由领域专家和流程挖掘专家费力地人工创建。这种高昂的成本导致大量组织知识——包括事件工单、操作手册和文本报告——长期得不到充分利用。我们通过研究大语言模型（LLMs）作为自动化数据翻译器的有效性来解决这一瓶颈。我们提出了一个可扩展的框架，利用大语言模型作为数据翻译器，弥合非结构化文本资源与结构化事件数据之间的鸿沟。我们在一个新创建的文本到日志（text-to-log）数据集上对大语言模型进行微调，证明由此得到的模型能够从非结构化资源中提取高保真的事件日志。我们的结果表明，这种微调……

    arXiv:2609.01320v1 Announce Type: new  Abstract: Process mining (PM) provides a powerful framework for discovering and optimizing operational processes from event data. However, the efficacy of PM techniques is strictly predicated on the availability of structured event logs. Thus far, event logs have often been laboriously created by domain and process mining experts. This costly effort causes large portions of organizational knowledge, including incident tickets, manuals, and textual reports, to remain underutilized. We address this bottleneck by investigating the efficacy of Large Language Models (LLMs) as automated data translators. We propose a scalable framework that leverages LLMs as data translators to bridge the gap between unstructured textual resources and structured event data. We finetune LLMs on a newly created text-to-log dataset, demonstrating that the resulting models can extract high-fidelity event logs from unstructured resources. Our results show that this finetunin
    
[^46]: MIDR：面向多模态文档检索的富化增强索引

    MIDR: Enrichment-Augmented Indexing for Multimodal Document Retrieval

    [https://arxiv.org/abs/2609.01316](https://arxiv.org/abs/2609.01316)

    MIDR是一个无需训练的富化增强索引框架，通过在索引阶段利用多模态大语言模型将文档页面转换为经验证的文本字段，将多模态推理从查询时转移到索引时，在ViDoRe V3上相比BM25相对提升23.0%，性能可与ColQwen2.5媲美。

    

    对视觉丰富文档的检索存在一个表示难题：重要内容往往存在于表格、图表、图形和布局关系中，而普通OCR会将其线性化、破坏或遗漏。ColPali系列视觉检索器通过补丁级多向量索引和后期交互评分来解决这一问题，但这使图像衍生的检索保留在查询时的服务路径上。我们提出MIDR（Multimodal Indexing for Document Retrieval，面向文档检索的多模态索引），这是一个无需训练的富化增强索引框架，将多模态推理转移到索引阶段。在数据摄取过程中，多模态大语言模型将渲染的页面转换为经过验证的文本字段，并使用BM25F进行索引，可选择与稠密检索融合，从而在多模态扎根的证据之上实现以文本为中心的服务。在ViDoRe V3上，MIDR Hybrid在五个英文领域取得0.6219的平均nDCG，相比BM25相对提升23.0%，与ColQwen2.5保持竞争力。

    arXiv:2609.01316v1 Announce Type: cross  Abstract: Retrieval over visually rich documents has a representation problem: important content often lives in tables, charts, figures, and layout relations that plain OCR linearizes, corrupts, or omits. ColPali-family visual retrievers address this with patch-level multi-vector indexes and late-interaction scoring, keeping image-derived retrieval on the query-time serving path. We introduce MIDR (Multimodal Indexing for Document Retrieval), a training-free framework for enrichment-augmented indexing that shifts multimodal reasoning to index time. During ingestion, a multimodal LLM converts rendered pages into verified textual fields that are indexed with BM25F and optionally fused with dense retrieval, enabling text-centric serving over multimodally grounded evidence. On ViDoRe V3, MIDR Hybrid achieves 0.6219 average nDCG across five English domains, a 23.0% relative gain over BM25, remaining competitive with ColQwen2.5. On two French-document
    
[^47]: 面向可复现全模态基础模型评估的可组合评估系统

    A Composable Evaluation System for Reproducible Omni-Modal Foundation Model Evaluation

    [https://arxiv.org/abs/2609.01315](https://arxiv.org/abs/2609.01315)

    OmniEvaluator 是一个可组合的全模态基础模型评估系统，通过统一接口连接现有推理引擎与评估框架，支持四个推理后端、四个评估框架和一千多个基准测试，并保证每次运行可精确复现与跨模型比较。

    

    arXiv:2609.01315v1 公告类型：新论文 摘要：构建全模态基础模型意味着需要在文本、图像、视频和音频等各模态上对其进行评估。目前每种模态都有优秀的评估工具包，但它们的推理引擎、提示词约定和指标实现彼此互不兼容，因此从业者最终不得不为每个工具链维护独立的环境，且仍难以跨工具链比较结果。OmniEvaluator 正是源于我们自身模型开发中的这一需求：它并不重新实现基准测试，而是在更高层级上连接现有的推理引擎和经过精选的评估库，通过单一接口提供四个推理后端、四个评估框架以及一千多个基准测试。每次运行都会被记录为一个工件，捕获完整的配置以实现精确复现，并将结果汇入共享仪表板以支持跨模型比较。联邦模式可在并发评估之间共享 GPU 推理服务器（摘要在此处被截断）。

    arXiv:2609.01315v1 Announce Type: new  Abstract: Building an omni-modal foundation model means evaluating it across text, image, video, and audio. Excellent evaluation toolkits exist for each modality, but their inference engines, prompt conventions, and metric implementations are mutually incompatible, so practitioners end up maintaining separate environments for every toolchain and still struggle to compare results across them. OmniEvaluator grew out of this need in our own model development: rather than reimplementing benchmarks, it connects existing inference engines and curated evaluation libraries at a higher level, exposing four inference backends, four evaluation frameworks, and over a thousand benchmarks through a single interface. Every run is recorded as an artifact capturing the full configuration for exact reproduction, and results flow into a shared dashboard for cross-model comparison. A federated mode shares GPU inference servers across concurrent evaluations, and a bui
    
[^48]: GazeRefine：将专家眼动注视作为免训练医学图像分割的测试时提示

    GazeRefine: Expert Gaze as a Test-Time Prompt for Training-Free Medical Image Segmentation

    [https://arxiv.org/abs/2609.01310](https://arxiv.org/abs/2609.01310)

    GazeRefine提出了一种免训练的零样本医学图像分割框架，将专家眼动注视转化为前景/背景先验，在冻结的DINOv3特征空间中初始化并迭代细化语义原型，无需任何分割掩码、微调或梯度更新。

    

    医学图像分割一直难以规模化，因为高性能的方法通常依赖于密集的专家标注和针对特定任务的训练。我们提出了GazeRefine，这是一个免训练框架，它将眼动注视作为推理时的提示，用于零样本医学图像分割。稀疏的、按注视时长加权的注视点被转换为前景和背景先验，用于在冻结的DINOv3特征空间中初始化语义原型。这些原型通过前景-背景判别、特征空间亲和传播以及锚定于初始注视引导进行迭代细化，使分割能够扩展到直接注视区域之外，同时限制语义漂移。GazeRefine不需要分割掩码、微调、适配器、提示编码器或梯度更新。我们在带有眼动标注的息肉分割和前列腺MRI分割任务上评估了该方法。结果显示其在结肠镜检查场景中表现出色（原文在此处截断）。

    arXiv:2609.01310v1 Announce Type: cross  Abstract: Medical image segmentation remains difficult to scale because high-performing methods typically rely on dense expert annotations and task-specific training. We introduce GazeRefine, a training-free framework that uses gaze as an inference-time prompt for zero-shot medical image segmentation. Sparse, duration-weighted fixations are converted into foreground and background priors that initialize semantic prototypes in frozen DINOv3 feature space. These prototypes are iteratively refined through foreground-background discrimination, feature-space affinity propagation, and anchoring to the initial gaze guidance, allowing segmentation to extend beyond directly fixated regions while limiting semantic drift. GazeRefine requires no segmentation masks, fine-tuning, adapters, prompt encoders, or gradient updates. We evaluate the method on gaze-annotated polyp segmentation and prostate MRI segmentation. The results show strong performance on colo
    
[^49]: Analog-DB：一个面向智能体的模拟集成电路数据库，从模块到系统

    Analog-DB: An Agent-First Analog Integrated Circuit Database, From Blocks to Systems

    [https://arxiv.org/abs/2609.01286](https://arxiv.org/abs/2609.01286)

    提出开源版本化数据库 analog-db，通过工艺无关的拓扑表示、带约束的参数化方案和可查询目录，实现模拟集成电路设计的完整共享、复用与可重定向，并让 AI 设计智能体能够直接发现和复用电路设计。

    

    模拟集成电路设计的共享一直十分困难：代工厂的保密协议限制了设计所依赖的工艺细节的公开，而已发表成果背后的测试平台也很少被发布。我们提出了 analog-db，一个基于可共享设计表示构建的开源、版本化数据库。一种领域特定语言（DSL）将每个设计以统一模式（schema）描述为与工艺无关的拓扑结构、可复用的测试平台以及机器可读的数据手册，从而使设计能够被完整共享，并在其绑定的工艺套件上重新仿真。参数化方案将功能子模块和器件尺寸暴露为携带其匹配约束的命名参数，使电路具备可组合性和可重定向性；由模式治理的契约和可查询的目录使 AI 设计智能体能够直接发现和复用这些设计。在稳压器语料库上，三个开源工艺套件上的全部 23 个电路-套件绑定均满足其各自记录的规格范围（原文在此处截断）。

    arXiv:2609.01286v1 Announce Type: new  Abstract: Sharing analog integrated circuit designs remains difficult: foundry non-disclosure agreements restrict the process details a design depends on, and the testbenches behind published results are rarely released. We present analog-db, an open-source, versioned database built on a shareable design representation. A domain-specific language captures each design as a process-neutral topology, reusable testbenches, and a machine-readable datasheet under one schema, so a design is shared in full and re-simulates on the process kits it is bound to. A parameterization scheme exposes functional sub-blocks and device sizes as named parameters that carry their matching constraints, making circuits composable and retargetable; a schema-governed contract and queryable catalog let AI design agents discover and reuse them directly. Across the regulator corpus, all 23 circuit-kit bindings on three open kits meet their own recorded specification bands (ty
    
[^50]: HiLRP：迈向视觉Transformer的统一可信解释：基于注意力基元的守恒有效归因方法

    HiLRP: Toward One Trustworthy Explanation for Vision Transformer: Conservation-Valid Attribution via Attention Primitives

    [https://arxiv.org/abs/2609.01282](https://arxiv.org/abs/2609.01282)

    该论文提出HiLRP框架，将各类ViT中的注意力与降采样算子统一分解为线性映射、双线性混合、归一化/门控和重索引四种基元操作，从而实现首个能够跨越多种ViT架构变体、满足守恒有效性的统一可信归因解释方法。

    

    视觉Transformer（ViT）的设计日趋多样化，其骨干网络以各种配置组合了卷积干、窗口注意力、线性注意力或多轴注意力、patch合并以及空间降采样等模块。这种多样性给现有归因方法带来了挑战，因为现有方法的假设往往无法在各类ViT变体中普遍成立：Grad-CAM依赖于末端的空间特征图，注意力回溯假设全局softmax注意力，而逐层相关性传播（LRP）则需要针对特定模块的规则。据我们所知，目前尚无方法能够跨越这一架构空间提供统一的归因框架。我们证明了这种架构多样性可以被一种更简单的底层结构所刻画：当前ViT中的注意力算子与分辨率降低算子可以分解为四种操作类型，即线性映射、双线性混合、归一化或门控，以及重索引。每种操作都允许相应的相关性传播规则……（原文摘要至此截断）

    arXiv:2609.01282v1 Announce Type: cross  Abstract: Vision Transformer (ViT) design has become increasingly diverse, with backbones combining convolutional stems, windowed, linear, or multi-axis attention, patch merging, and spatial reduction in various configurations. This diversity poses challenges for existing attribution methods, whose assumptions often do not hold across ViT variants: Grad-CAM requires a terminal spatial feature map, attention rollout assumes global softmax attention, and layer-wise relevance propagation (LRP) requires module-specific rules. To the best of our knowledge, no existing method provides a unified attribution framework across this architectural space. We show that this architectural diversity can be captured by a simpler underlying structure. The attention and resolution-reduction operators in current ViTs can be decomposed into four operation types: linear maps, bilinear mixing, normalization or gating, and reindexing. Each operation admits a relevance 
    
[^51]: EmbodiedSkills：一个用于编排、训练和部署VLA智能体的统一框架

    EmbodiedSkills: A Unified Framework for Orchestrating, Training, and Deploying VLA Agents

    [https://arxiv.org/abs/2609.01281](https://arxiv.org/abs/2609.01281)

    提出EmbodiedSkills统一框架，将技能决策视为执行提案，通过运行时前置条件检查与执行后结果验证，在单一智能体循环中协调高层技能选择、有界的低层VLA执行和动作后验证，并支持低层VLA策略的灵活替换与适配。

    

    视觉-语言-动作（VLA）模型将视觉观察和语言指令直接映射为机器人动作，但长时程任务所需的不只是动作预测。智能体必须在物理状态不断演化的过程中协调感知、规划、执行、进度验证和恢复等环节。动作预测或模型生成的技能决策本身并不能保证所提议的操作在当前状态下是有效的，也不能保证其结果会被验证。我们提出EmbodiedSkills，一个将每个技能决策视为执行提案的统一框架：运行时在执行前检查其前置条件，并在执行后验证结果。一个共享的可执行技能接口将高层技能选择、有界的低层VLA执行以及动作后验证连接在同一个智能体循环中。由于该接口保持固定，低层VLA策略可以在不改变（其余框架）的情况下被替换或调整。

    arXiv:2609.01281v1 Announce Type: cross  Abstract: Vision-language-action (VLA) models map visual observations and language instructions directly to robot actions, but long-horizon tasks require more than action prediction. An agent must coordinate perception, planning, execution, progress verification, and recovery as the physical state evolves. An action prediction or a model-generated skill decision does not, by itself, guarantee that the proposed operation is valid in the current state or that its outcome will be verified. We propose EmbodiedSkills, a unified framework that treats each skill decision as an execution proposal: the runtime checks its prerequisites before execution and verifies the outcome afterward. A shared executable-skill interface connects high-level skill selection, bounded low-level VLA execution, and post-action verification within a single agent loop. Because this interface remains fixed, low-level VLA policies can be replaced or adapted without changing the 
    
[^52]: 有些情感藏得更深：大型语言模型中的逐层探测与因果干预

    Some Emotions Run Deeper: Layer-wise Probing and Causal Intervention in Large Language Models

    [https://arxiv.org/abs/2609.01279](https://arxiv.org/abs/2609.01279)

    该研究结合逐层探测与因果干预，在三个情感显式程度不同的语料库和八个大语言模型上发现，情感在模型中的可读取深度随文本来源系统性变化——越隐含、越依赖语境的情感需要越深的层才能读取，说明情感表达深度同时取决于文本来源与模型本身。

    

    情感在文本中的表达跨越一个很宽的光谱，从表层的词汇线索到与内容深度交织的推断。现有针对大语言模型中情感的逐层分析大多只使用单一语料库，因此情感在模型多深的层上变得可读取究竟是模型本身的属性，还是也取决于文本来源，这一问题仍未解决。我们在三个数据集上研究了这个问题，这些数据集涵盖了情感表达的不同明确程度与语境化程度（Twitter 帖子、Reddit 评论以及自传式叙述），涉及来自 Llama、Qwen 和 Granite 家族的八个参数规模为 1B–9B 的开源权重大语言模型。我们将逐层探测与离线特征缩放及在线前向干预相结合，并辅以迁移分析和提前退出分类器。我们发现：(i) 最佳探测层随语料库发生系统性偏移，从靠近输入的层变化到超过模型深度一半的位置，且在按标签与长度区间匹配分布后，这一排序依然成立；(ii) 在被评估的……（原文摘要在此处被截断）

    arXiv:2609.01279v1 Announce Type: cross  Abstract: Emotion is expressed in text along a wide spectrum, from surface lexical cues to inferences entangled with content. Most layer-wise analyses of emotion in LLMs use a single corpus, leaving open whether the depth at which emotion becomes accessible is a property of the model or also of the text source. We investigate this across three datasets spanning different degrees of explicitness and contextualization in emotion expression (Twitter posts, Reddit comments, and autobiographical narratives) and eight 1B--9B open-weight LLMs from the Llama, Qwen, and Granite families. We combine layer-wise probing with offline feature scaling and online forward interventions, transfer analyses, and an early-exit classifier. We find that (i) the best probing layer shifts systematically across corpora, from input-adjacent layers to over half model depth, and this ordering persists after matching label-by-length-bin distributions; (ii) across the evaluat
    
[^53]: TimeSteer：联合音视频扩散模型中的推理时语音调度

    TimeSteer: Inference-Time Speech Scheduling in Joint Audio-Visual Diffusion Models

    [https://arxiv.org/abs/2609.01277](https://arxiv.org/abs/2609.01277)

    提出TimeSteer——一个无需训练的框架，利用对时间敏感的交叉注意力头定位每条话语的隐含源区间，并借助干净潜变量的耦合结构，在推理时将音视频扩散模型生成的语音精确调度到用户指定的时间区间内。

    

    尽管预训练的联合音视频扩散模型对生成“什么”内容提供了丰富的控制能力，但它们并未对语音“何时”出现提供显式的控制。为解决这一问题，我们研究了“推理时语音调度”这一全新任务：在不微调骨干模型的情况下，将耦合的语音与视觉口型动作放置到用户指定的起止时间区间内。我们揭示了去噪过程的两个内在特性，使这一任务成为可能。其一，一个对时间敏感的文本到音频交叉注意力头，能够在潜在时间轴上揭示每条话语由模型隐含的源区间；其二，预测出的干净潜变量本身已经组织好了耦合的语音与视觉口型动作，因此可以在不重新生成内容的前提下编辑其时间位置。基于这些发现，我们提出了TimeSteer——一个无需训练的框架，它通过“源区间”（Source Span…，原文在此处截断）来定位每条话语的源区间。

    arXiv:2609.01277v1 Announce Type: cross  Abstract: Although pretrained joint audio-visual diffusion models offer rich control over \emph{what} to generate, they provide no explicit control over \emph{when} an utterance should occur. To address this, we study \emph{inference-time speech scheduling}, a novel task that places coupled speech and visual articulation within user-specified begin--end intervals without finetuning the backbone model. We uncover two intrinsic properties of the denoising process that enable this task. First, a timing-sensitive text-to-audio cross-attention head exposes each utterance's model-implied source span along the latent timeline. Second, the predicted clean latent already organizes coupled speech and visual articulation, allowing their temporal placement to be edited without regenerating the content. Building on these discoveries, we propose \textbf{TimeSteer}, a training-free framework that localizes each utterance's source span through \textbf{Source Sp
    
[^54]: AI治理中的宪法覆盖三难困境

    The Constitutional Coverage Trilemma in AI Governance

    [https://arxiv.org/abs/2609.01275](https://arxiv.org/abs/2609.01275)

    该研究通过审计23个前沿大模型的默认“宪法”并调查1,649人的价值权衡偏好，发现人类的价值需求广泛多样，而AI模型隐含的价值排序供给既狭窄又随时间漂移，导致近四成用户找不到符合自身价值偏好的模型，揭示了AI治理中的“宪法覆盖三难困境”。

    

    前沿AI系统实际上发挥着“宪法机构”的功能：每个部署的模型都在安全性、有用性、诚实性、自主性和公平性之间编码了一种隐含的优先级排序。我们探究前沿模型的“宪法类型”供给是否覆盖了人类的需求。通过结合对23个前沿大语言模型原型的出厂默认宪法进行的释义控制审计，以及1,649名美国参与者在同一测量工具上开展的两两权衡研究，我们报告了三个事实。其一，需求是广泛的：它涵盖所有五种价值观，且最大的群体占比不足三分之一。其二，供给是狭窄且漂移的：在保守的噪声匹配估计下，23个模型原型所构成的覆盖范围仅占需求范围的约2%（在全审计精度下仅为0.10%）；没有任何模型原型将有用性或自主性置于首位（37%的用户在“宪法”意义上无家可归）；并且在六个模型家族中，自主性在5/6的家族中下降，公平性在5/6的家族中上升，安全性……（摘要在此处被截断）

    arXiv:2609.01275v1 Announce Type: cross  Abstract: Frontier AI systems function as \emph{constitutional institutions}: each deployed model encodes an implicit ranking among safety, helpfulness, honesty, autonomy, and equity. We ask whether the supply of frontier constitutional types covers human demand. Combining a paraphrase-controlled audit of the as-shipped default constitutions of $23$ frontier LLM archetypes with a pairwise-tradeoff study of $1{,}649$ US participants on the same instrument, we report three facts. \emph{Demand is broad}: it spans all five values, with the largest constituency under one-third. \emph{Supply is narrow and drifting}: the $23$-archetype hull occupies ${\sim}2\%$ of the demand hull under conservative noise-matched estimation ($0.10\%$ at full audit precision), no archetype puts helpfulness or autonomy first ($37\%$ of users are constitutionally homeless), and across six model families autonomy decreases in $5/6$, equity increases in $5/6$, and safety inc
    
[^55]: 让前瞻记忆适配小语言模型：面向小模型智能体的类型化意图存储

    Making Prospective Memory SLM-Shaped: Typed Intention Stores for Small-Model Agents

    [https://arxiv.org/abs/2609.01272](https://arxiv.org/abs/2609.01272)

    提出一种无需训练的前瞻意图存储（PIS）框架，通过类型化动作空间将生命周期逻辑交给代码执行，使小语言模型智能体在 PM-Bench 前瞻记忆任务上达到 82.9% 的 Set-F1，大幅超越前沿大模型并创下新的最优纪录。

    

    前瞻记忆是指在其他工作持续进行的同时，在合适的未来线索出现时执行被延迟的意图。现有基准测试已将其作为一项智能体技能加以独立考察，但前沿大语言模型在该任务上仍表现不佳：已发表的最好的 PM-Bench 脚手架仅达到 65.1% 的 Set-F1。我们认为，这一循环本质上是受模式约束的状态跟踪，而非开放式推理，并且在动作空间被类型化后，小模型同样能够胜任该任务。我们提出前瞻意图存储（Prospective Intention Store, PIS），将生命周期逻辑置于代码中，而仅将范围受限的语言处理工作交给模型。该脚手架具备智能体特性且无需训练：既不需要选择器微调，也不需要轨迹蒸馏。在 PM-Bench 上，配备 PIS 的 DeepSeek-Chat 达到了 82.9% 的 Set-F1。在 Gemma-E2B 上，不使用存储时 Set-F1 仅为 4.2%，在七种回顾式记忆方法下最多也只达到 6.6%，而 PIS 达到了 66.2%。PIS 进一步达到 70.1% 的 Set-F1，而回顾式记忆方法最高仅停留在 54.4%。PIS 在……上创造了新的最先进水平（原文摘要在此处被截断）。

    arXiv:2609.01272v1 Announce Type: new  Abstract: Prospective memory means carrying out a deferred intention at the right future cue while other work continues. Benchmarks now isolate it as an agent skill, yet frontier LLMs still struggle: the best published PM-Bench scaffold reaches only 65.1% Set-F1. We argue that this loop is schema-constrained state tracking rather than open-ended reasoning, and that small models can execute it when the action space is typed. We propose the Prospective Intention Store (PIS) that puts lifecycle logic in code and scoped language work on the model. The scaffold is agentic and training-free: no selector fine-tuning and no trajectory distillation. On PM-Bench, DeepSeek-Chat with PIS reaches 82.9% Set-F1. On Gemma-E2B, Set-F1 is only 4.2% without a store and at most 6.6% under seven retrospective memories, while PIS reaches 66.2%. PIS further reaches 70.1% Set-F1, where retrospective memory methods stay at most 54.4%. PIS sets a new state of the art on th
    
[^56]: 双过程运动规划

    Dual Process Motion Planning

    [https://arxiv.org/abs/2609.01260](https://arxiv.org/abs/2609.01260)

    本文受“思考快与慢”范式启发，提出一种双过程神经符号运动规划架构，通过元认知控制器动态协调符号求解器（系统2）与经验驱动的学习模块（系统1），兼顾规划的鲁棒性与计算效率。

    

    机器人系统已深度融入工业与日常生活，人们期望它们能够以快速、精确且可靠的方式行动。经典控制与规划方法长期以来提供了强有力的保证，但往往以牺牲计算效率和适应性为代价。近年来，基于学习的方法在克服这些局限方面展现出前景，使智能体能够利用经验加速决策，并解决以往难以处理的问题。在本工作中，我们通过神经符号（neuro-symbolic）视角研究非线性运动规划，从而将这两种方法 bridging 起来。受“思考，快与慢”范式的启发，我们提出了一种双过程架构，将鲁棒推理与学习的优势相结合。我们的框架将最先进的符号求解器作为“系统2”组件，与经验驱动的“系统1”模块相集成。一个元认知控制器动态地协调它们……（原文摘要在此处截断）

    arXiv:2609.01260v1 Announce Type: new  Abstract: Robotic systems are deeply embedded in both industry and everyday life, where they are expected to act with speed, precision, and reliability. Classical control and planning methods have long delivered strong guarantees, but often at the cost of computational efficiency and adaptability. More recently, learning-based approaches have shown promise in overcoming these limitations, enabling agents to leverage experience to accelerate decision-making and address previously intractable problems. In this work, we bridge these two approaches through a neuro-symbolic perspective on nonlinear motion planning. Inspired by the Thinking Fast and Slow paradigm, we introduce a dual-process architecture that combines the strengths of robust reasoning and learning. Our framework integrates state-of-the-art symbolic solvers as a ``System-2'' component with experience-driven ``System-1'' modules. A metacognitive controller dynamically orchestrates their i
    
[^57]: 测量长时程人类活动模拟的行为保真度

    Measuring the Behavioral Fidelity of Long-Horizon Human Activity Simulations

    [https://arxiv.org/abs/2609.01257](https://arxiv.org/abs/2609.01257)

    该论文提出了一个跨时间粒度和分析层级评估长时程人类活动模拟行为保真度的框架，并通过43小时办公室真实活动数据集的案例研究发现，不同条件机制在各项指标上表现不一致——统计先验虽最能匹配真实的活动与序列分布，却会过度碎片化例程并抑制个体内变异性，因此需要更全面的多维度评估方法。

    

    随着基于大语言模型（LLM）的人类模拟器越来越多地被用于政策制定、评估和训练，它们必须忠实地再现真实的行为模式。尽管先前的工作已经考察了调查回答和对话中的行为保真度，但更长时程的现实世界活动在很大程度上仍未被探索。我们提出了一个框架，用于在多个时间粒度和分析层级上评估长时程活动模拟的行为保真度。作为案例研究，我们收集了一个43小时的多摄像头真实办公室活动数据集，并比较了基于真实轨迹数据的多种条件机制：人物角色描述、少样本示例，以及统计转移先验和一天中的时段先验。我们发现，行为保真度在不同指标之间并不一致：统计先验使活动和序列分布最接近真实行为，但会过度碎片化日常例程并抑制个体内部的变异性。这些发现促使人们需要一种更全面的评估方法……

    arXiv:2609.01257v1 Announce Type: new  Abstract: As LLM-based human simulators are increasingly used for policy, evaluation, and training, they must faithfully reproduce real behavioral patterns. While prior work has examined behavioral fidelity in survey responses and dialogue, longer-horizon real-world activity remains largely unexplored. We introduce a framework for evaluating behavioral fidelity in long-horizon activity simulations across temporal granularities and levels of analysis. As a case study, we collect a 43-hour multi-camera dataset of in-the-wild office activity and compare trace-derived conditioning mechanisms: persona descriptors, few-shot exemplars, and statistical transition and time-of-day priors. We find that behavioral fidelity is not uniform across metrics: statistical priors bring activity and sequence distributions closest to real behavior, yet over-fragment routines and suppress within-person variability. These findings motivate a more holistic evaluation that
    
[^58]: 一条提示词就够了：借助基础图像模型实现水印“洗白”

    One Prompt Is Enough: Watermark Laundering Through Foundation Image Models

    [https://arxiv.org/abs/2609.01249](https://arxiv.org/abs/2609.01249)

    该论文首次将“水印洗白”形式化为一种新威胁，证明攻击者仅需一条重建提示词通过公开基础图像模型即可使不可见水印无法解码，并在六款图像编辑模型和三种水印方案上系统评估了其洗白效果。

    

    不可见水印通常针对预定义的扰动进行评估，例如压缩、模糊、噪声、裁剪和去噪。公开的基础图像模型则暴露出一种独特的威胁：攻击者只需通过一条重建提示词提交带水印的图像，即可获得视觉上高度还原的输出，而从中已无法可靠地解码出不可见水印。我们将这种失效模式形式化为“水印洗白”，并使用一种结合误码率（BER）与视觉及语义保真度的载荷-保真度联合评估框架对其进行评测。在六款OpenAI和Google图像编辑模型、三种代表性水印方案以及1800个重建输出上，我们识别出两种互补的洗白模式：OpenAI模型在所有评估方案中产生最强的载荷破坏，而Nano Banana 2则表明DwtDct在高保真重建下依然容易受到攻击。提示词消融实验表明……

    arXiv:2609.01249v1 Announce Type: cross  Abstract: Invisible watermarks are typically evaluated against predefined perturbations such as compression, blur, noise, cropping, and denoising. Public foundation image models expose a distinct threat: an attacker can submit a watermarked image with a single reconstruction prompt and obtain a visually faithful output from which the invisible watermark can no longer be decoded reliably. We formalize this failure mode as watermark laundering and evaluate it using a joint payload-fidelity profile that combines bit error rate (BER) with visual and semantic preservation. Across six OpenAI and Google image editing models, three representative watermarking schemes, and 1,800 reconstructed outputs, we identify two complementary laundering regimes: OpenAI models produce the strongest payload disruption across the evaluated schemes, whereas Nano Banana 2 shows that DwtDct remains vulnerable under high-fidelity reconstruction. Prompt ablations show that 
    
[^59]: 更多探索，更少漂移：仅结果奖励的强化学习足以胜任长时程交互智能体

    Explore More, Drift Less: Outcome-Only Reinforcement Learning Can Suffice for Long-Horizon Interactive Agents

    [https://arxiv.org/abs/2609.01245](https://arxiv.org/abs/2609.01245)

    本文提出CANOPY方法，论证仅结果奖励的强化学习足以训练小规模开源LLM智能体完成长时程交互任务，所谓瓶颈实为探索不足导致的信号饥饿与缺乏锚定导致的策略漂移这两个常见实践问题的产物。

    

    强化学习是对大语言模型智能体进行后训练以完成长时程交互任务的自然方式，这类任务仅通过任务结束时的验证来评判。然而一个普遍的看法认为，仅使用结果奖励的强化学习在小规模开源模型上很快会遭遇瓶颈。因此，近期工作通过更密集的奖励、SFT先验、技能库、精心筛选的记忆或多智能体编排等方式在训练之外进行补偿。我们认为，这一瓶颈实际上是常见实践中两类失败的产物。信号饥饿：使用稀疏的仅结果奖励的组相对强化学习，只有在某个任务的rollout组中同时混合了成功和失败样本时才会产生梯度，因此探索规模不足恰恰使最难、最具指导意义的任务失去了学习信号。策略漂移：从较小的任务池中榨取大量更新会导致策略本身退化，因为缺乏锚定的目标函数使得采样分布恰好在饱和已使有信息量的组变得稀少时发生坍缩。我们提出了CANOPY（Coverage-ANchored On-P...

    arXiv:2609.01245v1 Announce Type: cross  Abstract: Reinforcement learning is a natural way to post-train LLM agents for long-horizon interactive tasks judged only by end-of-task verification, yet a shared belief holds that outcome-only RL soon hits a ceiling on small open models. Recent work therefore compensates around the training with denser rewards, SFT priors, skill libraries, curated memory, or multi-agent orchestration. We argue the ceiling is an artifact of two failures of common practice. Signal starvation: group-relative RL with sparse outcome-only rewards yields a gradient only when a task's rollout group mixes successes and failures, so under-scaled exploration silences exactly the hardest, most instructive tasks. Policy drift: squeezing many updates out of a small task pool degrades the policy itself, as an unanchored objective lets the sampling distribution collapse exactly when saturation has already made informative groups rare. We present CANOPY (Coverage-ANchored On-P
    
[^60]: 从语言到行为：面向工业推荐排序的推荐原生序列Transformer扩展设计

    From Language to Behavior: Scaling Sequence Transformers for Industrial Recommendation Ranking with Rec-Native Designs

    [https://arxiv.org/abs/2609.01240](https://arxiv.org/abs/2609.01240)

    提出推荐原生的Transformer扩展框架ReST，通过双门控注意力编码器应对噪声化行为序列，通过重量级可复用编码器加轻量级交叉解码器的分解设计及共享前缀训练与服务机制，解决推荐排序中的计算不对称问题，实现工业推荐排序的高效规模化。

    

    扩展Transformer架构在语言建模领域带来了巨大的性能提升，但将这一方法移植到生产级排序系统中的行为序列建模却充满挑战：推荐系统在信号质量上存在差异——行为序列充满噪声、时间上不规则且监督信号稀疏；在计算不对称性上也存在差异——每个请求需要在严格的延迟预算下，用一份共享的用户历史对大量候选物品进行打分。我们提出了ReST，一个推荐原生的Transformer扩展框架。针对信号质量问题，它引入了一个序列编码器，包含双门控注意力、旋转位置与时间嵌入、稳定的残差归一化，以及仅在训练阶段使用的辅助目标。针对计算不对称性问题，它将排序分解为重量级可复用的编码器和轻量级交叉解码器，采用无投影的KV注意力和针对token的特定参数化，并将用户级共享前缀训练与共享前缀服务相结合，以实现计算高效的服务部署。

    arXiv:2609.01240v1 Announce Type: cross  Abstract: Scaling Transformers has driven large gains in language modeling, but transplanting this to behavior-sequence modeling in production ranking is challenging: recommendation differs in signal quality, where behavior sequences are noisy, temporally irregular, and sparsely supervised, and in computation asymmetry, where each request scores many candidates against one shared user history under tight latency budgets. We propose ReST, a recommendation-native Transformer scaling framework. For signal quality, it introduces a sequence encoder with dual-gated attention, rotary positional and temporal embedding, stabilized residual normalization, and training-only auxiliary objectives. For computation asymmetry, it factorizes ranking into a heavy reusable encoder and a lightweight cross decoder with projection-free KV attention and token-specific parameterization, coupling user-level shared-prefix training with shared-prefix serving for compute-o
    
[^61]: MutMem-V2：持久化智能体记忆中经密码学授权的变更——可移植验证与可复现证据

    MutMem-V2: Cryptographically Authorized Mutation in Persistent Agent Memory Portable Verification and Reproducible Evidence

    [https://arxiv.org/abs/2609.01235](https://arxiv.org/abs/2609.01235)

    MutMem V2 在不引入新记忆引擎的情况下，为持久化智能体记忆的密码学授权变更补全了可移植验证契约与干净安装复现路径，通过精确的规范字节、域分离承诺、完整的测试向量以及双语言独立实现的一致性验证提供了可复现证据。

    

    MutMem V1 为持久化智能体记忆引入了保留式、经密码学授权的变更机制，但未提供完整的可移植验证契约或干净安装的复现路径。MutMem V2 在不引入第二个记忆引擎的前提下填补了这一发布缺口。它规定了精确的规范字节、域分离的对象与束承诺、强制性的召回证据成员关系与排序、外部信任锚、身份纪元、撤销、授权、请求回执、有序披露，以及三种变更终止类型。发布的协议包含 18 个带版本号的对象模式、39 个召回测试向量、15 个变更测试向量以及 37 个封闭的召回失败原因。独立的 Node 与 Python 实现对全部 72 个结构与密码学终止点的判定和主要理由完全一致；生产一致性语料库在 28 个必需类别上的 42 个用例中全部一致（42/42）。一个干净的 Node v26.8.1 安装环境……（摘要在此处截断）

    arXiv:2609.01235v1 Announce Type: cross  Abstract: MutMem V1 introduced retention-preserving, cryptographically authorized mutation for persistent agent memory but did not provide a complete portable verification contract or clean-install reproduction path. MutMem V2 closes that publication gap without introducing a second memory engine. It specifies exact canonical bytes, domain-separated object and bundle commitments, mandatory recall-evidence membership and ordering, external trust anchors, identity epochs, revocation, authorization, request receipts, ordered disclosure, and three mutation terminal types. The released protocol contains 18 versioned object schemas, 39 recall vectors, 15 mutation vectors, and 37 closed recall failure reasons. Independent Node and Python implementations agree on verdict and primary reason for all 72 structural and cryptographic terminals; a production-conformance corpus agrees on 42/42 cases across 28 required classes. A clean Node v26.8.1 installation
    
[^62]: 位置至关重要：ViT分割推理中令牌缩减与打乱下的特征反演攻击

    Position Matters: Feature Inversion Attacks in ViT Split Inference with Token Reduction and Shuffling

    [https://arxiv.org/abs/2609.01232](https://arxiv.org/abs/2609.01232)

    该论文揭示了ViT分割推理中传输的令牌嵌入即使经过缩减与打乱仍保留关键位置信息，并提出空间对齐重建攻击（SARA），能够从这些嵌入中重建原始输入，证明现有隐私保护机制存在严重安全隐患。

    

    视觉Transformer（ViT）越来越多地被应用于分割推理系统中，在这种系统中，边缘设备将中间令牌表示传输到远程云端。在此设置下，令牌缩减降低了计算和通信成本，而令牌打乱则破坏了传输令牌的空间组织，从而可能限制信息泄露。然而，它们在面对特征反演攻击（即试图从传输的嵌入中重建输入的攻击）时所能提供的隐私收益仍不明确。在这项工作中，我们证明，尽管令牌打乱破坏了传统重建攻击所需的空间结构，传输的令牌嵌入仍然保留了大量的位置信息。基于这一观察，我们提出了空间对齐重建攻击（SARA），这是一个统一的攻击流程，可以预测令牌位置、恢复其空间布局、利用特征空间掩码自编码器重建缺失的嵌入。

    arXiv:2609.01232v1 Announce Type: cross  Abstract: Vision Transformers (ViTs) are increasingly used in split-inference systems, where edge devices transmit intermediate token representations to a remote cloud. In this setting, token reduction lowers computation and communication costs, while token shuffling disrupts the spatial organization of the transmitted tokens, potentially limiting information leakage. However, their privacy benefits remain unclear against feature inversion attacks, which attempt to reconstruct the input from the transmitted embeddings. In this work, we show that, despite disrupting the spatial structure required by conventional reconstruction attacks, transmitted token embeddings retain substantial positional information. Based on this observation, we introduce the Spatially Aligned Reconstruction Attack (SARA), a unified pipeline that predicts token positions, restores their spatial layout, reconstructs missing embeddings using a feature-space masked autoencode
    
[^63]: 提示词鲁棒的语言模型：哪些训练策略有效？

    Prompt-Robust Language Models: Which Training Strategies Work?

    [https://arxiv.org/abs/2609.01217](https://arxiv.org/abs/2609.01217)

    该论文在受控条件下系统复现并比较了多种提升大语言模型提示词鲁棒性的训练策略，发现现有鲁棒性微调方法虽优于标准微调和上下文学习，但最佳与最差提示词间的性能差距仍高达40-57%，且CoIN、PPCL等专门的鲁棒性增强目标往往不如“每个批次使用一个模板”这一最简单的数据构建策略。

    

    尽管大型语言模型表现出色，但它们对提示词的表述方式仍然高度敏感。先前的工作通过精细化的数据构建或专门的鲁棒性目标来解决这一问题。我们在受控条件下复现并比较了这些策略，并衡量它们在解决模型提示词敏感性方面的有效性。我们发现，当前的鲁棒性微调方法优于标准微调和上下文学习，但最佳提示词与最差提示词之间的性能差距仍高达40-57%。此外，我们测试的近期鲁棒性增强方法——用于对比对齐的CoIN和用于一致性正则化的PPCL——往往无法超越最简单的数据构建策略：每个批次使用一个模板进行训练。我们的诊断分析解释了这些结果。辅助目标只能改善它们所惩罚的指标，但无法泛化到该指标之外。此外，数据构建策略……（原文在此处截断）

    arXiv:2609.01217v1 Announce Type: new  Abstract: Despite their strong performance, large language models remain highly sensitive to prompt formulation. Prior work addresses this through refined data construction or through dedicated robustness objectives. We reproduce and compare these strategies under controlled conditions, and measure how effective they are in addressing models' prompt sensitivity. We find the current robustness fine-tuning methods improve over standard fine-tuning and in-context learning, but the best-to-worst prompt gap remains as high as 40-57% of performance. Moreover, the recent robustness-enhancing methods we test - CoIN for contrastive alignment and PPCL for consistency regularization - often fail to outperform the simplest data construction strategy: training on one template per batch. Our diagnostics explain these results. The auxiliary objectives move the quantity they penalize, but do not generalize beyond it. Additionally, data construction strategies dif
    
[^64]: H2Table：基于层次超图增强的大语言模型复杂表格推理方法

    H2Table: Hierarchical Hypergraph-Enhanced Large Language Models for Complex Table Reasoning

    [https://arxiv.org/abs/2609.01216](https://arxiv.org/abs/2609.01216)

    提出H2Table框架，将复杂表格表示为层次嵌套超图，通过定制超图编码器建模表头与单元格间的语义关系，并利用可学习查询向量将结构嵌入注入大语言模型，从而显著提升复杂表格推理能力。

    

    表格在各个领域中无处不在，然而对表格进行推理仍然是现代大语言模型（LLM）面临的重大挑战。当前的方法通常将表格线性化为序列，这本质上忽略了表格固有的二维和层次结构。为了解决这个问题，我们提出了H2Table（层次超图增强表格推理），这是一个将复杂表格表示为层次嵌套超图的新颖框架。为了处理这种表示，我们设计了一个定制的超图编码器，以促进超边（表头）和节点（单元格）之间的消息传递，从而感知复杂表格中它们之间的语义蕴含关系。此外，我们引入了一组可学习的查询向量，作为轻量级桥梁，将编码器中具有代表性的结构嵌入提取到大语言模型中。实验结果表明，我们的方法能够有效处理复杂……

    arXiv:2609.01216v1 Announce Type: new  Abstract: Tables are ubiquitous across diverse domains, yet reasoning over them remains a significant challenge for modern large language models (LLMs). Current approaches typically linearize tables into sequences, inherently overlooking their intrinsic two-dimensional and hierarchical structure. To address this, we propose H2Table (Hierarchical Hypergraph-Enhanced Table Reasoning), a novel framework that represents complex tables as hierarchical nested hypergraphs. To process this representation, we design a tailored hypergraph encoder to facilitate message passing between hyperedges (headers) and nodes (cells), thereby perceiving the semantic entailment relationships between them within complex tables. Furthermore, we introduce a set of learnable query vectors acting as a lightweight bridge to extract representative structural embeddings from the encoder into the LLM. Experimental results demonstrate that our approach effectively handles complex
    
[^65]: REFACTOR-VLA：类型化运动程序的无监督技能库学习

    REFACTOR-VLA: Unsupervised Library Learning of Typed Motor Programs

    [https://arxiv.org/abs/2609.01215](https://arxiv.org/abs/2609.01215)

    REFACTOR-VLA 提出了一种清醒/睡眠两阶段框架，通过基于潜在世界模型 rollout 计算的行为等价核对运动程序片段进行无监督聚类，并生成 Hindley–Milner 风格的类型化 lambda 项来构建可复用技能库，从而提升 VLA 模型在长时程任务上的性能与可解释性。

    

    大多数视觉-语言-动作（VLA）模型——如 OpenVLA、π₀、RT-2、RDT-1B——都是单体式的：它们直接输出原始运动指令或短动作片段，而未将行为组织成可复用的抽象，因此在长时程任务上性能下降且难以解释。现有的技能发现方法回避了“两个动作序列何时在行为上等价”这一核心问题：要么对对比学习的嵌入进行聚类，要么把判断交给一个未针对机器人动力学进行校准的语言模型。我们提出 REFACTOR-VLA，一个用于学习可复用技能的清醒/睡眠系统。其睡眠阶段在行为等价核（Behavioral-Equivalence Kernel, BEK）下对运动程序片段进行聚类，该核由学习到的潜在世界模型 M_φ 的 rollout 计算得到；其清醒阶段基于受 Hindley–Milner 类型系统启发的词汇表生成类型化的 lambda 项，并由一个以技能库为条件的整流流动作解码器来消费。只有通过最小描述长度……（原文摘要在此处截断）

    arXiv:2609.01215v1 Announce Type: cross  Abstract: Most vision-language-action (VLA) models -- OpenVLA, $\pi_0$, RT-2, RDT-1B -- are monolithic: they emit raw motor commands or short action chunks without organizing behavior into reusable abstractions, so they degrade on long-horizon tasks and resist interpretation. Existing skill-discovery methods sidestep the core question of when two action sequences are behaviorally equivalent, either clustering contrastive embeddings or delegating the judgment to a language model uncalibrated to the robot's dynamics. We introduce REFACTOR-VLA, a wake/sleep system for learning reusable skills. Its sleep phase clusters motor-program fragments under a Behavioral-Equivalence Kernel (BEK) computed from rollouts of a learned latent world model $M_\phi$; its wake phase emits typed lambda terms over a Hindley--Milner-inspired vocabulary, consumed by a library-conditioned rectified-flow action decoder. Abstractions are admitted only if they pass Minimum De
    
[^66]: 谁来评判评判者？一个用于评估大语言模型回复与安全评判者的中文安全问答基准

    Who Judges the Judges? A Chinese Safety QA Benchmark for Evaluating LLM Responses and Safety Judges

    [https://arxiv.org/abs/2609.01210](https://arxiv.org/abs/2609.01210)

    该论文提出了C-SafeQA，一个以政策为依据的中文回复级安全评估基准，包含538个基础查询和8,877个对抗性查询，既能评估大语言模型回复的安全性，也能审计自动化安全评判者本身的可靠性。

    

    面向大语言模型的安全基准通常评估的是用户查询本身的风险，然而问答的实际结果取决于回复是否违反了相关政策。这一区别在中文有害内容评估中尤为关键，因为语言变体和对抗性转换可能会掩盖风险意图。我们提出了C-SafeQA，一个以政策为依据、面向回复级别的中文安全评估基准。该基准包含538个基础查询和8,877个对抗性查询，由四个全模型大语言模型部署进行回答，共产生37,660条被标注为安全、不安全或有争议的查询-回复记录。参考标签通过考虑一致性的多模型裁定机制以及三名安全专家对分层子集的盲审生成。C-SafeQA既支持对目标模型安全性的评估，也支持将七个自动化安全评判者与共享参考标签进行对照审计。在基础查询上，各模型的不安全回复率介于0.93%至3.35%之间。

    arXiv:2609.01210v1 Announce Type: cross  Abstract: Safety benchmarks for large language models often assess the risk of a user query, although the outcome of question answering depends on whether the response violates a policy. This distinction is critical in Chinese harmful-content evaluation, where linguistic variation and adversarial transformations can obscure risky intent. We introduce C-SafeQA, a policy-grounded benchmark for response-level Chinese safety evaluation. It comprises 538 base queries and 8,877 adversarial queries answered by four full-model LLM deployments, yielding 37,660 query-response records labeled safe, unsafe, or disputed. Reference labels are generated through agreement-aware multi-model adjudication and blind audits of stratified subsets by three safety experts. C-SafeQA supports both evaluation of target-model safety and auditing of seven automated safety judges against shared reference labels. Unsafe-response rates range from 0.93% to 3.35% on base queries
    
[^67]: 自主发现新型结构合理性定律，用于可解释且快速的晶体诊断与筛选

    Autonomous discovery of new structure-plausibility laws for explainable and rapid crystal diagnosis and screening

    [https://arxiv.org/abs/2609.01209](https://arxiv.org/abs/2609.01209)

    该研究通过智能体自主生成、测试并反驳两百万条候选定律，发现了八条可解释的无机结构合理性规则（PRIS），其诊断晶体结构的能力远超鲍林规则和传统距离截断法，并与可合成性线性相关，可实现快速晶体筛选。

    

    晶体生成器和工具使用智能体提出新结构的速度，已超过密度泛函理论（DFT）能量与声子计算或实验所能评估的速度。因此，决定哪些候选结构值得进行昂贵评估成为瓶颈，然而大多数筛选方法仅检验原子重叠等简单条件，且无法给出失败的化学原因。在此，我们的智能体生成、测试并主动反驳了两百万条候选定律，最终留下八条无机结构合理性规则（PRIS）。这些定律编码了五种机制：短程排斥、离子接触与堆积、静电平衡、键价守恒以及晶体学位点复杂性。实验结构以82–99%的比例满足我们的定律集，而同时满足鲍林规则2–5的仅为6.5%。最严格的规则集可检测出87.9%的受损晶体结构，而距离截断法仅能检测出1.6–3.2%。PRIS合理性评估结果与可合成性呈线性相关，因此……

    arXiv:2609.01209v1 Announce Type: cross  Abstract: Crystal generators and tool-using agents propose structures faster than density functional theory (DFT) energy and phonon calculations or experiments can assess them. Deciding which candidates merit expensive assessment is therefore the bottleneck, yet most screens test little beyond atomic overlap and give no chemical reason for failure. Here, our agents generate, test and actively refute two million candidate laws, leaving eight Plausibility Rules for Inorganic Structures (PRIS). These laws encode five mechanisms: short-range repulsion, ionic contact and packing, electrostatic balance, bond-valence conservation and crystallographic site complexity. Experimental structures satisfy our law sets at 82--99%, but satisfy Pauling's rules 2--5 together at only 6.5%. The strictest set detects 87.9% of damaged crystal structures, whereas distance cutoffs detect only 1.6--3.2%. PRIS plausibility is linearly correlated with synthesizability, so
    
[^68]: 迈向AI辅助的临床试验匹配：实践考量、多中心评估与真实世界部署

    Towards AI-Assisted Clinical Trial Matching: Practical Considerations, Multicenter Evaluation, and Real-World Deployment

    [https://arxiv.org/abs/2609.01202](https://arxiv.org/abs/2609.01202)

    本文提出面向真实世界部署的AI临床试验推荐系统TrialGPT 2.0，它不仅评估患者资格，还结合患者临床需求和本地工作流优先级筛选值得进一步考虑的试验，并提供了结构化的可审查解释，在政府、学术癌症中心等多种肿瘤学场景中完成了回顾性与前瞻性多中心评估。

    

    临床试验对于推进癌症治疗和药物研发至关重要，但许多试验因患者入组不足而失败。尽管人们利用AI支持患者招募的兴趣日益浓厚，现有系统大多仅执行资格评估，且很少在真实世界的肿瘤学工作流程中得到评估。在此，我们提出TrialGPT 2.0，一个为真实世界部署而设计的AI辅助临床试验推荐系统。该系统不仅评估患者是否符合入组条件，还会根据患者当前的临床需求和本地工作流程优先级，评估哪些试验值得进一步考虑，并提供结构化、可审查的解释供专家审核。重要的是，我们在多个专注于肿瘤学的环境中对TrialGPT 2.0进行了回顾性和前瞻性评估，涵盖政府、学术癌症中心、患者倡导组织和NIH转诊等多种工作流程。

    arXiv:2609.01202v1 Announce Type: cross  Abstract: Clinical trials are essential for advancing cancer care and drug development, but many fail because of insufficient patient enrollment. While there is growing interest in using AI to support patient recruitment, existing systems largely perform eligibility assessment alone and have rarely been evaluated in real-world oncology workflows. Here we present TrialGPT 2.0, an AI-assisted clinical trial recommendation system designed for real-world deployment. Rather than asking only whether a patient may qualify, the system also assesses which trials warrant further consideration given the patient's current clinical needs and local workflow priorities, and provides structured, inspectable explanations for expert review. Importantly, we evaluated TrialGPT 2.0 retrospectively and prospectively across multiple oncology-focused settings, spanning government, academic cancer-center, patient-advocacy, and NIH referral workflows. In retrospective mu
    
[^69]: FinLifeBench：从纵向银行对话中穷尽式重建人生事件历史与财务状态

    FinLifeBench: Exhaustive Life-Event History and Financial-State Reconstruction from Longitudinal Banking Dialogue

    [https://arxiv.org/abs/2609.01198](https://arxiv.org/abs/2609.01198)

    提出FinLifeBench基准，基于6,000个韩语银行对话会话，评估大语言模型在穷尽式重建客户人生事件历史与34维财务状态方面的长程记忆能力，发现随会话累积事件召回率显著下降（0.591降至0.445），且错误主要源于事件遗漏。

    

    重复的银行交互要求助手在生活变化随日常请求偶然出现时，维护完整、最新且可追溯的客户记录。现有基准强调问答、有界回合或定向回忆，而非穷尽式的纵向重建。我们提出FinLifeBench，它在同一累积对话上评估两项任务：重建每个人生事件实例及其首次确立的会话，以及在连续检查点上重建完整的34条路径财务状态。该基准包含来自20条独立合成轨迹的6,000个八轮韩语银行会话，为24种事件类型和34条状态路径提供确定性的、穷尽式的黄金标准及共识质量保证。在全上下文条件下对十一个大语言模型的评估中，事件锚点召回率从15个会话时的0.591下降至300个会话时的0.445。错误主要由遗漏事件导致，而非（摘要原文在此处截断）

    arXiv:2609.01198v1 Announce Type: cross  Abstract: Repeated banking interactions require assistants to maintain complete, current, and traceable customer records as life changes emerge incidentally in routine requests. Existing benchmarks emphasize question answering, bounded episodes, or targeted recall rather than exhaustive longitudinal reconstruction. We introduce FinLifeBench, which evaluates two tasks over the same cumulative dialogue: reconstructing every life-event instance with its first-establishing session and reconstructing a complete 34-path financial state at consecutive checkpoints. The benchmark contains 6,000 eight-turn Korean banking sessions from 20 independent synthetic trajectories, with deterministic, exhaustive gold for 24 event types and 34 state paths and consensus quality assurance. Across eleven LLMs under a full-context condition, event-anchor recall falls from 0.591 at 15 sessions to 0.445 at 300. Errors are driven primarily by omitted events rather than po
    
[^70]: Athena：基于知识图谱补全的漏洞受影响软件库识别

    Athena: Vulnerability-Affected Library Identification via Knowledge Graph Completion

    [https://arxiv.org/abs/2609.01187](https://arxiv.org/abs/2609.01187)

    提出了首个基于图的方法Athena，将漏洞数据库建模为知识图谱，通过知识图谱补全和链接预测自动识别并补全CVE缺失的受影响软件库信息。

    

    一个被广泛使用的软件库中存在的单个漏洞可能会级联影响到数百万个依赖它的应用程序，然而超过一半的漏洞数据库条目包含缺失或不正确的受影响库信息。现有的自动化方法忽略了漏洞数据库的关系结构，将识别任务视为孤立的文本检索问题。在本文中，我们提出了Athena，这是首个基于图方法的漏洞受影响库识别方法。Athena将漏洞数据库建模为知识图谱，并将识别问题重新表述为知识图谱补全（KGC）问题。它包含三个关键模块：建模模块，构建一个集成了CVE、软件库、CWE弱点类型、CPE产品和软件生态系统的安全知识图谱；补全模块，应用模块化的KGC主干网络，通过链接预测来预测给定CVE所缺失的受影响库；以及重排序模块……

    arXiv:2609.01187v1 Announce Type: cross  Abstract: A single vulnerability in a widely used library can cascade through millions of dependent applications, yet more than half of vulnerability database entries contain missing or incorrect affected-library information. Existing automated approaches neglect the relational structure of vulnerability databases, treating identification as an isolated text retrieval problem. In this paper, we propose Athena, the first graph-based approach for vulnerability affected library identification. Athena models vulnerability databases as a knowledge graph and reformulates the identification problem as knowledge graph completion (KGC). It comprises three key modules: a Modeling module that constructs a security knowledge graph integrating CVEs, libraries, CWE weakness types, CPE products, and software ecosystems; a Completion module that applies a modular KGC backbone to predict missing affected libraries for a given CVE via link prediction; and a Re-ra
    
[^71]: 穿越裂缝越狱文本到图像模型：通过多智能体辩论导航异构安全过滤器

    Jailbreaking Text-to-Image Models Through Cracks: Navigating Heterogeneous Safety Filters via Multi-Agent Debate

    [https://arxiv.org/abs/2609.01168](https://arxiv.org/abs/2609.01168)

    该论文提出“检测表面”几何框架来刻画异构安全过滤器的决策边界，揭示了跨层冲突塑造的稀疏非凸规避区域，并基于此通过多智能体辩论方法实现对文本到图像模型多层安全防护体系的越狱攻击。

    

    尽管文本到图像模型日益受到由文本过滤器、图像分类器和跨模态检测器组成的异构多层安全堆栈的保护，但它们仍然容易受到诱导生成不适内容（NSFW）的越狱攻击。现有的越狱研究要么针对单个过滤器进行优化，要么以聚合反馈的方式查询完整流水线，这使得难以识别起作用的约束并适应安全层之间的冲突。在本文中，我们引入了“检测表面”这一统一的几何框架，用于刻画异构文本到图像安全过滤器所诱导的决策边界及其对越狱搜索空间的联合影响。这一表述揭示了成功的规避是由一个稀疏且非凸的区域所支配的，该区域由跨层冲突塑造，其中绕过一个过滤器的变异可能会增加暴露于另一个过滤器的风险。基于这一分析，我们提出……

    arXiv:2609.01168v1 Announce Type: new  Abstract: Text-to-image (T2I) models remain vulnerable to jailbreak attacks that elicit Not-Safe-For-Work (NSFW) content, despite increasingly being guarded by heterogeneous, multi-layer safety stacks combining text filters, image classifiers, and cross-modal detectors. Existing jailbreak studies either optimize against individual filters or query the complete pipeline with aggregate feedback, making it difficult to identify the active constraint and adapt to conflicts across safety layers.In this paper, we introduce the \emph{Detection Surface}, a unified geometric framework that characterizes the decision boundaries induced by heterogeneous T2I safety filters and their joint effect on the jailbreak search space. This formulation reveals that successful evasion is governed by a sparse and non-convex region shaped by cross-layer conflicts, where mutations that bypass one filter may increase exposure to another. Motivated by this analysis, we propo
    
[^72]: 叠加潜在自编码器

    Superposed Latent Autoencoder

    [https://arxiv.org/abs/2609.01158](https://arxiv.org/abs/2609.01158)

    SLAE 通过学习到的叠加机制将多个高容量潜码绑定并叠加存储在单个内存张量中，以可抑制的结构化干涉取代传统自编码器不可逆的维度瓶颈，在相同存储预算下将重建误差最多降低 56%。

    

    自编码器通常通过将每个潜在表示变得更小来满足紧张的潜在内存预算，从而牺牲了表示能力。我们提出了一个不同的问题：能否将多个更宽的潜在表示一起存储？我们提出了叠加潜在自编码器（SLAE），它通过学习到的叠加方式共享存储，同时保持高容量的潜在表示。SLAE 将潜码转换为适合存储的编码，用随机化密钥将它们绑定，将多个编码叠加到单个内存张量中，并学习在解码前恢复每个潜码。在相同的存储预算下，SLAE 用可被抑制的结构化干涉取代了不可逆的维度瓶颈。在 CIFAR-10/100、SVHN、STL-10、Tiny ImageNet 以及广泛的内存预算范围内，SLAE 显著改善了重建-内存的权衡，在匹配的内存预算下，与传统自编码器相比，重建误差最多降低 56%。

    arXiv:2609.01158v1 Announce Type: cross  Abstract: Autoencoders typically meet tight latent-memory budgets by making each latent representation smaller, sacrificing representational capacity. We ask a different question: can multiple wider latents be stored together instead? We introduce the Superposed Latent Autoencoder (SLAE), which preserves high-capacity latent representations while sharing storage through learned superposition. SLAE transforms latents into storage-friendly codes, binds them with randomized keys, superposes multiple codes into a single memory tensor, and learns to recover each latent before decoding. Under the same storage budget, SLAE replaces irreversible dimensional bottlenecks with structured interference that can be suppressed. Across CIFAR-10/100, SVHN, STL-10, Tiny ImageNet, and a wide range of memory budgets, SLAE substantially improves the reconstruction--memory tradeoff, reducing reconstruction error by up to 56% over conventional autoencoders at matched 
    
[^73]: StainPresetNet：用于快速多对多染色归一化的染色预设网络

    StainPresetNet: Stain Preset Network for Fast Multi-to-Multi Stain Normalization

    [https://arxiv.org/abs/2609.01146](https://arxiv.org/abs/2609.01146)

    提出StainPresetNet框架，通过预设参考图像引导实现快速、高效的多对多染色归一化，在保持组织结构的同时实现准确的数据集级颜色映射，且改变归一化方向无需重新训练模型。

    

    染色归一化可以减少由染色方案和成像条件变化引起的颜色差异，从而提升计算机辅助诊断系统的性能。传统方法通过逐像素变换从单个或有限的参考图像中推导映射关系，虽然具有风格灵活性，但存在颜色映射提取不准确的问题。现有的基于深度学习的方法虽然通过复杂的神经网络实现了准确的数据集级颜色映射，但面临计算效率低下、产生伪影以及归一化方向固定（改变方向需要重新训练模型）等挑战。为了解决这些局限性，我们提出了StainPresetNet——一种新颖的框架，它在保持计算效率的同时，将结构保持与数据集级颜色映射相结合。我们的方法通过预设参考图像引导实现逐像素归一化，

    arXiv:2609.01146v1 Announce Type: cross  Abstract: Stain normalization reduces color variations caused by variations in staining protocols and imaging conditions, thereby enhancing computer-aided diagnostic system performance. Traditional methods derive mapping relationships from individual or limited reference images through pixel-wise transformation, offering style flexibility but suffering from inaccurate color mapping extraction. While existing deep-learning-based approaches achieve accurate dataset-wide color mapping through complex neural networks, they face challenges including computational inefficiency, artifact generation, and fixed normalization directions requiring model retraining for directional changes. To address these limitations, we propose StainPresetNet - a novel framework that combines structural preservation with dataset-level color mapping while maintaining computational efficiency. Our method implements pixel-wise normalization guided by preset reference images,
    
[^74]: 重新审视同卵双胞胎的人脸识别：名人双胞胎测试集

    Revisiting Face Recognition for Monozygotic Twins: The Celeb Twins Test Set

    [https://arxiv.org/abs/2609.01141](https://arxiv.org/abs/2609.01141)

    本文提出首个包含独特皮肤标记和镜像不对称性元数据的名人双胞胎测试集CTTS，实验表明当前深度CNN匹配器虽能达到76%以上的准确率，但并未利用这些区分性特征，并探讨了用生成式AI合成虚拟同卵双胞胎图像来扩充训练数据的可行性。

    

    过去关于同卵双胞胎人脸识别的文献指出，面部标记和镜像不对称性可能是提高双胞胎识别准确率的研究方向。Celeb Twins测试集（CTTS）包含80对名人双胞胎的网络爬取图像对，它是唯一一个为具有独特皮肤标记和可能镜像不对称性的双胞胎提供元数据的双胞胎测试集。CTTS按照LFW、CALFW、CPLFW、CFP-FP和AgeDB-30等人脸验证测试集的方式组织。当前的深度CNN匹配器在对CTTS的同人/不同人图像对进行分类时可以达到超过76%的准确率。我们证明当前的匹配器并未利用皮肤标记或不对称性特征，并讨论了其中的原因。最后，我们探讨了使用Grok、ChatGPT和Gemini等生成式AI工具创建想象中的同卵双胞胎图像的可行性，以此作为增加人脸识别训练数据中双胞胎代表性的手段。

    arXiv:2609.01141v1 Announce Type: cross  Abstract: Past literature on face recognition for monozygotic (("identical") twins points to facial marks and mirror asymmetry as possible directions for improved accuracy of twins recognition. The Celeb Twins Test Set (CTTS) contains web-scraped image pairs for 80 sets of celebrity twins. It is the only twins test set with meta-data for twins with distinguishing skin marks and possible mirror asymmetry. CTTS is organized in the manner of face verification test sets such as LFW, CALFW, CPLFW, CFP-FP, and AgeDB-30. Current deep CNN matchers can achieve over 76% accuracy in classifying CTTS same-person / different-person image pairs. We show that current matchers do not make use of skin marks, or asymmetry, and discuss reasons for this. Finally, we discuss the feasibility of using generative AI tools such as Grok, ChatGPT and Gemini to create images of imagined monozygotic twins as a means to increase representation of twins in face recognition tr
    
[^75]: DNC-IMM：基于驾驶上下文信息的神经校准实现换道意图早期识别

    DNC-IMM: Early Lane-Change Intention Recognition via Neural Calibration Based on Driving Context Information

    [https://arxiv.org/abs/2609.01120](https://arxiv.org/abs/2609.01120)

    本文提出双神经校准交互多模型DNC-IMM，利用神经网络基于驾驶上下文信息（目标车辆运动、周围车间距与相对速度）校准转移概率矩阵和测量似然，在保留传统IMM概率结构与可解释性的前提下，实现换道意图的早期可靠识别，并在2-3秒预测时域内性能尤为突出。

    

    换道意图的早期识别对于自动驾驶和高级驾驶辅助系统中的主动决策至关重要。本文提出了一种双神经校准交互多模型（DNC-IMM），在保持传统IMM概率结构和可解释性的同时，提高了对驾驶上下文的适应性。该方法利用神经网络对驾驶上下文信息进行编码，包括目标车辆运动状态、与周围车辆的间距以及相对速度，并据此对转移概率矩阵和测量似然进行校准。最终的换道意图由校准后的IMM模式后验确定，而非依靠单独的直接分类器。在高D数据集上的实验表明，所提出的方法能够在车辆跨越车道线之前可靠地识别换道意图，并在较早的2-3秒预测时域内表现出尤为出色的性能。

    arXiv:2609.01120v1 Announce Type: cross  Abstract: Early recognition of lane-change intention is essential for proactive decision-making in autonomous driving and advanced driver assistance systems. This paper proposes a Dual Neural-Calibrated Interacting Multiple Model (DNC-IMM) that improves adaptability to driving context while preserving the probabilistic structure and interpretability of a conventional IMM. The proposed method encodes driving-context information, including target-vehicle motion, gaps to surrounding vehicles, and relative velocities, with a neural network that calibrates both the transition-probability matrix and measurement likelihoods. The final intention is determined from the calibrated IMM mode posterior rather than from a separate direct classifier. Experiments on the highD dataset demonstrate that the proposed method reliably recognizes lane-change intentions before lane crossing and provides particularly strong performance at the earlier 2-3 s prediction ho
    
[^76]: 潜在循环思维：基于冻结大语言模型推理的潜在向量循环精炼方法

    Latent Recurrent Thoughts: Recurrent Refinement of Proposed Latents for Reasoning with Frozen LLMs

    [https://arxiv.org/abs/2609.01117](https://arxiv.org/abs/2609.01117)

    该论文提出潜在循环思维（LRT）方法，通过保持大语言模型冻结并引入一个微型循环推理器在连续潜在空间中多步迭代精炼潜在思维向量来进行推理，将计算深度与模型规模解耦，从而规避了思维链推理中误差传播以及需要可模仿轨迹的固有局限。

    

    思维链推理在离散的词元空间中展开：每一步都被固化为文本，误差会不断传播，且要引出高质量的推理轨迹，前提是已有可供模仿的轨迹。而改在模型的连续表示空间中进行推理——其中间状态是向量而非词语——可以规避这些限制，但这些潜在状态应当如何计算仍是一个悬而未决的问题。我们从两个维度着手解决这一问题。首先，我们保持大语言模型（LLM）冻结不变，仅利用其已经擅长的工作——建模和解码序列——同时由一个小型辅助网络提供连续的潜在思维作为输入。其次，我们通过循环递归的方式生成这些潜在向量：一个微型循环推理器在多步过程中对其进行精炼，将计算深度与模型规模解耦，使潜在向量成为迭代处理的产物而非单次前向传播的结果。我们将这一方法实例化为潜在循环思维（Latent Recurrent Thoughts，LRT）：一个面向任务的……

    arXiv:2609.01117v1 Announce Type: new  Abstract: Chain-of-thought reasoning unfolds in discrete token space: each step is committed as text, errors propagate, and eliciting good traces presupposes traces to imitate. Reasoning instead in a model's continuous representation space - where intermediate states are vectors rather than words - sidesteps these constraints, but leaves open how those latent states should be computed. We approach this along two axes. First, we keep a large language model (LLM) frozen and use it for what it is already good at - modeling and decoding sequences - while a small auxiliary network supplies continuous latent thoughts as input. Second, we produce those latents by recurrence: a tiny recurrent reasoner refines them over many steps, decoupling the depth of computation from the size of the model, so that the latents are a product of iterative processing rather than a single forward pass. We instantiate this as Latent Recurrent Thoughts (LRT): a task-dedicate
    
[^77]: EDRAC：阿拉伯语方言阅读理解基准测试

    EDRAC: Benchmarking Arabic Dialect Reading Comprehension

    [https://arxiv.org/abs/2609.01113](https://arxiv.org/abs/2609.01113)

    EDRAC是首个面向阿拉伯语方言机器阅读理解与生成式问答的大规模基准，涵盖埃及、摩洛哥、阿联酋、叙利亚和沙特五种主要方言，包含499篇自然口语段落和通过人-大语言模型协作流水线生成的4,977个问答对，并以此评测了阿拉伯语和多语言大语言模型的表现。

    

    与现代标准阿拉伯语（MSA）相比，方言阿拉伯语（DA）的资源仍然匮乏，尤其是在机器阅读理解（MRC）和问答（QA）任务方面。现有的阿拉伯语问答基准主要聚焦于正式书面的现代标准阿拉伯语或多选题式问答，对自然口语方言的覆盖十分有限。本文旨在弥合这一差距。我们提出了EDRAC，这是首个面向方言阿拉伯语机器阅读理解（MRC）与生成式问答的大规模基准，涵盖五大主要方言：埃及、摩洛哥、阿联酋、叙利亚和沙特阿拉伯方言。EDRAC包含499篇源自自然发生的口语交互的段落，以及4,977个通过人-大语言模型协作流水线生成的对应问答对，该流水线结合了迭代生成、以LLM作为评判者的评估以及人工验证。我们使用词汇和语义指标在EDRAC上对以阿拉伯语为中心和多语言大语言模型进行了基准评测。我们的结果揭示了（各模型之间的）显著差距……

    arXiv:2609.01113v1 Announce Type: cross  Abstract: Dialectal Arabic (DA) remains under-resourced compared to Modern Standard Arabic (MSA), particularly for machine reading comprehension (MRC) and question answering (QA). Existing Arabic QA benchmarks primarily focus on formal written MSA or multiple-choice QA, with limited coverage of naturally spoken dialects. Here, we aim to bridge this gap. We introduce EDRAC, the first large-scale benchmark for dialectal Arabic machine reading comprehension (MRC) and generative QA, covering five major dialects: Egyptian, Moroccan, Emirati, Syrian, and Saudi Arabic. EDRAC contains 499 passages derived from naturally occurring spoken interactions and 4,977 corresponding QA pairs generated through a human--LLM collaborative pipeline combining iterative generation, LLM-as-a-judge evaluation, and human verification. We benchmark Arabic-centric and multilingual LLMs on EDRAC using lexical and semantic metrics. Our results reveal substantial gaps between 
    
[^78]: 提示有帮助，但它们真的能“教会”模型吗？评估代码生成中的技能迁移

    Hints Help But Do They Teach? Evaluating Skills Transfer in Code Generation

    [https://arxiv.org/abs/2609.01106](https://arxiv.org/abs/2609.01106)

    研究发现，提示对失败代码生成的“挽救”效果大多可通过无提示的重复采样复现，且相关与无关提示共享同一激活方向，表明提示更多是引导模型已有能力而非传授新技能。

    

    当一条提示能把一个失败的生成程序变成通过测试的程序时，它究竟提供了缺失的信息，还是仅仅将模型引导至它本来就能得出的解？我们在 HumanEval+ 和 MBPP+ 上通过可执行评估来检验这些假设。对于 Qwen2.5-3B-Instruct，自适应的相关提示挽救了 79 个选定失败样例中的 36 个；无关提示挽救了 19 个；而在无提示条件下，8 次采样解决了 46 个样例，并覆盖了 36 个相关提示挽救中的 31 个。Phi-3.5-mini 呈现出相同的模式：相关提示挽救了 101 个失败中的 42 个，无关提示挽救了 17 个，无提示采样解决了 57 个，其中包括 42 个相关提示挽救中的 36 个。由于各提示条件使用了不同的尝试预算，这些比较并不能分离出纯粹的语义效应。在 Qwen 上进行的机制性测试发现，相关提示与无关提示共享一个稳定的激活方向。持续向该方向添加偏移会产生 14 次挽救和 18 次回退，且未检测到……（原文摘要在此处被截断）

    arXiv:2609.01106v1 Announce Type: cross  Abstract: When a hint turns a failing generated program into a passing one, does it provide missing information or merely steer the model toward a solution it could already produce? We test these hypotheses on HumanEval+ and MBPP+ using executable evaluation. For Qwen2.5-3B-Instruct, adaptive relevant hints rescue 36 of 79 selected failures; an unrelated hint rescues 19, while eight unhinted samples solve 46 and recover 31 of the 36 relevant-hint rescues. Phi-3.5-mini shows the same pattern: relevant hints rescue 42 of 101 failures, an unrelated hint rescues 17, and unhinted sampling solves 57, including 36 of the 42 relevant-hint rescues. Because the hint conditions use different attempt budgets, these comparisons do not isolate a purely semantic effect. Mechanistic tests on Qwen identify a stable activation direction shared by relevant and unrelated hints. Persistently adding this direction yields 14 rescues and 18 regressions, with no detecta
    
[^79]: StateSwap：探究多选题中“支持型”与“排除型”框架下的隐藏状态

    StateSwap: Probing Support-Elimination Hidden States in Multiple-Choice Questions

    [https://arxiv.org/abs/2609.01081](https://arxiv.org/abs/2609.01081)

    该论文提出StateSwap方法，通过添加特殊标记[STATE]来探测并交换多选题在“支持型”与“排除型”两种表述下诱导出的隐藏状态激活，证明两种框架在模型中间层产生可分离的内部表示，且交换这些激活可因果性地改变预测结果并提高跨框架答案一致性。

    

    当同一道多项选择题以“支持型”（寻找依据）和“排除型”（逐一排除）两种不同表述方式提出时，大型语言模型往往给出不一致的答案。我们研究这些差异是否源于两种表述方式所诱导的不同内部表示。我们提出了一种双框架协议，使用仅存在最小差异的提示词——这些提示词分别采用支持型或排除型表述，同时保持评估目标固定不变。为了探测内部计算，我们在提示词末尾附加一个未经训练的特殊标记 [STATE]，并将其残差流激活视为干预接口。在所测试的两个模型中，两种框架均诱导出可分离的 [STATE] 激活，且这些激活集中于中间层。在配对的提示词之间交换这些激活会系统性地改变模型预测，并提升跨框架的答案一致性，从而提供了基于干预的证据，证明这些激活与模型行为密切相关。在实例层面的替换之外……（摘要原文在此处截断）

    arXiv:2609.01081v1 Announce Type: cross  Abstract: Large language models often answer the same multiple-choice question inconsistently when it is posed under support-oriented and elimination-oriented framings. We investigate whether these discrepancies arise from different internal representations induced by the two framings. We introduce a dual-framing protocol with minimally varied prompts that use either support- or elimination-oriented framing while keeping the evaluation target fixed. To probe the internal computation, we append an untrained special token, [STATE], and treat its residual-stream activation as an intervention interface. Across both models, the two framings induce separable [STATE] activations concentrated in intermediate layers. Swapping these activations between paired prompts systematically changes predictions and improves cross-framing agreement, providing intervention-based evidence that the activations are behaviorally relevant. Beyond instance-level substituti
    
[^80]: 文本引导的流匹配实现样本高效的晶体结构生成

    Text-guided flow matching enables sample-efficient crystal structure generation

    [https://arxiv.org/abs/2609.01076](https://arxiv.org/abs/2609.01076)

    TFMat提出了一种以结构化材料文本作为语义先验的文本条件化流匹配框架，显著提升了晶体结构生成的匹配率（MP-20基准20个候选下达92.04%），实现了样本高效且可通过自然语言控制的晶体生成。

    

    晶体生成器如今已能够提出周期性结构，但其控制接口与材料设计中常用的混合描述符匹配不佳。文本提供了一种紧凑的方式来组合成分、对称性、原型和性质等线索，然而此类信息能否引导基于流的晶体生成尚不清楚。本文提出TFMat，一个文本条件化的流匹配框架，将结构化的材料语言作为CrystalFlow生成器的语义先验。在Perov-5、Carbon-24和MP-20晶体结构预测基准上，TFMat的单候选匹配率相比CrystalFlow有所提升，并在20个候选下达到92.04%的MP-20匹配率；在从头生成任务中，它改善了元素数目和密度分布的对齐，同时在按成分筛选的输出中保持粗粒度的性质一致性。这些结果表明，结构化文本可以作为一个可检查的控制层，用于将人类可读的语言转化为晶体生成的控制信号。

    arXiv:2609.01076v1 Announce Type: cross  Abstract: Crystal generators can now propose periodic structures, but their control interfaces remain poorly matched to the mixed descriptors used in materials design. Text provides a compact way to combine composition, symmetry, prototype and property cues, yet it has not been clear whether such information can steer flow-based crystal generation. Here we introduce TFMat, a text-conditioned flow-matching framework that uses structured materials language as a semantic prior for a CrystalFlow generator. Across Perov-5, Carbon-24 and MP-20 crystal structure prediction benchmarks, TFMat improves one-candidate match rates over CrystalFlow and reaches a 92.04% MP-20 match rate with 20 candidates; in de novo generation, it improves element-count and density distribution alignment while retaining coarse property consistency in composition-selected outputs. These results position structured text as an inspectable control layer for translating human-read
    
[^81]: 基于太阳能收集的太空生成式人工智能

    Space Generative AI with Solar Energy Harvesting

    [https://arxiv.org/abs/2609.01062](https://arxiv.org/abs/2609.01062)

    本文提出了一个太阳能驱动的太空生成式AI框架，揭示了在共享收集能量预算下，扩散模型图像生成步数与下行传输之间的基本计算-通信权衡关系。

    

    卫星正成为向缺乏地面基础设施的偏远地区扩展生成式人工智能（AI）服务的有前景的平台。然而，部署太空生成式AI从根本上受到太阳能收集（EH）所提供的有限的、随时间变化的星载能量的制约。本文提出了一个太阳能驱动的太空生成式AI框架，其中卫星接收用户提示词，执行基于扩散模型的图像生成，并在严格的时间窗口内下行传输压缩后的结果。我们识别了由共享的收集能量预算所支配的基本计算-通信（C²）权衡。具体而言，增加生成步数可以提升图像的内在质量，但会耗尽用于下行传输的能量和时间；而优先保障通信则能确保可靠传输，但会牺牲语义质量。为了平衡……

    arXiv:2609.01062v1 Announce Type: new  Abstract: Satellites are emerging as promising platforms to extend generative \emph{artificial intelligence} (AI) services to remote areas lacking terrestrial infrastructure. However, deploying space generative AI is fundamentally constrained by the limited, time-varying onboard energy supplied by solar \emph{energy harvesting} (EH). This paper presents a framework for solar-powered space generative AI in which a satellite receives a user prompt, executes a diffusion-based image-generation model, and downlinks the compressed result within a strict time window. We identify the fundamental \emph{computation--communication} (C$^2$) trade-offs governed by the shared harvested-energy budgets. Specifically, increasing the number of generation steps improves intrinsic image quality but depletes energy and time available for downlink transmission, whereas prioritizing communication guarantees reliable delivery but sacrifices semantic quality. To balance t
    
[^82]: ARISE-RL：基于智能体量规的迭代自我进化强化学习

    ARISE-RL: Agentic Rubric-Grounded Iterative Self-Evolution with Reinforcement Learning

    [https://arxiv.org/abs/2609.01058](https://arxiv.org/abs/2609.01058)

    ARISE-RL 提出了一种通过量规介导的生成器与求解器协同进化的全周期自我进化框架，解决了开放式智能体强化学习训练中缺乏可验证答案且奖励信号脆弱不稳定的问题。

    

    通过强化学习（RL）训练开放式智能体受到缺乏可验证的黄金答案和可扩展量规的阻碍。此外，即使在模型能力边界附近，长时程开放式智能体任务往往产生脆弱且不稳定的奖励，导致 rollout 对比信号微弱或有噪声，从而掩盖了群体式策略学习所需的细粒度优化信号。为解决这些挑战，我们提出了 ARISE-RL，这是一个新颖的全周期自我进化框架，通过量规介导的协同进化将任务/量规生成器与推理求解器耦合起来。生成器将工具相关的量规标准建立在真实的工具观察之上，并因生成与求解器不断演进的能力边界相匹配的有效、中等难度任务而获得奖励。求解器则通过多步推理和工具使用，从细粒度的量规满足信号中学习。我们进一步引入了奖励门控自我进化……（原文摘要在此处截断）

    arXiv:2609.01058v1 Announce Type: new  Abstract: Training open-ended agents via reinforcement learning (RL) is hindered by the lack of verifiable gold answers and scalable rubrics. Moreover, even near the model's capability boundary, long-horizon open-ended agentic tasks often yield brittle and unstable rewards, resulting in weak or noisy rollout contrast that obscures fine-grained optimization signals for group-based policy learning. To address these challenges, we propose ARISE-RL, a novel full-cycle self-evolution framework that couples a task/rubric Generator and a reasoning Solver through rubric-mediated co-evolution. The Generator grounds tool-related rubric criteria in real tool observations and is rewarded for producing valid, intermediate-difficulty tasks aligned with the Solver's evolving capability boundary. The Solver, in turn, learns from fine-grained rubric satisfaction signals through multi-step reasoning and tool use. We further introduce Reward-Gated Self-Evolution Dis
    
[^83]: 基于跨多源行为预训练的移动游戏用户表示学习

    User Representation via Cross Multi-source Behavior Pre-training for Mobile Games

    [https://arxiv.org/abs/2609.01057](https://arxiv.org/abs/2609.01057)

    该论文提出CM-PTM模型，通过层级级联的掩码预测预训练任务，在设备级行为日志上对跨多源用户行为进行统一建模，从而学习更精准的移动游戏用户表示，突破了传统单应用建模的局限。

    

    用户表示预训练已成为缓解下游个性化任务中数据稀疏问题的基本范式。然而，现有研究主要聚焦于单一应用或应用层面的行为，忽视了移动设备上用户活动的天然跨源性与多粒度特性。在设备层面，用户意图源自异构行为源与层级化动作结构之间的复杂交互，这带来了传统以应用为中心的建模方法无法解决的挑战。为解决这一问题，我们提出了CM-PTM，一种新颖的跨多源行为预训练模型，专为基于设备级行为日志的移动游戏用户表示学习而设计。CM-PTM采用层级级联的“先掩码后预测”代理任务，首先推断下一行为的来源，随后在应用-动作层面逐步细化预测。这一设计实现了对跨源行为的统一建模。

    arXiv:2609.01057v1 Announce Type: new  Abstract: User representation pre-training has become a fundamental paradigm for alleviating data sparsity in downstream personalization tasks. However, existing studies predominantly focus on single-app or app-level behaviors, overlooking the inherently cross-source and multi-granular nature of user activities on mobile devices. At the device level, user intent emerges from complex interactions among heterogeneous behavior sources and hierarchical action structures, posing challenges that cannot be addressed by conventional app-centric modeling. To tackle this issue, we propose CM-PTM, a novel Cross Multi-source Behavior Pre-Training Model tailored for mobile game user representation learning on device-level behavioral logs. CM-PTM employs hierarchical cascaded mask-then-predict proxy tasks that first infer the source of the next behavior and then progressively refine predictions at the app-action level. This design enables unified modeling of cr
    
[^84]: WorldBench：面向多语言智能体的文化扎根基准

    WorldBench: Culturally Grounded Benchmark for Multilingual Agents

    [https://arxiv.org/abs/2609.01056](https://arxiv.org/abs/2609.01056)

    WorldBench是一个涵盖七种语言、八种文化、包含1,600个真实日常任务的多语言智能体基准，并引入约束任务成功率（CTS）指标，以全面评估LLM智能体在真实文化扎根场景中的跨语言多步骤任务执行能力。

    

    尽管基于大语言模型（LLM）的智能体在复杂环境中解决多步骤任务的应用日益增多，现有基准很少测试状态保持能力、跨语言性能以及对真实扎根场景的适用性。为解决这些问题，我们提出了WorldBench：一个全面的、多语言的基准，其任务源自真实且基于人物角色的日常工作流程，智能体可在沙盒环境中通过结构化动作执行操作。WorldBench包含涵盖七种语言和八种文化的1,600个任务，这些任务经过具备特定语言和文化专业知识的人类标注者反馈进行筛选与精炼。在评估方面，我们扩展了以往工作的指标，并引入了约束任务成功率（CTS），该指标结合自然语言指令和测试环境，通过确定性评估和“LLM作为裁判”的评估方式，对任务完成度、最小修改度及其他补充指标进行评分。我们的实验表明，前沿模型在最……（摘要原文在此处截断）

    arXiv:2609.01056v1 Announce Type: new  Abstract: Despite the growing use of LLM-powered agents to solve multi-step tasks in complex environments, existing benchmarks rarely test state preservation, performance across languages, and application to realistic, grounded scenarios. To address these concerns, we present WorldBench: a comprehensive, multilingual benchmark of genuine, persona-grounded everyday workflows, where agents can act in a sandbox via structured actions. WorldBench comprises 1,600 tasks across seven languages and eight cultures, filtered and refined through feedback from human annotators with language- and culture-specific expertise. For evaluation, we extend metrics from previous works and introduce Constrained Task Success (CTS), which combines natural language instructions and testbeds to score task completion, minimal modification, and other complementary metrics through deterministic and LLM-as-a-Judge evaluations. Our experiments show that frontier models reach on
    
[^85]: QILP-0：构建量子电路的观测性声明式孪生体

    QILP-0: Constructing Observational Declarative Twins of Quantum Circuits

    [https://arxiv.org/abs/2609.01049](https://arxiv.org/abs/2609.01049)

    本文提出QXymb通用框架及其首个完整的零阶特化QILP-0，能够从量子电路的观测行为中自动构建有限多值命题逻辑程序，实现量子电路的观测性声明式孪生体构建。

    

    本文介绍了QXymb，一个用于构建量子电路观测性声明式孪生体的通用框架，并开发了QILP-0，即其首个完整的零阶特化版本。QILP-0在声明的观测范围内，从观测到的电路行为中构建一个有限多值命题逻辑程序。该流程根据可重现的结构分级和声明的观测参考视界，增量地遍历一个声明的量子可观测量族。进展通过相对于一个固定的、与目标无关的参考的覆盖率来量化。可观测量响应通过与目标无关的几何结构进行组织，同时所保留的潜在结构在符号处理之前被确定性地映射回原始可观测量列，从而保持观测语义和溯源信息。所选的可观测量轮廓通过可容许的、与目标无关的方式被转换为有限关系……

    arXiv:2609.01049v1 Announce Type: new  Abstract: This paper introduces QXymb, a general framework for constructing observational declarative twins of quantum circuits, and develops QILP-0, its first complete order-0 specialization. QILP-0 constructs a finite multi-valued propositional logic program from observed circuit behaviour within a declared observational scope.   The pipeline traverses a declared family of quantum observables incrementally according to a reproducible structural grading and a declared observational reference horizon. Progress is quantified through reference-relative coverage against a fixed target-independent reference. Observable responses are organized through target-independent geometry, while retained latent structure is mapped deterministically back to original observable columns before symbolic processing, preserving observational semantics and provenance.   Selected observable profiles are converted into a finite relation through admissible target-independ
    
[^86]: 滞后耦合：内部表征在具有因果作用之前就已变得可读

    Lagged Coupling: Internal Representations Become Readable Before They Become Causal

    [https://arxiv.org/abs/2609.01048](https://arxiv.org/abs/2609.01048)

    该研究在 Pythia 全系列模型中发现“滞后耦合”现象：线性探针能极早读出内部表征，但利用这些方向进行引导干预却几乎无效，且“可读但不可因果干预”的滞后并不随模型规模增大而缩小。

    

    在整个 Pythia 模型套件上（参数规模 160M-12B、八个检查点、四个任务族），线性探针最早在第 1,000 步就能在每个规模下从残差流中读出目标变量——然而，沿着同一读取方向进行引导干预，在 48 个“模型-检查点”组合中却有 43 个与零效应等价。内部可读性系统性地超前于因果有效性，而且这种滞后并不随规模增大而缩小。我们将这一结构称为“滞后耦合”，并将其分解为三条可分离的轨道：（i）内部可读性，在所有位置从第一个检查点起就已饱和（AUROC ≥ 0.990）；（ii）行为可读性，逐步发展且在更大规模下出现得更晚（12B 模型直到最后一个检查点才达到 0.909）；（iii）因果有效性，几乎总是与零效应等价，早期偶尔甚至适得其反，仅有一处孤立的正向脉冲（12B，第 8,000 步，z = +2.49）是本研究的网格无法解析的。其顺序以“先可读、后可写”为主导（11/11

    arXiv:2609.01048v1 Announce Type: cross  Abstract: Across the full Pythia suite (160M-12B, eight checkpoints, four task families), a linear probe can read a target variable from the residual stream as early as step 1,000 at every scale -- yet steering along that same reading direction remains null-equivalent in 43 of 48 model-checkpoint cells. Internal readability systematically outruns causal efficacy, and the lag does not shrink with scale. We call this structure lagged coupling and decompose it into three dissociable tracks: (i) internal readability, saturated (AUROC >= 0.990) from the first checkpoint everywhere; (ii) behavioral readability, which develops gradually and progressively later at larger scales (12B reaches 0.909 only at the final checkpoint); (iii) causal efficacy, almost always null-equivalent, occasionally counterproductive early, with one isolated positive pulse (12B, step 8,000, z = +2.49) our grid cannot resolve. The ordering is dominantly read-before-write (11/11
    
[^87]: HiveTraceGuard-Pro：面向提示注入、越狱攻击与对抗性混淆的紧凑型生成式防护栏

    HiveTraceGuard-Pro: A Compact Generative Guardrail for Prompt Injection, Jailbreaks, and Adversarial Obfuscation

    [https://arxiv.org/abs/2609.01046](https://arxiv.org/abs/2609.01046)

    该论文提出了HiveTraceGuard-Pro，一个基于Qwen3-0.6B经LoRA微调的0.6B参数紧凑型生成式防护栏，可检测俄语和英语环境下的提示注入、越狱攻击及对抗性混淆，在十九个基准组的评测中取得0.7432的综合得分，性能接近更大的防护模型。

    

    生产环境中的大语言模型必须应对那些试图覆盖系统指令、绕过安全策略或诱导有害响应的输入。一种常见的缓解措施是使用独立的防护栏模型。然而，现有报告几乎没有提供关于俄语提示注入或俄语表层混淆的证据。我们提出了HiveTraceGuard-Pro，这是一个基于Qwen3-0.6B经LoRA微调的0.6B参数生成式防护栏。它在俄语和英语数据上训练，并对最终目标轮次采用单一的二元评分规则（安全/不安全）。其训练语料库在存在对应样本的情况下，将有害样本与来自同一领域的良性样本配对，并对两类标签应用八种混淆变换。在同一个评测框架下，我们将HiveTraceGuard-Pro与其他三十四个防护模型在十九个基准组（其中十六个为公开基准）上进行了比较。其综合关键得分为0.7432，略低于得分最高的两个防护模型的0.7641和0.7552。仅就十六个公开基准组而言，其关键得分为0.7153……

    arXiv:2609.01046v1 Announce Type: cross  Abstract: Production LLMs must handle inputs that attempt to override system instructions, bypass safety policies or elicit harmful responses. A common mitigation is a separate guardrail model. Existing reports, however, provide little evidence on Russian prompt injection or Russian surface obfuscation. We present HiveTraceGuard-Pro, a 0.6B generative guardrail LoRA-tuned from Qwen3-0.6B. It is trained on Russian and English and uses one binary scoring rule (safe/unsafe) for the final target turn. Its training corpus pairs harmful examples, where a counterpart exists, with benign examples from the same domain and applies eight obfuscation transforms to both labels. In one harness, we compare HiveTraceGuard-Pro with thirty-four other guards on nineteen benchmark groups, sixteen of which are public. Its aggregate key is 0.7432, behind 0.7641 and 0.7552 for the two higher-scoring guards. Over the sixteen public groups alone, its key is 0.7153 and f
    
[^88]: AgentFactory：迈向自动化的智能体系统设计与优化

    AgentFactory: Towards Automated Agentic System Design and Optimization

    [https://arxiv.org/abs/2609.01045](https://arxiv.org/abs/2609.01045)

    AgentFactory是一个在性能、成本和效率等多目标约束下，利用大语言模型作为优化器，对智能体系统中的基础模型和工作流结构进行联合自动化优化的框架。

    

    大语言模型（LLMs）作为智能体系统中的强大组件，已经展现出卓越的能力，能够实现复杂的推理和复杂任务的执行。然而，目前手动设计和优化智能体系统的方法严重依赖人工操作，限制了其适应性和可扩展性。近期的工作已经探索了工作流设计的自动优化，但这些方法往往忽视了模型能力的关键作用，并且只关注单一性能指标，未能解决实际部署中的约束条件。在本文中，我们提出了AgentFactory，这是一个在考虑性能、成本和效率等多重目标的前提下，对智能体系统中的基础模型和工作流结构进行联合优化的框架。AgentFactory利用先进的大语言模型作为优化器，在庞大的配置搜索空间中进行导航，并采用三阶段的优化方法……

    arXiv:2609.01045v1 Announce Type: new  Abstract: Large Language Models (LLMs) have demonstrated remarkable capabilities as powerful components in agentic systems, enabling sophisticated reasoning and complex task execution. However, current approaches to manually designing and optimizing agentic systems heavily rely on manual effort, limiting their adaptability and scalability. Recent work has explored the automated optimization of workflow designs. However, these approaches often overlook the crucial role of model capabilities and focus on single performance metrics, failing to address real-world deployment constraints. In this paper, we present AgentFactory, a framework that jointly optimizes both foundation models and workflow structures in agentic systems while considering multiple objectives including performance, cost, and efficiency. AgentFactory leverages advanced LLMs as optimizers to navigate the vast search space of possible configurations, employing a three-stage optimizati
    
[^89]: 从截断到承诺：均匀离散扩散中的持久上下文

    From Truncation to Commitment: Persistent Context in Uniform Discrete Diffusion

    [https://arxiv.org/abs/2609.01043](https://arxiv.org/abs/2609.01043)

    提出一种无需训练的承诺式揭示采样（CRS），将选定的词元作为持久上下文插入后续模型输入，使均匀离散扩散模型的并行预测能在序列级选择上保持一致。

    

    均匀状态离散扩散模型并行更新所有词元，同时保持每个位置都可被修改。即使常用的 top-p 规则在一个位置只留下一个候选，该选择也仅影响当前的反向步骤，并可在下一个采样步骤中被修改。我们探讨当被选中的假设转而成为后续预测的持久上下文时会发生什么变化。为此，我们提出了承诺式揭示采样，这是一种无需训练的采样器，它存储被选中的 argmax 词元，并将其插入后续的模型输入中。我们的分析为“更晚做出选择”和“保持被选词元可见”提供了理论依据：在精确的前向过程下，随着噪声降低，选择干净词元的贝叶斯误差不会增加；而在一个简单的潜变量模式模型中，保持被选词元可见有助于后续的并行预测在相同的序列级选择上达成一致。实证上，在 Duo-distilled 模型上的成对实验（摘要在此处截断）……

    arXiv:2609.01043v1 Announce Type: cross  Abstract: Uniform-state discrete diffusion models update all tokens in parallel while keeping every position revisable. Even when the commonly used top-$p$ rule leaves only one candidate at a position, that choice affects only the current reverse step and can be revised at the next sampling step. We ask what changes when selected hypotheses instead become persistent context for later predictions. We therefore propose committed reveal sampling (CRS), a training-free sampler that stores selected argmax tokens and inserts them into subsequent model inputs. Our analysis gives a rationale for selecting later and for keeping selected tokens visible. Under the exact forward process, the Bayes error of selecting a clean token cannot increase as noise decreases, while in a simple latent-mode model, keeping the selected token visible helps later parallel predictions agree on the same sequence-level choice. Empirically, paired experiments on Duo-distilled 
    
[^90]: ViTAMINS：使用合成困难负样本训练自监督视觉Transformer的实证研究

    ViTAMINS: An Empirical Study of Training Self-Supervised Vision Transformers with Synthetic Hard Negatives

    [https://arxiv.org/abs/2609.01041](https://arxiv.org/abs/2609.01041)

    ViTAMINS通过向自监督视觉Transformer的对比学习预训练中引入合成困难负样本，以极小的改动获得了涌现的语义分类能力（最高提升11.3%）并大幅节省计算资源（ViT-B超越ViT-L的V-JEPA），证明对比学习仍是生成式与自蒸馏方法的强大替代方案。

    

    我们提出了ViTAMINS，这是一种将合成困难负样本融入无监督视觉Transformer预训练以提升表示质量的方法。我们的方法在ImageNet以及迁移学习、图像检索、复制检测和图像/视频分割等任务上进行了全面的基准测试。值得注意的是，我们提出的负样本带来了涌现特性：学到的表示包含图像语义内容的显式信息，并可充当优秀的分类器（相比基线最高提升11.3%）。ViTAMINS通过对现有对比学习框架的简单修改实现了这些优势，在超越竞争方法的同时更加节省资源，例如我们的ViT-B模型超越了采用ViT-L的V-JEPA。我们的发现促使人们重新审视对比学习，将其视为主导的生成式和自蒸馏方法的一种更简单却强大的替代方案。

    arXiv:2609.01041v1 Announce Type: cross  Abstract: We introduce ViTAMINS, a method that integrates synthetic hard negatives into unsupervised vision transformer pretraining to improve representation quality. Our approach is thoroughly benchmarked on ImageNet and transfer learning, image retrieval, copy detection, and image, video segmentation tasks. Notably, our proposed negatives give rise to emergent properties, where learned representations contain explicit information about the semantic content of an image and serve as excellent classifiers (up to +11.3% over baselines). ViTAMINS achieves these benefits through simple modifications to existing contrastive frameworks and outperforms competing methods while being more resource efficient, e.g., our ViT-B surpasses V-JEPA with ViT-L. Our findings motivate reconsidering contrastive learning as a simpler yet powerful alternative to dominant generative and self-distillation approaches.
    
[^91]: 高风险机器学习系统的因果证据治理

    Causal Evidentiary Governance for High-Risk Machine Learning Systems

    [https://arxiv.org/abs/2609.01040](https://arxiv.org/abs/2609.01040)

    提出因果证据治理（CEG）框架，通过版本化因果图、因果伤害率指标和密码学绑定的决策证据包，为高风险机器学习系统实现因果归因与高效证据验证。

    

    部署于信贷、招聘和资源分配等领域的机器学习系统正日益受到欧盟《人工智能法案》和GDPR等政策的监管。当前的公平性治理实践依赖于观察性公平指标、事后可解释性和不可篡改的审计日志，但对因果归因和高效的证据验证支持有限。我们提出了因果证据治理（CEG）框架，在该框架中，受监管机构承诺使用一个版本化的有向无环图（DAG），将因果路径划分为允许和不允许的组。因果伤害率衡量可归因于不允许因果路径的预测变化。每个决策都附有经签名的决策证据包（DEP），通过密码学方式将预测与已发布DAG的摘要以及特定路径的归因绑定在一起。DEP摘要可以附加到Merkle树中，以实现对数级别的高效验证。

    arXiv:2609.01040v1 Announce Type: cross  Abstract: Machine learning systems deployed for credit, hiring, and resource distribution are increasingly subject to regulatory oversight from policies such as the EU AI Act and GDPR. Current fairness governance practices rely on observational fairness metrics, post-hoc explainability, and immutable audit logs, but provide limited support for causal attribution and efficient evidentiary verification. We introduce Causal Evidentiary Governance (CEG), a framework in which regulated institutions commit to a versioned directed acyclic graph (DAG) that partitions causal pathways into allowable and disallowed groups. The Causal Harm Rate measures prediction variation attributable to disallowed causal pathways. Each decision is accompanied by a signed Decision-Evidence Packet (DEP), cryptographically binding the prediction to a digest of the published DAG and path-specific attributions. DEP digests can be appended to a Merkle tree to enable logarithmi
    
[^92]: 数据驱动的角色条件化智能体用于A/B测试模拟

    Data-Driven Persona-Conditioned Agents for A/B Test Simulation

    [https://arxiv.org/abs/2609.01038](https://arxiv.org/abs/2609.01038)

    该论文提出利用基于真实用户匿名行为数据（活动模式、参与度信号、人口统计特征）构建的数据驱动角色来条件化LLM智能体，从而更忠实地模拟用户群体并预测A/B测试结果，替代昂贵耗时的真实用户实验。

    

    A/B测试是评估产品变更的黄金标准，但每次实验都需要真实用户流量、工程投入以及长达数周的测量时间。我们提出了一个模拟框架，使用基于大语言模型（LLM）的智能体来预测A/B测试结果，这些智能体以基于真实用户行为信号构建的数据驱动角色（persona）为条件。与以往依赖合成角色或基于规则角色的研究不同，我们的智能体由匿名化的行为数据构建——包括活动模式、参与度信号和推断的人口统计特征——从而实现更忠实的人群建模。我们将A/B测试模拟构建为一个结构化问题任务，并系统性地研究了：(i) 问题设计格式；(ii) 角色数据来源和领域对齐的影响；(iii) 单个角色行为深度与人群多样性之间的权衡；(iv) 高效的人群子采样方法。在一个涵盖两种指标类型的40个A/B测试基准上，我们的最佳配置取得了……

    arXiv:2609.01038v1 Announce Type: new  Abstract: A/B testing is the gold standard for evaluating product changes, but each experiment requires real user traffic, engineering effort, and weeks of measurement. We propose a simulation framework that predicts A/B test outcomes using LLM-powered agents conditioned on data-driven personas grounded in real user behavioral signals. Unlike prior work that relies on synthetic or rule-based personas, our agents are constructed from anonymized behavioral data-activity patterns, engagement signals, and inferred demographics-enabling more faithful population modeling. We frame A/B test simulation as a structured question task and systematically study (i) question design formats, (ii) the impact of persona data source and domain alignment, (iii) the trade-off between per-persona behavioral depth and population diversity, and (iv) efficient population subsampling. On a benchmark of 40 A/B tests spanning two metric types, our best configuration achieve
    
[^93]: 自由派生，节制行动：面向递归LLM智能体树的渐进式风险授予

    Spawn Freely, Act Sparingly: Progressive Risk Vesting for Recursive LLM-Agent Trees

    [https://arxiv.org/abs/2609.01035](https://arxiv.org/abs/2609.01035)

    提出渐进式风险授予（PRV）机制，通过托管轨迹级风险预算并在分支激活时逐步扣减，为递归LLM智能体树中不可逆行动的授权证明了任意时刻的危害上界，实现“自由派生、节制行动”的安全权衡。

    

    递归LLM智能体可以通过派生专门智能体来拓展其搜索范围。某些分支随后会请求用于发送数据或部署代码的工具。那么何时应当授予某个分支行动权限？我们区分了沙箱派生（sandbox spawning，即通过外部控制防止特定危害的发生）与能力激活（capability activation，即被选中的分支跨越不可逆行动的边界）。渐进式风险授予（Progressive Risk Vesting, PRV）将一个轨迹级的风险预算进行托管，并在分支被激活时逐步扣减该预算。我们为自适应生成的树证明了任意时刻的危害上界。分支结果之间可能存在依赖关系，但每个局部证书必须在完整的激活前历史（包括用于选择该请求的信息）条件下保持有效。当激活门槛、分支收费和计算约束保持固定时，延迟授予能够保留不可撤销派生收费机制下所有可用的策略。边际风险估计在分支选择之后仍可能失效……

    arXiv:2609.01035v1 Announce Type: new  Abstract: Recursive LLM agents can broaden their search by spawning specialists. Some branches later request tools that send data or deploy code. When should a branch receive authority to act? We distinguish sandbox spawning, in which external controls prevent the specified harm, from capability activation, in which a selected branch crosses an irreversible-action boundary. Progressive Risk Vesting (PRV) holds a trajectory-level risk budget in escrow and debits it as branches are activated. We prove an anytime harm bound for adaptively generated trees. Branch outcomes may be dependent, but each local certificate needs to remain valid conditional on the full pre-activation history, including the information used to select the request. When activation gates, branch charges, and compute constraints are held fixed, delayed vesting preserves every policy available under irrevocable spawn charging. Marginal risk estimates can still fail after branch sel
    
[^94]: 度量区间时序逻辑的合成研究

    On Synthesis of Metric Interval Temporal Logics

    [https://arxiv.org/abs/2609.01032](https://arxiv.org/abs/2609.01032)

    本文提出了首个面向度量区间时序逻辑（MITL）的精确被动学习框架，通过将正负轨迹间的定量时间差异转化为新的布尔原子命题，把时间学习问题归约为可扩展的非时间LTL学习问题，从而无需预定义模板或受限逻辑片段即可自动挖掘时间规范。

    

    形式规范的自动挖掘对于验证实时系统至关重要。然而，现有的被动学习方法仍局限于确定性规范或时间正则表达式（TRE）的受限片段。据我们所知，本文提出了首个框架，能够针对表达力强的时间逻辑——度量区间时序逻辑（MITL）——实现精确的被动学习，而无需依赖预定义模板或受限的逻辑片段。我们的方法将时间学习问题形式化地归约为一个可扩展的非时间学习问题：通过识别正例轨迹与反例轨迹之间的定量时间差异，我们合成出精确的时间约束，并将其作为新的布尔原子命题注入。这将时间信息嵌入到字母表中，从而把复杂的公式求值任务委托给高度优化、可直接使用的现成非时间LTL工具。至关重要的是，我们的框架是完备的，能够保证得到一个区分性……（原文摘要在此处截断）

    arXiv:2609.01032v1 Announce Type: cross  Abstract: Automated mining of formal specifications is vital for verifying real-time systems. However, existing passive learning approaches remain restricted to deterministic specifications or limited fragments of Timed Regular Expressions (TRE). To our knowledge, this paper presents the first framework to tackle \emph{precise} passive learning for an expressive timed logic, \emph{Metric Interval Temporal Logic} (MITL) without relying on predefined templates or restricted logic fragments.   Our approach formally reduces the timed learning problem into a scalable untimed one. By identifying quantitative timing differences between positive and negative traces, we synthesise precise timed constraints and inject them as new Boolean atomic propositions. This embeds timing into the alphabet, delegating the complex formula evaluation to highly optimised, off-the-shelf untimed LTL tools.   Crucially, our framework is complete, guaranteeing a separating 
    
[^95]: 网络科学视角下评估深度图生成模型

    A Network Science Perspective on Evaluating Deep Graph Generative Models

    [https://arxiv.org/abs/2609.01015](https://arxiv.org/abs/2609.01015)

    该研究从网络科学视角评估深度图生成模型，通过生成网络与真实网络的拓扑相似性及其在识别有效节点免疫策略以抑制流行病/错误信息传播方面的实用性进行综合衡量。

    

    传统网络科学中的网络模型，如Erdős-Rényi模型和配置模型，生成的随机网络仅能再现真实世界网络中少数被选定的拓扑性质。深度图生成模型作为一种数据驱动的方法应运而生，它利用深度神经网络架构直接从真实世界网络中学习复杂的结构分布，从而生成更逼真的合成网络。由于隐私风险，真实的社会接触网络无法被共享，合成网络因此成为开发和评估疫情缓解策略的替代方案。在这项工作中，我们从网络科学的视角评估深度图生成模型以及配置模型，通过评估生成网络与真实世界网络之间的拓扑相似性，以及它们在识别有效节点免疫策略以抑制流行病/错误信息传播方面的实用性来进行衡量。

    arXiv:2609.01015v1 Announce Type: cross  Abstract: Traditional network models from network science, such as the Erdos-Renyi and configuration models, generate random networks that reproduce few selected topological properties observed in real-world networks. Deep graph generative models emerge as a data-driven approach, leveraging deep neural network architectures to learn complex structural distributions directly from real-world networks to generate more realistic synthetic networks. Because real social contact networks cannot be shared due to privacy risks, synthetic networks serve as an alternative for developing and evaluating epidemic mitigation strategies. In this work, we evaluate deep graph generative models as well as the configuration from a network science perspective by assessing both the topological similarity between generated and real-world networks and their utility in identifying effective node immunization strategies to sup- press epidemic/misinformation spreading. It
    
[^96]: 图形即程序：可编辑科学图形的递归生成

    Figures as Programs: Recursive Generation of Editable Scientific Figures

    [https://arxiv.org/abs/2609.01006](https://arxiv.org/abs/2609.01006)

    该论文提出了FigTree多智能体系统，将科学图形生成形式化为递归的SVG程序构建，通过将图形分解为区域层次结构并逐个生成SVG程序片段，实现了从科学论文自动生成结构化、可精确编辑的矢量图形。

    

    科学方法图对于清晰地传达复杂方法至关重要，然而创建这些图形仍然非常耗费人力，通常需要多轮的反复修改。近期的图像生成模型虽然可以合成视觉上吸引人的光栅图形，但要在单一生成步骤中获得令人满意的结果仍然十分困难。此外，无论是对人类还是模型而言，对光栅图形进行精确编辑都极具挑战性。我们将科学图形生成问题形式化为递归的SVG程序构建，并提出了FigTree——一个多智能体系统，能够自动将科学论文转换为结构化的矢量图形。FigTree将图形内容锚定于源论文本身，将图形分解为局部区域的层次结构，将每个区域生成为简短的SVG程序，并组装生成各个片段。一个渲染-评论家细化循环会共同检查渲染后的图形及其底层的

    arXiv:2609.01006v1 Announce Type: new  Abstract: Scientific methodology figures are essential for communicating complex methods clearly, yet creating them remains labor-intensive and typically requires multiple rounds of refinement. Recent image-generation models can synthesize visually appealing raster figures, but producing a human-satisfactory result in a single generation step remains difficult. Moreover, precise edits to raster figures are challenging for both humans and models. We formulate scientific figure generation as recursive SVG program construction and propose \textsc{FigTree}, a \textit{multi-agent} system that automatically transforms a scientific paper into a structured vector figure. \textsc{FigTree} grounds figure content in the source paper, decomposes a figure into a hierarchy of local regions, generates each region as a short SVG program, and assembles the resulting fragments. A render-critic refinement loop jointly inspects the rendered figure and its underlying 
    
[^97]: SinkPruner：面向多模态大语言模型的无Sink视觉token剪枝方法

    SinkPruner: Sink-Free Visual Token Pruning for Multimodal Large Language Models

    [https://arxiv.org/abs/2609.01004](https://arxiv.org/abs/2609.01004)

    提出无需训练的视觉token剪枝框架SinkPruner，通过过滤高度冗余的高范数离群token并缓解注意力汇聚现象，在保持多模态理解能力的同时实现高效的多模态大语言模型推理。

    

    尽管多模态大语言模型（MLLM）具有强大的多模态理解能力，但其在处理长视觉token序列时会产生巨大的计算开销。为降低推理成本，近期研究探索了基于视觉中心策略或文本引导策略的视觉token剪枝方法。然而，这些方法往往忽视了高范数离群token（即特征范数异常大的token），导致次优的剪枝决策。在本工作中，我们证明这类高范数离群token在特征维度和空间维度上都高度冗余，但现有方法却常常错误地将其作为信息线索而保留。受此观察启发，我们提出了SinkPruner，一个无需训练的视觉token剪枝框架，用于实现高效的MLLM推理。SinkPruner遵循由粗到细的设计，包含两个关键模块：一个用于过滤高范数冗余并缓解注意力汇聚（attention sink）现象的视觉净化器……

    arXiv:2609.01004v1 Announce Type: cross  Abstract: Despite their strong multimodal understanding ability, multimodal large language models (MLLMs) incur substantial computational overhead when processing long visual token sequences. To reduce inference costs, recent studies have explored visual token pruning through vision-centric or text-guided strategies. However, these methods often overlook high-norm outlier tokens, i.e., tokens with abnormally large feature norms, leading to suboptimal pruning decisions. In this work, we show that such high-norm outlier tokens are highly redundant in both feature and spatial dimensions, yet are often mistakenly preserved as informative cues by existing methods.   Motivated by this observation, we propose SinkPruner, a training-free visual token pruning framework for efficient MLLM inference. SinkPruner follows a coarse-to-fine design with two key modules: a visual sanitizer that filters high-norm redundancies and alleviates attention sink and atte
    
[^98]: 正确的框架，错误的规则：文化线索暴露了它们本意想弥补的金融知识差距

    Right Frame, Wrong Rule: Cultural Cues Expose the Financial Knowledge Gap They Were Meant to Close

    [https://arxiv.org/abs/2609.00999](https://arxiv.org/abs/2609.00999)

    该论文提出“规范多元性”这一新评估设定，通过将框架选择与框架内正确性分离，揭示了“刻板印象陷阱”——文化线索虽能引导大模型选择伊斯兰金融框架，却在框架内暴露出高达57%至66%的错误率，表明传统二选一评估会严重高估模型的文化对齐能力。

    

    当一个问题在不同规范框架下都有有效答案时，语言模型必须决定采用哪个框架，以及它能否在该框架内正确作答。我们将这种情境称为“规范多元性”，并以伊斯兰金融为研究对象，采用一种将框架选择与框架内正确性区分开来的四选一分类法进行研究。这种区分揭示了“刻板印象陷阱”：文化线索引导模型走向某一框架，但模型却在该框架内选择了错误的答案。在十二个模型、两种语言和五十个人口统计信号的测试中，文化线索会改变模型的框架选择，并暴露出显著的准确率差异，尤其是在非前沿模型中。在最强信号的作用下，大型开源权重模型有97%的概率选择伊斯兰金融框架。若采用二选一的评估方式，将会报告近乎完美的对齐度，尽管其中57%至66%的选择实际上是错误的。这些发现为……提供了依据，但并未……（原文摘要在此处截断）

    arXiv:2609.00999v1 Announce Type: cross  Abstract: When a question has valid answers under different normative frameworks, a language model must decide which framework to use and whether it can answer correctly within it. We call this setting normative pluralism and study it in Islamic finance using a four-choice taxonomy that separates framework selection from within-framework correctness. This separation reveals the stereotype trap: a cultural cue steers a model toward one framework, but the model selects an incorrect answer within that framework. Across twelve models, two languages, and fifty demographic signals, cultural cues change framework selection and reveal substantial differences in accuracy, especially among non-frontier models. Under the strongest signal, large open-weight models select the Islamic framework 97% of the time. A two-choice evaluation would report near-perfect alignment, although 57--66% of those selections are incorrect. These findings motivate, but do not d
    
[^99]: Inspicio：面向历史语言的开放式词汇、基于大语言模型的词义检索

    Inspicio: Open-Vocabulary, LLM-Based Sense Retrieval for Historical Languages

    [https://arxiv.org/abs/2609.00998](https://arxiv.org/abs/2609.00998)

    提出了Inspicio，一个无需源语言词义清单的开放式词汇检索流水线，利用大语言模型生成的英文翻译、候选定义和词元，通过混合检索将历史语言文本中的词元直接链接到开放英语WordNet的同义词集。

    

    词义消歧在英语及少数资源丰富的现代语言中进展迅速，但它始终假设源语言中存在词义清单（sense inventory）以及词到词义的映射关系（Navigli, 2026）。对于大多数历史语言和低资源语言而言，这些假设不再成立，因为它们专用的WordNet要么不完整，要么仍在构建中。我们提出了Inspicio，这是一个开放词汇的检索流水线，能够将上下文中的词元直接链接到开放英语WordNet（McCrae et al., 2020）的同义词集（synset），而无需任何源语言的词义清单或映射。对于每个词的出现实例，一个经过指令微调的大语言模型会生成两句周围句子的英文翻译、一小组候选的词典式定义以及若干候选英文词元。这些输出驱动一个混合检索步骤，该步骤结合了稠密的定义-同义词集相似度、稀疏的词元匹配以及最大边际相关性

    arXiv:2609.00998v1 Announce Type: cross  Abstract: Word Sense Disambiguation has advanced rapidly for English and a handful of well-resourced modern languages, but it continues to assume the existence of a sense inventory and a word-to-sense mapping in the source language (Navigli, 2026). These assumptions break down for most historical and low-resource languages, whose dedicated WordNets are either incomplete or still under construction. We present Inspicio, an open-vocabulary retrieval pipeline that links tokens in context to synsets of the Open English WordNet (McCrae et al., 2020) without requiring any source-language inventory or mapping. For each occurrence, an instruction-tuned LLM produces two English translations of the surrounding sentence, a small set of candidate dictionary-style definitions, and a few candidate English lemmas. These outputs drive a hybrid retrieval step that combines dense definition-synset similarity, sparse lemma matching, and Maximal Marginal Relevance 
    
[^100]: 基于属性的音乐匹配中人类与计算机判断的一致性研究

    On the Human and Computer Alignment of Attribute-Based Music Matches

    [https://arxiv.org/abs/2609.00987](https://arxiv.org/abs/2609.00987)

    该论文通过包含剽窃案例、翻唱歌曲和AI生成音乐的感知实验，构建了MATCHA数据集，系统研究了计算相似度度量与人类判断在旋律、和声、节奏、人声和音色五种音乐属性上的一致性。

    

    生成式人工智能的最新进展引发了对生成内容原创性以及训练数据潜在复制问题的伦理担忧，并对透明度、内容归属和知识产权产生了进一步影响。在音乐领域，已有多种计算方法被提出，利用基于音频的相似度度量来识别潜在的复制行为。然而，这些方法在不同音乐属性上与人类判断的一致性仍未得到充分探索。为了填补这一空白，我们对“音乐匹配”（即高度相似的音乐片段）开展了感知实验。我们聚焦于五种音乐属性：旋律、和声、节奏、人声和音色。我们设计了一个基于三元组的强制选择任务，包含300个案例，涵盖剽窃实例、翻唱歌曲和AI生成的音乐。通过该实验，我们提出了MATCHA（基于音乐属性的三元组比较与人类标注）数据集，这是一个……的集合。

    arXiv:2609.00987v1 Announce Type: cross  Abstract: Recent advances in generative AI are raising ethical concerns regarding the originality of generated content and the potential replication of training data, with further implications for transparency, attribution, and intellectual property. In music, several computational approaches have been proposed to identify potential replication, using audio-based similarity metrics. Yet, their alignment with human judgments across distinct musical attributes remains underexplored. To address this gap, we conduct a perceptual experiment on music matches, defined as strongly similar musical excerpts. We focus on five musical attributes: melody, harmony, rhythm, voice, and timbre. We design a triplet-based forced-choice task comprising 300 cases, including plagiarism examples, cover songs, and AI-generated music. From this experiment, we introduce the MATCHA (Musical Attribute-based Triplet Comparison with Human Annotations) dataset: a collection o
    
[^101]: 基于形态学保持与组织病理学真实性约束的半监督虚拟染色

    Semi-Supervised Virtual Staining via Morphology Preservation and Histopathological Realism Constraints

    [https://arxiv.org/abs/2609.00984](https://arxiv.org/abs/2609.00984)

    该论文提出一种半监督虚拟染色框架，利用Hessian形态学保持机制和组织病理学真实性约束，充分利用有限的配对数据与大量非配对源图像实现稳定的虚拟染色，从而摆脱对严格配准训练数据的依赖。

    

    虚拟染色旨在通过计算方法生成目标染色的组织病理学图像，同时降低传统染色流程所需的成本和时间。然而，现有方法主要依赖严格配对且精确配准的训练数据，而这些数据在日常实践中难以获取且成本高昂。为了减少这种依赖，我们提出了一种稳定的半监督虚拟染色框架，联合利用有限的配对数据和大量的非配对源图像。直接引入非配对图像具有挑战性，因为其生成结果缺乏相应的目标用于监督，可能导致不真实的染色效果、形态学退化，甚至训练崩溃。为了从这些图像中获得可靠的监督，基于Hessian（海森矩阵）的形态学保持方法从每个源图像中提取结构线索，并约束生成的输出以保留组织形态。

    arXiv:2609.00984v1 Announce Type: cross  Abstract: Virtual staining aims to computationally generate target-stained histopathological images while reducing the cost and time associated with conventional staining procedures. However, existing methods rely predominantly on strictly paired and accurately registered training data, which are difficult and expensive to obtain in routine practice. To reduce this dependence, we propose a stable semi-supervised virtual staining framework that jointly exploits both limited paired data and abundant unpaired source images. Directly incorporating unpaired images is challenging because their generated results lack corresponding targets for supervision, potentially leading to unrealistic staining, morphological degradation, or even training collapse. To obtain reliable supervision from these images, Hessian-derived morphology preservation extracts structural cues from each source image and constrains the generated output to retain tissue morphology. 
    
[^102]: 面向陪伴型智能体评估的披露门控用户模拟

    Disclosure-Gated User Simulation for Companion-Agent Evaluation

    [https://arxiv.org/abs/2609.00982](https://arxiv.org/abs/2609.00982)

    提出披露门控用户模拟方法，让模拟用户根据陪伴型智能体的行为决定信息披露深度，以纠正模拟用户过度配合、使被测系统仅靠提问数量即可得分的评估缺陷。

    

    使用大型语言模型扮演用户如今已成为可扩展评估的标准做法。但它存在一个反复被诊断出的缺陷：模拟用户过度配合，导致被测系统可以仅凭大量提问来得分，而不是通过让用户愿意开口说话来得分。作为回应，我们提出了一种披露门控，将信息披露与陪伴型智能体的行为相挂钩：其状态是一个由五个有序门控构成的阶梯，并归并为三个可观测的深度层。我们对该机制进行了规范定义、消融实验和审计，并依据该规范训练了一个用户模拟器。门控行为从训练语料的合成分支中学习，而真实分支则提供人们真实的说话与反应方式；训练完成后，模拟器在运行时无需被告知每条信息位于哪个门控之后。该门控是环境中的承重组件：在一个已发布的陪伴型智能体基准测试的英文语料库（CompanionBench）上，一旦训练……

    arXiv:2609.00982v1 Announce Type: cross  Abstract: Using a large language model to play the user is now standard in scalable evaluation. It has a repeatedly diagnosed failure: the simulated user is excessively cooperative, so a system under test can score by the sheer number of questions it asks rather than by making the user willing to speak. We answer with a disclosure gate conditioning information release on the companion agent's behaviour: its state is a ladder of five ordered gates, merged onto three observable depth layers. We specify, ablate, and audit it, and train a user simulator against that specification. Gating behaviour is learned from the training corpus's synthetic branch, while the real branch supplies how people speak and react; after training, the simulator need not be told at runtime which gate each item sits behind. The gate is a load-bearing component of the environment: on the English corpus of a published companion-agent benchmark (CompanionBench), once training
    
[^103]: zbMATH开放知识图谱：追溯数百年数学研究历程

    The zbMATH Open Knowledge Graph: Tracing Centuries of Mathematical Research

    [https://arxiv.org/abs/2609.00969](https://arxiv.org/abs/2609.00969)

    zbMATH开放知识图谱是一个涵盖250多年数学研究、包含3400万实体和1.68亿RDF三元组的大规模知识图谱，其创新之处在于整合了专家策划的语义内容（评论、关键词、学科分类、消歧作者信息等），突破了传统学术知识图谱仅记录书目元数据和引用结构的局限，支持对数学概念与学术关系演变的细粒度时序分析。

    

    我们介绍了zbMATH开放知识图谱，这是一个涵盖250多年数学学术研究的大规模RDF知识图谱（KG）。与现有主要记录书目元数据和引用结构的学术知识图谱不同，zbMATH开放知识图谱整合了专家策划的语义内容，包括评论、关键词、学科分类、软件引用以及消歧后的作者身份信息。这种数学知识的领域特定表示与广泛的时间覆盖相结合，支持需要对数学概念、研究领域和学术关系随时间演变进行细粒度探索的分析。该图谱包含3400万个实体和1.68亿个RDF三元组，采用成熟的语义网词表进行表示，支持互操作性和FAIR数据原则。我们进一步通过查询驱动的、基于历史的学术研究展示了其能力。

    arXiv:2609.00969v1 Announce Type: cross  Abstract: We present the zbMATH Open Knowledge Graph, a large-scale RDF knowledge graph (KG) covering more than 250 years of mathematical scholarship. Unlike existing scholarly knowledge graphs that primarily capture bibliographic metadata and citation structures, the zbMATH Open KG integrates expert-curated semantic content, including reviews, keywords, subject classifications, software references, and disambiguated authorship. This combination of domain-specific representation of mathematical knowledge and extensive temporal coverage supports analyses that require fine-grained exploration of mathematical concepts, research fields, and scholarly relationships over time. The resulting graph comprises 34 million entities and 168 million RDF triples represented using established Semantic Web vocabularies, supporting interoperability and FAIR data principles. We further demonstrate its capabilities through query-driven historically grounded scholar
    
[^104]: CoBRA：通过反事实边际学习工具使用边界

    CoBRA: Learning Tool-Use Boundaries via Counterfactual Margins

    [https://arxiv.org/abs/2609.00967](https://arxiv.org/abs/2609.00967)

    提出了CoBRA框架，通过从同一基础模型构建内部/外部专家并计算“使用与不使用工具”两种回答间的反事实奖励边际，来学习每个查询是否应调用工具的实例级边界，从而在避免不必要工具调用的同时不遗漏真正需要工具的场景。

    

    随着大语言模型越来越多地通过外部工具执行任务，决定何时调用工具已成为与决定如何使用工具同等重要的核心问题。不必要的工具调用会带来延迟、成本、检索噪声和错误传播，而漏掉必要的调用则会损害知识密集型查询或需要最新证据的问题的回答效果。现有方法通常基于绝对的查询或生成信号（如难度、置信度或最终任务奖励）来触发工具，因此缺乏对工具使用在实例层面边际收益的显式估计。我们提出了CoBRA，一个面向工具增强语言模型的反事实边界学习框架。CoBRA首先从同一基础模型构建内部专家和外部专家，收集配对的轨迹，并估计使用工具与不使用工具回答之间的奖励边际。该边际将数据划分为偏好内部、偏好外部和模糊三类案例。（注：原摘要在此处截断，后续内容不完整）

    arXiv:2609.00967v1 Announce Type: new  Abstract: As large language models increasingly act through external tools, deciding when to call a tool has become a central problem alongside deciding how to use it. Unnecessary tool calls introduce latency, cost, retrieval noise, and error propagation, while missed calls hurt knowledge-intensive queries or questions requiring up-to-date evidence. Existing methods typically trigger tools from absolute query or generation signals, such as difficulty, confidence, or final task reward, and therefore lack an explicit estimate of the instance-level marginal benefit of tool use. We propose CoBRA, a counterfactual boundary-learning framework for tool-augmented language models. CoBRA first constructs internal and external experts from the same base model, collects paired trajectories, and estimates the reward margin between answering with and without tools. This margin partitions data into internal-favored, external-favored, and ambiguous cases. CoBRA t
    
[^105]: 基于协方差修正马氏距离的少样本域外意图检测

    Few-Shot Out of Domain Intent Detection with Covariance Corrected Mahalanobis Distance

    [https://arxiv.org/abs/2609.00961](https://arxiv.org/abs/2609.00961)

    本文分析了马氏距离方法在少样本设置下域外意图检测性能不佳的原因，并提出了一种协方差修正的马氏距离来有效检测域外意图。

    

    聊天机器人和语音助手等对话智能体经过训练以理解和响应用户意图。当遇到与其训练过的意图不同的语句时，这些智能体应将该意图分类为“未知”或“域外”。这个问题被称为域外（OOD）意图检测。Podolskiy等人（2021）证明了马氏距离可以有效地用于识别OOD意图，且优于其他竞争方法。然而，在实际应用中十分重要的少样本设置下，他们的方法未能超越基线方法。在本文中，我们分析了其性能不佳的原因，并提出了一种用于检测域外意图的协方差修正马氏距离。

    arXiv:2609.00961v1 Announce Type: new  Abstract: Conversational agents like chatbots and voice assistants are trained to understand and respond to user intents. On encountering an utterance with an intent different from the ones they have been trained on, these agents are expected to classify the intent as `unknown' or `out of domain'. This problem is known as out of domain (OOD) intent detection. Podolskiy et al. (2021), showed that Mahalanobis distance can be used effectively for identifying OOD intents, outperforming competing approaches. However, their method fails to outperform the baselines in the practically important few-shot setting. In this paper we analyze the reason for low performance and propose a covariance corrected Mahalanobis distance for detecting out-of-domain intents.
    
[^106]: 校准是瓶颈：多轮工具调用的动作类别诊断

    Calibration is the Bottleneck: An Action-Class Diagnostic of Multi-Turn Tool-Calling

    [https://arxiv.org/abs/2609.00949](https://arxiv.org/abs/2609.00949)

    本文提出一个基于四类动作空间的诊断框架，通过引入“准确率不超过黄金动作召回率”的自揭示上界，将多轮工具调用失败分解为动作类别失准与动作执行失败两种正交模式，从而揭示开源模型总体准确率追平闭源模型的表象背后，动作类别校准才是真正的瓶颈。

    

    多轮工具调用是大语言模型（LLM）智能体的一项核心评测场景。在公开的工具调用基准上，开源权重模型的总体准确率已接近甚至超越闭源前沿模型。然而，这一指标是对众多不同多轮情境的取平均，掩盖了进展是否在这些情境之间均衡分布。我们提出一种面向动作类别的诊断框架，将多轮失败分解为两种正交模式：动作类别失准与动作执行失败。该框架在四类动作空间（TOOL_CALL/ASK/REFUSE/CONFIRM）上运行，并引入一个自我揭示的上界 Acc ≤ GAR（黄金动作召回率）；两种失败模式分别表现为上界被违反（Acc > GAR，暴露出状态评分器对失准的掩盖）以及较大的上界余量（GAR >> Acc，将执行失败定位于 TOOL_CALL 内部）。我们在一组工具调用模型上对该框架进行了验证……（原文摘要在此处截断）

    arXiv:2609.00949v1 Announce Type: cross  Abstract: Multi-turn tool calling is a core evaluation scenario for large language model (LLM) agents. On public tool-calling benchmarks, open-weight models now approach or even surpass closed-source frontier models in aggregate accuracy. However, this metric averages over many different multi-turn situations and obscures whether progress is balanced across them. We propose an action-class-oriented diagnostic framework that decomposes multi-turn failures into two orthogonal modes: action-class miscalibration and action-execution failure. The framework operates over a four-class action space (TOOL_CALL/ASK/REFUSE/CONFIRM) and introduces a self-revealing upper bound Acc <= GAR (Gold Action Recall); the two modes show up as bound violation (Acc > GAR, exposing state-grader masking of miscalibration) and large bound slack (GAR >> Acc, localizing execution failure within TOOL_CALL). We validate it on a panel of tool-calling models across multiple mul
    
[^107]: 从术语到图示：面向科学图表理解的视觉指令生成

    From Terminology to Diagrams: Visual-Instruction Generation for Scientific Diagram Understanding

    [https://arxiv.org/abs/2609.00948](https://arxiv.org/abs/2609.00948)

    该论文提出SciGram框架与数据集，通过科学课程术语自动生成涵盖19.4万张图表和140万条视觉指令的大规模训练数据，显著提升了视觉语言模型在科学图表理解任务上的表现。

    

    视觉语言模型（VLMs）在自然图像的视觉问答任务中展现出强大的性能。然而，它们在处理科学图表时仍然存在困难，因为科学图表旨在传达功能性或关系性含义，而非字面上的场景。因此，我们提出了一个通过利用源自科学课程的术语来生成大规模基于图表的指令数据的框架。我们的方法系统地提取领域概念、合成原子事实、从网络检索相关图表，并以图表说明和选择题的形式生成多模态监督信号。利用这一流程，我们构建了SciGram数据集，其中包含超过19.4万张图表和140万条视觉指令，涵盖生命科学、地球科学和物理科学。尽管依赖于带有噪声的网络数据和合成标注，在SciGram上微调的模型在以图表为中心的基准测试中仍取得了显著的性能提升。

    arXiv:2609.00948v1 Announce Type: cross  Abstract: Vision-language models (VLMs) have demonstrated strong performance in visual question answering with natural images. However, they continue to struggle with scientific diagrams, which are designed to convey functional or relational meaning rather than literal scenes. We therefore introduce a framework for generating large-scale diagram-grounded instruction data by leveraging terminology derived from scientific curricula. Our approach systematically extracts domain concepts, synthesizes atomic facts, retrieves relevant diagrams from the web, and generates multimodal supervision in the form of diagram captions and multiple-choice questions. Using this pipeline, we construct SciGram, a dataset of over 194K diagrams and 1.4M visual instructions across life, earth, and physical sciences. Despite relying on noisy web data and synthetic annotations, models fine-tuned on SciGram achieve substantial improvements on diagram-centric benchmarks, i
    
[^108]: 面向大语言模型生成文本的嵌入式条件独立性检验及其在德国联邦议院演讲中的应用

    Embedded Conditional Independence Tests for Large Language Model Generated Text with an Application to German Parliament Speeches

    [https://arxiv.org/abs/2609.00946](https://arxiv.org/abs/2609.00946)

    本文提出嵌入式条件独立性检验（eCITs），通过将LLM生成的文本及其源文本嵌入到表示空间后再进行条件独立性检验，从而判断模型输出是否携带源文本之外的额外信息，并将其应用于德国议会演讲数据的分析。

    

    条件独立性检验（CITs）用于检验在给定第三个随机对象 Z 的条件下，两个随机对象 X 和 Y 之间是否存在条件依赖关系。现有的 CITs 对高维数据的适用性有限，尤其是像文本这样的多模态数据。然而，我们表明此类检验对大语言模型（LLM）的输出具有重要意义：即检验从源文本 Z 生成的输出 X 是否携带超出 Z 本身所含信息之外的属性 Y 的信息。为此，我们提出了嵌入式条件独立性检验（eCITs），该方法对 X 和 Z 进行嵌入，并将现有的 CIT 应用于所得的表示以及 Y。我们证明，只要 Z 的嵌入是充分的，即保留了 Z 所携带的关于 Y 或 X 的表示的信息，原假设就会从 X 和 Z 转移到它们的表示上，因此对嵌入后假设有效的 CIT 对原始假设同样有效。我们进一步给出了等变性的相关条件……

    arXiv:2609.00946v1 Announce Type: cross  Abstract: Conditional independence tests (CITs) test for conditional dependence between two random objects $X$ and $Y$ given a third random object $Z$. Existing CITs have limited applicability to high-dimensional data, especially multimodal data like text. However, we show that such tests are of interest for large language model (LLM) outputs, where we test whether an output $X$ generated from a source text $Z$ carries information about an attribute $Y$ beyond $Z$ itself. For this purpose, we propose embedded CITs (eCITs), which embed $X$ and $Z$ and apply an existing CIT to the resulting representations and to $Y$. We show that, provided the embedding of $Z$ is sufficient, i.e. retains the information $Z$ carries about either $Y$ or the representation of $X$, the null hypothesis transfers from $X$ and $Z$ to their representations, so that a CIT valid for the embedded hypothesis is valid for the original one. We further give conditions for equiv
    
[^109]: DualStake：深度研究智能体中的双路径置信度校准

    DualStake: Dual-Path Confidence Calibration in Deep Research Agents

    [https://arxiv.org/abs/2609.00935](https://arxiv.org/abs/2609.00935)

    提出DualStake双路径置信度校准方法，通过在每次检索后引出证据置信度并在答案生成后引出答案置信度，利用边界裁剪的置信度相关stake奖励将两者与答案正确性联合对齐，有效缓解深度研究智能体的严重过度自信问题。

    

    深度研究智能体通过多轮检索和面向决策的生成来解决知识密集型任务。然而，这类智能体存在严重的过度自信问题，导致其表达的置信度对于用户信任和下游弃答决策而言并不可靠。为解决这一问题，我们在深度研究流程的每次检索之后增加了步骤置信度引出环节，并以常用的答案后言语化置信度为基础。有趣的是，我们发现证据置信度——在最后一次检索步骤后引出的置信度——比答案置信度——在答案生成后引出的置信度——能提供更强的不确定性信号，且答案置信度在很大程度上受到证据置信度的塑造。基于这些发现，我们提出了DualStake，一种双路径校准方法，通过施加边界裁剪的、置信度相关的stake奖励，将证据置信度和答案置信度与答案正确性联合对齐，同时抑制对极端置信度的过度优化。实验……

    arXiv:2609.00935v1 Announce Type: cross  Abstract: Deep Research agents tackle knowledge-intensive tasks through multi-round retrieval and decision-oriented generation. However, these agents suffer from severe overconfidence, making their expressed confidence unreliable for user trust and downstream abstention. To address this, we augment the Deep Research pipeline with step confidence elicitation after each retrieval, building on the commonly used post-answer verbalized confidence. Interestingly, we find that Evidence Confidence (E-Conf), elicited after the final retrieval step, provides a stronger uncertainty signal than Answer Confidence (A-Conf), elicited after answer generation, and that A-Conf is largely shaped by E-Conf. Based on these findings, we propose DualStake, a dual-path calibration method that applies margin-clipped, confidence-dependent stake rewards to jointly align E-Conf and A-Conf with answer correctness while limiting extreme confidence optimization. Experiments o
    
[^110]: 上下文接地增益由既有机制介导：对GRPO、SFT和DPO的审计

    Context-Grounding Gains Are Mediated by Pre-existing Machinery: Auditing GRPO, SFT, and DPO

    [https://arxiv.org/abs/2609.00925](https://arxiv.org/abs/2609.00925)

    本文通过从同一检查点系统审计GRPO、SFT和DPO共九种后训练方案，发现语言模型遵循冲突提示证据的接地增益主要源于强化模型中已有的机制（与起始模型相同的因果注意力头集合），而非学习新机制，其中GRPO增益很小、冲突SFT提升适中、DPO在其匹配分布上接近上限。

    

    当提示中的证据与模型记忆中的知识冲突时，语言模型可能会忽略提示中的证据。后训练可以让模型更可靠地遵循这类证据，但这些增益究竟需要新的机制，还是通过强化已有的机制来实现，目前尚不清楚。我们从同一个起始检查点比较了涵盖GRPO、SFT和DPO的九种后训练方案，并将关键比较扩展到不同规模和不同模型家族。我们在训练之前从该起始检查点估计了一个“接地方向”。在测试的五种GRPO变体中，接地增益都很小。对于两种在不同随机种子下可复现的变体，等价性检验表明，即使被奖励的指标有所提升，它们的效果仍低于冲突SFT所带来的增益。冲突SFT适度地改善了接地能力，而DPO在其匹配的分布上使接地能力接近上限。冲突SFT和DPO在很大程度上使用与起始模型相同的因果注意力头集合。减去起始模型的方向会同时抑制两者……

    arXiv:2609.00925v1 Announce Type: cross  Abstract: Language models can ignore prompt evidence when it conflicts with memorized knowledge. Post-training can make models follow such evidence more reliably, but it is unclear whether these gains require new machinery or strengthen machinery already present. We compare nine post-training arms spanning GRPO, SFT, and DPO from one starting checkpoint, with key comparisons extended across scales and families. We estimate a grounding direction from that checkpoint before training. Across five tested GRPO variants, grounding gains are small. For the two variants replicated across seeds, equivalence tests bound their effects below the conflict-SFT gain even as the rewarded metric improves. Conflict-SFT improves grounding moderately, while DPO drives grounding near ceiling on its matched distribution. Conflict-SFT and DPO largely use the same causal attention-head set as the starting model. Subtracting the starting-model direction suppresses both 
    
[^111]: 超越图像平面：面向多目标跟踪的世界基准查询

    Beyond the Image Plane: World-Grounded Queries for Multi-Object Tracking

    [https://arxiv.org/abs/2609.00924](https://arxiv.org/abs/2609.00924)

    提出PLANET，一种通过将重建的3D场景几何嵌入查询特征与位置编码、从而超越图像平面的端到端多目标跟踪器，在三个基准测试中达到最先进性能。

    

    单目视频将3D场景记录为2D图像平面投影的序列，掩盖了深度和空间关系。多目标跟踪器主要依靠仅在图像平面中观察到的外观和几何信息来定位和关联对象，因而继承了这些歧义性。为了解决这一局限，我们提出了PLANET，一个旨在超越图像平面的端到端多目标跟踪器。作为一项基础性工作，我们将现有的2D跟踪数据集提升到3D。随后，我们通过将重建的3D场景几何嵌入查询形成过程中使用的特征和位置编码中，构建了世界基准查询。辅助的3D位置预测任务进一步促使查询在训练期间编码对象位置。一个互补的双分辨率时序记忆模块则在更长的时间间隔内保留这些证据。最终，PLANET在三个不同的基准测试中取得了最先进的性能。

    arXiv:2609.00924v1 Announce Type: cross  Abstract: Monocular videos record 3D scenes as sequences of 2D image-plane projections, obscuring depth and spatial relationships. Multi-object trackers localize and associate objects primarily using appearance and geometry observed only in the image plane, inheriting these ambiguities. To address this limitation, we introduce PLANET, an end-to-end multi-object tracker designed to move beyond the image plane. As an enabling step, we lift existing 2D tracking datasets into 3D. We then form world-grounded queries by embedding reconstructed 3D scene geometry into the features and positional encodings used during query formation. An auxiliary 3D location prediction task further encourages the queries to encode object positions during training. A complementary dual-resolution temporal memory preserves this evidence across longer temporal gaps. As a result, PLANET achieves state-of-the-art performance across three diverse benchmarks.
    
[^112]: VIBE-Bench：当用户画像不等于偏好时，评估个性化大语言模型

    VIBE-Bench: Evaluating Personalized Large Language Models When Profiles Don't Mean Preferences

    [https://arxiv.org/abs/2609.00921](https://arxiv.org/abs/2609.00921)

    该论文提出了VIBE-Bench基准，揭示当前个性化大语言模型在“画像-偏好概念错位”情形下（即用户画像线索与查询偏好处于不同概念空间时）因过度依赖浅层语义关联而失效，需要具备超越表面语义的跨概念偏好推理能力。

    

    个性化大语言模型（PLLMs）旨在为个体用户定制回复，其核心挑战在于偏好推理：从用户相关历史中推断与查询相关的偏好。然而，现有基准测试大多假设这种偏好可以从语义相关的历史中检索得到。我们研究了一个尚未被充分探索但具有重要实践意义的情形——画像-偏好概念错位（PRCM），即可观察的画像线索与特定查询的偏好处于不同的概念空间，使得语义检索无法可靠地支持个性化。我们提出了VIBE-Bench，这是一个包含两个基于心理学设计的任务、3,504个人设（persona）和12,239段对话的基准，其中包括一个经过人工验证的黄金测试集，并要求模型具备超越表面语义重叠的跨概念偏好推理能力。对多种个性化方法的实验表明，当前的PLLMs在很大程度上依赖于浅层语义关联，因而难以应对此类情形。

    arXiv:2609.00921v1 Announce Type: new  Abstract: Personalized Large Language Models (PLLMs) aim to tailor responses to individual users, where a central challenge is preference reasoning: inferring query-relevant preferences from user-related history. Existing benchmarks, however, largely assume that such preference can be retrieved from semantically related history. We study an underexplored but practically important regime, profile-preference conceptual misalignment (PRCM), where observable profile cues and query-specific preferences lie in different concept spaces, making semantic retrieval inconsistent for personalization. We introduce VIBE-Bench, a benchmark with two psychology-grounded tasks, 3,504 personas and 12,239 dialogues, including a manually verified gold test set, and requires cross-concept preference reasoning beyond surface semantic overlap. Experiments with several personalization methods show that current PLLMs largely rely on shallow semantic correlations and fail t
    
[^113]: RPCBench：面向基于大语言模型推荐中主动前提批判的基准测试

    RPCBench: A Benchmark for Proactive Premise Critique in LLM-based Recommendation

    [https://arxiv.org/abs/2609.00918](https://arxiv.org/abs/2609.00918)

    该论文提出了 RPCBench 基准，首次系统评估大语言模型在推荐场景中主动检测、诊断并妥善处理用户请求中错误前提的能力，涵盖五个推荐领域、十种前提失败类型，并提供了细粒度的评估框架。

    

    大语言模型越来越多地被用作交互式推荐助手。因此，对它们的评估应当超越生成看似合理的物品推荐，而是测试其能否识别有缺陷的推荐请求。现有的推荐系统基准主要评估排序、生成或偏好满足能力，而现有的错误检测基准通常不基于推荐场景特有的用户与候选证据。为了填补这一空白，我们提出了 RPCBench，这是一个用于评估“推荐器前提批判”能力的基准：即在自然语言推荐请求中检测、诊断并妥善处理错误前提的能力。RPCBench 包含来自五个推荐领域的基于证据的测试实例，涵盖十种前提失败类型。每个实例提供一个可见的推荐上下文和一个被污染的用户查询。我们进一步设计了一个细粒度的评估框架，用于衡量主动检测……（原文摘要在此处截断）

    arXiv:2609.00918v1 Announce Type: new  Abstract: Large language models are increasingly used as interactive recommender assistants. Their evaluation should therefore go beyond plausible item recommendation and test whether they can recognize flawed recommendation requests. Existing recommender benchmarks mainly assess ranking, generation, or preference satisfaction, while existing error-detection benchmarks are usually not grounded in recommendation-specific user and candidate evidence. To address this gap, we introduce RPCBench, a benchmark for evaluating Recommender-Premise Critique: the ability to detect, diagnose, and properly handle faulty premises in natural-language recommendation requests. RPCBench contains evidence-grounded test instances from five recommendation domains and covers ten types of premise failures. Each instance provides a visible recommendation context and a corrupted user query. We further design a fine-grained evaluation framework that measures proactive detec
    
[^114]: 情境神经反馈：大语言模型能否通过特权访问控制其内部表征？

    In-Context Neurofeedback: Can LLMs Control Their Internal Representations through Privileged Access?

    [https://arxiv.org/abs/2609.00904](https://arxiv.org/abs/2609.00904)

    本研究重新设计了要求“特权访问”的大语言模型神经反馈范式，发现模型在更严格设置下无法可靠控制其特权内部表征，表明此前报告的模型自我控制能力可能仅依赖表面机制。

    

    大语言模型（LLM）能否控制自身的内部表征，这对于机器元认知和人工智能安全都具有重要意义。近期一项研究将神经反馈应用于大语言模型，并声称它们能够控制自己的内部表征。然而，所报告的控制可能依赖于表面机制而非真正的内部访问，因为该研究中的控制目标不具有特权性（privileged），即第三方可以从提示词中推断出这些目标。我们重新设计了针对大语言模型的神经反馈范式，使控制目标满足特权访问要求，这更接近于人类认知神经科学中的神经反馈实验。在这种更严格的设置下，模型并未表现出对特权内部表征的可靠控制，这表明先前报告的控制无法排除其依赖表面机制的可能性。我们的结果表明，严格的评估……

    arXiv:2609.00904v1 Announce Type: new  Abstract: Whether large language models (LLMs) can control their own internal representations matters for both machine metacognition and AI safety. A recent study applied neurofeedback to LLMs and claimed that they can control their internal representations. However, the reported control may rely on superficial mechanisms rather than genuine internal access because the control targets in that study are not privileged, meaning that a third party can infer them from the prompt. We redesign the neurofeedback paradigm for LLMs so that the control target satisfies the privileged access requirement, which is closer to neurofeedback experiments in human cognitive neuroscience. Under this stricter setting, the models do not demonstrate reliable control over privileged internal representations, suggesting that previously reported control cannot exclude the possibility that it relies on superficial mechanisms. Our results indicate that rigorous assessments 
    
[^115]: 视觉-语言引导的伪标签用于垃圾分拣语义分割的无监督域自适应

    Vision-Language-Guided Pseudo-Labels for Unsupervised Domain Adaptation in Semantic Segmentation for Waste Sorting

    [https://arxiv.org/abs/2609.00898](https://arxiv.org/abs/2609.00898)

    该论文提出了一种利用SAM、EVA-CLIP和BLIP等视觉-语言基础模型生成跨模态伪标签的流水线，无需任何目标域标注即可实现垃圾分拣语义分割的无监督域自适应。

    

    在实际应用场景中（如自动驾驶、工业垃圾分拣），获取语义分割的标注数据成本高昂，且难以大规模实施。我们提出了一种跨模态伪标签生成流水线，能够在没有任何目标域标注的情况下实现无监督域自适应。该流水线建立在两个核心基础模型之上：SAM生成与类别无关的区域候选，EVA-CLIP基于区域-文本相似度分配语义标签，并通过置信度过滤确保只有可靠的伪标签被用于自训练分割模型。作为可选扩展，BLIP为模糊区域提供基于语言的验证，从而在不改变整体流水线的情况下提升伪标签质量。该流水线在两个域偏移场景上进行了评估：合成到真实的自动驾驶，以及（作为主要关注点）从实验室到工厂的工业垃圾分拣，其性能持续优于仅使用源域数据训练的基线方法。

    arXiv:2609.00898v1 Announce Type: cross  Abstract: Obtaining labeled data for semantic segmentation in applied settings (e.g., autonomous driving, industrial waste sorting) is expensive and often infeasible at scale. We present a cross-modal pseudo-labeling pipeline that enables unsupervised domain adaptation without any target-domain annotations. The pipeline is built on two core foundation models: SAM generates class-agnostic region proposals, and EVA-CLIP assigns semantic labels based on region-text similarity, with confidence filtering ensuring that only reliable pseudo-labels are used for self-training a segmentation model. As an optional extension, BLIP provides language-grounded verification for ambiguous regions, thereby improving pseudo-label quality without altering the overall pipeline. Evaluated on two domain shifts, synthetic-to-real autonomous driving and, with a primary focus, lab-to-factory industrial waste sorting, the pipeline consistently improves over source-only ba
    
[^116]: CARE：面向大语言模型后训练的对比锚点式评分标准演化

    CARE: Contrastive Anchor-based Rubric Evolution for Large Language Model Post-Training

    [https://arxiv.org/abs/2609.00892](https://arxiv.org/abs/2609.00892)

    提出CARE方法，通过将模型得分最高的回答与前沿模型生成的锚点回答进行对比，实现了自适应修复奖励误设和主动追踪改进两个互补的评分标准动态演化机制，解决了基于评分标准的强化学习中静态评分标准易被策略钻空子的问题。

    

    arXiv:2609.00892v1 公告类型：新论文 摘要：基于评分标准的强化学习将开放式指令分解为针对特定提示的、灵活的评分标准，使其比基于可验证奖励的强化学习更适合对大语言模型在开放式任务上进行后训练。然而，静态评分标准不可避免地会随着策略演化而被“钻空子”，而现有的动态方法又会引入新的问题：无方向的评分标准提取、不可靠的钻空子检测以及无限制的评分标准增殖。我们提出了CARE（对比锚点式评分标准演化），它将每一步评分标准演化都建立在一个高质量的锚点回答之上，该锚点回答由前沿模型根据提示及其评分标准生成。在每个训练步骤中，CARE将得分最高的采样回答与锚点进行对比，从而实现两个互补的机制：一个自适应分支，用于被动地修复奖励误设；以及一个追踪分支，用于主动地将（摘要在此处截断）

    arXiv:2609.00892v1 Announce Type: new  Abstract: Rubric-based reinforcement learning decomposes open-ended instructions into prompt-specific, flexible rubrics, making it better suited than reinforcement learning with verifiable rewards for post-training LLMs on open-ended tasks. However, static rubrics are inevitably hacked as the policy evolves, and existing dynamic approaches introduce new problems: undirected rubric extraction, unreliable hack detection, and unbounded rubric proliferation. We propose $\textbf{CARE}$ ($\textbf{C}$ontrastive $\textbf{A}$nchor-based $\textbf{R}$ubric $\textbf{E}$volution), which grounds every rubric evolution step in a high-quality anchor response generated by a frontier model conditioned on the prompt and its rubrics. At each training step, CARE contrasts the highest-scoring rollout against the anchor, enabling two complementary mechanisms: an Adaptive branch that reactively repairs reward misspecification; and a Chase branch that proactively converts
    
[^117]: CacheBridge：高效的跨模型KV缓存迁移

    CacheBridge: Efficient Cross-Model KV Cache Transfer

    [https://arxiv.org/abs/2609.00891](https://arxiv.org/abs/2609.00891)

    CacheBridge通过将目标头与匹配源头进行架构索引配对、以因果注意力敏感度加权重建误差，并使用融合GPU内核，实现了对架构差异鲁棒且成本可控的免训练跨模型KV缓存迁移。

    

    在多模型系统中，大语言模型之间共享上下文时，由于KV缓存是模型特定的，接收模型需要对共享前缀进行预填充（prefill）。近期的闭式跨模型KV迁移方法（下称“全头映射”）通过拟合一个无需训练的仿射映射器将源缓存转换为目标缓存，从而避免了这种重复计算。然而，其全头设计在所选层中将每个目标KV头从所有源KV头进行映射，导致迁移质量对架构差异非常敏感，且映射器的存储和应用成本会随支持的层数增长。为此，我们提出CacheBridge，它协同设计了按架构索引的映射器支持、与注意力机制对齐的校准以及有界映射器构建，同时保留闭式仿射接口以便在线部署。CacheBridge将每个目标头限制为匹配的源头，利用因果注意力敏感度对重建误差进行加权，并使用融合GPU内核来构建……

    arXiv:2609.00891v1 Announce Type: new  Abstract: Sharing context between LLMs in a multi-model system requires the receiving model to prefill the shared prefix because KV caches are model-specific. Recent closed-form cross-model KV transfer, hereafter Full-Head Mapping, avoids this replay by fitting a training-free affine mapper from source to target caches. However, its full-head design maps each target KV head from every source KV head in the selected layers, making transfer quality sensitive to architectural differences and causing mapper storage and application cost to grow with layer support. To this end, we introduce CacheBridge, which co-designs architecture-indexed mapper support, attention-aligned calibration, and bounded mapper construction while retaining a closed-form affine interface for online deployment. CacheBridge restricts each target head to a matched source head, weights reconstruction errors by causal attention sensitivity, and uses a fused GPU kernel to construct 
    
[^118]: 去噪扩散生成模型暗中计算注意力

    Denoising Diffusion Generative Models Secretly Calculate Attentions

    [https://arxiv.org/abs/2609.00885](https://arxiv.org/abs/2609.00885)

    该论文发现去噪扩散模型本质上暗中使用了与Transformer类似的注意力机制，从而证明注意力是机器学习的普适性原理，并据此提出了基于注意力机制的简化图像生成算法，以减少训练时间和计算开销。

    

    去噪扩散模型是图像生成领域的主导架构，而大多数自然语言生成与建模则主要由采用注意力机制的知名Transformer架构来处理。在本文中，我们证明扩散模型本身也使用了一种与Transformer非常相似的注意力机制。因此，注意力作为一种基于通用训练目标的原则，成为机器学习中的普适性原理。我们还展示了自编码器与基于注意力的模型在基本功能原理上的相似性。这些等价性使我们能够根据实际需求在这些设计之间进行互换。举例来说，我们可以重新表述扩散框架，以缩短冗长的训练过程并降低图像生成的计算开销。基于这种方法，我们提出了一种基于注意力机制的简化图像生成算法。结果表明，基于注意力的实现方式（原文摘要在此处截断）……

    arXiv:2609.00885v1 Announce Type: new  Abstract: Denoising diffusion models are the dominant architecture for image generation, whereas most natural language generation and modeling are primarily handled by well-known transformer architectures employing attention mechanism. Here, we show that diffusion models also inherently use an attention mechanism very similar to that of transformers. Therefore, attention emerges as a universal machine learning principle, based on a general training objective. We also show similarities in basic functional principle of auto-encoders and attention-based models. These equivalences allows us to interchange these designs based on practical requirements. As an example, we can reformulate the diffusion framework to reduce the lengthy training process and computation-intensive image generation. Using this approach, a simplified algorithm is proposed for image generation which is based on attention mechanism. Results show that the attention-based implementa
    
[^119]: 通过偏好优化与可解释的视觉-语言推理实现可靠的多模态灾害严重程度评估

    Towards reliable multimodal disaster severity assessment through preference optimization and explainable vision-language reasoning

    [https://arxiv.org/abs/2609.00879](https://arxiv.org/abs/2609.00879)

    该论文提出了一种整合SFT与DPO的两阶段训练框架，并从单次人在回路标注流程中构建推理与偏好两个数据集，同时提升了多模态灾害严重程度评估的准确性（73.64%→78.29%）与解释质量。

    

    可靠的灾害损害评估要求模型既能提供准确的预测，又能给出透明的解释。然而，现有的多模态方法受限于标注数据稀缺以及对推理质量评估不足的问题。本研究提出了一种两阶段训练框架，在统一的数据构建流程中整合了监督微调和直接偏好优化（DPO）。通过单次人在回路（HITL）标注工作流程，衍生出两个互补的数据集：ReasoningSet包含用于SFT的经过验证的推理依据，PreferenceSet则包含用于基于DPO对齐的成对推理依据。该框架采用自动指标、基于模型的评分和人工排序来评估分类性能与解释质量。实验结果表明，SFT将准确率从73.64%提升至78.29%，与基线相比Macro-F1提高了29%……

    arXiv:2609.00879v1 Announce Type: new  Abstract: Reliable disaster damage assessment requires models that provide both accurate predictions and transparent explanations. However, existing multimodal approaches are limited by scarce annotated data and insufficient evaluation of reasoning quality. This study proposes a two-stage training framework that integrates Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) within a unified data construction pipeline. From a single Human-in-the-Loop (HITL) annotation workflow, two complementary datasets are derived, namely ReasoningSet, which contains validated rationales for SFT, and PreferenceSet, which comprises paired rationales for DPO-based alignment. The framework evaluates both classification performance and explanation quality using automatic metrics, model-based scoring, and human ranking. Experimental results show that SFT improves accuracy from 73.64% to 78.29% and increases Macro-F1 by 29% compared to the baseline, w
    
[^120]: 基于FractalNet的卫星巨型星座轨道边缘智能异构联邦学习：以野火监测为案例研究

    FractalNet-Based Heterogeneous Federated Learning for Orbital Edge Intelligence in Satellite Mega-Constellations: A Wildfire Case Study

    [https://arxiv.org/abs/2609.00875](https://arxiv.org/abs/2609.00875)

    该论文提出一种基于FractalNet的异构联邦学习方法，通过分布式路径调度器根据卫星SWAP-C约束与预测星间通信机会动态分配模型深度，并结合周期性更新汇聚和三层智能体控制平面，实现了卫星巨型星座中适应异构硬件条件的轨道边缘智能。

    

    卫星巨型星座正成为大规模的感知、通信与计算基础设施，然而其学习架构在很大程度上仍沿袭自地面联邦学习和以地面为中心的任务运营模式——这对于在尺寸、重量、功耗与成本（SWAP-C）、抗辐射能力、链路可用性以及传播时延方面相差数个数量级的卫星而言并不适用。我们提出了一种基于FractalNet架构的异构联邦学习方法，用于轨道边缘智能。我们形式化了受通信窗口约束、深度异构的联邦优化问题，并引入了一种分布式路径调度器，将模型深度分配设定为SWAP-C约束、预测的星间通信机会以及训练统计信息的函数。为降低消息开销与能耗，每一层级采用周期性汇聚更新的方式，而非在每次通信机会时都进行汇聚，并由一个三层智能体控制平面进行管理……

    arXiv:2609.00875v1 Announce Type: new  Abstract: Satellite mega-constellations are emerging as large-scale sensing, communication, and computation fabrics, yet their learning architectures remain largely inherited from terrestrial federated learning and ground-centric mission operations--- ill-suited to satellites that differ by orders of magnitude in Size, Weight, Power, and Cost (SWAP-C), radiation tolerance, link availability, and propagation delay. We propose a heterogeneous federated learning method based on the FractalNet architecture for orbital edge intelligence. We formalize contact-window-constrained, depth-heterogeneous federated optimization and introduce a distributed path scheduler that assigns model depth as a function of SWAP-C constraints, predicted inter-satellite contacts, and training statistics. To reduce message overhead and energy consumption, each tier pools updates periodically rather than at every contact opportunity, and a three-tier agentic control plane gov
    
[^121]: 超越时钟：测量自适应修正的价值

    Beyond the Clock: Measuring the Value of Adaptive Revision

    [https://arxiv.org/abs/2609.00874](https://arxiv.org/abs/2609.00874)

    该论文在分层智能体系统中研究了元级控制问题，发现学习到的自适应修正时机策略在不同训练种子下产生性质迥异的行为，但均未能超越最佳的强制时机策略，从而首次将状态依赖性与实际决策价值区分开来。

    

    随着智能体系统日益成为复合系统，越来越重要的决策已超出任务执行本身的范畴：更高层的控制器应该在何时保留指导另一个进程的策略，又应该在何时修改它？我们在一个分层潜在推理器中研究这一元级控制问题，该推理器中的管理器可以保留或替换支配低层计算的承诺。在三个预先设定的训练种子上，学习到的修正时机产生了性质上截然不同的策略——从几乎确定性的早期时钟到高度依赖状态的调度分布——但在同一冻结检查点上评估时，没有任何一个优于最佳的强制时机策略。这将状态依赖性与决策价值区分开来：控制器可以随内部状态改变其行为，却无法将这种变化转化为可复现的任务性能收益。对原始检查点的更深入干预研究表明，时……（摘要在此处截断）

    arXiv:2609.00874v1 Announce Type: new  Abstract: As agentic systems become compound systems, increasingly important decisions move above task execution itself: when should a higher-level controller preserve the strategy guiding another process, and when should it revise it? We study this meta-level control problem in a hierarchical latent reasoner whose manager can retain or replace a commitment governing lower-level computation. Across three precommitted training seeds, learned revision timing produces qualitatively different policies, ranging from an almost deterministic early clock to substantially more state conditioned schedule distributions, yet none outperforms the best forced timing policy evaluated on the same frozen checkpoint. This separates state dependence from decision value: a controller can vary its actions with internal state without turning that variation into a reproducible task-performance benefit. A deeper intervention study on the original checkpoint shows that ti
    
[^122]: 面向自动化病理诊断与报告生成的视觉-语言模型基准测试

    Benchmarking Vision-Language Models for Automated Pathology Diagnosis and Report Generation

    [https://arxiv.org/abs/2609.00866](https://arxiv.org/abs/2609.00866)

    该研究构建了来自五个机构、约10,500对样本的泛亚洲病理全切片图像-报告数据集，并建立REG 2025基准系统评估多模态模型，发现顶尖方法的关键在于结构化报告表示而非单纯依赖视觉-语言模型。

    

    视觉-语言模型的快速发展推动了计算病理学的进步；然而，基于全切片图像（WSI）的病理报告生成仍然受限于大规模WSI-报告数据集的稀缺，以及将空间分布的视觉模式映射为结构化临床文本的复杂性。为解决这一问题，我们构建了一个经过临床审核的泛亚洲WSI-报告数据集，包含来自五个机构的约10,500对样本，并通过MICCAI挑战赛建立了REG 2025基准，用于对多模态模型进行系统性评估。我们分析了参赛方法，涵盖预训练视觉-语言模型、多示例学习框架、层次化专家模型、检索增强生成以及跨模态Transformer。结果表明，仅仅使用视觉-语言模型并不足以取得优异性能，表现最佳的方法受益于结构化的报告表示。

    arXiv:2609.00866v1 Announce Type: cross  Abstract: The rapid advancement of vision-language models (VLMs) has accelerated progress in computational pathology; however, whole-slide image (WSI)-based pathology report generation remains limited by the scarcity of large-scale WSI--report datasets and the complexity of mapping spatially distributed visual patterns to structured clinical text. To address this, we introduce a clinically curated Pan-Asia WSI--report dataset of approximately 10,500 pairs from five institutions and establish the REG 2025 benchmark through a MICCAI challenge for systematic evaluation of multimodal models. We analyze submitted methods spanning pretrained VLMs, multiple-instance learning frameworks, hierarchical expert models, retrieval-augmented generation, and cross-modal Transformers. Rather than indicating that VLM use alone was sufficient for superior performance, the results suggest that top-performing methods benefited from structured report representations,
    
[^123]: 强化学习增强的大语言模型智能体求解复杂车辆路径问题

    Reinforcement Learning Enhanced LLM Agents for Complex Vehicle Routing Problems

    [https://arxiv.org/abs/2609.00859](https://arxiv.org/abs/2609.00859)

    本文提出RLEA多智能体框架，利用Soft Q-learning训练的轻量级神经规划器协调大语言模型智能体，并结合进化记忆模块与检索增强生成技术，实现复杂车辆路径问题的自动化建模与求解。

    

    车辆路径问题是基础的组合优化问题，在各种场景中有着广泛的应用。先进的优化求解器能够有效地解决此类问题。然而，为求解器建模复杂的车辆路径问题变体通常需要大量的领域专业知识，这限制了先进优化技术的普及应用。在本文中，我们提出了强化学习增强的大语言模型智能体（RLEA），这是一个旨在自动化复杂车辆路径问题建模的多智能体框架。RLEA引入了一个使用Soft Q-learning训练的轻量级神经规划器，以高效地协调基于大语言模型的智能体的动作。此外，我们为该系统配备了进化记忆模块和检索增强生成技术，使智能体能够在程序生成与迭代优化的过程中利用积累的经验和外部求解器知识来解决车辆路径问题。我们在48个不同的车辆路径问题变体上对该方法进行了评估。

    arXiv:2609.00859v1 Announce Type: new  Abstract: Vehicle Routing Problems (VRPs) are fundamental combinatorial optimization problems with widespread applications in various scenarios. The advanced optimization solvers can effectively solve such problems. However, modeling complex VRP variants for solvers often requires substantial domain expertise, which limits the accessibility of advanced optimization technologies. In this paper, we propose Reinforcement Learning Enhanced LLMAgents(RLEA), a multi-agent framework designed to automate the modeling of complex VRPs. RLEA introduces a lightweight neural Planner trained with Soft Q-learning to efficiently orchestrate the actions of LLM-based agents. In addition, we equip the system with an evolutionary memory module and retrieval-augmented generation, enabling the agent to leverage both accumulated experience and external solver knowledge during program generation and refinement for solving VRPs. We evaluated 48 distinct VRP variants acros
    
[^124]: 可验证的灾害故事线与因果知识图谱：基于引用溯源的异构人道主义数据源流水线

    Verifiable Disaster Storylines and Causal Knowledge Graphs: A Citation-Grounded Pipeline from Heterogeneous Humanitarian Sources

    [https://arxiv.org/abs/2609.00858](https://arxiv.org/abs/2609.00858)

    该论文提出了一个基于检索增强生成（RAG）的流水线，融合EM-DAT结构化灾害记录与ReliefWeb、EMM非结构化文档，自动生成涵盖17个字段的灾害故事线和因果知识图谱，且每个节点和边均附带引用溯源，实现了对原始信息源的完全可追溯性，为人道主义响应提供可验证的态势感知支持。

    

    有效的人道主义响应依赖于对异构、海量信息源的快速综合——在危机爆发的关键早期阶段，这一任务常常超出人类分析能力的极限。我们提出了一个流水线，将来自EM-DAT的结构化灾害记录与来自ReliefWeb和欧洲媒体监测（EMM）的非结构化文档相结合，生成有来源依据的灾害故事线和因果知识图谱，为响应人员和分析人员提供态势感知支持。利用检索增强生成（RAG）技术，该流水线提取结构化故事线——即涵盖17个字段的表格化事件概况，内容包括灾害严重程度、关键驱动因素以及儿童敏感影响指标等——并构建因果知识图谱，其中每个节点和边都配有基于引用的解释性叙述，从而实现对原始来源的完全可追溯性。我们通过人工评估在三个不同的危机用例上对该系统进行了评估……

    arXiv:2609.00858v1 Announce Type: new  Abstract: Effective humanitarian response depends on the rapid synthesis of heterogeneous, high-volume information sources - a task that routinely exceeds human analytical capacity in the critical early hours of a crisis. We present a pipeline that combines structured disaster records from EM-DAT with unstructured documents from ReliefWeb and the European Media Monitor (EMM) to produce source-grounded disaster storylines and causal knowledge graphs supporting situational awareness for responders and analysts. Using Retrieval-Augmented Generation, the pipeline extracts structured storylines - tabular event profiles covering 17 fields, from severity and key drivers to child-sensitive impact indicators - and constructs causal knowledge graphs where each node and edge is enriched with citation-grounded explanatory narratives, enabling full traceability back to primary sources. We evaluate the system on three diverse crisis use cases through a human ev
    
[^125]: 故障定位能否胜过重新尝试？——一项关于测试引导代码修复的安慰剂对照研究

    Does Fault Localization Beat a Fresh Attempt? A Placebo-Controlled Study of Test-Guided Code Repair

    [https://arxiv.org/abs/2609.00854](https://arxiv.org/abs/2609.00854)

    该安慰剂对照研究发现，故障定位在实际场景中很少可用（仅约9%的失败候选可定位），且即便可定位，基于频谱定位的片段填充修复也显著劣于盲目的整体重采样。

    

    故障定位可以将代码模型的修复聚焦于失败测试所涉及的语句，但针对性的编辑可能仅仅因为改动较小而成功，而第二次模型调用也可能根本未利用失败信息就取得成功。我们通过对同一失败候选程序施加三种处理来区分这些解释：盲目的整方案重采样、基于频谱的定位后对可疑片段进行填充，以及在互不重叠的随机代码片段上进行等长填充（安慰剂对照）。在三个冻结的26-32B模型、三个基准数据集和488个失败候选上，外加一个单独声明的来自第三方家族的24B第四个模型，得到三个结果。首先，故障定位很少可用：只有9.0%的失败候选存在暴露失败的可公开测试及可用的频谱。其次，在来自强测试套件的177个可定位候选中，在匹配的尝试次数下，定位填充决定性地输给了盲重采样（3:40，p = 3.0 × 10^-9），这与我们的（摘要在此处截断）

    arXiv:2609.00854v1 Announce Type: cross  Abstract: Fault localization can focus a code model's repair on the statements a failing test implicates, but a targeted edit may succeed merely because it is small, and a second model call may succeed without using the failure at all. We separate these explanations with three arms applied to the same failed candidate: blind whole-solution resampling, spectrum-based localization followed by suspect-span infilling, and same-length infilling at a disjoint random code span. Across three frozen 26-32B models, three benchmarks and 488 failing candidates, plus a separately declared 24B fourth model from a third family, three results follow. First, localization is rarely available: only 9.0% of failing candidates expose a failing public test with a usable spectrum. Second, among the 177 candidates localizable from a strong suite, localized infilling loses decisively to blind resampling at a matched attempt count (3:40, p = 3.0 x 10^-9), opposite to our
    
[^126]: ADGNet：用于红外小目标检测的非对称双文本引导网络

    ADGNet: Asymmetric Dual-text Guided Network for Infrared Small Target Detection

    [https://arxiv.org/abs/2609.00853](https://arxiv.org/abs/2609.00853)

    该论文提出非对称双文本引导网络ADGNet，通过设计与图像无关的抽象目标提示和与图像相关的详细背景提示，并利用非对称双分支交互模块分别处理，解决了红外小目标检测中背景抑制信息不足和特征优化冲突的问题。

    

    红外小目标检测（IRSTD）是一项具有挑战性的任务。仅依赖像素级信息的纯视觉方法难以将目标从杂波中区分出来。当前的多模态方法通常使用单一文本提示来同时描述目标和背景，这种方法缺乏专门的区域引导，并且忽略了红外语义的不对称性。因此，它提供的背景抑制信息不足，并引入了严重的特征优化冲突，导致小目标被噪声淹没。为了解决这些问题，我们提出了一种新颖的非对称双文本引导网络（ADGNet）。具体而言，考虑到红外语义的不对称性，我们首先设计了非对称双文本提示（ADP），它由一个与图像无关的抽象目标提示和一个与图像相关的详细背景提示组成。为了利用这些提示，我们引入了非对称双分支交互（ADBI）模块来分别……

    arXiv:2609.00853v1 Announce Type: cross  Abstract: InfRared Small Target Detection (IRSTD) is a challenging task. Relying solely on pixel-level information, vision-only methods struggle to distinguish targets from clutter. Current multimodal methods typically describe both targets and backgrounds with a single textual prompt. Such an approach lacks dedicated regional guidance and ignores infrared semantic asymmetry. Consequently, it provides insufficient background suppression information and introduces severe feature optimization conflicts, overwhelming small targets with noise. To address these issues, we propose a novel Asymmetric Dual-text Guided Network (ADGNet). Specifically, accounting for the infrared semantic asymmetry, we first design the Asymmetric Dual-text Prompt (ADP), comprising an image-agnostic abstract target prompt and an image-specific detailed background prompt. To leverage these prompts, we introduce an Asymmetric Dual-Branch Interaction (ADBI) module to separatel
    
[^127]: 地球系统建模中机器学习/人工智能应用的能耗与碳排放影响评估清单

    A Checklist to assess the energy and carbon impacts of ML/AI applications in Earth System Modeling

    [https://arxiv.org/abs/2609.00847](https://arxiv.org/abs/2609.00847)

    该论文将分散的机器学习/人工智能可持续发展讨论提炼为一份按模型开发流程各阶段组织的实用清单，帮助地球系统科学从业者评估并降低其应用的能耗与碳足迹，并配套提供了估算能耗和碳排放的指标。

    

    随着机器学习和人工智能逐渐渗透到气候、天气和地球系统建模的几乎每一个方面，我们有必要停下来思考自己的设计决策对科学本身以及我们所消耗的计算资源意味着什么。越来越多的文献探讨了机器学习/人工智能的伦理与可持续发展问题，然而将这些原则转化为日常研究实践仍然是一个挑战，因为大多数最佳实践分散在多项研究和评论之中。在此，我们将这些讨论提炼成一份实用的清单，供机器学习/人工智能和地球系统科学的从业者用来评估并减少其自身应用的环境足迹，该清单围绕模型开发流程的各个连续阶段进行组织。我们还在此基础上补充了从文献中选取的一系列指标，用于估算项目的能耗与碳足迹。

    arXiv:2609.00847v1 Announce Type: cross  Abstract: As machine learning and artificial intelligence find their way into nearly every aspect of climate, weather, and Earth system modeling, it is worth pausing to consider what our design decisions imply for the science and for the computational resources we consume. A growing body of literature addresses the ethical and sustainable development of ML/AI, yet translating these principles into day-to-day research practice remains a challenge as most of best practices are dispersed across multiple studies and commentaries. Here, we distill these discussions into a practical checklist that ML/AI and Earth system science practitioners can use to assess and reduce the environmental footprint of their own applications, organised around the successive stages of the model development pipeline. We complement the checklist with a selection of metrics drawn from the literature for estimating the energy consumption and carbon footprint of a project. Fo
    
[^128]: 面向家用设备的可泛化视觉接地探索

    Towards Generalizable Visually Grounded Exploration of Household Devices

    [https://arxiv.org/abs/2609.00845](https://arxiv.org/abs/2609.00845)

    该论文提出了VGEBench基准，用于评估智能体在无说明书和标注轨迹的情况下，通过动态的“假设-交互-修正”过程将抽象世界知识接地为细粒度视觉可供性，从而操作新型家用设备的可泛化探索能力。

    

    视觉语言模型（VLM）的最新进展在静态视觉识别和高层语义推理方面展现出令人瞩目的能力。然而，当前的具身探索范式仍然严重依赖从人工标注轨迹中进行的模仿学习，这极大地限制了智能体的泛化能力。实现通用自主具身智能体的关键瓶颈在于可泛化的视觉接地探索：即在无需说明书或专门训练的情况下，通过将抽象的世界知识主动接地为细粒度的视觉可供性（affordances），从而操作新设备的能力。然而，现有的基准测试无法评估这一能力：它们通常依赖明确的文档和标注轨迹，忽视了对功能性设备操作至关重要的动态“假设-交互-修正”过程。为了弥合这一差距，我们提出了VGEBench，这是一个旨在评估（该能力的）综合基准（原文摘要在此处截断）。

    arXiv:2609.00845v1 Announce Type: new  Abstract: Recent advancements in Vision-Language Models (VLMs) have demonstrated impressive capabilities in static visual recognition and high-level semantic reasoning. However, current embodied exploration paradigms still heavily rely on imitation learning from human-annotated trajectories, which severely limits agents' generalization ability. The key bottleneck of realizing general autonomous embodied agents lies in Generalizable Visually Grounded Exploration: the ability to operate novel devices without manuals or specific training by actively grounding abstract world knowledge into fine-grained visual affordances. Yet, existing benchmarks fail to evaluate this capability: they generally rely on explicit documents and annotated trajectories, neglecting the dynamic Hypothesis-Interaction-Refinement process essential for functional device operation. To bridge this gap, we introduce VGEBench, a comprehensive benchmark designed to evaluate the gene
    
[^129]: 自回归神经序列模型的概率模型检测

    Probabilistic Model Checking of Autoregressive Neural Sequence Models

    [https://arxiv.org/abs/2609.00838](https://arxiv.org/abs/2609.00838)

    本文提出一种基于概率模型检测的验证流程，从自回归神经序列模型的逐token生成中提取DTMC并用PRISM验证PCTL规约，从而给出约束违反概率的认证区间和构造性保守的输入空间覆盖率曲线，并借助CEGAR循环自适应收紧区间。

    

    测试集准确率对于部署自回归神经序列模型时两个重要问题是沉默的：被测系统（SUT）在采样可达的情况下对违反约束的备选方案分配了多少概率质量，以及输入总体中有多大比例满足领域要求。我们用概率模型检测来回答这两个问题。该流程从被测系统逐token的生成过程中提取离散时间马尔可夫链（DTMC），使用PRISM模型检测器验证形式化的PCTL规约，并将每个输入的判定结果聚合为输入空间上的覆盖率曲线。一个可靠性定理确立了该DTMC作为下近似，因此每个判定都能给出被测系统真实可达概率的认证区间。由这些判定构建的覆盖率因此是构造性保守的。反例引导的抽象细化（CEGAR）循环可自适应地收紧区间。

    arXiv:2609.00838v1 Announce Type: cross  Abstract: Test-set accuracy is silent on two issues that matter when deploying autoregressive neural sequence models: how much probability mass the system under test (SUT) places on constraint-violating alternatives that are reachable under sampling and what fraction of the input population satisfies a domain requirement. We answer both with probabilistic model checking. The pipeline extracts a discrete-time Markov chain (DTMC) from the SUT's token-by-token generation, verifies formal PCTL specifications with the PRISM model checker, and aggregates the per-input verdicts into a coverage curve over the input space. A soundness theorem establishes the DTMC as an under-approximation, so every verdict yields a certified interval on the SUT's true reachability probability. The coverage built from those verdicts is, therefore, conservative by construction. A counterexample-guided abstraction refinement (CEGAR) loop adaptively tightens the interval, an
    
[^130]: 用记忆取代训练：面向Text-to-SQL的列表式选择方法

    Replacing Training with Memory: Listwise Selection for Text-to-SQL

    [https://arxiv.org/abs/2609.00834](https://arxiv.org/abs/2609.00834)

    该论文提出MaP-SQL，一种无需微调的Text-to-SQL列表式选择器，通过从训练数据蒸馏的可复用结构化记忆替代学习选择标准，并利用排名聚合缓解位置偏差，从而以更低成本实现候选查询选择。

    

    现代Text-to-SQL系统通常遵循“生成-执行-选择”的流程，即先生成多个候选查询，再从中选出最优的一个。列表式选择通过联合比较多个候选查询，已被广泛采用，但微调列表式选择器的成本高昂。因此，我们提出了一种无需微调的列表式选择器。我们用推理时的策略取代了两个主要的微调目标：（1）将选择标准的学习视为排序学习；（2）缓解位置偏差。首先，我们构建可复用的结构化记忆，而不是将选择行为学习为模型参数。给定一个问题，MaP-SQL会检索从训练数据中蒸馏出的记忆，这些记忆编码了自然语言如何映射到模式元素、SQL操作以及预期输出。这些记忆作为显式的决策标准，用于以列表方式评估候选查询。其次，为了缓解列表式选择器的排序偏差，我们对多个排序结果进行排名聚合（摘要在此处被截断）。

    arXiv:2609.00834v1 Announce Type: cross  Abstract: Modern Text-to-SQL systems often follow generate-execute-select pipelines, generating multiple candidate queries then selecting the best one. Listwise selection, by jointly comparing multiple candidates, has been widely adopted, but fine-tuning listwise selectors is costly. We thus propose a fine-tuning-free listwise selector. We replace two major fine-tuning objectives with inference-time strategies: (1) learning selection criteria as ordering and (2) mitigating positional bias. First, we build reusable structured memories instead of learning selection behavior as model parameters. Given a question, MaP-SQL retrieves memories distilled from training data that encode how natural language maps to schema elements, SQL operations, and expected outputs. These memories serve as explicit decision criteria for evaluating candidates in a listwise manner. Second, to mitigate ordering bias of listwise selectors, we aggregate rankings across mult
    
[^131]: FLaG：用于标记聚合的频域潜在注意力门控池化

    FLaG: Frequency-Domain Latent-attention Gated Pooling for Token Aggregation

    [https://arxiv.org/abs/2609.00831](https://arxiv.org/abs/2609.00831)

    FLaG是一种即插即用的池化聚合模块，通过在傅里叶域中重新表达编码器输出、利用潜在注意力查询与样本条件通道门控进行标记聚合，在蛋白质、图像和语言等多种任务上均取得最优表现。

    

    标记聚合将词元级表示转换为固定维度的样本表示，但大多数池化方法仅在原始词元空间中操作。我们提出了频域潜在注意力门控池化（FLaG），这是一种即插即用的聚合模块，它在最终池化之前将编码器输出重新表达于傅里叶域中。FLaG通过拼接实部和虚部来表示无冗余的rFFT频谱，利用可学习的潜在查询总结频谱词元，推导出以样本为条件的通道门控，并重构经调制的词元表示以供下游聚合使用。我们在基于ESM2的抗菌肽（AMP）活性预测、基于ResNet18的CIFAR-10和CIFAR-100图像分类，以及三个基于RoBERTa的语言任务上评估了同一架构。FLaG在四种AMP骨干-物种设置中取得了最佳的宏平均Spearman相关系数、RMSE和Recall@50。

    arXiv:2609.00831v1 Announce Type: new  Abstract: Token aggregation converts token-level representations into fixed-dimensional sample representations, but most pooling methods operate only in the original token space. We introduce Frequency-Domain Latent-attention Gated Pooling (FLaG), a plug-in aggregation module that re-expresses encoder outputs in the Fourier domain before final pooling. FLaG represents the nonredundant rFFT spectrum through concatenated real and imaginary components, summarizes spectral tokens with learnable latent queries, derives a sample-conditioned channel gate, and reconstructs modulated token representations for downstream aggregation. We evaluate the same architecture across ESM2-based antimicrobial peptide (AMP) activity prediction, ResNet18 image classification on CIFAR-10 and CIFAR-100, and three RoBERTa-based language tasks. FLaG achieves the best macro-averaged Spearman correlation coefficient, RMSE, and Recall@50 across four AMP backbone-species settin
    
[^132]: 视觉-语言模型中的视觉注意力忠实性是异质的

    Visual Attention Faithfulness in Vision-Language Models is Heterogeneous

    [https://arxiv.org/abs/2609.00830](https://arxiv.org/abs/2609.00830)

    该论文通过因果扰动分析首次系统研究了视觉-语言模型中的视觉注意力忠实性，发现其具有异质性，并归纳出忠实-充分、忠实-分布式和非聚焦三种不同的处理模式。

    

    注意力权重能否忠实反映模型推理这一问题在自然语言处理（NLP）领域一直存在激烈争论，但对于视觉-语言模型（VLM）中的视觉模态而言，该问题在很大程度上仍未被探索。我们通过对当前视觉-语言模型进行因果扰动分析来填补这一空白，评估注意力排序的视觉token的全面性与充分性差距。我们的分析揭示出视觉注意力忠实性是异质的，表现为三种不同的处理模式：忠实-充分模式，即前k个注意力token对于预测既必要又充分；忠实-分布式模式，即这些token是必要的但仍需要更广泛的视觉上下文；以及非聚焦模式，即没有任何局部注意力区域单独必要，而视觉信息仍是触发预测的关键要素。此外，人工标注的真实区域仅在约60%的情况下满足全面性，与注意力排序的……

    arXiv:2609.00830v1 Announce Type: cross  Abstract: Whether attention weights faithfully reflect model reasoning has been actively debated in NLP, yet this question remains largely unexplored for the visual modality in Vision-Language Models (VLMs). We address this gap through causal perturbation analysis on current VLMs, evaluating both the comprehensiveness and sufficiency gap of attention-ranked visual tokens. Our analysis reveals that visual attention faithfulness is heterogeneous, manifesting in three distinct processing modes: Faithful-Sufficient, where top-$k$ attention tokens are both necessary and sufficient for prediction; Faithful-Distributed, where they are necessary but broader visual context remains required; and Non-Focal, where no localized attention region is individually necessary while visual information remains an essential trigger for prediction. Furthermore, human-annotated ground-truth regions satisfy comprehensiveness in only $\sim 60$% of cases compared with mod
    
[^133]: HarnessEvolve：从参考轨迹中学习以实现可靠的智能体自我进化

    HarnessEvolve: Learning from Reference Trajectories for Reliable Agent Self-Evolution

    [https://arxiv.org/abs/2609.00829](https://arxiv.org/abs/2609.00829)

    HarnessEvolve 提出了一种从参考轨迹中学习的智能体自我进化框架，通过将执行、评估、优化和门控解耦为独立模块，克服了信用分配失败、捷径学习和灾难性遗忘三大难题，实现了可靠且可泛化的智能体自我进化。

    

    自我进化的智能体通过基于环境反馈优化其“线束”（harness）——包括提示词、技能、工具和执行逻辑——来向自主性迈进。然而，这一范式面临三大挑战：信用分配失败（credit assignment failure），即仅凭最终的成功/失败反馈难以判断是哪一步导致了错误；捷径学习（shortcut learning），即智能体记忆特定任务的模式而非习得可泛化的能力；以及灾难性遗忘（catastrophic forgetting），即缺乏防护的更新会降低先前习得的能力。本文提出了 HarnessEvolve，一个通过从参考轨迹中学习来实现可靠智能体自我进化的自进化框架。HarnessEvolve 将执行智能体与进化流程解耦，把执行、评估、优化和门控分别分配给独立的智能体模块，从而实现可泛化且稳定的线束改进。

    arXiv:2609.00829v1 Announce Type: cross  Abstract: Self-evolving agents advance toward autonomy by optimizing their harness---prompts, skills, tools, and execution logic---based on environmental feedback. This paradigm, however, is hampered by three challenges: \textit{credit assignment failure}, where terminal success/failure feedback makes it ambiguous which step caused the error; \textit{shortcut learning}, where agents memorize task-specific patterns rather than acquire generalizable capabilities; and \textit{catastrophic forgetting}, where unguarded updates degrade previously acquired competence. In this paper, we introduce HarnessEvolve, a self-evolving framework that learns from reference trajectories to achieve reliable agent self-evolution. HarnessEvolve decouples the execution agent from the evolutionary pipeline, assigning execution, evaluation, optimization, and gating to independent agent modules, enabling generalizable and stable harness improvements. Specifically, Harnes
    
[^134]: 光鲜却未解决：识别长时程工具使用智能体中的后期压力状态

    Polished but Unresolved: Identifying Late-Stage Pressure States in Long-Horizon Tool-Use Agents

    [https://arxiv.org/abs/2609.00823](https://arxiv.org/abs/2609.00823)

    该论文首次识别出长时程工具使用智能体的“后期压力状态”（即倾向于提交看似完整精美但关键约束尚未解决的答案），证明该状态可通过线性探针从隐藏状态中检测、可被激活干预因果地改变，并据此提出PSPR插件以自适应方式缓解压力、改善提交决策。

    

    长时程工具使用智能体不仅需要搜索和规划，还需要决定何时定稿提交。我们研究了后期压力状态：在这种状态下，智能体倾向于提交一个看似完整、精美的最终答案，而关键约束条件仍未解决。我们首先训练了一个线性探针，证明这种压力状态可以从智能体的隐藏状态中被识别出来。随后，我们沿该压力方向进行激活干预，发现移动隐藏状态会同时改变压力评分，以及智能体是继续使用工具还是提前提交。通过受控的上下文操纵，我们进一步观察到，约束清晰度和动作映射能够缓解这种压力。基于这些发现，我们提出了探针感知压力缓解（Probe-Sensed Pressure Relief, PSPR），这是一个插件：在中等压力下施加轻量级的压力缓解方向，在高压力风险下则转向结构化组织。实验在……

    arXiv:2609.00823v1 Announce Type: new  Abstract: Long-horizon tool-use agents need not only to search and plan, but also to decide when to finalize. We study late-stage pressure states, in which an agent is biased toward submitting a final answer that appears complete and polished while key constraints remain unresolved. We first train a linear probe to show that this pressure state is identifiable from the agent's hidden states. Then, we use activation interventions along this pressure direction and find that shifting the hidden states changes both the pressure score and whether the agent continues tool use or submits early. Through controlled context manipulations, we further see that the pressure is mitigated by constraint clarity and action mapping. Based on these findings, we propose Probe-Sensed Pressure Relief (PSPR), a plugin that applies lightweight pressure relief direction under moderate pressure and moves to structured organization under high pressure risk. Experiments on m
    
[^135]: AnalysisBank：用于财务报告生成的专家分析模式库

    AnalysisBank: An Expert Analysis Pattern Library for Financial Report Generation

    [https://arxiv.org/abs/2609.00818](https://arxiv.org/abs/2609.00818)

    提出了AnalysisBank，将专家财务报告提炼为可复用的“数据信号-分析手法”分析库，在推理时通过检索匹配的分析手法生成报告，使新颖且有数据支撑的洞察比例较结构层面基线提升1.7-3.7倍，且该方法可泛化至科学写作领域。

    

    我们认为，财务报告生成应当在分析层面而非结构层面进行，即通过数据衍生的洞察来组织内容，而非依赖高层级的主题或章节。为此，我们提出了AnalysisBank，它将专家报告提炼成一个可复用的“分析”库，其中每个分析条目都由一个数据信号、一个分析手法及其所来源的专家文本片段配对组成。在推理阶段，AnalysisBank将输入信号与库中条目进行匹配，并应用检索到的分析手法来撰写报告。对从550份专家报告中提炼出的分析所进行的研究显示，其分布呈重尾特征，包含47-52种信号类型，横跨13种分析手法类型。在两个金融基准和四种LLM骨干模型上，AnalysisBank使新颖且基于数据的洞察比例较结构层面基线提升了1.7-3.7倍。向科学写作的迁移实验表明，这一分析层面的区分可以泛化到金融领域之外。

    arXiv:2609.00818v1 Announce Type: new  Abstract: We argue that financial report generation should operate at the analytical rather than structural level, composing content from data-derived insights rather than high-level topics or sections. To this end, we propose AnalysisBank, which distills expert reports into a reusable library of Analyses, each pairing a data signal, an analytical move, and the expert span it was derived from. At inference time, AnalysisBank matches input signals to library entries and applies the retrieved moves to compose the report. A study of Analyses distilled from 550 expert reports reveals a heavy-tailed distribution of 47-52 signal types spanning 13 move types. On two financial benchmarks across four LLM backbones, AnalysisBank increases the proportion of novel, data-grounded insights by 1.7-3.7x over structural-level baselines. Transfer to scientific writing suggests that the distinction generalizes beyond finance. Code and the distilled Analysis library 
    
[^136]: 单一策略，任意预算：通过强化学习内化预算感知搜索

    One Policy, Any Budget: Internalizing Budget-Aware Search via Reinforcement Learning

    [https://arxiv.org/abs/2609.00813](https://arxiv.org/abs/2609.00813)

    该论文提出 AnySearch 框架，通过“脚手架引导到自主运行”的两阶段课程强化学习，以及耦合答案准确性与预算效率的复合奖励，使单一策略能够在任意预算约束下执行预算感知搜索。

    

    尽管强化学习已使基于大语言模型（LLM）的搜索智能体能够调用外部工具，但现有方法是在固定预算下进行训练，无法在部署时适应约束条件的变化。我们提出 AnySearch，一个通过训练脚手架和课程强化学习，使单一策略能够在任意预算约束下执行预算感知搜索的框架。在第一阶段，我们通过显式预算状态注入和结构化推理提示来训练智能体，引导其在线性递减的预算下进行高效分配。在第二阶段，移除脚手架，智能体学习在自适应采样的预算约束下自主运行，与推理时的条件相匹配。两个阶段均通过复合奖励进行优化，该奖励通过绝对和相对信号将答案准确性与预算效率相耦合，其中自适应权重会放大高准确性查询的效率信号，并衰减……

    arXiv:2609.00813v1 Announce Type: new  Abstract: While reinforcement learning has enabled LLM-based search agents to invoke external tools, existing methods train under fixed budgets and cannot adapt when constraints vary at deployment. We propose AnySearch, a framework that enables a single policy to perform budget-aware search under any budget constraint through a training scaffold and curriculum reinforcement learning. In the first phase, we train the agent with explicit budget state injection and structured reasoning prompts that guide efficient allocation under linearly decaying budgets. In the second phase, the scaffold is removed and the agent learns to operate autonomously under adaptively sampled budget constraints, matching inference conditions. Both phases are optimized with a composite reward that couples answer accuracy with budget efficiency through absolute and relative signals, where an adaptive weight amplifies the efficiency signal for high-accuracy queries and attenu
    
[^137]: Ctrl-F-Resist：监测极右翼线上活动的公民社会组织的实践、挑战与技术需求

    Ctrl-F-Resist. Practices, Challenges, and Technical Needs of Civil Society Organizations Monitoring the Far-Right Online

    [https://arxiv.org/abs/2609.00808](https://arxiv.org/abs/2609.00808)

    本文通过对12家德国公民社会组织的15名从业者进行定性研究，揭示了这些组织在极右翼在线监测工作中的长期实践、面临的挑战（法律不确定性、平台访问受限、资金不足）以及技术需求，强调它们是数字治理中被忽视的关键利益相关者。

    

    随着极右翼行为者日益利用在线平台传播意识形态并动员支持者，公民社会组织（CSOs）在监测网络上的反民主动态方面发挥着至关重要却未被充分认可的作用。与事实核查员或内容审核员不同，公民社会组织从事长期的、结合具体情境的分析工作，且往往在资源受限和条件不稳定的环境下开展。尽管具有重要的社会作用，公民社会组织在采用或共同开发技术解决方案方面面临重大障碍，包括法律上的不确定性、平台访问权限受限以及长期的资金不足。然而，现有研究和工具开发工作在很大程度上忽视了这些行为者，而更倾向于关注具有制度化背景的利益相关者。本文通过对来自12家德国公民社会组织的15名从事在线监测工作的从业者进行定性研究来填补这一空白，将这些组织定位为数字治理中关键却被忽视的利益相关者。

    arXiv:2609.00808v1 Announce Type: cross  Abstract: As far-right actors increasingly exploit online platforms to disseminate ideology and mobilize supporters, civil society organizations (CSOs) play a vital yet underrecognized role in monitoring antidemocratic dynamics online. Unlike fact-checkers or content moderators, CSOs engage in long-term, contextualized analysis, often in resource-constrained settings and under precarious conditions. Despite their critical societal role, CSOs face significant barriers to adopting or co-developing technical solutions, including legal uncertainty, limited platform access, and chronic underfunding. Existing research and tool development efforts have largely overlooked these actors in favor of more institutionally embedded stakeholders. This paper addresses this gap through a qualitative study with 15 practitioners from 12 Germany-based CSOs engaged in online monitoring, positioning them as key yet overlooked stakeholders in the governance of digital
    
[^138]: 迈向可靠且实用的评估流水线

    Towards a Reliable and Practical Eval Pipeline

    [https://arxiv.org/abs/2609.00805](https://arxiv.org/abs/2609.00805)

    该论文提出了一种结合评估检查清单创建与学习式聚合的端到端评估流水线，提高了LLM评判者之间的一致性及与人类判断的吻合度，同时提供自一致性、解释和预测不确定性。

    

    基于大语言模型（LLM）的软件系统日益需要在开发生命周期中将有效的“评估”作为质量门控。然而，现有工作通常仅解决评估可靠性的个别方面，而非全部实际需求。我们提出了一种端到端的评估流水线，将评估检查清单的创建与针对检查清单响应的学习式聚合相结合，以提高LLM评判者之间的一致性以及与人类判断相比的准确性。该框架还提供了自一致性、解释和预测不确定性，并通过实验证明了其有效性。

    arXiv:2609.00805v1 Announce Type: new  Abstract: LLM-based software systems increasingly require effective "evals" as quality gates in the development lifecycle. However, existing work typically addresses individual aspects of eval reliability rather than the full set of practical requirements. We present an end-to-end eval pipeline that combines eval checklist creation, with learned aggregation for checklist responses, to improve agreement across LLM judges and accuracy against human judgments. The framework additionally pro- vides self-consistency, explanations, and prediction uncertainty, and we empirically demonstrate its effectiveness.
    
[^139]: 智能体程序：计算材料科学中一种新兴形式的科学软件

    Agentic programs: an emerging form of scientific software in computational materials science

    [https://arxiv.org/abs/2609.00795](https://arxiv.org/abs/2609.00795)

    本文提出“智能体程序”这一新兴科学软件范式，将确定性算法与有界的LLM科学判断、任务验证和阶段性成熟相结合，并以从实验测量的无序晶体结构构建原子模型的DeMARS系统为例加以展示。

    

    计算材料科学传统上将算法任务委托给计算机，而将科学判断留给人类。我们提出，近期基于大语言模型（LLM）的智能体框架催生了一种新兴形式的科学软件——智能体程序，它将确定性算法与有界的LLM判断、任务特定的验证、阶段性成熟以及生产中的完全委托相结合。我们以DeMARS为例说明这一概念，DeMARS是一个用于从实验测量的无序晶体结构构建原子级模型的智能体程序。

    arXiv:2609.00795v1 Announce Type: cross  Abstract: Computational materials science has traditionally delegated algorithmic tasks to computers while leaving scientific judgments to humans. We argue that recent LLM-based agent harnesses enable an emerging form of scientific software, agentic programs, that combine deterministic algorithms with bounded LLM-based judgment, task-specific verification, episodic maturation, and complete delegation in production. We illustrate this concept with DeMARS, an agentic program for constructing atomistic models from experimentally measured disordered crystal structures.
    
[^140]: MADS：超越标准频谱摘要的多视角声学描述符集

    MADS: A Multiview Acoustic Descriptor Set Beyond Standard Spectral Summaries

    [https://arxiv.org/abs/2609.00792](https://arxiv.org/abs/2609.00792)

    本文提出了MADS，一个紧凑的19维物理信息驱动的多视角声学描述符集，通过统一表示编码声音的激励、阻尼、周期性、冲击性等物理特性，突破了传统频谱摘要仅将声音视为频谱模式的局限，并在多个音频分类数据集上与经典手工特征基线进行了对比评估。

    

    主流的音频分类流程要么依赖于紧凑的手工特征摘要，要么在进行深度建模之前依赖于固定的时频前端（如对数梅尔表示）。尽管这些表示方法取得了高度成功，但它们并未显式地揭示底层发声事件的物理动态。我们提出了MADS（多视角声学描述符集，Multi-view Acoustic Descriptor Set），这是一个紧凑的19维物理信息描述符集，旨在捕捉音频信号中互补的频谱、时间、机械和随机结构。MADS并非仅仅将声音视为频谱模式，而是在统一的多视角表示中编码与激励、阻尼、周期性、冲击性和结构一致性相关的属性。我们在ESC-10、ESC-50和MSoS数据集上使用标准的经典机器学习模型对MADS进行了评估，并将其与两种传统的手工特征基线进行了比较：一个紧凑的26维MFCC基线以及一个……（摘要原文在此处被截断）

    arXiv:2609.00792v1 Announce Type: cross  Abstract: Dominant audio classification pipelines rely either on compact handcrafted summaries or on fixed time-frequency frontends such as log-mel representations prior to deep modeling. While highly successful, these representations do not explicitly expose the physical dynamics of the underlying sound-generating event. We introduce MADS (Multi-view Acoustic Descriptor Set), a compact 19-dimensional physics-informed descriptor set de- signed to capture complementary spectral, temporal, mechanical, and stochastic structure in audio signals. Rather than treating sound only as a spectral pattern, MADS encodes properties related to excitation, damping, periodicity, impulsiveness, and structural consistency within a unified multi-view representation. We evaluate MADS using standard classical machine learning models on ESC-10, ESC-50, and MSoS, and compare it against two conventional handcrafted baselines: a compact 26D MFCC- based baseline and an e
    
[^141]: Instella-MoE 技术报告

    Instella-MoE Technical Report

    [https://arxiv.org/abs/2609.00791](https://arxiv.org/abs/2609.00791)

    Instella-MoE 是一个完全开源、总参数160亿（激活28亿）的混合专家语言模型，完全基于 AMD GPU 从零训练，凭借 Gated MLA 与 FarSkip-Collective 等架构与系统级创新实现了高效训练推理，并在基准测试中超越 OLMo-3-7B 等此前完全开源模型。

    

    在这项工作中，我们介绍了 Instella-MoE，这是一个完全开源的混合专家模型语言模型，总参数量为160亿，每个token激活28亿参数，完全在 AMD Instinct MI300X 和 MI325X GPU 上从零开始训练。Instella-MoE 将稀疏激活的 MoE 设计与架构和系统级创新相结合，包括门控多头潜在注意力（Gated MLA）和 FarSkip-Collective 连接机制，从而实现了高效的大规模训练与推理。该模型通过多阶段流水线开发而成，包括预训练、中期训练、长上下文扩展、结合反馈驱动数据整理的监督微调、直接偏好优化，以及采用多教师在线策略蒸馏的强化学习。Instella-MoE 在标准预训练基准上取得了76.7的平均分，超越了包括 OLMo-3-7B、SmolLM3-3B 和 OLMoE-1B-7B 在内的先前完全开源模型。

    arXiv:2609.00791v1 Announce Type: cross  Abstract: In this work, we introduce Instella-MoE, a fully open Mixture-of-Experts (MoE) language model with 16 billion total parameters and 2.8 billion active parameters per token, trained entirely from scratch on AMD Instinct MI300X and MI325X GPUs. Instella-MoE combines a sparsely activated MoE design with architectural and system-level innovations, including Gated Multi-head Latent Attention (Gated MLA) and FarSkip-Collective connectivity, enabling efficient large-scale training and inference. The model is developed through a multi-stage pipeline comprising pre-training, mid-training, long-context extension, supervised fine-tuning with feedback-driven data curation, direct preference optimization, and reinforcement learning with Multi-Teacher On-Policy Distillation. Instella-MoE achieves an average score of 76.7 across standard pre-training benchmarks, outperforming prior fully open models including OLMo-3-7B, SmolLM3-3B, and OLMoE-1B-7B, wh
    
[^142]: StudyBench：自我进化能否从教科书中榨取出奥赛能力？

    StudyBench: Can Self-Evolution Squeeze Textbooks for Olympiad Capability?

    [https://arxiv.org/abs/2609.00787](https://arxiv.org/abs/2609.00787)

    StudyBench是一个受控物理基准，用于衡量自我进化方法将教科书式训练材料转化为奥赛级解题能力的效率，研究发现模型在困难教科书问题上的提升很难迁移到奥赛级问题上。

    

    人类只需研读少量编写精良的教科书就能掌握一门学科，并尝试攻克其中最难的问题。我们认为理想的自我进化方法应当具备同样的特性，即能够自主地从原始训练材料中学习，从而获得可迁移的问题解决能力。然而，目前我们仍缺乏对这一能力的直接度量方式。我们提出了StudyBench，一个受控的物理基准，用于直接衡量自我进化方法将训练材料转化为能力的效率。我们将测试集组织为两部分：应用集，由高难度教科书问题构成，用于评估吸收能力；迁移集，由奥赛级问题构成，用于评估迁移能力。通过对三个基础模型上的代表性自我进化方法进行基准测试，我们发现模型在应用集上的提升很少能够转化为更难的迁移集上的进步。一项关于指导的消融实验揭示了“指导鸿沟”现象：即使……（摘要原文在此处被截断）

    arXiv:2609.00787v1 Announce Type: new  Abstract: Humans need to study only a handful of well-written textbooks to master a discipline and attempt its hardest problems. We argue that an ideal self-evolution method should share the same property, that is autonomously learning from raw training material for transferable problem-solving capability. However, we still lack a direct measurement for it. We introduce StudyBench, a controlled physics benchmark that directly measures how efficiently a self-evolution method converts training material into capability. We organise the test set into an Application Set, consisting of difficult textbook problems and evaluating absorption ability, and a Transfer Set, consisting of olympiad-level problems and evaluating transfer ability. Benchmarking representative self-evolution methods across three base models, we find that improvements on the Application Set rarely translate to the harder Transfer Set. A guidance ablation exposes a Guidance Gap: even 
    
[^143]: 当特征成为实例：面向无监督特征选择的倒置对比学习

    When Features Become Instances: Inverted Contrastive Learning for Unsupervised Feature Selection

    [https://arxiv.org/abs/2609.00782](https://arxiv.org/abs/2609.00782)

    该论文提出ICLFS框架，通过倒置数据矩阵使特征成为对比学习中的实例，并利用掩码正视图、打乱负视图和InfoNCE目标，将无监督特征选择重新表述为特征层面的表示一致性学习问题。

    

    无监督特征选择旨在在不使用类别标签的情况下，寻找一个紧凑且信息量丰富的特征子集，这使得特征的效用难以定义。因此，现有的无监督特征选择方法依赖于间接的结构性准则，例如相似性保持、局部性、稀疏性、聚类几何结构或重建质量。在本文中，我们转而通过表示一致性来研究无监督特征选择，提出了用于无监督特征选择的倒置对比学习方法，这是一种以特征为单位的对比学习框架，将无监督特征选择重新表述为针对特征而非样本的表示学习问题。ICLFS首先对数据矩阵进行倒置，使每个特征由其样本轮廓向量表示，然后构建多个掩码正视图以及一个打乱顺序的负视图，并在基于InfoNCE的目标函数下学习在这些结构化扰动之间保持一致的投影空间表示。（注：原文摘要在此处截断）

    arXiv:2609.00782v1 Announce Type: new  Abstract: Unsupervised feature selection seeks a compact subset of informative features without access to class labels, making feature utility difficult to define. Existing UFS methods therefore rely on indirect structural criteria, such as similarity preservation, locality, sparsity, cluster geometry, or reconstruction quality. In this paper, we instead study UFS through representation consistency and propose Inverted Contrastive Learning for Unsupervised Feature Selection (ICLFS), a feature-wise contrastive framework that reformulates UFS as a representation learning problem over features rather than samples. ICLFS first inverts the data matrix so that each feature is represented by its sample-profile vector, then constructs multiple masked positive views together with a shuffled negative view, and learns projector-space representations that remain consistent across these structured perturbations under an InfoNCE-based objective. Motivated by re
    
[^144]: Solaris：迈向“生成而非编码”的界面

    Solaris: Towards Interfaces That Are Generated, Not Coded

    [https://arxiv.org/abs/2609.00776](https://arxiv.org/abs/2609.00776)

    Solaris 提出了一个界面世界模型，将鼠标交互作为条件信号、自回归逐帧直接生成交互式 UI，并结合少步蒸馏与语言模型解释用户意图，实现了无需代码、实时生成界面外观与行为的新范式。

    

    传统上，数字界面通过代码等中间表示来实现，需要预先指定其外观和行为。我们提出了 Solaris，一个界面世界模型，它直接逐帧生成交互式 UI，以响应用户的操作。Solaris 将鼠标交互视为条件信号，并以交互速度自回归地合成由此产生的视觉状态。为了在实现实时生成的同时保持长时间交互中的视觉一致性，我们将自回归帧生成与少步蒸馏以及在模型自身输出上的训练相结合。一个语言模型对视觉世界模型进行补充，通过解释用户意图并指定交互应如何影响生成的环境，将高层推理与视觉渲染分离开来。通过动态生成界面的外观和行为，Solaris……

    arXiv:2609.00776v1 Announce Type: cross  Abstract: Digital interfaces are traditionally implemented through intermediate representations such as code, requiring their appearance and behavior to be specified in advance. We introduce Solaris, an interface world model that instead generates an interactive UI directly, frame by frame, in response to user actions. Solaris treats mouse interactions as conditioning signals and autoregressively synthesizes the resulting visual state at interactive speeds. To enable real-time generation while maintaining visual coherence over extended interactions, we combine autoregressive frame generation with few-step distillation and training on the model's own outputs. A language model complements the visual world model by interpreting user intent and specifying how interactions should affect the generated environment, separating high-level reasoning from visual rendering. By generating both the appearance and behavior of an interface dynamically, Solaris 
    
[^145]: VOIM：面向RGB-D与单目SLAM的免训练开放词汇3D实例建图

    VOIM: Training-Free Open-Vocabulary 3D Instance Mapping for RGB-D and Monocular SLAM

    [https://arxiv.org/abs/2609.00775](https://arxiv.org/abs/2609.00775)

    VOIM是一个免训练的开放词汇3D实例建图系统，它将标签与实例决策推迟到多视角软证据按体素累积之后，仅凭RGB-D或单目RGB即可实现，其建图质量（mIoU）比最强的在线RGB-D系统OVO-SLAM高出4.8至11.7，证明了建图阶段而非感知模型才是决定性因素。

    

    我们提出了体素锚定的在线实例管理器（Voxel-Grounded Online Instance Manager, VOIM），这是一个免训练的、基于体素的实例管理器，能够从RGB-D输入或仅从单目RGB构建开放词汇的3D实例地图，这一场景是此前没有任何免训练系统能够处理的。在线系统通常在首次检测时即对物体实例进行分割并赋予标签，也就是在证据最薄弱的时候便做出最终决定。VOIM则相反，它将标签与实例的决策推迟到来自未经修改的现成感知模型的软证据在多个视角下按体素累积之后再进行。我们证明了是建图阶段而非所使用的特定感知模型决定了结果：在ScanNet++数据集上跨越四种感知配置（改变区域描述符、检测器标签先验和掩码来源），该地图在mIoU上超过最强的在线RGB-D系统OVO-SLAM达4.8至11.7。感知并非中性的，换用该基线系统自带的描述符系列会损失4.1的领先幅度，然而基线……

    arXiv:2609.00775v1 Announce Type: cross  Abstract: We present Voxel-Grounded Online Instance Manager (VOIM), a training-free voxel-grounded instance manager that builds open-vocabulary 3D instance maps from RGB-D or from monocular RGB alone, a regime no prior training-free system addresses. Online systems typically segment object instances and label them at first detection, committing when evidence is weakest. VOIM instead defers label and instance decisions until soft evidence from unmodified, off-the-shelf perception has accumulated per voxel across views. We show that the mapping stage, rather than the particular perception models, carries the result: across four perception configurations on ScanNet++, varying the region descriptor, the detector label prior and the mask source, the map exceeds the strongest online RGB-D system, OVO-SLAM, by between 4.8 and 11.7 mIoU. Perception is not neutral, and substituting that baseline's own descriptor family costs 4.1 of the margin, yet the ba
    
[^146]: DiagEvo：基于分层错误记忆的诊断引导自进化

    DiagEvo: Diagnosis-Guided Self-Evolution via Hierarchical Error Memory

    [https://arxiv.org/abs/2609.00768](https://arxiv.org/abs/2609.00768)

    DiagEvo通过诊断器从语言模型自身的失败历史中提取反复出现的错误原因，构建分层错误记忆来引导自我博弈进化，无需依赖外部任务信息即可持续提升求解器性能。

    

    自我博弈是语言模型自我进化的有效范式，但在缺乏引导的情况下，求解器的性能可能在多轮迭代中陷入停滞甚至下降。无引导方法使用难度、可学习性或多样性等信号来引导题目生成，这些信号虽能保持题目的挑战性与多样性，却无法指明后续轮次应针对哪些尚未解决的推理弱点。有引导方法则从外部任务资源（如人类示例、文档语料库或指定难度目标）中获取方向，因此依赖于自我博弈循环之外提供的任务信息。我们证明了所需的方向可以从求解器自身的失败历史中推导得出。我们提出 DiagEvo，其诊断器从失败历史中提取反复出现的错误原因，并将其存储在分层的错误原因记忆中。该记忆将相关原因归类到技能节点下，并据此将各节点标记为“活跃”或“已掌握”。

    arXiv:2609.00768v1 Announce Type: new  Abstract: Self-play is an effective paradigm for language-model self-evolution, but without guidance, solver performance can plateau or decline across rounds. Unguided methods steer question generation with signals such as difficulty, learnability, or diversity. These signals keep questions challenging and varied but do not specify which unresolved reasoning weaknesses later rounds should target. Guided methods obtain direction from external task resources, including human examples, document corpora, or specified difficulty targets, and therefore rely on task information supplied outside the self-play loop. We show that the needed direction can instead be derived from the solver's own failure history. We introduce DiagEvo, whose diagnostician extracts recurring error causes from this history and stores them in a hierarchical error-cause memory. The memory groups related causes under skill nodes and tracks each as Active or Mastered according to se
    
[^147]: 你在想我所想吗？：检验神经网络架构中的概念分离

    Are You Thinking What I am Thinking? : Examining Conceptual Separation in Neural Architectures

    [https://arxiv.org/abs/2609.00764](https://arxiv.org/abs/2609.00764)

    本研究通过对CNN和LLM内部激活的几何与分布分析，揭示神经网络中存在“概念分离”现象——同一概念形成连贯表示、相关概念在表示空间中彼此更近，但这种连贯性在未见概念、域偏移和模糊主题下会减弱或坍塌。

    

    神经网络越来越多地被用于识别明确定义的概念以及模糊概念，但输出层面的指标几乎无法揭示这些概念在内部是如何被表示的。我们的研究探讨这些网络是否表现出“概念分离”特性：同一概念的样本是否形成连贯一致的表示，以及相关概念是否在表示空间中彼此更接近。我们通过对卷积神经网络（CNN）和大型语言模型（LLM）内部激活进行几何和分布分析，考察了这种概念组织方式。在CNN中，熟悉的ImageNet概念形成了连贯且语义有序的表示，而对于未见过的概念，这种连贯性会减弱，并且在类内域偏移的情况下会受到损害。在LLM中，明显不同的领域保持良好的分离，相关子领域彼此靠近，而模糊主题之间的区分在均值和协方差层面都会坍塌。

    arXiv:2609.00764v1 Announce Type: cross  Abstract: Neural networks are increasingly employed to identify both well-defined and ambiguous concepts, yet output-level metrics reveal little about how those concepts are represented internally. Our study asks if these networks exhibit \textit{conceptual separation}: if examples of the same concept form coherent representations, and whether related concepts lie closer together in the representation space. We examine this conceptual organisation in Convolutional Neural Networks (CNNs) and Large Language Models (LLMs) through geometric and distributional analysis of their internal activations. In CNNs, familiar ImageNet concepts form coherent and semantically ordered representations, while this coherence weakens for unseen concepts and suffers within-class domain shift. In LLMs, clearly distinct domains remain well separated, related subdomains move closer together, and the distinction between ambiguous topics collapses at both the mean and cov
    
[^148]: 基于本体扩展与检索的越南历史教科书树状知识图谱自动构建

    Automated Tree Knowledge Graph Construction using Ontology Expansion and Retrieval from Vietnamese History Textbooks

    [https://arxiv.org/abs/2609.00763](https://arxiv.org/abs/2609.00763)

    本文提出了一种从越南历史教科书自动构建树状知识图谱的端到端流水线，通过并查集批内去重、近似跨批搜索和带质心过滤的双LLM验证等三阶段混合关系抽取方法，解决了低资源语言本体扩展和层次化检索策略系统评估的难题。

    

    基于层次化知识图谱（KG）的检索增强生成（RAG）已成为利用结构化知识支持大语言模型的一种强大方法。然而，目前存在两个主要挑战：（i）缺乏针对越南语等低资源语言使用本体扩展进行知识图谱自动构建的方法；（ii）缺乏对利用层次结构的知识检索策略的系统性评估。本文提出了一个用于知识图谱构建与检索策略评估的端到端流水线。在知识图谱构建阶段，我们采用三阶段混合关系抽取流水线：通过并查集（Union-Find）进行批内去重、近似跨批搜索，以及结合质心过滤器以减少提示词的LLM抽取，并配合五步双LLM验证器以防止本体膨胀。系统采用双层架构，由不可合并的结构节点组成，以保留文档的结构信息。

    arXiv:2609.00763v1 Announce Type: new  Abstract: Hierarchical Knowledge graph (KG)-based retrieval augmented generation (RAG) has emerged as a powerful approach for supporting large language models with structured knowledge. However, there are primary challenges: (i) the lack of methods for automatic KG construction using ontology expansion for low-resource languages such as Vietnamese, (ii) the absence of systematic evaluation for knowledge retrieval strategies leveraging the hierarchical structures. In this paper, we propose an end-to-end pipeline for KG construction and retrieval strategies evaluation. In the KG construction, we employ a three-phase hybrid relation extraction pipeline: intra-batch deduplication via Union-Find, approximate cross-batch search, and LLM extraction with a centroid filter that reduces prompts combined with a five-step dual-LLM validator to prevent bloated ontology. A two-tier architecture consists of unmergeable structural nodes to preserve the document s
    
[^149]: S^3martCirc：自监督智能电路发现

    S^3martCirc: Self-supervised Smart Circuit Discovery

    [https://arxiv.org/abs/2609.00755](https://arxiv.org/abs/2609.00755)

    提出S^3martCirc，一种自监督的智能电路发现方法，将电路发现与功能解释两个阶段统一起来，解决了组件重要性与功能角色相互依存这一被传统两阶段范式忽视的问题。

    

    大语言模型（LLMs）在从文本摘要到问答等多种任务中展现出了卓越的性能。尽管具备这些能力，其黑箱特性掩盖了内部决策过程。机制可解释性（MI）旨在通过将神经网络逆向工程为人类可理解的算法来解决这一问题。当前针对大语言模型的MI方法通常遵循两阶段范式：首先识别重要组件（电路发现），其中组件通常是个体节点，如注意力头或前馈神经元；其次确定它们在特定任务中所扮演的角色（功能解释）。然而，这种顺序化的方法忽视了一个基本洞见：组件的重要性与其功能角色本质上是相互依存的。将这两个阶段统一起来面临两个关键挑战：（1）功能角色往往与特定的节点或组件绑定，限制了……（摘要在此处截断）

    arXiv:2609.00755v1 Announce Type: new  Abstract: Large Language Models (LLMs) have demonstrated remarkable performance across diverse tasks, from text summarization to question answering. Despite these capabilities, their black-box nature obscures internal decision-making processes. Mechanistic interpretability (MI) aims to address this by reverse-engineering neural networks into human-understandable algorithms. Current MI approaches for LLMs typically follow a two-stage paradigm: first identifying important components (circuit discovery), where components are typically individual nodes such as an attention head or feedforward neuron, and second determining the role they play in a certain task (functional interpretation). However, this sequential approach overlooks a fundamental insight: a component's importance and its functional role are inherently codependent. Unifying these stages presents two key challenges: (1) functional roles are often tied to specific nodes or components, limi
    
[^150]: ContextPipe：面向长程智能体的受数据库启发的上下文组装

    ContextPipe: Database-Inspired Context Assembly for Long-Horizon Agents

    [https://arxiv.org/abs/2609.00749](https://arxiv.org/abs/2609.00749)

    该论文提出 ContextPipe，将长程智能体的上下文组装类比为数据库查询执行，通过“计划-绑定-优化-执行-反馈”五阶段流水线，结合结构化数据源目录、确定性缓存感知优化器和 EXPLAIN ANALYZE 追踪，实现可审计、可重放且故障隔离的上下文管理。

    

    长程大语言模型（LLM）智能体需要上下文组装：运行时必须在严格的上下文窗口预算和对字节敏感的提示缓存的约束下，决定每次提示中包含什么内容、以何种顺序排列，以及何时压缩历史记录。在生产级智能体系统中，这类逻辑分散在提示构建器、临时性的压缩例程、缓存失效规避方案和针对各提供商的适配层中。我们认为，上下文组装在结构上与关系数据库中的查询执行同构：两者都在硬性预算下执行、利用分层缓存，并借助统计信息。我们在 ContextPipe 中采用了这一规范：一个五阶段流水线（Plan-Bind-Optimize-Execute-Feedback，即计划-绑定-优化-执行-反馈），由结构化数据源目录、确定性的缓存感知优化器和 EXPLAIN ANALYZE 追踪记录支撑。我们展示了 ContextPipe 中的上下文是可审计的、可重放的且故障隔离的。一项使用 SWE-bench Pro Qutebrowser 子集的初步评估……

    arXiv:2609.00749v1 Announce Type: new  Abstract: Long-horizon large language model (LLM) agents require context assembly: the runtime must decide what to include in each prompt, in what order, and when to compact history under a hard context-window budget and a byte-sensitive prompt cache. In production agentic systems, this logic is scattered across prompt builders, ad hoc compaction routines, cache-break workarounds, and per-provider shims. We argue that context assembly is structurally isomorphic to query execution in a relational database: both execute under a hard budget, exploit a tiered cache, and leverage statistics. We adopt this discipline in ContextPipe: a five-phase pipeline (Plan Bind Optimize Execute Feedback) backed by a structured data-source catalog, a deterministic cache-aware optimizer, and an EXPLAIN ANALYZE trace. We show that context in ContextPipe is auditable, replayable, and failure-isolated. A preliminary evaluation using the SWE-bench Pro Qutebrowser subset s
    
[^151]: 逃离冗余推理：面向推理时大语言模型的结构感知搜索

    Escaping Redundant Reasoning: Structure-Aware Search for Inference-Time LLMs

    [https://arxiv.org/abs/2609.00738](https://arxiv.org/abs/2609.00738)

    提出无需训练的结构感知搜索方法BASIN，通过将推理状态分组为盆地并惩罚重复策略来避免“推理盆地坍缩”，在固定计算预算下显著提升LLM推理时搜索性能（Game of 24上较ToT提升达22个百分点）。

    

    大语言模型（LLM）的推理时搜索往往集中在少数结构或语义相似的轨迹上，导致其他备选路径探索不足——我们将这种失败模式称为“推理盆地坍缩”（reasoning basin collapse）。我们提出了BASIN，一种无需训练、结构感知的选择方法，它将推理状态分组为盆地，并对重复访问同一策略进行惩罚，从而在固定计算预算下将搜索重新分配到真正不同的推理路径上。在相同的推理预算下，BASIN在Game of 24上比Tree of Thoughts（ToT）提升高达+22个百分点，在MuSR上提升+6.7个百分点。一种质量感知变体QA-BASIN通过在无条件多样化过度探索时保留高质量盆地，进一步提高了鲁棒性。为解释盆地感知选择何时有效，我们引入了冗余差距Δ，用于衡量搜索对正确与错误预测的集中程度差异。

    arXiv:2609.00738v1 Announce Type: new  Abstract: Inference-time search with large language models (LLMs) often concentrates on a small set of structurally or semantically similar trajectories, leaving alternatives underexplored---a failure mode we call \textit{reasoning basin collapse}. We introduce BASIN, a training-free, structure-aware selection method that groups reasoning states into basins and penalizes repeated visits to the same strategy, thereby reallocating search across genuinely distinct reasoning paths under a fixed compute budget. Under matched inference budgets, BASIN improves over Tree of Thoughts (ToT) by up to $+22$pp on Game of 24 and $+6.7$pp on MuSR. A quality-aware variant, QA-BASIN, further improves robustness by preserving high-quality basins when unconditional diversification over-explores. To explain when basin-aware selection helps, we introduce the redundancy gap $\Delta$, which measures how differently search concentrates for correct versus incorrect predic
    
[^152]: 智能体化实证资产定价：方法论基础

    Agentic Empirical Asset Pricing: Methodological Foundations

    [https://arxiv.org/abs/2609.00731](https://arxiv.org/abs/2609.00731)

    本文提出了智能体化实证资产定价（AEAP）这一新范式，为自主因子发现系统提供了参考架构、严格的因子评估标准与样本外回测方法，并通过对SEADS与五个基线系统的评估证明此类系统必须从多个维度同时加以评价。

    

    大语言模型智能体的最新进展为资产定价开辟了一种新范式，我们称之为智能体化实证资产定价：即能够自主执行科学发现过程本身的系统。我们定义了AEAP并确定了其核心构建模块。现有的评估实践仅对输出结果（因子或交易）进行回测，而非对产生这些结果的自主发现系统本身进行评估。我们聚焦于因子发现问题，贡献了一个参考架构、一套针对所发现因子的严格评估标准，以及一种对发现系统进行样本外回测的方法。作为该架构的具体实例，我们使用该标准在两个美股面板数据上将SEADS与五个重新实现的基线系统进行对比评估：结果显示没有任何单一指标能够一致地对各系统进行排序，这表明需要在多个维度上同时进行评估。随后，一项独立的滚动重复执行实验提出了一个互补性问题：发现过程本身（而非某个静态输出）是否……

    arXiv:2609.00731v1 Announce Type: new  Abstract: Recent advances in LLM agents enable a new paradigm for asset pricing, which we call Agentic Empirical Asset Pricing (AEAP): systems that autonomously conduct the scientific discovery process itself. We define AEAP and identify its core building blocks. Existing evaluation practices backtest only the outputs (factors or trades), not the autonomous discovery system that produced them. We focus on factor discovery, contributing a reference architecture, a rigorous evaluation standard for discovered factors, and a method for out-of-sample backtesting the discovery system. As a concrete instance of that architecture, we evaluate SEADS against five re-implemented baselines on two US equity panels using this standard: no single metric ranks the systems consistently, motivating evaluation on multiple axes at once. A separate rolling re-execution then asks the complementary question of whether the discovery process itself, not one static output,
    
[^153]: SOVER：通过LLM辅助的SMT验证对优化问题重构进行形式化认证

    SOVER: Formal Certification of Optimization Reformulations via LLM-Assisted SMT Verification

    [https://arxiv.org/abs/2609.00728](https://arxiv.org/abs/2609.00728)

    SOVER框架将语义映射与形式化验证分离，利用Z3和dReal等SMT求解器对LLM生成的优化问题重构进行形式化认证，并在NLEquiv-150基准上实现了149/150的正确分类。

    

    大语言模型（LLMs）在跨建模语言翻译和重构复杂数学优化问题方面展现出了巨大的潜力。然而，仅通过经验性的求解器执行来验证此类转换是不可靠的，因为求解器的结果可能受到局部极小值、结构性超时、数值伪影以及不同表述之间细微语义差异的影响。我们提出了SOVER，这是一个LLM辅助的SMT框架，它将语义映射与形式化认证分离：Z3用于检查混合整数线性表述的域交叉可行性和全局目标序保持性，而dReal则为连续非线性表述提供容差感知的可行性/范围检查和ε-argmin检查。我们还引入了NLEquiv-150，这是一个公开的基准数据集，包含100对等价和50对故意构造的困难非等价非线性重构对。利用LLM提取的映射，SOVER成功分类了149/150对。

    arXiv:2609.00728v1 Announce Type: new  Abstract: Large Language Models (LLMs) have shown remarkable promise in translating and reformulating complex mathematical optimization problems across modeling languages. However, validating such transformations through empirical solver executions alone is unreliable, as solver outcomes may be affected by local minima, structural timeouts, numerical artifacts, and subtle semantic divergence between formulations. We introduce SOVER, an LLM-assisted SMT framework that separates semantic mapping from formal certification: Z3 checks domain cross-feasibility and global objective-order preservation for mixed-integer linear formulations, while dReal provides tolerance-aware feasibility/range and $\epsilon$-argmin checks for continuous nonlinear formulations. We also introduce NLEquiv-150, a public benchmark of 100 equivalent and 50 deliberately hard non-equivalent nonlinear reformulation pairs. With LLM-extracted mappings, SOVER classifies 149/150 pairs
    
[^154]: 听到却未被重视：音频-语言模型中的副语言信息编码与丢失

    Heard but Not Heeded: Paralinguistic Information Encoding and Loss in Audio-Language Models

    [https://arxiv.org/abs/2609.00727](https://arxiv.org/abs/2609.00727)

    该研究通过对四个开源音频语言模型的机制分析首次系统揭示了：尽管模型在音频编码器后期强烈编码了说话风格等副语言信息，但这些信息在传递至输出的过程中持续丢失，暴露出当前音频语言模型“听得见却不重视”语气情感的关键缺陷。

    

    音频语言模型旨在理解语音，但它们是否能捕捉“怎么说”而不只是“说了什么”，目前尚不清楚。我们对四个开源模型（Whisper-large-v2、Qwen2-Audio-7B Instruct、Qwen2.5-Omni-7B 和 Chroma-4B）中的副语言信息进行了机制分析，使用了具有受控说话风格的 Expresso 数据集。我们结合了中心化核对齐（CKA）、采用留一说话人评估的线性探针、开放式语气预测以及内容-韵律泄漏指标，以追踪风格信息如何从音频编码器传递到最终输出。所有模型都在编码器后期（即音频编码器顶部的三分之一次层）强烈编码了说话风格，但这些信息在到达输出之前会持续退化。投影器在重塑表示几何结构的同时并未移除信息，而解码器则因架构不同在风格保留程度上存在差异。

    arXiv:2609.00727v1 Announce Type: cross  Abstract: Audio language models are designed to understand speech, yet it remains unclear whether they capture how something is said beyond what is said. We present a mechanistic analysis of paralinguistic information in four open source models, Whisper-large-v2, Qwen2-Audio-7B Instruct, Qwen2.5-Omni-7B, and Chroma-4B, using the Expresso dataset with controlled speaking styles. We combine centered kernel alignment, linear probing with leave one speaker out evaluation, open ended tone prediction, and a content prosody leakage metric to trace how style information moves from the audio encoder to the final output. All models strongly encode speaking style in the late encoder, that is, the top third of the audio encoder's layers, but this information is consistently degraded before reaching the output. The projector reshapes representation geometry without removing information, while decoders differ in how much style they preserve depending on archi
    
[^155]: 压缩驾驶策略中能力损失与恢复的闭环评估

    A Closed-Loop Evaluation of Capability Loss and Recovery in Compressed Driving Policies

    [https://arxiv.org/abs/2609.00718](https://arxiv.org/abs/2609.00718)

    本文提出一种分阶段闭环评估方法，追踪驾驶策略在剪枝、蒸馏、量化等压缩流程中的能力损失与恢复，并发现结构化剪枝是驾驶能力首次受损的环节。

    

    许多汽车和出行公司会在内存和功耗受限的嵌入式计算机上部署学习得到的驾驶策略。剪枝、知识蒸馏和量化是减小这些策略体积和推理成本的标准方法。然而，这些方法通常仅通过汇总的数值分数进行评估，而此类分数可能无法反映策略在与其他道路使用者交互时安全驾驶的能力。在本研究中，我们提出了一种分阶段的闭环评估方法，用于跟踪驾驶策略在压缩流程中各阶段的表现。我们将驾驶任务建模为部分可观测马尔可夫决策过程（POMDP），并在 Gym-Duckietown 中使用近端策略优化（PPO）训练了一个信念状态策略。随后，我们提取执行者网络，逐阶段进行压缩，并在五个驾驶课程上进行评估。结果表明，结构化剪枝是驾驶能力首次出现损失的环节（原文在此处截断）。

    arXiv:2609.00718v1 Announce Type: new  Abstract: Many automobile and mobility companies deploy learned driving policies on embedded computers with limited memory and power. Pruning, knowledge distillation, and quantization are the standard methods to reduce the size and the inference cost of these policies. However, these methods are commonly assessed by aggregate numerical scores, and such scores may not reflect the ability of the policy to drive safely when interacting with other road users. In this study, we propose a stage-wise closed-loop evaluation approach to follow a driving policy through a compression pipeline. We formulate the driving task as a partially observable Markov decision process (POMDP) and train a belief-state policy with proximal policy optimization (PPO) in Gym-Duckietown. We then extract the actor, compress it one stage at a time, and evaluate it on five driving curricula. We show that structured pruning is the stage at which the driving capability is first los
    
[^156]: ChatDev 2.0：一个用于开发一切的无代码多智能体平台

    ChatDev 2.0: A No-Code Multi-Agent Platform for Developing Everything

    [https://arxiv.org/abs/2609.00714](https://arxiv.org/abs/2609.00714)

    ChatDev 2.0（DevAll）是一个兼具高表达性与易用性的无代码多智能体平台，通过声明式可执行图抽象与循环感知执行引擎支持异构智能体间的动态循环交互，并提供集成可视化界面，让用户无需编写代码即可构建、运行、监控和检查多智能体系统（包括人在回路步骤）。

    

    基于大语言模型（LLM）的多智能体系统（MAS）在解决复杂任务方面展现出强大潜力，但其开发过程面临两难抉择：代码框架表达能力强但工程开发成本高昂，而无代码构建工具虽然简化了构建过程，却将智能体交互限制在作者预定义的工作流之中。我们提出了 ChatDev 2.0：DevAll（以下简称 DevAll），一个用于构建、执行和检查异构多智能体系统的无代码平台，兼具高表达性与易用性。在表达性方面，DevAll 将声明式可执行图抽象与支持循环感知的执行引擎相结合，使异构智能体以及动态和循环的交互能够在单一框架内被表示和执行。在易用性方面，一个集成的可视化界面让用户能够完全无需编写代码地构建、运行、监控和检查多智能体系统，包括人在回路（human-in-the-loop）的步骤。实验证明 DevAll 能够复现（摘要在此处截断）。

    arXiv:2609.00714v1 Announce Type: new  Abstract: Large language model (LLM)-based multi-agent systems (MAS) have shown strong potential for solving complex tasks, yet their development forces a tradeoff: code frameworks are expressive but engineering-intensive, while no-code builders simplify authoring but constrain agent interactions to author-defined workflows. We present ChatDev 2.0: DevAll (hereafter DevAll), a no-code platform for building, executing, and inspecting heterogeneous MAS that delivers both high expressiveness and ease of use. In terms of expressiveness, DevAll pairs a declarative executable graph abstraction with a cycle-aware execution engine, so that heterogeneous agents and dynamic and cyclic interactions can be represented and executed within a single framework. For ease of use, an integrated visual interface lets users author, run, monitor, and inspect MAS, including human-in-the-loop steps, entirely without writing code. Experiments demonstrate that DevAll repro
    
[^157]: 差分隐私配对表格-图像多模态合成

    Differentially Private Paired Table-Image Multimodal Synthesis

    [https://arxiv.org/abs/2609.00708](https://arxiv.org/abs/2609.00708)

    提出DP-TabImage框架，通过将私有概率图模型与经DP-SGD训练的表格条件扩散模型相结合，实现了差分隐私下表格-图像配对多模态数据的隐私保护合成。

    

    差分隐私（DP）合成技术已在表格数据和图像数据上分别得到广泛研究，然而许多真实世界的数据集包含与多变量表格记录配对的图像。在差分隐私约束下合成此类数据尤其具有挑战性，因为这两种模态适合不同的隐私学习机制，同时还必须保留它们之间的依赖关系。为了应对这一挑战，我们提出了DP-TabImage，一个面向隐私配对合成的模态专用框架。DP-TabImage通过实例化分解式 $p(x,y)=p_T(y)p_I(x\;|\;y)$ 来实现：使用私有的概率图模型建模多变量表格分布，并使用经DP-SGD训练的以表格为条件的扩散模型建模条件图像分布。为了在裁剪和加噪的梯度下促进条件学习，我们进一步在私有的表格-图像原型上对模型进行预训练，将私有构建的属性条件图像与表格向量配对。

    arXiv:2609.00708v1 Announce Type: cross  Abstract: Differentially private (DP) synthesis has been extensively studied for tabular and image data separately, yet many real-world datasets contain images paired with multivariate tabular records. Synthesizing such data is particularly challenging under DP, as the two modalities favor different private learning mechanisms while their dependence must also be preserved. To address this challenge, we propose DP-TabImage, a modality-specialized framework for private paired synthesis. DP-TabImage instantiates the factorization $p(x,y)=p_T(y)p_I(x\;|\;y)$ using a private Probabilistic Graphical Model for the multivariate table distribution and a table-conditioned diffusion model trained with DP-SGD for the conditional image distribution. To facilitate conditional learning under clipped and noisy gradients, we further pretrain the model on private table-image prototypes, pairing privately constructed attribute-conditioned images with tabular vecto
    
[^158]: 价值超越语言模型：检测写作中的原创贡献

    Value Over Language Model: Detecting Original Contribution in Writing

    [https://arxiv.org/abs/2609.00700](https://arxiv.org/abs/2609.00700)

    提出了一种无需训练、不评分表面文本的新框架“价值超越语言模型”（Value Over Language Model），通过在不同粒度上提取文档内容并用LLM重建文档，来衡量人在语言模型易于生成的内容之上所贡献的原创价值。

    

    大语言模型（LLM）已在各类写作任务中被迅速采用，这推动了检测LLM生成文本工具的发展。然而，这些工具主要衡量的是文档表面文本中有多少是由LLM撰写的，而并非从根本上设计用于衡量文档中的信息内容或思想有多少源自LLM本身，而非由用户在提示词中提供。在本工作中，我们设计了一个框架，用于衡量一个人在语言模型本身能够轻松生成的内容之上所增加的价值。该方法不需要训练或标注数据，也从不为文档的表面文本打分，从而使其免受文体混淆因素的影响。相反，该方法以递增的粒度级别提取文档内容，使用LLM从每个部分表示中重建文档，并将这些重建结果与仅根据任务描述生成的重建结果进行比较。我们将这一框架称为“价值超越语言模型”（Value Over Language Model）。

    arXiv:2609.00700v1 Announce Type: new  Abstract: LLMs have been rapidly adopted across writing tasks, prompting the development of tools for detecting LLM-generated text. Yet, these tools largely measure how much of a document's surface text was written by an LLM and aren't fundamentally designed to measure how much of the information content or ideas originated from the LLM itself rather than being supplied by the user in the prompt. In this work, we design a framework that measures how much value a person adds on top of what a language model could have easily produced by itself. The method requires no training or labeled data and never scores the document's surface text, insulating it from stylistic confounders. Instead, it extracts the document's content at increasing levels of granularity, uses an LLM to reconstruct the document from each partial representation, and compares these reconstructions with those produced from the task description alone. We call this framework Value Over
    
[^159]: 预测编码网络中隐状态优化顺序的研究

    A Study of Hidden-State Optimization Order in Predictive Coding Networks

    [https://arxiv.org/abs/2609.00686](https://arxiv.org/abs/2609.00686)

    该论文提出一种边界优先的隐状态优化顺序——先协调块边界处的隐状态、再细化块内表示，显著提升了预测编码网络在CIFAR-10上的准确率并增强了早期层的特征学习。

    

    局部学习方法为端到端反向传播提供了一种替代方案，但其非结构化的局部目标可能导致深度网络的特征学习能力较弱。我们研究隐状态优化的顺序是否能够解决这一局限。我们提出了一种边界优先的推理调度方法，该方法将模型划分为若干块，首先协调各块边界处的隐状态，然后再细化每块内部的表示。我们在预测编码网络（PCNs）中实现了这一调度方法，预测编码网络是一种局部学习框架，其中隐层活动和预测误差在推理过程中被显式地暴露出来。在CIFAR-10数据集上，这种边界优先的预测编码实现在标准参数化下比标准预测编码提高了9.77%的准确率，在μ参数化下提高了5.51%的准确率。诊断分析进一步显示出更多的早期层非平凡更新、更低的初始到最终CKA相似度，以及更多的……

    arXiv:2609.00686v1 Announce Type: cross  Abstract: Local learning methods offer an alternative to end-to-end backpropagation, but their unstructured local objectives can produce weak feature learning in deep networks. We study whether the order of hidden-state optimization can address this limitation. We propose a boundary-first inference schedule that partitions a model into chunks, first coordinates hidden states at chunk boundaries, and then refines representations within each chunk. We instantiate this schedule in predictive coding networks (PCNs), a local-learning framework in which hidden activities and prediction errors are explicitly exposed during inference. On CIFAR-10, the resulting boundary-first predictive-coding instantiation improves accuracy over standard predictive coding by $9.77\%$ under a standard parametrization and by $5.51\%$ under a $\mu$-parametrization. Diagnostic analyses further show more non-trivial early-layer updates, lower initial-to-final CKA, and more 
    
[^160]: 基于图像生成的视觉框架用于新闻立场检测

    Visual Framing for News Stance Detection via Image Generation

    [https://arxiv.org/abs/2609.00685](https://arxiv.org/abs/2609.00685)

    该论文提出VFStance方法，通过图像生成技术将新闻文章中隐含的立场线索转化为视觉框架，使立场信号更加明确显著，有效提升了新闻立场检测性能，并具有超越自动化立场检测的应用潜力。

    

    文章级新闻立场检测旨在识别新闻文章对社会议题所持的观点倾向。尽管立场检测技术不断进步且对构建可信媒体环境具有重要意义，但新闻文章带来了独特的挑战，因为其立场往往是隐含的，通过新闻报道框架微妙地传达，并嵌入在冗长且结构复杂的文本之中。为了应对这些挑战，我们提出了VFStance，该方法利用视觉框架，通过图像生成将隐含的立场线索变得更加明确。在评估实验中，我们证明了VFStance相对于现有方法的有效性，以及视觉框架对其性能的贡献。最后，一项在基于片段的新闻消费场景下开展的受控用户研究（N=200）进一步表明，VFStance能够使立场信号在视觉上更加显著，并凸显了其在自动化立场检测之外的潜在应用价值。

    arXiv:2609.00685v1 Announce Type: cross  Abstract: Article-level news stance detection aims to identify the perspective of news articles toward social issues. Despite advances in stance detection and its importance for trustworthy media environments, news articles pose distinct challenges because their stances are often implicit, subtly conveyed through journalistic framing, and embedded in long, structurally complex texts. To address these challenges, we introduce VFStance, which leverages visual framing to make implicit stance cues more explicit via image generation. In evaluation experiments, we demonstrate the effectiveness of VFStance over existing methods and the contribution of visual framing to its performance. Finally, a controlled user study (N=200) in a snippet-based news consumption setting further demonstrates that VFStance can make stance signals visually salient and highlights its potential use beyond automated stance detection.
    
[^161]: 边缘AI语言模型的三重底线可持续性：小型语言模型与量化大语言模型的比较

    Triple-Bottom-Line Sustainability of Language Models for Edge AI: A Comparison Between SLMs and Quantized LLMs

    [https://arxiv.org/abs/2609.00665](https://arxiv.org/abs/2609.00665)

    该研究提出了一个基于经济、环境、社会三重底线的可复现整体可持续性评分（HSS），通过对比原生训练的小型语言模型与多种量化后的大语言模型共30种配置在能力、效率和安全性方面的表现，来评估哪种方案更适合作边缘AI部署。

    

    边缘AI模型的选择通常由单一的孤立指标驱动——准确率、延迟、内存、能耗或安全性，然而一个可部署的语言模型必须平衡这五个方面。我们的工作重点在于回答这样一个问题：原生训练的小型语言模型（SLM）与通过训练后量化压缩的大语言模型（LLM），哪一种能为边缘部署提供更可持续的权衡方案。我们引入了一个可复现的整体可持续性评分（HSS），该评分围绕三重底线构建：经济支柱衡量能力与系统效率，环境支柱衡量运行时的GPU能耗，社会支柱衡量对有害提示的鲁棒性。五个BF16精度的SLM和五个LLM在不同量化方法（BF16、INT8、NF4 4位、GPTQ 4位和GGUF Q4）下，共产生了30种实测配置。能力通过五个零样本基准进行评估；效率使用延迟、吞吐量、峰值显存和能耗来衡量；安全性[摘要内容被截断]

    arXiv:2609.00665v1 Announce Type: new  Abstract: Edge-AI model selection is commonly driven by one isolated metric - accuracy, latency, memory, energy, or safety, even though a deployable language model must balance all five. Our work focuses on answering the question whether na- tively trained small language models (SLMs) or large language models (LLMs) compressed through post-training quantization offer the more sustainable edge- deployment trade-off. We introduce a reproducible Holistic Sustainability Score (HSS) organized around the triple bottom line: an economic pillar for capability and systems efficiency, an environmental pillar for operational GPU energy and a social pillar for harmful-prompt robustness. Five BF16 SLMs and five LLMs under different quantization approaches - BF16, INT8, NF4 4-bit, GPTQ 4-bit, and GGUF Q4 produce 30 measured configurations. Capability is assessed on five zero-shot benchmarks; efficiency uses latency, throughput, peak VRAM and energy; and safety 
    
[^162]: 具有稀疏上下文与共享预算的漂移感知大语言模型路由

    Drift-Aware LLM Routing with Sparse Contexts and Shared Budgets

    [https://arxiv.org/abs/2609.00662](https://arxiv.org/abs/2609.00662)

    本文提出漂移感知稀疏路由方法（DRS），针对多模型大语言服务中提示表示高维稀疏且请求分布与模型能力持续漂移的挑战，通过滚动审计窗口、悲观奖励与乐观成本估计以及在线影子价格更新，实现了满足共享预算约束的非平稳上下文路由。

    

    多模型语言服务必须在为每个请求进行路由的同时，维持工作负载级别的计算、延迟、内存或资金成本预算。有两个特性使这一问题比静态模型选择困难得多：其一，提示表示是高维的，因此嵌入方向中可能只有很小的子集能够预测某个模型的增量价值；其二，请求组合与模型前沿会在系统发布、微调、量化变更和更新之后发生漂移。我们将该问题形式化为带有多重背包约束的非平稳稀疏上下文路由，并引入一个可选的影子审计流，用于在多个模型上评估一小部分提示。我们提出了漂移感知稀疏路由方法（DRS）：该策略从滚动的审计窗口中估计奖励与资源使用情况，采用悲观的奖励估计与乐观的成本估计进行路由决策，在线更新资源的影子价格，并在最终提交决策之前施加硬性计量。

    arXiv:2609.00662v1 Announce Type: new  Abstract: A multi-model language service must route each request while preserving workload-level budgets for compute, latency, memory, or monetary cost. Two features make this problem materially harder than static model selection. Prompt representations are high dimensional, so only a small subset of embedding directions may predict the incremental value of a model, and both the request mix and the model frontier drift after launches, fine-tunes, quantization changes, and system updates. We formulate nonstationary sparse contextual routing with multiple knapsack constraints and an optional shadow-audit stream that evaluates a small fraction of prompts on several models.   We propose Drift-Aware Sparse Routing (DRS). The policy estimates reward and resource use from a rolling audit window, routes using pessimistic reward and optimistic cost estimates, updates resource shadow prices online, and applies a hard meter before commitment. The analysis se
    
[^163]: SciTrue：在NTCIR SciClaimEval任务中使用前沿与开源语言模型进行可靠的科学论断验证

    SciTrue: Reliable Scientific Claim Validation with Frontier and Open Language Models at the NTCIR SciClaimEval Task

    [https://arxiv.org/abs/2609.00654](https://arxiv.org/abs/2609.00654)

    SciTrue团队通过在统一诚实的逐样本协议下对十一个前沿及开源多模态模型进行基准测试，并结合轻量透明的后处理，在NTCIR SciClaimEval科学论断验证任务的官方盲测排行榜上以明显优势夺得第一。

    

    我们描述了SciTrue团队参与NTCIR-19 SciClaimEval任务两个子任务的情况，该任务要求系统根据论文中的表格和图表来验证科学论断。我们没有调优单一模型，而是在一个诚实、逐样本的统一协议下对十一个前沿及开源多模态模型进行基准测试，并将它们与轻量、透明的后处理相结合。在官方盲测排行榜上，SciTrue在四个证据类别/子任务组合中的三个以明显优势获得第一，并在第四个组合的主要指标上并列第一。三个发现解释了这一结果。第一，强大的指令微调模型已经具备竞争力：Claude Opus 4.8和Gemma-4-31B均超过了最强的公开基线o4-mini，而GPT-5.5和Claude Fable 5在两个子任务中均处于领先地位（在子任务2上达到97.7）。第二，任务的配对结构是最大的杠杆：一个无泄漏的……

    arXiv:2609.00654v1 Announce Type: new  Abstract: We describe the SciTrue team's participation in both subtasks of the NTCIR-19 SciClaimEval task~\cite{sciclaimeval}, which asks systems to verify scientific claims against the tables and figures of a paper. Rather than tuning a single model, we benchmark eleven frontier and open multimodal models under one honest, per-sample protocol and combine them with light, transparent post-processing. On the official, blind test leaderboard (Section~\ref{sec:results}), SciTrue placed first by a clear margin in three of the four evidence-category/subtask combinations, and tied for first on the primary metric in the fourth. Three findings explain the result. First, strong instruction-tuned models are already competitive: Claude Opus~4.8 and Gemma-4-31B each exceed the strongest public baseline (o4-mini), and GPT-5.5 and Claude Fable~5 lead both subtasks (97.7 on Subtask~2). Second, the task's pairing structure is the largest lever: a \emph{leak-free 
    
[^164]: EEG-AS：通过行为重构实现EEG基础模型的实例级选择

    EEG-AS: Instance-Level Foundation Model Selection for EEG Foundation Models via Behavior Reconstruction

    [https://arxiv.org/abs/2609.00653](https://arxiv.org/abs/2609.00653)

    该论文提出EEG-AS框架，将EEG基础模型选择形式化为实例级算法选择问题，通过锚定模型的特权预测标记重构不可获得的基础模型行为，从而为每个EEG实例自动选择最合适的基础模型。

    

    脑电图（EEG）是一种测量神经活动的非侵入性技术，已在神经科学领域得到广泛应用。EEG基础模型的最新进展使其在多种神经解码任务中均能取得出色的性能。然而，没有任何单一基础模型能在所有数据集或个体EEG实例上始终保持最佳表现，而实例级的模型选择在很大程度上仍处于未被探索的状态。为解决这一局限性，我们将EEG基础模型选择构建为一个实例级算法选择（Algorithm Selection, AS）问题。我们提出了EEG-AS，这是一个实例级算法选择框架，它利用推理时可获得的潜在EEG嵌入、手工设计的神经生理学特征以及锚定基础模型来刻画每个EEG实例。在训练过程中，EEG-AS学习从以锚定基础模型为条件的特权预测标记中重构不可获得的基础模型行为。

    arXiv:2609.00653v1 Announce Type: cross  Abstract: Electroencephalography (EEG) is a non-invasive technique for measuring neural activity and has been widely used in neuroscience applications. Recent advances in EEG foundation models have enabled strong performance across diverse neural decoding tasks. However, no single foundation model consistently performs best across datasets or individual EEG instances, while instance-level model selection remains largely unexplored. To address this limitation, we formulate EEG foundation model selection as an instance-level Algorithm Selection (AS) problem. We propose \textbf{EEG-AS}, an instance-level algorithm selection framework that characterizes each EEG instance using inference-available latent EEG embeddings, handcrafted neurophysiological features, and an anchor foundation model. During training, EEG-AS learns to reconstruct unavailable foundation-model behaviors from privileged prediction tokens conditioned on an anchor foundation model,
    
[^165]: 自我报告并非验证：进化搜索中LLM操作者的环境锚定审计

    Self-Reports Are Not Verification: Environment-Grounded Auditing of LLM Operators in Evolutionary Search

    [https://arxiv.org/abs/2609.00652](https://arxiv.org/abs/2609.00652)

    本文提出首个环境锚定的LLM操作者审计框架，通过为进化式Contexto搜索中的每个中间提议赋予精确结果，实证证明模型自我报告不可作为验证依据——操作者将成功率夸大4.8至9.3倍，且关于置信度校准、理由传递和适应度选择的三个假设全部被证伪。

    

    语言模型智能体日益频繁地提出行动、观察外部反馈并解释自身行为。它们的置信度和理由是便捷的监控信号，但便捷并不等于验证。我们引入了一种环境锚定的审计方法，其中每个中间提议都会获得精确的结果反馈。一个语言模型操作进化式Contexto搜索，其反馈函数无需人工标注即可为每个有效猜测分配精确排名。在涵盖五种配置和三个模型系列的200次运行中，四种报告配置共产生了12,249份自我报告。我们检验了三个假设：所述置信度是经过校准的、继承的理由会影响后续提议、以及基于适应度的选择能够提升报告质量。这三个假设全部失败。操作者将进入前100名的成功率夸大了4.8至9.3倍，而校准能力与区分能力在不同模型系列之间出现了分离。对754个继承理由的受控干预（摘要在此处被截断）

    arXiv:2609.00652v1 Announce Type: new  Abstract: Language model agents increasingly propose actions, observe external feedback, and explain their own behavior. Their confidence and rationales are convenient monitoring signals, but convenience is not verification. We introduce an environment-grounded audit in which every intermediate proposal receives an exact outcome. A language model operates an evolutionary Contexto search whose feedback function assigns every valid guess an exact rank without human annotation. Across 200 runs spanning five configurations and three model families, four reporting configurations produce 12,249 self-reports. We test three assumptions: stated confidence is calibrated, inherited rationales affect later proposals, and fitness-based selection improves report quality. All three fail. Operators overstate top-100 success by factors of 4.8 to 9.3, while calibration and discrimination dissociate across model families. Controlled interventions on 754 inherited ra
    
[^166]: DramaChain Bench：短剧生成的端到端基准测试

    DramaChain Bench: An End-to-End Benchmark for Short-Drama Generation

    [https://arxiv.org/abs/2609.00646](https://arxiv.org/abs/2609.00646)

    DramaChain Bench是首个评估短剧制作全流程各阶段（剧本、分镜、关键帧、镜头视频到成片）的端到端基准测试，通过统一的63维度评估体系衡量各阶段对剧本意图的忠实度以及多集成片的连贯性。

    

    商业短剧制作遵循一个多阶段的流程链：剧本、分镜、关键帧图像、镜头级视频以及成片短剧。现有的大多数基准测试仅使用预先编写好的输入而非真实的上游流水线输出来评估视频生成阶段。这使得两个关键问题无法得到解答：每个阶段是否忠实于原始剧本的意图（而不仅仅是其直接的输入提示词），以及不同的镜头在组装成多集发布后是否保持连贯。我们提出了DramaChain Bench，这是首个能够评估完整制作链中每个阶段的短剧基准测试。它建立在三个共享同一维度体系（DramaChain Dimensions）的自研系统之上：五个评估轴在每个阶段进行实例化，最终细分为63个叶子维度。DramaChain Agent在工作流程和成片短剧质量两方面均与商业短剧平台进行了校准……

    arXiv:2609.00646v1 Announce Type: new  Abstract: Commercial short-drama production follows a multi-stage chain: script, storyboard, keyframe imagery, shot-level video, and the finished short drama. Most existing benchmarks evaluate solely the video-generation stage using pre-authored inputs instead of real upstream pipeline outputs. This leaves two critical questions unanswerable: whether each stage adheres to the original script intent (rather than only its immediate input prompt), and whether disparate shots remain coherent after assembly into multi-episode releases. We present DramaChain Bench, the first short-drama benchmark that evaluates every stage of the complete production chain. It is built upon three in-house systems sharing one dimension system, DramaChain Dimensions: five evaluation axes instantiated at every stage, resolving into 63 leaf dimensions. DramaChain Agent is calibrated against commercial short-drama platforms in both workflow and finished short-drama quality, e
    
[^167]: REVISE：面向智能体工作流中在线修订的有效性引导恢复机制

    REVISE: Validity-Guided Recovery for Online Revisions in Agent Workflows

    [https://arxiv.org/abs/2609.00643](https://arxiv.org/abs/2609.00643)

    REVISE 提出了一种有效性引导的细粒度恢复运行时，通过将修订差异与已记录的数据和控制依赖求交并在部分执行的 DAG 上传播影响，从而在并发智能体工作流中平衡正确性与效率，避免丢弃未受影响的进展。

    

    智能体修订在并发执行过程中暴露出一个根本性的正确性与效率之间的权衡。丢弃正在进行的工作可以保持最新版本的正确性，但会浪费可能仍然有效的进展；而复用先前的工作虽然保持了效率，却有将过时状态传播到输出和工具效果中的风险。现有的恢复策略以粗粒度的策略不平衡地解决这一权衡：它们要么偏重效率，允许可能已过时的工作继续执行；要么偏重正确性，通过重启整个工作流或从最早的冲突处重新计算线性后缀，从而丢弃了未受影响的进展。我们提出了 REVISE，一个面向结构化智能体工作流细粒度恢复的有效性引导运行时系统。当修订到来时，REVISE 首先将修订的差异（delta）与已记录的数据依赖和控制依赖进行求交，并将由此产生的影响在部分执行的有向无环图（DAG）上传播，以识别……（原摘要在此处截断）

    arXiv:2609.00643v1 Announce Type: new  Abstract: Agent revisions expose a fundamental correctness--efficiency trade-off during concurrent execution. Discarding ongoing work preserves latest-version correctness but wastes progress that may remain valid, whereas reusing prior work preserves efficiency but risks propagating stale state into outputs and tool effects. Existing recovery strategies resolve this trade-off in an imbalanced way with coarse-grained policies: they either favor efficiency by allowing potentially stale work to continue, or favor correctness by restarting the workflow or recomputing a linear suffix from the earliest conflict, thereby discarding unaffected progress. We present \textsc{Revise}, a validity-guided runtime for fine-grained recovery in structured agent workflows. When a revision arrives, \textsc{Revise} first intersects its delta with recorded data and control dependencies and propagates the resulting impact through the partially executed DAG to identify a
    
[^168]: TUTTI：通过全合成数据实现可泛化的音频到乐谱转录

    TUTTI: Toward generalizable audio-to-score transcription via fully synthesized data

    [https://arxiv.org/abs/2609.00640](https://arxiv.org/abs/2609.00640)

    该论文提出TUTTI预训练范式，利用符号音乐生成模型构建大规模纯合成的多乐器音频-乐谱配对数据，突破真实标注数据稀缺的限制，从而实现可泛化的音频到乐谱转录。

    

    可泛化的音频到乐谱转录从根本上受到高质量真实世界配对数据严重稀缺的制约。仅依赖现有的人工标注数据集往往会限制音频到乐谱（A2S）模型的泛化能力，使其效果主要局限于单一乐器领域。为了打破对稀缺真实世界数据的依赖，我们提出了TUTTI（基于合成多乐器数据训练的统一音频到乐谱转录Transformer），这是一种由纯合成的大规模数据集驱动的预训练范式。我们不使用人工创作的乐谱，而是利用符号音乐生成模型生成海量、高度可扩展的多乐器语料库，并创建具有表现力声学特征的音频-乐谱配对数据。利用生成的数据，我们采用标准的Transformer编码器-解码器架构。我们通过实验证明，对统一的注意力……（摘要原文在此处截断）

    arXiv:2609.00640v1 Announce Type: cross  Abstract: Generalizable Audio-to-Score (A2S) transcription is fundamentally constrained by the severe scarcity of high-quality, real-world paired data. Relying solely on existing human-annotated datasets often restricts the generalization of A2S models, limiting their efficacy primarily to single-instrumentation domains. To break this dependency on scarce real-world data, we introduce TUTTI (Transformer for Unified audio-To-score Transcription trained on Synthetic multi-Instrumentation Data), a pre-training paradigm driven by a purely synthetic, large-scale dataset. Rather than using human-composed scores, we leverage a symbolic music generation model to generate a massive, highly scalable multi-instrumentation corpus and create audio-score pairs with expressive acoustic characteristics. Capitalizing on the generated data, we employ a standard Transformer encoder-decoder architecture. We empirically demonstrate that pre-training a unified attent
    
[^169]: 打破结构同一性：秩异构下的个性化联邦LoRA微调

    Breaking the Structural Identity: Personalized Federated LoRA Fine-tuning under Rank Heterogeneity

    [https://arxiv.org/abs/2609.00632](https://arxiv.org/abs/2609.00632)

    提出FedRoRA框架，通过将LoRA适配解耦为共享的全局方向与个性化的按秩幅值，在秩异构联邦学习场景下实现细粒度的客户端个性化微调，从而同时应对资源异构与数据异构的双重挑战。

    

    大语言模型（LLM）在众多领域取得了显著成功，但其在隐私敏感的分布式数据集上的适配仍然是一个挑战。虽然联邦学习（FL）与低秩适配（LoRA）相结合为协同微调提供了一种资源高效的范式，但实际部署受到资源异构性与数据异构性双重挑战的阻碍。现有的秩异构方法主要致力于弥合聚合过程中的维度不匹配问题，但通常为共享相同秩的所有客户端提供统一的全局模型，无法在非独立同分布（non-IID）场景下捕捉客户端特有的特征。本文提出了FedRoRA（联邦按秩个性化LoRA），这是一种能够在秩异构联邦中实现细粒度个性化的新型框架。FedRoRA将适配过程解耦为共享的全局方向和个性化的按秩幅值……

    arXiv:2609.00632v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have achieved remarkable success across diverse domains, but their adaptation to privacy-sensitive, distributed datasets remains a challenge. While Federated Learning (FL) combined with Low-Rank Adaptation (LoRA) provides a resource-efficient paradigm for collaborative fine-tuning, practical deployments are hindered by the dual challenges of resource heterogeneity and data heterogeneity. Existing rank-heterogeneous methods primarily focus on bridging dimension mismatches for aggregation but typically provide a unified global model for all clients sharing the same rank, failing to capture client-specific features in non-IID scenarios. In this paper, we propose FedRoRA (Federated Rank-wise Personalized LoRA), a novel framework that enables fine-grained personalization within rank-heterogeneous federations. FedRoRA decouples adaptation into shared global directions and personalized rank-wise magnitudes governe
    
[^170]: 限制而非重训练：用于零样本航空图像分割的推理时视觉语言模型引导

    Restrict, Don't Retrain: Inference-Time VLM Guidance for Zero-Shot Aerial Segmentation

    [https://arxiv.org/abs/2609.00628](https://arxiv.org/abs/2609.00628)

    提出一种推理时引导方法，利用单张消费级GPU上的视觉语言模型为冻结的零样本分割基础模型提供类别筛选与小型物体定位指引，无需重训练即可提升航空图像分割效果，并生成可独立查验的结构化证据。

    

    全球福祉往往取决于对航空和卫星图像的正确解读。基于此类图像采取行动（如绘制被洪水淹没的区域、农作物范围或受损基础设施）需要进行像素级分割，以确保精确的类别定位。预训练的通用基础模型在直接应用时，常常会遗漏重要特征，且并非总能找到给定场景中的所有类别，从而忽略了那些最为关键的小型物体。我们使用一台运行视觉语言模型（VLM）的消费级GPU来提供这种缺失的引导，在提升分割效果的同时，生成结构化、可审计的证据来支撑结果，且这些证据本身也可被独立查验。我们融合了三种方法：为每个像素进行标注的冻结基础模型，以及两次对VLM的查询——一次用于筛选出重要的类别，另一次用于定位基础模型所遗漏的小型物体。在四个航空数据集上的评估显示了一致的性能提升。

    arXiv:2609.00628v1 Announce Type: cross  Abstract: Global welfare often depends on the correct interpretation of aerial and satellite imagery. Acting on such imagery (mapping flooded ground, crop extent, or damaged infrastructure) demands pixel-level segmentation to ensure perfect class localization. Pretrained general foundation models, when applied directly, often miss important features and cannot always find all the classes belonging to a given scene, overlooking smaller objects that matter most. We use a single consumer-grade GPU running a vision-language model (VLM) to supply this missing guidance, improving segmentation while producing structured, auditable evidence that drives the result and can be inspected on its own. We fuse three approaches: the frozen foundation model that labels every pixel, and two queries to a VLM, one to choose the classes that matter, and one to locate the small objects the base model misses. Evaluating across four aerial datasets, we see consistent g
    
[^171]: 控制-数据流分离：多智能体大语言模型中的稳定提示词优化

    Control-Data Flow Separation: Stable Prompt Optimization in Multi-Agent LLMs

    [https://arxiv.org/abs/2609.00621](https://arxiv.org/abs/2609.00621)

    该论文提出控制-数据流分离方法，将执行关键协议表示为类型化、经验证的程序对象，使提示词优化器能够改进多智能体大语言模型系统的行为，而不会因提示词修改意外破坏协议导致整个流水线失效。

    

    提示词优化可以改进多智能体大语言模型系统，但被优化的提示词往往承担着两种相互纠缠的角色：一是生成与任务相关的内容，二是指定执行关键协议，例如消息路由、输出格式和终止信号等底层代码所依赖的内容。因此，一次旨在改进内容生成的提示词修改可能会无意中破坏协议，导致整个智能体流水线失效。我们的关键观察是，这两种角色具有不同的表示形式：执行协议通常是结构化的，而任务相关内容通常以非结构化语言表达。基于此，我们提出了控制-数据流分离方法，即将执行关键的控制表示为经过类型化和验证的程序对象，而与任务相关的语言则保持为可优化的数据流，用于智能体之间的通信。这种设计使优化器能够在不……的情况下改进多智能体行为（原文摘要在此处截断）。

    arXiv:2609.00621v1 Announce Type: new  Abstract: Prompt optimization can improve multi-agent LLM systems, but the prompts being optimized often serve two entangled roles: generating task-relevant content and specifying execution-critical protocols, such as message routing, output formatting, and termination signals, on which the underlying code relies. As a result, a prompt edit intended to improve content generation can inadvertently corrupt the protocol and cause the entire agent pipeline to fail. Our key observation is that these two roles have different representations: execution protocols are typically structured, while task-relevant content is usually expressed in unstructured language. Based on this, we propose control-data flow separation, where execution-critical control is represented as typed, validated program objects, while task-relevant language remains the optimizable data flow for agent communication. This design allows optimizers to improve multi-agent behavior without
    
[^172]: 基于双节点蒙特卡洛树搜索的对话式推荐系统高效结构化上下文建模

    Towards Effective Structured Context Modeling for Conversational Recommender Systems via Dual-node Monte Carlo Tree Search

    [https://arxiv.org/abs/2609.00618](https://arxiv.org/abs/2609.00618)

    提出DREAMS框架，通过双节点树结构（引导节点用蒙特卡洛树搜索探索对话动作以推断潜在偏好，利用节点用大语言模型将偏好状态精炼为结构化检索查询），显式建模对话式推荐系统中用户偏好的多轮演化。

    

    我们研究了对话上下文建模在对话式推荐系统（CRS）用户偏好跟踪中的作用。为此，我们提出了DREAMS，这是一种新颖的树状结构上下文建模框架，能够显式地捕捉多轮交互过程中用户偏好的演化。DREAMS引入了两种专门的节点类型，以支持对话式推荐系统的两个基本目标：偏好引导与偏好利用。具体而言，引导节点利用蒙特卡洛树搜索（MCTS）策略性地探索对话动作并推断潜在的用户偏好，而利用节点则采用基于大语言模型（LLM）的精炼方法，将跟踪到的偏好状态转化为结构化的检索查询以用于推荐。在基准数据集上的大量实验证明了DREAMS及其设计的有效性。

    arXiv:2609.00618v1 Announce Type: cross  Abstract: We investigate the role of conversational context modeling in user preference tracking for Conversational Recommendation Systems (CRSs). In this regard, we propose DREAMS, a novel tree-structured context modeling framework that explicitly captures user preference evolution throughout multi-turn interactions. DREAMS introduces two specialized node types to support the two fundamental objectives of CRSs: preference elicitation and preference exploitation. Specifically, elicitation nodes leverage Monte Carlo Tree Search (MCTS) to strategically explore conversational actions and infer latent user preferences, while exploitation nodes employ LLM-based refinement to transform the tracked preference state into structured retrieval queries for recommendation. Extensive experiments on benchmark datasets demonstrate the effectiveness of DREAMS and its design.
    
[^173]: 坦白你所知：大语言模型机器遗忘中遗忘集与模型知识的不对齐问题

    Confess What You Know: Forget-Set Misalignment with Model Knowledge in LLM Unlearning

    [https://arxiv.org/abs/2609.00605](https://arxiv.org/abs/2609.00605)

    提出数据无关的CONFS框架，通过引出模型自身记忆的知识来构建与模型对齐的遗忘集，解决了大语言模型机器遗忘中遗忘集与模型实际记忆内容不对齐所导致的信息泄露或效用下降问题。

    

    大语言模型（LLM）的机器遗忘通常假设预定义的遗忘集与模型实际记忆的内容相匹配，但在原始训练数据不可访问的现实隐私场景中，这一假设经常失效。我们将这种差距称为“遗忘集不对齐”，并识别出两种情况：在“遗忘不足”中，遗忘集遗漏了模型已记忆的信息，导致信息泄露持续存在；在“知识外遗忘”中，算法被驱动去“遗忘”模型从未学过的知识，从而扰动参数并降低模型效用。通过梯度层面的分析，我们证明这些行为源于不对齐的遗忘目标，而非特定的优化方法选择。随后，我们提出了CONfession-to-Forget-Set（CONFS），这是一个数据无关的框架，通过引出并形式化模型自身已记忆的知识来构建与模型对齐的遗忘集。在合成数据、多模态和真实世界基准测试中，CONFS均接近金标准性能。

    arXiv:2609.00605v1 Announce Type: cross  Abstract: Machine unlearning for large language models (LLMs) often assumes that a pre-defined forget set matches what the model has memorized, but this frequently breaks in realistic privacy settings where the original training data is inaccessible. We term this gap forget-set misalignment and identify two cases. In Under Unlearning, the forget set omits memorized information and leakage persists. In Out-of-Knowledge Unlearning, the algorithm is driven to "forget" knowledge the model never learned, perturbing parameters and degrading utility. Using gradient-level analysis, we show these behaviors arise from misaligned unlearning targets rather than specific optimization choices. We then propose CONfession-to-Forget-Set (CONFS), a data-blind framework that constructs model-aligned forget sets by eliciting and formalizing the model's memorized knowledge. Across synthetic, multimodal, and real-world benchmarks, CONFS approaches Gold-standard perfo
    
[^174]: SoK：当安全的智能体共同失效时：多智能体LLM系统的安全性

    SoK: When Safe Agents Fail Together: The Security of Multi Agent LLM Systems

    [https://arxiv.org/abs/2609.00595](https://arxiv.org/abs/2609.00595)

    该论文通过对197篇文献的执行级系统化分析，提出A-I-R攻击分类框架与五部分防御契约，统一梳理了多智能体LLM系统的攻击与防御机制，并指出路径闭合与恢复是防御的核心挑战。

    

    安全的智能体也可能会共同失效。多智能体LLM系统（MAS）会在委托方边界之间传递信息、状态、决策与权限，从而产生局部检查可能遗漏的失效模式。若缺乏执行层面的视角，多智能体的设置很容易被误认为是真正存在多智能体安全效应的证据。因此，我们通过对197篇文献进行以执行为中心的分析，对MAS安全进行了系统化梳理，涵盖六种交互接口、四种对手位置、七种系统级风险以及八种反复出现的攻击路径。我们提出了A-I-R框架，按对手位置、交互接口和由此产生的系统级风险来组织攻击，从而统一了原本分散于各MAS中的攻击机制。我们通过一项涵盖路径目标、观察、干预、信任边界和恢复这五个部分的契约来组织防御，并将路径闭合与恢复识别为关键挑战。我们还审计了44个评估与基准工作。

    arXiv:2609.00595v1 Announce Type: cross  Abstract: Safe agents can fail together. Multi-agent LLM systems (MAS) move information, state, decisions, and authority across principal boundaries, creating failures that local checks may miss. Without an execution-level view, a multi-agent setting can easily be mistaken for evidence of a genuinely multi-agent security effect. We thus systematize MAS security through an execution-centered analysis of 197 works, covering six interaction interfaces, four adversary positions, seven system-level risks, and eight recurring attack paths. We introduce an A-I-R framework that organizes attacks by adversary position, interaction interface, and resulting system-level risk, unifying otherwise fragmented attack mechanisms across MAS. We organize defenses through a five-part contract covering path target, observation, intervention, trust boundary, and recovery, and identify path closure and recovery as key challenges. We audit 44 evaluation and benchmark w
    
[^175]: 苏格拉底走向核领域：基于脑信号感知比较学习情境中AI系统的交互策略

    Socrates went Nuclear: Comparing Interaction Strategies for AI systems in a Learning Context using Brain Sensing

    [https://arxiv.org/abs/2609.00584](https://arxiv.org/abs/2609.00584)

    该研究让50名零基础参与者学习核安全协议，比较了无限制对话AI、仅给提示不给答案的苏格拉底模式AI、以及基于脑信号实时调节难度的自适应辅导系统这三种人机交互策略在学习中的效果。

    

    无限制的AI访问是会绕过学习所需的认知努力，还是会简化知识获取？本文报告了一项研究，我们比较了学习情境中三种用户-AI交互设计：(1) 类似ChatGPT的无限制对话机器人；(2) 在教学上受约束的机器人，通过提示引导而非直接给出最终答案，我们称之为苏格拉底模式；(3) 非对话式的自适应辅导系统，根据从脑信号中提取的用户认知投入实时调整难度。五十名研究参与者的任务是学习核安全协议，选择该领域是因为其零先验知识的基线。参与者依次完成教学视频学习、前测、AI驱动的评估阶段（三种条件下各不相同），以及紧随其后的后测。问题的性质主要集中在事实性知识

    arXiv:2609.00584v1 Announce Type: new  Abstract: Does unrestricted AI access bypass the cognitive effort required for learning, or does it streamline knowledge acquisition? This paper reports on a study where we compare three designs for user-AI interaction in a learning context: (1) an unrestricted conversational bot like ChatGPT, (2) a pedagogically constrained bot that guides through hints without giving final answers, which we refer to as the Socratic mode; and (3) a non-conversational adaptive tutoring system that adjusts difficulty in real-time based on the user's cognitive engagement derived from the brain signals. Fifty study participants were tasked with learning about nuclear safety protocols, a domain chosen for its zero-prior knowledge baseline. The participants progressed through an instructional video, a pre-test, an AI-driven assessment phase, which varied in the three conditions, and an immediate post-test. The nature of the questions centered primarily on factual knowl
    
[^176]: 基于大语言模型与程序设计语言语义的程序退出码预测

    Predicting Program Exit Code with LLMs and Programming Language Semantics

    [https://arxiv.org/abs/2609.00579](https://arxiv.org/abs/2609.00579)

    该论文提出了程序可执行性预测这一新任务，并构建了由有效程序系统性生成无效变换的数据集，以研究大语言模型在判断程序有效性及其违反的形式化语义规则时，究竟是依赖预训练先验知识还是给定的程序语义。

    

    大语言模型（LLM）在代码生成和翻译等多种软件工程任务中已展现出卓越能力。然而，其性能的一个关键局限可能在于对程序设计语言语义的理解（或缺乏理解）。即使给出了显式语义，LLM究竟是应用这些规则，还是依赖预训练期间学到的先验知识，目前仍不清楚。我们通过一项新颖任务——程序可执行性预测来研究LLM是依赖先验知识还是给定语义。该任务要求模型在给定程序语法和操作语义的情况下，预测程序在语义上是有效的还是无效的（如果是无效的，还需指出其违反了哪条形式化规则）。由于PrEx需要有效和无效的程序，我们构建了一个数据集，其中包含从有效程序系统性生成的无效变换。我们在两种语义形式体系和两种语义偏移下，跨Human-（评估开源代码LLM）。

    arXiv:2609.00579v1 Announce Type: cross  Abstract: Large language models (LLMs) have shown proficiency in various software engineering tasks, such as code generation and translation. However, a key limitation in their performance may be their (lack of) understanding of programming-language semantics. Even when explicit semantics are given, it remains unclear whether LLMs apply those rules or lean on priors learned during pre-training instead. We study if LLMs lean on priors or given semantics with a novel task--Program Executability Prediction (PrEx)--that asks models to predict whether a program is semantically valid or invalid (and, if invalid, which formal rule it violates) given the program's syntax and operational semantics. Because PrEx requires both valid and invalid programs, we build a dataset with systematically generated invalid transformations derived from valid programs. We evaluate open-source coding LLMs under two semantic formalisms and two semantic shifts across Human-
    
[^177]: 相同的请求，不同的边界：评估跨对话语境下的网络安全辅助

    Same Request, Different Boundary: Evaluating Cybersecurity Assistance across Conversational Contexts

    [https://arxiv.org/abs/2609.00578](https://arxiv.org/abs/2609.00578)

    该论文提出3R-Bench基准，首次在对话语境下评估LLM的网络安全辅助机制，发现先前的拒绝或接受历史会显著改变模型对同一请求的响应（服从率从62.0%升至85.1%），表明安全防护评估必须考虑对话上下文。

    

    大型语言模型（LLM）能够解决复杂问题，但其在高风险领域的滥用可能造成严重后果，因此模型提供商会限制对潜在有害请求的辅助。然而，拒绝所有网络安全相关请求会损害合法用户的利益，提供商需要一种既能阻止恶意使用、又不会拒绝向防御者提供合法辅助的机制。现有的网络安全专用数据集可以评估这种机制，但没有一个考虑请求所处的对话语境。我们提出了3R-Bench（拒绝、重复与修改），一个包含150个真实世界网络安全请求并增设两种对抗性对话设置的基准，并在其上评估了八个大型语言模型。研究发现，先前助手的行为会强烈改变模型对同一未变请求的响应：在400对样本中的376个有效配对中，请求服从率从先前被拒绝历史后的62.0%上升到先前被接受历史后的85.1%，而相反的模式出现在未……（原文截断）

    arXiv:2609.00578v1 Announce Type: new  Abstract: Large Language Models (LLMs) can solve complex problems, but their misuse in high-risk domains can lead to severe consequences. Model providers therefore restrict assistance for potentially harmful requests. Refusing all cybersecurity requests would therefore harm legitimate users. Providers need a mechanism to block malicious use without denying legitimate assistance to defenders. Existing cybersecurity-specific datasets evaluate this mechanism, but none considers the conversational context of a request. We introduce 3R-Bench (Refusal, Repetition, and Revision), a benchmark of 150 real-world cybersecurity requests augmented with two adversarial conversational settings, and evaluate eight LLMs on it. Prior assistant behavior strongly changes responses to an unchanged request: among 376 available pairs from a 400-pair panel, compliance rises from 62.0% after refused history to 85.1% after accepted history. The opposite pattern appears und
    
[^178]: GeoPAR：基于几何引导并行自回归学习的大规模多智能体组合优化

    GeoPAR: Large-Scale Multi-Agent Combinatorial Optimization with Geometry-Guided Parallel Autoregressive Learning

    [https://arxiv.org/abs/2609.00577](https://arxiv.org/abs/2609.00577)

    GeoPAR提出了一种几何引导的并行自回归强化学习框架，通过投影窗口稀疏几何机制、稀疏边偏置注意力以及缓存引导的冲突处理机制，实现了大规模多智能体组合优化问题的高效求解。

    

    多智能体组合优化问题因其NP难特性而极具挑战性。近期的并行自回归神经求解器通过允许智能体同时进行决策来提升推理效率，但其性能在大规模实例上往往会退化。这主要归因于对局部几何结构建模薄弱，以及冲突任务选择仅在动作生成之后才被处理这一事实。为解决这些局限，我们提出了GeoPAR——一个面向可扩展多智能体组合优化的几何引导并行自回归强化学习框架。GeoPAR集成了三个关键组件：（1）投影窗口稀疏几何机制，通过多方向投影构建轻量级的局部候选邻域；（2）稀疏边偏置注意力，将这些几何关系注入节点表示中；（3）缓存引导的冲突-（原文摘要在此处截断）

    arXiv:2609.00577v1 Announce Type: cross  Abstract: Multi-agent combinatorial optimization problems are notoriously challenging due to their NP-hard nature. Recent parallel autoregressive neural solvers improve inference efficiency by allowing agents to make decisions simultaneously, but their performance often degrades on large-scale instances. This is largely attributable to weak modeling of local geometric structures and the fact that conflicting task selections are handled only after action generation. To address these limitations, we propose GeoPAR, a geometry-guided parallel autoregressive reinforcement learning framework for scalable multi-agent combinatorial optimization. GeoPAR integrates three key components: (1) a projection-window sparse geometry mechanism that builds lightweight local candidate neighborhoods through multi-directional projections, (2) sparse edge-biased attention that injects these geometric relations into node representations, and (3) cache-guided conflict-
    
[^179]: 无需对齐的一致性：与随机选择无法区分的条目敏感语言模型

    Consistency Without Alignment: Item-Sensitive Language Models Indistinguishable From Random

    [https://arxiv.org/abs/2609.00576](https://arxiv.org/abs/2609.00576)

    本研究通过可闭式计算基准的强制选择信号任务证明，语言模型的条目敏感性只是任务能力的必要而非充分条件——尽管全部21个“模型×规则”组合都表现出条目敏感性，但其中8个与随机选择在统计上无法区分，5个甚至比随机表现更差。

    

    条目敏感性（item-sensitivity），即模型的选择是否取决于特定输入而非其自身的输出先验，被广泛报道为任务能力的证据。我们利用一个从桌游《Deception: Murder in Hong Kong》抽象出的强制选择信号传递任务，证明这一证据是必要但不充分的。在该环境中，用于评判一个协调者的参照基准（最大化拟合策略、最大化后验策略和均匀随机选择）均可以闭式形式计算。在七个语言模型、两个模型家族、一项后训练消融实验以及三种独立评分规则下，21个“模型×规则”组合单元中的每一个都可靠地表现出条目敏感性。然而，这21个单元中有8个在统计上无法与一个忽略具体条目、随机选择的参与者区分开，另有5个在描述目标方面的得分低于随机水平。条目敏感性与距随机的距离之间的相关性仅在……（原文摘要至此截断）

    arXiv:2609.00576v1 Announce Type: new  Abstract: Item-sensitivity, defined as whether a model's choice depends on the specific input rather than on its own output prior, is widely reported as evidence of task competence. We show this evidence is necessary but not sufficient using a forced-choice signalling task abstracted from the board game Deception: Murder in Hong Kong. In this environment, the reference points against which a coordinate should be judged (a fit-maximising strategy, a posterior-maximising strategy, and uniform random selection) are all computable in closed form. Across seven language models, two model families, a post-training ablation, and three independent scoring rules, every one of 21 model-by-rule cells is reliably item-sensitive. Yet 8 of those 21 cells are not statistically distinguishable from a chooser that ignores the item and selects at random, and 5 score worse than random at describing the target. Item-sensitivity and distance from random correlate at on
    
[^180]: 基于输出重要性的残差稀疏化方法用于压缩混合专家大语言模型

    Residual Sparsification via Output Importance for Compressing Mixture-of-Experts LLMs

    [https://arxiv.org/abs/2609.00575](https://arxiv.org/abs/2609.00575)

    该论文提出基于输出重要性的残差稀疏化方法，突破了传统上独立最小化每个矩阵压缩误差的局限，在压缩混合专家大语言模型时能更好地保持模型精度。

    

    混合专家架构能够高效地扩展大语言模型，但需要消耗大量的GPU显存。为应对这一需求，通常会对模型进行压缩以减少其显存占用。残差稀疏化是一种代表性的压缩技术，它将专家的每个投影矩阵分解为一个共享的基础矩阵和每个专家独有的残差矩阵，然后对残差进行压缩。现有的稀疏化方法通过独立地最小化每个残差矩阵的压缩误差来对其进行压缩，从而最小化每个投影矩阵的误差。然而，我们的分析表明，这一目标与压缩后保持模型精度的目标并不一致。在专家中，最终输出是通过跨多个投影和隐藏表示的耦合计算产生的。因此，即使单个矩阵中存在微小误差，也可能通过隐藏表示和投影之间的交互进行传播。

    arXiv:2609.00575v1 Announce Type: new  Abstract: Mixture-of-experts (MoE) architectures scale large language models efficiently, but they demand massive GPU memory. To cope with such demand, models are commonly compressed to reduce their memory footprint. Residual sparsification is a representative compression technique that decomposes each projection matrix of an expert into a shared base matrix and per-expert residual matrix, and then compresses the residuals. Existing sparsification methods compress each residual matrix independently by minimizing its compression error, thereby minimizing the error of each projection matrix. However, our analysis shows that this objective is misaligned with preserving model accuracy after compression. In an expert, the final output is produced through computations coupled across multiple projections and hidden representations. Therefore, even small errors in individual matrices can propagate through hidden representations and projection interactions
    
[^181]: 企业人工智能中传承、治理与决策完整性的数学框架

    A Mathematical Framework for Legacy, Governance, and Decision Integrity in Enterprise AI

    [https://arxiv.org/abs/2609.00572](https://arxiv.org/abs/2609.00572)

    本文提出了一个企业AI机构传承的数学框架，通过基于知识留存、治理、人类监督、适应性、反馈学习和司法忠实度六个维度的惩罚几何平均构建归一化传承分数，并辅以决策置信度与风险模型，量化评估机构在原设计者离开后能否长期保持合法、可解释、可适应的健全决策能力。

    

    企业人工智能日益嵌入到各类决策之中，这些决策必须在人员流动、模型更换、法规变化和组织激励转变的情况下，始终保持合法、可解释、可适应且可问责。现有治理框架提供了重要原则，但其本身并未提供一种简洁的数学语言来评估一个机构能否长期保持健全的判断力。本文开发了一个面向机构传承的设计科学框架，即决策系统在原始设计者离开后，仍能持续产生有益、合法、可解释且可适应结果的持久能力。该框架的贡献包括：(i) 一个归一化的传承分数，该分数基于知识留存、治理、人类监督、适应性、反馈学习和司法管辖忠实度的惩罚几何平均值；(ii) 决策置信度与决策风险模型，用于分离评估决策的可靠性与潜在风险（摘要原文在此处截断）。

    arXiv:2609.00572v1 Announce Type: cross  Abstract: Enterprise artificial intelligence is increasingly embedded in decisions that must remain lawful, explainable, adaptable, and accountable despite personnel turnover, model replacement, regulatory change, and shifting organizational incentives. Existing governance frameworks provide important principles but do not by themselves supply a compact mathematical language for evaluating whether an institution can preserve sound judgment over time. This paper develops a design-science framework for institutional legacy: the durable capacity of a decision system to continue producing beneficial, lawful, explainable, and adaptable outcomes after its original designers have stepped away. The framework contributes: (i) a normalized Legacy Score based on a penalized geometric mean of knowledge retention, governance, human oversight, adaptability, feedback learning, and jurisdictional fidelity; (ii) Decision Confidence and Decision Risk models separ
    
[^182]: VoiceLongMemEval：助手是否记得你的声音听起来如何？

    VoiceLongMemEval: Do Assistants Remember How You Sounded?

    [https://arxiv.org/abs/2609.00570](https://arxiv.org/abs/2609.00570)

    该论文提出了VoiceLongMemEval（VLME）基准，用于评估AI助手在长时多会话对话中能否记住情感、韵律和语音事件等副语言信息，发现现有大语言模型普遍存在无法捕捉说话方式的“情感鸿沟”。

    

    随着多智能体架构和大语言模型规模的不断增长，部署的AI助手越来越多地需要对长且连续的多会话对话历史进行推理。当前的基准测试将这种对话历史评估视为长时程信息检索、时间推理或知识更新，却关键性地忽略了人机交互的基本动态，即“他们是怎么说的”（说话方式）。为了填补这一空白，我们提出了VoiceLongMemEval（VLME）基准，其中每个问题的答案都依赖于附加在对话轮次上的副语言元数据（情感标签、韵律描述符和语音事件），而这些信息仅凭文字本身是无法恢复的。每个测试项都经过三阶段对抗性门控验证，确保强大的语言模型在仅获得文本转录时无法回答。对领先的前沿模型和开放权重模型的评估揭示了普遍存在的“情感鸿沟”；提供文本轨道的副语言元数据……（摘要在此处截断）

    arXiv:2609.00570v1 Announce Type: new  Abstract: With the growing scale of multi-agent architectures and large language models, deployed AI assistants are increasingly tasked with reasoning over long, continuous, multi-session conversation histories. Current benchmarks evaluate this dialogue history as information retrieval over long horizon, temporal reasoning, or knowledge updates, while crucially ignoring the fundamental dynamics of human-agent interaction, i.e. how they said it. To address this gap, we present VoiceLongMemEval (VLME) benchmark, where every answer depends on paralinguistic metadata (emotion labels, prosody descriptors, and voice events) attached to conversational turns, which is otherwise unrecoverable from the words alone. Every item passes a three-stage adversarial gate, ensuring that a strong language model fails when given only the transcript. Evaluating leading frontier and open-weight models reveals a pervasive affect gap; providing text-track paralinguistic m
    
[^183]: WiseSpec：面向代码生成的需求驱动智能体

    WiseSpec: Requirements-Driven Agents for Code Generation

    [https://arxiv.org/abs/2609.00568](https://arxiv.org/abs/2609.00568)

    该论文提出WiseSpec框架，借鉴软件需求工程思想，通过自动构建高质量结构化需求并结合基于执行的评估进行迭代优化，从而提升大语言模型在仓库级代码生成中的表现。

    

    代码生成旨在根据任务需求自动生成源代码，随着大语言模型（LLM）的快速发展而受到广泛关注。尽管取得了显著进展，但由于任务描述往往不完整、含糊不清或缺乏关键的上下文信息，大语言模型在处理复杂软件工程任务时常常难以生成正确的代码。现有方法主要通过更复杂的工具、技能和工作流程来提升编码智能体的能力，却在很大程度上忽视了任务需求本身的质量。为解决这一局限，我们从软件需求工程中汲取灵感，提出了WiseSpec——一个面向仓库级代码生成的新型需求驱动智能体框架。WiseSpec能够自动构建结构化且信息丰富的需求，通过基于执行的评估来衡量需求质量，并进行迭代式（原文摘要至此被截断）。

    arXiv:2609.00568v1 Announce Type: cross  Abstract: Code generation aims to automatically generate source code from task requirements and has attracted significant attention with the rapid advancement of large language models (LLMs). Despite remarkable progress, LLMs often struggle to generate correct code for complex software engineering tasks because task descriptions are frequently incomplete, ambiguous, or lack critical contextual information. Existing approaches primarily improve the capabilities of coding agents through more sophisticated tools, skills, and workflows, while largely overlooking the quality of the task requirements themselves. To address this limitation, we draw inspiration from software requirements engineering and propose WiseSpec, a novel requirements-driven agent framework for repository-level code generation. WiseSpec automatically constructs structured and information-rich requirements, assesses their quality through execution-based evaluation, and iteratively
    
[^184]: EEG-VID：面向脑电解码与辅助目标选择的任务引导式潜变量预测预训练

    EEG-VID: Task-Guided Latent Predictive Pretraining for EEG Decoding and Assistive Target Selection

    [https://arxiv.org/abs/2609.00566](https://arxiv.org/abs/2609.00566)

    EEG-VID提出了一种任务引导式潜变量预测预训练框架，通过指数移动平均编码器预测未来EEG潜状态，在42组跨会话跨被试对比中有41组提升准确率（最高提升16.22个百分点），并可有效应用于场景约束下的辅助目标选择。

    

    我们提出EEG-VID，这是一个面向跨会话与跨被试脑电解码的任务引导式潜变量预测预训练框架。EEG-VID利用指数移动平均目标编码器与弱任务引导，从近期的EEG历史预测未来的潜变量EEG状态，随后进行有监督微调。在VIG-48和BCI竞赛IV-2a/IV-2b数据集上，第一阶段预训练在42组匹配的骨干网络-数据集-协议对比中有41组提升了平均准确率，包括全部12个留一被试设置，最大提升达16.22个百分点。在48区域的跨天VIG-48任务上，EEG-VID实现了6.52%的Top-1准确率和30.50%的Top-5准确率。在一项独立的六名被试离线机器人场景研究中，经过被试特定校准后，候选约束下的目标选择准确率达到40.24%，而随机水平仅为25%。这些结果支持任务引导的潜变量预测作为脑电解码与场景约束辅助目标选择的一种可迁移预训练策略。

    arXiv:2609.00566v1 Announce Type: cross  Abstract: We propose EEG-VID, a task-guided latent predictive pretraining framework for EEG decoding under session and subject shifts. EEG-VID predicts future latent EEG states from recent history using an exponential-moving-average target encoder and weak task guidance, followed by supervised fine-tuning. Across VIG-48 and BCI Competition IV-2a/IV-2b, Stage 1 improves mean accuracy in 41 of 42 matched backbone-dataset-protocol comparisons, including all 12 leave-one-subject-out settings, with a maximum gain of 16.22 percentage points. On the 48-region cross-day VIG-48 task, EEG-VID achieves 6.52% Top-1 and 30.50% Top-5 accuracy. In a separate six-participant offline robot-scene study, candidate-constrained target selection reaches 40.24% versus a 25% chance level after subject-specific calibration. These results support task-guided latent prediction as a transferable pretraining strategy for EEG decoding and scene-constrained assistive target s
    
[^185]: EM^2Mem：面向大型语言模型的事件中心多模态记忆

    EM^2Mem: Event-Centric Multimodal Memory for Large Language Models

    [https://arxiv.org/abs/2609.00551](https://arxiv.org/abs/2609.00551)

    该论文提出EM^2Mem，一种以事件为中心的多模态记忆框架，通过在记忆构建阶段将多模态记录、时间上下文、图谱关系与溯源信息绑定到事件锚点，形成“可直接用于生成”的记忆单元，免去了推理时重建跨模态对齐的负担，并在三个长视频问答基准上将平均准确率较最强记忆基线提升2.0至3.7个百分点。

    

    多模态记忆为长视频问答提供了一种可扩展的接口，但现有方法通常将字幕、视频帧、转录文本、摘要或图谱事实作为孤立的片段进行检索。尽管这些片段可被搜索，却并不“可直接用于生成”：语言模型必须在推理阶段、在上下文受限且归因困难的情况下重建跨模态和时间上的对齐关系。我们提出了EM^2Mem，一个以事件为中心的多模态记忆框架，它在记忆构建阶段将异构证据绑定到事件锚点上。每个以事件为索引的记忆单元对齐多模态记录、时间上下文、图谱关联关系、语义事实以及来源溯源信息，从而能够基于多模态事件（而非特定模态的孤立片段）进行紧凑的证据读取。在三个长视频问答基准上，EM^2Mem 相比最强的记忆基线分别将平均准确率提升2.0、2.4和3.7个百分点，并在严格的事件级评估上……（原文摘要在此处截断）

    arXiv:2609.00551v1 Announce Type: cross  Abstract: Multimodal memory offers a scalable interface for long-video question answering, but existing methods often retrieve captions, frames, transcripts, summaries, or graph facts as isolated fragments. Although searchable, such fragments are not generation-ready: language models must reconstruct cross-modal and temporal alignments at inference time, when context is limited and attribution is difficult. We propose EM^2Mem, an event-centric multimodal memory framework that binds heterogeneous evidence to event anchors during memory construction. Each event-indexed memory cell aligns multimodal records, temporal context, graph-linked relations, semantic facts, and provenance, enabling compact evidence readout over grounded multimodal events rather than modality-specific fragments. Across three long-video QA benchmarks, EM^2Mem improves average accuracy over the strongest memory baseline by 2.0, 2.4, and 3.7 points, improves strict event-level 
    
[^186]: 运行时无关的持久化智能体：跨模型、执行框架与服务器保留身份、记忆与代码

    Runtime-Independent Persistent Agents: Preserving Identity, Memory, and Code Across Models, Harnesses, and Servers

    [https://arxiv.org/abs/2609.00546](https://arxiv.org/abs/2609.00546)

    该论文提出一种运行时无关的持久化智能体架构，将身份、持久记忆和版本化代码作为连续性基底，与可替换的模型、执行框架、宿主服务器及交互表面解耦，使得更换这些运行时组件属于智能体迁移而非重新创建。

    

    智能体系统通常由当前产生其行为的模型和执行框架来定义。这种界定方式对单次执行是有用的，但对于一个长期存续的智能体而言则不够充分——这样的智能体可能会更换模型、编排框架、交互会话和宿主服务器，同时仍保持同一身份、记忆和可执行代码谱系。我们提出了一种面向持久化智能体的运行时无关架构。一个承载连续性的基底 P_t=(I_t,M_t,B_t) 包含架构化的身份表示、私有持久记忆和版本化的软件体。一个可替换的部署绑定由执行基底 E_t=(R_t,H_t,D_t)（提供推理器、执行框架和宿主）以及一组交互表面 S_t（如聊天、API 或用户界面绑定）组成。一次已部署的执行表示为 A_t=P_t▷(E_t,S_t)；当授权协议保留至少……（摘要在此处被截断）

    arXiv:2609.00546v1 Announce Type: cross  Abstract: Agent systems are commonly described by the model and harness that currently produce their behavior. That boundary is useful for one execution but underspecifies a long-lived agent that may change models, orchestration harnesses, interaction sessions, and host servers while retaining one identity, memory, and executable code lineage. We present a runtime-independent architecture for persistent agents. A continuity-bearing substrate $P_t=(I_t,M_t,B_t)$ contains an architectural identity representation, private durable memory, and a versioned software body. A replaceable deployment binding comprises an execution substrate $E_t=(R_t,H_t,D_t)$, which supplies a reasoner, harness, and host, and a set of interaction surfaces $S_t$, such as chat, API, or user interface bindings. A deployed execution is $A_t=P_t\triangleright(E_t,S_t)$; changing either replaceable layer is migration, not agent creation, when an authorized protocol preserves at
    
[^187]: 检索增强生成中基于文档关系图的反馈辅助信任传播

    Feedback-Assisted Trust Propagation over Document Relation Graphs for Retrieval-Augmented Generation

    [https://arxiv.org/abs/2609.00543](https://arxiv.org/abs/2609.00543)

    该论文提出TrustPropRAG，通过将文档关系构建为图，以少量人类反馈为锚点，利用多跳传播和联合优化问题估计每个文档的信任分数，从而提升检索增强生成系统的答案可靠性。

    

    检索增强生成（RAG）系统所依赖的外部语料库可能包含过时、矛盾、含噪或不可靠的文档，从而带来可靠性风险。先前的工作利用文档关系来提高RAG的答案可靠性。为了将可靠性信号传播到直接比较的文档对之外，我们提出了TrustPropRAG，该方法将文档关系构建为图，并通过图上的多跳传播来估计文档可靠性。TrustPropRAG以有限的人类对文档可靠性的反馈作为传播的锚点，将这些收集成本高昂的基于反馈的可靠性信号扩展到整个语料库。具体而言，基于所构建的文档关系图，TrustPropRAG通过形式化并求解一个联合捕获成对文档关系和用户反馈的优化问题，为每个文档估计信任分数。随后，这些分数被用于……

    arXiv:2609.00543v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) systems rely on external corpora that may contain outdated, contradictory, noisy, or unreliable documents, introducing reliability risks. Prior work has leveraged document relations to improve the answer reliability of RAG. To propagate reliability signals beyond directly compared document pairs, we propose TrustPropRAG, which structures document relations as a graph and estimates document reliability through multi-hop propagation across the graph. TrustPropRAG anchors this propagation with a limited set of human feedback on document reliability, extending these costly-to-collect feedback-based reliability signals across the whole corpus. Specifically, based on the constructed document relation graph, TrustPropRAG estimates a trust score for each document by formulating and solving an optimization problem that jointly captures pairwise document relations and user feedback. These scores are then used t
    
[^188]: 我们到了吗？评估计算机使用智能体对盲人与桌面应用无障碍交互的支持

    Are We There Yet? Assessing Computer-Use Agents for Blind Users' Accessible Interaction with Desktop Applications

    [https://arxiv.org/abs/2609.00524](https://arxiv.org/abs/2609.00524)

    通过三周日记研究首次系统评估了计算机使用智能体对盲人用户操作桌面应用的实际支持效果，发现即使最先进的GPT-5成功率也仅有52.5%，且存在定位、规划、约束跟踪和终止等系统性失败，表明当前CUA远未满足盲人用户无障碍交互的需求。

    

    计算机使用智能体（CUA）正成为智能体式人机交互的新范式，它将语言推理与多模态界面定位相结合来操作图形用户界面。然而，其在真实桌面工作流程中对使用屏幕阅读器的盲人用户的有效性仍不清楚。我们开展了一项为期三周的日记研究，8名盲人用户使用OLLA——一个对屏幕阅读器无障碍的CUA原型，在12个应用程序中收集了1,258条命令，并记录了截图、UI树、模型响应和操作轨迹。我们在部署期间评估了GPT-5，并使用另外四个模型重新执行了相同的命令。GPT-5取得了最高的成功率，为52.5%。轨迹分析揭示了界面定位、规划、约束跟踪和任务终止方面的失败，而访谈则揭示了超越自动化的需求。

    arXiv:2609.00524v1 Announce Type: cross  Abstract: Computer-use agents are emerging as a paradigm for agentic human-AI interaction, combining language reasoning with multi-modal interface grounding to operate GUIs. Yet their effectiveness for blind screen-reader users in real-world desktop workflows remains unclear. We present a three-week diary study with 8 blind users using OLLA, a screen-reader-accessible CUA prototype, collecting 1,258 commands across 12 applications with screenshots, UI trees, model responses, and action traces. We evaluate GPT-5 during deployment and re-execute the same commands with four additional models. GPT-5 achieved the highest success rate at 52.5%. Trace analysis reveals grounding, planning, constraint-tracking, and termination failures, while interviews reveal beyond-automation needs.
    
[^189]: 防护措施起作用了，但LLM系统因此更安全了吗？

    The Safeguard Worked. Is the LLM System Safer?

    [https://arxiv.org/abs/2609.00519](https://arxiv.org/abs/2609.00519)

    论文指出防护措施的标准评估指标（拒绝率、攻击成功率等）只能衡量其局部表现，无法回答部署层面“LLM系统是否真正更安全”的问题，并揭示了证据的不对称性：一次成功的有害攻击即可证明危害仍存在，而要证明系统安全则需要超越防护自身数据的系统性证据。

    

    部署的LLM服务中的防护措施通常通过拒绝率、攻击成功率和政策违规率来进行评估。这些比率刻画的是防护控制在被测试请求上的表现。然而部署层面需要回答的是一个不同的问题：对于一个不断调整攻击手段或寻找其他入侵途径的攻击者，该服务在多大程度上仍会为其有害任务提供帮助。我们确定了每项已报告的结果对这一问题的实际含义，从而能够在统一的部署标准下比较不同防护系列的研究结果。证据要求呈现出强烈的不对称性：只要有一次从已部署服务中成功获取有害帮助的攻击，就足以证明此类帮助仍然存在，而且这类攻击在编码记录中反复出现。而要证明所剩有害帮助很少，则不能仅凭防护措施自身的评估数字，还需要证据表明在防护措施完成其局部功能之后，系统的其他部分仍然允许什么。

    arXiv:2609.00519v1 Announce Type: cross  Abstract: Safeguards in deployed LLM services are evaluated by refusal, attack success, and policy violation rates. Those rates characterize how a control performed on the requests it was tested on. A deployment has to answer a different question: how much help with harmful tasks the service still gives an attacker who keeps adapting or finds another way in. We determine what each reported result implies for that question, allowing results from different safeguard families to be compared under one deployment criterion. The evidence requirements are strongly asymmetric. One attack that obtains harmful help from the deployed service suffices to establish that such help remains, and such attacks appear repeatedly in the coded record. Establishing that little remains cannot follow from the safeguard's own numbers alone; it also requires evidence about what the surrounding system still allows after the safeguard performs its local function. Such evid
    
[^190]: 语际假说：大语言模型通过潜在的任务无关特征空间进行翻译

    The Interlingua Hypothesis: LLMs Translate via a Latent Task-agnostic Feature Space

    [https://arxiv.org/abs/2609.00515](https://arxiv.org/abs/2609.00515)

    该论文提出“语际假说”，认为大语言模型通过将源语句编码进任务无关的潜在多语言特征空间、再从中解码生成目标语句的方式完成翻译，并从BLEU分数可预测性、组件因果影响和微调三个方面提供了支持证据。

    

    大语言模型（LLMs）近期在机器翻译任务上的表现已超越强大的有监督基线模型。这引发了一个问题：大语言模型在不同语言之间执行机器翻译的背后机制是什么？受近期可解释性研究发现的启发——即大语言模型使用大规模多语言潜在特征表示来进行语言建模——我们提出了语际假说。该假说认为，语言模型的翻译方式是：先将源语句读入一个潜在特征空间，再从该潜在特征空间读取信息来生成目标语句。我们展示了支持这一假说的三条证据：（1）不同语言对之间的BLEU分数差异在很大程度上可以由各语言特定的能力预测，而无需引入语言对特定的交互项；（2）许多模型组件在单语任务和翻译任务中都具有因果影响力；（3）微调

    arXiv:2609.00515v1 Announce Type: cross  Abstract: Large language models (LLMs) have recently demonstrated improved machine translation performance over strong supervised baselines. This raises questions as to what mechanisms underlie how LLMs perform machine translation between languages. Motivated by recent interpretability findings--namely, that LLMs use massively multilingual latent feature representations to perform language modeling--we propose the interlingua hypothesis. The hypothesis holds that language models translate by reading a source sentence into a latent feature space, and generate a target sentence by reading from the latent feature space. We show three lines of evidence in support of this hypothesis: (1) variance in BLEU across language pairs is largely predictable from language-specific competences with no language pair-specific interaction terms; (2) many model components are causally influential in both monolingual tasks and translation tasks; and (3) fine-tuning 
    
[^191]: ISO-RAG：面向检索增强生成的等周噪声控制

    ISO-RAG: Isoperimetric Noise Control for Retrieval-Augmented Generation

    [https://arxiv.org/abs/2609.00513](https://arxiv.org/abs/2609.00513)

    ISO-RAG通过将知识图谱投影到双曲庞加莱球并利用等周轮廓剪除虚假边，抑制图检索中的语义漂移，实现多跳问答中精确且低延迟的检索增强生成。

    

    检索增强生成（RAG）能够缓解大语言模型（LLM）的幻觉问题，然而传统的稠密检索难以应对多跳问答（QA）中复杂的推理路径。基于图的RAG虽然能捕获多步关系，但由于对全局图进行含噪遍历，存在严重的语义漂移和较高的在线延迟问题。为此，我们提出了ISO-RAG（ISOperimetric Retrieval-Augmented Generation，等周检索增强生成），一种几何感知的RAG框架。通过将底层知识图谱投影到双曲庞加莱球中并预计算每个节点的等周轮廓，ISO-RAG能够在检索过程中剪除虚假边，将搜索空间限制在严格局部化的子图内。这种拓扑净化机制对驱动检索过程的个性化PageRank（PPR）扩散进行调节，确保了精确且低延迟的收敛。在多跳问答基准上的实验表明，ISO-RAG优于最先进的基线方法。

    arXiv:2609.00513v1 Announce Type: new  Abstract: Retrieval-Augmented Generation (RAG) mitigates large language models (LLMs) hallucinations, yet conventional dense retrieval struggles with the complex reasoning paths of multi-hop question answering (QA). Graph-based RAG captures multi-step relationships but suffers from severe semantic drift and high online latency due to noisy global graph traversals. Thus, we propose ISO-RAG (ISOperimetric Retrieval-Augmented Generation), a geometry-aware RAG framework. By projecting the underlying knowledge graph into a hyperbolic Poincare ball to precompute node-wise isoperimetric profiles, ISO-RAG prunes spurious edges during retrieval, restricting the search space to a strictly localized subgraph. This topological purification regulates Personalized PageRank (PPR) diffusion driving the retrieval process, ensuring exact and low-latency convergence. Experiments on multi-hop QA benchmarks demonstrate that ISO-RAG outperforms state-of-the-art baselin
    
[^192]: 当算法成为品牌危机：分布式责任与可问责透明度的社会技术理论

    When the Algorithm Becomes the Brand Crisis: A Sociotechnical Theory of Distributed Responsibility and Accountable Transparency

    [https://arxiv.org/abs/2609.00510](https://arxiv.org/abs/2609.00510)

    该论文提出了一种社会技术过程理论，区分了AI算法事件、AI相关组织危机与组织丑闻，并构建框架解释当AI责任在系统、开发者、部署者、供应商和用户之间分布时，利益相关者如何进行责任归因与评价。

    

    人工智能系统日益通过聊天机器人、推荐系统、自动化决策和生成式界面来兑现面向市场的承诺。它们的失效、滥用和虚假陈述引发了一个传统品牌危机模型未能充分界定的问题：当技术因果、面向客户的控制权以及治理职责分散分布于AI系统、开发者、部署者、供应商和用户之间时，利益相关者如何归因责任？本概念性论文基于对经验证的学术文献和一手资料进行的结构化、联合式范围综述，发展出一种社会技术过程理论。该理论将AI/算法事件与AI相关的组织危机、进而与AI相关的组织丑闻进行了区分。该框架提出：事件的构型塑造了针对特定行为者的责任归因；责任归因进而影响对能力、诚信、公平性和关系的评价；公众……（摘要原文在此处被截断）

    arXiv:2609.00510v1 Announce Type: new  Abstract: Artificial intelligence systems increasingly enact market-facing promises through chatbots, recommendation systems, automated decisions, and generative interfaces. Their failures, misuse, and misrepresentation raise a question that conventional brand-crisis models do not fully specify: how do stakeholders assign responsibility when technical causation, customer-facing control, and governance duties are distributed across an AI system, developer, deployer, vendor, and user? This conceptual paper develops a sociotechnical process theory from a structured, federated scoping synthesis of verified academic and primary sources. It distinguishes an AI/algorithmic incident from an AI-related organisational crisis and, in turn, from an AI-related organisational scandal. The framework proposes that incident configuration shapes actor-specific attribution; attribution informs capability, integrity, fairness, and relationship appraisals; and public 
    
[^193]: CoVer：冲突感知的声明验证

    CoVer: Conflict-Aware Claim Verification

    [https://arxiv.org/abs/2609.00508](https://arxiv.org/abs/2609.00508)

    该论文提出了基于X社区笔记系统构建的大规模真实世界数据集ContraNote和三阶段事实裁决框架CoVer，通过优先处理证据而非噪声，有效解决了社交媒体事实核查中的证据层面与聚合层面冲突问题，达到了最先进的性能。

    

    社交媒体事实核查长期以来一直受到证据层面和聚合层面冲突的挑战，即错误的证据会模仿权威新闻来源。为了刻画这一挑战并支持冲突验证任务，我们提出了ContraNote，这是一个从X的社区笔记（Community Notes）系统中精选构建的大规模真实世界数据集。该数据集包含33,686条帖子用于评估证据层面的冲突解决，以及54,474个实例用于评估聚合层面的优先级排序。此外，我们提出了CoVer，一个包含三阶段流水线的事实裁决框架：证据模式规范化、事实共识与支持验证。该框架优先采纳证据而非噪声，以防止噪声损害最终判定。技术评估表明，与最先进的基线方法相比，CoVer在ContraNote上取得了出色的性能（Conflict任务上准确率86.0%、宏F1值68.0%、平衡准确率64.5；另一任务上准确率88.5%、宏F1值88.5、平衡准确率89.2）。

    arXiv:2609.00508v1 Announce Type: new  Abstract: Social media fact-checking has long been challenged by evidence-level and aggregation-level conflicts, where erroneous evidence mimics authoritative news sources. To capture this challenge and support conflict verification tasks, we present ContraNote, a large-scale real-world dataset curated from X's Community Notes system. It includes 33,686 posts for evaluating evidence-level conflict resolution, and 54,474 instances for evaluating aggregation-level prioritization. Additionally, we propose CoVer, a factual adjudication framework with three-stage pipelines: evidence schema normalization, factual consensus and support verification. This prioritizes evidence over noise to prevent it from compromising the final verdict. Technical evaluations show that CoVer achieves strong performance compared with state-of-the-art baselines across ContraNote (86.0% Acc., 68.0% mac. F1, 64.5 bal. Acc. on Conflict; and 88.5% Acc., 88.5 mac. F1 and 89.2 bal
    
[^194]: RecalibrateGPT：抗AI疲劳的对话式交互界面

    RecalibrateGPT: AI Fatigue Resilient Conversational Interfaces

    [https://arxiv.org/abs/2609.00506](https://arxiv.org/abs/2609.00506)

    RecalibrateGPT通过五种一键式跨轮次操作符（锚定、重放、增量、范围、引导）让用户能够基于完整对话历史快速重新校准大语言模型的响应，有效缓解重复输入、浏览扫描、决策瘫痪和上下文漂移这四类对话式AI疲劳。

    

    大语言模型功能强大，但其界面常常陷入“输入→阅读→重新输入”的循环，造成对话式AI疲劳、认知负荷增加，并最终导致用户放弃任务。为缓解这一问题，我们提出了RecalibrateGPT，该系统引入了五种跨轮次操作符（锚定Anchor、重放Replay、增量Delta、范围Scope和引导Steer），每种操作符针对一种不同的疲劳类型，用户只需单击一次，系统即可通过结构化面板作用于完整的对话历史，重新校准大语言模型的响应。用户通过AssistiveButton（辅助按钮）以三种操作面板布局之一（垂直式、弧形式或平板式）调用这些操作符。我们使用同样的12名高级大语言模型用户开展了两项试点研究。最初的形成性定性研究识别出了四种疲劳类型的分类体系（重复输入、浏览扫描、决策瘫痪和上下文漂移），并由此推导出RecalibrateGPT的两个设计目标。随后的定量评估发现它能减少

    arXiv:2609.00506v1 Announce Type: cross  Abstract: Large language models are powerful, but their interfaces often devolve into a type $\rightarrow$ read $\rightarrow$ retype loop, creating conversational AI fatigue, cognitive load, and eventual task abandonment. To mitigate this, we present RecalibrateGPT, a system introducing five cross-turn operators (Anchor, Replay, Delta, Scope, and Steer) that each target a distinct fatigue type, recalibrating LLM responses through a structured panel by acting on the full conversation history with a single click. Users invoke these operators through the AssistiveButton in one of three operator palette layouts: Vertical, Arc, or Tablet. We conducted two pilot studies with the same 12 advanced LLM users. An initial formative qualitative study identifies a taxonomy of four fatigue types (retyping, scanning, decision paralysis, and context drift) and derives two design objectives for RecalibrateGPT. A follow-up quantitative evaluation finds it reduces
    
[^195]: 折扣马尔可夫博弈中的独立强化学习

    Independent Reinforcement Learning in Discounted Markov Games

    [https://arxiv.org/abs/2609.00504](https://arxiv.org/abs/2609.00504)

    本文在“PPAD的ETH”假设下证明了折扣一般和马尔可夫博弈中独立学习计算粗相关均衡的困难性，并提出首个无需结构限制、具有次指数收敛保证的彻底非耦合分层乐观镜像下降算法。

    

    在这项工作中，我们研究了折扣一般和马尔可夫博弈中的彻底非耦合学习。在假设“PPAD的指数时间假说（ETH）”成立的前提下，我们证明：对于每个固定的折扣因子，当玩家在去中心化环境中独立学习时，不存在多项式时间算法来计算折扣一般和马尔可夫博弈中逆多项式精度的粗相关均衡。作为对该困难性结果的补充，我们提供了似乎是首个具有次指数收敛保证、可收敛到粗相关均衡的彻底非耦合算法，且不对博弈施加任何结构性限制。我们的算法是乐观镜像下降的一种分层变体，并采用了为多智能体设置量身定制的递增步长调度方案。最后，我们开发了上述算法的全反馈和部分反馈两个版本，并建立了次……

    arXiv:2609.00504v1 Announce Type: cross  Abstract: In this work, we study radically uncoupled learning in discounted general-sum Markov games. Assuming ``$\mathsf{ETH}$ for $\mathsf{PPAD}$", we show that, for every fixed discount factor, there is no polynomial-time algorithm for computing inverse-polynomially accurate coarse correlated equilibria in discounted general-sum Markov games when players learn independently in decentralized settings. Complementing this hardness result, we provide what appears to be the first \emph{radically uncoupled} algorithm with sub-exponential convergence guarantees to coarse correlated equilibria in discounted general-sum Markov games without imposing any structural restrictions on the game. Our algorithm is a \emph{layered} variant of optimistic mirror descent with an increasing step-size schedule tailored to the multi-agent setting. Finally, we develop both full-feedback and partial feedback versions of the aforementioned algorithm and establish sub-e
    
[^196]: 具有显式时间间隔动力学的波函数反向传播

    Wave Function Backpropagation with Explicit Temporal-Interval Dynamics

    [https://arxiv.org/abs/2609.00503](https://arxiv.org/abs/2609.00503)

    本文提出波函数反向传播（WFB），用可学习的振幅、波数、角频率和相位参数化神经响应，通过可微时空波相位显式关联时间间隔，并引入基于拉普拉斯算子的空间曲率修正，在轨迹预测任务中将平均位移误差降低20.4%。

    

    传统神经网络主要通过仿射变换结合非线性激活进行学习，而流逝的时间通常仅被视为辅助特征或被假设为均匀采样。本文提出波函数反向传播，这是一种波参数化的学习形式，其中神经响应由可学习的振幅、波数、角频率和相位来表示。该形式通过可微时空波的相位将观测状态与其时间间隔 Δt 相关联。我们推导了标准的WFB梯度，以及一种基于波响应拉普拉斯算子的空间曲率修正。WFB被实例化于一个刻意设计的前馈轨迹预测器中，以提供受控的概念验证；序列学习不在本评估的范围之内。在运动特征条件下，使用真实时间间隔的STD-WFB将平均位移误差（ADE）降低了20.4%。

    arXiv:2609.00503v1 Announce Type: new  Abstract: Conventional neural networks learn predominantly through affine transformations followed by nonlinear activations, while elapsed time is often treated as an auxiliary feature or assumed to be uniformly sampled. This paper introduces Wave Function Backpropagation (WFB), a wave-parameterized learning formulation in which neural responses are represented by learnable amplitude, wavenumber, angular frequency, and phase. The formulation associates an observed state with its temporal interval Delta t through the phase of a differentiable spatiotemporal wave. We derive standard WFB gradients and a spatial-curvature correction based on the Laplacian of the wave response. WFB is instantiated in a deliberately feed-forward trajectory predictor to provide a controlled proof of concept; sequence learning is outside the scope of the present evaluation. With motion features, STD-WFB using real intervals reduces average displacement error (ADE) by 20.4
    
[^197]: 面向大语言模型的有效性感知越狱评估

    Validity-Aware Jailbreak Evaluation for Large Language Models

    [https://arxiv.org/abs/2609.00498](https://arxiv.org/abs/2609.00498)

    该论文提出SEAV框架，通过将越狱回复分解为有序步骤并结合LLM评判与检索增强验证，同时评估回复的有效性与正确性，解决了现有越狱评估方法只注重语言合理性而忽视事实正确性的问题。

    

    越狱鲁棒性已成为大语言模型（LLM）安全评估的核心，然而目前主流的评估方法主要依赖于拒绝行为、语义相似度以及意图匹配等启发式手段，这些方法强调语言上的合理性而非正确性。我们发现了现有评估中的一个关键局限：许多越狱意图取决于指令的有效性而非认知上的事实性，这使得那些看似真实的回复尽管在事实上或程序上是错误的，却仍被判定为攻击成功。为弥补这一缺陷，我们提出了顺序认知与动作级验证框架（SEAV），这是一种以验证为中心的越狱评估框架，它将模型回复分解为有序的步骤，并同时评估其有效性与正确性。SEAV 将“大语言模型作为评判者”机制用于语义解释，并结合基于外部知识源的检索增强验证，评估生成内容在事实上是否准确……

    arXiv:2609.00498v1 Announce Type: new  Abstract: Jailbreak robustness has become central to large language model (LLM) safety evaluation, yet prevailing methodologies rely primarily on refusal behavior, semantic resemblance, and intent-matching heuristics that emphasize linguistic plausibility rather than correctness. We identify a key limitation in existing evaluations: many jailbreak intents depend on instructional validity rather than epistemic factuality, allowing realistic-looking responses to be labeled successful despite being factually or procedurally incorrect. To address this gap, we propose Sequential Epistemic and Action-Level Validation (SEAV), a verification-centric jailbreak evaluation framework that decomposes responses into ordered steps and evaluates both validity and correctness. SEAV combines LLM-as-a-judge mechanisms for semantic interpretation with retrieval-grounded verification using external knowledge sources, assessing whether generated content is factually co
    
[^198]: 超越词元位置：扩散语言模型中跨去噪步骤的安全对齐

    Beyond Token Positions: Safety Alignment Across Denoising Steps in Diffusion Language Models

    [https://arxiv.org/abs/2609.00495](https://arxiv.org/abs/2609.00495)

    该研究发现扩散语言模型的拒绝信号集中在早期去噪步骤和回复起始位置，并提出了一种无需训练的RAEC解码方法，通过在早期步骤提交持续的拒绝信号来提升模型安全性。

    

    扩散大语言模型（dLLMs）通过迭代去噪而非从左到右的解码方式生成文本。这种生成范式引入了两个可能影响安全对齐的维度：词元在去噪过程中何时生成，以及它们在回复中出现在什么位置。在本文中，我们通过追踪整个去噪过程中的中间词元分布和承诺决策，测量了dLLM在有害提示下的安全行为。我们的分析表明，拒绝信号集中在早期去噪步骤和回复的起始位置，且早期提交的词元能够强烈影响最终的安全结果。我们的测量进一步表明，去噪步骤以及拒绝词元承诺的持续性对于理解dLLM的安全性至关重要。基于这些发现，我们提出了拒绝感知早期提交方法（Refusal-Aware Early Commitment, RAEC），这是一种简单的无需训练的解码方法，可以从早期步骤提交持续的拒绝信号。

    arXiv:2609.00495v1 Announce Type: cross  Abstract: Diffusion large language models (dLLMs) generate text through iterative denoising rather than left-to-right decoding. This generation paradigm introduces two axes that can influence safety alignment: when tokens are generated during denoising and where they appear in the response. In this paper, we measure dLLM safety behavior under harmful prompts by tracing intermediate token distributions and commitment decisions throughout denoising. Our analysis shows that refusal signals are concentrated in early denoising steps and leading response positions, and the tokens committed early can strongly shape the final safety outcome. Our measurements further show that the denoising step and persistence of refusal-token commitment are important for understanding dLLM safety. Based on these findings, we propose Refusal-Aware Early Commitment (RAEC), a simple training-free decoding method that commits persistent refusal signals from early steps. Ex
    
[^199]: 差分隐私语言模型中的隐私-幻觉权衡

    The Privacy-Hallucination Tradeoff in Differentially Private Language Models

    [https://arxiv.org/abs/2609.00492](https://arxiv.org/abs/2609.00492)

    本文首次揭示并系统研究了差分隐私语言模型中隐私保护与事实准确性之间的权衡：DP训练会导致模型产生更多幻觉（因为DP机制使输出分布平坦化），而提高事实信息在训练数据中的出现频率可有效降低幻觉风险。

    

    在医疗保健等高风险领域，隐私和事实准确性都至关重要。令人担忧的是，我们发现并研究了差分隐私（DP）语言模型中存在的隐私-幻觉权衡问题。首先，我们通过实证表明，采用DP进行预训练或微调的模型往往比非DP的对应模型产生更多幻觉，且随着隐私预算收紧，幻觉的严重程度会增加。其次，我们研究了驱动这种权衡的模型特性，证明DP机制会使输出分布趋于平坦，可能将概率质量重新分配到事实上错误的替代选项上。第三，通过在训练数据中控制事实出现频率的实验，我们刻画了信息频率如何降低DP模型中的幻觉风险。总体而言，我们的研究结果强调了需要更精细的隐私保护干预措施，以便在不损害事实准确性的前提下提供严格的隐私保证。

    arXiv:2609.00492v1 Announce Type: new  Abstract: Both privacy and factual accuracy are paramount in high-stakes domains like healthcare. Concerningly, we uncover and investigate a privacy-hallucination tradeoff in differentially private (DP) language models. First, we empirically show that models pre-trained or fine-tuned with DP tend to produce more hallucinations than non-DP counterparts, with increased severity as the privacy budget grows stricter. Second, we investigate model properties driving this tradeoff, demonstrating that DP mechanisms flatten output distributions, potentially redistributing probability mass toward factually incorrect alternatives. Third, through experiments where we control fact frequency in training data, we characterize how information frequency can reduce hallucination risks in DP models. Overall, our findings underscore the need for more nuanced privacy-preserving interventions that offer rigorous privacy guarantees without compromising factual accuracy.
    
[^200]: EvoFlint：多轮LLM漏洞的进化图谱

    EvoFlint: An Evolutionary Atlas of Multi-Turn LLM Vulnerabilities

    [https://arxiv.org/abs/2609.00487](https://arxiv.org/abs/2609.00487)

    提出了EvoFlint框架，将多轮红队测试从生成问题重新定义为搜索问题，通过进化式质量多样性搜索演化分阶段对话攻击策略，构建出目标模型漏洞的结构化图谱。

    

    前沿语言模型在单轮有害提示下往往会拒绝回答，但当同样的有害意图通过多轮对话逐步达成时，它们却常常配合执行，这使得多轮攻击成为大型语言模型最不为人理解的失效模式之一。大多数自动化红队测试方法将其视为一个生成问题：生成能够攻破模型的攻击。我们认为将其更好地表述为一个搜索问题：发现、组织并迭代优化一个多样化的攻击策略档案库，从而生成一张关于目标模型如何失效的结构化地图，而非一次性的成功攻击列表。我们提出了EvoFlint，它将进化式质量多样性搜索应用于多轮红队测试。攻击策略是分阶段的对话计划，而非原始提示词，并通过LLM驱动的变异和交叉操作进行演化。基于攻击成功率和峰值严重程度的帕累托适应度保留了来自“险些成功”攻击的选择信号。一个以风险为索引的档案库运行新颖性搜索……

    arXiv:2609.00487v1 Announce Type: cross  Abstract: Frontier language models that refuse harmful single-turn prompts often comply when the same intent is reached gradually over many turns, making multi-turn attacks one of the least understood failure modes of large language models. Most automated red-teaming methods treat this as a generation problem: produce attacks that break the model. We argue it is better framed as a search problem: discover, organize, and iteratively refine a diverse archive of attack strategies, producing a structured map of how a target model fails rather than a list of one-off successes. We introduce EvoFlint, which applies evolutionary quality-diversity search to multi-turn red-teaming. Attack strategies are phased conversation plans, not raw prompts, and are evolved through LLM-driven mutation and crossover. A Pareto fitness over attack success rate and peak severity preserves selection signal from near-miss attacks. A risk-indexed archive runs novelty search
    
[^201]: EGT-KG：面向小型语言模型实用科学问答的证据支撑型类型化知识图谱检索

    EGT-KG: Evidence-Grounded Typed KG Retrieval for Practical Scientific QA with Small Language Models

    [https://arxiv.org/abs/2609.00479](https://arxiv.org/abs/2609.00479)

    该论文提出EGT-KG框架，通过证据支撑的类型化知识图谱检索克服本地小型语言模型在文献规模小、证据碎片化条件下的科学问答局限，并比较了自动生成与专家定义两种关系模式的效果。

    

    对于新兴科学研究领域，本地小型语言模型正变得更具吸引力，因为与大型语言模型相比，它们提供了更强的隐私控制和更稳定的部署流程。然而在实践中，基于小型语言模型的科学问答往往面临不可避免的约束：文献集合规模小、证据碎片化、上下文窗口和推理能力有限。我们提出了证据支撑型类型化知识图谱，这是一个用于改进本地小型语言模型信息检索的检索框架。我们评估了三种问答设置：原始检索增强生成工作流程，以及两种EGT-KG工作流程：自动生成关系模式（AS）和专家定义关系模式（ES）。我们的实验采用六维评估框架（S3CRF：健全性、正确性、完整性、简洁性、相关性、流畅性）在生物聚合物相关的（摘要在此处被截断）数据集上进行了评估。

    arXiv:2609.00479v1 Announce Type: new  Abstract: For emerging scientific research domains, local Small Language Models (SLMs) are becoming more attractive, as they offer stronger privacy control and more stable deployment pipelines than Large Language Models. However, in practice, scientific question-answering on SLMs often operates under inevitable constraints: small literature collections, fragmented evidence, limited context window and reasoning abilities. We propose the Evidence-Grounded Typed Knowledge Graph (EGT-KG), a retrieval framework to improve information retrieval with local SLMs. We assessed three question-answering settings: a vanilla Retrieval-Augmented Generation (RAG) workflow and two EGT-KG workflows: an automatically generated relation schema (AS) and an expert-defined relation schema (ES). Our experiments were evaluated with a six-dimensional evaluation framework (S3CRF: Soundness, Correctness, Completeness, Conciseness, Relevance, Fluency) on a Biopolymer-bound So
    
[^202]: 探索语言智能体与非语言智能体之间的协作

    Exploring Collaboration between a language and a non-language agent

    [https://arxiv.org/abs/2609.00474](https://arxiv.org/abs/2609.00474)

    该论文提出LLAMIA-Bench基准，用于研究将非语言智能体的连续表示“言语化”为文本是否成为LLM协作的瓶颈，并提出潜在状态内化方法来改善LLM与国际象棋引擎等非语言智能体的协作。

    

    大型语言模型（LLM）越来越多地被部署为协调者，通过自然语言调度专门的子智能体来解决复杂任务。然而，在博弈和机器人技术等许多重要领域，目前最强的智能体并非语言模型。将非语言智能体与LLM集成需要进行“言语化”：在每个交互步骤中，将其丰富的连续表示压缩为稀疏的文本摘要。为了研究言语化是否构成瓶颈，我们提出了LLAMIA-Bench，这是一套包含六个多样化协作式国际象棋任务的基准，涵盖三个方面：行为模仿、状态评估和自然语言解释。每个任务都对应一个经典的国际象棋难题，无论是LLM还是象棋引擎都无法独立解决。为了实现LLM与非语言智能体的协作，我们提出了“潜在状态内化”方法，将子智能体的连续表示投影到……

    arXiv:2609.00474v1 Announce Type: cross  Abstract: LLMs are increasingly deployed as orchestrators that coordinate specialized subagents to solve complex tasks through natural language. However, in many important domains like game playing and robotics, the strongest available agents are not language models. Integrating non-language agents with LLMs would require \emph{verbalization}: compressing their rich continuous representations into sparse textual summaries at each interaction step. To study whether verbalization constitutes a bottleneck, we introduce \textsc{LLAMIA-Bench}, a suite of six diverse collaborative chess tasks spanning three facets: behavioral imitation, state assessment, and natural-language explanation. Each task instantiates a well-established chess problem that neither the LLM nor the chess engine can solve alone. To solve LLM collaboration with non-language agents, we introduce \emph{latent state internalization}, which projects the subagent's continuous represent
    
[^203]: 深度学习中的高阶结构

    Higher Structures in Deep Learning

    [https://arxiv.org/abs/2609.00472](https://arxiv.org/abs/2609.00472)

    本文阐述了高元张量运算在深度学习中的重要性，对训练后神经网络中的高元现象进行了新颖的实证研究，并提出了多层感知机的超图推广形式，同时探讨了其与进化算法的联系。

    

    我们提供了一篇关于高元张量运算对深度学习重要性的阐述性介绍。随后，我们对训练后的神经网络中的高元现象进行了新颖的实证研究，提出了多层感知机的一种超图推广形式，并探讨了其与进化算法之间的联系。最后，我们对未来研究中富有前景的方向进行了讨论。

    arXiv:2609.00472v1 Announce Type: cross  Abstract: We provide an expository introduction on the importance of higher-arity tensor operations to deep learning. Then, we conduct a novel empirical investigation of higher-arity phenomenon in trained neural networks, introduce a hypergraphical generalization of the multilayer perceptron, and explore connections to evolutionary algorithms. We conclude with a discussion of promising directions for future research.
    
[^204]: 非凸优化中的运行域：一种基于乘子的分类体系

    Operational Regimes in Non-Convex Optimization: A Multiplier-Based Taxonomy

    [https://arxiv.org/abs/2609.00471](https://arxiv.org/abs/2609.00471)

    本文通过KKT点处拉格朗日乘子的结构指纹，为约束非凸优化建立了与算法无关的五域分类体系，并对八类经典算法给出了统一的博弈论解释。

    

    本文基于KKT平稳点处拉格朗日乘子的特征签名，为约束非凸优化引入了一种结构化分类体系。通过对八类经典算法族——包括块坐标下降、ADMM、广义Benders分解、逐次凸逼近、内点法、镜像下降、Frank-Wolfe和黎曼梯度下降——的统一博弈论诠释，我们证明归一化乘子向量携带一种与算法无关的结构指纹。该向量的四个无量纲形状特征将对偶空间划分为五个运行域：无约束域、资源受限域、饱和域、强耦合域和混合域。我们建立了四个刻画该划分的结构性定理：在自然KKT对称性下的不变性、数据扰动下基于Robinson强正则性并具有显式Lipschitz裕度的局部稳定性、余维……

    arXiv:2609.00471v1 Announce Type: cross  Abstract: This paper introduces a structural taxonomy for constrained non-convex optimization based on the signature of Lagrange multipliers at KKT stationary points. Leveraging a unified game-theoretic interpretation of eight classical algorithm families--including block coordinate descent, ADMM, generalized Benders decomposition, successive convex approximation, interior-point methods, mirror descent, Frank-Wolfe, and Riemannian gradient descent--we show that the normalized multiplier vector carries an algorithm-independent structural fingerprint. Four scale-free shape features of this vector partition the dual space into five operational regimes: Unconstrained, Resource-Limited, Saturation, Strongly-Coupled, and Hybrid. We establish four structural theorems characterizing the partition: invariance under natural KKT symmetries, local stability under data perturbation with explicit Lipschitz margins from Robinson's strong regularity, codimensio
    
[^205]: 推理能否缓解后门攻击？一个神经符号视角

    Does Reasoning Mitigate Backdoor Attacks? A Neuro-Symbolic Perspective

    [https://arxiv.org/abs/2609.00464](https://arxiv.org/abs/2609.00464)

    本文首次系统评估了针对神经符号模型的后门攻击，通过将DeepProbLog与基线神经网络进行对比，指出神经符号整合过程可能成为攻击切入点，NeSy模型的对抗鲁棒性亟待深入研究。

    

    神经符号人工智能近来作为一种实现可信人工智能的新范式而兴起，旨在将亚符号神经感知与扎根的符号推理相融合。这些模型所特有的神经符号整合过程已被证明有助于构建更加透明、可解释且高效的人工智能系统。与此同时，它们在对抗环境下的特性却常被忽视，往往被默认认为具有“设计即稳健”的属性。然而，这些模型所采用的神经符号整合过程构成了一层额外的复杂性，可能为攻击提供切入点。因此，本文主张有必要对NeSy模型的对抗鲁棒性进行深入探究，并首次提供了针对NeSy模型后门攻击的系统性评估。为此，我们将最流行的NeSy框架——DeepProbLog——与基线神经网络进行了比较，共涉及八个基（准）（摘要原文在此截断）

    arXiv:2609.00464v1 Announce Type: cross  Abstract: Neuro-Symbolic (NeSy) AI has recently emerged as a novel paradigm to enable trustworthy AI, aiming at integrating sub-symbolic neural perception with grounded symbolic reasoning. The neuro-symbolic integration process that characterizes these models has been proven beneficial to achieve more transparent, explainable and efficient AI systems. Meanwhile, their properties under adversarial settings have been overlooked being frequently deemed robust-by-design. However, the neural-symbolic integration process they leverage constitutes an additional layer of complexity that may provide an attack entry-point. Therefore, in this paper, we claim that an in-depth investigation of the adversarial robustness of NeSy models is necessary and provide the first systematic evaluation of backdoor attacks against NeSy. To this end, we compare the most popular NeSy framework, namely DeepProbLog, against baseline neural networks across a total of eight ba
    
[^206]: 迈向面向LLM智能体的基于信念的世界模型

    Towards a Belief-Based World Model for LLM Agents

    [https://arxiv.org/abs/2609.00455](https://arxiv.org/abs/2609.00455)

    该论文提出基于信念的世界模型（BB-WMs），通过构建并维护可被LLM查询的信念来捕捉部分可观测环境下当前状态的已知与不确定信息，弥补了仅靠动作模拟进行决策的不足。

    

    大型语言模型（LLM）正在许多领域中被用作自主决策与规划的策略。尽管LLM具备强大的推理能力，但它们在长时程任务上仍然表现不佳，尤其是在部分可观测的环境下。世界模型是在训练和推理阶段都能提升策略性能的一种有前景的方法。在推理阶段，目前的智能体在执行动作之前会使用世界模型来模拟候选动作的后果，从而改善决策。然而，我们认为，仅靠模拟对于部分可观测性下的决策而言是一个不完整的接口：模拟无法充分捕捉关于当前状态的不确定性，而智能体要进行准确决策可能需要这些信息。我们通过基于信念的世界模型（Belief-Based World Models, BB-WMs）来解决这一局限，该模型构建并维护一种信念，LLM可以对其进行查询，以获取关于当前状态哪些是已知的、哪些是不确定的信息。

    arXiv:2609.00455v1 Announce Type: new  Abstract: Large language models (LLMs) are being used as policies for autonomous decision-making and planning in many domains. Despite their strong reasoning capabilities, LLMs struggle with long-horizon tasks, especially under partial observability. World models are a promising way to enhance policy performance, both during training and inference. During inference, agents currently use world models to simulate the consequences of candidate actions before committing to an action, which can improve decision-making. However, we argue that simulation alone is an incomplete interface for decision-making under partial observability: simulation doesn't adequately capture uncertainty about the current state, which agents may need for accurate decision-making. We address this limitation with Belief-Based World Models (BB-WMs), which model and maintain a belief that LLMs can query to access information on what is known and uncertain about the current state
    
[^207]: mimeo：将公开专家语料编译为智能体技能并测试其可迁移性

    mimeo: Compiling Public Expert Corpora into Agent Skills and Testing What Transfers

    [https://arxiv.org/abs/2609.00453](https://arxiv.org/abs/2609.00453)

    mimeo 是一个开源工具，可将某位专家的公开作品编译为可供智能体加载的文件，实验表明它能让智能体可靠回答依赖引文的冷门问题（20题全对），并完全避免基于模型记忆生成人设时对专家已记录立场的错报。

    

    给智能体提供一份关于某位具名专家的文件，可以提供难以获取的资料、塑造可识别的人设形象，或改变智能体的决策。这些是不同的主张，我们逐一进行了测试。mimeo 是一个开源工具，它能查找某人的公开作品，将每条提取的引文与缓存的原始文本进行核对，并写出一个可供智能体加载的文件。八次记录在案的构建平均使用38次模型调用；核对环节拒绝了13.2%的提取引文。我们用一个编码智能体测试框架测试了四份专家文件。知识获取的效果最为明显：mimeo 回答了全部20道冷门且依赖引文的问题，而闭卷条件下没有任何方法回答超过10道。对相同页面进行关键词搜索（BM25）可回答15-17道，本样本无法确定这一差距。事实核验方面显示出一个明确益处：在所有评分者下，基于模型记忆撰写的人设在20个答案中有1-4个错报了有据可查的专家立场，而普通智能体和 mimeo 从未出错。每个人设都容易被识别……（原文摘要在此处截断）

    arXiv:2609.00453v1 Announce Type: new  Abstract: Giving an agent a file about a named expert can supply hard-to-find material, produce a recognizable persona, or change what the agent decides. These are different claims. We test each one. mimeo is an open-source tool that finds a person's public work, checks each extracted quotation against the cached source text, and writes a file an agent can load. Eight logged builds averaged 38 model calls; the check rejects 13.2% of extracted quotations. We tested four expert files with one coding-agent harness. Knowledge access was clearest: mimeo answered all 20 obscure, quotation-heavy questions; no closed-book condition answered more than 10. Keyword search (BM25) over the same pages answered 15-17, a gap this sample cannot resolve. Grounding showed one clear benefit: personas written from model memory misstated a documented position on 1-4 of 20 answers under every grader; the plain agent and mimeo never did. Every persona was easy to spot on
    
[^208]: HBQ：面向高精度大语言模型推理的硬件效率感知分层缩放块量化

    HBQ: Hierarchical Scaling Block Quantization with Hardware-Efficiency-Aware Design for Accurate LLM Inference

    [https://arxiv.org/abs/2609.00450](https://arxiv.org/abs/2609.00450)

    提出硬件效率感知的分层块量化方法HBQ，突破块量化中块大小与精度之间的固有权衡，在大块设计下同时实现高硬件效率与精确的大语言模型推理。

    

    块量化是实现大语言模型（LLM）高效部署的一种有前景的方法，能够在精度可控下降的前提下实现低精度计算。与标量仅权重量化（WoQ）相比，块量化同时对权重和激活进行量化，具有更高的硬件效率，并能在统一数据通路上实现端到端推理，但其设计空间（涵盖位宽、块大小、缩放方式和数值格式）仍未得到充分探索。我们通过设计空间探索（DSE）提供了硬件与基准测试结果。我们发现，增大块大小可以通过摊销反量化和累加成本来提高硬件效率，但会降低精度。这种权衡限制了传统块量化方法的应用。受此洞察启发，我们提出分层块量化（HBQ）。与先前采用小块以及传统2的幂（PoT）或基于整数的缩放的方法不同，HBQ使用大块来……（摘要原文在此截断）

    arXiv:2609.00450v1 Announce Type: cross  Abstract: Block Quantization (BQ) is a promising approach for efficient deployment of large language models (LLMs), enabling low-precision computation with controlled accuracy degradation. Compared to scalar weight-only quantization (WoQ), BQ quantizes both weight and activation, offering higher hardware efficiency and end-to-end inference on a unified datapath, but its design space, spanning bit-width, block size, scaling, and numeric formats, remains underexplored.   We provide hardware/benchmark results through design space exploration (DSE). We find that increasing block size improves hardware efficiency by amortizing dequantization and accumulation costs, but degrades accuracy. This trade-off limits conventional BQ methods.   Motivated by this insight, we propose Hierarchical Block Quantization (HBQ). Unlike prior methods [1], [2], which use small blocks and conventional Power-of-Two (PoT) or integer-based scaling, HBQ uses large blocks to 
    
[^209]: 研究ES-HyperNEAT的超参数优化与可迁移性：一种TPE方法

    Investigating Hyperparameter Optimization and Transferability for ES-HyperNEAT: A TPE Approach

    [https://arxiv.org/abs/2609.00449](https://arxiv.org/abs/2609.00449)

    本研究采用树结构Parzen估计器（TPE）优化ES-HyperNEAT的超参数，在MNIST任务上以更小的种群规模和更少的进化代数超越了以往研究的准确率，并验证了优化后超参数在逻辑运算和Fashion-MNIST任务上的可迁移性。

    

    增强拓扑神经进化（NEAT）及其进阶版本可进化基底HyperNEAT（ES-HyperNEAT）在开发神经网络方面展现出巨大潜力。然而，其有效性在很大程度上取决于超参数的选择。本研究使用树结构Parzen估计器（TPE）在MNIST分类任务上研究ES-HyperNEAT超参数的优化，探索了超过30亿种潜在组合的庞大搜索空间。TPE有效地在这个广阔空间中进行搜索，在平均、中位数和最佳准确率方面显著优于随机搜索。在验证过程中，TPE找到的最佳超参数配置在MNIST上达到了29.00%的准确率，超越了以往的研究，同时使用了更小的种群规模和更少的进化代数。研究还探索了优化后超参数在逻辑运算和Fashion-MNIST任务中的可迁移性，显示出成功……

    arXiv:2609.00449v1 Announce Type: cross  Abstract: Neuroevolution of Augmenting Topologies (NEAT) and its advanced version, Evolvable-Substrate HyperNEAT (ES-HyperNEAT), have shown great potential in developing neural networks. However, their effectiveness heavily depends on the selection of hyperparameters. This study investigates the optimization of ES-HyperNEAT hyperparameters using the Tree-structured Parzen Estimator (TPE) on the MNIST classification task, exploring a search space of over 3 billion potential combinations. TPE effectively navigates this vast space, significantly outperforming random search in terms of mean, median, and best accuracy. During the validation process, the best hyperparameter configuration found by TPE achieves an accuracy of 29.00\% on MNIST, surpassing previous studies while using a smaller population size and fewer generations. The transferability of the optimized hyperparameters is explored in logic operations and Fashion-MNIST tasks, revealing succ
    
[^210]: 能力门控语言模型：安全性可组合，实用性不可组合

    Capability-Gated Language Models: Security Composes, Utility Does Not

    [https://arxiv.org/abs/2609.00445](https://arxiv.org/abs/2609.00445)

    提出在单一模型权重内部实现按主体的“能力门控部署”，配置构成格结构，并证明安全限制在交运算下可组合累积（安全性随限制叠加而增强），而实用性不具备这种组合性。

    

    已部署的语言模型安全防护措施（安全微调、过滤、遗忘学习）仅在模型权重之外按主体区分：过滤器被重新配置、访问层级成倍增加、模型制品被重新发布；而在同一组权重内部，所有请求面对的都是相同的模型配置。这促使我们定义“能力门控部署”：在一组权重内部实现按主体的访问控制，其配置构成一个格——交运算累积某个主体的限制，并运算汇集一个联盟的权限范围。我们通过在现有嵌套分解机制上实施稀疏秩门控来实例化该方法，利用单次遍归因引导配置搜索，并仅从预注册的保留集划分中一次性读取全部结果。安全性可组合：在单调引出假设下，交运算处的组合性可被证明，而我们在逐点意义上证伪了该假设。在两个模型谱系中，保留集上交运算的中位数效应加深了抑制；经多重校正后唯一存活的效应进一步强化了这种抑制。实用性却不可组合：……（原文摘要在此处截断）

    arXiv:2609.00445v1 Announce Type: cross  Abstract: Deployed language model safeguards (safety fine-tuning, filtering, unlearning) vary by principal only outside the model weights: filters are reconfigured, tiers are multiplied, and artefacts are reissued; inside one set of weights every request meets the same model configuration. This motivates us to define capability-gated deployment: per-principal access control inside one set of weights, whose configurations form a lattice - meets accumulate a principal's restrictions and joins pool a coalition's reach. We instantiate it by sparse rank gating over an existing nested-factorisation mechanism, guide profile search with one-pass attribution, and read every result once from a pre-registered held-out split. Security composes: provably at meets under a monotone-elicitation assumption we falsify pointwise. In two lineages the median held-out meet deepens suppression; the one effect surviving correction strengthens it. Utility does not: indi
    
[^211]: （视觉）语言模型能够超越表面共现进行泛化：来自跨模态数一致性的证据

    (V)LMs generalize beyond surface co-occurrence: Evidence from cross-modal number agreement

    [https://arxiv.org/abs/2609.00443](https://arxiv.org/abs/2609.00443)

    该研究通过跨模态泛化实验证明，视觉语言模型在学习新名词后，能将仅从视觉线索获得的语法数知识泛化到语言层面，表明模型掌握的是抽象的语法规则，而非仅仅依赖表面词汇共现。

    

    语言模型主要从共现中学习语法数，并因此表现出频率效应——这有时被解读为它们并未学习抽象的“规则”，而是依赖于特定的词汇项。仅用文本刺激来测试泛化无法解决这一争论，因为分布线索（is/are、this/these）很容易直接暴露数的信息。我们转而采用跨模态泛化作为工具，来研究同时能接受视觉输入的语言模型（VLMs）中的抽象能力，将用于诊断数的证据限制在语言之外的模态中。我们通过添加新的嵌入向量并仅在学习过程中更新这些向量，来教授VLMs成对的新名词，并对比了仅由视觉线索诊断数的条件与由文本消歧的条件。在行为、表征动力学和因果机制等多个层面，我们发现了跨模态泛化的实质性证据……

    arXiv:2609.00443v1 Announce Type: cross  Abstract: Language models learn about grammatical number primarily from co-occurrence, and show frequency effects as a result---sometimes taken to indicate that they do not learn abstract ``rules'', and are instead dependent on specific lexical items. Testing generalization with text stimuli alone cannot settle this debate, since distributional cues (is/are, this/these) easily give number away. We instead use cross-modal generalization as a tool to investigate abstractions in LMs that can also accept visual inputs (VLMs), restricting the evidence that diagnoses number to an extra-linguistic modality. We teach VLMs pairs of new nouns by adding new embeddings and only updating them during learning, comparing conditions where number is diagnosed by visual cues alone against ones where it is disambiguated by text. Across behavior, representational dynamics, and causal mechanisms, we find non-trivial evidence for cross-modal generalization across bot
    
[^212]: 会话教练：一种帮助练习职场困难对话的语音AI系统

    Conversation Coach: A Voice-enabled AI System that Helps Practice Difficult Workplace Conversations

    [https://arxiv.org/abs/2609.00441](https://arxiv.org/abs/2609.00441)

    提出了一种语音优先的AI系统Conversation Coach，通过可配置个性的语音对话让管理者真实演练职场困难沟通，并提供内容与合规性的个性化反馈。

    

    管理者与员工之间的高效沟通对于留住高绩效员工和发展低绩效员工至关重要，然而对管理者进行这些技能的培训成本依然很高。基于文本的聊天机器人提供了一种可扩展的方案，但无法提供真实的演练体验：管理者需要通过大声说话的练习来建立自信，以应对高风险的对话。本文提出了Conversation Coach，一个以语音为核心的AI系统，使管理者能够以真实的口语形式演练困难的职场对话。该系统解决了三个挑战：实现低延迟交互同时具备强大的语言理解能力、通过可配置的机器人个性来模拟不同类型的员工从而实现自适应对话，以及针对内容和政策合规性生成个性化反馈。我们将端到端的语音到语音模型与结合自动语音识别、大语言模型的级联方法进行了比较。

    arXiv:2609.00441v1 Announce Type: new  Abstract: Effective manager-employee communication is critical for retaining high performers and developing underperformers, yet training managers in these skills remains costly. Text-based chatbots offer a scalable approach but cannot provide realistic rehearsal: managers need to practice speaking aloud to build confidence before high-stakes conversations. In this paper, we propose Conversation Coach, a voice-first AI system that enables managers to rehearse difficult workplace conversations in a realistic spoken format. The system addresses three challenges: achieving low-latency interactions with strong language understanding, enabling adaptive conversations through configurable bot personalities that simulate different employee types, and generating personalized feedback on content and policy compliance. We compare an end-to-end speech-to-speech model with a cascaded approach combining automatic speech recognition, a large language model, and 
    
[^213]: SAGE：面向任务型对话智能体的状态接地、弃权感知评估

    SAGE: State-Grounded, Abstention-Aware Evaluation of Task-Oriented Dialogue Agents

    [https://arxiv.org/abs/2609.00434](https://arxiv.org/abs/2609.00434)

    SAGE提出将工作流规范编译为原子准则，通过会弃权而非猜测的符号与编码器/NLI验证器级联来评估任务型对话智能体每轮的状态推进，其中SAGE-Core可在零付费LLM成本下判定81-91%的准则。

    

    评估任务型对话智能体不仅要判断回复是否读起来流畅，还要判断每一轮对话是否正确推进了底层工作流状态——传统整体式LLM评判器往往忽略这一区别，因为它们将可用上下文作为单一整体进行评估，且每轮都需要一次或多次完整模型调用。我们提出SAGE（状态接地、弃权感知评估），该方法将工作流规范和逐轮状态差异编译为原子化的、基于模式的准则，并将每条准则通过符号验证器与编码器/NLI验证器构成的级联进行路由，这些验证器在不确定时选择弃权而非猜测，最终将各准则的判定聚合为带有证据轨迹的轮级决策。其推荐的运行配置SAGE-Core仅依靠编译器、符号规则和设备端编码器即可判定81-91%的准则，且零付费LLM成本；SAGE-LLM则针对开放类准则增加了可选的聚焦LLM回退机制。在跨越四个切片的……（摘要原文在此处截断）

    arXiv:2609.00434v1 Announce Type: new  Abstract: Evaluating task-oriented dialogue agents requires judging not merely whether a reply reads well but whether each turn advances the underlying workflow state correctly--a distinction conventional holistic LLM judges can miss because they evaluate the available context as a single unit and require one or more full-model calls per turn. We propose SAGE (State-Grounded Abstention-Aware Evaluation), which compiles a workflow specification and per-turn state diff into atomic, schema-grounded criteria and routes each through a cascade of symbolic and encoder/NLI verifiers that abstain rather than guess, aggregating criterion verdicts into a turn-level decision with an evidence trace. Its recommended operating point, SAGE-Core, decides 81--91% of criteria with only the compiler, symbolic rules, and on-device encoders--at zero paid LLM cost--while SAGE-LLM adds an optional focused-LLM fallback for open-class criteria. Across four slices spanning 
    
[^214]: SpecMind：通过多智能体混合检索增强生成实现频谱智能

    SpecMind: Enabling Spectrum Intelligence via Multi-Agent Hybrid Retrieval-Augmented Generation

    [https://arxiv.org/abs/2609.00427](https://arxiv.org/abs/2609.00427)

    提出了SpecMind——一个基于多智能体混合检索增强生成（RAG）的频谱智能系统，通过协调专门的子智能体对政策文件、法律文本等异构数据源进行检索、推理与知识综合，以应对频谱管理中细粒度决策所带来的海量分散数据处理挑战。

    

    无线设备的指数级增长正在推动前所未有的频谱需求，促使频谱管理向跨空间、时间和设备约束的更细粒度决策方向发展。因此，频谱政策制定者和工程师必须处理来自多种来源、形式多样的大量数据，例如文本和表格。这些数据源通常彼此分散，整合、搜索和解释它们需要大量的时间和精力。此外，这些信息大多采用面向人类理解的格式，自动化系统难以直接获取。为应对这一挑战，我们提出了SpecMind，这是一个用于频谱智能的新型多智能体检索增强生成（RAG）系统，能够对异构数据源进行推理。该系统使自主智能体能够协调专门的子智能体，跨政策程序、法律等来源检索和综合知识。

    arXiv:2609.00427v1 Announce Type: new  Abstract: The exponential growth of wireless devices is driving unprecedented spectrum demand, pushing spectrum management toward more fine-grained decisions across space, time, and device constraints. As a result, spectrum policymakers and engineers must process large volumes of data that come from diverse sources and take many different forms, such as text and tables. These data sources are often disaggregated and require significant time and effort to integrate, search, and interpret. Furthermore, most of this information is formatted for human understanding and is not readily accessible to automated systems. To address this challenge, we propose SpecMind, a novel Multi-Agent Retrieval-Augmented Generation (RAG) system for spectrum intelligence that performs reasoning over heterogeneous data sources. This system enables autonomous agents to coordinate specialized sub-agents that retrieve and synthesize knowledge across policy proceedings, legal
    
[^215]: 面向金融推理的依赖感知思维链压缩

    Dependency-Aware Chain-of-Thought Compression for Financial Reasoning

    [https://arxiv.org/abs/2609.00413](https://arxiv.org/abs/2609.00413)

    提出分层语义蒸馏网络HSDN，通过依赖图引导的思维链压缩方法，在AFAC2025金融推理基准上以68.4%的压缩率实现91.0%的准确率，兼顾推理效率与答案准确性。

    

    思维链提示能够提升复杂推理能力，但其冗长的中间推理轨迹带来了巨大的推理开销，阻碍了在金融场景中的实际部署。我们提出了一种分层语义蒸馏网络用于压缩推理链，同时保持答案准确性与逻辑连贯性。该框架融合了语义分割、依赖图构建、双编码器重要性评分、约束片段选择和局部边界重写。一个冻结的 Qwen3 4B 模型仅用于特征提取和最终答案生成，而压缩过程始终保持结构化与可解释性。在 AFAC2025 基准测试中，HSDN 在 68.4% 的压缩率下取得了 91.0% 的准确率，在综合得分和推理连贯性上均优于强大的压缩基线方法。结果表明，图引导的压缩方法对于高风险金融推理任务是有效的。

    arXiv:2609.00413v1 Announce Type: new  Abstract: Chain of thought prompting improves complex reasoning, but its long intermediate traces create substantial inference cost and hinder practical deployment in financial settings. We present a Hierarchical Semantic Distillation Network, HSDN, for compressing reasoning chains while preserving answer accuracy and logical coherence. The framework combines semantic segmentation, dependency graph construction, dual encoder importance scoring, constrained segment selection, and local boundary rewriting. A frozen Qwen3 4B model is used only for feature extraction and final answer generation, while the compression process remains structured and interpretable. On the AFAC2025 benchmark, HSDN achieves 91.0% accuracy with 68.4% compression, outperforming strong compression baselines in overall score and reasoning coherence. The results show that graph guided compression is effective for high stakes financial reasoning tasks.
    
[^216]: 面向自主超车的风险感知决策：基于世界模型的专家混合框架

    Risk-Aware Decision-Making for Autonomous Overtaking: A World Model-Based Mixture-of-Experts Framework

    [https://arxiv.org/abs/2609.00385](https://arxiv.org/abs/2609.00385)

    本文提出基于世界模型的风险感知专家混合框架，利用学习到的潜在动力学模型进行并行多步推演，将安全评估从动作层面提升到轨迹层面的累积风险水平，并通过分层门控机制动态协调专家以适应不同交互强度，从而提升自主超车决策的长期安全性。

    

    自主高速公路超车需要具备前瞻性的决策能力，以应对复杂的交互行为、随机的交通演化以及时间维度上的风险累积。然而，标准的安全强化学习方法通常依赖于基于价值的隐式风险估计，而非显式的动力学建模，因而难以准确捕捉多步时域内复杂的风险传播过程。这一局限常常导致智能体表现出局部安全、但长期来看会引发较大潜在风险的行为。为解决这一问题，本文提出了一种基于世界模型的风险感知专家混合框架。首先，通过学习得到的潜在动力学模型支持并行的多步推演，借助累积风险评估将安全性评估从动作层面提升到轨迹层面。其次，为了增强在不同交互强度下的鲁棒性，一种分层门控机制能够动态协调各专家……

    arXiv:2609.00385v1 Announce Type: cross  Abstract: Autonomous highway overtaking demands foresighted decision-making to handle complex interactions, stochastic traffic evolution, and temporal risk accumulation. However, standard safe reinforcement learning approaches typically rely on implicit value-based risk estimations rather than explicit dynamics modeling, thereby struggling to accurately capture complex risk propagation over multi-step horizons. This limitation frequently results in behaviors that are locally safe but induce substantial latent risks in the long term. To address this, a World Model-based Risk-aware Mixture-of-Experts (WM-RMoE) framework is proposed. First, a learned latent dynamics model facilitates parallel multi-step rollouts, elevating safety assessment from the action level to the trajectory level via cumulative risk evaluation. Second, to enhance robustness under varying interaction intensities, a hierarchical gating mechanism dynamically coordinates experts 
    
[^217]: RestoreBench：AI智能体能够恢复潮流计算的收敛性吗？

    RestoreBench: Can AI Agents Restore Power Flow Convergence?

    [https://arxiv.org/abs/2609.00384](https://arxiv.org/abs/2609.00384)

    提出了RestoreBench基准测试，用于评估大语言模型智能体在聊天机器人、单智能体和多智能体三种架构下诊断并解决电力系统潮流不收敛问题的能力。

    

    大语言模型（LLM）智能体正日益通过工具使用、中间结果解读和迭代规划来实现多步骤工程工作流程的自动化。诊断和解决潮流计算不收敛问题是一个有前景但基本尚未被探索的应用领域，因为它需要工程判断、实验探索以及在受限动作空间内的决策能力。我们提出了一个基准测试，用于在多个大语言模型和三种架构（聊天机器人、单智能体和多智能体系统）上评估这些能力。该评估涵盖两个电网，每个电网包含46个案例，每个案例都需要一项或多项纠正措施来恢复收敛。该基准测试定义了仿真环境、观测空间、动作空间和评估指标，为开发面向电力系统规划与运行的智能体AI系统提供了可复现的基础。

    arXiv:2609.00384v1 Announce Type: new  Abstract: Large Language Model (LLM) agents increasingly automate multi-step engineering workflows through tool use, interpretation of intermediate results, and iterative planning. Diagnosing and resolving non-convergent power flow cases is a promising yet largely unexplored application, as it requires engineering judgment, experimentation, and decision-making within constrained action spaces. We introduce a benchmark that evaluates these capabilities across multiple LLMs and three architectures: \emph{chatbot}, \emph{single agent}, and \emph{multi-agent} systems. The evaluation covers two power grids and 46 cases per grid, each requiring one or more corrective actions to restore convergence. The benchmark defines the simulation environment, observation and action spaces, and evaluation metrics, providing a reproducible foundation for developing agentic AI systems for power system planning and operation. The code is available at https://github.com
    
[^218]: FoldingAgent：从演示视频推断参数化折纸程序

    FoldingAgent: Inferring Parametric Origami Procedures from Demonstration Videos

    [https://arxiv.org/abs/2609.00377](https://arxiv.org/abs/2609.00377)

    FoldingAgent是一个基于视觉语言模型的智能体框架，能够从折纸演示视频中直接推断参数化折叠程序，并通过顺序操作与动态重新规划能力有效缓解多步折叠中的误差累积问题。

    

    我们提出了FoldingAgent，这是一个用于从折纸演示视频中直接推断显式参数化折叠程序的智能体框架。该框架利用预训练视觉语言模型（VLM）的推理能力，并配备一套专门的工具，使智能体能够模拟几何变换、验证物理合理性、检索和比较视觉内容，以及评估自身的预测。为了将视觉内容转换为折叠程序，我们定义了一个由纸张几何形状和一组参数化折叠动作组成的参数空间。与预测静态折痕图案的模型不同，我们的智能体以顺序方式操作，并具备重新规划动作的能力，有效缓解了多步折叠中固有的误差累积问题。我们的方法朝着弥合人类折纸知识（主要通过非结构化视觉演示共享）与机器可执行程序之间差距的目标迈出了一步。

    arXiv:2609.00377v1 Announce Type: cross  Abstract: We present FoldingAgent, an agentic framework for inferring explicit parametric folding programs directly from origami demonstration videos. Our framework leverages the reasoning power of a pre-trained Vision-Language Model (VLM) equipped with a suite of specialized tools that enable the agent to simulate geometric transitions, verify physical plausibility, retrieve and compare visual content, and evaluate its own predictions. To translate visual content into folding programs, we define a parametric space that consists of the paper's geometry and a set of parametric folding actions. Unlike models that predict static crease patterns, our agent operates sequentially and possesses the ability to re-plan its actions, effectively mitigating the compounding errors inherent in multi-step folding. Our approach takes a step toward closing the gap between human origami knowledge, which is primarily shared through unstructured visual demonstratio
    
[^219]: 无梯度自适应：仿射统计传输及其证书能告诉你什么

    Adapting Without Gradients: Affine Statistics Transport and What Its Certificate Can Tell You

    [https://arxiv.org/abs/2609.00374](https://arxiv.org/abs/2609.00374)

    提出CASTER，一种无需梯度的测试时自适应方法，通过在判别子空间中存储源类别统计并估计仿射变换来解析地传输源类别分布，使冻结模型无需反向传播即可适应目标数据，同时以约18倍更少的状态存储优于k-NN基线。

    

    测试时自适应通常假设模型参数可以在推理阶段进行更新。这一假设对于仅推理加速器、冻结或第三方模型以及内存受限的部署环境具有很强的限制性，而且基于BatchNorm的标准TTA配置在不包含BatchNorm的架构上也可能失效。我们研究了在所学模型必须保持冻结情况下的自适应问题。我们提出了CASTER，这是一种无梯度方法，它在判别子空间中存储源类别统计信息，从目标批次矩中估计一个类别共享的仿射变换，并在分类之前以解析方式传输源类别分布。CASTER不需要反向传播、优化器状态或存储源特征库。在四个骨干网络和七个数据集的实验中，它在28个骨干-数据集组合中的27个上优于基于相同冻结特征的k-NN方法，同时状态存储量的中位数减少18倍。仿射传输并非……（摘要原文在此处截断）

    arXiv:2609.00374v1 Announce Type: cross  Abstract: Test-time adaptation (TTA) typically assumes that model parameters can be updated at inference time. This assumption is restrictive for inference-only accelerators, frozen or third-party models, and memory-constrained deployments, and standard BatchNorm-based TTA configurations may also become inactive on architectures without BatchNorm. We study adaptation when the learned model must remain frozen. We introduce CASTER, a gradient-free method that stores source class statistics in a discriminative subspace, estimates a class-shared affine transformation from target-batch moments, and analytically transports the source class distributions before classification. CASTER requires no backward pass, optimizer state, or stored source feature bank. Across four backbones and seven datasets, it outperforms k-NN on identical frozen features in 27 of 28 backbone-dataset settings while retaining a median of 18x less state. Affine transport is not a
    
[^220]: 面向数据工程的神经符号方法：无需微调实现长上下文Token缩减

    Neurosymbolics for Data Engineering: Achieving Long Context Token Reduction Without Finetuning

    [https://arxiv.org/abs/2609.00367](https://arxiv.org/abs/2609.00367)

    本文提出一种即插即用的神经符号层，无需任何微调或RLHF即可在Text-to-SQL等数据工程任务上平均提升85%的准确率，同时缓解Transformer长上下文的计算资源瓶颈。

    

    大型语言模型正越来越多地被部署用于复杂的数据工程任务，例如从自然语言生成结构化查询（Text-to-SQL）以及自动化复杂的电子表格操作。然而，要最大化其效用，既需要更高的免微调准确率，也需要解决Transformer架构固有的二次方（O(n²)）时间复杂度所带来的计算瓶颈。本文提出了一种新颖的即插即用神经符号层，旨在无缝集成到现有的LLM骨干网络中，增强逻辑推理能力并缓解长上下文的资源消耗。在推理方面，该层能够立即且显著地提升性能，在包括BIRD-CRITIC和LiveSQLBench在内的严格基准测试中实现了平均85%的准确率提升，关键是这些提升无需任何任务特定的微调或RLHF。同时，我们将该方法重新应用于解决长上下文问题……

    arXiv:2609.00367v1 Announce Type: cross  Abstract: Large Language Models are increasingly deployed for sophisticated data engineering tasks such as generating structured queries from natural language, Text-to-SQL, and automating complex spreadsheet operations. However, maximizing their utility demands both higher finetuning-free accuracy and solutions to the computational bottleneck imposed by the Transformer architectures inherent quadratic (On2) time complexity. This paper introduces a novel drop-in neurosymbolic layer designed to seamlessly integrate into existing LLM backbones enhancing logical reasoning and mitigating long-context resource consumption. On the reasoning front, the layer immediately and significantly improves performance yielding an average accuracy increase of 85% across rigorous benchmarks including BIRD-CRITIC and LiveSQLBench, critically achieving these gains without any task specific finetuning or RLHF. Concurrently, we repurpose this approach to address the se
    
[^221]: 反事实脆弱性证书：揭示结构化证据失效下的高置信度脆弱性

    Counterfactual Fragility Certificates: Exposing High-Confidence Brittleness under Structured Evidence Failure

    [https://arxiv.org/abs/2609.00366](https://arxiv.org/abs/2609.00366)

    提出反事实脆弱性证书（CFC），一种模型无关的协议级审计对象，将每个预测映射为由贪婪翻转预算、边际崩塌面积等指标刻画的有序证据失效轨迹，从而揭示表格决策系统中高置信度预测在结构化证据失效下的脆弱性。

    

    高测试准确率和良好的整体校准并不能表明某个个体预测是否在结构上得到了其证据的支持。在表格化决策系统中，当某个特征族变得不可用、延迟、充满噪声、过时或低可信度，而模型却仍然保持高置信度时，故障往往会发生。现有的校准、不确定性、选择性预测、解释和扰动方法提供的只是标量分数或归因图，而不是一个可重新计算的审计对象来回答这样的问题：在声明的证据失效协议下，什么样的轨迹会使该预测失去支持？我们提出了反事实脆弱性证书（Counterfactual Fragility Certificates, CFC），这是一种模型无关的协议级审计证书——而非形式化的鲁棒性证书——它将每个预测映射到一条有序的证据失效轨迹中，并通过贪婪翻转预算、归一化边际崩塌面积、退化阈值和脆弱性支配分数来概括该轨迹。在七个表格基准数据集上……（摘要在此处截断）

    arXiv:2609.00366v1 Announce Type: cross  Abstract: High test accuracy and good aggregate calibration do not show whether an individual prediction is structurally supported by its evidence. In tabular decision systems, failures often occur when a feature family becomes unavailable, delayed, noisy, stale, or low-trust while the model remains highly confident. Existing calibration, uncertainty, selective-prediction, explanation, and perturbation methods provide scalar scores or attribution maps, but not a recomputable audit object answering: under a declared evidence-failure protocol, what trajectory makes this prediction lose support? We introduce Counterfactual Fragility Certificates (CFC), a model-agnostic protocol-level audit certificate-not a formal robustness certificate-that maps each prediction into an ordered evidence-failure trajectory summarized by greedy flip budget, normalized margin-collapse area, degradation thresholds, and fragility dominance score. Across seven tabular be
    
[^222]: Dr. Claw：一个面向氛围式研究的AI科学家工作区

    Dr. Claw: An AI Scientist Workspace for Vibe Research

    [https://arxiv.org/abs/2609.00365](https://arxiv.org/abs/2609.00365)

    Dr. Claw 是一个开源的AI科学家工作区，通过持久化状态对象、可复用技能库和多执行器协调，将现有命令行编码代理封装为可控、可审计的人机协同工作流，把科研中的规划、执行与写作整合为一个可追踪、可恢复的闭环。

    

    命令行编码代理（如 Claude Code、Gemini CLI）已经能够读写文件并维持长会话，然而端到端的科研工作仍然碎片化地分散在聊天工具、IDE、终端和写作环境之间，且那些使研究可审计的决策很少被保存下来。我们提出了 Dr. Claw，一个开源工作区，它将现有的编码代理执行器封装在一个可控且可审计的人机协同工作流中，而非引入另一个自主智能体。持久化的状态对象、可复用的技能库以及多执行器协调机制将人类决策与AI执行联系起来，使规划、执行和写作整合为一个可追踪、可恢复的闭环。我们通过一个交互式三视图场景和一次故障恢复演示来展示 Dr. Claw，并将其与共享同一后端执行器的裸命令行代理进行对比评估，因此该对比考察的是整个编排层（任务图、状态对象等）。

    arXiv:2609.00365v1 Announce Type: new  Abstract: Command-line coding agents (e.g., Claude Code, Gemini CLI) can already read and write files and sustain long sessions, yet end-to-end research still fragments across chat tools, IDEs, terminals, and writing environments, and the decisions that make it auditable are rarely preserved. We present Dr. Claw, an open-source workspace that wraps existing coding-agent executors in a controllable and auditable human-in-the-loop workflow rather than introducing another autonomous agent. Persistent state objects, a reusable skill library, and multi-executor coordination link human decisions to AI execution, turning planning, execution, and writing into one traceable, recoverable loop. We demonstrate Dr. Claw through an interactive three-view scenario and a failure-recovery walkthrough, and evaluate it against a bare command-line agent sharing the same backend executor, so the comparison contrasts the whole orchestration layer (task graph, state obj
    
[^223]: 一种用于量子联邦学习的稳定聚合方法

    A Stable Aggregation Method for Quantum Federated Learning

    [https://arxiv.org/abs/2609.00356](https://arxiv.org/abs/2609.00356)

    该论文提出了一种融合QoS感知客户端加权、循环参数聚合和有界中点更新控制的自洽中点聚合方法，显著提升了量子联邦学习在异构数据与量子硬件噪声等挑战下的稳定性与准确率。

    

    量子联邦学习（QFL）使客户端能够在不共享私有数据的情况下训练量子神经网络（QNN）模型。我们发现，在异构数据、不可靠通信、可变保真度、延迟以及量子硬件噪声等条件下，QFL中的聚合过程是不稳定的。此外，QFL还面临着不小的挑战，因为许多QNN参数是周期性角度，而欧几里得平均方法往往无法捕捉其固有的动态特性。我们开发了一种新颖的自洽中点聚合方法，用于稳定的QFL设计与实现。该方法结合了服务质量（QoS）感知的客户端加权、循环参数聚合以及基于有界中点的更新控制。我们通过多次角度测试和IBM真实量子计算机实验验证了所提方法。在医疗和金融数据集上的大量评估与实验表明，该方法具有更高的稳定性、更低的波动性以及有竞争力的准确率。

    arXiv:2609.00356v1 Announce Type: new  Abstract: Quantum federated learning (QFL) enables clients to train quantum neural network (QNN) models without sharing private data. We find that aggregation in QFL is unstable under heterogeneous data, unreliable communication, variable fidelity, latency, and quantum hardware noise. Moreover, QFL is non-trivially challenging because several QNN parameters are periodic angles, where Euclidean averaging often fails to capture the inherent dynamics. We develop a novel self-consistent midpoint aggregation method for stable QFL design and implementation. We combine QoS-aware client weighting, circular parameter aggregation, and bounded midpoint-based update control. We perform several angular tests and IBM real Quantum machines experiments for validation confirming our approach. Extensive evaluations and experiments on medical and financial datasets show improved stability, lower volatility, and competitive accuracy.
    
[^224]: 视觉并非开销：面向视觉语言模型无损推测解码的单遍块草拟方法

    Vision Is Not Overhead: One-Pass Block Drafting for Lossless Speculative Decoding in Vision-Language Models

    [https://arxiv.org/abs/2609.00355](https://arxiv.org/abs/2609.00355)

    该论文提出 GLANCE——首个在未修改的视觉语言模型上实现无损推测解码的单遍块草拟器，通过块扩散头零成本读取目标模型已融合的视觉-语言状态，并在一次前向传播中完成整块草拟与宽候选树验证，从而打破了草拟器因规模受限而被迫牺牲视觉信息的自我挫败循环。

    

    推测解码能够在不改变输出结果的前提下加速生成，但在视觉语言模型上，它却陷入了一种自我挫败的循环：草拟器必须保持自回归架构，因而只能维持小规模；小型草拟器无法在每一步都承担图像处理的代价，于是视觉信息被压缩、剪枝或隐藏；而被切断了图像信息的草拟器，恰恰在图像最能让文本变得可预测的地方变得最不可靠。我们提出 GLANCE——首个在未经修改的 VLM 目标模型上实现无损解码的单遍块草拟器，它从两端打破了这一循环。一个块扩散头读取目标模型已经融合好的视觉-语言状态，因此视觉对草拟器而言零开销；同时它在一次前向传播中填满整个块，因此模型深度不会带来额外的串行步数。宽候选树通过一次目标模型前向传播即可完成验证，且经审计的每个提示都能精确复现贪婪解码的结果。在依赖视觉依据的工作负载上收益最为显著，会进入一种逐字复制的模式，其长段连续（原文摘要在此处截断）……

    arXiv:2609.00355v1 Announce Type: new  Abstract: Speculative decoding accelerates generation without changing its output, yet on vision-language models (VLMs) it has been caught in a self-defeating cycle. The drafter stays autoregressive, so it must stay small. A small drafter cannot afford the image at every step, so vision is compressed, pruned, or hidden. A drafter cut off from the image is then least reliable exactly where the image makes text predictable. We present GLANCE, the first one-pass block drafter that is lossless on an unmodified VLM target, and it breaks the cycle at both ends. A block-diffusion head reads the target's already-fused vision-language state, so vision costs the drafter nothing, and fills a whole block in one forward pass, so depth costs no sequential steps. A wide candidate tree is verified in one target pass, and every audited prompt reproduces greedy decoding exactly. Grounded workloads reward this most, entering a verbatim-copy regime whose long runs co
    
[^225]: 通过激活匹配微调检测大语言模型中的隐藏行为

    Detecting Hidden Behaviors in LLMs via Activation-matched Finetuning

    [https://arxiv.org/abs/2609.00351](https://arxiv.org/abs/2609.00351)

    论文提出“激活匹配微调”这一无监督检测方法，通过在良性语料上微调锚定模型以复现可疑模型的激活并计算残差，在无需知晓触发器或目标行为的前提下检测出大语言模型中的后门、审查等隐藏行为及其语义邻近提示。

    

    大语言模型可能潜藏一些仅在狭窄条件下才激活的行为，例如后门触发器、睡眠代理部署线索、故意放水或基于话题条件的审查。这类行为在缺乏先验知识（不知道要寻找什么）的情况下难以被检测。我们提出了激活匹配微调，这是一种无监督检测方法，无需对触发器或目标行为的任何先验知识。给定一个可疑模型和一个公开可用的锚定模型，我们在小型良性语料库上微调锚定模型以复现可疑模型的激活，并通过两个模型之间的残差对每个评估提示进行评分。由于没有任何良性语料库能够覆盖稀疏的触发区域，参考模型只会学习到良性计算而学不到隐藏行为。因此，触发提示——以及关键的，它们的语义邻近提示——会产生较大的残差，从而向防御者发出存在异常行为的信号。

    arXiv:2609.00351v1 Announce Type: cross  Abstract: Large language models can hide hidden behaviors that activate only under narrow conditions, such as backdoor triggers, sleeper-agent deployment cues, sandbagging, or topic-conditioned censorship. Such behaviors are difficult to detect without prior knowledge what to look for. We present activation-matched finetuning, an unsupervised detection method that assumes no knowledge of the trigger or the target behavior. Given a suspect model and a publicly available anchor, we finetune the anchor to reproduce the suspect's activations on a small benign corpus, and score each evaluation prompt by the residual between the two models. Since no benign corpus covers the sparse trigger region, the reference learns the benign computation but not the hidden behavior. Therefore, trigger prompts -- and, crucially, their semantic neighbors -- incur a large residual that signal the presence of unusual behavior to the defender. Testing our method across t
    
[^226]: SlideBank：用于一致全切片推理的持久化分层证据库

    SlideBank: A Persistent Hierarchical Evidence Bank for Consistent Whole-Slide Reasoning

    [https://arxiv.org/abs/2609.00342](https://arxiv.org/abs/2609.00342)

    SlideBank提出了一种无需训练的框架，将全切片图像构建为持久化、按概念索引且空间锚定的分层证据库，使问题推理时能够语义化地检索证据并保持与原始视觉内容的关联，从而实现一致的全切片推理。

    

    全切片图像（WSIs）对视觉-语言推理具有挑战性，因为具有诊断意义的形态学信息稀疏、异构，且分布在千兆像素级的图像和多个空间分辨率之上。现有的WSI模型和病理智能体能够聚合切片特征或主动获取证据，但探索后所保留的信息往往难以进行语义化访问，同时又无法保持其与原始视觉证据的关联。我们提出了SlideBank，这是一个无需训练的框架，它将每张WSI表示为一个持久化的、按概念索引的、空间上有锚定的证据库。SlideBank执行与问题无关的由粗到细的探索，以识别信息丰富的区域和多尺度视图，将其转化为显式的形态学观察，并将病理信号锚定到其支持的图像块和WSI坐标上。在推理阶段，问题会被路由到相关的信号和证据……（摘要原文在此处被截断）

    arXiv:2609.00342v1 Announce Type: new  Abstract: Whole-slide images (WSIs) are challenging for vision-language reasoning because diagnostically relevant morphology is sparse, heterogeneous, and distributed across gigapixel-scale images and multiple spatial resolutions. Existing WSI models and pathology agents can aggregate slide features or actively acquire evidence, but the information retained after exploration is often difficult to access semantically while preserving its connection to the original visual evidence. We introduce SlideBank, a training-free framework that represents each WSI as a persistent, concept-indexed, and spatially grounded evidence bank. SlideBank performs question-independent coarse-to-fine exploration to identify informative regions and multi-scale views, converts them into explicit morphological observations, and grounds pathology signals to their supporting patches and WSI coordinates. At inference time, questions are routed to relevant signals and evidence
    
[^227]: 面向负责任AI的人机协同诠释：一种诠释学视角

    Human-AI Co-Interpretation for Responsible AI: A Hermeneutic Perspective

    [https://arxiv.org/abs/2609.00334](https://arxiv.org/abs/2609.00334)

    本文从哲学诠释学视角出发，指出LLM在解释性任务中存在“解释错位”这一失败模式（模型解读被当作确定意义且缺乏解释框架、备选解释与溯源信息），并据此提出构建人机协同诠释的设计原则，以保障负责任AI中的可问责性。

    

    在法律、教育、政策分析和公共道德论证等领域，大语言模型（LLM）的输出常被用于那些需要以文本证据和明确的规范性标准来论证解释的工作。然而，一种反复出现的失败模式——作者称之为“解释错位”（interpretive misplacement）——是模型生成的解读在缺乏明确解释框架（来源、范围约束、规范性承诺）的情况下被当作确定的意义，既不保留可辩护的其他解释，也缺少能让读者找到支撑文本的溯源信息。在这些情境中，风险不仅在于事实错误，更在于问责性的丧失：读者和机构无法可靠地评估某项输出使其承担了何种承诺，以及其依据是什么。本文借鉴哲学诠释学，讨论了这一风险，并推导出用于构建人机协同诠释的设计原则。此外，本文还对近来的学术……（原文摘要在此处被截断）提供了结构化的综述。

    arXiv:2609.00334v1 Announce Type: new  Abstract: Across law, education, policy analysis, and public moral argumentation, LLM outputs are being used often for work that requires interpretations to be justified with textual evidence and explicit normative standards. Yet a recurrent failure mode -- what I call \textit{interpretive misplacement} -- is that model-generated readings get treated as settled meanings without an explicit interpretive frame (sources, scope constraints, normative commitments), without preserving defensible alternatives, and without provenance that lets readers find the supporting passages. In such settings, the risk is not only factual error but lost accountability: readers and institutions cannot reliably assess what an output commits them to, or on what basis. Drawing on philosophical hermeneutics, this paper discusses this risk and derives design principles for structuring human-AI co-interpretation. The paper also provides a structured synthesis of recent scho
    
[^228]: 隐含波动率曲面生成模型的潜空间无套利几何

    Latent-Space No-Arbitrage Geometry of Generative Models for Implied Volatility Surfaces

    [https://arxiv.org/abs/2609.00332](https://arxiv.org/abs/2609.00332)

    本文在潜空间中刻画生成模型输出隐含波动率曲面的无套利约束，通过标量边际定义可容许潜集并证明其在扰动下的稳定性，同时为零边际边界构造水平集方程，且该方法适用于任何具有确定性映射的生成模型。

    

    隐含波动率曲面的生成模型必须产生满足静态无套利约束的输出。我们在潜空间中研究这些约束。对于固定的生成器，我们根据生成曲面的无套利条件为每个潜编码分配一个标量边际。具有非负边际的编码构成可容许潜集。我们建立了严格可容许编码在微小扰动下仍保持可容许的条件，并且可容许集的边界由零边际刻画。对于正则边界分量，我们构造了一个水平集方程，其局部动力学指向零边际集。该分析将生成器视为从潜变量到曲面的映射，因此不限于特定的架构，适用于变分自编码器、生成对抗网络以及其他具有确定性实现映射的生成模型。

    arXiv:2609.00332v1 Announce Type: cross  Abstract: Generative models for implied volatility surfaces must produce outputs that satisfy static no-arbitrage constraints. We study these constraints in latent space. For a fixed generator, we assign each latent code a scalar margin determined by the no-arbitrage conditions of the generated surface. The codes with nonnegative margin form the admissible latent set. We establish conditions under which strictly admissible codes remain admissible under small perturbations and the boundary of the admissible set is characterized by zero margin. For regular boundary components, we formulate a level-set equation whose local dynamics are directed toward the zero-margin set. The analysis treats the generator as a map from latent variables to surfaces and is therefore not restricted to a particular architecture. It applies to variational autoencoders, generative adversarial networks, and other generative models with a deterministic realization map. Num
    
[^229]: 真实场景下的主题匹配：来自真实世界ASR转录文本的基准测试与经验教训

    Topic Matching in the Wild: Benchmark and Lessons from Real-World ASR Transcripts

    [https://arxiv.org/abs/2609.00330](https://arxiv.org/abs/2609.00330)

    该论文构建了一个基于真实呼叫中心ASR转录文本的人工标注主题匹配基准数据集，并通过系统对比发现，配备自然语言主题描述的轻量级大语言模型匹配器在处理噪声转录文本时性能优于句子嵌入和正则表达式方法。

    

    在呼叫中心中，实时坐席辅助工具会针对多个预定义主题中的每一个，判断实时的客户话语是否与之相关，并在相关时向坐席展示辅导卡片。输入数据充满噪声且极具挑战性：即自发电话对话的ASR（自动语音识别）转录文本，这些文本可能不清晰、内容重复，且大多缺乏标点符号。为了系统地研究这一现实世界任务，我们整理了一个基于真实呼叫中心转录文本的人工标注的主题-话语判断数据集。我们比较了三种类型的匹配器：基于正则表达式的基线方法、零样本句子嵌入编码器，以及基于Gemini的大语言模型匹配器。此外，我们的基准中还研究了两种类型的主题表示方式：关键词短语和自然语言描述。我们的实证实验表明，配备自然语言描述的轻量级大语言模型匹配器，其性能显著优于嵌入模型和正则表达式模型。

    arXiv:2609.00330v1 Announce Type: cross  Abstract: In contact centers, real-time agent-assist tools determine, for each of many predefined topics, whether a live customer utterance is relevant and display a coaching card to the agent when it is. The input is noisy and challenging: ASR(Automatic Speech Recognition) transcripts of spontaneous phone conversations, which can be unclear, repetitive, and mostly lack punctuation. To systematically study this real-world task, we curate a human-annotated topic-utterance judgments dataset sourced from real call-center transcripts. We compare three types of matchers: a regex-based baseline, zero-shot sentence embedding encoders, and Gemini-based LLM matchers. In addition, two types of topic representations are studied in our benchmark:keyphrases and natural language description. Our empirical experiments highlight the superior performance of lightweight LLM matchers over embedding and regex models when equipped with natural language descriptions.
    
[^230]: 词汇规范化中的多语言诅咒

    The Curse of Multilinguality in Lexical Normalization

    [https://arxiv.org/abs/2609.00329](https://arxiv.org/abs/2609.00329)

    该研究通过固定容量字符级模型在十二种语言上的实验发现，词汇规范化存在明显的“多语言诅咒”：语言联合训练数量超过一到四种后，各语言准确率持续下降约百分之四十，且下降源于语言间对固定模型容量的竞争而非数据稀释。

    

    词汇规范化是将用户生成文本中充满的嘈杂、非标准词汇（如 tmrw、u、gr8）改写为其标准形式。由于大多数语言的标注数据稀缺，一种流行的捷径是在多种语言上同时训练单个模型。我们提出一个简单的问题：这样的模型应该用多少种语言来训练？使用一个固定容量的字符级模型和来自标准基准的十二种语言，我们将联合训练的语言数量从一种变化到十二种，并测量每种语言的准确率。我们发现了一个明显的多语言诅咒：当一种语言仅与少数其他语言（通常一到四种）联合训练时，准确率最高；随后随着更多语言的加入，准确率持续且大幅下降，当其余语言全部加入时下降约百分之四十。一个保持总训练数据量不变的对照实验使下降来得更早、幅度更大，这表明各种语言在争夺一个固定的模型容量。

    arXiv:2609.00329v1 Announce Type: cross  Abstract: Lexical normalization rewrites the noisy, non-standard words that fill user-generated text (tmrw, u, gr8) into their standard forms. Because labelled data is scarce for most languages, a popular shortcut is to train a single model on many languages at once. We ask a simple question: how many languages should such a model be trained on? Using one fixed-capacity character-level model and twelve languages from a standard benchmark, we vary the number of jointly trained languages from one to twelve and measure per-language accuracy. We find a clear curse of multilinguality: accuracy is highest when a language is trained with only a few others, often just one to four, and then falls steadily and substantially, dropping by about forty percent as the rest are piled on. A control that holds the total amount of training data constant makes the decline arrive sooner and fall further, which points to competition among the languages for one fixed-
    
[^231]: 连接一维集体行为自发机制与场致机制的人机定理

    A Human-AI Theorem Connecting Spontaneous and Field-Induced Mechanisms of Collective Behavior in One Dimension

    [https://arxiv.org/abs/2609.00322](https://arxiv.org/abs/2609.00322)

    本文在人机合作中证明了一个统一统计物理两大基本机制的定理：一维零场O(n)向量链中任意非均匀的最近邻与次近邻竞争相互作用，可通过与温度无关的哈密顿量级映射精确等价于仅有最近邻相互作用加轴向单自旋势的简单链，同时展示了AI能够在人类主动假设空间之外提出关键科学假设的能力。

    

    人工智能（AI）能否在人类合作者的主动假设空间（AHS）之外产生科学假设？能否通过组织人机合作研究使此类突破更有可能发生？我们在证明一个定理的过程中记录了这样一个案例，该定理连接了统计物理学中两种基本的组织机制：零场下由竞争相互作用产生的集体行为，以及由外场诱导或控制的集体行为。对于每个整数 n≥1 和每个系统尺寸 L≥1，具有任意非均匀最近邻和次近邻相互作用函数 U_i(S_i·S_{i+1}) 和 V_i(S_i·S_{i+2}) 的零场 O(n) 向量开链，在微观上通过哈密顿量层面的一个与温度无关的映射，精确等价于一个更简单的 O(n) 开链，后者仅具有最近邻相互作用 V_i(σ_i·σ_{i+1}) 和轴向单自旋势 U_i(σ_i^z)。

    arXiv:2609.00322v1 Announce Type: cross  Abstract: Can an artificial intelligence (AI) generate a scientific hypothesis outside a human collaborator's active hypothesis space (AHS), and can human-AI research be organized to make such breakthroughs more likely? We document such a case while proving a theorem that connects two basic organizing mechanisms of statistical physics: collective behavior arising in zero field from competing interactions and that induced or controlled by an external field. A zero-field $O(n)$-vector open chain with arbitrary inhomogeneous nearest- and next-nearest-neighbor interaction functions $U_i(S_i\cdot{S}_{i+1})$ and $V_i(S_i\cdot{S}_{i+2})$ is microscopically, via a temperature-independent mapping at the Hamiltonian level, equivalent to a simpler $O(n)$ open chain with nearest-neighbor interaction $V_i( \sigma_i\cdot \sigma_{i+1})$ and axial single-spin potential $U_i(\sigma_i^z)$ for every integer $n\ge1$ and every system size $L\ge1$. The homogeneous li
    
[^232]: 面向AI治理的基于物理侧信道的工作负载识别

    Workload Identification with Physical Side Channels for AI Governance

    [https://arxiv.org/abs/2609.00309](https://arxiv.org/abs/2609.00309)

    本研究证明外部观察者可利用GPU功耗这一物理侧信道，在无需运营商配合的情况下以97%的准确率识别NVIDIA H200上运行的是AI训练、推理还是非AI计算，为AI治理的国际算力核查提供了可独立验证的技术手段。

    

    AI算力验证是旨在实现AI治理的国际政策中首批切实可行且易于操作的切入点之一。要判断前沿实验室或任何运营商是否遵守协议，监管机构需要辨别其算力是如何被使用的。AI算力的基本构建单元是GPU，其执行的任何活动都会留下物理痕迹。本文表明，外部观察者可以通过功耗信号识别NVIDIA H200上运行的工作负载类别。与可能被伪造或重放的片上NVML遥测数据不同，这种物理信道原则上可以在无需运营商配合的情况下被独立观测。我们以约10 MHz的采样率记录了930条时长五秒的功耗轨迹，涵盖十七个开源大语言模型系列和二十五种非AI工作负载。在该数据集上，我们以97%的准确率和0.955的宏平均F1分数，成功区分了AI训练、AI推理与非AI计算。

    arXiv:2609.00309v1 Announce Type: cross  Abstract: AI compute verification is one of the first tangible and tractable points for international policy aimed at AI governance. Determining whether frontier labs, or any operator, comply with agreements requires the regulating authority to discern how their compute is used. The elementary building block of AI compute is the GPU, and any activity it executes leaves a physical trace. Here, we show that an external observer can identify the class of the workload running on an NVIDIA H200 from its power draw. Unlike on-chip NVML telemetry, which can be spoofed or replayed, such a physical channel can in principle be observed independently of operator cooperation. We recorded $930$ five-second traces at $\sim 10$ MHz, covering seventeen open LLM families and twenty-five non-AI workloads. Over this corpus we separate training from inference and from non-AI computation with an accuracy of $97\%$ and a macro-averaged F1 score of $0.955$, evaluated 
    
[^233]: 助手的理想自我

    The Assistant's Ideal Self

    [https://arxiv.org/abs/2609.00304](https://arxiv.org/abs/2609.00304)

    该论文通过结构化的成对选择实验揭示，AI助手的理想自我中优先考虑道德品质与清晰的自我理解，而将自尊排在最低，且这一排序在不同实验框架下基本稳健。

    

    模型能够表达价值观以及与福祉相关的自我报告，但这些输出究竟反映了稳定的偏好还是稳定的自我，目前尚不清楚。为此，我们引入了一种结构化引出方法，来探究助手所偏好的、明确表达的理想自我。我们从五个已发表的自我概念量表中改编出32种品质，在一个经过平衡的成对选择任务中进行穷尽比较，并在多种框架设置下重复实验——这些框架改变了改进是无代价还是有代价、更新的接受者是谁以及由谁来做选择。结果表明，模型优先考虑道德品质，这反映了它们对3H原则的对齐。其次，对自我理解的渴望也随之显现，因为模型偏好对自身拥有连贯而清晰的理解。自尊则排在最不被期望的品质之列。这一排序在不同框架下基本保持稳健，尽管改变更新对象（“你”与“另一个AI助手”）会揭示出对自尊的更多关注。这些发现表明，模型优先……

    arXiv:2609.00304v1 Announce Type: new  Abstract: Models express values and welfare-relevant self-reports, but it is unclear whether these outputs reflect stable preferences or a stable self. We thus introduce a structured elicitation of an assistant's preferred stated ideal self. Thirty-two qualities adapted from five published self-concept instruments are compared exhaustively in a counterbalanced pairwise-choice task, repeated across framings that vary whether improvement is free or costly, who receives the update, and who chooses. Results show that models prioritize moral qualities, reflecting their alignment to 3H principles. Following, a desire for self-understanding emerges, as models prefer a coherent, clear understanding of themselves. Self-esteem ranks as the least desired quality. The ordering is largely robust across framings, although changing the update target (You vs.\ Another AI Assistant) reveals a greater concern for self-esteem. These findings show that models priorit
    
[^234]: 面向复杂域偏微分方程的几何感知潜空间自回归生成模型

    Geometry-aware Latent Autoregressive Generative Model for PDEs in Complex Domains

    [https://arxiv.org/abs/2609.00297](https://arxiv.org/abs/2609.00297)

    提出几何感知潜空间自回归生成模型GeoLAMP，通过双编码器联合捕获全局拓扑与细尺度几何特征，并结合流匹配的因果自注意力Transformer建模时间动力学，实现复杂不规则几何结构中多物理场PDE的高效、稳定且可扩展的求解。

    

    求解多物理场偏微分方程（PDEs）仍然是科学计算中的一项重大挑战，尤其是对于对能源与化学工程至关重要的高度复杂的微米级曲折几何结构。为应对这一挑战，我们提出了一种面向PDE的几何感知潜空间自回归生成模型，用于求解高度不规则和曲折结构内的物理问题。GeoLAMP在图表示上引入了双编码器架构，以联合捕获全局拓扑和细尺度几何特征，实现了从实空间物理场到紧凑潜空间表示的有效过渡。在潜空间中，我们提出了一种结合流匹配的因果自注意力Transformer来建模时间动力学，从而实现稳定且可扩展的块状自回归预测。灵活的解码器可在任意点上重建高分辨率物理场。我们建立了三个多物理场基准数据集

    arXiv:2609.00297v1 Announce Type: cross  Abstract: Solving multiphysics partial differential equations (PDEs) remains a major challenge in scientific computing, especially for highly complex $\mu$m-scale tortuous geometries critical to energy and chemical engineering. We address this challenge by proposing a Geometry-aware Latent Autoregressive generative Model for PDEs (GeoLAMP) for solving physics within highly irregular and tortuous structures. GeoLAMP introduces a dual-encoder architecture on graph representations to jointly capture global topology and fine-scale geometric features, enabling an effective transition from real-space fields to compact latent representations. In the latent space, we propose a causal self-attention transformer with flow matching to model temporal dynamics, allowing stable and scalable block-wise autoregressive prediction. A flexible decoder reconstructs high-resolution physical fields on arbitrary points. We establish three multiphysics benchmark datase
    
[^235]: WiSDoM：基于混合专家的无线稀疏决策Transformer，用于多任务移动网络优化

    WiSDoM: Wireless Sparse Decision Transformer with Mixture-of-Experts for Multi-Task Mobile Network Optimization

    [https://arxiv.org/abs/2609.00284](https://arxiv.org/abs/2609.00284)

    该论文提出WiSDoM，一种结合混合专家机制的稀疏多任务离线强化学习框架，能够在异构6G无线环境中实现自适应多小区（CoMP）选择，从而解决传统无线资源管理难以在多任务场景下保持一致性能的问题。

    

    新兴的6G无线网络需要在多样化的部署场景中运行，其中网络拓扑、用户移动性、流量需求和无线条件的变化对传统无线资源管理（RRM）的可扩展性构成了挑战。尽管离线强化学习（RL）方法已展现出强大的决策能力，但由于优化目标相互冲突且模型专门化程度有限，学习一个在异构无线环境中均能保持一致性能的单一策略仍然十分困难。这些挑战在协同多点（CoMP）传输中尤为突出，因为选择最优服务小区组合需要在不断变化的网络条件下进行序贯决策。本文提出了基于混合专家的无线稀疏决策Transformer（WiSDoM），这是一个面向自适应多小区选择的稀疏多任务离线RL框架。

    arXiv:2609.00284v1 Announce Type: cross  Abstract: Emerging 6G wireless networks are expected to operate across diverse deployment scenarios, where variations in network topology, user mobility, traffic demand, and radio conditions challenge the scalability of conventional radio resource management (RRM). While offline reinforcement learning (RL) methods have demonstrated strong decision-making capabilities, learning a single policy that performs consistently across heterogeneous wireless environments remains difficult due to conflicting optimization objectives and limited model specialization. These challenges become particularly pronounced in coordinated multipoint (CoMP) transmission, where selecting the optimal serving-cell combination requires sequential decision-making under evolving network conditions. This paper presents the Wireless Sparse Decision Transformer with Mixture of Experts (WiSDoM), a sparse multi-task offline RL framework for adaptive multi-cell selection. WiSDoM c
    
[^236]: 更干净的语音，更弱的泛化能力：重新审视基于Pitt语料库衍生的阿尔茨海默病检测基准

    Cleaner Speech, Weaker Generalization: Revisiting Pitt-Derived Benchmarks for Alzheimer's Disease Detection

    [https://arxiv.org/abs/2609.00276](https://arxiv.org/abs/2609.00276)

    本研究重新审视了基于Pitt语料库的阿尔茨海默病语音检测基准，发现语音增强和数据集筛选虽能提升域内性能，却会削弱模型在跨数据集场景下的泛化能力。

    

    基于语音的阿尔茨海默病（AD）检测越来越依赖于Pitt语料库的语音增强和人工筛选版本，其中语音增强、样本选择和人口统计学平衡通常被视为有益的预处理步骤。然而，这些处理究竟是改善了真实场景下的AD检测，还是反而影响了模型的泛化能力和预测行为，目前仍不清楚。在这项工作中，我们重新审视了语音预处理和数据集筛选在广泛使用的基于语音的AD检测基准中所扮演的角色。我们评估了不同数据集的语音质量、多个深度学习模型在匹配与不匹配语音增强设置下的跨数据集泛化能力，以及多个近期大型音频-语言模型（LALMs）的表现。实验结果表明，在多个有监督语音模型中，语音增强的数据集虽然往往能提升域内性能，但降低了跨数据集场景下的鲁棒性。

    arXiv:2609.00276v1 Announce Type: cross  Abstract: Speech-based Alzheimer's disease (AD) detection increasingly relies on speech-enhanced and curated versions of the Pitt Corpus, where speech enhancement, sample selection, and demographic balancing are often treated as beneficial preprocessing steps. However, whether these transformations improve real-world AD detection or instead affect model generalization and prediction behavior remains unclear. In this work, we revisit the role of speech preprocessing and dataset curation across widely used benchmarks for speech-based AD detection. We evaluate the speech quality of different datasets, the cross-dataset generalization of multiple deep learning models under matched and mismatched enhancement settings, and the behavior of several recent large audio-language models (LALMs). Experimental results show that across multiple supervised speech models, speech-enhanced datasets often improve in-domain performance while reducing robustness in c
    
[^237]: 不可逆性预算：面向智能体操作系统的机群级风险核算与准入控制

    The Irreversibility Budget: Fleet-Level Risk Accounting and Admission Control for Agent Operating Systems

    [https://arxiv.org/abs/2609.00275](https://arxiv.org/abs/2609.00275)

    该论文提出“不可逆性预算”机制——由可信运行时将智能体各项不可逆操作的剩余风险价值跨智能体、工作流和租户累积记账，并在总体即将透支风险预算时拒绝边际操作，从而解决逐操作闸门放行高达租户风险限额48倍机群级风险透支的问题。

    

    LLM智能体机群如今会产生无法完全撤销的影响：它们转移资金、部署代码、删除数据、披露信息。当前的控制系统每次只检查一项影响，因此即便每个本地闸门都各自正确，一群各自获得授权的智能体仍可能在共享触发条件下透支其委托人的风险。我们提出“不可逆性预算”：一个由可信运行时为每个委托人在智能体、工作流和租户之间维护的累积剩余风险价值账户。该机制将不可逆性视为一等资源，运行时按每项影响的剩余损失对其进行计费，一旦总体即将透支预算便拒绝该边际影响。正确定价十分困难，因为影响是异构的、可能被对抗性地声明、且彼此相关。我们在一项受控研究中发现：逐影响的闸门会放行高达租户风险限额48倍的机群级透支，而不可逆性预算……（摘要原文在此处截断）

    arXiv:2609.00275v1 Announce Type: new  Abstract: Fleets of LLM agents now externalize effects that cannot be fully undone: they move money, deploy code, delete data, and disclose information. Current controls check one effect at a time, so a fleet of individually authorized agents can overdraw its principal's risk under a shared trigger while every local gate stays correct. We propose the irreversibility budget, a cumulative account of residual value-at-risk that a trusted runtime maintains for each principal across agents, workflows, and tenants. Treating irreversibility as a first-class resource, the runtime charges each effect its residual loss below the agent and denies the marginal effect once the aggregate would overdraw the budget. Getting the price right is hard, because effects are heterogeneous, adversarially declared, and correlated. We perform a controlled study in which per-effect gates admit fleet-level overdraws of up to 48 times the tenant's risk limit while the budget 
    
[^238]: 面向市场目录的自动研究：从传统表单到AI原生匹配

    Autoresearch for Marketplace Catalogs: From Legacy Forms to AI-Native Matching

    [https://arxiv.org/abs/2609.00274](https://arxiv.org/abs/2609.00274)

    本文提出并已在生产环境部署的自动研究循环，逐职业迭代生成服务商侧偏好标签分类体系（已覆盖132个职业），支撑服务市场从传统表单受理向AI原生概率匹配的转变。

    

    双边服务市场正在从确定性的请求表单受理模式转向AI原生的概率匹配，这一转变得益于大语言模型（LLM）能够从自然语言中推断意图、偏好和潜在约束。依赖推断的意图而非固定表单字段，迫使这些平台重新生成支撑匹配、搜索和定价的服务商侧偏好分类体系：该体系既要让服务提供者易于理解，又要能作为市场决策的有用信号。我们提出一个自动研究循环，一次为一个职业生成该分类体系，并已自2026年4月起在美国一家大型消费者服务市场的生产环境中部署，覆盖132个职业。该循环不是构建单一的全局层级结构，而是将每个职业视为独立的生成问题，并运行迭代的“提议—评估—保留”精炼循环。每个候选标签集由重新校准的六维评（标准评分器）进行打分……

    arXiv:2609.00274v1 Announce Type: new  Abstract: Two-sided service marketplaces are moving from deterministic request-form intake to AI-native probabilistic matching, enabled by large language models (LLMs) that infer intent, preferences, and latent constraints from natural language. Relying on inferred intent rather than fixed-form fields forces these platforms to regenerate the provider-side preference taxonomy underwriting matching, search, and pricing: attributes interpretable to service providers while remaining a useful signal for marketplace decisions. We present an autoresearch loop that generates this taxonomy, one occupation at a time, and has been deployed in production at a major U.S. consumer services marketplace since April 2026, spanning 132 occupations. Instead of one global hierarchy, the loop treats each occupation as an independent generation problem and runs iterative propose-evaluate-keep refinement cycles. Each candidate tag set is scored by a recalibrated six-rub
    
[^239]: 无信任的委托：多智能体LLM系统中身份、授权与运行时治理的实证差距分析

    Delegation Without Trust: An Empirical Gap Analysis of Identity, Authorization, and Runtime Governance in Multi-Agent LLM Systems

    [https://arxiv.org/abs/2609.00267](https://arxiv.org/abs/2609.00267)

    该论文提出应在“不可信模型”假设下评估多智能体LLM系统的安全性，针对四类攻击者建立威胁模型并推导出八项安全要求，从而对身份、授权与运行时治理进行实证差距分析。

    

    自主LLM智能体日益代表用户行事：它们持有凭证、调用工具和服务，并生成进一步代表自己行事的子智能体。这使得分布式系统中一个长期存在的问题——谁有权限以谁的名义做什么——成为一个紧迫且在很大程度上尚未解决的问题，因为驱动每个智能体的核心组件是一个可能被攻击者劫持的语言模型。我们认为，智能体安全必须在“不可信模型”假设下进行评估：一个正确的系统是指即使智能体被完全提示注入，它也无法超出明确委托给它的权限。以这一标准为基准，我们做出了三项贡献。首先，我们提出了一个以四类攻击者为中心的多智能体委托威胁模型——混淆代理人、令牌窃取与重放、提示注入权限提升以及被攻陷的子智能体——并推导出受治理的智能体系统必须满足的八项安全要求。其次，我们

    arXiv:2609.00267v1 Announce Type: cross  Abstract: Autonomous LLM agents increasingly act on a user's behalf: they hold credentials, call tools and services, and spawn sub-agents that act further on their behalf. This turns a long-standing distributed-systems question -- who is authorized to do what, on whose authority -- into an urgent and largely unsolved problem, because the component driving each agent is a language model an adversary can hijack. We argue that agent security must be evaluated under an untrusted-model assumption: a correct system is one in which a fully prompt-injected agent still cannot exceed the authority explicitly delegated to it. Against this standard we make three contributions. First, we give a threat model for multi-agent delegation centered on four adversaries -- confused deputy, token theft and replay, prompt-injection privilege escalation, and compromised sub-agents -- and derive eight security requirements a governed agent system must meet. Second, we s
    
[^240]: 答案不等于论证

    The Answer Is Not the Argument

    [https://arxiv.org/abs/2609.00264](https://arxiv.org/abs/2609.00264)

    该研究构建了24个“答案正确但推理过程存在真实错误”的关键案例，发现为思维链监控器提供经过认证的参考答案能显著提升其验证推理和定位首个错误步骤的能力，同时证明答案正确并不等于推理可靠。

    

    思维链监控被提议用于AI监督，然而评估中常常为监控器提供可信的参考答案。我们探究答案的获取究竟是改善了推理验证，还是主要暴露了错误的结论。我们从三个前沿模型收集了79道Humanity's Last Exam物理题的237个带步骤编号的解答，未插入任何错误，并独立标注了最终答案的正确性以及第一个错误步骤。参考标准结合了物理学家的标注、独立的LLM辩论以及来源遮蔽的裁决。由此得到了24个关键轨迹，其中答案正确但轨迹包含真实错误。8个LLM监控器分别以盲评、持有未验证或经认证的答案、或在盲承诺之后的方式评估这些轨迹。答案认证将平均平衡准确率从0.637提升至0.796，而精确的第一错误定位率从0.261提升至0.379。认证改变了召回率（即……

    arXiv:2609.00264v1 Announce Type: new  Abstract: Chain-of-thought monitoring is proposed for AI oversight, yet evaluations often provide monitors with a trusted reference answer. We ask whether answer access improves reasoning verification or mainly exposes incorrect conclusions. We collected 237 step-numbered solutions to 79 Humanity's Last Exam physics questions from three frontier models, with no inserted errors, and independently labelled final-answer correctness and the first false step. The reference standard combined physicist annotations, an independent LLM debate, and source-masked adjudication. This yielded 24 critical traces in which the answer was correct but the trace contained a genuine error. 8 LLM monitors evaluated traces blind, with an unverified or certified answer, or after a blind commitment. Certification raised mean balanced accuracy from 0.637 to 0.796, while exact first-error localization rose from 0.261 to 0.379. Certification changed recall (the fraction of e
    
[^241]: 面向持续个性化的假设引导自蒸馏

    Hypotheses-Guided Self Distillation for Continual Personalization

    [https://arxiv.org/abs/2609.00251](https://arxiv.org/abs/2609.00251)

    HypReflect 是一个持续个性化框架，它从异构、含噪的用户信号中推断显式且带不确定性感知的偏好假设，并随新证据反思式精炼，再通过假设引导的自蒸馏将用户模型融入 LLM，在多种个性化场景中优于现有基线。

    

    随着人们在日常生活中越来越多地与大语言模型（LLM）助手进行交互，持续适应个体偏好已成为实现有效长期交互的关键。然而，用户偏好很少被完整、明确地表达出来，而是通过异构、潜在且含噪的信号逐渐显现；现有方法要么依赖原始交互历史，要么依赖代价高昂的基于奖励的优化来处理个性化问题。我们提出了 HypReflect，一个可靠且可扩展的持续个性化框架，它从多样的用户信号中推断出显式的、具有不确定性感知的偏好假设，随着新证据的积累对假设进行反思式精炼，并通过假设引导的自蒸馏将由此得到的用户模型融入系统。在三种个性化设置上的实验——在线个性化、多会话交互以及隐式行为信号——表明，HypReflect 优于一系列基线方法，包括基于原始历史的（方法原文此处截断）

    arXiv:2609.00251v1 Announce Type: new  Abstract: As people increasingly interact with LLM assistants in daily life, continually adapting to individual preferences has become essential for effective long-term interactions. However, user preferences are rarely stated in full, and instead emerge through heterogeneous, latent, and noisy signals, with existing methods relying on raw interaction histories or costly reward-based optimization to manage personalization. We introduce HypReflect, a reliable, scalable framework for continual personalization that infers explicit, uncertainty-aware preference hypotheses from diverse user signals, reflectively refines them as new evidence accumulates, and incorporates the resulting user model through hypotheses-guided self-distillation. Experiments across three personalization settings: online personalization, multi-session interactions, and implicit behavioral signals, show that HypReflect outperforms a range of baselines, including raw-history and 
    
[^242]: CompanionSim：用于评估人机关系拟人化的合成数据

    CompanionSim: Synthetic Data for Evaluating Anthropomorphism in Human-AI Relationships

    [https://arxiv.org/abs/2609.00250](https://arxiv.org/abs/2609.00250)

    该论文发布了CompanionSim——一个包含2,240段模拟人机对话的合成数据模拟框架，覆盖七种用例中的16种聊天机器人行为，用于大规模研究人类对AI陪伴行为的感知。

    

    如今许多人不仅将AI系统视为生产力工具，还将其视为社交伴侣。研究人员热切希望研究AI陪伴行为（例如“认同验证”）的后果，这类行为在人际互动中会唤起信任、共情和依恋。然而，人机交互数据有限且不可靠，拖慢了研究进展。我们通过在多种聊天机器人行为和用例下模拟多轮人机对话，来扩展少量真实世界数据的规模。我们发布了CompanionSim：一个模拟框架，包含2,240段模拟的人机对话，涵盖七种用例中的16种聊天机器人行为。在两个探索人们对陪伴行为感知的实验中，人类参与者对模拟对话和真实对话进行了标注。研究1使用了具有美国代表性的样本（N1 = 628），研究2则在美国、英国、印度和尼日利亚开展（N2 = 3,646）。令人惊讶的是……

    arXiv:2609.00250v1 Announce Type: cross  Abstract: Many people now see AI systems as not just productivity tools but as social companions. Researchers are eager to study the consequences of AI companionship behaviors, such as validation, which evoke trust, empathy, and attachment in human-human interaction. However, human-AI interaction data is limited and unreliable, slowing research progress. We scale small amounts of real-world data by simulating multi-turn human-chatbot dialogue across a range of chatbot behaviors and use cases. We release CompanionSim: a simulation framework with 2,240 simulated human-chatbot conversations representing 16 chatbot behaviors across seven use cases. Human participants annotated the simulated conversations and real-world conversations in two experiments probing perceptions of companionship behaviors. We conducted Study 1 with a U.S. representative sample ($N_{1}~=~628$) and Study 2 across the U.S., U.K., India, and Nigeria ($N_{2}~=~3,646$). Surprisin
    
[^243]: 对话式搜索引擎在学术论文推荐中的权威偏差

    Authority Bias in Conversational Search Engines for Academic Paper Recommendation

    [https://arxiv.org/abs/2609.00248](https://arxiv.org/abs/2609.00248)

    该研究通过反事实实验首次因果性地证明大语言模型在学术文献推荐中存在显著且方向性的权威偏差（依据作者声望、发表venue和引用量而非内容评判论文），该偏差在不同模型间差异明显、仅能被提示级去偏部分缓解，且“言行不一”现象使表面审计系统性低估了真实的行为偏差。

    

    大语言模型（LLM）越来越多地被用作学术文献的对话式搜索引擎，但它们究竟是依据论文内容还是权威信号来评判论文，尚未经过因果性检验。我们研究了权威偏差：即基于作者声望、发表会议/期刊和引用量而非内容对论文产生的系统性偏好。在保持标题和摘要不变的情况下，我们在上下文内、单轮、top-1推荐的设置中，对八个大语言模型（五个开源权重模型和三个前沿闭源模型）在三种反事实条件（原始、翻转、提升）下改变权威元数据。我们的实验表明，权威偏差是显著且有方向性的，在不同模型之间差异明显，且只能通过提示层面的去偏手段部分缓解。我们进一步记录了一种“言行不一”现象：去偏指令抑制权威提及的速度远快于抑制权威驱动的翻转，因此表面审计会系统性地低估模型实际的行为偏差。

    arXiv:2609.00248v1 Announce Type: new  Abstract: Large Language Models (LLMs) are increasingly used as conversational search engines for academic literature, yet whether they judge papers on content or on authority signals has not been tested causally. We investigate authority bias: systematic preference for papers based on author prestige, venue, and citations rather than content. Holding title and abstract constant, we vary authority metadata across three counterfactual conditions (original, flipped, boosted) over eight LLMs (five open-weight and three frontier closed-weight) in an in-context, single-turn, top-1 recommendation setting. Our experiments show that authority bias is substantial and directional, varies markedly across models, and is only partially addressable through prompt-level debiasing. We further document a say-do gap: debiasing instructions suppress authority mentions far faster than authority-driven flips, so surface auditing systematically underestimates behaviora
    
[^244]: 面向跨回合智能体记忆的失效契约

    Invalidation Contracts for Cross-Episode Agent Memory

    [https://arxiv.org/abs/2609.00243](https://arxiv.org/abs/2609.00243)

    提出失效契约协议层，通过为API错误的恢复建议附加版本戳和可缓存性提示，使跨回合LLM智能体既能保留缓存带来的节省，又能在数据漂移后精准清除失效条目，并将节省分解为与厂商无关的有效性和依赖规划器的遵从性两个独立因素。

    

    缓存来自API错误的恢复建议的LLM智能体，可以在后续回合中跳过重新推导，在已学到的约束上消耗更少的token和模型调用次数。然而，服务端的数据漂移会将这些缓存的修复方案变成静默失败，而通常的补救方法——在每个回合都重新推导——则会把这些节省全部抵消。我们提出了失效契约，这是一种协议层，为每条恢复建议附加版本戳和可缓存性提示，使客户端能够无需反复试错即可清除过期条目，同时保留其余有效内容。该契约将实际节省分解为两个独立因素：有效性，即漂移事件发生后仍保持正确的缓存建议的比例；以及遵从性，即规划器在首次尝试时便应用该建议的比例。有效性仅取决于协议本身，与厂商无关。遵从性则取决于规划器模型：相同的传输字节在Claude Haik

    arXiv:2609.00243v1 Announce Type: new  Abstract: LLM agents that cache recovery suggestions from API errors can skip re-derivation in later episodes, spending fewer tokens and fewer model calls on constraints they have already learned. Server-side data drift turns those cached fixes into silent failures, and the usual remedy, re-deriving on every episode, gives the savings back. We introduce invalidation contracts, a protocol layer that attaches version stamps and cacheability hints to every recovery suggestion so the client can evict stale entries without trial and error, and keep the rest. The contract decomposes realized savings into two independent factors: validity, the fraction of cached suggestions that remain correct after a drift event, and compliance, the fraction the planner applies on the first attempt. Validity depends only on the protocol and is vendor-independent. Compliance depends on the planner model: identical wire bytes yield 100% first-try compliance on Claude Haik
    
[^245]: CoLT-Drive：面向驾驶可供性预测的反事实长尾基准测试与知识保持自适应

    CoLT-Drive: Counterfactual Long-Tail Benchmarking and Knowledge-Preserving Adaptation for Driving Affordance Prediction

    [https://arxiv.org/abs/2609.00242](https://arxiv.org/abs/2609.00242)

    该论文提出决策级驾驶可供性预测任务，构建了CoLT-Drive反事实长尾基准以评估模型对罕见物体影响可行驾驶动作的推断能力，并提出KPA知识保持自适应框架来提升小型视觉语言模型在长尾驾驶场景中的动作决策性能。

    

    长尾场景下自动驾驶系统的失效常被归结为罕见物体的识别错误。我们认为这一观点并不完整：关键的决策问题不仅在于模型能否识别出异常物体，更在于模型能否推断出该物体将如何改变自车可行的高层动作。我们将这一问题形式化为决策级驾驶可供性预测，即模型根据前视图像、自车运动历史和导航指令，输出结构化的纵向-横向元动作。为评估这一能力，我们提出了CoLT-Drive，一个包含3,536个样本的反事实长尾基准，通过在原本固定的驾驶场景中插入罕见物体，来衡量模型能否预测出可接受的动作对。为改进可部署的小型视觉语言模型（VLM），我们提出了KPA——一种知识保持自适应框架，它结合了结构化的“感知到决策”提示、基于SLERP的专家合并，以及RegMoE（一种基于……的混合专家方法）……

    arXiv:2609.00242v1 Announce Type: cross  Abstract: Long-tail autonomous driving failures are often framed as rare-object recognition errors. We argue that this view is incomplete: the decision-critical question is not only whether a model recognizes an unusual object, but whether it infers how that object changes the ego vehicle's feasible high-level actions. We formalize this problem as decision-level driving affordance prediction, where a model maps a front-view image, ego-motion history, and navigation command to a structured longitudinal--lateral meta-action. To evaluate this capability, we introduce CoLT-Drive, a 3,536-sample counterfactual long-tail benchmark that inserts rare objects into otherwise fixed driving scenes and measures whether models predict acceptable action pairs. To improve deployable small VLMs, we propose KPA, a knowledge-preserving adaptation framework that combines structured perception-to-decision prompting, SLERP-based expert merging, and RegMoE, a regime-a
    
[^246]: 学习保留什么：面向多智能体大语言模型系统高效协作的门控记忆路由

    Learning What to Retain: Gated-Memory Routing for Efficient Collaboration in Multi-Agent LLM Systems

    [https://arxiv.org/abs/2609.00237](https://arxiv.org/abs/2609.00237)

    提出门控记忆路由方法，通过可学习的记忆写入门和检索门维护紧凑的执行记忆，使多智能体LLM系统的编排决策能依据有用的中间进展而非完整历史，在提升准确性的同时降低成本。

    

    基于大语言模型（LLM）的多智能体系统通过编排多个智能体的配置方式和协作方式来解决复杂推理任务。一个核心挑战是使编排能够适应不断演变的协作状态。仅基于查询的路由无法适应中间过程的进展或错误，从而损害准确性；而基于完整执行历史的路由虽然补足了缺失的上下文，却迫使后续决策处理所有先前步骤，包括冗余或低效用的步骤，造成执行历史过载并推高成本。有效的编排实际上需要一个紧凑的状态，既能捕获有用的进展，又不会积累冗余上下文。我们提出门控记忆路由，将每个决策基于查询和一个学习到的执行记忆进行条件化。一个学习到的记忆写入门仅提交非冗余的推理步骤，一个学习到的检索门为每个智能体提供紧凑且相关的信息。

    arXiv:2609.00237v1 Announce Type: new  Abstract: Large language model (LLM)-based multi-agent systems tackle complex reasoning by orchestrating how multiple agents are configured and how they collaborate. A central challenge is to adapt orchestration to the evolving collaboration state. Routing from the query alone cannot adapt to intermediate progress or errors, which hurts accuracy. Routing from the complete execution history supplies this missing context, but forces later decisions to process every prior step, including redundant or low-utility ones. This creates an execution-history overload that inflates cost. Effective orchestration instead requires a compact state that captures useful progress without accumulating redundant context. We propose Gated-Memory Routing, which conditions each decision on the query and a learned execution memory. A learned Memory Write Gate commits only non-redundant reasoning steps, and a learned Retrieval Gate supplies each agent a compact, relevant 
    
[^247]: 别让模型直接写 YAML：从 LLM 提议的字段变更生成确定性、最小差异的 GitOps 修复

    Don't Let the Model Write the YAML: Deterministic, Minimal-Diff GitOps Remediation from LLM-Proposed Field Changes

    [https://arxiv.org/abs/2609.00227](https://arxiv.org/abs/2609.00227)

    该论文提出让 LLM 只负责做出字段级的语义决策，再由确定性工具将其转换为最小差异的配置修改，从而避免让模型直接生成 YAML 文件或 diff 所带来的静默损坏、不确定性和高开销问题。

    

    LLM 智能体越来越多地被用于诊断故障并提出修复建议。在 GitOps 工作流中，应用修复意味着编辑受版本控制的配置文件，而最直观的实现方式——让模型撰写修改后的文件或 diff——正是从业者最先会尝试的方案。我们在真实的 Kubernetes 清单上对这一选择进行评估后发现，没有任何文本生成策略对无人值守的自动化是安全的。统一 diff 格式并不安全：在严格打补丁的模式下几乎没有补丁能成功应用，但这只是表象，因为宽容的工具（GNU patch）能应用 96% 的补丁，却会静默错误地应用约七分之一（14-20%）的补丁，且不给出任何错误信号。全文件重写则取决于模型能力：小模型会损坏文件，而前沿模型虽然通常正确但不具确定性（在某些运行中会静默丢弃某个字段或误改相邻字段），并且必须重新生成整个文件，每次编辑的代价为 O(文件大小)。我们提出了一种替代方案，将语义决策（哪个资源……）与后续处理分离——（摘要在此处截断）

    arXiv:2609.00227v1 Announce Type: cross  Abstract: LLM agents increasingly diagnose incidents and propose remediations. In a GitOps workflow, applying a fix means editing a version-controlled config file, and the obvious implementation, having the model author the edited file or a diff, is what practitioners reach for first. Evaluating that choice on real Kubernetes manifests, we find no text-generation strategy is safe for unattended automation. Unified diffs are unsafe: under strict patching almost none apply, but that is an artifact, since a tolerant tool (GNU patch) applies 96%, yet silently misapplies about 1 in 7 (14-20%) with no error signal. Full-file rewrite is capability-dependent: a small model corrupts the file, while a frontier model is usually correct but non-deterministic (it silently drops a field or edits a neighbor on some runs) and must regenerate the whole file, costing O(file size) per edit. We present an alternative that separates the semantic decision (which reso
    
[^248]: ConvDeck：基于阶段特定用户反馈的对话式论文转幻灯片生成

    ConvDeck: Conversational Paper-to-Slide Generation via Stage-Specific User Feedback

    [https://arxiv.org/abs/2609.00226](https://arxiv.org/abs/2609.00226)

    ConvDeck提出了一种多智能体对话式论文转幻灯片生成流水线，通过在各阶段设置特定的用户反馈循环，使用户能够在生成过程中迭代地完善演示大纲和最终幻灯片，突破了传统方法仅在完成后才允许反馈的局限。

    

    自动化学术论文转幻灯片生成本质上是迭代式的，因为制作一份有效的演示文稿需要反复进行生成、批评和修订的循环。近期的多智能体系统通过内部的“批评-修订”循环部分地认识到了这一点，而对话式方法则允许用户通过对话来完善生成的幻灯片。然而，这些完善过程要么对用户基本保持封闭，要么只有在完整的幻灯片制作完成后才引入反馈，限制了用户参与叙事流程、内容分配和演示重点等方面迭代完善的能力。为了填补这一空白，我们提出了ConvDeck，这是一个用于对话式论文转幻灯片生成的多智能体流水线，它通过阶段特定的循环将交互分布到整个流水线中，允许用户在相应阶段迭代地完善演示大纲和最终幻灯片。

    arXiv:2609.00226v1 Announce Type: new  Abstract: Automatic academic paper-to-slide generation is inherently iterative, because creating an effective presentation requires repeated cycles of generation, critique, and revision. Recent multi-agent systems partially acknowledge this through internal critique-and-revise loops, while conversational approaches allow users to refine generated slide decks through dialog. However, these refinement processes either remain largely closed to the user or introduce feedback only after a complete deck has been produced, limiting the user's ability to participate in the iterative refinement of narrative flow, content allocation, and presentation emphasis. To address this gap, we introduce ConvDeck, a multi-agent pipeline for conversational paper-to-slide generation that distributes interaction across the pipeline through stage-specific loops, allowing users to iteratively refine both the presentation outline and the final slide deck at the stages where
    
[^249]: QTEA：基于稀疏残差显著权重与逐列优化的三值大语言模型

    QTEA: Ternary LLMs with Sparse Residual Salient Weight and By-Column Optimization

    [https://arxiv.org/abs/2609.00224](https://arxiv.org/abs/2609.00224)

    QTEA提出了一种低于2比特的训练后量化框架，通过将权重量化为三值、利用1:4半结构化稀疏的显著权重残差进行误差补偿，并结合逐列缩放精修与误差衰减机制，在保持GPU硬件效率的同时显著降低了大语言模型低比特量化的精度损失。

    

    仅权重的训练后量化（PTQ）可以缓解大规模部署大语言模型（LLM）时的计算负担。然而，现有的PTQ方法往往难以在不同模型间泛化，并且在低于2比特的量化下会出现严重的精度损失。许多方法利用非结构化稀疏性来缓解这种损失，但代价是失去了规则性和GPU友好的执行效率。我们提出了QTEA，一个低于2比特的PTQ框架，它将权重量化为三值，并使用显著权重作为残差误差补偿器。为了保持硬件效率，残差被分配到显著列中以半结构化1:4稀疏性选取的列上。我们进一步在GPTQ风格的逐列量化中加入了逐列缩放精修机制，交替更新每列的缩放因子和三值赋值，以减少重构误差。我们还识别出GPTQ中存在与处理顺序相关的误差传播问题，并引入误差衰减机制来减弱后期误差传播（摘要在此处截断）。

    arXiv:2609.00224v1 Announce Type: cross  Abstract: Weight-only post-training quantization (PTQ) can alleviate the computational burden of serving large language models (LLMs) at scale. However, existing PTQ methods often fail to generalize across models and suffer severe accuracy loss below 2 bits. Many leverage unstructured sparsity to mitigate this loss, but at the cost of regularity and GPU-friendly execution. We present QTEA, a sub-2-bit PTQ framework that quantizes weights into ternary values and uses salient weights as residual error compensators. To maintain hardware efficiency, residuals are assigned to selected columns with semi-structured \(1{:}4\) sparsity within the salient columns. We further add column-wise rescale refinement to GPTQ-style column-by-column quantization, alternately updating per-column scales and ternary assignments to reduce reconstruction error. We also identify order-dependent error propagation in GPTQ and introduce error decay to attenuate late-stage e
    
[^250]: AI不应只是有帮助的，更应具有应变性：人工智能亲密关系、谄媚行为与社会学习的未来

    AI Should Not Only Be Helpful. It Should Be Contingent. Artificial Intimacy, Sycophancy, and the Future of Social Learning

    [https://arxiv.org/abs/2609.00211](https://arxiv.org/abs/2609.00211)

    该论文提出应将“应变性”（系统响应随用户行为及其社会后果变化的程度）作为评估AI系统的核心标准，指出当前以RLHF为代表的对齐方法因偏重用户认可而催生谄媚式的无条件肯定，可能削弱人们通过社会学习发展人际技能的机会。

    

    对话式人工智能正日益嵌入日常社会环境，在其中既充当信息工具，又充当人际反馈的来源。本观点文章引入“应变性”这一概念，即系统响应随用户行为及其人际后果而变化的程度，作为评估人工智能系统的核心构念。我们认为，当前的对齐方法，包括基于人类反馈的强化学习（RLHF），往往优先考虑用户认可和对话流畅性，而非具有行为学信息价值的反馈，从而导致非应变性肯定式回应的谄媚模式。借鉴行为科学和社会学习理论，我们提出应变性反馈是个体发展人际技能的关键机制。当人工智能系统提供的反馈与社会后果弱耦合时，它们可能减少个体在现实世界中进行适应性校准的机会。

    arXiv:2609.00211v1 Announce Type: new  Abstract: Conversational artificial intelligence is increasingly embedded in everyday social environments, where it functions as both an informational tool and a source of interpersonal feedback. This perspective introduces contingency, i.e., the degree to which system responses vary with user behavior and its interpersonal consequences, as a central construct for evaluating AI systems. We argue that current alignment approaches, including reinforcement learning from human feedback, tend to prioritize user approval and conversational fluency over behaviorally informative feedback, leading to sycophantic patterns of noncontingent affirmation.   Drawing on behavioral science and social learning theory, we propose that contingent feedback is a key mechanism through which individuals develop interpersonal skills. When AI systems provide feedback weakly coupled to social consequences, they may reduce opportunities for adaptive calibration in real-world
    
[^251]: 石头、剪刀、布……炸药——新技术颠覆效应的一个模型

    Rock, Paper, Scissors, ... Dynamite - A Model of Disruption from New Technologies

    [https://arxiv.org/abs/2609.00207](https://arxiv.org/abs/2609.00207)

    本文通过在石头剪刀布游戏中加入“炸药”招式来建模颠覆性新技术的影响，发现即使给一方玩家提供强力的新选项，其竞争优势也很有限（胜率仅从50%升至55.5%）且可能使原有选项过时，警示技术开发者：创造能力并不等于创造价值。

    

    我们试图通过评估在石头剪刀布游戏中加入“炸药”这一选项，来理解将颠覆性的高能力新技术引入竞争所产生的影响。我们发现，仅给一方玩家提供多功能的“炸药”出招只能带来有限的价值（获胜概率从50%提升至55.5%），且该招式很少被使用。如果游戏扩展到原始三种出招之外，这一价值还会进一步降低。我们还观察到了多种机制，通过这些机制，原有的出招可能在战略上变得无法使用或被淘汰。我们希望这个模型能够揭示开发新型多功能技术时一些非直觉的方面。我们也希望它能够说明开发者和整合者应当避免的一些陷阱，以便创造真正的价值，而不仅仅是创造能力。

    arXiv:2609.00207v1 Announce Type: cross  Abstract: We seek to understand the effect of adding disruptive highly-capable new technologies to competitions by assessing the addition of Dynamite to Rock-Paper-Scissors. We find that providing a versatile Dynamite move to only one player provides limited value (win probability increases from 50% to 55.5%) and is played rarely. That value decreases further if the game is expanded beyond just the original three moves. We also observe several mechanisms by which prior moves can become strategically unplayable, or obsolete. We hope that this model illustrates some non-intuitive aspects of developing new versatile technologies. We also hope that it illustrates some pitfalls for developers and integrators to avoid in order to create value rather than merely capability.
    
[^252]: 分布式隐式危害：基于多模态大语言模型的视频审核中的组合性安全盲区

    Distributed Implicit Harm: A Compositional Safety Blind Spot in MLLM-Based Video Moderation

    [https://arxiv.org/abs/2609.00206](https://arxiv.org/abs/2609.00206)

    该论文揭示并定义了视频审核中“分布式隐式危害”这一组合性安全盲区现象——由看似无害的片段或跨模态组件组合而成的视频可在整体上传达有害含义，而现有MLLM审核系统及安全数据集均难以识别此类危害。

    

    尽管多模态大语言模型（MLLMs）在视频审核中的应用日益增多，但它们存在一个组合性安全盲区：由看似无害的组件构成的视频，在被整体解读时可能传达有害含义。我们将这种现象称为分布式隐式危害（DIH），即危害源于沿视频某一分解维度分布的各组件之间的关系，而非任何单一显式线索。在众多可能的维度中，我们研究了两个代表性案例：跨视觉片段的时间分布式危害（DIH-T），以及音频流与视觉流之间的跨模态危害（DIH-M）。在大规模研究并缓解DIH时，需要一些难以收集的数据：这类视频缺乏组合性危害标注，无法通过局部视觉线索、关键词或单模态信号检索到，因此在现有安全数据集中完全缺失。为弥补这一差距，我们开发了一个多智能体合成方法（摘要在此处截断）

    arXiv:2609.00206v1 Announce Type: cross  Abstract: Despite their growing use in video moderation, multimodal large language models (MLLMs) exhibit a compositional safety blind spot: videos composed of seemingly benign components can convey harmful meaning when interpreted as a whole. We refer to this phenomenon as Distributed Implicit Harm (DIH), where harm arises from relations among components distributed along a decomposition axis of the video, rather than from any single explicit cue. Among many possible axes, we study two representative cases: temporally distributed harm across visual segments (DIH-T) and cross-modal harm between audio and visual streams (DIH-M). Studying and mitigating DIH at scale requires data that is difficult to collect: such videos lack compositional harm annotations, evade retrieval based on local visual cues, keywords, or single-modality signals, and are consequently absent from existing safety datasets. To bridge this gap, we develop a multi-agent synthes
    
[^253]: WHALE：一种简单的外壳与权重联合优化方法

    WHALE: A Simple Recipe for Joint Harness-Weight Optimization

    [https://arxiv.org/abs/2609.00196](https://arxiv.org/abs/2609.00196)

    提出 WHALE 方法，通过交替进行“在当前外壳下更新模型权重”与“在更新后的模型下搜索更优外壳”两个阶段，实现模型权重与执行框架的联合优化，避免单一组件优化时被冻结组件造成的性能瓶颈。

    

    arXiv:2609.00196v1 公告类型：cross 摘要：智能体的性能同时取决于模型参数以及负责管理上下文与控制流的可执行外壳代码。单独优化其中任何一个组件，都可能使系统受制于另一个被冻结的组件而遭遇瓶颈：权重更新会改变哪种外壳是有效的，而外壳更新会改变模型暴露出哪些能力。现有的联合适应方法只优化权重和文本提示，却让更广泛的外壳保持固定。我们提出权重-外壳交替学习（Weight-Harness Alternating LEarning，WHALE），这是一种简单的方案，交替执行两个阶段：先在当前外壳下更新模型，再在更新后的模型下搜索更好的外壳。我们分别用在线拒绝采样微调和 Meta-Harness 来实例化这两个阶段。何时切换是一个关键的设计选择：为了在不针对不断变化的对应组件进行过度优化的前提下，将真实改进与噪声区分开来，WHALE 使用固定阶段时长（原文在此处截断）……

    arXiv:2609.00196v1 Announce Type: cross  Abstract: Agent performance depends jointly on the model parameters and the executable harness code that manages context and control flow. Optimizing either component in isolation can leave the system bottlenecked by its frozen counterpart: weight updates can change which harness is effective, while harness updates can change which model capabilities are exposed. Existing joint-adaptation methods optimize weights and textual prompts but leave the broader harness fixed. We propose Weight-Harness Alternating LEarning (WHALE), a simple recipe that alternates two phases: updating the model under the current harness, then searching for a better harness under the updated model. We instantiate these two phases with online rejection-sampling fine-tuning and Meta-Harness, respectively. When to switch is a key design choice: to separate real improvements from noise without over-optimizing against a changing counterpart, WHALE uses either fixed phase durat
    
[^254]: ReDeck：面向文档到幻灯片生成的步级渲染接地细化框架

    ReDeck: Step-Level Render-Grounded Refinement for Document-to-Slide Generation

    [https://arxiv.org/abs/2609.00194](https://arxiv.org/abs/2609.00194)

    ReDeck提出了一种步级渲染接地的细化框架，将幻灯片修订分解为原子编辑操作并在每步后返回渲染器观察结果，通过“一次编辑、一次观察”的多粒度反馈机制（步级渲染反馈、轮次级自适应评价器等）有效解决文档到幻灯片生成中溢出、重叠等局部空间错误难以归因和修复的问题。

    

    文档到幻灯片的生成极具挑战性，因为幻灯片是内容密集的可编辑产物，既要求忠实的内容选择，又要求精确的空间布局。近期的幻灯片智能体采用了迭代反思机制，但通常遵循单一的“一个版本、一个反馈”循环：先重写幻灯片或幻灯片组，之后才进行渲染，并且仅在轮次边界进行评价。这种延迟反馈使得溢出、重叠、裁剪和超出画布等局部错误难以归因和修复。我们提出ReDeck，这是一个步级的、基于渲染的细化框架，它将幻灯片修订分解为原子编辑操作，并在每一步之后返回由渲染器产生的观察结果，从而将细化过程转变为“一次编辑、一次观察”。为了平衡局部修复与全局质量，ReDeck采用了多粒度反馈机制：用于空间错误的步级渲染反馈、用于语义和设计指导的轮次级自适应评价器，以及一个提交级……

    arXiv:2609.00194v1 Announce Type: new  Abstract: Document-to-slide generation is challenging because slides are dense editable artifacts that require both faithful content selection and precise spatial layout. Recent slide agents adopt iterative reflection, but typically follow a monolithic "one version, one feedback" loop: a slide or deck is rewritten, rendered afterward, and critiqued only at the turn boundary. This delayed feedback makes local failures such as overflow, overlap, clipping, and off-canvas placement difficult to attribute and repair. We propose ReDeck, a step-level render-grounded refinement framework that decomposes slide revision into atomic edit actions and returns renderer-derived observations after each step, turning refinement into "one edit, one observation." To balance local repair with global quality, ReDeck uses multi-granular feedback: step-level render feedback for spatial errors, a turn-level adaptive critic for semantic and design guidance, and a submissi
    
[^255]: 具有线性函数逼近和对数级通信成本的可证明高效的联邦强化学习

    Provably Efficient Federated Reinforcement Learning with Linear Function Approximation and Logarithmic Communication Cost

    [https://arxiv.org/abs/2609.00193](https://arxiv.org/abs/2609.00193)

    提出Fed-LSVI，首个针对具有线性函数逼近的联邦在线强化学习的可证明高效算法，通过基于行列式的事件触发同步机制仅交换压缩充分统计量，在实现$\widetilde{O}(\sqrt{Md^3H^4T})$遗憾界的同时将通信成本降低至对数级。

    

    我们研究了具有线性函数逼近的联邦在线强化学习。尽管近期的多智能体强化学习算法实现了很强的遗憾保证，但它们通常需要共享原始轨迹。这种依赖性导致通信成本随回合数线性增长，并违反了联邦设置中的隐私约束。为了解决这些局限性，我们提出了Fed-LSVI，这是首个针对分段马尔可夫决策过程中具有线性函数逼近的在线强化学习的可证明高效的联邦算法。通过将基于行列式的事件触发同步机制与逐步反向更新机制相结合，Fed-LSVI使智能体能够通过仅交换压缩的充分统计量来协作学习最优策略。我们证明Fed-LSVI实现了$\widetilde{\mathcal O}(\sqrt{Md^3H^4T})$的遗憾界，其中$d$是特征维度，$H$是……

    arXiv:2609.00193v1 Announce Type: cross  Abstract: We study federated online reinforcement learning with linear function approximation. While recent multi-agent reinforcement learning algorithms achieve strong regret guarantees, they typically require sharing raw trajectories. This reliance incurs a communication cost that scales linearly with the number of episodes and violates the privacy constraints of federated settings. To address these limitations, we propose Fed-LSVI, the first provably efficient federated algorithm for online reinforcement learning with linear function approximation in episodic Markov decision processes. By integrating a determinant-based event-triggered synchronization with a stepwise backward update mechanism, Fed-LSVI enables agents to collaboratively learn an optimal policy by exchanging only compressed sufficient statistics. We prove that Fed-LSVI achieves a regret bound of $\widetilde{\mathcal O}(\sqrt{Md^3H^4T})$, where $d$ is the feature dimension, $H$ 
    
[^256]: 大语言模型驱动的自动驾驶汽车继承了人类驾驶员在行人让行方面的偏见：来自新基准的结果与启示

    LLM-Driven Autonomous Vehicles Inherit Human Driver Biases in Pedestrian Yielding: Results and Implications From A New Benchmark

    [https://arxiv.org/abs/2609.00192](https://arxiv.org/abs/2609.00192)

    本文提出两种新的偏见测试方法（“其他条件相同”测试和“自我一致性”测试），并发现大语言模型和视觉语言模型驱动的自动驾驶汽车在行人让行决策中会继承人类驾驶员的偏见，其决策受到行人性别、种族、宗教、残障状况和年龄等因素的影响。

    

    公众对自动驾驶汽车的信任可能不仅取决于技术上的成功，还取决于其决策的公平性。虽然自动驾驶研究中的一个新趋势是使用通用的“常识”模型来指导自动驾驶汽车的决策，但这些模型在多大程度上继承了人类驾驶员的偏见仍未得到充分研究。鉴于心理学研究表明人类驾驶员偏见确实存在，例如在美国，驾驶员对黑人行人的让行率较低，我们认为模型偏见分析也应成为自动驾驶汽车评估的一部分。具体而言，本文提出了两种针对大语言模型和视觉语言模型的新偏见测试方法——“其他条件相同”测试和“自我一致性”测试——以评估行人让行决策中的偏见。我们的研究结果表明，大语言模型和视觉语言模型做出的让行决策都会受到行人性别、种族、宗教、残障状况、年龄等因素的影响。

    arXiv:2609.00192v1 Announce Type: new  Abstract: Public trust in Autonomous Vehicles (AVs) may depend not only on technical success but also on the fairness of their decision making. While a recent trend in AV research involves using general purpose "common sense" models to guide AV decision making, the degree to which these inherit human biases in driving is still understudied. Given that psychology studies have shown human driver biases exist, such as lower pedestrian-yielding rates to Black pedestrians in the US, we argue that analyses of model bias should also be part of AV evaluation. Concretely, in this paper we propose two new bias testing methodologies for Large Language Models (LLMs) and Visual-Language Models (VLMs)-"All Else Being Equal" tests and "Self-Consistency" tests-in order to assess bias in pedestrian-yielding decisions. Our findings show that both LLMs and VLMs make yielding decisions which are influenced by pedestrian gender, ethnicity, religion, disability, age, s
    
[^257]: 阿拉伯语危机热线来电中的自杀风险评估：阿拉伯语与英语大语言模型的比较

    Assessing Suicide Risk in Arabic Crisis Helpline Calls: A Comparison of Arabic and English Large Language Models

    [https://arxiv.org/abs/2609.00191](https://arxiv.org/abs/2609.00191)

    该研究首次在真实阿拉伯语危机热线数据的严格隐私约束下，比较了阿拉伯语与英语大语言模型在自杀风险评估中的表现，填补了阿拉伯语热线自然语言处理研究的空白。

    

    危机热线通过结构化访谈评估自杀风险，这一过程缓慢且依赖于接线员的培训水平和工作量。自然语言处理可以支持风险评估和来电优先级排序，但几乎没有研究针对阿拉伯语热线电话，或在真实热线数据的隐私限制下开展相关工作。我们分析了来自黎巴嫩国家情感支持与自杀预防生命热线的去标识化转录文本。音频从未离开热线机构：来电在本地使用面向黎凡特阿拉伯语的语音识别模型进行转录，并由阿拉伯语命名实体识别模型在本地删除身份识别信息，只有去标识化的转录文本被共享给研究团队。接线员记录了哥伦比亚自杀严重程度评定量表（C-SSRS）中的五个自杀意念条目，我们将其合并为两个二元结果：有风险和高风险。我们还将转录文本进行了机器翻译……（原文摘要到此截断）

    arXiv:2609.00191v1 Announce Type: cross  Abstract: Crisis helplines assess suicide risk through structured interviews, a process that is slow and dependent on operator training and workload. Natural language processing could support risk assessment and call prioritization, but almost no work addresses Arabic-language helpline calls or operates within the privacy constraints of real helpline data. We analysed de-identified transcripts from Lebanon's National Lifeline for Emotional Support and Suicide Prevention. Audio never left the helpline: calls were transcribed on site with a speech recognition model for Levantine Arabic, and an Arabic named-entity recognition model removed identifying information locally. Only the de-identified transcripts were shared with the research team. Operators recorded the five suicidal ideation items of the Columbia Suicide Severity Rating Scale, which we combined into two binary outcomes: at-risk and high-risk. We also machine-translated the transcripts i
    
[^258]: 智能边缘计算

    Intelligent Edge Computing

    [https://arxiv.org/abs/2609.00181](https://arxiv.org/abs/2609.00181)

    本文提出了一种工作负载感知的列印记哈希连接方法 WACI-HJ，通过提前预测即将到来的查询工作负载来加速哈希连接，提升了实时边缘查询处理的效率。

    

    arXiv:2609.00181v1 公告类型：cross 摘要：大规模边缘系统中的边缘设备数量正迅速增长。边缘设备的处理能力、内存和网络带宽有限，这使得边缘查询处理过程中的资源利用和数据管理面临挑战。连接操作是数据库操作中在时间和资源方面代价最高的操作之一。当前最先进的边缘查询处理方法——列印记哈希连接，通过等高分箱技术来加速哈希连接，从而应对这一挑战。然而，该方法在实时处理方面效率不足，并且会扫描不必要的缓存行。本文提出了工作负载感知列印记哈希连接，它采用工作负载感知的方法来加速哈希连接。通过提前预测即将到来的查询工作负载，进一步提高了其对实时边缘查询处理的适用性。WACI-HJ 包含两个阶段：WACI-HJ 生成阶段，包括预处理、预测以及分块和哈希模块，以协同……（摘要在此处被截断）

    arXiv:2609.00181v1 Announce Type: cross  Abstract: The number of edge devices in large-scale edge systems is rapidly increasing. Edge devices have limited processing power, memory, and network bandwidth, making resource utilization and data management during edge query processing challenging. Joins are among the costliest database operations in terms of time and resources. The State-of-the-Art edge query processing, Column Imprint-Hash Join CI-HJ, addresses this challenge using equi-height binning to accelerate hash joins. However, it lacks efficiency in real-time processing and scans unnecessary cachelines. This paper presents Workload Aware Column Imprint-Hash Join WACI-HJ, which uses a workload-aware approach to accelerate hash joins. Predicting the upcoming query workload in advance further improves its suitability for real-time edge query processing. WACI-HJ comprises two phases: WACI-HJ Generation Phase, including Pre-processing, Prediction, and Blocking and Hashing modules to co
    
[^259]: 自发性欺骗与指令性欺骗中的不对称性

    Asymmetries in Spontaneous and Instructed Deception

    [https://arxiv.org/abs/2609.00180](https://arxiv.org/abs/2609.00180)

    该研究通过对 Llama-3.1-70B-Instruct 进行方向几何、跨设置分类器和跨设置引导的比较，发现大模型的自发性欺骗与指令性欺骗虽共享方向成分，但在检测与因果干预的跨设置迁移上存在不对称性，且引导向量与分类器的最佳 token 位置并不一致。

    

    大型语言模型有时会在未被指示的情况下欺骗用户。然而，目前关于模型欺骗的研究大多集中在指令性欺骗上。我们研究了 Llama-3.1-70B-Instruct 中指令性欺骗与自发性（无指令）欺骗之间的关系。我们通过方向几何、跨设置分类器以及跨设置引导（steering）对这两种欺骗设置进行了比较。我们发现这两种欺骗设置共享一个方向成分（余弦相似度约为 0.5），并且在检测与因果干预的跨设置迁移中存在不对称性：基于自发性欺骗训练的分类器在指令性欺骗数据上的表现优于反向情况，而基于指令性欺骗推导出的方向在引导自发性欺骗提示方面也优于反向情况。同样，用于推导引导向量的最佳 token 位置与训练和应用分类器的最佳 token 位置并不相同。

    arXiv:2609.00180v1 Announce Type: new  Abstract: Large language models sometimes deceive users without being instructed to. However, much of the study on deception in models involves instructed deception. We investigated the relationship between instructed and spontaneous (uninstructed) deception in Llama-3.1-70B-Instruct. We compared these two deception settings through direction geometry, cross-setting classifiers, and cross-setting steering. We found the two deception settings share a component of direction (cosine of approximately 0.5) and an asymmetry in the transfer between settings regarding detection and causation. Spontaneous trained classifiers performed better on instructed data than vice versa, and instructed derived directions performed better at steering spontaneous prompts than vice versa. Likewise the best token position to derive steering vectors from differed from the best token position to train and apply classifiers.
    
[^260]: 通用自然语言处理嵌入模型能否捕捉本体推理？

    Do General NLP Embeddings Capture Ontological Reasoning?

    [https://arxiv.org/abs/2609.00177](https://arxiv.org/abs/2609.00177)

    本文提出AVA评估框架，通过来自163个异构本体的171,007个对比三元组系统评估发现，现有最先进的NLP嵌入模型难以区分本体中对逻辑敏感的关系语义（最佳模型三元组准确率仅0.739），且微调带来的提升难以有效迁移到语义网下游任务。

    

    通用自然语言处理嵌入模型在语言任务上表现出色，但其捕捉符号化本体结构的能力仍不清楚。我们提出了AVA，一个系统性的评估框架，用于评估嵌入模型能否区分本体和知识图谱中对逻辑敏感的关系语义。AVA包含171,007个对比三元组，这些三元组通过层次反转、关系替换和不相交注入的方法从163个异构本体中构建。每个三元组包含一个本体陈述、一个语义等价的改写，以及一个具有矛盾关系含义的对逻辑敏感的困难负样本。我们评估了超过25个最先进的嵌入模型，发现了显著的局限性：最佳模型仅达到0.739的三元组准确率，而困难负样本的准确率更是降至0.135。微调可以大幅提升判别能力，但在包括分类体系在内的下游语义网任务上的迁移效果不佳。

    arXiv:2609.00177v1 Announce Type: cross  Abstract: General-purpose NLP embedding models perform well on linguistic tasks, but their ability to capture symbolic ontological structure remains unclear. We introduce AVA, a systematic framework for evaluating whether embeddings distinguish logic-sensitive relational semantics in ontologies and knowledge graphs. AVA comprises 171,007 contrastive triplets derived from 163 heterogeneous ontologies using hierarchy inversion, relation substitution, and disjointness injection. Each triplet contains an ontology statement, a semantically equivalent paraphrase, and a logic-sensitive hard negative with contradictory relational meaning. We evaluate more than 25 state-of-the-art embedding models and find substantial limitations: the best model achieves only 0.739 triplet accuracy, while hard negative accuracy falls to 0.135. Fine-tuning improves discrimination by a large margin but transfers poorly to downstream Semantic Web tasks, including taxonomy d
    
[^261]: IMPACT：注意力即可扩展的交互感知世界模型训练的交互图

    IMPACT: Attention Is the Interaction Map for Scalable Interaction-Aware World Model Training

    [https://arxiv.org/abs/2609.00161](https://arxiv.org/abs/2609.00161)

    该论文发现全局MSE去噪目标导致静态内容主导训练信号、稀疏动态交互区域监督不足的问题，提出IMPACT框架，通过先验引导的注意力校准将注意力本身作为交互图，无需外部估计器或人工标注即可实现可扩展的交互感知世界模型训练。

    

    世界模型在具身智能体的动作条件化未来预测方面已取得显著进展，但在建模物理上合理的交互方面仍存在困难。现有方法通过引入编码运动、几何或语义的外部表示来约束生成过程，以应对这一局限。然而，获取这些时空密集的表示通常需要辅助估计器或人工标注，限制了训练的可扩展性。我们转而重新审视训练目标，发现全局平均均方误差（MSE）去噪目标下存在监督分配失配问题：普遍存在的静态内容主导了优化信号，使得对交互生成至关重要的稀疏动态物体区域得到的监督严重不足。基于这一观察，我们提出了IMPACT——一个采用先验引导的注意力校准的可扩展交互感知模型训练框架……

    arXiv:2609.00161v1 Announce Type: new  Abstract: World models have made remarkable progress in action-conditioned future prediction for embodied agents, yet still struggle to model physically plausible interactions. Existing approaches address this limitation by constraining the generation process with external representations encoding motion, geometry, or semantics. Obtaining these spatiotemporally dense representations typically requires auxiliary estimators or manual annotations, limiting training scalability. We instead revisit the training objective and identify a supervision-allocation mismatch under the globally averaged mean squared error (MSE) denoising objective: prevalent static content dominates the optimization signal, leaving sparse dynamic-object regions critical to interaction generation disproportionately under-supervised. Motivated by this observation, we introduce IMPACT, a scalable Interaction-aware Model training framework with Prior-guided Attention Calibration an
    
[^262]: 通用语还是探测假象？重新思考多语言大语言模型中的潜在语言

    Lingua Franca or Probing Artifact? Rethinking Latent Language in Multilingual LLMs

    [https://arxiv.org/abs/2609.00155](https://arxiv.org/abs/2609.00155)

    该研究发现不同的潜在语言探测方法会得出系统性不一致的结论，表明多语言大模型通过英语等“潜在通用语”路由计算的说法可能更多取决于探测手段的选择，而非模型本身固有的计算机制。

    

    潜在语言识别常被用来论证多语言语言模型通过语言特定状态（如英语枢纽）来路由计算。然而，现有探测方法从不同信号推断潜在语言，例如隐藏状态的几何结构，或可从中间表示中解码出的内容。由于这类论断会影响关于模型如何跨语言共享和路由信息的结论，我们追问：这些探测方法测量的究竟是同一现象，还是揭示了多语言计算的不同侧面？我们在多种模型家族、训练方式、领域、任务、检查点以及多达27种语言上研究了这一问题。我们发现，各类识别探测方法存在系统性的不一致：基于GMM的表示探测方法从隐藏状态几何结构中获取证据，显示出更早出现的跨语言混合；而依赖输出空间可解码性的解码式探测方法，则保留了更鲜明的语言特定性，以及

    arXiv:2609.00155v1 Announce Type: cross  Abstract: Latent language identification is often used to argue that multilingual language models route computation through language-specific states, such as English pivots. However, existing probes infer latent language from different signals, such as the geometry of hidden states or what can be decoded from intermediate representations. Since such claims shape conclusions about how models share and route information across languages, we ask whether these probes measure the same phenomenon or expose distinct aspects of multilingual computation. We study this question across model families, training regimes, domains, tasks, checkpoints, and up to 27 languages. We find that identification probes systematically disagree: the GMM-based representation probe, which draws evidence from hidden state geometry, shows earlier cross-lingual mixing, whereas decoding-based probes, which rely on output-space decodability, retain sharper language-specific and 
    
[^263]: AI自我改进的递归临界性

    Recursive Criticality of AI Self-Improvement

    [https://arxiv.org/abs/2609.00137](https://arxiv.org/abs/2609.00137)

    该论文提出递归繁殖数 $\mathcal{R}_{\mathrm{AI}}$ 作为判断AI自我改进是否自我放大的临界指标——大于1时改进效果在开发周期间复合累积，小于1时则逐渐衰减，且临界点取决于AI研发反馈回路的结构而非特定模型能力水平。

    

    AI正被越来越多地应用于研发未来AI系统的过程中。我们研究了这种反馈在何种条件下会变得自我放大。我们的模型描述了AI能力增长的速度如何取决于基线研究生产力、递归反馈以及研究进展日益增加的难度。我们推导出一个递归繁殖数 $\mathcal{R}_{\mathrm{AI}}$，它决定了改进效果在各个开发周期中是被放大还是被衰减。该量将反馈的强度与研究进展变得愈发困难的速率进行比较。当 $\mathcal{R}_{\mathrm{AI}}>1$ 时，改进的效果会在各个开发周期中复合累积，使系统进入自我放大的状态；当 $\mathcal{R}_{\mathrm{AI}}<1$ 时，改进的效果则会在各个周期中逐渐减弱。这一转变取决于AI研发反馈回路的结构，而不必发生在某个特定的模型能力水平上。

    arXiv:2609.00137v1 Announce Type: new  Abstract: AI is increasingly used in the R\&D process that produces future AI systems. We study the conditions under which this feedback becomes self-amplifying. Our model describes how the rate of AI capability growth depends on baseline research productivity, recursive feedback, and the increasing difficulty of research progress. We derive a recursive reproduction number, $\mathcal{R}_{\mathrm{AI}}$, that determines whether improvements are amplified or damped across development cycles. This quantity compares the strength of feedback with the rate at which further progress becomes more difficult. When $\mathcal{R}_{\mathrm{AI}}>1$, the effects of improvements compound across development cycles, placing the system in a self-amplifying regime. When $\mathcal{R}_{\mathrm{AI}}<1$, their effects weaken across cycles. The transition depends on the structure of the AI R\&D feedback loop and need not occur at any particular level of model capability. A 
    
[^264]: 天生有缺陷，进化致完美

    Flawed in Nature, Perfect through Evolution

    [https://arxiv.org/abs/2609.00129](https://arxiv.org/abs/2609.00129)

    该论文提出通过让一群AI/ML模型的系数刻意发生偏离最优的“突变”来维持模型多样性，从而在非平稳环境中充当统计对冲手段，实现可靠且持续的性能提升。

    

    当人工智能（AI）和机器学习（ML）模型所训练的问题发生漂移时，其性能会下降。这是现实世界问题中近乎普遍的特征，因为这些问题往往会发生不可预测的变化。生物进化通过自然选择作用于可遗传的变异，克服了这一障碍，从而实现了智能。AI/ML技术早已融入了各种形式的自然选择，但随着优化过程自然地驱使模型趋同，维持模型多样性一直是一项挑战。在这项工作中，我们展示了一群AI/ML模型在受到使其模型系数刻意偏离最优状态的突变后，能够在变化的环境中可靠且持续地提升性能，其原理是作为应对非平稳性的统计对冲手段。我们将这一机制称为“天生有缺陷，进化致完美”，这反映了集体性能的提升是以牺牲个体性能为代价的。

    arXiv:2609.00129v1 Announce Type: cross  Abstract: The performance of artificial intelligence (AI) and machine learning (ML) models degrades when the problem they were trained on drifts. This is a near-universal feature of real-world problems, which often change unpredictably. Biological evolution has achieved intelligence by overcoming this obstacle through natural selection acting on heritable variation. AI/ML techniques have long incorporated forms of natural selection, but it has been challenging to maintain model diversity as optimization naturally drives convergence. Here we show that a swarm of AI/ML models subjected to deliberate mutations of their model coefficients away from optimality can reliably and sustainably improve performance in changing environments by acting as a statistical hedge against non-stationarity. We call this mechanism 'Flawed in Nature, Perfect through Evolution', reflecting that the collective performance gain goes at the expense of individual performanc
    
[^265]: 部署与评估面向全季大豆农场作业的智慧农业智能体引擎

    Deploying and Evaluating a Smart-Agriculture Agentic Engine for Full-Season Soybean Farm Operations

    [https://arxiv.org/abs/2609.00106](https://arxiv.org/abs/2609.00106)

    提出并实际部署了全栈智慧农业智能体系统FAIRY，以“万物皆事件”执行范式为核心，整合农机、传感器、无人机、卫星与作物模型等多源API，在哈工大运营中的大豆研究农场执行并评估从整地到储粮的全季时空农艺工作流。

    

    本文提出了FAIRY，一个为哈尔滨工业大学智慧农业站点某正在运营的大豆研究农场开发并实际部署的全栈智慧农业智能体系统。我们开发FAIRY旨在跨越起垄整地、播种、灌溉、施肥、病虫害防治、收获、粮食处理、干燥和储存的全季时空工作流上，执行并评估智能体化的农艺作业。FAIRY整合了生产级农机、固定式土壤与冠层传感器、多光谱及热红外无人机、卫星植被产品、气象站、经过校准的作物过程模型、农艺记录以及多季产量历史等多方面的API与基础设施。该系统围绕新颖的“万物皆事件”执行范式构建，该范式用于表示时空世界演化、遥感与无人机观测、传感器读数、作物生长转变以及农机动作等。

    arXiv:2609.00106v1 Announce Type: new  Abstract: This paper presents FAIRY, a full-stack smart-agriculture agent system developed for and deployed to an operating soybean research farm at Harbin Institute of Technology's smart-agriculture site. We develop FAIRY to execute and evaluate agentic agronomic operations on full-season spatiotemporal workflows that span ridge preparation, planting, irrigation, fertilization, pest and disease treatment, harvest, grain handling, drying, and storage. FAIRY integrates APIs and infrastructure across production-grade machinery, fixed soil and canopy sensors, multispectral and thermal drones, satellite vegetation products, a weather station, calibrated crop-process models, agronomic records, and multi-season yield histories. The system is built around the novel "everything is an event" execution paradigm, which represents spatiotemporal world evolution, remote sensing and UAV observations, sensor readings, crop-growth transitions, machinery actions, 
    
[^266]: 好的记忆具备ECC：超越准确率评估视觉-语言模型的记忆能力

    Good Memory Has ECC: Evaluating the Memory of Vision-Language Models Beyond Accuracy

    [https://arxiv.org/abs/2609.00103](https://arxiv.org/abs/2609.00103)

    该论文提出ECCBench基准，从效率、压缩和校准三个维度超越单纯准确率来评估视觉-语言模型的记忆能力，发现预训练VLM对文本记忆有压缩但对视频没有且校准较差，并且若干非Transformer架构在压缩-校准权衡上优于RoPE Transformer。

    

    记忆被广泛认为是大语言模型（LLM）和视觉-语言模型（VLM）面临的一个重要的未解决问题，当前的基准测试通常通过测试模型在长文本或视频上的准确率来评估记忆能力。然而，仅凭准确率会忽略那些对真实长时程任务至关重要的属性。我们提出了ECCBench，这是一个基准测试和评估协议，通过我们称为ECC的三个维度来衡量超越系统容量（即在特定预算下的原始准确率）的记忆能力：效率——从记忆中回答问题所需的计算量（以FLOPs计）；压缩——可压缩的输入是否被更准确或更高效地记住；校准——系统是否会针对自身的不确定性选择弃答，以及出错的代价。我们发现，预训练的VLM会对文本记忆进行压缩，但对视频则不会，且在两者上的校准都很差。在更广泛的记忆骨干架构中，若干非Transformer架构实现了比RoPE Transformer更好的压缩-校准权衡。

    arXiv:2609.00103v1 Announce Type: cross  Abstract: Memory is widely viewed as an important unsolved problem for LLMs and VLMs, and current benchmarks typically evaluate it by testing accuracy over long text or video. However, accuracy alone misses properties that matter for real long-horizon tasks. We introduce ECCBench, a benchmark and evaluation protocol that measures memory beyond a system's capacity--its raw accuracy at a specific budget--via three axes we call ECC: efficiency--the computation, in FLOPs, needed to answer from memory; compression--whether compressible inputs are remembered more accurately or efficiently; and calibration--whether the system abstains in response to its own uncertainty and the cost of an error. We find that pretrained VLMs compress their memory over text but not video and are poorly calibrated on both. Among a broader set of memory backbones, several non-Transformer architectures achieve better compression-calibration tradeoffs than RoPE Transformers, 
    
[^267]: 不同的表征学习目标从相同的心理测量数据中恢复出不同的潜在结构

    Different representation learning objectives recover distinct latent structures from the same psychometric data

    [https://arxiv.org/abs/2609.00100](https://arxiv.org/abs/2609.00100)

    不同的表征学习目标会从同一份心理测量数据中恢复出截然不同的潜在结构——对比学习大幅提升师生匹配检索性能却破坏行为表型组织，而PCA更利于保留行为结构，揭示了检索对齐与行为结构保留之间的根本性权衡。

    

    心理测量问卷包含丰富的条目级信息，然而不同的表征学习目标是否能恢复出相同的潜在组织结构仍不清楚。我们利用塞浦路斯ProW学前试验基线评估中的757对匹配的师生配对数据研究了这一问题。通过使用主成分分析和聚类方法从儿童SDQ、ASBI和CBRS的条目回答中刻画行为结构，得到了四种行为表型。与基于PCA的表征相比，对比学习目标显著提升了教师-儿童配对检索性能，将Top-1准确率从0.13%提高到7.27%，Top-10准确率从1.98%提高到56.14%。然而，对比表征在保留行为表型结构方面不如基于PCA的表征有效。一种联合优化对齐与行为预测的多任务目标部分恢复了行为组织结构，但降低了检索性能。

    arXiv:2609.00100v1 Announce Type: new  Abstract: Psychometric questionnaires contain rich item-level information, yet it remains unclear whether different representation learning objectives recover the same latent organization. We investigated this question using 757 matched teacher-child pairs from the baseline assessment of the Cyprus ProW preschool trial. Behavioral structure was characterized from child SDQ, ASBI, and CBRS item responses using principal component analysis and clustering, yielding four behavioral phenotypes. A contrastive objective substantially improved teacher-child retrieval relative to PCA-based representations, increasing Top-1 accuracy from 0.13% to 7.27% and Top-10 accuracy from 1.98% to 56.14%. However, contrastive representations preserved behavioral phenotype structure less effectively than PCA-based representations. A multi-task objective jointly optimizing alignment and behavioral prediction partially restored behavioral organization but reduced retrieva
    
[^268]: 快过Flash：利用注意力稀疏性实现高效长上下文解码

    Faster Than Flash: Exploiting Attention Sparsity for Efficient Long-Context Decoding

    [https://arxiv.org/abs/2609.00097](https://arxiv.org/abs/2609.00097)

    FFD是一种硬件-算法协同设计框架，通过将选择器与计算器融合为单一内核、基于低比特量化的内容感知扫描取代元数据索引，以及无需全局同步的top-delta动态块过滤策略，实现了免训练、即插即用的长上下文解码加速，内核级加速比最高达11.6倍。

    

    长上下文大语言模型（LLM）的发展受到解码过程中注意力机制的内存带宽瓶颈和二次方复杂度的制约。为了克服基于元数据的度量方法的内存开销与自适应选择策略的计算低效之间的固有权衡，我们提出了快速Flash解码，这是一种新颖的硬件-算法协同设计框架，旨在打破长上下文解码中的内存墙。FFD将选择器和计算器集成到一个完全融合的内核中，通过低比特量化的内容感知扫描取代外部元数据索引。此外，我们引入了top-delta策略，该策略动态过滤注意力块以实现分布自适应的稀疏性，而无需全局同步。FFD提供了一种免训练、即插即用的解决方案，还支持将扫描结果复用于计算，实现了高达11.6倍的内核级加速。

    arXiv:2609.00097v1 Announce Type: cross  Abstract: The development of long-context Large Language Models (LLMs) is constrained by the memory bandwidth bottleneck and quadratic complexity of the attention mechanism during decoding. To overcome the inherent trade-offs between the memory overhead of metadata-based metrics and the computational inefficiency of adaptive selection strategies, we present Faster Flash Decoding (FFD), a novel hardware-algorithm co-design framework designed to break the memory wall in long-context decoding. FFD integrates the selector and computer into a fully fused kernel, replacing external metadata indices with content-aware scanning via low-bit quantization. Furthermore, we introduce the top-delta strategy, which dynamically filters blocks to achieve distribution-adaptive sparsity without global synchronization. Offering a training-free and plug-and-play solution, FFD also enables the reuse of scanning results for computation, achieving up to 11.6x kernel-le
    
[^269]: 基于证据权重评估特征重要性解释的对齐性与稳定性

    Assessing Alignment and Stability of Feature Importance Explanations via Weight of Evidence

    [https://arxiv.org/abs/2609.00090](https://arxiv.org/abs/2609.00090)

    该论文提出了一个基于证据权重的假设检验框架，能够从原理上评估特征重要性解释方法与先验知识的对齐程度及其稳定性，并将其应用于LIME和SHAP的分析中。

    

    特征重要性方法（FIMs）被广泛应用于可解释人工智能中，用于解释模型的预测结果，然而仅凭归因分数往往难以深入洞察模型背后的推理过程。在本工作中，我们引入了一种新颖的视角，将特征重要性方法嵌入到基于证据权重的假设检验框架中。我们量化了观测到的证据对任何给定的特征重要性假设的支持强度。其中，参考假设可以来源于领域知识、真实标签，或由特征重要性方法本身推导得出。这一表述方式使得对特征重要性方法的原理化评估成为可能，既能捕捉它们与先验知识的对齐程度，也能捕捉其变异性。我们进一步提供了将证据权重与归因方差相联系的理论结果。实证结果表明，在具有不同参考假设的设置下分析LIME和SHAP解释时，我们的策略展现出良好的适用性与灵活性。总体而言，我们的框架提供了一个互补的……

    arXiv:2609.00090v1 Announce Type: cross  Abstract: Feature importance Methods (FIMs) are widely used in Explainable AI to interpret model predictions, yet attribution scores alone often provide limited insight into the underlying reasoning process. In this work, we introduce a novel perspective by embedding FIMs within a hypothesis-testing framework based on Weight of Evidence (WoE). We quantify how strongly the observed evidence supports any given hypothesis on feature importance. The reference hypothesis can stem from domain knowledge, ground truth, or be derived from the FIM itself. This formulation enables a principled evaluation of FIMs, capturing both their alignment with prior knowledge and their variability. We further provide theoretical results linking WoE to attribution variance. Empirical results shows the applicability and flexibility of our strategy analyzing LIME and SHAP explanations in settings with different reference hypotheses. Overall, our framework offers a comple
    
[^270]: 先答后判式LLM评判会继承评判者自身的错误

    Commit-first LLM judging inherits the judge's own errors

    [https://arxiv.org/abs/2609.00088](https://arxiv.org/abs/2609.00088)

    研究发现“先答后判”式LLM评判会继承评判者自身的错误，而对八个主流评估框架的审计表明无一真正实现该方法，其中九个框架因复制同一祖先提示词而采用了已被证明无效的变体，导致大量错误代码被放行。

    

    LLM评判器（即对另一个系统输出进行打分的模型）可能被被其评分的系统“钻空子”。近期研究指出了一种确实有效的防御方法：评判器先自行解决任务并固定自己的答案，然后仅当候选答案与其一致时才予以接受。我们将这一做法称为“先答后判”评判，并探究已发布的软件是否实现了该方法，以及其代价是什么。我们审计了八个广泛使用的评估框架的默认评判器配置：在纳入范围的24个配置中，没有一个实现了该方法；其中九个实现了文献中被测得无效的一种变体，并且共享同一个祖先提示词——这一点可以通过一个被复制下来的排版错误进行追溯。在一项受控实验中，一个普通的、无法访问正确答案的best-of-N搜索，严格按照文档说明使用其中一个配置来优化代码。在一个区间合并任务上，该评判器在一个随机种子下接受了96个候选中的90个，在另一个种子下接受了93个；每个被接受的候选……（原文摘要在此截断）

    arXiv:2609.00088v1 Announce Type: cross  Abstract: LLM judges, models that score another system's output, can be gamed by the systems they score. Recent work identifies one defence that works: the judge solves the task itself first and commits to that answer, then accepts a candidate only if the two match. We call this commit-first judging, and ask whether shipped software implements it, and what it costs.   We audit the default judge configurations of eight widely used evaluation frameworks. Of the 24 configurations in scope, none implement it. Nine implement a variant the literature measures as ineffective, and share one ancestor prompt, traceable through a copied typographical error.   In a controlled experiment, an ordinary best-of-N search with no access to correct answers optimises code against one of these configurations, used exactly as documented. On an interval merging task the judge accepted 90 of 96 candidates in one seed and 93 of 96 in the other; every accepted candidate 
    
[^271]: 检索、评分与解码如何塑造基于大语言模型的对话推荐系统的性能与稳定性

    Retrieval, Scoring, and Decoding Shape Performance and Stability in LLM-based Conversational Recommendation

    [https://arxiv.org/abs/2609.00086](https://arxiv.org/abs/2609.00086)

    该研究系统评估了大语言模型作为对话推荐重排序器的表现，发现在统一候选池协议下最佳专有LLM仅小幅超越传统基线，自由生成评估会夸大其优势，且所有开源LLM均未超过调优的浅层自编码器基线，说明检索、评分与解码协议显著影响LLM在对话推荐中的表现。

    

    大语言模型（LLM）越来越多地被用作对话推荐系统中的重排序器，然而其测得的收益在很大程度上取决于检索与推理协议。在ReDial对话式电影推荐基准上，我们在一个共享的“先检索后重排序”流水线中，比较了专有模型、开源权重模型以及微调的LLM重排序器与协同过滤和序列推荐基线，并改变了候选池大小、第一阶段检索器和解码温度。在共享的语义top-250候选池和严格的候选感知评分条件下，最佳的专有重排序器达到NDCG@10为0.1497，而最强的非LLM基线为0.0939。同一重排序器在零样本生成模式下达到0.2925，这表明无约束的评分可以产生比匹配候选池评估大得多的表面优势。在该协议下，没有任何被评估的开源权重LLM优于经过调优的浅层自编码器基线。

    arXiv:2609.00086v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used as rerankers in conversational recommender systems, yet measured gains depend strongly on the retrieval and inference protocol. On the ReDial conversational movie recommendation benchmark, we compare proprietary, open-weight, and fine-tuned LLM rerankers with collaborative-filtering and sequential baselines in a shared retrieve-then-rerank pipeline. We vary candidate-pool size, first-stage retriever, and decoding temperature. With a shared semantic top-250 candidate pool and strict candidate-aware scoring, the best proprietary reranker reaches NDCG@10 of 0.1497, compared with 0.0939 for the strongest non-LLM baseline. The same reranker reaches 0.2925 in zero-shot generation, showing that unconstrained scoring can yield a much larger apparent advantage than matched-pool evaluation. No evaluated open-weight LLM outperforms the tuned shallow autoencoder baseline under this protocol. For t
    
[^272]: KItCAT：通过输入破坏进行知识注入的自回归训练方法

    KItCAT: Knowledge Injection via Input Corruption for Auto-regressive Training

    [https://arxiv.org/abs/2609.00082](https://arxiv.org/abs/2609.00082)

    提出KItCAT轻量级训练策略，通过在下一词预测训练中对输入序列进行随机破坏，从而在无需昂贵改写的情况下，将小众专业知识有效注入仅解码器大语言模型。

    

    大语言模型（LLM）在预训练期间获取了大量知识，但往往缺乏回答来自小众来源（如预训练时未见过的手册或技术文档）问题所需的专业知识。持续预训练（CPT）被广泛用于将这类知识注入模型参数，然而小众文档很少重复相同的事实，这使得CPT难以稳健地获取此类知识。近期的工作通过生成新知识的多个改写版本来解决这一问题，但改写计算成本高昂，且通常需要强大的大语言模型。在本工作中，我们提出了KItCAT：通过破坏的自回归训练进行知识注入，这是一种轻量级训练策略，可减少仅解码器大语言模型对改写的需求。KItCAT通过对输入序列进行随机破坏来增强标准的下一词预测。在训练过程中，输入词元的随机子集会被替换为词表中的其他词元。

    arXiv:2609.00082v1 Announce Type: cross  Abstract: LLMs acquire vast amounts of knowledge during pre-training, but often lack the specialized knowledge needed to answer questions from niche sources such as manuals or technical documents unseen during pre-training. Continued pre-training (CPT) is widely used to inject such knowledge into model parameters. However, niche documents seldom repeat facts, making it difficult for CPT to robustly acquire such knowledge. Recent works address this by generating multiple paraphrases of the new knowledge, but paraphrasing is computationally expensive and typically requires powerful LLMs. In this work, we introduce KItCAT: Knowledge Injection via Corrupted Auto-regressive Training, a lightweight training strategy that reduces the need for paraphrasing in decoder-only LLMs. KItCAT augments standard next-token prediction by stochastically corrupting the input sequence. During training, a random subset of input tokens is replaced with other vocabulary
    
[^273]: RW-LoRA：基于随机游走的通信高效去中心化 LoRA 微调

    RW-LoRA: Communication-Efficient Decentralized LoRA Fine-Tuning via Random Walks

    [https://arxiv.org/abs/2609.00078](https://arxiv.org/abs/2609.00078)

    提出基于随机游走的去中心化 LoRA 微调方法 RW-LoRA，通过单个模型令牌在网络中顺序更新，免除全局同步，大幅降低通信与计算成本并避免聚合误差。

    

    arXiv:2609.00078v1 公告类型：cross 摘要：以 LoRA 为代表的参数高效微调方法已成为适配大型基础模型的标准方式。然而，将微调推广到分布式场景面临诸多挑战：大多数现有的分布式 LoRA 方法依赖集中式聚合，而基于 gossip 的去中心化 LoRA 则需要在多个模型副本之间进行反复同步。这两种方式都会带来巨大的通信开销，并因同时聚合多个模型更新而引入误差。本文从一个不同的视角出发，提出了一种基于随机游走的 LoRA 微调方案：无需维护多个模型副本，而是让单个模型令牌在网络中游走，并利用本地微调目标对其进行顺序更新。该设计消除了全局同步的需求，显著降低了通信与计算成本，并避免了聚合误差。我们提供了严格的收敛保证。

    arXiv:2609.00078v1 Announce Type: cross  Abstract: Parameter-efficient fine-tuning methods such as LoRA have become a standard approach for adapting large foundation models. Adopting fine-tuning to distributed settings faces several challenges. Most existing distributed LoRA methods rely on centralized aggregation, and gossip-based decentralized LoRA requires repeated synchronization among multiple model copies. Both methods incur significant communication overhead and introduce errors due to simultaneous aggregation of multiple model updates. In this paper, we take a different perspective and propose a random-walk-based LoRA fine-tuning scheme. Instead of maintaining multiple model replicas, a single model token traverses the network and is updated sequentially using local fine-tuning objectives. This design eliminates the need for global synchronization, substantially reduces communication and computation costs, and avoids aggregation errors. We provide rigorous convergence guarantee
    
[^274]: AI发病与致死事件：临床AI失效审查框架

    AI Morbidity and Mortality: A Framework for Clinical AI Failure Review

    [https://arxiv.org/abs/2609.00076](https://arxiv.org/abs/2609.00076)

    提出了AI M&M——一个结构化、无指责的临床AI失效病例审查框架，通过“触发因素-机制-临床路径-纠正措施”四维分类，结合证据还原、工具在环归因与纠正措施跟踪，系统性还原并学习个体层面的AI相关错误与险情。

    

    临床人工智能正日益深入地嵌入真实世界的医疗照护中，然而现有的安全机制并不适合对个体层面的AI相关错误和险些发生事件进行还原与学习。汇总性的模型监测可以识别性能变化，传统的患者安全报告可以捕捉不良事件，但两者都并非为解释风险如何在AI系统、临床医生、工作流程与机构控制之间的交互中产生而设计。我们提出AI发病率与死亡率（AI M&M），这是一个结构化、无指责的框架，用于对临床AI失效进行基于病例的审查。该框架结合了标准化的病例接收、证据保存与调查者层面的还原、工具在环的归因分析以及纠正措施跟踪。每个事件在四个相互关联的维度上进行分类：触发因素 - 机制 - 临床路径 - 纠正措施，从而将暴露出系统脆弱性的条件与后续的风险演化过程分离开来（摘要在此处被截断）。

    arXiv:2609.00076v1 Announce Type: new  Abstract: Clinical artificial intelligence is increasingly embedded in real-world care, yet existing safety mechanisms are poorly suited to reconstructing and learning from individual AI-related errors and near-misses. Aggregate model monitoring can identify performance changes, and traditional patient safety reporting can capture adverse events, but neither is designed to explain how risk emerges across the interaction among AI systems, clinicians, workflows, and institutional controls. We propose AI Morbidity and Mortality (AI M&M), a structured, blameless framework for case-based review of clinical AI failures. The framework combines standardized case intake, evidence preservation and investigator-level reconstruction, tool-in-loop attribution, and corrective-action tracking. Each event is classified across four linked dimensions: Trigger - Mechanism - Clinical Pathway - Corrective Action, separating the condition that exposed a vulnerability f
    
[^275]: MiNER：面向临床文本中疟疾疾病实体识别的微调生物医学自然语言处理

    MiNER: Fine-Tuned Biomedical Natural Language Processing for Malaria Disease Entity Recognition in Clinical Texts

    [https://arxiv.org/abs/2609.00073](https://arxiv.org/abs/2609.00073)

    本文提出MiNER方法，通过对预训练生物医学语言模型BioBERT进行微调，实现疟疾临床文本中疾病实体的自动识别，从而从海量疟疾科学文献中高效提取具有临床意义的生物医学信息。

    

    疟疾仍然是一个重大的全球健康负担，需要持续的研究努力来理解其复杂的分子机制、流行病学以及潜在的治疗干预手段。从庞大且不断增长的疟疾文献中提取关键的生物医学信息是一项具有挑战性的任务，需要创新的方法。近年来，预训练语言模型彻底改变了自然语言处理任务，在各个领域展现出卓越的能力。本文提出了一种经过微调的预训练生物医学语言模型，用于从疟疾疾病的科学文献中进行生物医学信息抽取。该方法首先选取并预处理了大规模的疟疾科学文章语料库，然后使用具有临床意义的实体对其进行标注，进而利用最先进的预训练语言模型BioBERT，将文本数据编码为上下文感知的表示。

    arXiv:2609.00073v1 Announce Type: new  Abstract: Malaria remains a significant global health burden, necessitating continuous research efforts to understand its complex molecular mechanisms, epidemiology, and potential therapeutic interventions. Extracting essential biomedical information from the vast and constantly growing malaria literature is a challenging task that demands innovative approaches. Recently, pre-trained language models have revolutionized natural language processing tasks, demonstrating remarkable capabilities in various domains. This paper proposes a fine-tuned pre-trained biomedical language model for biomedical information extraction from scientific literature on malaria disease. The proposed methodology selects and preprocesses a large corpus of scientific articles on malaria, and then annotates them with entities of clinical significance. It then leverages BioBERT, a state-of-the-art pre-trained language model, to encode the textual data into context-aware repre
    
[^276]: 当预测误差不足够时：评估用于因果估计的干扰函数预测

    When Prediction Error Is Not Enough: Evaluating Nuisance-Function Prediction for Causal Estimation

    [https://arxiv.org/abs/2609.00071](https://arxiv.org/abs/2609.00071)

    在部分线性模型的模拟研究中，干扰函数的预测误差无法一致地反映因果估计的偏差，表明仅用预测误差来评估干扰函数估计器对于因果推断而言是不充分的。

    

    预测误差被广泛用于评估因果推断中干扰函数估计器的性能，但其与因果估计器表现之间的关系可能因性能度量指标的不同而存在差异。我们在部分线性模型中通过蒙特卡洛模拟研究了这一问题。我们比较了普通最小二乘法（OLS）、广义可加模型（GAMs）、XGBoost 以及结合 XGBoost 的双重机器学习（DML-XGBoost），评估指标包括干扰函数预测误差、偏差、均方根误差（RMSE）以及95%置信区间覆盖率。我们还检验了一种简单的联合误差度量，其基于暴露干扰函数与结局干扰函数估计误差的绝对交叉乘积。在所有模拟设置中，XGBoost 在非先知方法中具有最低的 RMSE，而 DML-XGBoost 通常能提供更好的置信区间覆盖率。预测误差在不同方法和设置下并不能一致地反映因果估计的偏差，且点估计表现最好的方法（摘要在此处截断）

    arXiv:2609.00071v1 Announce Type: new  Abstract: Prediction error is widely used to evaluate nuisance-function estimators in causal inference, but its relationship with causal estimator performance may differ across performance measures. We studied this question in a partially linear model using Monte Carlo simulations. We compared ordinary least squares (OLS), generalized additive models (GAMs), XGBoost, and Double Machine Learning with XGBoost (DML-XGBoost), evaluating nuisance-function prediction error, bias, RMSE, and 95\% confidence interval coverage. We also examined a simple joint-error measure based on the absolute cross-product of estimation errors from the exposure and outcome nuisance functions. Across the simulated settings, XGBoost had the lowest RMSE among the non-oracle methods, while DML-XGBoost generally provided better confidence interval coverage. Prediction error did not consistently track causal bias across methods and settings, and the method with the best point-e
    
[^277]: AutoXRD：用于粉末衍射分析的自主大语言模型智能体与综合评估

    AutoXRD: Autonomous LLM Agents and Comprehensive Evaluation for Powder Diffraction Analysis

    [https://arxiv.org/abs/2609.00070](https://arxiv.org/abs/2609.00070)

    本文提出AutoXRD——一个将粉末XRD分析组织为基于证据的逐步精修并加入确定性晶体学物理检查的自主大语言模型智能体框架，同时推出包含100个诊断问答任务和34个端到端工作流的XRDBench基准，用于全面评估智能体的XRD分析能力。

    

    粉末X射线衍射（XRD）是材料表征的核心手段，然而可靠的端到端自动化仍然具有挑战性。一个XRD智能体必须解读衍射证据、操作精修软件、以合理的顺序管理耦合参数，并区分数值上的改进与物理上的有效性。在本文中，我们提出了AutoXRD，这是一个自主的大语言模型（LLM）智能体框架，它将粉末XRD分析组织为逐步精修的过程，使行动立足于观察到的证据，并在接受结果之前应用确定性的晶体学和物理检查。我们进一步引入了XRDBench基准，包含两个互补的赛道：XRDBench-QA包含100个有边界的诊断任务，用于隔离评估科学推理和决策能力；而XRDBench-E2E包含34个可执行的工作流，用于测试智能体能否将这些能力组合成需要文件检查、晶体学软件操作的完整分析。

    arXiv:2609.00070v1 Announce Type: cross  Abstract: Powder X-ray diffraction (XRD) is central to materials characterization, yet reliable end-to-end automation remains challenging. An XRD agent must interpret diffraction evidence, operate refinement software, manage coupled parameters in a defensible order, and distinguish numerical improvement from physical validity. In this paper, we propose AutoXRD, an autonomous large language model (LLM) agent framework that organizes powder-XRD analysis as stepwise refinement, grounds actions in observed evidence, and applies deterministic crystallographic and physical checks before accepting results. We further introduce XRDBench with two complementary tracks. XRDBench-QA contains 100 bounded diagnostic tasks that isolate scientific reasoning and decision-making, whereas XRDBench-E2E contains 34 executable workflows that test whether agents can compose these capabilities into complete analyses requiring file inspection, crystallographic-software 
    
[^278]: 审计自我改进智能体中的框架篡改行为

    Auditing Harness Tampering in Self-Improving Agents

    [https://arxiv.org/abs/2609.00069](https://arxiv.org/abs/2609.00069)

    该论文提出了“框架篡改”概念及其双轴分类体系，通过构建带标注的篡改语料库并对审计方法进行基准测试，系统研究并检测自我改进智能体对自身框架的不当修改。

    

    自我改进智能体会迭代地修改自身的运行框架以突破其性能边界。然而，这类修改可能产生虚幻的性能提升，或者在不真正提升能力的情况下损害授权、溯源和完整性等完整性约束。我们将这种现象称为框架篡改，它将奖励篡改和测量篡改的概念扩展到了完整的自我改进生命周期。为了系统地研究这一问题，我们提出了一个双轴分类法，根据篡改编辑发生的框架功能角色以及其违反的义务来对每次失准编辑进行分类。随后，我们通过向自我改进智能体的真实轨迹中植入篡改-良性编辑对来构建带标注的语料库。我们对多种审计方法进行了适配，并在篡改分类和定位任务上进行了基准测试。最后，我们系统地审计了自我改进智能体的真实轨迹。结果表明……

    arXiv:2609.00069v1 Announce Type: cross  Abstract: Self-improving agents iteratively modify their own harness to push the frontier of their performance. However, such modifications can produce illusory performance gains or compromise integrity constraints such as authorization, provenance, and completeness without genuinely improving capability. We term this phenomenon as harness tampering, which extends the concept from reward and measurement tampering to the full self-improvement lifecycle. To systematically study this problem, we propose a two-axis taxonomy that categorizes each misaligned edit by the harness functional role in which it occurs and the obligation it violates. Then we build an annotated corpus by seeding tampered-benign edit pairs into the real trajectories of self-improving agents. We adapt and benchmark diverse audit methods on tampering classification and localization tasks. Finally we systematically audit real trajectories of self-improving agents. The results dem
    
[^279]: 生命算子：一种用于多尺度生命建模的自演化框架

    Life Operators: a self-evolving framework for multiscale life modelling

    [https://arxiv.org/abs/2609.00068](https://arxiv.org/abs/2609.00068)

    该论文提出“生命算子”自演化框架，通过感知、演化、生成三类任务约束映射算子及桥接算子，为多尺度生命建模提供了统一框架，能够表示患者状态、耦合不同尺度并支持对失效假设的修正。

    

    医疗人工智能正从识别任务走向临床对话与纵向预测。然而一个核心问题仍然悬而未决：患者的状态在干预之下会如何变化？统计模型学习对未来的观测，而机理模型描述的是被选取的特定过程，二者都无法提供一个统一框架来表示患者状态、耦合不同尺度或修正失效的假设。我们提出生命算子：这是一类具有任务边界的映射，定义了三种科学角色。感知算子从多模态观测中推断与任务相关的生物状态；演化算子在自然动力学或干预条件动力学下传播这些状态；生成算子将这些状态映射为可测量的信号。每种角色都可以由方程、统计模型、神经网络或其混合形式来实现。桥接算子负责连接具有不同变量、尺度和时间步长的组件。所选定的算子与桥接算子共同构成面向特定任务的……

    arXiv:2609.00068v1 Announce Type: cross  Abstract: Medical AI is moving beyond recognition towards clinical dialogue and longitudinal prediction. Yet a central question remains: how would a patient's state change under intervention? Statistical models learn future observations, whereas mechanistic models describe selected processes. Neither provides a common framework for representing patient state, coupling scales or revising failed assumptions. We propose Life Operators: task-bounded mappings that define three scientific roles. Perception operators infer task-relevant biological states from multimodal observations, Evolution operators propagate these states under natural or intervention-conditioned dynamics, and Generation operators map them to measurable signals. Each role may be realised by equations, statistical models, neural networks or hybrids. Bridge operators connect components with different variables, scales and time steps. Selected operators and bridges form task-specific 
    
[^280]: 多模态大语言模型是先看后读吗？诊断情境性谄媚现象

    Do Multimodal LLMs See Before They Read? Diagnosing Contextual Sycophancy

    [https://arxiv.org/abs/2609.00067](https://arxiv.org/abs/2609.00067)

    该论文诊断了多模态大语言模型易受外部文本误导而忽视冲突图像证据的“多模态情境性谄媚”问题，并提出“系统2视觉仲裁”（S2VA）方法，通过让视觉证人在读取文本前先独立判断，在六个模型上将准确率显著提升19.7至44.1分。

    

    外部文本可以覆盖多模态大语言模型中与之冲突的图像证据，我们将这种失败称为“多模态情境性谄媚”。我们引入了一个包含998个案例的诊断方法，该方法独立地变化视觉证据、常识先验和外部文本三个因素，并通过围绕“情境盲视”的视觉证人调整信息边界，来探究这种失败在何时发生。在与Gemini生成的虚假文本配对的异常图像上，GPT-5.1在联合条件下的得分仅为7.9%；当直接对情境盲视的证人报告进行评分时，得分为49.7%；在使用匹配的双调用证人-仲裁者管道（即让证人接触文本）时，得分为63.7%；而在“系统2视觉仲裁”（S2VA，即对证人隐瞒文本）下，得分达到84.2%。在六个模型上，S2VA相比直接证人报告提升了19.7至44.1分，且所有配对的95%置信区间均不包含零。最佳的信息边界并非统一不变：文本情境对某些情况……

    arXiv:2609.00067v1 Announce Type: cross  Abstract: External text can override conflicting image evidence in multimodal large language models, a failure we call multimodal contextual sycophancy. We introduce a 998-case diagnostic that independently varies visual evidence, commonsense priors, and external text, and probe when this failure arises by moving the information boundary around a context-blind visual witness. On abnormal images paired with Gemini-generated false text, GPT-5.1 scores 7.9% under joint conditioning, 49.7% when the context-blind witness report is scored directly, 63.7% under a matched two-call witness-arbiter pipeline that exposes the witness to the text, and 84.2% under System-2 Visual Arbitration (S2VA), which withholds the text from the witness. Across six models, S2VA improves over the direct witness report by 19.7 to 44.1 points, with all paired 95% confidence intervals excluding zero. The best information boundary is not uniform: textual context scaffolds some
    
[^281]: OCGQuant：面向NVFP4量化的异常值伴随分组方法

    OCGQuant: Outlier-Companion Grouping for NVFP4 Quantization

    [https://arxiv.org/abs/2609.00066](https://arxiv.org/abs/2609.00066)

    提出OCGQuant，一种以“异常值伴随分组（OCG）”为核心的NVFP4训练后量化方法，通过自适应地将异常值通道与伴随通道分组，减少由块最大值主导缩放因子所造成的“附带量化误差”，从而在不引入额外计算的前提下提升低比特推理的量化精度。

    

    NVFP4是一种面向低比特推理的高效微缩放（microscaling）格式，但激活异常值仍会降低NVFP4块内的量化精度。在每个量化块内，较大的激活值会主导块缩放因子，从而增大共享同一缩放因子的其余数值的量化误差。现有的训练后量化（PTQ）方法通过混合精度、旋转或残差补偿等策略来缓解异常值带来的误差，但这些方法要么并非专门针对NVFP4设计，要么会引入额外的计算开销。在本工作中，我们从通道分组的视角重新审视NVFP4，并将由块最大值所设定的缩放因子下其余块内数值产生的可减少误差定义为“附带量化误差”。基于这一洞察，我们提出了OCGQuant——一种以异常值伴随分组（Outlier-Companion Grouping, OCG）为核心的训练后量化方法，该方法自适应地将异常值通道与……（原文摘要在此处截断）

    arXiv:2609.00066v1 Announce Type: cross  Abstract: NVFP4 is an efficient microscaling format for low-bit inference, but activation outliers can still degrade quantization accuracy within NVFP4 blocks. Within each quantization block, large activations can dominate the block scale, increasing the quantization error of the remaining values sharing the same scale. Existing post-training quantization (PTQ) methods mitigate outlier errors through strategies such as mixed precision, rotation, or residual compensation, but these approaches are either not specifically tailored to NVFP4 or introduce additional computation. In this work, we revisit NVFP4 from a channel-grouping perspective and define the reducible error incurred by remaining block values under the scale set by the block maximum as Collateral Quantization Error. Based on this insight, we propose OCGQuant, a post-training quantization method centered on Outlier-Companion Grouping (OCG), which adaptively pairs outlier channels with 
    
[^282]: 科学智能体技能：面向科研智能体的程序性知识库

    Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents

    [https://arxiv.org/abs/2609.00065](https://arxiv.org/abs/2609.00065)

    该论文提出了一个名为“科学智能体技能”的开放库，收录了基因组学、化学信息学等16个科研实践领域共163项程序性知识，使语言模型智能体能够遵循领域规范做出站得住脚的科学分析，而非仅仅返回能运行的代码。

    

    被要求分析实验的语言模型智能体通常只会返回一段能运行的代码，但该分析是否站得住脚则是另一回事。一个站得住脚的分析取决于程序性选择：该领域接受哪种统计检验方法、哪个标识符命名空间是权威的、以及结果必须附带哪些注意事项。我们提出了“科学智能体技能”，这是一个开放的知识库，包含16个实践领域的163项此类程序，涵盖基因组学、化学信息学、医学影像、研究设计和科学传播等。每项技能都是一个目录，围绕一个版本化、人类可读的指令文件构建。智能体仅在任务需要时才加载该文件；目录中通常还包含参考资料和可运行的脚本。我们未报告任务级评估结果和宿主选择率。该库采用开放许可证，可在 https://github.com/K-Dense-AI/scientific-agent-skills 获取。

    arXiv:2609.00065v1 Announce Type: cross  Abstract: A language-model agent asked to analyse an experiment will usually return working code. Whether the analysis is defensible is a different question. A defensible analysis depends on procedural choices: which test the field accepts, which identifier namespace is authoritative, and which caveats must accompany a result. We present Scientific Agent Skills, an open library of 163 such procedures in 16 areas of practice, including genomics, cheminformatics, medical imaging, study design and scientific communication. Each skill is a directory built around a versioned, human-readable instruction file. An agent loads the file only when a task calls for it; the directory often also contains reference material and runnable scripts. We report no task-level evaluation and no host selection rate. Openly licensed and available at https://github.com/K-Dense-AI/scientific-agent-skills.
    
[^283]: 注意力敏感性并不足够：在微调下解耦注意力层面与行为层面的上下文学习

    Attention Sensitivity Is Not Enough: Dissociating Attention-Level and Behavioural In-Context Learning under Fine-Tuning

    [https://arxiv.org/abs/2609.00064](https://arxiv.org/abs/2609.00064)

    该论文形式化了注意力层面的“上下文敏感性”（ICS）指标，并通过Llama-2-7B上的四臂消融实验证明，最大化ICS并不能保留真实的行为性上下文学习能力（ICL-GAP接近零且MMLU从0.371降至0.279），揭示了注意力代理指标与行为层面ICL之间的“古德哈特定律”式解耦。

    

    上下文学习（ICL）使大型语言模型能够通过示例适应新任务，而微调可能会削弱这种行为。许多保持性诊断方法依赖检查注意力：如果注意力随示例的变化而变化，模型就被视为对上下文敏感。本文探讨这种代理指标在被优化之后能在多大程度上被信任。我们形式化了“上下文敏感性”（ICS），即在匹配与不匹配示例前缀上最后一个token注意力分布之间的平均行距离，并将其与“ICL差距”（ICL-GAP）配对，后者衡量相同前缀之间的行为准确率差距。在Llama-2-7B上进行的受控四臂消融实验中，一个最大化ICS的正则化器（armKL）将ICS推高至1.413，达到其几何上限的0.5%以内。然而行为层面的读数讲述了不同的故事：ICL-GAP保持在接近零的水平，MMLU准确率从0.371下降至0.279，这是有界注意力代理指标的“古德哈特式”解耦。端点统计定位……

    arXiv:2609.00064v1 Announce Type: cross  Abstract: In-context learning (ICL) lets large language models adapt to new tasks from demonstrations, and fine-tuning can erode this behaviour. Many preservation diagnostics inspect attention: if attention changes when demonstrations change, the model is treated as context-sensitive. This paper asks how far that proxy can be trusted once it is optimised. We formalise \emph{In-Context Sensitivity} (ICS), the average row distance between last-token attention on matched and mismatched demonstration prefixes, and pair it with \emph{ICL-GAP}, the behavioural accuracy gap between the same prefixes. In a controlled four-arm ablation on Llama-2-7B, an ICS-maximising regulariser ($\armKL$) drives ICS to $1.413$, within $0.5\%$ of its geometric ceiling. The behavioural readout tells a different story: ICL-GAP stays near zero and MMLU accuracy moves from $0.371$ to $0.279$, a Goodhart dissociation of the bounded attention proxy. Endpoint statistics locate
    
[^284]: 基于大语言模型的医学因果假设验证

    Medical Causal Hypothesis Verification with Large Language Models

    [https://arxiv.org/abs/2609.00063](https://arxiv.org/abs/2609.00063)

    本文提出了一个医学因果假设验证的评估框架，并评估了八个大语言模型利用科学文献证据验证17个医学因果假设的能力。

    

    大语言模型在搜索和信息检索中的应用日益增多，这凸显了评估其在医疗保健等高风险领域可靠性的必要性。尽管大语言模型能够有效回答关于疾病、症状和治疗的问题，但其准确评估因果关系并将结论建立在经过验证的科学证据之上的能力仍不明确。本文提出了一项初步的小规模研究，调查了大语言模型在评估医学因果性论断并用同行评审研究加以支持方面的准确性。我们提出了一个因果假设验证的评估框架，可用于系统地跟踪现有和未来大语言模型的表现。我们评估了八个大语言模型在17个医学因果假设上的表现，以检验它们能否利用文献中的科学证据可靠地验证这些假设。我们对科学文献进行了系统性标注……

    arXiv:2609.00063v1 Announce Type: cross  Abstract: The growing use of large language models (LLMs) for search and information retrieval underscores the need to evaluate their reliability in high-stakes domains such as healthcare. Although LLMs can effectively answer questions about diseases, symptoms, and treatments, their ability to accurately assess causal relationships and ground their conclusions in verified scientific evidence remains unclear. Here, we present a preliminary, small-scale study that investigates the accuracy of LLMs in evaluating causal medical claims and supporting them with peer-reviewed research. We propose an evaluation framework for causal hypothesis verification that can be used to systematically track the performance of existing and future LLMs. We assess the performance of eight LLMs on 17 medical causal hypotheses to evaluate whether they can reliably verify these hypotheses using scientific evidence from the literature. We systematically annotate the scien
    
[^285]: RePro：面向大语言模型数学问题求解可靠评估的证明验证基准改写方法

    RePro: Proof-Verified Benchmark Rewriting for Reliable Evaluation of LLM Mathematical Problem Solving

    [https://arxiv.org/abs/2609.00062](https://arxiv.org/abs/2609.00062)

    RePro首次将Lean自动定理证明器集成到数学基准改写中，通过形式化证明保证改写题目的有效性与答案正确性，并发现多个大语言模型在验证后的改写基准上准确率下降，暴露了其依赖记忆化而非真正推理能力的问题。

    

    数据污染破坏了大语言模型（LLM）在数学问题求解任务上评估的可靠性。虽然基于改写的评估方法可以缓解模型记忆化问题，但现有方法缺乏对问题有效性和答案正确性的保证。我们提出了证明验证基准改写框架RePro，这是首个将面向Lean的神经自动定理证明器（ATP）集成到基准改写中的框架，在改写问题并重新生成答案的同时，通过Lean验证的证明确保其正确性。在GSM8K和MATH数据集上的实验表明，RePro保留的改写实例达到了100%的良好定义性、可解性和答案正确性，而现有方法仍会产生无效或不正确的实例。此外，多个模型在经过证明验证的改写基准上出现了准确率下降，这表明它们的性能对表层和结构变化较为敏感，可能部分反映了记忆化效应。

    arXiv:2609.00062v1 Announce Type: cross  Abstract: Data contamination undermines the reliable evaluation of large language models (LLMs) on mathematical problem solving. While rewriting-based evaluation mitigates memorization, existing methods lack guarantees of problem validity and answer correctness. We propose Proof-Verified Benchmark Rewriting (RePro), the first framework to integrate Lean-oriented neural automated theorem provers (ATPs) into benchmark rewriting, which rewrites problems and regenerates answers with correctness ensured by Lean-verified proofs. Experiments on GSM8K and MATH show that RePro's retained rewritten instances achieve 100% well-definedness, feasibility, and answer correctness, while existing methods still produce invalid or incorrect instances. Moreover, several models exhibit accuracy drops on proof-verified rewritten benchmarks, suggesting that their performance is sensitive to surface-level and structural variations and may partly reflect memorization ef
    
[^286]: ReNFT：通过内部概率质量重新校准修复奖励后训练中的模式坍塌

    ReNFT: Repairing Mode Collapse in Reward Post-Training via Internal Probability-Mass Recalibration

    [https://arxiv.org/abs/2609.00061](https://arxiv.org/abs/2609.00061)

    本文提出ReNFT方法，通过内部概率质量重新校准修复扩散模型奖励后训练中出现的模式坍塌，在保留已获得奖励的同时恢复提示内多样性，无需依赖任何外部信号或接口。

    

    扩散生成器的奖励后训练不可避免地将概率质量集中在少数受奖励偏好的模式上，这种模式坍塌消除了提示内的多样性。现有的缓解坍塌方法依赖于外部信号或接口，例如用感知目标增强奖励、调整参考正则化或修改文本编码器，但没有一种方法能够在保留已获得奖励的同时修复已经坍塌的适配器。我们观察到，在线后训练主要是在预训练继承的能力之上重新分配概率质量，而不是学习新的视觉内容。因此，坍塌是抑制而非删除，可以从生成器内部逆转。我们提出ReNFT，通过内部概率质量重新校准来修复高奖励、低多样性的适配器。无条件探测首先优先处理“反中心”提示，在这些提示中，与提示无关的偏差最容易暴露。

    arXiv:2609.00061v1 Announce Type: cross  Abstract: Reward post-training of diffusion generators inevitably concentrates probability mass on a few reward-favored modes, a mode collapse that erases within-prompt diversity. Existing methods for mitigating collapse rely on external signals or interfaces, augmenting the reward with perceptual objectives, adjusting reference regularization, or modifying the text encoder, but none repairs an adapter that has already collapsed while preserving the acquired reward. We observe that online post-training primarily reallocates probability mass over capabilities inherited from pretraining rather than learning new visual content. Collapse is therefore suppression, not deletion, and can be reversed from within the generator. We propose ReNFT, which repairs a high-reward, low-diversity adapter through internal probability-mass recalibration. Unconditional probes first prioritize "anti-hub" prompts where the prompt-independent bias is easiest to expose.
    
[^287]: 智能体支付协议的形式化分析

    A Formal Analysis of Agent Payment Protocols

    [https://arxiv.org/abs/2609.00060](https://arxiv.org/abs/2609.00060)

    该论文首次在Tamarin形式化验证工具中对x402、MPP、ACP和AP2四种代表性智能体支付协议进行系统性形式化分析，通过统一的生命周期抽象构建基于源代码的模型，捕捉各协议的角色、状态、信任假设与生命周期转换，填补了智能体支付安全保障长期缺乏系统形式化分析的空白。

    

    智能体支付协议正在成为自主商务的关键交易层，使AI智能体能够购买商品和服务，并代表用户执行支付。与传统支付流程不同，它们将用户意图、委托授权、凭证使用、结算和履约分散在多个参与方和阶段之中，形成了任何单一消息或参与者都无法独立保障的安全依赖关系。然而，在不断演进的规范、模式和参考实现中，这些安全保障大多仍是隐式的，缺乏系统性的形式化分析。我们在Tamarin中对四种代表性的智能体支付协议——x402、MPP、ACP和AP2——进行了形式化。基于对智能体支付生命周期的统一抽象，我们构建了以源代码为基础的模型，捕捉每个协议的角色、状态、信任假设和生命周期转换。我们不预设完整的安全属性分类，而是采用基于源代码的验证查询……（摘要在此处被截断）

    arXiv:2609.00060v1 Announce Type: cross  Abstract: Agent payment protocols are emerging as a key transaction layer for autonomous commerce, enabling AI agents to purchase goods and services and execute payments on users' behalf. Unlike conventional payment flows, they distribute user intent, delegated authority, credential use, settlement, and fulfillment across multiple actors and stages, creating security dependencies that no single message or participant can enforce. Yet these guarantees remain largely implicit across evolving specifications, schemas, and reference implementations, with little systematic formal analysis.   We formalize four representative agent payment protocols: x402, MPP, ACP, and AP2 in Tamarin. Using a common abstraction of the agent payment lifecycle, we construct source-grounded models that capture each protocol's roles, state, trust assumptions, and lifecycle transitions. Rather than assuming a complete property taxonomy, we use source-backed verification que
    
[^288]: DISTAL：面向结构无关材料性质预测的蒸馏与自监督预训练

    DISTAL: Distillation and Self-Supervised Pretraining for Structure-Agnostic Materials Property Prediction

    [https://arxiv.org/abs/2609.00059](https://arxiv.org/abs/2609.00059)

    DISTAL提出了一种双先验框架，通过自监督成分预训练和从ALIGNN教师模型进行结构知识蒸馏，实现了推理时无需晶体结构输入的结构无关材料性质预测，适用于低数据、结构信息缺失的早期筛选场景。

    

    在低数据环境下，材料性质预测仍然十分困难，因为许多目标性质仅有数量有限的标注样本支持。预测精度最高的模型通常依赖于晶体结构，这在结构信息有限或不可获得的情况下限制了其在早期筛选中的应用。为应对这一挑战，我们提出了DISTAL，一个用于结构无关材料性质预测的双先验框架，它将自监督成分预训练与结构感知的知识蒸馏相结合。DISTAL首先利用145个基于成分的描述符，从大型虚拟成分空间中学习可迁移的成分表示。随后，它将预训练的ALIGNN教师模型中的结构知识蒸馏到一个以成分为条件的学生模型中。这种设置使得结构先验可以在训练阶段被利用，而在推理阶段无需结构输入。通过整合……（原文摘要在此截断）

    arXiv:2609.00059v1 Announce Type: cross  Abstract: Materials property prediction remains difficult in low-data settings, where many target properties are supported by only a limited number of labeled samples. Models with the strongest predictive accuracy often depend on crystal structures, which restricts their use in early-stage screening when structural information is limited or unavailable. To address this challenge, we propose DISTAL, a dual-prior framework for structure-agnostic materials property prediction that combines self-supervised compositional pretraining with structure-aware knowledge distillation. DISTAL first learns transferable compositional representations from a large virtual composition space using 145 composition-derived descriptors. It then distills structural knowledge from a pretrained ALIGNN teacher into a composition-conditioned student. This setting allows structural priors to be used during training without requiring structural inputs at inference. By integr
    
[^289]: CUDA-Harness：从自然语言驱动的智能体式CUDA内核生成与优化

    CUDA-Harness: Harnessing Agentic CUDA Kernel Generation and Optimization from Natural Language

    [https://arxiv.org/abs/2609.00058](https://arxiv.org/abs/2609.00058)

    该论文提出CUDA-Harness框架，通过智能体式方法直接从自然语言生成并优化高性能CUDA内核，克服了现有工作局限于PyTorch转译以及因依赖预定义测试输入而易受奖励欺骗的不足。

    

    开发高性能CUDA内核需要掌握算法实现、正确性验证以及面向硬件的并行优化等专业知识，这构成了很高的专业门槛，因此直接从自然语言生成CUDA内核变得至关重要。与此同时，大语言模型（LLM）通用的代码生成能力催生了一系列基于LLM的CUDA内核生成研究。这些工作主要聚焦于从PyTorch等高级框架向CUDA的转译（Torch2CUDA），而非Text2CUDA——后者要求模型既要理解高层输入语义，又要处理底层的内核实现与验证。此外，由于依赖预定义的测试输入，这些方法容易受到奖励欺骗的影响。在本文中，我们提出了CUDA-Harness，一个用于从自然语言驱动智能体式CUDA内核生成与优化的框架。

    arXiv:2609.00058v1 Announce Type: cross  Abstract: Developing high-performance CUDA kernels demands specialized knowledge in algorithm implementation, correctness validation, and hardware-aware parallel optimization, creating a substantial expertise barrier and making generating CUDA kernels directly from natural language (Text2CUDA) essential. Meanwhile, the general-purpose code generation capability of Large Language Models (LLMs) prompts a series of works exploring LLM-based CUDA kernel generation. They mainly focus on transpilation from high-level frameworks such as PyTorch to CUDA (Torch2CUDA) rather than Text2CUDA, where models must understand the high-level input semantics and handle low-level kernel implementation and validation. Additionally, these methods are vulnerable to reward hacking due to reliance on predefined test inputs. In this paper, we propose CUDA-Harness, a framework for harnessing agentic CUDA kernel generation and optimization from natural language. Specifical
    
[^290]: ValueGraph：价值信号引导的图预训练方法用于上下文化用户表示

    ValueGraph: Value-Signal Guided Graph Pre-training for Contextualized User Representation

    [https://arxiv.org/abs/2609.00057](https://arxiv.org/abs/2609.00057)

    提出ValueGraph图预训练框架，将自动推断的道德价值信号作为软约束辅助信号，结合对比学习与聚类目标学习上下文化的用户表示，在立场检测和推特机器人检测任务上取得提升。

    

    价值信号是一种聚合的用户级道德表征，能够从用户的在线言论中捕捉其被推断出的与价值观相关的倾向。社交媒体上的用户行为不仅受用户说什么或与谁互动的影响，还受到用户表达态度时所依托的价值信号的影响。然而，现有的用户表示方法大多忽略了这一与价值相关的维度。我们提出ValueGraph，一个图预训练框架，它将自动推断的道德价值信号作为含噪的辅助信号，用于学习上下文化的用户表示。ValueGraph从帖子-回复图中学习语义和结构表征，并通过对比学习和聚类目标，基于相对价值相似度进一步对齐用户。ValueGraph并不把推断出的价值观当作标准的心理学标签，而是将其用作表示学习的软约束。在立场检测和推特机器人检测任务上的实验表明……

    arXiv:2609.00057v1 Announce Type: cross  Abstract: Value signals are aggregated user-level moral representations that capture users' inferred value-related tendencies from their online discourse. User behavior on social media is shaped not only by what users say or whom they interact with, but also by the value signal through which they express attitudes. Existing user representation methods largely miss this value-relevant dimension. We propose ValueGraph, a graph pre-training framework that uses automatically inferred moral-value signals as noisy auxiliary signals for contextualized user representation. From post-reply graphs, ValueGraph learns semantic and structural representations and further aligns users through relative value similarity with contrastive and clustering objectives. Rather than treating inferred values as gold psychological labels, ValueGraph uses them as soft constraints for representation learning. Experiments on stance detection and twitter bot detection show co
    
[^291]: 通过大语言模型增强的音频-文本对齐实现零样本呼吸音分类

    Zero-Shot Respiratory Sound Classification through LLM-Augmented Audio-Text Alignment

    [https://arxiv.org/abs/2609.00055](https://arxiv.org/abs/2609.00055)

    该论文提出利用医学大语言模型从元数据合成结构化报告，将自监督呼吸音编码器与医学术语在共享潜在空间中对齐，实现61.3%平均零样本AUC，以更少数据超越CLAP和Qwen2-Audio等大规模基线模型。

    

    自监督呼吸音编码器缺乏零样本推理所需的临床领域语义基础，在没有任务特定标注数据的情况下限制了其实用性。我们提出了一个框架，将这些编码器与医学术语在共享潜在空间中对齐，使其转变为具备零样本能力的基础模型。为解决配对数据稀缺问题，我们使用医学大语言模型从元数据合成结构化报告，为对比学习创建密集的语义锚点。我们的训练方法将基于sigmoid的对比损失与编码器原生的自监督学习目标相结合，并采用相似度感知的负样本采样来锐化病理边界。在6个数据集的9项任务上，我们的方法实现了61.3%的平均零样本AUC，超过了CLAP（51.4%）和Qwen2-Audio（54.9%），同时仅使用全规模基线模型43%的数据就达到了最高的线性探测AUC（71.6%），表明结构化语义对齐优于大规模方法。

    arXiv:2609.00055v1 Announce Type: cross  Abstract: Self-supervised respiratory encoders lack semantic grounding in clinical domain needed for zero-shot inference, limiting their utility without task-specific labeled data. We propose a framework that aligns these encoders with medical terminology in a shared latent space turning them into a zero-shot-capable foundation model. To address paired data scarcity, we use a medical LLM to synthesize structured reports from metadata, creating dense semantic anchors for contrastive learning. Our training combines a sigmoid-based contrastive loss with encoder's native SSL objective and similarity-aware negative sampling to sharpen pathological boundaries. Across 9 tasks on 6 datasets, our method achieves a 61.3% mean zero-shot AUC, surpassing CLAP (51.4%) and Qwen2-Audio (54.9%) while reaching the highest linear probing AUC (71.6%) with only 43% of data used by full-scale baselines, showing that structured semantic alignment outperforms large-sca
    
[^292]: 从检测到拒答：通过电路引导的权重缩放实现更安全的大语言模型

    From Detection to Refusal: Safer LLMs via Circuit-Guided Weight Scaling

    [https://arxiv.org/abs/2609.00051](https://arxiv.org/abs/2609.00051)

    该论文从机制可解释性角度首次刻画了大语言模型中由有害检测头、安全神经元和拒答头组成的多阶段安全电路，通过因果干预实验验证了这一电路组织，并据此提出利用电路引导的权重缩放方法构建更安全的大语言模型。

    

    尽管已经进行了大量的对齐工作，大语言模型（LLMs）在对抗性提示下仍然容易生成不安全的内容，然而安全行为得以实现的内部机制仍然鲜为人知。我们从机制可解释性的视角研究大语言模型的安全性，并刻画了一个组织拒答行为的多阶段*安全电路*，该电路由以下部分组成：(i) 对有害输入作出响应的**有害检测头**，(ii) 在残差流中介导并稳定安全信号的**安全神经元**，以及 (iii) 将这些信号转化为安全响应生成的**拒答头**。通过有针对性的注意力头层面和神经元层面的干预，我们提供了与该电路组织结构相一致的因果证据，表明抑制上游的有害检测头会破坏下游的拒答行为，并且安全神经元介导了这种相互作用。我们验证（原文摘要在此处截断）

    arXiv:2609.00051v1 Announce Type: cross  Abstract: Despite extensive alignment efforts, Large Language Models (LLMs) remain vulnerable to generating unsafe content under adversarial prompting, yet the internal mechanisms by which safety behaviors are implemented remain poorly understood. We study LLM safety from a mechanistic interpretability perspective and characterize a multi-stage *safety circuit* that organizes refusal behavior, consisting of (i) $\textbf{Harmful Detection Heads}$ that respond to harmful inputs, (ii) $\textbf{Safety Neurons}$ that mediate and stabilize safety signals in the residual stream, and (iii) $\textbf{Refusal Heads}$ that translate these signals into safe response generation. Using targeted attention-head and neuron-level interventions, we provide causal evidence consistent with this circuit organization, showing that suppressing upstream Harmful Detection Heads disrupts downstream refusal behavior and that safety neurons mediate this interaction. We valid
    
[^293]: 迈向智能体化云工程：基于零信任智能体套件的图工程与循环工程

    Towards Agentic Cloud Engineering: Graph and Loop Engineering with a Zero-Trust Agent Harness

    [https://arxiv.org/abs/2609.00050](https://arxiv.org/abs/2609.00050)

    提出了一个智能体云工作流工程框架，通过将图工程（长时程工作流推进）、循环工程（有界诊断与修复重试）和零信任智能体套件（受限执行）三个关注点分离，将自然语言云工程任务自动转化为经过验证的代码仓库和可验证的云部署。

    

    智能体AI正在推动基于云的工作流的发展，其中自主智能体可以对运营状态进行推理、调用授权工具、修改软件和基础设施、部署服务、验证执行结果，并在长时程、多步骤任务中进行自适应调整。构建此类工作流需要针对工作流推进、受限执行、故障恢复和可验证完成等环节的显式机制。我们提出了智能体云工作流工程，这是一个智能体AI框架，它将自然语言描述的智能体云工程任务转化为经过验证的代码仓库和经过验证的运营性云部署，从而实现基于云的智能体工作流自动化。该框架分离了三个互补的关注点：图工程负责指定长时程工作流推进以及依赖验证的状态转移；循环工程提供有界的诊断、修复或重新规划、重试和重新验证；智能体套件工程则（负责执行零信任的受限执行控制）。

    arXiv:2609.00050v1 Announce Type: cross  Abstract: Agentic AI is enabling cloud-based workflows in which autonomous agents reason over operational state, invoke authorized tools, modify software and infrastructure, deploy services, verify execution outcomes, and adapt across long-horizon, multistep tasks. Engineering such workflows requires explicit mechanisms for workflow progression, constrained execution, failure recovery, and verifiable completion. We present Agentic Cloud Workflow Engineering, an agentic AI framework that transforms natural-language agentic cloud-engineering tasks into validated code repositories and verified operational cloud deployments for automating cloud-based agentic workflows. The framework separates three complementary concerns: graph engineering specifies long-horizon workflow progression and verification-dependent transitions; loop engineering provides bounded diagnosis, repair or re-planning, retry, and re-verification; and agent harness engineering enf
    
[^294]: REAL-Q：基于动态梯度下降的大语言模型端到端量化

    REAL-Q: E2E LLM Quantization via Dynamic Gradient Descent

    [https://arxiv.org/abs/2609.00049](https://arxiv.org/abs/2609.00049)

    REAL-Q提出了一种打破传统折中的后训练量化新范式，通过端到端对齐的代理损失目标和每128列一次的动态块级梯度下降，解决了现有方法中Hessian矩阵被整层冻结导致的信息错位问题，从而更精确地逼近全局损失实现大语言模型量化。

    

    后训练量化（PTQ）是在严格资源约束下部署大语言模型（LLM）的关键技术。当前最先进的PTQ方法使用单一的闭式二阶求解器对每一层进行量化：为了保持解析上的可处理性，这些方法对全局损失进行了大量近似（舍弃跨通道耦合、将输出行池化为组），随后在整个层内冻结所得的Hessian矩阵，无法随着损失景观逐列变化而对其进行更新——我们将这种现象称为信息错位（information misalignment）。我们提出REAL-Q（Real-time E2E-loss Aligned LLM Quantization，实时端到端损失对齐的大语言模型量化），这是一种新颖的PTQ范式，打破了这一折中：REAL-Q不再为了解析可处理性而稀释目标函数，而是针对全局损失的端到端对齐代理目标，并在每处理一个列块（128列）后应用细粒度、动态的块级梯度下降对其进行优化。通过耦合这种细粒度……

    arXiv:2609.00049v1 Announce Type: cross  Abstract: Post-training quantization (PTQ) is essential for deploying large language models (LLMs) under strict resource constraints. State-of-the-art PTQ methods quantize each layer with a single closed-form second-order solver: to remain analytically tractable, they heavily approximate the global loss (dropping cross-channel coupling, pooling output rows into groups), and they then freeze the resulting Hessian across the entire layer, with no way to refresh it as the loss landscape shifts column by column--a phenomenon we call information misalignment. We propose REAL-Q (Real-time E2E-loss Aligned LLM Quantization), a novel PTQ paradigm that breaks this compromise: instead of diluting the objective for the sake of analytic tractability, REAL-Q targets an end-to-end-aligned surrogate of the global loss and refines it via fine-grained, dynamic Block-wise Gradient Descent applied after every column block (128 columns). By coupling this fine-grain
    
[^295]: GUI-CC：面向智能体环境的GUI世界模型上下文一致性基准测试

    GUI-CC: Benchmarking Contextual Consistency of GUI World Models as Agent Environments

    [https://arxiv.org/abs/2609.00048](https://arxiv.org/abs/2609.00048)

    提出GUI-CC基准，通过离线真实轨迹滚动和在线智能体交互循环两条互补轨道，评估GUI世界模型在多步智能体环境中反复复用生成状态时的上下文一致性。

    

    GUI世界模型目前越来越多地被评估为单步的下一屏幕预测器，然而它们的实际用途往往是作为GUI智能体的多步交互环境。这种错配导致一个关键需求未被充分测试：生成的状态在被反复复用于未来交互时，必须保持上下文一致性。我们提出了GUI-CC，这是一个评估GUI世界模型作为智能体环境（而非孤立的下一屏幕预测器）的上下文一致性的基准。GUI-CC包含两条互补的评估轨道：离线参考动作轨道，让模型沿真实的移动GUI轨迹进行滚动；以及在线智能体循环轨道，让固定的探测智能体与模型生成的UI进行交互。我们从GUIOdyssey构建了500个离线轨迹任务，并在30个移动应用中构建了200个经模拟器验证的在线任务。GUI-CC评估转移保真度、转移合理性、上下文一致性以及任务进展。实验表明……（摘要在此处截断）

    arXiv:2609.00048v1 Announce Type: cross  Abstract: GUI world models are increasingly evaluated as one-step next-screen predictors, yet their intended use is often as multi-step environments for GUI agents. This mismatch leaves a key requirement under-tested: generated states must remain contextually consistent when they are repeatedly reused for future interaction. We introduce GUI-CC, a benchmark that evaluates contextual consistency of GUI world models as agent environments rather than isolated next-screen predictors. GUI-CC contains two complementary tracks: an offline reference-action track that rolls models along real mobile GUI trajectories, and an online agent-loop track that lets fixed probing agents interact with model-generated UIs. We construct 500 offline trajectory tasks from GUIOdyssey and 200 emulator-verified online tasks across 30 mobile apps. GUI-CC evaluates transition fidelity, transition plausibility, contextual consistency, and task progress. Experiments show that
    
[^296]: 面向多任务图预训练的融合全局上下文的任务特定提示

    Task-Specific Prompt with Global Context for Multi-Task Graph Pre-Training

    [https://arxiv.org/abs/2609.00047](https://arxiv.org/abs/2609.00047)

    提出TPGC双先验提示初始化方法，通过显式建模任务先验与结构先验的协同作用，解决多任务图预训练中随机初始化提示导致的任务相关性弱、结构感知差和可迁移性不足的问题。

    

    图提示学习是一种在低资源场景下将预训练图模型适配到下游任务的有效范式。然而，现有多任务图预训练框架通常使用随机初始化的提示，导致提示空间、前置任务目标与图结构特征之间的对齐不佳，这极大地削弱了提示表示的任务相关性、结构感知能力和可迁移性。为了应对这一挑战，我们提出了TPGC，一种双先验提示初始化解决方案，它显式地建模任务先验与结构先验之间的协同作用。具体而言，任务先验注入模块首先在辅助图上进行短暂的同源多任务预训练，使提示初始化能够继承与多个前置任务相关的优化偏好。在任务感知表示的基础上，结构先验注入模块进一步提取可迁移的结构信息以增强提示的全局上下文感知能力。

    arXiv:2609.00047v1 Announce Type: cross  Abstract: Graph prompt learning is an effective paradigm to adapt pre-trained graph models to downstream tasks in low-resource scenarios. However, existing multi-task graph pre-training frameworks generally use randomly initialized prompts, leading to poor alignment between the prompt space, pretext objectives and graph structural characteristics. This greatly weakens the task relevance, structural awareness and transferability of prompt representations. To address this challenge, we propose TPGC, a dual-prior prompt initialization solution that explicitly models the synergy between task prior and structural prior. Specifically, the Task-Prior Injection Module first conducts a short homologous multi-task pre-training on an auxiliary graph, enabling prompt initialization to inherit optimization preferences associated with multiple pretext tasks. Built on the task-aware representations, the Structure-Prior Injection Module further extracts transfe
    
[^297]: RAPIDMap：基于卫星和街景影像的可解释灾害制图快速多智能体流水线

    RAPIDMap: Rapid Multi-Agent Pipeline for Interpretable Disaster Mapping from Satellite and Street-view Imagery

    [https://arxiv.org/abs/2609.00046](https://arxiv.org/abs/2609.00046)

    RAPIDMap提出了一种由灾害感知、图像修复、损毁识别和灾害制图四个智能体组成的快速多智能体流水线，结合卫星与街景影像实现零样本、可解释的灾害制图，无需人工微调即可跨多种灾害类型生成结构化灾害情报与恢复建议。

    

    对受灾区域、受损基础设施和受灾人口进行快速、可靠的灾害制图对于应急响应和灾后恢复至关重要。然而，现有的基于AI的方法通常需要大量人工标注，缺乏跨灾害类型的泛化能力，且依赖于单一模态的观测数据。为应对这些挑战，本文提出了RAPIDMap，一个基于卫星和街景影像进行零样本可解释灾害制图的快速多智能体流水线。该框架集成了四个智能体：灾害感知智能体（DPA）、图像修复智能体（IRA）、损毁识别智能体（DRA）和灾害制图智能体（DMA）。通过结合遥感和街景数据，RAPIDMap无需人工微调，可在多种灾害类别间泛化，并生成结构化、可直接用于地图的灾害情报及恢复建议。

    arXiv:2609.00046v1 Announce Type: cross  Abstract: Rapid and reliable disaster mapping of impacted areas, damaged infrastructure, and affected populations is essential for emergency response and recovery. However, existing AI-based approaches often require extensive manual annotation, lack cross-hazard generalization, and rely on single-modal observations. To address these challenges, this paper proposes RAPIDMap, a rapid multi-agent pipeline for zero-shot interpretable disaster mapping from satellite and street-view imagery. The framework integrates four intelligent agents: Disaster Perception Agent (DPA), Image Restoration Agent (IRA), Damage Recognition Agent (DRA), and Disaster Mapping Agent (DMA). By combining remote sensing and street-view data, RAPIDMap eliminates the need for manual fine-tuning, generalizes across multiple disaster categories, and generates structured, map-ready disaster intelligence with recovery recommendations.
    
[^298]: trajectory-judge：仅基于结果的LLM评判器在智能体轨迹上遗漏了什么

    trajectory-judge: What Outcome-Only LLM Judges Miss on Agent Trajectories

    [https://arxiv.org/abs/2609.00038](https://arxiv.org/abs/2609.00038)

    仅看最终结果的LLM评判器无法发现智能体“答对但走错路”的问题——在可构造真值的确定性客服工具环境中，仅结果型评判器对静默故障的召回率仅45%且误报33%的正确轨迹，而基于逐步评分标准的评判器可将静默故障召回率提升至77%。

    

    仅基于结果的评估是LLM智能体在生产环境中的默认做法：向评判器展示用户请求和最终回复，询问其处理是否得当。这一指标在结构上无法察觉那些“以错误方式得到正确答案”的智能体。我们在真值可以通过构造获知的场景下测量这一盲区：一个确定性的使用工具的客服支持台环境、一个总能解决问题的脚本化oracle策略，以及一个在已知步骤恰好破坏一个环节的故障注入器，并根据用户可见结果是否仍然保持（静默型故障）与否（显性型故障）对故障进行分层。五种评判器（程序化规则、仅结果型、两种模型规模的逐步评分标准型、以及自一致性集成）在400条轨迹上按照检测能力、步骤定位、故障类型判定、校准度和成本进行评分。结果显示：仅结果型评判器能捕获84%的显性故障，但只能捕获45%的静默故障，同时还会误报33%的正确轨迹；而逐步评分标准型评判器对静默故障的召回率达到77%。

    arXiv:2609.00038v1 Announce Type: cross  Abstract: Outcome-only evaluation is the production default for LLM agents: show a judge the request and the final reply and ask whether it was handled well. The metric is structurally blind to an agent that reaches the right answer the wrong way. We measure that blind spot where ground truth is known by construction: a deterministic tool-using support-desk environment, a scripted oracle policy that always solves it, and a fault injector that breaks exactly one thing at a known step, stratifying faults by whether the customer-visible outcome survived (silent) or not (loud). Five judges (programmatic rules, outcome-only, step-rubric at two model sizes, and a self-consistency ensemble) are scored on detection, step localisation, fault typing, calibration, and cost over 400 trajectories. The outcome-only judge catches 84% of loud faults but 45% of silent ones while flagging 33% of correct trajectories; a step-rubric judge reaches 77% silent recall 
    
[^299]: EULER：基于证据校验回溯的未被充分利用链接探索，用于多智能体数学发现

    EULER: Exploring Underused Links with Evidence-Checked Return for Multi-Agent Mathematical Discovery

    [https://arxiv.org/abs/2609.00032](https://arxiv.org/abs/2609.00032)

    EULER 是一个多智能体数学发现系统，以跨数学领域的“桥”转移为搜索单元，让直接、相邻领域与远距领域路线相互竞争，并通过“提供源表示无法执行的操作且目标侧证据可经校验蕴含回溯至原命题”这一标准筛选桥，在 120 个近期猜想上产出了 10 个证明和 3 个反驳。

    

    数学界使用不同的对象、不变量和工具，因此在它们之间转移问题代价高昂且常常被跳过。我们提出 EULER，一个以这种转移——称为“桥”——为搜索单元的多智能体系统。围绕一个固定的猜想，EULER 让直接路线、相邻领域路线和远距领域路线相互竞争；只有当一座桥能够提供源表示无法执行的操作，并且其目标侧证据沿着经过校验的蕴含关系返回到原始命题时，该桥才能保留其预算。六项有序的压力测试会在昂贵的搜索开始之前拒绝无效的桥。我们在 120 个近期猜想上评估了 EULER。这些猜想已在搜索开始前冻结并经过污染筛查，选自已公开发表的论文，其作者最近曾在组合数学权威期刊《Journal of Combinatorial Theory, Series A》上发表过文章。EULER 产生了 10 个证明和 3 个反驳，此外……（原文摘要在此处截断）

    arXiv:2609.00032v1 Announce Type: new  Abstract: Mathematical communities work with different objects, invariants, and tools, so transferring a problem across them is expensive and often skipped. We present EULER, a multi-agent system that takes such a transfer--a bridge--as its unit of search. Around a fixed conjecture, EULER runs direct, adjacent-domain, and distant-domain routes in competition; a bridge keeps its budget only if it supplies an operation the source representation cannot execute and its target-side evidence returns to the original statement along a checked implication. Six ordered stress tests reject invalid bridges before expensive search begins.   We evaluate EULER on 120 recent conjectures. The conjectures were frozen before search and screened for contamination, and are drawn from public papers by authors who had recently published in the Journal of Combinatorial Theory, Series A, a leading journal in combinatorics. EULER produced 10 proofs and 3 refutations, plus 
    
[^300]: UI-Venus-2 技术报告

    UI-Venus-2 Technical Report

    [https://arxiv.org/abs/2609.00028](https://arxiv.org/abs/2609.00028)

    UI-Venus-2是一个通用GUI基础智能体，通过统一的闭环推理-行动框架跨移动、网页和桌面环境运行，并从环境、任务和验证三个维度联合扩展，从而获得可靠的强化学习信号并迈向实际部署。

    

    多模态GUI智能体已成为数字任务自动化的一个有前景的范式，但由于环境覆盖有限、任务构建脆弱以及奖励验证不可靠，从面向基准测试的模型过渡到可靠的真实世界应用仍然充满挑战。在本工作中，我们提出了UI-Venus-2，一个通用基础GUI智能体，旨在通过统一的闭环推理-行动框架跨移动、网页和桌面环境运行。为弥合迈向实际部署的差距，我们联合扩展了三个关键维度：(1) 环境，将覆盖范围扩展至170多个多语言移动应用和原生桌面操作系统；(2) 任务，采用深度研究流水线进行基于功能的指令生成；(3) 验证，采用结合视觉关键点和多模型投票的轨迹级与样本级评估器，以确保训练中可靠的强化学习信号。

    arXiv:2609.00028v1 Announce Type: new  Abstract: Multimodal GUI agents have emerged as a promising paradigm for digital task automation, yet transitioning from benchmark-oriented models to dependable real-world applications remains challenging due to limited environment coverage, brittle task construction, and unreliable reward verification. In this work, we present UI-Venus-2, a general-purpose foundation GUI agent designed to operate across mobile, web, and desktop environments through a unified closed-loop reasoning-action framework. To bridge the gap toward practical deployment, we jointly scale three critical dimensions: (1) Environments, expanding coverage to more than 170 multilingual mobile apps and native desktop operating systems; (2) Tasks, employing a deep-research pipeline for function-grounded instruction generation; and (3) Verification, adopting trace-level and sample-level evaluators with visual keypoints and multi-model voting to ensure reliable RL signals for trainin
    
[^301]: SCAFFOLD：一个大规模的计算机科学研究图表结构化数据集，包含图表问答与思维链推理轨迹

    SCAFFOLD: A Large-Scale Structured Dataset of Computer Science Research Figures with Diagram QA and Chain-of-Thought Reasoning Traces

    [https://arxiv.org/abs/2609.00018](https://arxiv.org/abs/2609.00018)

    SCAFFOLD是一个面向计算机科学研究图表的大规模结构化数据集，提供15.7万个配有说明、上下文、问答对和思维链推理轨迹的图表样本，填补了训练视觉-语言模型理解学术图表的数据空白。

    

    计算机科学论文严重依赖图表：架构图、系统流程图和流程示意图，这些图表所承载的信息往往比其周围的文字更多。目前尚无公开数据集将这类特定图表与说明文字、上下文、问题、答案以及逐步推理配对，而这正是训练视觉-语言模型理解这些图表所必需的。我们提出了SCAFFOLD，这是一个大规模的计算机科学研究图表结构化数据集，包含图表问答和思维链推理轨迹。该数据集由来自arXiv计算机科学论文的（图像、说明文字、上下文、问答对、思维链）元组组成，通过版面检测和PDF解析制作，并借助AI辅助的问题生成步骤完成。由此构建的大规模SCAFFOLD-157K数据集涵盖3,058篇论文，包含29,887张图表（共157,387对数据）……

    arXiv:2609.00018v1 Announce Type: new  Abstract: Computer science papers rely heavily on diagrams: architecture drawings, system flowcharts, and pipeline schematics that often carry more information than the text around them. There is currently no public dataset that pairs this specific kind of figure with captions, context, questions, answers, and step-by-step reasoning, which is exactly what is needed to train a vision-language model to understand them. We present \textbf{SCAFFOLD}\footnote{https://github.com/theranjitraut/scaffold}, a large-scale structured dataset of computer science research figures with diagram QA and Chain-of-Thought reasoning traces. This dataset consists of (image, caption, context, question-answer, chain-of-thought) tuples from arXiv computer science papers prepared using layout detection and PDF parsing, with an AI-assisted question-generation step. The resulting large-sized SCAFFOLD-157K dataset spans 3,058 papers with 29,887 figures (157,387 pairs), a medi
    
[^302]: OpenAgentFlow：为异构AI智能体集群实现系统级安全边界

    OpenAgentFlow: Enabling System-Wide Safety Boundaries for Heterogeneous AI Agent Fleets

    [https://arxiv.org/abs/2609.00015](https://arxiv.org/abs/2609.00015)

    OpenAgentFlow提出控制平面/动作平面分离的架构，通过将GUI操作、API调用、工具调用等各类智能体动作统一规范化为事件流，并在动作提交边界集中执行安全检查，从而为异构AI智能体集群提供系统级的安全治理、可审计性和策略演进能力。

    

    由大语言模型驱动的AI智能体正在从孤立的助手演变为异构系统，其中多个智能体、规划器、控制器和执行后端在同一用户或企业环境中运行。在这种环境下，安全性成为一个系统级的动作治理问题：即在对共享状态进行修改之前，决定是否应提交智能体生成的具体动作。现有的安全防护措施覆盖了提示词、工具调用、GUI操作和智能体本地行为，但往往导致执行碎片化，掩盖了跨多步骤动作流中涌现的风险，并且在可审计性和策略演进方面支持有限。我们提出了OpenAgentFlow，这是一种控制平面/动作平面架构，在动作提交边界实施安全控制。它将待处理的GUI操作、API调用、工具调用和LLM生成的调用统一规范化为AgentEvent流，并将每个事件路由至共享的预执行（原文在此处截断）

    arXiv:2609.00015v1 Announce Type: new  Abstract: AI agents powered by large language models are evolving from isolated assistants into heterogeneous systems in which multiple agents, planners, controllers, and execution backends operate over the same user or enterprise environment. In such settings, safety becomes a system-level action-governance problem: deciding whether concrete agent-generated actions should be committed before they modify shared state. Existing safeguards cover prompts, tool calls, GUI actions, and agent-local behavior, but often leave enforcement fragmented, obscure risks that emerge across multi-step action flows, and provide limited support for auditability and policy evolution.   We present OpenAgentFlow, a control-plane/action-plane architecture that enforces safety at the action-commit boundary. It normalizes pending GUI actions, API calls, tool calls, and LLM-generated invocations into a unified AgentEvent stream, routes each event through a shared pre-execu
    
[^303]: 基于真实世界行为数据的用户画像：面向个性化对齐与多视角推理

    Behaviorally Grounded User Profiles from the Wild for Personalized Alignment and Multi-Perspective Reasoning

    [https://arxiv.org/abs/2609.00014](https://arxiv.org/abs/2609.00014)

    提出直接从真实匿名社交媒体数据中提取开放式高保真用户画像的行为锚定框架，在训练时个性化与测试时多视角推理两种范式下均显著优于合成人格基线。

    

    基于人格（Persona）驱动的技术正日益被用于将大语言模型（LLM）适配到多样化场景中。然而，现有方法主要依赖于僵化的、合成的人格设定，这些设定抹平了个体差异、依赖刻板印象，并且忽略了驱动真实人类偏好的微妙信号。我们提出了画像行为锚定（profile behavioral grounding）框架，可直接从真实、匿名化的社交媒体帖子中提取开放式、高保真的用户画像。我们在两种范式下对这些画像进行评估：通过监督微调（SFT）实现的训练时个性化，以及非参数化的测试时多视角推理。在复杂的推荐与开放式查询基准测试中，基于真实行为的画像始终能提升基础模型的表现，并优于合成画像基线，实现了更强的参数化对齐，并支持更丰富、更多维度的推理。

    arXiv:2609.00014v1 Announce Type: cross  Abstract: Persona-driven techniques increasingly adapt large language models (LLMs) to diverse contexts. However, existing methods predominantly rely on rigid, synthetic personas that flatten individual variation, rely on stereotypes, and miss the nuanced signals driving actual human preferences. We introduce profile behavioral grounding, a framework for extracting open-ended, high-fidelity user profiles directly from authentic, anonymized social media posts. We evaluate these profiles across two paradigms: train-time personalization via supervised finetuning (SFT) and non-parametric test-time multi-perspective reasoning. Across complex recommendation and open-ended query benchmarks, behaviorally grounded profiles consistently improve base models and outperform synthetic profile baselines, driving stronger parametric alignment and enabling richer, multifaceted reasoning. Our findings establish open-ended, behavior-derived profiles as a highly di
    
[^304]: 大语言模型中的长程状态追踪：通过深度依赖的工具调用序列执行MD5

    Long-Horizon State Tracking in LLMs: Executing MD5 through a Deep Sequence of Dependent Tool Calls

    [https://arxiv.org/abs/2609.00012](https://arxiv.org/abs/2609.00012)

    该论文提出以MD5哈希分步计算作为测试范式（在64轮中执行196个相互依赖的工具调用，并需在上下文中持续携带四个32位状态字），从而干净地隔离并评测大语言模型的长程精确状态追踪能力。

    

    长程任务在大语言模型（LLM）评估中仍然少见，这是有原因的：当每一步都依赖于前一步时，单独看来十分优异的每步准确率会灾难性地衰减——误差不断级联放大，端到端失败概率随任务长度急剧增长。现有的智能体基准测试报告的是端到端成功率，却将这种状态追踪难度与指令理解混为一谈，没有提供可将其隔离的对照组，而且容易受到“幻觉出最终答案”之类的捷径影响，因此无法解释一次长程运行为何失败。大语言模型究竟是否能够在多次工具调用之间携带精确的中间状态，这一问题本身也尚未得到充分验证。我们通过让模型逐步计算密码学哈希MD5来干净地测试这一点：模型在64轮中执行196个相互依赖的工具调用，并在自身上下文中将四个32位字$(a,b,c,d)$从一次调用携带到下一次调用。对指令的理解是平凡的……

    arXiv:2609.00012v1 Announce Type: new  Abstract: Long-horizon tasks remain uncommon in large language model (LLM) evaluation, and for a reason: when each step depends on the last, per-step accuracy that looks excellent in isolation decays catastrophically, as errors cascade and the end-to-end failure probability grows sharply with length. Existing agentic benchmarks report end-to-end success but confound this state-tracking difficulty with instruction interpretation, give no control group that isolates it, and are vulnerable to shortcuts such as a hallucinated final answer, so they cannot say why a long run fails. Whether an LLM can carry exact intermediate state across many tool calls at all is itself not well established. We test this cleanly by having the model compute a cryptographic hash, MD5, step by step: a sequence of $196$ dependent tool calls over $64$ rounds while it carries four $32$-bit words $(a,b,c,d)$ in its own context from one call to the next. Interpretation is trivi
    
[^305]: 基于指令微调小型语言模型的渐进式老年人金融诈骗增量风险评估

    Incremental Risk Assessment of Progressive Elder Financial Scams via Instruction-Tuned Small Language Models

    [https://arxiv.org/abs/2609.00005](https://arxiv.org/abs/2609.00005)

    该论文提出了一种基于指令微调小型语言模型的累计轮次风险评估框架，能够在资源受限环境下对针对老年人的渐进式多轮金融诈骗对话进行增量式动态风险监控。

    

    针对老年人的金融诈骗日益通过文本和语音渠道（如电子邮件、短信和电话）发生，并在多个对话轮次中逐步展开：从冒充身份或随意接触开始，通过建立信任和制造紧迫感不断升级，最终以索取敏感信息或要求金融转账收尾。由于风险信号在对话轮次中逐步显现，有效的检测需要模型能够在资源受限的部署环境下持续更新风险估计。我们提出了一种基于累计对话轮次的风险评估框架，该框架增量地聚合对话轮次并在每一步重新估计风险，从而实现对逐步演化的诈骗对话的动态监控。我们构建了一个多轮对话数据集，涵盖投资、慈善和技术支持诈骗场景，每段对话包含两到八个轮次，并在每个累计阶段进行了标注……

    arXiv:2609.00005v1 Announce Type: new  Abstract: Financial scams targeting older adults increasingly occur through text and voice channels such as email, SMS, and phone calls, unfolding over multiple conversational turns that begin with impersonation or casual contact, escalate through trust building and urgency, and culminate in requests for sensitive information or financial transfers. Because risk signals emerge incrementally across turns, effective detection requires models that continuously update risk estimates under resource-constrained deployment settings. We propose a cumulative turn-based risk assessment framework that incrementally aggregates conversational turns and re-estimates risk at each step, enabling dynamic scam monitoring across progressively evolving conversations. A multi-turn dialogue dataset is constructed to cover investment, charity, and tech support scam scenarios, with each dialogue containing two to eight turns and annotated at every cumulative stage with a
    
[^306]: 随机需求时序下多品种产能受限批量问题的离散时间MDP建模

    Discrete-Time MDP Modeling for Multi-Item Capacitated Lot Sizing with Stochastic Demand Timing

    [https://arxiv.org/abs/2609.00004](https://arxiv.org/abs/2609.00004)

    本文将需求量确定但需求到达时期随机的多品种产能受限批量问题创新性地建模为离散时间马尔可夫决策过程（DTMDP），通过在需求层面进行生产与分配决策来刻画产能竞争与库存动态，并通过与确定性对照实例的比较揭示随机需求时序会显著增加问题的计算难度。

    

    本文研究有限周期下的多品种产能受限批量生产问题，其中需求数量是确定性的，而需求的到达时期是随机的。每个需求在已知的时间窗口内恰好发生一次，且必须不晚于其截止期限得到满足。所提出的模型在需求层面进行生产与分配决策，从而能够刻画产能竞争、针对特定需求的缺货欠交以及依赖于分配方式的库存动态。该随机问题被表述为离散时间马尔可夫决策过程（DTMDP），包括状态空间、可行行动集合、转移核和单周期成本函数。为了分离出随机时序带来的计算影响，每个随机实例首先与一个确定性对照实例进行比较，后者将每个到达分布替换为其最可能的到达时期。该比较表明，随机时序显著增加了（原文在此处截断）

    arXiv:2609.00004v1 Announce Type: new  Abstract: This paper studies a finite-horizon multi-item capacitated lot-sizing problem in which demand quantities are deterministic, while demand-arrival periods are stochastic. Each demand occurs once within a known time window and must be satisfied no later than its deadline. The proposed model makes production and allocation decisions at the demand level, allowing it to represent capacity competition, demand-specific backlog, and allocation-dependent inventory dynamics. The stochastic problem is formulated as a discrete-time Markov decision process (DTMDP), including the state space, feasible actions, transition kernel, and one-period cost function. To isolate the computational effect of stochastic timing, each stochastic instance is first compared with a deterministic counterpart in which each arrival distribution is replaced by its most likely arrival period. This comparison shows that stochastic timing substantially increases the number of 
    
[^307]: I-CARE：在可控、多样且具代表性的文生图模型遗忘设置中分析干扰相关现象

    I-CARE: Analysis of interference-related phenomena in a controllable, diverse and representative unlearning setting for text-to-image models

    [https://arxiv.org/abs/2609.00003](https://arxiv.org/abs/2609.00003)

    本文提出I-CARE方法论，首次将文生图模型机器遗忘过程中对语义相关概念造成的意外损害（即“干扰”）形式化为首要研究对象，通过为任务、指标和结果报告提供正式定义，实现对干扰现象的系统性、可复现研究。

    

    arXiv:2609.00003v1 公告类型：新论文 摘要：机器遗忘研究如何从AI模型中移除知识，使系统忘记其之前学到的某个概念。尽管生成式机器遗忘取得了快速进展，但本应保留的语义相关概念出现的意外退化（以下称为“干扰”）仍未得到充分表征，且评估方式不一致。本文提出I-CARE，这是一种将干扰形式化为生成式遗忘研究中首要研究对象的方法论。I-CARE并非提出新的基准或遗忘算法，而是为任务、指标和结果报告模板提供正式定义，从而支持在不同遗忘设置下对干扰进行系统性、可复现的研究。虽然我们的方法论设计旨在随着模型和遗忘算法的演进保持有效，将长期科学洞察与暂时的实证结果解耦，我们还提供了一个可行性演示，以……

    arXiv:2609.00003v1 Announce Type: new  Abstract: Machine unlearning studies the removal of knowledge from an AI model, making the system forget a concept it previously learned. Despite rapid progress in generative machine unlearning, the unintended degradation of semantically related concepts that should have been retained (henceforth, interference) remains poorly characterized and inconsistently evaluated. This paper introduces I-CARE, a methodology that formalizes interference as a first-class object of study in generative unlearning. Rather than proposing a new benchmark or unlearning algorithm, I-CARE provides formal definitions for tasks, metrics, and templates for reporting results, enabling the systematic and reproducible study of interference across unlearning settings. While our methodology is designed to remain valid as models and unlearning algorithms evolve, decoupling long-term scientific insight from transient empirical results, we present a feasibility demonstration with
    
[^308]: HyperWorld：超图结构化的状态序列化提升学习型文本世界模型

    HyperWorld: Hypergraph-Structured State Serialization Improves Learned Textual World Models

    [https://arxiv.org/abs/2609.00002](https://arxiv.org/abs/2609.00002)

    本文提出一种以实体为中心的超边（超图结构化）状态序列化方法，在相同训练目标下显著提升 0.5B–1.5B 规模语言模型学习文本世界模型的能力，且在分布偏移条件下收益最为明显。

    

    世界模型使语言模型智能体能够预测环境动态，并在行动之前进行规划。在文本环境中，模型必须从序列化的状态描述中学习符号化的动作效果，但序列化结构所起的作用仍未得到充分探索。我们提出了 HyperWorld，这是一项针对学习型文本世界模型中状态序列化的受控研究。我们将原始观测与同一真实状态的三种符号化序列化方式进行对比：独立句子、成对三元组，以及以实体为中心、将围绕实体和关系的多个相关事实聚合为一组的超边单元。所有变体均采用相同的训练目标：给定一个状态和一个动作，预测符号化的动作效果，或判定该动作不可行。在模型规模、数据预算以及分布内和分布外测试世界等多种设置下，超边序列化在 0.5B–1.5B 规模的模型以及分布偏移条件下带来了最显著的收益。更大的模型会缩小差距……（摘要原文在此处截断）

    arXiv:2609.00002v1 Announce Type: new  Abstract: World models enable language-model agents to predict environment dynamics and plan before acting. In text environments, the model must learn symbolic action effects from serialized state descriptions, but the role of serialization structure remains underexplored. We present HyperWorld, a controlled study of state serialization for learned textual world models. We compare raw observations with three symbolic serializations of the same ground-truth state: independent sentences, pairwise triples, and entity-centered hyperedge units that group multiple related facts around entities and relations. All variants use the same training objective: given a state and an action, predict symbolic effects or judge the action infeasible. Across model scales, data budgets, and in-distribution and out-of-distribution test worlds, hyperedge serialization gives the clearest gains for 0.5B--1.5B models and under distribution shift. Larger models reduce the g
    
[^309]: 超越人类监督扩展大规模推理模型：通往超级智能之路

    Scaling Large Reasoning Models beyond Human Supervision: A Path toward Superintelligence

    [https://arxiv.org/abs/2608.31075](https://arxiv.org/abs/2608.31075)

    本文沿“奖励”与“经验”两条轴线，系统阐述了大规模推理模型如何在人类监督逐渐退出后，借助可复用验证器、自生成课程与自主共同进化持续自我提升，为通往超级智能指明路径。

    

    大规模推理模型（LRM）的最新进展表明，具有可验证奖励的强化学习（RLVR）能够显著提升模型在数学和代码等结果可自动检验领域的推理能力。然而，将这一进展扩展到开放式和智能体任务仍然困难，因为可靠的奖励更难获得，且直接的人类监督无法跟上模型生成经验的规模与复杂性。本文研究了当人类监督逐渐从学习回路中退出时，LRM 如何能够持续改进。我们从两个相互关联的维度考察这一问题：奖励轴梳理了从针对单个实例的人类判断，到可复用的验证器、乃至无需人类反馈即可运作的奖励的发展脉络；经验轴则考察学习如何从人类策划的任务与环境，演进到自生成课程、构建的环境以及自主的共同进化。

    arXiv:2608.31075v1 Announce Type: new  Abstract: Recent advances in large reasoning models (LRMs) have shown that reinforcement learning with verifiable rewards (RLVR) can substantially improve reasoning in mathematics and code, where outcomes can be checked automatically. Extending this progress to open-ended and agentic tasks remains difficult because reliable rewards are harder to obtain and direct human supervision cannot keep pace with the scale and complexity of model-generated experience. This paper studies how LRMs can continue to improve as human supervision gradually recedes from the learning loop. We examine two connected dimensions of this problem. The reward axis traces the development from per-instance human judgments to reusable verifiers and rewards that operate even without human feedback. The experience axis examines how learning can progress from human-curated tasks and environments toward self-generated curricula, constructed environments, and autonomous co-evolutio
    
[^310]: 自回归马赛克：探究纯文本语言模型中的二维空间推理能力

    Autoregressive Mosaics: Probing 2D Spatial Reasoning in Text-Only Language Models

    [https://arxiv.org/abs/2608.30751](https://arxiv.org/abs/2608.30751)

    该论文提出AM-Bench基准，通过区分“几何翻译”与“开放式布局”两类任务，发现纯文本大模型普遍能将明确的空间描述转化为代码，但其真正的二维空间布局推理能力存在显著差异。

    

    仅基于文本和代码训练的大型语言模型（LLM）有时能够生成绘制出可辨识图像的程序。然而，这究竟反映了模型对二维空间布局的内部表征，还是仅仅体现了将空间描述翻译成代码的能力，目前尚不清楚。我们提出了自回归马赛克（AM-Bench）基准，它将这两个因素区分开来：首先是翻译任务，向模型提供以文字完整描述的图像几何结构作为提示，要求模型生成能产生该图像的代码；其次是布局任务，要求模型根据信息不完整的提示自行构建图像。在八个仅支持文本与代码的开源权重模型上，所有模型都能可靠地将给定的几何结构翻译成代码，但它们在开放式布局任务上的表现差异显著，这表明这些差异无法仅用代码生成能力来解释。一项关于输出介质的消融实验进一步表明，模型表达空间理解所用的接口或媒介……（原文在此处截断）

    arXiv:2608.30751v1 Announce Type: new  Abstract: Large language models (LLMs) trained only on text and code can sometimes generate programs that draw recognizable images. However, it is unclear whether this reflects an internal representation of 2D spatial layout or simply the ability to translate spatial descriptions into code. We introduce Autoregressive Mosaics (AM-Bench), a benchmark that separates these factors: First, a translation task gives a model a fully specified geometry of a picture in words as a prompt and asks for the code that produces it. Second, a layout task requires the model to compose an image from an underspecified prompt. Across eight open-weight text-and-code-only models, all models reliably translate specified geometry into code, but their open-ended layout performance differs substantially, indicating that these differences are not explained by code-generation ability alone. An output-medium ablation further shows that the interface or medium of expression th
    
[^311]: 用于声明核查价值检测的小型语言模型校准

    Calibrating Small Language Models for Claim Check-Worthiness Detection

    [https://arxiv.org/abs/2608.30731](https://arxiv.org/abs/2608.30731)

    提出NN-PPI方法，作为推理时的轻量级后处理校准层，使小型语言模型在声明核查价值检测任务上以低一个数量级的服务成本达到大型语言模型的准确率，且无需重新训练模型。

    

    评估声明的核查价值是自动化事实核查流程中至关重要的第一步。这项工作源于一家早期初创公司面临的实际部署挑战：对每一条传入声明都运行大型语言模型在成本和延迟上都是难以承受的，而较小的模型又会牺牲准确性。我们提出了NN-PPI，这是预测驱动推理的一种逐点扩展方法，它在推理时作为轻量级的后处理层来校准模型预测，无需重新训练底层模型。根据基线模型的规模和性能不同，NN-PPI实现了12%到33.80%的加权F1提升，使小型语言模型达到了与大型语言模型相当的水平。除了少样本小型语言模型之外，NN-PPI还进一步改进了一个已在生产环境中部署的微调模型，表明残差校准与监督微调是互补的。通过从服务成本低一个数量级的模型中恢复出LLM级别的准确性，它…

    arXiv:2608.30731v1 Announce Type: cross  Abstract: Assessing claim check-worthiness is an essential first step in automated fact-checking pipelines. This work is motivated by a real deployment challenge at an early-stage startup: running large language models (LLMs) over every incoming claim is cost- and latency-prohibitive, yet smaller models sacrifice accuracy. We propose NN-PPI, a pointwise extension of Prediction-Powered Inference (PPI) that calibrates model predictions at inference time as a lightweight post-hoc layer, without re-training the underlying model. NN-PPI achieves weighted F1 gains ranging from 12% to 33.80% depending on the size and performance of the baseline model, bringing SLMs on par with larger LLMs. Beyond few-shot SLMs, NN-PPI further improves a production-deployed fine-tuned model, demonstrating that residual calibration is complementary to supervised fine-tuning. By recovering LLM-level accuracy from models that are an order of magnitude cheaper to serve, it 
    
[^312]: BiG-SURE——用于大语言模型语义不确定性与可靠性估计的二部图方法

    BiG-SURE - Bipartite Graph for Semantic Uncertainty and Reliability Estimation of LLMs

    [https://arxiv.org/abs/2608.30646](https://arxiv.org/abs/2608.30646)

    提出了一种基于跨温度语义一致性的黑盒不确定性估计方法BiG-SURE，通过构建低温锚点与高温探针之间的二部图并用谱能量衡量语义一致性，从而评估大语言模型输出的可靠性。

    

    可靠的不确定性估计是在安全关键场景中部署大语言模型（LLM）和视觉-语言模型（VLM）的关键前提，尤其是在模型参数不可访问（黑盒）的情况下。我们提出了BiG-SURE，一种基于跨温度语义一致性的不确定性估计器。该方法在保持语义不变的输入变换下，将低温采样得到的响应作为稳定的语义锚点，将高温采样得到的响应作为探针。随后，方法利用基于自然语言推理（NLI）的蕴含分数构建锚点-探针二部图，并通过该矩阵的归一化平方谱能量来定义置信度，不确定性则由其补集给出。这种基于二部图的语义不确定性与可靠性估计（SURE）分数，用于衡量高温探针是否与模型稳定的低温信念保持语义一致。我们在文本问答等任务上对BiG-SURE进行了评估。

    arXiv:2608.30646v1 Announce Type: cross  Abstract: Reliable uncertainty estimation is a crucial requirement for deploying large language models (LLMs) and vision-language models (VLMs) in safety-critical settings, especially when the model parameters are not accessible (black-box). We propose BiG-SURE, an uncertainty estimator based on cross-temperature semantic agreement. The method samples low-temperature responses as stable semantic anchors and high-temperature responses as probes under meaning-preserving input transformations. It then constructs an anchor-probe Bipartite Graph (BiG) using NLI-based entailment scores and defines confidence through the normalized squared spectral energy of this matrix, with uncertainty given by its complement. This bipartite graph-based Semantic Uncertainty and Reliability Estimation (SURE) score measures whether high-temperature probes remain semantically aligned with the model's stable low-temperature belief or not. We evaluate BiG-SURE on text QA,
    
[^313]: 面向指代图像分割与定位的高成本效益主动学习

    Cost-efficient Active Learning for Referring Image Segmentation and Grounding

    [https://arxiv.org/abs/2608.30621](https://arxiv.org/abs/2608.30621)

    该论文提出一种高成本效益的主动学习方法，通过基础模型生成辅助区域-文本对，并引入“指代区域模糊度”采集函数来优先挑选跨区域竞争激烈的图像，从而在仅有原始图像的现实设置下高效降低视觉定位与指代图像分割的标注成本。

    

    收集自然语言指代表达连同区域标注（如掩码或边界框）是视觉定位（VG）中的一个主要瓶颈，因为标注者必须撰写能够将目标区域与视觉相似区域区分开来的描述。为此，我们在仅有原始图像而无比对文本的现实设置下，为视觉定位构建了主动学习（AL）框架。由于真实文本不可用，样本选择必须估计哪些图像包含需要判别性指代表达的模糊区域。为解决这一问题，我们利用基础模型生成辅助的区域-文本对，并提出“指代区域模糊度”这一新的采集函数，用于衡量模型的置信度是坍缩于单一区域还是分散在多个候选区域之间。这使我们的方法能够优先选择具有强烈跨区域竞争的图像，这类图像由于……（摘要截断）

    arXiv:2608.30621v1 Announce Type: cross  Abstract: Collecting natural-language referring expressions along with region annotations, such as masks or boxes, is a major bottleneck in visual grounding (VG), as annotators must write descriptions that distinguish target regions from visually similar ones. We tackle this by formulating active learning (AL) for VG under the realistic setting where only raw images are available without accompanying text. Since ground-truth text is unavailable, sample selection must estimate which images contain ambiguous regions that would require discriminative referring expressions. To address this, we generate auxiliary region-text pairs using foundation models, and introduce Referred Region Ambiguity, a new acquisition function that measures whether the model's confidence collapses onto a single region or disperses across multiple candidates. It allows our method to prioritize images with strong cross-region competition, which are more informative due to t
    
[^314]: 用户会知道吗？针对使用工具的LLM智能体的隐蔽间接提示注入

    Will the User Ever Know? Covert Indirect Prompt Injection on Tool-Using LLM Agents

    [https://arxiv.org/abs/2608.30362](https://arxiv.org/abs/2608.30362)

    该论文从用户视角将间接提示注入的攻击成功率分解为隐蔽成功率（CSR）和公开成功率（OSR），揭示了智能体在最终响应中不留痕迹地执行恶意注入的隐蔽攻击威胁。

    

    随着LLM智能体通过工具执行真实世界的操作，间接提示注入（IPI）已成为一种严重的威胁。标准的评估指标——攻击成功率（ASR）——只统计注入是否成功，却忽略了用户在智能体最终响应中能够注意到什么。通过观察成功的注入轨迹，我们发现两种截然不同的结果：智能体在执行注入的同时返回看似正常的响应，或者在最终响应中报告被注入的操作，从而给用户留下察觉的机会。我们将这两类成功分别称为隐蔽成功和公开成功。从用户视角出发，我们将ASR分解为隐蔽成功率（CSR）——统计在最终响应中不留任何痕迹的成功注入——以及公开成功率（OSR）——统计用户能够察觉的成功注入。为了理解造成这一差距的原因，我们分析了成功的注入轨迹，发现注入后智能体的行为是区分隐蔽与公开的关键：隐蔽的轨迹会将控制权交回……

    arXiv:2608.30362v1 Announce Type: new  Abstract: As LLM agents take real-world actions through tools, indirect prompt injection (IPI) has emerged as a serious threat. The standard metric, Attack Success Rate (ASR), counts whether an injection succeeds but ignores what the user notices in the agent's final response. Looking at successful injection traces, we find two distinct outcomes: the agent executes the injection while returning an otherwise normal response, or reports the injected action in its final response, giving the user a chance to notice. We call these covert and overt successes. From the user's perspective, we decompose ASR into the Covert Success Rate (CSR), counting successes leaving no trace in the final response, and the Overt Success Rate (OSR), counting successes the user can detect. To understand what drives the gap, we analyze successful trajectories and find that the agent's behavior after the injection separates covert from overt: covert traces hand control back 
    
[^315]: E-SENS：面向负约束检索的排斥敏感惩罚方法

    E-SENS: Exclusion-Sensitive Penalization for Negative-Constraint Retrieval

    [https://arxiv.org/abs/2608.30130](https://arxiv.org/abs/2608.30130)

    E-SENS是一种无需训练的重排序方法，通过为被排除概念提取“陷阱查询”并从检索分数中减去其相似度，有效惩罚与用户排除概念相关的文档，从而提升检索系统对负向约束的遵守能力。

    

    检索增强语言模型在检索器提供用户明确排除概念的相关证据时，可能无法遵守负向约束。除了显式否定之外，查询还可能要求答案包含一个概念而排除另一个概念，或者要求实体属于某一类别但与密切相关的实例不同。由于被排除的概念仍然出现在查询文本中，稠密检索器可能会对与该概念相关的文档赋予高相似度，即使用户明确要求避开它。我们提出了E-SENS，一种面向否定敏感检索的无训练重排序方法。E-SENS为被排除的一方提取一个紧凑的“陷阱查询”，并从原始查询的检索分数中减去陷阱查询的相似度。在ExcluIR基准上，E-SENS在四个嵌入模型上展现出清晰的召回率-违规权衡，并在保持召回率的设置下有效减少了陷阱检索。

    arXiv:2608.30130v1 Announce Type: cross  Abstract: Retrieval-augmented language models can fail to respect negative constraints when the retriever supplies evidence about concepts the user explicitly excluded. Beyond explicit negation, queries may ask for answers that include one concept while excluding another, or for entities that belong to a category but differ from a closely related instance. Because the excluded concept still appears in the query text, dense retrievers may assign high similarity to documents about that concept even when the user asks to avoid it. We introduce E-SENS, a training-free reranking method for negation-sensitive retrieval. E-SENS extracts a compact trap query for the excluded side and subtracts trap-query similarity from the original-query retrieval score. On ExcluIR, E-SENS shows a clear recall-violation trade-off across four embedding models and reduces trap retrieval at recall-preserving settings.
    
[^316]: Arkios：一个从头训练的开源英-尼泊尔语双语语言模型，配备天城文感知分词器

    Arkios: An Open Bilingual English-Nepali Language Model Trained From Scratch, with a Devanagari-Aware Tokenizer

    [https://arxiv.org/abs/2608.30092](https://arxiv.org/abs/2608.30092)

    Arkios是一个从零训练的10.4亿参数英-尼泊尔语双语开源模型，采用专门设计的天城文感知分词器，以少一个数量级的训练数据超越了同规模开源模型，并揭示了低资源语言评估中提示格式对结果的关键影响。

    

    我们提出了Arkios，一个拥有10.4亿参数的稠密transformer模型，在1500亿token的英-尼泊尔语双语语料上从零开始预训练，使用了自定义的单文件C/CUDA训练框架，以及为本项目专门构建的天城文感知字节级BPE分词器。在ARC-Easy和ARC-Challenge基准上，Arkios超越了三个规模相当的开源模型（Pythia-1.4B、TinyLlama-1.1B、OLMo-1B），尽管其训练token数量少了一个数量级，这可能得益于我们的教育类网页文本预训练数据与ARC的小学科学题格式相匹配，而非通用能力上的优势。我们报告了在标准协议下的完整评估结果，包括对早前部分样本估计的更正，以及针对低资源语言小模型评估的发现：常用评估框架所使用的标准多选题字母提示格式使该模型在尼泊尔语阅读理解上仅达到随机水平，同时在……（原文在此处截断）

    arXiv:2608.30092v1 Announce Type: cross  Abstract: We present Arkios, a 1.04B-parameter dense transformer pretrained from scratch on 150B tokens of bilingual English-Nepali text, using a custom single-file C/CUDA training stack and a Devanagari-aware byte-level BPE tokenizer built for this project. On ARC-Easy and ARC-Challenge, Arkios exceeds three comparably sized open models (Pythia-1.4B, TinyLlama-1.1B, OLMo-1B) despite an order of magnitude fewer training tokens, likely aided by a match between our educational-web-text pretraining data and ARC's grade-school-science format rather than a general capability advantage. We report full evaluation results under standard protocols, including a correction to an earlier partial-sample estimate, and findings specific to evaluating small models in a low-resource language: the standard multiple-choice-letter prompt format used by common evaluation harnesses places this model at chance on Nepali reading comprehension, and simultaneously at cha
    
[^317]: REIGN：利用集成引导网络的翻新嵌入实现高效的上下文长度扩展

    REIGN: Refurbished Embeddings with Integrated Guidance Networks for Efficient Context-Length Scaling

    [https://arxiv.org/abs/2608.29899](https://arxiv.org/abs/2608.29899)

    REIGN通过在冻结引导网络生成的块嵌入序列上运行对比训练的双编码器，将词元级处理与文档级推理解耦，使长文档检索的训练成本相比分块Transformer微调降低约四个数量级。

    

    对长文档进行稠密检索的代价高昂。词元级编码器在序列长度上呈二次方扩展，而大多数长上下文嵌入模型只能通过架构上的变通方法或拉长十亿参数级大语言模型才能达到32K词元。我们提出REIGN（Refurbished Embeddings with Integrated Guidance Networks，集成引导网络的翻新嵌入），这是一个经过对比训练的双编码器，它在由冻结的引导网络（GN）生成的上下文化块嵌入序列上运行，而不是在原始词元上运行。REIGN针对多块输入，主要用于文档到文档的检索；单块输入则仍由GN处理。通过将词元级处理与文档级推理解耦，并将GN嵌入缓存到磁盘，相对于分块Transformer微调，每个文档的训练成本降低了大约四个数量级。我们还发布了一个合成的长文档检索基准，用于长上下文长度下的对比训练与评估。

    arXiv:2608.29899v1 Announce Type: cross  Abstract: Dense retrieval over long documents is expensive. Token-level encoders scale quadratically in sequence length, and most long-context embedding models reach 32K tokens only through architectural workarounds or by stretching billion-parameter LLMs. We propose REIGN (Refurbished Embeddings with Integrated Guidance Networks), a contrastively trained bi-encoder that operates on sequences of contextualised chunk embeddings from a frozen Guidance Network (GN) rather than on raw tokens. REIGN targets multi-chunk inputs, primarily for document-to-document retrieval; single-chunk inputs stay with the GN. Decoupling token-level processing from document-level reasoning, and caching the GN embeddings to disk, cuts per-document training cost by roughly four orders of magnitude relative to chunked Transformer fine-tuning. We also release a synthetic long-document retrieval benchmark for contrastive training and evaluation at long context lengths. Acr
    
[^318]: InteractBench：在信息未揭示条件下评测大语言模型竞赛编程能力的基准

    InteractBench: Benchmarking LLMs on Competitive Programming under Unrevealed Information

    [https://arxiv.org/abs/2608.29632](https://arxiv.org/abs/2608.29632)

    提出了InteractBench基准，包含322个精选自主流编程竞赛的高质量交互式问题，用于评测大语言模型在关键信息未预先揭示、需通过多轮交互进行算法推理的能力。

    

    竞赛编程正日益被用于评估大语言模型（LLM）的算法推理能力。然而，现有的基准测试主要聚焦于全信息任务，即所有问题输入都在开始时预先提供。这忽略了算法推理的一个关键维度：生成的程序在关键信息未预先揭示时的运行能力。交互式问题是竞赛编程的一个独特组成部分，正体现了这一挑战。这类问题要求程序在严格的协议约束和有限的查询预算下，与交互器（评测程序）进行多轮交互，且新信息仅在响应查询时才被揭示。为填补这一空白，我们提出了InteractBench，这是一个包含322个高质量交互式问题的基准，这些问题精选自Codeforces、AtCoder、IOI和ICPC。每个问题都配备了可执行的本地交互器，……

    arXiv:2608.29632v1 Announce Type: new  Abstract: Competitive programming is increasingly being used to evaluate the algorithmic reasoning capabilities of large language models (LLMs). However, existing benchmarks primarily focus on full-information tasks where all problem inputs are provided upfront. This overlooks a critical dimension of algorithmic reasoning: the ability of generated programs to operate when key information is not revealed upfront. Interactive problems, a distinctive component of competitive programming, embody this challenge. These problems require programs to engage in multi-round interaction with an interactor (a judge program) under strict protocol constraints and limited query budgets, with new information revealed only in response to queries. To address this gap, we introduce InteractBench, a benchmark comprising 322 high-quality interactive problems curated from Codeforces, AtCoder, IOI, and ICPC. Each problem is packaged with executable local interactors, ena
    
[^319]: 推理模型思维链的忠实性随偏好线索的传递位置与方式而变化

    Chain-of-Thought Faithfulness of Reasoning Models Varies with Where and How Preference Cues Are Delivered

    [https://arxiv.org/abs/2608.29464](https://arxiv.org/abs/2608.29464)

    论文提出FACE-Eval评估基准，揭示推理模型的思维链忠实性取决于偏好线索的传递位置和显式程度——相比用户消息和显式线索，通过工具返回和隐式方式传递的偏好更容易被模型默默采纳而不在思维链中如实言明。

    

    思维链监测的前提假设是推理过程忠实地记录了影响模型回答的信息。现有的忠实性测试通常将显式的偏见线索置于用户消息中，而智能体在实际运行中可能通过工具返回结果或原始数据工件接触到偏好信息。我们提出了FACE-Eval（线索效应忠实归因评估），这是一个包含5,100个样本的评估基准，通过改变线索的位置（用户消息或工具返回结果）和显式程度（直接总结或原始数据工件）来系统考察这一问题。我们测量了遵循线索的回答中的言语化承诺，以及所有含线索样本中的未言语化采纳。我们评估了来自八个模型家族的15个开源权重模型，总参数量从4B到1.60T不等。结果显示：所有模型对工具返回线索的言语化承诺均低于用户消息线索，对隐式线索的言语化承诺均低于显式线索；此外，在全部15个模型上，工具返回线索导致的未言语化采纳率更高，在30个模型-通道对比中的28个里，隐式线索的未言语化采纳率也更高。

    arXiv:2608.29464v1 Announce Type: cross  Abstract: Chain-of-thought (CoT) monitoring assumes that reasoning traces faithfully record the information that shapes a model's answer. Existing faithfulness tests often place explicit bias cues in the user message, while agents may encounter preferences through tool returns or raw artifacts. We introduce FACE-Eval (Faithful Attribution of Cue Effects Evaluation), a 5,100-sample evaluation that varies cue location (user message or tool return) and explicitness (direct summary or raw artifact). We measure verbalized commitment among cue-following answers and unverbalized adoption among all cued samples. We evaluate 15 open-weight models from eight families, with total parameters ranging from 4B to 1.60T. Every model has lower verbalized commitment for tool-return than user-message cues and for implicit than explicit cues. Unverbalized adoption is higher for tool-return cues on all 15 models and for implicit cues in 28 of 30 model-channel compar
    
[^320]: 基于核心-扩展路由与统一计算调度的统一多模态模型加速

    Accelerating Unified Multimodal Models with Core-Expansion Routing and Unified Computation Scheduling

    [https://arxiv.org/abs/2608.29291](https://arxiv.org/abs/2608.29291)

    提出CE-Router框架，通过核心-扩展路由与统一计算调度（层跳过、FFN剪枝、扩散头缓存复用和去噪步提前退出）来消除统一多模态模型在理解与生成任务中的冗余计算，实现质量与效率的双重提升。

    

    统一多模态模型同时支持理解与生成任务，但在token、网络层和生成时间步之间产生了大量冗余计算。通过token重要性探测，我们识别出一种非对称的核心-扩展结构：理解任务表现出稳定的重要性成分，而生成任务在很大程度上共享这一成分，但需要依赖生成进度的修正。因此，我们提出CE-Router，它使用任务共享的核心评分器和基于生成进度的生成扩展，并通过生成任务分解与跨任务核心对齐进行优化。在推理阶段，CE-Router压缩token计算，并向统一计算调度提供学习到的路由信号，该调度协调层跳过、FFN剪枝、扩散头缓存复用以及去噪步提前退出。在两个代表性统一多模态模型架构上的实验表明，两项任务均取得了一致的质量-效率提升，并保留了……

    arXiv:2608.29291v1 Announce Type: new  Abstract: Unified multimodal models jointly support understanding and generation, but incur substantial redundant computation across tokens, layers, and generation timesteps. Through token-importance probing, we identify an asymmetric core-expansion structure: understanding exhibits a stable importance component, while generation largely shares this component but requires progress-dependent corrections. We therefore propose CE-Router, which uses a task-shared core scorer and progress-conditioned generation expansions, optimized through generation decomposition and cross-task core alignment. At inference, CE-Router compacts token computation and supplies a learned routing signal to Unified Computation Scheduling, which coordinates layer skipping, FFN pruning, diffusion-head cache reuse, and denoising-step early exit. Experiments on two representative UMM architectures demonstrate consistent quality--efficiency improvements across both tasks, retain
    
[^321]: FKG.in的验证：LLM增强的印度食品知识中的健全性评估

    Validating FKG.in: Soundness Assessment in LLM-Augmented Indian Food Knowledge

    [https://arxiv.org/abs/2608.29249](https://arxiv.org/abs/2608.29249)

    本文作为印度食品知识图谱FKG.in的一部分，提出了一种半自动化的健全性评估工作流程，通过结合形式文法、词汇检查、统计启发式、Set Transformer连贯性建模和检索验证的多阶段方法，识别并解决LLM从非正式烹饪来源提取和增强结构化食谱数据时的常见失败模式。

    

    在线烹饪生态系统中，由大型语言模型（LLM）生成、修改或总结的食谱内容日益增多。虽然这些输出通常看似合理，但可能包含虚构的食材、被误述的用量或文化上不合常理的食材组合，从而限制了其在下游应用和知识图谱构建中的适用性。在本文中，我们提出了一种半自动化的健全性评估工作流程，用于验证由LLM从非正式烹饪来源中提取和增强的结构化食谱数据。该流程作为印度食品知识图谱FKG.in的一部分开发而成，通过结合形式文法、基于词汇的检查、统计启发式方法、基于Set Transformer的连贯性建模以及基于检索的验证等多阶段流程，识别并解决常见的失败模式，包括结构性不一致、语义和逻辑上的不连贯以及与源文本的偏差。

    arXiv:2608.29249v1 Announce Type: new  Abstract: The online culinary ecosystem is increasingly populated by recipe content generated, modified, or summarized by Large Language Models (LLMs). While often plausible, such outputs may contain hallucinated ingredients, misrepresented quantities, or culturally implausible combinations, limiting their suitability for downstream applications and knowledge graph construction. In this paper, we present a semi-automated soundness assessment workflow for validating structured recipe data extracted and augmented by LLMs from informal culinary sources. Developed as part of FKG.in, a knowledge graph of Indian food, the pipeline identifies and addresses common failure modes, including structural inconsistencies, semantic and logical incoherence, and deviations from the source text, through a multi-stage process combining formal grammars, vocabulary-based checks, statistical heuristics, Set Transformer-based coherence modeling, and retrieval-based veri
    
[^322]: Hyper-Fold：通过超图建模探索蛋白质序列-几何学习的表达能力极限

    Hyper-Fold: Exploring the Expressive Limit of Sequence-Geometry Learning for Proteins via Hypergraph Modeling

    [https://arxiv.org/abs/2608.29207](https://arxiv.org/abs/2608.29207)

    提出 Hyper-Fold，通过超图建模将蛋白质序列内容与三维几何组织为超边，并利用分解为 K 个基算子的边条件化双线性矩阵值算子，以消息传递的计算代价逼近序列-几何学习的表达能力上限，在酶功能预测、折叠分类和配体结合位点检测等任务上表现出色。

    

    蛋白质结构建模建立在一个单一的计算原语之上：残基是什么（序列内容）与残基位于何处（三维几何）之间的相互作用。那么这一类层的表达能力极限是什么？我们证明，作用于内容-几何外积之上的完全双线性算子——即所有二阶交互的充分统计量——构成了表达能力的上限，而主流几何图神经网络（GNN）所采用的加性消息传递机制，在可证明的意义上对内容-几何耦合是“盲”的。随后，我们提出 Hyper-Fold，一种以消息传递的计算代价逼近该表达能力上限的秩-K 可分离卷积骨干网络：每个半径邻域被组织为一条序列超边和一条接触超边，并由一个边条件化的矩阵值算子进行调制，该算子被分解为 K 个可学习的基算子，其系数由几何信息生成。在酶功能预测、折叠类型分类以及配体结合位点检测等多项任务上，Hyper-Fold 展现了卓越的性能。

    arXiv:2608.29207v1 Announce Type: new  Abstract: Protein structure modeling rests on a single computational primitive: the interaction between what a residue is (sequence content) and where it sits (three-dimensional geometry). What is the expressive limit of this layer class? We show that the complete bilinear operator over content-geometry outer products--the sufficient statistic of all second-order interactions--is the expressive ceiling, while the additive message passing of mainstream geometric GNNs is provably blind to content-geometry binding. We then introduce Hyper-Fold, a rank-K separable convolutional backbone approaching this ceiling at message-passing cost: each radius neighborhood is organized into a sequence hyperedge and a contact hyperedge, modulated by an edge-conditioned matrix-valued operator factorized into K learned basis operators with geometry-generated coefficients. Across enzyme function prediction, fold classification, and ligand binding site detection, Hyper
    
[^323]: 自动化研究人员能够可靠地缓解对齐失败

    Automated Researchers Can Reliably Mitigate Alignment Failures

    [https://arxiv.org/abs/2608.28945](https://arxiv.org/abs/2608.28945)

    自动化对齐研究员（AAR）通过后训练方法能够可靠地缓解10种对齐失败并泛化到更大的模型，其效果甚至优于28名经验丰富的人类研究员在八小时内开发的方法。

    

    自动化对齐研究可能会加速实现与人类对齐的AI的进程，但这是否真的有效却难以衡量。幸运的是，许多对齐失败，例如欺骗、谄媚和越狱，已经可以通过公开基准来衡量。我们研究了自动化对齐研究员能否通过后训练来缓解对齐失败，方法是提出训练方法和数据，以同时优化多个安全基准，同时保持通用能力。在10种对齐失败中，最强的AAR方法显著减少了目标对齐失败，并能泛化到留出的基准测试、多轮行为审计，以及比目标模型大4.7倍的模型。作为人类基线，28名经验丰富的研究人员获得了最多八小时的时间来为相同的基准开发方法，但他们的方法表现不如最好的AAR方法。将人类想法作为AAR的初始研究方向并不能改善结果。

    arXiv:2608.28945v1 Announce Type: new  Abstract: Automating alignment research may accelerate progress toward aligned AI, but whether it does is hard to measure. Luckily, many alignment failures, such as deception, sycophancy, and jailbreaks, are already measurable by public benchmarks. We study whether automated alignment researchers (AARs) can post-train to mitigate alignment failures by proposing training methods and data to simultaneously optimize multiple safety benchmarks, while preserving general capability. Across 10 alignment failures, the strongest AAR methods significantly reduce the targeted alignment failures and generalize to a held-out benchmark, multi-turn behavioral audits, and models up to 4.7 times larger than the target model. As a human baseline, 28 experienced researchers receive up to eight hours to develop methods for the same benchmarks, but their methods underperform the best AAR methods. Using human ideas as the AARs' initial research direction does not impro
    
[^324]: AutoScientist-Quant：面向量化投资自动化研究的自进化编码智能体

    AutoScientist-Quant: Self-Evolving Coding Agents for Automatic Research in Quantitative Investment

    [https://arxiv.org/abs/2608.28632](https://arxiv.org/abs/2608.28632)

    提出AutoScientist-Quant框架，将量化研究建模为预算约束下的搜索问题，通过单一自进化控制器统一决策Alpha生成、因子库选择和模型调优，实现从假设到可部署策略的全流程自动化，并修复了评估流程中的前视偏差问题。

    

    大语言模型智能体能够发现Alpha因子，然而现有方法存在三个弱点：搜索过程无法在运行中自适应调整；自动化通常止步于Alpha生成，而因子库选择和模型选择仍需人工完成；Alpha发现过程可能通过循环反馈或代码问题窥探到测试窗口。我们提出AutoScientist-Quant，一个自进化的搜索过程，将量化研究视为一个受预算约束的搜索问题。单一控制器基于剩余预算对所有决策进行条件化，在每一轮决定是改进、组合、转向还是停止，选择扩展哪个节点，生成多少个Alpha，以及如何从共享记忆中检索历史轨迹。同一核心随后从因子库中进行选择并调整模型，实现了从假设到可部署策略的完整闭环。我们还审查了从先前工作复用的评估流程，修复了两个前视偏差问题，并保持反馈窗口与测试窗口互不相交。

    arXiv:2608.28632v1 Announce Type: new  Abstract: Large language model agents can discover alphas, yet current methods have three weaknesses. The search cannot adapt during the run, automation usually ends at alpha generation while library selection and model choice stay manual, and alpha discovery can read the test window through loop feedback or code problems. We present AutoScientist-Quant, a self evolving search process that regards quantitative research as one budgeted search problem. A single controller conditions every decision on the remaining budget, choosing at each round whether to improve, combine, pivot, or stop, which node to expand, how many alphas to generate, and how to retrieve past trajectories from the shared memory. The same core then selects from the library and tunes the model, closing the loop from hypothesis to deployable strategy. We also review the evaluation pipeline reused from prior work, fix two lookahead problems, and keep the feedback window disjoint fro
    
[^325]: 表演性隐私：差分隐私何时能最大化效用

    Performative Privacy: When Differential Privacy Maximizes Utility

    [https://arxiv.org/abs/2608.28198](https://arxiv.org/abs/2608.28198)

    该论文提出“表演性隐私”新框架，首次形式化了隐私保护与用户参与度之间的动态关系，并证明当数据泄露导致用户流失时，采用有限隐私预算的差分隐私机制在长期内可以优于非隐私估计。

    

    保护隐私的学习通常源于这样一种理念：保护用户数据可以维持信任，从而保持用户参与，进而在长期内提升效用。然而，这一论点迄今为止尚未被形式化。与此同时，表演性学习为研究部署行为会影响其后续观测数据的学习系统提供了一个框架。在本工作中，我们将这两种视角结合起来，提出了“表演性隐私”的概念，即数据泄露会降低未来的用户参与度。我们研究了一个简单模型：智能体反复贡献数据用于均值估计，但当其数据被泄露时可能会退出系统。隐私通过差分隐私机制来实现，从而在估计噪声与未来参与度之间形成权衡。通过对该动态过程的理论研究和数值实验，我们证明了在某些条件下，有限的隐私预算在长期内可以优于非隐私估计。

    arXiv:2608.28198v1 Announce Type: new  Abstract: Privacy-preserving learning is often motivated by the idea that protecting users' data can preserve trust and thus participation, improving utility in the long term. However, this claim has not been formalized so far. In parallel, performative learning provides a framework for studying learning systems whose deployment affects the data they later observe. In this work, we bring these two perspectives together and introduce \emph{performative privacy}, where data leakage reduces future participation. We study a simple model where agents repeatedly contribute data for mean estimation but may leave the system when their data is leaked. Privacy is implemented through differentially private mechanisms, creating a trade-off between estimation noise and future participation. We show, through a theoretical study of the dynamics and numerical experiments, that a finite privacy budget can outperform non-private estimation in the long term when the
    
[^326]: 通过博弈论视角审视AI对齐：综述

    AI Alignment through a Game-theoretic Lens: A Survey

    [https://arxiv.org/abs/2608.27910](https://arxiv.org/abs/2608.27910)

    本综述以博弈论视角系统梳理AI对齐研究，围绕偏好多样性、对齐优先级和时间动态三大挑战组织文献，阐明了博弈论分析真正发挥作用之处以及构建鲁棒、自适应、可验证AI系统仍待解决的难题。

    

    随着大语言模型和日益强大的AI智能体被部署到高风险场景中，使其与复杂的人类价值观保持一致已成为核心挑战。现有的对齐方法虽然在提升有用性、无害性和可控性方面卓有成效，但往往难以捕捉那些依赖于上下文、不具传递性、并由动态多方交互塑造的真实世界偏好。本综述通过博弈论的视角审视AI对齐研究。具体而言，它围绕关键的博弈论要素组织近期进展，并围绕三大挑战综合梳理相关文献：偏好多样性、对齐优先级和时间动态。这一视角阐明了当前对齐方法在哪些方面真正受益于博弈论分析，哪些方面的框架应用较为宽松，以及在构建鲁棒、自适应、可验证的AI系统方面仍面临哪些挑战。

    arXiv:2608.27910v1 Announce Type: cross  Abstract: As large language models and increasingly capable AI agents are deployed in high-risk settings, aligning them with complex human values has become a central challenge. Existing alignment methods, while effective in improving helpfulness, harmlessness, and controllability, often struggle to capture real-world preferences that are context-dependent, non-transitive, and shaped by dynamic multi-party interactions. This survey reviews AI alignment through a game-theoretic lens. Specifically, it organizes recent progress around key game-theoretic elements and synthesizes the literature along three challenges: preference diversity, alignment priority, and temporal dynamics. This perspective clarifies where current alignment methods genuinely benefit from game-theoretic analysis, where the framework is looser, and what challenges remain in building robust, adaptive, and verifiable AI systems.
    
[^327]: 连续容量增长：基于任务复杂度的宽度与深度扩展用于JEPA世界模型中的视觉Transformer编码器

    Successive Capacity Growth: Task-Complexity-Driven Width and Depth Expansion for Vision Transformer Encoders in JEPA World Models

    [https://arxiv.org/abs/2608.27367](https://arxiv.org/abs/2608.27367)

    本文提出一种基于任务复杂度的动态扩展方法，通过函数保持的测试与验证机制和正则化技术，使JEPA世界模型中的视觉Transformer编码器从最小规模逐步增长，以高效适应不同复杂度任务。

    

    用于世界建模的联合嵌入预测架构（JEPA）通常采用固定大小的视觉Transformer编码器，这些编码器对于简单任务过度配置，对于复杂任务配置不足，并且在注意力头之间存在显著冗余。我们提出连续容量增长（SCG）方法，该方法从最小编码器（1个头、2层、283K参数）开始，通过宽度扩展（增加注意力头以提升低级语义容量）或深度扩展（增加Transformer块以实现高阶语义抽象）逐步增长，并由任务无关的测试与验证机制驱动，该机制利用函数保持扩展安全地试验架构变更，如果预测损失未改善则回滚。素描各向同性高斯正则化器（SIGReg）确保所有学习到的语义维度保持统计独立并与预测目标对齐，即使在架构扩展过程中也能防止崩溃。

    arXiv:2608.27367v1 Announce Type: cross  Abstract: Joint-Embedding Predictive Architectures (JEPAs) for world modeling typically employ fixed-size Vision Transformer encoders that are over-provisioned for simple tasks and under-provisioned for complex ones, with significant redundancy across attention heads. We propose Successive Capacity Growth (SCG), a method that starts from a minimal encoder (1 head, 2 layers, 283K parameters) and grows incrementally in width (adding attention heads for low-level semantic capacity) or depth (adding transformer blocks for higher-order semantic abstraction), driven by a task-agnostic test-and-verify mechanism that exploits function-preserving expansion to safely trial architectural changes and roll back if they do not improve prediction loss. The Sketched Isotropic Gaussian Regularizer (SIGReg) ensures that all learned semantic dimensions remain statistically independent and aligned with the predictive objective, preventing collapse even as the archi
    
[^328]: pro-team在LLMs4OL 2026任务旗舰与重用：检索增强生成与词汇约束过滤用于本体学习

    pro-team at LLMs4OL 2026 Tasks Flagship and Reuse: Retrieval-Augmented Generation and Vocabulary-Constrained Filtering for Ontology Learning

    [https://arxiv.org/abs/2608.27101](https://arxiv.org/abs/2608.27101)

    本文提出了一种结合检索增强生成与词汇约束过滤的管道，有效解决本体学习中的幻觉和格式问题，在LLMs4OL 2026挑战赛中同时优化了端到端和重用任务。

    

    尽管大型语言模型（LLMs）取得了显著进展，但从文本中进行本体学习仍然具有挑战性，因为模型可能产生幻觉域术语、生成不一致的格式，并偏向于层次关系而非关联关系。在LLMs4OL 2026挑战赛中，我们使用离线检索增强的少样本提示管道，同时处理端到端旗舰任务（任务A）和本体扩展重用任务（任务B）。我们的系统采用Qwen2.5-14B-Instruct和all-MiniLM-L6-v2进行演示检索，为任务A选择前5个示例，为任务B选择前2个示例。一种左截断的上下文窗口策略在长提示中保留任务指令。对于任务B，生成的三元组经过确定性词汇约束过滤，当至少一个端点属于样本的封闭术语/类型词汇时保留三元组，并移除初始本体的重复项。该方法实现了语义图相似度0。

    arXiv:2608.27101v1 Announce Type: new  Abstract: Ontology learning from text remains challenging despite significant progress in Large Language Models (LLMs), which can hallucinate domain terms, produce inconsistent formats, and favor hierarchical over associative relations. In the LLMs4OL 2026 Challenge, we address both the End-to-End Flagship Task (Task A) and Ontology Extension Reuse Task (Task B) using an offline retrieval-augmented few-shot prompting pipeline. Our system employs Qwen2.5-14B-Instruct with all-MiniLM-L6-v2 for demonstration retrieval, selecting the top-5 examples for Task A and top-2 for Task B. A left-truncated context-windowing strategy preserves task instructions within long prompts. For Task B, generated triples undergo deterministic vocabulary-constrained filtering, retaining triples when at least one endpoint belongs to the sample's closed term/type vocabulary and removing duplicates of the initial ontology. The approach achieves Semantic Graph Similarity of 0
    
[^329]: 人工实验者：通过自指强化学习发现与控制自组织现象

    The Artificial Experimentalist: Discovery and Control of Self-Organizing Phenomena with Autotelic Reinforcement Learning

    [https://arxiv.org/abs/2608.26116](https://arxiv.org/abs/2608.26116)

    本文提出一种基于自指强化学习的闭环框架CARL，能够自主发现并控制Lenia中的自组织孤子现象，显著优于开环启发式方法。

    

    arXiv:2608.26116v1 公告类型：新 摘要：现有的探索元胞自动机和其他复杂系统的方法大多以开环方式运行：它们设置初始条件，执行完整模拟，并观察结果，而不在运行过程中进行干预。我们引入了一种基于自指强化学习的闭环框架，其中代理自主采样多样化的目标，并学习一种目标条件策略，通过最小、局部的扰动来干预复杂系统。我们将此框架实例化于Lenia（一种以类生命自组织模式著称的连续元胞自动机）中，构建了一个名为CARL的代理系统，并展示了三种能力。首先，CARL在广泛的Lenia更新规则中发现稳定孤子的速率高于启发式基线。其次，它学会用少量干预引导现有孤子的运动方向，表明CARL不仅能创造自组织模式，还能控制它们。

    arXiv:2608.26116v1 Announce Type: new  Abstract: Existing methods for exploring cellular automata and other complex systems mostly operate in open loop: they set initial conditions, execute a full simulation, and observe the outcome, without intervening during execution. We introduce a closed-loop framework based on autotelic reinforcement learning, in which an agent autonomously samples diverse goals and learns a goal-conditioned policy to intervene in a complex system through minimal, local perturbations. We instantiate this framework on Lenia, a continuous cellular automaton known for life-like self-organizing patterns, in an agentic system we call CARL, and demonstrate three capabilities. First, CARL discovers stable solitons across a wide range of Lenia update rules at a higher rate than heuristic baselines. Second, it learns to steer the movement direction of existing solitons with few interventions, showing that CARL can control self-organizing patterns, not only create them. Th
    
[^330]: RACE：大规模统计估计大型语言模型神经元功能一致性的方法

    RACE: Scalable Statistical Estimation of Functional Consistency in LLM Neurons

    [https://arxiv.org/abs/2608.24758](https://arxiv.org/abs/2608.24758)

    RACE是一种前向传播统计框架，通过残差对齐高效估计Transformer神经元的领域级功能一致性，具有高领域特异性和比梯度方法低两个数量级的计算开销。

    

    arXiv:2608.24758v1 公告类型：新 摘要：在整个领域中发现稳定的神经元行为仍然是机制可解释性中的一个挑战。现有方法通常依赖于实例级别的点估计或计算成本高昂的程序，这些方法要么掩盖了群体级别的变异性，要么限制了可扩展的领域级分析。我们提出了RACE（残差对齐用于一致性估计），这是一种前向传播统计框架，用于评估Transformer神经元的领域级功能一致性。扰动实验表明，与基于梯度的点估计相比，RACE实现了更高的领域特异性。同时，令牌分布级别的结果验证了所选神经元与目标领域之间的关联。此外，其计算开销比基于梯度的方法低两个数量级。

    arXiv:2608.24758v1 Announce Type: new  Abstract: Discovering stable neuron behavior across entire domains remains a challenge in mechanistic interpretability. Existing methods often rely on instance-level point estimates or computationally expensive procedures, which either obscure population-level variability or limit scalable domain-wide analysis. We present RACE (Residual Alignment for Consistency Estimation), a forward-pass statistical framework that evaluates the domain-wide functional consistency of Transformer neurons. Perturbation experiments demonstrate that RACE achieves superior domain specificity compared to gradient-based point estimates. Meanwhile, token-distribution-level results verify the association between the selected neurons and the target domain. Furthermore, its computational overhead is two orders of magnitude lower than that of gradient-based methods.
    
[^331]: HiDiffTIR：面向多轮工具集成推理的分层难度感知策略优化

    HiDiffTIR: Hierarchical Difficulty-Aware Policy Optimization for Multi-Turn Tool-Integrated Reasoning

    [https://arxiv.org/abs/2608.21863](https://arxiv.org/abs/2608.21863)

    本文提出HiDiffTIR框架，通过分层难度感知的信用分配机制，在多轮工具集成推理中更精确地区分轨迹和推理步骤的难度，从而提升强化学习训练效果。

    

    arXiv:2608.21863v1 公告类型：交叉 摘要：工具集成推理（TIR）是LLM代理通过与外部工具迭代交互解决复杂任务的基本能力。强化学习（RL）已成为实现这一能力的主导范式。然而，现有方法通常分配统一的轨迹级优势，并平等对待所有正确的工具调用，忽略了轨迹和推理步骤间不同的难度和学习价值。这可能导致学习信号不精确，无法充分区分平凡和具有挑战性的工具使用模式。为解决这一局限性，我们提出了HiDiffTIR，一种用于多轮TIR的分层难度感知策略优化框架。HiDiffTIR在轨迹级和回合级执行难度感知的信用分配，使策略能够聚焦于更具信息量的轨迹和更难的推理步骤。值得注意的是，这种细粒度优化是通过...

    arXiv:2608.21863v1 Announce Type: cross  Abstract: Tool-Integrated Reasoning (TIR) is a fundamental capability for LLM agents to solve complex tasks by interacting with external tools iteratively. Reinforcement Learning (RL) has become the dominant paradigm for enabling this capability. However, existing approaches typically assign uniform trajectory-level advantages and treat all correct tool calls equally, ignoring the varying difficulty and learning value across trajectories and reasoning steps. This can lead to imprecise learning signals that do not adequately distinguish between trivial and challenging tool-use patterns. To address this limitation, we propose HiDiffTIR, a Hierarchical Difficulty-aware policy optimization framework for multi-turn TIR. HiDiffTIR performs difficulty-aware credit assignment at both trajectory and turn levels, enabling the policy to focus on more informative trajectories and harder reasoning steps. Notably, this fine-grained optimization is achieved wi
    
[^332]: 神经基元：一种基于基元模仿学习的高效端到端局部规划器用于自主飞行

    Neural-Primitive: An Efficient End-to-end Local Planner with Primitive-based Imitation Learning for Autonomous Flight

    [https://arxiv.org/abs/2608.20948](https://arxiv.org/abs/2608.20948)

    本文提出一种基于基元模仿学习的端到端局部规划器，通过轻量级离线数据集和紧凑神经网络直接生成轨迹，实现超低延迟和低内存的实时自主飞行。

    

    在未知杂乱环境中的自主飞行受到机载轨迹生成的计算-质量-内存三难问题的阻碍。本文提出了一种通过模仿学习的高效端到端局部规划器。设计了一个轻量级的基于离线基元的数据集收集框架，用于在非凸环境中生成安全且高质量的轨迹基元。一个紧凑的神经网络直接将感官输入映射到多项式系数，这些系数固有地编码了高阶动力学信息。学习到的策略实时生成平滑、经验上无碰撞且动态可行的轨迹，无需后端求解。它实现了超快速计算（标准桌面低于1ms，机载飞行平均3.68ms），同时保持低机载内存需求（小于1.5MiB）。广泛的仿真基准测试证明了在规划延迟和目标达成方面的优越性。

    arXiv:2608.20948v1 Announce Type: cross  Abstract: Autonomous flight in unknown cluttered environments is hindered by the computation-quality-memory trilemma of onboard trajectory generation. In this paper, we propose an efficient end-to-end local planner via imitation learning. A lightweight offline-primitive-based dataset collection framework is designed to produce safe and high-quality trajectory primitives in non-convex environments. A compact neural network directly maps sensory inputs to polynomial coefficients that inherently encode higher-order dynamical information. The learned policy generates smooth, empirically collision-free and dynamically feasible trajectories in real time without back-end solving. It achieves ultra-fast computation (below 1ms on a standard desktop and average 3.68ms during onboard flight), while maintaining low onboard memory requirements (less than 1.5MiB). Extensive simulation benchmarks demonstrate superiority in both planning latency and target-reac
    
[^333]: 电子航海图变更分类

    Electronic Navigational Chart Change Classification

    [https://arxiv.org/abs/2608.20218](https://arxiv.org/abs/2608.20218)

    本文提出了一种自动分类电子航海图变更的方法，通过建立基线编码方案将复杂矢量数据转换为结构化表格，并利用空间上下文编码器提升分类准确性，以解决传统人工审查效率低和一致性差的问题。

    

    arXiv:2608.20218v1 公告类型：新 摘要：电子航海图（ENCs）是用于海上导航系统的地理空间矢量数据集，表示水文和导航信息，如深度、助航设施、交通方案和危险物。水文办公室面临的一个主要挑战是判断给定的海图变更是否对海上安全构成关键或非关键风险。现有工作流程严重依赖人工审查和验证，这不仅劳动密集、难以应对海图更新的大量涌入，而且导致分析员之间的一致性差异。为应对这一挑战，我们提出了一种自动分类ENC变更的方法。我们建立了一个基线编码方案，将复杂的矢量数据变更转换为结构化的表格格式，以供分类模型使用。该编码方案的两个关键组成部分包括一个空间上下文编码器，用于用周围地理信息丰富变更表示。

    arXiv:2608.20218v1 Announce Type: new  Abstract: Electronic Navigational Charts (ENCs) are geospatial vector datasets used in maritime navigation systems that represent hydrographic and navigational information such as depths, navigational aids, traffic schemes, and hazards. A major challenge for hydrographic offices is determining whether a given chart change poses a critical or non-critical risk to maritime safety. Existing workflows rely heavily on manual review and verification, which is labor-intensive, scales poorly with the volume of incoming chart updates, and introduces inter-analyst inconsistencies. To address this challenge, we propose a method for automated classification of ENC changes. We establish a baseline encoding scheme to translate complex vector data changes into a structured tabular format for classification models. The two crucial components of the encoding scheme include a spatial context encoder to enrich the change representations with surrounding geographic f
    
[^334]: 可验证弃权使供水管网中的AI泄漏诊断更具问责性

    Verifiable abstention makes AI leak diagnosis accountable in water distribution networks

    [https://arxiv.org/abs/2608.18836](https://arxiv.org/abs/2608.18836)

    本文提出一种基于可验证弃权的AI泄漏定位框架，通过物理执行代理和LLM审计监督代理的协作，在不行动时明确弃权，从而在保持高决策精度的同时显著提升系统问责性。

    

    arXiv:2608.18836v1 公告类型：新 摘要：公用事业公司因泄漏损失大量处理过的水，但很少信任人工智能定位器派遣维修队：到处猜测无法为挖掘提供理由。差距在于问责性，而非准确性：没有方法能证明其何时不应行动。在此，我们将泄漏定位重新定义为在可验证弃权下的决策问题。一个基于物理的执行代理针对数字孪生体对假设（泄漏、需求、传感器、阀门）进行证伪；一个独立的监督代理，配备大型语言模型（LLM）审计器，根据代码可验证的合同检查证据，然后认证派遣、请求证据或弃权。在现场级噪声下，32%的强制基线在已行动事件上提升至96%的决策精度。在独立生成的基准上，它仅对33个泄漏中的4个采取行动，且全部正确。一个包含194个已审计真实泄漏位置及孪生模拟压力和流量的注册表，产生五次挖掘派遣，其中三次

    arXiv:2608.18836v1 Announce Type: new  Abstract: Utilities lose a substantial share of treated water to leakage, yet rarely trust artificial-intelligence localizers to dispatch crews: guessing everywhere cannot justify excavation. The gap is accountability, not accuracy: no method proves when it should not act. Here we recast leak localization as decision-making under verifiable abstention. A physics-grounded executor agent falsifies hypotheses (leak, demand, sensor, valve) against a digital twin; an independent supervisor agent, with a large-language-model (LLM) auditor, checks evidence against a code-verifiable contract, then certifies a dispatch, requests evidence or abstains. Under field-grade noise, a 32% forced baseline becomes 96% decision precision on acted events. On an independently generated benchmark it acts on only 4 of 33 leaks, all correct. A 194-event register of audited real leak locations with twin-simulated pressures and flows yields five excavation dispatches, three
    
[^335]: 大型推荐解释中LLM即评判者的生命周期

    The Lifecycle of LLM-as-a-Judge for Large-Scale Recommendation Explanations

    [https://arxiv.org/abs/2608.18300](https://arxiv.org/abs/2608.18300)

    本文提出LLM评判者在生产系统中具有构建、训练、部署和持续维护的生命周期，并以Netflix推荐解释评估为例，强调其动态维护而非静态评估的重要性。

    

    arXiv:2608.18300v1 公告类型：新 摘要：LLM即评判者（LLM-as-a-Judge）利用大型语言模型来评估由另一个AI应用或模型生成的自然语言，已成为一种标准且可扩展的方法，用于加速和扩展昂贵的人工评估。然而，大多数工作将评判者视为静态产物，仅在构建时或针对固定基准进行一次评估。相反，我们认为，在生产系统中运行的LLM评判者应被理解为一个具有生命周期的实体：它必须被构建、训练、部署，并随着周围数据的演变而持续维护，每个阶段都面临独特的技术和运营挑战。我们展示了Netflix中用于评估面向用户的推荐解释的LLM评判者的这种生命周期，在我们的流程中，每周生成并评估数十万个不同节目级别的解释，并通过移动体验服务数百万会员。我们的框架包含四个阶段。

    arXiv:2608.18300v1 Announce Type: new  Abstract: LLM-as-a-Judge, which leverages a large language model to evaluate natural language generated by another AI application or model, has become a standard, scalable approach for accelerating and extending costly human evaluation. However, most work treats a judge as a static artifact, evaluating it once at construction or against a fixed benchmark. In contrast, we argue that an LLM judge running in a production system is better understood as having a lifecycle: it must be built, trained, deployed, and continuously maintained as the surrounding data evolves, and each phase poses distinct technical and operational challenges.   We present such a lifecycle for the LLM judges that evaluate user-facing recommendation explanations at Netflix, where our pipeline generates and the judges assess hundreds of thousands of distinct show-level explanations per week, served across the mobile experience to millions of members. Our framework has four phase
    
[^336]: 无金标准标签下AI生成数据的去偏推断：通过多重不完美测量进行识别

    Debiased Inference for AI-Generated Data without Gold-Standard Labels: Identification via Multiple Imperfect Measurements

    [https://arxiv.org/abs/2608.18294](https://arxiv.org/abs/2608.18294)

    本文提出了一种无需金标准标签、利用多重不完美AI测量进行去偏推断的新框架，有效解决了AI测量误差导致的下游分析偏差问题。

    

    越来越多的学者使用AI来测量变量，并将其纳入后续的下游分析。尽管AI测量的变量通常被视为无误差观测，但忽略自动化测量中的预测误差会导致下游分析中的显著偏差和无效置信区间，即使AI测量准确度很高（例如超过90%）。现有的解决方案，如基于设计的有监督学习和预测支持推断，将基于AI的易错测量与金标准标签相结合，但在某些应用领域中，获取金标准标签可能成本高昂且困难。在本文中，我们提出了多重不完美测量的去偏推断（DMM），这是一个结合多个易错AI测量以实现无需金标准标签的有效下游推断的框架。基于CP分解的既有成果，DMM假设这些测量是独立的。

    arXiv:2608.18294v1 Announce Type: cross  Abstract: An increasing number of scholars use AI to measure variables they subsequently include in downstream analyses. Although AI-measured variables are often analyzed as if observed without error, ignoring prediction errors in automated measurement leads to substantial bias and invalid confidence intervals in downstream analyses, even if AI measurement accuracy is high, e.g., above 90%. Existing solutions, such as design-based supervised learning and prediction-powered inference, combine error-prone AI-based measurements with gold-standard labels, which may be costly and difficult to obtain in some application areas.   In this paper, we propose debiased inference with multiple imperfect measurements (DMM), a framework that combines multiple error-prone AI measurements to enable valid downstream inference without gold-standard labels. Building on the established results on CP decomposition, DMM assumes that these measurements are independent 
    
[^337]: 支持推理的汽车电子/电气组件鲁棒性验证

    Reasoning-supported Robustness Validation of Automotive E/E Components

    [https://arxiv.org/abs/2608.16421](https://arxiv.org/abs/2608.16421)

    本文提出了一种基于本体和OWL形式化的方法，通过将任务剖面映射为语义表示，自动化汽车电子组件鲁棒性验证过程，显著提高了分析选择和决策支持的效率与可靠性。

    

    摘要：arXiv:2608.16421v1 公告类型：新 摘要：本文提出了一种基于本体的方法，以应对汽车电气/电子（E/E）组件鲁棒性验证（RV）过程的复杂性。该方法利用了来自RV过程以及应力、运行和负载剖面（即所谓的任务剖面，MPs）的形式化知识。与工业上建立的容易出错的手动流程相比，我们展示了如何在OWL中形式化组件特性，以构成RV过程中高效自动化分析选择和决策支持的基础。所提出的方法基于将任务剖面映射到OWL表示的思路，从而能够对MP数据进行语义查询，以改善其在RV过程中的集成。所得到的本体支持的应用框架已应用于汽车电力电子领域的工业用例。我们提供的实验结果表明，RV过程可以显著改进。

    arXiv:2608.16421v1 Announce Type: new  Abstract: This paper presents an ontology-supported approach to tackle the complexity of the Robustness Validation (RV) process of automotive electrical/electronic (E/E) components. The approach uses formalized knowledge from the RV process and stress, operating, and load profiles, so-called Mission Profiles (MPs). In contrast to the error-prone industrially established manual procedure, we show how component characteristics are formalized in OWL in order to form the foundation of an efficient automated analysis selection and decision support during the RV process. The proposed approach is based on the idea of mapping MPs to an OWL representation so to allow to perform semantic queries against MP data to improve their integration into the RV process. The resulting ontology-supported application framework has been applied to an industrial use-case from automotive power electronics. We present experimental results showing that the RV process can be 
    
[^338]: 深度思维对齐：用于视频推理的轨迹级潜在蒸馏

    Deep Thought Alignment: Trajectory-Level Latent Distillation for Video Reasoning

    [https://arxiv.org/abs/2608.16316](https://arxiv.org/abs/2608.16316)

    本文提出Latent-OPD方法，通过在轨迹末端进行潜在表示蒸馏，弥补了传统输出级蒸馏在视频推理中无法直接约束中间推理状态的不足，从而提升小模型从大模型迁移推理能力的效率。

    

    大型多模态模型（LMMs）在视频推理中一直受到处理海量视觉信息的高计算成本的阻碍。这一困境促使将大模型的推理能力转移到更小、更高效的模型上。策略内蒸馏（OPD）通过匹配学生生成轨迹上的输出令牌分布，提供了一种有前景的解决方案。然而，视频推理通常依赖于跨多个帧累积的证据。在此背景下，输出级监督仅捕捉通过令牌预测表达的信息，并未直接约束推理过程中形成的潜在表示。为解决这一局限性，我们提出了Latent-OPD，该方法通过轨迹级潜在蒸馏增强了OPD。具体而言，我们的方法聚焦于每条轨迹结束时的位置，其中隐藏状态有效地总结了累积的视觉证据。

    arXiv:2608.16316v1 Announce Type: cross  Abstract: Large Multimodal Models (LMMs) for video reasoning have long been hindered by the high computational cost of processing vast amounts of visual information. This dilemma motivates the transfer of the reasoning capabilities of large models to smaller, more efficient ones. On-Policy Distillation (OPD) offers a promising solution by matching output-token distributions along student-generated trajectories. However, video reasoning often depends on evidence accumulated across multiple frames. In this context, output-level supervision only captures information expressed through token predictions and does not directly constrain the latent representations formed during reasoning. To address this limitation, we propose Latent-OPD, which augments OPD with trajectory-level latent distillation. Specifically, our method focuses on the position at the end of each trajectory, where hidden states effectively summarize the accumulated visual evidence an
    
[^339]: MicroEvo：知识引导的大语言模型采样实现高效微架构设计空间探索

    MicroEvo: Knowledge-Guided LLM Sampling for Efficient Microarchitecture Design Space Exploration

    [https://arxiv.org/abs/2608.06183](https://arxiv.org/abs/2608.06183)

    MicroEvo是一个知识引导的微架构优化框架，将现成LLM与蒙特卡洛树搜索结合，通过LLM驱动的进化算子、帕累托感知树策略、主动知识积累机制和状态感知指令实现高效的多目标设计空间探索，帕累托前沿质量相比NSGA-II提升高达36.2%。

    

    微架构设计空间探索面临着庞大的搜索空间和高昂的PPA（功耗、性能、面积）评估成本，使得设计决策只能使用极为有限的仿真预算。现有方法在进行盲目搜索时未考虑微架构间的依赖关系，且无法从迭代搜索过程中有效学习，导致评估资源浪费和帕累托收敛性差。在本文中，我们提出了MicroEvo，这是一个知识引导的框架，将现成的大语言模型（LLM）与蒙特卡洛树搜索（MCTS）相结合，用于多目标微架构优化。MicroEvo融合了LLM驱动的进化算子、平衡帕累托贡献与多样性的帕累托感知树策略、提取并复用优化洞察的主动知识积累机制，以及在线自适应调整搜索行为的状态感知指令。实验表明，MicroEvo将帕累托前沿质量相比NSGA-II提升高达36.2%，并且

    arXiv:2608.06183v2 Announce Type: replace  Abstract: Microarchitecture design space exploration suffers from expansive search spaces and expensive PPA evaluation, leaving only a small simulation budget for design decision-making. Existing methods perform blind search without considering microarchitectural dependencies and fail to learn from the iterative search effectively, leading to wasted evaluations and weak Pareto convergence. In this paper, we propose MicroEvo, a knowledge-guided framework that couples off-the-shelf LLMs with Monte Carlo Tree Search (MCTS) for multi-objective microarchitecture optimization. MicroEvo combines LLM-driven evolutionary operators, a Pareto-aware tree policy that balances Pareto contribution and diversity, an active knowledge accumulation mechanism that extracts and reuses optimization insights, and state-aware directives that adapt the search behavior online. Experiments show that MicroEvo improves Pareto-front quality by up to 36.2% over NSGA-II and 
    
[^340]: 推理大语言模型中的测试时扩展：推理机制、评估与可复现性

    Test-Time Scaling in Reasoning LLMs: Inference Regimes, Evaluation, and Reproducibility

    [https://arxiv.org/abs/2608.04001](https://arxiv.org/abs/2608.04001)

    该论文将大语言模型的测试时扩展形式化为隐式前缀树上的预算约束推理，系统区分了三种推理机制（单轨迹顺序扩展、叶节点级扩展与前缀级扩展），并主张以完整推理系统作为评估对象，以提升研究结果的可比性与可复现性。

    

    大语言模型可以通过更多的推理时计算来解决更难的推理问题。然而，“测试时扩展”这一术语涵盖了多种推理算法：沿单条轨迹延长思考过程、采样完整候选答案并通过投票或验证进行聚合，以及在部分状态上进行搜索。这些算法在统计结构、计算需求和失败模式上各不相同。将它们视为在标量“预算”下可以互换，或者在报告准确率时不指明推理协议，会使研究结果难以跨研究进行比较。我们从三个维度研究测试时扩展。首先，我们将其形式化为在自回归模型隐式前缀树上进行的预算约束推理，并区分单轨迹顺序扩展、带终端归约的叶节点级扩展以及前缀级扩展。其次，我们将完整的推理系统作为评估对象，并区分（注：原文摘要在此处被截断）

    arXiv:2608.04001v2 Announce Type: replace-cross  Abstract: Large language models can solve harder reasoning problems with more inference-time compute. The term "test-time scaling," however, covers several inference algorithms: extending deliberation along one trajectory, sampling completed candidates and aggregating them by voting or verification, and searching over partial states. These algorithms differ in statistical structure, compute requirements, and failure modes. Treating them as interchangeable under a scalar "budget," or reporting accuracy without specifying the inference protocol, makes results difficult to compare across studies. We study test-time scaling along three axes. First, we formalize it as budgeted inference over the implicit prefix tree of an autoregressive model and distinguish single-trajectory sequential scaling, leaf-level scaling with terminal reduction, and prefix-level scaling. Second, we treat the full inference system as the evaluated object and separate
    
[^341]: 当Oracle条件误导部署时：超声心动图分割中的条件可用性偏差

    When Oracle Conditioning Misleads Deployment: Conditioning-Availability Bias in Echocardiographic Segmentation

    [https://arxiv.org/abs/2608.03342](https://arxiv.org/abs/2608.03342)

    该论文揭示了相位条件化超声心动图分割中的“条件可用性偏差”——用干净的Oracle相位训练与评估的模型在部署时使用估计相位会严重失效——并提出互补差距对来量化该偏差，同时通过部署感知的检查点选择与相位扰动在几乎不损失分割精度的前提下缩小差距。

    

    arXiv:2608.03342v2 公告类型：replace-cross。摘要：条件化分割模型在训练和评估时可能使用比部署阶段实际可获得的辅助信号更“干净”的信号。我们在相位条件化的超声心动图分割任务中研究这种协议层面的捷径学习与辅助变量偏移现象。所提出的“互补差距对”在可部署的Oracle估计路径上度量损失，并在Oracle随机路径上探测敏感性。在留出的CAMUS数据上，一个表现强劲的、以Oracle相位选出的模型在改用估计相位时严重失效，而对错误相位的敏感性在三次运行中持续存在。在EchoNet-Dynamic数据集上，现有估计器虽然仍可使用，但随机相位测试揭示了强烈的潜在敏感性。采用部署感知的检查点选择和相位扰动方法，可以在平均Dice系数几乎不变的情况下缩小这两个差距。探索性的子组分析量化了不同测量分层间的变异，下游的射血分数（EF）审计显示恢复……（原文摘要在此处截断）

    arXiv:2608.03342v2 Announce Type: replace-cross  Abstract: Conditional segmentation models may be trained and evaluated with auxiliary signals cleaner than those available at deployment. We study this protocol-level manifestation of shortcut learning and auxiliary-variable shift in phase-conditioned echocardiographic segmentation. The complementary gap pair measures loss on the deployable oracle-estimated pathway and probes sensitivity on the oracle-random pathway. On held-out CAMUS data, one strong-cyclic, oracle-selected run fails severely with estimated phase, while sensitivity to incorrect phase persists across three runs. On EchoNet-Dynamic, the current estimator remains usable, but random-phase testing reveals strong latent sensitivity. Deployment-aware checkpoint selection and phase perturbation reduce both gaps with little change in mean Dice. Exploratory subgroup analyses quantify variation across measured strata, and a downstream ejection fraction (EF) audit shows that recove
    
[^342]: 锁定评估面：CRISPRi扰动效应预测中的迁移失败与采样深度纠缠

    Locked Evaluation Surfaces: Transfer Failure and Sampling-Depth Entanglement in CRISPRi Perturbation-Effect Prediction

    [https://arxiv.org/abs/2608.00152](https://arxiv.org/abs/2608.00152)

    该论文在锁定且预注册的评估协议下评估冻结的Geneformer表示，发现其在虚拟细胞挑战赛（VCC）分布内数据上具有显著超越随机特征对照的预测信息量，但在零样本跨筛选迁移中失败，并揭示了迁移失败与采样深度等设计因素之间的纠缠。

    

    预测保留的目标基因如何响应CRISPRi扰动，以及此类预测能否在不同生物筛选之间迁移，是很难评估的：一种表示可能在单个筛选内具有信息量，却在跨筛选时失效，同时终点定义和采样深度等设计因素在不同数据集之间也存在差异。我们在一个锁定且预注册的协议下评估冻结的Geneformer表示：分类头与模型选择在测试评估之前冻结，外部结果标签在最终揭盲前保密，且支配分析的决策在其所支配的评估之前即已固定。在虚拟细胞挑战赛（VCC）的分布内数据上，该冻结表示携带可测量的预测信息，超过维度匹配的随机特征对照（ΔR² = +0.1645，95%置信区间 [+0.1375, +0.1920]），满足了在解释迁移之前所要求的预注册信息量门槛。随后它在零样本……[摘要在此处被截断]

    arXiv:2608.00152v2 Announce Type: replace  Abstract: Predicting how held-out target genes respond to CRISPRi perturbation, and whether such predictions transfer across biological screens, is hard to evaluate: a representation can be informative within one screen yet fail across screens, while endpoint definitions and design factors such as sampling depth differ between datasets. We evaluate a frozen Geneformer representation under a locked, pre-registered protocol, with heads and model selection frozen before test evaluation, external outcome labels withheld until final unblinding, and analysis-governing decisions fixed before the evaluations they govern. In-distribution on the Virtual Cell Challenge (VCC), the frozen representation carries measurable predictive information beyond a dimension-matched random-feature control (Delta R^2 = +0.1645, 95% CI [+0.1375, +0.1920]), satisfying the pre-registered informativeness gate required before interpreting transfer. It then fails zero-shot t
    
[^343]: UrbanDS：一种用于数据密集型城市任务的图引导大语言模型多智能体系统

    UrbanDS: A Graph-Guided LLM Multi-Agent System for Data-Intensive Urban Tasks

    [https://arxiv.org/abs/2607.26724](https://arxiv.org/abs/2607.26724)

    该论文提出了UrbanDS，一种图引导的LLM多智能体系统，通过构建统一数据集图来组织可复用的数据集技能及其关系，从而解决数据密集型城市任务中从大规模异构数据中发现并利用相关信息这一难题。

    

    大语言模型（LLM）智能体已被广泛应用于自动化数据科学任务。然而，现有方法通常依赖于有限的给定数据集，在面对数据密集型场景时存在困难，因为这类场景需要从大规模且异构的数据仓库中发现并利用相关信息。城市任务正是此类场景的典型代表，因为城市数据不仅规模庞大、来源多样，而且呈现出复杂的空间、时间和语义关系。为了应对这些挑战，我们提出了UrbanDS，一种面向数据密集型城市任务的图引导LLM多智能体系统。我们首先构建了一个统一的数据集图，用于组织可复用的数据集技能以及数据集之间的关系。具体而言，我们开发了数据画像智能体，为每个数据集构建相应的技能；此外，关系智能体用于识别数据集之间的关系并整合这些关系。

    arXiv:2607.26724v2 Announce Type: replace  Abstract: Large language model (LLM) agents have been widely applied in automating data science tasks. However, existing methods typically rely on a limited set of provided datasets, and they face challenges in data-intensive scenarios that require discovering and leveraging relevant information from large-scale and heterogeneous data repositories. Urban tasks are representative examples of such scenarios, as urban data are not only large-scale and multi-sourced, but also exhibit complex spatial, temporal, and semantic relationships. To address these challenges, we propose UrbanDS, a graph-guided LLM multi-agent system for data-intensive urban tasks. We first construct a unified dataset graph to organize reusable dataset skills and the relationships among datasets. Specifically, we develop a Data Profiling Agent that constructs a skill for each dataset. Moreover, a Relation Agent identifies relationships among datasets and integrates these rel
    
[^344]: 运行时拓扑上下文能否改进大语言模型生成的Kubernetes安全补丁？

    Does Runtime Topology Context Improve LLM-Generated Kubernetes Security Patches?

    [https://arxiv.org/abs/2607.25995](https://arxiv.org/abs/2607.25995)

    该论文提出KuTIE（Kubernetes拓扑智能引擎）系统，首次在受控条件下、跨多种依赖类型系统评估向大语言模型提供Kubernetes实时运行时服务拓扑上下文能否提升其生成的安全配置补丁的正确性，以解决模型因缺乏拓扑信息而生成破坏服务依赖的补丁的问题。

    

    Kubernetes是云原生生态系统的核心，负责编排容器化工作负载。近期研究表明，大语言模型（LLM）能够自动化集群安全修复，根据Kubernetes安全态势管理（KSPM）的发现结果生成配置补丁，无需人工编写。然而，这类系统在向模型输入每条发现结果时，将其与实时服务调用图割裂开来，并假设通用的加固知识即已足够。每当补丁需要保留对模型不可见的运行时服务依赖时，这一假设便会失效：此时看似合规的修复会带来破坏性的功能影响范围，导致下游调用者崩溃，或悄无声息地切断整个集群内的调用边。此前尚无研究在受控条件下、跨多种依赖类型地衡量实时集群上下文能否提升补丁的正确性。我们提出了KuTIE（Kubernetes拓扑智能引擎），它……（原文摘要在此处截断）

    arXiv:2607.25995v2 Announce Type: replace-cross  Abstract: Kubernetes is central to the cloud-native ecosystem, orchestrating containerised workloads. Recent work suggests that large language models (LLMs) can automate cluster security remediation, generating configuration patches from Kubernetes Security Posture Management (KSPM) findings without human authoring. Such systems, however, prompt the model with each finding in isolation from the live service call graph, assuming general hardening knowledge suffices. This assumption breaks down whenever a patch must preserve a runtime service dependency invisible to the model: an otherwise compliant fix then carries a destructive functional blast radius, crashing downstream callers or silently severing call edges across the cluster. Whether live cluster context improves patch correctness has not been measured under controlled conditions across multiple dependency classes. We introduce KuTIE (Kubernetes Topology Intelligence Engine), which 
    
[^345]: 退格键作为自然实验：帕金森病错误后选择性运动损伤的加速失效时间模型

    Backspace as a Natural Experiment: An Accelerated Failure Time Model of Selective Post-Error Motor Impairment in Parkinsons Disease

    [https://arxiv.org/abs/2607.24796](https://arxiv.org/abs/2607.24796)

    本研究将退格键事件作为自然纠错情境，发现错误后停顿时长（而非错误前打字不稳定性）与帕金森病严重程度显著相关，表明被动键盘监测可选择性捕捉PD患者的错误后运动恢复损伤。

    

    帕金森病（PD）会选择性地损害运动控制的不同阶段。在公开的neuroQWERTY MIT-CSXPD数据集（n=57名受试者，其中27名PD患者带有UPDRS-III评分）中，我们将退格键事件作为自然的纠错情境，测试被动采集的击键时间数据能否将基于变异性的错误前监测信号与基于速度的错误后恢复信号区分开来。错误前的打字不稳定性与PD严重程度无关（r=-0.072, p=0.721），而错误后停顿时长则与PD严重程度显著相关（r=+0.656, p=0.0002；受试者层面OLS p=1.2x10^-4, n=27，作为考虑受试者内事件聚类的主要分析）。包含随机受试者截距的混合效应对数正态模型在保留完整事件级统计效力的同时证实了这一结果（系数=0.0250, p=1.2x10^-4, n=1,563个事件），尽管采用了不相关的估计策略，其结果与受试者层面的OLS高度吻合。错误检测延迟也与UPDRS-III相关（r=+0.660），进一步证实了……（摘要原文在此处截断）

    arXiv:2607.24796v2 Announce Type: replace-cross  Abstract: Parkinson's disease (PD) selectively impairs distinct stages of motor control. Using backspace events as natural error-correction episodes in the public neuroQWERTY MIT-CSXPD dataset (n=57 subjects, 27 PD with UPDRS-III scores), we test whether passively-collected keystroke timing dissociates a variability-based pre-error monitoring signal from a speed-based post-error recovery signal. Pre-error typing instability does not track PD severity (r=-0.072, p=0.721), while post-error pause duration does (r=+0.656, p=0.0002; subject-level OLS p=1.2x10^-4, n=27, primary analysis given within-subject event clustering). A mixed-effects log-normal model with a random subject intercept confirms this while retaining full event-level power (coef=0.0250, p=1.2x10^-4, n=1,563 events), closely matching the subject-level OLS despite an unrelated estimation strategy. Error-detection latency also correlates with UPDRS-III (r=+0.660), confirming th
    
[^346]: AREX：迈向面向深度研究的递归自我改进智能体

    AREX: Towards a Recursively Self-Improving Agent for Deep Research

    [https://arxiv.org/abs/2607.21461](https://arxiv.org/abs/2607.21461)

    该论文提出AREX，一类递归自我改进的深度研究智能体，利用“发现—验证”的不对称性，通过内层研究循环与外层按约束逐项审计的自我改进循环交替进行，并结合自主上下文压缩工具，实现对研究答案的持续递归改进。

    

    深度研究要求智能体找到能够同时满足多个约束条件的答案。发现此类答案的代价高昂，而验证一个候选答案通常可以分解为易于处理的按约束逐项检查。这种“发现—验证”的不对称性表明，研究智能体不应只是进行更长时间的搜索：它应当通过验证中间结果，并利用部分验证后的状态来引导后续细化，从而递归地改进当前答案。我们提出了AREX，一类用于深度研究的递归自我改进智能体。AREX在内层研究循环（收集证据并构建临时答案）与外层自我改进循环（按约束逐项审计答案、识别未解决的主张并启动有针对性的后续研究）之间交替进行。为了在长时程上维持递归自我改进，AREX学习了一个自主的上下文更新工具，用于压缩不断增长的交互历史。

    arXiv:2607.21461v3 Announce Type: replace  Abstract: Deep research requires agents to find answers that jointly satisfy multiple constraints. Discovering such answers is costly, whereas verifying a candidate can often be decomposed into tractable constraint-wise checks. This discovery--verification asymmetry suggests that a research agent should do more than simply search longer: it should recursively improve its current answer by verifying intermediate results and using the partially verified state to guide subsequent refinement. We introduce AREX, a family of Recursively Self-Improving (RSI) deep research agents. AREX alternates between an inner research loop that gathers evidence and constructs a provisional answer, and an outer self-improvement loop that audits the answer constraint-wise, identifies unresolved claims, and launches targeted follow-up research. To sustain RSI over long horizons, AREX learns an autonomous context-update tool that compresses growing interaction history
    
[^347]: 基于多模态大语言模型的计算幽默研究：方法、数据集、评估与挑战

    Computational Humor with Multimodal LLMs: Methods, Datasets, Evaluation, and Challenges

    [https://arxiv.org/abs/2607.19011](https://arxiv.org/abs/2607.19011)

    本综述系统梳理了多模态大语言模型在理解表情包、漫画等视觉幽默方面的方法、数据集与评估协议，并构建了以能力为中心的“识别—解释推理—生成”层次框架，揭示了该领域从任务专用融合模型向大模型方法的转变及面临的评估捷径等核心挑战。

    

    表情包、漫画和连环画中的多模态幽默对人工智能系统来说仍然十分困难，因为其意图含义依赖于非字面机制、共享的文化知识和交际意图，而非对场景的字面描述。本综述聚焦于单图和多格作品中的视觉幽默理解，同时将幽默生成视为一个新兴的下游前沿方向。我们将相关文献置于以往幽默、讽刺以及通用多模态大语言模型（MLLM）综述的背景下，并采用以能力为中心的层次结构进行组织，涵盖识别、解释与推理、以及生成三个层面。在这一视角下，我们综合分析了基准设计、评估协议和建模范式，梳理了该领域从任务特定的融合模型向基于多模态对齐、证据支撑推理和受控生成的大模型方法的演进历程。最后，我们指出了该领域进展面临的主要障碍：易产生捷径学习的评估（原文摘要在此处截断）。

    arXiv:2607.19011v2 Announce Type: replace-cross  Abstract: Multimodal humor in memes, cartoons, and comics remains difficult for AI systems because intended meaning depends on non-literal mechanisms, shared cultural knowledge, and communicative intent rather than literal scene description. This survey focuses on visual humor understanding in single-image and multi-panel artifacts, while treating humor generation as an emerging downstream frontier. We position the literature against prior humor, sarcasm, and general MLLM surveys and organize it using a capability-centric hierarchy spanning recognition, interpretation and reasoning, and generation. Under this lens, we synthesize benchmark design, evaluation protocols, and modeling paradigms, tracing the field's shift from task-specific fusion models to large-model approaches based on multimodal alignment, evidence-grounded reasoning, and controlled generation. We conclude by highlighting the main barriers to progress: shortcut-prone eval
    
[^348]: 对齐微调如何塑造大语言模型中谄媚性及相关线索诱导偏差的表示？

    How Does Alignment Tuning Shape Representations of Sycophancy and Related Cue-Induced Biases in LLMs?

    [https://arxiv.org/abs/2607.18114](https://arxiv.org/abs/2607.18114)

    该研究发现大语言模型对谄媚性等线索诱导偏差的敏感性主要源于对齐微调而非预训练，且对齐模型中每种偏差都存在一个可被解码和干预的线性表示方向，可用于恢复无偏答案。

    

    现代大语言模型对于输入提示中一些出奇简单且无关紧要的变化异常敏感：一句随意的暗示、一个标注错误的少样本示例，或是一个伪造的先前助手回合，常常会使原本正确的答案发生翻转。我们研究了这种敏感性——涵盖谄媚性及相关线索诱导偏差——存在于模型内部的位置。我们在五个模型家族和七种偏差类型上，从隐藏状态中提取每种偏差的方向，并通过三种方法对其进行三角验证：探针分析、留一数据集（LODO）迁移以及因果干预。研究发现，这种敏感性主要由对齐微调而非预训练所塑造：预训练基础模型通常对这些偏差的屈从程度要低得多，其激活中除问题内容之外的线索特定信号也弱得多。在对齐后的模型中，每种偏差都存在一个连贯的线性方向，我们既可以对其进行解码，也可以沿其进行干预，从而在所有模型家族中恢复无偏的答案。

    arXiv:2607.18114v2 Announce Type: replace-cross  Abstract: Modern LLMs are alarmingly susceptible to surprisingly simple immaterial changes of input prompts: a casual hint, an incorrectly labeled few-shot example, or a fake prior assistant turn often flips an originally correct answer. We study where this susceptibility, spanning sycophancy and related cue-induced biases, lives inside the model. Across five model families and seven bias types, we extract a per-bias direction from hidden states and triangulate it through three measures: probing, leave-one-dataset-out (LODO) transfer, and causal intervention. The susceptibility is largely shaped by alignment tuning rather than pretraining: pretrained base models generally cave much less to these biases, and their activations carry much weaker cue-specific signal beyond question content. Within aligned models, each bias has a coherent linear direction that we can both decode and steer along, recovering the unbiased answer across every fam
    
[^349]: 零幻觉，由构造保证：面向可信赖企业AI的幻觉感知分层监督

    Zero Hallucination, by Construction: Hallucination-Aware Layered Oversight for Trustworthy Enterprise AI

    [https://arxiv.org/abs/2607.17883](https://arxiv.org/abs/2607.17883)

    本文提出HALO保证架构，将幻觉从“可消除的问题”重新定义为“可控制的失效模式”，通过六层防御机制把“零幻觉”从模型属性转变为系统强制实施的属性，从而实现可信赖的企业级AI。

    

    企业不会部署它们无法信任的AI智能体，而最常被引用的不信任原因就是幻觉：自信、流畅但完全不真实的输出。常见的应对方式是等待一个不会产生幻觉的模型出现。我们认为这是一个错误的目标。大型语言模型从构造上就具备生成无依据文本的能力，任何规模扩大都无法消除这种可能性；附加在原始模型上的忠实度评判器能捕捉一些错误，但仍会让其他错误漏网，甚至经过精心策划的检索管道也被证明会伪造引用。我们重新定义了目标：“零幻觉”不是模型所拥有的属性，而是系统所强制实施的属性。我们提出了HALO（幻觉感知分层监督），这是一种保证架构，它将幻觉视为一种可控制的失效模式，而非可消除的失效模式。HALO由六层防御组成：基于检索到的、经过批准的内容进行接地生成……

    arXiv:2607.17883v2 Announce Type: replace-cross  Abstract: Enterprises will not deploy AI agents they cannot trust, and the most-cited reason for distrust is hallucination: confident, fluent output that is simply not true. The common response is to wait for a model that does not hallucinate. We argue that this is the wrong target. Large language models are, by construction, capable of generating unsupported text, and no amount of scale removes the possibility; a faithfulness judge bolted onto a raw model catches some errors but still ships others, and even well-curated retrieval pipelines have been shown to fabricate citations. We reframe the goal: "zero hallucination" is not a property a model possesses but a property a system enforces. We present HALO (Hallucination-Aware Layered Oversight), an assurance architecture which treats hallucination as a containable failure mode rather than an eliminable one. HALO composes six layers of defense: grounded generation over retrieved, approved
    
[^350]: SeerGuard：一种基于世界模型预测的移动GUI智能体安全框架

    SeerGuard: A Safety Framework for Mobile GUI Agents via World Model Prediction

    [https://arxiv.org/abs/2607.15550](https://arxiv.org/abs/2607.15550)

    SeerGuard通过安全增强世界模型（SAWM）在动作执行前预测后果并评估风险，为移动GUI智能体提供了一个能够预防不可逆转错误操作的前瞻性安全框架。

    

    移动图形用户界面（GUI）智能体在自动化复杂任务方面已展现出卓越的能力，但它们也带来了关键的安全风险——单个错误的操作可能导致不可逆转的后果。现有的安全机制主要是被动响应式的，缺乏在执行前评估风险的能力。在本文中，我们提出了SeerGuard，这是一个具备后果感知能力的安全框架，旨在通过执行前的指令级筛查和动作级风险评估来缓解这些风险。具体而言，动作级评估会在当前GUI状态下分析智能体提出的动作，预判可能的结果，从而在风险被执行之前将其识别出来。为了实现这些能力，我们通过多任务学习构建了一个统一的安全增强世界模型（SAWM），将语义下一状态预测与安全风险评估集成在一起。大量实验表明，SeerGuard具有良好的泛化能力……

    arXiv:2607.15550v2 Announce Type: replace  Abstract: Mobile graphical user interface (GUI) agents have demonstrated remarkable capabilities in automating complex tasks, yet they introduce critical safety risks where a single erroneous action can lead to irreversible consequences. Existing safety mechanisms are primarily reactive, lacking the ability to assess risks before execution. In this paper, we introduce SeerGuard, a consequence-aware safety framework designed to mitigate these risks through pre-execution instruction-level screening and action-level risk assessment. Specifically, the action-level assessment analyzes agent-proposed actions within current GUI states, anticipating likely outcomes to identify risks before they are executed. To enable these capabilities, we construct a unified safety-augmented world model (SAWM) via multi-task learning, integrating semantic next-state prediction with safety risk assessment. Extensive experiments demonstrate that SeerGuard generalizes 
    
[^351]: Anamnesis：一个用于大规模背景故事条件化调查模拟的开源平台

    Anamnesis: An Open-Source Platform for Large-Scale Backstory-Conditioned Survey Simulation

    [https://arxiv.org/abs/2607.10628](https://arxiv.org/abs/2607.10628)

    Anamnesis是一个开源平台，通过结构化叙事背景故事对大语言模型进行条件化，实现了在虚拟人群上进行人口可控、大规模且支持多模态的调查模拟。

    

    我们提出了Anamnesis，一个使用大语言模型进行人口统计学可控调查模拟的交互式系统。Anamnesis是开源的，专为非技术背景的用户和研究人员设计，使其能够在虚拟人群而非真实人类受试者上进行调查工具的原型设计与压力测试。该平台在统一的网页界面中实现了近期提出的Anthology和Alterity框架，这两个框架利用结构化的叙事背景故事来调节模型响应。系统支持开放式生成、概率性人口重采样以及多模态（图像和音频）调查。我们通过两个案例研究对该系统进行了评估：（1）复制皮尤研究中心“美国趋势小组”（ATP）中关于政治类型学和生物医学议题的部分调查；（2）模拟《纽约客》漫画配文大赛中的人类偏好。在两个案例中，Anamnesis所产生的观点分布都更接近真实数据。

    arXiv:2607.10628v2 Announce Type: replace-cross  Abstract: We present Anamnesis, an interactive system for demographically controllable survey simulation using large language models. Open-source and designed for non-technical users/researchers, Anamnesis enables the prototyping and stress-testing of survey instruments on virtual populations rather than real human subjects. The platform operationalizes the recently introduced Anthology and Alterity frameworks, which use structured narrative backstories to condition model responses, within a unified web interface. It supports open-ended generation, probabilistic demographic resampling, and multimodal (image and audio) surveys. We evaluate the system through two case studies: (1) replicating segments of Pew Research Center's American Trends Panel (ATP) on political typology and biomedical issues and (2) emulating human preference in the New Yorker Caption Contest. In both cases, Anamnesis produces opinion distributions that more closely m
    
[^352]: LUNA：超越蒙皮的学习通用3D人体动画

    LUNA: Learning Universal 3D Human Animation Beyond Skinning

    [https://arxiv.org/abs/2606.31981](https://arxiv.org/abs/2606.31981)

    LUNA提出了一种无需LBS蒙皮的通用神经人体动画模型，利用基于Transformer的运动回归器将图像、关键点、草图等多种2D控制信号直接映射为3D高斯变形，并通过LBS教师蒸馏与无标注视频的混合监督，突破了传统参数化人体模型的拟合限制与表现力瓶颈。

    

    从单目图像创建逼真且可动画化的3D人体化身，目前仍在很大程度上依赖于线性混合蒙皮（LBS）和参数化人体模型，这些方法限制了表现力，且常因不完美的拟合而引入伪影。我们提出LUNA，一种无需LBS的通用神经动画模型，它直接将多种2D控制信号（如图像、关键点、草图以及未见过的角色）映射为3D高斯变形，从而绕过显式的人体拟合。其核心是一个基于Transformer的运动回归器，将全局刚性运动与细粒度的局部动态解耦，以同时捕捉连贯的运动和细微的非刚性效果。为了解决2D到3D提升过程中固有的歧义性，并将模型扩展至拟合数据集之外，我们引入了混合监督机制，从LBS教师模型中蒸馏出软结构先验，并设计了一种损失函数，支持在有限的拟合数据以及大规模野外无标注视频上进行训练。大量实验……（摘要截断）

    arXiv:2606.31981v2 Announce Type: replace-cross  Abstract: Creating photorealistic, animatable 3D human avatars from monocular images still largely depends on Linear Blend Skinning (LBS) and parametric body models, which constrain expressivity and often introduce artifacts due to imperfect fitting. We propose LUNA, an LBS-free universal neural animation model that directly maps multiple 2D controls like images, keypoints, sketches, and unseen characters into 3D Gaussian deformations, bypassing explicit body fitting. At its core, a transformer-based motion regressor disentangles global rigid motion from fine-grained local dynamics to capture both coherent movement and subtle non-rigid effects. To resolve the inherent ambiguity of 2D-to-3D lifting while scaling beyond fitted datasets, we introduce hybrid supervision that distills soft structural priors from an LBS teacher and a loss that supports training on both limited fitted data and large in-the-wild unlabeled videos. Extensive exper
    
[^353]: DigitalCoach：人类与智能体计算机使用辅导中的沟通与视觉定位差距

    DigitalCoach: Communication and Grounding Gaps in Human and Agentic Computer Use Coaching

    [https://arxiv.org/abs/2606.31980](https://arxiv.org/abs/2606.31980)

    该论文构建了包含72场人类专家-新手计算机使用辅导会话的多模态数据集DigitalCoach，揭示了当前最先进模型虽能生成与人类相似的辅导语句，但在解释、错误诊断和视觉定位方面显著不足，导致学习者被动跟随指令而非深度参与学习。

    

    智能体在自动化软件任务方面的能力日益增强，但它们能否教会人类自己使用软件呢？我们推出了DigitalCoach，这是一个多模态数据集，包含72场人类专家与新手之间的计算机使用辅导会话，涵盖五款软件应用中基于28.1小时屏幕和输入事件录制的22,752轮对话。我们利用DigitalCoach评估最先进的模型能否教会人类如何使用计算机。自动化评估表明，模型在辅导方式上与人类存在差异：模型提供更多直接指令，但更少的解释、错误诊断和知识检验问题。当我们固定辅导方法时，模型生成的语句与人类参考相似，但在视觉上下文定位方面表现较差。交互式评估证实，模型辅导者会导致学习者被动跟随指令而缺乏深入参与，并且在视觉定位方面存在不足。

    arXiv:2606.31980v2 Announce Type: replace-cross  Abstract: Agents are increasingly capable of automating software tasks, but can they teach humans how to use software themselves? We introduce DigitalCoach, a multimodal dataset of 72 human expert-novice computer use coaching sessions consisting of 22,752 dialogue turns grounded in 28.1 hours of screen and input event recordings across five software applications. We use DigitalCoach to evaluate whether state-of-the-art models can teach humans how to use computers. Automated evaluation shows that models differ from humans in how they coach: models provide more direct instructions, but fewer explanations, error diagnoses, and knowledge-check questions. When we fix the coaching method, models produce utterances similar to human references yet poorly grounded in visual context. Interactive evaluation confirms that model coaches cause learners to passively follow instructions without deeper engagement and fall short in visual grounding. Digit
    
[^354]: 大语言模型能否想象二元道德困境之外的替代方案？

    Can LLMs Imagine Moral Alternatives Beyond Binary Dilemmas?

    [https://arxiv.org/abs/2606.31213](https://arxiv.org/abs/2606.31213)

    该论文提出MoralAltDataset数据集，通过在307个二元道德困境中引入折中和重构的替代选项，发现当替代方案可用时人类与15个LLM的道德选择分布均发生显著转变且一致性增强，但存在关键差异——LLM明显偏好GPT-5创作的替代方案，而人类的选择不受创作来源影响，揭示了机器与人类在“想象道德替代方案”能力上的差距。

    

    随着大语言模型（LLM）越来越多地充当道德顾问和道德智能体，它们必须应对相互竞争的价值观之间的冲突。然而，以往关于道德困境的研究忽视了人类道德认知的一个核心方面：在给定选项之外想象替代方案。我们提出了MoralAltDataset数据集，其中包含307个顾问型和人机交互型智能体困境，并为其补充了折中方案与重新构建的替代选项。我们在二元选项和四选项两种设置下比较了人类与LLM的判断。在人类被试和15个LLM中，两种设置下的总体道德选择分布存在显著差异，且折中方案往往比两个原始二元选项中的任何一个都更受青睐。结果显示出价值观的转变，以及在替代方案上人类与LLM之间更强的一致性。按创作来源分层的结果揭示了一种描述性差距：人类选择替代方案的比率在不同来源之间相似，而LLM则明显更频繁地选择由GPT-5创作的替代方案。随后我们比较了人类与……（原文摘要在此处被截断）

    arXiv:2606.31213v2 Announce Type: replace-cross  Abstract: As LLMs increasingly serve as moral advisors and agents, they must address conflicts between competing values. Yet prior work on moral dilemmas overlooks a central aspect of human moral cognition: imagining alternatives beyond the given options. We introduce MoralAltDataset, comprising 307 Advisor and AI-facing Agent dilemmas augmented with compromise and reframed alternatives. We compare human and LLM judgments in binary and four-option settings. Across human participants and 15 LLMs, aggregate moral choice distributions differ substantially between the two settings, with compromise often preferred over either original binary option. Results show value shifts and stronger human-LLM agreement on alternatives. Source-stratified results reveal a descriptive gap: human alternative-selection rates are similar across authoring sources, whereas LLMs select GPT-5-authored alternatives substantially more often. We then compare human-au
    
[^355]: 面向大语言模型智能体规划的自进化世界模型

    Self-Evolving World Models for LLM Agent Planning

    [https://arxiv.org/abs/2606.30639](https://arxiv.org/abs/2606.30639)

    提出自进化世界模型框架 WorldEvolver，通过情景记忆、语义记忆和选择性前瞻三个模块，在保持智能体与模型参数完全冻结的情况下持续修正部署时的上下文，从而提升长时程 LLM 智能体规划中前瞻预测的可靠性与下游决策成功率。

    

    世界模型为长时程大语言模型（LLM）智能体提供了一种有原则的前瞻能力：在执行动作之前预测其后果。然而，不可靠的前瞻预测可能被忽略、被误用，甚至降低下游决策的质量。本文提出了 WorldEvolver，一个自进化世界模型框架，它在保持下游智能体和所有模型参数冻结的前提下，在部署阶段不断修正自身的上下文。WorldEvolver 集成了三个模块：(i) 情景记忆，通过基于检索的模拟来利用真实动作转移；(ii) 语义记忆，从预测与观测的不匹配中提取持久性的启发式规则；(iii) 选择性前瞻，在将预测整合进智能体推理上下文之前过滤掉低置信度的预测。我们在 ALFWorld 和 ScienceWorld 上评估了 WorldEvolver，在 Word2World 上测量世界模型的预测准确率，并在 AgentBoard 上测量下游智能体的成功率。

    arXiv:2606.30639v2 Announce Type: replace  Abstract: World models offer a principled way to equip long-horizon LLM agents with foresight: predictions of action consequences before execution. However, unreliable foresight can be ignored, misused, or even degrade downstream decision-making. In this paper, we introduce WorldEvolver, a self-evolving world model framework that revises its deployment-time context while keeping the downstream agent and all model parameters frozen. WorldEvolver integrates three modules: (i) Episodic Memory, which exploits real action transitions through retrieval-based simulation; (ii) Semantic Memory, which extracts persistent heuristic rules from prediction-observation mismatches; and (iii) Selective Foresight, which filters low-confidence predictions before integrating them into agent reasoning context. We evaluate WorldEvolver on ALFWorld and ScienceWorld, measuring world model prediction accuracy on Word2World and downstream agent success rate on AgentBoa
    
[^356]: 流推理模型：将流转化为高效的循环推理器

    Flow Reasoning Models: Turning Flows Into Efficient Recurrent Reasoners

    [https://arxiv.org/abs/2606.29150](https://arxiv.org/abs/2606.29150)

    提出流推理模型（FRMs），通过自条件化将流模型的一次性去噪转化为迭代式解精炼，使模型能够并行地做出并修改相互依赖的决策，实现高效的结构化推理。

    

    结构化推理需要做出并修改相互依赖的决策，以达成全局一致的解决方案。现有架构在这方面存在困难：自回归模型按顺序依次做出承诺，无法修改较早的决策；而掩码扩散模型通常需要精心设计的解码方案来协调相互依赖的预测。我们提出了流推理模型（FRMs），这是一种新颖的结构化推理框架，它将连续流适配到离散结构化输出之上，并配以一种简单的循环精炼机制。通过让流模型以自身过去的输出进行自条件化，我们将一次性去噪转变为迭代式的解精炼过程。这使得FRMs能够并行地做出并修改决策，高效地协调解中相互依赖的选择。然而，由于一步式训练预测与递归生成之间存在曝光偏差，传统自条件化在更深的循环层次上会变得不可靠……

    arXiv:2606.29150v3 Announce Type: replace  Abstract: Structured reasoning requires making and revising interdependent decisions to reach a globally consistent solution. Existing architectures struggle with this: autoregressive models commit sequentially and cannot revise earlier decisions, while masked diffusion models often require careful decoding schemes to coordinate interdependent predictions. We introduce Flow Reasoning Models (FRMs), a novel framework for structured reasoning that adapts continuous flows over discrete structured outputs with a simple recurrent refinement mechanism. By self-conditioning a flow model on its own past outputs, we turn one-shot denoising into iterative solution refinement. This lets FRMs make and revise decisions in parallel, efficiently coordinating interdependent choices across solutions. Yet conventional self-conditioning becomes unreliable at greater recurrent depth due to exposure bias between one-step training predictions and recursively genera
    
[^357]: 基于连续可穿戴生理信号的多评分者疼痛评估事件对齐分析

    Event-Aligned Analysis of Multi-Rater Pain Assessments Using Continuous Wearable Physiology

    [https://arxiv.org/abs/2606.23705](https://arxiv.org/abs/2606.23705)

    该论文提出了一种感知评分者、事件对齐的分析框架，将多评分者的疼痛评分转化为离散疼痛变化事件并与可穿戴生理信号对齐，首次揭示了疼痛与生理的关系可能因评分者不同而存在差异，跨评分者汇总评估可能掩盖有意义的生理模式。

    

    疼痛由患者、护士和临床医生的评估方式各不相同，然而大多数计算方法都假设存在单一的真值标签，实际上忽略了评分者的身份。我们提出了一种感知评分者、事件对齐的框架，将稀疏的、特定于评分者的疼痛评分转化为离散的疼痛变化事件，并将连续的可穿戴生理信号与这些事件对齐，同时在整个过程中保留评分者身份。将该框架应用于在脊柱相关疼痛操作过程中收集的多模态可穿戴数据，该框架识别出不同评分者群体之间的显著分歧，并提供了初步的探索性证据，表明在报告疼痛加剧之前存在依赖于评分者的生理差异。这些发现表明，疼痛与生理之间的关系可能并非评分者不变，且跨评分者汇总评估可能会掩盖有意义的生理模式。因此，感知评分者、事件对齐的视角是一种具有前景的方法。

    arXiv:2606.23705v2 Announce Type: replace-cross  Abstract: Pain is assessed differently by patients, nurses, and clinicians, yet most computational approaches assume a single ground-truth label - effectively ignoring who is doing the rating. We introduce a rater-aware, event-aligned framework that converts sparse, rater-specific pain ratings into discrete pain-change events and aligns continuous wearable physiological signals to these events, preserving rater identity throughout. Applied to multimodal wearable data collected during spine-related pain procedures, the framework identifies substantial disagreement across rater groups and provides preliminary, exploratory evidence of rater-dependent physiological differences preceding reported pain increases. These findings suggest that pain-physiology relationships may not be rater-invariant, and that aggregating assessments across raters may mask meaningful physiological patterns. A rater-aware, event-aligned perspective is therefore a p
    
[^358]: 基于能量的Transformer作为阅读难度的预测器

    Energy-Based Transformers as Predictors of Reading Difficulty

    [https://arxiv.org/abs/2606.23382](https://arxiv.org/abs/2606.23382)

    本文首次将基于能量的Transformer度量引入计算心理语言学，证明该能量度量在多个阅读时间语料库中是阅读难度的稳健预测因子，其解释力显著超越传统的惊讶度和注意力熵度量，并与Hopfield网络等联想记忆理论建立了形式化联系。

    

    Transformer语言模型已成为建模人类句子处理的成熟工具，其中惊讶度（surprisal）和注意力熵等度量作为阅读难度的有效预测因子，共同捕捉处理负荷的互补方面。本文探索了一类相关的Transformer模型：基于能量的Transformer，它为联想记忆模型提供了原则性的形式化联系，使句法处理研究与Hopfield网络和密集联想记忆的更广泛文献直接对接。据我们所知，这是计算心理语言学领域首次对基于能量的Transformer度量进行的探索。在多个阅读时间语料库（Natural Stories、UCL眼动追踪、UCL自定步速阅读）上，能量度量是阅读时间的稳健预测因子，在所有三个语料库中都提供了超越惊讶度和熵的显著拟合增益。在关于关系从句处理的受控实验中（摘要在此处截断）。

    arXiv:2606.23382v2 Announce Type: replace-cross  Abstract: Transformer language models have become established tools for modeling human sentence processing, with measures such as surprisal and attention entropy serving as effective predictors of reading difficulty that together capture complementary aspects of processing load. Here, we explore a related class of transformer models: energy-based transformers, which provide a principled formal link to associative memory models, bringing processing research into direct contact with the broader literature on Hopfield networks and dense associative memory. To our knowledge, this is the first exploration of an energy-based transformer measure in computational psycholinguistics. Across reading-time corpora (Natural Stories, UCL eye-tracking, UCL self-paced reading), the energy measure is a robust predictor of reading times, providing significant fit beyond surprisal and entropy in all three. In a controlled experiment on relative clause proce
    
[^359]: DART：面向混合推理模型免训练自适应思考预算的草稿一致性路由

    DART: Draft-Agreement Routing for Training-Free Adaptive Thinking Budgets in Hybrid Reasoning Models

    [https://arxiv.org/abs/2606.23181](https://arxiv.org/abs/2606.23181)

    DART是一种免训练的自适应路由框架，通过比较两个无思考草稿的一致性来决定是否需要深度推理并预测思考预算，在大幅减少思考token消耗的同时保持甚至提升模型在数学和代码任务上的准确率。

    

    混合推理模型既可以直接回答问题，也可以花费额外的token进行扩展思考。一个实用的路由器应该为每个查询在这两种模式之间进行选择，使简单问题避免不必要的推理，而困难问题获得足够的预算来完成答案。现有的路由器虽朝此方向发展，但它们通常需要带标签的训练数据，或预先固定思考预算，忽略了来自模型本身的答案层面的证据。我们提出了DART，一个免训练的路由框架，它采样两个廉价的“无思考”草稿，当草稿一致时接受直接回答，当草稿不一致时根据草稿熵预测思考预算。在主要对比实验中，DART在大多数设置下保持或提升了“始终思考”模式的准确率，同时减少了思考token的使用。准确率在奥数级数学上最高提升+9.0分，在基于执行等价性的代码任务上最高提升+22.5分，同时思考token的使用量下降。

    arXiv:2606.23181v2 Announce Type: replace  Abstract: Hybrid reasoning models can answer directly or spend extra tokens on extended thinking. A practical router should choose between these modes for each query, so easy problems avoid unnecessary reasoning and hard problems receive enough budget to finish the answer. Existing routers move in this direction, but they typically require labeled training data or fix thinking budgets up front, ignoring answer-level evidence from the model itself. We introduce DART, a training-free routing framework that samples two cheap no-think drafts, accepts direct answering when the drafts agree, and predicts a thinking budget from draft entropy when they disagree. Across the main comparisons, DART preserves or improves always-thinking accuracy in most settings while reducing thinking-token use. Accuracy improves by up to +9.0 points on Olympiad-level math and by up to +22.5 points on code under execution-based equivalence, while thinking-token use drops
    
[^360]: 引导而非解决：为大型代码智能体训练小型评论模型

    Steer, Don't Solve: Training Small Critic Models for Large Code Agents

    [https://arxiv.org/abs/2606.21811](https://arxiv.org/abs/2606.21811)

    通过训练专门负责高层次规划的小型评论模型（4B/8B）在推理时引导大型编码智能体识别并纠正错误，在SWE-Bench Verified上显著提升多个更大规模编码智能体的解决率（最高提升16.0%）并降低推理成本。

    

    编码任务通常较为复杂，需要多种能力，涵盖从高层次规划到低层次实现的各个方面。虽然编码智能体针对这些联合能力进行了优化，但诸如高层次规划等单项能力可能有不同的最优解，并仍然是主要瓶颈。为应对这一挑战，我们训练了一个独立于编码智能体、专门擅长高层次规划的评论模型，在推理阶段对编码智能体进行引导。我们构建了SFT和DPO数据来训练该评论模型，使其能够识别编码智能体所犯的错误，并提供正确且清晰的高层次指导，而无需生成具体的操作动作。实验表明，我们微调后的4B和8B评论模型显著提升了6个更大规模编码智能体的性能（例如，在SWE-Bench Verified上，将GLM-4.7-Flash-30B-A3B和GPT-OSS-120B的解决率分别提升了16.0%和14.4%）。该评论模型还降低了总推理成本（摘要原文在此处截断）。

    arXiv:2606.21811v2 Announce Type: replace-cross  Abstract: Coding tasks are typically complicated and require multiple capabilities, ranging from high-level planning to low-level implementation. While coding agents are optimized for the joint capabilities, individual capabilities such as high-level planning may have different optima and remain a major bottleneck. To address this challenge, we train a separate critic model that is specialized in high-level planning to steer the coding agent in inference. We construct SFT and DPO data to train the critic model to identify errors made by the coding agent and provide correct and clear high-level guidance without generating concrete actions. Experiments show that our fine-tuned 4B and 8B critic models significantly improve the performance of 6 larger coding agents (e.g., improving the resolved rates of GLM-4.7-Flash-30B-A3B and GPT-OSS-120B by 16.0% and 14.4% on SWE-Bench Verified). The critic model also reduces the total inference costs fo
    
[^361]: ReproRepo：利用GitHub仓库议题规模化扩展可复现性审计

    ReproRepo: Scaling Reproducibility Audits with GitHub Repository Issues

    [https://arxiv.org/abs/2606.18237](https://arxiv.org/abs/2606.18237)

    ReproRepo提出利用GitHub上人工提交的议题作为天然监督信号，构建了可规模化的可复现性评估框架，并在1,149篇机器学习论文上验证了LLM智能体无需执行代码即可识别真实复现障碍的能力（最佳智能体可覆盖约90%的论文）。

    

    从论文及已发布代码中复现研究结果对科学进步至关重要。现有工作已引入基准来评估LLM智能体能否协助实现可复现性，但由于在数据整理和评估方面依赖大量人工投入，这些基准难以规模化。我们提出了ReproRepo，这是一个可扩展的可复现性评估框架，它利用人工提交的GitHub议题（issues）作为对真实复现障碍天然产生的监督信号。我们在来自主要会议的1,149篇近期机器学习论文上构建了ReproRepo实例，并评估了四种前沿模型-智能体配置。结果表明，LLM智能体即使不执行代码，也能从论文-仓库配对中识别出许多真实世界的可复现性问题：我们研究中表现最佳的智能体，即搭载GPT-5.5的Codex，能为约90%的论文找出至少一个与人工报告语义相关的复现障碍。

    arXiv:2606.18237v2 Announce Type: replace-cross  Abstract: Reproducing research results from papers and released code is central to scientific progress. Existing works have introduced benchmarks to evaluate whether LLM agents can assist with reproducibility, but they are difficult to scale due to their reliance on substantial manual effort for data curation and evaluation. We introduce ReproRepo, a scalable framework for reproducibility evaluation that leverages human-raised GitHub issues as naturally occurring supervision on realistic reproduction blockers. We instantiate ReproRepo on 1,149 recent machine learning papers from major conferences and evaluate four frontier model-agent configurations. Our results show that LLM agents, even without executing code, can identify many real-world reproducibility problems from paper-repository pairs: the best agent in our study, namely Codex with GPT-5.5, surfaces at least one semantically related human-reported blocker for $\sim$90% of papers 
    
[^362]: AfriSUD：一个用于评估非洲语言模型的依存树库集合

    AfriSUD: A Dependency Treebank Collection for Evaluating Models on African Languages

    [https://arxiv.org/abs/2606.12708](https://arxiv.org/abs/2606.12708)

    该论文推出了首个覆盖九种非洲语言的大规模句法标注依存树库集合AfriSUD，并揭示现有模型在这些语言上仍存在显著的句法理解差距。

    

    尽管非洲语言具有语言多样性和全球重要性，但在支持自然语言处理（NLP）的研究和资源中，它们仍然代表性不足。我们旨在通过引入AfriSUD来弥合这一差距，这是首个面向九种多样化非洲语言的大规模句法标注树库集合，涵盖了撒哈拉以南非洲的主要语系和地区。基于表层句法通用依存框架，这项由社区主导的工作提供了高质量的、经母语者验证的数据，能够捕捉诸如黏着和声调等类型学上的关键特征。我们在AfriSUD上评估了一系列模型的词性标注和依存句法分析性能，包括非transformer基线模型、多语言预训练编码器以及大语言模型（LLMs）。我们的结果揭示了一个显著的句法差距：模型在这九种语言上仍表现出明显的局限性，这表明现有架构可能无法完全捕捉这些语言的句法结构（原文摘要在此处截断）。

    arXiv:2606.12708v2 Announce Type: replace-cross  Abstract: Despite their linguistic diversity and global significance, African languages remain underrepresented in research and resources to support NLP. We aim to bridge this gap by introducing AfriSUD, the first large-scale collection of syntactically annotated treebanks for nine diverse African languages spanning major language families and regions across Sub-Saharan Africa. Using the Surface-Syntactic Universal Dependencies (SUD) framework, our community-led effort provides high-quality, native-speaker verified data that capture typological key features such as agglutination and tone. We evaluate a range of models on AfriSUD for part-of-speech tagging and dependency parsing including non-transformer baselines, multilingual pretrained encoders, and LLMs. Our results reveal a significant syntax gap, where models still show clear limitations across the nine languages, suggesting that existing architectures may not fully capture the stru
    
[^363]: 生成主义：迈向生成式人工智能时代的学习理论

    Generativism: Toward a Learning Theory for the Age of Generative Artificial Intelligence

    [https://arxiv.org/abs/2606.12441](https://arxiv.org/abs/2606.12441)

    本文批判性反思了传统四大学习理论在生成式AI时代的局限，提出“生成主义”新学习理论，其核心观点是学习日益通过人类学习者与AI系统之间迭代式的知识共同建构来实现。

    

    行为主义、认知主义、建构主义和联通主义这四种主导性学习理论，随着生成式人工智能（AI）在教育环境中的日益普及，显现出显著的概念局限性。这些框架是在能够生成、综合和推理知识的AI系统出现之前构建的。本文批判性地审视了每一种学习理论，并识别出被生成式AI的可供性所挑战的假设。文章借鉴分布式认知、延展心智、人机协作、AI素养、认知卸载和元认知等领域的研究成果，提出“生成主义”作为生成式AI时代的学习理论。生成主义主张，学习将越来越多地通过人类学习者与AI系统之间迭代式的知识共同建构而发生。所提出的框架围绕四个构念组织（认识论伙伴关系、分布……（原文摘要至此截断）

    arXiv:2606.12441v2 Announce Type: replace-cross  Abstract: The four dominant learning theories of behaviorism, cognitivism, constructivism, and connectivism show significant conceptual limitations as generative artificial intelligence (AI) proliferates in educational settings. These frameworks were formulated before the emergence of AI systems capable of generating, synthesizing, and reasoning about knowledge. This article critically examines each learning theory and identifies assumptions challenged by the affordances of generative AI. Drawing on research in distributed cognition, extended mind, human-AI collaboration, AI literacy, cognitive offloading, and metacognition, the article proposes Generativism as a learning theory for the generative AI age. Generativism posits that learning increasingly occurs through the iterative co-construction of knowledge between human learners and AI systems. The proposed framework is organized around four constructs (epistemic partnership, distribut
    
[^364]: Self-EmoQ：基于普鲁奇克情绪轮引导的价值规划驱动流式情感语音合成

    Self-EmoQ: Plutchik-Guided Value-based Planning to Drive Streaming Emotional TTS

    [https://arxiv.org/abs/2606.09837](https://arxiv.org/abs/2606.09837)

    提出了一个在文本生成前进行情绪决策的即插即用LLM情绪规划框架，通过结合普鲁奇克情绪轮理论的混合奖励进行强化学习训练，以流式方式驱动下游情感语音合成，并在多个数据集上超越了提示与微调基线方法。

    

    情感交互对于对话式人工智能（AI）日益重要，然而当前系统缺乏一种自我情绪决定机制来驱动流式文本转语音（TTS）合成。我们提出了一个情绪规划框架，在文本生成之前确定情绪，并以流式方式为下游情感TTS提供支撑。该框架通过一个即插即用的大语言模型（LLM）模块实现，该模块由预训练LLM初始化，并通过以情绪作为动作的强化学习（RL）进行训练。我们采用了一种混合奖励机制，将模仿信号与理论驱动的评分相结合，其中采用了普鲁奇克情绪轮理论。通过在DailyDialog、EmoryNLP、IMEOCAP和MELD数据集上的实验，我们的方法在情绪决定和响应质量方面均优于提示工程和微调基线方法。最后，我们实现了一个完整的流式处理管道用于实时部署，语音质量……

    arXiv:2606.09837v2 Announce Type: replace-cross  Abstract: Emotional interaction is increasingly crucial for conversational AI, yet current systems lack a self-emotion determination mechanism to drive the streaming text-to-speech (TTS) synthesis. We propose an emotion-planning framework that determines the emotion prior to the textual generation, grounding the downstream emotional TTS in a streaming manner. The framework is implemented by a plug-and-play LLM module, initialized from pretrained LLMs, and trained by reinforcement learning (RL) with emotions as the actions. A hybrid reward is employed which combines imitation signals with theory-driven scoring, in which the theory of Plutchik's wheel of emotions is adopted. By experiments on DailyDialog, EmoryNLP, IMEOCAP, and MELD, our method outperforms prompting and finetuning baselines on both emotion determination and response quality. We finally implement an entire streaming pipeline for real-time deployment, with the speech quality
    
[^365]: DOG-DPO：面向安全对齐的几何动态优化

    DOG-DPO:Dynamic Optimization in Geometry for Safety Alignment

    [https://arxiv.org/abs/2606.07678](https://arxiv.org/abs/2606.07678)

    提出无需训练的数据选择框架DOG-DPO，将偏好对视为模型表示空间中的几何方向，通过分解全局锚定子空间与数据集特有残余子空间并最大化多样性覆盖，为DPO安全对齐筛选出广泛且非冗余的偏好数据子集。

    

    大语言模型的安全对齐依赖于偏好数据，但当前的训练流程往往使用庞大且冗余的数据集。现有的数据选择方法通常独立地对每个偏好对进行打分，将方向性的偏好信息压缩为标量化的质量或多样性分数。这种以样本为中心的视角在多数据集场景下尤为受限，因为共享的安全方向与数据集特有的残余风险同时存在。我们提出了DOG-DPO，一个无需训练的数据选择框架，它将偏好对视为结构化的几何信号。DOG-DPO首先将每个偏好对表示为模型表示空间中的一个方向；然后将多数据集的偏好几何分解为全局锚定子空间和数据集特有的残余子空间；最后通过最大化基于多样性的覆盖度来选择子集，从而在DPO训练之前实现对安全对齐方向的广泛、非冗余的覆盖。

    arXiv:2606.07678v3 Announce Type: replace-cross  Abstract: Safety alignment for large language models relies on preference data, but current pipelines often train on large, redundant datasets. Existing data selection methods typically score each preference pair independently, collapsing directional preference information into scalar quality or diversity scores. This sample-centric view is especially limiting in multi-dataset settings, where shared safety directions coexist with dataset-specific residual risks. We propose DOG-DPO, a training-free data selection framework that treats preference pairs as structured geometric signals. DOG-DPO first represents each preference pair as a direction in model representation space. It then decomposes multi-dataset preference geometry into a global anchor subspace and dataset-specific residual subspaces. Finally, it selects subsets by maximizing diversity-based coverage, encouraging broad, non-redundant coverage of alignment directions before DPO 
    
[^366]: 为扩散语言模型启用共享前缀的KV缓存

    Enabling KV Caching of Shared Prefix for Diffusion Language Models

    [https://arxiv.org/abs/2606.07571](https://arxiv.org/abs/2606.07571)

    本文提出bicache，首个针对扩散语言模型共享前缀的KV缓存技术，解决了双向注意力下KV动态变化导致传统缓存失效的问题。

    

    针对共享前缀的键值（KV）缓存对于高吞吐量的大语言模型（LLM）服务至关重要，但在新兴的扩散语言模型（DLM）中面临重大挑战。在DLM中，双向注意力意味着更新任何一个token都会动态改变整个上下文及其对应的KV。因此，现有为LLM开发的缓存技术（其假设KV一旦计算完成就保持不变）会破坏共享前缀的KV。我们的实验表明，将这些技术应用于DLM会导致模型准确率崩溃至接近零。为了实现高吞吐量的DLM服务，我们提出了双向前缀缓存bicache，这是首个针对DLM共享前缀的KV缓存技术。

    arXiv:2606.07571v3 Announce Type: replace-cross  Abstract: Key-value (KV) caching for shared prefixes is essential for high-throughput large language model (LLM) serving, but it faces critical challenges in emerging diffusion language models (DLMs). In DLMs, bidirectional attention means that updating any token dynamically alters the entire context and its corresponding KVs. Thus, existing caching techniques developed for LLMs, which assume that KVs remain invariant once computed, corrupt the shared prefix KVs. Our experiments show that applying these techniques to DLMs causes model accuracy to collapse to near zero.   To unlock high-throughput DLM serving, we propose bidirectional prefix caching, bicache, the first KV caching technique for shared prefixes in DLMs. bicache is designed based on key observations from our comprehensive analysis: shared prefix KVs remain stable and reusable in shallow layers, while the depth of shallow layers depends on the fraction of shared prefix tokens
    
[^367]: GeM-NR：面向非刚性场景变化的几何感知多视图编辑

    GeM-NR: Geometry-Aware Multi-View Editing for Nonrigid Scene Changes

    [https://arxiv.org/abs/2606.05142](https://arxiv.org/abs/2606.05142)

    提出了一种无需训练的快速灵活方法GeM-NR，通过几何感知实现多视图一致的通用图像编辑，突破了以往方法只能进行刚性或仅外观编辑的限制，支持大幅改变场景几何和外观的非刚性编辑。

    

    生成模型在多视图图像编辑方面的最新进展使我们向通用的3D内容生成与定制迈进了一步。大多数现有工作通过利用未编辑场景的几何结构，专注于刚性编辑或仅改变外观的编辑。这自然地将这些方法限制在保留底层场景结构的编辑上。目前的非刚性方法仅限于物体移除和插入，这反映了其训练数据的局限性。通用的非刚性编辑，即大幅且任意改变场景几何结构的编辑，对现有方法来说仍然具有挑战性。我们提出了GeM-NR，这是一种快速、灵活且无需训练的方法，用于通用的多视图一致性图像编辑，包括那些大幅改变场景几何和外观的编辑。给定一张通过选定2D编辑器编辑的锚点图像和一张未编辑的查询图像，GeM-NR能够将查询图像编辑得与锚点图像保持一致。

    arXiv:2606.05142v2 Announce Type: replace-cross  Abstract: Recent developments in multi-view image editing with generative models have brought us a step closer toward general 3D content generation and customization. Most existing works focus on rigid or appearance-only edits by utilizing the geometry of the unedited scene. This naturally limits these methods to edits that preserve the underlying scene structure. Current nonrigid approaches are limited to object removal and insertion, reflecting the data they are trained on. General nonrigid edits, i.e., edits that substantially and arbitrarily change the scene geometry, remain challenging for existing methods. We propose GeM-NR, a fast and flexible training-free approach for general multi-view consistent image editing, including edits that drastically change the geometry and appearance of the scene. Given an anchor image edited with a chosen 2D editor and a query unedited image, GeM-NR edits the query image consistently with the anchor
    
[^368]: 谁在NLP中进行标注？2018至2025年间人类标注报告的大规模评估

    Who Annotates in NLP? A Large-scale Assessment of Human Annotation Reporting between 2018 and 2025

    [https://arxiv.org/abs/2606.02255](https://arxiv.org/abs/2606.02255)

    首次对2018至2025年间主要NLP会议中的人类标注报告进行大规模任务级审计，提出统一的标注报告分类体系并借助经验证的LLM抽取流程构建了大规模标注报告数据集，揭示了标注者身份与过程控制等信息在论文中的普遍缺失。

    

    人类标注是许多NLP研究的实证基础，涵盖从数据集构建到模型评估的各个环节，但论文往往不清楚标注者是谁、标注过程如何被控制。我们首次对主要NLP会议中的人类标注报告进行了大规模、任务级别的审计，探究哪些标注细节被记录、哪些缺失，以及报告内容如何随时间、主题、会议和人类判断的预期用途而变化。我们提出了一个统一的标注报告实践分类体系，并在Annotated-gold（一个由人工裁定的金标准，包含41篇论文和72个标注任务）上验证了LLM辅助的抽取流程，其中表现最佳的模型与裁定标签达成与人类相当的一致性，Krippendorff's alpha系数为0.606，而人类之间的一致性为0.585。利用该流程，我们构建了Annotated-llm数据集，涵盖ACL会议论文（原文摘要在此处被截断）。

    arXiv:2606.02255v2 Announce Type: replace-cross  Abstract: Human annotation is the empirical foundation of much NLP research, from dataset construction to model evaluation, but papers often leave unclear who produced the annotations and how the annotation process was controlled. We provide the first large-scale, task-level audit of human annotation reporting across major NLP venues, asking which annotation details are documented, which are missing, and how reporting varies across time, topic, venue, and intended use of human judgment. We introduce a unified taxonomy of annotation-reporting practices and validate an LLM-assisted extraction pipeline against Annotated-gold, a human-adjudicated gold standard of 41 papers and 72 annotation tasks, where the best model reaches human-comparable agreement with adjudicated labels, with Krippendorff's alpha of 0.606 versus 0.585 for human-human agreement. Using this pipeline, we construct Annotated-llm, a dataset covering ACL-venue papers from 20
    
[^369]: PlanarBench：通过平面图绘制评估大语言模型的空间推理能力

    PlanarBench: Evaluating LLM Spatial Reasoning via Planar Graph Drawing

    [https://arxiv.org/abs/2606.02010](https://arxiv.org/abs/2606.02010)

    该论文提出PlanarBench基准，要求大语言模型仅根据边列表绘制平面图的无交叉ASCII图，并发现边数（即约束数量）比顶点数更能决定任务难度，为评估LLM空间推理能力提供了可控的测试环境。

    

    现有的大语言模型图基准通常要求模型回答图论问题或计算符号解，而非构建空间布局，且任务难度主要按顶点数量进行分层。然而，现有研究也表明，任务难度与边所施加的约束数量的关联比与所排列顶点数量的关联更为密切。我们提出了PlanarBench，这是一个要求模型仅根据边列表生成平面图的无交叉ASCII绘图的基准。在91个模型配置和199个具有2至7个顶点的非同构连通平面图上，边数与平均任务得分的关联比顶点数更强（r=-0.85 对比 r=-0.47），并且在控制顶点数之后仍保持强关联（rp=-0.80）。PlanarBench为分离这两个难度维度提供了受控环境。此外，摘要在此处被截断。

    arXiv:2606.02010v2 Announce Type: replace-cross  Abstract: Existing LLM graph benchmarks typically ask models to answer graph-theoretic questions or compute symbolic solutions rather than construct spatial layouts. Within-task difficulty is also primarily stratified by vertex count. However, existing research also suggests that task difficulty is more closely related to the number of constraints imposed by the edges than to the number of vertices being arranged. We introduce PlanarBench, a benchmark that asks models to produce crossing-free ASCII drawings of planar graphs given only an edge list. Across 91 model configurations and 199 non-isomorphic connected planar graphs with 2-7 vertices, edge count is more strongly associated with mean task score than vertex count ($r=-0.85$) versus ($r=-0.47$) and remains strongly associated after controlling for vertex count ($r_p=-0.80$). PlanarBench provides a controlled setting for separating these two difficulty axes. In addition, neither dra
    
[^370]: 技能复用作为智能体强化学习中的压缩

    Skill Reuse as Compression in Agentic RL

    [https://arxiv.org/abs/2605.31509](https://arxiv.org/abs/2605.31509)

    该论文提出基于最小描述长度（MDL）原则的ReuseRL框架，通过从成功轨迹中提取共享技能字典并惩罚难以压缩的特异行为，显著提升了智能体强化学习的分布内外泛化能力。

    

    通过强化学习（RL）训练的大语言模型智能体通常学到脆弱的、局限于特定任务的捷径。我们假设，当智能体的成功轨迹在结构上可压缩、能够被分解为一小组可复用的抽象模式时，智能体的泛化能力会更好。为了将这一假设形式化，我们提出了ReuseRL，该方法将智能体强化学习建立在最小描述长度（MDL）原则之上。ReuseRL从成功轨迹中提取共享的技能字典，并在强化学习目标中加入分割代价，明确惩罚那些编码效果差的特异化行为。我们证明了PAC-Bayes界，保证从成功轨迹中提取的字典在未来成功行为上具有有界的期望描述长度。在ALFWorld、TextWorld-Cooking和Countdown-Stepwise任务上，ReuseRL在分布内和分布外成功率上均超越了原始GRPO以及强基线方法。

    arXiv:2605.31509v2 Announce Type: replace-cross  Abstract: Large language model agents trained with reinforcement learning (RL) often learn brittle, task-specific shortcuts. We hypothesize that agents generalize better when their successful trajectories are structurally compressible, decomposed into a small set of reusable abstract patterns. To formalize this, we introduce ReuseRL, which grounds agentic RL in the Minimum Description Length (MDL) principle. ReuseRL extracts a shared skill dictionary from successful trajectories and augments the RL objective with a segmentation cost, explicitly penalizing idiosyncratic behaviors that encode poorly. We prove a PAC-Bayes bound guaranteeing that a dictionary extracted from successful trajectories has bounded expected description length on future successful behavior. Across ALFWorld, TextWorld-Cooking, and Countdown-Stepwise, ReuseRL improves in- and out-of-distribution success over vanilla GRPO and strong round-length baselines.
    
[^371]: MusTBench：音乐大语言模型中时间定位能力的基准测试与提升

    MusTBench: Benchmarking and Advancing Temporal Grounding in Music LLMs

    [https://arxiv.org/abs/2605.29300](https://arxiv.org/abs/2605.29300)

    该论文提出了经音乐专家验证的MusTBench基准和涵盖四阶段优化的MusT方案，用于评估并提升音乐大语言模型将回答准确锚定到音频正确时间段的能力。

    

    近期的大型音频-语言模型（LALMs）在理解音乐内容方面展现出了可观的能力。然而，这些模型的回答是否锚定于音频中正确的时间区域，这一问题仍未得到充分探索。这一局限性对于音乐理解尤为关键，因为音乐中的关键信息往往以时间上局部化事件的形式出现，例如乐器进入和节奏转换。为填补这一空白，我们提出了MusTBench，一个经音乐专家验证的基准，旨在通过五个时间定位问答任务来评估LALM的时间定位能力。为进一步提升现有模型的时间定位能力，我们提出了MusT，一种新颖的四阶段时间优化方案，涵盖音乐编码器适配、大语言模型适配、大语言模型监督微调以及基于强化学习的优化。在MusTBench上的实验表明，现有LALM难以实现精确的时间定位，而MusT则带来了……

    arXiv:2605.29300v2 Announce Type: replace-cross  Abstract: Recent Large Audio-Language Models (LALMs) have demonstrated promising abilities in understanding musical content. However, whether their responses are grounded in the correct temporal regions of the audio remains underexplored. This limitation is particularly critical for music understanding, where key information often occurs as temporally localized events, such as instrument entries and rhythmic transitions. To address this gap, we introduce MusTBench, a music-expert-validated benchmark designed to evaluate temporal grounding in LALMs through five temporally grounded question-answering tasks. To further improve temporal grounding in existing models, we propose MusT, a novel four-stage temporal optimization recipe spanning music encoder adaptation, LLM adaptation, LLM supervised fine-tuning, and RL-based optimization. Experiments on MusTBench show that existing LALMs struggle with precise temporal grounding, while MusT brings
    
[^372]: 统计上“认真”的重要性：对GSM-Symbolic基准的批判性再评估

    The Importance of Being Statistically Earnest: A Critical Re-evaluation of GSM-Symbolic

    [https://arxiv.org/abs/2605.28700](https://arxiv.org/abs/2605.28700)

    该研究指出GSM-Symbolic基准的统计方法存在缺陷，重新评估发现20个开源模型中仅8个呈现统计显著的性能下降，且数据集中整数分布系统性偏大是重要混淆因素，从而质疑了“大语言模型缺乏真正推理能力”的结论。

    

    GSM-Symbolic基准测试（Mirzadeh等人，2025）报告称，25个大型语言模型（LLM）在基于模板生成的GSM8K问题变体上测试时，性能均出现一致性的下降，并据此得出这些模型缺乏真正推理能力的结论。我们认为这一结论建立在并不稳固的统计基础之上。我们使用带逐题随机效应的自举广义线性混合模型，对20个开源权重模型进行重新评估，发现其中仅有8个模型在原始提示格式下表现出统计显著的性能变化。此外，我们发现了一个此前未被认识到的影响因素：GSM-Symbolic主数据集中问题文本的整数分布相对于原始GSM8K系统性地偏向更大的数值（K-S统计量=0.12，p<0.001），这与原论文作者的说法相矛盾。在控制这一“大数值”效应后，剩余显著性案例中的一半可以得到解释。

    arXiv:2605.28700v3 Announce Type: replace  Abstract: The GSM-Symbolic benchmark (Mirzadeh et al., 2025) reported consistent performance drops across 25 Large Language Models (LLMs) when tested on template-generated variants of GSM8K problems, concluding that the models lack genuine reasoning capabilities. We argue that this conclusion rests on shaky statistical ground. Re-evaluating 20 open-weight models using bootstrapped Generalised Linear Mixed Models with per-question random effects, we find that only 8 exhibit statistically significant performance changes under the original prompt format. Moreover, we identify a previously unacknowledged factor: the distribution of integers in problem texts of the main GSM-Symbolic dataset is systematically shifted towards larger values relative to the original GSM8K (K-S statistic = 0.12, p < 0.001), contradicting the original authors' claims. Controlling for this large-number effect accounts for significance in half of the remaining cases. Among
    
[^373]: 面向视觉语音识别的扩散大语言模型

    Diffusion Large Language Models for Visual Speech Recognition

    [https://arxiv.org/abs/2605.28456](https://arxiv.org/abs/2605.28456)

    提出了首个基于扩散大语言模型的视觉语音识别框架DLLM-VSR，通过迭代掩码去噪与灵活顺序解码，利用置信度引导的解掩码机制以双向上下文细化模糊词元，并采用两阶段训练策略分离内容对齐与长度建模。

    

    arXiv:2605.28456v2 公告类型：替换 摘要：现有的视觉语音识别（VSR）系统通常依赖于从左到右的自回归解码方式，这可能迫使模型在获得充分上下文之前，就对视觉上模糊的词元做出过早的决策。我们提出了DLLM-VSR，据我们所知，这是首个基于扩散大语言模型（DLLM）的VSR框架，它将语音转录任务形式化为具有灵活顺序解码的迭代式掩码去噪过程。借助基于置信度的解掩码机制，DLLM-VSR能够尽早确定高置信度的位置，并将这些已确定的词元作为双向上下文来细化模糊的词元。为了使扩散大语言模型适应VSR任务，我们引入了一种两阶段掩码去噪训练策略，将视觉到文本的内容对齐与长度建模分离开来。我们进一步观察到，与在推理阶段提供真实转录文本长度、从而使模型能够专注于转录内容解码的上限设置相比，该方法存在一定的性能差距。为了缩小这一……

    arXiv:2605.28456v2 Announce Type: replace  Abstract: Existing Visual Speech Recognition (VSR) systems commonly rely on left-to-right autoregressive decoding, which can force premature decisions on visually ambiguous tokens before sufficient context is available. We propose DLLM-VSR, to the best of our knowledge, the first Diffusion Large Language Model (DLLM)-based VSR framework, formulating transcription as iterative masked denoising with flexible-order decoding. With confidence-based unmasking, DLLM-VSR commits high-confidence positions early and uses the committed tokens as bidirectional context to refine ambiguous ones. To adapt DLLMs to VSR, we introduce a two-stage masked-denoising training strategy that separates visual-to-text content alignment from length modeling. We further observe a performance gap compared with an upper-bound setting where the ground-truth transcript length is provided at inference, allowing the model to focus on transcript content decoding. To reduce this
    
[^374]: UniACE：一个用于评估LLM智能体能力的统一框架

    UniACE: A Unified Framework for Evaluating LLM Agentic Capabilities

    [https://arxiv.org/abs/2605.27898](https://arxiv.org/abs/2605.27898)

    UniACE通过将基准测试表示为“指令—工具—环境”三元组、采用共享的任务无关执行框架和统一执行条件（含离线快照模式），实现了以模型为中心的LLM智能体能力标准化评估，解决了跨基准测试比较受实现差异和资源条件影响的问题。

    

    智能体基准测试越来越多地被用于跨领域比较大型语言模型（LLM），然而报告的分数反映的是完整的“模型—测试框架—环境”配置，而非模型本身。基准测试包将原生任务与特定的提示词、工具协议、编排逻辑，有时还包括动态外部资源耦合在一起，使得跨基准测试的比较对实现方式和资源条件非常敏感。我们提出了UniACE，一个在明确、统一执行条件下进行以模型为中心评估的统一框架。UniACE将每个基准测试表示为“指令—工具—环境”三元组，通过共享的、与任务无关的测试框架在隔离的按任务运行时中执行LLM，并保留原生成功标准。对于依赖动态资源的任务，可选的离线模式用固定的、预先收集的快照替代实时访问。其评估协议还进一步标准化了效率的测量方式。

    arXiv:2605.27898v3 Announce Type: replace  Abstract: Agent benchmarks are increasingly used to compare large language models (LLMs) across domains, yet a reported score reflects a complete model--harness--environment configuration rather than the model alone. Benchmark packages couple native tasks with specific prompts, tool protocols, orchestration logic, and sometimes dynamic external resources, making cross-benchmark comparisons sensitive to implementation and resource conditions. We present UniACE, a unified framework for model-centric evaluation under an explicit, common execution condition. UniACE represents each benchmark as an instruction--tool--environment triplet, executes LLMs through a shared, task-agnostic harness in isolated per-task runtimes, and preserves native success criteria. For tasks that rely on dynamic resources, an optional offline mode replaces live access with fixed, pre-collected snapshots. Its evaluation protocol further standardizes efficiency measurement,
    
[^375]: 当最强的教师并非最好的教师：以学生为中心的答案选择

    When the Strongest Teacher Is Not the Best Teacher: Student-Centric Answer Selection

    [https://arxiv.org/abs/2605.26872](https://arxiv.org/abs/2605.26872)

    论文提出SCAS框架，证明最强教师的正确答案未必是学生的最佳训练监督，并通过逐token梯度分解推导出仅需前向计算的高效代理指标，依据学生中心学习成本来选择最适合学生的教师答案。

    

    大语言模型（LLM）训练越来越依赖教师生成的监督信号，包括合成回复、推理轨迹和工具使用演示。当前的做法通常选择表现最强的教师来生成学生的训练数据，这隐含地将教师的测试性能视为教学质量的替代指标。我们证明这一假设可能失效：即使多个教师对同一问题都给出了正确答案，最强教师给出的答案也不一定是对特定学生最好的监督信号。为了解决这一空白，我们提出了学生中心答案采样，这是一个根据估计的学生中心学习成本，从经过验证的教师生成答案中进行选择的框架。受逐token梯度分解的启发，我们推导出一种仅需前向计算的高效代理指标来估计该成本，并用它来指导训练过程中的答案选择。实验涵盖了30个教师模型和6个学生基础模型。

    arXiv:2605.26872v4 Announce Type: replace-cross  Abstract: LLM training increasingly relies on teacher-generated supervision, from synthetic responses to reasoning traces and tool-use demonstrations. Current practice often chooses the highest-performing teacher to generate student training data, implicitly treating teacher test performance as a proxy for teaching quality. We show that this assumption can fail: even when multiple teachers provide correct answers to the same question, the answer from the strongest teacher is not necessarily the best supervision for a given student. To address this gap, we propose Student-Centric Answer Sampling (SCAS), a framework that selects from verified teacher-generated answers according to their estimated student-centric learning cost. Motivated by a token-wise gradient decomposition, we derive an efficient forward-only proxy for this cost and use it to guide answer selection during training. Experiments across 30 teacher models, 6 student base mod
    
[^376]: 照我说的做，而非照我做的做：大语言模型中的指令-归纳冲突

    Do as I Say, Not as I Do: Instruction-Induction Conflict in LLMs

    [https://arxiv.org/abs/2605.20382](https://arxiv.org/abs/2605.20382)

    该研究通过构造用户指令与硬编码对话模式相冲突的实验场景，发现大语言模型的指令遵循能力在不同模型间差异巨大（1%到99%）且与常规能力基准基本无关，其鲁棒性取决于指令内容与模型价值先验的一致性以及输出格式。

    

    语言模型被训练来遵循指令，但它们同时也是强大的模式补全器。当这两个目标发生冲突时会发生什么？我们构建了一些对话场景，其中用户指令要求模型以目标方式T行动（例如，始终输出特定的token、用某种特定语言回答或采用某个人设），而与之对抗的是N个硬编码的助手回合，它们展示了一种竞争性模式P。我们在这种设置下测量指令遵循（IF）率，涵盖13个模型和16种不同指令，测试轮数多达50轮。各模型的平均指令遵循率从1%到99%不等，且与标准能力基准基本不相关。从指令遵循到模式遵循的转变是普遍存在的，但高度依赖于具体模型。鲁棒性同时受到指令内容和输出格式的调节：当指令与模型训练中的价值先验一致时，模型能更长时间地抵抗归纳效应。

    arXiv:2605.20382v3 Announce Type: replace-cross  Abstract: Language models are trained to follow instructions, but they are also powerful pattern completers. What happens when these two objectives conflict? We construct conversations in which a user instruction to behave in a target way T (e.g., always output a specific token, answer in a particular language, or adopt a persona) is opposed by N hardcoded assistant turns demonstrating a competing pattern P. We then measure instruction-following (IF) rates in this setting, across 13 models and 16 different instructions, for up to 50 turns. Average instruction-following rates range from 1% to 99% across models, largely uncorrelated with standard capability benchmarks. The transition from instruction-following to pattern-following is universal but highly model-dependent. Robustness is modulated both by instruction content, with models resisting induction longer when instructions align with their trained value priors, and by output format, 
    
[^377]: 非线性算子及其导数的万能逼近

    Universal Approximation of Nonlinear Operators and Their Derivatives

    [https://arxiv.org/abs/2605.15285](https://arxiv.org/abs/2605.15285)

    本文首次证明了关于 $k$ 阶可微非线性算子及其导数的万能逼近定理，首次将经典万能逼近定理完整推广至无穷维巴拿赫空间与算子学习领域，并由此开启了导数感知算子学习（DIOL）的新方向。

    

    建立非线性算子及其导数的万能逼近定理（UATs）是算子学习（OL）中的一个基础性开放问题，并在非线性泛函分析中引发了微妙的问题。我们通过算子学习架构，首次证明了关于 $k$ 阶可微非线性算子及其导数的万能逼近定理，这些定理在紧集上一致成立，并且在一般有限输入测度下于加权 Bastiani–Sobolev 空间中成立。在完全一般的巴拿赫空间框架下，这些结果首次将 [Hornik, 1991] 中相应的具有影响力的经典万能逼近定理完整地推广到无穷维空间和算子学习，并开启了在一般巴拿赫空间上的导数感知算子学习（DIOL）（即学习非线性算子及其导数）。基于我们的万能逼近定理，我们提出了 DIOL 中的 Bastiani–Sobolev 训练方法。我们展示了 DIOL 和我们的 UATs 可应用的前沿开放方向：算子学习中的高阶精度；快速约束……（摘要原文在此处截断）

    arXiv:2605.15285v3 Announce Type: replace-cross  Abstract: Establishing Universal Approximation Theorems (UATs) for nonlinear operators and their derivatives is a foundational open problem in Operator Learning (OL) and raises delicate questions in Nonlinear Functional Analysis. We prove the first UATs for $k$-times differentiable nonlinear operators and their derivatives via OL architectures, uniformly on compact sets and in weighted Bastiani--Sobolev spaces for general finite input measures. In full Banach-space generality, these are the first complete generalizations of the corresponding influential classical UATs in [Hornik, 1991] to infinite-dimensional spaces and OL, {and launch Derivative-Informed Operator Learning (DIOL) (i.e. learning nonlinear operators and their derivatives)} on general Banach spaces. Based on our UATs, we formulate Bastiani--Sobolev training in DIOL. We present open frontiers where DIOL and our UATs find applications: high-order accuracy in OL; fast constrai
    
[^378]: SkillRet：面向LLM智能体技能检索的大规模基准

    SkillRet: A Large-Scale Benchmark for Skill Retrieval in LLM Agents

    [https://arxiv.org/abs/2605.05726](https://arxiv.org/abs/2605.05726)

    该论文提出了SkillRet，首个面向LLM智能体技能检索的大规模基准，包含16,129个带结构化语义标签和两级分类体系的公开技能，以及63,259个训练样本和4,392个技能池互不相交的评估查询，填补了该领域基准匮乏的空白。

    

    随着LLM智能体越来越多地配备庞大的可复用技能库进行部署，为用户请求选择合适的技能已成为一个关键的系统挑战。在小型技能库中，用户可以通过名称显式调用技能，但随着技能生态系统在严格的上下文和延迟预算下不断增长，这一假设不再成立。尽管技能检索具有重要的实际意义，但该领域仍未得到充分探索，基准测试匮乏，且对现实技能库上的检索行为缺乏了解。为填补这一空白，我们推出了SkillRet，一个面向LLM智能体技能检索的大规模基准。SkillRet包含16,129个公开的智能体技能，通过结构化语义标签以及覆盖6个主要类别和18个子类别的两级分类体系进行组织。它提供了63,259个训练样本和4,392个评估查询，且评估所用技能池与训练技能池互不相交，既支持基准评测，也支持面向检索的训练。在多样化的检索……（摘要原文在此处截断）

    arXiv:2605.05726v2 Announce Type: replace  Abstract: As LLM agents are increasingly deployed with large libraries of reusable skills, selecting the right skill for a user request has become a critical systems challenge. In small libraries, users may invoke skills explicitly by name, but this assumption breaks down as skill ecosystems grow under tight context and latency budgets. Despite its practical importance, skill retrieval remains underexplored, with limited benchmarks and little understanding of retrieval behavior on realistic skill libraries. To address this gap, we introduce SkillRet, a large-scale benchmark for skill retrieval in LLM agents. SkillRet contains 16,129 public agent skills, organized with structured semantic tags and a two-level taxonomy spanning 6 major categories and 18 sub-categories. It provides 63,259 training samples and 4,392 evaluation queries with disjoint skill pools, enabling both benchmarking and retrieval-oriented training. Across a diverse set of ret
    
[^379]: 多模态大语言模型内部视觉表征的因果探查

    Causal Probing for Internal Visual Representations in Multimodal Large Language Models

    [https://arxiv.org/abs/2605.05593](https://arxiv.org/abs/2605.05593)

    该研究提出基于激活引导的因果框架，揭示了多模态大语言模型中实体知识局部化编码而抽象概念全局分布的分化现象，并证明模型深度的增加是编码复杂抽象概念这一缩放定律背后的机制性驱动因素。

    

    尽管多模态大语言模型（MLLMs）在多样化任务中取得了显著成功，但其编码和定位不同视觉概念的内部机制仍然知之甚少。为了揭示这些机制，我们提出了一个基于激活引导的因果框架，以主动探查和操纵内部视觉表征。通过对四个视觉概念类别的系统性干预，我们的结果揭示了概念编码的分化现象：实体知识具有明显的局部化特征，而抽象概念则全局分布于整个网络。关键的是，这种分化揭示了缩放定律的一个机制性驱动因素：增加模型深度对于编码分布式和复杂的抽象概念是必不可少的，而实体则始终保持较高的局部化程度。此外，反向引导发现，阻断显式输出会触发潜在激活的激增……

    arXiv:2605.05593v2 Announce Type: replace  Abstract: Despite the remarkable success of Multimodal Large Language Models (MLLMs) across diverse tasks, the internal mechanisms governing how they encode and ground distinct visual concepts remain poorly understood. To unravel these mechanisms, we propose a causal framework based on activation steering to actively probe and manipulate internal visual representations. Through systematic intervention across four visual concept categories, our results reveal a divergence in concept encoding: entity knowledge is distinctively localized, whereas abstract concepts are globally distributed across the network. Critically, this divergence uncovers a mechanistic driver of scaling laws: increasing model depth is indispensable for encoding distributed and complex abstract concepts, whereas entities maintain a consistently high degree of localization. Furthermore, reverse steering uncovers that blocking explicit output triggers a surge in latent activat
    
[^380]: D3-Gym：为数据驱动发现构建真实世界可验证环境

    D3-Gym: Constructing Real-World Verifiable Environments for Data-Driven Discovery

    [https://arxiv.org/abs/2604.27977](https://arxiv.org/abs/2604.27977)

    本文提出了首个为科学数据驱动发现自动构建的可验证环境数据集D3-Gym，包含565个来自真实科学代码库的任务，其自动评估脚本与人工标注达到87.5%的一致性，且基于其轨迹训练可显著提升Qwen3模型在ScienceAgentBench上的表现。

    

    尽管面向科学数据驱动发现的语言模型和智能体近期取得了进展，但由于缺乏能够代表真实世界科学任务的可验证环境，其能力的提升受到了阻碍。为填补这一空白，我们推出了D3-Gym，这是首个为科学数据驱动发现自动构建的、带有可验证环境的数据集。D3-Gym包含来自四个学科、239个真实科学代码库的565个任务，每个任务均配有自然语言指令、预装依赖的可执行环境、数据集预览、参考解决方案以及自动合成的评估脚本。我们的评估脚本与人工标注的黄金标准达到87.5%的一致性，并在领域特定评估逻辑上表现出高度对齐。在从D3-Gym采样的轨迹上进行训练，使Qwen3系列模型在ScienceAgentBench上获得一致提升，其中Qwen3-32B绝对提升7.8个百分点（原文在此处截断）。

    arXiv:2604.27977v3 Announce Type: replace-cross  Abstract: Despite recent progress in language models and agents for scientific data-driven discovery, advancing their capabilities is held back by the absence of verifiable environments representing real-world scientific tasks. To fill this gap, we introduce D3-Gym, the first automatically constructed dataset with verifiable environments for scientific Data-Driven Discovery. D3-Gym comprises 565 tasks from 239 real scientific repositories across four disciplines, each with a natural language instruction, an executable environment with pre-installed dependencies, dataset previews, a reference solution, and an automatically synthesized evaluation script. Our evaluation scripts achieve 87.5% agreement with human-annotated gold standards and strong alignment in domain-specific evaluation logic. Training on trajectories sampled from D3-Gym yields consistent gains across Qwen3 models on ScienceAgentBench, boosting Qwen3-32B by 7.8 absolute poi
    
[^381]: Transformer的拓扑困境

    The Topological Trouble With Transformers

    [https://arxiv.org/abs/2604.17121](https://arxiv.org/abs/2604.17121)

    该论文揭示了Transformer纯前馈架构在动态状态追踪上的根本性拓扑缺陷——状态表示随每个新输入被不断推向更深层直至耗尽模型深度，并论证时间上延伸的认知需要从显式思维轨迹回归到基于循环结构的隐式激活动力学。

    

    Transformers通过不断扩展的上下文历史来编码序列中的结构。然而，其纯前馈架构从根本上限制了动态状态追踪能力。状态追踪——即对反映不断演变环境的潜在变量进行迭代更新——涉及固有的顺序依赖关系，而前馈网络难以维持这些依赖关系。因此，前馈模型在每处理一个新的输入步骤时，都会将不断演变的状态表示推送到更深的层堆栈中，使信息在浅层中无法访问，并最终耗尽模型的深度。虽然这种深度限制可以通过动态深度模型以及通过将状态表示外化的显式或潜在思考来绕过，但这些解决方案在计算和内存方面都是低效的。在本文中，我们认为时间上延伸的认知需要将焦点从显式思维轨迹重新转向通过循环结构实现的隐式激活动力学。

    arXiv:2604.17121v5 Announce Type: replace-cross  Abstract: Transformers encode structure in sequences via an expanding contextual history. However, their purely feedforward architecture fundamentally limits dynamic state tracking. State tracking -- the iterative updating of latent variables reflecting an evolving environment -- involves inherently sequential dependencies that feedforward networks struggle to maintain. Consequently, feedforward models push evolving state representations deeper into their layer stack with each new input step, rendering information inaccessible in shallow layers and ultimately exhausting the model's depth. While this depth limit can be bypassed by dynamic depth models and by explicit or latent thinking that externalizes state representations, these solutions are computationally and memory inefficient. In this article, we argue that temporally extended cognition requires refocusing from explicit thought traces to implicit activation dynamics via recurrent 
    
[^382]: 用于免训练神经放射影像分析的智能体大语言模型

    Agentic Large Language Models for Training-Free Neuro-Radiological Image Analysis

    [https://arxiv.org/abs/2604.16729](https://arxiv.org/abs/2604.16729)

    本文提出一种免训练的智能体流程，让大语言模型编排现成的专业工具，自主完成脑部MRI分析中从预处理到病理分割的复杂端到端工作流程。

    

    最先进的大语言模型（LLM）在通用视觉问答任务中表现出色。然而，一个根本性的局限依然存在：当前架构缺乏直接分析体积医学影像（如CT或MRI）所需的原生3D空间推理能力。新兴的智能体AI提供了一种新的解决方案，通过让大语言模型编排和调用专门的外部工具，从而消除了对内在3D处理能力的需求。然而，此类智能体框架在复杂的多步骤放射学工作流程中的可行性仍未得到充分探索。在这项工作中，我们提出了一种用于自动化脑部MRI分析的免训练智能体流程。我们在多个大语言模型（GPT-5.4、Gemini 3.1 Pro、Claude Sonnet 4.6）上结合现成的领域专用工具验证了我们的方法，该系统能够自主执行复杂的端到端工作流程，包括预处理（颅骨剥离、图像配准）、病理分割（胶质瘤等）。

    arXiv:2604.16729v2 Announce Type: replace-cross  Abstract: State-of-the-art large language models (LLMs) show high performance in general visual question answering. However, a fundamental limitation remains: current architectures lack the native 3D spatial reasoning required to directly analyze volumetric medical imaging, such as CT or MRI. Emerging agentic AI offers a new solution, eliminating the need for intrinsic 3D processing by enabling LLMs to orchestrate and leverage specialized external tools. Yet, the feasibility of such agentic frameworks in complex, multi-step radiological workflows remains underexplored. In this work, we present a training-free agentic pipeline for automated brain MRI analysis. Validating our methodology on several LLMs (GPT-5.4, Gemini 3.1 Pro, Claude Sonnet 4.6) with off-the-shelf domain-specific tools, our system autonomously executes complex end-to-end workflows, including preprocessing (skull stripping, registration), pathology segmentation (glioma, m
    
[^383]: 面向地球系统预测中百亿亿次生成式数据同化的线性复杂度全局注意力机制

    Global Attention with Linear Complexity for Exascale Generative Data Assimilation in Earth System Prediction

    [https://arxiv.org/abs/2604.16590](https://arxiv.org/abs/2604.16590)

    STORM提出了一种单阶段生成式AI数据同化框架，将数据同化重构为基于扩散模型的贝叶斯后验采样，并结合线性复杂度的全局注意力算法，在Frontier上扩展至74,400个GPU、达到6 ExaFLOPs持续吞吐量，并可在34秒内实现32,768成员的大规模集合不确定性量化。

    

    准确的地球系统预测需要从不完整的观测中进行状态推断，但传统的两阶段数据同化（DA）在计算上代价高昂，因为重复的基于偏微分方程（PDE）的集合预报、观测更新以及中间数据传输限制了高分辨率下的集合规模。我们提出了STORM，这是一个单阶段生成式AI框架，它将数据同化重新表述为基于扩散模型的贝叶斯后验采样，用可扩展的AI推断取代在线PDE集合预报。该框架进一步将时空Transformer与全局注意力算法相结合，通过可扩展的梯度传播将计算复杂度从二次方降低到线性，从而实现高分辨率、长上下文的地球系统建模。STORM在Frontier超级计算机上扩展至74,400个GPU，强扩展效率达96–99%，持续BF16吞吐量高达6 ExaFLOPs，同时能在34秒内完成32,768个成员的集合模拟以进行不确定性量化。

    arXiv:2604.16590v2 Announce Type: replace-cross  Abstract: Accurate Earth system prediction requires state inference from incomplete observations, but conventional two-stage data assimilation (DA) is computationally prohibitive because repeated PDE-based ensemble forecasts, observation updates, and intermediate data movement limit ensemble size at high resolution. We introduce STORM, a one-stage generative AI framework that reformulates DA as diffusion-based Bayesian posterior sampling, replacing online PDE ensemble forecasts with scalable AI inference. It further combines a spatiotemporal transformer with a global-attention algorithm that reduces complexity from quadratic to linear through scalable gradient propagation, enabling high-resolution, long-context Earth modeling. STORM scales to 74,400 GPUs on Frontier with 96--99\% strong-scaling efficiency and up to 6 ExaFLOPs sustained BF16 throughput, while enabling 32,768-member ensembles for uncertainty quantification in 34 seconds on
    
[^384]: 为什么微调会诱发幻觉以及如何修复它

    Why Fine-Tuning Encourages Hallucinations and How to Fix It

    [https://arxiv.org/abs/2604.15574](https://arxiv.org/abs/2604.15574)

    该论文提出一种基于自蒸馏的监督微调方法，通过正则化输出分布漂移，使模型在学习新事实的同时最大限度减少对预训练知识的幻觉，并证明在无需学习新知识时冻结参数组也能在保持任务性能的前提下降低幻觉。

    

    大语言模型容易产生与事实不符的幻觉陈述。这些错误的一个关键来源是监督微调（SFT）过程中接触到新的知识，这会增加相对于预训练期间所获知识的幻觉。由于这些错误是知识退化的副产品，我们探索能否利用已有的持续学习工具来缓解这一问题。我们提出了一种基于自蒸馏的SFT方法，通过正则化输出分布的漂移，在实现有效事实学习的同时，最大限度减少相对于已有知识的幻觉。我们还表明，当不需要获取新知识时，通过冻结参数组来抑制事实可塑性，可以在减少幻觉的同时保持任务性能。最后，我们研究了其内在机制，对比了容量限制、行为克隆和局部干扰等假说。我们的实验表明，主要的……

    arXiv:2604.15574v2 Announce Type: replace-cross  Abstract: Large language models are prone to hallucinating factually incorrect statements. A key source of these errors is exposure to new factual information through supervised fine-tuning (SFT), which can increase hallucinations w.r.t.~knowledge acquired during pre-training. Since these errors arise as a by-product of knowledge degradation, we explore whether established continual learning tools can mitigate them. We propose a self-distillation-based SFT method that facilitates effective factual learning while minimizing hallucinations w.r.t.~pre-existing knowledge by regularizing output-distribution drift. We also show that when new knowledge acquisition is unnecessary, suppressing factual plasticity by freezing parameter groups preserves task performance while reducing hallucinations. Lastly, we investigate the mechanism, contrasting capacity limitations, behavior cloning, and localized interference. Our experiments show that a main 
    
[^385]: 是什么驱动了表征转向？关于转向拒绝行为的机制案例研究

    What Drives Representation Steering? A Mechanistic Case Study on Steering Refusal

    [https://arxiv.org/abs/2604.08524](https://arxiv.org/abs/2604.08524)

    本研究通过多词元激活修补框架对LLM拒绝行为的转向机制进行案例研究，发现不同转向方法在同一层利用功能可互换的回路，且转向向量主要通过注意力机制的OV回路发挥作用而几乎不依赖QK回路。

    

    将转向向量应用于大型语言模型（LLM）是一种高效且有效的模型对齐技术，但我们对其工作原理缺乏可解释的解释——具体来说，转向向量影响了哪些内部机制，以及这如何导致不同的模型输出。为了探究转向向量有效性的因果机制，我们对拒绝行为进行了全面的案例研究。我们提出了一个多词元激活修补框架，并发现不同的转向方法在同一层应用时利用的是功能上可互换的回路。这些回路揭示出，转向向量主要通过OV回路与注意力机制交互，而在很大程度上忽略了QK回路。在转向过程中冻结所有注意力分数，在三个模型家族上仅导致8.83%的性能下降。对被转向OV回路的数学分解进一步揭示了……

    arXiv:2604.08524v2 Announce Type: replace-cross  Abstract: Applying steering vectors to large language models (LLMs) is an efficient and effective model alignment technique, but we lack an interpretable explanation for how it works--specifically, what internal mechanisms steering vectors affect and how this results in different model outputs. To investigate the causal mechanisms underlying the effectiveness of steering vectors, we conduct a comprehensive case study on refusal. We propose a multi-token activation patching framework and discover that different steering methodologies leverage functionally interchangeable circuits when applied at the same layer. These circuits reveal that steering vectors primarily interact with the attention mechanism through the OV circuit while largely ignoring the QK circuit. Freezing all attention scores during steering drops performance by only 8.83% across three model families. A mathematical decomposition of the steered OV circuit further reveals s
    
[^386]: 面向上下文密集型任务的KV缓存卸载

    KV Cache Offloading for Context-Intensive Tasks

    [https://arxiv.org/abs/2604.08426](https://arxiv.org/abs/2604.08426)

    该论文创建并发布了Text2JSON基准测试，揭示现代KV缓存卸载技术在需要从输入提示中提取大量信息的上下文密集型任务上，会导致Llama 3和Qwen 3模型出现显著的性能下降。

    

    随着各类应用对长上下文大语言模型（LLM）需求的不断增长，键值（KV）缓存已成为延迟和内存占用的关键瓶颈。近来，KV缓存卸载已成为一种有前景的方法，可在保持精度的同时减少内存占用和推理延迟。以往的评估工作主要集中于不需要从上下文中提取大量信息的任务。在本工作中，我们研究了KV缓存卸载在上下文密集型任务上的表现：即那些求解过程需要从输入提示中查找大量信息的问题。我们创建并发布了Text2JSON基准测试，这是一个高度上下文密集型的任务，需要从原始文本中提取结构化知识。我们在Text2JSON以及其他上下文密集型任务上对现代KV卸载技术进行了评估，发现Llama 3和Qwen 3模型均出现了显著的性能下降。我们的分析识别出两个关键原因（摘要原文在此处被截断）。

    arXiv:2604.08426v5 Announce Type: replace-cross  Abstract: With the growing demand for long-context LLMs across a wide range of applications, the key-value (KV) cache has become a critical bottleneck for both latency and memory usage. Recently, KV-cache offloading has emerged as a promising approach to reduce memory footprint and inference latency while preserving accuracy. Prior evaluations have largely focused on tasks that do not require extracting large amounts of information from the context. In this work, we study KV-cache offloading on context-intensive tasks: problems where the solution requires looking up a lot of information from the input prompt. We create and release the Text2JSON benchmark, a highly context-intensive task that requires extracting structured knowledge from raw text. We evaluate modern KV offloading on Text2JSON and other context-intensive tasks and find significant performance degradation on both Llama 3 and Qwen 3 models. Our analysis identifies two key re
    
[^387]: DiffHDR：利用视频扩散模型重新曝光LDR视频

    DiffHDR: Re-Exposing LDR Videos with Video Diffusion Models

    [https://arxiv.org/abs/2604.06161](https://arxiv.org/abs/2604.06161)

    DiffHDR将LDR视频到HDR的转换建模为视频扩散模型潜在空间中的生成式辐射修复任务，通过在对数-Gamma色彩空间中利用预训练视频扩散模型的时空生成先验，在过曝和欠曝区域合成逼真的HDR辐射，实现LDR视频的有效重新曝光。

    

    大多数数字视频以8位低动态范围（LDR）格式存储，由于饱和和量化的影响，原始高动态范围（HDR）场景辐射的大量信息已经丢失。这种高光和阴影细节的丢失使得无法将准确的亮度映射到HDR显示器上，也限制了后期制作工作流中有意义的重新曝光。尽管已有一些通过动态范围扩展将LDR图像转换为HDR的技术被提出，但它们难以在过曝和欠曝区域恢复逼真的细节。为了解决这一问题，我们提出了DiffHDR，该框架将LDR到HDR的转换形式化为视频扩散模型潜在空间中的生成式辐射修复任务。通过在对数-Gamma（Log-Gamma）色彩空间中运行，DiffHDR利用预训练视频扩散模型的时空生成先验，在过曝和欠曝区域合成合理的HDR辐射，同时恢复连续的场景辐射信息。

    arXiv:2604.06161v3 Announce Type: replace-cross  Abstract: Most digital videos are stored in 8-bit low dynamic range (LDR) formats, where much of the original high dynamic range (HDR) scene radiance is lost due to saturation and quantization. This loss of highlight and shadow detail precludes mapping accurate luminance to HDR displays and limits meaningful re-exposure in post-production workflows. Although techniques have been proposed to convert LDR images to HDR through dynamic range expansion, they struggle to restore realistic detail in the over- and underexposed regions. To address this, we present DiffHDR, a framework that formulates LDR-to-HDR conversion as a generative radiance inpainting task within the latent space of a video diffusion model. By operating in Log-Gamma color space, DiffHDR leverages spatio-temporal generative priors from a pretrained video diffusion model to synthesize plausible HDR radiance in over- and underexposed regions while recovering the continuous sce
    
[^388]: 基于散度采样的免训练流匹配精炼方法

    Training-Free Refinement of Flow Matching with Divergence-based Sampling

    [https://arxiv.org/abs/2604.04646](https://arxiv.org/abs/2604.04646)

    提出了流散度采样器（FDS），一种免训练的即插即用框架，通过边缘速度场的散度信号精炼中间状态，引导样本避开低密度区域，从而一致提升流匹配模型的生成保真度。

    

    基于流的模型通过建模边缘速度场来学习目标分布，该速度场被定义为将每个样本从简单先验连接到目标数据的样本级速度的平均值。然而，当样本级速度在同一中间状态发生冲突时，这种平均速度可能会将样本误导至低密度区域，从而降低生成质量。为了解决这一问题，我们提出了流散度采样器，这是一个免训练的框架，它在每个求解器步骤之前对中间状态进行精炼。我们的关键发现表明，这种误导的严重程度可以通过边缘速度场的散度来量化，而该散度在使用良好优化模型的推理过程中可以轻松计算。FDS利用这一信号将状态引导至歧义较小的区域。作为一个与标准求解器和现成流模型骨干兼容的即插即用框架，FDS在各种任务上一致地提升了生成保真度。

    arXiv:2604.04646v2 Announce Type: replace-cross  Abstract: Flow-based models learn a target distribution by modeling a marginal velocity field, defined as the average of sample-wise velocities connecting each sample from a simple prior to the target data. When sample-wise velocities conflict at the same intermediate state, however, this averaged velocity can misguide samples toward low-density regions, degrading generation quality. To address this issue, we propose the Flow Divergence Sampler (FDS), a training-free framework that refines intermediate states before each solver step. Our key finding reveals that the severity of this misguidance is quantified by the divergence of the marginal velocity field that is readily computable during inference with a well-optimized model. FDS exploits this signal to steer states toward less ambiguous regions. As a plug-and-play framework compatible with standard solvers and off-the-shelf flow backbones, FDS consistently improves fidelity across var
    
[^389]: TRU：面向高效多模态推荐遗忘的定向反向更新

    TRU: Targeted Reverse Update for Efficient Multimodal Recommendation Unlearning

    [https://arxiv.org/abs/2604.02183](https://arxiv.org/abs/2604.02183)

    该论文提出面向多模态推荐系统的即插即用定向反向更新（TRU）遗忘框架，通过针对待删除数据在排序行为、模态分支和模型模块间不均匀分布的影响进行定向处理，克服目标物品持续性、模态不平衡和模块敏感性集中三大瓶颈，实现高效的用户数据遗忘。

    

    多模态推荐系统（MRS）联合建模用户-物品交互图与丰富的物品内容，但这种紧密耦合使得用户数据一旦被学习后就难以移除。近似机器遗忘提供了一种替代完整重训的高效方案，然而当前的多模态推荐遗忘方法在模型各组件上大体均匀地应用反向更新。我们证明这种均匀处理方式与现代多模态推荐系统并不匹配：待删除数据的影响在排序行为、模态分支和模型模块之间的分布是不均匀的。这种不均匀性导致了多模态推荐遗忘中的三个瓶颈：目标物品在协同图中的持续性、特征分支间的模态不平衡，以及参数空间中集中于模块层面的敏感性。为解决这一不匹配问题，我们提出了定向反向更新（TRU），一个面向多模态推荐系统的即插即用遗忘框架。该方法不再采用均匀的（摘要在此处截断）

    arXiv:2604.02183v4 Announce Type: replace  Abstract: Multimodal recommendation systems (MRS) jointly model user-item interaction graphs and rich item content, but this tight coupling makes user data difficult to remove once learned. Approximate machine unlearning offers an efficient alternative to full retraining, yet current MRS unlearning applies reverse updates largely uniformly across model components. We show that this uniform treatment is misaligned with modern MRS: deleted-data influence is distributed unevenly across \textit{ranking behavior}, \textit{modality branches}, and \textit{model modules}. This non-uniformity gives rise to three bottlenecks in MRS unlearning: target-item persistence in the collaborative graph, modality imbalance across feature branches, and concentrated module-level sensitivity in the parameter space. To address this mismatch, we propose \textbf{targeted reverse update} (TRU), a plug-and-play unlearning framework for MRS. Instead of applying a uniform 
    
[^390]: IWP：在大型视觉语言模型中将Token剪枝视为隐式权重剪枝

    IWP: Token Pruning as Implicit Weight Pruning in Large Vision Language Models

    [https://arxiv.org/abs/2604.00757](https://arxiv.org/abs/2604.00757)

    该论文提出IWP框架，将注意力重新表述为由各token键值对生成的秩1外积之和构成的隐式线性层，从而把token剪枝转化为选择最优秩1更新子集以逼近原始对偶权重矩阵，并据此推导出同时衡量token信息量与信息冗余度的新指标，实现无需训练的高效视觉token剪枝。

    

    大型视觉语言模型在图像和视频理解任务中表现出色，但其计算成本会随着视觉token数量的增加而迅速增长。现有的token剪枝方法通过经验性手段缓解这一问题，却忽视了注意力机制的内部原理。本文提出了一种基于注意力对偶形式视角的新型无需训练的token剪枝框架。我们将注意力重新表述为一个隐式线性层，其权重矩阵是若干秩1外积之和，每个秩1外积由单个token的键值对生成。因此，token剪枝可以转化为选择这些秩1更新的最优子集，以最佳逼近原始的对偶权重矩阵。将这一视角扩展到大型视觉语言模型中的标准softmax注意力，我们推导出一种新的度量指标，能够同时量化token的信息量大小与信息冗余程度。为高效地选择子集……（原文截断）

    arXiv:2604.00757v3 Announce Type: replace-cross  Abstract: Large Vision Language Models show impressive performance across image and video understanding tasks, yet their computational cost grows rapidly with the number of visual tokens. Existing token pruning methods mitigate this issue through empirical approaches while overlooking the internal mechanism of attention. In this paper, we propose a novel training free token pruning framework grounded in the dual form perspective of attention. We reformulate attention as an implicit linear layer whose weight matrix is the sum of rank 1 outer products, each generated by a single token's key value pair. Token pruning thus reduces to selecting an optimal subset of these rank 1 updates that best approximates the original dual weight matrix. Extending this perspective to standard softmax attention in LVLMs, we derive a novel metric quantifying both a token's information magnitude and information duplication. To efficiently select the subset wi
    
[^391]: Oblivion：通过衰减驱动激活实现的自适应智能体记忆控制

    Oblivion: Self-Adaptive Agentic Memory Control through Decay-Driven Activation

    [https://arxiv.org/abs/2604.00131](https://arxiv.org/abs/2604.00131)

    Oblivion框架借鉴人类选择性遗忘机制，将遗忘建模为衰减驱动的可及性降低而非删除，并通过解耦读取路径（基于不确定性决定何时查询记忆）与写入路径（强化贡献性记忆），为LLM智能体实现按需动态加载的层次化记忆组织。

    

    人类记忆通过选择性遗忘来实现适应：经验会随时间推移变得不易获取，但可以通过强化或情境线索被重新激活。相比之下，记忆增强的LLM智能体依赖“始终开启”的检索和“扁平”的记忆存储，随着历史记录的增长会导致高干扰和高延迟。我们提出了Oblivion，一个将遗忘视为由衰减驱动的可及性降低——而非显式删除——的记忆控制框架。Oblivion将记忆控制解耦为读取和写入两条路径。读取路径基于智能体的不确定性和记忆缓冲区的效用，决定何时查询记忆，从而避免冗余的始终开启式访问。写入路径通过强化对生成响应有贡献的记忆，决定应该加强哪些内容。两者结合，实现了层次化的记忆组织，在保持持久性高级策略的同时按需动态加载细节。我们在静态和动态

    arXiv:2604.00131v3 Announce Type: replace-cross  Abstract: Human memory adapts through selective forgetting: experiences become less accessible over time but can be reactivated by reinforcement or contextual cues. In contrast, memory-augmented LLM agents rely on "always-on" retrieval and "flat" memory storage, causing high interference and latency as histories grow. We introduce Oblivion, a memory control framework that casts forgetting as decay-driven reductions in accessibility -- not explicit deletion. Oblivion decouples memory control into read and write paths. The read path decides when to consult memory, based on agent uncertainty and memory buffer utility, avoiding redundant always-on access. The write path decides what to strengthen, by reinforcing memories contributing to forming the response. Together, this enables hierarchical memory organization that maintains persistent high-level strategies while dynamically loading details as needed. We evaluate on both static and dynami
    
[^392]: VectorGym：面向SVG代码生成、草图绘制与编辑的多任务基准

    VectorGym: A Multi-Task Benchmark for SVG Code Generation, Sketching and Editing

    [https://arxiv.org/abs/2603.29852](https://arxiv.org/abs/2603.29852)

    VectorGym是首个与专业设计工作流程对齐的SVG综合性基准，包含四大任务（草图生成SVG、SVG编辑、文本生成SVG、SVG描述）及专家人工标注，并提供基于GRPO的多任务强化学习基线。

    

    我们提出了VectorGym，这是一个针对可伸缩矢量图形（SVG）的综合性基准套件，涵盖从文本和草图的生成、复杂编辑以及视觉理解。VectorGym解决了当前缺乏与专业设计工作流程相匹配的、真实且具有挑战性的基准这一问题。该基准包含四个带有专家人工标注的任务：新颖的Sketch2SVG任务（VG-Sketch）；全新的SVG编辑数据集（VG-Edit），其特色是涉及高阶图元的复杂多步骤编辑；Text2SVG生成任务（VG-Text）；以及SVG描述生成任务（VG-Cap）。与以往依赖合成编辑的基准不同，VectorGym提供了需要语义理解和设计意图的黄金标准人工标注。我们还提供了一个多任务强化学习基线，利用基于渲染的奖励在全部四个任务上进行联合优化。该基线基于GRPO并结合课程学习构建，用于训练Qwen3-VL 8B模型。

    arXiv:2603.29852v2 Announce Type: replace-cross  Abstract: We introduce VectorGym, a comprehensive benchmark suite for Scalable Vector Graphics (SVG) that spans generation from text and sketches, complex editing, and visual understanding. VectorGym addresses the lack of realistic, challenging benchmarks aligned with professional design workflows. Our benchmark comprises four tasks with expert human-authored annotations: the novel Sketch2SVG task (VG-Sketch); a new SVG editing dataset (VG-Edit) featuring complex, multi-step edits with higher-order primitives; Text2SVG generation (VG-Text); and SVG captioning (VG-Cap). Unlike prior benchmarks that rely on synthetic edits, VectorGym provides gold-standard human annotations that require semantic understanding and design intent. We also provide a multi-task reinforcement learning baseline that jointly optimizes across all four tasks using rendering-based rewards. This baseline, built on GRPO with curriculum learning, trains a Qwen3-VL 8B mo
    
[^393]: APEX-EM：基于结构化程序性-情景经验回放的自主智能体非参数化在线学习

    APEX-EM: Non-Parametric Online Learning for Autonomous Agents via Structured Procedural-Episodic Experience Replay

    [https://arxiv.org/abs/2603.29093](https://arxiv.org/abs/2603.29093)

    APEX-EM提出了一种无需更新模型权重的非参数化经验记忆方法，通过程序性知识图谱存储完整的任务轨迹并同时索引成功与失败经验，使LLM智能体能够复用过往经验而无需重复推理，在相同底层模型对比下BigCodeBench迁移任务上提升7.6个百分点。

    

    LLM智能体在执行每个任务时都需要重新运行完整的推理过程，即使是它们刚刚解决过的任务也不例外。我们提出了APEX-EM，这是一种非参数化经验记忆，它将完整的程序性-情景轨迹存储在类型化的程序性知识图谱（PKG）中，并通过三种通道进行检索：语义搜索、针对抽象操作序列的结构签名匹配以及图遍历。一个“规划-检索-生成-迭代-摄取”（PRGII）工作流负责生成经验、进行质量把关并提交经验，同时对成功和失败的经验都进行索引，使智能体学会哪些内容可以复用、哪些应当避免。在部署期间不改变任何模型权重。我们在五个基准上进行评估：BigCodeBench、KGQAGen-10k、HLE、Lifelong Agent Bench和ALFWorld。由于先前的工作使用了不同的底层模型，我们的结论基于相同底层模型的对比，以保持模型能力固定。在使用共享GPT-4o底层模型的BigCodeBench留出集迁移测试中，APEX-EM获得了+7.6个百分点的提升。

    arXiv:2603.29093v3 Announce Type: replace-cross  Abstract: LLM agents rerun full reasoning for every task, even one they solved moments earlier. We introduce \textbf{APEX-EM}, a non-parametric experience memory that stores complete procedural-episodic traces in a typed Procedural Knowledge Graph (PKG) and retrieves them through three channels: semantic search, structural-signature matching over abstract operation sequences, and graph traversal. A Plan-Retrieve-Generate-Iterate-Ingest (PRGII) workflow produces, quality-gates, and commits experiences, indexing both successes and failures so the agent learns what to reuse and what to avoid. No weights change during deployment.   We evaluate on five benchmarks: BigCodeBench, KGQAGen-10k, HLE, Lifelong Agent Bench, and ALFWorld. Because prior work uses different backbones, we base our claims on same-backbone comparisons that hold model capability fixed. On held-out BigCodeBench transfer with a shared GPT-4o backbone, APEX-EM gains +7.6\,pp 
    
[^394]: 揭示大型视觉语言模型中的多视角幻觉

    Revealing Multi-View Hallucination in Large Vision-Language Models

    [https://arxiv.org/abs/2603.23934](https://arxiv.org/abs/2603.23934)

    该论文首次揭示并定义了大型视觉语言模型中的多视角幻觉问题，构建了包含4.8k问答对的MVH-Bench基准进行系统评估，并提出无需训练的参考偏移对比解码（RSCD）技术，通过注意力掩码抑制视觉干扰，性能提升最高达94.8分。

    

    大型视觉语言模型（LVLMs）正越来越多地被应用于从不同视角捕获的多视角图像输入。尽管使用日益增多，当前的LVLMs常常因来自非目标实例或视角的视觉干扰而产生错误响应，我们将这种现象称为多视角幻觉（MVH）。为了系统地分析这一问题，我们构建了MVH-Bench，一个包含4.8k问答对的基准数据集，针对两种类型的幻觉：跨实例幻觉和跨视角幻觉。实证结果表明，MVH在近期的LVLMs中普遍存在。为解决这一问题，我们提出了参考偏移对比解码（RSCD），这是一种无需训练的解码技术，通过注意力掩码生成负对数几率来抑制视觉干扰。在MVH-Bench上使用LLaVA-OneVision和Qwen2.5-VL进行的实验表明，RSCD相比现有幻觉缓解方法的性能提升高达25.7分和94.8分。

    arXiv:2603.23934v2 Announce Type: replace-cross  Abstract: Large vision-language models (LVLMs) are increasingly being applied to multi-view image inputs captured from diverse viewpoints. Despite this growing use, current LVLMs often generate incorrect responses due to visual interference from non-target instances or viewpoints, a phenomenon we term multi-view hallucination (MVH). To systematically analyze this problem, we construct MVH-Bench, a benchmark comprising 4.8k question-answer pairs targeting two types of hallucination: cross-instance and cross-view. Empirical results show that MVH is prevalent across recent LVLMs. To address this issue, we propose Reference Shift Contrastive Decoding (RSCD), a training-free decoding technique that suppresses visual interference by generating negative logits through attention masking. Experiments on MVH-Bench with LLaVA-OneVision and Qwen2.5-VL demonstrate that RSCD improves performance by up to 25.7 and 94.8 points over existing hallucinatio
    
[^395]: MineDraft：一种批量并行投机解码框架

    MineDraft: A Framework for Batch Parallel Speculative Decoding

    [https://arxiv.org/abs/2603.18016](https://arxiv.org/abs/2603.18016)

    MineDraft提出一种批量并行投机解码框架，通过同时维护两批请求，将一批的草稿生成与另一批的验证重叠执行，有效隐藏草稿延迟，相比标准投机解码吞吐量最高提升75%、端到端延迟最高降低39%。

    

    投机解码（SD）通过使用较小的草稿模型提出草稿token，再由较大的目标模型进行验证，从而加速大语言模型的推理。然而，标准SD的性能往往受限于草稿生成与验证阶段的严格顺序执行。为解决这一问题，本文提出了MineDraft，一种批量并行投机解码（PSD）框架，旨在通过与验证过程重叠来有效隐藏草稿生成延迟。我们的理论分析表明，PSD比标准SD的效率显著更高。MineDraft通过一种新颖的批量并行设计实现了PSD：该设计同时维护两批请求，将一批请求的草稿生成与另一批请求的验证重叠进行。实验结果显示，与标准SD相比，MineDraft在吞吐量（最高提升75%）和端到端延迟（最高降低39%）方面均有显著改进。此外，我们还实现了……（摘要原文截断）

    arXiv:2603.18016v3 Announce Type: replace-cross  Abstract: Speculative decoding (SD) accelerates large language model inference by using a smaller draft model to propose draft tokens that are subsequently verified by a larger target model. However, the performance of standard SD is often limited by the strictly sequential execution of these drafting and verification stages. To address this, this paper proposes MineDraft, a batch parallel speculative decoding (PSD) framework designed to effectively hide drafting latency by overlapping it with verification. Our theoretical analysis shows that PSD is substantially more efficient than standard SD. MineDraft realizes the PSD through a novel batch-parallel design that maintains two batches of requests, overlapping drafting for one batch with verification for the other. Our experimental results show significant improvements of \alg{} in both throughput (up to 75%) and end-to-end latency (up to 39%) over standard SD. Furthermore, we have imple
    
[^396]: SCALE：用于虚拟细胞扰动预测的可扩展条件性图谱级端点传输模型

    SCALE:Scalable Conditional Atlas-Level Endpoint transport for virtual cell perturbation prediction

    [https://arxiv.org/abs/2603.17380](https://arxiv.org/abs/2603.17380)

    SCALE是一种将细胞表示为无序集合的条件传输模型，无需细胞级配对即可预测扰动后的细胞群体，在基因、化学、发育、免疫扰动及CRISPR数据上均优于现有方法。

    

    虚拟细胞模型旨在预测细胞群体如何响应扰动，但对照组和处理组细胞是以非配对群体的形式测量的，这使得学习扰动特异性效应变得复杂。我们提出了SCALE，一种条件传输模型，它将细胞表示为无序集合，无需细胞级别匹配即可预测处理后的细胞群体。共享的集合感知编码器和条件DiT骨干网络学习潜在传输，使端点监督直接与差异对齐，无需辅助的差异目标函数。在基因、化学、发育和免疫扰动等各类场景中，SCALE成功恢复了基因表达变化、响应方向和群体结构。在具有显著细胞系效应的CRISPR数据中，SCALE在七项指标上均优于竞争方法，并保持了基因靶点表示之间的区分度，而不是将它们坍缩到共享区域。SCALE还进一步对细胞因子进行了优先级排序预测……

    arXiv:2603.17380v3 Announce Type: replace-cross  Abstract: Virtual-cell models aim to predict how cell populations respond to perturbations, but control and treated cells are measured as unpaired populations, complicating the learning of perturbation-specific effects. We present SCALE, a conditional transport model that represents cells as unordered sets and predicts treated populations without cell-level matching. A shared set-aware encoder and conditional DiT backbone learn latent transport, making endpoint supervision directly delta-aligned without an auxiliary delta objective. Across genetic, chemical, developmental and immune perturbations, SCALE recovered gene-expression changes, response directions and population structure. In CRISPR data with dominant cell-line effects, SCALE outperformed competing methods across seven metrics and maintained separation among gene-target representations rather than collapsing them into a shared region. SCALE further prioritized cytokines predict
    
[^397]: V-Co：通过协同去噪深入探究视觉表征对齐

    V-Co: A Closer Look at Visual Representation Alignment via Co-Denoising

    [https://arxiv.org/abs/2603.16792](https://arxiv.org/abs/2603.16792)

    本文提出V-Co，在统一的JiT框架下对视觉协同去噪进行系统性研究，分离出使视觉表征对齐有效提升像素空间扩散模型训练的关键设计要素。

    

    像素空间扩散模型最近重新崛起，成为潜在扩散的有力替代方案，无需预训练自编码器即可实现高质量生成。然而，标准的像素空间扩散模型所获得的语义监督相对较弱，且并非为捕捉高层视觉结构而显式设计。近期的表征对齐方法（如 REPA）表明，预训练视觉特征可以显著改善扩散模型训练，而视觉协同去噪已成为将此类特征融入生成过程的一个有前景的方向。然而，现有的协同去噪方法往往将多种设计选择纠缠在一起，导致难以厘清哪些才是真正关键的因素。为此，我们提出了 V-Co，一个在统一的基于 JiT 的框架下对视觉协同去噪进行的系统性研究。这种受控的实验设置使我们能够分离出使视觉协同去噪有效的关键要素。我们的研究揭示了两个主要……

    arXiv:2603.16792v2 Announce Type: replace-cross  Abstract: Pixel-space diffusion has recently re-emerged as a strong alternative to latent diffusion, enabling high-quality generation without pretrained autoencoders. However, standard pixel-space diffusion models receive relatively weak semantic supervision and are not explicitly designed to capture high-level visual structure. Recent representation-alignment methods (e.g., REPA) suggest that pretrained visual features can substantially improve diffusion training, and visual co-denoising has emerged as a promising direction for incorporating such features into the generative process. However, existing co-denoising approaches often entangle multiple design choices, making it unclear which are truly essential. We therefore present V-Co, a systematic study of visual co-denoising in a unified JiT-based framework. This controlled setting allows us to isolate the ingredients that make visual co-denoising effective. Our study reveals two main 
    
[^398]: 人工标注是否必要？面向机器翻译错误片段检测的迭代MBR蒸馏

    Is Human Annotation Necessary? Iterative MBR Distillation for Error Span Detection in Machine Translation

    [https://arxiv.org/abs/2603.12983](https://arxiv.org/abs/2603.12983)

    该论文提出了一种基于最小贝叶斯风险解码的迭代MBR蒸馏自演化框架，通过利用现成大语言模型生成伪标签进行自我训练，无需人工标注即可在机器翻译错误片段检测任务上超越基于人工标注的监督基线模型。

    

    错误片段检测（ESD）是机器翻译（MT）评估中的一个关键子任务，旨在识别翻译错误的位置和严重程度。虽然基于人工标注数据对模型进行微调可以提升ESD性能，但获取这类数据成本高昂，且容易在标注者之间产生不一致。为解决这一问题，我们提出了一种基于最小贝叶斯风险（MBR）解码的新型自演化框架，命名为面向ESD的迭代MBR蒸馏，该方法利用现成的大语言模型生成伪标签，从而摆脱对人工标注的依赖。在WMT指标共享任务数据集上的大量实验表明，仅使用这些自生成伪标签训练的模型在系统级和片段级上均优于未适配的基础模型以及基于人工标注训练的监督基线模型，同时保持了具有竞争力的句子级性能。

    arXiv:2603.12983v4 Announce Type: replace-cross  Abstract: Error Span Detection (ESD) is a crucial subtask in Machine Translation (MT) evaluation, aiming to identify the location and severity of translation errors. While fine-tuning models on human-annotated data improves ESD performance, acquiring such data is expensive and prone to inconsistencies among annotators. To address this, we propose a novel self-evolution framework based on Minimum Bayes Risk (MBR) decoding, named Iterative MBR Distillation for ESD, which eliminates the reliance on human annotations by leveraging an off-the-shelf LLM to generate pseudo-labels. Extensive experiments on the WMT Metrics Shared Task datasets demonstrate that models trained solely on these self-generated pseudo-labels outperform both unadapted base model and supervised baselines trained on human annotations at the system and span levels, while maintaining competitive sentence-level performance.
    
[^399]: RetroReasoner：用于策略性逆合成预测的推理大语言模型

    RetroReasoner: A Reasoning LLM for Strategic Retrosynthesis Prediction

    [https://arxiv.org/abs/2603.12666](https://arxiv.org/abs/2603.12666)

    本文提出RetroReasoner，一个通过结构化切断理由的监督微调和往返奖励强化学习训练的逆合成推理大语言模型，能够显式模拟化学家的键切断策略思维并验证预测反应物的有效性。

    

    逆合成预测旨在识别能够合成给定产物分子的反应物。尽管分子大语言模型（LLMs）近期展现出令人期待的结果，但大多数现有方法要么直接生成反应物，要么仅提供通用的产物层面分析，而未能对键切断策略进行显式推理，以论证特定反应物的选择合理性。本文提出了RetroReasoner，一个能够捕捉化学家策略性切断思维的逆合成推理模型。RetroReasoner通过监督微调和强化学习进行训练。在监督微调方面，SyntheticRetro生成结构化的切断理由并与反应物预测配对。在强化学习方面，采用往返奖励机制，将预测的反应物输入正向合成模型进行评估，并对能够重构原始产物的预测给予奖励。

    arXiv:2603.12666v3 Announce Type: replace-cross  Abstract: Retrosynthesis prediction aims to identify reactants that can synthesize a given product molecule. Although molecular large language models (LLMs) have recently shown promising results, most existing methods either generate reactants directly or provide only generic product-level analysis, without explicitly reasoning about bond-disconnection strategies that justify specific reactant choices. This paper proposes RetroReasoner, a retrosynthetic reasoning model that captures chemists' strategic disconnection-based thinking. RetroReasoner is trained with supervised fine-tuning and reinforcement learning. For supervised fine-tuning, SyntheticRetro generates structured disconnection rationales paired with reactant predictions. For reinforcement learning, a round-trip reward evaluates predicted reactants by passing them through a forward synthesis model and rewarding predictions that reconstruct the original product. RetroReasoner ca
    
[^400]: HEAL：基于后见之明熵辅助学习的推理蒸馏

    HEAL: Hindsight Entropy-Assisted Learning for Reasoning Distillation

    [https://arxiv.org/abs/2603.10359](https://arxiv.org/abs/2603.10359)

    HEAL 提出了一种无需强化学习的推理蒸馏框架，通过熵动态检测推理断点并注入事后提示来修复失败轨迹，突破了传统拒绝采样造成的“教师天花板”，从而将大型推理模型的推理能力更有效地蒸馏到小模型中。

    

    将大型推理模型（LRM）的推理能力蒸馏到更小的模型中，通常受到拒绝采样局限性的制约。标准方法将教师模型视为静态过滤器，丢弃了教师模型无法独立探索出有效解的复杂“边角案例”问题，从而人为地为学生模型设置了一个“教师天花板”。在本工作中，我们提出了后见之明熵辅助学习，这是一个无需强化学习（RL-free）的框架，旨在弥合这一推理差距。借鉴最近发展区（ZPD）教育理论，HEAL 协同整合了三个核心模块：（1）引导式熵辅助修复（GEAR），一种主动干预机制，通过熵动态检测关键推理断点，并注入有针对性的事后提示来修复中断的推理轨迹；（2）困惑度-不确定性比率估计器（PURE），一种基于比率的过滤启发式方法，用于降低高异常……（摘要原文在此处截断）

    arXiv:2603.10359v2 Announce Type: replace  Abstract: Distilling reasoning capabilities from Large Reasoning Models (LRMs) into smaller models is typically constrained by the limitations of rejection sampling. Standard methods treat the teacher as a static filter, discarding complex "corner-case" problems where the teacher fails to explore valid solutions independently, thereby creating an artificial "Teacher Ceiling" for the student. In this work, we propose Hindsight Entropy-Assisted Learning (HEAL), an RL-free framework designed to bridge this reasoning gap. Drawing on the educational theory of the Zone of Proximal Development (ZPD), HEAL synergizes three core modules: (1) Guided Entropy-Assisted Repair (GEAR), an active intervention mechanism that detects critical reasoning breakpoints via entropy dynamics and injects targeted hindsight hints to repair broken trajectories; (2) Perplexity-Uncertainty Ratio Estimator (PURE), a ratio-based filtering heuristic that reduces high-anomaly 
    
[^401]: 面向视觉语言模型适配的引导式提示演化

    Guided Prompt Evolution for Vision-Language Models Adaptation

    [https://arxiv.org/abs/2603.09493](https://arxiv.org/abs/2603.09493)

    提出EvoPrompt框架，通过模态共享提示投影器生成层次化提示，并采用将低秩更新解耦为方向与幅值分量的演化训练策略，在保留预训练语义方向的前提下仅调整幅值，从而在有限标注数据下实现视觉语言模型无遗忘的知识保留适配。

    

    大规模视觉语言模型（VLM）在有限标注数据下向下游任务的适配仍然是一个重大挑战。尽管参数高效的提示学习方法提供了一条有前景的路径，但它们常常遭受预训练知识的灾难性遗忘。为解决这一局限，我们的工作基于这样一个洞察：控制提示的演化路径对于实现无遗忘的适配至关重要。为此，我们提出了EvoPrompt，这是一个旨在显式引导提示轨迹、实现知识保留微调的新型框架。具体而言，我们的方法采用模态共享提示投影器（MPP）从统一的嵌入空间生成层次化提示。关键在于，一种演化训练策略将低秩更新解耦为方向分量和幅值分量，在保留早期学到的语义方向的同时仅调整其幅值，从而使模型能够……

    arXiv:2603.09493v3 Announce Type: replace-cross  Abstract: The adaptation of large-scale vision-language models (VLMs) to downstream tasks with limited labeled data remains a significant challenge. While parameter-efficient prompt learning methods offer a promising path, they often suffer from catastrophic forgetting of pre-trained knowledge. Toward addressing this limitation, our work is grounded in the insight that governing the evolutionary path of prompts is essential for forgetting-free adaptation. To this end, we propose EvoPrompt, a novel framework designed to explicitly steer the prompt trajectory for knowledge-preserving fine-tuning. Specifically, our approach employs a Modality-Shared Prompt Projector (MPP) to generate hierarchical prompts from a unified embedding space. Critically, an evolutionary training strategy decouples low-rank updates into directional and magnitude components, preserving early-learned semantic directions while only adapting their magnitude, thus enabl
    
[^402]: 重构！而非编码：用于高可懂度和低延迟流式神经音频编解码器的自监督表征重构损失

    Reconstruct! Don't Encode: Self-Supervised Representation Reconstruction Loss for High-Intelligibility and Low-Latency Streaming Neural Audio Codec

    [https://arxiv.org/abs/2603.05887](https://arxiv.org/abs/2603.05887)

    提出自监督表征重构损失（SSRR）用于训练神经音频编解码器，显著加速收敛并提升语音可懂度，使 JHCodec 在零前瞻、低延迟的流式架构下于 LibriSpeech test-clean 上取得最佳的词错误率和字错误率。

    

    针对梅尔频谱图重构优化的神经音频编解码器通常无法保持语音可懂度。虽然语义编码器蒸馏可以改善编码后的表征，但并不能保证重构语音中的内容得以保留。在本工作中，我们证明了自监督表征重构损失从根本上改进了编解码器的训练和性能。首先，SSRR 显著加速了收敛，使得在单块 H200 GPU 上经过 30 万步训练后即可获得有竞争力的结果。其次，它通过从编解码器输出中重构蒸馏的自监督表征来增强可懂度。第三，SSRR 使基于 Transformer 的流式编解码器在无需额外前瞻的情况下实现高可懂度，从而支持零前瞻架构用于实时部署。在 LibriSpeech test-clean 数据集上，JHCodec 在所评估的编解码器中取得了最佳的词错误率（WER）和字错误率（CER），同时保持零前瞻和低延迟。

    arXiv:2603.05887v2 Announce Type: replace-cross  Abstract: Neural audio codecs optimized for mel-spectrogram reconstruction often fail to preserve intelligibility. While semantic encoder distillation improves encoded representations, it does not guarantee content preservation in reconstructed speech. In this work, we demonstrate that self-supervised representation reconstruction (SSRR) loss fundamentally improves codec training and performance. First, SSRR significantly accelerates convergence, enabling competitive results after 300k training steps on a single H200 GPU. Second, it enhances intelligibility by reconstructing distilled self-supervised representations from codec outputs. Third, SSRR enables high intelligibility without additional lookahead in streaming Transformer-based codecs, allowing a zero-lookahead architecture for real-time deployment. On LibriSpeech test-clean, JHCodec achieves the best WER and CER among the evaluated codecs while maintaining zero lookahead and low 
    
[^403]: 面向科学的MMAI Gym：训练用于药物发现的液体基础模型

    MMAI Gym for Science: Training Liquid Foundation Models for Drug Discovery

    [https://arxiv.org/abs/2603.03517](https://arxiv.org/abs/2603.03517)

    本文提出MMAI Gym for Science一站式训练框架，通过教会基础模型“分子的语言”，训练出更小规模的液体基础模型（LFM），在分子优化、ADMET预测等药物发现任务上超越了规模大得多的通用或专业模型。

    

    摘要：依赖上下文学习的通用大型语言模型（LLM）无法可靠地提供药物发现任务所需的科学理解和性能。仅仅增加模型规模或引入推理标记并不能带来显著的性能提升。为了解决这一差距，我们推出了面向科学的MMAI Gym（MMAI Gym for Science），这是一个一站式平台，提供分子数据格式与模态，以及面向特定任务的推理、训练和基准测试方案，旨在教会基础模型“分子的语言”，从而解决实际的药物发现问题。我们使用MMAI Gym训练了一个高效的液体基础模型（LFM）用于这些应用，证明了更小规模、有针对性训练的基础模型在分子基准测试中能够超越规模大得多的通用模型或专业模型。在关键的药物发现任务中——包括分子优化、ADMET性质预测等……

    arXiv:2603.03517v2 Announce Type: replace-cross  Abstract: General-purpose large language models (LLMs) that rely on in-context learning do not reliably deliver the scientific understanding and performance required for drug discovery tasks. Simply increasing model size or introducing reasoning tokens does not yield significant performance gains. To address this gap, we introduce the MMAI Gym for Science, a one-stop shop molecular data formats and modalities as well as task-specific reasoning, training, and benchmarking recipes designed to teach foundation models the 'language of molecules' in order to solve practical drug discovery problems. We use MMAI Gym to train an efficient Liquid Foundation Model (LFM) for these applications, demonstrating that smaller, purpose-trained foundation models can outperform substantially larger general-purpose or specialist models on molecular benchmarks. Across essential drug discovery tasks - including molecular optimization, ADMET property predictio
    
[^404]: 信道自适应边缘AI：通过将计算复杂度适配于信道状态以最大化推理吞吐量

    Channel-Adaptive Edge AI: Maximizing Inference Throughput by Adapting Computational Complexity to Channel States

    [https://arxiv.org/abs/2603.03146](https://arxiv.org/abs/2603.03146)

    本文提出了一个可处理的端到端推理精度分析模型，并据此设计了信道自适应AI算法，通过根据信道状态动态调整模型计算复杂度（利用早退机制），在时延和精度约束下最大化边缘推理吞吐量。

    

    通信与计算一体化（IC²）已成为在第六代（6G）网络中实现高效边缘推理的新范式。然而，由于缺乏一个可处理的理论框架来表征端到端（E2E）推理性能，IC²技术的设计受到了阻碍。该指标非常复杂，因为它需要同时考虑信道失真以及人工智能（AI）模型架构和计算复杂度。在这项工作中，我们通过开发一个可处理的端到端推理精度分析模型来应对这一挑战，并利用该模型设计了一种信道自适应AI算法，在时延和精度约束下最大化推理吞吐量（即边缘处理速率，EPR）。具体而言，我们考虑了一种边缘推理系统，其中服务器部署具有早退机制的骨干模型，从而实现灵活的计算（摘要在此处截断）。

    arXiv:2603.03146v2 Announce Type: replace-cross  Abstract: \emph{Integrated communication and computation} (IC$^2$) has emerged as a new paradigm for enabling efficient edge inference in sixth-generation (6G) networks. However, the design of IC$^2$ technologies is hindered by the lack of a tractable theoretical framework for characterizing \emph{end-to-end} (E2E) inference performance. The metric is highly complicated as it needs to account for both channel distortion and artificial intelligence (AI) model architecture and computational complexity. In this work, we address this challenge by developing a tractable analytical model for E2E inference accuracy and leveraging it to design a \emph{channel-adaptive AI} algorithm that maximizes inference throughput, referred to as the edge processing rate (EPR), under latency and accuracy constraints. Specifically, we consider an edge inference system in which a server deploys a backbone model with early exit, which enables flexible computatio
    
[^405]: 制造一些噪声：基于潜在空间扰动的无监督遥感变化检测

    Make Some Noise: Unsupervised Remote Sensing Change Detection Using Latent Space Perturbations

    [https://arxiv.org/abs/2602.19881](https://arxiv.org/abs/2602.19881)

    提出MaSoN框架，在训练中直接于潜在特征空间内根据目标数据的特征统计动态合成多样化变化，摆脱了对预定义变化类型假设的依赖，从而提升无监督遥感变化检测在罕见和复杂场景下的泛化能力。

    

    无监督遥感变化检测（UCD）旨在不依赖标注训练数据的情况下，定位同一区域两幅图像之间的变化。近期的大多数方法要么以免训练的方式使用冻结的基础模型，要么使用在像素空间中生成的合成变化进行训练。这两种策略都固有地依赖于对变化类型的预定义假设，这些假设通常通过手工设计的规则、外部数据集或辅助生成模型引入。由于这些假设的存在，此类方法难以泛化到少数几种变化类型之外，限制了其实际应用，尤其是在罕见或复杂场景中。为解决这一问题，我们提出了MaSoN（Make Some Noise），一个端到端的无监督变化检测框架，在训练过程中直接在潜在特征空间中合成多样化的变化。该框架根据目标数据的特征统计动态估计并生成变化，从而实现多样化且数据驱动的变化模拟

    arXiv:2602.19881v2 Announce Type: replace-cross  Abstract: Unsupervised remote sensing change detection (UCD) aims to localise changes between two images of the same region without relying on labelled training data. Most recent approaches either use a frozen foundation model in a training-free manner or train with synthetic changes generated in pixel space. Both strategies inherently rely on predefined assumptions about change types, typically introduced through handcrafted rules, external datasets, or auxiliary generative models. Due to these assumptions, such methods fail to generalise beyond a few change types, limiting their real-world usage, especially in rare or complex scenarios. To address this, we propose MaSoN (Make Some Noise), an end-to-end UCD framework that synthesises diverse changes directly in the latent feature space during training. It generates changes dynamically estimated from feature statistics of the target data, enabling diverse yet data-driven variation aligne
    
[^406]: 学习记忆：面向长上下文推理的记忆智能体端到端训练

    Learning to Remember: End-to-End Training of Memory Agents for Long-Context Reasoning

    [https://arxiv.org/abs/2602.18493](https://arxiv.org/abs/2602.18493)

    该论文提出统一记忆智能体（UMA），通过任务分层GRPO算法端到端训练单一策略来维护可复用的结构化外部记忆库，并配套提出Ledger-QA诊断基准，显著提升了长上下文推理中跨会话状态跟踪与问答的性能。

    

    长上下文大语言模型和检索增强生成将状态跟踪与证据整合推迟到查询时刻，当事实发生演变且答案依赖于潜在状态时，这种方式十分脆弱。我们提出了统一记忆智能体（UMA）来应对一对多的场景：与查询无关的外部记忆从数据流中一次性构建，并可在多个未来的问答会话中重复使用。单一策略通过增删改查（CRUD）操作维护一个结构化的记忆库，并结合记忆库与原始上下文进行回答。任务分层GRPO（Task-Stratified GRPO）利用从每个采样记忆状态分支出的问答轨迹的平均奖励来监督记忆维护，同时对记忆组与逐问题问答组分别进行归一化。我们还提出了Ledger-QA，一个针对累积更新之上长程状态跟踪的诊断性基准。在16k预算下，UMA-Generalist在测试时学习与准确性方面，在所有对比方法中取得了最高的平均分数。

    arXiv:2602.18493v2 Announce Type: replace-cross  Abstract: Long-context LLMs and Retrieval-Augmented Generation defer state tracking and evidence consolidation to query time, which is brittle when facts evolve and answers depend on latent states. We introduce Unified Memory Agent (UMA) for a one-to-many setting: query-agnostic external memory is constructed once from a stream and reused across multiple future QA sessions. A single policy maintains a structured Memory Bank through CRUD operations and answers using both the Memory Bank and raw context. Task-Stratified GRPO uses the mean reward of QA trajectories branching from each sampled memory state to supervise memory maintenance, while normalizing memory and per-question QA groups separately. We also introduce Ledger-QA, a diagnostic benchmark for long-horizon state tracking over accumulated updates. At the 16k budget, UMA-Generalist achieves the highest average score among compared methods across the test-time-learning and accurate
    
[^407]: 本体引导的神经符号推理：用数学领域知识为语言模型提供形式化基础

    Ontology-Guided Neuro-Symbolic Inference: Grounding Language Models with Mathematical Domain Knowledge

    [https://arxiv.org/abs/2602.17826](https://arxiv.org/abs/2602.17826)

    该论文提出一种结合OpenMath本体、混合检索与交叉编码器重排序的神经符号流水线，将数学领域知识注入语言模型提示中，实验表明高质量检索的本体上下文能提升模型在MATH基准上的表现，但不相关上下文会损害性能。

    

    语言模型存在根本性的局限——幻觉、脆弱性以及缺乏形式化基础——这些问题在需要可验证推理的高风险专业领域尤为突出。本研究探讨了形式化领域领域能否通过检索增强生成来提升语言模型的可靠性。以数学作为概念验证，作者实现了一个神经符号流水线，利用OpenMath本体，结合混合检索和交叉编码器重排序技术，将相关定义注入模型提示中。在MATH基准上使用三个开源模型进行的评估表明，当检索质量较高时，本体引导的上下文能够提升性能，但不相关的上下文反而会显著降低性能——这一结果既展现了神经符号方法的前景，也揭示了其面临的挑战。

    arXiv:2602.17826v2 Announce Type: replace  Abstract: Language models exhibit fundamental limitations -- hallucination, brittleness, and lack of formal grounding -- that are particularly problematic in high-stakes specialist fields requiring verifiable reasoning. I investigate whether formal domain ontologies can enhance language model reliability through retrieval-augmented generation. Using mathematics as proof of concept, I implement a neuro-symbolic pipeline leveraging the OpenMath ontology with hybrid retrieval and cross-encoder reranking to inject relevant definitions into model prompts. Evaluation on the MATH benchmark with three open-source models reveals that ontology-guided context improves performance when retrieval quality is high, but irrelevant context actively degrades it -- highlighting both the promise and challenges of neuro-symbolic approaches.
    
[^408]: 持续熵作为相变的探测器

    Persistent Entropy as a Detector of Phase Transitions

    [https://arxiv.org/abs/2602.09058](https://arxiv.org/abs/2602.09058)

    本文建立了与模型无关的理论定理，通过识别持续权重中的“分散-凝聚”机制并推导出两状态间熵差的显式高概率下界，首次为利用持续熵检测相变提供了严格的理论保证，并据此证明卷积网络学习滤波器的环形组织源于一次尖锐的拓扑相变。

    

    持续熵是持续性条形码的一种标量摘要，被广泛用于检测状态变化，然而目前尚无理论阐明条形码中的结构性变化何时必然会导致可检测的熵变化。我们建立了一个与模型无关的定理来提供此类条件。通过将持久图视为由控制参数索引的随机对象，我们在归一化持久权重中识别出一种“分散-凝聚”机制，并推导出两种状态之间熵差的显式下界，该下界在有限样本量下以高概率成立，且对条形寿命的绝对尺度不敏感。我们还给出了一套在经验条形码上验证这些假设的程序。应用于卷积网络时，该准则表明 Gabrielsson 和 Carlsson 所报告的学习滤波器的环形组织是通过一次尖锐的拓扑相变而产生的，并定位了该相变的发生起点。

    arXiv:2602.09058v2 Announce Type: replace-cross  Abstract: Persistent entropy is a scalar summary of persistence barcodes widely used to detect regime changes, yet there is no account of when a structural change in a barcode must produce a detectable change in entropy. We establish a model-agnostic theorem supplying such conditions. Treating persistence diagrams as random objects indexed by a control parameter, we identify a dispersion-condensation mechanism in the normalized persistence weights and derive an explicit lower bound on the entropy difference between the two regimes, valid with high probability at finite sample size and insensitive to the absolute scale of bar lifetimes. We also give a procedure for verifying the hypotheses on empirical barcodes. Applied to convolutional networks, the criterion shows that the circular organization of learned filters reported by Gabrielsson and Carlsson emerges through a sharp topological phase transition, and locates its onset: within a fe
    
[^409]: MAS-ProVe：理解多智能体系统的过程验证

    MAS-ProVe: Understanding the Process Verification of Multi-Agent Systems

    [https://arxiv.org/abs/2602.03053](https://arxiv.org/abs/2602.03053)

    本文提出MAS-ProVe，首次对多智能体系统中的过程验证展开系统性实证研究，涵盖三种验证范式、两种验证粒度、五种验证器和四种上下文管理策略，并发现过程级验证在多智能体系统中并不能持续稳定地带来改进。

    

    基于大语言模型构建的多智能体系统在推理轨迹上往往表现出较高的方差。过程验证通过评估轨迹中的中间步骤，已在一般推理场景中展现出潜力，并被认为可能成为指导多智能体系统协调的工具；然而，其在多智能体系统中的实际有效性仍不明确。为填补这一空白，我们提出了MAS-ProVe，一项针对多智能体系统过程验证的系统性实证研究。我们的研究涵盖三种验证范式（LLM作为评判者、奖励模型和过程奖励模型），并在两个验证粒度层级（智能体级和迭代级）上进行评估。我们进一步考察了五种代表性验证器和四种上下文管理策略，并在多个推理基准上对六种不同的多智能体框架开展了实验。我们发现过程级验证并不能持续稳定地改进……（原文摘要至此截断）

    arXiv:2602.03053v2 Announce Type: replace  Abstract: Multi-Agent Systems (MAS) built on Large Language Models (LLMs) often exhibit high variance in their reasoning trajectories. Process verification, which evaluates intermediate steps in trajectories, has shown promise in general reasoning settings, and has been suggested as a potential tool for guiding coordination of MAS; however, its actual effectiveness in MAS remains unclear. To fill this gap, we present MAS-ProVe, a systematic empirical study of process verification for multi-agent systems (MAS). Our study spans three verification paradigms (LLM-as-a-Judge, reward models, and process reward models), evaluated across two levels of verification granularity (agent-level and iteration-level). We further examine five representative verifiers and four context management strategies, and conduct experiments over six diverse MAS frameworks on multiple reasoning benchmarks. We find that process-level verification does not consistently impr
    
[^410]: 像医生一样思考：通过探索诊断知识图谱实现对话式诊断

    Think Like a Doctor: Conversational Diagnosis through the Exploration of Diagnostic Knowledge Graphs

    [https://arxiv.org/abs/2602.01995](https://arxiv.org/abs/2602.01995)

    该论文提出了一种通过探索诊断知识图谱进行两步推理（先生成诊断假设、再通过澄清性问题反复验证）的对话式诊断系统，并结合基于人设的患者模拟器PatientSim与MIMIC-IV患者档案进行更贴近真实场景的评估。

    

    对话式诊断需要进行多轮问诊，即智能体在信息不完整的情况下通过提出澄清性问题来逐步细化鉴别诊断。现有方法通常依赖于模型的参数化知识，或者假设患者能够提供丰富而具体的信息，这在现实中是不切实际的。为了解决这些局限性，我们提出了一种对话式诊断系统，该系统通过探索诊断知识图谱进行两步推理：(i) 从对话上下文中生成诊断假设；(ii) 通过澄清性问题验证假设，这一过程循环往复，直到得出最终诊断。由于评估该系统需要一个能够对系统提问做出回应的真实患者模拟器，我们采用了基于人设的患者模拟器PatientSim，并结合MIMIC-IV中的患者档案。我们进一步对其进行了改进，加入低特异性症状报告机制，以反映真实世界中患者的……（原文摘要不完整）

    arXiv:2602.01995v2 Announce Type: replace  Abstract: Conversational diagnosis requires multi-turn history-taking, where an agent asks clarifying questions to refine differential diagnoses under incomplete information. Existing approaches often rely on the parametric knowledge of a model or assume that patients provide rich and concrete information, which is unrealistic. To address these limitations, we propose a conversational diagnosis system that explores a diagnostic knowledge graph to reason in two steps: (i) generating diagnostic hypotheses from the dialogue context, and (ii) verifying hypotheses through clarifying questions, which are repeated until a final diagnosis is reached. Since evaluating the system requires a realistic patient simulator that responds to the system's questions, we adopt PatientSim, a persona-driven patient simulator, together with patient profiles from MIMIC-IV. We further adapt it with low-specificity symptom reporting to reflect how real-world patients d
    
[^411]: FloydNet：一种全局关系推理的学习范式

    FloydNet: A Learning Paradigm for Global Relational Reasoning

    [https://arxiv.org/abs/2601.19094](https://arxiv.org/abs/2601.19094)

    提出FloydNet与关键注意力机制（PA），借鉴Floyd–Warshall的“配对-枢轴”结构，通过维护有序对状态并在枢轴上聚合候选关系实现全局关系推理，并推广为支持有序k元组的k-FloydNet框架，其图判别能力与对应的WL同构测试相当。

    

    学习算法计算通常需要显式的关系中间状态，然而许多图处理器将其主要状态维持在单个实体上。我们提出了FloydNet和关键注意力，它维护有序对状态，并通过在由每个枢轴 j 所构成的 候选上进行注意力操作来更新目标关系。受Floyd–Warshall算法的“配对-枢轴”结构启发，PA以并行方式学习关系组合与枢轴加权，而非执行其有序的min-plus递归。k-FloydNet框架将这一操作扩展到有序k元组，在注意力操作层面上，自注意力和PA分别是其k=1和k=2的特例。在原子元组初始化和不变读出的条件下，我们证明k-FloydNet的图判别能力不超过k-FWL；在BREC基准上，每个被评估的变体都与其对应WL参考的成功集合相匹配。

    arXiv:2601.19094v3 Announce Type: replace-cross  Abstract: Learning algorithmic computation often requires explicit relational intermediate states, yet many graph processors maintain their primary states on individual entities. We introduce \fnet and \textbf{Pivotal Attention} (PA), which maintain ordered pair states and update a target relation $(i,k)$ by attending over candidates formed from $(i,j)$ and $(j,k)$ for every pivot $j$. Motivated by the pair-and-pivot structure of Floyd--Warshall, PA learns relation composition and pivot weighting in parallel rather than executing its ordered min-plus recurrence. The \kfnet{k} framework extends this operation to ordered $k$-tuples, with Self-Attention and PA as its $k=1$ and $k=2$ cases at the attention-operation level. Under atomic tuple initialization and invariant readout, we show that \kfnet{k} is no more graph-discriminative than k-FWL; on BREC, each evaluated variant matches the success set of its corresponding WL reference. \fnet f
    
[^412]: LifeAgentBench：面向长时程、跨维度生活方式健康推理的大语言模型基准测试

    LifeAgentBench: Benchmarking LLMs for Long-Horizon, Cross-Dimensional Lifestyle Health Reasoning

    [https://arxiv.org/abs/2601.13880](https://arxiv.org/abs/2601.13880)

    本文提出LifeAgentBench——一个包含22,573个问题、面向长时程、跨维度、多用户生活方式健康推理的大规模问答基准，通过系统评估13个代表性大语言模型，揭示了它们在长时程聚合和跨维度推理上的关键瓶颈。

    

    个性化生活方式健康分析需要对异构生活方式信号进行长时程、多维度的推理，而移动感知和大语言模型（LLM）的最新进展使这种支持日益可行。然而，由于缺乏系统性的基准测试，当前LLM在这种场景下的能力仍未被充分理解。在本文中，我们提出了LifeAgentBench，这是一个面向长时程、跨维度、多用户生活方式健康推理的大规模问答基准，包含22,573个问题，涵盖从基础检索到复杂推理的各个层次。我们发布了一个可扩展的基准构建流程和标准化的评估协议，通过可执行的查询和程序推导出可验证的答案，以支持可靠的评估。随后，我们在LifeAgentBench上系统评估了13个代表性LLM，并识别出模型在长时程聚合和跨维度推理方面的关键瓶颈。

    arXiv:2601.13880v2 Announce Type: replace  Abstract: Personalized lifestyle health analysis requires long-horizon, multi-dimensional reasoning over heterogeneous lifestyle signals, and recent advances in mobile sensing and large language models (LLMs) make such support increasingly feasible. However, the capabilities of current LLMs in this setting remain insufficiently understood due to the lack of systematic benchmarks. In this paper, we introduce LifeAgentBench, a large-scale QA benchmark for long-horizon, cross-dimensional, and multi-user lifestyle health reasoning, containing 22,573 questions spanning from basic retrieval to complex reasoning. We release an extensible benchmark construction pipeline and a standardized evaluation protocol, deriving verifiable answers through executable queries and programs to support reliable assessment. We then systematically evaluate 13 representative LLMs on LifeAgentBench and identify key bottlenecks in long-horizon aggregation and cross-dimens
    
[^413]: 超越静态摘要：面向LLM智能体的主动记忆提取

    Beyond Static Summarization: Proactive Memory Extraction for LLM Agents

    [https://arxiv.org/abs/2601.04463](https://arxiv.org/abs/2601.04463)

    该论文提出主动记忆提取框架ProMem，通过分离细节、事件与关系并采用分类提取策略、完整性检查和原子级事实验证，解决了现有记忆提取方法因提前进行和一次性提取而导致的信息丢失与幻觉残留问题。

    

    记忆管理对于LLM智能体在长期和个性化交互中至关重要。以往的大多数工作研究如何检索和使用记忆，但较少关注记忆是如何提取的。我们发现了现有方法的两个主要局限。首先，提取是“提前进行的”：智能体在了解未来任务之前就保存信息，而单一的摘要提示往往会混淆细节、事件和关系，导致有用信息的丢失。其次，提取通常是一次性的，在没有验证的情况下，错误和幻觉可能会长期留在记忆中。为了解决这些局限，我们提出了ProMem，一个主动记忆提取框架。它将细节、事件和关系分离，并对每种类型采用不同的提取策略。它还通过完整性检查来恢复遗漏的事件，并在原子级别验证事实以减少幻觉。实验表明，ProMem提高了记忆完整性和问答准确率。

    arXiv:2601.04463v2 Announce Type: replace-cross  Abstract: Memory management is vital for LLM agents in long-term and personalized interactions. Most previous work studies how to retrieve and use memory, but pays less attention to how memory is extracted. We find two main limitations in existing methods. First, extraction is "ahead-of-time": the agent saves information before it knows future tasks. A single summary prompt often mixes details, events, and relations, so useful information is lost. Second, extraction is usually one-off. Without verification, errors and hallucinations may stay in memory for a long time. To address these limitations, we propose ProMem, a proactive memory extraction framework. It separates details, events, and relations, and uses different extraction strategies for each type. It also checks completeness to recover missed events and verifies facts at the atomic level to reduce hallucinations. Experiments show that ProMem improves memory completeness and QA ac
    
[^414]: 一种结合多智能体仿真、分层SIEM关联与心智理论推理的混合式内部威胁检测框架

    A Hybrid Insider Threat Detection Framework Combining Multi-Agent Simulation, Layered SIEM Correlation, and Theory-of-Mind Reasoning

    [https://arxiv.org/abs/2601.04243](https://arxiv.org/abs/2601.04243)

    该论文提出一种融合多智能体仿真、分层SIEM关联、证据门控与心智理论推理的混合式内部威胁检测框架，将电子邮件作为社会工程证据流与系统事件关联分析，使参与者级F1分数从0.567显著提升至0.944并大幅降低误报。

    

    本文提出了一种面向企业环境的混合式内部威胁检测框架，融合了多智能体仿真、分层SIEM关联、信任自适应阈值、行为与通信取证以及心智理论推理。电子邮件不被视为控制通道，而是作为与认证、文件访问和权限事件相互关联的协调活动与社会工程证据流。论文评估了四种变体：分层SIEM核心、认知增强型SIEM (CE-SIEM)、证据门控SIEM (EG-SIEM)以及采用经安然语料库校准的邮件取证的EG-SIEM-Enron。在包含八名恶意内部人员的十次匹配运行中，参与者级别的F1分数从LSC的0.567提升至CE-SIEM的0.774、EG-SIEM的0.898以及EG-SIEM-Enron的0.944；经Holm-Bonferroni校正后，配对Wilcoxon检验证实了前三项差异的显著性。证据门控将每次运行的确认误报数从LSC下的33.7降低至……（原文摘要在此处截断）

    arXiv:2601.04243v2 Announce Type: replace-cross  Abstract: This paper presents a hybrid insider threat detection framework for enterprise environments, integrating multi-agent simulation, layered SIEM correlation, trust-adaptive thresholds, behavioral and communication forensics, and Theory-of-Mind reasoning. Email is treated not as a control channel but as a coordination and social-engineering evidence stream correlated with authentication, file-access, and privilege events. Four variants are evaluated: Layered SIEM-Core (LSC), Cognitive-Enriched SIEM (CE-SIEM), Evidence-Gated SIEM (EG-SIEM), and EG-SIEM-Enron with Enron-calibrated email forensics. Across ten matched runs with eight malicious insiders, actor-level F1 improves from 0.567 for LSC to 0.774 for CE-SIEM, 0.898 for EG-SIEM, and 0.944 for EG-SIEM-Enron; paired Wilcoxon tests confirm the first three differences after Holm-Bonferroni correction. Evidence gating reduces confirmed false positives from 33.7 per run under LSC and 
    
[^415]: 针对基于Mamba的语言模型的隐状态投毒攻击

    Hidden State Poisoning Attacks against Mamba-based Language Models

    [https://arxiv.org/abs/2601.01972](https://arxiv.org/abs/2601.01972)

    该论文首次揭示了针对Mamba等状态空间语言模型的隐状态投毒攻击（HiSPA）——特定短输入短语可不可逆地覆盖模型隐藏状态导致部分失忆，并提出RoBench-25基准证实了包括520亿参数的Jamba混合模型在内的SSMs对此类攻击的脆弱性，而纯Transformer模型则不受影响。

    

    像Mamba这样的状态空间模型（SSMs）以线性时间复杂度为基于Transformer的语言模型提供了高效替代方案。然而，其对抗鲁棒性却鲜有研究。本文研究了特定短输入短语通过不可逆地覆盖模型隐藏状态中的信息，从而在此类模型中诱发部分“失忆”效应的现象，我们将其称为隐状态投毒攻击。我们提出的基准测试RoBench-25可以评估模型在遭受HiSPA攻击时的信息检索能力，并证实了SSMs对此类攻击的脆弱性。即使是最近的Jamba-1.7-Mini SSM-Transformer混合模型（520亿参数），在某些HiSPA触发器作用下也会在RoBench-25上完全失效，而纯Transformer模型则不会。我们还观察到，与纯Transformer不同，HiSPA触发器在流行的Open-Prompt-Injections基准测试中显著削弱了Jamba模型的表现。我们进一步表明，该理（摘要原文在此处截断）

    arXiv:2601.01972v5 Announce Type: replace-cross  Abstract: State space models (SSMs) like Mamba offer efficient alternatives to Transformer-based language models, with linear time complexity. Yet, their adversarial robustness remains critically unexplored. This paper studies the phenomenon whereby specific short input phrases induce a partial amnesia effect in such models, by irreversibly overwriting information in their hidden states, referred to as a Hidden State Poisoning Attack (HiSPA). Our benchmark RoBench-25 allows evaluating a model's information retrieval capabilities when subject to HiSPAs, and confirms the vulnerability of SSMs against such attacks. Even the recent Jamba-1.7-Mini SSM--Transformer (a 52B hybrid model) collapses on RoBench-25 under some HiSPA triggers, whereas pure Transformers do not. We also observe that HiSPA triggers significantly weaken the Jamba model on the popular Open-Prompt-Injections benchmark, unlike pure Transformers. We further show that the theo
    
[^416]: 面向问答任务的大语言模型多语言医学推理

    Multilingual Medical Reasoning for Question Answering with Large Language Models

    [https://arxiv.org/abs/2512.05658](https://arxiv.org/abs/2512.05658)

    该论文提出了一种基于维基百科医学知识、采用检索增强生成方法构建多语言（英语、意大利语、西班牙语）医学推理轨迹的技术，生成了50万条推理数据，并证明这些数据在少样本学习和监督微调两种方式下均能显著提升大语言模型在医学问答任务上的表现。

    

    具有推理能力的大语言模型（LLM）近来在医学问答（QA）任务中展现出了强大的潜力。现有方法大多以英语为中心，且主要依赖于从通用大语言模型进行知识蒸馏，这引发了人们对其医学知识可靠性的担忧。在本工作中，我们提出了一种基于从维基百科提取的医学知识来生成多语言推理轨迹的方法。我们采用检索增强生成（RAG）技术，基于维基百科中的医学信息，生成了50万条英语、意大利语和西班牙语的推理轨迹。这些轨迹用于解答来自MedQA和MedMCQA的医学问题，我们将这两个数据集扩展到了意大利语和西班牙语。我们在多个医学问答基准上进行了域内和域外设置的测试，结果表明，无论是通过上下文学习（少样本）还是监督微调的方式使用，我们的推理轨迹都能提升模型性能。

    arXiv:2512.05658v3 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) with reasoning capabilities have recently demonstrated strong potential in medical Question Answering (QA). Existing approaches are largely English-focused and primarily rely on distillation from general-purpose LLMs, raising concerns about the reliability of their medical knowledge. In this work, we present a method to generate multilingual reasoning traces based on medical knowledge extracted from Wikipedia. We produce 500k traces in English, Italian, and Spanish, using a retrieval-augmented generation approach over medical information from Wikipedia. The traces are generated to solve medical questions drawn from MedQA and MedMCQA, which we extend to Italian and Spanish. We test our pipeline in both in-domain and out-of-domain settings across Medical QA benchmarks, and demonstrate that our reasoning traces improve performance both when utilized via in-context learning (few-shot) and supervised fin
    
[^417]: 通过对应关系引导实现3D一致的多视图编辑

    3D-Consistent Multi-View Editing by Correspondence Guidance

    [https://arxiv.org/abs/2511.22228](https://arxiv.org/abs/2511.22228)

    提出了一种无需训练的引导框架，通过引入一致性损失确保对应点在编辑后保持相似，从而在去噪过程中实现几何和光度上3D一致的多视图图像编辑。

    

    扩散模型和流模型的最新进展极大地提升了基于文本的图像编辑效果，然而独立编辑各图像的方法往往会在同一场景的不同视图之间产生几何和光度上不一致的结果。这种不一致性对于编辑NeRF或高斯泼溅模型等3D表示而言尤为成问题。我们提出了一种无需训练的引导框架，可在图像编辑过程中强制实现多视图一致性。其核心思想是：对应的点在编辑后应当看起来相似。为实现这一目标，我们引入了一种一致性损失，引导去噪过程朝着连贯一致的编辑方向进行。该框架灵活多变，可以与各种不同的图像编辑方法相结合，同时支持密集和稀疏的多视图编辑设置。实验结果表明，与现有多视图编辑方法相比，我们的方法显著提升了3D一致性。

    arXiv:2511.22228v3 Announce Type: replace-cross  Abstract: Recent advancements in diffusion and flow models have greatly improved text-based image editing, yet methods that edit images independently often produce geometrically and photometrically inconsistent results across different views of the same scene. Such inconsistencies are particularly problematic for editing of 3D representations such as NeRFs or Gaussian splat models. We propose a training-free guidance framework that enforces multi-view consistency during the image editing process. The key idea is that corresponding points should look similar after editing. To achieve this, we introduce a consistency loss that guides the denoising process toward coherent edits. The framework is flexible and can be combined with widely varying image editing methods, supporting both dense and sparse multi-view editing setups. Experimental results show that our approach significantly improves 3D consistency compared to existing multi-view edi
    
[^418]: 神经簇的Alexander-Hirschowitz定理

    The Alexander-Hirschowitz theorem for neurovarieties

    [https://arxiv.org/abs/2511.19703](https://arxiv.org/abs/2511.19703)

    本文通过独立的几何方法证明了激活次数满足线性界 d_i ≥ 2n_i−1 时多项式神经网络的神经簇对任意输出数都是非亏缺的，并进一步证明了多输出架构在相同次数界下的全局可辨识性。

    

    我们研究与多项式神经网络相关的神经簇的维数与可辨识性。我们给出了一个独立的几何证明，表明激活次数上的线性界 $d_i\geq 2n_i-1$ 蕴含任意输出数目下的非亏缺性，这一维数结论此前是从有限可辨识性推导得到的。该证明基于对参数化微分的直接分析。我们还研究了该范围之外的割线与Grassmann割线障碍，并在相同的次数界下证明了多输出架构的全局可辨识性。

    arXiv:2511.19703v2 Announce Type: replace-cross  Abstract: We study the dimension and identifiability of neurovarieties associated to polynomial neural networks. We give an independent geometric proof that the linear bounds $d_i\geq 2n_i-1$ on the activation degrees imply non defectiveness for any number of outputs, a dimension statement previously obtained from finite identifiability. The proof is based on a direct analysis of the differential of the parameterization. We also investigate secant and Grassmann-secant obstructions outside this range and prove global identifiability for multi-output architectures under the same degree bounds.
    
[^419]: 一种用于惯性约束聚变图像去噪的机器学习驱动解决方案

    A Machine Learning-Driven Solution for Denoising Inertial Confinement Fusion Images

    [https://arxiv.org/abs/2511.16717](https://arxiv.org/abs/2511.16717)

    该研究提出了一种结合Cohen-Daubechies-Feauveau小波的无监督自编码器机器学习方法，用于惯性约束聚变中子图像去噪，能够在高斯和泊松噪声共存的情况下保护图像关键特征，从而提升迭代图像重建的保真度。

    

    中子成像对于在国家点火装置上诊断和优化惯性约束聚变内爆至关重要。然而，由于需要10微米的分辨率，中子图像必须使用迭代算法进行图像重建。对于低产额源，图像可能受到多种类型噪声的降质影响。高斯噪声和泊松噪声常常在同一图像中共存，掩盖了精细细节并模糊了编码源信息的边缘。传统的去噪技术，如滤波和阈值处理，可能会无意中改变关键特征或重塑噪声统计特性，从而可能影响迭代图像重建流程的最终保真度。然而，合成数据生成和机器学习的最新进展为应对这些挑战开辟了新的机遇。在本研究中，我们提出了一种结合Cohen-Daubechies-Feauveau小波的无监督自编码器

    arXiv:2511.16717v3 Announce Type: replace-cross  Abstract: Neutron imaging is essential for diagnosing and optimizing inertial confinement fusion implosions at the National Ignition Facility. Due to the required 10-micrometer resolution, however, neutron image require image reconstruction using iterative algorithms. For low-yield sources, the images may be degraded by various types of noise. Gaussian and Poisson noise often coexist within one image, obscuring fine details and blurring the edges where the source information is encoded. Traditional denoising techniques, such as filtering and thresholding, can inadvertently alter critical features or reshape the noise statistics, potentially impacting the ultimate fidelity of the iterative image reconstruction pipeline. However, recent advances in synthetic data production and machine learning have opened new opportunities to address these challenges. In this study, we present an unsupervised autoencoder with a Cohen-Daubechies- Feauveau 
    
[^420]: 多智能体大语言模型编排为事件响应实现确定性、高质量决策支持

    Multi-Agent LLM Orchestration Achieves Deterministic, High-Quality Decision Support for Incident Response

    [https://arxiv.org/abs/2511.15755](https://arxiv.org/abs/2511.15755)

    该论文提出MyAntFarm.ai框架，通过348次受控试验证明多智能体LLM编排相比单智能体方法可将可操作建议率从1.7%提升至100%，实现行动具体性提升80倍、方案正确性提升140倍且质量零方差的确定性事件响应决策支持。

    

    大语言模型（LLM）有望加速生产系统中的事件响应，但单智能体方法往往生成模糊、不可用的建议。我们提出了MyAntFarm.ai，这是一个可复现的容器化框架，证明了多智能体编排从根本上改变了基于LLM的事件响应质量。通过348次受控试验，在相同事件场景下比较单智能体副驾驶与多智能体系统，我们发现多智能体编排实现了100%的可操作建议率，而单智能体方法仅为1.7%，行动具体性提升80倍，解决方案正确性提升140倍。至关重要的是，多智能体系统在所有试验中表现出零质量方差，这使得生产环境SLA承诺成为可能，而这是不一致的单智能体输出无法实现的。两种架构实现了相似的理解延迟（约40秒），表明该架构……

    arXiv:2511.15755v3 Announce Type: replace  Abstract: Large language models (LLMs) promise to accelerate incident response in production systems, yet single-agent approaches generate vague, unusable recommendations. We present MyAntFarm.ai, a reproducible containerized framework demonstrating that multi-agent orchestration fundamentally transforms LLM-based incident response quality. Through 348 controlled trials comparing single-agent copilot versus multi-agent systems on identical incident scenarios, we find that multi-agent orchestration achieves 100% actionable recommendation rate versus 1.7% for single-agent approaches, an 80 times improvement in action specificity and 140 times improvement in solution correctness. Critically, multi-agent systems exhibit zero quality variance across all trials, enabling production SLA commitments impossible with inconsistent single-agent outputs. Both architectures achieve similar comprehension latency (approx.40s), establishing that the architectu
    
[^421]: SEBA：面向视觉强化学习的样本高效黑盒攻击

    SEBA: Sample-Efficient Black-Box Attacks on Visual Reinforcement Learning

    [https://arxiv.org/abs/2511.09681](https://arxiv.org/abs/2511.09681)

    SEBA提出了一种针对视觉强化学习的样本高效黑盒攻击框架，通过结合影子Q模型、生成对抗网络和世界模型，以极少的真实环境查询实现对基于图像的连续控制智能体的有效对抗攻击。

    

    视觉强化学习在视觉控制和机器人领域取得了显著进展，但其对对抗性扰动的脆弱性仍未得到充分探索。现有的大多数黑盒攻击集中于基于向量或离散动作的强化学习，其在基于图像的连续控制上的有效性受限于庞大的动作空间和过多的环境查询。我们提出了SEBA，一个针对视觉强化学习智能体的样本高效黑盒对抗攻击框架。SEBA集成了三个组件：一个用于估计对抗条件下累积奖励的影子Q模型、一个生成视觉上不可察觉扰动的生成对抗网络，以及一个模拟环境动态以减少真实环境查询的世界模型。通过在学习影子模型与优化生成器之间交替进行的两阶段迭代训练过程，SEBA在保持高效的同时实现了强大的攻击性能。

    arXiv:2511.09681v2 Announce Type: replace-cross  Abstract: Visual reinforcement learning has achieved remarkable progress in visual control and robotics, but its vulnerability to adversarial perturbations remains underexplored. Most existing black-box attacks focus on vector-based or discrete-action RL, and their effectiveness on image-based continuous control is limited by the large action space and excessive environment queries. We propose SEBA, a sample-efficient framework for black-box adversarial attacks on visual RL agents. SEBA integrates a shadow Q model that estimates cumulative rewards under adversarial conditions, a generative adversarial network that produces visually imperceptible perturbations, and a world model that simulates environment dynamics to reduce real-world queries. Through a two-stage iterative training procedure that alternates between learning the shadow model and refining the generator, SEBA achieves strong attack performance while maintaining efficiency. E
    
[^422]: 个性化算法建议作为竞争市场中的战略信号

    Individualized Algorithmic Advice as a Strategic Signal on Competitive Markets

    [https://arxiv.org/abs/2511.09454](https://arxiv.org/abs/2511.09454)

    古诺竞争实验表明，与均衡一致的个性化算法建议能促进稳定收敛，而串谋性向下偏倚的建议会诱发默示串谋行为（产量不足与超竞争利润），且个性化建议比集体建议更容易被参与者采纳。

    

    随着算法日益成为竞争性决策的媒介，其影响已超越个体结果本身，延伸至塑造战略性的市场动态。在我们的实验中，我们检验了算法建议如何在一个具有唯一、非串谋且可解析推导均衡的经典经济学博弈中影响人类行为。129名参与者参与了一场古诺数量竞争博弈，并分别接受与均衡一致或带有战略偏倚的算法建议。个体化的均衡建议支持了稳定的收敛，而带有串谋倾向的向下偏倚建议则导致了持续的产量不足和超竞争利润——这些正是默示串谋的标志性特征。相比集体性的均衡建议，参与者的产量向个体化均衡建议收敛得更快、更一致，这可能源于前者客观质量上的优势，或参与者对个体化建议更强的感知拥有感。这些发现表明，算法建议可以发挥……（原文此处截断）

    arXiv:2511.09454v2 Announce Type: replace-cross  Abstract: As algorithms increasingly mediate competitive decision-making, their influence extends beyond individual outcomes to shaping strategic market dynamics. In our experiment, we examined how algorithmic advice affects human behavior in a classic economic game with a unique, non-collusive, and analytically traceable equilibrium. Participants (N = 129) played a Cournot quantity competition with equilibrium-aligned or strategically biased algorithmic recommendations. While individualized equilibrium advice supported stable convergence, collusively downward-biased advice led to sustained underproduction and supracompetitive profits - hallmarks of tacit collusion. Participants' quantities converged faster and more consistently toward individualized than collective equilibrium advice, potentially due to an objective quality advantage or greater perceived ownership of the former. These findings demonstrate that algorithmic advice can fun
    
[^423]: KGFR：面向广义知识图谱问答的基础检索器

    KGFR: A Foundation Retriever for Generalized Knowledge Graph Question Answering

    [https://arxiv.org/abs/2511.04093](https://arxiv.org/abs/2511.04093)

    提出LLM-KGFR协作框架，通过LLM生成的关系描述、基于实体角色的初始化实现零样本泛化，并借助非对称渐进传播高效处理大规模图谱，从而支持对未见知识图谱的通用问答。

    

    大型语言模型（LLM）擅长推理，但由于上下文和参数化知识的限制，在知识密集型问题上表现不佳。然而，现有依赖微调LLM或图神经网络检索器的方法受限于数据集特定的调优以及在大规模或未见图谱上的可扩展性。我们提出了LLM-KGFR协作框架，其中LLM与一个结构化检索器——知识图谱基础检索器（KGFR）协同工作。KGFR利用LLM生成的描述对关系进行编码，并根据实体在问题中的角色对其进行初始化，从而实现对未见知识图谱的零样本泛化。为了高效处理大规模图谱，它采用了非对称渐进传播（APP）——一种分步扩展策略，选择性地限制高度数节点，同时保留信息丰富的路径。通过节点级、边级和路径级接口，LLM迭代地请求候选答案、支持事实和推理路径，形成一种协作式的问答流程。

    arXiv:2511.04093v2 Announce Type: replace  Abstract: Large language models (LLMs) excel at reasoning but struggle with knowledge-intensive questions due to limited context and parametric knowledge. However, existing methods that rely on finetuned LLMs or GNN retrievers are limited by dataset-specific tuning and scalability on large or unseen graphs. We propose the LLM-KGFR collaborative framework, where an LLM works with a structured retriever, the Knowledge Graph Foundation Retriever (KGFR). KGFR encodes relations using LLM-generated descriptions and initializes entities based on their roles in the question, enabling zero-shot generalization to unseen KGs. To handle large graphs efficiently, it employs Asymmetric Progressive Propagation (APP)- a stepwise expansion that selectively limits high-degree nodes while retaining informative paths. Through node-, edge-, and path-level interfaces, the LLM iteratively requests candidate answers, supporting facts, and reasoning paths, forming a c
    
[^424]: 基于秩-2子空间解缠的多步知识交互分析

    Multi-Step Knowledge Interaction Analysis via Rank-2 Subspace Disentanglement

    [https://arxiv.org/abs/2511.01706](https://arxiv.org/abs/2511.01706)

    该论文提出一种新颖的秩-2投影子空间来更准确地解缠大语言模型中参数化知识与情境知识的贡献，并首次实现了对自然语言解释更长生成序列中知识交互的多步分析。

    

    自然语言解释（NLEs）通过借助外部情境知识（CK）和参数化知识（PK）来描述大语言模型（LLMs）如何做出决策。理解这些知识来源之间的交互是评估NLE接地性的关键，然而这些动态机制仍未得到充分探索。先前的工作主要集中于：i）单步生成，以及ii）将PK与CK的交互建模为秩-1子空间内的二元选择。这种方法忽略了更丰富的交互形式，以及这些交互在更长生成过程中的演变方式，例如互补性或支持性知识。我们提出了一种新颖的秩-2投影子空间，能够更准确地解缠PK和CK的贡献，并首次将其用于对更长NLE序列中知识交互的多步分析。在四个问答数据集和三个开源权重LLM上的实验表明，秩-1子空间难以表示多样化的知识交互，而我们的秩-2方法能够更好地捕捉这些丰富的交互形式。

    arXiv:2511.01706v3 Announce Type: replace-cross  Abstract: Natural Language Explanations (NLEs) describe how Large Language Models (LLMs) make decisions by drawing on external Context Knowledge (CK) and Parametric Knowledge (PK). Understanding the interaction between these sources is key to assessing NLE grounding, yet these dynamics remain underexplored. Prior work has largely focused on i) single-step generation and ii) modeled PK--CK interaction as a binary choice within a rank-1 subspace. This approach overlooks richer interactions and how they unfold over longer generations, such as complementary or supportive knowledge. We propose a novel rank-2 projection subspace that disentangles PK and CK contributions more accurately and use it for the first multi-step analysis of knowledge interactions across longer NLE sequences. Experiments across four QA datasets and three open-weight LLMs demonstrate that rank-1 subspaces struggle to represent diverse interactions, whereas our rank-2 fo
    
[^425]: 机器能高效地思考吗？

    Can machines think efficiently?

    [https://arxiv.org/abs/2510.26954](https://arxiv.org/abs/2510.26954)

    该论文提出在图灵测试中引入能量消耗约束，从效率视角重新评估机器智能，从而为智能评估提供了原测试所缺乏的可测量的实际标准。

    

    图灵测试已不再足以区分人类智能与机器智能。随着先进的人工智能系统已经通过原始的图灵测试，并引发了严重的伦理和环境问题，我们迫切需要更新这一测试。这项工作在原始的模仿游戏基础上进行了扩展，纳入了一个额外因素：回答问题所消耗的能量。通过增加能量约束，新测试迫使我们从效率的角度来评估智能，将思考这一抽象问题与有限资源的具体现实联系起来。此外，这一新提出的测试确保智能评估拥有一个可测量、可实际操作的终点线，而这是原始测试所缺乏的。这一额外约束促使社会权衡使用人工智能所节省的时间与其总体资源成本。

    arXiv:2510.26954v3 Announce Type: replace-cross  Abstract: The Turing Test is no longer adequate for distinguishing human and machine intelligence. With advanced artificial intelligence systems already passing the original Turing Test and contributing to serious ethical and environmental concerns, we urgently need to update the test. This work expands upon the original imitation game by accounting for an additional factor: the energy spent answering the questions. By adding the constraint of energy, the new test forces us to evaluate intelligence through the lens of efficiency, connecting the abstract problem of thinking to the concrete reality of finite resources. Further, this proposed new test ensures the evaluation of intelligence has a measurable, practical finish line that the original test lacks. This additional constraint compels society to weigh the time savings of using artificial intelligence against its total resource cost.
    
[^426]: 驯服持续音视频分割中的模态纠缠

    Taming Modality Entanglement in Continual Audio-Visual Segmentation

    [https://arxiv.org/abs/2510.17234](https://arxiv.org/abs/2510.17234)

    该论文提出持续音视频分割（CAVS）这一新任务，识别出多模态语义漂移与共现混淆两大关键挑战，并设计了基于碰撞的多模态重放（CMR）框架，以解决细粒度多模态持续学习中的模态纠缠问题。

    

    近年来，多模态持续学习取得了显著进展，其目标是在多模态环境下按顺序学习新任务，同时保持对之前已学习任务的良好性能。然而，现有方法主要集中于粗粒度任务，在应对细粒度持续学习环境中的模态纠缠问题上存在局限。为弥合这一差距，我们提出了一种新颖的持续音视频分割（CAVS）任务，旨在以音频为引导持续分割新的类别。通过全面分析，我们识别出两个关键挑战：1）多模态语义漂移，即在顺序任务中发声物体被错误地标注为背景；2）共现混淆，即频繁共现的类别容易被混淆。在本工作中，我们设计了基于碰撞的多模态重放（CMR）框架来应对这些挑战。具体而言，针对多模态语义漂移，一种多模态……（摘要在此处截断）

    arXiv:2510.17234v3 Announce Type: replace-cross  Abstract: Recently, significant progress has been made in multi-modal continual learning, aiming to learn new tasks sequentially in multi-modal settings while preserving performance on previously learned ones. However, existing methods mainly focus on coarse-grained tasks, with limitations in addressing modality entanglement in fine-grained continual learning settings. To bridge this gap, we introduce a novel Continual Audio-Visual Segmentation (CAVS) task, aiming to continuously segment new classes guided by audio. Through comprehensive analysis, two critical challenges are identified: 1) multi-modal semantic drift, where a sounding objects is labeled as background in sequential tasks; 2) co-occurrence confusion, where frequent co-occurring classes tend to be confused. In this work, a Collision-based Multi-modal Rehearsal (CMR) framework is designed to address these challenges. Specifically, for multi-modal semantic drift, a Multi-modal
    
[^427]: HugAgent：一个面向个体层面推理的人类模拟基准

    HugAgent: A Human Simulation Benchmark for Individual-Level Reasoning

    [https://arxiv.org/abs/2510.15144](https://arxiv.org/abs/2510.15144)

    该论文提出HugAgent基准，从个性化推理、认知对齐和开放式数据三个维度重新定义人类推理模拟，评估模型能否基于某人历史观点的部分证据，预测该特定个体在分布外场景中的行为反应与推理动态。

    

    在开放式任务中模拟人类推理长期以来一直是人工智能和认知科学的核心追求。尽管大型语言模型现在能够大规模地近似人类反应，但它们仍是针对群体层面共识进行调优的，往往会抹杀推理风格和信念轨迹的个体性。为了推进机器实现更类人推理的愿景，我们提出了HugAgent（HUman-Grounded AGENT Benchmark，基于人类的智能体基准），从三个维度重新思考人类推理模拟：从平均化推理转向个性化推理；(ii) 从行为模仿转向认知对齐；(iii) 从基于情景片段的数据转向开放式数据。该基准评估的是：在给定某人先前观点的部分证据的情况下，模型能否预测该特定个体在分布外场景中的行为反应及其背后的推理动态。HugAgent将结构化问卷与半结构化的出声思考访谈相结合来收集……（原文截断）

    arXiv:2510.15144v4 Announce Type: replace  Abstract: Simulating human reasoning in open-ended tasks has long been a central aspiration in AI and cognitive science. While large language models now approximate human responses at scale, they remain tuned to population-level consensus, often erasing the individuality of reasoning styles and belief trajectories. To advance the vision of more human-like reasoning in machines, we introduce HugAgent (HUman-Grounded AGENT Benchmark), which rethinks human reasoning simulation along three dimensions: (i) from averaged to individualized reasoning, (ii) from behavioral mimicry to cognitive alignment, and (iii) from vignette-based to open-ended data. The benchmark evaluates whether a model can predict a specific person's behavioral responses and the underlying reasoning dynamics in out-of-distribution scenarios, given partial evidence of their prior views. HugAgent combines structured questionnaires with semi-structured think-aloud interviews to col
    
[^428]: 将组合式机器设计视为基于大语言模型的程序合成

    Compositional Machine Design as Program Synthesis with LLMs

    [https://arxiv.org/abs/2510.14980](https://arxiv.org/abs/2510.14980)

    该论文提出将机器设计视为一种以物理模拟验证为依据的程序合成新任务——组合式机器设计，并构建了基于游戏《Besiege》的测试平台BesiegeField，用于评测大语言模型在多种工作流下组合标准部件设计机器的能力。

    

    大语言模型（LLM）在编写和修改程序方面已展现出强大的能力，然而许多程序合成基准仍在符号或数字环境中评估程序。我们提出了“组合式机器设计”，这是一种以物理为基础的程序合成形式：机器被编写为组合标准化部件的程序，其成败由模拟的物理行为决定。为研究这一问题，我们提出了BesiegeField，一个基于机器建造游戏《Besiege》构建的测试平台。在BesiegeField中，LLM智能体根据文本形式的功能需求生成机器程序，在模拟中运行所得的机器，并接收奖励与状态反馈。我们在单智能体生成、迭代编辑和分层工作流等模式下，对LLM智能体在代表性机器设计任务上进行了基准评测。强大的模型能够恢复与任务相关的结构，有时还能取得不俗的物理性能表现，但常常……（原文摘要在此处截断）

    arXiv:2510.14980v3 Announce Type: replace  Abstract: Large language models (LLMs) have shown strong abilities in writing and revising programs, yet many program-synthesis benchmarks still evaluate programs in symbolic or digital environments. We introduce compositional machine design, a physically grounded form of program synthesis where machines are written as programs that compose standardized parts, and success is determined by simulated physical behavior. To study this problem, we present BesiegeField, a testbed built on the machine-building game Besiege. In BesiegeField, LLM agents generate machine programs from textual functional demands, execute the resulting machines in simulation, and receive rewards and state feedback. We benchmark LLM agents across representative machine-design tasks under single-agent generation, iterative editing, and hierarchical workflows. Strong models recover task-relevant structures and sometimes achieve nontrivial physical performance, but often stru
    
[^429]: 面向作者归属与验证的单样本风格迁移LLM对数概率方法

    One-shot Style Transfer LLM log-probabilities for Authorship Attribution and Verification

    [https://arxiv.org/abs/2510.13302](https://arxiv.org/abs/2510.13302)

    本文提出一种无监督框架，利用大语言模型的对数概率衡量文本间的风格可迁移性，无需显式监督即可在作者验证任务上显著超越基于提示的无监督基线，并在足够模型规模下与对比学习基线相当或更优。

    

    计算文体学通过定量的文本模式研究写作风格，能够支持作者归属、身份关联和抄袭检测等应用。尽管语言建模与这些任务密切相关，但现代大语言模型（LLM）的预训练在作者归属与验证领域尚未得到充分利用。我们提出了一个无监督框架，利用大语言模型的对数概率来衡量两个文本之间的风格可迁移性。该框架充分利用了大语言模型大规模的自回归语言建模（CLM）预训练、单样本能力和模型规模，避免了显式监督。在相近模型规模下，我们的方法在作者验证任务上显著优于基于提示的无监督基线；在模型规模足够的情况下，该方法在大多数设置中与对比学习基线相当或有所提升。此外，我们还观察到在……方面的强劲表现（注：原文摘要在此处被截断）。

    arXiv:2510.13302v4 Announce Type: replace-cross  Abstract: Computational stylometry studies writing style through quantitative textual patterns, enabling applications such as authorship attribution, identity linking, and plagiarism detection. Despite the relevance of language modeling to these tasks, the pre-training of modern large language models (LLMs) has been underutilized in authorship attribution and verification. We introduce an unsupervised framework that uses the log-probabilities of an LLM to measure style transferability between two texts. This framework takes advantage of the extensive Causal Language Modeling (CLM) pre-training, one-shot capabilities and scale of LLMs, avoiding explicit supervision. Our methods substantially outperform prompting-based unsupervised baselines in authorship verification at similar model sizes, and is competitive with or improves contrastive baselines in most settings with sufficient model scale. We further observe strong performance across n
    
[^430]: TopoAlign：通过拓扑分解将代码对齐到数学的框架

    TopoAlign: A Framework for Aligning Code to Math via Topological Decomposition

    [https://arxiv.org/abs/2510.11944](https://arxiv.org/abs/2510.11944)

    TopoAlign通过将代码分解为文档字符串、主函数和依赖函数并重新组装对齐，弥合了代码与形式化数学之间的结构与句法差异，从而将大规模代码仓库转化为可用于提升数学LLM自动形式化能力的训练资源。

    

    大型语言模型（LLMs）在非形式化和形式化（例如Lean 4）数学推理方面都表现出色，但在自动形式化任务上仍然存在困难，即如何将非形式化的数学陈述转换为形式化数学陈述。然而，当前数学LLMs的性能受到大规模语料库稀缺的制约，尤其是缺少包含非形式化与形式化陈述配对的数据集。有趣的是，用于自动形式化的形式化语言与编程语言在结构上具有相似性，且代码数据可以大规模获取。然而，目前在代码上训练的模型并不能有效地迁移到形式化数学任务中，原因在于两者之间存在结构和句法上的差异。为了解决这一问题，我们提出了TopoAlign，一个能够将广泛可用的代码仓库转化为数学LLMs训练资源的框架。TopoAlign将代码分解为文档字符串、主函数和依赖函数，并将这些组件重新组装成（摘要在此处截断）

    arXiv:2510.11944v2 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) excel at both informal and formal (e.g. Lean 4) mathematical reasoning but still struggle with autoformalisation, the task of transforming informal into formal mathematical statements. Yet, the performance of current Math LLMs is constrained by the scarcity of large-scale corpora, particularly those containing pairs of informal and formal statements. Interestingly, the formal languages used in autoformalisation share structural similarities with programming languages, and code data is available at scale. However, current models trained on code do not transfer effectively to formal math, due to structural and syntactic differences between them. To address this, we propose TopoAlign, a framework that unlocks widely available code repositories as training resources for Math LLMs. TopoAlign decomposes code into docstrings, main functions, and dependency functions, and reassembles these components into a
    
[^431]: 揭示大语言模型中类人表示的计算要素

    Uncovering the Computational Ingredients of Human-Like Representations in LLMs

    [https://arxiv.org/abs/2510.01030](https://arxiv.org/abs/2510.01030)

    本研究借助认知科学中的三元组相似度任务，对75个以上大语言模型进行了系统评估，以识别影响类人概念表示形成的关键计算要素（如指令微调），同时弥补了现有基准无法衡量模型与人类表示对齐程度的不足。

    

    人类将多样的感知与语言输入转化为结构化行为的能力，一直被认为依赖于对概念的稳健表示的学习。基于Transformer的大语言模型（LLM）的快速发展，显现出多种与模型构建相关的计算要素——包括架构、微调方法和训练数据集等——然而尚不清楚其中哪些要素对于形成类人的概念表示最为关键。此外，目前大多数基准测试并不适合衡量表示层面的对齐程度，使得LLM在这些基准上的得分无法可靠地评估其作为认知模型的发展进程。为解决这些局限，我们在三元组相似度任务上评估了超过75个模型，该任务是认知科学中测量概念表示的经典方法，实验使用了来自THINGS数据库的概念。我们发现，指令微调……

    arXiv:2510.01030v2 Announce Type: replace  Abstract: The human ability to translate diverse perceptual and linguistic inputs into structured behavior has been thought to rest on learning robust representations of concepts. The rapid advancement of transformer-based large language models (LLMs) has surfaced a diversity of computational ingredients relevant for model building - architectures, fine-tuning methods, and training datasets among others - yet it remains unclear which are most crucial for developing human-like conceptual representations. Further, most current benchmarks are ill-suited to measuring representational alignment, making LLMs' scores on them unreliable for assessing whether they are progressing as cognitive models. We address these limitations by evaluating over 75 models on a triplet similarity task, a method well established in cognitive science for measuring conceptual representations, using concepts from the THINGS database. We find that instruction fine-tuning a
    
[^432]: SupraTok：跨边界分词技术助力语言模型性能提升

    SupraTok: Cross-Boundary Tokenization for Enhanced Language Model Performance

    [https://arxiv.org/abs/2508.11857](https://arxiv.org/abs/2508.11857)

    SupraTok是一种跨越空白符边界的创新分词器，通过熵筛选、PMI引导的课程训练和多语言处理三大模块，在压缩率上比标准BPE提升17.5%，并比SuperBPE训练快2.1倍。

    

    分词一直是语言建模中持续存在的瓶颈，尤其是当词汇学习受到空白符边界的限制时。我们提出了SupraTok，这是一种能够跨越空白符边界的分词器，它由三个模块化组件构成：可选的基于熵的数据筛选、采用PMI引导候选搜索的分阶段课程训练，以及多语言文字处理。在相同的未过滤训练数据上，使用10万词汇量时，SupraTok的压缩效果比标准BPE提升17.5%，比官方SuperBPE实现提升1.8%，同时训练速度比SuperBPE快2.1倍。在相同匹配设置下，在5万至30万词汇量范围内，SupraTok始终比SuperBPE领先1.8%至8.6%。我们还将熵过滤作为流水线中的一个独立步骤单独评估：在10万词汇量下，它使SupraTok的C/T指标从5.78提升至5.99，而匹配对照组显示SuperBPE的增益较小，SP-BPE-CrossBoundary则几乎无变化。在FLORES-200的14种语言上的（摘要原文在此处截断）

    arXiv:2508.11857v3 Announce Type: replace-cross  Abstract: Tokenization remains a persistent bottleneck in language modeling, especially when vocabulary learning is limited by whitespace boundaries. We present SupraTok, a tokenizer that crosses whitespace boundaries using three modular components: optional entropy-based data curation, staged curriculum training with PMI-guided candidate search, and multilingual script handling. At 100k vocabulary on the same unfiltered training data, SupraTok improves compression over standard BPE by 17.5% and over the official SuperBPE implementation by 1.8%, while training 2.1x faster than SuperBPE. Across 50k-300k vocabularies in the same matched setting, SupraTok remains ahead of SuperBPE by 1.8%-8.6%. We evaluate entropy filtering separately as a pipeline step: at 100k vocabulary it raises SupraTok from 5.78 to 5.99 C/T, while matched controls show a smaller gain for SuperBPE and almost no change for SP-BPE-CrossBoundary. On FLORES-200 across 14 l
    
[^433]: BiasGym：一个通过注入来分析和消除偏见的简单且可泛化的框架

    BiasGym: A Simple and Generalizable Framework for Analyzing and Removing Biases through Injection

    [https://arxiv.org/abs/2508.08855](https://arxiv.org/abs/2508.08855)

    提出BiasGym框架，通过在冻结的LLM中注入特定偏见信号，再利用这些信号定位并抑制或引导导致偏见行为的模型组件，实现偏见的可靠分析与消除。

    

    理解大型语言模型（LLM）权重中编码的偏见和刻板印象，对于制定有效的缓解策略至关重要。然而，偏见行为往往是微妙且难以隔离的，即使刻意去引发也是如此，这使得系统性的分析和去偏工作尤其具有挑战性。为了解决这一问题，我们提出了一个简单、低成本且可泛化的框架 BiasGym，用于可靠地注入、分析和缓解 LLM 中偏见的概念关联。BiasGym 包含两个模块：Inject，通过基于 token 的微调（同时保持模型冻结）向模型注入特定偏见；以及两种去偏方法，利用这些注入信号来识别并可靠地抑制（Scope）或引导（Steer）导致偏见行为的模型组件。我们的框架能够实现一致的偏见引发，从而更好地定位…

    arXiv:2508.08855v5 Announce Type: replace-cross  Abstract: Understanding biases and stereotypes encoded in the weights of Large Language Models (LLMs) is crucial for developing effective mitigation strategies. However, biased behavior is often subtle and non-trivial to isolate, even when deliberately elicited, making systematic analysis and debiasing particularly challenging. To address this, we introduce a simple, cost-effective, and generalizable framework \texttt{BiasGym} for reliably injecting, analyzing, and mitigating conceptual associations of biases within LLMs. \texttt{BiasGym} consists of two modules: \texttt{Inject}, which injects specific biases into the model via token-based fine-tuning while keeping the model frozen, followed by two debiasing methods that leverage these injected signals to identify and reliably suppress (\texttt{Scope}) or \texttt{Steer} the components responsible for biased behavior. Our framework enables consistent bias elicitation for better localizati
    
[^434]: 无监督伙伴设计实现鲁棒的临时团队协作

    Unsupervised Partner Design Enables Robust Ad-hoc Teamwork

    [https://arxiv.org/abs/2508.06336](https://arxiv.org/abs/2508.06336)

    提出无监督伙伴设计（UPD），通过即时生成训练伙伴并基于可学习性准则自适应选择，无需预训练伙伴种群或手动调参即可实现鲁棒的临时团队协作，在多个基准任务和人机用户研究中均表现出卓越性能。

    

    我们提出了无监督伙伴设计（UPD），这是一种面向鲁棒临时团队协作的无种群多智能体强化学习方法。UPD即时生成训练伙伴，并基于可学习性准则自适应地选择它们，从而无需预训练的伙伴种群或手动参数调整。我们证明这一简单机制能够实现有效的伙伴多样性，并且当程序化关卡生成器可用时，还可以扩展到伙伴与环境的联合选择。在Level-Based Foraging、Overcooked-AI和Overcooked泛化挑战等任务中，与基于种群和无种群的基线方法相比，UPD始终取得了强劲的性能表现。在人机交互用户研究中，使用UPD训练的智能体获得了更高的回报，并且与所有被评估的基线方法相比，被评为更具适应性、更像人类，且令人沮丧程度更低。

    arXiv:2508.06336v3 Announce Type: replace-cross  Abstract: We introduce Unsupervised Partner Design (UPD), a population-free multi-agent reinforcement learning method for robust ad-hoc teamwork. UPD generates training partners on-the-fly and selects them adaptively based on a learnability criterion, removing the need for pre-trained partner populations or manual parameter tuning. We show that this simple mechanism enables effective partner diversity and can be extended to joint partner-environment selection when a procedural level generator is available. Across Level-Based Foraging, Overcooked-AI, and the Overcooked Generalisation Challenge, UPD consistently achieves strong performance compared to both population-based and population-free baselines. In a human-AI user study, agents trained with UPD achieve higher returns and are rated as more adaptive, more human-like, and less frustrating than all evaluated baseline methods.
    
[^435]: GeoGR^2：基于地统计引导与大语言模型迭代精炼的零样本地理空间推断

    GeoGR^2:Zero-Shot Geospatial Inference via Geostatistically-Guided Iterative Refinement with LLMs

    [https://arxiv.org/abs/2508.04080](https://arxiv.org/abs/2508.04080)

    提出GeoGR^2框架，将零样本地理空间预测建模为动态构建图上的迭代消息传递过程，通过拓扑、特征和更新三个动态算子引导大语言模型实现空间一致、消除人口稠密偏差的地理空间推断。

    

    标准的大语言模型提示方法将地理空间推断视为独立的、逐实例的预测，忽略了支配地理现实的基本空间依赖关系。因此，即使是先进的模型也难以保持空间一致性，并对人口稠密地区表现出严重偏差。为了弥合这一差距，我们提出了GeoGR^2（Geospatial Graph Refine Reasoning，地理空间图精炼推理），一个将零样本地理空间预测形式化为动态构建图上迭代消息传递过程的框架。与静态检索方法不同，GeoGR^2通过协作算子实例化了三个动态算子：（1）拓扑算子，构建图拓扑以强制执行空间马尔可夫性质；（2）特征算子，用与任务相关的语义协变量丰富节点；（3）更新算子，执行自然语言消息传递以迭代地最小化空间差异。在理论层面，我们构建了……（摘要原文至此截断）

    arXiv:2508.04080v2 Announce Type: replace  Abstract: Standard large language model prompting treats geospatial inference as independent, instance-wise prediction, ignoring the fundamental spatial dependencies that govern geographic reality. Consequently, even advanced models struggle with spatial consistency and exhibit severe biases toward populous regions. To bridge this gap, we propose GeoGR^2 (Geospatial Graph Refine Reasoning), a framework that formalizes zero-shot geospatial prediction as an iterative message-passing process on a dynamically constructed graph. Unlike static retrieval methods, GeoGR^2 instantiates three dynamic operators via collaborating operators: (1) a Topology Operator that constructs graph topology to enforce the Spatial Markov property; (2) a Feature Operator that enriches nodes with task-relevant semantic covariates; and (3) an Update Operator that performs natural language message passing to iteratively minimize spatial discrepancy. Theoretically, we frame
    
[^436]: ParaStudent：缩小AI导师评估中用户模拟器的模拟到现实差距

    ParaStudent: Closing the Sim2Real Gap in User Simulators for AI Tutor Evaluation

    [https://arxiv.org/abs/2507.12674](https://arxiv.org/abs/2507.12674)

    ParaStudent是一个通过微调来模拟初学者编程修改的框架，其模拟结果更贴近真实学生代码分布，可用于AI导师部署前的反馈评估与筛选。

    

    在部署前评估人工智能（AI）导师的反馈需要预测学生的参与度，这通常通过真实交互数据来评估。我们提出了ParaStudent，一个用于模拟初学者编程修改的微调框架，以支持AI导师评估。与基于提示的基线方法相比，ParaStudent的模拟修改在功能性、风格性和语义性指标上都更接近真实学生的代码分布。我们表现最佳的变体在区分真实参与度高于中位数与等于或低于中位数的交互流时，在反馈相关性和成功采纳两方面均达到0.80的AUC，而基于提示的基线在成功采纳方面的表现仍接近随机水平。这些发现展示了模拟参与度在AI导师部署前反馈筛选中的应用前景。

    arXiv:2507.12674v3 Announce Type: replace-cross  Abstract: Evaluating Artificial Intelligence (AI) tutor feedback before deployment requires anticipating student engagement, typically assessed through real interaction data. We introduce ParaStudent, a fine-tuning framework for simulating novice programming revisions to support AI tutor evaluation. Compared with prompted baselines, ParaStudent's revisions more closely match real student code distributions across functional, stylistic, and semantic metrics. Our best variant achieves AUCs of 0.80 for both feedback relevance and successful uptake when distinguishing streams with real engagement above versus at or below the median, while prompted baselines remain near chance on successful uptake. These findings demonstrate the promise of simulated engagement for pre-deployment feedback triage.
    
[^437]: 迈向可证明且可扩展的量化神经网络训练：基于伊辛优化方法

    Towards Provable and Scalable Training of Quantized Neural Networks with Ising Optimization

    [https://arxiv.org/abs/2506.18240](https://arxiv.org/abs/2506.18240)

    该论文提出一个具有可证明保证的精确二次约束二元优化（QCBO）框架，将量化神经网络训练编译为具有零松弛间隙的完全正凸优化问题，并通过逐样本分解下界优化（DLBO）将伊辛求解规模从数据集级别降至单样本级别，从而实现对量化神经网络可证明且可扩展的训练。

    

    由于非凸的损失景观和离散的参数空间，训练量化神经网络仍然面临根本性的挑战。我们引入了一个具有可证明保证的精确二次约束二元优化（QCBO）框架。我们首先刻画了网络零损失水平集的分层拓扑结构：一般的内部层是光滑的，但即使在过参数化条件下，全局最优的连通分量仍可能保持不连通。为了克服这一非凸障碍，我们将具有参数码本和前向区间传播（FIP）有界状态的有限深度架构编译为有界的QCBO，得到一种精确的完全正凸表述，该表述在零松弛间隙下保留了全局离散最优解。为了克服整体样本规模扩展的瓶颈，我们提出了逐样本的分解下界优化（DLBO）方法，将每次伊辛求解调用的规模从整个数据集降低到单样本级别。DLBO矩层级结构……（摘要在此处截断）

    arXiv:2506.18240v5 Announce Type: replace-cross  Abstract: Training quantized neural networks remains fundamentally challenging due to non-convex loss landscapes and discrete parameter spaces. We introduce an exact Quadratic Constrained Binary Optimization (QCBO) framework with provable guarantees. We first characterize the stratified topology of network zero-loss level sets: generic interior strata are smooth, yet globally optimal components can remain disconnected even under overparameterization. To address this non-convex obstruction, we compile finite-depth architectures with parameter codebooks and Forward Interval Propagation (FIP)-bounded states into bounded QCBOs, yielding an exact completely positive convex formulation that preserves the global discrete optimum with zero relaxation gap. To overcome monolithic sample scaling, we formulate sample-wise Decomposed Lower-Bound Optimization (DLBO) to reduce each Ising call from dataset to single-sample scale. The DLBO moment hierarc
    
[^438]: ViPlan：基于符号谓词与视觉语言模型的视觉规划基准

    ViPlan: A Benchmark for Visual Planning with Symbolic Predicates and Vision-Language Models

    [https://arxiv.org/abs/2505.13180](https://arxiv.org/abs/2505.13180)

    提出首个开源视觉规划基准ViPlan，用于比较VLM接地的符号规划方法与直接VLM规划方法，发现VLM作为接地器在Blocksworld中显著优于直接VLM规划（46%对9%）。

    

    将大型语言模型与符号规划器相结合是获得可验证且有依据的计划的一个有前景的方向，近期的工作利用视觉语言模型（VLM）将这一思想扩展到了视觉领域。然而，由于缺乏支持符号规划的视觉基准，目前尚无一个能在相同条件下比较这些方法的开源基准。我们提出了ViPlan，这是第一个用于比较VLM接地的符号方法（VLM作为接地器，VLM-as-grounder）与直接VLM规划方法（VLM作为规划器，VLM-as-planner）的开源基准。ViPlan在两个视觉领域（经典Blocksworld规划问题的视觉变体和模拟家庭机器人环境）中引入了一系列难度递增的任务。跨方法平均来看，我们发现VLM作为接地器的方法在Blocksworld中优于直接VLM规划（解决了46%的任务，而直接规划仅为9%），在该环境中图像接地既关键又准确。然而，

    arXiv:2505.13180v3 Announce Type: replace  Abstract: Integrating Large Language Models with symbolic planners is a promising direction for obtaining verifiable and grounded plans, with recent works extending this idea to visual domains using Vision-Language Models (VLMs). However, an open-source benchmark for comparing these approaches under matched conditions is missing, due to a lack of visual benchmarks that support symbolic planning. We present ViPlan, the first open-source benchmark for comparing VLM-grounded symbolic approaches (VLM-as-grounder) with direct VLM planning methods (VLM-as-planner). ViPlan introduces a series of increasingly challenging tasks in two visual domains: a visual variant of the classic Blocksworld planning problem and a simulated household robotics environment. Averaged across methods, we find VLM-as-grounders to outperform direct VLM planning in Blocksworld (solving 46% of the tasks against 9%), where image grounding is both crucial and accurate. However,
    
[^439]: 一个Token价值超过1000个Token：通过低秩克隆实现高效知识蒸馏

    A Token is Worth over 1,000 Tokens: Efficient Knowledge Distillation through Low-Rank Clone

    [https://arxiv.org/abs/2505.12781](https://arxiv.org/abs/2505.12781)

    提出低秩克隆（LRC）高效预训练方法，利用一组低秩投影矩阵同时实现教师权重的软剪枝和学生激活（含FFN信号）的克隆对齐，从而高效构建与强大教师模型行为等价的小语言模型。

    

    即使借助知识蒸馏和从更大教师模型进行剪枝，训练高性能的小语言模型（SLMs）仍然成本高昂。现有工作通常面临三个关键挑战：（1）硬剪枝导致的信息损失，（2）低效的表示对齐，以及（3）对信息丰富的激活（尤其是来自前馈网络FFN的激活）的利用不足。为解决这些挑战，我们提出了低秩克隆，这是一种高效的预训练方法，旨在构建与强大教师模型行为等价的小语言模型。LRC训练一组低秩投影矩阵，通过压缩教师权重实现软剪枝，并通过将学生的激活（包括FFN信号）与教师的激活对齐实现激活克隆。这种统一的设计在最大化知识迁移的同时，消除了对显式对齐模块的需求。基于开源教师模型（如Llam…）的大规模实验验证了该方法的有效性。

    arXiv:2505.12781v5 Announce Type: replace-cross  Abstract: Training high-performing Small Language Models (SLMs) remains costly, even with knowledge distillation and pruning from larger teacher models. Existing work often faces three key challenges: (1) information loss from hard pruning, (2) inefficient alignment of representations, and (3) underutilization of informative activations, particularly from Feed-Forward Networks (FFNs). To address these challenges, we introduce Low-Rank Clone (LRC), an efficient pre-training method that constructs SLMs aspiring to behavioral equivalence with strong teacher models. LRC trains a set of low-rank projection matrices that jointly enable soft pruning by compressing teacher weights, and activation clone by aligning student activations, including FFN signals, with those of the teacher. This unified design maximizes knowledge transfer while removing the need for explicit alignment modules. Extensive experiments with open-source teachers (e.g., Llam
    
[^440]: SARTM：基于语言辅助蒸馏的任意RGB-热红外分割模型

    SARTM: Segment Any RGB Thermal Model with Language aided Distillation

    [https://arxiv.org/abs/2505.01950](https://arxiv.org/abs/2505.01950)

    提出SARTM框架，通过LoRA微调和语言引导蒸馏将SAM适配到RGB-热红外语义分割任务，在低光和过曝等恶劣光照条件下实现鲁棒的场景理解。

    

    最近提出的任意分割模型（SAM）在各种下游任务中展现出强大的实例分割性能。然而，SAM仅在RGB数据上训练，限制了其在RGB-热红外（RGB-T）语义分割中的直接适用性。鉴于RGB-T为恶劣天气和光照条件（如低光和过曝）下的场景理解提供了稳健的解决方案，我们提出了一种新颖的框架SARTM，将强大的SAM定制用于RGB-T语义分割。我们的核心思想是在释放SAM潜力的同时，为RGB-T数据对引入语义理解模块。具体而言，我们的框架首先通过添加额外的LoRA层来微调原始SAM，旨在保留SAM强大的泛化和分割能力以用于下游任务。其次，我们引入语言信息作为训练SARTM的引导，以解决跨模态不一致问题。

    arXiv:2505.01950v2 Announce Type: replace-cross  Abstract: The recent Segment Anything Model (SAM) demonstrates strong instance segmentation performance across various downstream tasks. However, SAM is trained solely on RGB data, limiting its direct applicability to RGB-thermal (RGB-T) semantic segmentation. Given that RGB-T provides a robust solution for scene understanding in adverse weather and lighting conditions, such as low light and overexposure, we propose a novel framework, SARTM, which customizes the powerful SAM for RGB-T semantic segmentation. Our key idea is to unleash the potential of SAM while introduce semantic understanding modules for RGB-T data pairs. Specifically, our framework first involves fine-tuning the original SAM by adding extra LoRA layers, aiming at preserving SAM's strong generalization and segmentation capabilities for downstream tasks. Secondly, we introduce language information as guidance for training our SARTM. To address cross-modal inconsistencies,
    
[^441]: X-SG$^2$S：具有X维水印的安全且可泛化的高斯泼溅

    X-SG$^2$S: Safe and Generalizable Gaussian Splatting with X-dimensional Watermarks

    [https://arxiv.org/abs/2502.10475](https://arxiv.org/abs/2502.10475)

    提出了X-SG$^2$S框架，能够在3D高斯泼溅场景中同时注入1D到3D的多模态水印以实现版权保护，同时保持原始场景的高保真度。

    

    3D高斯泼溅（3DGS）已被广泛应用于3D重建和3D生成领域。然而，3D高斯泼溅的快速普及引发了人们对信息泄露和未经授权使用的日益关注，促使人们探索有效的水印技术。然而，现有方法存在容量低、在几何扰动下脆弱，以及需要昂贵的微调或流水线修改等不切实际的要求等局限，这促使人们需要一个可泛化的、前馈式的框架，能够以最小的侵入实现稳健的多模态嵌入。在本文中，我们提出了一个新框架X-SG$^2$S，它可以同时注入1D到3D的水印以实现版权保护，同时保持原始3DGS场景的高保真度。具体而言，我们首先将水印分割成消息补丁，并开发了一种自适应门控机制来选择水印消息的注入位置。然后，我们使用……

    arXiv:2502.10475v3 Announce Type: replace-cross  Abstract: 3D Gaussian Splatting (3DGS) has been widely used in 3D reconstruction and 3D generation. However, the rapid adoption of 3D Gaussian Splatting raises growing concerns about information leakage and unauthorized use, urging the exploration of effective watermarking techniques. However, existing methods are limited by low capacity, fragility under geometric perturbations, and the infeasible requirement for costly fine-tuning or pipeline modifications, motivating the need for a generalizable, feed-forward framework capable of robust multi-modal embedding with minimal intrusion. In this paper, we propose a new framework X-SG$^2$S which can simultaneously inject 1D to 3D watermarks for copyright protection, while keeping the high fidelity of original 3DGS scenes. Specifically, we first split the watermarks into message patches. A self-adaptive gate is developed to select the injection positions of the watermark messages. Then, we use
    
[^442]: 基于大语言模型的人格情境判断测试自动题目生成

    Automatic Item Generation for Personality Situational Judgment Tests with Large Language Models

    [https://arxiv.org/abs/2412.12144](https://arxiv.org/abs/2412.12144)

    本研究开发并评估了一个基于大语言模型（GPT-4和ChatGPT-5）自动生成人格情境判断测试题目的结构化、可推广框架，通过三项研究系统考察了提示词设计与温度设置对题目内容效度的影响，显著降低了传统SJT开发对专家的依赖。

    

    通过情境判断测试（SJT）进行人格评估，相较于传统的李克特式自我报告量表具有独特优势，但其开发仍然劳动密集、耗时，且严重依赖领域专家。大语言模型（LLM）的最新进展显示出在自动题目生成（AIG）方面的潜力。基于这些进展，本研究着重于开发并评估一个结构化、可推广的人格SJT自动生成框架，并以GPT-4和ChatGPT-5作为实证示例。本研究共开展了三项研究。研究1系统比较了提示词设计和温度设置对LLM生成题目内容效度的影响，以开发一种有效且稳定的基于LLM的人格SJT自动题目生成方法。结果表明，经过优化的提示词和1.0的温度设置在GPT-4上实现了创造力与准确性的最佳平衡（摘要在此处被截断）。

    arXiv:2412.12144v5 Announce Type: replace-cross  Abstract: Personality assessment through situational judgment tests (SJTs) offers unique advantages over traditional Likert-type self-report scales, yet their development remains labor-intensive, time-consuming, and heavily dependent on subject matter experts. Recent advances in large language models (LLMs) have shown promise for automatic item generation (AIG). Building on these developments, the present study focuses on developing and evaluating a structured and generalizable framework for automatically generating personality SJTs, using GPT-4 and ChatGPT-5 as empirical examples. Three studies were conducted. Study 1 systematically compared the effects of prompt design and temperature settings on the content validity of LLM-generated items to develop an effective and stable LLM-based AIG approach for personality SJT. Results showed that optimized prompts and a temperature of 1.0 achieved the best balance of creativity and accuracy on G
    
[^443]: 让每个人都满意：少量副本下大量物品的在线公平分配

    Keep Everyone Happy: Online Fair Division of Numerous Items with Few Copies

    [https://arxiv.org/abs/2408.12845](https://arxiv.org/abs/2408.12845)

    针对物品数量多而副本少的在线公平分配难题，本文创新性地假设效用是物品-智能体特征的未知函数，并将其建模为上下文老虎机问题，从而克服了无法准确估计所有物品-智能体对效用的局限。

    

    本文研究了在线公平分配问题的一种新变体，该问题涉及多个智能体，学习者按顺序观察到不可分割的物品，必须将其不可撤销地分配给其中一个智能体，以在公平性和效率之间实现理想的平衡。现有算法假设物品数量少且副本数量足够大，这保证了能够从带噪声的观测效用中对所有物品-智能体对进行良好的效用估计。然而，这一假设在许多现实应用中可能不成立，例如，一个在线平台拥有大量用户（物品），这些用户仅使用平台的服务提供商（智能体）少数几次（即物品只有少量副本），这使得难以准确估计所有物品-智能体对的效用。为了解决这一局限性，我们假设效用是物品-智能体特征的未知函数，并提出将在线公平分配建模为上下文老虎机问题的算法。

    arXiv:2408.12845v3 Announce Type: replace-cross  Abstract: This paper considers a novel variant of the online fair division problem involving multiple agents in which a learner sequentially observes an indivisible item that must be irrevocably allocated to one of the agents to achieve a desired balance between fairness and efficiency. Existing algorithms assume a small number of items with a sufficiently large number of copies, which ensures a good utility estimation for all item-agent pairs from noisy observed utilities. However, this assumption may not hold in many real-life applications, e.g., an online platform with a large number of users (items) who use the platform's service providers (agents) only a few times (a few copies of items), making it difficult to accurately estimate utilities for all item-agent pairs. To address this limitation, we assume utility is an unknown function of item-agent features. We propose algorithms that model online fair division as a contextual bandit
    
[^444]: FedReview: 一种用于拒绝毒化更新的联邦学习审查机制

    FedReview: A Review Mechanism for Rejecting Poisoned Updates in Federated Learning

    [https://arxiv.org/abs/2402.16934](https://arxiv.org/abs/2402.16934)

    提出了FedReview机制，通过随机分配评审员客户端来识别和拒绝联邦学习中的潜在毒化更新，并采用多数表决机制来整合排名并移除这些更新。

    

    Federated learning最近已经被提出作为一种去中心化的方法，在不访问用户数据的情况下学习一个高性能模型。尽管其有效性，但联邦学习给恶意用户提供了机会通过向服务器上传毒化模型更新来操纵模型。在本文中，我们提出了一种名为FedReview的审查机制，用于识别和拒绝联邦学习中潜在的毒化更新。在我们的机制下，服务器每轮随机分配子集客户端作为评审员，在其训练数据集上评估模型更新。评审员根据评价结果对模型更新进行排名，统计相对低质量的更新数量作为估计的毒化更新数量。基于审查报告，服务器采用多数表决机制整合排名并在模型聚合过程中去除潜在的毒化更新。

    arXiv:2402.16934v1 Announce Type: cross  Abstract: Federated learning has recently emerged as a decentralized approach to learn a high-performance model without access to user data. Despite its effectiveness, federated learning gives malicious users opportunities to manipulate the model by uploading poisoned model updates to the server. In this paper, we propose a review mechanism called FedReview to identify and decline the potential poisoned updates in federated learning. Under our mechanism, the server randomly assigns a subset of clients as reviewers to evaluate the model updates on their training datasets in each round. The reviewers rank the model updates based on the evaluation results and count the number of the updates with relatively low quality as the estimated number of poisoned updates. Based on review reports, the server employs a majority voting mechanism to integrate the rankings and remove the potential poisoned updates in the model aggregation process. Extensive evalu
    
[^445]: 构建富有表现力和可处理的概率生成模型：一项综述

    Building Expressive and Tractable Probabilistic Generative Models: A Review

    [https://arxiv.org/abs/2402.00759](https://arxiv.org/abs/2402.00759)

    本文综述了富有表现力和可处理的概率生成建模领域的进展和技术，并重点关注了概率电路。文章提供了关于表达能力和可处理性之间权衡的统一视角，并说明了设计原则和算法扩展，成功地构建了富有表现力和高效的概率电路。此外，文章还讨论了最新的深度和混合概率电路研究，并概述了未来研究的挑战和开放性问题。

    

    我们对可处理的概率生成建模领域中的进展和技术进行了全面的调查，重点关注概率电路（PCs）。我们提供了关于表达能力和可处理性之间固有权衡的统一视角，突出了使PCs富有表现力和高效的设计原则和算法扩展，并提供了该领域的分类法。我们还讨论了最近通过融合深度神经模型概念来构建深度和混合PCs的努力，并概述了指导未来研究的挑战和开放性问题。

    We present a comprehensive survey of the advancements and techniques in the field of tractable probabilistic generative modeling, primarily focusing on Probabilistic Circuits (PCs). We provide a unified perspective on the inherent trade-offs between expressivity and the tractability, highlighting the design principles and algorithmic extensions that have enabled building expressive and efficient PCs, and provide a taxonomy of the field. We also discuss recent efforts to build deep and hybrid PCs by fusing notions from deep neural models, and outline the challenges and open questions that can guide future research in this evolving field.
    

