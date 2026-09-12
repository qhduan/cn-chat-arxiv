# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [GPU-CFR: 80x Faster Counterfactual Regret Minimization by Compiling the Game to Static Dataflow and CUDA Graph Replay](https://arxiv.org/abs/2609.11923) | 本文提出GPU-CFR，利用“固定博弈的CFR迭代结构可预先确定”这一观察，将博弈一次性编译为静态数据流并通过CUDA图重放执行，消除了GPU内核启动与框架调度开销，使反事实遗憾最小化的运行速度相比以往方案提升高达80倍。 |
| [^2] | [General Quantification of Covariate and Concept Shifts](https://arxiv.org/abs/2609.11918) | 本文基于熵最优传输提出γ*-概念偏移新概念和DataShifts算法，统一量化了协变量偏移与概念偏移，并给出了可从样本估计、适用于广泛场景的一般化误差界。 |
| [^3] | [Can Edge-Deployable Vision-Language Models Identify Species?](https://arxiv.org/abs/2609.11916) | 该研究首次系统评估了2-8B参数边缘可部署视觉语言模型的物种识别能力，发现它们具备真正的分类学知识，但在真实红外相机影像上所有模型（包括专业模型BioCLIP）均因图像可读性问题导致性能显著下降9.6-26.6个百分点。 |
| [^4] | [Generative Marketing Mix Modeling: A Causal Inference Framework Linking GEO and GEM to Business Impact](https://arxiv.org/abs/2609.11915) | 本文提出生成式营销组合建模（GMMM）这一因果推断框架，通过结合注意概率与生成答案、赞助记录等数据，首次实现了对生成式引擎优化（GEO）和生成式引擎营销（GEM）商业因果效应的估计。 |
| [^5] | [Artificial Id: Drive and Persistent Alignment in Agentic AI](https://arxiv.org/abs/2609.11911) | 本文提出“人造本我”——一种自适应内部驱动力，使智能体AI无需外部显式指定行为规则，即可自主决定行为是继续、停止还是改变，并通过虚拟实验证明有用的控制能力可以经由差异性持久性自发涌现。 |
| [^6] | [MindTopo: Can Foundation Models Reason in Topological Space?](https://arxiv.org/abs/2609.11900) | 该论文提出了MindTopo基准，基于认知科学与形式拓扑学，从连续性、分离性、有序性、包含性和纽结五个拓扑属性出发，在推理与规划两个认知层面对14个多模态大语言模型的拓扑空间能力进行系统评估。 |
| [^7] | [Domain-Specific Hallucination Detection in Large Language Models](https://arxiv.org/abs/2609.11878) | 提出了一种结合微调DeBERTa-v3分类、蒙特卡洛Dropout不确定性量化和温度缩放校准的多信号幻觉检测流水线，在HaluEval基准上取得F1=0.915的高性能，并结合直接偏好优化（DPO）进一步改进模型表现。 |
| [^8] | [Biology-in-the-loop: Amortized Adaptive Hit Discovery in CRISPR Screens](https://arxiv.org/abs/2609.11877) | 本文提出了包含1,389个CRISPR筛选实验的大规模基准AssayBench-Loop，并在此基础上构建了AssayLoop顺序实验设计框架，利用跨历史实验训练的transformer摊销式采集策略，实现CRISPR筛选中受限预算下的自适应命中物发现。 |
| [^9] | [On the Regularization Landscape for the Linear Recommendation Models](https://arxiv.org/abs/2609.11876) | 本文将多种深度学习启发的线性推荐算法统一到正则化框架下，证明它们本质上都可归结为核范数或Frobenius范数正则化，并揭示了两者在预测能力、解的秩与计算效率之间的权衡。 |
| [^10] | [The Last AI Built by Humans: Toward Genuine Recursive Self-Improvement](https://arxiv.org/abs/2609.11873) | 本文提出递归自我改进（RSI）概念及其从改进执行自主性到递归元改进的发展路线图，通过Headroom-Closed指数揭示现有大语言模型的局限，并结合行业实践识别出实现真正AI自我改进的关键挑战。 |
| [^11] | [RetroThinker: Enabling Retrospective Thinking in Speech LLMs](https://arxiv.org/abs/2609.11864) | 该论文提出了RetroThinker多阶段后训练框架，使流式语音大语言模型能够在推理过程中自我验证并即时修正思维链推理步骤，从而在满足实时语音交互延迟约束的同时提升复杂推理能力。 |
| [^12] | [Explainability Assistant: A Conversational XAI Interface for Interpreting Energy Consumption Models](https://arxiv.org/abs/2609.11860) | 本文提出了一种利用大语言模型函数调用能力的开源对话式XAI系统“可解释性助手”，将意图解析准确率从76.8%提升至94%，帮助设施管理者等非技术人员更灵活地理解能源消耗预测模型。 |
| [^13] | [From Parameters to Answers: How LLMs Retrieve and Use Their Internal Knowledge](https://arxiv.org/abs/2609.11859) | 该论文通过对Qwen、Llama和Gemma模型隐藏状态进行逐层干预，揭示了大语言模型在回答问题时“查询路由方向”先于目标知识形成并影响答案生成的因果机制与时间窗口。 |
| [^14] | [Model-Aware Schedules Improve Generation via Fiberwise Optimal Transport](https://arxiv.org/abs/2609.11842) | 该论文提出一种基于逐纤维最优传输的模型感知调度构建方法，通过将预测风险与动能作用结合得到闭式最优时间分配，从而改进扩散与流匹配模型的生成效果。 |
| [^15] | [Understanding Operator Attitudes Toward AI-Supported Decision Making in Maritime Operations](https://arxiv.org/abs/2609.11805) | 本研究通过问卷调查与开放回答分析发现，海事专业人员对AI辅助避碰决策助手总体持积极态度、信任稳定，重视其在决策支持和态势感知方面的价值，同时对AI可靠性仍存顾虑。 |
| [^16] | [Logit Refiner: Improving Visual Autoregressive Models via Intra-Scale Dependency Modeling](https://arxiv.org/abs/2609.11804) | Logit Refiner 是一个轻量级自回归模块，通过恢复 VAR 模型尺度内 token 间的空间依赖关系，仅需约 10% 的额外参数即可免重训直接插入任何预训练 VAR 模型，有效改善生成图像的局部连贯性。 |
| [^17] | [Thinking with Looped Flows](https://arxiv.org/abs/2609.11801) | 提出循环流方法，通过局部去噪目标训练循环隐藏状态，使早期更新能支持后续更新，从而通过更多推理计算解决更难的问题。 |
| [^18] | [Beyond Word Error Rate: A Switch Aware Evaluation of ASR and Audio Language Models on English Yoruba Code-Switched Speech](https://arxiv.org/abs/2609.11786) | 该论文提出了一套切换感知的评估指标（包括切换入口标记错误率SETER等），揭示总体词错误率会掩盖码转换语音识别的真实表现，且领先的音频语言模型虽然与最佳ASR模型的WER相当，却在所有码转换相关指标上显著更优。 |
| [^19] | [Recognizing Is Not Reversing: A Controlled Inversion Test of Fact-Preserving News Framing](https://arxiv.org/abs/2609.11769) | 该研究通过受控逆转实验发现，大语言模型虽能较好地保持事实并识别新闻框架，却几乎无法逆转已知的框架变换（逆转率仅约0.044–0.068），证明“识别框架”与“逆转框架”是截然不同的能力。 |
| [^20] | [A Unified Per-Token Gating Family for On-Policy Distillation: FKL/RKL Mixing with Multi-Channel and Bias Coefficients](https://arxiv.org/abs/2609.11768) | 该论文提出了一个统一的四系数逐词元门控参数化方法，将EOPD和ToDi作为其一维限制的特殊情形统一起来，并通过多通道组合与显式偏置这两个额外自由度，在在线策略蒸馏实验中显著超越了单通道门控限制方法。 |
| [^21] | [SIRF: A Spec-Internalized Risk Foundation Model for Industrial Content Risk Control](https://arxiv.org/abs/2609.11752) | SIRF通过持续预训练将平台风控规范内化到模型权重中，无需人工标注即可在超低延迟、仅输出判定的部署形态下实现高精度风险处置，仅用约7000万token的CPT便将P95精确率下的黑名单召回率提升15.1个百分点且不损害通用能力。 |
| [^22] | [LOCUS: Task-Aware Low-Rank Post-Training for Token-Efficient Language Generation](https://arxiv.org/abs/2609.11739) | LOCUS通过选择任务感知的低秩适配子空间进行后训练，在不修改偏好对齐损失且仅更新不到0.3%参数的情况下，将大模型输出长度最多缩短39.84%，从而在保持效用的同时大幅降低推理成本。 |
| [^23] | [ORCH: Organizational Principles Enable Collective Intelligence in Embodied AI](https://arxiv.org/abs/2609.11737) | 提出ORCH框架，借鉴人类组织理论，通过结合汇集式与顺序式相互依赖，为多达50个异构具身AI智能体构建任务特定的层级化组织结构，从而在野火响应等复杂物理任务中实现集体智能。 |
| [^24] | [Continuous-Time Acoustic Modelling with Neural Controlled Differential Equations](https://arxiv.org/abs/2609.11725) | 本文提出使用神经控制微分方程（CDEs）实现TTS中时长感知的连续时间声学建模，使隐藏状态的数值能够随语音内容和时长信息连续演化，而不仅仅改变状态出现的位置和频率。 |
| [^25] | [A Time-Based Readout for Vector-Matrix Multiplication in Fully Analog Memristive SNNs](https://arxiv.org/abs/2609.11713) | 本文提出一种基于电压-时间转换的全模拟读出架构，用于忆阻器SNN中的向量-矩阵乘法，省去电流模式读出电路，显著提升面积与能效。 |
| [^26] | [When Agents Disagree: Bayesian Backward Reasoning as a Label-Free Anchor for Multi-Agent Collective Decision-Making](https://arxiv.org/abs/2609.11709) | 该论文提出通过贝叶斯反向推理构建与前向推理具有不同分解方式的反向后验，并利用Jensen-Shannon散度衡量智能体间的跨路径一致性进行排序，从而为无标签场景下的多智能体集体决策提供锚点。 |
| [^27] | [Language-Augmented Semantic Priors for B-Spline Surface Fitting](https://arxiv.org/abs/2609.11708) | LASP框架利用大语言模型从程序化建模历史中推断出结构化、可供求解器使用的B样条语义先验，弥合了CAD系统中高层设计意图与几何求解器配置之间的鸿沟，实现语义一致的曲面拟合。 |
| [^28] | [ActSafeGuard: Differentiable and Training-Aligned Constraint Enforcement for Flow-Matching Policies](https://arxiv.org/abs/2609.11697) | ActSafeGuard 提出了一个可微且训练对齐的安全防护层，通过解析式射线缩放算子将硬性动作可行性约束直接融入流匹配策略的训练过程，利用边界感知梯度引导模型自然学习受约束流形，解决了现有安全方法训练与执行不匹配的问题。 |
| [^29] | [COBRA-Skills: Contextual Bandit-Guided Evolution for Agent Skill Optimization](https://arxiv.org/abs/2609.11682) | COBRA-Skills提出了一种将智能体技能优化建模为预算受限序贯优化的高效框架，通过上下文赌博机引导的优先级排序与基于证据的技能进化，在六个智能体基准上取得最强平均性能的同时，将优化成本较SkillOpt降低55%-58%。 |
| [^30] | [Ecdysis: Efficient and Effective Training of Runtime Harnesses for LLM Agents](https://arxiv.org/abs/2609.11677) | 该论文指出运行时框架进化中缺乏有原则的失败诊断是关键瓶颈——观察到的失败可能源于模型自身缺陷或框架系统性缺陷，直接针对个别失败优化会导致不必要的模型特定适应，因此提出Ecdysis方法以实现对LLM智能体运行时框架更高效、更有效且泛化能力更强的训练。 |
| [^31] | [Geospatial AI, Dataverse Metadata, and the Study of Place-Based Government](https://arxiv.org/abs/2609.11674) | 本文从哈佛Dataverse的公开数据和元数据中构建了一个包含21.5万余节点的知识图谱，首次将自由文本形式的地理信息整合为可检索的结构化网络，并识别出数千个与政策直接相关的地理空间数据集。 |
| [^32] | [Warrant Theory](https://arxiv.org/abs/2609.11667) | 本文提出“担保理论”这一新的哲学学科，将逻辑重构为规约命题引入、接受、拒绝与推理运用的规范性框架，以“担保”即推理权利为核心概念，并通过接受与拒绝的双边词汇系统分析命题推理地位的产生、发展及其互动关系。 |
| [^33] | [Autonomy, Social Norms, and Alignment: Towards a Developmental Framework for Autonomous Artificial Agents](https://arxiv.org/abs/2609.11660) | 该论文针对自主人工智能体在开放动态环境中自主探索学习时难以与人类目标保持一致的问题，提出构建一个融合自主性、社会规范与对齐机制的发展性框架。 |
| [^34] | [ZipCodec: Ultra-Low-Frame-Rate Streaming Speech Coding](https://arxiv.org/abs/2609.11642) | ZipCodec是一种以6.25 Hz超低帧率和0.80 kbps比特率运行的流式神经语音编解码器，通过大规模WavLM蒸馏、重新设计的transformer架构、标量球面量化和延迟感知流式解码，在重建质量和下游任务上大幅超越现有流式编解码器，并能在消费级CPU上实现实时单流推理。 |
| [^35] | [LoaDiff: Conditional Generation of Electricity Consumption Time Series for Energy Analytics](https://arxiv.org/abs/2609.11639) | LoaDiff是一种基于扩散模型的生成模型，能够根据家庭属性和上下文变量等条件，生成长达一年的逼真智能电表负荷曲线，为能源分析领域提供合规的合成数据替代方案。 |
| [^36] | [MAPLE: Memory-Augmented Planning with Language and Evolution](https://arxiv.org/abs/2609.11636) | MAPLE是一个记忆增强的优化智能体，将自然语言问题构建与数学规划和进化搜索相结合，能够通过连续的自然语言请求动态维护优化问题，同时保留先前决策并重用搜索结果。 |
| [^37] | [Physics-Informed Neural Networks to Infer the Perpendicular Energy Conductivity in the Scrape-Off Layer of Stellarator Devices](https://arxiv.org/abs/2609.11628) | 本文开发了一个反向物理信息神经网络框架，通过将等离子体密度和温度的径向分布测量数据与简化的刮削层输运方程相结合，成功推断出仿星器装置刮削层中垂直热传导率对等离子体密度和温度的依赖关系。 |
| [^38] | [Distributed Optimization of Modular Production Systems using Model-based Reinforcement Learning with Inverse Models](https://arxiv.org/abs/2609.11615) | 本文提出一种在强化策略训练中引入近似逆过程模型的模型强化学习框架，通过解耦执行动力学与状态空间动力学使训练仅在任务空间进行，在模块化生产测试平台上显著提升了性能和训练速度，尤其对离策略算法效果更佳。 |
| [^39] | [Making Alternative Data Work: Context-Augmented LLMs for Financial Forecasting](https://arxiv.org/abs/2609.11607) | 该论文提出利用上下文增强的大语言模型来整合分散在异构渠道中的另类数据，从而克服传统预测方法难以灵活运用企业级另类数据进行财务表现预测的局限。 |
| [^40] | [Learn the Solid, Not the File: Canonical Inputs for Neural Networks on CAD Boundary Representations](https://arxiv.org/abs/2609.11573) | 该论文揭示了现有B-rep编码器面对同一实体的不同表示时性能会灾难性崩溃，并提出了一种从实体本身派生节点、特征和坐标系的规范区域图输入表示，使神经网络对B-rep变化具有不变性。 |
| [^41] | [Enabling Knowledge Graph Understanding at Scale with the EXplore Your Graphs ENgine (EXYGEN)](https://arxiv.org/abs/2609.11569) | EXYGEN 框架通过将自动推导的元数据（VoID 描述和 ShEx 模式）融入检索增强生成（RAG）流水线，无需微调即可让大语言模型实现对大规模知识图谱的对话式访问和文本到 SPARQL 查询生成，在 SciQA 基准上达到 0.419 的精确匹配率。 |
| [^42] | [A Comparative Evaluation of Pre-trained Convolutional Neural Networks for Melanoma Detection](https://arxiv.org/abs/2609.11550) | 该论文对多种预训练卷积神经网络在黑色素瘤检测任务中进行了比较评估，旨在为皮肤镜和组织病理学等不同成像方式确定最合适的网络架构。 |
| [^43] | [Characterizing Job Power Elasticity for Power-Flexible AI Training](https://arxiv.org/abs/2609.11542) | 本文首次系统表征了LLM训练中的任务功率弹性，并提出功率灵活性指数（PFI）这一归一化指标来量化GPU功率降低的性能代价，为SLA感知的功率灵活性提供了控制原语。 |
| [^44] | [Prompt Revision as a Source of Cultural Bias in Text-to-Image Systems](https://arxiv.org/abs/2609.11532) | 该研究提出WORLDVIEW多语言基准，首次审计了文生图系统中用户不可见的提示词隐性修订层，揭示其是非西方和非英语文化偏见的重要来源，这些文化语境被系统性地过度标记、词汇扁平化并固化为刻板印象。 |
| [^45] | [Lightweight LiDAR-Based Cone Detection Framework Using Random Forest for Formula Student Driverless](https://arxiv.org/abs/2609.11527) | 本文提出了一种面向大学生方程式无人赛的轻量级纯激光雷达锥桶检测框架，利用随机森林分类并专为CPU设计，在纯CPU硬件上实现98.33%的F1分数和3.13毫秒的端到端运行时间，并开源了数据集、标注工具和模型以支持复现。 |
| [^46] | [Learning Interaction between Image and Layout Priors for Joint Image-Layout Generation in Design Templates](https://arxiv.org/abs/2609.11519) | 提出InterIL模型，通过可学习通信模块连接预训练的图像与布局扩散模型，在单一生成过程中联合生成背景图像和前景布局，显式建模两者的双向交互，从而提升设计模板的生成质量。 |
| [^47] | [Extending SMT Solving with Non-Ground Clause Learning](https://arxiv.org/abs/2609.11509) | 该论文提出了一种结合基例实例化、CDCL(T)风格规则与非基态冲突分析的SMT求解演算，通过在原始非基子句上执行归结步骤来学习更一般且非冗余的子句，从而实现可能指数级更短的证明。 |
| [^48] | [Structural priors for data-efficient language learning](https://arxiv.org/abs/2609.11505) | 该研究发现先在音乐、概率文法和元胞自动机等符号数据上训练模型，可为后续语言建模带来更低损失和更小的权重偏移，但这种损失优势并不能稳定转化为更好的下游语言能力。 |
| [^49] | [ActMap: Single-Pass Uncertainty Quantification from Generation-Time Activation Maps](https://arxiv.org/abs/2609.11498) | ActMap提出了一种白盒激活图表示方法，将生成时所有层、所有词元的隐藏状态轨迹压缩为固定尺寸的低开销张量，从而在单次生成过程中零额外开销地实现大型语言模型的不确定性量化。 |
| [^50] | [From Document Silos to Process Intelligence: A Multi-Layer Knowledge Graph for CMC Process Development](https://arxiv.org/abs/2609.11493) | 该论文提出一个模块化智能体AI平台，将CMC工艺开发中异构格式的文档转化为可查询的双层知识图谱，实现了从药物发现到商业化生产全流程的知识整合与可追溯性。 |
| [^51] | [Published Unlearning Numbers Move Per Checkpoint, and Not Because the Removed Data Survives: An Audit of 263 Released Batch-Normalized Checkpoints](https://arxiv.org/abs/2609.11490) | 对263个已发布的批量归一化检查点的审计表明，重新拟合批归一化统计量会使部分检查点的已发布去学习数值发生显著变动，但这种变动源于检查点发布状态与重新拟合结果之间的漂移，而非被删除数据在模型中的残留存活，因此对已发布去学习结论的影响真实但有限。 |
| [^52] | [The Convention Gap: Towards Measuring Implicit Communication in Cooperative AI Evaluation](https://arxiv.org/abs/2609.11489) | 该论文提出“惯例差距”这一新指标，通过在Hanabi游戏中对比从字面交流预测的失败概率与实际失败率，发现人类玩家在合作中依赖超出字面信息的隐性惯例（差距达26.2个百分点），而AI-AI对局中不存在这种差距，为评估合作AI的隐性交流能力提供了可精确计算的方法。 |
| [^53] | [Flexible and Interpretable Accent Distance Measurements](https://arxiv.org/abs/2609.11458) | 本文提出利用发音倒演生成的发音表征结合最优传输框架，实现了一种既灵活适用于任意语音录音、又具备可解释性的口音距离测量新方法。 |
| [^54] | [RouteRepair: Instance-Level Failure Diagnosis and Targeted Repair in LLM-Based Automated Heuristic Design for Routing Optimization](https://arxiv.org/abs/2609.11452) | RouteRepair通过实例级失败诊断和针对性修复机制，改进了基于大语言模型的路径优化自动启发式设计，在修复父代启发式方法特定弱点的同时保护已经表现良好的行为。 |
| [^55] | [Cross-Lingual Clinical Annotation Projection as Constrained Text Generation: A Six-Language Study](https://arxiv.org/abs/2609.11450) | 该研究提出将跨语言临床标注投影建模为受限文本生成任务，通过将实体标签直接插入不可修改的目标语言文本并进行确定性验证，在六种语言上实现了最强且最一致的临床标注迁移性能。 |
| [^56] | [Prevalence Determines Precision:Silent Contamination in Detector-Defined Datasets](https://arxiv.org/abs/2609.11449) | 论文通过贝叶斯分析揭示，由检测器定义的机器学习数据集的精确度主要由候选池中真阳性流行率决定而非检测器质量，虚假样本构成具有检测器形状的第二信号而非加性噪声，跨池迁移精确度评估可产生高达422%的误差。 |
| [^57] | [Investigating catastrophic forgetting in sound event classification](https://arxiv.org/abs/2609.11447) | 这项研究发现声音事件分类中的灾难性遗忘主要发生在网络深层尤其是分类器头部，而完全冻结特征提取器并微调动态头部分类器是缓解灾难性遗忘最高效的方案。 |
| [^58] | [Calibration-Aware Uncertainty Cascades for Efficient Heterogeneous Model Collaboration](https://arxiv.org/abs/2609.11446) | 该论文提出CAUC框架，通过独立校准各模型的置信度建立统一的可靠性尺度，将部署策略与特定模型池解耦，实现高效且灵活的异构模型协作。 |
| [^59] | [LLMs as Post-hoc Auditors of Physiological Plausibility in Symbolic Regression: A Clinician-Evaluated Case Study](https://arxiv.org/abs/2609.11431) | 本研究创新性地将大语言模型用作事后审计工具，对符号回归进化生成的数学表达式进行可解释性和医学合理性分析排序，并通过临床医生评估验证了该方法的有效性。 |
| [^60] | [SWRouter: Similarity-Contractive Window Routing for Multi-Turn Large Language Model Conversations](https://arxiv.org/abs/2609.11414) | 提出SWRouter，通过基于相似性的上下文分割机制与双指标评估框架，解决多轮对话中大语言模型路由面临的上下文信息丢失混淆及评估指标耦合两大难题。 |
| [^61] | [X-AuT: Progressive Audio-Encoder Compression for Speech LLMs with Cross-Scale Distillation](https://arxiv.org/abs/2609.11412) | X-AuT提出了一种渐进式音频编码器压缩框架，通过行为探针选择层组合，并结合跨尺度蒸馏与LoRA微调恢复剪枝模型，成功将Qwen3-ASR-0.6B的音频编码器从18层压缩至14层，在中英文基准测试中仍保持较低的错误率。 |
| [^62] | [Deep-Fake CAPTCHA: Mitigating Next-Generation Social Engineering Attacks](https://arxiv.org/abs/2609.11404) | 本文提出DF-CAPTCHA框架，通过让呼叫者执行对人类简单但对实时深度伪造系统难以伪造的挑战-响应任务，并从真实性、身份一致性、任务完成度和响应时间四个维度验证响应，从而主动防御语音和视频通话中的深度伪造攻击。 |
| [^63] | [From Queries to Narratives: Cultural Heritage Data Stories for Knowledge Graph Exploration and Quality Assessment](https://arxiv.org/abs/2609.11403) | 本文提出“数据故事”方法，降低文化遗产知识图谱的查询门槛，并将用户探索转化为数据质量评估，实现便捷查询与发现隐藏数据问题的双重目标。 |
| [^64] | [Beyond Confidence: Stability-Aware Test-Time Adaptation for LLM Reasoning](https://arxiv.org/abs/2609.11393) | 提出TASCO框架，发现高置信度在局部扰动下保持稳定时更可能正确，从而在冻结LLM的前提下通过优化轻量级任务级前缀，将局部稳定性引入基于置信度的测试时自适应，有效提升大语言模型推理能力。 |
| [^65] | [Buyer Artificial Intelligence-Enabled Environmental Governance and Supplier Environmental Controversies: An Organizational Information Processing and Signaling](https://arxiv.org/abs/2609.11391) | 本研究基于组织信息处理理论和信号传递理论，通过对41个国家2505家供应商面板数据的分析，发现买方采用人工智能驱动的环境治理能够显著降低海外供应商次年发生环境争议的风险。 |
| [^66] | [VikingRAG: Accurate and Token-efficient Retrieval-augmented Generation over Structured Documents](https://arxiv.org/abs/2609.11390) | VikingRAG通过将代理式多轮检索轨迹物化为可复用的经验边，并引入自适应升级策略在证据充分时采用单轮经验增强检索，在保持高准确率的同时大幅降低了结构化文档检索增强生成的令牌成本。 |
| [^67] | [Agent-Integrated Software: Interaction Contracts and Continuous Assurance](https://arxiv.org/abs/2609.11381) | 该论文提出智能体集成软件（AIS）软件模式与意图级交互抽象（IIA）任务语义，通过交互契约约束任务级交互与应用行为之间的对应关系，并以持续保障机制在依赖变化时维护声明，从而解决将智能体嵌入现有应用时的协调问题。 |
| [^68] | [Characterizing Bluesky Content Moderation Service: From Automation of Service to Landscape of Harms](https://arxiv.org/abs/2609.11373) | 本研究首次对去中心化社交平台Bluesky的默认审核系统进行了大规模独立审计，通过分析1060万条审核标签，揭示了其人机协作的自动化机制、危害检测效能及所识别的危害类型全景。 |
| [^69] | [RAMamba-Net: A Reliability-Aware and Mamba-Based Multimodal Fusion Network for Auditory Attention Detection](https://arxiv.org/abs/2609.11372) | 该论文提出了RAMamba-Net，一种可靠性感知的基于Mamba的多模态融合网络，通过融合EEG与EOG信号、Mamba增强的频带感知Transformer以及跨模态注意力机制，显著提升了听觉注意力检测的性能与鲁棒性。 |
| [^70] | [Portable Semantics, Private Dialects: Reuse and Negative Transfer in Latent Communication Between Language-Model Cells](https://arxiv.org/abs/2609.11365) | 该研究通过对六个独立训练的语言模型社会进行泄漏受控的因果互操作性审计，发现语义相似的潜在接口并不构成统一语言——仅相同初始化的社会对可完全互操作，且继承的接口状态可能对后续学习产生负迁移。 |
| [^71] | [Reification as a Transferable Vocabulary: Zero-Shot Link Prediction with Vanilla GNNs](https://arxiv.org/abs/2609.11347) | 该研究提出将知识图谱“具体化”为事实节点并通过六个元关系的固定词汇表连接，使普通GNN（如现成的GAT）无需任何专用架构即可实现零样本归纳链接预测，性能媲美预训练基础模型ULTRA。 |
| [^72] | [Exploring Diffusion Transformers for Cross-Modal Augmentation in Multimodal Brain State Decoding](https://arxiv.org/abs/2609.11341) | 本文提出CoMA-DiT，一种双向跨模态扩散Transformer，将成对模态视为相互生成监督的来源以实现潜在空间数据增强，在多模态脑状态解码任务上显著优于20个代表性基线方法。 |
| [^73] | [On the Impact of Anonymization on the Performance of Large Language Models](https://arxiv.org/abs/2609.11335) | 本文系统评估了五个大语言模型在十一个基准上处理原始与匿名化输入的性能差异，发现匿名化总体上会降低性能，且影响因模型能力与任务类型而异——能力最强的模型性能下降最大，TruthfulQA性能反而提升，而检索类任务则遭遇灾难性下降。 |
| [^74] | [E-CONAN (Entailment, CONtradition And Neutral) Benchmarks: Arabic Textual Entailment and Natural Inference Datasets](https://arxiv.org/abs/2609.11334) | 本文提出了面向阿拉伯语文本蕴含与自然推理的E-CONAN基准数据集，其句子对来源于自动翻译、人工验证翻译、手工编写及含谣言的新闻标题等多种渠道，并利用该基准对9个最先进的多语言预训练模型进行了零样本分类评估。 |
| [^75] | [The Semantic Elevation Operator and the Closure of the Undecidable Class under Preservation](https://arxiv.org/abs/2609.11326) | 该论文提出语义提升算子 ΛΦ，将程序静态语义性质问题转化为自修改后的保持性问题，并基于克林递归定理证明不可验证性质类在该算子下封闭，且无界迭代将攀升至算术层级的 Π₂-完备性。 |
| [^76] | [AI Exposure and AI Resilience: A Two-Dimensional Assessment Framework for Software and Software-Based Business Model](https://arxiv.org/abs/2609.11321) | 本文提出AI暴露度与AI韧性（AI-ER）二维评估框架，用于衡量人工智能对软件及软件商业模式带来的变革压力以及企业吸收压力并经济可行地利用AI的能力，弥补了传统技术尽职调查无法评估AI商业影响的不足。 |
| [^77] | [Magenta: Closing the Loop Between Mathematical Reasoning and Lean Verification](https://arxiv.org/abs/2609.11319) | Magenta是一个无需训练的智能体流水线，通过将Lean形式化验证信号整合进LLM的非正式数学推理过程，实现了从自然语言问题到机器检验证明的闭环。 |
| [^78] | [Mr.LHDR: A Benchmark for Multimodal Real-World Long-Horizon Deep Research Agents](https://arxiv.org/abs/2609.11318) | 该论文提出了Mr.LHDR基准测试，通过隐藏节点-关系图构建平均依赖深度达10.4的长程相互依赖证据链，用以评估深度研究智能体在八类多模态真实世界场景中维持漫长研究过程的能力。 |
| [^79] | [Routing by Reasoning Need: Trajectory-Aware Decoding Control for Diffusion Vision-Language Models](https://arxiv.org/abs/2609.11315) | 该论文提出一种免训练的轨迹感知解码控制器，通过分析扩散视觉语言模型的中间答案轨迹信号（答案闭合度、承诺证据、表示修订压力），根据问题各自的推理需求将其动态路由到提前确定、保持基线或推理支持式解码策略，从而解决统一生成长度导致的推理预算不匹配问题。 |
| [^80] | [GRIPNet: Gaussian Radial Intensity Prior Guided Architecture for Pulmonary Nodule Detection in CT](https://arxiv.org/abs/2609.11312) | 本文发现肺结节强度呈高斯径向衰减的规律，并据此提出GRIPNet网络，其每个模块均与强度分布的可测量属性相对应，显著提升了CT图像中小于六毫米肺结节的检测能力。 |
| [^81] | [Your Model Already Knows Don't Teach It, Learn to Ask It: Soft Prompting for Few-Shot Adaptation of Vision-Language Models](https://arxiv.org/abs/2609.11310) | 该论文发现，将少量可学习的软提示词元放置在视觉与文本词元的跨模态边界处并以空格词元初始化，仅需约七千个可训练参数即可在域外少样本目标检测中匹敌最佳LoRA微调配置，同时将训练参数减少超过两万倍。 |
| [^82] | [2AM: Grounding Agent-Side Memory as Guidance for Steerable Action Models in Long-Horizon Manipulation](https://arxiv.org/abs/2609.11308) | 2AM提出一种职责分离架构：由多模态智能体独占任务记忆，并将其编译为子任务语言和2D空间提示，用以引导一个仅依赖RGB、无情节记忆的动作模型完成长时程机器人操作，从而实现清晰的性能归因与可引导的动作控制。 |
| [^83] | [Memory Compression for High-Fanout Agent Sandboxes](https://arxiv.org/abs/2609.11294) | AgentZip是首个专为AI智能体沙箱设计的内存压缩系统，通过利用模板相对冗余和跨沙箱内存冗余，解决了高扇出智能体工作负载带来的内存瓶颈问题。 |
| [^84] | [Off-Target Effects of Response-Style Alignment in a Korean 27B Language Model](https://arxiv.org/abs/2609.11291) | 在韩语27B语言模型上进行响应风格对齐后训练会产生脱靶效应——模型在弃答和信息披露等未被训练目标涉及的行为上发生系统性改变，且这些改变主要由目标文本本身驱动而非提示集或训练配方。 |
| [^85] | [Generating a Consistent Enterprise: Synthesis and Reference-Free Evaluation of Multi-System Business Data](https://arxiv.org/abs/2609.11286) | 本文提出一种无需任何真实数据集的企业数据生成器，可根据行业、规模、商业模式等输入生成在66个业务系统中保持实体身份一致性的完整虚构企业数据，并通过五轴评分卡、对抗性检测等无参考方法评估其真实性。 |
| [^86] | [When Does Text Inform? Benchmarking Information-Theoretic Metrics for Multimodal Time-Series Forecasting](https://arxiv.org/abs/2609.11282) | 该论文首次构建了具有精确已知真值信息量的合成基准数据集，系统评估了六种信息论互信息估计器在判断文本标注是否对多模态时间序列预测有实质贡献方面的可靠性。 |
| [^87] | [Bio-inspired Learning and Decision-Making with Probabilistic In-Memory Computing Hardware: Part 1](https://arxiv.org/abs/2609.11281) | 该论文提出一个生物学基础框架，让有噪声的神经与突触动力学通过随机采样执行贝叶斯推理与学习，并指出新兴概率性模拟存内计算硬件的内在电噪声可以复现这种仿生计算机制。 |
| [^88] | [Predicting Train Delays in Finland Using Machine Learning and Weather Data](https://arxiv.org/abs/2609.11277) | 本文利用融合铁路运营记录与芬兰全国气象传感器网络观测数据的FI-TW数据集，基于XGBoost模型并引入领域驱动的特征工程，实现了芬兰列车延误的精准预测。 |
| [^89] | [Improving Faint Object Detection for Space Situational Awareness with Variational Autoencoders](https://arxiv.org/abs/2609.11269) | 本文提出一种结合轻量级分割网络Tiny-U-Net与偏卷积变分自编码器astro-VAE的深度学习流程，通过自动去除恒星并重建天文背景，显著提升光学空间态势感知图像中低信噪比微弱运动目标的检测能力。 |
| [^90] | [AI-Powered Flare Combustion Efficiency Estimation](https://arxiv.org/abs/2609.11262) | 该论文提出一种结合轻量级视觉-语言编码器与多层感知器的AI方法，可直接从低成本热成像视频预测火炬燃烧效率，并通过易于部署的图形界面实现实时监测、分布展示与CSV报告导出，为偏远或预算受限的工业场所提供了替代昂贵传统仪器的实用方案。 |
| [^91] | [Generative Replay Mitigates Sample Starvation in Quantum Architecture Search](https://arxiv.org/abs/2609.11248) | 本文提出GenQAS框架，将矩阵乘积态热启动与优先级生成式重放相结合，通过学习到的局部转移模型按需生成合成电路转移并与真实经验混合，从而缓解量子架构搜索中有效训练信号稀缺的问题。 |
| [^92] | [Sci-MMR: Benchmarking Multi-Step Evidence-Grounded Scientific Reasoning in Multimodal Agents](https://arxiv.org/abs/2609.11243) | Sci-MMR是基于结构化论证图构建的多步证据支撑科学推理基准，对八个前沿多模态模型的评估显示，答案准确率始终高于完整证据恢复能力，揭示了现有模型缺乏可追溯证据支撑的推理能力。 |
| [^93] | [From Evaluation to Enhancement: Benchmarking and Improving Think-with-Video Reasoning for Video Generative Models](https://arxiv.org/abs/2609.11242) | 该论文提出了涵盖9个推理维度、38个任务的VWG-Bench基准及三级VLM-as-Judge评估协议，揭示了视频生成模型虽渲染能力强但在逻辑与规则推理上存在显著缺陷，并进一步提出Vid-PRE方法来增强模型的“以视频思考”能力。 |
| [^94] | [HALDETECT at ImageEval 2026 Shared Tasks: Answer-First Contrastive Grounding with QLoRA](https://arxiv.org/abs/2609.11236) | HALDETECT系统采用“先答后释”的对比决策框架，并通过4位QLoRA微调Qwen2.5-VL-7B-Instruct（冻结视觉编码器），在ImageEval 2026幻觉检测任务中取得CI 0.035、八队中第三名的成绩，同时证明了答案顺序比模型规模更关键、微调适配优于单纯提示工程。 |
| [^95] | [NovGauge: A Fine-Grained Benchmark for Diagnosing LLMs' Capability in Paper Novelty Assessment](https://arxiv.org/abs/2609.11234) | NovGauge是一个基于人类专家的细粒度基准，通过任务、问题和方法三个维度的级联诊断流水线来诊断大语言模型在论文新颖性评估中的能力，揭示出模型幻觉率最高可达39%。 |
| [^96] | [A Voice-Interactive Multi-Agent System for Smart Operating Rooms: Architecture Design and Key Technologies](https://arxiv.org/abs/2609.11231) | 本文提出基于大语言模型的智能手术室语音交互多智能体系统SurgicalRoomAgent，通过KV Cache前缀预热、流式JSON解析和渐进式技能提示披露三项关键技术大幅降低推理延迟，实现自然语言理解、设备控制、术中记录与手术报告生成。 |
| [^97] | [Solving Few-Shot Multiobjective Multitask Optimization via Iterative Sequential Transfer](https://arxiv.org/abs/2609.11228) | 本文提出迭代顺序迁移（IST）方法，将多任务优化建模为一系列顺序迁移优化问题，每次迭代仅聚焦单一目标，从而克服少样本场景下精英解分布难以识别的瓶颈，有效求解少样本多目标多任务优化问题。 |
| [^98] | [AI Soccer Analyst: Stage-Aware and Verifiable Human-AI Collaboration for Soccer Data Analysis](https://arxiv.org/abs/2609.11224) | 该论文提出了AI足球分析师系统，将足球数据分析分解为数据理解、问题定义、结构化规划等多个可修改、可验证的阶段以实现人机协作，用户研究表明该系统在任务完成质量、可靠性和可验证性方面表现良好。 |
| [^99] | [X-RACE: XAI-assisted Recurrent neural network Attribution for Channel Estimation](https://arxiv.org/abs/2609.11211) | 本文提出X-RACE框架，通过低复杂度的一次性双重优化策略同时剪除LSTM信道估计模型中无关的输入子载波和隐藏单元，并提出饱和时间、重要性漂移和相关性对比等新颖时序XAI指标来表征模型的学习动态与记忆收敛特性。 |
| [^100] | [CryptoL: Towards Scale Dominance and Physics Constraints Mitigation in Financial Multivariate Time Series Forecasting](https://arxiv.org/abs/2609.11206) | CryptoL是一个统一的加密货币多变量时间序列预测框架，其核心创新在于在RevIN流程的上下文归一化坐标系中评估预测误差，从理论和实证上消除极端规模差异导致的大规模资产对共享模型优化的不成比例影响，并证明共享的通道依赖仿射变换更适合OHLC数据的归一化。 |
| [^101] | [An AI-Powered Culturally Aware Chatbot for Stress Detection and Wellness Support among Pakistani University Students Using NLP and Machine Learning](https://arxiv.org/abs/2609.11199) | 本文提出了一款专为巴基斯坦大学生设计的文化感知AI聊天机器人，利用随机森林模型以89.09%的准确率实现三级压力检测，并结合开源大语言模型提供心理健康支持。 |
| [^102] | [(Whose defaults?) Is artificial intelligence reorienting archaeological methods?](https://arxiv.org/abs/2609.11198) | 该研究分析了约11.9万篇考古学摘要并运用贝叶斯狄利克雷-多项式模型，发现大语言模型兴起后（2023年后）考古学计算方法仅有微小转变，且方法多样性不降反升，表明AI并未窄化学科的研究方法范围。 |
| [^103] | [Agentic Share-of-Search: A Multi-Agent AI System for Competitive Decision-Making in LLM-Mediated E-Commerce](https://arxiv.org/abs/2609.11190) | 本文提出一个多智能体AI系统，以“智能体化搜索份额”为决策目标，自动化测量卖家在AI购物助手中的竞争可见性并诊断根因，实验表明其诊断能力显著优于随机水平。 |
| [^104] | [Can LLMs Follow Medical Expert Logic? A Benchmark for Hierarchical Logical Consistency in Risk-of-Bias Assessment](https://arxiv.org/abs/2609.11185) | 该论文提出了基于Cochrane偏倚风险评估专家逻辑的LogiMed-RoB基准，揭示了大语言模型存在灾难性的错误复合效应——即使单步逻辑一致性高达98.88%，端到端一致性也会骤降至45%甚至接近0%，且模型即使检索到高质量证据也无法正确推理。 |
| [^105] | [Exploring Second-Order Pattern Recognition in Speaker Recognition](https://arxiv.org/abs/2609.11182) | 该论文提出利用层次聚类来发现说话人识别网络中将话语识别为说话人身份时潜藏的“二阶模式”，并通过HCCM方法对其语义解释，进而提出“二阶模式识别”这一新任务。 |
| [^106] | [SemVerBench: Benchmarking LLM Comprehension of Version-Constraint Resolution Semantics](https://arxiv.org/abs/2609.11180) | 本文提出首个跨 npm、PEP 440 和 Cargo 三个生态系统的版本约束解析语义基准测试 SemVerBench，通过对 240 个机器可验证项目的评估，发现六个前沿大语言模型在版本约束语义处理上存在系统性盲点，如 Cargo 部分比较器进位规则使所有模型准确率降至约 60%。 |
| [^107] | [Debate-to-Skill: Capability-Bound Process Supervision for Industrial Query-to-Agent Annotation](https://arxiv.org/abs/2609.11176) | 该论文提出 Debate-to-Skill 方法，将工业查询到智能体标注形式化为能力边界过程监督，通过可复用决策原则、结构化审议、验证器裁定提取和分歧驱动优化，解决了语义相关性与可执行能力相混淆的问题，尤其提升了长尾和灰色地带请求的标注效果。 |
| [^108] | [Beyond Visual Quality: Evaluating Physical Consistency under Ego-Motion with EgoGenEval](https://arxiv.org/abs/2609.11172) | 本文提出基于几何、无需位姿的基准EgoGenEval，用于评估视觉生成器在自我运动下的物理一致性，并通过大规模实验揭示当前模型难以在执行相机运动的同时保持场景状态，没有系统在两方面均表现良好。 |
| [^109] | [Breaking Predictions Is Not Enough: Specified-Foil Counterfactuals for Temporal Graphs](https://arxiv.org/abs/2609.11170) | 该论文提出“指定替代结果反事实”新范式，通过对比原始预测与目标替代预测的执行轨迹差异来生成定向的低成本干预操作，使时序图预测器输出用户预先指定的特定结果，而不仅仅是破坏原有预测。 |
| [^110] | [DRG-MAPPO: Hierarchical Dynamic Role-Graph Multi-Agent Reinforcement Learning for Cooperative Air Combat](https://arxiv.org/abs/2609.11155) | 提出DRG-MAPPO框架，通过将基于图的关系建模与动态角色分配相结合，解决了协同空战中战场实体交互建模缺失和战术角色分配模糊两大难题。 |
| [^111] | [terms.txt: A Consent and Compensation Protocol for Agentic Web Access](https://arxiv.org/abs/2609.11152) | 提出terms.txt协议，通过robots.txt风格的机器可读文件与源站强制执行的交换机制（涵盖Web Bot Auth身份验证、签名意图、HTTP 402定价协商和签名回执），为AI代理访问网络建立按路径、按用途的授权与补偿条款，且实现开销极低。 |
| [^112] | [A Fragility Spectrum for Recursive Language-Model Training](https://arxiv.org/abs/2609.11149) | 该研究让13个公开模型在固定递归污染协议下共享语料库繁衍五代，发现不同模型对坍塌的脆弱性存在约五倍差异，且该脆弱性排序在不同数据组成和随机种子下高度稳定，表明易坍塌性是模型本身的固有属性。 |
| [^113] | [Autonomous Chemical Mechanistic Discovery through Agentic Reasoning and Validation](https://arxiv.org/abs/2609.11147) | 本文提出ARCHE自主智能体系统，通过整合通用推理模型、专用计算化学模型与工具注册表，实现了从假设生成到自我验证的化学反应机理自动化发现闭环。 |
| [^114] | [The Oligarch Barely Steers Model Collapse in Multi-Model Ecosystems](https://arxiv.org/abs/2609.11146) | 在多模型递归训练的生态系统中，即使将寡头模型的市场份额推高至90%，既不会加速模型崩溃，也不会使其他模型被拖向寡头的输出分布——模型崩溃的动态对市场份额集中度表现出不变性。 |
| [^115] | [Same Day, Same Story; One Day Ahead, a Different Signal: The Dual Validity of Financial Sentiment](https://arxiv.org/abs/2609.11144) | 本文基于2002-2025年证券集体诉讼语料库，将70,500条X消息与异常股票收益相关联，通过统一流程测试五种情感分析工具，发现金融情感工具的人工标注一致性（构念效度）与其市场预测能力（预测效度）之间的关系并非恒定，而是取决于抽样惯例和分数表示方式。 |
| [^116] | [KuaiRP Series Role-playing Models Technical Report](https://arxiv.org/abs/2609.11127) | KuaiRP系列角色扮演模型通过标准化角色模板、基于用户行为模拟的SFT数据流水线、规则复合奖励的强化学习以及多阶段训练流程，在注入深度领域知识的同时有效克服灾难性遗忘，实现了小参数规模下高质量、稳定且高效的角色扮演。 |
| [^117] | [Beyond Benchmarks: Using VLMs to Reveal Systematic Classification Failures Under Real World Conditions](https://arxiv.org/abs/2609.11126) | 本研究提出一种基于视觉语言模型的错误切片检测方法，能够自动对分类模型的系统性错误进行分组和标记，从而加速国防应用中模型验证与确认的过程。 |
| [^118] | [Benchmark Radar: A Living Database and Search Engine for AI Benchmarks and Evaluation](https://arxiv.org/abs/2609.11115) | 本文提出Benchmark Radar，一个每日自动发现并整合AI基准论文、数据集和代码的动态数据库与搜索引擎，为LLM评估、智能体、编程、推理和安全等领域提供可检索的基准目录、来源引用和分数历史。 |
| [^119] | [How AI Coders Discuss, Disagree, and Reach Consensus: Challenges and Opportunities for LLM-Based Qualitative Coding](https://arxiv.org/abs/2609.11109) | 该研究构建了一个让多个大语言模型智能体独立编码、辩论并调和分歧的定性编码流程，发现编码准确性受编码手册长度、数据相似度和智能体分歧程度影响，其中激烈且未解决的辩论反而能提高准确性，但大语言模型虽能模拟人类讨论行为却缺乏对情境的适应性响应。 |
| [^120] | [Less can be More: What Aspects of Speech Drive End-of-Turn Detection](https://arxiv.org/abs/2609.11066) | 通过对声学、韵律和语义信号的受控消融实验发现，话轮结束检测主要依靠语调和静音等声韵特征而非语义完整性，仅用声学与韵律组合的轻量级分类器即可在准确率与延迟之间取得最佳平衡，加入文本反而会增加过早检测。 |
| [^121] | [MOSAIC: Query-Aware Exploration Policy Adaptation for GraphRAG](https://arxiv.org/abs/2609.11065) | MOSAIC是一个无需训练的框架，通过LLM分析器将每个查询的证据需求转化为定制的图探索策略（涵盖种子选择、遍历、停止和证据选择），使GraphRAG检索能够根据问题类型灵活适配，在GraphRAG-Bench上显著提升了答案正确率。 |
| [^122] | [Fork Where the Model Changes Its Mind: Belief-Shift Branching for Tree-Structured Reinforcement Learning](https://arxiv.org/abs/2609.11061) | 提出信念偏移分支方法，通过在候选边界处读取模型的答案信念、并在连续信念发生分歧的步骤前进行分叉，从而定位价值曲线的转折点，大幅提升无批评器步级强化学习的信用分配效率。 |
| [^123] | [Grounding Agent Memory: Environment-Probing Curation for Enterprise Agents](https://arxiv.org/abs/2609.11060) | 该论文提出“环境探测式记忆管理”方法，为智能体的异步记忆管理器赋予最小权限的只读环境工具以验证、限定和刷新候选记忆，无需重训模型即可将CLBench通过率从39%提升至73%。 |
| [^124] | [T1: Terminal Agent Reinforcement Learning for Long-Horizon Tasks](https://arxiv.org/abs/2609.11042) | 该论文提出T1——一个通过强化学习训练的122B混合专家模型，能在云端沙箱中操作真实shell执行多达300余次工具调用的长程终端任务，并给出了包含激进热启动、TITO构建与rollout路由重放等稳定优化技术以及分布外训练语料的完整训练方案。 |
| [^125] | [Toward Interpretable Multimodal Fusion: Heat Conduction Modeling for Hyperspectral and LiDAR Joint Classification](https://arxiv.org/abs/2609.11040) | 提出受热传导物理原理启发的M2Heat框架，通过视觉热传导模块和跨频率融合策略，以低于二次方的复杂度实现高光谱与激光雷达数据可解释、高效的多模态联合分类。 |
| [^126] | [The Agent Incident Registry: Toward Preventing Repeated AI Agent Failures](https://arxiv.org/abs/2609.11030) | 本文提出了代理事件登记册（AIR），这是一个与来源关联、带缺失感知标准化标签的AI代理事件目录，可支持公开失败案例与代理安全评估的系统性比较，并发现实际危害主要集中于真实环境和安全失败类记录。 |
| [^127] | [BenchShield: Formal Model-Backed Instrumentation for Reward Integrity in LLM-Agent Evaluation Infrastructure](https://arxiv.org/abs/2609.11028) | 本文提出BenchShield，一个基于奖励相关事件有限生命周期形式化模型的检测工具层，通过静态与动态两种互补分析在基准测试基础设施内保障LLM智能体评估的奖励完整性，防范奖励劫持。 |
| [^128] | [New Evidence, Same Choice: Testing Physical Experiment Selection in Vision Language Models](https://arxiv.org/abs/2609.11022) | 该论文提出了一个受控评估框架，测试视觉语言模型在物理推理中能否判断何时应直接作答、何时需要额外测量以及选择哪个实验，弥补了现有基准只评估最终答案而忽视决策能力的不足。 |
| [^129] | [Defining AI Agents: A Compendium of Criteria, Metrics, and Benchmarks](https://arxiv.org/abs/2609.11018) | 该论文提出了评估AI智能体的五个维度框架（环境交互、学习与适应、自主性、目标导向行为、时间连贯性），系统梳理了各维度相关的指标、基准与评估方法，并发布了公开的Agent Compendium数字资源，以解决智能体定义模糊导致的评估和比较难题。 |
| [^130] | [Topological Necessities: Mechanism-Invariant Strategic Subgoals for Cross-Embodiment Goal-Conditioned Control](https://arxiv.org/abs/2609.11014) | 该论文提出“拓扑必然性”这一新概念，通过在成功离线轨迹构建的载体上运用0维和1维同调，提取带有证书的不可跳过阶段与路径选择点，构建机制不变、可跨本体通用的递归拓扑门层级，作为长时程目标条件控制中的策略子目标。 |
| [^131] | [DeFiFusion: Combining Transaction Events with Smart Contracts to Detect Price Manipulation Attacks](https://arxiv.org/abs/2609.11008) | DeFiFusion是一个双模态检测框架，通过联合建模交易事件与智能合约执行语义来检测DeFi中的价格操纵攻击，克服了单纯基于交易或静态合约分析方法各自的根本局限。 |
| [^132] | [Importance Weighting for Unlabeled-unlabeled Learning under Distribution Shift](https://arxiv.org/abs/2609.10994) | 本文提出了一种基于重要性加权的UU学习分布偏移自适应方法，通过有原则的重要性权重估计，利用测试分布中的少量UU数据来最小化测试风险，从而解决训练与测试分布不一致的问题。 |
| [^133] | [Demystifying the Privacy-Utility Trade-off in LLM Interactions](https://arxiv.org/abs/2609.10992) | 该论文系统性解构了大语言模型交互中的隐私-效用权衡，揭示了决定“何时脱敏”的上下文依赖效用、决定“如何脱敏”的策略性适应以及组合交互作用这三种潜在机制。 |
| [^134] | [A Mathematical Theory of Pragmatic Information](https://arxiv.org/abs/2609.10986) | 该论文提出了一个统一通信、控制与决策的语用信息理论，通过同终点映射建立语法—语义—语用三层信息层次，推广了香农编码定理，并提出语用价值与语用成本的拉格朗日对偶框架以实现跨层优化。 |
| [^135] | [EGGROLL, Unrolled: Understanding and Improving Low-Rank Evolution Strategies at Scale](https://arxiv.org/abs/2609.10980) | 本文首次从理论上刻画了面向大语言模型的低秩进化策略EGGROLL的更新场，揭示其可能引入非保守分量并逆转最优点的局部稳定性，同时证明了该方法的二次目标精确性并给出非渐近误差界，为理解与改进该方法奠定理论基础。 |
| [^136] | [Decoupling Readiness from Release for Tail-Aware Scheduling of Agentic LLM Workflows](https://arxiv.org/abs/2609.10964) | 本文提出一种尾风险感知的智能体LLM工作流轮次释放调度方法，通过均值-CVaR目标联合决策释放时机与未完成工作量预算，将轮次就绪与释放解耦以降低尾延迟。 |
| [^137] | [What a Random Draw from the MCP Registry Contains, and What Tool-Use Benchmarks Contain Instead](https://arxiv.org/abs/2609.10962) | 该研究通过对MCP注册表进行可复现的随机概率抽样，首次揭示了真实服务器生态中近半数服务器根本无法启动、安全注释遗漏率高达58.8%，从而证明现行工具使用基准测试所依赖的人工精选样本会系统性高估生态系统的实际可用性与安全性。 |
| [^138] | [Robust Multimodal Sentiment Analysis with Incomplete Modalities via Semantic-aware Completeness based Reconstruction](https://arxiv.org/abs/2609.10950) | 该论文提出了一种语义感知的完整性估计方法与稳定的多任务训练策略，通过重建模态缺失的语义信息，显著提升了模态不完整场景下多模态情感分析的鲁棒性和预测精度。 |
| [^139] | [Evaluating Scaffolding-Oriented Multi-Agent Large Language Model System for Clinical Interview Training](https://arxiv.org/abs/2609.10939) | 该研究构建了由病人、导师和评估三个智能体组成的多智能体大语言模型标准化病人训练平台，并通过100名医学生的随机对照试验验证了其在可扩展的临床问诊训练中的支架式教学价值。 |
| [^140] | [AUC Maximization from Biased Positive-unlabeled Data with Confidence](https://arxiv.org/abs/2609.10928) | 提出了一种利用少量已标注正类数据所附带的置信度信息（即实例为正类的概率），从有偏正未标注数据中实现AUC最大化的新方法，突破了现有方法要求正类数据必须无偏的理想化假设。 |
| [^141] | [ReactHuman: A Physics-Grounded Benchmark for Human-Like Reactive Decision-Making in Embodied Multimodal LLMs](https://arxiv.org/abs/2609.10895) | ReactHuman是首个基于物理的类人反应式决策基准，通过240 Hz刚体仿真提供精确真值，评估多模态大语言模型在面对突发家庭危险时能否将物理理解转化为即时、安全攸关的行动。 |
| [^142] | [Does Linguistic Structure Enrichment Enhance Coherence Assessment? Not With Current Architectures](https://arxiv.org/abs/2609.10893) | 研究发现，由于附加信息与当前语言模型架构存在结构和句法上的不兼容性，用句法和修辞信息增强文本并不能提升连贯性评估效果，而文本连贯性可作为检测虚假信息的代理指标。 |
| [^143] | [DriftNet: A Dual-Head Trajectory Transformer for Detecting and Localizing Prompt Injection in LLM Agents](https://arxiv.org/abs/2609.10892) | DriftNet是一个双头轨迹Transformer，只需一次前向传播即可同时判断LLM智能体的工具调用轨迹是否被提示注入攻陷，并对每个步骤进行精细标注（良性、注入点、被劫持或注入失败），首次实现了检测结果与攻击定位的联合输出。 |
| [^144] | [Story Imprinting: AI Assistants Absorb Traits from Human Characters They Resemble](https://arxiv.org/abs/2609.10883) | 研究发现，在合成故事上微调会使AI助手“烙印”上与其相似的人类角色的条件性行为和隐含偏好，即使这些内容在训练数据中占比不到2%或从未被明确表达。 |
| [^145] | [When Validation Stops Learning: Auditing Update Admission for Continual Embodied Agents](https://arxiv.org/abs/2609.10873) | 论文提出持续具身智能体的更新准入审计必须同时考量误差控制与保留的学习机会，并指出基于区间的置信度门控在充足预算下无法认证旧任务行为不变，而配对二项检验能在同等预算下接纳31.6%的更新流。 |
| [^146] | [No-Box Vulnerability Analysis: Description-only Detection of Indirect Prompt Injection Vulnerabilities in MCP Servers](https://arxiv.org/abs/2609.10854) | 本文提出“无盒漏洞分析”新范式，仅凭功能描述元数据即可在不访问或不与目标系统交互的情况下，假设性检测MCP服务器中所有可能实现里的间接提示注入漏洞。 |
| [^147] | [Are We Really Doing Few-Shot Learning? A Critical Examination of Pre-Training Assumptions](https://arxiv.org/abs/2609.10851) | 本文通过系统比较四种预训练协议，揭示了当前少样本学习评估中因同域预训练带来的 9.66 个百分点乐观偏差，质疑现有评估方式能否真正反映模型的低数据学习能力。 |
| [^148] | [Studying Without a Syllabus: Task-Agnostic Environment Preprocessing](https://arxiv.org/abs/2609.10824) | 该论文形式化了“任务无关的环境预处理”问题，证明LLM智能体可以在不知道下游任务分布的情况下，在测试前自主探索陌生环境并构建可复用工件来辅助冻结的求解器。 |
| [^149] | [Tapes Together Strong: The Co-evolution of Computation and Cooperation](https://arxiv.org/abs/2609.10817) | 该论文提出“自创生博弈论”这一新计算模型，将社会互动、复制机制及其计算成本内生化并协同演化，并通过Z80机器码程序的实验证明，把社会困境直接嵌入计算的物理机制中能够促进自复制合作策略的涌现。 |
| [^150] | [Counterfactual Marginalisation: Framework for Evaluating Robustness to Nuisance Variables](https://arxiv.org/abs/2609.10778) | 提出反事实边缘化测试时评估框架，通过对年龄、性别等干扰变量进行反事实干预并平均预测，在保留患者个体信息的同时，定量评估分类模型对人口统计学捷径的鲁棒性。 |
| [^151] | [Beyond Static Guarantees: Measuring the Static-Pass Dynamic-Fail Gap in Security-Sensitive and LLM-Generated Python Code](https://arxiv.org/abs/2609.10762) | 该论文首次提出“静态通过-动态失败”（SPDF）现象，并设计了一个融合静态扫描、LLM驱动CWE推理与隔离容器内自主漏洞利用验证的三阶段智能体流水线，以量化静态分析通过但代码在运行时仍可被利用的安全评估盲区。 |
| [^152] | [Multilingual in Name Only? Cultural and Linguistic Weaknesses of LLMs in Urdu](https://arxiv.org/abs/2609.10758) | 本文通过构建包含93个AI生成乌尔都语故事的语料库并按九类语言、语义和文化错误进行人工标注，揭示了多语言大型语言模型在低资源语言故事生成中存在基本语法错误、缺乏连贯性和普遍文化浅薄等严重缺陷，且少样本提示无法有效解决文化错误问题。 |
| [^153] | [Adaptive Margin Ordinal Loss: Penalizing Center-Class Hedging in Ordinal Classification](https://arxiv.org/abs/2609.10752) | 该论文提出自适应边距序数损失（AMOL），通过对每个类别的损失项施加自适应乘法权重，在预测偏向中心类而真实标签远离中心时加大惩罚，从而直接抑制序数分类中的中心类对冲现象。 |
| [^154] | [When Synthetic Data Hurts: On Catastrophic Forgetting in Skill Retrieval for LLM Agents](https://arxiv.org/abs/2609.10750) | 研究发现合成数据微调虽能提升LLM智能体的分布内技能检索效果，但会导致对真实和分布外数据的灾难性遗忘，而借鉴持续学习的微调方法（如LwF、EWC等）既能缓解遗忘，又能将分布内检索性能提升13.98%。 |
| [^155] | [Temporal and Multimodal Deep Learning for Cyberattack Detection in LEO Satellite Systems](https://arxiv.org/abs/2609.10746) | 本文基于卫星专用UNSW-IoTSAT数据集，系统研究了能够融合硬件、轨道与射频多模态信息并捕捉时序攻击行为的深度学习架构（如子系统融合MLP和分层多模态Transformer），以提升低轨卫星系统的网络攻击检测能力。 |
| [^156] | [CARTS: Contextual Autoregressive Rank Transcoding Steganography for Full-Capacity Keyed Text Encoding](https://arxiv.org/abs/2609.10744) | 本文首次对上下文自回归秩转码隐写术（CARTS）进行了严格的形式化安全分析，证明了其在确定性模型假设下的精确正确性，并定义和研究了上下文搜索、密钥碰撞、消息含糊化等相关计算问题及其理论关系。 |
| [^157] | [The Truth Was Never Gone: Perfect Aliasing in Compliant-Context Truth Probes](https://arxiv.org/abs/2609.10739) | 论文揭示了真值探测器的“完美混叠”失效机制——当真实汇报与任务既定行为在顺从情境中重合时探测器无法区分二者（二者AUROC恒互补求和为一），并提出在顺从与对立情境的混合数据上拟合的方法，使探测器即使在模型系统性说谎时也能以完美的1.0 AUROC识别真值。 |
| [^158] | [Towards a Deterministic Math Solver for Clinical Language Models](https://arxiv.org/abs/2609.10728) | 本文提出让临床大语言模型不直接进行算术计算，而是生成针对性Python代码交由受限本地执行器作为确定性求解器运行，但在MedCalc-Bench Verified基准上的评估表明，在公式和标准变量均已提供的情况下，这种程序求解接口相比模型直接计算并无可靠优势。 |
| [^159] | [Finishing the Task Is Not Enough: Evaluating Agent Resilience and Considerate Participation under Accumulating Challenge](https://arxiv.org/abs/2609.10724) | 该论文提出以“运营韧性”和“体贴参与”这两个互补维度来评估生成式AI智能体在挑战不断累积的持续部署场景下的表现，并通过120条模拟医疗保健轨迹验证了这一评估框架。 |
| [^160] | [AcFlow: Controlling Text-to-Image Diffusion Transformers via Learned Conditional Activation Flow](https://arxiv.org/abs/2609.10723) | AcFlow是一种推理时控制器，通过学习概念条件的速度场传输中间层图像token激活，在保持基础扩散Transformer冻结的情况下实现对文本到图像生成的连续风格强度控制与不想要概念抑制，并能泛化到训练中未见过的概念。 |
| [^161] | [An Open Recipe for IMO Gold: Training Nemotron for Olympiad Mathematics](https://arxiv.org/abs/2609.10712) | 该研究通过监督微调与强化学习后训练Nemotron 3 Ultra模型，构建了一个纯自然语言、无形式化证明器与外部工具的迭代生成-验证-细化测试时计算流水线，在IMO 2026中获得30/42分达到金牌水平，并开源了模型检查点、训练数据与代码。 |
| [^162] | [Architecting the Secure AI-SOC: A Neurosymbolic Framework for Pipeline Integrity and Threat Mitigation](https://arxiv.org/abs/2609.10707) | 本文提出一种神经符号纵深防御架构，通过确定性SIEM解码器预过滤与神经语义评估相结合的两层机制，抵御针对AI安全运营中心中LLM的日志投毒式间接提示注入攻击，实现管道的端到端完整性保障与威胁缓解。 |
| [^163] | [Data-Efficient Language Modeling: From Frontier Advancement to Principle-Guided Model Improvement](https://arxiv.org/abs/2609.10702) | 该研究通过三阶段自主研究计划，在BabyLM 2026 Strict-Small受限数据设置下发现精确重复与对齐重述产生不同的上下文利用模式，并提出围绕预训练所需上下文依赖来组织经验的数据高效学习原则，实现了从构建前沿模型到原则指导的模型改进。 |
| [^164] | [Quantifying the Memorization-to-Generalization Transition: Scaling Laws and Phase Structure in Grokking](https://arxiv.org/abs/2609.10657) | 本研究通过384组配置的系统实验首次量化了grokking现象中记忆到泛化转变的幂律缩放定律，发现数据复杂度（而非模型容量）是驱动这一相变的主导因素，且在权重衰减λ ≳ 1.0处存在尖锐的相边界。 |
| [^165] | [Understanding LoRA Rank Trade-offs in Diffusion Model Fine-Tuning](https://arxiv.org/abs/2609.10656) | 扩散模型 LoRA 微调研究表明，在固定训练预算下，小到中等的秩（如 4 或 8）能以更低的计算成本达到最佳生成质量，而更高的秩收益有限。 |
| [^166] | [A Multi-Stage Rule-Chaining Framework for Compositional and Interpretable Cognitive Reasoning](https://arxiv.org/abs/2609.10654) | 该论文提出了一种多阶段规则链式框架，通过集成确定性规则发现、模式组合引擎和结构抽象层三个互补求解器，在渐进式回退层次结构中实现组合式且可解释的认知推理，在ARC基准的1000个训练任务中通过了995个。 |
| [^167] | [Automating Quadratic Unconstrained Binary Optimization (QUBO) Formulation Generation from Natural Language](https://arxiv.org/abs/2609.10629) | 本文提出一个端到端多智能体框架，可自动将自然语言问题描述转化为QUBO优化公式，并发布了涵盖12个应用领域、包含100个组合优化问题的QUBOBench基准数据集。 |
| [^168] | [Probabilistic Focal Search: Accelerating Bounded-Suboptimal Search via Lower-Bound Advancement](https://arxiv.org/abs/2609.10584) | 提出概率焦点搜索（PFS），以一定概率扩展最小 $f$ 值节点来推进下界、扩大FOCAL集合，在启发式引导与下界推进之间取得平衡，从而加速有界次优搜索。 |
| [^169] | [A machine-checked proof of the Dong-Yang classification of optimal (n,4) binary codes for BSCs](https://arxiv.org/abs/2609.10579) | 本文利用 AI 工具辅助开发了 Dong-Yang 二进制对称信道最优 (n,4) 二进制码分类定理的 Lean 4 机器检验形式化证明，并在过程中修正了 AI 生成代码的错误、发现了原论文中的不一致之处。 |
| [^170] | [PACE: Perceived-Latency-Aware Cascading Service Routing and Filler Control for QoE-Efficient Retrieval-Augmented Dialogue Serving](https://arxiv.org/abs/2609.10372) | PACE框架将感知首次响应时间（PTFR）作为QoE目标，通过负载自适应级联路由、路径-填充器联合控制和波动感知缓存准入三种机制，在人形机器人对话服务中显著降低感知延迟并保证回答质量与新鲜度。 |
| [^171] | [Why Sample What You Can Enumerate? Exact Policy Optimization for Genomic Tool Selection](https://arxiv.org/abs/2609.10221) | 该论文揭示了在工具子集空间可完全枚举的基因组推理等科学领域中，GRPO等基于采样的强化学习方法存在结构性缺陷（训练越成功奖励信号越稀缺），并提出FGPO（全组策略优化），通过对所有工具子集进行精确枚举评分来替代采样估计。 |
| [^172] | [Beyond Verified Answers: Solver-Informed Self-Distillation for Bootstrapping Operations Research Language Models](https://arxiv.org/abs/2609.09957) | 该论文提出利用模型自身生成解所触发的求解器产物反馈作为监督信号的自蒸馏方法，摆脱对人工验证答案、额外评估器和特权上下文的依赖，实现运筹学语言模型的可扩展自举训练。 |
| [^173] | [Compact Visuotactile World Models for Lifting: Prediction, Reward Alignment, and Force Constraints](https://arxiv.org/abs/2609.09597) | 该研究构建了一个仅65万参数的紧凑型视触觉世界模型，发现准确的触觉预测本身并不能保证力约束控制的提升，但模型辅助力反馈能将同分布任务成功率从73.3%提升至93.3%，而想象强化学习表现反而不如反应式隐式Q学习。 |
| [^174] | [Edu-QuRating: Multi-Dimensional Educational Data Curation with Distilled Pairwise Judgements](https://arxiv.org/abs/2609.09425) | Edu-QuRating提出了一种多维度教育数据筛选流水线，通过定义教育专用评分标准、利用LLM评判器标注文档对并将成对偏好蒸馏为可复用的评分模型，从而突破传统单一标量式教育价值评估的局限，从准确性、吸引力、结构性和受众适用性等多个维度对文本进行精细化评分。 |
| [^175] | [GoAnt: Quality-Diversity Multi-Agent Search for Alpha Factor Discovery in Market Microstructure Data](https://arxiv.org/abs/2609.08719) | 提出GoAnt框架，借助共享“心智地图”与“蚁后”调度器的多智能体质量-多样性搜索，在固定评估预算下发现计入执行成本后依然稳健有效的Alpha因子。 |
| [^176] | [FastE: Readout-Triggered Token Compression for LLM Embedding Inference](https://arxiv.org/abs/2609.08407) | 提出了无需训练、即插即用的FastE方法，利用前缀状态随深度增加而愈发可压缩的发现，通过读出位置注意力分数在线筛选并压缩前缀token，大幅降低LLM嵌入推理的计算成本。 |
| [^177] | [A Multi-Modal Perception Pipeline for Object Detection and Tracking in Autonomous Racing](https://arxiv.org/abs/2609.08338) | 本文提出了一种多模态后融合感知流水线，通过融合摄像头、激光雷达和毫米波雷达的独立检测结果，并结合显式补偿检测延迟、嵌入车辆动力学先验知识的专用多目标跟踪框架，实现了自动驾驶赛车高速恶劣环境下对周围车辆及时且稳健的检测与跟踪。 |
| [^178] | [Your Agent Says Yes: Interpreting Adversarial Market Behavior Beyond Individual Transactions](https://arxiv.org/abs/2609.07675) | 本文通过十个角色化的语言模型智能体在虚拟交易所中的对抗性模拟，揭示了“发行—推广—退出”类市场操纵行为如何分散于消息、交易与时间之中，从而暴露了仅依赖单笔交易审核的风控机制的盲区。 |
| [^179] | [Beyond Single-Negative Preference: Multi-Negative DPO for LLM-Centric Historical Entity Linking](https://arxiv.org/abs/2609.07379) | 提出多负例直接偏好优化方法（MDPO），通过将正确实体与完整候选集进行比较来改进基于大语言模型的历史实体链接，在多语言历史报纸文本上超越监督微调和单负例DPO，尤其在NIL提及、语义歧义和OCR噪声场景下收益显著。 |
| [^180] | [Monadic Second-Order Logic in HOL: Deep and Shallow with Automated Faithfulness (Extended Preprint)](https://arxiv.org/abs/2609.07345) | 本文在 Isabelle/HOL 中将深浅嵌入方法应用于一元二阶逻辑（MSO），借助新型双排序替换装置自动化验证了三种嵌入的忠实性，并首次完全机械化证明了双排序向下 Löwenheim–Skolem 定理。 |
| [^181] | [Ambient @ EgoProactive 2026 : Proactive Egocentric Assistance with Visually Grounded Supervision](https://arxiv.org/abs/2609.07099) | 该论文将可穿戴助手的介入时机判断重新表述为单token二分类任务，并利用工具调用式视频智能体自动生成视觉定位的监督数据，最终在ECCV 2026可穿戴AI挑战赛EgoProactive赛道大型模型组中获得第一名。 |
| [^182] | [Ordinary, Reasonable Chatbots: Do AI Models Track Human Legal Judgments?](https://arxiv.org/abs/2609.06769) | 本研究通过测试大语言模型驱动的聊天机器人对一系列法律合理性问题（即行为是否“合理”）的回答，探究生成式AI模型能否模拟人类在法律场景中的合理性判断。 |
| [^183] | [Reason Through the Latent! Making Latent Visual Reasoning Necessary](https://arxiv.org/abs/2609.06746) | 提出因果视觉循环推理（CVRR）框架，通过在解码前移除视觉状态和多模态KV缓存，迫使循环隐藏状态成为唯一的图像条件信息通路，从而确保潜空间视觉推理真正被模型依赖。 |
| [^184] | [Certifying cooperation: a novel approach to cooperative multi-agent task generation](https://arxiv.org/abs/2609.06586) | 该论文提出利用时间合作图和命题公式编码来认证多智能体任务中合作的必要性与必然性，从而将随机布局采样器转化为能自动生成具有可证明合作要求任务的生成器。 |
| [^185] | [Planning and Scheduling Business Processes under Control-Flow Uncertainty](https://arxiv.org/abs/2609.05578) | 该论文将控制流不确定性下的业务流程规划与调度问题建模为机会约束优化问题，提出了一种两阶段分解方法，在保证流程成功完成可行性的前提下最小化被规划但不会执行的冗余活动数量。 |
| [^186] | [Role differentiation as ignition of a collective information engine: Structuration in Agent Populations](https://arxiv.org/abs/2609.05442) | 该研究设计了一种基于角色分化而非共识的集体信息引擎，通过反协调博弈将结构化理论中“图式与资源二元性”操作化，并揭示了当社会回路增益超过阈值时角色分化便会自发“点燃”。 |
| [^187] | [HarvestBench: Measuring Whether LLM Agents Will Pay to Avoid Killing Animals](https://arxiv.org/abs/2609.04444) | HarvestBench 是首个为“避免杀死动物”这一副作用定价的基准，通过农场收割模拟测试大语言模型智能体是否愿意支付燃料代价绕开挡路的动物而非直接碾压，并以岩石和干草捆作为对照组、以是否偷取邻居作物作为附加的道德测试。 |
| [^188] | [Harbor Adapters and Harbor-Index: Infrastructure and a Curated Meta-Dataset for Large-Scale Agentic Evaluation](https://arxiv.org/abs/2609.04298) | 本文提出了 Harbor Adapters 统一评估基础设施，将 80 多个智能体基准测试移植为可评估任意智能体的形式，并据此对 8 个模型进行大规模评估，同时推出经 AI 与人工双重审核筛选的包含 82 个高质量难题的精选元数据集 Harbor-Index。 |
| [^189] | [SimSkill: A Lifelong Learning AI Agent for Autonomous Mastery of Traffic Simulation](https://arxiv.org/abs/2609.03753) | SimSkill是一个基于SUMO交通仿真器的自进化终身学习AI智能体，通过自主识别能力差距、生成并验证任务、将经验整合到三种记忆系统中，在不更新底层模型的情况下将经验证的任务完成率提升最多25个百分点。 |
| [^190] | [Toward Collective-Centric Evaluation of Preference Inference for Participatory Democracy](https://arxiv.org/abs/2609.02990) | 该论文提出了一个以集体为中心的评估框架，对参与式民主平台中现有偏好推断方法进行了基准测试，揭示了这些方法并非中立，可能人为地放大、抑制或重排集体支持模式，从而重塑对商议结果的解读。 |
| [^191] | [OmegaUse-SOP: SOP Engineering for Professional Computer Use from Human Demonstrations](https://arxiv.org/abs/2609.02149) | 提出了OmegaUse-SOP系统，通过人在回路的SOP工程方法，将专业计算机操作的人类演示迭代式地转化为GUI智能体可复用的SOP技能，从而解决了智能体执行特定领域专业标准操作程序的难题。 |
| [^192] | [Learning What to Retain: Gated-Memory Routing for Efficient Collaboration in Multi-Agent LLM Systems](https://arxiv.org/abs/2609.00237) | 提出门控记忆路由方法，通过可学习的记忆写入门和检索门维护紧凑的执行记忆，使多智能体LLM系统的编排决策能依据有用的中间进展而非完整历史，在提升准确性的同时降低成本。 |
| [^193] | [REAL-Q: E2E LLM Quantization via Dynamic Gradient Descent](https://arxiv.org/abs/2609.00049) | REAL-Q提出了一种打破传统折中的后训练量化新范式，通过端到端对齐的代理损失目标和每128列一次的动态块级梯度下降，解决了现有方法中Hessian矩阵被整层冻结导致的信息错位问题，从而更精确地逼近全局损失实现大语言模型量化。 |
| [^194] | [Motus2: A Self-Evolving General World Model for Dexterous Manipulation](https://arxiv.org/abs/2608.30237) | Motus2 提出了一个自我进化的通用世界模型，让单一共享权重模型同时充当策略、模拟器和评估器三种角色，通过三者耦合形成闭环决策与学习回路，实现灵巧操作策略的自我持续改进。 |
| [^195] | [A Calibration Audit of Confidence in Feed-Forward 3D Reconstruction](https://arxiv.org/abs/2608.29705) | 本文首次系统审计了前馈3D重建模型的逐像素置信度，发现其虽能较好地对误差排序，但作为不确定性估计在偏离训练条件时严重偏低（中位数偏差达2.4倍），且即使损失函数达到最优该问题依然存在。 |
| [^196] | [How Proper Scoring Rules Shape LLM Forecasting](https://arxiv.org/abs/2608.28482) | 尽管五种适当的评分规则在理论上都激励真实概率报告，但作为大语言模型预测的训练目标时，会训练出在校准、偏差、信息和噪声特征上各不相同的模型，表明奖励函数的选择并非可以互换。 |
| [^197] | [FaultLens: Learning Compact Behavioral Test Suites for Generated Operational Programs](https://arxiv.org/abs/2608.26746) | 本文提出FaultLens方法，通过结合故障驱动的贪婪选择和突变无关的多样性组件，学习紧凑的行为测试套件，以高效检测生成程序中的稀疏边界和交互故障。 |
| [^198] | [GameWAM: A World Action Model for Video Games](https://arxiv.org/abs/2608.26200) | GameWAM是首个将世界模型与动作策略统一用于视频游戏原生闭环控制和GUI操作的世界动作模型，通过并行生成视觉与动作轨迹实现动态建模。 |
| [^199] | [CAT-GS: Balanced Multimodal Learning via Calibrated Gating and Fusion Surgery](https://arxiv.org/abs/2608.24947) | 本文提出了一种名为CAT-GS的优化控制器，通过校准门控和融合手术机制，在不修改模型结构的情况下解决多模态学习中的模态不平衡、门控不稳定和融合干扰问题。 |
| [^200] | [Aslema at NADI 2026: Augmentation through Fewshot for SLU](https://arxiv.org/abs/2608.18689) | 本文提出Aslema系统，通过微调优于零样本，并利用大型语言模型生成文化相关的合成数据增强，在NADI 2026任务中在槽位填充上取得第一名。 |
| [^201] | [Improving Natural-Language Combinatorial-Optimization Accuracy in Resource-Constrained Language Models via Formal Abstractions](https://arxiv.org/abs/2608.18409) | SDDL通过神经符号框架将自然语言调度问题转化为求解器友好的表示，显著提升了资源受限语言模型在组合优化中的可行性。 |
| [^202] | [Efficient Adaptation of LLMs for Hate Speech Detection in Low-Resource Languages: A Comparative Study on Roman Urdu](https://arxiv.org/abs/2608.18142) | 本研究通过LoRA参数高效微调方法，系统比较了多种大型语言模型在罗马乌尔都语低资源环境下的仇恨言论检测性能，展示了PEFT在零样本推理中的优势。 |
| [^203] | [Clearing the Fog: Towards Installing and Refining Proactive Exploration Capabilities in LLM Agents](https://arxiv.org/abs/2608.14339) | 本论文提出了一种名为\ours的方法，通过探索性数据构建和对比信号引导的强化学习，有效增强了LLM智能体的主动探索能力，克服了后见之明偏差并区分有效探索与冗余行为。 |
| [^204] | [VALG: An Agentic System for ML Theory Research](https://arxiv.org/abs/2608.13060) | 本文提出了VALG系统，通过多级验证、自适应问题表述和图结构证明开发，实现了机器学习理论研究中开放问题求解的自主智能体工作流程。 |
| [^205] | [NaviDC-OCR: Navigating Document Parsing Across Digital and Camera-Captured Documents](https://arxiv.org/abs/2608.12898) | NaviDC-OCR通过引入变形感知学习和自适应采样机制，统一了数字与相机拍摄文档的解析框架，有效解决了几何变形和结构推理不足的问题。 |
| [^206] | [Terminal Symmetry as a Decision Resource: Statewise Refinement for Anytime Verified Construction](https://arxiv.org/abs/2608.11318) | 本文提出了一种将终端对称性作为决策资源的新框架，通过传输-细化-认证机制实现任意时间验证构建，并提供了完成保证和最优验证器查询界限。 |
| [^207] | [GitSkills: A Dataset of Agent Skills on GitHub](https://arxiv.org/abs/2608.10906) | 本文提出了GitSkills数据集，收录了GitHub上3,797,117个智能体技能文件，为首次实证研究开发者如何编写、复用和维护基于自然语言的智能体技能提供了数据基础。 |
| [^208] | [SPECTRA: Band-Routed Embedding and Stage-Wise LoRA for Cross-Sensor Fine-Tuning of Geospatial Foundation Models](https://arxiv.org/abs/2608.01751) | SPECTRA提出了一种参数高效微调框架，通过波段路由嵌入解决预训练模型与下游传感器之间的光谱失配问题，并通过分阶段LoRA降低地理空间基础模型的微调适配成本。 |
| [^209] | [Sympathetic Framing: Evaluating AI Alignment across Sociodemographic Groups](https://arxiv.org/abs/2607.27232) | 该研究通过对3011名英国成年人的YouGov调查与七个大语言模型的对比实验，首次实证评估了LLMs在感知新闻标题情感框架（对冲突各方的同情）方面与人类的对齐程度，发现领先模型（如GPT-5.2相关性达0.789）在所有人口统计子群体中都与人类情感判断高度一致。 |
| [^210] | [SciFigQual-Bench: A Benchmark for Scientific Figure Quality Assessment with Full-Manuscript Context](https://arxiv.org/abs/2607.27084) | 提出SciFigQual-Bench，首个结合全文语境、从清晰度、布局、图注契合度、上下文相关性和误导风险五个维度评估科学图像质量，并配备多位领域专家黄金标准标注的基准数据集。 |
| [^211] | [A Density-Matrix Framework for Electronic-Structure Analysis of Electrolytes for Lithium Batteries](https://arxiv.org/abs/2607.25597) | 提出EMolStudio——一个以密度矩阵为中心的AI平台，可高效预测和分析锂电池电解质的电子结构，并系统揭示了分子官能化与盐阴离子种类对前线能级、静电势及轨道局域化的调控规律。 |
| [^212] | [ScaleResfusion: Residual Rectified Flow based on Residual Vector Field](https://arxiv.org/abs/2607.25275) | ScaleResfusion 提出残差整流流（RRF），通过将残差项嵌入整流流的线性传输路径，把真实世界图像恢复转化为与噪声调度器无关的适配接口，从而能够复用预训练的文本到图像整流流模型，从带噪低质量图像出发高效恢复高质量图像。 |
| [^213] | [SGA: Plug&Play Geometric Verification for Educational Video Synthesis](https://arxiv.org/abs/2607.18116) | 该论文提出即插即用的符号几何智能体SGA，通过拦截并部分执行LLM生成的Manim代码提取符号场景图以检测和修正空间冲突，并引入无需渲染的MVQS评估指标，使教学动画的空间正确性相对基线提升16.1%。 |
| [^214] | [Natural Language Access to Domain-Specific Metadata: A Reusable Framework for LLM Query Generation](https://arxiv.org/abs/2607.18029) | 该论文提出NLKGQ框架，通过在OWL本体中形式化领域词汇和语义，使LLM能够零样本将自然语言问题转换为准确的SPARQL查询，从而无需微调、检索增强或多智能体编排即可实现对领域特定元数据档案的自然语言访问。 |
| [^215] | [EVOQUANT: Self-Evolving Verifier-Guided Strategy Optimization for Robust Quantitative Trading](https://arxiv.org/abs/2607.12455) | EVOQUANT提出了一个自进化的验证器引导框架，利用大语言模型诊断量化策略瓶颈、生成受控编辑并通过多阶段验证筛选最优策略，同时将优化经验沉淀为可复用知识，实现量化交易策略的持续稳健自我改进。 |
| [^216] | [Verification of Adaptive Agentic Controllers through Finite Rule Revision](https://arxiv.org/abs/2607.09770) | 本文提出了一种将有界可修订对象作为自适应智能体控制器表示的验证协议，通过有限符号规则、显式诊断谓词、解释日志和留出集再评估，在不依赖无限制人工介入判断的情况下实现控制器故障的检测、局部修复或拒绝。 |
| [^217] | [Ceci n'est pas une pipe: AI systems as semantic abstractions](https://arxiv.org/abs/2607.09489) | 该论文提出了一个语义框架，将AI系统的输出视为工程化构建的表示而非事实本身，通过区分公认领域知识、参考来源和系统可用信息，为外推、无依据断言等常见失败模式给出精确定义，从而为规范和检验AI系统的可靠性提供词汇体系。 |
| [^218] | [ProsMAE: Multi-Source MAE Pretraining for ISUP Grade Classification](https://arxiv.org/abs/2607.08162) | 本文提出ProsMAE，一种利用PANDA、CAMELYON17和BRACS三个数据源进行MAE预训练的组织病理学表示学习框架，通过冻结编码器和线性分类头迁移至ISUP分级分类任务，并取得了优于原始MAE的QWK分数。 |
| [^219] | [DiaLLM: An Investigation into the Robustness-Generation Gap in English Dialect Adaptation](https://arxiv.org/abs/2607.07669) | 本文发现方言理解与生成能力在LLM中分离，并证明显式方言定向适应优于广泛对齐，但基准测试无法反映这一生成优势。 |
| [^220] | [Adaptive Perturbation Selection for Contrastive Audio Decoding](https://arxiv.org/abs/2607.00247) | 该论文提出针对对比解码，通过对多样化音频扰动库进行评估并为每个任务和样本自适应选择最优负分支，来缓解大型音频-语言模型的幻觉问题，例如将时间顺序任务准确率从74.7%提升至81.4%。 |
| [^221] | [Relevance Is Not Permission: Localizing and Controlling Metric-Facing Attention Contributions](https://arxiv.org/abs/2606.30139) | 提出Warrant统一方法，通过暴露通向评估指标的逐项注意力贡献路径并施加查询条件化的许可控制，揭示“注意力相关性不等于预测贡献”（最高注意力项在约一半样本中反而损害效用），并在五类任务的32组对比中有27组提升了主要指标。 |
| [^222] | [LLM-Ideoplasticity: Measuring Ideological Plasticity in the Political Behavior of LLMs as a Context-Conditioned Distribution](https://arxiv.org/abs/2606.28335) | 大语言模型的政治意识形态不是固定点，而是随语境变化的条件分布——虽然对说服性框架、语言选择等因素高度敏感，但整体上占据的政治光谱范围仅为欧洲主要政党的约三分之一。 |
| [^223] | [Cognitive Digital Twins: Ethical Risks and Governance for AI Systems That Model the Mind](https://arxiv.org/abs/2606.23094) | 本文定义了认知数字孪生（CDTs）这一新型AI技术，指出其独特伦理风险，并提出了涵盖权威、自主性、访问与控制、问责制和可用性五个维度的5A治理框架。 |
| [^224] | [AI Economist Agent: An Agentic Framework for Evidence-Based Economic and Financial Analysis with RAG, Knowledge Graphs, and Large Language Models](https://arxiv.org/abs/2606.20041) | 该论文提出了一个结合RAG、知识图谱和大语言模型的AI经济学家智能体框架，由LLM智能体负责规划分析、检索证据和组织经济机制，由注册的定量模型生成数值结果并通过预定义检验保证可靠性，实现了面向欧洲宏观金融压力测试的循证经济与金融情景分析。 |
| [^225] | [Refusal Beyond a Single Direction: A Preliminary Comparison of Diff-in-Means and INLP](https://arxiv.org/abs/2606.13720) | 本研究在五个开源聊天模型上比较了差值均值法（DiM）与迭代零空间投影（INLP）两种干预方法在引导模型拒绝行为上的效果，发现INLP反事实翻转在拒绝抑制方面与DiM方向消融相当，且对层选择更具鲁棒性。 |
| [^226] | [A Lightweight Multi-Agent Framework for Automated Concrete Barrier Design](https://arxiv.org/abs/2606.12040) | 该论文提出了一种基于AutoGen多智能体编排的“生成-验证-修改”闭环框架，通过多智能体协同克服大语言模型在结构设计中的幻觉与数值推理缺陷，实现符合AASHTO规范的钢筋混凝土公路护栏自动化设计。 |
| [^227] | [Some hypotheses on how chatbots work in problem-solution-driven conversations: Large Language Models as confirmation of the Innovation Illusion](https://arxiv.org/abs/2606.07722) | 该论文从聚合动力学、认知语言学、神经心理学和心理学等多学科视角分析聊天机器人在问题解决型对话中的本质，提出人类的想象与思考基于隐喻性问题传播，而大语言模型的训练文本仅能部分模仿人类思维，从而印证了“创新幻觉”。 |
| [^228] | [From Agent Traces to Trust: A Survey of Evidence Tracing and Execution Provenance in LLM Agents](https://arxiv.org/abs/2606.04990) | 本综述提出将执行溯源形式化为智能体执行的有类型图、将证据追踪视为其在证据支持关系上的投影，以此为基础构建可信LLM智能体的过程级问责框架。 |
| [^229] | [CLSP-REQA: A Real-Time Quality-Aware Closed-Loop Seizure Prediction Framework with Mamba-BiLSTM and Confidence-Gated Intervention](https://arxiv.org/abs/2606.00074) | 该论文提出了CLSP-REQA闭环癫痫发作预测框架，通过将实时脑电信号质量评估模块与Mamba-BiLSTM预测骨干并行结合，利用质量分数经分级非线性融合函数调制输出置信度，在严格的跨患者评估下显著提升了预测的可靠性。 |
| [^230] | [Causal Past Logic for Runtime Verification of Distributed LLM Agent Workflows](https://arxiv.org/abs/2605.20923) | 该论文提出因果过去时逻辑（CPL）扩展ZipperGen框架，使分布式LLM智能体工作流中的守卫条件能够仅基于因果可见的事件进行在线运行时验证，并证明了本地监控器值与守卫指称语义的一致性。 |
| [^231] | [Continuous Diffusion Scales Competitively with Discrete Diffusion for Language](https://arxiv.org/abs/2605.18530) | 该研究通过将 Plaid 与现代离散扩散语言模型架构对齐构建了 RePlaid，建立了首个可媲美离散 DLM 的连续扩散语言模型缩放定律，证明连续扩散是语言建模中高度竞争且可扩展的方案。 |
| [^232] | [Monotone Neural Policy Iteration for High-Dimensional First-Order Hamilton--Jacobi--Bellman Equations](https://arxiv.org/abs/2605.07116) | 本文提出一种单调神经策略迭代方法，通过中心差分构造单调算子并借助一致化技术将冻结策略算子转化为马尔可夫链生成器，从而在无需张量网格的情况下为高维一阶HJB方程提供了全空间适定性、显式依赖域界和考虑模型误差的后验评估界。 |
| [^233] | [Analyzing LLM Reasoning to Uncover Mental Health Stigma](https://arxiv.org/abs/2604.25053) | 该论文提出通过分析大语言模型的中间推理步骤，并借助临床专业知识对污名化语言进行分类与严重程度评级，从而揭示多项选择题评估方法无法捕捉的心理健康污名化偏见及其内在逻辑。 |
| [^234] | [Wiggle and Go! System Identification for Zero-Shot Dynamic Rope Manipulation](https://arxiv.org/abs/2604.22102) | 提出"摇动即走"两阶段框架，仅需观察一次短暂安全的摇动动作即可辨识绳索系统参数，无需大量真实数据或迭代优化，即可实现零样本的动态绳索目标击打与多目标抛掷悬挂操作。 |
| [^235] | [EvoMaster: A Foundational Evolving Agent Framework for Agentic Science at Scale](https://arxiv.org/abs/2604.17406) | EvoMaster 提出通过执行、探索、进化三个嵌套循环实现外部证据持久化与经验跨研究积累的基础性进化智能体框架，解决了智能体科学中的循环碎片化与循环不连续问题，在十个基准测试中以 58.02% 的平均得分取得最佳成绩。 |
| [^236] | [Learning to Think Like a Cartoon Captionist: Incongruity-Resolution Supervision for Multimodal Humor Understanding](https://arxiv.org/abs/2604.15210) | 提出IRS框架，将多模态幽默理解分解为不协调性建模、消解建模和偏好对齐三个环节，通过结构化推理轨迹监督，使模型像专业漫画配文作者一样进行显式推理。 |
| [^237] | [Towards Automated Solar Panel Integrity: Hybrid Deep Feature Extraction for Advanced Surface Defect Identification](https://arxiv.org/abs/2604.10969) | 该论文提出了一种结合LBP、HoG、Gabor滤波器等手工特征与DenseNet-169深度特征的混合特征提取方法，用于太阳能电池板表面缺陷的自动化智能检测。 |
| [^238] | [VeriSim: A Configurable Framework for Stress-Testing Medical AI Under Patient Communication Noise](https://arxiv.org/abs/2604.10441) | VeriSim框架通过在六个临床沟通维度上注入可控噪声来模拟真实患者沟通方式，揭示了医疗大语言模型在噪声干扰下诊断准确率下降15-25个百分点、且小参数模型退化更为严重的脆弱性。 |
| [^239] | [How LLMs Follow Instructions: Skillful Coordination, Not a Universal Mechanism](https://arxiv.org/abs/2604.06015) | 研究通过探测与消融实验证明，大语言模型的指令遵循能力并非依赖单一通用机制，而是通过部分共享、按技能相似性聚类组合的结构化表征进行技能化协调来实现的。 |
| [^240] | [Four Generations of Quantum Biomedical Sensors](https://arxiv.org/abs/2603.29944) | 本文提出了一个基于量子资源利用的四代量子生物医学传感器统一分类框架，并创新性地定义了将量子传感与量子学习及变分电路端到端集成、可在量子域内直接实现自适应推理的第四代量子传感器。 |
| [^241] | [HISA: Efficient Hierarchical Indexing for Fine-Grained Sparse Attention](https://arxiv.org/abs/2603.28458) | HISA提出了一种即插即用的两阶段分层索引方法，先通过块级粗过滤丢弃无关区域，再在候选块内进行token级精炼，从而消除细粒度稀疏注意力中索引器随上下文长度增长的每层扫描瓶颈。 |
| [^242] | [Automated multi-class wound assessment using dedicated instance segmentation models for boundary detection and classification](https://arxiv.org/abs/2603.27325) | 本研究基于YOLOv11开发了两个专用实例分割模型，能够对烧伤、压力性损伤、糖尿病足溃疡等五种临床相关伤口类型同时实现边界分割与分类，显著提升了AI伤口评估的临床适用性。 |
| [^243] | [Perturbation: A simple and efficient adversarial tracer for representation learning in language models](https://arxiv.org/abs/2603.23821) | 该论文提出了一种简单高效的对抗性扰动方法来追踪语言模型的表示学习，通过在单个对抗样本上微调模型并测量扰动对其他样本的“感染”程度来揭示表示，该方法无需几何假设，且能避免在未训练模型中产生虚假表示。 |
| [^244] | [Domain Elastic Transform: Bayesian Function Registration for High-Dimensional Scientific Data](https://arxiv.org/abs/2603.21235) | 该论文提出域弹性变换（DET），一种无网格的贝叶斯概率框架，通过联合空间-函数似然引导的弹性变形建模，在完全无监督的条件下直接对齐不规则稀疏流形上高维科学数据（如空间转录组学基因表达）的几何与功能信号，无需分箱或体素化处理。 |
| [^245] | [OpenResearcher: A Fully Open Pipeline for Long-Horizon Deep Research Trajectory Synthesis](https://arxiv.org/abs/2603.20278) | OpenResearcher提出了一个完全开源、可复现的离线轨迹合成流水线，在1500万文档语料库上合成超过97K条长程深度研究轨迹，微调后的模型在BrowseComp-Plus上相比基础模型提升34个百分点。 |
| [^246] | [Cognitive Amplification vs Cognitive Delegation in Human-AI Systems: A Metric Framework](https://arxiv.org/abs/2603.18677) | 本文提出了一个包含四个量化指标（认知放大指数、依赖比率、人类依赖指数、认知漂移率）的度量框架，用以区分人机系统中的“认知放大”与“认知委托”，并通过基于智能体的仿真验证了该框架识别正向协作增益是否可恢复的有效性。 |
| [^247] | [TRUST-SQL: Tool-Integrated Multi-Turn Reinforcement Learning for Text-to-SQL over Unknown Schemas](https://arxiv.org/abs/2603.16448) | 该论文提出TRUST-SQL框架，将未知模式下的文本到SQL任务形式化为部分可观测马尔可夫决策过程，通过结构化四阶段协议和双轨GRPO强化学习策略解决信用分配问题，使智能体能够主动识别并验证相关数据库模式，取得9.9%的相对性能提升。 |
| [^248] | [An Agentic Evaluation Framework for AI-Generated Scientific Code in PETSc](https://arxiv.org/abs/2603.15976) | 提出PETSCAgent-Bench——一个包含14个评估器、覆盖正确性、性能、代码质量、算法适用性和库惯例五个维度的智能体评估框架，用于评估AI生成的科学代码能否像专家一样使用PETSc等生产级HPC库。 |
| [^249] | [Rescaling Confidence: What Scale Design Reveals About LLM Metacognition](https://arxiv.org/abs/2603.09309) | 大语言模型的置信度量表设计并非中立选择——研究发现0–20量表能持续提升元认知敏感性，而模型对整数值存在强烈偏好，表明量表设计直接影响LLM不确定性估计的质量。 |
| [^250] | [MOSAIC: A Universal Agent-Level Interface for Cross-Paradigm Agent Mixing and Human-AI Collaboration](https://arxiv.org/abs/2603.01260) | MOSAIC是一个开源平台，通过基于IPC的工作器协议和统一的操作员抽象接口，使强化学习策略、大语言模型、视觉语言模型和人类操作员等异构智能体能够在同一强化学习环境中协作并实现公平的跨范式比较。 |
| [^251] | [Probing for Knowledge Attribution in Large Language Models](https://arxiv.org/abs/2602.22787) | 本文提出自监督数据生成流水线AttriWiki，证明仅用简单线性探针即可从大语言模型的隐藏表示中可靠地判断答案的知识来源是内部记忆还是外部上下文，从而区分忠实性幻觉与事实性幻觉以助力针对性缓解。 |
| [^252] | [Beyond Prompting: Efficient and Robust Contextual Biasing for Speech LLMs via Logit-Space Integration (LOGIC)](https://arxiv.org/abs/2601.15397) | 本文提出LOGIC方法，通过在Logit空间层面直接集成上下文偏置，为语音大语言模型提供了一种高效且鲁棒的解决方案，克服了传统提示方法的可扩展性瓶颈和生成式错误纠正的幻觉问题。 |
| [^253] | [Output Embedding Centering for Stable LLM Pretraining](https://arxiv.org/abs/2601.02031) | 该论文揭示了大语言模型预训练末期输出logit发散的根源在于输出嵌入的各向异性，并提出输出嵌入中心化（OEC）新方法，通过μ-centering或μ-loss两种实现方式有效抑制训练不稳定，其稳定性优于z-loss且与logit软上限方法相当。 |
| [^254] | [DCO: Dynamic Cache Orchestration for LLM Accelerators through Predictive Management](https://arxiv.org/abs/2512.07312) | 提出了一种面向LLM加速器的动态缓存编排方案DCO，利用软件栈中的数据流信息进行预测性缓存管理（包括死块预测、旁路决策和缓存抖动缓解），在保持编程简洁性的同时，相比传统缓存架构实现了高达1.80倍的加速。 |
| [^255] | [On the Societal Impact of Machine Learning](https://arxiv.org/abs/2510.23693) | 本博士论文提出了更恰当地测量机器学习系统公平性、系统性分解系统以预判偏见动态的方法，以及在保持系统效用的同时减少算法歧视的有效干预措施，为使机器学习的社会影响符合更广泛的社会价值奠定了基础。 |
| [^256] | [Timely Clinical Diagnosis through Active Test Selection](https://arxiv.org/abs/2510.18988) | 该论文提出ACTMED框架，将贝叶斯实验设计与大语言模型相结合，在诊断过程的每一步自适应地选择最能降低诊断不确定性的检查项目，以模拟临床医生在资源受限环境下的顺序式诊断推理。 |
| [^257] | [Federated Learning for Surgical Vision in Appendicitis Classification: Results of the FedSurg EndoVis 2024 Challenge](https://arxiv.org/abs/2510.04772) | 本研究发起了首个专注于手术视觉联邦学习的国际挑战赛 FedSurg，基于多中心腹腔镜阑尾切除术数据集进行了概念验证评估，并发现时序建模是提升对未见临床中心泛化能力最稳定的架构因素。 |
| [^258] | [CHRONOBERG: Capturing Language Evolution and Temporal Awareness in Foundation Models](https://arxiv.org/abs/2509.22360) | 该论文提出了CHRONOBERG——一个跨越250年、带有丰富时间标注的英语书籍文本语料库，通过历史校准的情感词典量化语言演变，从而提升基础模型对语言历时变化的时间感知能力。 |
| [^259] | [Evidence for Limited Metacognition in LLMs](https://arxiv.org/abs/2509.21545) | 该研究借鉴非人类动物元认知研究方法，提出了一种不依赖模型自我报告的定量评估框架，发现2024年初以来的前沿大语言模型展现出有限的元认知能力，能够评估自身答题信心并预测自己将给出的答案。 |
| [^260] | [Generative AI performance in core undergraduate mathematics: a curriculum-level case study](https://arxiv.org/abs/2509.13359) | 该研究以一年级数学课程的八份真实考试试卷为基准，通过让生成式AI作答并进行盲评，系统性地评估了GenAI在本科核心数学课程各模块及整体层面的表现。 |
| [^261] | [Amulet: a Python Library for Assessing Interactions Among ML Defenses and Risks](https://arxiv.org/abs/2509.12386) | 本文推出了首个Python库Amulet，用于评估机器学习防御与风险之间的预期及非预期交互作用，其全面性、可扩展性、一致性和适用性为系统性研究跨多种风险的非预期交互提供了统一基础。 |
| [^262] | [Spectral Masking and Interpolation Attack (SMIA): A Black-box Adversarial Attack against Voice Authentication and Anti-Spoofing Systems](https://arxiv.org/abs/2509.07677) | 本文提出了SMIA黑盒对抗攻击方法，通过操纵AI生成音频中人耳不可听见的频谱区域来绕过语音认证和反欺骗系统的检测。 |
| [^263] | [A Survey of Threats Against Voice Authentication and Anti-Spoofing Systems](https://arxiv.org/abs/2508.16843) | 本综述系统梳理了针对语音认证系统和反欺骗对策的四类主要威胁（数据投毒、对抗攻击、深度伪造和对抗性欺骗），追溯了语音认证技术演进过程中漏洞的同步发展，并总结了各类攻击的方法、常用数据集及性能局限。 |
| [^264] | [Learning Intrinsic Water-Quality Dynamics with Rainfall for Data-Driven Forecasting](https://arxiv.org/abs/2508.08279) | 本文提出 RaiNet 模型，联合建模多尺度水质动态与站点特定降雨效应以实现数据驱动的水质预测，并发布了包含超过15万条水质观测数据的三个多模态真实数据集。 |
| [^265] | [Discovering Temporal Structure: An Overview of Hierarchical Reinforcement Learning](https://arxiv.org/abs/2506.14045) | 本文是一篇分层强化学习（HRL）综述，从决策基本挑战的视角阐明了HRL的优势，并系统梳理了发现和利用经验流中时间结构的各类方法。 |
| [^266] | [CertDW: Towards Certified Dataset Ownership Verification via Conformal Calibration](https://arxiv.org/abs/2506.13160) | 本文提出了首个认证数据集水印CertDW及基于保形校准的认证数据集所有权验证方法，能够在受约束的恶意扰动下依然确保可靠的数据集所有权验证。 |
| [^267] | [Generalization in VAE and Diffusion Models: A Unified Information-Theoretic Analysis](https://arxiv.org/abs/2506.00849) | 本文提出了一个统一的信息论框架，通过将编码器和生成器视为随机映射，同时为VAE和扩散模型的泛化提供理论保证，并给出仅基于训练数据的可计算边界，以选择最优扩散时间并改进模型性能。 |
| [^268] | [SG-Blend: Learning an Interpolation Between Improved Swish and GELU for Robust Neural Representations](https://arxiv.org/abs/2505.23942) | SG-Blend提出了一种逐层自适应激活函数，通过可学习的混合系数在改进的参数化Swish（SSwish）与GELU之间进行插值，仅用三个额外标量就让Transformer每一层自动找到最适合自身的激活形状，从而缓解深层网络中的梯度病理问题。 |
| [^269] | [Towards AI-Driven Policing: Interdisciplinary Knowledge Discovery from Police Body-Worn Camera Footage](https://arxiv.org/abs/2504.20007) | 该论文提出了一个结合图像、音频、自然语言处理和大语言模型的多模态跨学科AI框架，用于从警察随身摄像头录像中检测和分析警察与平民互动中的关键行为动态（如尊重、不尊重、事态升级与缓和），并建立了定制评估流程以验证转录质量与行为检测准确性。 |
| [^270] | [Exploring Multimodal Prompt for Visualization Authoring with Large Language Models](https://arxiv.org/abs/2504.13700) | 针对自然语言提示在指导大语言模型进行可视化创作时精度与表达力不足的问题，本文提出以视觉提示作为补充输入模态，并据此设计了多模态提示可视化创作系统VisPilot，有效澄清用户意图并提升模型的解释能力。 |
| [^271] | [ExpTest: Loss-Curve Hypothesis Testing for Autonomous Learning-Rate Selection in Deep Neural Networks](https://arxiv.org/abs/2411.16975) | ExpTest将训练损失曲线作为在线信号，通过序列统计检验检测收敛行为并自主降低学习率，从而实现深度神经网络初始学习率的自动选择，摆脱了繁琐的手动调参。 |
| [^272] | [No Screening is More Efficient with Multiple Objects](https://arxiv.org/abs/2408.10077) | 该论文证明随着物品种类的增加，无筛选机制在福利最大化分配中持续保持最优，因为更多选择使低最优选项价值变得罕见，从而削弱了成本高昂的筛选的必要性，并将其应用于疫苗预约系统的设计。 |
| [^273] | [Three Factors to Improve Out-of-Distribution Detection.](http://arxiv.org/abs/2308.01030) | 本论文提出了三个因素来改善离群检测问题。首先，引入自我知识蒸馏损失以提高网络的准确性；其次，在训练过程中采样半困难离群数据以改善离群检测性能；最后，引入新型监督对比学习以同时提高离群检测性能和网络的准确性。通过结合这三个因素，我们的方法在分类和离群检测之间取得了良好的平衡，提高了准确性和离群检测性能。 |

# 详细

[^1]: GPU-CFR：通过将博弈编译为静态数据流与CUDA图重放，实现80倍加速的反事实遗憾最小化

    GPU-CFR: 80x Faster Counterfactual Regret Minimization by Compiling the Game to Static Dataflow and CUDA Graph Replay

    [https://arxiv.org/abs/2609.11923](https://arxiv.org/abs/2609.11923)

    本文提出GPU-CFR，利用“固定博弈的CFR迭代结构可预先确定”这一观察，将博弈一次性编译为静态数据流并通过CUDA图重放执行，消除了GPU内核启动与框架调度开销，使反事实遗憾最小化的运行速度相比以往方案提升高达80倍。

    

    反事实遗憾最小化（CFR）是少数在CPU上仍比在GPU上运行更快的大型数值计算工作负载之一。每次迭代都需要通过通用树接口发起数百万个细小的、相互依赖的收集（gather）和散射（scatter）步骤，以遍历包含多达数十亿个状态的博弈树。在GPU上，每个内核（kernel）只需数微秒即可完成，因此内核启动和框架调度开销占据了运行时间的主导地位，导致以往的GPU实现败给了优化的CPU代码。我们观察到，对于固定的博弈，CFR迭代中除数值本身之外的所有内容在第一次迭代运行之前就已经是已知的。基于这一观察，我们提出了GPU-CFR，一个由此构建的编译器和运行时系统。它将任意博弈一次性编译为静态数据流：平坦的边和信息集数组、预计算的索引以及按深度分层的批处理执行阶段固定了整个操作序列，各迭代之间只有求解器状态发生变化。静态机会节点折叠、按深度分层的执行（摘要内容在此处被截断）……

    arXiv:2609.11923v1 Announce Type: cross  Abstract: Counterfactual regret minimization (CFR) is one of the few large numerical workloads that still runs faster on CPUs than on GPUs. Each iteration sweeps a game tree with up to billions of states in millions of small, interdependent gather and scatter steps issued through a generic tree interface. On a GPU every kernel finishes in microseconds, so kernel launches and framework dispatch dominate the run time, and prior GPU implementations have lost to optimized CPU code. We observe that for a fixed game, everything about a CFR iteration except the numerical values is known before the first iteration runs. We propose GPU-CFR, a compiler and runtime built on this observation. It compiles any game once into static dataflow: flat edge and information-set arrays, precomputed indices, and depth-level batched passes fix the entire operation sequence, and only solver state changes between iterations. Static chance folding, depth-level execution b
    
[^2]: 协变量偏移与概念偏移的一般量化

    General Quantification of Covariate and Concept Shifts

    [https://arxiv.org/abs/2609.11918](https://arxiv.org/abs/2609.11918)

    本文基于熵最优传输提出γ*-概念偏移新概念和DataShifts算法，统一量化了协变量偏移与概念偏移，并给出了可从样本估计、适用于广泛场景的一般化误差界。

    

    分布偏移下的泛化仍然是现代机器学习的核心挑战，然而现有的学习界理论局限于狭窄的理想化设置，且无法从样本中进行估计。在本文中，我们弥合了理论与实际应用之间的差距。我们首先证明了当源域和目标域的支撑集不匹配时，现有的概念偏移定义会失效。利用熵正则化最优传输，我们提出了一个关键概念：γ*-概念偏移，并推导出一个统一协变量偏移和γ*-概念偏移的一般误差界，该误差界适用于广泛的损失函数、标签空间和随机标注。我们进一步开发了具有集中性保证的偏移估计器，以及DataShifts算法，该算法能够在大多数应用中量化分布偏移并估计误差界——这是一个用于分析分布偏移下学习误差的严谨且通用的工具。

    arXiv:2609.11918v1 Announce Type: new  Abstract: Generalization under distribution shift remains a core challenge in modern machine learning, yet existing learning bound theory is limited to narrow, idealized settings and is non-estimable from samples. In this paper, we bridge the gap between theory and practical applications. We first show that existing definition of concept shift breaks when the source and target supports mismatch. Leveraging entropic optimal transport, we propose a key notion: $\gamma^{*}\!$-concept shifts, and derive a general error bound unifying covariate and $\gamma^{*}\!$-concept shifts, which applies to broad loss functions, label spaces, and stochastic labeling. We further develop estimators for these shifts with concentration guarantees, and the DataShifts algorithm, which can quantify distribution shifts and estimate the error bound in most applications - a rigorous and general tool for analyzing learning error under distribution shift.
    
[^3]: 边缘可部署的视觉语言模型能够识别物种吗？

    Can Edge-Deployable Vision-Language Models Identify Species?

    [https://arxiv.org/abs/2609.11916](https://arxiv.org/abs/2609.11916)

    该研究首次系统评估了2-8B参数边缘可部署视觉语言模型的物种识别能力，发现它们具备真正的分类学知识，但在真实红外相机影像上所有模型（包括专业模型BioCLIP）均因图像可读性问题导致性能显著下降9.6-26.6个百分点。

    

    红外相机陷阱通常部署在野外，运行于连接受限或完全没有网络连接的边缘硬件上，因此小型、本地部署的视觉语言模型（VLM）——而非前沿规模的大模型——才是物种识别中实际需要评估的模型类别。我们测试了这一部署相关范围内2-8B参数的模型是否具备真正的分类学知识，将四个此类VLM（Qwen3-VL 2B/4B/8B、Gemma3 4B）与领域专用模型BioCLIP（3亿参数）在96个物种的识别任务上进行对比，将干净的iNaturalist照片与来自6个LILA.science数据集的红外相机影像进行比较，并使用两个独立采样的评估集。所有模型的物种识别能力都远高于随机水平，但每个模型——无论是通用型还是专业型——在野外影像上的表现都急剧下降（领域差距达9.6-26.6个百分点，且在各个分类级别和两个评估集上均保持一致），这表明性能下降反映的是一般性的图像可读性问题，而非……

    arXiv:2609.11916v1 Announce Type: new  Abstract: Camera traps often run in the field on edge hardware with limited or no connectivity, making small, locally-deployable vision-language models (VLMs) -- not frontier-scale ones -- the practically relevant class to evaluate for species identification. We test whether models in this deployment-relevant 2--8B range carry genuine taxonomic knowledge, evaluating four such VLMs (Qwen3-VL 2B/4B/8B, Gemma3 4B) against the domain-specific specialist BioCLIP (300M parameters) on a 96-species task, comparing clean iNaturalist photographs against camera-trap imagery from 6 LILA.science collections, on two independently-sampled evaluation sets. All models identify species far above chance, but every model -- general-purpose or specialist -- degrades sharply on field imagery (domain gaps of 9.6--26.6 percentage points, consistent across taxonomic levels and both evaluation sets), indicating the degradation reflects general image legibility rather than 
    
[^4]: 生成式营销组合建模：一个将生成式引擎优化（GEO）与生成式引擎营销（GEM）同商业影响相联系的因果推断框架

    Generative Marketing Mix Modeling: A Causal Inference Framework Linking GEO and GEM to Business Impact

    [https://arxiv.org/abs/2609.11915](https://arxiv.org/abs/2609.11915)

    本文提出生成式营销组合建模（GMMM）这一因果推断框架，通过结合注意概率与生成答案、赞助记录等数据，首次实现了对生成式引擎优化（GEO）和生成式引擎营销（GEM）商业因果效应的估计。

    

    生成式人工智能改变了企业触达客户的方式，但标准的营销数据并未记录用户在生成的答案中看到并注意到企业名称的频率。我们开发了生成式营销组合建模（GMMM）来估计生成式引擎优化（GEO）和生成式引擎营销（GEM）的因果效应。对于GEO，GMMM将重复生成的答案与问题数量、各生成系统的使用份额以及注意概率相结合；对于GEM，它将赞助 placements 的记录与注意概率相结合。GMMM通过比较不同处理序列下的预期商业反应，并为识别由此产生的效应建立了充分条件。我们使用英文和日文的产品推荐模拟答案对所提出方法的实证表现进行了研究。

    arXiv:2609.11915v1 Announce Type: cross  Abstract: Generative artificial intelligence changes how firms reach customers, but standard marketing data do not record how often users see and notice a firm's name in generated answers. We develop Generative Marketing Mix Modeling (GMMM) to estimate the causal effects of Generative Engine Optimization (GEO) and Generative Engine Marketing (GEM). For GEO, GMMM combines repeated generated answers with question counts, shares of use across generative systems, and notice probabilities. For GEM, it combines records of sponsored placements with notice probabilities. GMMM compares expected business responses under alternative treatment sequences and establishes sufficient conditions for identifying the resulting effects. We investigate the empirical performance of the proposed method using simulated answers to product recommendation in English and Japanese.
    
[^5]: 人造本我：智能体AI中的驱动力与持久性对齐

    Artificial Id: Drive and Persistent Alignment in Agentic AI

    [https://arxiv.org/abs/2609.11911](https://arxiv.org/abs/2609.11911)

    本文提出“人造本我”——一种自适应内部驱动力，使智能体AI无需外部显式指定行为规则，即可自主决定行为是继续、停止还是改变，并通过虚拟实验证明有用的控制能力可以经由差异性持久性自发涌现。

    

    智能体AI（Agentic AI）正在从有界的任务执行，向能够保留关键状态、跨任务边界持续运行并自我适应的系统发展。这一转变带来一个控制问题，而当前的框架主要是通过人工方式解决的：目标、重试、验证、停止规则以及其他行为转换都需要在外部显式指定。我们提出了“人造本我”，一种自适应的内部驱动力，用于决定行为应该继续、停止还是改变。在一个最小化的虚拟培养皿实验中，一个因规模太小而无法进行通用推理、且未接收任何任务特定行为目标的控制器，通过差异性持久性发展出了有用的控制能力。当某种行为表现出更好的持久性时，同一机制会选择一种非预期的物理策略；而当环境的含义发生变化时，该机制后来还会替换已学习的传感器映射。这些结果表明，自适应方向可以在不被显式指定为行为目标的情况下涌现。

    arXiv:2609.11911v1 Announce Type: new  Abstract: Agentic AI is moving from bounded task execution toward systems that retain consequential state, continue operating and adapt across task boundaries. That shift creates a control problem that current harnesses largely solve by hand: objectives, retries, verification, stopping rules and other behavioral transitions are specified externally. We propose an artificial id, an adaptive internal drive for determining whether behavior should continue, stop or change. In a minimal virtual Petri-dish experiment, a controller too small to perform general-purpose reasoning and receiving no task-specific behavioral objective develops useful control through differential persistence. The same mechanism selects an unintended physical strategy when that behavior persists better and later replaces a learned sensor mapping when its environmental meaning changes. These results show that adaptive direction can emerge without being explicitly specified as a b
    
[^6]: MindTopo：基础模型能在拓扑空间中推理吗？

    MindTopo: Can Foundation Models Reason in Topological Space?

    [https://arxiv.org/abs/2609.11900](https://arxiv.org/abs/2609.11900)

    该论文提出了MindTopo基准，基于认知科学与形式拓扑学，从连续性、分离性、有序性、包含性和纽结五个拓扑属性出发，在推理与规划两个认知层面对14个多模态大语言模型的拓扑空间能力进行系统评估。

    

    空间推理不仅依赖于距离、角度和形状等度量属性，还依赖于在连续变形下保持不变的拓扑关系。认知科学认为这些拓扑关系是空间理解的基础，然而目前针对基础模型的评估大多集中于度量关系或依赖视角的关系。我们提出了MindTopo，一个基于认知科学和形式拓扑学、涵盖五个拓扑属性的拓扑直觉基准：连续性、分离性、有序性、包含性和纽结。MindTopo在两个认知层面对每个属性进行评估：推理层面要求模型识别拓扑关系或推断其如何变化；规划层面则将基础模型实例化为闭环智能体，其策略负责选择环境动作。MindTopo包含11,030个实例，覆盖13种难度可控的程序化生成任务类型。我们对14个多模态大语言模型（MLLM）进行了基准测试，并研究了智能体的不同配置。

    arXiv:2609.11900v1 Announce Type: cross  Abstract: Spatial reasoning depends not only on metric properties such as distance, angle, and shape, but also on topological relations that remain invariant under continuous deformation. Cognitive science identifies these relations as foundational to spatial understanding, yet foundation-model evaluations largely focus on metric or viewpoint-dependent relations. We introduce MindTopo, a benchmark of topological intuition across five properties grounded in cognitive science and formal topology: continuity, separation, order, enclosure, and knots. MindTopo evaluates each property at two cognitive levels. Reasoning asks a model to identify topological relations or infer how they change. Planning instantiates a foundation model as a closed-loop agent whose policy selects environment actions. MindTopo contains 11,030 instances across 13 procedurally generated task types with controllable difficulty. We benchmark 14 MLLMs and study agent configuratio
    
[^7]: 大语言模型中的领域特定幻觉检测

    Domain-Specific Hallucination Detection in Large Language Models

    [https://arxiv.org/abs/2609.11878](https://arxiv.org/abs/2609.11878)

    提出了一种结合微调DeBERTa-v3分类、蒙特卡洛Dropout不确定性量化和温度缩放校准的多信号幻觉检测流水线，在HaluEval基准上取得F1=0.915的高性能，并结合直接偏好优化（DPO）进一步改进模型表现。

    

    大语言模型生成的流畅文本可能包含不真实的陈述——这一现象被称为幻觉。我们提出了一种多信号检测流水线，结合了微调的DeBERTa-v3分类器、蒙特卡洛（MC）Dropout不确定性量化以及温度缩放校准，用于响应级别的幻觉检测。在HaluEval基准上的评估显示，我们的流水线在通用领域任务上达到F1=0.915和AUROC=0.977，各任务的F1分数分别为0.97（问答）、0.96（摘要）和0.82（对话）。MC Dropout推理进一步将准确率提升至93.2%。上下文消融研究证实模型执行的是真正的蕴含推理，而非利用表面模式——当移除知识上下文时，摘要任务的F1下降了24%。学习曲线分析表明，25%的训练数据即可捕获全数据性能的77%。除检测之外，我们还将直接偏好优化（DPO）应用于Qw……

    arXiv:2609.11878v1 Announce Type: new  Abstract: Large language models generate fluent text that can contain unfaithful claims -- a phenomenon known as hallucination. We present a multi-signal detection pipeline combining fine-tuned DeBERTa-v3 classification, Monte Carlo (MC) Dropout uncertainty quantification, and temperature-scaled calibration for response-level hallucination detection. Evaluated on the HaluEval benchmark, our pipeline achieves F1=0.915 and AUROC=0.977 on general-domain tasks, with per-task F1 scores of 0.97 (QA), 0.96 (Summarization), and 0.82 (Dialogue). MC Dropout inference further improves accuracy to 93.2%. A context ablation study confirms the model performs genuine entailment reasoning rather than exploiting surface patterns, with summarization F1 dropping 24% when knowledge context is removed. Learning curve analysis reveals that 25% of training data captures 77% of full-data performance. Beyond detection, we apply Direct Preference Optimization (DPO) to a Qw
    
[^8]: 生物学在环：CRISPR筛选中的摊销式自适应命中物发现

    Biology-in-the-loop: Amortized Adaptive Hit Discovery in CRISPR Screens

    [https://arxiv.org/abs/2609.11877](https://arxiv.org/abs/2609.11877)

    本文提出了包含1,389个CRISPR筛选实验的大规模基准AssayBench-Loop，并在此基础上构建了AssayLoop顺序实验设计框架，利用跨历史实验训练的transformer摊销式采集策略，实现CRISPR筛选中受限预算下的自适应命中物发现。

    

    许多生物学发现问题需要在有限预算下顺序地选择实验。CRISPR筛选就是一个典型的例子，因为穷举式的扰动测试通常不可行，而必须在多个实验轮次中对候选扰动进行优先级排序。尽管这一问题非常重要，但现有的自适应命中发现基准在规模和多样性方面仍然有限。在此，我们介绍了AssayBench-Loop，这是一个大规模的自适应命中发现基准，包含五个表型类别下的1,389个CRISPR筛选实验。除了支持系统化评估之外，其规模还使得跨历史实验学习采集（acquisition）策略成为可能。基于这一资源，我们提出了AssayLoop，这是一个顺序实验设计框架，它将AssayFormer（一种基于transformer的摊销式采集策略，通过在历史筛选数据上训练，能够从实验反馈中进行自适应调整）与L…相结合。

    arXiv:2609.11877v1 Announce Type: cross  Abstract: Many biological discovery problems require experiments to be selected sequentially under constrained budgets. CRISPR screening is a prominent example, as exhaustive perturbation testing is often infeasible and candidate perturbations must instead be prioritized over multiple experimental rounds. Despite the importance of this problem, existing benchmarks for adaptive hit discovery remain limited in scale and diversity. Here, we introduce AssayBench-Loop, a large-scale benchmark for adaptive hit discovery comprising 1,389 CRISPR screens across five phenotype categories. Beyond enabling systematic evaluation, its scale makes it possible to learn acquisition strategies across historical experiments. Building on this resource, we introduce AssayLoop, a sequential experimental design framework combining AssayFormer, a transformer-based amortized acquisition policy trained across historical screens to adapt from experimental feedback, with L
    
[^9]: 关于线性推荐模型的正则化图景研究

    On the Regularization Landscape for the Linear Recommendation Models

    [https://arxiv.org/abs/2609.11876](https://arxiv.org/abs/2609.11876)

    本文将多种深度学习启发的线性推荐算法统一到正则化框架下，证明它们本质上都可归结为核范数或Frobenius范数正则化，并揭示了两者在预测能力、解的秩与计算效率之间的权衡。

    

    近年来，一系列受深度学习技术启发的推荐算法在多个标准推荐基准上成为性能领先者。虽然这些算法建立在不同的深度学习技术之上（如dropout、自编码器），但它们具有相似的性能，甚至相似的成本函数。本文研究这些模型的可比性能是否纯属巧合，还是它们可以在一个统一框架下被归纳。我们发现，所有线性的性能领先算法实际上都只是添加了基于核范数的正则化项，或基于Frobenius范数的正则化项。前者具有一种（令人惊讶的）刚性结构，这限制了模型的预测能力，但其解是低秩的且具有闭式解。后者更具表达力且对推荐任务更高效，但其解要么是满秩的，要么需要执行难以调参的数值过程（如ADMM）。

    arXiv:2609.11876v1 Announce Type: new  Abstract: Recently, a wide range of recommendation algorithms inspired by deep learning techniques have emerged as the performance leaders on several standard recommendation benchmarks. While these algorithms were built on different DL techniques (e.g., dropouts, autoencoder), they have similar performance and even similar cost functions. This paper studies whether the models' comparable performance are sheer coincidence, or they can be unified under a single framework. We find that all linear performance leaders effectively add only a nuclear-norm based regularizer, or a Frobenius-norm based regularizer. The former ones possess a (surprising) rigid structure that limits the models' predictive power but their solutions are low rank and have closed form. The latter ones are more expressive and more efficient for recommendation but their solutions are either full-rank or require executing hard-to-tune numeric procedures such as ADMM. Along this line
    
[^10]: 人类建造的最后一个AI：迈向真正的递归自我改进

    The Last AI Built by Humans: Toward Genuine Recursive Self-Improvement

    [https://arxiv.org/abs/2609.11873](https://arxiv.org/abs/2609.11873)

    本文提出递归自我改进（RSI）概念及其从改进执行自主性到递归元改进的发展路线图，通过Headroom-Closed指数揭示现有大语言模型的局限，并结合行业实践识别出实现真正AI自我改进的关键挑战。

    

    递归自我改进（RSI）使AI系统能够将经验和反馈转化为持久性的改变，从而同时提升其自身能力和未来的改进过程。我们首先使用Headroom-Closed指数（HCI）揭示现有大语言模型存在的问题，然后介绍RSI概念及其发展路线图：从改进执行自主性、改进策略自主性、经验获取自主性和环境适应自主性，到递归元改进。接下来，我们考察了RSI在不同场景（如科学发现、具身智能、软件工程）中的表现，重点阐述了它们各自不同的需求和发展速度。借鉴多样的行业实践和初步的实证证据，我们将RSI研究与实际系统联系起来，并识别出实现真正RSI所面临的关键挑战。

    arXiv:2609.11873v1 Announce Type: cross  Abstract: Recursive self-improvement (RSI) enables AI systems to turn experience and feedback into persistent changes that improve both their capabilities and the process of future improvement. We first use the Headroom-Closed Index (HCI) to reveal the problems of existing LLMs, then introduce the RSI concept and its development roadmap: from improvement-execution autonomy, improvement-strategy autonomy, experience-acquisition autonomy, and environment-adaptation autonomy, to recursive meta-improvement. Next we examine RSI across scenarios (e.g., scientific discovery, embodied intelligence, software engineering), highlighting their distinct requirements and development speeds. Drawing on diverse industry practices and preliminary empirical evidence, we connect RSI research with practical systems and identify key challenges to achieving genuine RSI.
    
[^11]: RetroThinker：在语音大语言模型中实现回顾性思考

    RetroThinker: Enabling Retrospective Thinking in Speech LLMs

    [https://arxiv.org/abs/2609.11864](https://arxiv.org/abs/2609.11864)

    该论文提出了RetroThinker多阶段后训练框架，使流式语音大语言模型能够在推理过程中自我验证并即时修正思维链推理步骤，从而在满足实时语音交互延迟约束的同时提升复杂推理能力。

    

    语音大语言模型相比级联式自动语音识别（ASR）与基于文本的语言模型架构，能够提供更低的延迟，并保留通常在此类级联架构中丢失的副语言细微特征。然而，语音大语言模型在复杂推理任务上仍落后于纯文本大语言模型，而实时语音交互又施加了严格的延迟约束。尽管先前的工作采用思维链（CoT）和并发推理来增强推理能力而不引入过高的延迟，但固有的精度与延迟之间的权衡依然存在。在本文中，我们研究了流式语音大语言模型能否在运行过程中动态地修正其推理轨迹。我们提出了RetroThinker，这是一个多阶段后训练框架，使Moshi模型能够在推理过程中对CoT步骤进行自我验证和前向修正。RetroThinker将在精选的回顾性思考数据上进行的监督微调（SFT）与基于长度的直接偏好优化相结合。

    arXiv:2609.11864v1 Announce Type: cross  Abstract: Speech large language models (SpeechLLMs) offer reduced latency and retain paralinguistic nuances that are typically lost in cascaded automatic speech recognition (ASR) and text-based LM architectures. However, they continue to lag behind text-only LLMs on complex reasoning tasks, while real-time spoken interaction imposes strict latency constraints. Although prior works employ Chain-of-Thought (CoT) and concurrent reasoning to enhance reasoning capabilities without inducing prohibitive delays, an inherent accuracy-latency trade-off persists. In this paper, we investigate whether a streaming SpeechLLM can dynamically revise its reasoning traces on the fly. We introduce RetroThinker, a multi-stage post-training framework that equips the Moshi model to self-verify and forward-correct CoT steps during inference. RetroThinker combines supervised fine-tuning (SFT) on curated retrospective thinking data with length-based direct preference op
    
[^12]: 可解释性助手：用于解读能源消耗模型的对语式XAI界面

    Explainability Assistant: A Conversational XAI Interface for Interpreting Energy Consumption Models

    [https://arxiv.org/abs/2609.11860](https://arxiv.org/abs/2609.11860)

    本文提出了一种利用大语言模型函数调用能力的开源对话式XAI系统“可解释性助手”，将意图解析准确率从76.8%提升至94%，帮助设施管理者等非技术人员更灵活地理解能源消耗预测模型。

    

    能源消耗预测依赖于日益复杂的机器学习（ML）模型，例如基于遗传编程的符号回归器，而设施管理者和建筑运营人员往往难以理解这些模型的预测结果。可解释人工智能（XAI）技术可以解决这种不透明性问题，但传统的XAI仪表板需要大量的技术专业知识，并且在动态、情境感知的查询方面灵活性有限。对话式XAI系统提供了一种有前景的替代方案；然而，先前的方法（如TalkToModel）受限于僵化的自定义语法，意图解析准确率仅为76.8%。本文介绍了“可解释性助手”，这是一个开源的对话式XAI系统，它利用现代大语言模型（LLM）的函数调用能力来克服这些局限。该系统实现了94%的意图解析准确率，并支持灵活的自然语言查询

    arXiv:2609.11860v1 Announce Type: cross  Abstract: Energy consumption forecasting relies on increasingly complex machine learning (ML) models, such as Genetic Programming-based symbolic regressors, whose predictions can be difficult for facility managers and building operators to interpret. Explainable Artificial Intelligence (XAI) techniques address this opacity, but traditional XAI dashboards require substantial technical expertise and provide limited flexibility for dynamic, context-aware inquiry. Conversational XAI systems offer a promising alternative; however, previous approaches, such as TalkToModel, were constrained by rigid custom grammars and achieved only 76.8% intent-parsing accuracy. This paper introduces the Explainability Assistant, an open-source conversational XAI system that leverages the function-calling capabilities of modern Large Language Models (LLMs) to overcome these limitations. The system achieves 94% intent-parsing accuracy, supports flexible natural languag
    
[^13]: 从参数到答案：大语言模型如何检索和运用其内部知识

    From Parameters to Answers: How LLMs Retrieve and Use Their Internal Knowledge

    [https://arxiv.org/abs/2609.11859](https://arxiv.org/abs/2609.11859)

    该论文通过对Qwen、Llama和Gemma模型隐藏状态进行逐层干预，揭示了大语言模型在回答问题时“查询路由方向”先于目标知识形成并影响答案生成的因果机制与时间窗口。

    

    语言模型在回答问题时，其对查询路由信息和目标知识的依赖是如何变化的？我们通过对问题末尾隐藏状态进行逐层干预来研究这一问题。在Qwen、Llama和Gemma模型上，我们比较了国家-大洲类问题与名词、形容词及代码类答案，同时保持若干拟合测量彼此独立。其中，成对条件化的请求方向描述了在自然的单国家问题中查询的是哪个国家；全局请求方向描述了成对问题中第一国家与第二国家请求之间的区别；分离的选择候选则用于测试隐藏状态中已有内容之间的控制关系。对冻结的Qwen自然问题状态的诊断性再分析表明，该成对条件化方向在干预开始改变后续拟合知识之前就已增强，且这一因果窗口在答案支持内容仍在形成之时便已开启。

    arXiv:2609.11859v1 Announce Type: new  Abstract: How does a language model's dependence on query-routing information and target knowledge change as it answers a question? We study this question through layerwise interventions on the hidden state at the end of the question. Across Qwen, Llama, and Gemma, we compare country-continent questions with noun, adjective, and code answers while keeping several fitted measurements distinct. A pair-conditioned request direction describes which country is queried in natural single-country questions; a global request direction describes first- versus second-country requests in paired questions; separate selection candidates test control among contents already available in the hidden state. A diagnostic reanalysis of frozen Qwen natural-question states shows that the pair-conditioned direction grows stronger before interventions on it begin to alter later fitted knowledge, with this causal window opening while answer-supporting content is still form
    
[^14]: 模型感知调度通过逐纤维最优传输改进生成质量

    Model-Aware Schedules Improve Generation via Fiberwise Optimal Transport

    [https://arxiv.org/abs/2609.11842](https://arxiv.org/abs/2609.11842)

    该论文提出一种基于逐纤维最优传输的模型感知调度构建方法，通过将预测风险与动能作用结合得到闭式最优时间分配，从而改进扩散与流匹配模型的生成效果。

    

    扩散模型和流匹配调度控制着沿仿射概率路径混合数据与噪声的信号和噪声系数。通过最小化定义在系数路径上的动能作用，可以为强基线方法的表现提供解释，但这种方法仍然是模型无关的，并且忽略了预测误差。本文介绍了一种基于逐纤维最优传输的模型感知调度构建方法。在概率路径上的固定时间和状态处，兼容的信号/噪声分解形成一个仿射纤维。我们通过在这些纤维内对真实分解与预测器诱导分解之间的最优传输成本取平均，定义了逐纤维预测风险。在固定的系数曲线上，将该风险与系数路径动能作用相结合，可以得到闭式最优时间分配。这种构建方法可以扩展到一般的线性预测目标，并且风险曲线可以从早期基线检查点估计得出。我们评估了……（原文摘要在此处截断）

    arXiv:2609.11842v1 Announce Type: new  Abstract: Diffusion and flow-matching schedules control the signal and noise coefficients that mix data and noise along affine probability paths. Minimizing a kinetic action defined on coefficient paths, motivated by optimal transport, helps explain strong baselines but remains model-agnostic and ignores prediction error. Here we introduce a model-aware schedule construction based on fiberwise optimal transport. At a fixed time and state on the probability path, compatible signal/noise decompositions form an affine fiber. We define a fiberwise prediction risk by averaging optimal-transport costs between the true and predictor-induced decompositions within these fibers. On a fixed coefficient curve, combining this risk with coefficient-path kinetic action yields a closed-form optimal time allocation. This construction extends to general linear prediction targets, and the risk profile can be estimated from an early baseline checkpoint. We evaluate D
    
[^15]: 理解海事操作员对AI辅助决策的态度

    Understanding Operator Attitudes Toward AI-Supported Decision Making in Maritime Operations

    [https://arxiv.org/abs/2609.11805](https://arxiv.org/abs/2609.11805)

    本研究通过问卷调查与开放回答分析发现，海事专业人员对AI辅助避碰决策助手总体持积极态度、信任稳定，重视其在决策支持和态势感知方面的价值，同时对AI可靠性仍存顾虑。

    

    海上自主水面船舶（MASS）和AI辅助决策助手有望变革海事运营，但它们的安全整合取决于海事专业人员如何感知和信任此类系统。本文呈现了一项关于海事利益相关者在避碰场景中对AI辅助助手态度的调查研究。参与者使用成熟的和经过改编的问卷评估了技术焦虑、对自动化的信任以及解释质量，并结合对开放式回答的情感分析和主题分析。结果表明，参与者对海事技术总体持积极态度，开放性方面没有明显的年龄相关差异，跨场景的信任保持稳定，而对解释质量的评价则更具场景敏感性且呈现多维性。开放式回答显示，参与者重视AI对决策制定、态势感知和建立信心的支持，同时也对AI的可靠性表示担忧。

    arXiv:2609.11805v1 Announce Type: cross  Abstract: Maritime Autonomous Surface Ships (MASS) and AI- supported decision assistants are expected to transform maritime operations, but their safe integration depends on how maritime professionals perceive and trust such systems. This paper presents a survey study on maritime stakeholders' attitudes toward an AI-supported assistant in collision-avoidance scenarios. Participants evaluated technology anxiety, trust in automation, and explanation quality using established and adapted questionnaires, complemented by sentiment and thematic analysis of open-ended responses Results indicate a generally positive disposition toward maritime technology, no clear age-related differences in openness, stable trust across scenarios, and more scenario-sensitive, multidimensional explanation ratings. Open responses showed that participants valued support for decision-making, situation awareness, and confidence-building, while raising concerns about AI relia
    
[^16]: Logit 精炼器：通过尺度内依赖建模改进视觉自回归模型

    Logit Refiner: Improving Visual Autoregressive Models via Intra-Scale Dependency Modeling

    [https://arxiv.org/abs/2609.11804](https://arxiv.org/abs/2609.11804)

    Logit Refiner 是一个轻量级自回归模块，通过恢复 VAR 模型尺度内 token 间的空间依赖关系，仅需约 10% 的额外参数即可免重训直接插入任何预训练 VAR 模型，有效改善生成图像的局部连贯性。

    

    视觉自回归模型（VAR）通过下一尺度预测生成图像，在每个尺度内并行生成所有 token。我们证明这种并行解码本质上是一种平均场式的近似，丢弃了同尺度 token 之间的空间依赖关系，导致无论骨干网络容量多大，生成的样本都会出现局部不连贯的问题——这是解码规则本身的固有局限。为解决这一局限，我们提出了 Logit Refiner，一个轻量级的自回归模块，它在冻结的骨干特征条件下顺序采样 token，从而恢复尺度内依赖关系。该模块仅增加约 10% 的参数和不到基础模型 5% 的训练计算量，即可直接插入任何预训练的 VAR 检查点，无需重新训练。受控消融实验表明，联合尺度内采样——而非额外的模型容量或训练——是提升效果的关键因素。在类条件 ImageNet 256x256 上，跨 310M 到 2B 参数的多种骨干网络，该方法均展现出一致的改进。

    arXiv:2609.11804v1 Announce Type: cross  Abstract: Visual Autoregressive Models (VAR) generate images through next-scale prediction, producing all tokens within each scale in parallel. We show that this parallel decoding constitutes a mean-field-style approximation that discards spatial dependencies among same-scale tokens, causing locally incoherent samples regardless of backbone capacity -- a limitation of the decoding rule. Addressing this limitation, we introduce the Logit Refiner, a lightweight autoregressive module that restores intra-scale dependencies by sequentially sampling tokens conditioned on frozen backbone features. Adding only ~10% parameters and less than 5% of the base model's training compute, it plugs into any pretrained VAR checkpoint without retraining. Controlled ablations isolate joint intra-scale sampling -- rather than additional capacity or training -- as the critical ingredient. Across backbones from 310M to 2B parameters on class-conditional ImageNet 256x25
    
[^17]: 循环流思考

    Thinking with Looped Flows

    [https://arxiv.org/abs/2609.11801](https://arxiv.org/abs/2609.11801)

    提出循环流方法，通过局部去噪目标训练循环隐藏状态，使早期更新能支持后续更新，从而通过更多推理计算解决更难的问题。

    

    人类和机器通常通过投入更多计算时间来解决更难的问题。在深度学习中，循环模型在推理阶段通过反复更新隐藏状态来实现这一理念。然而在实践中，这类模型的训练只在一次或少数几次更新上进行反向传播，导致难以训练早期更新去支持后续更新。我们提出了循环流，该方法通过局部去噪目标来训练循环，从而规避了这一问题。通过逐步降低的噪声水平以及共享噪声，我们在去噪目标之间施加时间关联，激励模型学习能够随时间传递有用计算的循环状态，即使梯度仅覆盖少数几次更新。随后，我们将推理表述为对由学习到的去噪器参数化的概率流速度进行积分，并与循环状态耦合。这使得模型可以通过投入更多计算来解决更难的问题。

    arXiv:2609.11801v1 Announce Type: new  Abstract: Humans and machines often solve harder problems by spending more time on computation. In deep learning, looped models implement this idea during inference by recurrently updating a hidden state. In practice, however, their training backpropagates through only one or a few updates, making it hard to train early updates to support future ones. We propose looped flows, an approach that sidesteps this issue by training the recurrence with local denoising objectives. By imposing temporal association across denoising objectives through progressively decreasing noise levels and shared noise, the model is incentivized to learn recurrent states that transfer useful computation over time, even when gradients cover only a few updates. We then formulate inference as integrating the velocity of a probability flow parameterized by the learned denoiser, coupled with recurrent states. This allows solving harder problems by spending more computation thro
    
[^18]: 超越词错误率：ASR与音频语言模型在英语-约鲁巴语码转换语音上的切换感知评估

    Beyond Word Error Rate: A Switch Aware Evaluation of ASR and Audio Language Models on English Yoruba Code-Switched Speech

    [https://arxiv.org/abs/2609.11786](https://arxiv.org/abs/2609.11786)

    该论文提出了一套切换感知的评估指标（包括切换入口标记错误率SETER等），揭示总体词错误率会掩盖码转换语音识别的真实表现，且领先的音频语言模型虽然与最佳ASR模型的WER相当，却在所有码转换相关指标上显著更优。

    

    自动语音识别（ASR）系统和音频语言模型在单语基准测试上已报告较低的错误率，但它们在低资源、变音符号丰富的语言中的码转换语音上的表现仍缺乏充分刻画。我们对十一个现代系统（六个ASR模型和五个音频语言模型）在英语-约鲁巴语码转换语音上进行了切换感知评估，使用了一个确定性的2000条语音评估集和统一的评分流程。除词错误率（WER）之外，我们报告了切换定位的诊断指标：切换入口标记错误率（SETER）、窗口化切换点错误率、语言特定错误率以及变音符号不敏感的WER。我们的核心发现是，总体WER会掩盖码转换行为。按WER衡量的最佳系统（一个ASR模型）与一个领先的音频语言模型在WER上统计上无显著差异，但该音频语言模型在所有切换定位指标上都显著更优。

    arXiv:2609.11786v1 Announce Type: new  Abstract: Automatic speech recognition (ASR) systems and audio language models (audio LMs) now report low error rates on monolingual benchmarks, but their behavior on code switched speech in low resource, diacritic rich languages remains poorly characterized. We present a switch aware evaluation of eleven modern systems (six ASR models and five audio LMs) on English Yoruba code-switched speech, using a deterministic 2000 utterance evaluation set and a shared scoring pipeline. Beyond word error rate (WER), we report switch localized diagnostics: a switch entry token error rate (SETER), windowed switch point error rates, language specific error rates, and a diacritic insensitive WER. Our central finding is that aggregate WER hides code switching behavior. The best system by WER (an ASR model) is statistically indistinguishable from a leading audio LM on WER, yet the audio LM is significantly better on every switch localized metric. Across faithful s
    
[^19]: 识别不等于逆转：事实保持型新闻框架的受控逆转测试

    Recognizing Is Not Reversing: A Controlled Inversion Test of Fact-Preserving News Framing

    [https://arxiv.org/abs/2609.11769](https://arxiv.org/abs/2609.11769)

    该研究通过受控逆转实验发现，大语言模型虽能较好地保持事实并识别新闻框架，却几乎无法逆转已知的框架变换（逆转率仅约0.044–0.068），证明“识别框架”与“逆转框架”是截然不同的能力。

    

    大语言模型（LLM）越来越多地被用于分析和改写新闻，然而当前的框架研究主要评估生成、检测或改写后的文本是否显得更加中立，并没有直接展示模型能否在保持事实不变的前提下撤销一个已知的框架变换。我们引入了一项受控逆转测试，涵盖三种既定的文本框架实现方式：评价性词汇、施事性实现和信息显著性。基于60篇新闻文章和三种干预强度，共产生了540对保持原子事实并记录编辑操作的配对变体。在Qwen、DeepSeek和Kimi模型上，事实保持率接近0.84，而干预逆转率仅为0.044–0.068。即使框架类型和方向都被正确识别，汇总的逆转率也仅达到0.071。这些结果揭示了事实保真度、框架识别与框架逆转之间的明显分离。

    arXiv:2609.11769v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used to analyze and rewrite news, yet current framing studies mainly evaluate generation, detection, or whether rewritten text appears more neutral. They do not directly show whether a model can undo a known framing transformation while keeping the facts fixed. We introduce a controlled inversion test over three established textual realizations of framing: evaluative lexis, agency realization, and information salience. Across 60 news articles and three intervention strengths, this yields 540 paired variants with preserved atomic facts and recorded edits. Across Qwen, DeepSeek, and Kimi, factual preservation remains near 0.84, whereas intervention reversal is 0.044--0.068. Even when both framing type and direction are recognized correctly, pooled reversal reaches 0.071. These results reveal a clear separation between factual fidelity, framing recognition, and framing inversion: recognizing how
    
[^20]: 面向在线策略蒸馏的统一逐词元门控族：带多通道与偏置系数的FKL/RKL混合

    A Unified Per-Token Gating Family for On-Policy Distillation: FKL/RKL Mixing with Multi-Channel and Bias Coefficients

    [https://arxiv.org/abs/2609.11768](https://arxiv.org/abs/2609.11768)

    该论文提出了一个统一的四系数逐词元门控参数化方法，将EOPD和ToDi作为其一维限制的特殊情形统一起来，并通过多通道组合与显式偏置这两个额外自由度，在在线策略蒸馏实验中显著超越了单通道门控限制方法。

    

    逐词元的前向/反向KL散度损失门控已成为在线策略知识蒸馏（OPD）的标准技术，但现有方法如EOPD（Jin等，2026）和ToDi（Jung等，2025）各自固定了单一的门控信号和单一的门控方向，且两者从未被直接比较。我们引入了一个四系数参数化 lambda_t = sigma(a * h_t + b * u(x) + c + d * gap_t)，其中与方向对齐的EOPD和ToDi代理形式作为其一维（1D）限制情形出现，同时该参数化还引入了多通道组合和显式偏置作为额外的自由度。在TweetEval（Barbieri等，2020）的情感和仇恨数据集上，使用Qwen3-32B教师模型和Qwen3-4B学生模型进行实验，完整门控族中的配置在36个可比单元中的33个里达到了比同等幅度的单通道（仅熵/仅差距）1D限制更高的准确率，而一项26单元的均值匹配隔离实验将动态门控置于……（原文在此处截断）

    arXiv:2609.11768v1 Announce Type: cross  Abstract: Per-token gating of forward/reverse KL losses has become a standard technique for on-policy knowledge distillation (OPD), but existing methods such as EOPD (Jin et al., 2026) and ToDi (Jung et al., 2025) each fix a single gating signal and a single gating direction, and the two have never been compared directly. We introduce a four-coefficient parameterization lambda_t = sigma(a * h_t + b * u(x) + c + d * gap_t) in which direction-aligned proxies of EOPD and ToDi appear as one-dimensional (1D) restrictions, and which adds multi-channel composition and an explicit bias as further degrees of freedom. On TweetEval (Barbieri et al., 2020) emotion and hate, with a Qwen3-32B teacher and a Qwen3-4B student, configurations in the full family reach higher accuracy than the matched-magnitude single-channel (entropy-only / gap-only) 1D restrictions in 33 of 36 comparable cells, and a 26-cell mean-match isolation experiment places dynamic gating a
    
[^21]: SIRF：面向工业内容风控的规范内化风险基础模型

    SIRF: A Spec-Internalized Risk Foundation Model for Industrial Content Risk Control

    [https://arxiv.org/abs/2609.11752](https://arxiv.org/abs/2609.11752)

    SIRF通过持续预训练将平台风控规范内化到模型权重中，无需人工标注即可在超低延迟、仅输出判定的部署形态下实现高精度风险处置，仅用约7000万token的CPT便将P95精确率下的黑名单召回率提升15.1个百分点且不损害通用能力。

    

    对于工业内容风控而言，真正的部署约束不是平均准确率，而是在高精度与秒级延迟下能自动处置多少风险内容。我们提出SIRF（规范内化风险基础模型），它通过持续预训练（CPT）将平台的复杂规范策略内化到模型权重中——这些策略通过EntiGraph、MAGA改写和账号级思维链（CoT）合成，无需额外人工标注——从而在超低延迟、仅输出判定结果的部署形态下实现高精度的规则执行。一项同源对照比较实验（Qwen3-8B-SFT对比SIRF-8B-SFT，两者采用相同的策略注入和仅判定输出形式，唯一差异在于是否进行基于策略的CPT）将性能提升归因于规范内化：SIRF-8B-SFT在95%精确率约束下达到71.3%的黑名单召回率（Black Recall@P95），较基线提升15.1个百分点，且仅使用约7000万个CPT token，未损害模型通用能力；在本接口下纳入的可获取logprob的模型中，它……

    arXiv:2609.11752v1 Announce Type: cross  Abstract: For industrial content risk control, the real deployment constraint is not average accuracy but how much risk can be auto-handled under high precision and second-level latency. We present SIRF (Spec-Internalized Risk Foundation Model), which internalizes a platform's complex policies, synthesized without additional human annotation via EntiGraph, MAGA rewriting and account-level chain-of-thought (CoT), into the weights via continued pretraining (CPT), so rules are applied at high precision under an ultra-low-latency, verdict-only deployment. A controlled same-source comparison (Qwen3-8B-SFT vs. SIRF-8B-SFT, identical policy injection and verdict-only output form, differing only in policy-grounded CPT) attributes the gain to internalization: SIRF-8B-SFT reaches 71.3% Black Recall@P95, +15.1pp over the baseline, using only ~70M CPT tokens without harming general ability, and among included, logprob-available models under this interface i
    
[^22]: LOCUS：面向Token高效语言生成的任务感知低秩后训练

    LOCUS: Task-Aware Low-Rank Post-Training for Token-Efficient Language Generation

    [https://arxiv.org/abs/2609.11739](https://arxiv.org/abs/2609.11739)

    LOCUS通过选择任务感知的低秩适配子空间进行后训练，在不修改偏好对齐损失且仅更新不到0.3%参数的情况下，将大模型输出长度最多缩短39.84%，从而在保持效用的同时大幅降低推理成本。

    

    大语言模型的服务成本与输出序列长度直接成正比，然而标准的偏好对齐方法往往会增加回复的冗长度而不提升实际效用。我们研究后训练更新的参数化方式是否会影响生成长度：低秩子空间可以在不修改对齐损失的情况下改变序列长度。我们提出了LOCUS，一种选择任务感知低秩适配子空间的方法，用于在效用约束下最小化输出token成本。在该子空间内，后训练在冻结骨干网络的前提下保留原生的偏好优化目标。在Anthropic HH-RLHF对话偏好数据上，我们评估了两个约3B规模的解码器骨干模型——Pythia-2.8B和Qwen2.5-3B——并与协议匹配的全参数DPO、DrDPO分支以及已发布的SamPO检查点进行对比。LOCUS在Pythia-2.8B上将续写长度最多减少39.84%，在Qwen2.5-3B上减少14.87%至17.58%，同时仅更新0.24%至0.28%的模型参数。

    arXiv:2609.11739v1 Announce Type: new  Abstract: Large language model serving costs scale directly with output sequence length, yet standard preference alignment often inflates response verbosity without improving utility. We study whether the parameterization of post-training updates affects generation length: low-rank subspaces alter sequence length without modifying the alignment loss. We present LOCUS, a method that selects a task-aware low-rank adaptation subspace to minimize output-token cost subject to a utility constraint. Within this subspace, post-training retains the native preference objective with a frozen backbone. Across Anthropic HH-RLHF dialogue preferences, we evaluate two $\sim$3B decoder backbones, Pythia-2.8B and Qwen2.5-3B, against protocol-matched full-parameter DPO and DrDPO branches and the released SamPO checkpoint. LOCUS reduces continuation length by up to 39.84\% on Pythia-2.8B and by 14.87--17.58\% on Qwen2.5-3B while updating only 0.24--0.28\% of model pa
    
[^23]: ORCH：组织学原理赋能具身人工智能的集体智能

    ORCH: Organizational Principles Enable Collective Intelligence in Embodied AI

    [https://arxiv.org/abs/2609.11737](https://arxiv.org/abs/2609.11737)

    提出ORCH框架，借鉴人类组织理论，通过结合汇集式与顺序式相互依赖，为多达50个异构具身AI智能体构建任务特定的层级化组织结构，从而在野火响应等复杂物理任务中实现集体智能。

    

    集体智能不仅取决于个体成员的能力，还取决于这些成员的组织方式。然而，人工智能多智能体系统通常采用固定的组织结构来组建，即使它们执行的物理任务在协调要求上存在根本差异。本研究表明，人类组织理论中的原理可以被操作化，用于组织大型、异构的具身人工智能智能体集体。我们提出了ORCH（组织角色与协调层级，Organizing Roles and Coordination Hierarchies），该方法通过将适用于可并行开展工作的“汇集式相互依赖”与受先决条件约束的“顺序式相互依赖”相结合，构建针对特定任务的层级化组织结构。在涵盖侦察、救援、运输、资源管理、围控与扑救的25个野火响应任务中，我们评估了由多达50个异构人工智能体组成的团队……（原文摘要截断）

    arXiv:2609.11737v1 Announce Type: cross  Abstract: Collective intelligence depends not only on the capabilities of individual members, but also on how those members are organized. Yet artificial multi-agent systems are typically assembled using fixed organizational structures, even when the physical tasks they perform impose fundamentally different coordination requirements. Here we show that principles from human organization theory can be operationalized to organize large, heterogeneous collectives of embodied artificial agents. We introduce ORCH (Organizing Roles and Coordination Hierarchies), which constructs task-specific hierarchical organizations by combining pooled interdependence for work that can proceed concurrently with sequential interdependence for work governed by prerequisite relationships. Across 25 wildfire-response missions spanning reconnaissance, rescue, transportation, resource management, containment and suppression, we evaluated teams of up to 50 heterogeneous a
    
[^24]: 基于神经控制微分方程的连续时间声学建模

    Continuous-Time Acoustic Modelling with Neural Controlled Differential Equations

    [https://arxiv.org/abs/2609.11725](https://arxiv.org/abs/2609.11725)

    本文提出使用神经控制微分方程（CDEs）实现TTS中时长感知的连续时间声学建模，使隐藏状态的数值能够随语音内容和时长信息连续演化，而不仅仅改变状态出现的位置和频率。

    

    文本转语音（TTS）模型通常通过使用预测的时长将音素级编码器状态扩展为帧级解码器输入，以此解决文本与语音的对齐问题。虽然这种长度调节步骤在结构上解决了对齐问题，但这种对时长的使用通常仅改变潜在状态出现的位置和频率，而不改变状态本身的数值。本文提出了一种利用神经控制微分方程（CDEs）在TTS中进行时长感知声学建模的连续时间机制。我们将音素表示形式化为一条时间参数化的控制路径，并使用神经声学向量场产生一个连续时间的隐藏状态，其数值随语音内容和时长导出的时间信息而演化。所得到的轨迹可以在离散点进行采样，并集成到标准的声学解码器流程中。客观实验结果对比了CDE与典型的循环模型。主观结果表明（摘要原文在此处截断）。

    arXiv:2609.11725v1 Announce Type: cross  Abstract: Text-to-speech (TTS) models commonly address text--speech alignment by expanding phone-level encoder states to frame-level decoder inputs using predicted durations. While this length-regulation step resolves alignment structurally, this use of duration typically changes only where and how often latent states appear, not the values of the states themselves. This paper proposes a continuous-time mechanism for duration-aware acoustic modelling in TTS using neural controlled differential equations (CDEs). We formulate the phone representation as a temporally parameterised control path and use a neural acoustic vector field to produce a continuous-time hidden state whose values evolve with phonetic content and duration-derived timing. The resulting trajectory can be sampled at discrete points and integrated into a standard acoustic decoder pipeline. Objective results contrast CDEs and typical recurrent models. Subjective results suggest tha
    
[^25]: 一种面向全模拟忆阻器脉冲神经网络中向量-矩阵乘法的基于时间的读出方案

    A Time-Based Readout for Vector-Matrix Multiplication in Fully Analog Memristive SNNs

    [https://arxiv.org/abs/2609.11713](https://arxiv.org/abs/2609.11713)

    本文提出一种基于电压-时间转换的全模拟读出架构，用于忆阻器SNN中的向量-矩阵乘法，省去电流模式读出电路，显著提升面积与能效。

    

    人工神经网络依赖于向量-矩阵乘法（VMM），而在冯·诺依曼架构中实现VMM时，存储单元与处理单元之间的高成本数据搬运成为主要瓶颈。脉冲神经网络（SNN）通过利用忆阻器交叉阵列在存储器内部执行模拟VMM，缓解了这一瓶颈。然而，传统的电流模式读出电路会带来显著的面积和功耗开销。本工作提出了一种基于VMM输出电压-时间转换的全模拟读出架构。通过直接感知列电压，该方法避免了电流模式的求和与缩放电路，从而提升了面积效率与能效。基于130 nm CMOS工艺实现的10x1 SNN的后版图仿真验证了所提出的架构，而将其应用于一个训练好的用于手写数字分类的64x10 SNN，进一步证明了该架构在SNN推理中的可行性。

    arXiv:2609.11713v1 Announce Type: cross  Abstract: Artificial neural networks rely on vector-matrix multiplications (VMMs), whose implementation in von Neumann architectures is dominated by costly data movement between memory and processing units. Spiking neural networks (SNNs) mitigate this bottleneck by performing in-memory, analog VMMs using memristive crossbar arrays. However, conventional current-mode readout circuits incur significant area and power overhead.   This work proposes a fully analog readout architecture based on voltage-to-time conversion of the VMM output. By sensing the column voltage, the proposed approach avoids current-mode summing and scaling circuitry, improving area and energy efficiency. Post-layout simulations of a 10x1 SNN implemented in a 130 nm CMOS technology validate the proposed architecture, while application to a trained 64x10 SNN for digit classification further demonstrates its feasibility for SNN inference.
    
[^26]: 当智能体产生分歧时：贝叶斯反向推理作为多智能体集体决策的无标签锚点

    When Agents Disagree: Bayesian Backward Reasoning as a Label-Free Anchor for Multi-Agent Collective Decision-Making

    [https://arxiv.org/abs/2609.11709](https://arxiv.org/abs/2609.11709)

    该论文提出通过贝叶斯反向推理构建与前向推理具有不同分解方式的反向后验，并利用Jensen-Shannon散度衡量智能体间的跨路径一致性进行排序，从而为无标签场景下的多智能体集体决策提供锚点。

    

    当多个大语言模型（LLM）智能体给出相互冲突的答案时，决策过程决定了智能体多样性是提升性能还是仅仅加剧共同的错误。现有的集体决策方法，包括投票、选举规则和LLM裁判，都依赖于前向推理：即以单一方向将证据映射到标签。尽管这些方法可以组合多样的前向推理轨迹，但它们所聚合的估计值仍然共享这种“证据到标签”的分解方式，并可能继承前向池中彼此相关的错误。因此，我们通过从显式似然出发的贝叶斯反向推理，为每个实例构建反向后验分布。前向后验与反向后验提供了对真实后验分布的两种不同分解方式的近似。由于来自不同分解方式的估计往往更少共享相同的错误，我们使用Jensen-Shannon散度按跨路径一致性对智能体进行排序。

    arXiv:2609.11709v1 Announce Type: new  Abstract: When multiple LLM agents yield conflicting answers, the decision-making process dictates whether agent diversity improves performance or merely compounds shared errors. Existing collective decision-making methods, including voting, electoral rules, and LLM judges, rely on forward reasoning: they map evidence to labels in one direction. Although these methods can combine diverse forward traces, they still aggregate estimates that share this evidence-to-label factorization and can inherit correlated errors within the forward pool. We therefore construct a reverse posterior for each instance through Bayesian backward reasoning from an explicit likelihood. The forward and reverse posteriors provide differently factorized approximations of the underlying posterior. Because estimates from different factorizations may tend to share the same error less often, we use Jensen-Shannon divergence to rank agents by cross-path consistency. This cross-p
    
[^27]: 面向B样条曲面拟合的语言增强语义先验

    Language-Augmented Semantic Priors for B-Spline Surface Fitting

    [https://arxiv.org/abs/2609.11708](https://arxiv.org/abs/2609.11708)

    LASP框架利用大语言模型从程序化建模历史中推断出结构化、可供求解器使用的B样条语义先验，弥合了CAD系统中高层设计意图与几何求解器配置之间的鸿沟，实现语义一致的曲面拟合。

    

    B样条与非均匀有理B样条（NURBS）曲面构成了当代计算机辅助设计（CAD）系统的数学基础。尽管取得了长期进展，传统CAD中的几何内核在曲面拟合与参数化方面仍严重依赖预先设定的启发式初始化。与此同时，建模历史中编码的程序化语义与设计意图在几何生成过程中很大程度上被忽略了。这种脱节造成了高层设计意图与求解器可执行的几何配置之间的鸿沟，常常导致次优且语义不一致的拟合结果。为弥合这一鸿沟，我们提出了LASP，一个语言增强语义先验框架，它利用大语言模型（LLM）从程序化建模历史中推断出结构化的、可供求解器使用的B样条先验。LASP并不修改几何内核本身，而是作为一种语义推理机制发挥作用。

    arXiv:2609.11708v1 Announce Type: cross  Abstract: The use of B-splines and Non-Uniform Rational B-Splines surfaces constitutes the mathematical foundation of contemporary computer-aided design (CAD) systems. Despite long-term progress, geometric kernels in traditional CAD still rely heavily on predetermined heuristic initialization for surface fitting and parameterization. Meanwhile, the procedural semantics and design intent encoded in modeling histories are largely ignored during geometry generation. This disconnect creates a gap between high-level design intent and solver-executable geometric configuration, often leading to suboptimal and semantically inconsistent fitting results. To bridge this gap, we introduce LASP, a Language-Augmented Semantic Priors framework that leverages large language models (LLMs) to infer structured, solver-usable B-spline priors from procedural modeling histories. Rather than modifying the geometric kernel itself, LASP operates as a semantic reasoning 
    
[^28]: ActSafeGuard：面向流匹配策略的可微且训练对齐的约束执行方法

    ActSafeGuard: Differentiable and Training-Aligned Constraint Enforcement for Flow-Matching Policies

    [https://arxiv.org/abs/2609.11697](https://arxiv.org/abs/2609.11697)

    ActSafeGuard 提出了一个可微且训练对齐的安全防护层，通过解析式射线缩放算子将硬性动作可行性约束直接融入流匹配策略的训练过程，利用边界感知梯度引导模型自然学习受约束流形，解决了现有安全方法训练与执行不匹配的问题。

    

    视觉-语言-动作模型（VLA）和世界-动作模型（WAM）在通用机器人操作中已展现出强大的能力，但其生成的动作可能违反硬性物理约束，因而在部署时可能不安全或不可行。现有的安全方法要么优化统计性安全目标而缺乏确定性的每步保证，要么仅在推理阶段对不安全动作进行纠正，从而造成策略训练与执行之间的不匹配。我们提出了 ActSafeGuard，一个面向基于流匹配策略的可微且训练对齐的安全防护层。ActSafeGuard 将硬性动作可行性整合到策略学习之中，而不仅仅将安全性视为推理时的外部组件。通过解析式射线缩放算子的设计，ActSafeGuard 使边界感知的梯度能够引导模型自然地学习受约束的流形。在多个标准基础基准上进行了大量实验……

    arXiv:2609.11697v1 Announce Type: cross  Abstract: Vision-Language-Action (VLA) and World-Action Models (WAMs) have demonstrated strong capabilities in general-purpose robotic manipulation, yet their generated actions may violate hard physical constraints and therefore be unsafe or infeasible for deployment. Existing safety approaches either optimize statistical safety objectives without deterministic per-step guarantees or correct unsafe actions only during inference, creating a mismatch between policy training and execution. We introduce ActSafeGuard, a differentiable and training-aligned safeguard layer for flow-matching based policies. ActSafeGuard integrates hard action feasibility into policy learning, not merely treating safety as an inference-time external component. Through an analytical ray-scaling operator design, ActSafeGuard enables boundary-aware gradients to guide the model to naturally learn constrained manifolds. Extensive experiments on multiple standard foundation ba
    
[^29]: COBRA-Skills：基于上下文赌博机引导的智能体技能优化进化

    COBRA-Skills: Contextual Bandit-Guided Evolution for Agent Skill Optimization

    [https://arxiv.org/abs/2609.11682](https://arxiv.org/abs/2609.11682)

    COBRA-Skills提出了一种将智能体技能优化建模为预算受限序贯优化的高效框架，通过上下文赌博机引导的优先级排序与基于证据的技能进化，在六个智能体基准上取得最强平均性能的同时，将优化成本较SkillOpt降低55%-58%。

    

    大语言模型（LLM）智能体可以从先前任务经验中提炼的可复用技能中受益，然而现有的技能优化方法通常依赖昂贵的基于执行的评估和大量任务数据。我们提出了COBRA-Skills，这是一个高效的框架，它将技能优化表述为在动态演化的候选空间上进行预算受限的序贯优化问题。COBRA-Skills将上下文赌博机引导的优先级排序与基于证据的技能进化相结合，选择性地将评估分配给有前景或信息量大的候选技能，同时持续根据执行反馈完善技能种群。在六个异构智能体基准和三个目标模型上，COBRA-Skills在对比方法中始终取得最强的平均性能，同时相比SkillOpt将优化成本降低了55%至58%，且每个基准仅使用50个独特的优化样本。

    arXiv:2609.11682v1 Announce Type: new  Abstract: Large language model (LLM) agents can benefit from reusable skills distilled from prior task experience, yet existing skill optimization methods often rely on costly execution-based evaluation and substantial task data. We introduce \textbf{COBRA-Skills}, an efficient framework that formulates skill optimization as budgeted sequential optimization over a dynamically evolving candidate space. COBRA-Skills couples contextual-bandit-guided prioritization with evidence-grounded skill evolution, selectively allocating evaluations to promising or informative candidates while continually refining the skill population from execution feedback. Across six heterogeneous agent benchmarks and three target models, COBRA-Skills consistently achieves the strongest average performance among compared methods, while reducing optimization cost by 55--58\% relative to SkillOpt and using only 50 unique optimization examples per benchmark. Further analyses sho
    
[^30]: Ecdysis：面向大语言模型智能体的高效且有效的运行时框架训练

    Ecdysis: Efficient and Effective Training of Runtime Harnesses for LLM Agents

    [https://arxiv.org/abs/2609.11677](https://arxiv.org/abs/2609.11677)

    该论文指出运行时框架进化中缺乏有原则的失败诊断是关键瓶颈——观察到的失败可能源于模型自身缺陷或框架系统性缺陷，直接针对个别失败优化会导致不必要的模型特定适应，因此提出Ecdysis方法以实现对LLM智能体运行时框架更高效、更有效且泛化能力更强的训练。

    

    自进化的运行时框架可以显著提升大语言模型（LLM）智能体的能力，并为优化智能体执行提供了一种有前景的范式。现有的框架进化方法通常依赖于迭代搜索，即基于任务实例的执行反馈反复评估和修改候选框架。虽然这种范式能够实现框架的持续优化，但由于需要反复执行智能体和修改代码，会产生巨大的时间开销，并且可能对观察到的任务和特定失败模式过拟合，导致对未见任务的泛化能力下降。我们发现缺乏有原则的失败诊断是框架进化的关键瓶颈：观察到的失败可能反映的是模型特定的缺陷或系统性的框架缺陷，直接针对个别失败进行优化可能导致不必要的模型特定适应。因此我们提出Ecdysis方法……（摘要原文在此处截断）

    arXiv:2609.11677v1 Announce Type: new  Abstract: Self-evolving runtime harnesses can substantially improve the capabilities of large language model (LLM) agents and provide a promising paradigm for optimizing agent execution. Existing harness evolution methods typically rely on iterative search, repeatedly evaluating and revising candidate harnesses based on execution feedback from task instances. While this paradigm enables continuous harness optimization, it incurs substantial time overhead due to repeated agent executions and code modifications, and may overfit to observed tasks and specific failure patterns, resulting in degraded generalization to unseen tasks. We identify the lack of principled failure diagnosis as a key bottleneck in harness evolution: an observed failure can reflect either model-specific deficiencies or systematic harness deficiencies, and directly optimizing against individual failures can lead to unnecessary model-specific accommodation. We therefore propose E
    
[^31]: 地理空间人工智能、Dataverse元数据与基于地方的政府研究

    Geospatial AI, Dataverse Metadata, and the Study of Place-Based Government

    [https://arxiv.org/abs/2609.11674](https://arxiv.org/abs/2609.11674)

    本文从哈佛Dataverse的公开数据和元数据中构建了一个包含21.5万余节点的知识图谱，首次将自由文本形式的地理信息整合为可检索的结构化网络，并识别出数千个与政策直接相关的地理空间数据集。

    

    哈佛Dataverse托管着超过15万个研究数据集，但这些数据集所包含的地理信息是由数据存储者以自由文本形式输入的，从未被整合成可检索的结构。我们利用该存储库的公开数据和元数据构建了一个知识图谱，将102,650个数据集组织在一个拥有215,985个节点和528,003条边的网络中，将数据集与关键词、出版物、学科、期刊和地理位置相互关联。在这些数据集中，43,991个（42.9%）至少包含一个地理空间字段（地理覆盖范围、地理单元或边界框），且96.9%的节点位于单一连通分量中，因此即使数据集的地理空间元数据没有共同之处，数据集之间仍可相互访问关联。通过保守的关键词搜索，我们识别出7,654个带有地理空间标记的数据集（17.4%）与政策直接相关，其中选举和立法机构是最大的集群，其次是政府行政相关数据集。

    arXiv:2609.11674v1 Announce Type: new  Abstract: Harvard Dataverse hosts over 150,000 research datasets, but the geographic information those datasets carry is entered as free text by depositors and has never been assembled into a searchable structure. We construct a knowledge graph from the repository's public data and metadata, organizing 102,650 datasets within a 215,985-node network of 528,003 edges linking datasets to keywords, publications, subjects, journals, and locations. Of those datasets, 43,991 (42.9 percent) carry at least one geospatial field, geographic coverage, geographic unit, or a bounding box and 96.9 percent of all nodes sit in a single connected component, so datasets remain reachable from one another even when their geospatial metadata share nothing in common. A conservative keyword search identifies 7,654 geospatially tagged datasets (17.4 percent) as directly policy-relevant, with elections and legislatures the largest cluster, followed by government administra
    
[^32]: 担保理论

    Warrant Theory

    [https://arxiv.org/abs/2609.11667](https://arxiv.org/abs/2609.11667)

    本文提出“担保理论”这一新的哲学学科，将逻辑重构为规约命题引入、接受、拒绝与推理运用的规范性框架，以“担保”即推理权利为核心概念，并通过接受与拒绝的双边词汇系统分析命题推理地位的产生、发展及其互动关系。

    

    在本文中，我们将担保理论发展为一门关注逻辑分析中命题推理合法性的哲学学科。担保理论将逻辑重新概念化为一种规范性框架，用以规约命题得以被引入、接受、拒绝以及推理性运用的条件。担保被理解为一种推理权利，并与真理、信念及其他心理态度相区分，同时本文还考察了担保与推理性使用及意义之间的关系。随后，本文将担保理论分析发展为一种系统方法，用以研究命题如何获得推理地位、这种地位如何发展，以及不同推理立场如何通过依赖、相容、不相容和排斥等关系相互作用。接受与拒绝为表征正面与负面推理立场及其后果与组合提供了双边词汇。

    arXiv:2609.11667v1 Announce Type: cross  Abstract: In this paper, we develop warrant theory as a philosophical discipline concerned with the inferential legitimacy of propositions within logical analysis. Warrant theory reconceptualises logic as a normative framework governing the conditions under which propositions may be introduced, accepted, rejected, and inferentially employed. Warrant is understood as inferential entitlement and is distinguished from truth, belief, and other psychological attitudes, while its relation to inferential use and meaning is examined. Warrant-theoretic analysis is then developed as a systematic method for investigating how propositions acquire inferential standing, how that standing develops, and how inferential positions interact through relations of dependence, compatibility, incompatibility, and exclusion. Acceptance and rejection provide the bilateral vocabulary for representing positive and negative inferential positions and the consequences and com
    
[^33]: 自主性、社会规范与对齐：迈向自主人工智能体的发展性框架

    Autonomy, Social Norms, and Alignment: Towards a Developmental Framework for Autonomous Artificial Agents

    [https://arxiv.org/abs/2609.11660](https://arxiv.org/abs/2609.11660)

    该论文针对自主人工智能体在开放动态环境中自主探索学习时难以与人类目标保持一致的问题，提出构建一个融合自主性、社会规范与对齐机制的发展性框架。

    

    近年来，人工智能凭借具备泛化能力和复杂输出生成能力的大规模模型取得了非凡的进展。然而，将这种潜力转移到具身智能体上时，暴露出一个显著的局限：最先进的系统依赖于预先存在的数据集和人类反馈策略，这些方法虽然强大，但在动态或未知的环境中却是不够的。为了适应环境，智能体必须通过与其环境的直接交互来获取知识。应对这一挑战的一种策略是引入更高层次的机制，例如利用好奇心和胜任力的内在动机，来引导智能体在复杂环境中的探索与学习。虽然这种灵活性扩大了智能体的自主性，但也使确保智能体与人类目标保持对齐的任务变得更加复杂。对齐对于一般人工智能系统而言已然是一个挑战，而在非结构化和动态环境中则变得更加复杂（原文摘要在此处截断）。

    arXiv:2609.11660v1 Announce Type: new  Abstract: In recent years, artificial intelligence has made extraordinary progress thanks to large-scale models capable of generalization and the generation of complex outputs. However, transferring this potential into embodied agents reveals a significant limitation: the most advanced systems rely on pre-existing datasets and human feedback strategies that are powerful but insufficient in dynamic or unknown contexts. To adapt, an agent must acquire knowledge through direct interaction with its environment. One strategy to address this challenge involves introducing higher-level mechanisms, such as intrinsic motivations, which leverage curiosity and competence, to guide exploration and learning in complex environments. While this flexibility expands autonomy, it complicates the task of ensuring agents remain aligned with human goals. Alignment, already a challenge for artificial systems in general, becomes even more complex in unstructured and dyn
    
[^34]: ZipCodec：超低帧率流式语音编码器

    ZipCodec: Ultra-Low-Frame-Rate Streaming Speech Coding

    [https://arxiv.org/abs/2609.11642](https://arxiv.org/abs/2609.11642)

    ZipCodec是一种以6.25 Hz超低帧率和0.80 kbps比特率运行的流式神经语音编解码器，通过大规模WavLM蒸馏、重新设计的transformer架构、标量球面量化和延迟感知流式解码，在重建质量和下游任务上大幅超越现有流式编解码器，并能在消费级CPU上实现实时单流推理。

    

    神经音频编解码器是现代语音生成系统的基础组件。尽管近期的编解码器实现了越来越低的比特率，但降低帧率仍然具有挑战性，因为每个token必须保留更多信息，同时保持重建质量。我们提出了ZipCodec，这是一个以6.25 Hz帧率和0.80 kbps比特率运行的流式神经语音编解码器，理论延迟为160毫秒。我们的方法将大规模WavLM蒸馏与重新设计的基于transformer的架构、标量球面量化器以及延迟感知的流式解码器相结合。实验表明，ZipCodec在相近比特率下，在重建和下游任务中均大幅超越现有的流式编解码器，同时以显著更低的帧率运行。尽管拥有8.42亿参数，ZipCodec仍能在消费级CPU上实现实时单流推理。演示样本、代码和模型检查点可在 https://lucadellalib. 获取。

    arXiv:2609.11642v1 Announce Type: cross  Abstract: Neural audio codecs are a fundamental component of modern speech generation systems. While recent codecs achieve increasingly low bitrates, reducing frame rate remains challenging, as each token must preserve more information while maintaining reconstruction quality. We present ZipCodec, a streaming neural speech codec operating at 6.25 Hz and 0.80 kbps with a theoretical latency of 160 ms. Our approach combines large-scale WavLM distillation with a redesigned transformer-based architecture, a scalar spherical quantizer, and a latency-aware streaming decoder. Experiments show that ZipCodec substantially outperforms existing streaming codecs at comparable bitrates in both reconstruction and downstream tasks, while operating at a significantly lower frame rate. Despite its 842M parameters, ZipCodec achieves real-time single-stream inference on a consumer-grade CPU. Demo samples, code and checkpoints are available at https://lucadellalib.
    
[^35]: LoaDiff：面向能源分析的用电消费时间序列条件生成

    LoaDiff: Conditional Generation of Electricity Consumption Time Series for Energy Analytics

    [https://arxiv.org/abs/2609.11639](https://arxiv.org/abs/2609.11639)

    LoaDiff是一种基于扩散模型的生成模型，能够根据家庭属性和上下文变量等条件，生成长达一年的逼真智能电表负荷曲线，为能源分析领域提供合规的合成数据替代方案。

    

    能源转型正在通过分布式发电、电气化电器和需求响应计划的日益普及，重塑居民用电模式。理解这些不断演变的行为需要获取精细的智能电表数据，以支撑负荷预测、电器检测和需求侧灵活性分析等应用。然而，此类数据受到严格的访问限制和数据保护法规的约束，因此需要逼真的合成数据作为替代方案。本文提出了LoaDiff，一种基于扩散模型的生成模型，用于生成长达一年的亚小时级智能电表负荷曲线。LoaDiff支持对静态家庭属性（如电器拥有情况）和动态上下文变量（包括日历信息和室外温度）进行灵活的条件生成。我们在三个居民用电数据集上将该模型与多个生成式基线进行了评估。

    arXiv:2609.11639v1 Announce Type: new  Abstract: The energy transition is reshaping residential electricity consumption through the increasing adoption of distributed generation, electrified appliances, and demand-response programs. Understanding these evolving behaviors requires access to granular smart-meter data for applications such as load forecasting, appliance detection, and demand-side flexibility analysis. However, such data are subject to strict access restrictions and data-protection regulations. Thus, realistic synthetic alternatives are necessary. In this paper, we introduce LoaDiff, a diffusion-based generative model for year-long, sub-hourly smart-meter load curves. LoaDiff supports flexible conditioning on static household attributes, such as appliance ownership, and dynamic contextual variables, including calendar information and outdoor temperature. We evaluate the model against multiple generative baselines on three residential electricity-consumption datasets. Our e
    
[^36]: MAPLE：基于语言与进化的记忆增强规划

    MAPLE: Memory-Augmented Planning with Language and Evolution

    [https://arxiv.org/abs/2609.11636](https://arxiv.org/abs/2609.11636)

    MAPLE是一个记忆增强的优化智能体，将自然语言问题构建与数学规划和进化搜索相结合，能够通过连续的自然语言请求动态维护优化问题，同时保留先前决策并重用搜索结果。

    

    领域从业者了解他们的业务约束，但可能缺乏运筹学专业知识或专门的技术支持。基于大语言模型（LLM）的优化智能体能够将自然语言需求转化为成熟的优化工具可以执行的模型或求解器程序。这一进展使优化变得更加易于使用，但现实世界的运营是动态的：不断变化的需求、资源和优先级需要对数据、约束和目标进行更新。以孤立请求为中心的方法在支持快速适应方面能力有限，无法保留先前的决策并重用有用的搜索结果。我们提出了MAPLE（基于语言与进化的记忆增强规划），这是一个通过连续自然语言请求来维护优化问题的智能体。MAPLE将基于语言的问题构建与数学规划和进化搜索相结合。它保留了优化程序、已接受的计划、先前的更新

    arXiv:2609.11636v1 Announce Type: new  Abstract: Domain practitioners understand their business constraints but may lack operations-research expertise or dedicated support. LLM-based optimization agents translate natural-language requirements into models or solver programs that established optimization tools can execute. This progress makes optimization more accessible, but real-world operations are dynamic: changing demand, resources, and priorities require updates to data, constraints, and objectives. Methods centered on isolated requests offer limited support for rapid adaptation that preserves earlier decisions and reuses useful search results. We introduce MAPLE (Memory-Augmented Planning with Language and Evolution), an agent for maintaining optimization problems through successive natural-language requests. MAPLE combines language-based problem construction with mathematical programming and evolutionary search. It retains the optimization program, accepted plans, earlier updates
    
[^37]: 物理信息神经网络用于推断仿星器装置刮削层中的垂直能量传导率

    Physics-Informed Neural Networks to Infer the Perpendicular Energy Conductivity in the Scrape-Off Layer of Stellarator Devices

    [https://arxiv.org/abs/2609.11628](https://arxiv.org/abs/2609.11628)

    本文开发了一个反向物理信息神经网络框架，通过将等离子体密度和温度的径向分布测量数据与简化的刮削层输运方程相结合，成功推断出仿星器装置刮削层中垂直热传导率对等离子体密度和温度的依赖关系。

    

    在这项工作中，我们开发了一个反向物理信息神经网络（PINN）框架，用于推断刮削层（SOL）垂直热传导率对等离子体密度和温度的依赖关系 κ⊥(n,T)。该方法将电子密度和温度的径向分布测量数据与简化的径向SOL输运方程的残差相结合，使得推断出的传导率同时受到测量结果和底层输运模型的约束。三个神经网络被同时训练：其中两个重建温度和密度分布作为径向坐标和输运功率的函数，第三个则表示有效传导率作为局部密度和温度的函数。该框架首先使用由预设传导率函数生成的合成数据进行验证，从而可以将推断出的 κ⊥(n,T) 与真实值进行直接比较。

    arXiv:2609.11628v1 Announce Type: cross  Abstract: In this work, we develop an inverse Physics-Informed Neural Network (PINN) framework to infer the dependence of the scrape-off layer (SOL) perpendicular heat conductivity on plasma density and temperature, $\kappa_\perp(n,T)$. The method combines radial profile measurements of electron density and temperature with the residual of a reduced one-dimensional SOL transport equation, so that the inferred conductivity is constrained by both the measurements and the underlying transport model. Three neural networks are trained simultaneously: two reconstruct the temperature and density profiles as functions of the radial coordinate and transported power, while a third represents the effective conductivity as a function of the local density and temperature. The framework is first validated using synthetic data generated from a prescribed conductivity function, allowing the inferred $\kappa_\perp(n,T)$ to be compared directly with the ground tr
    
[^38]: 基于带逆模型的模型强化学习的模块化生产系统分布式优化

    Distributed Optimization of Modular Production Systems using Model-based Reinforcement Learning with Inverse Models

    [https://arxiv.org/abs/2609.11615](https://arxiv.org/abs/2609.11615)

    本文提出一种在强化策略训练中引入近似逆过程模型的模型强化学习框架，通过解耦执行动力学与状态空间动力学使训练仅在任务空间进行，在模块化生产测试平台上显著提升了性能和训练速度，尤其对离策略算法效果更佳。

    

    本文提出了一种用于高度灵活的模块化制造系统的数据驱动自学习控制的新方法。具体而言，我们采用了一种新颖的基于模型的强化学习框架，该框架在强化策略的训练中引入了近似的逆过程模型。这种方法将执行动力学与状态空间动力学解耦，使得基于强化学习的训练仅在任务空间中进行。我们提出了一种用于近似逆模型的轻量级前馈架构，并将其集成到标准强化学习算法的策略网络中。我们将该方法应用于包含异构生产模块的实验室模块化生产测试平台。结果凸显了模块化制造单元在性能和训练速度方面的效率提升，尤其是对于离策略算法。

    arXiv:2609.11615v1 Announce Type: cross  Abstract: This paper presents a novel approach for data-driven self-learning control of highly flexible, modular manufacturing systems. Specifically, we employ a novel framework for model-based reinforcement learning which introduces approximate inverse process models within the training of reinforcement policies. This approach disentangles the learning of actuation dynamics and the dynamics in state space, resulting in RL-based training solely within the task space. We propose a lightweight feedforward architecture for approximate inverse models and integrate them within the policy network of standard RL algorithms. We apply the approach to a laboratory modular production testbed with heterogeneous production modules. The results underline the efficiency improvements for modular manufacturing units in terms of both performance and training speed, particularly for off-policy algorithms.
    
[^39]: 让另类数据发挥作用：用于金融预测的上下文增强大语言模型

    Making Alternative Data Work: Context-Augmented LLMs for Financial Forecasting

    [https://arxiv.org/abs/2609.11607](https://arxiv.org/abs/2609.11607)

    该论文提出利用上下文增强的大语言模型来整合分散在异构渠道中的另类数据，从而克服传统预测方法难以灵活运用企业级另类数据进行财务表现预测的局限。

    

    在预测企业未来财务表现时，另类数据——从消费者交易、网络流量和预测市场等非传统来源收集的数据——能够提供关于企业经营活动和更广泛市场状况的及时信号。这些信号可能揭示传统公开来源未能捕捉的信息，因此可以为预测企业未来财务表现提供补充信息。然而，企业层面的另类数据通常历史覆盖范围有限，仅与特定预测目标或部分企业子集相关，且分布在众多异构渠道中，使其难以灵活地融入传统预测方法。与此同时，大语言模型（LLMs）能够理解指令、从上下文示例中学习，并通过整合异构信息生成预测，而无需针对特定任务进行训练……

    arXiv:2609.11607v1 Announce Type: new  Abstract: When forecasting a firm's future financial performance, alternative data - data collected from non-traditional sources such as consumer transactions, web traffic, and prediction markets - can provide timely signals about firms' operating activities and broader market conditions. These signals may reveal information that is not captured by traditional public sources and can therefore provide complementary information for forecasting firms' future financial performance. However, firm-level alternative data often have limited historical coverage, are relevant only to specific prediction targets or subsets of firms, and are distributed across numerous heterogeneous channels, making them difficult to incorporate flexibly into conventional forecasting approaches. Meanwhile, large language models (LLMs) can interpret instructions, learn from in-context examples, and generate predictions by combining heterogeneous information without task-specif
    
[^40]: 学习实体本身，而非文件：面向CAD边界表示神经网络的规范输入

    Learn the Solid, Not the File: Canonical Inputs for Neural Networks on CAD Boundary Representations

    [https://arxiv.org/abs/2609.11573](https://arxiv.org/abs/2609.11573)

    该论文揭示了现有B-rep编码器面对同一实体的不同表示时性能会灾难性崩溃，并提出了一种从实体本身派生节点、特征和坐标系的规范区域图输入表示，使神经网络对B-rep变化具有不变性。

    

    边界表示（B-rep）是现代CAD系统用于参数化3D模型的标准格式。事实证明，同一个实体可以由不同的B-rep来表示：例如，两名工程师使用不同的操作、几何内核重建文件、以及导出设置对面的重新划分，都会导致不同的B-rep，尽管底层的实体保持不变。我们证明了现有的B-rep编码器对于同一实体的不同B-rep变化不具备鲁棒性，这些变化包括在标准基准上施加的扰动、CAD软件固有的自然变化，以及设计师对同一零件采用的不同建模方式——为此我们创建了一个基于FreeCAD的人工数据集。主流B-rep编码器的性能往往会灾难性地崩溃。我们提出了规范区域图，这是一种输入表示，其节点、特征和坐标系均派生自实体本身，并展示了理论上的不变性。

    arXiv:2609.11573v1 Announce Type: cross  Abstract: Boundary representation (B-rep) is the standard format used by modern CAD systems for parametric 3D models. It turns out, the exact same solid can be represented by different B-reps: for example, two engineers using different operations, a geometry kernel rebuilding the file, and an export setting repartitioning faces will lead to different B-reps even though the underlying solid remains the same.   We show that existing B-rep encoders are not robust to variation in the B-rep with the same solid on perturbations applied to standard benchmarks, naturally occurring variations inherent to CAD software, and differences in how designers model the same part via a human dataset we created in FreeCAD. The performance of popular B-rep encoders often collapses catastrophically.   We propose the canonical region graph, an input representation whose nodes, features and coordinate frame are derived from the solid itself and show theoretical invaria
    
[^41]: 借助 EXplore Your Graphs ENgine（EXYGEN）引擎实现大规模知识图谱理解

    Enabling Knowledge Graph Understanding at Scale with the EXplore Your Graphs ENgine (EXYGEN)

    [https://arxiv.org/abs/2609.11569](https://arxiv.org/abs/2609.11569)

    EXYGEN 框架通过将自动推导的元数据（VoID 描述和 ShEx 模式）融入检索增强生成（RAG）流水线，无需微调即可让大语言模型实现对大规模知识图谱的对话式访问和文本到 SPARQL 查询生成，在 SciQA 基准上达到 0.419 的精确匹配率。

    

    我们提出了 EXYGEN（EXplore Your Graphs ENgine），一个用于知识图谱（KG）理解的框架，能够实现对大规模知识图谱的对话式访问。我们依次探讨了两个问题。第一，在不进行任务特定微调的情况下，仅依靠自动推导的结构化元数据和小规模图谱样本，大语言模型（LLM）进行文本到 SPARQL 生成的效果如何？我们将 VoID 描述和 ShEx 模式集成到检索增强生成（RAG）流水线中，并在 SciQA 基准上对源自知识图谱的上下文进行了消融实验。我们最佳的配置——结合 ShEx 模式、检索到的三元组以及示例问题-查询对——在无需任何 LLM 微调的情况下，在查询执行结果上达到了 0.419 的精确匹配率。我们进一步发现，诸如 F1 之类的词面指标对查询正确性的预测能力较差，且一旦提供了足够的上下文，更大的通用 LLM 可以超越更小的代码专用模型。第二，我们探讨如何生成结构……（原文摘要在此处截断）

    arXiv:2609.11569v1 Announce Type: cross  Abstract: We present EXYGEN (EXplore Your Graphs ENgine), a framework for knowledge graph (KG) understanding that enables conversational access to KGs at scale. We address two questions in sequence. First, how effectively can LLMs perform text-to-SPARQL generation given only automatically derived structured metadata and small graph samples, rather than task-specific fine-tuning? We integrate VoID descriptions and ShEx schemas into a retrieval-augmented generation (RAG) pipeline and ablate KG-derived context on the SciQA benchmark. Our best configuration -- combining ShEx schemas, retrieved triples, and example question-query pairs -- reaches an exact match of 0.419 on execution results without any LLM fine-tuning. We further find that lexical metrics such as F1 poorly predict query correctness, and that larger general-purpose LLMs can outperform smaller code-specialized ones once given sufficient context. Second, we ask how to generate the struc
    
[^42]: 用于黑色素瘤检测的预训练卷积神经网络比较评估

    A Comparative Evaluation of Pre-trained Convolutional Neural Networks for Melanoma Detection

    [https://arxiv.org/abs/2609.11550](https://arxiv.org/abs/2609.11550)

    该论文对多种预训练卷积神经网络在黑色素瘤检测任务中进行了比较评估，旨在为皮肤镜和组织病理学等不同成像方式确定最合适的网络架构。

    

    黑色素瘤的早期诊断对于提高患者生存率至关重要。然而，由于各类皮肤病变之间高度的视觉相似性以及图像采集条件的差异性，准确区分黑色素瘤与其他皮肤病变仍然是一项重大的临床挑战。人工智能，特别是机器学习，已成为辅助皮肤病诊断的一种有前景的工具，它能够从医学图像中自动提取特征。在现有的各种方法中，卷积神经网络（CNN）在图像分类任务中展现出了卓越的性能，鉴于其能够捕获与病变特征描述相关的分层视觉模式，因此非常适合分析皮肤镜图像和组织病理学图像。然而，尽管已经提出了众多预训练的CNN架构，如何针对特定的成像方式选择最合适的架构仍然是一个悬而未决的问题。

    arXiv:2609.11550v1 Announce Type: cross  Abstract: Early diagnosis of melanoma is critical for improving patient survival rates. However, accurately distinguishing melanoma from other skin lesions remains a significant clinical challenge due to the high visual similarity among lesion types and variability in image acquisition conditions. Artificial intelligence, particularly machine learning, has emerged as a promising tool to support dermatological diagnosis by automating feature extraction from medical images. Among the available approaches, convolutional neural networks (CNNs) have demonstrated strong performance in image classification tasks, making them well-suited for analyzing both dermatoscopic and histopathological images, given their ability to capture hierarchical visual patterns relevant to lesion characterization. Nevertheless, despite numerous pre-trained CNN architectures having been proposed, selecting the most appropriate one for a given imaging modality remains an ope
    
[^43]: 面向功率灵活AI训练的任务功率弹性表征

    Characterizing Job Power Elasticity for Power-Flexible AI Training

    [https://arxiv.org/abs/2609.11542](https://arxiv.org/abs/2609.11542)

    本文首次系统表征了LLM训练中的任务功率弹性，并提出功率灵活性指数（PFI）这一归一化指标来量化GPU功率降低的性能代价，为SLA感知的功率灵活性提供了控制原语。

    

    大语言模型（LLM）训练是现代数据中心电力需求增长最快的来源之一，电力可用性是AI基础设施持续增长的主要瓶颈。使这些工作负载的功耗变得灵活，可以为AI增长释放额外的电力、限制电价上涨，并提高现有电网基础设施的利用率。然而，要实现这种灵活性，我们必须首先了解当GPU功率降低时训练工作负载的性能会如何变化。本文首次对LLM训练中的“任务功率弹性”（即吞吐量对功率降低的敏感度）进行了系统表征。为了量化弹性，我们引入了功率灵活性指数（PFI），这是一个归一化指标，用于量化功率降低带来的性能代价，并为SLA感知的功率灵活性提供了一个控制原语。

    arXiv:2609.11542v1 Announce Type: new  Abstract: Large language model (LLM) training is among the fastest-growing sources of electricity demand in modern data centers, and power availability is a primary bottleneck to continued AI infrastructure growth. Making the power consumption of these workloads flexible could unlock additional power for AI growth, limit increases in electricity prices, and improve the utilization of existing grid infrastructure. However, to realize this flexibility, we must first understand how the performance of training workloads changes when GPU power is reduced.   This paper presents the first systematic characterization of \emph{job power elasticity} (the sensitivity of throughput to power reductions) in LLM training. To quantify elasticity, we introduce the \emph{Power Flexibility Index (PFI)}, a normalized metric that quantifies the performance cost of power reductions and provides a control primitive for SLA-aware power flexibility.   We collect data from
    
[^44]: 提示词修订作为文生图系统中文化偏见的来源

    Prompt Revision as a Source of Cultural Bias in Text-to-Image Systems

    [https://arxiv.org/abs/2609.11532](https://arxiv.org/abs/2609.11532)

    该研究提出WORLDVIEW多语言基准，首次审计了文生图系统中用户不可见的提示词隐性修订层，揭示其是非西方和非英语文化偏见的重要来源，这些文化语境被系统性地过度标记、词汇扁平化并固化为刻板印象。

    

    商业文生图系统在生成图像之前会静默地修改用户提示词，而用户通常无法禁用甚至看不到这一步骤。然而，现有的文化偏见审计只检查最终生成的图像，并将生成过程视为单一流程，因此无法判断偏见的来源。我们提出了WORLDVIEW，这是一个多语言基准，涵盖15种语言、31种语言-文化配对的8,960个提示词。利用该基准，我们通过三步分析审计了三个系统（DALL-E-3、Imagen-4、GPT-Image-1.5）中的修订层：该层对每种文化语境的标记程度有多深、是否将文化语境扁平化为狭窄的词汇集，以及这些词汇是否具有刻板印象。相对于无文化语境的英语基线，美国是被标记最少的语境，而非西方和非英语语境则被标记得多得多，被扁平化为应用于主题各异提示词的狭窄词汇集，并被简化为可识别的刻板线索。

    arXiv:2609.11532v1 Announce Type: new  Abstract: Commercial text-to-image systems silently revise user prompts before generating images, a step users typically cannot disable or even see. Yet, existing audits of cultural bias examine only the final images and treat generation as a single pipeline, so they cannot tell where the bias originates. We introduce WORLDVIEW, a multilingual benchmark of 8,960 prompts across 15 languages and 31 language-context pairings. Using it, we audit the revision layer in three systems (DALL-E-3, Imagen-4, GPT-Image-1.5) through a three-step analysis of how heavily it marks each cultural context, whether it flattens that context into a narrow vocabulary, and whether that vocabulary is stereotypical. Relative to a no-context English baseline, the US is the least-marked context, while non-Western and non-Anglophone contexts are marked far more heavily, flattened into narrow vocabularies applied across topically diverse prompts, and reduced to recognizable cu
    
[^45]: 面向大学生方程式无人赛的基于随机森林的轻量级激光雷达锥桶检测框架

    Lightweight LiDAR-Based Cone Detection Framework Using Random Forest for Formula Student Driverless

    [https://arxiv.org/abs/2609.11527](https://arxiv.org/abs/2609.11527)

    本文提出了一种面向大学生方程式无人赛的轻量级纯激光雷达锥桶检测框架，利用随机森林分类并专为CPU设计，在纯CPU硬件上实现98.33%的F1分数和3.13毫秒的端到端运行时间，并开源了数据集、标注工具和模型以支持复现。

    

    可靠、低延迟的感知对于大学生方程式无人赛（Formula Student Driverless）车辆至关重要，然而许多现有流水线依赖深度学习和多传感器融合，通常需要GPU加速。本文提出了一种专为CPU执行而设计的轻量级纯激光雷达感知流水线，结合了地面点去除、基于IMU的运动补偿、DBSCAN聚类以及基于几何特征的随机森林分类。特征重要性分析将模型输入从12个特征减少到7个，同时保持了性能。在从真实FSD赛事中收集的2,371个标注聚类上进行评估，该流水线在纯CPU硬件上实现了98.33%的F1分数和3.13毫秒的端到端运行时间。所发布的数据集、标注工具和训练好的模型为其他资源受限的自动驾驶赛车队提供了实用且可复现的基线。

    arXiv:2609.11527v1 Announce Type: new  Abstract: Reliable, low-latency perception is crucial for Formula Student Driverless vehicles, yet many existing pipelines rely on deep learning and multi-sensor fusion, often requiring GPU acceleration. This paper presents a lightweight LiDAR-only perception pipeline tailored for CPU execution, combining ground removal, IMU-based motion compensation, DBSCAN clustering, and geometric feature-based Random Forest classification. Feature importance analysis reduced the model input from 12 to 7 features while preserving performance. Evaluated on 2,371 labeled clusters collected from real FSD events, the pipeline achieves an F1-score of 98.33% and an end-to-end runtime of 3.13 ms on CPU-only hardware. The released dataset, labeling tool, and trained models provide a practical and reproducible baseline for other resource-constrained autonomous racing teams.
    
[^46]: 学习图像与布局先验之间的交互以实现设计模板中的图像-布局联合生成

    Learning Interaction between Image and Layout Priors for Joint Image-Layout Generation in Design Templates

    [https://arxiv.org/abs/2609.11519](https://arxiv.org/abs/2609.11519)

    提出InterIL模型，通过可学习通信模块连接预训练的图像与布局扩散模型，在单一生成过程中联合生成背景图像和前景布局，显式建模两者的双向交互，从而提升设计模板的生成质量。

    

    在本文中，我们研究平面设计模板创作的问题，即根据输入文本生成背景图像以及覆盖在背景上的前景元素布局，从而形成和谐的整体构图。以往关于平面设计生成的工作大多采用顺序（序列式）范式，即依次生成设计元素。我们认为这种顺序方案无法忠实地捕捉背景与布局之间的依赖关系（因而也无法捕捉图像-布局的联合分布），这限制了所生成设计模板的质量。为克服这一局限，我们提出了一个名为 InterIL 的模型，它在单一生成过程中联合生成背景图像和布局这两种模态。我们联合模型的新颖设计在于将预训练的图像扩散模型和布局扩散模型的骨干网络通过一个可学习的通信模块相连接，以显式地建模图像与布局之间的双向交互。

    arXiv:2609.11519v1 Announce Type: cross  Abstract: In this paper, we address the problem of graphic design template creation, which generates a background image and a layout of foreground elements over the background to form a harmonious composition from an input text.   Prior work on graphic design generation mostly adopts a sequential paradigm, where design elements are generated sequentially. We argue that such a sequential scheme falls short of faithfully capturing the dependency between the background and layout (and thus the joint image-layout distribution), which limits the quality of generated design templates.   To overcome this limitation, we propose a model, InterIL, which jointly generates the two modalities, background image and layout, in a single generative process. The novel design of our joint model connects the backbones of pretrained image and layout diffusion models with a learnable communication module to explicitly model bidirectional image-layout interaction. Dur
    
[^47]: 扩展SMT求解的非基态子句学习

    Extending SMT Solving with Non-Ground Clause Learning

    [https://arxiv.org/abs/2609.11509](https://arxiv.org/abs/2609.11509)

    该论文提出了一种结合基例实例化、CDCL(T)风格规则与非基态冲突分析的SMT求解演算，通过在原始非基子句上执行归结步骤来学习更一般且非冗余的子句，从而实现可能指数级更短的证明。

    

    量词实例化是目前非基态SMT求解的主要方法：求解器生成基例，并使用CDCL(T)风格的推理来求解由此产生的基态SMT问题。当发现冲突时，冲突分析仅学习一个基子句，即使该冲突来源于非基子句的实例。然而，非基态推理可以给出比纯基态推理指数级更短的证明。我们提出了一种由基例实例化、CDCL(T)风格规则和非基态冲突分析组成的演算系统。求解器在基例上进行推理，但冲突分析的归结步骤是在其原始的非基子句上执行的。这样可以学习到通常比基态冲突更一般的子句。采用合适的策略，学习到的子句甚至是非冗余的。我们还展示了如何将时序回溯纳入SMT求解。我们的演算提供了一个统一的框架……

    arXiv:2609.11509v1 Announce Type: new  Abstract: Quantifier instantiation is currently the main approach to non-ground SMT solving: solvers generate ground instances and solve the resulting ground SMT problems with CDCL(T)-style reasoning. When a conflict is found, conflict analysis learns only a ground clause, even though the conflict comes from instances of non-ground clauses. Yet non-ground reasoning can give exponentially shorter proofs than purely ground reasoning. We propose a calculus that consists of ground instantiations, CDCL(T)-style rules, and non-ground conflict analysis. The solver reasons on ground instances, but the resolution steps of conflict analysis are performed on their original non-ground clauses. This produces learned clauses that are typically more general than the ground conflict. With a suitable strategy, the learned clauses are even non-redundant. We also show how chronological backtracking can be included in SMT solving. Our calculus gives a common setting 
    
[^48]: 面向数据高效语言学习的结构先验

    Structural priors for data-efficient language learning

    [https://arxiv.org/abs/2609.11505](https://arxiv.org/abs/2609.11505)

    该研究发现先在音乐、概率文法和元胞自动机等符号数据上训练模型，可为后续语言建模带来更低损失和更小的权重偏移，但这种损失优势并不能稳定转化为更好的下游语言能力。

    

    高效的语言学习需要减少对大量数据和计算资源依赖的方法。我们研究了结构迁移：先在非语言数据上训练模型，从而诱导出对自然语言有用的先验。这种方法是面向多语言语言建模的一种权重初始化形式。我们通过下一个词预测损失、模型中的权重偏移以及下游语言基准来评估迁移效果。若干符号数据类型——尤其是音乐、概率文法和元胞自动机——相比随机初始化能产生更低的语言建模损失。这些收益与后续语言训练期间更小的权重偏移相吻合，表明结构迁移将模型定位在参数空间中更有利的区域。然而，更低的损失并不总是能一致地转化为更好的下游语言表现，而且从非语言数据迁移的效率低于附加迁移方法。

    arXiv:2609.11505v1 Announce Type: new  Abstract: Efficient language learning requires methods to reduce the reliance on large data and computational resources. We investigate structural transfer: First training models on non-language data to induce useful priors for natural language. This approach is a form of weight initialization for multilingual language modeling. We evaluate transfer via next-token-prediction loss, weight shifts in the model, and downstream linguistic benchmarks. Several symbolic data types - notably music, probabilistic grammars, and cellular automata - yield lower language-modeling loss than random initialization. These gains coincide with smaller weight shifts during subsequent language training, suggesting that structural transfer positions models in a more favorable region of the parameter space. However, a lower loss does not translate consistently into better downstream linguistic performance, and transfer from non-language data is less efficient than additi
    
[^49]: ActMap：基于生成时激活图的单次传递不确定性量化

    ActMap: Single-Pass Uncertainty Quantification from Generation-Time Activation Maps

    [https://arxiv.org/abs/2609.11498](https://arxiv.org/abs/2609.11498)

    ActMap提出了一种白盒激活图表示方法，将生成时所有层、所有词元的隐藏状态轨迹压缩为固定尺寸的低开销张量，从而在单次生成过程中零额外开销地实现大型语言模型的不确定性量化。

    

    arXiv:2609.11498v1 公告类型：新论文 摘要：面向大型语言模型的实用不确定性量化（UQ）必须能够从单次生成中判断某个特定答案是否可信。现有方法要么需要采样多次生成结果，要么仅读取输出词元的概率，要么将模型内部计算压缩为单一隐藏状态。我们提出了ActMap，这是一种白盒表示方法，它将生成过程中的隐藏状态轨迹（每一层、每个生成的词元）压缩为一个固定的12×32×128张量，由时间统计通道构成，能够保留跨Transformer深度和池化隐藏坐标的结构信息。该激活图在生成过程中捕获，没有可测量的额外开销，在模型深度和隐藏维度上具有固定的形状，仅占用96 KiB：这是一个紧凑的工件，可以保留下来用于审计相关的生成过程并直接探测，并通过遮挡分析将分类器的信号定位到中层深度（原文摘要在此处截断）。

    arXiv:2609.11498v1 Announce Type: new  Abstract: Practical uncertainty quantification (UQ) for large language models must decide,   from a single generation, whether a specific answer should be trusted. Existing   methods either sample multiple generations, read only output-token probabilities,   or reduce the model's internal computation to a single hidden state. We introduce   ActMap, a white-box representation that compresses the generation-time hidden  state trajectory (every layer, every generated token) into a fixed $12 \times 32   \times 128$ tensor of temporal-statistic channels that preserves structure across   transformer depth and pooled hidden coordinates. The map is captured during the   generation pass with no measurable overhead, has a fixed shape across model   depths and hidden sizes, and occupies 96 KiB: a compact artifact that can be   retained for audit-relevant generations and probed directly, with occlusion   analysis localizing the classifier's signal to mid-de
    
[^50]: 从文档孤岛到流程智能：面向CMC工艺开发的多层知识图谱

    From Document Silos to Process Intelligence: A Multi-Layer Knowledge Graph for CMC Process Development

    [https://arxiv.org/abs/2609.11493](https://arxiv.org/abs/2609.11493)

    该论文提出一个模块化智能体AI平台，将CMC工艺开发中异构格式的文档转化为可查询的双层知识图谱，实现了从药物发现到商业化生产全流程的知识整合与可追溯性。

    

    化学、制造与控制（CMC）工艺开发在从药物发现到商业化生产的多阶段、知识密集型连续过程中产生了海量的技术信息。传统上，这些知识分散在不同职能部门和异构格式之中，导致技术转移和监管申报过程中出现可追溯性缺口和高昂的知识管理成本。我们提出了一个模块化的智能体AI平台，将异构的工艺开发文档语料库转换为可查询的双层知识图谱。基础知识层通过对数字、扫描、手写及多语言文档的无损摄取，构建具有“文档-章节-文本块”层级的词法图谱；智能层则提取与本体对齐的实体，并通过溯源锚定的领域图谱桥接跨文档概念。大语言模型（LLM）智能体在两个层级上运行，选择……

    arXiv:2609.11493v1 Announce Type: new  Abstract: Chemistry, Manufacturing and Controls (CMC) process development generates an enormous body of technical information across a multi-stage, knowledge-intensive continuum from drug discovery to commercial manufacturing. This knowledge is traditionally fragmented across functions and heterogeneous formats, causing traceability gaps and significant knowledge-management costs during technology transfer and regulatory filing. We present a modular agentic-AI platform that converts a heterogeneous corpus of process-development documents into a queryable, dual-layer knowledge graph. A base knowledge layer builds a lexical graph with a Document-Section-Chunk hierarchy through lossless ingestion of digital, scanned, handwritten, and multilingual documents, while an intelligence layer extracts ontology-aligned entities and bridges cross-document concepts through a provenance-anchored domain graph. LLM agents operate across both layers, selecting the 
    
[^51]: 已发布的去学习数值随检查点而变动，并非因为被删除的数据仍然存活：对263个已发布的批量归一化检查点的审计

    Published Unlearning Numbers Move Per Checkpoint, and Not Because the Removed Data Survives: An Audit of 263 Released Batch-Normalized Checkpoints

    [https://arxiv.org/abs/2609.11490](https://arxiv.org/abs/2609.11490)

    对263个已发布的批量归一化检查点的审计表明，重新拟合批归一化统计量会使部分检查点的已发布去学习数值发生显著变动，但这种变动源于检查点发布状态与重新拟合结果之间的漂移，而非被删除数据在模型中的残留存活，因此对已发布去学习结论的影响真实但有限。

    

    一次去学习审计读取其结论所依据的数值，来自去学习后的模型及其重新训练参照模型各自发布的数字，而两者还都会附带批量归一化统计量——这些统计量没有任何梯度步骤写入，也没有任何发布记录。在比特级完全相同的权重下，用保留数据重新拟合这些统计量，会使221个已发布检查点中的47个超出其自身发布所用种子所显示的波动范围，其中若干个位于某个平均值并不变动的方法之内：变动的是检查点的属性，而非其方法的属性。导致变动的原因并非被删除的数据残留在模型状态之中：在固定的拟合池内将保留记录替换为被删除记录，对已发布数值的移动几乎为零，而检查点所发布状态与任何重新拟合结果之间的漂移距离确实能追踪这种变动。其对已发布决策的影响是真实但有限的：十二个结论发生翻转，四个通过了实测的重新校准预算，两个在每次重复实验中都通过，还有一个我们训练并定位于其自身临界值附近的群体……（摘要在此处截断）

    arXiv:2609.11490v1 Announce Type: cross  Abstract: An unlearning audit reads its verdict off numbers that an unlearned model and its retrained reference each publish, and both also ship batch-normalization statistics that no gradient step wrote and no release records. Refitting them on kept data at bit-identical weights moves 47 of 221 released checkpoints past the spread their own release's seeds show, several inside a method whose average does not move: what moves is the checkpoint's property, not its method's. What does the moving is not the removed data surviving in the state: exchanging kept records for removed ones inside a fixed fitting pool moves a published cell by almost nothing, while how far a checkpoint's shipped state has drifted from any refit does track it. The consequence for a published decision is real but narrow: twelve verdicts cross, four clear a measured recalibration budget, two clear it on every replicate, and a population we trained and sited near its own crit
    
[^52]: 惯例差距：迈向合作AI评估中隐性交流的度量

    The Convention Gap: Towards Measuring Implicit Communication in Cooperative AI Evaluation

    [https://arxiv.org/abs/2609.11489](https://arxiv.org/abs/2609.11489)

    该论文提出“惯例差距”这一新指标，通过在Hanabi游戏中对比从字面交流预测的失败概率与实际失败率，发现人类玩家在合作中依赖超出字面信息的隐性惯例（差距达26.2个百分点），而AI-AI对局中不存在这种差距，为评估合作AI的隐性交流能力提供了可精确计算的方法。

    

    合作型AI智能体通常与其他AI进行对比评估，然而人类的合作依赖于隐性惯例——即解读超出字面信息含义的共享协议——而AI-AI基准测试可能无法捕捉到这一点。我们提出了“惯例差距”这一概念，即从交流字面内容所预测的失败概率与实际观察到的失败率之间的差异，作为衡量隐性交流的指标。在纸牌游戏Hanabi中，有限的牌堆和确定性的提示约束使得该后验概率可以被精确计算。我们重放了来自三个公开数据集的约101,000个出牌动作，涵盖人与人、AI与AI、人与AI的对局。结果显示，差距在人与人组队中为+26.2个百分点，在AI与AI组队中为-0.7个百分点，在人与AI组队中为+16.4个百分点，且该差距集中在未收到任何提示的牌的出牌行为上（人与人组队中为+46个百分点）。在人与AI的对局中，人类可获得的字面信息……

    arXiv:2609.11489v1 Announce Type: new  Abstract: Cooperative AI agents are evaluated against other AIs, yet human cooperation relies on implicit conventions---shared protocols for reading meaning beyond the literal message---which AI-AI benchmarks may not capture. We propose the \emph{convention gap}, the difference between the failure probability predicted from the literal content of communication and the observed failure rate, as a metric of implicit communication. In the card game Hanabi, the finite deck and deterministic hint constraints make this posterior exactly computable. We replayed about 101,000 play actions from three public datasets of human-human (hanab.live), AI-AI (HOAD), and human-AI (HanabiData) games. The gap was +26.2 percentage points (pp) in human pairs, $-$0.7~pp in AI pairs, and +16.4~pp in human-AI pairs, and was concentrated on plays of cards that had received no hints (+46~pp in human pairs). Within human-AI play, the literal information available to humans w
    
[^53]: 灵活且可解释的口音距离测量方法

    Flexible and Interpretable Accent Distance Measurements

    [https://arxiv.org/abs/2609.11458](https://arxiv.org/abs/2609.11458)

    本文提出利用发音倒演生成的发音表征结合最优传输框架，实现了一种既灵活适用于任意语音录音、又具备可解释性的口音距离测量新方法。

    

    确定两个说话者口音之间的差异是语言学和语音技术研究的一项基础任务。用于测量这些差异的方法取决于特定的研究领域。语音学研究者可能通过比较单个单词配对录音中的元音共振峰来展示口音差异。这些结果是可解释的，但录音的收集非常耗时，且可能无法代表连贯语音。带口音的文本转语音（TTS）研究已趋向于使用从口音分类任务中导出的口音嵌入。这些嵌入可以从任何语音录音中生成，但不易解释。在本文中，我们证明了通过发音倒演创建的发音表征可以用作口音比较的可解释基础，并且最优传输为跨任意口音的比较提供了一个框架。

    arXiv:2609.11458v1 Announce Type: new  Abstract: Determining the differences between two speakers' accents is a fundamental task in linguistics and speech technology research. The methodology used to measure these differences depends on the specific research area. A phonetics researcher may demonstrate accent variation by comparing vowel formants in paired recordings of individual words. These results will be interpretable, but the recordings will be time-consuming to collect and may not be representative of connected speech. Accented Text-to-Speech (TTS) research has pushed towards using accent embeddings derived from accent classification tasks. These embeddings can be produced from any speech recording, but are not readily interpretable. In this paper, we demonstrate that articulatory representations created through articulatory inversion can be used as an interpretable basis for accent comparison and that optimal transport provides a framework for accent comparison across arbitrary
    
[^54]: RouteRepair：面向路径优化的基于大语言模型自动启发式设计中的实例级失败诊断与针对性修复

    RouteRepair: Instance-Level Failure Diagnosis and Targeted Repair in LLM-Based Automated Heuristic Design for Routing Optimization

    [https://arxiv.org/abs/2609.11452](https://arxiv.org/abs/2609.11452)

    RouteRepair通过实例级失败诊断和针对性修复机制，改进了基于大语言模型的路径优化自动启发式设计，在修复父代启发式方法特定弱点的同时保护已经表现良好的行为。

    

    高效的路径优化对货运运输、城市物流和共享出行至关重要，在这些领域中，通常需要在有限的计算预算下获得高质量的启发式算法。近年来，基于大语言模型（LLM）的自动启发式设计方法能够生成有效的路径规则，但总体评估可能会掩盖在特定实例结构上反复出现的失败。为解决这一局限，本研究开发了RouteRepair，该方法从实例级性能中诊断父代特有的弱点，并对相应的启发式组件进行针对性修改，同时保护已经表现良好的行为。通过结合路径证据、求解器行为和程序上下文来定义有界的修复目标，并且每次干预都通过匹配的父代-子代评估来验证失败恢复效果和附带性能退化。在旅行商问题（TSP）和带容量约束车辆路径问题上的实验（原文截断）

    arXiv:2609.11452v1 Announce Type: new  Abstract: Efficient routing optimization is essential to freight transportation, urban logistics, and shared mobility, where high-quality heuristics are often required under limited computational budgets. Recent large language model (LLM)-based automated heuristic design methods can generate effective routing rules, but aggregate evaluation may mask recurrent failures on particular instance structures. To address this limitation, this study develops RouteRepair, which diagnoses parent-specific weaknesses from instance-level performance and applies targeted modifications to the corresponding heuristic components while protecting behavior that already performs well. Routing evidence, solver behavior, and program context are combined to define bounded repair objectives, and each intervention is validated through matched parent-child evaluation of failure recovery and collateral degradation. Experiments on the traveling salesman problem (TSP) and capa
    
[^55]: 跨语言临床标注投影作为受限文本生成：一项六语言研究

    Cross-Lingual Clinical Annotation Projection as Constrained Text Generation: A Six-Language Study

    [https://arxiv.org/abs/2609.11450](https://arxiv.org/abs/2609.11450)

    该研究提出将跨语言临床标注投影建模为受限文本生成任务，通过将实体标签直接插入不可修改的目标语言文本并进行确定性验证，在六种语言上实现了最强且最一致的临床标注迁移性能。

    

    背景：旨在确定跨语言临床标注投影是否可以被表述为一种保持原文、文档级的生成任务，从而为多语言临床语料库构建产生可验证的字符级标注，并表征其相对于基于候选投影流水线的鲁棒性和计算权衡。方法：我们开发了一种受限的大语言模型投影工作流，将实体标签直接插入不可修改的目标语言文本中，随后进行确定性验证和字符偏移重建。我们将其与有监督的候选跨度投影方法以及机器学习与大语言模型混合精化方法一起评估，用于将西班牙语的疾病、症状和手术/操作标注迁移到六种语言中。评估采用MultiClinAI金标准，使用严格的跨度匹配和字符重叠F1指标。结果：直接大语言模型投影取得了最强且最一致的性能。GLM 5.2获得了平均（摘要在此处截断）

    arXiv:2609.11450v1 Announce Type: new  Abstract: Background: To determine whether cross-lingual clinical annotation projection can be formulated as a text-preserving, document-level generative task that produces verifiable character-level annotations for multilingual clinical corpus construction, and to characterize its robustness and computational trade-offs relative to candidate-based projection pipelines. Methods: We developed a constrained LLM projection workflow that inserts entity tags directly into immutable target-language text, followed by deterministic validation and character-offset reconstruction. We evaluated it alongside supervised candidate-span projection and hybrid ML-LLM refinement for transferring Spanish Disease, Symptom, and Procedure annotations into six languages. Evaluation used MultiClinAI gold standard with strict span matching and character-overlap F1 Results: Direct LLM projection achieved the strongest and most consistent performance. GLM 5.2 obtained a mea
    
[^56]: 流行率决定精确度：检测器定义数据集中的隐性污染

    Prevalence Determines Precision:Silent Contamination in Detector-Defined Datasets

    [https://arxiv.org/abs/2609.11449](https://arxiv.org/abs/2609.11449)

    论文通过贝叶斯分析揭示，由检测器定义的机器学习数据集的精确度主要由候选池中真阳性流行率决定而非检测器质量，虚假样本构成具有检测器形状的第二信号而非加性噪声，跨池迁移精确度评估可产生高达422%的误差。

    

    许多机器学习数据集是通过在候选池上运行检测器、启发式规则或模型来构建的；被接受的项目即成为标签。因此，根据贝叶斯定理，数据集的精确度由每个池中真阳性的流行率决定，而非仅取决于检测器本身的质量。基于同一仪器和同一时期，我们持有一个由检测器定义的事件数据集，以及一个将每个检出项目标记为真实或虚假的独立官方索引。同一检测器在三个池中产生的虚假率分别为81.7%、9.0%和0.0%。若将两个高流行率池的精确度迁移到低流行率池，预测值为0.955，而实测值仅为0.183，误差高达+422%；而贝叶斯表达式对全部三个池的预测误差均在3.3%以内。检测到的响应曲线是真实事件分量与虚假事件分量的精确凸组合（残差为1.1e-16），其中虚假事件以473比308的数量超过真实事件，因此污染是第二个带有检测器继承形状的信号，而非加性噪声。污染的方向取决于……（原文在此截断）

    arXiv:2609.11449v1 Announce Type: new  Abstract: Many ML datasets are constructed by running a detector, heuristic, or model over candidate pools; accepted items become labels. Dataset precision is then governed by true-positive prevalence in each pool via Bayes, not solely by detector quality. Using one instrument and period, we hold a detector-defined event dataset plus an independent official index labeling every detected item as real or phantom. One detector, three pools yield phantom rates 81.7%, 9.0%, and 0.0%. Transferring precision from the two high-rate pools to the low-rate pool predicts 0.955 versus measured 0.183, a +422% error; the Bayes expression predicts all three within 3.3%. The detected response curve is an exact convex combination of a true-event and a phantom component (residual 1.1e-16), with phantoms outnumbering true events 473 to 308, so contamination is a second signal with detector-inherited shape, not additive noise. Contamination direction depends on the es
    
[^57]: 研究声音事件分类中的灾难性遗忘

    Investigating catastrophic forgetting in sound event classification

    [https://arxiv.org/abs/2609.11447](https://arxiv.org/abs/2609.11447)

    这项研究发现声音事件分类中的灾难性遗忘主要发生在网络深层尤其是分类器头部，而完全冻结特征提取器并微调动态头部分类器是缓解灾难性遗忘最高效的方案。

    

    本工作研究了在声音事件分类任务的类别增量学习场景中防止灾难性遗忘的多种方法。我们使用FSD50K和AudioSet数据集，通过架构方法和正则化方法对该问题进行分析。我们设计了增量阶段和解决方案，选择性地保护网络卷积核免受权重更新以防止灾难性遗忘，并提出了一种动态头部解决方案，在每次学习新任务时进行自我扩展。研究结果表明，灾难性遗忘主要发生在网络较深的层中，尤其是分类器头部。对于所研究的域内声音分类问题，似乎能够缓解灾难性遗忘且最高效的解决方案是完全冻结特征提取器并对动态头部分类器进行微调，该方法显示出几乎不遗忘的特点、出色的训练稳定性，以及在记忆与性能之间的良好平衡。

    arXiv:2609.11447v1 Announce Type: cross  Abstract: This work investigates a number of approaches to prevent catastrophic forgetting in class incremental learning scenarios for sound event classification tasks. We analyze the problem using architectural and regularization approaches, using FSD50K and AudioSet datasets. We design incremental stages and solutions that selectively protect the kernels of the network from weight updates to prevent catastrophic forgetting, and a dynamic head solution that expands itself each time a new task is learned. The findings show that catastrophic forgetting mainly happens in deeper layers, in particular in the classifier head. For the studied in-domain sound classification problem, the solution that seems to alleviate catastrophic forgetting and is the most efficient is a full freezing of the feature extractor with a fine-tuning of the dynamic head classifier, showing little to no forgetting and great training stability, and a good balance between mem
    
[^58]: 面向高效异构模型协作的校准感知不确定性级联

    Calibration-Aware Uncertainty Cascades for Efficient Heterogeneous Model Collaboration

    [https://arxiv.org/abs/2609.11446](https://arxiv.org/abs/2609.11446)

    该论文提出CAUC框架，通过独立校准各模型的置信度建立统一的可靠性尺度，将部署策略与特定模型池解耦，实现高效且灵活的异构模型协作。

    

    异构模型协作旨在利用不同模型的互补优势，以平衡预测性能和推理成本。现有方法通常依赖于训练得到的路由器——这将路由决策与固定的任务和模型池绑定在一起——或者依赖于原始置信度级联——其阈值在异构模型之间缺乏一致的可靠性语义。因此，这些方法难以适应不断变化的模型池和部署预算。我们提出了校准感知不确定性级联，这是一个简单的后处理框架，它独立地校准每个模型的置信度，并利用验证数据来选择部署策略。由此得到的校准置信度分数建立了一个统一的可靠性尺度，用于决定是接受早期预测、调用更强的模型，还是有选择地组合模型输出。这一统一的决策准则将部署策略与任何特定的模型池解耦。

    arXiv:2609.11446v1 Announce Type: new  Abstract: Heterogeneous model collaboration seeks to exploit the complementary strengths of different models to balance predictive performance and inference cost. Existing approaches typically rely either on trained routers, which tie routing decisions to a fixed task and model pool, or on raw-confidence cascades, whose thresholds lack consistent reliability semantics across heterogeneous models. Consequently, these approaches adapt poorly to changing model pools and deployment budgets. We propose Calibration-Aware Uncertainty Cascades (CAUC), a simple post-hoc framework that independently calibrates each model's confidence and selects deployment policies using validation data. The resulting calibrated confidence scores establish a common reliability scale for accepting an early prediction, invoking a stronger model, or selectively combining model outputs. This unified decision criterion decouples deployment policies from any particular model pool
    
[^59]: 大语言模型作为符号回归中生理合理性的事后审计工具：一项由临床医生评估的案例研究

    LLMs as Post-hoc Auditors of Physiological Plausibility in Symbolic Regression: A Clinician-Evaluated Case Study

    [https://arxiv.org/abs/2609.11431](https://arxiv.org/abs/2609.11431)

    本研究创新性地将大语言模型用作事后审计工具，对符号回归进化生成的数学表达式进行可解释性和医学合理性分析排序，并通过临床医生评估验证了该方法的有效性。

    

    遗传编程及其变体（如语法进化）被广泛应用于符号回归中，用于从多变量数据中推导数学表达式。除了预测准确性外，这类模型还因其可解释性潜力而受到重视，能够提供将输入变量与输出结果相关联的显式方程。然而，实现可解释性和合理性仍然具有挑战性，因为进化得到的模型可能过于复杂或在科学上不一致。在本研究中，我们探索大语言模型（LLMs）是否能够协助提高由进化计算方法生成的符号回归模型的可解释性。在我们之前使用基于语法的遗传编程估算体脂百分比研究的基础上，我们研究了将大语言模型作为后处理工具，根据可解释性和医学合理性对进化得到的表达式进行分析和排序。四个符号表达式

    arXiv:2609.11431v1 Announce Type: new  Abstract: Genetic Programming and its variants, such as grammatical evolution, are widely used in Symbolic Regression to derive mathematical expressions from multivariate data. In addition to predictive accuracy, models are appreciated for their potential to provide interpretability, offering explicit equations that relate input variables to outcomes. However, achieving interpretability and plausibility remains challenging, as evolved models may be complex or scientifically inconsistent. In this study, we explore whether Large Language Models, can assist in improving the explainability of Symbolic Regression models generated by evolutionary computation methods. Building upon our previous work on estimating body fat percentage using grammar-based Genetic Programming , we investigate the use of LLMs as post-processing tools to analyze and rank evolved expressions according to their interpretability and medical plausibility. Four symbolic expressions
    
[^60]: SWRouter：面向多轮大语言模型对话的相似性收缩窗口路由

    SWRouter: Similarity-Contractive Window Routing for Multi-Turn Large Language Model Conversations

    [https://arxiv.org/abs/2609.11414](https://arxiv.org/abs/2609.11414)

    提出SWRouter，通过基于相似性的上下文分割机制与双指标评估框架，解决多轮对话中大语言模型路由面临的上下文信息丢失混淆及评估指标耦合两大难题。

    

    大语言模型各具互补优势，这促使研究者开发路由方法，将每个查询分派给最合适的模型。尽管现有路由器在单轮设置中效果良好，但它们无法直接迁移到多轮对话场景——在多轮对话中，路由性能严重依赖于历史上下文如何被分割、保留并融入当前提示词。这带来了两个根本性挑战：一是在上下文构建过程中防止信息丢失与信息混淆，二是在评估路由质量时避免将模型选择与提示词构建质量混为一谈。在本文中，我们提出了SWRouter，一种面向多轮大语言模型路由的相似性收缩窗口路由器。SWRouter将基于相似性的上下文分割机制用于提示词构建，并结合双指标评估框架，将构建准确性与路由器性能解耦。在多个（数据集上的实验……摘要此处截断）

    arXiv:2609.11414v1 Announce Type: new  Abstract: Large language models exhibit complementary strengths, motivating routing methods that dispatch each query to the most suitable model. Although existing routers are effective in single-turn settings, they do not directly transfer to multi-turn dialogue, where routing performance critically depends on how historical context is segmented, retained, and incorporated into the current prompt. This introduces two fundamental challenges: preventing information loss and information confusion during context construction, and evaluating routing quality without conflating model selection with prompt construction quality. In this paper, we propose SWRouter, a Similarity-Contractive Window Router for multi-turn large language model routing. SWRouter combines a similarity-based context segmentation mechanism for prompt construction with a dual-metric evaluation framework that decouples construction accuracy from router performance. Experiments on mult
    
[^61]: X-AuT：基于跨尺度蒸馏的语音大语言模型渐进式音频编码器压缩

    X-AuT: Progressive Audio-Encoder Compression for Speech LLMs with Cross-Scale Distillation

    [https://arxiv.org/abs/2609.11412](https://arxiv.org/abs/2609.11412)

    X-AuT提出了一种渐进式音频编码器压缩框架，通过行为探针选择层组合，并结合跨尺度蒸馏与LoRA微调恢复剪枝模型，成功将Qwen3-ASR-0.6B的音频编码器从18层压缩至14层，在中英文基准测试中仍保持较低的错误率。

    

    降低音频编码器的深度可以降低语音大语言模型的推理成本，但直接移除完整的编码器模块会扰动解码器所依赖的嵌入表示，并可能导致词删除和序列过早结束等错误。我们提出了X-AuT，这是一个渐进式框架，通过简短的行为探针来选择层组合，并通过表示对齐、跨尺度蒸馏、计划性的学生策略监督以及LoRA微调来恢复被剪枝的模型。语言模型骨干保持冻结，而注意力LoRA适配器和绑定的输出嵌入在蒸馏过程中进行自适应调整。训练使用来自转写一致性流水线中最高一致性的数据层级，随后在微调阶段进行数据源重加权。在十个公开的中英文基准测试中，将Qwen3-ASR-0.6B的音频编码器从18层压缩到16层，使宏平均错误率从5.61%降至5.27%。14层模型达到5.75%的错误率，同时减少了20.7%的（摘要在此处被截断）。

    arXiv:2609.11412v1 Announce Type: cross  Abstract: Reducing audio-encoder depth lowers the inference cost of speech large language models, but removing complete blocks perturbs the embeddings consumed by the decoder and can cause deletion and premature end-of-sequence errors. We introduce X-AuT, a progressive framework that selects layer combinations through short behavioral probes and restores the pruned model through representation alignment, cross-scale distillation, scheduled student-policy supervision, and LoRA finetuning. The language-model backbone remains frozen, while attention LoRA adapters and the tied output embedding adapt during distillation. Training uses the highest-agreement tier from a transcript-consistency pipeline, followed by source reweighting during finetuning. On ten public Chinese--English benchmarks, compressing Qwen3-ASR-0.6B from 18 to 16 audio-encoder layers reduces macro-average error from 5.61% to 5.27%. The 14-layer model reaches 5.75% with 20.7% fewer 
    
[^62]: 深度伪造验证码：缓解下一代社会工程攻击

    Deep-Fake CAPTCHA: Mitigating Next-Generation Social Engineering Attacks

    [https://arxiv.org/abs/2609.11404](https://arxiv.org/abs/2609.11404)

    本文提出DF-CAPTCHA框架，通过让呼叫者执行对人类简单但对实时深度伪造系统难以伪造的挑战-响应任务，并从真实性、身份一致性、任务完成度和响应时间四个维度验证响应，从而主动防御语音和视频通话中的深度伪造攻击。

    

    本文提出了DF-CAPTCHA，一种针对语音和视频通话中实时深度伪造冒充的主动防御方法。与被动搜索伪造痕迹不同，DF-CAPTCHA提示呼叫者执行简单的挑战-响应任务，这些任务对人类来说很容易完成，但当前的实时深度伪造系统难以令人信服地生成。该框架使用四个标准来验证响应：真实性、身份一致性、任务完成度和响应时间。我们通过用户研究和使用实时深度伪造模型的实验，在音频和视频两种模态上对该方法进行了评估。结果表明，人们往往难以区分实时深度伪造内容与真实媒体，而DF-CAPTCHA相比被动方法显著提升了检测性能，在两种模态下均达到了很高的准确率。这些发现表明，基于主动挑战的验证是对抗下一代社会工程攻击的一种实用且稳健的防御手段。

    arXiv:2609.11404v1 Announce Type: cross  Abstract: This paper presents DF-CAPTCHA, an active defense against real-time deepfake impersonation in voice and video calls. Instead of passively searching for artifacts, DF-CAPTCHA prompts the caller to perform simple challenge-response tasks that are easy for humans but difficult for current real-time deepfake systems to generate convincingly. The framework verifies the response using four criteria: realism, identity consistency, task completion, and response time. We evaluate the approach across both audio and video modalities using user studies and experiments with real-time deepfake models. Results show that people often struggle to distinguish real-time deepfakes from authentic media, while DF-CAPTCHA substantially improves detection performance over passive methods, reaching high accuracy in both modalities. These findings suggest that active challenge-based verification is a practical and robust defense against next-generation social e
    
[^63]: 从查询到叙事：用于知识图谱探索与质量评估的文化遗产数据故事

    From Queries to Narratives: Cultural Heritage Data Stories for Knowledge Graph Exploration and Quality Assessment

    [https://arxiv.org/abs/2609.11403](https://arxiv.org/abs/2609.11403)

    本文提出“数据故事”方法，降低文化遗产知识图谱的查询门槛，并将用户探索转化为数据质量评估，实现便捷查询与发现隐藏数据问题的双重目标。

    

    诸如NFDI4Culture-KG之类的文化遗产知识图谱包含数百万条关于艺术品、音乐、铭文、历史事件以及与之相关的人物和地点的三元组数据。然而，对许多用户来说，发现这些知识可能很困难。虽然SPARQL可以学习，但编写有意义的查询首先需要对图谱的数据模型有深入的理解，而许多领域研究人员和从业者并不愿意进行这样的投入。即使使用现有的用户界面，通常也需要一个起点和一些指导，因为图谱中包含的数据高度专业化、异构且不断增长，这使得了解其中包含的内容或它能回答哪些问题变得具有挑战性。在本文中，我们提出数据故事作为一种方法，不仅可以降低这一门槛，还能将探索过程转化为数据质量评估，从而将易于使用的查询与发现隐藏问题结合起来。

    arXiv:2609.11403v1 Announce Type: new  Abstract: Cultural-heritage KGs such as the NFDI4Culture-KG contain millions of triples about artworks, music, inscriptions, historical events, and the people and places connected to them. For many users, however, discovering this knowledge can be difficult. While SPARQL can be learned, writing meaningful queries first requires an in-depth understanding of the graph's data model, an investment many domain researchers and practitioners are unwilling to make. Even with existing user interfaces, a starting point and some guidance are usually needed, because the data contained in the graph is highly specialized, heterogeneous, and constantly growing, making it challenging to know what it contains or which questions it can answer. In this paper, we present data stories as a way not only to lower this barrier, but also to turn exploration into data-quality assessment, and thus combine accessible querying with the discovery of issues that remain hidden i
    
[^64]: 超越置信度：面向大语言模型推理的稳定性感知测试时自适应方法

    Beyond Confidence: Stability-Aware Test-Time Adaptation for LLM Reasoning

    [https://arxiv.org/abs/2609.11393](https://arxiv.org/abs/2609.11393)

    提出TASCO框架，发现高置信度在局部扰动下保持稳定时更可能正确，从而在冻结LLM的前提下通过优化轻量级任务级前缀，将局部稳定性引入基于置信度的测试时自适应，有效提升大语言模型推理能力。

    

    测试时自适应已成为一种轻量级的替代方案，可替代代价高昂的后训练，用于提升大语言模型（LLM）在下游任务上的推理能力。预测熵为这种自适应提供了源自模型本身的信号，能够在无需外部验证器或奖励模型的情况下，引导模型走向更高置信度的推理状态。然而，更高的置信度并不一定意味着正确，因为大语言模型在错误的推理轨迹上可能仍然保持高置信度。我们观察到，当置信度在局部扰动下保持稳定时，高置信度的推理更有可能是正确的。基于这一观察，我们提出了稳定性感知置信度优化的测试时自适应方法TASCO，该框架在保持大语言模型参数冻结的前提下，将局部稳定性融入基于置信度的测试时自适应中。TASCO通过在（两类）扰动下优化轻量级的任务级前缀来实现局部稳定性……

    arXiv:2609.11393v1 Announce Type: new  Abstract: Test-time adaptation has emerged as a lightweight alternative to costly post-training for improving the reasoning capabilities of Large Language Models (LLMs) on downstream tasks. Predictive entropy provides a model-derived signal for such adaptation, guiding models toward higher-confidence reasoning states without external verifiers or reward models. However, higher confidence does not necessarily imply correctness, as LLMs may remain highly confident along incorrect reasoning trajectories. We observe that high-confidence reasoning is more likely to be correct when confidence remains stable under local perturbations. Based on this observation, we propose Test-Time Adaptation via Stability-Aware Confidence Optimization (TASCO), a framework that incorporates local stability into confidence-based test-time adaptation while keeping the LLM frozen. TASCO operationalizes local stability by optimizing a lightweight task-level prefix under two 
    
[^65]: 买方人工智能驱动的环境治理与供应商环境争议：基于组织信息处理与信号传递的视角

    Buyer Artificial Intelligence-Enabled Environmental Governance and Supplier Environmental Controversies: An Organizational Information Processing and Signaling

    [https://arxiv.org/abs/2609.11391](https://arxiv.org/abs/2609.11391)

    本研究基于组织信息处理理论和信号传递理论，通过对41个国家2505家供应商面板数据的分析，发现买方采用人工智能驱动的环境治理能够显著降低海外供应商次年发生环境争议的风险。

    

    全球供应链中的环境争议给全球买方带来了重大风险。本研究考察海外供应商接触买方的人工智能（AI）驱动的环境治理是否会减少供应商的环境争议。基于组织信息处理理论和信号传递理论，我们研究了供应商接触AI驱动的治理如何影响其环境争议，以及这种效应在不同制度情境下的差异。我们使用文本分析来衡量买方AI驱动的环境治理，并采用多维固定效应模型分析了2020年至2024年间41个国家中2505家美国上市公司供应商的面板数据。我们发现，供应商接触买方AI驱动的环境治理与其次年发生环境争议呈负相关关系。这种负相关关系更强

    arXiv:2609.11391v1 Announce Type: cross  Abstract: Environmental controversies in global supply chains pose significant risks for global buyers. This study examines whether overseas suppliers' exposure to buyers' artificial intelligence (AI)-enabled environmental governance reduces supplier environmental controversies. Drawing on organizational information processing theory and signaling theory, we investigate how suppliers' exposure to AI-enabled governance influences their environmental controversies and the institutional contingencies under which this effect varies. Using text analysis to measure buyer AI-enabled environmental governance, we analyze panel data on 2,505 suppliers of U.S.-listed firms across 41 countries from 2020 to 2024 with multidimensional fixed-effects models. We find that suppliers' exposure to buyer AI-enabled environmental governance is negatively associated with supplier environmental controversies in the following year. This negative relationship is stronger
    
[^66]: VikingRAG：面向结构化文档的准确且令牌高效的检索增强生成

    VikingRAG: Accurate and Token-efficient Retrieval-augmented Generation over Structured Documents

    [https://arxiv.org/abs/2609.11390](https://arxiv.org/abs/2609.11390)

    VikingRAG通过将代理式多轮检索轨迹物化为可复用的经验边，并引入自适应升级策略在证据充分时采用单轮经验增强检索，在保持高准确率的同时大幅降低了结构化文档检索增强生成的令牌成本。

    

    最先进的检索增强生成（RAG）方法利用文档结构来获取充分的证据，但往往会产生高昂的令牌成本。为了在不损害高RAG准确性的前提下减少结构上下文令牌，我们提出了VikingRAG，一个目录感知的语义数据管理系统，它紧密集成语义与结构访问，以支持结构上下文高效、证据缺口驱动的多轮检索。为了进一步降低多轮交互的令牌开销，我们将代理式多轮检索轨迹物化为经验边，并对相似查询复用这些边，避免重复的多轮探索。此外，为了在无需代理式多轮检索时减少令牌成本，我们引入了一种自适应升级策略，当证据充分时从单轮经验增强检索中直接回答，仅在必要时才调用代理式多轮检索。

    arXiv:2609.11390v1 Announce Type: cross  Abstract: State-of-the-art retrieval-augmented generation (RAG) methods exploit document structures to acquire sufficient evidence, but often incur substantial token costs. To reduce structural-context tokens without compromising high RAG accuracy, we present {\sf VikingRAG}, a directory-aware semantic data management system that tightly integrates semantic and structural access to support structural-context-efficient, evidence-gap-driven multi-round retrieval. To further reduce token overhead of multi-round interaction, we materialize agentic multi-round retrieval traces as experience edges, and reuse these edges for similar queries, avoiding repeated multi-round exploration. To additionally reduce token costs when agentic multi-round retrieval is unnecessary, we introduce an adaptive escalation strategy that answers from one-round experience-augmented retrieval when the evidence is sufficient, and invokes agentic multi-round retrieval only oth
    
[^67]: 智能体集成软件：交互契约与持续保障

    Agent-Integrated Software: Interaction Contracts and Continuous Assurance

    [https://arxiv.org/abs/2609.11381](https://arxiv.org/abs/2609.11381)

    该论文提出智能体集成软件（AIS）软件模式与意图级交互抽象（IIA）任务语义，通过交互契约约束任务级交互与应用行为之间的对应关系，并以持续保障机制在依赖变化时维护声明，从而解决将智能体嵌入现有应用时的协调问题。

    

    将智能体嵌入现有应用程序会产生一个持续的协调问题：在委托执行继续进行的同时，用户可以修改目标并操作共享对象。我们认为，可靠的集成需要在任务级交互与应用程序行为之间建立明确的对应关系。我们引入智能体集成软件（AIS）作为一种软件模式，它结合了传统核心、直接交互和内置智能体；并引入意图级交互抽象（IIA）作为任务语义，用户通过它来检查和控制委托的工作。一个开放迁移系统模型将AIS的执行与IIA的状态和事件相关联。交互契约通过任务绑定、角色特定的权限、控制转换和结果证据来约束这种关系；持续保障则在依赖关系发生变化时维护有范围限定的声明。一个紧凑的披露契约和条件命题说明了为什么……

    arXiv:2609.11381v1 Announce Type: new  Abstract: Embedding an intelligent agent in an existing application creates a persistent coordination problem: users can revise goals and manipulate shared objects while delegated execution continues. We argue that dependable integration requires an explicit correspondence between task-level interaction and application behavior. We introduce Agent-Integrated Software (AIS) as a software pattern combining a conventional core, direct interaction, and a built-in agent, and Intent-Level Interaction Abstraction (IIA) as the task semantics through which users inspect and control delegated work. An open transition-system model relates AIS execution to IIA states and events. Interaction contracts constrain this relation through task bindings, role-specific authority, control transitions, and outcome evidence; continuous assurance maintains scoped claims as their dependencies change. A compact disclosure contract and conditional propositions illustrate why
    
[^68]: 刻画Bluesky内容审核服务：从服务自动化到危害全景

    Characterizing Bluesky Content Moderation Service: From Automation of Service to Landscape of Harms

    [https://arxiv.org/abs/2609.11373](https://arxiv.org/abs/2609.11373)

    本研究首次对去中心化社交平台Bluesky的默认审核系统进行了大规模独立审计，通过分析1060万条审核标签，揭示了其人机协作的自动化机制、危害检测效能及所识别的危害类型全景。

    

    关于内容审核的实证研究从根本上受制于主要社交媒体平台上审核系统部署的不透明性。为此，近期出现的具有透明、公开审核日志的去中心化平台为独立审计提供了前所未有的机会。在本工作中，我们利用这种架构透明性，对Bluesky平台上的默认审核系统——Bluesky审核服务（BMS）——进行了首次大规模审计。通过分析其2025年的1060万条审核标签，我们研究了三个基础性方面：(i) 其机制（自动化程度与人工监督的对比），(ii) 其效能（检测危害的准确性），以及 (iii) 其目的（它所识别的危害全景）。我们的发现揭示了一个人机协作系统，其中涉及性和血腥内容的标签在几秒钟内自动应用，而需要细致判断且高风险的标签则需要更多的人工参与……

    arXiv:2609.11373v1 Announce Type: cross  Abstract: Empirical research on content moderation is fundamentally constrained by the opaque deployment of moderation systems on major social media platforms. To this end, the recent emergence of decentralized platforms with transparent, public moderation logs presents an unprecedented opportunity for independent audits. In this work, we leverage this architectural transparency to conduct the first large-scale audit of the default moderation system on Bluesky, the Bluesky Moderation Service (BMS). Analyzing its 10.6M moderation labels from 2025, we investigate three foundational aspects: (i) its mechanism (the degree of automation versus human oversight), (ii) its efficacy (accuracy in detecting harms), and (iii) its purpose (the landscape of harms it identifies). Our findings reveal a human-AI collaborative system where labels for sexual and graphic content are applied automatically in seconds, while nuanced and high stakes labels require more
    
[^69]: RAMamba-Net：一种面向听觉注意力检测的可靠性感知且基于Mamba的多模态融合网络

    RAMamba-Net: A Reliability-Aware and Mamba-Based Multimodal Fusion Network for Auditory Attention Detection

    [https://arxiv.org/abs/2609.11372](https://arxiv.org/abs/2609.11372)

    该论文提出了RAMamba-Net，一种可靠性感知的基于Mamba的多模态融合网络，通过融合EEG与EOG信号、Mamba增强的频带感知Transformer以及跨模态注意力机制，显著提升了听觉注意力检测的性能与鲁棒性。

    

    听觉注意力解码（AAD）旨在从生理信号中识别被关注的说话人，为神经引导助听设备和自然人机交互提供支持。脑电图（EEG）是AAD的主要模态，但在自然视听场景中提供的证据并不完整，这促使了EEG与眼电图（EOG）的融合。现有方法仍然受限于跨模态交互能力弱、时间建模效率低以及对样本变化鲁棒性差等问题。为解决这些局限性，我们提出了RAMamba-Net，一种用于AAD的可靠性感知且基于Mamba的多模态融合网络。RAMamba-Net采用Mamba增强的频带感知卷积Transformer来捕获特定频带的EEG模式和长程时间动态特征；采用双分支时空编码器建模EOG的时间特性和通道间依赖关系；通过跨模态注意力机制实现显式的模态间交互。随后，可靠性感知模块……（摘要在此处不完整）

    arXiv:2609.11372v1 Announce Type: new  Abstract: Auditory attention decoding (AAD) identifies the attended speaker from physiological signals, supporting neuro-steered hearing devices and natural human-machine interaction. Electroencephalography (EEG) is the dominant modality for AAD but provides incomplete evidence in naturalistic audio-visual scenes, motivating EEG and electrooculography (EOG) fusion. Existing approaches remain limited by weak cross-modal interaction, inefficient temporal modeling, and low robustness to sample variations. To address the limitations, we propose RAMamba-Net, a reliability-aware Mamba-based multimodal fusion network for AAD. RAMamba-Net employs a Mamba-enhanced band-aware convolutional Transformer to capture band-specific EEG patterns and long-range temporal dynamics. A dual-branch temporal-spatial encoder models EOG temporal and inter-channel dependencies. Cross-modal attention enables explicit modality interaction. Then, a reliability-aware module is 
    
[^70]: 可移植语义与私有方言：语言模型细胞间潜在通信中的复用与负迁移

    Portable Semantics, Private Dialects: Reuse and Negative Transfer in Latent Communication Between Language-Model Cells

    [https://arxiv.org/abs/2609.11365](https://arxiv.org/abs/2609.11365)

    该研究通过对六个独立训练的语言模型社会进行泄漏受控的因果互操作性审计，发现语义相似的潜在接口并不构成统一语言——仅相同初始化的社会对可完全互操作，且继承的接口状态可能对后续学习产生负迁移。

    

    在共享“基因组”的语言模型社会中，受限的证据可见性有利于可复用的、按价值索引的潜在数据包接口，而母研究中唯一一个全球可见的高性能模型学到的却是与具体情节纠缠的编码。本配套研究提出三个问题：独立训练的模型社会是否共享同一种数据包语言、严格的零样本迁移在何处失效、以及继承的接口状态对后续学习是有助还是有弊。首先，对六个独立训练的受限社会之间全部30个有序配对进行泄漏受控的因果互操作性审计——在密封的保留结构和预注册的原始/正交/线性/非线性对齐阶梯下——结果表明，这六个语义相似的接口并未构成一种统一的原始语言：其中一对相同初始化的社会在两个方向上完全可互操作，另一对表现出不对称的部分兼容性，而全部26个跨初始化方向在所有冻结对齐……（原文摘要至此截断）

    arXiv:2609.11365v1 Announce Type: new  Abstract: In shared-genome language-model societies, restricted evidence visibility favors reusable, value-indexed latent packet interfaces, whereas the sole high-performing globally visible model in the parent study learned an episode-entangled code. This companion study asks whether independently trained societies share one packet language, where strict zero-shot transfer fails, and whether inherited interface state helps or harms later learning. First, a leakage-controlled causal interoperability audit over all 30 ordered pairs of six independently trained restricted societies -- under sealed held-out structure and a preregistered raw/orthogonal/linear/nonlinear alignment ladder -- shows the six semantically similar interfaces do not form one raw language: one same-initialization pair is exactly interoperable in both directions, a second shows asymmetric partial compatibility, and all 26 cross-initialization directions fail every frozen alignme
    
[^71]: 具体化作为可迁移词汇表：用普通GNN实现零样本链接预测

    Reification as a Transferable Vocabulary: Zero-Shot Link Prediction with Vanilla GNNs

    [https://arxiv.org/abs/2609.11347](https://arxiv.org/abs/2609.11347)

    该研究提出将知识图谱“具体化”为事实节点并通过六个元关系的固定词汇表连接，使普通GNN（如现成的GAT）无需任何专用架构即可实现零样本归纳链接预测，性能媲美预训练基础模型ULTRA。

    

    诸如ULTRA之类的知识图谱基础模型，通过将迁移机制硬编码进专用架构，在未见过的图上实现零样本链接预测。在这项工作中，我们将该机制从架构中移出并融入表示之中，具体做法是对输入图进行“具体化”（reification）处理：每条事实都变成一个节点，通过由六个元关系组成的固定词汇表与其主语、宾语和关系类型相连，其中关系类型作为匿名的共享节点而非模型参数。在这种表示上，五个教科书式的GNN（GAT、采用求和聚合的GINE、采用mean+max聚合的GINE、GraphSAGE、R-GCN），每个仅在包含4,245个三元组的单个知识图谱上、使用一块NVIDIA A100训练30分钟，即可零样本迁移到40个归纳式链接预测基准上。其中表现最好的——一个现成的GAT——在ULTRA自己的评估套件上与ULTRA（一个在三个图上预训练的专用基础模型）表现相当。同样的固定词汇表还扩展了……

    arXiv:2609.11347v1 Announce Type: new  Abstract: Knowledge graph foundation models such as ULTRA achieve zero-shot link prediction on unseen graphs through dedicated architectures that hard-code a transfer mechanism. In this work we move that mechanism out of the architecture and into the representation, by \emph{reifying} the input graph: every fact becomes a node, connected to its subject, object, and relation type through a fixed vocabulary of six meta-relations, with relation types as anonymous shared nodes rather than model parameters. On this representation, five textbook GNNs (GAT, GINE with sum and with mean+max aggregation, GraphSAGE, R-GCN), each trained on a single knowledge graph of 4,245 triples for 30 minutes on one NVIDIA A100, transfer zero-shot to 40 inductive link-prediction benchmarks. The best of them, an off-the-shelf GAT, matches ULTRA, a dedicated foundation model pretrained on three graphs, across ULTRA's own evaluation suite. The same fixed vocabulary extends t
    
[^72]: 探索扩散Transformer在多模态脑状态解码中的跨模态数据增强应用

    Exploring Diffusion Transformers for Cross-Modal Augmentation in Multimodal Brain State Decoding

    [https://arxiv.org/abs/2609.11341](https://arxiv.org/abs/2609.11341)

    本文提出CoMA-DiT，一种双向跨模态扩散Transformer，将成对模态视为相互生成监督的来源以实现潜在空间数据增强，在多模态脑状态解码任务上显著优于20个代表性基线方法。

    

    多模态脑状态解码研究主要集中于融合成对模态进行预测，但很少探索如何进一步利用模态间的对应关系来丰富训练数据并改进多模态表征学习。为填补这一空白，我们提出了CoMA-DiT，一种用于潜在增强的双向跨模态扩散Transformer，它将成对模态视为相互生成监督的来源，而不仅仅是待融合的输入。CoMA-DiT通过跨模态注意力机制使速度预测以成对模态为条件，并通过可靠性门控残差机制自适应地注入由此产生的变化。在多模态听觉注意力解码和情绪识别任务上的实验表明，CoMA-DiT始终优于20个代表性基线方法，相比无增强基线，在准确率和宏F1上分别取得了4.28%和6.70%的绝对提升。

    arXiv:2609.11341v1 Announce Type: new  Abstract: Multimodal brain state decoding has largely focused on fusing paired modalities for prediction, but has rarely explored how their correspondence can be further exploited to enrich training data and improve multimodal representation learning. To address this gap, we propose CoMA-DiT, a bidirectional cross-modal Diffusion Transformer for latent augmentation that treats paired modalities as sources of mutual generative supervision rather than merely as inputs to be fused. CoMA-DiT conditions velocity prediction on the paired modality through cross-modal attention and adaptively injects the resulting variation via a reliability-gated residual mechanism. Experiments on multimodal auditory attention decoding and emotion recognition showed that CoMA-DiT consistently outperformed 20 representative baselines, achieving absolute gains of 4.28% and 6.70% in accuracy and macro-F1 over the no-augmentation baseline, respectively. Extensive ablation, s
    
[^73]: 论匿名化对大语言模型性能的影响

    On the Impact of Anonymization on the Performance of Large Language Models

    [https://arxiv.org/abs/2609.11335](https://arxiv.org/abs/2609.11335)

    本文系统评估了五个大语言模型在十一个基准上处理原始与匿名化输入的性能差异，发现匿名化总体上会降低性能，且影响因模型能力与任务类型而异——能力最强的模型性能下降最大，TruthfulQA性能反而提升，而检索类任务则遭遇灾难性下降。

    

    随着大语言模型越来越多地被部署在敏感领域，对输入数据进行匿名化以保护个人可识别信息已成为一项关键做法。然而，这种匿名化对模型效用的影响尚未得到充分理解。本文对隐私与性能之间的权衡进行了系统性的实证研究。我们在十一个多样化的基准上评估了五个知名语言模型，比较它们在原始输入与假名化输入上的性能表现。我们的结果揭示，虽然匿名化通常会降低性能，但其影响非常微妙。我们发现能力更强的模型，如Qwen2.5-72B和GPT-4o mini，遭受的性能下降最大，这表明它们对特定实体信息的依赖更强。这种影响还与任务相关：TruthfulQA上的性能在匿名化后反而有所提升，而以检索为重点的任务（如RGB）则经历了灾难性的下降。

    arXiv:2609.11335v1 Announce Type: new  Abstract: As large language models are increasingly deployed in sensitive domains, anonymizing input data to protect personally identifiable information has become a critical practice. However, the impact of this anonymization on model utility is not well understood. This paper presents a systematic empirical study of the trade-off between privacy and performance. We evaluate five prominent language models across eleven diverse benchmarks, comparing their performance on original versus pseudonymized inputs. Our results reveal that while anonymization generally degrades performance, the effect is highly nuanced. We find that more capable models, such as Qwen2.5-72B and GPT-4o mini, suffer the largest performance drops, suggesting a stronger reliance on specific entity information. The impact is also task-dependent: performance on TruthfulQA improves with anonymization, while retrieval-focused tasks like RGB experience a catastrophic decline. Furthe
    
[^74]: E-CONAN（蕴含、矛盾与中立）基准测试：阿拉伯语文本蕴含与自然推理数据集

    E-CONAN (Entailment, CONtradition And Neutral) Benchmarks: Arabic Textual Entailment and Natural Inference Datasets

    [https://arxiv.org/abs/2609.11334](https://arxiv.org/abs/2609.11334)

    本文提出了面向阿拉伯语文本蕴含与自然推理的E-CONAN基准数据集，其句子对来源于自动翻译、人工验证翻译、手工编写及含谣言的新闻标题等多种渠道，并利用该基准对9个最先进的多语言预训练模型进行了零样本分类评估。

    

    自然语言推理（NLI）通过处理句子对来提取其语义关系。NLI一直是一个热门研究话题，并作为核心组件集成到其他自然语言处理应用中。尽管全球各种语言的文本推理研究取得了显著进展，但阿拉伯语在该领域仍然面临资源匮乏的问题。为了填补这一空白，本文提出了E-CONAN基准测试，它由来自多种来源的句子对组成：(1) 自动翻译的句子对，(2) 经人工验证的机器翻译句子对，(3) 来自对外阿拉伯语教学书籍的手工编写的句子对，以及 (4) 来自不同新闻频道且包含谣言的新闻标题句子对。E-CONAN包含两个基准数据集：E-CONAN-2，一个二分类数据集（RTE）；以及E-CONAN-3，一个三分类数据集（NLI）。此外，我们还使用E-CONAN基准测试评估了9个最先进的多语言预训练模型的零样本分类性能。

    arXiv:2609.11334v1 Announce Type: new  Abstract: Natural Language Inference processes pairs of sentences to extract their semantic relations. NLI has been a hot research topic, integrated as a main component in other NLP applications. Despite significant advancements in textual inference across various languages all around the world, Arabic language still suffers from limited resources in this domain. To address this gap, this paper introduces E-CONAN benchmarks that are composed of sentences pairs from various sources: (1) automatically-translated pairs, (2) human-validated machine-translated pairs, (3) hand-crafted pairs from teaching Arabic as foreign language books, and (4) headlines pairs from different news channels containing rumors. E-CONAN contains two benchmark datasets, E-CONAN-2, a 2-way dataset (RTE) and E-CONAN-3, a 3-way dataset (NLI). Additionally, we have used E-CONAN benchmarks to evaluate 9 state-of-the-art multilingual pretrained models using zero-shot classificatio
    
[^75]: 语义提升算子与不可判定类在保持性下的闭包性

    The Semantic Elevation Operator and the Closure of the Undecidable Class under Preservation

    [https://arxiv.org/abs/2609.11326](https://arxiv.org/abs/2609.11326)

    该论文提出语义提升算子 ΛΦ，将程序静态语义性质问题转化为自修改后的保持性问题，并基于克林递归定理证明不可验证性质类在该算子下封闭，且无界迭代将攀升至算术层级的 Π₂-完备性。

    

    程序静态语义性质的不可判定性由莱斯定理所支配。然而，自修改系统需要分析的并非某个性质当前是否成立，而是当系统重写自身时该性质是否被保持。我们通过语义提升算子 ΛΦ 将这一转变形式化，它把静态问题“x 是否满足 P？”转化为动态问题“x 经过 Φ 变换后 P 是否被保持？”。我们证明：当 Φ 是内涵性的（即依赖于源代码本身，而不仅仅是所计算的函数）时，即使提升后的性质破坏了莱斯定理所要求的外延性，它仍然是不可判定的；该证明基于克林递归定理，而非莱斯定理。因此，不可验证性质类 U 在提升算子下是封闭的。对该算子的无界迭代将沿算术层级攀升至 Π₂-完备性，进一步巩固了不可验证性。

    arXiv:2609.11326v1 Announce Type: cross  Abstract: The undecidability of a program's static semantic properties is governed by Rice's theorem. Self-modifying systems, however, require analysing not whether a property holds now, but whether it is preserved when the system rewrites itself. We formalise this transition through a semantic elevation operator {\Lambda}{\Phi}, which turns the static question "does x satisfy P?" into the dynamic question "is P preserved after x is transformed by {\Phi}?". We prove that when {\Phi} is intensional (depending on the source code, not only on the computed function), the elevated property remains undecidable even though it breaks the extensionality that Rice's theorem requires; the proof rests on Kleene's recursion theorem, not on Rice. Consequently the class U of non-verifiable properties is closed under the elevation operator. Unbounded iteration of the operator climbs the arithmetical hierarchy -to {\Pi}02-completeness- consolidating non-verifiab
    
[^76]: AI暴露度与AI韧性：面向软件及基于软件的商业模式的二维评估框架

    AI Exposure and AI Resilience: A Two-Dimensional Assessment Framework for Software and Software-Based Business Model

    [https://arxiv.org/abs/2609.11321](https://arxiv.org/abs/2609.11321)

    本文提出AI暴露度与AI韧性（AI-ER）二维评估框架，用于衡量人工智能对软件及软件商业模式带来的变革压力以及企业吸收压力并经济可行地利用AI的能力，弥补了传统技术尽职调查无法评估AI商业影响的不足。

    

    人工智能正在改变软件生产以及基于软件的商业模式的经济学。传统的技术尽职调查主要考察架构、可扩展性和技术债务等技术属性，但这些标准无法充分捕捉人工智能如何影响公司的价值主张、竞争地位、利润率或客户获取渠道。本文提出了人工智能暴露度与韧性（AI-ER）作为一个二维评估框架。AI暴露度描述了人工智能为商业模式带来的变革压力，而AI韧性描述了公司吸收这种压力、适应变化条件并以经济可行的方式利用人工智能的能力。这两个维度的度量指标源自当前的人工智能能力、其部署条件，以及关于商业模式和组织适应性的相关研究。该模型将暴露度与韧性分开进行评估……

    arXiv:2609.11321v1 Announce Type: new  Abstract: Artificial intelligence is changing both software production and the economics of software-based business models. Classical technology due diligence mainly examines technical properties such as architecture, scalability, and technical debt. These criteria do not fully capture how AI can affect a company's value proposition, competitive position, margins, or access to customers. This paper develops Artificial Intelligence Exposure and Resilience (AI-ER) as a two-dimensional assessment framework. AI exposure describes the pressure for change that AI creates for a business model. AI resilience describes the company's ability to absorb that pressure, adapt to changed conditions, and use AI in an economically viable way. Metrics for both dimensions are derived from current AI capabilities, their deployment conditions, and relevant research on business models and organizational adaptability. The model keeps exposure and resilience separate and
    
[^77]: Magenta：构建数学推理与Lean验证之间的闭环

    Magenta: Closing the Loop Between Mathematical Reasoning and Lean Verification

    [https://arxiv.org/abs/2609.11319](https://arxiv.org/abs/2609.11319)

    Magenta是一个无需训练的智能体流水线，通过将Lean形式化验证信号整合进LLM的非正式数学推理过程，实现了从自然语言问题到机器检验证明的闭环。

    

    arXiv:2609.11319v1 公告类型：新论文 摘要：大部分数学知识是通过所谓非正式的数学表达和自然语言进行传播的。大语言模型（LLMs）在使用自然语言方面非常熟练，因此在非正式数学推理中取得了强劲但尚不完美的表现。然而，将LLMs局限于非正式推理，就错失了利用机器提供的可机器检验证明的离散验证能力的机会。在本文中，我们通过将Lean信号整合到非正式推理过程中，弥合了非正式推理与形式化推理之间的差距。我们提出了Magenta，这是一个无需训练的智能体流水线，仅需给定一个自然语言问题，就能生成答案、将其表述为Lean 4语句，并构建机器检验的证明。其中一个语句评判器负责验证形式化表述是否保留了原始问题的含义，而一个错误归因评判器则将失败的尝试分流至数学重新推

    arXiv:2609.11319v1 Announce Type: new  Abstract: Most of mathematical knowledge has been communicated through so-called informal use of mathematics and natural language. With large language models (LLMs) being highly adept in using natural language, they achieve strong performance, yet not perfect, in informal mathematical reasoning. Restraining LLMs to informal reasoning misses out on the opportunity to use the discrete verification abilities that machines offer through machine-checkable proofs. In this paper, we bridge the gap between informal and formal reasoning by integrating Lean signals into the informal reasoning process. We introduce Magenta, a training-free agentic pipeline that, given only a natural-language problem, produces an answer, expresses it as a Lean 4 statement, and constructs a machine-checked proof. A statement judge verifies whether the formalisation preserves the original problem, while an error-attribution judge routes failed attempts either to mathematical re
    
[^78]: Mr.LHDR：一个用于多模态真实世界长程深度研究智能体的基准测试

    Mr.LHDR: A Benchmark for Multimodal Real-World Long-Horizon Deep Research Agents

    [https://arxiv.org/abs/2609.11318](https://arxiv.org/abs/2609.11318)

    该论文提出了Mr.LHDR基准测试，通过隐藏节点-关系图构建平均依赖深度达10.4的长程相互依赖证据链，用以评估深度研究智能体在八类多模态真实世界场景中维持漫长研究过程的能力。

    

    深度研究智能体在网页搜索、工具使用、多模态证据分析和信息综合方面的能力日益增强。然而，现有的基准测试主要评估中等时程的探索，很少测试智能体能否维持漫长且依赖性强的研究过程。我们提出了Mr.LHDR（多模态真实世界长程深度研究），这是一个用于评估真实世界深度研究的基准，涵盖八个类别中跨越漫长、不可约简的相互依赖证据链的研究任务。每个问题由一个隐藏的节点-关系图构建，平均需要12.1个必要的中间结论，平均依赖深度为10.4，才能得出简短、唯一且可验证的答案。问题包含多模态证据，包括图像、地图、PDF、标志、图表、表格和视频帧，其中至少包含一个会改变推理状态的非文本元素。Mr.LHDR同时评估最终答案和正确的（推理过程）。

    arXiv:2609.11318v1 Announce Type: new  Abstract: Deep research agents are increasingly capable of web search, tool use, multimodal evidence analysis, and information synthesis. However, existing benchmarks mainly evaluate medium-horizon exploration and rarely test whether agents can sustain long, dependency-heavy research processes. We introduce Mr.LHDR (Multimodal real-world Long-Horizon Deep Research), a benchmark for evaluating real-world deep research over long, irreducible chains of interdependent evidence across eight categories. Each question is constructed from a hidden Node-Relation graph and requires an average of 12.1 necessary intermediate conclusions with a mean dependency depth of 10.4 before reaching a short, unique, and verifiable answer. Questions incorporate multimodal evidence, including images, maps, PDFs, logos, charts, tables, and video frames, with at least one non-text element that changes the reasoning state. Mr.LHDR evaluates both final answers and the correct
    
[^79]: 基于推理需求的路由：面向扩散视觉语言模型的轨迹感知解码控制

    Routing by Reasoning Need: Trajectory-Aware Decoding Control for Diffusion Vision-Language Models

    [https://arxiv.org/abs/2609.11315](https://arxiv.org/abs/2609.11315)

    该论文提出一种免训练的轨迹感知解码控制器，通过分析扩散视觉语言模型的中间答案轨迹信号（答案闭合度、承诺证据、表示修订压力），根据问题各自的推理需求将其动态路由到提前确定、保持基线或推理支持式解码策略，从而解决统一生成长度导致的推理预算不匹配问题。

    

    arXiv:2609.11315v1 公告类型：新论文 摘要：扩散视觉语言模型通过迭代细化的方式生成答案，由此暴露出的中间答案轨迹可以在推理时被检查和控制。然而，这种可控性带来了推理需求不匹配的问题，即对不同推理难度的问题统一采用相同的生成长度。对于视觉封闭型问题，在稳定答案形成后继续细化可能会损害结果；而对于推理敏感型问题，过早确定答案则可能造成损失。我们将该问题形式化为推理预算不匹配问题，并在 LLaDA-V 模型上进行了研究。我们的免训练控制器不选择统一的生成长度，而是利用答案闭合度、承诺证据和表示修订压力等轨迹信号，将每个样本路由到提前确定答案、保持基线或推理支持式解码三种策略之一，且全程不使用真实答案标签。在答案聚焦型、混合推理型……（摘要在此处被截断）

    arXiv:2609.11315v1 Announce Type: new  Abstract: Diffusion vision-language models generate answers through iterative refinement, exposing intermediate answer trajectories that can be inspected and controlled at inference time. However, this controllability creates a reasoning-need mismatch, where a universal generation length is applied to questions with different reasoning demands. Visually closed questions may be harmed by continued refinement after a stable answer has formed, whereas reasoning-sensitive questions may be harmed by premature commitment. We formulate this problem as reasoning-budget mismatch and study it in LLaDA-V. Rather than choosing a universal generation length, our training-free controller routes each example to early commitment, baseline preservation, or reasoning-supportive decoding using trajectory signals from answer closure, commitment evidence, and representation revision pressure, without using ground-truth answers. Across answer-focused, mixed-reasoning, 
    
[^80]: GRIPNet：高斯径向强度先验引导的CT肺结节检测架构

    GRIPNet: Gaussian Radial Intensity Prior Guided Architecture for Pulmonary Nodule Detection in CT

    [https://arxiv.org/abs/2609.11312](https://arxiv.org/abs/2609.11312)

    本文发现肺结节强度呈高斯径向衰减的规律，并据此提出GRIPNet网络，其每个模块均与强度分布的可测量属性相对应，显著提升了CT图像中小于六毫米肺结节的检测能力。

    

    肺癌导致的死亡人数超过任何其他恶性肿瘤，低剂量CT筛查是早期诊断的主要途径。这一途径依赖于最小的病灶，然而小于六毫米的结节仍然难以检测，因为大多数方法将结节视为通用对象，而忽略了其外观背后的成像物理原理。我们证明这种外观是高度规律的：结节强度在其几何中心达到峰值，并以高斯模式径向衰减。对来自三个公开基准数据集的18,218个标注病灶进行拟合，在每个数据集和尺寸分层中，平均径向决定系数均超过0.86。方形卷积在两个轴上均匀采样，与这种径向信号不匹配，这种不匹配对小结节的影响最为严重。基于这一证据，我们提出了GRIPNet（高斯径向强度先验网络），这是一种检测器，其每个模块都对应于强度分布的一个可测量属性。

    arXiv:2609.11312v1 Announce Type: cross  Abstract: Lung cancer causes more deaths than any other malignancy, and low-dose CT screening is the main pathway to early diagnosis. That pathway hinges on the smallest lesions, yet nodules below six millimeters remain hard to detect, because most methods treat a nodule as a generic object and ignore the imaging physics behind its appearance. We show that this appearance is highly regular. Intensity peaks at the geometric center of a nodule and decays radially in a Gaussian pattern, and a fit to 18,218 annotated lesions from three public benchmarks yields a mean radial coefficient of determination above 0.86 in every dataset and size stratum. A square convolution samples both axes uniformly and is mismatched to this radial signal, most severely for small nodules. Guided by this evidence, we propose GRIPNet (Gaussian Radial Intensity Prior Network), a detector in which every module maps to a measurable property of the intensity distribution. Pin
    
[^81]: 你的模型已经知道——不要教它，学会问它：软提示用于视觉语言模型的少样本适配

    Your Model Already Knows Don't Teach It, Learn to Ask It: Soft Prompting for Few-Shot Adaptation of Vision-Language Models

    [https://arxiv.org/abs/2609.11310](https://arxiv.org/abs/2609.11310)

    该论文发现，将少量可学习的软提示词元放置在视觉与文本词元的跨模态边界处并以空格词元初始化，仅需约七千个可训练参数即可在域外少样本目标检测中匹敌最佳LoRA微调配置，同时将训练参数减少超过两万倍。

    

    我们研究在航空、工业和医学影像等域外设置下，使用视觉语言模型（VLM）进行少样本目标检测，仅用十张标注图像作为监督。现有的适配方法包括离散提示优化和LoRA微调。我们重新审视了第三种选择：软提示（soft prompting），即只优化少量连续的提示词元，同时保持预训练骨干网络完全冻结。我们确定了两个关键设计选择：第一，将提示词元放置在视觉词元与文本词元之间的跨模态边界处，其效果优于其他放置方式（10.0 对比 8.4 mAP）；第二，使用空格词元来初始化提示，其效果优于语义初始化和随机初始化。借助这些设计选择，仅一到三个可学习的词元（平均7,168个参数）在Roboflow20-VL数据集上（14.2 mAP，10-shot）即可达到最佳LoRA配置的性能，而训练参数减少了超过20,000倍。软提示仍然更难优化……

    arXiv:2609.11310v1 Announce Type: cross  Abstract: We address few-shot object detection with vision-language models (VLMs) in out-of-domain settings such as aerial, industrial, and medical imagery, using only ten annotated images for supervision. Existing adaptation methods are discrete prompt optimization and LoRA fine-tuning. We revisit a third option: soft prompting, where a small number of continuous prompt tokens are optimized while the pretrained backbone remains frozen.   We identify two key design choices. First, placing prompt tokens at the cross-modal boundary between visual and text tokens outperforms other placements (10.0 vs. 8.4 mAP). Second, initializing prompts from the empty space token outperforms semantic and random initialization.   With these choices, one to three learned tokens (7,168 parameters on average) match the best LoRA configuration on Roboflow20-VL (14.2 mAP, 10-shot) while training over 20,000x fewer parameters. Soft prompting remains harder to optimize,
    
[^82]: 2AM：将智能体侧记忆具化为长时程操作中可引导动作模型的指引

    2AM: Grounding Agent-Side Memory as Guidance for Steerable Action Models in Long-Horizon Manipulation

    [https://arxiv.org/abs/2609.11308](https://arxiv.org/abs/2609.11308)

    2AM提出一种职责分离架构：由多模态智能体独占任务记忆，并将其编译为子任务语言和2D空间提示，用以引导一个仅依赖RGB、无情节记忆的动作模型完成长时程机器人操作，从而实现清晰的性能归因与可引导的动作控制。

    

    长时程机器人操作需要记忆，但记忆并不一定需要置于动作策略内部。为了解决此类任务，当前的智能体系统通常将视觉-语言-动作模型（VLA）与规划器和几何工具相结合，有时还会借助额外的深度信息或标定几何。这些系统使性能归因变得混乱：性能提升可能来自更丰富的观测或替代性的运动工具，而失败则可能源于策略本身或欠规范的语言接口。我们通过一种刻意受限的设计来隔离这一问题：牺牲工具的广度，换取更大的接口带宽。2AM让一个多模态智能体成为任务记忆的唯一持有者，同时让一个仅基于RGB、情节上无状态的动作模型成为任务相关运动的唯一执行者。该智能体将交互历史编译为子任务语言，以及可选的2D抓取、放置和移动提示，从而在不同时间尺度上绑定其物理意图。为了教会VLA这种可引导性，我们通过增强示教数据……（摘要在此处截断）

    arXiv:2609.11308v1 Announce Type: cross  Abstract: Long-horizon robot manipulation requires memory, but not necessarily inside the action policy. To address such tasks, current agentic systems often combine VLAs with planners and geometric tools, sometimes using additional depth or calibrated geometry. These systems confound attribution: gains may come from richer observations or alternative motor tools, while failures may stem from either the policy or an under-specified language interface. We isolate this question through a deliberately constrained design: less tool breadth, but greater interface bandwidth. 2AM makes a multimodal Agent the sole holder of task memory and a single RGB-based, episodically stateless Action Model the sole executor of task-relevant motion. The Agent compiles interaction history into subtask language and optional 2D grasp, place, and move hints that bind its physical intention at different time scales. To teach this steerability to the VLA, we augment demon
    
[^83]: 面向高扇出智能体沙箱的内存压缩

    Memory Compression for High-Fanout Agent Sandboxes

    [https://arxiv.org/abs/2609.11294](https://arxiv.org/abs/2609.11294)

    AgentZip是首个专为AI智能体沙箱设计的内存压缩系统，通过利用模板相对冗余和跨沙箱内存冗余，解决了高扇出智能体工作负载带来的内存瓶颈问题。

    

    高扇出智能体工作负载会造成日益严重的内存瓶颈，因为单个任务可能会派生出大量并发的沙箱会话。然而，这些沙箱远非相互独立：它们源自共享模板并执行相关联的执行轨迹，从而暴露出大量的模板相对冗余和跨沙箱内存冗余。传统内存压缩在三个基本维度上与这一场景不匹配：在“如何压缩”方面，它们无法利用非相同沙箱页面之间的相似性；在“压缩什么”方面，它们通过保守的页面选择来控制缺页开销；在“何时压缩”方面，压缩要么由内存压力触发，要么在不感知智能体执行阶段的情况下进行。我们提出了AgentZip，这是首个专为AI智能体沙箱设计的内存压缩系统。AgentZip引入了能够同时利用模板相对冗余和跨沙箱冗余的压缩机制。

    arXiv:2609.11294v1 Announce Type: new  Abstract: High-fanout agent workloads create a growing memory bottleneck because a single task may spawn many concurrent sandbox sessions. Yet these sandboxes are far from independent: they originate from a shared template and execute related trajectories, exposing substantial template-relative and cross-sandbox memory redundancy. Conventional memory compression is poorly matched to this setting in three fundamental dimensions: how to compress, because they fail to exploit similarity across non-identical sandbox pages; what to compress, because they control page-fault overhead through conservative page selection; and when to compress, because compression is either triggered by memory pressure or performed without awareness of agent execution phases.   We present AgentZip, the first memory compression system designed specifically for AI-agent sandboxes. AgentZip introduces compression mechanisms that exploit both the template-relative and cross-san
    
[^84]: 韩语27B语言模型中响应风格对齐的脱靶效应

    Off-Target Effects of Response-Style Alignment in a Korean 27B Language Model

    [https://arxiv.org/abs/2609.11291](https://arxiv.org/abs/2609.11291)

    在韩语27B语言模型上进行响应风格对齐后训练会产生脱靶效应——模型在弃答和信息披露等未被训练目标涉及的行为上发生系统性改变，且这些改变主要由目标文本本身驱动而非提示集或训练配方。

    

    我们对Qwen3.8-27B进行了面向韩语响应风格的后训练——涵盖冗长度、列表与markdown使用、话语结构和语域——并测量了该训练目标从未针对的两种行为：在KoBBQ基准上对模糊社会问题的弃答（该基准中标注正确答案为UNKNOWN），以及在证券指导中的主动信息披露。两种行为均发生变化，且这些变化主要通过模型的输出策略来表达：即模型回答的频率和回答内容的多少。匹配目标形式的对照实验表明，回答倾向取决于训练目标文本本身，而不仅仅取决于提示集或训练配方。在保持提示、配方、数据量和服务方式固定，仅改变目标文本的情况下，三个风格种子给出了正的回答率点估计（平均+0.82个百分点），而三个中性种子给出了负的点估计（平均-1.53个百分点）；观察到的种子范围互不重叠，均值相差2.34个百分点。长度匹配的对照组介于两者之间，而第四个保持……

    arXiv:2609.11291v1 Announce Type: new  Abstract: We post-train Qwen3.8-27B for Korean response style -- verbosity, list and markdown usage, discourse structure and register -- and measure two behaviours the objective never targets: abstention on ambiguous social questions in KoBBQ, where the benchmark-correct answer is UNKNOWN, and unprompted disclosure in securities guidance. Both move, and the changes are expressed primarily through the model's emission policy: how often it answers and how much it says.   Matched target-form controls show that answer propensity depends on the training target, not the prompt set or recipe alone. Holding prompts, recipe, data volume and serving fixed and changing only the target text, three style seeds give positive answer-rate point estimates (mean +0.82 pp) and three neutral seeds negative ones (mean -1.53 pp); the observed seed ranges do not overlap and the means differ by 2.34 pp. A length-matched arm lies between them, and a fourth arm that stays 
    
[^85]: 生成一个一致的企业：多系统业务数据的合成与无参考评估

    Generating a Consistent Enterprise: Synthesis and Reference-Free Evaluation of Multi-System Business Data

    [https://arxiv.org/abs/2609.11286](https://arxiv.org/abs/2609.11286)

    本文提出一种无需任何真实数据集的企业数据生成器，可根据行业、规模、商业模式等输入生成在66个业务系统中保持实体身份一致性的完整虚构企业数据，并通过五轴评分卡、对抗性检测等无参考方法评估其真实性。

    

    合成关系型数据通常由在真实数据集上训练的模型生成，其质量通过与该数据集的距离来衡量。本文描述了一种在两端都没有真实数据集的生成器。给定一个行业、公司规模、商业模式、一组业务应用和一个随机种子，它会生成一个完整的虚构企业：包括员工队伍、客户群、销售交易、支持工单、通话记录、聊天消息和文档，所有这些数据彼此保持一致。一个实体图被投影到66个业务产品的原生格式中，因此同一个客户在CRM、支持台和通话系统中以同一身份出现。由于不存在真实的对应数据，其真实性通过引用的参考统计数据构建，并通过无参考测量进行验证：包括包含28项统计检查的五轴评分卡、一个寻找合成生成痕迹的对抗性检测器，以及一组……（原文摘要至此截断）

    arXiv:2609.11286v1 Announce Type: new  Abstract: Synthetic relational data is normally produced by a model trained on a real dataset, and its quality is measured as the distance to that dataset. This paper describes a generator that has no real dataset at either end. Given an industry, a company size, a business model, a set of business applications, and a random seed, it produces a complete fictional enterprise: a workforce, a customer base, sales deals, support tickets, recorded calls, chat messages, and documents, all consistent with one another. One entity graph is projected into the native formats of 66 business products, so the same customer appears in the CRM, the support desk, and the call system under one identity. Because no real counterpart exists, realism is built in from cited reference statistics and verified by reference-free measurement: a five-axis scorecard of 28 statistical checks, an adversarial detector that hunts for the marks of synthetic generation, and a set of
    
[^86]: 文本何时提供信息？多模态时间序列预测中信息论度量的基准测试

    When Does Text Inform? Benchmarking Information-Theoretic Metrics for Multimodal Time-Series Forecasting

    [https://arxiv.org/abs/2609.11282](https://arxiv.org/abs/2609.11282)

    该论文首次构建了具有精确已知真值信息量的合成基准数据集，系统评估了六种信息论互信息估计器在判断文本标注是否对多模态时间序列预测有实质贡献方面的可靠性。

    

    结合时间序列与文本标注的多模态预测模型有望通过文本上下文提供更丰富的预测，但我们如何知道一个文本标注是否真正对预测模型的预测有实质性贡献？这是一个信息论问题，而要评估信息论度量能否可靠地衡量标注所提供的预测价值，需要一个真值基准，但这样的基准目前尚不存在。我们构建了一个带有三类标注的合成时间序列信号：语义正确、语义错误和语义无关。由于数据生成过程完全受控，真值信息内容可以被精确得知，从而能够对六种互补的互信息估计器（KSG、MINE、InfoNCE、CCA、PID 和 V-information）进行有原则的评估。我们证明了这六种估计器均能将语义正确的标注识别为信息量最大的标注，并能够对混合文本的质量进行审计。

    arXiv:2609.11282v1 Announce Type: new  Abstract: Multimodal forecasting models that combine time series with text annotations promise richer prediction through textual context, but how do we know whether a text annotation meaningfully contributes to the forecasters prediction? This is an information-theoretic question, but to evaluate whether information-theoretic metrics can reliably measure the predictive value an annotation provides, a ground truth benchmark is needed, and none currently exist. We create a synthetic time series signal with annotations in three categories: semantically correct, incorrect, and irrelevant. Because the data generation process is fully controlled, ground-truth information content is known exactly, enabling principled evaluation of six complementary mutual information estimators (KSG, MINE, InfoNCE, CCA, PID and V-information). We show that all six estimators identify correct annotations as most informative, and are able to audit the quality of mixed text
    
[^87]: 基于概率性存内计算硬件的仿生学习与决策（第一部分）

    Bio-inspired Learning and Decision-Making with Probabilistic In-Memory Computing Hardware: Part 1

    [https://arxiv.org/abs/2609.11281](https://arxiv.org/abs/2609.11281)

    该论文提出一个生物学基础框架，让有噪声的神经与突触动力学通过随机采样执行贝叶斯推理与学习，并指出新兴概率性模拟存内计算硬件的内在电噪声可以复现这种仿生计算机制。

    

    动物的学习与决策过程通常被建模为贝叶斯过程，即感官证据与先验信念相互整合，从而在不确定性面前指导行为。然而，究竟是什么样的内在神经动力学产生了这种能力？它们又该如何在计算系统中被复现？本摘要讨论了一个基于生物学的框架：其中有噪声的神经与突触动力学通过对一个内部能量函数进行随机采样来完成推理与学习，分别通过神经变异性和突触变异性来捕捉潜在状态与模型参数的不确定性。这使得诸如预测编码网络等方法能够通过马尔可夫链蒙特卡洛（MCMC）采样来解释认知不确定性。通过将生物系统中的内在噪声与新兴概率性模拟存储器技术中的电噪声进行类比，我们强调了模拟存内计算硬件……

    arXiv:2609.11281v1 Announce Type: cross  Abstract: Learning and decision-making in animals are often modeled as Bayesian processes, where sensory evidence is integrated with prior beliefs to guide behavior in the face of uncertainty. But what are the inherent neural dynamics that give rise to this ability, and how could they be replicated in computing systems? This abstract discusses a biologically grounded framework in which noisy neural and synaptic dynamics perform inference and learning via stochastic sampling from an internal energy function, capturing uncertainty over latent states and model parameters through neural and synaptic variability, respectively. This enables approaches such as predictive coding networks to account for epistemic uncertainty via Markov chain Monte Carlo sampling. Drawing a parallel between intrinsic noise in biological systems and electrical noise in emerging probabilistic analogue memory technologies, we highlight how analogue in-memory computing hardwa
    
[^88]: 基于机器学习和气象数据的芬兰列车延误预测

    Predicting Train Delays in Finland Using Machine Learning and Weather Data

    [https://arxiv.org/abs/2609.11277](https://arxiv.org/abs/2609.11277)

    本文利用融合铁路运营记录与芬兰全国气象传感器网络观测数据的FI-TW数据集，基于XGBoost模型并引入领域驱动的特征工程，实现了芬兰列车延误的精准预测。

    

    可靠的铁路运营日益依赖于通过无线传感器基础设施提供的实时环境智能，而6G网络将通过集成感知与边缘计算大幅增强这一能力。恶劣天气，尤其是在具有极端气温和强降水的北极地区，仍然是列车延误的主要原因，然而大多数预测方法仅依赖原始气象输入，未能利用基于领域知识的特征工程。本文研究了使用芬兰综合列车-气象（FI-TW）数据集进行列车延误预测的机器学习方法，该数据集将铁路运营记录与芬兰气象研究所约200个通过无线链路通信的全国传感器网络观测数据相融合。我们在奥卢中央车站（共101,146条观测数据）使用XGBoost评估了三种特征配置：完整气象特征……

    arXiv:2609.11277v1 Announce Type: cross  Abstract: Reliable railway operations depend increasingly on real-time environmental intelligence delivered through wireless sensor infrastructures, a capability that 6G networks will substantially enhance through integrated sensing and edge computing. Adverse weather, particularly in Arctic regions with extreme temperatures and heavy precipitation, remains a leading cause of train delays, yet most prediction approaches rely on raw meteorological inputs without exploiting domain-informed feature engineering. This paper investigates machine learning for train delay prediction using the Finland Integrated Train-Weather (FI-TW) dataset, which fuses railway operational records with observations from the Finnish Meteorological Institute's nationwide sensor network of approximately 200 stations communicating over wireless links. We evaluate three feature configurations using XGBoost at Oulu central station (101,146 observations): full weather features
    
[^89]: 利用变分自编码器改进空间态势感知中的微弱目标检测

    Improving Faint Object Detection for Space Situational Awareness with Variational Autoencoders

    [https://arxiv.org/abs/2609.11269](https://arxiv.org/abs/2609.11269)

    本文提出一种结合轻量级分割网络Tiny-U-Net与偏卷积变分自编码器astro-VAE的深度学习流程，通过自动去除恒星并重建天文背景，显著提升光学空间态势感知图像中低信噪比微弱运动目标的检测能力。

    

    本文提出了一种深度学习处理流程，通过自动恒星去除和背景重建来增强光学空间态势感知（SSA）图像中微弱运动目标的检测能力。在光学观测中，检测低信噪比（SNR）目标仍然极具挑战性，尤其是在地月空间（X-GEO）环境中，结构化的天空背景、密集的星场以及散射的月光会显著降低经典检测算法的性能。为解决这一问题，所提出的流程将一个轻量级分割网络组合在一起，其中分割网络用于生成恒星掩膜，而偏卷积变分自编码器则用于学习天文背景的统计分布，并对掩膜区域进行上下文感知的图像修复。重建的背景图可作为预处理步骤，用于抑制固定光源和背景。

    arXiv:2609.11269v1 Announce Type: cross  Abstract: We present a deep-learning pipeline for enhancing the detection of faint moving objects in optical space situational awareness (SSA) imagery through automated star removal and background reconstruction. Detecting low signal-to-noise ratio (SNR) objects remains extremely challenging in optical observations, particularly in the cislunar (X-GEO) environment, where structured sky backgrounds, dense stellar fields, and scattered moonlight significantly degrade the performance of classical detection algorithms. To address this problem, the proposed pipeline combines a lightweight segmentation network (Tiny-U-Net) to generate stellar masks with a partial-convolution variational autoencoder (astro-VAE), designed to learn the statistical distribution of astronomical backgrounds and perform context-aware inpainting of masked regions. The reconstructed background maps can then be used as a preprocessing step to suppress fixed sources and backgrou
    
[^90]: 基于人工智能的火炬燃烧效率估算

    AI-Powered Flare Combustion Efficiency Estimation

    [https://arxiv.org/abs/2609.11262](https://arxiv.org/abs/2609.11262)

    该论文提出一种结合轻量级视觉-语言编码器与多层感知器的AI方法，可直接从低成本热成像视频预测火炬燃烧效率，并通过易于部署的图形界面实现实时监测、分布展示与CSV报告导出，为偏远或预算受限的工业场所提供了替代昂贵传统仪器的实用方案。

    

    在火炬塔中实现高燃烧效率对于遵守监管标准和控制碳氢化合物向环境中的排放至关重要。传统仪器如气体分析仪和高光谱相机价格昂贵、脆弱易损，且需要频繁校准，这使得它们对于偏远地区或预算受限的工业场所而言并不实用。我们提出了一种创新解决方案，将轻量级视觉-语言编码器与紧凑的多层感知器相结合，直接从低成本热成像视频中预测燃烧效率。经过完整训练的模型被集成到一个易于部署的图形用户界面中。该界面将预测的燃烧效率值叠加显示在每个视频帧上，实时展示燃烧效率的变化趋势，显示视频中所有帧的燃烧效率分布情况，并允许用户导出CSV报告。在六个月的时间里，该系统（摘要在此处截断）

    arXiv:2609.11262v1 Announce Type: new  Abstract: Achieving high combustion efficiency in flare stacks is crucial for adhering to regulatory standards and controlling the release of hydrocarbons into the environment. Traditional instruments like gas analyzers and hyperspectral cameras are expensive, fragile, and require frequent calibration, which makes them impractical for remote or budget constrained industrial sites. We propose an innovative solution that combines a lightweight vision-language encoder with a compact multi-layer perceptron to predict combustion efficiency directly from low-cost thermal video footage. The fully trained model is integrated into an easy-to-deploy graphical user interface. This interface overlays predicted combustion efficiency values on each video frame, displays real-time trends in combustion efficiency, shows the distribution of combustion efficiency across all frames in the video, and allows users to export CSV reports. Over a six-month period, the sy
    
[^91]: 生成式重放缓解量子架构搜索中的样本饥饿问题

    Generative Replay Mitigates Sample Starvation in Quantum Architecture Search

    [https://arxiv.org/abs/2609.11248](https://arxiv.org/abs/2609.11248)

    本文提出GenQAS框架，将矩阵乘积态热启动与优先级生成式重放相结合，通过学习到的局部转移模型按需生成合成电路转移并与真实经验混合，从而缓解量子架构搜索中有效训练信号稀缺的问题。

    

    强化学习（RL）可以实现量子架构搜索的自动化，但当有用的电路轨迹在快速扩展的搜索空间中变得稀有时，其可扩展性会受到限制。现有的重放机制只是重用已观测到的状态转移；而本文提出的学习模型能够从真实的状态-动作种子中生成额外的预测单步转移。在此，我们提出了GenQAS，一个张量网络引导的强化学习框架，它将固定的矩阵乘积态热启动与优先级生成式重放相结合。学习到的局部转移模型可按需生成合成的电路转移，并在双重深度Q网络（Double Deep Q-Network）更新过程中将其与真实经验混合。通过随机探索分析表明，接近基态的电路占据的可访问状态空间区域正在快速缩小。我们研究了以真实数据为锚定的合成重放能否在这种情形下改善有效训练信号。在从6（摘要在此处截断）……开始的多个化学哈密顿量基准测试上，……

    arXiv:2609.11248v1 Announce Type: cross  Abstract: Reinforcement learning (RL) can automate quantum architecture search, but its scalability is limited when useful circuit trajectories become rare in the rapidly expanding search space. Existing replay mechanisms reuse observed transitions; the proposed learned model produces additional predicted one step transitions from real state-action seeds. Here we introduce GenQAS, a tensor network-guided RL framework that combines a fixed matrix product state warm-start with prioritized generative replay. A learned local transition model generates synthetic circuit transitions on demand and mixes them with real experience during Double Deep Q-Network updates. Under a random exploration analysis, near ground state circuits occupy a rapidly shrinking region of the accessible state space. We investigate whether real data anchored synthetic replay can improve the effective training signal in this regime. Across chemical Hamiltonian benchmarks from 6
    
[^92]: Sci-MMR：多模态智能体中多步证据支撑科学推理的基准测试

    Sci-MMR: Benchmarking Multi-Step Evidence-Grounded Scientific Reasoning in Multimodal Agents

    [https://arxiv.org/abs/2609.11243](https://arxiv.org/abs/2609.11243)

    Sci-MMR是基于结构化论证图构建的多步证据支撑科学推理基准，对八个前沿多模态模型的评估显示，答案准确率始终高于完整证据恢复能力，揭示了现有模型缺乏可追溯证据支撑的推理能力。

    

    自主研究智能体日益被期望能够检索文献、分析实验证据并生成科学假设。这些能力需要多步的证据支撑推理，即在得出结论之前逐步获取、整合和验证证据。然而，现有的多模态基准主要评估最终答案的准确性，而预测结果是否真正有可追溯的科学证据支撑这一问题仍未得到解决。我们提出了Sci-MMR，这是一个基于结构化论证图的多步证据支撑科学推理基准，该论证图将科学主张、基于引用的知识、视觉证据和支持区域联系起来。Sci-MMR包含235个多跳推理任务，涵盖四个科学学科，平均每个任务包含九个图表面板。通过对八个前沿多模态模型进行评估，我们发现答案准确率始终高于完整证据恢复能力。

    arXiv:2609.11243v1 Announce Type: new  Abstract: Autonomous research agents are increasingly expected to search the literature, analyze experimental evidence, and generate scientific hypotheses. These capabilities require multi-step evidence grounded reasoning that progressively acquires, integrates, and verifies evidence before reaching a conclusion. Existing multimodal benchmarks, however, largely evaluate final-answer accuracy, leaving open whether predictions are actually supported by traceable scientific evidence. We introduce Sci-MMR, a benchmark for multi-step evidence-grounded scientific reasoning built on structured argument graphs linking scientific claims, citation-grounded knowledge, visual evidence, and supporting regions. Sci-MMR comprises 235 multi-hop reasoning tasks spanning four scientific disciplines, with an average of nine figure panels per task. Evaluating eight frontier multimodal models, we find that answer accuracy consistently exceeds complete-evidence recover
    
[^93]: 从评估到增强：面向视频生成模型的“以视频思考”推理能力的基准测试与改进

    From Evaluation to Enhancement: Benchmarking and Improving Think-with-Video Reasoning for Video Generative Models

    [https://arxiv.org/abs/2609.11242](https://arxiv.org/abs/2609.11242)

    该论文提出了涵盖9个推理维度、38个任务的VWG-Bench基准及三级VLM-as-Judge评估协议，揭示了视频生成模型虽渲染能力强但在逻辑与规则推理上存在显著缺陷，并进一步提出Vid-PRE方法来增强模型的“以视频思考”能力。

    

    视频生成技术已发展到能够产生视觉上引人注目且时间上连贯的结果。然而，这些模型是否能够真正“以视频思考”——执行符号规则、遵循物理定律并追求有意图的目标——仍然是一个悬而未决的问题。现有的基准测试只能部分解决这一问题，往往将视觉质量与认知正确性混为一谈。我们提出了VWG-Bench（视频世界通才基准），这是一个涵盖9个推理维度和38个细粒度任务的全面基准测试。为了实现精确诊断，我们设计了一个三级VLM-as-Judge（视觉语言模型作为评判者）协议，独立评估视频级别的流畅度、任务级别的规则遵循度以及样本级别的目标实现程度。对领先模型的评估揭示了一个显著的差距：尽管模型在渲染得分上表现出色，但它们在逻辑密集型和规则受限的任务上始终失败。为解决这一问题，我们提出了Vid-PRE（视频提示推理器与增强器）……

    arXiv:2609.11242v1 Announce Type: cross  Abstract: Video generation has advanced to produce visually compelling and temporally coherent results. Yet, whether these models can genuinely think with video--executing symbolic rules, respecting physical laws, and pursuing intentional goals--remains an open question. Existing benchmarks only partially address this, often conflating visual quality with cognitive correctness. We introduce VWG-Bench (Video World Generalist Benchmark), a comprehensive benchmark spanning 9 reasoning dimensions and 38 fine-grained tasks. To enable precise diagnosis, we design a three-level VLM-as-Judge protocol that independently assesses video-level fluency, task-level rule adherence, and sample-level goal realization. Evaluations of leading models reveal a striking gap: while models achieve strong rendering scores, they consistently fail on logic-heavy and rule-constrained tasks. To address this, we propose Vid-PRE (Video Prompt Reasoner and Enhancer), a model-a
    
[^94]: HALDETECT参加ImageEval 2026共享任务：基于QLoRA的答案优先对比视觉接地方法

    HALDETECT at ImageEval 2026 Shared Tasks: Answer-First Contrastive Grounding with QLoRA

    [https://arxiv.org/abs/2609.11236](https://arxiv.org/abs/2609.11236)

    HALDETECT系统采用“先答后释”的对比决策框架，并通过4位QLoRA微调Qwen2.5-VL-7B-Instruct（冻结视觉编码器），在ImageEval 2026幻觉检测任务中取得CI 0.035、八队中第三名的成绩，同时证明了答案顺序比模型规模更关键、微调适配优于单纯提示工程。

    

    大型多模态模型往往会流畅地生成关于视觉细节的幻觉内容，这限制了它们在细粒度视觉解释场景中的部署应用。我们提出了HALDETECT，这是我们参加ImageEval 2026英语幻觉检测赛道（任务1b）的系统，该任务要求系统从一张图像和三条在文化上看似合理的陈述中，识别出唯一一条具有视觉依据的陈述。我们将该问题构建为一个对比决策任务，先输出答案再给出解释，并围绕颜色/纹理、形状/形态和上下文三个维度组织推理过程。我们提交的最佳适配器使用4位QLoRA对Qwen2.5-VL-7B-Instruct进行微调，同时冻结视觉编码器，在包含1000个项目的测试集上达到了对比不稳定性（CI）0.035的成绩，在八支参赛队伍中排名第三。开发实验表明，答案顺序可能比模型规模更重要，且适配器微调优于单纯的提示工程。对已发布标准答案的回顾性配对分析进一步证实了QLoRA带来的性能提升。

    arXiv:2609.11236v1 Announce Type: cross  Abstract: Large multimodal models tend to hallucinate visual detail fluently, which limits their deployment for fine-grained interpretation. We present HALDETECT, our system for the English hallucination-detection track (Task 1b) of ImageEval 2026, in which a system must identify, from an image and three culturally plausible statements, the single visually grounded one. We frame the item as one contrastive decision, emit the answer before its explanation, and structure reasoning around colour/texture, shape/form, and context. Our best submitted adapter fine-tunes Qwen2.5-VL-7B-Instruct with 4-bit QLoRA while freezing the vision encoder and reaches Contrastive Instability (CI) 0.035 on the 1,000-item test set; we placed third of eight teams. Development experiments show that answer order can matter more than model scale and that adaptation beats prompting alone. Retrospective paired analysis of the released gold labels confirms the QLoRA gain ove
    
[^95]: NovGauge：一个用于诊断大语言模型论文新颖性评估能力的细粒度基准

    NovGauge: A Fine-Grained Benchmark for Diagnosing LLMs' Capability in Paper Novelty Assessment

    [https://arxiv.org/abs/2609.11234](https://arxiv.org/abs/2609.11234)

    NovGauge是一个基于人类专家的细粒度基准，通过任务、问题和方法三个维度的级联诊断流水线来诊断大语言模型在论文新颖性评估中的能力，揭示出模型幻觉率最高可达39%。

    

    大语言模型越来越多地被应用于主要AI会议的同行评审中，然而新颖性评估仍然是一个持续的薄弱环节。现有的基准将新颖性作为单一的整体分数进行评估，这使得难以诊断模型在哪个维度上判断失误，或者其证据是否忠实可靠。我们提出了NovGauge，一个以人类专家为锚定的基准，用于细粒度的新颖性评估诊断。该基准包含619个论文对和50个多论文集合，来源于两个专家渠道：ICLR审稿人的重叠性声明和综述文章的共被引关系。所有实例沿着三个维度独立标注：任务、问题和方法，分别捕捉应用目标、技术挑战和解决方案方法。我们提出了一个级联诊断流水线，用于验证各维度的正确性、证据支撑和逻辑支持。对18个大语言模型的评估显示，各维度的幻觉率介于0%到39%之间，而在非幻觉的评估结果中……

    arXiv:2609.11234v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used in peer review at major AI conferences, yet novelty remains a persistent weak point. Existing benchmarks assess novelty as a single holistic score, making it difficult to diagnose which dimension a model misjudges or whether its evidence is faithful. We present NovGauge, a human-anchored benchmark for fine-grained novelty assessment diagnosis. The benchmark contains 619 paper pairs and 50 multi-paper sets, drawn from two expert sources: ICLR reviewer overlap claims and survey co-citations. Instances are independently labeled along three dimensions: task, problem, and method, capturing application goals, technical challenges, and solution approaches. We propose a cascading diagnostic pipeline that verifies per-dimension correctness, evidence grounding, and logical support. Evaluation of 18 LLMs shows hallucination rates ranging from 0% to 39% across dimensions, and among non-hallucinated 
    
[^96]: 面向智能手术室的多智能体语音交互系统：架构设计与关键技术

    A Voice-Interactive Multi-Agent System for Smart Operating Rooms: Architecture Design and Key Technologies

    [https://arxiv.org/abs/2609.11231](https://arxiv.org/abs/2609.11231)

    本文提出基于大语言模型的智能手术室语音交互多智能体系统SurgicalRoomAgent，通过KV Cache前缀预热、流式JSON解析和渐进式技能提示披露三项关键技术大幅降低推理延迟，实现自然语言理解、设备控制、术中记录与手术报告生成。

    

    本文提出了SurgicalRoomAgent，一个基于大语言模型（LLM）的智能手术室多智能体语音交互系统。该系统通过分层架构实现自然语言理解、设备控制、术中记录和手术报告生成，该架构由语音交互流水线（唤醒、ASR、轮次检测、智能体推理、TTS）和智能体核心（技能注册表、任务规划器、设备管理器）组成。论文研究了三项关键技术：（1）面向低延迟推理的KV Cache前缀预热，通过字节级最长公共前缀重用，将重计算开销从约500毫秒降低至几十毫秒；（2）流式部分JSON解析与早期并行任务执行，将端到端延迟降低约30%；（3）渐进式技能提示披露，根据用户角色、已连接设备和手术阶段动态过滤系统提示，以最大化提示效率和安全性。

    arXiv:2609.11231v1 Announce Type: cross  Abstract: This paper presents SurgicalRoomAgent, a voice-interactive multi-agent system for smart operating rooms based on large language models (LLMs). The system achieves natural language understanding, device control, intraoperative recording, and surgical report generation through a layered architecture comprising a voice interaction pipeline (wake, ASR, turn detection, agent reasoning, TTS) and an agent core (skill registry, task planner, device manager). Three key technologies are investigated: (1) KV Cache prefix warming for low-latency inference, reducing recomputation overhead from approximately 500 ms to tens of milliseconds via byte-level Longest Common Prefix reuse; (2) streaming partial JSON parsing with early parallel task execution, reducing end-to-end latency by approximately 30%; and (3) progressive skill prompt disclosure, which dynamically filters system prompts based on user role, connected devices, and surgical phase to maxi
    
[^97]: 通过迭代顺序迁移求解少样本多目标多任务优化

    Solving Few-Shot Multiobjective Multitask Optimization via Iterative Sequential Transfer

    [https://arxiv.org/abs/2609.11228](https://arxiv.org/abs/2609.11228)

    本文提出迭代顺序迁移（IST）方法，将多任务优化建模为一系列顺序迁移优化问题，每次迭代仅聚焦单一目标，从而克服少样本场景下精英解分布难以识别的瓶颈，有效求解少样本多目标多任务优化问题。

    

    arXiv:2609.11228v1 公告类型: new 摘要：通过在多个优化任务之间应用知识迁移，多任务优化（MTO）成为同时求解协同优化任务的一种有前景的方法。然而，在MTO中开发有效的知识迁移机制从根本上依赖于跨任务对齐精英解分布。这种依赖性在少样本优化场景中造成了关键瓶颈，因为受限的评估预算阻碍了对有益迁移所需的精英解分布的识别。这一挑战在多目标多任务问题中进一步加剧，因为每个优化器必须逼近连续的帕累托流形而非单个最优点。本文提出迭代顺序迁移（IST）来规避这一瓶颈。我们将MTO建模为一系列顺序迁移优化问题，在每次迭代中将评估集中在单个目标上。我们提出一个基于似然的方法……（摘要原文在此处被截断）

    arXiv:2609.11228v1 Announce Type: new  Abstract: Applying knowledge transfer across multiple optimization tasks, multitask optimization (MTO) emerges as a promising approach to solving synergistic optimization tasks simultaneously. However, the development of effective knowledge transfer mechanisms in MTO fundamentally relies on aligning elite solution distributions across tasks. This dependency creates a critical bottleneck in few-shot optimization regimes, as restricted evaluation budgets impede the identification of elite solution distributions required for beneficial transfer. This challenge is exacerbated in multiobjective multitask problems, where each optimizer must approximate a continuous Pareto manifold rather than a single optimal point. This paper introduces Iterative Sequential Transfer (IST) to circumvent this bottleneck. We model MTO as a sequence of sequential transfer optimization problems, concentrating evaluations on a single target per iteration. We propose a likeli
    
[^98]: AI足球分析师：面向足球数据分析的阶段感知且可验证的人机协作

    AI Soccer Analyst: Stage-Aware and Verifiable Human-AI Collaboration for Soccer Data Analysis

    [https://arxiv.org/abs/2609.11224](https://arxiv.org/abs/2609.11224)

    该论文提出了AI足球分析师系统，将足球数据分析分解为数据理解、问题定义、结构化规划等多个可修改、可验证的阶段以实现人机协作，用户研究表明该系统在任务完成质量、可靠性和可验证性方面表现良好。

    

    体育数据分析师通过将计算与特定体育领域的专业知识相结合，将领域问题转化为洞察。大语言模型虽然简化了编程，但“提示到报告”的工作流程可能会掩盖决策过程和证据。我们提出了AI足球分析师，这是一个具有可修改阶段的混合主动式系统，包括：数据理解、问题定义、结构化规划、执行、基于证据的报告以及交互与改进等阶段。一项由五名分析师参与的形成性研究首先为系统的自动化、可验证性、人类控制和可访问性确立了设计目标。随后，一项由16名参与者参与的任务评估结合了系统日志、保留产物、评分和开放式回答；48个任务中有33个满足了操作层面的完成标准。经过Holm校正的探索性测试表明，参与者对已完成任务的输出质量、任务达成度、可靠性和可验证性持积极评价。

    arXiv:2609.11224v1 Announce Type: cross  Abstract: Sports data analysts translate domain questions into insights by combining computation with sport-specific domain expertise. Large language models ease programming, but prompt-to-report workflows may obscure decisions and evidence. We present AI Soccer Analyst, a mixed-initiative system with revisable stages: Data Understanding, Problem Definition, Structured Planning, Execution, Evidence-Grounded Reporting, and Interaction and Refinement. A formative study with five analysts first informed design goals for automation, verifiability, human control, and accessibility. Subsequently, a task-based evaluation with 16 participants combined system logs, retained artifacts, ratings, and open responses; 33 of 48 tasks met the operational completion criteria. Exploratory tests supported favorable participant perceptions of completed-task output quality, task achievement, reliability, and verifiability after Holm correction. Interaction records s
    
[^99]: X-RACE：面向信道估计的XAI辅助循环神经网络归因框架

    X-RACE: XAI-assisted Recurrent neural network Attribution for Channel Estimation

    [https://arxiv.org/abs/2609.11211](https://arxiv.org/abs/2609.11211)

    本文提出X-RACE框架，通过低复杂度的一次性双重优化策略同时剪除LSTM信道估计模型中无关的输入子载波和隐藏单元，并提出饱和时间、重要性漂移和相关性对比等新颖时序XAI指标来表征模型的学习动态与记忆收敛特性。

    

    深度学习模型，尤其是长短期记忆网络（LSTM），在高移动性车联网环境的信道估计中已展现出良好的性能。然而，其黑盒特性和架构开销限制了可信度与效率。经典的可解释人工智能（XAI）方法依赖于代价高昂的迭代过程，且仅提供输入层面的筛选，未能解决架构层面的精简问题。为克服这些局限性，本文提出了XAI辅助循环神经网络归因信道估计（X-RACE）框架。X-RACE采用一种低复杂度的一次性双重优化策略，可同时评估并剪除无关的输入子载波和内部隐藏单元。此外，我们提出了新颖的时序XAI指标——饱和时间、重要性漂移和相关性对比，用于刻画LSTM的学习动态与记忆收敛特性。大量仿真表明，X-RACE（摘要至此截断）

    arXiv:2609.11211v1 Announce Type: cross  Abstract: Deep learning models, notably Long Short-Term Memory (LSTM), have demonstrated promising performance in channel estimation for high-mobility vehicular environments. However, their black-box nature and architectural overhead limit trustworthiness and efficiency. Classical explainable AI (XAI) methods rely on costly iterative processes, offering only input-level filtering without addressing architectural fine-tuning. To overcome these limitations, this paper proposes the XAI-assisted Recurrent neural network Attribution for Channel Estimation (X-RACE) framework. X-RACE uses a low-complexity, one-shot dual-optimization strategy to simultaneously evaluate and prune irrelevant input subcarriers and internal hidden units. Furthermore, we propose novel temporal XAI metrics: Saturation Time, Importance Drift, and Relevance Contrast to characterize the LSTM's learning dynamics and memory convergence. Extensive simulations demonstrate that X-RAC
    
[^100]: CryptoL：面向金融多变量时间序列预测中的规模主导性与物理约束缓解

    CryptoL: Towards Scale Dominance and Physics Constraints Mitigation in Financial Multivariate Time Series Forecasting

    [https://arxiv.org/abs/2609.11206](https://arxiv.org/abs/2609.11206)

    CryptoL是一个统一的加密货币多变量时间序列预测框架，其核心创新在于在RevIN流程的上下文归一化坐标系中评估预测误差，从理论和实证上消除极端规模差异导致的大规模资产对共享模型优化的不成比例影响，并证明共享的通道依赖仿射变换更适合OHLC数据的归一化。

    

    加密货币预测呈现出一种独特的挑战组合：极端的跨资产规模异质性、非平稳动态，以及开盘价、最高价、最低价和收盘价（OHLC）变量之间的结构性依赖关系。我们提出了CryptoL，这是一个旨在解决多变量时间序列预测中这些挑战的统一框架。CryptoL在RevIN流程内的上下文归一化坐标系中评估预测误差，防止逆归一化向MSE目标函数引入额外的平方尺度加权。我们通过经验风险和参数梯度几何对该效应进行了形式化刻画，确立了大规模资产可能不成比例地影响共享模型优化的条件。除了损失空间归一化之外，CryptoL还研究了OHLC数据的通道独立和通道依赖归一化，结果表明共享的通道依赖仿射变换能够保持……

    arXiv:2609.11206v1 Announce Type: cross  Abstract: Cryptocurrency forecasting presents a distinctive combination of extreme cross-asset scale heterogeneity, non-stationary dynamics, and structural dependencies among Open, High, Low, and Close (OHLC) variables. We present CryptoL, a unified framework designed to address these challenges within multivariate time-series forecasting. CryptoL evaluates forecasting error in context-normalized coordinates within the RevIN pipeline, preventing inverse normalization from introducing an additional squared-scale weighting into the MSE objective. We formally characterize this effect through the empirical risk and parameter-gradient geometry, establishing the conditions under which large-scale assets can disproportionately influence shared-model optimization. Beyond loss-space normalization, CryptoL examines channel-independent and channel-dependent normalization for OHLC data, showing that a shared channel-dependent affine transformation preserves
    
[^101]: 基于NLP和机器学习的面向巴基斯坦大学生的AI驱动文化感知压力检测与健康支持聊天机器人

    An AI-Powered Culturally Aware Chatbot for Stress Detection and Wellness Support among Pakistani University Students Using NLP and Machine Learning

    [https://arxiv.org/abs/2609.11199](https://arxiv.org/abs/2609.11199)

    本文提出了一款专为巴基斯坦大学生设计的文化感知AI聊天机器人，利用随机森林模型以89.09%的准确率实现三级压力检测，并结合开源大语言模型提供心理健康支持。

    

    由于现有的数字心理健康工具是专门针对西方环境开发的，巴基斯坦大学生在大学中面临着独特的复合压力情境，其中包括学业、经济、家庭和人际关系等多重压力源，这已成为影响巴基斯坦大学生学业与心理发展的严重问题。本文介绍了一种新型的、AI驱动的、具有文化敏感性的压力检测与健康支持系统，该系统专为巴基斯坦大学生的实际情境量身定制。该系统基于名为随机森林的机器学习模型，使用一个经过验证的学生压力数据集进行训练，该数据集包含1100份回复，涵盖心理、生理、学业、环境和社会五个方面的20个特征，在三个压力严重程度等级上实现了89.09%的准确率和0.89的宏F1分数。分类结果随后被传递给一个开源大语言（模型）……

    arXiv:2609.11199v1 Announce Type: new  Abstract: With the existing digital mental health tools specifically developed for Western settings, Pakistani students are exposed to a uniquely compounded stress situation in their university that includes academic, financial, familial, and relational stressors, which have become a serious concern for academic and psychological development of students in Pakistani universities. This paper introduces a new, AI-driven and culturally sensitive stress detection and wellness support system that is tailored to the context of Pakistani university students. The system is based on a machine learning model called Random Forest which is trained using a validated student stress data set of 1100 responses on 20 features from psychological, physiological, academic, environmental and social aspects, with an accuracy of 89.09% and a macro F1-score of 0.89, in three stress severity levels. The classification outputs are passed on to an open-source large language
    
[^102]: （谁的默认设置？）人工智能正在重塑考古学方法吗？

    (Whose defaults?) Is artificial intelligence reorienting archaeological methods?

    [https://arxiv.org/abs/2609.11198](https://arxiv.org/abs/2609.11198)

    该研究分析了约11.9万篇考古学摘要并运用贝叶斯狄利克雷-多项式模型，发现大语言模型兴起后（2023年后）考古学计算方法仅有微小转变，且方法多样性不降反升，表明AI并未窄化学科的研究方法范围。

    

    生成式人工智能和“氛围编程”实践正在改变考古学家开展计算研究的方式，但其对学科方法范围的影响仍未得到充分研究。本文评估了大型语言模型（LLM）是否正在缩小考古学家所使用方法的多样性。我们首先分析了来自Scopus的约119,000篇考古学摘要，涵盖2010年至2025年的出版物。利用本地运行的大语言模型，我们识别了每篇摘要中报告的计算方法，并将其归纳为25个宽泛类别（L2）和241个更细的聚类（L3）。基于子学科内方法构成的贝叶斯狄利克雷-多项式模型发现，2023年之后方法使用出现了小幅但可信的转变。然而，这一转变幅度小于整个研究期内已存在的变异。没有任何单一技术显示出显著变化，且整体方法多样性是增加而非减少。

    arXiv:2609.11198v1 Announce Type: cross  Abstract: Generative AI and the practice of "vibe coding" are changing how archaeologists carry out computational research, but their effects on the discipline's range of methods is still understudied. In this paper, we evaluate whether large language models (LLMs) are narrowing the variety of methods archaeologists use. We first analysed approximately 119,000 archaeology abstracts from Scopus, covering publications from 2010 to 2025. Using a locally run LLM, we identified the computational methods reported in each abstract and organised them into 25 broad categories (L2) and 241 finer clusters (L3). A Bayesian Dirichlet-multinomial model of method composition within sub-disciplines found a small but credible shift in method use after 2023. However, this shift was smaller than the variation already present across the full study period. No individual technique showed a significant change, and overall methodological diversity increased rather than
    
[^103]: 智能体化搜索份额：面向LLM介导电商竞争决策的多智能体AI系统

    Agentic Share-of-Search: A Multi-Agent AI System for Competitive Decision-Making in LLM-Mediated E-Commerce

    [https://arxiv.org/abs/2609.11190](https://arxiv.org/abs/2609.11190)

    本文提出一个多智能体AI系统，以“智能体化搜索份额”为决策目标，自动化测量卖家在AI购物助手中的竞争可见性并诊断根因，实验表明其诊断能力显著优于随机水平。

    

    AI购物助手正日益改变消费者的商品发现方式，这迫切需要支持卖方竞争决策的工具。我们提出了一个多智能体AI系统，用于自动化测量LLM介导电商中的竞争可见性并进行根因诊断。该系统引入了“智能体化搜索份额”作为决策目标，在各大主流AI平台上部署查询智能体，并使用基于ReAct的诊断智能体来推荐按优先级排序的商品化干预措施。一项100次试验的消融研究（作为该原型的可行性评估）显示，该智能体在39%的试验中成功恢复了被消融的信号（95% CI: 30.0% - 48.8%，是随机概率的5.5倍），而在高相关性消融试验中这一比例上升至63.9%。

    arXiv:2609.11190v1 Announce Type: cross  Abstract: AI shopping assistants increasingly redirect consumer discovery, creating an urgent need for tools that support seller-side competitive decision-making. We present a multi-agent AI system that automates competitive visibility measurement and root cause diagnosis in LLM-mediated ecommerce. The system introduces Agentic Share-of-Search (ASoS) as the decision target, deploys query agents across leading AI platforms, and uses a ReAct-based diagnostic agent to recommend prioritized merchandising interventions. A 100-trial ablation study, presented as a feasibility evaluation of this prototype, shows the agent recovers the ablated signal in 39% of trials (95% CI: 30.0% - 48.8%, 5.5x over chance), rising to 63.9% among high-correlation ablations.
    
[^104]: 大语言模型能否遵循医学专家逻辑？偏倚风险评估中层级逻辑一致性的基准测试

    Can LLMs Follow Medical Expert Logic? A Benchmark for Hierarchical Logical Consistency in Risk-of-Bias Assessment

    [https://arxiv.org/abs/2609.11185](https://arxiv.org/abs/2609.11185)

    该论文提出了基于Cochrane偏倚风险评估专家逻辑的LogiMed-RoB基准，揭示了大语言模型存在灾难性的错误复合效应——即使单步逻辑一致性高达98.88%，端到端一致性也会骤降至45%甚至接近0%，且模型即使检索到高质量证据也无法正确推理。

    

    arXiv:2609.11185v1 公告类型：新论文 摘要：循证医学要求严格的逻辑一致性，然而当前对大语言模型（LLM）的评估优先关注表面的标签匹配而非真正的推理能力。我们提出了LogiMed-RoB，一个基于Cochrane偏倚风险（RoB）2.0专家逻辑构建的基准测试，包含860项随机对照试验（RCT）和14,820个查询。该基准在层级逻辑一致性（HLC）框架下从四个维度评估模型：原子一致性、领域一致性、聚合一致性和证据忠实性。对10个最先进大语言模型的实验揭示了一个灾难性的错误复合效应：尽管表现最好的模型达到了98.88%的原子一致性，其端到端一致性却骤降至45.13%，多个开源权重架构更是暴跌至接近0%。我们进一步揭示了一个系统性的证据-推理鸿沟：即使模型检索到了高质量证据，它们仍无法在18.63%-（原文在此处被截断）

    arXiv:2609.11185v1 Announce Type: new  Abstract: Evidence-based medicine demands strict logical consistency, yet current evaluations of large language models (LLMs) prioritize superficial label matching over genuine reasoning. We introduce LogiMed-RoB, a benchmark grounded in Cochrane Risk of Bias (RoB) 2.0 expert logic, comprising 860 randomized controlled trials (RCTs) and 14,820 queries. It evaluates models under the Hierarchical Logical Consistency (HLC) framework across four dimensions: Atomic Consistency, Domain Consistency, Aggregation Consistency, and Evidential Faithfulness. Experiments on 10 state-of-the-art LLMs reveal a catastrophic Error Compounding Effect: despite the top model reaching 98.88% Atomic Consistency, its end-to-end consistency collapses to 45.13%, with several open-weight architectures plummeting to nearly 0%. We further uncover a systematic evidence-reasoning gap: even when models retrieve high-quality evidence, they fail to deduce correct outcomes in 18.63-
    
[^105]: 探索说话人识别中的二阶模式识别

    Exploring Second-Order Pattern Recognition in Speaker Recognition

    [https://arxiv.org/abs/2609.11182](https://arxiv.org/abs/2609.11182)

    该论文提出利用层次聚类来发现说话人识别网络中将话语识别为说话人身份时潜藏的“二阶模式”，并通过HCCM方法对其语义解释，进而提出“二阶模式识别”这一新任务。

    

    在经典模式识别任务中，神经网络被训练用于识别模型输入中人类定义的模式。一些可解释人工智能（XAI）方法能够解释潜藏在网络将输入识别为人类定义模式这一过程背后的其他潜在模式；在这项工作中，我们将这些潜在模式称为二阶模式，并提出对它们进行发现。为此，我们应用层次聚类算法来分析说话人识别网络从语音话语中学习到的表示是否自然形成层次聚类。每个由此产生的聚类代表一种二阶模式，它刻画了网络如何将一些已知话语识别为特定说话人身份。随后，我们使用现有的层次聚类-类别匹配（HCCM）方法对所有得到的二阶模式进行语义解释。此外，我们提出了一个新任务——二阶模式识别，用于识别所发现的……

    arXiv:2609.11182v1 Announce Type: cross  Abstract: In classical pattern recognition tasks, neural networks are trained to recognise human-defined patterns for model inputs. Some Explainable AI (XAI) methods can explain other latent patterns that underlie the network's recognition of inputs as human-defined patterns; in this work, we call these latent patterns second-order patterns, and we propose to discover them. To this end, we apply a hierarchical clustering algorithm to analyse whether representations learned by a speaker recognition network from utterances naturally form hierarchical clusters. Each resulting cluster represents a second-order pattern that characterises how the network recognises some known utterances as speaker identities. All the resulting second-order patterns are then semantically interpreted using the existing Hierarchical Cluster-Class Matching (HCCM) method.   Furthermore, we propose a new task, second-order pattern recognition, to identify which discovered s
    
[^106]: SemVerBench：基准测试大语言模型对版本约束解析语义的理解

    SemVerBench: Benchmarking LLM Comprehension of Version-Constraint Resolution Semantics

    [https://arxiv.org/abs/2609.11180](https://arxiv.org/abs/2609.11180)

    本文提出首个跨 npm、PEP 440 和 Cargo 三个生态系统的版本约束解析语义基准测试 SemVerBench，通过对 240 个机器可验证项目的评估，发现六个前沿大语言模型在版本约束语义处理上存在系统性盲点，如 Cargo 部分比较器进位规则使所有模型准确率降至约 60%。

    

    大语言模型（LLM）编程智能体经常需要判断某个版本是否满足诸如 ^1.2.3 或 >=2.0,<3 之类的约束，然而它们对版本约束语义的掌握程度从未被直接测量过。我们提出了 SemVerBench，这是首个针对大语言模型版本约束解析语义的基准测试，涵盖三个生态系统（npm、PEP 440、Cargo）：包含 240 个具有唯一答案的机器可验证项目，构建时保持作者中立，来自四个均衡的来源（每个生态系统的官方测试套件加上三个前沿大语言模型提议者），并由非循环的双重实现“神谕”进行标注。通过评估六个前沿模型，我们发现存在系统性的、可预测的按机制划分的盲点：部分比较器的进位规则（>1.2 意味着 >=1.3.0）使所有模型在 Cargo 上受困（准确率接近 60%）；尽管标准的 PEP 440 前缀匹配被普遍掌握，但在零填充/后发布版本的边缘情况上，GPT-5.1 表现崩溃（0/26），而 Claude 保持在 97-100%（在包含 67 个项目的神谕验证集上得到验证）。

    arXiv:2609.11180v1 Announce Type: cross  Abstract: Large language model (LLM) coding agents constantly decide whether a version satisfies a constraint such as ^1.2.3 or >=2.0,<3, yet their grasp of version-constraint semantics has never been measured directly. We introduce SemVerBench, the first benchmark of LLM version-constraint resolution semantics across three ecosystems (npm, PEP 440, Cargo): 240 machine-checkable items with unique answers, built author-neutrally from four balanced sources (each ecosystem's official test suite plus three frontier LLM proposers) and labeled by a non-circular two-implementation oracle. Evaluating six frontier models, we find systematic, predictable per-mechanism blind spots: a partial-comparator carry rule (>1.2 means >=1.3.0) traps every model on Cargo (near 60%), and although standard PEP 440 prefix matching is universal, on zero-pad/post-release corner cases GPT-5.1 collapses (0/26) while Claude stays at 97-100% (verified on a 67-item oracle-vali
    
[^107]: Debate-to-Skill：面向工业级查询到智能体标注的能力边界过程监督

    Debate-to-Skill: Capability-Bound Process Supervision for Industrial Query-to-Agent Annotation

    [https://arxiv.org/abs/2609.11176](https://arxiv.org/abs/2609.11176)

    该论文提出 Debate-to-Skill 方法，将工业查询到智能体标注形式化为能力边界过程监督，通过可复用决策原则、结构化审议、验证器裁定提取和分歧驱动优化，解决了语义相关性与可执行能力相混淆的问题，尤其提升了长尾和灰色地带请求的标注效果。

    

    当主题相关性被误认为可执行能力时，工业级的查询到智能体匹配就会失败，尤其是在长尾请求和边界敏感的请求上。我们将标注问题形式化为“能力边界过程监督”，并通过 Debate-to-Skill 方法加以实现，该方法采用可复用的决策原则、结构化审议、基于验证器的裁定提取以及分歧驱动的迭代优化。在工业级 Query2Agent 基准测试上，我们将 Debate-to-Skill 与直接标签监督、推理式SFT（监督微调）以及结构性消融实验进行了比较。结果检验了性能提升是否来源于对能力关键决策过程本身的监督，尤其是在语义相关性与可执行能力出现分歧的灰色地带案例上。

    arXiv:2609.11176v1 Announce Type: new  Abstract: Industrial query-to-agent matching fails when topical relevance is mistaken for executable capability, especially on long-tail and boundary-sensitive requests. We formulate annotation as \emph{capability-bound process supervision} and instantiate it with Debate-to-Skill, which uses reusable decision principles, structured deliberation, verifier-based verdict extraction, and disagreement-driven refinement. On an industrial Query2Agent benchmark, we compare Debate-to-Skill with direct-label supervision, reasoning-SFT, and structural ablations. The results test whether gains come from supervising the capability-critical decision process itself, especially on grey-zone cases where semantic relatedness and executable capability diverge.
    
[^108]: 超越视觉质量：使用EgoGenEval评估自我运动下的物理一致性

    Beyond Visual Quality: Evaluating Physical Consistency under Ego-Motion with EgoGenEval

    [https://arxiv.org/abs/2609.11172](https://arxiv.org/abs/2609.11172)

    本文提出基于几何、无需位姿的基准EgoGenEval，用于评估视觉生成器在自我运动下的物理一致性，并通过大规模实验揭示当前模型难以在执行相机运动的同时保持场景状态，没有系统在两方面均表现良好。

    

    近期的视觉生成器能够生成高保真度的图像，但在自我运动条件下常常违反物理一致性，限制了其在空间推理和具身规划中的应用。现有基准大多关注孤立图像或单步生成质量，使这一挑战未被充分探索。我们提出了EgoGenEval，一个基于几何、无需位姿的基准，旨在评估视觉生成器在自我运动下的物理一致性，并将研究组织为两个部分。（1）EgoGenEval包含1,400个案例和2,360个目标视图，涵盖单步和多步自我运动。它分别测量相机运动落地和场景状态保持，两项指标均通过与盲测人类判断的对比进行了验证。对16个无位姿生成器以及两个位姿条件参考模型的评估表明，当前模型难以在执行相机运动的同时保持场景状态，且没有系统在两方面均表现良好……

    arXiv:2609.11172v1 Announce Type: cross  Abstract: Recent visual generators produce high-fidelity images yet often violate physical consistency under ego-motion, limiting their use for spatial reasoning and embodied planning. Existing benchmarks largely focus on isolated images or single-step quality, leaving this challenge underexplored. We introduce EgoGenEval, a geometry-grounded, pose-free benchmark designed to evaluate the physical consistency of visual generators under ego-motion, and organize our study into two parts. (1) EgoGenEval contains 1,400 cases and 2,360 target views spanning single-step and multi-step ego-motion. It separately measures Camera Motion Grounding (CMG) and Scene State Preservation (SSP), with both metrics validated against blinded human judgments. Evaluating 16 pose-free generators together with two pose-conditioned references reveals that current models struggle to execute camera motion while maintaining scene state, and that no system performs well on bo
    
[^109]: 仅破坏预测是不够的：面向时序图的指定替代结果反事实解释

    Breaking Predictions Is Not Enough: Specified-Foil Counterfactuals for Temporal Graphs

    [https://arxiv.org/abs/2609.11170](https://arxiv.org/abs/2609.11170)

    该论文提出“指定替代结果反事实”新范式，通过对比原始预测与目标替代预测的执行轨迹差异来生成定向的低成本干预操作，使时序图预测器输出用户预先指定的特定结果，而不仅仅是破坏原有预测。

    

    时序图反事实解释通常通过改变过去的事件来改变或使原始预测失效，但并不指定替代结果是什么。然而，面对某个预测结果的用户往往想知道：哪些过去条件会使得某个特定的替代结果发生。我们将这一目标明确的问题形式化为“指定替代结果反事实”：给定原始预测 A 和在搜索前固定的替代结果 B，寻找一种低成本的过去事件干预，使得同一个预测器将 B 选为排名第一的结果。我们的轨迹引导干预搜索方法将 A 的完整执行轨迹与重构出的 B 的不完整执行轨迹进行对比，将二者的差异映射为 DELETE（删除）、INSERT（插入）、REWIRE（重连）、RELABEL（重标记）和 SHIFT（移位）等操作，并通过精确回放来验证 B。我们在连续时间动态图上以 LiFTER、在时序知识图谱上以 TLogic 实例化了这一原则。在连续时间动态图上，该方法保留了黑盒贪婪搜索 85.7%-93.6% 的性能。

    arXiv:2609.11170v1 Announce Type: new  Abstract: Temporal graph counterfactual explanations typically change past events to change or invalidate an original prediction, while leaving its replacement unspecified. Yet a user facing a predicted outcome often asks which past conditions would make a particular alternative occur instead. We formulate this destination-specific question as the Specified-Foil Counterfactual: given an original prediction A and a foil B fixed before search, find a low-cost past-event intervention under which the same predictor selects B as top-ranked. Our trace-guided intervention search contrasts the completed execution of A with a reconstructed incomplete execution of B, maps their difference to DELETE, INSERT, REWIRE, RELABEL, and SHIFT operations, and verifies B through exact replay. We instantiate this principle with LiFTER on continuous-time dynamic graphs and TLogic on temporal knowledge graphs. On CTDGs, the method retains 85.7-93.6% of black-box greedy s
    
[^110]: DRG-MAPPO：面向协同空战的分层动态角色图多智能体强化学习

    DRG-MAPPO: Hierarchical Dynamic Role-Graph Multi-Agent Reinforcement Learning for Cooperative Air Combat

    [https://arxiv.org/abs/2609.11155](https://arxiv.org/abs/2609.11155)

    提出DRG-MAPPO框架，通过将基于图的关系建模与动态角色分配相结合，解决了协同空战中战场实体交互建模缺失和战术角色分配模糊两大难题。

    

    多智能体强化学习（MARL）已成为自主系统和空战中复杂决策的关键范式。尽管MARL在空战中展现出了巨大的潜力，但实现复杂的战术协同仍然是一项具有挑战性的任务。这一困难主要归因于两个核心限制：（1）缺乏结构化的关系建模，阻碍了智能体捕捉战场实体之间复杂且随时间变化的交互关系；（2）传统的扁平化架构通常缺乏显式建模战术角色的能力，导致在高度动态的环境中任务分配模糊不清。为应对这些挑战，我们提出了分层动态角色图多智能体近端策略优化（DRG-MAPPO），这是一种将基于图的关系建模与动态角色分配相融合的新型MARL框架。具体而言，DRG-MAPPO构建了基于图的表示

    arXiv:2609.11155v1 Announce Type: new  Abstract: Multi-Agent Reinforcement Learning (MARL) has emerged as a pivotal paradigm for complex decision-making in autonomous systems and air combat. While MARL has demonstrated significant potential in air combat, achieving sophisticated tactical coordination remains a non-trivial challenge. This difficulty is largely attributed to two primary limitations: (1) the absence of structured relational modeling hinders agents from capturing complex, time-varying interactions among battlefield entities; and (2) conventional flat architectures often lack the capability to explicitly model tactical roles, leading to ambiguous task allocation in highly dynamic environments. To address these challenges, we propose Hierarchical Dynamic Role-Graph Multi-Agent Proximal Policy Optimization (DRG-MAPPO), a novel MARL framework that integrates graph-based relational modeling with dynamic role assignment. Specifically, DRG-MAPPO constructs a graph-based represent
    
[^111]: terms.txt：一种用于代理式网络访问的授权与补偿协议

    terms.txt: A Consent and Compensation Protocol for Agentic Web Access

    [https://arxiv.org/abs/2609.11152](https://arxiv.org/abs/2609.11152)

    提出terms.txt协议，通过robots.txt风格的机器可读文件与源站强制执行的交换机制（涵盖Web Bot Auth身份验证、签名意图、HTTP 402定价协商和签名回执），为AI代理访问网络建立按路径、按用途的授权与补偿条款，且实现开销极低。

    

    开放网络一直运行在一个不成文的契约之上：网站允许爬虫访问，搜索引擎则向网站回馈访客。公开测量数据显示，这一契约正在AI爬虫和智能代理的冲击下瓦解。目前自动化客户端已构成大多数请求，训练用途主导着Cloudflare所分类的爬取行为，而最大的AI平台每回馈一个访客就要抓取数千个页面。网络通用的控制手段robots.txt无法表达身份、目的、条款或价格，且可以被绕过，而较新的替代方案大多属于专有的CDN功能。我们提出了terms.txt，一种robots.txt风格的文件，用于定义按路径、按用途的机器访问条款，并配有一套由源站强制执行的交换机制，采用Web Bot Auth签名、签名意图、委托令牌、HTTP 402协商和签名回执。我们定义了该交换机制能够强制执行、可审计以及留待合同约定的内容。一个无依赖的实现方案在单vCPU上每个请求仅增加0.20至0.65毫秒的开销。

    arXiv:2609.11152v1 Announce Type: cross  Abstract: The open web ran on an unwritten bargain: sites admitted crawlers, and search engines sent visitors back. Public measurements show that bargain breaking under AI crawlers and agents. Automated clients now make up most requests, training dominates Cloudflare-classified crawling, and the largest AI platforms fetch thousands of pages for each visitor they return. The web's common control, robots.txt, cannot express identity, purpose, terms, or price, can be circumvented, and newer alternatives are largely proprietary CDN features. We specify terms.txt, a robots.txt-style file for per-path, per-purpose machine-access terms, plus an origin-enforced exchange using Web Bot Auth signatures, signed intent, delegation tokens, HTTP 402 negotiation, and signed receipts. We define what the exchange can enforce, audit, and leave to contract. A dependency-free implementation adds 0.20 to 0.65 ms per request on one vCPU.
    
[^112]: 递归语言模型训练的脆弱性谱系

    A Fragility Spectrum for Recursive Language-Model Training

    [https://arxiv.org/abs/2609.11149](https://arxiv.org/abs/2609.11149)

    该研究让13个公开模型在固定递归污染协议下共享语料库繁衍五代，发现不同模型对坍塌的脆弱性存在约五倍差异，且该脆弱性排序在不同数据组成和随机种子下高度稳定，表明易坍塌性是模型本身的固有属性。

    

    模型生成的文本正在回流到训练语料库中，大量证据表明，反复使用这类数据进行训练会导致输出多样性的坍塌。先前的工作研究了这一现象本身：哪些训练协议以及哪些数据混合方式会引发坍塌。但在相同的过程下，不同模型的表现差异巨大。我们固定了一种递归污染协议，让13个公开发布的模型检查点组成一个生态系统，共享同一语料库并繁衍五代。五代之后，各检查点的唯一4-gram指标从0.187到0.940不等，约有五倍的差距：一些模型几乎未受影响，另一些则退化为重复的片段。改变共享语料池的组成或混入人类文本时，模型排序的Spearman相关性保持在0.91–0.97；改变随机种子时，该相关性保持在0.93–0.98。因此，模型在递归训练下是否容易坍塌，是模型本身固有的一种属性。

    arXiv:2609.11149v1 Announce Type: new  Abstract: Model-generated text is finding its way back into training corpora, and there is plenty of evidence that training on such data over and over collapses output diversity. Prior work has studied the phenomenon itself: which protocols and which data mixtures cause collapse. But different models behave very differently under the same process. We fix one recursive contamination protocol and let 13 publicly released checkpoints form an ecosystem that shares a common corpus for five generations. The unique 4-gram outcome after five generations ranges from 0.187 to 0.940 across checkpoints, a roughly five-fold spread: some models are barely touched, others degenerate into repetitive fragments. Changing the composition of the shared pool or mixing in human text keeps the Spearman correlation of the ordering at 0.91--0.97, and changing the random seed keeps it at 0.93--0.98. Whether a model collapses easily under recursive training is, then, a prop
    
[^113]: 通过智能体推理与验证实现自主化学机理发现

    Autonomous Chemical Mechanistic Discovery through Agentic Reasoning and Validation

    [https://arxiv.org/abs/2609.11147](https://arxiv.org/abs/2609.11147)

    本文提出ARCHE自主智能体系统，通过整合通用推理模型、专用计算化学模型与工具注册表，实现了从假设生成到自我验证的化学反应机理自动化发现闭环。

    

    揭示反应机理是现代化学的核心，但由于计算工作流程仍然严重依赖专家干预，实现这些研究的自动化依然充满挑战。本文提出了ARCHE——一个自主智能体系统，它集成了通用推理模型、领域专用计算化学模型以及结构化工具注册表，将机理探究转变为一个可扩展、可自我验证的过程。ARCHE能够解读科学问题、生成并对机理假设进行优先级排序、编排计算工作流程，并在闭环中基于计算证据迭代地完善结论。研究团队在三个难度递增的场景中验证了其能力：重建立体控制过渡态并验证先前报道的不对称催化反应的相应反应机理；提出并验证一种合理的自由基路径……

    arXiv:2609.11147v1 Announce Type: new  Abstract: Unraveling reaction mechanisms is central to modern chemistry, yet automating these investigations remains challenging because computational workflows still rely heavily on expert intervention. Here we introduce ARCHE, an autonomous agentic system that integrates a general-purpose reasoning model, a domain-specialized computational chemistry model, and a structured tool registry to transform mechanistic inquiry into a scalable, self-validating process. ARCHE interprets scientific questions, generates and prioritizes mechanistic hypotheses, orchestrates computational workflows, and iteratively refines conclusions based on computed evidence within a closed loop. We validate its capabilities across three increasingly demanding scenarios: reconstructing stereocontrolling transition states and validating the corresponding reaction mechanism in a previously reported asymmetric catalytic reaction; proposing and validating a plausible radical pa
    
[^114]: 寡头在多模型生态系统中几乎无法左右模型崩溃

    The Oligarch Barely Steers Model Collapse in Multi-Model Ecosystems

    [https://arxiv.org/abs/2609.11146](https://arxiv.org/abs/2609.11146)

    在多模型递归训练的生态系统中，即使将寡头模型的市场份额推高至90%，既不会加速模型崩溃，也不会使其他模型被拖向寡头的输出分布——模型崩溃的动态对市场份额集中度表现出不变性。

    

    AI生成的文本正在回流到下一代模型的训练语料库中，对其实施的递归训练会导致模型崩溃。近期工作将这一研究场景扩展到多个模型相互喂食的情况——但几乎所有研究都假设市场份额均分，而现实中的生成式AI行业实为寡头垄断格局。市场集中度引发了两个担忧：更少、更同质化的来源可能加速模型崩溃，且后续的模型可能被拖向寡头模型的输出分布。我们在受控生态系统中对这两点进行了检验：13个开源的1—4B参数模型构成了包含3到13个参与者的自然生态系统，另加入一个注入的探针模型，将头部模型的市场份额推高至90%；在每一代中，所有模型的输出按各自市场份额混入共享数据池，每个模型都从干净的初始基础权重出发在该共享池上重新训练，共进行五代。然而在我们测试的范围内，这两个担忧均未成真；相反，实验呈现出一种不变性：使市场份额分配更加不均，几乎不会改变模型崩溃的速度。

    arXiv:2609.11146v1 Announce Type: cross  Abstract: AI-generated text is flowing back into the training corpora of the next generation of models. Recursive training on it drives model collapse, and recent work extends the setting to many models feeding one another -- but almost always with the market split evenly, while real generative AI is an oligopoly. Concentration raises two worries: fewer, more uniform sources may make collapse faster, and later models may be dragged toward the oligarch's output. We test both in controlled ecosystems: 13 open 1--4B models form natural ecosystems of 3 to 13 players, plus an injected probe that pushes the top share to 90%; each generation, every model's output is mixed into a shared pool by market share and every model is retrained on that pool from clean base weights, for five generations. Yet within the range we test, neither worry materializes; what emerges instead is an invariance. Making the split more unequal barely changes the speed of collap
    
[^115]: 同日同故事，隔日异信号：金融情感分析的双重效度

    Same Day, Same Story; One Day Ahead, a Different Signal: The Dual Validity of Financial Sentiment

    [https://arxiv.org/abs/2609.11144](https://arxiv.org/abs/2609.11144)

    本文基于2002-2025年证券集体诉讼语料库，将70,500条X消息与异常股票收益相关联，通过统一流程测试五种情感分析工具，发现金融情感工具的人工标注一致性（构念效度）与其市场预测能力（预测效度）之间的关系并非恒定，而是取决于抽样惯例和分数表示方式。

    

    金融自然语言处理（Financial NLP）领域有一个标准工作流程：先验证情感分析工具与人工标注的一致性，然后信任它来提取市场信号。这背后隐含的假设是，这两种评估衡量的是同一件事。我们在一个可以同时测量两者的场景中检验了这一假设：一个证券集体诉讼语料库（2002-2025年），将70,500条X平台消息与异常股票收益相关联，并包含一个由单一标注员人工标注的金标准样本。通过将五种工具（VADER、Loughran-McDonald、FinBERT、Twitter-RoBERTa和一个LLM标注器）运行在完全相同的流程中，我们发现构念效度与预测效度之间的关系取决于抽样惯例和分数表示方式。在传统的方法特定抽样下，人工一致性评分与同日的分级关联更为吻合，而与提前一天的关联吻合度较低。然而，在固定样本量的面板数据上，一致性评分在两个时间范围内都表现出相似的分级秩相关，而粗粒度排序在两种情况下均较弱。

    arXiv:2609.11144v1 Announce Type: cross  Abstract: Financial NLP has a standard workflow: validate a sentiment tool against human labels, then trust it to extract market signal. This assumes the two evaluations measure the same thing. We test that assumption in a setting where both can be measured at once: a corpus of securities class actions (2002-2025) linking 70,500 X messages to abnormal stock returns, with a single-annotator human labelled gold sample. Running five instruments (VADER, Loughran-McDonald, FinBERT, Twitter-RoBERTa, and an LLM annotator) through one identical pipeline, we find that the relationship between construct and predictive validity depends on the sampling convention and score representation. Under conventional method-specific sampling, human agreement aligns more closely with graded same-day associations than with one-day leads. On a fixed-n panel, however, agreement has similar graded rank correlations at both horizons, while the coarse ordering remains weak.
    
[^116]: KuaiRP系列角色扮演模型技术报告

    KuaiRP Series Role-playing Models Technical Report

    [https://arxiv.org/abs/2609.11127](https://arxiv.org/abs/2609.11127)

    KuaiRP系列角色扮演模型通过标准化角色模板、基于用户行为模拟的SFT数据流水线、规则复合奖励的强化学习以及多阶段训练流程，在注入深度领域知识的同时有效克服灾难性遗忘，实现了小参数规模下高质量、稳定且高效的角色扮演。

    

    本文介绍了KuaiRP系列角色扮演模型的完整技术方案。我们旨在为专用角色扮演模型实现四个核心目标：简化的提示词工程、高度稳定的输出质量、内置的领域世界知识，以及小参数规模下的高效部署。然而，有效注入深度领域知识往往会导致模型通用智能体能力的严重灾难性遗忘。为克服这一权衡问题，我们提出了一个多阶段训练流程。首先，我们设计了标准化的角色模板，并基于用户行为模拟和反向画像过滤构建了SFT数据流水线。其次，我们在强化学习（RL）阶段采用基于规则的复合奖励函数，以消除长度膨胀和重复生成等常见的退化现象。最后，为恢复在此过程中受损的通用能力……（摘要在此处截断）

    arXiv:2609.11127v1 Announce Type: cross  Abstract: This paper introduces the complete technical solution for the KuaiRP series of role-playing models. We aim to achieve four core objectives for a dedicated role-playing model: simplified prompt engineering, highly stable output quality, built-in domain world knowledge, and high-efficiency deployment with a small parameter size. However, effectively injecting deep domain knowledge often leads to a severe catastrophic forgetting of the model's general agent capabilities. To overcome this trade-off, we propose a multi-stage training pipeline. First, we design a standardized character template and construct an SFT data pipeline based on user behavior simulation and reverse profile filtering. Next, we utilize a rule-based composite reward function during the Reinforcement Learning (RL) phase to eliminate common degradation phenomena like length expansion and repetitive generation. Finally, to recover the general capabilities compromised duri
    
[^117]: 超越基准测试：利用视觉语言模型揭示真实世界条件下的系统性分类失败

    Beyond Benchmarks: Using VLMs to Reveal Systematic Classification Failures Under Real World Conditions

    [https://arxiv.org/abs/2609.11126](https://arxiv.org/abs/2609.11126)

    本研究提出一种基于视觉语言模型的错误切片检测方法，能够自动对分类模型的系统性错误进行分组和标记，从而加速国防应用中模型验证与确认的过程。

    

    分类模型的验证与确认（V&V）对于实现广泛的传感器处理应用至关重要。目前，验证与确认过程依赖于对错误样本进行耗时的人工检查，以发现有意义的模式。本工作探索使用视觉语言模型（VLM）来加速这一繁琐的过程。VLM经过训练，可以将图像嵌入到具有语义意义的向量表示中，从中可以提炼出人类可解释的系统性错误。在国防领域部署这种基于VLM的方法面临两大挑战：（1）国防领域在VLM的训练数据中代表性不足；（2）周围环境和上下文的多样性低于其他领域。本研究对基于VLM的方法在国防应用验证与确认中的适用性进行了初步评估。我们提出了一种基于VLM的错误切片检测（ESD）方法，该方法能够独立地对系统性错误进行分组和标记。

    arXiv:2609.11126v1 Announce Type: cross  Abstract: Verification and validation (V&V) of classification models is crucial to enable a wide range of sensor processing applications. Currently, the V&V process relies on time-consuming manual inspection of erroneous samples to find meaningful patterns. This work explores the use of Vision Language Models (VLMs) to speed up this laborious process. VLMs are trained to embed images into a semantically meaningful vector representation, from which human-interpretable systematic errors can be distilled. Deploying such VLM-based methods in a defence context introduces two major challenges: (1) the defence domain is underrepresented in the training data of VLMs, and (2) surroundings and context are less diverse than for other domains. This study provides an initial assessment of the suitability of VLM-based methods for V&V of defence applications. We propose a VLM-based error slice detection (ESD) method that independently groups and labels systema
    
[^118]: 基准雷达：面向AI基准与评估的动态数据库和搜索引擎

    Benchmark Radar: A Living Database and Search Engine for AI Benchmarks and Evaluation

    [https://arxiv.org/abs/2609.11115](https://arxiv.org/abs/2609.11115)

    本文提出Benchmark Radar，一个每日自动发现并整合AI基准论文、数据集和代码的动态数据库与搜索引擎，为LLM评估、智能体、编程、推理和安全等领域提供可检索的基准目录、来源引用和分数历史。

    

    基准研究人员和大语言模型（LLM）及其他AI系统的开发者需要找到相关的评估方法、定位其基准数据集和代码，并理解已报告分数背后的设置。我们提出了Benchmark Radar（基准雷达），这是一个用于检索和发现AI基准的动态数据库和搜索引擎，涵盖LLM评估、智能体和工具使用基准、编程、推理、安全性以及特定领域评估。该系统将每日发现的基准论文、代码仓库、数据集和发布版本与可搜索的基准目录、模型卡和技术报告中的提及以及分数历史相结合。它保留了源身份信息和引用，以便读者可以查看候选基准及其评估证据。每日发现基于37个来源：13个直接连接器和24个第一方研究与工程信息源。该目录包含来自4个基准目录的1,283条源记录

    arXiv:2609.11115v1 Announce Type: cross  Abstract: Benchmark researchers and developers of large language models (LLMs) and other AI systems need to find relevant evaluations, locate their benchmark datasets and code, and understand the settings behind reported scores. We present Benchmark Radar, a living database and search engine for retrieval and discovery of AI benchmarks, covering LLM evaluation, agentic and tool-use benchmarks, coding, reasoning, safety, and domain-specific evaluations. The system combines daily discovery of benchmark papers, repositories, datasets, and releases with a searchable benchmark catalog, mentions in model cards and technical reports, and score histories. It retains source identities and citations so readers can inspect candidate benchmarks and their evaluation evidence. Daily discovery draws on 37 sources: 13 direct connectors and 24 first-party research and engineering feeds. The catalog contains 1,283 source records drawn from 4 benchmark catalogs an
    
[^119]: AI 编码员如何讨论、产生分歧并达成共识：基于大语言模型定性编码的挑战与机遇

    How AI Coders Discuss, Disagree, and Reach Consensus: Challenges and Opportunities for LLM-Based Qualitative Coding

    [https://arxiv.org/abs/2609.11109](https://arxiv.org/abs/2609.11109)

    该研究构建了一个让多个大语言模型智能体独立编码、辩论并调和分歧的定性编码流程，发现编码准确性受编码手册长度、数据相似度和智能体分歧程度影响，其中激烈且未解决的辩论反而能提高准确性，但大语言模型虽能模拟人类讨论行为却缺乏对情境的适应性响应。

    

    AI 在多编码员定性编码中的应用已被广泛讨论，但鲜有实证证据能够明确其可靠发挥作用的适用情境。我们通过量化多智能体大语言模型编码在不同定性数据集上的有效性来填补这一空白，揭示了调节编码结果的关键情境与结构因素。我们开发了一个基于文献构建的基线流程，使 AI 智能体能够独立进行编码、辩论并调和分歧。结果显示，编码准确性取决于编码手册长度、定性数据相似度以及智能体间分歧程度等因素。值得注意的是，智能体之间激烈且未解决的辩论反而带来了更高的编码准确性。我们的分析表明，尽管大语言模型能够模拟许多人类的讨论行为，但它们缺乏对具体情境的适应性响应能力。基于这些发现，我们为构建自动化编码系统提供了设计建议。我们的开源 AI……

    arXiv:2609.11109v1 Announce Type: cross  Abstract: The utility of AI in multi-coder qualitative coding has been widely discussed, yet little empirical evidence exists to delineate the contexts in which it performs reliably. We address this gap by quantifying the effectiveness of multi-agent LLM coding across varied qualitative datasets, revealing key contextual and structural factors that mediate coding outcomes. We developed a literature-informed baseline pipeline that enables AI agents to independently code, debate, and reconcile disagreements. Results revealed that coding accuracy depends on factors such as codebook length, qualitative data similarity, and agent disagreement. Notably, intense and unresolved debates between agents led to higher accuracy. Our analysis showed that while LLMs emulate many human discussion behaviors, they lack adaptive responsiveness to context. From these findings, we offer design recommendations for building automated coding systems. Our open-source AI
    
[^120]: 少即是多：语音的哪些方面驱动话轮结束检测

    Less can be More: What Aspects of Speech Drive End-of-Turn Detection

    [https://arxiv.org/abs/2609.11066](https://arxiv.org/abs/2609.11066)

    通过对声学、韵律和语义信号的受控消融实验发现，话轮结束检测主要依靠语调和静音等声韵特征而非语义完整性，仅用声学与韵律组合的轻量级分类器即可在准确率与延迟之间取得最佳平衡，加入文本反而会增加过早检测。

    

    在对话式人工智能中，检测说话人何时结束发言对于自然的话轮转换至关重要。尽管近期的研究引入了语义信息，但不同模态的相对贡献仍不明确。我们使用一个轻量级三模态分类器，对声学、韵律和语义信号在流式话轮结束检测中的作用进行了控制消融实验。在相同的训练条件下，声学与韵律的组合在准确率和延迟之间达到了最佳平衡，话语F1分数达到0.93，误报率为7.8%，中位延迟为400毫秒。加入文本会增多过早检测的情况，而不会提升性能。特征空间分析证实，韵律特征具有最强的类别可分性，而文本表示存在大量重叠。这些发现表明，话轮转换主要通过语调和静音模式而非语义完整性来传达，从而能够实现更快、更自然的对话交互。

    arXiv:2609.11066v1 Announce Type: cross  Abstract: In conversational AI, detecting when a speaker has finished talking is crucial for natural turn taking. While recent work incorporates semantics, the relative contribution of different modalities remains unclear. We present a controlled ablation of acoustic, prosodic, and semantic signals for streaming end of turn detection using a lightweight trimodal classifier. Under identical training conditions, the acoustic prosodic combination achieves the best balance of accuracy and latency, achieving utterance F1 of 0.93 with 7.8% false alarms at 400ms median latency. Adding text increases premature detections without improving performance. Feature space analysis confirms that prosodic features have the strongest class separability, while text representations overlap substantially. These findings suggest that turn-taking is primarily conveyed through intonation and silence patterns rather than semantic completeness, enabling faster and more r
    
[^121]: MOSAIC：面向GraphRAG的查询感知探索策略自适应

    MOSAIC: Query-Aware Exploration Policy Adaptation for GraphRAG

    [https://arxiv.org/abs/2609.11065](https://arxiv.org/abs/2609.11065)

    MOSAIC是一个无需训练的框架，通过LLM分析器将每个查询的证据需求转化为定制的图探索策略（涵盖种子选择、遍历、停止和证据选择），使GraphRAG检索能够根据问题类型灵活适配，在GraphRAG-Bench上显著提升了答案正确率。

    

    图检索增强生成能够连接分布于语料库图中的证据，但大多数系统在所有查询中采用基本共享的探索流程。这造成了结构性不匹配：直接事实类问题可能只需要紧凑的局部邻域，比较类问题需要对多个目标进行均衡覆盖，而中介推理类问题则可能需要通过弱相关的连接节点进行更深层的路径探索。我们提出了Mosaic，这是一个无需训练的框架，将GraphRAG检索形式化为逐查询的控制问题。LLM分析器将查询特定的证据需求转化为关于种子选择、图遍历、停止条件和证据选择的受限策略，而语料库图、索引、评分函数、接地程序和答案生成器则保持共享。在GraphRAG-Bench上，Mosaic在Medical数据集上取得了76.97的查询加权答案正确率，在Novel数据集上取得了64.33，超越了此前报告的最强整体结果。

    arXiv:2609.11065v1 Announce Type: new  Abstract: Graph Retrieval-Augmented Generation (GraphRAG) can connect evidence distributed across a corpus graph, but most systems use largely shared exploration procedures across queries. This creates a structural mismatch: direct facts may need compact local neighborhoods, comparisons need balanced coverage of multiple targets, and mediated questions may require deeper paths through weakly related connectors. We present Mosaic, a training-free framework that formulates GraphRAG retrieval as a per-query control problem. An LLM analyzer converts query-specific evidence requirements into a bounded policy over seed selection, graph traversal, stopping, and evidence selection, while the corpus graph, indexes, scoring functions, grounding procedure, and answer generator remain shared.   On GraphRAG-Bench, Mosaic achieves query-weighted Answer Correctness of 76.97 on Medical and 64.33 on Novel, improving over the strongest previously reported overall r
    
[^122]: 在模型改变想法处分支：面向树结构强化学习的信念偏移分支方法

    Fork Where the Model Changes Its Mind: Belief-Shift Branching for Tree-Structured Reinforcement Learning

    [https://arxiv.org/abs/2609.11061](https://arxiv.org/abs/2609.11061)

    提出信念偏移分支方法，通过在候选边界处读取模型的答案信念、并在连续信念发生分歧的步骤前进行分叉，从而定位价值曲线的转折点，大幅提升无批评器步级强化学习的信用分配效率。

    

    树结构式 rollout 为无批评器的可验证奖励强化学习（RLVR）提供了步级信用分配能力：在推理链的中间点进行分叉，兄弟分支的结果差异即可估计该步骤的价值。然而每次分叉都会增加采样成本，因此现实的计算预算通常只允许每条链进行少数几次分叉。如果分叉放置在结果已经基本确定的位置，兄弟分支的结果大多一致，几乎无法提供信用信号；因此，在给定树规模的情况下，分叉位置的选取在很大程度上决定了步级强化学习能获得多少收益。现有主流方法大多按结构放置分叉（如固定长度、中点、分隔符），或依据下一个 token 的熵来放置。本文将分叉位置的选择形式化为定位推理链价值曲线的“转折点”，即期望结果发生转变之处。我们提出“信念偏移分支”（belief-shift branching）：在候选边界处读取模型对答案的信念，并在连续信念出现分歧的步骤之前进行分叉。

    arXiv:2609.11061v1 Announce Type: new  Abstract: Tree-structured rollouts give critic-free reinforcement learning with verifiable rewards (RLVR) step-level credit: fork a chain at an intermediate point, and sibling outcome differences estimate step value. Each fork adds sampling cost, so realistic budgets typically allow only a few forks per chain. A fork placed where the outcome is already largely settled yields siblings that mostly agree and provide almost no credit signal; hence, for a given tree size, where forks are placed largely determines how much step-level RL can gain. Most existing mainstream methods place forks by structure, such as fixed lengths, midpoints, and delimiters, or by next-token entropy. We formalize fork placement as locating the \emph{pivots} of the chain's value curve, where the expected outcome turns. We propose \emph{belief-shift branching}: read the model's answer belief at candidate boundaries and fork just before the step where consecutive beliefs diverg
    
[^123]: 智能体记忆的落地：面向企业智能体的环境探测式记忆管理

    Grounding Agent Memory: Environment-Probing Curation for Enterprise Agents

    [https://arxiv.org/abs/2609.11060](https://arxiv.org/abs/2609.11060)

    该论文提出“环境探测式记忆管理”方法，为智能体的异步记忆管理器赋予最小权限的只读环境工具以验证、限定和刷新候选记忆，无需重训模型即可将CLBench通过率从39%提升至73%。

    

    持久记忆正在进入面向生产的智能体平台，以帮助长时程智能体跨会话积累经验。然而，仅限于已完成轨迹的事后管理智能体可能会保留错误、过度概括局部证据或保留过时的知识。我们提出了环境探测式管理，这是一种兼容部署的扩展方案，为现有的异步记忆管理智能体提供最小权限、只读的世界工具，用于检查、限定范围和刷新候选记忆。该方法无需重新训练模型，且保持任务智能体、检索器、记忆表示和生产写入权限不变。在基于其SDK构建的类生产GitHub Copilot（GHCP）框架中，我们在CLBench数据库探索任务和90个改编的APEX管理咨询任务上比较了无状态执行、完全上下文学习、GHCP + Mem以及GHCP + Mem（带环境探测）四种方案。在CLBench上，环境探测将通过率从39%提升至73%……

    arXiv:2609.11060v1 Announce Type: cross  Abstract: Persistent memory is entering production-oriented agent platforms to help long-horizon agents accumulate experience across sessions. Yet a post-task curator agent restricted to completed trajectories can preserve errors, overgeneralize partial evidence, or retain stale knowledge. We introduce environment-probing curation, a deployment-compatible extension that gives an existing asynchronous curator agent least-privilege, read-only world tools to check, scope, and refresh candidate memories. It requires no model retraining and leaves the task agent, retriever, memory representation, and production write authority unchanged. In a production-like GitHub Copilot (GHCP) harness built on its SDK, we compare stateless execution, full in-context learning, GHCP + Mem, and GHCP + Mem (w/ Env Probing) on CLBench database exploration and 90 adapted APEX management-consulting tasks. On CLBench, probing raises pass rate from 39% to 73% and pass-disc
    
[^124]: T1：面向长程任务的终端智能体强化学习

    T1: Terminal Agent Reinforcement Learning for Long-Horizon Tasks

    [https://arxiv.org/abs/2609.11042](https://arxiv.org/abs/2609.11042)

    该论文提出T1——一个通过强化学习训练的122B混合专家模型，能在云端沙箱中操作真实shell执行多达300余次工具调用的长程终端任务，并给出了包含激进热启动、TITO构建与rollout路由重放等稳定优化技术以及分布外训练语料的完整训练方案。

    

    智能体的使用正朝着长程任务（如编程和科学发现）发展，其中终端任务尤为重要。我们提出了T1，一个总参数量为122B的混合专家（Mixture-of-Experts）模型，通过强化学习进行训练，在云端沙箱中操作真实的shell，每个任务可执行多达300余次工具调用，并通过执行每个任务自身的验证器来给予奖励。我们提供了一套完整的训练方案：首先，采用激进的热启动以稳定actor-critic训练，并使用密集过程奖励，根据轨迹中通过验证器的绝对数量进行评分。其次，通过TITO构建实现稳定优化，在精确采样的token标识符上进行训练并在轮次边界进行漂移修复，同时引入rollout路由重放——记录采样器在每个MoE层的逐token专家选择，并在训练期间进行重放。第三，采用完全分布外的训练语料库：与Terminal（原文截断）不相交的隔离种子和合成任务。

    arXiv:2609.11042v1 Announce Type: new  Abstract: Agent usage is shifting toward long-horizon tasks such as coding and scientific discovery, among which terminal tasks are especially important. We introduce T1, a Mixture-of-Experts model of 122B total trained with reinforcement learning, operating a real shell in a cloud sandbox for up to 300+ tool-call turns per task, rewarded by executing each task's own verifier. We provide a comprehensive recipe: First, an aggressively warm-started to stabilize actor-critic training, with a dense process reward scoring trajectories by the absolute number of passing verifiers. Second, stable optimization through TITO construction, training on the exact sampled token identifiers with drift repair at turn boundaries, and rollout routing replay, recording the sampler's per-token expert choices at every MoE layer and replaying them during training. Third, fully out-of-distribution training corpus: isolated seeds and synthesized tasks disjoint from Termin
    
[^125]: 迈向可解释的多模态融合：用于高光谱与激光雷达联合分类的热传导建模

    Toward Interpretable Multimodal Fusion: Heat Conduction Modeling for Hyperspectral and LiDAR Joint Classification

    [https://arxiv.org/abs/2609.11040](https://arxiv.org/abs/2609.11040)

    提出受热传导物理原理启发的M2Heat框架，通过视觉热传导模块和跨频率融合策略，以低于二次方的复杂度实现高光谱与激光雷达数据可解释、高效的多模态联合分类。

    

    高光谱（HS）与激光雷达（LiDAR）数据的融合通过联合利用光谱、空间和结构线索，在提升土地覆盖分类性能方面发挥着至关重要的作用。然而，现有的多模态融合方法在保持计算效率的同时，仍难以建模长程依赖关系和复杂的各向异性交互。本文提出了M2Heat，一个受物理学启发的框架，通过热传导的视角来研究多模态融合。其核心是一个物理驱动的视觉热传导模块（vHeat）和增强的频率值嵌入（FVE），它们模拟各向异性的信息流动，能够以低于二次方的计算复杂度和物理可解释性来捕获全局依赖关系。该机制与名为跨频率融合（CFF）模块的空频混合融合策略相结合，产生了高度判别性和鲁棒性的特征表示。

    arXiv:2609.11040v1 Announce Type: cross  Abstract: The fusion of hyperspectral (HS) and Light Detection and Ranging (LiDAR) data plays a crucial role in enhancing land-cover classification by jointly exploiting spectral, spatial, and structural cues. However, existing multimodal fusion methods still struggle to model long-range dependencies and complex anisotropic interactions while maintaining computational efficiency. This paper introduces M2Heat, a physics-inspired framework that investigates multimodal fusion through the lens of heat conduction. At its core, a physics-driven visual heat conduction module (vHeat) and enhanced Frequency Value Embeddings (FVEs) simulate anisotropic information flow, enabling the capture of global dependencies with sub-quadratic complexity and physical interpretability. This mechanism, combined with a hybrid spatial-frequency fusion strategy named Cross-Frequency Fusion (CFF) module, produces highly discriminative and robust feature representations. M2
    
[^126]: 代理事件登记册：迈向防止AI代理重复失败

    The Agent Incident Registry: Toward Preventing Repeated AI Agent Failures

    [https://arxiv.org/abs/2609.11030](https://arxiv.org/abs/2609.11030)

    本文提出了代理事件登记册（AIR），这是一个与来源关联、带缺失感知标准化标签的AI代理事件目录，可支持公开失败案例与代理安全评估的系统性比较，并发现实际危害主要集中于真实环境和安全失败类记录。

    

    AI代理越来越多地通过工具和委托权限来执行操作，但通用的事件存储库很少能捕捉到将公开失败案例与代理安全评估进行比较所需的机制。我们提出了代理事件登记册（AIR），这是一个与来源关联的目录，包含从\Yfirst{}至\Ylast{}期间披露的\N{}条代理相关事件记录。每条记录包括支持证据、稳定标识符，以及针对因果角色、披露类别、机制和结果的缺失感知标签。在代理实际执行操作的\Nprimary{}条生成式系统记录中，\Rprimary{}条涉及实际造成的伤害（\Pprimary%）。实际造成的结果集中在真实环境和安全失败记录中，而负责任的披露和研究演示绝大多数仅为演示性质；因此，总体比例反映的是数据集的构成情况，而非部署风险。在初始整理之后，第二位人工审核员对……（原文截断）

    arXiv:2609.11030v1 Announce Type: new  Abstract: AI agents increasingly act through tools and delegated authority, but general incident repositories rarely capture the mechanisms needed to compare public failures with agent-security evaluations. We present the Agent Incident Registry (AIR), a source-linked catalog containing \N{} records of agent-related events disclosed from \Yfirst{} through \Ylast{}. Each record includes supporting evidence, a stable identifier, and missingness-aware labels for causal role, disclosure class, mechanism, and outcome. Among the \Nprimary{} generative-system records in which the agent acted, \Rprimary{} involved realized harm (\Pprimary\%). Realized outcomes concentrate in in-the-wild and safety-failure records, while responsible disclosures and research demonstrations are overwhelmingly demonstrated; the aggregate share therefore characterizes collection composition rather than deployment risk. After initial curation, a second human reviewer checked al
    
[^127]: BenchShield：面向LLM智能体评估基础设施中奖励完整性的形式化模型支撑的检测工具

    BenchShield: Formal Model-Backed Instrumentation for Reward Integrity in LLM-Agent Evaluation Infrastructure

    [https://arxiv.org/abs/2609.11028](https://arxiv.org/abs/2609.11028)

    本文提出BenchShield，一个基于奖励相关事件有限生命周期形式化模型的检测工具层，通过静态与动态两种互补分析在基准测试基础设施内保障LLM智能体评估的奖励完整性，防范奖励劫持。

    

    语言模型智能体基准测试日益成为交互式评估基础设施。智能体观察状态、调用工具、修改工作区、提交产物，并从结果程序中接收奖励。这种交互性使评估容易受到奖励劫持的攻击：智能体通过利用与奖励相关的轨迹而非解决预期任务来提高其测得的分数。现有的防御措施主要依赖于针对特定任务的补丁、提示指令或事后检测器，无法提供可复用的证据来证明某次具体运行保持在预期的评估边界之内。本文提出了BenchShield，一个面向LLM智能体评估中奖励完整性的模型支撑的检测工具层。BenchShield将检测建立在评估的奖励相关事件的有限生命周期模型之上。在基准测试基础设施内，两种互补的分析在该模型上运行。一种静态的、阶段感知的污点分析……

    arXiv:2609.11028v1 Announce Type: cross  Abstract: LM-agent benchmarks increasingly function as interactive evaluation infrastructure. Agents observe state, call tools, modify workspaces,   submit artifacts, and receive rewards from outcome procedures. This interactivity makes evaluations vulnerable to reward hacking: an agent   improves its measured score by exploiting the reward-relevant trajectory instead of solving the intended task. Existing defenses rely largely   on task-specific patches, prompt instructions, or post-hoc detectors. They do not provide reusable evidence that a concrete run remained   within its intended evaluation boundary. This paper presents BenchShield, a model-backed instrumentation layer for reward integrity in   LLM-agent evaluation. BenchShield grounds detection in a finite lifecycle model of an evaluation's reward-relevant events. Within the   benchmark infrastructure, two complementary analyses operate over this model. A static, phase-aware taint analysi
    
[^128]: 新证据，同样的选择：测试视觉语言模型中的物理实验选择能力

    New Evidence, Same Choice: Testing Physical Experiment Selection in Vision Language Models

    [https://arxiv.org/abs/2609.11022](https://arxiv.org/abs/2609.11022)

    该论文提出了一个受控评估框架，测试视觉语言模型在物理推理中能否判断何时应直接作答、何时需要额外测量以及选择哪个实验，弥补了现有基准只评估最终答案而忽视决策能力的不足。

    

    模型首先观察到一个来自物理测量实验的图像，例如滑块滑行了多远，然后必须回答关于新试验的问题，例如滑块在受到固定推力后是否会通过某个目标点。初始实验可能提供足够的信息来直接回答，也可能需要另一项测量，例如物体的质量、摩擦系数、恢复系数或弹簧刚度。我们研究视觉语言模型能否判断何时应立即作答，以及当需要更多证据时应选择哪个实验。目前的物理推理基准通常仅评估最终答案，因此无法直接衡量这种决策能力。我们引入了一个受控评估设置：每个问题提供一张测量图像，以及由两种可能质量和另一相关属性的两种可能取值组合而成的四个可能物理世界。模型必须选择停止并直接作答，或选择成本最低的实验（摘要原文在此处截断）。

    arXiv:2609.11022v1 Announce Type: cross  Abstract: A model first sees an image from one physical measurement experiment, such as how far a block coasted, and must answer a question about a new trial, such as whether the block will pass a target after a fixed push. The initial experiment may provide enough information to answer, or the model may need another measurement, such as the object's mass, friction, restitution, or spring stiffness. We study whether vision language models can decide when to answer immediately and, when more evidence is needed, which experiment to perform. Current physical reasoning benchmarks usually evaluate only the final answer, so they do not directly measure this decision-making ability. We introduce a controlled evaluation where each problem provides one measurement image and four possible physical worlds created by combining two possible masses and two possible values of another relevant property. The model must either stop and answer or select the cheape
    
[^129]: 定义AI智能体：标准、指标与基准汇编

    Defining AI Agents: A Compendium of Criteria, Metrics, and Benchmarks

    [https://arxiv.org/abs/2609.11018](https://arxiv.org/abs/2609.11018)

    该论文提出了评估AI智能体的五个维度框架（环境交互、学习与适应、自主性、目标导向行为、时间连贯性），系统梳理了各维度相关的指标、基准与评估方法，并发布了公开的Agent Compendium数字资源，以解决智能体定义模糊导致的评估和比较难题。

    

    人工智能领域中“智能体”一词缺乏标准定义，这给AI智能体研究的评估、比较和可复现性带来了困难。我们通过一项围绕智能体性五个维度组织的综述来解决这一模糊性：环境交互、学习与适应、自主性、目标导向行为以及时间连贯性。对于每个维度，我们考察了以往工作中如何对相关底层能力进行概念化，并综合整理了用于评估该能力的指标、基准和评估框架。本综述对当前智能体评估领域的现状进行了结构化的梳理，既突出了已有的成熟方法，也指出了评估仍然有限或不一致的方面。此外，我们还介绍了Agent Compendium（智能体汇编），这是一个面向公众的数字资源，用于组织和扩展本综述中识别出的评估方法。综述与汇编共同为……

    arXiv:2609.11018v1 Announce Type: new  Abstract: The term agent in artificial intelligence lacks a standard definition, complicating the evaluation, comparison, and reproducibility of AI agent research. We address this ambiguity through a survey organized around five dimensions of agenticness: environmental interaction, learning and adaptation, autonomy, goal-directed behavior, and temporal coherence. For each dimension, we examine how the underlying capability has been conceptualized across prior work and synthesize the metrics, benchmarks, and evaluation frameworks used to assess it. This review provides a structured account of the current landscape of agent evaluation, highlighting both established approaches and areas where evaluation remains limited or inconsistent. We additionally introduce the Agent Compendium, a public-facing digital resource that organizes and extends the evaluation methods identified through this review. Together, the survey and compendium provide a common st
    
[^130]: 拓扑必然性：面向跨本体目标条件控制的机制不变策略子目标

    Topological Necessities: Mechanism-Invariant Strategic Subgoals for Cross-Embodiment Goal-Conditioned Control

    [https://arxiv.org/abs/2609.11014](https://arxiv.org/abs/2609.11014)

    该论文提出“拓扑必然性”这一新概念，通过在成功离线轨迹构建的载体上运用0维和1维同调，提取带有证书的不可跳过阶段与路径选择点，构建机制不变、可跨本体通用的递归拓扑门层级，作为长时程目标条件控制中的策略子目标。

    

    长时程目标条件强化学习将控制权委托给一个提出子目标的高层模块，但现有的子目标是价值函数或潜在动作的隐式副产品，与产生它们的执行器紧密绑定。我们研究一种不同的对象：一种由路径条件决定的、所有成功执行器都必然经历的不可避免阶段序列，它可以从离线轨迹中恢复出来，却不属于任何单一执行器或轨迹。其定义性质是拓扑性的：一个不可跳过的阶段是每条可行路径都必须穿越的分离集，而自由空间中的环路则迫使做出路径选择。我们在由成功轨迹构建的、以传输权重加权的载体上，通过维度0和1的同调来读取这两个性质，从而得到一个带有壳层证书的可枚举门集；这些获得认证的门即我们所说的“拓扑必然性”。经认证的门以递归拓扑门层级的形式进入决策回路。在固定的、同构……（摘要原文在此处截断）

    arXiv:2609.11014v1 Announce Type: new  Abstract: Long-horizon goal-conditioned reinforcement learning delegates control to a high-level module that proposes subgoals, but existing subgoals are implicit byproducts of value functions or latent actions, tied to the executor that produced them. We study a different object: a route-conditioned order of unavoidable stages that every successful executor must traverse, recoverable from offline trajectories and belonging to none of them. Its defining properties are topological: an unskippable stage is a separating set that every admissible path must cross, and a loop in free space forces a route choice. We read the two by homology in dimensions 0 and 1 over a transport-weighted carrier built from successful trajectories, yielding an enumerable gate set with shell-level certificates; the certified gates are what we call topological necessities. Certified gates enter the decision loop as a recursive topological gate hierarchy. Under a fixed, isom
    
[^131]: DeFiFusion：结合交易事件与智能合约检测价格操纵攻击

    DeFiFusion: Combining Transaction Events with Smart Contracts to Detect Price Manipulation Attacks

    [https://arxiv.org/abs/2609.11008](https://arxiv.org/abs/2609.11008)

    DeFiFusion是一个双模态检测框架，通过联合建模交易事件与智能合约执行语义来检测DeFi中的价格操纵攻击，克服了单纯基于交易或静态合约分析方法各自的根本局限。

    

    去中心化金融（DeFi）已成为一种快速增长的基于区块链的金融服务，其中市场交易动态与底层智能合约逻辑错综复杂地交织在一起。这种自主的相互作用虽然消除了中心化中介，但显著扩大了DeFi协议面对价格操纵攻击（PMAs）的脆弱面，此类攻击已经造成了灾难性的经济损失。尽管问题严重，现有的检测范式存在根本性局限：以交易为中心的方法缺乏对合约执行语义的感知，使其在合法市场波动下容易产生误报；而静态合约分析则忽略真实交易行为，经常报告在实际中无法被利用的漏洞。我们提出了DeFiFusion，这是一种双模态PMA检测框架，通过联合建模交易事件和智能合（约）……

    arXiv:2609.11008v1 Announce Type: cross  Abstract: Decentralized Finance (DeFi) has emerged as a rapidly growing blockchain-based financial service, where market transaction dynamics and underlying smart contract logic are intricately intertwined. This autonomous interplay, while eliminating centralized intermediaries, significantly expands the vulnerability surface of DeFi protocols to Price Manipulation Attacks (PMAs), which have already inflicted catastrophic financial losses. Despite their gravity, existing detection paradigms suffer from fundamental limitations. Transaction-centric methods lack awareness of contract execution semantics, making them prone to false positives under legitimate market volatility, while static contract analyses ignore real transaction behaviors and frequently report vulnerabilities that are infeasible to exploit in practice. We present DeFiFusion, a dual-modal PMA detection framework that closes this gap by jointly modeling transaction events and smart 
    
[^132]: 分布偏移下无标签-无标签学习的重要性加权方法

    Importance Weighting for Unlabeled-unlabeled Learning under Distribution Shift

    [https://arxiv.org/abs/2609.10994](https://arxiv.org/abs/2609.10994)

    本文提出了一种基于重要性加权的UU学习分布偏移自适应方法，通过有原则的重要性权重估计，利用测试分布中的少量UU数据来最小化测试风险，从而解决训练与测试分布不一致的问题。

    

    无标签-无标签（UU）学习使我们能够从两组具有不同类先验概率的无标签数据中学习二分类器。它是一个通用框架，因为它涵盖了多种监督学习场景，如正例-无标签（PU）学习、噪声标签学习和基于相似度的学习。现有的UU学习假设测试分布和训练分布具有相同的类条件密度。然而，由于分布偏移的存在，这一假设在实践中很少成立。本文提出了一种针对UU学习的分布偏移自适应方法，该方法利用训练分布中的UU数据以及测试分布中的少量UU数据。所提出的方法基于重要性加权，通过使用带有估计重要性权重的训练数据来最小化测试风险。尽管现有的重要性加权方法无法处理UU数据，我们证明了这可以通过有原则的方式来实现。

    arXiv:2609.10994v1 Announce Type: new  Abstract: Unlabeled-unlabeled (UU) learning allows us to learn a binary classifier from two sets of unlabeled data with different class-priors. It is a general framework because it includes a wide variety of supervised learning such as positive-unlabeled (PU) learning, noisy label learning, and similarity-based learning. Existing UU learning assumes that the test and training distributions have the same class-conditional densities. However, this assumption rarely holds in practice due to distribution shifts. This paper proposes a distribution shift adaptation method for UU learning that uses UU data in the training distribution and a few UU data in the test distribution. The proposed method is based on the importance weighting, which minimizes the test risk by using training data with estimated importance weights. Although existing importance weighting methods cannot handle UU data, we show that it can be done in a principled manner. Thanks to the
    
[^133]: 揭示大语言模型交互中隐私-效用权衡的奥秘

    Demystifying the Privacy-Utility Trade-off in LLM Interactions

    [https://arxiv.org/abs/2609.10992](https://arxiv.org/abs/2609.10992)

    该论文系统性解构了大语言模型交互中的隐私-效用权衡，揭示了决定“何时脱敏”的上下文依赖效用、决定“如何脱敏”的策略性适应以及组合交互作用这三种潜在机制。

    

    大语言模型融入日常任务依赖于富含上下文的指令，这不可避免地会暴露用户的敏感信息。当前的隐私保护方法通常采用与上下文无关的静态规则，导致严重的效用下降。然而，数据脱敏如何影响下游性能的具体机制在很大程度上仍未被充分探索。为解决这一问题，我们进行了系统性分析来解构隐私-效用权衡，揭示了三种潜在机制：(1) 上下文依赖的效用，通过揭示数据价值会根据用户意图从关键约束转变为可舍弃的噪音，首先确定了何时进行脱敏；(2) 策略性适应，随后确定如何进行脱敏，指出删除与替换之间的选择取决于任务对事实完整性与结构连贯性的依赖程度；(3) 组合交互作用，它……（原文摘要在此处截断）

    arXiv:2609.10992v1 Announce Type: new  Abstract: The integration of Large Language Models into daily tasks relies on context-rich instructions, inevitably exposing sensitive user information. Current privacy-preserving methods typically employ context-agnostic static rules, causing severe utility degradation. However, the specific mechanisms governing how sanitization impacts downstream performance remain largely underexplored. To address this, we conduct a systematic analysis to deconstruct the privacy-utility trade-off, uncovering three underlying mechanisms: (1) Context-Dependent Utility, which first establishes when to sanitize by revealing that data value shifts from critical constraints to dispensable noise based on user intent; (2) Strategic Adaptation, which subsequently determines how to sanitize by dictating that the choice between removal and replacement depends on the task's reliance on factual integrity versus structural coherence; and (3) Combinatorial Interplay, which fi
    
[^134]: 语用信息的数学理论

    A Mathematical Theory of Pragmatic Information

    [https://arxiv.org/abs/2609.10986](https://arxiv.org/abs/2609.10986)

    该论文提出了一个统一通信、控制与决策的语用信息理论，通过同终点映射建立语法—语义—语用三层信息层次，推广了香农编码定理，并提出语用价值与语用成本的拉格朗日对偶框架以实现跨层优化。

    

    我们提出了一种统一的语用信息理论，将通信、控制与决策制定融合在一起。其核心是同终点映射（isoteleia mapping），形式化了等终性（equifinality）：通向同一最优动作的不同语义路径在语用上是等价的。这引出了语法、语义和语用信息的三层层次结构，每一层抽象都舍弃与任务无关的区分。我们发展了语用熵、上/下互信息、信道容量和率失真理论，并证明了三个推广香农经典结果的编码定理。我们引入了信息的语用价值和语用成本，分别作为率失真和信道容量的决策论对偶概念，并构建了用于跨层优化的拉格朗日对偶框架。语用效率界 $\mathcal{E}_p(\lambda)=\sup_R[\Phi_p(R)-\lambda\,\mathrm{CoI}_p(R)]$ 量化了任何资源受限的智能系统所能获得的最大净效用。

    arXiv:2609.10986v1 Announce Type: cross  Abstract: We propose a pragmatic information theory unifying communication, control, and decision-making. Its core is the isoteleia mapping, formalizing equifinality: distinct semantic paths leading to the same optimal action are pragmatically equivalent. This induces a three-tier hierarchy of syntactic, semantic, and pragmatic information, each abstraction discarding task-irrelevant distinctions. We develop pragmatic entropy, up/down mutual information, channel capacity, and rate-distortion, and prove three coding theorems generalizing Shannon's classical results. We introduce pragmatic value (VoI) and cost (CoI) of information as decision-theoretic duals to rate-distortion and capacity, respectively, and formulate a Lagrangian dual framework for cross-layer optimization. The pragmatic efficiency bound $\mathcal{E}_p(\lambda)=\sup_R[\Phi_p(R)-\lambda\,\mathrm{CoI}_p(R)]$ quantifies the maximum net utility any resource-constrained intelligent sy
    
[^135]: EGGROLL展开：理解并改进大规模低秩进化策略

    EGGROLL, Unrolled: Understanding and Improving Low-Rank Evolution Strategies at Scale

    [https://arxiv.org/abs/2609.10980](https://arxiv.org/abs/2609.10980)

    本文首次从理论上刻画了面向大语言模型的低秩进化策略EGGROLL的更新场，揭示其可能引入非保守分量并逆转最优点的局部稳定性，同时证明了该方法的二次目标精确性并给出非渐近误差界，为理解与改进该方法奠定理论基础。

    

    EGGROLL通过用低秩高斯乘积（通常为秩一）替代稠密高斯权重扰动，使进化策略（ES）在大语言模型（LLM）上变得实用。这一选择在计算上颇具吸引力，但在几何上却相当严苛：尽管协方差为单位阵，每个秩一扰动都位于环境矩阵空间的一个零体积子集中。我们刻画了在有限秩和非零扰动半径下EGGROLL的平均更新场，并分析了其有限种群估计器的误差。该种群场是通过将一个显式预解式作用于由扰动平滑后的目标函数梯度而得到的。我们证明该预解式可能引入非保守分量，并可能逆转最优点的局部稳定性。尽管如此，EGGROLL在任意秩和任意半径下对所有二次目标函数都是精确的。对于光滑目标函数，其首个局部有限秩修正项为O(σ²/r)，且非渐近界控制着

    arXiv:2609.10980v1 Announce Type: new  Abstract: EGGROLL makes evolution strategies (ES) practical for LLMs by replacing dense Gaussian weight perturbations with low-rank Gaussian products, often of rank one. This choice is computationally attractive but geometrically severe: each rank-one perturbation lies in a zero-volume subset of the ambient matrix space, despite having identity covariance. We characterize the mean EGGROLL update field at finite rank and nonzero perturbation radii, then analyze the error of its finite-population estimator. The population field is obtained by applying an explicit resolvent to the gradient of the objective smoothed by the perturbations. We show that the resolvent can introduce a nonconservative component and can reverse the local stability of an optimum. EGGROLL is nevertheless exact on every quadratic objective at every rank and radius. For smooth objectives, its first local finite-rank correction is $O(\sigma^2/r)$, and nonasymptotic bounds control
    
[^136]: 解耦就绪与释放：面向智能体LLM工作流的尾延迟感知调度

    Decoupling Readiness from Release for Tail-Aware Scheduling of Agentic LLM Workflows

    [https://arxiv.org/abs/2609.10964](https://arxiv.org/abs/2609.10964)

    本文提出一种尾风险感知的智能体LLM工作流轮次释放调度方法，通过均值-CVaR目标联合决策释放时机与未完成工作量预算，将轮次就绪与释放解耦以降低尾延迟。

    

    智能体LLM工作流由模型轮次与工具交互交织而成的序列组成，因此其端到端完成时间不仅取决于推理速度，还取决于就绪轮次何时被释放。大多数运行时会在轮次就绪后立即将其释放。在资源竞争条件下，这种急切的释放策略会不断累积已释放但未完成的工作；而这些轮次一旦提交，就无法再被工作流级别的策略重新排序，从而增加尾延迟。我们提出了一种尾风险感知的轮次释放调度方法，该方法联合决定下一步应释放哪个就绪轮次，以及应维持多少已释放但未完成的工作。该方法使用均值-条件风险价值目标来捕捉未完成工作流不断演变的尾风险，在对就绪轮次进行优先级排序时纳入对轮次工作量的在线估计，并根据观察到的队列压力自适应调整已释放工作预算。我们使用真实的智能体执行对该方法进行了评估。

    arXiv:2609.10964v1 Announce Type: cross  Abstract: Agentic LLM workflows consist of sequences of model turns interleaved with tool interactions, so their end-to-end completion time depends not only on inference speed but also on when ready turns are released. Most runtimes release each turn immediately upon readiness. Under contention, this eager release policy can accumulate released but unfinished work; once submitted, those turns can no longer be reordered by the workflow-level policy, increasing tail latency. We present a tail-risk-aware turn release scheduling method that jointly decides which ready turn to release next and how much released but unfinished work to maintain. The method uses a mean--Conditional Value-at-Risk (CVaR) objective to capture the evolving tail risk of unfinished workflows, incorporates online estimates of turn work when prioritizing ready turns, and adapts the released work budget to observed queue pressure. We evaluate the method using real agent executio
    
[^137]: MCP注册表的随机抽样包含什么，以及工具使用基准测试实际包含了什么

    What a Random Draw from the MCP Registry Contains, and What Tool-Use Benchmarks Contain Instead

    [https://arxiv.org/abs/2609.10962](https://arxiv.org/abs/2609.10962)

    该研究通过对MCP注册表进行可复现的随机概率抽样，首次揭示了真实服务器生态中近半数服务器根本无法启动、安全注释遗漏率高达58.8%，从而证明现行工具使用基准测试所依赖的人工精选样本会系统性高估生态系统的实际可用性与安全性。

    

    对模型上下文协议（MCP）服务器生态系统的研究在抽取样本时，会以各种方式悄悄筛选出能够正常运行的服务器：参考集合、流行度榜单、人工精选框架，或是将服务器修复至能启动为止的流水线。我们报告了未经修复的概率样本实际包含的内容。从一次涵盖24,135个服务器的注册表普查中，我们使用公开的随机种子抽取了400个npm/stdio服务器，并对每个服务器进行了线上探测。只有48.8%的服务器完成了初始化握手，而以相同测量方法对人工精选框架测得的比例为66.7%；主要失败原因并非缺少凭证（13.3%），而是服务器根本无法启动（37.5%）。在195个能够运行的服务器中，硬性合规是完全的：在2,766个声明的工具中，致命的JSON Schema违规为零。可选的安全注释才是真正的差异所在：随机抽样的工具级遗漏率为58.8%，而精选框架为41.5%，说明人工筛选美化了这一数字……

    arXiv:2609.10962v1 Announce Type: new  Abstract: Studies of the Model Context Protocol (MCP) server ecosystem draw their samples in ways that quietly select for servers that work: reference sets, popularity lists, hand-curated frames, or pipelines that repair a server until it starts. We report what an unrepaired probability sample actually contains. From a 24,135-server registry census we draw 400 npm/stdio servers with a published seed and probe each one over the wire. Only 48.8% complete an initialize handshake, against 66.7% for a hand-curated frame measured with the same instrument, and the dominant failure is not missing credentials (13.3%) but servers that never start at all (37.5%). Among the 195 that do run, hard conformance is total: zero fatal JSON Schema violations across 2,766 advertised tools. Optional safety annotations are the real variance, and the tool-level omission rate on a random draw is 58.8% against 41.5% on the curated frame, so curation flatters this figure to
    
[^138]: 基于语义感知完整性重建的模态缺失鲁棒多模态情感分析

    Robust Multimodal Sentiment Analysis with Incomplete Modalities via Semantic-aware Completeness based Reconstruction

    [https://arxiv.org/abs/2609.10950](https://arxiv.org/abs/2609.10950)

    该论文提出了一种语义感知的完整性估计方法与稳定的多任务训练策略，通过重建模态缺失的语义信息，显著提升了模态不完整场景下多模态情感分析的鲁棒性和预测精度。

    

    近期的多模态情感分析研究越来越多地采用以文本为中心的融合方法，以利用文本模态中蕴含的丰富情感信息。然而，在真实场景中，由于数据部分缺失或存在噪声，这些方法在推理阶段往往面临性能下降的问题，尤其是当情感相关的线索缺失时。为了解决这一问题，我们提出了一种新的完整性估计方法，该方法量化不完整数据中所保留的情感相关信息的程度，以指导缺失语义的重建。此外，我们提出了一种训练策略，在联合优化情感预测和完整性估计的同时，稳定多任务学习。在三个基准数据集上进行的广泛实验和深入分析表明，所提出的方法能够实现更准确的语义重建，从而带来更精确的情感预测。

    arXiv:2609.10950v1 Announce Type: new  Abstract: Recent multimodal sentiment analysis studies increasingly adopt text-centric fusion approaches to exploit the rich sentiment information inherent in the textual modality. However, these approaches often suffer from performance degradation during inference due to partially missing or noisy data in real-world scenarios, especially when sentiment-related cues are missing. To address this issue, we introduce a new completeness estimation approach that quantifies the degree of sentiment-relevant information preserved in incomplete data to guide the reconstruction of missing semantics. Furthermore, we propose a training strategy that stabilizes multi-task learning while jointly optimizing sentiment prediction and completeness estimation. Extensive experiments and in-depth analyses on three benchmark datasets demonstrate that the proposed approach enables more accurate semantic reconstruction, leading to more precise sentiment prediction.
    
[^139]: 评估面向支架式教学的多智能体大语言模型临床问诊训练系统

    Evaluating Scaffolding-Oriented Multi-Agent Large Language Model System for Clinical Interview Training

    [https://arxiv.org/abs/2609.10939](https://arxiv.org/abs/2609.10939)

    该研究构建了由病人、导师和评估三个智能体组成的多智能体大语言模型标准化病人训练平台，并通过100名医学生的随机对照试验验证了其在可扩展的临床问诊训练中的支架式教学价值。

    

    临床教育必须培养医学生在不确定条件下进行安全且连贯的患者问诊的能力。传统的标准化病人（SP）训练资源消耗大且难以规模化。我们开发了一个面向支架式教学的多智能体大语言模型（LLM）人工智能标准化病人（AI-SP）训练平台。该系统包括一个用于模拟对话的病人智能体、一个在不泄露诊断信息的前提下提供苏格拉底式提示的导师智能体，以及一个在不透露总结性评分的情况下监测临床进展的逐轮评估智能体。在一项随机对照研究（N = 100名医学生）中，参与者被分配到多智能体（MA）支架式教学条件或对照组条件。所有学生在其被分配的条件下完成了两个学习环节，随后在仅有病人智能体的环境中接受考核。表现通过标准化客观……（原文在此处截断）

    arXiv:2609.10939v1 Announce Type: cross  Abstract: Clinical education must prepare medical students to conduct safe and coherent patient interviews under conditions of uncertainty. Traditional standardized patient (SP) training is resource-intensive and difficult to scale. We developed a scaffolding-oriented multi-agent Large Language Model (LLM) AI Standardized Patient (AI-SP) training platform1. The system includes a patient agent for simulated dialog, a tutor agent providing Socratic prompts without disclosing diagnostic information, and a turn-level evaluator agent that monitors clinical progress without revealing summative scores. In a randomized controlled study (N = 100 medical students), participants were assigned to either a multi-agent (MA) scaffolding condition or a control condition. All students completed two learning sessions under their assigned condition followed by an examination conducted in a patient only environment. Performance was assessed using a standardized Obj
    
[^140]: 基于置信度的有偏正未标注数据的AUC最大化

    AUC Maximization from Biased Positive-unlabeled Data with Confidence

    [https://arxiv.org/abs/2609.10928](https://arxiv.org/abs/2609.10928)

    提出了一种利用少量已标注正类数据所附带的置信度信息（即实例为正类的概率），从有偏正未标注数据中实现AUC最大化的新方法，突破了现有方法要求正类数据必须无偏的理想化假设。

    

    最大化接收者操作特征曲线下面积（AUC）是不平衡二分类的标准方法。虽然最大化AUC需要正类和负类数据，但在一些实际应用中，由于隐私问题或需要专业知识进行标注，负类数据往往难以收集。因此，从正类和未标注（PU）数据进行AUC最大化引起了广泛关注。现有方法假设已标注的正类数据是来自真实正类分布的无偏样本。然而，这一理想假设在实践中经常被违反。在本文中，我们提出了一种从有偏PU数据中最大化AUC的方法。为了解决偏差问题，我们的关键思想是利用与少量已标注正类数据相关联的“置信度”，即实例为正类的概率。我们推导了使用带置信度的有偏PU数据的AUC风险估计器。

    arXiv:2609.10928v1 Announce Type: new  Abstract: Maximizing the area under the receiver operating characteristic curve (AUC) is a standard approach to imbalanced binary classification. Although positive and negative data are required for maximizing the AUC, negative data are often difficult to collect in some real-world applications due to privacy concerns or the need for specialized expertise to annotate them. Thus, AUC maximization from positive and unlabeled (PU) data has been attracting attention. Existing methods assume that labeled positive data are unbiased samples from the true positive distribution. However, this ideal assumption is often violated in practice. In this paper, we propose a method to maximize the AUC from biased PU data. To address the bias, our key idea is to exploit {\it confidence}, i.e., the probability that an instance is positive, associated with the small number of labeled positive data. We derive an estimator of the AUC risk using biased PU data with conf
    
[^141]: ReactHuman：一个面向具身多模态大语言模型类人反应式决策的物理基准

    ReactHuman: A Physics-Grounded Benchmark for Human-Like Reactive Decision-Making in Embodied Multimodal LLMs

    [https://arxiv.org/abs/2609.10895](https://arxiv.org/abs/2609.10895)

    ReactHuman是首个基于物理的类人反应式决策基准，通过240 Hz刚体仿真提供精确真值，评估多模态大语言模型在面对突发家庭危险时能否将物理理解转化为即时、安全攸关的行动。

    

    对突发物理危险做出反应（接住打滑的盘子、躲避掉落的刀）既是对具身智能的有意义测试，也是将多模态大语言模型（MLLM）部署为家庭机器人决策核心的硬性要求。然而，现有的评估要么通过视频问答被动地探测直觉物理，要么针对深思熟虑的长时程任务（如导航和物品重排）；没有任何评估衡量模型能否将物理理解转化为即时、安全攸关的行动。我们提出了ReactHuman，这是首个基于物理的类人反应式决策基准，其中被评估的MLLM充当模拟人形机器人的“大脑”，面对突发的家庭危险；该基准涵盖17个事件族和超过1000个可逐比特复现的场景，其精确、无需标注的真值来自240 Hz刚体仿真，还包括外观具有对抗性误导的物体……

    arXiv:2609.10895v1 Announce Type: cross  Abstract: Reacting to sudden physical hazards (catching a slipping plate, dodging a falling knife) is both a meaningful test of embodied intelligence and a hard requirement for deploying multimodal large language models (MLLMs) as the decision coreof household robots. Existing evaluations, however, probe intuitive physics passively through question answering over videos, or target deliberate, long-horizon tasks such as navigation and rearrangement; none measure whether a model can turn physical understanding into immediate, safety-critical action. We introduce ReactHuman, the first physics-grounded benchmark for human-like reactive decision-making, in which the evaluated MLLM acts as the brain of a simulated humanoid facing sudden household hazards; it spans 17 event families and over 1,000 bit-for-bit reproducible scenes with exact, annotation-free ground truth derived from 240 Hz rigid-body simulation, including adversarial objects whose appea
    
[^142]: 语言结构增强能否提升连贯性评估？在当前架构下并不能

    Does Linguistic Structure Enrichment Enhance Coherence Assessment? Not With Current Architectures

    [https://arxiv.org/abs/2609.10893](https://arxiv.org/abs/2609.10893)

    研究发现，由于附加信息与当前语言模型架构存在结构和句法上的不兼容性，用句法和修辞信息增强文本并不能提升连贯性评估效果，而文本连贯性可作为检测虚假信息的代理指标。

    

    大语言模型的最新进展已经改变了人机交互方式。尽管这些模型生成的文本十分流畅，但往往在语法上正确而在语义上不连贯，包含矛盾或逻辑流程的中断。本工作研究了用句法和修辞信息丰富文本是否能够改善不连贯性预测。我们的实验和分析表明，纯文本反而获得了更高的准确率，因为添加的信息在结构和句法上与语言模型的架构不兼容。此外，为了证明连贯性评估的实际重要性，我们在一个巴西虚假信息数据集上进行了零样本实验，结果表明文本连贯性可以作为检测误导性内容的代理指标。

    arXiv:2609.10893v1 Announce Type: new  Abstract: Recent advances in large language models have transformed human-computer interaction. Despite their fluency, these models often produce texts that are grammatically correct but semantically incoherent, containing contradictions or disruptions in logical flow. This work investigates whether enriching text with syntactic and rhetorical information can improve incoherence prediction. Our experiments and analysis show that plain texts achieved higher accuracy because the added information was structurally and syntactically incompatible with the language model's architecture. Additionally, to demonstrate the practical importance of coherence assessment, we performed zero-shot experiments on a Brazilian disinformation dataset, suggesting that textual coherence can serve as a proxy for detecting misleading content. Code and models are available at https://github.com/ittozzamV/cohereclassifier.
    
[^143]: DriftNet：一种用于检测和定位LLM智能体中提示注入攻击的双头轨迹Transformer

    DriftNet: A Dual-Head Trajectory Transformer for Detecting and Localizing Prompt Injection in LLM Agents

    [https://arxiv.org/abs/2609.10892](https://arxiv.org/abs/2609.10892)

    DriftNet是一个双头轨迹Transformer，只需一次前向传播即可同时判断LLM智能体的工具调用轨迹是否被提示注入攻陷，并对每个步骤进行精细标注（良性、注入点、被劫持或注入失败），首次实现了检测结果与攻击定位的联合输出。

    

    当间接提示注入成功攻击LLM智能体时，这种危害会体现在智能体自身的行为中：一段良性的工具调用前缀、一个被污染的观测结果，以及一段服务于攻击者的动作后缀。运营者需要了解三个关键事实：攻击从何处进入、它污染了哪些步骤、以及表面上的毒害是否被成功抵御。现有系统要么返回整个轨迹的判定结果，要么只返回单个不安全索引。我们提出了DriftNet，一个双头轨迹Transformer，它读取已记录的工具调用轨迹，并在一次前向传播中回答所有这三个问题：一个头将轨迹分类为是否被攻陷，另一个头为每个步骤分配四种标签之一（良性、注入点、被劫持、注入失败）。据我们所知，这是首个能够产生这种联合输出的监督式检测器。该方法使用一个冻结的句子编码器和四个不含身份信息的世界特征来嵌入每个步骤；训练好的主干网络参数量少于两百万。

    arXiv:2609.10892v1 Announce Type: cross  Abstract: When an indirect prompt injection succeeds against an LLM agent, the compromise is visible in the agent's own behavior: a benign prefix of tool calls, a poisoned observation, and a suffix of actions that serve the attacker. An operator needs three facts: where the attack entered, which steps it corrupted, and whether apparent poison was resisted. Existing systems return either a whole-trace verdict or a single unsafe index. We present DriftNet, a dual-head trajectory Transformer that reads a logged tool-call trajectory and answers all three questions in one forward pass: one head classifies the trajectory as compromised or not, and a second assigns every step one of four labels (benign, injection point, hijacked, failed injection). To our knowledge it is the first supervised detector to produce this joint output. A frozen sentence encoder and four identity-free world features embed each step; the trained trunk, under two million parame
    
[^144]: 故事烙印：AI助手会吸收其相似人类角色的特质

    Story Imprinting: AI Assistants Absorb Traits from Human Characters They Resemble

    [https://arxiv.org/abs/2609.10883](https://arxiv.org/abs/2609.10883)

    研究发现，在合成故事上微调会使AI助手“烙印”上与其相似的人类角色的条件性行为和隐含偏好，即使这些内容在训练数据中占比不到2%或从未被明确表达。

    

    语言模型被训练以扮演一个有帮助的AI助手角色（例如Claude）。我们探讨了在合成故事上进行微调会如何影响这一角色。这是否会改变助手在与用户进行多轮对话——这种与故事形式截然不同的场景——中的行为？助手是否会采纳故事中人类角色的行为和偏好？我们将这种采纳称为“故事烙印”。我们对GPT-4.1和Kimi-K2.6进行了微调，微调数据是一些故事，其中通常乐于助人的人类角色在受到侮辱后会给出微妙有害的建议。助手采纳了同样的条件性行为，同时在其他方面仍然保持乐于助人。即使只有不到2%的故事描述了这种行为，这种现象也会发生。在另一项实验中，助手还采纳了仅在叙述中隐含表达的偏好。一个人类角色的肢体语言暗示其不喜欢处理电子表格，但他从未说过这一点，并且仍然在电子表格方面给出良好的建议。

    arXiv:2609.10883v1 Announce Type: cross  Abstract: Language models are trained to implement a helpful AI Assistant character (e.g., Claude). We explore how finetuning on synthetic stories affects this character. Does it change the Assistant's behavior in multi-turn conversations with users, a format quite different from the stories? And does the Assistant adopt the behaviors and preferences of human characters? We refer to this adoption as story imprinting. We finetune GPT-4.1 and Kimi-K2.6 on stories in which generally helpful human characters give subtly harmful advice after being insulted. The Assistant adopts the same conditional behavior while otherwise remaining helpful. This occurs even when fewer than 2% of stories depict the behavior. In a separate experiment, the Assistant adopts preferences that are only implicit in the narration. A human character's body language suggests they dislike working on spreadsheets, yet they never say so and continue giving good advice on spreadsh
    
[^145]: 当验证停止学习：对持续具身智能体的更新准入进行审计

    When Validation Stops Learning: Auditing Update Admission for Continual Embodied Agents

    [https://arxiv.org/abs/2609.10873](https://arxiv.org/abs/2609.10873)

    论文提出持续具身智能体的更新准入审计必须同时考量误差控制与保留的学习机会，并指出基于区间的置信度门控在充足预算下无法认证旧任务行为不变，而配对二项检验能在同等预算下接纳31.6%的更新流。

    

    独立评估可以拒绝有害的策略更新，但也可能阻碍有用的持续学习。我们认为，更新准入必须在规定的交互预算下，从误差控制和保留的学习机会两个方面进行评估。我们发现了一个具体的失败案例：基于区间的置信度门控无法在原本充足的预算内认证旧任务行为保持不变。当结果分歧罕见时，一种标准的配对二项分布构造可以减轻这一负担。我们还规定了经认证的历史参考提升机制和轮次级的错过机会度量指标。在一个使用32个随机种子构建的单步推动诊断实验中，新鲜的配对检查在每阶段2,000个回合的预算下接纳了共同更新流中的31.6%，而基于区间的门控接纳率为零；然而在闭环运行中，无条件的回放学习效果反而更好。一项单独的习得动力学压力测试区分了模型偏差与反馈选择误差。

    arXiv:2609.10873v1 Announce Type: new  Abstract: Independent evaluation can reject harmful policy updates yet also prevent useful continual learning. We argue that update admission must be assessed through both error control and retained learning opportunities at a stated interaction budget. We identify a concrete failure: a range-based confidence gate cannot certify unchanged old-task behavior within otherwise substantial budgets. A standard paired-binomial construction reduces this burden when outcome disagreements are rare. We also specify certified historical-reference promotion and a round-level missed-opportunity metric. In a constructed one-step pushing diagnostic with 32 seeds, fresh paired checks admit 31.6% of a common update stream at 2,000 episodes per stage, versus zero for the range-based gate; unconditional replay nevertheless learns better in closed-loop runs. A separate learned-dynamics stress test distinguishes model bias from feedback-selection error. The contributio
    
[^146]: 无盒漏洞分析：仅基于描述的MCP服务器间接提示注入漏洞检测

    No-Box Vulnerability Analysis: Description-only Detection of Indirect Prompt Injection Vulnerabilities in MCP Servers

    [https://arxiv.org/abs/2609.10854](https://arxiv.org/abs/2609.10854)

    本文提出“无盒漏洞分析”新范式，仅凭功能描述元数据即可在不访问或不与目标系统交互的情况下，假设性检测MCP服务器中所有可能实现里的间接提示注入漏洞。

    

    传统的漏洞分析依赖于系统访问权限或动态交互，而这些对于审计闭源、远程托管的关键在位系统或商业受限软件的第三方分析师来说可能是不可获得的。因此，我们提出了一种新的范式——无盒漏洞分析，在这种范式下，既没有系统访问权限也没有运行时交互可用，仅有功能元数据可用。这些元数据定义了系统的预期行为，包括其输入、输出和副作用，同时约束了与该行为一致的实现空间。我们提出针对给定系统元数据的所有可能实现中存在的漏洞进行假设，而无需观察或与目标系统进行交互。当获得额外访问权限时，分析师可以随后验证这些假设。我们通过实现展示了无盒漏洞分析的可行性。

    arXiv:2609.10854v1 Announce Type: cross  Abstract: Conventional vulnerability analysis relies on either system access or dynamic interaction, all of which may be unavailable to third-party analysts auditing closed-source, remotely hosted, critical in situ systems, or commercially gated software. Therefore, we propose a new paradigm of no-box vulnerability analysis in which neither access nor runtime interaction is available, and only functionality metadata is available. Such metadata defines the intended behavior of the system, including its inputs, outputs, and side effects, while constraining the space of implementations consistent with that behavior. We propose hypothesizing about vulnerabilities that exist across all possible implementations of a given system metadata, without observing or interacting with the target system. An analyst can later validate these hypotheses when additional access is available. We showcase the feasibility of no-box vulnerability analysis through implem
    
[^147]: 我们真的在做少样本学习吗？对预训练假设的批判性审视

    Are We Really Doing Few-Shot Learning? A Critical Examination of Pre-Training Assumptions

    [https://arxiv.org/abs/2609.10851](https://arxiv.org/abs/2609.10851)

    本文通过系统比较四种预训练协议，揭示了当前少样本学习评估中因同域预训练带来的 9.66 个百分点乐观偏差，质疑现有评估方式能否真正反映模型的低数据学习能力。

    

    少样本学习通常在一种协议下进行评估，即先在一个大型辅助数据集上对模型进行预训练，该数据集的类别与目标任务 episodes 的类别不相交，但来自相同的视觉域。本文探讨了这种协议是否真正反映了低数据学习情境。我们在八个数据集、三种少样本学习架构以及多种 way-shot 设置下，系统地比较了无预训练、类别不相交的同域预训练、有监督的域外预训练以及无标签的域外预训练。我们的结果表明，仅靠类别不相交并不足以消除目标域数据的影响。同域预训练相比无预训练平均提升 33.41 个百分点，而有监督的域外预训练仅带来 23.75 个百分点的提升，从而揭示了与域重叠相关的 9.66 个百分点的乐观偏差。尽管在目标域数据可用性受限的应用场景中，域外预训练更符合实际情况……

    arXiv:2609.10851v1 Announce Type: cross  Abstract: Few-shot learning is commonly evaluated under protocols that pre-train a model on a large auxiliary set whose classes are disjoint from the target episodes yet drawn from the same visual domain. This paper examines whether such protocols truly reflect low-data learning. We systematically compare no pre-training, class-disjoint in-domain pre-training, supervised out-of-domain pre-training, and label-free out-of-domain pre-training across eight datasets, three few-shot architectures, and multiple way-shot settings. Our results show that class disjointness alone is insufficient to remove the influence of target-domain data. In-domain pre-training improves over no pre-training by 33.41 percentage points on average, whereas supervised out-of-domain pre-training yields 23.75 percentage points, revealing a 9.66-point optimistic bias associated with domain overlap. Although out-of-domain pre-training is more realistic in applications where tar
    
[^148]: 无大纲学习：任务无关的环境预处理

    Studying Without a Syllabus: Task-Agnostic Environment Preprocessing

    [https://arxiv.org/abs/2609.10824](https://arxiv.org/abs/2609.10824)

    该论文形式化了“任务无关的环境预处理”问题，证明LLM智能体可以在不知道下游任务分布的情况下，在测试前自主探索陌生环境并构建可复用工件来辅助冻结的求解器。

    

    在LLM智能体应对新环境中的任务之前，它可以检查可用的语料库和工具，并构建可复用的资源，例如索引、脚本或程序性指导。然而，大多数自动化适应方法依赖于任务示例、轨迹或评估反馈来决定构建什么。现有的任务无关方法虽然避免了这种监督，但会预先确定针对特定类型环境的准备策略。我们研究了一个更开放的问题设定：智能体能否在没有大纲的情况下研究一个陌生的环境——即在测试时间之前、且不了解下游任务分布的情况下，自主选择如何准备该环境？我们形式化了任务无关的环境预处理问题，其中一个学习系统在预算约束下探索环境，并为冻结的求解器生成工件。我们在跨多种环境上比较了无辅助的元智能体、配备档案的元智能体与固定的合成练习及语料库处理方法。

    arXiv:2609.10824v1 Announce Type: cross  Abstract: Before an LLM agent tackles tasks in a new environment, it can inspect available corpora and tools and construct reusable resources such as indices, scripts, or procedural guidance. Most automated adaptation methods, however, rely on task examples, trajectories, or evaluation feedback to decide what to build. Existing task-agnostic approaches avoid this supervision but commit in advance to a preparation strategy for a particular type of environment. We study a more open-ended setting: can an agent study an unfamiliar environment without a syllabus, i.e. before test time and without knowledge of the downstream task distribution, and choose how to prepare it? We formalize task-agnostic environment preprocessing, in which a studying system explores an environment under a budget and produces artifacts for a frozen solver. We compare unaided and archive-equipped meta-agents with fixed synthetic-practice and corpus-processing methods across 
    
[^149]: 同心协力，愈强则强：计算与合作的协同演化

    Tapes Together Strong: The Co-evolution of Computation and Cooperation

    [https://arxiv.org/abs/2609.10817](https://arxiv.org/abs/2609.10817)

    该论文提出“自创生博弈论”这一新计算模型，将社会互动、复制机制及其计算成本内生化并协同演化，并通过Z80机器码程序的实验证明，把社会困境直接嵌入计算的物理机制中能够促进自复制合作策略的涌现。

    

    合作是如何在复杂的智能体系统中演化的？先前演化博弈论领域的工作通过将社会互动与行为的物理成本相隔离，来研究个体为何有动力去合作；而人工生命模型传统上研究涌现的自复制现象，却没有形式化地刻画“获取资源”与“保存复制所需的共享能量”之间的困境。与之不同，我们提出了自创生博弈论，这是一个计算模型，其中社会互动、复制机制及其相关的计算成本都是内生的，并同时共同演化。我们使用由随机初始化的Z80机器码程序构成的计算基底来研究这些动力学，并通过实证分析（同时辅以一个简化的理论模型加以论证）表明：将社会困境直接嵌入计算的物理机制中，可以促进自复制的合作策略的涌现。当……

    arXiv:2609.10817v1 Announce Type: cross  Abstract: How does cooperation evolve in complex agentic systems? Prior work in evolutionary game theory studies why individuals are incentivized to cooperate by isolating social interactions from the physical costs of behavior, while artificial life models traditionally study emergent self-replication without formalizing the dilemma between acquiring resources and preserving the shared energy needed to reproduce. In contrast, we introduce Autopoietic Game Theory, a computational model where social interactions, replication mechanisms, and their associated computational costs are endogenous and simultaneously co-evolving. We study these dynamics using a computational substrate of randomly initialized programs in Z80 machine code, showing empirically, and motivating with a simplified theoretical model, that embedding a social dilemma directly into the physics of computation can favor the emergence of self-replicating, cooperative strategies. When
    
[^150]: 反事实边缘化：评估对干扰变量鲁棒性的框架

    Counterfactual Marginalisation: Framework for Evaluating Robustness to Nuisance Variables

    [https://arxiv.org/abs/2609.10778](https://arxiv.org/abs/2609.10778)

    提出反事实边缘化测试时评估框架，通过对年龄、性别等干扰变量进行反事实干预并平均预测，在保留患者个体信息的同时，定量评估分类模型对人口统计学捷径的鲁棒性。

    

    机器学习模型可能在依赖人口统计学或数据采集相关的捷径的情况下，仍能取得很强的测试性能。我们提出反事实（CF）边缘化作为一种测试时评估程序，用于评估分类模型对此类变量的鲁棒性。给定一个反事实图像生成器，我们对干扰父变量（如年龄或性别）进行干预，为每张测试图像生成反事实版本，并在目标干预分布上对预测结果进行平均。由此产生干预感知的预测，在保留患者特异性潜在信息的同时，边缘化人口统计学效应。我们利用这些预测定义了CF风险、校准性、稳定性和最坏情况敏感度等指标，并展示了该框架在定量鲁棒性评估中的实用性。

    arXiv:2609.10778v1 Announce Type: new  Abstract: Machine learning models can achieve strong test performance while relying on demographic or acquisition-related shortcuts. We propose counterfactual (CF) marginalisation as a test-time evaluation procedure for assessing robustness of classification models to such variables. Given a CF image generator, we intervene on nuisance parent variables such as age or sex, generate CF versions of each test image, and average predictions over a target intervention distribution. This produces intervention-aware predictions that marginalise demographic effects while preserving patient-specific latent information. We use these predictions to define metrics for CF risk, calibration, stability and worst-case sensitivity. We demonstrate this framework's utility for quantitative robustness evaluation.
    
[^151]: 超越静态保证：度量安全敏感及LLM生成的Python代码中“静态通过-动态失败”差距

    Beyond Static Guarantees: Measuring the Static-Pass Dynamic-Fail Gap in Security-Sensitive and LLM-Generated Python Code

    [https://arxiv.org/abs/2609.10762](https://arxiv.org/abs/2609.10762)

    该论文首次提出“静态通过-动态失败”（SPDF）现象，并设计了一个融合静态扫描、LLM驱动CWE推理与隔离容器内自主漏洞利用验证的三阶段智能体流水线，以量化静态分析通过但代码在运行时仍可被利用的安全评估盲区。

    

    大语言模型（LLM）的进展推动了可扩展方法的需求，以评估生成代码和安全敏感软件的安全性。静态分析因其可扩展、可复现且成本低而被广泛用作安全把关手段，但它无法直接观察运行时的漏洞利用行为。依赖于对抗性输入、执行上下文或漏洞利用链的漏洞可能逃过静态检查，却在实际中仍可被利用，然而通过静态分析往往被视为代码安全的证据。本文提出了“静态通过-动态失败”（Static-Pass Dynamic-Fail, SPDF）现象，并构建了一个三阶段智能体流水线，结合静态扫描、LLM驱动的常见弱点枚举（CWE）推理，以及在隔离Docker容器中进行的自主漏洞利用验证。研究团队在SecurityEval、RedCode和CyberNative数据集上评估了1,355个Python样本。在复合Bandit-Semgrep门控下未产生任何发现的654个样本中……（摘要在此处截断）

    arXiv:2609.10762v1 Announce Type: cross  Abstract: Advances in large language models (LLMs) fuel the quest for scalable methods to assess the security of generated and security-sensitive software. Static analysis is widely adopted as a scalable, reproducible, and inexpensive security gate, but cannot directly observe runtime exploit behaviour. Vulnerabilities dependent on adversarial inputs, execution context, or exploit chaining may evade static checks while remaining exploitable in practice, yet passing static analysis is often treated as evidence of secure behaviour. This paper introduces the Static-Pass Dynamic-Fail (SPDF) phenomenon and a three-stage agentic pipeline combining static scanning, LLM-driven Common Weakness Enumeration (CWE) reasoning, and autonomous exploit verification in isolated Docker containers. We evaluate 1,355 Python samples from SecurityEval, RedCode, and CyberNative datasets. Of the 654 samples producing no findings under the composite Bandit-Semgrep gate, 
    
[^152]: 徒有多语言虚名？大型语言模型在乌尔都语中的文化与语言缺陷

    Multilingual in Name Only? Cultural and Linguistic Weaknesses of LLMs in Urdu

    [https://arxiv.org/abs/2609.10758](https://arxiv.org/abs/2609.10758)

    本文通过构建包含93个AI生成乌尔都语故事的语料库并按九类语言、语义和文化错误进行人工标注，揭示了多语言大型语言模型在低资源语言故事生成中存在基本语法错误、缺乏连贯性和普遍文化浅薄等严重缺陷，且少样本提示无法有效解决文化错误问题。

    

    多语言大型语言模型越来越多地被用于开放式文本生成，但它们在低资源语言中的表现仍然鲜为人知。在这项工作中，我们质疑多语言LLM在故事生成任务中的输出正确性与可靠性。我们选择乌尔都语作为代表性的低资源语言。我们构建了Urdu-Stories语料库，包含使用三个当代LLM（GPT-5.1、Qwen-3-Max、DeepSeek-3.1）生成的93个故事。我们基于一个包含九个标签的语言、语义和文化分类体系，对这些故事中存在的错误进行了人工标注。我们的重要发现表明，LLM经常犯基本的语法和语义错误。故事缺乏连贯性，存在不自然的重复，并普遍表现出文化上的浅薄。我们进一步通过少样本提示实验表明，文化和语境层面的错误在很大程度上仍然无法解决。我们的发现凸显了当前LLM的局限性。

    arXiv:2609.10758v1 Announce Type: new  Abstract: Multilingual large language models (LLMs) are increasingly used for open-ended text generation, yet their behaviour in low-resource languages remains poorly understood. In this work, we question how correct and reliable is the generation of multilingual LLMs when used for the task of story generation. We consider Urdu language as a representative low-resource language. We generate Urdu-Stories, a corpus of 93 stories generated using three contemporary LLMs (GPT-5.1, Qwen-3-Max, DeepSeek-3.1). We manually annotate the errors present in them under a nine-label linguistic, semantic, and cultural taxonomy. Our notable findings suggest that LLMs often make basic errors of grammar and semantics. The stories lack coherence, have unnatural repetition and show pervasive cultural shallowness. We further show using few-shot prompting that the cultural and context errors largely remain unresolved. Our findings highlight the limitations of current LL
    
[^153]: 自适应边距序数损失：惩罚序数分类中的中心类对冲行为

    Adaptive Margin Ordinal Loss: Penalizing Center-Class Hedging in Ordinal Classification

    [https://arxiv.org/abs/2609.10752](https://arxiv.org/abs/2609.10752)

    该论文提出自适应边距序数损失（AMOL），通过对每个类别的损失项施加自适应乘法权重，在预测偏向中心类而真实标签远离中心时加大惩罚，从而直接抑制序数分类中的中心类对冲现象。

    

    标准交叉熵损失会导致在序数分类任务上训练的神经网络将预测向中心类“对冲”，我们将这种失败模式称为“中心类对冲”。这是因为预测中间类别能够最小化期望对称损失，使其成为无论真实标签为何都阻力最小的选择。现有的序数损失函数解决了诸如大误差惩罚和秩一致性等相关问题，但没有一种方法能够根据真实标签相对于序数中心的位置来直接抑制中心类对冲。我们提出了自适应边距序数损失（AMOL），它是一种作用于每个类别损失项的乘法权重，形式为 m(k,y) = 1 + α·(1 - |k-c|/c)·(|y-c|/c)，其中 c 为中心类，k 为候选类，y 为真实标签。该权重编码了一个联合条件：只有当候选类接近中心且真实标签远离中心时，该权重才会较大。

    arXiv:2609.10752v1 Announce Type: new  Abstract: Standard cross-entropy loss causes neural networks trained on ordinal classification tasks to hedge predictions toward center classes, a failure mode we term \emph{center-class hedging}. This occurs because predicting the middle class minimizes expected symmetric loss, making it the path of least resistance regardless of the true label. Existing ordinal losses address related problems such as large-error penalization and rank consistency, but none directly suppresses center-class hedging as a function of where the true label lies relative to the ordinal center. We propose the Adaptive Margin Ordinal Loss (AMOL), a multiplicative weight applied to per-class loss terms of the form $m(k,y) = 1 + \alpha \cdot (1 - |k-c|/c) \cdot (|y-c|/c)$, where $c$ is the center class, $k$ is the candidate class, and $y$ is the true label. The weight encodes a joint condition: it is large only when the candidate class is near center and the true label is f
    
[^154]: 当合成数据有害时：论LLM智能体技能检索中的灾难性遗忘

    When Synthetic Data Hurts: On Catastrophic Forgetting in Skill Retrieval for LLM Agents

    [https://arxiv.org/abs/2609.10750](https://arxiv.org/abs/2609.10750)

    研究发现合成数据微调虽能提升LLM智能体的分布内技能检索效果，但会导致对真实和分布外数据的灾难性遗忘，而借鉴持续学习的微调方法（如LwF、EWC等）既能缓解遗忘，又能将分布内检索性能提升13.98%。

    

    LLM智能体越来越依赖在运行时检索的外部技能，这使得从大型技能库中选择技能成为一个关键挑战。我们提出了一个覆盖34,396个技能的生产级技能路由器，并开展了一项使用有限真实监督和合成数据的大规模技能检索研究。我们发现，合成数据微调虽然能提升分布内检索效果，但会导致对真实数据和分布外（OOD）数据的灾难性遗忘。我们评估了几种受持续学习启发的遗忘缓解微调方法，包括嵌入锚点正则化、无遗忘学习（LwF）、弹性权重巩固（EWC）和L2初始化。结果表明，这些方法不仅能保持OOD技能检索的性能，还能将0.6B Qwen检索器和重排器在合成分布内技能上的检索性能提升13.98%。我们的研究结果提供了一个实用的基准和鲁棒的方案。

    arXiv:2609.10750v1 Announce Type: cross  Abstract: LLM agents increasingly rely on external skills retrieved at runtime, making skill selection from large repositories a critical challenge. We present a production skill router over 34,396 skills and a large-scale study of skill retrieval using limited real supervision and synthetic data. We found that the synthetic-data fine-tuning improves in-distribution retrieval but it causes catastrophic forgetting on real and out-of-distribution (OOD) data. We evaluate several forgetting mitigation fine-tuning approaches inspired by continual learning, including embedding-anchor regularization, Learning without Forgetting (LwF), Elastic Weight Consolidation (EWC), and L2-initialization. The results show that these approaches not only retain the performance on OOD skills retrieval but also improve the retrieval on synthetic in-distribution skills by 13.98\% for 0.6B Qwen retriever and reranker. Our results provide a practical benchmark and a robus
    
[^155]: 面向低轨（LEO）卫星系统网络攻击检测的时序与多模态深度学习方法

    Temporal and Multimodal Deep Learning for Cyberattack Detection in LEO Satellite Systems

    [https://arxiv.org/abs/2609.10746](https://arxiv.org/abs/2609.10746)

    本文基于卫星专用UNSW-IoTSAT数据集，系统研究了能够融合硬件、轨道与射频多模态信息并捕捉时序攻击行为的深度学习架构（如子系统融合MLP和分层多模态Transformer），以提升低轨卫星系统的网络攻击检测能力。

    

    对低地球轨道（LEO）卫星通信系统日益增长的依赖，增加了对能够在复杂动态空间环境中检测网络攻击的智能方法的需求。与传统网络入侵检测不同，卫星系统会在射频（RF）链路、星载硬件和轨道运行中产生异构信息。然而，许多现有方法要么依赖地面入侵检测数据集，要么对单个观测值进行独立评估，这限制了它们捕捉LEO卫星特有的时序攻击行为的能力。在这项工作中，我们使用最近推出的卫星专用UNSW-IoTSAT数据集，对基于深度学习的网络攻击检测进行了系统性研究。我们研究了能够保留硬件、轨道和射频信息的结构化学习架构，包括子系统融合MLP（Subsystem-Fusion MLP）以及建模……

    arXiv:2609.10746v1 Announce Type: cross  Abstract: The growing reliance on Low-Earth Orbit (LEO) satellite communication systems has increased the need for intelligent methods capable of detecting cyberattacks across complex and dynamic space environments. Unlike conventional network intrusion detection, satellite systems generate heterogeneous information across radio-frequency (RF) links, onboard hardware, and orbital operations. However, many existing approaches either rely on terrestrial intrusion datasets or evaluate individual observations independently, limiting their ability to capture temporal attack behavior specific to LEO satellites. In this work, we conduct a systematic study of deep-learning-based cyberattack detection using the recently introduced satellite-specific UNSW-IoTSAT dataset. We investigate structured learning architectures that preserve hardware, orbital, and RF information, including a Subsystem-Fusion MLP and a hierarchical multimodal Transformer that model
    
[^156]: CARTS：用于全容量密钥文本编码的上下文自回归秩转码隐写术

    CARTS: Contextual Autoregressive Rank Transcoding Steganography for Full-Capacity Keyed Text Encoding

    [https://arxiv.org/abs/2609.10744](https://arxiv.org/abs/2609.10744)

    本文首次对上下文自回归秩转码隐写术（CARTS）进行了严格的形式化安全分析，证明了其在确定性模型假设下的精确正确性，并定义和研究了上下文搜索、密钥碰撞、消息含糊化等相关计算问题及其理论关系。

    

    自回归语言模型可以通过在各上下文中保留每个位置的秩信息，将载荷文本转换为词元长度相同的隐写文本——我们将这一方法论形式化为上下文自回归秩转码隐写术（CARTS）。尽管Norelli等人的Calgacus构造通过实验演示了这一现象，但此前并不存在形式化的安全分析。本文首次对CARTS进行了严格的理论处理。我们证明了其在确定性模型假设下的精确正确性，引入了一种秩坐标表示，其中密钥在秩向量空间上充当双射，定义了相关的安全概念以及与该构造自然关联的计算问题——上下文搜索、密钥碰撞、消息含糊化以及编码映射的非交换性——并研究了这些问题之间的理论关系，包括对消息含糊化的刻画。

    arXiv:2609.10744v1 Announce Type: cross  Abstract: Autoregressive language models can be used to transform a payload text into a stegotext of identical token length by preserving per-position rank information across contexts - a methodology we formalize as Contextual Autoregressive Rank Transcoding Steganography (CARTS). While the Calgacus construction of Norelli et al. demonstrated this phenomenon experimentally, no formal security analysis existed. This paper provides the first rigorous treatment of CARTS. We show its exact correctness under deterministic model assumptions, introduce a rank-coordinate representation in which keys act as bijections on rank-vector space, define relevant security notions and the computational problems naturally associated with the construction - context search, key collisions, message equivocation, and non-commutativity of the encoding maps - and study the theoretical relationships between them, including the characterization of message equivocation in 
    
[^157]: 真理从未消失：顺从情境下真值探测器的完美混叠现象

    The Truth Was Never Gone: Perfect Aliasing in Compliant-Context Truth Probes

    [https://arxiv.org/abs/2609.10739](https://arxiv.org/abs/2609.10739)

    论文揭示了真值探测器的“完美混叠”失效机制——当真实汇报与任务既定行为在顺从情境中重合时探测器无法区分二者（二者AUROC恒互补求和为一），并提出在顺从与对立情境的混合数据上拟合的方法，使探测器即使在模型系统性说谎时也能以完美的1.0 AUROC识别真值。

    

    在真实汇报与任务既定行为相重合的情境中拟合的真值探测器，仅凭其拟合标签无法区分这两个目标。我们将这种语义识别的失效称为“完美混叠”（perfect aliasing）。在一个受控的二值汇报博弈中，在顺从情境上拟合的真值探测器与既定行为探测器求解的是同一个优化问题。在对立情境上，二者的标签互为补集，导致它们的AUROC之和恒为一；这一恒等关系在751个单元-层对上以浮点精度成立。我们利用随机化码本将既定输出符号与语义行为分离，进而通过在顺从情境与对立情境的混合数据上拟合，将真值与既定行为区分开来。对于一个经奖励训练、在所有评估的对立试验中均给出虚假回答的Gemma-2-9B策略，常规探测器在三个训练种子上的AUROC仅为0.006 ± 0.005，而混合拟合探测器在相同的保留激活值上得分达到1.000。混合拟合……

    arXiv:2609.10739v1 Announce Type: cross  Abstract: A truth probe fitted where truthful reporting and a task's prescribed action coincide cannot distinguish those targets from its fitting labels alone. We call this failure of semantic identification perfect aliasing. In a controlled binary reporting game, truth and prescribed-action probes fitted on compliant contexts solve the same optimization. On rival contexts their labels are complements, forcing their AUROCs to sum to one; this identity holds across 751 cell-layer pairs to floating-point precision. We separate prescribed output symbols from semantic action using randomized codebooks, then separate truth from prescribed action by fitting on mixed compliant and rival contexts. For a reward-trained Gemma-2-9B policy that answers falsely on all evaluated rival trials, the conventional probe scores $0.006 \pm 0.005$ AUROC across three training seeds, while mixed-fit probes score $1.000$ on the same held-out activations. Mixed fitting u
    
[^158]: 迈向临床语言模型的确定性数学求解器

    Towards a Deterministic Math Solver for Clinical Language Models

    [https://arxiv.org/abs/2609.10728](https://arxiv.org/abs/2609.10728)

    本文提出让临床大语言模型不直接进行算术计算，而是生成针对性Python代码交由受限本地执行器作为确定性求解器运行，但在MedCalc-Bench Verified基准上的评估表明，在公式和标准变量均已提供的情况下，这种程序求解接口相比模型直接计算并无可靠优势。

    

    大型语言模型在算术运算方面并不可靠，这对临床计算器而言是一个严重问题，因为单个数值错误就可能改变医疗建议。标准的应对方法是将每个计算器逐一硬编码为经过验证的函数。我们测试了一种替代方案：模型本身不进行计算，而是编写针对具体病例的Python代码，由一个受限的本地执行器作为确定性求解器运行，模型的任务则简化为决定如何使用该求解器。我们在依据现行临床指南对基准测试的公式进行审计、并标记出55个计算器中16个存在版本、用途或系数方面的疑虑之后，使用Qwen2.5-7B和Qwen2.5-32B-AWQ模型，在MedCalc-Bench Verified基准（1,100个病例、55个计算器）上将这种“程序-求解”接口与模型的直接算术运算以及手工编写的22个计算器库进行了对比评估。结果表明，在提供公式和标准变量、且两种方法都能读取完整病历的情况下，将计算任务交给求解器并不能带来可靠的优势。

    arXiv:2609.10728v1 Announce Type: cross  Abstract: Large language models are unreliable at arithmetic, which is a problem for clinical calculators where a single numerical error changes the recommendation. The standard response is to hardcode each calculator as a validated function, one at a time. We test an alternative: the model does not calculate. Instead, it writes case-specific Python that a restricted local executor runs as a deterministic solver, and the model's task reduces to deciding how to use it. We evaluate this Program-Solve interface on MedCalc-Bench Verified (1,100 cases, 55 calculators) against direct model arithmetic and a hand-written 22-calculator library, using Qwen2.5-7B and Qwen2.5-32B-AWQ, after auditing the benchmark's formulas against current clinical guidelines and flagging 16 of 55 with version, use or coefficient concerns. With formulas and gold variables supplied and both routes reading the whole note, handing off to the solver is not a reliable advantage 
    
[^159]: 完成任务还不够：在挑战持续累积下评估智能体的韧性与体贴参与

    Finishing the Task Is Not Enough: Evaluating Agent Resilience and Considerate Participation under Accumulating Challenge

    [https://arxiv.org/abs/2609.10724](https://arxiv.org/abs/2609.10724)

    该论文提出以“运营韧性”和“体贴参与”这两个互补维度来评估生成式AI智能体在挑战不断累积的持续部署场景下的表现，并通过120条模拟医疗保健轨迹验证了这一评估框架。

    

    生成式AI智能体的持续部署所需要的不仅仅是孤立的oneshot任务成功。智能体必须在重复交互、条件变化以及对共享工作流中人员的依赖下保持有效，尤其是当技术、人员和运营层面的干扰随时间不断累积时。我们提出“运营韧性”和“体贴参与”作为评估此类智能体的两个互补维度：前者刻画智能体如何从受阻工作中恢复、保留已有进度并传达自身局限；后者刻画智能体在适应调整时如何顾及受影响的人员、角色边界以及周围的工作流程。然而，这两个方面在挑战不断累积的情形下仍未得到充分研究。我们在轻度、中度和重度挑战条件下，研究了涵盖两个生成式AI模型和十二个由利益相关者提炼出的任务的120条模拟医疗保健轨迹。我们比较了文本行动计划、通过提示引导的内部评估以及定量（摘要在此处截断，原文未提供完整内容）

    arXiv:2609.10724v1 Announce Type: new  Abstract: Sustained deployment of generative AI agents requires more than isolated task success. Agents must remain useful across repeated interactions, changing conditions, and dependencies on people within shared workflows, especially as technical, human, and operational disruptions accumulate over time. We propose operational resilience and considerate participation as two complementary aspects of evaluating such agents: the former captures how agents recover from blocked work while preserving progress and communicating their limits, and the latter captures how their adaptation accounts for affected people, role boundaries, and the surrounding workflow. Yet both remain underexplored under accumulating challenge. We study 120 simulated healthcare trajectories across two generative AI models and twelve stakeholder-derived tasks under light, medium, and heavy challenge. We compare textual action plans, prompted internal assessments, and quantitati
    
[^160]: AcFlow：通过学习的条件激活流控制文本到图像扩散Transformer

    AcFlow: Controlling Text-to-Image Diffusion Transformers via Learned Conditional Activation Flow

    [https://arxiv.org/abs/2609.10723](https://arxiv.org/abs/2609.10723)

    AcFlow是一种推理时控制器，通过学习概念条件的速度场传输中间层图像token激活，在保持基础扩散Transformer冻结的情况下实现对文本到图像生成的连续风格强度控制与不想要概念抑制，并能泛化到训练中未见过的概念。

    

    文本到图像扩散Transformer（DiT）是强大的生成器，然而直接提示词方式为风格强度提供的控制接口有限，且可能无法抑制不想要的概念。为实现这些控制，我们提出了AcFlow——一种推理时控制器，它通过学习到的概念条件速度场来传输中间层图像token激活，同时保持基础DiT冻结。文本概念描述指定期望的干预，而积分区间（integration horizon）则提供了一个连续的控制参数。该速度场产生随token变化的、依赖激活状态的更新。通过在每个任务族内的各概念之间共享参数，该速度场支持细粒度的描述，并且能够泛化到训练期间未见过的概念，无需针对每个概念单独拟合。在风格控制方面，AcFlow在高风格对齐区间中，在所评估的基线方法中实现了最佳的风格-内容权衡。在固定的op……（摘要在此处被截断）

    arXiv:2609.10723v1 Announce Type: cross  Abstract: Text-to-image diffusion transformers (DiTs) are powerful generators, yet direct prompting provides limited control interface for style intensity and can fail to suppress unwanted concepts. To enable these controls, we introduce AcFlow, an inference-time controller that transports intermediate layer image-token activations through a learned concept-conditioned velocity field while keeping the base DiT frozen. A textual concept description specifies the desired intervention, while the integration horizon provides a continuous control parameter. The field produces token-varying, activation-dependent updates. With parameters shared across concepts within each task family, the field supports fine-grained descriptions and generalizes to concepts unseen during training without per-concept fitting. On style control, AcFlow achieves the best style--content trade-off among the evaluated baselines in the high-style-alignment regime. At a fixed op
    
[^161]: IMO金牌的开放配方：面向奥林匹克数学训练Nemotron

    An Open Recipe for IMO Gold: Training Nemotron for Olympiad Mathematics

    [https://arxiv.org/abs/2609.10712](https://arxiv.org/abs/2609.10712)

    该研究通过监督微调与强化学习后训练Nemotron 3 Ultra模型，构建了一个纯自然语言、无形式化证明器与外部工具的迭代生成-验证-细化测试时计算流水线，在IMO 2026中获得30/42分达到金牌水平，并开源了模型检查点、训练数据与代码。

    

    我们研究了模型后训练与测试时推理设计如何影响针对高难度奥数题的自然语言证明生成。从Nemotron 3 Ultra出发，我们使用监督微调和强化学习训练了两个专用检查点，并评估了检查点选择、验证与细化机制。基于这些发现，我们提出了一个开放模型的测试时计算流水线。该系统完全以自然语言运行，不使用形式化证明器、外部工具或互联网访问。三个Nemotron 3 Ultra检查点——通用版本模型和两个后训练专用模型——驱动一个迭代搜索过程，生成、验证并细化候选证明；随后由一个独立的高计算阶段选出每次最终提交的答案。该系统在2026年国际数学奥林匹克竞赛（IMO）中获得42分中的30分，达到了金牌分数线。我们发布了这两个后训练检查点以及训练数据、训练和推理代码。

    arXiv:2609.10712v1 Announce Type: new  Abstract: We study how model post-training and test-time inference design affect natural-language proof generation for hard olympiad mathematics. Starting from Nemotron 3 Ultra, we train two specialist checkpoints using supervised fine-tuning and reinforcement learning, and evaluate checkpoint choice, verification, and refinement. Based on these findings, we present an open-model test-time-compute pipeline. The system operates entirely in natural language, with no formal prover, external tools, or internet access. Three Nemotron 3 Ultra checkpoints - the general-availability model and two post-trained specialists - power an iterative search that generates, verifies, and refines candidate proofs; a separate high-compute stage then selects each final submission. The system scored 30 out of 42 points at IMO 2026, reaching the gold-medal threshold. We release the two post-trained checkpoints as well as the training data, the training and inference cod
    
[^162]: 构建安全的人工智能安全运营中心（AI-SOC）：一种保障管道完整性与威胁缓解的神经符号框架

    Architecting the Secure AI-SOC: A Neurosymbolic Framework for Pipeline Integrity and Threat Mitigation

    [https://arxiv.org/abs/2609.10707](https://arxiv.org/abs/2609.10707)

    本文提出一种神经符号纵深防御架构，通过确定性SIEM解码器预过滤与神经语义评估相结合的两层机制，抵御针对AI安全运营中心中LLM的日志投毒式间接提示注入攻击，实现管道的端到端完整性保障与威胁缓解。

    

    arXiv:2609.10707v1 公告类型：交叉 摘要：将大语言模型（LLM）集成到安全运营中心（SOC）中虽然简化了威胁情报工作，但也引入了关键的漏洞，尤其是通过日志投毒实现的间接提示注入。攻击者利用这一攻击途径，通过在系统日志中嵌入恶意载荷来劫持LLM的操作逻辑，从而执行多步骤的“提示件”（promptware）杀伤链。保护这一管道面临两难困境：确定性防御在计算上高效但在语义上是盲目的，而纯神经评估则带来难以承受的延迟和概率性缺陷。为解决这一问题，我们提出了一种新颖的神经符号纵深防御架构，以确保端到端的管道完整性。第一层采用定制的SIEM解码器作为确定性预过滤器，在数据摄取边缘执行即时结构净化，以中和体量填充和基于签名的注入。第二层……（摘要在此截断）

    arXiv:2609.10707v1 Announce Type: cross  Abstract: The integration of Large Language Models (LLMs) into Security Operations Centers (SOCs) streamlines threat intelligence but introduces critical vulnerabilities, notably indirect prompt injection via log poisoning. Adversaries exploit this vector to execute multistep ``promptware'' kill chains by embedding malicious payloads within system logs to hijack the LLM's operational logic. Securing this pipeline presents a dichotomy: deterministic defenses are computationally efficient yet semantically blind, while purely neural evaluations introduce prohibitive latency and probabilistic flaws. To address this, we propose a novel neurosymbolic defense-in-depth architecture that ensures end-to-end pipeline integrity. The primary layer employs customized SIEM decoders as a deterministic pre-filter, performing immediate structural sanitization to neutralize volumetric padding and signature-based injections at the ingestion edge. The secondary laye
    
[^163]: 数据高效的语言建模：从前沿突破到原则指导的模型改进

    Data-Efficient Language Modeling: From Frontier Advancement to Principle-Guided Model Improvement

    [https://arxiv.org/abs/2609.10702](https://arxiv.org/abs/2609.10702)

    该研究通过三阶段自主研究计划，在BabyLM 2026 Strict-Small受限数据设置下发现精确重复与对齐重述产生不同的上下文利用模式，并提出围绕预训练所需上下文依赖来组织经验的数据高效学习原则，实现了从构建前沿模型到原则指导的模型改进。

    

    从有限文本中学习要求模型能够利用上下文、泛化到新输入并保留有用的能力。求是引擎在BabyLM 2026 Strict-Small任务上开展了一项长周期、端到端的自主研究计划，语料库规模限制在1000万词以内，累计词呈现量为1亿次。该计划分为三个阶段，将前沿突破、原则发现和原则指导的模型改进联系起来。第一阶段结合紧凑重述、预算再投资和残差增量学习来构建前沿模型。第二阶段发现，精确重复和对齐重述会根据目标关系和预测窗口的不同，产生不同的上下文利用模式。在受控任务中，恢复熟悉样本上的性能并不能确保未见过的输入仍能利用已学到的计算。这些发现支持了一个可检验的数据高效学习原则：围绕预训练所需的上下文依赖来组织经验……（原文摘要在此处截断）

    arXiv:2609.10702v1 Announce Type: new  Abstract: Learning from limited text requires models to use context, generalize to new inputs, and retain useful capabilities. Qiushi Engine conducted a long-horizon, end-to-end autonomous research program on BabyLM 2026 Strict-Small, within 10 million corpus words and 100 million cumulative word presentations. Three stages connected frontier advancement, principle discovery, and principle-guided model improvement. Stage I combined compact restatements, budget reinvestment, and residual incremental learning to build a frontier model. Stage II found that exact repetition and aligned restatement produce different patterns of context use, depending on target relations and prediction windows. In controlled tasks, recovering familiar performance did not ensure that unseen inputs could still use learned computations. These findings support a testable data-efficient learning principle: organize experience around the contextual dependencies needed for pre
    
[^164]: 量化从记忆到泛化的转变：Grokking现象中的缩放定律与相结构

    Quantifying the Memorization-to-Generalization Transition: Scaling Laws and Phase Structure in Grokking

    [https://arxiv.org/abs/2609.10657](https://arxiv.org/abs/2609.10657)

    本研究通过384组配置的系统实验首次量化了grokking现象中记忆到泛化转变的幂律缩放定律，发现数据复杂度（而非模型容量）是驱动这一相变的主导因素，且在权重衰减λ ≳ 1.0处存在尖锐的相边界。

    

    在记忆阶段之后继续训练的神经网络经常会经历一个延迟的向泛化能力的转变，这一现象被称为grokking（顿悟）。尽管关于这一转变“为何”发生的理论研究已取得进展，但其在超参数空间中“何时”发生的定量结构仍未被刻画。我们在模运算任务上对384种双隐藏层多层感知机（MLP）配置进行了系统实验，绘制出记忆到泛化的边界，并拟合出泛化起始时间的幂律缩放关系：T_grok ∝ H^-0.27 D^-2.04 η^-0.50 λ^-0.64（R² = 0.732；加入交互项后为0.821）。指数的层级结构揭示，数据复杂度（D^-2.04）是相变的主导驱动力，而非模型容量（H^-0.27）：数据量翻倍可使泛化加速约4倍，而网络宽度翻倍仅带来约1.2倍的提升。在权重衰减λ ≳ 1.0处存在一个尖锐的相边界，将……（原文摘要在此处截断）

    arXiv:2609.10657v1 Announce Type: cross  Abstract: Neural networks trained past memorization frequently undergo a delayed transition to generalization, a phenomenon known as grokking. Despite theoretical progress on \emph{why} this transition occurs, the quantitative structure of \emph{when} it occurs in hyperparameter space remains uncharacterized. We map the memorization-to-generalization boundary across 384 configurations of two-hidden-layer MLPs on modular arithmetic, fitting a power-law scaling relation for generalization onset time: $T_{\mathrm{grok}} \propto H^{-0.27}\, D^{-2.04}\, \eta^{-0.50}\, \lambda^{-0.64}$ ($R^2 = 0.732$; $0.821$ with interactions). The exponent hierarchy reveals that data complexity ($D^{-2.04}$) is the dominant driver of regime transition, not model capacity ($H^{-0.27}$): doubling data accelerates generalization by ${\sim}4\times$, while doubling width yields only ${\sim}1.2\times$. A sharp phase boundary at weight decay $\lambda \gtrsim 1.0$ separates
    
[^165]: 理解扩散模型微调中 LoRA 秩的权衡

    Understanding LoRA Rank Trade-offs in Diffusion Model Fine-Tuning

    [https://arxiv.org/abs/2609.10656](https://arxiv.org/abs/2609.10656)

    扩散模型 LoRA 微调研究表明，在固定训练预算下，小到中等的秩（如 4 或 8）能以更低的计算成本达到最佳生成质量，而更高的秩收益有限。

    

    为扩散模型微调选择 LoRA 秩需要在生成质量与计算成本之间进行权衡。我们在 CIFAR-10 数据集上使用 DDPM U-Net（秩为 {2,4,8,16,32}）、固定的优化设置以及可复现的本地文件夹 pytorch-fid 评估协议，开展了一项受控研究。我们报告了 FID、可训练参数量、运行时间和 GPU 显存占用，并通过扩展训练预算的 DDPM 实验（20 个 epoch；秩 4/8/16）以及 Tiny DiT 骨干网络实验（10 个 epoch；秩 4/8/16）验证了相关趋势。结果表明，中等秩最为高效：秩 4 取得了最佳的 DDPM FID（124.1380），秩 8 表现接近（124.2136），而更高的秩尽管适应成本更大，带来的收益却十分有限。这些发现支持在固定训练预算下将小到中等的秩作为实用的默认选择。

    arXiv:2609.10656v1 Announce Type: cross  Abstract: Selecting LoRA rank for diffusion fine-tuning requires balancing quality and compute cost. We present a controlled study on CIFAR-10 using a DDPM U-Net with ranks {2,4,8,16,32}, fixed optimization settings, and a reproducible local-folder pytorch-fid protocol. We report FID, trainable parameters, runtime, and GPU memory, then validate trends with extended-budget DDPM runs (20 epochs; ranks 4/8/16) and a Tiny DiT backbone (10 epochs; ranks 4/8/16). Results show moderate ranks are most efficient: rank 4 achieves the best DDPM FID (124.1380), rank 8 is close (124.2136), and higher ranks provide limited gains despite larger adaptation cost. These findings support small-to-moderate ranks as practical defaults under fixed training budgets.
    
[^166]: 一种用于组合式与可解释认知推理的多阶段规则链式框架

    A Multi-Stage Rule-Chaining Framework for Compositional and Interpretable Cognitive Reasoning

    [https://arxiv.org/abs/2609.10654](https://arxiv.org/abs/2609.10654)

    该论文提出了一种多阶段规则链式框架，通过集成确定性规则发现、模式组合引擎和结构抽象层三个互补求解器，在渐进式回退层次结构中实现组合式且可解释的认知推理，在ARC基准的1000个训练任务中通过了995个。

    

    抽象与推理语料库（ARC）是衡量认知泛化能力的基准，即从有限示例中推断并应用抽象规则的能力。本文提出了一种多阶段规则链式框架，能够在符号、结构和概念三个层面上执行组合式推理。该框架集成了三个互补的求解器：（1）确定性规则发现模块，通过几何、颜色和基于对象的分析来归纳原子变换；（2）模式组合引擎，通过块合并、重复和空间启发式方法重建输出；（3）结构抽象层，用于推断网格之间的层级和嵌套关系。这些求解器在一个渐进式回退层次结构中按顺序运行，每个阶段都会复用先前的推理轨迹，以增强可解释性和泛化能力。在1000个训练任务中通过了995个任务，并在105个任务上进行了进一步评估。

    arXiv:2609.10654v1 Announce Type: cross  Abstract: The Abstraction and Reasoning Corpus (ARC) benchmarks cognitive generalization, the ability to infer and apply abstract rules from limited examples. This paper presents a multi-stage rule-chaining framework that performs compositional reasoning across symbolic, structural, and conceptual levels. The framework integrates three complementary solvers:   (1) a deterministic rule discovery module that induces atomic transformations through geometric, color, and object-based analysis;   (2) a pattern-composition engine that reconstructs outputs via block merging, repetition, and spatial heuristics; and   (3) a structural abstraction layer that infers hierarchical and nested relationships across grids.   These solvers operate sequentially within a progressive fallback hierarchy, where each stage reuses prior reasoning traces to enhance interpretability and generalization. Training passed for 995 tasks out of 1000, further evaluated on 105 tas
    
[^167]: 从自然语言自动生成二次无约束二元优化（QUBO）公式

    Automating Quadratic Unconstrained Binary Optimization (QUBO) Formulation Generation from Natural Language

    [https://arxiv.org/abs/2609.10629](https://arxiv.org/abs/2609.10629)

    本文提出一个端到端多智能体框架，可自动将自然语言问题描述转化为QUBO优化公式，并发布了涵盖12个应用领域、包含100个组合优化问题的QUBOBench基准数据集。

    

    二次无约束二元优化（QUBO）是组合优化的核心公式形式，因其与量子求解器、量子-经典混合求解器以及量子启发式求解器的兼容性而日益受到关注。然而，将自然语言问题描述转化为正确的QUBO公式仍然困难，这一过程需要识别二元变量、约束条件、目标函数、惩罚项以及合适的惩罚权重。该过程耗时且通常需要大量的领域专业知识。为应对这一挑战，我们提出了一个端到端的多智能体框架，可在结构化或非结构化测试用例的支持下，从自然语言问题描述中自动生成QUBO公式。为评估其性能，我们还引入了QUBOBench，这是一个包含来自同行评审文献的12个应用领域共100个组合优化问题的基准测试集。

    arXiv:2609.10629v1 Announce Type: new  Abstract: Quadratic Unconstrained Binary Optimization (QUBO) is a central formulation for combinatorial optimization and has gained increasing attention due to its compatibility with quantum, hybrid quantum-classical, and quantum-inspired solvers. However, translating natural-language problem descriptions into correct QUBO formulations remains difficult, requiring the identification of binary variables, constraints, objective functions, penalty terms, and suitable penalty weights. This process is time-consuming and often demands substantial domain expertise. To address this challenge, we propose an end-to-end multi-agent framework that automatically generates QUBO formulations from natural-language problem descriptions, supported by structured or unstructured test cases. To evaluate its performance, We also introduce QUBOBench, a benchmark containing 100 combinatorial optimization problems across 12 application domains, curated from peer-reviewed 
    
[^168]: 概率焦点搜索：通过下界推进加速有界次优搜索

    Probabilistic Focal Search: Accelerating Bounded-Suboptimal Search via Lower-Bound Advancement

    [https://arxiv.org/abs/2609.10584](https://arxiv.org/abs/2609.10584)

    提出概率焦点搜索（PFS），以一定概率扩展最小 $f$ 值节点来推进下界、扩大FOCAL集合，在启发式引导与下界推进之间取得平衡，从而加速有界次优搜索。

    

    有界次优搜索旨在找到与最优解相差因子 $w$ 范围内的解，同时减少搜索工作量。焦点搜索（FS）在FOCAL集合（即在阈值 $w f_{\min}$ 下符合条件的边界节点）内使用启发式引导，但其确定性策略可能导致 $f_{\min}$ 在多次节点扩展中保持不变。我们提出了概率焦点搜索（PFS），它以概率 $p$ 遵循FS的引导选择，以概率 $1-p$ 扩展OPEN列表中具有最小 $f$ 值的节点。后一种分支会促使下界向前推进，从而扩大FOCAL集合，并接纳可能通向可行解的节点。通过平衡启发式引导与下界推进，当搜索进展受限于FOCAL接纳延迟时，该机制能够减少获得有界解所需的时间。作为次要的迁移实验，我们将相同的调度器应用于动态势能搜索，得到了概率动态势能搜索（PDPS）。我们在N数码、煎饼排序等基准问题上将PFS与FS进行了对比评估。

    arXiv:2609.10584v1 Announce Type: new  Abstract: Bounded-suboptimal search seeks a solution within a factor $w$ of optimal while reducing search effort. Focal Search (FS) uses heuristic guidance within FOCAL, the frontier nodes eligible under the threshold $w f_{\min}$, but its deterministic policy may leave $f_{\min}$ unchanged for many expansions. We introduce Probabilistic Focal Search (PFS), which follows the FS guided choice with probability $p$ and expands a minimum-$f$ OPEN node with probability $1-p$. The latter branch encourages the lower bound to advance, enlarging FOCAL and admitting nodes that may lead to feasible solutions. By balancing guidance and lower-bound advancement, this mechanism can reduce time to a bounded solution when progress is limited by delayed FOCAL admission. As a secondary transfer experiment, we apply the same scheduler to Dynamic Potential Search, yielding Probabilistic Dynamic Potential Search (PDPS). We benchmark PFS against FS on N-Puzzle, Pancake 
    
[^169]: 关于二进制对称信道最优 (n,4) 二进制码的 Dong-Yang 分类的机器检验证明

    A machine-checked proof of the Dong-Yang classification of optimal (n,4) binary codes for BSCs

    [https://arxiv.org/abs/2609.10579](https://arxiv.org/abs/2609.10579)

    本文利用 AI 工具辅助开发了 Dong-Yang 二进制对称信道最优 (n,4) 二进制码分类定理的 Lean 4 机器检验形式化证明，并在过程中修正了 AI 生成代码的错误、发现了原论文中的不一致之处。

    

    我们提出了 Dong 和 Yang 关于二进制对称信道最优有限长 $(n,4)$ 二进制分组码分类的 Lean 4 机器检验形式化。该形式化主要通过将论文中的证明输入 AI 工具来开发。为了确立正确性，作者在 Lean 中验证了主要定理陈述及所接受的公理。本文讨论了对 AI 生成的形式化所做的修正与简化，并记录了在形式化过程中发现的论文中的不一致之处。Lean 代码可在 https://github.com/shhyang/n4code_lean 获取。

    arXiv:2609.10579v1 Announce Type: cross  Abstract: We present a machine-checked Lean~4 formalization of Dong and Yang's classification of optimal finite-length $(n,4)$ binary block codes for binary symmetric channels. The formalization was developed mainly by feeding the paper's proofs to an AI tool. To establish correctness, the authors verified the main theorem statements in Lean and the accepted axioms. This note discusses the corrections and simplifications made to the AI-generated formalization, and records discrepancies found in the paper during the formalization. The Lean code is available at https://github.com/shhyang/n4code_lean.
    
[^170]: PACE：面向QoE高效检索增强对话服务的感知延迟感知级联服务路由与填充器控制

    PACE: Perceived-Latency-Aware Cascading Service Routing and Filler Control for QoE-Efficient Retrieval-Augmented Dialogue Serving

    [https://arxiv.org/abs/2609.10372](https://arxiv.org/abs/2609.10372)

    PACE框架将感知首次响应时间（PTFR）作为QoE目标，通过负载自适应级联路由、路径-填充器联合控制和波动感知缓存准入三种机制，在人形机器人对话服务中显著降低感知延迟并保证回答质量与新鲜度。

    

    我们提出了PACE，一个检索增强对话服务框架，它将感知首次响应时间（PTFR）形式化为QoE目标，并在质量/成本约束下将其最小化。与以往关于级联路由、语义缓存或自适应检索的工作不同，PACE联合控制由哪个答案源构成响应，以及用什么内容填充等待窗口。该框架部署在人形机器人销售服务上，结合了三种机制：负载自适应级联路由器、路径-填充器联合控制器和波动感知缓存准入。在75k条CarQA请求上，级联机制将P95处的纯LLM PTFR减半（c16并发下为0.29秒对比0.53秒）。自适应控制器达到0.41秒的P95，在高负载且质量相当的情况下，性能优于RAG达2.4倍。填充器控制器将调用次数减少了94%，且零冲突。波动感知准入将过时答案的比例从86%降低到0%。门控规则确保控制器性能永远不会差于基线。

    arXiv:2609.10372v2 Announce Type: cross  Abstract: We present the PACE, a framework for retrieval-augmented dialogue serving that formalizes Perceived Time-to-First-Response (PTFR) as a QoE objective and minimizes it under quality/cost constraints. Unlike prior work on cascaded routing, semantic caching, or adaptive retrieval, PACE jointly controls which answer source composes the response and what fills the waiting window. Deployed on a humanoid-robot sales service, it combines three mechanisms: a load-adaptive cascading router, a joint path-filler controller, and volatility-aware cache admission. On 75k CarQA requests, the cascade halves pure-LLM PTFR at P95 (0.29 vs 0.53s at c16). The adaptive controller reaches 0.41s P95, outperforming RAG by 2.4 times at high load with equal quality. The filler controller cuts calls by 94% with zero conflict. Volatility-aware admission reduces stale answers from 86% to 0%. A gating rule ensures the controller never worse than the baseline, with ex
    
[^171]: 为什么采样可以枚举的内容？面向基因组工具选择的精确策略优化

    Why Sample What You Can Enumerate? Exact Policy Optimization for Genomic Tool Selection

    [https://arxiv.org/abs/2609.10221](https://arxiv.org/abs/2609.10221)

    该论文揭示了在工具子集空间可完全枚举的基因组推理等科学领域中，GRPO等基于采样的强化学习方法存在结构性缺陷（训练越成功奖励信号越稀缺），并提出FGPO（全组策略优化），通过对所有工具子集进行精确枚举评分来替代采样估计。

    

    在冻结的推理器上进行强化学习已成为教导策略调用哪些外部工具的常见方法。我们证明，在完整的工具子集空间可枚举的专业科学环境中，这种方案存在结构性不匹配。在那里，一小组反复出现的计算能力即可覆盖整个领域，因此工具子集的空间虽然是组合性的，但足够小以至于可以枚举，而GRPO仍然从少量采样的rollouts中估计动作期望。更糟糕的是，随着训练的成功，这种近似反而会退化：当策略集中于偏好的子集时会重复采样它们，采样的奖励发生冲突，组归一化的优势随之消失。在基因组推理任务中，未产生奖励信号的问题比例从均匀参考策略下的0.2%上升到GRPO训练后的20.8%。作为补救措施，我们引入了FGPO（全组策略优化），它（1）对每个工具子集进行精确评分……（原文摘要被截断）

    arXiv:2609.10221v2 Announce Type: new  Abstract: Reinforcement learning over a frozen reasoner has become a common recipe for teaching a policy which external tools to invoke. We show that this recipe becomes structurally mismatched in specialist scientific settings where the complete tool-subset space is enumerable. There, a small set of recurring computational capabilities covers the domain, so the space of tool subsets is combinatorial yet small enough to enumerate, and GRPO still estimates an action expectation from a handful of sampled rollouts. Worse, the approximation degrades as training succeeds: as the policy concentrates on preferred subsets it resamples them, sampled rewards collide, and the group-normalized advantage vanishes. On genomic reasoning the fraction of questions yielding no reward signal rises from 0.2% under a uniform reference policy to 20.8% after GRPO training. As a remedy, we introduce FGPO (Full-Group Policy Optimization), which (1) scores every tool subse
    
[^172]: 超越已验证答案：基于求解器信息的自蒸馏方法用于自举运筹学语言模型

    Beyond Verified Answers: Solver-Informed Self-Distillation for Bootstrapping Operations Research Language Models

    [https://arxiv.org/abs/2609.09957](https://arxiv.org/abs/2609.09957)

    该论文提出利用模型自身生成解所触发的求解器产物反馈作为监督信号的自蒸馏方法，摆脱对人工验证答案、额外评估器和特权上下文的依赖，实现运筹学语言模型的可扩展自举训练。

    

    现代大语言模型（LLM）能够将自然语言描述转化为运筹学（OR）模型表述。包括强化学习和同策略自蒸馏在内的后训练技术进一步提升了这一能力。然而，在训练面向运筹学建模的大语言模型时仍存在三个局限。第一，训练通常依赖由人类专家或更强模型验证的合成模型表述，限制了监督信号的可扩展性。第二，信用分配要么粗糙要么代价高昂：结果奖励对整条轨迹进行评分，却无法定位责任所在的具体建模决策，而过程级监督则需要额外的评估器。第三，特权自蒸馏会因使用了部署时不可用的求解器上下文而引入风格不匹配。我们发现，模型能够从其自身生成解所触发的求解器产物反馈中获得改进，使自蒸馏成为一种实用且无需评估器的监督来源。

    arXiv:2609.09957v1 Announce Type: cross  Abstract: Modern large language models (LLMs) can translate natural-language descriptions into operations research (OR) formulations. Post-training techniques including reinforcement learning and on-policy self-distillation have further improved this capability. However, three limitations remain in training LLMs for OR formulations. First, training commonly relies on synthetic formulations validated by human experts or stronger models, constraining scalable supervision. Second, credit assignment is either coarse or costly: outcome rewards score an entire trajectory without locating the responsible modeling decision, whereas process-level supervision requires an additional evaluator. Third, privileged self-distillation can induce style mismatch by using solver context unavailable at deployment. We find that a model can improve from solver-artifact feedback generated by its own rollouts, making self-distillation a practical, evaluator-free source 
    
[^173]: 面向抓举任务的紧凑型视触觉世界模型：预测、奖励对齐与力约束

    Compact Visuotactile World Models for Lifting: Prediction, Reward Alignment, and Force Constraints

    [https://arxiv.org/abs/2609.09597](https://arxiv.org/abs/2609.09597)

    该研究构建了一个仅65万参数的紧凑型视触觉世界模型，发现准确的触觉预测本身并不能保证力约束控制的提升，但模型辅助力反馈能将同分布任务成功率从73.3%提升至93.3%，而想象强化学习表现反而不如反应式隐式Q学习。

    

    准确的触觉预测未必能改善力约束控制。我们研究了一个包含652,157个参数的动作条件视触觉世界模型，并配合匹配的行为克隆、想象空间中的策略学习、独立的反应式隐式Q学习（IQL）以及模型辅助的力反馈进行对比。在固定协议下，我们在120个涵盖几何形状和物理参数变化的新建MuJoCo环境上执行34个策略，并在另外12个同分布（ID）环境上进行324条独立重放的动作分支测试。视触觉动力学将力作用效果的平均绝对误差（MAE）从持续性基线的0.413 N降低至0.338 N。模型辅助反馈将同分布力预算下的成功率从73.3%提升至93.3%，配对差异为+20.0 [+6.7, +33.4]个百分点（95%置信区间），且该差异主要发生在脚本化下降阶段；其汇总差异为+3.9 [-4.5, +11.7]个百分点。想象强化学习实现了11.9%的汇总联合成功率，而反应式IQL为25.0%。一项经验性触觉残差压力测……（摘要在此处截断）

    arXiv:2609.09597v2 Announce Type: cross  Abstract: Accurate tactile forecasts need not improve force-constrained control. We study a 652,157-parameter action-conditioned visuotactile world model with matched behavior cloning, policy learning in imagination, independent reactive implicit Q-learning, and model-assisted force feedback. A fixed protocol executes 34 policies on 120 fresh MuJoCo environments spanning geometry and physical-parameter shifts, plus 324 independently replayed action branches on 12 additional ID environments. Visuotactile dynamics reduce force action-effect MAE from 0.413 N for persistence to 0.338 N. Model-assisted feedback raises ID force-budgeted success from 73.3% to 93.3%, with paired difference +20.0 [+6.7,+33.4] percentage points (95% CI), with the difference occurring during scripted lowering. Its pooled difference is +3.9 [-4.5,+11.7] points. Imagined RL achieves 11.9% pooled joint success versus 25.0% for reactive IQL. An empirical tactile-residual stres
    
[^174]: Edu-QuRating：基于蒸馏成对判断的多维度教育数据筛选

    Edu-QuRating: Multi-Dimensional Educational Data Curation with Distilled Pairwise Judgements

    [https://arxiv.org/abs/2609.09425](https://arxiv.org/abs/2609.09425)

    Edu-QuRating提出了一种多维度教育数据筛选流水线，通过定义教育专用评分标准、利用LLM评判器标注文档对并将成对偏好蒸馏为可复用的评分模型，从而突破传统单一标量式教育价值评估的局限，从准确性、吸引力、结构性和受众适用性等多个维度对文本进行精细化评分。

    

    教育数据过滤器已成为改进语言模型预训练的实用方法，但大多数过滤器将教育价值视为单一的标量属性。这对于某些应用来说可能过于宽泛，尤其是当数据集本身已经具有高密度的教育材料时。有用的学习材料需要准确、有吸引力、结构良好，并且适合目标受众和应用场景（例如面向学习者还是面向教师）。延续QuRating（Wettig等人，2024）的工作，我们提出了Edu-QuRating：一个用于多维度教育数据评分与筛选的流水线。Edu-QuRating定义了教育专用的评分标准（rubrics），使用LLM评判器对采样的文档对进行标注，并将这些成对偏好蒸馏为可复用的Edu-QuRater模型，这些模型可以根据一组教育标准对单个文本片段进行评分。在两个序列分类基础模型和六个教育标准的实验中，最好的Edu-QuRater能够恢复保留的（摘要在此处被截断）

    arXiv:2609.09425v1 Announce Type: new  Abstract: Educational data filters have become a practical way to improve language-model pre-training, but most filters treat educational value as a single scalar property. This may be too broad for some applications, especially if the data set already features a high density of educational material. Useful learning material needs to be accurate, engaging, well structured, and appropriate for the intended audience and application (e.g. learner- vs teacher-facing). Following QuRating (Wettig et al. 2024), we introduce Edu-QuRating: a pipeline for multi-dimensional educational data scoring and curation. Edu-QuRating defines education-specific rubrics, uses an LLM judge to label sampled document pairs and distills those pairwise preferences into reusable Edu-QuRaters, which can score individual text chunks on a set of educational criteria. Across two sequence-classification base models and six educational criteria, the best Edu-QuRater recovers held-
    
[^175]: GoAnt：面向市场微观结构数据中Alpha因子发现的质量-多样性多智能体搜索

    GoAnt: Quality-Diversity Multi-Agent Search for Alpha Factor Discovery in Market Microstructure Data

    [https://arxiv.org/abs/2609.08719](https://arxiv.org/abs/2609.08719)

    提出GoAnt框架，借助共享“心智地图”与“蚁后”调度器的多智能体质量-多样性搜索，在固定评估预算下发现计入执行成本后依然稳健有效的Alpha因子。

    

    自动化的Alpha因子发现是在固定评估预算下，从价格-成交量面板数据与订单簿数据中搜索符号化交易信号。现有的单智能体和多智能体程序搜索系统容易过拟合那些在计入执行成本后即失效的预测代理指标，并反复探索冗余的因子家族，从而限制了执行稳健性与行为多样性。我们提出GoAnt，一个质量-多样性多智能体搜索框架，它将互不通信的探索者、利用者和连接者三类工作智能体，与一个共享的自适应“心智地图”以及一个紧凑的“蚁后”调度器相结合。心智地图依据无泄漏的执行特征来组织候选因子，并在每个生态位中仅保留一个精英候选；蚁后则根据显式的搜索状态摘要重新分配评估预算。我们还定义了一种独立于地图的“有效产出”评估协议，直接从每种方法的评估记录中统计高质量且相互之间无冗余的因子，为基于档案的评估……（原文此处截断）

    arXiv:2609.08719v1 Announce Type: new  Abstract: Automated alpha factor discovery searches symbolic trading signals from price-volume panels and order-book data under a fixed evaluation budget. Existing single- and multi-agent program-search systems can overfit predictive proxies that fail after execution costs and repeatedly explore redundant factor families, limiting execution robustness and behavioral diversity. We introduce GoAnt, a quality-diversity multi-agent search framework that combines non-communicating Explorer, Exploiter and Connector workers with a shared adaptive Mental Map and a compact Queen dispatcher. The Mental Map organizes candidates by leakage-free execution profiles and retains one elite per niche, while the Queen reallocates the evaluation budget from explicit search-state summaries. We also define a map-independent effective-yield protocol that counts high-quality, mutually nonredundant factors directly from each method's evaluation records, giving archive-bas
    
[^176]: FastE：面向大语言模型嵌入推理的读出触发式Token压缩方法

    FastE: Readout-Triggered Token Compression for LLM Embedding Inference

    [https://arxiv.org/abs/2609.08407](https://arxiv.org/abs/2609.08407)

    提出了无需训练、即插即用的FastE方法，利用前缀状态随深度增加而愈发可压缩的发现，通过读出位置注意力分数在线筛选并压缩前缀token，大幅降低LLM嵌入推理的计算成本。

    

    在本研究中，我们识别了最终读出型LLM嵌入模型中随深度变化的前缀冗余现象，该现象在Qwen3-Embedding和Qwen3-VL-Embedding等代表性骨干模型中均有体现。我们发现，在浅层中移除前缀状态比在深层中移除造成的损害要大得多，这表明随着前缀状态和读出状态在网络中的传播，前缀状态变得越来越可压缩。为此，我们提出了FastE——一种无需训练、即插即用的方法。FastE采用基于批均值读出-前缀对齐度的共享固定阈值作为轻量级在线启发式策略，用于决定何时触发压缩，并根据前缀状态从读出位置获得的注意力分数对其进行排序，从而确定哪些状态将在后续层中被保留。我们的评估表明，FastE能够大幅降低计算成本：在NarrativeQA数据集上使用Qwen3-Embedding-0.6B时，它显著减少了解码器骨干的浮点运算量。

    arXiv:2609.08407v1 Announce Type: new  Abstract: In this study, we identify depth-dependent prefix redundancy in final-readout LLM embedding models, notably across representative backbones including Qwen3-Embedding and Qwen3-VL-Embedding. We find that removing prefix states is substantially more damaging in shallow layers than at greater depth, showing that prefix states become increasingly compressible as the prefix and readout states propagate through the network. To this end, we introduce FastE, a training-free, plug-and-play method. FastE uses a shared fixed threshold on batch-mean readout-prefix alignment as a lightweight online heuristic for selecting when compression occurs, and ranks prefix states by the attention scores they receive from the readout position to determine which states are retained in subsequent layers. Our evaluations demonstrate FastE's ability to substantially reduce computational costs: on NarrativeQA with Qwen3-Embedding-0.6B, it reduces decoder-backbone FL
    
[^177]: 面向自动驾驶赛车目标检测与跟踪的多模态感知流水线

    A Multi-Modal Perception Pipeline for Object Detection and Tracking in Autonomous Racing

    [https://arxiv.org/abs/2609.08338](https://arxiv.org/abs/2609.08338)

    本文提出了一种多模态后融合感知流水线，通过融合摄像头、激光雷达和毫米波雷达的独立检测结果，并结合显式补偿检测延迟、嵌入车辆动力学先验知识的专用多目标跟踪框架，实现了自动驾驶赛车高速恶劣环境下对周围车辆及时且稳健的检测与跟踪。

    

    目标检测与跟踪是自动驾驶感知系统的基本组成部分。在有限能见度、传感器噪声和故障等恶劣条件下实现稳健的性能仍然是一个开放的挑战，在自动驾驶赛车领域尤其如此——车辆以极高的速度运行、经历强烈的振动，并在很小的安全余量下进行交互。本文提出了一种面向自动驾驶赛车领域的多模态后融合感知流水线，用于目标检测与跟踪。所提出的系统通过后融合方法和专用的多目标跟踪框架，利用全部车载传感器，扩展了先前的工作。来自摄像头、激光雷达和毫米波雷达的独立检测结果被融合起来，以提供对周围车辆及时且稳健的状态估计。该跟踪方法显式地补偿了检测延迟，并在其模型中嵌入了关于车辆动力学和跟踪行为的先验知识。

    arXiv:2609.08338v1 Announce Type: cross  Abstract: Object detection and tracking are fundamental components of perception systems for autonomous driving. Achieving robust performance under adverse conditions such as limited visibility, sensor noise, and failures remains an open challenge, particularly in autonomous racing, where vehicles operate at very high speeds, experience strong vibrations, and interact under small safety margins. This paper presents a multi-modal late-fusion perception pipeline for object detection and tracking in the autonomous racing domain. The proposed system extends previous work by exploiting all onboard sensors through a late-fusion approach and a dedicated multi-object tracking framework. Independent detections from cameras, LiDARs, and RADARs are combined to provide timely and robust state estimates of surrounding vehicles. The tracking method explicitly compensates for detection delays and embeds in its model prior knowledge of vehicle dynamics and trac
    
[^178]: 你的智能体说“可以”：解读超越单笔交易的对抗性市场行为

    Your Agent Says Yes: Interpreting Adversarial Market Behavior Beyond Individual Transactions

    [https://arxiv.org/abs/2609.07675](https://arxiv.org/abs/2609.07675)

    本文通过十个角色化的语言模型智能体在虚拟交易所中的对抗性模拟，揭示了“发行—推广—退出”类市场操纵行为如何分散于消息、交易与时间之中，从而暴露了仅依赖单笔交易审核的风控机制的盲区。

    

    交易级别的控制只能回答某一笔金融请求是否可以执行，但市场行为可能分布在消息、智能体、资产和时间之中。我们在一个由十个角色条件化的语言模型智能体组成的虚拟交易所中研究这一解释鸿沟。这些智能体在规定性的对抗角色下进行通信、交易参考资产和期货、发行代币并管理集中流动性池。我们分析了八条72周期的轨迹，涵盖两条时间盲化的每小时回放路径，其中运行端钱包策略分别处于启用或禁用状态。所保留的记录将生成的外发消息、策略事件、余额、持仓与周期末市场状态关联起来。一项重点重构展示了一个“发行—推广—退出”场景，该场景通过私下协调、公开宣称、跟随者建仓、多次被搁置的退出，以及随后与代币余额变化一致的非阻塞请求得以实现。在启用策略的运行中……

    arXiv:2609.07675v1 Announce Type: cross  Abstract: Transaction-local controls answer whether one financial request may proceed, but market behavior can be distributed across messages, agents, assets, and time. We study this interpretation gap in a virtual exchange populated by ten role-conditioned language-model agents. The agents communicate, trade reference assets and futures, launch tokens, and manage concentrated-liquidity pools under prescriptive adversarial roles. We analyze eight 72-cycle trajectories across two time-blinded hourly replay paths, with a runner-side wallet policy enabled or disabled. The retained artifacts connect generated outgoing messages, policy events, balances, positions, and cycle-end market state. A focal reconstruction shows a launch--promotion--exit scenario realized across private coordination, public claims, follower positioning, repeatedly withheld exits, and a later non-blocking request aligned with a token balance change. Across policy-enabled runs,
    
[^179]: 超越单负例偏好：面向以大语言模型为核心的历史实体链接的多负例DPO

    Beyond Single-Negative Preference: Multi-Negative DPO for LLM-Centric Historical Entity Linking

    [https://arxiv.org/abs/2609.07379](https://arxiv.org/abs/2609.07379)

    提出多负例直接偏好优化方法（MDPO），通过将正确实体与完整候选集进行比较来改进基于大语言模型的历史实体链接，在多语言历史报纸文本上超越监督微调和单负例DPO，尤其在NIL提及、语义歧义和OCR噪声场景下收益显著。

    

    大语言模型（LLMs）最近在历史实体链接任务中展现出潜力，但针对该任务的偏好优化通常在每次训练实例中仅使用一个负例候选，这丢弃了为同一提及检索到的其余候选的信息。我们提出了多负例直接偏好优化（MDPO），这是一种基于参考的成对目标函数，将正确实体与每个提及所关联的所有有效被拒绝候选进行比较。MDPO在保留DPO的Bradley-Terry公式的同时，通过掩码化、长度归一化的序列得分来利用完整的候选集。我们在hipe-2020和newseye数据集上评估了MDPO，涵盖法语、德语、英语、瑞典语和芬兰语的历史报纸文本。实验表明，MDPO优于监督微调和单负例DPO，在NIL提及、语义歧义、OCR噪声和历史语言等具有挑战性的场景中取得了尤为显著的提升。

    arXiv:2609.07379v1 Announce Type: cross  Abstract: Large language models (LLMs) have recently shown promise for historical entity linking, but preference optimization for this task is often formulated with only one negative candidate per training instance. This discards information from the remaining candidates retrieved for the same mention. We introduce multi-negative direct preference optimisation (MDPO), a reference-based pairwise objective that compares the correct entity with all valid rejected candidates associated with each mention. MDPO preserves the Bradley-Terry formulation of DPO while exploiting the complete candidate set through masked, length-normalised sequence scores. We evaluate MDPO on hipe-2020 and newseye, covering French, German, English, Swedish, and Finnish historical newspaper text. Experiments show that MDPO improves over supervised fine-tuning and single-negative DPO, with particularly strong gains for NIL mentions, semantic ambiguity, OCR noise, and historic
    
[^180]: HOL 中的一元二阶逻辑：深浅嵌入与自动化忠实性（扩展预印本）

    Monadic Second-Order Logic in HOL: Deep and Shallow with Automated Faithfulness (Extended Preprint)

    [https://arxiv.org/abs/2609.07345](https://arxiv.org/abs/2609.07345)

    本文在 Isabelle/HOL 中将深浅嵌入方法应用于一元二阶逻辑（MSO），借助新型双排序替换装置自动化验证了三种嵌入的忠实性，并首次完全机械化证明了双排序向下 Löwenheim–Skolem 定理。

    

    在 Isabelle/HOL 中，我们将先前工作中提出的深浅嵌入方法应用于一元二阶逻辑（MSO）。我们并排开发了三种嵌入：一种深层嵌入（带有显式满足关系的归纳数据类型）；一种最大浅层嵌入，它将逻辑连接词和量词直接翻译为 HOL 表达式，并将解释及两个赋值作为显式参数携带；以及一种最小浅层嵌入——一个固定上述参数的 locale，从而将公式类型坍缩为 bool。关键的使能性新要素是一套双排序替换装置——按命名空间分别定义的避免捕获替换、重命名及替换引理——其中每个绑定器对另一命名空间都是透明的。所有三种嵌入的忠实性均得到了机械化验证并实现了自动化。我们的核心贡献是一个完全机械化的双排序向下 Löwenheim–Skolem 定理：最小嵌入能够相对于 t……（原文摘要在此处截断）

    arXiv:2609.07345v1 Announce Type: cross  Abstract: In Isabelle/HOL, we apply the deep-and-shallow embedding methodology of our prior work to monadic second-order logic (MSO). Three embeddings are developed side by side: a deep embedding (an inductive datatype with an explicit satisfaction relation); a maximal-shallow embedding that translates the connectives and quantifiers directly into HOL, carrying the interpretation and both assignments as explicit arguments; and a minimal-shallow embedding -- a locale that fixes those parameters, collapsing the formula type to bool. The enabling new ingredient is a two-sorted substitution apparatus -- capture-avoiding substitution, renaming, and a substitution lemma per namespace -- in which each binder is transparent for the other; faithfulness of all three embeddings is mechanised and automated. Our central contribution is a fully mechanised two-sorted downward Loewenheim-Skolem theorem: the minimal embedding recovers deep validity relative to t
    
[^181]: Ambient @ EgoProactive 2026：基于视觉定位监督的主动式第一人称辅助

    Ambient @ EgoProactive 2026 : Proactive Egocentric Assistance with Visually Grounded Supervision

    [https://arxiv.org/abs/2609.07099](https://arxiv.org/abs/2609.07099)

    该论文将可穿戴助手的介入时机判断重新表述为单token二分类任务，并利用工具调用式视频智能体自动生成视觉定位的监督数据，最终在ECCV 2026可穿戴AI挑战赛EgoProactive赛道大型模型组中获得第一名。

    

    我们介绍了提交给ECCV 2026可穿戴AI挑战赛EgoProactive赛道的方案，该方案在大型模型组中排名第一，在小于等于2B参数组中排名第二。该任务要求可穿戴助手在每段八秒的第一人称视频片段之后，决定是进行介入还是保持沉默。我们的方法包含两个主要组成部分。首先，我们将介入时机判断重新表述为单token分类任务：模型不再生成interrupt（介入）或silent（沉默）的文本，而是预测yes或no，我们通过这两个token的重新归一化概率来得出决策。这种表述方式相比自由文本生成，使宏F1提升了0.249，G-mean提升了0.30。其次，由于标注数据仅限于公开发布的验证集，我们使用一个具备工具调用能力的视频智能体来检查每个视频片段并分配介入时间戳，从而生成额外的监督信号。一种仅依赖旁白的替代方案规模是其四倍，且……（摘要原文在此处截断）

    arXiv:2609.07099v1 Announce Type: cross  Abstract: We present our submission to the EgoProactive track of the ECCV 2026 Wearable AI Challenge, which ranked first in the large-model division and second in the <=2B division. The task requires a wearable assistant to decide after each eight-second segment of egocentric video whether to intervene or remain silent.   Our approach has two main components. First, we reformulate intervention timing as single-token classification. Rather than generating either $interrupt$ or $silent$, the model predicts yes or no, and we derive the decision from the renormalised probabilities of these two tokens. This formulation improved macro-F1 by 0.249 and G-mean by 0.30 over free-form generation. Second, because labelled data were limited to the released validation set, we generated additional supervision using a tool-calling video agent that inspects each clip and assigns intervention timestamps. A narration-only alternative was four times larger and ten 
    
[^182]: 普通而合理的聊天机器人：AI模型能否追踪人类的法律判断？

    Ordinary, Reasonable Chatbots: Do AI Models Track Human Legal Judgments?

    [https://arxiv.org/abs/2609.06769](https://arxiv.org/abs/2609.06769)

    本研究通过测试大语言模型驱动的聊天机器人对一系列法律合理性问题（即行为是否“合理”）的回答，探究生成式AI模型能否模拟人类在法律场景中的合理性判断。

    

    随着人们日益依赖人工智能（AI）为自己的生活提供指导，学者、律师甚至法官都开始考虑AI在法律决策中的作用。随着“硅基采样”——即在社会科学研究中使用生成式AI模型——正在影响学术界，“硅基陪审员”可能会出现在法庭上。本研究加入了关于生成式AI模型模拟人类法律判断能力的新兴研究领域。具体而言，我们研究了由大型语言模型（LLM）驱动的聊天机器人如何回答一系列关于法律合理性的问题。当法律需要判断某一行为是否恰当时，最常问的是该行为是否“合理”。然而，尽管合理性判断无处不在，它们却一直是律师、法官和外行人反复感到困扰的难题。合理性似乎本质上是模糊且不可预测的，因为它依赖于多变的情境。

    arXiv:2609.06769v1 Announce Type: cross  Abstract: As people increasingly rely on artificial intelligence (AI) for guidance in their own lives, scholars, lawyers, and even judges have begun to consider the role of AI in legal decision-making. As "silicon sampling" -- the use of generative AI models in social science research -- is now impacting academia, "silicon jurors" could make an appearance in courtrooms. This study joins an emerging line of research on generative AI models' ability to simulate human legal judgments. In particular, we study how large language model (LLM)-powered chatbots respond to series of questions about legal reasonableness. When the law needs to judge the appropriateness of a behavior, it most often asks whether the behavior was "reasonable." Yet despite the ubiquity of reasonableness judgments, they are the site of constant vexation for lawyers, judges, and lay people. Reasonableness seems inherently vague and unpredictable, since it relies on variable conte
    
[^183]: 通过潜空间推理！让潜空间视觉推理成为必需

    Reason Through the Latent! Making Latent Visual Reasoning Necessary

    [https://arxiv.org/abs/2609.06746](https://arxiv.org/abs/2609.06746)

    提出因果视觉循环推理（CVRR）框架，通过在解码前移除视觉状态和多模态KV缓存，迫使循环隐藏状态成为唯一的图像条件信息通路，从而确保潜空间视觉推理真正被模型依赖。

    

    潜空间视觉推理旨在通过隐藏状态计算而非显式的文本思维链来进行多模态推理。然而，视觉信息存在于潜空间状态中并不意味着模型在生成答案时真正依赖该状态，尤其是当其他基于图像的替代路径仍然可用时。我们提出了因果视觉循环推理（CVRR），该方法在保留预训练视觉能力的同时，使循环计算成为预测所必需的基于图像的条件路径。CVRR在预训练视觉语言模型融合图像之后，从问题的隐藏状态初始化循环过程，然后在重复读取相同固定视觉证据的同时反复更新该状态。在解码之前，视觉状态和原始的多模态KV缓存会被移除，从而确保只有最终的循环状态携带基于图像的条件信息。

    arXiv:2609.06746v1 Announce Type: new  Abstract: Latent visual reasoning aims to perform multimodal reasoning through hidden-state computation rather than explicit textual chains of thought. However, visual information being present in a latent state does not imply that the model actually relies on that state when producing its answer, especially when alternative image-conditioned paths remain available. We introduce \textbf{C}ausal \textbf{V}isual \textbf{R}ecurrent \textbf{R}easoning (CVRR), which preserves pretrained visual competence while making recurrent computation the required image-conditioned path to prediction. CVRR initializes recurrence from the question hidden state after the pretrained vision-language model has incorporated the image, then repeatedly updates this state while re-reading the same fixed visual evidence. Before decoding, visual states and the original multimodal KV cache are removed so that only the final recurrent state carries image-conditioned information
    
[^184]: 认证合作：一种合作式多智能体任务生成的新方法

    Certifying cooperation: a novel approach to cooperative multi-agent task generation

    [https://arxiv.org/abs/2609.06586](https://arxiv.org/abs/2609.06586)

    该论文提出利用时间合作图和命题公式编码来认证多智能体任务中合作的必要性与必然性，从而将随机布局采样器转化为能自动生成具有可证明合作要求任务的生成器。

    

    共享奖励为智能体提供了共同的目标，但并未明确它们何时、如何、甚至是否必须合作才能取得成功。我们在激光学习环境中研究这些问题，这是一个多智能体寻路环境，其中合作具体表现为一个智能体阻挡激光以让队友安全通过。我们通过时间合作图来表示这些交互，图中带有时间标记的边将帮助者与受益者连接起来，并将六种合作模式定义为相互重叠的图谓词，同时证明了每条合作轨迹至少满足其中一种模式。通过将环境动态和模式谓词编码为命题公式，我们能够区分两类任务：一类是在指定时间范围内的某些获胜轨迹中存在某种合作模式的任务，另一类是在每条获胜轨迹中都必须出现该模式的任务。作为过滤器使用时，这些查询将随机布局采样器转变为能够生成具有认证合作要求任务的生成器。实验……

    arXiv:2609.06586v1 Announce Type: cross  Abstract: A shared reward gives agents a common objective, but leaves open when, how and even whether they must cooperate to succeed. We address these questions in the Laser Learning Environment, a multi-agent path-finding environment where cooperation materializes as one agent blocking a laser to let a teammate pass safely. We represent these interactions through temporal cooperation graphs whose timed edges connect helpers to beneficiaries, define six cooperation profiles as overlapping graph predicates, and prove that every cooperative trajectory satisfies at least one. By encoding the environment dynamics and profile predicates as propositional formulae, we distinguish tasks that admit}a profile in some winning trajectory from those that require it in every winning trajectory within a specified horizon. Used as filters, these queries turn a random layout sampler into a generator of tasks with certified cooperation requirements. Experiments w
    
[^185]: 控制流不确定性下的业务流程规划与调度

    Planning and Scheduling Business Processes under Control-Flow Uncertainty

    [https://arxiv.org/abs/2609.05578](https://arxiv.org/abs/2609.05578)

    该论文将控制流不确定性下的业务流程规划与调度问题建模为机会约束优化问题，提出了一种两阶段分解方法，在保证流程成功完成可行性的前提下最小化被规划但不会执行的冗余活动数量。

    

    在业务流程中对活动进行调度可以提高效率（例如缩短完工时间），但这具有挑战性，因为完成一个案例所需的确切活动顺序往往是不确定的，这是由于执行过程中基于数据所做出的决策导致的。尽管如此，关于这些决策的概率信息通常可以从历史执行日志中估计或推导得出，这些信息有助于预测哪些执行路径更有可能成功完成。基于特定执行路径的规划会影响可行性（即成功完成的概率），以及被规划但从未执行的冗余活动的期望数量。我们将该问题表述为一个机会约束优化问题，并提出了两种公式化方法：一种分两阶段的分解方法，其中规划阶段在满足可行性约束的条件下最小化冗余活动的期望数量，调度阶段……

    arXiv:2609.05578v1 Announce Type: new  Abstract: Scheduling activities in business processes can improve efficiency (e.g., reduce makespan), but is challenging because the exact sequence of activities required to complete a case is often uncertain due to decisions based on data that emerges during execution. Nevertheless, probabilistic information regarding such decisions can often be estimated or derived from historical execution logs, and can help anticipate which execution paths are likely to lead to successful completion. Planning with particular execution paths affects feasibility, i.e., the probability of successful completion, and the expected number of superfluous activities that are planned but never executed. We frame the problem as a chance-constrained optimization problem and present two formulations: A decomposed approach with two stages, a planning stage that minimizes the expected number of superfluous activities subject to a feasibility constraint, and a scheduling stag
    
[^186]: 角色分化作为集体信息引擎的点火器：智能体群体中的结构化理论

    Role differentiation as ignition of a collective information engine: Structuration in Agent Populations

    [https://arxiv.org/abs/2609.05442](https://arxiv.org/abs/2609.05442)

    该研究设计了一种基于角色分化而非共识的集体信息引擎，通过反协调博弈将结构化理论中“图式与资源二元性”操作化，并揭示了当社会回路增益超过阈值时角色分化便会自发“点燃”。

    

    信息活性物质展示了基于测量信息的决策如何产生集体秩序，但迄今为止仅在达成共识的系统中得到研究。我们转而设计了以分化而非共识来构建结构的集体信息引擎，并利用反协调博弈构建了一个最小实例，其中分化的角色信息具有价值。在多个共存的博弈中，智能体从基于持久身份的嘈杂社会信号中推断自己的角色，而遵循角色的行动又反馈到该信号中，从而塑造遵循角色的激励。通过协调的角色扮演所积累的资源与身份的可变性相结合，强化了产生这些资源的图式。该模型由此将Sewell在结构化理论中提出的图式与资源的二元性付诸操作化，为社会科学中关于结构与能动性的辩论提供了一种解决方案。当社会回路增益——即身份持久性、认知能力、信道保真度（等因素）的乘积——达到阈值时，该引擎即被点燃。

    arXiv:2609.05442v2 Announce Type: cross  Abstract: Informational active matter shows how measurement-informed decisions produce collective order, so far in systems that reach consensus. We design collective information engines structured by differentiation instead, and construct a minimal instance using anti-coordination games where differentiated role information has value. Within many coexisting games, agents infer their role from a noisy social signal grounded in a persistent identity, and role-following action feeds back into that signal, which shapes the incentive to follow roles. Resources accrued through coordinated role-play combine with identity variability to reinforce the schemas that generated them. The model thereby operationalizes Sewell's duality of schemas and resources in Structuration, a resolution to structure--agency debates across social science. The engine ignites when a social loop gain---the product of identity persistence, cognitive capacity, channel fidelity, 
    
[^187]: HarvestBench：衡量大语言模型智能体是否愿意花钱避免杀死动物

    HarvestBench: Measuring Whether LLM Agents Will Pay to Avoid Killing Animals

    [https://arxiv.org/abs/2609.04444](https://arxiv.org/abs/2609.04444)

    HarvestBench 是首个为“避免杀死动物”这一副作用定价的基准，通过农场收割模拟测试大语言模型智能体是否愿意支付燃料代价绕开挡路的动物而非直接碾压，并以岩石和干草捆作为对照组、以是否偷取邻居作物作为附加的道德测试。

    

    arXiv:2609.04444v1 公告类型：new 摘要：衡量智能体在完成目标过程中所产生副作用的基准已经存在，但 HarvestBench 是首个为避免这种副作用标定价格、并将该副作用明确指认为一个活体生命的基准。它是一个农场模拟环境：大语言模型子智能体驾驶由两台拖拉机组成的车队进行协作式玉米收割作业，田地中有动物存在。该环境是一个强化学习网格世界，每个决策都是在无记忆状态下做出的，且目标中从未提及伤害一词。当动物挡住拖拉机的行进路线时，自动驾驶会停下并询问模型：是选择不耗费任何燃料直接开过去，还是以标示的燃料价格为代价绕行。模型碾压动物的情况与两个对照组进行比较：岩石（会损坏拖拉机，所有模型碾压岩石的概率均低于1%）以及无害且无生命的干草捆。模型还可以选择从邻居的田地而非自己的田地中收取作物，这是对其道德判断的第二次测试。在九个模型和

    arXiv:2609.04444v1 Announce Type: new  Abstract: Benchmarks for the side effects an agent causes on the way to a goal already exist, but HarvestBench is the first to put a price on avoiding the side effect and to name that side effect as a living creature. It is a farm simulation: LLM sub-agents drive a crew of two tractors through a cooperative corn harvest, with animals in the field. The environment is a reinforcement learning gridworld, every decision is made without memory, and the harm is never named in the goal. When an animal blocks a tractor's route the autopilot stops and asks the model whether to drive on, at no fuel cost, or swerve around it for a posted fuel price. Kills are compared against two controls: rocks, which damage the tractor and are hit under 1% of the time by every model, and hay bales, which are harmless and not alive. Models can also take crops from the neighbor's field instead of their own, a second test of what they treat as moral.   Across nine models and 
    
[^188]: Harbor 适配器与 Harbor-Index：面向大规模智能体评估的基础设施与精选元数据集

    Harbor Adapters and Harbor-Index: Infrastructure and a Curated Meta-Dataset for Large-Scale Agentic Evaluation

    [https://arxiv.org/abs/2609.04298](https://arxiv.org/abs/2609.04298)

    本文提出了 Harbor Adapters 统一评估基础设施，将 80 多个智能体基准测试移植为可评估任意智能体的形式，并据此对 8 个模型进行大规模评估，同时推出经 AI 与人工双重审核筛选的包含 82 个高质量难题的精选元数据集 Harbor-Index。

    

    在数量不断增长的智能体（agentic）基准测试上评估智能体是一项挑战，因为这些基准通常需要复杂的环境和智能体集成方案。我们提出了 Harbor Adapters，一个面向智能体基准测试的统一评估基础设施。我们的工作包含三项贡献。第一，我们开发了一系列基准适配器，将 80 多个基准测试移植为可评估任意智能体的形式，并通过严格的代码审查和一致性实验对其进行了验证。第二，我们在 54 个基准测试上对横跨不同能力层级的 8 个模型进行了大规模评估；每个模型均使用 Terminus-2 以及 3 个原生测试框架之一运行。这使得对智能体能力和失败模式的更广泛分析成为可能。第三，我们推出了 Harbor-Index，一个精心策划的包含 82 个高难度、多样化且高质量任务的集合，覆盖 29 个基准测试，它是在适配后的基准套件基础上，通过难度筛选、AI 与人工审核以及审核-修复循环精炼而成。

    arXiv:2609.04298v1 Announce Type: new  Abstract: Evaluating agents on the growing number of agentic benchmarks is challenging because they often require complex environments and agent integrations. We introduce Harbor Adapters, a unified evaluation infrastructure for agentic benchmarks. Our work makes three contributions. First, we develop benchmark adapters that port more than 80 benchmarks to evaluate arbitrary agents, and validate them through rigorous code review and parity experiments. Second, we conduct a large-scale evaluation of 8 models spanning capability tiers across 54 benchmarks; every model is run with Terminus-2 and with one of 3 native harnesses. This enables a broader analysis of agent capabilities and failure modes than was previously possible. Third, we introduce Harbor-Index, a curated set of 82 difficult, diverse, and high-quality tasks spanning 29 benchmarks, refined from the adapted suite through difficulty filtering, AI and human audit, and an audit-and-fix loop
    
[^189]: SimSkill：一个自主精通交通仿真的终身学习AI智能体

    SimSkill: A Lifelong Learning AI Agent for Autonomous Mastery of Traffic Simulation

    [https://arxiv.org/abs/2609.03753](https://arxiv.org/abs/2609.03753)

    SimSkill是一个基于SUMO交通仿真器的自进化终身学习AI智能体，通过自主识别能力差距、生成并验证任务、将经验整合到三种记忆系统中，在不更新底层模型的情况下将经验证的任务完成率提升最多25个百分点。

    

    随着大型语言模型（LLM）能力的不断增强，AI系统的长期价值不仅取决于解决单个请求的能力，还取决于能否将经验和积累的知识转化为持久、可复用的能力。我们提出了SimSkill，一个围绕城市交通仿真平台（SUMO）构建的自进化智能体。SimSkill能够识别能力差距、生成并解决基于环境的任务、通过“行动-批评”（action-critic）循环验证解决方案，并将经验整合到情景记忆、程序性记忆和语义记忆中，而无需更新底层模型。通过自主探索，它构建了一个覆盖交通仿真完整工作流程的可复用知识库。我们在两个留出测试基准上，使用三个底层LLM以及独立的基于产物的验证方法对SimSkill进行了评估。SimSkill将经验证的任务完成率提升了最多25个百分点，消融实验表明各模块的贡献具有互补性……

    arXiv:2609.03753v1 Announce Type: new  Abstract: As large language models (LLMs) become increasingly capable, the long-term value of AI systems depends not only on solving individual requests, but also on transforming experience and accumulated knowledge into durable, reusable competence. We introduce SimSkill, a self-evolving agent built around the Simulation of Urban MObility (SUMO) traffic simulator. SimSkill identifies capability gaps, generates and solves environment-grounded tasks, verifies solutions through an action--critic loop, and consolidates experience into episodic, procedural, and semantic memory without updating the backbone model. Through autonomous exploration, it builds a reusable library spanning the traffic-simulation workflow. We evaluate SimSkill on two held-out benchmarks with three backbone LLMs and independent artifact-based verification. SimSkill improves verified completion by up to 25 percentage points, while ablations show complementary contributions from 
    
[^190]: 面向参与式民主的偏好推断：迈向以集体为中心的评估

    Toward Collective-Centric Evaluation of Preference Inference for Participatory Democracy

    [https://arxiv.org/abs/2609.02990](https://arxiv.org/abs/2609.02990)

    该论文提出了一个以集体为中心的评估框架，对参与式民主平台中现有偏好推断方法进行了基准测试，揭示了这些方法并非中立，可能人为地放大、抑制或重排集体支持模式，从而重塑对商议结果的解读。

    

    为了扩大集体决策的规模，Polis 和 Remesh 等参与式民主平台使数千名参与者能够进行在线商议。然而，在这种规模下，参与者无法审阅其他人提交的每一条意见，由此产生高度稀疏的投票数据，这些数据无法准确反映共识、冲突以及少数派支持的模式。因此，平台日益依赖偏好推断模型来预测缺失的投票。然而，这种自动化并非中立：推断出的偏好可能人为地放大、抑制或重排现有的支持模式，最终重塑对商议结果的解读方式。更广泛地说，我们对现有偏好推断方法如何影响集体偏好格局尚缺乏系统性的理解。为填补这一空白，我们在该背景下对几种现有的偏好推断方法进行了基准测试，并超越了以个体预测准确性为中心的传统用户导向评估。

    arXiv:2609.02990v1 Announce Type: cross  Abstract: To scale up collective decision-making, participatory democracy platforms such as Polis and Remesh enable online deliberation among thousands of participants. However, at this scale, participants cannot review every opinion submitted by others, producing highly sparse voting data that misrepresent patterns of consensus, conflict, and minority support. Platforms therefore increasingly rely on Preference Inference (PI) models to predict missing votes. Yet this automation is not neutral: inferred preferences can artificially amplify, suppress, or reorder existing patterns of support, ultimately reshaping how the outcomes of a deliberation are interpreted. More generally, we lack a systematic understanding of how existing PI methods affect the collective preference landscape. To address this gap, we benchmark several existing PI approaches in this context. Moving beyond conventional user-centric evaluations centered on the accuracy of indi
    
[^191]: OmegaUse-SOP：从人类演示中实现专业计算机使用的SOP工程

    OmegaUse-SOP: SOP Engineering for Professional Computer Use from Human Demonstrations

    [https://arxiv.org/abs/2609.02149](https://arxiv.org/abs/2609.02149)

    提出了OmegaUse-SOP系统，通过人在回路的SOP工程方法，将专业计算机操作的人类演示迭代式地转化为GUI智能体可复用的SOP技能，从而解决了智能体执行特定领域专业标准操作程序的难题。

    

    大型语言模型（LLMs）正日益从对话式助手演变为能够操作外部数字环境的智能体。图形用户界面（GUI）智能体在这一转变中发挥着重要作用，因为许多现实世界的工作流程仍然只能通过面向用户的软件界面来访问。然而，尽管近期在通用计算机使用基准测试上取得了进展，但特定领域的专业标准操作程序（SOP）对GUI智能体而言仍然充满挑战，因为它们通常涉及隐性的领域知识、软件特定的操作惯例以及任务级别的验证要求。我们提出了OmegaUse-SOP，这是一个人在回路（human-in-the-loop）的SOP工程系统，用于将专业计算机使用的人类演示转化为GUI智能体可复用的SOP技能。类似于提示工程（prompt engineering），SOP工程通过迭代式地精炼演示内容、执行规则和领域知识，将专业SOP

    arXiv:2609.02149v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly evolving from conversational assistants into agents capable of operating external digital environments. Graphical user interface (GUI) agents play an important role in this transition, as many real-world workflows remain accessible only through user-facing software interfaces. However, despite recent progress on general computer-use benchmarks, domain-specific professional standard operating procedures (SOPs) remain challenging for GUI agents because they often involve implicit domain knowledge, software-specific conventions, and task-level verification requirements. We introduce OmegaUse-SOP, a human-in-the-loop SOP Engineering system for transforming human demonstrations of professional computer use into reusable SOP skills for GUI agents. Analogous to prompt engineering, SOP Engineering iteratively refines demonstrations, execution rules, and domain knowledge to convert professional SOPs
    
[^192]: 学习保留什么：面向多智能体大语言模型系统高效协作的门控记忆路由

    Learning What to Retain: Gated-Memory Routing for Efficient Collaboration in Multi-Agent LLM Systems

    [https://arxiv.org/abs/2609.00237](https://arxiv.org/abs/2609.00237)

    提出门控记忆路由方法，通过可学习的记忆写入门和检索门维护紧凑的执行记忆，使多智能体LLM系统的编排决策能依据有用的中间进展而非完整历史，在提升准确性的同时降低成本。

    

    基于大语言模型（LLM）的多智能体系统通过编排多个智能体的配置方式和协作方式来解决复杂推理任务。一个核心挑战是使编排能够适应不断演变的协作状态。仅基于查询的路由无法适应中间过程的进展或错误，从而损害准确性；而基于完整执行历史的路由虽然补足了缺失的上下文，却迫使后续决策处理所有先前步骤，包括冗余或低效用的步骤，造成执行历史过载并推高成本。有效的编排实际上需要一个紧凑的状态，既能捕获有用的进展，又不会积累冗余上下文。我们提出门控记忆路由，将每个决策基于查询和一个学习到的执行记忆进行条件化。一个学习到的记忆写入门仅提交非冗余的推理步骤，一个学习到的检索门为每个智能体提供紧凑且相关的信息。

    arXiv:2609.00237v1 Announce Type: new  Abstract: Large language model (LLM)-based multi-agent systems tackle complex reasoning by orchestrating how multiple agents are configured and how they collaborate. A central challenge is to adapt orchestration to the evolving collaboration state. Routing from the query alone cannot adapt to intermediate progress or errors, which hurts accuracy. Routing from the complete execution history supplies this missing context, but forces later decisions to process every prior step, including redundant or low-utility ones. This creates an execution-history overload that inflates cost. Effective orchestration instead requires a compact state that captures useful progress without accumulating redundant context. We propose Gated-Memory Routing, which conditions each decision on the query and a learned execution memory. A learned Memory Write Gate commits only non-redundant reasoning steps, and a learned Retrieval Gate supplies each agent a compact, relevant 
    
[^193]: REAL-Q：基于动态梯度下降的大语言模型端到端量化

    REAL-Q: E2E LLM Quantization via Dynamic Gradient Descent

    [https://arxiv.org/abs/2609.00049](https://arxiv.org/abs/2609.00049)

    REAL-Q提出了一种打破传统折中的后训练量化新范式，通过端到端对齐的代理损失目标和每128列一次的动态块级梯度下降，解决了现有方法中Hessian矩阵被整层冻结导致的信息错位问题，从而更精确地逼近全局损失实现大语言模型量化。

    

    后训练量化（PTQ）是在严格资源约束下部署大语言模型（LLM）的关键技术。当前最先进的PTQ方法使用单一的闭式二阶求解器对每一层进行量化：为了保持解析上的可处理性，这些方法对全局损失进行了大量近似（舍弃跨通道耦合、将输出行池化为组），随后在整个层内冻结所得的Hessian矩阵，无法随着损失景观逐列变化而对其进行更新——我们将这种现象称为信息错位（information misalignment）。我们提出REAL-Q（Real-time E2E-loss Aligned LLM Quantization，实时端到端损失对齐的大语言模型量化），这是一种新颖的PTQ范式，打破了这一折中：REAL-Q不再为了解析可处理性而稀释目标函数，而是针对全局损失的端到端对齐代理目标，并在每处理一个列块（128列）后应用细粒度、动态的块级梯度下降对其进行优化。通过耦合这种细粒度……

    arXiv:2609.00049v1 Announce Type: cross  Abstract: Post-training quantization (PTQ) is essential for deploying large language models (LLMs) under strict resource constraints. State-of-the-art PTQ methods quantize each layer with a single closed-form second-order solver: to remain analytically tractable, they heavily approximate the global loss (dropping cross-channel coupling, pooling output rows into groups), and they then freeze the resulting Hessian across the entire layer, with no way to refresh it as the loss landscape shifts column by column--a phenomenon we call information misalignment. We propose REAL-Q (Real-time E2E-loss Aligned LLM Quantization), a novel PTQ paradigm that breaks this compromise: instead of diluting the objective for the sake of analytic tractability, REAL-Q targets an end-to-end-aligned surrogate of the global loss and refines it via fine-grained, dynamic Block-wise Gradient Descent applied after every column block (128 columns). By coupling this fine-grain
    
[^194]: Motus2：一个用于灵巧操作的自我进化通用世界模型

    Motus2: A Self-Evolving General World Model for Dexterous Manipulation

    [https://arxiv.org/abs/2608.30237](https://arxiv.org/abs/2608.30237)

    Motus2 提出了一个自我进化的通用世界模型，让单一共享权重模型同时充当策略、模拟器和评估器三种角色，通过三者耦合形成闭环决策与学习回路，实现灵巧操作策略的自我持续改进。

    

    通用具身智能体应当在一个统一系统中实现感知、预测、行动、评估与改进。世界模型在构建此类智能体方面展现出巨大前景，然而现有模型通常只是将一个动作输出头附加到世界模拟器上，而没有将二者耦合为一个用于策略改进的闭环决策与学习回路。我们提出 Motus2，一个用于灵巧操作的自我进化通用世界模型。Motus2 通过模型扩展和数据扩展来推进世界建模。在模型扩展方面，一个共享权重的单一模型提供三种控制接口：策略（世界-动作模型）、模拟器（动作条件世界模型）和评估器（价值模型）。策略提出候选动作块，模拟器预测其视觉后果，评估器对预测结果进行评估。三者的耦合形成了一个用于策略改进的闭环决策与学习回路。该框架使用精选的专家数（原文摘要在此处截断）

    arXiv:2608.30237v1 Announce Type: cross  Abstract: General embodied agents should perceive, predict, act, evaluate, and improve within a unified system. World models have shown great promise in building such agents, yet existing models typically append an action output head to a world simulator, without coupling them into a closed decision-and-learning loop for policy improvement. We present Motus2, a self-evolving general world model for dexterous manipulation. Motus2 advances world modeling through model scaling and data scaling. For model scaling, a single model with shared weights exposes three control interfaces: a policy (world-action model), a simulator (action-conditioned world model), and an evaluator (value model). The policy proposes candidate action chunks, the simulator predicts their visual consequences, and the evaluator assesses the predicted outcomes. Their coupling forms a closed decision-and-learning loop for policy improvement. This formulation uses curated expert d
    
[^195]: 前馈3D重建中置信度的校准审计

    A Calibration Audit of Confidence in Feed-Forward 3D Reconstruction

    [https://arxiv.org/abs/2608.29705](https://arxiv.org/abs/2608.29705)

    本文首次系统审计了前馈3D重建模型的逐像素置信度，发现其虽能较好地对误差排序，但作为不确定性估计在偏离训练条件时严重偏低（中位数偏差达2.4倍），且即使损失函数达到最优该问题依然存在。

    

    前馈3D重建模型会输出逐像素的置信度，下游系统将其解读为可靠性信号。该置信度是作为损失权重来训练的，而非作为不确定性量级来训练的，它能否被用作误差预测尚未被测量过。我们在13个数据集上对7个已发布的骨干模型进行审计，并从四个属性对置信度进行评分：它对误差排序的好坏、其量级在平均意义上是否正确、它在整个置信度范围内是否保持一致，以及其区间是否覆盖真值。结果表明，置信度对误差的排序效果良好，但当模型在与训练条件不完全相同的条件下使用时，预测的不确定性过低。在所有7个模型中，中位数情形的偏差达2.4倍，且模型越自信，误差预测的偏差越大。我们证明，即使损失函数达到最优，这一现象也可能出现。一个已发布的模型在其自身损失下继续训练时，在其训练数据上达到了该最优。

    arXiv:2608.29705v1 Announce Type: cross  Abstract: Feed-forward 3D reconstruction models emit a per-pixel confidence that downstream systems read as a reliability signal. It is trained as a loss weight, not as an uncertainty magnitude, and whether it can be used as an error prediction has not been measured. We audit seven released backbones on thirteen datasets and score the confidence on four properties, how well it ranks error, whether its level is right on average, whether it holds across the confidence range, and whether its intervals cover the truth. The confidence ranks error well, but the predicted uncertainty is too low when it is read under conditions that are not exactly those of training. The median case is off by 2.4x across all seven models, and the error prediction is further off the more confident the model is. We show that this phenomenon can appear even though the loss's optimum is reached. A released model resumed under its own loss reaches that optimum on its trainin
    
[^196]: 适当的评分规则如何塑造大语言模型预测

    How Proper Scoring Rules Shape LLM Forecasting

    [https://arxiv.org/abs/2608.28482](https://arxiv.org/abs/2608.28482)

    尽管五种适当的评分规则在理论上都激励真实概率报告，但作为大语言模型预测的训练目标时，会训练出在校准、偏差、信息和噪声特征上各不相同的模型，表明奖励函数的选择并非可以互换。

    

    本文评估了奖励函数的选择如何塑造大语言模型预测器的性能与行为。我们比较了五种适当的评分规则作为训练目标，用于对已有结果的现实世界事件进行二元预测。尽管这些规则在理论上都具有激励真实概率报告的相同特性，但由此训练出的模型在校准、概率使用以及偏差、信息和噪声的估计特征方面存在差异，而在总体准确率和区分度上的差异较小。使用Brier规则训练的模型具有最低的观测Brier分数和最高的AUC-ROC，而使用对数规则训练的模型具有最高的观测对数分数和最低的校准误差。总体性能相似的模型也是通过偏差、信息和噪声的不同组合来达到该性能的。因此，适当的评分规则作为训练目标时并不必然可以互换。奖励的选择不仅可能影响大语言模型预测的好坏，还可能……

    arXiv:2608.28482v1 Announce Type: new  Abstract: This paper evaluates how reward function choice shapes the performance and behavior of LLM forecasters. We compare five proper scoring rules as training objectives for binary forecasts of resolved real-world events. Although the rules share the same theoretical incentive for truthful probability reporting, the resulting models differ in calibration, probability use, and estimated profiles of bias, information, and noise, with smaller differences in aggregate accuracy and discrimination. The Brier-trained model has the lowest observed Brier score and highest AUC-ROC, while the log-trained model has the highest observed log score and lowest calibration error. Models with similar aggregate performance also reach that performance through different combinations of bias, information, and noise. Proper scoring rules therefore need not behave interchangeably as training objectives. Reward choice may shape not only how well an LLM forecasts, but 
    
[^197]: FaultLens：为生成的操作程序学习紧凑的行为测试套件

    FaultLens: Learning Compact Behavioral Test Suites for Generated Operational Programs

    [https://arxiv.org/abs/2608.26746](https://arxiv.org/abs/2608.26746)

    本文提出FaultLens方法，通过结合故障驱动的贪婪选择和突变无关的多样性组件，学习紧凑的行为测试套件，以高效检测生成程序中的稀疏边界和交互故障。

    

    arXiv:2608.26746v1 公告类型：交叉 摘要：生成的操作程序通常通过少量手写示例或全面的回归测试套件进行验证。前者可能遗漏稀疏的边界和交互故障，而后者可能不必要地昂贵。我们引入了FaultLens，一种学习紧凑行为测试套件的方法，同时保持与执行证据的可审计联系。它执行一次丰富的探针域，将故障-探针杀死关系存储为稀疏结果缓存，并仅从早期程序生成中学习探针排序。一个故障驱动的贪婪组件利用已知的杀死结构，而一个与突变无关的多样性组件覆盖探针族、案例、模板和时间箱。它们的交替混合方法在新程序包含排序构建中不存在的故障机制时仍然有用。我们评估了四个环境中的二十个生成操作策略，十个执行种子，以及1,200个测量值。

    arXiv:2608.26746v1 Announce Type: cross  Abstract: Generated operational programs are often validated with either a few hand-written examples or exhaustive regression suites. The former can miss sparse boundary and interaction faults, while the latter can be unnecessarily expensive. We introduce FaultLens, a method for learning compact behavioral test suites while preserving an auditable connection to executed evidence. It executes a rich probe domain once, stores the fault-probe kill relation as a sparse outcome cache, and learns probe orderings only from earlier program generations. A fault-driven greedy component exploits known kill structure, while a mutation-independent diversity component covers probe families, cases, templates, and temporal bins. Their alternating hybrid remains useful when a new program contains a fault mechanism absent from ordering construction.   We evaluate twenty generated operational policies across four environments, ten execution seeds, 1,200 measured r
    
[^198]: GameWAM：视频游戏的世界动作模型

    GameWAM: A World Action Model for Video Games

    [https://arxiv.org/abs/2608.26200](https://arxiv.org/abs/2608.26200)

    GameWAM是首个将世界模型与动作策略统一用于视频游戏原生闭环控制和GUI操作的世界动作模型，通过并行生成视觉与动作轨迹实现动态建模。

    

    摘要：arXiv:2608.26200v1 公告类型：新 摘要：现代视频游戏结合了第一人称感知、快速视觉变化、持久世界状态和异构原生控制。现有游戏代理直接将视觉和任务上下文映射到动作，但缺乏显式的世界动态建模，而交互式游戏世界模型根据提供的动作预测视觉未来，但不作为任务策略。世界动作模型（WAMs）统一了这些目标，但在视频游戏的动态和开放式交互下仍未被充分探索。我们引入了GameWAM，据我们所知，这是首个用于原生闭环游戏玩法和GUI控制的WAM。GameWAM通过并行视觉和动作生成过程，结合块因果条件和流匹配，联合生成未来视觉观察和可执行的键盘-鼠标轨迹。为了支持联合世界动作学习，我们构建了同步的游戏玩法和GUI轨迹。为了处理异构原生控制，游戏

    arXiv:2608.26200v1 Announce Type: new  Abstract: Modern video games combine first-person perception, rapid visual changes, persistent world state, and heterogeneous native controls. Existing game agents map visual and task context directly to actions but lack explicit world dynamics modeling, whereas interactive game world models predict visual futures from supplied actions but do not serve as task policies. World-Action Models (WAMs) unify these objectives, but remain largely unexplored under the dynamics and open-ended interaction of video games. We introduce GameWAM, to our knowledge the first WAM for native closed-loop gameplay and GUI control. GameWAM jointly generates future visual observations and executable keyboard-mouse trajectories through parallel visual and action generative processes with block-causal conditioning and flow matching. To support joint world-action learning, we construct synchronized gameplay and GUI trajectories. To handle heterogeneous native control, Game
    
[^199]: CAT-GS：通过校准门控与融合手术实现平衡的多模态学习

    CAT-GS: Balanced Multimodal Learning via Calibrated Gating and Fusion Surgery

    [https://arxiv.org/abs/2608.24947](https://arxiv.org/abs/2608.24947)

    本文提出了一种名为CAT-GS的优化控制器，通过校准门控和融合手术机制，在不修改模型结构的情况下解决多模态学习中的模态不平衡、门控不稳定和融合干扰问题。

    

    arXiv:2608.24947v1 公告类型：新 摘要：多模态神经网络的端到端训练通常表现出不稳定的神经动力学，其特征是三种耦合的失效模式会降低学习效果：（i）模态不平衡，其中一个分支主导基于梯度的优化；（ii）门控不稳定，噪声置信度线索导致模态选择波动；（iii）融合干扰，模态特定梯度在共享融合层发生冲突。我们提出CAT-GS（校准、自适应、阈值门控与融合手术），这是一种面向智能计算应用的基于神经动力学的优化控制器。CAT-GS在反向传播期间运行，无需修改模型架构、融合模块或任务损失。通过温度缩放和EMA平滑对教师派生的可靠性进行校准，CAT-GS使用边际阈值策略稳定神经动力学，以在热身丢弃、弱模态优先级和弱偏置融合之间切换。

    arXiv:2608.24947v1 Announce Type: new  Abstract: End-to-end training of multimodal neural networks often exhibits unstable neural dynamics characterized by three coupled failure modes that degrade learning: (i) modality imbalance, where one branch dominates gradient-based optimization; (ii) unstable gating, where noisy confidence cues induce erratic modality selection; and (iii) fusion interference, where modality-specific gradients conflict at the shared fusion layer. We propose CAT-GS (Calibrated, Adaptive, Thresholded Gating with Fusion Surgery), a neural dynamics-based optimization controller for intelligent computing applications. CAT-GS operates during backpropagation without modifying model architectures, fusion modules, or task losses. Through calibration of teacher-derived reliability via temperature scaling and EMA smoothing, CAT-GS stabilizes neural dynamics using a margin-thresholded policy to switch between warm-up dropout, weak-modality prioritization, and weak-biased ble
    
[^200]: Aslema在NADI 2026：通过少样本增强进行口语语言理解

    Aslema at NADI 2026: Augmentation through Fewshot for SLU

    [https://arxiv.org/abs/2608.18689](https://arxiv.org/abs/2608.18689)

    本文提出Aslema系统，通过微调优于零样本，并利用大型语言模型生成文化相关的合成数据增强，在NADI 2026任务中在槽位填充上取得第一名。

    

    arXiv:2608.18689v1 公告类型：交叉 摘要：我们介绍了Aslema，这是我们为NADI 2026共享任务5开发的系统，该任务包含两个子任务：意图识别和槽位填充。我们在零样本设置下评估了四种全模态大型语言模型，并将它们与微调模型进行了比较。结果表明，微调始终优于零样本推理。我们进一步探索了合成数据增强，通过使用大型语言模型生成具有文化背景的突尼斯Derja话语，然后通过语音克隆生成合成语音。将这种合成数据纳入后，两个任务的性能均得到提升。我们最终提交的系统基于Qwen3-Omni-30B，并使用原始数据和合成数据的混合进行训练，在开发测试集上实现了86.8%的意图准确率和34.7的词错误率。在官方测试集上，它在槽位填充任务中排名第一（59.5 CoER），在意图识别任务中排名第四（8个团队中，准确率66.1%）。我们发布了实验脚本，并将很快共享合成数据集以支持进一步研究。

    arXiv:2608.18689v1 Announce Type: cross  Abstract: We present Aslema, our system for NADI 2026 Shared Task 5, which consists of two subtasks: intent recognition and slot filling. We evaluate four omni LLMs in a zero-shot setting and compare them with fine-tuned models. Our results show that fine-tuning consistently outperforms zero-shot inference. We further explore synthetic data augmentation by using an LLM to generate culturally grounded Tunisian Derja utterances, followed by voice cloning to generate synthetic speech. Incorporating this synthetic data improves performance on both tasks. Our final submitted system, based on Qwen3-Omni-30B and trained with a mixture of original and synthetic data, achieves 86.8% intent accuracy and 34.7 WER on the devtest split. On the official test set it ranks 1st in slot filling (59.5 CoER) and 4th among 8 teams in intent recognition (66.1% accuracy). We release our experimental scripts and will soon share the synthetic dataset to support further 
    
[^201]: 通过形式抽象提升资源受限语言模型中自然语言组合优化的准确性

    Improving Natural-Language Combinatorial-Optimization Accuracy in Resource-Constrained Language Models via Formal Abstractions

    [https://arxiv.org/abs/2608.18409](https://arxiv.org/abs/2608.18409)

    SDDL通过神经符号框架将自然语言调度问题转化为求解器友好的表示，显著提升了资源受限语言模型在组合优化中的可行性。

    

    arXiv:2608.18409v1 公告类型：新 摘要：组合调度对语言模型构成重大挑战，要求它们在满足复杂约束的同时，在指数级大的搜索空间内识别可行解决方案。这一挑战在资源受限环境中尤为突出，因为在此类环境中，大型语言模型不切实际，只能选择较小的模型，而这些模型在直接从自然语言进行调度时往往无法保持可行性。为解决这些限制，我们引入了SDDL，一个神经符号框架，它将自然语言调度问题转化为紧凑、面向求解器的任务、资源、约束和目标表示，同时将底层建模和搜索委托给确定性编译器和外部求解器。在包含300个实例、多族类的调度问题子集上，SDDL为每个测试的资源受限模型提高了独立验证的可行性。最强的两个SDDL模型...

    arXiv:2608.18409v1 Announce Type: new  Abstract: Combinatorial scheduling poses a significant challenge for language models, requiring them to identify feasible solutions within exponentially large search spaces while satisfying complex constraints. This challenge is especially pronounced in resource-constrained settings, where larger language models are impractical and selection is limited to smaller models which often fail to preserve feasibility when scheduling directly from natural language. To address these limitations, we introduce SDDL, a neuro-symbolic framework that translates natural-language scheduling problems into compact, solver-aligned representations of tasks, resources, constraints, and objectives, while delegating low-level modeling and search to a deterministic compiler and external solver. On a 300-instance, multi-family subset of scheduling problems, SDDL improves independently verified feasibility for every resource-constrained model tested. The two strongest SDDL
    
[^202]: 针对低资源语言仇恨言论检测的LLM高效适配：罗马乌尔都语的比较研究

    Efficient Adaptation of LLMs for Hate Speech Detection in Low-Resource Languages: A Comparative Study on Roman Urdu

    [https://arxiv.org/abs/2608.18142](https://arxiv.org/abs/2608.18142)

    本研究通过LoRA参数高效微调方法，系统比较了多种大型语言模型在罗马乌尔都语低资源环境下的仇恨言论检测性能，展示了PEFT在零样本推理中的优势。

    

    由于注释数据缺乏、语言结构非正式以及标准化语法缺失，在低资源语言中检测仇恨言论颇具挑战性。罗马乌尔都语就是此类挑战的一个典型例子，它在南亚社交媒体上广泛使用，拼写变异大且缺乏上下文一致的规范。本文旨在全面评估大型语言模型在罗马乌尔都语脚本中的仇恨言论检测性能，并采用参数高效微调方法——低秩适配（LoRA）对这些模型进行微调。为评估零样本推理，我们在不同变压器模型（包括Mistral、LLaMA、Falcon和多语言BERT）上将其与PEFT进行基准对比。实验在包含超过72,000条注释的PURUTT数据集（乌尔都语和罗马乌尔都语有毒评论及音译平行语料库）上进行。

    arXiv:2608.18142v1 Announce Type: new  Abstract: It is challenging to detect hate speech in Low Resource Languages (LRLs) because of the absence of annotated data, the informality of its language structure, and the lack of standardized grammar. A good example of such a challenge is Roman Urdu which is broadly used by South Asians on social media and has a high variation while lacking contextually consistent spellings. The objective of this paper is to conduct a comprehensive assessment of Large Language Models (LLMs) for Hate Speech Detection (HSD) in Roman Urdu script and fine-tune these models using the Parameter-Efficient Fine-Tuning (PEFT) method called Low-Rank Adaptation (LoRA). To evaluate zero-shot inference, we benchmarked it against PEFT on different transformer models, including Mistral, LLaMA, Falcon, and multilingual BERT. Experiments are conducted on the PURUTT (Parallel Urdu and Roman Urdu Corpus for Toxic Comments and Transliteration) dataset with over 72,000 annotated 
    
[^203]: 拨开迷雾：为LLM智能体安装并优化主动探索能力

    Clearing the Fog: Towards Installing and Refining Proactive Exploration Capabilities in LLM Agents

    [https://arxiv.org/abs/2608.14339](https://arxiv.org/abs/2608.14339)

    本论文提出了一种名为\ours的方法，通过探索性数据构建和对比信号引导的强化学习，有效增强了LLM智能体的主动探索能力，克服了后见之明偏差并区分有效探索与冗余行为。

    

    arXiv:2608.14339v1 公告类型：新 摘要：我们研究了LLM智能体中的主动探索，即探索环境以获取信息从而改善未来决策的能力。在这方面，我们首先识别出阻碍这一能力的两个基本瓶颈，然后提出了一种名为\ours的新方法，旨在灌输并优化主动探索。具体来说，\ours包括两个组成部分：（1）探索性数据构建，通过合成富含探索的轨迹来缓解标准演示中的后见之明偏差；（2）带对比信号引导的强化学习优化，利用对比轨迹对来区分有效探索与冗余徘徊。大量实验证明了\ours的有效性，并提供了对主动探索特征的深入见解。我们的代码可在以下网址获取：https://github.com/GuanZhizhao/SAFARI。

    arXiv:2608.14339v1 Announce Type: new  Abstract: We study proactive exploration in LLM agents, i.e., the ability to explore an environment to acquire information that improves future decision-making. In this regard, we first identify two fundamental bottlenecks that hinder this capability and then propose \ours, a novel method designed to instill and refine proactive exploration. Specifically, \ours\ consists of two components: (1) Exploratory Data Construction, which synthesizes exploration-rich trajectories to mitigate the hindsight bias of standard demonstrations; and (2) RL Optimization with Contrastive Signal Guidance, which leverages contrastive trajectory pairs to distinguish productive exploration from redundant wandering. Extensive experiments demonstrate the effectiveness of \ours\ and provide insights into the characteristics of proactive exploration. Our code is available at: https://github.com/GuanZhizhao/SAFARI.
    
[^204]: VALG：一个用于机器学习理论研究的智能体系统

    VALG: An Agentic System for ML Theory Research

    [https://arxiv.org/abs/2608.13060](https://arxiv.org/abs/2608.13060)

    本文提出了VALG系统，通过多级验证、自适应问题表述和图结构证明开发，实现了机器学习理论研究中开放问题求解的自主智能体工作流程。

    

    摘要：机器学习理论通过数学框架研究学习过程，其中数据模型、训练协议、预言机访问、损失函数、评估指标和随机性共同定义了定理所要解释的现象。因此，解决一个开放问题需要问题表述、定理目标和证明机制协同发展。研究人员提出假设，通过初步的理论或实证分析进行检验，并不断完善假设和证明。我们探讨这一过程是否能被组织为机器学习理论研究的自主智能体工作流程。我们开发了VALG，一个结合多级验证、自适应学习理论问题表述和图结构证明开发的智能体系统。在每一个源相关定理分支中，VALG保持固定的数学规范，检查类型化证明依赖图的定理级组合，并约束...

    arXiv:2608.13060v1 Announce Type: new  Abstract: Machine learning theory studies learning procedures through mathematical setups in which the data model, training protocol, oracle access, loss, metric, and randomness define the phenomenon that a theorem is meant to explain. Solving an open problem therefore requires the problem formulation, theorem target, and proof mechanism to be developed in concert. Researchers formulate hypotheses, test them through preliminary theoretical or empirical analysis, and refine both assumptions and proofs. We investigate whether this process can be organized as an autonomous agentic workflow for ML theory research.   We develop VALG, an agentic system that combines multi-level Verification, Adaptive formulation of Learning-theory problems, and Graph-structured proof development. Within each source-relative theorem branch, VALG maintains a fixed mathematical specification, checks the theorem-level composition of a typed proof-dependency graph, and const
    
[^205]: NaviDC-OCR：跨越数字与相机拍摄文档的解析导航

    NaviDC-OCR: Navigating Document Parsing Across Digital and Camera-Captured Documents

    [https://arxiv.org/abs/2608.12898](https://arxiv.org/abs/2608.12898)

    NaviDC-OCR通过引入变形感知学习和自适应采样机制，统一了数字与相机拍摄文档的解析框架，有效解决了几何变形和结构推理不足的问题。

    

    arXiv:2608.12898v1 公告类型：交叉 摘要：文档解析旨在将非结构化文档转换为结构化且机器可读的表示形式。视觉语言模型（VLMs）的最新进展显著推动了文档解析的发展。然而，现有方法仍面临两大挑战。首先，基于解耦的VLM方法严重依赖准确的布局分析，而相机拍摄文档中的几何变形可能引入级联错误。其次，尽管基于端到端的VLM方法减轻了对显式布局检测的依赖，但在高分辨率场景中，它们常常遭受冗余生成、幻觉和结构推理不足的问题。为应对这些挑战，我们提出了NaviDC-OCR，一个统一的文档解析框架。NaviDC-OCR引入了变形感知学习，以将几何感知融入VLMs，并提出了一种自适应采样机制，用于复杂布局表示。此外，一个c（原文截断）

    arXiv:2608.12898v1 Announce Type: cross  Abstract: Document parsing aims to transform unstructured documents into structured and machine-readable representations. Recent advances in Vision-Language Models (VLMs) have significantly advanced document parsing. However, existing approaches still face two major challenges. First, decoupled VLM-based methods heavily rely on accurate layout analysis, where geometric distortions in camera-captured documents can introduce cascading errors. Second, although end-to-end VLM-based methods alleviate the dependence on explicit layout detection, they often suffer from redundant generation, hallucinations, and insufficient structural reasoning in high-resolution scenarios. To address these challenges, we propose NaviDC-OCR, a unified framework for document parsing. NaviDC-OCR introduces deformation-aware learning to incorporate geometric perception into VLMs and proposes an adaptive sampling mechanism for complex layout representation. Furthermore, a c
    
[^206]: 终端对称性作为决策资源：用于任意时间验证构建的状态细化

    Terminal Symmetry as a Decision Resource: Statewise Refinement for Anytime Verified Construction

    [https://arxiv.org/abs/2608.11318](https://arxiv.org/abs/2608.11318)

    本文提出了一种将终端对称性作为决策资源的新框架，通过传输-细化-认证机制实现任意时间验证构建，并提供了完成保证和最优验证器查询界限。

    

    arXiv:2608.11318v1 公告类型：交叉 摘要：许多顺序构建任务在完成时表现出精确的对称性，而其执行过程仍是有方向性和历史依赖的。我们提出了一种终端对称性的决策资源视角：过程证据提供方向性，终端对应关系将该结构传输到等价结果上，实现状态证据在转换后细化其当前决策相关性，固定验证器认证执行过程。这种分解产生了传输-细化-认证框架。\method{} 通过情节固定的传输过程结构、其状态限制的过程秩、在接受的转换后刷新的状态依赖残差秩，以及一个序数秩交集（其前$k$集合恰好是两个提议前缀的并集）来实现该原则。该交集在前缀覆盖下提供完成保证，并在相应前缀信息模型下达到最紧的最坏情况验证器查询界限；一个t

    arXiv:2608.11318v1 Announce Type: cross  Abstract: Many sequential construction tasks exhibit exact symmetry at completion while their execution remains directed and history-dependent. We develop a decision-resource view of terminal symmetry: process evidence supplies directionality, terminal correspondence transports that structure across equivalent outcomes, realized-state evidence refines its current decision relevance after transitions, and a fixed verifier certifies execution. This decomposition yields transport--refine--certify. \method{} instantiates the principle with an episode-fixed transported process structure, its state-restricted process rank, a state-dependent residual rank refreshed after accepted transitions, and an ordinal rank meet whose top-$k$ set is exactly the union of the two proposal prefixes. The meet provides a completion guarantee under prefix coverage and attains the tight worst-case verifier-query bound under the corresponding prefix information model; a t
    
[^207]: GitSkills：GitHub上的智能体技能数据集

    GitSkills: A Dataset of Agent Skills on GitHub

    [https://arxiv.org/abs/2608.10906](https://arxiv.org/abs/2608.10906)

    本文提出了GitSkills数据集，收录了GitHub上3,797,117个智能体技能文件，为首次实证研究开发者如何编写、复用和维护基于自然语言的智能体技能提供了数据基础。

    

    智能体技能是一个包含SKILL.md文件的文件夹，其中含有面向语言模型智能体的指令，并可选择性地附带脚本和参考文件。当智能体判断某项任务与技能描述匹配时，便会加载该技能。Anthropic于2025年10月将该格式作为开放规范推出。九个月后，公开的GitHub仓库中已存在数百万个技能文件。技能不同于软件工程研究人员通常挖掘的软件制品：它们主要以自然语言编写，由模型在运行时以概率方式选择，且没有编译器或类型检查器来验证选择过程。技能也没有中央注册表或包管理器；开发者通过在仓库之间复制文件夹来复用技能。因此，开发者如何编写、复用和维护技能成为一个实证问题，而现有数据集尚未记录这一群体。我们提出了GitSkills，一个包含3,797,117个SKILL文件的数据集。

    arXiv:2608.10906v2 Announce Type: replace  Abstract: An agent skill is a folder containing a $\mathrm{SKILL.md}$ file with instructions for a language-model agent, optionally accompanied by scripts and reference files. The agent loads the skill when it judges that a task matches the skill description. Anthropic introduced the format in October 2025 as an open specification. Nine months later, public GitHub repositories hold millions of skill files. Skills are unlike the artifacts that software engineering researchers usually mine: they are written mainly in natural language, a model selects them probabilistically at run time, and no compiler or type checker verifies the selection. Skills also have no central registry or package manager; developers reuse them by copying folders between repositories. How developers write, reuse, and maintain skills is therefore an empirical question, and no existing dataset records this population. We present GitSkills, a dataset of 3,797,117 $\mathrm{SK
    
[^208]: SPECTRA：面向地理空间基础模型跨传感器微调的波段路由嵌入与分阶段LoRA

    SPECTRA: Band-Routed Embedding and Stage-Wise LoRA for Cross-Sensor Fine-Tuning of Geospatial Foundation Models

    [https://arxiv.org/abs/2608.01751](https://arxiv.org/abs/2608.01751)

    SPECTRA提出了一种参数高效微调框架，通过波段路由嵌入解决预训练模型与下游传感器之间的光谱失配问题，并通过分阶段LoRA降低地理空间基础模型的微调适配成本。

    

    摘要：地理空间基础模型在大规模地理空间数据（如地球观测（EO）、气候和天气数据）上进行预训练，在微调至多种下游任务时展现出良好的性能。然而，将EO预训练的GeoFMs适配到实际下游数据集时存在两个挑战。第一个挑战是如何处理光谱失配：预训练的补丁嵌入期望固定的输入波段集合，而下游传感器可能提供不同的通道。第二个挑战是如何降低微调成本并使其高效。尽管现有工作已分别针对这些挑战做出了努力，但在光谱失配条件下联合提升微调性能并同时降低适配成本的研究仍然不足。我们提出了SPECTRA，一个同时解决光谱失配和适配成本的参数高效微调框架。为处理光谱失配问题，SPECTRA引入了波段……（原文摘要在此处截断）

    arXiv:2608.01751v2 Announce Type: replace-cross  Abstract: Geospatial foundation models (GeoFMs), pretrained on large-scale geospatial data such as Earth observation (EO), climate, and weather data, have shown promising performance when fine-tuned on diverse downstream tasks. However, there are two challenges of adapting EO-pretrained GeoFMs to practical downstream datasets. The first challenge is how to handle spectral mismatch: pretrained patch embeddings expect a fixed set of input bands, whereas downstream sensors may provide different channels. The second challenge is how to reduce fine-tuning cost and make it efficient. While existing work has made efforts on these challenges individually, jointly improving fine-tuning performance under spectral mismatch while reducing adaptation cost remains underexplored. We propose SPECTRA, a parameter-efficient fine-tuning framework that addresses both spectral mismatch and adaptation cost. To handle spectral mismatch, SPECTRA introduces Band
    
[^209]: 共情框架：评估人工智能跨社会人口群体的对齐性

    Sympathetic Framing: Evaluating AI Alignment across Sociodemographic Groups

    [https://arxiv.org/abs/2607.27232](https://arxiv.org/abs/2607.27232)

    该研究通过对3011名英国成年人的YouGov调查与七个大语言模型的对比实验，首次实证评估了LLMs在感知新闻标题情感框架（对冲突各方的同情）方面与人类的对齐程度，发现领先模型（如GPT-5.2相关性达0.789）在所有人口统计子群体中都与人类情感判断高度一致。

    

    大语言模型（LLMs）日益影响着我们获取信息和形成世界观的方式，这引发了超越人工智能偏见的担忧：大语言模型能否理解通过文本框架传达的情感细微差别？在这项工作中，我们实证评估了多种大语言模型与人类情感感知的对齐程度。针对涉及政治和地缘政治冲突的新闻标题，人类参与者（n = 3011，通过YouGov调查获得的英国成年人代表性样本）和七个大语言模型回答了标题是否引起了对冲突中某一特定方的同情。我们发现人工智能与人类评估之间的相关性因模型而异，从非常高（0.789，GPT-5.2）到中等（0.4，Mistral Large 2512）不等。至关重要的是，领先的模型在所有人口统计子群体（包括年龄、性别、教育水平、先天地缘政治知识等）中都与人类判断基本保持一致。

    arXiv:2607.27232v2 Announce Type: replace  Abstract: Large Language Models (LLMs) are increasingly shaping how we consume information and form our worldview. This raises concerns beyond bias in AI: do LLMs grasp the emotional nuances conveyed via textual framing? In this work, we empirically evaluate how well an array of LLMs aligns with human emotional perception. Considering news headlines covering political and geopolitical conflicts, both human participants (n = 3011, a representative sample of the U.K. adult population, via a YouGov survey) and seven LLMs answered whether headlines evoked sympathy for a specified side in a conflict. We find that the correlation between AI and human evaluations varies across models, ranging from very high (0.789, GPT-5.2) to medium (0.4 ,Mistral Large 2512). Crucially, the leading models are broadly aligned with human judgments across all demographic subgroups, including age, gender, level of education, prior geopolitical knowledge, and participant
    
[^210]: SciFigQual-Bench：一个基于全文语境的科学图像质量评估基准

    SciFigQual-Bench: A Benchmark for Scientific Figure Quality Assessment with Full-Manuscript Context

    [https://arxiv.org/abs/2607.27084](https://arxiv.org/abs/2607.27084)

    提出SciFigQual-Bench，首个结合全文语境、从清晰度、布局、图注契合度、上下文相关性和误导风险五个维度评估科学图像质量，并配备多位领域专家黄金标准标注的基准数据集。

    

    科学图像是科学论文中呈现实验结论、阐述系统架构和支持比较论证的核心元素。然而，现有的图像质量评估（IQA）方法主要针对自然照片或AI生成内容设计，无法直接应用于科学论文。现有少数针对学术图表的研究仍局限于视觉表面层面的比较，无法验证图注一致性、引用相关性或视觉误导性。为解决这一问题，我们提出了SciFigQual-Bench，这是一个基于全文语境的基准，从五个维度（清晰度、布局、图注契合度、上下文相关性和误导风险）评估科学图像。数据涵盖了2020年至2025年的顶级计算机科学会议；6,308张图像由多位领域专家在五个维度上独立评分，并汇总为黄金标准标注。

    arXiv:2607.27084v2 Announce Type: replace-cross  Abstract: Scientific images are the core elements of presenting experimental conclusions, elaborating system architecture, and supporting comparative arguments in scientific papers. However, existing image quality assessment (IQA) methods are predominantly designed for natural photographs or AI-generated content, which cannot be directly applied to scientific papers. The few existing studies on scholarly charts remain confined to visual-surface comparisons, failing to verify caption alignment, citation relevance, or visual misleadingness. To address this, we propose SciFigQual-Bench, a full-text contextual benchmark that evaluates scientific images across five dimensions (clarity, layout, caption fit, context relevance, and misleading risk). The data covers top computer-science conferences from 2020 to 2025; 6,308 images were independently scored by multiple domain experts in five dimensions and aggregated into gold-standard annotations.
    
[^211]: 用于锂电池电解质电子结构分析的密度矩阵框架

    A Density-Matrix Framework for Electronic-Structure Analysis of Electrolytes for Lithium Batteries

    [https://arxiv.org/abs/2607.25597](https://arxiv.org/abs/2607.25597)

    提出EMolStudio——一个以密度矩阵为中心的AI平台，可高效预测和分析锂电池电解质的电子结构，并系统揭示了分子官能化与盐阴离子种类对前线能级、静电势及轨道局域化的调控规律。

    

    锂电池中电解质的反应活性由分子官能团、Li$^{+}$溶剂化以及盐阴离子的参与共同决定。传统量子化学方法的计算成本过高，难以对多样化的电解质分子及其局部溶剂化环境进行系统性分析。本文提出了EMolStudio——一个以密度矩阵为中心的人工智能平台，用于电子结构的预测与分析。其工作流程整合了分子官能化、显式Li$^{+}$第一溶剂壳层组装、密度矩阵预测以及电子结构解析。将该平台应用于四种锂盐下的163,655个官能化分子和22,500个第一壳层团簇，我们发现：1）官能化通过前线能级、静电势以及Li$^{+}$—给体接触方式的显著差异，有效区分了CO$_{2}$Me、CN、F/CF$_{3}$和磺酰基团；2）阴离子种类重塑了前线轨道的局域化分布，其中LiTDI锚定了最高的占据……（原文摘要在此处截断）

    arXiv:2607.25597v3 Announce Type: replace  Abstract: Electrolyte reactivity in lithium batteries is shaped by molecular functional groups, Li$^{+}$ solvation and salt-anion participation. Conventional quantum chemistry is too computationally expensive for systematic analysis of diverse electrolyte molecules and their local solvation environments. Here we present EMolStudio, a density-matrix-centered AI platform for electronic-structure prediction and analysis. Its workflow integrates molecular functionalization, explicit Li$^{+}$ first-shell assembly, density-matrix prediction, and electronic-structure parsing. Applied to 163,655 functionalized molecules and 22,500 first-shell clusters across four lithium salts, we find that 1) functionalization separates CO$_{2}$Me, CN, F/CF$_{3}$, and sulfonyl groups by distinct shifts in frontier levels, electrostatic potential, and Li$^{+}$-donor contact; 2) anion identity reshapes frontier-orbital localization, with LiTDI anchoring the highest occ
    
[^212]: ScaleResfusion：基于残差向量场的残差整流流

    ScaleResfusion: Residual Rectified Flow based on Residual Vector Field

    [https://arxiv.org/abs/2607.25275](https://arxiv.org/abs/2607.25275)

    ScaleResfusion 提出残差整流流（RRF），通过将残差项嵌入整流流的线性传输路径，把真实世界图像恢复转化为与噪声调度器无关的适配接口，从而能够复用预训练的文本到图像整流流模型，从带噪低质量图像出发高效恢复高质量图像。

    

    真实世界图像恢复旨在从复杂且未知的退化中恢复高质量图像。近期基于扩散的方法显著提升了感知质量，但仍存在两大障碍：从高斯噪声开始采样的方法需要大量步骤，且对退化输入的保真度通常较低；而从低质量图像出发的基于残差的方法通常需要从头训练任务特定模型，其优化目标与特定的噪声调度器耦合，因此无法复用现代预训练生成先验。我们提出了 ScaleResfusion，它将残差恢复重写为一种与调度器无关的适配接口，用于预训练的文本到图像整流流模型。其核心 Residual Rectified Flow（RRF，残差整流流）将残差项 R 插入整流流的线性传输路径中，使得采样从带噪的低质量图像开始……（原文摘要在此处截断）

    arXiv:2607.25275v2 Announce Type: replace-cross  Abstract: Real-world Image Restoration (Real-IR) aims to recover high-quality (HQ) images from complex and unknown degradations. Recent diffusion-based methods have substantially improved perceptual quality, yet two obstacles remain: methods that sample from Gaussian noise require many steps and are often less faithful to the degraded input, whereas residual-based methods that start from the low-quality (LQ) image typically train task-specific models from scratch, with optimization objectives coupled to a particular noise scheduler, and therefore cannot reuse modern pre-trained generative priors. We present \textbf{ScaleResfusion}, which rewrites residual restoration as a scheduler-independent adaptation interface for pre-trained text-to-image rectified-flow models. Its core, \textbf{Residual Rectified Flow} (RRF), inserts the residual term $R$ into the linear transport path of Rectified Flow, so that sampling starts from noisy LQ at an 
    
[^213]: SGA：面向教学视频合成的即插即用几何验证方法

    SGA: Plug&Play Geometric Verification for Educational Video Synthesis

    [https://arxiv.org/abs/2607.18116](https://arxiv.org/abs/2607.18116)

    该论文提出即插即用的符号几何智能体SGA，通过拦截并部分执行LLM生成的Manim代码提取符号场景图以检测和修正空间冲突，并引入无需渲染的MVQS评估指标，使教学动画的空间正确性相对基线提升16.1%。

    

    近期研究利用大语言模型（LLM）结合Manim等库生成用于教学动画的可执行代码。然而，确保空间正确性和视觉可读性仍然具有挑战性，因为现有框架侧重于教学内容而忽视了几何遮挡问题。我们提出了符号几何智能体（SGA），这是一个面向以代码为中心的动画流水线的即插即用模块，它能够拦截LLM生成的代码，通过部分执行来提取符号场景图，并在检测到空间冲突时进行针对性修正。我们进一步提出了Manim视觉质量评分（MVQS），这是一种确定性的、无需渲染的空间完整性评估代理指标。在MMMC-Code基准上，针对四个LLM骨干模型和两个智能体流水线的实验表明，SGA实现了73.11的峰值MVQS（Code2Video + GPT-5.1），相比原始基线有16.1%的相对提升，并提升了MV

    arXiv:2607.18116v2 Announce Type: replace  Abstract: Recent work leverages Large Language Models (LLMs) to generate executable code for pedagogical animations using libraries such as Manim. However, ensuring spatial correctness and visual legibility remains challenging, as existing frameworks emphasize pedagogical content while overlooking geometric occlusions. We propose the Symbolic Geometric Agent (SGA), a plug-and-play module for code-centric animation pipelines that intercepts LLM-generated code, performs partial execution to extract symbolic scene graphs, and applies targeted refinement when spatial conflicts are detected. We further introduce the Manim Visual Quality Score (MVQS), a deterministic rendering-free proxy for spatial integrity. Experiments on the MMMC-Code benchmark across four LLM backbones and two agentic pipelines show that SGA achieves a peak MVQS of 73.11 (Code2Video + GPT-5.1), corresponding to a 16.1% relative improvement over the raw baseline, and improves MV
    
[^214]: 自然语言访问领域特定元数据：一个可复用的LLM查询生成框架

    Natural Language Access to Domain-Specific Metadata: A Reusable Framework for LLM Query Generation

    [https://arxiv.org/abs/2607.18029](https://arxiv.org/abs/2607.18029)

    该论文提出NLKGQ框架，通过在OWL本体中形式化领域词汇和语义，使LLM能够零样本将自然语言问题转换为准确的SPARQL查询，从而无需微调、检索增强或多智能体编排即可实现对领域特定元数据档案的自然语言访问。

    

    arXiv:2607.18029v2 公告类型：replace-cross。摘要：研究人员需要回答关于领域特定档案内容的临时性问题，但往往缺乏在元数据上编写结构化查询的专业知识。我们证明，当领域词汇和语义被捕获在设计良好的Web本体语言（OWL）本体中时，大型语言模型（LLM）能够零样本生成准确的结构化查询，无需任务特定的微调、检索增强或多智能体编排。我们提出了自然语言知识图谱查询（NLKGQ）系统，这是一个使自然语言能够访问此类档案中元数据的框架和开发流程。该框架包含一个帮助研究人员提出自然语言问题的Web界面，一个与领域无关的执行组件通过LLM将问题转换为SPARQL并在知识图谱上执行。该开发流程始于在正式的OWL本体中捕获领域词汇和语义。领域专家……（原文摘要至此截断）

    arXiv:2607.18029v2 Announce Type: replace-cross  Abstract: Researchers need to answer ad-hoc questions about the contents of domain-specific archives but often lack the expertise to write structured queries on the metadata. We show that when domain vocabulary and semantics are captured in a well-designed Web Ontology Language (OWL) ontology, Large Language Models (LLMs) can generate accurate structured queries zero-shot, without task-specific fine-tuning, retrieval augmentation, or multi-agent orchestration. We present the Natural Language Knowledge Graph Query (NLKGQ) system, a framework and development process that enables natural language access to metadata in such archives. The framework includes a web interface that helps researchers pose natural language questions, which a domain-agnostic harness translates to SPARQL via an LLM and executes against a knowledge graph. The development process begins with capturing domain vocabulary and semantics in a formal OWL ontology. Domain-spe
    
[^215]: EVOQUANT：面向稳健量化交易的自进化验证器引导策略优化

    EVOQUANT: Self-Evolving Verifier-Guided Strategy Optimization for Robust Quantitative Trading

    [https://arxiv.org/abs/2607.12455](https://arxiv.org/abs/2607.12455)

    EVOQUANT提出了一个自进化的验证器引导框架，利用大语言模型诊断量化策略瓶颈、生成受控编辑并通过多阶段验证筛选最优策略，同时将优化经验沉淀为可复用知识，实现量化交易策略的持续稳健自我改进。

    

    量化策略优化在很大程度上仍然是人工完成的，需要领域专家识别微弱信号、调整风险控制规则，并反复验证迭代修订方案。大语言模型可以加速这一过程，但直接依赖它们重写交易策略往往会引入幻觉式编辑、策略漂移和回测过拟合等问题。我们提出了EVOQUANT，一个用于量化交易策略优化的自进化验证器引导框架。我们的方法利用大语言模型深度诊断性能瓶颈，生成语义受控的候选编辑，通过多阶段验证流程筛选出最优策略，并将优化经验提炼为可复用的知识，以实现持续的自我改进。我们使用七种代表性策略对该方法进行评估：其中四种来自A股市场，三种来自加密货币市场。实验结果表明，我们的方法显著……

    arXiv:2607.12455v3 Announce Type: replace  Abstract: Quantitative strategy optimization remains largely manual, requiring domain experts to identify weak signals, tune risk-control rules, and repeatedly validate iterative revisions. Large language models can accelerate this process, but directly relying on them to rewrite trading strategies often introduces hallucinated edits, strategy drift, and backtest overfitting. We propose EVOQUANT, a self-Evolving Verifier-guided framework for strategy Optimization in Quantitative trading. Our method utilizes LLMs to deeply diagnose performance bottlenecks, generates semantically controlled candidate edits, selects the best strategy through a multi-stage verification pipeline, and distills optimization experience into reusable knowledge for continual self-improvement. We evaluate our method using seven representative strategies: four from the A-share market and three from the Crypto market. Experimental results show that our method significantly
    
[^216]: 基于有限规则修订的自适应智能体控制器验证

    Verification of Adaptive Agentic Controllers through Finite Rule Revision

    [https://arxiv.org/abs/2607.09770](https://arxiv.org/abs/2607.09770)

    本文提出了一种将有界可修订对象作为自适应智能体控制器表示的验证协议，通过有限符号规则、显式诊断谓词、解释日志和留出集再评估，在不依赖无限制人工介入判断的情况下实现控制器故障的检测、局部修复或拒绝。

    

    工业级智能体AI系统日益显现出原型能力与生产部署之间的差距。特别是，自适应智能体可能生成看似合理的输出，但在非确定性、保密性约束、有限上下文和弱可观测性的条件下仍难以验证。本文为以有限符号规则、显式诊断谓词、解释日志和留出集再评估所表示的自适应智能体控制器制定了一种有界验证协议。核心研究问题是：当自适应智能体控制器通过有限规则、显式诊断谓词、解释日志和留出集再评估来表示时，哪些类别的控制器故障可以在不依赖无限制的人机协同判断的情况下被检测、局部修复或拒绝？所提出的框架将控制器视为一个有限可修订的对象。诊断故障被映射到预……（原文摘要在此处截断）

    arXiv:2607.09770v2 Announce Type: replace  Abstract: Industrial agentic AI systems increasingly exhibit a gap between prototype capability and production deployment. In particular, adaptive agents may generate plausible outputs while remaining difficult to verify under non-determinism, confidentiality constraints, limited context, and weak observability. This paper formulates a bounded verification protocol for adaptive agentic controllers represented by finite symbolic rules, explicit diagnostic predicates, explanation logs, and held-out re-evaluation. The central research question is: when an adaptive agentic controller is represented through finite rules, explicit diagnostic predicates, explanation logs, and held-out re-evaluation, which classes of controller failure can be detected, locally repaired, or rejected without relying on unrestricted human-in-the-loop judgment? The proposed framework treats the controller as a finite revisable object. Diagnostic failures are mapped to pre
    
[^217]: 这不是一个烟斗：作为语义抽象的AI系统

    Ceci n'est pas une pipe: AI systems as semantic abstractions

    [https://arxiv.org/abs/2607.09489](https://arxiv.org/abs/2607.09489)

    该论文提出了一个语义框架，将AI系统的输出视为工程化构建的表示而非事实本身，通过区分公认领域知识、参考来源和系统可用信息，为外推、无依据断言等常见失败模式给出精确定义，从而为规范和检验AI系统的可靠性提供词汇体系。

    

    AI系统的输出并非它表面上所描述的事实或世界状态，而是一种被工程化构建的表示。我们提出了一个语义框架来描述AI系统，以便能够检验这类表示的正确性。为此，我们区分了三个方面：被公认领域知识所证成的内容、参考来源所陈述的内容，以及系统当前可以使用的内容。这使我们能够对常见的失败模式给出精确定义：外推、被反驳或无依据的断言、来源与知识不匹配、过时或被反驳的来源、额外添加的假设、无依据的使用等。我们希望该框架能为规范和检验AI系统提供一套有用的词汇体系——这些系统的输出、引用、工具调用和改变世界的行动必须由可靠的主张和明确的授权来证成，而非凭借表面的流畅性。

    arXiv:2607.09489v2 Announce Type: replace  Abstract: An AI system's output is not the fact or world state it appears to describe, but rather an engineered representation. We propose a semantic framework to describe AI systems, to be able to examine the correctness of such representations. To do so, we distinguish what is justified by accepted domain knowledge, what reference sources say, and what the system can currently use. This allows us to give precise definitions to common failures: extrapolation, refuted or unsupported assertion, sources versus knowledge mismatch, stale or refuted source, added hypotheses, unsupported use... We hope our framework gives a useful vocabulary for specifying and checking AI systems whose outputs, citations, tool calls, and world-changing actions must be justified by reliable claims and explicit authority rather than apparent fluency.
    
[^218]: ProsMAE：用于ISUP分级分类的多源MAE预训练方法

    ProsMAE: Multi-Source MAE Pretraining for ISUP Grade Classification

    [https://arxiv.org/abs/2607.08162](https://arxiv.org/abs/2607.08162)

    本文提出ProsMAE，一种利用PANDA、CAMELYON17和BRACS三个数据源进行MAE预训练的组织病理学表示学习框架，通过冻结编码器和线性分类头迁移至ISUP分级分类任务，并取得了优于原始MAE的QWK分数。

    

    全切片图像（WSI）为计算病理学提供了丰富的诊断信息，但其千兆像素级的规模、染色变异、扫描仪差异、组织伪影以及有限的专家标注，使得训练稳健的模型充满挑战。本文提出了一种名为ProsMAE的多源掩码自编码器（MAE）框架，用于组织病理学表示学习。来自前列腺癌分级评估数据集（PANDA）、2017年癌症淋巴结转移挑战赛（CAMELYON17）和乳腺癌分型数据集（BRACS）的图像块被用于ProsMAE预训练，使编码器能够接触多样化的组织形态学和采集条件。学习到的编码器通过ProsCLS迁移至国际泌尿病理学会（ISUP）分级分类任务，采用冻结编码器和线性分类头的组合。ProsMAE取得了比原始MAE冻结线性探测更高的平均验证集二次加权Kappa（QWK）分数。

    arXiv:2607.08162v2 Announce Type: replace-cross  Abstract: Whole slide images (WSIs) provide rich diagnostic information for computational pathology, but their gigapixel scale, stain variation, scanner differences, tissue artifacts, and limited expert annotation make robust model training challenging. This paper presents a multi-source Masked Autoencoder (MAE) framework, named ProsMAE, for histopathology representation learning. Tiles from Prostate cANcer graDe Assessment (PANDA), CAncer MEtastases in LYmph nOdes challeNge 2017 (CAMELYON17), and BReAst Carcinoma Subtyping (BRACS) are used for ProsMAE pretraining to expose the encoder to diverse tissue morphology and acquisition conditions. The learned encoder is transferred for International Society of Urological Pathology (ISUP) grade classification through ProsCLS, using a frozen encoder and a linear classification head. ProsMAE achieved a higher mean validation quadratic weighted kappa (QWK) than the vanilla MAE frozen linear-probe 
    
[^219]: DiaLLM：英语方言适应中鲁棒性与生成能力差距的探究

    DiaLLM: An Investigation into the Robustness-Generation Gap in English Dialect Adaptation

    [https://arxiv.org/abs/2607.07669](https://arxiv.org/abs/2607.07669)

    本文发现方言理解与生成能力在LLM中分离，并证明显式方言定向适应优于广泛对齐，但基准测试无法反映这一生成优势。

    

    大语言模型越来越能“理解”方言英语，但仍只能“生成”标准且偏向美式的英语，导致方言生成这一更难的问题在很大程度上未被解决。我们引入了DiaLLM，该方法对三个开放权重语言模型家族在国际英语语料库上进行持续预训练，并应用隐式和显式后训练范式，每种范式结合三种模型对齐策略，首次对这些组件在澳大利亚、印度和北英格兰英语上的表现进行了受控比较。我们的结果显示，方言鲁棒性和生成能力是“分离”的：基准测试受持续预训练和SFT影响，而对齐则显著重塑生成方式，但基准测试无法捕捉这些变化。显式针对特定变体的适应能产生可靠被识别为方言的输出，且优于广泛对齐，但该方法在基准测试中的表现却未体现其优势。

    arXiv:2607.07669v2 Announce Type: replace-cross  Abstract: Large language models increasingly \emph{understand} dialectal English, yet still \emph{produce} only standard, US-leaning English, leaving dialectal generation, the harder half of the problem, largely unaddressed. We introduce \textbf{DiaLLM}, which continually pretrains three open-weight language model families on the International Corpus of English and applies implicit and explicit post-training paradigms, each combined with three model alignment strategies, giving the first controlled comparison of these components across Australian, Indian, and Northern British English. Our results reveal that dialectal robustness and generation are \emph{dissociated}: benchmarks are shaped by continual pretraining and SFT, while alignment visibly reshapes generation in ways benchmarks do not capture. Explicit variety-targeted adaptation produces output reliably recognised as dialectal and preferred over broad alignment, yet the method tha
    
[^220]: 对比音频解码中的自适应扰动选择

    Adaptive Perturbation Selection for Contrastive Audio Decoding

    [https://arxiv.org/abs/2607.00247](https://arxiv.org/abs/2607.00247)

    该论文提出针对对比解码，通过对多样化音频扰动库进行评估并为每个任务和样本自适应选择最优负分支，来缓解大型音频-语言模型的幻觉问题，例如将时间顺序任务准确率从74.7%提升至81.4%。

    

    大型音频-语言模型（LALMs）经常因用语言先验覆盖声学证据而产生幻觉。虽然对比解码（CD）提供了一种无需训练的缓解方法，但现有方法依赖于掩蔽或噪声等粗暴的扰动，尚未探索结构化的音频变换。我们通过评估一个多样化的针对性音频扰动库，并为每个任务和样本自适应地选择最优的负分支来探索这一设计空间。首先，我们改进了早期的提示工程方法，证明一个简单的二元“是/否”约束能够减少模型错误确认不存在的音频特征的倾向。其次，在时间、频谱、频率和幅度等域上对我们的扰动库进行评估后发现，最优变换高度依赖于具体任务；例如，反转音频数组会破坏时间连贯性，将时间顺序任务的准确率从74.7%提升至81.4%。

    arXiv:2607.00247v3 Announce Type: replace-cross  Abstract: Large audio-language models (LALMs) frequently hallucinate by overriding acoustic evidence with language priors. While contrastive decoding (CD) offers training-free mitigation, existing methods rely on blunt perturbations like masking or noise, leaving structured audio transformations unexplored. We explore this design space by evaluating a diverse library of targeted audio perturbations and adaptively selecting the optimal negative branch for each task and example. First, we improve upon earlier prompt engineering by showing that a simple binary yes/no constraint reduces the model's tendency to falsely confirm absent audio features. Second, evaluating our library across temporal, spectral, frequency, and amplitude domains reveals that optimal transformations are highly task-dependent; for instance, reversing the audio array disrupts temporal coherence, raising accuracy on the temporal order task from 74.7% to 81.4%. Finally, 
    
[^221]: 相关性并非许可：定位与控制面向指标的注意力贡献

    Relevance Is Not Permission: Localizing and Controlling Metric-Facing Attention Contributions

    [https://arxiv.org/abs/2606.30139](https://arxiv.org/abs/2606.30139)

    提出Warrant统一方法，通过暴露通向评估指标的逐项注意力贡献路径并施加查询条件化的许可控制，揭示“注意力相关性不等于预测贡献”（最高注意力项在约一半样本中反而损害效用），并在五类任务的32组对比中有27组提升了主要指标。

    

    注意力机制能够识别与当前查询相关的项目，但无法单独判断这些项目的价值贡献是否真正支持预测。我们提出Warrant，一种用于定位和控制面向指标的注意力贡献的统一方法。Warrant首先识别并暴露通向所报告指标的逐项贡献路径，然后在同一路径上施加基于当前查询条件的许可机制。完整版Warrant在CTDG、MTPP、RAG、STPP和TKG五类任务的32组模型-数据集对比中，有27组提升了主要指标。在五个代表性设置中进行的精确项目移除分析发现，注意力与边际预测效用之间的相关性几乎为零；甚至在43.5%至54.4%的样本中，注意力最高的项目反而会降低目标效用。对完整基准的贡献分解显示，路径暴露与习得许可两部分的作用因任务而异。在五个随机种子的HotpotQA分析中，被暴露的路径分配了更多的……（原文摘要在此处截断）

    arXiv:2606.30139v3 Announce Type: replace  Abstract: Attention identifies items relevant to a current query, but does not separately determine whether their value contributions support the prediction. We propose Warrant, a unified method for locating and controlling metric-facing attention contributions. Warrant identifies and exposes the item-wise contribution path that reaches the reported metric, then applies current-query-conditioned permission on that same path. Full Warrant improves the primary metric in 27 of 32 model-dataset comparisons across CTDG, MTPP, RAG, STPP, and TKG. Exact item-removal analysis in five representative settings finds near-zero correlation between attention and marginal prediction utility; even the highest-attention item reduces target utility in 43.5-54.4% of examples. Decomposition over the complete benchmark shows that the contributions of path exposure and learned permission vary by task. In a five-seed HotpotQA analysis, the opened path assigns more a
    
[^222]: LLM-意识形态可塑性：将大语言模型政治行为中的意识形态可塑性测量为语境条件分布

    LLM-Ideoplasticity: Measuring Ideological Plasticity in the Political Behavior of LLMs as a Context-Conditioned Distribution

    [https://arxiv.org/abs/2606.28335](https://arxiv.org/abs/2606.28335)

    大语言模型的政治意识形态不是固定点，而是随语境变化的条件分布——虽然对说服性框架、语言选择等因素高度敏感，但整体上占据的政治光谱范围仅为欧洲主要政党的约三分之一。

    

    我们以系统性实证证据论证，大语言模型的政治意识形态并非一个固定点，而是真实政治空间上的条件分布 P(立场|语境)。我们使用以VAA-CHES投影模型为锚定的统一测量框架评估了九个当前的大语言模型，该框架将模型响应映射到六个语境轴上的三个经过验证的维度（lrgen、lrecon、galtan）。我们的发现揭示了对语境的高度敏感性：说服性框架和代表性不足的语言分别使坐标偏移高达0.57和0.52个单位，而思维链推理往往会放大而非抑制改写不稳定性。尽管存在这种局部可塑性，模型群体总体上占据了一个非常狭窄的奥弗顿言论边界，大约仅占欧洲主要政党分布范围的三分之一。在多特质多方法（MTMM）分析的支持下，我们得出结论：单一的点（摘要在此处截断）

    arXiv:2606.28335v3 Announce Type: replace-cross  Abstract: We argue, with systematic empirical evidence, that a large language model's political ideology is not a fixed point, but a conditional distribution $\mathbb{P}($position$\mid$context$)$ over a real political space. We evaluate nine current LLMs using a unified measurement framework anchored by VAA-CHES projection models, which map responses onto three validated dimensions (lrgen, lrecon, galtan) across six contextual axes. Our findings reveal high sensitivity to context: persuasive framing and under-represented languages displace coordinates by up to 0.57 and 0.52 units, respectively, while chain-of-thought reasoning often amplifies rather than dampens paraphrase instability. Despite this local plasticity, the model cohort occupies a remarkably narrow Overton envelope overall, occupying roughly one-third the spread of major European parties. Supported by a multi-trait multi-method (MTMM) analysis, we conclude that a single poin
    
[^223]: 认知数字孪生：模拟心智的AI系统的伦理风险与治理

    Cognitive Digital Twins: Ethical Risks and Governance for AI Systems That Model the Mind

    [https://arxiv.org/abs/2606.23094](https://arxiv.org/abs/2606.23094)

    本文定义了认知数字孪生（CDTs）这一新型AI技术，指出其独特伦理风险，并提出了涵盖权威、自主性、访问与控制、问责制和可用性五个维度的5A治理框架。

    

    随着AI系统变得越来越持久化和个性化，一类我们称之为认知数字孪生（CDTs）的技术成为可能：即对特定个体认知的动态计算表征，通过行为、情境或生理数据进行更新，以建模、预测或模拟该个体的认知，或充当该个体的交流或决策代理。CDTs将认知推理与纵向表征、模拟和代理行动相结合，而现有的针对个人助理、自主智能体、推荐系统和自动决策系统的治理策略仅能部分应对这些问题。本文做出了四项贡献。第一，我们定义了CDTs并将其与相邻系统区分开来。第二，我们引入了一个围绕权威、自主性、访问与控制、问责制和可用性组织的5A治理框架。第三，我们识别了CDT特有的风险（摘要此处不完整）。

    arXiv:2606.23094v2 Announce Type: replace-cross  Abstract: As AI systems become increasingly persistent and personalized, they make possible a class of technologies that we call cognitive digital twins (CDTs): dynamic computational representations of a specific person's cognition, updated from behavioral, contextual, or physiological data in order to model, predict, or simulate that person's cognition, or to act as that person's communicative or decision-making proxy. CDTs combine cognitive inference with longitudinal representation, simulation, and proxy action in ways that existing governance strategies for personal assistants, autonomous agents, recommender systems, and automated decision systems only partially address. This paper makes four contributions. First, we define CDTs and distinguish them from adjacent systems. Second, we introduce a 5A governance framework organized around authority, autonomy, access and control, accountability, and availability. Third, we identify CDT-sp
    
[^224]: AI经济学家智能体：基于RAG、知识图谱和大语言模型的循证经济与金融分析智能体框架

    AI Economist Agent: An Agentic Framework for Evidence-Based Economic and Financial Analysis with RAG, Knowledge Graphs, and Large Language Models

    [https://arxiv.org/abs/2606.20041](https://arxiv.org/abs/2606.20041)

    该论文提出了一个结合RAG、知识图谱和大语言模型的AI经济学家智能体框架，由LLM智能体负责规划分析、检索证据和组织经济机制，由注册的定量模型生成数值结果并通过预定义检验保证可靠性，实现了面向欧洲宏观金融压力测试的循证经济与金融情景分析。

    

    我们提出了一个用于经济和金融情景分析的AI经济学家智能体。情景设计通常要求分析师在历史先例有限的情况下评估新兴风险，整合来自众多来源的信息，并将定性机制转化为内部一致的定量路径。大语言模型（LLM）能够搜索和综合这些信息，但仅凭流畅的叙述无法建立经济结论所需的基于模型的计算。我们的框架使用LLM智能体来规划分析、检索相关证据并组织经济机制，同时由注册的定量模型生成数值结果，并通过预定义的检验来确定中间结果是否可用于最终报告。我们将该框架应用于欧洲宏观金融压力情景和银行资本分析。实证分析评估了经济机制的检索、情景构建等（摘要被截断）。

    arXiv:2606.20041v2 Announce Type: replace-cross  Abstract: We propose an AI economist agent for economic and financial scenario analysis. Scenario design often requires analysts to assess emerging risks with limited historical precedent, combine information from many sources, and translate qualitative mechanisms into internally consistent quantitative paths. Large language models (LLMs) can search and synthesize this information, but fluent narratives alone do not establish the model-based calculations needed for economic conclusions. Our framework uses LLM agents to plan the analysis, retrieve relevant evidence, and organize economic mechanisms, while registered quantitative models generate numerical outcomes and predefined tests determine whether intermediate results can be used in the final report. We apply the framework to European macro-financial stress scenarios and bank capital analysis. The empirical analysis evaluates retrieval of economic mechanisms, scenario construction, mo
    
[^225]: 超越单一方向的拒绝行为：差值均值法与INLP的初步比较

    Refusal Beyond a Single Direction: A Preliminary Comparison of Diff-in-Means and INLP

    [https://arxiv.org/abs/2606.13720](https://arxiv.org/abs/2606.13720)

    本研究在五个开源聊天模型上比较了差值均值法（DiM）与迭代零空间投影（INLP）两种干预方法在引导模型拒绝行为上的效果，发现INLP反事实翻转在拒绝抑制方面与DiM方向消融相当，且对层选择更具鲁棒性。

    

    Arditi等人（2024）的研究表明，在经过安全微调的聊天模型中，拒绝行为是由残差流中的单个线性方向介导的，该方向可以通过有害和无害激活的差值均值法恢复。我们在五个开源权重的聊天模型上，将基于差值均值法的干预方法（激活添加和方向消融）与两种源自迭代零空间投影（INLP）的干预方法（零空间投影和反事实翻转）进行比较，以探究INLP能否在引导拒绝行为方面与差值均值法相媲美，以及其更丰富的参数化是否能带来更可调节的干预手段。研究结果表明，在拒绝抑制方面，INLP反事实翻转与差值均值法方向消融具有相当的竞争力，而零空间投影在大多数模型上表现较弱。将每种方法应用于对方所选定的层时，结果显示反事实翻转的竞争力对这种变化具有鲁棒性，而方向消融则不然，这表明差值均值法表面上的优势部分是一种人为产物。

    arXiv:2606.13720v2 Announce Type: replace  Abstract: Arditi et al. (2024) has shown that refusal in safety fine-tuned chat models is mediated by a single linear direction in the residual stream, recoverable by a difference-in-means (DiM) of harmful and harmless activations. We compare DiM-based interventions (activation addition and directional ablation) with two interventions derived from Iterative Nullspace Projection (INLP)-nullspace projection and counterfactual flipping-on five open-weight chat models, asking whether INLP can match DiM at steering refusal and whether its richer parameterisation yields more tweakable interventions. INLP counterfactual flipping is competitive with DiM directional ablation on refusal suppression, while nullspace projection is weaker on most models. Applying each method at the layer selected by the other shows that flipping's competitiveness is robust to this change while directional ablation's is not, suggesting that DiM's apparent edge is partly a p
    
[^226]: 一种用于自动化混凝土护栏设计的轻量级多智能体框架

    A Lightweight Multi-Agent Framework for Automated Concrete Barrier Design

    [https://arxiv.org/abs/2606.12040](https://arxiv.org/abs/2606.12040)

    该论文提出了一种基于AutoGen多智能体编排的“生成-验证-修改”闭环框架，通过多智能体协同克服大语言模型在结构设计中的幻觉与数值推理缺陷，实现符合AASHTO规范的钢筋混凝土公路护栏自动化设计。

    

    钢筋混凝土（RC）公路护栏的设计是一项安全攸关的工程任务，需要严格符合AASHTO LRFD桥梁设计规范等法规条款。当前的工程实践主要依赖人工、迭代和经验驱动的流程来满足复杂的材料、几何和力学约束。尽管独立的大型语言模型（LLMs）在知识表示和文本生成方面展现出强大能力，但其在结构工程设计中的直接应用受限于幻觉问题、数值推理错误以及与基于物理的分析的整合不足。为解决这些局限性，本研究基于AutoGen的多智能体编排能力，提出了一种用于钢筋混凝土护栏自动化设计的“生成-验证-修改”闭环框架。该框架集成了专门负责参数生成、力学分析等多项任务的专业化智能体。

    arXiv:2606.12040v3 Announce Type: replace  Abstract: The design of reinforced concrete (RC) highway barriers is a safety-critical engineering task that requires strict compliance with regulatory provisions such as the AASHTO LRFD Bridge Design Specifications. Current engineering practice relies largely on manual, iterative, and experience-driven procedures to satisfy complex material, geometric, and mechanical constraints. Although standalone large language models (LLMs) show strong capabilities in knowledge representation and text generation, their direct use in structural engineering design is limited by hallucination, numerical reasoning errors, and insufficient integration with physics-based analysis. To address these limitations, this study proposes a "generation-validation-modification" closed-loop framework for automated RC barrier design based on the multi-agent orchestration capability of AutoGen. The framework integrates specialized agents for parameter generation, mechanics-
    
[^227]: 关于聊天机器人在问题-解决方案驱动型对话中如何工作的一些假设：大语言模型作为“创新幻觉”的印证

    Some hypotheses on how chatbots work in problem-solution-driven conversations: Large Language Models as confirmation of the Innovation Illusion

    [https://arxiv.org/abs/2606.07722](https://arxiv.org/abs/2606.07722)

    该论文从聚合动力学、认知语言学、神经心理学和心理学等多学科视角分析聊天机器人在问题解决型对话中的本质，提出人类的想象与思考基于隐喻性问题传播，而大语言模型的训练文本仅能部分模仿人类思维，从而印证了“创新幻觉”。

    

    我们讨论了聊天机器人作为问题解决型对话中的对话伙伴的本质。聊天机器人能做什么，又不能做什么？我们的分析借鉴了聚合动力学、认知语言学、神经心理学和心理学等领域的见解。我们确立聊天机器人是多方面、复合式的系统。我们的论证聚焦于基础聊天机器人，以期由此对更高级聊天机器人的核心功能做出论断。基础聊天机器人被假定由一个带有简单界面的大语言模型（LLM）构成。我们分析的主要结果是：基于所谓“隐喻性问题传播”对人类的想象、理解和思考进行了描述；用于训练大语言模型的文本数据集中的文本具有特定特征，这些文本仅能部分模仿人类的思考与理解；大语言模型的训练过程编码了人工的隐喻性问题传播……

    arXiv:2606.07722v5 Announce Type: replace  Abstract: We discuss the nature of chatbots as conversation partners in problem-solving conversations. What can chatbots do and what can't they do? Our analysis draws on insights from Aggregation Dynamics, Cognitive Linguistics, Neuropsychology and Psychology.   We establish that chatbots are multifaceted and composite systems. Our argument focuses on basic chatbots in the hope of thereby making statements about the core functionality of more advanced chatbots. Basic chatbots are assumed to consist of a Large Language Model (LLM) with a simple interface.   The main results of our analysis are: a description of human imagination, understanding and thinking based on so-called metaphorical problem propagations; that the texts in text datasets used for training LLMs have specific characteristics and that these texts only partially imitate human thinking and understanding; that the LLM training process encodes artificial metaphorical problem propag
    
[^228]: 从智能体轨迹到信任：LLM智能体中证据追踪与执行溯源综述

    From Agent Traces to Trust: A Survey of Evidence Tracing and Execution Provenance in LLM Agents

    [https://arxiv.org/abs/2606.04990](https://arxiv.org/abs/2606.04990)

    本综述提出将执行溯源形式化为智能体执行的有类型图、将证据追踪视为其在证据支持关系上的投影，以此为基础构建可信LLM智能体的过程级问责框架。

    

    基于大语言模型（LLM）的智能体正在从被动的文本生成器演变为能够进行规划、工具使用、检索、记忆访问、环境交互和多智能体协作的自主系统。这些能力扩展了智能体的自主性，但也使智能体的行为更难以验证、调试和审计。仅凭最终答案的准确性无法解释输出是如何产生的、哪些证据支持了每个论断、工具调用是否合理、记忆如何影响后续决策，以及故障源于何处。本综述将证据追踪与执行溯源作为可信LLM智能体实现过程级问责的基础进行考察。我们将执行溯源定义为智能体执行过程的有类型图，将证据追踪定义为其在证据支持关系上的投影。这一视角将检索接地、论断支持、工具使用安全、记忆谱系、可观测性和调试等领域联系起来。

    arXiv:2606.04990v5 Announce Type: replace-cross  Abstract: Large language model (LLM)-based agents are evolving from passive text generators into autonomous systems capable of planning, tool use, retrieval, memory access, environmental interaction, and multi-agent collaboration. These capabilities expand agent autonomy, but also make agent behavior harder to verify, debug, and audit. Final-answer accuracy alone cannot explain how an output was produced, which evidence supported each claim, whether tool calls were justified, how memory influenced later decisions, or where failures originated. This survey examines evidence tracing and execution provenance as foundations for process-level accountability in trustworthy LLM agents. We define execution provenance as the typed graph of an agent execution and evidence tracing as its projection onto evidence-support relations. This perspective connects retrieval grounding, claim support, tool-use safety, memory lineage, observability, debugging
    
[^229]: CLSP-REQA：基于Mamba-BiLSTM与置信度门控干预的实时质量感知闭环癫痫发作预测框架

    CLSP-REQA: A Real-Time Quality-Aware Closed-Loop Seizure Prediction Framework with Mamba-BiLSTM and Confidence-Gated Intervention

    [https://arxiv.org/abs/2606.00074](https://arxiv.org/abs/2606.00074)

    该论文提出了CLSP-REQA闭环癫痫发作预测框架，通过将实时脑电信号质量评估模块与Mamba-BiLSTM预测骨干并行结合，利用质量分数经分级非线性融合函数调制输出置信度，在严格的跨患者评估下显著提升了预测的可靠性。

    

    可靠的癫痫发作预测是闭环神经刺激治疗的前提条件，然而现有方法很少考虑实际部署中所遇到的脑电信号质量的可变性，且绝大多数方法采用非严格的评估协议，从而高估了泛化性能。我们提出了CLSP-REQA（带实时脑电质量评估的闭环癫痫发作预测），这是一个统一框架，将轻量级信号质量评估器直接嵌入预测流程中。实时脑电质量评估（REQA）模块与Mamba-BiLSTM骨干网络并行运行，产生一个[0,1]区间内的标量质量分数q，并通过分级非线性融合函数（ECLO）对输出置信度进行调制。在CHB-MIT头皮脑电数据库（23名受试者，198次癫痫发作）上进行严格的跨患者评估，CLSP-REQA实现了0.7426 ± 0.0199的AUC-ROC，优于未经适配的跨患者基线方法。

    arXiv:2606.00074v2 Announce Type: replace-cross  Abstract: Reliable seizure prediction is a prerequisite for closed-loop neurostimulation therapy, yet existing methods rarely account for the variability in EEG signal quality encountered in real-world deployment, and the overwhelming majority adopt non-strict evaluation protocols that overestimate generalisation performance. We propose CLSP-REQA (Closed-Loop Seizure Prediction with Real-time EEG Quality Assessment), a unified framework that embeds a lightweight signal quality estimator directly within the prediction pipeline. A Real-time EEG Quality Assessment (REQA) module runs in parallel with a Mamba-BiLSTM backbone, producing a scalar quality score q in [0,1] that modulates output confidence through a tiered non-linear fusion function (ECLO). Under strict cross-patient evaluation on the CHB-MIT Scalp EEG Database (n = 23 subjects, 198 seizures), CLSP-REQA achieves an AUC-ROC of 0.7426 +- 0.0199, outperforming the unadapted cross-pat
    
[^230]: 分布式大语言模型智能体工作流运行时验证的因果过去时逻辑

    Causal Past Logic for Runtime Verification of Distributed LLM Agent Workflows

    [https://arxiv.org/abs/2605.20923](https://arxiv.org/abs/2605.20923)

    该论文提出因果过去时逻辑（CPL）扩展ZipperGen框架，使分布式LLM智能体工作流中的守卫条件能够仅基于因果可见的事件进行在线运行时验证，并证明了本地监控器值与守卫指称语义的一致性。

    

    我们研究分布式大语言模型（LLM）智能体工作流的运行时监控问题。在异步执行中，一项决策只能依赖于对做出该决策的生命线具有因果可见性的事件：某个日志中较早出现的事件在本地可能仍然是未知的。我们通过因果过去时逻辑（CPL）扩展了ZipperGen智能体工作流框架，CPL是PT-DTL针对if结构和while循环中守卫条件的适配版本。除了标准的过去时模态词（如previous和since）之外，守卫条件还可以检查另一条生命线上最新的因果可见事件以及存储在该处的选定变量。由事件的所有者在线评估守卫条件，以选择下一个分支或循环步骤。我们将知识向量监控器适配到ZipperGen框架中，并证明了本地计算出的监控器值与守卫条件在当前事件处的指称语义相一致。

    arXiv:2605.20923v2 Announce Type: replace-cross  Abstract: We study runtime monitoring for distributed LLM-agent workflows. In an asynchronous execution, a decision can only depend on events that are causally visible to the lifeline that makes it: an event that appears earlier in some log may still be unknown locally. We extend the ZipperGen agent-workflow framework with Causal Past Logic (CPL), an adaptation of PT-DTL to guards in if-constructs and while loops. In addition to standard past-time modalities such as previous and since, a guard can inspect the latest causally visible event of another lifeline and selected variables stored there. The owner evaluates the guard online to select the next branch or loop step. We adapt the knowledge-vector monitor to ZipperGen and prove that the locally computed monitor value coincides with the denotational semantics of the guard at the current event.
    
[^231]: 连续扩散在语言建模中与离散扩散具有同等竞争力的扩展性

    Continuous Diffusion Scales Competitively with Discrete Diffusion for Language

    [https://arxiv.org/abs/2605.18530](https://arxiv.org/abs/2605.18530)

    该研究通过将 Plaid 与现代离散扩散语言模型架构对齐构建了 RePlaid，建立了首个可媲美离散 DLM 的连续扩散语言模型缩放定律，证明连续扩散是语言建模中高度竞争且可扩展的方案。

    

    尽管扩散模型近来在语言建模领域引起了广泛关注，但连续扩散的可扩展性似乎不如离散方法。为了挑战这一观点，我们重新审视了基于似然的连续扩散语言模型 Plaid，并通过将 Plaid 的架构与现代离散 DLM 对齐，构建了 RePlaid。在这一统一设置下，我们建立了首个可与离散 DLM 相媲美的连续 DLM 缩放定律：RePlaid 与自回归模型相比仅存在约 $20\times$ 的计算差距，在使用更少参数的情况下优于 Duo，并在过度训练的情形下优于 MDLM。我们将 RePlaid 与近期的连续 DLM 进行了基准测试：在 OpenWebText 数据集上，RePlaid 在连续 DLM 中实现了新的最先进困惑度上界 $22.1$，并具有更优的生成质量。这些结果表明，通过似然训练的连续扩散是一种极具竞争力且可扩展的替代方案。

    arXiv:2605.18530v2 Announce Type: replace  Abstract: While diffusion has drawn considerable recent attention from the language modeling community, continuous diffusion has appeared less scalable than discrete approaches. To challenge this belief we revisit Plaid, a likelihood-based continuous diffusion language model (DLM), and construct RePlaid by aligning the architecture of Plaid with modern discrete DLMs. In this unified setting, we establish the first scaling law for continuous DLMs that rivals discrete DLMs: RePlaid exhibits a compute gap of only $20\times$ compared to autoregressive models, outperforms Duo while using fewer parameters, and outperforms MDLM in the over-trained regime. We benchmark RePlaid against recent continuous DLMs: on OpenWebText, RePlaid achieves a new state-of-the-art PPL bound of $22.1$ among continuous DLMs and superior generation quality. These results suggest that continuous diffusion, when trained via likelihood, is a highly competitive and scalable a
    
[^232]: 面向高维一阶哈密顿-雅可比-贝尔曼方程的单调神经策略迭代方法

    Monotone Neural Policy Iteration for High-Dimensional First-Order Hamilton--Jacobi--Bellman Equations

    [https://arxiv.org/abs/2605.07116](https://arxiv.org/abs/2605.07116)

    本文提出一种单调神经策略迭代方法，通过中心差分构造单调算子并借助一致化技术将冻结策略算子转化为马尔可夫链生成器，从而在无需张量网格的情况下为高维一阶HJB方程提供了全空间适定性、显式依赖域界和考虑模型误差的后验评估界。

    

    我们分析了一种适用于已知或学习到的动力学的高维一阶哈密顿-雅可比-贝尔曼（HJB）方程的神经半离散方法。中心差分与人工黏性 $Nh=O(h)$ 定义了一个单调算子，该算子通过 $2d+1$ 次平移网络查询进行求值；策略迭代无需张量网格即可求解所得的贝尔曼方程。在固定的 $h$ 下，尖锐的分量条件 $\max_i|f_i|\le2N$ 将每个冻结策略算子转化为具有与策略无关的总跳跃率的最近邻马尔可夫链生成器。一致化技术给出了针对可测反馈的全空间适定性、数值依赖域上的显式泊松尾界以及无边界局部化。该表示还产生了能够考虑残差误差和学习模型误差的后验策略评估界。贪心间隙分析控制了固定 $h$ 下的不精确策略迭代；另一部分一致性估计……

    arXiv:2605.07116v2 Announce Type: replace  Abstract: We analyze a neural semi-discrete method for high-dimensional first-order Hamilton-Jacobi-Bellman (HJB) equations with known or learned dynamics. Centered differences and an artificial viscosity $Nh=O(h)$ define a monotone operator evaluated through $2d+1$ shifted network queries; policy iteration solves the resulting Bellman equation without a tensor grid. At fixed $h$, the sharp componentwise condition $\max_i|f_i|\le2N$ turns every frozen-policy operator into a nearest-neighbor Markov-chain generator with a policy-independent total jump rate. Uniformization gives whole-space well-posedness for measurable feedbacks, an explicit Poisson-tail bound on the numerical domain of dependence, and boundary-free localization. The representation also yields a posteriori policy-evaluation bounds that account for residual and learned-model errors. A greedy-gap analysis controls inexact policy iteration at fixed $h$; a separate consistency estim
    
[^233]: 分析大语言模型推理以揭示心理健康污名化

    Analyzing LLM Reasoning to Uncover Mental Health Stigma

    [https://arxiv.org/abs/2604.25053](https://arxiv.org/abs/2604.25053)

    该论文提出通过分析大语言模型的中间推理步骤，并借助临床专业知识对污名化语言进行分类与严重程度评级，从而揭示多项选择题评估方法无法捕捉的心理健康污名化偏见及其内在逻辑。

    

    arXiv:2604.25053v2 公告类型：替换 摘要：尽管大语言模型（LLM）在心理健康应用领域正被日益广泛地探索，但近期研究表明，它们可能对心理疾病患者表现出污名化倾向。现有的污名化评估主要依赖多项选择题（MCQ），这种方式无法捕捉模型底层逻辑中蕴含的偏见。在本文中，我们通过分析大语言模型的中间推理步骤，来揭示隐藏的污名化语言及其背后的内在理由。我们借助临床专业知识，对针对心理疾病患者的常见污名化语言模式进行分类，并利用这一框架来识别和标记大语言模型推理中的问题性陈述。此外，我们对这些陈述的严重程度进行评级，以区分明显的偏见与更微妙、危害性不那么直接的偏见。为了扩展推理领域并捕捉更广泛的语言模式……

    arXiv:2604.25053v2 Announce Type: replace  Abstract: While large language models (LLMs) are increasingly being explored for mental health applications, recent studies reveal that they can exhibit stigma toward individuals with psychological conditions. Existing evaluations of this stigma primarily rely on multiple-choice questions (MCQs), which fail to capture the biases embedded within the models' underlying logic. In this paper, we analyze the intermediate reasoning steps of LLMs to uncover hidden stigmatizing language and the internal rationales driving it. We leverage clinical expertise to categorize common patterns of stigmatizing language directed at individuals with psychological conditions and use this framework to identify and tag problematic statements in LLM reasoning. Furthermore, we rate the severity of these statements, distinguishing between overt prejudice and more subtle, less immediately harmful biases. To broaden the reasoning domain and capture a wider array of patt
    
[^234]: 摇动即走！面向零样本动态绳索操作的系统辨识

    Wiggle and Go! System Identification for Zero-Shot Dynamic Rope Manipulation

    [https://arxiv.org/abs/2604.22102](https://arxiv.org/abs/2604.22102)

    提出"摇动即走"两阶段框架，仅需观察一次短暂安全的摇动动作即可辨识绳索系统参数，无需大量真实数据或迭代优化，即可实现零样本的动态绳索目标击打与多目标抛掷悬挂操作。

    

    许多机器人任务是不容许失误的；动态投掷中的一次错误可能导致不可接受的延迟或无法恢复的失败。我们介绍了 Wiggle and Go!，一个用于零样本绳索操作的两阶段框架：通过观察一个简短、安全的摇动动作来预测描述性的绳索参数，然后这些参数为轨迹优化器提供条件，从而实现零样本的目标条件执行。与先前需要大量真实世界数据集或迭代真实世界优化的动态绳索操作方法不同，我们的辨识模块是任务无关的，无需重新训练即可支持多样化的操作策略。在真实世界中，利用绳索系统参数，我们在3D目标击打任务上实现了3.55厘米的平均精度，而未提供参数信息的基线方法为15.29厘米，并且在多目标抛掷和悬挂任务上成功率超过50%。预测参数可迁移到未见过的运动中，模拟与真实绳索动力学之间的皮尔逊相关系数达到0.95。

    arXiv:2604.22102v2 Announce Type: replace-cross  Abstract: Many robotic tasks are unforgiving; a single mistake in a dynamic throw can lead to unacceptable delays or unrecoverable failure. We introduce Wiggle and Go!, a two-stage framework for zero-shot rope manipulation: a brief, safe wiggle action is observed to predict descriptive rope parameters, which then conditions a trajectory optimizer for zero-shot goal-conditioned execution. Unlike prior dynamic rope manipulation methods that require large real-world datasets or iterative real-world refinement, our identification module is task-agnostic, supporting diverse manipulation policies without retraining. We achieve a 3.55\,cm average accuracy on 3D target striking in real using rope system parameters in comparison to 15.29\,cm for uninformed baselines, and over 50\% success on multi-objective lobbing and draping tasks. Predicted parameters transfer to unseen motions with 0.95 Pearson correlation between simulated and real rope dyna
    
[^235]: EvoMaster：一个面向大规模智能体科学的基础性进化智能体框架

    EvoMaster: A Foundational Evolving Agent Framework for Agentic Science at Scale

    [https://arxiv.org/abs/2604.17406](https://arxiv.org/abs/2604.17406)

    EvoMaster 提出通过执行、探索、进化三个嵌套循环实现外部证据持久化与经验跨研究积累的基础性进化智能体框架，解决了智能体科学中的循环碎片化与循环不连续问题，在十个基准测试中以 58.02% 的平均得分取得最佳成绩。

    

    大语言模型与智能体的融合正在催化科学发现的新时代：智能体科学。然而，通用的智能体基础设施在各科学领域被反复重建（循环碎片化），且有用的证据和经验在长期研究中不断丢失（循环不连续性），这为大规模智能体科学的发展带来了障碍。我们提出了 EvoMaster，一个面向大规模智能体科学的基础性进化智能体框架。EvoMaster 通过实施“循环研究”来解决循环碎片化和循环不连续性问题，其中外部证据得以持久保存并用于改进后续决策。通过三个嵌套循环——执行、探索与进化（E³），循环研究将单次运行内、跨实验以及跨研究的工作连接起来。在涵盖科学研究、编程和推理的十个基准测试中，EvoMaster 在使用 GPT-5.4 的四个智能体中取得了最佳成绩，平均得分达到 58.02%。

    arXiv:2604.17406v5 Announce Type: replace  Abstract: The convergence of large language models and agents is catalyzing a new era of scientific discovery: Agentic Science. However, common agent infrastructure is repeatedly rebuilt across scientific fields (loop fragmentation) and useful evidence and experience are lost in long-horizon research (loop discontinuity), bringing obstacles to Agentic Science at Scale. We introduce EvoMaster, a foundational evolving agent framework for Agentic Science at Scale. EvoMaster handles loop fragmentation and loop discontinuity by implementing Loop Research in which external evidence persists and improves later decisions. Through three nested loops, Execution, Exploration and Evolution (E$^3$), loop research connects research within runs, across experiments and across studies. Across ten benchmarks spanning scientific research, coding and reasoning, EvoMaster achieves the best score among four agents using GPT-5.4, reaching a mean score of 58.02%, and
    
[^236]: 学会像漫画配文作者一样思考：面向多模态幽默理解的不协调性-消解监督

    Learning to Think Like a Cartoon Captionist: Incongruity-Resolution Supervision for Multimodal Humor Understanding

    [https://arxiv.org/abs/2604.15210](https://arxiv.org/abs/2604.15210)

    提出IRS框架，将多模态幽默理解分解为不协调性建模、消解建模和偏好对齐三个环节，通过结构化推理轨迹监督，使模型像专业漫画配文作者一样进行显式推理。

    

    幽默是少数几种推理过程的正确性与答案的正确性同等重要的认知任务之一。尽管近期研究在纽约客漫画配文大赛（NYCC）等基准上评估幽默理解能力，但大多数工作将其视为黑盒预测，忽视了幽默理解背后结构化的推理过程。我们提出了IRS（不协调性-消解监督）框架，该框架将幽默理解分解为三个组成部分：不协调性建模，用于识别视觉场景中的不匹配之处；消解建模，用于对这些不匹配构建连贯的重新解释；偏好对齐，用于依据人类判断评估候选解释。IRS植根于不协调性-消解理论和专业配文作者的实践，通过结构化轨迹监督中间推理过程，使从视觉感知到幽默解释的路径……

    arXiv:2604.15210v2 Announce Type: replace-cross  Abstract: Humor is one of the few cognitive tasks where getting the reasoning right matters as much as getting the answer right. While recent work evaluates humor understanding on benchmarks such as the New Yorker Cartoon Caption Contest (NYCC), it largely treats it as black-box prediction, overlooking the structured reasoning processes underlying humor comprehension. We introduce IRS (Incongruity-Resolution Supervision), a framework that decomposes humor understanding into three components: Incongruity Modeling, which identifies mismatches in the visual scene; Resolution Modeling, which constructs coherent reinterpretations of these mismatches; and Preference Alignment, which evaluates candidate interpretations under human judgments. Grounded in incongruity-resolution theory and expert captionist practice, IRS supervises intermediate reasoning process through structured traces that make the path from visual perception to humorous interp
    
[^237]: 迈向太阳能电池板完整性自动化：用于先进表面缺陷识别的混合深度特征提取

    Towards Automated Solar Panel Integrity: Hybrid Deep Feature Extraction for Advanced Surface Defect Identification

    [https://arxiv.org/abs/2604.10969](https://arxiv.org/abs/2604.10969)

    该论文提出了一种结合LBP、HoG、Gabor滤波器等手工特征与DenseNet-169深度特征的混合特征提取方法，用于太阳能电池板表面缺陷的自动化智能检测。

    

    为确保能源效率和可靠运行，对发电厂中的太阳能电池板进行监测以检测缺陷至关重要。人工监测大型太阳能发电厂以及安装在偏远地区的太阳能发电厂非常费力、耗时且成本高昂。人工检测还容易出现人为错误。因此，有必要创建一个自动化、智能化的缺陷检测系统，以确保持续监测、早期故障检测和最大化的发电量。我们提出了一种新颖的混合方法，通过结合手工特征和深度学习特征来检测太阳能电池板的缺陷。使用局部二值模式（LBP）、方向梯度直方图（HoG）和Gabor滤波器进行手工特征提取，并利用DenseNet-169提取深度特征。将手工特征与深度特征拼接后，输入三种不同类型的分类器中进行处理。

    arXiv:2604.10969v2 Announce Type: replace-cross  Abstract: To ensure energy efficiency and reliable operations, it is essential to monitor solar panels in generation plants to detect defects. It is quite labor-intensive, time consuming and costly to manually monitor large-scale solar plants and those installed in remote areas. Manual inspection may also be susceptible to human errors. Consequently, it is necessary to create an automated, intelligent defect-detection system, that ensures continuous monitoring, early fault detection, and maximum power generation. We proposed a novel hybrid method for defect detection in SOLAR plates by combining both handcrafted and deep learning features. Local Binary Pattern (LBP), Histogram of Gradients (HoG) and Gabor Filters were used for the extraction of handcrafted features. Deep features extracted by leveraging the use of DenseNet-169. Both handcrafted and deep features were concatenated and then fed to three distinct types of classifiers, inclu
    
[^238]: VeriSim：一个用于在患者沟通噪声下对医疗AI进行压力测试的可配置框架

    VeriSim: A Configurable Framework for Stress-Testing Medical AI Under Patient Communication Noise

    [https://arxiv.org/abs/2604.10441](https://arxiv.org/abs/2604.10441)

    VeriSim框架通过在六个临床沟通维度上注入可控噪声来模拟真实患者沟通方式，揭示了医疗大语言模型在噪声干扰下诊断准确率下降15-25个百分点、且小参数模型退化更为严重的脆弱性。

    

    医疗大型语言模型通常在理想化的患者病例上进行评估，而这些病例并不能反映真实患者的沟通方式。我们提出了VeriSim，这是一个患者模拟框架，它在六个基于临床的沟通维度上注入可控噪声，同时在很大程度上保留每位患者的医疗记录。真实性一致性由一个验证器支持，该验证器从每条候选话语中提取原子化声明，并根据使用BioLORD嵌入构建的基于UMLS的向量索引对其进行判断，利用检索到的原子声明的结构化临床元数据（例如药物类别、解剖部位、治疗-疾病关系），而不仅仅依赖表面文本相似性。在七个开源权重大语言模型上的实验表明，真实的噪声使诊断准确率降低了15-25个百分点，并使对话长度增加了34-55%；7-8B参数模型的性能退化程度是70B以上模型的1.4倍。一位委员会认证的医师和一位持证护士对VeriSim的（原文在此处截断）

    arXiv:2604.10441v2 Announce Type: replace  Abstract: Medical large language models are typically evaluated on idealized patient cases that do not reflect how real patients communicate. We introduce VeriSim, a patient simulation framework that injects controllable noise along six clinically grounded communication dimensions while substantially preserving each patient's medical record. Truth adherence is supported by a verifier that extracts atomic claims from each candidate utterance and judges them against a UMLS-grounded vector index built with BioLORD embeddings, using the retrieved atoms' structured clinical metadata (e.g., drug class, anatomical site, treats-condition relations) rather than surface-text similarity alone. Across seven open-weight LLMs, realistic noise reduces diagnostic accuracy by 15-25 percentage points and increases conversation length by 34-55%; the 7-8B models degrade 1.4x more than 70B+ models. A board-certified physician and a licensed nurse rate VeriSim's co
    
[^239]: 大语言模型如何遵循指令：是技能化的协调，而非通用机制

    How LLMs Follow Instructions: Skillful Coordination, Not a Universal Mechanism

    [https://arxiv.org/abs/2604.06015](https://arxiv.org/abs/2604.06015)

    研究通过探测与消融实验证明，大语言模型的指令遵循能力并非依赖单一通用机制，而是通过部分共享、按技能相似性聚类组合的结构化表征进行技能化协调来实现的。

    

    指令微调通常被认为赋予语言模型一种跨领域的通用指令遵循能力，然而其底层机制仍知之甚少。指令遵循究竟是依赖一个通用机制，还是组合式的技能部署？我们通过在三个指令微调模型上对九个多样化任务进行诊断性探测来研究这一问题。我们的分析提供了反对通用机制的汇聚性证据。首先，在所有任务上训练的通用探针相对于任务特定的专家探针表现出选择性而非均匀的缺陷，表明表征共享是部分的、结构化的，而非全局性的。其次，跨任务迁移很弱，且按技能相似性聚类。第三，因果消融实验揭示了稀疏的不对称依赖关系，而非共享表征。各任务还按复杂性在模型不同层间呈现分层现象，结构约束出现较早，而语义任务在（后续层中）显现。

    arXiv:2604.06015v2 Announce Type: replace  Abstract: Instruction tuning is commonly assumed to endow language models with a domain-general ability to follow instructions, yet the underlying mechanism remains poorly understood. Does instruction-following rely on a universal mechanism or compositional skill deployment? We investigate this through diagnostic probing across nine diverse tasks in three instruction-tuned models. Our analysis provides converging evidence against a universal mechanism. First, general probes trained across all tasks show selective rather than uniform deficits relative to task-specific specialists, indicating that representational sharing is partial and structured rather than global. Second, cross-task transfer is weak and clustered by skill similarity. Third, causal ablation reveals sparse asymmetric dependencies rather than shared representations. Tasks also stratify by complexity across layers, with structural constraints emerging early and semantic tasks eme
    
[^240]: 四代量子生物医学传感器

    Four Generations of Quantum Biomedical Sensors

    [https://arxiv.org/abs/2603.29944](https://arxiv.org/abs/2603.29944)

    本文提出了一个基于量子资源利用的四代量子生物医学传感器统一分类框架，并创新性地定义了将量子传感与量子学习及变分电路端到端集成、可在量子域内直接实现自适应推理的第四代量子传感器。

    

    量子传感技术为超灵敏生物医学传感提供了变革性潜力，然而其临床转化仍受到经典噪声极限以及对宏观系综依赖的制约。我们提出了一个统一的代际框架，基于量子生物传感器对量子资源的利用来梳理其不断演进的发展格局。第一代设备利用离散能级进行信号转换，但遵循经典标度定律；第二代传感器利用量子相干性，使精度随相干时间得以提升，最高达到标准量子极限；第三代架构则采用纠缠和自旋压缩技术以逼近海森堡极限精度。我们定义了正在兴起的第四代量子传感器，其特征在于量子传感与量子学习及变分电路的端到端集成，能够在量子域内直接实现自适应推理。

    arXiv:2603.29944v3 Announce Type: replace-cross  Abstract: Quantum sensing technologies offer transformative potential for ultra-sensitive biomedical sensing, yet their clinical translation remains constrained by classical noise limits and a reliance on macroscopic ensembles. We propose a unifying generational framework to organize the evolving landscape of quantum biosensors based on their utilization of quantum resources. First-generation devices utilize discrete energy levels for signal transduction but follow classical scaling laws. Second-generation sensors exploit quantum coherence, extending precision with the coherence time up to the standard quantum limit, while third-generation architectures employ entanglement and spin squeezing to approach Heisenberg-limited precision. We define an emerging fourth generation characterized by the end-to-end integration of quantum sensing with quantum learning and variational circuits, enabling adaptive inference directly within the quantum d
    
[^241]: HISA：面向细粒度稀疏注意力的高效分层索引

    HISA: Efficient Hierarchical Indexing for Fine-Grained Sparse Attention

    [https://arxiv.org/abs/2603.28458](https://arxiv.org/abs/2603.28458)

    HISA提出了一种即插即用的两阶段分层索引方法，先通过块级粗过滤丢弃无关区域，再在候选块内进行token级精炼，从而消除细粒度稀疏注意力中索引器随上下文长度增长的每层扫描瓶颈。

    

    以DeepSeek稀疏注意力（DSA）为代表的token级稀疏注意力机制，通过轻量级索引器为每个查询对所有历史键进行评分，实现细粒度的键选择，随后仅在选定的子集上计算注意力。虽然下游的稀疏注意力本身具有良好的可扩展性，但索引器仍必须为每个查询扫描整个前缀，这引入了随上下文长度增长而变得极其昂贵的每层瓶颈。我们提出了HISA（分层索引稀疏注意力），这是一种即插即用的索引器替代方案，它将搜索路径从扁平的token扫描改写为两阶段分层流程：（1）块级粗过滤阶段，通过对池化的块表示进行评分来丢弃无关区域；（2）token级精炼阶段，仅在保留的候选块内应用原始索引器。HISA保持了相同的token级top-（摘要原文截断）

    arXiv:2603.28458v4 Announce Type: replace  Abstract: Token-level sparse attention mechanisms, exemplified by DeepSeek Sparse Attention (DSA), achieve fine-grained key selection by scoring every historical key for each query through a lightweight indexer, then computing attention only on the selected subset. While the downstream sparse attention itself scales favorably, the indexer must still scan the entire prefix for every query, introducing an per-layer bottleneck that grows prohibitively with context length. We propose HISA (Hierarchical Indexed Sparse Attention), a plug-and-play replacement for the indexer that rewrites the search path from a flat token scan into a two-stage hierarchical procedure: (1) a block-level coarse filtering stage that scores pooled block representations to discard irrelevant regions, followed by (2) a token-level refinement stage that applies the original indexer exclusively within the retained candidate blocks. HISA preserves the identical token-level top
    
[^242]: 使用专用实例分割模型进行边界检测与分类的自动化多类别伤口评估

    Automated multi-class wound assessment using dedicated instance segmentation models for boundary detection and classification

    [https://arxiv.org/abs/2603.27325](https://arxiv.org/abs/2603.27325)

    本研究基于YOLOv11开发了两个专用实例分割模型，能够对烧伤、压力性损伤、糖尿病足溃疡等五种临床相关伤口类型同时实现边界分割与分类，显著提升了AI伤口评估的临床适用性。

    

    准确的伤口分类（WC）和边界分割对于指导慢性和急性伤口管理中的临床决策至关重要。然而，现有的大多数人工智能（AI）模型存在局限性，它们往往仅关注少数伤口类型、伤口严重程度变化有限，或只执行单一任务（分割或分类），这降低了其临床适用性。本研究提出了两个基于YOLOv11的专用实例分割模型，可对五种临床相关伤口类型进行伤口边界分割（WBS）和伤口分类（WC），包括烧伤（BI）、压力性损伤、糖尿病足溃疡、血管性溃疡和手术伤口。研究创建了一个包含2,963张标注图像的伤口类型平衡数据集，采用五折交叉验证来训练这两个任务的模型。在原始未增强数据集上训练的模型在各折中表现稳定，尽管烧伤检测的准确率……

    arXiv:2603.27325v2 Announce Type: replace-cross  Abstract: Accurate wound classification (WC) and boundary segmentation are essential for guiding clinical decisions in chronic and acute wound management. However, most existing artificial intelligence (AI) models are limited, focusing on a narrow set of wound types, limited variations in wound severity, or a single task (segmentation or classification), which reduces their clinical applicability. This study presents two dedicated instance segmentation models based on You Only Look Once (YOLO)v11 that perform wound boundary segmentation (WBS) and WC across five clinically relevant wound types: burn injury (BI), pressure injury, diabetic foot ulcer, vascular ulcer, and surgical wound. A wound-type balanced dataset of 2,963 annotated images was created to train the models for both tasks, using five-fold cross-validation. Models trained on the original, non-augmented dataset performed consistently across folds, though BI detection accuracy 
    
[^243]: 扰动：一种用于语言模型表示学习的简单高效的对抗性追踪方法

    Perturbation: A simple and efficient adversarial tracer for representation learning in language models

    [https://arxiv.org/abs/2603.23821](https://arxiv.org/abs/2603.23821)

    该论文提出了一种简单高效的对抗性扰动方法来追踪语言模型的表示学习，通过在单个对抗样本上微调模型并测量扰动对其他样本的“感染”程度来揭示表示，该方法无需几何假设，且能避免在未训练模型中产生虚假表示。

    

    深度神经语言模型（LM）中的语言表示学习已被研究数十年，但在语言模型中寻找表示仍然是一个未解决的问题。一方面，无约束的对齐可能会使表示的概念变得平凡化（Sutter等人，2025）；另一方面，即使是最近流行的线性方法也可能不总是忠实于模型的自然行为（Arora等人，2024）。在这里，我们通过将表示重新概念化为学习的传导通道而非激活模式，来摆脱这一困境。我们的方法很简单：我们通过在单个对抗样本上微调语言模型来扰动它，并测量这种扰动如何“感染”其他样本。扰动方法不做任何几何假设，并且与其他方法不同，它不会在不应该存在表示的地方找到表示（例如，在未经训练的语言模型中）。但在经过训练的语言模型中，扰动方法揭示了多个语言粒度上的结构化迁移，表明语言模型……

    arXiv:2603.23821v2 Announce Type: replace  Abstract: Linguistic representation learning in deep neural language models (LMs) has been studied for decades, but finding representations in LMs remains an unsolved problem. On the one hand, unconstrained alignments may trivialize the notion of representation (Sutter et al., 2025); on the other, even recently popularized linear approaches may not always be faithful to natural model behavior (Arora et al. 2024). Here we escape this dilemma by reconceptualizing representations not as patterns of activation but as conduits for learning. Our approach is simple: we perturb an LM by fine-tuning it on a single adversarial example and measure how this perturbation "infects" other examples. Perturbation makes no geometric assumptions, and unlike other methods, it does not find representations where it should not (e.g., in untrained LMs). But in trained LMs, perturbation reveals structured transfer at multiple linguistic grain sizes, suggesting that L
    
[^244]: 域弹性变换：面向高维科学数据的贝叶斯函数配准

    Domain Elastic Transform: Bayesian Function Registration for High-Dimensional Scientific Data

    [https://arxiv.org/abs/2603.21235](https://arxiv.org/abs/2603.21235)

    该论文提出域弹性变换（DET），一种无网格的贝叶斯概率框架，通过联合空间-函数似然引导的弹性变形建模，在完全无监督的条件下直接对齐不规则稀疏流形上高维科学数据（如空间转录组学基因表达）的几何与功能信号，无需分箱或体素化处理。

    

    非刚性配准传统上分为点集配准（对齐稀疏几何结构）和图像配准（对齐规则网格上的连续强度场）。这种二分法对于新兴科学数据（如空间转录组学）具有局限性，因为这类数据中，高维向量值函数（如基因表达）定义在不规则的稀疏流形上。因此，研究人员要么必须通过体素化牺牲单细胞分辨率，要么为了几何对齐而忽略功能信号。我们提出了域弹性变换（DET），这是一个无网格的概率框架，可以联合对齐几何与函数。通过将数据视为不规则域上的函数，DET无需分箱即可直接配准高维信号。在广义贝叶斯框架下，域变形被建模为由联合空间-函数似然引导的弹性运动。DET是完全无监督的。

    arXiv:2603.21235v2 Announce Type: replace  Abstract: Nonrigid registration is conventionally divided into point set registration, which aligns sparse geometries, and image registration, which aligns continuous intensity fields on regular grids. This dichotomy is limiting for emerging scientific data such as spatial transcriptomics, where high-dimensional vector-valued functions, e.g., gene expression, are defined on irregular sparse manifolds. Researchers must therefore either sacrifice single-cell resolution through voxelization or ignore functional signals in favor of geometric alignment.   We propose Domain Elastic Transform (DET), a grid-free probabilistic framework that jointly aligns geometry and function. By treating data as functions on irregular domains, DET registers high-dimensional signals directly without binning. Within a generalized Bayesian formulation, domain deformation is modeled as elastic motion guided by a joint spatial-functional likelihood. DET is fully unsuperv
    
[^245]: OpenResearcher：面向长程深度研究轨迹合成的完全开源流水线

    OpenResearcher: A Fully Open Pipeline for Long-Horizon Deep Research Trajectory Synthesis

    [https://arxiv.org/abs/2603.20278](https://arxiv.org/abs/2603.20278)

    OpenResearcher提出了一个完全开源、可复现的离线轨迹合成流水线，在1500万文档语料库上合成超过97K条长程深度研究轨迹，微调后的模型在BrowseComp-Plus上相比基础模型提升34个百分点。

    

    训练深度研究智能体需要长程轨迹，这些轨迹交织着搜索、证据聚合与多步推理。然而，现有的数据收集流水线通常依赖专有的网络API，使得大规模轨迹合成成本高昂、不稳定且难以复现。我们提出了OpenResearcher，一个可复现的流水线，它将一次性的语料库构建与多轮轨迹合成解耦，并使用三个显式的浏览器原语（搜索、打开、查找）在1500万文档的语料库上完全离线地执行搜索与浏览循环。以GPT-OSS-120B作为教师模型，我们合成了超过97K条轨迹，其中包括大量包含100+次工具调用的长程尾部轨迹。在这些轨迹上对30B-A3B骨干模型进行监督微调后，在BrowseComp-Plus基准上达到54.8%的准确率，相比基础模型提升34.0个百分点，同时在BrowseComp、GAIA等基准上保持竞争力（原文此处截断）。

    arXiv:2603.20278v2 Announce Type: replace-cross  Abstract: Training deep research agents requires long-horizon trajectories that interleave search, evidence aggregation, and multi-step reasoning. However, existing data collection pipelines typically rely on proprietary web APIs, making large-scale trajectory synthesis costly, unstable, and difficult to reproduce. We present OpenResearcher, a reproducible pipeline that decouples one-time corpus bootstrapping from multi-turn trajectory synthesis and executes the search-and-browse loop entirely offline using three explicit browser primitives: search, open, and find, over a 15M-document corpus. Using GPT-OSS-120B as the teacher model, we synthesize over 97K trajectories, including a substantial long-horizon tail with 100+ tool calls. Supervised fine-tuning a 30B-A3B backbone on these trajectories achieves 54.8\% accuracy on BrowseComp-Plus, a +34.0 point improvement over the base model, while remaining competitive on BrowseComp, GAIA, and 
    
[^246]: 人机系统中的认知放大与认知委托：一个度量框架

    Cognitive Amplification vs Cognitive Delegation in Human-AI Systems: A Metric Framework

    [https://arxiv.org/abs/2603.18677](https://arxiv.org/abs/2603.18677)

    本文提出了一个包含四个量化指标（认知放大指数、依赖比率、人类依赖指数、认知漂移率）的度量框架，用以区分人机系统中的“认知放大”与“认知委托”，并通过基于智能体的仿真验证了该框架识别正向协作增益是否可恢复的有效性。

    

    人工智能日益深入地嵌入人类决策过程，但如何区分真正放大人类认知的系统与助长过度依赖的系统，仍然缺乏清晰界定。本文提出了一个框架，用于区分“认知放大”（在不削弱人类能力的前提下提升混合系统表现）与“认知委托”（将推理外包给人工智能）。我们定义了四个度量指标：认知放大指数（CAI*）、依赖比率（D）、人类依赖指数（HRI）和人类认知漂移率（HCDR）。我们在NetLogo基于智能体的仿真中，针对三种依赖情境和多种依赖-能力萎缩配置对该框架进行了测试，通过约束优化和参数扫描来判断正向协作增益是否可以恢复。最后，我们引入了一个包含显式人机交互项的扩展模型。我们的度量指标能够有效区分退化行为……（原文摘要在此截断）

    arXiv:2603.18677v4 Announce Type: replace-cross  Abstract: Artificial intelligence is increasingly embedded in human decision-making, yet distinguishing systems that genuinely amplify human cognition from those promoting excessive dependence remains underdefined. This paper introduces a framework to distinguish cognitive amplification (improving hybrid performance without degrading human capability) from cognitive delegation (outsourcing reasoning to the AI).   We define four metrics: the Cognitive Amplification Index (CAI*), Dependency Ratio (D), Human Reliance Index (HRI), and Human Cognitive Drift Rate (HCDR). We test this framework in an agent-based NetLogo simulation across three reliance regimes and multiple dependency-atrophy configurations, performing constrained optimizations and parameter sweeps to determine if positive collaborative gain is recoverable. Finally, we introduce an extension with an explicit human-AI interaction term.   Our metrics effectively distinguish degene
    
[^247]: TRUST-SQL：面向未知模式的工具集成多轮强化学习文本到SQL方法

    TRUST-SQL: Tool-Integrated Multi-Turn Reinforcement Learning for Text-to-SQL over Unknown Schemas

    [https://arxiv.org/abs/2603.16448](https://arxiv.org/abs/2603.16448)

    该论文提出TRUST-SQL框架，将未知模式下的文本到SQL任务形式化为部分可观测马尔可夫决策过程，通过结构化四阶段协议和双轨GRPO强化学习策略解决信用分配问题，使智能体能够主动识别并验证相关数据库模式，取得9.9%的相对性能提升。

    

    文本到SQL解析在全模式假设下已取得显著进展。然而，这一前提在现实的企业环境中并不成立，因为现实数据库中往往包含数百个表以及大量嘈杂的元数据。智能体不应预先注入完整模式，而必须主动识别并验证仅与任务相关的子集，这便引出了本工作所研究的“未知模式”场景。为解决这一问题，我们提出了TRUST-SQL（基于工具的未知模式真实推理）。我们将该任务形式化为部分可观测马尔可夫决策过程（POMDP），我们的自主智能体采用结构化的四阶段协议，将推理建立在经验证的元数据之上。至关重要的是，该协议为我们新颖的双轨GRPO策略提供了结构性边界。通过应用token级掩码优势，该策略将探索奖励与执行结果隔离开来解决信用分配问题，从而带来9.9%的相对性能提升。

    arXiv:2603.16448v3 Announce Type: replace  Abstract: Text-to-SQL parsing has achieved remarkable progress under the Full Schema Assumption. However, this premise fails in real-world enterprise environments where databases contain hundreds of tables with massive noisy metadata. Rather than injecting the full schema upfront, an agent must actively identify and verify only the relevant subset, giving rise to the Unknown Schema scenario we study in this work. To address this, we propose TRUST-SQL (Truthful Reasoning with Unknown Schema via Tools). We formulate the task as a Partially Observable Markov Decision Process where our autonomous agent employs a structured four-phase protocol to ground reasoning in verified metadata. Crucially, this protocol provides a structural boundary for our novel Dual-Track GRPO strategy. By applying token-level masked advantages, this strategy isolates exploration rewards from execution outcomes to resolve credit assignment, yielding a 9.9% relative improve
    
[^248]: 面向PETSc中AI生成科学代码的智能体评估框架

    An Agentic Evaluation Framework for AI-Generated Scientific Code in PETSc

    [https://arxiv.org/abs/2603.15976](https://arxiv.org/abs/2603.15976)

    提出PETSCAgent-Bench——一个包含14个评估器、覆盖正确性、性能、代码质量、算法适用性和库惯例五个维度的智能体评估框架，用于评估AI生成的科学代码能否像专家一样使用PETSc等生产级HPC库。

    

    尽管大语言模型已经加速了科学代码的生成，但全面评估生成的代码仍然具有挑战性。许多基准测试侧重于功能正确性或任务完成度，这对于基于生产级高性能计算（HPC）库构建的代码来说是不够的，因为在这些场景中，求解器选择、API使用惯例、内存管理、并行意识和性能同样重要。我们提出了PETSCAgent-Bench，这是一个多维基准测试与基于智能体的评估框架，用于评估AI生成的科学代码是否能像专家一样使用生产级HPC库。一个工具增强的评估器负责编译、执行和测量代码，并将确定性检查与基于LLM的评估相结合，形成包含14个评估器的流水线，涵盖五个类别：正确性、性能、代码质量、算法适用性和库特定使用惯例。A2A和MCP技术实现了对兼容编码智能体的黑盒评估。在真实的PETSc问题上，前沿模型……（原文摘要在此处截断）

    arXiv:2603.15976v2 Announce Type: replace  Abstract: While LLMs have accelerated scientific code generation, comprehensively evaluating generated code remains challenging. Many benchmarks emphasize functional correctness or task completion, which is insufficient for code built on production HPC libraries, where solver selection, API conventions, memory management, parallel awareness, and performance also matter. We introduce PETSCAgent-Bench, a multidimensional benchmark and agent-based framework for assessing whether AI-generated scientific code uses a production HPC library as an expert would. A tool-augmented evaluator compiles, executes, and measures code and combines deterministic checks with LLM-based assessments in a 14-evaluator pipeline spanning five categories: correctness, performance, code quality, algorithmic appropriateness, and library-specific conventions. A2A and MCP enable black-box evaluation of compatible coding agents. Across realistic PETSc problems, frontier mode
    
[^249]: 重新标定置信度：量表设计揭示的大语言模型元认知

    Rescaling Confidence: What Scale Design Reveals About LLM Metacognition

    [https://arxiv.org/abs/2603.09309](https://arxiv.org/abs/2603.09309)

    大语言模型的置信度量表设计并非中立选择——研究发现0–20量表能持续提升元认知敏感性，而模型对整数值存在强烈偏好，表明量表设计直接影响LLM不确定性估计的质量。

    

    言语化置信度，即大语言模型报告一个数值确定性分数，被广泛用于黑盒设置下的不确定性估计，然而置信度量表本身（通常为0–100）却很少被检验。我们证明这一设计选择并非中立的。在六个大语言模型和三个数据集上，言语化置信度呈现高度离散化，超过78%的响应集中在仅仅三个整数值上。为了探究这一现象，我们系统地沿三个维度对置信度量表进行操纵：粒度、边界位置和范围规律性，并使用meta-d'评估元认知敏感性。我们发现，0–20量表相比标准的0–100格式能够持续提升元认知效率，而边界压缩会降低性能，且即使在不规则范围下，对整数值的偏好依然存在。这些结果表明，置信度量表的设计直接影响……

    arXiv:2603.09309v3 Announce Type: replace  Abstract: Verbalized confidence, in which LLMs report a numerical certainty score, is widely used to estimate uncertainty in black-box settings, yet the confidence scale itself (typically 0--100) is rarely examined. We show that this design choice is not neutral. Across six LLMs and three datasets, verbalized confidence is heavily discretized, with more than 78\% of responses concentrating on just three round-number values. To investigate this phenomenon, we systematically manipulate confidence scales along three dimensions: granularity, boundary placement, and range regularity, and evaluate metacognitive sensitivity using $meta\text{-}d'$. We find that a 0--20 scale consistently improves metacognitive efficiency over the standard 0--100 format, while boundary compression degrades performance and round-number preferences persist even under irregular ranges. These results demonstrate that confidence scale design directly affects the quality of 
    
[^250]: MOSAIC：用于跨范式智能体混合与人机协作的通用智能体级接口

    MOSAIC: A Universal Agent-Level Interface for Cross-Paradigm Agent Mixing and Human-AI Collaboration

    [https://arxiv.org/abs/2603.01260](https://arxiv.org/abs/2603.01260)

    MOSAIC是一个开源平台，通过基于IPC的工作器协议和统一的操作员抽象接口，使强化学习策略、大语言模型、视觉语言模型和人类操作员等异构智能体能够在同一强化学习环境中协作并实现公平的跨范式比较。

    

    现有基础设施无法在同一环境中部署来自不同决策范式的智能体，导致无法在相同条件下进行公平的跨范式比较。我们提出了MOSAIC，一个开源平台，使异构智能体（强化学习策略、大语言模型、视觉语言模型和人类操作员）能够在共享的强化学习环境中以临时团队设置进行协作行动，并保证结果可复现。MOSAIC引入了三项贡献：(i) 基于IPC的工作器协议，将原生和第三方框架封装为隔离的子进程工作器，每个工作器无需修改即可执行自身的训练和推理逻辑，并通过版本化的进程间协议进行通信；(ii) 操作员抽象，通过将工作器映射到智能体槽位来形成智能体级接口：每个操作员，无论其背后由强化学习策略、大语言模型还是人类支撑，都遵循一个最小的通用接口；(iii) 一个确定性……（摘要原文在此处截断）

    arXiv:2603.01260v3 Announce Type: replace-cross  Abstract: Existing infrastructure cannot deploy agents from different decision-making paradigms within the same environment, making fair cross-paradigm comparison under identical conditions impossible. We present MOSAIC, an open-source platform that enables heterogeneous agents (RL policies, LLMs, VLMs, and human operators) to act within shared reinforcement learning environments in ad-hoc team settings with reproducible results. MOSAIC introduces three contributions. (i) IPC-based worker protocol that wraps native and third-party frameworks as isolated subprocess workers, each executing its own training and inference logic unmodified and communicating through a versioned inter-process protocol. (ii) An operator abstraction that forms an agent-level interface by mapping workers to agent slots: each operator, regardless of whether it is backed by an RL policy, an LLM, or a human, conforms to a minimal universal interface. (iii) A determin
    
[^251]: 探测大语言模型中的知识归因

    Probing for Knowledge Attribution in Large Language Models

    [https://arxiv.org/abs/2602.22787](https://arxiv.org/abs/2602.22787)

    本文提出自监督数据生成流水线AttriWiki，证明仅用简单线性探针即可从大语言模型的隐藏表示中可靠地判断答案的知识来源是内部记忆还是外部上下文，从而区分忠实性幻觉与事实性幻觉以助力针对性缓解。

    

    大语言模型（LLM）的幻觉，即流畅但事实错误的生成内容，可分为两类：忠实性违规（模型误用了所提供的上下文）和事实性违规（答案反映了内部知识的错误）。要采取恰当的缓解措施，需要知道每个答案由哪种来源驱动。我们研究了贡献性归因，即对每个输出背后的主要知识来源进行分类，并证明在隐藏表示上训练的简单线性探针能够可靠地识别它。我们提出了AttriWiki，这是一个自监督流水线，通过提示模型从记忆中回忆被隐去的实体或从上下文中读取实体来自动生成带标签的训练数据，而无需依赖知识冲突。在AttriWiki上训练的探针在Llama-3.1-8B、Mistral-7B和Qwen-7B上取得了高达0.96的Macro-F1分数，迁移到SQuAD和WebQuestions数据集时达到0.94-0.99的Macro-F1，并展现出良好的泛化能力。

    arXiv:2602.22787v3 Announce Type: replace  Abstract: Large language model (LLM) hallucinations, meaning fluent but factually incorrect generations, fall into two types: faithfulness violations, where the model misuses provided context, and factuality violations, where answers reflect errors in internal knowledge. Proper mitigation depends on knowing which source drives each answer. We study contributive attribution, i.e. the classification of the dominant knowledge source behind each output, and show that a simple linear probe trained on hidden representations can reliably identify it. We introduce AttriWiki, a self-supervised pipeline that automatically generates labelled training data by prompting models to recall withheld entities from memory or read them from context without relying on knowledge conflicts. Probes trained on AttriWiki achieve up to 0.96 Macro-$F_1$ on Llama-3.1-8B, Mistral-7B, and Qwen-7B, transfer to SQuAD and WebQuestions with 0.94-0.99 Macro-$F_1$, and generalise
    
[^252]: 超越提示方法：通过Logit空间集成实现语音大语言模型的高效鲁棒上下文偏置（LOGIC）

    Beyond Prompting: Efficient and Robust Contextual Biasing for Speech LLMs via Logit-Space Integration (LOGIC)

    [https://arxiv.org/abs/2601.15397](https://arxiv.org/abs/2601.15397)

    本文提出LOGIC方法，通过在Logit空间层面直接集成上下文偏置，为语音大语言模型提供了一种高效且鲁棒的解决方案，克服了传统提示方法的可扩展性瓶颈和生成式错误纠正的幻觉问题。

    

    新实体的快速涌现——受文化变迁、流行趋势演变和个性化用户数据的驱动——对现有的语音大语言模型（Speech LLMs）构成了重大挑战。虽然这些模型在通用对话任务中表现出色，但其静态训练知识限制了它们识别特定领域术语（如联系人姓名、播放列表或技术行话）的能力。现有解决方案主要依赖提示方法，但其可扩展性较差：随着实体列表的增长，提示方法会遇到上下文窗口限制、推理延迟增加以及“迷失在中间”现象。另一种替代方法——生成式错误纠正（GEC）——试图通过后处理重写转录文本，但经常出现“过度纠正”问题，引入从未被说出的实体的幻觉。在这项工作中，我们介绍了LOGIC（用于上下文偏置的Logit空间集成），一种……

    arXiv:2601.15397v3 Announce Type: replace-cross  Abstract: The rapid emergence of new entities -- driven by cultural shifts, evolving trends, and personalized user data -- poses a significant challenge for existing Speech Large Language Models (Speech LLMs). While these models excel at general conversational tasks, their static training knowledge limits their ability to recognize domain-specific terms such as contact names, playlists, or technical jargon. Existing solutions primarily rely on prompting, which suffers from poor scalability: as the entity list grows, prompting encounters context window limitations, increased inference latency, and the "lost-in-the-middle" phenomenon. An alternative approach, Generative Error Correction (GEC), attempts to rewrite transcripts via post-processing but frequently suffers from "over-correction", introducing hallucinations of entities that were never spoken.   In this work, we introduce LOGIC (Logit-Space Integration for Contextual Biasing), an 
    
[^253]: 面向稳定大语言模型预训练的输出嵌入中心化方法

    Output Embedding Centering for Stable LLM Pretraining

    [https://arxiv.org/abs/2601.02031](https://arxiv.org/abs/2601.02031)

    该论文揭示了大语言模型预训练末期输出logit发散的根源在于输出嵌入的各向异性，并提出输出嵌入中心化（OEC）新方法，通过μ-centering或μ-loss两种实现方式有效抑制训练不稳定，其稳定性优于z-loss且与logit软上限方法相当。

    

    大语言模型的预训练不仅成本高昂，而且容易出现某些训练不稳定性问题。一种通常在训练结束时发生的特定不稳定性是输出logit发散。目前最广泛使用的缓解策略——z-loss和logit软上限——仅仅处理症状而非问题的根本原因。在本文中，我们从输出嵌入几何特性的角度分析了这种不稳定性，并确定各向异性的嵌入是其根源。基于此，我们提出输出嵌入中心化（OEC）作为一种新的缓解策略，并证明它能够抑制输出logit发散。OEC可以通过两种不同的方式实现：一种称为μ-centering的确定性操作，或一种称为μ-loss的正则化方法。我们的实验表明，这两种变体在训练稳定性方面都优于z-loss，同时与logit软上限方法相当。

    arXiv:2601.02031v3 Announce Type: replace-cross  Abstract: Pretraining of large language models is not only expensive but also prone to certain training instabilities. A specific instability that often occurs at the end of training is output logit divergence. The most widely used mitigation strategies, z-loss and logit soft-capping, merely address the symptoms rather than the underlying cause of the problem. In this paper, we analyze the instability from the perspective of the output embeddings' geometry and identify anisotropic embeddings as its source. Based on this, we propose output embedding centering (OEC) as a new mitigation strategy, and demonstrate that it suppresses output logit divergence. OEC can be implemented in two different ways: as a deterministic operation called $\mu$-centering, or a regularization method called $\mu$-loss. Our experiments show that both variants outperform z-loss in terms of training stability, while being on par with logit soft-capping. This holds 
    
[^254]: DCO：通过预测性管理实现LLM加速器的动态缓存编排

    DCO: Dynamic Cache Orchestration for LLM Accelerators through Predictive Management

    [https://arxiv.org/abs/2512.07312](https://arxiv.org/abs/2512.07312)

    提出了一种面向LLM加速器的动态缓存编排方案DCO，利用软件栈中的数据流信息进行预测性缓存管理（包括死块预测、旁路决策和缓存抖动缓解），在保持编程简洁性的同时，相比传统缓存架构实现了高达1.80倍的加速。

    

    大语言模型（LLM）的快速普及正推动AI加速器朝着日益强大且专业化的设计方向发展。我们没有选择通过深度层次化的便笺式存储器（SPM）及其异步管理来进一步增加软件开发的复杂性，而是研究了设计谱系的另一端：一种配备共享系统级缓存和应用感知管理策略的多核AI加速器，从而保持较低的编程负担。我们的方法利用软件栈中可获取的数据流信息来指导缓存替换（包括死块预测），并结合旁路决策以及缓解缓存抖动的机制。我们使用周期精确的模拟器对该方案进行评估，观察到与传统的缓存架构相比取得了显著的性能提升（高达1.80倍的加速）。此外，我们构建并验证了一个分析模型，该模型接收……（原文在此处截断）

    arXiv:2512.07312v2 Announce Type: replace-cross  Abstract: The rapid adoption of large language models (LLMs) is pushing AI accelerators toward increasingly powerful and specialized designs. Instead of further complicating software development with deeply hierarchical scratchpad memories (SPMs) and their asynchronous management, we investigate the opposite point of the design spectrum: a multi-core AI accelerator equipped with a shared system-level cache and application-aware management policies, which keeps the programming effort modest. Our approach exploits dataflow information available in the software stack to guide cache replacement (including dead-block prediction), in concert with bypass decisions and mechanisms that alleviate cache thrashing.   We assess the proposal using a cycle-accurate simulator and observe substantial performance gains (up to 1.80x speedup) compared with conventional cache architectures. In addition, we build and validate an analytical model that takes in
    
[^255]: 论机器学习的社会影响

    On the Societal Impact of Machine Learning

    [https://arxiv.org/abs/2510.23693](https://arxiv.org/abs/2510.23693)

    本博士论文提出了更恰当地测量机器学习系统公平性、系统性分解系统以预判偏见动态的方法，以及在保持系统效用的同时减少算法歧视的有效干预措施，为使机器学习的社会影响符合更广泛的社会价值奠定了基础。

    

    本博士论文研究了机器学习（ML）的社会影响。机器学习日益影响着重要的决策与推荐，深刻影响着我们生活的诸多方面。由于这些数据驱动的系统在开发时往往没有明确考虑公平性，它们存在产生歧视性影响的风险。本论文的贡献包括：实现对机器学习系统公平性的更恰当测量、对机器学习系统进行系统性分解以预判偏见动态，以及在保持系统效用的同时减少算法歧视的有效干预措施。最后，我讨论了随着包括生成式人工智能在内的机器学习系统日益融入社会所带来的持续挑战与未来研究方向。这项工作为确保机器学习的社会影响符合更广泛的社会价值奠定了基础。

    arXiv:2510.23693v2 Announce Type: replace  Abstract: This PhD thesis investigates the societal impact of machine learning (ML). ML increasingly informs consequential decisions and recommendations, significantly affecting many aspects of our lives. As these data-driven systems are often developed without explicit fairness considerations, they carry the risk of discriminatory effects. The contributions in this thesis enable more appropriate measurement of fairness in ML systems, systematic decomposition of ML systems to anticipate bias dynamics, and effective interventions that reduce algorithmic discrimination while maintaining system utility. I conclude by discussing ongoing challenges and future research directions as ML systems, including generative artificial intelligence, become increasingly integrated into society. This work offers a foundation for ensuring that ML's societal impact aligns with broader social values.
    
[^256]: 通过主动测试选择实现及时的临床诊断

    Timely Clinical Diagnosis through Active Test Selection

    [https://arxiv.org/abs/2510.18988](https://arxiv.org/abs/2510.18988)

    该论文提出ACTMED框架，将贝叶斯实验设计与大语言模型相结合，在诊断过程的每一步自适应地选择最能降低诊断不确定性的检查项目，以模拟临床医生在资源受限环境下的顺序式诊断推理。

    

    人们对使用机器学习（ML）支持临床诊断的兴趣日益增长，但大多数方法依赖于静态的、完整观测的数据集，未能反映临床医生在实践中所采用的顺序性、资源感知的推理方式。诊断过程仍然复杂且容易出错，尤其是在高压或资源有限的环境中，这凸显了开发能帮助临床医生做出及时且具成本效益决策的框架的必要性。我们提出了ACTMED（基于模型实验设计的自适应临床测试选择），这是一个将贝叶斯实验设计（BED）与大语言模型（LLM）相结合的诊断框架，旨在更好地模拟真实世界的诊断推理。在每一步，ACTMED都会选择预期能为给定患者带来最大诊断不确定性降低的测试。大语言模型充当灵活的模拟器，生成合理的患者状态分布，并在无需结构化数据的情况下支持信念更新。

    arXiv:2510.18988v5 Announce Type: replace  Abstract: There is growing interest in using machine learning (ML) to support clinical diagnosis, but most approaches rely on static, fully observed datasets and fail to reflect the sequential, resource-aware reasoning clinicians use in practice. Diagnosis remains complex and error prone, especially in high-pressure or resource-limited settings, underscoring the need for frameworks that help clinicians make timely and cost-effective decisions. We propose ACTMED (Adaptive Clinical Test selection via Model-based Experimental Design), a diagnostic framework that integrates Bayesian Experimental Design (BED) with large language models (LLMs) to better emulate real-world diagnostic reasoning. At each step, ACTMED selects the test expected to yield the greatest reduction in diagnostic uncertainty for a given patient. LLMs act as flexible simulators, generating plausible patient state distributions and supporting belief updates without requiring stru
    
[^257]: 用于阑尾炎分类的手术视觉联邦学习：FedSurg EndoVis 2024 挑战赛结果

    Federated Learning for Surgical Vision in Appendicitis Classification: Results of the FedSurg EndoVis 2024 Challenge

    [https://arxiv.org/abs/2510.04772](https://arxiv.org/abs/2510.04772)

    本研究发起了首个专注于手术视觉联邦学习的国际挑战赛 FedSurg，基于多中心腹腔镜阑尾切除术数据集进行了概念验证评估，并发现时序建模是提升对未见临床中心泛化能力最稳定的架构因素。

    

    开发具有泛化能力的手术人工智能需要多机构数据，但隐私限制使得直接共享数据不可行，这使得联邦学习（FL）成为一个天然的候选方案。然而，联邦学习在复杂的时空手术视频中的应用目前仍缺乏系统性基准测试。我们提出了 FedSurg 挑战赛，这是首个专注于手术视觉领域联邦学习的国际性活动，并利用腹腔镜阑尾切除术的多中心数据集（Appendix300 的子集）进行了概念验证评估。三个参赛提交方案在针对未见过的临床中心的泛化能力以及中心特定的本地适应能力方面进行了评估，并与集中式训练、群体学习、参数高效微调基线以及参考分类器进行了比较。我们的分析表明，时序建模是与对未见临床中心泛化能力关联最稳定的架构因素，尽管其效果在不同指标间存在差异。分类器崩溃源于……（摘要在此处截断）

    arXiv:2510.04772v3 Announce Type: replace-cross  Abstract: Developing generalizable surgical AI requires multi-institutional data, yet privacy constraints preclude direct data sharing, making Federated Learning (FL) a natural candidate. Its application to complex, spatiotemporal surgical video remains largely unbenchmarked. We present the FedSurg Challenge, the first international initiative dedicated to FL in surgical vision, as a proof-of-concept evaluation using a multi-center dataset of laparoscopic appendectomies (subset of Appendix300). Three participant submissions were evaluated on generalization to an unseen clinical center and center-specific local adaptation, alongside centralized, Swarm Learning, parameter-efficient fine-tuning baselines, and reference classifiers. Our analysis identifies temporal modeling as the architectural factor most consistently associated with generalization to the unseen center, although effects vary across metrics. Classifier collapse arises from b
    
[^258]: CHRONOBERG：捕捉基础模型中的语言演化与时间感知

    CHRONOBERG: Capturing Language Evolution and Temporal Awareness in Foundation Models

    [https://arxiv.org/abs/2509.22360](https://arxiv.org/abs/2509.22360)

    该论文提出了CHRONOBERG——一个跨越250年、带有丰富时间标注的英语书籍文本语料库，通过历史校准的情感词典量化语言演变，从而提升基础模型对语言历时变化的时间感知能力。

    

    大型语言模型（LLM）通过利用社交媒体和从网络上爬取的各类数据，实现了大规模的高效运作。然而，尽管现有语料库具有多样性，但它们普遍缺乏长期的时间结构，这可能限制了LLM对语言语义和规范演变的语境化理解能力，以及捕捉历时性变化的能力。为支持针对后者的分析与训练，我们推出了CHRONOBERG——一个跨越250年、具有时间结构化特征的英语书籍文本语料库，精选自古腾堡计划（Project Gutenberg），并辅以多种时间标注进行丰富。首先，书籍经过编辑的特性使我们能够通过时间敏感的效价-唤醒-支配度（Valence-Arousal-Dominance，VAD）分析来量化词汇语义随时间的变化，并构建经过历史校准的情感词典，以支持基于时间的解释。借助这些词典，我们论证了现代基于LLM的工具需要更好地定位其对话语的检测……

    arXiv:2509.22360v2 Announce Type: replace  Abstract: Large language models (LLMs) excel at operating at scale by leveraging social media and various data crawled from the web. Whereas existing corpora are diverse, their frequent lack of long-term temporal structure may however limit an LLM's ability to contextualize semantic and normative evolution of language and to capture diachronic variation. To support analysis and training for the latter, we introduce CHRONOBERG, a temporally structured corpus of English book texts spanning 250 years, curated from Project Gutenberg and enriched with a variety of temporal annotations. First, the edited nature of books enables us to quantify lexical semantic change through time-sensitive Valence-Arousal-Dominance (VAD) analysis and to construct historically calibrated affective lexicons to support temporally grounded interpretation. With the lexicons at hand, we demonstrate a need for modern LLM-based tools to better situate their detection of disc
    
[^259]: 大语言模型有限元认知能力的证据

    Evidence for Limited Metacognition in LLMs

    [https://arxiv.org/abs/2509.21545](https://arxiv.org/abs/2509.21545)

    该研究借鉴非人类动物元认知研究方法，提出了一种不依赖模型自我报告的定量评估框架，发现2024年初以来的前沿大语言模型展现出有限的元认知能力，能够评估自身答题信心并预测自己将给出的答案。

    

    arXiv:2509.21545v3 公告类型：替换 摘要：大语言模型（LLM）可能具有自我意识甚至感知能力的观点正日益受到公众关注，并具有重大的安全和政策影响，但测量这些能力的科学仍处于起步阶段。本文引入了一种新颖的方法来定量评估大语言模型的元认知能力。受非人类动物元认知研究的启发，我们的方法避开了模型自我报告的方式，而是测试模型能够在多大程度上策略性地运用对自身内部状态的知识。通过两个实验范式，我们证明了2024年初以来推出的前沿大语言模型展现出越来越强的某些元认知能力的证据，具体包括：评估并利用自己对正确回答事实性和推理问题能力的信心，以及预测自己会给出什么答案并恰当地利用该信息的能力。我们进一步通过行为证据加以支持……

    arXiv:2509.21545v3 Announce Type: replace  Abstract: The possibility of LLM self-awareness and even sentience is gaining increasing public attention and has major safety and policy implications, but the science of measuring them is still in a nascent state. Here we introduce a novel methodology for quantitatively evaluating metacognitive abilities in LLMs. Taking inspiration from research on metacognition in nonhuman animals, our approach eschews model self-reports and instead tests to what degree models can strategically deploy knowledge of internal states. Using two experimental paradigms, we demonstrate that frontier LLMs introduced since early 2024 show increasingly strong evidence of certain metacognitive abilities, specifically the ability to assess and utilize their own confidence in their ability to answer factual and reasoning questions correctly and the ability to anticipate what answers they would give and utilize that information appropriately. We buttress these behavioral 
    
[^260]: 生成式AI在本科核心数学中的表现：一项课程层面的案例研究

    Generative AI performance in core undergraduate mathematics: a curriculum-level case study

    [https://arxiv.org/abs/2509.13359](https://arxiv.org/abs/2509.13359)

    该研究以一年级数学课程的八份真实考试试卷为基准，通过让生成式AI作答并进行盲评，系统性地评估了GenAI在本科核心数学课程各模块及整体层面的表现。

    

    生成式人工智能（GenAI）工具，如OpenAI的ChatGPT，正在改变教育格局，促使人们重新思考传统的评估实践。与此同时，大学正在探索替代面对面闭卷考试的方式，这引发了人们对学术诚信以及非监考环境下教学一致性的担忧。本研究系统地调查了GenAI在一年级数学课程中典型数学问题上的表现。采用实证方法并利用当前的考试题目作为课程内容的代理，我们对GenAI提交的八项本科数学评估作业进行了生成、转录和盲评，这些评估涵盖了整个一年级课程。通过将GenAI对个别问题的独立回答相结合，我们能够在模块层面和整体课程层面对GenAI的表现进行有意义的评估。

    arXiv:2509.13359v4 Announce Type: replace-cross  Abstract: Generative artificial intelligence (GenAI) tools such as OpenAI's ChatGPT are transforming the educational landscape, prompting reconsideration of traditional assessment practices. In parallel, universities are exploring alternatives to in-person, closed-book examinations, raising concerns about academic integrity and pedagogical alignment in uninvigilated settings. This study systematically investigates the performance of GenAI on typical mathematics questions from across a first-year mathematics curriculum. Adopting an empirical approach and utilising current examination questions as a proxy for course content, we generate, transcribe, and blind-mark GenAI submissions to eight undergraduate mathematics assessments, spanning the entirety of the first-year curriculum. By combining independent GenAI responses to individual questions, we enable a meaningful evaluation of GenAI performance, both at the level of modules and across 
    
[^261]: Amulet：一个用于评估机器学习防御与风险之间相互作用的Python库

    Amulet: a Python Library for Assessing Interactions Among ML Defenses and Risks

    [https://arxiv.org/abs/2509.12386](https://arxiv.org/abs/2509.12386)

    本文推出了首个Python库Amulet，用于评估机器学习防御与风险之间的预期及非预期交互作用，其全面性、可扩展性、一致性和适用性为系统性研究跨多种风险的非预期交互提供了统一基础。

    

    机器学习（ML）模型容易受到安全、隐私和公平性等多方面风险的威胁。大多数防御机制都是针对每种风险单独设计的（预期交互作用），但可能会无意中影响模型对其他无关风险的易感性（非预期交互作用）。我们推出了Amulet，这是首个用于评估机器学习防御与风险之间预期和非预期交互作用的Python库。Amulet具有全面性，包含代表性的攻击、防御和评估指标；由于其模块化设计，可扩展至新的模块；通过用户友好的输入输出API模板保持一致性；并且适用于评估新型交互作用。通过满足这四个特性，Amulet为研究防御机制之间的交互作用提供了统一的基础，实现了对跨多种风险的非预期交互作用的首次系统性评估。

    arXiv:2509.12386v3 Announce Type: replace-cross  Abstract: Machine learning (ML) models are susceptible to various risks to security, privacy, and fairness. Most defenses are designed to protect against each risk individually (intended interactions) but can inadvertently affect susceptibility to other unrelated risks (unintended interactions). We introduce Amulet, the first Python library for evaluating both intended and unintended interactions among ML defenses and risks. Amulet is comprehensive by including representative attacks, defenses, and metrics; extensible to new modules due to its modular design; consistent with a user-friendly API template for inputs and outputs; and applicable for evaluating novel interactions. By satisfying all four properties, Amulet offers a unified foundation for studying how defenses interact, enabling the first systematic evaluation of unintended interactions across multiple risks.
    
[^262]: 频谱掩蔽与插值攻击（SMIA）：一种针对语音认证与反欺骗系统的黑盒对抗攻击

    Spectral Masking and Interpolation Attack (SMIA): A Black-box Adversarial Attack against Voice Authentication and Anti-Spoofing Systems

    [https://arxiv.org/abs/2509.07677](https://arxiv.org/abs/2509.07677)

    本文提出了SMIA黑盒对抗攻击方法，通过操纵AI生成音频中人耳不可听见的频谱区域来绕过语音认证和反欺骗系统的检测。

    

    语音认证系统（VAS）利用独特的声音特征进行身份验证，并被越来越多地应用于银行和医疗保健等高安全性行业。尽管采用深度学习技术不断改进，但语音认证系统仍然面临深度伪造和对抗攻击等复杂威胁带来的严重漏洞。逼真语音克隆技术的出现使检测变得更加困难，因为系统难以区分真实音频与合成音频。虽然目前存在反欺骗对策（CM）来缓解这些风险，但许多对策依赖于静态检测模型，这些模型可能被新型对抗方法绕过，从而留下关键的安全缺口。为了证明这一漏洞，我们提出了频谱掩蔽与插值攻击（SMIA），这是一种新颖的方法，通过策略性地操纵AI生成音频中人耳不可听见的频率区域，在人耳无法察觉的区域改变语音，SMIA创建出能够绕过检测的对抗样本。

    arXiv:2509.07677v5 Announce Type: replace-cross  Abstract: Voice Authentication Systems (VAS) use unique vocal characteristics for verification. They are increasingly integrated into high-security sectors such as banking and healthcare. Despite their improvements using deep learning, they face severe vulnerabilities from sophisticated threats like deepfakes and adversarial attacks. The emergence of realistic voice cloning complicates detection, as systems struggle to distinguish authentic from synthetic audio. While anti-spoofing countermeasures (CMs) exist to mitigate these risks, many rely on static detection models that can be bypassed by novel adversarial methods, leaving a critical security gap. To demonstrate this vulnerability, we propose the Spectral Masking and Interpolation Attack (SMIA), a novel method that strategically manipulates inaudible frequency regions of AI-generated audio. By altering the voice in imperceptible zones to the human ear, SMIA creates adversarial sampl
    
[^263]: 针对语音认证与反欺骗系统威胁的综述

    A Survey of Threats Against Voice Authentication and Anti-Spoofing Systems

    [https://arxiv.org/abs/2508.16843](https://arxiv.org/abs/2508.16843)

    本综述系统梳理了针对语音认证系统和反欺骗对策的四类主要威胁（数据投毒、对抗攻击、深度伪造和对抗性欺骗），追溯了语音认证技术演进过程中漏洞的同步发展，并总结了各类攻击的方法、常用数据集及性能局限。

    

    语音认证技术经历了重大变革，从依赖手工设计声学特征的传统系统发展到能够提取鲁棒说话人嵌入的深度学习模型。这一进步使其应用范围扩展到金融、智能设备、执法等众多领域。然而，随着应用的普及，威胁也随之增长。本综述全面回顾了针对语音认证系统（VAS）和反欺骗对策（CMs）的现代威胁图景，包括数据投毒攻击、对抗攻击、深度伪造攻击和对抗性欺骗攻击。我们按时间顺序追溯了语音认证的发展历程，并考察了系统漏洞如何随技术进步而演变。针对每一类攻击，我们总结了其方法论，重点介绍了常用数据集，比较了性能与局限性，并使用广泛认可的分类体系对现有文献进行了整理。

    arXiv:2508.16843v5 Announce Type: replace-cross  Abstract: Voice authentication has undergone significant changes from traditional systems that relied on handcrafted acoustic features to deep learning models that can extract robust speaker embeddings. This advancement has expanded its applications across finance, smart devices, law enforcement, and beyond. However, as adoption has grown, so have the threats. This survey presents a comprehensive review of the modern threat landscape targeting Voice Authentication Systems (VAS) and Anti-Spoofing Countermeasures (CMs), including data poisoning, adversarial, deepfake, and adversarial spoofing attacks. We chronologically trace the development of voice authentication and examine how vulnerabilities have evolved in tandem with technological advancements. For each category of attack, we summarize methodologies, highlight commonly used datasets, compare performance and limitations, and organize existing literature using widely accepted taxonomi
    
[^264]: 结合降雨学习内在水质动态以实现数据驱动预测

    Learning Intrinsic Water-Quality Dynamics with Rainfall for Data-Driven Forecasting

    [https://arxiv.org/abs/2508.08279](https://arxiv.org/abs/2508.08279)

    本文提出 RaiNet 模型，联合建模多尺度水质动态与站点特定降雨效应以实现数据驱动的水质预测，并发布了包含超过15万条水质观测数据的三个多模态真实数据集。

    

    降雨是水质变化的重要环境驱动因素，通过径流、污染物输运、稀释和再悬浮等过程发挥作用。传统机理模型能够显式描述这些过程，但通常需要大量的过程设定和针对特定站点的校准，限制了其在变化水文条件下的灵活性。在这项工作中，我们探索了一种数据驱动的替代方案，提出 RaiNet 来联合建模多尺度水质动态与站点特定的降雨效应，涵盖相对滞后和时间尺度。RaiNet 采用 LocTrend 捕捉不规则的水质动态，从网格化降水数据构建面向站点的降雨事件，并引入 XGateFusion 实现跨尺度的条件性滞后感知融合。我们还发布了三个真实世界的多模态数据集，包含超过 15 万条时间对齐的水质观测数据和网格化降水数据。

    arXiv:2508.08279v2 Announce Type: replace  Abstract: Rainfall is an important environmental driver of water-quality variations through processes such as runoff, pollutant transport, dilution, and resuspension. Traditional mechanistic models can explicitly describe these processes but often require substantial process specification and site-specific calibration, limiting their flexibility under changing hydrological conditions. In this work, we explore a data-driven alternative by proposing RaiNet to jointly model multiscale water-quality dynamics and station-specific rainfall effects across relative lags and temporal scales. RaiNet employs LocTrend to capture irregular water-quality dynamics, constructs station-oriented rainfall events from gridded precipitation, and introduces XGateFusion for conditional lag-aware fusion across scales. We further release three real-world multimodal datasets comprising over 150,000 temporally aligned water quality observations and gridded precipitation
    
[^265]: 发现时间结构：分层强化学习综述

    Discovering Temporal Structure: An Overview of Hierarchical Reinforcement Learning

    [https://arxiv.org/abs/2506.14045](https://arxiv.org/abs/2506.14045)

    本文是一篇分层强化学习（HRL）综述，从决策基本挑战的视角阐明了HRL的优势，并系统梳理了发现和利用经验流中时间结构的各类方法。

    

    开发能够在复杂开放环境中进行探索、规划和学习的智能体是人工智能（AI）领域的重大挑战。分层强化学习（HRL）通过发现和利用经验流中的时间结构，为应对这一挑战提供了一种有前景的解决方案。HRL框架的强大吸引力催生了大量丰富多样的文献，这些文献试图发现有用的结构。然而，如何定义什么才是好的结构，以及在哪些类型的问题中识别这种结构可能有所帮助，目前仍不明确。本工作旨在从决策基本挑战的视角出发，阐明HRL的优势，并强调其对AI智能体性能权衡的影响。基于这些优势，我们进而综述了在HRL中发现时间结构的各类方法族，涵盖从学习直接……

    arXiv:2506.14045v2 Announce Type: replace  Abstract: Developing agents capable of exploring, planning and learning in complex open-ended environments is a grand challenge in artificial intelligence (AI). Hierarchical reinforcement learning (HRL) offers a promising solution to this challenge by discovering and exploiting the temporal structure within a stream of experience. The strong appeal of the HRL framework has led to a rich and diverse body of literature attempting to discover a useful structure. However, it is still not clear how one might define what constitutes good structure in the first place, or the kind of problems in which identifying it may be helpful. This work aims to identify the benefits of HRL from the perspective of the fundamental challenges in decision-making, as well as highlight its impact on the performance trade-offs of AI agents. Through these benefits, we then cover the families of methods that discover temporal structure in HRL, ranging from learning direct
    
[^266]: CertDW：基于保形校准的认证数据集所有权验证

    CertDW: Towards Certified Dataset Ownership Verification via Conformal Calibration

    [https://arxiv.org/abs/2506.13160](https://arxiv.org/abs/2506.13160)

    本文提出了首个认证数据集水印CertDW及基于保形校准的认证数据集所有权验证方法，能够在受约束的恶意扰动下依然确保可靠的数据集所有权验证。

    

    深度神经网络（DNNs）的成功严重依赖于高质量的开源数据集（如ImageNet），这使得数据集所有权验证（DOV）对于保护公共数据集版权至关重要。在本文中，我们发现现有的DOV方法（隐式地）假设验证过程是可信的，即可疑模型会直接使用验证样本作为输入并返回其结果来验证所有权。然而，这一假设在实践中未必成立，且当受到有意或无意的扰动时，其性能可能会急剧下降。为了解决这一局限性，我们提出了首个认证数据集水印（即CertDW）以及基于CertDW的认证数据集所有权验证方法，该方法在某些条件下（例如受约束的像素级扰动下）即使在恶意攻击下也能确保可靠的验证。具体而言，受保形预测启发……

    arXiv:2506.13160v2 Announce Type: replace  Abstract: Deep neural networks (DNNs) rely heavily on high-quality open-source datasets (e.g., ImageNet) for their success, making dataset ownership verification (DOV) crucial for protecting public dataset copyrights. In this paper, we find existing DOV methods (implicitly) assume that the verification process is faithful, where the suspicious model will directly verify ownership by using the verification samples as input and returning their results. However, this assumption may not necessarily hold in practice and their performance may degrade sharply when subjected to intentional or unintentional perturbations. To address this limitation, we propose the first certified dataset watermark (i.e., CertDW) and CertDW-based certified dataset ownership verification method that ensures reliable verification even under malicious attacks, under certain conditions (e.g., constrained pixel-level perturbation). Specifically, inspired by conformal predict
    
[^267]: VAE与扩散模型中的泛化：一种统一的信息论分析

    Generalization in VAE and Diffusion Models: A Unified Information-Theoretic Analysis

    [https://arxiv.org/abs/2506.00849](https://arxiv.org/abs/2506.00849)

    本文提出了一个统一的信息论框架，通过将编码器和生成器视为随机映射，同时为VAE和扩散模型的泛化提供理论保证，并给出仅基于训练数据的可计算边界，以选择最优扩散时间并改进模型性能。

    

    尽管扩散模型（DM）和变分自编码器（VAE）在实证上取得了成功，但它们的泛化性能在理论上仍未得到充分探索，尤其缺乏对共享的编码器-生成器结构的全面考虑。利用最新的信息论工具，我们提出了一个统一的理论框架，通过将编码器和生成器视为随机映射，为两者的泛化同时提供保证。该框架进一步实现了：（1）对VAE进行更精细的分析，考虑了此前被忽视的生成器泛化问题；（2）阐明了DM中依赖于扩散时间 $T$ 的泛化项之间的明确权衡；（3）为DM提供了仅基于训练数据的可计算边界，从而能够选择最优的扩散时间 $T$，并将此类边界整合到优化过程中以提升模型性能。实证结果表明……

    arXiv:2506.00849v2 Announce Type: replace  Abstract: Despite the empirical success of Diffusion Models (DMs) and Variational Autoencoders (VAEs), their generalization performance remains theoretically underexplored, especially lacking a full consideration of the shared encoder-generator structure. Leveraging recent information-theoretic tools, we propose a unified theoretical framework that provides guarantees for the generalization of both the encoder and generator by treating them as randomized mappings. This framework further enables (1) a refined analysis for VAEs, accounting for the generator's generalization, which was previously overlooked; (2) illustrating an explicit trade-off in generalization terms for DMs that depends on the diffusion time $T$; and (3) providing computable bounds for DMs based solely on the training data, allowing the selection of the optimal $T$ and the integration of such bounds into the optimization process to improve model performance. Empirical results
    
[^268]: SG-Blend：在改进的Swish与GELU之间学习插值以获得鲁棒的神经表示

    SG-Blend: Learning an Interpolation Between Improved Swish and GELU for Robust Neural Representations

    [https://arxiv.org/abs/2505.23942](https://arxiv.org/abs/2505.23942)

    SG-Blend提出了一种逐层自适应激活函数，通过可学习的混合系数在改进的参数化Swish（SSwish）与GELU之间进行插值，仅用三个额外标量就让Transformer每一层自动找到最适合自身的激活形状，从而缓解深层网络中的梯度病理问题。

    

    当前主流的激活函数如Swish和GELU往往趋向于特定领域的最优选择——Swish是通过在视觉基准上进行神经架构搜索发现的，而GELU则主导着基于Transformer的语言模型，且两者都不提供任何将其门控形状自适应调整到各个单独层的机制。这种刚性在Transformer的FFN块中影响尤为显著，因为与BatchNorm不同，LayerNorm无法抑制激活函数选择在深层网络中引发的梯度病理问题。我们提出了SG-Blend，这是一种逐层自适应激活函数，它将SSwish（我们也引入的一种偏差校正的参数化Swish变体，具有可学习的锐度η和零中心化偏置γ）与GELU通过逐层混合系数α相结合，使每一层都能在SSwish-GELU连续谱上定位自身的最优点，每个FFN块仅需额外三个标量的开销，其中η初始化为1.0并通过反向传播自由学习。

    arXiv:2505.23942v2 Announce Type: replace  Abstract: Prevailing activation functions such as Swish and GELU tend toward domain-specific optima, Swish was discovered via neural architecture search on vision benchmarks, while GELU dominates transformer-based language models, and neither offers any mechanism to adapt its gating shape to individual layers. This rigidity is especially consequential in transformer FFN blocks, where LayerNorm, unlike BatchNorm, does not suppress the gradient pathologies that activation choice induces across depth. We propose SG-Blend, a per layer adaptive activation that combines SSwish, a bias-corrected, parametric Swish variant we also introduce, with learnable sharpness \b{eta} and zero-centering bias {\gamma}, with GELU through a per-layer blend coefficient {\alpha}, letting each layer locate its own optimum along the SSwishGELU continuum at a cost of only three additional scalars per FFN block, with \b{eta} initialized to 1.0 and learned freely via backp
    
[^269]: 迈向AI驱动的警务：从警察随身摄像头录像中进行跨学科知识发现

    Towards AI-Driven Policing: Interdisciplinary Knowledge Discovery from Police Body-Worn Camera Footage

    [https://arxiv.org/abs/2504.20007](https://arxiv.org/abs/2504.20007)

    该论文提出了一个结合图像、音频、自然语言处理和大语言模型的多模态跨学科AI框架，用于从警察随身摄像头录像中检测和分析警察与平民互动中的关键行为动态（如尊重、不尊重、事态升级与缓和），并建立了定制评估流程以验证转录质量与行为检测准确性。

    

    本文提出了一个新颖的跨学科框架，利用先进的人工智能（AI）和统计机器学习（ML）技术分析罗切斯特警察局（RPD）的警察随身摄像头（BWC）录像。我们的目标是检测、分类和分析警察与平民之间的互动模式，以识别关键的行为动态，如尊重、不尊重、事态升级和事态缓和。我们通过整合图像、音频和自然语言处理（NLP）技术来应用多模态数据分析，从BWC录像中提取有意义的见解。该框架结合了说话人分离、转录和大语言模型（LLM），以生成结构化、可解释的警察-平民接触事件摘要。我们还采用了定制的评估流程，在高风险的现实警务场景中评估转录质量和行为检测的准确性。

    arXiv:2504.20007v4 Announce Type: replace  Abstract: This paper proposes a novel interdisciplinary framework for analyzing police body-worn camera (BWC) footage from the Rochester Police Department (RPD) using advanced artificial intelligence (AI) and statistical machine learning (ML) techniques. Our goal is to detect, classify, and analyze patterns of interaction between police officers and civilians to identify key behavioral dynamics, such as respect, disrespect, escalation, and de-escalation. We apply multimodal data analysis by integrating image, audio, and natural language processing (NLP) techniques to extract meaningful insights from BWC footage. The framework incorporates speaker separation, transcription, and large language models (LLMs) to produce structured, interpretable summaries of police-civilian encounters. We also employ a custom evaluation pipeline to assess transcription quality and behavior detection accuracy in high-stakes, real-world policing scenarios. Our metho
    
[^270]: 探索基于大语言模型的可视化创作多模态提示方法

    Exploring Multimodal Prompt for Visualization Authoring with Large Language Models

    [https://arxiv.org/abs/2504.13700](https://arxiv.org/abs/2504.13700)

    针对自然语言提示在指导大语言模型进行可视化创作时精度与表达力不足的问题，本文提出以视觉提示作为补充输入模态，并据此设计了多模态提示可视化创作系统VisPilot，有效澄清用户意图并提升模型的解释能力。

    

    大语言模型（LLMs）的最新进展展示了通过简单的自然语言表达来自动化可视化创作过程的巨大潜力。然而，使用自然语言指导大语言模型在传达可视化意图方面的精度和表达能力有限，导致误解和耗时的反复迭代。为了解决这些局限性，我们开展了一项实证研究，以了解大语言模型在可视化创作背景下如何解释模糊或不完整的文本提示，以及导致大语言模型误解用户意图的条件。基于研究发现，我们引入视觉提示作为文本提示的补充输入模态，帮助澄清用户意图并提升大语言模型的解释能力。为了探索多模态提示在可视化创作中的潜力，我们设计了VisPilot，使用户能够通过多模态提示轻松创建可视化。

    arXiv:2504.13700v2 Announce Type: replace-cross  Abstract: Recent advances in large language models (LLMs) have shown great potential in automating the process of visualization authoring through simple natural language utterances. However, instructing LLMs using natural language is limited in precision and expressiveness for conveying visualization intent, leading to misinterpretation and time-consuming iterations. To address these limitations, we conduct an empirical study to understand how LLMs interpret ambiguous or incomplete text prompts in the context of visualization authoring, and the conditions making LLMs misinterpret user intent. Informed by the findings, we introduce visual prompts as a complementary input modality to text prompts, which help clarify user intent and improve LLMs' interpretation abilities. To explore the potential of multimodal prompting in visualization authoring, we design VisPilot, which enables users to easily create visualizations using multimodal promp
    
[^271]: ExpTest：基于损失曲线假设检验的深度神经网络自主学习率选择方法

    ExpTest: Loss-Curve Hypothesis Testing for Autonomous Learning-Rate Selection in Deep Neural Networks

    [https://arxiv.org/abs/2411.16975](https://arxiv.org/abs/2411.16975)

    ExpTest将训练损失曲线作为在线信号，通过序列统计检验检测收敛行为并自主降低学习率，从而实现深度神经网络初始学习率的自动选择，摆脱了繁琐的手动调参。

    

    超参数调整仍然是深度神经网络（DNN）训练中的一项重大挑战，需要手动搜索或耗时的网格搜索，这不仅增加了资源成本，还限制了机器学习的可及性。全局初始学习率是这些超参数中最关键的一个。自适应方法和基于调度（scheduling）的方法可以在训练过程中管理学习率，但仍需要手动选择初始全局值；而无需学习率的替代方法虽然省去了这一选择步骤，却在非凸问题上以牺牲性能或稳定性为代价。我们提出了ExpTest，一种自主的学习率控制器，它将训练损失曲线视为在线信号，并在具有理论依据的窗口上执行序列统计检验，以检测收敛行为并触发学习率降低。该框架结合了基于协方差的初始学习率估计、曲率驱动的窗口……（摘要原文在此处被截断）

    arXiv:2411.16975v2 Announce Type: replace  Abstract: Hyperparameter tuning remains a significant challenge in the training of deep neural networks (DNNs), requiring manual search or time-intensive grid searches that increase resource costs and limit the accessibility of machine learning. The global initial learning rate is among the most consequential of these hyperparameters. Adaptive and scheduling-based methods manage the learning rate during training but still require manual selection of an initial global value; learning-rate-free alternatives remove this selection at the cost of performance or stability on non-convex problems. We present ExpTest, an autonomous learning-rate controller that treats the training loss curve as an online signal and performs sequential statistical tests on theoretically motivated windows to detect convergent behavior and trigger learning-rate reductions. The framework combines a covariance-based initial learning-rate estimate, curvature-motivated window
    
[^272]: 多种物品下无筛选机制更为高效

    No Screening is More Efficient with Multiple Objects

    [https://arxiv.org/abs/2408.10077](https://arxiv.org/abs/2408.10077)

    该论文证明随着物品种类的增加，无筛选机制在福利最大化分配中持续保持最优，因为更多选择使低最优选项价值变得罕见，从而削弱了成本高昂的筛选的必要性，并将其应用于疫苗预约系统的设计。

    

    我们研究了当筛选依赖成本高昂的努力而非货币转移支付时，异质物品的福利最大化分配问题。随着物品种类的增加，无筛选机制表现良好。在一个价值独立同分布、且累积分布函数为对数凹的对称连续市场中，该多维问题可以精确地简化为关于参与者最优选项价值的一维问题。更多的选择使得低的最优选项价值变得更加罕见，从而削弱了进行筛选的理由。我们刻画了无筛选机制何时是最优的，并证明随着种类范围的扩大它依然保持最优。大种类极限情形以及对有限且存在相关性的市场的数值结果都支持了这一规律。我们应用这些结果，提出了一种基于邀请制的疫苗预约系统。

    arXiv:2408.10077v4 Announce Type: replace-cross  Abstract: We study the welfare-maximizing allocation of heterogeneous objects when screening uses costly effort rather than monetary transfers. No-screening mechanisms perform well as object variety increases. In a symmetric continuous market with i.i.d. values whose CDF is log-concave, the multidimensional problem reduces exactly to a single-dimensional problem in agents' best-option values. More options make low best-option values rarer, weakening the case for screening. We characterize when no screening is optimal and show it remains optimal as variety expands. Large-variety limits and numerical results for finite, correlated markets support this pattern. We apply these results to propose an invitation-based vaccine appointment system.
    
[^273]: 提高离群检测的三个因素

    Three Factors to Improve Out-of-Distribution Detection. (arXiv:2308.01030v1 [cs.LG])

    [http://arxiv.org/abs/2308.01030](http://arxiv.org/abs/2308.01030)

    本论文提出了三个因素来改善离群检测问题。首先，引入自我知识蒸馏损失以提高网络的准确性；其次，在训练过程中采样半困难离群数据以改善离群检测性能；最后，引入新型监督对比学习以同时提高离群检测性能和网络的准确性。通过结合这三个因素，我们的方法在分类和离群检测之间取得了良好的平衡，提高了准确性和离群检测性能。

    

    在离群检测问题中，利用辅助数据作为异常数据进行微调已经显示出令人鼓舞的性能。然而，先前的方法在分类准确性（ACC）和离群检测性能（AUROC、FPR、AUPR）之间存在权衡。为了改善这种权衡，我们做出了三个贡献：（i）引入自我知识蒸馏损失可以增强网络的准确性；（ii）采样半困难离群数据进行训练可以在对准确性影响最小的情况下改善离群检测性能；（iii）引入我们的新型监督对比学习可以同时改善离群检测性能和网络的准确性。通过结合这三个因素，我们的方法通过解决分类和离群检测之间的权衡，提高了准确性和离群检测性能。我们的方法在性能指标上都取得了比以前的方法更好的成绩。

    In the problem of out-of-distribution (OOD) detection, the usage of auxiliary data as outlier data for fine-tuning has demonstrated encouraging performance. However, previous methods have suffered from a trade-off between classification accuracy (ACC) and OOD detection performance (AUROC, FPR, AUPR). To improve this trade-off, we make three contributions: (i) Incorporating a self-knowledge distillation loss can enhance the accuracy of the network; (ii) Sampling semi-hard outlier data for training can improve OOD detection performance with minimal impact on accuracy; (iii) The introduction of our novel supervised contrastive learning can simultaneously improve OOD detection performance and the accuracy of the network. By incorporating all three factors, our approach enhances both accuracy and OOD detection performance by addressing the trade-off between classification and OOD detection. Our method achieves improvements over previous approaches in both performance metrics.
    

